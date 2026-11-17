"""The serving side reads safetensors headers and nothing else.

A checkpoint header carries dtype / shape / nbytes; it does not carry a
quant method. Which scheme classifies its tensors used to be decided a
second time on the serving path by downloading `config.json` and falling
back silently when that failed — a second identity for a fact the spec's
`quant_context` already declares (CLAUDE.md §2.4). Now the header
fetchers take the scheme as an argument, and only spec generation — the
snapshot writer, custom-model registration — resolves one from a config.
"""
from __future__ import annotations

import inspect
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.quant_schemes import SCHEME_NONE, resolve_scheme
from backend.hf.metadata import _fetch_metadata_hf, scan_local_hf_repo


def _write_safetensors(path: Path, tensors: dict) -> None:
    """A minimal safetensors file: u64 header length, JSON header, zeros."""
    header = {}
    offset = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * offset)


def _scan(tensors: dict, config: dict) -> dict:
    """The repo's own config selects the scheme, as registration does."""
    scheme, num_bits, unknown = resolve_scheme(config)
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        _write_safetensors(repo / "model.safetensors", tensors)
        return scan_local_hf_repo(repo, scheme=scheme, num_bits=num_bits,
                                  unknown_method=unknown)


class TestHeadersAreEnough(unittest.TestCase):
    def test_without_a_scheme_no_tensor_is_a_sidecar(self):
        """No quant config selects no scheme, and a tensor named like some
        scheme's sidecar is an architectural parameter: a norm's `.scale`
        gain stays in `total_params`."""
        meta = _scan({
            "model.layers.0.mlp.down_proj.weight": ("BF16", [64, 128], 64 * 128 * 2),
            "model.layers.0.norm.scale": ("BF16", [64], 64 * 2),
            "model.layers.0.mlp.down_proj.weight_scale_inv": ("F32", [1, 1], 4),
        }, {})
        self.assertEqual(meta["quant_scheme"], SCHEME_NONE.name)
        self.assertEqual(meta["total_params"], 64 * 128 + 64 + 1)
        self.assertEqual(meta["quantization_metadata_params"], 0)

    def test_an_unknown_method_is_reported_and_classifies_nothing(self):
        meta = _scan({
            "model.layers.0.mlp.down_proj.weight": ("I8", [64, 128], 64 * 128),
            "model.layers.0.mlp.down_proj.weight_scale": ("F32", [64, 1], 64 * 4),
        }, {"quantization_config": {"quant_method": "nonesuch"}})
        self.assertEqual(meta["quant_method_unknown"], "nonesuch")
        self.assertEqual(meta["quantization_metadata_params"], 0)
        self.assertEqual(meta["total_params"], 64 * 128 + 64)

    def test_a_packed_name_with_no_declared_width_is_not_unpacked(self):
        """`.weight_packed` under no scheme is not packed and has no
        semantic width: its elements count as stored, one each."""
        meta = _scan({
            "model.layers.0.mlp.down_proj.weight_packed": ("I32", [64, 128], 64 * 128 * 4),
            "model.layers.0.norm.weight": ("BF16", [64], 64 * 2),
        }, {})
        self.assertEqual(meta["total_params"], 64 * 128 + 64)

    def test_a_config_declared_layout_classifies_its_sidecars(self):
        """DeepSeek-V4's FP8 config says `scale_fmt: ue8m0`, which stores
        the block scales at `.scale` (the `fp8_ue8m0` layout)."""
        config = {"quantization_config": {
            "quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128]}}
        meta = _scan({
            "layers.0.attn.wq_a.weight": ("F8_E4M3", [128, 128], 128 * 128),
            "layers.0.attn.wq_a.scale": ("F8_E8M0", [1, 1], 1),
        }, config)
        self.assertEqual(meta["quant_scheme"], "fp8+fp8_ue8m0")
        self.assertEqual(meta["total_params"], 128 * 128)
        self.assertEqual(meta["quantization_metadata_params"], 1)

    def test_mixed_mxfp4_plain_and_packed_weights_keep_bf16_unpacked(self):
        # Both U8 naming variants are real packed weights; the same .weight
        # suffix on BF16 attention/norm tensors is not evidence of packing.
        meta = _scan({
            "experts.0.w1.weight_packed": ("U8", [8, 16], 128),
            "experts.0.w2.weight": ("U8", [8, 16], 128),
            "experts.0.w1.weight_scale": ("U8", [8, 1], 8),
            "experts.0.w2.weight_scale": ("U8", [8, 1], 8),
            "attention.weight": ("BF16", [8, 32], 512),
            "norm.weight": ("BF16", [8], 16),
        }, {"quantization_config": {"quant_method": "mxfp4-pack-quantized"}})
        self.assertEqual(meta["total_params"], 2 * 128 * 2 + 256 + 8)
        self.assertEqual(meta["quantization_metadata_params"], 16)
        self.assertEqual(meta["checkpoint_storage_bytes"], 800)
        self.assertEqual(meta["index_total_size"], 800)

    def test_the_hub_fetcher_takes_no_config(self):
        params = inspect.signature(_fetch_metadata_hf).parameters
        self.assertNotIn("config", params)
        self.assertIs(params["scheme"].default, inspect.Parameter.empty)
        self.assertIs(params["num_bits"].default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main()
