"""Providers decline the cells their kernel cannot run, before inputs exist.

Each case here is a cell a sweep handed a kernel on every pass and the
library refused at launch: Gemma-2's fp32 cells to FlashAttention, FlashInfer
and sgl_kernel's rmsnorm (whose dispatch tables stop at fp16/bf16), and
192-wide V cells to FlashInfer's dense wrappers (whose V-width dispatch is
64/128/256/512). The decline comes from the provider's declaration, so the
smoke cell, the published capability record and the stale-row audit read
the same limit. Each provider must also still take the neighbouring cell it
does run — a gate that refuses everything also "fixes" the failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from kernel_bench.env import Env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import (
    KernelSignature, Provider, all_providers, register)

FP32 = {"activation": "fp32", "kv": "fp32", "output": "fp32",
        "weight": "fp32"}
BF16 = {"activation": "bf16", "kv": "bf16", "output": "bf16",
        "weight": "bf16"}

# Gemma-2-9b's planned shapes (H100 job file), and MiMo's 192-QK attention.
_GEMMA_ATTN = {"B": 1, "s_query": 1024, "s_kv": 1024, "n_heads": 16,
               "n_kv_heads": 8, "d_head": 128, "kv_block_size": 16}
_GEMMA_DECODE = dict(_GEMMA_ATTN, s_query=1)
_GEMMA_VERIFY = dict(_GEMMA_ATTN, s_query=4)
_V192 = {"B": 1, "s_query": 1024, "s_kv": 1024, "n_heads": 4,
         "n_kv_heads": 1, "d_head": 192, "kv_block_size": 16,
         "s_window": 512}
_V192_DECODE = dict(_V192, s_query=1)
_ROPE = {"B": 1, "s_query": 1024, "n_heads": 16, "n_kv_heads": 8,
         "d_rope": 256}
_NORM = {"B": 1, "s_query": 1024, "d_model": 3584}


def _cell(signature, dims, dtype_env, phase="prefill") -> Invocation:
    return Invocation.from_job({
        "signature": signature, "op": signature, "dims": dict(dims),
        "phase": phase, "dtype_env": dict(dtype_env)})


def _provider(library, impl) -> Provider:
    for p in all_providers():
        if (p.library, p.kernel_impl) == (library, impl):
            return p
    raise LookupError(f"{library}:{impl} is not registered")


_ENV = Env(torch_version=None, device=None, device_name=None,
           gpu_sku=None, peak_bw_gbps=None)


# (provider, signature, phase, the failing cell, the clause that declines it,
#  the neighbouring cell it still runs). The FIRST case is the one the default
# suite runs: fa2 on Gemma-2 fp32, the largest single source of failures.
CASES = [
    ("fa2", "flash_attention_v2", "gqa_attention", "prefill",
     dict(_GEMMA_ATTN, d_head=256), FP32, "activation:fp32",
     dict(_GEMMA_ATTN, d_head=256), BF16),
    ("fa2", "flash_attention_v2_kvcache", "gqa_attention", "decode",
     _GEMMA_DECODE, FP32, "activation:fp32", _GEMMA_DECODE, BF16),
    ("fa3", "flash_attention_v3", "gqa_attention", "prefill",
     _GEMMA_ATTN, FP32, "activation:fp32", _GEMMA_ATTN, BF16),
    ("fa3", "flash_attention_v3_kvcache", "gqa_attention", "decode",
     _GEMMA_DECODE, FP32, "activation:fp32", _GEMMA_DECODE, BF16),
    ("flashinfer", "flashinfer_ragged_prefill", "gqa_attention", "prefill",
     _GEMMA_ATTN, FP32, "activation:fp32", _GEMMA_ATTN, BF16),
    ("flashinfer", "flashinfer_paged_decode", "gqa_attention", "decode",
     _GEMMA_DECODE, FP32, "activation:fp32", _GEMMA_DECODE, BF16),
    ("flashinfer", "flashinfer_paged_verify", "gqa_attention", "decode",
     _GEMMA_VERIFY, FP32, "activation:fp32", _GEMMA_VERIFY, BF16),
    # V 192 wide: refused. MiMo's 192-QK / 128-V cell: V is 128, served.
    ("flashinfer", "flashinfer_ragged_prefill", "gqa_attention", "prefill",
     _V192, BF16, "operand_enum:v[-1]", dict(_V192, d_v_head=128), BF16),
    ("flashinfer", "flashinfer_paged_decode", "gqa_attention", "decode",
     _V192_DECODE, BF16, "operand_enum:v[-1]",
     dict(_V192_DECODE, d_head=256), BF16),
    ("flashinfer", "flashinfer_paged_verify", "gqa_attention", "decode",
     dict(_V192, s_query=4), BF16, "operand_enum:v[-1]",
     dict(_V192, s_query=4, d_v_head=128), BF16),
    ("flashinfer", "flashinfer_rope", "rope", "prefill",
     _ROPE, FP32, "activation:fp32", _ROPE, BF16),
    ("sglang", "sgl_rmsnorm", "rmsnorm", "prefill",
     _NORM, FP32, "activation:fp32", _NORM, BF16),
]


@pytest.mark.parametrize(
    "library,impl,signature,phase,bad,bad_dt,clause,good,good_dt", CASES,
    ids=[f"{c[1]}-{c[6]}" for c in CASES])
def test_provider_declines_what_its_kernel_refuses(
        library, impl, signature, phase, bad, bad_dt, clause, good, good_dt):
    p = _provider(library, impl)
    refused = list(p.refusals(_cell(signature, bad, bad_dt, phase), _ENV))
    assert any(token.startswith(clause) for token in refused), refused
    ok = _cell(signature, good, good_dt, phase)
    assert list(p.refusals(ok, _ENV)) == []


def test_sglang_fused_add_rmsnorm_keeps_fp32():
    """sgl_kernel's fused_add_rmsnorm is sglang's own kernel, whose dispatch
    includes float (fused_add_rms_norm_kernel.cu:43) — the rmsnorm limit is
    FlashInfer's, and copying it here would drop cells that run."""
    p = _provider("sglang", "sgl_fused_add_rmsnorm")
    assert p.accepts(_cell("fused_add_rmsnorm", _NORM, FP32), _ENV)


class TestOperandEnum:
    """The extent comes off the tensor's layout for THIS cell, so the
    asymmetric-V rule lives in the signature only."""

    def test_reads_the_variant_width_not_the_default_one(self):
        serves = KernelSignature(signature="gqa_attention",
                                 activation=("bf16",),
                                 dtype_slots=(("kv", ("bf16",)),),
                                 variant_dims=("s_window", "d_v_head"),
                                 operand_enum=(("v", -1, (128,)),))
        narrow = _cell("gqa_attention", dict(_V192, d_v_head=128), BF16)
        same = _cell("gqa_attention", _V192, BF16)
        assert list(serves.refusals(narrow)) == []
        assert list(serves.refusals(same)) == [
            "operand_enum:v[-1]:192 is not served (serves 128)"]

    def test_is_published(self):
        serves = KernelSignature(signature="gqa_attention",
                                 operand_enum=(("v", -1, (64, 128)),))
        assert "v.shape[-1] in [64, 128]" in (
            serves.describe()["shape_constraints"])

    @pytest.mark.parametrize("entry,match", [
        (("w", -1, (64,)), "not a contract tensor"),
        (("v", 4, (64,)), "outside"),
        (("v", -1, ()), "empty"),
    ])
    def test_registration_refuses_a_bad_entry(self, entry, match):
        p = Provider(library="torch", signature="gqa_attention",
                     kernel_impl="probe", import_name="torch",
                     serves=KernelSignature(signature="gqa_attention",
                                            kv_block_sizes=(),
                                            operand_enum=(entry,)),
                     build=lambda inv, env, inputs: None)
        with pytest.raises(ValueError, match=match):
            register(p)
