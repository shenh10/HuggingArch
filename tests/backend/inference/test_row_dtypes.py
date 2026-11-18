"""Each per-op row states the element dtype of every tensor its shape
columns name (`input_dtypes` per `shape_in` input, `output_dtype` for
`shape_out`, `port_dtypes` per `shape_outs` port), as the walk billed the
read or write — so the per-op table can show a format change, which the
shapes alone never do."""
from __future__ import annotations

import pytest

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot, snapshot_config

LLAMA = "meta-llama/Llama-3.2-3B"


@pytest.fixture(scope="module")
def decode_rows():
    est = estimate_capacity(
        config=snapshot_config(LLAMA), model_id=LLAMA,
        resolved=resolve_snapshot(LLAMA), gpu_sku="H100-SXM",
        parallel=ParallelConfig(), num_gpus=1, gpu_memory_fraction=0.9,
        avg_input_tokens=1024, avg_output_tokens=64,
        prefill_batch_size=2, decode_batch_size=2,
        selection_mode="theoretical_only", measurement_scope=None,
        precision={"preset": "w_fp8_kv_fp8"})
    return est["per_op_timing_decode"]


def _row(rows, path):
    return next(r for r in rows if r.get("display_path") == path)


def test_an_fp8_gemm_reads_fp8_and_writes_the_activation_dtype(decode_rows):
    """Under FP8 weights the projection's kernel eats its operand in FP8
    (the quantised copy the row bills) and writes BF16: the two sides of
    the shape column differ in dtype where they agree in nothing else."""
    o_proj = _row(decode_rows, "self_attn.o_proj")
    assert o_proj["input_dtypes"] == ["F8_E4M3"]
    assert o_proj["output_dtype"] == "BF16"


def test_every_input_has_its_own_slot(decode_rows):
    """One dtype per input tensor, in `shape_in` order: a residual add
    states both operands, not the first one's for the pair."""
    add = _row(decode_rows, "out__block")
    assert add["input_dtypes"] == ["BF16", "BF16"]
    for row in decode_rows:
        shape_in = row.get("shape_in")
        if not shape_in:
            assert row.get("input_dtypes") is None, row.get("display_path")
            continue
        arity = len(shape_in) if isinstance(shape_in[0], list) else 1
        assert len(row["input_dtypes"]) == arity, row.get("display_path")


def test_the_weight_operand_is_named_with_its_shape_and_dtype(decode_rows):
    """A projection's own weight is an operand the activation columns never
    name: the row lists it, as one rank holds it, at its resident dtype —
    the FP8 the preset stores it at, not the activation's BF16."""
    o_proj = _row(decode_rows, "self_attn.o_proj")
    assert [(w["shape"], w["dtype"]) for w in o_proj["weights"]] == [
        ([3072, 3072], "F8_E4M3")]
    assert o_proj["weights"][0]["name"].endswith("o_proj.weight")
