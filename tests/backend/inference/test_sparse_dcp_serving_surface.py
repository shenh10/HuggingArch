"""Pin the public B300 serving rows for both phases of sparse DCP8."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot, snapshot_config


def serving_surface():
    model = "moonshotai__Kimi-K3.1-Pro"
    result = estimate_capacity(
        config=snapshot_config(model), resolved=resolve_snapshot(model),
        gpu_sku="B300", num_gpus=8, framework="sglang",
        parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                tp_moe_expert=4, ep_moe=2, dcp=8),
        selection_mode="theoretical_only", avg_input_tokens=4096,
        avg_output_tokens=1024, prefill_batch_size=1, decode_batch_size=1,
    )
    surface = {}
    for phase, field in (("prefill", "per_op_timing"), ("decode", "per_op_timing_decode")):
        rows = result[field]
        assert rows, phase
        identities = [
            [row["aggregation_id"], row["op"],
             {key: val for key, val in row.items() if "dtype" in key}]
            for row in rows
        ]
        surface[phase] = {
            "rows_by_op": dict(Counter(row["op"] for row in rows)),
            "ordered_identity_and_dtypes_sha256": hashlib.sha256(
                json.dumps(identities, sort_keys=True).encode()
            ).hexdigest(),
        }
    return surface


@pytest.mark.regression
def test_sparse_dcp8_serving_surface():
    golden = Path(__file__).resolve().parents[2] / "golden" / "kimi_k31_dcp8.json"
    assert serving_surface() == json.loads(golden.read_text())
