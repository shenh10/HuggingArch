"""Chunked-prefill SOL follows the endpoint's materialized launch schedule."""
from __future__ import annotations

import pytest

from backend.inference.capacity import estimate_capacity
from backend.inference.row_roofline import ROW, column_names
from backend.inference.sol_closure import check_closure
from tests.sweep_parity import Deployment, FixtureBench, diff, layout_at, report


pytestmark = pytest.mark.regression


MODEL = "meta-llama/Llama-3.2-1B"
CHUNK = 2048
ENDS = (2048, 4096, 6144)


def _estimate(prefix: int, *, end: int, chunked: bool, **over):
    deployment = Deployment(
        MODEL, knobs={"s_prefix_cached": prefix}, **over)
    return estimate_capacity(
        **deployment.env(),
        avg_input_tokens=end,
        prefill_batch_size=1,
        prefill_chunk_size=CHUNK if chunked else 0,
    )


def test_multi_chunk_sol_is_independent_single_chunk_estimates():
    with FixtureBench() as bench:
        deployment = Deployment(
            MODEL,
            selection_mode="envelope",
            measurement_scope=bench.scope,
            knobs={"s_prefix_cached": 0},
        )
        endpoint = estimate_capacity(
            **deployment.env(),
            avg_input_tokens=ENDS[-1],
            prefill_batch_size=2,
            prefill_chunk_size=CHUNK,
        )
        chunks = [
            _estimate(
                start,
                end=end,
                chunked=False,
                selection_mode="envelope",
                measurement_scope=bench.scope,
            )
            for start, end in zip((0, *ENDS[:-1]), ENDS, strict=True)
        ]

    phase = endpoint["phases"]["prefill"]
    check_closure(endpoint)
    assert phase["op_table_total"]["t_sol_s"] == pytest.approx(
        sum(row["t_op_s"] for row in endpoint["per_op_timing"]), rel=1e-12)
    summed = [chunk["phases"]["prefill"] for chunk in chunks]
    for column in ("flops_per_step", "mem_read_bytes", "mem_write_bytes",
                   "total_bytes", "t_compute_s", "t_memory_s", "t_comm_s",
                   "t_sol_s"):
        assert phase[column] == pytest.approx(
            2 * sum(chunk[column] for chunk in summed), rel=1e-12), column

    endpoint_rows = {row["uid"]: row for row in endpoint["per_op_timing"]}
    chunk_rows = [
        {row["uid"]: row for row in chunk["per_op_timing"]}
        for chunk in chunks
    ]
    assert endpoint_rows.keys() == chunk_rows[0].keys()
    for uid, row in endpoint_rows.items():
        for column in (*column_names(published_by=ROW, additive=True),
                       "count_per_step"):
            assert row[column] == pytest.approx(
                2 * sum(rows[uid][column] for rows in chunk_rows), rel=1e-12
            ), (uid, column)


def test_chunk_launches_repeat_weights_and_grow_attention_prefix_bytes():
    chunks = [
        _estimate(start, end=end, chunked=False)
        for start, end in zip((0, *ENDS[:-1]), ENDS, strict=True)
    ]
    rows = [{row["uid"]: row for row in chunk["per_op_timing"]}
            for chunk in chunks]

    embedding = rows[0][0]
    one_chunk = embedding["weight_read_bytes_per_rank"]
    assert one_chunk > 0
    assert all(rows[i][0]["weight_read_bytes_per_rank"]
               == one_chunk for i in (1, 2))
    assert sum(rows[i][0]["weight_read_bytes_per_rank"]
               for i in range(3)) == 3 * one_chunk

    attention = next(row for row in rows[0].values()
                     if row["op"] == "flash_attention")
    access = [rows[i][attention["uid"]]["access_bytes_per_rank"]
              for i in range(3)]
    assert access[0] > 0
    assert access[1] > access[0]
    assert access[2] > access[1]


def test_measured_is_at_least_sol_on_each_single_chunk_estimate(monkeypatch):
    monkeypatch.setattr(
        FixtureBench, "latency_us",
        staticmethod(lambda _key, _impl="k1": 1000.0))
    with FixtureBench() as bench:
        deployment = Deployment(
            MODEL,
            selection_mode="envelope",
            measurement_scope=bench.scope,
            knobs={"s_prefix_cached": 0},
        )
        endpoint = estimate_capacity(
            **deployment.env(),
            avg_input_tokens=ENDS[-1],
            prefill_batch_size=2,
            prefill_chunk_size=CHUNK,
        )
        chunks = [
            _estimate(
                start,
                end=end,
                chunked=False,
                selection_mode="envelope",
                measurement_scope=bench.scope,
            )
            for start, end in zip((0, *ENDS[:-1]), ENDS, strict=True)
        ]
        for chunk in chunks:
            phase = chunk["phases"]["prefill"]
            assert phase["t_measured_s"] is not None
            assert phase["t_measured_s"] >= phase["t_sol_s"]
        assert endpoint["phases"]["prefill"]["t_measured_s"] == pytest.approx(
            2 * sum(chunk["phases"]["prefill"]["t_measured_s"]
                    for chunk in chunks), rel=1e-12)


def test_a_one_chunk_endpoint_keeps_the_exact_prior_sol_payload():
    chunked = _estimate(0, end=CHUNK, chunked=True)
    unchunked = _estimate(0, end=CHUNK, chunked=False)
    diffs = diff(chunked, unchunked, path="one-chunk")
    assert not diffs, report(diffs)


def test_a_swept_long_context_reads_the_direct_schedule():
    from backend.inference.capacity import estimate_capacity_sweep

    deployment = Deployment(MODEL, knobs={"s_prefix_cached": 0})
    kwargs = {
        **deployment.env(),
        "avg_input_tokens": ENDS[-1],
        "prefill_batch_size": 2,
        "prefill_chunk_size": CHUNK,
    }
    direct = estimate_capacity(**kwargs)
    kwargs.pop("avg_input_tokens")
    swept = estimate_capacity_sweep(seq_lens=[ENDS[-1]], **kwargs)[0]
    assert swept["phases"]["prefill"]["t_sol_s"] == pytest.approx(
        direct["phases"]["prefill"]["t_sol_s"], rel=1e-12)
    assert [
        (row["uid"], *(row[column]
                       for column in column_names(published_by=ROW,
                                                  additive=True)))
        for row in swept["per_op_timing"]
    ] == [
        (row["uid"], *(row[column]
                       for column in column_names(published_by=ROW,
                                                  additive=True)))
        for row in direct["per_op_timing"]
    ]


def test_a_parallel_sweep_point_reads_the_direct_schedule():
    from backend.inference import parallel_sweep

    deployment = Deployment(MODEL, knobs={"s_prefix_cached": 0})
    payload = {
        key: value for key, value in deployment.env().items()
        if key not in parallel_sweep.PER_POINT_FIELDS
    }
    payload.update(
        gpu_skus=[deployment.gpu_sku],
        avg_input_tokens=ENDS[-1],
        prefill_chunk_size=CHUNK,
    )
    swept = layout_at(payload)[deployment.gpu_sku]["sol"]
    direct = estimate_capacity(
        **deployment.env(),
        avg_input_tokens=ENDS[-1],
        prefill_batch_size=swept["phases"]["prefill"]["batch"],
        prefill_chunk_size=CHUNK,
    )
    assert swept["phases"]["prefill"]["t_sol_s"] == pytest.approx(
        direct["phases"]["prefill"]["t_sol_s"], rel=1e-12)
    assert [
        (row["uid"], row["t_op_s"], row["weight_read_bytes_per_rank"])
        for row in swept["per_op_timing"]
    ] == [
        (row["uid"], row["t_op_s"], row["weight_read_bytes_per_rank"])
        for row in direct["per_op_timing"]
    ]
