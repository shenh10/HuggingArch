"""Endpoint folding of one ``PointMeasure`` per prefill chunk."""
from __future__ import annotations

from dataclasses import replace

import pytest

from backend.inference.batch_sweep import apply_measured, fold_chunk_point_measures
from backend.inference.phases import phase_binding
from backend.measure.auto_sweep import MEASURED, MISSING, MeasuredFormula, PointMeasure
from backend.measure.lookup import Key


C = 8192


def _key(op: str, s_kv: int) -> Key:
    return Key(op, f'{{"s_query":8192,"s_kv":{s_kv}}}', "{}",
               "H100-SXM", "prefill")


def _chunk_measure(
        s_kv: int, *, token_inputs=None, context=None, cadence=1,
        token_weight=1, token_launches=3.0, attention_launches=2.0):
    token_inputs = {"flops": 100, "anchor": None, "dispatch": None}
    context = context or {"gpu": "H100-SXM", "framework": "vllm"}
    text = ("" if token_weight == 1 else f"({token_weight})*") \
        + "n_0*T_0 + n_1*T_1"
    measure = PointMeasure(
        (_key("gemm", C), _key("gqa_attention", s_kv)),
        (token_launches, attention_launches), (None, None),
        ("gemm", "gqa_attention"), ("", ""), measure_formula(text),
        context, (token_inputs, {"flops": 200, "anchor": None,
                                 "dispatch": None}))
    return {
        "measure": measure.to_json(),
        "terms": [["per_op_timing", "token"],
                  ["per_op_timing", "attention"]],
        "cadences": (cadence, 1),
    }


def measure_formula(text):
    from backend.measure.auto_sweep import MeasuredFormula
    return MeasuredFormula(text, (False, False))


def _map(chunks=3, batch=2):
    return {
        "deployment": "tp=1", "phase": "prefill", "endpoint_index": 0,
        "endpoint": {"seq_len": chunks * C, "s_prefix_cached": 0,
                     "s_verify": 1, "batch": batch},
        "chunks": [{"index": i, "start": i * C, "end": (i + 1) * C,
                    "width": C, "launch_batch": 1}
                   for i in range(chunks)],
    }


def _three_chunks():
    return _map(), [
        _chunk_measure(C), _chunk_measure(2 * C), _chunk_measure(3 * C)]


def _select(times):
    by_key = {key: time for key, time in times.items()}

    def select(key, _context, _op, _read):
        if key not in by_key:
            return None, MISSING, None
        return by_key[key], MEASURED, {"source": f"measured:{key.shape_json}",
                                       "source_badge": {}, "utilization": {}}
    return select


def test_an_endpoint_is_the_sum_of_its_chunk_point_measures():
    chunk_map, chunks = _three_chunks()
    endpoint = fold_chunk_point_measures(chunk_map, chunks)
    measure = PointMeasure.from_json(endpoint["measure"])

    # Two endpoint repetitions: token work is one exact c-token key with six
    # per-chunk launch counts; each growing context is its own physical key.
    assert measure.launches == (18.0, 4.0, 4.0, 4.0)
    assert [key.shape_json for key in measure.keys] == [
        f'{{"s_query":8192,"s_kv":{value}}}'
        for value in (C, C, 2 * C, 3 * C)]
    assert endpoint["terms"] == [
        ["per_op_timing", "token"], ["per_op_timing", "attention"],
        ["per_op_timing", "attention"], ["per_op_timing", "attention"]]

    times = {
        measure.keys[0]: 0.002,
        measure.keys[1]: 0.003,
        measure.keys[2]: 0.005,
        measure.keys[3]: 0.007,
    }
    folded = measure.fold(_select(times))
    assert folded.track.t_s == pytest.approx(
        18 * 0.002 + 4 * (0.003 + 0.005 + 0.007))
    # The old endpoint-sized cell is not one of the requested keys.
    assert _key("gqa_attention", 3 * C)._replace(signature="old") not in times


def test_one_missing_context_key_makes_the_whole_endpoint_incomplete():
    chunk_map, chunks = _three_chunks()
    measure = PointMeasure.from_json(
        fold_chunk_point_measures(chunk_map, chunks)["measure"])
    times = {
        measure.keys[0]: 0.002,
        measure.keys[1]: 0.003,
        measure.keys[3]: 0.007,
    }
    folded = measure.fold(_select(times))
    assert folded.track.t_s is None
    coverage = folded.track.coverage()
    assert coverage["measured"] == 3
    assert coverage["states"]["missing"] == [
        f"gqa_attention {measure.keys[2].shape_json}"]


def test_one_graph_row_owns_all_of_its_chunk_terms():
    chunk_map, chunks = _three_chunks()
    endpoint = fold_chunk_point_measures(chunk_map, chunks)
    measure = PointMeasure.from_json(endpoint["measure"])
    times = {key: 0.002 * (i + 1) for i, key in enumerate(measure.keys)}
    point = {
        "per_op_timing": [
            {"uid": "token"}, {"uid": "attention"}],
        "phases": {"prefill": {}},
    }
    folded = measure.fold(_select(times))
    apply_measured(point, {"prefill": {**endpoint, "phase": "prefill"}},
                   {"prefill": folded})
    token, attention = point["per_op_timing"]
    assert token["t_measured_s"] == pytest.approx(0.036)
    # This leaf aggregates all three distinct Lq terms, not only the last one.
    assert attention["t_measured_s"] == pytest.approx(0.072)
    assert attention["measured_state"] == MEASURED
    assert point["phases"]["prefill"]["t_measured_s"] == pytest.approx(0.108)


def test_duplicate_keys_with_different_fold_laws_do_not_collapse():
    base = _chunk_measure(C)
    variants = {
        "formula weight": _chunk_measure(C, token_weight=3),
        "selection inputs": {
            **_chunk_measure(C),
            "measure": replace(PointMeasure.from_json(
                _chunk_measure(C)["measure"]),
                inputs=({"flops": 101, "anchor": None, "dispatch": None},
                        {"flops": 200, "anchor": None, "dispatch": None})
            ).to_json(),
        },
        "selection context": _chunk_measure(C, context={
            "gpu": "B300-SXM", "framework": "vllm"}),
        "per-term selection context": {
            **_chunk_measure(C),
            "measure": replace(
                PointMeasure.from_json(_chunk_measure(C)["measure"]),
                contexts=({"gpu": "B300-SXM", "framework": "vllm"},
                          {"gpu": "B300-SXM", "framework": "vllm"})).to_json(),
        },
        "cadence": _chunk_measure(C, cadence=3),
    }
    for name, variant in variants.items():
        endpoint = fold_chunk_point_measures(
            _map(chunks=2, batch=1), [base, variant])
        measure = PointMeasure.from_json(endpoint["measure"])
        assert len(measure.keys) == (
            4 if "selection context" in name else 3), name
        assert measure.launches[0] == measure.launches[1] == 3.0, name
        if "selection context" in name:
            assert measure.contexts[0] != measure.contexts[1]


def _projected_chunk_measure(model_id, signature, wanted):
    """Plan a real framework's 3x4096 endpoint and select one attention row."""
    from kernel_bench.plan import PlanPoint, build_jobs_over_points
    from kernel_bench.prefill_chunks import decompose_prefill, prefill_chunk_map
    from tests.snapshots import resolve_snapshot

    endpoint = PlanPoint(12_288, 0, 1, 1)
    chunks = decompose_prefill(endpoint, prefill_chunk_size=4096)
    by_point = build_jobs_over_points(
        model_id=model_id, phase="prefill", avg_output_tokens=0,
        points=[chunk.point for chunk in chunks],
        resolved=resolve_snapshot(model_id), with_unplaced=True)
    selected = []
    for chunk in chunks:
        jobs = by_point[chunk.point][0]
        rows = [job for job in jobs if job["signature"] == signature
                and wanted(job["dims"])]
        assert rows, f"{model_id} lost its {signature} row at {chunk}"
        selected.append(rows[0])
    return prefill_chunk_map(endpoint, prefill_chunk_size=4096), selected


def _one_attention_term_measure(job):
    signature, shape, dtype, phase = job["physical_key"]
    key = Key(signature, shape, dtype, "B300", phase)
    return {
        "measure": PointMeasure(
            (key,), (1.0,), (None,), ("attention",), ("",),
            MeasuredFormula("n_0*T_0", (False,)),
            {"gpu": "B300", "framework": "vllm"},
            ({"flops": 1, "anchor": None, "dispatch": None},),
        ).to_json(),
        "terms": [["per_op_timing", "attention"]],
        "cadences": (1,),
    }


def test_gpt_oss_swa_projects_repeated_chunk_keys_and_merges_launches():
    """gpt-oss' distant SWA chunks share one projected exact key.

    At ``L=12288,c=4096``, the second chunk and tail both project to
    ``s_kv=4223``.  Their exact keys collapse, while endpoint repetition and
    launch counts still add.
    """
    _assert_projected_context_keys_merge_as_launch_counts(
        "openai/gpt-oss-20b", "gqa_attention",
        lambda dims: dims.get("s_window") == 128, (4096, 4223, 4223))


def test_glm_kpool_projects_repeated_chunk_keys_and_merges_launches():
    """GLM's k-pool projection is exercised by a second real framework.

    All three 4096-token launches select an effective 2051-key cache, so the
    exact key collapses to one lookup with six endpoint launches at batch 2.
    """
    _assert_projected_context_keys_merge_as_launch_counts(
        "zai-org/GLM-5.3-Flash", "mla_sparse_attention",
        lambda dims: dims.get("n_topk_keys") == 2051,
        (2051, 2051, 2051))


def _assert_projected_context_keys_merge_as_launch_counts(
        model_id, signature, wanted, effective_s_kv):
    chunk_map, planned = _projected_chunk_measure(
        model_id, signature, wanted)
    assert [job["dims"]["s_kv"] for job in planned] == list(effective_s_kv)
    keys = [tuple(job["physical_key"]) for job in planned]
    if signature == "gqa_attention":
        assert keys[1] == keys[2]
        assert keys[0] != keys[1]
    else:
        assert len(set(keys)) == 1

    endpoint = fold_chunk_point_measures(
        {**chunk_map, "endpoint": {**chunk_map["endpoint"], "batch": 2}},
        [_one_attention_term_measure(job) for job in planned])
    measure = PointMeasure.from_json(endpoint["measure"])
    by_shape = {key.shape_json: launch for key, launch
                in zip(measure.keys, measure.launches, strict=True)}
    expected_counts = {
        keys[0][1]: 2 * effective_s_kv.count(effective_s_kv[0]),
        keys[-1][1]: 2 * effective_s_kv.count(effective_s_kv[-1]),
    }
    assert by_shape == expected_counts
    times = {key: 0.003 if key.shape_json == keys[-1][1] else 0.002
             for key in measure.keys}
    folded = measure.fold(_select(times)).track
    assert folded.t_s == pytest.approx(sum(
        launch * times[key] for key, launch in zip(
            measure.keys, measure.launches, strict=True)))


def test_a_one_chunk_endpoint_is_returned_unchanged():
    chunk = _chunk_measure(C)
    chunk_map = _map(chunks=1, batch=1)
    assert fold_chunk_point_measures(chunk_map, [chunk]) == chunk


def test_chunk_causal_work_rebuilds_the_endpoint_triangle():
    endpoint = phase_binding(phase="prefill", avg_input_tokens=3 * C,
                             avg_output_tokens=0, s_prefix_cached=0)
    chunks = []
    for start, end in ((0, C), (C, 2 * C), (2 * C, 3 * C)):
        chunks.append(phase_binding(phase="prefill", avg_input_tokens=end,
                                    avg_output_tokens=0,
                                    s_prefix_cached=start))
    assert sum(chunk["s_pairs"] for chunk in chunks) == endpoint["s_pairs"]


def _long_estimate(policy=None, *, parallel=None, missing_last=False):
    from backend.inference.capacity import estimate_capacity
    from tests.sweep_parity import Deployment, FixtureBench

    def wants_missing(key):
        return ("missing" if key.signature == "gqa_attention"
                and '"s_kv":5120' in key.shape_json else "measured")

    policy = policy or (wants_missing if missing_last else None)
    bench = FixtureBench(policy)
    bench.__enter__()
    try:
        deployment = Deployment(
            "meta-llama/Llama-3.2-1B", parallel=parallel,
            selection_mode="envelope", measurement_scope=bench.scope)
        estimate = estimate_capacity(
            **deployment.env(), avg_input_tokens=5 * 1024,
            prefill_batch_size=2, prefill_chunk_size=2048)
    finally:
        bench.__exit__(None, None, None)
    return bench, estimate


def test_runtime_long_prefill_reads_each_chunk_key_once_including_its_tail():
    bench, estimate = _long_estimate()
    phase = estimate["phases"]["prefill"]
    assert phase["t_measured_s"] is not None
    assert phase["coverage"]["measured"] == phase["coverage"]["total"]

    attention = [key for key in bench.keys_asked
                 if key.signature == "gqa_attention" and key.phase == "prefill"]
    assert [key.shape_json for key in attention] == [
        f'{{"B":1,"d_head":64,"kv_block_size":16,"n_heads":32,'
        f'"n_kv_heads":8,"s_kv":{context},"s_query":{query}}}'
        for context, query in ((2048, 2048), (4096, 2048),
                               (5120, 1024))]
    token_keys = [key for key in bench.keys_asked
                  if key.phase == "prefill"
                  and key.signature in ("gemm", "rmsnorm", "silu_and_mul")]
    assert token_keys
    assert any('"s_query":2048' in key.shape_json for key in token_keys)
    assert any('"s_query":1024' in key.shape_json for key in token_keys)
    assert not any('"s_query":5120' in key.shape_json for key in token_keys)


def test_runtime_endpoint_is_incomplete_when_one_context_key_is_missing():
    _bench, estimate = _long_estimate(missing_last=True)
    phase = estimate["phases"]["prefill"]
    assert phase["t_measured_s"] is None
    assert phase["coverage"]["measured"] == phase["coverage"]["total"] - 1
    assert len(phase["coverage"]["states"]["missing"]) == 1
    assert '"s_kv":5120' in phase["coverage"]["states"]["missing"][0]


def test_runtime_short_context_keeps_the_existing_measure_path():
    from backend.inference.capacity import estimate_capacity
    from tests.sweep_parity import Deployment, FixtureBench

    with FixtureBench() as bench:
        deployment = Deployment(
            "meta-llama/Llama-3.2-1B", selection_mode="envelope",
            measurement_scope=bench.scope)

        def measured_view(chunked):
            estimate = estimate_capacity(
                **deployment.env(), avg_input_tokens=2048,
                prefill_chunk_size=2048 if chunked else 0)
            phase = estimate["phases"]["prefill"]
            return (
                phase["t_measured_s"], phase["coverage"],
                [(row.get("uid"), row.get("t_measured_s"),
                  row.get("measured_state"), row.get("measured_key"))
                 for row in estimate["per_op_timing"]])

        assert measured_view(True) == measured_view(False)


# ---------------------------------------------------------------------------
# A sweep folds one endpoint once and hands it to every point of it.
# ---------------------------------------------------------------------------
#
# Two prefill batches, two decode batches each: the second point at one
# prefill batch reads the fold the first wrote (`capacity._sweep_cell`'s
# `endpoints` memo, keyed by `capacity._endpoint_memo_key`), and the points
# at the other prefill batch must not. Order matters: the default run takes
# the FIRST case only (CLAUDE.md §5.5), so it is the point that both reads a
# memo and belongs to the second endpoint -- the one that turns red whether
# the memo's contents go wrong or its key stops telling the two endpoints
# apart.
_FOLD_POINTS = ("p2d8", "p2d1", "p1d8", "p1d1")
_FOLD_GRID = [{"key": "p1d1", "prefill_batch_size": 1, "decode_batch_size": 1},
              {"key": "p1d8", "prefill_batch_size": 1, "decode_batch_size": 8},
              {"key": "p2d1", "prefill_batch_size": 2, "decode_batch_size": 1},
              {"key": "p2d8", "prefill_batch_size": 2, "decode_batch_size": 8}]
_FOLD_SEQ = 5 * 1024
_FOLD_CHUNK = 2048


@pytest.fixture(scope="module")
def folded_sweep():
    """The sweep of `_FOLD_GRID` at one long context, and the deployment a
    direct estimate at a point of it reads — one sweep for the whole
    parametrization (the walk is the cost of it), under a stand-in that
    measures every key, kept open so the direct estimates read the same
    stand-in."""
    from backend.inference.capacity import estimate_capacity_sweep
    from tests.sweep_parity import Deployment, FixtureBench
    with FixtureBench() as bench:
        deployment = Deployment(
            "meta-llama/Llama-3.2-1B", selection_mode="envelope",
            measurement_scope=bench.scope, knobs={"s_prefix_cached": 0})
        yield estimate_capacity_sweep(
            seq_lens=[_FOLD_SEQ], batch_grid=_FOLD_GRID,
            prefill_chunk_size=_FOLD_CHUNK, **deployment.env()), deployment


@pytest.mark.parametrize("key", _FOLD_POINTS)
def test_each_swept_point_of_a_chunked_endpoint_is_the_direct_estimate(
        key, folded_sweep):
    """A point of a (context x batch) sweep whose prefill is chunked is the
    direct estimate at that point, whole: the folded per-op rows, the folded
    phase and its measured total, the capacity fields.

    The sweep folds each chunked endpoint once and hands the result to every
    point that shares it, so the points at one prefill batch differ only in
    their decode batch -- this is what reads that memo, and what fails if the
    key ever stops determining the fold (the grid holds two prefill batches,
    whose folds differ by a factor of the batch)."""
    from backend.inference.capacity import estimate_capacity
    from tests.sweep_parity import diff, report, without_breakdown
    swept, deployment = folded_sweep
    point = next(p for p in _FOLD_GRID if p["key"] == key)
    direct = estimate_capacity(
        **deployment.env(), avg_input_tokens=_FOLD_SEQ,
        prefill_batch_size=point["prefill_batch_size"],
        decode_batch_size=point["decode_batch_size"],
        prefill_chunk_size=_FOLD_CHUNK)
    direct.pop("vs_naive_delta")
    prefill = direct["phases"]["prefill"]
    # The point this guards is a folded endpoint read by more than one point:
    # assert the fixture still produces one.
    assert len(prefill["prefill_chunk_map"]["chunks"]) >= 2
    assert prefill["batch"] == point["prefill_batch_size"]
    assert prefill["t_measured_s"] is not None
    diffs = diff(swept[key][0], without_breakdown(direct, keyed=True))
    assert not diffs, f"{key}:\n{report(diffs)}"


def test_the_endpoint_memo_key_names_the_width_the_schedule_does_not():
    """The memo key tells two foldings apart whenever the fold would differ,
    and that includes the verify width — which a schedule does NOT carry
    (`PrefillChunkSchedule`'s endpoint is materialized at `s_verify=1`
    whatever the request asks for), while the fold prices the launches at
    the preparation's width. Without the width in the key, lifting the memo
    out of one context would serve a K=15 fold to a K=1 point."""
    from backend.inference import capacity
    from tests.sweep_parity import Deployment
    deployment = Deployment("meta-llama/Llama-3.2-1B",
                            knobs={"s_prefix_cached": 0})
    plan = capacity.CapacitySweepPlan.of(
        seq_lens=[_FOLD_SEQ], prefill_chunk_size=_FOLD_CHUNK,
        **deployment.env())
    (cell,) = plan.cells()
    (schedule,) = plan.prefill_chunk_schedules
    at = {width: capacity._preparation_at(cell.walk, s_verify=width)
          for width in (1, 4)}
    key = capacity._endpoint_memo_key
    # The schedule says the same thing at either width: only the key can
    # tell the two folds apart.
    assert schedule.endpoint.s_verify == 1
    assert key("sol", at[1], schedule) == key("sol", at[1], schedule)
    assert key("sol", at[1], schedule) != key("sol", at[4], schedule)
    assert key("sol", at[1], schedule) != key(
        "sol", at[1], schedule.with_endpoint_batch(2))
    assert key("sol", at[1], schedule) != key("measures", at[1], schedule)


def test_runtime_pp2_long_prefill_keeps_pipeline_coverage_honest():
    from backend.inference.parallel_config import ParallelConfig
    bench, estimate = _long_estimate(parallel=ParallelConfig(pp=2))
    phase = estimate["phases"]["prefill"]
    assert phase["t_measured_s"] is None
    assert phase["coverage"]["states"]["unplaced"]
    attention = [key for key in bench.keys_asked
                 if key.signature == "gqa_attention" and key.phase == "prefill"]
    assert [key.shape_json for key in attention] == [
        f'{{"B":1,"d_head":64,"kv_block_size":16,"n_heads":32,'
        f'"n_kv_heads":8,"s_kv":{context},"s_query":{query}}}'
        for context, query in ((2048, 2048), (4096, 2048),
                               (5120, 1024))]
    assert not any('"s_query":5120,' in key.shape_json
                   for key in bench.keys_asked
                   if key.signature != "gqa_attention")


def test_a_chunked_endpoint_counts_every_request_s_tokens():
    """Folding an endpoint's chunks changes its time, not its workload: the
    tokens one step processes are the endpoint's `B × s_query`, as the
    unchunked estimate of the same request says. The fold once took the
    first chunk's workload (one launch of one request), so at 128K the
    Scaling panel's prefill rate was short by the prefill batch."""
    from backend.inference.capacity import estimate_capacity
    from tests.sweep_parity import Deployment
    deployment = Deployment("meta-llama/Llama-3.2-1B", parallel=None,
                            selection_mode="envelope")
    whole, chunked = (estimate_capacity(
        **deployment.env(), avg_input_tokens=5 * 1024, prefill_batch_size=2,
        prefill_chunk_size=chunk)["phases"]["prefill"] for chunk in (None, 2048))
    assert len(chunked["prefill_chunk_map"]["chunks"]) >= 2
    assert chunked["binding"] == whole["binding"]
    assert chunked["binding"]["effective_tokens_per_step"] == 2 * 5 * 1024
    sol = chunked["throughput"]["sol"]
    assert sol["per_group"] == pytest.approx(2 * 5 * 1024 / chunked["t_sol_s"])
