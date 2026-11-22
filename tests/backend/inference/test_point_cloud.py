"""One point cloud: coordinates, grid rungs, and absent points as data."""
from __future__ import annotations

import pytest
import ast
import pathlib

from backend.inference.capacity import CapacityKnobs, CapacitySweepPlan
from backend.inference.frontier import (POINT_DIMS, SINGLE_GPU_MODE,
                                        absent_point, frontier_view,
                                        parallel_mode)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.sweep_grid import (CONTEXT_GRID, CURATED_CONTEXTS,
                                          GPU_COUNT_GRID,
                                          SCALING_CONTEXTS,
                                          SCALING_DECODE_LADDER,
                                          SCALING_DEGREES)


def test_only_the_point_cloud_module_makes_points():
    """The vocabulary may live in `frontier`, but point construction — served
    values and absence alike — has one producer interface."""
    root = pathlib.Path(__file__).resolve().parents[3] / "backend"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name != "point_cloud.py":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    called = node.func
                    name = (called.id if isinstance(called, ast.Name)
                            else called.attr if isinstance(called, ast.Attribute)
                            else "")
                    if name in {"absent_point", "cloud_point"}:
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


def test_only_the_producer_appends_cloud_points():
    """Adapters may enumerate records; only `produce_cloud` fills a cloud."""
    root = pathlib.Path(__file__).resolve().parents[3] / "backend"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (path.name == "point_cloud.py"
                    and isinstance(node, ast.Attribute)
                    and node.attr == "append"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "points"):
                continue
            if (isinstance(node, ast.Attribute)
                    and node.attr == "append"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "points"):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []


def test_every_surface_context_is_a_measured_context():
    assert 65536 not in CONTEXT_GRID
    assert set(CURATED_CONTEXTS) <= set(CONTEXT_GRID)
    assert list(CURATED_CONTEXTS) == SCALING_CONTEXTS


def test_the_gpu_and_decode_ladders_are_grid_subsets():
    assert [x for x in GPU_COUNT_GRID if x <= 128] == SCALING_DEGREES
    expected = (1, 8, 16, 32, 64, 128, 256, 512, 1024)
    assert expected == SCALING_DECODE_LADDER


def test_an_absent_point_states_every_coordinate_and_a_closed_reason():
    coords = {name: 0 for name in POINT_DIMS}
    point = absent_point(coords, "capacity", "the cache does not fit")
    assert point["refused"] == "capacity"
    assert point["reason"] == "the cache does not fit"
    with pytest.raises(ValueError, match="missing \\['context'\\]"):
        absent_point({k: v for k, v in coords.items() if k != "context"},
                     "capacity", "missing")
    with pytest.raises(ValueError, match="unknown point refusal"):
        absent_point(coords, "somewhere_else", "not a vocabulary word")


def test_a_served_point_also_states_every_coordinate():
    from backend.inference.point_cloud import CloudRecord, produce_cloud

    coords = {name: 0 for name in POINT_DIMS}
    with pytest.raises(ValueError, match="missing=\\['context'\\]"):
        produce_cloud([CloudRecord(
            coordinates={k: v for k, v in coords.items()
                        if k != "context"},
            estimate=_est(0.01))])


def test_the_producer_keeps_each_records_source_token():
    """A Gallery record's coverage index follows its filled point."""
    from backend.inference.point_cloud import CloudRecord, produce_cloud

    coords = {name: 0 for name in POINT_DIMS}
    cloud = produce_cloud([
        CloudRecord(coordinates=coords, estimate=_est(0.01), token="source"),
    ])
    assert cloud.tokens[id(cloud.points[0])] == "source"


def test_a_view_admits_an_absent_point_but_no_front_uses_it():
    coords = {name: 0 for name in POINT_DIMS}
    cloud = [
        {**coords, "gpu_sku": "A", "tok_s_per_user": 10,
         "tok_s_per_gpu": 10},
        absent_point({**coords, "gpu_sku": "A"}, "capacity", "too large"),
    ]
    view = frontier_view(cloud, dims=POINT_DIMS, fix={}, group_by=["gpu_sku"])
    assert len(view["series"][0]["frontier"]) == 1
    assert view["values"]["gpu_sku"] == ["A"]


def test_batch_rung_controls_keep_grid_order():
    base = {name: 0 for name in POINT_DIMS}
    keys = ("b1024", "b128", "b64", "max")
    cloud = [{**base, "batch_key": key} for key in keys]
    view = frontier_view(cloud, dims=POINT_DIMS, fix={}, group_by=["gpu_sku"])
    assert view["values"]["batch_key"] == ["b64", "b128", "b1024", "max"]


def test_parallel_mode_is_the_config_shape_not_a_label():
    assert parallel_mode(None) == "tp1"
    assert parallel_mode(ParallelConfig(tp_attention=1)) == "tp1"
    assert parallel_mode(ParallelConfig(
        dp_attention=8, ep_moe=8)) == "dp_ep"
    assert parallel_mode(ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, ep_moe=8)) == "tp_ep"
    assert parallel_mode(ParallelConfig(tp_attention=3)) == "custom"


def test_a_plan_point_maps_to_the_cloud_coordinates():
    from backend.inference.point_cloud import point_coordinates
    from kernel_bench.plan import PlanPoint

    point = PlanPoint(16384, 8192, 8, 64, 1.5)
    coordinates = point_coordinates(
        parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                tp_moe_expert=2), gpu_sku="H100-SXM",
        context=point.seq_len, batch_key="b64",
        draft_tokens=point.s_verify - 1, acceptance=0.8,
        ep_load_factor=point.ep_load_factor)
    assert coordinates == {
        "mode": "tp_tp", "num_gpus": 2, "layout": "TP2", "gpu_sku": "H100-SXM",
        "context": 16384, "batch_key": "b64", "draft_tokens": 7,
        "acceptance": 0.8, "ep_load_factor": 1.5,
        "rho_dcp": 1.0, "dcp_failed_owners": 0,
    }


def _est(tpot: float, max_batch: int = 64) -> dict:
    # `per_request` is the rate one request sees, as the producer publishes
    # it (`phases.phase_throughput`): the point reads it, it divides nothing.
    return {
        "max_batch_per_gpu": max_batch,
        "phases": {"decode": {"throughput": {"sol": {
            "time_per_output_token_s": tpot,
            "per_request": 1 / tpot,
            "per_gpu": 1 / tpot,
        }, "concurrent_requests": max_batch}}},
    }


def test_an_estimate_cloud_states_coordinates_and_keeps_absences():
    knobs = CapacityKnobs(
        parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                tp_moe_expert=2),
        s_verify=3, acceptance_rate=0.7, ep_load_factor=1.5)
    plan = CapacitySweepPlan.of(
        seq_lens=[1024, 2048], config={}, gpu_sku="H100-SXM", knobs=knobs,
        batch_grid=[{"key": "b1", "prefill_batch_size": 1,
                     "decode_batch_size": 1},
                    {"key": "b64", "prefill_batch_size": 64,
                     "decode_batch_size": 64}])
    cloud = plan.throughput_cloud([
        [_est(0.01), _est(0.02, max_batch=2)],
        [_est(0.03), _est(0.04, max_batch=2)],
    ])
    served = cloud[0]
    assert {k: served[k] for k in POINT_DIMS} == {
        "mode": "tp_tp", "num_gpus": 2, "layout": "TP2", "gpu_sku": "H100-SXM",
        "context": 1024, "batch_key": "b1", "draft_tokens": 2,
        "acceptance": 0.7, "ep_load_factor": 1.5,
        "rho_dcp": 1.0, "dcp_failed_owners": 0,
    }
    absent = cloud[1]
    assert absent["refused"] == "capacity"
    assert absent["context"] == 1024 and absent["batch_key"] == "b64"
    assert len(cloud) == 4


def _cloud_point(gpu: str, mode: str, n: int, ctx: int, key: str, batch: int,
                 per_gpu: float) -> dict:
    return {"mode": mode, "num_gpus": n, "gpu_sku": gpu, "context": ctx,
            "batch_key": key, "draft_tokens": 0, "acceptance": 1.0,
            "ep_load_factor": 1.0, "rho_dcp": 1.0, "dcp_failed_owners": 0,
            "batch": batch, "layout": f"{mode}{n}",
            "tok_s_per_user": 1000.0 / batch, "tok_s_per_gpu": per_gpu}


def test_a_line_view_keeps_the_best_point_at_each_x_and_says_which():
    """Several points of one line at one X (a dimension neither held nor
    grouped): the line keeps the one with the most tokens/s per GPU, with
    its own coordinates, so a hover can name the deployment."""
    from backend.inference.frontier import line_view
    cloud = [
        _cloud_point("B300", "dp_ep", 8, 1024, "max", 512, 90.0),
        _cloud_point("B300", "tp_tp", 8, 1024, "max", 256, 120.0),
        _cloud_point("B300", "dp_ep", 1, 1024, "max", 64, 50.0),
        _cloud_point("B300", "dp_ep", 8, 4096, "max", 128, 40.0),
        absent_point({name: 0 for name in POINT_DIMS}
                     | {"gpu_sku": "B300", "context": 1024},
                     "capacity", "too large"),
    ]
    v = line_view(cloud, dims=POINT_DIMS, x="num_gpus",
                  fix={"context": [1024]}, group_by=["gpu_sku"])
    (line,) = v["series"]
    assert [(p["num_gpus"], p["mode"]) for p in line["points"]] == [
        (1, "dp_ep"), (8, "tp_tp")]
    # Its controls are `frontier_view`'s: one vocabulary across forms.
    assert v["values"] == frontier_view(
        cloud, dims=POINT_DIMS, fix={"context": [1024]},
        group_by=["gpu_sku"])["values"]


def test_a_line_along_batch_reads_each_points_own_batch():
    """`batch` is the point's per-GPU decode batch: the `max` rung lands on
    its ceiling, and a ladder rung equal to that ceiling is the same X."""
    from backend.inference.frontier import line_view
    cloud = [_cloud_point("H200", "tp_tp", 2, 1024, key, b, per_gpu)
             for key, b, per_gpu in (("b1", 1, 10.0), ("b64", 64, 300.0),
                                     ("max", 64, 300.0), ("max", 96, 320.0))]
    cloud[3]["context"] = 4096
    (line,) = line_view(cloud, dims=POINT_DIMS, x="batch",
                        fix={"context": [1024]}, group_by=["context"])["series"]
    assert [p["batch"] for p in line["points"]] == [1, 64]
    with pytest.raises(ValueError, match="unknown line axis"):
        line_view(cloud, dims=POINT_DIMS, x="batch_key", fix={}, group_by=[])


def test_the_one_gpu_point_is_every_ladders_first_rung_once():
    """One point per deployment: the one-GPU deployment's coordinate is
    `tp1` (`parallel_mode`). A view holds and groups it into every mode with
    a ladder of its own, draws it once in a front over several modes, and
    grows no one-point line for a mode whose every degree past one refused."""
    from backend.inference.frontier import SINGLE_GPU_MODE, line_view
    one = _cloud_point("B300", SINGLE_GPU_MODE, 1, 1024, "max", 64, 50.0)
    cloud = [
        one,
        _cloud_point("B300", "dp_ep", 8, 1024, "max", 512, 90.0),
        _cloud_point("B300", "tp_tp", 8, 1024, "max", 256, 120.0),
        absent_point({**{name: 0 for name in POINT_DIMS}, "mode": "cp_ep",
                      "num_gpus": 8, "gpu_sku": "B300", "context": 1024},
                     "layout", "CP over a recurrent mixer"),
    ]
    held = line_view(cloud, dims=POINT_DIMS, x="num_gpus",
                     fix={"mode": ["dp_ep"], "num_gpus": [1]}, group_by=[])
    assert held["series"][0]["points"] == [one]
    lines = line_view(cloud, dims=POINT_DIMS, x="num_gpus", fix={},
                      group_by=["mode"])
    by_mode = {s["group"]["mode"]: [p["num_gpus"] for p in s["points"]]
               for s in lines["series"]}
    assert by_mode == {"dp_ep": [1, 8], "tp_tp": [1, 8], "cp_ep": []}
    assert lines["values"]["mode"] == ["cp_ep", "dp_ep", "tp_tp"]
    (front,) = frontier_view(cloud, dims=POINT_DIMS, fix={},
                             group_by=["gpu_sku"])["series"]
    assert sum(p is one for p in front["frontier"]) <= 1
    # A cloud with no ladder keeps the one-GPU deployment as its own mode.
    alone = frontier_view([one], dims=POINT_DIMS, fix={}, group_by=["mode"])
    assert [s["group"]["mode"] for s in alone["series"]] == [SINGLE_GPU_MODE]


def test_both_clouds_keep_a_point_past_the_context_that_fits_and_drop_one_that_does_not():
    """One rule for both producers: KV capacity (`capacity.fits`) decides
    whether a point exists; a context past the model's published length
    keeps its values, flagged."""
    knobs = CapacityKnobs(parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                                  tp_moe_expert=2))
    plan = CapacitySweepPlan.of(
        seq_lens=[1048576], config={}, gpu_sku="H100-SXM", knobs=knobs,
        batch_grid=[{"key": "b1", "prefill_batch_size": 1, "decode_batch_size": 1},
                    {"key": "b64", "prefill_batch_size": 1, "decode_batch_size": 64}])
    past = lambda est: {**est, "beyond_max_context": True}
    cloud = plan.throughput_cloud([[past(_est(0.01, max_batch=2)),
                                past(_est(0.02, max_batch=2))]])
    kept, dropped = cloud
    assert kept["beyond_max_context"] is True and "refused" not in kept
    assert dropped["refused"] == "capacity"


def _two_phase_estimate(*, tpot: float, batch: int, ttft: float,
                        prompt: int, gpus: int = 1) -> dict:
    """An estimate as the producer publishes it: each phase's rates named
    once (`phases.phase_throughput`), including the rate ONE request sees."""
    def rates(tokens: float, t: float, per_request: float) -> dict:
        return {"sol": {"per_request": per_request, "per_gpu": tokens / t / gpus,
                        "time_per_output_token_s": t / (tokens / batch)},
                "concurrent_requests": batch}
    return {"phases": {
        "decode": {"batch": batch, "t_sol_s": tpot,
                   "throughput": rates(batch, tpot, 1 / tpot)},
        "prefill": {"batch": 1, "t_sol_s": ttft,
                    "coverage": {"measured": 0, "total": 3},
                    "throughput": {"sol": {"per_request": prompt / ttft,
                                           "per_gpu": prompt / ttft / gpus}}}}}


def test_one_point_carries_both_phases_of_one_estimate():
    """A cloud point is a decode rung that also states the deployment's
    prefill: both phases read off the ONE estimate, neither recomputed."""
    from backend.inference.frontier import cloud_point

    pt = cloud_point(_two_phase_estimate(tpot=0.01, batch=8, ttft=0.05,
                                         prompt=1000))
    assert pt["tok_s_per_user"] == pytest.approx(100.0)
    assert pt["tok_s_per_gpu"] == pytest.approx(800.0)
    assert pt["prefill_tok_s_per_user"] == pytest.approx(20000.0)
    assert pt["prefill_tok_s_per_gpu"] == pytest.approx(20000.0)
    assert pt["prefill_batch"] == 1 and pt["prefill_t_s"] == 0.05
    assert pt["prefill_coverage"] == {"measured": 0, "total": 3}
    # A phase the estimate does not carry is simply not on the point.
    decode_only = _two_phase_estimate(tpot=0.01, batch=8, ttft=0.05, prompt=1000)
    decode_only["phases"].pop("prefill")
    assert "prefill_tok_s_per_gpu" not in cloud_point(decode_only)


def _phase_point(**over) -> dict:
    """A cloud point as the PRODUCER emits it, which includes its mode
    coordinate: a one-GPU deployment is ONE deployment whatever mode was
    asked for, and `parallel_mode` stamps it `SINGLE_GPU_MODE`.

    A probe that stamps a one-GPU point `tp_tp` is a cloud shape no
    producer can make, and a view's expansion through `mode_ladders` then
    never runs under the test — which is how `view_forms` came to group on
    the raw coordinate unseen."""
    base = {name: 0 for name in POINT_DIMS}
    base.update(gpu_sku="B300", mode="tp_tp", num_gpus=1, context=1024,
                batch_key="b1", batch=1,
                tok_s_per_user=10.0, tok_s_per_gpu=10.0,
                prefill_tok_s_per_user=900.0, prefill_tok_s_per_gpu=900.0)
    base.update(over)
    if int(base["num_gpus"]) == 1:
        base["mode"] = SINGLE_GPU_MODE
    return base


def test_a_ladder_that_starts_at_the_one_gpu_point_moves_with_num_gpus():
    """Every mode's ladder starts at the one-GPU deployment, which the
    cloud carries once under `SINGLE_GPU_MODE` (`mode_ladders`). Asked on
    the raw mode coordinate, that point is a mode of its own: a mode whose
    only priced degrees are one GPU and one more falls into two groups of
    a single X apiece, nothing "moves", and "vs GPUs" is disabled with the
    untrue sentence "decode does not move with num_gpus here" — a gate
    that lies about the cloud it reads.

    `view_forms` asks through the same expansion every view groups by."""
    from backend.inference.frontier import view_forms

    cloud = [_phase_point(num_gpus=1, tok_s_per_gpu=10.0,
                          prefill_tok_s_per_gpu=900.0),
             _phase_point(num_gpus=2, mode="dp_ep", tok_s_per_gpu=8.0,
                          prefill_tok_s_per_gpu=700.0)]
    assert [p["mode"] for p in cloud] == [SINGLE_GPU_MODE, "dp_ep"]
    forms = view_forms(cloud)
    assert forms["decode"]["num_gpus"] is None, forms["decode"]["num_gpus"]
    assert forms["prefill"]["num_gpus"] is None, forms["prefill"]["num_gpus"]


def test_a_cloud_says_which_phase_moves_with_which_axis():
    """Which (phase, form) pairs a cloud can answer is derived FROM the
    cloud, not declared per phase: prefill is stamped on every decode rung
    of a layout, so a prefill line along the decode batch is flat and is
    refused with that reason, while decode's moves and is a view."""
    from backend.inference.frontier import view_forms

    cloud = [
        _phase_point(batch_key="b1", batch=1, tok_s_per_gpu=10.0),
        _phase_point(batch_key="b8", batch=8, tok_s_per_gpu=50.0),
        _phase_point(num_gpus=2, batch_key="b1", batch=1, tok_s_per_gpu=8.0,
                     prefill_tok_s_per_gpu=700.0),
        _phase_point(num_gpus=2, batch_key="b8", batch=8, tok_s_per_gpu=40.0,
                     prefill_tok_s_per_gpu=700.0),
    ]
    forms = view_forms(cloud)
    assert forms["decode"]["batch"] is None
    assert forms["decode"]["num_gpus"] is None
    assert "same at every batch" in forms["prefill"]["batch"]
    assert forms["prefill"]["num_gpus"] is None
    # The front needs no axis, so it is a view of either phase.
    assert forms["decode"]["frontier"] is None and forms["prefill"]["frontier"] is None
    # One context in this cloud: nothing to run a line along, for either.
    assert "one context" in forms["decode"]["context"]
    assert "one context" in forms["prefill"]["context"]
    # A cloud whose points price no prefill is no view of that phase at all.
    bare = [{k: v for k, v in p.items() if not k.startswith("prefill_")}
            for p in cloud]
    assert all(why == "no point of this sweep prices prefill"
               for why in view_forms(bare)["prefill"].values())


def test_a_line_runs_along_the_context_for_either_phase():
    """`context` is a coordinate of every cloud, so a line along it is a
    view of either phase — the same holds, the phase's own Y."""
    from backend.inference.frontier import line_view

    cloud = [_phase_point(context=1024, tok_s_per_gpu=10.0,
                          prefill_tok_s_per_gpu=900.0),
             _phase_point(context=4096, tok_s_per_gpu=6.0,
                          prefill_tok_s_per_gpu=700.0)]
    for phase, field, want in (("decode", "tok_s_per_gpu", [10.0, 6.0]),
                               ("prefill", "prefill_tok_s_per_gpu", [900.0, 700.0])):
        (line,) = line_view(cloud, dims=POINT_DIMS, x="context", fix={},
                            group_by=["gpu_sku"], phase=phase)["series"]
        assert [p["context"] for p in line["points"]] == [1024, 4096]
        assert [p[field] for p in line["points"]] == want


def test_a_deployments_line_is_its_own_batch_curve_and_drawn_once():
    """Grouped by `layout`, each line is one deployment through its batches
    — the one-GPU deployment once, not once per mode's ladder — and the
    card's envelope (`envelope_by`) is the old one-front-per-card answer."""
    cloud = [
        _cloud_point("B300", "tp1", 1, 1024, "b1", 1, 10.0),
        _cloud_point("B300", "tp1", 1, 1024, "b8", 8, 40.0),
        _cloud_point("B300", "dp_ep", 8, 1024, "b1", 1, 12.0),
        _cloud_point("B300", "dp_ep", 8, 1024, "b8", 8, 90.0),
        _cloud_point("B300", "tp_tp", 8, 1024, "b1", 1, 8.0),
        _cloud_point("B300", "tp_tp", 8, 1024, "b8", 8, 60.0),
    ]
    v = frontier_view(cloud, dims=POINT_DIMS, fix={"context": [1024]},
                      group_by=["gpu_sku", "layout"], envelope_by=["gpu_sku"])
    lines = {s["group"]["layout"]: [p["batch"] for p in s["frontier"]]
             for s in v["series"]}
    assert lines == {"tp11": [1, 8], "dp_ep8": [1, 8], "tp_tp8": [1, 8]}
    # A group by deployment states what the deployment fixes.
    assert {(s["group"]["mode"], s["group"]["num_gpus"]) for s in v["series"]} \
        == {("tp1", 1), ("dp_ep", 8), ("tp_tp", 8)}
    alone = frontier_view(cloud, dims=POINT_DIMS, fix={"context": [1024]},
                          group_by=["gpu_sku"])
    assert v["envelopes"] == alone["series"]
    with pytest.raises(ValueError, match="whole lines"):
        frontier_view(cloud, dims=POINT_DIMS, fix={}, group_by=["layout"],
                      envelope_by=["gpu_sku"])


def test_a_layout_name_is_one_deployment():
    """`layout` is a coordinate, so two deployments may not share a name:
    every field of `ParallelConfig` a deployment can differ in shows in it."""
    import itertools
    from backend.inference.frontier import layout_label
    names = {}
    for tp, ffn, moe, ep, cp, dcp, dp, rep, pp, sp, scheme in itertools.product(
            (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2),
            (1, 2), (False, True), ("pcp", "ulysses")):
        if cp == 1 and scheme != "pcp":
            continue   # a scheme with no context parallel is no deployment
        cfg = ParallelConfig(tp_attention=tp, tp_dense_ffn=ffn,
                             tp_moe_expert=moe, ep_moe=ep, cp=cp, dcp=dcp,
                             dp_attention=dp, dp_replicas=rep, pp=pp,
                             sequence_parallel=sp, cp_scheme=scheme)
        name = layout_label(cfg)
        assert name not in names, (name, names.get(name), cfg)
        names[name] = cfg
    assert layout_label(ParallelConfig()) == "TP1"
    assert layout_label(ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                       tp_moe_expert=8)) == "TP8"
    assert layout_label(ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                       dp_attention=4, ep_moe=8)) == "TP2·EP8·DP4"
