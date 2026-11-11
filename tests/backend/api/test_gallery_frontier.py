"""The gallery's decode Pareto frontier: which points a layout draws, which
of them survive the frontier, what a refused layout reports, and how
Kernel Bench rows fold on.

The estimator is stubbed: these pin the frontier's reading of its rows —
the same questions the overview's point tests asked of the old line
charts (a batch that does not fit is not drawn, past the published
context is empty, no room for a request draws nothing).
"""
from __future__ import annotations

import time

import pytest

import backend.api.gallery_frontier as f
from backend.code_trace.provider import recording
from backend.measure.measure_matrix import MeasureMatrix
from backend.measure.auto_sweep import MeasuredFormula, PointMeasure
from backend.measure.lookup import Key
from backend.inference.parallel_config import ParallelConfig
from backend.inference.row_roofline import FloorForm
from backend.inference.parallel_sweep import class_cards
from backend.inference.sweep_grid import (
    FRONTIER_BATCH_GRID,
    FRONTIER_DRAFT_TOKENS,
    GALLERY_SEQ_LENS,
)


@pytest.fixture(autouse=True)
def _one_computation():
    """Each test drives the worker side of a Gallery job directly — the
    fold, the read, a bundle written — so each is one recorded computation,
    as the job it stands in for is (`measure.pool.submit` runs every job in
    one: `code_trace.provider.run_recorded`). What it keeps is stored with
    the code it ran."""
    with recording():
        yield


def _pt(user, gpu, batch=1):
    return {"tok_s_per_user": user, "tok_s_per_gpu": gpu, "batch": batch}


def test_the_frontier_keeps_only_points_nothing_beats_on_both_axes():
    pts = [_pt(100, 100), _pt(50, 400), _pt(40, 300), _pt(100, 90),
           _pt(10, 1000), _pt(60, 50)]
    front = f.pareto_front(pts)
    assert [(p["tok_s_per_user"], p["tok_s_per_gpu"]) for p in front] == [
        (100, 100), (50, 400), (10, 1000)]
    assert f.pareto_front([]) == []


def _one_term(t):
    """A step of one term, `t` seconds at any floors."""
    return FloorForm.termwise([(0, (t, 0.0, 0.0, 0.0))], 1).to_json()


def _timed_item(seq, *, batch=4, plain_s=0.01, drafts=None,
                speculative=False, measure_record=None):
    """A theory point as `_layout_points` stores it: its decode step, the
    plain step's floor form and each K's (`draft`), constants here."""
    dec = {"phase": "decode", "batch": batch, "t_sol_s": plain_s,
           "binding": {"tokens_per_step": 1, "effective_tokens_per_step": batch,
                       "speculative_draft_tokens": 1 if speculative else 0,
                       "acceptance_rate": 1.0,
                       "effective_tokens_formula": {"symbolic": "B × 1",
                                                    "substituted": f"{batch} × 1"}},
           "throughput": {"groups": 1, "num_gpus": 2}}
    return {"seq": seq, "label": "TP2", "num_gpus": 2,
            "sol": {"phases": {"decode": dec}},
            "floors": None if speculative else _one_term(plain_s),
            **({} if measure_record is None else
                {"measureRecord": measure_record}),
            "draft": {str(k): _one_term(t)
                      for k, t in (drafts or {}).items()}}


def _inline_measures(monkeypatch):
    """Read the measures a test placed on its points, as the pool reader
    would read them from the records those points name."""
    monkeypatch.setattr(
        f, "_measureReader",
        lambda _core, _gpu_sku, _view, _timings: lambda _i, point: (
            (point.get("measures") or {}).get("decode")))


def _fold_into(dec, *, t_measured=None, part=0.0, terms=(), width=1):
    """What the measured fold states of a point's step
    (`batch_sweep.apply_measured(blend_terms=True)`): its measured total
    (None unless every term is measured — then the one term of `_one_term`
    is), which step terms are measured and their share."""
    if t_measured is not None:
        terms, part = (0,), t_measured
    dec.update(t_measured_s=t_measured, measured_terms=list(terms),
               measured_part_s=part, total_rows=width)


def _fold_nothing(items, **_kw):
    """A fold that measures nothing: each point's coverage as it stood."""
    for item in items:
        _fold_into(item["sol"]["phases"]["decode"])


def _serving_law_record(*, max_batch=100, draft_max_batch=None,
                        speculative=False, measure=False):
    """A minimal current serving-law record: a real law graph (so projection
    is the path under test) over stub capacity facts, with no measure law."""
    graph = {
        "inputs": ["B", "s_input", "s_verify"],
        "definitions": [],
        "outputs": {"t_sol_s": "s_input*s_verify/1000000000000",
                    "term_0": "s_input*s_verify/1000000000000"},
    }
    return {
        "schema": f.walkRecordSchema,
        "walked_on": "B200",
        "framework": "vllm",
        "parallel": ParallelConfig(tp_attention=2).to_dict(),
        "speculative": speculative,
        "gpu_memory_fraction": 0.9,
        "max_context_tokens": max(GALLERY_SEQ_LENS),
        "parallel_warnings": [],
        "phases": {
            name: {"law": graph, "collectives": [], "terms": [],
                   **({"measurePoints": {}} if measure else {})}
            for name in ("prefill", "decode")
        },
        "capacity": [
            {
                "gpu_sku": "B200",
                "context": seq,
                "max_decode_batch": max_batch,
                "max_prefill_batch": 1,
                "beyond_max_context": seq == GALLERY_SEQ_LENS[-1],
                "infeasibility_reason": None,
            }
            for seq in GALLERY_SEQ_LENS
        ],
        **({} if draft_max_batch is None else {
            "draft_capacity": [
                {
                    "gpu_sku": "B200",
                    "context": seq,
                    "max_decode_batch": draft_max_batch,
                    "max_prefill_batch": 1,
                    "beyond_max_context": seq == GALLERY_SEQ_LENS[-1],
                    "infeasibility_reason": None,
                }
                for seq in GALLERY_SEQ_LENS
            ]}),
    }


def _projecting_record(monkeypatch, calls, *, fail=None, fail_k=None,
                       fail_all=False, max_batch=100):
    """A speculative record projected through the real grid, recording each
    projection's contexts and K; `fail` rejects one K (`fail_all`, all)."""
    record = _serving_law_record(max_batch=max_batch, speculative=True)
    real = f.servingLawPoints

    def projected(record_arg, **kwargs):
        calls.append({"contexts": list(kwargs["contexts"]),
                      "draft_tokens": kwargs["draftTokens"]})
        if fail is not None and kwargs["draftTokens"] and (
                fail_all or kwargs["draftTokens"] == fail_k):
            calls[-1]["raised"] = str(fail)
            raise fail
        return real(record_arg, **kwargs)

    monkeypatch.setattr(f, "servingLawPoints", projected)
    return record


def _theory_at(res, seq):
    return [p for p in res["theory"] if p["seq"] == str(seq)]


def _batch(point):
    return point["sol"]["phases"]["decode"]["batch"]


def test_a_layout_draws_the_batches_that_fit_and_max_once():
    res = f._restamp_layout(
        _serving_law_record(max_batch=100), "B200", ParallelConfig(tp_attention=2))
    pts = _theory_at(res, GALLERY_SEQ_LENS[0])
    assert sorted(_batch(p) for p in pts) == [1, 2, 4, 8, 16, 32, 64, 100]
    assert all(p["label"] == "TP2" and p["num_gpus"] == 2 for p in pts)
    assert _theory_at(res, GALLERY_SEQ_LENS[-1]) == []
    assert len(FRONTIER_BATCH_GRID) == 12


def test_a_batch_that_does_not_fit_is_not_drawn():
    """62 requests fit per GPU: 64 and above are absent, max is 62 once."""
    res = f._restamp_layout(
        _serving_law_record(max_batch=62), "B200", ParallelConfig(tp_attention=2))
    pts = _theory_at(res, GALLERY_SEQ_LENS[0])
    assert sorted(_batch(p) for p in pts) == [1, 2, 4, 8, 16, 32, 62]


def test_a_layout_with_no_room_for_a_request_draws_nothing():
    """Weights / KV leave max_batch 0: every context is empty, the page
    names the model under the chart."""
    res = f._restamp_layout(
        _serving_law_record(max_batch=0), "B200", ParallelConfig())
    assert res["theory"] == []


def test_a_refused_layout_is_reported_not_failed(monkeypatch):
    def boom(req):
        raise ValueError("does not deploy")

    def refuse(resolved, cfg):
        from backend.arch.eval.refusal import SpecRefusal
        raise SpecRefusal(
            "heads do not divide", declaration="num_heads",
            fix="divide heads across the tensor-parallel degree")

    monkeypatch.setattr(f, "resolve_estimate_environment", boom)
    monkeypatch.setattr(f, "_resolve_one", lambda mid: (None, object(), "v"))
    monkeypatch.setattr(f, "layout_verdict", refuse)
    res = f._layout_law((
        "test/Model", ["B200"], 0.9,
        f._GALLERY_PRECISION.canonical(), ParallelConfig().to_dict()))
    assert res == {"refused": "heads do not divide"}

    # Through the projection the frontier reads it by (`_restamp_layout`), as
    # every walk record is — not handed to the fold directly.
    projected = f._restamp_layout(res, "B200", ParallelConfig())
    assert projected == {"refused": "heads do not divide"}

    sk = str(GALLERY_SEQ_LENS[0])
    entry = f._fold(
        "test/Model", [ParallelConfig(), ParallelConfig(tp_attention=2)],
        [projected, {"theory": [{**_timed_item(sk), "label": "TP2",
                                 "num_gpus": 2}], "draft_errors": []}])
    assert entry["error"] is None
    assert entry["layouts"][0]["refused"] == "heads do not divide"
    assert entry["layouts"][1]["points"] == 1
    out = f._read_entry(entry, f.FrontierView("B200"))
    assert [p["batch"] for p in out["frontier"][sk]] == [4]


def test_an_engine_bug_is_not_downgraded_and_does_not_sink_the_warm(monkeypatch):
    """A plain ValueError from the law walk is an error, not a refusal.

    The diagnostic walk may fail with the same engine bug. It is reported
    beside the original failure, but must not replace it — and it must not
    escape the per-layout boundary and cancel the remaining Gallery models.
    """
    def boom(req):
        raise ValueError("engine bug")

    def diagnostic(resolved, cfg):
        raise ValueError("same engine bug")

    monkeypatch.setattr(f, "resolve_estimate_environment", boom)
    monkeypatch.setattr(f, "_resolve_one", lambda mid: (None, object(), "v"))
    monkeypatch.setattr(f, "layout_verdict", diagnostic)
    res = f._layout_law((
        "test/Model", ["B200"], 0.9,
        f._GALLERY_PRECISION.canonical(), ParallelConfig().to_dict()))
    assert res == {"error": "engine bug", "diagnostic": "same engine bug"}


def test_a_speculative_view_reads_the_drafts_step_and_keeps_one_plus_alpha_k():
    """Under K drafted tokens a point's step is its form at that K, and each
    request keeps 1 + α·K tokens of it: tpot is the step over that."""
    sk = str(GALLERY_SEQ_LENS[0])
    item = _timed_item(sk, drafts={2: 0.016})
    view = f.FrontierView("B200", draft_tokens=2, acceptance_rate=0.5)
    dec = f._at_view(item, view)["sol"]["phases"]["decode"]
    assert dec["t_sol_s"] == 0.016
    assert dec["binding"]["effective_tokens_per_step"] == 4 * (1 + 0.5 * 2)
    rates = dec["throughput"]["sol"]
    assert rates["time_per_output_token_s"] == 0.016 / 2
    assert rates["per_gpu"] == 4 * 2 / 0.016 / 2
    # Another acceptance reads the same form; the plain view is untouched.
    hi = f._at_view(item, f.FrontierView("B200", draft_tokens=2,
                                         acceptance_rate=1.0))
    assert hi["sol"]["phases"]["decode"]["throughput"]["sol"][
        "time_per_output_token_s"] == 0.016 / 3
    plain = f._at_view(item, f.FrontierView("B200"))["sol"]["phases"]["decode"]
    assert plain["throughput"]["sol"]["time_per_output_token_s"] == 0.01
    assert item["sol"]["phases"]["decode"]["t_sol_s"] == 0.01


def test_a_point_with_no_step_under_the_view_is_not_drawn():
    """No form at that K: not drawn there. A batch served only under a
    draft (its own step verifies drafts): not drawn under plain decode."""
    sk = str(GALLERY_SEQ_LENS[0])
    both = _timed_item(sk, batch=4, drafts={1: 0.012})
    plain_only = _timed_item(sk, batch=8, plain_s=0.02)
    drafted_only = _timed_item(sk, batch=6, drafts={1: 0.015}, speculative=True)
    entry = {"model_id": "m", "layouts": [], "error": None,
             "draft_unavailable": None,
             "_theory": [both, plain_only, drafted_only]}
    k1 = f._read_entry(entry, f.FrontierView("B200", draft_tokens=1))
    assert sorted(p["batch"] for p in k1["frontier"][sk]) == [4, 6]
    assert f._at_view(plain_only, f.FrontierView("B200", draft_tokens=2)) is None
    assert f._at_view(drafted_only, f.FrontierView("B200")) is None


def test_a_plain_view_omits_a_batch_served_only_by_the_draft(monkeypatch):
    """A draft's KV footprint can make its max batch lower than plain decode.

    That extra batch is a K-step and must remain available to K views, but it
    is not a plain point. Its only spelling of "which step it is" is the decode
    binding's `speculative_draft_tokens`.
    """
    sk = str(GALLERY_SEQ_LENS[0])
    entry = {"model_id": "m", "layouts": [], "error": None,
             "draft_unavailable": None}
    entry["_theory"] = f._restamp_layout(
        _serving_law_record(max_batch=2, draft_max_batch=3,
                            speculative=True),
        "B200", ParallelConfig(tp_attention=2))["theory"]
    by_batch = {point["sol"]["phases"]["decode"]["batch"]: point
                for point in entry["_theory"] if point["seq"] == sk}
    assert 3 in by_batch and by_batch[3]["sol"]["phases"]["decode"][
        "binding"]["speculative_draft_tokens"] == 1
    plain = f._read_entry(entry, f.FrontierView("B200"))
    assert sorted(point["batch"] for point in plain["frontier"][sk]) == [2]
    k1 = f._read_entry(entry, f.FrontierView("B200", draft_tokens=1))
    assert sorted(point["batch"] for point in k1["frontier"][sk]) == [3]
    for point in entry["_theory"]:
        at_k1 = f._at_view(point, f.FrontierView("B200", draft_tokens=1))
        if at_k1 is not None:
            assert at_k1["sol"]["phases"]["decode"]["binding"][
                "speculative_draft_tokens"] == 1
    folded: list = []

    def fold(items, **_kwargs):
        folded.extend(item["sol"]["phases"]["decode"]["batch"]
                      for item in items)
        for item in items:
            item["sol"]["phases"]["decode"]["coverage"] = {
                "measured": 1, "total": 1}
            _fold_into(item["sol"]["phases"]["decode"], t_measured=0.02)

    monkeypatch.setattr(f, "measure_points", fold)
    for view, want in ((f.FrontierView("B200"), {1, 2}),
                       (f.FrontierView("B200", draft_tokens=1), {1, 2, 3})):
        folded.clear()
        rates = f._measured_rates(
            entry, lambda _i, _point: {"any": 1}, view, None)
        assert set(folded) == want
        assert f._rates_cover(entry, rates, view)

    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    monkeypatch.setattr(
        f, "_measureReader",
        lambda _core, _gpu, _view, _timings: lambda _i, point: (
            {"any": 1}
            if point["sol"]["phases"]["decode"]["binding"][
                "speculative_draft_tokens"] == 0 else None))
    plain = f._read_entry(entry, f.FrontierView("B200"))
    assert plain["measured_track"]["state"] == "reads"

    monkeypatch.setattr(
        f, "_measureReader",
        lambda _core, _gpu, _view, _timings: lambda _i, point: (
            {"any": 1}
            if point["sol"]["phases"]["decode"]["binding"][
                "speculative_draft_tokens"] == 1 else None))
    drafted = f._read_entry(entry, f.FrontierView("B200", draft_tokens=1))
    assert drafted["measured_track"]["state"] == "reads"


def test_a_model_with_no_mtp_draft_reads_an_empty_speculative_frontier():
    sk = str(GALLERY_SEQ_LENS[0])
    entry = {"model_id": "m", "layouts": [], "error": None,
             "draft_unavailable": "this model declares no MTP stack",
             "_theory": [_timed_item(sk)]}
    out = f._read_entry(entry, f.FrontierView("B200", draft_tokens=3))
    assert all(pts == [] for pts in out["frontier"].values())
    assert out["draft_unavailable"] == "this model declares no MTP stack"
    assert len(f._read_entry(entry, f.FrontierView("B200"))["frontier"][sk]) == 1


def test_a_speculative_view_is_its_own_request_but_not_its_own_entry():
    plain, k2 = f.FrontierView("B200"), f.FrontierView("B200", draft_tokens=2)
    assert plain.walk_parts() == k2.walk_parts()
    assert plain.key_parts() != k2.key_parts()
    assert (f.FrontierView("B200", draft_tokens=2, acceptance_rate=0.6).key_parts()
            != k2.key_parts())
    # Acceptance is not a key under plain decode.
    assert f.FrontierView("B200", acceptance_rate=0.6).key_parts() == plain.key_parts()
    # One entry serves every floor, K and acceptance: each point carries its
    # step as a function of the floors at every K, substituted when drawn.
    stored = f._Stored("m", "v1")
    floored = f.FrontierView("B200", comm_latency_us=5.0)
    assert (stored.entry_key(plain) == stored.entry_key(k2)
            == stored.entry_key(floored) == stored.entry_key(f.FrontierView(
                "B200", launch_floor_us=9.0, draft_tokens=3,
                acceptance_rate=0.6)))
    assert floored.key_parts() != plain.key_parts()
    fraction = f.FrontierView("B200", gpu_memory_fraction=.8)
    assert not fraction.keeps_walk()
    assert stored.entry_key(fraction) != stored.entry_key(plain)


def test_a_serving_entry_draws_each_k_and_floor_off_its_step_forms():
    """A serving entry's point keeps every K's step as a function of the
    floors: a view substitutes its own floors (here a launch-bound term
    `max(a, n·L)`) and keeps 1 + α·K tokens of it. A K with no form has no
    step; a plain point without its form is not an entry of this schema."""
    sk = str(GALLERY_SEQ_LENS[0])
    item = _timed_item(sk)
    item["draft"] = {"2": FloorForm(0.01, ((0.004, 0.0, 0.0, 1000.0),)).to_json()}
    view = f.FrontierView("B200", draft_tokens=2, acceptance_rate=0.5,
                          launch_floor_us=1.0)
    dec = f._at_view(item, view)["sol"]["phases"]["decode"]
    assert dec["t_sol_s"] == pytest.approx(0.01 + 0.004)
    assert dec["throughput"]["sol"]["time_per_output_token_s"] == pytest.approx(
        0.014 / 2)
    slow = view._replace(launch_floor_us=6.0)
    assert f._at_view(item, slow)["sol"]["phases"]["decode"][
        "t_sol_s"] == pytest.approx(0.01 + 0.006)
    assert f._at_view(item, f.FrontierView("B200", draft_tokens=1)) is None
    item.pop("floors")
    with pytest.raises(ValueError, match="no step form"):
        f._at_view(item, f.FrontierView("B200"))


def test_the_route_refuses_a_draft_length_off_the_pickers(monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api  # noqa: F401  # registers routes
    from backend.api._app import app
    from backend.api.routes import gallery as route

    monkeypatch.setattr(route, "build_frontier", lambda *a: (_ for _ in ()).throw(
        AssertionError("swept")))
    client = TestClient(app)
    for params in ({"draft_tokens": 5}, {"draft_tokens": 2, "acceptance_rate": 0},
                   {"draft_tokens": 2, "acceptance_rate": 1.5}):
        resp = client.get("/gallery/frontier", params={"gpu_sku": "B200", **params})
        assert resp.status_code in (400, 422), params


def test_the_route_refuses_a_gpu_outside_the_gallery_series(monkeypatch):
    """The picker offers `sweep_grid.GALLERY_GPUS`; another SKU is not swept
    cold — a card the sheet lists but the Gallery does not publish included.
    RTX-4090 is that second case and not a hypothetical one: it is a card of
    the SWEEP series, and the Gallery's narrower series is what this route
    admits."""
    from fastapi.testclient import TestClient

    import backend.api  # noqa: F401  # registers routes
    from backend.api._app import app
    from backend.api.routes import gallery as route

    monkeypatch.setattr(route, "build_frontier", lambda *a: (_ for _ in ()).throw(
        AssertionError("swept")))
    client = TestClient(app)
    assert client.get("/gallery/frontier",
                      params={"gpu_sku": "A100-40G-PCIe"}).status_code == 400
    refused = client.get("/gallery/frontier", params={"gpu_sku": "RTX-4090"})
    assert refused.status_code == 400
    assert "RTX-4090" not in refused.json()["detail"].split(";")[0]


def test_measured_rows_fold_onto_the_frontier(monkeypatch):
    """Kernel Bench rows ride the same points as SOL: a complete measured
    step publishes tok/s beside the estimate, and the measured Pareto is
    folded from those. The theory half is not served."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    _inline_measures(monkeypatch)

    def apply(items, **_kw):
        for it in items:
            dec = it["sol"]["phases"]["decode"]
            _fold_into(dec, t_measured=0.02)
            dec["coverage"] = {"measured": 3, "total": 3}

    monkeypatch.setattr(f, "measure_points", apply)
    _inline_measures(monkeypatch)
    sk = str(GALLERY_SEQ_LENS[0])
    out = f._read_entry({
        "model_id": "m", "frontier": {sk: [_pt(100, 100)]},
        "layouts": [], "error": None,
        "_theory": [{**_timed_item(sk, batch=1), "measures": {"decode": {}}}],
    }, f.FrontierView("B200"))
    assert "_theory" not in out
    pt = out["frontier"][sk][0]
    assert pt["tok_s_per_user"] == pytest.approx(100.0)
    # One token per step over the measured 20 ms, on 2 GPUs.
    assert pt["tok_s_per_user_measured"] == pytest.approx(50.0)
    assert pt["tok_s_per_gpu_measured"] == pytest.approx(25.0)
    meas = out["frontier_measured"][sk][0]
    assert meas["tok_s_per_user_measured"] == pytest.approx(50.0)


def test_the_partial_track_swaps_measured_terms_at_the_view_s_floors(
        monkeypatch):
    """The partial frontier: the point's step form with the fold's measured
    terms' SOL swapped for their measured share (`auto_sweep.blend`,
    `FloorForm.terms_at`), at the view's floors — another floor moves the
    SOL terms left, never the measured share."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))

    def apply(items, **_kw):
        for it in items:
            dec = it["sol"]["phases"]["decode"]
            _fold_into(dec, part=0.004, terms=(1,), width=2)
            dec["coverage"] = {"measured": 1, "total": 2,
                               "states": {"missing": ["k"]}}

    monkeypatch.setattr(f, "measure_points", apply)
    _inline_measures(monkeypatch)
    sk = str(GALLERY_SEQ_LENS[0])
    item = _timed_item(sk, batch=1)
    # Term 0 launch-bound (2 launches), term 1 compute-bound (6 ms): the
    # fold measured term 1 at 4 ms, so the floors move only term 0.
    item["floors"] = FloorForm.termwise(
        [(0, (0.001, 0.0, 0.0, 2.0)), (1, (0.006, 0.0, 0.0, 0.0))], 2).to_json()
    entry = {"model_id": "m", "frontier": {}, "layouts": [], "error": None,
             "_theory": [{**item, "measures": {"decode": {}}}]}
    for launch_us, sol_s in ((1.0, 0.007), (3000.0, 0.012)):
        out = f._read_entry(entry, f.FrontierView(
            "B200", launch_floor_us=launch_us))
        pt = out["frontier_partial"][sk][0]
        partial_s = 0.004 + (sol_s - 0.006)
        assert out["frontier"][sk][0]["tpot_s"] == pytest.approx(sol_s)
        assert pt["tpot_partial_s"] == pytest.approx(partial_s)
        assert pt["partial_measured_share"] == pytest.approx(0.004 / partial_s)
        assert out["frontier_measured"][sk] == []


def test_the_term_mask_keeps_every_step_term():
    """The rates keep a point's measured step terms as a bit mask; reading
    it back names exactly those terms, high and sparse ones included."""
    for terms in ([], [0], [1], [0, 2, 5], list(range(3, 130, 7))):
        assert f._terms(f._termMask(terms)) == terms


def test_a_form_not_over_the_formula_s_terms_is_an_engine_error(monkeypatch):
    """A point whose step form has a term past the measured formula's is
    another step: the fold refuses it (`MeasurementFoldMismatch`), so the
    model draws its SOL frontier under `engine_error` rather than a partial
    line priced off misaligned terms."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))

    def apply(items, **_kw):
        for it in items:
            dec = it["sol"]["phases"]["decode"]
            _fold_into(dec, part=0.004, terms=(0,), width=1)
            dec["coverage"] = {"measured": 1, "total": 1}

    monkeypatch.setattr(f, "measure_points", apply)
    _inline_measures(monkeypatch)
    sk = str(GALLERY_SEQ_LENS[0])
    item = _timed_item(sk, batch=1)
    item["floors"] = FloorForm.termwise(
        [(0, (0.001, 0.0, 0.0, 0.0)), (1, (0.006, 0.0, 0.0, 0.0))], 2).to_json()
    stored = f._Stored("m", None)
    out = stored.drawn(f.FrontierView("B200"), {
        "model_id": "m", "frontier": {}, "layouts": [], "error": None,
        "_theory": [{**item, "measures": {"decode": {}}}]})
    assert out["measured_track"]["state"] == "engine_error"
    assert "measured formula" in out["measured_track"]["error"]
    assert out["frontier"][sk]
    assert out["frontier_partial"][sk] == []


def test_the_measured_fold_does_not_walk(monkeypatch):
    """Joining Kernel Bench rows is a lookup over the cached theory, not
    another estimate walk or another law production."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    monkeypatch.setattr(f, "_layout_law", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("walked")))
    monkeypatch.setattr(f, "measure_points", _fold_nothing)
    _inline_measures(monkeypatch)
    sk = str(GALLERY_SEQ_LENS[0])
    f._read_entry({
        "model_id": "m", "frontier": {sk: []}, "layouts": [], "error": None,
        "_theory": [{**_timed_item(sk, batch=1), "measures": {}}],
    }, f.FrontierView("B200"))


def test_incomplete_measured_coverage_does_not_publish_tok_s(monkeypatch):
    """A mixed step has no measured tok/s: the page never draws a partial
    total. Same all-or-nothing the overview asked of `_safe_measured`."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    monkeypatch.setattr(f, "measure_points", _fold_nothing)
    _inline_measures(monkeypatch)
    sk = str(GALLERY_SEQ_LENS[0])
    item = _timed_item(sk, batch=1)
    item["sol"]["phases"]["decode"]["coverage"] = {
        "measured": 2, "total": 3, "states": {"missing": ["k"]}}
    out = f._read_entry({
        "model_id": "m", "frontier": {sk: [_pt(100, 100)]},
        "layouts": [], "error": None,
        "_theory": [{**item, "measures": {"decode": {}}}],
    }, f.FrontierView("B200"))
    pt = out["frontier"][sk][0]
    assert "tok_s_per_user_measured" not in pt
    assert out["frontier_measured"][sk] == []


def test_every_k_is_one_projection_of_every_context(monkeypatch):
    """Each K of `FRONTIER_DRAFT_TOKENS` is a projection of the same serving
    law at its own verify width — one law, every K read off it."""
    calls: list = []
    record = _projecting_record(monkeypatch, calls)
    res = f._restamp_layout(record, "B200", ParallelConfig(tp_attention=2))
    assert [c["draft_tokens"] for c in calls] == [0, *FRONTIER_DRAFT_TOKENS]
    assert all(c["contexts"] == list(GALLERY_SEQ_LENS) for c in calls)
    assert res["draft_errors"] == []
    point = _theory_at(res, GALLERY_SEQ_LENS[0])[0]
    assert sorted(point["draft"]) == sorted(
        str(k) for k in FRONTIER_DRAFT_TOKENS)


def test_a_layout_no_card_serves_projects_no_draft(monkeypatch):
    """No room for a request anywhere: no plain point, so no speculative
    projection either."""
    calls: list = []
    record = _projecting_record(monkeypatch, calls, max_batch=0)
    res = f._restamp_layout(record, "B200", ParallelConfig(tp_attention=2))
    assert res["theory"] == [] and res["draft_errors"] == []
    assert [c["draft_tokens"] for c in calls] == [0]


def test_a_cached_point_keeps_what_the_frontier_reads(monkeypatch):
    """The frontier draws decode: a projected point keeps its decode phase
    and the record that owns its measure, not the prefill half, the rest of
    the walk, or a second key table."""
    view = f.FrontierView("B200")
    walk_key = f._Stored("test/Model", "v1").serving_key(view)
    res = f._restamp_layout(
        _serving_law_record(measure=True), "B200",
        ParallelConfig(tp_attention=2), walk_key=walk_key)
    point = _theory_at(res, GALLERY_SEQ_LENS[0])[0]
    assert point["sol"] == {"phases": {"decode": point["sol"]["phases"]["decode"]}}
    assert point["measureRecord"] == f.StepLawKey.of_layout(
        walk_key, ParallelConfig(tp_attention=2)).record_name
    assert "measures" not in point


def test_one_fold_reads_each_measure_record_once(monkeypatch):
    """A Gallery fold expands points only for the request, and reads every
    named canonical record once — not once per point and never into the
    Gallery cache."""
    sk = str(GALLERY_SEQ_LENS[0])
    core = {
        "model_id": "m", "layouts": [], "error": None,
        "_measurePool": "bundle-key",
        "_theory": [_timed_item(sk, batch=1, measure_record="record"),
                    _timed_item(sk, batch=2, measure_record="record")],
    }
    reads: list = []
    monkeypatch.setattr(
        f, "sharedSolPool",
        type("Pool", (), {"record": staticmethod(
            lambda *_args: reads.append(1) or {"record": True})}))
    measures = [
        PointMeasure(
            (Key("op", '{"B":1}', '{}', "B200", "decode"),), (float(batch),),
            (None,), ("op",), ("",), MeasuredFormula("T_0*n_0", (False,)),
            {"framework": "vllm", "gpu": "B200"}, ({"any": 1},))
        for batch in (1, 2)
    ]
    record_matrix = MeasureMatrix.of_measures(measures)
    record_matrix.point_rows.update({
        (int(GALLERY_SEQ_LENS[0]), 1, 0): 0,
        (int(GALLERY_SEQ_LENS[0]), 2, 0): 1,
    })
    monkeypatch.setattr(
        f, "servingMeasureMatrix",
        lambda _record, *, card, points: record_matrix)

    def apply(items, **_kwargs):
        for item in items:
            item["sol"]["phases"]["decode"]["coverage"] = {
                "measured": 1, "total": 1}
            _fold_into(item["sol"]["phases"]["decode"], t_measured=0.05)

    monkeypatch.setattr(f, "measure_points", apply)
    view = f.FrontierView("B200")
    rates = f._measured_rates(
        core, f._measureReader(core, "B200", view, {}), view, None)
    assert reads == [1]
    assert rates["timings"]["measure_records"] == 1
    assert rates["timings"]["measure_points"] == 2


def test_a_failed_speculative_projection_is_every_ks_published_error(monkeypatch):
    """No K is dropped silently: a speculative projection that fails is an
    error of the layout at every K, published with that K's view and not
    with another's."""
    calls: list = []
    record = _projecting_record(
        monkeypatch, calls, fail=ValueError("verify refused"), fail_all=True)
    cfg = ParallelConfig(tp_attention=2)
    res = f._restamp_layout(record, "B200", cfg)
    assert res["theory"]
    assert [(e["layout"], e["draft_tokens"], e["error"]) for e in res["draft_errors"]] == [
        ("TP2", k, "verify refused") for k in FRONTIER_DRAFT_TOKENS]
    entry = f._fold("test/Model", [cfg], [res])
    entry["draft_unavailable"] = None
    k = FRONTIER_DRAFT_TOKENS[1]
    out = f._read_entry(entry, f.FrontierView("B200", draft_tokens=k))
    assert out["draft_errors"] == [
        {"layout": "TP2", "draft_tokens": k, "error": "verify refused"}]
    assert all(pts == [] for pts in out["frontier"].values())
    assert f._read_entry(entry, f.FrontierView("B200"))["draft_errors"] == []
    assert "_draft_errors" not in out


def test_one_k_that_fails_is_that_ks_error(monkeypatch):
    """A K whose law does not price is that K's error; the other K's draw."""
    bad = FRONTIER_DRAFT_TOKENS[-1]
    calls: list = []
    record = _projecting_record(
        monkeypatch, calls, fail=ValueError("verify width refused"), fail_k=bad)
    res = f._restamp_layout(record, "B200", ParallelConfig(tp_attention=2))
    assert [c["draft_tokens"] for c in calls] == [0, *FRONTIER_DRAFT_TOKENS]
    assert [(e["draft_tokens"], e["error"]) for e in res["draft_errors"]] == [
        (bad, "verify width refused")]
    point = _theory_at(res, GALLERY_SEQ_LENS[0])[0]
    assert sorted(point["draft"]) == sorted(
        str(k) for k in FRONTIER_DRAFT_TOKENS if k != bad)


def _kept(monkeypatch, tmp_path, *, state="reads"):
    """Hermetic frontier caches over `tmp_path`, a model with a version, the
    measured track in `state`, and a record of which part of an entry each
    read asked for (`reads`) and how many measured folds ran (`folds`)."""
    from backend.measure.cache import (DiskBackend, KeyedCache, MemoryBackend,
                                       TieredBackend)
    from backend.inference.sol_pool import SolPool
    from backend.measure.scope import MeasurementScope
    entries = KeyedCache("f", DiskBackend(tmp_path / "e"))
    views = KeyedCache("v", TieredBackend(MemoryBackend(max_bytes=1 << 20),
                                          DiskBackend(tmp_path / "v")))
    reads: list = []
    real_get = entries.get

    def get(key, **kw):
        reads.append(key)
        return real_get(key, **kw)
    monkeypatch.setattr(entries, "get", get)
    monkeypatch.setattr(f, "_FRONTIER", entries)
    monkeypatch.setattr(f, "_FRONTIER_VIEWS", views)
    monkeypatch.setattr(f, "sharedSolPool", SolPool(tmp_path / "sol-pool"))
    monkeypatch.setattr(f, "_resolve_one", lambda mid: (None, object(), "v1"))
    monkeypatch.setattr(
        f, "_measureReader",
        lambda _core, _gpu_sku, _view, _timings: lambda _i, point: (
            {"any": 1} if point.get("measureRecord") else None))
    scope = MeasurementScope(None, "lake", 1, 1, "r")
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        (scope, "reads") if state == "reads" else (f.NO_MEASUREMENTS, state)))
    folds: list = []

    def apply(items, **_kw):
        folds.append(len(items))
        for it in items:
            _fold_into(it["sol"]["phases"]["decode"], t_measured=0.05)
            it["sol"]["phases"]["decode"]["coverage"] = {"measured": 1, "total": 1}
    monkeypatch.setattr(f, "measure_points", apply)
    return entries, reads, folds


def _built_entry(monkeypatch, cards=("B200",), calls=None):
    """A one-layout model's serving-law result, with the projection stubbed
    to constant points; these tests own Gallery's cache and fold behavior."""
    monkeypatch.setattr(f, "draft_refusal", lambda mid: None)

    def restamp(_walked, _gpu_sku, cfg, _walk_key=None, _view=None):
        return {
            "theory": [
                _timed_item(str(seq), measure_record="test-record",
                           drafts={k: 0.012 for k in FRONTIER_DRAFT_TOKENS})
                for seq in GALLERY_SEQ_LENS
            ],
            "draft_errors": [],
        }

    monkeypatch.setattr(f, "_restamp_layout", restamp)
    cfg = ParallelConfig(tp_attention=2).to_dict()
    plan = {"cards": list(cards), "layoutsByCard": {c: [cfg] for c in cards},
            "layouts": [cfg]}
    return plan, [{"parallel": cfg, "walk": {"schema": f.walkRecordSchema}}]


def test_a_kept_view_is_read_without_its_entry(monkeypatch, tmp_path):
    """A view drawn once is kept: the next request for it reads the
    drawing, not the entry — whatever the entry's size."""
    _entries, reads, folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    view = f.FrontierView("B200")
    built = f._finish_frontier("test/Model", view, plan, results)
    assert built["frontier_measured"][str(GALLERY_SEQ_LENS[0])]
    reads.clear()
    got = f._frontier_read("test/Model", view)
    assert got == {"drawn": built}
    assert reads == []
    assert folds == [len(f._restamp_layout(results[0], "B200",
                                         ParallelConfig(tp_attention=2))["theory"])]


def test_floors_reuse_the_fold_and_each_k_folds_its_own_formula(
        monkeypatch, tmp_path):
    """Another floor or K draws off the entry's first part. Floors and
    acceptance reuse a K's measured rates (step times, which neither
    moves); each K has its own fold under the same Kernel Bench snapshot."""
    _entries, reads, folds = _kept(monkeypatch, tmp_path)
    real_reader = f._measureReader
    record_reads: list = []

    def reader(*args, **kwargs):
        record_reads.append(1)
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(f, "_measureReader", reader)
    plan, results = _built_entry(monkeypatch)
    model = "test/Model"
    f._finish_frontier(model, f.FrontierView("B200"), plan, results)
    record_reads.clear()
    stored = f._Stored(model, "v1")
    cases = ((f.FrontierView("B200", comm_latency_us=5.0), 0),
             (f.FrontierView("B200", draft_tokens=1), 1),
             (f.FrontierView("B200", draft_tokens=3, acceptance_rate=0.6), 1),
             (f.FrontierView("B200", draft_tokens=3, acceptance_rate=0.9), 0))
    for view, reader_reads in cases:
        reads.clear()
        got = f._frontier_read(model, view)
        assert "drawn" in got
        assert reads == [stored.entry_key(view)]
        assert len(record_reads) == reader_reads
        record_reads.clear()
    assert len(folds) == 3
    # A new snapshot folds again, off the records the entry names.
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        MeasurementScope(None, "lake", 2, 2, "r"), "reads"))
    reads.clear()
    assert "drawn" in f._frontier_read(model, f.FrontierView("B200"))
    assert reads == [stored.entry_key(f.FrontierView("B200"))]
    assert record_reads == [1] and len(folds) == 4


def test_an_entry_missing_its_measure_record_draws_sol(monkeypatch, tmp_path):
    """A record the artifact did not carry is an engine/deployment error, not
    a second Gallery key table or a lake outage: the SOL entry still draws
    and no walk is started to paper over the cache."""
    entries, _reads, _folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model = "test/Model"
    stored = f._Stored(model, "v1")
    core = f._restamp(
        model, {**plan, "layouts": [{"parallel": p, "walk": r} for p, r in
                                     zip(plan["layouts"], results, strict=True)]},
        "B200")
    for point in core["_theory"]:
        point.pop("measureRecord", None)
    entries.put(stored.entry_key(f.FrontierView("B200")), core)
    monkeypatch.setattr(f, "_layout_law", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("walked")))
    got = f._frontier_read(model, f.FrontierView("B200"))
    assert got["drawn"]["measured_track"]["state"] == "engine_error"
    assert "has no K=0 measure" in got["drawn"]["measured_track"]["error"]
    assert got["drawn"]["frontier"][str(GALLERY_SEQ_LENS[0])]
    # A view that reads no measurements draws off the same entry.
    assert "drawn" in f._frontier_read(model, f.FrontierView("B200", draft_tokens=1))


def test_another_gpu_of_the_class_is_priced_off_the_kept_walk(monkeypatch, tmp_path):
    """Switching GPU within a walk class walks nothing: the first GPU's
    request walked the model for every card of the class and kept it; the
    other card's entry is priced off that walk — its own capacity, its own
    price — kept, and drawn."""
    entries, _reads, _folds = _kept(monkeypatch, tmp_path)
    calls: list = []
    plan, results = _built_entry(monkeypatch, cards=("GB200", "B200"), calls=calls)
    f._finish_frontier("test/Model", f.FrontierView("B200"), plan, results)
    walked = len(calls)
    got = f._frontier_read("test/Model", f.FrontierView("GB200"))
    assert "drawn" in got and got["drawn"]["frontier"][str(GALLERY_SEQ_LENS[0])]
    assert len(calls) == walked
    assert entries.get(f._Stored("test/Model", "v1").entry_key(
        f.FrontierView("GB200"))) is not None
    # A view at another HBM fraction does not read the class walk: it walks
    # its own card alone.
    monkeypatch.setattr(f, "frontier_layouts", lambda caps, gpu: [ParallelConfig()])
    monkeypatch.setattr(f, "class_frontier_layouts", lambda caps, cards: [ParallelConfig()])
    monkeypatch.setattr(f, "sweep_caps", lambda *a, **k: {})
    got = f._frontier_read("test/Model", f.FrontierView("GB200", 0.8))
    assert got["walk"]["cards"] == ["GB200"]


def test_the_lakes_answer_reaches_the_pricing_and_never_the_walk(
        monkeypatch, tmp_path):
    """A class walk is theory: its key carries no measurement scope, and a
    card priced off a kept walk is priced under whatever the lake answered
    this request — down or up — without walking again. So a lake outage
    never re-walks a class, and never keeps a walk under an outage's key."""
    _entries, _reads, folds = _kept(monkeypatch, tmp_path)
    calls: list = []
    plan, results = _built_entry(monkeypatch, cards=("GB200", "B200"),
                                 calls=calls)
    f._finish_frontier("test/Model", f.FrontierView("B200"), plan, results)
    walked = len(calls)
    stored = f._Stored("test/Model", "v1")
    # One walk key, whatever the track: the walk tier is keyed under `none`.
    keys = set()
    for state in ("reads", "unavailable", "none"):
        _track(monkeypatch, state)
        keys.add(stored.walk_identity(f.FrontierView("GB200")).key)
    assert len(keys) == 1
    # The lake down: the other card is priced off the walk all the same,
    # its drawing SOL-only and labelled for the outage.
    _track(monkeypatch, "unavailable")
    down = f._frontier_read("test/Model", f.FrontierView("GB200"))["drawn"]
    assert down["measured_track"] == {"state": "unavailable"}
    assert down["frontier"][str(GALLERY_SEQ_LENS[0])]
    assert len(calls) == walked
    # The lake back: the same entry folds its rates, still without walking.
    _track(monkeypatch, "reads")
    up = f._frontier_read("test/Model", f.FrontierView("GB200"))["drawn"]
    assert up["measured_track"]["state"] == "reads"
    assert up["frontier_measured"][str(GALLERY_SEQ_LENS[0])]
    assert len(calls) == walked and folds


def test_the_kept_entry_names_records_and_carries_no_point_measures(
        monkeypatch, tmp_path):
    entries, _reads, folds = _kept(monkeypatch, tmp_path, state="none")
    plan, results = _built_entry(monkeypatch)
    f._finish_frontier("test/Model", f.FrontierView("B200"), plan, results)
    stored = f._Stored("test/Model", "v1")
    core = entries.get(stored.entry_key(f.FrontierView("B200")))
    assert core["_measurePool"]
    assert all(t.get("measureRecord") == "test-record" for t in core["_theory"])
    assert all("measures" not in t for t in core["_theory"])
    assert folds == []


# --- Kernel Bench coverage of the frontier ---------------------------------

def _covered_item(seq, *, batch, tpot, coverage):
    """A theory point and the coverage the measured fold states of it; a
    fully measured point gets a measured rate."""
    rates = {"concurrent_requests": batch,
             "sol": {"time_per_output_token_s": tpot, "per_gpu": batch / tpot}}
    return {"seq": seq, "label": f"b{batch}", "num_gpus": 1,
            "sol": {"phases": {"decode": {"batch": batch, "throughput": rates}}},
            "floors": _one_term(tpot),
            "measures": {"decode": {"coverage": coverage}}}


def _read_folded(monkeypatch, items):
    """`_read_entry` with the rows read: the stubbed fold states each
    point's coverage as its measure names it."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    _inline_measures(monkeypatch)

    def apply(pts, **_kw):
        for it in pts:
            cov = it["measures"]["decode"]["coverage"]
            dec = it["sol"]["phases"]["decode"]
            dec["coverage"] = cov
            tpot = dec["throughput"]["sol"]["time_per_output_token_s"]
            _fold_into(dec, t_measured=(tpot * 2 if cov["measured"] == cov["total"]
                                        else None))

    def at_view(item, _view, fold=None):
        """The point as stored, its measured rate off the fold's time."""
        t = (fold or {}).get("t_measured_s")
        if t is None:
            return item
        dec = item["sol"]["phases"]["decode"]
        batch = dec["batch"]
        return {**item, "sol": {"phases": {"decode": {**dec, "throughput": {
            **dec["throughput"], "measured": {
                "time_per_output_token_s": t, "per_gpu": batch / t}}}}}}
    monkeypatch.setattr(f, "measure_points", apply)
    monkeypatch.setattr(f, "_at_view", at_view)
    return f._read_entry({"model_id": "m", "layouts": [], "error": None,
                          "_theory": items}, f.FrontierView("B200"))


def test_coverage_of_a_fully_measured_frontier(monkeypatch):
    sk = str(GALLERY_SEQ_LENS[0])
    full = {"measured": 4, "total": 4}
    out = _read_folded(monkeypatch, [
        _covered_item(sk, batch=1, tpot=0.01, coverage=full),
        _covered_item(sk, batch=8, tpot=0.02, coverage=full)])
    assert len(out["frontier_measured"][sk]) == 2
    assert out["measured_track"]["coverage"][sk] == {
        "points": 2, "complete": 2, "best": {"measured": 4, "total": 4},
        "median": 1.0}


def test_coverage_counts_only_the_points_on_the_frontier(monkeypatch):
    """A point the frontier drops (beaten on both axes) is not drawn: its
    unmeasured keys are not listed. Each key counts the frontier points
    that need it, most needed first."""
    sk = str(GALLERY_SEQ_LENS[0])
    out = _read_folded(monkeypatch, [
        _covered_item(sk, batch=1, tpot=0.01, coverage={
            "measured": 3, "total": 4, "states": {"missing": ["gemm"]}}),
        _covered_item(sk, batch=8, tpot=0.02, coverage={
            "measured": 1, "total": 4, "states": {
                "missing": ["attn", "gemm"], "unplaced": ["rope"]}}),
        # Slower and fewer tokens per GPU than batch 8: off the frontier.
        _covered_item(sk, batch=2, tpot=0.05, coverage={
            "measured": 0, "total": 4, "states": {"ambiguous": ["norm"]}}),
    ])
    cov = out["measured_track"]["coverage"][sk]
    assert cov["points"] == 2 and cov["complete"] == 0
    assert cov["best"] == {"measured": 3, "total": 4}
    assert cov["median"] == (0.75 + 0.25) / 2
    assert cov["states"] == {
        "missing": {"keys": 2, "top": [{"key": "gemm", "points": 2},
                                       {"key": "attn", "points": 1}]},
        "unplaced": {"keys": 1, "top": [{"key": "rope", "points": 1}]}}


def test_coverage_lists_the_most_needed_keys_and_counts_them_all():
    many = [f"k{i:02d}" for i in range(f.COVERAGE_KEYS_LISTED + 5)]
    held = f._PointCoverage()
    held.add(0, {"measured": 0, "total": 20, "states": {"missing": many}})
    held.add(1, {"measured": 1, "total": 20, "states": {"missing": many[-1:]}})
    kept = held.to_json()
    # Each key is named once, however many points hold it.
    assert len(kept["keys"]) == len(many)
    cov = f.frontier_coverage(list(kept["points"].values()), kept["keys"])
    missing = cov["states"]["missing"]
    assert missing["keys"] == len(many)
    assert len(missing["top"]) == f.COVERAGE_KEYS_LISTED
    assert missing["top"][0] == {"key": many[-1], "points": 2}


def test_coverage_of_a_frontier_nothing_measures(monkeypatch):
    sk = str(GALLERY_SEQ_LENS[0])
    out = _read_folded(monkeypatch, [_covered_item(sk, batch=1, tpot=0.01, coverage={
        "measured": 0, "total": 2, "states": {
            "missing": ["gemm"],
            "declined": [{"key": "fp4 gemm", "reason": "no dtype token"}]}})])
    cov = out["measured_track"]["coverage"][sk]
    assert cov["complete"] == 0 and cov["median"] == 0.0
    assert cov["best"] == {"measured": 0, "total": 2}
    assert cov["states"]["declined"] == {"keys": 1, "top": [
        {"key": "fp4 gemm", "reason": "no dtype token", "points": 1}]}
    # A context no layout serves has no points, and says so.
    assert out["measured_track"]["coverage"][str(GALLERY_SEQ_LENS[1])] == {
        "points": 0, "complete": 0, "best": None, "median": None}


def test_a_frontier_point_the_fold_did_not_reach_is_refused(monkeypatch):
    """A partial fold drops the measured overlay instead of the whole page.

    A point no measure reached cannot be reported as zero coverage, and one
    stale/partial overlay must not 500 the entire Gallery response. The SOL
    drawing remains valid and is labelled as an engine error for this read.
    """
    sk = str(GALLERY_SEQ_LENS[0])
    item = _covered_item(sk, batch=1, tpot=0.01, coverage={"measured": 0, "total": 2})
    item.pop("measures")
    out = _read_folded(monkeypatch, [item])
    assert out["frontier"][sk]
    assert out["measured_track"]["state"] == "engine_error"
    assert "has no K=0 measure" in out["measured_track"]["error"]


def test_no_coverage_when_the_lake_is_unavailable(monkeypatch):
    from backend.measure.scope import NONE
    monkeypatch.setattr(f, "_frontier_scope", lambda: (NONE, "unavailable"))
    sk = str(GALLERY_SEQ_LENS[0])
    out = f._read_entry({"model_id": "m", "layouts": [], "error": None,
                         "_theory": [_timed_item(sk)]}, f.FrontierView("B200"))
    assert out["frontier"][sk]
    assert out["measured_track"] == {"state": "unavailable"}


def test_a_fold_the_lake_drops_mid_request_draws_sol_only(
        monkeypatch, tmp_path):
    """The lake answered its version — the view reads a measured track —
    but the rows' lookup failed: the model still draws, SOL-only, its track
    "unavailable", rather than the whole frontier 503ing. That drawing is
    never kept under the measured track's key: the next read the lake
    answers folds the rates again instead of serving the outage's drawing."""
    from backend.measure.lookup import MeasurementsUnavailable
    _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model, view = "test/Model", f.FrontierView("B200")
    sk = str(GALLERY_SEQ_LENS[0])

    def unavailable(items, **_kw):
        raise MeasurementsUnavailable(
            "measurements unavailable: the Kernel Bench service did not "
            "answer (ReadTimeout)")

    monkeypatch.setattr(f, "measure_points", unavailable)
    built = f._finish_frontier(model, view, plan, results)
    assert built["measured_track"] == {"state": "unavailable"}
    assert built["frontier"][sk]
    assert built["frontier_measured"][sk] == []

    folds: list = []

    def apply(items, **_kw):
        folds.append(len(items))
        for it in items:
            dec = it["sol"]["phases"]["decode"]
            _fold_into(dec, t_measured=0.05)
            dec["coverage"] = {"measured": 1, "total": 1}

    monkeypatch.setattr(f, "measure_points", apply)
    got = f._frontier_read(model, view)["drawn"]
    assert folds  # the degraded drawing was not kept: the fold ran again
    assert got["measured_track"]["state"] == "reads"
    assert got["frontier_measured"][sk]


def _track(monkeypatch, state):
    """The measured track every read sees from now on: the lake answering
    (`reads`, the scope `_kept` folds under) or not (`unavailable`)."""
    from backend.measure.scope import MeasurementScope
    scope = MeasurementScope(None, "lake", 1, 1, "r")
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        (scope, "reads") if state == "reads" else (f.NO_MEASUREMENTS, state)))


def test_a_flapping_lake_reads_both_drawings_back(monkeypatch, tmp_path):
    """A lake that flaps between answering and not moves the measured
    track between "reads" and "unavailable". Each is drawn off the entry
    once and kept — the unavailable one under scope `none`, whatever the
    outage — so the next flip reads its drawing, not the entry (a model's
    entry is up to ~150 MB)."""
    _entries, reads, _folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model, view = "test/Model", f.FrontierView("B200")
    sk = str(GALLERY_SEQ_LENS[0])
    measured = f._finish_frontier(model, view, plan, results)
    _track(monkeypatch, "unavailable")
    first = f._frontier_read(model, view)["drawn"]
    assert first["measured_track"] == {"state": "unavailable"}
    def sol(drawing):
        return {k: [(p["layout"], p["batch"], p["tok_s_per_gpu"]) for p in pts]
                for k, pts in drawing["frontier"].items()}
    assert sol(first) == sol(measured)
    assert first["frontier_measured"][sk] == []
    for state, want in (("reads", measured), ("unavailable", first),
                        ("reads", measured)):
        _track(monkeypatch, state)
        reads.clear()
        assert f._frontier_read(model, view) == {"drawn": want}
        assert reads == [], state


def test_a_fold_the_lake_drops_keeps_the_sol_drawing(monkeypatch, tmp_path):
    """The SOL drawing a failed fold draws is the unavailable track's: a
    later request that finds the lake down reads it back without the
    entry."""
    from backend.measure.lookup import MeasurementsUnavailable
    _entries, reads, _folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model, view = "test/Model", f.FrontierView("B200")

    def unavailable(items, **_kw):
        raise MeasurementsUnavailable("measurements unavailable: ReadTimeout")

    monkeypatch.setattr(f, "measure_points", unavailable)
    built = f._finish_frontier(model, view, plan, results)
    _track(monkeypatch, "unavailable")
    reads.clear()
    assert f._frontier_read(model, view) == {"drawn": built}
    assert reads == []


def test_a_fold_mismatch_is_an_engine_error_not_a_lake_outage(
        monkeypatch, tmp_path):
    """A deterministic coverage hole is our bug, not Kernel Bench's state.

    The SOL drawing still ships, but the degraded drawing is labelled
    `engine_error` and is not kept: a later request must retry the fold
    rather than read back what looks like an unavailable lake.
    """
    _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model, view = "test/Model", f.FrontierView("B200")
    sk = str(GALLERY_SEQ_LENS[0])

    def incomplete(core, _measureAt, _view, _scope):
        plain = [str(i) for i, point in enumerate(core["_theory"])
                 if not f._stepDraftTokens(point["sol"]["phases"]["decode"])]
        return {"rates": {}, "timings": {},
                "coverage": {"keys": [], "points": {
                    i: [1, 1, []] for i in plain[1:]}}}

    monkeypatch.setattr(f, "_measured_rates", incomplete)
    built = f._finish_frontier(model, view, plan, results)
    assert built["measured_track"]["state"] == "engine_error"
    assert "did not cover" in built["measured_track"]["error"]
    assert built["frontier"][sk]
    assert built["frontier_measured"][sk] == []
    stored = f._Stored(model, "v1")
    assert f._FRONTIER_VIEWS.get(stored.view_key(
        view, f.NO_MEASUREMENTS, "engine_error")) is None
    assert f._FRONTIER_VIEWS.get(stored.view_key(
        view, f.NO_MEASUREMENTS, "unavailable")) is None


def test_one_request_reads_the_lake_once(monkeypatch):
    """A frontier request reads the lake's version once and draws every
    model under that answer: one response reads one lake, and a lake that
    does not answer costs one timeout, not one per model."""
    import asyncio
    asked: list = []
    track = (f.NO_MEASUREMENTS, "unavailable")

    def scope():
        asked.append(1)
        return track

    seen: list = []

    def read(model_id, view, track=None, *, cache_only=False):
        seen.append(track)
        return {"drawn": {"model_id": model_id}}

    monkeypatch.setattr(f, "_frontier_scope", scope)
    monkeypatch.setattr(f, "_frontier_read", read)
    monkeypatch.setattr(f, "iter_snapshot_ids", lambda: ["a/1", "b/2", "c/3"])
    out = asyncio.run(f.build_frontier(f.FrontierView("B200")))
    assert [m["model_id"] for m in out["models"]] == ["a/1", "b/2", "c/3"]
    assert asked == [1]
    assert seen == [track] * 3


def test_a_speculative_view_without_its_k_measure_is_an_engine_error(monkeypatch):
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    sk = str(GALLERY_SEQ_LENS[0])
    out = f._read_entry({"model_id": "m", "layouts": [], "error": None,
                         "_theory": [_timed_item(sk, drafts={1: 0.012})]},
                        f.FrontierView("B200", draft_tokens=1))
    assert out["frontier"][sk]
    assert out["measured_track"]["state"] == "engine_error"
    assert "no K=1 point with a measure record" in out["measured_track"]["error"]


def test_a_kept_view_carries_its_coverage_and_floors_reuse_the_fold(
        monkeypatch, tmp_path):
    """The coverage is the fold's, read once per measurement scope: another
    view's floors redraw it from the kept rates without folding again, and
    a kept view carries it."""
    _entries, _reads, folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    view = f.FrontierView("B200")
    first = f._finish_frontier("test/Model", view, plan, results)
    sk = str(GALLERY_SEQ_LENS[0])
    cov = first["measured_track"]["coverage"][sk]
    assert cov["points"] == len(first["frontier"][sk]) > 0
    assert cov["complete"] == cov["points"]
    n = len(folds)
    other = f._frontier_read("test/Model", view._replace(comm_latency_us=7.0))
    assert len(folds) == n
    assert other["drawn"]["measured_track"]["coverage"][sk]["points"] == len(
        other["drawn"]["frontier"][sk])
    assert f._frontier_read("test/Model", view)["drawn"]["measured_track"] == (
        first["measured_track"])


def test_a_cold_frontier_publishes_kept_models_at_once_and_computes_in_the_clients_order(
        monkeypatch):
    """While the frontier computes, each finished model is published with
    the response's head (`offload.report_partial`), so a page draws it at
    once. A model already kept lands first even when the client did not ask
    for it — its read waits behind nothing that computes — and of the models
    that compute, the one the client put first (`priority`) computes first."""
    import asyncio

    from backend.api import offload

    track = (f.NO_MEASUREMENTS, "none")
    monkeypatch.setattr(f, "_frontier_scope", lambda: track)
    monkeypatch.setattr(f, "_model_concurrency", lambda: 1)
    monkeypatch.setattr(f, "iter_snapshot_ids", lambda: ["a/1", "b/2", "c/3"])

    def read(model_id, view, track=None, *, cache_only=False):
        if model_id == "c/3":
            return {"drawn": {"model_id": model_id}}
        return {"walk": {"cards": [], "layouts": []}}

    computed: list = []

    async def compute(model_id, view, track, got):
        computed.append(model_id)
        await asyncio.sleep(0.3)
        return {"model_id": model_id}

    monkeypatch.setattr(f, "_frontier_read", read)
    monkeypatch.setattr(f, "_compute_frontier", compute)
    view = f.FrontierView("B200")

    async def scenario():
        with pytest.raises(offload.StillComputing) as info:
            await offload.within_deadline(
                f.frontier_request_key(view, track),
                lambda: f.build_frontier(view), hint=["b/2"])
        partial = info.value.partial
        offload.set_deadline(None)
        whole = await offload.within_deadline(
            f.frontier_request_key(view, track), lambda: f.build_frontier(view))
        return partial, whole

    monkeypatch.setattr(offload, "_deadline_s", 0.15)
    try:
        partial, whole = asyncio.run(scenario())
    finally:
        offload.set_deadline(None)
    assert partial["head"]["seq_lens"] == list(GALLERY_SEQ_LENS)
    assert partial["head"]["config"]["gpu_sku"] == "B200"
    assert [m["model_id"] for m in partial["items"]][:1] == ["c/3"]
    assert computed == ["b/2", "a/1"]
    assert [m["model_id"] for m in whole["models"]] == ["a/1", "b/2", "c/3"]
    assert {k: whole[k] for k in partial["head"]} == partial["head"]

def _route_scenario(monkeypatch, scopes, states_by_call):
    """Two requests to `/gallery/frontier` (the route coroutine, one event
    loop): the first misses the deadline and its finished response is held
    for a retry; the lake's answer between them is `scopes[i]`. Each build
    draws its one model under `states_by_call[n]` (the n-th build's state,
    or the track's when None). Returns the second response and how many
    builds drew under the request's track (not counting the SOL partial).
    """
    import asyncio

    from backend.api import offload
    from backend.api.routes import gallery as route

    calls = {"scope": 0, "builds": 0}

    def scope():
        got = scopes[min(calls["scope"], len(scopes) - 1)]
        calls["scope"] += 1
        return got

    def read(model_id, view, track=None, *, cache_only=False):
        if cache_only:
            return {}
        time.sleep(0.3)
        if track[1] == "none":               # the SOL partial a "reads" build publishes first
            return {"drawn": {"model_id": model_id, "measured_track": {"state": "none"}}}
        state = states_by_call[min(calls["builds"], len(states_by_call) - 1)]
        calls["builds"] += 1
        return {"drawn": {"model_id": model_id,
                          "measured_track": {"state": state or track[1]}}}

    monkeypatch.setattr(f, "_frontier_scope", scope)
    monkeypatch.setattr(f, "_model_concurrency", lambda: 1)
    monkeypatch.setattr(f, "iter_snapshot_ids", lambda: ["a/1"])
    monkeypatch.setattr(f, "_frontier_read", read)
    ask = dict(gpu_sku="B200", gpu_memory_fraction=0.9, comm_latency_us=2.0,
               launch_floor_us=1.0, draft_tokens=0, acceptance_rate=0.7,
               priority="", have_tag=0, have=0)

    async def scenario():
        with pytest.raises(offload.StillComputing):
            await route.gallery_frontier(**ask)
        await asyncio.sleep(0.6)             # finished, held for the retry
        offload.set_deadline(None)
        return await route.gallery_frontier(**ask)

    monkeypatch.setattr(offload, "_deadline_s", 0.1)
    try:
        second = asyncio.run(scenario())
    finally:
        offload.set_deadline(None)
    return second, calls["builds"]


def test_a_response_drawn_while_the_lake_was_unavailable_is_not_served_once_it_answers(
        monkeypatch):
    """Production, 2026-11-09: after Kernel Bench came back, the bare
    frontier request kept answering every model "unavailable" — the whole
    response, held for a retry, was keyed on the view alone. Its identity
    includes the track it was drawn under (`frontier_request_key`)."""
    from backend.measure.scope import MeasurementScope
    lake = MeasurementScope(None, "lake", 1, 1, "r")
    second, builds = _route_scenario(
        monkeypatch, [(f.NO_MEASUREMENTS, "unavailable"), (lake, "reads")], [None])
    assert [m["measured_track"]["state"] for m in second["models"]] == ["reads"]
    assert builds == 2


def test_a_response_degraded_mid_request_is_not_held(monkeypatch):
    """Under a "reads" track a model whose fold the lake failed mid-request
    is drawn "unavailable": that response is not the track's answer, so it
    is not held for the retry (`offload.report_degraded`)."""
    from backend.measure.scope import MeasurementScope
    lake = MeasurementScope(None, "lake", 1, 1, "r")
    second, builds = _route_scenario(
        monkeypatch, [(lake, "reads")], ["unavailable", None])
    assert [m["measured_track"]["state"] for m in second["models"]] == ["reads"]
    assert builds == 2


def test_a_bare_frontier_request_and_its_explicit_defaults_are_one_identity(monkeypatch):
    """The page sends the floors explicitly; the smoke and a bare link do
    not. Both are one view, so one response identity."""
    from fastapi.testclient import TestClient

    import backend.api  # noqa: F401  # registers routes
    from backend.api._app import app
    from backend.api.routes import gallery as route

    keys: list = []

    async def capture(key, start, **_kw):
        keys.append(key)
        return {}

    async def track(_view):
        return (f.NO_MEASUREMENTS, "none")

    monkeypatch.setattr(route, "within_deadline", capture)
    monkeypatch.setattr(route, "frontier_track", track)
    client = TestClient(app)
    for params in ({"gpu_sku": "B200"},
                   {"gpu_sku": "B200", "comm_latency_us": "2", "launch_floor_us": "1"},
                   {"gpu_sku": "B200", "comm_latency_us": "2.0",
                    "launch_floor_us": "1.0", "gpu_memory_fraction": "0.9",
                    "draft_tokens": "0", "acceptance_rate": "0.7"}):
        assert client.get("/gallery/frontier", params=params).status_code == 200
    assert len(keys) == 3 and len(set(keys)) == 1


def test_a_measured_cache_hit_returns_without_computing(monkeypatch):
    """The lake being readable never invalidates a drawing already kept for
    its own track (`frontier_model` must not fall back to a second walk)."""
    import asyncio

    view = f.FrontierView("B200")
    track = (object(), "reads")
    monkeypatch.setattr(
        f, "_frontier_read",
        lambda *_args, **_kwargs: {"drawn": {"model_id": "kept"}})
    monkeypatch.setattr(
        f, "_compute_frontier",
        lambda *_args: (_ for _ in ()).throw(AssertionError("computed")))
    assert asyncio.run(f.frontier_model("m", view, track)) == {
        "model_id": "kept"}


def test_a_measured_request_publishes_sol_before_the_fold(monkeypatch):
    """A lake read or fold may take time; the model's SOL partial is public
    first and the measured drawing replaces it by model id."""
    import asyncio

    view = f.FrontierView("B200")
    track = (object(), "reads")
    monkeypatch.setattr(f, "_frontier_scope", lambda: track)
    monkeypatch.setattr(f, "iter_snapshot_ids", lambda: ["a/1"])
    monkeypatch.setattr(f, "_model_concurrency", lambda: 1)
    monkeypatch.setattr(
        f, "_frontier_read",
        lambda model, view_arg, track_arg=None, *, cache_only=False:
        {} if cache_only or track_arg[1] == "reads" else {"walk": {}})
    monkeypatch.setattr(f, "_layout_law", lambda item: {"schema": "law"})
    monkeypatch.setattr(f, "_finish_frontier", lambda *args, **kwargs: {
        "model_id": args[0], "track": kwargs.get("track")})
    published: list = []
    monkeypatch.setattr(f, "report_partial", lambda item=None, **_kw:
                        published.append(item))

    async def compute(model_id, view_arg, track_arg, got):
        return {"model_id": model_id, "computed": True,
                "stage": "measured" if track_arg[1] == "reads" else "sol"}

    monkeypatch.setattr(f, "_compute_frontier", compute)
    out = asyncio.run(f.build_frontier(view))
    assert out["models"] == [{"model_id": "a/1", "computed": True,
                              "stage": "measured"}]
    assert [item["stage"] for item in published
            if item is not None][:-1] == ["sol", "measured"]


def test_a_layout_the_walk_does_not_deploy_projects_as_its_answer():
    """A walk's refusal, error or missing peak is that layout's answer on
    every card of the class (`_restamp_layout`), never an absent one; a record
    that is not a current serving law is an error, not a silent None."""
    cfg = ParallelConfig()
    assert f._restamp_layout({"error": "boom"}, "B200", cfg) == {"error": "boom"}
    no_peak = f._restamp_layout({"no_peak": ["fp4", "mma"]}, "H100-SXM", cfg)
    assert no_peak == {"error": str(f.NoPeakForDtype("H100-SXM", "fp4", "mma"))}
    with pytest.raises(ValueError, match="not a serving-law"):
        f._restamp_layout({"schema": "serving-law-v1"}, "B200", cfg)
