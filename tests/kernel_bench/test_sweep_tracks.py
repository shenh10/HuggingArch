"""The plan enumerates the points a surface serves, from the surface's track
table (`sweep_grid.TrackSpec`).

A measurement is only worth taking if some served point looks it up. The plan
used to build its decode cells with no output length, so decode attention
ran over a cache of the context alone (`s_kv = 1024`) while every served
point sits at the context plus half the output (`s_kv = 1152`). Those rows
were measured and never read.

These tests pin the enumeration to the Scaling surface: the output length is
the caller's to state (no default), the track table is the enumeration's
whole input (the enumerator names no surface), and `ROUTINE_TRACKS` plans
every speculative width the Scaling panel offers.
"""
from __future__ import annotations

import inspect
import json
from collections import defaultdict
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.sweep_grid import (
    ROUTINE_BATCH_GRID,
    ROUTINE_OUTPUT_TOKENS,
    ROUTINE_TRACKS,
    TrackSpec,
)
from tests.snapshots import resolve_snapshot

LLAMA = "meta-llama/Llama-3.2-3B"
CARD = "B300"


def _key(job: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return tuple(job["physical_key"])


class TestDecodeServesTheOutputLength(unittest.TestCase):
    def test_decode_attention_runs_over_context_plus_half_the_output(self):
        from kernel_bench.plan import build_jobs
        jobs, _dropped = build_jobs(
            model_id=LLAMA, batch_size=1, seq_len=1024, phase="decode",
            avg_output_tokens=256, resolved=resolve_snapshot(LLAMA))
        attn = [j for j in jobs if j["signature"] == "gqa_attention"]
        self.assertTrue(attn, "Llama decode plans its attention")
        self.assertEqual({j["dims"]["s_kv"] for j in attn}, {1152})


class TestNoHiddenOutputLength(unittest.TestCase):
    """The output length has no neutral value — 0 is the bug — so the two
    planners refuse to run without it and their signatures carry no
    default."""

    def test_omitting_it_raises(self):
        from kernel_bench.plan import build_jobs_over_batches
        from kernel_bench.sweep import build_sweep_jobs
        with self.assertRaises(TypeError):
            build_jobs_over_batches(model_id=LLAMA, batch_sizes=[1],
                                    seq_len=1024, phase="decode")
        with self.assertRaises(TypeError):
            build_sweep_jobs(model_id=LLAMA, gpu=CARD, tracks=ROUTINE_TRACKS,
                             parallel_configs=[])

    def test_the_signatures_state_no_default(self):
        from kernel_bench.plan import build_jobs_over_batches
        from kernel_bench.sweep import build_sweep_jobs
        for fn in (build_jobs_over_batches, build_sweep_jobs):
            p = inspect.signature(fn).parameters["avg_output_tokens"]
            self.assertIs(p.kind, inspect.Parameter.KEYWORD_ONLY, fn.__name__)
            self.assertIs(p.default, inspect.Parameter.empty, fn.__name__)
        for name in ("tracks", "parallel_configs"):
            p = inspect.signature(build_sweep_jobs).parameters[name]
            self.assertIs(p.default, inspect.Parameter.empty, name)


class TestTheTrackTableIsTheWholeInput:
    """`ROUTINE_TRACKS` × a pair of deployments × two Scaling contexts give
    exactly the cells those tracks ask, and nothing in the enumerator knows
    the surface's name."""

    _SEQS = (1024, 2048)

    class _Cap:
        max_batch = 4096
        max_prefill_batch = 4096
        beyond_max_context = False

    class _PastContext(_Cap):
        beyond_max_context = True

    def _calls(
            self, monkeypatch, cap_at=None, *, prefill_chunk_size=8192,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        import kernel_bench.plan as pl
        import kernel_bench.sweep as sw
        from kernel_bench.sweep import preset_parallel_configs
        calls: List[Dict[str, Any]] = []

        def fake(**kw):
            calls.append(kw)
            return {point: ([], []) for point in kw["points"]}
        monkeypatch.setattr(pl, "build_jobs_over_points", fake)
        cap_at = cap_at or (lambda seq: self._Cap())
        monkeypatch.setattr(sw, "_ceilings_for_deployment",
                            lambda **kw: {s: cap_at(s) for s in kw["seq_lens"]})
        _jobs, stats = sw.build_sweep_jobs(
            model_id=LLAMA, gpu=CARD, seq_lens=list(self._SEQS),
            tracks=ROUTINE_TRACKS,
            parallel_configs=preset_parallel_configs([{}, {"tp_attention": 2,
                "tp_dense_ffn": 2, "tp_moe_expert": 2}]),
            avg_output_tokens=ROUTINE_OUTPUT_TOKENS,
            resolved=resolve_snapshot(LLAMA),
            prefill_chunk_size=prefill_chunk_size)
        return calls, stats

    def test_the_cells_are_the_track_points(self, monkeypatch):
        from backend.inference.calibration_overlay import parallel_signature
        from kernel_bench.sweep import (
            DECODE_BATCH_CEILING, PREFILL_TOKEN_BUDGET, preset_parallel_configs,
        )
        calls, stats = self._calls(monkeypatch)
        got: Dict[Tuple[str, str, int, int], Tuple[int, ...]] = {}
        walked = set()
        for kw in calls:
            # One walk per (deployment, phase): every context and batch of
            # it is a point of that walk.
            cell = (parallel_signature(kw["parallel"]), kw["phase"])
            assert cell not in walked, f"cell walked twice: {cell}"
            walked.add(cell)
            for point in kw["points"]:
                context = (*cell, point.seq_len, point.s_prefix_cached)
                got[context] = tuple(sorted(set(
                    (*got.get(context, ()), point.batch))))
            assert kw["avg_output_tokens"] == ROUTINE_OUTPUT_TOKENS
        presets = {parallel_signature(pc) for _l, pc in
                   preset_parallel_configs([{}, {"tp_attention": 2,
                       "tp_dense_ffn": 2, "tp_moe_expert": 2}])}
        keys = tuple(bp["key"] for bp in ROUTINE_BATCH_GRID)
        want: Dict[Tuple[str, str, int, int], Tuple[int, ...]] = {}
        for p in presets:
            for seq in self._SEQS:
                pre_ceil = max(1, min(self._Cap.max_prefill_batch,
                                      PREFILL_TOKEN_BUDGET // seq))
                dec_ceil = min(self._Cap.max_batch, DECODE_BATCH_CEILING)
                def batches(field, ceil):
                    out = []
                    for k in keys:
                        b = ceil if k == "max" else next(
                            bp[field] for bp in ROUTINE_BATCH_GRID
                            if bp["key"] == k)
                        if k == "max" or b <= ceil:
                            out.append(b)
                    return tuple(sorted(set(out)))
                want[(p, "prefill", seq, 0)] = batches("prefill_batch_size", pre_ceil)
                want[(p, "decode", seq, 0)] = batches("decode_batch_size", dec_ceil)
        assert got == want
        assert stats["reused"] == []

    def test_h2_records_the_chunk_campaign_and_its_exact_key_debt(self, monkeypatch):
        """The campaign knob and rollout stage are explicit in stats.

        Default policy says 8192 and H2 emits launch-chunk cells.  0 and None
        are both explicit unchunked campaigns.  The exact-key counts are this
        campaign's plan, not an assumed full-corpus rollout.
        """
        _calls, stats = self._calls(monkeypatch)
        assert stats["prefill_chunk_size"] == 8192
        assert stats["prefill_chunk_jobs"] == "chunked-batch-h2"
        assert stats["prefill_chunk_campaign"] == "h2-chunked-jobs/c=8192"
        assert stats["prefill_chunk_rollout"] == {
            "stage": "h2-plan-rekey", "mode": "chunked",
            "measurement_rollout": "staged-corpus-campaign-pending-h4",
            "full_corpus_assumed": False, "exact_key_only": True,
            "missing_context_key_is_incomplete": True,
        }
        assert "distinct_prefill_chunk_keys" in stats
        assert "distinct_prefill_context_keys" in stats

        _calls, unchunked = self._calls(monkeypatch, prefill_chunk_size=0)
        assert unchunked["prefill_chunk_size"] == 0
        assert unchunked["prefill_chunk_jobs"] == "unchunked-batch-h2"
        assert unchunked["prefill_chunk_campaign"] == "h2-unchunked-jobs/c=0"
        assert unchunked["prefill_chunk_rollout"]["mode"] == "unchunked"

    def test_one_decomposition_feeds_one_walk_and_keeps_endpoint_mapping(
            self, monkeypatch):
        """A long endpoint's launch points are one cell of one walk."""
        import kernel_bench.plan as pl
        import kernel_bench.sweep as sw
        from kernel_bench.sweep import preset_parallel_configs
        from kernel_bench.plan import PlanPoint
        calls = []

        def fake(**kw):
            calls.append(kw)
            return {point: ([{
                "signature": "gqa_attention",
                "dims": {"s_kv": point.seq_len, "s_query": point.seq_len
                         - point.s_prefix_cached},
                "dtype_env": {}, "phase": kw["phase"],
                "physical_key": [
                    "gqa_attention",
                    f'{{"s_kv":{point.seq_len},"s_query":'
                    f'{point.seq_len - point.s_prefix_cached}}}',
                    "{}", kw["phase"]],
            }], []) for point in kw["points"]}

        monkeypatch.setattr(pl, "build_jobs_over_points", fake)
        monkeypatch.setattr(
            sw, "_ceilings_for_deployment",
            lambda **kw: {16384: self._Cap()})
        _jobs, stats = sw.build_sweep_jobs(
            model_id=LLAMA, gpu=CARD, seq_lens=[16384],
            tracks=(TrackSpec("plain", "prefill", ("b1",)),),
            parallel_configs=preset_parallel_configs([{}]),
            avg_output_tokens=ROUTINE_OUTPUT_TOKENS,
            resolved=resolve_snapshot(LLAMA))

        prefill = [kw for kw in calls if kw["phase"] == "prefill"]
        assert len(prefill) == 1
        assert prefill[0]["points"] == [
            PlanPoint(8192, 0, 1, 1), PlanPoint(16384, 8192, 1, 1)]
        assert stats["prefill_chunk_endpoints"] == 1
        assert stats["prefill_chunk_terms"] == 2
        assert stats["prefill_chunk_map"] == [{
            "deployment": "tp=1", "phase": "prefill", "endpoint_index": 0,
            "endpoint": {"seq_len": 16384, "s_prefix_cached": 0,
                         "s_verify": 1, "batch": 1},
            "chunks": [
                {"index": 0, "start": 0, "end": 8192, "width": 8192,
                 "launch_batch": 1},
                {"index": 1, "start": 8192, "end": 16384, "width": 8192,
                 "launch_batch": 1},
            ],
        }]
        from backend.inference.prefill_geometry import (
            PrefillEndpointGeometry, prefill_chunk_map as runtime_map)
        runtime_chunk_map = runtime_map(
            PrefillEndpointGeometry(16384, 0, 1, 1),
            prefill_chunk_size=8192)
        assert runtime_chunk_map is not None
        assert stats["prefill_chunk_map"] == [{
            **runtime_chunk_map,
            "deployment": "tp=1",
        }]

        manifest = stats["distinct_prefill_context_key_manifest"]
        manifest_by_signature = [
            record for records
            in stats["distinct_prefill_context_key_manifest_by_signature"].values()
            for record in records]
        assert manifest_by_signature == manifest
        manifest_keys = {tuple(record["physical_key"]) for record in manifest}
        assert len(manifest_keys) == stats["distinct_prefill_context_keys"]
        assert manifest_keys == {
            ("gqa_attention", '{"s_kv":8192,"s_query":8192}', "{}", "prefill"),
            ("gqa_attention", '{"s_kv":16384,"s_query":8192}', "{}", "prefill"),
        }
        assert {record["model"] for record in manifest} == {LLAMA}
        assert {record["page"] for record in manifest} == {None}
        assert all(record["dtype"] == {} for record in manifest)
        assert all(record["variant"] == {} for record in manifest)

    def test_h4_can_union_manifest_facets_across_models(self, monkeypatch):
        """Campaign staging groups exact keys without folding provenance.

        The manifest keeps the raw physical key unionable while also carrying
        model/page/dtype/variant read from pre-dedup planner jobs.  Two models
        can therefore share an exact-key campaign denominator and still expose
        their distinct model coverage (and page/dtype/variant slices).
        """
        import kernel_bench.plan as pl
        import kernel_bench.sweep as sw
        from kernel_bench.sweep import preset_parallel_configs

        def fake(**kw):
            jobs = []
            for page in (16, 32):
                for dtype in ({"activation": "bf16"},
                              {"activation": "fp8"}):
                    for variant in ({}, {"s_window": 128}):
                        dims = {"s_kv": 8192, "s_query": 8192,
                                "kv_block_size": page, **variant}
                        jobs.append({
                            "signature": "gqa_attention",
                            "dims": dims,
                            "dtype_env": dtype,
                            "variant": variant,
                            "phase": kw["phase"],
                            "physical_key": [
                                "gqa_attention", json.dumps(dims, sort_keys=True),
                                json.dumps(dtype, sort_keys=True), kw["phase"]],
                        })
            return {point: (jobs, []) for point in kw["points"]}

        monkeypatch.setattr(pl, "build_jobs_over_points", fake)
        monkeypatch.setattr(
            sw, "_ceilings_for_deployment",
            lambda **kw: {16384: self._Cap()})
        resolved = resolve_snapshot(LLAMA)
        manifests = []
        for model in (LLAMA, "openai/gpt-oss-20b"):
            _jobs, stats = sw.build_sweep_jobs(
                model_id=model, gpu=CARD, seq_lens=[16384],
                tracks=(TrackSpec("plain", "prefill", ("b1",)),),
                parallel_configs=preset_parallel_configs([{}]),
                avg_output_tokens=ROUTINE_OUTPUT_TOKENS, resolved=resolved)
            manifests.extend(stats["distinct_prefill_context_key_manifest"])
            assert stats["distinct_prefill_context_keys"] == 8

        exact_keys = {tuple(record["physical_key"]) for record in manifests}
        assert len(manifests) == 16
        # The two models deliberately request the same eight physical keys;
        # model provenance remains two records per exact key after the union.
        assert len(exact_keys) == 8
        for field in ("model", "page", "dtype", "variant"):
            grouped = defaultdict(set)
            for record in manifests:
                value = record[field]
                if isinstance(value, dict):
                    value = tuple(sorted(value.items()))
                grouped[value].add(tuple(record["physical_key"]))
            expected = {
                "model": {LLAMA, "openai/gpt-oss-20b"},
                "page": {16, 32},
                "dtype": {(("activation", "bf16"),),
                          (("activation", "fp8"),)},
                "variant": {(), (("s_window", 128),)},
            }[field]
            assert set(grouped) == expected, field
            expected_counts = {"model": 8, "page": 4, "dtype": 4,
                               "variant": 4}[field]
            assert {len(keys) for keys in grouped.values()} == {
                expected_counts}, field

    def test_a_bad_chunk_knob_is_refused_before_a_model_is_resolved(self):
        from kernel_bench.sweep import build_sweep_jobs

        with pytest.raises(ValueError):
            build_sweep_jobs(
                model_id=LLAMA, gpu=CARD, tracks=ROUTINE_TRACKS,
                parallel_configs=[], avg_output_tokens=1,
                prefill_chunk_size=-1)

    def test_a_context_past_the_model_is_no_point(self, monkeypatch):
        """The plan asks `capacity.point_served`: a context past the model's
        published length has no point, so no cell."""
        calls, stats = self._calls(
            monkeypatch, cap_at=lambda seq: (self._PastContext() if seq > 1024
                                             else self._Cap()))
        assert calls and all(point.seq_len <= 1024
                             for kw in calls for point in kw["points"])
        assert all(p[3] <= 1024 for p in stats["points"])

    def test_routing_skew_is_a_track_axis_not_a_deployment_field(
            self, monkeypatch):
        """A ρ campaign plans one point per skew, all on the one walk family."""
        import kernel_bench.plan as pl
        import kernel_bench.sweep as sw
        from backend.inference.parallel_config import ParallelConfig

        asked = []

        def fake(**kw):
            asked.extend(kw["points"])
            return {point: ([], []) for point in kw["points"]}

        class Cap:
            max_batch = 1
            max_prefill_batch = 1
            beyond_max_context = False

        monkeypatch.setattr(pl, "build_jobs_over_points", fake)
        monkeypatch.setattr(sw, "_ceilings_for_deployment",
                            lambda **kw: {seq: Cap() for seq in kw["seq_lens"]})
        _jobs, stats = sw.build_sweep_jobs(
            model_id=LLAMA, gpu=CARD, seq_lens=[1024],
            tracks=[TrackSpec("skewed", "decode", ("b1",),
                              ep_load_factors=(1.0, 1.5))],
            parallel_configs=[("tp1", ParallelConfig())],
            avg_output_tokens=256, resolved=resolve_snapshot(LLAMA))
        assert {point.ep_load_factor for point in asked} == {1.0, 1.5}
        assert {point[-1] for point in stats["points"]} == {1.0, 1.5}

    def test_the_enumerator_names_no_surface(self):
        import kernel_bench.plan as pl
        import kernel_bench.sweep as sw
        for fn in (sw.build_sweep_jobs, sw._feasible_batches,
                   sw._per_gpu_ceiling, pl.build_jobs_over_points):
            src = inspect.getsource(fn).lower()
            assert "gallery" not in src, fn.__name__
            assert "routine_tracks" not in src, fn.__name__


class TestWorkersPlanTheSameJobs(unittest.TestCase):
    """Deployments planned in forked workers merge back to the list one
    process plans: same jobs, same order, same stats."""

    def test_parallel_equals_serial(self):
        from kernel_bench.sweep import build_sweep_jobs, preset_parallel_configs
        resolved = resolve_snapshot(LLAMA)

        def plan(workers):
            return build_sweep_jobs(
                model_id=LLAMA, gpu=CARD, seq_lens=[1024],
                tracks=ROUTINE_TRACKS,
                parallel_configs=preset_parallel_configs(
                    [{}, {"tp_attention": 2, "tp_dense_ffn": 2},
                     {"tp_attention": 4, "tp_dense_ffn": 4}]),
                avg_output_tokens=ROUTINE_OUTPUT_TOKENS, resolved=resolved,
                workers=workers)

        serial, parallel = plan(1), plan(3)
        self.assertTrue(serial[0])
        self.assertEqual(json.dumps(parallel, sort_keys=True, default=str),
                         json.dumps(serial, sort_keys=True, default=str))


def test_a_two_value_planner_migrates_drops_into_the_denominator(
        monkeypatch, tmp_path, capsys):
    """Old callers cannot classify their textual drops, but may not erase them.

    The compatibility shape is migrated as `legacy_drop`: the note stays as
    detail, and the sidecar/report still count the row.
    """
    import kernel_bench.plan as pl
    import kernel_bench.sweep as sw
    from backend.inference.parallel_config import ParallelConfig
    from kernel_bench.cli import main

    note = ("fused_qk_rope [blk]: no shape projection for signature 'rope' "
            "(shape_in=[[8, 32, 128, 128]])")

    def fake(**kw):
        return {point: ([], [note]) for point in kw["points"]}

    class Cap:
        max_batch = 1
        max_prefill_batch = 1
        beyond_max_context = False

    monkeypatch.setattr(pl, "build_jobs_over_points", fake)
    monkeypatch.setattr(sw, "_ceilings_for_deployment",
                        lambda **kw: {seq: Cap() for seq in kw["seq_lens"]})
    monkeypatch.setattr(sw, "_feasible_batches",
                        lambda cap, phase, seq, keys, **_: [1])
    _jobs, stats = sw.build_sweep_jobs(
        model_id=LLAMA, gpu=CARD,
        tracks=[TrackSpec("legacy", "decode", ("b1",))],
        parallel_configs=[("tp1", ParallelConfig())],
        seq_lens=[1024], avg_output_tokens=256,
        resolved=resolve_snapshot(LLAMA))

    assert stats["dropped"] == [note.split(" (shape_in=")[0]]
    assert stats["unplaced"] == [{
        "op": "fused_qk_rope", "aggregation_id": "blk",
        "reason": "legacy_drop",
        "detail": "no shape projection for signature 'rope' "
                  "(shape_in=[[8, 32, 128, 128]])",
    }]
    assert pl.unplaced_by_reason(stats["unplaced"])[0].startswith("legacy_drop")

    plan = tmp_path / "plan.jsonl"
    pl.write_plan([], str(plan), unplaced=stats["unplaced"])
    assert main(["report", "--jobs", str(plan), "--gpu", CARD]) == 0
    out = capsys.readouterr().out
    assert "structural 1" in out
    assert "legacy_drop: 1 row(s)" in out


class TestTrackSpecRefusesWhatTheEnumeratorCannotRead(unittest.TestCase):
    def test_unknown_values_raise(self):
        with self.assertRaises(ValueError):
            TrackSpec("x", "encode", ("b1",))
        with self.assertRaises(ValueError):
            TrackSpec("x", "decode", ("b3",))
        with self.assertRaises(ValueError):
            TrackSpec("x", "decode", ())
        with self.assertRaises(ValueError):
            TrackSpec("x", "prefill", ("b1",), prefix_new_tokens=0)
        with self.assertRaises(ValueError):
            TrackSpec("x", "prefill", ("b1",), geometry="chunked")


def test_frontier_tracks_are_the_dense_decode_ladder():
    from backend.inference.sweep_grid import FRONTIER_BATCH_GRID, FRONTIER_TRACKS
    assert len(FRONTIER_TRACKS) == 1
    t = FRONTIER_TRACKS[0]
    assert t.phase == "decode"
    assert t.batch_keys == tuple(bp["key"] for bp in FRONTIER_BATCH_GRID)
    assert "b8" in t.batch_keys and "b1024" in t.batch_keys


def test_sweep_surfaces_are_scaling_and_the_frontier():
    from kernel_bench.sweep import sweep_surfaces
    from backend.inference.sweep_grid import FRONTIER_TRACKS
    surfaces = sweep_surfaces()
    assert set(surfaces) == {"routine", "frontier"}
    assert surfaces["frontier"]["tracks"] is FRONTIER_TRACKS
    assert surfaces["frontier"]["deployments"] == "frontier"
    assert surfaces["routine"]["deployments"] == "routine"


class TestTheFrontierTableAsksTheDenseDecodeLadder:
    """`FRONTIER_TRACKS` at one context: every power-of-two decode batch
    that fits, plus max, and nothing in the enumerator names the frontier."""

    class _Cap:
        max_batch = 100
        max_prefill_batch = 100
        beyond_max_context = False

    def test_the_cells_include_the_frontier_rungs(self, monkeypatch):
        import kernel_bench.plan as pl
        import kernel_bench.sweep as sw
        from backend.inference.sweep_grid import FRONTIER_TRACKS, GALLERY_OUTPUT_TOKENS
        from kernel_bench.sweep import preset_parallel_configs
        calls: List[Dict[str, Any]] = []

        def fake(**kw):
            calls.append(kw)
            return {point: ([], []) for point in kw["points"]}
        monkeypatch.setattr(pl, "build_jobs_over_points", fake)
        monkeypatch.setattr(sw, "_ceilings_for_deployment",
                            lambda **kw: {s: self._Cap() for s in kw["seq_lens"]})
        sw.build_sweep_jobs(
            model_id=LLAMA, gpu=CARD, seq_lens=[1024],
            tracks=FRONTIER_TRACKS,
            parallel_configs=preset_parallel_configs([{}]),
            avg_output_tokens=GALLERY_OUTPUT_TOKENS,
            resolved=resolve_snapshot(LLAMA))
        assert len(calls) == 1
        batches = tuple(sorted({p.batch for p in calls[0]["points"]}))
        assert calls[0]["phase"] == "decode"
        # The max point is the rung (max_batch_rung): 100 -> 64
        assert batches == (1, 8, 16, 32, 64)


MIMO = "XiaomiMiMo/MiMo-7B-RL"     # dense, with an MTP stack
SPEC_SEQS = [1024, 16384]


@pytest.mark.regression          # four Scaling sweeps and a routine plan
class TestThePlanCoversScalingSpeculation(unittest.TestCase):
    """Every speculative preset the Scaling panel offers asks for decode
    keys at its verify width (K + 1). The routine plan, built from
    `ROUTINE_TRACKS` alone (no width handed in by the caller), plans every
    one of them."""

    @classmethod
    def setUpClass(cls):
        from backend.api._models import BatchPoint, InferenceEstimateSweepRequest
        from backend.api.estimate_env import resolve_estimate_environment
        from backend.inference.capacity import sweep_points
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.sweep_grid import (
            ROUTINE_BATCH_GRID, ROUTINE_OUTPUT_TOKENS, SPEC_PRESETS,
        )
        from backend.measure.auto_sweep import PointMeasure
        from kernel_bench.sweep import build_sweep_jobs
        cls.demand: Dict[str, Set[Tuple[str, ...]]] = {}
        for name, preset in SPEC_PRESETS.items():
            req = InferenceEstimateSweepRequest(
                model_id=MIMO, gpu_sku=CARD, parallel={},
                gpu_memory_fraction=0.9,
                avg_output_tokens=ROUTINE_OUTPUT_TOKENS, seq_lens=SPEC_SEQS,
                batch_grid=[BatchPoint(**bp) for bp in ROUTINE_BATCH_GRID],
                speculative_draft_tokens=int(preset["K"]),
                acceptance_rate=float(preset["alpha"]), selection_mode="latest")
            env = resolve_estimate_environment(req)
            plan, walked = sweep_points(
                seq_lens=SPEC_SEQS,
                batch_grid=[bp.model_dump() for bp in req.batch_grid], **env)
            keys = cls.demand.setdefault(name, set())
            for pts in plan.shape(walked).values():
                for pt in pts:
                    m = ((pt or {}).get("measures") or {}).get("decode")
                    if m is None:
                        continue
                    for k in PointMeasure.from_json(m["measure"]).key_set():
                        keys.add((k.signature, k.shape_json, k.dtype_env_json,
                                  k.phase))
        jobs, _stats = build_sweep_jobs(
            model_id=MIMO, gpu=CARD, seq_lens=SPEC_SEQS, tracks=ROUTINE_TRACKS,
            parallel_configs=[("tp=1", ParallelConfig())],
            avg_output_tokens=ROUTINE_OUTPUT_TOKENS,
            resolved=resolve_snapshot(MIMO))
        cls.supply = {_key(j) for j in jobs}

    def test_every_preset_is_planned(self):
        from backend.inference.sweep_grid import SPEC_PRESETS
        for name in SPEC_PRESETS:
            with self.subTest(preset=name):
                self.assertTrue(self.demand[name])
                self.assertEqual(sorted(self.demand[name] - self.supply), [])

    def test_the_presets_ask_for_different_widths(self):
        """The presets are not one key set restated: each widens the query
        block (s_query = K + 1) of the attention it verifies with."""
        from backend.inference.sweep_grid import SPEC_PRESETS
        for name, preset in SPEC_PRESETS.items():
            widths = {json.loads(k[1])["s_query"] for k in self.demand[name]
                      if k[0] == "gqa_attention"}
            self.assertEqual(widths, {int(preset["K"]) + 1}, name)


if __name__ == "__main__":
    unittest.main()
