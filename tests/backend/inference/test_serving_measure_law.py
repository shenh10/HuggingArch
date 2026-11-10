"""A serving-law record owns the measured row keys, not just SOL times."""
from __future__ import annotations

import copy
import json
import math
import time
import zlib
from typing import Any

import pytest
from dataclasses import replace

from backend.api._models import InferenceEstimateRequest
from backend.api.estimate_env import resolve_estimate_environment
from backend.api.gallery_frontier import (
    _GALLERY_PRECISION, _layout_law,
)
import backend.api.gallery_frontier as gallery
from backend.inference.batch_sweep import _measured_formula_weight, measure_points
from backend.inference.capacity import estimate_pending
from backend.inference.card import Card
from backend.inference.parallel_config import ParallelConfig
from backend.inference.parallel_sweep import class_cards
from backend.inference.sweep_grid import (
    FRONTIER_BATCH_GRID,
    FRONTIER_DRAFT_TOKENS,
    GALLERY_SEQ_LENS,
)
from backend.inference.serving_view import servingLawPointView
from backend.measure.auto_sweep import (
    blend,
    Folded, MEASURED, MISSING, PointMeasure, key_label,
)
from backend.inference.defaults import (COMM_LATENCY_US_DEFAULT,
                                       LAUNCH_FLOOR_US_DEFAULT)
from backend.inference.sol_law import (
    FloorCard,
    LawGraph,
    cardBinding,
    draftForwardsLaw,
    servingMeasureMatrix,
    servingMeasurePoint,
    servingLawPoint,
    servingLawRecord,
)
from tests.snapshots import snapshot_config
from tests.sweep_parity import FixtureBench

MODEL = "Qwen/Qwen3-30B-A3B"
PARALLEL = ParallelConfig(tp_attention=2, tp_dense_ffn=2, ep_moe=2)
LONG_CASES = [
    (MODEL, PARALLEL),
    ("deepseek-ai/DeepSeek-V4.1-Flash",
     ParallelConfig(ep_moe=8, dp_attention=8)),
]


def request(gpu: str) -> InferenceEstimateRequest:
    return InferenceEstimateRequest(
        model_id=MODEL, gpu_sku=gpu, avg_input_tokens=1024,
        avg_output_tokens=256, parallel=PARALLEL.to_dict(),
        framework="vllm", selection_mode="theoretical_only")


def same(got, want, path="point"):
    """Whole public shape: same keys, non-numbers equal, numbers to 1e-12."""
    if isinstance(got, dict):
        assert set(got) == set(want), f"{path}: {sorted(got)} != {sorted(want)}"
        for key in got:
            same(got[key], want[key], f"{path}.{key}")
        return
    if isinstance(got, (int, float)) and not isinstance(got, bool):
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=0), path
    else:
        assert got == want, path


def without_anomaly_numbers(measure: dict[str, Any]) -> dict[str, Any]:
    """The pool's lookup semantics: anomaly numerators are not selection facts."""
    point = PointMeasure.from_json(measure["measure"])
    point = replace(point, inputs=tuple(
        None if value is None else {
            "anchor": value.get("anchor"),
            "dispatch": value.get("dispatch"),
        } for value in point.inputs), aggregation_ids=(), sol=())
    return {**measure, "measure": point.to_json(),
            **({"terms": []} if "terms" in measure else {})}


def _reference_fold(measure: PointMeasure, select):
    """The former product fold, kept only as this file's oracle."""
    rows = []
    times = []
    for i, key in enumerate(measure.keys):
        if measure.formula.zero_cost[i]:
            rows.append((MEASURED, 0.0, measure.labels[i], measure.reasons[i]))
            times.append(0.0)
            continue
        if key is None:
            rows.append((measure.unkeyed[i], None, measure.labels[i],
                         measure.reasons[i] or None))
            times.append(None)
            continue
        context = (measure.contexts[i] if measure.contexts
                   else measure.context)
        t, state, detail = select(
            key, context, measure.labels[i],
            measure.inputs[i] if measure.inputs else None)
        if t is None:
            rows.append((state, None, key_label(key), None))
            times.append(None)
            continue
        rows.append((MEASURED, float(t) * measure.launches[i],
                     key_label(key), detail))
        times.append(float(t))
    return Folded(measure, tuple(rows), tuple(times))


def _oracle_select(key, _context, label, _read):
    checksum = zlib.crc32(key.signature.encode()) + len(label)
    if checksum % 3:
        return checksum % 97 / 1e6 + 0.0001, MEASURED, {
            "source": "oracle", "source_badge": {"cross_check": "passed"},
            "utilization": {}}
    return None, MISSING, None


_STRUCTURAL_TERMS = {}


def _structural_terms(measure: PointMeasure) -> tuple[int, ...]:
    """The formula's weighted terms, including a draft's forwards share."""
    key = (measure.formula.text, len(measure.formula.zero_cost))
    if key not in _STRUCTURAL_TERMS:
        _STRUCTURAL_TERMS[key] = tuple(
            term for term, zero in enumerate(measure.formula.zero_cost)
            if not zero and float(_measured_formula_weight(
                measure.formula, term, len(measure.formula.zero_cost))) != 1.0)
    return _STRUCTURAL_TERMS[key]


def oracle(gpu: str):
    snapshot_config(MODEL)
    environment = resolve_estimate_environment(request(gpu))
    record = servingLawRecord(
        environment, gpu_sku=gpu, anchorContext=1024, outputTokens=256,
        parallel=PARALLEL, capacityCards=(gpu,), capacityContexts=(1024,))
    pool = servingLawPointView(servingLawPoint(
        record, card=Card.of(gpu), context=1024, outputTokens=256,
        decodeBatch=16, prefillBatch=1, comm_latency_us=2,
        launch_floor_us=1))
    poolMeasure = servingMeasurePoint(
        record, card=Card.of(gpu), context=1024, batch=16)

    pending = estimate_pending(**resolve_estimate_environment(
        request(gpu).model_copy(update={"decode_batch_size": 16})))
    direct = pending.finish()
    directMeasures = pending.measures(direct)

    # Before the fold, the PointMeasure itself is byte-for-byte the same fact.
    assert without_anomaly_numbers(poolMeasure) == without_anomaly_numbers(
        directMeasures["decode"])
    poolItem = {"sol": copy.deepcopy(pool), "measures": {"decode": poolMeasure}}
    directItem = {"sol": servingLawPointView(direct),
                  "measures": {"decode": directMeasures["decode"]}}
    with FixtureBench() as bench:
        measure_points(
            [poolItem, directItem], selection_mode="latest",
            measurement_scope=bench.scope)
    # The partial track is the one fact the two read differently: the
    # direct walk blends its rows' SOL, the pool point leaves it to a
    # reader's floors (`test_pool_partial_is_a_direct_walk_s_at_every_k`).
    for item in (poolItem, directItem):
        decode = item["sol"]["phases"]["decode"]
        decode.pop("t_partial_s")
        decode.pop("partial_measured_share")
        decode["throughput"].pop("partial")
    same(poolItem["sol"], directItem["sol"])
    return record


def test_pool_measured_points_match_a_direct_walk_on_two_card_classes():
    for gpu in ("H100-SXM", "B200"):
        record = oracle(gpu)
        assert record["schema"] == "serving-law-v7"
        assert "measurePoints" in record["phases"]["decode"]


def _point_request(model: str, gpu: str, parallel: ParallelConfig,
                    batch: int, *, draft_tokens: int = 0) -> InferenceEstimateRequest:
    return InferenceEstimateRequest(
        model_id=model, gpu_sku=gpu, avg_input_tokens=1024,
        avg_output_tokens=256, parallel=parallel.to_dict(),
        decode_batch_size=batch, framework="vllm",
        speculative_draft_tokens=draft_tokens,
        speculative_draft="recursive" if draft_tokens else None,
        selection_mode="theoretical_only")


def _direct_measure(model: str, gpu: str, parallel: ParallelConfig,
                    batch: int, *, draft_tokens: int = 0) -> dict[str, Any]:
    pending = estimate_pending(**resolve_estimate_environment(
        _point_request(model, gpu, parallel, batch,
                       draft_tokens=draft_tokens)))
    result = pending.finish()
    return pending.measures(result)["decode"]


def test_a_stored_k1_measure_is_the_direct_speculative_walk():
    """One public K=1 point: the record table and a direct estimate state the
    same verify + draft `PointMeasure`, not two key derivations."""
    model = "Qwen/Qwen3.5-35B-A3B"
    parallel = ParallelConfig()
    snapshot_config(model)
    request = _point_request(model, "H100-SXM", parallel, 1,
                             draft_tokens=1)
    record = servingLawRecord(
        resolve_estimate_environment(request), gpu_sku="H100-SXM",
        anchorContext=1024, outputTokens=256, parallel=parallel,
        capacityCards=("H100-SXM",), capacityContexts=(1024,),
        measureContexts=(1024,), measureBatchGrid=[{"decode_batch_size": 1}])
    table = record["phases"]["decode"]["measurePoints"]
    assert table["schema"] == "measure-points-v7"
    ks = {int(point["draftTokens"]) for point in table["points"]}
    assert ks == {0, *FRONTIER_DRAFT_TOKENS}
    stored = servingMeasurePoint(
        record, card=Card.of("H100-SXM"), context=1024, batch=1,
        draft_tokens=1)
    direct = _direct_measure(model, "H100-SXM", parallel, 1, draft_tokens=1)
    assert (PointMeasure.from_json(stored["measure"])
            == PointMeasure.from_json(
                without_anomaly_numbers(direct)["measure"]))


def test_stored_measure_points_are_point_keys_after_json():
    """A v6 record's concrete keys survive JSON and equal the plan road.

    The record is walked on the class representative; the checked point is on
    another card of the class, so this also pins the GPU field of every
    stored physical key. The direct side is an estimate's own
    `PendingEstimate.measures`, which asks `point_keys` at that point.
    """
    for model, parallel in LONG_CASES:
        snapshot_config(model)
        cards = class_cards("H100-SXM")
        record = _layout_law(
            (model, cards, 0.9, _GALLERY_PRECISION.canonical(),
             parallel.to_dict()))
        assert record.get("schema") == "serving-law-v7", (model, record)
        throughJson = json.loads(json.dumps(record))
        table = throughJson["phases"]["decode"]["measurePoints"]
        assert table["cardless"]
        assert all("gpu" not in point for point in table["points"])
        publicPoints = {(int(point["context"]), int(point["batch"]),
                         int(point.get("draftTokens", 0)))
                        for point in table["points"]}
        for fact in record["capacity"]:
            if int(fact["context"]) not in GALLERY_SEQ_LENS:
                continue
            ceiling = int(fact["max_decode_batch"] or 0)
            if (ceiling <= 0 or fact.get("infeasibility_reason")
                    or fact.get("beyond_max_context")):
                continue
            expected = {
                (int(fact["context"]), int(item["decode_batch_size"]), 0)
                for item in FRONTIER_BATCH_GRID
                if item["decode_batch_size"] is not None
                and int(item["decode_batch_size"]) <= ceiling
            } | {(int(fact["context"]), ceiling, 0)}
            if record.get("speculative"):
                draftFacts = record.get("draft_capacity") or []
                draftCeiling = next(
                    (int(item.get("max_decode_batch") or 0)
                     for item in draftFacts
                     if item["gpu_sku"] == fact["gpu_sku"]
                     and int(item["context"]) == int(fact["context"])), 0)
                if draftCeiling > 0:
                    expected |= {
                        (int(fact["context"]), int(item["decode_batch_size"]), k)
                        for k in FRONTIER_DRAFT_TOKENS
                        for item in FRONTIER_BATCH_GRID
                        if item["decode_batch_size"] is not None
                        and int(item["decode_batch_size"]) <= draftCeiling
                    } | {(int(fact["context"]), draftCeiling, k)
                         for k in FRONTIER_DRAFT_TOKENS}
            missing = expected - publicPoints
            assert not missing, (model, fact["gpu_sku"], fact["context"],
                                 sorted(missing))
        for sku in cards:
            wanted = sorted(publicPoints)
            matrix = servingMeasureMatrix(
                throughJson, card=Card.of(sku), points=wanted)
            matrixFolds = matrix.fold(_oracle_select)
            for (context, batch, draft_tokens), row in matrix.point_rows.items():
                measure = PointMeasure.from_json(servingMeasurePoint(
                    throughJson, card=Card.of(sku), context=context,
                    batch=batch, draft_tokens=draft_tokens)["measure"])
                reference = _reference_fold(measure, _oracle_select)
                got = matrixFolds[row]
                where = (model, sku, context, batch, draft_tokens)
                assert reference.track == got.track, where
                assert reference.rows == got.rows, where
                if reference.track.total_s is None:
                    assert got.total != got.total
                else:
                    assert math.isclose(
                        got.total, reference.track.total_s, rel_tol=1e-12), where
                structural = _structural_terms(measure)
                want = reference.subtotal(structural)
                gotShare = got.subtotal(structural)
                if want is None or gotShare is None:
                    assert want is None and gotShare is None
                else:
                    assert math.isclose(
                        gotShare, want, rel_tol=1e-12), (*where, structural)
            priced = gallery._restamp_layout(
                throughJson, sku, parallel)
            assert "error" not in priced, (model, sku, priced)
        point = next(item for item in throughJson["phases"]["decode"]
                     ["measurePoints"]["points"]
                     if item["context"] == 1024
                     and item["batch"] == 1
                     and int(item.get("draftTokens", 0)) == 0)
        gpu = cards[0]
        started = time.monotonic()
        stored = servingMeasurePoint(
            throughJson, card=Card.of(gpu), context=1024, batch=1)
        elapsed = time.monotonic() - started
        matrix = servingMeasureMatrix(
            throughJson, card=Card.of(gpu), points=[(1024, 1, 0)])
        assert matrix.point_rows[(1024, 1, 0)] == 0
        direct = _direct_measure(model, gpu, parallel, point["batch"])
        assert (PointMeasure.from_json(stored["measure"])
                == PointMeasure.from_json(
                    without_anomaly_numbers(direct)["measure"])), model
        assert stored["terms"] == []
        assert elapsed < 0.05, (model, elapsed)

        if not record.get("speculative"):
            continue
        point = next(item for item in table["points"]
                     if item["context"] == 1024 and item["batch"] == 1
                     and int(item["draftTokens"]) == 1)
        stored = servingMeasurePoint(
            throughJson, card=Card.of(gpu), context=1024, batch=1,
            draft_tokens=1)
        direct = _direct_measure(model, gpu, parallel, point["batch"],
                                 draft_tokens=1)
        assert (PointMeasure.from_json(stored["measure"])
                == PointMeasure.from_json(
                    without_anomaly_numbers(direct)["measure"])), model


def test_offline_serving_projection_never_asks_point_keys(monkeypatch):
    """Drawing SOL from a v6 record is offline: it reads no measurement key."""
    import backend.api.gallery_frontier as gallery
    import backend.inference.calibration_overlay as overlay
    import backend.inference.sol_law as law

    snapshot_config(MODEL)
    record = _layout_law((MODEL, ["H100-SXM"], 0.9,
                          _GALLERY_PRECISION.canonical(),
                          PARALLEL.to_dict()))
    assert record.get("schema") == "serving-law-v7", record
    for module, name in ((law, "point_keys"), (overlay, "point_keys")):
        monkeypatch.setattr(
            module, name,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("point_keys called")))
    out = gallery._restamp_layout(
        json.loads(json.dumps(record)), "H100-SXM", PARALLEL)
    assert out["theory"]


MTP_MODEL = "XiaomiMiMo/MiMo-7B-RL"
MTP_PARALLEL = ParallelConfig(tp_attention=2, tp_dense_ffn=2)


def _gemmMissing(key):
    return "missing" if key.signature == "gemm" else "measured"


@pytest.mark.parametrize("draftTokens,speculative",
                         [(3, True), (0, True), (0, False)])
def test_pool_partial_is_a_direct_walk_s_at_every_k(draftTokens, speculative):
    """The partial track by two roads at one point: the pool's (the record's
    stored lookups at K, folded, then the measured terms priced off the
    step's per-term floor form, `FloorForm.terms_at`, and blended,
    `auto_sweep.blend`) and a direct estimate's (its own walk at K, the
    measure's per-term SOL, `MatrixFold.partial`). Only gemm is missing, so
    the blend is genuinely mixed. At K = 0 the record either carries a draft
    (the Gallery's plain view of an MTP model: the law's terms run past the
    plain step's, weighted 0) or does not."""
    gpu, context, batch = "H100-SXM", 1024, 8
    snapshot_config(MTP_MODEL)
    spec = {} if not speculative else {
        "speculative_draft_tokens": 1, "speculative_draft": "recursive",
        "acceptance_rate": 0.7}
    base = InferenceEstimateRequest(
        model_id=MTP_MODEL, gpu_sku=gpu, avg_input_tokens=context,
        avg_output_tokens=256, parallel=MTP_PARALLEL.to_dict(),
        framework="vllm", selection_mode="theoretical_only")
    record = servingLawRecord(
        resolve_estimate_environment(base.model_copy(update=spec)),
        gpu_sku=gpu, anchorContext=context, outputTokens=256,
        parallel=MTP_PARALLEL, capacityCards=(gpu,),
        capacityContexts=(context,), measureContexts=(context,),
        measureBatchGrid=[{"decode_batch_size": batch}],
        **({"draft_forwards": draftForwardsLaw} if speculative else {}))
    pool = servingLawPointView(servingLawPoint(
        record, card=Card.of(gpu), context=context, outputTokens=256,
        decodeBatch=batch, prefillBatch=1, verifyWidth=draftTokens + 1,
        draftTokens=draftTokens, acceptanceRate=0.7,
        comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT))
    poolMeasure = servingMeasurePoint(
        record, card=Card.of(gpu), context=context, batch=batch,
        draft_tokens=draftTokens)
    pending = estimate_pending(**resolve_estimate_environment(
        base.model_copy(update={
            "decode_batch_size": batch,
            **({"speculative_draft_tokens": draftTokens,
                "speculative_draft": "recursive",
                "acceptance_rate": 0.7} if draftTokens else {})})))
    direct = pending.finish()
    poolItem = {"sol": copy.deepcopy(pool), "measures": {"decode": poolMeasure}}
    directItem = {"sol": servingLawPointView(direct),
                  "measures": {"decode": pending.measures(direct)["decode"]}}
    with FixtureBench(_gemmMissing) as bench:
        measure_points([poolItem, directItem], selection_mode="envelope",
                       measurement_scope=bench.scope, blend_terms=True)
    poolDecode = poolItem["sol"]["phases"]["decode"]
    directDecode = directItem["sol"]["phases"]["decode"]
    assert poolDecode["t_partial_s"] is None  # the pool blends at the reader
    assert poolDecode["measured_terms"] == directDecode["measured_terms"]
    assert poolDecode["measured_part_s"] == pytest.approx(
        directDecode["measured_part_s"], rel=1e-12)

    graph = LawGraph.from_json(record["phases"]["decode"]["law"])
    floorCard = FloorCard.of(record, Card.of(gpu),
                             comm_latency_us=COMM_LATENCY_US_DEFAULT,
                             launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    point = {**cardBinding(record, Card.of(gpu),
                           comm_latency_us=COMM_LATENCY_US_DEFAULT,
                           launch_floor_us=LAUNCH_FLOOR_US_DEFAULT),
             "B": batch, "s_input": context, "s_output": 256, "s_prefix": 0,
             "s_verify": draftTokens + 1, "rho": 1}
    form = graph.floorForm(point, floorCard)
    assert form.at(*floorCard.floors_s) == pytest.approx(
        directDecode["t_sol_s"], rel=1e-9)
    blended = blend(
        form.at(*floorCard.floors_s),
        form.terms_at(*floorCard.floors_s, poolDecode["measured_terms"]),
        poolDecode["measured_part_s"])
    assert 0 < directDecode["partial_measured_share"] < 1
    assert blended["t_s"] == pytest.approx(directDecode["t_partial_s"], rel=1e-9)
    assert blended["measured_share"] == pytest.approx(
        directDecode["partial_measured_share"], rel=1e-9)
