"""The canonical SOL layer keeps laws, not expanded page points."""
from __future__ import annotations

import json

import pytest

from backend.api._models import InferenceEstimateRequest
from backend.api.estimate_env import resolve_estimate_environment
from backend.arch.eval.precision import AS_SHIPPED
from backend.inference.capacity import (
    CapacityKnobs,
    _capacity_pending,
    _capacity_preparation,
    finish_estimates,
)
from backend.inference.card import Card
from backend.inference.defaults import (
    COMM_LATENCY_US_DEFAULT,
    LAUNCH_FLOOR_US_DEFAULT,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.row_roofline import FloorForm
from backend.inference.sol_law import (
    FloorCard,
    LawGraph,
    cardBinding,
    draftForwardsLaw,
    servingLawPoints,
    servingLawPoint,
    servingLawRecord,
)
from tests.snapshots import snapshot_config


MODEL = "meta-llama/Llama-3.2-3B"
DRAFT_MODEL = "deepseek-ai/DeepSeek-V3"
NESTED_EXTREMUM_MODEL = "Qwen/Qwen3.8-Flash-Next"


def _environment(gpu: str = "H100-SXM"):
    snapshot_config(MODEL)
    request = InferenceEstimateRequest(
        model_id=MODEL,
        gpu_sku=gpu,
        parallel=ParallelConfig(
            tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2).to_dict(),
        precision={"canonical": AS_SHIPPED.canonical()},
        avg_input_tokens=4096,
        avg_output_tokens=256,
        selection_mode="envelope",
    )
    return resolve_estimate_environment(request)


def _direct(environment, *, context: int, batch: int):
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku=environment["gpu_sku"],
        knobs=knobs)
    pending = _capacity_pending(
        preparation, avg_input_tokens=context, avg_output_tokens=256,
        prefill_batch_override=1, decode_batch_override=batch,
        breakdown=False, fold_prefill_chunks=False)
    return finish_estimates([pending])[0]


def test_one_law_substitutes_every_point_without_breakdown():
    environment = _environment()
    parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                              tp_moe_expert=2)
    record = servingLawRecord(
        environment, gpu_sku="H100-SXM", anchorContext=4096,
        outputTokens=256, parallel=parallel,
        capacityCards=("H100-SXM", "H200"), capacityContexts=(8192,))
    encoded = json.dumps(record, separators=(",", ":"))
    assert len(encoded) < 150_000
    assert "per_op_timing" not in record
    assert "flops_by_module" not in record
    assert "activation_peak" not in record
    assert "kv_cache" not in record

    graph = LawGraph.from_json(record["phases"]["decode"]["law"])
    binding = cardBinding(
        record, Card.of("H100-SXM"), comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    point = {**binding, "B": 8, "s_input": 8192, "s_output": 256,
             "s_prefix": 0, "s_verify": 1, "rho": 1}
    answer = graph.evaluate(point)
    direct = _direct(environment, context=8192, batch=8)
    assert answer["t_sol_s"] == pytest.approx(
        direct["phases"]["decode"]["t_sol_s"], rel=1e-12, abs=0.0)

    h200 = _environment("H200")
    onH200 = graph.evaluate({
        **cardBinding(record, Card.of("H200"),
                      comm_latency_us=COMM_LATENCY_US_DEFAULT,
                      launch_floor_us=LAUNCH_FLOOR_US_DEFAULT),
        "B": 8, "s_input": 8192, "s_output": 256, "s_prefix": 0,
        "s_verify": 1, "rho": 1,
    })
    assert onH200["t_sol_s"] == pytest.approx(
        _direct(h200, context=8192, batch=8)["phases"]["decode"]["t_sol_s"],
        rel=1e-12, abs=0.0)

    served = servingLawPoint(
        record, card=Card.of("H200"), context=8192, outputTokens=256,
        decodeBatch=8, comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    assert served["max_decode_batch_per_gpu"] == (
        _direct(h200, context=8192, batch=8)["max_decode_batch_per_gpu"])
    assert served["phases"]["decode"]["throughput"]["sol"]["per_gpu"] == (
        _direct(h200, context=8192, batch=8)["phases"]["decode"]["throughput"]["sol"]["per_gpu"])


def test_a_point_grid_is_the_same_point_projection_read_once():
    """The grid shares the card, capacity and prefill read, not a second law."""
    environment = _environment()
    parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                              tp_moe_expert=2)
    record = servingLawRecord(
        environment, gpu_sku="H100-SXM", anchorContext=4096,
        outputTokens=256, parallel=parallel,
        capacityCards=("H100-SXM", "H200"), capacityContexts=(8192,))
    grid = servingLawPoints(
        record, card=Card.of("H200"), contexts=(8192,),
        decodeBatches=(1, 8, None), outputTokens=256,
        comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    oneAtATime = [
        servingLawPoint(
            record, card=Card.of("H200"), context=8192, outputTokens=256,
            decodeBatch=batch, comm_latency_us=COMM_LATENCY_US_DEFAULT,
            launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
        for batch in (1, 8, None)
    ]
    assert grid == [oneAtATime]


def test_a_decode_floor_form_is_the_law_at_every_floor():
    """A projected decode step's floor form (`floors`) substituted at any
    floors is the law evaluated with those floors bound into the card
    (`cardBinding`) — a second road through the graph, not through the
    form. Exercises the link-latency arms of a TP2 all-reduce ladder and
    the launch arms, across batches and verify widths."""
    environment = _environment()
    parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                              tp_moe_expert=2)
    record = servingLawRecord(
        environment, gpu_sku="H100-SXM", anchorContext=4096,
        outputTokens=256, parallel=parallel,
        capacityCards=("H100-SXM",), capacityContexts=(8192,))
    card = Card.of("H100-SXM")
    law = LawGraph.from_json(record["phases"]["decode"]["law"])
    [row] = servingLawPoints(
        record, card=card, contexts=(8192,), decodeBatches=(1, 64),
        outputTokens=256, verifyWidth=3,
        comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT, floorForms=True)
    for point, batch in zip(row, (1, 64), strict=True):
        form = FloorForm.from_json(point["phases"]["decode"]["floors"])
        assert form.terms
        for comm_us, launch_us in ((0.0, 0.0), (2.3, 3.7), (40.0, 0.1),
                                   (0.5, 25.0)):
            direct = float(law.evaluate({
                **cardBinding(record, card, comm_latency_us=comm_us,
                              launch_floor_us=launch_us),
                "B": batch, "s_input": 8192, "s_output": 256,
                "s_prefix": 0, "s_verify": 3, "rho": 1})["t_sol_s"])
            assert form.at(comm_us * 1e-6, launch_us * 1e-6) == pytest.approx(
                direct, rel=1e-12)
    # Without the flag a point publishes the time alone.
    [[plain, _]] = servingLawPoints(
        record, card=card, contexts=(8192,), decodeBatches=(1, 64),
        outputTokens=256, comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    assert "floors" not in plain["phases"]["decode"]


def test_a_batched_floor_form_is_the_point_one():
    """The whole-grid form (`floorForm_batch`, the numpy program) is the
    one-point form (`floorForm`, the scalar program) at every point — a
    second compile of the same graph, read by the same reader. TP2 puts a
    collective's latency ladder (`Piecewise`) and `Max` arms in the law,
    the constructs a numpy program must print element-wise. A grid whose
    first batch is unserved checks each point gets its own form back."""
    environment = _environment()
    record = servingLawRecord(
        environment, gpu_sku="H100-SXM", anchorContext=4096,
        outputTokens=256,
        parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                                tp_moe_expert=2),
        capacityCards=("H100-SXM",), capacityContexts=(8192,))
    card = Card.of("H100-SXM")
    floors = dict(comm_latency_us=COMM_LATENCY_US_DEFAULT,
                  launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    law = LawGraph.from_json(record["phases"]["decode"]["law"])
    floorCard = FloorCard.of(record, card, **floors)
    points = [{**cardBinding(record, card, **floors), "B": batch,
               "s_input": context, "s_output": 256, "s_prefix": 0,
               "s_verify": verify, "rho": 1}
              for batch in (1, 64, 4096) for context in (1024, 8192)
              for verify in (1, 3)]
    for point, batched in zip(
            points, law.floorForm_batch(points, floorCard), strict=True):
        scalar = law.floorForm(point, floorCard)
        assert (batched.index, batched.width) == (scalar.index, scalar.width)
        assert batched.constant == pytest.approx(scalar.constant, rel=1e-12)
        for ours, theirs in zip(batched.terms, scalar.terms, strict=True):
            assert ours == pytest.approx(theirs, rel=1e-12)

    [row] = servingLawPoints(
        record, card=card, contexts=(8192,), decodeBatches=(10**9, 1, 8),
        outputTokens=256, servedOnly=True, **floors)
    assert row[0] is None
    for point, batch in zip(row[1:], (1, 8), strict=True):
        assert point == servingLawPoint(
            record, card=card, context=8192, outputTokens=256,
            decodeBatch=batch, **floors)


def test_a_law_binds_no_card_of_another_walk_class():
    """A law's terms are the rows of one walk, and which rows exist is a fact
    of the card's walk class (`Card.klass`): a record names the card it was
    walked on, and a card of another class can neither bind into its laws nor
    carry capacity facts in it — it walks its own record. Without the gate
    the symbols bind and the number looks right: DeepSeek-V4-Flash's H100
    decode priced off a B300 walk carried four `quantise_activation` rows
    H100's own walk does not have (260 rows against 256, +2.0% t_sol)."""
    environment = _environment()
    parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                              tp_moe_expert=2)
    with pytest.raises(ValueError, match="card class"):
        servingLawRecord(
            environment, gpu_sku="H100-SXM", anchorContext=4096,
            outputTokens=256, parallel=parallel,
            capacityCards=("H100-SXM", "B200"), capacityContexts=(8192,))
    record = servingLawRecord(
        environment, gpu_sku="H100-SXM", anchorContext=4096,
        outputTokens=256, parallel=parallel,
        capacityCards=("H100-SXM", "H200"), capacityContexts=(8192,))
    assert record["walked_on"] == "H100-SXM"
    # B200 (compute capability 10.0) is refused even with a capacity fact
    # forged for it: the gate is the binding, not the capacity lookup.
    forged = {**record, "capacity": [
        *record["capacity"], {**record["capacity"][0], "gpu_sku": "B200"}]}
    with pytest.raises(ValueError, match="card class"):
        cardBinding(forged, Card.of("B200"),
                    comm_latency_us=COMM_LATENCY_US_DEFAULT,
                    launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    with pytest.raises(ValueError, match="card class"):
        servingLawPoint(forged, card=Card.of("B200"), context=8192,
                        outputTokens=256, decodeBatch=8,
                        comm_latency_us=COMM_LATENCY_US_DEFAULT,
                        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    # A record that does not say what it was walked on binds no card.
    unnamed = {k: v for k, v in record.items() if k != "walked_on"}
    with pytest.raises(KeyError):
        cardBinding(unnamed, Card.of("H100-SXM"),
                    comm_latency_us=COMM_LATENCY_US_DEFAULT,
                    launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)


def test_cse_keeps_an_as_written_extremum_over_cases():
    """A draft's roofline holds `Max` over `Cases`; CSE must not ask SymPy to
    order those arms while no workload value is bound."""
    snapshot_config(DRAFT_MODEL)
    request = InferenceEstimateRequest(
        model_id=DRAFT_MODEL,
        gpu_sku="RTX-4090",
        parallel=ParallelConfig(
            tp_attention=32, tp_dense_ffn=32, tp_moe_expert=32).to_dict(),
        precision={"canonical": AS_SHIPPED.canonical()},
        avg_input_tokens=1024,
        avg_output_tokens=256,
        speculative_draft="recursive",
        speculative_draft_tokens=2,
        acceptance_rate=0.8,
        selection_mode="theoretical_only",
    )
    environment = resolve_estimate_environment(request)
    parallel = ParallelConfig.from_dict(request.parallel)
    record = servingLawRecord(
        environment, gpu_sku="RTX-4090", anchorContext=1024,
        outputTokens=256, parallel=parallel,
        capacityCards=("RTX-4090",), capacityContexts=(1024,),
        draft_forwards=draftForwardsLaw)
    law = record["phases"]["decode"]["law"]
    assert law["definitions"]
    assert "program" not in law
    encoded = json.dumps(record, separators=(",", ":"))
    assert "law_inert_extremum" not in encoded
    with pytest.raises(ValueError, match="law text only"):
        LawGraph.from_json({**law, "program": "def forbidden(): pass"})

    graph = LawGraph.from_json(law)
    binding = cardBinding(
        record, Card.of("RTX-4090"), comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    answer = graph.evaluate({**binding, "B": 1, "s_input": 1024,
                             "s_output": 256, "s_prefix": 0, "s_verify": 2,
                             "rho": 0.8})
    assert 0 < answer["t_sol_s"] < 1
    for batch in (1, None):
        expected = servingLawPoint(
                record, card=Card.of("RTX-4090"), context=1024,
                outputTokens=256, decodeBatch=batch, prefillBatch=1,
                verifyWidth=2, draftTokens=1, acceptanceRate=.8,
                comm_latency_us=COMM_LATENCY_US_DEFAULT,
                launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
        other = servingLawPoint(
                record, card=Card.of("RTX-4090"), context=1024,
                outputTokens=256, decodeBatch=batch, prefillBatch=1,
                verifyWidth=2, draftTokens=1, acceptanceRate=.6,
                comm_latency_us=COMM_LATENCY_US_DEFAULT,
                launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
        assert (other["phases"]["decode"]["t_sol_s"]
                == expected["phases"]["decode"]["t_sol_s"])
        assert (other["phases"]["decode"]["throughput"]["sol"]["per_gpu"]
                != expected["phases"]["decode"]["throughput"]["sol"]["per_gpu"])


def test_draft_rows_enter_the_collective_sidecar(monkeypatch):
    """A draft term's collectives are facts of the phase it was summed into."""
    import types
    import backend.inference.sol_law as sol_law

    main = {"main": True}
    draft = {"draft": True}
    seen = []

    class Step:
        terms = ()

        def law(self, _name):
            return 0

        def term_laws(self, _name):
            return []

    monkeypatch.setattr(
        sol_law.StepSum, "of",
        lambda *rows, **_kwargs: Step())
    monkeypatch.setattr(
        sol_law.LawGraph, "of",
        lambda *_exprs: types.SimpleNamespace(to_json=dict))
    monkeypatch.setattr(
        sol_law, "_collectives",
        lambda rows: seen.append(tuple(rows)) or ())

    phase = sol_law._phase([main], draft_rows=[draft])
    assert phase["collectives"] == []
    assert seen == [(main, draft)]


def test_lifting_substitutes_card_symbols_without_ordering_extremum_arms():
    """A prefix law can hold nested `Max`/`Min` arms over `Cases`; replacing
    only its card symbols must not ask SymPy to order those arms."""
    snapshot_config(NESTED_EXTREMUM_MODEL)
    request = InferenceEstimateRequest(
        model_id=NESTED_EXTREMUM_MODEL,
        gpu_sku="H100-SXM",
        parallel=ParallelConfig(
            tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4).to_dict(),
        precision={"canonical": AS_SHIPPED.canonical()},
        avg_input_tokens=1024,
        avg_output_tokens=256,
        selection_mode="theoretical_only",
    )
    record = servingLawRecord(
        resolve_estimate_environment(request), gpu_sku="H100-SXM",
        anchorContext=1024, outputTokens=256,
        parallel=ParallelConfig(
            tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4),
        capacityCards=("H100-SXM",), capacityContexts=(1024,))
    graph = LawGraph.from_json(record["phases"]["prefill"]["law"])
    binding = cardBinding(
        record, Card.of("H100-SXM"), comm_latency_us=COMM_LATENCY_US_DEFAULT,
        launch_floor_us=LAUNCH_FLOOR_US_DEFAULT)
    answer = graph.evaluate({**binding, "B": 1, "s_input": 1024,
                             "s_output": 256, "s_prefix": 0, "s_verify": 1,
                             "rho": 1})
    assert 0 < answer["t_sol_s"] < 1
