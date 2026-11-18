"""Sweep parity (docs/auto_sweep.md §3.5): a swept point is the estimate
a direct call at that point returns.

The harness is `tests/sweep_parity.py`. What holds today, and what each
later step makes hold:

  * the parallel sweep's phase numbers on every card equal a direct estimate
    on that card at that card's batch — on the cheap card pair (`CARDS`) and,
    for a model whose fused graph depends on the card, across the whole
    Scaling series with the per-op row set compared too
    (`test_a_mixed_series_card_is_the_direct_walk_on_that_card`);
  * per-op rows walked at one batch and read at another batch and card
    through their laws equal the rows a walk there emits, on the columns the
    laws state (roofline and shape);
  * the Context x Batch sweep walks once and its (context, batch) points,
    priced by substitution, equal direct estimates (step 4);
  * the Kernel Bench stand-in answers each key by policy and counts lookups;
  * every point of the parallel sweep reads the measured times of its own
    keys, all of the layout's points in one lookup (step 2).

The checks a later step turns green are strict xfails naming that step: a
fix flips them to failures, and the step removes the mark.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.arch.eval.analysis import Point
from backend.inference.card import Card
from backend.inference.parallel_config import ParallelConfig
from backend.inference.parallel_sweep import DEFAULT_SWEEP_GPUS
from backend.inference.phases import Speculation, step_tokens
from tests.sweep_parity import (AMBIGUOUS, MISSING, ROW_LISTS, Deployment, FixtureBench,
                                diff, law_view, measured_rows, phase_scalars, report,
                                response_cell, rows_at, without_breakdown)

LLAMA = "meta-llama/Llama-3.2-3B"
QWEN_MOE = "Qwen/Qwen3-30B-A3B"
GPT_OSS = "openai/gpt-oss-20b"   # sliding-window attention
#: A model whose FUSED GRAPH depends on the card: vLLM gates
#: `fused_kda_decode` at `min_compute_capability: "9.0"`, so its decode walk
#: on an Ada card carries the three KDA ops the Hopper and Blackwell walks
#: have fused into one. The only model here that can tell a walk of the
#: wrong card class from the right one — see `_row_shape`.
KIMI_K3 = "moonshotai/Kimi-K3"
CTX = 4096
#: The cheap card pair the parity checks run on: two cards, deliberately of
#: DIFFERENT walk classes (compute capability 9.0 and 10.0), so a point read
#: on one is never read off the other's walk.
CARDS = ("H100-SXM", "B200")
#: The cards a point or a row is READ on: the deployment's own card,
#: another card of ITS class (H100-SXM and H200 are one class — the
#: re-pricing a sweep actually pays for) and a card of another class
#: (which a sweep walks for itself, `_class_walks`). Both legs of the
#: reuse rule, in the order a sweep pays them.
POINT_CARDS = ("H100-SXM", "H200", "B200")
#: The Scaling panel's whole series — four walk classes
#: (`framework_bindings.card_class`). `CARDS` is two of them; the mixed-series
#: oracle below runs all of it, which is the sweep's real shape.
SERIES = tuple(DEFAULT_SWEEP_GPUS)

TP2 = ParallelConfig(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2)
EP8 = ParallelConfig(dp_attention=8, ep_moe=8)
PP2 = ParallelConfig(pp=2)


def _parallel_cell(dep: Deployment, cards=CARDS, context: int = CTX) -> dict:
    """One layout of the parallel sweep, every card of `cards`, its points
    measured from one lookup: `{card: estimate}`
    (`parallel_sweep._estimate_layout`, `batch_sweep.measure_points`)."""
    from backend.inference import parallel_sweep as ps
    env = dep.env()
    payload = {k: v for k, v in env.items() if k not in ps.PER_POINT_FIELDS}
    payload.update(gpu_skus=list(cards), num_gpus=env["parallel"].cluster_size,
                   parallel=env["parallel"], avg_input_tokens=context)
    from backend.inference.batch_sweep import measure_points
    from tests.sweep_parity import layout_at
    out = layout_at(payload)
    measure_points([pt for pt in out.values() if "sol" in pt],
                   selection_mode=env["selection_mode"],
                   measurement_scope=env["measurement_scope"])
    return {card: pt["sol"] for card, pt in out.items()}


def _direct_at_point(dep: Deployment, card: str, swept: dict) -> dict:
    """The direct estimate on `card` at the batches the swept point ran."""
    phases = swept["phases"]
    return dep.on(card).direct(avg_input_tokens=CTX,
                               prefill_batch=phases["prefill"]["batch"],
                               decode_batch=phases["decode"]["batch"])


def _phase_diffs(swept: dict, direct: dict, view) -> list:
    out = []
    for ph in ("prefill", "decode"):
        out += diff(view(swept["phases"][ph]), view(direct["phases"][ph]),
                    path=f"phases.{ph}")
    return out


_LAYOUTS = [
    pytest.param(Deployment(LLAMA, parallel=TP2), id="llama-tp2"),
    pytest.param(Deployment(QWEN_MOE, parallel=EP8), id="qwen3-moe-dp8ep8",
                 marks=pytest.mark.regression),
    pytest.param(Deployment(LLAMA, parallel=PP2), id="llama-pp2",
                 marks=pytest.mark.regression),
]


@pytest.mark.parametrize("dep", _LAYOUTS)
def test_parallel_sweep_phase_numbers_equal_direct(dep):
    """Walk once per card class, restamp per card, rescale to the card's
    batch: every phase number equals a direct estimate's there."""
    swept = _parallel_cell(dep)
    for card in CARDS:
        direct = _direct_at_point(dep, card, swept[card])
        diffs = _phase_diffs(swept[card], direct, phase_scalars)
        for key in ("max_batch_per_gpu", "max_prefill_batch_per_gpu",
                    "infeasibility_reason", "max_context_tokens", "beyond_max_context"):
            diffs += diff(swept[card][key], direct[key], path=key)
        assert not diffs, f"{card}:\n{report(diffs)}"


#: What a point publishes about the deployment rather than about its step:
#: its capacity verdict, the batch it admits, and the model's own context.
#: An infeasible card has only these.
_CARD_VERDICT = ("infeasibility_reason", "max_batch_per_gpu",
                 "max_prefill_batch_per_gpu", "max_decode_batch_per_gpu",
                 "max_context_tokens", "beyond_max_context")


def _row_shape(rows) -> list:
    """Each per-op row's identity, in order: which op ran, where in the
    graph, how many instances the step runs, and the format it ran in.

    This is the part a card's CLASS decides, and the reason the oracle below
    cannot be a comparison of step times. Fusion happens in the walk, so a
    kernel a framework runs only above some compute capability ABSORBS the
    rows it replaces: Kimi-K3's decode step is 1092 rows on an Ada card and
    1012 on Hopper and Blackwell, because vLLM gates `fused_kda_decode` at
    9.0. A card priced off another class's walk therefore publishes a
    different row LIST, and comparing the list in ORDER also catches a
    regrouping that keeps the count.

    DO NOT simplify this back to a number comparison. Measured by reverting
    `_estimate_layout` to one anchor for the whole series, so that every
    card was priced off the Ada walk:

      * Kimi-K3 (here)                             RED — rows 1092 vs 1012
      * Qwen3-30B-A3B, the second case here        GREEN
      * Llama-3.2-3B, `test_parallel_sweep.py::
        test_two_gpus_match_per_sku_singles`       GREEN

    Those last two deploy the same graph on every class, so no number of
    theirs moves and no amount of numeric tolerance would have caught the
    bug. A capability-split model compared on its ROW SET is what catches
    it, which is why Kimi-K3 is the first case below.

    The format is part of the identity because a card decides it too: the
    launches a quantised step runs are chosen per card
    (`serving.build_serving_layers`, by compute capability), so
    DeepSeek-V4-Flash's routed activation is FP8 on a B200 and BF16 on an
    H100-SXM walk — the same op, at the same path, a different row. A
    length comparison misses it (that one moved 270 rows to 274, but a
    regrouping need not) and `t_sol` misses it twice over.
    """
    return [(r.get("op"), r.get("display_path"), r.get("count_per_step"),
             r.get("act_dtype")) for r in rows or ()]


#: Mixed-series cases, priced across the whole Scaling series. Kimi-K3 is
#: FIRST because it is the one that turns red when a card is priced off
#: another class's walk: its fused decode graph differs between Ada and
#: everything above it. Qwen3-MoE is second as the breadth case — it serves
#: on more of the series, so it compares more cards' verdicts — but it
#: deploys one graph on every class and so discriminates nothing on its own
#: (`_row_shape`). Keep that order: the default suite and the deploy gate
#: see only the first.
_MIXED_SERIES = [
    pytest.param(Deployment(KIMI_K3, parallel=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, ep_moe=8)), id="kimi-k3-tp8ep8"),
    pytest.param(Deployment(QWEN_MOE, parallel=EP8), id="qwen3-moe-dp8ep8"),
]


# No `regression` mark: the comment above says the default suite and the
# deploy gate see the first case, and a mark here would have made that false.
# Others catch the MECHANISM going wrong — `test_a_layout_walks_once_per_card_class`
# counts the walks, and the two refusal tests catch a cross-class re-price being
# attempted. This is the only one that catches a wrong walk PUBLISHED as a card's
# own numbers: a swept card's rows and verdicts against a fresh direct walk on
# that card. A check nothing runs cannot be told from one that passes, so this
# one runs; `conftest` deselects the second case, so the gate pays for one cell.
@pytest.mark.parametrize("dep", _MIXED_SERIES)
def test_a_mixed_series_card_is_the_direct_walk_on_that_card(dep):
    """Every card of the Scaling series, priced in ONE sweep cell, is what a
    fresh direct walk on that card gives — feasible or infeasible.

    The sweep walks once per card CLASS and restamps within it
    (`parallel_sweep._estimate_layout`), so this is the check that the class
    is the right unit of reuse. It compares the per-op ROW SET, not only the
    step time, because the row set is what a class changes (`_row_shape`):
    reverting `_estimate_layout` to one anchor for the series turns the
    Kimi-K3 case red on `per_op_timing_decode.rows` — the Ada walk's 1092
    decode rows against B300's own 1012 — where every number-only check
    stays green on a capability-blind model.

    A card that admits no request runs no step (`capacity.without_step`), so
    it is compared on the verdict it publishes (`_CARD_VERDICT`) and on
    publishing no rows — from that card's own walk either way.
    """
    swept = _parallel_cell(dep, cards=SERIES)
    assert sorted(swept) == sorted(SERIES)
    priced = 0
    for card in SERIES:
        point = swept[card]
        direct = _direct_at_point(dep, card, point)
        diffs = [d for key in _CARD_VERDICT
                 for d in diff(point.get(key), direct.get(key), path=key)]
        if point.get("infeasibility_reason"):
            # No step to compare: this card publishes none, and the rows of
            # the walk it was priced from are not its answer.
            diffs += [f"{lst}: published by a card that admits no request"
                      for lst in ROW_LISTS if point.get(lst)]
        else:
            priced += 1
            diffs += _phase_diffs(point, direct, phase_scalars)
            for lst in ROW_LISTS:
                diffs += diff(_row_shape(point.get(lst)),
                              _row_shape(direct.get(lst)), path=f"{lst}.rows")
        assert not diffs, f"{card}:\n{report(diffs)}"
    # A series on which nothing was priced would pass every assertion above
    # without comparing one row.
    assert priced, f"no card of {SERIES} served {dep.model_id}"


#: A row's measurement identity. A row re-read at another batch does not
#: carry it: its key at that point is `point_keys`' answer, which the sweep
#: pipeline states on the re-read row (`batch_sweep`'s docstring).
IDENTITY = ("shape_json", "dtype_env_json", "signature")


def _without_identity(x):
    """`x` with every measurement-identity field removed, at any depth: what
    `rows_at` alone gives (a re-read row states no identity; a sweep point's
    rows are keyed by `point_keys` in `capacity._point_rows`)."""
    if isinstance(x, dict):
        return {k: _without_identity(v) for k, v in x.items() if k not in IDENTITY}
    if isinstance(x, list):
        return [_without_identity(v) for v in x]
    return x


def test_parallel_sweep_phase_display_equals_direct():
    """The whole phase at the card's batch — every field a point read with no
    breakdown shows (`without_breakdown`) — is the direct estimate's
    there."""
    dep = Deployment(LLAMA, parallel=TP2)
    swept = _parallel_cell(dep, cards=("B200",))
    direct = _direct_at_point(dep, "B200", swept["B200"])
    diffs = _phase_diffs(swept["B200"], without_breakdown(direct, keyed=True), lambda ph: ph)
    assert not diffs, report(diffs)


_ROW_CASES = [
    pytest.param(Deployment(LLAMA), id="llama-tp1"),
    pytest.param(Deployment(QWEN_MOE, parallel=EP8), id="qwen3-moe-dp8ep8",
                 marks=pytest.mark.regression),
    pytest.param(Deployment(LLAMA, parallel=PP2), id="llama-pp2",
                 marks=pytest.mark.regression),
    pytest.param(Deployment(GPT_OSS), id="gpt-oss-swa",
                 marks=pytest.mark.regression),
]


@pytest.mark.parametrize("dep", _ROW_CASES)
def test_rows_read_at_a_point_equal_the_walk_there(dep):
    """Rows walked at B=1, read at B and on another card through their laws
    (`rows_at`), equal a walk at B on that card on every column the laws
    state.

    Off the walk a sweep would price each card from (`_class_walks`): the
    card's own class's. The laws are card-parametric, but which rows there
    ARE is not — a card of another class fuses the graph differently — so
    reading a point off another class's walk compares a row to whichever
    row happens to sit at its index."""
    at = _class_walks(dep, POINT_CARDS)
    for card in POINT_CARDS:
        anchor = at[card][0]
        for batch in (3, 64):
            walked = dep.on(card).direct(avg_input_tokens=CTX, prefill_batch=batch,
                                         decode_batch=batch)
            for lst in ROW_LISTS:
                if lst not in anchor:
                    continue
                diffs = diff(law_view(rows_at(anchor[lst], batch, card=card,
                                              breakdown=True)),
                             law_view(walked[lst]), path=lst)
                assert not diffs, f"{card} B={batch}:\n{report(diffs)}"


def test_rows_read_at_a_point_are_the_walk_there_whole():
    """Every field of a row read at another batch — formula text, wire dims,
    node dims, per-tensor byte splits — is the walk's there (identity
    aside, `_without_identity`)."""
    dep = Deployment(LLAMA)
    anchor = dep.direct(avg_input_tokens=CTX, prefill_batch=1, decode_batch=1)
    walked = dep.direct(avg_input_tokens=CTX, prefill_batch=64, decode_batch=64)
    read = rows_at(anchor["per_op_timing_decode"], 64, card=dep.gpu_sku,
                   breakdown=True)
    assert not any(k in r for r in read for k in IDENTITY), (
        "a row read at another batch carries no measurement identity")
    diffs = diff(read, _without_identity(walked["per_op_timing_decode"]))
    assert not diffs, report(diffs)


def _class_walks(dep: Deployment, cards) -> dict:
    """Per card of `cards`, the walk a sweep prices it off: that walk's
    anchor estimate at B=1, and the preparation the point is read on.

    `dep`'s own card's walk when the engine lets a prepared walk be
    re-priced on the card — one card class — else a walk on the card
    itself, which is what a sweep does for a card of another class
    (`parallel_sweep._estimate_layout`: one walk per class, and so one
    anchor per class here).

    The rule is ASKED of the engine (`capacity._restamp_gpu`'s refusal) and
    never restated here. That is the point: the engine's refusal is the
    only statement of when a walk may be re-priced elsewhere, so reverting
    it prices B200 off the H100-SXM walk again — which is the row diff
    `test_a_point_is_the_walk_there_whole` reports."""
    from backend.inference import capacity as cap
    env = dep.env()
    knobs = cap.CapacityKnobs.from_kwargs(
        {k: v for k, v in env.items()
         if k not in ("config", "gpu_sku", "avg_output_tokens")})
    own = cap._capacity_preparation(config=env["config"], gpu_sku=dep.gpu_sku,
                                    knobs=knobs)
    anchors: dict = {}
    out = {}
    for card in cards:
        try:
            walk_card, p = dep.gpu_sku, (
                own if card == dep.gpu_sku else cap._restamp_gpu(
                    own, Card.of(card)))
        except ValueError:
            walk_card, p = card, cap._capacity_preparation(
                config=env["config"], gpu_sku=card, knobs=knobs)
        if walk_card not in anchors:
            anchors[walk_card] = dep.on(walk_card).direct(
                avg_input_tokens=CTX, prefill_batch=1, decode_batch=1)
        out[card] = (anchors[walk_card], p)
    return out


def _walk_card(dep: Deployment, card: str) -> str:
    """The card whose class `card` reads: the deployment's when the two walk
    alike, else `card` itself (`capacity._class_walks` states the rule)."""
    from backend.inference.card import Card

    own = Card.of(dep.gpu_sku)
    return own.sku if own.walks_alike(Card.of(card)) else card


def _stamped_at(dep: Deployment, anchor: dict, p, card: str, batch: int, *,
                breakdown: bool) -> tuple:
    """`anchor` (walked on the card `_class_walks` names for `card`) re-read
    on `card` at `batch` — exactly that batch, as a direct estimate runs it
    (`capacity._direct_batches`): the card restamp and the batch rescale
    (`capacity._stamp_gpu_on_result`), as a sweep reads a point. Returns the
    restamped result and its per-op rows there.

    `p` is the preparation `_class_walks` gives — the one a sweep would
    read this point on — and it supplies this card's ceilings. What is
    under test is the ROW LAW read at another batch and card (`rows_at`
    below) and the phase re-read from it."""
    from backend.inference import capacity as cap
    ceilings = cap.capacity_ceilings(p, avg_input_tokens=CTX,
                                     avg_output_tokens=dep.avg_output_tokens)
    batches = cap._direct_batches(ceilings, prefill_batch_override=batch,
                                  decode_batch_override=batch)
    rows = {k: rows_at(anchor.get(k) or [], b, card=card, breakdown=breakdown)
            for k, b in batches.items()}
    # The anchor is a response: its rows' laws are read off their text.
    walks = {k: SimpleNamespace(analysis=response_cell()) for k in rows}
    from backend.inference.card import Card
    return cap._stamp_gpu_on_result(anchor, p, ceilings,
                                    Card.of(_walk_card(dep, card)),
                                    batches=batches,
                                    rows=rows, walks=walks, point=Point({}),
                                    breakdown=breakdown), rows


_BREAKDOWN_CASES = [
    pytest.param(Deployment(LLAMA, parallel=TP2), id="llama-tp2"),
    pytest.param(Deployment(QWEN_MOE, parallel=EP8), id="qwen3-moe-dp8ep8",
                 marks=pytest.mark.regression),
]


@pytest.mark.parametrize("dep", _BREAKDOWN_CASES)
def test_a_point_read_with_or_without_breakdown_is_the_same_point(dep):
    """One point read twice, with `breakdown` and without: the reading
    without it is the reading with it less what the breakdown shows
    (`without_breakdown`), byte for byte — at the walked batch and at
    another, on the walked card and on another. The numbers, the laws they
    are read off and the checks they pass take one path either way."""
    import json

    from backend.inference.module_rows import served
    at = _class_walks(dep, (dep.gpu_sku, "B200"))
    for card in (dep.gpu_sku, "B200"):
        anchor, p = at[card]
        for batch in (1, 64):
            shown, shown_rows = _stamped_at(dep, anchor, p, card, batch, breakdown=True)
            bare, bare_rows = _stamped_at(dep, anchor, p, card, batch, breakdown=False)
            with_ = served(without_breakdown({**shown, **shown_rows}, keyed=False))
            without = served(without_breakdown({**bare, **bare_rows}, keyed=False))
            assert (json.dumps(without, sort_keys=True, default=str)
                    == json.dumps(with_, sort_keys=True, default=str)), (
                f"{card} B={batch}:\n{report(diff(without, with_))}")
            assert all("flops_by_module" in ph for ph in shown["phases"].values() if ph)


_POINT_CASES = [
    pytest.param(Deployment(LLAMA), id="llama-tp1"),
    # A node dim that is a parameter's shape (the hyper-connection mix), and
    # a compressor's once-every-m-steps cadence.
    pytest.param(Deployment("deepseek-ai/DeepSeek-V4-Flash", parallel=ParallelConfig(
        tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4)), id="dsv4-flash-tp4",
                 marks=pytest.mark.regression),
    pytest.param(Deployment(QWEN_MOE, parallel=EP8), id="qwen3-moe-dp8ep8",
                 marks=pytest.mark.regression),
    pytest.param(Deployment(GPT_OSS), id="gpt-oss-swa",
                 marks=pytest.mark.regression),
]


@pytest.mark.parametrize("dep", _POINT_CASES)
def test_a_point_is_the_walk_there_whole(dep):
    """One walk per card CLASS at B=1; at every batch of {1, 3, 16, 64} on
    every card of `POINT_CARDS`, the rows read through their laws and the
    phases re-read from them are the direct estimate's there — every field,
    prefill and decode (identity aside, `_without_identity`): integers and
    text exactly, floats to 1e-12 relative.

    Which walk a card is priced off is the engine's answer, not this
    test's (`_class_walks`): a card of the walk's class is re-priced off
    it, a card of another class is walked for itself, and nothing here
    restates that rule. So the check spans the two cases a sweep has —
    H200 off the H100-SXM walk, B200 off its own — and it is also the
    mutation witness for the rule itself: revert the engine's refusal and
    B200 is priced off the H100-SXM walk again, which `per_op_timing.rows`
    below reports as DeepSeek-V4-Flash's 270 H100 rows against B200's own
    274, the routed activation BF16 where that card runs it FP8.

    The ROW SET and its order are compared first and in their own right
    (`_row_shape`) — the whole-row diff that follows would bury the row
    that moved under sixty thousand byte differences, and a row count or a
    `t_sol` would not see it at all."""
    assert dep.gpu_sku in POINT_CARDS, "the walked card must be one of them"
    at = _class_walks(dep, POINT_CARDS)
    for card in POINT_CARDS:
        anchor, p = at[card]
        for batch in (1, 3, 16, 64):
            phases = _stamped_at(dep, anchor, p, card, batch,
                                 breakdown=True)[0]["phases"]
            assert phases["prefill"]["batch"] == phases["decode"]["batch"] == batch
            walked = _without_identity(dep.on(card).direct(
                avg_input_tokens=CTX, prefill_batch=batch, decode_batch=batch))
            diffs = []
            for lst in ROW_LISTS:
                if lst in anchor:
                    read = rows_at(anchor[lst], batch, card=card, breakdown=True)
                    diffs += diff(_row_shape(read), _row_shape(walked[lst]),
                                  path=f"{lst}.rows")
                    diffs += diff(_without_identity(read), walked[lst], path=lst)
            for ph in ("prefill", "decode"):
                diffs += diff(_without_identity(phases[ph]), walked["phases"][ph],
                              path=f"phases.{ph}")
            assert not diffs, f"{card} B={batch}:\n{report(diffs)}"


_GRID = [{"key": "b1", "prefill_batch_size": 1, "decode_batch_size": 1},
         {"key": "b64", "prefill_batch_size": 64, "decode_batch_size": 64},
         {"key": "max", "prefill_batch_size": -1, "decode_batch_size": None}]
_SEQS = (1024, CTX)


def _context_batch(dep: Deployment) -> dict:
    from backend.inference.capacity import estimate_capacity_sweep
    return estimate_capacity_sweep(seq_lens=list(_SEQS), batch_grid=_GRID, **dep.env())


def test_context_batch_points_equal_direct(monkeypatch):
    """One walk for the whole sweep; each (context, batch) point of it is
    the direct estimate there: every capacity field, every phase number, and
    every per-op row on the columns its laws state. The fusion delta
    (`vs_naive_delta`, a step time at the walk's point) is not carried to a
    point."""
    from backend.inference import capacity
    walks = []
    real = capacity._capacity_pending
    monkeypatch.setattr(capacity, "_capacity_pending",
                        lambda *a, **k: walks.append(k["avg_input_tokens"]) or real(*a, **k))
    dep = Deployment(LLAMA)
    swept = _context_batch(dep)
    assert walks == [_SEQS[0]]
    for bp in _GRID:
        for i, seq in enumerate(_SEQS):
            direct = dep.direct(avg_input_tokens=seq, prefill_batch=bp["prefill_batch_size"],
                                decode_batch=bp["decode_batch_size"])
            point = swept[bp["key"]][i]
            rest = lambda e: {k: v for k, v in e.items()
                              if k not in (*ROW_LISTS, "phases", "vs_naive_delta")}
            diffs = diff(rest(point), rest(direct))
            diffs += _phase_diffs(point, direct, phase_scalars)
            for lst in ROW_LISTS:
                if lst in direct:
                    diffs += diff(law_view(point[lst]), law_view(direct[lst]), path=lst)
            assert "vs_naive_delta" not in point
            assert not diffs, f"{bp['key']} seq={seq}:\n{report(diffs)}"


def test_context_batch_point_is_the_walk_there_whole():
    """Every field of a Context x Batch point — the whole phases, the per-op
    rows with their measurement identity (`point_keys`, asked for by a
    request that looks measurements up), the capacity fields — is the direct
    estimate's there, as a point read with no breakdown shows it
    (`without_breakdown`; the anchor's fusion delta not carried)."""
    dep = Deployment(LLAMA, selection_mode="latest")
    swept = _context_batch(dep)
    for bp in _GRID:
        for i, seq in enumerate(_SEQS):
            direct = dep.direct(avg_input_tokens=seq, prefill_batch=bp["prefill_batch_size"],
                                decode_batch=bp["decode_batch_size"])
            direct.pop("vs_naive_delta")
            diffs = diff(swept[bp["key"]][i], without_breakdown(direct, keyed=True))
            assert not diffs, f"{bp['key']} seq={seq}:\n{report(diffs)}"


def test_a_request_that_looks_nothing_up_keys_no_point():
    """A theory request asks no point for its rows' keys: its points measure
    nothing and their rows re-read at another point state no identity and
    no key columns — and every number is the one a request that looks
    measurements up reads."""
    from backend.inference.capacity import sweep_points
    theory = sweep_points(seq_lens=list(_SEQS), batch_grid=_GRID,
                          **Deployment(LLAMA).env())[1]
    keyed = sweep_points(seq_lens=list(_SEQS), batch_grid=_GRID,
                         **Deployment(LLAMA, selection_mode="latest").env())[1]
    for seq_theory, seq_keyed in zip(theory, keyed, strict=True):
        for bare, full in zip(seq_theory, seq_keyed, strict=True):
            assert bare["measures"] == {} and full["measures"]
            diffs = diff(_without_identity(without_breakdown(bare["sol"], keyed=False)),
                         _without_identity(without_breakdown(full["sol"], keyed=False)))
            assert not diffs, report(diffs)


#: Contexts with a cached prefix beside plain ones: (prompt, cached prefix).
_PREFIXED = ((1024, 0), (CTX, CTX - 1024), (CTX, 0), (2 * CTX, 2 * CTX - 1024))


def test_a_cached_prefix_is_a_point_of_the_one_walk(monkeypatch):
    """A context whose prompt is partly cached is a point of the same walk
    as the plain contexts — the cached prefix is a count the laws read —
    and every field of it is the direct estimate with that prefix cached,
    as a point read with no breakdown shows it (the anchor's fusion delta
    not carried)."""
    from dataclasses import replace

    from backend.inference import capacity
    walks = []
    real = capacity._capacity_pending
    monkeypatch.setattr(capacity, "_capacity_pending",
                        lambda *a, **k: walks.append(k["avg_input_tokens"]) or real(*a, **k))
    dep = Deployment(LLAMA, selection_mode="latest")
    swept = capacity.estimate_capacity_sweep(
        seq_lens=[seq for seq, _ in _PREFIXED],
        s_prefixes=[prefix for _, prefix in _PREFIXED],
        batch_grid=_GRID[:2], **dep.env())
    assert walks == [_PREFIXED[0][0]]
    for bp in _GRID[:2]:
        for i, (seq, prefix) in enumerate(_PREFIXED):
            direct = replace(dep, knobs={"s_prefix_cached": prefix}).direct(
                avg_input_tokens=seq, prefill_batch=bp["prefill_batch_size"],
                decode_batch=bp["decode_batch_size"])
            direct.pop("vs_naive_delta")
            diffs = diff(swept[bp["key"]][i], without_breakdown(direct, keyed=True))
            assert not diffs, f"{bp['key']} seq={seq} prefix={prefix}:\n{report(diffs)}"


# ---------------------------------------------------------------------------
# The measured side, against the Kernel Bench stand-in.
# ---------------------------------------------------------------------------


def test_the_stand_in_answers_each_key_by_policy():
    """A measured key reads its row; a missing one and an ambiguous one
    (two kernels: `latest` cannot pick) read none; one estimate is one
    lookup."""
    def policy(key):
        if key.signature.startswith("gemm"):
            return MISSING
        if "rmsnorm" in key.signature:
            return AMBIGUOUS
        return "measured"

    with FixtureBench(policy) as bench:
        dep = Deployment(LLAMA, selection_mode="latest", measurement_scope=bench.scope)
        est = dep.direct(avg_input_tokens=CTX, prefill_batch=1, decode_batch=1)
        assert bench.lookups == 1
        states = {}
        for r in est["per_op_timing_decode"]:
            sig = r.get("signature") or ""
            m = r.get("t_measured_s")
            if sig.startswith("gemm") or "rmsnorm" in sig:
                assert m is None, (sig, m)
            states.setdefault(sig, m is not None)
        assert any(states.values()), "the stand-in measured no row"


def test_parallel_sweep_measured_is_each_points_own():
    """Each card's point reads the measured times of its own keys — its
    card, its batch — not the anchor point's: its rows and its phase totals
    equal a direct estimate there. One lookup serves the whole layout."""
    with FixtureBench() as bench:
        dep = Deployment(LLAMA, selection_mode="latest", measurement_scope=bench.scope)
        swept = _parallel_cell(dep)
        assert bench.lookups == 1
        for card in CARDS:
            direct = _direct_at_point(dep, card, swept[card])
            direct = without_breakdown(direct, keyed=True)
            for ph in ("prefill", "decode"):
                assert swept[card]["phases"][ph]["t_measured_s"] == pytest.approx(
                    direct["phases"][ph]["t_measured_s"], rel=1e-12), (card, ph)
            for lst in ("per_op_timing", "per_op_timing_decode"):
                cols = ("t_measured_s", "measured_state", "measured_key", "source",
                        "source_badge")
                pick = lambda rows: [{k: r.get(k) for k in cols} for r in rows]
                diffs = diff(pick(swept[card][lst]), pick(direct[lst]), path=f"{card}.{lst}")
                assert not diffs, report(diffs)
                got, want = measured_rows(swept[card][lst]), measured_rows(direct[lst])
                assert got.keys() == want.keys(), (card, lst)
                for uid, m in want.items():
                    assert (got[uid] is None) == (m is None), (card, lst, uid)
                    if m is not None:
                        assert got[uid] == pytest.approx(m, rel=1e-12), (card, lst, uid)


def test_context_batch_measured_is_each_points_own():
    """Every (context, batch) point reads its own keys' measured times, all
    of the sweep's points in one lookup: its rows, its phase totals and its
    coverage equal a direct estimate there."""
    def policy(key):   # decode is partly measured, prefill whole
        return "missing" if key.signature == "gemm" and key.phase == "decode" else "measured"
    with FixtureBench(policy) as bench:
        # Envelope: the fixture rows name their own implementation (`k1`),
        # while the estimator's latest selection asks for the framework's
        # declared implementation. This test is about a point reading its own
        # keys, not about that declaration contract.
        dep = Deployment(LLAMA, selection_mode="envelope", measurement_scope=bench.scope)
        swept = _context_batch(dep)
        assert bench.lookups == 1
        for bp in _GRID:
            for i, seq in enumerate(_SEQS):
                direct = dep.direct(avg_input_tokens=seq,
                                    prefill_batch=bp["prefill_batch_size"],
                                    decode_batch=bp["decode_batch_size"])
                point = swept[bp["key"]][i]
                direct.pop("vs_naive_delta")
                # The whole point a sweep shows: totals, coverage,
                # throughputs, the per-op rows' times, sources and badges.
                diffs = diff(point, without_breakdown(direct, keyed=True),
                             path=f"{bp['key']}.{seq}")
                assert not diffs, report(diffs)
                assert point["phases"]["prefill"]["t_measured_s"] is not None
                assert point["phases"]["decode"]["t_measured_s"] is None


def test_a_rejected_draft_reads_back_the_tokens_the_phase_published():
    """A sweep re-reads a phase's binding, so it must read the binding it
    was given — including an acceptance rate of ZERO.

    `α = 0.0` is a legal binding: every draft token is rejected, so a verify
    forward keeps the one token the step would have decoded anyway and
    `1 + α·K` is 1. Read with `or`, that zero became the default 1.0 — every
    draft accepted — and the two readers of one binding disagreed by `1 + K`:
    at K=3 and batch 8 the phase published 8 tok/step where the sweep's
    re-read published 32. The pin is the agreement, not either number, so it
    holds whatever `step_tokens` later decides a step produces.
    """
    from backend.inference.batch_sweep import _speculation_of
    for alpha in (0.0, 0.25, 1.0):
        binding = {"s_verify": 4, "acceptance_rate": alpha, "tokens_per_step": 1.0}
        published = Speculation.verifying(binding["s_verify"], alpha)
        reread = _speculation_of(binding)
        assert reread.acceptance_rate == published.acceptance_rate, alpha
        assert (step_tokens("decode", 8, 1.0, reread)
                == step_tokens("decode", 8, 1.0, published)), alpha
    # A binding that states no rate is the unspeculated default, still 1.0.
    assert _speculation_of({"s_verify": 4}).acceptance_rate == 1.0
