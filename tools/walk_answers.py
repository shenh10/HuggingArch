"""Does "the same answers" imply "the same rows"? — the oracle of
`docs/walk_identity.md` §4/§5.

The proposal there says a walk is identified by the answers it gives to the
questions it cannot defer, and that two cards whose answers agree on one
model's graph share a walk. That rests on one theorem:

    agreeing answers  =>  the same per-op row set, the same dispatch,
                          the same refusal.

This tool tests it, and nothing in the engine depends on it. It reads the
declarations (a framework binding's card conditions, the sheet's peak
columns), groups the cards by the answers they give, then walks each card
and compares what came out. A group whose members disagree is a
counterexample: a question is missing from the declared set, and naming it
is the point. Two cards in different groups that agree is not a defect —
the grouping is then finer than it needs to be, which only costs walks.

    python tools/walk_answers.py --model Qwen/Qwen3-30B-A3B
    python tools/walk_answers.py --corpus            # every snapshot
    python tools/walk_answers.py --corpus --jobs 6 --out /tmp/answers.json

The answer set is computed DELIBERATELY COARSE: only the questions
`walk_identity.md` §2 names. A missing question therefore shows up as a
violation rather than hiding inside an over-fine key, which is the
direction that teaches something.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)

from backend.api._models import InferenceEstimateRequest
from backend.api.gallery_frontier import _GALLERY_PRECISION, _layout_law
from backend.api.estimate_env import resolve_estimate_environment
from backend.arch.dtypes import UnknownDtype, peak_kind
from backend.arch.snapshots import iter_snapshot_ids
from backend.inference import gpu_specs
from backend.inference.calibration_overlay import key_columns, point_keys
from backend.inference.capacity import estimate_pending
from backend.inference.framework_bindings import ServingDeployment, get_binding
from backend.inference.gpu_specs import NoPeakForDtype
from backend.inference.parallel_sweep import DEFAULT_SWEEP_GPUS, class_cards

#: The framework every walk here is taken under. The questions are a
#: binding's, so the binding is named rather than defaulted.
FRAMEWORK = "vllm"


def class_reps(series: Optional[List[str]] = None) -> List[str]:
    """One card per walk class of the Scaling series, in series order."""
    cards = list(series or DEFAULT_SWEEP_GPUS)
    return list(dict.fromkeys(class_cards(gpu, cards)[0] for gpu in cards))


def _conditions(binding: Any) -> List[Tuple[Optional[str], Any]]:
    """Every dispatch condition a framework binding declares, with the op
    that declares it: an op implementation's, a kernel dispatch's, a
    precision launch's. A precision launch names no op — it is keyed by the
    quant schemes a checkpoint allows — so it carries None and is never
    restricted away.
    """
    out: List[Tuple[Optional[str], Any]] = []
    for op, impl in binding.op_implementations:
        out.extend((op, when) for when in impl.whens)
    for dispatch in binding.kernel_dispatches:
        out.extend((dispatch.op, when) for when in dispatch.whens)
    for launch in binding.precision_launches:
        out.extend((None, when) for when in launch.whens)
    return out


def card_question_answers(gpu_sku: str, binding: Any,
                          ops: Optional[set] = None) -> Dict[str, bool]:
    """The card half of the answer set: for each DISTINCT card-key triple a
    condition of `binding` declares, whether this card satisfies it.

    Keyed by the triple and not by the op, so two conditions with the same
    card keys are one question — the answer is a fact of the card and the
    declaration, never of which op asked.

    With `ops`, only the conditions an op in that set declares are asked:
    the refinement `walk_identity.md` §3.2 proposes, where a condition no
    op of this graph has cannot move this graph's rows. `ops` MUST be the
    union over every card compared, never one card's: a card whose
    condition failed does not carry the fused op it would have made, so
    restricting by that card's own ops would drop the very question that
    separates it (and silently claim the two cards agree).
    """
    deployment = ServingDeployment.at(gpu_sku=gpu_sku, activation_dtype=None)
    out: Dict[str, bool] = {}
    for op, condition in _conditions(binding):
        if ops is not None and op is not None and op not in ops:
            continue
        question = json.dumps(
            {"device": list(condition.device or ()),
             "min_compute_capability": condition.min_compute_capability,
             "below_compute_capability": condition.below_compute_capability},
            sort_keys=True, separators=(",", ":"))
        out[question] = bool(condition.holds_for_card(deployment))
    return out


def launch_answers(gpu_sku: str, binding: Any) -> Dict[str, bool]:
    """The precision half: which activation launches this card's deployed
    binding takes, and which candidates it did not satisfy.

    A selected launch decides the cast nodes (`materialize_casts` reads the
    resolution `serving.py` writes with `with_launches`) and a leaf's
    required edge dtype, so it changes the node set and belongs to the
    answers rather than to the price. The unavailable ones are kept too:
    `ResolvedPrecision` records one as a selector gap, which is also an
    output.
    """
    deployed = binding.deployed(
        ServingDeployment.at(gpu_sku=gpu_sku, activation_dtype=None))
    out: Dict[str, bool] = {}
    for launch in deployed.precision_launches_for():
        out[f"launch:{launch.name or launch.activation_format}"] = True
    for launch in deployed.unavailable_precision_launches_for():
        out.setdefault(
            f"launch:{launch.name or launch.activation_format}", False)
    return out


def peak_column_answers(gpu_sku: str,
                        families: Optional[List[str]] = None) -> Dict[str, bool]:
    """The sheet half: per MMA family, whether this card lists a peak
    column (`gpu_specs.peak_families`). With `families`, only the ones a
    walk needs; without, every family the sheet knows (the coarse form,
    which is what `CardClass` carries today)."""
    listed = set(gpu_specs.peak_families(gpu_specs.get_gpu(gpu_sku)))
    wanted = families if families is not None else sorted(
        gpu_specs.PEAK_COLUMNS)
    return {family: family in listed for family in wanted}


def _symbolic(formula: Any) -> str:
    """A displayed formula's symbolic half alone (`phases` writes
    `{"symbolic": ..., "substituted": ...}`): the law the walk wrote, not
    the numbers this card puts in it."""
    if isinstance(formula, dict):
        return str(formula.get("symbolic") or "")
    return str(formula or "")


def _fingerprint(value: Any) -> str:
    """A value as one comparable string.

    An EMPTY container carries its type: `{}`, `[]`, `()` and `None` are
    four different answers about a row — "no routed boundary" is not "a
    routed boundary with nothing in it" — and rendering them all as the
    same two characters would let one pass for another. Review asked for
    this sentinel before Stage A.
    """
    if isinstance(value, dict):
        if not value:
            return "<empty dict>"
        return "{" + ",".join(
            sorted(f"{k!r}:{_fingerprint(v)}" for k, v in value.items())) + "}"
    if isinstance(value, (list, tuple)):
        if not value:
            return f"<empty {type(value).__name__}>"
        return "[" + ",".join(_fingerprint(v) for v in value) + "]"
    return repr(value)


def row_signature(model_id: str, gpu_sku: str, parallel: Dict[str, Any]
                  ) -> Dict[str, Any]:
    """What a card's walk produced, as the thing two cards must agree on:
    per row list, each row's op, lookup identity and dispatch.

    This is the comparison CLAUDE.md §1.5 asks of anything that re-prices a
    walk on another card — the per-op row SET and ORDER including dtype, not
    a total — plus the dispatch, because a record stores it and two cards
    that dispatch differently cannot share one.
    """
    request = InferenceEstimateRequest(
        model_id=model_id, gpu_sku=gpu_sku, avg_input_tokens=4096,
        avg_output_tokens=256, parallel=parallel, framework=FRAMEWORK,
        selection_mode="theoretical_only", decode_batch_size=8)
    pending = estimate_pending(**resolve_estimate_environment(request))
    result = pending.finish()
    rows: Dict[str, Any] = {}
    dtypes: set = set()
    ops: set = set()
    for name, walk in pending.walk_of_rows.items():
        listed = result.get(name) or []
        if not listed:
            continue
        for row in listed:
            for field in ("act_dtype", "weight_dtype", "metadata_dtype"):
                if row.get(field):
                    dtypes.add(str(row[field]))
            # Both spellings: the row's own op and the physical op its key
            # resolves to (`RowIdentity.op_of` renames attention variants),
            # because a binding entry is keyed by either.
            if row.get("op"):
                ops.add(str(row["op"]))
        keyed = key_columns(listed, walk.analysis)
        points = point_keys(walk.analysis, keyed, walk.workload,
                            walk.deployment, model_id=walk.model_id,
                            spec_hash=walk.spec_hash)
        rows[name] = [
            (# the row's identity in the graph, and its op as the walk
             # spelled it before the key resolved a variant
             str(row.get("aggregation_id") or ""), row.get("uid"),
             row.get("op"), row.get("count_per_step"), row.get("is_comm"),
             row.get("comm_op"), row.get("role"), row.get("conversion"),
             # the structure a cast or a collective adds, and the geometry
             # one launch runs — both change with a card and both reach a
             # stored record
             _fingerprint(row.get("routed")),
             _fingerprint(row.get("shape_in")),
             _fingerprint(row.get("shape_out")),
             _fingerprint(row.get("comm_payload_shape")),
             # the law's SHAPE, never its substituted value: the symbolic
             # half is the walk's, the substituted half is the card's price
             # (`max(FLOPs / peak_flops, ...)` reads this card's peaks), so
             # comparing the whole formula compares prices and reports every
             # pair of cards as a counterexample.
             _symbolic(row.get("t_op_formula")),
             _symbolic(row.get("access_bytes_formula")),
             # what the row is measured as
             point.op, point.signature, point.shape_json,
             point.dtype_env_json, point.launches, str(point.cadence),
             point.declined,
             _fingerprint(point.launch_shape_in),
             _fingerprint(point.launch_shape_out),
             _fingerprint(None if point.dispatch is None
                          else point.dispatch.as_dict()))
            for row, point in zip(keyed, points, strict=True)]
        ops.update(str(point.op) for point in points if point.op)
    return {"rows": rows, "dtypes": sorted(dtypes), "ops": sorted(ops),
            "refusal": result.get("infeasibility_reason")}


def _one_card(item: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """One (model, card) walk, its failure kept as the answer it is.

    A card the layout does not FIT on carries no rows, and that is not a
    walk difference: the HBM budget is on the pricing side of the split
    (`CardClass`: "HBM, bandwidths and `gpus_per_node` are substituted when
    the walk is priced on the card"), so a capacity refusal is a per-card
    outcome by design and says nothing about the structure. Such a card is
    reported and left out of the comparison; counting it as "different
    rows" would read the dense 32B that does not fit on a 24 GB card as a
    counterexample to the theorem.
    """
    model_id, gpu_sku, parallel = item
    try:
        got = row_signature(model_id, gpu_sku, parallel)
    except Exception as exc:
        # NOT dropped: a raised refusal is an output two cards with equal
        # answers must agree on (`NoPeakForDtype` is exactly the structural
        # refusal the answers are supposed to predict). It is compared as
        # its type, so only "both raised the same kind" counts as agreeing.
        # `dtypes` and `ops` are stated empty, not omitted: `audit_model`
        # reads them off every card it compares, so a record without them
        # crashed the audit — which is why this path, added to compare a
        # structural refusal, had never compared one.
        return {"gpu": gpu_sku, "raised": type(exc).__name__,
                "detail": str(exc)[:200], "dtypes": [], "ops": [],
                "rows": {"__raised__": [type(exc).__name__]}}
    if not any(got["rows"].values()):
        # Empty rows are only excused when the walk SAID it was a capacity
        # refusal. Anything else empty is kept and compared, so a card that
        # silently produced nothing cannot hide as "infeasible".
        reason = got.get("refusal")
        if reason:
            return {"gpu": gpu_sku, "infeasible": reason}
        return {"gpu": gpu_sku, **got,
                "rows": {"__empty__": ["no rows and no stated refusal"]}}
    return {"gpu": gpu_sku, **got}


def _group(answers: Dict[str, Dict[str, bool]]) -> Dict[str, List[str]]:
    """Cards by the answers they gave, the group's key its answer set."""
    out: Dict[str, List[str]] = {}
    for gpu, answer in answers.items():
        out.setdefault(_fingerprint(answer), []).append(gpu)
    return out


def audit_model(model_id: str, parallel: Dict[str, Any],
                cards: List[str]) -> Dict[str, Any]:
    """One model: group the cards by their answers, walk each card, and
    report every disagreement inside a group (a missing question) and every
    agreement across groups (an over-fine grouping)."""
    binding = get_binding(FRAMEWORK)
    signatures = {}
    for card in cards:
        got = _one_card((model_id, card, parallel))
        signatures[card] = got
    walked = {card: got for card, got in signatures.items()
              if "error" not in got and "infeasible" not in got}
    # The dtypes any card's walk needed: the restriction that makes the
    # peak half exact rather than the card's whole column set. Taken over
    # every card so the restriction is not a function of the card being
    # grouped.
    needed = sorted({dtype for got in walked.values()
                     for dtype in got["dtypes"]})
    # The ops any card produced. The union, for the reason
    # `card_question_answers` states.
    ops = {op for got in walked.values() for op in got["ops"]}
    # The MMA family of each dtype the walk needed, off the one declaration
    # (`dtypes.peak_kind`) and never by matching the strings: the rows spell
    # a dtype `BF16` and a peak column `bf16`, so `in` found nothing and the
    # whole peak half silently left the key — the self-blind check CLAUDE.md
    # warns about, found by review.
    families = set()
    for dtype in needed:
        try:
            families.add(peak_kind(dtype))
        except UnknownDtype:
            # A dtype with no MMA column (block scales, index tables) asks
            # no peak question; it is not a gap to paper over.
            continue
    families = sorted(families)
    launches = {card: launch_answers(card, binding) for card in walked}
    coarse = {card: {**card_question_answers(card, binding),
                     **peak_column_answers(card)} for card in walked}
    refined = {card: {**card_question_answers(card, binding, ops),
                      **peak_column_answers(card, families),
                      **launches[card]}
               for card in walked}
    violations: List[Dict[str, Any]] = [
        {"answers": key, "cards": [members[0], other],
         "what": "same answers, different rows"}
        for key, members in _group(refined).items()
        for other in members[1:]
        if _fingerprint(walked[other]["rows"])
        != _fingerprint(walked[members[0]]["rows"])]
    missed: List[List[str]] = []
    groups = _group(refined)
    keys = sorted(groups)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            a, b = groups[left][0], groups[right][0]
            if _fingerprint(walked[a]["rows"]) == _fingerprint(
                    walked[b]["rows"]):
                missed.append([a, b])
    return {
        "model": model_id,
        "cards_walked": sorted(walked),
        "cards_refused": {card: got["error"] for card, got in
                          signatures.items() if "error" in got},
        "cards_infeasible": {card: got["infeasible"] for card, got in
                             signatures.items() if "infeasible" in got},
        "needed_dtypes": needed,
        "restricted_families": families,
        "ops_asked": len([1 for op, _c in _conditions(binding)
                          if op is None or op in ops]),
        "conditions_declared": len(_conditions(binding)),
        "groups_coarse": len(_group(coarse)),
        "groups_refined": len(groups),
        "refined_groups": {key: members for key, members in groups.items()},
        "violations": violations,
        "over_fine": missed,
    }


def _job(item: Tuple[str, Dict[str, Any], List[str]]) -> Dict[str, Any]:
    model_id, parallel, cards = item
    try:
        return audit_model(model_id, parallel, cards)
    except Exception as exc:
        return {"model": model_id, "fatal": f"{type(exc).__name__}: {exc}"}


#: The compared fields of a row, in `row_signature`'s order — named here
#: so `--mutate` can report which one a mutation exercised.
SIGNATURE_FIELDS = (
    "aggregation_id", "uid", "op", "count_per_step", "is_comm", "comm_op",
    "role", "conversion", "routed", "shape_in", "shape_out",
    "comm_payload_shape", "t_op_formula(symbolic)",
    "access_bytes_formula(symbolic)", "key.op", "key.signature",
    "key.shape_json", "key.dtype_env_json", "key.launches", "key.cadence",
    "key.declined", "launch_shape_in", "launch_shape_out", "dispatch")


def mutation_test(model_id: str, parallel: Dict[str, Any],
                  cards: List[str]) -> int:
    """Does this comparison actually compare? — the oracle of the oracle.

    "No counterexamples" is worth nothing if the comparison is blind, and
    two of this tool's first three bugs were exactly that (a peak family
    matched by string containment, every empty walk dropped). So each
    compared field is mutated on ONE card of a group and the audit must
    report a violation; a field whose mutation stays green is not in the
    comparison and the conclusion does not cover it. Returns the number of
    fields that failed to redden.
    """
    groups = _group_of_model(model_id, parallel, cards)
    victim = next((members[0] for members in groups if len(members) > 1), None)
    if victim is None:
        print("no group with two cards at this layout: nothing to mutate")
        return 1
    print(f"mutating {victim} (grouped with "
          f"{[m for ms in groups if victim in ms for m in ms if m != victim]})")
    original = globals()["row_signature"]
    blind: List[str] = []

    def audit(mutate: Any) -> int:
        def spy(mid: str, gpu: str, par: Dict[str, Any]) -> Dict[str, Any]:
            got = original(mid, gpu, par)
            if gpu == victim and got["rows"]:
                got = {**got, "rows": mutate(got["rows"])}
            return got
        globals()["row_signature"] = spy
        try:
            return len(audit_model(model_id, parallel, cards)["violations"])
        finally:
            globals()["row_signature"] = original

    if audit(lambda rows: rows):
        print("  baseline already violates: fix that before mutating")
        return 1
    print("  baseline clean")
    for index, field in enumerate(SIGNATURE_FIELDS):
        def mutate(rows: Dict[str, Any], at: int = index) -> Dict[str, Any]:
            return {name: [tuple("MUTATED" if j == at else value
                                 for j, value in enumerate(row))
                           if k == 0 else row
                           for k, row in enumerate(listed)]
                    for name, listed in rows.items()}
        red = audit(mutate)
        print(f"  {field:32s} {'red' if red else 'GREEN — NOT COMPARED'}")
        if not red:
            blind.append(field)
    print(f"blind fields: {blind or 'none'}")
    return len(blind)


#: The one path of a stored record that names the card by design: the
#: record says which card its rows were walked on.
RECORD_NAMES_THE_CARD = ("walked_on",)


#: Lists a record stores a MAPPING in, with the fields that key each
#: entry. Read positionally they diff on the order the caller named its
#: cards in, which is the caller's and not the record's.
RECORD_MAPPINGS: Dict[str, Tuple[str, ...]] = {
    "capacity": ("gpu_sku", "context"),
    "draft_capacity": ("gpu_sku", "context"),
    "points": ("gpu", "context", "batch", "draftTokens"),
}


def _normalised(value: Any, key: str = "") -> Any:
    """A record with its mapping-bearing lists (`RECORD_MAPPINGS`) in a
    canonical order, so a diff reads them as the mappings they are."""
    if isinstance(value, dict):
        return {name: _normalised(item, name) for name, item in value.items()}
    if isinstance(value, list):
        listed = [_normalised(item) for item in value]
        fields = RECORD_MAPPINGS.get(key)
        if fields and all(isinstance(item, dict) for item in listed):
            return sorted(listed, key=lambda item: tuple(
                str(item.get(field)) for field in fields))
        return listed
    return value


def _leaves(value: Any, path: str = "") -> Dict[str, Any]:
    """Every leaf of a record by its path."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            out.update(_leaves(item, f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(value, (list, tuple)):
        out = {}
        for index, item in enumerate(value):
            out.update(_leaves(item, f"{path}[{index}]"))
        return out
    return {path: value}


def record_test(model_id: str, parallel: Dict[str, Any],
                cards: List[str]) -> int:
    """Is the comparison COMPLETE? — by construction rather than by a list.

    `row_signature` compares a curated projection of a row, so "it compares
    everything that matters" rests on a field list and the mutation check.
    This closes it the other way round: build the whole record the pool
    stores (`gallery_frontier._layout_law`) for two cards of one answer
    group and diff every leaf. A differing leaf is allowed only if it is a
    NUMBER — the card's price is numeric by construction (`cardBinding`
    binds peaks, bandwidth, fabric and the launch floor, and nothing else,
    into a law graph that is already fixed) — or the one path that names the
    card on purpose. Any differing string, bool or structure is a
    counterexample this tool would otherwise have to have been told to look
    for.

    Returns the number of non-numeric differences found.
    """
    groups = _group_of_model(model_id, parallel, cards)
    pair = next((members for members in groups if len(members) > 1), None)
    if pair is None:
        print("no group with two cards at this layout: nothing to diff")
        return 1
    left, right = pair[0], pair[1]
    precision = _GALLERY_PRECISION.canonical()
    # BOTH records cover BOTH cards and differ only in which card was
    # WALKED on. Building each over its own card alone would diff the
    # record's per-card sections — its capacity view and the batches its
    # measure grid enumerates are each card's HBM ceiling — and report them
    # as counterexamples, which they are not: a record is stored for a
    # class with every card of the class in it (`capacityCards`). The
    # question is whether walking it on either card gives the same record.
    records = {
        left: _layout_law((model_id, [left, right], 0.9, precision, parallel)),
        right: _layout_law((model_id, [right, left], 0.9, precision, parallel)),
    }
    a = _leaves(_normalised(records[left]))
    b = _leaves(_normalised(records[right]))
    only = sorted(set(a) ^ set(b))
    differ = sorted(path for path in set(a) & set(b) if a[path] != b[path])
    numeric = [path for path in differ
               if isinstance(a[path], (int, float))
               and not isinstance(a[path], bool)
               and isinstance(b[path], (int, float))
               and not isinstance(b[path], bool)]
    declared = [path for path in differ
                if path.split(".")[-1] in RECORD_NAMES_THE_CARD
                or path in RECORD_NAMES_THE_CARD]
    bad = [path for path in differ
           if path not in numeric and path not in declared]
    print(f"record diff: walked on {left} vs on {right}, both covering "
          f"both (one answer group): "
          f"{len(a)} leaves, {len(differ)} differ, "
          f"{len(numeric)} numeric (price), {len(declared)} declared, "
          f"{len(only)} present on one side only")
    for path in (bad + only)[:12]:
        print(f"  NOT A PRICE: {path} "
              f"{str(a.get(path))[:40]!r} vs {str(b.get(path))[:40]!r}")
    return len(bad) + len(only)


def refusal_test(model_id: str, parallel: Dict[str, Any],
                 cards: List[str]) -> int:
    """Does a STRUCTURAL refusal reach the comparison? — the second half of
    the oracle's own check.

    A refusal raised by the walk (`NoPeakForDtype`: the card lists no
    column for a dtype this walk needs) is precisely what the answers are
    supposed to predict, so two cards with equal answers must agree on it.
    The path that compares it existed but had never run: the record it
    returned omitted the fields `audit_model` reads off every compared
    card, so the first card to raise would have crashed the audit instead.
    This forces one card of a group to raise and requires a violation.
    """
    groups = _group_of_model(model_id, parallel, cards)
    pair = next((members for members in groups if len(members) > 1), None)
    if pair is None:
        print("no group with two cards at this layout: nothing to force")
        return 1
    victim = pair[0]
    original = globals()["row_signature"]

    def spy(mid: str, gpu: str, par: Dict[str, Any]) -> Dict[str, Any]:
        if gpu == victim:
            raise NoPeakForDtype("fp8", "forced by --refusal")
        return original(mid, gpu, par)

    # Against a CLEAN baseline, not in isolation: "a violation mentioning
    # this card exists" would also pass on a model that already violates,
    # so the check asserts the forced refusal ADDS one. Review asked for
    # this before Stage A.
    before = audit_model(model_id, parallel, cards)["violations"]
    if before:
        print(f"baseline already violates ({len(before)}): this model cannot "
              f"test the refusal path")
        return 1
    globals()["row_signature"] = spy
    try:
        found = audit_model(model_id, parallel, cards)["violations"]
    finally:
        globals()["row_signature"] = original
    added = [violation for violation in found if victim in violation["cards"]]
    reported = len(found) > len(before) and bool(added)
    print(f"forcing a structural refusal on {victim} (grouped with "
          f"{[card for card in pair if card != victim]}): baseline "
          f"{len(before)} violations, forced {len(found)}, naming it "
          f"{len(added)} — "
          f"{'reported' if reported else 'NOT REPORTED — the oracle is blind'}")
    return 0 if reported else 1


def _group_of_model(model_id: str, parallel: Dict[str, Any],
                    cards: List[str]) -> List[List[str]]:
    return list(audit_model(model_id, parallel, cards)
                ["refined_groups"].values())


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--corpus", action="store_true",
                        help="every snapshot in the library")
    parser.add_argument("--parallel", default='{"ep_moe": 8, "dp_attention": 8}',
                        help="the layout to walk, as JSON")
    parser.add_argument("--cards", default=None,
                        help="comma-separated; default is one card per walk "
                             "class of the Scaling series")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default=None, help="write the report as JSON")
    parser.add_argument("--record", action="store_true",
                        help="diff the whole stored record between two cards "
                             "of one answer group; every difference must be "
                             "a price (numeric). Needs one --model")
    parser.add_argument("--refusal", action="store_true",
                        help="force a structural refusal on one card and "
                             "check the audit reports it; needs one --model")
    parser.add_argument("--mutate", action="store_true",
                        help="mutate each compared field of one card and "
                             "check the audit reddens (the oracle of the "
                             "oracle); needs one --model")
    args = parser.parse_args(argv)

    cards = (args.cards.split(",") if args.cards else class_reps())
    parallel = json.loads(args.parallel)
    models = (sorted(iter_snapshot_ids()) if args.corpus
              else list(args.models or []))
    if not models:
        parser.error("name --model or pass --corpus")
    print(f"cards {cards}\nlayout {parallel}\nmodels {len(models)}",
          flush=True)
    if args.mutate or args.refusal or args.record:
        if len(models) != 1:
            parser.error("--mutate / --refusal / --record take one --model")
        failed = 0
        if args.mutate:
            failed += mutation_test(models[0], parallel, cards)
        if args.refusal:
            failed += refusal_test(models[0], parallel, cards)
        if args.record:
            failed += record_test(models[0], parallel, cards)
        return 1 if failed else 0

    items = [(model, parallel, cards) for model in models]
    reports: List[Dict[str, Any]] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for report in pool.map(_job, items):
                reports.append(report)
                _say(report)
    else:
        for item in items:
            report = _job(item)
            reports.append(report)
            _say(report)

    violations = [v for report in reports for v in report.get("violations", [])]
    over_fine = [report for report in reports if report.get("over_fine")]
    fatal = [report for report in reports if report.get("fatal")]
    walks_today = sum(len(report.get("cards_walked") or ())
                      for report in reports)
    walks_grouped = sum(report.get("groups_refined") or 0
                        for report in reports)
    print("\n== summary", flush=True)
    print(f"models reported {len(reports)}, fatal {len(fatal)}")
    print(f"THEOREM violations (same answers, different rows): "
          f"{len(violations)}")
    for violation in violations[:20]:
        print(f"  {violation['cards']}")
    print(f"over-fine pairs (different answers, same rows): "
          f"{sum(len(r['over_fine']) for r in over_fine)} "
          f"in {len(over_fine)} models")
    print(f"card walks: {walks_today} today -> {walks_grouped} grouped")
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1))
        print(f"report written to {args.out}")
    return 1 if violations or fatal else 0


def _say(report: Dict[str, Any]) -> None:
    if report.get("fatal"):
        print(f"  {report['model']}: FATAL {report['fatal'][:90]}", flush=True)
        return
    mark = "VIOLATION" if report["violations"] else "ok"
    unfit = len(report.get("cards_infeasible") or ())
    print(f"  {report['model']}: {len(report['cards_walked'])} cards"
          + (f" (+{unfit} unfit)" if unfit else "")
          + f", coarse {report['groups_coarse']} -> refined "
          f"{report['groups_refined']} groups, {mark}"
          + (f", over-fine {len(report['over_fine'])}"
             if report["over_fine"] else ""), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
