"""Measured rows whose achieved rates were computed by an older walk.

`achieved_tflops` and `achieved_bw_gbps` are the plan's per-launch
numerators (`flops_per_rank`, `access_bytes_per_rank`) over the row's own
p50. The p50 is the measurement; the numerators are what the walk said the
launch moves on the day the plan was written. When the walk corrects that
— an index gather billed for the whole table it picks from, an FP8 GEMM
billed for the activation cast its timed launch does not do, a context-
parallel rank billed for keys outside its own queries' window — every row
already in the lake keeps the old numerator. On 2026-10-27 that left 332
rows above the card's peak bandwidth and many more below it reading 2-30x
high, and nothing re-stamped them: `hbench restamp` rewrites a results
CSV, not the lake.

This plans every (model, GPU) the lake holds rows of again, at today's
engine, over both sweep surfaces (`kernel_bench.sweep.plan_surface`), then
plans each deployment the rows left unpriced at the rows' own batches and
query lengths (`own_points`, `kernel_bench.sweep.plan_points`) — a row
whose batch the capacity has since moved off the grid is still read by
every estimate of its physical key, so its numerator matters — and joins
on the PHYSICAL cell — `(gpu, op, shape_json, dtype_env_json,
phase)`, the identity the sweep dedups jobs by, because one card runs a
kernel at one shape the same way whichever deployment cut the shape. Any
model's plan on that GPU may answer a cell. A row whose cell no plan of
today produces is left and counted (`unplanned`: a key the walk no longer
gives — an axis added to the cell, a dtype slot re-keyed); a cell two of today's
jobs price differently is left and counted (`ambiguous`) — that is a walk
inconsistency to fix, not a number to pick.
Pairs that differ only by floating-point rounding are one price. A genuinely
ambiguous cell is asked again by its exact lake selector — model, deployment,
and node — from plan provenance; only a node the walk still prices two ways
remains ambiguous.

    python -m tools.restamp_achieved --out .cache/restamp_achieved.json
    python -m tools.restamp_achieved --apply .cache/restamp_achieved.json

Planning reads a copy of the catalog (`tools.lake_rows`) and caches each
(model, GPU, surface) plan under `--cache`, so a stopped run resumes. The
Kernel Bench service holds the lake, so `--apply` runs with it stopped
(`ha` stops and starts it), and writes through `store.restamp`: only the
roofline columns, in place, each row keeping its batch and review status.

`--stale-only` stops after the surface join and classifies every leftover
key by what today's plans no longer carry: the point, the routed geometry,
another shape axis, a dtype slot, or the op itself. It is the cheap report
that decides whether the (much longer) own-point pass is worth running; an
`op-gone` row's neighbouring op is evidence, never a proposed rename.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from backend.inference.benchmark_identity import (
    canonical_identity_json, physical_cell_of,
)
from kernel_bench.emit import UTILIZATION_CEILING

SURFACES = ("routine", "frontier")


def precisions(extra: Sequence[str] = ()) -> Tuple[Optional[str], ...]:
    """The precision ids a surface is replanned under: as shipped, plus any
    the caller names (`--precision`).

    The join is on the PHYSICAL cell and `dtype_env_json` is part of it, so
    a row measured under a non-shipped precision matches no as-shipped job
    and lands in `unplanned` — a bucket that means "the walk no longer
    gives this key", i.e. a defect indicator. `kernel_bench sweep
    --precision` made such rows possible (GLM-5.3-Flash's INT8 FFN is the
    first), so they have to be plannable here too.

    Why the caller names them rather than this planning every declared
    preset: a row does not record the profile it was measured under, so
    there is nothing to derive from, and planning all of them costs the
    whole tool a multiple of its runtime (six presets today) for a class of
    rows that is empty until someone measures one. The operator who ran the
    sweep knows which profile they passed; `unplanned_hint` makes the
    report say so when rows land there.

    When a row does carry its profile, this becomes derivable and the flag
    should go."""
    from backend.arch.precision_profile import PRESETS
    unknown = sorted(set(extra) - set(PRESETS))
    if unknown:
        raise SystemExit(f"unknown precision preset(s) {unknown}; "
                         f"one of {sorted(PRESETS)}")
    rest = tuple(p for p in dict.fromkeys(extra) if p != "as_shipped")
    return (None, *rest)


def unplanned_hint(unplanned: Sequence[Any], asked: Sequence[str]) -> str:
    """What to say when rows went unplanned and no extra precision was
    asked for. `unplanned` is a defect indicator, so a row that is merely
    measured under another profile must not be left looking like one."""
    if not unplanned or asked:
        return ""
    return ("note: rows a non-shipped precision measured match no "
            "as-shipped cell and land here. If any of these were measured "
            "with `kernel_bench sweep --precision <preset>`, re-run with "
            "the same `--precision <preset>` so their cells are planned.")
#: A numerator that moved by less than this is the plan's own rounding.
_SAME = 1e-3
#: The achieved columns carry four decimals (`emit.achieved_from_latency`),
#: so a numerator read back off a row is uncertain by half a unit of the
#: last one times its p50: a 5 us launch at 0.0001 TFLOPS reads back as
#: 500 FLOPs whatever it was billed. Per microsecond of p50, in the
#: numerator's own unit.
_READBACK = {"flops": 0.5e-4 * 1e12 * 1e-6, "bytes": 0.5e-4 * 1e9 * 1e-6}
#: Rows per `store.restamp` transaction.
_CHUNK = 20000

#: (op, shape_json, dtype_env_json, phase), the two JSON texts canonical.
Cell = Tuple[str, str, str, str]
Pair = Tuple[Optional[float], Optional[float]]


class _Provenance:
    """Plan prices by exact lake selector, and the fresh-prefill subset.

    Kernel Bench's historical prefill tracks have no cached prefix; when an
    own-point pass offers several prefix contexts for the same under-keyed
    cell, the fresh one is the row's own context."""

    def __init__(self):
        self.rows: Dict[Tuple[str, str, str, str, Cell], set] = defaultdict(set)
        self.fresh: Dict[Tuple[str, str, str, str, Cell], set] = defaultdict(set)
_POINT_AXES = frozenset(("B", "s_query"))
_CONTEXT_AXIS = "s_kv"
_CONTEXT_ALL = -1
_ROUTED_AXES = frozenset(("n_experts", "n_tokens", "s_expert_rows"))


def _surface_seqs() -> set:
    from kernel_bench.sweep import sweep_surfaces
    return {s for spec in sweep_surfaces().values() for s in spec["seq_lens"]}


def cell_of(op: str, shape: Any, dtype_env: Any, phase: str) -> Cell:
    """The physical cell, from a row's JSON texts or a legacy plan's dicts.

    Current plan jobs carry ``physical_key``; this shared canonicalizer is
    for row-shaped identity fields and the restamp cache, which stores cells
    rather than jobs.  It is not a second restamp key in ``emit``.
    """
    return physical_cell_of(op, shape, dtype_env, phase)


def _plan_file(cache: Path, model: str, gpu: str, surface: str,
               precision: Optional[str] = None) -> Path:
    # The precision is part of the cache identity: two profiles plan
    # different cells for the same (model, gpu, surface), so sharing one
    # file would let whichever ran first answer for both.
    at = f"__{precision}" if precision else ""
    return cache / gpu / f"{model.replace('/', '__')}__{surface}{at}.json"


def _provenance_file(cells: Path) -> Path:
    return cells.with_name(f"{cells.stem}.provenance.json")


def _walked_context(point) -> tuple:
    """A plan point as the provenance cache's context record."""
    if point is None:
        return (None, 0, None, None)
    return (point.seq_len, point.s_prefix_cached, point.s_verify, point.batch)


def _same_pair(a: Pair, b: Pair) -> bool:
    """Whether two numerator pairs differ only by the plan's own rounding."""
    for x, y in zip(a, b, strict=True):
        if (x is None) != (y is None):
            return False
        if x is None or y is None:
            continue
        if abs(x - y) > _SAME * max(abs(x), abs(y), 1.0):
            return False
    return True


def _fold_pairs(pairs) -> set:
    """A set with numerically equivalent plans represented once."""
    out: list = []
    for pair in sorted(pairs, key=str):
        if not any(_same_pair(pair, kept) for kept in out):
            out.append(pair)
    return set(out)


def _write_cells(out: Path, jobs) -> None:
    """Each physical cell `jobs` price, with every distinct numerator pair
    they give it — `[op, shape, dtype_env, phase, [[flops, bytes], …]]`."""
    cells: Dict[Cell, set] = defaultdict(set)
    for job in jobs:
        cells[cell_of(job["op"], job["dims"], job["dtype_env"], job["phase"])].add(
            (job.get("flops_per_rank"), job.get("access_bytes_per_rank")))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps([[*cell, sorted(v, key=str)] for cell, v in cells.items()]))
    tmp.replace(out)


def _write_provenance(cells: Path, jobs) -> None:
    """The cells `jobs` price beside the exact lake selector that priced
    them: model, deployment, and node. The physical join stays primary; this
    is the tie-breaker for a cell today's plans price in two genuinely
    different boundary forms."""
    by_selector: Dict[Tuple[Tuple[str, str, str, Cell], tuple], set] = (
        defaultdict(set))
    for job in jobs:
        node = job.get("node_spec_id")
        if not node:
            continue
        key = (job["model"], job["parallel"], node,
               cell_of(job["op"], job["dims"], job["dtype_env"], job["phase"]))
        context = _walked_context(job.get("_point"))
        by_selector[(key, context)].add((job.get("flops_per_rank"),
                                         job.get("access_bytes_per_rank")))
    out = _provenance_file(cells)
    rows = []
    rows.extend([*key[:3], *key[3], sorted(_fold_pairs(pairs), key=str),
                 list(context)]
                for (key, context), pairs in by_selector.items())
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, sort_keys=True))
    tmp.replace(out)


def replan(task: Tuple[str, str, str, Optional[str], str]
           ) -> Tuple[str, str, str, Optional[str], str]:
    """`(model, gpu, surface, precision, why not)`: plan one surface for one
    model on one GPU at one precision into the cache (`_write_cells`). A
    cached plan is kept."""
    model, gpu, surface, precision, cache = task
    out = _plan_file(Path(cache), model, gpu, surface, precision)
    provenance = _provenance_file(out)
    if out.exists() and provenance.exists():
        return model, gpu, surface, precision, ""
    from kernel_bench.sweep import plan_surface
    try:
        from backend.arch.precision_profile import PrecisionProfile
        profile = (PrecisionProfile.from_request({"preset": precision})
                   if precision else None)
        jobs, _stats = plan_surface(model, gpu, surface, precision=profile)
    except Exception as e:  # a model the engine refuses is counted, not fatal
        return model, gpu, surface, precision, f"{type(e).__name__}: {e}"
    _write_cells(out, jobs)
    _write_provenance(out, jobs)
    return model, gpu, surface, precision, ""


def own_points(phase: str, parallel: Any,
               asked: List[Tuple[int, int, int]], *,
               split_contexts: bool = False) -> List[List[Any]]:
    """The plan calls that reach the rows' own points: each a list of
    `plan.PlanPoint`, one call per context so a context the deployment
    refuses (`prefix_refusal`) loses only its own points.

    A row states its rank's `B` and `s_query`; the point is the request
    batch and the context. The batch is `B`, or `B` times the attention's
    data-parallel degree when each rank holds its share of the requests; the
    context is any of the surfaces' — the join keeps only what matches the
    row's cell, so a candidate that misses costs time, not truth. A decode
    row's `s_query` is the verify width; a prefill row's is the new tokens
    of its rank, `cp` of them making the request's, over no cached prefix or
    the rest of a longer context cached.

    Each asked point also says which decode context it needs: `0` for a cell
    whose physical shape names no context axis (one representative context
    prices the same B and query width), a surface sequence length named by the
    row's `s_kv`, or `-1` to try every context when `s_kv` is present but is
    not itself a surface context. Decode's selected contexts are one call:
    `build_jobs_over_points` shares each context's walk and compiled laws over
    every batch and verify width in it. `split_contexts` is the fallback after
    a refusal, so one context cannot lose the rest."""
    from kernel_bench.plan import PlanPoint
    seqs = sorted(_surface_seqs())
    dp = max(1, int(getattr(parallel, "dp_attention", 1) or 1))
    cp = max(1, int(getattr(parallel, "cp", 1) or 1))
    calls: Dict[Any, set] = defaultdict(set)
    for b, sq, context in asked:
        for batch in {b, b * dp}:
            if phase == "decode":
                selected = (seqs if context == _CONTEXT_ALL
                            else [context or min(seqs)])
                if split_contexts:
                    for seq in selected:
                        calls[seq].add(PlanPoint(seq, 0, sq, batch))
                else:
                    for seq in selected:
                        calls["all"].add(PlanPoint(seq, 0, sq, batch))
            else:
                new = sq * cp
                calls[0].add(PlanPoint(new, 0, 1, batch))
                for seq in seqs:
                    if seq > new:
                        calls[seq].add(PlanPoint(seq, seq - new, 1, batch))
    return [sorted(pts, key=lambda p: (p.seq_len, p.s_prefix_cached,
                                       p.s_verify, p.batch))
            for _, pts in sorted(calls.items())]


def _points_file(cache: Path, model: str, gpu: str, parallel: str, phase: str,
                 asked: Sequence[Tuple[int, ...]]) -> Path:
    import hashlib
    asked = [(b, sq, point[2] if len(point) > 2 else 0)
             for point in asked for b, sq in (point[:2],)]
    # Pair-only payloads keep the cache names made by the first pass; a
    # context selector beyond "the representative" is part of the cache key.
    payload = ([(b, sq) for b, sq, context in asked if context == 0]
               if all(context == 0 for _b, _s, context in asked)
               else sorted(asked))
    tag = hashlib.sha1(json.dumps(payload).encode()).hexdigest()[:12]
    return (cache / "points" / gpu
            / f"{model.replace('/', '__')}__{parallel}__{phase}__{tag}.json")


def replan_points(task: Tuple[str, str, str, str,
                              Tuple[Tuple[int, int, int], ...], str]
                  ) -> Tuple[str, str, str, str, int, int]:
    """`(model, gpu, parallel, phase, calls planned, calls refused)`: one
    deployment's phase planned at its rows' own points (`own_points`) into
    the cache. A cached plan of the same points is kept."""
    model, gpu, parallel_sig, phase, asked, cache = task
    out = _points_file(Path(cache), model, gpu, parallel_sig, phase, list(asked))
    provenance = _provenance_file(out)
    if out.exists() and provenance.exists():
        return model, gpu, parallel_sig, phase, 0, 0
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.sweep_grid import ROUTINE_OUTPUT_TOKENS
    from kernel_bench.sweep import plan_points, resolve_sweep_model
    try:
        parallel = ParallelConfig.from_dict(
            {} if parallel_sig == "tp=1" else
            {k: int(v) for k, v in (kv.split("=") for kv in parallel_sig.split(","))})
        _config, resolved = resolve_sweep_model(model)
    except Exception:  # a deployment or model the engine refuses plans nothing
        _write_cells(out, [])
        return model, gpu, parallel_sig, phase, 0, 1
    jobs: List[dict] = []
    done = refused = 0
    calls = own_points(phase, parallel, list(asked))
    split_once = False
    at = 0
    while at < len(calls):
        points = calls[at]
        at += 1
        try:
            by_point = plan_points(model, gpu, parallel, phase, points,
                                   avg_output_tokens=ROUTINE_OUTPUT_TOKENS,
                                   resolved=resolved)
        except Exception:
            if phase == "decode" and not split_once:
                # The combined decode call is the fast path. Split only after
                # it refuses, so one context cannot lose the other contexts'
                # points; a model that refuses every context still counts all
                # of them below.
                split_once = True
                calls = own_points(phase, parallel, list(asked),
                                   split_contexts=True)
                at = 0
                continue
            refused += 1
            continue
        done += 1
        for walked, (point_jobs, _dropped) in by_point.items():
            jobs.extend(dict(job, _point=walked) for job in point_jobs)
    _write_cells(out, jobs)
    _write_provenance(out, jobs)
    return model, gpu, parallel_sig, phase, done, refused


def _moved(old: Optional[float], new: Optional[float], slack: float) -> bool:
    """Whether `new` differs from `old`, a numerator read back off rows
    whose longest p50 makes it uncertain by `slack`."""
    if new is None:
        return False
    if old is None:
        return True
    return abs(new - old) > max(_SAME * max(abs(old), abs(new)), slack)


def _load_cells(path: Path, gpu: str, priced: Dict[Tuple[str, Cell], set]) -> None:
    if path.exists():
        for op, shape, env, phase, pairs in json.loads(path.read_text()):
            priced[(gpu, (op, shape, env, phase))].update(
                _fold_pairs(tuple(p) for p in pairs))


def _load_provenance(path: Path, gpu: str,
                     provenance: _Provenance
                     ) -> None:
    if not path.exists():
        return
    fresh = provenance.fresh
    for row in json.loads(path.read_text()):
        model, parallel, node, op, shape, env, phase, pairs = row[:8]
        point = row[8] if len(row) > 8 else None
        key = (gpu, model, parallel, node,
               (op, shape, env, phase))
        provenance.rows[key].update(_fold_pairs(tuple(p) for p in pairs))
        if point is not None and point[1] == 0:
            fresh[key].update(_fold_pairs(tuple(p) for p in pairs))


def _priced(gpu: str, g: dict, priced: Dict[Tuple[str, Cell], set]) -> set:
    return _fold_pairs(priced.get(
        (gpu, cell_of(g["op"], g["shape_json"], g["dtype_env_json"],
                      g["phase"])), set()))


def _states_a_point(g: dict) -> bool:
    """Whether the row's own shape names the batch and query length a plan
    can be asked at (`own_points`)."""
    dims = json.loads(g["shape_json"])
    return (isinstance(dims.get("B"), int)
            and isinstance(dims.get("s_query"), int))


def _point_context(g: dict, seqs: set) -> int:
    """Which decode contexts a row's own cell can need (`own_points`). A
    shape that names no context axis is context-free: the physical key would
    be the same at every surface context. A named `s_kv` that is itself one
    of those contexts pins it; any other named cache length needs them all."""
    if g["phase"] != "decode":
        return 0
    dims = json.loads(g["shape_json"])
    cache_len = dims.get(_CONTEXT_AXIS)
    if cache_len is None:
        return 0
    if isinstance(cache_len, int) and cache_len in seqs:
        return cache_len
    return _CONTEXT_ALL


def _changed_axes(old: Dict[str, Any], new: Dict[str, Any]) -> set:
    """The axes whose presence or value differs between two shapes."""
    return ({k for k in old.keys() | new.keys()
             if old.get(k) != new.get(k)})


def _stale_reasons(unplanned: List[dict],
                   priced: Dict[Tuple[str, Cell], set]) -> List[dict]:
    """One diagnostic record per unplanned group, from today's own cells.

    This does not rename a key or choose a successor: it says which part of
    the identity is gone. `op-gone` names only the nearest current op as
    evidence, never as a mapping — a fused row may have replaced the old
    launch, or the old op may simply no longer be planned on that card."""
    by_op: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = (
        defaultdict(list))
    by_signature: Dict[Tuple[str, str, str], List[Tuple[str, Dict[str, Any]]]] = (
        defaultdict(list))
    by_axes: Dict[Tuple[str, str, frozenset],
                  Tuple[str, Dict[str, Any]]] = {}
    by_point: Dict[Tuple[str, str, str, str, Tuple[Tuple[str, Any], ...]],
                   Dict[str, Any]] = {}
    by_op_axes: Dict[Tuple[str, str, str, str, frozenset],
                     Dict[str, Any]] = {}
    by_routed_base: Dict[Tuple[str, str, str, str,
                                Tuple[Tuple[str, Any], ...]],
                          Dict[str, Any]] = {}
    for gpu, cell in priced:
        op, shape, env, phase = cell
        dims = json.loads(shape)
        by_op[(gpu, phase, op, env)].append(dims)
        by_signature[(gpu, phase, op)].append((env, dims))
        by_axes.setdefault((gpu, phase, frozenset(dims)), (op, dims))
        op_key = (gpu, phase, op, env)
        by_op_axes.setdefault((*op_key, frozenset(dims)), dims)
        by_point.setdefault((*op_key, tuple(sorted(
            (k, v) for k, v in dims.items() if k not in _POINT_AXES))), dims)
        by_routed_base.setdefault((*op_key, tuple(sorted(
            (k, v) for k, v in dims.items()
            if k not in _POINT_AXES | _ROUTED_AXES))), dims)

    out: List[dict] = []
    for g in unplanned:
        dims = json.loads(g["shape_json"])
        env = canonical_identity_json(g["dtype_env_json"])
        candidates = by_op.get((g["gpu"], g["phase"], g["op"], env), ())
        signature = by_signature.get((g["gpu"], g["phase"], g["op"]), ())
        record = {**{k: g.get(k, "") for k in (
                    "gpu", "model", "parallel", "op", "shape_json",
                    "dtype_env_json", "phase")},
                  "parallel": g.get("parallel", "tp=1"),
                  "node_spec_id": g.get("node_spec_id", ""),
                  "n": g["n"]}
        if not signature:
            axes = set(dims)
            nearest = by_axes.get((g["gpu"], g["phase"], frozenset(axes)))
            if nearest is None:
                nearest = max(
                    ((op, shape)
                     for (_gpu, phase, _axes), (op, shape) in by_axes.items()
                     if _gpu == g["gpu"] and phase == g["phase"]),
                    key=lambda item: (
                        len(axes & item[1].keys()) / len(axes | item[1].keys()),
                        -len(axes ^ item[1].keys())),
                    default=None)
            record["reason"] = "op-gone"
            if nearest is not None:
                record["nearest_op"] = nearest[0]
                record["nearest_shape_json"] = json.dumps(
                    nearest[1], sort_keys=True, separators=(",", ":"))
        else:
            if not candidates:
                record["reason"] = "dtype-gone"
            else:
                op_key = (g["gpu"], g["phase"], g["op"], env)
                point = tuple(sorted((k, v) for k, v in dims.items()
                                     if k not in _POINT_AXES))
                if (*op_key, point) in by_point:
                    shape = by_point[(*op_key, point)]
                    reason = "point-gone"
                    changed = _changed_axes(dims, shape)
                elif _ROUTED_AXES & dims.keys():
                    base = tuple(sorted((k, v) for k, v in dims.items()
                                        if k not in _POINT_AXES | _ROUTED_AXES))
                    shape = by_routed_base.get((*op_key, base), candidates[0])
                    reason = "routed-geometry-gone"
                    changed = _changed_axes(dims, shape)
                else:
                    reason = "shape-gone"
                    shape = by_op_axes.get((*op_key, frozenset(dims)),
                                           candidates[0])
                    changed = _changed_axes(dims, shape)
                record.update(reason=reason,
                              nearest_shape_json=json.dumps(
                                  shape, sort_keys=True, separators=(",", ":")),
                              changed_axes=sorted(changed))
        record["states_a_point"] = _states_a_point(g)
        out.append(record)
    return out


def plan_restamp(groups: List[dict], cache: Path, workers: int, *,
                 own_points: bool = True,
                 extra_precisions: Sequence[str] = (),
                 ) -> Tuple[List[dict], Counter, Dict[str, str], List[dict],
                            List[dict]]:
    """`(rewrites, counts, refused, moved, unplanned)`: each group of rows (one
    physical cell on one GPU) whose numerators today's plans move, with the
    new pair; row counts per outcome; the (model, GPU, surface) plans the
    engine refused; per moved group its utilization before and after; and each
    unplanned group with the part of its key today's cells no longer carry.

    Two stages. The surfaces first — every (model, GPU) over both — and
    then, for the rows they leave unpriced or whose physical cell is
    ambiguous without their own node's provenance, each (model, GPU,
    deployment, phase) at the rows' own points (`own_points`): a batch the
    capacity has since moved, a deployment the surface no longer plans, or a
    boundary form only the row's node identifies still has a cell today's
    walk prices."""
    tasks = sorted(
        {(g["model"], g["gpu"], s, p, str(cache))
         for g in groups for s in SURFACES
         for p in precisions(extra_precisions)},
        key=lambda t: (t[0], t[1], t[2], t[3] or ""))
    refused: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (model, gpu, surface, precision, why) in enumerate(
                pool.map(replan, tasks), 1):
            at = f" {precision}" if precision else ""
            if why:
                refused[f"{model} @ {gpu} ({surface}{at})"] = why
            print(f"  [{i}/{len(tasks)}] {model} @ {gpu} {surface}{at}"
                  + (f": {why}" if why else ""), flush=True)
    priced: Dict[Tuple[str, Cell], set] = defaultdict(set)
    exact = _Provenance()
    for model, gpu, surface, precision, _ in tasks:
        path = _plan_file(cache, model, gpu, surface, precision)
        _load_cells(path, gpu, priced)
        _load_provenance(_provenance_file(path), gpu, exact)

    if own_points:
        asked: Dict[Tuple[str, str, str, str], set] = defaultdict(set)
        seqs = _surface_seqs()
        for g in groups:
            pairs = _priced(g["gpu"], g, priced)
            selector = (g["gpu"], g["model"], g["parallel"],
                        g.get("node_spec_id", ""),
                        cell_of(g["op"], g["shape_json"], g["dtype_env_json"],
                                g["phase"]))
            exact_pairs = (_fold_pairs(exact.rows.get(selector, set()))
                           if len(pairs) > 1 else {()})
            fresh_pairs = (_fold_pairs(exact.fresh.get(selector, set()))
                           if len(pairs) > 1 else {()})
            if pairs and (len(exact_pairs) == 1 or len(fresh_pairs) == 1):
                continue
            if _states_a_point(g):
                dims = json.loads(g["shape_json"])
                asked[(g["model"], g["gpu"], g["parallel"], g["phase"])].add(
                    (dims["B"], dims["s_query"], _point_context(g, seqs)))
        point_tasks = sorted(
            ((*k, tuple(sorted(v)), str(cache)) for k, v in asked.items()),
            key=lambda t: -len(t[4]))
        print(f"own points: {len(point_tasks)} deployment-phases, "
              f"{sum(len(t[4]) for t in point_tasks)} (B, s_query) pairs",
              flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (model, gpu, par, phase, done, bad) in enumerate(
                    pool.map(replan_points, point_tasks), 1):
                if bad:
                    refused[f"{model} @ {gpu} {par} {phase} (own points)"] = (
                        f"{bad} of {done + bad} plan calls refused")
                if i % 50 == 0 or i == len(point_tasks):
                    print(f"  [{i}/{len(point_tasks)}] own points", flush=True)
    for model, gpu, par, phase, pts, _ in point_tasks:
        path = _points_file(cache, model, gpu, par, phase, list(pts))
        _load_cells(path, gpu, priced)
        _load_provenance(_provenance_file(path), gpu, exact)

        # The first point pass can make a cell ambiguous: one deployment's
        # point adds a second price to a cell another deployment already
        # answered. Ask the row's own deployment then, now that its node is
        # known to need provenance.
    if own_points:
        asked = defaultdict(set)
        for g in groups:
            pairs = _priced(g["gpu"], g, priced)
            if len(pairs) <= 1 or not _states_a_point(g):
                continue
            selector = (g["gpu"], g["model"], g["parallel"],
                        g.get("node_spec_id", ""),
                        cell_of(g["op"], g["shape_json"],
                                g["dtype_env_json"], g["phase"]))
            own = _fold_pairs(exact.rows.get(selector, set()))
            fresh = _fold_pairs(exact.fresh.get(selector, set()))
            if len(own) == 1 or len(fresh) == 1:
                continue
            dims = json.loads(g["shape_json"])
            asked[(g["model"], g["gpu"], g["parallel"], g["phase"])].add(
                (dims["B"], dims["s_query"], _point_context(g, seqs)))
        fallback_tasks = sorted(
            ((*key, tuple(sorted(points)), str(cache))
             for key, points in asked.items()),
            key=lambda task: -len(task[4]))
        if fallback_tasks:
            print(
                f"exact-node points: {len(fallback_tasks)} deployment-phases, "
                f"{sum(len(task[4]) for task in fallback_tasks)} points",
                flush=True)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for i, (model, gpu, par, phase, done, bad) in enumerate(
                        pool.map(replan_points, fallback_tasks), 1):
                    if bad:
                        refused[f"{model} @ {gpu} {par} {phase} "
                                "(exact-node points)"] = (
                                    f"{bad} of {done + bad} plan calls refused")
                    if (i % 50 == 0 or i == len(fallback_tasks)):
                        print(
                            f"  [{i}/{len(fallback_tasks)}] exact-node points",
                            flush=True)
            for model, gpu, par, phase, pts, _ in fallback_tasks:
                path = _points_file(cache, model, gpu, par, phase,
                                    list(pts))
                _load_cells(path, gpu, priced)
                _load_provenance(_provenance_file(path), gpu, exact)

    rewrites: List[dict] = []
    moved: List[dict] = []
    unplanned: List[dict] = []
    counts: Counter = Counter()
    for g in groups:
        pairs = _priced(g["gpu"], g, priced)
        selector = (g["gpu"], g["model"], g["parallel"],
                    g.get("node_spec_id", ""),
                    cell_of(g["op"], g["shape_json"], g["dtype_env_json"],
                            g["phase"]))
        resolved = False
        if not pairs:
            counts["unplanned"] += g["n"]
            unplanned.append(g)
            continue
        if len(pairs) > 1:
            own = _fold_pairs(exact.rows.get(selector, set()))
            if len(own) == 1:
                pairs = own
                resolved = True
            else:
                if not own:
                    counts["provenance-gone"] += g["n"]
                    moved.append({
                        **{c: g[c] for c in ("gpu", "model", "parallel",
                                              "node_spec_id", "op", "shape_json",
                                              "dtype_env_json", "phase")},
                        "n": g["n"], "provenance_gone": True})
                    continue
                fresh = _fold_pairs(exact.fresh.get(selector, set()))
                if len(fresh) == 1:
                    pairs = fresh
                    resolved = True
                else:
                    counts["ambiguous"] += g["n"]
                    moved.append({
                        **{c: g[c] for c in ("gpu", "model", "parallel",
                                              "node_spec_id", "op", "shape_json",
                                              "dtype_env_json", "phase")},
                        "n": g["n"], "ambiguous": True})
                    continue
        flops, nbytes = next(iter(pairs))
        p50 = g["max_p50_us"]
        if not (_moved(g["old_flops"], flops, _READBACK["flops"] * p50)
                or _moved(g["old_bytes"], nbytes, _READBACK["bytes"] * p50)):
            counts["unchanged"] += g["n"]
            continue
        key = {c: g[c] for c in ("gpu", "op", "shape_json", "dtype_env_json", "phase")}
        if resolved:
            key.update(model=g["model"], parallel=g["parallel"],
                       node_spec_id=g["node_spec_id"])
        ratio_b = (nbytes / g["old_bytes"]) if g["old_bytes"] and nbytes is not None else None
        ratio_f = (flops / g["old_flops"]) if g["old_flops"] and flops is not None else None
        change = {**key, "n": g["n"],
                  "mbu": g["max_mbu"], "mfu": g["max_mfu"],
                  "new_mbu": None if ratio_b is None or g["max_mbu"] is None
                  else g["max_mbu"] * ratio_b,
                  "new_mfu": None if ratio_f is None or g["max_mfu"] is None
                  else g["max_mfu"] * ratio_f}
        if resolved:
            change["resolved_provenance"] = True
        # The write gate's rule (`emit.above_sol`): a numerator that makes a
        # measured launch faster than the card is not what that launch
        # moved — today's law over-bills it, or the launch skipped work the
        # cell has. Either way it is a finding, not a number to stamp.
        if max(change["new_mbu"] or 0, change["new_mfu"] or 0) > UTILIZATION_CEILING:
            counts["above_peak"] += g["n"]
            change["refused"] = True
            moved.append(change)
            continue
        counts["restamped"] += g["n"]
        rewrites.append({**key, "flops_per_rank": flops,
                         "access_bytes_per_rank": nbytes})
        moved.append(change)
    if unplanned:
        no_point = [g for g in unplanned if not _states_a_point(g)]
        by_op: Counter = Counter()
        for g in unplanned:
            by_op[(g["gpu"], g["op"])] += g["n"]
        print(f"unplanned after {'own points' if own_points else 'surfaces'}: "
              f"{len(unplanned)} groups, "
              f"{sum(g['n'] for g in unplanned)} rows "
              f"({len(no_point)} groups state no (B, s_query) point)", flush=True)
        for (gpu, op), n in by_op.most_common(20):
            print(f"  {n:8d}  {op:32s} {gpu}", flush=True)
        hint = unplanned_hint(unplanned, extra_precisions)
        if hint:
            print(f"  {hint}", flush=True)
    stale = _stale_reasons(unplanned, priced)
    if stale:
        reasons: Counter = Counter()
        for row in stale:
            reasons[(row["gpu"], row["reason"], row["op"])] += row["n"]
        print("unplanned, by reason:")
        for (gpu, reason, op), n in reasons.most_common(30):
            print(f"  {n:8d}  {reason:24s} {op:32s} {gpu}", flush=True)
    return rewrites, counts, refused, moved, stale


def report(counts: Counter, refused: Dict[str, str], moved: List[dict]) -> None:
    for why, n in counts.most_common():
        print(f"{why:>10}: {n} rows")
    for what, why in sorted(refused.items()):
        print(f"  not planned: {what}: {why}")
    ambiguous = [m for m in moved if m.get("ambiguous")]
    gone = [m for m in moved if m.get("provenance_gone")]
    resolved = [m for m in moved if m.get("resolved_provenance")]
    if ambiguous:
        print(f"\nstill ambiguous: {len(ambiguous)} groups, "
              f"{sum(m['n'] for m in ambiguous)} rows "
              "(today prices this exact node two ways)")
        reasons: Counter = Counter()
        for m in ambiguous:
            reasons[(m["gpu"], m["phase"], m["op"])] += m["n"]
        for (gpu, phase, op), n in reasons.most_common(20):
            print(f"  {n:8d}  {op:32s} {phase:8s} {gpu}")
    if gone:
        print(f"\nno exact node provenance: {len(gone)} groups, "
              f"{sum(m['n'] for m in gone)} rows "
              "(another deployment's plan answered the physical cell)")
    if resolved:
        print(f"\nresolved by node provenance: {len(resolved)} groups, "
              f"{sum(m['n'] for m in resolved)} rows")
    by_op: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for m in moved:
        if not any(m.get(flag) for flag in
                   ("refused", "ambiguous", "provenance_gone")):
            by_op[(m["op"], m["gpu"])].append(m)
    print("\nrestamped, by op (rows; groups whose worst MBU/MFU was over 1.0 before -> after):")
    for (op, gpu), ms in sorted(by_op.items(), key=lambda kv: -sum(m["n"] for m in kv[1])):
        over = [m for m in ms if max(m["mbu"] or 0, m["mfu"] or 0) > 1.0]
        after = [m for m in ms if max(m["new_mbu"] or 0, m["new_mfu"] or 0) > 1.0]
        ratio = [m["new_mbu"] / m["mbu"] for m in ms if m["mbu"] and m["new_mbu"] is not None]
        med = f"{statistics.median(ratio):.3f}" if ratio else "-"
        print(f"  {sum(m['n'] for m in ms):8d}  {op:32s} {gpu:9s} median new/old bw {med}"
              f"  over peak {len(over)} -> {len(after)}")
    for label, picked in (
            ("restamped group(s) still claim more than the card's peak "
             "(within the write gate's ceiling)",
             [m for m in moved if not m.get("refused")
              and not m.get("ambiguous")
              and not m.get("provenance_gone")
              and max(m["new_mbu"] or 0, m["new_mfu"] or 0) > 1.0]),
            ("group(s) NOT restamped: today's numerator puts the launch above "
             f"{UTILIZATION_CEILING}x the card's peak",
             [m for m in moved if m.get("refused")])):
        if not picked:
            continue
        print(f"\n{len(picked)} {label}:")
        for m in sorted(picked, key=lambda m: -max(m["new_mbu"] or 0, m["new_mfu"] or 0))[:40]:
            print(f"  {m['gpu']} {m['op']} {m['phase']} {m['shape_json']} "
                  f"MBU {m['mbu'] or 0:.3f} -> {m['new_mbu'] or 0:.3f} "
                  f"MFU {m['mfu'] or 0:.3f} -> {m['new_mfu'] or 0:.3f}")


def apply(rewrites: List[dict]) -> int:
    """Rewrite the planned rows' achieved rates from their own p50, through
    `store.restamp`, owner by owner; returns the rows updated.

    A rewrite with a model/deployment/node selector came from an ambiguous
    physical cell resolved by that provenance, and touches only its own
    rows. A cell-only rewrite remains the common fast path."""
    from backend.service.bench import store
    from backend.service.bench.lake import Lake, LakeConfig
    from kernel_bench.emit import achieved_from_latency
    if not rewrites:
        return 0
    # One read over the whole lake: more than the service's budget.
    lake = Lake.open(replace(LakeConfig.from_env(), memory_limit="4GB"))
    try:
        lake.require_free("restamping achieved rates")
        patches: Dict[str, List[dict]] = defaultdict(list)

        def patch(targets, by_key, key_of):
            for t in targets:
                r = by_key[key_of(t)]
                tflops, bw = achieved_from_latency(
                    r["flops_per_rank"], r["access_bytes_per_rank"],
                    float(t["p50_us"]))
                patches[t["owner"]].append({
                    "id": t["id"],
                    "achieved_tflops": float(tflops) if tflops else None,
                    "achieved_bw_gbps": float(bw) if bw else None})

        cell_only = [r for r in rewrites if not all(
            k in r for k in ("model", "parallel", "node_spec_id"))]
        selected = [r for r in rewrites if all(
            k in r for k in ("model", "parallel", "node_spec_id"))]
        if cell_only:
            by_cell = {(r["gpu"], r["op"], r["shape_json"],
                        r["dtype_env_json"], r["phase"]): r
                       for r in cell_only}
            targets = lake.read(
                "SELECT m.id::VARCHAR AS id, b.owner::VARCHAR AS owner, m.p50_us, "
                "m.gpu, m.op, m.shape_json, m.dtype_env_json, m.phase "
                "FROM lake.measurements m JOIN lake.batches b USING (batch_id) "
                "JOIN (SELECT unnest(?::VARCHAR[]) AS gpu, unnest(?::VARCHAR[]) AS op, "
                "unnest(?::VARCHAR[]) AS shape_json, unnest(?::VARCHAR[]) AS dtype_env_json, "
                "unnest(?::VARCHAR[]) AS phase) r USING (gpu, op, shape_json, dtype_env_json, phase) "
                "WHERE m.latency_us IS NOT NULL AND m.p50_us > 0",
                [[k[i] for k in by_cell] for i in range(5)])
            patch(targets, by_cell, lambda t: (
                t["gpu"], t["op"], t["shape_json"], t["dtype_env_json"],
                t["phase"]))
        if selected:
            by_selected = {(r["gpu"], r["model"], r["parallel"],
                            r["node_spec_id"], r["op"], r["shape_json"],
                            r["dtype_env_json"], r["phase"]): r
                           for r in selected}
            targets = lake.read(
                "SELECT m.id::VARCHAR AS id, b.owner::VARCHAR AS owner, m.p50_us, "
                "m.gpu, m.model, m.parallel, m.node_spec_id, m.op, "
                "m.shape_json, m.dtype_env_json, m.phase "
                "FROM lake.measurements m JOIN lake.batches b USING (batch_id) "
                "JOIN (SELECT unnest(?::VARCHAR[]) AS gpu, unnest(?::VARCHAR[]) AS model, "
                "unnest(?::VARCHAR[]) AS parallel, unnest(?::VARCHAR[]) AS node_spec_id, "
                "unnest(?::VARCHAR[]) AS op, unnest(?::VARCHAR[]) AS shape_json, "
                "unnest(?::VARCHAR[]) AS dtype_env_json, unnest(?::VARCHAR[]) AS phase) r "
                "USING (gpu, model, parallel, node_spec_id, op, shape_json, "
                "dtype_env_json, phase) "
                "WHERE m.latency_us IS NOT NULL AND m.p50_us > 0",
                [[k[i] for k in by_selected] for i in range(8)])
            patch(targets, by_selected, lambda t: (
                t["gpu"], t["model"], t["parallel"], t["node_spec_id"],
                t["op"], t["shape_json"], t["dtype_env_json"], t["phase"]))
        done = 0
        for owner, rows in patches.items():
            for start in range(0, len(rows), _CHUNK):
                done += store.restamp(lake, owner=owner, rows=rows[start:start + _CHUNK])
                print(f"  {owner}: {done} rows", flush=True)
        return done
    finally:
        lake.close()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="write the planned rewrites to this JSON file")
    ap.add_argument("--apply", metavar="REWRITES",
                    help="apply a planned JSON file (the Kernel Bench service stopped)")
    ap.add_argument("--cache", default=".cache/restamp_achieved_plans",
                    help="where each (model, GPU, surface) plan is kept")
    ap.add_argument("--workers", type=int, default=8,
                    help="plans run at once (one process each)")
    ap.add_argument("--model", action="append", help="only these models (repeatable)")
    ap.add_argument("--precision", action="append", default=[], metavar="PRESET",
                    help="also plan every surface under this precision preset "
                         "(repeatable). A row measured under a non-shipped "
                         "precision (`kernel_bench sweep --precision`) has "
                         "dtypes no as-shipped cell carries, so it is "
                         "unplannable without the same preset named here. "
                         "Not the default because a row does not record the "
                         "profile it was measured under, so there is nothing "
                         "to derive, and each extra preset re-plans every "
                         "(model, GPU, surface).")
    ap.add_argument("--stale-only", action="store_true",
                    help="classify surface leftovers without the own-point "
                         "replanning pass (a diagnosis, never an apply plan)")
    args = ap.parse_args(argv)
    if args.apply:
        with open(args.apply) as f:
            print(f"applied: {apply(json.load(f)['rewrites'])} rows")
        return 0

    from tools.lake_rows import query
    where, params = "", []
    if args.model:
        where, params = "AND model IN (SELECT unnest(?::VARCHAR[]))", [args.model]
    # Old numerators read back off the row: achieved × p50. Every row of a
    # group was stamped from one job, so the group's max is any row's value
    # up to the four-decimal rounding of the column.
    groups = query(
        "SELECT gpu, model, parallel, node_spec_id, op, shape_json, dtype_env_json, phase, count(*) AS n, "
        "max(achieved_tflops * p50_us * 1e6) AS old_flops, "
        "max(achieved_bw_gbps * p50_us * 1e3) AS old_bytes, "
        "max(achieved_bw_gbps / nullif(peak_bw_at_test, 0)) AS max_mbu, "
        "max(achieved_tflops / nullif(peak_tflops_at_test, 0)) AS max_mfu, "
        "max(p50_us) AS max_p50_us "
        "FROM lake.measurements WHERE latency_us IS NOT NULL AND p50_us > 0 "
        f"{where} GROUP BY ALL", params)
    print(f"{sum(g['n'] for g in groups)} measured rows in {len(groups)} groups", flush=True)
    (rewrites, counts, refused, moved, unplanned) = plan_restamp(
        groups, Path(args.cache), args.workers,
        own_points=not args.stale_only,
        extra_precisions=tuple(args.precision))
    report(counts, refused, moved)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rewrites": rewrites, "counts": counts,
                       "refused": refused, "moved": moved,
                       "unplanned": unplanned}, f)
        print(f"wrote {len(rewrites)} group rewrites to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
