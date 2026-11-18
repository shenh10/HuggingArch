"""The sweep parity harness: a swept point against a direct estimate at the
same point, and a Kernel Bench stand-in for the measured side
(docs/auto_sweep.md §3.5).

A sweep prices many points off few walks: one walk per cell (model, layout,
phase, precision, pp stage), then each point (workload counts, card) by
substitution into
the row laws. The contract is that a swept point is the estimate a direct
call at that point returns. This module is the one statement of "the same":

  * `diff(a, b)` — every difference between two results: integers, bools,
    strings and None exactly; floats to a relative 1e-12 (substitution sums in
    another order than a walk, which moves the last bits and nothing else);
    a key or a list element one side lacks is a difference too.
  * `Deployment` — an estimate's arguments from a corpus snapshot (no network),
    and the direct estimate at a point (`direct`).
  * `rows_at(rows, batch, card)` — per-op rows re-read at another batch and
    card through the row laws, the per-point rows a sweep stands on; the
    sweep payloads drop per-op rows (`sweep_runtime.served_sweep_point`),
    so the rows are compared here, not through the API.
  * `FixtureBench` — a Kernel Bench service in process (`measure.lookup`'s
    client pointed at an `httpx.MockTransport`): each key it is asked is
    answered by a policy (measured, missing, ambiguous), and every request
    is counted, so a test can say how many lookups a sweep made.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

from tests.snapshots import snapshot_config

#: Relative tolerance for floats: substitution evaluates the same law as the
#: walk, in another summation order.
REL = 1e-12


@dataclass(frozen=True)
class Difference:
    path: str
    kind: str          # "number" | "value" | "type" | "missing" | "length"
    a: Any
    b: Any

    def __str__(self) -> str:
        return f"{self.path}: {self.kind} {self.a!r} != {self.b!r}"


def _close(x: float, y: float, rel: float) -> bool:
    if x == y:
        return True
    if math.isnan(x) and math.isnan(y):
        return True
    return abs(x - y) <= rel * max(abs(x), abs(y))


def diff(a: Any, b: Any, *, rel: float = REL, path: str = "") -> List[Difference]:
    """Every difference between `a` and `b` (module docstring). Two integers
    are equal only when equal; a float against a number is equal within
    `rel` (a sum that one side keeps as an int and the other as a float is
    the same value)."""
    out: List[Difference] = []
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a or k not in b:
                out.append(Difference(p, "missing", a.get(k, "<absent>"),
                                      b.get(k, "<absent>")))
                continue
            out.extend(diff(a[k], b[k], rel=rel, path=p))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(Difference(path, "length", len(a), len(b)))
        # A length difference is recorded above; the common prefix still diffs.
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            out.extend(diff(x, y, rel=rel, path=f"{path}[{i}]"))
        return out
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a is not b and a != b:
            out.append(Difference(path, "value", a, b))
        return out
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            out.append(Difference(path, "type", a, b))
        elif not _close(float(a), float(b), rel):
            out.append(Difference(path, "number", a, b))
        return out
    if type(a) is not type(b):
        out.append(Difference(path, "type", a, b))
    elif a != b:
        out.append(Difference(path, "value", a, b))
    return out


def report(diffs: Sequence[Difference], limit: int = 25) -> str:
    """The first `limit` differences, one per line, for an assertion message."""
    lines = [str(d) for d in diffs[:limit]]
    if len(diffs) > limit:
        lines.append(f"... and {len(diffs) - limit} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A deployment, and the direct estimate at one of its points.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Deployment:
    """An estimate's seq-invariant arguments for a corpus snapshot: what
    `backend.api.estimate_env.resolve_estimate_environment` builds for a
    request, without fetching (the snapshot is the record)."""
    model_id: str
    parallel: Any = None              # ParallelConfig; None is TP1
    gpu_sku: str = "H100-SXM"
    avg_output_tokens: int = 256
    selection_mode: str = "theoretical_only"
    measurement_scope: Any = None
    gpu_memory_fraction: float = 0.9
    knobs: Mapping[str, Any] = field(default_factory=dict)

    def env(self) -> Dict[str, Any]:
        """The flat estimate arguments (`estimate_capacity` / the sweeps)."""
        from backend.inference.parallel_config import ParallelConfig
        parallel = self.parallel if self.parallel is not None else ParallelConfig()
        return dict(
            config=snapshot_config(self.model_id), model_id=self.model_id,
            gpu_sku=self.gpu_sku, gpu_memory_fraction=self.gpu_memory_fraction,
            num_gpus=parallel.cluster_size, parallel=parallel,
            avg_output_tokens=self.avg_output_tokens,
            selection_mode=self.selection_mode,
            measurement_scope=self.measurement_scope, **dict(self.knobs))

    def on(self, gpu_sku: str) -> "Deployment":
        return replace(self, gpu_sku=gpu_sku)

    def direct(self, *, avg_input_tokens: int,
               prefill_batch: Optional[int] = None,
               decode_batch: Optional[int] = None) -> dict:
        """The direct estimate at one point: its own walk, its own lookup."""
        from backend.inference.capacity import estimate_capacity
        return estimate_capacity(
            **self.env(), avg_input_tokens=avg_input_tokens,
            prefill_batch_size=prefill_batch, decode_batch_size=decode_batch)


#: The per-op row lists an estimate carries, by phase.
ROW_LISTS = ("per_op_timing", "per_op_timing_decode", "per_op_timing_draft")


#: The context a parity layout walks BEFORE the one it is asked for.
#: A sweep prices many contexts off few walks: the first context of a
#: stage takes the walk and every later one substitutes its length into
#: that walk's laws (`capacity.ContextAnchors.at`, `own` False). Asked for
#: one context only, the oracle compares the anchor walk itself and never
#: prices a substitution — the mechanism every swept point past the first
#: context stands on, and the one this harness exists to hold to a direct
#: estimate (CLAUDE.md §1.5: cross-point repricing is compared row for
#: row, not by `t_sol` alone).
WARM_UP_CONTEXT = 512


def layout_at(payload: Mapping[str, Any], *,
              warm_up: int = WARM_UP_CONTEXT) -> Dict[str, dict]:
    """`parallel_sweep._estimate_layout` at the context a direct estimate's
    arguments name (`avg_input_tokens`), read off a walk taken at ANOTHER
    context (`warm_up`, priced first): `{card: entry}`, each
    `{"context", "sol", "measures", "ladder"}` (or the card's error).

    So the entry compared is a substituted point, which is what a swept
    point is. `warm_up=<the asked-for context>` asks for the anchor walk
    itself, for a caller that wants to compare that."""
    from backend.inference import parallel_sweep as ps
    want = int(payload["avg_input_tokens"])
    body = {k: v for k, v in payload.items() if k != "avg_input_tokens"}
    contexts = [int(warm_up), want] if int(warm_up) != want else [want]
    out = ps._estimate_layout({**body, "contexts": contexts})
    at = contexts.index(want)
    return {card: entries[at] for card, entries in out.items()}


def response_cell(book: Any = None) -> Any:
    """The walk a response's rows are read off, as `batch_sweep.rows_at`
    and `phase_at` take it (`cell`): their laws, read back off their text
    into a book of their own (`LawBook`, or `book`). A response carries no
    walk to admit a point against."""
    from types import SimpleNamespace

    from backend.inference.row_roofline import LawBook
    return SimpleNamespace(laws=SimpleNamespace(book=book or LawBook()))


def rescale_rows(rows: List[dict], counts: Mapping[str, int],
                 peak_flops: Optional[float], peak_bw: Optional[float],
                 fabric: Any, cell: Any, *, breakdown: bool,
                 launch_s: float = 0.0) -> List[dict]:
    """`batch_sweep.rows_at` at the point `counts` moves the rows to."""
    from backend.arch.eval.analysis import Point
    from backend.inference.batch_sweep import rows_at as read_rows
    return read_rows(rows, Point(counts), cell=cell, peak_flops=peak_flops,
                     peak_bw=peak_bw, fabric=fabric, breakdown=breakdown,
                     launch_s=launch_s)


def rescale_phase(phase: Dict[str, Any], rows: List[dict],
                  counts: Mapping[str, int], peak_flops: Optional[float],
                  peak_bw: Optional[float], fabric: Any, *, cell: Any,
                  breakdown: bool, draft_rows: Optional[List[dict]] = None,
                  draft_cell: Any = None, launch_s: float = 0.0) -> Dict[str, Any]:
    """`batch_sweep.phase_at`, with the phase's rows (and a draft's, off
    `draft_cell`) read at `counts` on the card first (`rescale_rows`)."""
    from backend.arch.eval.analysis import Point
    from backend.inference.batch_sweep import phase_at
    if not phase or not phase.get("batch"):
        return phase
    return phase_at(
        phase, rescale_rows(rows, counts, peak_flops, peak_bw, fabric, cell,
                            breakdown=breakdown, launch_s=launch_s),
        Point(counts), peak_flops, peak_bw, fabric, cell=cell,
        breakdown=breakdown,
        draft_rows=(rescale_rows(draft_rows, counts, peak_flops, peak_bw,
                                 fabric, draft_cell, breakdown=breakdown,
                                 launch_s=launch_s)
                    if draft_rows else draft_rows),
        launch_s=launch_s)


def rows_at(rows: List[dict], batch: int, *, card: str,
            breakdown: bool) -> List[dict]:
    """`rows` (walked at some batch on some card) read at `batch` on `card`,
    through each row's laws: its shape law (`batch_sweep.rows_at`) and its
    roofline on the card — every row's own
    peak re-read for its `act_dtype`, the collective on the card's fabric
    (`capacity._stamp_gpu_on_result`'s reading). The rows are a response's,
    so their laws are read off their text (`LawBook`). `breakdown` as
    `batch_sweep.rows_at`'s."""
    from backend.inference.comm_cost import Fabric
    from backend.inference.defaults import (COMM_LATENCY_US_DEFAULT,
                                            LAUNCH_FLOOR_US_DEFAULT)
    from backend.inference.gpu_specs import get_gpu, peak_hbm_bw
    from backend.inference.pipeline import leaf_peak
    gpu = get_gpu(card)
    on_card = []
    for r in rows:
        adt = r.get("act_dtype")
        peak = leaf_peak(gpu, adt) if adt else None
        on_card.append({**r, "peak_flops_per_sec_for_op": float(peak) if peak else 0.0})
    return rescale_rows(on_card, {"B": batch}, None, peak_hbm_bw(gpu),
                        Fabric.of(gpu, COMM_LATENCY_US_DEFAULT * 1e-6),
                        response_cell(), breakdown=breakdown,
                        launch_s=LAUNCH_FLOOR_US_DEFAULT * 1e-6)


#: The per-op columns a row's laws state at any point: its roofline
#: (`row_roofline.EXPR_NAMES`, as `batch_sweep.rows_at` writes them back)
#: and its shape (`shape_law`). What a sweep reads off a row at a point; the
#: rest of a row (formula text, wire dims, per-tensor byte splits) is display
#: and is compared by the full-row check.
LAW_COLUMNS = ("flops_per_rank", "access_bytes_per_rank", "act_out_bytes_per_rank",
               "t_compute_s", "t_memory_s", "t_comm_s", "t_sol_s", "t_op_s", "bound",
               "arithmetic_intensity")
SHAPE_COLUMNS = ("shape_in", "shape_in_exact", "shape_out", "shape_outs")


def without_breakdown(result: Mapping[str, Any], *, keyed: bool) -> Dict[str, Any]:
    """`result` as a point read with no breakdown shows it (`batch_sweep.
    rows_at` / `phase_at` with `breakdown=False`): each phase without its
    module table, its total and its formulas; each row without its formula
    popovers, a collective's per-phase rows, link and buffer shapes, a
    fused kernel's absorbed parts' output shapes, the access facts its
    popover prints (`point_law["access"]`) — and, unless the point
    was `keyed` (its request looks measurements up), what its key is read
    off (`node_dims`, `comm_payload_shape`: read where it is keyed)."""
    def phase(ph: Any) -> Any:
        if not isinstance(ph, Mapping):
            return ph
        return {k: phase(v) for k, v in ph.items()
                if k not in ("flops_by_module", "formulas", "op_table_total")}

    def row(r: Mapping[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in r.items()
               if not k.endswith("_formula")
               and k not in ("comm_phases", "comm_bw_bytes_per_sec", "comm_bw_label",
                             "comm_collective_in_shape", "comm_collective_out_shape")
               and (keyed or k != "node_dims")
               and (keyed or k != "comm_payload_shape" or not r.get("is_comm"))}
        if out.get("point_law") is not None:
            # The access facts its popover prints (`point_law["access"]`).
            out["point_law"] = {**out["point_law"], "access": None}
        badge = out.get("source_badge") or {}
        if badge.get("absorbed"):
            out["source_badge"] = {**badge, "absorbed": [
                {k: v for k, v in part.items() if k != "shape_out"}
                for part in badge["absorbed"]]}
        return out

    out = dict(result)
    if "phases" in out:
        out["phases"] = {k: phase(v) for k, v in out["phases"].items()}
    for lst in ROW_LISTS:
        if out.get(lst) is not None:
            out[lst] = [row(r) for r in out[lst]]
    return out


def law_view(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Each row's `uid` and the columns its laws state (`LAW_COLUMNS`,
    `SHAPE_COLUMNS`)."""
    return [{"uid": r.get("uid"), **{k: r.get(k) for k in (*LAW_COLUMNS, *SHAPE_COLUMNS)}}
            for r in rows]


def phase_scalars(phase: Mapping[str, Any]) -> Dict[str, Any]:
    """A phase's numbers and verdicts: every top-level field that is a number,
    a bool or None, and the same one level into the dicts that hold numbers
    (`binding`, `t_sol_by_bound`, `throughput`'s rates, `pipeline_time`'s
    sums). Text (`note`, `formulas`) and the module table with its total are
    display, compared by the full-phase check."""
    def scalar(v: Any) -> bool:
        return v is None or isinstance(v, (bool, int, float))

    def numbers(d: Any) -> Any:
        if isinstance(d, Mapping):
            out = {k: numbers(v) for k, v in d.items()
                   if scalar(v) or isinstance(v, Mapping)}
            return {k: v for k, v in out.items() if v != {}}
        return d
    out: Dict[str, Any] = {}
    for k, v in phase.items():
        if k in ("formulas", "flops_by_module", "op_table_total", "note"):
            continue
        if scalar(v):
            out[k] = v
        elif isinstance(v, Mapping):
            got = numbers(v)
            if got:
                out[k] = got
    return out


# ---------------------------------------------------------------------------
# A Kernel Bench service in process.
# ---------------------------------------------------------------------------


#: What a policy answers for one key: `measured` (one row), `missing` (none),
#: `ambiguous` (two rows of the framework, two kernels — `latest` cannot pick).
MEASURED, MISSING, AMBIGUOUS = "measured", "missing", "ambiguous"


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


class FixtureBench:
    """A Kernel Bench service in process (module docstring).

    `policy(key) -> MEASURED | MISSING | AMBIGUOUS` decides each key's answer
    (default: every key measured). A measured key's latency is a fixed
    function of the key (`latency_us`), so two runs that ask the same key read
    the same time. `lookups` counts the lookups made under a reading scope
    (`measure.lookup.lookup` calls — what "one lookup per sweep" counts);
    `requests` the HTTP requests that reached the service (a key the client
    already holds is not asked again); `keys_asked` every key the service was
    asked, in order.

    Use as a context manager around the code under test; it points the lookup
    client at the stand-in and restores it after. `scope` is the measurement
    scope an estimate passes to read it (`measurement_scope=`)."""

    def __init__(self, policy: Optional[Callable[[Any], str]] = None, *,
                 framework: str = "vllm"):
        from backend.measure.scope import MeasurementScope
        self.policy = policy or (lambda key: MEASURED)
        self.framework = framework
        self.scope = MeasurementScope(owner=None, lake_id="fixture-lake", snapshot=1,
                                      shared_snapshot=1, rules_rev="fixture")
        self.requests = 0
        self.lookups = 0
        self.keys_asked: List[Any] = []
        self._saved: Dict[str, Any] = {}

    @staticmethod
    def latency_us(key: Any, impl: str = "k1") -> float:
        """The measured latency the stand-in reports for `key`."""
        return 1.0 + _seed(key.signature, key.shape_json, key.dtype_env_json,
                           key.gpu, key.phase, impl) % 997 / 10.0

    def _row(self, key: Any, impl: str, base: Dict[str, str]) -> Dict[str, str]:
        from backend.inference.gpu_specs import get_gpu, peak_hbm_bw
        gpu = get_gpu(key.gpu)
        bw = (peak_hbm_bw(gpu) or 0.0) / 1e9 if gpu is not None else 3350.0
        return {**base, "gpu": key.gpu, "phase": key.phase, "op": key.signature,
                "shape_json": key.shape_json, "dtype_env_json": key.dtype_env_json,
                "framework": self.framework, "kernel_impl": impl,
                "node_spec_id": f"n{_seed(key.signature, key.shape_json, key.gpu, key.phase, key.dtype_env_json)}",
                "latency_us": repr(self.latency_us(key, impl)),
                # Memory-bound at the card's own bandwidth: no peak drift, so
                # the time read back is the latency above.
                "achieved_tflops": "0.0", "achieved_bw_gbps": repr(bw * 0.9),
                "peak_bw_at_test": repr(bw), "origin": "shared"}

    def _answer(self, request: httpx.Request) -> httpx.Response:
        from backend.inference.op_calibration import PhysicalKey, ROW_COLUMNS
        import tests.corpus_row as corpus
        from backend.inference import benchmarks
        self.requests += 1
        body = json.loads(request.content)
        base = benchmarks.validate_text(corpus.to_csv([corpus.row()]), source="fixture")[1][0]
        cols = [*ROW_COLUMNS, "origin"]
        values: Dict[str, int] = {}
        rows = []
        for k in body["keys"]:
            key = PhysicalKey(k["signature"], k["physical_shape"], k["dtype_env_json"],
                              k["gpu"], k["phase"])
            self.keys_asked.append(key)
            state = self.policy(key)
            impls = {MEASURED: ("k1",), MISSING: (), AMBIGUOUS: ("k1", "k2")}[state]
            rows.append([[values.setdefault(r[c], len(values)) for c in cols]
                         for r in (self._row(key, i, base) for i in impls)])
        return httpx.Response(200, json={
            "lake_id": self.scope.lake_id, "snapshot": self.scope.snapshot,
            "columns": cols, "values": list(values), "excluded_mine": 0, "rows": rows})

    def __enter__(self) -> "FixtureBench":
        from backend.measure import lookup
        self._saved = {"url": os.environ.get("HUGGINGARCH_BENCH_URL"),
                       "secret": os.environ.get("HUGGINGARCH_BENCH_SECRET"),
                       "clients": dict(lookup._clients)}
        os.environ["HUGGINGARCH_BENCH_URL"] = "http://bench.fixture"
        os.environ["HUGGINGARCH_BENCH_SECRET"] = "fixture"
        lookup.forget_rows()
        lookup._clients.clear()
        lookup._clients[os.getpid()] = httpx.Client(
            transport=httpx.MockTransport(self._answer), trust_env=False)
        real = self._saved["lookup"] = lookup.lookup

        def counted(scope: Any, keys: Sequence[Any]) -> Any:
            if scope.reads:
                self.lookups += 1
            return real(scope, keys)
        lookup.lookup = counted
        return self

    def __exit__(self, *exc: Any) -> None:
        from backend.measure import lookup
        for env, name in (("url", "HUGGINGARCH_BENCH_URL"),
                          ("secret", "HUGGINGARCH_BENCH_SECRET")):
            if self._saved[env] is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = self._saved[env]
        lookup.lookup = self._saved["lookup"]
        lookup.forget_rows()
        lookup._clients.clear()
        lookup._clients.update(self._saved["clients"])


def measured_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[int, Optional[float]]:
    """Each row's measured time by uid (None where it has none)."""
    return {r["uid"]: r.get("t_measured_s") for r in rows}
