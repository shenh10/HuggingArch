"""The Kernel Bench page's views, read from the lake's derived file.

`browse(lake, q, derived)` answers what `GET /api/bench/browse` returns
before the route's display fields (`_as_row`, `_coverage`): the same JSON
shape the old Postgres `bench_browse` (migration 0022, dropped by 0026; in
git history) returned, without `pending`: nothing is ever in flight.

Views: envelope, rows, workload, curves, coverage; every view but curves
also returns `facets` (each filter's values under the other filters).
Scopes (`Query.source`, `run`, `upload`): the shared corpus, the caller's rows ("mine"), shared
∪ mine (a row counted once), or one of the caller's sets (run / upload).
Sorts and filters as in 0022; bounds kept as product limits (workload
candidates 50,000; curves 50,000 envelopes per op / 4,000 points).

What every view reads. Not the lake: the browse table (`ROWS`) of the
lake's derived file (`derived.Derived`), one row per measurement of the
snapshot the file holds, every quantity of the browse surface (`_FIELDS`)
already derived from the measurement, its batch and its input's
annotations (`rows_select`, the one place they are derived), with the
batches as of that snapshot (`BATCHES`, whose review state decides the
scope) and each physical shape's place on a curve (`PLACEMENT`). The file's
one refresh keeps them at the lake's snapshot by its change feed; a request
first waits for the file to reach the snapshot it starts at
(`Derived.reach`), so it answers the lake as it is, and never reads the
lake's files instead. Computed per request over the lake, the envelope
view's grouping read every data file and joined two million annotations
(2.4 s on a 4.28M-row lake), the curves view placed every shape in Python
(11 s, a 62 MB answer for every op at once); over the derived file the
envelope view is 0.7 s, the rest 0.03-0.3 s, one op's curves at most 1.2 s.

An envelope is one physical op — (signature, physical shape) at one
(dtype env, gpu) and phase: one kernel at one shape on one card, whichever
deployment (`parallel`, kept on the row) cut the shape — over the scope's
rows of annotated inputs: its winner is the fastest row (ties: the smaller id), its gain the
runner-up's latency over the winner's, less one; its candidates the 20
fastest; its kernels and models every distinct one among its rows. The
envelope's model, framework and roofline check are its winner's; the kernel
filter matches an envelope any of whose rows ran that kernel.

Query shape: project only the columns a view reads; envelope groups the
browse rows by their envelope key (`envelope`, the hash of phys, env and
phase the derived file stores) with the per-group aggregates the sort and
the page need, then builds candidate lists, kernels and models for the
page's groups only; rows find the page's ids over the narrow columns, then
read those rows; facets and coverage count the groups (`GROUPS`), not the
rows.
"""
from __future__ import annotations

import json
import math
import statistics
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

from backend.arch.dims import call_token_axes, symbols as dim_symbols
from backend.inference.benchmarks import OUTCOME_VOCABULARY
from backend.inference.op_signatures import all_signatures
from backend.service.bench import store
from backend.service.bench.lake import Lake

if TYPE_CHECKING:
    from backend.service.bench.derived import Derived


@dataclass(frozen=True)
class Query:
    """One browse request, validated by the route."""
    view: str = "envelope"            # envelope | rows | workload | curves | coverage
    caller: Optional[str] = None       # the signed-in user, if any
    source: str = ""                   # "" (shared ∪ mine) | "shared" | "mine"
    run: str = ""                      # one of the caller's runs …
    upload: str = ""                   # … or uploads (the route resolves an admin's reads)
    set_owner: Optional[str] = None    # owner of run/upload (the caller, or anyone for an admin)
    filters: dict = field(default_factory=dict)   # model, signature, gpu, framework, dtype, phase, kernel, outcome, check, key, batch, context
    sort: str = ""
    desc: Optional[bool] = None        # None: the view's own order (workload descending, others ascending)
    page: int = 0
    page_size: int = 50


class Refused(ValueError):
    """A request beyond a view's bounds; the message names the bound (400)."""


def browse(lake: Lake, q: Query, derived: "Derived") -> dict:
    """One view (`Query`) over the derived file, once it holds the lake's
    snapshot now (`Derived.reach`: waits for the refresh; `LakeUnavailable`
    when it cannot reach it in time)."""
    b = _Browse(lake, q)
    b.check_request()
    derived.reach(lake, lake.snapshot_id())
    return b.answer()


# The derived file's tables the views read (`derived.Derived` builds them).
#: One row per measurement: `batch_id`, `input_id`, `env`, `phys` and every
#: quantity of `_FIELDS` that has SQL, by its name (`rows_select`).
ROWS = f"{store.DERIVED}.browse_rows"
#: The lake's batches at the file's snapshot: their review state and owner
#: decide the scope.
BATCHES = f"{store.DERIVED}.batches"
#: The browse rows grouped by everything a scope, a facet or the coverage
#: view tells rows apart by — batch, within peaks, every filter column —
#: with how many rows each group has, and how many pass each roofline check
#: (`n`, `n_<check>`; `groups_select`): what facets and coverage count, a
#: few thousand groups for millions of rows.
GROUPS = f"{store.DERIVED}.browse_groups"
#: Per physical shape (`phys`), its place on a curve (`_Placement.entry`):
#: axis (NULL: unplaced), x, rest (the shape without its token axes, JSON),
#: factors (x's factors, a JSON list); integers as text (`_Placement.entry`).
PLACEMENT = f"{store.DERIVED}.placement"

# The row as a view reads it: the browse row (m) and its batch (b).
_FROM = f"FROM {ROWS} m JOIN {BATCHES} b ON b.batch_id = m.batch_id"


@dataclass(frozen=True)
class _Field:
    """One quantity of the browse surface, declared once: its SQL over the
    measurement (m), its batch (b) and its input's annotations (a) — what
    the derived file stores it as, under its name (`rows_select`) — and
    the parts of the page that read it: a row's struct (`shown`), a facet
    the page filters by (`filter`, the page's name for it), a sort key
    (`sort`, the page's name; the rows view takes every sort key, `views`
    the other views that take it), or a roofline check (`check`). The
    page's names are not always the column's: the `kernel` filter is the
    catalog kernel, the `kernel` sort the implementation. An envelope
    aggregate has no row SQL (the envelope view computes it)."""
    name: str
    sql: Optional[str]
    shown: bool = False
    filter: Optional[str] = None
    sort: Optional[str] = None
    views: tuple = ()
    check: bool = False


def _positive(sql: str) -> str:
    return f"(CASE WHEN {sql} > 0 THEN {sql} END)"


_DTYPE = "json_extract_string(m.dtype_env_json, '$.activation')"   # see facets
#: The views that read a latency: only a measurement has one.
_LATENCY_VIEWS = ("envelope", "workload", "curves")
_WORKLOAD, _ENVELOPE = ("workload",), ("workload", "envelope")

# The surface. Filters are listed in facet order. A row's rates are shown
# only when positive; MFU / MBU and the check are the store's
# (`measured_utilization`'s rule, parity-tested there). The row is 0021b's
# `_bench_row_json`; `origin` is added per request. The roofline columns
# keep the lake's names: the shared-row rule (`store.SHARED_ROW`) reads them
# from the browse row as it reads them from a measurement.
_FIELDS = (
    _Field("model", "m.model", shown=True, filter="model", sort="model"),
    _Field("signature", "a.signature", shown=True, filter="signature", sort="op", views=_WORKLOAD),
    _Field("gpu", "m.gpu", shown=True, filter="gpu", sort="gpu"),
    _Field("framework", "m.framework", shown=True, filter="framework", sort="framework",
           views=_WORKLOAD),
    _Field("dtype", _DTYPE, filter="dtype", sort="dtype"),
    _Field("phase", "m.phase", shown=True, filter="phase", sort="phase"),
    _Field("catalog_kernel", "a.catalog_kernel", shown=True, filter="kernel"),
    # What happened (`benchmarks.OUTCOMES`). Filtered in `scope`, not with the
    # other filters: it decides which rows are in view at all.
    _Field("outcome", store.OUTCOME, shown=True, filter="outcome"),
    _Field("outcome_detail", "m.outcome_detail", shown=True),
    # Which clock timed it (`benchmarks.TIMING_METHODS`); NULL on a row timed
    # before the clock recorded itself, which the page marks.
    _Field("timing_method", "m.timing_method", shown=True),
    _Field("id", "m.id", shown=True),
    _Field("op", "m.op", shown=True),
    _Field("framework_version", "m.framework_version", shown=True, sort="version"),
    _Field("kernel_impl", "m.kernel_impl", shown=True, sort="kernel", views=_WORKLOAD),
    _Field("parallel", "m.parallel", shown=True),
    _Field("shape_json", "m.shape_json", shown=True),
    _Field("physical_shape", "a.physical_shape", shown=True, sort="shape", views=_WORKLOAD),
    _Field("dtype_env_json", "m.dtype_env_json", shown=True),
    _Field("node_spec_id", "m.node_spec_id", shown=True, sort="layer", views=_WORKLOAD),
    _Field("latency_us", "m.latency_us", shown=True, sort="latency", views=_ENVELOPE),
    _Field("gain", None, sort="gain", views=("envelope",)),
    _Field("n_candidates", None, sort="candidates", views=("envelope",)),
    _Field("achieved_tflops", "m.achieved_tflops", shown=True),
    _Field("achieved_bw_gbps", "m.achieved_bw_gbps", shown=True),
    _Field("peak_tflops_at_test", "m.peak_tflops_at_test", shown=True),
    _Field("peak_bw_at_test", "m.peak_bw_at_test", shown=True),
    _Field("samples", "m.samples", shown=True, sort="samples", views=_WORKLOAD),
    _Field("date", "m.date", shown=True, sort="date"),
    _Field("upload_id", "b.upload_id", shown=True),
    _Field("run_id", "b.run_id", shown=True),
    _Field("workload_batch", "a.workload_batch", shown=True),
    _Field("workload_context", "a.workload_context", shown=True),
    _Field("tflops", _positive("m.achieved_tflops"), sort="tflops", views=_ENVELOPE),
    _Field("gbps", _positive("m.achieved_bw_gbps"), sort="gbps", views=_ENVELOPE),
    _Field("mfu", _positive(store.MFU), sort="mfu", views=_ENVELOPE),
    _Field("mbu", _positive(store.MBU), sort="mbu", views=_ENVELOPE),
    _Field("ai", "(CASE WHEN m.achieved_tflops > 0 AND m.achieved_bw_gbps > 0"
                 " THEN 1000 * m.achieved_tflops / m.achieved_bw_gbps END)", sort="ai"),
    # Whether the row is within its recorded peaks: what the shared-row rule
    # asks of a row (`store.shared_row`).
    _Field("within_peak", store.WITHIN_PEAK),
    _Field("over", f"(NOT {store.WITHIN_PEAK})", check=True),
    _Field("ok", f"(({store.MFU} IS NOT NULL OR {store.MBU} IS NOT NULL) AND {store.WITHIN_PEAK})",
           check=True),
)


def rows_select(source: str) -> str:
    """The browse row of every measurement of `source` (a relation with the
    lake's measurement columns, aliased m): its keys and every quantity of
    `_FIELDS` with SQL, under its name — with its batch from `BATCHES` and
    its input's annotations from `store.ANNOT` (a placeholder's are NULL).
    The derived file's only definition of the browse table."""
    return ("SELECT m.batch_id, m.input_id, m.env, a.phys, "
            # The envelope (module docstring) as one key: an envelope view
            # groups millions of rows by it.
            "md5_number_upper(a.phys::VARCHAR || chr(31) || m.env::VARCHAR || chr(31) || m.phase)"
            " AS envelope, "
            + ", ".join(f'{f.sql} AS "{f.name}"' for f in _FIELDS if f.sql is not None)
            + f" FROM {source} m JOIN {BATCHES} b ON b.batch_id = m.batch_id"
            f" LEFT JOIN {store.ANNOT} a ON a.input_id = m.input_id")


def groups_select() -> str:
    """The derived file's only definition of `GROUPS`, over `ROWS`."""
    keys = ["batch_id", "within_peak", *(f.name for f in _FIELDS if f.filter)]
    return (f"SELECT {', '.join(_col(k) for k in keys)}, count(*) AS n, "
            + ", ".join(f'count(*) FILTER (WHERE {_col(f.name)}) AS "n_{f.name}"'
                        for f in _FIELDS if f.check)
            + f" FROM {ROWS} m GROUP BY ALL")


def _col(name: str) -> str:
    """A stored quantity of the browse row, by its `_Field` name."""
    return f'm."{name}"'


class _Surface:
    """The lookups the views read, each derived from `_FIELDS`: every
    quantity is the browse row's column of its name."""

    def __init__(self, fields: tuple) -> None:
        self.sql = {f.name: _col(f.name) for f in fields if f.sql is not None}
        self.row = {f.name: _col(f.name) for f in fields if f.shown}
        self.dims = {f.filter: _col(f.name) for f in fields if f.filter}
        self.checks = {f.name: _col(f.name) for f in fields if f.check}
        sortable = [f for f in fields if f.sort]
        self.sorts = {
            "rows": {f.sort: _col(f.name) for f in sortable if f.sql is not None},
            "workload": {f.sort: _col(f.name) for f in sortable if "workload" in f.views},
            # the envelope view sorts its grouped CTE, by column name
            "envelope": {f.sort: f.name for f in sortable if "envelope" in f.views},
        }
        #: what an envelope's winner can carry beyond a row column
        self.derived = {f.name for f in fields if f.sql is not None and not f.shown}


_SURFACE = _Surface(_FIELDS)
# Rows of annotated inputs — the members of envelopes (a placeholder has none).
_PHYSICAL = f"{_col('signature')} IS NOT NULL AND {_col('physical_shape')} IS NOT NULL"

_WORKLOAD_CANDIDATES = 50_000
_CURVE_ENVELOPES = 50_000
_CURVE_POINTS = 4_000
_CANDIDATES = 20


class _Sql:
    """A statement being built: text and its positional parameters, in
    order. Names come only from the whitelists above; values go in params."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.params: list[Any] = []

    def add(self, text: str, *params: Any) -> "_Sql":
        self.parts.append(text)
        self.params.extend(params)
        return self

    def text(self) -> str:
        return " ".join(self.parts)


class _Browse:
    """One request: its scope, filters, page and the views over them."""

    def __init__(self, lake: Lake, q: Query) -> None:
        self.lake, self.q = lake, q
        self.view = q.view or "envelope"
        self.size = min(max(q.page_size, 1), 200)
        self.off = max(q.page, 0) * self.size
        self.f = {k: v for k, v in q.filters.items() if v is not None and v != ""}
        # Measurements unless the request names another outcome: every view
        # without the filter reads what it read before outcomes were more
        # than `measured` / `oom`, and the facets list the rest.
        self.f.setdefault("outcome", "measured")
        # `outcome=all` (rows view only, `check_request`): measurements and
        # failures together, so a batch's page can show what failed beside
        # what measured. Everywhere the filter would read it (scope, facets)
        # it is absent instead.
        self.all_outcomes = self.f["outcome"] == "all"
        if self.all_outcomes:
            del self.f["outcome"]
        self.sort = q.sort or "latency"
        in_set = bool(q.run or q.upload)
        # Whose rows are "mine" on the page: the set's owner when an admin
        # reads someone's set, else the caller.
        self.reader = (q.set_owner or q.caller) if in_set else q.caller
        self._scope: Optional[tuple[str, list]] = None

    # ---------------------------------------------------------------- scope

    def scope(self, s: _Sql, outcome: bool = True) -> _Sql:
        """The scope's row predicate over m and b — and the outcome filter.

        One outcome is in view at a time, decided here, once, for every view
        this class builds: measurements unless the request names another
        (`outcome`). A row recording an outcome has no latency, and left in
        a view that reads one it inflated the candidate count or became a
        winner the page drew as `0.0us` — so the views that read latencies
        (`_LATENCY_VIEWS`) refuse any other outcome, and the rows and
        coverage views show and count the one asked for. The rows view alone
        takes `all` (`self.all_outcomes`, `check_request`): measurements and
        failures together. `outcome=False` leaves it out, for the facets,
        which list each filter's values under the OTHER filters.
        """
        if self._scope is None:
            self._scope = self._resolve_scope()
        text, params = self._scope
        if outcome and not self.all_outcomes:
            s.add(f"{_SURFACE.sql['outcome']} = ? AND", self.f["outcome"])
        return s.add("(").add(text, *params).add(")")

    def _resolve_scope(self) -> tuple[str, list]:
        q = self.q
        if q.run or q.upload:
            owner = q.set_owner or q.caller
            if owner is None:
                return "FALSE", []
            ref = (store.SetRef(owner, "run", q.run.strip()) if q.run
                   else store.SetRef(owner, "upload", q.upload))
            where, params = ref.where()
            return f"m.batch_id IN (SELECT batch_id FROM {BATCHES} WHERE {where})", params
        parts, params = [], []
        if q.source in ("", "shared"):
            parts.append(store.shared_row(_col("within_peak")))
        if q.caller is not None and q.source in ("", "mine"):
            parts.append("b.owner = ?::UUID")
            params.append(q.caller)
        return (f"({' OR '.join(parts)})" if parts else "FALSE"), params

    def origin(self, s: _Sql) -> _Sql:
        if self.reader is None:
            return s.add("'shared'")
        return s.add("CASE WHEN b.owner = ?::UUID THEN 'mine' ELSE 'shared' END", self.reader)

    def row_struct(self, s: _Sql) -> _Sql:
        s.add("struct_pack(" + ", ".join(f"{k} := {v}" for k, v in _SURFACE.row.items())
              + ", origin := ")
        return self.origin(s).add(")")

    # -------------------------------------------------------------- filters

    def filters(self, s: _Sql, dims=tuple(_SURFACE.dims), check: bool = True) -> _Sql:
        """`AND`-ed row filters on `dims` (and the roofline check)."""
        for d in dims:
            if d in self.f and d != "outcome":          # `scope` filters it
                s.add(f"AND {_SURFACE.dims[d]} = ?", self.f[d])
        if check and self.check():
            s.add("AND " + _SURFACE.checks[self.check()])
        return s

    def check(self) -> Optional[str]:
        """The roofline check the request asks for, if it names one."""
        c = self.f.get("check")
        return c if c in _SURFACE.checks else None

    def one_envelope(self, s: _Sql) -> _Sql:
        """Narrow to the one envelope `key` names, as the envelope view
        returns it (the rows and curves views ask for its members)."""
        key = self.f.get("key")
        if isinstance(key, dict):
            s.add(f"AND {_col('signature')} = ? AND {_col('physical_shape')} = ?"
                  f" AND json({_col('dtype_env_json')}) = json(?)"
                  f" AND {_col('gpu')} = ? AND {_col('phase')} = ?",
                  key.get("signature"), key.get("physical_shape"),
                  json.dumps(key.get("dtype_env_json")), key.get("gpu"), key.get("phase"))
        return s

    def desc(self) -> bool:
        if self.q.desc is None:
            return self.view == "workload"
        return self.q.desc

    def order(self, key: str, tie: str) -> str:
        d = "DESC" if self.desc() else "ASC"
        return f"{key} {d} NULLS LAST, {tie} {d}"

    # --------------------------------------------------------------- answer

    def answer(self) -> dict:
        views = {"envelope": self.envelope, "rows": self.rows, "workload": self.workload,
                 "curves": self.curves, "coverage": self.coverage}
        out: dict = {}
        if self.view != "curves":
            out["facets"] = self.facets()
        out.update(views[self.view]())
        return out

    def check_request(self) -> None:
        """Refuse what the view cannot answer before reading anything."""
        sorts = _SURFACE.sorts
        if self.view not in ("envelope", "rows", "workload", "curves", "coverage"):
            raise Refused(f"unknown view {self.view}")
        if self.all_outcomes:
            if self.view != "rows":
                raise Refused("outcome=all lists measurements and failures together; "
                              "only the rows view shows both")
            outcome = "measured"   # the latency-view guard below reads it
        else:
            outcome = self.f["outcome"]
            if outcome not in OUTCOME_VOCABULARY:
                raise Refused(f"outcome is one of {', '.join(sorted(OUTCOME_VOCABULARY))}, "
                              f"not {outcome}")
        if self.view in _LATENCY_VIEWS and outcome != "measured":
            raise Refused(f"the {self.view} view reads latencies, and only a "
                          f"measurement has one; see outcome={outcome} in the rows view")
        if self.view == "envelope" and self.sort not in sorts["envelope"]:
            *head, last = sorts["envelope"]
            raise Refused(f"envelopes sort by {', '.join(head)} or {last}, not {self.sort}")
        if self.view == "rows" and self.sort not in sorts["rows"]:
            raise Refused(f"rows do not sort by {self.sort}")
        if self.view == "workload":
            if "model" not in self.f or "phase" not in self.f:
                raise Refused("pick a model and a phase to see its workloads")
            if self.sort not in sorts["workload"]:
                raise Refused(f"workload rows do not sort by {self.sort}")

    def read(self, s: _Sql) -> list[dict]:
        return self.lake.read(s.text(), s.params)

    # --------------------------------------------------------------- facets

    def facets(self) -> dict:
        """Each filter's values over the scope under every other filter, read
        from the groups (`GROUPS`): a group passes the roofline check when
        any of its rows does."""
        check = self.check()
        dims = _SURFACE.dims
        s = _Sql().add("WITH d AS (SELECT DISTINCT " + ", ".join(f"{e} AS {d}" for d, e in dims.items()))
        s.add(f"FROM {GROUPS} m JOIN {BATCHES} b ON b.batch_id = m.batch_id WHERE")
        self.scope(s, outcome=False)
        if check:
            s.add(f'AND m."n_{check}" > 0')
        s.add(")")
        cols = []
        for d in dims:
            col = _Sql().add(f"list_sort(list(DISTINCT {d}) FILTER (WHERE {d} IS NOT NULL")
            for o in dims:
                if o != d and o in self.f:
                    col.add(f"AND {o} = ?", self.f[o])
            col.add(f")) AS {d}")
            cols.append(col)
        s.add("SELECT")
        for n, col in enumerate(cols):
            s.add(("," if n else "") + col.text(), *col.params)
        s.add("FROM d")
        row = self.read(s)[0]
        return {d: _plain(row[d] or []) for d in dims}

    # ------------------------------------------------------------- envelope

    def envelopes(self, s: _Sql, winner: tuple[str, ...], extra_where: str = "") -> _Sql:
        """CTEs `g` (one row per envelope in scope, by its key `envelope`:
        count, winner id and latency, the runner-up's latency, and the
        `winner` fields — row columns or display rates — the caller reads)
        and `e` (`g` filtered, with the gain). Filters on key columns (op,
        gpu, phase, dtype — equal on every member) select rows before
        grouping; a `key` selects the one envelope it names there too. The
        model, framework and check filters are the winner's, the kernel
        filter any member's. Each winner field is its own `arg_min_null` (the
        winner's value, NULL included — `arg_min` skips NULLs): a
        struct carried through the grouping for every row in scope was
        twice as slow at 4.28M rows."""
        kernel = self.f.get("kernel")
        check = self.check()
        fastest = f"({_col('latency_us')}, {_col('id')})"
        names = (*(d for d in ("model", "framework") if d in self.f), *winner)
        exprs = {c: _SURFACE.sql[c] for c in dict.fromkeys(names)}
        if check:
            exprs["passes_check"] = _SURFACE.checks[check]
        s.add(f"WITH g AS (SELECT m.envelope, any_value(m.phys) AS phys, count(*) AS n_candidates,"
              f" arg_min({_col('id')}, {fastest}) AS winner_id,"
              f" arg_min_null({_col('latency_us')}, {fastest}) AS latency_us,"
              # the two smallest latencies: the runner-up's is the second
              f" min_by({_col('latency_us')}, {_col('latency_us')}, 2)[2] AS second_latency_us")
        s.add("".join(f", arg_min_null({e}, {fastest}) AS {c}" for c, e in exprs.items()))
        if kernel:
            s.add(f", bool_or({_col('catalog_kernel')} = ?) AS has_kernel", kernel)
        s.add(f"{_FROM} WHERE {_PHYSICAL} AND")
        self.scope(s)
        self.filters(s, dims=("signature", "gpu", "phase", "dtype"), check=False)
        self.one_envelope(s)
        s.add("GROUP BY m.envelope)")
        s.add(", e AS MATERIALIZED (SELECT envelope, phys, n_candidates, winner_id, latency_us,"
              " second_latency_us, second_latency_us / NULLIF(latency_us, 0) - 1 AS gain"
              + "".join(f", {c}" for c in winner if c != "latency_us") + " FROM g WHERE TRUE")
        if kernel:
            s.add("AND has_kernel")
        for d in ("model", "framework"):
            if d in self.f:
                s.add(f"AND {d} = ?", self.f[d])
        if check:
            s.add("AND passes_check")
        return s.add(extra_where + ")")

    def envelope(self) -> dict:
        sort_col = _SURFACE.sorts["envelope"][self.sort]
        order = self.order(sort_col, "winner_id")
        s = self.envelopes(_Sql(), winner=(sort_col,) if sort_col in _SURFACE.derived else ())
        s.add(f", p AS MATERIALIZED (SELECT * FROM e ORDER BY {order} LIMIT ? OFFSET ?)", self.size, self.off)
        # The page's members, read once: the winner's row, candidate lists,
        # kernels, models.
        s.add(", c AS (SELECT p.winner_id, min_by(")
        self.row_struct(s)
        s.add(f", ({_col('latency_us')}, {_col('id')})) AS winner,"
              f" min_by(struct_pack(id := {_col('id')}, latency_us := {_col('latency_us')},"
              f" framework := {_col('framework')}, kernel_impl := {_col('kernel_impl')},"
              f" catalog_kernel := {_col('catalog_kernel')}, model := {_col('model')}),"
              f" ({_col('latency_us')}, {_col('id')}), {_CANDIDATES}) AS candidates,"
              f" list_sort(list(DISTINCT {_col('catalog_kernel')}) FILTER"
              f" (WHERE {_col('catalog_kernel')} IS NOT NULL)) AS kernels,"
              f" list_sort(list(DISTINCT {_col('model')}) FILTER (WHERE {_col('model')} IS NOT NULL)) AS models"
              f" FROM p JOIN {ROWS} m ON m.envelope = p.envelope"
              f" JOIN {BATCHES} b ON b.batch_id = m.batch_id WHERE {_PHYSICAL} AND")
        self.scope(s)
        s.add("GROUP BY p.winner_id)")
        s.add(f"SELECT t.total, x.* FROM (SELECT count(*) AS total FROM e) t LEFT JOIN ("
              f" SELECT p.{sort_col} AS sort_key, p.winner_id, p.n_candidates, p.gain,"
              " c.winner, c.candidates, c.kernels, c.models FROM p JOIN c ON c.winner_id = p.winner_id"
              f") x ON true ORDER BY {self.order('x.sort_key', 'x.winner_id')}")
        got = self.read(s)
        total = int(got[0]["total"])
        items = []
        for r in got:
            if r["winner_id"] is None:
                continue
            w = _row(r["winner"])
            items.append({
                "winner": w, "n_candidates": int(r["n_candidates"]), "gain": _plain(r["gain"]),
                "candidates": _plain(r["candidates"]), "models": list(r["models"] or []),
                "kernels": list(r["kernels"] or []),
                "key": {k: w[k] for k in ("signature", "physical_shape", "dtype_env_json",
                                          "gpu", "phase")}})
        return {"total": total, "items": items}

    # ----------------------------------------------------------------- rows

    def page_of_rows(self, s: _Sql, source: str, totals: str) -> list[dict]:
        """Finish a statement whose CTE `source` has (id, k): `totals` over
        it, then the page's rows in order."""
        s.add(f", p AS (SELECT * FROM {source} ORDER BY {self.order('k', 'id')} LIMIT ? OFFSET ?)",
              self.size, self.off)
        s.add(f"SELECT t.*, x.* FROM (SELECT {totals} FROM {source}) t LEFT JOIN (SELECT p.k AS sort_key, p.id AS page_id,")
        self.row_struct(s)
        s.add(f"AS row FROM p JOIN {ROWS} m ON {_col('id')} = p.id"
              f" JOIN {BATCHES} b ON b.batch_id = m.batch_id"
              f") x ON true ORDER BY {self.order('x.sort_key', 'x.page_id')}")
        return self.read(s)

    def rows(self) -> dict:
        """One page of the scope's rows under the filters (the envelope
        view's scope and filters), or of one envelope's (`key` as the
        envelope view returns it — its rows from every deployment)."""
        s = _Sql().add(f"WITH f AS (SELECT {_col('id')} AS id, {_SURFACE.sorts['rows'][self.sort]} AS k"
                       f" {_FROM} WHERE")
        self.scope(s)
        self.one_envelope(s)
        self.filters(s)
        s.add(")")
        got = self.page_of_rows(s, "f", "count(*) AS total")
        total = int(got[0]["total"])
        return {"total": total,
                "items": [_row(r["row"]) for r in got if r["page_id"] is not None]}

    # ------------------------------------------------------------- workload

    def workload(self) -> dict:
        model, phase = self.f["model"], self.f["phase"]
        batch, context = self.f.get("batch"), self.f.get("context")
        wb, wc = _col("workload_batch"), _col("workload_context")
        s = _Sql().add(f"SELECT list_sort(list(DISTINCT {wb})"
                       f" FILTER (WHERE {wb} IS NOT NULL)) AS batches,"
                       f" count(*) FILTER (WHERE {wb} IS NULL) AS unplaced,"
                       f" list(DISTINCT {wc}) FILTER (WHERE {wb} = ?) AS contexts,"
                       f" count(*) FILTER (WHERE {wb} = ? AND {wc} IN (?, 0)) AS candidates",
                       batch, batch, context)
        s.add(f"{_FROM} WHERE {_col('model')} = ? AND {_col('phase')} = ? AND", model, phase)
        self.scope(s)
        st = self.read(s)[0]
        out: dict = {"batches": list(st["batches"] or []), "unplaced": int(st["unplaced"])}
        if batch is None:
            return {**out, "contexts": []}
        # Real contexts ascending, then 0 (shapes without a context dimension).
        out["contexts"] = sorted(st["contexts"] or [], key=lambda c: (c is None, c == 0, c or 0))
        if context is None:
            return out
        out["candidates"] = n = int(st["candidates"])
        if n > _WORKLOAD_CANDIDATES:
            raise Refused(f"this workload has {n} measurements; the view reads up to 50,000 — pick a source")
        # The candidates — context c and context 0 (they belong to every
        # context) — and per (layer, op) the fastest of them.
        s = _Sql().add(f"WITH c AS (SELECT {_col('id')} AS id, {_col('node_spec_id')} AS node_spec_id,"
                       f" {_col('op')} AS op, {_col('latency_us')} AS latency_us,"
                       f" {_SURFACE.sorts['workload'][self.sort]} AS k {_FROM}"
                       f" WHERE {_col('model')} = ? AND {_col('phase')} = ? AND {wb} = ?"
                       f" AND {wc} IN (?, 0) AND", model, phase, batch, context)
        self.scope(s)
        self.filters(s, dims=[d for d in _SURFACE.dims if d not in ("model", "phase")])
        s.add("), w AS (SELECT min_by(struct_pack(id, latency_us, k), (latency_us, id)) AS best"
              " FROM c GROUP BY node_spec_id, op),"
              " pick AS (SELECT best.id AS id, best.latency_us AS latency_us, best.k AS k FROM w)")
        got = self.page_of_rows(s, "pick", "count(*) AS total, COALESCE(sum(latency_us), 0) AS total_latency_us")
        return {**out, "total": int(got[0]["total"]),
                "total_latency_us": _plain(got[0]["total_latency_us"]),
                "items": [_row(r["row"]) for r in got if r["page_id"] is not None]}

    # --------------------------------------------------------------- curves

    def curves(self) -> dict:
        """The ops in view, and — when the request names one op (its
        `signature` filter) —
        that op's curves: its envelopes along the tokens one kernel call
        runs over (`_Placement`, stored per physical shape in `PLACEMENT`).

        Without an op the answer lists the ops (`{signature}` each): every
        op with an annotated row under the scope and every filter, a shape
        with no physical identity included. The page draws an op's curves
        when it asks for them by name: one answer with every op's curves
        was 187k points, 62 MB on the 4.28M-row lake.

        For the op: a shape with its token axes taken out — a GEMM's (n, k)
        — at one gpu and dtype is one curve, across phases (each point says
        its phase). A point needs an MFU or an MBU: a memory-only op records
        no FLOP rate. The longest curves first, each whole, up to 4,000
        points (the first curve always); envelopes with no token axis are
        `unplaced`; more than 50,000 envelopes in view answers `too_many`."""
        listed = _Sql().add(f"SELECT DISTINCT {_col('signature')} AS signature {_FROM}"
                            f" WHERE {_col('signature')} IS NOT NULL AND")
        self.scope(listed)
        self.filters(listed)
        self.one_envelope(listed)
        op = self.f.get("signature")
        if op is not None:
            listed.add(f"AND {_col('signature')} = ?", op)
        signatures = sorted(r["signature"] for r in self.read(listed))
        if op is None:
            return {"ops": [{"signature": sig} for sig in signatures]}
        if op not in signatures:
            return {"ops": []}
        # The op's envelopes counted first, on the rates alone: an op over
        # the bound answers `too_many` without carrying its winners' text
        # through the grouping.
        rates = ("tflops", "gbps", "mfu", "mbu")
        drawn = " AND (mfu IS NOT NULL OR mbu IS NOT NULL)"
        counted = self.envelopes(_Sql(), winner=rates, extra_where=drawn)
        n_op = self.read(counted.add("SELECT count(*) AS n FROM e"))[0]["n"]
        if n_op > _CURVE_ENVELOPES:
            return {"ops": [{"signature": op, "too_many": True}]}
        shown = ("phase", "physical_shape", "gpu", "dtype_env_json", "framework", "model",
                 "catalog_kernel", "kernel_impl", "latency_us")
        s = self.envelopes(_Sql(), winner=(*rates, *shown), extra_where=drawn)
        s.add("SELECT e.*, coalesce(e.catalog_kernel, e.kernel_impl) AS kernel,"
              f" pl.axis, pl.x, pl.rest, pl.factors FROM e JOIN {PLACEMENT} pl ON pl.phys = e.phys")
        return {"ops": [_curves_of(op, self.read(s))]}

    # ------------------------------------------------------------- coverage

    def coverage(self) -> dict:
        """Rows per catalog kernel under the gpu / dtype / phase / check
        filters, and the rows no catalog kernel claims, per framework — of
        the outcome in view (measurements unless one is named) — and
        `outcomes`, the rows of every outcome under the same filters, so the
        page can say how many of a kernel's attempts failed and how."""
        check = self.check()
        n = f'sum(m."n_{check}")' if check else "sum(m.n)"
        s = _Sql().add(f"SELECT {_col('catalog_kernel')} AS kernel, {_col('framework')} AS framework,"
                       f" {_col('outcome')} AS outcome, {n} AS n FROM {GROUPS} m"
                       f" JOIN {BATCHES} b ON b.batch_id = m.batch_id WHERE")
        self.scope(s, outcome=False)
        self.filters(s, dims=("gpu", "dtype", "phase"), check=False)
        s.add(f"GROUP BY ALL HAVING {n} > 0")
        kernels: dict = {}
        unclassified: dict = {}
        outcomes: dict = {}
        for r in self.read(s):
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + int(r["n"])
            if r["outcome"] != self.f["outcome"]:
                continue
            if r["kernel"] is not None:
                kernels[r["kernel"]] = kernels.get(r["kernel"], 0) + int(r["n"])
            else:
                f = r["framework"] or ""
                unclassified[f] = unclassified.get(f, 0) + int(r["n"])
        return {"kernels": kernels, "unclassified": unclassified, "outcomes": outcomes}


class _Placement:
    """Where a physical shape sits on a curve, from the declarations
    (`call_tokens()`): its token axes, x and the rest of it. Applied once
    per physical shape by the derived file's refresh (`entry`, stored in
    `PLACEMENT`); a view reads the stored answer."""

    def __init__(self, runs: dict, totals: list[str], factors: list[str]) -> None:
        self.runs = runs
        self.totals = totals
        self.factors = factors

    def place(self, signature: str, physical_shape: str):
        """(axis, x, rest, rest's key, x's factors) of a shape, or None:
        unplaced — no token axis, more than one total, or no contract for
        the signature."""
        shape = json.loads(physical_shape)

        def has(a: str) -> bool:
            v = shape.get(a)
            return isinstance(v, int) and not isinstance(v, bool) and v > 0
        runs = self.runs.get(signature)
        if runs is None:
            # No contract: no declaration says what its axes are.
            return None
        totals = [a for a in self.totals if has(a)]
        if totals:
            # A total is the count itself; every factor present moves with
            # it — the declared factors (dims.yaml `call_tokens: factor`),
            # not only the ones the contract's operands lay out: a
            # combine writes its token side, so its B and s_query are on
            # no operand and still grow with the routed rows.
            if len(totals) != 1:
                return None
            x_axes = totals
            taken = set(totals) | {a for a in self.factors if has(a)} | {
                a for r in runs for a in r if has(a)}
        else:
            x_axes = next((list(r) for r in runs if all(has(a) for a in r)), [])
            taken = set(x_axes)
        if not x_axes:
            return None
        rest = {k: v for k, v in shape.items() if k not in taken}
        return ("×".join(x_axes), math.prod(shape[a] for a in x_axes),
                rest, json.dumps(rest, sort_keys=True), tuple(shape[a] for a in x_axes))

    def entry(self, phys: int, signature: str, physical_shape: str) -> dict:
        """A `PLACEMENT` row: the placement of one physical shape (axis NULL:
        unplaced)."""
        at = self.place(signature, physical_shape)
        if at is None:
            return {"phys": phys, "axis": None, "x": None, "rest": None, "factors": None}
        axis, x, _rest, key, factors = at
        # Integers as text: exact past 2^53, and read back several times
        # faster than DuckDB's 128-bit integers.
        return {"phys": phys, "axis": axis, "x": str(x), "rest": key,
                "factors": json.dumps(list(factors))}


#: A `PLACEMENT` row's types (`_Placement.entry`).
PLACEMENT_TYPES = {"phys": "UBIGINT", "axis": "VARCHAR", "x": "VARCHAR", "rest": "VARCHAR",
                   "factors": "VARCHAR"}


def _curves_of(signature: str, envelopes: list[dict]) -> dict:
    """One op's envelopes (each with its stored placement) onto curves;
    points are built only for the curves the answer keeps."""
    members: dict = {}
    unplaced = 0
    for e in envelopes:
        if e["axis"] is None:
            unplaced += 1
            continue
        key = (e["gpu"], e["dtype_env_json"], e["axis"], e["rest"])
        members.setdefault(key, []).append((int(e["x"]), tuple(json.loads(e["factors"])), e))
    kept, upto = [], 0
    for key in sorted(members, key=lambda k: (-len(members[k]), k)):
        upto += len(members[key])
        if upto > _CURVE_POINTS and kept:
            break
        gpu, dtype, axis, rest = key
        placed = sorted(members[key], key=lambda m: (m[0], m[2]["physical_shape"], m[2]["phase"],
                                                     m[2]["latency_us"]))
        pts = [{"x": x, "shape": e["physical_shape"], "phase": e["phase"], "framework": e["framework"],
                "kernel": e["kernel"], "model": e["model"], "latency_us": e["latency_us"],
                "tflops": e["tflops"], "gbps": e["gbps"], "mfu": e["mfu"], "mbu": e["mbu"]}
               for x, _, e in placed]
        kept.append({"gpu": gpu,
                     "dtype_env_json": json.loads(dtype), "axis": axis,
                     "shape": json.loads(rest),
                     "saturation": _saturation(pts), "points": pts,
                     "lines": _lines([(p["phase"], f) for p, (_, f, _) in zip(pts, placed, strict=True)])})
    return {"signature": signature, "points": sum(len(m) for m in members.values()),
            "curves_total": len(members), "unplaced": unplaced, "curves": kept}


def _lines(points: list[tuple]) -> list[list[int]]:
    """The lines through a curve's points (each a (phase, x's factors) in
    x order), as point indices: one factor of x moving, the phase and every
    other factor held. A product x is a grid of launches — `B=4 s_query=1`
    and `B=1 s_query=4` share x and are different calls — so a line through
    every point would join launches of different series; a point on no line
    is a line of its own."""
    lines: list[list[int]] = []
    width = len(points[0][1]) if points else 0
    for moving in range(width):
        groups: dict = {}
        for i, (phase, f) in enumerate(points):
            groups.setdefault((phase, f[:moving] + f[moving + 1:]), []).append(i)
        lines.extend(g for g in groups.values() if len(g) > 1)
    on = {i for g in lines for i in g}
    return lines + [[i] for i in range(len(points)) if i not in on]


def _saturation(points: list[dict]) -> dict:
    """Where a curve turns compute-bound: the token count at which its
    arithmetic intensity reaches the card's ridge (peak FLOP/s over peak
    bytes/s). A point's intensity over the ridge is its MFU over its MBU —
    both divide the same work by the same latency, against the same card's
    two peaks — so the crossing is where MFU/MBU reaches 1, interpolated on
    log tokens between the last point below and the first at or above (the
    median ratio of the points at one x). `crossed` with `tokens`;
    `compute` — at or above from the first point (`tokens`: that point's);
    `memory` — below at every point (`tokens`: the last point's);
    `no_flops` — no point has both ratios (an op that records no FLOP rate)."""
    by_x: dict = {}
    for p in points:
        f, m = p["mfu"], p["mbu"]
        if f is not None and m is not None and f > 0 and m > 0:
            by_x.setdefault(p["x"], []).append(math.log(float(f) / float(m)))
    if not by_x:
        return {"state": "no_flops"}
    xs = sorted(by_x)
    r = [statistics.median(by_x[x]) for x in xs]
    if r[0] >= 0:
        return {"state": "compute", "tokens": xs[0]}
    for x0, r0, x1, r1 in zip(xs[:-1], r[:-1], xs[1:], r[1:], strict=True):
        if r0 < 0 <= r1:
            t = -r0 / (r1 - r0)
            return {"state": "crossed",
                    "tokens": math.exp(math.log(x0) + t * (math.log(x1) - math.log(x0)))}
    return {"state": "memory", "tokens": xs[-1]}


def call_tokens() -> dict:
    """What the curves view places shapes by (`Query.filters["call_tokens"]`),
    from the declarations: dims.yaml's `total` and `factor` token axes, and per signature
    the runs its contract lays the `factor` axes out in
    (`OpSignature.token_runs`) — folded rows multiply, separate axes do not,
    and a `context` axis runs with the batch (dims.yaml `kind: batch`)."""
    roles = call_token_axes()
    batch = [n for n, sym in sorted(dim_symbols().items()) if sym.kind == "batch"]
    return {"total": list(roles["total"]),
            "factor": list(roles["factor"]),
            "runs": {s.name: [list(r) for r in s.token_runs(roles["factor"], roles["context"], batch)]
                     for s in all_signatures()}}


def _plain(v: Any) -> Any:
    """A lake value as JSON: ids as text, decimals as numbers."""
    if isinstance(v, uuid.UUID):
        return str(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def _row(r: dict) -> dict:
    out = _plain(r)
    for k in store.JSON_COLUMNS:
        if isinstance(out.get(k), str):
            out[k] = json.loads(out[k])
    return out
