"""Writes to the Kernel Bench lake. No users, no permissions: callers pass
explicit owners and batch references (the routes decide who may act).

Tables (DuckLake; created by `ensure_schema`). Times are TIMESTAMP in UTC
(the lake's session time zone): DuckDB hands TIMESTAMPTZ to Python only
through `pytz`, which is not a dependency.

  batches(batch_id BIGINT, owner UUID, upload_id UUID, piece INTEGER,
          run_id VARCHAR NULL, label VARCHAR NULL, created_at TIMESTAMP,
          status VARCHAR  -- 'private' | 'nominated' | 'shared' | 'rejected'
          within_peak BOOLEAN, nominated_at, decided_at TIMESTAMP NULL,
          decided_by UUID NULL)
      unique (owner, upload_id, piece) — enforced by the writers;
      batch_id = md5_number_upper(owner ‖ 0x1F ‖ upload_id ‖ 0x1F ‖ piece) >> 1
  inputs(input_id BIGINT, op, phase, kernel_impl VARCHAR, shape_json VARCHAR
         (canonical JSON text))
      an input's identity only; unique (op, phase, kernel_impl, shape_json);
      input_id = md5_number_upper(op ‖ 0x1F ‖ phase ‖ 0x1F ‖ kernel_impl ‖
      0x1F ‖ shape_json) >> 1
  measurements(every column of backend.inference.benchmarks.ALL_COLUMNS with
         its type (`MEASUREMENT_TYPES`; `shape_json` and `dtype_env_json` as
         canonical JSON text, an empty cell as NULL), id UUID, batch_id
         BIGINT, input_id BIGINT, env UBIGINT, created_at TIMESTAMP)
      env = md5_number_upper(dtype_env_json ‖ 0x1F ‖ gpu) (`env_key`): an
      envelope is one kernel at one shape on one card, so the deployment that
      cut the shape (`parallel`, still on the row) is not in it;
      written sorted by (batch_id, input_id); the measurement is never
      updated — only the roofline columns (`restamp`) and, once, `env` on a
      lake whose env still names the deployment (`rekey_env`)
  lake_meta(key VARCHAR, value VARCHAR)
      the lake's own facts (docs/measured_calibration.md §4.6): `lake_id` (a uuid written when the
      lake is created — its identity against the lease), `import_state`
      (`none` on a lake created empty; `verified` on the lake the one-time
      import from the Postgres tables built, with that import's
      `import_watermark` (JSON) — history: those tables are gone, nothing
      sets either any more), `lease_claimed_at`

Ids are hashes of the identities they name, not counters: two processes
writing for different owners are not serialized by anything (the owner lock
is per owner, and DuckLake lets two appends commit side by side), so a
`max + 1` would hand both the same id. A hash of a unique identity is the
same in every process and needs no coordination; 63 bits leave a collision
probability near n²/2⁶⁴, and a writer refuses one it sees. Registering a new
input additionally takes `lake.inputs_lock()`, so `inputs` never holds one
key twice.

Annotations — the grouping labels of an input (signature, physical_shape,
workload_batch, workload_context, catalog_kernel) and `phys =
md5_number_upper(signature ‖ 0x1F ‖ physical_shape)` — are not lake data:
they are pure functions of the input under the annotation rules
(`annotate`, composed by the routes; their revision is `rules_rev`). They
live in the lake's derived file (`derived.Derived`, `ANNOT`), which its one
refresh keeps at one entry per input of the snapshot it holds: an input
`annotate` refuses has its entry with every annotation NULL — a
placeholder: its rows are in scope but in no envelope. Nothing here writes
them; an upload's new inputs are annotated by the refresh that brings the
derived file to the upload's snapshot, before any reader reads that
snapshot.

  annot(input_id BIGINT, signature, physical_shape VARCHAR, workload_batch,
        workload_context INTEGER, catalog_kernel VARCHAR, phys UBIGINT)
      in the derived file, one row per input

A row is shared iff its batch is 'shared' and (not within_peak or the row is
within both recorded peaks — `SHARED_ROW` below, the one SQL statement of the
rule, parity-tested against `measured_utilization`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import numpy as np

from backend.inference import benchmarks as _benchmarks
from backend.inference.benchmark_identity import canonical_identity_json
from backend.service.bench.lake import Lake


# A batch's review status, in the order a set takes the first its batches
# have (`sets.list_sets`).
Status = Literal["private", "nominated", "shared", "rejected"]
# annotate(input) -> {signature, physical_shape, workload_batch,
#                     workload_context, catalog_kernel}
Annotate = Callable[[Mapping[str, Any]], Mapping[str, Any]]

# The measurement columns holding a JSON object as text (the row validator
# decodes exactly these: `benchmarks._check_shape_json`,
# `_check_dtype_env_json`); readers that hand rows out decode them.
JSON_COLUMNS = ("shape_json", "dtype_env_json")

# An input's columns besides its id: the input key (what `annotate` reads;
# `lake.inputs` holds exactly it and is unique on it), then the annotations
# `annotate` gives (the derived file's `annot`). Both tables' DDL, INPUT_KEY,
# ANNOTATIONS and an `annot` entry's types are read from here.
_INPUT_COLUMNS = (
    ("op", "VARCHAR", "key"), ("phase", "VARCHAR", "key"),
    ("kernel_impl", "VARCHAR", "key"), ("shape_json", "VARCHAR", "key"),
    ("signature", "VARCHAR", "annotation"), ("physical_shape", "VARCHAR", "annotation"),
    ("workload_batch", "INTEGER", "annotation"), ("workload_context", "INTEGER", "annotation"),
    ("catalog_kernel", "VARCHAR", "annotation"),
)
INPUT_KEY = tuple(name for name, _type, role in _INPUT_COLUMNS if role == "key")
ANNOTATIONS = tuple(name for name, _type, role in _INPUT_COLUMNS if role == "annotation")

# The `measurements` table's type of every corpus column — this table's
# definition, checked against `ALL_COLUMNS` below. The numbers are the ones
# `validate_row` parses as numbers; everything else is text.
MEASUREMENT_TYPES: dict[str, str] = {
    **{c: "VARCHAR" for c in _benchmarks.ALL_COLUMNS},
    **{c: "DOUBLE" for c in ("latency_us", "p50_us", "p95_us", "achieved_tflops",
                             "achieved_bw_gbps", "peak_tflops_at_test", "peak_bw_at_test",
                             "cross_check_rel_err", "cross_check_cosine",
                             "measured_load_factor")},
    **{c: "INTEGER" for c in ("samples", "count_per_step", "schema_version")},
}
if list(MEASUREMENT_TYPES) != list(_benchmarks.ALL_COLUMNS):
    raise AssertionError("MEASUREMENT_TYPES is out of step with ALL_COLUMNS")

# A measured row against the peaks it recorded, over alias m —
# `measured_utilization`'s rule: a ratio only over a positive peak (NULL
# otherwise), an absent achieved rate counting as 0. The row is within its
# peaks when no ratio exceeds 1.
MFU: str = ("(CASE WHEN m.peak_tflops_at_test > 0 THEN"
            " coalesce(m.achieved_tflops, 0) / m.peak_tflops_at_test END)")
MBU: str = ("(CASE WHEN m.peak_bw_at_test > 0 THEN"
            " coalesce(m.achieved_bw_gbps, 0) / m.peak_bw_at_test END)")
WITHIN_PEAK: str = f"(coalesce({MFU} <= 1, true) AND coalesce({MBU} <= 1, true))"


def outcome_of(alias: str) -> str:
    """The SQL of a row's outcome over `alias`, read as
    `benchmarks.outcome_of` reads it: trimmed, empty or NULL as
    `"measured"`. The one SQL spelling of that rule — the duplicate match,
    the browse filter and `OUTCOME` all go through it."""
    return f"coalesce(nullif(trim({alias}.outcome::VARCHAR), ''), 'measured')"


#: A row's outcome over alias m (`outcome_of`).
OUTCOME: str = outcome_of("m")


#: SQL predicate over alias m: this row carries a MEASUREMENT. A row whose
#: `outcome` records something else has no latency and no peaks, and every
#: query that reads one of those must say so — the alternative, discovered
#: by review, is an envelope whose winner has no latency (rendered as
#: 0.0us), a candidate count that does not match the candidates, and
#: `min(latency) < shared_latency` comparing None to a float. Empty or NULL
#: is a measurement: every row written before the column has numbers.
IS_MEASUREMENT: str = f"({OUTCOME} = 'measured')"


def shared_row(within: str) -> str:
    """The shared-row rule over the batch alias b, given the SQL of whether
    the row is within its recorded peaks (`within`: `WITHIN_PEAK` over a
    measurement, or a column a derived table stored it in)."""
    return f"(b.status = 'shared' AND (NOT b.within_peak OR {within}))"


# SQL predicate over aliases m (measurements) and b (batches).
SHARED_ROW: str = shared_row(WITHIN_PEAK)

# The lake's data tables, by name, with their DDL (what a backup counts and
# a restore checks); its metadata table (identity, import state) follows.
SCHEMA: tuple[tuple[str, str], ...] = (
    ("batches", """CREATE TABLE IF NOT EXISTS lake.batches (
        batch_id BIGINT NOT NULL, owner UUID NOT NULL, upload_id UUID NOT NULL,
        piece INTEGER NOT NULL, run_id VARCHAR, label VARCHAR,
        created_at TIMESTAMP NOT NULL, status VARCHAR NOT NULL,
        within_peak BOOLEAN NOT NULL, nominated_at TIMESTAMP,
        decided_at TIMESTAMP, decided_by UUID)"""),
    ("inputs", "CREATE TABLE IF NOT EXISTS lake.inputs (input_id BIGINT NOT NULL, "
     + ", ".join(f"{name} {type_} NOT NULL" for name, type_, role in _INPUT_COLUMNS
                 if role == "key") + ")"),
    ("measurements", "CREATE TABLE IF NOT EXISTS lake.measurements ("
     + ", ".join(f"{c} {t}" for c, t in MEASUREMENT_TYPES.items())
     + ", id UUID NOT NULL, batch_id BIGINT NOT NULL, input_id BIGINT NOT NULL,"
       " env UBIGINT NOT NULL, created_at TIMESTAMP NOT NULL)"),
)
TABLES = tuple(name for name, _ddl in SCHEMA)
META_DDL = "CREATE TABLE IF NOT EXISTS lake.lake_meta (key VARCHAR NOT NULL, value VARCHAR)"

_SEP = "chr(31)"
#: The physical hash of an annotation entry (its signature and shape).
PHYS = f"md5_number_upper(signature || {_SEP} || physical_shape)"
_INPUT_ID = (f"(md5_number_upper(op || {_SEP} || phase || {_SEP} || kernel_impl || {_SEP} "
             "|| shape_json) >> 1)::BIGINT")

#: The alias the lake's derived file is attached under (`derived.Derived`),
#: and the annotations' table in it — what every reader joins as `annot`.
DERIVED = "derived"
ANNOT = f"{DERIVED}.annot"
#: An `annot` entry: the input's id and its annotations, typed; the table
#: adds the physical hash.
ANNOT_ENTRY = {"input_id": "BIGINT"} | {
    name: type_ for name, type_, role in _INPUT_COLUMNS if role == "annotation"}
#: The annotations table's columns, as DDL.
ANNOT_COLUMNS = ("input_id BIGINT NOT NULL, "
                 + ", ".join(f"{name} {type_}" for name, type_ in ANNOT_ENTRY.items()
                             if name != "input_id") + ", phys UBIGINT")


def env_key(t: str) -> str:
    """The SQL of a measurement's `env` over the relation `t` (its
    `dtype_env_json` and `gpu`): with `phys` and the phase, the envelope key
    — one kernel at one shape on one card; the deployment that cut the shape
    is not in it (the calibration key, `op_calibration.PhysicalKey`)."""
    return f"md5_number_upper({t}.dtype_env_json::VARCHAR || {_SEP} || {t}.gpu::VARCHAR)"


@dataclass(frozen=True)
class SetRef:
    """A run or an upload of one owner, as the API names it."""
    owner: str
    kind: Literal["run", "upload"]
    id: str

    def where(self) -> tuple[str, list]:
        """The predicate over `batches` (unaliased) naming this set."""
        column = {"run": "run_id", "upload": "upload_id::VARCHAR"}[self.kind]
        return f"owner = ?::UUID AND {column} = ?", [self.owner, self.id]


@dataclass(frozen=True)
class UploadResult:
    batch_id: Optional[int]   # None when every row was a duplicate (no batch)
    upload_id: str
    inserted: int
    duplicates: int     # rows whose 12-column identity the owner already has


def has_tables(lake: Lake) -> bool:
    """Whether the lake holds any table — a brand-new lake does not. Reads
    only, so it can decide before `ensure_schema` writes anything."""
    return bool(lake.read("SELECT count(*) AS n FROM duckdb_tables() "
                          "WHERE database_name = 'lake'")[0]["n"])


def ensure_schema(lake: Lake, **meta: str) -> None:
    """Create the tables if absent (idempotent); bring an existing lake's
    `measurements` up to today's corpus columns; drop the annotation columns
    a lake written before annotations moved to memory still has on `inputs`
    (DuckLake drops a column in the catalog; the data files are not
    rewritten). `meta`: `lake_meta` entries to write in the same transaction
    when the lake has none yet — a new lake's identity and import state are
    never apart from its tables."""
    def create(cur):
        for _name, ddl in SCHEMA:
            cur.execute(ddl)
        cur.execute(META_DDL)
        # A corpus column the schema has gained since this lake was written.
        # `CREATE TABLE IF NOT EXISTS` above does nothing to a table that
        # already exists, while `upload` names every column of ALL_COLUMNS —
        # so without this an additive schema bump (v14's two driver columns
        # being the case that found it) leaves every existing lake erroring
        # on the next upload, and only a freshly created one works. Which is
        # the half a test on a temp lake cannot see.
        #
        # Asked generically off MEASUREMENT_TYPES rather than naming the
        # columns of the day, so the next additive bump needs no edit here.
        # DuckLake adds a column in the catalog without rewriting the data
        # files: rows already written read it as NULL, which every consumer
        # already treats as the empty string, i.e. UNKNOWN. That is the
        # correct answer for them and the one thing not to paper over — an
        # old row must not come out of a migration claiming a driver nobody
        # observed.
        present = {c for (c,) in cur.execute(
            "SELECT column_name FROM duckdb_columns() WHERE database_name = 'lake' "
            "AND table_name = 'measurements'").fetchall()}
        for c, type_ in MEASUREMENT_TYPES.items():
            if c not in present:
                cur.execute(f'ALTER TABLE lake.measurements ADD COLUMN "{c}" {type_}')
        for (c,) in cur.execute(
                "SELECT column_name FROM duckdb_columns() WHERE database_name = 'lake' "
                "AND table_name = 'inputs' AND column_name NOT IN (SELECT unnest(?::VARCHAR[]))",
                [["input_id", *INPUT_KEY]]).fetchall():
            cur.execute(f'ALTER TABLE lake.inputs DROP COLUMN "{c}"')
        if meta and not cur.execute("SELECT 1 FROM lake.lake_meta LIMIT 1").fetchone():
            for k, v in meta.items():
                cur.execute("INSERT INTO lake.lake_meta VALUES (?, ?)", [k, v])
    lake.transact(create)


def meta(lake: Lake) -> dict[str, str]:
    """The lake's `lake_meta` entries."""
    return {r["key"]: r["value"] for r in lake.read("SELECT key, value FROM lake.lake_meta")}


def set_meta(lake: Lake, **entries: str) -> None:
    """Write `lake_meta` entries (replacing those keys), in one transaction."""
    def write(cur):
        for k, v in entries.items():
            cur.execute("DELETE FROM lake.lake_meta WHERE key = ?", [k])
            cur.execute("INSERT INTO lake.lake_meta VALUES (?, ?)", [k, v])
    lake.transact(write)


def _rows_frame(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """The rows as one string column per corpus column, JSON cells canonical
    (`canonical_identity_json`: `validate_row` has made each an object); a
    column a CSV may omit (`moe_block_member`) reads as empty."""
    frame = {c: [r.get(c) or "" for r in rows] for c in _benchmarks.ALL_COLUMNS}
    for c in JSON_COLUMNS:
        frame[c] = [canonical_identity_json(v) for v in frame[c]]
    return {c: np.array(v, dtype=str) for c, v in frame.items()}


def _typed(c: str) -> str:
    return f"CAST(NULLIF(src.{c}::VARCHAR, '') AS {MEASUREMENT_TYPES[c]}) AS {c}"


#: An incoming row is the owner's duplicate when the 12 identity columns
#: match AND it says the same thing happened. The outcome belongs in this
#: match because a measurement and a recorded refusal of one cell share
#: every identity column by design: without it, uploading the refusal first
#: made the later real measurement a "duplicate" and it was dropped — not
#: refused, not counted, silently absent, and only an entire-batch revocation
#: could undo it. With it the two coexist and the readers arbitrate: every
#: query that reads a latency asks `IS_MEASUREMENT`, so the measurement is
#: what anything shows, and the refusal stays as the record that this cell
#: was once impossible here.
_IDENTITY_MATCH = " AND ".join((
    *(f"m.{c} = src.{c}::VARCHAR" for c in _benchmarks.IDENTITY_COLUMNS),
    f"{outcome_of('m')} = {outcome_of('src')}"))


def records(types: Mapping[str, Any]) -> str:
    """A relation of the records in one JSON text parameter, with the
    columns `types` names — one parameter parsed by DuckDB: binding a list of
    Python dicts converts value by value (~20 s for 27k entries; this 0.2 s)."""
    # The column names and types are the callers' constants (no quotes).
    return f"(SELECT unnest(from_json(?::JSON, '{json.dumps([types])}'), recursive := true))"


def upload(lake: Lake, *, owner: str, upload_id: str, run_id: Optional[str],
           label: Optional[str], rows: Sequence[Mapping[str, Any]],
           status: Status = "private",
           decided_by: Optional[str] = None) -> UploadResult:
    """One batch from validated CSV rows (strings as `validate_text` yields
    them). Under `lake.owner_lock(owner)`: register the inputs the lake
    lacks under `lake.inputs_lock()` (their annotations are the derived
    file's: the refresh that reaches this upload's snapshot adds them before
    any reader reads it); then, in one transaction, the new inputs, the batch, and the rows whose identity the
    owner does not already have. No batch is written when every row is such
    a duplicate. Raises `ValueError` when the owner already has a batch with
    this `upload_id`."""
    frame = _rows_frame(rows)
    keys = sorted({tuple(str(frame[c][i]) for c in INPUT_KEY) for i in range(len(rows))})
    key_frame = {c: np.array([k[j] for k in keys], dtype=str) for j, c in enumerate(INPUT_KEY)}

    def missing_inputs(cur) -> list[dict]:
        """The rows' inputs the lake lacks: input_id and the key."""
        cur.register("new_keys", key_frame)
        out = cur.execute(
            f"SELECT {_INPUT_ID} AS input_id, * FROM (SELECT "
            + ", ".join(f"{c}::VARCHAR AS {c}" for c in INPUT_KEY)
            + " FROM new_keys) k ANTI JOIN lake.inputs i USING (" + ", ".join(INPUT_KEY) + ")")
        names = [d[0] for d in out.description]
        return [dict(zip(names, r, strict=True)) for r in out.fetchall()]

    def write(cur) -> UploadResult:
        if cur.execute("SELECT 1 FROM lake.batches WHERE owner = ?::UUID AND upload_id = ?::UUID",
                       [owner, upload_id]).fetchone():
            raise ValueError(f"owner {owner} already has upload {upload_id}")
        new = missing_inputs(cur)
        if new:
            key = {"input_id": "BIGINT", **dict.fromkeys(INPUT_KEY, "VARCHAR")}
            cur.execute(f"INSERT INTO lake.inputs SELECT {', '.join(key)} FROM {records(key)}",
                        [json.dumps(new)])
            clash = cur.execute(
                "SELECT input_id FROM lake.inputs GROUP BY input_id HAVING count(*) > 1 LIMIT 1"
            ).fetchone()
            if clash:
                raise RuntimeError(f"input_id collision on {clash[0]}")
        batch_id = cur.execute(
            "SELECT (md5_number_upper(?::UUID::VARCHAR || {s} || ?::UUID::VARCHAR || {s} || '0')"
            " >> 1)::BIGINT".format(s=_SEP), [owner, upload_id]).fetchone()[0]
        if cur.execute("SELECT 1 FROM lake.batches WHERE batch_id = ?", [batch_id]).fetchone():
            raise RuntimeError(f"batch_id collision on {batch_id}")
        cur.register("src", frame)
        # ORDER BY input_id: each Parquet row group of the batch then spans
        # a disjoint input_id range (row groups written in parallel may land
        # out of order, which zone maps do not mind).
        # Columns NAMED, not positional. A column added to an existing lake
        # lands at the END of the table (DuckLake appends it in the catalog),
        # so a positional insert of "corpus columns, then the five of our
        # own" stops lining up the moment any lake has been upgraded rather
        # than created — and lining up wrongly is worse than failing, since
        # neighbouring VARCHAR columns would accept each other's values.
        inserted = cur.execute(
            "INSERT INTO lake.measurements ("
            + ", ".join([*_benchmarks.ALL_COLUMNS, "id", "batch_id",
                         "input_id", "env", "created_at"])
            + ") SELECT "
            + ", ".join(_typed(c) for c in _benchmarks.ALL_COLUMNS)
            + ", uuid(), ?, i.input_id, {env}, now()::TIMESTAMP "
              "FROM src JOIN lake.inputs i ON i.op = src.op::VARCHAR "
              "AND i.phase = src.phase::VARCHAR AND i.kernel_impl = src.kernel_impl::VARCHAR "
              "AND i.shape_json = src.shape_json::VARCHAR "
              "WHERE NOT EXISTS (SELECT 1 FROM lake.measurements m JOIN lake.batches b "
              "USING (batch_id) WHERE b.owner = ?::UUID AND {match}) "
              "ORDER BY i.input_id".format(env=env_key("src"), match=_IDENTITY_MATCH),
            [batch_id, owner]).fetchone()[0]
        if inserted:
            cur.execute(
                "INSERT INTO lake.batches VALUES (?, ?::UUID, ?::UUID, 0, ?, ?, now()::TIMESTAMP, ?, false, "
                "CASE WHEN ? = 'nominated' THEN now()::TIMESTAMP END, "
                "CASE WHEN ? IN ('shared', 'rejected') THEN now()::TIMESTAMP END, ?::UUID)",
                [batch_id, owner, upload_id, run_id, label, status, status, status, decided_by])
        return UploadResult(batch_id=batch_id if inserted else None, upload_id=upload_id,
                            inserted=inserted, duplicates=len(rows) - inserted)

    with lake.owner_lock(owner):
        new = lake.transact(missing_inputs)
        if not new:
            # Inputs are never deleted: every key stays registered.
            return lake.transact(write)
        with lake.inputs_lock():
            return lake.transact(write)


def batches_of(lake: Lake, ref: SetRef) -> list[dict]:
    """The batches a set reference names (kind=run: the owner's batches with
    that run_id; kind=upload: that upload_id's pieces), earliest first."""
    where, params = ref.where()
    return lake.read(f"SELECT * FROM lake.batches WHERE {where} "
                     "ORDER BY created_at, upload_id, piece", params)


def set_status(lake: Lake, refs: Sequence[SetRef], *, from_status: Sequence[Status],
               to_status: Status, within_peak: Optional[bool] = None,
               decided_by: Optional[str] = None) -> list[dict]:
    """Move the referenced batches currently in `from_status` to `to_status`
    (nominate: private→nominated; withdraw: nominated→private; promote:
    nominated→shared; reject: nominated→rejected). `within_peak` is set when
    given; `nominated_at` follows nomination, `decided_*` a promote or reject.
    Returns the changed batches."""
    def move(cur) -> list[dict]:
        ids: list[int] = []
        for ref in refs:
            where, params = ref.where()
            ids += [r[0] for r in cur.execute(
                f"SELECT batch_id FROM lake.batches WHERE {where} AND status IN "
                f"(SELECT unnest(?::VARCHAR[]))", [*params, list(from_status)]).fetchall()]
        if not ids:
            return []
        decided = to_status in ("shared", "rejected")
        cur.execute(
            "UPDATE lake.batches SET status = ?, within_peak = coalesce(?, within_peak), "
            "nominated_at = CASE WHEN ? = 'nominated' THEN now()::TIMESTAMP "
            "WHEN ? = 'private' THEN NULL ELSE nominated_at END, "
            "decided_at = CASE WHEN ? THEN now()::TIMESTAMP ELSE decided_at END, "
            "decided_by = CASE WHEN ? THEN ?::UUID ELSE decided_by END "
            "WHERE batch_id IN (SELECT unnest(?::BIGINT[]))",
            [to_status, within_peak, to_status, to_status, decided, decided, decided_by, ids])
        out = cur.execute("SELECT * FROM lake.batches WHERE batch_id IN "
                          "(SELECT unnest(?::BIGINT[])) ORDER BY created_at, upload_id, piece",
                          [ids])
        names = [d[0] for d in out.description]
        return [dict(zip(names, r, strict=True)) for r in out.fetchall()]
    return lake.transact(move)


def delete_set(lake: Lake, ref: SetRef) -> int:
    """Delete the referenced batches and their rows; returns rows deleted.
    Under the owner lock, so an upload's identity anti-join never skips a
    row as the duplicate of one being deleted."""
    where, params = ref.where()

    def delete(cur) -> int:
        ids = [r[0] for r in cur.execute(
            f"SELECT batch_id FROM lake.batches WHERE {where}", params).fetchall()]
        if not ids:
            return 0
        n = cur.execute("DELETE FROM lake.measurements WHERE batch_id IN "
                        "(SELECT unnest(?::BIGINT[]))", [ids]).fetchone()[0]
        cur.execute("DELETE FROM lake.batches WHERE batch_id IN (SELECT unnest(?::BIGINT[]))",
                    [ids])
        return n

    with lake.owner_lock(ref.owner):
        return lake.transact(delete)


def delete_rows(lake: Lake, *, owner: str, ids: Sequence[str]) -> int:
    """Delete the owner's rows by id (UUID text); returns rows deleted. A
    batch left without rows goes with them."""
    if not ids:
        return 0

    def delete(cur) -> int:
        batches = [r[0] for r in cur.execute(
            "SELECT DISTINCT m.batch_id FROM lake.measurements m JOIN lake.batches b "
            "USING (batch_id) WHERE b.owner = ?::UUID AND m.id IN (SELECT unnest(?::UUID[]))",
            [owner, list(ids)]).fetchall()]
        if not batches:
            return 0
        n = cur.execute(
            "DELETE FROM lake.measurements WHERE batch_id IN (SELECT unnest(?::BIGINT[])) "
            "AND id IN (SELECT unnest(?::UUID[]))", [batches, list(ids)]).fetchone()[0]
        cur.execute(
            "DELETE FROM lake.batches WHERE batch_id IN (SELECT unnest(?::BIGINT[])) "
            "AND batch_id NOT IN (SELECT DISTINCT batch_id FROM lake.measurements "
            "WHERE batch_id IN (SELECT unnest(?::BIGINT[])))", [batches, batches])
        return n

    with lake.owner_lock(owner):
        return lake.transact(delete)


#: The columns `restamp` may rewrite: the two halves of every utilization a
#: row shows (`MFU`, `MBU`, `WITHIN_PEAK` are SQL over exactly these). None
#: is the measurement — that is the latency, which nothing here touches.
ROOFLINE_COLUMNS = ("peak_tflops_at_test", "peak_bw_at_test",
                    "achieved_tflops", "achieved_bw_gbps")


def restamp(lake: Lake, *, owner: str,
            rows: Sequence[Mapping[str, Any]]) -> int:
    """Rewrite the owner's rows' roofline columns; returns rows updated.

    A row's utilization is `achieved / peak`, and neither half is the
    measurement. The PEAKS are what the GPU table said when the row was
    measured: when the table is corrected — the RTX-4090's bf16 column was
    the vendor sheet's figure at a boost clock the card clears, 8.3% under
    what it holds — the rows carrying the old number show a utilization of a
    peak that never existed. The ACHIEVED rates are the plan's per-launch
    numerators over the measured p50: when the walk corrects what a launch
    moves — 2026-10-24, an FP8 GEMM billed for the activation conversion its
    timed launch does not do, a partial rotation billed for whole heads —
    the rows stamped with the old numerator claim up to 3.5x the bandwidth
    the launch used, and some of them more than the card has.

    Each row names its id and the columns to set (`ROOFLINE_COLUMNS`); a
    column it leaves out keeps its value. Updated in place, so the row keeps
    its batch and with it its review status — a nominated or shared row is
    corrected where it is, not re-uploaded as a private one. Only the
    owner's rows. What the new numbers are is the caller's business:
    `tools.restamp_peaks` reads peaks from the functions the emitter used,
    and achieved rates come from a fresh plan (`kernel_bench.emit.restamp_row`).
    """
    patch = [{"id": str(r["id"]),
              **{c: None if r.get(c) is None else float(r[c]) for c in ROOFLINE_COLUMNS}}
             for r in rows]
    if not patch:
        return 0
    types = {"id": "UUID", **dict.fromkeys(ROOFLINE_COLUMNS, "DOUBLE")}

    def update(cur) -> int:
        # One JSON parameter (`_records`), not `executemany`: binding row by
        # row ran a 5,000-row patch out of the lake's `memory_limit`.
        return cur.execute(
            "UPDATE lake.measurements AS m SET "
            + ", ".join(f"{c} = coalesce(r.{c}, m.{c})" for c in ROOFLINE_COLUMNS)
            + f" FROM {records(types)} AS r WHERE m.id = r.id AND m.batch_id IN ("
            "SELECT batch_id FROM lake.batches WHERE owner = ?::UUID)",
            [json.dumps(patch), owner]).fetchone()[0]

    with lake.owner_lock(owner):
        return lake.transact(update)


def rekey_env(lake: Lake) -> int:
    """Recompute `env` where it is not `env_key` — a lake written while env
    still named the deployment (`parallel`). A storage key, not a fact: the
    rows' columns do not change. Idempotent; returns the rows rekeyed."""
    stale = f"env <> {env_key('measurements')}"
    if not lake.read(f"SELECT EXISTS (SELECT 1 FROM lake.measurements WHERE {stale}) AS s")[0]["s"]:
        return 0
    lake.require_free("rekeying env")
    return lake.transact(lambda cur: cur.execute(
        f"UPDATE lake.measurements SET env = {env_key('measurements')} WHERE {stale}").fetchone()[0])
