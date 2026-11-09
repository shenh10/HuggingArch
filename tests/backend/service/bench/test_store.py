"""store.py: the schema (and a lake written with the retired annotation
columns), upload (inputs, batch, identity duplicates, keys), the in-memory
annotations, review transitions, deletion, and the shared-row rule."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import uuid

import pytest

from backend.inference import benchmarks as _benchmarks
from backend.inference.op_calibration import measured_utilization
from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeConfig
from backend.service.bench.store import SetRef
from tests.backend.service.bench.conftest import _rows, derive
from tests.backend.service.bench.pg import scratch_database


def _md5_upper(text: str) -> int:
    """DuckDB's md5_number_upper: the first 8 digest bytes, little-endian."""
    return int.from_bytes(hashlib.md5(text.encode()).digest()[:8], "little")


def _up(lake, owner, rows, annotate, *, run_id="run-1", upload_id=None, **kw):
    return store.upload(lake, owner=owner, upload_id=upload_id or str(uuid.uuid4()),
                        run_id=run_id, label="lbl", rows=rows, **kw)


def _count(lake, sql="SELECT count(*) AS n FROM lake.measurements", params=()):
    return lake.read(sql, params)[0]["n"]


# --- schema ------------------------------------------------------------------

def test_ensure_schema_is_idempotent_and_types_every_column(lake):
    store.ensure_schema(lake)
    cols = {r["column_name"]: r["data_type"] for r in lake.read(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_catalog = 'lake' AND table_name = 'measurements'")}
    for c in _benchmarks.ALL_COLUMNS:
        assert cols[c] == store.MEASUREMENT_TYPES[c]
    assert {c: cols[c] for c in ("id", "batch_id", "input_id", "env", "created_at")} == {
        "id": "UUID", "batch_id": "BIGINT", "input_id": "BIGINT", "env": "UBIGINT",
        "created_at": "TIMESTAMP"}
    tables = {r["table_name"] for r in lake.read(
        "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'lake'")}
    assert tables == {"batches", "inputs", "measurements", "lake_meta"}
    assert [r["column_name"] for r in lake.read(
        "SELECT column_name FROM information_schema.columns WHERE table_catalog = 'lake' "
        "AND table_name = 'inputs' ORDER BY ordinal_position")] == ["input_id", *store.INPUT_KEY]


# The `inputs` table and its insert as the store wrote them while annotations
# were stored (before they moved to memory).
_OLD_INPUTS = """CREATE TABLE lake.inputs (
        input_id BIGINT NOT NULL, op VARCHAR NOT NULL, phase VARCHAR NOT NULL,
        kernel_impl VARCHAR NOT NULL, shape_json VARCHAR NOT NULL,
        signature VARCHAR, physical_shape VARCHAR, workload_batch INTEGER,
        workload_context INTEGER, catalog_kernel VARCHAR, annotation_rev VARCHAR,
        phys UBIGINT)"""


def test_a_lake_with_stored_annotations_opens_and_serves(tmp_path, make_rows, annotate, new_owner):
    """A lake written while annotations were stored: `ensure_schema` drops
    the annotation columns, the stored labels are never read (stale ones
    stay unread), and uploads and views work."""
    from backend.service.bench.browse import Query, browse
    cfg = LakeConfig(catalog=str(tmp_path / "catalog.ducklake"), data=str(tmp_path / "data"),
                     cache_dir=str(tmp_path / "cache"))
    owner = new_owner()
    [row] = make_rows(1)
    key = {k: row[k] for k in store.INPUT_KEY}

    def old_lake(cur):
        cur.execute(_OLD_INPUTS)
        cur.execute("INSERT INTO lake.inputs VALUES ((md5_number_upper(? || chr(31) || ? || chr(31)"
                    " || ? || chr(31) || ?) >> 1)::BIGINT, ?, ?, ?, ?, 'stale:sig', '{}', 9, 9,"
                    " 'stale', 'old-rev', 1)", [*key.values(), *key.values()])
    lk = Lake.open(cfg)
    lk.transact(old_lake)
    lk.close()

    lk = Lake.open(cfg)
    try:
        store.ensure_schema(lk)
        assert [r["column_name"] for r in lk.read(
            "SELECT column_name FROM information_schema.columns WHERE table_catalog = 'lake' "
            "AND table_name = 'inputs' ORDER BY ordinal_position")] == ["input_id", *store.INPUT_KEY]
        d = derive(lk, annotate)
        [a] = lk.read(f"SELECT * FROM {store.ANNOT}")
        assert a["signature"] == annotate(key)["signature"] != "stale:sig"
        # the old input is known: an upload of its row registers nothing new
        r = _up(lk, owner, [row], annotate, status="shared")
        assert r.inserted == 1 and _count(lk, "SELECT count(*) AS n FROM lake.inputs") == 1
        [e] = browse(lk, Query(source="shared"), d)["items"]
        assert e["key"]["signature"] == annotate(key)["signature"]
    finally:
        lk.close()


def test_a_lake_written_before_a_column_existed_gains_it_and_keeps_serving(
        tmp_path, make_rows, annotate, new_owner):
    """An EXISTING lake must gain a corpus column the schema has since added.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that is already
    there, while `upload` names every column of `ALL_COLUMNS` — so without
    the additive branch in `ensure_schema`, an additive schema bump leaves
    every real lake erroring on its next upload while a freshly created one
    works perfectly. Every other test in this file uses the `lake` fixture,
    which creates the table at today's schema, so none of them can see it.

    The column is dropped and re-added generically rather than named, because
    the next additive bump must be covered without editing this test.
    """
    cfg = LakeConfig(catalog=str(tmp_path / "catalog.ducklake"),
                     data=str(tmp_path / "data"),
                     cache_dir=str(tmp_path / "cache"))
    owner = new_owner()
    rows = make_rows(1)
    # The two columns v14 added, as a lake written at v13 did not have them.
    absent = ("kmd_version", "umd_version")

    lk = Lake.open(cfg)
    try:
        store.ensure_schema(lk)
        lk.transact(lambda cur: [
            cur.execute(f'ALTER TABLE lake.measurements DROP COLUMN "{c}"')
            for c in absent])
        present = {r["column_name"] for r in lk.read(
            "SELECT column_name FROM information_schema.columns WHERE "
            "table_catalog = 'lake' AND table_name = 'measurements'")}
        assert not (present & set(absent)), "the old lake is set up without them"
    finally:
        lk.close()

    lk = Lake.open(cfg)
    try:
        store.ensure_schema(lk)
        cols = {r["column_name"]: r["data_type"] for r in lk.read(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_catalog = 'lake' AND table_name = 'measurements'")}
        for c in _benchmarks.ALL_COLUMNS:
            assert cols[c] == store.MEASUREMENT_TYPES[c], c
        # And the upgraded lake still takes rows — the half that actually
        # broke, and which a column-list assertion alone would not catch.
        r = _up(lk, owner, rows, annotate, status="shared")
        assert r.inserted == 1
        [m] = lk.read("SELECT kmd_version, umd_version FROM lake.measurements")
        assert m["kmd_version"] == rows[0]["kmd_version"]
    finally:
        lk.close()


def test_the_insert_lands_each_value_in_its_own_column(
        lake, make_rows, annotate, new_owner):
    """The insert must be ORDINAL-INDEPENDENT: a column's name decides where
    its value goes, not its position in the SELECT.

    The other existing-lake test above cannot show this. Its named-`INSERT`
    mutation fails with `ConversionException: Could not convert string
    '575.57.08' to INT128` — red at a TYPE boundary, because the two v14
    `VARCHAR`s happen to drift onto the UUID/integer store-owned columns. So
    it proves the insert is typed right, not ordered right, and it would have
    stayed green through the case this branch itself called the worse one:
    adjacent same-type columns silently accepting each other's values.

    Constructed by swapping the NAMES of two same-type provenance columns
    (`run_id` / `box`, both free-form VARCHAR) so their names and ordinals
    disagree while every other column stays aligned. A positional insert then
    transposes exactly those two and nothing else — no conversion error
    anywhere, the upload succeeds, and the row is quietly wrong. That is the
    failure a reviewer demonstrated against this branch, and it is why this
    asserts on TWO values rather than one: a single read-back cannot tell a
    correct insert from a transposition.
    """
    owner = new_owner()
    rows = make_rows(1, run_id="KNOWN-RUN-ID", box="KNOWN-BOX-ID")

    def swap_names(cur):
        for a, b in (("run_id", "_swap"), ("box", "run_id"), ("_swap", "box")):
            cur.execute(
                f'ALTER TABLE lake.measurements RENAME COLUMN "{a}" TO "{b}"')
    lake.transact(swap_names)

    ordinals = {r["column_name"]: r["ordinal_position"] for r in lake.read(
        "SELECT column_name, ordinal_position FROM information_schema.columns "
        "WHERE table_catalog = 'lake' AND table_name = 'measurements'")}
    declared = list(_benchmarks.ALL_COLUMNS)
    assert ordinals["run_id"] > ordinals["box"], "the swap did not take"
    assert declared.index("run_id") < declared.index("box"), (
        "the schema no longer declares these two in the order this relies on")

    assert _up(lake, owner, rows, annotate, status="shared").inserted == 1

    [m] = lake.read("SELECT run_id, box FROM lake.measurements")
    assert (m["run_id"], m["box"]) == ("KNOWN-RUN-ID", "KNOWN-BOX-ID"), (
        f"values landed in each other's columns: run_id={m['run_id']!r} "
        f"box={m['box']!r} — the insert is matching by POSITION, so a lake "
        f"whose physical column order differs from `ALL_COLUMNS` silently "
        f"stores every row's provenance against the wrong column")


def test_a_driverless_row_stays_unknown_through_the_lake(
        lake, make_rows, annotate, new_owner):
    """A row carrying no driver must come back out carrying no driver.

    That is what every row written before schema 14 looks like, and what
    `tools/migrate_schema.py` writes for them on purpose: empty, never a
    presumed value. The store must not helpfully substitute anything, and the
    one comparison that reads the pair must answer UNKNOWN rather than
    "native" — the latter would be a claim about ~295,000 rows nobody made.
    """
    owner = new_owner()
    old, new = make_rows(2)
    old = dict(old, kmd_version="", umd_version="")
    _up(lake, owner, [old, new], annotate, status="shared")
    seen = {r["shape_json"]: r for r in lake.read(
        "SELECT shape_json, kmd_version, umd_version FROM lake.measurements")}
    assert len(seen) == 2

    pre = seen[old["shape_json"]]
    assert not (pre["kmd_version"] or "") and not (pre["umd_version"] or "")
    assert _benchmarks.is_driver_substituted(pre) is None

    # The row that DID carry a driver still has it, so the assertion above is
    # about provenance and not about a write that silently dropped a column.
    post = seen[new["shape_json"]]
    assert post["kmd_version"] == new["kmd_version"] != ""
    assert _benchmarks.is_driver_substituted(post) is False


# --- upload --------------------------------------------------------------------

def test_upload_writes_inputs_batch_and_rows(lake, make_rows, annotate, new_owner):
    owner, upload_id = new_owner(), str(uuid.uuid4())
    rows = make_rows(3)
    r = _up(lake, owner, rows, annotate, upload_id=upload_id)
    assert (r.inserted, r.duplicates, r.upload_id) == (3, 0, upload_id)
    [b] = lake.read("SELECT * FROM lake.batches")
    assert b["batch_id"] == r.batch_id == _md5_upper(f"{owner}\x1f{upload_id}\x1f0") >> 1
    assert (str(b["owner"]), str(b["upload_id"]), b["piece"], b["run_id"], b["label"],
            b["status"], b["within_peak"], b["nominated_at"], b["decided_at"]) == (
        owner, upload_id, 0, "run-1", "lbl", "private", False, None, None)
    derive(lake, annotate)
    inputs = lake.read(f"SELECT * FROM lake.inputs i JOIN {store.ANNOT} a USING (input_id) "
                       "ORDER BY input_id")
    assert len(inputs) == 3
    for i in inputs:
        ann = annotate(i)
        assert {a: i[a] for a in store.ANNOTATIONS} == ann
        assert i["phys"] == _md5_upper(f"{ann['signature']}\x1f{ann['physical_shape']}")
        assert i["input_id"] == _md5_upper("\x1f".join(i[c] for c in store.INPUT_KEY)) >> 1
    m = lake.read("SELECT m.*, i.op AS i_op, i.shape_json AS i_shape FROM lake.measurements m "
                  "JOIN lake.inputs i USING (input_id) ORDER BY m.shape_json")
    assert len(m) == 3 and all(x["batch_id"] == r.batch_id for x in m)
    src = sorted(rows, key=lambda x: x["shape_json"])
    for x, row in zip(m, src, strict=True):
        assert x["i_op"] == row["op"] and x["i_shape"] == x["shape_json"] == row["shape_json"]
        assert x["env"] == _md5_upper(f"{row['dtype_env_json']}\x1f{row['gpu']}")
        assert x["latency_us"] == float(row["latency_us"]) and x["samples"] == int(row["samples"])
        assert x["measured_load_factor"] is None and x["moe_block_member"] is None


def test_json_cells_are_stored_canonical(lake, annotate, new_owner):
    owner = new_owner()
    [row] = _rows(1)
    shape = json.loads(row["shape_json"])
    loose = json.dumps(dict(reversed(list(shape.items()))), indent=1)
    env = json.dumps(dict(reversed(list(json.loads(row["dtype_env_json"]).items()))))
    _up(lake, owner, [{**row, "shape_json": loose, "dtype_env_json": env}], annotate)
    [m] = lake.read("SELECT shape_json, dtype_env_json FROM lake.measurements")
    assert m == {"shape_json": row["shape_json"], "dtype_env_json": row["dtype_env_json"]}
    # the canonical spelling is the same identity
    assert _up(lake, owner, [row], annotate).duplicates == 1


def test_identity_duplicates(lake, make_rows, annotate, new_owner):
    owner, other = new_owner(), new_owner()
    _up(lake, owner, make_rows(3), annotate)
    # same identities, different measurement / provenance: still duplicates
    again = make_rows(3, latency_us="1.0", contributor="@else", run_id="r2")
    r = _up(lake, owner, again, annotate, run_id="r2")
    assert (r.batch_id, r.inserted, r.duplicates) == (None, 0, 3)
    assert _count(lake, "SELECT count(*) AS n FROM lake.batches") == 1
    # partial overlap
    r = _up(lake, owner, make_rows(5), annotate)
    assert (r.inserted, r.duplicates) == (2, 3) and r.batch_id is not None
    # identity is per owner
    r = _up(lake, other, make_rows(5), annotate)
    assert (r.inserted, r.duplicates) == (5, 0)
    assert _count(lake) == 10
    # an identity column differing is a new row
    r = _up(lake, owner, make_rows(1, gpu="H200"), annotate)
    assert r.inserted == 1
    # inputs are shared across owners, registered once
    assert _count(lake, "SELECT count(*) AS n FROM lake.inputs") == 5


def test_reused_upload_id_is_refused(lake, make_rows, annotate, new_owner):
    owner, upload_id = new_owner(), str(uuid.uuid4())
    _up(lake, owner, make_rows(1), annotate, upload_id=upload_id)
    with pytest.raises(ValueError, match="already has upload"):
        _up(lake, owner, make_rows(2), annotate, upload_id=upload_id)
    assert _count(lake) == 1


def test_upload_with_a_decided_status(lake, make_rows, annotate, new_owner):
    admin = new_owner()
    _up(lake, new_owner(), make_rows(1), annotate, status="shared", decided_by=admin)
    [b] = lake.read("SELECT status, decided_at, decided_by FROM lake.batches")
    assert b["status"] == "shared" and b["decided_at"] is not None
    assert str(b["decided_by"]) == admin


# --- set references and review --------------------------------------------------

def test_batches_of(lake, make_rows, annotate, new_owner):
    owner = new_owner()
    a = _up(lake, owner, make_rows(1), annotate, run_id="r")
    b = _up(lake, owner, make_rows(2), annotate, run_id="r")
    c = _up(lake, owner, make_rows(3), annotate, run_id=None)
    _up(lake, new_owner(), make_rows(1), annotate, run_id="r")
    run = store.batches_of(lake, SetRef(owner, "run", "r"))
    assert [x["batch_id"] for x in run] == [a.batch_id, b.batch_id]
    up = store.batches_of(lake, SetRef(owner, "upload", c.upload_id))
    assert [x["batch_id"] for x in up] == [c.batch_id]
    assert store.batches_of(lake, SetRef(new_owner(), "run", "r")) == []


def test_review_transitions(lake, make_rows, annotate, new_owner):
    owner, admin = new_owner(), new_owner()
    a = _up(lake, owner, make_rows(1), annotate, run_id="r")
    b = _up(lake, owner, make_rows(2), annotate, run_id="r")
    run = SetRef(owner, "run", "r")

    nominated = store.set_status(lake, [run], from_status=["private"], to_status="nominated",
                                 within_peak=True)
    assert {x["batch_id"] for x in nominated} == {a.batch_id, b.batch_id}
    assert all(x["status"] == "nominated" and x["within_peak"] and x["nominated_at"]
               for x in nominated)
    # nothing private left to nominate
    assert store.set_status(lake, [run], from_status=["private"], to_status="nominated") == []

    ref_a = SetRef(owner, "upload", a.upload_id)
    [w] = store.set_status(lake, [ref_a], from_status=["nominated"], to_status="private")
    assert (w["status"], w["nominated_at"], w["within_peak"]) == ("private", None, True)

    [p] = store.set_status(lake, [run], from_status=["nominated"], to_status="shared",
                           decided_by=admin)
    assert (p["batch_id"], p["status"], str(p["decided_by"])) == (b.batch_id, "shared", admin)
    assert p["decided_at"] is not None and p["nominated_at"] is not None

    store.set_status(lake, [ref_a], from_status=["private"], to_status="nominated")
    [r] = store.set_status(lake, [ref_a, run], from_status=["nominated"], to_status="rejected",
                           decided_by=admin)
    assert (r["batch_id"], r["status"]) == (a.batch_id, "rejected")
    # rows never change on review (b's first row duplicated a's)
    assert _count(lake) == 2


# --- deletion ---------------------------------------------------------------------

def test_delete_set(lake, make_rows, annotate, new_owner):
    owner, other = new_owner(), new_owner()
    _up(lake, owner, make_rows(2), annotate, run_id="r")
    _up(lake, owner, make_rows(3), annotate, run_id="r")      # 1 new row
    keep = _up(lake, owner, make_rows(4), annotate, run_id="s")
    _up(lake, other, make_rows(2), annotate, run_id="r")
    assert store.delete_set(lake, SetRef(owner, "run", "r")) == 3
    assert store.delete_set(lake, SetRef(owner, "run", "r")) == 0
    left = lake.read("SELECT owner::VARCHAR AS owner, batch_id FROM lake.batches ORDER BY owner")
    assert sorted((x["owner"], x["batch_id"]) for x in left) == sorted(
        [(owner, keep.batch_id), (other, left[0]["batch_id"] if left[0]["owner"] == other
                                  else left[1]["batch_id"])])
    assert _count(lake) == 1 + 2
    # a deleted row's identity can be uploaded again
    assert _up(lake, owner, make_rows(2), annotate).inserted == 2


def test_delete_rows(lake, make_rows, annotate, new_owner):
    owner, other = new_owner(), new_owner()
    a = _up(lake, owner, make_rows(2), annotate)
    b = _up(lake, owner, make_rows(3), annotate)          # 1 new row
    o = _up(lake, other, make_rows(1), annotate)
    ids = {x["batch_id"]: [str(r["id"]) for r in lake.read(
        "SELECT id FROM lake.measurements WHERE batch_id = ?", [x["batch_id"]])]
        for x in lake.read("SELECT batch_id FROM lake.batches")}
    # another owner's ids are not deletable
    assert store.delete_rows(lake, owner=owner, ids=ids[o.batch_id]) == 0
    assert store.delete_rows(lake, owner=owner, ids=[]) == 0
    assert store.delete_rows(lake, owner=owner, ids=ids[a.batch_id][:1]) == 1
    # the emptied batch goes with its last row; an id already gone is no row
    assert store.delete_rows(lake, owner=owner, ids=ids[b.batch_id] + ids[a.batch_id][:1]) == 1
    assert {x["batch_id"] for x in lake.read("SELECT batch_id FROM lake.batches")} == {
        a.batch_id, o.batch_id}
    assert _count(lake) == 2


def test_restamp_rewrites_the_roofline_columns_in_place(lake, make_rows, annotate, new_owner):
    """A restamped row keeps its batch — so a nominated row stays nominated —
    and its latency; a column the patch leaves out keeps its value; another
    owner's row is not touched."""
    owner, other = new_owner(), new_owner()
    mine = _up(lake, owner, make_rows(2), annotate, status="nominated")
    theirs = _up(lake, other, make_rows(1), annotate)
    cols = "id, batch_id, p50_us, " + ", ".join(store.ROOFLINE_COLUMNS)
    before = {str(r["id"]): r for r in lake.read(f"SELECT {cols} FROM lake.measurements")}
    [a, b] = [i for i, r in before.items() if r["batch_id"] == mine.batch_id]
    [o] = [i for i, r in before.items() if r["batch_id"] == theirs.batch_id]
    patch = [{"id": a, "achieved_bw_gbps": 12.5, "achieved_tflops": 0.0},
             {"id": b, "peak_bw_at_test": 999.0},
             {"id": o, "achieved_bw_gbps": 1.0}]
    assert store.restamp(lake, owner=owner, rows=patch) == 2
    after = {str(r["id"]): r for r in lake.read(f"SELECT {cols} FROM lake.measurements")}
    assert (after[a]["achieved_bw_gbps"], after[a]["achieved_tflops"]) == (12.5, 0.0)
    assert after[b]["peak_bw_at_test"] == 999.0
    for c in store.ROOFLINE_COLUMNS:
        if c not in ("achieved_bw_gbps", "achieved_tflops"):
            assert after[a][c] == before[a][c]
        if c != "peak_bw_at_test":
            assert after[b][c] == before[b][c]
    assert after[o] == before[o]
    assert all(after[i]["p50_us"] == before[i]["p50_us"]
               and after[i]["batch_id"] == before[i]["batch_id"] for i in before)
    assert {r["status"] for r in lake.read(
        "SELECT status FROM lake.batches WHERE batch_id = ?", [mine.batch_id])} == {"nominated"}
    assert store.restamp(lake, owner=owner, rows=[]) == 0


# --- the shared-row rule -------------------------------------------------------------------

_PEAK_CASES = [
    # achieved_tflops, achieved_bw_gbps, peak_tflops, peak_bw
    ("690", "920", "989", "3350"),
    ("990", "920", "989", "3350"),          # over the compute peak
    ("690", "3351", "989", "3350"),         # over the bandwidth peak
    ("1000", "4000", "989", "3350"),        # over both
    ("989", "3350", "989", "3350"),         # exactly at both
    ("690", "920", "", "3350"),             # no compute peak: not disqualifying
    ("690", "4000", "989", ""),
    ("690", "920", "0", "0"),               # zero peaks
    ("690", "920", "-1", "-5"),             # non-positive peaks
    ("", "920", "989", "3350"),             # no achieved rate counts as 0
    ("0", "0", "989", "3350"),
    ("1.0000000000000002", "1", "1", "1"),  # one ulp over
    ("0.30000000000000004", "1", "0.3", "1"),
    ("7", "1", "3", "1"),
]


def test_peak_rule_matches_measured_utilization(lake):
    """`MFU`, `MBU` and `WITHIN_PEAK` against `measured_utilization` on the
    same values: the same ratios (None where it has none), and a row within
    iff the Python rule reports no ratio over 1."""
    con = lake._con.cursor()
    try:
        for case in _PEAK_CASES:
            row = dict(zip(("achieved_tflops", "achieved_bw_gbps", "peak_tflops_at_test",
                            "peak_bw_at_test"), case, strict=True))
            py = measured_utilization(row)
            got = con.execute(
                f"SELECT {store.MFU}, {store.MBU}, {store.WITHIN_PEAK} FROM (SELECT "
                + ", ".join(f"CAST(NULLIF(?, '') AS DOUBLE) AS {k}" for k in row)
                + ") m", list(row.values())).fetchone()
            assert got == (py["mfu"], py["mbu"], py["roof_status"] in ("ok", None)), (case, got, py)
    finally:
        con.close()


def test_shared_row(lake, annotate, new_owner):
    """SHARED_ROW over stored rows: every batch status × within_peak, rows
    within and over their peaks."""
    over = _rows(2, achieved_tflops=lambda i: "2000" if i else "100")   # row 1 is over
    expected = set()
    for status in ("private", "nominated", "shared", "rejected"):
        for within in (False, True):
            owner = new_owner()
            r = _up(lake, owner, over, annotate, status=status)
            lake.transact(lambda cur, w=within, b=r.batch_id: cur.execute(
                "UPDATE lake.batches SET within_peak = ? WHERE batch_id = ?", [w, b]))
            for m in lake.read("SELECT id, achieved_tflops FROM lake.measurements "
                               "WHERE batch_id = ?", [r.batch_id]):
                if status == "shared" and (not within or m["achieved_tflops"] < 989):
                    expected.add(m["id"])
    got = {x["id"] for x in lake.read(
        "SELECT m.id FROM lake.measurements m JOIN lake.batches b USING (batch_id) "
        f"WHERE {store.SHARED_ROW}")}
    assert got == expected and len(expected) == 3


# --- two processes, one owner, one Postgres catalog -----------------------------------

class _SlowLake(Lake):
    """A lake whose write transactions hold for a while before committing,
    so two processes' transactions overlap."""

    def transact(self, fn):
        def slow(cur):
            out = fn(cur)
            import time
            time.sleep(0.5)
            return out
        return super().transact(slow)


def _race_upload(config, owner, n, ready, start, results, lock_owner):
    import contextlib
    lk = _SlowLake.open(config)
    if not lock_owner:
        lk.owner_lock = lambda owner: contextlib.nullcontext()
    try:
        ready.put(True)
        start.wait(30)
        r = store.upload(lk, owner=owner, upload_id=str(uuid.uuid4()), run_id="race",
                         label=None, rows=_rows(n))
        results.put(r.inserted)
    finally:
        lk.close()


def _race(config, lock_owner: bool) -> tuple[list[int], int, int]:
    """Two processes upload the same rows for one owner at the same moment;
    returns (inserted per process, rows stored, distinct identities)."""
    lk = Lake.open(config)
    store.ensure_schema(lk)
    # the inputs exist already (another owner's upload), so neither process
    # takes the inputs lock and only the owner lock can serialize them
    store.upload(lk, owner=str(uuid.uuid4()), upload_id=str(uuid.uuid4()), run_id=None,
                 label=None, rows=_rows(4))
    ctx = multiprocessing.get_context("spawn")
    ready, start, results = ctx.Queue(), ctx.Event(), ctx.Queue()
    owner = str(uuid.uuid4())
    procs = [ctx.Process(target=_race_upload,
                         args=(config, owner, 4, ready, start, results, lock_owner))
             for _ in range(2)]
    for p in procs:
        p.start()
    # Release both only once both have imported and attached: a fixed sleep
    # let a process still starting under load begin after the other had
    # committed, so the unlocked control saw no race.
    for _ in procs:
        ready.get(timeout=120)
    start.set()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    inserted = sorted(results.get(timeout=5) for _ in procs)
    [c] = lk.read(
        "SELECT count(*) AS n, count(DISTINCT (" + ", ".join(_benchmarks.IDENTITY_COLUMNS)
        + ")) AS d FROM lake.measurements m JOIN lake.batches b USING (batch_id) "
        "WHERE b.owner = ?::UUID", [owner])
    lk.close()
    return inserted, c["n"], c["d"]


def test_two_process_same_owner_upload_race(tmp_path):
    with scratch_database("bench_lake_test_store") as dsn:
        config = LakeConfig(catalog=f"postgres:{dsn}", data=str(tmp_path / "data"),
                            cache_dir=str(tmp_path / "cache"))
        assert _race(config, lock_owner=True) == ([0, 4], 4, 4)
    # control: without the owner lock the same race does store duplicates,
    # so the test above can see the race it guards against
    with scratch_database("bench_lake_test_store") as dsn:
        config = LakeConfig(catalog=f"postgres:{dsn}", data=str(tmp_path / "data2"),
                            cache_dir=str(tmp_path / "cache2"))
        assert _race(config, lock_owner=False) == ([4, 4], 8, 4)


def test_rekey_env_drops_the_deployment_from_an_old_lake(lake, make_rows, annotate, new_owner):
    """A lake written while `env` named the deployment is rekeyed once:
    rows of one shape on one card under two deployments share an env."""
    owner = new_owner()
    base = make_rows(1)[0]
    rows = [{**base, "spec_hash": "a" * 16, "parallel": "tp=8"},
            {**base, "spec_hash": "b" * 16, "parallel": "ep=8"}]
    _up(lake, owner, rows, annotate)
    lake.transact(lambda cur: cur.execute(
        "UPDATE lake.measurements SET env = md5_number_upper(dtype_env_json || chr(31) || gpu"
        " || chr(31) || parallel)"))
    assert len({r["env"] for r in lake.read("SELECT env FROM lake.measurements")}) == 2
    assert store.rekey_env(lake) == 2
    assert store.rekey_env(lake) == 0
    (env,) = {r["env"] for r in lake.read("SELECT env FROM lake.measurements")}
    assert env == _md5_upper(f"{base['dtype_env_json']}\x1f{base['gpu']}")
