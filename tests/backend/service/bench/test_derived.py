"""The lake's derived file (`derived.Derived`): the annotations, the browse
tables and the lookup's cells of one lake snapshot, kept by one refresh.

What it promises: a start reuses the file the last one left (no input is
annotated again); a snapshot that moved is followed by the change feed (only
new inputs annotated, only changed rows rewritten); other annotation rules
rebuild it; and a file that does not hold what the lake holds is refused
loudly, never served. The browse tables are checked against a second route:
the same tables recomputed from the lake at the snapshot."""
from __future__ import annotations

import json
import uuid

import pytest

from backend.service.bench import browse as B
from backend.service.bench import derived as D
from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeConfig
from tests.backend.service.bench.conftest import make_annotator


def _upload(lake, owner, rows, *, status="shared"):
    out = store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id=None, label=None,
                       rows=rows, status=status,
                       decided_by=owner if status == "shared" else None)
    return out


def _counting(annotate):
    seen: list = []

    def counted(inp):
        seen.append((inp["op"], inp["phase"], inp["kernel_impl"], inp["shape_json"]))
        return annotate(inp)
    return counted, seen


def _table(lake, sql: str) -> list:
    return sorted(tuple(str(v) for v in r.values()) for r in lake.read(sql))


def _config(tmp_path) -> LakeConfig:
    return LakeConfig(catalog=str(tmp_path / "catalog.ducklake"), data=str(tmp_path / "data"),
                      cache_dir=str(tmp_path / "cache"))


def test_a_start_reuses_the_file_the_last_one_left(tmp_path, make_rows, new_owner):
    """A new process under the same rules finds the file at the lake's
    snapshot: nothing is annotated or rebuilt before it serves."""
    lk = Lake.open(_config(tmp_path))
    store.ensure_schema(lk)
    _upload(lk, new_owner(), make_rows(4))
    first, seen = _counting(make_annotator())
    assert D.Derived(annotate=first, rules=lambda: "r").refresh(lk)["how"] == "rebuilt"
    assert len(seen) == 4
    lk.close()
    lk = Lake.open(_config(tmp_path))
    try:
        again, seen = _counting(make_annotator())
        assert D.Derived(annotate=again, rules=lambda: "r").refresh(lk)["how"] == "current"
        assert seen == []
        assert lk.read(f"SELECT count(*) AS n FROM {store.ANNOT}")[0]["n"] == 4
    finally:
        lk.close()


def test_a_refresh_annotates_only_the_inputs_it_lacks(lake, make_rows, new_owner):
    counted, seen = _counting(make_annotator())
    d = D.Derived(annotate=counted, rules=lambda: "r")
    rows = make_rows(5)
    _upload(lake, new_owner(), rows[:2])
    d.refresh(lake)
    _upload(lake, new_owner(), rows[:4])          # two known inputs, two new
    assert d.refresh(lake)["how"] == "changes"
    assert len(seen) == 4 == len(set(seen))


def test_other_rules_rebuild_every_annotation(lake, make_rows, new_owner):
    _upload(lake, new_owner(), make_rows(3))
    D.Derived(annotate=make_annotator(), rules=lambda: "one").refresh(lake)
    other = make_annotator(prefix="sig2")
    assert D.Derived(annotate=other, rules=lambda: "two").refresh(lake)["how"] == "rebuilt"
    got = lake.read(f"SELECT * FROM lake.inputs i JOIN {store.ANNOT} a USING (input_id)")
    assert len(got) == 3
    for i in got:
        assert {a: i[a] for a in store.ANNOTATIONS} == other(i)


def test_an_input_annotate_refuses_is_a_placeholder(lake, make_rows, new_owner):
    """Its entry is there — one per input — with no annotations: its rows
    are in scope and in no envelope."""
    def refuse_one(inp):
        if json.loads(inp["shape_json"])["s_ctx"] == 4096:
            raise ValueError("no signature")
        return make_annotator()(inp)
    _upload(lake, new_owner(), make_rows(3))       # s_ctx 2048, 4096, 6144
    D.Derived(annotate=refuse_one, rules=lambda: "r").refresh(lake)
    got = lake.read(f"SELECT signature, phys FROM {store.ANNOT}")
    assert len(got) == 3 and sum(r["signature"] is None and r["phys"] is None for r in got) == 1


def test_a_catalog_that_cannot_load_fails_the_refresh(lake, make_rows, new_owner):
    _upload(lake, new_owner(), make_rows(1))
    d = D.Derived(annotate=make_annotator(), rules=lambda: "r")
    d.refresh(lake)
    held = d.held(lake)

    def broken(inp):
        raise ImportError("no kernel_bench")
    _upload(lake, new_owner(), make_rows(2))
    with pytest.raises(ImportError):
        D.Derived(annotate=broken, rules=lambda: "r").refresh(lake)
    assert d.held(lake) == held
    # nothing of the failed refresh was kept: the file is where it was
    assert lake.read(f"SELECT newest FROM {D.A}.meta")[0]["newest"] == held[1]


def test_annotations_go_in_batches(lake, make_rows, new_owner, monkeypatch):
    """One JSON text per `BATCH` inputs — one holding every input's outgrew
    the lake's memory limit at 300k inputs — and the answer is whole."""
    _upload(lake, new_owner(), make_rows(5))
    monkeypatch.setattr(D, "BATCH", 2)
    sizes: list = []
    real = D.Derived._pending

    def measured(cur, select, order):
        for chunk in real(cur, select, order):
            sizes.append(len(chunk))
            yield chunk
    monkeypatch.setattr(D.Derived, "_pending", staticmethod(measured))
    D.Derived(annotate=make_annotator(), rules=lambda: "r").refresh(lake)
    assert sizes[:3] == [2, 2, 1]                 # the inputs; the placement follows
    assert lake.read(f"SELECT count(DISTINCT input_id) AS n FROM {store.ANNOT}")[0]["n"] == 5


def test_the_browse_tables_follow_the_lake(lake, make_rows, new_owner):
    """Inserts, a delete, a restamp and a review decision, each followed by
    the change feed: the browse rows, groups and placement equal the same
    tables recomputed from the lake at the snapshot (the second route)."""
    d = D.Derived(annotate=make_annotator(), rules=lambda: "r")
    alice, bob = new_owner(), new_owner()
    rows = make_rows(6)
    _upload(lake, alice, rows[:3])
    d.refresh(lake)
    second = _upload(lake, bob, rows[2:], status="private")
    ids = [r["id"] for r in lake.read("SELECT id::VARCHAR AS id FROM lake.measurements ORDER BY id")]
    store.delete_rows(lake, owner=alice, ids=[i for i in ids if i in {
        r["id"] for r in lake.read("SELECT m.id::VARCHAR AS id FROM lake.measurements m "
                                   "JOIN lake.batches b USING (batch_id) WHERE b.owner = ?::UUID",
                                   [alice])}][:1])
    bob_ids = [r["id"] for r in lake.read(
        "SELECT m.id::VARCHAR AS id FROM lake.measurements m JOIN lake.batches b "
        "USING (batch_id) WHERE b.owner = ?::UUID", [bob])]
    store.restamp(lake, owner=bob, rows=[{"id": bob_ids[0], "achieved_tflops": 1.25}])
    store.set_status(lake, [store.SetRef(bob, "upload", second.upload_id)],
                     from_status=["private"], to_status="shared", within_peak=True)
    assert d.refresh(lake)["how"] == "changes"
    snap = lake.snapshot_id()
    fresh = B.rows_select(f"(SELECT * FROM lake.measurements AT (VERSION => {snap}))")
    cols = "* EXCLUDE (envelope)"
    assert _table(lake, f"SELECT {cols} FROM {B.ROWS}") == _table(lake, f"SELECT {cols} FROM ({fresh})")
    assert _table(lake, f"SELECT envelope FROM {B.ROWS}") == _table(lake, f"SELECT envelope FROM ({fresh})")
    assert _table(lake, f"SELECT * FROM {B.GROUPS}") == _table(
        lake, B.groups_select().replace(B.ROWS, f"({fresh})"))
    assert _table(lake, f"SELECT * FROM {B.BATCHES}") == _table(lake, "SELECT * FROM lake.batches")
    tokens = B.call_tokens()
    place = B._Placement(tokens["runs"], tokens["total"], tokens["factor"])
    want = [place.entry(r["phys"], r["signature"], r["physical_shape"]) for r in lake.read(
        f"SELECT DISTINCT phys, signature, physical_shape FROM {store.ANNOT} WHERE phys IS NOT NULL")]
    assert _table(lake, f"SELECT * FROM {B.PLACEMENT}") == sorted(
        tuple(str(e[c]) for c in B.PLACEMENT_TYPES) for e in want)


def test_a_file_that_does_not_hold_the_lake_is_refused_loudly(lake, make_rows, new_owner):
    """A table the refresh leaves short of the lake (here an annotation lost
    behind the builder's back) fails the refresh by name; the file stays
    where it was and reads behind it keep waiting — never a short answer."""
    d = D.Derived(annotate=make_annotator(), rules=lambda: "r")
    _upload(lake, new_owner(), make_rows(2))
    d.refresh(lake)
    held = d.held(lake)
    lake.transact(lambda cur: cur.execute(
        f"DELETE FROM {store.ANNOT} WHERE input_id = (SELECT min(input_id) FROM {store.ANNOT})"))
    _upload(lake, new_owner(), make_rows(3))
    with pytest.raises(D.Inconsistent, match="annot has"):
        d.refresh(lake)
    assert d.held(lake) == held


def test_other_token_axes_rebuild_the_placement(lake, make_rows, new_owner, caplog):
    """The curves' placement is a Python function of the declared token
    axes (`call_tokens()`), outside the annotation rules: a file built
    under other axes - or other placement code - is of another form and is
    rebuilt, never served with stale curves."""
    _upload(lake, new_owner(), make_rows(2))
    tokens = B.call_tokens()
    D.Derived(annotate=make_annotator(), rules=lambda: "r", tokens=lambda: tokens).refresh(lake)
    same = D.Derived(annotate=make_annotator(), rules=lambda: "r", tokens=lambda: tokens)
    assert same.refresh(lake)["how"] == "current"
    other = {**tokens, "total": [*tokens["total"], "n_other"]}
    moved = D.Derived(annotate=make_annotator(), rules=lambda: "r", tokens=lambda: other)
    with caplog.at_level("WARNING", logger=D.__name__):
        assert moved.refresh(lake)["how"] == "rebuilt"
    assert "the derived tables' form changed" in caplog.text
