"""The Kernel Bench service (docs/measured_calibration.md §5.1): every
request needs the shared secret from a loopback peer; identity comes only
from the headers the HA API sets; `/internal/bench/lookup` answers rows per
requested key — selected by the current annotations, at a pinned snapshot,
shared rows strict and the owner's counted — and `/internal/bench/version`
names what a measured estimate is a function of."""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

import tests.corpus_row as _corpus
from backend.measure.scope import MeasurementScope
from backend.service.bench import annotations as A
from backend.service.bench import store

from ..conftest import BENCH_SECRET, as_user

ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ADMIN = "33333333-3333-4333-8333-333333333333"


def _row(i: int, **over) -> dict:
    shape = json.loads(_corpus.row()["shape_json"])
    shape["s_ctx"] = 2048 * (i + 1)
    return _corpus.row(shape_json=json.dumps(shape, sort_keys=True, separators=(",", ":")),
                       **over)


def _upload(client, user: str, rows: list, *, public: bool = False) -> dict:
    r = client.post("/api/bench/upload", headers=as_user(user, admin=public),
                    json={"csv_text": _corpus.to_csv(rows), "label": "x", "make_public": public})
    assert r.status_code == 200, r.text
    return r.json()


def _key(row: dict) -> dict:
    """The lookup key an estimate builds for the row's physical op."""
    ann = A.row_annotations(row)
    return {"signature": ann["signature"], "physical_shape": ann["physical_shape"],
            "dtype_env_json": row["dtype_env_json"], "gpu": row["gpu"], "phase": row["phase"]}


def _lookup(client, keys, owner=None, snapshot=None):
    r = client.post("/internal/bench/lookup",
                    json={"keys": keys, "owner": owner, "snapshot": snapshot})
    assert r.status_code == 200, r.text
    out = r.json()
    cols, values = out["columns"], out["values"]
    out["rows"] = [[dict(zip(cols, (values[i] for i in cells), strict=True)) for cells in per_key]
                   for per_key in out["rows"]]
    return out


def _version(client) -> dict:
    r = client.get("/internal/bench/version")
    assert r.status_code == 200, r.text
    return r.json()


class TestTheBoundary:
    def test_every_request_needs_the_secret_from_loopback(self, bench_app, bench_lake):
        for client in (TestClient(bench_app, client=("127.0.0.1", 1)),
                       TestClient(bench_app, client=("127.0.0.1", 1),
                                  headers={"X-HA-Internal": "wrong"}),
                       TestClient(bench_app, client=("10.0.0.5", 1),
                                  headers={"X-HA-Internal": BENCH_SECRET})):
            for method, path in (("GET", "/health"), ("GET", "/internal/bench/version"),
                                 ("POST", "/internal/bench/lookup"), ("GET", "/api/bench/list"),
                                 ("GET", "/api/bench/providers")):
                r = client.request(method, path, headers=as_user(ADMIN, admin=True), json={})
                assert r.status_code == 403, (method, path, r.text)
        ok = TestClient(bench_app, client=("::1", 1), headers={"X-HA-Internal": BENCH_SECRET})
        assert ok.get("/health").json()["status"] == "ok"

    def test_no_secret_configured_refuses_everything(self, bench_app, bench_lake, monkeypatch):
        monkeypatch.delenv("HUGGINGARCH_BENCH_SECRET")
        client = TestClient(bench_app, client=("127.0.0.1", 1), headers={"X-HA-Internal": ""})
        assert client.get("/health").status_code == 403

    def test_identity_is_the_forwarded_headers(self, bench_client, bench_lake):
        assert bench_client.get("/api/bench/list").status_code == 401
        assert bench_client.get("/api/bench/list", headers={"X-HA-User": ""}).status_code == 401
        assert bench_client.get("/api/bench/list", headers=as_user(ALICE)).status_code == 200
        # Admin only by the admin header.
        assert bench_client.get("/api/bench/nominations",
                                headers=as_user(ALICE)).status_code == 403
        assert bench_client.get("/api/bench/nominations",
                                headers=as_user(ALICE, admin=True)).status_code == 200


class TestLookup:
    def test_rows_per_key_and_the_owners_over_the_shared(self, bench_client, bench_lake):
        a, b, c = _row(0), _row(1), _row(2)
        _upload(bench_client, ADMIN, [a, b], public=True)
        _upload(bench_client, ALICE, [_row(0, contributor="alice", kernel_impl="mine_kernel")])
        keys = [_key(a), _key(b), _key(c)]
        anon = _lookup(bench_client, keys)
        assert [len(k) for k in anon["rows"]] == [1, 1, 0]
        assert {r["origin"] for k in anon["rows"] for r in k} == {"shared"}
        mine = _lookup(bench_client, keys, owner=ALICE)
        assert [r["origin"] for r in mine["rows"][0]] == ["shared", "mine"]
        assert mine["rows"][0][1]["contributor"] == "alice"
        assert mine["excluded_mine"] == 0
        # Bob sees no row of Alice's.
        assert [len(k) for k in _lookup(bench_client, keys, owner=BOB)["rows"]] == [1, 1, 0]
        # The cells are the CSV text the validator reads.
        assert anon["rows"][0][0]["latency_us"] == "387.5"
        assert anon["rows"][0][0]["shape_json"] == a["shape_json"]

    def test_a_shared_row_of_the_owner_comes_once_per_origin(self, bench_client, bench_lake):
        a = _row(0)
        _upload(bench_client, ADMIN, [a], public=True)
        got = _lookup(bench_client, [_key(a)], owner=ADMIN)["rows"][0]
        assert [r["origin"] for r in got] == ["shared", "mine"]

    def test_the_answer_is_pinned_at_the_snapshot(self, bench_client, bench_lake):
        a, b = _row(0), _row(1)
        first = _upload(bench_client, ALICE, [a])
        pinned = _version(bench_client)["snapshot"]
        # After the pin: a row of a NEW input (its annotations join `annot`
        # now), and the first upload deleted.
        _upload(bench_client, ALICE, [b])
        assert bench_client.delete(f"/api/bench/{first['upload_id']}",
                                   headers=as_user(ALICE)).json()["deleted"]
        keys = [_key(a), _key(b)]
        then = _lookup(bench_client, keys, owner=ALICE, snapshot=pinned)
        assert then["snapshot"] == pinned
        assert [len(k) for k in then["rows"]] == [1, 0]
        now = _lookup(bench_client, keys, owner=ALICE)
        assert now["snapshot"] > pinned and [len(k) for k in now["rows"]] == [0, 1]

    def test_review_state_is_pinned_too(self, bench_client, bench_lake):
        """`batches` is read at the snapshot as well: a promote after the pin
        does not make a row shared in the pinned answer."""
        a = _row(0)
        up = _upload(bench_client, ALICE, [a])
        assert bench_client.post(f"/api/bench/{up['upload_id']}/contribute",
                                 headers=as_user(ALICE)).status_code == 200
        pinned = _version(bench_client)["snapshot"]
        r = bench_client.post("/api/bench/promote", headers=as_user(ADMIN, admin=True),
                              json={"sets": [{"owner": ALICE, "kind": "upload",
                                              "id": up["upload_id"]}]})
        assert r.status_code == 200, r.text
        assert _lookup(bench_client, [_key(a)], snapshot=pinned)["rows"] == [[]]
        assert len(_lookup(bench_client, [_key(a)])["rows"][0]) == 1

    def test_an_expired_snapshot_is_503_not_empty(self, bench_client, bench_lake):
        _upload(bench_client, ALICE, [_row(0)])
        r = bench_client.post("/internal/bench/lookup",
                              json={"keys": [_key(_row(0))], "snapshot": 10 ** 9})
        assert r.status_code == 503, r.text

    def test_a_legacy_shape_is_found_under_the_current_key(self, bench_client, bench_lake,
                                                           monkeypatch):
        """Membership is the service's current annotations (`current_shape`,
        then `physical_identity`), not a projection of the stored shape: a
        row whose stored shape spells a retired vocabulary — here a stand-in
        migration that reads a stored `d_head` 64 as today's 128 — is found
        under the current key, and sent as stored. (Today's real retired
        letters fail the current validator, so they never reach a lookup.)"""
        def migrate(op, shape_json):
            dims = json.loads(shape_json)
            if dims.get("d_head") == 64:
                dims["d_head"] = 128
            return json.dumps(dims, sort_keys=True, separators=(",", ":"))
        monkeypatch.setattr(A, "current_shape", migrate)
        stored = json.loads(_row(0)["shape_json"])
        stored["d_head"] = 64
        row = _corpus.row(shape_json=json.dumps(stored, sort_keys=True, separators=(",", ":")))
        _upload(bench_client, ADMIN, [row], public=True)
        current = _key(row)
        raw = {**current, "physical_shape": A.physical_identity(
            row["op"], row["shape_json"])["physical_shape"]}
        assert json.loads(current["physical_shape"])["d_head"] == 128
        assert json.loads(raw["physical_shape"])["d_head"] == 64
        got = _lookup(bench_client, [current, raw])["rows"]
        assert len(got[0]) == 1 and got[1] == []
        assert json.loads(got[0][0]["shape_json"])["d_head"] == 64

    def test_the_write_path_refuses_bad_rows_and_lookup_does_not_revalidate(
            self, bench_client, bench_lake, monkeypatch):
        old = _corpus.row(contributor="old", shape_json=json.dumps(
            {"B": 1, "H": 4, "s_ctx": 77}, sort_keys=True, separators=(",", ":")))
        good = _row(0)
        bench_client.get("/api/bench/list", headers=as_user(ALICE))       # the lake serves
        refused = bench_client.post("/api/bench/upload", headers=as_user(ALICE), json={
            "csv_text": _corpus.to_csv([old]), "label": "old"})
        assert refused.status_code == 422
        assert "unknown symbol 'H'" in refused.json()["detail"]
        _upload(bench_client, ALICE, [good])
        keys = [_key(old), _key(good)]
        report = bench_client.get("/api/bench/calibration", headers=as_user(ALICE)).json()
        assert report["rows_rejected"] == 0
        from backend.inference import benchmarks

        def unreachable(_row):
            raise AssertionError("lookup must not run the row validator")

        monkeypatch.setattr(benchmarks, "validate_row", unreachable)
        got = _lookup(bench_client, keys, owner=ALICE)
        assert got["excluded_mine"] == 0
        assert [len(k) for k in got["rows"]] == [0, 1]

    def test_bounds(self, bench_client, bench_lake):
        bench_client.get("/api/bench/list", headers=as_user(ALICE))
        r = bench_client.post("/internal/bench/lookup", json={"keys": [], "owner": "nobody"})
        assert r.status_code == 400
        assert _lookup(bench_client, [])["rows"] == []


class TestVersion:
    def test_a_private_upload_moves_the_snapshot_not_the_shared_rows(
            self, bench_client, bench_lake):
        _upload(bench_client, ADMIN, [_row(0)], public=True)
        v1 = _version(bench_client)
        assert v1["shared_snapshot"] <= v1["snapshot"] and v1["rules_rev"]
        _upload(bench_client, ALICE, [_row(1)])
        v2 = _version(bench_client)
        assert v2["snapshot"] > v1["snapshot"]
        assert v2["shared_snapshot"] == v1["shared_snapshot"]
        assert (v2["lake_id"], v2["rules_rev"]) == (v1["lake_id"], v1["rules_rev"])
        _upload(bench_client, ADMIN, [_row(2)], public=True)
        v3 = _version(bench_client)
        assert v3["shared_snapshot"] > v2["shared_snapshot"]
        # A shared row deleted moves it too.
        up = _upload(bench_client, ADMIN, [_row(3)], public=True)
        v4 = _version(bench_client)
        bench_client.delete(f"/api/bench/{up['upload_id']}", headers=as_user(ADMIN, admin=True))
        assert _version(bench_client)["shared_snapshot"] > v4["shared_snapshot"]

    def test_the_calibration_report_names_the_owners_scope(self, bench_client, bench_lake):
        _upload(bench_client, ALICE, [_row(0), _row(1)])
        report = bench_client.get("/api/bench/calibration", headers=as_user(ALICE)).json()
        v = _version(bench_client)
        assert (report["rows_used"], report["rows_rejected"]) == (2, 0)
        assert report["measurement_scope"] == MeasurementScope.from_version(v, ALICE).digest
        assert bench_client.get("/api/bench/calibration").status_code == 401

    def test_a_failed_attempt_is_not_counted_as_calibrating(self, bench_client, bench_lake):
        """A valid row recording an outcome is sent by the lookup and skipped
        by the estimate's registry, so it must not read as a row that
        calibrates: counted apart, as `rows_not_measured`."""
        from backend.inference import benchmarks as _benchmarks
        failed = dict(_row(1), outcome="error", outcome_detail="RuntimeError: boom",
                      cross_check="unknown",
                      **dict.fromkeys(_benchmarks.MEASUREMENT_COLUMNS, ""),
                      cross_check_rel_err="", cross_check_cosine="",
                      xcheck_typical="", xcheck_worst="")
        _upload(bench_client, ALICE, [_row(0), failed])
        report = bench_client.get("/api/bench/calibration", headers=as_user(ALICE)).json()
        assert (report["rows_used"], report["rows_not_measured"],
                report["rows_rejected"]) == (1, 1, 0)


def test_rules_rev_follows_the_declared_rule_boundary(monkeypatch):
    """Connection and lookup execution code is infrastructure; the validator
    and physical identity are rules.  The revision must say which is which."""
    from pathlib import Path
    from importlib import import_module
    from backend.service.bench import lake as bench_lake
    from backend.service.bench import lookup as bench_lookup
    from backend.service.bench import rules_revision
    from backend.inference import benchmarks
    from backend.inference.op_calibration import physical_identity

    owners = {entry.owner.__module__ if callable(entry.owner)
              else entry.owner.__name__ for entry in rules_revision.rule_inputs()}
    assert "backend.inference.benchmarks" in owners
    assert "backend.inference.op_calibration" in owners
    assert "backend.service.bench.data_rules" in owners
    assert not {"backend.service.bench.lake", "backend.service.bench.lookup",
                "backend.service.bench.serving"} & owners

    def with_probe(path: Path, marker: str) -> str:
        rules_revision.rules_rev.cache_clear()
        original = path.read_bytes()
        path.write_bytes(original + f"\n# {marker}\n".encode())
        try:
            rules_revision.rules_rev.cache_clear()
            return rules_revision.rules_rev()
        finally:
            path.write_bytes(original)
            rules_revision.rules_rev.cache_clear()

    assert with_probe(Path(bench_lake.__file__), "connection-budget probe") == (
        rules_revision.rules_rev())
    assert with_probe(Path(bench_lookup.__file__), "lookup execution probe") == (
        rules_revision.rules_rev())
    assert with_probe(Path(benchmarks.__file__), "validator probe") != (
        rules_revision.rules_rev())
    identity_module = import_module(physical_identity.__module__)
    assert with_probe(Path(identity_module.__file__),
                      "physical identity probe") != rules_revision.rules_rev()


def test_repeated_op_shapes_compute_their_physical_identity_once(monkeypatch):
    """The annotation owner memoizes the physical half by (op, shape_json).

    Inputs stream in that order, so the memo is one entry, not a second
    process-wide table. Phase and kernel impl still annotate every input.
    """
    calls = {"shape": 0, "identity": 0}

    def current_shape(op, shape_json):
        calls["shape"] += 1
        return {"op": op, "shape": shape_json}

    def physical_identity(op, shape):
        calls["identity"] += 1
        return {"signature": f"{op}:sig", "physical_shape": shape["shape"]}

    class Catalog:
        @staticmethod
        def catalog_kernel(kernel_impl):
            return kernel_impl.upper()

    monkeypatch.setattr(A, "current_shape", current_shape)
    monkeypatch.setattr(A, "physical_identity", physical_identity)
    monkeypatch.setattr(A, "_catalog", lambda: Catalog)
    A._shape_identity.cache_clear()
    try:
        shape = _row(0)["shape_json"]
        rows = [
            {"op": "linear", "shape_json": shape, "phase": phase,
             "kernel_impl": impl}
            for phase in ("decode", "prefill")
            for impl in ("a", "b")
        ]
        answers = [A.row_annotations(row) for row in rows]
    finally:
        A._shape_identity.cache_clear()
    assert calls == {"shape": 1, "identity": 1}
    assert {answer["catalog_kernel"] for answer in answers} == {"A", "B"}


@pytest.mark.parametrize("rows_each", [1, 40], ids=["inlined", "files"])
def test_divergent_copies_of_a_lake_have_divergent_scopes(tmp_path, annotate, make_rows,
                                                          new_owner, rows_each):
    """A lake copied (a restore, another machine, or beside itself as here)
    keeps its lake id and its history's snapshot ids; once the copies
    diverge, one snapshot id names two different contents — even with the
    same commit time, forced here. The scope names the content
    (`Lake.content_id`), so the two never share a cache key (the remote tier
    is shared by every machine), while their common history still does.
    Both a small write (rows inlined in the catalog) and a large one (data
    files) are told apart."""
    import shutil

    import duckdb

    from backend.service.bench.lake import Lake, LakeConfig
    from backend.service.bench.lookup import SharedRevision

    def opened(root):
        return Lake.open(LakeConfig(catalog=str(root / "catalog.ducklake"),
                                    data=str(root / "data"), cache_dir=str(root / "cache")))

    def scope(lake):
        (snap, content), (shared, shared_content) = SharedRevision().current(lake)
        return MeasurementScope(None, store.meta(lake)["lake_id"], snap, shared, "r",
                                content, shared_content)

    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir(mode=0o700)
    a = opened(a_dir)
    store.ensure_schema(a, lake_id="one-lineage", import_state="none")
    store.annotate_all(a, annotate)
    rows = make_rows(1 + 2 * rows_each)
    owner = new_owner()
    store.upload(a, owner=owner, upload_id=str(uuid.uuid4()), run_id=None, label=None,
                 rows=rows[:1], annotate=annotate, status="shared", decided_by=owner)
    common = scope(a)
    a.close()
    shutil.copytree(a_dir, b_dir)
    con = duckdb.connect(str(b_dir / "catalog.ducklake"))   # the copy's own data path
    con.execute("UPDATE ducklake_metadata SET value = ? WHERE key = 'data_path'",
                [str(b_dir / "data") + "/"])
    con.close()
    a, b = opened(a_dir), opened(b_dir)
    assert scope(a).key == scope(b).key == common.key        # one history
    for lake, mine in ((a, rows[1:1 + rows_each]), (b, rows[1 + rows_each:])):
        store.annotate_all(lake, annotate)                   # the copies diverge
        store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id=None,
                     label=None, rows=mine, annotate=annotate, status="shared",
                     decided_by=owner)
    first = a.snapshot_id()
    assert first == b.snapshot_id()
    a.close()
    b.close()
    for root in (a_dir, b_dir):                    # the same commit times in both
        con = duckdb.connect(str(root / "catalog.ducklake"))
        con.execute("UPDATE ducklake_snapshot SET snapshot_time = "
                    "TIMESTAMPTZ '2026-10-10 00:00:00+00' WHERE snapshot_id > ?",
                    [common.snapshot])
        con.close()
    a, b = opened(a_dir), opened(b_dir)
    meta = {r["k"]: r["v"] for r in a.read(
        "SELECT snapshot_id::VARCHAR AS k, snapshot_time::VARCHAR AS v FROM lake.snapshots()")}
    assert meta == {r["k"]: r["v"] for r in b.read(
        "SELECT snapshot_id::VARCHAR AS k, snapshot_time::VARCHAR AS v FROM lake.snapshots()")}
    changes = "SELECT changes_made AS c FROM __ducklake_metadata_lake.main.ducklake_snapshot_changes"
    kinds = " ".join(r["c"] for r in a.read(changes + f" WHERE snapshot_id > {common.snapshot}"))
    assert ("inserted_into_table" in kinds) == (rows_each > 1), kinds
    sa, sb = scope(a), scope(b)
    assert (sa.lake_id, sa.snapshot, sa.shared_snapshot) == (
        sb.lake_id, sb.snapshot, sb.shared_snapshot) == ("one-lineage", first, first)
    assert sa.key != sb.key
    assert scope(a).key == sa.key                  # stable for one content
    assert a.content_id(common.snapshot) == b.content_id(common.snapshot) \
        == common.snapshot_content                 # the common history, read again
    a.close()
    b.close()


def test_the_service_serves_without_torch(tmp_path):
    """The service annotates every input with the kernel catalog's names and
    answers lookups, and never loads torch (it grew the process from ~73 to
    ~540 MiB for its whole life)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    code = f"""
import sys, json, uuid
sys.path.insert(0, {str(root)!r})
import tests.corpus_row as C
from fastapi.testclient import TestClient
from backend.service.bench import annotations as A, server
app = server.create_app(start_lake=False)
c = TestClient(app, client=("127.0.0.1", 1), headers={{"X-HA-Internal": "s"}})
admin = {{"X-HA-User": str(uuid.uuid4()), "X-HA-Admin": "1"}}
r = c.post("/api/bench/upload", headers=admin, json={{"csv_text": C.to_csv([C.row()]), "make_public": True}})
assert r.status_code == 200, r.text
row = C.row()
ann = A.row_annotations(row)
key = {{"signature": ann["signature"], "physical_shape": ann["physical_shape"],
       "dtype_env_json": row["dtype_env_json"], "gpu": row["gpu"], "phase": row["phase"]}}
v = c.get("/internal/bench/version").json()
r = c.post("/internal/bench/lookup", json={{"keys": [key]}})
assert r.status_code == 200 and len(r.json()["rows"][0]) == 1, r.text
assert ann["catalog_kernel"] and v["rules_rev"]
print("torch" in sys.modules)
"""
    env = {"HUGGINGARCH_BENCH_SECRET": "s", "BENCH_LAKE_CATALOG": str(tmp_path / "l" / "c.ducklake"),
           "BENCH_LAKE_DATA": str(tmp_path / "l" / "data"), "BENCH_LAKE_LEASE": "off",
           "BENCH_LAKE_BACKUP": "off", "HUGGINGARCH_BENCH_MAINTENANCE": "off",
           "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "False", out.stdout


def test_a_lookup_over_its_row_bound_is_refused_before_reading_rows(
        bench_client, bench_lake, monkeypatch):
    """Past `MAX_ROWS` candidates the lookup is a 503 — before any wide
    column is read or any row assembled, never a truncated answer — and
    `/health` counts it beside the rows-per-key telemetry."""
    from backend.service.bench import lookup as LK
    a, b = _row(0), _row(1)
    _upload(bench_client, ADMIN, [a, b], public=True)
    keys = [_key(a), _key(b)]
    assert [len(k) for k in _lookup(bench_client, keys)["rows"]] == [1, 1]
    monkeypatch.setattr(LK, "MAX_ROWS", 1)

    def no_reads(*a, **k):
        raise AssertionError("read the rows of a refused lookup")
    monkeypatch.setattr(LK, "_entries", no_reads)
    r = bench_client.post("/internal/bench/lookup", json={"keys": keys})
    assert r.status_code == 503 and "over the lookup's bound" in r.json()["detail"]
    stats = bench_client.get("/health").json()["lookups"]
    assert (stats["lookups"], stats["refused"], stats["rows"], stats["max_rows_per_key"]) == (
        2, 1, 4, 1)
    assert stats["rows_per_key"] == 1.0
