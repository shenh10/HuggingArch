"""/api/bench/* of the Kernel Bench service over a local lake (a .ducklake
catalog file and a data directory under tmp_path): who may read and change
which set — the store knows no users, the routes scope every call, from the
identity the HA API forwards (`X-HA-User` / `X-HA-Admin`; the API's own side
is `test_bench_proxy`) — the bounds, the 503 without a lake, and the review
path from contribute to the shared corpus.

The submitter profiles are Supabase calls, intercepted by httpx_mock
(conftest); nothing else leaves the process."""
from __future__ import annotations

import json

import pytest

import tests.corpus_row as _corpus

from .conftest import TEST_SUPABASE_URL, as_user

ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ADMIN = "33333333-3333-4333-8333-333333333333"


def _auth(sub: str) -> dict:
    """What the HA API forwards for `sub` (ADMIN is the admin)."""
    return as_user(sub, admin=sub == ADMIN)


@pytest.fixture
def client(bench_client):
    """The service, called as the HA API calls it."""
    return bench_client


def _csv(n: int = 1, **over) -> str:
    """`n` valid rows of distinct identities (row i at s_ctx 2048·(i+1)); an
    override is a string, or a callable of the row index."""
    rows = []
    for i in range(n):
        shape = json.loads(_corpus.row()["shape_json"])
        shape["s_ctx"] = 2048 * (i + 1)
        fields = {"shape_json": json.dumps(shape, sort_keys=True, separators=(",", ":"))}
        fields.update({k: v(i) if callable(v) else v for k, v in over.items()})
        rows.append(_corpus.row(**fields))
    return _corpus.to_csv(rows)


def _upload(client, sub: str, n: int = 1, *, label: str = "batch", public: bool = False,
            **over) -> dict:
    r = client.post("/api/bench/upload", headers=_auth(sub),
                    json={"csv_text": _csv(n, **over), "label": label, "make_public": public})
    assert r.status_code == 200, r.text
    return r.json()


def _overstamp_first(client, sub: str, run_id: str) -> None:
    """Install a legacy over-peak row directly, as a lake written before the
    upload gate holds."""
    from backend.service.bench import routes as R
    from backend.service.bench import store
    rows = client.get(f"/api/bench/rows?run_id={run_id}", headers=_auth(sub)).json()["rows"]
    with R._writing() as lake:
        store.restamp(lake, owner=sub, rows=[{
            "id": rows[0]["id"], "achieved_tflops": 2000.0,
            "achieved_bw_gbps": 0.0}])


def _shared_rows(client) -> int:
    """How many rows the shared corpus holds (the signed-out rows view)."""
    return len(client.get("/api/bench/browse?view=rows&source=shared&page_size=200")
               .json()["items"])


class TestWithoutALake:
    def test_every_endpoint_names_the_missing_setting(self, client, monkeypatch, bench_gate):
        monkeypatch.delenv("BENCH_LAKE_CATALOG", raising=False)
        monkeypatch.delenv("BENCH_LAKE_DATA", raising=False)
        for method, path in (("GET", "/api/bench/list"),
                             ("GET", "/api/bench/summary?run_id=r"),
                             ("DELETE", "/api/bench/run/r")):
            r = client.request(method, path, headers=_auth(ALICE))
            assert r.status_code == 503, r.text
            assert "BENCH_LAKE_CATALOG" in r.json()["detail"]
        r = client.post("/api/bench/upload", headers=_auth(ALICE), json={"csv_text": _csv()})
        assert r.status_code == 503 and "BENCH_LAKE_DATA" in r.json()["detail"]


class TestGating:
    @pytest.mark.parametrize("method, path", [
        ("POST", "/api/bench/upload"), ("GET", "/api/bench/list"),
        ("GET", "/api/bench/summary?run_id=r"), ("GET", "/api/bench/rows"),
        ("DELETE", "/api/bench/run/r"), ("DELETE", f"/api/bench/{ALICE}"),
        ("POST", "/api/bench/run/r/contribute"), ("POST", f"/api/bench/{ALICE}/contribute"),
        ("GET", "/api/bench/nominations"), ("POST", "/api/bench/promote"),
        ("POST", "/api/bench/reject"), ("POST", "/api/bench/rows/delete"),
    ])
    def test_anonymous_is_401(self, client, method, path):
        assert client.request(method, path, json={}).status_code == 401

    def test_the_shared_dump_is_retired(self, client):
        assert client.get("/api/bench/shared").status_code in (404, 405)

    @pytest.mark.parametrize("method, path", [
        ("GET", "/api/bench/nominations"), ("POST", "/api/bench/promote"),
        ("POST", "/api/bench/reject"),
    ])
    def test_review_is_admin_only(self, client, bench_lake, httpx_mock, method, path):
        r = client.request(method, path, headers=_auth(ALICE),
                           json={"sets": [{"owner": ALICE, "kind": "run", "id": "r"}]})
        assert r.status_code == 403, r.text


class TestUpload:
    def test_upload_lists_a_private_batch_and_skips_duplicates(self, client, bench_lake):
        up = _upload(client, ALICE, 3, label="sweep", run_id="run-a")
        assert (up["row_count"], up["duplicates"], up["anomalous_count"], up["shared"]) == (3, 0, 0, False)
        again = _upload(client, ALICE, 3, label="again", run_id="run-a")
        assert (again["row_count"], again["duplicates"]) == (0, 3)
        (run,) = client.get("/api/bench/list", headers=_auth(ALICE)).json()["uploads"]
        assert (run["kind"], run["id"], run["upload_id"], run["label"]) == (
            "run", "run-a", up["upload_id"], "sweep")
        assert (run["row_count"], run["status"]) == (3, "private")
        assert [b["status"] for b in run["batches"]] == ["private"]
        assert client.get("/api/bench/list", headers=_auth(BOB)).json() == {"uploads": []}
        assert _shared_rows(client) == 0

    def test_invalid_csv_is_422(self, client, bench_lake):
        bad = _corpus.row()
        bad["latency_us"] = "not-a-number"
        r = client.post("/api/bench/upload", headers=_auth(ALICE),
                        json={"csv_text": _corpus.to_csv([bad])})
        assert r.status_code == 422, r.text

    def test_a_file_of_two_runs_is_422(self, client, bench_lake):
        r = client.post("/api/bench/upload", headers=_auth(ALICE),
                        json={"csv_text": _csv(2, run_id=lambda i: f"run-{i}")})
        assert r.status_code == 422 and "run-0, run-1" in r.json()["detail"]

    def test_a_row_above_the_cards_peak_is_422(self, client, bench_lake):
        r = client.post("/api/bench/upload", headers=_auth(ALICE), json={
            "csv_text": _csv(2, achieved_tflops=lambda i: "2000.0" if i == 0 else "500.0"),
            "label": "above peak"})
        assert r.status_code == 422, r.text
        assert "peak" in r.json()["detail"]
        assert client.get("/api/bench/list", headers=_auth(ALICE)).json()["uploads"] == []

    def test_non_admin_publishing_is_refused(self, client, bench_lake, httpx_mock):
        r = client.post("/api/bench/upload", headers=_auth(ALICE),
                        json={"csv_text": _csv(), "make_public": True})
        assert r.status_code == 403, r.text
        assert client.get("/api/bench/list", headers=_auth(ALICE)).json() == {"uploads": []}

    def test_admin_publishes_straight_to_shared(self, client, bench_lake, httpx_mock):
        up = _upload(client, ADMIN, 2, public=True)
        assert up["shared"] is True
        assert _shared_rows(client) == 2
        (s,) = client.get("/api/bench/list", headers=_auth(ADMIN)).json()["uploads"]
        assert (s["status"], s["is_public"]) == ("shared", True)


class TestReading:
    def test_summary_rows_and_overlap_of_ones_own_set(self, client, bench_lake):
        up = _upload(client, ALICE, 3, run_id="run-a")
        _overstamp_first(client, ALICE, "run-a")
        h = _auth(ALICE)
        s = client.get("/api/bench/summary?run_id=run-a", headers=h).json()
        assert (s["row_count"], s["mfu_over"], s["anomalous"], s["checked"]) == (3, 1, 1, 3)
        assert client.get(f"/api/bench/summary?upload_id={up['upload_id']}", headers=h).json() == s
        page = client.get("/api/bench/rows?run_id=run-a&limit=2", headers=h).json()
        assert (page["total"], len(page["rows"])) == (3, 2)
        row = page["rows"][0]
        assert {"mfu", "mbu", "roof_status", "node_label", "arith_intensity"} <= row.keys()
        assert sorted(r["roof_status"] for r in client.get(
            "/api/bench/rows?run_id=run-a", headers=h).json()["rows"]) == ["mfu", "ok", "ok"]
        assert client.get("/api/bench/rows?run_id=run-a&limit=2&offset=2",
                          headers=h).json()["total"] is None
        ov = client.get("/api/bench/overlap?run_id=run-a", headers=h).json()
        assert (ov["in_shared"], ov["new"]) == (0, ov["op_shapes"])

    def test_keyset_paging_and_its_bounds(self, client, bench_lake):
        _upload(client, ALICE, 3, run_id="run-a")
        h = _auth(ALICE)
        first = client.get("/api/bench/rows?run_id=run-a&limit=2&since=00000000-0000-0000-0000-000000000000",
                           headers=h).json()
        assert len(first["rows"]) == 2 and first["next"] == first["rows"][-1]["id"]
        rest = client.get(f"/api/bench/rows?run_id=run-a&limit=2&after={first['next']}",
                          headers=h).json()
        assert len(rest["rows"]) == 1 and rest["next"] is None
        assert client.get("/api/bench/rows?run_id=run-a&after=1),x",
                          headers=h).status_code == 400
        assert client.get(f"/api/bench/rows?after={first['next']}", headers=h).status_code == 400
        assert client.get("/api/bench/rows?run_id=run-a&limit=0", headers=h).status_code == 400

    def test_unscoped_rows_are_retired(self, client, bench_lake):
        """Every row of the caller's was the browser's calibration overlay;
        the server calibrates with them itself now (`calibrate_with_mine`)."""
        _upload(client, ALICE, 2)
        r = client.get("/api/bench/rows", headers=_auth(ALICE))
        assert r.status_code == 400
        assert "run_id or upload_id required" in r.json()["detail"]

    def test_another_owners_set_is_403_and_an_admin_reads_it(self, client, bench_lake, httpx_mock):
        _upload(client, ALICE, 2, run_id="run-a")
        for path in ("summary", "rows", "overlap"):
            r = client.get(f"/api/bench/{path}?run_id=run-a&owner={ALICE}", headers=_auth(BOB))
            assert r.status_code == 403, (path, r.text)
            r = client.get(f"/api/bench/{path}?run_id=run-a&owner={ALICE}", headers=_auth(ADMIN))
            assert r.status_code == 200, (path, r.text)
        assert client.get(f"/api/bench/summary?run_id=run-a&owner={ALICE}",
                          headers=_auth(ADMIN)).json()["row_count"] == 2
        # Without `owner` a reader reads their own run of that name: none.
        assert client.get("/api/bench/summary?run_id=run-a",
                          headers=_auth(BOB)).json()["row_count"] == 0

    def test_batch_review_sample_rows_come_from_the_rows_view(self, client, bench_lake, httpx_mock):
        """The batch review's sample (BenchReview `sampleRowsUrl`): one page of
        the set's rows from the Kernel Bench rows view, each with its physical
        op and MFU / MBU; an admin names the owner."""
        _upload(client, ALICE, 2, run_id="run-a")
        q = f"view=rows&source=mine&run=run-a&page=0&page_size=200&owner={ALICE}"
        for who in (ALICE, ADMIN):
            items = client.get(f"/api/bench/browse?{q}", headers=_auth(who)).json()["items"]
            assert len(items) == 2
            assert all(r["origin"] == "mine" and {"signature", "op", "latency_us", "mfu", "mbu"} <= r.keys()
                       for r in items)

    def test_a_set_needs_a_valid_name(self, client, bench_lake):
        h = _auth(ALICE)
        assert client.get("/api/bench/summary", headers=h).status_code == 400
        assert client.get("/api/bench/summary?upload_id=u1)", headers=h).status_code == 400


class TestDeleting:
    def test_deletes_are_the_owners(self, client, bench_lake):
        up = _upload(client, ALICE, 1, run_id="")
        _upload(client, ALICE, 2, run_id="run-a")
        # Bob names Alice's upload and run: his own, which do not exist.
        assert client.delete(f"/api/bench/{up['upload_id']}", headers=_auth(BOB)).json()["deleted"] is False
        assert client.delete("/api/bench/run/run-a", headers=_auth(BOB)).json()["deleted"] is False
        assert len(client.get("/api/bench/list", headers=_auth(ALICE)).json()["uploads"]) == 2
        assert client.delete(f"/api/bench/{up['upload_id']}", headers=_auth(ALICE)).json()["deleted"] is True
        assert client.delete("/api/bench/run/run-a", headers=_auth(ALICE)).json()["deleted"] is True
        assert client.get("/api/bench/list", headers=_auth(ALICE)).json() == {"uploads": []}
        assert client.delete("/api/bench/not-an-id", headers=_auth(ALICE)).status_code == 400

    def test_deleting_rows(self, client, bench_lake):
        _upload(client, ALICE, 2, run_id="run-a")
        ids = [r["id"] for r in client.get("/api/bench/rows?run_id=run-a",
                                           headers=_auth(ALICE)).json()["rows"]]
        r = client.post("/api/bench/rows/delete", headers=_auth(BOB), json={"ids": ids})
        assert r.json() == {"deleted": 0}
        r = client.post("/api/bench/rows/delete", headers=_auth(ALICE), json={"ids": ids[:1]})
        assert r.json() == {"deleted": 1}
        assert client.get("/api/bench/summary?run_id=run-a",
                          headers=_auth(ALICE)).json()["row_count"] == 1
        bad = client.post("/api/bench/rows/delete", headers=_auth(ALICE), json={"ids": ["x"]})
        assert bad.status_code == 400
        assert client.post("/api/bench/rows/delete", headers=_auth(ALICE),
                           json={"ids": []}).status_code == 400

    def test_restamping_rows(self, client, bench_lake):
        """Any subset of the roofline columns; a row that sets none, or a
        rate out of range, is refused before anything is written."""
        _upload(client, ALICE, 1, run_id="run-a")
        [row] = client.get("/api/bench/rows?run_id=run-a", headers=_auth(ALICE)).json()["rows"]

        def restamp(*rows):
            return client.post("/api/bench/rows/restamp", headers=_auth(ALICE),
                               json={"rows": list(rows)})

        assert restamp({"id": row["id"]}).status_code == 400
        assert restamp({"id": row["id"], "achieved_bw_gbps": -1.0}).status_code == 400
        assert restamp({"id": row["id"], "peak_bw_at_test": 0.0}).status_code == 400
        assert restamp({"id": row["id"], "achieved_tflops": 0.0,
                        "achieved_bw_gbps": 7.0}).json() == {"updated": 1}
        assert restamp({"id": row["id"], "achieved_tflops": 2000.0}).status_code == 422
        [after] = client.get("/api/bench/rows?run_id=run-a", headers=_auth(ALICE)).json()["rows"]
        assert (after["achieved_tflops"], after["achieved_bw_gbps"]) == (0.0, 7.0)
        assert after["p50_us"] == row["p50_us"]
        assert after["peak_bw_at_test"] == row["peak_bw_at_test"]


class TestReview:
    def _nominations(self, client, httpx_mock) -> list:
        httpx_mock.add_response(
            method="GET", url=f"{TEST_SUPABASE_URL}/rest/v1/profiles?id=in.%28{ALICE}%29&select=id%2Cusername",
            json=[{"id": ALICE, "username": "alice"}])
        httpx_mock.add_response(method="GET", url=f"{TEST_SUPABASE_URL}/auth/v1/admin/users/{ALICE}",
                                json={"id": ALICE, "email": "alice@example.com"})
        r = client.get("/api/bench/nominations", headers=_auth(ADMIN))
        assert r.status_code == 200, r.text
        return r.json()["nominations"]

    def test_contribute_promote_then_shared_in_browse(self, client, bench_lake, httpx_mock):
        up = _upload(client, ALICE, 3, run_id="run-a")
        _overstamp_first(client, ALICE, "run-a")
        r = client.post("/api/bench/run/run-a/contribute?within_peak=true", headers=_auth(ALICE))
        assert r.status_code == 200 and r.json()["batches"] == 1
        # Nothing private is left to nominate.
        assert client.post("/api/bench/run/run-a/contribute",
                           headers=_auth(ALICE)).status_code == 404
        (n,) = self._nominations(client, httpx_mock)
        assert (n["owner"], n["kind"], n["id"], n["run_id"]) == (ALICE, "upload", up["upload_id"], "run-a")
        assert n["row_count"] == 2            # what promote shares: the rows within peak
        assert n["submitter"] == {"username": "alice", "email": "alice@example.com"}
        signed_out = client.get("/api/bench/browse?view=envelope&source=shared").json()
        assert signed_out["items"] == [] and "pending" not in signed_out
        r = client.post("/api/bench/promote", headers=_auth(ADMIN),
                        json={"sets": [{"owner": n["owner"], "kind": n["kind"], "id": n["id"]}]})
        assert r.status_code == 200, r.text
        assert r.json()["promoted"] == [{"owner": ALICE, "kind": "upload", "id": up["upload_id"],
                                         "batches": 1}]
        assert _shared_rows(client) == 2
        shared = client.get("/api/bench/browse?view=envelope&source=shared").json()
        assert len(shared["items"]) == 2
        assert all(e["winner"]["origin"] == "shared" and e["winner"]["roof_status"] == "ok"
                   for e in shared["items"])
        (run,) = client.get("/api/bench/list", headers=_auth(ALICE)).json()["uploads"]
        assert (run["status"], run["is_public"]) == ("shared", True)
        # Promoting again finds nothing nominated.
        again = client.post("/api/bench/promote", headers=_auth(ADMIN),
                            json={"sets": [{"owner": ALICE, "kind": "run", "id": "run-a"}]})
        assert again.status_code == 404
        # Deleting a shared batch takes its rows out of the shared corpus.
        assert client.delete("/api/bench/run/run-a", headers=_auth(ALICE)).json()["deleted"]
        assert _shared_rows(client) == 0

    def test_reject_keeps_the_batch_its_owners(self, client, bench_lake, httpx_mock):
        up = _upload(client, ALICE, 1, run_id="")
        assert client.post(f"/api/bench/{up['upload_id']}/contribute",
                           headers=_auth(ALICE)).json()["batches"] == 1
        r = client.post("/api/bench/reject", headers=_auth(ADMIN),
                        json={"sets": [{"owner": ALICE, "kind": "upload", "id": up["upload_id"]}]})
        assert r.status_code == 200, r.text
        assert r.json()["rejected"][0]["batches"] == 1
        (s,) = client.get("/api/bench/list", headers=_auth(ALICE)).json()["uploads"]
        assert (s["status"], s["is_public"]) == ("rejected", False)
        assert client.get("/api/bench/summary?upload_id=" + up["upload_id"],
                          headers=_auth(ALICE)).json()["row_count"] == 1

    def test_review_refuses_malformed_set_names(self, client, bench_lake, httpx_mock):
        for ref in ({"owner": ALICE, "kind": "upload", "id": "u1)"},
                    {"owner": "not-a-uuid", "kind": "run", "id": "r"}):
            r = client.post("/api/bench/promote", headers=_auth(ADMIN), json={"sets": [ref]})
            assert r.status_code == 400, r.text
        assert client.post("/api/bench/reject", headers=_auth(ADMIN),
                           json={"sets": []}).status_code == 400

    def test_contribute_names_ones_own_set(self, client, bench_lake):
        up = _upload(client, ALICE, 1, run_id="run-a")
        assert client.post(f"/api/bench/{up['upload_id']}/contribute",
                           headers=_auth(BOB)).status_code == 404
        assert client.post("/api/bench/run/run-a/contribute",
                           headers=_auth(BOB)).status_code == 404


class TestBrowse:
    def test_mine_needs_sign_in(self, client, bench_lake):
        r = client.get("/api/bench/browse?source=mine")
        assert r.status_code == 401

    def test_refusals_are_400(self, client, bench_lake):
        # Unscoped rows are one page of the scope, like the envelope view;
        # any page depth is answered (an empty page past the end).
        assert client.get("/api/bench/browse?view=rows").status_code == 200
        far = client.get("/api/bench/browse?view=rows&page=200&page_size=50")
        assert far.status_code == 200 and far.json()["items"] == []
        r = client.get("/api/bench/browse?upload=nope", headers=_auth(ALICE))
        assert r.status_code == 400
        assert client.get("/api/bench/browse?key=%7B", headers=_auth(ALICE)).status_code == 400

    def test_a_set_of_ones_own_and_coverage(self, client, bench_lake):
        _upload(client, ALICE, 2, run_id="run-a")
        h = _auth(ALICE)
        rows = client.get("/api/bench/browse?view=rows&run=run-a", headers=h).json()
        assert len(rows["items"]) == 2
        assert all(r["origin"] == "mine" and "node_label" in r for r in rows["items"])
        assert client.get("/api/bench/browse?view=rows&run=run-a",
                          headers=_auth(BOB)).json()["items"] == []
        cov = client.get("/api/bench/browse?view=coverage&source=mine", headers=h).json()
        assert "libraries" in cov and "kernels" not in cov
        assert sum(sum(k.values()) for k in cov["libraries"].values()) \
            + sum(cov["unclassified"].values()) == 2



def test_startup_annotates_and_starts_the_maintainer(bench_lake, bench_gate, monkeypatch):
    """The background start (`serving.Gate.start`): the lake opened, every
    input annotated into the derived file by `row_annotations` (the composition only
    this module may make), then serving: the shared rows' revision starts
    at the snapshot then and the maintainer starts."""
    from backend.service.bench import lake as lake_mod
    from backend.service.bench import store
    from backend.service.bench import routes as B
    calls: list = []
    monkeypatch.setattr(B._maintain, "start", lambda *a: calls.append(a))
    lake = lake_mod.get_lake()
    store.ensure_schema(lake)
    store.upload(lake, owner=ALICE, upload_id="44444444-4444-4444-8444-444444444444", run_id=None,
                 label=None, rows=_corpus_rows(2), status="shared", decided_by=ADMIN)
    bench_gate.start()
    assert bench_gate.phase == "serving"
    got = lake.read(f"SELECT * FROM lake.inputs i LEFT JOIN {store.ANNOT} a USING (input_id)")
    assert len(got) == 2
    for i in got:
        assert {a: i[a] for a in store.ANNOTATIONS} == B.row_annotations(i)
    assert calls == [(lake, None)]
    from backend.service.bench.lookup import snapshot_now
    assert B.shared_revision.current(lake) == (snapshot_now(lake),) * 2


class TestNotServing:
    """Every lake endpoint answers 503 with the gate's reason; writes also
    when only writes are refused."""

    def test_while_importing_or_refused_every_endpoint_is_503(self, client, bench_lake, bench_gate):
        for phase, reason in (("importing", "the import from the Postgres tables is running"),
                              ("unavailable", "not the authoritative lake — restore from backup")):
            bench_gate.phase, bench_gate.reason = phase, reason
            for method, path in (("GET", "/api/bench/list"), ("GET", "/api/bench/browse"),
                                 ("DELETE", "/api/bench/run/r")):
                r = client.request(method, path, headers=_auth(ALICE))
                assert r.status_code == 503 and reason in r.json()["detail"], r.text
            r = client.post("/api/bench/upload", headers=_auth(ALICE), json={"csv_text": _csv()})
            assert r.status_code == 503 and reason in r.json()["detail"]

    def test_low_disk_refuses_writes_not_reads(self, client, bench_lake, monkeypatch):
        from collections import namedtuple

        from backend.service.bench import lake as lake_mod
        _upload(client, ALICE, 2)
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(lake_mod.shutil, "disk_usage",
                            lambda p: usage(10 * 2**30, 10 * 2**30 - 100 * 2**20, 100 * 2**20))
        r = client.post("/api/bench/upload", headers=_auth(ALICE), json={"csv_text": _csv(1, gpu="H200")})
        assert r.status_code == 503 and "100 MB free" in r.json()["detail"]
        assert client.delete("/api/bench/run/r", headers=_auth(ALICE)).status_code == 503
        assert client.get("/api/bench/list", headers=_auth(ALICE)).status_code == 200


def test_health_reports_the_lake_and_never_fails(client, bench_lake, bench_gate, monkeypatch):
    client.get("/api/bench/list", headers=_auth(ALICE))       # starts the gate
    detail = client.get("/health").json()["lake"]
    assert detail["state"] == "serving" and detail["lease"] == "off"
    assert detail["backup"] == "off" and detail["free_mb"] > 0

    def boom():
        raise RuntimeError("broken")
    monkeypatch.setattr(bench_gate, "health", boom)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["lake"] == {"error": "RuntimeError: broken"}


def _corpus_rows(n: int) -> list[dict]:
    from backend.inference import benchmarks as _benchmarks
    return _benchmarks.validate_text(_csv(n), source="test")[1]
