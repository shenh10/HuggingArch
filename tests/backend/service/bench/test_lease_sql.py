"""The lease (`public.bench_lake_lease`, `bench_lake_claim`,
`bench_lake_transfer`, in the baseline `supabase/init.sql`) on the local
Postgres 17 (socket /tmp, port 55432), the database built from init.sql
alone by /tmp/pg_fresh_db.sh:
one row, the claim's three answers (lake and machine must both match), two
first claims racing, a cloned volume on another machine refused, the
operator's transfer, service role only. Skipped when the cluster or psql is
not there."""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

from tests.backend.service.bench.pg import SOCKET, scratch_database

_ROOT = Path(__file__).resolve().parents[4]
_INIT = _ROOT / "supabase/init.sql"
_TRANSFER_LAKE = (_ROOT / "supabase" / "migrations"
                  / "0027_bench_lake_transfer_lineage.sql")
_PSQL = "/tmp/pgenv/bin/psql"
_FRESH = "/tmp/pg_fresh_db.sh"
_DB = f"bench_lease_test_{os.getpid()}"
A, B = str(uuid.uuid4()), str(uuid.uuid4())


@pytest.fixture
def con():
    if not (os.path.exists(_PSQL) and os.path.exists(_FRESH)):
        pytest.skip(f"{_PSQL} or {_FRESH} not present")
    with scratch_database(_DB), tempfile.TemporaryDirectory() as src:
        # init.sql alone: a source tree with the baseline and no migrations.
        (Path(src) / "supabase" / "migrations").mkdir(parents=True)
        (Path(src) / "supabase" / "init.sql").write_text(_INIT.read_text())
        out = subprocess.run(["bash", _FRESH, _DB, src], capture_output=True, text=True,
                              timeout=300, check=False)
        assert out.returncode == 0, out.stderr
        out = subprocess.run(
            [_PSQL, "-h", SOCKET, "-d", _DB, "-v", "ON_ERROR_STOP=1",
             "-f", str(_TRANSFER_LAKE)], capture_output=True, text=True,
            timeout=300, check=False)
        assert out.returncode == 0, out.stderr
        conns = []

        def connect():
            import pg8000.native
            c = pg8000.native.Connection("postgres", unix_sock=SOCKET, database=_DB)
            conns.append(c)
            return c
        try:
            yield connect
        finally:
            for c in conns:
                c.close()


def _claim(c, lake, machine="m"):
    return c.run("SELECT public.bench_lake_claim(CAST(:l AS uuid), :m)", l=lake, m=machine)[0][0]


def _lease(c):
    return c.run("SELECT lake_id::text, machine, claimed_at FROM public.bench_lake_lease")


def test_claim_answers(con):
    c = con()
    assert _claim(c, A, "m-1") == "claimed"
    first = _lease(c)
    assert [r[:2] for r in first] == [[A, "m-1"]]
    time.sleep(0.01)
    assert _claim(c, A, "m-1") == "ok"
    [(lake, machine, at)] = _lease(c)
    assert (lake, machine) == (A, "m-1") and at > first[0][2]
    assert _claim(c, B, "m-1") == "refused"           # another lake
    assert _claim(c, A, "m-2") == "refused"           # a cloned volume on another machine
    assert _claim(c, B, "m-2") == "refused"
    assert [r[:2] for r in _lease(c)] == [[A, "m-1"]]
    for bad in ("NULL, 'm'", f"CAST('{A}' AS uuid), ''", f"CAST('{A}' AS uuid), NULL"):
        with pytest.raises(Exception, match="are required"):
            c.run(f"SELECT public.bench_lake_claim({bad})")


def test_transfer_moves_the_machine_only_from_the_holder(con):
    c = con()
    assert c.run("SELECT public.bench_lake_transfer('m-1', 'm-2')") == [["refused"]]   # no lease
    _claim(c, A, "m-1")
    assert c.run("SELECT public.bench_lake_transfer('m-9', 'm-2')") == [["refused"]]
    assert c.run("SELECT public.bench_lake_transfer('m-1', 'm-2')") == [["moved"]]
    assert _claim(c, A, "m-1") == "refused" and _claim(c, A, "m-2") == "ok"
    assert [r[:2] for r in _lease(c)] == [[A, "m-2"]]


def test_transfer_can_move_a_new_lake_lineage_only_from_the_holder(con):
    c = con()
    _claim(c, A, "m-1")
    assert c.run(
        "SELECT public.bench_lake_transfer('m-9', 'm-2', CAST(:a AS uuid), "
        "CAST(:b AS uuid))", a=A, b=B) == [["refused"]]
    assert c.run(
        "SELECT public.bench_lake_transfer('m-1', 'm-2', CAST(:a AS uuid), "
        "CAST(:b AS uuid))", a=A, b=B) == [["moved"]]
    assert [r[:2] for r in _lease(c)] == [[B, "m-2"]]


def test_one_row_only(con):
    c = con()
    _claim(c, A)
    with pytest.raises(Exception, match=r"check|duplicate"):
        c.run("INSERT INTO public.bench_lake_lease (id, lake_id, machine) VALUES (false, CAST(:l AS uuid), 'x')", l=B)
    with pytest.raises(Exception, match="duplicate"):
        c.run("INSERT INTO public.bench_lake_lease (lake_id, machine) VALUES (CAST(:l AS uuid), 'x')", l=B)


def test_two_first_claims_racing_one_wins(con):
    a, b = con(), con()
    a.run("BEGIN")
    assert _claim(a, A, "m-1") == "claimed"     # not yet committed
    answer = []
    t = threading.Thread(target=lambda: answer.append(_claim(b, A, "m-2")))
    t.start()
    time.sleep(0.3)
    assert answer == []                          # waits on the first claim's row
    a.run("COMMIT")
    t.join(5)
    assert answer == ["refused"]


def test_service_role_only(con):
    c = con()
    privs = dict(c.run(
        "SELECT r, has_function_privilege(r, 'public.bench_lake_claim(uuid, text)', 'EXECUTE')"
        " FROM unnest(ARRAY['anon', 'authenticated', 'service_role']) r"))
    assert privs == {"anon": False, "authenticated": False, "service_role": True}
    table = dict(c.run(
        "SELECT r, has_table_privilege(r, 'public.bench_lake_lease', 'SELECT')"
        " FROM unnest(ARRAY['anon', 'authenticated', 'service_role']) r"))
    assert table == {"anon": False, "authenticated": False, "service_role": True}
    assert c.run("SELECT relrowsecurity FROM pg_class WHERE oid = 'public.bench_lake_lease'::regclass") == [[True]]
    transfer = dict(c.run(
        "SELECT r, has_function_privilege(r, 'public.bench_lake_transfer(text, text)', 'EXECUTE')"
        " FROM unnest(ARRAY['anon', 'authenticated', 'service_role']) r"))
    assert transfer == {"anon": False, "authenticated": False, "service_role": False}
    transfer_lake = dict(c.run(
        "SELECT r, has_function_privilege(r, 'public.bench_lake_transfer("
        "text, text, uuid, uuid)', 'EXECUTE')"
        " FROM unnest(ARRAY['anon', 'authenticated', 'service_role']) r"))
    assert transfer_lake == {"anon": False, "authenticated": False, "service_role": False}
    assert not c.run("SELECT has_table_privilege('service_role', 'public.bench_lake_lease', 'UPDATE')")[0][0]
