"""serving.Gate: the startup's decisions (docs/measured_calibration.md §5.2)
— a brand-new lake decided before anything is written, the lease
(PostgREST mocked by httpx_mock; lake and machine), a lake left mid-import,
the mandatory backup, the disk check — and what reads and writes get
meanwhile."""
from __future__ import annotations

import json
import re
import uuid
from collections import namedtuple

import httpx
import pytest

from backend.service.bench import serving, store
from backend.service.bench import lake as lake_mod
from backend.service.bench.backup import BackupConfig
from backend.service.bench.lake import Lake, LakeConfig, LakeUnavailable

from ..conftest import TEST_SERVICE_KEY, TEST_SUPABASE_URL
from .conftest import make_annotator

LEASE_URL = re.compile(re.escape(f"{TEST_SUPABASE_URL}/rest/v1/bench_lake_lease") + r"(\?.*)?$")
CLAIM_URL = f"{TEST_SUPABASE_URL}/rest/v1/rpc/bench_lake_claim"
OTHER = "99999999-9999-4999-8999-999999999999"


@pytest.fixture
def env(monkeypatch):
    """The lease on, machine m-1, backups off (the backup tests turn them on)."""
    for k in ("BENCH_LAKE_LEASE", "BENCH_LAKE_BACKUP_DATA",
              "BENCH_LAKE_MACHINE", "HUGGINGARCH_BENCH_RESTORE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FLY_MACHINE_ID", "m-1")
    monkeypatch.setenv("BENCH_LAKE_BACKUP", "off")
    return monkeypatch


@pytest.fixture
def lake_dir(tmp_path):
    return tmp_path / "lake"


@pytest.fixture
def opened(lake_dir):
    """The gate's `open_lake`: one Lake for the test, closed after."""
    lakes = []

    def open_lake():
        if not lakes:
            lakes.append(Lake.open(LakeConfig(catalog=str(lake_dir / "catalog.ducklake"),
                                              data=str(lake_dir / "data"))))
        return lakes[0]
    yield open_lake
    for lk in lakes:
        lk.close()


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _gate(opened, **kw):
    return serving.Gate(annotate=make_annotator(), open_lake=opened, **kw)


def _dev_schema(lake, **meta):
    """An unleased lake that is allowed to write: its own development role."""
    store.ensure_schema(lake, lake_role="development", **meta)


def _no_holder(httpx_mock):
    httpx_mock.add_response(method="GET", url=LEASE_URL, json=[])


def _claim(httpx_mock, answer):
    httpx_mock.add_response(method="POST", url=CLAIM_URL, json=answer)


def _claims(httpx_mock) -> list[dict]:
    return [json.loads(r.content) for r in httpx_mock.get_requests(method="POST", url=CLAIM_URL)]


# --- the lease -----------------------------------------------------------------------------

def test_a_new_lake_with_no_lease_holder_claims_it(env, opened, httpx_mock):
    _no_holder(httpx_mock)
    _claim(httpx_mock, "claimed")
    served = []
    gate = _gate(opened, on_serving=lambda lake, backup, read_only=False: served.append((lake, backup)))
    lake = gate.reader()
    meta = store.meta(lake)
    assert meta["import_state"] == "none"
    assert _claims(httpx_mock) == [{"p_lake": meta["lake_id"], "p_machine": "m-1"}]
    assert gate.lease_state == "claimed" and served == [(lake, None)]
    req = httpx_mock.get_requests(method="POST", url=CLAIM_URL)[0]
    assert req.headers["apikey"] == TEST_SERVICE_KEY
    assert meta["data_rules"] == serving.data_rules.registered_version()
    assert gate.health()["data_rules"]["state"] == "current"


def test_an_empty_lake_while_another_holds_the_lease_writes_nothing(env, opened, httpx_mock):
    httpx_mock.add_response(method="GET", url=LEASE_URL, json=[{"lake_id": OTHER}])
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="restore it from backup"):
        gate.reader()
    assert not store.has_tables(opened())
    assert gate.phase == "unavailable" and gate._retry_at is None      # not retried


def test_another_lake_holding_the_lease_is_refused(env, opened, httpx_mock):
    _no_holder(httpx_mock)
    _claim(httpx_mock, "refused")
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="not the authoritative lake"):
        gate.reader()
    with pytest.raises(LakeUnavailable):
        gate.writer()
    assert gate.lease_state == "refused"


def test_the_same_lake_on_the_same_machine_is_ok_and_a_lease_error_fails_closed(
        env, opened, httpx_mock):
    _no_holder(httpx_mock)
    _claim(httpx_mock, "claimed")
    _gate(opened).reader()
    _claim(httpx_mock, "ok")
    gate = _gate(opened)
    gate.reader()
    assert gate.lease_state == "ok"
    httpx_mock.add_exception(httpx.ConnectError("down"), method="POST", url=CLAIM_URL)
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="unreachable"):
        gate.reader()                                   # claimed before, still closed
    assert gate.lease_state == "unreachable" and gate._retry_at is not None


def test_a_cloned_volume_on_another_machine_is_refused(env, opened, httpx_mock):
    _no_holder(httpx_mock)
    _claim(httpx_mock, "claimed")
    lake_id = store.meta(_gate(opened).reader())["lake_id"]
    env.setenv("FLY_MACHINE_ID", "m-2")                 # the same lake files, another machine
    _claim(httpx_mock, "refused")
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="on machine m-2"):
        gate.reader()
    assert _claims(httpx_mock)[-1] == {"p_lake": lake_id, "p_machine": "m-2"}


def test_no_machine_id_is_refused(env, opened, httpx_mock):
    env.delenv("FLY_MACHINE_ID")
    _no_holder(httpx_mock)
    with pytest.raises(LakeUnavailable, match="no machine id"):
        _gate(opened).reader()
    env.setenv("BENCH_LAKE_MACHINE", "dev-box")
    _claim(httpx_mock, "claimed")
    _gate(opened).reader()
    assert _claims(httpx_mock)[-1]["p_machine"] == "dev-box"


def test_unreachable_and_never_claimed_is_503_then_retried(env, opened, httpx_mock):
    _no_holder(httpx_mock)
    httpx_mock.add_exception(httpx.ConnectError("down"), method="POST", url=CLAIM_URL)
    clock = FakeClock()
    gate = _gate(opened, clock=clock)
    with pytest.raises(LakeUnavailable, match="unreachable"):
        gate.reader()
    with pytest.raises(LakeUnavailable, match="unreachable"):
        gate.reader()                                   # not retried yet
    assert len(httpx_mock.get_requests(method="POST")) == 1
    clock.t += serving.RETRY_AFTER
    _claim(httpx_mock, "claimed")
    gate.reader()
    assert gate.phase == "serving"


def test_a_lease_answering_an_error_is_unreachable(env, opened, httpx_mock):
    httpx_mock.add_response(method="GET", url=LEASE_URL, status_code=404,
                            json={"message": "relation does not exist"})
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="404"):
        gate.reader()
    assert not store.has_tables(opened())


def test_the_lease_off_is_explicit(env, opened, httpx_mock, caplog):
    env.setenv("BENCH_LAKE_LEASE", "off")
    gate = _gate(opened)
    with caplog.at_level("WARNING"):
        gate.reader()
    assert gate.lease_state == "off" and "lease is OFF" in caplog.text
    assert httpx_mock.get_requests() == []
    assert gate.health()["lease"] == "off"


def test_an_unleased_copy_of_an_identified_lake_is_read_only(env, opened, make_rows, annotate):
    """`LEASE=off` is not a licence to fork an authoritative lake_id."""
    env.setenv("BENCH_LAKE_LEASE", "off")
    lake = opened()
    _dev_schema(lake, lake_id="11111111-1111-4111-8111-111111111111")
    store.set_meta(lake, lake_role="authority")
    serving.data_rules.initialize(lake)
    store.annotate_all(lake, annotate)
    store.upload(lake, owner="11111111-1111-4111-8111-111111111111",
                 upload_id="22222222-2222-4222-8222-222222222222", run_id=None,
                 label=None, rows=make_rows(1), annotate=annotate)
    gate = _gate(opened)
    assert gate.reader().read(
        "SELECT count(*) AS n FROM lake.measurements")[0]["n"] == 1
    with pytest.raises(LakeUnavailable, match=r"read-only.*lease holder"):
        gate.writer()
    health = gate.health()
    assert health["state"] == "serving" and health["read_only"] is True


def test_a_read_only_copy_behind_the_data_rules_does_not_serve_unvalidated_rows(
        env, opened):
    env.setenv("BENCH_LAKE_LEASE", "off")
    lake = opened()
    _dev_schema(lake, lake_id="11111111-1111-4111-8111-111111111111")
    store.set_meta(lake, lake_role="authority")
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="read-only copy's data rules"):
        gate.reader()
    health = gate.health()
    assert health["state"] == "degraded" and health["read_only"] is True
    assert health["data_rules"]["state"] == "failed"


def test_an_unleased_development_lake_has_its_own_writable_identity(env, opened):
    env.setenv("BENCH_LAKE_LEASE", "off")
    _dev_schema(opened(), lake_id="11111111-1111-4111-8111-111111111111")
    gate = _gate(opened)
    gate.writer()
    assert store.meta(gate.lake)["lake_role"] == "development"


def test_a_development_lake_cannot_become_the_authority_by_turning_the_lease_on(
        env, opened):
    _dev_schema(opened(), lake_id="11111111-1111-4111-8111-111111111111")
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="development lake cannot claim"):
        gate.reader()
    assert gate.phase == "unavailable"


def test_no_supabase_and_the_lease_on_is_503(env, opened, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL")
    with pytest.raises(LakeUnavailable, match="Supabase is not configured"):
        _gate(opened).reader()


# --- import_state ---------------------------------------------------------------------------

def test_a_lake_left_mid_import_is_refused(env, opened):
    """The import from the Postgres tables cannot resume: those tables are
    gone (migration 0026)."""
    env.setenv("BENCH_LAKE_LEASE", "off")
    store.ensure_schema(opened(), lake_id="x", import_state="importing")
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="restore the lake from backup"):
        gate.reader()
    assert gate._retry_at is None


@pytest.mark.parametrize("state", ["verified", "none"])
def test_an_imported_or_importless_lake_serves(env, opened, state):
    env.setenv("BENCH_LAKE_LEASE", "off")
    _dev_schema(opened(), lake_id="x", import_state=state)
    gate = _gate(opened)
    gate.reader()
    assert gate.phase == "serving"


def test_a_lake_made_before_lake_meta_gets_an_identity(env, opened):
    env.setenv("BENCH_LAKE_LEASE", "off")
    store.ensure_schema(opened())
    gate = _gate(opened)
    meta = store.meta(gate.reader())
    assert meta["import_state"] == "none" and len(meta["lake_id"]) == 36


def test_an_old_lake_migrates_its_data_rules_before_it_serves(
        env, opened, make_rows, annotate, new_owner):
    env.setenv("BENCH_LAKE_LEASE", "off")
    lake = opened()
    _dev_schema(lake, lake_id="x", import_state="none")
    store.annotate_all(lake, annotate)
    store.upload(lake, owner=new_owner(), upload_id=str(uuid.uuid4()), run_id=None,
                 label=None, rows=make_rows(1), annotate=annotate)
    assert "data_rules" not in store.meta(lake)
    gate = _gate(opened)
    gate.reader()
    assert store.meta(lake)["data_rules"] == serving.data_rules.registered_version()
    assert gate.health()["data_rules"] == {
        "state": "current", "registered": serving.data_rules.registered_version(),
        "lake": serving.data_rules.registered_version(),
        "migration": {"rekeyed_rows": 0, "env_rekeyed_rows": 0,
                      "dropped_shared_rows": 0, "reasons": {}},
    }


def test_a_lake_behind_is_degraded_while_it_migrates_and_failed_migrations_stay_degraded(
        env, opened, monkeypatch):
    env.setenv("BENCH_LAKE_LEASE", "off")
    _dev_schema(opened(), lake_id="x", import_state="none")
    gate = None
    real = serving.data_rules.migrate

    def migrating(lake):
        health = gate.health()
        assert health["state"] == "degraded"
        assert health["data_rules"]["state"] == "behind"
        assert "migrating" in health["reason"]
        return real(lake)

    monkeypatch.setattr(serving.data_rules, "migrate", migrating)
    gate = _gate(opened)
    gate.reader()
    assert gate.phase == "serving"

    def failed(_lake):
        raise RuntimeError("the plan is truncated")

    store.set_meta(opened(), data_rules="legacy")
    monkeypatch.setattr(serving.data_rules, "migrate", failed)
    failed_gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="the plan is truncated"):
        failed_gate.reader()
    health = failed_gate.health()
    assert health["state"] == "degraded"
    assert health["data_rules"]["state"] == "failed"
    assert health["data_rules"]["error"] == "RuntimeError: the plan is truncated"


def test_an_unregistered_lake_version_is_refused_before_any_schema_write(
        env, opened, monkeypatch):
    env.setenv("BENCH_LAKE_LEASE", "off")
    _dev_schema(opened(), lake_id="x", import_state="none")
    store.set_meta(opened(), data_rules="from-a-newer-deployment")

    def refused(*_args, **_kwargs):
        raise AssertionError("an unregistered lake must not reach ensure_schema")

    monkeypatch.setattr(store, "ensure_schema", refused)
    gate = _gate(opened)
    with pytest.raises(LakeUnavailable, match="outside this deployment"):
        gate.reader()
    health = gate.health()
    assert health["state"] == "degraded"
    assert health["data_rules"] == {
        "state": "failed", "registered": serving.data_rules.registered_version(),
        "lake": "from-a-newer-deployment",
        "error": "the lake's data rules are 'from-a-newer-deployment', outside "
                 "this deployment's registered chain",
    }


# --- annotations --------------------------------------------------------------------------

def test_every_input_is_annotated_in_memory_before_the_lake_serves(env, opened, make_rows):
    """The gate builds `annot` from every input with the `annotate` it was
    given before it serves; nothing of it is stored."""
    env.setenv("BENCH_LAKE_LEASE", "off")
    lake = opened()
    _dev_schema(lake, lake_id="x", import_state="none")
    store.annotate_all(lake, make_annotator())
    store.upload(lake, owner="11111111-1111-4111-8111-111111111111", upload_id=
                 "22222222-2222-4222-8222-222222222222", run_id=None, label=None,
                 rows=make_rows(3), annotate=make_annotator())
    lake.transact(lambda cur: cur.execute("DROP TABLE annot"))     # a new process
    other = make_annotator(prefix="sig2")
    seen = []

    def on_serving(lk, backup, read_only=False):
        seen.append(lk.read("SELECT * FROM lake.inputs i JOIN annot a USING (input_id)"))
    gate = serving.Gate(annotate=other, open_lake=opened, on_serving=on_serving)
    gate.reader()
    [got] = seen
    assert len(got) == 3
    for i in got:
        assert {a: i[a] for a in store.ANNOTATIONS} == other(i)


def test_a_catalog_that_cannot_load_holds_the_startup(env, opened, make_rows):
    env.setenv("BENCH_LAKE_LEASE", "off")
    lake = opened()
    _dev_schema(lake, lake_id="x", import_state="none")
    store.annotate_all(lake, make_annotator())
    store.upload(lake, owner="11111111-1111-4111-8111-111111111111", upload_id=
                 "22222222-2222-4222-8222-222222222222", run_id=None, label=None,
                 rows=make_rows(1), annotate=make_annotator())

    def broken(inp):
        raise ImportError("no kernel_bench")
    gate = serving.Gate(annotate=broken, open_lake=opened)
    with pytest.raises(LakeUnavailable, match="no kernel_bench"):
        gate.reader()


# --- backups and disk ------------------------------------------------------------------------

def test_startup_backs_up_before_writes(env, opened, tmp_path, make_rows, annotate):
    env.setenv("BENCH_LAKE_LEASE", "off")
    env.delenv("BENCH_LAKE_BACKUP")
    lake = opened()
    _dev_schema(lake, lake_id="x", import_state="none")
    store.annotate_all(lake, annotate)
    store.upload(lake, owner="11111111-1111-4111-8111-111111111111", upload_id=
                 "22222222-2222-4222-8222-222222222222", run_id=None, label=None,
                 rows=make_rows(2), annotate=annotate)
    config = BackupConfig(root=str(tmp_path / "bucket"))
    gate = _gate(opened, backup=lambda: config)
    gate.writer()
    current = json.loads((tmp_path / "bucket" / "current").read_text())
    assert current["snapshot"] == lake.snapshot_id()
    st = gate.health()["backup"]
    assert st["newest_recoverable_snapshot"] == lake.snapshot_id() and st["last_error"] is None


def test_unconfigured_backups_refuse_writes_not_reads(env, opened):
    env.setenv("BENCH_LAKE_LEASE", "off")
    env.delenv("BENCH_LAKE_BACKUP")
    gate = _gate(opened, backup=lambda: None)
    gate.reader()
    with pytest.raises(LakeUnavailable, match="backups are not configured"):
        gate.writer()
    assert gate.health()["backup"]["configured"] is False
    env.setenv("BENCH_LAKE_BACKUP", "off")              # the explicit opt-out
    gate = _gate(opened, backup=lambda: pytest.fail("not read when off"))
    gate.writer()
    assert gate.health()["backup"] == "off"


def test_a_failed_startup_backup_holds_writes(env, opened, tmp_path, monkeypatch):
    env.setenv("BENCH_LAKE_LEASE", "off")
    env.delenv("BENCH_LAKE_BACKUP")
    monkeypatch.setattr(serving.Trigger, "start", lambda self: None)   # no retry thread here
    config = BackupConfig(root=str(tmp_path / "bucket"))
    (tmp_path / "bucket").write_text("not a directory")
    gate = _gate(opened, backup=lambda: config)
    gate.reader()                                       # reads serve
    with pytest.raises(LakeUnavailable, match="the backup is failing"):
        gate.writer()
    (tmp_path / "bucket").unlink()
    gate.backup.run(opened())                           # what the trigger's retry does
    gate.writer()


def test_low_disk_refuses_writes(env, opened, monkeypatch):
    env.setenv("BENCH_LAKE_LEASE", "off")
    usage = namedtuple("usage", "total used free")
    free = [lake_mod.MIN_FREE_BYTES - 1]
    monkeypatch.setattr(lake_mod.shutil, "disk_usage", lambda p: usage(0, 0, free[0]))
    gate = _gate(opened)
    gate.reader()
    with pytest.raises(LakeUnavailable, match="writes refused"):
        gate.writer()
    assert gate.health()["free_mb"] == 499
    free[0] = lake_mod.MIN_FREE_BYTES
    gate.writer()


def test_an_unopenable_lake_is_503_with_the_reason(env, monkeypatch):
    def open_lake():
        raise LakeUnavailable("BENCH_LAKE_CATALOG and BENCH_LAKE_DATA must be set")
    gate = serving.Gate(annotate=make_annotator(), open_lake=open_lake)
    with pytest.raises(LakeUnavailable, match="BENCH_LAKE_CATALOG"):
        gate.reader()
    assert gate.health()["state"] == "unavailable"


def _backed_up_lake(tmp_path, make_rows, annotate) -> int:
    """A lake with two rows, backed up into tmp_path/bucket; its row count."""
    from backend.service.bench.backup import Backup, LocalBucket
    src = Lake.open(LakeConfig(catalog=str(tmp_path / "src" / "catalog.ducklake"),
                               data=str(tmp_path / "src" / "data")))
    store.ensure_schema(src, lake_id="built-elsewhere", import_state="verified")
    store.annotate_all(src, annotate)
    store.upload(src, owner="11111111-1111-4111-8111-111111111111",
                 upload_id="22222222-2222-4222-8222-222222222222", run_id=None, label=None,
                 rows=make_rows(2), annotate=annotate)
    Backup(LocalBucket(str(tmp_path / "bucket"))).run(src)
    n = src.read("SELECT count(*) AS n FROM lake.measurements")[0]["n"]
    src.close()
    return n


def test_restore_fills_an_empty_volume_once(env, tmp_path, make_rows, annotate):
    """The cutover's offline import (or a lost volume) comes back from the
    backup's `current` snapshot before anything opens the lake; with a lake
    present it does nothing."""
    n = _backed_up_lake(tmp_path, make_rows, annotate)
    vol = tmp_path / "vol" / "bench_lake"
    vol.mkdir(parents=True, mode=0o700)
    env.setenv("HUGGINGARCH_BENCH_RESTORE", "on")
    env.setenv("BENCH_LAKE_CATALOG", str(vol / "catalog.ducklake"))
    env.setenv("BENCH_LAKE_DATA", str(vol / "data") + "/")
    env.setenv("BENCH_LAKE_BACKUP_DATA", str(tmp_path / "bucket"))
    assert serving.restore_from_env() == str(vol / "catalog.ducklake")
    lake = Lake.open(LakeConfig.from_env())
    try:
        assert lake.read("SELECT count(*) AS n FROM lake.measurements")[0]["n"] == n
        assert store.meta(lake)["lake_id"] == "built-elsewhere"
    finally:
        lake.close()
    assert serving.restore_from_env() is None


def test_restore_is_off_unless_asked(env, tmp_path):
    env.setenv("BENCH_LAKE_CATALOG", str(tmp_path / "vol" / "catalog.ducklake"))
    env.setenv("BENCH_LAKE_DATA", str(tmp_path / "vol" / "data"))
    assert serving.restore_from_env() is None
    assert not (tmp_path / "vol").exists()


def test_restore_refuses_a_partly_filled_volume_or_a_backup_without_current(
        env, opened, tmp_path, make_rows, annotate):
    vol = tmp_path / "vol"
    vol.mkdir(mode=0o700)
    env.setenv("HUGGINGARCH_BENCH_RESTORE", "on")
    env.setenv("BENCH_LAKE_CATALOG", str(vol / "catalog.ducklake"))
    env.setenv("BENCH_LAKE_DATA", str(vol / "data"))
    env.setenv("BENCH_LAKE_BACKUP_DATA", str(tmp_path / "empty-bucket"))
    with pytest.raises(RuntimeError, match="current"):
        serving.restore_from_env()
    _backed_up_lake(tmp_path, make_rows, annotate)
    env.setenv("BENCH_LAKE_BACKUP_DATA", str(tmp_path / "bucket"))
    (vol / "stray").write_text("x")
    with pytest.raises(RuntimeError, match="empty"):
        serving.restore_from_env()
    # Through the gate, a failed restore leaves the store unserved with the reason.
    gate = _gate(opened, restore=serving.restore_from_env)
    with pytest.raises(LakeUnavailable, match="restore from backup failed"):
        gate.reader()


def test_the_gate_drops_stored_annotation_columns_of_an_old_lake(env, opened, make_rows):
    """A lake written while annotations were stored on `inputs` (docs/measured_calibration.md §4.2) is
    migrated by the gate at startup — not only a brand-new lake gets
    `ensure_schema` — so an upload, which inserts the input key alone,
    works on it."""
    env.setenv("BENCH_LAKE_LEASE", "off")
    lake = opened()
    _dev_schema(lake, lake_id="x", import_state="none")
    lake.transact(lambda cur: cur.execute(
        "ALTER TABLE lake.inputs ADD COLUMN signature VARCHAR"))
    gate = serving.Gate(annotate=make_annotator(), open_lake=opened)
    lk = gate.reader()
    assert [r["column_name"] for r in lk.read(
        "SELECT column_name FROM information_schema.columns WHERE table_catalog = 'lake' "
        "AND table_name = 'inputs' ORDER BY ordinal_position")] == ["input_id", *store.INPUT_KEY]
    res = store.upload(lk, owner="11111111-1111-4111-8111-111111111111",
                       upload_id="22222222-2222-4222-8222-222222222222", run_id=None,
                       label=None, rows=make_rows(1), annotate=make_annotator())
    assert res.inserted == 1
