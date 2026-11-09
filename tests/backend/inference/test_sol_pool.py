"""The shared SOL pool's immutable, seekable record store."""
from __future__ import annotations

import json

from backend.measure.scope import NONE, ResultIdentity
from backend.inference.sol_pool import SolPool


def identity(name: str) -> ResultIdentity:
    return ResultIdentity(kind="sol-test", version=name, content={"name": name},
                          scope=NONE, reads=False)


def test_a_bundle_is_written_once_and_reads_the_same_records(tmp_path):
    pool = SolPool(tmp_path)
    who = identity("walk")
    records = [(f"record-{i}", {"value": i, "padding": "x" * 10_000})
               for i in range(20)]

    assert pool.write(who, records, metadata={"surface": "test"})
    # An existing immutable bundle wins; the second writer is a no-op, not an
    # overwrite.
    assert pool.write(who, [(f"other-{i}", i) for i in range(3)])
    assert pool.recordIds(who.key) == tuple(name for name, _ in records)
    assert list(pool.records(who.key)) == records
    assert pool.record(who.key, "record-7") == records[7][1]
    view = pool.sequence(who.key)
    assert view is not None
    assert len(view) == len(records)
    assert view[3] == records[3][1]
    assert view[8:10] == [value for _name, value in records[8:10]]


def test_a_reader_seeks_one_record_without_parsing_its_neighbours(tmp_path):
    pool = SolPool(tmp_path)
    who = identity("seek")
    records = [(f"record-{i}", {"index": i, "body": "y" * 20_000})
               for i in range(12)]
    assert pool.write(who, records)

    wanted = pool.record(who.key, "record-11")
    assert wanted["index"] == 11
    shards = list((tmp_path / who.key[:2] / who.key).glob("records-*.jsonl"))
    assert shards
    # Every stored record is independently addressable; the reader does not
    # materialise or parse the bundle as one JSON document.
    lines = []
    index = json.loads((tmp_path / who.key[:2] / who.key / "index.json")
                       .read_text(encoding="utf-8"))
    for name in index["order"]:
        slot = index["records"][name]
        shard = next(shard for shard in shards
                     if shard.name == f"records-{slot[0]:05d}.jsonl")
        encoded = shard.read_bytes()[slot[1]:slot[1] + slot[2]]
        assert len(encoded) == slot[2]
        lines.append(encoded)
    assert len(lines) == len(records)
    assert json.loads(lines[11]) == wanted


def test_a_corrupt_record_is_a_miss_and_the_manifest_survives(tmp_path):
    pool = SolPool(tmp_path, hotRecords=0)
    who = identity("corrupt")
    assert pool.write(who, [("good", {"value": 1})])
    shard = next((tmp_path / who.key[:2] / who.key).glob("records-*.jsonl"))
    encoded = shard.read_bytes()
    shard.write_bytes(encoded.replace(b'{"value":1}', b'{"value":2}'))

    assert pool.record(who.key, "good") is None
    assert pool.manifest(who.key)["record_count"] == 1


def test_the_hot_cache_budget_counts_json_bytes(tmp_path):
    pool = SolPool(tmp_path, hotRecords=10, hotBytes=12_000)
    who = identity("hot")
    records = [(f"record-{i}", {"padding": "x" * 10_000}) for i in range(2)]
    assert pool.write(who, records)

    assert pool.record(who.key, "record-0") is not None
    assert pool.record(who.key, "record-1") is not None
    assert list(pool._hot) == [(who.key, "record-1")]


def test_a_pinned_record_survives_the_hot_tiers_churn(tmp_path):
    """The Gallery's serving walks re-read their records on every cold
    build; pinned, a record stays parsed whatever the hot LRU does."""
    pool = SolPool(tmp_path, hotRecords=1, hotBytes=1, pinBytes=1 << 20)
    who = identity("pinned")
    records = [(f"record-{i}", {"padding": "x" * 10_000}) for i in range(3)]
    assert pool.write(who, records)

    assert pool.pinned_record(who.key, "record-0") is not None
    assert pool.record(who.key, "record-1") is not None   # churns the hot LRU
    assert pool.record(who.key, "record-2") is not None
    assert list(pool._pinned) == [(who.key, "record-0")]
    assert pool.record(who.key, "record-2") is not None   # still not pinned
    assert list(pool._pinned) == [(who.key, "record-0")]


def test_the_pin_cap_refuses_new_pins_and_never_drops_a_pinned_record(tmp_path):
    """A pinned record was promised residency: the cap is an admission
    rule, not an eviction rule. Past it further pins are refused — counted,
    and never quietly kept in the LRU instead: a refused pin re-reads from
    disk, so the layers stay separate answers rather than a fallback."""
    pool = SolPool(tmp_path, hotRecords=0, hotBytes=1, pinBytes=15_000)
    who = identity("budget")
    records = [(f"record-{i}", {"padding": "x" * 10_000}) for i in range(3)]
    assert pool.write(who, records)

    assert pool.pinned_record(who.key, "record-0") is not None
    assert list(pool._pinned) == [(who.key, "record-0")]
    assert pool.pinned_record(who.key, "record-1") is not None  # refused
    assert list(pool._pinned) == [(who.key, "record-0")]        # still
    assert list(pool._hot) == []                                # no fallback
    assert pool.pinRefused == 1
    assert pool.pinned_record(who.key, "record-0") == records[0][1]
    assert pool.pinRefused == 1                                 # a hit


def test_a_record_larger_than_the_whole_pin_cap_is_refused(tmp_path):
    pool = SolPool(tmp_path, hotRecords=8, pinBytes=5_000)
    who = identity("toolarge")
    assert pool.write(who, [("record-0", {"padding": "x" * 10_000})])
    assert pool.pinned_record(who.key, "record-0") is not None
    assert list(pool._pinned) == [] and list(pool._hot) == []
    assert pool.pinRefused == 1


def test_a_pool_without_a_pin_budget_reads_pins_from_disk(tmp_path):
    pool = SolPool(tmp_path, pinBytes=0)
    who = identity("nopin")
    assert pool.write(who, [("record-0", {"v": 1})])
    assert pool.pinned_record(who.key, "record-0") == {"v": 1}
    assert list(pool._pinned) == [] and list(pool._hot) == []
    assert pool.record(who.key, "record-0") == {"v": 1}
    assert list(pool._hot) == [(who.key, "record-0")]


def test_a_pool_crossing_a_process_keeps_its_pin_budget(tmp_path):
    import pickle

    pool = SolPool(tmp_path, pinBytes=123_456)
    back = pickle.loads(pickle.dumps(pool))
    assert back is not pool and back._pinBytesTarget == 123_456


def test_a_view_crosses_a_process_boundary(tmp_path):
    """A worker hands a pool view back to the server process (the measured
    Scaling sweep's theory half): it is pickled as where the pool is, and
    the receiver seeks the same records through a lock of its own. It once
    failed with "cannot pickle '_thread.lock'" and the sweep answered 500."""
    import pickle
    from concurrent.futures import ProcessPoolExecutor

    pool = SolPool(tmp_path)
    who = identity("cross")
    records = [(f"record-{i}", {"value": i}) for i in range(5)]
    assert pool.write(who, records)
    view = pool.sequence(who.key)
    pool.record(who.key, "record-0")   # a hot entry is not part of the value
    back = pickle.loads(pickle.dumps(view))
    assert back.pool is not pool and back.pool.root == pool.root
    assert list(back) == [value for _name, value in records]
    with ProcessPoolExecutor(max_workers=1) as ex:
        returned = ex.submit(_view_in_worker, str(tmp_path), who.key).result()
    assert list(returned) == [value for _name, value in records]


def _view_in_worker(root: str, key: str):
    """The view as a worker returns it, read back here."""
    return SolPool(root).sequence(key)
