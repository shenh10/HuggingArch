"""The batch pages over the lake (backend/service/bench/sets.py), each view
against a brute-force reference computed from the rows the test inserted."""
from __future__ import annotations

import json
import uuid

import pytest

from backend.inference import benchmarks as _benchmarks
from backend.inference.op_calibration import measured_utilization
from backend.inference.benchmark_identity import canonical_identity_json
from backend.service.bench import sets, store
from backend.service.bench.store import SetRef
from tests.backend.service.bench.conftest import derive, make_annotator


def _overlap(lake, ref, **kw):
    """`sets.overlap` over the lake's derived file (the fixture's rules)."""
    return sets.overlap(lake, ref, derive(lake, make_annotator(), rules="test"), **kw)


def _shape(i):
    """Row i's shape: the corpus row's with s_ctx 2048·(i+1) — distinct
    identities that `validate_text` accepts (the export validates)."""
    return json.dumps({"B": 1, "d_head": 128, "n_heads": 24, "s_ctx": 2048 * (i + 1)},
                      sort_keys=True, separators=(",", ":"))


@pytest.fixture
def rows_of(make_rows):
    return lambda n, **over: make_rows(n, shape_json=_shape, **over)


def _upload(lake, annotate, owner, rows, *, run_id=None, label=None):
    upload_id = str(uuid.uuid4())
    res = store.upload(lake, owner=owner, upload_id=upload_id, run_id=run_id,
                       label=label, rows=rows)
    assert res.inserted == len(rows)
    return upload_id


def _move(lake, owner, upload_id, frm, to, within_peak=None):
    changed = store.set_status(lake, [SetRef(owner, "upload", upload_id)],
                               from_status=[frm], to_status=to, within_peak=within_peak)
    assert changed


def _over(i):
    """Row 0 runs faster than its recorded compute peak."""
    return "2000.0" if i == 0 else "500.0"


def _within(r) -> bool:
    return measured_utilization(r)["roof_status"] in ("ok", None)


def _summary_of(rows) -> dict:
    """The set summary, by brute force over CSV rows."""
    def facet(col):
        return sorted({r[col] for r in rows if r[col]})
    util = [measured_utilization(r) for r in rows]
    return {
        "row_count": len(rows),
        "frameworks": facet("framework"), "gpus": facet("gpu"), "models": facet("model"),
        "ops": facet("op"), "phases": facet("phase"),
        "mfu_over": sum(u["mfu"] is not None and u["mfu"] > 1 for u in util),
        "mbu_over": sum(u["mbu"] is not None and u["mbu"] > 1 for u in util),
        "anomalous": sum(u["roof_status"] not in ("ok", None) for u in util),
        "checked": sum(u["mfu"] is not None or u["mbu"] is not None for u in util),
        "measured": sum(_benchmarks.is_measurement(r) for r in rows),
        "refused": sum(not _benchmarks.is_measurement(r) for r in rows),
        **{f"refused_{o}": sum((r.get("outcome") or "measured") == o for r in rows)
           for o in ("oom", "mismatch", "error", "crash", "timeout")},
    }


def _summary_fields(d) -> dict:
    return {k: d[k] for k in _summary_of([])}


@pytest.fixture
def corpus(lake, rows_of, annotate, new_owner):
    """Owner A: run r1 = batch 1 (3 rows, shared within peak) + batch 2
    (2 rows, private); a standalone upload (2 rows, nominated within peak).
    Owner B: one shared batch overlapping part of A's run."""
    a, b = new_owner(), new_owner()
    run1 = rows_of(3, model="m/one", achieved_tflops=_over)
    run2 = rows_of(2, model="m/two", framework="sglang", gpu="H200")
    solo = rows_of(2, model="m/solo", achieved_tflops=_over, phase="decode")
    other = rows_of(2, model="m/other", contributor="b",
                      latency_us=lambda i: "100.0" if i == 0 else "900.0")
    u1 = _upload(lake, annotate, a, run1, run_id="r1", label="first")
    u2 = _upload(lake, annotate, a, run2, run_id="r1", label="second")
    u3 = _upload(lake, annotate, a, solo, label="solo")
    u4 = _upload(lake, annotate, b, other, label="theirs")
    _move(lake, a, u1, "private", "nominated", within_peak=True)
    _move(lake, a, u1, "nominated", "shared")
    _move(lake, a, u3, "private", "nominated", within_peak=True)
    _move(lake, b, u4, "private", "nominated")
    _move(lake, b, u4, "nominated", "shared")
    return {"a": a, "b": b, "u1": u1, "u2": u2, "u3": u3, "u4": u4,
            "run1": run1, "run2": run2, "solo": solo, "other": other}


# --------------------------------------------------------------------- list

def test_list_groups_batches_into_sets_with_derived_status(lake, corpus):
    out = sets.list_sets(lake, corpus["a"])
    assert [(s["kind"], s["id"]) for s in out] == [("upload", corpus["u3"]), ("run", "r1")]
    solo, run = out
    assert _summary_fields(run) == _summary_of(corpus["run1"] + corpus["run2"])
    assert _summary_fields(solo) == _summary_of(corpus["solo"])
    # A run: its earliest batch's upload id and label, the earliest time,
    # each batch's own status; still has a private batch → "private".
    assert run["run_id"] == "r1" and run["upload_id"] == corpus["u1"]
    assert run["label"] == "first"
    assert run["created_at"] == run["batches"][0]["created_at"] < run["batches"][1]["created_at"]
    assert [(b["upload_id"], b["status"], b["row_count"]) for b in run["batches"]] == [
        (corpus["u1"], "shared", 3), (corpus["u2"], "private", 2)]
    assert (run["status"], run["is_public"], run["nominated"]) == ("private", True, False)
    assert run["batches"][0]["within_peak"] is True
    assert (solo["status"], solo["is_public"], solo["nominated"], solo["run_id"]) == (
        "nominated", False, True, None)
    assert solo["created_at"].endswith("+00:00") and "T" in solo["created_at"]
    # Another owner's sets are not listed; theirs are theirs.
    assert [s["id"] for s in sets.list_sets(lake, corpus["b"])] == [corpus["u4"]]
    assert sets.list_sets(lake, str(uuid.uuid4())) == []


def test_status_rollup_order(lake, rows_of, annotate, new_owner):
    owner = new_owner()
    u1 = _upload(lake, annotate, owner, rows_of(1, model="x/1"), run_id="r")
    u2 = _upload(lake, annotate, owner, rows_of(1, model="x/2"), run_id="r")
    status = lambda: sets.list_sets(lake, owner)[0]["status"]
    assert status() == "private"
    _move(lake, owner, u1, "private", "nominated")
    assert status() == "private"
    _move(lake, owner, u2, "private", "nominated")
    assert status() == "nominated"
    _move(lake, owner, u1, "nominated", "shared")
    assert status() == "nominated"
    _move(lake, owner, u2, "nominated", "rejected")
    assert status() == "shared"


# ------------------------------------------------------------------ summary

def test_summary_of_run_upload_and_missing_set(lake, corpus):
    a = corpus["a"]
    assert sets.summary(lake, SetRef(a, "run", "r1")) == _summary_of(corpus["run1"] + corpus["run2"])
    assert sets.summary(lake, SetRef(a, "upload", corpus["u2"])) == _summary_of(corpus["run2"])
    assert sets.summary(lake, SetRef(a, "upload", corpus["u3"])) == _summary_of(corpus["solo"])
    # A set is the owner's: another owner naming it reads nothing.
    assert sets.summary(lake, SetRef(corpus["b"], "run", "r1")) == _summary_of([])
    assert sets.summary(lake, SetRef(a, "run", "nope"))["row_count"] == 0


def test_a_batch_summary_counts_each_refusal_outcome(lake, rows_of, annotate, new_owner):
    """A batch whose rows failed has nothing to roofline-check (checked = 0);
    its summary still says what happened — per outcome — which is what the
    batch pages' Check column shows instead of a bare "–"."""
    good = rows_of(2)
    refused = []
    for i, outcome in enumerate(("oom", "error")):
        r = dict(good[i], outcome=outcome, outcome_detail="boom", cross_check="unknown")
        for c in ("latency_us", "p50_us", "p95_us", "samples", "count_per_step",
                  "achieved_tflops", "achieved_bw_gbps", "peak_tflops_at_test",
                  "peak_bw_at_test", "cross_check_rel_err", "cross_check_cosine",
                  "xcheck_typical", "xcheck_worst"):
            r[c] = ""
        refused.append(r)
    owner = new_owner()
    _upload(lake, annotate, owner, good + refused)
    (s,) = sets.list_sets(lake, owner)
    assert _summary_fields(s) == _summary_of(good + refused)
    assert (s["checked"], s["measured"], s["refused"]) == (2, 2, 2)
    assert (s["refused_oom"], s["refused_error"]) == (1, 1)
    assert (s["refused_mismatch"], s["refused_crash"], s["refused_timeout"]) == (0, 0, 0)
    assert _summary_fields(s["batches"][0]) == _summary_of(good + refused)


# --------------------------------------------------------------------- rows

def _ids(lake, ref):
    return sets.rows(lake, ref, limit=1000)["rows"]


def test_rows_offset_paging(lake, corpus):
    ref = SetRef(corpus["a"], "run", "r1")
    every = _ids(lake, ref)
    ids = [r["id"] for r in every]
    assert len(ids) == 5 and ids == sorted(ids)
    pages, offset = [], 0
    while True:
        p = sets.rows(lake, ref, limit=2, offset=offset)
        assert (p["offset"], p["limit"]) == (offset, 2)
        assert p["total"] == (5 if offset == 0 else None)
        if not p["rows"]:
            break
        pages += [r["id"] for r in p["rows"]]
        offset += 2
    assert pages == ids
    assert sets.rows(lake, ref, limit=2, count=False)["total"] is None


def test_rows_columns_and_review_fields(lake, corpus):
    a = corpus["a"]
    full = _ids(lake, SetRef(a, "run", "r1"))
    expected = {(r["model"], json.dumps(json.loads(r["shape_json"]), sort_keys=True)): r
                for r in corpus["run1"] + corpus["run2"]}
    for r in full:
        src = expected[(r["model"], json.dumps(r["shape_json"], sort_keys=True))]
        assert isinstance(r["shape_json"], dict) and isinstance(r["dtype_env_json"], dict)
        u = measured_utilization(src)
        assert r["mfu"] == pytest.approx(u["mfu"]) and r["mbu"] == pytest.approx(u["mbu"])
        assert r["user_id"] == a
        assert r["latency_us"] == float(src["latency_us"])
        shared_batch = r["model"] == "m/one"
        assert r["status"] == ("shared" if shared_batch else "private")
        # The shared batch was nominated within peak: its over-peak row is
        # neither shared nor offered.
        assert r["is_public"] == (shared_batch and _within(src))
        assert (r["nominated_at"] is not None) == (shared_batch and _within(src))
        assert r["upload_id"] == (corpus["u1"] if shared_batch else corpus["u2"])
    corpus_cols = sets.rows(lake, SetRef(a, "run", "r1"), limit=10, columns="corpus")["rows"]
    assert set(corpus_cols[0]) == {*_benchmarks.ALL_COLUMNS, "id", "upload_id", "mfu", "mbu"}
    assert [r["id"] for r in corpus_cols] == [r["id"] for r in full]
    with pytest.raises(ValueError):
        sets.rows(lake, SetRef(a, "run", "r1"), limit=1, columns="everything")


def test_rows_keyset_paging(lake, corpus):
    ref = SetRef(corpus["a"], "run", "r1")
    ids = [r["id"] for r in _ids(lake, ref)]
    walked, after = [], None
    while True:
        p = sets.rows(lake, ref, limit=2, after=after) if after else \
            sets.rows(lake, ref, limit=2, since=ids[0])
        assert set(p) == {"rows", "limit", "next"}
        walked += [r["id"] for r in p["rows"]]
        if p["next"] is None:
            break
        assert p["next"] == p["rows"][-1]["id"]
        after = p["next"]
    assert walked == ids
    assert [r["id"] for r in sets.rows(lake, ref, limit=10, before=ids[2])["rows"]] == ids[:2]
    assert [r["id"] for r in sets.rows(lake, ref, limit=10, since=ids[1], before=ids[4])["rows"]] \
        == ids[1:4]
    assert [r["id"] for r in sets.rows(lake, ref, limit=10, after=ids[1])["rows"]] == ids[2:]


def _envelopes(rows, annotate):
    """Per op shape (signature, physical_shape, dtypes, gpu, phase — not
    the deployment): the fastest row's (latency, framework, kernel_impl) and
    the member count."""
    out: dict = {}
    for r in rows:
        ann = annotate(r)
        key = (ann["signature"], ann["physical_shape"],
               canonical_identity_json(r["dtype_env_json"]), r["gpu"], r["phase"])
        out.setdefault(key, []).append(r)
    return {k: (min(float(r["latency_us"]) for r in v), len(v)) for k, v in out.items()}


def test_overlap_against_shared(lake, corpus, annotate):
    a = corpus["a"]
    # Shared: A's batch 1 within its peaks, all of B's batch.
    shared_rows = [r for r in corpus["run1"] if _within(r)] + corpus["other"]
    shared = _envelopes(shared_rows, annotate)
    for ref, rows in ((SetRef(a, "run", "r1"), corpus["run1"] + corpus["run2"]),
                      (SetRef(a, "upload", corpus["u3"]), corpus["solo"])):
        mine = _envelopes(rows, annotate)
        both = [k for k in mine if k in shared]
        faster = [k for k in both if mine[k][0] < shared[k][0]]
        out = _overlap(lake, ref)
        assert out["op_shapes"] == len(mine)
        assert out["in_shared"] == len(both)
        assert out["new"] == len(mine) - len(both)
        assert out["faster_than_shared"] == len(faster)
        assert "pending" not in out
        ops = {k[0] for k in mine}
        assert out["by_op"] == [
            {"op": op, "op_shapes": sum(k[0] == op for k in mine),
             "in_shared": sum(k[0] == op for k in both),
             "faster": sum(k[0] == op for k in faster)} for op in sorted(ops)]
        listed = {(r["signature"], r["physical_shape"], r["gpu"], r["phase"]):
                  (r["latency_us"], r["shared_latency_us"], r["shared_candidates"])
                  for r in out["in_shared_list"]}
        assert listed == {(k[0], k[1], k[3], k[4]): (mine[k][0], shared[k][0], shared[k][1])
                          for k in both}
        ratios = [r["latency_us"] / r["shared_latency_us"] for r in out["in_shared_list"]]
        assert ratios == sorted(ratios)
    run = _overlap(lake, SetRef(a, "run", "r1"))
    # Shapes seq 2048 (B's), 4096 (both), 6144 (A's own shared row); the
    # run's H200 batch is new. B's 100 µs beats the run at seq 2048.
    assert (run["op_shapes"], run["in_shared"], run["new"], run["faster_than_shared"]) == (5, 3, 2, 0)
    assert len(_overlap(lake, SetRef(a, "run", "r1"), limit=1)["in_shared_list"]) == 1
    assert _overlap(lake, SetRef(a, "run", "nope"))["op_shapes"] == 0


def test_overlap_winner_framework(lake, rows_of, annotate, new_owner):
    owner, other = new_owner(), new_owner()
    _upload(lake, annotate, owner, rows_of(1, model="w/1", framework="vllm",
                                             latency_us="50.0"), run_id="w")
    _upload(lake, annotate, owner, rows_of(1, model="w/2", framework="sglang",
                                             latency_us="40.0"), run_id="w")
    u = _upload(lake, annotate, other, rows_of(1, model="w/3", framework="trtllm",
                                                 latency_us="45.0"))
    _move(lake, other, u, "private", "nominated")
    _move(lake, other, u, "nominated", "shared")
    (row,) = _overlap(lake, SetRef(owner, "run", "w"))["in_shared_list"]
    assert (row["latency_us"], row["framework"], row["shared_framework"],
            row["shared_kernel_impl"], row["shared_candidates"]) == (40.0, "sglang", "trtllm",
                                                                     "flash_attention_v3", 1)


def test_overlap_matches_shared_across_deployments(lake, rows_of, annotate, new_owner):
    """A set's op shape is in shared when shared measured the same kernel
    shape on the same card under another deployment (0024)."""
    owner, other = new_owner(), new_owner()
    _upload(lake, annotate, owner, rows_of(1, model="d/1", parallel="ep_moe=8",
                                             latency_us="30.0"), run_id="d")
    u = _upload(lake, annotate, other, rows_of(1, model="d/2", parallel="tp_attention=8",
                                                 latency_us="45.0"))
    _move(lake, other, u, "private", "nominated")
    _move(lake, other, u, "nominated", "shared")
    out = _overlap(lake, SetRef(owner, "run", "d"))
    assert (out["op_shapes"], out["in_shared"], out["faster_than_shared"]) == (1, 1, 1)


# -------------------------------------------------------------- nominations

def test_nominations_per_upload_offered_rows(lake, corpus, rows_of, annotate):
    a = corpus["a"]
    later = _upload(lake, annotate, a, rows_of(1, model="m/late"), run_id="r1", label="late")
    _move(lake, a, later, "private", "nominated")
    out = sets.nominations(lake)
    assert [(n["owner"], n["kind"], n["id"]) for n in out] == [
        (a, "upload", later), (a, "upload", corpus["u3"])]
    late, solo = out
    assert late["run_id"] == "r1" and late["label"] == "late"
    assert late["nominated_at"] > solo["nominated_at"]
    # Nominated within peak: the over-peak row is not offered.
    offered = [r for r in corpus["solo"] if _within(r)]
    assert _summary_fields(solo) == _summary_of(offered)
    assert solo["row_count"] == 1 and solo["batches"][0]["within_peak"] is True
    _move(lake, a, later, "nominated", "rejected")
    assert [n["id"] for n in sets.nominations(lake)] == [corpus["u3"]]


# ----------------------------------------------------------------- deleting

def test_any_shared_is_the_owners_shared_rows(lake, corpus):
    a, b = corpus["a"], corpus["b"]

    def ids(owner, upload_id):
        page = sets.rows(lake, SetRef(owner, "upload", upload_id), limit=100)["rows"]
        return {r["id"]: r for r in page}

    run1 = ids(a, corpus["u1"])               # shared within peak: row 0 is over peak
    over = [i for i, r in run1.items() if not _within(r)]
    within = [i for i, r in run1.items() if _within(r)]
    assert over and within
    assert sets.any_shared(lake, a, within)
    assert not sets.any_shared(lake, a, over)              # left out of the shared corpus
    assert not sets.any_shared(lake, a, list(ids(a, corpus["u2"])))   # private
    assert not sets.any_shared(lake, b, within)            # not b's rows
    assert sets.any_shared(lake, b, list(ids(b, corpus["u4"])))
    assert not sets.any_shared(lake, a, [])
