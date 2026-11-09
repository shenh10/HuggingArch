"""`browse` against a brute-force Python reference over the rows inserted.

One corpus (module-scoped lake): two owners, batches in every review state,
a within-peak shared batch with rows over their peaks, a placeholder input,
latency ties, several inputs folding into one physical op. Every view is
recomputed here with plain loops over the inserted rows — the semantics of
0022 `bench_browse` and 0021b `_bench_recompute_envelope`, on the rows — and
compared whole.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
import random
import sys
import uuid
from typing import Optional
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.inference.benchmark_identity import workload_coordinates
from backend.inference.op_calibration import measured_utilization
from backend.inference.benchmark_identity import canonical_identity_json
from backend.service.bench import browse as browse_mod, store
from backend.service.bench.browse import Query, Refused
from backend.service.bench.derived import Derived
from backend.service.bench.lake import Lake, LakeConfig
from tests.corpus_row import row as corpus_row

A = "aaaaaaaa-0000-4000-8000-000000000001"   # the caller
B = "bbbbbbbb-0000-4000-8000-000000000002"   # someone else
# How the curves view places this corpus's synthetic signatures (the route
# derives it from the catalog: `browse.call_tokens()`). `attn` either folds
# B and s_query into one axis or keeps them apart; both are exercised.
_RUNS = {"sig:moe": [["s_query"]], "sig:normb": [["B"]]}
AXES = {"total": ["n_tokens"], "factor": ["B", "s_query"], "runs": {**_RUNS, "sig:attn": [["B", "s_query"]]}}
SEPARATE = {"total": ["n_tokens"], "factor": ["B", "s_query"], "runs": {**_RUNS, "sig:attn": [["B"], ["s_query"]]}}
# Attention with its context: decode shapes (with s_kv) run along B×s_kv,
# prefill shapes (none here) along B.
CONTEXT = {"total": ["n_tokens"], "factor": ["B", "s_query"], "runs": {**_RUNS, "sig:attn": [["B", "s_kv"], ["B"], ["s_query"]]}}
# `moe` with no contract: its shapes name `n_tokens`, and are still unplaced.
NO_MOE = {"total": ["n_tokens"], "factor": ["B", "s_query"], "runs": {k: v for k, v in AXES["runs"].items() if k != "sig:moe"}}
CATALOG = {"ka": "KA", "kb": "KB"}          # kc: no catalog kernel


# --------------------------------------------------------------------- corpus

def annotate_with(catalog: dict, fold_note: bool = True):
    """Signature from op; physical shape = the shape without `note` (two
    inputs, one physical op); `pending` has no signature — refused, a
    placeholder."""
    def annotate(inp):
        if inp["op"] == "pending":
            raise ValueError("op pending has no signature")
        shape = json.loads(inp["shape_json"])
        if fold_note:
            shape.pop("note", None)
        physical = canonical_identity_json(shape)
        batch, context = workload_coordinates(inp["phase"], physical)
        return {"signature": f"sig:{inp['op']}", "physical_shape": physical,
                "workload_batch": batch, "workload_context": context,
                "catalog_kernel": catalog.get(inp["kernel_impl"])}
    return annotate


def _shape(rnd: random.Random, op: str, phase: str) -> dict:
    if op == "attn":
        b, s = rnd.choice([1, 2, 4]), rnd.choice([128, 256])
        shape = ({"B": b, "s_query": s, "d": 64} if phase == "prefill"
                 else {"B": b, "s_query": 1, "s_kv": s, "d": 64})
        if rnd.random() < 0.3:
            shape["note"] = 1
        return shape
    if op == "moe":
        t = rnd.choice([64, 128])
        return {"n_tokens": t, "s_query": t // 4, "E": 8}
    if op == "norm":
        return {"d": 64, "h": rnd.choice([8, 16])}
    if op == "normb":
        return {"B": rnd.choice([1, 2]), "d": 64}
    return {"B": 1, "s_query": 128}                # pending


_seq = iter(range(10**9))


def _csv_rows(rnd: random.Random, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        op = rnd.choice(["attn", "attn", "moe", "norm", "normb", "pending"])
        phase = rnd.choice(["prefill", "decode"])
        act = rnd.choice(["bf16", "fp8"])
        out.append(corpus_row(
            model=rnd.choice(["m1", "m2"]), spec_hash=f"{next(_seq):016x}", phase=phase,
            parallel="tp=1" if rnd.random() < 0.1 else "tp=8",
            framework=rnd.choice(["vllm", "sglang"]), gpu=rnd.choice(["H100", "A100"]),
            op=op, node_spec_id=rnd.choice(["l.a", "l.b"]),
            shape_json=json.dumps(_shape(rnd, op, phase)),
            dtype_env_json=json.dumps({"activation": act, "kv": "bf16", "metadata": "fp32",
                                       "output": "bf16", "weight": "bf16"}),
            kernel_impl=rnd.choice(["ka", "kb", "kc"]),
            latency_us=str(rnd.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])),
            achieved_tflops=rnd.choice(["", "0", "100.0", "500.0", "1200.0"]),
            peak_tflops_at_test=rnd.choice(["", "0", "1000.0", "1000.0"]),
            achieved_bw_gbps=rnd.choice(["", "200.0", "4000.0"]),
            peak_bw_at_test=rnd.choice(["3000.0", "3000.0", ""]),
            date=rnd.choice(["2026-05-16", "2026-06-01"]),
            samples=str(rnd.choice([10, 100, 1000]))))
    return out


# (owner, status, within_peak, run_id, rows)
BATCHES = [
    (B, "shared", False, None, 120),
    (B, "shared", True, "rb", 100),
    (A, "private", False, "r1", 80),
    (A, "shared", False, "r1", 60),
    (A, "nominated", False, None, 60),
    (B, "private", False, None, 60),
    (B, "rejected", False, None, 40),
    (A, "shared", True, None, 60),
]


def _num(v):
    return float(v) if v not in ("", None) else None


def _open(tmp: Path) -> Lake:
    lk = Lake.open(LakeConfig(catalog=str(tmp / "catalog.ducklake"), data=str(tmp / "data"),
                              cache_dir=str(tmp / "cache")))
    store.ensure_schema(lk)
    _ANNOTATORS[lk] = annotate_with(CATALOG)
    return lk


# The annotation rules each test lake is read under (`_open`, `build`, or a
# test naming other ones), and the derived file per (lake, rules, placement
# axes): a view reads the file built under the rules and axes it names.
_ANNOTATORS: dict = {}
_DERIVED: dict = {}


def derived_for(lk: Lake, axes: Optional[dict] = None) -> Derived:
    """The lake's derived file under its annotator and `axes` (the curves'
    placement; AXES by default), refreshed now: another annotator or other
    axes are other rules, and rebuild it."""
    axes = AXES if axes is None else axes
    ann = _ANNOTATORS[lk]
    key = (id(lk), id(ann), json.dumps(axes, sort_keys=True))
    if key not in _DERIVED:
        rules = f"{id(ann)}:{key[2]}"
        _DERIVED[key] = Derived(annotate=ann, rules=lambda: rules, tokens=lambda: axes)
    _DERIVED[key].refresh(lk)
    return _DERIVED[key]


def browse(lk: Lake, q: Query) -> dict:
    """`browse.browse` over the lake's derived file (`derived_for`, the
    curves' axes from the query's `call_tokens`). A curves query is answered
    as the page asks it: the ops listed, then each op's curves by name."""
    d = derived_for(lk, q.filters.get("call_tokens"))
    if q.view != "curves":
        return browse_mod.browse(lk, q, d)
    listed = browse_mod.browse(lk, q, d)["ops"]
    assert all(set(o) == {"signature"} for o in listed)
    return {"ops": [browse_mod.browse(lk, dataclasses.replace(
        q, filters={**q.filters, "signature": o["signature"]}), d)["ops"][0] for o in listed]}


def build(lk: Lake, batches, annotate, seed: int = 7) -> list[dict]:
    """Upload `batches`; return the reference records of every row (the lake
    read under `annotate`)."""
    _ANNOTATORS[lk] = annotate
    rnd = random.Random(seed)
    meta = {}
    for owner, status, within, run_id, n in batches:
        rows = _csv_rows(rnd, n)
        # a latency tie at the winner, on one physical op
        rows[1] = {**rows[0], "spec_hash": f"{next(_seq):016x}", "node_spec_id": "l.tie",
                   "kernel_impl": "kb" if rows[0]["kernel_impl"] != "kb" else "ka"}
        upload_id = str(uuid.uuid4())
        res = store.upload(lk, owner=owner, upload_id=upload_id, run_id=run_id, label=None,
                           rows=rows,
                           status="private" if status == "shared" and within else status)
        assert res.inserted == n
        if status == "shared" and within:
            store.set_status(lk, [store.SetRef(owner, "upload", upload_id)],
                             from_status=["private"], to_status="shared", within_peak=True)
        for r in rows:
            meta[r["spec_hash"]] = (r, owner, status, within, run_id, upload_id)
    return reference(lk, meta)


def reference(lk: Lake, meta: dict) -> list[dict]:
    """The reference records of every row of `meta` (spec_hash → upload),
    with its input's annotations as `annot` holds them."""
    derived_for(lk)
    stored = lk.read("SELECT m.id, m.spec_hash, a.signature, a.physical_shape, a.catalog_kernel,"
                     " a.workload_batch, a.workload_context FROM lake.measurements m"
                     f" LEFT JOIN {store.ANNOT} a USING (input_id)")
    out = []
    for s in stored:
        r, owner, status, within, run_id, upload_id = meta[s["spec_hash"]]
        out.append({
            **{k: r[k] for k in ("op", "framework", "framework_version", "kernel_impl", "gpu",
                                 "phase", "parallel", "model", "node_spec_id", "date")},
            "outcome": r.get("outcome") or "measured",
            "outcome_detail": r.get("outcome_detail") or None,
            "timing_method": r.get("timing_method") or None,
            "id": str(s["id"]), "shape_json": json.loads(r["shape_json"]),
            "dtype_env_json": json.loads(r["dtype_env_json"]),
            "dtype_text": canonical_identity_json(r["dtype_env_json"]),
            "signature": s["signature"], "physical_shape": s["physical_shape"],
            "catalog_kernel": s["catalog_kernel"], "workload_batch": s["workload_batch"],
            "workload_context": s["workload_context"],
            **{k: _num(r[k]) for k in ("latency_us", "achieved_tflops", "achieved_bw_gbps",
                                       "peak_tflops_at_test", "peak_bw_at_test")},
            "samples": int(r["samples"]),
            "owner": owner, "status": status, "within_peak": within, "run_id": run_id,
            "upload_id": upload_id, "activation": json.loads(r["dtype_env_json"])["activation"],
        })
    return out


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    lk = _open(tmp_path_factory.mktemp("lake"))
    rows = build(lk, BATCHES, annotate_with(CATALOG))
    yield lk, rows
    lk.close()


# ------------------------------------------------------------------ reference

def pos(x):
    return x if x is not None and x > 0 else None


def rates(r) -> dict:
    """The page's rates; MFU / MBU and the roofline check as
    `measured_utilization` has them (an absent achieved rate reads 0)."""
    mu = measured_utilization(r)
    tf, bw = r["achieved_tflops"], r["achieved_bw_gbps"]
    return {"tflops": pos(tf), "gbps": pos(bw), "mfu": pos(mu["mfu"]), "mbu": pos(mu["mbu"]),
            "ai": 1000 * tf / bw if pos(tf) and pos(bw) else None,
            "over": mu["roof_status"] not in ("ok", None), "ok": mu["roof_status"] == "ok"}


def shared(r) -> bool:
    if r["status"] != "shared":
        return False
    if not r["within_peak"]:
        return True
    status = measured_utilization(r)["roof_status"]
    return status in ("ok", None)


# A scope: (Query fields, row predicate, reader)
SCOPES = {
    "shared": ({"source": "shared", "caller": A}, shared, A),
    "mine": ({"source": "mine", "caller": A}, lambda r: r["owner"] == A, A),
    "union": ({"source": "", "caller": A}, lambda r: shared(r) or r["owner"] == A, A),
    "anonymous": ({"source": ""}, shared, None),
    "run": ({"run": "r1", "caller": A}, lambda r: r["owner"] == A and r["run_id"] == "r1", A),
    "admin-run": ({"run": "rb", "caller": A, "set_owner": B},
                  lambda r: r["owner"] == B and r["run_id"] == "rb", B),
}


def in_scope(rows, scope):
    return [r for r in rows if SCOPES[scope][1](r)]


def query(scope, **kw) -> Query:
    return Query(**SCOPES[scope][0], **kw)


def ref_row(r, reader) -> dict:
    out = {k: r[k] for k in (
        "id", "op", "signature", "framework", "framework_version", "kernel_impl",
        "catalog_kernel", "gpu", "phase", "parallel", "shape_json", "physical_shape",
        "dtype_env_json", "model", "node_spec_id", "latency_us", "achieved_tflops",
        "achieved_bw_gbps", "peak_tflops_at_test", "peak_bw_at_test", "samples", "date",
        "upload_id", "run_id", "workload_batch", "workload_context", "outcome",
        "outcome_detail", "timing_method")}
    out["origin"] = "mine" if reader is not None and r["owner"] == reader else "shared"
    return out


def ordered(items, key, ident, desc):
    have = sorted((x for x in items if key(x) is not None), key=lambda x: (key(x), ident(x)),
                  reverse=desc)
    none = sorted((x for x in items if key(x) is None), key=ident, reverse=desc)
    return have + none


def row_passes(r, f, skip=()):
    dims = {"model": r["model"], "signature": r["signature"], "gpu": r["gpu"],
            "framework": r["framework"], "dtype": r["activation"], "phase": r["phase"],
            "kernel": r["catalog_kernel"], "outcome": r["outcome"]}
    f = {"outcome": "measured", **f}
    for d, v in dims.items():
        if d not in skip and f.get(d) and v != f[d]:
            return False
    return not (f.get("check") in ("over", "ok") and "check" not in skip and not rates(r)[f["check"]])


def ref_envelopes(rows, f, reader):
    groups: dict = {}
    for r in rows:
        if r["signature"] is None or r["physical_shape"] is None:
            continue
        key = (r["signature"], r["physical_shape"], r["dtype_text"], r["gpu"], r["phase"])
        groups.setdefault(key, []).append(r)
    out = []
    for members in groups.values():
        members.sort(key=lambda r: (r["latency_us"], r["id"]))
        w = members[0]
        second = members[1]["latency_us"] if len(members) > 1 else None
        rt = rates(w)
        e = {
            "winner": ref_row(w, reader), "n_candidates": len(members),
            "gain": second / w["latency_us"] - 1 if second is not None and w["latency_us"] else None,
            "candidates": [{"id": c["id"], "latency_us": c["latency_us"], "framework": c["framework"],
                            "kernel_impl": c["kernel_impl"], "catalog_kernel": c["catalog_kernel"],
                            "model": c["model"]} for c in members[:20]],
            "models": sorted({c["model"] for c in members if c["model"] is not None}),
            "kernels": sorted({c["catalog_kernel"] for c in members if c["catalog_kernel"] is not None}),
            "key": {"signature": w["signature"], "physical_shape": w["physical_shape"],
                    "dtype_env_json": w["dtype_env_json"], "gpu": w["gpu"], "phase": w["phase"]},
            "_rates": rt, "_w": w,
        }
        keep = all(not f.get(d) or v == f[d] for d, v in (
            ("signature", w["signature"]), ("gpu", w["gpu"]), ("phase", w["phase"]),
            ("dtype", w["activation"]), ("model", w["model"]), ("framework", w["framework"])))
        if f.get("kernel") and f["kernel"] not in e["kernels"]:
            keep = False
        if f.get("check") in ("over", "ok") and not rt[f["check"]]:
            keep = False
        if keep:
            out.append(e)
    return out


ENV_SORT = {"latency": lambda e: e["_w"]["latency_us"], "gain": lambda e: e["gain"],
            "candidates": lambda e: e["n_candidates"], "tflops": lambda e: e["_rates"]["tflops"],
            "gbps": lambda e: e["_rates"]["gbps"], "mfu": lambda e: e["_rates"]["mfu"],
            "mbu": lambda e: e["_rates"]["mbu"]}


def public(e):
    return {k: v for k, v in e.items() if not k.startswith("_")}


def ref_envelope_view(rows, scope, f, sort, desc, page, size):
    reader = SCOPES[scope][2]
    envs = ordered(ref_envelopes(in_scope(rows, scope), f, reader), ENV_SORT[sort],
                   lambda e: e["_w"]["id"], desc)
    return {"total": len(envs),
            "items": [public(e) for e in envs[page * size:(page + 1) * size]]}


def ref_facets(rows, f):
    out = {}
    for d, col in (("model", "model"), ("signature", "signature"), ("gpu", "gpu"),
                   ("framework", "framework"), ("dtype", "activation"), ("phase", "phase"),
                   ("kernel", "catalog_kernel"), ("outcome", "outcome")):
        out[d] = sorted({r[col] for r in rows if r[col] is not None and row_passes(r, f, skip=(d,))})
    return out


ROW_SORT = {
    "latency": lambda r: r["latency_us"], "model": lambda r: r["model"],
    "framework": lambda r: r["framework"], "version": lambda r: r["framework_version"],
    "kernel": lambda r: r["kernel_impl"], "gpu": lambda r: r["gpu"], "phase": lambda r: r["phase"],
    "op": lambda r: r["signature"], "layer": lambda r: r["node_spec_id"],
    "shape": lambda r: r["physical_shape"], "dtype": lambda r: r["activation"],
    "samples": lambda r: r["samples"], "date": lambda r: r["date"],
    "tflops": lambda r: rates(r)["tflops"], "gbps": lambda r: rates(r)["gbps"],
    "mfu": lambda r: rates(r)["mfu"], "mbu": lambda r: rates(r)["mbu"],
    "ai": lambda r: rates(r)["ai"],
}


def strip(out: dict) -> dict:
    """The answer without its facets; a curve's lines as a set."""
    out = {k: v for k, v in out.items() if k != "facets"}
    for o in out.get("ops") or []:
        for c in o.get("curves") or []:
            c["lines"] = sorted(c["lines"])
    return out


# ---------------------------------------------------------------------- tests

@pytest.mark.parametrize("scope", list(SCOPES))
@pytest.mark.parametrize("sort", list(ENV_SORT))
@pytest.mark.parametrize("desc", [False, True])
def test_envelope_every_sort_and_scope(corpus, scope, sort, desc):
    lk, rows = corpus
    got = browse(lk, query(scope, sort=sort, desc=desc, page_size=200))
    assert strip(got) == ref_envelope_view(rows, scope, {}, sort, desc, 0, 200)
    assert got["facets"] == ref_facets(in_scope(rows, scope), {})


@pytest.mark.parametrize("sort", ["latency", "gain", "mfu"])
def test_envelope_pages(corpus, sort):
    lk, rows = corpus
    for page in range(4):
        got = browse(lk, query("union", sort=sort, desc=True, page=page, page_size=7))
        assert strip(got) == ref_envelope_view(rows, "union", {}, sort, True, page, 7)


FILTERS = [
    {"model": "m1"}, {"signature": "sig:attn"}, {"gpu": "A100"}, {"framework": "sglang"},
    {"dtype": "fp8"}, {"phase": "decode"}, {"kernel": "KA"}, {"kernel": "KB", "gpu": "H100"},
    {"check": "over"}, {"check": "ok"}, {"model": "m2", "phase": "prefill", "check": "ok"},
    {"signature": "sig:moe", "dtype": "bf16", "framework": "vllm"},
]


@pytest.mark.parametrize("scope", ["shared", "union", "run"])
@pytest.mark.parametrize("f", FILTERS, ids=lambda f: "-".join(f"{k}={v}" for k, v in f.items()))
def test_envelope_filters_and_facets(corpus, scope, f):
    lk, rows = corpus
    got = browse(lk, query(scope, filters=f, sort="mfu", desc=True, page_size=200))
    assert strip(got) == ref_envelope_view(rows, scope, f, "mfu", True, 0, 200)
    assert got["facets"] == ref_facets(in_scope(rows, scope), f)


def test_corpus_exercises_what_it_claims(corpus):
    """The cases the reference comparisons rely on are in the corpus."""
    lk, rows = corpus
    envs = ref_envelopes(in_scope(rows, "union"), {}, A)
    assert any(len(e["candidates"]) > 1 and e["candidates"][0]["latency_us"]
               == e["candidates"][1]["latency_us"] for e in envs), "a latency tie at the winner"
    assert any(e["gain"] == 0 for e in envs)
    # within-peak shared batches: some of their rows are not shared
    wp = [r for r in rows if r["status"] == "shared" and r["within_peak"]]
    assert any(shared(r) for r in wp) and any(not shared(r) for r in wp)
    # a placeholder: rows in scope, in no envelope
    ph = [r for r in rows if r["signature"] is None]
    assert ph
    got = browse(lk, query("union", page_size=200))
    ids = {c["id"] for e in got["items"] for c in e["candidates"]}
    assert not ids & {r["id"] for r in ph}
    # two inputs, one physical op
    assert any(len({(c["kernel_impl"]) for c in e["candidates"]}) > 1 for e in envs)
    # one physical op measured under two deployments: one envelope
    by_id = {r["id"]: r for r in rows}
    assert any(len({by_id[c["id"]]["parallel"] for c in e["candidates"]}) > 1 for e in envs)
    # shared ∪ mine counts a row of the caller's shared batch once
    both = [r for r in rows if r["owner"] == A and shared(r)]
    assert both
    assert got["total"] == len(envs)


def test_upload_scope_and_origin(corpus):
    lk, rows = corpus
    upload_id = next(r["upload_id"] for r in rows if r["owner"] == A and r["status"] == "nominated")
    pred = lambda r: r["upload_id"] == upload_id
    got = browse(lk, Query(upload=upload_id, caller=A, page_size=200))
    envs = ordered(ref_envelopes([r for r in rows if pred(r)], {}, A), ENV_SORT["latency"],
                   lambda e: e["_w"]["id"], False)
    assert strip(got)["items"] == [public(e) for e in envs]
    assert all(e["winner"]["origin"] == "mine" for e in got["items"])
    # someone else's upload id names nothing of the caller's
    other = next(r["upload_id"] for r in rows if r["owner"] == B)
    assert browse(lk, Query(upload=other, caller=A))["items"] == []


def test_admin_reads_owner_set_with_owner_origin(corpus):
    lk, _ = corpus
    got = browse(lk, query("admin-run", page_size=200))
    assert got["items"] and all(e["winner"]["origin"] == "mine" for e in got["items"])


def test_mine_without_caller_is_empty(corpus):
    lk, _ = corpus
    got = browse(lk, Query(source="mine"))
    assert got["items"] == [] and got["total"] == 0


def test_total_counts_every_envelope_and_pages_go_to_the_end(corpus):
    lk, _rows = corpus
    whole = browse(lk, query("union", page_size=200))
    n = whole["total"]
    assert n == len(whole["items"]) and n > 5
    last = browse(lk, query("union", page_size=5, page=(n - 1) // 5))
    assert last["total"] == n and last["items"] == whole["items"][(n - 1) // 5 * 5:]


# ----------------------------------------------------------------------- rows

def ref_rows(rows, scope, f, sort, desc, page, size, key=None):
    reader = SCOPES[scope][2]
    sel = [r for r in in_scope(rows, scope) if row_passes(r, f)]
    if key is not None:
        sel = [r for r in sel if (r["signature"], r["physical_shape"], r["dtype_env_json"], r["gpu"],
                                  r["phase"]) == (
            key["signature"], key["physical_shape"], key["dtype_env_json"], key["gpu"],
            key["phase"])]
    sel = ordered(sel, ROW_SORT[sort], lambda r: r["id"], desc)
    return {"total": len(sel),
            "items": [ref_row(r, reader) for r in sel[page * size:(page + 1) * size]]}


@pytest.mark.parametrize("sort", list(ROW_SORT))
@pytest.mark.parametrize("desc", [False, True])
def test_rows_of_a_set_every_sort(corpus, sort, desc):
    lk, rows = corpus
    got = browse(lk, query("run", view="rows", sort=sort, desc=desc, page_size=50, page=1))
    assert strip(got) == ref_rows(rows, "run", {}, sort, desc, 1, 50)


@pytest.mark.parametrize("scope", ["shared", "union", "mine"])
@pytest.mark.parametrize("f", [{}, {"check": "over"}, {"kernel": "KB", "dtype": "fp8"},
                               {"signature": "sig:attn", "gpu": "H100", "framework": "vllm"}])
def test_rows_of_a_model_and_phase(corpus, scope, f):
    lk, rows = corpus
    f = {"model": "m1", "phase": "decode", **f}
    for sort in ("mfu", "shape"):
        got = browse(lk, query(scope, view="rows", filters=f, sort=sort, desc=True, page_size=30))
        assert strip(got) == ref_rows(rows, scope, f, sort, True, 0, 30)
        assert got["facets"] == ref_facets(in_scope(rows, scope), f)


@pytest.mark.parametrize("scope", ["shared", "union", "anonymous"])
@pytest.mark.parametrize("f", [{}, {"gpu": "A100", "check": "ok"}])
def test_rows_unscoped(corpus, scope, f):
    """The rows view takes the envelope view's scope and filters: no batch,
    model or op shape needed, one page of the whole scope."""
    lk, rows = corpus
    for sort in ("latency", "op"):
        got = browse(lk, query(scope, view="rows", filters=f, sort=sort, page_size=40, page=1))
        assert strip(got) == ref_rows(rows, scope, f, sort, False, 1, 40)


def test_rows_total_counts_every_row(corpus):
    lk, _rows = corpus
    got = browse(lk, query("union", view="rows", page_size=5))
    pages = -(-got["total"] // 200)
    seen = sum(len(browse(lk, query("union", view="rows", page_size=200, page=p))["items"]) for p in range(pages))
    assert seen == got["total"] > 200 and len(got["items"]) == 5


def test_rows_of_an_envelope(corpus):
    lk, rows = corpus
    envs = browse(lk, query("union", sort="candidates", desc=True, page_size=5))["items"]
    for e in envs:
        key = json.loads(json.dumps(e["key"]))       # as the page sends it back
        got = browse(lk, query("union", view="rows", filters={"key": key}, page_size=200))
        assert strip(got) == ref_rows(rows, "union", {}, "latency", False, 0, 200, key=key)
        assert got["total"] == e["n_candidates"]
        assert got["items"][0]["id"] == e["winner"]["id"]


def test_rows_of_a_set_include_placeholders(corpus):
    lk, _rows = corpus
    got = browse(lk, query("run", view="rows", page_size=200))
    assert any(r["signature"] is None for r in got["items"])


# ------------------------------------------------------------------- workload

WORKLOAD_SORTS = ["latency", "layer", "op", "kernel", "framework", "shape", "samples",
                  "tflops", "gbps", "mfu", "mbu"]


def ref_workload(rows, scope, f, batch, context, sort, desc, page, size):
    reader = SCOPES[scope][2]
    sel = [r for r in in_scope(rows, scope) if r["model"] == f["model"] and r["phase"] == f["phase"]]
    out = {"batches": sorted({r["workload_batch"] for r in sel if r["workload_batch"] is not None}),
           "unplaced": sum(r["workload_batch"] is None for r in sel)}
    if batch is None:
        return {**out, "contexts": []}
    ctx = {r["workload_context"] for r in sel if r["workload_batch"] == batch}
    out["contexts"] = sorted(ctx, key=lambda c: (c is None, c == 0, c or 0))
    if context is None:
        return out
    cand = [r for r in sel if r["workload_batch"] == batch and r["workload_context"] in (context, 0)]
    out["candidates"] = len(cand)
    picks: dict = {}
    for r in cand:
        if not row_passes(r, f, skip=("model", "phase")):
            continue
        k = (r["node_spec_id"], r["op"])
        if k not in picks or (r["latency_us"], r["id"]) < (picks[k]["latency_us"], picks[k]["id"]):
            picks[k] = r
    p = ordered(list(picks.values()), ROW_SORT[sort], lambda r: r["id"], desc)
    return {**out, "total": len(p), "total_latency_us": sum(r["latency_us"] for r in p),
            "items": [ref_row(r, reader) for r in p[page * size:(page + 1) * size]]}


@pytest.mark.parametrize("scope", ["shared", "union", "run"])
@pytest.mark.parametrize("phase,batch,context", [
    ("prefill", None, None), ("prefill", 2, None), ("prefill", 2, 128), ("prefill", 1, 0),
    ("decode", 4, 256), ("decode", 1, 0)])
def test_workload(corpus, scope, phase, batch, context):
    lk, rows = corpus
    f = {"model": "m2", "phase": phase, "batch": batch, "context": context}
    got = browse(lk, query(scope, view="workload", filters=f, page_size=200))
    want = ref_workload(rows, scope, f, batch, context, "latency", True, 0, 200)
    got = strip(got)
    assert got.pop("total_latency_us", 0) == pytest.approx(want.pop("total_latency_us", 0))
    assert got == want


@pytest.mark.parametrize("sort", WORKLOAD_SORTS)
@pytest.mark.parametrize("desc", [None, False, True])
def test_workload_sorts_and_filters(corpus, sort, desc):
    lk, rows = corpus
    f = {"model": "m1", "phase": "prefill", "batch": 2, "context": 256, "framework": "vllm"}
    got = strip(browse(lk, query("union", view="workload", filters=f, sort=sort, desc=desc,
                                 page_size=3, page=1)))
    want = ref_workload(rows, "union", f, 2, 256, sort, desc is not False, 1, 3)
    assert got.pop("total_latency_us") == pytest.approx(want.pop("total_latency_us"))
    assert got == want


def test_workload_has_context_zero_rows(corpus):
    _, rows = corpus
    assert any(r["workload_context"] == 0 and r["workload_batch"] == 1 and r["phase"] == "prefill"
               for r in rows)


# --------------------------------------------------------------------- curves

def ref_saturation(points):
    """The token count where intensity reaches the ridge: intensity/ridge is
    MFU/MBU, so walk x upward and find where the (median per x) ratio first
    goes from below 1 to at or above it, geometric interpolation between."""
    ratios = {}
    for p in points:
        if p["mfu"] and p["mbu"]:
            ratios.setdefault(p["x"], []).append(p["mfu"] / p["mbu"])
    if not ratios:
        return {"state": "no_flops"}
    xs = sorted(ratios)
    med = []
    for x in xs:
        v = sorted(ratios[x])
        n = len(v)
        med.append(v[n // 2] if n % 2 else math.sqrt(v[n // 2 - 1] * v[n // 2]))
    if med[0] >= 1:
        return {"state": "compute", "tokens": xs[0]}
    for i in range(1, len(xs)):
        if med[i - 1] < 1 <= med[i]:
            frac = math.log(med[i - 1]) / (math.log(med[i - 1]) - math.log(med[i]))
            return {"state": "crossed", "tokens": xs[i - 1] * (xs[i] / xs[i - 1]) ** frac}
    return {"state": "memory", "tokens": xs[-1]}


def ref_lines(axis, points):
    """Pairwise: two points are neighbours along an axis of x when their
    phase and every other axis of x are equal; a line is a component of one
    axis's neighbours, two points or more, in x order; a point with no
    neighbour is a line alone. Lines compare as a set (`strip`)."""
    names = axis.split("×")
    shapes = [json.loads(p["shape"]) for p in points]
    lines = []
    for a in names:
        others = [b for b in names if b != a]
        comp = list(range(len(points)))
        for i, j in itertools.combinations(range(len(points)), 2):
            if points[i]["phase"] == points[j]["phase"] and all(shapes[i][b] == shapes[j][b] for b in others):
                comp[j] = comp[i] = min(comp[i], comp[j])
        for root in set(comp):
            members = [i for i in range(len(points)) if comp[i] == root]
            if len(members) > 1:
                lines.append(members)
    alone = [[i] for i in range(len(points)) if not any(i in line for line in lines)]
    return sorted(lines + alone)


def ref_curves(rows, scope, f, axes, max_env=50_000, max_points=4_000, key=None):
    """x: the one `total` axis a shape has, else the product of the first
    run of factors (as the signature's contract lays them out) the shape
    has; the key: gpu, dtype and the shape without them (no phase); a point
    needs an MFU or an MBU. A signature with no runs (no contract) is
    unplaced, whatever axes its shapes name."""
    sel = in_scope(rows, scope)
    if key is not None:
        sel = [r for r in sel if (r["signature"], r["physical_shape"], r["dtype_env_json"], r["gpu"],
                                  r["phase"]) == (
            key["signature"], key["physical_shape"], key["dtype_env_json"], key["gpu"],
            key["phase"])]
    sigs = sorted({r["signature"] for r in sel if r["signature"] is not None and row_passes(r, f)})
    envs = [e for e in ref_envelopes(sel, f, None)
            if e["_rates"]["mfu"] is not None or e["_rates"]["mbu"] is not None]
    ops = []
    for sig in sigs:
        mine = [e for e in envs if e["key"]["signature"] == sig]
        if len(mine) > max_env:
            ops.append({"signature": sig, "too_many": True})
            continue
        curves: dict = {}
        unplaced = 0
        for e in mine:
            w = e["_w"]
            shape = json.loads(w["physical_shape"])
            def has(a):
                return type(shape.get(a)) is int and shape[a] > 0
            totals = [a for a in axes["total"] if has(a)]
            runs = axes["runs"].get(sig, [])
            if sig not in axes["runs"]:  # no contract: nothing says what its axes are
                x_axes = taken = []
            elif len(totals) == 1:      # the count itself; the factors move with it
                x_axes, taken = totals, totals + [a for r in runs for a in r if has(a)]
            elif totals:
                x_axes = taken = []
            else:                       # the first run of factors the shape has, whole
                x_axes = taken = next((r for r in runs if all(has(a) for a in r)), [])
            if not x_axes:
                unplaced += 1
                continue
            rest = {k: v for k, v in shape.items() if k not in taken}
            axis = "×".join(x_axes)
            x = 1
            for a in x_axes:
                x *= shape[a]
            ck = (w["gpu"], w["dtype_text"], axis, json.dumps(rest, sort_keys=True))
            c = curves.setdefault(ck, {"gpu": w["gpu"],
                                       "dtype_env_json": w["dtype_env_json"], "axis": axis,
                                       "shape": rest, "points": []})
            c["points"].append({"x": x, "shape": w["physical_shape"], "phase": w["phase"],
                                "framework": w["framework"],
                                "kernel": w["catalog_kernel"] or w["kernel_impl"], "model": w["model"],
                                "latency_us": w["latency_us"], "tflops": e["_rates"]["tflops"],
                                "gbps": e["_rates"]["gbps"], "mfu": e["_rates"]["mfu"],
                                "mbu": e["_rates"]["mbu"]})
        kept, upto = [], 0
        for ck in sorted(curves, key=lambda k: (-len(curves[k]["points"]), k)):
            c = curves[ck]
            upto += len(c["points"])
            if upto <= max_points or not kept:
                c["points"].sort(key=lambda p: (p["x"], p["shape"], p["phase"], p["latency_us"]))
                c["saturation"] = ref_saturation(c["points"])
                c["lines"] = ref_lines(c["axis"], c["points"])
                kept.append(c)
        ops.append({"signature": sig, "points": sum(len(c["points"]) for c in curves.values()),
                    "curves_total": len(curves), "unplaced": unplaced, "curves": kept})
    return {"ops": ops}


@pytest.mark.parametrize("scope", ["shared", "union", "mine"])
@pytest.mark.parametrize("f", [{}, {"gpu": "H100"}, {"kernel": "KA", "check": "ok"}, {"model": "m1"}])
@pytest.mark.parametrize("axes", [AXES, SEPARATE, CONTEXT, NO_MOE],
                         ids=["attn-folded", "attn-separate", "attn-context", "moe-no-contract"])
def test_curves(corpus, scope, f, axes):
    lk, rows = corpus
    got = browse(lk, query(scope, view="curves", filters={**f, "call_tokens": axes}))
    assert "facets" not in got
    assert strip(got) == ref_curves(rows, scope, f, axes)
    attn = next(o for o in got["ops"] if o["signature"] == "sig:attn")
    if axes is CONTEXT:
        # Decode (s_kv) on B×s_kv with s_query kept; prefill on B: never one curve.
        assert {c["axis"] for c in attn["curves"]} == {"B×s_kv", "B"}
        assert all("s_query" in c["shape"] for c in attn["curves"])
        assert all(len({p["phase"] for p in c["points"]}) == 1 for c in attn["curves"])
    if axes is NO_MOE:
        moe = next(o for o in got["ops"] if o["signature"] == "sig:moe")
        assert (moe["curves"], moe["points"]) == ([], 0) and moe["unplaced"] > 0
    if axes is SEPARATE:
        # Kept apart, B is the x and s_query stays in the curve: prefill and
        # decode attention never share a curve.
        assert {c["axis"] for c in attn["curves"]} == {"B"}
        assert all("s_query" in c["shape"] for c in attn["curves"])
        assert all(len({p["phase"] for p in c["points"]}) == 1 for c in attn["curves"])
    if not f and axes is AXES:
        assert any(o.get("unplaced") for o in got["ops"])
        assert any(o.get("curves") for o in got["ops"])
        curves = [c for o in got["ops"] for c in o.get("curves", [])]
        # The corpus exercises each rule: a total axis, a factor product,
        # one curve across phases, and a point with an MBU but no MFU.
        assert {c["axis"] for c in curves} >= {"n_tokens", "B×s_query", "B"}
        assert any(len({p["phase"] for p in c["points"]}) > 1 for c in curves)
        assert any(p["mfu"] is None and p["mbu"] is not None for c in curves for p in c["points"])
        # Two launches at one x (B=2 s_query=4 beside B=8 s_query=1) sit on
        # one curve and never on one line.
        assert any(len({p["x"] for p in c["points"]}) < len(c["points"]) for c in curves)
        assert all(len({c["points"][i]["x"] for i in line}) == len(line) for c in curves for line in c["lines"])


def test_curves_of_an_envelope(corpus):
    """An envelope's key narrows the charts to that one op shape — the
    drill-down the rows view opens."""
    lk, rows = corpus
    envs = browse(lk, query("union", sort="candidates", desc=True, page_size=5))["items"]
    for e in envs:
        key = json.loads(json.dumps(e["key"]))       # as the page sends it back
        got = browse(lk, query("union", view="curves", filters={"key": key, "call_tokens": AXES}))
        assert strip(got) == ref_curves(rows, "union", {}, AXES, key=key)
        assert {o["signature"] for o in got["ops"]} == {key["signature"]}


def test_curves_bounds(corpus, monkeypatch):
    lk, rows = corpus
    monkeypatch.setattr(browse_mod, "_CURVE_ENVELOPES", 10)
    monkeypatch.setattr(browse_mod, "_CURVE_POINTS", 4)
    got = strip(browse(lk, query("union", view="curves", filters={"call_tokens": AXES})))
    want = ref_curves(rows, "union", {}, AXES, max_env=10, max_points=4)
    assert got == want
    assert any(o.get("too_many") for o in got["ops"])
    assert any(len(o.get("curves", [])) < o.get("curves_total", 0) for o in got["ops"])


# ------------------------------------------------------------------- coverage

@pytest.mark.parametrize("scope", ["shared", "union", "run"])
@pytest.mark.parametrize("f", [{}, {"gpu": "A100", "dtype": "bf16"}, {"phase": "decode", "check": "over"},
                               {"model": "m1", "kernel": "KA", "framework": "vllm"}])
def test_coverage(corpus, scope, f):
    lk, rows = corpus
    got = browse(lk, query(scope, view="coverage", filters=f))
    sel = [r for r in in_scope(rows, scope)
           if row_passes(r, f, skip=("model", "signature", "framework", "kernel", "outcome"))]
    kernels: dict = {}
    unclassified: dict = {}
    outcomes: dict = {}
    for r in sel:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        if r["outcome"] != f.get("outcome", "measured"):
            continue
        if r["catalog_kernel"] is not None:
            kernels[r["catalog_kernel"]] = kernels.get(r["catalog_kernel"], 0) + 1
        else:
            unclassified[r["framework"] or ""] = unclassified.get(r["framework"] or "", 0) + 1
    assert strip(got) == {"kernels": kernels, "unclassified": unclassified,
                          "outcomes": outcomes}
    assert got["facets"] == ref_facets(in_scope(rows, scope), f)


# -------------------------------------------------------------- outcomes

def test_failed_attempts_are_rows_a_page_can_ask_for_by_outcome(tmp_path):
    """A failed attempt (`error`, `timeout`, …) is a row of the shared table
    like a measurement. Every view that reads a latency keeps to
    measurements and refuses another outcome; the rows view shows the
    outcome asked for, the facets list every outcome present, and coverage
    counts them. If the scope stopped excluding them, the envelope would
    gain winners with no latency; if it excluded them everywhere, no page
    could see a failure at all."""
    lk = _open(tmp_path)
    try:
        good = _csv_rows(random.Random(11), 6)
        broken = []
        for i, (outcome, detail) in enumerate((("error", "RuntimeError: boom"),
                                              ("timeout", "one launch over the budget"),
                                              ("error", "ValueError: bad layout"))):
            r = dict(good[i], spec_hash=f"{next(_seq):016x}", op="norm",
                     shape_json=json.dumps({"d": 64, "h": 8}), kernel_impl="ka",
                     outcome=outcome, outcome_detail=detail, cross_check="unknown")
            for c in ("latency_us", "p50_us", "p95_us", "samples", "count_per_step",
                      "achieved_tflops", "achieved_bw_gbps", "peak_tflops_at_test",
                      "peak_bw_at_test", "cross_check_rel_err", "cross_check_cosine",
                      "xcheck_typical", "xcheck_worst"):
                r[c] = ""
            broken.append(r)
        _uploaded(lk, good + broken, A, "shared", annotate_with(CATALOG))
        q = {"source": "shared", "caller": A, "page_size": 200}

        envelope = browse(lk, Query(**q))
        assert all(e["winner"]["latency_us"] is not None for e in envelope["items"])
        assert envelope["facets"]["outcome"] == ["error", "measured", "timeout"]

        assert browse(lk, Query(view="rows", **q))["total"] == len(good)
        errors = browse(lk, Query(view="rows", filters={"outcome": "error"}, **q))
        assert errors["total"] == 2
        assert {r["outcome_detail"] for r in errors["items"]} == {
            "RuntimeError: boom", "ValueError: bad layout"}
        assert all(r["latency_us"] is None for r in errors["items"])

        coverage = browse(lk, Query(view="coverage", **q))
        assert coverage["outcomes"] == {"measured": len(good), "error": 2, "timeout": 1}
        assert sum(coverage["kernels"].values()) + sum(
            coverage["unclassified"].values()) == len(good)
        timeouts = browse(lk, Query(view="coverage", filters={"outcome": "timeout"}, **q))
        assert timeouts["kernels"] == {"KA": 1}

        with pytest.raises(Refused, match="only a measurement has one"):
            browse(lk, Query(filters={"outcome": "error"}, **q))
        with pytest.raises(Refused, match="outcome is one of"):
            browse(lk, Query(view="rows", filters={"outcome": "broken"}, **q))
    finally:
        lk.close()


def test_the_rows_view_lists_measurements_and_failures_together(tmp_path):
    """`outcome=all` is the rows view's way to see a batch as it ran — the
    measurements and the attempts that failed, each failure carrying its
    detail. A view that reads a latency cannot answer it."""
    lk = _open(tmp_path)
    try:
        good = _csv_rows(random.Random(12), 3)
        failed = dict(good[0], spec_hash=f"{next(_seq):016x}", op="norm",
                      shape_json=json.dumps({"d": 64, "h": 8}), kernel_impl="ka",
                      outcome="error", outcome_detail="RuntimeError: boom",
                      cross_check="unknown")
        for c in ("latency_us", "p50_us", "p95_us", "samples", "count_per_step",
                  "achieved_tflops", "achieved_bw_gbps", "peak_tflops_at_test",
                  "peak_bw_at_test", "cross_check_rel_err", "cross_check_cosine",
                  "xcheck_typical", "xcheck_worst"):
            failed[c] = ""
        _uploaded(lk, [*good, failed], A, "shared", annotate_with(CATALOG))
        q = {"source": "shared", "caller": A, "page_size": 200}

        page = browse(lk, Query(view="rows", filters={"outcome": "all"}, **q))
        assert page["total"] == 4
        # No latency sorts last (NULLS LAST); the failure keeps its detail.
        assert [r["outcome"] for r in page["items"][-1:]] == ["error"]
        assert page["items"][-1]["outcome_detail"] == "RuntimeError: boom"
        assert {r["outcome"] for r in page["items"][:3]} == {"measured"}
        # `all` is not an outcome the facets pretend to filter by; they list
        # the real ones, over the whole scope.
        assert page["facets"]["outcome"] == ["error", "measured"]

        for view in ("envelope", "workload", "curves", "coverage"):
            with pytest.raises(Refused, match="only the rows view"):
                browse(lk, Query(view=view, filters={"outcome": "all"}, **q))
    finally:
        lk.close()


# ------------------------------------------------------------------- refusals

@pytest.mark.parametrize("q,message", [
    (Query(view="roofline"), "unknown view roofline"),
    (Query(sort="model"), "envelopes sort by latency, gain, candidates, tflops, gbps, mfu or mbu, not model"),
    (Query(view="rows", run="r1", caller=A, sort="nope"), "rows do not sort by nope"),
    (Query(view="workload", filters={"model": "m1"}), "pick a model and a phase to see its workloads"),
    (Query(view="workload", filters={"model": "m1", "phase": "decode"}, sort="date"),
     "workload rows do not sort by date"),
])
def test_refusals(corpus, q, message):
    lk, _ = corpus
    with pytest.raises(Refused) as e:
        browse(lk, q)
    assert str(e.value) == message


def test_deepest_page_is_answered(corpus):
    lk, _ = corpus
    assert browse(lk, Query(page=199, page_size=50))["items"] == []


def test_workload_candidates_bound(corpus, monkeypatch):
    lk, rows = corpus
    monkeypatch.setattr(browse_mod, "_WORKLOAD_CANDIDATES", 2)
    f = {"model": "m2", "phase": "prefill", "batch": 2, "context": 128}
    n = ref_workload(rows, "union", f, 2, 128, "latency", True, 0, 1)["candidates"]
    assert n > 2
    with pytest.raises(Refused) as e:
        browse(lk, query("union", view="workload", filters=f))
    assert str(e.value) == f"this workload has {n} measurements; the view reads up to 50,000 — pick a source"


# ------------------------------------------ annotations computed in memory

def _uploaded(lk: Lake, rows: list[dict], owner: str, status: str, annotate) -> dict:
    """Upload `rows` as one batch; their `meta` entries for `reference`."""
    upload_id = str(uuid.uuid4())
    store.upload(lk, owner=owner, upload_id=upload_id, run_id=None, label=None, rows=rows, status=status)
    return {r["spec_hash"]: (r, owner, status, False, None, upload_id) for r in rows}


def test_a_catalog_change_takes_effect_at_the_next_start(tmp_path):
    """Annotations are derived under the annotation rules: under another
    catalog (other rules) the derived file is rebuilt, and every row is
    grouped by the new rules at once."""
    lk = _open(tmp_path)
    try:
        meta = _uploaded(lk, _csv_rows(random.Random(3), 120), A, "shared", annotate_with(CATALOG))
        # `note` no longer folds (more physical ops) and kc becomes catalog kernel KC
        new = _ANNOTATORS[lk] = annotate_with({**CATALOG, "kc": "KC"}, fold_note=False)
        assert derived_for(lk).refresh(lk)["how"] == "current"
        assert lk.read(f"SELECT count(*) AS n FROM {store.ANNOT} WHERE signature IS NOT NULL"
                       )[0]["n"] == len({
            (r["op"], r["phase"], r["kernel_impl"], canonical_identity_json(r["shape_json"]))
            for r, *_ in meta.values() if r["op"] != "pending"})
        del new
        rows = reference(lk, meta)
        assert any(r["shape_json"].get("note") for r in rows)
        for sort in ("latency", "candidates"):
            got = browse(lk, Query(source="shared", caller=A, sort=sort, page_size=200))
            assert strip(got) == ref_envelope_view(rows, "shared", {}, sort, False, 0, 200)
            assert "KC" in got["facets"]["kernel"]
    finally:
        lk.close()


def test_an_uploaded_input_is_in_envelopes_at_once(tmp_path):
    """An upload's new inputs are annotated by the refresh the next view
    waits for: that view groups them; a refused one (no signature) is a
    placeholder — in the rows, in no envelope."""
    lk = _open(tmp_path)
    try:
        meta = _uploaded(lk, _csv_rows(random.Random(4), 40), A, "shared", annotate_with(CATALOG))
        before = browse(lk, query("shared", page_size=200))["total"]
        fresh = _csv_rows(random.Random(5), 1)[0]
        fresh.update(op="brandnew", phase="prefill", shape_json=json.dumps({"B": 3, "d": 7}))
        pending = {**fresh, "spec_hash": f"{next(_seq):016x}", "op": "pending"}
        meta.update(_uploaded(lk, [fresh, pending], A, "shared", annotate_with(CATALOG)))
        rows = reference(lk, meta)
        got = browse(lk, query("shared", page_size=200))
        assert got["total"] == before + 1
        assert strip(got) == ref_envelope_view(rows, "shared", {}, "latency", False, 0, 200)
        assert "sig:brandnew" in got["facets"]["signature"]
        listed = browse(lk, query("shared", view="rows", page_size=200))
        assert {r["op"] for r in listed["items"]} >= {"brandnew", "pending"}
        assert next(r for r in listed["items"] if r["op"] == "pending")["signature"] is None
    finally:
        lk.close()


def test_one_shape_cut_by_two_deployments_is_one_envelope(tmp_path):
    """The deployment that cut a per-rank shape is provenance, not part of
    the kernel's identity (0024): one card runs it the same way under tp=8
    and ep=8. Both rows are candidates of one envelope; the faster wins; the
    envelope's rows still say which deployment each was measured for."""
    lk = _open(tmp_path)
    try:
        base = _csv_rows(random.Random(2), 1)[0]
        base.update(op="normb", phase="prefill", shape_json=json.dumps({"B": 2, "d": 64}),
                    achieved_tflops="100.0", peak_tflops_at_test="1000.0",
                    achieved_bw_gbps="200.0", peak_bw_at_test="3000.0")
        rows = [{**base, "spec_hash": f"{next(_seq):016x}", "latency_us": "5.0",
                 "parallel": "tp_attention=8"},
                {**base, "spec_hash": f"{next(_seq):016x}", "latency_us": "3.0",
                 "parallel": "ep_moe=8"}]
        store.upload(lk, owner=A, upload_id=str(uuid.uuid4()), run_id=None, label=None,
                     rows=rows, status="shared")
        items = browse(lk, query("shared"))["items"]
        assert len(items) == 1
        env = items[0]
        assert env["n_candidates"] == 2 and env["winner"]["latency_us"] == 3.0
        assert "parallel" not in env["key"]
        got = browse(lk, query("shared", view="rows", filters={"key": env["key"]}))["items"]
        assert sorted(r["parallel"] for r in got) == ["ep_moe=8", "tp_attention=8"]
        curves = browse(lk, query("shared", view="curves", filters={"call_tokens": AXES}))
        (op,) = curves["ops"]
        assert op["points"] == 1 and "parallel" not in op["curves"][0]
    finally:
        lk.close()


def test_an_op_without_physical_shapes_is_listed_by_curves(tmp_path):
    """An input with a signature but no physical shape is in no envelope;
    the curves view still lists its op (0022 listed every signature in
    scope), with no points."""
    lk = _open(tmp_path)
    try:
        base = annotate_with(CATALOG)

        def annotate(inp):
            a = dict(base(inp))
            if inp["op"] == "norm":
                a["physical_shape"] = None
            return a
        rows = build(lk, [(A, "shared", False, None, 120)], annotate, seed=5)
        assert any(r["signature"] == "sig:norm" for r in rows)
        got = browse(lk, query("shared", view="curves", filters={"call_tokens": AXES}))
        assert strip(got) == ref_curves(rows, "shared", {}, AXES)
        norm = next(o for o in got["ops"] if o["signature"] == "sig:norm")
        assert norm == {"signature": "sig:norm", "points": 0, "curves_total": 0, "unplaced": 0,
                        "curves": []}
        env = browse(lk, query("shared", page_size=200))
        assert all(e["key"]["signature"] != "sig:norm" for e in env["items"])
    finally:
        lk.close()


def test_placement_states_each_rule():
    """`_Placement.place`, outcome by outcome — written out, not recomputed
    the way `ref_curves` does: (axis, x, rest) or None (unplaced)."""
    place = browse_mod._Placement({"fold": [["B", "s_query"]], "apart": [["B"], ["s_query"]],
                                   "perm": [["s_query"]]}, ["n_tokens"], ["B", "s_query"])

    def at(sig, shape):
        got = place.place(sig, json.dumps(shape))
        return None if got is None else got[:3]
    # A folded run: the product, both leave the shape.
    assert at("fold", {"B": 4, "s_query": 128, "d_model": 64}) == ("B×s_query", 512, {"d_model": 64})
    # Axes kept apart: the first one is the x; the other stays in the shape.
    assert at("apart", {"B": 4, "s_query": 128, "s_kv": 128}) == ("B", 4, {"s_query": 128, "s_kv": 128})
    assert at("apart", {"s_query": 1, "s_kv": 64}) == ("s_query", 1, {"s_kv": 64})
    # One total: it is the x, and the factors present leave with it.
    assert at("perm", {"n_tokens": 125, "s_query": 16, "n_experts": 8}) == ("n_tokens", 125, {"n_experts": 8})
    # The product is exact past 2^53 (9,007,199,254,740,992).
    assert at("fold", {"B": 96259267, "s_query": 93673913}) == ("B×s_query", 9016982202401771, {})
    # A run is placed only whole; a non-positive or non-integer axis is not a token count.
    assert at("fold", {"B": 0, "s_query": 4, "d": 1}) is None
    assert at("apart", {"B": 2.5, "s_query": 4}) == ("s_query", 4, {"B": 2.5})
    # No token axis, a signature with no contract, or two totals: unplaced.
    assert at("fold", {"d_model": 64, "n_heads": 8}) is None
    assert at("unknown", {"B": 4, "s_query": 4}) is None
    assert at("unknown", {"n_tokens": 4, "s_query": 2}) is None    # a total is no contract either
    two = browse_mod._Placement({"x": [["B"]]}, ["n_tokens", "n_other"], ["B"])
    assert two.place("x", json.dumps({"n_tokens": 4, "n_other": 2, "B": 1})) is None


def test_token_runs_read_the_layout_grammar():
    """`OpSignature.token_runs` on the declaration grammar: a product tuple
    folds its factors, a name stands alone, and a factor that is a term of
    a sum (`("+", …)`, at the top or nested in a product) is no run."""
    from backend.inference.op_signatures import OpSignature, TensorLayout

    def runs(*axes):
        return OpSignature(name="t", inputs=(TensorLayout("x", axes),)).token_runs(("B", "s_query"))
    assert runs(("B", "s_query"), "d") == (("B", "s_query"),)
    assert runs("B", "h", "s_query", "d") == (("B",), ("s_query",))
    assert runs(("n_experts", "s_query"), "d") == (("s_query",),)
    assert runs((2, "d_ffn"), ("B", "s_query")) == (("B", "s_query"),)
    assert runs(("+", "B", "s_query"), "d") == ()
    assert runs(("B", ("+", "s_query", "d_pad")), "d") == ()
    assert runs(("B", "s_query"), ("+", "d_q", "d_k")) == (("B", "s_query"),)   # a sum of other axes


def test_saturation_states_each_case():
    """`_saturation` written out: where MFU/MBU (intensity over the ridge)
    first reaches 1 along the tokens, on log tokens."""
    def pts(*xr):
        return [{"x": x, "mfu": r * 0.1, "mbu": 0.1} for x, r in xr]
    sat = browse_mod._saturation
    # Crossing between 64 (ratio 1/2) and 256 (ratio 2): geometric middle, 128.
    got = sat(pts((16, 0.25), (64, 0.5), (256, 2.0), (1024, 4.0)))
    assert got["state"] == "crossed" and got["tokens"] == pytest.approx(128.0)
    assert sat(pts((8, 1.0), (64, 3.0))) == {"state": "compute", "tokens": 8}
    assert sat(pts((8, 0.1), (64, 0.5))) == {"state": "memory", "tokens": 64}
    assert sat([{"x": 4, "mfu": None, "mbu": 0.8}]) == {"state": "no_flops"}
    # Two points at one x (two phases): their median ratio decides.
    got = sat(pts((64, 0.5), (256, 0.5), (256, 8.0), (256, 2.0)))
    assert got["state"] == "crossed" and got["tokens"] == pytest.approx(128.0)


def test_token_runs_put_the_context_with_the_batch():
    """A contract with a `context` axis runs first along batch × context;
    one without is unchanged."""
    from backend.inference.op_signatures import OpSignature, TensorLayout
    attn = OpSignature(name="a", inputs=(TensorLayout("q", ("B", "h", "s_query", "d")),
                                         TensorLayout("k", ("B", "h", "s_kv", "d"))))
    assert attn.token_runs(("B", "s_query"), ("s_kv",), ("B",)) == (("B", "s_kv"), ("B",), ("s_query",))
    scored = OpSignature(name="i", inputs=(TensorLayout("q", (("B", "s_query"), "d")),
                                           TensorLayout("k", ("B", "s_kv", "d"))))
    assert scored.token_runs(("B", "s_query"), ("s_kv",), ("B",)) == (("B", "s_kv"), ("B", "s_query"))
    gemm = OpSignature(name="g", inputs=(TensorLayout("x", (("B", "s_query"), "k")),))
    assert gemm.token_runs(("B", "s_query"), ("s_kv",), ("B",)) == (("B", "s_query"),)


def test_call_tokens_derives_attention_from_the_catalog():
    runs = browse_mod.call_tokens()["runs"]
    assert runs["gqa_attention"][0] == ["B", "s_kv"]
    assert runs["gemm"] == [["B", "s_query"]]
    assert runs["grouped_gemm"] == [["s_expert_rows"]]
    # The dispatch boundary: the permute reads the token side, so it runs
    # along it; the combine writes it, so its contract keeps the routed run.
    assert runs["moe_permute"][0] == ["B", "s_query"]
    assert runs["moe_combine"] == [["s_expert_rows"]]


def test_a_dispatch_boundary_curve_runs_along_its_routed_total():
    """The boundary cells name both sides of the dispatch, which grow
    together; their routed total `n_tokens` is what a curve runs along, so
    the batch, the query and the per-expert rows all move with it and only
    the experts and the width stay in the curve's identity."""
    tokens = browse_mod.call_tokens()
    place = browse_mod._Placement(tokens["runs"], tokens["total"], tokens["factor"])
    for signature in ("moe_permute", "moe_combine", "grouped_gemm"):
        shape = {"n_experts": 256, "s_expert_rows": 1, "n_tokens": 256,
                 "d_model": 7168}
        if signature != "grouped_gemm":
            shape.update(B=1, s_query=1)
        axis, x, rest, _key, _factors = place.place(signature, json.dumps(shape, sort_keys=True))
        assert (axis, x) == ("n_tokens", 256), signature
        assert rest == {"n_experts": 256, "d_model": 7168}, signature


def test_a_capped_context_is_no_context_run():
    """A contract that reads min(context, cap) keys (sparse attention's
    top-k) has no batch × context run; the catalog's sparse attention runs
    along B, and dense attention along B×s_kv."""
    from backend.inference.op_signatures import OpSignature, TensorLayout
    sparse = OpSignature(name="s", inputs=(TensorLayout("q", ("B", "s_query", "d")),
                                           TensorLayout("c", ("B", "s_kv", "d")),
                                           TensorLayout("i", ("B", "s_query", "n_topk_keys"))),
                         attends_at_most=("n_topk_keys",))
    assert sparse.token_runs(("B", "s_query"), ("s_kv",), ("B",)) == (("B",), ("s_query",))
    runs = browse_mod.call_tokens()["runs"]
    assert runs["mla_sparse_attention"][0] == ["B"]
    assert runs["mla_attention"][0] == ["B", "s_kv"]
    with pytest.raises(ValueError, match="attends_at_most"):
        from backend.inference.op_signatures import register_signature
        register_signature(OpSignature(name="bad", inputs=(TensorLayout("q", ("B", "s_kv")),),
                                       attends_at_most=("n_topk_keys",)))


@pytest.mark.parametrize("view", ["curves", "envelope", "rows", "workload", "coverage"])
def test_no_view_reads_the_lakes_files(corpus, view, monkeypatch):
    """Every view is a read of the derived file (`browse.ROWS`, `GROUPS`,
    `BATCHES`, `PLACEMENT`): computed per request over the lake's files,
    the views took 2-11 s on a 4.28M-row lake and the page's concurrent
    requests ran past the API's limit. Curves first: the one that read most."""
    lk, rows = corpus
    derived_for(lk)
    seen: list = []
    real = lk.read

    def read(sql, params=()):
        seen.append(sql)
        return real(sql, params)
    monkeypatch.setattr(lk, "read", read)
    r = next(r for r in rows if r["workload_batch"] is not None)
    f = ({"model": r["model"], "phase": r["phase"], "batch": r["workload_batch"],
          "context": r["workload_context"]} if view == "workload" else {})
    browse(lk, query("union", view=view, filters=f))
    assert seen and not [q for q in seen if "lake.measurements" in q or "lake.batches" in q
                         or "lake.inputs" in q]
