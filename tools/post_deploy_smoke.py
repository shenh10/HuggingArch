"""Post-deploy content smoke: does the deployed site actually read Kernel Bench rows?

    python tools/post_deploy_smoke.py https://www.huggingarch.com \
        [--expect-version <engine hash>] [--version-wait 900]

Why this exists: production's measured overlay matched zero Kernel Bench rows
for weeks (the lake was keyed in an old format) while every check we had
looked at status shapes — HTTP 200, 63 models, `measured_track.state ==
"reads"`. Each of those is true of a site whose lookups return nothing, so
none of them could fail. This smoke asserts content instead:

1. Estimate: take real shared decode measurements from the site's own
   Kernel Bench (`/api/bench/browse`), ask `/inference/estimate` for the same
   model / GPU / per-rank batch, and require that every per-op row whose key
   is one the lake holds (same dtypes) comes back `measured`. Zero-cost rows
   (views, transposes: `measured_reason == "zero_launch"`) are always
   "measured" and are excluded — they say nothing about lookups.
2. Gallery: `/gallery/frontier` for the landing GPU (polled while 503) answers
   200 with models, no model's measured track is `engine_error`, and some
   model whose track `reads` has measured frontier points; a revisit answers
   200 within 5 s (the cache serves it).
3. Health: `kernel_bench.lookups` must not report keys asked with zero rows
   answered.

Nothing about the corpus is hard-coded: the model, GPU and batch come from
browse. Standard library only (runs on a bare CI runner); urllib honours
`https_proxy`. Exit status 1 on any failure, with a summary.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

# How many (model, gpu, batch) reproductions to try before giving up, and the
# revisit budget the Gallery promises for a cached GPU view.
MAX_ESTIMATE_TRIES = 6
REVISIT_BUDGET_S = 5.0
_DTYPE_FIELDS = (("activation", "act_dtype"), ("weight", "weight_dtype"),
                 ("output", "output_dtype"))


@dataclass
class Report:
    """PASS / FAIL lines, printed as they come."""
    fails: list = field(default_factory=list)
    out: Callable[[str], None] = print

    def check(self, ok: bool, msg: str) -> bool:
        self.out(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            self.fails.append(msg)
        return ok

    def note(self, msg: str) -> None:
        self.out("  " + msg)


# ---------------------------------------------------------------------------
# Assertions over parsed responses (pure: the unit tests feed canned JSON).

def row_key(row: dict) -> str:
    """A browse row's lookup key, in the estimate's `measured_key` form."""
    return f"{row['signature']} {row['physical_shape']}"


def _dtypes_agree(env: dict, op_row: dict) -> bool:
    for env_name, row_name in _DTYPE_FIELDS:
        a, b = env.get(env_name), op_row.get(row_name)
        if a and b and str(a).lower() != str(b).lower():
            return False
    return True


def parallel_of(text: Optional[str]) -> dict:
    """A row's `parallel` cell (`tp_attention=8,ep_moe=8`) as the estimate
    request's `parallel` mapping: the deployment that cut the measured
    per-rank shape, which the estimate must name to reproduce it."""
    out: dict = {}
    for part in (text or "").split(","):
        name, _, value = part.partition("=")
        if name and value.isdigit():
            out[name.strip()] = int(value)
    return out


def candidate_groups(rows: Iterable[dict]) -> list:
    """Shared decode measurements grouped by what an estimate request can
    name — (model, gpu, deployment, per-rank batch `B` of the physical
    shape) — each with its keys → dtype env. Largest groups first (more keys
    to hit)."""
    groups: dict = defaultdict(dict)
    for r in rows:
        if r.get("phase") != "decode" or r.get("outcome", "measured") != "measured":
            continue
        try:
            shape = json.loads(r["physical_shape"])
        except (KeyError, TypeError, ValueError):
            continue
        b = shape.get("B") if isinstance(shape, dict) else None
        if not isinstance(b, int) or b < 1:
            continue
        groups[(r["model"], r["gpu"], r.get("parallel") or "", b)][row_key(r)] = (
            r.get("dtype_env_json") or {})
    return sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def lake_keys_in_estimate(lake_keys: dict, estimate: dict) -> list:
    """The estimate's decode per-op rows asking for a key the lake holds
    (same dtypes), zero-cost rows excluded."""
    rows = estimate.get("per_op_timing_decode") or []
    return [r for r in rows
            if not r.get("zero_cost")
            and r.get("measured_key") in lake_keys
            and _dtypes_agree(lake_keys[r["measured_key"]], r)]


def check_estimate(rep: Report, label: str, hits: list, estimate: dict) -> None:
    """Every hit must be measured, and the decode phase must count at least
    one measured row that costs something."""
    missing = [r["measured_key"] for r in hits if r.get("measured_state") != "measured"]
    rep.check(not missing,
              f"{label}: {len(hits) - len(missing)}/{len(hits)} per-op rows whose key "
              f"the shared lake holds come back measured"
              + (f" (missing e.g. {missing[0]})" if missing else ""))
    rows = estimate.get("per_op_timing_decode") or []
    costed = [r for r in rows if not r.get("zero_cost")]
    measured = [r for r in costed if r.get("measured_state") == "measured"]
    cov = ((estimate.get("phases") or {}).get("decode") or {}).get("coverage") or {}
    rep.note(f"{label}: decode coverage {cov.get('measured')}/{cov.get('total')}, "
             f"non-zero-cost measured {len(measured)}/{len(costed)}")
    rep.check(len(measured) > 0, f"{label}: decode measures at least one non-zero-cost row")


def check_frontier(rep: Report, gpu: str, status: Optional[int], body: Optional[dict]) -> None:
    rep.check(status == 200, f"frontier {gpu} answered 200 (got {status})")
    models = (body or {}).get("models") or []
    rep.check(len(models) > 0, f"frontier {gpu} has models ({len(models)})")
    states: dict = defaultdict(int)
    reading, drawn = 0, 0
    for m in models:
        state = (m.get("measured_track") or {}).get("state")
        states[state] += 1
        if state == "reads":
            reading += 1
            if any((m.get("frontier_measured") or {}).values()):
                drawn += 1
    rep.note(f"frontier {gpu}: measured_track states {dict(states)}; "
             f"{drawn}/{reading} reading models have measured points")
    rep.check(states.get("engine_error", 0) == 0,
              f"frontier {gpu}: no measured track is engine_error ({states.get('engine_error', 0)})")
    if reading:
        rep.check(drawn > 0, f"frontier {gpu}: some model that reads measurements draws measured points")


def check_revisit(rep: Report, gpu: str, status: Optional[int], seconds: float) -> None:
    rep.check(status == 200 and seconds < REVISIT_BUDGET_S,
              f"frontier {gpu} revisit answered {status} in {seconds:.1f}s "
              f"(budget {REVISIT_BUDGET_S:.0f}s)")


def check_health(rep: Report, health: Optional[dict]) -> None:
    kb = ((health or {}).get("detail") or {}).get("kernel_bench") or {}
    lk = kb.get("lookups")
    if not rep.check(isinstance(lk, dict), "health reports kernel_bench.lookups"):
        return
    keys, rows = int(lk.get("keys") or 0), int(lk.get("rows") or 0)
    rep.note(f"health: kernel_bench lookups={lk.get('lookups')} keys={keys} rows={rows}")
    rep.check(not (keys > 0 and rows == 0),
              f"health: lookups answered rows for the keys asked (keys={keys}, rows={rows})")


# ---------------------------------------------------------------------------
# HTTP.

class Client:
    def __init__(self, base: str, insecure: bool = False):
        self.base = base.rstrip("/")
        self.ctx = ssl.create_default_context()
        if insecure:
            self.ctx.check_hostname = False
            self.ctx.verify_mode = ssl.CERT_NONE

    def call(self, path: str, body: Optional[dict] = None,
             timeout: float = 300) -> tuple:
        """(status, parsed JSON or None). Never raises on an HTTP status."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method="POST" if data else "GET",
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self.ctx) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.load(e)
            except ValueError:
                return e.code, None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            print(f"  {path}: {e}", flush=True)
            return None, None


def wait_for_version(c: Client, expect: str, wait_s: float) -> bool:
    """A rolling deploy: poll /inference/version until the new engine answers."""
    deadline = time.time() + wait_s
    while True:
        _, body = c.call("/inference/version", timeout=60)
        got = (body or {}).get("code_version")
        if got == expect:
            print(f"  version {got} is live", flush=True)
            return True
        if time.time() > deadline:
            return False
        print(f"  version {got}, waiting for {expect}", flush=True)
        time.sleep(15)


def run_estimate(rep: Report, c: Client) -> None:
    _, browse = c.call("/api/bench/browse?view=rows&source=shared&phase=decode&page_size=1",
                       timeout=120)
    gpus = ((browse or {}).get("facets") or {}).get("gpu") or []
    if not rep.check(bool(gpus), "browse lists GPUs with shared decode measurements"):
        return
    rows: list = []
    for gpu in gpus:
        _, page = c.call("/api/bench/browse?view=rows&source=shared&phase=decode"
                         f"&gpu={urllib.request.quote(gpu)}&page_size=200", timeout=120)
        rows.extend((page or {}).get("items") or [])
    groups = candidate_groups(rows)
    rep.note(f"browse: {len(rows)} shared decode rows, {len(groups)} (model, gpu, parallel, B) groups")
    tried = 0
    for (model, gpu, parallel, b), keys in groups:
        if tried >= MAX_ESTIMATE_TRIES:
            break
        tried += 1
        request = {"model_id": model, "gpu_sku": gpu, "decode_batch_size": b}
        if parallel_of(parallel):
            request["parallel"] = parallel_of(parallel)
        status, est = c.call("/inference/estimate", request)
        label = f"estimate {model} @{gpu} {parallel or 'tp1'} B={b}"
        if status != 200 or not est:
            rep.note(f"{label}: {status}, trying the next group")
            continue
        hits = lake_keys_in_estimate(keys, est)
        if not hits:
            rep.note(f"{label}: no per-op key matches the {len(keys)} lake keys, next group")
            continue
        check_estimate(rep, label, hits, est)
        return
    rep.check(False, f"no estimate reproduced a shared measurement's key in {tried} tries")


def run_gallery(rep: Report, c: Client, wait_s: float) -> None:
    deadline = time.time() + wait_s
    path = "/gallery/frontier"   # the route's default is the landing GPU
    while True:
        status, body = c.call(path, timeout=120)
        if status != 503 or time.time() > deadline:
            break
        after = float((body or {}).get("retry_after_s") or 5)
        rep.note(f"frontier 503 ({(body or {}).get('reason')}), retry in {after:.0f}s")
        time.sleep(min(max(after, 1.0), 30.0))
    gpu = ((body or {}).get("config") or {}).get("gpu_sku") or "landing"
    check_frontier(rep, gpu, status, body)
    if status != 200:
        return
    t0 = time.time()
    status, _ = c.call(f"/gallery/frontier?gpu_sku={urllib.request.quote(gpu)}", timeout=60)
    check_revisit(rep, gpu, status, time.time() - t0)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base_url")
    ap.add_argument("--expect-version", default="",
                    help="wait until /inference/version answers this engine hash")
    ap.add_argument("--version-wait", type=float, default=900)
    ap.add_argument("--gallery-wait", type=float, default=600)
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification")
    a = ap.parse_args(argv)
    rep = Report(out=lambda s: print(s, flush=True))
    c = Client(a.base_url, a.insecure)
    if a.expect_version and not rep.check(
            wait_for_version(c, a.expect_version, a.version_wait),
            f"deployed engine {a.expect_version} answers /inference/version"):
        return _summary(rep)
    run_estimate(rep, c)
    run_gallery(rep, c, a.gallery_wait)
    _, health = c.call("/health", timeout=60)
    check_health(rep, health)
    return _summary(rep)


def _summary(rep: Report) -> int:
    if rep.fails:
        print(f"RESULT FAIL ({len(rep.fails)}):", flush=True)
        for f in rep.fails:
            print(f"  - {f}", flush=True)
        return 1
    print("RESULT OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
