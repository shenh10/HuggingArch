"""Re-key the lake's measurements to the leaf's own dtype environment.

The dtype half of a measurement key used to be stated per MODEL: every row
carried all five slots — `kv` the step's widest cache, `metadata` the step's
most common scale dtype (fp32 when it had none), `weight` the model's weight
dtype even on a leaf that streams none. The key is now the LEAF's own
environment (`RowIdentity.dtype_env`, `benchmark_identity.py`), which is what
the estimate looks up with and what this harness plans with, and the lookup
matches it exactly (`service/bench/lookup.py`). Every row written before is
therefore under a key nobody asks for.

This rewrites them, once per lake. The new key of an old row is not derived
here: the row's model is planned again (`plan.build_jobs_over_points`, the
same derivation a new run keys its jobs by) and the row takes the key its
node (`node_spec_id`) gets there. A row is re-keyed only when that answer is
the old key with slots dropped — every slot the new key states, the old one
stated with the same value.

One value may change: `output`. Until the walk stated the dtype a leaf
writes, every row's `output` was the BF16 filler
(`compute_per_row_dtype_env_json`'s default), so an fp16 model's rows and a
cache writer's rows were stored at `output=bf16` whatever they wrote. Such a
row takes today's `output` when the launch it timed wrote that dtype without
being told: its provider declares no output (`KernelSignature.output`, the
one gate that reads the slot — a provider that declares one was handed the
filler and wrote it), and a launch that declares none writes the dtype it
computes in (the cell's activation, or the `computes_at` it declares). A
new output the launch would not have written by itself — a cache record a
reference only quantises to when the slot asks — is left (`disagrees`).

Anything else is left as it is and counted:

  - `no_node`: the node is not in today's graph (a spec renamed it);
  - `disagrees`: the leaf's own environment contradicts what the row was
    stored under (a scale slot that was fp32 filler for a ue8m0 checkpoint:
    the row is of another kernel, not of this key);
  - `ambiguous`: two plans of the model agree with the row and with each
    other not;
  - `no_model`: the model no longer resolves to a spec.

One slot may be added: `kv_format`. A row's `kv` said an FP8 cache before
the key named the cache's record (`CachedTensor.format_name`), so DeepSeek-
V4's FlashMLA record and a per-tensor-scaled cache were one key. Such a row
takes today's `kv_format` when the launch it timed could only have touched
that record: its provider declares exactly that one (`KernelSignature.
kv_format`) — vLLM's V4 q/KV insert allocates the packed record and nothing
else, FlashMLA's FP8 sparse decode reads `fp8_ds_mla` and nothing else. A
provider that declares none, or several, chose by the slot or ignored it,
and its old row says neither (`disagrees`).

Another slot may be added: `activation_granularity`. A row written before a
launch's activation scales were keyed says neither the group it quantised at
nor that it had no scale. Such a row takes the one granularity its measured
provider declares today (`KernelSignature.activation_granularity`). This is
a key-vocabulary migration, not a re-plan: the recorded implementation, not
today's graph, is the evidence that this launch quantised at that group, so
the slot is added even when the model's node has since moved. The row is
selected by its recorded `kernel_impl`, not by signature+library: the record
names the launch that was timed, while a signature and library name every
kernel of that class, including cells this launch never measured. A provider
with no declaration did not make the slot's value unambiguous, and one with
several did not either — several refuses the whole rekey loudly rather than
guessing. A key that already states a granularity is never rewritten,
whatever that value is.

Re-keyed rows are counted `rekeyed` (slots dropped), `rekeyed_output`
(slots dropped and the filler `output` replaced) and `rekeyed_kv_format`
(slots dropped, the filler `output` replaced where it was, and the record
the provider declares added).

What this does NOT migrate, and why: the routed MoE exchange's cell. A
dispatch now keys as `routed_all_to_all` with three routing axes in its
shape (`benchmark_identity._collective_identity`), so a row measured under
the flat `all_to_all` cell is under a key nobody asks for — the same
situation as the dtype slots above. Two reasons it is not here. It is not
this migration's SHAPE: every rewrite above changes `dtype_env_json`, one
column, in one `UPDATE`, while this one changes `op` and `shape_json`,
which are INPUT KEY columns (`service/bench/store._INPUT_COLUMNS`) whose
hash is `input_id` — so it would have to register new inputs under
`inputs_lock()`, remap `input_id` on the measurement rows and refresh the
derived annotations. And it has no subjects: the lake holds 4,436,047
measurement rows and NOT ONE of them is any bare collective (no
`all_to_all`, `all_reduce`, `all_gather`, `reduce_scatter` or `send_recv`
row, and none whose `kernel_impl` names an exchange or DeepEP), because no
provider serves one — the fused all-reduce is the only collective anyone
has measured. The first routed-exchange row will be written after a
provider exists, i.e. already under the new cell.

Which plans: the model as shipped and, for each plain weight dtype the old
rows were stored at, every weight at that dtype — a BF16 run of an FP8
checkpoint was a BF16 kernel, and its leaves are keyed as the BF16 profile
keys them. Each at a few meshes (a leaf's dtype environment does not depend
on the mesh; the meshes only make the comm nodes exist).

    python -m kernel_bench.rekey_lake --out rekey.json   # what would change
    python -m kernel_bench.rekey_lake --apply rekey.json # change it

The lake is the one `BENCH_LAKE_*` names (`LakeConfig.from_env`). Planning
takes minutes and only reads, so it can run on a copy of the catalog; the
Kernel Bench service holds the lake, so `--apply` runs with it stopped, and
only writes. A group the lake no longer holds under its old key is skipped.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

#: (phase, node_spec_id) -> the keys today's plans give that node.
NodeKeys = Dict[Tuple[str, str], Set[str]]


def _profiles(weight_dtypes: Set[str]) -> List[Tuple[str, object]]:
    """The precision profiles a model's old rows are replayed under."""
    from backend.arch.precision_profile import AS_SHIPPED, PrecisionProfile
    from backend.arch.dtypes import dtype_bytes
    out = [("as_shipped", AS_SHIPPED)]
    for dt in sorted(weight_dtypes):
        try:
            plain = dtype_bytes(dt) >= 2
        except (KeyError, ValueError):
            plain = False
        if plain:
            out.append((f"weights={dt}", PrecisionProfile.canonicalize(
                {"weights": [{"select": {"roles": "weight_class"},
                              "format": {"dtype": dt}}]})))
    return out


def replay(task: Tuple[str, str, Tuple[str, ...]]) -> Tuple[str, str, Optional[List[NodeKeys]], str]:
    """`(model, gpu, [node keys per profile] or None, why not)`: the model
    planned again on `gpu` under each profile of `_profiles`."""
    model, gpu, weight_dtypes = task
    from backend.inference.framework_bindings import DEFAULT_FRAMEWORK
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.parallel_sweep import config_for_degree
    from kernel_bench.plan import PlanPoint, build_jobs_over_points
    from kernel_bench.sweep import _deployed_binding, resolve_sweep_model
    try:
        _config, resolved = resolve_sweep_model(model)
    except RuntimeError as e:  # no config or no spec: counted, not fatal
        return model, gpu, None, f"{type(e).__name__}: {e}"
    meshes = [ParallelConfig(), config_for_degree("dp_ep", 8), config_for_degree("tp_tp", 8)]
    per_profile: List[NodeKeys] = []
    for _name, precision in _profiles(set(weight_dtypes)):
        keys: NodeKeys = defaultdict(set)
        try:
            binding = _deployed_binding(DEFAULT_FRAMEWORK, gpu, precision, resolved)
        except ValueError:  # a profile the model refuses keys nothing
            per_profile.append({})
            continue
        for mesh in meshes:
            for phase in ("prefill", "decode"):
                try:
                    planned = build_jobs_over_points(
                        model_id=model, points=[PlanPoint(1024, 0, 1, 1)], phase=phase,
                        avg_output_tokens=256, parallel=mesh, resolved=resolved,
                        binding=binding, precision=precision, gpu=gpu)
                except ValueError:  # a refusal (a mesh, a point) keys nothing
                    continue
                for jobs, _dropped in planned.values():
                    for job in jobs:
                        keys[(phase, job["node_spec_id"])].add(job["physical_key"][2])
        per_profile.append(dict(keys))
    return model, gpu, per_profile, ""


#: What every row's `output` slot was before the walk stated it: a history
#: of the rows already written, so not read off today's default.
OUTPUT_FILLER = "bf16"


def rekeyed(old: str, candidates: Set[str],
            written: Optional[str] = None,
            record: Optional[str] = None,
            granularity: Optional[str] = None
            ) -> Tuple[Optional[str], str]:
    """`(new key, how)` when exactly one candidate is `old` with slots
    dropped (`how` ""), or, failing that, exactly one is that with the
    filler `output` replaced by `written` (`how` "output") — `written` the
    dtype the measured launch wrote whatever the cell's `output` said, None
    when it read the slot — or, failing that, exactly one is that with the
    `kv_format` the old key never had set to `record` (`how` "kv_format") —
    `record` the one cache record the measured launch could touch, None
    when its provider does not say. `granularity`, when a row's provider
    uniquely declares that missing slot, is added first without asking the
    candidates: it is a migration of the key the recorded launch used, not a
    fresh derivation of today's cell. Else `(None, reason)`."""
    was = json.loads(old)

    if granularity is not None and "activation_granularity" not in was:
        new = json.dumps({**was, "activation_granularity": granularity},
                         sort_keys=True, separators=(",", ":"))
        return new, "activation_granularity"

    if not candidates:
        return None, "no_node"

    def agree(replaced: Dict[str, str]) -> Set[str]:
        stated = {**was, **replaced}
        return {c for c in candidates
                if all(stated.get(slot) == value
                       for slot, value in json.loads(c).items())}

    how, agreeing, filled = "", agree({}), {}
    if (not agreeing and written is not None
            and was.get("output") == OUTPUT_FILLER):
        filled = {**filled, "output": written}
        how, agreeing = "output", agree(filled)
    if (not agreeing and record is not None and "kv" in was
            and "kv_format" not in was):
        how, agreeing = "kv_format", agree({**filled, "kv_format": record})
    if not agreeing:
        return None, "disagrees"
    if len(agreeing) > 1:
        return None, "ambiguous"
    return next(iter(agreeing)), how


@lru_cache(maxsize=None)
def _provider(kernel_impl: str):
    """The registered provider a row's `kernel_impl` names, or None."""
    from kernel_bench.providers import all_providers, catalog_kernel
    name = catalog_kernel(kernel_impl)
    return next((p for p in all_providers() if p.kernel_impl == name), None)


def written_by(kernel_impl: str, old: str) -> Optional[str]:
    """The dtype the launch a row timed wrote without reading the cell's
    `output` slot, or None when it read it — its provider declares an
    output (`KernelSignature.output`) — or it is no registered kernel. A
    provider that declares no output writes the cell's activation
    (`KernelSignature.writes`; registration makes a kernel that computes in
    another dtype declare what it writes)."""
    provider = _provider(kernel_impl)
    if provider is None or provider.serves is None or provider.serves.output:
        return None
    return provider.serves.writes(json.loads(old).get("activation"))[0]


def record_of(kernel_impl: str) -> Optional[str]:
    """The one cache record the launch a row timed could read or write
    beside its output, whatever the cell said: its provider declares exactly
    one (`KernelSignature.kv_format`). None for a provider that declares
    none or several — it chose by the cell's slot, or never read it — and
    for no registered kernel."""
    provider = _provider(kernel_impl)
    if provider is None or provider.serves is None:
        return None
    declared = provider.serves.kv_format
    return declared[0] if len(declared) == 1 else None


def granularity_of(kernel_impl: str, old: str) -> Optional[str]:
    """The activation granularity the launch a row timed used, when its
    provider's declaration says that unambiguously and the old key left the
    slot unsaid; None when no such migration is licensed.

    Selection is by the row's recorded implementation (`catalog_kernel`
    resolves an `impl_id` to its provider): that column says which launch
    timed the row. Signature+library would be wider — all DeepGEMM GEMMs,
    say — and would claim rows measured by another implementation of the
    same op. The VALUE still comes only from that provider's `serves`
    declaration; this function never names a granularity itself.
    """
    provider = _provider(kernel_impl)
    if provider is None or provider.serves is None:
        return None
    was = json.loads(old)
    if ("activation_granularity" in was
            or was.get("activation") not in provider.serves.activation
            or was.get("weight") not in provider.serves.weight):
        return None
    declared = provider.serves.activation_granularity
    if not declared:
        return None
    if len(declared) != 1:
        what = ", ".join(declared)
        raise ValueError(
            f"ambiguous activation_granularity declaration for "
            f"{provider.kernel_impl}: serves {what}; cannot re-key "
            f"{kernel_impl}")
    return declared[0]


def plan_rekey(groups: List[dict], workers: int) -> Tuple[List[Tuple[dict, str]], Counter, Dict[str, str]]:
    """`(rewrites, counts, models refused)`: each old group (its distinct
    `model, gpu, phase, node_spec_id, kernel_impl, dtype_env_json` and row
    count) with its new key where it has one."""
    granularities = [granularity_of(g["kernel_impl"], g["dtype_env_json"])
                     for g in groups]
    by_model: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for g, granularity in zip(groups, granularities, strict=True):
        if granularity is None:
            by_model[(g["model"], g["gpu"])].add(
                json.loads(g["dtype_env_json"]).get("weight", ""))
    tasks = [(m, gpu, tuple(sorted(w - {""}))) for (m, gpu), w in sorted(by_model.items())]
    replays: Dict[Tuple[str, str], Optional[List[NodeKeys]]] = {}
    refused: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for model, gpu, per_profile, why in pool.map(replay, tasks):
            replays[(model, gpu)] = per_profile
            if per_profile is None:
                refused[f"{model} @ {gpu}"] = why
            print(f"  replayed {model} @ {gpu}" + (f": {why}" if why else ""), flush=True)
    rewrites: List[Tuple[dict, str]] = []
    counts: Counter = Counter()
    for g, granularity in zip(groups, granularities, strict=True):
        if granularity is not None:
            # A vocabulary migration is licensed by the recorded launch, so
            # it does not need (and cannot be blocked by) today's plan.
            new, how = rekeyed(g["dtype_env_json"], set(),
                               granularity=granularity)
        else:
            per_profile = replays[(g["model"], g["gpu"])]
            if per_profile is None:
                counts["no_model"] += g["n"]
                continue
            node = (g["phase"], g["node_spec_id"])
            candidates = set().union(
                *(keys.get(node, set()) for keys in per_profile))
            new, how = rekeyed(
                g["dtype_env_json"], candidates,
                written_by(g["kernel_impl"], g["dtype_env_json"]),
                record_of(g["kernel_impl"]), granularity=granularity)
        if new is None:
            counts[how] += g["n"]
        elif new == g["dtype_env_json"]:
            counts["already"] += g["n"]
        else:
            counts[f"rekeyed_{how}" if how else "rekeyed"] += g["n"]
            rewrites.append((g, new))
    return rewrites, counts, refused


def apply(lake, rewrites: List[Tuple[dict, str]]) -> int:
    """Rewrite each group's `dtype_env_json` and its `env` in one lake
    transaction; returns the rows changed."""
    from backend.service.bench.store import env_key
    if not rewrites:
        return 0
    lake.require_free("re-keying dtype environments")
    cols = ("model", "gpu", "phase", "node_spec_id", "kernel_impl", "dtype_env_json")
    values = [[g[c] for c in cols] + [new] for g, new in rewrites]

    def run(cur) -> int:
        cur.execute("CREATE OR REPLACE TEMP TABLE rekey (model VARCHAR, gpu VARCHAR, "
                    "phase VARCHAR, node_spec_id VARCHAR, kernel_impl VARCHAR, old VARCHAR, "
                    "new VARCHAR)")
        cur.executemany("INSERT INTO rekey VALUES (?, ?, ?, ?, ?, ?, ?)", values)
        n = cur.execute(
            "UPDATE lake.measurements AS m SET dtype_env_json = r.new FROM rekey r "
            "WHERE m.model = r.model AND m.gpu = r.gpu AND m.phase = r.phase "
            "AND m.node_spec_id = r.node_spec_id AND m.kernel_impl = r.kernel_impl "
            "AND m.dtype_env_json = r.old").fetchone()[0]
        cur.execute(f"UPDATE lake.measurements SET env = {env_key('measurements')} "
                    f"WHERE env <> {env_key('measurements')}")
        return n
    return lake.transact(run)


def main(argv: Optional[List[str]] = None) -> None:
    from backend.service.bench.lake import Lake, LakeConfig
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="write the rewrites planned to this JSON file")
    ap.add_argument("--apply", metavar="REWRITES",
                    help="apply a planned JSON file (the Kernel Bench service stopped)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", action="append", help="only these models (repeatable)")
    args = ap.parse_args(argv)
    config = LakeConfig.from_env()
    if args.apply:
        # One UPDATE over the whole lake: more than the service's budget.
        config = replace(config, memory_limit="4GB")
    lake = Lake.open(config)
    try:
        if args.apply:
            with open(args.apply) as f:
                rewrites = [(g, new) for g, new in json.load(f)]
            print(f"applied: {apply(lake, rewrites)} rows")
            return
        where, params = "", []
        if args.model:
            where, params = "WHERE model IN (SELECT unnest(?::VARCHAR[]))", [args.model]
        groups = lake.read(
            "SELECT model, gpu, phase, node_spec_id, kernel_impl, dtype_env_json, "
            "count(*) AS n "
            f"FROM lake.measurements {where} GROUP BY ALL", params)
    finally:
        lake.close()
    print(f"{sum(g['n'] for g in groups)} rows in {len(groups)} groups", flush=True)
    rewrites, counts, refused = plan_rekey(groups, args.workers)
    for why, n in counts.most_common():
        print(f"{why:>10}: {n} rows")
    for model, why in sorted(refused.items()):
        print(f"  not resolved: {model}: {why}")
    by_model: Counter = Counter()
    for g, _new in rewrites:
        by_model[g["model"]] += g["n"]
    for model, n in by_model.most_common():
        print(f"  {model}: {n} rows re-keyed")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rewrites, f)
        print(f"wrote {len(rewrites)} rewrites to {args.out}")


if __name__ == "__main__":
    main()
