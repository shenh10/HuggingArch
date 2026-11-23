"""Replay the analysis on the product's deployments — validate's last gate.

`valid: true` means the spec analyses on the deployments the product offers
for it. This module is that check: it picks the deployments, runs the
product's own analysis entry point (`estimate_capacity`, prefill and decode)
on each, and reports every refusal with its node through the validator's
finding shape (`backend.arch.validate._analysable`). It reaches the arch
orchestrator as an injected gate (`replay_gate`), because arch does not
import inference.

The deployment set is one deployment per parallelism type at its highest
legal degree up to `MAX_DEGREE` — divisibility and padding fail first at the
highest power of two — chosen by the product's own legality code: the
scaling ladders as the layout solves them (`legal_parallel_degrees`), the gallery presets, the CP
legality solver, and the layersplit cache predicate below. Pipeline
parallelism is not replayed: a spec declares nothing about it, every input
the walk binds is resolved per occurrence at `pp = 1`
(`Analysis.input_references`),
and the rest is placement arithmetic, so a spec that analyses at `pp = 1`
and not at `pp > 1` is an engine defect, not a spec error.

The replay is the engine's own analysis (`THEORY`): no framework binding
chooses a deployment, rewrites the graph or refuses one. Whether a
framework runs a model class at some parallelism is a fact about that
framework, which an estimate naming it reports; it says nothing about
whether the spec analyses.

Two cost controls leave the meaning of `valid` unchanged. The orchestrator
runs this gate only when every other gate passed. The result is memoised on
disk by everything it depends on — the resolved spec, the config, the
engine, a pending op proposal and the replay constants — so re-validating an
unchanged spec reads it back, and any edit misses. The deployments run in a
process pool that holds a host-wide grant of at least `MIN_WORKERS` workers.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.measure import pool
from backend.arch.eval.program import Program
from backend.arch.eval.refusal import (
    LayoutRefusal, refusal_stage, stage_context,
)
from backend.arch.eval.transforms import Deployment
from backend.arch.ops_registry import PENDING_OP_PATCH_ENV
from backend.arch.validate._analysable import Refusal, grouped_findings
from backend.arch.validate._orchestrator import GateOutcome
from backend.code_trace.provider import Trace, provider, recording
from backend.fingerprint import data_version

from .capacity import estimate_capacity
from .framework_bindings import THEORY
from .parallel_config import CpScheme, ParallelConfig
from .parallel_sweep import (
    SweepCaps,
    config_for_degree,
    layout_verdict,
    legal_parallel_degrees,
)
from .sol_closure import check_closure
from .sweep_grid import AUDIT_PARALLEL

logger = logging.getLogger(__name__)

#: The highest degree replayed per parallelism type.
MAX_DEGREE = 8


@dataclass(frozen=True)
class ReplayWorkload:
    """The workload every deployment is replayed at: the SOL surface's."""
    gpu_sku: str = "H100-SXM"
    #: Every model as it ships (`precision` request form).
    precision: Dict[str, str] = field(
        default_factory=lambda: {"preset": "as_shipped"})
    avg_input_tokens: int = 4096
    avg_output_tokens: int = 1024
    selection_mode: str = "theoretical_only"
    #: No framework binding: the replay judges the spec, not a framework.
    framework: str = THEORY

    def estimate_kwargs(self) -> dict:
        return asdict(self)


REPLAY_WORKLOAD = ReplayWorkload()
#: The replay waits for at least this many pool workers rather than running
#: serially, which would put a long spec's 40-iteration loop past the
#: generate budget; it uses up to `MAX_WORKERS` when the host has them.
MIN_WORKERS = 4
MAX_WORKERS = 8

class ReplayCapacityTimeout(TimeoutError):
    """No worker grant within the wait budget. Infrastructure: the spec was
    never judged."""


@dataclass(frozen=True)
class ReplayDeployment:
    label: str
    parallel: ParallelConfig


# -- which deployments ----------------------------------------------------

def layersplit_refusal(analysis: Any) -> Optional[str]:
    """Why layersplit cannot place this model's cache, or None when it can:
    the refusal `CacheView.on_layer_split` raises, over the model's walk
    (`analysis`, one rank). The Inference page asks it through
    `/inference/legal-cp`."""
    return cache_view_from_analysis(analysis).layer_split_refusal()


def _top(ladder: Sequence[Tuple[int, ParallelConfig]]
         ) -> Optional[Tuple[int, ParallelConfig]]:
    legal = [(x, cfg) for x, cfg in ladder
             if x <= MAX_DEGREE and not cfg.validate()]
    return max(legal, key=lambda pair: pair[0]) if legal else None


def _top_deployable(resolved: Any, ladder: Sequence[Tuple[int, ParallelConfig]]
                    ) -> Optional[Tuple[int, ParallelConfig]]:
    """The highest rung of `ladder` the layout deploys (`layout_verdict`):
    which degrees deploy is the layout's answer, not a cap on heads — it
    refuses a degree a head solve would allow and accepts, with copies, one
    a head solve would refuse."""
    deployable = []
    for x, cfg in ladder:
        try:
            layout_verdict(resolved, cfg)
        except LayoutRefusal:
            continue
        deployable.append((x, cfg))
    return _top(deployable)


def _top_legal(resolved: Any, mode: str, caps: Any
               ) -> Optional[Tuple[int, ParallelConfig]]:
    """The highest degree of a scaling mode up to `MAX_DEGREE` the layout
    accepts (`legal_parallel_degrees`)."""
    accepted = legal_parallel_degrees(
        resolved, mode, caps, x_max=MAX_DEGREE)["accepted"]
    return _top([(rung["x"], ParallelConfig.from_dict(rung["parallel"]))
                 for rung in accepted])


def product_deployments(resolved: Any) -> Tuple[ReplayDeployment, ...]:
    """One deployment per parallelism type at its highest legal degree:
    one GPU, TP, EP with DP-attention, CP under each scheme the cache allows,
    and PP 2."""
    walked = Deployment(Program.of(resolved)).analyze()
    caps = SweepCaps.of(walked)
    moe = caps.moe_enabled
    out = [ReplayDeployment("single", ParallelConfig())]

    top = _top_legal(resolved, "tp_tp", caps)
    if top is not None:
        out.append(ReplayDeployment(f"tp{top[0]}", top[1]))

    if moe:
        top = _top_legal(resolved, "dp_ep", caps)
        if top is not None:
            out.append(ReplayDeployment(f"ep{top[0]}+dp{top[0]}", top[1]))
    else:
        out.append(ReplayDeployment("audit ep8+dp8",
                                    ParallelConfig(**AUDIT_PARALLEL)))

    layersplit = layersplit_refusal(walked) is None
    for scheme in (s.value for s in CpScheme):
        if scheme == CpScheme.LAYERSPLIT.value and not layersplit:
            continue
        if moe:
            ladder = [(x, replace(config_for_degree("cp_ep", x),
                                  cp_scheme=scheme))
                      for x in (2, 4, 8) if caps.ep_divides(x)]
        else:
            ladder = [(x, ParallelConfig(cp=x, cp_scheme=scheme))
                      for x in (2, 4, 8)]
        top = _top_deployable(resolved, ladder)
        if top is not None:
            out.append(ReplayDeployment(f"cp{top[0]} {scheme}", top[1]))

    return tuple(out)


def _describe(parallel: ParallelConfig) -> str:
    """The fields a deployment sets away from one GPU, as `name=value`."""
    base = asdict(ParallelConfig())
    parts = [f"{name}={value}" for name, value in asdict(parallel).items()
             if value != base[name]]
    return ", ".join(parts) or "one GPU"


# -- one deployment ---------------------------------------------------------

def replay_one(config: dict, resolved: Any,
               deployment: ReplayDeployment) -> Optional[Refusal]:
    """Run the analysis entry point on one deployment: None when it
    succeeds, its refusal otherwise. Infrastructure raises."""
    try:
        with stage_context("analysis"):
            result = estimate_capacity(
                config=config, resolved=resolved,
                num_gpus=deployment.parallel.cluster_size,
                parallel=deployment.parallel,
                **REPLAY_WORKLOAD.estimate_kwargs())
            # A result is valid only when it closes, as the SOL surface
            # requires.
            with stage_context("closure"):
                check_closure(result)
    except (OSError, ImportError, MemoryError):
        raise
    except Exception as exc:  # every other raise is the analysis refusing
        return Refusal.of(exc, deployment=deployment.label,
                          stage=refusal_stage(exc))
    return None


# The job a forked worker reads: set before the pool forks, so the resolved
# spec is inherited rather than pickled.
_JOB: Optional[Tuple[dict, Any, Tuple[ReplayDeployment, ...]]] = None


def _worker(index: int) -> Tuple[Optional[Refusal], Trace]:
    """One deployment in a pool worker, and the trace of what it ran there:
    the worker's recording is not the parent's, so it ships back."""
    config, resolved, deployments = _JOB
    with recording() as rec:
        outcome = replay_one(config, resolved, deployments[index])
    return outcome, rec["trace"]


# -- host-wide worker grant -------------------------------------------------

def _replay_dir() -> Path:
    root = os.environ.get("HA_VALIDATE_REPLAY_DIR") or os.path.join(
        tempfile.gettempdir(), "huggingarch-validate-replay")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _host_slots() -> int:
    raw = os.environ.get("HA_VALIDATE_REPLAY_SLOTS")
    return int(raw) if raw else max(MIN_WORKERS, pool.usable_cpus() - 2)


def _locked(handle: Any) -> bool:
    """Take an exclusive lock on one slot file without waiting."""
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


@contextmanager
def worker_grant(wanted: int) -> Iterator[int]:
    """Hold between `MIN_WORKERS` and `wanted` of the host's replay slots
    (one `fcntl` lock file per worker), waiting up to
    `HA_VALIDATE_REPLAY_WAIT_SECONDS` for the minimum. Yields the count."""
    slots_dir = _replay_dir() / "slots"
    slots_dir.mkdir(exist_ok=True)
    total = _host_slots()
    need = min(MIN_WORKERS, wanted, total)
    deadline = time.monotonic() + float(
        os.environ.get("HA_VALIDATE_REPLAY_WAIT_SECONDS") or 300)
    waited = False
    while True:
        with ExitStack() as held:
            count = 0
            for i in range(total):
                if count >= wanted:
                    break
                handle = held.enter_context(open(slots_dir / f"slot_{i}", "a"))
                if _locked(handle):
                    count += 1
            if count >= need:
                if waited:
                    logger.info("validate replay: granted %d workers", count)
                yield count
                return
        if time.monotonic() > deadline:
            raise ReplayCapacityTimeout(
                f"no grant of {need} replay workers within the wait budget "
                f"({total} host slots)")
        if not waited:
            logger.info("validate replay: waiting for %d workers", need)
            waited = True
        time.sleep(1.0)


def _pool_enabled() -> bool:
    return not (os.environ.get("HUGGINGARCH_NO_SWEEP_POOL")
                or "PYTEST_CURRENT_TEST" in os.environ)


def replay(config: dict, resolved: Any,
           deployments: Sequence[ReplayDeployment]) -> List[Refusal]:
    """Every deployment's refusal, in deployment order."""
    deployments = tuple(deployments)
    if not _pool_enabled() or len(deployments) < 2:
        outcomes = [replay_one(config, resolved, d) for d in deployments]
        return [r for r in outcomes if r is not None]
    global _JOB
    with worker_grant(min(MAX_WORKERS, len(deployments))) as workers:
        _JOB = (config, resolved, deployments)
        try:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=context) as pool:
                outcomes = []
                for outcome, trace in pool.map(_worker, range(len(deployments))):
                    provider().merge(trace)
                    outcomes.append(outcome)
                return [r for r in outcomes if r is not None]
        except BrokenProcessPool as exc:
            raise OSError(f"validate replay pool broke: {exc}") from exc
        finally:
            _JOB = None


# -- memo -------------------------------------------------------------------

def _pending_patch_hash() -> str:
    path = os.environ.get(PENDING_OP_PATCH_ENV)
    if not path:
        return ""
    patch_dir = Path(path).parent
    h = hashlib.sha256()
    for item in sorted(patch_dir.rglob("*")):
        if item.is_file():
            h.update(item.relative_to(patch_dir).as_posix().encode())
            h.update(item.read_bytes())
    return h.hexdigest()


def memo_key(config: dict, resolved: Any) -> str:
    """Everything the replay result depends on but the code: the resolved
    spec and the engine data it reads (model yaml and drafts as resolved,
    `data_version`), the config, a pending op proposal, and the replay's own
    constants. The code is the entry's trace (`replay_result`)."""
    with provider().untraced():
        version = data_version(resolved)
    parts = [
        version,
        json.dumps(config, sort_keys=True, default=str),
        _pending_patch_hash(),
        json.dumps([asdict(REPLAY_WORKLOAD), MAX_DEGREE,
                    [s.value for s in CpScheme]], sort_keys=True),
    ]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _memo_path(key: str) -> Path:
    path = _replay_dir() / "memo"
    path.mkdir(exist_ok=True)
    return path / f"{key}.json"


def _memo_read(key: str) -> Optional["ReplayResult"]:
    """The memoised result, or None to replay: no entry, one that does not
    read back as a whole result (a partial write, an older layout), or one
    whose trace the code this process runs no longer satisfies."""
    path = _memo_path(key)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        trace = Trace.from_json(data["trace"])
        result = ReplayResult.from_json(data)
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("validate replay memo %s unreadable (%s); replaying",
                       path.name, exc)
        return None
    p = provider()
    if not p.valid(trace):
        return None
    p.merge(trace)
    return result


def _memo_write(key: str, result: "ReplayResult", trace: Trace) -> None:
    path = _memo_path(key)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({**result.to_json(), "trace": trace.to_json()}),
                   encoding="utf-8")
    tmp.replace(path)


# -- the gate ---------------------------------------------------------------

@dataclass(frozen=True)
class ReplayResult:
    """The deployments replayed and their refusals."""
    deployments: Tuple[Tuple[str, str], ...]   # (label, parallel config)
    refusals: Tuple[Refusal, ...]

    def to_json(self) -> dict:
        return {"deployments": [list(d) for d in self.deployments],
                "refusals": [asdict(r) for r in self.refusals]}

    @classmethod
    def from_json(cls, data: dict) -> "ReplayResult":
        return cls(tuple(tuple(d) for d in data["deployments"]),
                   tuple(Refusal(**r) for r in data["refusals"]))

    def report(self) -> dict:
        """One row per deployment, with the causes it refused on."""
        refused: dict = {}
        for refusal in self.refusals:
            refused.setdefault(refusal.deployment, []).append(
                {"kind": refusal.kind, "node": refusal.node,
                 "stage": refusal.stage})
        return {"deployments": [
            {"label": label, "parallel": parallel,
             "refusals": refused.get(label, [])}
            for label, parallel in self.deployments]}


def replay_result(config: dict, resolved: Any) -> ReplayResult:
    """The replay over `product_deployments`, memoised."""
    key = memo_key(config, resolved)
    cached = _memo_read(key)
    if cached is not None:
        return cached
    with recording() as rec:
        deployments = product_deployments(resolved)
        result = ReplayResult(
            tuple((d.label, _describe(d.parallel)) for d in deployments),
            tuple(replay(config, resolved, deployments)))
    _memo_write(key, result, rec["trace"])
    return result


def replay_gate(resolved: Any, config: Any, model_result: Any) -> GateOutcome:
    """The orchestrator's downstream gate: the replay's refusals as
    findings, one per cause, and the per-deployment report."""
    result = replay_result(dict(config), resolved)
    return GateOutcome(grouped_findings(result.refusals), result.report())
