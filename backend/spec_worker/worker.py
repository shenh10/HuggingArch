"""Spec-worker agent execution + post-success persistence hooks.

`stream_agent` is a thin wrapper over `backend._agent_runner.stream_agent_core`.
The spec-specific concern is what happens on a successful run:

  • `_maybe_snapshot` writes the HF model_info + config + forward source
    to `arch/snapshots/<safe_id>.json`. Fast path uses data captured
    at prompt-prep time so no extra HF round-trip is needed.
  • `_publish_to_cache` writes the freshly-generated spec yaml + snapshot +
    component bundle to Supabase when configured, or to the host-local shared
    cache on an offline self-host run. A validate gate prevents poisoning the
    shared tier with structurally-broken specs.

Both hooks are best-effort: any failure is logged and swallowed so the
user-facing /generate-spec stream is not affected.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from backend import supabase_env
from backend.spec_worker.self_register import stamp_match_model_id as _stamp_match_model_id

import json
import hashlib
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Generator, Optional, Literal, Tuple

from backend._agent_runner import scrub_paths, stream_agent_core
from backend._agent_runner.runner import sse as _sse
from backend.agents.base import AgentEnd
from backend.custom_id import is_custom_id
from backend.hf.metadata import stored_without_checkpoint
from backend.spec_worker.overlay import (
    has_resumable_draft,
    is_anonymous_overlay,
    is_generate_leaf,
    land_anonymous_overlay,
    RunEnd,
)

from backend.arch.loader import SpecLoadError
from backend.fingerprint import engine_version
from backend.arch.validate._analysable import (
    ENGINE_KIND,
    engine_gap_entries,
    only_unattributed,
)

from .limits import (
    GENERATE_TIMEOUT_KNOB,
    OP_CONTINUE_SEC,
    OP_CONTINUE_TIMEOUT_KNOB,
    ROUND_BUDGET_SEC,
    ROUND_TIMEOUT_KNOB,
    VALIDATE_SEC,
    max_repair_rounds,
)
from .runtime import (
    _AGENT_HARD_CAP_SEC,
    _LOG_DIR,
    _locks_lock,
    _model_procs,
    bind_job_log,
    clear_job_log_sidecar,
    get_agent_cli,
    owns,
    release,
)
from .bundle import (
    BlockedBundle,
    BundleFile,
    FormalSource,
    NormalizedBundle,
    SpecBundleError,
    normalize_bundle,
)
from . import records, shared_spec_miss
from .outcome import RecordOutcome, RunErrorKind
from .overlay import _safe, authored_yaml, run_overlay_owner
from ._ground_truth import ForwardSourceUnavailable


logger = logging.getLogger(__name__)


def _harvest_component_yamls(
    component_dirs: list[Path],
    repo_root: Path,
    *,
    use_overlay: bool,
    started_at: float,
    declared_drafts: set[str],
) -> list[dict]:
    """Collect components that must travel with the generated model spec.

    A private retry can reuse a draft written by an earlier attempt, so mtime
    is not a valid inclusion rule there. Harvest every draft declared by the
    final model YAML. Shared-tree runs retain the freshness filter to avoid
    publishing unrelated repository components.
    """
    harvested: list[dict] = []
    for component_dir in component_dirs:
        if not component_dir.is_dir():
            continue
        for path in component_dir.glob("*.yaml"):
            try:
                if use_overlay:
                    if path.stem not in declared_drafts:
                        continue
                    relative = f"backend/arch/components/drafts/{path.name}"
                else:
                    if path.stat().st_mtime < started_at - 1:
                        continue
                    relative = path.relative_to(repo_root).as_posix()
                harvested.append({
                    "path": relative,
                    "body": path.read_text(encoding="utf-8"),
                })
            except OSError:  # one unreadable component must not abort publish
                continue
    return harvested


def _fresh_model_yaml(started_at: float, root: Optional[str] = None,
                      prefer_stem: Optional[str] = None):
    """The model yaml written by THIS run (mtime >= started_at), newest first,
    or None. Globs the per-run overlay's `models/` dir when `root` is given (a
    signed-in, isolated run), else the shared tree. Single source of truth
    shared by the publish gate AND the success-verdict check, so both agree on
    'did the agent actually produce a spec'. 1s slack absorbs clock skew."""
    if root:
        models_dir = Path(root) / "models"
    else:
        models_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "backend" / "arch" / "models"
        )
    fresh = [
        p for p in models_dir.glob("*.yaml")
        if p.is_file() and p.stat().st_mtime >= started_at - 1
    ]
    return _preferred_model_yaml(fresh, prefer_stem=prefer_stem)


def _preferred_model_yaml(fresh: list[Path], prefer_stem: Optional[str] = None):
    """Newest fresh file, or the strict target file when an identity is named.

    A generate must not mistake a fresh agent base (or a stale identity left
    by an older leaf) for its target. Passing `prefer_stem` therefore makes a
    missing target terminal even when other YAML files are fresh.
    """
    if prefer_stem is not None:
        fresh = [path for path in fresh if path.stem == prefer_stem]
    if not fresh:
        return None
    fresh.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return fresh[0]


def _existing_model_yaml(root: Optional[str]):
    """The model yaml already present in an isolated RUN overlay, regardless of
    mtime — the spec `seed_run_overlay` wrote at the start of a REFINE run. Used
    as the success-verdict/publish fallback when the agent made no edit this run:
    a refine whose seeded spec is ALREADY valid is a success (nothing to fix),
    not a 'nothing generated' failure. The gate then re-validates it, so a
    still-broken untouched spec still fails on its real errors.

    Callers gate this on the `seeded` fact: this run started from a spec it
    means to keep — a refine seeded it, or a continued generation re-opened
    the draft an unfinished run left (`resume.prepare_resume`). That is
    essential, not just the `root` check: a NEW generate reuses the
    per-(user, model, kind) leaf, so without the fact a failed re-generate
    (exit 0, no write) could pass on a spec it did not write. Only a run that
    starts from its leaf's spec may fall back here."""
    if not root:
        return None
    return _existing_model_yaml_with_stem(root)


def _existing_model_yaml_with_stem(root: Optional[str], prefer_stem: Optional[str] = None):
    if not root:
        return None
    models_dir = Path(root) / "models"
    cands = sorted(
        (p for p in models_dir.glob("*.yaml") if p.is_file()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if prefer_stem is not None:
        cands = [path for path in cands if path.stem == prefer_stem]
    return cands[0] if cands else None


def _fresh_model_names(root: Optional[str]) -> list[str]:
    models_dir = Path(root) / "models" if root else (
        Path(__file__).resolve().parent.parent.parent
        / "backend" / "arch" / "models"
    )
    return [path.name for path in sorted(models_dir.glob("*.yaml")) if path.is_file()]


def _no_output_reason(model_id: str, root: Optional[str]) -> str:
    expected = f"models/{_safe(model_id)}.yaml"
    names = ", ".join(_fresh_model_names(root)) or "none"
    return (
        "agent exited cleanly but wrote no model spec — "
        f"expected {expected}; fresh model files: {names} (see log)"
    )


@dataclass(frozen=True)
class Downgrade:
    """Why a clean agent exit is not a success. `kind` is what the repair
    loop dispatches on — only a `validation` failure hands the agent a fix
    list; `errors` is that list, unformatted. `reason` is the human line the
    runner writes into the `done` event; `error_kind` is its machine key and
    `engine_gaps` the engine refusals of a `blocked_by_engine` verdict
    (`engine_gap_entries`)."""
    kind: Literal[
        "no_output", "validation", "crash", "missing_primitive",
        "cache_warning", "blocked_by_engine",
    ]
    reason: str
    errors: Tuple[str, ...] = ()
    error_kind: Optional[RunErrorKind] = None
    engine_gaps: Tuple[dict, ...] = ()


def _yaml_map(directory: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.yaml")):
        if path.is_file():
            out[path.stem] = path.read_text(encoding="utf-8")
    return out


def _overlay_views(target_root: Optional[str]) -> tuple[dict[str, str], dict[str, str]]:
    if not target_root:
        return {}, {}
    root = Path(target_root)
    return (
        _yaml_map(root / "models"),
        _yaml_map(root / "components" / "drafts"),
    )


def _shared_tree_views(
    repo_root: Path,
    started_at: float,
    safe_id: str,
) -> tuple[dict[str, str], dict[str, str], list[dict], FormalSource]:
    """Reconnect auth-off self-host runs to the pre-Batch-3 shared-tree view.

    Batch 3 will give these runs an anonymous overlay. Until then, mtime is the
    only way to distinguish this run's shared-tree writes from the deployed
    formal layer. Fresh models/drafts become run inputs. Model names are global,
    but a fresh draft excludes only the drafts tier: namesakes in library and
    adapters stay formal so their signatures participate in conflict checks.
    """
    arch_root = repo_root / "backend" / "arch"
    models_dir = arch_root / "models"
    fresh_models = [
        path for path in models_dir.glob("*.yaml")
        if path.is_file() and path.stat().st_mtime >= started_at - 1
    ]
    run_models = {
        path.stem: path.read_text(encoding="utf-8")
        for path in fresh_models if path.stem != safe_id
    }

    component_files = _harvest_component_yamls(
        [arch_root / "components" / "library",
         arch_root / "components" / "drafts"],
        repo_root,
        use_overlay=False,
        started_at=started_at,
        declared_drafts=set(),
    )
    run_drafts = {
        Path(file["path"]).stem: file["body"]
        for file in component_files
        if file["path"].startswith(
            "backend/arch/components/drafts/")
    }

    fresh_model_stems = {path.stem for path in fresh_models}
    fresh_draft_components = {
        ("drafts", Path(file["path"]).stem)
        for file in component_files
        if file["path"].startswith(
            "backend/arch/components/drafts/")
    }
    # Build the formal view after excluding this run's names. Constructing the
    # unfiltered deployment source first would sign a malformed fresh draft and
    # let that parser error bypass the SpecBundleError boundary below.
    formal_source = FormalSource.from_deployment(
        root=arch_root,
        exclude_models=fresh_model_stems,
        exclude_components=fresh_draft_components,
    )
    return (
        run_models,
        run_drafts,
        component_files,
        formal_source,
    )


def _normalize_shared_tree_bundle(
    model_id: str,
    spec_yaml: str,
    started_at: float,
    *,
    extra_files: Optional[list] = None,
) -> NormalizedBundle:
    """Normalize an auth-off run against its fresh shared-tree provenance."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    run_models, run_drafts, component_files, formal_source = (
        _shared_tree_views(repo_root, started_at, _safe(model_id)))
    return normalize_bundle(
        model_id,
        spec_yaml,
        [*component_files, *(extra_files or [])],
        formal_source,
        run_models=run_models,
        run_drafts=run_drafts,
    )


def _bundle_failure(exc: SpecBundleError) -> Downgrade:
    detail = "; ".join(exc.errors)
    return Downgrade(
        "validation",
        f"generated spec bundle is incomplete: {detail}",
        tuple(exc.errors),
        RunErrorKind.SPEC_BUNDLE_INCOMPLETE,
    )


def _load_failure(exc: SpecLoadError) -> Downgrade:
    """Migrate blockers / schema load errors are repairable validation findings."""
    detail = str(exc)
    return Downgrade(
        "validation",
        f"generated spec failed validation: {detail}",
        (detail,),
        RunErrorKind.SPEC_INVALID,
    )


def _harvest_formal_proposals(
    target_root: Optional[str],
    seed_bundle: Optional[NormalizedBundle | BlockedBundle] = None,
) -> tuple[list[BundleFile], list[str]]:
    """Round-trip PR-only formal proposals from the lazy authoring area."""
    if not target_root:
        return [], []
    root = Path(target_root)
    seeded = {
        file.path: file for file in (seed_bundle.proposal_files if seed_bundle else ())
    }
    harvested: list[BundleFile] = []
    notes: list[str] = []
    seen: set[str] = set()
    reference_root = Path(__file__).resolve().parent.parent.parent / "backend" / "arch"
    for tier in ("library", "adapters"):
        directory = root / "components" / "formal_proposals" / tier
        for path in sorted(directory.glob("*.yaml")) if directory.is_dir() else []:
            intended = f"backend/arch/components/{tier}/{path.name}"
            seen.add(intended)
            body = path.read_text(encoding="utf-8")
            old = seeded.get(intended)
            if old is not None:
                harvested.append(BundleFile(
                    path=intended, body=body,
                    formal_body_sha256=old.formal_body_sha256))
                continue
            formal = reference_root / "components" / tier / path.name
            if not formal.is_file():
                notes.append(
                    f"formal proposal omitted: no current {tier} component "
                    f"named {path.stem}")
                continue
            baseline = hashlib.sha256(
                formal.read_bytes()).hexdigest()
            harvested.append(BundleFile(
                path=intended, body=body, formal_body_sha256=baseline))
    notes.extend(
        f"formal proposal withdrawn: {intended}"
        for intended in sorted(set(seeded) - seen)
    )
    return harvested, notes


def _normalize_overlay_bundle(
    model_id: str,
    target_root: str,
    *,
    seed_bundle: Optional[NormalizedBundle | BlockedBundle] = None,
    extra_files: Optional[list] = None,
) -> Optional[NormalizedBundle]:
    safe_id = _safe(model_id)
    identity = Path(target_root) / "models" / f"{safe_id}.yaml"
    if not identity.is_file():
        return None
    spec_yaml = identity.read_text(encoding="utf-8")
    proposals, proposal_notes = _harvest_formal_proposals(
        target_root, seed_bundle=seed_bundle)
    files = [
        *(seed_bundle.record_files(include_proposals=False)
          if seed_bundle else []),
        *proposals,
        *(extra_files or []),
    ]
    run_models, run_drafts = _overlay_views(target_root)
    bundle = normalize_bundle(
        model_id, spec_yaml, files, FormalSource.from_deployment(),
        run_models=run_models, run_drafts=run_drafts,
    )
    if not proposal_notes:
        return bundle
    return replace(bundle, notes=tuple(dict.fromkeys((
        *bundle.notes, *proposal_notes))))


def _is_commit_sha(text: str) -> bool:
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def _code_revision() -> str:
    """Git SHA of the checkout the agent and the verdict gate both import.

    The image has no `.git`; the deploy passes the commit as the build arg
    `HA_GIT_SHA`. A value that is not a full 40-hex commit (an empty build
    arg, a short or mangled SHA) names no commit, and the answer is
    `unknown` rather than a guess."""
    sha = os.environ.get("HA_GIT_SHA") or os.environ.get("GIT_SHA")
    if sha:
        sha = sha.strip().lower()
        return sha if _is_commit_sha(sha) else "unknown"
    try:
        root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=2,
            check=False,
        )
        if proc.returncode == 0 and _is_commit_sha(proc.stdout.strip()):
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("git rev-parse failed: %s", exc)
    return "unknown"


@dataclass(frozen=True)
class RunRevision:
    """The checkout a run started on, and the one its verdict gate validated
    against.

    The backend serves from a git checkout that keeps taking commits while
    runs are in flight. The code a run started on is gone from disk by then,
    so the gate validates against the checkout as it is; the agent's own
    in-loop validate imports the same tree. A moved checkout does not void
    the run: the verdict stands and carries both revisions, plus the files
    that changed in between, so a reader can judge whether it mattered.
    Validator changes that affect the spec surface as ordinary validate
    errors, which the repair loop can act on."""
    started: str
    judged: str

    @classmethod
    def since(cls, started: Optional[str]) -> "RunRevision":
        """Read HEAD now, right before the gate's validate starts."""
        judged = _code_revision()
        return cls(started or judged, judged)

    @property
    def moved(self) -> bool:
        return self.started != self.judged

    def changed_paths(self) -> tuple[str, ...]:
        """Files changed between the two revisions; () if git cannot say."""
        if not self.moved:
            return ()
        try:
            proc = subprocess.run(
                ["git", "diff", "--name-only", self.started, self.judged],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("git diff failed: %s", exc)
            return ()
        if proc.returncode != 0:
            return ()
        return tuple(line for line in proc.stdout.splitlines() if line.strip())

    def stamp(self, max_paths: int = 20) -> str:
        if not self.moved:
            return f"code: {self.judged[:12]}"
        text = (f"code: run started at {self.started[:12]}, "
                f"validated at {self.judged[:12]} "
                "(the checkout moved during the run")
        paths = self.changed_paths()
        if paths:
            shown = ", ".join(paths[:max_paths])
            more = f", +{len(paths) - max_paths} more" if len(paths) > max_paths else ""
            noun = "file" if len(paths) == 1 else "files"
            text += f"; {len(paths)} {noun} changed: {shown}{more}"
        return text + ")"


def blocked_reason(gaps) -> str:
    """The `done` message of a run the engine blocks: the capabilities it
    lacks, then where each was refused. Display text; the `engine_gaps`
    entries it is written from are the machine form."""
    capabilities = ", ".join(dict.fromkeys(g["capability"] for g in gaps))
    where = "; ".join(
        f"{g['capability']} at `{g['node'] or 'the model'}` "
        f"({', '.join(g['deployments'])}; {', '.join(g['stages'])})"
        for g in gaps)
    return (f"The analysis engine does not yet support: {capabilities}. The "
            "spec states these correctly; it is blocked until the engine "
            f"supports them. Refused: {where}")[:4000]


#: Appended to a blocked run's message when its record did not save.
BLOCKED_NOT_SAVED = ("The blocked spec could not be saved to your "
                     "generations; it remains in this run's workspace.")


def _run_snapshot_json(model_id: str, target_root: Optional[str]) -> dict:
    """The snapshot this run wrote (overlay or shared tree), or {}."""
    from backend.arch.snapshots import snapshot_path
    snap_path = snapshot_path(model_id, target_root)
    snapshot_json: dict = {}
    if snap_path.is_file():
        try:
            snapshot_json = json.loads(snap_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            snapshot_json = {}
    return snapshot_json


def _record_snapshot(model_id: str, target_root: Optional[str],
                     snapshot_fields: Optional[dict] = None) -> dict:
    """The snapshot a run's record keeps: the one the run wrote, and what
    its caller keeps beside it (`snapshot_fields`: a described model's
    description, so a later turn reads it from the record, not the
    browser). A model stored without a checkpoint (written from a
    description) authored its snapshot in the agent's workspace: of it the
    record keeps the config alone — everything else a snapshot holds is a
    fact about a checkpoint or its source."""
    snapshot_json = _run_snapshot_json(model_id, target_root)
    if stored_without_checkpoint(model_id):
        snapshot_json = {"model_id": model_id,
                         "config": snapshot_json.get("config") or {}}
    return {**snapshot_json, **(snapshot_fields or {})}


def _record_bundle(model_id: str, target_root: Optional[str],
                   started_at: float, *,
                   seed_bundle: Optional[NormalizedBundle | BlockedBundle] = None,
                   extra_files: Optional[list] = None,
                   spec_yaml: Optional[str] = None) -> tuple[str, list]:
    """The bundle a generation record stores: the run's spec normalized with
    its seed bundle and the op-registry files it carries, stamped with its
    match id, and the files `record_files()` keeps. Raises the
    normalization's `SpecBundleError` / `SpecLoadError`. A shared-tree run
    (no `target_root`) normalizes `spec_yaml` against its fresh provenance."""
    if target_root:
        bundle = _normalize_overlay_bundle(
            model_id, target_root, seed_bundle=seed_bundle,
            extra_files=extra_files)
        if bundle is None:
            raise SpecBundleError(
                f"no model spec at models/{_safe(model_id)}.yaml")
    else:
        bundle = _normalize_shared_tree_bundle(
            model_id, spec_yaml or "", started_at, extra_files=extra_files)
    return (_stamp_match_model_id(bundle.spec_yaml, model_id),
            bundle.record_files())


def blocked_records_enabled() -> bool:
    """Whether this deployment keeps engine-blocked specs as pending records
    (`HA_RECORD_BLOCKED_SPECS=1`). Off until every instance runs a backend
    that knows pending records: an older one serves the newest record that
    is not `rejected`, a blocked one included (supabase/README.md, the 0002
    rollout)."""
    return os.environ.get("HA_RECORD_BLOCKED_SPECS", "").strip().lower() in (
        "1", "true", "yes", "on")


def _record_blocked(model_id: str, log_file: IO, started_at: float,
                    prep_result: dict, downgrade: Downgrade) -> dict:
    """Keep a blocked run's spec as the user's pending record
    (`spec_generations.record_pending`), never in a shared cache. Returns
    the `done.record` value; an anonymous or single-user run keeps none."""
    user_id = prep_result.get("user_id")
    target_root = prep_result.get("target_root")
    if not (user_id and target_root and supabase_env.configured()):
        return {"outcome": RecordOutcome.NONE.value}
    if not blocked_records_enabled():
        outcome = {"outcome": RecordOutcome.FAILED.value,
                   "reason": "this deployment does not keep blocked specs yet "
                             "(HA_RECORD_BLOCKED_SPECS)"}
        log_file.write(f"[spec-record] blocked_by_engine {outcome}\n")
        log_file.flush()
        return outcome
    try:
        spec_yaml, files = _record_bundle(
            model_id, target_root, started_at,
            seed_bundle=prep_result.get("seed_bundle"),
            extra_files=prep_result.get("op_registry_files") or [])
        agent_cli = prep_result.get("agent_cli")
        outcome = records.record_pending(
            model_id=model_id, user_id=user_id, spec_yaml=spec_yaml,
            snapshot_json=_record_snapshot(
                model_id, target_root, prep_result.get("snapshot_fields")),
            new_component_yamls=files, agent_cli=agent_cli,
            agent_model=_run_agent_model(prep_result, agent_cli),
            engine_gaps=list(downgrade.engine_gaps),
            engine_version=engine_version(),
            run_started_at=prep_result.get("run_started_at"),
        )
    except (SpecBundleError, SpecLoadError) as exc:
        outcome = {"outcome": RecordOutcome.FAILED.value,
                   "reason": f"bundle normalization failed: {exc}"[:500]}
    except Exception as exc:  # storage must not break the run's verdict
        logger.warning("blocked record failed for %s", model_id, exc_info=True)
        outcome = {"outcome": RecordOutcome.FAILED.value,
                   "reason": f"{exc.__class__.__name__}: {exc}"[:500]}
    log_file.write(f"[spec-record] blocked_by_engine {outcome}\n")
    log_file.flush()
    return outcome


def _invoke_validate(model_id: str, spec_stem: Optional[str],
                     overlay_dir: Optional[str]) -> dict:
    """The same CLI entry the agent is allowed to run.

    The sandboxed agent does `python -m backend.inference.validate` against the
    checkout on disk (`PYTHONPATH` = repo root). The long-lived backend
    process keeps whatever it imported at boot. A fresh interpreter for
    the verdict gate is the same package the agent runs, without a
    second validate implementation. Callers stamp the result with a
    `RunRevision`.
    """
    root = str(Path(__file__).resolve().parents[2])
    cmd = [sys.executable, "-m", "backend.inference.validate", model_id]
    if spec_stem:
        cmd.extend(["--spec", spec_stem])
    if overlay_dir:
        cmd.extend(["--overlay-dir", overlay_dir])
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not existing else root + os.pathsep + existing
    if overlay_dir:
        env["HA_SPEC_OVERLAY_DIR"] = overlay_dir
    proc = subprocess.run(
        cmd, cwd=root, capture_output=True, text=True,
        timeout=VALIDATE_SEC, env=env, check=False,
    )
    stdout = (proc.stdout or "").strip()
    if not stdout:
        err = (proc.stderr or f"validate exited {proc.returncode}").strip()
        raise RuntimeError(err[-500:])
    return json.loads(stdout)


def _verify_spec_output(model_id: str, log_file: IO, started_at: float,
                        root: Optional[str] = None, *, seeded: bool = False,
                        code_revision: Optional[str] = None
                        ) -> tuple[Optional[Downgrade], Optional[dict]]:
    """Verdict gate for a clean agent exit. The agent CLI reports success on a
    plain exit-0 — but a run that wrote NO fresh model yaml did not actually
    generate anything (e.g. the codex Bubblewrap sandbox fails to launch, so
    every command errors and the agent exits 0 having written nothing). Return
    a human-readable downgrade reason in that case so the runner rewrites the
    `done` event to success=False. A produced YAML must also clear the publish
    validator's hard errors; diagnostic failures are not a successful SpecGen.
    Checks the per-run overlay when `root` is given.

    Returns (downgrade, validate_result): the downgrade is None on success,
    and the validate result dict travels with it so `_publish_to_cache` can
    consume the same verdict instead of re-running validate (which re-fetches
    safetensors metadata). Spec-eval failures (a graph that does not
    compose, a dim that does not resolve) arrive as `errors[]` and are a
    repairable `validation` downgrade. A raise out of `validate` is
    infrastructure (config / Hub), so the result dict is only non-None when
    validation completed."""
    def _fail(downgrade: Downgrade) -> tuple[Downgrade, None]:
        log_file.write(f"[verdict] DOWNGRADE success -> failure: {downgrade.reason}\n")
        log_file.flush()
        return downgrade, None

    safe_id = _safe(model_id)
    model_yaml = _fresh_model_yaml(started_at, root, prefer_stem=safe_id)
    if model_yaml is None:
        # No fresh write. An op-gap the agent recorded takes precedence — it is
        # a real, actionable signal (chain the op writer), not an empty run.
        from ._op_chain import find_fresh_gap
        gap = find_fresh_gap(model_id, started_at)
        if gap:
            return _fail(Downgrade(
                "missing_primitive",
                f"missing primitive {gap['op_name']!r} — chaining op writer",
                errors=(json.dumps(gap),),
            ))
        # No fresh write and no gap. Only a run that started from its leaf's
        # spec — a refine's seed, or a continued generation's draft — may fall
        # back to it: an agent that correctly made NO edit (the spec is
        # already valid) leaves no fresh yaml — judge it on the spec that
        # exists, not as 'nothing generated'. A NEW generate does not set
        # `seeded`, so it still reports no_output on a leaf it did not write.
        model_yaml = (_existing_model_yaml_with_stem(root, prefer_stem=safe_id)
                      if seeded else None)
        if model_yaml is None:
            return _fail(Downgrade(
                "no_output",
                _no_output_reason(model_id, root)))
    if root:
        try:
            _normalize_overlay_bundle(model_id, root)
        except SpecBundleError as exc:
            return _fail(_bundle_failure(exc))
        except SpecLoadError as exc:
            return _fail(_load_failure(exc))
    else:
        try:
            _normalize_shared_tree_bundle(
                model_id, model_yaml.read_text(encoding="utf-8"), started_at)
        except SpecBundleError as exc:
            return _fail(_bundle_failure(exc))
        except SpecLoadError as exc:
            return _fail(_load_failure(exc))
    try:
        revision = RunRevision.since(code_revision)
        log_file.write(f"[verdict] {revision.stamp()}\n")
        log_file.flush()
        result = _invoke_validate(model_id, model_yaml.stem, root)
        errors = tuple(str(error) for error in (result.get("errors") or []))
        if not errors:
            return None, result
        if result.get("blocked_by_engine"):
            # Every error is an analysis engine refusal: no spec edit
            # resolves it, so the run stops here instead of repairing.
            gaps = tuple(engine_gap_entries(result))
            return _fail(Downgrade(
                "blocked_by_engine", blocked_reason(gaps), errors,
                error_kind=RunErrorKind.BLOCKED_BY_ENGINE, engine_gaps=gaps))
        # The repair loop gets what the spec can fix; engine refusals stay
        # in the reason for the user.
        engine = {r.get("detail") for r in result.get("analysis_refusals") or []
                  if r.get("kind") == ENGINE_KIND}
        repairable = tuple(e for e in errors if e not in engine)
        # The full error list (capped) travels in the reason: a final failure
        # shows the user everything, not just 3 lines. The repair loop reads
        # `errors` itself.
        return _fail(Downgrade(
            "validation",
            "generated spec failed validation: " + "; ".join(errors)[:4000],
            repairable,
            error_kind=(RunErrorKind.ANALYSIS_REFUSAL_UNATTRIBUTED
                        if only_unattributed(result) else None)))
    except Exception as exc:
        return _fail(Downgrade(
            "crash",
            "generated spec validation crashed: "
            f"{exc.__class__.__name__}: {str(exc)[:200]}"))


def _publish_to_cache(
    model_id: str,
    log_file,
    started_at: float,
    user_id: Optional[str] = None,
    agent_cli: Optional[str] = None,
    agent_model: Optional[str] = None,
    target_root: Optional[str] = None,
    val_result: Optional[dict] = None,
    extra_files: Optional[list] = None,
    seeded: bool = False,
    seed_bundle: Optional[NormalizedBundle] = None,
    code_revision: Optional[str] = None,
    run_started_at: Optional[str] = None,
    record_result: Optional[dict] = None,
    snapshot_fields: Optional[dict] = None,
) -> Optional[Downgrade]:
    """Write-through to Supabase `cached_specs` so other VMs can skip
    the agent for this model_id.

    When `user_id` is set (a signed-in user triggered this run via the
    service-layer /api/spec/generate route), also append a per-user
    record to `spec_generations` so an admin can review who generated
    what. Both writes reuse the artifacts assembled here; both are gated
    on the validate verdict (`val_result`, already computed by
    `_verify_spec_output` in `_on_success` — validate re-fetches
    safetensors metadata, so it runs once per run, not once per hook).

    Reads the identity/drafts the agent wrote in this run's overlay, then
    normalizes the complete provenance bundle before any persistent write.

    `record_result`, when given, is filled with the per-user record's
    outcome (`done.record`).

    Silent-fail by design: a Supabase outage must NOT break the
    user-facing /generate-spec stream. The local disk write already
    happened in _maybe_snapshot + the agent's file writes; the cache
    is a pure optimization.

    Skipped when:
      • SUPABASE_* env isn't configured (dev / no-service builds)
      • the expected model yaml didn't get written (already a different
        terminal verdict, don't pile on)
    """
    try:
        # Per-user runs isolate ALL artifacts under a private overlay;
        # host/anonymous runs harvest from their `__anonymous__` kind leaf
        # (S4 §2.5). `use_overlay` drives where we read; cache publish is
        # separate — a private overlay must not populate cached_specs.
        use_overlay = bool(target_root)
        private_overlay = use_overlay and not is_anonymous_overlay(target_root)
        from backend.arch.snapshots import snapshot_path
        snap_path = snapshot_path(model_id, target_root if use_overlay else None)

        # Locate the model yaml via the shared finder (fresh write first, else a
        # refine overlay's seeded spec), so the publish gate and the success
        # verdict agree on what "produced" means — a no-op-but-valid refine
        # re-publishes its unchanged, valid spec instead of silently skipping.
        safe_id = _safe(model_id)
        model_yaml_path = _fresh_model_yaml(
            started_at, target_root, prefer_stem=safe_id)
        if model_yaml_path is None and seeded:
            model_yaml_path = _existing_model_yaml_with_stem(
                target_root, prefer_stem=safe_id)
        if model_yaml_path is None:
            log_file.write("[cache-publish] SKIP: no fresh model yaml found\n")
            log_file.flush()
            return None

        spec_yaml_text = model_yaml_path.read_text(encoding="utf-8")

        # Publish-time validate gate.
        #
        # The cross-VM cache is shared mutable state: a poisoned row sits
        # there until manually deleted, and any other VM (or the same VM
        # after a `rm`) re-hydrates the bad spec from it on the next
        # /analyze call — looks like local files "regenerate themselves"
        # to the user. The agent's own "success" verdict is too forgiving
        # to gate on: it can pass a spec that still raises at compute time
        # (e.g. a slice node binding dim via the `dim:` shorthand while the
        # op's flops_fn reads `d["in"]` → `KeyError: 'in'` at every
        # compute). So we gate on the validate verdict instead.
        #
        # `val_result` is the verdict run's validate result, handed down by
        # `_on_success` (None there means validate crashed → downgrade → we
        # never get here). A standalone caller may omit it; the gate then
        # re-validates. Policy:
        #   • Engine crash (KeyError, unbound op param, NameError, …)  → SKIP
        #   • validate() returns errors:[...]                            → SKIP
        #   • param_mismatches present but no fatal errors              → ALLOW
        #     (a structurally-complete-but-numerically-wrong spec is
        #      still a useful iteration checkpoint; downstream Phase 4
        #      consumers can refine from it)
        # The fallback validate, like the verdict gate, runs against the
        # checkout as it is now and logs the revision it ran at.
        if val_result is None:
            log_file.write(
                f"[cache-publish] validate "
                f"{RunRevision.since(code_revision).stamp()}\n")
            log_file.flush()
            try:
                val_result = _invoke_validate(
                    model_id, model_yaml_path.stem, target_root)
            except Exception as e:
                log_file.write(
                    f"[cache-publish] SKIP — validate raised "
                    f"{e.__class__.__name__}: {str(e)[:200]}\n"
                )
                log_file.flush()
                return
        val_errors = val_result.get("errors") or []
        if val_errors:
            log_file.write(
                f"[cache-publish] SKIP — spec failed validate; "
                f"first errors: {val_errors[:3]}\n"
            )
            log_file.flush()
            return

        snapshot_json = _record_snapshot(model_id, target_root, snapshot_fields)

        try:
            # Self-register only after normalization: a failed bundle must
            # leave no record/cache/PR side effect, and no unreviewed base
            # may travel.
            spec_yaml_text, new_component_yamls = _record_bundle(
                model_id, target_root, started_at, seed_bundle=seed_bundle,
                extra_files=extra_files, spec_yaml=spec_yaml_text)
        except SpecBundleError as exc:
            log_file.write(
                f"[cache-publish] SKIP — bundle normalization failed: "
                f"{'; '.join(exc.errors)}\n")
            log_file.flush()
            return _bundle_failure(exc)

        # Prefer the per-user agent the run actually used (injected via
        # prep_result); fall back to the globally-configured one.
        if agent_cli is None:
            try:
                agent_cli = get_agent_cli()
            except (RuntimeError, ImportError):
                agent_cli = None

        # Publication and artifact location are separate facts. A private
        # overlay harvests locally but never populates the global cache; an
        # anonymous overlay publishes globally without landing in backend/arch.
        # Custom IDs retain their dedicated custom cache, which is their
        # loader-facing identity channel.
        if is_anonymous_overlay(target_root) and is_custom_id(model_id):
            land_anonymous_overlay(
                model_id, target_root, spec_yaml_text,
                new_component_yamls=new_component_yamls,
                snapshot_src=snap_path if snap_path.is_file() else None,
            )

        publish_global_cache = (not use_overlay) or not private_overlay
        if not publish_global_cache:
            log_file.write(
                "[cache-publish] SKIP (private user run → spec_generations only)  "
                f"{model_id}  yaml={len(spec_yaml_text)}B  "
                f"+components={len(new_component_yamls)}\n"
            )
            log_file.flush()
        elif supabase_env.configured():
            ok = shared_spec_miss.upsert_after_success(
                model_id=model_id,
                spec_yaml=spec_yaml_text,
                snapshot_json=snapshot_json,
                new_component_yamls=new_component_yamls,
                agent_cli=agent_cli,
                generated_by=user_id,
            )
            log_file.write(
                f"[cache-publish] {'OK' if ok else 'FAIL'}  {model_id}  "
                f"yaml={len(spec_yaml_text)}B  "
                f"+components={len(new_component_yamls)}\n"
            )
            log_file.flush()
        else:
            try:
                shared_spec_miss.write_local_after_success(
                    model_id=model_id,
                    spec_yaml=spec_yaml_text,
                    snapshot_json=snapshot_json,
                    new_component_yamls=new_component_yamls,
                    agent_cli=agent_cli,
                )
            except Exception as cache_exc:
                logger.warning(
                    "[cache-publish] local shared cache write failed for %s: %s",
                    model_id, cache_exc)
                log_file.write(
                    f"[cache-publish] LOCAL-FAIL  {model_id}: {cache_exc}\n")
                log_file.flush()
                return Downgrade(
                    kind="cache_warning",
                    reason=(
                        "Spec generated, but its local shared-cache write failed; "
                        "the artifact exists only in the scratch run overlay. "
                        "See the generation log for [cache-publish] LOCAL-FAIL."),
                )
            log_file.write(
                f"[cache-publish] LOCAL-OK  {model_id}  "
                f"yaml={len(spec_yaml_text)}B  "
                f"+components={len(new_component_yamls)}\n"
            )
            log_file.flush()

        # The run's record: one working candidate per (owner, model). A later
        # successful generate overwrites generated/rejected/closed; a
        # contributed/merged row stays frozen; a pending record from an
        # earlier attempt is dropped. `records` decides where it is kept —
        # `spec_generations` on a Supabase deployment (where an unattributed
        # run still keeps none), the host-local store otherwise, so an
        # offline run leaves the record review and fix are pointed at.
        # A records outage does not fail the run: it is reported as the
        # `failed` record outcome.
        recorded = records.record(
            model_id=model_id,
            user_id=user_id or "",
            spec_yaml=spec_yaml_text,
            snapshot_json=snapshot_json,
            new_component_yamls=new_component_yamls,
            agent_cli=agent_cli,
            agent_model=agent_model,
            run_started_at=run_started_at,
        )
        log_file.write(
            f"[spec-record] {recorded.get('outcome')}  "
            f"{model_id}  owner={run_overlay_owner(user_id or '')}\n"
        )
        log_file.flush()
        if record_result is not None:
            record_result.update(recorded)
        return None
    except Exception as e:
        try:
            log_file.write(f"[cache-publish] SKIP ({e.__class__.__name__}: {e})\n")
            log_file.flush()
        except OSError:
            pass


def _maybe_snapshot(model_id: str, log_file, captured: Optional[dict] = None,
                    target_root: Optional[str] = None) -> None:
    """After a successful /generate-spec run, persist the model's config +
    safetensors metadata + forward source to backend/arch/snapshots/.

    Fast path (Plan B): when `captured` is provided (dict with
    config/metadata/forward_source filled in at prompt-prep time), write
    directly from memory — zero extra HF round-trips on the done event.

    Slow path: re-fetch via snapshot_one (preserved for manual / batch
    refreshes from the CLI). Best-effort: failure is logged but doesn't
    block the success event.

    Skips custom:* model_ids (no HF snapshot semantics).
    """
    try:
        if is_custom_id(model_id):
            return

        # Signed-in runs write the snapshot into their private overlay, not the
        # shared snapshots/ dir (zero shared-tree footprint).
        if captured and captured.get("config") and captured.get("metadata"):
            from backend.arch.snapshots import release_dates_for, write_snapshot
            out = write_snapshot(
                model_id,
                captured["config"],
                captured["metadata"],
                forward_source=captured.get("forward_source") or None,
                model_str=captured.get("model_str") or None,
                release_date=release_dates_for([model_id]).get(model_id),
                overlay_dir=target_root,
            )
            n_groups = len(captured["metadata"]["header"]["tensors"])
            size_kb = out.stat().st_size / 1024
            log_file.write(
                f"[snapshot] OK (from prompt cache)  {model_id:40s}  "
                f"{n_groups} tensor groups  {size_kb:.1f} KB  → {out.name}\n"
            )
            log_file.flush()
            return

        # Fallback re-fetch writes to the SHARED snapshots dir; skip it for a
        # private (overlay) run that had no captured data.
        if target_root:
            log_file.write("[snapshot] SKIP (private run, no captured snapshot data)\n")
            log_file.flush()
            return
        from backend.hf.snapshot import snapshot_one
        result = snapshot_one(model_id, skip_if_exists=False)
        log_file.write(f"[snapshot] {result}\n")
        log_file.flush()
    except Exception as e:
        try:
            log_file.write(f"[snapshot] SKIP ({e.__class__.__name__}: {e})\n")
            log_file.flush()
        except OSError:
            pass


def _on_success(model_id: str, log_file: IO, started_at: float, prep_result: dict) -> Optional[str]:
    """Spec-worker `on_success` hook fed to `stream_agent_core`.

    Order matters: snapshot first (local truth), then verdict validation, then
    cross-VM publish (which reads the freshly-written snapshot + model yaml).

    Returns an optional DOWNGRADE REASON: when the agent exited cleanly but
    produced no valid spec, the string is returned so the runner rewrites the
    `done` event to success=False (a clean exit alone is too forgiving — see
    `_verify_spec_output`). None keeps the success verdict.
    """
    snapshot_data = prep_result.get("snapshot_data")
    # user_id / agent_cli are injected into prep_result by the service-layer
    # run_fn (backend/service/routes/spec.py); both are absent on the core
    # anonymous /generate-spec path, leaving the per-user record a no-op and
    # the agent_cli falling back to the global default.
    user_id = prep_result.get("user_id")
    agent_cli = prep_result.get("agent_cli")
    agent_model = _run_agent_model(prep_result, agent_cli)
    # Per-run overlay root (always set for generate/refine).
    target_root = prep_result.get("target_root")
    # This run started from its leaf's spec: a refine seeded it
    # (prepare_refine_prompt) or a continued generation re-opened the draft
    # (prepare_resume). Only then may the verdict/publish gate fall back to
    # the overlay's spec on a no-edit run — a NEW generate reuses the leaf and
    # could otherwise pass on a spec it did not write (see _existing_model_yaml).
    seeded = bool(prep_result.get("seeded_spec"))
    # This round's verdict replaces the last round's.
    prep_result.pop("terminal", None)
    _maybe_snapshot(model_id, log_file, snapshot_data, target_root=target_root)
    downgrade, val_result = _verify_spec_output(
        model_id, log_file, started_at, root=target_root, seeded=seeded,
        code_revision=prep_result.get("code_revision"),
    )
    if downgrade:
        # The runner's contract is a reason string; the typed verdict rides
        # on `prep_result`, which the repair hook receives alongside it.
        note_downgrade(prep_result, downgrade)
        if downgrade.kind == "missing_primitive" and downgrade.errors:
            prep_result["missing_primitive_gap"] = json.loads(downgrade.errors[0])
        if downgrade.kind == "blocked_by_engine":
            record = _record_blocked(
                model_id, log_file, started_at, prep_result, downgrade)
            prep_result["terminal"]["record"] = record
            if record["outcome"] == RecordOutcome.FAILED.value:
                return f"{downgrade.reason} {BLOCKED_NOT_SAVED}"
        return downgrade.reason
    # What the verdict judged rides on the run's `done` (`checks`: the
    # checks it ran and the ones it had no checkpoint for).
    if (val_result or {}).get("checks"):
        prep_result.setdefault("terminal", {})["checks"] = val_result["checks"]
    record: dict = {}
    publish_downgrade = _publish_to_cache(
        model_id, log_file, started_at, user_id=user_id, agent_cli=agent_cli,
        agent_model=agent_model,
        target_root=target_root, val_result=val_result,
        extra_files=prep_result.get("op_registry_files") or [],
        seeded=seeded,
        seed_bundle=prep_result.get("seed_bundle"),
        code_revision=prep_result.get("code_revision"),
        run_started_at=prep_result.get("run_started_at"),
        record_result=record,
        snapshot_fields=prep_result.get("snapshot_fields"),
    )
    if record:
        prep_result.setdefault("terminal", {})["record"] = record
    if publish_downgrade is not None:
        if publish_downgrade.kind == "cache_warning":
            prep_result["done_message"] = publish_downgrade.reason
            return None
        note_downgrade(prep_result, publish_downgrade)
        return publish_downgrade.reason
    return None


def note_downgrade(prep_result: dict, downgrade: Downgrade) -> None:
    """Put a downgrade on `prep_result`: the typed verdict the repair hook
    reads, and `terminal` — what the run's `done` event carries beside its
    message when this verdict is final: the machine key and errors, and the
    engine refusals of a blocked run (the runner adds `terminal` to `done`;
    the job registry carries it into the run's log row)."""
    prep_result["downgrade"] = downgrade
    terminal = prep_result.setdefault("terminal", {})
    if downgrade.error_kind:
        terminal["error_kind"] = downgrade.error_kind.value
        terminal["errors"] = list(downgrade.errors)
    if downgrade.engine_gaps:
        terminal["engine_gaps"] = list(downgrade.engine_gaps)


def _run_agent_model(prep_result: dict, agent_cli: Optional[str]) -> Optional[str]:
    """The model id this run actually used (migration 0041): the run's
    injected env first (a BYOK credential's resolved model), else the host
    agent's env var; None when the agent runs on the endpoint's default."""
    from backend.agents.base import AGENT_ENV_VARS
    names = AGENT_ENV_VARS.get(agent_cli or "")
    var = names.model if names else None
    if not var:
        return None
    env = prep_result.get("agent_env") or {}
    return env.get(var) or os.environ.get(var)


def _max_repair_rounds() -> int:
    """Verdict-gate repair attempts per run (`limits.max_repair_rounds`, which
    the lock TTL is sized from too)."""
    return max_repair_rounds()

def _repair_prompt_fn(model_id: str):
    """Build the runner's repair hook for this model. Only verdict-gate
    VALIDATION failures are repairable — including engine exceptions
    `validate` turned into `errors[]`. 'no yaml written' and an
    infrastructure raise out of `validate` describe an agent that never
    produced anything fixable, so they get no retry. A repairable run
    re-spawns the agent in the SAME workspace with the validator's error
    list and has it fix its own files in place.

    A round its own timer stopped (`prep_result["_round_timed_out"]`, set by
    the runner) is judged on the files it left: when they fail validation
    the agent continues from them with the current errors, and when it
    authored YAML (draft components, say) but not yet the model spec, it
    continues from those files to write the spec."""
    def _make(reason: str, prep_result: dict) -> Optional[str]:
        downgrade = prep_result.get("downgrade")
        if not isinstance(downgrade, Downgrade):
            return None
        from ._prompts.tasks import _PY
        timed_out = bool(prep_result.get("_round_timed_out"))
        if downgrade.kind == "no_output" and timed_out:
            files = authored_yaml(prep_result.get("target_root"))
            if not files:
                return None
            listed = "\n".join(f"- `{name}`" for name in files)
            return (
                f"Your previous round on {model_id} reached its time budget "
                "before it wrote the model spec. The files you wrote are in "
                f"place in this workspace:\n{listed}\n\n"
                "Continue from them: complete the components you started and "
                f"write the model spec to `models/{_safe(model_id)}.yaml`. "
                f"Then run `{_PY} -m backend.inference.validate {model_id}` "
                "and iterate until it reports PASS or `blocked_by_engine: true`."
            )
        if downgrade.kind != "validation":
            return None  # missing_primitive is chained by stream_agent, not repaired
        errors = "; ".join(downgrade.errors)[:4000]
        if timed_out:
            return (
                f"Your previous round on {model_id} reached its time budget. "
                "The files you wrote are in place in this workspace; continue "
                "from them and edit them in place.\n\n"
                f"Current validation errors:\n{errors}\n\n"
                f"After each fix re-run `{_PY} -m backend.inference.validate "
                f"{model_id}` and iterate until it reports PASS or `blocked_by_engine: true`."
            )
        return (
            f"The spec you just wrote for {model_id} FAILS the verdict "
            "validator. Fix your own files in place — do not rewrite from "
            "scratch, and do not touch files you did not create.\n\n"
            f"Validation errors:\n{errors}\n\n"
            f"After each fix re-run `{_PY} -m backend.inference.validate "
            f"{model_id}` and iterate until it reports PASS or `blocked_by_engine: true`."
        )
    return _make


def _upload_run_log(model_id: str, prep_result: dict, run_key: Optional[str],
                    log_path: str, success: Optional[bool]) -> None:
    """Bind this run's log to the triggering user in the DB (success OR
    failure), and to the generation record the run wrote, when it wrote one
    (`done.record` = `recorded` with its id). Signed-in runs only;
    best-effort so logging never breaks the run or leaks an exception into
    the generator. A missing terminal verdict is recorded as failure — never
    `success: null`."""
    user_id = prep_result.get("user_id")
    if success is None:
        success = False
    if not user_id or not supabase_env.configured():
        clear_job_log_sidecar(log_path)
        return
    try:
        from backend.service import spec_generation_logs
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            log_text = f.read()
        # The row is owner-readable (/api/account/gen-logs/{id}) and the raw
        # log — prompt echo included — carries the overlay root, which embeds
        # the deploy home + the owner's user UUID.
        log_text = scrub_paths(log_text, prep_result.get("target_root"))
        uploaded = bool(spec_generation_logs.record(
            user_id=user_id, model_id=model_id, run_key=run_key,
            success=success, log=log_text,
            agent_cli=prep_result.get("agent_cli"),
            agent_model=_run_agent_model(prep_result, prep_result.get("agent_cli")),
            record_id=_written_record_id(prep_result),
        ))
    except Exception:
        logger.warning("spec_generation_logs upload failed", exc_info=True)
        return
    if uploaded:
        clear_job_log_sidecar(log_path)


def _written_record_id(prep_result: dict) -> Optional[str]:
    """The id of the generation record this run wrote (`done.record`), or
    None when it wrote none."""
    written = (prep_result.get("terminal") or {}).get("record") or {}
    if written.get("outcome") != RecordOutcome.RECORDED.value:
        return None
    return written.get("id")


def _stage_snapshot(model_id: str, snapshot_data: Optional[dict],
                    target_root: Optional[str], log_path: Optional[str]) -> None:
    """Drop the prefetched snapshot into the overlay BEFORE the agent launches.

    The sandboxed agent has no network, so its mandated write → validate →
    iterate loop only works when `load_snapshot` can resolve config +
    metadata from the overlay (see snapshots.load_snapshot). Review and fix
    prep stage it once for their own validate and the stream stages it again
    for the job log; generate / refine re-write it on success via
    `_maybe_snapshot`. Best-effort: a missing
    or partial capture leaves the run exactly as before (agent-side
    validate then needs network it may not have — logged, not raised).
    `log_path` is None at prep time, before a job log exists; the outcome
    then goes to the server log.
    """
    if not target_root or is_custom_id(model_id):
        return
    if not snapshot_data or not snapshot_data.get("config") \
            or not snapshot_data.get("metadata"):
        return
    try:
        from backend.arch.snapshots import write_snapshot
        out = write_snapshot(
            model_id,
            snapshot_data["config"],
            snapshot_data["metadata"],
            forward_source=snapshot_data.get("forward_source") or None,
            model_str=snapshot_data.get("model_str") or None,
            overlay_dir=target_root,
        )
        _note(log_path, f"[snapshot] staged for offline validate → {out.name}")
    except Exception as exc:  # staging must never break a run
        _note(log_path, "[snapshot] staging failed (agent validate will need "
                        f"network): {exc}")


def _note(log_path: Optional[str], line: str) -> None:
    if log_path is None:
        logger.info(line)
        return
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def _chain_op_then_continue(
    model_id: str,
    prep_result: dict,
    run_key: str,
    agent_cli: str,
    agent_env: Optional[dict],
    log_path: str,
    ts: str,
) -> Generator[str, None, None]:
    """Op writer in this job, then a second generate turn, then one PR."""
    from backend.arch.ops_registry import PENDING_OP_PATCH_ENV
    from backend.agents.base import host_agent_credentials
    from backend.op_workers.pipeline.chain import run_propose_review
    from ._prompts.tasks import continuation_after_op_prompt

    gap = prep_result["missing_primitive_gap"]
    op_name = gap["op_name"]
    yield _sse({
        "type": "log",
        "text": (
            f"Spec writer stopped on missing primitive `{op_name}`; "
            "running op writer..."
        ),
    })

    frames: list[str] = []

    def _on_frame(frame: str) -> None:
        frames.append(frame)

    chain = run_propose_review(
        op_name, gap, consume_slot=False, on_frame=_on_frame,
    )
    for frame in frames:
        yield frame

    if not chain.ready:
        prep_result["chain_success"] = False
        yield _sse({
            "type": "done", "success": False,
            "message": (
                f"{chain.message}. Proposal is in "
                f"backend/arch/ops_proposals/{op_name}/ — not a PR. "
                "Open PR from your account after a "
                "successful generate."
            ),
        })
        return

    yield _sse({
        "type": "log",
        "text": (
            f"Op `{op_name}` passed the auto-merge envelope; "
            "continuing spec generation..."
        ),
    })

    cont_prompt = continuation_after_op_prompt(
        prep_result.get("prompt") or "", op_name,
    )
    op_files = [
        {"path": path, "body": body}
        for path, body in chain.proposal_files.items()
    ]
    prep_result["op_registry_files"] = op_files
    cont_prep = {
        **prep_result,
        "prompt": cont_prompt,
        "missing_primitive_gap": None,
        "downgrade": None,
        "op_registry_files": op_files,
    }
    if agent_env is not None:
        cont_env = {**agent_env, PENDING_OP_PATCH_ENV: chain.patch_path}
    else:
        cont_env = {
            **host_agent_credentials(agent_cli),
            PENDING_OP_PATCH_ENV: chain.patch_path,
        }
    target_root = prep_result.get("target_root")
    if target_root:
        cont_env["HA_SPEC_OVERLAY_DIR"] = target_root

    last_done = None
    cont_log = log_path[:-4] + "_continue.log" if log_path.endswith(".log") else log_path + ".continue"
    for frame in stream_agent_core(
        key=run_key,
        prep_result=cont_prep,
        agent_cli=agent_cli,
        log_path=cont_log,
        log_header=f"=== {model_id} continue after {op_name} @ {ts} sha={_code_revision()} ===",
        launch_banner="Resuming spec writer...",
        procs_dict=_model_procs,
        locks_lock=_locks_lock,
        release_fn=lambda _k: None,
        owns=owns,
        on_success=lambda _k, log_file, started_at, pr: _on_success(
            model_id, log_file, started_at, pr),
        env=cont_env,
        hard_cap_sec=OP_CONTINUE_SEC,
        deadline_at=prep_result.get("_run_deadline_at"),
        round_budget_sec=ROUND_BUDGET_SEC,
        timeout_hint=OP_CONTINUE_TIMEOUT_KNOB,
        round_timeout_hint=ROUND_TIMEOUT_KNOB,
        repair_prompt_fn=_repair_prompt_fn(model_id),
        max_repair_rounds=_max_repair_rounds(),
    ):
        if frame.startswith("data: "):
            try:
                evt = json.loads(frame[len("data: "):].strip())
            except (ValueError, KeyError):
                evt = None
            if evt and evt.get("type") == "done":
                last_done = evt
        yield frame

    success = bool(last_done and last_done.get("success"))
    prep_result["chain_success"] = success
    # The continuation's session is the one a later continuation re-enters.
    if cont_prep.get("_session_id"):
        prep_result["_session_id"] = cont_prep["_session_id"]
    if last_done is None:
        last_done = {"type": "done", "success": False,
                     "message": "continuation ended without a done event"}
    elif success and op_files:
        last_done = {
            **last_done,
            "message": (
                (last_done.get("message") or "ok")
                + f". Primitive `{op_name}` rides this generation; "
                  "Open PR from your account ships spec and op together."
            ),
        }
    yield _sse(last_done)


def frame_event(frame: str) -> Optional[dict]:
    """The event dict inside one `data: <json>` frame the agent streams, or
    None for anything else (the background job buffers dicts, not frames)."""
    if not frame.startswith("data: "):
        return None
    try:
        return json.loads(frame[6:].strip())
    except json.JSONDecodeError as exc:
        logger.debug("agent SSE frame is not JSON: %s", exc)
        return None


class _LeafRecord:
    """Keeps a generate leaf's `RunEnd` current through one job, so the
    leaf always says how its last run stands: recorded as unfinished as soon
    as the agent session is known (a run a server restart cuts off can still
    be re-entered), and again with the terminal `done`. Inert for any other
    leaf."""

    _UNFINISHED = "The run stopped before it finished."

    def __init__(self, model_id: str, target_root: Optional[str]):
        self.root = target_root if is_generate_leaf(target_root, model_id) else None
        self._session: Optional[str] = None
        self.finished = False

    def progress(self, prep_result: dict) -> None:
        session = prep_result.get("_session_id")
        if self.root and session and session != self._session:
            self._session = session
            RunEnd.unfinished(prep_result, self._UNFINISHED).write(self.root)

    def finish(self, event: dict, prep_result: Optional[dict]) -> dict:
        """Record a terminal `done` and mark a failed one `resumable`. A
        `done` from a failed prep (`prep_result` None) started no agent and
        leaves the earlier record as it is."""
        if self.root is None or event.get("type") != "done":
            return event
        self.finished = True
        if prep_result is not None:
            RunEnd.from_done(event, prep_result).write(self.root)
        if event.get("success"):
            return event
        return {**event, "resumable": has_resumable_draft(self.root)}

    def abandon(self, prep_result: dict) -> None:
        """The job ended without a terminal `done` (a crash, or the
        generator closed): the run is unfinished."""
        if self.root and not self.finished:
            RunEnd.unfinished(prep_result, self._UNFINISHED).write(self.root)


def generation_job(model_id: str, prep_fn, *, stamp: Optional[dict] = None,
                   run_key: Optional[str] = None,
                   target_root: Optional[str] = None,
                   failover=None):
    """The generator `start_job` consumes for a generate / refine run.

    Prep (metadata fetch, prompt build) runs inside the job thread, off the
    request path, so the POST returns at once and the browser polls. `stamp`
    carries what the service layer adds to `prep_result` — the requesting
    user and a BYOK agent choice / credential env. If prep raises, the run
    lock this job holds is released here: `stream_agent`, whose `finally`
    releases it on every other path, never starts.

    `target_root` is the run's overlay leaf; a generate leaf's run record
    is kept by `_LeafRecord`, and a failed `done` there carries `resumable`
    — whether the leaf holds a draft a continued generation can pick up.

    `failover` (a `failover.Failover`) keeps the run going past an
    unavailable agent service or a timed-out round (`stream_agent`).
    """
    def _run():
        record = _LeafRecord(model_id, target_root)
        yield {"type": "log", "text": "Starting analysis..."}
        yield {"type": "log", "text": "Fetching model metadata..."}
        try:
            prep_result = prep_fn()
        except ForwardSourceUnavailable as exc:
            release(run_key or model_id)
            yield record.finish(exc.done_event(), None)
            return
        except Exception as e:
            release(run_key or model_id)
            yield record.finish({"type": "done", "success": False,
                                 "message": f"Metadata fetch failed: {e}"}, None)
            return
        except BaseException:
            release(run_key or model_id)
            raise
        prep_result.update(stamp or {})
        try:
            for frame in stream_agent(model_id, prep_result, run_key=run_key,
                                      failover=failover):
                event = frame_event(frame)
                if event is not None:
                    record.progress(prep_result)
                    yield record.finish(event, prep_result)
        finally:
            record.abandon(prep_result)
    return _run


def _ends_on_agent(event: dict) -> dict:
    """A failed terminal `done` whose agent ended without judging the spec
    (`agent_end`: unavailable, timed out, killed, context exhausted) names
    that as the run's `error_kind`: the run did not fail on its spec.
    `failed` names no cause and leaves the event as it is; a run error kind
    is the agent's own word for the rest (`RunErrorKind(end)` raises on
    drift)."""
    end = event.get("agent_end")
    if event.get("success") or end in (None, AgentEnd.FAILED.value):
        return event
    return {**event, "error_kind": RunErrorKind(end).value}


def stream_agent(model_id: str, prep_result: dict, *,
                 run_key: Optional[str] = None,
                 failover=None) -> Generator[str, None, None]:
    """Run the configured agent CLI and stream normalized SSE events.

    `run_key` is the opaque per-run concurrency/registry key threaded through
    `stream_agent_core` (procs_dict / release_fn / status lookup). It defaults
    to `model_id` — the core anonymous / single-user path, unchanged. The
    service layer passes a composite `f"{user_id}|{model_id}"` so two users
    generating the same model don't collide on a global lock. Artifact naming
    (snapshot / cache row / log file) always uses the REAL `model_id`, never the
    composite key — hence the on_success wrapper that re-binds the true model_id.

    `failover` (a `failover.Failover`), when given, decides after each failed
    agent `done` whether the run goes on — with the next agent backend when
    the service was unavailable, or from the draft when a round timed out —
    under the same lock, log and overall deadline. Only the last `done`
    leaves this generator; one that ended on the agent's side names it as
    its `error_kind` (`_ends_on_agent`).
    """
    run_key = run_key or model_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = _safe(model_id)
    log_path = os.path.join(_LOG_DIR, f"{safe_id}_{ts}.log")
    # BYOK (Phase 2): the service-layer run_fn may inject a per-user agent
    # choice + credential env into prep_result. Absent on the core anonymous
    # path, where we fall back to the globally-configured agent and its
    # allowlisted host credentials. The full server environment is never
    # inherited by the subprocess.
    agent_cli = prep_result.get("agent_cli") or get_agent_cli()
    target_root = prep_result.get("target_root")
    _stage_snapshot(model_id, prep_result.get("snapshot_data"),
                    target_root, log_path)
    revision = _code_revision()
    prep_result["code_revision"] = revision
    # When the run began: a record keeps it, and a run that began earlier
    # never overwrites the pending record of one that began later.
    prep_result.setdefault(
        "run_started_at", datetime.now(timezone.utc).isoformat())
    bind_job_log(
        run_key, log_path,
        user_id=prep_result.get("user_id"), model_id=model_id)
    if failover is not None:
        failover.start(prep_result, agent_cli)
    final_success = None
    # Hold the spec-gen slot across an op-writer chain and across the
    # agents a failover runs; no agent's finally may drop it before the run
    # is over.
    _release_held = {"on": True}

    def _release_after_chain(key: str) -> None:
        if not _release_held["on"]:
            release(key)

    log_header = f"=== {model_id} @ {ts} sha={revision} ==="
    attempt = 0
    try:
        while True:
            # The CLI this attempt spawns: a continued generation re-enters
            # its session only with the same CLI (`RunEnd`).
            prep_result["_agent_cli"] = agent_cli
            # Per-user run: point the agent's in-loop `validate`/`structure`
            # at its private overlay so they resolve the in-progress spec
            # that isn't in the shared tree. A BYOK env is a scrubbed dict
            # (no server secrets); None keeps the host's own credentials.
            agent_env = prep_result.get("agent_env")
            if target_root and agent_env is not None:
                agent_env = {**agent_env, "HA_SPEC_OVERLAY_DIR": target_root}
            continued = None
            for frame in stream_agent_core(
                key=run_key,
                prep_result=prep_result,
                agent_cli=agent_cli,
                log_path=log_path,
                log_header=log_header,
                launch_banner="Launching coding agent...",
                procs_dict=_model_procs,
                owns=owns,
                locks_lock=_locks_lock,
                release_fn=_release_after_chain,
                # stream_agent_core calls on_success(key, ...); re-bind the real
                # model_id so snapshot/publish/record naming ignores the composite key.
                on_success=lambda _k, log_file, started_at, pr: _on_success(
                    model_id, log_file, started_at, pr),
                env=agent_env,
                hard_cap_sec=_AGENT_HARD_CAP_SEC,
                # Every attempt after the first spends what is left of the
                # run's one deadline, not a fresh budget.
                deadline_at=prep_result.get("_run_deadline_at"),
                round_budget_sec=ROUND_BUDGET_SEC,
                timeout_hint=GENERATE_TIMEOUT_KNOB,
                round_timeout_hint=ROUND_TIMEOUT_KNOB,
                repair_prompt_fn=_repair_prompt_fn(model_id),
                max_repair_rounds=_max_repair_rounds(),
                resume_session=prep_result.get("resume_session"),
                append_log=attempt > 0,
            ):
                evt = frame_event(frame)
                if evt is None or evt.get("type") != "done":
                    yield frame
                    continue
                # A missing-primitive gap is not a terminal failure: suppress
                # the first `done` and chain the op writer, then a second
                # generate turn.
                if prep_result.get("missing_primitive_gap"):
                    yield from _chain_op_then_continue(
                        model_id, prep_result, run_key, agent_cli, agent_env,
                        log_path, ts,
                    )
                    final_success = prep_result.get("chain_success")
                    break
                # A stopped run no longer owns its key and is not continued.
                if failover is not None and owns(run_key):
                    continued = failover.next(evt, prep_result)
                if continued is not None:
                    break
                if evt.get("success") and prep_result.get("done_message"):
                    evt = {**evt, "message": prep_result["done_message"]}
                evt = _ends_on_agent(evt)
                final_success = bool(evt.get("success"))
                yield _sse(evt)
            if continued is None:
                break
            agent, why = continued
            agent_cli = agent.cli
            attempt += 1
            log_header = (f"\n=== {model_id} continues with {agent_cli} "
                          f"(attempt {attempt + 1}) ===\n[failover] {why}")
            yield _sse({"type": "log", "text": why})
    finally:
        _release_held["on"] = False
        release(run_key)
        if final_success is None:
            final_success = False
        # Covers abort/exception as well as a clean done. The daemon job
        # thread drives this generator, so this still runs if the browser
        # disconnected.
        _upload_run_log(model_id, prep_result, run_key, log_path, final_success)
