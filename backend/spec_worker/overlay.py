"""Per-user spec-generation overlays (out-of-tree write isolation).

A spec-gen run must NOT write into the shared repo tree
(`backend/arch/{models,components/drafts,snapshots}`) — that would make
one generation globally visible with no PR and race a concurrent generate
or refine of the same model. Instead each run gets its own overlay directory:

    <SPEC_RUNS_ROOT>/<owner>/<kind>__<safe_model_id>/
        models/                 — the agent writes <name>.yaml here
        components/drafts/      — new draft components
        snapshots/              — per-run snapshot json
        backend/arch/      — YAML-only reference tree for the agent
        .agent-home/            — per-run CLI state
        .resume.json            — how the last generate run ended (`RunEnd`)

`owner` is the signed-in user id when per-user isolation is on, otherwise
the reserved `__anonymous__` owner (S4 §2.5). Generate and refine never
share a leaf.

and serving materializes a user's stored spec into:

    <USER_SPECS_ROOT>/<user_id>/{models,components/drafts}/

The loader layers these dirs on top of the shared read-only `components/library`
via `SpecLoader(overlay_model_dirs=..., overlay_draft_dirs=...)`.

Home/permission note: in the container the backend runs as root while the agent
drops to `HA_AGENT_USER` (`specagent`) with `HOME=/home/specagent`, so a
`~`-relative path would resolve differently for each and be unwritable by the
agent. We therefore use an ABSOLUTE root (env `HA_SPEC_RUNS_ROOT` /
`HA_USER_SPECS_ROOT`, set under the agent-writable `/app/.cache` in the image)
and chown freshly-created dirs to the agent user. In dev (no privilege drop)
both processes share one home and the default `~/.huggingarch/...` works.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from backend.agents.run_user import untrusted_run_user
from backend.arch.loader import SpecLoader
from backend.arch.snapshots import snapshot_path, store_snapshot
from backend.hf.metadata import fetch_config
from backend.custom_id import is_custom_id, model_path as custom_model_path, strip_prefix
from backend.spec_worker.self_register import stamp_match_model_id
from backend.spec_worker.bundle import BlockedBundle, NormalizedBundle
from backend.spec_worker.limits import LOCK_TTL_SEC

logger = logging.getLogger(__name__)


# Self-host / per-user-off / core anonymous runs share this reserved owner
# (S4 §2.5). It is not a user id; it only namespaces kind-isolated leaves.
ANONYMOUS_OWNER = "__anonymous__"

# Short post-success land onto one serving target (model id / custom cache).
# S4 takes over publication; this only serializes the copy and writes via
# temp+rename so a crash cannot leave a half-written serving file.
_land_guard = threading.Lock()
_land_locks: dict[str, threading.Lock] = {}
formal_tiers = ("library", "adapters")

def _safe(s: str) -> str:
    return s.replace("/", "__")


def _under(root: Path, *names: str) -> Path:
    """`root/<names...>` with the result proven to stay under `root`. A
    model id or user id is request input; `..` in one of them would
    otherwise point `rmtree` at the store above the overlay."""
    path = root.joinpath(*names)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"overlay path {path} escapes {root}")
    return path


def spec_runs_root() -> Path:
    v = os.environ.get("HA_SPEC_RUNS_ROOT")
    return Path(v) if v else Path.home() / ".huggingarch" / "spec_runs"


def user_specs_root() -> Path:
    v = os.environ.get("HA_USER_SPECS_ROOT")
    return Path(v) if v else Path.home() / ".huggingarch" / "user_specs"


def is_user_spec_path(path) -> bool:
    """Whether a spec file lives in a user's saved overlay — the one place
    a broken spec is the user's to regenerate rather than the library's."""
    try:
        Path(path).resolve().relative_to(user_specs_root().resolve())
    except (ValueError, OSError):
        return False
    return True


def _chown_to_agent(*paths: Path) -> None:
    """Give the unprivileged agent user ownership of freshly-created overlay
    dirs so it can write there. No-op in dev (no privilege drop) or if the
    user doesn't exist / we lack permission."""
    user = untrusted_run_user()
    if not user:
        return
    for p in paths:
        # per-item: one un-chownable path must not abort the rest
        with suppress(LookupError, PermissionError, OSError):
            shutil.chown(p, user=user)


def _copy_agent_references(overlay: Path) -> None:
    """Populate the workspace with only the YAML references SpecGen may read.

    Mirroring repo-relative paths keeps prompt instructions valid without
    granting access to the real repository, its docs, tests, or Python source.
    """
    source = Path(__file__).resolve().parents[1] / "arch"
    target = overlay / "backend" / "arch"

    def _copy_read_only(src_file: Path, dst_dir: Path) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=".specgen-ref-", dir=dst_dir)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            shutil.copyfile(src_file, temp_path)
            temp_path.chmod(0o444)
            os.replace(temp_path, dst_dir / src_file.name)
        finally:
            with suppress(OSError):
                temp_path.unlink()

    # The dim-symbol registry: the vocabulary a spec binds, and the worlds and
    # groups a `dim_symbols:` row names.
    target.mkdir(parents=True, exist_ok=True)
    _copy_read_only(source / "dims.yaml", target)
    for relative in (
        Path("models"),
        Path("components") / "library",
        Path("components") / "adapters",
        Path("components") / "drafts",
    ):
        src = source / relative
        dst = target / relative
        dst.mkdir(parents=True, exist_ok=True)
        expected = set()
        for yaml_path in src.glob("*.yaml"):
            expected.add(yaml_path.name)
            _copy_read_only(yaml_path, dst)
        # A deploy may remove or rename a reference YAML while overlays persist.
        for stale in dst.glob("*.yaml"):
            if stale.name not in expected:
                with suppress(OSError):
                    stale.unlink()


def _prepare(overlay: Path, *, agent_workspace: bool = False,
             mode: str | None = None) -> Path:
    models = overlay / "models"
    drafts = overlay / "components" / "drafts"
    snaps = overlay / "snapshots"
    formal_proposals = [
        overlay / "components" / "formal_proposals" / tier
        for tier in formal_tiers
    ]
    private_dirs = [models, drafts, snaps, *formal_proposals]
    if mode in ("fresh", "seed"):
        # These are authored harvest inputs. A new generate starts with none;
        # a seed also starts with none before writing the one supplied bundle.
        # Neither mode may erase another kind's leaf (the leaf itself is
        # kind-scoped), and seed never performs a second erase after writing.
        for authored in private_dirs:
            shutil.rmtree(authored, ignore_errors=True)
        # The last run's record names a session this mode clears.
        with suppress(OSError):
            (overlay / RUN_END_FILE).unlink()
    elif mode is not None and mode != "resume":
        raise ValueError(
            f"overlay mode must be fresh, seed or resume, not {mode!r}")
    if agent_workspace:
        # CLI state and temp files must not bleed between reruns. Authored YAML
        # is explicitly managed by `mode` above. A resume continues the earlier
        # run: its authored YAML and its CLI state (the agent session) stay.
        cleared = (".tmp",) if mode == "resume" else (".agent-home", ".tmp")
        for name in (".agent-home", ".tmp"):
            if name in cleared:
                shutil.rmtree(overlay / name, ignore_errors=True)
            private_dirs.append(overlay / name)
    for d in private_dirs:
        d.mkdir(parents=True, exist_ok=True)
    overlay.chmod(0o700)
    if agent_workspace:
        _copy_agent_references(overlay)
    # chown the whole freshly-made subtree (leaf-first is fine; also the
    # per-user parent so the agent can traverse it).
    _chown_to_agent(overlay.parent, overlay, models,
                    overlay / "components", drafts, snaps, *private_dirs[3:])
    return overlay


def overlay_leaf_name(model_id: str, *, kind: str = "generate") -> str:
    """Leaf directory under the per-user overlay root. Generate and refine
    (and review__/fix__ elsewhere) each get their own leaf so one in-flight
    agent cannot rmtree another's `.agent-home`."""
    if kind not in ("generate", "refine"):
        raise ValueError(f"overlay kind must be generate or refine, not {kind!r}")
    return f"{kind}__{_safe(model_id)}"


def run_overlay_owner(user_id: str | None = "") -> str:
    """Owner directory under the runs root. Empty/missing user id is the
    reserved anonymous owner, not a shared-tree write. The reserved
    literal is not a user id — callers that pass it as one are rejected."""
    uid = (user_id or "").strip()
    if uid == ANONYMOUS_OWNER:
        raise ValueError(
            f"{ANONYMOUS_OWNER!r} is a reserved overlay owner, not a user id")
    return uid if uid else ANONYMOUS_OWNER


def is_anonymous_overlay(target_root: str | Path | None) -> bool:
    """True when `target_root` is a managed leaf under `__anonymous__`."""
    if not target_root:
        return False
    target = Path(target_root).resolve()
    runs = spec_runs_root().resolve()
    if not target.is_relative_to(runs):
        return False
    rel = target.relative_to(runs)
    return bool(rel.parts) and rel.parts[0] == ANONYMOUS_OWNER


def new_run_overlay(user_id: str, model_id: str, *, kind: str = "generate") -> Path:
    """Create + return a fresh per-(user, model, kind) run overlay root,
    agent-writable. Idempotent: reused across a user's re-runs of the same
    model and kind. Generate and refine do not share a leaf."""
    ov = _prepare(
        _under(spec_runs_root(), _safe(user_id), overlay_leaf_name(model_id, kind=kind)),
        agent_workspace=True,
        mode="fresh",
    )
    with suppress(Exception):  # GC is best-effort; never break a run
        gc_overlays(spec_runs_root())
    return ov


class RunLeafRefusal(LookupError):
    """A generate request the state of its leaf refuses; `refusal()` is the
    response body a route answers with (409)."""

    status = ""

    def __init__(self, model_id: str, message: str):
        super().__init__(message)
        self.model_id = model_id

    def refusal(self) -> dict:
        return {"success": False, "status": self.status, "message": str(self),
                "resumable": self.status == "draft_kept"}


class NoResumableDraft(RunLeafRefusal):
    """A continued generation found no earlier draft in its leaf."""

    status = "no_draft"

    def __init__(self, model_id: str):
        super().__init__(
            model_id,
            f"There is no earlier draft of {model_id} to continue. "
            "Start a new generation.")


class DraftKept(RunLeafRefusal):
    """A new generation would discard the draft an unfinished run kept."""

    status = "draft_kept"

    def __init__(self, model_id: str):
        super().__init__(
            model_id,
            f"An earlier generation of {model_id} stopped before it finished "
            "and kept its draft. Continue it, or start over to discard it.")


def generate_leaf(user_id: str, model_id: str) -> Path:
    """The generate leaf of `(user, model)`; `user_id` "" is the anonymous
    owner, as in `allocate_run_overlay`."""
    return _under(spec_runs_root(), _safe(run_overlay_owner(user_id)),
                  overlay_leaf_name(model_id, kind="generate"))


def has_resumable_draft(target_root: str | Path | None) -> bool:
    """Whether a leaf holds a draft a continued generation picks up: YAML the
    agent authored (a model spec, draft components, formal proposals) left by
    a run that did not succeed. A run that succeeded leaves its spec, not a
    draft; a leaf with no run record (stopped before its session was known)
    counts as unfinished."""
    if not authored_yaml(target_root):
        return False
    end = RunEnd.read(target_root)
    return end is None or not end.success


def is_generate_leaf(target_root: str | Path | None, model_id: str) -> bool:
    """Whether `target_root` is this model's generate leaf."""
    return bool(target_root) and Path(target_root).name == overlay_leaf_name(
        model_id, kind="generate")


def resume_run_overlay(user_id: str, model_id: str) -> Path:
    """Re-open a user's generate leaf to continue its draft: authored YAML and
    the agent's CLI state stay, temp files are cleared. Raises
    `NoResumableDraft` when the leaf holds no draft — checked again after the
    leaf is prepared, so a draft pruned in between is refused rather than
    continued as an empty workspace."""
    leaf = generate_leaf(user_id, model_id)
    if not has_resumable_draft(leaf):
        raise NoResumableDraft(model_id)
    ov = _prepare(leaf, agent_workspace=True, mode="resume")
    if not has_resumable_draft(ov):
        raise NoResumableDraft(model_id)
    with suppress(Exception):  # GC is best-effort; never break a run
        gc_overlays(spec_runs_root())
    return ov


RUN_END_FILE = ".resume.json"


@dataclass(frozen=True)
class RunEnd:
    """How the last generate run in a leaf ended, kept beside its draft: the
    agent CLI and session it ran (a continued generation re-enters that
    session), whether it succeeded, its closing message, whether a timer
    stopped it, and how its agent's last invocation ended (`agent_end`, the
    `AgentEnd` the `done` carried, kept as the word it wrote; None while the
    run is unfinished, and in a record written before the field existed).
    What an ending means for a continuation is `resume`'s to say — this is
    the record of it."""

    success: bool
    message: str
    timed_out: bool
    session_id: str | None
    agent_cli: str | None
    ended_at: float
    agent_end: str | None = None

    @classmethod
    def unfinished(cls, prep_result: dict, message: str) -> "RunEnd":
        """The record of a run still going, or one that stopped without a
        terminal `done`: written as soon as its agent session is known, so a
        run the server's restart cuts off can still be re-entered."""
        return cls(
            success=False, message=message, timed_out=False,
            session_id=prep_result.get("_session_id"),
            agent_cli=prep_result.get("_agent_cli"),
            ended_at=time.time(),
        )

    @classmethod
    def from_done(cls, event: dict, prep_result: dict) -> "RunEnd":
        return cls(
            success=bool(event.get("success")),
            message=str(event.get("message") or ""),
            timed_out=bool(event.get("timed_out")),
            session_id=prep_result.get("_session_id"),
            agent_cli=prep_result.get("_agent_cli"),
            ended_at=time.time(),
            # A `done` without `agent_end` is one whose CLI finished its turn.
            agent_end=str(event.get("agent_end") or "") or None,
        )

    def write(self, root: str | Path) -> None:
        """Best-effort: a run's outcome never depends on this record."""
        with suppress(OSError):
            _atomic_write_text(Path(root) / RUN_END_FILE,
                               json.dumps(asdict(self)))

    @classmethod
    def read(cls, root: str | Path) -> "RunEnd | None":
        """The leaf's record; None when it has none. An unreadable record
        is logged and counts as none: the continuation then cold-starts."""
        path = Path(root) / RUN_END_FILE
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                success=bool(raw["success"]),
                message=str(raw.get("message") or ""),
                timed_out=bool(raw.get("timed_out")),
                session_id=raw.get("session_id") or None,
                agent_cli=raw.get("agent_cli") or None,
                ended_at=float(raw.get("ended_at") or 0.0),
                agent_end=str(raw.get("agent_end") or "") or None,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("unreadable run record %s: %s", path, exc)
            return None


def allocate_run_overlay(
    model_id: str,
    *,
    kind: str,
    user_id: str = "",
    seed_bundle: NormalizedBundle | None = None,
    resume: bool = False,
    discard_draft: bool = False,
) -> str:
    """Create the isolated generate/refine leaf for this run and return it.

    Per-user runs own `<user>/<kind>__<model>`. Self-host / per-user-off /
    core anonymous runs own `__anonymous__/<kind>__<model>` so two in-flight
    kinds never share a write root. A refine seed is one normalized bundle;
    generate leaves start empty of authored yaml. `resume` re-opens the
    generate leaf with its draft and agent session (`resume_run_overlay`).
    A new generate over a draft an unfinished run kept raises `DraftKept`
    unless the caller chose to discard it (`discard_draft`).
    """
    owner = run_overlay_owner(user_id)
    if resume:
        if kind != "generate" or seed_bundle is not None:
            raise ValueError("only a generate run continues an earlier draft")
        return str(resume_run_overlay(user_id, model_id))
    if (kind == "generate" and seed_bundle is None and not discard_draft
            and has_resumable_draft(generate_leaf(user_id, model_id))):
        raise DraftKept(model_id)
    if seed_bundle is not None:
        return seed_run_overlay(owner, model_id, seed_bundle, kind=kind)
    return str(new_run_overlay(owner, model_id, kind=kind))


def load_seed_spec_yaml(model_id: str) -> str | None:
    """YAML to seed a refine overlay: custom cache spec, else the currently
    resolved shared/repo spec. None when nothing is on disk to edit.

    An unnamed spec's name defaults to its filename stem, so the overlay copy
    (`<safe_model_id>.yaml`) would get a different name than the shared file
    and not supersede it. Pin an explicit `name:` matching the resolved spec
    so the overlay copy dedups over the shared one.
    """
    if is_custom_id(model_id):
        path = custom_model_path(strip_prefix(model_id)) / "spec.yaml"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return None
    import yaml as _yaml
    config = fetch_config(model_id)
    if config is None:
        logger.info("no refine seed for %s: config unavailable", model_id)
        return None
    try:
        resolved = SpecLoader().match(config, model_id=model_id)
        if resolved is None:
            return None
        spec = resolved.spec
        src = spec.source_path
        if not (src and os.path.isfile(src)):
            return None
        with open(src, encoding="utf-8") as f:
            text = f.read()
        raw = load_yaml(text)
        if isinstance(raw, dict) and "name" not in raw and spec.name:
            text = f"name: {spec.name}\n" + text
        return text
    except (OSError, _yaml.YAMLError, ValueError) as exc:
        logger.info("no refine seed for %s: %s", model_id, exc)
        return None


def _land_lock(model_id: str) -> threading.Lock:
    """In-process mutex for one serving target. Generate/refine run locks are
    already split by kind; this is the short land/publish mutex S4 §2.5
    describes, not a second DB lock."""
    key = _safe(model_id)
    with _land_guard:
        lock = _land_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _land_locks[key] = lock
        return lock


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` via a same-directory temp file then `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".land-", suffix=".tmp", dir=path.parent)
    tmp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        tmp.chmod(0o644)
        os.replace(tmp, path)
    finally:
        with suppress(OSError):
            tmp.unlink()


def _atomic_copyfile(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".land-", suffix=".tmp", dir=dest.parent)
    tmp = Path(temp_name)
    try:
        os.close(fd)
        shutil.copyfile(src, tmp)
        tmp.chmod(0o644)
        os.replace(tmp, dest)
    finally:
        with suppress(OSError):
            tmp.unlink()


def land_anonymous_overlay(
    model_id: str,
    target_root: str | Path | None,
    spec_yaml: str,
    new_component_yamls=None,
    snapshot_src: Path | None = None,
) -> None:
    """Copy a finished anonymous custom run to its dedicated local cache.

    Hub model identities never land in ``backend/arch``: their shared tier is
    the DB/local shared cache. ``custom:`` ids are the exception because the
    custom loader resolves that cache as part of the id itself. Same-target
    lands are serialized in-process and use temp+rename.
    """
    if (not is_anonymous_overlay(target_root) or not spec_yaml
            or not is_custom_id(model_id)):
        return
    with _land_lock(model_id):
        if is_custom_id(model_id):
            dest = custom_model_path(strip_prefix(model_id)) / "spec.yaml"
            _atomic_write_text(dest, spec_yaml)
            return
        repo_arch = Path(__file__).resolve().parents[1] / "arch"
        models = repo_arch / "models"
        _atomic_write_text(models / f"{_safe(model_id)}.yaml", spec_yaml)
        drafts = repo_arch / "components" / "drafts"
        for comp in (new_component_yamls or []):
            body = comp.get("body") if isinstance(comp, dict) else None
            if not body:
                continue
            name = os.path.basename((comp.get("path") or "").strip()) or "component.yaml"
            _atomic_write_text(drafts / name, body)
        if snapshot_src is not None:
            src = Path(snapshot_src)
            if src.is_file():
                _atomic_copyfile(src, snapshot_path(model_id))


# ---------------------------------------------------------------------------
# Bounded-disk GC. Both roots are PURE CACHE — every overlay is rebuildable
# from the `spec_generations` row — so we can prune the oldest leaf dirs
# without a cron. Called opportunistically on overlay creation.
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    with suppress(ValueError, TypeError):
        return int(os.environ.get(name, "").strip() or default)
    return default


def _recent_mtime(leaf: Path) -> float:
    """Most-recent mtime of a leaf overlay, sampling the subdirs the agent
    writes YAML into (their mtime bumps on each file write, unlike the leaf
    dir's own) and every entry under `.agent-home`: a round that only reads
    and talks to its model still appends to its CLI session transcript
    there. The CLI home is small, so walking it stays cheap."""
    best = 0.0
    for p in (leaf, leaf / "models", leaf / "snapshots",
              leaf / "components" / "drafts",
              *(leaf / "components" / "formal_proposals" / tier
                for tier in formal_tiers)):
        with suppress(OSError):
            best = max(best, p.stat().st_mtime)
    for dirpath, _dirnames, filenames in os.walk(leaf / ".agent-home"):
        for p in (dirpath, *(os.path.join(dirpath, f) for f in filenames)):
            with suppress(OSError):
                best = max(best, os.stat(p).st_mtime)
    return best


def gc_overlays(root: Path | None = None, *, max_entries: int | None = None,
                min_age_s: int | None = None) -> int:
    """Prune the oldest per-(user, model) overlay leaves beyond a cap, skipping
    any touched within `min_age_s`. The default is the spec lock TTL
    (`limits.LOCK_TTL_SEC`), the longest a run can hold its leaf, so an
    in-flight run is never pruned under its agent. Pure cache: rebuildable from
    spec_generations. `root=None` GCs both roots. Returns dirs removed."""
    cap = max_entries if max_entries is not None else _env_int("HA_SPEC_OVERLAY_MAX", 200)
    min_age = min_age_s if min_age_s is not None else _env_int(
        "HA_SPEC_OVERLAY_MIN_AGE_S", LOCK_TTL_SEC)
    roots = [root] if root is not None else [spec_runs_root(), user_specs_root()]
    now = time.time()
    removed = 0
    for r in roots:
        removed += _gc_one(r, cap, min_age, now)
    return removed


def _gc_one(root: Path, cap: int, min_age: float, now: float) -> int:
    if not root.exists():
        return 0
    leaves: list[tuple[float, Path]] = []  # (mtime, <root>/<user>/<model>)
    with suppress(OSError):
        for user_dir in root.iterdir():
            if not user_dir.is_dir():
                continue
            with suppress(OSError):  # one bad user dir must not abort the rest
                leaves.extend((_recent_mtime(md), md)
                              for md in user_dir.iterdir() if md.is_dir())
    if len(leaves) <= cap:
        return 0
    leaves.sort(key=lambda t: t[0])  # oldest first
    removed = 0
    for mtime, d in leaves[:len(leaves) - cap]:
        if now - mtime < min_age:
            continue  # protect a fresh / in-flight overlay
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
        with suppress(OSError):  # drop the now-empty <user> parent
            d.parent.rmdir()
    return removed


def _write_spec_into(ov: Path, bundle: NormalizedBundle | BlockedBundle) -> None:
    """Write one normalized bundle's serving view into a prepared overlay.

    The spec is stamped with its `match.model_id` on the way in: the loader
    resolves by model_id, and a stored spec_yaml (esp. one generated before
    self-registration existed) may carry only `match.model_type` — unstamped it
    would not resolve out of the overlay (review/analyze/refine would see "No
    spec matched"). `stamp_match_model_id` is a no-op when already pinned.
    Drafts are the only component files a loader can see. Other bundle files
    (notably op proposals) stay record/PR material, and formal component
    proposals are deliberately absent here. A `BlockedBundle` is the record
    as saved, so its agent base models land beside the identity (a
    normalized bundle has flattened them away) and a draft saved under the
    retired root lands with the others.
    """
    model_id = bundle.model_id
    models = overlay_models_dir(ov)
    (models / f"{_safe(model_id)}.yaml").write_text(
        stamp_match_model_id(bundle.spec_yaml, model_id), encoding="utf-8")
    drafts = overlay_drafts_dir(ov)
    for file in bundle.serving_files:
        name = PurePosixPath(file.path).name
        if file.draft_name is not None:
            (drafts / name).write_text(file.body, encoding="utf-8")
        elif file.model_name not in (None, _safe(model_id)):
            (models / name).write_text(file.body, encoding="utf-8")


def _seed_formal_proposals(ov: Path,
                           bundle: NormalizedBundle | BlockedBundle) -> None:
    """Re-seed PR-only formal proposals in a non-loader authoring area."""
    for file in bundle.proposal_files:
        pure = PurePosixPath(file.path)
        if len(pure.parts) != 5 or pure.parts[:3] != (
                "backend", "arch", "components") or pure.parts[3] not in formal_tiers:
            continue
        tier = pure.parts[3]
        dest = ov / "components" / "formal_proposals" / tier / pure.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(file.body, encoding="utf-8")


def materialize_user_spec(user_id: str, model_id: str,
                          bundle: NormalizedBundle,
                          snapshot_json: dict | None = None) -> str:
    """Write a stored spec + its draft components, and the snapshot it was
    judged against, into a per-model serving overlay (cleared fresh each
    call) and return the overlay root as a string. Serving reads the config
    and checkpoint record from that snapshot, the same one validate read, so
    the page and the verdict compare against one record. Pure cache — rebuilt
    on demand from spec_generations, so a stale or missing overlay is never
    authoritative."""
    root = _under(user_specs_root(), _safe(user_id), _safe(model_id))
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    ov = _prepare(root)
    _write_spec_into(ov, bundle)
    if snapshot_json:
        store_snapshot(model_id, snapshot_json, str(ov))
    return str(ov)


def seed_run_overlay(user_id: str, model_id: str, bundle: NormalizedBundle,
                     *, kind: str = "refine") -> str:
    """Seed a refine workspace with one normalized bundle."""
    if bundle.model_id != model_id:
        raise ValueError("seed bundle model_id does not match the run")
    ov = _prepare(
        _under(spec_runs_root(), _safe(user_id), overlay_leaf_name(model_id, kind=kind)),
        agent_workspace=True,
        mode="seed",
    )
    _write_spec_into(ov, bundle)
    _seed_formal_proposals(ov, bundle)
    with suppress(Exception):
        gc_overlays(spec_runs_root())
    return str(ov)


def seed_review_overlay(reviewer_id: str, record_id: str, model_id: str,
                        bundle: NormalizedBundle) -> str:
    """Create a fresh RUN overlay for an agent REVIEW and seed it with the spec
    under review so the read-only review agent has it to inspect + resolve via
    its in-loop `validate`/`structure`. Keyed on `(reviewer, review__<record>)`
    — a distinct leaf from generate__ / refine__ overlays, so a review never
    clobbers an in-flight generation. Returns the overlay root."""
    ov = _prepare(
        _under(spec_runs_root(), _safe(reviewer_id), f"review__{_safe(record_id)}"),
        agent_workspace=True,
        mode="seed",
    )
    with suppress(Exception):  # GC is best-effort; never break a run
        gc_overlays(spec_runs_root())
    _write_spec_into(ov, bundle)
    _seed_formal_proposals(ov, bundle)
    return str(ov)


def seed_fix_overlay(fixer_id: str, record_id: str, model_id: str,
                     bundle: NormalizedBundle | BlockedBundle) -> str:
    """Create a fresh WRITABLE RUN overlay for an agent FIX and seed it with the
    record's current draft so the fix agent has it to read + edit in place. Keyed
    on `(fixer, fix__<record>)` — a distinct leaf from the review overlay
    (`review__<record>`) and the per-(user, model) generate/refine overlay, so a
    fix never clobbers an in-flight review or generation. The post-success
    harvest reads the fixed yaml back from this same overlay (target_root).
    Returns the overlay root."""
    ov = _prepare(
        _under(spec_runs_root(), _safe(fixer_id), f"fix__{_safe(record_id)}"),
        agent_workspace=True,
        mode="seed",
    )
    with suppress(Exception):  # GC is best-effort; never break a run
        gc_overlays(spec_runs_root())
    _write_spec_into(ov, bundle)
    _seed_formal_proposals(ov, bundle)
    return str(ov)


def seed_recheck_overlay(actor_id: str, record_id: str, model_id: str,
                         bundle: NormalizedBundle,
                         snapshot_json: dict | None) -> str:
    """A fresh overlay for a deterministic re-check of a stored record: its
    bundle, and the snapshot it was generated against, so validate reads the
    same config and checkpoint index. No agent runs here. Keyed on `(actor,
    recheck__<record>)`, apart from every other leaf. Returns the root."""
    ov = _prepare(
        _under(spec_runs_root(), _safe(actor_id), f"recheck__{_safe(record_id)}"),
        mode="seed",
    )
    with suppress(Exception):  # GC is best-effort; never break a run
        gc_overlays(spec_runs_root())
    _write_spec_into(ov, bundle)
    if snapshot_json:
        store_snapshot(model_id, snapshot_json, str(ov))
    return str(ov)


def stage_forward_source(overlay: str | Path, files: list) -> list[str]:
    """Write complete module source files into `<overlay>/source/` so a
    sandboxed agent can read and cite them by line. The ground-truth prompt
    only carries excerpts (selected `forward` bodies, no line numbers, no
    `Model.forward`, no configuration class); these are the whole files.

    `files` (`[{path, text}]`) must be real module files — see
    `_ground_truth.full_module_sources`; each file's on-disk sibling
    `configuration_*.py` is added so config-class defaults are citable.
    Returns the staged absolute paths. A write failure raises: no agent runs
    without the source it is told to cite. Every spec-gen agent reads them at
    the same path: claude's isolated run via its `Read(<workspace>/source/**)`
    rule; codex / kimi use the same overlay as cwd without a path-scoped
    policy."""
    return _write_forward_source(Path(overlay) / "source", files)


def _write_forward_source(dest: Path, files: list) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    for f in files:
        path, text = f.get("path") or "", f.get("text") or ""
        if text and path:
            staged.setdefault(Path(path).name, text)
        parent = Path(path).parent if path else None
        if parent and parent.is_dir():
            for cfg in sorted(parent.glob("configuration_*.py")):
                with suppress(OSError):
                    staged.setdefault(cfg.name, cfg.read_text(encoding="utf-8"))
    out = []
    for name, text in staged.items():
        target = dest / name
        with suppress(OSError):
            target.unlink()
        target.write_text(text, encoding="utf-8")
        target.chmod(0o444)
        out.append(str(target))
    _chown_to_agent(dest)
    return out


def stage_vllm_source(overlay: str | Path, files) -> list[str]:
    """Write vLLM's model files (`hf.vllm_source.VllmModelSource.files`)
    under `<overlay>/source/` at their path in the vLLM repo
    (`source/vllm/models/hy_v4/nvidia/mtp.py`), read-only, beside the
    modeling files `stage_forward_source` writes: the agents read and cite
    them the same way. Returns the staged absolute paths."""
    dest = (Path(overlay) / "source").resolve()
    out = []
    for f in files:
        target = (dest / f["path"]).resolve()
        if dest not in target.parents:
            raise ValueError(f"vLLM source path {f['path']!r} leaves the source dir")
        target.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            target.unlink()
        target.write_text(f["text"], encoding="utf-8")
        target.chmod(0o444)
        out.append(str(target))
    _chown_to_agent(dest)
    return out


def overlay_models_dir(overlay: Path) -> Path:
    return overlay / "models"


def overlay_drafts_dir(overlay: Path) -> Path:
    return overlay / "components" / "drafts"



def authored_yaml(target_root: str | Path | None) -> list[str]:
    """The YAML an agent authored in a leaf — model specs, draft components,
    formal proposals — workspace-relative. The reference tree under
    `backend/arch` is not authored."""
    if not target_root:
        return []
    root = Path(target_root)
    dirs = [overlay_models_dir(root), overlay_drafts_dir(root),
            *(root / "components" / "formal_proposals" / tier
              for tier in formal_tiers)]
    return [path.relative_to(root).as_posix()
            for directory in dirs
            for path in sorted(directory.glob("*.yaml")) if path.is_file()]


def authored_files(target_root: str | Path | None) -> list[str]:
    """`authored_yaml` plus the config a model written from a description
    keeps in the leaf's snapshot (`snapshots/custom:*.json`, the one
    snapshot an agent may write)."""
    if not target_root:
        return []
    root = Path(target_root)
    return [*authored_yaml(root),
            *(path.relative_to(root).as_posix()
              for path in sorted((root / "snapshots").glob("custom:*.json"))
              if path.is_file())]


def continue_from_record(user_id: str, model_id: str,
                         bundle: NormalizedBundle,
                         snapshot_json: dict | None) -> str:
    """Re-open a user's generate leaf for the next turn on a model they keep:
    the agent session stays (the turn re-enters it), and the authored files
    are replaced by the stored record — which holds any edit the user made
    by hand since, and outlives a leaf the GC reaped. Returns the leaf."""
    leaf = generate_leaf(user_id, model_id)
    ov = _prepare(leaf, agent_workspace=True, mode="resume")
    for authored in (overlay_models_dir(ov), overlay_drafts_dir(ov),
                     ov / "snapshots"):
        shutil.rmtree(authored, ignore_errors=True)
        authored.mkdir(parents=True, exist_ok=True)
    _chown_to_agent(overlay_models_dir(ov), overlay_drafts_dir(ov),
                    ov / "snapshots")
    _write_spec_into(ov, bundle)
    if snapshot_json:
        store_snapshot(model_id, snapshot_json, str(ov))
    # The turn edits these in place, as the agent user.
    _chown_to_agent(*(path for authored in (overlay_models_dir(ov),
                                            overlay_drafts_dir(ov),
                                            ov / "snapshots")
                      for path in authored.iterdir()))
    with suppress(Exception):  # GC is best-effort; never break a run
        gc_overlays(spec_runs_root())
    return str(ov)


def scratch_user_spec(user_id: str, model_id: str, bundle: NormalizedBundle,
                      snapshot_json: dict | None) -> str:
    """A throwaway overlay holding one candidate of a user's model — a hand
    edit to validate before it is kept. Not the serving overlay
    (`materialize_user_spec`), which readers are using: a candidate that
    fails must never be what they read. The caller removes it."""
    scratch = _under(user_specs_root(), "_scratch", _safe(user_id))
    scratch.mkdir(parents=True, exist_ok=True)
    ov = _prepare(Path(tempfile.mkdtemp(prefix=f"{_safe(model_id)}.",
                                        dir=scratch)))
    _write_spec_into(ov, bundle)
    if snapshot_json:
        store_snapshot(model_id, snapshot_json, str(ov))
    return str(ov)


def resolve_run_workspace(target_root: str | Path | None) -> str:
    """Validate and normalize the current run overlay as an agent workspace.

    Exposing the roots of both stores made every user's artifacts visible to
    every other run because all agents share one OS account. Resolve symlinks
    and refuse unless the target is a managed overlay leaf.
    """
    if not target_root:
        raise ValueError("An isolated SpecGen run requires one workspace")
    try:
        target = Path(target_root).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"SpecGen workspace does not exist: {target_root}") from exc
    roots = (spec_runs_root().resolve(), user_specs_root().resolve())
    managed_leaf = any(
        target.is_relative_to(root)
        and len(target.relative_to(root).parts) == 2
        for root in roots
    )
    if not managed_leaf:
        raise ValueError(f"SpecGen workspace is outside managed overlay roots: {target}")
    return str(target)
