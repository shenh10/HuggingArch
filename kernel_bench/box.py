"""Machine-local execution facts for kernel benchmark runs."""
from __future__ import annotations

import os
import socket
import string
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 (CI) — tomli is pinned in uv.lock
    import tomli as tomllib


if TYPE_CHECKING:
    from .env import Env


class BoxProfileError(ValueError):
    """Raised when a box profile is missing or malformed."""


@dataclass(frozen=True)
class BoxProfile:
    path: Path
    name: str
    python: Optional[str]
    constraints: Optional[str]
    command_prefix: Tuple[str, ...]
    env: Mapping[str, str]
    providers: Mapping[str, str]
    # Per-library env file override (`[env_requirements] fa2 = "..."`) — a
    # box whose torch/CUDA stack only matches a PREBUILT wheel (no CUDA
    # toolkit on the box) pins that wheel here instead of the generated
    # env's source-build spec. The file replaces `envs/<lib>.txt` for this
    # box only, AND it replaces the box-wide `constraints`: a box that states
    # one env's requirements in full is stating them in full, and layering the
    # default torch pin on top is what made a second wheel stack unresolvable
    # (`torch==2.8.0+cu128` from the box vs `torch==2.13.0` from vLLM).
    env_requirements: Mapping[str, str] = None
    #: Per-library environment overlay (`[library_env.<lib>] VAR = "..."`) —
    #: `[env]` says what every env on this box runs in; this says what ONE
    #: env runs in, for a box where that is not the same thing.
    #:
    #: The case that needed it: a box whose kernel driver tops out at CUDA
    #: 12.9 still runs CUDA-13 wheels, through NVIDIA's forward-compatibility
    #: user-mode driver on `$LD_LIBRARY_PATH`. Which user-mode driver an env
    #: loads is a property of that env's wheels, and putting it in `[env]`
    #: would silently change it for the CUDA-12 envs too — re-timing every
    #: row they go on to measure. Keeping it scoped here is still right, for
    #: that reason alone.
    #:
    #: What this no longer has to carry is the ROW's side of it. Since schema
    #: 14 every row records the driver it actually ran under
    #: (`kmd_version` / `umd_version`, observed from `/proc`), so the driver
    #: is a fact ON the row and is NOT inferred from this declaration. Do not
    #: reconstruct an env-to-overlay mapping from `[library_env]` to answer
    #: "which driver measured that row" — that is the framework-to-overlay
    #: second table §1.4 forbids, and it would be wrong whenever an
    #: `$LD_LIBRARY_PATH` failed to take effect or this file was edited after
    #: the row was written. Ask the row.
    #:
    #: Values are `${VAR}`-expanded from the running environment like `[env]`,
    #: so a machine-local path is written `${KERNEL_BENCH_WORK_DIR}/…`.
    library_env: Mapping[str, Mapping[str, str]] = None
    #: The machine's scratch disk, where the measuring loop keeps its queue
    #: and its results (`jobs/`, `results/`, `queue.log`). Node-local — a
    #: reallocation wipes it, which is why none of what runs there may live
    #: only there. None when the profile states none, and the commands that
    #: need one say so.
    #:
    #: A profile names a HARDWARE + wheel stack (`h100-sxm-cu128`), not a
    #: machine, so this and the `[providers]` venv paths beside it are
    #: written as `${KERNEL_BENCH_WORK_DIR}/…` and expanded from the
    #: environment of whichever machine with that stack is running: two
    #: H100 boxes share the profile and differ only in where their disk is
    #: mounted. A literal path here put one allocated machine's name in the
    #: repository and went stale the first time the box was reallocated.
    work_dir: Optional[str] = None
    #: How many cards the box has (`devices = 2`). A run drives the ones
    #: it is shown (`CUDA_VISIBLE_DEVICES`), never more than these
    #: (`BoxEnvs.devices`); a collective cell's group is judged against that
    #: (`Env.devices`). A box that states none has one.
    devices: int = 1


def _optional_string(data: Mapping[str, object], key: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise BoxProfileError(f"{key} must be a non-empty string")
    return value


def _string_table(data: Mapping[str, object], key: str) -> Dict[str, str]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise BoxProfileError(f"[{key}] must be a TOML table")
    bad = [name for name, item in value.items()
           if not isinstance(name, str) or not isinstance(item, str)]
    if bad:
        raise BoxProfileError(f"[{key}] values must be strings: {bad}")
    # An empty value is not a declaration, here or in `[library_env]`: written
    # down it looks deliberate and behaves like absence — `env_requirements`
    # with `vllm = ""` resolves the generated env under the box constraints,
    # which is exactly what naming an override was meant to stop.
    empty = sorted(name for name, item in value.items() if not item.strip())
    if empty:
        raise BoxProfileError(
            f"[{key}] states nothing for {empty}; remove the line or give it "
            f"a value — an empty string reads as a decision and acts as none")
    return dict(value)


def _nested_string_table(data: Mapping[str, object], key: str) -> Dict[str, Dict[str, str]]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise BoxProfileError(f"[{key}] must be a TOML table")
    out: Dict[str, Dict[str, str]] = {}
    for name, item in value.items():
        if not isinstance(item, dict):
            raise BoxProfileError(
                f"[{key}.{name}] must be a TOML table of VAR = \"value\"")
        bad = [var for var, text in item.items() if not isinstance(text, str)]
        if bad:
            raise BoxProfileError(f"[{key}.{name}] values must be strings: {bad}")
        empty = sorted(var for var, text in item.items() if not text.strip())
        if empty:
            raise BoxProfileError(
                f"[{key}.{name}] states nothing for {empty}; an empty value "
                f"reads as a decision and acts as none — say what the variable "
                f"is, or leave it out and inherit `[env]`")
        out[str(name)] = dict(item)
    return out


def load_box_profile(path: str | os.PathLike[str]) -> BoxProfile:
    profile_path = Path(path)
    try:
        with profile_path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise BoxProfileError(f"box profile not found: {profile_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise BoxProfileError(f"invalid box profile {profile_path}: {exc}") from exc

    allowed = {"name", "python", "constraints", "command_prefix", "env",
               "providers", "env_requirements", "library_env", "work_dir",
               "devices"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise BoxProfileError(f"unknown box profile field(s): {unknown}")

    raw_prefix = data.get("command_prefix", [])
    if (not isinstance(raw_prefix, list)
            or any(not isinstance(item, str) or not item for item in raw_prefix)):
        raise BoxProfileError("command_prefix must be an array of non-empty strings")
    devices = data.get("devices", 1)
    if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
        raise BoxProfileError(
            f"devices must be a positive integer, got {devices!r}")

    return BoxProfile(
        path=profile_path,
        name=_optional_string(data, "name") or profile_path.stem,
        python=_optional_string(data, "python"),
        constraints=_optional_string(data, "constraints"),
        command_prefix=tuple(raw_prefix),
        env=_string_table(data, "env"),
        providers=_string_table(data, "providers"),
        env_requirements=_string_table(data, "env_requirements"),
        library_env=_nested_string_table(data, "library_env"),
        work_dir=_optional_string(data, "work_dir"),
        devices=devices,
    )


def find_box_profile(explicit: Optional[str], envs_dir: str,
                     environ: Mapping[str, str] = os.environ,
                     hostname: Optional[str] = None) -> Optional[BoxProfile]:
    """Resolve an explicit profile or the current host's conventional file."""
    requested = explicit or environ.get("KERNEL_BENCH_BOX")
    boxes_dir = Path(envs_dir) / "boxes"
    if requested:
        # A queue is handed the profile the way supervise.sh spells it — a
        # path relative to the repo root — while its own CWD (and every
        # child env's) is the work dir, not the repo. Resolve against the
        # places the profile can live before reporting anything, so a missing
        # file says what was tried instead of a doubled path nobody passed.
        name = requested if requested.endswith(".toml") else f"{requested}.toml"
        root = Path(envs_dir).resolve().parents[1]
        candidates = (Path(requested), root / requested, boxes_dir / Path(name).name)
        for candidate in candidates:
            if candidate.is_file():
                return load_box_profile(candidate)
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise BoxProfileError(f"box profile not found: {requested} (tried {tried})")

    host_path = boxes_dir / f"{hostname or socket.gethostname()}.toml"
    return load_box_profile(host_path) if host_path.is_file() else None


def _overlay(merged: Dict[str, str], values: Mapping[str, str]) -> Dict[str, str]:
    for name, value in values.items():
        expanded = string.Template(value).safe_substitute(merged)
        merged[name] = os.path.expanduser(expanded)
    return merged


def apply_profile_env(base_env: Mapping[str, str],
                      profile: Optional[BoxProfile]) -> Dict[str, str]:
    """Overlay profile variables, expanding earlier environment variables."""
    merged = dict(base_env)
    if profile is None:
        return merged
    return _overlay(merged, profile.env)


def apply_library_env(base_env: Mapping[str, str],
                      profile: Optional[BoxProfile],
                      library: str) -> Dict[str, str]:
    """`base_env` plus what the box states for THIS library's env only.

    A library the box says nothing about gets `base_env` unchanged, so the
    overlay costs nothing to not use.
    """
    merged = dict(base_env)
    if profile is None or not profile.library_env:
        return merged
    return _overlay(merged, profile.library_env.get(library, {}))


def library_requirements(profile: Optional[BoxProfile], library: str,
                         envs_dir: str | os.PathLike[str],
                         *, asked_constraints: Optional[str] = None
                         ) -> Tuple[str, Optional[str]]:
    """Which requirements files materialize `library`'s env on this box:
    `(env_file, constraints)`, the second None when none applies.

    The profile's `constraints` is the box's DEFAULT torch build — the pin for
    the envs that do not state one themselves — and an `[env_requirements]`
    override states one env in full, so for that library the default steps
    aside. Both at once is how a box with a SECOND wheel stack becomes
    unresolvable: the default pins `torch==2.8.0+cu128`, vLLM 0.32.0 pins
    `torch==2.13.0`, and no resolution satisfies both.

    `asked_constraints` — what an operator passed as `--constraints` — is not
    that default and does not step aside for anything. It is the reach for when
    a driver is misbehaving ("every env on this box, this torch, now"), and an
    env that cannot satisfy it must fail to resolve rather than quietly resolve
    without it: a silently ignored emergency pin is worse than no flag at all.
    So an explicit file goes to every env, overrides included.

    One place, because `run-all` asks the same question three times (resolve,
    measure, `up`) and the three copies of the expression were already three
    lines that had to be kept in step by hand.
    """
    override = (profile.env_requirements.get(library)
                if profile and profile.env_requirements else None)
    env_file = override or str(Path(envs_dir) / f"{library}.txt")
    if asked_constraints:
        return env_file, asked_constraints
    if override:
        return env_file, None
    return env_file, (profile.constraints if profile else None)


def _pins(paths: Sequence[Optional[str]], root: Path) -> Dict[str, str]:
    """`package -> version` for every `==` pin in these requirements files
    and the files they `-r` include — what uv will resolve the env from,
    read for the versions it cannot choose. Paths are as `library_
    requirements` gives them: relative to the repo, includes relative to the
    including file."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    out: Dict[str, str] = {}
    todo = [Path(p) if Path(p).is_absolute() else root / p for p in paths if p]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        if not path.is_file():
            # A requirements file the box names and nobody wrote is a
            # declaration error, not "nothing pinned".
            raise BoxProfileError(f"requirements file not found: {path}")
        seen.add(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line.startswith(("-r ", "-c ")):
                todo.append(path.parent / line[3:].strip())
                continue
            try:
                req = Requirement(line)
            except InvalidRequirement:
                continue
            exact = [s.version for s in req.specifier if s.operator == "=="]
            if len(exact) == 1:
                out[canonicalize_name(req.name)] = exact[0]
    return out


def gpu_of(profile: Optional[BoxProfile]) -> Optional[str]:
    """The SKU a box measures on, from its profile name (`h100-sxm-cu128`
    measures `H100-SXM`); None without a profile or a matching SKU."""
    if profile is None:
        return None
    from backend.inference.gpu_specs import load_gpu_specs
    name = (profile.name or "").lower()
    best = None
    for sku in load_gpu_specs():
        # Every part of the SKU's name, not just its family: `h100` alone
        # made `h100-sxm-cu128` an H100-PCIe (the longer name won).
        parts = sku.sku.lower().split("-")
        if all(part in name for part in parts) and (
                best is None or len(sku.sku) > len(best)):
            best = sku.sku
    return best


@dataclass(frozen=True)
class BoxEnvs:
    """The envs one box runs, from everything that decides them: the profile,
    the card, the env files, an operator's `--constraints`, and where each
    source build's interpreter is (`--env`, the profile's `[providers]`,
    `KERNEL_BENCH_ENV_<LIB>`). One object for the plan that judges jobs
    against these envs and for the `run-all` that starts them, so the two
    cannot read different answers to "which envs does this box have"."""
    profile: Optional[BoxProfile]
    gpu: Optional[str]
    envs_dir: str
    overrides: Mapping[str, str]
    asked_constraints: Optional[str]
    environ: Mapping[str, str]
    root: Path

    @classmethod
    def of(cls, profile: Optional[BoxProfile], envs_dir: str, *,
           gpu: Optional[str] = None,
           overrides: Optional[Mapping[str, str]] = None,
           asked_constraints: Optional[str] = None,
           environ: Optional[Mapping[str, str]] = None,
           root: Optional[Path] = None) -> "BoxEnvs":
        """The box's own card unless one is named, and a named one must BE
        the box's: judging an H100's kernels on an RTX 4090's compute
        capability writes every sm_90 kernel's cells off as gaps."""
        own = gpu_of(profile)
        if gpu and own and gpu != own:
            raise BoxProfileError(
                f"--gpu {gpu} is not the card of box {profile.name} ({own})")
        return cls(profile, gpu or own, str(envs_dir), dict(overrides or {}),
                   asked_constraints,
                   dict(environ if environ is not None
                        else apply_profile_env(os.environ, profile)),
                   root or Path(__file__).resolve().parents[1])

    def python_for(self, lib: str) -> Optional[str]:
        """The interpreter of a source-build env, or None when this box has
        not provisioned it."""
        declared = ((self.profile.providers or {}).get(lib)
                    if self.profile else None)
        return (self.overrides.get(lib)
                or (expand_box_path(declared, self.environ) if declared else None)
                or self.environ.get(f"KERNEL_BENCH_ENV_{lib.upper()}"))

    def requirements(self, lib: str) -> Tuple[str, Optional[str]]:
        return library_requirements(self.profile, lib, self.envs_dir,
                                    asked_constraints=self.asked_constraints)

    def premises(self, manifest: Mapping[str, Mapping[str, object]]) -> str:
        """What a judgement against these envs rests on, as a short digest a
        plan records on every job (`judged_on`): a run whose own premises
        differ asks the question again instead of trusting the answer.

        The FACTS `envs` judges with, not the paths and names that lead to
        them: the card, and per library whether its env exists and the
        versions its requirements pin (file contents, so an edited pin or a
        different `--envs-dir` is a different premise and an aliased path is
        not). A requirements file that is missing is recorded as missing;
        judging against it raises."""
        import hashlib
        import json
        facts: Dict[str, object] = {"gpu": self.gpu}
        if self.devices != 1:
            # Stated only when it is not one card, so a plan judged before
            # this fact existed (every one of them, on one card) keeps the
            # digest it carries and is not judged again for nothing.
            facts["devices"] = self.devices
        for lib, info in sorted(manifest.items()):
            if info["source_build"]:
                facts[lib] = bool(self.python_for(lib))
                continue
            try:
                facts[lib] = _pins(self.requirements(lib), self.root)
            except BoxProfileError as exc:
                facts[lib] = f"unreadable: {exc}"
        return hashlib.sha1(json.dumps(facts, sort_keys=True).encode()
                            ).hexdigest()[:12]

    def envs(self) -> Dict[str, "Env"]:
        """`library -> Env` as DECLARED, for `providers.candidates`.

        Every pip library has an env; a source build only where `python_for`
        finds one. Each env carries every module its library's kernels gate
        on, at the version its requirements pin (`requirements`: the env
        file or the box's override, and the constraints) or "unknown" where
        nothing pins it, which `Provider._version_ok` allows as it allows an
        unknown installed version. The card is the SKU's: its
        `compute_capability` is what `min_sm` is judged on, and an unknown
        SKU judges nothing. Assumes an env imports every module its kernels
        gate on; where that is wrong the run's own gate declines the cell."""
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name

        from backend.inference.gpu_specs import get_gpu

        from .env import Env
        from .providers import dependency_manifest, get_library

        spec = get_gpu(self.gpu) if self.gpu else None
        sm = (f"sm_{spec.compute_capability.replace('.', '')}"
              if spec and spec.compute_capability else None)
        out: Dict[str, Env] = {}
        for lib, info in dependency_manifest().items():
            if info["source_build"] and not self.python_for(lib):
                continue
            pins = ({} if info["source_build"] else
                    _pins(self.requirements(lib), self.root))
            library = get_library(lib)

            def pinned(import_name: str) -> str:
                line = library.spec_for(import_name) or import_name
                return pins.get(canonicalize_name(Requirement(line).name),
                                "unknown")
            out[lib] = Env(
                torch_version=pins.get("torch"), cuda_available=True,
                device_name=None, gpu_sku=self.gpu, peak_bw_gbps=None, sm=sm,
                libs={name: pinned(name) for name in info["import_names"]},
                library=lib, devices=self.devices)
        return out

    @property
    def devices(self) -> int:
        """The cards a run started from here can drive together: the ones it
        will be shown (`CUDA_VISIBLE_DEVICES`, when set), never more than the
        box has (`BoxProfile.devices`). The same answer the run's own process
        gives (`Env.devices`, its visible device count), so a plan and the run
        judge a collective cell on one fact: a single-card queue's plan puts
        a pair's cells in its gaps, with the reason, instead of queueing what
        its run would decline."""
        have = self.profile.devices if self.profile is not None else 1
        shown = self.environ.get("CUDA_VISIBLE_DEVICES")
        if shown is None:
            return have
        # CUDA reads the list up to its first invalid entry (`-1` hides
        # every card).
        n = 0
        for entry in shown.split(","):
            entry = entry.strip()
            if not entry or entry.startswith("-"):
                break
            n += 1
        return min(have, n)


def require_work_dir(profile: Optional[BoxProfile],
                     environ: Mapping[str, str] = os.environ) -> Path:
    """The machine's scratch disk, or a refusal that names the fix.

    The queue keeps its jobs, its results and its log there, and every one
    of those is per-machine. A command that needs one and has no profile
    is being run somewhere nobody declared, which is the state the shell
    scripts were in.

    The profile states the SHAPE of the path and the environment fills in
    where this machine mounts it (`${KERNEL_BENCH_WORK_DIR}`), so one
    profile serves every machine with the same hardware and wheel stack.
    An unexpanded variable is a refusal rather than a directory literally
    called `${KERNEL_BENCH_WORK_DIR}`, which is what `expandvars` leaves
    behind and what a queue would then happily create.
    """
    if profile is None:
        raise BoxProfileError(
            "no box profile: pass --box, set $KERNEL_BENCH_BOX, or put "
            "kernel_bench/envs/boxes/<name>.toml in place")
    if not profile.work_dir:
        raise BoxProfileError(
            f"box {profile.name} ({profile.path}) declares no `work_dir`; "
            f"the queue keeps jobs/ and results/ there")
    path = expand_box_path(profile.work_dir, environ)
    if "$" in path:
        raise BoxProfileError(
            f"box {profile.name} ({profile.path}) states work_dir="
            f"{profile.work_dir!r}, and this machine's environment does not "
            f"say where that is; export KERNEL_BENCH_WORK_DIR=<the node's "
            f"local disk> before running the queue")
    return Path(path)


def expand_box_path(value: str, environ: Mapping[str, str] = os.environ) -> str:
    """A profile path with `${VAR}` filled in from `environ` and `~` from the
    user — the one place that expansion happens, so `work_dir` and the
    `[providers]` interpreters cannot drift apart on how they read a path."""
    return os.path.expanduser(string.Template(value).safe_substitute(environ))


def prefix_command(command: Sequence[str],
                   profile: Optional[BoxProfile]) -> list[str]:
    prefix = list(profile.command_prefix) if profile else []
    return prefix + list(command)
