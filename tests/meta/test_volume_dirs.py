"""Every directory a start-time pull writes into lives on the persistent
volume in production.

The set is derived, not listed: the pulls are the `python -m backend.<m>
pull` lines of `scripts/entrypoint.sh`, each module declares the
directories it writes (`VOLUME_DIRS`, by environment variable), production's
value of each is fly.toml's `[env]` (which overrides the image) or else the
Dockerfile's runtime stage, and the volume is fly.toml's `[[mounts]]`
destination. A pull into a directory off the volume repeats every start —
and a cache whose disk tier is off the volume starts cold every deploy
(the Scaling panel's 2026-11 regression: `HA_SWEEP_CACHE_DIR` under
`/app/var`)."""
from __future__ import annotations

import importlib
from pathlib import Path, PurePosixPath
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def _pulled_modules() -> list[str]:
    text = (ROOT / "scripts" / "entrypoint.sh").read_text(encoding="utf-8")
    return re.findall(r"^python -m (backend\.\w+) pull\b", text, re.M)


def _runtime_env() -> dict[str, str]:
    """The Dockerfile's runtime-stage `ENV` assignments."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    runtime = re.split(r"^FROM \S+ AS runtime\s*$", text, flags=re.M)[1]
    return dict(re.findall(r"^\s*(?:ENV\s+)?([A-Z_][A-Z0-9_]*)=(\S+?)\s*\\?$",
                           runtime, re.M))


def test_every_pulled_directory_is_on_the_volume():
    fly = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    mounts = [PurePosixPath(m["destination"]) for m in fly["mounts"]]
    env = {**_runtime_env(), **fly["env"]}
    modules = _pulled_modules()
    assert "backend.sweep_cache_artifact" in modules, modules
    checked = set()
    for name in modules:
        for variable in importlib.import_module(name).VOLUME_DIRS:
            assert variable in env, (
                f"{name} writes ${variable}, which production does not set: "
                "it would resolve under the (unmounted) home directory")
            path = PurePosixPath(env[variable])
            assert any(path.is_relative_to(m) for m in mounts), (
                f"{name} writes {variable}={path}, off the volume "
                f"{[str(m) for m in mounts]}: it is lost on every restart")
            checked.add(variable)
    assert checked
