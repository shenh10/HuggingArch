"""Every directory a start-time pull writes into lives on the persistent
volume in production.

The set is derived, not listed: the pulls are the `python -m backend.<m>
pull` lines of `scripts/entrypoint.sh`, each module declares the
directories it writes (`VOLUME_DIRS`, by environment variable), production's
value of each is the host's environment (fly.toml's `[env]`, or the Mac's
docker-compose.yml `environment:`; either overrides the image) or else the
Dockerfile's runtime stage, and the volume is the host's mount (fly.toml's
`[[mounts]]`, the compose service's named volume). Every host is checked. A pull into a directory off the volume repeats every start —
and a cache whose disk tier is off the volume starts cold every deploy
(the Scaling panel's 2026-11 regression: `HA_SWEEP_CACHE_DIR` under
`/app/var`)."""
from __future__ import annotations

import importlib
from pathlib import Path, PurePosixPath
import re
import tomllib

import pytest
import yaml

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


def _fly() -> tuple[list[PurePosixPath], dict[str, str]]:
    fly = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    return [PurePosixPath(m["destination"]) for m in fly["mounts"]], fly["env"]


def _mac() -> tuple[list[PurePosixPath], dict[str, str]]:
    """The compose service's named volumes (not bind mounts or tmpfs) and
    its `environment:`."""
    compose = yaml.safe_load(
        (ROOT / "deploy" / "mac" / "docker-compose.yml").read_text(encoding="utf-8"))
    app = compose["services"]["app"]
    named = set(compose.get("volumes") or {})
    mounts = [PurePosixPath(v.split(":")[1]) for v in app["volumes"]
              if v.split(":")[0] in named]
    return mounts, {k: str(v) for k, v in app["environment"].items()}


@pytest.mark.parametrize("host", [_fly, _mac], ids=["fly", "mac"])
def test_every_pulled_directory_is_on_the_volume(host):
    mounts, host_env = host()
    assert mounts, "the host mounts no persistent volume"
    env = {**_runtime_env(), **host_env}
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
