"""The self-hosted production host's environment, as its container gets it
(deploy/self-hosted/): `runtime.env` under the compose service's literal
`environment:`. Values compose interpolates from the host's secrets file
(`${...}`) are the machine's, not declared here, and are left out.

Read by scripts/release_warm.py (it warms and publishes the release artifacts
under this environment) and tests/meta/test_volume_dirs.py (the volume gate);
`root` is the checkout whose files are read, so a release reads its own ref."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import yaml

HOST = Path("deploy") / "self-hosted"


def runtime_env(root: Path) -> dict[str, str]:
    """`runtime.env`'s KEY=value lines."""
    env = {}
    for line in (root / HOST / "runtime.env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            env[name.strip()] = value.strip()
    return env


def _app(root: Path) -> dict:
    compose = yaml.safe_load((root / HOST / "docker-compose.yml").read_text(encoding="utf-8"))
    return compose["services"]["app"], set(compose.get("volumes") or {})


def host_env(root: Path) -> dict[str, str]:
    """What the host sets over the image: runtime.env, then the service's
    literal `environment:` (which compose applies over `env_file`)."""
    app, _ = _app(root)
    literal = {k: str(v) for k, v in app["environment"].items() if "${" not in str(v)}
    return {**runtime_env(root), **literal}


def volume_mounts(root: Path) -> list[PurePosixPath]:
    """Container paths of the service's named volumes (not bind mounts)."""
    app, named = _app(root)
    return [PurePosixPath(v.split(":")[1]) for v in app["volumes"] if v.split(":")[0] in named]
