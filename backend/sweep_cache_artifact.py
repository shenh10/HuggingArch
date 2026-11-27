"""The Scaling sweep's answer cache as a release artifact.

A finished sweep answer (`backend.api.sweep_runtime._SWEEP`) is a
deterministic function of its request and the engine, stored with the trace
of the code that computed it, and shareable across machines (it is the entry
the remote tier holds too). A cold one takes minutes; publishing a warm
machine's answers lets a fresh volume serve them at once.

Shareable: the answers (``*.json`` at the top of the directory). The
``steps/`` and ``projection/`` beside them are one machine's local work and
never ship: the former is unfinished sweep progress, the latter the projected
rows Gallery and Scaling recompute from the canonical SOL pool. The mechanism
is `backend.release_artifacts.JsonCacheArtifact`.

    python -m backend.sweep_cache_artifact pull
    python -m backend.sweep_cache_artifact publish --cache-dir DIR
"""
from __future__ import annotations

from backend.measure.cache import env_bytes
from backend.release_artifacts import CacheTier, JsonCacheArtifact


def sweep_disk_bytes() -> int:
    """The one byte budget of the sweep cache directory."""
    return env_bytes("HA_SWEEP_CACHE_BYTES", 512 << 20)


SWEEP_CACHE = JsonCacheArtifact(
    name="sweep",
    prefix="sweep-cache-v1",
    dir_env="HA_SWEEP_CACHE_DIR",
    default_dir="sweep",
    pin_env="HA_SWEEP_CACHE_ARTIFACT",
    entries=(CacheTier(),),
    # The shared projected rows behind Gallery and Scaling answers. They are
    # a machine-local acceleration of the same canonical laws, not a release
    # artifact: their keys already carry the record and substitution axes, and
    # publishing them would duplicate the pool's derivable content.
machine_local=(CacheTier("projection", compress=True),),
)

pull = SWEEP_CACHE.pull
VOLUME_DIRS = SWEEP_CACHE.volume_dirs


if __name__ == "__main__":
    raise SystemExit(SWEEP_CACHE.main())
