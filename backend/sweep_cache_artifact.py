"""The Scaling sweep's answer cache as a release artifact.

A finished sweep answer (`backend.api.sweep_runtime._SWEEP`) is a
deterministic function of its request and the engine, stored with the trace
of the code that computed it, and shareable across machines (it is the entry
the remote tier holds too). A cold one takes minutes; publishing a warm
machine's answers lets a fresh volume serve them at once.

Shareable: the answers (``*.json`` at the top of the directory). The
``steps/`` beside them are one machine's unfinished work and never ship.
The mechanism is `backend.release_artifacts.JsonCacheArtifact`.

    python -m backend.sweep_cache_artifact pull
    python -m backend.sweep_cache_artifact publish --cache-dir DIR
"""
from __future__ import annotations

from backend.release_artifacts import CacheTier, JsonCacheArtifact


SWEEP_CACHE = JsonCacheArtifact(
    name="sweep",
    prefix="sweep-cache-v1",
    dir_env="HA_SWEEP_CACHE_DIR",
    default_dir="sweep",
    pin_env="HA_SWEEP_CACHE_ARTIFACT",
    entries=(CacheTier(),),
)

pull = SWEEP_CACHE.pull
VOLUME_DIRS = SWEEP_CACHE.volume_dirs


if __name__ == "__main__":
    raise SystemExit(SWEEP_CACHE.main())
