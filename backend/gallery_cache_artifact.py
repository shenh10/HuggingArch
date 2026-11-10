"""The Gallery response cache as a release artifact.

The canonical SOL pool and this cache are different layers: the pool owns
immutable laws, while these are pre-materialized response entries. They are
deterministic for one engine/spec snapshot, and can be expensive to draw, so
a release may publish a partial cache (for example completed Gallery GPUs)
and let runtime warm extend it on the persistent volume.

Shareable: model facts (the directory's own tier) and frontier entries
(``frontier/``, gzip-compressed: megabytes of JSON each, a sixth of that on
disk). Measured views (``frontier-views/``) belong to one
Kernel Bench snapshot and are folded live: they never ship, and a newly
installed artifact clears the machine's (they were drawn off the entries it
replaces). The mechanism is `backend.release_artifacts.JsonCacheArtifact`.

    python -m backend.gallery_cache_artifact pull
    python -m backend.gallery_cache_artifact publish --cache-dir DIR
"""
from __future__ import annotations

from backend.release_artifacts import CacheTier, JsonCacheArtifact


GALLERY_CACHE = JsonCacheArtifact(
    name="gallery",
    # v2: frontier entries are gzip files (`.json.gz`); a v1 reader would
    # refuse them, so the two never share an index.
    prefix="gallery-cache-v2",
    dir_env="HA_GALLERY_CACHE_DIR",
    default_dir="gallery",
    pin_env="HA_GALLERY_CACHE_ARTIFACT",
    entries=(CacheTier(), CacheTier("frontier", compress=True)),
    machine_local=(CacheTier("frontier-views"),),
)

pull = GALLERY_CACHE.pull
VOLUME_DIRS = GALLERY_CACHE.volume_dirs


if __name__ == "__main__":
    raise SystemExit(GALLERY_CACHE.main())
