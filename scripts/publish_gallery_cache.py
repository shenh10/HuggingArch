"""Publish deterministic Gallery response entries as an indexed artifact.

Measured views are deliberately excluded: they belong to a Kernel Bench
snapshot and are folded live. Model facts and frontier entries are the
expensive deterministic projections a new machine can safely reuse
(`backend.gallery_cache_artifact`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.gallery_cache_artifact import GALLERY_CACHE

build = GALLERY_CACHE.build
publish = GALLERY_CACHE.publish


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(publish(arguments.cache_dir.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
