"""Publish a warm machine's finished Scaling sweep answers as an indexed
artifact (`backend.sweep_cache_artifact`); the unfinished ``steps/`` stay.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.sweep_cache_artifact import SWEEP_CACHE

build = SWEEP_CACHE.build
publish = SWEEP_CACHE.publish


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(publish(arguments.cache_dir.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
