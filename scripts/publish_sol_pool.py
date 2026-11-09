"""Publish the complete canonical SOL pool through its engine-version index
(`backend.sol_pool_artifact`; transport: `backend.release_artifacts`)."""
from __future__ import annotations

import argparse
from pathlib import Path

from backend.inference.sol_data import walkRecordSchema
from backend.release_artifacts import publish as publish_artifact
from backend.release_artifacts import write_archive


def build(root: Path, destination: Path) -> int:
    manifests = sorted(root.glob("*/*/manifest.json"))
    if not manifests:
        raise ValueError(f"no SOL bundles under {root}")
    # A bundle being written lives beside its key until it is swapped in
    # (`SolPool._swapIn`: `<key>.<pid>.<thread>.new` / `.old`, the writer's
    # lock `<key>.writer`); a key itself has no dot. Only bundles ship.
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {
            ".bootstrap.json", ".writer"}
        and "." not in path.relative_to(root).parts[1]
    )
    write_archive(destination, "sol", (
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in files))
    return len(manifests)


def publish(root: Path) -> str:
    return publish_artifact(walkRecordSchema, build, root, "manifests")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(publish(arguments.pool_dir.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
