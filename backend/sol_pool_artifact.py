"""Bootstrap the local SOL pool from a versioned storage artifact.

The canonical pool is a set of immutable, content-addressed bundles.  A
machine therefore needs only one download per engine version: fetch the
compressed archive, verify its digest, and unpack it on the persistent volume.
Per-record remote reads belong in ``SolPool`` itself (the next step for this
layer); this module is the smaller startup boundary that lets a new machine
reuse a complete warm corpus without baking it into the image. The transport
(index, download, digest) is `backend.release_artifacts`'.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from backend.release_artifacts import fetch_verified


DIR_ENV = "HA_SOL_POOL_DIR"
PIN_ENV = "HA_SOL_POOL_ARTIFACT"
#: The directories (by environment variable) a pull writes into, which must
#: outlive a restart (`tests/meta/test_volume_dirs.py`).
VOLUME_DIRS = (DIR_ENV,)


def _pool_root() -> Path:
    return Path(os.environ.get(
        DIR_ENV, str(Path.home() / ".cache/huggingarch/sol-pool")))


def _safe_archive(archive: Path) -> None:
    """Refuse an archive that is not exactly the canonical ``sol/`` tree."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name.rstrip("/")
            if name == "sol" or name.startswith("sol/"):
                continue
            raise ValueError(f"SOL pool archive has non-pool member {name!r}")
        if not any(member.name.rstrip("/") == "sol" for member in members):
            raise ValueError("SOL pool archive has no sol/ root")


def _write_bootstrap(root: Path, artifact: str, digest: str,
                     manifest_count: int) -> None:
    marker = root / ".bootstrap.json"
    staging = root / f".bootstrap.json.{os.getpid()}.tmp"
    try:
        with staging.open("w", encoding="utf-8") as out:
            json.dump({
                "artifact": artifact,
                "sha256": digest,
                "manifests": manifest_count,
            }, out, sort_keys=True)
            out.write("\n")
        os.replace(staging, marker)
    finally:
        staging.unlink(missing_ok=True)


def _has_bootstrap(root: Path, artifact: str, digest: str) -> bool:
    """Whether ``root`` is exactly a previously installed artifact.

    Manifest presence alone cannot answer that question. A runtime warm that
    started before an artifact existed leaves valid immutable bundles behind,
    and that partial state must not make later starts skip the complete,
    digest-verified corpus.
    """
    try:
        marker = json.loads(
            (root / ".bootstrap.json").read_text(encoding="utf-8"))
        expected = int(marker["manifests"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    if marker.get("artifact") != artifact or marker.get("sha256") != digest:
        return False
    return expected > 0 and len(list(root.glob("*/*/manifest.json"))) == expected


def _unpack(archive: Path, root: Path, artifact: str, digest: str) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        unpacked = staging / "sol"
        manifests = list(unpacked.glob("*/*/manifest.json"))
        if not manifests:
            raise ValueError("SOL pool archive has no manifests")
        if root.exists() or root.is_symlink():
            if root.is_symlink():
                raise RuntimeError(f"SOL pool path {root} is a symlink")
            backup = Path(tempfile.mkdtemp(
                prefix=f".{root.name}.previous.", dir=root.parent))
            backup.rmdir()
            root.rename(backup)
        else:
            backup = None
        try:
            unpacked.rename(root)
        except BaseException:
            if backup is not None and not root.exists():
                backup.rename(root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        _write_bootstrap(root, artifact, digest, len(manifests))
        print(f"SOL pool artifact: unpacked {len(manifests)} bundles to {root}",
              flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def pull(timeout: float = 900.0) -> int:
    """Fetch, verify and unpack the configured artifact; 0 on success/no-op."""
    from backend.inference.sol_data import walkRecordSchema
    root = _pool_root()

    def install(archive: Path, artifact: str, digest: str) -> None:
        _safe_archive(archive)
        _unpack(archive, root, artifact, digest)

    return fetch_verified(
        "SOL pool", walkRecordSchema, PIN_ENV, root,
        lambda artifact, digest: _has_bootstrap(root, artifact, digest),
        install, timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pull"])
    parser.add_argument("--timeout", type=float, default=900.0)
    arguments = parser.parse_args(argv)
    return pull(arguments.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
