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
#: The factored trace bodies a published archive ships
#: (`scripts.publish_sol_pool`): one shared table of every distinct
#: ``(item, hash)`` pair, the shared binding-item sets whole traces have in
#: common as blocks of table rows, and each bundle's traces as their blocks
#: plus their own rows. Expanded on install — the pool's own on-disk format
#: is unchanged, and an archive with no table unpacks exactly as before.
TRACE_TABLE = "traces-table.json"
TRACE_BLOCKS = "trace-blocks.json"
TRACE_REFS = "traces.refs.json"
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


def _expand_traces(unpacked: Path) -> None:
    """Rebuild every bundle's ``traces.json`` from the shared table.

    A record is served only while its trace validates against the running
    code, so the bodies cannot be left out of the install; factored, the
    whole pool ships each item once instead of every record's full
    inventory. A malformed table, block or row number fails the install —
    present-but-invalid is startup-fatal by design.
    """
    table_file = unpacked / TRACE_TABLE
    if not table_file.exists():
        return
    try:
        table = json.loads(table_file.read_text(encoding="utf-8"))
        entries = table["entries"]
        if (table.get("schema") != 1 or not isinstance(entries, list)
                or not all(
                    isinstance(entry, list) and len(entry) == 2
                    and all(isinstance(side, str) for side in entry)
                    for entry in entries)):
            raise ValueError("trace table has an unknown shape")
        blocks = {}
        blocks_file = unpacked / TRACE_BLOCKS
        if blocks_file.exists():
            stored = json.loads(blocks_file.read_text(encoding="utf-8"))
            if stored.get("schema") != 1 or not isinstance(
                    stored.get("blocks"), dict):
                raise ValueError("trace blocks have an unknown shape")
            blocks = stored["blocks"]
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as error:
        raise ValueError(f"SOL pool archive has a bad trace factoring") from error

    def rows_of(trace: str, rows) -> list:
        if not isinstance(rows, list):
            raise ValueError(f"trace {trace!r} has no row list")
        out = []
        for row in rows:
            if not isinstance(row, int) or not 0 <= row < len(entries):
                raise ValueError(f"trace {trace!r} names row {row!r}")
            out.append(list(entries[row]))
        return out

    for refs_file in sorted(unpacked.glob(f"*/*/{TRACE_REFS}")):
        try:
            refs = json.loads(refs_file.read_text(encoding="utf-8"))
            if not isinstance(refs, dict):
                raise ValueError("trace refs are not a mapping")
            bodies = {}
            for trace_id, spec in refs.items():
                if not isinstance(spec, dict):
                    raise ValueError(f"trace {trace_id!r} has no factoring")
                body = []
                for block in spec.get("blocks", []):
                    if block not in blocks:
                        raise ValueError(f"trace {trace_id!r} names no block")
                    body.extend(rows_of(trace_id, blocks[block]))
                body.extend(rows_of(trace_id, spec.get("rows", [])))
                # The pool writes a trace's items sorted; so does the
                # expansion, whatever order the factoring listed them in.
                bodies[trace_id] = sorted(map(tuple, body))
        except (OSError, ValueError, TypeError,
                json.JSONDecodeError) as error:
            raise ValueError(
                f"SOL pool archive has a bad {refs_file.name} "
                f"for {refs_file.parent.name}") from error
        traces = refs_file.with_name("traces.json")
        traces.write_text(
            json.dumps(bodies, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n", encoding="utf-8")
        refs_file.unlink()
    table_file.unlink()
    blocks_file.unlink(missing_ok=True)


def _unpack(archive: Path, root: Path, artifact: str, digest: str) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        unpacked = staging / "sol"
        _expand_traces(unpacked)
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
