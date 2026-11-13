"""Rewrite a SOL pool's traces from the one-aggregate import-time format
to the per-binding one (`backend.code_trace.import_state`).

A trace recorded before the aggregate was removed names one
``<import-time>`` item over everything the trees' import touched, and the
skeleton items of non-tree modules whole. Both moved on any addition — a
def, an import alias, a method — anywhere they cover, so a deploy that
added code took the whole warm corpus with it. This tool rewrites each
record's trace, against the tree the pool was recorded on, into what the
provider records today:

  * the items it already names, kept as they are;
  * each legacy ``<module>`` skeleton item replaced by that module's
    top-level binding items (additions among them are new items);
  * the binding items and import-executed defs of the traced trees'
    static import closure of the modules the trace names;
  * the lock file.

Records and shards are copied byte for byte; only each bundle's
``traces.json`` and the ``traces`` map of its ``index.json`` change. Run
this on a copy, point a serving tree at it, and count what validates
before replacing anything."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.code_trace.import_state import ImportGraph
from backend.code_trace.provider import TRACED_EXCLUDE, TRACED_TREES
from backend.code_trace.source import FileIdentity, digest

IMPORT_ITEM = "<import-time>"
LOCK_ITEM = "<uv.lock>"


def trace_id(items: list) -> str:
    return digest("\n".join(f"{item}\t{held}" for item, held in items))


def excluded_rels() -> frozenset:
    """TRACED_EXCLUDE's module names as the graph names their files."""
    prefix = TRACED_TREES[0].split(".")[0] + "."
    return frozenset(name[len(prefix):].replace(".", "/") + ".py"
                     for name in TRACED_EXCLUDE)


def migrate(pool: Path, backend: Path, import_time: Path) -> int:
    body = json.loads(import_time.read_text(encoding="utf-8"))
    executed = {item: held for item, held in body}
    identities: dict = {}

    def loader(path: str) -> FileIdentity:
        rel = os.path.relpath(path, backend).replace(os.sep, "/")
        identity = identities.get(rel)
        if identity is None:
            identity = identities[rel] = FileIdentity(
                rel, Path(path).read_text(encoding="utf-8"))
        return identity

    graph = ImportGraph(str(backend), TRACED_TREES, loader, excluded_rels())
    lock = (backend.parent / "uv.lock")
    rewritten = 0
    for manifest_path in sorted(pool.glob("*/*/manifest.json")):
        bundle = manifest_path.parent
        index = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
        bodies = json.loads((bundle / "traces.json").read_text(encoding="utf-8"))
        records = index.get("traces", {})
        new_bodies: dict = {}
        mapping: dict = {}
        for record_id, old_id in records.items():
            new_id = mapping.get(old_id)
            if new_id is None:
                items = [(item, held) for item, held in bodies[old_id]
                         if item != IMPORT_ITEM]
                out: list = []
                for item, held in items:
                    if item.endswith("::<module>"):
                        # A non-tree module's skeleton (a tree module's
                        # was never kept): its bindings replace it.
                        rel = item[: -len("::<module>")]
                        identity = (graph.identity(rel)
                                    if rel in graph.rels else None)
                        if identity is None and (backend / rel).exists():
                            identity = loader(str(backend / rel))
                        if identity is not None:
                            out.extend(identity.bindings().items())
                            continue
                    out.append((item, held))
                seeds = {item.split("::", 1)[0] for item, _h in out
                         if "::" in item}
                state = graph.state_items(dict(out), str(lock))
                out.extend(state.items())
                for rel in graph.closure(seeds):
                    for item, held in executed.items():
                        # Defs the import ran, not its module skeletons:
                        # the bindings above replace those.
                        if (item.startswith(rel + "::")
                                and not item.endswith("::<module>")):
                            out.append((item, held))
                merged = sorted(set(out))
                new_id = trace_id(merged)
                mapping[old_id] = new_id
                new_bodies[new_id] = [[item, held] for item, held in merged]
            records[record_id] = new_id
        staging = Path(tempfile.mkdtemp(
            prefix=f".{bundle.name}.", dir=bundle.parent))
        try:
            for path in bundle.iterdir():
                if path.name in ("traces.json", "index.json"):
                    continue
                shutil.copy2(path, staging / path.name)
            _write(staging / "traces.json", new_bodies)
            _write(staging / "index.json", index)
            retired = bundle.with_name(bundle.name + f".{os.getpid()}.old")
            bundle.rename(retired)
            staging.rename(bundle)
            shutil.rmtree(retired, ignore_errors=True)
            rewritten += 1
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return rewritten


def _write(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True,
                        help="the backend/ tree the pool was recorded on")
    parser.add_argument("--import-time", type=Path, required=True,
                        help="the old import-time trace body (the trace "
                             "store's <id>.json for the aggregate item)")
    arguments = parser.parse_args()
    print(f"migrated {migrate(arguments.pool.resolve(),
                             arguments.backend.resolve(),
                             arguments.import_time.resolve())} bundles",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
