"""Rewrite a SOL pool's traces from the one-aggregate import-time format
to the per-binding one (`backend.code_trace.source.FileIdentity`).

A trace recorded before the aggregate was removed names one
``<import-time>`` item over everything the trees' import touched, and the
skeleton items of non-tree modules whole. Both moved on any addition —
def, import alias, method — anywhere they cover, so a deploy that added
code took the whole warm corpus with it. This tool rewrites each record's
trace, against the tree the pool was recorded on, into what that provider
records today:

  * the items it already names, kept as they are;
  * each legacy ``<module>`` skeleton item replaced by that module's
    top-level binding items (additions among them are new items);
  * the binding items of the traced trees' static import closure of the
    modules the trace names, the state a registry read reaches;
  * the defs of those modules the import itself executed (named in the
    old import-time trace body, ``--import-time``), whose bodies the one
    aggregate covered;
  * the lock file.

Records and shards are copied byte for byte; only each bundle's
``traces.json`` and the ``traces`` map of its ``index.json`` change. Run
this on a copy, point a serving tree at it, and count what validates
before replacing anything."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.code_trace.source import FileIdentity, digest

IMPORT_ITEM = "<import-time>"
LOCK_ITEM = "<uv.lock>"
TREES = ("backend.arch", "backend.inference", "backend.measure", "backend.hf")


def trace_id(items: list) -> str:
    return digest("\n".join(f"{item}\t{held}" for item, held in items))


class Tree:
    """One backend tree's items and import graph, as traces name them."""

    def __init__(self, backend: pathlib.Path, trees: tuple) -> None:
        self.root = backend
        # Module names in imports are spelled from the package ("backend.x"),
        # whatever the tree's directory happens to be called.
        self.prefix = trees[0].split(".")[0] if trees else "backend"
        rels: set = set()
        for tree in trees:
            parts = tree.split(".")
            if len(parts) < 2 or parts[0] != self.prefix:
                continue
            base = pathlib.Path(*parts[1:])
            directory = backend / base
            if directory.is_dir():
                for path in directory.rglob("*.py"):
                    rels.add(path.relative_to(backend).as_posix())
        self.rels = frozenset(rels)
        self._identities: dict = {}
        self._name_to_rel = {}
        for rel in rels:
            module = self.prefix + "." + rel[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            self._name_to_rel[module] = rel

    def identity(self, rel: str):
        identity = self._identities.get(rel)
        if identity is None and (self.root / rel).exists():
            identity = self._identities[rel] = FileIdentity(
                rel, (self.root / rel).read_text(encoding="utf-8"))
            self._identities[rel] = identity
        return identity

    def bindings(self, rel: str) -> dict:
        identity = self.identity(rel)
        return {item: held for item, held in identity.items().items()
                if "<binding>:" in item} if identity else {}

    def edges(self) -> dict:
        out: dict = {}
        for rel in self.rels:
            identity = self.identity(rel)
            if identity is None:
                continue
            here = self.prefix + "." + rel[:-3].replace("/", ".")
            here = here.removesuffix(".__init__")
            for level, module, aliases in identity._imports:
                base = module or ""
                if level:
                    package = here
                    for _ in range(level):
                        package = package.rsplit(".", 1)[0]
                    base = (package + ("." + module if module else "")).strip(".")
                targets = []
                if base in self._name_to_rel:
                    targets.append(self._name_to_rel[base])
                for alias in aliases:
                    name = f"{base}.{alias}" if base else alias
                    for candidate in (name, name.rsplit(".", 1)[0]):
                        if candidate in self._name_to_rel:
                            targets.append(self._name_to_rel[candidate])
                out.setdefault(rel, set()).update(
                    target for target in targets if target != rel)
        return out

    def closure(self, seeds: set, edges: dict) -> set:
        seen, stack = set(seeds), list(seeds)
        while stack:
            for target in edges.get(stack.pop(), ()):
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return seen

    def lock(self) -> str:
        path = self.root.parent / "uv.lock"
        return digest(path.read_text(encoding="utf-8")) if path.exists() else ""


def migrate(pool: pathlib.Path, tree: Tree, import_time: pathlib.Path) -> int:
    if import_time is not None:
        body = json.loads(import_time.read_text(encoding="utf-8"))
        executed = {item: held for item, held in body}
    else:
        executed = {}
    edges = tree.edges()
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
                        rel = item[: -len("::<module>")]
                        bindings = tree.bindings(rel)
                        if bindings:
                            out.extend(bindings.items())
                            continue
                    out.append((item, held))
                seeds = {item.split("::", 1)[0] for item, _h in out
                         if "::" in item} & set(tree.rels)
                for rel in tree.closure(seeds, edges):
                    out.extend(tree.bindings(rel).items())
                    for item, held in executed.items():
                        # Defs the import ran, not its module skeletons:
                        # the bindings above replace those.
                        if (item.startswith(rel + "::")
                                and not item.endswith("::<module>")):
                            out.append((item, held))
                lock = tree.lock()
                if lock:
                    out.append((LOCK_ITEM, lock))
                merged = sorted(set(out))
                new_id = trace_id(merged)
                mapping[old_id] = new_id
                new_bodies[new_id] = [[item, held] for item, held in merged]
            records[record_id] = new_id
        staging = pathlib.Path(tempfile.mkdtemp(
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


def _write(path: pathlib.Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=pathlib.Path, required=True)
    parser.add_argument("--backend", type=pathlib.Path, required=True,
                        help="the backend/ tree the pool was recorded on")
    parser.add_argument("--import-time", type=pathlib.Path, required=True,
                        help="the old import-time trace body (the trace "
                             "store's <id>.json for the aggregate item)")
    arguments = parser.parse_args()
    tree = Tree(arguments.backend.resolve(), TREES)
    print(f"migrated {migrate(arguments.pool.resolve(), tree,
                             arguments.import_time)} bundles", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
