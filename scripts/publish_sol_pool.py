"""Publish the complete canonical SOL pool through its engine-version index
(`backend.sol_pool_artifact`; transport: `backend.release_artifacts`)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from backend.inference.sol_data import walkRecordSchema
from backend.release_artifacts import publish as publish_artifact
from backend.release_artifacts import write_archive


#: Trace bodies ship factored three ways. One shared table holds every
#: distinct ``(item, hash)`` pair; the binding items a trace's import
#: closure adds are near-identical across a warm corpus (a handful of
#: distinct sets for thousands of traces), so each such set ships once as
#: a block and a trace's ``traces.refs.json`` names its blocks by id plus
#: its own rows in the table. A full functions-mode pool holds a few
#: thousand distinct pairs but gigabytes of per-record inventories, every
#: one a subset of that one table (`sol_pool_artifact` expands the files
#: back on install; the pool's own format is unchanged).
TRACE_TABLE = "traces-table.json"
TRACE_BLOCKS = "trace-blocks.json"
TRACE_REFS = "traces.refs.json"
_BLOCK_MIN_TRACES = 2


def build(root: Path, destination: Path) -> int:
    manifests = sorted(root.glob("*/*/manifest.json"))
    if not manifests:
        raise ValueError(f"no SOL bundles under {root}")
    # A bundle being written lives beside its key until it is swapped in
    # (`SolPool._swapIn`: `<key>.<pid>.<thread>.new` / `.old`, the writer's
    # lock `<key>.writer`); a key itself has no dot. Only bundle files ship
    # (a pool root holds nothing else but install markers and this table).
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != ".writer"
        and len(path.relative_to(root).parts) == 3
        and "." not in path.relative_to(root).parts[1])
    bodies: dict[Path, dict] = {}
    pairs: set = set()
    for path in files:
        if path.name != "traces.json":
            continue
        bundle_bodies = json.loads(path.read_text(encoding="utf-8"))
        bodies[path] = bundle_bodies
        for body in bundle_bodies.values():
            pairs.update((str(item), str(held)) for item, held in body)
    table = sorted(pairs)
    row_of = {pair: row for row, pair in enumerate(table)}

    # The blocks: each distinct set of binding items whole traces share.
    binding_sets: dict = {}
    for bundle_bodies in bodies.values():
        for body in bundle_bodies.values():
            key = frozenset(pair for pair in map(tuple, body)
                            if "<binding>:" in pair[0])
            if key:
                binding_sets.setdefault(key, 0)
                binding_sets[key] += 1
    shared = sorted((pairs_, count) for pairs_, count in binding_sets.items()
                    if count >= _BLOCK_MIN_TRACES)
    blocks = {hashlib.sha256(
        ("\n".join(f"{item}\t{held}" for item, held in sorted(pairs_))
         ).encode("utf-8")).hexdigest()[:16]: pairs_
        for pairs_, _count in shared}
    # Largest first, so a trace names the fewest blocks that cover it and
    # every block it names is whole inside it (expansion adds the block).
    ordered_blocks = sorted(blocks.items(), key=lambda kv: -len(kv[1]))

    def members() -> Iterable[tuple[str, bytes]]:
        if table:
            yield TRACE_TABLE, (
                json.dumps({"schema": 1, "entries": table},
                           sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False) + "\n").encode("utf-8")
        if blocks:
            yield TRACE_BLOCKS, (
                json.dumps({"schema": 1, "blocks": {
                    block: [row_of[pair] for pair in sorted(
                        pairs_, key=lambda p: row_of[p])]
                    for block, pairs_ in sorted(blocks.items())}},
                    sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False) + "\n").encode("utf-8")
        for path in files:
            bundle_bodies = bodies.get(path)
            if bundle_bodies is None:
                yield path.relative_to(root).as_posix(), path.read_bytes()
                continue
            refs = {}
            for trace_id, body in bundle_bodies.items():
                named = [tuple(pair) for pair in body]
                # Greedy, largest first, each block's pairs consumed once:
                # the blocks a trace names are disjoint in it, so the
                # expansion adds every item once.
                remaining = {pair for pair in named
                             if "<binding>:" in pair[0]}
                trace_blocks = []
                for block, pairs_ in ordered_blocks:
                    if pairs_ <= remaining:
                        trace_blocks.append(block)
                        remaining -= pairs_
                covered = {pair for block in trace_blocks
                           for pair in blocks[block]}
                rows = [row_of[pair] for pair in named if pair not in covered]
                refs[trace_id] = {"blocks": trace_blocks, "rows": rows}
            yield "/".join((*path.relative_to(root).parts[:-1], TRACE_REFS)), (
                json.dumps(refs, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False) + "\n").encode("utf-8")

    write_archive(destination, "sol", members())
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
