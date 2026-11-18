"""Run parallel-sweep cells in this (fresh) process, each as one recorded
computation under the process's provider (`HA_CODE_TRACE`), and print one
JSON line: per cell, its result and its trace.

    HA_CODE_TRACE=functions python -m tests.backend.code_trace.sweep_child \\
        Qwen/Qwen3-30B-A3B dp_ep,tp_ep 0,3

The plan is built before the first recording, as a server builds a request's
before its cached computation: the memos it fills are hits inside the cells.
A cell written `10!` drops the walk registry first (`WALK_CELLS`), so the cell
walks again while every other memo — ours and SymPy's — is hot.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main(model: str, modes: str, cells: str,
         gpus: str = "H100-SXM,B200") -> None:
    from backend.code_trace.provider import provider, recording
    from backend.inference import parallel_sweep as ps

    config = json.loads((ROOT / "backend" / "arch" / "snapshots"
                         / (model.replace("/", "__") + ".json")).read_text())["config"]
    env = dict(config=config, gpu_sku="B300", gpu_memory_fraction=0.9,
               num_gpus=8, avg_output_tokens=256, parallel=None,
               selection_mode="theoretical_only", model_id=model)
    plan = ps.ParallelSweepPlan.of(modes=modes.split(","), contexts=[1024],
                                   gpu_skus=gpus.split(","), base_env=env)
    p = provider()
    t0 = time.perf_counter()
    if p.traced:
        p.seal()
    seal_seconds = time.perf_counter() - t0
    from backend.inference.walk_registry import WALK_CELLS

    out = []
    for token in cells.split(","):
        if token.endswith("!"):
            WALK_CELLS.clear()
        cell = plan.cells[int(token.rstrip("!"))]
        t0 = time.perf_counter()
        with recording() as rec:
            result = ps.run_layout_cell(cell)
        out.append({"cell": f"{'/'.join(cell.modes)}@{cell.x}",
                    "seconds": time.perf_counter() - t0,
                    "result": json.loads(json.dumps(result, sort_keys=True,
                                                    default=str)),
                    "trace": rec["trace"].to_json()})
    print(json.dumps({"provider": type(p).__name__, "cells": out,
                      "seal_seconds": seal_seconds,
                      "global_items": len(p.global_trace.items)
                      if getattr(p, "global_trace", None) else None}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
