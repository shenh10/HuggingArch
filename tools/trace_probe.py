"""Stage-1 probe for verifying traces: which repo functions does one sweep
cell actually execute, and what does recording that cost?

Measurement only — nothing in `backend/` reads this. It answers, for a few
layouts of a model's parallel sweep (SOL / theoretical only):

  * overhead: wall time of a cell with vs without `sys.monitoring` PY_START
    recording (callback returns DISABLE; `restart_events()` at each begin);
  * how many distinct repo functions a cell executes, and the trace size;
  * how many distinct traces the layouts of one model produce;
  * whether `frontier.py::frontier_view` is in a cell's trace (it is only
    reached by `ParallelSweepPlan.assemble`);
  * the memo hole: a cell's trace recorded in a process that already ran
    other cells (memo hits skip code) against the same cell's trace recorded
    cold in a fresh process.

A trace item is (path, def-qualname) → hash of the def's AST (docstrings
dropped), one per outermost def (nested defs, lambdas and genexprs belong to
the def around them; module-level lambdas to the module), plus one
`<module>` item per touched module: its AST with every def reduced to its
decorators and name (so adding, removing or rebinding a def changes it).
Functions executed during import are reported separately: they ran before
any trace began, so they are a dependency of every trace.

    .venv/bin/python -m tools.trace_probe --model meta-llama/Llama-3.2-3B \
        --modes tp_tp --cells 0,1,2,3 --gpus H100-SXM,B200
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
BACKEND = str(ROOT / "backend") + os.sep


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class Recorder:
    """PY_START on every code object, first execution only (DISABLE), added
    to every active trace. `begin` re-arms the events process-wide."""

    def __init__(self) -> None:
        self.tool = None
        for tid in (3, 4, 5):
            try:
                sys.monitoring.use_tool_id(tid, "trace_probe")
                self.tool = tid
                break
            except ValueError:
                continue
        assert self.tool is not None
        self.active: List[Set[Any]] = []
        mon = sys.monitoring
        disable = mon.DISABLE

        def on_start(code, _offset):
            if code.co_filename.startswith(BACKEND):
                for t in self.active:
                    t.add(code)
            return disable

        mon.register_callback(self.tool, mon.events.PY_START, on_start)
        mon.set_events(self.tool, mon.events.PY_START)

    def begin(self) -> Set[Any]:
        sys.monitoring.restart_events()
        t: Set[Any] = set()
        self.active.append(t)
        return t

    def end(self, t: Set[Any]) -> Set[Any]:
        self.active.remove(t)
        return t


# ---------------------------------------------------------------------------
# Code object → trace item
# ---------------------------------------------------------------------------


def _strip_doc(body: list) -> list:
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[1:]
    return body


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class _ModuleIndex:
    """One source file: its outermost defs (line spans, qualname, hash) and
    its skeleton hash."""

    def __init__(self, path: str) -> None:
        src = pathlib.Path(path).read_text(encoding="utf-8")
        tree = ast.parse(src)
        self.defs: List[Tuple[int, int, str, str]] = []

        def visit(body: list, prefix: str) -> None:
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                    clone = ast.parse(ast.unparse(node)).body[0]
                    clone.body = _strip_doc(clone.body) or [ast.Pass()]
                    self.defs.append((start, node.end_lineno, prefix + node.name,
                                      _sha(ast.dump(clone))))
                elif isinstance(node, ast.ClassDef):
                    visit(node.body, prefix + node.name + ".")
                elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    for field in ("body", "orelse", "finalbody"):
                        visit(getattr(node, field, []) or [], prefix)
                    for h in getattr(node, "handlers", []) or []:
                        visit(h.body, prefix)

        visit(tree.body, "")

        class _Skel(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                return ast.FunctionDef(
                    name=node.name, args=ast.arguments(
                        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[],
                        defaults=[]),
                    body=[ast.Pass()], decorator_list=node.decorator_list,
                    returns=None, type_params=[])
            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                node.body = _strip_doc(node.body) or [ast.Pass()]
                self.generic_visit(node)
                return node

        tree.body = _strip_doc(tree.body)
        self.skeleton = _sha(ast.dump(_Skel().visit(tree)))
        self.defs.sort()

    def item(self, line: int) -> Optional[Tuple[str, str]]:
        best = None
        for start, end, qual, h in self.defs:
            if start <= line <= end:
                best = (qual, h)  # outermost spans are unique per line
                break
        return best


class Items:
    def __init__(self) -> None:
        self._mods: Dict[str, _ModuleIndex] = {}
        self.bytes_parsed = 0

    def _mod(self, path: str) -> _ModuleIndex:
        m = self._mods.get(path)
        if m is None:
            m = self._mods[path] = _ModuleIndex(path)
        return m

    def of(self, codes: Set[Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for co in codes:
            rel = os.path.relpath(co.co_filename, ROOT)
            mod = self._mod(co.co_filename)
            out[f"{rel}::<module>"] = mod.skeleton
            it = mod.item(co.co_firstlineno)
            if it is not None:
                out[f"{rel}::{it[0]}"] = it[1]
        return out


def trace_id(items: Dict[str, str]) -> str:
    return _sha(json.dumps(sorted(items.items())))


# ---------------------------------------------------------------------------
# Child: one fresh process
# ---------------------------------------------------------------------------


def child(args: argparse.Namespace) -> None:
    rec = Recorder() if args.trace else None
    imp = rec.begin() if rec else None
    t0 = time.perf_counter()
    from backend.arch.loader import get_default_loader  # noqa: F401
    from backend.inference import parallel_sweep as ps
    import_s = time.perf_counter() - t0
    if rec:
        rec.end(imp)

    snap = ROOT / "backend" / "arch" / "snapshots" / (args.model.replace("/", "__") + ".json")
    config = json.loads(snap.read_text())["config"]
    env = dict(config=config, gpu_sku="B300", gpu_memory_fraction=0.9,
               num_gpus=8, avg_output_tokens=256, parallel=None,
               selection_mode="theoretical_only", model_id=args.model)
    plan_t = rec.begin() if rec else None
    t0 = time.perf_counter()
    plan = ps.ParallelSweepPlan.of(modes=args.modes.split(","), contexts=[1024],
                                   gpu_skus=args.gpus.split(","), base_env=env)
    plan_s = time.perf_counter() - t0
    if rec:
        rec.end(plan_t)
    if args.list:
        print(json.dumps([f"{'/'.join(c.modes)}@{c.x}" for c in plan.cells]))
        return
    picks = [int(i) for i in args.cells.split(",")]
    cells_out = []
    results = []
    for i in picks:
        cell = plan.cells[i]
        t = rec.begin() if rec else None
        t0 = time.perf_counter()
        r = ps.run_layout_cell(cell)
        dt = time.perf_counter() - t0
        results.append(r)
        entry = {"cell": f"{'/'.join(cell.modes)}@{cell.x}", "seconds": dt,
                 "ok": "cards" in r}
        if rec:
            entry["codes"] = rec.end(t)
        cells_out.append(entry)
    asm_codes = None
    if args.assemble:
        # assemble over just the picked cells: enough to reach the frontier.
        sub = type(plan)(**{**plan.__dict__, "cells": tuple(plan.cells[i] for i in picks)})
        t = rec.begin() if rec else None
        sub.assemble(results)
        if rec:
            asm_codes = rec.end(t)

    out: Dict[str, Any] = {"import_s": import_s, "plan_s": plan_s, "cells": []}
    if rec:
        t0 = time.perf_counter()
        items = Items()
        out["import"] = items.of(imp)
        out["plan"] = items.of(plan_t)
        for e in cells_out:
            codes = e.pop("codes")
            e["n_code_objects"] = len(codes)
            e["items"] = items.of(codes)
        if asm_codes is not None:
            out["assemble"] = items.of(asm_codes)
        out["itemize_s"] = time.perf_counter() - t0
    out["cells"] = cells_out
    pathlib.Path(args.out).write_text(json.dumps(out))


# ---------------------------------------------------------------------------
# Parent: orchestrate fresh processes, report
# ---------------------------------------------------------------------------


def _run(args: argparse.Namespace, *, cells: str, trace: bool, assemble: bool = False,
         listing: bool = False) -> Any:
    out = f"/tmp/trace_probe_{os.getpid()}.json"
    cmd = [sys.executable, "-m", "tools.trace_probe", "--child", "--model", args.model,
           "--modes", args.modes, "--gpus", args.gpus, "--cells", cells, "--out", out]
    if trace:
        cmd.append("--trace")
    if assemble:
        cmd.append("--assemble")
    if listing:
        cmd.append("--list")
        return json.loads(subprocess.check_output(cmd, cwd=ROOT, text=True).strip().splitlines()[-1])
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads(pathlib.Path(out).read_text())


def _fn_count(items: Dict[str, str]) -> int:
    return sum(1 for k in items if not k.endswith("::<module>"))


def _mods(items: Dict[str, str]) -> int:
    return sum(1 for k in items if k.endswith("::<module>"))


def parent(args: argparse.Namespace) -> None:
    names = _run(args, cells="0", trace=False, listing=True)
    picks = [int(i) for i in args.cells.split(",")] if args.cells != "all" else list(range(len(names)))
    print(f"{args.model}: {len(names)} cells; probing {[names[i] for i in picks]}")
    report: Dict[str, Any] = {"model": args.model, "cells": {}}

    cold: Dict[int, Dict[str, str]] = {}
    for i in picks:
        plain = [_run(args, cells=str(i), trace=False)["cells"][0]["seconds"]
                 for _ in range(args.repeat)]
        traced_runs = [_run(args, cells=str(i), trace=True) for _ in range(args.repeat)]
        traced = [r["cells"][0]["seconds"] for r in traced_runs]
        e = traced_runs[0]["cells"][0]
        cold[i] = e["items"]
        size = len(json.dumps(sorted(e["items"].items())))
        row = {"plain_s": min(plain), "traced_s": min(traced),
               "code_objects": e["n_code_objects"], "functions": _fn_count(e["items"]),
               "modules": _mods(e["items"]), "trace_json_bytes": size,
               "trace_id": trace_id(e["items"]), "ok": e["ok"],
               "itemize_s": traced_runs[0]["itemize_s"],
               "import_fns": _fn_count(traced_runs[0]["import"]),
               "import_s_traced": traced_runs[0]["import_s"]}
        report["cells"][names[i]] = row
        print(f"  cold {names[i]:>10}: plain {row['plain_s']:.2f}s traced {row['traced_s']:.2f}s"
              f"  fns {row['functions']} mods {row['modules']} codes {row['code_objects']}"
              f"  {size/1024:.0f} KiB id {row['trace_id']} ok={e['ok']}")

    # Same cells in one process, in order: memo hits from earlier cells.
    seq_plain = _run(args, cells=",".join(map(str, picks)), trace=False)
    seq = _run(args, cells=",".join(map(str, picks)), trace=True, assemble=True)
    print(f"  import: {_fn_count(seq['import'])} repo fns executed during import "
          f"({_mods(seq['import'])} modules); import {seq_plain['import_s']:.2f}s plain, "
          f"{seq['import_s']:.2f}s traced")
    print(f"  plan build: {_fn_count(seq['plan'])} fns")
    report["import_fns"] = _fn_count(seq["import"])
    report["warm"] = {}
    for i, e, p in zip(picks, seq["cells"], seq_plain["cells"], strict=True):
        missing = sorted(set(cold[i]) - set(e["items"]))
        report["warm"][names[i]] = {"plain_s": p["seconds"], "traced_s": e["seconds"],
                                    "functions": _fn_count(e["items"]),
                                    "missing_vs_cold": len(missing),
                                    "missing": missing}
        print(f"  warm {names[i]:>10}: plain {p['seconds']:.2f}s traced {e['seconds']:.2f}s"
              f"  fns {_fn_count(e['items'])}  missing-vs-cold {len(missing)}")
    ids = {trace_id(v) for v in cold.values()}
    union = set().union(*cold.values())
    inter = set(union)
    for v in cold.values():
        inter &= set(v)
    print(f"  distinct cold traces: {len(ids)} / {len(cold)}; union {_fn_count(dict.fromkeys(union))}"
          f" fns, intersection {_fn_count(dict.fromkeys(inter))} fns")
    fv = "backend/inference/frontier.py::frontier_view"
    print(f"  frontier_view in any cell trace: {any(fv in v for v in cold.values())};"
          f" in assemble trace: {fv in seq.get('assemble', {})}")
    report.update(distinct=len(ids), union=_fn_count(dict.fromkeys(union)),
                  intersection=_fn_count(dict.fromkeys(inter)),
                  frontier_in_cells=any(fv in v for v in cold.values()),
                  frontier_in_assemble=fv in seq.get("assemble", {}),
                  cold_items={names[i]: cold[i] for i in picks},
                  diff_vs_first={names[i]: sorted(set(cold[i]) ^ set(cold[picks[0]]))[:40]
                                 for i in picks[1:]})
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(report, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--modes", default="tp_tp")
    ap.add_argument("--gpus", default="H100-SXM,B200")
    ap.add_argument("--cells", default="0")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    (child if args.child else parent)(args)


if __name__ == "__main__":
    main()
