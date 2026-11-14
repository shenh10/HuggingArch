"""Cache-write quantisation hints — does the spec store the cache the way
the forward writes it?

A cache slot's stored format has no checkpoint to diff against: the cache is
runtime state, so the safetensors header that anchors every weight byte says
nothing about it, and the only witness is the forward source. A reference
forward often quantises a write in place and then copies it into a buffer
allocated at the model's dtype (`act_quant(kv, 64, ..., True)` followed by
`self.kv_cache[...] = kv`): the buffer's BF16 is a container, the stored
values are FP8. Reading the `register_buffer` and not the write bills the
slot at BF16, and nothing downstream notices (DeepSeek-V4.1-Flash shipped at
3.6x its paper's per-token KV that way).

This scan reads the recorded forward source with `ast`: a WRITE is an
assignment into a subscript of a buffer the source registers
(`register_buffer("<name>", ...)`), and the write is QUANTISED when, earlier
in the same function, the written tensor is the first argument of a call
whose name says it quantises (and not dequantises), or is assigned from
one — directly or through later assignments that read it (DeepSeek-V3.2
quantises its latent, dequantises it back by hand, and writes that). What it asks of the spec, against the formats the as-shipped walk
stores its per-token cached tensors at:

  * a quantising call that names its precision (`fp4_act_quant`) — some
    slot stored at that element width;
  * every buffer filled quantised — as many cached tensors stored quantised
    (scaled, or narrower than the model's dtype); a precision named or not,
    one quantised slot does not answer for two quantised buffers;
  * one that quantises a slice of the row (`act_quant(kv[..., :-rd], ...)`)
    — some slot whose format keeps an unquantised tail.

Writes are matched to the spec by format, not to individual slots: the
source's buffer names say nothing the spec can be keyed by. A requirement
no slot meets is one warning naming the source line. Layer 2: the findings
are hints for the spec author and never fail validation.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

from backend.arch.cache_formats import CacheFormat
from backend.arch.dtypes import DTYPES, canonical, dtype_bytes
from backend.arch.eval import cache as cache_mod
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.hf.forward_source import recorded_forward_files

from ._finding import Finding

_QUANTISES = re.compile(r"quant", re.I)
_DEQUANTISES = re.compile(r"dequant", re.I)
#: A precision written into a function's name: `fp4_act_quant`,
#: `quantize_nvfp4`, `int8_quant`. Read through `dtypes.canonical`, so only
#: a spelling the dtype table knows names a width.
_PRECISION = re.compile(r"(?:nv|mx)?fp\d+|int\d+", re.I)
_SPELLINGS = frozenset(spelling.upper() for row in DTYPES.values()
                       for spelling in row.spellings)


@dataclass(frozen=True)
class QuantisedWrite:
    """One cache write the forward quantises first."""

    path: str
    line: int
    buffer: str
    call: str           # the quantising call, as written
    width: Optional[float]   # element bytes its name states; None if none
    partial: bool       # it quantises a slice of the written tensor

    @property
    def where(self) -> str:
        return f"{self.path.rsplit('/', 1)[-1]}:{self.line}"


def _callee(call: ast.Call) -> Optional[str]:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _quantising(call: ast.Call) -> bool:
    name = _callee(call) or ""
    return bool(_QUANTISES.search(name)) and not _DEQUANTISES.search(name)


def _root(expr: ast.AST) -> Optional[str]:
    """The variable an expression reads its tensor from: `kv` for `kv`,
    `kv[..., :-rd]`, `kv.squeeze(1)`."""
    while True:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Subscript):
            expr = expr.value
        elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            expr = expr.func.value
        else:
            return None


def _width(call: ast.Call) -> Optional[float]:
    """The element bytes a quantising call's name states, when it names a
    precision the dtype table spells."""
    match = _PRECISION.search(_callee(call) or "")
    if match is None or match.group(0).upper() not in _SPELLINGS:
        return None
    return float(dtype_bytes(canonical(match.group(0))))


def _buffers(trees: List[Tuple[str, ast.AST]]) -> Set[str]:
    out: Set[str] = set()
    for _path, tree in trees:
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and _callee(node) == "register_buffer"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                out.add(node.args[0].value)
    return out


def _reach(assigns: List[ast.Assign], start: int, names: Set[str],
           until: int) -> Set[str]:
    """The variables that still hold a quantising call's result at line
    `until`: its own `names`, then every plain-name assignment between
    reading one of them (`kv = (kv_fp8 * kv_scale).to(dtype)`, the
    simulated-precision round trip) — and none that overwrites one from
    something else."""
    held = set(names)
    for node in assigns:
        if not start < node.lineno < until:
            continue
        reads = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
        for target in node.targets:
            if isinstance(target, ast.Name):
                if reads & held:
                    held.add(target.id)
                else:
                    held.discard(target.id)
    return held


def quantised_writes(files: List[Dict[str, str]]) -> List[QuantisedWrite]:
    """Every write into a registered buffer that the forward quantises
    first, one per (write, quantising call).

    A frozen source may also carry CUDA/C++ implementation evidence. Those
    files are not Python AST inputs; retain the Python scan instead of
    aborting the entire check on a native kernel's syntax.
    """
    trees = [(f.get("path") or "", ast.parse(f.get("text") or ""))
             for f in files
             if not (f.get("path") or "").endswith(
                 (".cu", ".cuh", ".cpp", ".cc", ".c", ".h", ".hpp"))]
    buffers = _buffers(trees)
    out: List[QuantisedWrite] = []
    for path, tree in trees:
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assigns = sorted((n for n in ast.walk(fn)
                              if isinstance(n, ast.Assign)),
                             key=lambda n: n.lineno)
            # (line, call, variables it quantises or yields)
            quants: List[Tuple[int, ast.Call, Set[str]]] = []
            for node in ast.walk(fn):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and _quantising(node.value)):
                    # `x_fp8, x_scale = act_quant(x)`: the data is the first
                    # name; the scale beside it is the format's own sidecar.
                    quants.append((node.lineno, node.value, {
                        _root(t.elts[0] if isinstance(t, ast.Tuple) and t.elts
                              else t) for t in node.targets}))
                elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and _quantising(node.value) and node.value.args):
                    quants.append((node.lineno, node.value,
                                   {_root(node.value.args[0])}))
            for node in assigns:
                for target in node.targets:
                    if not (isinstance(target, ast.Subscript)
                            and isinstance(target.value, ast.Attribute)
                            and target.value.attr in buffers):
                        continue
                    written = _root(node.value)
                    for line, call, names in quants:
                        if line >= node.lineno:
                            continue
                        held = _reach(assigns, line, names, node.lineno)
                        if written not in held:
                            continue
                        first = call.args[0] if call.args else None
                        out.append(QuantisedWrite(
                            path=path, line=node.lineno,
                            buffer=target.value.attr,
                            call=f"{ast.unparse(call)} (line {line})",
                            width=_width(call),
                            partial=isinstance(first, ast.Subscript)
                            and _root(first) in held))
    return out


def _stored_formats(resolved) -> Iterator[Tuple[CacheFormat, bool, str]]:
    """`(format, quantised, name)` of every per-token cached tensor as
    shipped; quantised is scaled, or narrower than the model's dtype."""
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=1).analyze()
    model_width = float(dtype_bytes(analysis.resolved_precision.model_dtype))
    for entry in cache_mod.derive(analysis).values.values():
        for tensor in entry.tensors:
            if not tensor.recurrent:
                fmt = tensor.resolved_format
                yield fmt, (fmt.scale is not None
                            or fmt.element_bytes < model_width), tensor.name


def run_cache_write_check(model_id: str, resolved) -> List[Finding]:
    writes = quantised_writes(recorded_forward_files(model_id))
    if not writes:
        return []
    formats = list(_stored_formats(resolved))
    widths = {fmt.element_bytes for fmt, _q, _n in formats}
    tailed = any(fmt.tail is not None for fmt, _q, _n in formats)
    quantised = {name for _f, q, name in formats if q}
    stored = sorted({fmt.label() for fmt, _q, _n in formats})

    findings: List[Finding] = []
    seen: Set[Tuple[str, str]] = set()

    def flag(write: QuantisedWrite, need: str, detail: str) -> None:
        if (write.call, need) in seen:
            return
        seen.add((write.call, need))
        findings.append(Finding(
            kind="cache_write_quantised_unmodelled",
            detail=(f"{write.where}: the forward writes `{write.buffer}` "
                    f"after `{write.call}`, {detail} The spec stores its "
                    f"per-token cache as {stored}. A cache slot is stored "
                    f"at the format its write is quantised to, not the "
                    f"buffer's dtype: declare it under "
                    f"`quant_context.cache_formats`."),
            extra={"buffer": write.buffer, "line": write.line,
                   "requires": need}))

    for write in writes:
        if write.width is not None and write.width not in widths:
            flag(write, f"width:{write.width:g}",
                 f"which stores {write.width:g}-byte elements; no cache "
                 f"slot is stored at that width.")
        if write.partial and not tailed:
            flag(write, "tail",
                 "which quantises only a slice of the row; no cache slot "
                 "keeps the rest unquantised (`unquantized_tail`).")
    # Each buffer the forward fills quantised is at least one cached tensor
    # the spec stores quantised. Fewer means some buffer was read at its
    # allocation dtype. Which one the names cannot say, so the hint names
    # every quantised write of a buffer beyond the count.
    buffers = sorted({w.buffer for w in writes})
    if len(buffers) > len(quantised):
        for write in writes:
            flag(write, "quantised",
                 f"which stores it quantised; the forward fills "
                 f"{len(buffers)} buffers quantised ({buffers}) and the "
                 f"spec stores {len(quantised)} cached tensors quantised "
                 f"({sorted(quantised)}).")
    return findings
