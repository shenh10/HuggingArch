"""Peak activation memory via DAG liveness analysis.

Classical algorithm for activation footprint on a computation DAG:
  1. Topologically order the leaf nodes.
  2. For each tensor T produced at step i, its last consumer j is the
     max index over every leaf that reads T (or `len(steps)` if T is a
     block output — survives past the block boundary).
  3. Walk in order. At step i add T_i to the live set; sum of live
     tensor sizes is the instantaneous footprint. Peak = max over steps.
  4. After step i, retire every tensor whose last_consumer == i.

This yields the upper-bound activation memory an executor can't go below
without recompute / offload. Used as the basis for the
`estimated_max_gpu_memory_fraction` recommendation — the highest value
the user can safely set for `gpu_memory_fraction` while leaving room for
this activation peak + a small framework overhead (CUDA context, NCCL
buffers, dispatcher pools).

Substrate: one exact ``Analysis``. Its physical shapes plus the program's
edges carry everything liveness needs — each leaf's identity
(`aggregation_id`), its typed `inputs_obj` (producer backrefs, alias
identities for refs that name an enclosing scope's input), and its
output shape. The old shard-on-read replay threaded the LM head's logits into
every MTP stack's hidden input, so on the eleven MTP-bearing models the
reported peak belonged to an edge the replay invented, not to the model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend.arch.eval.refusal import node_context
from backend.arch.eval.activation_edges import leaf_input_ids, tensor_id
from backend.arch.eval.cache import Edges

import sympy as sp
from backend.arch.eval.analysis import Point

_P_act, _P_ovh, _M_tot = sp.symbols("P_act P_overhead M_total")


def activation_peak_fraction_expr() -> sp.Expr:
    """The GPU memory fraction left for batch / activations after peak
    activations + framework overhead: `1 - (P_act + P_ovh) / M_tot`.
    Spec-independent — the inputs are scalar bytes."""
    return 1 - (_P_act + _P_ovh) / _M_tot


def activation_peak_fraction_value(peak_act: int, overhead: int,
                                   mem_total: int) -> float:
    """Numeric eval of the fraction, clamped to [0.1, 1.0]."""
    raw = float(activation_peak_fraction_expr().subs({
        _P_act: int(peak_act),
        _P_ovh: int(overhead),
        _M_tot: int(mem_total),
    }))
    return max(0.1, min(1.0, raw))


# Reserved per-GPU for framework overhead (CUDA context, NCCL, dispatcher
# pools). A conservative headroom that keeps `1 - estimated_max_fraction`
# large enough even on small cards.
FRAMEWORK_OVERHEAD_BYTES = 1 * 1024 ** 3  # 1 GiB

# The `_FUSED_ATTENTION_OPS` hard-coded whitelist was retired here: the fusion
# rewrite structurally eliminates the same `attention_score` + `softmax` intermediates by
# rewriting the naive `[qk_score, softmax, attn_apply]` chain into a
# single `flash_attention` leaf with the same downstream output bytes.
# `tests/backend/inference/test_activation_peak_physical.py` proves byte-equivalence and
# stays as the permanent retirement guard — if anyone re-introduces a
# whitelist OR breaks the fusion path, that test fails loudly.


# ---------------------------------------------------------------------------
# Leaf extraction over the eval analysis
# ---------------------------------------------------------------------------


def _eval_leaf_records(
    program,
    shapes,
    precision,
    *,
    rewrite=None,
    edges=None,
    top_level: bool = False,
    differing: Optional[list] = None,
) -> Tuple[
    Dict[Tuple[int, str], list],
    Dict[Tuple[int, str], Dict[str, int]],
]:
    """The liveness rows and external-input bytes per block template.

    `top_level` also groups each stack's model-level ops (outside every
    block) as one liveness group of their own, keyed `(-1, stack)`: an
    encoder's step can be model-level ops alone (gemma-4's audio
    projection) or carry them around its blocks (a `patchify`, a merger).

    Build one `{id, op, inputs, out_bytes}` row per leaf, with inputs resolved to the
    producer's identity or, for a ref that names an enclosing scope's input,
    to that alias. External tensors retain their own physical input size;
    treating every alias as one model-wide residual invents geometry for
    auxiliary and multi-input blocks.

    Each produced tensor is priced at the producing leaf's role through
    the precision profile (`ResolvedPrecision.edge_bytes`). A fused node's
    ports use the absorbed member that publishes that port. A block's
    external residual has no producer in this template and takes the
    model's own width.

    Each byte count is a number where the shapes are a point's, and a law
    in the workload's counts where they are a walk's (`_tensor_bytes`).
    Two readers of one external input that state its size as different
    laws are appended to `differing` (`(owner, sid, first, second)`), for
    the point to decide; without `differing`, or between numbers, they are
    refused here.
    """
    from backend.arch.eval.layer import Rewrite

    rewrite = rewrite or Rewrite()

    ownership: Dict[int, Tuple[int, str]] = {}
    for contribution in program.contributions:
        if contribution.count == 0:
            continue
        if contribution.block_index is None:
            if not top_level:
                continue
            key = (-1, contribution.stack)
        else:
            key = (contribution.block_index,
                   contribution.node.template or "")
        pending = [contribution.node]
        while pending:
            owned = pending.pop()
            previous = ownership.setdefault(owned.uid, key)
            if previous != key:
                raise ValueError(
                    f"activation node uid {owned.uid} belongs to block "
                    f"invocations {previous[0]} and {key[0]}"
                )
            pending.extend(owned.children or ())

    for fused_uid in (rewrite.origins or {}):
        if program.by_uid(fused_uid) is None:
            continue
        # A launch belongs to the block invocation of the origin it runs
        # with (`Rewrite.home`, `FusedNode.home`): for one across a layer
        # seam, the reading layer's.
        home = rewrite.home(fused_uid)
        if home in ownership:
            ownership[fused_uid] = ownership[home]

    # The edge → producer join, made once in eval (absorption-resolved),
    # per iteration class of each leaf (`activation_edges.leaf_input_ids`):
    # a template whose layers run different launches is simulated once per
    # class, each its own group (`(block, template, class)`).
    input_ids = leaf_input_ids(program, rewrite, edges=edges)

    by_template: Dict[Tuple[Any, ...], list] = {}
    external_by_template: Dict[Tuple[Any, ...], Dict[str, int]] = {}

    def _record(node, owner, inputs) -> None:
        shape = shapes.values.get(node.uid)
        main_bpe = precision.edge_bytes(node.role,
                                        program.stack_of(node))
        out_bytes = _tensor_bytes(
            shape.out if shape is not None else None, main_bpe)
        outputs = {tensor_id(node.uid, "out"): out_bytes}
        if shape is not None:
            for port, dims in shape.ports.items():
                outputs[tensor_id(node.uid, port)] = _tensor_bytes(
                    dims, _port_bpe(node, port, precision, program))
        by_template.setdefault(owner, []).append({
            "id": node.aggregation_id, "uid": node.uid, "op": node.op,
            "inputs": inputs,
            "out_bytes": out_bytes,
            "outputs": outputs})
        refs = tuple(node.inputs_obj or ())
        if shape is None:
            if refs:
                raise ValueError(
                    f"{node.display_path}: activation leaf has inputs but no "
                    "physical shape"
                )
            return
        if len(refs) != len(shape.ins):
            raise ValueError(
                f"{node.display_path}: activation input refs/shapes have "
                f"arity {len(refs)} != {len(shape.ins)}"
            )
        external = external_by_template.setdefault(owner, {})
        for ref, dims in zip(refs, shape.ins, strict=True):
            if ref.src is not None or not ref.external_sid:
                continue
            input_bytes = _tensor_bytes(
                dims, precision.default_edge_bytes)
            previous = external.setdefault(ref.external_sid, input_bytes)
            if previous != input_bytes and differing is not None and (
                    isinstance(previous, sp.Basic)
                    or isinstance(input_bytes, sp.Basic)):
                differing.append(
                    (owner, ref.external_sid, previous, input_bytes))
            elif previous != input_bytes:
                raise ValueError(
                    f"block invocation {owner[0]} ({owner[1]!r}) external "
                    f"input {ref.external_sid!r} "
                    f"has conflicting physical sizes {previous} and "
                    f"{input_bytes} bytes"
                )

    for node in program.nodes():
        if node.children:
            continue
        with node_context(node.display_path):
            block_owner = ownership.get(node.uid)
            if block_owner is None:
                continue
            for iterations, inputs in input_ids[node.uid].items():
                _record(node, (*block_owner, _class_key(iterations)), inputs)

    return by_template, external_by_template


def _class_key(iterations) -> Tuple[Any, ...]:
    """A class of iterations as a key: its iterations, in order."""
    return tuple(sorted(iterations, key=lambda it: (it is None, it or 0)))



def _tensor_bytes(dims, bytes_per_element):
    """One tensor's bytes: its extents, each floored (`as_extent`), times
    its element width, whole — an `int` where the extents are numbers, the
    same as a law where they are a walk's (an empty shape is no tensor)."""
    from backend.arch.eval.dim import as_extent, value_of

    if not dims:
        return 0
    values = [value_of(dim) for dim in dims]
    if any(isinstance(v, sp.Basic) and v.free_symbols for v in values):
        return sp.floor(sp.Mul(*(sp.floor(v) for v in values))
                        * sp.Rational(bytes_per_element))
    elements = 1
    for dim in dims:
        elements *= as_extent(dim, rounding="floor")
    return int(elements * bytes_per_element)


def _port_bpe(node, port, precision, program) -> float:
    """Width of one published tensor: the member that emits that port."""
    from backend.arch.eval.launch import FusedNode
    publishers = node.publishers() if isinstance(node, FusedNode) else ()
    for origin, _origin_port, name in publishers:
        if (name or "out") == (port or "out"):
            member = program.by_uid(origin)
            if member is None:
                return precision.default_edge_bytes
            return precision.edge_bytes(member.role, program.stack_of(member))
    return precision.edge_bytes(node.role, program.stack_of(node))


# ---------------------------------------------------------------------------
# Liveness walker
# ---------------------------------------------------------------------------


def _peak_for_leaves(
    records: List[dict],
    external_bytes: Mapping[str, int],
) -> dict:
    """Run liveness over a single block's leaf records.

    Returns `{peak_bytes, peak_node_id, peak_step, num_steps,
    top_live_at_peak}`. Inputs are already resolved producer
    uids (via `inputs_obj.src`) so no `_resolve_input`
    heuristic is needed — the dict-lookup either hits a sibling
    producer or falls through to the external-residual bucket.

    Block-output survival isn't tracked explicitly: by construction
    the block's declared `outputs:` IS the residual stream (the final
    `add(x, mlp_out)` in a pre-norm transformer block), which the
    `ext_last` bookkeeping already keeps live throughout the walk via
    `live[@ext:x] = external_bytes[x]`. Tracking outputs separately
    would either double-count residual or extend a within-block tensor
    past the last consumer in the SAME block — there's no downstream
    consumer in this single-block analysis to keep it alive for.
    """
    if not records:
        return {"peak_bytes": 0, "peak_node_id": None, "peak_step": None,
                "num_steps": 0, "top_live_at_peak": []}

    id_to_idx = {
        output_id: i
        for i, row in enumerate(records)
        for output_id in row.get("outputs", {row["id"]: row.get("out_bytes", 0)})
    }

    # last_consumer[producer_id] = highest step index that reads the output.
    # Default: self-step (a tensor is live at least when it is produced).
    last_consumer: dict = dict(id_to_idx)
    # External inputs (block-level residual "x") tracked separately.
    ext_last: dict = {}

    for i, r in enumerate(records):
        for src in r["inputs"]:
            if src in id_to_idx:
                last_consumer[src] = max(last_consumer[src], i)
            elif src:
                ext_last[src] = max(ext_last.get(src, -1), i)

    live: dict = {}
    for ext in ext_last:
        if ext not in external_bytes:
            raise ValueError(
                f"activation liveness has no physical size for external "
                f"input {ext!r}"
            )
        live[f"@ext:{ext}"] = external_bytes[ext]

    peak_bytes = sum(live.values())
    peak_step = -1
    peak_node_id: Optional[str] = None
    live_at_peak: List[Tuple[str, int]] = []

    for i, r in enumerate(records):
        outputs = r.get("outputs", {r["id"]: r.get("out_bytes", 0)})
        for output_id, output_bytes in outputs.items():
            out_b = int(output_bytes or 0)
            if out_b > 0:
                live[output_id] = out_b
        cur = sum(live.values())
        if cur > peak_bytes:
            peak_bytes = cur
            peak_step = i
            peak_node_id = r["id"]
            live_at_peak = sorted(live.items(), key=lambda kv: -kv[1])[:5]
        for pid, lc in list(last_consumer.items()):
            if lc == i and pid in live:
                del live[pid]
        for ext, lc in list(ext_last.items()):
            key = f"@ext:{ext}"
            if lc == i and key in live:
                del live[key]

    uid_name = {r["uid"]: r["id"] for r in records}
    port_owner = {str(uid): name for uid, name in uid_name.items()}

    def _live_name(key):
        """A tensor's published name: its producer's id, a named port's
        `#port=<port>` after it (`tensor_id`), an external input's alias."""
        if isinstance(key, str) and key.startswith("@ext:"):
            return key[5:]
        if isinstance(key, str) and "#port=" in key:
            uid, port = key.split("#port=", 1)
            return f"{port_owner.get(uid, uid)}#port={port}"
        return uid_name.get(key, key)

    return {
        "peak_bytes": peak_bytes,
        "peak_node_id": peak_node_id,
        "peak_step": peak_step,
        "num_steps": len(records),
        "top_live_at_peak": [{"id": _live_name(k), "bytes": v}
                             for k, v in live_at_peak],
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def activation_peak_at(step, *, framework_binding=None,
                       overhead_bytes: int = FRAMEWORK_OVERHEAD_BYTES) -> dict:
    """A prefill or encode step's activation peak: its cell's liveness
    (`Cell.liveness`, folded off the cell's walk once) read at the step's
    point. The liveness reads the whole model's serving layers, timing the
    stacks the walk accounts for (`Program.included_stack_names`) — those
    the step's own rows time."""
    walk = step.walk
    if walk.rule.phase not in ("prefill", "encode"):
        raise ValueError(
            "activation capacity requires a prefill or encode analysis, "
            f"got {walk.rule.phase!r}")
    liveness = step.cell.liveness(framework_binding,
                                  walk.program.included_stack_names())
    return liveness.at(step.point, overhead_bytes=overhead_bytes)


class ActivationLiveness:
    """One step's activation liveness, folded off its cell's serving layers
    once.

    Which tensor is live at which step of a block is the graph's — the
    serving graph the forward runs (`ServingLayers.program`), its leaves in
    order and the edges between them — and holds at every point of the
    cell. Only the bytes move with the workload: each tensor's is a law in
    the workload's counts (`_tensor_bytes`), written into the cell's book
    (`Cell.laws`) beside the rows' laws, and `at` reads them all at a point
    in one call and runs the liveness there (`_peak_for_leaves`).
    """

    def __init__(self, serving, book) -> None:
        from backend.arch.layout.physical import physical_axis_partitions

        analysis = serving.analysis
        self._analysis = analysis
        self._book = book
        with analysis.assumed():
            precision = analysis.resolved_precision
            self._batch = self._law(analysis.binding.raw("B"))
            self._seq_len = (None if analysis.rule.phase == "encode"
                             else self._law(analysis.binding.raw("s_ctx")))
            self._divided_by = {
                axis: self._law(degree)
                for axis, degree in physical_axis_partitions(analysis).items()}
            #: External inputs whose sizes two readers state as different
            #: laws: equal or not at a point, which `at` checks.
            self._differing: List[tuple] = []
            records_by_template, external_by_template = _eval_leaf_records(
                serving.program,
                serving.shapes,
                precision,
                rewrite=serving.rewrite,
                edges=Edges(analysis, serving.program, serving.rewrite),
                # An encode step's model-level ops are its own work.
                top_level=analysis.rule.phase == "encode",
                differing=self._differing,
            )
        self._fused_ops = tuple(sorted({
            m["fused_op"] for m in serving.fusion_matches
        }))
        duplicate_names = {
            name
            for _index, name, _class in records_by_template
            if len({key[0] for key in records_by_template
                    if key[1] == name}) > 1
        }
        # A block whose layers run different launches is simulated once per
        # iteration class (`Rewrite.iteration_classes`), each a group of its
        # own, and `at` publishes the worst class under the block's name.
        #: `(label, records, external bytes)` per block invocation, every
        #: byte count a number or its law.
        self._blocks: List[Tuple[str, List[dict], Dict[str, object]]] = []
        for key, records in records_by_template.items():
            block_index, template, _iterations = key
            label = (
                f"{template} (model-level ops)" if block_index == -1
                else f"{template}#block={block_index}"
                if template in duplicate_names else template
            )
            # A block simulated once per iteration class publishes its worst
            # class under the block's own name (`at`): the classes are how
            # the liveness is computed, not a new kind of block.
            self._blocks.append((
                label,
                [{**record,
                  "out_bytes": self._law(record["out_bytes"]),
                  "outputs": {tensor: self._law(nbytes) for tensor, nbytes
                              in record["outputs"].items()}}
                 for record in records],
                {sid: self._law(nbytes) for sid, nbytes
                 in external_by_template.get(key, {}).items()}))
        self._differing = [
            (owner, sid, self._law(first), self._law(second))
            for owner, sid, first, second in self._differing]
        self._texts = tuple(dict.fromkeys(
            law for law in self._laws() if isinstance(law, sp.Basic)))

    def _law(self, value):
        """A number as itself; a law in the workload's counts as the law
        the cell's book holds (`LawBook.write`)."""
        from backend.arch.eval.dim import value_of

        value = value_of(value)
        if isinstance(value, sp.Basic):
            if value.free_symbols:
                return self._book.write(value)
            return int(value)
        return value

    def _laws(self):
        yield self._batch
        yield self._seq_len
        yield from self._divided_by.values()
        for _label, records, external in self._blocks:
            for record in records:
                yield record["out_bytes"]
                yield from record["outputs"].values()
            yield from external.values()
        for _owner, _sid, first, second in self._differing:
            yield first
            yield second

    def at(self, point, *, overhead_bytes: int = FRAMEWORK_OVERHEAD_BYTES
           ) -> dict:
        """The liveness at `point` (the workload's counts by name; `{}` off
        an analysis that is already a point's): each block's peak, the
        worst block's, and what drives it."""
        if self._texts:
            self._analysis.admit(point)
            read = Point(point).laws(self._book, self._texts)

            def value(law):
                return int(read[law]) if isinstance(law, sp.Basic) else law
        else:
            def value(law):
                return law
        for (block_index, template, _class), sid, first, second in \
                self._differing:
            if value(first) != value(second):
                raise ValueError(
                    f"block invocation {block_index} ({template!r}) external "
                    f"input {sid!r} "
                    f"has conflicting physical sizes {value(first)} and "
                    f"{value(second)} bytes"
                )
        batch = value(self._batch)
        seq_len = None if self._seq_len is None else value(self._seq_len)
        divided_by = {axis: value(degree)
                      for axis, degree in self._divided_by.items()}
        by_block: Dict[str, dict] = {}
        external_by_block: Dict[str, Dict[str, int]] = {}
        for label, records, external in self._blocks:
            external_bytes = {sid: value(nbytes)
                              for sid, nbytes in external.items()}
            peak = _peak_for_leaves(
                [{**record,
                  "out_bytes": value(record["out_bytes"]),
                  "outputs": {tensor: value(nbytes) for tensor, nbytes
                              in record["outputs"].items()}}
                 for record in records],
                external_bytes,
            )
            # One entry per block: its worst iteration class.
            if (label not in by_block
                    or peak["peak_bytes"] > by_block[label]["peak_bytes"]):
                by_block[label] = peak
                external_by_block[label] = external_bytes
        if not by_block:
            return {"peak_activation_bytes_per_rank": 0, "peak_by_block": {},
                    "batch": batch, "seq_len": seq_len,
                    "divided_by": divided_by,
                    "overhead_bytes": overhead_bytes,
                    "fused_ops": ()}

        # Worst-case block drives the recommendation.
        worst_name = max(by_block, key=lambda k: by_block[k]["peak_bytes"])
        worst = by_block[worst_name]
        residual_bytes = max(
            external_by_block.get(worst_name, {}).values(),
            default=0,
        )

        return {
            "peak_activation_bytes_per_rank": worst["peak_bytes"],
            "worst_block": worst_name,
            "peak_node_id": worst["peak_node_id"],
            "num_steps": worst["num_steps"],
            "top_live_at_peak": worst["top_live_at_peak"],
            "batch": batch,
            "seq_len": seq_len,
            # Which axes the walk actually divided, and by how much. Replaces
            # the single `tp_shard` the display used to print as `÷N`: there
            # is no one N, and naming the axes is the only form of the
            # answer that stays true for a model whose attention and FFN
            # shard differently.
            "divided_by": divided_by,
            "residual_bytes": residual_bytes,
            "overhead_bytes": overhead_bytes,
            "fused_ops": self._fused_ops,
            "peak_by_block": {k: v["peak_bytes"] for k, v in by_block.items()},
        }


def estimated_max_fraction(
    *,
    peak_activation_bytes: int,
    overhead_bytes: int,
    mem_total_bytes: int,
) -> dict:
    """Recommend the highest safe value for `gpu_memory_fraction`.

    The existing budget model in capacity.py treats `fraction × total`
    as the pool for weights + KV cache, so the remaining
    `(1 − fraction) × total` must cover runtime activation + framework
    overhead. The estimate below gives the user the ceiling — set
    higher and you risk OOM during prefill.
    """
    reserved = int(peak_activation_bytes) + int(overhead_bytes)
    if mem_total_bytes <= 0:
        return {"fraction": 0.0, "reserved_bytes": reserved,
                "mem_total_bytes": mem_total_bytes}

    # Render through the module's own fraction expression
    # so the tooltip matches every other formula on the Inference tab.
    # P_act / P_ovh / M_tot are opaque scalar inputs — liveness is a DAG
    # algorithm, so there's no closed-form dim expansion here.
    expr = activation_peak_fraction_expr()
    clamped = activation_peak_fraction_value(
        peak_activation_bytes, overhead_bytes, mem_total_bytes,
    )
    symbolic_str = str(expr).replace("**", "^")

    def _fmt(v: int) -> str:
        return str(int(v))
    substituted = symbolic_str
    for name, val in (("P_act", peak_activation_bytes),
                      ("P_ovh", overhead_bytes),
                      ("M_tot", mem_total_bytes)):
        substituted = substituted.replace(name, _fmt(val))

    return {
        "fraction": clamped,
        "reserved_bytes": reserved,
        "mem_total_bytes": mem_total_bytes,
        "formula": {
            "label": "Estimated max GPU memory fraction",
            "symbolic": symbolic_str,
            "substituted": substituted,
            "result": f"{clamped:.3f}",
            "description": "frac = 1 − (P_act + P_ovh) / M_tot, clamped to [0.1, 1.0]",
        },
    }
