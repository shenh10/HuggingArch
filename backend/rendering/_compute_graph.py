"""The final whole-model compute graph, serialized for web/src/pages/ComputeGraph.tsx.

What the inference page's ``/inference/compute-graph`` endpoint renders: the
POST-PASS IR tree (parallelism comm injection + kernel fusion applied), with
each block stack collapsed into one expandable ``×count`` compound, fused
leaves carrying their pre-fusion subgraph as children, engine-computed shapes
splatted onto every node, and per-stage (pp_rank) slices derived from the
whole.

Dependency contract: rendering is the presentation layer. It may READ arch
and inference facts (``FUSION_SCHEMES``, a stage's ``StageView``); neither may
import rendering. Nothing here re-derives a fact the walk already knows —
shapes come from the analysis layers, styling from `_consts`, node dicts from
the same `NodeBuilder` the Inspector uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.arch.ops import is_out_in_matrix, is_side_buffer
from backend.arch.eval.dim import as_extent, expr_of
from backend.arch.eval.program import BlockStage, OpStage
from backend.arch.op_names import is_view_name
from backend.arch.spec_tree import iter_tree

from backend.inference.fusion_plans import FUSION_SCHEMES

from ._consts import STYLE_BY_OP, STYLE_BY_ROLE
from ._node_build import NodeBuilder, strip_transport_keys
from ._shapes import _fmt_shape, _fmt_shape_factored, build_edges, dim_glossary



# Zero-cost layout/view ops are `op_names.is_view_name` — primitives via
# `OpSpec.is_view_op`, head_split/head_broadcast/head_merge via
# `view_only_components()`. `concat` is not a view (it materializes).

# A view is absorbed into the adjacent op that executes it (its host's
# prologue/epilogue) — grounded in vLLM's DeepseekV2Attention.forward:
# layout ops are free reinterprets fused into the neighbouring
# elementwise/materialize kernel (rope writes the rotated slice, the
# rmsnorm reads the reshaped input, `k = k.new_empty(); k[..]=k_nope;
# k[..]=k_pe` materializes + broadcasts k_pe). GEMMs and the attention
# kernel are NOT hosts — they read prepared tensors via strides, so a view
# feeding/following them stays a standalone free reinterpret. Which ops
# host is derived from their declarations (`ops.hosts_views`; a fused op
# answers for the ops it absorbed) and stamped by `NodeBuilder` as
# `hosts_views`.

def _producer_id(ref: str, candidate_ids) -> str:
    """An input ref is `<sid>` or `<sid>.<port>`; return the matching
    candidate id (sids contain dots, so match against known ids)."""
    if ref in candidate_ids:
        return ref
    head = ref.rsplit(".", 1)[0]
    return head if head in candidate_ids else ref


def _lifted_input_roles(fused: Dict[str, Any]) -> List[Optional[str]]:
    """The roles a fused node's outer inputs play inside it: each outer
    input takes the role an absorbed node (its `input_roles`, stamped by
    `NodeBuilder` from the declaration) reads it in. Empty when none does."""
    inner: Dict[str, Dict[str, Any]] = {}

    def walk(ns) -> None:
        for n in ns:
            if n.get("id"):
                inner[str(n["id"])] = n
            walk(n.get("children") or ())

    walk(fused.get("children") or ())
    role_for_src: Dict[str, str] = {}
    for n in inner.values():
        roles = n.get("input_roles") or ()
        for src, role in zip(n.get("inputs") or (), roles, strict=False):
            if role and src not in inner:
                role_for_src[str(src)] = role
    lifted = [role_for_src.get(str(s)) for s in fused.get("inputs") or ()]
    return lifted if any(lifted) else []


def _absorb_view_ops(siblings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Absorb each zero-cost view op into the adjacent elementwise /
    materialize op that executes it (its host's prologue/epilogue), per
    the vLLM convention. The view moves into the host's children —
    collapsed the host reads as itself, expanded it reveals the layout
    work it does internally. Views feeding/following a GEMM or the
    attention kernel are NOT absorbed (those read prepared tensors via
    strides) — they stay as standalone free reinterprets. Cost is
    unchanged (views are 0/0).

    Bottom-up; only same-scope, single-consumer views are moved.
    """
    for n in siblings:
        if n.get("children"):
            n["children"] = _absorb_view_ops(n["children"])

    ids = {n["id"] for n in siblings}
    by_id = {n["id"]: n for n in siblings}
    consumers: Dict[str, List[Dict[str, Any]]] = {n["id"]: [] for n in siblings}
    for n in siblings:
        for inp in (n.get("inputs") or []):
            pid = _producer_id(inp, ids)
            if pid in consumers and pid != n["id"]:
                consumers[pid].append(n)

    def is_view(n: Dict[str, Any]) -> bool:
        return is_view_name(n["op"])

    def host_of(n, seen):
        cons = consumers.get(n["id"], [])
        if len(cons) != 1:
            return None
        c = cons[0]
        if c["id"] in seen:
            return None
        if is_view(c):
            return host_of(c, seen | {c["id"]})
        return c if c.get("hosts_views") else None

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for n in siblings:
        if is_view(n):
            h = host_of(n, {n["id"]})
            if h is not None and h["id"] in ids:
                groups.setdefault(h["id"], []).append(n)

    moved: set = set()
    for hid, views in groups.items():
        host = by_id[hid]
        absorbed = {v["id"] for v in views}
        moved |= absorbed
        # Rewire: drop host inputs that pointed at an absorbed view, add
        # the absorbed views' external producers (the layout is now
        # internal to the host). `inputs` and `input_ports` are parallel
        # arrays (see _node_build.py:161-166) — must rebuild both in
        # lockstep, otherwise port info on the absorbed view's source
        # (e.g. `kv_a_proj_with_mqa.output_0`) gets dropped and the
        # frontend `expandDst` can't route the cross-scope edge.
        new_inputs: List[str] = []
        new_ports: List[str] = []
        seen_refs: set = set()
        def _add(ref, port):
            if ref not in seen_refs:
                seen_refs.add(ref)
                new_inputs.append(ref)
                new_ports.append(port)
        host_inputs = host.get("inputs") or []
        host_ports = host.get("input_ports") or [""] * len(host_inputs)
        for inp, port in zip(host_inputs, host_ports, strict=True):
            if _producer_id(inp, ids) not in absorbed:
                _add(inp, port)
        for v in views:
            v_inputs = v.get("inputs") or []
            v_ports = v.get("input_ports") or [""] * len(v_inputs)
            for vinp, vport in zip(v_inputs, v_ports, strict=True):
                if _producer_id(vinp, ids) not in absorbed:
                    _add(vinp, vport)
        host["inputs"] = new_inputs
        host["input_ports"] = new_ports
        host["children"] = views + (host.get("children") or [])
        host["expandable"] = True

    return [n for n in siblings if n["id"] not in moved]


def _format_axes(axes) -> str:
    """Render per-axis symbolic exprs (e.g. ['B','S','N*D']) for the
    tooltip. Multi-input form is list-of-lists, single-input is flat —
    same shape convention as `_shapes._fmt_shape`. Multiplication keeps `*`
    so the rendered axis label is copy-paste safe."""
    def _pretty(a):
        return str(a).replace(" ", "")
    if not axes:
        return ""
    if isinstance(axes[0], (list, tuple)):
        return ", ".join("[" + ",".join(_pretty(a) for a in sub) + "]" for sub in axes)
    return "[" + ",".join(_pretty(a) for a in axes) + "]"


def _build_shape_index(program, shapes, full_shapes, name_to_value=None):
    """Physical and semantic shape layers, keyed by analysis uid."""
    def dims(axes):
        return ([as_extent(axis, rounding="floor") for axis in axes]
                if axes is not None else None)

    def axes(axes_value):
        return [str(expr_of(axis)) for axis in axes_value] \
            if axes_value is not None else None

    def inputs(shape):
        if shape is None or not shape.ins:
            return None
        values = [dims(row) for row in shape.ins]
        return values[0] if len(values) == 1 else values

    def input_axes(shape):
        if shape is None or not shape.ins:
            return None
        values = [axes(row) for row in shape.ins]
        return values[0] if len(values) == 1 else values

    idx: Dict[int, Dict[str, Any]] = {}
    for uid, shape in shapes.values.items():
        if program.by_uid(uid) is None:
            continue
        full = full_shapes.values.get(uid)
        so, so_axes = dims(shape.out), axes(shape.out)
        si, si_axes = inputs(shape), input_axes(shape)
        full_out = dims(full.out) if full is not None else None
        full_in = inputs(full)
        entry: Dict[str, Any] = {}
        if so:
            entry["tensor_out"] = _fmt_shape_factored(
                so, so_axes, name_to_value
            )
            if full_out and full_out != so:
                entry["tensor_out_full"] = _fmt_shape_factored(
                    full_out, so_axes, name_to_value
                )
            if so_axes:
                entry["tensor_out_symbolic"] = _format_axes(so_axes)
        if si:
            entry["tensor_in"] = _fmt_shape(si)
            if full_in and full_in != si:
                entry["tensor_in_full"] = _fmt_shape(full_in)
            if si_axes:
                entry["tensor_in_symbolic"] = _format_axes(si_axes)
        if shape.ports:
            entry["tensor_outs"] = {
                port: _fmt_shape_factored(
                    dims(row), axes(row), name_to_value
                )
                for port, row in shape.ports.items() if row
            }
            full_ports = full.ports if full is not None else {}
            different = {
                port: _fmt_shape_factored(
                    dims(full_ports[port]), axes(row), name_to_value
                )
                for port, row in shape.ports.items()
                if port in full_ports and dims(full_ports[port]) != dims(row)
            }
            if different:
                entry["tensor_outs_full"] = different
        if entry:
            idx[uid] = entry
    return idx


def _stamp_read_ports(nodes: List[Dict[str, Any]], shapes, full_shapes,
                      name_to_value) -> None:
    """Shape the renderer's input nodes no analysis uid stands for, from the
    port of a node that reads them: a model input (`_model_input`), and a
    block's formal past the carried stream (`formal` > 0) — its tensor is
    not the residual the stack carries. A model input that only a block
    stack reads takes the shape of that stack's formal."""
    def port(uid: Any, index: int) -> Dict[str, Any]:
        shape, full = shapes.values.get(uid), full_shapes.values.get(uid)
        if shape is None or index >= len(shape.ins):
            return {}
        row = shape.ins[index]
        dims = [as_extent(axis, rounding="floor") for axis in row]
        axes = [str(expr_of(axis)) for axis in row]
        out = {"tensor_out": _fmt_shape_factored(dims, axes, name_to_value)}
        if full is not None and index < len(full.ins):
            whole = [as_extent(axis, rounding="floor")
                     for axis in full.ins[index]]
            if whole != dims:
                out["tensor_out_full"] = _fmt_shape_factored(
                    whole, axes, name_to_value)
        return out

    wanted: Dict[str, Dict[str, Any]] = {}
    stamped: List[str] = []

    def collect(ds: List[Dict[str, Any]]) -> None:
        for d in ds:
            if d.get("_model_input") or "formal" in d:
                wanted[d["id"]] = d
            if d.get("_model_input") or d.get("formal", 0) > 0:
                stamped.append(d["id"])
            collect(d.get("children") or ())

    found: Dict[str, Dict[str, Any]] = {}

    def read(ds: List[Dict[str, Any]]) -> None:
        for d in ds:
            uid = d.get("uid")
            if uid is not None:
                for index, ref in enumerate(d.get("inputs") or ()):
                    if ref in wanted and ref not in found:
                        shaped = port(uid, index)
                        if shaped:
                            found[ref] = shaped
            read(d.get("children") or ())

    collect(nodes)
    read(nodes)
    for d in nodes:
        if d.get("op") != "block_stack":
            continue
        formals = {c["formal"]: c["id"] for c in d.get("children") or ()
                   if "formal" in c}
        for index, ref in enumerate(d.get("inputs") or ()):
            inner = formals.get(index)
            if ref in wanted and ref not in found and inner in found:
                found[ref] = found[inner]
    for ref in stamped:
        for key, value in found.get(ref, {}).items():
            wanted[ref].setdefault(key, value)

    def unmark(ds: List[Dict[str, Any]]) -> None:
        for d in ds:
            d.pop("_model_input", None)
            unmark(d.get("children") or ())

    unmark(nodes)


def _attach_tensor_shapes(nodes: List[Dict[str, Any]],
                          shape_index: Dict[int, Dict[str, Any]]) -> None:
    """Splat `tensor_out` / `tensor_in` / `tensor_outs` onto each UI node
    so the frontend ELK renderer can label edges with the producer's
    shape.

    The join is on `uid`. A node's `id` is a path by the time the graph
    is assembled — `_namespace` prefixes `__block__<template>/` — so it
    is not an identity. `uid` is stamped by NodeBuilder and never
    rewritten. Nodes with no uid are renderer constructs (`block_stack`,
    `_block_input`) and pick up a residual-stream shape below.

    Existing fields on the node are not overwritten so callers that
    pre-populated shapes win."""
    for n in nodes:
        info = shape_index.get(n.get("uid"))
        if info:
            for k, v in info.items():
                n.setdefault(k, v)
        if n.get("children"):
            _attach_tensor_shapes(n["children"], shape_index)


def serialize_post_tree(post_tree, resolved_spec, *,
                         pre_tree, fusion_matches=(),
                         program, shapes, full_shapes,
                         pre_program, pre_shapes, pre_full_shapes,
                         kv=None,
                         ) -> Dict[str, Any]:
    """Serialize the post-passes IR tree into ONE whole-model graph the
    inference page renders with its ELK renderer (web/src/pages/ComputeGraph.tsx).

    Node dicts are built by the SAME `NodeBuilder` the Inspector uses
    (`rendering._node_build`) — only the source tree differs: the
    Inspector walks `iter_tree(resolved)` (raw spec), this feeds the
    POST-PASS tree (kernel fusion + parallelism comm injection applied)
    via `NodeBuilder.build_roots`. So styling (op/role → CSS class),
    nested-component expansion, and the UI dict shape are identical to
    the Inspector; comm primitives pick up the shared `s-comm` style.

    A FUSED op is modeled as a compound that REPLACED its subgraph: the
    post-tree leaf is re-styled `fused` and given `children` = the
    pre-fusion subgraph it absorbed (pulled from `pre_tree` via the
    `diagnostics` match records), so collapsed it reads as the fused
    kernel and expanded it reveals the original score→softmax→apply (or
    allreduce→add→rmsnorm, …) it stands in for — the same compound /
    expand machinery every other component uses.

    On top of those dicts we assemble the whole-model overview in
    spine order — `embed_tokens → [decoder block ×N] → norm →
    lm_head` — collapsing each block STACK into one expandable compound
    (labeled `<stack> ×count`). Repetition (N identical layers) is the
    box + its count, not N unrolled copies. `assignments:` stacks split
    into their per-template blocks (DSv3: dense ×3 → MoE ×58); aux heads
    (MTP) branch off the last main block.

    `kv` is the deployment's `KVMarks` (`KVMarks.of_serving`): every node
    that writes or reads the cache carries ``kv``, and cache edges carry
    ``kind: "cache"``.
    """
    if program is None or shapes is None or full_shapes is None:
        raise ValueError(
            "serialize_post_tree needs program, shapes, and full_shapes"
        )
    if pre_program is None or pre_shapes is None or pre_full_shapes is None:
        raise ValueError(
            "serialize_post_tree needs the pre-fusion program and shapes"
        )
    if kv is None:
        raise ValueError(
            "serialize_post_tree needs the deployment's KV marks: a graph "
            "without them draws no cache"
        )
    # Per-rank weight dims for linear leaves under TP, derived from the
    # forward-transform per-rank shapes: a linear's weight is [in, out] where
    # in = last dim of its per-rank input, out = last dim of its per-rank
    # output (column-parallel shrinks out; row-parallel shrinks in). Empty at
    # TP=1 (Inspector / single-rank) → falls through to the spec IR's globals.
    def _last_dim(shape):
        if not shape:
            return None
        head = shape[0]
        row = head if isinstance(head, (list, tuple)) else shape
        return int(row[-1]) if row else None
    sharded_weight_dims: Dict[int, Dict[str, int]] = {}

    def _index_linear(prog, layer) -> None:
        for uid, shape in layer.values.items():
            node = prog.by_uid(uid)
            if node is None or not is_out_in_matrix(node.declaration):
                continue
            shape_in = [list(row) for row in shape.ins]
            wi, wo = _last_dim(shape_in), _last_dim(shape.out)
            if wi is not None and wo is not None:
                sharded_weight_dims[uid] = {"in": wi, "out": wo}

    # Pre first so absorbed linears (inside qkv_parallel_linear etc.)
    # have a per-rank W; post second so a rebased shape wins.
    _index_linear(pre_program, pre_shapes)
    _index_linear(program, shapes)
    node_builder = NodeBuilder(resolved_spec, STYLE_BY_OP, STYLE_BY_ROLE,
                               sharded_weight_dims=sharded_weight_dims)

    # fused_uid → LIST of absorbed-subgraph aggregation_id
    # tuples (the pre-fusion nodes each fused leaf replaced). A LIST,
    # not a single tuple: a hybrid model fuses the SAME template-local
    # spec_id in several blocks (DSv3 dense + MoE both fuse
    # `post_attention_layernorm`), so there's one MatchInfo per block —
    # and they differ only in the block-specific comm node id
    # (post_attn_add__comm1 vs __comm3). Each block picks the tuple
    # whose ids all resolve in ITS pre-tree index. Join on fused_uid:
    # the rewrite uid is the one identity the program node and the tree
    # replacement always share.
    fused_matches: Dict[int, List[tuple]] = {}
    for match in fusion_matches:
        fused_matches.setdefault(match["fused_uid"], []).append(
            tuple(match["matched_aggregation_ids"]))

    # Refs absorbed by a PARALLEL fusion (e.g. mlp.up_proj absorbed by
    # merged_column_parallel_linear on mlp.gate_proj at port output_1).
    # The fusion rewrite updates live consumers' resolved_inputs at match
    # time, but the absorbed subgraph of a DIFFERENT fusion (silu_and_mul's
    # pre-fusion children silu / mul) is pulled out of pre_idx without
    # that rewrite, so refs to a parallel-fused sibling dangle. Map:
    # absorbed-agg_id → (anchor_agg_id, "output_<i>"). Built once;
    # consumed by `_subscope_fused`.
    # The anchor is the fused node as the tree draws it, joined by the
    # rewrite uid like `fused_matches` above: a parallel fusion's match
    # names one sibling (`gate_proj`) while the tree node is another
    # (`up_proj`), and a ref redirected to the match's name reaches no node.
    tree_id_of_uid: Dict[int, str] = {}

    def _index_tree(node) -> None:
        tree_id_of_uid[node.uid] = node.aggregation_id
        for child in node.children or ():
            _index_tree(child)

    for root in post_tree.top_level:
        _index_tree(root)
    for _meta, roots in post_tree.blocks:
        for root in roots:
            _index_tree(root)

    def _anchor(match) -> str:
        if match["fused_uid"] not in tree_id_of_uid:
            raise ValueError(
                f"fusion {match['fused_op']!r} at "
                f"{match['fused_aggregation_id']!r} has no node in the "
                "fused tree to anchor its absorbed outputs")
        return tree_id_of_uid[match["fused_uid"]]

    absorbed_to_port: Dict[str, tuple] = {}
    for match in fusion_matches:
        scheme = FUSION_SCHEMES.get(match["fused_op"])
        if scheme and scheme.naive_pattern.is_parallel():
            for i, sib_sid in enumerate(match["matched_aggregation_ids"]):
                absorbed_to_port[sib_sid] = (_anchor(match), f"output_{i}")
        # Nodes-form fusions with an exposed primary or named secondary
        # output port (a norm's optional cast is the primary; fused_qk_rope
        # exposes q_out / k_out): a node ABSORBED here is still
        # referenced by a node that lives inside a DIFFERENT fused
        # compound (attention_score inside flash_attention reads the
        # pre-fusion q_rope). Since `_expand_fused` rebuilds that
        # compound's children from the PRE-fusion tree, those refs point
        # at the absorbed node and must be redirected to
        # `<anchor>.<port>`. Mirrors the parallel case above.
        for absorbed_sid, port_name in match.get("absorbed_output_ports", ()):
            absorbed_to_port[absorbed_sid] = (_anchor(match), port_name)

    # Per-template pre-fusion node index: {aggregation_id: pre SpecIRNode}.
    # Built per block for scope locality (each block's _expand_fused only
    # consults its own pre_idx), but keyed by aggregation_id so lookups
    # against `matched` tuples (which carry aggregation_id) are direct
    # equality. Top-level nodes have bare display_path == aggregation_id.
    pre_index_by_template: Dict[str, Dict[str, Any]] = {}
    if pre_tree is not None:
        def _index(node, idx):
            idx[node.aggregation_id] = node
            for c in (node.children or ()):
                _index(c, idx)
        for meta, roots in pre_tree.blocks:
            idx: Dict[str, Any] = {}
            for r in roots:
                _index(r, idx)
            # Keyed by the invocation (`BlockMeta.label`): the groups of one
            # template that see different histories are different blocks.
            pre_index_by_template.setdefault(meta.label, idx)
        top_idx: Dict[str, Any] = {}
        for r in pre_tree.top_level:
            _index(r, top_idx)
        pre_index_by_template["__top__"] = top_idx

    def _expand_fused(dicts: List[Dict[str, Any]],
                      pre_idx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Re-style fused leaves blue and give them the pre-fusion
        subgraph as children (a compound that replaced its subgraph).

        Also prunes the OTHER matched siblings from this scope. Cost
        treats every layer sharing a template-local spec_id as fused
        (`count_per_step` aggregates across templates). A historical
        template-key collision could graft only one block, so a hybrid
        model's collision-victim block (DSv3 MoE shares
        `self_attn.attn_apply` with the dense block) keeps its matched
        qk_score / softmax as tree siblings. Without pruning they'd show
        BOTH standalone AND inside the fused compound. Pruning makes
        every block render fused, consistent with the roofline."""
        prune: set = set()
        for d in dicts:
            sid = d.get("id")
            candidates = fused_matches.get(d.get("uid"), [])
            # Pick the match whose absorbed ids all resolve in THIS
            # block's pre-tree (disambiguates dense vs MoE by their
            # block-specific comm node id); fall back to the first.
            matched = next(
                (t for t in candidates if all(x in pre_idx for x in t)),
                candidates[0] if candidates else None,
            )
            if matched:
                d["style"] = "fused"
                pre_nodes = [pre_idx[m] for m in matched if m in pre_idx]
                # A launch that took a component whole (a view-only wrapper
                # it read through, `fusion.ViewChains`) absorbed the wrapper
                # and its leaves; the wrapper is drawn with its leaves
                # inside it, so a leaf under another absorbed node is not
                # drawn a second time beside it.
                taken = {id(node) for node in pre_nodes}

                def _under_taken(node) -> bool:
                    parent = node.parent
                    while parent is not None:
                        if id(parent) in taken:
                            return True
                        parent = parent.parent
                    return False

                pre_nodes = [n for n in pre_nodes if not _under_taken(n)]
                if pre_nodes:
                    # Sub-scope the absorbed subgraph under the fused
                    # node's own id. The fused node inherits its anchor
                    # member's spec_id (e.g. `post_attention_layernorm`,
                    # or the first member of a parallel fusion), which
                    # ALSO appears among the matched children — so the
                    # compound and one of its children would collide
                    # without a sub-prefix. Only INTERNAL refs (to other
                    # matched nodes) get the prefix; external refs (to the
                    # block-level producers — self_attn / residual x) are
                    # left bare so the later block-level `_namespace`
                    # resolves them to the real block nodes, so the
                    # compound's input edges reach its entry
                    # children. (Over-prefixing them was leaving the
                    # absorbed add's residual input dangling.)
                    d["children"] = _subscope_fused(
                        node_builder.build_roots(pre_nodes), sid, pre_idx)
                    d["expandable"] = True
                    d["fused"] = d.get("op")  # mark: this compound IS a fused op
                    # A fused kernel that ate the attention score / apply
                    # plays their parts: its outer inputs take the roles the
                    # absorbed anchors gave them, so the structure view
                    # still groups at the component.
                    if not d.get("input_roles"):
                        lifted = _lifted_input_roles(d)
                        if lifted:
                            d["input_roles"] = lifted
                    # Parallel fusion: each child sibling produces ONE
                    # of the compound's output ports (sibling[i] →
                    # `output_<i>`). The annotation lets an outer-scope
                    # edge with `src_port=output_<i>` resolve to the
                    # SPECIFIC inner exit child — without it, an output_1
                    # edge from `gate_proj` to silu_and_mul.mul has only
                    # "some exit child" to go to, and fanning it to every
                    # exit smears it across both gate_proj_inner and
                    # up_proj.
                    scheme = FUSION_SCHEMES.get(d.get("op"))
                    if scheme and scheme.naive_pattern.is_parallel():
                        for i, c in enumerate(d["children"]):
                            c["output_port"] = f"output_{i}"
                    # Other matched siblings (not this anchor) → prune,
                    # except a member the launch took only some runs of: it
                    # keeps a node of its own for the rest.
                    match = next(
                        (m for m in fusion_matches
                         if m["fused_uid"] == d.get("uid")), {})
                    kept = set(match.get("kept_aggregation_ids", ()))
                    prune.update(m for m in matched
                                 if m != sid and m not in kept)
                    continue  # children are the absorbed subgraph, don't recurse
            if d.get("children"):
                d["children"] = _expand_fused(d["children"], pre_idx)
        # A launch is never one of another launch's absorbed members, even
        # where it stands under a member's name: DeepSeek-V3's first dense
        # layer quantises its input norm in its own launch
        # (`fused_rmsnorm_quant`, at `input_layernorm`), while the seam
        # launch took that norm's other runs.
        launches = {m["fused_uid"] for m in fusion_matches}
        return [d for d in dicts
                if d.get("id") not in prune or d.get("uid") in launches]

    def _prepend_block_input_nodes(
        children: List[Dict[str, Any]], inputs: List[str],
    ) -> List[Dict[str, Any]]:
        """Prepend a virtual `input` node per declared block input (formal
        0 receives the stack's carried hidden state), mirroring the Inspector's
        per-block builder (`backend/rendering/builders.py`).

        Rationale: without an explicit producer for the block input, a
        consumer that reads `x` only renders its `x` edge if it happens
        to be a scope ENTRY (no in-scope producer) — the frontend's
        entry-fan then routes the block_stack's backbone edge into it.
        That breaks for the post-attention residual `add`: under TP>1 it
        fuses into `allreduce_fused_rmsnorm` (all_reduce → add → rmsnorm),
        so the add gains an in-scope producer (the all_reduce) and is no
        longer an entry — its block-input `x` residual then has no path
        and renders as a dangling / missing arrow. Giving `x` a real,
        resolvable node makes every `x` consumer (entry or not, nested in
        a fused compound or not) receive its edge through the ordinary
        build_edges + expandDst machinery. The backbone fans into this
        input node (the sole entry), which then feeds both
        input_layernorm and the residual add.

        Shape is stamped later (post `_attach_tensor_shapes`) from the
        block's shape-preserving residual width — see the input-node
        shape pass below.
        """
        # Collect the bare refs actually consumed inside this scope, so we
        # only materialize input nodes that something reads.
        consumed: set = set()

        def _collect(ds: List[Dict[str, Any]]) -> None:
            for d in ds:
                for inp in (d.get("inputs") or ()):
                    consumed.add(inp)
                _collect(d.get("children") or ())

        _collect(children)
        input_nodes: List[Dict[str, Any]] = []
        for formal, inp_name in enumerate(inputs):
            if inp_name not in consumed:
                continue
            input_nodes.append({
                "id": inp_name,
                "op": "input",
                "label": "input",
                "style": "io",
                "inputs": [],
                "input_ports": [],
                "_block_input": True,  # marker for the shape-stamp pass
                # Which formal: the block_stack's `inputs[formal]` is the
                # tensor bound to it, so an outer edge lands on this node
                # (the frontend's entry fan cannot tell two formals apart).
                # Past the carried stream, its tensor is its readers' input
                # port (`_stamp_read_ports`).
                "formal": formal,
            })
        return input_nodes + children

    def _subscope_fused(child_dicts: List[Dict[str, Any]],
                        sid: str,
                        pre_idx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sub-scope an absorbed fused subgraph under the fused node's
        id. Prefixes child ids with `sid/`, and an input ref with `sid/`
        ONLY when it points at another node WITHIN the subgraph
        (internal edge). External refs — to the block-level producers
        the subgraph reads (self_attn partial, residual x) — are left
        bare so the later block-level `_namespace` resolves them to the
        real block nodes (the compound's input edge then reaches these
        entry children).

        External refs that pointed at a node which has SINCE been
        absorbed by a parallel fusion (e.g. silu_and_mul's `mul` reading
        `mlp.up_proj`, which got merged into `mlp.gate_proj` at port
        `output_1`) are redirected to `<anchor>.<port>` via
        `absorbed_to_port`. Without this rewrite the ref dangles since
        the absorbed sibling no longer exists in the post-fusion tree."""
        internal: set = set()
        def _collect(ds):
            for d in ds:
                internal.add(d["id"])
                _collect(d.get("children") or ())
        _collect(child_dicts)

        def rewrite(d: Dict[str, Any]) -> None:
            inputs = list(d.get("inputs") or [])
            ports = list(d.get("input_ports") or ([""] * len(inputs)))
            if len(ports) < len(inputs):
                ports = ports + [""] * (len(inputs) - len(ports))
            new_inputs: List[str] = []
            new_ports: List[str] = []
            for inp, port in zip(inputs, ports, strict=False):
                # The producer is the ref less the port the ref declares
                # (`input_ports`), not less whatever follows its last dot:
                # a parameter named `hc_norm.weight` is not port `weight`
                # of a member named `hc_norm`.
                pid_internal = (inp[:-len(port) - 1]
                                if port and inp.endswith(f".{port}") else inp)
                if pid_internal in internal:
                    new_inputs.append(f"{sid}/{inp}")
                    new_ports.append(port)
                    continue
                pid_abs = _producer_id(inp, absorbed_to_port)
                if pid_abs in absorbed_to_port:
                    anchor_sid, anchor_port = absorbed_to_port[pid_abs]
                    new_inputs.append(
                        f"{anchor_sid}.{anchor_port}" if anchor_port
                        else anchor_sid)
                    new_ports.append(anchor_port)
                    continue
                # A copied pre-fusion member can name a COMPONENT wrapper as
                # its producer. Building the full tree would drop that wrapper
                # by module absorption and redirect this ref to its surviving
                # ancestor; building only `matched` roots has no parent to do
                # that. Ask the block's complete pre-tree which producer the
                # unfused graph would show, so the fused copy does not dangle.
                pid_pre = _producer_id(inp, pre_idx)
                producer = pre_idx.get(pid_pre)
                while (producer is not None
                       and producer.parent is not None
                       and producer.module_name is None
                       and producer.provenance == "spec"
                       and producer.children):
                    producer = producer.parent
                if producer is not None:
                    visible = producer.aggregation_id
                    if visible != pid_pre:
                        new_inputs.append(visible + inp[len(pid_pre):])
                        new_ports.append(port)
                        continue
                new_inputs.append(inp)
                new_ports.append(port)
            d["inputs"] = new_inputs
            d["input_ports"] = new_ports
            d["id"] = f"{sid}/{d['id']}"
            for c in (d.get("children") or ()):
                rewrite(c)
        for d in child_dicts:
            rewrite(d)
        return child_dicts

    def _namespace(dicts: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
        """Prefix every node id + input ref in a block's subtree with the
        block's unique id.

        spec_ids are unique only WITHIN one block template — the
        Inspector renders each template separately so it never collides.
        This whole-model graph puts dense / MoE / MTP blocks in ONE
        canvas, where `self_attn` / `mlp` / `input_layernorm` recur in
        every block. Without namespacing, the renderer's id-keyed maps collapse
        the duplicates: expand clicks resolve to the wrong node and edges
        route to a sibling block. Prefixing id AND inputs uniformly keeps
        every intra-block edge intact (both endpoints get the same
        prefix) while making ids globally unique. Ports ride in the
        parallel `input_ports` array, so prefixing the whole `inputs`
        string is safe."""
        def rewrite(d: Dict[str, Any]) -> None:
            d["id"] = f"{prefix}/{d['id']}"
            if d.get("inputs"):
                d["inputs"] = [f"{prefix}/{inp}" for inp in d["inputs"]]
            for c in (d.get("children") or ()):
                rewrite(c)
        for d in dicts:
            rewrite(d)
        return dicts

    # Index the post_tree's top-level nodes + group block stacks by
    # their spec.model layer name (`stack_name`). One `layers` stack
    # can map to MULTIPLE post_tree blocks (DSv3: dense ×3 + MoE ×58
    # both under stack_name="layers" via `assignments:`); they are drawn
    # in the order the walk runs them (`Stack.walk_order`: a carrying
    # stack's variants by the first iteration each covers), so the chain
    # follows the carried stream and a stack's last group is its exit.
    # The graph is of the step the analysis walked: the language model's
    # stacks, not the encoders (another phase's walk, read here as the
    # tensors they hand over).
    walked = program.walked_stack_names
    top_level = [n for n in post_tree.top_level
                 if program.stack_of(n) in walked]
    top_by_id = {n.display_path: n for n in top_level}
    walk_rank = {stage.template.meta.label: rank
                 for stack in program.stacks
                 for rank, stage in enumerate(stack.walk_order())
                 if isinstance(stage, BlockStage)}
    # What each invocation's formals are bound to (`Stage.upstream`, by
    # position). Formal 0 of a carrying stack reads the carried stream, drawn
    # as the chain below; the formals past it read these.
    upstream_of = {stage.template.meta.label: stage.upstream
                   for stack in program.stacks
                   for stage in stack.walk_order()
                   if isinstance(stage, BlockStage)}
    blocks_in_walk = sorted(
        ((meta, roots) for meta, roots in post_tree.blocks
         if roots and program.stack_of(roots[0]) in walked),
        key=lambda block: walk_rank[block[0].label])
    stack_blocks: Dict[str, List[Any]] = {}
    for meta, roots in blocks_in_walk:
        stack_blocks.setdefault(meta.stack_name or meta.template_name, []) \
            .append((meta, roots))

    nodes: List[Dict[str, Any]] = []
    # Backbone chain: ids of the MAIN residual-stream nodes, so we wire
    # `prev → next` sequential edges. Aux heads (MTP) and side buffers
    # (rotary inv-freq) branch off but don't advance the backbone.
    backbone: List[str] = []
    last_main_block: List[str] = []  # last non-aux block id (MTP branches off it)

    # Declared cross-stack wiring (7.2): each aux group's `inputs:` and each
    # main layer's `inputs:` name the stages they read. The evaluator already
    # resolves these in dependency order; the graph draws them through one
    # table of the names a reader can use for something drawn outside its
    # own block: a model input, a model-level node, a stack or an aux group
    # (its LAST block group — the one whose output the name stands for; a
    # stack split by `assignments:` draws dense → MoE, and a reader of the
    # stack reads the MoE group), and an appended stream (its writer stack).
    # One name naming two endpoints is refused.
    endpoints: Dict[str, str] = {}

    def _endpoint(name: str, drawn: str) -> None:
        if endpoints.setdefault(name, drawn) != drawn:
            raise ValueError(
                f"{name!r} names two endpoints of the graph: "
                f"{endpoints[name]!r} and {drawn!r}")

    stack_last: Dict[str, str] = {}
    group_last: Dict[str, str] = {}
    for meta, _roots in blocks_in_walk:
        stack_last[meta.stack_name or meta.template_name] = \
            f"__block__{meta.label}"
        if meta.group_name:
            group_last[meta.group_name] = f"__block__{meta.label}"
    for name in program.walked_inputs:
        _endpoint(name, name)
    handed = [h for h in program.handoffs if h.stack not in walked]
    for handoff in handed:
        _endpoint(handoff.name, handoff.name)
    for name, drawn in (*stack_last.items(), *group_last.items()):
        _endpoint(name, drawn)
    for node in top_level:
        _endpoint(node.display_path, node.display_path)
    # A model-level op a launch replaced is read by the name its spine entry
    # declares, and the launch stands at that entry's place in the spine
    # (`layer_path`): a launch across the last layer's seam takes the final
    # norm and is named for the seam (`norm__seam@block`).
    spine_names: Dict[tuple, str] = {}

    def _spine_ops(entries) -> None:
        for entry in entries:
            if entry.kind == "op":
                spine_names[tuple(entry.layer_path)] = entry.name
            _spine_ops(entry.children or ())

    _spine_ops(iter_tree(program.resolved).spine)
    for node in top_level:
        name = spine_names.get(tuple(node.layer_path or ()))
        if name is not None and name != node.display_path:
            _endpoint(name, node.display_path)
    for stream in post_tree.streams.values():
        if stream.stack_name in stack_last:
            _endpoint(stream.name, stack_last[stream.stack_name])
    path_last: Dict[tuple, str] = {}
    for meta, _roots in blocks_in_walk:
        path_last[tuple(meta.layer_path)] = f"__block__{meta.label}"
    # Which box holds each node of the walk, so a fact that names a uid can
    # be drawn at the block that owns it.
    block_of_uid: Dict[int, str] = {}
    for meta, roots in blocks_in_walk:
        pending = list(roots)
        while pending:
            held = pending.pop()
            block_of_uid[held.uid] = f"__block__{meta.label}"
            pending.extend(held.children or ())
    # The names a model-level reader settles in ITS OWN scope, ahead of the
    # table above: inside an aux group its stages go by their local names
    # (Nemotron's MTP head reads `layers`, the group's stack), and an inner
    # stream belongs to the stack in the reader's scope, never to the graph
    # as a whole.
    group_scope: Dict[str, Dict[str, str]] = {}
    # The program's spine: its aux heads carry their sampled views.
    for entry in iter_tree(program.resolved).spine:
        if entry.kind != "group":
            continue
        local = {child.name: path_last[tuple(child.layer_path)]
                 for child in entry.children
                 if tuple(child.layer_path) in path_last}
        for child in entry.children:
            if child.kind == "op":
                group_scope[child.name] = dict(local)
    # A launch that replaced a group's op reads in that op's scope: it
    # stands at the op's place in the spine (`layer_path`).
    for node in top_level:
        name = spine_names.get(tuple(node.layer_path or ()))
        if name in group_scope and node.display_path != name:
            group_scope[node.display_path] = group_scope[name]
    # A stack's INNER stream read from outside it: DeepSeek-V4.1's model-level
    # `hc_exit` reduces the backbone's final `hc_pre_mix`. Which write that
    # is, is the walk's own answer, asked once
    # (`Program.inner_streams_read_after_their_stack`); the renderer only maps
    # the uid it names to the box drawn around it.
    #
    # Per reader, not one name for the graph: stream names are authored and
    # the scope is what makes them unique, so two groups may each declare `s`
    # and each read their own. That is why the walk answers per reader; a
    # single global name → endpoint entry would instead refuse the whole
    # graph for the legal case.
    #
    # And NOT the last block drawn for the stack: drawing order is the walk's
    # schedule over templates, and an `assignments:` stack draws one box per
    # template, so the box that happens to come last need not hold the final
    # layer's write at all. On V4.1-Flash the final `hc_pre_mix` write is
    # layer 39, in the `[compress_ratio=1]` selection template, while the
    # last box drawn for that stack is the shared-keys one — display order
    # would hang the exit reduce off a region that does not own the write.
    for reader, streams in program.inner_streams_read_after_their_stack().items():
        for sid, final in streams.items():
            if final.uid in block_of_uid:
                group_scope.setdefault(reader, {})[sid] = block_of_uid[final.uid]
    group_inputs: Dict[str, List[str]] = {}
    # `resolved_spec`'s spine, not the spliced walk's: sampled-head views
    # stay a safety-net extra, not a backbone hop.
    spine = iter_tree(resolved_spec).spine
    for entry in spine:
        if entry.kind == "group" and entry.aux_kind:
            group_inputs[entry.name] = list(entry.inputs)

    # Model-level ops a fused kernel absorbed along with a later one
    # (`SpecIRTree.fused_layers`): a ref to one reads what it read, the
    # kernel's own inputs.
    absorbed: Dict[str, Any] = {}

    def _note_absorbed(entries) -> None:
        for entry in entries:
            if tuple(entry.layer_path) in post_tree.fused_layers:
                absorbed[entry.name] = entry
            _note_absorbed(entry.children)

    _note_absorbed(spine)

    def _through(names) -> List[str]:
        out: List[str] = []
        for name in names:
            if name in absorbed:
                out.extend(_through(absorbed[name].inputs))
            else:
                out.append(name)
        return out

    def _resolve_refs(names) -> List[str]:
        # A name outside the table is a node drawn by its own path (a leaf
        # inside a model-level component); `_settle_refs` below refuses a
        # name that is neither.
        return [endpoints.get(n, n) for n in names]

    # What the model receives from outside its graph (`model_inputs:`): one
    # boundary input each, read by name like any stage. Not an IR node — its
    # tensor is the one its readers' input ports carry (`_stamp_read_ports`).
    nodes.extend({"id": name, "op": "input", "label": name,
                  "style": "io", "inputs": [], "input_ports": [],
                  "_model_input": True}
                 for name in program.walked_inputs)
    # What an encoder hands the language model (`Program.handoffs`): the
    # encoder runs in its own encode walk, so the step receives the tensor
    # at the boundary, as it receives a model input.
    nodes.extend({"id": handoff.name, "op": "input",
                  "label": f"{handoff.name} (from the {handoff.stack} encoder)",
                  "style": "io", "inputs": [], "input_ports": [],
                  "_model_input": True}
                 for handoff in handed)

    # Whole-stream joins a model-level op runs first (`OpStage.prelude`,
    # laid down by `Program.lowered`): drawn just ahead of the op, which is
    # drawn reading the join. Their elements — and a one-element stream the
    # op reads as is — are drawn from the stack that appends the stream
    # (`Program.whole_stream_pairings`), once every stack is placed: an aux
    # tower may be placed after the op that reads it.
    top_by_uid = {n.uid: n for n in top_level}
    prelude_of = {
        stage.name: [top_by_uid[n.uid] for n in stage.prelude]
        for stack in program.stacks if stack.name in walked
        for stage in stack.stages
        if isinstance(stage, OpStage) and stage.prelude}
    lowered_uids = {n.uid for nodes_ in prelude_of.values() for n in nodes_}
    joins_of: Dict[str, List[Any]] = {}
    for row in program.whole_stream_pairings:
        if (row.representation == "concat" and not row.reader_iterations
                and row.reader_stack in walked):
            joins_of.setdefault(row.reader, []).append(row)

    joined_readers: List[Dict[str, Any]] = []

    def _drawn_from_writers(d: Dict[str, Any],
                            writers: Dict[str, str]) -> None:
        d["inputs"] = [writers.get(inp, inp) for inp in d.get("inputs") or ()]
        d["input_ports"] = [""] * len(d["inputs"])
        for child in d.get("children") or ():
            _drawn_from_writers(child, writers)

    def _block_is_moe(roots) -> bool:
        """Whether the block routes tokens to experts: a node of it is
        declared in a dispatch zone (`Program.has_dispatch_zone`)."""
        stack = list(roots)
        while stack:
            node = stack.pop()
            if node.in_dispatch_zone:
                return True
            stack.extend(node.children or ())
        return False

    def _emit_block(meta, roots, declared_inputs=None) -> None:
        stack_id = f"__block__{meta.label}"
        # A derived group names the history length its layers see; the
        # groups of one declaration otherwise draw as identical boxes.
        derived = meta.derived_group()
        label = (f"{meta.stack_name or meta.template_name} × {meta.count}"
                 + (f"  [{derived['axis']}={derived['visible']}]"
                    if derived is not None else "")
                 + ("  [loop gate]" if meta.loop_gate
                    else f"  [pass {meta.pass_}]" if meta.pass_ > 1 else "")
                 + (f"  [{meta.aux_kind}]" if meta.aux_kind else ""))
        is_aux = bool(meta.aux_kind)
        # Aux attachment comes from the group's declared `inputs:` (an MTP
        # head reads the backbone norm; a vision tower is READ BY the main
        # stack — the old last-main-block heuristic drew that one backwards).
        # Main blocks chain off the backbone unless the layer declares its
        # own inputs (Kimi's decoder reads its `embed_merge`).
        if declared_inputs:
            inp = declared_inputs
        elif is_aux:
            declared = _resolve_refs(group_inputs.get(meta.group_name or "", []))
            # An aux group that declares no `inputs:` is an ENCODER (a
            # vision/audio tower): its input is an image, not a hidden
            # state, so there is no backbone edge to draw. The old
            # last-main-block fallback drew `embed_tokens → vision_tower`
            # — an encoder reading the text stream, backwards.
            inp = declared
        else:
            inp = [backbone[-1]] if backbone else []
        if inp:
            # One input per formal, in formal order: an MTP head reads the
            # backbone norm AND the token ids, not the norm alone.
            inp = [inp[0], *_resolve_refs(upstream_of.get(meta.label, ())[1:])]
        nodes.append({
            "id": stack_id,
            "op": "block_stack",
            "label": label,
            "style": "moe",          # purple compound — same vocab as Inspector
            "count": meta.count,
            "aux_kind": meta.aux_kind,
            "is_moe": _block_is_moe(roots),
            "inputs": inp,
            "input_ports": [""] * len(inp),  # backbone refs are main outputs
            "children": _namespace(
                _prepend_block_input_nodes(
                    _absorb_view_ops(
                        _expand_fused(
                            node_builder.build_roots(roots),
                            pre_index_by_template.get(meta.label, {}))),
                    # The formals, then the streams the block reads: the
                    # appended ones its template names (`reads:`) and the
                    # stack's inner ones (`StackReads.inner`), which a
                    # component nested in the block may read itself (CSA2's
                    # borrowed compressor reads `compress_kv`). Each is an
                    # input of the block like a formal past the carried
                    # stream; only those something reads are drawn.
                    list(dict.fromkeys(
                        [*meta.inputs, *meta.stream_reads, *meta.reads.inner]))),
                stack_id),
            "expandable": True,
            **({"derived_group": derived} if derived is not None else {}),
        })
        if is_aux:
            return  # aux branch: don't advance the main backbone
        backbone.append(stack_id)
        last_main_block.clear()
        last_main_block.append(stack_id)

    for entry in spine:
        lname = entry.name
        # An encoder group names no attachment of its own; its block stack
        # reads the model input its template stage names.
        attached = _through(entry.inputs) or [
            ref for child in entry.children if child.kind != "op"
            for ref in child.inputs if ref in program.model_inputs]
        layer_inputs = _resolve_refs(attached)
        if lname in stack_blocks:
            for meta, roots in stack_blocks[lname]:
                # A stack's declared `inputs:` applies to its FIRST block;
                # the rest chain off it in walk order.
                _emit_block(meta, roots, declared_inputs=layer_inputs or None)
                layer_inputs = []
        elif lname in top_by_id:
            node = top_by_id[lname]
            *joins, d = _absorb_view_ops(_expand_fused(
                node_builder.build_roots([*prelude_of.get(lname, ()), node]),
                pre_index_by_template.get("__top__", {})))
            nodes.extend(joins)
            if joins_of.get(lname):
                # A joined stream is read through its join; one element
                # read as is (no join) keeps the stream's name until its
                # writer is drawn.
                through = {}
                for row in joins_of[lname]:
                    if row.through_uid is not None \
                            and row.through_uid not in top_by_uid:
                        raise ValueError(
                            f"{lname}: the join it reads {row.stream!r} "
                            "through is not a node of the fused tree")
                    through[row.stream] = (
                        row.stream if row.through_uid is None
                        else top_by_uid[row.through_uid].aggregation_id)
                layer_inputs = [through[name] if name in through
                                else endpoints.get(name, name)
                                for name in _through(entry.inputs)]
                joined_readers.extend([*joins, d])
            if is_side_buffer(node.declaration):
                d["inputs"] = []  # standalone source, off the backbone
                d["input_ports"] = []
            else:
                # An op that declares no input reads its sequential
                # predecessor; one that declares inputs reads them.
                d["inputs"] = (layer_inputs if attached
                               else [backbone[-1]] if backbone else [])
                d["input_ports"] = [""] * len(d["inputs"])
                backbone.append(node.display_path)
            nodes.append(d)
        # else: spine entry with no matching node (skip)

    # Safety net: any top_level node / block not reached via spine
    # order — append standalone so nothing silently vanishes.
    placed = {n["id"] for n in nodes}
    nodes.extend(
        _absorb_view_ops(_expand_fused(
            node_builder.build_roots([node]),
            pre_index_by_template.get("__top__", {})))[0]
        for node in top_level
        if node.display_path not in placed and node.uid not in lowered_uids
    )
    for meta, roots in blocks_in_walk:
        bid = f"__block__{meta.label}"
        if bid not in placed:
            _emit_block(meta, roots)
    # The appending stack as drawn: by its own name, or — inside an aux
    # group — by the group's (`_emit_block`).
    writers: Dict[str, str] = {}
    for rows in joins_of.values():
        for row in rows:
            drawn = (endpoints[row.writer] if row.writer in endpoints
                     else endpoints.get(row.writer_stack))
            if drawn is None:
                raise ValueError(
                    f"stream {row.stream!r}: its writer {row.writer!r} "
                    f"(stack {row.writer_stack!r}) is not drawn")
            for name in (row.stream,
                         *(row.element_ref(j) for j in range(row.visible))):
                writers[name] = drawn
    for d in joined_readers:
        _drawn_from_writers(d, writers)

    # Splat engine-computed shapes onto every node so an edge can be
    # labelled from the producer's `tensor_out`. Keyed by uid, so namespacing
    # of `id` does not matter. Pre and post merge: uids only pre has are
    # the nodes fusion absorbed, which the fused compound's children are
    # rebuilt from. Post is applied second.
    glossary = dim_glossary(program.base_binding)
    shape_index = _build_shape_index(
        pre_program, pre_shapes, pre_full_shapes, name_to_value=glossary,
    )
    shape_index.update(_build_shape_index(
        program, shapes, full_shapes, name_to_value=glossary,
    ))
    _attach_tensor_shapes(nodes, shape_index)
    _stamp_read_ports(nodes, shapes, full_shapes, glossary)

    # Backbone residual-stream shape propagation. The synthetic
    # `block_stack` wrappers (a UI grouping of `count` repeated layers)
    # have no analysis UID, so `_attach_tensor_shapes` leaves them
    # without a `tensor_out` — and the backbone edge into the next stack
    # / final norm then renders shapeless. A repeated block stack is
    # provably shape-preserving (you can only stack `×count` identical
    # layers if each maps [B,S,H] → [B,S,H]), so the residual-stream
    # shape flows unchanged along the backbone. Inherit it from the
    # upstream backbone node for any backbone entry left unshaped; nodes with
    # their own analysis shape (embed_tokens, model.norm,
    # lm_head's [B,S,V]) keep it and re-seed the carry.
    by_id = {n["id"]: n for n in nodes}
    prev_shape = None
    prev_shape_full = None
    for bid in backbone:
        n = by_id.get(bid)
        if n is None:
            continue
        if n.get("tensor_out"):
            prev_shape = n["tensor_out"]
            prev_shape_full = n.get("tensor_out_full")
        elif prev_shape is not None:
            n["tensor_out"] = prev_shape
            n.setdefault("tensor_in", prev_shape)
            if prev_shape_full:
                n.setdefault("tensor_out_full", prev_shape_full)

    # Aux block stacks (vision tower, MTP head) are not on the backbone, so
    # the residual propagation above skips them — but their declared
    # consumers draw real edges (vision_tower → decoder layers), which the
    # render invariant requires a shape for. A stack's product is its last
    # child's output; take that.
    for n in nodes:
        if n.get("op") == "block_stack" and not n.get("tensor_out"):
            children = [c for c in (n.get("children") or [])
                        if not c.get("_block_input")]
            for c in reversed(children):
                if c.get("tensor_out"):
                    n["tensor_out"] = c["tensor_out"]
                    break

    # Stamp the block-input `x` pseudo-nodes with the residual-stream
    # shape. They carry no analysis UID (they are a rendering
    # construct, like the block_stack wrappers), so `_attach_tensor_shapes`
    # leaves them unshaped — and `build_edges` would then raise on the
    # `x → consumer` edge (render invariant: no shapeless arrows). A block
    # is shape-preserving, so the residual entering it equals what feeds
    # the block_stack (its backbone producer) or the stack's own carried
    # tensor_out. Inherit from there.
    for n in nodes:
        if n.get("op") != "block_stack":
            continue
        producer = (n.get("inputs") or [None])[0]
        residual_shape = (
            (by_id.get(producer) or {}).get("tensor_out")
            or n.get("tensor_out")
            or n.get("tensor_in")
        )
        if not residual_shape:
            continue
        for c in (n.get("children") or ()):
            if c.get("_block_input") and not c.get("tensor_out"):
                c["tensor_out"] = residual_shape

    # Every input ref names a node of the graph. A ref the builders left as
    # a name — a stack, an aux group or a stream read from inside a
    # model-level component or a block — is drawn at that name's endpoint;
    # a ref that is neither is refused with the reader, the name and what
    # the table holds, rather than dropped.
    all_ids: set = set()

    def _collect_ids(ns: List[Dict[str, Any]]) -> None:
        for n in ns:
            all_ids.add(n["id"])
            _collect_ids(n.get("children") or [])

    _collect_ids(nodes)

    def _settle_refs(ns: List[Dict[str, Any]], scope: Dict[str, str]) -> None:
        for n in ns:
            names = {**endpoints, **group_scope.get(n["id"], scope)}
            ins = n.get("inputs") or []
            ports = list(n.get("input_ports") or [""] * len(ins))
            ports += [""] * (len(ins) - len(ports))
            settled = []
            for ref, port in zip(ins, ports, strict=True):
                suffix = f".{port}" if port and ref.endswith(f".{port}") else ""
                base = ref[: len(ref) - len(suffix)]
                if base not in all_ids:
                    # A block's refs carry its namespace; the name after it
                    # is the one the reader wrote.
                    local = base.split("/", 1)[1] if base.startswith(
                        "__block__") and "/" in base else base
                    if local not in names:
                        raise ValueError(
                            f"{n['id']}: input {ref!r} names no node of the "
                            f"graph and no stack, group, stream or model "
                            f"input (endpoints: {sorted(names)!r})")
                    base = names[local]
                settled.append(base + suffix)
            if ins:
                n["inputs"] = settled
            _settle_refs(n.get("children") or [],
                         group_scope.get(n["id"], scope))

    _settle_refs(nodes, {})

    # Derive edges at every nesting level: ComputeGraph.tsx consumes `graph.edges[]`
    # as the root-edge source and each compound's `child_edges` as the
    # inner-scope source. The shared `build_edges` walker emits sibling
    # edges per scope, attaches `child_edges` to each compound, and
    # returns the root list — so DSv3's dense/MoE/MTP block_stacks get
    # the proper input_layernorm → self_attn → norm → mlp chain instead
    # of spurious feedthroughs from embed_tokens into every child.
    # Cache marks go on the finished tree: compounds carry the union of
    # their descendants, and the edge builder reads the cache kind off each
    # source node.
    kv.stamp(nodes)
    edges = build_edges(nodes)
    strip_transport_keys(nodes)

    return {"nodes": nodes, "edges": edges}


def slice_pipeline_fragment(graph: Dict[str, Any], stage: Any
                            ) -> Dict[str, Any]:
    """The fragment of the whole-model graph one pipeline stage runs.

    `stage` is the stage's view of the analysis the graph was drawn from
    (`layout.stages.StageView`): a block stack keeps the layers of it the
    stage runs, counted (a stack that spans a cut is drawn in both stages
    with each one's count), and a model-level node is kept where placement
    puts it. A node the graph adds beside the program's (a collective after
    an output head) stays with what it reads, and a model input with the
    nodes that read it. An input produced on an earlier stage arrives
    across the cut: it is drawn as a `recv` boundary node naming the stage
    that sends it — the send/recv the pipeline runs. An input that is
    neither here nor on an earlier stage is refused.
    """
    placement = stage.placement
    program = placement.program

    def producers(node: Dict[str, Any]) -> List[tuple]:
        """`(producer id, port suffix)` per input: a port rides in the
        parallel `input_ports`."""
        ins = node.get("inputs") or []
        ports = list(node.get("input_ports") or [""] * len(ins))
        ports += [""] * (len(ins) - len(ports))
        out = []
        for ref, port in zip(ins, ports, strict=True):
            suffix = f".{port}" if port and ref.endswith(f".{port}") else ""
            out.append((ref[: len(ref) - len(suffix)], suffix))
        return out

    total: Dict[str, int] = {}
    for template in program.blocks:
        label = f"__block__{template.meta.label}"
        total[label] = total.get(label, 0) + template.count
    scope = program.included_stack_names(include_aux=True)

    def kept_on(index: int) -> tuple:
        blocks = placement.block_counts(index)
        held: Dict[str, int] = {}
        for at, template in enumerate(program.blocks):
            label = f"__block__{template.meta.label}"
            held[label] = held.get(label, 0) + blocks[at]
        runs = placement.view(index).rows(scope).values
        placed = {n.aggregation_id: runs.get(n.uid, 0) > 0
                  for n in program.nodes() if n.aggregation_id}
        keep: set = set()
        for node in graph["nodes"]:
            nid = node["id"]
            if node.get("op") == "block_stack":
                if held.get(nid, 0):
                    keep.add(nid)
            elif nid in placed:
                if placed[nid]:
                    keep.add(nid)
            elif any(ref in keep for ref, _ in producers(node)):
                keep.add(nid)
        # A model input goes where it is read.
        for node in graph["nodes"]:
            if node.get("op") == "input" and any(
                    node["id"] == ref
                    for other in graph["nodes"] if other["id"] in keep
                    for ref, _ in producers(other)):
                keep.add(node["id"])
        return keep, held

    keep, held = kept_on(stage.stage)
    earlier = [kept_on(index)[0] for index in range(stage.stage)]

    def sender(ref: str, reader: str) -> int:
        stages = [index for index, kept in enumerate(earlier) if ref in kept]
        if not stages:
            raise ValueError(
                f"stage {stage.stage}: {reader} reads {ref!r}, which no "
                "stage up to it runs")
        return stages[-1]

    by_id = {n["id"]: n for n in graph["nodes"]}
    recv: Dict[str, Dict[str, Any]] = {}

    def arrived(ref: str, reader: str) -> str:
        if ref in keep:
            return ref
        if ref not in recv:
            source = by_id[ref]
            recv[ref] = {
                "id": f"recv/{ref}", "op": "recv",
                "label": f"recv from stage {sender(ref, reader)}",
                "style": "io", "inputs": [], "input_ports": [],
                **{key: source[key] for key in ("tensor_out", "tensor_out_full")
                   if key in source},
            }
        return recv[ref]["id"]

    out_nodes = []
    for node in graph["nodes"]:
        if node["id"] not in keep:
            continue
        d = dict(node)
        d["inputs"] = [arrived(ref, d["id"]) + suffix
                       for ref, suffix in producers(d)]
        if d.get("op") == "block_stack" and held[d["id"]] != total[d["id"]]:
            d["count"] = held[d["id"]]
            d["label"] = d["label"].replace(
                f"× {total[d['id']]}",
                f"× {held[d['id']]} of {total[d['id']]}", 1)
        out_nodes.append(d)
    out_edges = []
    for e in graph["edges"]:
        if e.get("dst") not in keep:
            continue
        if e.get("src") not in keep:
            src = arrived(e["src"], e["dst"])
            e = {**e, "src": src,
                 "id": f"{src}->{e['dst']}:{e['id'].rsplit(':', 1)[-1]}"}
        out_edges.append(e)
    layers = placement.ranges[stage.stage]
    return {"nodes": [*recv.values(), *out_nodes], "edges": out_edges,
            "stage": {"pp": placement.pp, "rank": stage.stage,
                      "layers": [layers.start, layers.stop],
                      "total_layers": placement.n_iterations,
                      "label": stage.describe()}}
