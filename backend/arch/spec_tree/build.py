"""`iter_tree` — the walk from authored spec to `SpecIRTree`.

`SpecTreeWalker` expands component invocations, `<param>` bindings and
multiplicity into the IR tree; `finalize_ir_tree` then wires parents and
uids, the dispatch zone, and the checkpoint weight names. This file is
the whole of how the tree comes to be — the types it builds live in
`nodes`, the questions answered off the tree live in `projections`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from backend.arch.ops import COMBINE_OP, DISPATCH_OP, is_identity_source
from backend.arch.schema import (CheckpointAddress, HISTORY_AXIS, OWN_NAME,
                                 resolve_config_bool)
from backend.arch import dims as dims_registry
from backend.arch.block_label import differing_slots, distinct_block_labels
from backend.arch.ops import OP_REGISTRY
from backend.arch.eval.dim import eval_dim_expr, names_of, value_of
from backend.arch.eval.binding import rederived_values
from backend.arch.spec_id_compose import compose_weight_name

from backend.arch.eval.refusal import SpecRefusal
from backend.arch.dim_resolution import config_path_value
from backend.arch.schema.model import (LayerMap, NearestLayer,
                                       StreamLayerRef, relative_distance)

from .nodes import (
    BlockMeta, InputRef, SpecIRNode, SpecIRTree, SpineEntry, StackReads,
    STACK_ENTRY, StreamAt, TreeStream,
)


def absent_repeat(count: int) -> bool:
    """A resolved repeat of 0 means this stack is not in the checkpoint.

    Omitted or unresolved repeats stay 1 (the historical default). Callers
    skip the stack when this is true so every consumer — leaves, blocks,
    rendering — sees the same absence.
    """
    return int(count) <= 0


def _copy_recipe(ir_node: SpecIRNode, graph: Any) -> None:
    """Copy nested-recipe facts off `graph` onto the call node.

    Same object the walk expands; the graph itself is not stored.
    """
    ir_node.formals = tuple(graph.inputs or ())
    ir_node.secondary_outputs = tuple(
        (k, str(v)) for k, v in (graph.secondary_outputs or {}).items())
    ir_node.stream_out = tuple(
        (k, str(v)) for k, v in (graph.stream_out or {}).items())
    ir_node.stream_reads = tuple(graph.reads or ())
    ir_node.kv_cached = tuple(graph.kv_cached or ())
    ir_node.kv_entries = tuple(graph.cache.entries if graph.cache else ())
    ir_node.kv_states = tuple(graph.kv_states or ())
    ir_node.kv_label = graph.kv_label
    ir_node.attn_variant = graph.attn_variant
    ir_node.sparse_dcp = graph.sparse_dcp


def iter_blocks(resolved) -> List[BlockMeta]:
    """Enumerate every block-template invocation in `spec.model` as a
    flat list of BlockMeta records.

    Walks `spec.model` directly — does NOT call SpecEngine. This is
    layer-0 metadata; engine.compute_model and weight_breakdown both
    consume it instead of re-walking the model list.

    Memoized on the ResolvedSpec instance, like iter_leaves.
    """
    if resolved._spec_tree_blocks_cache is not None:
        return resolved._spec_tree_blocks_cache
    out = enumerate_blocks(resolved)
    resolved._spec_tree_blocks_cache = out
    return out


def _reads_of(layer) -> StackReads:
    concat = tuple(
        (name, axis)
        for name in layer.concat_streams
        for axis in (layer.streams[name].axis,)
        if axis is not None)
    inner = tuple(layer.inner_streams)
    carried = layer.carried_stream
    shapes = tuple(
        (name, tuple(str(axis) for axis in spec.shape))
        for name, spec in layer.streams.items()
        if spec.shape)
    return StackReads(
        name=layer.name or "",
        each=tuple(layer.read_streams),
        prefix=tuple(layer.prefix_streams),
        concat=concat,
        inner=inner,
        carried=carried,
        shapes=shapes,
    )


def spine_of_spec(spec) -> Tuple[
        Dict[Tuple[int, ...], Tuple[str, ...]],
        Dict[Tuple[int, ...], StackReads]]:
    """Appended names and read sets by integer `layer_path`. No dim resolve.

    The same walk `stream_spine` / `iter_tree` uses. Corpus pins that have
    only an `ArchSpec` (no config) read this instead of `ModelLayer` stream
    fields.
    """
    appended: Dict[Tuple[int, ...], Tuple[str, ...]] = {}
    stack_reads: Dict[Tuple[int, ...], StackReads] = {}

    def walk(layers, path: Tuple[int, ...]) -> None:
        for index, layer in enumerate(layers or ()):
            here = (*path, index)
            stack_reads[here] = _reads_of(layer)
            if layer.layers:
                walk(layer.layers, here)
                continue
            names = tuple(layer.appended_streams)
            if names:
                appended[here] = names

    walk(spec.model, ())
    return appended, stack_reads


def layer_at(spec, path: Tuple[int, ...]):
    """The `ModelLayer` a `layer_path` locates. Construction pointer, not order."""
    layer = spec.model[path[0]]
    for index in path[1:]:
        layer = layer.layers[index]
    return layer


def model_spine(resolved) -> Tuple[SpineEntry, ...]:
    """Authored interleave of `spec.model`: ops, blocks, nested groups,
    each with its `repeat:` resolved."""

    def repeat_of(layer) -> Optional[int]:
        if layer.repeat is None:
            return None
        value = eval_dim_expr(str(layer.repeat), resolved.dim_values)
        unbound = sorted(names_of(value))
        if unbound:
            raise ValueError(
                f"model layer {layer.name!r}: count expression "
                f"{layer.repeat!r} leaves {unbound} unbound under this "
                f"model's dims")
        return int(value)

    def kind_of(layer) -> Literal["op", "block", "assignments", "group"]:
        if layer.group and layer.layers:
            return "group"
        if layer.op:
            return "op"
        if layer.assignments:
            return "assignments"
        return "block"

    def walk(layers, path: Tuple[int, ...], stack: str,
             aux_kind: Optional[str], group: Optional[str]
             ) -> Tuple[SpineEntry, ...]:
        out: List[SpineEntry] = []
        for index, layer in enumerate(layers or ()):
            here = (*path, index)
            kind = kind_of(layer)
            if kind == "group":
                inner = layer.name
                inner_aux = layer.aux_kind or aux_kind
                children = walk(layer.layers, here, inner, inner_aux, layer.name)
                out.append(SpineEntry(
                    layer_path=here,
                    name=layer.name,
                    kind="group",
                    inputs=tuple(layer.inputs or ()),
                    stack_name=inner,
                    aux_kind=inner_aux,
                    group=layer.name,
                    children=children,
                    repeat=repeat_of(layer),
                ))
                continue
            out.append(SpineEntry(
                layer_path=here,
                name=layer.name,
                kind=kind,
                inputs=tuple(layer.inputs or ()),
                stack_name=stack,
                aux_kind=aux_kind,
                group=group,
                repeat=repeat_of(layer),
            ))
        return tuple(out)

    return walk(resolved.spec.model, (), "", None, None)


def stream_spine(resolved) -> Tuple[
        Dict[str, TreeStream], Dict[Tuple[int, ...], StackReads]]:
    """Appended streams and per-path read sets, copied from the loader
    memo `visibility_groups` already reads. Do not re-resolve iterations."""
    appended = resolved.appended_streams()
    streams: Dict[str, TreeStream] = {}
    _names, stack_reads = spine_of_spec(resolved.spec)

    def walk(layers, path: Tuple[int, ...]) -> None:
        for index, layer in enumerate(layers or ()):
            here = (*path, index)
            if layer.layers:
                walk(layer.layers, here)
                continue
            for name, spec in layer.appended_streams.items():
                rec = appended[name]
                crop = spec.slice
                streams[name] = TreeStream(
                    name=name,
                    stack_name=layer.name,
                    layer_path=here,
                    iterations=rec.iterations,
                    writers=rec.writers,
                    from_=spec.from_ or "",
                    slice=(None if crop is None else (crop.axis, crop.start)),
                    kind="before" if spec.before is not None else "after",
                )

    walk(resolved.spec.model, ())
    return streams, stack_reads


def scope_checkpoint_root(address, enclosing: str) -> str:
    """A group's root: its own declared one, else the enclosing scope's."""
    if address is not None and address.root is not None:
        return address.root
    return enclosing


def with_root(address, root: str):
    """`address` resolved against its scope: carries the root its scope's
    entries hang off, so `checkpoint_prefix` needs no second look at the
    spec. A scope that declares no stack address is root-only."""
    if address is None:
        return CheckpointAddress(root=root)
    return address.model_copy(update={"root": root})


class _BlockEnumerator:
    """One enumerate_blocks() run: dim resolution shared by the three
    layer-form branches (template / assignments / group)."""

    def __init__(self, resolved):
        self.resolved = resolved
        self.spec = resolved.spec
        self.dim_values = dict(resolved.dim_values)
        self.out: List[BlockMeta] = []
        self.streams, self.stack_reads = stream_spine(resolved)
        self.root = self.spec.checkpoint_root

    def reads_at(self, layer_path) -> StackReads:
        path = tuple(layer_path)
        try:
            return self.stack_reads[path]
        except KeyError as exc:
            raise KeyError(
                f"no stack_reads for layer_path {path}: broken tree, "
                f"the enumerator walked a path stream_spine did not"
            ) from exc

    def block_io(self, template_name: str) -> Dict[str, Any]:
        """This invocation's formals, declared output roots, recipe
        stream / cache interface, and the names it is shown by, as
        `BlockMeta` fields.

        Copied from the named `BlockGraph`. `get_block` raising is a
        broken spec and is not swallowed. The graph object itself is not
        stored.

        The wrapper a block calls is what the loader expanded inline
        shorthand into (`synth_template` / `synth_params`), else the
        wrapper a hand-written one-node adapter binds
        (`BlockGraph.adapted_wrapper`); both are looked up in the spec's
        namespace, the one `get_block` reads.
        """
        block = self.spec.get_block(template_name)
        adapter = block.adapted_wrapper(self.resolve) or (None, {})
        return dict(
            inputs=tuple(block.inputs or ("x",)),
            outputs=tuple(block.outputs or ()),
            stream_out=tuple((k, str(v))
                             for k, v in (block.stream_out or {}).items()),
            stream_reads=tuple(block.reads or ()),
            kv_cached=tuple(block.kv_cached or ()),
            wrapper_op=block.synth_template or adapter[0],
            wrapper_params=dict(block.synth_params or adapter[1] or {}),
        )

    def resolve(self, op: str):
        """The block or component `op` names in the spec's namespace."""
        spec = self.spec
        return (spec.blocks or {}).get(op) or (spec.components or {}).get(op)

    def resolve_int(self, expr: Any, absent: int = 1) -> int:
        """An integer count from a dim expression. `absent` is what an
        UNWRITTEN expression means (`repeat:` omitted is one layer); an
        expression that is written and does not resolve is a spec error,
        never `absent` — that is how a typo in `repeat:` became one layer."""
        from backend.arch.eval.dim import eval_dim_expr
        if expr is None:
            return absent
        value = eval_dim_expr(str(expr), self.dim_values)
        unbound = sorted(names_of(value))
        if unbound:
            raise ValueError(
                f"count expression {expr!r} leaves {unbound} unbound under "
                f"this model's dims")
        return int(value)

    def backbone_layer_count(self) -> Optional[int]:
        """The main backbone stack's layer count — the tail index a
        `checkpoint.index: {after: backbone}` aux stack ships its tensors
        under (DSv3 `model.layers.61.*` ← n_layers = 61). Read from the
        backbone model layer's own count binding — its `repeat:`
        expression or explicit assignment indices, the same source the
        spec binds the layer count from everywhere else. None when the
        spec has no backbone stack or the count doesn't resolve."""
        from backend.arch.eval.dim import eval_dim_expr
        for lyr in self.resolved.spec.model:
            if lyr.group:
                continue
            if lyr.assignments:
                return sum(
                    len(g["indices"])
                    for g in self.resolved.resolve_assignment_groups(lyr))
            if lyr.template and lyr.repeat is not None:
                return int(eval_dim_expr(str(lyr.repeat), self.dim_values))
        return None

    def assignment_metas(self, container, count_multiplier: int,
                         **extras) -> None:
        """One BlockMeta per assignment group of a per-layer container."""
        for g in self.resolved.resolve_assignment_groups(container):
            count = count_multiplier * len(g["indices"])
            if absent_repeat(count):
                continue
            raw = {k: str(v) for k, v
                   in (g["dim_overrides"] or {}).items()}
            ovr = {k: self.resolve_int(v) for k, v in raw.items()}
            site = (f"layer {container.name!r} assignment "
                    f"{g['assignment_index']} ({g['template']})")
            for k, v in raw.items():
                self.resolved.refuse_literal_rebinding(site, k, v, ovr[k],
                                                       rebinds=True)
            path = extras.get("layer_path", ())
            self.out.append(BlockMeta(
                template_name=g["template"],
                count=count,
                layer_indices=tuple(g["indices"]),
                dim_overrides=ovr,
                dim_overrides_raw=raw,
                assignment_index=g["assignment_index"],
                reads=self.reads_at(path),
                **self.block_io(g["template"]),
                **extras,
            ))

    def template_branch(self, layer, layer_path) -> None:
        try:
            count = self.resolve_int(layer.repeat, absent=1)
        except ValueError as exc:
            raise ValueError(f"model layer {layer.name!r}: {exc}") from exc
        if absent_repeat(count):
            return
        site = self.resolved.site
        self.out.append(BlockMeta(
            tied_to=layer.call,
            tied_to_path=(
                (_named_earlier(self.resolved.spec.model[:layer_path[0]],
                                layer.call),)
                if layer.call is not None else None),
            template_name=layer.template,
            count=count,
            count_raw=(None if layer.repeat is None else str(layer.repeat)),
            stack_name=layer.name,
            aux_kind=None,
            checkpoint=self.stack_address(layer),
            layer_indices=None,
            dim_overrides={},
            layer_path=layer_path,
            address=site.address if site is not None else None,
            reads=self.reads_at(layer_path),
            **self.block_io(layer.template),
        ))

    def stack_address(self, layer) -> CheckpointAddress:
        """Where a model-level stack's tensors live: its declared address,
        else its own name under the model's root — the rule a model-level
        op's name follows. A stack without `repeat:` has no index, and its
        tensors hang off the root itself."""
        if layer.repeat is None:
            return CheckpointAddress(root=self.root)
        declared = layer.checkpoint or CheckpointAddress(
            prefix=".".join(part for part in (self.root, layer.name) if part))
        return with_root(declared, self.root)

    def tail_index_base(self, checkpoint) -> Optional[int]:
        """The backbone's layer count when `checkpoint` indexes after it —
        the tail index its tensors ship under starts there — else None."""
        follows_backbone = checkpoint is not None and checkpoint.follows_backbone
        return self.backbone_layer_count() if follows_backbone else None

    def group_branch(self, layer, layer_path) -> None:
        group_repeat = self.resolve_int(layer.repeat) if layer.repeat else 1
        # The group's declared checkpoint address, plus — for an index
        # after the backbone — the tail index it needs. Resolved here
        # (the one place that sees the whole `model:` list) and carried
        # on the meta, so `checkpoint_prefix(meta)` stays the single
        # implementation and needs no second look at the spec.
        root = scope_checkpoint_root(layer.checkpoint, self.root)
        group_checkpoint = with_root(layer.checkpoint, root)
        group_backbone_layers = self.tail_index_base(group_checkpoint)
        for sub_index, sub in enumerate(layer.layers):
            # Every stack of a group is the group's: its name is the stack
            # the program buckets it under and `select: {stack:}` names. Where
            # its tensors live is its checkpoint address, not its name.
            stack = layer.name
            # Sub-layer with `assignments:` — heterogeneous templates
            # share ONE HF stack (e.g. MiMo-V2.5 vision: 4 full-attn +
            # 24 sink-windowed blocks all live under `visual.blocks.X.*`).
            # Mirror the top-level assignments path, but carry the
            # group's aux_kind and the sub-layer's stack_name through
            # so emitted leaves bucket correctly.
            if sub.assignments:
                self.assignment_metas(
                    sub, group_repeat,
                    outer_repeat=group_repeat,
                    outer_repeat_raw=(None if layer.repeat is None
                                      else str(layer.repeat)),
                    stack_name=stack,
                    aux_kind=layer.aux_kind,
                    group_name=layer.name,
                    checkpoint=group_checkpoint,
                    backbone_layers=group_backbone_layers,
                    layer_path=(*layer_path, sub_index),
                    # A call of a group: each assignment runs the called
                    # group's at the same place.
                    **({"tied_to": layer.call,
                        "tied_to_path": (_named_earlier(
                            self.resolved.spec.model[:layer_path[0]],
                            layer.call), sub_index)}
                       if layer.call is not None else {}),
                )
                continue
            if not sub.template:
                continue   # primitive / embedding sub-entries handled
                            # by _walk_model_layers, not as blocks
            sub_count = self.resolve_int(sub.repeat) if sub.repeat else 1
            count = group_repeat * sub_count
            if absent_repeat(count):
                continue
            site = self.resolved.site
            # A second indexed stack in the group (Qwen3-VL's
            # `deepstack_merger_list.X` beside `blocks.X`) declares its own
            # address; it indexes locally.
            own = sub.checkpoint
            sub_path = (*layer_path, sub_index)
            # A call of an earlier template call beside it, or a template
            # call in a call of a group: the called one at the same place.
            if sub.call is not None:
                tied_to, tied_to_path = sub.call, (
                    *layer_path, _named_earlier(layer.layers[:sub_index],
                                                sub.call))
            elif layer.call is not None:
                tied_to, tied_to_path = layer.call, (
                    _named_earlier(self.resolved.spec.model[:layer_path[0]],
                                   layer.call), sub_index)
            else:
                tied_to = tied_to_path = None
            self.out.append(BlockMeta(
                tied_to=tied_to,
                tied_to_path=tied_to_path,
                template_name=sub.template,
                count=count,
                count_raw=(None if sub.repeat is None else str(sub.repeat)),
                outer_repeat_raw=(None if layer.repeat is None
                                  else str(layer.repeat)),
                stack_name=stack,
                aux_kind=layer.aux_kind,
                group_name=layer.name,
                checkpoint=(with_root(own, root) if own is not None
                            else group_checkpoint),
                backbone_layers=(self.tail_index_base(own) if own is not None
                                 else group_backbone_layers),
                layer_indices=None,
                outer_repeat=group_repeat,
                dim_overrides={},
                layer_path=sub_path,
                address=site.address if site is not None else None,
                reads=self.reads_at(sub_path),
                **self.block_io(sub.template),
            ))

    def run(self) -> List[BlockMeta]:
        for index, layer in enumerate(self.resolved.spec.model):
            if layer.template:
                self.template_branch(layer, (index,))
            elif layer.assignments:
                self.assignment_metas(
                    layer, 1,
                    stack_name=layer.name,
                    aux_kind=None,
                    checkpoint=self.stack_address(layer),
                    layer_path=(index,),
                )
            elif layer.group and layer.layers:
                self.group_branch(layer, (index,))
            # Other entries (top-level embedding/lm_head/norm) are not
            # blocks; they're handled by the top-level leaf walker in
            # collect().
        spec = self.resolved.spec
        # A block's short name, told apart from every other template of
        # this model (`distinct_block_labels`).
        stacks: Dict[Optional[str], List[str]] = {}
        layers: Dict[Optional[str], Dict[str, set]] = {}
        for meta in self.out:
            stacks.setdefault(meta.stack_name, []).append(meta.template_name)
            layers.setdefault(meta.stack_name, {}).setdefault(
                meta.template_name, set()).update(
                meta.layer_indices if meta.layer_indices is not None
                else range(int(meta.count)))
        names = distinct_block_labels(stacks.values(), self.resolve,
                                      layers=[layers[s] for s in stacks])
        families = distinct_block_labels(stacks.values(), self.resolve,
                                         wrappers_only=True)
        kinds: Dict[Tuple[Optional[str], str], List[str]] = {}
        for meta in self.out:
            kinds.setdefault((meta.stack_name, families[meta.template_name]),
                             []).append(meta.template_name)
        slots: Dict[Tuple[Optional[str], str], Tuple] = {}
        for (stack, _kind), members in kinds.items():
            if len(set(members)) < 2:
                continue
            for t, found in differing_slots(members, self.resolve).items():
                slots[(stack, t)] = tuple(
                    (slot, name, getattr(self.resolve(name), "description",
                                         None) or "")
                    for slot, name in found)
        return self.distinguish_by_dims([
            replace(group, short_name=names[group.template_name],
                    family_name=families[group.template_name],
                    distinct_slots=slots.get(
                        (group.stack_name, group.template_name), ()),
                    iteration_owner=(
                        iteration_owner(spec, group.layer_path)
                        if group.layer_path else ()))
            for meta in self.out
            for group in self.visibility_groups(meta)])

    def distinguish_by_dims(self, metas: List[BlockMeta]) -> List[BlockMeta]:
        """`metas` with `distinct_dims` set on every invocation that shares
        its stack and `label` with another: the dims whose bound values tell
        them apart, each invocation's own `dims:` value or else the model's.

        A node's identity is its invocation's label and its path, and every
        per-node table — weights, rows, measured joins — is keyed by it. Two
        `per_layer.assign` entries may bind one template in one stack at
        different `dims:` (DeepSeek-V4.1's pure CSA consumers at compress
        ratio 2 and 1); each holds its own layers' weights and computes at
        its own extents, so under one label the two collide and the weight
        walk refuses them. Invocations that share a label and differ in no
        bound dim are left as they are: nothing declared tells them apart,
        and that refusal names the declaration to merge."""
        by_label: Dict[Tuple[Optional[str], str], List[int]] = {}
        for index, meta in enumerate(metas):
            by_label.setdefault((meta.stack_name, meta.label), []).append(index)
        out = list(metas)
        for members in by_label.values():
            if len(members) < 2:
                continue
            dims = sorted({dim for index in members
                           for dim in metas[index].dim_overrides})
            bound = {index: {dim: self.bound_dim(metas[index], dim)
                             for dim in dims} for index in members}
            differing = [dim for dim in dims
                         if len({bound[index][dim] for index in members}) > 1]
            if not differing:
                continue
            for index in members:
                out[index] = replace(metas[index], distinct_dims=tuple(
                    (dim, bound[index][dim]) for dim in differing))
        return out

    def bound_dim(self, meta: BlockMeta, dim: str) -> str:
        """The value `dim` takes inside `meta`'s invocation, as text: its
        own `dims:` binding, else the model's."""
        value = meta.dim_overrides.get(dim, self.dim_values.get(dim))
        if value is None:
            raise ValueError(
                f"{meta.template_name}: an invocation binds {dim!r} in its "
                f"`dims:` and the model binds no default for it")
        resolved = value_of(value)
        try:
            return str(int(resolved))
        except (TypeError, ValueError):
            return str(resolved)

    def layer_at(self, path: Tuple[int, ...]):
        """The model layer a `BlockMeta.layer_path` locates."""
        return layer_at(self.resolved.spec, path)

    def visibility_groups(self, meta: BlockMeta) -> List[BlockMeta]:
        """`meta` split by what its layers see of their stack's `read: all`
        stream: one invocation per visible length m, with its own layers,
        count and `Sar` binding (`BlockMeta.visible`).

        A template reading the prefix of a stream sees a history of m
        elements, and m grows along the stack: the layers one declaration
        covers do not share a shape, so one evaluation cannot stand for
        them. Grouped by m they do, and each group is a variant of the
        stack like any other — the chain walks it in layer order and holds
        it to the carried stream's fixed point wherever it occurs more than
        once. A layer that sees nothing (m = 0) has nothing to read, and a
        template that reads there is refused: that layer is written with a
        variant that does not read the stream. A template that does not
        read it is not split."""
        if not meta.layer_path:
            return [meta]
        layer = self.layer_at(meta.layer_path)
        (name,) = layer.prefix_streams or (None,)
        if name is None:
            return [meta]
        reads = any(name in graph.reads for _graph_name, graph
                    in self.resolved.spec.reachable_graphs(
                        [meta.template_name]))
        if not reads:
            return [meta]
        stream = self.resolved.appended_streams()[name]
        indices = (meta.layer_indices if meta.layer_indices is not None
                   else tuple(range(meta.count)))
        by_length: Dict[int, List[int]] = {}
        for index in indices:
            by_length.setdefault(stream.visible(index), []).append(index)
        if 0 in by_length:
            raise ValueError(
                f"model layer {layer.name!r}: template "
                f"{meta.template_name!r} reads stream {name!r} (`read: all`) "
                f"at layer(s) {by_length[0]}, where no element of it has "
                f"been appended yet; write those layers with a variant that "
                f"does not read it")
        return [
            replace(meta, count=len(group), layer_indices=tuple(group),
                    dim_overrides={**meta.dim_overrides, HISTORY_AXIS: m},
                    dim_overrides_raw={**meta.dim_overrides_raw,
                                       HISTORY_AXIS: str(m)},
                    visible=((name, m),))
            for m, group in sorted(by_length.items())]


def declared_blocks(metas: List[BlockMeta]) -> List[BlockMeta]:
    """`metas` as their declarations authored them: the derived groups of a
    `read: all` stack (`_BlockEnumerator.visibility_groups`) joined back into
    one record per declaration — every layer, the whole count, the authored
    dims — in the order each declaration first appears. For a consumer whose
    unit is the declaration (an editor edits it) rather than the invocation
    (a walk evaluates it). A declaration with one invocation is returned as
    is."""
    joined: Dict[Tuple[Tuple[int, ...], Optional[int]], List[BlockMeta]] = {}
    for meta in metas:
        key = (tuple(meta.layer_path), meta.assignment_index)
        joined.setdefault(key, []).append(meta)
    out: List[BlockMeta] = []
    for group in joined.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        first = group[0]
        out.append(replace(
            first,
            count=sum(meta.count for meta in group),
            layer_indices=tuple(sorted(
                i for meta in group for i in meta.layer_indices)),
            dim_overrides={k: v for k, v in first.dim_overrides.items()
                           if k != HISTORY_AXIS},
            dim_overrides_raw={k: v for k, v in first.dim_overrides_raw.items()
                               if k != HISTORY_AXIS},
            visible=()))
    return out


def enumerate_blocks(resolved) -> List[BlockMeta]:
    """Mirror the evaluator's layer-traversal, producing only metadata.
    The two paths must stay in lock-step on what counts as a block
    invocation — when this disagrees, downstream consumers silently
    skip leaves or double-count them."""
    return _BlockEnumerator(resolved).run()

def compute_dispatch_zone(graph) -> set:
    """Set of node names downstream of any `permute` op and upstream
    of any `combine` op in `graph` (boundaries themselves excluded).

    The MoE expert bank is the canonical case: `permute` writes the
    `[n_expert_copies, M, d]` buffer, the bank's weights lead with the same
    expert axis, and `combine` reduces back. The walk reads the zone
    (`SpecIRNode.in_dispatch_zone`) for the share of that axis a token runs,
    `n_experts_active / n_route_targets`."""
    gnodes = graph.graph
    dispatches = {n for n, gn in gnodes.items() if gn.op == DISPATCH_OP}
    if not dispatches:
        return set()
    combines = {n for n, gn in gnodes.items() if gn.op == COMBINE_OP}
    fwd: Dict[str, List[str]] = {n: [] for n in gnodes}
    for n, gn in gnodes.items():
        for src in (gn.inputs or []):
            # Engine uses `gn.source_names` (split-axis-aware), but for
            # zone reachability the raw input refs are equivalent — node
            # references in `inputs` resolve to producer nodes.
            base = src.split(".")[0] if isinstance(src, str) else src
            if src in fwd:
                fwd[src].append(n)
            elif base in fwd:
                fwd[base].append(n)
    zone: set = set()
    for d in dispatches:
        seen: set = set()
        frontier = list(fwd.get(d, []))
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in combines:
                continue
            zone.add(cur)
            frontier.extend(fwd.get(cur, []))
    return zone


def reads_into_dispatch_zone(src) -> bool:
    """Whether a node whose one input is `src`, standing in `src`'s own
    graph, lies in that graph's dispatch zone: `compute_dispatch_zone`'s
    rule asked of one edge — downstream of the dispatch op itself, or of a
    node the zone already holds (a combine never is one, so nothing past
    it is). For a node a pass grafts onto an edge after the tree is built
    (`inference.materialize_cast`)."""
    return src.op == DISPATCH_OP or src.in_dispatch_zone
def iter_tree(resolved) -> SpecIRTree:
    """Walk the spec ONCE and emit a full SpecIRTree — every spec graph
    node (component invocations, weight leaves, AND non-weight ops like
    add/silu/mul/softmax) appears as a SpecIRNode.

    Output contains one entry per block invocation, in spec.model order, plus
    top-level entries (embed_tokens, model.norm, lm_head, group
    sub-layers). All `<param>` substitution / rename
    propagation / role inheritance / dispatch-zone marking is applied
    upfront — consumers (Evaluator for derived layers, _node_build for UI) just
    iterate the tree and add their own per-node logic.

    Memoized on the ResolvedSpec instance. Mode-agnostic: dim values
    are resolved to int when `dim_values` are int (the production
    case); raw expression strings are preserved on every node
    (`dims_raw` / `dim_raw`) so symbolic-mode consumers can re-resolve
    against a sympy-Symbol dim dict without re-walking the spec.

    Phase A invariant: parent / uid are wired at build time, and
    resolved_inputs are normalized + inputs_obj populated
    by `finalize_ir_tree` before return — consumers receive a fully
    finalized tree.
    """
    if resolved._spec_tree_ir_cache is not None:
        return resolved._spec_tree_ir_cache
    walker = SpecTreeWalker(resolved)
    out = walker.build_ir_tree()
    wire_parents_and_uids(out)
    finalize_ir_tree(out)
    (out.stream_reads, out.stream_read_forms,
     out.stream_writes) = _StreamTable(resolved, out).build()
    resolved._spec_tree_ir_cache = out
    return out


def iteration_owner(spec, path: Tuple[int, ...]) -> Tuple[int, ...]:
    """The model layer whose iterations a block declaration at `path` runs:
    the layer itself when it repeats, else the repeated group around it
    (DeepSeek's `mtp`). An inner stream is declared on that layer and its
    layers are numbered in it."""
    path = tuple(path)
    if (layer_at(spec, path).repeat is None and len(path) > 1
            and layer_at(spec, path[:-1]).repeat is not None):
        return path[:-1]
    return path


def _spans(indices) -> str:
    """`[0, 1, 2, 6, 10]` as `0-2, 6, 10`."""
    out: List[str] = []
    run: List[int] = []
    for i in sorted(set(indices)):
        if run and i != run[-1] + 1:
            out.append(str(run[0]) if len(run) == 1
                       else f"{run[0]}-{run[-1]}")
            run = []
        run.append(i)
    if run:
        out.append(str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return ", ".join(out) or "none"


class StreamReadRefusal(SpecRefusal):
    """A reader's `reads_from` the spec writer fixes. `meta` is the block
    invocation whose declaration it names (None when the declaration
    selects no layer)."""

    def __init__(self, message: str, *, fix: str,
                 meta: Optional[BlockMeta] = None) -> None:
        super().__init__(message, declaration="reads_from", fix=fix)
        self.meta = meta


class _StreamTable:
    """`SpecIRTree.stream_reads` / `stream_writes`: which layer's write of an
    inner stream each reading layer reads, as its declaration's `reads_from`
    says, and the node each layer writes it from. Built once with the tree;
    every refusal names the declaration to fix and the layers that write
    the stream. Nothing here looks for a writer by the stream's name: a
    reader's source is the layer, or the set, its declaration names."""

    def __init__(self, resolved, tree: SpecIRTree):
        self.resolved = resolved
        self.spec = resolved.spec
        self.tree = tree
        self.isolated = resolved.site is not None
        # Every layer by position, with its name path (`a/b`) for messages.
        self.names: Dict[Tuple[int, ...], str] = {}
        self.declared: Dict[Tuple[int, ...], set] = {}

        def walk(layers, path, prefix) -> None:
            for index, layer in enumerate(layers or ()):
                here, name = (*path, index), (*prefix, layer.name)
                self.names[here] = "/".join(name)
                self.declared[here] = set(layer.inner_streams)
                walk(layer.layers, here, name)

        walk(self.spec.model, (), ())
        self.inner = set().union(*self.declared.values())
        self.reads: Dict[Tuple[Tuple[int, ...], int, str], StreamAt] = {}
        self.forms: Dict[Tuple[Tuple[int, ...], int, str], str] = {}
        self.writes: Dict[Tuple[Tuple[int, ...], int, str], int] = {}
        # The block invocation each read is declared by.
        self.readers: Dict[Tuple[Tuple[int, ...], int, str], BlockMeta] = {}
        self.depth: Dict[Tuple[int, ...], int] = {}
        # Every layer one declaration selects, over its block invocations.
        self.decl_layers: Dict[Tuple[Tuple[int, ...], Optional[int]],
                               List[int]] = {}

    def build(self):
        self.check_declarations()
        if not self.inner:
            return {}, {}, {}
        invocations = []
        for meta, roots in self.tree.blocks:
            owner = meta.iteration_owner
            layers = (tuple(meta.layer_indices)
                      if meta.layer_indices is not None
                      else tuple(range(int(meta.count))))
            if layers:
                self.depth[owner] = max(self.depth.get(owner, 0),
                                        max(layers) + 1)
            invocations.append((meta, roots, owner, layers))
        for meta, _roots, _owner, layers in invocations:
            self.decl_layers.setdefault(
                (tuple(meta.layer_path), meta.assignment_index), []).extend(
                    layers)
        for found in self.decl_layers.values():
            found.sort()
        for meta, roots, owner, layers in invocations:
            self.record_writes(meta, roots, owner, layers)
        self.record_seeds()
        for meta, roots, owner, layers in invocations:
            self.record_reads(meta, roots, owner, layers)
        for (stack, layer, name), (source, at) in self.reads.items():
            if (source, at, name) not in self.writes:
                raise StreamReadRefusal(
                    f"layer {layer} of stack {self.names[stack]!r} reads "
                    f"stream {name!r} from layer {at} of stack "
                    f"{self.names[source]!r}, which does not write it",
                    fix=self.writers_fix(source, name),
                    meta=self.readers[(stack, layer, name)])
        return self.reads, self.forms, self.writes

    # -- declarations ------------------------------------------------------

    def declarations(self):
        """`(layer path, assignment index or None, declaration)` for every
        declaration that selects layers."""
        for path in self.names:
            layer = layer_at(self.spec, path)
            if layer.template is not None:
                yield path, None, layer
            for index, assign in enumerate(layer.assignments or ()):
                yield path, index, assign

    def where(self, path, index) -> str:
        return (f"per_layer.assign[{index}] of {self.names[path]!r}"
                if index is not None else f"stack {self.names[path]!r}")

    def check_declarations(self) -> None:
        """A `reads_from` names an inner stream."""
        appended = set(self.spec.stream_names().produced)
        for path, index, decl in self.declarations():
            if not decl.reads_from:
                continue
            where = self.where(path, index)
            for stream in decl.reads_from:
                if stream in appended or stream not in self.inner:
                    raise StreamReadRefusal(
                        f"{where} declares reads_from[{stream!r}], but "
                        f"{stream!r} is not an inner stream",
                        fix=("`reads_from` names the layer an inner stream "
                             "(one a stack declares with neither `carries`, "
                             "`from` nor `read`) is read from; an appended "
                             "stream is read with `read: each` / `read: all` "
                             "/ `read: concat` or named in `inputs:`, and a "
                             "carried stream arrives on the block's first "
                             "input"))

    def declaration(self, meta: BlockMeta):
        layer = layer_at(self.spec, meta.layer_path)
        if meta.assignment_index is not None and layer.assignments:
            return layer.assignments[meta.assignment_index]
        return layer

    # -- writes ------------------------------------------------------------

    def record_writes(self, meta, roots, owner, layers) -> None:
        found: Dict[str, List[int]] = {}

        def publish(stream_out, nodes) -> None:
            by_name = {node.name: node for node in nodes or ()}
            for name, target in stream_out:
                if str(target) in by_name:
                    found.setdefault(name, []).append(by_name[str(target)].uid)

        def visit(nodes) -> None:
            for node in nodes:
                if node.is_component or node.children:
                    visit(node.children)
                    publish(node.stream_out, node.children)

        visit(roots)
        publish(meta.stream_out, roots)
        for name, writes in found.items():
            if name not in self.declared.get(owner, ()):
                continue
            if len(writes) > 1:
                raise StreamReadRefusal(
                    f"template {meta.template_name!r} writes stream {name!r} "
                    f"{len(writes)} times in one layer of stack "
                    f"{self.names[owner]!r}",
                    fix=(f"a layer writes {name!r} once: one component "
                         f"`stream_out`s it from a node"),
                    meta=meta)
            for layer in layers:
                self.writes[(owner, layer, name)] = writes[0]

    def record_seeds(self) -> None:
        """Each inner stream's `seed`: the top-level node of the layer it
        names writes the stream at the stack's entry (`STACK_ENTRY`)."""
        by_path = {tuple(node.layer_path): node
                   for node in self.tree.top_level}
        for owner in self.declared:
            layer = layer_at(self.spec, owner)
            for name, decl in layer.inner_streams.items():
                if decl.seed is None:
                    continue
                (path,) = self.spec.layers_named(owner, decl.seed)
                if path in by_path:
                    self.writes[(owner, STACK_ENTRY, name)] = \
                        by_path[path].uid

    def writers(self, stack, name: str) -> List[int]:
        return sorted(layer for (s, layer, n) in self.writes
                      if s == stack and n == name and layer != STACK_ENTRY)

    def writers_fix(self, stack, name: str) -> str:
        """Which layers of `stack` write `name`, and the config field that
        lists exactly those layers, when the config has one."""
        writers = self.writers(stack, name)
        listed = self.config_lists(writers, self.depth.get(stack, 0))
        return (f"Name a layer that writes {name!r}. In stack "
                f"{self.names[stack]!r} layers {_spans(writers)} write it; "
                f'"i - 1" reads the layer before, and `{{nearest: <set>}}` '
                f"the nearest earlier layer of a set the config lists"
                + (f" — {' or '.join(listed)} lists exactly these layers"
                   if listed else ""))

    def config_lists(self, layers: List[int], depth: int) -> List[str]:
        """The `{nearest: ...}` sets the config spells `layers` with: a list
        of layer numbers equal to them, or a per-layer list whose entries
        equal one value exactly at them. A suggestion for a refusal's fix;
        nothing resolves through it."""
        want = set(layers)
        out: List[str] = []

        def walk(value, path) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, (*path, str(key)))
                return
            ref = "config." + ".".join(path)
            numbers = ([value] if isinstance(value, int)
                       and not isinstance(value, bool) else value)
            if isinstance(numbers, list) and numbers and all(
                    isinstance(v, int) and not isinstance(v, bool)
                    for v in numbers) and set(numbers) == want:
                out.append(f"`{{nearest: {{from: {ref}}}}}`")
                return
            if isinstance(value, list) and len(value) >= depth > 0:
                head = [str(v) for v in value[:depth]]
                out.extend(
                    f"`{{nearest: {{from: {ref}, is: {v}}}}}`"
                    for v in dict.fromkeys(head)
                    if {i for i, x in enumerate(head) if x == v} == want)

        if want:
            walk(self.resolved.config or {}, ())
        return out

    # -- reads -------------------------------------------------------------

    def record_reads(self, meta, roots, owner, layers) -> None:
        read = set(meta.stream_reads)
        pending = list(roots)
        while pending:
            node = pending.pop()
            read.update(node.stream_reads)
            pending.extend(node.children or ())
        read &= self.inner
        declared = self.declaration(meta).reads_from or {}
        if not layers:
            return
        # A source declared for a stream no component of these layers reads
        # would be wrong without anything noticing.
        for name in sorted(set(declared) - read):
            raise StreamReadRefusal(
                f"{self.where(meta.layer_path, meta.assignment_index)} "
                f"declares reads_from[{name!r}], but template "
                f"{meta.template_name!r} reads no stream {name!r}",
                fix=(f"delete reads_from[{name}], or bind a component that "
                     f"`reads: [{name}]`"),
                meta=meta)
        for name in sorted(read):
            value = declared.get(name)
            if value is None:
                if self.isolated:
                    continue
                stack = owner if name in self.declared[owner] else None
                raise StreamReadRefusal(
                    f"layer {layers[0]} of stack {self.names[owner]!r} runs "
                    f"{meta.template_name!r}, which reads stream {name!r}, "
                    f"and its declaration names no layer it reads it from",
                    fix=(f"write `reads_from: {{{name}: <layer>}}` on the "
                         f"declaration that selects these layers — the layer "
                         f"whose write they read. "
                         + (self.writers_fix(stack, name) if stack is not None
                            else f"{name!r} is declared on another stack: "
                                 f"name it, `{{stack: <stack>, layer: "
                                 f"<layer>}}`")),
                    meta=meta)
            sources = self.sources(meta, owner, name, value)
            for layer in layers:
                key = (owner, layer, name)
                self.reads[key] = self.check_source(
                    meta, owner, layer, name, sources[layer])
                self.forms[key] = self.form(value)
                self.readers[key] = meta

    @staticmethod
    def form(value) -> str:
        """A `reads_from` value as a person reads it."""
        if isinstance(value, NearestLayer):
            return f"nearest {value.nearest}"
        if isinstance(value, StreamLayerRef):
            return f"{value.stack} layer {value.layer}"
        return str(value)

    def sources(self, meta, owner, name, value) -> Dict[int, StreamAt]:
        """Each layer of the declaration and the layer its `reads_from`
        names. Every form is a map from reading layer to source: `{map: …}`
        writes it out, and a layer number, `"i - k"`, `{stack, layer}` and
        `{nearest: <set>}` are shorthand for one."""
        layers = self.decl_layers[(tuple(meta.layer_path),
                                   meta.assignment_index)]
        if isinstance(value, LayerMap):
            return self.mapped(meta, owner, name, value, layers)
        if isinstance(value, NearestLayer):
            return {layer: (owner, self.nearest(meta, owner, layer, name,
                                                value.nearest))
                    for layer in layers}
        if isinstance(value, StreamLayerRef):
            at = self.point(meta, owner, name, value)
            return {layer: at for layer in layers}
        distance = relative_distance(value)
        if distance is None:
            at = (owner, self.number(meta, name, value))
            return {layer: at for layer in layers}
        step = self.number(meta, name, distance)
        if step < 1:
            raise StreamReadRefusal(
                f"reads_from[{name!r}] = {value!r} is not a layer before the "
                f"reading one",
                fix='"i - k" reads the layer k before, k >= 1', meta=meta)
        return {layer: (owner, layer - step) for layer in layers}

    def mapped(self, meta, owner, name, value: LayerMap,
               layers: List[int]) -> Dict[int, StreamAt]:
        """`{map: …}`: a source for exactly the layers the declaration
        selects — written per layer, or a config list aligned with them."""
        where = self.where(meta.layer_path, meta.assignment_index)
        if isinstance(value.map, str):
            listed = config_path_value(value.map[len("config."):],
                                       self.resolved.config or {})
            if not isinstance(listed, list) or len(listed) != len(layers) \
                    or not all(isinstance(v, int) and not isinstance(v, bool)
                               for v in listed):
                raise StreamReadRefusal(
                    f"{where}: reads_from[{name!r}] = {{map: {value.map}}} "
                    f"is {listed!r}, not a list of {len(layers)} layer "
                    f"numbers, one per selected layer ({_spans(layers)})",
                    fix=self.writers_fix(owner, name), meta=meta)
            return {layer: (owner, at) for layer, at in zip(
                layers, listed, strict=True)}
        out = {self.number(meta, name, reader): self.point(
                   meta, owner, name, source)
               for reader, source in value.map.items()}
        missing = sorted(set(layers) - set(out))
        extra = sorted(set(out) - set(layers))
        if missing or extra:
            raise StreamReadRefusal(
                f"{where}: reads_from[{name!r}] maps layers {_spans(out)}, "
                f"and the declaration selects layers {_spans(layers)}"
                + (f"; no source for {_spans(missing)}" if missing else "")
                + (f"; {_spans(extra)} are not selected" if extra else ""),
                fix=("map each selected layer to the layer whose write it "
                     "reads, and no other. " + self.writers_fix(owner, name)),
                meta=meta)
        return out

    def point(self, meta, owner, name, value) -> StreamAt:
        """One source layer: a number of the reader's stack, or
        `{stack, layer}`."""
        if not isinstance(value, StreamLayerRef):
            return (owner, self.number(meta, name, value))
        source = self.spec.layer_position(value.stack.split("/"))
        if source is None:
            stacks = sorted(self.names[path] for path, names
                            in self.declared.items() if name in names)
            raise StreamReadRefusal(
                f"reads_from[{name!r}] names stack {value.stack!r}, which "
                f"is no model layer",
                fix=(f"name the stack by its path from the top of "
                     f"`model:`; the stacks that declare {name!r}: "
                     f"{stacks}"),
                meta=meta)
        return (source, self.number(meta, name, value.layer))

    def check_source(self, meta, owner, layer, name, found) -> StreamAt:
        """A source declares the stream, lies in its stack, and — in the
        reader's own stack — comes before the reader."""
        source, at = found
        if name not in self.declared.get(source, ()):
            raise StreamReadRefusal(
                f"layer {layer} of stack {self.names[owner]!r} reads stream "
                f"{name!r} from stack {self.names[source]!r}, which does not "
                f"declare it",
                fix=(f"declare `{name}` under the `streams:` of the stack "
                     f"whose layers write it, or name that stack: "
                     f"`{{stack: <stack>, layer: <layer>}}`"),
                meta=meta)
        depth = self.depth.get(source, 0)
        if at == STACK_ENTRY and (source, at, name) in self.writes:
            return (source, at)
        if source == owner and not 0 <= at < layer:
            raise StreamReadRefusal(
                f"layer {layer} of stack {self.names[owner]!r} reads stream "
                f"{name!r} from layer {at}, which is not a layer before it",
                fix=(f"a layer reads a write of an earlier layer of its "
                     f"stack; what a first layer reads before any write is "
                     f"the stream's `seed`, or it is written with a variant "
                     f"that does not read {name!r}. "
                     + self.writers_fix(owner, name)),
                meta=meta)
        if not 0 <= at < depth:
            raise StreamReadRefusal(
                f"layer {layer} of stack {self.names[owner]!r} reads stream "
                f"{name!r} from layer {at} of stack {self.names[source]!r}, "
                f"which has layers 0..{depth - 1}",
                fix=self.writers_fix(source, name), meta=meta)
        return (source, at)

    def nearest(self, meta, owner, layer, name, members) -> int:
        """The nearest layer of the declared set before `layer`. Every
        member writes the stream — a set that names a layer that does not
        is a wrong set, whichever member a reader lands on — and nothing
        outside the set is a candidate."""
        if name not in self.declared.get(owner, ()):
            raise StreamReadRefusal(
                f"layer {layer} of stack {self.names[owner]!r} reads stream "
                f"{name!r} from the nearest layer of a set, but the stack "
                f"does not declare {name!r}",
                fix=(f"declare `{name}` under the `streams:` of the stack "
                     f"whose layers write it"),
                meta=meta)
        found = sorted(self.resolved._config_list_indices(
            members, self.depth.get(owner, 0), base=members.base or 0))
        silent = [m for m in found if (owner, m, name) not in self.writes]
        if silent:
            raise StreamReadRefusal(
                f"reads_from[{name!r}] = nearest {members}: layers "
                f"{_spans(silent)} of stack {self.names[owner]!r} are in the "
                f"set and do not write {name!r}",
                fix=self.writers_fix(owner, name), meta=meta)
        before = [m for m in found if m < layer]
        if not before:
            raise StreamReadRefusal(
                f"layer {layer} of stack {self.names[owner]!r} reads stream "
                f"{name!r} from the nearest layer of {members} before it, "
                f"and the set has none before it ({_spans(found)})",
                fix=(f"a layer reads a write of an earlier layer of its "
                     f"stack; what a first layer reads before any write is "
                     f"the stream's `seed`, or it is written with a variant "
                     f"that does not read {name!r}. "
                     + self.writers_fix(owner, name)),
                meta=meta)
        return before[-1]

    def number(self, meta, name: str, expr: str) -> int:
        value = eval_dim_expr(str(expr), self.resolved.dim_values)
        if names_of(value) or value != int(value):
            raise StreamReadRefusal(
                f"reads_from[{name!r}]: {expr!r} resolves to {value}, not one "
                f"layer number",
                fix=("a `reads_from` value is a layer number — an integer or "
                     'a dim expression of the model\'s dims — "i - k", or '
                     "`{nearest: <set>}`"),
                meta=meta)
        return int(value)


def wire_parents_and_uids(tree: SpecIRTree) -> None:
    """Pre-normalization pass: every node gets `parent` and `uid`.

    These are pure structural facts (don't depend on `resolved_inputs`
    being canonical) so they run before `finalize_ir_tree`
    in `iter_tree`. That lets every `iter_leaves` / `iter_tree`
    consumer derive `display_path` / `local_path` correctly — without
    this, a leaf's parent is None and `display_path` collapses to the
    bare local name.

    Idempotent: deterministic walk order + monotonic uid counter +
    write-through assignment. Safe to call twice (second call rewrites
    identical values).
    """
    uid_counter = [0]

    def _walk(nodes: List["SpecIRNode"], parent: Optional["SpecIRNode"]) -> None:
        for n in nodes:
            n.parent = parent
            n.uid = uid_counter[0]
            uid_counter[0] += 1
            if n.children:
                _walk(n.children, n)

    _walk(tree.top_level, None)
    for _meta, roots in tree.blocks:
        _walk(roots, None)
    # Seed the tree's mutation-time uid allocator so any future
    # `tree.make_node(...)` produces uids that don't collide with the
    # build-time ones above.
    tree._next_uid = uid_counter[0]


def finalize_ir_tree(tree: SpecIRTree) -> None:
    """One-pass per-scope finalization over `inputs_obj`: the walker
    emits each input ref with `src=None` and the producer's nested
    spec_id (pure parent_prefix + local_name) in `external_sid`. This
    pass looks the key up in the per-scope canonical index and sets
    `src=node` + clears `external_sid` for in-scope hits. External /
    cross-block refs that the local index can't resolve keep
    `src=None` with `external_sid` holding the string for downstream
    wrapper-alias / namespace handling. `SpecIRNode.resolved_inputs`
    is a @property derived from inputs_obj, so this single pass
    populates both views.

    `display_path` and the walker-emitted ref keys are now both
    structural / pure-nested — they live in the same id space, no
    denest step needed.

    Per-scope, not flat: spec_ids are template-local and the
    component-scope suffix makes two blocks' `post_attn_add` nodes
    have different canonical spec_ids while sharing the same local
    name. A flat map would last-wins-collide and cross-link the blocks.

    Also populates `weight_name` on weight-bearing leaves by walking
    each leaf's parent chain through `compose_weight_name`. The walk
    requires `parent` to be wired (done by `wire_parents_and_uids`
    before this pass) so display paths survive fusion (set
    at walker emit time, see `build_ir_node`).

    Idempotent: re-running on an already-finalized tree is a no-op.
    Re-wires `parent` / `uid` defensively so trees mutated by callers
    that bypass the SpecIRTree mutation API still come back consistent.
    """
    def _build_scope_index(
        roots: List[SpecIRNode],
    ) -> Dict[str, SpecIRNode]:
        """Per-scope index keyed by both the full display_path and the
        scope-suffix-stripped form. Compute-only primitives carry a
        `__<scope>` suffix on their display_path so cross-component
        local-name collisions disambiguate, but `_resolve_input_ref`
        emits the bare sibling name (`qk_score`, not
        `qk_score__gqa_attention`) — the suffix-stripped key absorbs
        that asymmetry. Siblings within one scope have unique base
        names by construction, so both keys are unambiguous per scope.
        """
        idx: Dict[str, SpecIRNode] = {}

        def _walk(nodes: List[SpecIRNode]) -> None:
            for n in nodes:
                if n.display_path:
                    idx[n.display_path] = n
                    if n.component_scope:
                        suffix = f"__{n.component_scope}"
                        if n.display_path.endswith(suffix):
                            bare = n.display_path[: -len(suffix)]
                            idx.setdefault(bare, n)
                if n.children:
                    _walk(n.children)

        _walk(roots)
        return idx

    def _finalize_scope(
        roots: List[SpecIRNode],
        scope_idx: Dict[str, SpecIRNode],
    ) -> None:
        for n in roots:
            if n.inputs_obj:
                rebuilt: List[InputRef] = []
                for ref in n.inputs_obj:
                    key = (
                        ref.src.display_path if ref.src is not None
                        else ref.external_sid
                    )
                    resolved = scope_idx.get(key)
                    if resolved is not None:
                        rebuilt.append(InputRef(
                            src=resolved, port=ref.port, external_sid="",
                        ))
                    else:
                        rebuilt.append(InputRef(
                            src=None, port=ref.port, external_sid=key,
                            grafted=ref.grafted,
                        ))
                n.inputs_obj = tuple(rebuilt)
            if n.children:
                _finalize_scope(n.children, scope_idx)

    top_idx = _build_scope_index(tree.top_level)
    _finalize_scope(tree.top_level, top_idx)
    for _meta, roots in tree.blocks:
        block_idx = _build_scope_index(roots)
        _finalize_scope(roots, block_idx)

    # Re-wire parent / uid: idempotent for pristine `iter_tree` output,
    # essential for trees mutated by callers that bypass the tree
    # mutation API and synthesize bare SpecIRNodes.
    wire_parents_and_uids(tree)
    # Populate HF weight names on weight-bearing leaves. Done AFTER
    # parent wiring so the walk has a stable chain to follow.
    populate_weight_names(tree)
    # `tied_weight.to` names a model layer. Resolved to a uid once, here,
    # after uids exist, so every consumer asks the uid and none re-reads
    # the config or hunts for a role.
    resolve_tied_weights(tree)


def resolve_tied_weights(tree: SpecIRTree) -> None:
    """Set `tied_to_uid` on every node whose `tied_to` names a layer in
    the model list, by the scope the schema gives the name
    (`ModelSpec` validation): an earlier top-level layer, else an earlier
    sibling in the tied layer's group. A name is not an address — a
    group's layer may share its name with a model-root layer — so the
    target is found by `layer_path`. A name no scope declares is a spec
    error."""
    by_path = {tuple(n.layer_path): n for n in tree.top_level
               if n.layer_path}

    def target_of(node: SpecIRNode) -> Optional[SpecIRNode]:
        path = tuple(node.layer_path)
        scopes = [((), path[:1])]
        if len(path) > 1:
            scopes.append((path[:-1], path[-1:]))
        for scope, before in scopes:
            for at, candidate in by_path.items():
                if (at[:-1] == scope and at[-1:] < before
                        and candidate.name == node.tied_to):
                    return candidate
        return None

    def _walk(nodes: List[SpecIRNode]):
        for n in nodes:
            yield n
            yield from _walk(n.children or [])

    roots = list(tree.top_level)
    for _meta, block_roots in tree.blocks:
        roots.extend(block_roots)
    tied: List[SpecIRNode] = (_tie_template_calls(tree, _walk)
                              + _tie_group_ops(tree, _walk)
                              + _tie_shared_modules(_walk(roots)))
    for node in _walk(roots):
        if node.tied_to is None or node.tied_to_uid is not None:
            continue
        target = target_of(node)
        if target is None:
            raise ValueError(
                f"{node.display_path}: `tied_weight.to: {node.tied_to}` names "
                f"no earlier top-level layer and no earlier sibling in its "
                f"group; the model list declares "
                f"{sorted(n.name for n in by_path.values())}")
        node.tied_to_uid = target.uid
        tied.append(node)
    # A target that is itself tied at this binding (its conditional tie is
    # on) does not own the weight; the reader reads its owner's.
    owner_of = {node.uid: node.tied_to_uid for node in tied}
    for node in tied:
        seen = {node.uid}
        while node.tied_to_uid in owner_of:
            if node.tied_to_uid in seen:
                raise ValueError(
                    f"{node.display_path}: `tied_weight` chain loops")
            seen.add(node.tied_to_uid)
            node.tied_to_uid = owner_of[node.tied_to_uid]


def _tie_shared_modules(nodes) -> List[SpecIRNode]:
    """`module:` says where a node's weight lives in the checkpoint, so
    weight-holding siblings at one address are ONE stored tensor: the
    parameter stored there (an identity source), which the others run and
    read as a tied head reads its table (absorbed MLA's per-head GEMMs at
    `kv_b_proj`). An address with no such parameter, or two, is refused."""
    tied: List[SpecIRNode] = []
    for parent in nodes:
        at: Dict[str, List[SpecIRNode]] = {}
        for child in parent.children or ():
            if child.module_name is not None and _holds_weight(child):
                at.setdefault(child.module_name, []).append(child)
        for module, sharing in at.items():
            if len(sharing) == 1:
                continue
            held = [n for n in sharing if is_identity_source(n.op)]
            if len(held) != 1:
                raise ValueError(
                    f"{parent.display_path}: {[n.name for n in sharing]} store "
                    f"at module {module!r}, one tensor, which {len(held)} "
                    f"stored parameters there declare (one must)")
            (owner,) = held
            for node in sharing:
                if node is not owner and node.tied_to_uid is None:
                    node.tied_to, node.tied_to_uid = owner.name, owner.uid
                    tied.append(node)
    return tied


def _holds_weight(node: SpecIRNode) -> bool:
    """Whether a leaf's op may store a weight: a registered op that
    declares one (`OpSpec.has_weight`), or an op the registry does not know
    (a spec's own op). A component holds none of its own."""
    spec = OP_REGISTRY.get(node.op or "")
    return not node.children and (spec is None or spec.has_weight())


def _named_earlier(layers: Sequence[Any], name: str) -> int:
    """The index of the last of `layers` called `name` (the schema has
    checked one is)."""
    return max(index for index, layer in enumerate(layers)
               if layer.name == name)


def _tie_group_ops(tree: SpecIRTree, walk) -> List[SpecIRNode]:
    """A call of a group (`ModelLayer.call`): each weight-holding leaf of
    its op layers reads the leaf at the same place in the called group —
    the loader wrote the call out as that group, so the places agree (its
    template calls are tied as calls, `_tie_template_calls`)."""
    def ops(at: Tuple[int, ...]) -> Dict[Tuple[int, ...], List[SpecIRNode]]:
        out: Dict[Tuple[int, ...], List[SpecIRNode]] = {}
        for node in tree.top_level:
            where = tuple(node.layer_path or ())
            if where[:len(at)] == at and len(where) > len(at):
                out.setdefault(where[len(at):], []).append(node)
        return out
    tied: List[SpecIRNode] = []
    for path, target in tree.group_ties.items():
        theirs = ops(target)
        for place, nodes in ops(path).items():
            for node, owner in zip(walk(nodes), walk(theirs[place]),
                                   strict=True):
                if _holds_weight(owner):
                    node.tied_to, node.tied_to_uid = owner.name, owner.uid
                    tied.append(node)
    return tied


def _tie_template_calls(tree: SpecIRTree, walk) -> List[SpecIRNode]:
    """A call of a template call (`BlockMeta.tied_to`, from
    `ModelLayer.call`): each weight-holding leaf reads the leaf at the same
    place in the called one — the same template, so the same nodes in the
    same order."""
    by_path = {(meta.layer_path, meta.assignment_index): roots
               for meta, roots in tree.blocks}
    tied: List[SpecIRNode] = []
    for meta, roots in tree.blocks:
        if meta.tied_to is None:
            continue
        theirs = by_path[(meta.tied_to_path, meta.assignment_index)]
        for node, target in zip(walk(roots), walk(theirs), strict=True):
            if _holds_weight(target):
                node.tied_to, node.tied_to_uid = meta.tied_to, target.uid
                tied.append(node)
    return tied


def populate_weight_names(tree: SpecIRTree) -> None:
    """For every leaf, walk the parent chain top-down and accumulate
    `weight_name` via `compose_weight_name`. The leaf's value is
    stored on `ir.weight_name`; intermediate-node values are not
    persisted (only `compose_tensor_name` consumes the leaf form).

    Component invocations and intermediate non-leaf nodes are skipped
    (no on-disk weight). `compose_tensor_name` decides whether the
    leaf actually has a `.weight` / `.bias` suffix; this pass just
    establishes the base name so its op-suffix logic stays trivial.
    """
    def _walk(nodes: List[SpecIRNode]) -> None:
        for n in nodes:
            chain: List[SpecIRNode] = []
            cur: Optional[SpecIRNode] = n
            while cur is not None:
                chain.append(cur)
                cur = cur.parent
            chain.reverse()
            wn = ""
            for link in chain:
                # The checkpoint's path is built from the segments the
                # checkpoint has modules for, under the names IT uses. A node
                # contributing nothing is skipped; the leaves below it hang
                # off whatever is above.
                if link.module_name is None:
                    continue
                wn = compose_weight_name(wn, link.module_name)
            n.weight_name = wn
            if n.children:
                _walk(n.children)

    _walk(tree.top_level)
    for _meta, roots in tree.blocks:
        _walk(roots)


# ---------------------------------------------------------------------------
# Internal walker
# ---------------------------------------------------------------------------
def is_pure_delegation(node) -> bool:
    """Whether this node does nothing but invoke a component.

    "Nothing but" is read off the schema, not off a list kept here: a
    `GraphNode` field marked `BINDING_ONLY` says which component to call, on
    what edge, with what params and under what checkpoint names. Everything
    else says something about THIS node — `repeat` multiplies it, `role`
    reclassifies its subtree, `dims` rebinds its extents — and a node that
    says something cannot be inlined away.

    The failure this prevents is silent. A one-node template whose node
    carried `repeat: n_experts` and `role: ffn_moe` would inline, and the
    expert count and the role would both simply not happen.
    """
    fields = type(node).model_fields
    for name, value in node.__dict__.items():
        if (fields[name].json_schema_extra or {}).get("binding_only"):
            continue
        if value != fields[name].default:
            return False
    # `module` is marked binding-only because an inlined node contributes no
    # segment either way — its name never reaches a path. That holds for the
    # two answers that claim nothing: omitted (the name is a spec artefact)
    # and `null` (there is no module here). It does NOT hold for a node that
    # names a segment: inlining would erase a module the checkpoint has.
    return not isinstance(node.module, str)


def module_name_of(node, name: str) -> Optional[str]:
    """The checkpoint's module at a graph node's position, or None for none.

    Three author-facing states collapse to two here, once:

        module omitted   -> the node's own name (the common case, ~97%)
        module: null     -> None, no module at this position
        module: "x"      -> "x"
    """
    return name if node.module is OWN_NAME else node.module


class SpecTreeWalker:
    def __init__(self, resolved):
        self.resolved = resolved
        self.spec = resolved.spec
        self.dim_values: Dict[str, Any] = dict(resolved.dim_values)
        # Components AND inline blocks are both reachable as `op:` refs from
        # graph nodes — same as `_node_build.NodeBuilder`.
        self.components = {
            **(self.spec.components or {}),
            **(self.spec.blocks or {}),
        }
        self.comp_roles = self.spec.component_roles or {}
        self.custom_ops = self.spec.custom_ops or {}
        # Active dim context — top of stack is the dim dict in scope at
        # the current recursion frame. Pushed on component entry when the
        # call site supplies dim overrides.
        self._dims_stack: List[Dict[str, Any]] = [dict(self.dim_values)]
        # Per-graph dispatch-zone stack. When walking a BlockGraph, we
        # compute the set of node names that lie between any `permute`
        # and `combine` op (the MoE expert bank's interior) and push it
        # here. `_emit_node` checks the top frame to know whether the
        # current node is in an active region — used to derive the
        # active-multiplier when a `repeat:` declaration is present
        # without an explicit `active:`.
        self._dispatch_zone_stack: List[set] = []
        # Scope chain for resolving raw input refs to canonical spec_ids.
        # Each frame describes ONE BlockGraph scope:
        #   sibling_names — names in this graph's `graph:` (potential
        #     sibling refs, also for parsing `<sibling>.<port>`)
        #   prefix — id prefix that fully qualifies a sibling name
        #     (e.g. `layers.X.hc_attn_pre.`)
        #   aliases — `{wrapper_input_alias: caller_raw_ref}` populated
        #     at compound entry from `zip(sub_graph.inputs, node.inputs)`.
        #     An alias ref resolves by walking up to the caller's frame
        #     and recursively resolving `caller_raw_ref` there.
        # Frames pushed at compound entry, popped at exit. Top of stack
        # = innermost (current) scope.
        self._scope_stack: List[Tuple[set, str, Dict[str, str]]] = []
        # Per scope, the steps switched off by `when:` → the one input each
        # hands to its readers in its place.
        self._bypassed_stack: List[Dict[str, str]] = []

    # -- IR tree (full spec walk, all nodes) --------------------------------

    def build_ir_tree(self) -> SpecIRTree:
        """Walk spec → SpecIRTree (top_level + per-block-meta IR roots).

        Reuses every helper that the leaf walker uses (dim resolution,
        id, rename, params_stack, dispatch zone) so the
        IR tree shape is consistent with iter_leaves' canonical names.
        """
        streams, stack_reads = stream_spine(self.resolved)
        tree = SpecIRTree(
            streams=streams, stack_reads=stack_reads,
            spine=model_spine(self.resolved),
            # A call of a group: its path and the called group's.
            group_ties={
                (index,): (_named_earlier(self.spec.model[:index],
                                          layer.call),)
                for index, layer in enumerate(self.spec.model)
                if layer.group and layer.call is not None},
        )

        # Top-level: walk spec.model for embed/lm_head/norm + group sub-layers.
        # Each non-template entry produces one SpecIRNode at the top_level
        # list. Group entries recurse, with their sub-layers emitted at the
        # top_level list too (with stack_name = group.name) — this matches
        # the structure in `_walk_model_layers` for leaves.
        self._build_top_level_ir(self.spec.model, prefix="", template="",
                                 multiplicity=1, active_multiplicity=1,
                                 role=None, stack_name=None,
                                 out=tree.top_level,
                                 root=self.spec.checkpoint_root,
                                 absent=tree.absent)

        # Block templates: one IR-root list per BlockMeta, post inline.
        for meta in iter_blocks(self.resolved):
            block = self.spec.get_block(meta.template_name)
            if block is None:
                tree.blocks.append((meta, []))
                continue
            parent_dims = self._dims_stack[-1]
            dims = dict(parent_dims)
            dims.update(rederived_values(
                meta.dim_overrides, meta.dim_overrides_raw, parent_dims))
            dims.update(meta.dim_overrides)
            self._dims_stack.append(dims)
            # An isolated component's stage is the block holding its call
            # (`CallSite.block`). That call is where the component's cache
            # is declared, so it is kept — the adapter rule would inline a
            # one-node delegation — and marked for what it is.
            site = self.resolved.site
            at_site = site is not None and meta.template_name == site.block
            try:
                roots: List[SpecIRNode] = []
                self._build_ir_subgraph(
                    block,
                    parent_prefix="",
                    params_stack=[],
                    wrapper_id=None,
                    outer_prefix="",
                    template=meta.label,
                    multiplicity=int(meta.count),
                    active_multiplicity=int(meta.count),
                    role=None,
                    component_path=(),
                    stack_name=meta.stack_name,
                    out=roots,
                    component_name=meta.template_name,
                    at_site=at_site,
                )
                # The assignment's `dims:` binds names for everything the
                # template contains, so it belongs at the OUTSIDE of the
                # subtree — one frame beyond the outermost node, which is
                # where the roots carry it.
                for root in roots:
                    root.block_dims_raw = dict(meta.dim_overrides_raw)
                tree.blocks.append((meta, roots))
            finally:
                self._dims_stack.pop()

        return tree

    def _build_top_level_ir(self, layers, prefix, template, multiplicity,
                            active_multiplicity, role, stack_name, out,
                            root="", path: Tuple[int, ...] = (),
                            absent: Optional[set] = None):
        """Top-level IR emission for spec.model entries (non-template).
        Emits one SpecIRNode per leaf-like entry; recurses into groups.
        `root` is the checkpoint path of the scope the entries sit in;
        `path` is the `layer_path` of the group they are declared in. An
        absent group's path goes into `absent`."""
        for index, layer in enumerate(layers or []):
            here = (*path, index)
            if layer.template or layer.assignments:
                continue   # block-template entries handled via iter_blocks
            if layer.group and layer.layers:
                # `repeat: 0` means the aux head is absent (optional MTP).
                # Coercing 0 to 1 used to keep wrapper primitives (fc / norms)
                # in the leaf list after the inner template's BlockMeta count
                # had already dropped to zero.
                if layer.repeat:
                    rep = self._layer_repeat(layer)
                    if isinstance(rep, int) and absent_repeat(rep):
                        if absent is not None:
                            absent.add(here)
                        continue
                    if not isinstance(rep, int):
                        rep = 1
                else:
                    rep = 1
                # Group sub-entry `name:` is the canonical HF path
                # relative to the model root — same convention as
                # top-level entries outside a group. Prepending
                # `layer.name.` here would double-prefix structural
                # paths (kimi_k25 `vision_tower.…`, MiMo-V2.5
                # `audio_encoder.…`, Qwen3.5 `model.visual.…`) since
                # the sub-entry already encodes the full path.
                # `stack_name=layer.name` is preserved as the aux-
                # bucketing key (group's semantic tag).
                self._build_top_level_ir(
                    layer.layers,
                    prefix=prefix,
                    template=template,
                    multiplicity=multiplicity * rep,
                    active_multiplicity=active_multiplicity * rep,
                    role=role,
                    stack_name=layer.name,
                    out=out,
                    root=scope_checkpoint_root(layer.checkpoint, root),
                    path=here,
                    absent=absent,
                )
                continue
            if not layer.op:
                continue
            node_role = (
                layer.role
                or self.comp_roles.get(layer.op)
                or role
            )
            tied_to = (
                layer.tied_weight.to
                if layer.tied_weight is not None
                and (layer.tied_weight.when is None
                     or self._resolve_config_bool(layer.tied_weight.when))
                else None
            )
            checkpoint_copy_src = (
                layer.tied_weight.checkpoint_copy.src
                if tied_to is not None
                and layer.tied_weight.checkpoint_copy is not None
                else None
            )
            dims_resolved: Dict[str, Any] = {}
            dims_raw: Dict[str, str] = {}
            shape_resolved: Optional[Tuple[Any, ...]] = None
            shape_raw: Optional[Tuple[str, ...]] = None
            try:
                for k, v in (layer.dims or {}).items():
                    dims_raw[k] = str(v)
                    # An extent may be left in the symbols the walk binds (a
                    # media layer's padded length in its input's axes);
                    # `_resolve_dim_in` refuses any other free name.
                    dims_resolved[k] = (self._resolve_dim_in(
                        v, self._dims_stack[-1], []) if v else 0)
                dim_resolved = self._dim_int(layer.dim) if layer.dim else None
                # Explicit n-D shape (parameter_tensor at top level).
                if layer.shape:
                    shape_raw = tuple(str(s) for s in layer.shape)
                    shape_resolved = tuple(
                        self._dim_int(s) for s in layer.shape
                    )
            except ValueError as exc:
                raise ValueError(f"model layer {layer.name!r}: {exc}") from exc
            for k, v in dims_raw.items():
                self.resolved.refuse_literal_rebinding(
                    f"model layer {layer.name!r}", k, v, dims_resolved[k])
            # Same rule as `_node_has_bias`, which a graph node goes
            # through — a model-root `linear` and a graph `linear` cannot
            # disagree about whether their bias is on disk. Two copies of
            # one list is how `conv1d` came to be in neither.
            bias_decl = (OP_REGISTRY[layer.op].bias_decl()
                         if layer.op in OP_REGISTRY else None)
            has_bias = (
                False if bias_decl is None
                else bool(bias_decl.flag({
                    k: self._dim_int(v) for k, v in layer.dims.items()
                    if k == bias_decl.when})))
            # Honor `repeat:` on top-level flat ops: multiple sibling banks
            # of identical structure (MiMo-V2.5's 20 `speech_embeddings`).
            # multiplicity / active_multiplicity multiply by own_rep so
            # materialize sums the right byte total. Whether the checkpoint
            # writes an index per copy is the layer's `checkpoint.index`.
            own_rep: Optional[int] = None
            own_rep_raw: Optional[str] = None
            own_mult = multiplicity
            own_active_mult = active_multiplicity
            if layer.repeat:
                rep = self._layer_repeat(layer)
                if isinstance(rep, int) and rep > 0:
                    own_rep = rep
                    # Carried only where the resolved count was accepted, so the
                    # two forms answer for the same copies. A raw kept beside a
                    # rejected resolution would let a re-resolving consumer
                    # revive a count this walk declined to apply.
                    own_rep_raw = str(layer.repeat)
                    own_mult = multiplicity * rep
                    own_active_mult = active_multiplicity * rep
            # Top-level layers (embed_tokens / lm_head / norms / …)
            # are model-root scope — `template == ""`, so the
            # `aggregation_id` property collapses to bare `display_path`
            # for these. No per-BlockMeta duplication concern.
            # A model-root entry may invoke a component, not just a
            # primitive: `op: hc_head` is one call of a pure function at
            # the root, which `template:` used to have to express because
            # this branch always emitted a leaf. Expressing it as a stack
            # of one cost the call its wiring — `template:` carries no
            # `inputs:` into the IR, so the body's refs to `fn` / `scale`
            # / `base` stayed unresolved and the parameters they name had
            # no reader. `template:` is for layer stacks; a component call
            # goes down the same path a block graph's call goes down.
            is_comp = layer.op in self.components
            declaration = (None if is_comp or not layer.op
                           else self._declaration_for(layer.op))
            if layer.op and not is_comp and declaration is None:
                raise ValueError(
                    f"{layer.name}: op {layer.op!r} is neither a registered "
                    f"primitive, a custom op of this spec, nor a component — a "
                    f"model layer is laid out and rendered by its declaration, "
                    f"and this one has none")
            # A model-level op is the module its name (or declared `path`)
            # names under the scope root. A model-level component call is a
            # pure function at the root, not a module: its body's tensors
            # hang off the root itself.
            address = layer.checkpoint
            ir_node = SpecIRNode(
                name=layer.name,
                op=layer.op,
                is_component=is_comp,
                layer_path=here,
                module_name=(None if is_comp
                             else f"{address.prefix}.{address.path}"
                             if address and address.prefix
                             else address.path if address and address.path
                             else layer.name),
                checkpoint_indexed=bool(address and (address.index
                                                     or address.prefix)),
                checkpoint_index_after=(address.prefix
                                        if address and address.prefix
                                        else None),
                checkpoint_root=("" if address and address.path else root),
                declaration=declaration,
                call_params=tuple(sorted(
                    (str(k), str(v)) for k, v in (layer.params or {}).items())),
                role=node_role,
                template=template,
                stack_name=stack_name,
                component_path=(),
                dims=dims_resolved,
                dims_raw=dims_raw,
                frame=self._dims_stack[-1],
                dim=dim_resolved,
                dim_raw=str(layer.dim) if layer.dim else None,
                shape=shape_resolved,
                shape_raw=shape_raw,
                weight_axes=(
                    tuple(layer.weight_axes) if layer.weight_axes is not None else None
                ),
                expert_weight=layer.expert_weight,
                stored_axes=(
                    tuple(layer.stored_axes) if layer.stored_axes is not None else None
                ),
                stored_split=self._stored_split(
                    layer.stored_split, layer.name, declaration, dims_resolved, []),
                stored_without_graph_reader=layer.stored_without_graph_reader,
                host_offload=layer.host_offload,
                multiplicity=own_mult,
                active_multiplicity=own_active_mult,
                repeat=own_rep,
                repeat_raw=own_rep_raw,
                inputs=tuple(layer.inputs or ()),
                # A component invocation needs its call-site edges as refs,
                # not just as names: `_callee_aliases` zips the callee's
                # declared `inputs:` against these to build the body's
                # frame, and an empty tuple zips to nothing — the body
                # would see no `x` at all. `finalize_ir_tree` fills `src`
                # for the ones naming a model-root sibling; a stack name
                # (`layers`) stays unresolved and arrives as that stack's
                # output. A primitive's `inputs:` are its edges for the same
                # reason: without them a two-input `concat` would read one
                # operand, and its second would never reach the walk.
                inputs_obj=tuple(
                    InputRef(src=None, port=port, external_sid=sid,
                             grafted=grafted)
                    for sid, port, grafted in (
                        self._resolve_input_ref(r)
                        for r in (layer.inputs or ())
                    )
                ),
                children=[],
                in_dispatch_zone=False,
                has_bias=has_bias,
                tied_to=tied_to,
                checkpoint_copy_src=checkpoint_copy_src,
                segment=layer.segment,
                boundary_role=layer.boundary_role,
                # A top-level layer's `axis` is as load-bearing as a graph
                # node's — `hc_expand` is `broadcast axis: -2`, and dropping
                # it here silently appended the hidden-copies axis at the
                # END, transposing the whole residual stream's layout.
                axis=layer.axis,
            )
            if is_comp:
                self._recurse_into_component(
                    ir_node, layer.name, layer, layer.op,
                    parent_prefix=prefix, params_stack=[],
                    template=template, component_path=(),
                    stack_name=stack_name, node_id=f"{prefix}{layer.name}",
                    node_mult=own_mult, node_active_mult=own_active_mult,
                    node_role=node_role)
            out.append(ir_node)

    def _build_ir_subgraph(self, graph, parent_prefix, params_stack,
                           wrapper_id, outer_prefix, template,
                           multiplicity, active_multiplicity, role,
                           component_path, stack_name, out,
                           aliases: Optional[Dict[str, str]] = None,
                           component_name: str = "",
                           at_site: bool = False):
        """Mirror `_build_graph`'s logic + dispatch-zone
        push, but emit a SpecIRNode for every graph entry (leaves AND
        components AND non-weight ops). Component invocations recurse
        into their sub-graph and the children populate the wrapper's
        `.children`.

        `component_name` is the wrapping component's name (the `op:`
        being invoked, or the block template's name at the outermost
        call). It scopes the spec_id of compute-only primitives in
        this graph via `compose_child_spec_id(..., component_scope=)`
        so cross-component name collisions (`softmax`, `q_pe_rope`,
        `out`, …) auto-disambiguate. Must stay in lockstep with the
        engine — same component_name reaches both sides because the
        IR build and `_eval_graph` walk the same template/component
        invocation chain.

        `at_site`: this graph is the block an isolated component's stage
        runs, and its one node is the call at the component's site — kept
        rather than inlined, and built by `build_ir_node(at_site=True)`."""
        # 1. Skip nodes with resolved repeat=0, and steps whose `when:`
        # switch resolves to 0 (a shape-preserving one is read through; any
        # other is absent with its readers).
        skipped: set = set()
        bypassed: Dict[str, str] = {}
        for name, node in graph.graph.items():
            where = f"{component_name}/{name}" if component_name else name
            try:
                repeat = (self._resolve_dim_in(
                    node.copies, self._dims_stack[-1], params_stack)
                    if node.copies else None)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from exc
            if repeat == 0:
                skipped.add(name)
            elif self._switched_off(where, node, params_stack) is False:
                skipped.add(name)
                continue
            through = self._reads_through(graph, name, params_stack)
            if through is not None:
                # A component's caller reads its output through
                # `output_names`, which `_live_output` already resolved
                # through this step. A block template's outputs are read
                # by name, so there a read-through output would name nothing.
                if wrapper_id is None and name in (graph.outputs or ()):
                    raise ValueError(
                        f"{name}: a step its readers read through (a "
                        "switched-off `when:`, or a sole live operand) "
                        "cannot be a block template output")
                bypassed[name] = through
        # A switched-off step that hands nothing on exists only to feed
        # steps switched off with it; a live reader, or a template output,
        # would be left without its operand.
        for name in skipped:
            node = graph.graph[name]
            if node.when is None:
                continue
            where = f"{component_name}/{name}" if component_name else name
            readers = [other for other, reader in graph.graph.items()
                       if other not in skipped and other not in bypassed
                       and any(str(ref) == name
                               or str(ref).startswith(name + ".")
                               for ref in reader.inputs or ())]
            if readers or name in (graph.outputs or ()):
                raise ValueError(
                    f"{where}: switched off by `when: {node.when}`, it hands "
                    f"nothing on (its op does not keep its input's shape), "
                    f"yet {readers or 'the template output'} still reads it. "
                    f"Switch its readers with the same `when:`.")
        # A built step that still reads a skipped sibling has an operand the
        # model does not have, and its op states no value for one: the spec
        # either skips the step with its branch or reads something else.
        for name, node in graph.graph.items():
            if name in skipped or name in bypassed:
                continue
            gone = [str(ref) for ref in node.inputs or ()
                    if self._repeats_zero(graph, str(ref), params_stack)]
            if gone:
                where = f"{component_name}/{name}" if component_name else name
                raise ValueError(
                    f"{where}: reads {gone}, which this binding does not "
                    f"build (`repeat:` resolves to 0), and "
                    f"{self._resolve_op(node.op, params_stack)!r} declares no "
                    f"value for an absent operand. Give the step the same "
                    f"`repeat:` as its branch, or read an operand that "
                    f"exists here.")

        # 2. Alias-adapter inline-flatten.
        #
        # An adapter is a template whose entire body is ONE node delegating to
        # a component, carrying the `params` / `module` / `inner_modules` that
        # specialise it. It has no structure of its own, so it contributes no
        # namespace: the template IS the component it delegates to, and the
        # walk descends into the sub-graph directly rather than wrapping it in
        # a node named `layer`.
        #
        # Derived from the graph rather than declared. This used to require
        # `primary: true`, which meant a flag whose two readings — "fold this
        # node's name away" and "replace this whole template" — differed by
        # which branch happened to see it first. Marking `transformer` primary
        # in the eight-node `mtp_block_with_head` got the second reading and
        # discarded embed_tokens / eh_proj / shared_head.head and three norms:
        # 1,956,140,032 parameters, no error. Being an adapter is a property of
        # the graph — one node, delegating — so nothing has to be asserted and
        # nothing can be asserted wrongly.
        if wrapper_id is None and not at_site:
            live = [n for n in graph.graph
                    if n not in skipped and n not in bypassed]
            if len(live) == 1 and is_pure_delegation(graph.graph[live[0]]):
                only_node = graph.graph[live[0]]
                only_op = self._resolve_op(only_node.op, params_stack)
                if only_op in self.components:
                    sub_graph = self.components[only_op]
                    if only_node.inner_modules:
                        sub_graph = self._apply_inner_modules(
                            sub_graph, only_node.inner_modules)
                    sub_frame = self._build_frame(sub_graph, only_node)
                    self._build_ir_subgraph(
                        sub_graph,
                        parent_prefix=parent_prefix,
                        params_stack=[*params_stack, sub_frame],
                        wrapper_id=None,
                        outer_prefix=outer_prefix,
                        template=template,
                        multiplicity=multiplicity,
                        active_multiplicity=active_multiplicity,
                        role=role,
                        component_path=component_path,
                        stack_name=stack_name,
                        out=out,
                        component_name=only_op,
                    )
                    return

        # 3. Compute dispatch zone for THIS graph (per-graph, not cumulative).
        zone = compute_dispatch_zone(graph)
        self._dispatch_zone_stack.append(zone)
        # Push a scope frame for input-ref resolution: this graph's
        # sibling names + prefix + caller-supplied alias map (for
        # compound entry). At top-level the aliases dict is empty.
        sibling_names = set(graph.graph.keys())
        self._scope_stack.append((sibling_names, parent_prefix, dict(aliases or {})))
        self._bypassed_stack.append(bypassed)
        try:
            # Producers first, whatever order `graph:` was written in: the
            # evaluator walks these children in sequence, and a reader
            # built ahead of its producer finds no shape to read.
            for name in graph.dependency_order():
                if name in skipped or name in bypassed:
                    continue
                node = graph.graph[name]
                ir_node = self.build_ir_node(
                    name, node, parent_prefix, params_stack,
                    wrapper_id, outer_prefix,
                    template, multiplicity, active_multiplicity,
                    role, component_path, stack_name,
                    component_name, at_site=at_site,
                )
                out.append(ir_node)
        finally:
            self._dispatch_zone_stack.pop()
            self._scope_stack.pop()
            self._bypassed_stack.pop()

    def build_ir_node(self, name, node, parent_prefix, params_stack,
                       wrapper_id, outer_prefix,
                       template, multiplicity, active_multiplicity,
                       role, component_path, stack_name,
                       component_name: str = "",
                       at_site: bool = False) -> SpecIRNode:
        """Build a single SpecIRNode (recursing into children for
        component invocations). All dim resolution / role inheritance
        applied here.

        `at_site`: the node is the call an isolated component's stage makes
        at its site (provenance `site`). The call is the stage, not a level
        of it, so its body is scoped as the stage's own — the same ids it
        had when the component itself was the stage.

        `component_name` is the wrapping component's name; used to
        scope compute-only primitives' spec_ids (mirrors engine's
        `_eval_graph` component-scope semantics — both sides must
        agree on the spec_id of every node)."""
        op_name = self._resolve_op(node.op, params_stack)
        is_comp = op_name in self.components

        node_role = self._inherit_role(node, op_name, is_comp, role)

        node_id, scope, is_compute_only = self._node_spec_id(
            name, op_name, wrapper_id, component_name)

        # Dispatch zone membership (this graph's zone — top of stack).
        in_zone = bool(self._dispatch_zone_stack
                       and name in self._dispatch_zone_stack[-1])

        try:
            (node_mult, node_active_mult, own_repeat, own_active,
             own_repeat_raw, own_active_raw) = self._resolve_multiplicities(
                node, name, multiplicity, active_multiplicity, params_stack)
            (dims_resolved, dims_raw, dim_resolved, dim_raw,
             shape_resolved, shape_raw) = self._resolve_node_dims(
                node, params_stack)
        except ValueError as exc:
            where = f"{component_name}/{name}" if component_name else name
            raise ValueError(f"{where}: {exc}") from exc

        cop = self.custom_ops.get(op_name)
        from_custom = cop is not None and cop.out_key is not None
        declaration = None if is_comp else self._declaration_for(op_name)
        if not is_comp and declaration is None:
            raise ValueError(
                f"{name}: op {op_name!r} is neither a registered primitive, a "
                f"custom op of this spec, nor a component — a leaf is laid out "
                f"and rendered by its declaration, and this one has none")
        has_bias = self._node_has_bias(node, op_name, params_stack)
        # Axis. `axes` is the full permutation (transpose / permute); the
        # DimMap reads it to derive their dim-correspondence.
        axis = node.axis
        axes = node.axes

        # Resolve raw input refs to canonical spec_ids using the active
        # scope chain. Wrapper-input aliases walk up to the caller's
        # raw ref; sibling refs gain the current scope's prefix.
        # Walker emits InputRefs with `src=None` and the nested-form
        # sibling lookup key in `external_sid`; `finalize_ir_tree`
        # walks the per-scope index to set src for in-scope producers
        # and leaves src=None / external_sid populated for cross-block
        # refs that need wrapper-alias / namespace resolution downstream.
        raw_inputs = self._live_inputs(node)
        initial_inputs_obj = tuple(
            InputRef(src=None, port=port, external_sid=sid, grafted=grafted)
            for sid, port, grafted in (self._resolve_input_ref(r)
                                       for r in raw_inputs)
        )

        # `aggregation_id` is a @property derived from
        # template + display_path — no stored field to populate. The
        # parent chain wired in `finalize_ir_tree` makes display_path
        # work; until then any aggregation_id query falls back to the
        # node's local name, which is what callers in flight expect.
        ir_node = SpecIRNode(
            name=name,
            op=op_name,
            is_component=is_comp,
            module_name=module_name_of(node, name),
            checkpoint_indexed=node.checkpoint is not None,
            tensor_suffix={f".{declared}": stored for declared, stored
                           in (node.tensor_suffix or {}).items()},
            from_custom_op=from_custom,
            declaration=declaration,
            local_formula=node.formula,
            call_params=tuple(sorted(
                (str(k), str(v)) for k, v in (node.params or {}).items())),
            role=node_role,
            template=template,
            stack_name=stack_name,
            component_path=component_path,
            dims=dims_resolved,
            dims_raw=dims_raw,
            frame=self._dims_stack[-1],
            dim=dim_resolved,
            dim_raw=dim_raw,
            shape=shape_resolved,
            shape_raw=shape_raw,
            weight_axes=(
                tuple(node.weight_axes) if node.weight_axes is not None else None
            ),
            expert_weight=node.expert_weight,
            stored_axes=(
                tuple(node.stored_axes) if node.stored_axes is not None else None
            ),
            stored_split=self._stored_split(
                node.stored_split, name, declaration, dims_resolved, params_stack),
            stored_without_graph_reader=node.stored_without_graph_reader,
            host_offload=node.host_offload,
            multiplicity=node_mult,
            active_multiplicity=node_active_mult,
            repeat=own_repeat,
            active=own_active,
            repeat_raw=own_repeat_raw,
            active_raw=own_active_raw,
            # Which keyword declared the copies, carried onto the IR so a
            # consumer asking what a reader of this node moves asks the
            # declaration rather than guessing from the count.
            chained=node.chain is not None,
            inputs=raw_inputs,
            inputs_obj=initial_inputs_obj,
            children=[],
            in_dispatch_zone=in_zone,
            axis=axis,
            axes=list(axes) if axes else None,
            has_bias=has_bias,
            boundary_role=node.boundary_role,
            # Phase A object-ref companion fields. `component_scope`
            # mirrors the value passed to `compose_child_spec_id` in
            # `_node_spec_id` — store it explicitly so `display_path`
            # derivation doesn't need to parse the `__<scope>` suffix off
            # `spec_id` (the scope string itself can contain `__`).
            component_scope=scope,
            is_compute_only=is_compute_only,
            provenance="site" if at_site else "spec",
        )

        # Component recursion.
        if is_comp:
            self._recurse_into_component(
                ir_node, name, node, op_name,
                parent_prefix=parent_prefix, params_stack=params_stack,
                template=template, component_path=component_path,
                stack_name=stack_name, node_id="" if at_site else node_id,
                node_mult=node_mult, node_active_mult=node_active_mult,
                node_role=node_role,
                inner_modules=node.inner_modules)

        return ir_node


    def _inherit_role(self, node, op_name, is_comp, role):
        """Call-site `role:` > invoked component's declared role > parent's role."""
        # Role inheritance: call-site `role:` override > invoked
        # component's declared role > parent's propagated role. Spec
        # authors must explicitly tag wrapper-vs-inner role differences
        # (e.g. an `ffn_moe` wrapper that `repeat:`s a `ffn_dense`
        # per-expert FFN must set `role: ffn_moe` on the call site so
        # `quant_context.per_role.ffn_moe` reaches the inner leaves).
        # The loader's `_check_role_propagation` static check warns when
        # the pattern appears without a tag.
        node_role = (
            (node.role if node.role else None)
            or (self.comp_roles.get(op_name) if is_comp else None)
            or role
        )
        return node_role

    def _declaration_for(self, op_name):
        """The op declaration a leaf carries: the spec's custom op, else the
        registered primitive — the loader refuses a custom op that collides
        with a primitive's name (`_check_custom_op_collisions`), so the two
        lookups never overlap; None when the name is neither (a component
        or a block)."""
        cop = self.custom_ops.get(op_name)
        if cop is not None:
            return cop
        return OP_REGISTRY.get(op_name)

    def _node_spec_id(self, name, op_name, wrapper_id, component_name):
        """(spec_id, component_scope, is_compute_only) for this node."""
        from backend.arch.spec_id_compose import compose_child_spec_id
        # Spec id. Compute-only primitives get the parent component
        # name appended (`__<component>`) so cross-component name
        # collisions auto-disambiguate. Weight-bearing leaves and
        # component invocations never carry the suffix — the former
        # because their spec_id feeds tensor_name → HF matching, the
        # latter because they're aggregator nodes, not divergence-
        # checked leaves.
        op_reg = OP_REGISTRY.get(op_name)
        is_compute_only = (
            op_reg is not None and not op_reg.has_weight()
        )
        scope = component_name if (is_compute_only and component_name) else None
        node_id = compose_child_spec_id(
            wrapper_id or "", name, component_scope=scope,
        )
        return node_id, scope, is_compute_only

    def _resolve_multiplicities(self, node, name, multiplicity,
                            active_multiplicity, params_stack):
        """(node_mult, node_active_mult, own_repeat, own_active, raw exprs)."""
        # Repeat / active multipliers (cumulative for descendants).
        node_mult = multiplicity
        node_active_mult = active_multiplicity
        own_repeat: Optional[int] = None
        own_active: Optional[int] = None
        # The expression each count came FROM, kept beside the count itself so a
        # consumer with a different dim binding can re-resolve rather than
        # inherit this walk's constant. Set to whatever was actually resolved —
        # including the substituted fallbacks below — so the two forms cannot
        # describe different copies.
        own_repeat_raw: Optional[str] = None
        own_active_raw: Optional[str] = None
        # The count multiplies this node and everything under it whichever
        # keyword declares it; `GraphNode.copies` is the one place that reads
        # the pair, so a chain keeps the multiplier a bank has always had.
        if node.copies:
            try:
                rep = self._resolve_dim_in(node.copies,
                                           self._dims_stack[-1], params_stack)
            except ValueError:
                rep = None
            if isinstance(rep, int) and rep > 0:
                own_repeat = rep
                own_repeat_raw = str(node.copies)
                node_mult = multiplicity * rep
                act_raw: Optional[str] = own_repeat_raw
                if node.active:
                    try:
                        act = self._resolve_dim_in(
                            node.active, self._dims_stack[-1], params_stack)
                    except ValueError:
                        act = rep
                    if not isinstance(act, int) or act <= 0:
                        act = rep
                    else:
                        act_raw = str(node.active)
                else:
                    act = rep
                own_active = act
                own_active_raw = act_raw
                node_active_mult = active_multiplicity * act
        return (node_mult, node_active_mult, own_repeat, own_active,
                own_repeat_raw, own_active_raw)

    def _resolve_node_dims(self, node, params_stack):
        """Resolved + raw dims/dim/shape for this node."""
        # Resolved dims for this node (raw expr also kept).
        dims_resolved: Dict[str, Any] = {}
        dims_raw: Dict[str, str] = {}
        for k, v in (node.dims or {}).items():
            dims_raw[k] = str(v)
            dims_resolved[k] = self._resolve_dim_in(
                v, self._dims_stack[-1], params_stack)
            self.resolved.refuse_literal_rebinding(
                f"{node.op!r} node", k, v, dims_resolved[k])
        dim_resolved = None
        dim_raw = None
        if node.dim:
            dim_raw = str(node.dim)
            dim_resolved = self._resolve_dim_in(
                node.dim, self._dims_stack[-1], params_stack)
        # Explicit n-D shape (`parameter_tensor` and similar source ops).
        shape_resolved: Optional[Tuple[Any, ...]] = None
        shape_raw: Optional[Tuple[str, ...]] = None
        if node.shape:
            shape_raw = tuple(str(s) for s in node.shape)
            shape_resolved = tuple(
                self._resolve_dim_in(s, self._dims_stack[-1], params_stack)
                for s in node.shape)
        return (dims_resolved, dims_raw, dim_resolved, dim_raw,
                shape_resolved, shape_raw)

    def _node_has_bias(self, node, op_name, params_stack) -> bool:
        """Whether this node's op carries a bias on disk."""
        # has_bias: an op that declares a `.bias` tensor whose switch is on at
        # this node (`TensorDecl.flag`: the node's `bias:` dim, or the op's
        # answer where the node leaves it unset). rmsnorm has gain only.
        #
        # `conv1d` belongs here for the same reason `linear` does — it
        # declares a `.bias` TensorDecl gated on the same flag. It was
        # missing while Nemotron kept its conv bias as a sibling
        # `parameter_vector`, and the omission is not inert: the index
        # `Analysis.tensor_index()` builds is per-uid-first-row, so a node
        # holding `.weight` and `.bias` only shows `.weight` and the
        # companion claim is what puts the bias back. Without it the bias
        # is unmatched on one side and absent from `weight_breakdown` on
        # the other, while `Analysis.tensors()` still has it.
        spec = OP_REGISTRY.get(op_name)
        decl = spec.bias_decl() if spec is not None else None
        if decl is None:
            return False
        return self._has_bias(node, decl, params_stack)

    def _recurse_into_component(self, ir_node, name, node, op_name, *,
                          parent_prefix, params_stack, template,
                          component_path, stack_name, node_id, node_mult,
                          node_active_mult, node_role, inner_modules=None):
        """Expand a component invocation's subgraph into ir_node.children.

        `node` is the call site — a `GraphNode` inside a block graph, or a
        `ModelLayer` when the model root invokes a component with `op:`.
        Only the fields both declare are read here (`dims`, `dim`, `inputs`,
        `params`); `inner_modules` is a graph-node field and arrives as an
        argument so this does not have to ask whether the call site has one.
        """
        sub_graph = self.components[op_name]
        if inner_modules:
            sub_graph = self._apply_inner_modules(sub_graph, inner_modules)
        # Read off the SAME graph object the walk recurses into, so the
        # declared output and the emitted children cannot come from two
        # different copies of the component.
        sub_frame = self._build_frame(sub_graph, node)

        # Push dim overrides from this call site.
        new_dims = dict(self._dims_stack[-1])
        overrides = {k: self._resolve_dim_in(v, new_dims, params_stack)
                     for k, v in (node.dims or {}).items()}
        new_dims.update(rederived_values(
            overrides, node.dims or {}, self._dims_stack[-1]))
        new_dims.update(overrides)
        if node.dim:
            new_dims["dim"] = self._resolve_dim_in(node.dim, new_dims,
                                                   params_stack)
        self._dims_stack.append(new_dims)
        ir_node.output_names = tuple(
            self._live_output(sub_graph, str(o), [*params_stack, sub_frame])
            for o in (sub_graph.outputs or ()))
        _copy_recipe(ir_node, sub_graph)
        # Bind sub_graph input aliases to caller-side raw refs so
        # references inside the body (`mixes.in: [norm, fn]`) can
        # resolve `fn` up to the actual block-scope source.
        sub_aliases: Dict[str, str] = {}
        sub_aliases.update(zip(
            list(sub_graph.inputs or []),
            list(self._live_inputs(node)),
            strict=False,
        ))
        try:
            child_path = (*component_path, (name, op_name))
            self._build_ir_subgraph(
                sub_graph,
                parent_prefix=f"{node_id}." if node_id else "",
                params_stack=[*params_stack, sub_frame],
                wrapper_id=node_id,
                outer_prefix=parent_prefix,
                template=template,
                multiplicity=node_mult,
                active_multiplicity=node_active_mult,
                role=node_role,
                component_path=child_path,
                stack_name=stack_name,
                aliases=sub_aliases,
                out=ir_node.children,
                component_name=op_name,
            )
        finally:
            self._dims_stack.pop()

    def _resolve_input_ref(self, ref: str) -> Tuple[str, str, bool]:
        """Resolve a raw input reference (as written in YAML's `in:`)
        against the current scope chain.

        Returns `(spec_id, port, grafted)` — `spec_id` is the producing
        node's id in **nested form** (`<scope_prefix><local_name>`),
        `port` is the secondary-output port name or "" for the main
        output, `grafted` whether step 3 rewrote the ref to its caller's
        name (`InputRef.grafted`).

        `display_path` is now itself pure-nested, so the walker-emitted
        ref key and the producer's `display_path` share an id space:
        `finalize_ir_tree` resolves each ref by direct lookup, no
        denest step. Primary-fold semantics live in `weight_name`
        (HF naming only) and never touch structural ids.

        At the model root (no scope frame — a `ModelLayer.inputs`
        entry, primitive or component call) the ref is a whole layer
        name and is returned as `(ref, "")` with no dot-splitting:
        root names are checkpoint paths with literal dots
        (`mtp.pre_fc_norm_embedding`), never `<node>.<port>`, and the
        evaluator's stage wiring reads `inputs:` the same way.

        Resolution order inside a scope frame, from innermost outward:
          1. **Whole-ref match** — if the full `ref` is a sibling
             name in this scope (including yaml names that contain
             literal dots like `gate.sigmoid`), it's a port-less
             reference; return `(<prefix><ref>, "")`. This MUST be
             tried before the head-tail split, otherwise dotted
             literal names get sliced into a wrong (sid, port) pair.
          2. **Head-tail split** — if `ref`'s `<head>` (up to the
             first dot) is a sibling, return `(<prefix><head>,
             <tail>)`. The tail is the secondary-output port.
          3. **Wrapper input alias** — if `<head>` is bound by the
             enclosing component invocation's `inputs:` list to a
             caller-side raw ref, walk one frame up and re-resolve
             the grafted ref in the caller's scope.
          4. **Fallthrough** — return `(head, tail)` or `(ref, "")`
             as appropriate. Forward refs / cross-block refs hit
             this; consumers fall back to raw `inputs` if needed.
        """
        if not self._scope_stack:
            # Model root: a `ModelLayer.inputs` entry names a layer whole,
            # as the evaluator's stage wiring reads it — root names are
            # checkpoint paths with literal dots, not `<node>.<port>`.
            return (ref, "", False)
        sibling_names, prefix, aliases = self._scope_stack[-1]
        # 1. Whole-ref sibling match (handles literal-dot names).
        if ref in sibling_names:
            spec_id = f"{prefix}{ref}" if prefix else ref
            return (spec_id, "", False)
        head, dot, tail = ref.partition(".")
        # 2. Head-tail sibling match — head is the producer, tail
        # is the port on its secondary output.
        if head in sibling_names:
            spec_id = f"{prefix}{head}" if prefix else head
            return (spec_id, tail, False)
        # 3. Wrapper input alias — graft caller-side raw ref in.
        if head in aliases:
            caller_ref = aliases[head]
            new_ref = f"{caller_ref}.{tail}" if tail else caller_ref
            saved = self._scope_stack.pop()
            try:
                sid, port, _grafted = self._resolve_input_ref(new_ref)
                return (sid, port, True)
            finally:
                self._scope_stack.append(saved)
        # 4. Fallthrough — unresolved.
        return (head, tail, False) if dot else (ref, "", False)

    # -- helpers (mirror engine + _node_build) -------------------------------

    def _resolve_op(self, op: str, params_stack: List[Dict[str, str]]) -> str:
        """Graft `<param>` placeholders against the active params
        frame. Mirrors engine._resolve_op + _node_build._resolve_op."""
        if op.startswith("<") and op.endswith(">"):
            pname = op[1:-1]
            if params_stack and pname in params_stack[-1]:
                return params_stack[-1][pname]
        return op

    def _build_frame(self, sub_graph, calling_node) -> Dict[str, str]:
        """Push a params frame for component invocation: defaults first,
        then caller bindings overlay. Same semantics as engine and
        _node_build."""
        frame: Dict[str, str] = {}
        for pname, pspec in (sub_graph.params or {}).items():
            if pspec.default is not None:
                frame[pname] = pspec.default
        for pname, pval in (calling_node.params or {}).items():
            frame[pname] = str(pval)
        return frame

    def _apply_inner_modules(self, graph, overrides: Dict[str, Optional[str]]):
        """Apply a call site's `inner_modules` to the component it invokes.

        Keyed by the component's OWN node names, and the component is not
        modified — the overrides land on a copy, so two consumers of one
        library component cannot see each other's answer about their own
        checkpoints.

        A key the component does not declare is an error rather than a no-op.
        A silently-ignored override is a statement about a checkpoint with no
        effect, which is the failure this whole field exists to end.
        """
        unknown = set(overrides) - set(graph.graph)
        if unknown:
            raise ValueError(
                f"inner_modules names {sorted(unknown)!r}, which the invoked "
                f"component does not declare (has {sorted(graph.graph)!r})")
        new_graph = {
            name: (gn.model_copy(update={"module": overrides[name]})
                   if name in overrides else gn)
            for name, gn in graph.graph.items()
        }
        return graph.model_copy(update={"graph": new_graph})

    def _stored_split(self, split, name, declaration, dims, params_stack):
        """`(spec, parts)` for a declared `stored_split`, after the checks
        that need the op and the bound dims: the axis is one of the op's
        weight axes, `parts` resolves to at least 2 and divides that axis."""
        if split is None:
            return None
        # A component invocation has no declaration; an op that stores no
        # weight declares an empty `weight_shape`.
        shape = tuple(declaration.weight_shape) if declaration is not None else ()
        if not shape:
            raise ValueError(
                f"{name}: `stored_split` cuts a stored weight, and this "
                f"node's op stores none")
        if split.axis >= len(shape):
            raise ValueError(
                f"{name}: `stored_split.axis` {split.axis} is not one of the "
                f"op's {len(shape)} weight axes {list(shape)}")
        parts = self._resolve_dim_in(split.parts, dims, params_stack)
        try:
            parts = int(parts)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name}: `stored_split.parts` {split.parts!r} does not "
                f"resolve to a count under this model's dims") from None
        if parts < 2:
            raise ValueError(
                f"{name}: `stored_split.parts` resolves to {parts}; a tensor "
                f"the checkpoint stores whole declares no `stored_split`")
        extent = self._resolve_dim_in(shape[split.axis], dims, params_stack)
        try:
            extent = int(extent)
        except (TypeError, ValueError):
            extent = None
        if extent is not None and extent % parts:
            raise ValueError(
                f"{name}: `stored_split.parts` {parts} does not divide weight "
                f"axis {split.axis} ({shape[split.axis]} = {extent}); check "
                f"`parts` against the checkpoint's shard count, or the "
                f"binding of {shape[split.axis]!r}")
        return (split, parts)

    def _resolve_dim(self, expr: Any) -> Any:
        return self._resolve_dim_in(expr, self._dims_stack[-1], [])

    def _layer_repeat(self, layer) -> Any:
        """A model layer's `repeat:`, an error naming the layer when it
        does not resolve."""
        try:
            return self._resolve_dim(layer.repeat)
        except ValueError as exc:
            raise ValueError(f"model layer {layer.name!r}: {exc}") from exc

    def _resolve_dim_in(self, expr: Any, dims: Dict[str, Any],
                        params_stack: List[Dict[str, str]]) -> Any:
        """Resolve a dim expression to int when possible. Substitutes
        `<param>` first, then evaluates against `dims` via
        `eval.dim.eval_dim_expr`. Falls back to `resolved.get_dim`."""
        if expr is None:
            return None
        s = str(expr)
        # Graft <param> references.
        if s.startswith("<") and s.endswith(">"):
            pname = s[1:-1]
            for frame in reversed(params_stack):
                if pname in frame:
                    s = frame[pname]
                    break
        if s in dims:
            v = dims[s]
            try:
                # Whole only when it IS whole: a `constant` symbol (a
                # clamp bound of 7.5) is not truncated.
                return int(v) if int(v) == v else v
            except (TypeError, ValueError):
                return v
        from backend.arch.eval.dim import eval_dim_expr as _eval_dim_expr, names_of
        merged = dict(dims)
        for k, v in self.dim_values.items():
            merged.setdefault(k, v)
        try:
            v = _eval_dim_expr(s, merged)
        except ValueError as exc:
            raise ValueError(
                f"dim expression {s!r} does not resolve: {exc}") from exc
        # A symbol left standing is fine only if the registry knows it (a
        # runtime axis the evaluator binds later). A name registered nowhere
        # is a typo, and it used to ride through as a free symbol and become
        # a 0-shaped leaf.
        known = dims_registry.registered_for(self.resolved.spec.dim_symbols)
        unregistered = sorted(
            name for name in names_of(v) if name not in known)
        if unregistered:
            raise ValueError(
                f"dim expression {s!r} names {unregistered}, which neither "
                f"backend/arch/dims.yaml nor this spec's `dim_symbols:` "
                f"declares")
        # The only symbols a later walk binds are the engine's (batch,
        # sequence, phase). Any other left standing is a parameter no
        # `dims:` gave a value — a component asking for a dim its caller
        # never bound — and it would surface much later as a symbolic
        # weight byte count that cannot become an integer. A `constant`
        # symbol (a clamp bound) is the exception: one the config does not
        # publish is unbound by design — absent, not defaulted — and sizes
        # nothing, so it stays the symbol and the walk records no value.
        kinds = dims_registry.kinds()
        unbound = sorted(
            name for name in names_of(v)
            if kinds.get(name) != "constant"
            and (name in self.resolved.spec.dim_symbols
                 or not dims_registry.base_symbols()[name].engine_supplied))
        if unbound:
            hints = "; ".join(self._binding_hints(unbound, merged))
            raise ValueError(
                f"unresolved dim {', '.join(unbound)} (expression {s!r}): no "
                f"`dims:` binding in the model or at the call site gives it a "
                f"value" + (f". Bind it in the model's `dims:` — {hints}"
                            if hints else ""))
        try:
            return int(v)
        except (TypeError, ValueError):
            return v

    @staticmethod
    def _binding_hints(unbound, bound) -> List[str]:
        """How dims.yaml tells an author to bind each unbound symbol. A
        derived one is unbound because an input is: its inputs are named
        too, since binding those is the repair."""
        from backend.arch.eval.dim import names_of, eval_dim_expr
        rows = dims_registry.base_symbols()
        hints, seen, queue = [], set(), list(unbound)
        while queue:
            name = queue.pop(0)
            row = rows.get(name)
            if name in seen or row is None:
                continue
            seen.add(name)
            if row.binding_hint:
                hints.append(f"`{name}`: {row.binding_hint}")
            if row.when_unbound:
                queue.extend(sorted(
                    str(i) for i in names_of(eval_dim_expr(row.when_unbound, {}))
                    if str(i) not in bound))
        return hints

    def _dim_int(self, expr: Any) -> int:
        """A written dim expression as the integer it resolves to here.
        Nothing, or a symbol that did not bind, is an error with the
        expression in it — it used to become 0, and a 0-shaped leaf was
        then dropped from the tree without a word."""
        return self._dim_int_in(expr, [])

    def _dim_int_in(self, expr, params_stack) -> int:
        v = self._resolve_dim_in(expr, self._dims_stack[-1], params_stack)
        if v is None:
            raise ValueError(f"dim expression {expr!r} resolved to nothing")
        try:
            return int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"dim expression {expr!r} = {v!r} is not an integer in this "
                f"binding") from exc

    def _switched_off(self, name, node, params_stack) -> Optional[bool]:
        """None when `node` has no `when:` or its switch is on; otherwise
        whether the off step hands its first input on (True: a
        shape-preserving step is the identity on that input) or is simply
        absent (False: its readers must be switched off with it).

        Only a weightless primitive can be switched; a `when:` on anything
        else is rejected rather than bypassed."""
        if node.when is None:
            return None
        op_name = self._resolve_op(node.op, params_stack)
        decl = OP_REGISTRY.get(op_name)
        if decl is None or decl.has_weight():
            raise ValueError(
                f"{name}: `when:` needs a weightless primitive, and "
                f"{op_name!r} is not one")
        if self._switch_on(name, node, params_stack):
            return None
        return decl.shape_kind == "preserve"

    def _switch_on(self, name, node, params_stack) -> bool:
        """Whether a `when:` step runs at this binding."""
        try:
            value = self._resolve_dim_in(node.when, self._dims_stack[-1],
                                         params_stack)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        try:
            return int(value) != 0
        except (TypeError, ValueError):
            raise ValueError(
                f"{name}: `when: {node.when}` resolved to {value!r}, not a "
                "0/1 switch") from None

    def _live_output(self, graph, name: str, params_stack) -> str:
        """The step that produces a declared output at this binding: the
        output itself, or what it reads through to (`_reads_through`), as
        its readers inside the graph would."""
        seen = set()
        while name not in seen:
            seen.add(name)
            through = self._reads_through(graph, name, params_stack)
            if through is None:
                return name
            name = through
        raise ValueError(f"{name}: steps read through in a cycle")

    def _reads_through(self, graph, name: str, params_stack) -> Optional[str]:
        """What the step `name` hands its readers instead of being built,
        or None when it is built.

        Two declarations make a step the identity at a binding: a `when:`
        switch that resolves to 0, and an op that declares what an absent
        operand stands for (`OpSpec.absent_operand_identity`) whose other
        operands are siblings resolved `repeat: 0` — `moe_out: add
        [combine, shared_experts]` in a model with no shared expert is
        `combine`, as the forward's `if self.shared_experts:` says. Any other
        op keeps the absent edge.
        """
        node = graph.graph.get(name)
        if node is None:
            return None
        if node.when is not None:
            off = self._switched_off(name, node, params_stack)
            return str(node.inputs[0]) if off else None
        refs = [str(ref) for ref in node.inputs or ()]
        live = [ref for ref in refs
                if not self._repeats_zero(graph, ref, params_stack)]
        if len(live) == len(refs) or len(live) != 1:
            return None
        decl = OP_REGISTRY.get(self._resolve_op(node.op, params_stack))
        if decl is None or decl.absent_operand_identity is None:
            return None
        return live[0]

    def _repeats_zero(self, graph, ref: str, params_stack) -> bool:
        """Whether `ref` names a sibling whose copy count resolved to 0 — by
        its full name first, as the absence marking reads it, then by its
        head."""
        sibling = graph.graph.get(ref) or graph.graph.get(ref.partition(".")[0])
        if sibling is None or not sibling.copies:
            return False
        return self._resolve_dim_in(
            sibling.copies, self._dims_stack[-1], params_stack) == 0

    def _live_inputs(self, node) -> Tuple[str, ...]:
        """`node.inputs` with every switched-off sibling read through."""
        through = self._bypassed_stack[-1] if self._bypassed_stack else {}
        out = []
        for ref in node.inputs or ():
            seen: set = set()
            while ref in through and ref not in seen:
                seen.add(ref)
                ref = through[ref]
            out.append(ref)
        return tuple(out)

    def _resolve_config_bool(self, field_ref: str) -> bool:
        return resolve_config_bool(self.resolved.config, field_ref)

    def _has_bias(self, node, decl, params_stack) -> bool:
        expr = (node.dims or {}).get(decl.when) if decl.when else None
        return bool(decl.flag(
            {} if expr is None
            else {decl.when: self._dim_int_in(expr, params_stack)}))
