"""PhysicalCollective events and grafting them onto the spec tree."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from enum import Enum
from math import prod
from typing import (Any, Dict, Iterator, List, Mapping, Optional, Sequence,
                    Tuple)

from backend.arch.eval.dim import Dim, as_python, expr_of, value_of, varies
from backend.arch.eval.transition import RedistributeSite
from backend.arch.eval.layer import Layer
from backend.arch.eval.program import Program
from backend.arch.eval.values import Cost, Shape
from backend.arch.spec_tree import InputRef, SpecIRNode, SpecIRTree

from .comm_volume import collective_per_rank_bytes


def flow_port(port: str) -> str:
    return port or "out"


@dataclass(frozen=True)
class FlowRef:
    """One graph value feeding a physical collective event: a node's output
    port, or — `node_uid` None — a tensor no node in the graph writes, read
    by name (`InputRef.external_sid`: a model input, an isolated module's
    input)."""

    node_uid: Optional[int]
    port: str = "out"
    external: str = ""

    @classmethod
    def read_by(cls, ref: Any) -> Optional["FlowRef"]:
        """The value input ref `ref` reads; None for an unnamed edge with
        no producer, which names no value."""
        if ref.src is not None:
            return cls(ref.src.uid, flow_port(ref.port))
        if ref.external_sid:
            return cls(None, flow_port(ref.port), ref.external_sid)
        return None


@dataclass(frozen=True, order=True)
class RewriteTarget:
    """One consumer input that must read the event's output."""

    node_uid: int
    input_index: int


def refs_reading(root: Any, source_uid: Optional[int], source_port: str,
                 external: str = "") -> Tuple[RewriteTarget, ...]:
    """Every input ref reading `source_uid.source_port` on `root` — a
    wrapper's own entry — and anywhere inside its body: the inputs a node
    grafted on that edge for the region `root` owns must take over
    (CLAUDE.md 2.8)."""
    found: List[RewriteTarget] = []
    source = FlowRef(source_uid, flow_port(source_port), external)

    def visit(node: Any) -> None:
        for index, ref in enumerate(node.inputs_obj or ()):
            if FlowRef.read_by(ref) == source:
                found.append(RewriteTarget(node.uid, index))
        for child in node.children or ():
            visit(child)

    visit(root)
    return tuple(sorted(set(found)))


def rewrite_refs(node_of: Any, targets: Sequence[RewriteTarget],
                 source: FlowRef, graft: SpecIRNode, what: str) -> None:
    """Repoint each target input at `graft`, the node grafted on the edge
    from `source`; each must still read `source` (`node_of` resolves a
    target's uid; `what` names the graft in a refusal)."""
    for target in targets:
        consumer = node_of(target.node_uid)
        if consumer is None:
            raise ValueError(
                f"{what} cannot resolve rewrite UID {target.node_uid}")
        refs = list(consumer.inputs_obj or ())
        if not 0 <= target.input_index < len(refs):
            raise ValueError(
                f"{what} rewrite {target!r} is outside "
                f"{consumer.display_path}'s {len(refs)} inputs")
        actual = FlowRef.read_by(refs[target.input_index])
        if actual != source:
            raise ValueError(
                f"{what} rewrite {target!r} reads {actual!r}, not semantic "
                f"source {source!r}")
        refs[target.input_index] = InputRef(src=graft, port="")
        consumer.inputs_obj = tuple(refs)


class Placement(str, Enum):
    """Where the synthetic node belongs relative to an existing node."""

    BEFORE = "before"
    AFTER = "after"


class PostTransform(str, Enum):
    """Typed local work between a collective buffer and the next flow."""

    NONE = "none"
    SAMPLED_POSITION_SELECT = "sampled_position_select"
    LOCAL_SLICE = "local_slice"
    REPLICA_SELECT = "replica_select"
    #: Drop a padded cut's padding after it is gathered: keep the leading
    #: logical extent of one axis (vLLM's `logits[..., :org_vocab_size]`).
    PADDING_TRIM = "padding_trim"
    ONLINE_ATTENTION_COMBINE = "online_attention_combine"


class LogicalBindingGap(str, Enum):
    """A declared binding migration not yet represented by ``EdgeState``."""

    DCP_HEAD_FLOW = "dcp_head_flow"


class PayloadRole(str, Enum):
    """Whether payload bytes describe the main flow or an auxiliary buffer."""

    FLOW_BUFFER = "flow_buffer"
    AUXILIARY_STATISTIC = "auxiliary_statistic"
    RUNTIME_STAGE = "runtime_stage"
    PACKED_ATTENTION_PARTIAL = "packed_attention_partial"


@dataclass(frozen=True)
class PhysicalCollective:
    """Lossless physical communication event before tree grafting.

    Rows are an accounting projection of this record.  Tree materialization
    additionally consumes the flow source, every rewrite target, placement,
    attribution and optional event-to-event chain.  Keeping those facts here
    prevents SSA fanout and DCP's multi-node lowering from being reconstructed
    from display anchors after they have already been discarded.
    """

    event_id: str
    site: RedistributeSite
    op: str
    source: Optional[FlowRef]
    source_event: Optional[str]
    rewrites: Tuple[RewriteTarget, ...]
    placement_uid: int
    placement: Placement
    attribution_uid: int
    report_anchor_uid: int
    tree_anchor_event: Optional[str]
    count_uid: int
    count: int
    world: str
    group_size: int
    base_stride: int
    #: Per-rank shapes as the emitter built them: `Dim`s wherever the axis
    #: is tracked (they compare and convert as their integer values).
    flow_in_shape: Tuple[Any, ...]
    collective_in_shape: Tuple[Any, ...]
    collective_out_shape: Tuple[Any, ...]
    flow_out_shape: Tuple[Any, ...]
    payload_shape: Tuple[Any, ...]
    payload_role: PayloadRole
    payload_dtype_bytes: float
    payload_bytes: float
    bytes_per_rank: float
    #: `payload_bytes` as the emitter built it: a `Dim` over the axes the
    #: buffer is made of (its value is `payload_bytes`), so a row can read
    #: the same payload at another batch.
    payload_extent: Any = field(default=None, compare=False, repr=False)
    runtime_work: Any = None
    payload_symbolic_shape: Tuple[str, ...] = ()
    post_transform: PostTransform = PostTransform.NONE
    binding_gap: Optional[LogicalBindingGap] = None
    meta: Tuple[Tuple[str, Any], ...] = field(default=())
    dedup_key: Optional[Tuple[Any, ...]] = field(
        default=None, compare=False, repr=False
    )

    def at_width(self, width: float) -> "PhysicalCollective":
        """This event moving its payload at `width` bytes per element: the
        same buffer, its bytes restated by the one rule an event is built
        by (`payload_fields`)."""
        return replace(self, **payload_fields(
            self.op, self.group_size, self.count, self.payload_shape, width))


def payload_fields(op: str, group_size: int, count: int,
                   payload_shape: Sequence[Any], width: float
                   ) -> Dict[str, Any]:
    """A collective event's bytes from its payload buffer and element
    width: the buffer's bytes as a typed extent and its value, and what
    each rank moves over a step (`collective_per_rank_bytes` x the launches
    a step makes). The one statement of them — when an event is built
    (`materializer._Session._event`) and when a later pass restates its
    width (`PhysicalCollective.at_width`)."""
    extent = payload_bytes(payload_shape, width)
    payload = value_of(extent)
    return {
        "payload_dtype_bytes": float(width),
        "payload_bytes": payload,
        "payload_extent": extent,
        "bytes_per_rank": (collective_per_rank_bytes(op, group_size, payload)
                           * count),
    }


@dataclass(frozen=True)
class MaterializedCommTree:
    """A copied IR tree with one node for every typed physical event."""

    tree: SpecIRTree
    program: Program
    events: Tuple[PhysicalCollective, ...]
    node_uid_by_event: Dict[str, int]
    cost: Layer[Cost]
    shapes: Layer[Shape]
    instances: Layer[int]
    comm_bytes: Layer[float]
    comm_meta: Layer[Dict[str, Any]]

    def with_payload_widths(self, widths: Mapping[int, float]
                            ) -> "MaterializedCommTree":
        """The same graft, with the collectives at the node uids `widths`
        names moving their payload at that element width.

        A collective's width is chosen when it is grafted, from what its
        source writes then; a later pass that changes what an edge holds
        (`inference.materialize_casts`: a MoE dispatch's tokens quantised
        before it, so the exchange after it moves the narrow copy and its
        scales) restates it here, and every layer is projected again from
        the restated events — one payload, read wherever it is."""
        if not widths:
            return self
        events = []
        for event in self.events:
            width = widths.get(self.node_uid_by_event[event.event_id])
            if width is None or width == event.payload_dtype_bytes:
                events.append(event)
                continue
            events.append(event.at_width(width))
        events = tuple(events)
        nodes = {event_id: self.program.by_uid(uid)
                 for event_id, uid in self.node_uid_by_event.items()}
        cost, shapes, instances, comm_bytes, comm_meta = _comm_layers(
            self.program, events, nodes)
        return replace(self, events=events, cost=cost, shapes=shapes,
                       instances=instances, comm_bytes=comm_bytes,
                       comm_meta=comm_meta)


def payload_bytes(local_shape: Sequence[Any], width: float) -> Any:
    """The bytes of one per-rank buffer of elements `width` bytes wide,
    typed like its axes: a `Dim` whose expression is the product of the axis
    expressions when any axis is tracked, its value the integer element
    count times the width."""
    value = prod(value_of(d) for d in local_shape) * width
    if not varies(value):
        value = float(value)
    if not any(isinstance(d, Dim) for d in local_shape):
        return value
    return Dim(prod(expr_of(d) for d in local_shape) * width, value)


def payload_substituted(shape: Sequence[Any], dtype_bytes: float) -> str:
    """A collective payload's substituted line, `(16*1*2048*2)`: each axis's
    whole value and the element width. The comm row's popover reads it at
    the walk's batch, and a row re-read at another batch renders it from that
    batch's axes through this same function."""
    terms = [str(int(dim)) for dim in shape]
    terms.append(f"{dtype_bytes:g}")
    return f"({'*'.join(terms)})"


def _copy_program_tree(analysis) -> Tuple[Any, Dict[int, SpecIRNode]]:
    """Deep-copy the program's IR tree and index the copy by uid.

    UIDs are the only topology key; display paths and aggregation ids are
    reporting facts and are not unique across repeated template invocations.
    The copy must preserve the uid set exactly, alias nothing, and be
    reachable-checked by the caller after grafting.
    """
    source_tree = analysis.program.ir
    copied_tree = copy.deepcopy(source_tree)
    original_nodes = tuple(analysis.program.nodes())
    copied_nodes = tuple(_walk_tree_nodes(copied_tree))
    original = {node.uid: node for node in original_nodes}
    copied = {node.uid: node for node in copied_nodes}
    if len(original) != len(original_nodes) or len(copied) != len(copied_nodes):
        raise ValueError("communication tree contains duplicate program UIDs")
    if set(copied) != set(original):
        raise ValueError(
            "copied communication tree does not preserve the program UID set"
        )
    if any(copied[uid] is original[uid] for uid in copied):
        raise ValueError("communication tree copy aliases a program node")
    return copied_tree, copied


class _CommGraft:
    """Graft one event at a time into the copied tree.

    The three bookkeeping maps (event → node, event → semantic source,
    placement → current tail) are what the per-event steps share; keeping
    them on the object is what lets `apply` read as one line per step.
    """

    def __init__(self, copied_tree, copied):
        self.tree = copied_tree
        self.copied = copied
        self.nodes_by_event: Dict[str, SpecIRNode] = {}
        self.semantic_sources: Dict[str, FlowRef] = {}
        self.after_tail: Dict[int, SpecIRNode] = {}

    # -- one event ---------------------------------------------------------

    def apply(self, event, seq: int) -> SpecIRNode:
        placement, attribution, report_anchor = self._anchors(event)
        source, source_port, semantic_source = self._source(event)
        dims = self._comm_dims(event, report_anchor)
        structural_anchor = (
            self.nodes_by_event.get(event.tree_anchor_event)
            if event.tree_anchor_event is not None else placement
        )
        if structural_anchor is None:
            raise ValueError(
                f"event {event.event_id} cannot resolve its structural anchor"
            )
        comm_name = _collision_free_comm_name(
            self.tree, structural_anchor,
            self._base_name(event, report_anchor, placement, seq),
        )
        comm_node = self.tree.make_node(
            name=comm_name,
            op=event.op,
            is_component=False,
            from_custom_op=False,
            role=attribution.role,
            template=attribution.template,
            component_path=attribution.component_path,
            dims=dims,
            inputs=(),
            inputs_obj=(InputRef(src=source, port=source_port),),
            children=[],
            provenance="comm",
        )
        self._insert(event, comm_node, placement, structural_anchor)
        self._apply_rewrites(event, comm_node, semantic_source)
        self.nodes_by_event[event.event_id] = comm_node
        self.semantic_sources[event.event_id] = semantic_source
        return comm_node

    def _anchors(self, event):
        placement = self.copied.get(event.placement_uid)
        attribution = self.copied.get(event.attribution_uid)
        report_anchor = self.copied.get(event.report_anchor_uid)
        if placement is None or attribution is None or report_anchor is None:
            raise ValueError(
                f"event {event.event_id} cannot resolve placement, "
                "attribution or report anchor in the copied tree"
            )
        return placement, attribution, report_anchor

    def _source(self, event):
        """(source node, port, semantic FlowRef) — either a real program
        node or an earlier event's comm node."""
        if event.source is not None:
            source = self.copied.get(event.source.node_uid)
            if source is None:
                raise ValueError(
                    f"event {event.event_id} cannot resolve source UID "
                    f"{event.source.node_uid} in the copied tree"
                )
            source_port = (
                "" if event.source.port == "out" else event.source.port
            )
            return source, source_port, event.source
        source = self.nodes_by_event.get(str(event.source_event))
        semantic_source = self.semantic_sources.get(str(event.source_event))
        if source is None or semantic_source is None:
            raise ValueError(
                f"event {event.event_id} cannot resolve dependency "
                f"{event.source_event!r}"
            )
        return source, "", semantic_source

    @staticmethod
    def _comm_dims(event, report_anchor) -> Dict[str, Any]:
        dims: Dict[str, Any] = {
            "__comm_world__": event.group_size,
            "__comm_base_stride__": event.base_stride,
            "__anchor_spec_id__": report_anchor.display_path,
        }
        event_meta = dict(event.meta)
        if "lse_combine" in event_meta:
            dims["__comm_lse_combine__"] = int(
                event_meta["lse_combine"]
            )
        return dims

    @staticmethod
    def _base_name(event, report_anchor, placement, seq: int) -> str:
        is_sampled_gather = dict(event.meta).get(
            "logical_owner:sampled_position"
        ) is True
        boundary_role = report_anchor.boundary_role
        is_vocab_boundary = (
            event.op == "all_reduce"
            and event.site == RedistributeSite.STAGE_CLOSE
            and boundary_role == "input_embedding"
        ) or (
            event.op == "all_gather"
            and event.site == RedistributeSite.MODEL_OUTPUT
            and boundary_role == "output_head"
            and not is_sampled_gather
        )
        if is_vocab_boundary:
            return f"{report_anchor.display_path}.vocab_{event.op}"
        return f"comm{seq}__{event.op}__on__{placement.name}"

    def _insert(self, event, comm_node, placement, structural_anchor) -> None:
        if event.tree_anchor_event is not None:
            tree_anchor = self.nodes_by_event.get(event.tree_anchor_event)
            if tree_anchor is None:
                raise ValueError(
                    f"event {event.event_id} cannot resolve tree anchor "
                    f"{event.tree_anchor_event!r}"
                )
            _insert_after(self.tree, tree_anchor, comm_node)
            if event.placement == Placement.AFTER:
                self.after_tail[event.placement_uid] = comm_node
        elif event.placement == Placement.BEFORE:
            _insert_before(self.tree, placement, comm_node)
        else:
            tail = self.after_tail.get(event.placement_uid, placement)
            _insert_after(self.tree, tail, comm_node)
            self.after_tail[event.placement_uid] = comm_node

    def _apply_rewrites(self, event, comm_node, semantic_source) -> None:
        """Repoint every declared consumer input at the comm node; the
        current ref must still read the semantic source."""
        rewrite_refs(self.copied.get, event.rewrites, semantic_source,
                     comm_node, f"event {event.event_id}")

    # -- final integrity -----------------------------------------------------

    def node_uid_by_event(self) -> Dict[str, int]:
        return {event_id: node.uid
                for event_id, node in self.nodes_by_event.items()}

    def validate(self, copied: Dict[int, SpecIRNode], events) -> None:
        reachable = {node.uid for node in _walk_tree_nodes(self.tree)}
        synthetic = {node.uid for node in self.nodes_by_event.values()}
        if not synthetic.issubset(reachable):
            raise ValueError(
                "materialized communication nodes are not all reachable from "
                "the copied tree roots"
            )
        if len(reachable) != len(copied) + len(events):
            raise ValueError(
                "materialized communication tree lost or duplicated nodes"
            )
        _validate_scope_keys(self.tree)


def _comm_layers(
    program: Program,
    events: Sequence[PhysicalCollective],
    nodes_by_event: Dict[str, SpecIRNode],
) -> Tuple[
    Layer[Cost],
    Layer[Shape],
    Layer[int],
    Layer[float],
    Layer[Dict[str, Any]],
]:
    """Project one event set onto the copied program's physical layers."""
    nodes_with_depth = tuple(_walk_tree_nodes_with_depth(program.ir))
    depths = {node.uid: depth for node, depth in nodes_with_depth}
    nodes_by_uid = {node.uid: node for node, _depth in nodes_with_depth}
    costs: Dict[int, Cost] = {}
    shapes: Dict[int, Shape] = {}
    comm_bytes: Dict[int, float] = {}
    comm_meta: Dict[int, Dict[str, Any]] = {}
    counts: Dict[int, int] = {}
    for event in events:
        node = nodes_by_event[event.event_id]
        if node.uid in costs:
            raise ValueError(
                f"event {event.event_id} has duplicate node uid {node.uid}"
            )
        shape_in = tuple(event.flow_in_shape)
        shape_out = tuple(event.flow_out_shape)
        local_flops = (event.runtime_work.flops if event.runtime_work is not None
                       else dict(event.meta).get("sparse_dcp_local_flops", 0))
        costs[node.uid] = Cost(flops=local_flops)
        shapes[node.uid] = Shape(
            ins=(shape_in,),
            out=shape_out,
        )
        per_instance = collective_per_rank_bytes(
            event.op, event.group_size, event.payload_bytes
        )
        comm_bytes[node.uid] = per_instance
        counts[node.uid] = event.count
        if per_instance * event.count != event.bytes_per_rank:
            raise ValueError(
                f"event {event.event_id} count projection disagrees with "
                f"its physical bytes: {per_instance} * {event.count} != "
                f"{event.bytes_per_rank}"
            )
        symbolic_terms = list(event.payload_symbolic_shape) or [
            f"axis{index}" for index in range(len(event.payload_shape))
        ]
        symbolic_terms.append(
            "dt_lse"
            if event.payload_role == PayloadRole.AUXILIARY_STATISTIC
            else "dt_a"
        )
        tensor_symbolic = f"({'*'.join(symbolic_terms)})"
        report_anchor = nodes_by_uid[event.report_anchor_uid]
        comm_meta[node.uid] = {
            "parent_aggregation_id": report_anchor.aggregation_id,
            "template_name": node.template,
            "depth": depths[node.uid],
            "comm_world": event.group_size,
            "comm_base_stride": event.base_stride,
            "tensor_bytes": event.payload_bytes,
            # The shapes' whole extents (`materializer._int_shape`), laws
            # where they grow with the workload; read at a point they are
            # the integers a row prints, and the payload's substituted line
            # is printed there (`payload_substituted`).
            "tensor_shape": [as_python(value_of(dim)) for dim in event.payload_shape],
            "tensor_symbolic": tensor_symbolic,
            "op": event.op,
            "flow_in_shape": [as_python(value_of(dim)) for dim in shape_in],
            "collective_in_shape": [as_python(value_of(dim))
                                    for dim in event.collective_in_shape],
            "collective_out_shape": [as_python(value_of(dim))
                                     for dim in event.collective_out_shape],
            # The same three shapes as the emitter built them (`Dim`s where
            # the axis is tracked), so a row can read them at another batch.
            "payload_extents": tuple(event.payload_shape),
            "collective_in_extents": tuple(event.collective_in_shape),
            "collective_out_extents": tuple(event.collective_out_shape),
            "flow_out_shape": [as_python(value_of(dim)) for dim in shape_out],
            "payload_role": event.payload_role.value,
            "payload_dtype_bytes": event.payload_dtype_bytes,
            "tensor_bytes_extent": event.payload_extent,
            "post_transform": event.post_transform.value,
            "binding_gap": (
                event.binding_gap.value
                if event.binding_gap is not None else None
            ),
            **{key: value for key, value in event.meta
               if key in ("cut", "source_occurrence", "sink_occurrences")
               or key.startswith("sparse_dcp_")},
            **(dict(event.runtime_work.metadata()) if event.runtime_work is not None else {}),
        }
    # A runtime plan can replace its producer's materialized output traffic
    # (e.g. MLA FP32 split partials). This is a fact on the existing producer,
    # not a synthetic extra producer launch or a second evaluation.
    for event in events:
        work = event.runtime_work
        if work is not None and work.producer_uid is not None:
            if work.producer_write_bytes is None:
                raise ValueError('runtime producer write override has no byte law')
            target = comm_meta.setdefault(work.producer_uid, {})
            if "runtime_write_bytes" in target:
                raise ValueError('duplicate runtime producer output override')
            target["runtime_write_bytes"] = work.producer_write_bytes
            target["runtime_work_basis"] = work.work_basis
    return (
        Layer(program, costs),
        Layer(program, shapes),
        Layer(program, counts),
        Layer(program, comm_bytes),
        Layer(program, comm_meta),
    )


def _walk_tree_nodes(tree: SpecIRTree) -> Iterator[SpecIRNode]:
    def visit(node: SpecIRNode) -> Iterator[SpecIRNode]:
        yield node
        for child in node.children or ():
            yield from visit(child)

    for root in tree.top_level:
        yield from visit(root)
    for _meta, roots in tree.blocks:
        for root in roots:
            yield from visit(root)


def _walk_tree_nodes_with_depth(
    tree: SpecIRTree,
) -> Iterator[Tuple[SpecIRNode, int]]:
    def visit(
        node: SpecIRNode, depth: int
    ) -> Iterator[Tuple[SpecIRNode, int]]:
        yield node, depth
        for child in node.children or ():
            yield from visit(child, depth + 1)

    for root in tree.top_level:
        yield from visit(root, 1)
    for _meta, roots in tree.blocks:
        for root in roots:
            yield from visit(root, 1)


def _siblings_of(
    tree: SpecIRTree, node: SpecIRNode
) -> List[SpecIRNode]:
    if node.parent is not None:
        return node.parent.children
    candidates = [tree.top_level, *(roots for _meta, roots in tree.blocks)]
    for siblings in candidates:
        if any(candidate is node for candidate in siblings):
            return siblings
    raise ValueError(f"node UID {node.uid} is not attached to the copied tree")


def _collision_free_comm_name(
    tree: SpecIRTree, anchor: SpecIRNode, base: str
) -> str:
    """Choose a local name that cannot collide in the anchor's scope."""
    used = {node.name for node in _siblings_of(tree, anchor)}
    if base not in used:
        return base
    suffix = 2
    while f"{base}__{suffix}" in used:
        suffix += 1
    return f"{base}__{suffix}"


def _validate_scope_keys(tree: SpecIRTree) -> None:
    """Every mutation scope must remain indexable without last-wins loss."""
    scopes = [tree.top_level, *(roots for _meta, roots in tree.blocks)]
    for index, roots in enumerate(scopes):
        nodes = []

        def visit(node: SpecIRNode) -> None:
            nodes.append(node)
            for child in node.children or ():
                visit(child)

        for root in roots:
            visit(root)
        for label, keys in (
            ("display_path", [node.display_path for node in nodes]),
            ("aggregation_id", [node.aggregation_id for node in nodes]),
        ):
            seen = set()
            duplicates = set()
            for key in keys:
                if key in seen:
                    duplicates.add(key)
                seen.add(key)
            if duplicates:
                raise ValueError(
                    f"materialized communication scope {index} has duplicate "
                    f"{label} keys {sorted(duplicates)!r}"
                )


def _insert_before(
    tree: SpecIRTree, anchor: SpecIRNode, node: SpecIRNode
) -> None:
    siblings = _siblings_of(tree, anchor)
    index = next(
        (i for i, candidate in enumerate(siblings) if candidate is anchor),
        None,
    )
    if index is None:
        raise ValueError(f"cannot locate insertion anchor UID {anchor.uid}")
    siblings.insert(index, node)
    node.parent = anchor.parent


def _insert_after(
    tree: SpecIRTree, anchor: SpecIRNode, node: SpecIRNode
) -> None:
    siblings = _siblings_of(tree, anchor)
    index = next(
        (i for i, candidate in enumerate(siblings) if candidate is anchor),
        None,
    )
    if index is None:
        raise ValueError(f"cannot locate insertion anchor UID {anchor.uid}")
    siblings.insert(index + 1, node)
    node.parent = anchor.parent


def _coalesce_events(
    events: Sequence[PhysicalCollective],
    node_order: Dict[int, int],
) -> List[PhysicalCollective]:
    """Merge one SSA finalize while retaining every consumer target.

    Every reader of one value that needs one transition lowers the same
    collective, already billed once per producer occurrence they read
    (`_Session._value_count`); the merge unions their rewrite targets. A
    reader that names the producer (a stage's name, a stream) has no ref to
    rewrite and settles the value after its producer; one that holds a ref
    is rewritten. The merged event grafts BEFORE the earliest rewritten
    reader when there is one, else AFTER the producer, and is attributed to
    the producer when its readers settle it in both places. Anything else
    different is a real inconsistency and raises.
    """
    out: List[PhysicalCollective] = []
    by_key: Dict[Tuple[Any, ...], int] = {}
    canonical_event_ids: Dict[str, str] = {}
    for raw_event in events:
        event = replace(
            raw_event,
            source_event=canonical_event_ids.get(
                str(raw_event.source_event), raw_event.source_event
            ),
            tree_anchor_event=canonical_event_ids.get(
                str(raw_event.tree_anchor_event),
                raw_event.tree_anchor_event,
            ),
        )
        key = event.dedup_key
        if key is None or key not in by_key:
            if key is not None:
                by_key[key] = len(out)
            out.append(event)
            continue
        index = by_key[key]
        existing = out[index]
        left, right = (
            replace(
                side,
                event_id="",
                rewrites=(),
                placement=Placement.BEFORE,
                placement_uid=0,
                attribution_uid=0,
                report_anchor_uid=0,
                count_uid=0,
            )
            for side in (existing, event)
        )
        if left != right:
            diffs = sorted(
                f
                for f in left.__dataclass_fields__
                if getattr(left, f) != getattr(right, f)
            )
            raise ValueError(
                f"SSA finalize {key!r} lowered to inconsistent events "
                f"(differing: {diffs})"
            )
        rewrites = tuple(sorted(set((*existing.rewrites, *event.rewrites))))
        if existing.placement != event.placement:
            if existing.source is None:
                raise ValueError(
                    f"SSA finalize {key!r}: a chained event settled both "
                    "before a reader and after its producer"
                )
            producer_uid = existing.source.node_uid
            existing = replace(
                existing,
                count_uid=producer_uid,
                attribution_uid=producer_uid,
                report_anchor_uid=producer_uid,
            )
        if rewrites:
            existing = replace(
                existing,
                placement=Placement.BEFORE,
                placement_uid=min(
                    (target.node_uid for target in rewrites),
                    key=lambda uid: node_order[uid],
                ),
            )
        out[index] = replace(existing, rewrites=rewrites)
        canonical_event_ids[raw_event.event_id] = existing.event_id
    return out
