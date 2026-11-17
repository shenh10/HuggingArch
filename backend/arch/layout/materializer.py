"""Lower Redistribute records to PhysicalCollective events."""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from math import prod
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple
import sympy as sp

from backend.arch.eval.refusal import SpecRefusal, node_context
from backend.arch.eval.dim import expr_of, value_of, whole_count, require
from backend.arch.eval.layer import Layer
from backend.arch.eval.occurrence import Occurrence
from backend.arch.eval.program import Program
from backend.arch.eval.transition import RedistributeSite, Redistribute

from .groups import same
from .materialize_comm import (
    FlowRef,
    flow_port,
    LogicalBindingGap,
    MaterializedCommTree,
    payload_fields,
    PayloadRole,
    PhysicalCollective,
    Placement,
    PostTransform,
    RewriteTarget,
    _coalesce_events,
    _copy_program_tree,
    _CommGraft,
    _comm_layers,
    refs_reading,
)
from .materialize_flow import _output_axes
from .materialize_graph import validate_event_graph
from .materialize_rows import (
    dcp_trio,
    endpoint_rows,
    exchange_row,
    gather_row,
    send_row,
    reduction_row,
    scatter_row,
)
from backend.code_trace.memo import memo_property

_DEFAULT_SOURCE = object()


def _int_shape(shape: Sequence[Any]) -> Tuple[Any, ...]:
    """The whole-number reading of a typed per-rank shape: each extent a
    nonnegative integer, a law where it grows with the workload
    (`whole_count`)."""
    return tuple(
        whole_count(dim, lambda at, dim=dim: ValueError(
            f"physical shape {tuple(at(value_of(d)) for d in shape)!r} has "
            f"a non-integral or negative extent {at(value_of(dim))!r}"))
        for dim in shape)


def _scaled_one_axis(
        before: Sequence[int], after: Sequence[int], group_size: int, *,
        expand: bool) -> bool:
    """Exactly one axis scales by `group_size` (or none, when the group is 1)."""
    if len(before) != len(after):
        return False
    changed = [
        axis for axis, (src, dst) in enumerate(zip(before, after, strict=True))
        if src != dst
    ]
    if len(changed) != (0 if group_size == 1 else 1):
        return False
    for axis, (src, dst) in enumerate(zip(before, after, strict=True)):
        if axis not in changed:
            continue
        left, right = (src, dst) if expand else (dst, src)
        if right != left * group_size:
            return False
    return True


def _collective_geometry_ok(
        op: str, group_size: int,
        collective_in: Sequence[int], collective_out: Sequence[int]) -> bool:
    if op == "runtime_stage":
        return tuple(collective_in) == tuple(collective_out)
    if op == "all_gather":
        return _scaled_one_axis(
            collective_in, collective_out, group_size, expand=True)
    if op == "reduce_scatter":
        return _scaled_one_axis(
            collective_in, collective_out, group_size, expand=False)
    if op == "all_reduce":
        return tuple(collective_out) == tuple(collective_in)
    if op == "all_to_all":
        return (len(collective_in) == len(collective_out)
                and prod(collective_out) == prod(collective_in))
    if op == "send_recv":
        return tuple(collective_out) == tuple(collective_in)
    return False


def _require_collective_geometry(
        event_id: str, op: str, group_size: int,
        collective_in: Sequence[int], collective_out: Sequence[int]) -> None:
    if op not in ("all_gather", "all_reduce", "all_to_all", "reduce_scatter",
                  "send_recv", "runtime_stage"):
        raise ValueError(
            f"event {event_id} has unsupported collective {op!r}")
    if not _collective_geometry_ok(
            op, group_size, collective_in, collective_out):
        raise ValueError(
            f"event {event_id} {op} has invalid collective geometry "
            f"{tuple(collective_in)!r} -> {tuple(collective_out)!r} for "
            f"group size {group_size}")


def _require_payload_flow(
        event_id: str, op: str, payload_role: PayloadRole,
        payload_dims: Sequence[int],
        flow_in: Sequence[int], flow_out: Sequence[int],
        collective_in: Sequence[int], collective_out: Sequence[int],
        post_transform: PostTransform) -> None:
    if payload_role == PayloadRole.RUNTIME_STAGE:
        if (op != "runtime_stage" or tuple(flow_in) != tuple(flow_out)
                or tuple(payload_dims) != (0,) or post_transform != PostTransform.NONE):
            raise ValueError(f"event {event_id} has invalid local runtime stage")
        return
    expected = collective_out if op == "all_gather" else collective_in
    if tuple(payload_dims) != tuple(expected):
        raise ValueError(
            f"event {event_id} {op} collective payload must be "
            f"{tuple(expected)!r}, got {tuple(payload_dims)!r}")
    if not isinstance(payload_role, PayloadRole):
        raise TypeError(
            f"event {event_id} has invalid payload role {payload_role!r}")
    if payload_role == PayloadRole.PACKED_ATTENTION_PARTIAL:
        if (op != "all_to_all"
                or post_transform != PostTransform.ONLINE_ATTENTION_COMBINE
                or len(flow_in) != 4 or len(flow_out) != 4
                or len(collective_in) != 3
                or tuple(collective_out) != tuple(collective_in)
                or tuple(collective_in[:2]) != (flow_in[0] * flow_in[2], flow_in[1])
                or any(flow_in[i] != flow_out[i] for i in (0, 2, 3))
                or flow_out[1] <= 0 or flow_in[1] % flow_out[1]):
            raise ValueError(f"event {event_id} has invalid packed attention combine geometry")
        extra = collective_in[-1] - flow_in[-1]
        require(sp.Or(*(sp.Eq(extra, size) for size in (2, 4, 8))),
                lambda at: ValueError(f"event {event_id} invalid packed LSE slot {at(extra)}"))
        return
    if payload_role == PayloadRole.FLOW_BUFFER:
        if tuple(collective_in) != tuple(flow_in):
            raise ValueError(
                f"event {event_id} main collective input "
                f"{tuple(collective_in)!r} does not equal flow "
                f"{tuple(flow_in)!r}")
        if (post_transform == PostTransform.NONE
                and tuple(collective_out) != tuple(flow_out)):
            raise ValueError(
                f"event {event_id} without a post-transform changes "
                f"{tuple(collective_out)!r} to {tuple(flow_out)!r}")
        return
    if (payload_role != PayloadRole.AUXILIARY_STATISTIC
            or post_transform != PostTransform.NONE
            or tuple(flow_in) != tuple(flow_out)):
        raise ValueError(
            f"event {event_id} auxiliary collective must preserve its "
            f"main flow, got {tuple(flow_in)!r} -> {tuple(flow_out)!r}")


def _require_post_transform(
        event_id: str, op: str, payload_role: PayloadRole,
        post_transform: PostTransform,
        flow_in: Sequence[int], flow_out: Sequence[int],
        collective_out: Sequence[int]) -> None:
    legal = {
        PostTransform.NONE: True,
        PostTransform.SAMPLED_POSITION_SELECT: (
            op == "all_gather" and payload_role == PayloadRole.FLOW_BUFFER),
        PostTransform.LOCAL_SLICE: (
            op in ("all_gather", "all_reduce")
            and payload_role == PayloadRole.FLOW_BUFFER),
        PostTransform.REPLICA_SELECT: (
            op == "all_gather" and payload_role == PayloadRole.FLOW_BUFFER),
        PostTransform.PADDING_TRIM: (
            op == "all_gather" and payload_role == PayloadRole.FLOW_BUFFER),
        PostTransform.ONLINE_ATTENTION_COMBINE: (
            op == "all_to_all" and payload_role == PayloadRole.PACKED_ATTENTION_PARTIAL),
    }
    if not legal.get(post_transform, False):
        raise ValueError(
            f"event {event_id} has invalid {post_transform!r} for {op}")
    if post_transform == PostTransform.SAMPLED_POSITION_SELECT:
        changed = [
            axis for axis, (raw, semantic) in enumerate(
                zip(collective_out, flow_out, strict=True)
            )
            if raw != semantic
        ]
        if tuple(flow_out) != tuple(flow_in) or len(changed) != 1:
            raise ValueError(
                f"event {event_id} sampled-position selection must "
                f"restore exactly its input shape, got {tuple(flow_in)!r} -> "
                f"{tuple(collective_out)!r} -> {tuple(flow_out)!r}")
        return
    if post_transform == PostTransform.PADDING_TRIM:
        changed = [
            axis for axis, (raw, kept) in enumerate(
                zip(collective_out, flow_out, strict=True)
            )
            if raw != kept
        ]
        if len(changed) != 1 or not 0 < flow_out[changed[0]] < collective_out[changed[0]]:
            raise ValueError(
                f"event {event_id} padding trim must shorten exactly one "
                f"axis, got {tuple(collective_out)!r} -> {tuple(flow_out)!r}")
        return
    if post_transform in (
            PostTransform.LOCAL_SLICE, PostTransform.REPLICA_SELECT):
        changed = [
            axis for axis, (raw, local) in enumerate(
                zip(collective_out, flow_out, strict=True)
            )
            if raw != local
        ]
        if len(changed) != 1 or any(
                local <= 0 or raw % local
                for raw, local in zip(collective_out, flow_out, strict=True)):
            raise ValueError(
                f"event {event_id} {post_transform.value} must reduce "
                f"exactly one divisible axis, got {tuple(collective_out)!r} "
                f"-> {tuple(flow_out)!r}")


def _require_binding_gap(
        event_id: str, binding_gap: Optional[LogicalBindingGap],
        post_transform: PostTransform) -> None:
    if binding_gap is None:
        return
    if not isinstance(binding_gap, LogicalBindingGap):
        raise TypeError(
            f"event {event_id} has invalid binding gap {binding_gap!r}")
    if post_transform != PostTransform.NONE and not (
            binding_gap == LogicalBindingGap.DCP_HEAD_FLOW
            and post_transform in (
                PostTransform.LOCAL_SLICE, PostTransform.REPLICA_SELECT,
                PostTransform.ONLINE_ATTENTION_COMBINE)):
        raise ValueError(
            f"event {event_id} cannot combine binding gap {binding_gap!r} "
            f"with post-transform {post_transform!r}")


class _Session:
    def __init__(
        self,
        analysis: Any,
        act_dtype_bytes: Optional[float] = None,
        *,
        stage: Any = None,
        timing_scope: Optional[FrozenSet[str]] = None,
        sm_count: Optional[int] = None,
    ) -> None:
        self.analysis = analysis
        self.sm_count = sm_count
        #: The pipeline stage this rank runs (`stages.StageView`), when the
        #: view is one stage's: its rows count what runs there, and it sends
        #: the hops across its own exit cut and no other. None is the whole
        #: pipeline — every cut.
        self.stage = stage
        self.program = analysis.program
        #: The stacks whose hops are priced: a hop serving no stack of the
        #: scope (an MTP head's read of the embedding when the MTP is not
        #: timed) is not this view's. The backbone when None.
        self.timing_scope = timing_scope
        #: A collective moves the tensor at the width the walk priced it:
        #: the model's own activation width under its precision profile.
        self.act_bytes = (
            analysis.resolved_precision.default_edge_bytes
            if act_dtype_bytes is None else float(act_dtype_bytes))
        if stage is None:
            # Tests may pass a stub instead of a full Analysis. With no stage
            # selection there is nothing to recount, so the declared instance
            # layer is already the input.
            self._counts = getattr(
                analysis, "instances", Layer(self.program, {})
            )
        else:
            self._counts = analysis.recount(
                analysis.instances,
                stage.rows(self.program.included_stack_names(
                    include_aux=True)))
        self._event_seq = 0
        self._node_order = {
            node.uid: index for index, node in enumerate(self.program.nodes())
        }

    # -- shared readings --------------------------------------------------

    def _node(self, uid: int):
        return self.program.by_uid(uid)

    def _count(self, uid: int) -> int:
        count = self._counts.values.get(uid)
        if count is None:
            raise ValueError(
                f"{self._node(uid).display_path}: no instance count on the "
                "analysis — cannot attribute a layer count"
            )
        return int(value_of(count))

    def _partial_provenance(self, record: Any, value: Any):
        """The transition-authored source contract for one pending value."""
        provenance = value.provenance
        if provenance is None:
            raise ValueError(
                f"{self._node(record.node).display_path}: "
                f"{value.reducer!r} Partial has no lowering provenance"
            )
        return provenance

    def _consumer_of(self, uid: int, port: str = "out") -> int:
        return consumer_of(self.program, uid, port)

    def _targets_for_source(
        self, root_uid: int, source_uid: int, source_port: str
    ) -> Tuple[RewriteTarget, ...]:
        """Every matching ref on a wrapper entry and inside its body."""
        root = self._node(root_uid)
        found = refs_reading(root, source_uid, source_port)
        if not found:
            raise ValueError(
                f"{root.display_path}: no input in the component entry reads "
                f"{self._node(source_uid).display_path}.{flow_port(source_port)}"
            )
        return found

    def _reads_by_name(self, record: Redistribute) -> bool:
        """Whether the edge's ref reads its producer by a name, not a node.

        `finalize_ir_tree` leaves `src` unset on a ref it cannot resolve in
        the reader's own scope — a stage's name at the model root, a
        component's formal inside its body. The walk still resolved it,
        through the frame, to the tensor that name was handed
        (`record.producer`).
        """
        refs = tuple(self._node(record.node).inputs_obj or ())
        if record.input_index >= len(refs):
            return False
        ref = refs[record.input_index]
        return ref.src is None and bool(ref.external_sid)

    def _record_topology(
        self, record: Redistribute
    ) -> Tuple[
        RedistributeSite, FlowRef, Tuple[RewriteTarget, ...], int, Placement
    ]:
        site = record.site
        if not isinstance(site, RedistributeSite):
            raise TypeError(
                f"redistribute record at uid {record.node} has no typed site"
            )
        port = flow_port(record.producer_port)
        if site in (RedistributeSite.EDGE, RedistributeSite.INTERNAL_IN):
            if record.producer < 0 or record.input_index < 0:
                raise ValueError(
                    f"{site.value} transition must name a producer and input"
                )
            if site == RedistributeSite.EDGE and self._reads_by_name(record):
                # The reader names the producer's stage, not a node: a
                # model-level op reading a stack's exit by the stack's name
                # (a vision tower into the embedding merge). The ref has no
                # node to rewrite, so the value settles as its stage hands
                # it on — after the producer, the place a stage close
                # settles a stack's carried stream. Every reader of that
                # handoff shares the one event (`dedup_key`).
                return (
                    site,
                    FlowRef(record.producer, port),
                    (),
                    record.producer,
                    Placement.AFTER,
                )
            target = RewriteTarget(record.node, record.input_index)
            return (
                site,
                FlowRef(record.producer, port),
                (target,),
                record.node,
                Placement.BEFORE,
            )
        if site == RedistributeSite.INTERNAL_OUT:
            consumer_uids = consumer_roots(self.program, record.node, port)
            targets = tuple(
                target
                for consumer_uid in consumer_uids
                for target in self._targets_for_source(
                    consumer_uid, record.node, port)
            )
            return (
                site,
                FlowRef(record.node, port),
                targets,
                consumer_uids[0],
                Placement.BEFORE,
            )
        if site in (
            RedistributeSite.STAGE_CLOSE,
            RedistributeSite.STAGE_CROSSING,
            RedistributeSite.CACHE_ENDPOINT,
            RedistributeSite.MODEL_OUTPUT,
        ):
            if record.producer < 0:
                raise ValueError(f"{site.value} transition must name a producer")
            return (
                site,
                FlowRef(record.producer, port),
                (),
                record.node,
                Placement.AFTER,
            )
        raise ValueError(f"unsupported redistribute site {site!r}")

    def _event(
        self,
        *,
        record: Any,
        anchor_uid: int,
        op: str,
        group_size: int,
        count_uid: int,
        world: str,
        count: Optional[int] = None,
        flow_in_shape: Sequence[Any],
        collective_in_shape: Sequence[Any],
        collective_out_shape: Sequence[Any],
        flow_out_shape: Sequence[Any],
        payload_shape: Sequence[Any],
        payload_role: PayloadRole = PayloadRole.FLOW_BUFFER,
        payload_dtype_bytes: Optional[float] = None,
        post_transform: PostTransform = PostTransform.NONE,
        binding_gap: Optional[LogicalBindingGap] = None,
        meta: Tuple[Tuple[str, Any], ...] = (),
        runtime_work: Any = None,
        source: Any = _DEFAULT_SOURCE,
        source_event: Optional[str] = None,
        rewrites: Optional[Tuple[RewriteTarget, ...]] = None,
        placement_uid: Optional[int] = None,
        placement: Optional[Placement] = None,
        attribution_uid: Optional[int] = None,
        tree_anchor_event: Optional[str] = None,
        event_id: Optional[str] = None,
        dedup_key: Optional[Tuple[Any, ...]] = None,
        site: Optional[RedistributeSite] = None,
        family: Any = None,
    ) -> PhysicalCollective:
        record_site, default_source, default_rewrites, default_placement_uid, \
            default_placement = self._record_topology(record)
        if event_id is None:
            self._event_seq += 1
            event_id = f"comm{self._event_seq}"
        resolved_family = self._resolved_family(world, family, event_id)
        size, base_stride = self._family_geometry(resolved_family)
        if size != group_size:
            raise ValueError(
                f"world {world!r} resolves degree {size}, not event group "
                f"size {group_size}")
        if count is None:
            count = self._count(count_uid)
        dtype_bytes = (
            self.act_bytes if payload_dtype_bytes is None
            else float(payload_dtype_bytes)
        )
        flow_in = _int_shape(flow_in_shape)
        collective_in = _int_shape(collective_in_shape)
        collective_out = _int_shape(collective_out_shape)
        flow_out = _int_shape(flow_out_shape)
        payload_dims = _int_shape(payload_shape)
        _require_payload_flow(
            event_id, op, payload_role, payload_dims,
            flow_in, flow_out, collective_in, collective_out, post_transform)
        _require_collective_geometry(
            event_id, op, group_size, collective_in, collective_out)
        if (payload_role == PayloadRole.PACKED_ATTENTION_PARTIAL
                and flow_in[1] != flow_out[1] * group_size):
            raise ValueError(f"event {event_id} combine head cut differs from group size")
        _require_post_transform(
            event_id, op, payload_role, post_transform,
            flow_in, flow_out, collective_out)
        _require_binding_gap(event_id, binding_gap, post_transform)
        # The emitter's typed payload is the one source: the integer shape
        # above and the bytes below are both its value (`payload_fields`).
        payload = payload_fields(op, group_size, count, payload_shape,
                                 dtype_bytes)
        resolved_site = record_site if site is None else site
        if not isinstance(resolved_site, RedistributeSite):
            raise TypeError(f"event {event_id} has invalid site {resolved_site!r}")
        resolved_source = default_source if source is _DEFAULT_SOURCE else source
        if (resolved_source is None) == (source_event is None):
            raise ValueError(
                f"event {event_id} must name exactly one of source or "
                "source_event"
            )
        if resolved_source is not None and not isinstance(resolved_source, FlowRef):
            raise TypeError(
                f"event {event_id} has invalid flow source {resolved_source!r}"
            )
        return PhysicalCollective(
            event_id=event_id,
            site=resolved_site,
            op=op,
            source=resolved_source,
            source_event=source_event,
            rewrites=default_rewrites if rewrites is None else rewrites,
            placement_uid=(default_placement_uid
                           if placement_uid is None else placement_uid),
            placement=default_placement if placement is None else placement,
            attribution_uid=(anchor_uid
                             if attribution_uid is None else attribution_uid),
            report_anchor_uid=anchor_uid,
            tree_anchor_event=tree_anchor_event,
            count_uid=count_uid,
            count=count,
            world=world,
            group_size=group_size,
            base_stride=base_stride,
            flow_in_shape=tuple(flow_in_shape),
            collective_in_shape=tuple(collective_in_shape),
            collective_out_shape=tuple(collective_out_shape),
            flow_out_shape=tuple(flow_out_shape),
            payload_shape=tuple(payload_shape),
            payload_role=payload_role,
            runtime_work=runtime_work,
            **payload,
            payload_symbolic_shape=self._payload_symbols(
                resolved_source,
                rank=len(payload_dims),
                auxiliary=(
                    payload_role == PayloadRole.AUXILIARY_STATISTIC
                ),
            ),
            post_transform=post_transform,
            binding_gap=binding_gap,
            meta=meta,
            dedup_key=dedup_key,
        )

    def _payload_symbols(
        self,
        source: Optional[FlowRef],
        *,
        rank: int,
        auxiliary: bool,
    ) -> Tuple[str, ...]:
        """Semantic axis expressions for a physical payload buffer."""
        if source is None:
            return ()
        shape = self.analysis.shapes.values.get(source.node_uid)
        if shape is None:
            return ()
        axes = _output_axes(shape, source.port)
        if axes is None:
            return ()
        if auxiliary and len(axes) == rank + 1:
            axes = axes[:-1]
        if len(axes) != rank:
            return ()
        return tuple(str(expr_of(axis)) for axis in axes)

    # -- per-record lowerings --------------------------------------------

    def events(self) -> List[PhysicalCollective]:
        """Lower records, coalescing one SSA finalize without losing targets."""
        self._event_seq = 0
        raw: List[PhysicalCollective] = []
        terminal_events: Dict[int, str] = {}
        indices_by_identity = {
            id(record): index for index, record in enumerate(
                self.analysis.redistributes)
        }
        for record_index, record in enumerate(self.analysis.redistributes):
            dependency_index = (
                indices_by_identity.get(id(record.dependency))
                if record.dependency is not None else None)
            if record.dependency is not None and dependency_index is None:
                raise ValueError(
                    f"{self._node(record.node).display_path}: ordered "
                    "transition depends on a redistribute outside this "
                    "analysis")
            with node_context(self._node(record.node).display_path):
                if record.crossing is not None and not self._pays(record):
                    continue
                source_event = terminal_events.get(dependency_index)
                if dependency_index is not None and source_event is None:
                    raise ValueError(
                        f"{self._node(record.node).display_path}: ordered "
                        "transition depends on a redistribute that produced "
                        "no event")
                produced = self._lower(record, source_event=source_event)
                raw.extend(produced)
                if produced:
                    terminal_events[record_index] = produced[-1].event_id
        from .sparse_dcp_prefill import sparse_dcp_prefill_events
        raw.extend(sparse_dcp_prefill_events(self))
        events = _coalesce_events(raw, self._node_order)
        validate_event_graph(self.analysis, events)
        return events

    def _value_count(self, record: Redistribute) -> Optional[int]:
        """How many times one rank settles the value an edge reads: once per
        producer occurrence its readers on this view read, however many
        readers read it and from whatever repetition context. None for any
        other site, which its record's own node counts.

        The readers that share the transition are the records with the same
        producer value and states (`_finalize_readers`). Each occurrence a
        reader reads is the one the walk bound (`Analysis.input_references`).
        A reader in the producer's own template iteration (a sibling read,
        `Reference.local`) prices each
        occurrence at its own instances per iteration — an expert bank reads
        the routed rows per expert; a reader in another context, a stage's
        name or a stream read, at the producer's."""
        if record.site != RedistributeSite.EDGE or record.producer < 0:
            return None
        readers = self._finalize_readers[self._finalize_key(record)]
        sources: set = set()
        per_occurrence: set = set()
        here = 0
        for reader in readers:
            occurrences = self._occurrences_here(reader.node)
            if not occurrences:
                continue
            here += 1
            reference = self.analysis.input_reference(
                reader.node, reader.input_index)
            if reference.local:
                per_occurrence.add(
                    Fraction(self._count(reader.node), len(occurrences)))
            for iteration in occurrences:
                sources.update(
                    found.occurrence for found in reference.at(iteration))
        if not here:
            # A pipeline stage that runs none of the readers runs none of
            # the transition, like every row it does not run.
            return 0
        if not sources:
            raise ValueError(
                f"{self._node(record.producer).display_path}: an edge "
                "transition whose readers on this view read no occurrence of "
                "their producer")
        if not per_occurrence:
            producer = record.producer
            per_occurrence.add(Fraction(
                int(value_of(self.analysis.instances.values[producer])),
                len(self._domains[producer])))
        if len(per_occurrence) != 1:
            raise ValueError(
                f"{self._node(record.producer).display_path}: readers of one "
                f"value in its own iteration price it at different instances "
                f"per occurrence {sorted(per_occurrence)!r}")
        count = len(sources) * per_occurrence.pop()
        if count.denominator != 1:
            raise ValueError(
                f"{self._node(record.producer).display_path}: settles "
                f"{count} times, not a whole number")
        return int(count)

    def _occurrences_here(self, uid: int) -> Tuple[Optional[int], ...]:
        """The iterations of `uid` this view runs: all of them for the whole
        pipeline, those placed on the view's stage for one stage."""
        iterations = self._domains[uid]
        if self.stage is None:
            return iterations
        placement = self.stage.placement
        return tuple(
            iteration for iteration in iterations
            if placement.stage(Occurrence(uid, iteration)) == self.stage.stage)

    @memo_property
    def _domains(self) -> Dict[int, Tuple[Optional[int], ...]]:
        return dict(self.program.occurrence_domains())

    @memo_property
    def _finalize_readers(self) -> Dict[Tuple[Any, ...], List[Redistribute]]:
        """The edge records that settle one value one way, by
        `_finalize_key`."""
        out: Dict[Tuple[Any, ...], List[Redistribute]] = {}
        for record in self.analysis.redistributes:
            if record.site == RedistributeSite.EDGE and record.producer >= 0:
                out.setdefault(self._finalize_key(record), []).append(record)
        return out

    @staticmethod
    def _finalize_key(record: Redistribute) -> Tuple[Any, ...]:
        return (record.producer, flow_port(record.producer_port),
                record.arrived, record.required, tuple(record.collectives))

    def _pays(self, record: Redistribute) -> bool:
        """Whether this view prices a hop: it serves a timed stack, and it
        crosses this stage's exit cut (any cut, for the whole pipeline)."""
        scope = (self.program.included_stack_names()
                 if self.timing_scope is None else self.timing_scope)
        return bool(record.crossing.serves & scope) and (
            self.stage is None or record.cut == self.stage.stage)

    def _lower(self, record: Redistribute, *,
               source_event: str | None = None) -> List[PhysicalCollective]:
        if not record.collectives:
            raise ValueError(
                f"{self._node(record.node).display_path}: comm record with "
                "no collectives has no materialized form"
            )
        names = tuple(name for _world, name in record.collectives)
        if record.site == RedistributeSite.CACHE_ENDPOINT:
            return endpoint_rows(
                self, record, anchor_uid=record.producer,
                meta=(("pcp_cache_write", True),),
            )
        if record.site == RedistributeSite.MODEL_OUTPUT:
            return endpoint_rows(self, record, anchor_uid=record.node)
        if record.site == RedistributeSite.INTERNAL_OUT:
            uid, port = published_producer(
                self.program, record.node,
                record.producer_port)
            if uid != record.node:
                record = replace(record, node=uid, producer=uid,
                                 producer_port=port)
        if "all_gather_lse" in names:
            return dcp_trio(self, record)
        if len(record.collectives) != 1:
            raise ValueError(
                f"{self._node(record.node).display_path}: multi-collective "
                "transition needs an explicit physical event chain "
                f"(record {record.collectives!r})"
            )
        builders = {
            "all_reduce": reduction_row,
            "reduce_scatter": scatter_row,
            "all_to_all": exchange_row,
            "all_gather": gather_row,
            "send_recv": send_row,
        }
        out: List[PhysicalCollective] = []
        for world, name in record.collectives:
            builder = builders.get(name)
            if builder is None:
                raise ValueError(
                    f"{self._node(record.node).display_path}: no "
                    f"lowering for collective {name!r} "
                    f"(record {record.collectives!r})"
                )
            if source_event is not None and name != "all_to_all":
                raise ValueError(
                    f"{self._node(record.node).display_path}: an ordered "
                    f"transition to {name!r} has no chained lowering")
            if name == "all_to_all":
                out.append(exchange_row(
                    self, record, world, source_event=source_event))
            else:
                out.append(builder(self, record, world))
        return out

    def _family_size(self, label: str) -> int:
        rule = self.analysis.rule
        family = rule.families.get(label)
        if family is None:
            raise ValueError(
                f"world {label!r} resolved no family — cannot size the "
                "collective group"
            )
        return family.group_size

    def _resolved_family(self, world: str, family: Any, event_id: str):
        registered = self.analysis.rule.families.get(world)
        if family is not None:
            if world != family.id:
                raise ValueError(
                    f"event {event_id} world {world!r} does not name its "
                    f"value family {family.id!r}")
            if registered is not None and not same(registered, family):
                raise ValueError(
                    f"event {event_id} world {world!r} resolves family "
                    f"{registered!r}, not value family {family!r}")
            return family
        if registered is None:
            raise ValueError(
                f"world {world!r} resolved no physical process-group family")
        return registered

    @staticmethod
    def _family_geometry(family: Any) -> Tuple[int, int]:
        """Group size and physical rank stride from resolved membership."""
        if family.group_size == 1:
            return 1, 1
        strides = {
            abs(right - left)
            for group in family.groups
            for left, right in pairwise(sorted(group))
        }
        if len(strides) != 1 or 0 in strides:
            raise ValueError(
                f"family {family.id!r} has non-uniform physical rank strides "
                f"{sorted(strides)!r} in groups {family.groups!r}"
            )
        return family.group_size, strides.pop()


def collect_events(
    analysis: Any,
    *,
    stage: Any = None,
    timing_scope: Optional[FrozenSet[str]] = None,
    sm_count: Optional[int] = None,
) -> List[PhysicalCollective]:
    return _Session(
        analysis, stage=stage, timing_scope=timing_scope, sm_count=sm_count,
    ).events()


def materialize_comm_events(
    analysis: Any,
    *,
    stage: Any = None,
    timing_scope: Optional[FrozenSet[str]] = None,
    sm_count: Optional[int] = None,
) -> List[PhysicalCollective]:
    """Every comm record lowered without discarding graph topology."""
    return collect_events(
        analysis,
        stage=stage,
        timing_scope=timing_scope,
        sm_count=sm_count,
    )


def materialize_comm_tree(
    analysis: Any,
    *,
    stage: Any = None,
    timing_scope: Optional[FrozenSet[str]] = None,
    sm_count: Optional[int] = None,
) -> MaterializedCommTree:
    """Graft the typed physical events into a copied full ``SpecIRTree``.

    UIDs are the only topology key.  Display paths and aggregation ids remain
    reporting facts; neither is unique across repeated template invocations.
    `stage` (a `stages.StageView`) selects one pipeline stage: what runs
    there, and the hops it sends across its exit cut, whole, as the graph's
    dataflow holds the value. `timing_scope` names the stacks whose hops are
    priced.
    """
    events = tuple(collect_events(
        analysis,
        stage=stage,
        timing_scope=timing_scope,
        sm_count=sm_count,
    ))
    copied_tree, copied = _copy_program_tree(analysis)
    graft = _CommGraft(copied_tree, copied)
    for seq, event in enumerate(events, 1):
        graft.apply(event, seq)
    graft.validate(copied, events)
    # The grafted graph, walked on the analysis's own schedule (an encode
    # step's encoder alone).
    program = Program(resolved=analysis.program.resolved, ir=copied_tree,
                      schedule=analysis.program.schedule)
    cost, shapes, instances, comm_bytes, comm_meta = _comm_layers(
        program, events, graft.nodes_by_event
    )
    return MaterializedCommTree(
        tree=copied_tree,
        program=program,
        events=events,
        node_uid_by_event=graft.node_uid_by_event(),
        cost=cost,
        shapes=shapes,
        instances=instances,
        comm_bytes=comm_bytes,
        comm_meta=comm_meta,
    )


def lower_record(
    analysis: Any, record: Any, act_dtype_bytes: float = 2.0
) -> List[PhysicalCollective]:
    return _Session(analysis, float(act_dtype_bytes))._lower(record)


def emit_event(analysis: Any, act_dtype_bytes: float = 2.0, **kwargs: Any):
    return _Session(analysis, float(act_dtype_bytes))._event(**kwargs)


def dcp_events(
    analysis: Any, record: Any, act_dtype_bytes: float = 2.0
) -> List[PhysicalCollective]:
    return dcp_trio(_Session(analysis, float(act_dtype_bytes)), record)


def reduction_event(
    analysis: Any, record: Any, world: str, act_dtype_bytes: float = 2.0
) -> PhysicalCollective:
    return reduction_row(_Session(analysis, float(act_dtype_bytes)), record, world)


def _descendant_uids(node: Any) -> set:
    found: set = set()
    stack = list(node.children or ())
    while stack:
        child = stack.pop()
        found.add(child.uid)
        stack.extend(child.children or ())
    return found


def published_producer(program: Any, uid: int, port: str = "out"
                       ) -> Tuple[int, str]:
    """The name readers use for `uid.port`.

    A component's output leaf is read under the component's name. A
    gather leaving that leaf therefore leaves the outermost wrapper
    whose output it is.
    """
    wanted = flow_port(port)
    wrappers = [
        node for node in program.nodes()
        if node.children
        and program.output_leaf(node)[0].uid == uid
        and flow_port(program.output_leaf(node)[1]) == wanted
    ]
    if not wrappers:
        return uid, wanted
    nested: set = set()
    for node in wrappers:
        nested.update(_descendant_uids(node))
    outer = [node for node in wrappers if node.uid not in nested]
    if len(outer) != 1:
        raise ValueError(
            f"{program.by_uid(uid).display_path}: output leaf of "
            f"{len(outer)} unrelated wrappers "
            f"{[node.display_path for node in outer]}"
        )
    return outer[0].uid, "out"


def consumer_of(program: Any, uid: int, port: str = "out") -> int:
    """The unique region that owns every reader of `uid.port`.

    The exit exchange is one event. Nested readers collapse into a
    wrapper that already lists the edge. Sibling readers under one
    parent collapse into that parent — `_targets_for_source` then walks
    the body and rewrites every slot. A flat inlined fan-out (no parent)
    still has one event; `consumer_roots` names every reader and this
    returns the first. Readers that do not share a parent have escaped
    the region and still fail.
    """
    return consumer_roots(program, uid, port)[0]


def consumer_roots(program: Any, uid: int, port: str = "out"
                   ) -> Tuple[int, ...]:
    """Every rewrite root for `uid.port`'s exit exchange.

    One wrapper, or sibling readers that share a parent, collapse to that
    region. A flat inlined graph (permute → SwiGLU w1 and w3, no parent
    pointer) keeps every maximal reader so each input can be rewritten.
    """
    wanted = flow_port(port)
    consumers = [
        node
        for node in program.nodes()
        if any(
            ref.src is not None
            and ref.src.uid == uid
            and flow_port(ref.port) == wanted
            for ref in (node.inputs_obj or ())
        )
    ]
    nested: set = set()
    for node in consumers:
        nested.update(_descendant_uids(node))
    maximal = [node for node in consumers if node.uid not in nested]
    if not maximal:
        where = program.by_uid(uid).display_path
        raise SpecRefusal(
            f"{where}: no consumer of port "
            f"{wanted!r} to anchor the exit exchange",
            declaration="cache",
            fix=("this node is cached but no node of the graph reads it; "
                 "cache the tensor the graph reads (the one the source "
                 "writes to the cache and splits back)"),
            node=where)
    if len(maximal) == 1:
        return (maximal[0].uid,)
    parent_uids = {
        getattr(getattr(node, "parent", None), "uid", None) for node in maximal
    }
    if len(parent_uids) == 1 and None not in parent_uids:
        return (parent_uids.pop(),)
    if parent_uids == {None}:
        return tuple(node.uid for node in maximal)
    raise ValueError(
        f"{program.by_uid(uid).display_path}: expected exactly one "
        f"region owning port {wanted!r} to anchor the exit exchange, "
        f"found {[node.display_path for node in maximal]}"
    )


def family_geometry(family: Any) -> Tuple[int, int]:
    return _Session._family_geometry(family)


def node_order(analysis: Any) -> dict:
    return {
        node.uid: index for index, node in enumerate(analysis.program.nodes())
    }
