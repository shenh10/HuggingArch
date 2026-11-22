"""`Rule` — the walk's transition rule base, with the identity answers to
the per-rank projection questions.

`Analysis.local_tensors` / `capacity.weights` / `Analysis.physical_shapes` /
`Analysis.rank_definitions` ask the walk's rule what ONE RANK holds. That answer is layout knowledge, and
eval does not import the layout package to reach it — so the question is part
of the rule's contract, and this base carries the answer for a rule that
distributes nothing: the rank's tensors are the model's tensors, priced by
the global join, its shapes are the walk's shapes, and no law of a node comes
to another value at the rank. `LayoutPropagation` overrides each with its
projection.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, Optional, Sequence

from ..ops import DISPATCH_OP
from .capacity import UnpricedTensor, Weights
from .routed import RoutedLaunch
from .storage import is_expert_axis
from .values import Extent


class Rule:
    """Identity rule: nothing is distributed, so local == global."""

    #: Identity has no sequence cut and no cache family.
    striped_cache_owners: frozenset[int] = frozenset()
    resident_sequence_family = None
    cache_family = None
    #: The phase the walk runs (`sharding.PREFILL` / `DECODE` / `ENCODE`),
    #: which the composition root hands the rule it builds
    #: (`LayoutPropagation.phase`, from `Deployment.phase`). The identity
    #: rule is built for no phase, and a question about the phase asked of
    #: its walk has no answer (`fusion.GraphCondition.phase`).
    phase = None

    def world_degree(self, world: str) -> int:
        """The degree the rank group labelled `world` (`dims.yaml`
        `worlds:`) runs at in this walk's phase: one, for a rule that
        distributes nothing."""
        return 1

    @property
    def is_trivial(self) -> bool:
        """Nothing is distributed, so nothing needs deciding."""
        return True

    def history_state(self, entry: Any) -> Any:
        """Where a history of tensors placed as `entry` lives — a
        `read: all` stream's `[B, S, m, D]` value. Asked only of a rule
        that is not trivial, as every placement question is; the identity
        rule records no states."""
        raise NotImplementedError(
            "the identity rule places nothing; the walk does not ask it")

    def model_input_state(self, name: str, shape: Sequence[Any], *,
                          encoder: bool) -> Any:
        """Where model input `name`, `[batch, rows, width?]`, lives as it
        arrives, read by an encoder stack when `encoder`. Asked only of a
        rule that is not trivial, as `history_state` is."""
        raise NotImplementedError(
            "the identity rule places nothing; the walk does not ask it")

    def stage_of(self, analysis: Any
                 ) -> Callable[[Any, int, Optional[int]], int]:
        """`(program, uid, iteration) -> stage`: the pipeline stage node
        `uid` of `program` — the walked graph, or one a rewrite built over
        it — runs on at `iteration`. The one answer to which stage a run is
        on. One stage here: nothing is cut."""
        return _one_stage

    def rank_shapes(self, analysis: Any) -> Any:
        """The rank's tensor shapes — the walk's own, undivided."""
        return analysis.shapes

    def rank_definitions(self, analysis: Any, definitions: Any,
                         global_shape: Any, local_shape: Any, node: Any, *,
                         query_shapes: Any = None, query_node: Any = None,
                         cache: Any = None) -> Dict[str, Any]:
        """Each of a node's definitions (`binding.Definition`) that comes to
        another value at this rank. Nothing is cut here, so none does."""
        return {}

    def local_tensors(self, analysis: Any) -> Iterator:
        """The rank's stored tensors — the model's own join, undivided."""
        yield from analysis.tensors()

    def local_axis_segments(self, analysis: Any) -> Dict[Any, Any]:
        """Each stored tensor's axes as the partitions a placement holds them
        in — `(uid, tensor_index, segment_index)` to, per axis, the
        `(global, local)` extent of each partition along it (one partition
        for an axis that packs nothing; a packed `gate|up` output, two).
        Nothing is distributed here, so each axis is one partition held
        whole."""
        return {(row.uid, row.tensor_index, row.segment_index):
                tuple(((axis, axis),) for axis in row.tensor.shape)
                for row in analysis.tensors()}

    def resident_weights(self, analysis: Any) -> Weights:
        """The rank's weight ledger, priced off the global join.

        `resident` rather than `saved`: what a rank holds is what the
        deployment RUNS at, which equals the checkpoint's dtype whenever the
        model is served as it was saved and differs exactly when someone asked
        for a conversion.
        """
        program = analysis.program
        parameters: Extent = 0
        metadata: Extent = 0
        buffers: Extent = 0
        by_stack: Dict[str, Extent] = {}
        by_uid: Dict[int, Weights] = {}
        for row in analysis.tensors():
            uid, tensor, rep, count = (
                row.uid, row.tensor, row.rep, row.count)
            elements = tensor.elements
            if not elements:
                continue
            fmt = rep.resident if rep is not None else None
            if fmt is None:
                raise UnpricedTensor(
                    f"{tensor.name} has no format, so its bytes are unknown. A "
                    f"tensor counted at a guessed width is a device budget that "
                    f"looks right until the load fails."
                )
            own = fmt.weight_bytes_for(elements) * count
            meta = fmt.metadata_bytes * count
            # Which bucket is the TENSOR's own answer — `StorageKind.BUFFER` is
            # set by the op that owns it, so a position table is not classified
            # here by what its name ends in. That predicate exists on the
            # header side and is the one that files `rotary_emb.inv_freq` as a
            # quantisation scale.
            node_weights = by_uid.get(uid, Weights())
            dtype = fmt.logical_dtype.upper()
            if tensor.is_parameter:
                parameters = parameters + own
                node_weights = node_weights.add(parameters=own, metadata=meta,
                                                dtype=dtype)
            else:
                buffers = buffers + own
                node_weights = node_weights.add(buffers=own, metadata=meta,
                                                dtype=dtype)
            metadata = metadata + meta
            by_uid[uid] = node_weights
            node = program.by_uid(uid)
            stack = program.stack_of(node) if node is not None else ""
            by_stack[stack] = by_stack.get(stack, 0) + own + meta
        return Weights(parameters=parameters, metadata=metadata,
                       buffers=buffers, by_stack=by_stack, by_uid=by_uid)

    def routed(self, analysis: Any) -> Dict[int, RoutedLaunch]:
        """Each leaf on a MoE routed buffer, by uid, and the launch it
        runs on the rank (`RoutedLaunch`): its dispatch's token operand, and
        the routed buffer the leaf itself is handed — its first input with
        an expert axis (`storage.is_expert_axis`), or, for the dispatch,
        the buffer it writes. Each leaf reads its own tensor, so rows a
        layout gathered inside the zone are the rows that leaf launches.

        The tensors are found on the walk's shapes (an axis's identity) and
        read at the rank on the one per-rank projection
        (`Analysis.physical_shapes`, the walk's own shapes where nothing is
        distributed), at the same positions — so this record and every
        per-rank shape reader see one number."""
        program = analysis.program
        shapes = analysis.shapes.values
        rank = analysis.physical_shapes().values
        out: Dict[int, RoutedLaunch] = {}

        def expert_axis(dims: Any) -> Optional[int]:
            return next((axis for axis, extent in enumerate(dims or ())
                         if is_expert_axis(extent)), None)

        for uid, shape in shapes.items():
            node = program.by_uid(uid)
            if node is None or node.children:
                continue
            dispatch = program.dispatch_of(node)
            if dispatch is None or dispatch not in shapes:
                continue
            if node.op == DISPATCH_OP:
                axis, buffer = expert_axis(shape.out), rank[uid].out
            else:
                index = next((i for i, dims in enumerate(shape.ins)
                              if expert_axis(dims) is not None), None)
                if index is None:
                    continue
                axis = expert_axis(shape.ins[index])
                buffer = rank[uid].ins[index]
            if axis is None:
                raise ValueError(
                    f"{node.display_path}: the dispatch writes no buffer "
                    "with an expert axis")
            out[uid] = RoutedLaunch.of(dispatch, rank[dispatch].ins[0],
                                       buffer, axis,
                                       where=node.display_path)
        return out


def _one_stage(_program: Any, _uid: int, _iteration: Optional[int]) -> int:
    return 0
