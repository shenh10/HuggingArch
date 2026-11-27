"""Fusion — the graph transform, and the only one there is.

What a fusion plan declares
---------------------------
`backend/inference/fusion_plans/*.yaml` names a fused op and the naive subgraph
it replaces, in one of two forms:

    chain     nodes: {add: {op: add, inputs: [external.x, external.residual]},
                      rmsnorm: {op: rmsnorm, inputs: [add.out]}}
    parallel  parallel: {op: linear, shared_inputs: [external.x], min_count: 2}

and an `io_mapping` from the fused op's operands to matched ports. The registry
is `extra="forbid"` and carries no cost arithmetic: cost preservation is the
`Cost.combine` law used when the cost layer crosses the rewrite.

So the transform produces a correspondence, not a result
--------------------------------------------------------
`Rewrite` says which old nodes each new node came from; every layer then crosses
with its OWN combine, which is the part a single synthesised result cannot
express:

    cost   Cost.combine     sums — a fused QKV does all three projections
    shape  Shape.combine    spans — reads the first's input, makes the last's output
    bytes  NOT a sum        flash attention exists precisely to avoid
                            materialising the score matrix, so the traffic of
                            the fused op is less than the traffic of its parts

That last row is why `rebase` takes the combine per layer instead of the
transform declaring one. A scheme that changes cost rather than preserving it
would have to say so; none of the seventeen do, which is the registry's own
stated design.

Ordering
--------
Fusion applies AFTER the walk, not before. The fused op's cost is defined by its
parts, so the parts have to be evaluated for it to exist at all — and there is
no op spec for `fused_add_rmsnorm` to evaluate directly. That is the opposite of
what a graph transform usually implies, and it follows from the registry
carrying no cost of its own.
"""
from __future__ import annotations

from backend.arch.axis_propagation import dim_map
from backend.arch.ops import (
    OP_REGISTRY, ShapeKind, attention_kernel, repeats_input,
)

from dataclasses import dataclass, field
from typing import (
    Any, ClassVar, Dict, FrozenSet, Iterator, List, Mapping, Optional,
    Sequence, Tuple,
)

import sympy as sp

from .fusion_order import refuse_tied_preemption
from ..activation_formats import ACTIVATION_FORMATS
from .analysis import OwnCopies
from .cache import (Edges, cached_node_uids, is_view, main_port,
                    state_carrier_uids)
from .dim import exact, is_whole, same_law, value_of, where

from .launch import FusedNode
from .layer import Layer, Rewrite
from .occurrence import Occurrence
from .traffic import TensorReaders, reader_format
from .values import Cost, Moved, Shape, product
from backend.code_trace.memo import memo_property


# -- what a match produced ------------------------------------------------

def _routed_span(program: Any, members: Sequence[Any]) -> bool:
    """Whether a fusion span runs on dispatched tokens: its members agree
    (siblings share the zone), and a span that straddled the zone boundary
    would be a kernel no framework runs."""
    flags = {program.in_zone(m) for m in members}
    if len(flags) > 1:
        raise ValueError(
            f"fusion span {sorted(m.uid for m in members)!r} straddles the "
            "dispatch zone: some members are routed and some are not")
    return flags.pop() if flags else False


@dataclass(frozen=True)
class FusedProgram:
    """The program with some nodes replaced, as a view rather than a rebuild.

    Delegates everything to the program it wraps. Rebuilding a `SpecIRTree`
    would mean re-deriving adapter transparency, weight naming and multiplicity for
    nodes whose only change is that several of them are now one.
    """

    base: Any
    fused: Tuple[FusedNode, ...] = ()
    absorbed: FrozenSet[int] = frozenset()
    #: This pass's rewrite: which runs of each node each of its launches
    #: took (`Rewrite.taken`), the one statement every run count here is
    #: read off.
    rewrite: Rewrite = field(default_factory=Rewrite)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @memo_property
    def routed_uids(self) -> FrozenSet[int]:
        """The base's `Program.routed_uids`, and each fused kernel whose
        origins run on dispatched tokens (a span cannot straddle the zone,
        `_routed_span`)."""
        return self._with_fused(self.base.routed_uids)

    @memo_property
    def dispatch_members(self) -> FrozenSet[int]:
        """The base's `Program.dispatch_members`, and each fused kernel that
        absorbed one: a launch that took a dispatch boundary in is one."""
        return self._with_fused(self.base.dispatch_members)

    def _with_fused(self, members: FrozenSet[int]) -> FrozenSet[int]:
        return members | frozenset(
            fused.uid for fused in self.fused
            if any(uid in members for uid in fused.origins))

    def nodes(self) -> Iterator[Any]:
        """The base's nodes in its order, the absorbed ones left out and each
        fused kernel standing where it runs: once its first member's turn
        has come and every tensor it reads from outside the span has been
        written. A consumer that reads the order as a schedule (activation
        liveness) sees a launch between its producers and its readers,
        not after the whole layer."""
        order = list(self.base.nodes())
        slots = self._slots(order)
        by_slot: Dict[int, List[Tuple[int, int, FusedNode]]] = {}
        for fused in self.fused:
            slot, depth = slots[fused.uid]
            by_slot.setdefault(slot, []).append((depth, fused.uid, fused))
        for index, node in enumerate(order):
            if node.uid not in self.absorbed:
                yield node
            for _depth, _uid, fused in sorted(by_slot.get(index, ()),
                                              key=lambda row: row[:2]):
                yield fused

    def _slots(self, order: Sequence[Any]) -> Dict[int, Tuple[int, int]]:
        """Fused uid -> `(slot, depth)`: the base position it follows, and
        how many fused kernels of this pass it reads through at that slot
        (a reader after its producer). The slot is the later of its first
        member's position and the end of each outside producer it reads —
        a component producer ends at its last descendant, an absorbed one
        at the kernel that absorbed it."""
        index = {node.uid: i for i, node in enumerate(order)}
        end = dict(index)
        for i, node in enumerate(order):
            parent = node.parent
            while parent is not None and parent.uid in end:
                end[parent.uid] = max(end[parent.uid], i)
                parent = parent.parent
        # Origin -> its kernel, this pass's and every earlier pass's (an
        # earlier pass's members are gone from `order`; its kernel stands
        # in it).
        owner: Dict[int, int] = {}
        program: Any = self
        while isinstance(program, FusedProgram):
            for fused in program.fused:
                for uid in fused.origins:
                    owner.setdefault(uid, fused.uid)
            program = program.base
        mine = {fused.uid: fused for fused in self.fused}
        slots: Dict[int, Tuple[int, int]] = {}

        def done(uid: int) -> Tuple[int, int]:
            if uid in mine:
                return slot_of(mine[uid])
            if uid in end and uid not in self.absorbed:
                return end[uid], -1
            if uid in owner:
                return done(owner[uid])
            raise ValueError(f"fused program has no position for uid {uid}")

        def slot_of(fused: FusedNode) -> Tuple[int, int]:
            if fused.uid not in slots:
                # A launch across a seam runs where the reading iteration
                # starts it: its previous-iteration members ran a layer
                # earlier, and their position says nothing of this one.
                slot = min(index[uid] for uid in fused.origins
                           if uid not in fused.previous_iteration)
                depth = 0
                for ref in fused.inputs_obj:
                    if ref.src is None:
                        continue
                    at, through = done(ref.src.uid)
                    if at > slot:
                        slot, depth = at, through + 1
                    elif at == slot:
                        depth = max(depth, through + 1)
                slots[fused.uid] = (slot, depth)
            return slots[fused.uid]

        for fused in self.fused:
            slot_of(fused)
        return slots

    def walked_nodes(self) -> Iterator[Any]:
        """`nodes` of the walked stacks: a kernel fuses walked nodes only."""
        walked = {node.uid for node in self.base.walked_nodes()}
        mine = {fused.uid for fused in self.fused}
        for node in self.nodes():
            if node.uid in walked or node.uid in mine:
                yield node

    def by_uid(self, uid: int) -> Any:
        for node in self.fused:
            if node.uid == uid:
                return node
        return self.base.by_uid(uid)

    def occurrence_layers(self, uid: int) -> Tuple[Optional[int], ...]:
        """The iterations a node runs at, read off this pass's one statement
        of which runs it took (`Rewrite.taken`): a launch's own, a node a
        launch took only some runs of the ones it has left, and any other
        node as before."""
        if uid in self.rewrite.taken:
            return _ordered(self.rewrite.iterations(uid))
        if self.rewrite.partly(uid):
            return _ordered(frozenset(self.base.occurrence_layers(uid))
                            - self.rewrite.taken_at(uid))
        return self.base.occurrence_layers(uid)

    def block_of(self, uid: int) -> Optional[int]:
        """A fused kernel runs in the layer its home ran in
        (`FusedNode.home`) — the reading iteration's, for a launch across a
        seam; a match never spans two block invocations otherwise."""
        for node in self.fused:
            if node.uid == uid:
                return self.base.block_of(node.home)
        return self.base.block_of(uid)

    def repeated_in(self, uid: int) -> Tuple[int, ...]:
        """And it repeats as they did — a match never spans two repeat
        scopes either, so its home origin answers for all of them."""
        for node in self.fused:
            if node.uid == uid:
                return self.base.repeated_in(node.home)
        return self.base.repeated_in(uid)


def _ordered(iterations: Any) -> Tuple[Optional[int], ...]:
    return tuple(sorted(iterations, key=lambda it: (it is None, it or 0)))


@dataclass(frozen=True)
class _Taken:
    """What one match takes: the launch's home and the iterations of it the
    launch runs at, and per member which of its iterations each of the
    launch's runs takes (the same one, or for a member of the previous
    layer the writing one) — `Rewrite.taken`."""

    home: int
    iterations: FrozenSet[Optional[int]]
    by_member: Mapping[int, Mapping[Optional[int], Optional[int]]]


@dataclass
class _Runs:
    """The iterations each node still runs at on its own within one
    rewrite — the one statement of how often a node runs, from which every
    count is read (`Rewrite.taken`, `FusedProgram.occurrence_layers`).

    A launch within one iteration takes every run its members have left. A
    launch across a seam takes the runs of the seams it spans
    (`_Binding.pairs`): the reading layer's members at the reading
    iterations, the previous layer's at the writing ones. A member of a
    layer whose first iteration has no seam before it — the norm that opens
    a stack — keeps that run, and so stays in the graph; the next match (or
    pass) takes what is left. Node-level all-or-nothing is what made a seam
    impossible to fuse; this is that answer at iteration grain."""

    program: Any
    left: Dict[int, FrozenSet[Optional[int]]]

    @classmethod
    def of(cls, program: Any) -> "_Runs":
        return cls(program, {})

    def has(self, uid: int) -> FrozenSet[Optional[int]]:
        """The iterations `uid` still runs at on its own."""
        if uid in self.left:
            return self.left[uid]
        return frozenset(self.program.occurrence_layers(uid))

    def take(self, match: _Binding, members: Sequence[int],
             seams: Seams) -> Optional[_Taken]:
        """What `match` takes, None for a match with no run left to take
        or whose members do not run as one launch."""
        if match.pairs is None:
            took = frozenset.intersection(*(self.has(uid) for uid in members))
            if not took:
                return None
            return _Taken(members[0], took, {
                uid: {it: it for it in took} for uid in members})
        # The reading members of one launch run at one set of iterations:
        # each formal read the binding crossed is read by them.
        readers = {seams.readers[key] for key in match.crossed}
        if len({frozenset(self.program.occurrence_layers(uid))
                for uid in readers}) != 1:
            return None
        writers = {source.uid for _it, source in match.pairs}
        if len(writers) != 1:
            return None
        previous = match.previous
        reading = frozenset.intersection(
            *(self.has(uid) for uid in members if uid not in previous))
        writing = frozenset.intersection(
            *(self.has(uid) for uid in members if uid in previous))
        # A launch across a seam runs in the reading layer, after the layer
        # its previous-layer members belong to: a reader in that layer would
        # read a tensor not yet written (the MTP layer's closing add, read by
        # its own shared-head norm before the stack's exit). So a pair whose
        # writing run its own layer reads, outside the launch, stays; the
        # others are taken, and the layers that differ in this fall into
        # iteration classes of their own (`Rewrite.iteration_classes`).
        inside = frozenset(members)
        pairs = [(it, source) for it, source in match.pairs
                 if it in reading and source.iteration in writing
                 and not any(seams.read_in_layer(uid, source.iteration)
                             - inside for uid in previous)]
        if not pairs:
            return None
        home = min(uid for uid, _index in match.crossed)
        return _Taken(home, frozenset(it for it, _source in pairs), {
            uid: ({it: source.iteration for it, source in pairs}
                  if uid in previous else {it: it for it, _source in pairs})
            for uid in members})

    def commit(self, taken: _Taken) -> FrozenSet[int]:
        """Take `taken`'s runs; the members it took every run of."""
        whole: set = set()
        for uid, at in taken.by_member.items():
            rest = self.has(uid) - frozenset(at.values())
            if rest:
                self.left[uid] = rest
            else:
                self.left.pop(uid, None)
                whole.add(uid)
        return frozenset(whole)


# -- matching -------------------------------------------------------------

def _producers(node: Any,
               edges: Optional[Mapping[int, Tuple]] = None
               ) -> List[Tuple[Optional[int], str]]:
    """`(producer uid, port)` per declared input, uid `None` when the ref is
    unresolved (it names an enclosing scope's input rather than a node).

    `edges` is a boundary view's resolved table (`BoundaryView.edges`): the
    same list with every wrapper producer replaced by the leaf that writes
    the port the input reads."""
    if edges is not None and node.uid in edges:
        return [(src if isinstance(src, int) else None, port)
                for src, port in edges[node.uid]]
    return [(ref.src.uid if ref.src is not None else None,
             ref.port or "out")
            for ref in (node.inputs_obj or ())]


def _producer_ids(node: Any,
                  edges: Optional[Mapping[int, Tuple]] = None
                  ) -> List[Tuple[Any, str]]:
    """`_producers`, with an unresolved ref carrying its IDENTITY instead of
    `None`. An alias names the enclosing scope's tensor (`external_sid`), and
    two operands reading the same alias read the same tensor — treating that
    as "no producer" is how `q_a_proj`/`kv_a_proj`, both fed by the block
    input, could not be grouped as siblings. The same blindness cost `moe_out`
    its second input shape in `traffic`."""
    if edges is not None and node.uid in edges:
        return list(edges[node.uid])
    out: List[Tuple[Any, str]] = []
    for ref in (node.inputs_obj or ()):
        if ref.src is not None:
            out.append((ref.src.uid, ref.port or "out"))
        else:
            out.append((("alias", ref.external_sid), ref.port or "out"))
    return out


@dataclass
class ViewChains:
    """Where a quantised write is the same write made above the views it
    reads through.

    A view moves no bytes: it re-indexes its input (`OpSpec.is_view_op`, a
    view-only component as a whole, `cache.is_view`). So quantising a
    view's output IS quantising its input, wherever the view hands every
    run of elements that share one scale (`ActivationFormat.scale_block`)
    to one such run — the quantised tensor is then the same elements and
    the same scales, only indexed differently. That is a question of the
    view's dim correspondence (`axis_propagation.dim_map`) and the format,
    asked of each leaf view, and of nothing named:

      * it moves as many elements as it is handed — a narrowing or a
        repetition changes what the cast would quantise;
      * its innermost output axis is made of its innermost input axis alone:
        that axis as it was (`from`), the trailing part of it (`split`), or
        a merge ending in it;
      * one scale covers as many elements on both sides, and that run
        divides both innermost extents — a 128-group over a 192-wide row
        straddles rows, so DeepSeek-V3's `q_nope_slice` does not commute.

    The chain also has to be the tensor's only path: every link, and the
    tensor it starts from, is read by the next link alone, and by nothing
    past the graph's edges (`traffic.TensorReaders`, the question the cast
    graft asks before moving a cast above a copy). A second reader of the
    wide tensor keeps the cast out of the producer's launch, as fusion's
    closure does for any member (`Fusion._closed`).

    The serving graph moves a cast above a view only where the view keeps
    the innermost axis whole (`traffic.ElementCopy`); after a view that
    regroups it (`head_merge`) the cast stays on the tensor the GEMM is
    handed, and a standalone cast is measured there. Only fusion, which
    answers what the framework launches, reads through: a plan naming the cast as a member
    of the producer's launch binds the producer across the chain, and the
    chain's views join that launch, which then writes the quantised tensor
    the views re-index.
    """

    program: Any
    #: Each view leaf's tensors ON ONE RANK, by uid: the cast groups the
    #: rank's own tensor, so a scale block must divide the rank's innermost
    #: extent, not the model's. None where the walk is distributed and the
    #: caller handed no per-rank layer.
    shapes: Optional[Mapping[int, Any]]
    #: Who reads each tensor (`traffic.TensorReaders`).
    tensor_readers: TensorReaders
    _roots: Dict[Tuple[int, int], Any] = field(default_factory=dict)

    @classmethod
    def of(cls, program: Any, analysis: Optional[Any],
           local_shapes: Any = None, graft: Any = None) -> "ViewChains":
        """Over the per-rank shapes the caller materialised
        (`layout.physical_kernel_shapes`, which this package may not
        import). A walk that distributes nothing is its own per-rank answer
        (`Rule.is_trivial`: local == global). Who reads a tensor past the
        edges is asked of the graph this fusion pass started from (a fused
        program's base: absorbing nodes changes no such reader), which
        `graft` made from the walked one (`Fusion.rewrite`)."""
        base = program
        while isinstance(base, FusedProgram):
            base = base.base
        readers = TensorReaders.of(analysis, program, graph=base,
                                   rewrite=graft)
        if local_shapes is not None:
            return cls(program, local_shapes.values, readers)
        if analysis is not None and analysis.rule.is_trivial:
            return cls(program, analysis.shapes.values, readers)
        # No rank's tensor to group a quantised write by: either the walk
        # distributes and the caller handed no per-rank layer, or there is
        # no walk at all (a rewrite asked of a graph alone, which is how the
        # matcher's own tests ask what a pattern binds).
        return cls(program, None, readers)

    def root(self, node: Any, index: int
             ) -> Optional[Tuple[int, str, FrozenSet[int]]]:
        """`(producer uid, port, chain)` when input `index` of `node` reaches
        a producer through views its quantised write commutes with — `chain`
        the views' uids, wrappers and leaves — else None."""
        key = (node.uid, index)
        if key not in self._roots:
            self._roots[key] = self._resolve(node, index)
        return self._roots[key]

    def through(self, bound: Sequence[Any]) -> FrozenSet[int]:
        """The views a match reads through: those of a member's input whose
        producer is another member."""
        inside = {member.uid for member in bound}
        out: set = set()
        for member in bound:
            for index in range(len(member.inputs_obj or ())):
                found = self.root(member, index)
                if found is not None and found[0] in inside:
                    out |= found[2]
        return frozenset(out)

    def _resolve(self, node: Any, index: int
                 ) -> Optional[Tuple[int, str, FrozenSet[int]]]:
        fmt = None if isinstance(node, FusedNode) else node.output_format
        refs = tuple(node.inputs_obj or ())
        if fmt is None or not 0 <= index < len(refs):
            return None
        reader, at, link, port = node, index, refs[index].src, refs[index].port
        chain: set = set()
        while link is not None and is_view(link):
            if not self.tensor_readers.only(link, port, [(reader, at)]):
                return None
            leaves = ([link] if not link.children
                      else [leaf for leaf in _descendants(link)
                            if not leaf.children])
            if not all(self._keeps_blocks(leaf, fmt) for leaf in leaves):
                return None
            chain |= {link.uid, *(leaf.uid for leaf in leaves)}
            inputs = tuple(link.inputs_obj or ())
            if len(inputs) != 1:
                return None
            reader, at, link, port = link, 0, inputs[0].src, inputs[0].port
        if not chain or link is None:
            return None
        if not self.tensor_readers.only(link, port, [(reader, at)]):
            return None
        return link.uid, port or "out", frozenset(chain)

    def _keeps_blocks(self, leaf: Any, fmt: Any) -> bool:
        if self.shapes is None:
            raise ValueError(
                f"{leaf.display_path}: whether a cast "
                "commutes with this view is a question of the rank's own "
                "tensor, and this distributed walk was handed no per-rank "
                "shapes (`Framework.rewrite(..., local_shapes=)`)")
        shape = self.shapes.get(leaf.uid)
        if shape is None or len(shape.ins) != 1:
            return False
        handed, written = tuple(shape.ins[0]), tuple(shape.out or ())
        if not handed or not written or not same_law(product(handed),
                                                     product(written)):
            return False
        sources = dim_map(leaf.op, leaf, in_shapes=shape.ins)
        if not sources or len(sources) != len(written):
            return False
        last, inner = sources[-1], (0, len(handed) - 1)
        refs = tuple(last.refs)
        from_inner = (
            (last.kind == "from" and refs == (inner,))
            or (last.kind == "split" and refs == (inner,) and last.part == 1)
            or (last.kind == "merge" and refs and refs[-1] == inner))
        block = fmt.scale_block(handed[-1])
        return (from_inner
                and same_law(block, fmt.scale_block(written[-1]))
                and is_whole(exact(handed[-1]) / exact(block))
                and is_whole(exact(written[-1]) / exact(block)))


def _descendants(node: Any) -> Iterator[Any]:
    for child in node.children or ():
        yield child
        yield from _descendants(child)


def _scopes(program: Any) -> Dict[int, Tuple[str, Optional[int]]]:
    """uid → `(template, parent uid)`, the position a match may not cross.

    A fusion is one kernel launch; its operands live in one component
    instance of one block. Without this constraint the matcher fused
    gemma-4's vision and audio embedders into a single QKV kernel and
    Kimi-K3's once-per-model `add_pe` with a per-layer `norm0` —
    The matcher rejects both as component-boundary crossings, read off
    `component_paths`. Parenthood here is read off the tree itself.
    """
    out: Dict[int, Tuple[str, Optional[int]]] = {}
    for node in program.nodes():
        out.setdefault(node.uid, (node.template, None))
        for child in node.children:
            out[child.uid] = (child.template, node.uid)
    return out


CROSSES = ("component",)


@dataclass(frozen=True)
class BoundaryView:
    """The candidate graph a boundary-crossing plan matches over.

    A framework kernel can span our component boundaries: DeepSeek-V4's
    mHC post+pre launch reads the leaves of `hc_post` and writes those of
    `hc_pre`, and its big-fuse ends in the sublayer RMSNorm, a leaf of the
    block itself. A binding entry says so with `crosses:`; everything else
    keeps matching the template graph exactly as before (`crosses` unset →
    no view, and `_producer_ids` / `_consumers` / `_scopes` answer as they
    always did).

    Two things change for an entry that opts in:

    * **the producer of an edge.** A leaf's declared input names the
      component it reads, not the leaf inside it that writes the port
      (`ffn_norm.in = hc_ffn_pre[out]`). `edges` is that list with every
      wrapper producer replaced by `Program.output_leaf(wrapper, port)` —
      secondary ports included, so `post_exp.in = hc_attn_pre[post]`
      resolves to the `hc_split_sinkhorn` leaf and keeps the port. A formal
      of a nested component needs no resolution: the walk already bound it
      to the outer producer.
    * **the scope a match may not leave.** `scopes` is the BLOCK OCCURRENCE
      (`Program.block_of`) and the repeated instances inside it
      (`Program.repeated_in`), instead of the immediate parent. A node with no
      block occurrence — a model-level component instance, the network-exit
      `hc_head` — is not in the view at all, so a `crosses: component` plan
      cannot reach it.

    `_scopes`'s own false positives stay rejected, and the two the wider
    scope could have let through are the ones it is written against:
    gemma-4's image and video embedders are two occurrences of one template,
    and Kimi-K3's vision tower ends in a norm whose producer resolves into a
    component that runs 27 times. Different block, or different repeat —
    either way, not one launch. `tests/backend/arch/eval/test_eval_fusion.py`
    asserts both, with the legitimate match of the same scheme beside it.
    """

    leaves: Tuple[Any, ...]
    scopes: Mapping[int, Any]
    edges: Mapping[int, Tuple[Tuple[Any, str], ...]]

    @classmethod
    def of(cls, program: Any, leaves: Sequence[Any]) -> "BoundaryView":
        scopes = {}
        for node in leaves:
            block = program.block_of(node.uid)
            if block is not None:
                scopes[node.uid] = ("block", block,
                                    program.repeated_in(node.uid))
        inside = [node for node in leaves if node.uid in scopes]
        edges = {node.uid: tuple(_writing_leaves(program, node))
                 for node in inside}
        return cls(tuple(inside), scopes, edges)


def _writing_leaves(program: Any, node: Any) -> Iterator[Tuple[Any, str]]:
    """`node`'s inputs with every component producer resolved to the leaf
    that writes the port the input reads."""
    for ref in (node.inputs_obj or ()):
        port = ref.port or "out"
        if ref.src is None:
            yield (("alias", ref.external_sid), port)
            continue
        if not ref.src.children:
            yield (ref.src.uid, port)
            continue
        leaf, written = program.output_leaf(ref.src, port)
        yield (leaf.uid, written or "out")


def _input_ports(pattern: Any, bound: Mapping[str, Any]
                 ) -> Dict[Tuple[int, int], str]:
    """`(member uid, input index) -> port`: the `external.<name>` the plan
    declares at each member input (`NaivePattern.input_ports`). A chain
    pattern names every input of a bound node (`_bind` binds a node only
    when its inputs line up with the declared refs); a parallel form names
    each sibling's inputs by `shared_inputs`."""
    out: Dict[Tuple[int, int], str] = {}
    if pattern.parallel is not None:
        names = [ref.partition(".")[2]
                 for ref in pattern.parallel.shared_inputs]
        for node in bound.values():
            for index, name in enumerate(names):
                out[(node.uid, index)] = name
        return out
    for pattern_name, node in bound.items():
        spec = pattern.nodes[pattern_name]
        for index, ref in enumerate((*spec.inputs, *spec.optional_inputs)):
            head, _, tail = ref.partition(".")
            # An edge to an optional node the match does not have reads
            # what that node would have read (`PatternNode.read_through`),
            # as the matcher bound it: a router with no gate temperature
            # reads its logits, the launch's `gating` port.
            while head != "external" and head not in bound:
                through = pattern.nodes[head].read_through()
                if through is None:
                    break
                head, _, tail = through.partition(".")
            if head == "external":
                out[(node.uid, index)] = tail
    return out


def _external_inputs(members: Sequence[Any],
                     ports: Mapping[Tuple[int, int], str],
                     order: Sequence[str],
                     shared: Mapping[str, str],
                     edges: Optional[Mapping[int, Tuple]] = None,
                     internal: FrozenSet[int] = frozenset(),
                     crossed: FrozenSet[Tuple[int, int]] = frozenset(),
                     ) -> List[Tuple[int, int, Any, Tuple[str, ...]]]:
    """`(member_uid, input_index, ref, ports)` for every tensor entering the
    match from outside: one entry per distinct outside producer, named by
    the plan's input ports it fills (`ports`, `_input_ports`), in the plan's
    port order (`order`, `NaivePattern.input_ports`).

    The single source for the fused boundary's input side: `Fusion.rewrite`
    stores the refs and their ports on the `FusedNode` and `rebase_analysis`
    reads the dims off the same list, so the views are aligned by
    construction. Which operand is which is the plan's declaration, not the
    order the members' uids happen to sort in — that order differs between
    a layer and a seam, and between two layers' templates.

    The dedup is the fusion's own semantic — three parallel projections
    fused into one kernel read their shared activation once, and an MLA
    kernel reads the latent that is both its K and its V once — so one
    entry can fill several ports (`{k, v}`). A port that two tensors fill
    is refused: an operand is one tensor. `internal` names further nodes
    inside the launch that no pattern node binds — the views a quantised
    write was read through (`ViewChains.through`) — so an edge out of them
    is no input. `crossed` names the formal reads a match bound across a
    seam (`_Binding.crossed`): their producer is a member of the previous
    iteration, so they are no input either.
    """
    inside = {n.uid for n in members} | set(internal)
    rank = {port: position for position, port in enumerate(order)}
    first: Dict[Any, Tuple[int, int, Any]] = {}
    named: Dict[Any, set] = {}
    by_port: Dict[str, Any] = {}
    for member in sorted(members, key=lambda n: n.uid):
        resolved = (edges or {}).get(member.uid)
        for index, ref in enumerate(tuple(member.inputs_obj or ())):
            if (member.uid, index) in crossed:
                continue
            src = ref.src
            # Membership is judged on the view's producer — the leaf that
            # writes what this input reads — while the ref stays the one the
            # graph declares, because that is what traffic walks.
            producer = (resolved[index][0] if resolved is not None
                        else (src.uid if src is not None else None))
            if isinstance(producer, int) and producer in inside:
                continue
            port = ports.get((member.uid, index))
            if port not in rank:
                raise ValueError(
                    f"{member.display_path or member.name}: input {index} "
                    f"enters the launch from outside but the plan names it "
                    f"{port!r}, not one of its input ports {list(order)!r}")
            key = (
                src.uid if src is not None else (
                    "alias",
                    ref.external_sid or f"{member.uid}:{index}",
                ),
                ref.port or "out",
            )
            if by_port.setdefault(port, key) != key:
                raise ValueError(
                    f"{member.display_path or member.name}: port {port!r} "
                    f"is filled by two tensors in one launch; an operand "
                    f"is one tensor")
            first.setdefault(key, (member.uid, index, ref))
            named.setdefault(key, set()).add(port)
    # A tensor of several ports is read as the one the others are declared
    # to share it with (`NaivePattern.shared_ports`) — listed first, the
    # rest after it in the plan's order (`OpSignature.at_ports`).
    entries = []
    for key, names in named.items():
        if len(names) == 1:
            entries.append((*first[key], tuple(names)))
            continue
        read = [name for name in names if name not in shared]
        if len(read) != 1 or any(shared[name] != read[0]
                                 for name in names if name != read[0]):
            member, index, _ref = first[key]
            raise ValueError(
                f"uid {member} input {index}: one tensor fills the ports "
                f"{sorted(names, key=rank.__getitem__)}, and its plan does "
                f"not declare which it is read as (`shared_ports`: "
                f"{dict(shared)})")
        (read_as,) = read
        rest = sorted(names - {read_as}, key=rank.__getitem__)
        entries.append((*first[key], (read_as, *rest)))
    return sorted(entries, key=lambda entry: rank[entry[3][0]])


def _boundary_outputs(members: Sequence[Any],
                      consumers: Mapping[int, List[int]]
                      ) -> Tuple[Optional[int], Tuple[int, ...]]:
    """`(sink_uid, exposed_uids)` — which member outputs survive the fusion.

    A member the outside still reads (or that nothing reads at all — a graph
    output) keeps its tensor in HBM; the rest were the intermediates the
    kernel exists to not materialise. The LAST surviving member in program
    order is the fused op's own output; the others are boundary ports. The
    closure rule already guaranteed every outside-read member is one the
    scheme exposes, so no scheme knowledge is needed here.
    """
    inside = {n.uid for n in members}
    visible = [
        n.uid for n in sorted(members, key=lambda m: m.uid)
        if any(c not in inside for c, _port in consumers.get(n.uid, ()))
        or not consumers.get(n.uid)
    ]
    if not visible:
        return None, ()
    return visible[-1], tuple(visible[:-1])


def _named_member(pattern: Any, bound: Mapping[str, Any], uid: int) -> int:
    """The member a launch whose output `uid` writes is named for. A grafted
    activation cast (`provenance == "cast"`) is named for the GEMM it
    quantises for (`<gemm>__cast`, `materialize_cast`), so a launch that
    writes its output quantised is named for the member the cast reads,
    read through along the pattern's one internal edge into it, not for
    the GEMM downstream."""
    names = {node.uid: name for name, node in bound.items()}
    name = names.get(uid)
    while name is not None and bound[name].provenance == "cast":
        through = (pattern.nodes or {})[name].internal_inputs()
        if len(through) != 1:
            break
        name = through[0].rsplit(".", 1)[0]
        uid = bound[name].uid
    return uid


def _walked_output(pattern: Any, bound: Mapping[str, Any], head: str,
                   port: str) -> Optional[Tuple[Any, str]]:
    """The walked tensor a match's primary output stands for: the node and
    port `io_mapping` names, an absent optional sink falling through to its
    input (`NaivePattern.port_node`), and a grafted cast read through to the
    member it re-encodes (`_named_member`) — a condition asks the walk about
    the launch's output, and the walk ran no cast."""
    name = pattern.port_node(head, bound)
    if name is None:
        return None
    node = bound[name]
    named = _named_member(pattern, bound, node.uid)
    if name == head and named == node.uid:
        return node, main_port(port)
    return next(n for n in bound.values() if n.uid == named), ""


def _consumers(nodes: Sequence[Any],
               edges: Optional[Mapping[int, Tuple]] = None
               ) -> Dict[int, List[Tuple[int, str]]]:
    """uid → `(reader uid, port read)` pairs. A match may only absorb a node
    whose every consumer is inside the match, or fusing it would cut an edge
    the rest of the graph still reads.

    The port is carried because one leaf can write several (`hc_split_sinkhorn`
    writes `pre` to a reader inside its own component and `post` / `comb` to
    readers outside it): a boundary-crossing match is closed per port
    (`Fusion._closed`)."""
    out: Dict[int, List[Tuple[int, str]]] = {}
    for node in nodes:
        for producer, port in _producers(node, edges):
            if producer is not None:
                out.setdefault(producer, []).append((node.uid, port))
    return out


@dataclass
class OperandRoles:
    """Which operands each node's output becomes — the set of input roles
    (`OpSpec.input_roles`: q, k, v, …) it reaches, unioned over every path
    downstream. Each path stops at the first op that declares roles and
    contributes the role of the position it enters there; a node whose
    output enters a score as K on one path and an apply as V on another
    reaches {k, v} (DeepSeek-V4's shared `kv_pe_rope`).

    This is the identity a fusion port names. `fused_qk_rope` exposes
    `q_out` and `k_out` from two ropes with the same op and no edge between
    them; which one is Q is not in the pair at all, it is in what the pair
    feeds — the attention score reads its query at position 0 and its keys
    at 1, and says so in its registration. A pattern node's `feeds` is
    checked against this.

    Read off the unfused graph. The identity belongs to the model's
    dataflow, and a fusion applied earlier replaces the score with a node
    that declares no roles.
    """

    consumers: Mapping[int, Tuple[Tuple[Any, int], ...]]
    _memo: Dict[int, FrozenSet[str]] = field(
        default_factory=dict, init=False, repr=False)

    @classmethod
    def of(cls, program: Any) -> "OperandRoles":
        while isinstance(program, FusedProgram):
            program = program.base
        consumers: Dict[int, List[Tuple[Any, int]]] = {}
        for node in program.nodes():
            for position, ref in enumerate(node.inputs_obj or ()):
                if ref.src is not None:
                    consumers.setdefault(ref.src.uid, []).append(
                        (node, position))
        return cls({uid: tuple(c) for uid, c in consumers.items()})

    def of_node(self, uid: int) -> FrozenSet[str]:
        if uid in self._memo:
            return self._memo[uid]
        roles: set = set()
        seen: set = set()
        todo = [uid]
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            for consumer, position in self.consumers.get(current, ()):
                declared = tuple(consumer.declaration.input_roles or ()) \
                    if consumer.declaration is not None else ()
                if not declared:
                    todo.append(consumer.uid)
                    continue
                if position < len(declared) and declared[position]:
                    roles.add(declared[position])
        self._memo[uid] = frozenset(roles)
        return self._memo[uid]


@dataclass(frozen=True)
class SeamEdge:
    """One writer a formal read resolves to across the seam between two
    iterations of a layer loop: `producer`'s `port` is what input `index`
    of the reader reads, at the iterations `pairs` names — each
    `(reader iteration, source occurrence)` one seam."""

    producer: int
    port: str
    pairs: FrozenSet[Tuple[Optional[int], Occurrence]]


@dataclass(frozen=True)
class Seams:
    """Where a formal of a layer template is written by the previous
    iteration, read off the walk's references (`Analysis.reference_table`):
    `(reader uid, input) -> SeamEdge` per writer, the iterations it writes
    for in its edge.

    The walk evaluates a template once and binds, beside each value an
    input reads, the occurrences that produced it at every iteration
    (`occurrence.Reference`). The residual a layer reads is the previous
    layer's exit (`Program.threads_loop`), so the norm that opens layer i+1
    and the add that closed layer i are one producer–consumer edge at
    iterations 1..c−1 of a segment, the previous segment's exit at its
    first, and the stack's last exit for the model-level reader after it.
    Nothing here is new: the edge is the walk's, and the matcher follows it
    as it follows a sibling edge (`_dag_bindings`).

    A seam whose two occurrences run on different pipeline stages is a cut,
    not a seam: the value is sent there, and the add runs where it was
    made. The stage is the walk's placement (`Rule.stage_of`), read here
    and asked of no caller."""

    edges: Mapping[Tuple[int, int], Tuple[SeamEdge, ...]]
    #: The walked node whose iterations a reader's runs are: the reader
    #: itself, or the reader a grafted cast reads for.
    readers: Mapping[Tuple[int, int], int] = field(default_factory=dict)
    #: `(writing leaf, iteration) -> the nodes reading that run inside its
    #: own iteration`, as the walk bound them (`Reference.local`).
    in_layer: Mapping[Tuple[int, Any], FrozenSet[int]] = field(
        default_factory=dict)

    def read_in_layer(self, uid: int, iteration: Any) -> FrozenSet[int]:
        """The nodes that read run `iteration` of `uid` inside its own
        iteration."""
        return self.in_layer.get((uid, iteration), frozenset())

    @classmethod
    def of(cls, program: Any, analysis: Any, nodes: Sequence[Any],
           graft: Any = None) -> "Seams":
        # The walk's bound references; a rewrite with no walk behind it (a
        # test's bare program) has no iteration to cross into.
        table = None if analysis is None else analysis.reference_table
        if table is None:
            return cls({}, {}, {})
        # Which pipeline stage each occurrence runs on is the walk's
        # placement (`Rule.stage_of`): a seam a cut separates is a hop, so
        # no caller can make one by forgetting to say where the cut is.
        stage_of = analysis.rule.stage_of(analysis)
        present = {node.uid for node in nodes}

        out: Dict[Tuple[int, int], Tuple[SeamEdge, ...]] = {}
        readers: Dict[Tuple[int, int], int] = {}
        for node in nodes:
            for index, ref in enumerate(node.inputs_obj or ()):
                if ref.src is not None:
                    continue
                key = (node.uid, index)
                if not table.covers(node.uid):
                    # A grafted cast reads what its first reader's input
                    # was bound to (`CastGraft.rewrite`).
                    key = (graft.inputs.get(key) if graft is not None
                           else None)
                    if key is None or not table.covers(key[0]):
                        continue
                if key[1] >= table.arity[key[0]]:
                    continue
                reference = table.at(*key)
                found: Dict[Tuple[int, str], set] = {}
                for it in sorted(program.occurrence_layers(key[0]),
                                 key=lambda i: (i is None, i)):
                    for source in reference.at(it):
                        if not program.threads_loop(key[0], source):
                            continue
                        written, port = source.occurrence, source.port
                        writer = program.by_uid(written.uid)
                        if writer is not None and writer.children:
                            # A layer whose exit is a component's port:
                            # the leaf that writes it (`_writing_leaves`).
                            leaf, leaf_port = program.output_leaf(
                                writer, port or "out")
                            written = Occurrence(leaf.uid, written.iteration)
                            port = leaf_port or "out"
                        if written.uid not in present:
                            continue
                        if stage_of(program, source.occurrence.uid,
                                    written.iteration) \
                                != stage_of(program, node.uid, it):
                            continue
                        found.setdefault((written.uid, port),
                                         set()).add((it, written))
                if found:
                    readers[(node.uid, index)] = key[0]
                    out[(node.uid, index)] = tuple(
                        SeamEdge(uid, port, frozenset(pairs))
                        for (uid, port), pairs in sorted(found.items()))
        # Which runs a node's own layer reads, as the walk bound them: a
        # local reference is a read of the same iteration.
        in_layer: Dict[Tuple[int, Any], set] = {}
        for node in nodes:
            if not table.covers(node.uid):
                # A node a pass grafted reads what its edge names in its own
                # iteration (it stands in its producer's scope), at the runs
                # it declares (`Program.occurrence_layers`: a pipeline hop
                # only at its cut).
                for ref in node.inputs_obj or ():
                    if ref.src is None:
                        continue
                    leaf = (program.output_leaf(ref.src, ref.port or "out")[0]
                            if ref.src.children else ref.src)
                    for it in program.occurrence_layers(node.uid):
                        in_layer.setdefault((leaf.uid, it), set()).add(
                            node.uid)
                continue
            for index in range(table.arity[node.uid]):
                reference = table.at(node.uid, index)
                if not reference.local:
                    continue
                for it in program.occurrence_layers(node.uid):
                    for source in reference.at(it):
                        written = source.occurrence.uid
                        writer = program.by_uid(written)
                        if writer is not None and writer.children:
                            written = program.output_leaf(
                                writer, source.port or "out")[0].uid
                        in_layer.setdefault((written, it), set()).add(
                            node.uid)
        return cls(out, readers, {key: frozenset(found)
                                  for key, found in in_layer.items()})

    def consumers(self) -> Dict[int, List[Tuple[int, str]]]:
        """`_consumers`' entries for the reads across a seam: a writer of
        the previous iteration is read by the formal's readers."""
        out: Dict[int, List[Tuple[int, str]]] = {}
        for (reader, _index), found in self.edges.items():
            for edge in found:
                out.setdefault(edge.producer, []).append((reader, edge.port))
        return out


@dataclass
class _Binding:
    """One partial binding of a pattern: the graph node per pattern node,
    the external operands, which side of a seam each pattern node is on
    (0 the reading iteration, 1 the previous one), and the seams crossed —
    the formal reads bound to a previous-iteration writer and the
    iteration pairs every one of them agrees on."""

    bound: Dict[str, Any] = field(default_factory=dict)
    external: Dict[str, Any] = field(default_factory=dict)
    side: Dict[str, int] = field(default_factory=dict)
    crossed: FrozenSet[Tuple[int, int]] = frozenset()
    pairs: Optional[FrozenSet[Tuple[Optional[int], Occurrence]]] = None

    def copy(self) -> "_Binding":
        return _Binding(dict(self.bound), dict(self.external),
                        dict(self.side), self.crossed, self.pairs)

    def across(self, key: Tuple[int, int], edge: SeamEdge
               ) -> Optional["_Binding"]:
        """This binding with the seam `key` bound to `edge`'s writer: the
        iterations it spans are those every crossed seam spans."""
        pairs = (edge.pairs if self.pairs is None
                 else self.pairs & edge.pairs)
        if not pairs:
            return None
        out = self.copy()
        out.crossed = self.crossed | {key}
        out.pairs = pairs
        return out

    @property
    def previous(self) -> FrozenSet[int]:
        """The members of the previous iteration."""
        return frozenset(node.uid for name, node in self.bound.items()
                         if self.side.get(name, 0))


def _dag_matches(pattern: Any, nodes: Sequence[Any],
                 scopes: Mapping[int, Any],
                 operands: OperandRoles,
                 edges: Optional[Mapping[int, Tuple]] = None,
                 views: Optional[ViewChains] = None,
                 ) -> List[Dict[str, Any]]:
    """`_dag_bindings`' member maps, within one iteration (no seams)."""
    return [binding.bound for binding in _dag_bindings(
        pattern, nodes, scopes, operands, edges, views)]


def _dag_bindings(pattern: Any, nodes: Sequence[Any],
                  scopes: Mapping[int, Any],
                  operands: OperandRoles,
                  edges: Optional[Mapping[int, Tuple]] = None,
                  views: Optional[ViewChains] = None,
                  seams: Optional[Seams] = None,
                  ) -> List[_Binding]:
    """Every occurrence of a declared node-and-edge subgraph.

    Anchored on the pattern's SINKS — the nodes no other pattern node
    consumes — and matched backwards along declared inputs, so each pattern
    node's producers are stated and a candidate either has them or does not.
    One sink is a chain or a fan-in; several sinks (`fused_qk_rope`'s q and k
    ropes) enumerate candidates per sink under one shared binding, which is
    where the external table earns its keep: `external.q_pe` and
    `external.k_pe` must resolve to DIFFERENT producers or the two slots
    would bind one rope twice.

    Every binding is confined to one scope — `(template, parent)` per member,
    all equal — because a fusion is one kernel launch and its operands live
    in one component instance. `_scopes` records the false positives this
    exists to prevent.

    A pattern node that declares `feeds` binds only a node whose output
    becomes that operand (`OperandRoles`), which is what fixes the two ropes
    to `q_rope` and `k_rope` rather than to the order they were met in —
    and not one whose output also becomes another operand some other node of
    the pattern `feeds`: a tensor that turns into both q and k tells the two
    slots apart no better than its op does. One member set found under two
    different bindings is refused: its port names would depend on that order.

    A pattern node declared `optional` may be absent: its reader then binds
    straight to the node's one internal input, and every non-optional node
    must still be bound. One plan thereby covers a kernel whose naive form
    has a step some models fold away (the indexer's head weights). An
    optional SINK that is absent ends the chain one step earlier: its
    internal input is anchored instead (a router with no routed-scaling
    multiply).

    An edge into a node whose quantised write commutes with the views it
    reads through (`views`, `ViewChains`) also binds the producer at the
    other end of those views: quantising after them is quantising before
    them. The views themselves are no pattern node; the rewrite takes them
    into the launch (`ViewChains.through`).

    An input that reads a formal of a layer template the previous iteration
    wrote (`Seams`) binds that writer as a producer, as a sibling edge
    would: the writer and what the pattern binds above it are the previous
    iteration's side of the match, confined to one scope of their own, and
    the binding records the iteration pairs it spans. One seam deep — the
    previous side follows no further seam.
    """
    declared = dict(pattern.nodes or {})
    if len(declared) < 2:
        # A one-node pattern is a rename, not a fusion: it matches every node
        # of that op anywhere in the model. `ParallelPatternForm` rejects the
        # same thing with `min_count >= 2` and calls it degenerate; the chain
        # form had no such check, so `fused_moe` — declared as a single
        # `linear` — matched seven plain projections in a DENSE Llama. The
        # three plans that do this are reachable from no framework binding,
        # which is the other half of the evidence that they are unfinished.
        return []
    consumed = {ref.split(".", 1)[0]
                for spec in declared.values()
                for ref in spec.inputs
                if not ref.startswith("external.")}
    sinks = sorted(name for name in declared if name not in consumed)
    required = {name for name, spec in declared.items() if not spec.optional}
    # The operands the pattern's `feeds` tell apart. A node whose output
    # becomes two of them (DeepSeek-V4's MTP `enorm` / `hnorm`, which feed the
    # whole block after them) is neither one, so it binds neither slot.
    fed = frozenset(spec.feeds for spec in declared.values() if spec.feeds)
    if not sinks:
        return []                       # cyclic: not a subgraph
    by_uid = {n.uid: n for n in nodes}
    crossing = {} if seams is None else seams.edges

    def _bind(pattern_name: str, node: Any, state: _Binding,
              side: int) -> List[_Binding]:
        spec = declared[pattern_name]
        if not spec.accepts_op(node.op):
            return []
        if spec.feeds is not None and (
                spec.feeds not in operands.of_node(node.uid)
                or operands.of_node(node.uid) & (fed - {spec.feeds})):
            return []
        if state.bound.get(pattern_name) not in (None, node):
            return []
        if state.side.get(pattern_name, side) != side:
            return []                   # one node, one iteration
        if any(n.uid == node.uid for k, n in state.bound.items()
               if k != pattern_name):
            return []                   # one graph node per pattern slot
        state = state.copy()
        state.bound[pattern_name] = node
        state.side[pattern_name] = side
        refs = list(spec.inputs)
        optional = list(spec.optional_inputs)
        actual = _producer_ids(node, edges)
        if len(actual) < len(refs) or len(actual) > len(refs) + len(optional):
            return []
        states = [state]
        for index, (ref, (producer, port)) in enumerate(zip(
                refs + optional[:len(actual) - len(refs)], actual,
                strict=True)):
            head, _, tail = ref.partition(".")
            following: List[_Binding] = []
            for current in states:
                if head == "external":
                    if current.external.get(tail, producer) != producer:
                        continue        # two operands claim one producer
                    current = current.copy()
                    current.external[tail] = producer
                    following.append(current)
                    continue
                found = _bind_edge(head, tail, producer, port, current, side)
                if not found:
                    root = None if views is None else views.root(node, index)
                    if root is not None:
                        found = _bind_edge(head, tail, root[0], root[1],
                                           current, side)
                if not found and side == 0:
                    for edge in crossing.get((node.uid, index), ()):
                        for across in _bind_edge(head, tail, edge.producer,
                                                 edge.port, current, 1):
                            joined = across.across((node.uid, index), edge)
                            if joined is not None:
                                found.append(joined)
                following.extend(found)
            states = following
            if not states:
                return []
        return states

    def _bind_edge(head: str, tail: str, producer: Any, port: str,
                   state: _Binding, side: int) -> List[_Binding]:
        """Bind the pattern edge `head.tail` to the graph edge
        `(producer, port)`. A node declared `optional` that the producer
        is not is read through: the edge binds to the node's own one input
        instead, and the node is absent from the match. That input may be
        the pattern's own external operand — a step the kernel folds in
        that stands on the launch's input (a router's gate temperature on
        the logits), absent where the checkpoint does not spell it — and
        the operand then binds to this producer."""
        if not isinstance(producer, int) or producer not in by_uid:
            return []
        if not tail or port == tail:
            found = _bind(head, by_uid[producer], state, side)
            if found:
                return found
        spec = declared[head]
        if not spec.optional or head in state.bound:
            return []
        through = spec.read_through()
        if through is None:
            return []
        through_head, _, through_tail = through.partition(".")
        if through_head == "external":
            if state.external.get(through_tail, producer) != producer:
                return []               # two operands claim one producer
            current = state.copy()
            current.external[through_tail] = producer
            return [current]
        return _bind_edge(through_head, through_tail, producer, port,
                          state, side)

    def _extend(index: int, state: _Binding) -> List[_Binding]:
        if index == len(sinks):
            if not required <= set(state.bound):
                return []
            # One launch, one scope per iteration it spans — the views a
            # member was read through included (`ViewChains.through`): each
            # is placed by its own outermost node, the one whose parent is
            # not itself a view of the chain.
            chain = (frozenset() if views is None
                     else views.through(tuple(state.bound.values())))
            for side in set(state.side.values()):
                placed = {scopes.get(n.uid)
                          for name, n in state.bound.items()
                          if state.side[name] == side}
                if side == 0:
                    placed |= {
                        scopes.get(uid) for uid in chain
                        if scopes.get(uid, (None, None))[1] not in chain}
                if len(placed) > 1:
                    return []
            return [state]
        return _anchor(sinks[index], index, state)

    def _anchor(name: str, index: int, state: _Binding) -> List[_Binding]:
        """Bind the pattern sink `name` to every candidate, then the next
        sink. An optional sink may also be absent: the chain then ends at
        its one internal input, anchored in its place — after the
        candidates WITH the sink, so the larger match is met first."""
        if name in state.bound:
            return _extend(index + 1, state)
        spec = declared[name]
        out: List[_Binding] = []
        taken = {n.uid for n in state.bound.values()}
        for node in nodes:
            if node.uid in taken or not spec.accepts_op(node.op):
                continue
            for found in _bind(name, node, state, 0):
                out.extend(_extend(index + 1, found))
        if spec.optional:
            (through,) = spec.internal_inputs()
            out.extend(_anchor(through.rsplit(".", 1)[0], index, state))
        return out

    matches = _extend(0, _Binding())
    # The members are what a fusion IS, so one member set is one match —
    # and it must bind one way. Two bindings of the same set put the port
    # names on whichever the enumeration met first.
    seen: Dict[FrozenSet[int], Dict[str, int]] = {}
    unique: List[_Binding] = []
    for match in matches:
        bound = match.bound
        key = frozenset(n.uid for n in bound.values())
        binding = {name: n.uid for name, n in bound.items()}
        if key not in seen:
            seen[key] = binding
            unique.append(match)
        elif seen[key] != binding:
            names = {n.uid: n.name for n in bound.values()}
            raise ValueError(
                f"fusion pattern binds {sorted(names.values())!r} two ways "
                f"({ {k: names[u] for k, u in seen[key].items()} !r} and "
                f"{ {k: names[u] for k, u in binding.items()} !r}); its "
                "ports would be named by match order. Declare `feeds:` on "
                "the pattern nodes, or the operand roles downstream.")
    return unique


def _parallel_matches(form: Any, nodes: Sequence[Any],
                      scopes: Mapping[int, Any],
                      program: Any = None,
                      edges: Optional[Mapping[int, Tuple]] = None
                      ) -> List[Dict[str, Any]]:
    """Maximal sibling sets: same op, same producers, one scope, none reading
    another.

    One match per set rather than one per permutation — the siblings are
    interchangeable, and enumerating orders would be `N!` matches describing the
    same fusion. The grouping key carries the alias IDENTITY where a producer
    is the enclosing scope's input (`q_a_proj` and `kv_a_proj` both read the
    block's `x` — same tensor, no uid), and the PARENT MODULE, because the
    kernels these schemes stand for are module wrappers: vLLM's
    `MergedColumnParallelLinear` / `QKVParallelLinear` merge the linears of
    one parent module only — the matcher keys its groups the same way and
    records why (the DSv3 router gate and the shared-expert projections all
    read `post_attention_layernorm`, and are separate kernel launches). The
    parent module is finer than the component tree: DSA's `indexer.wq_b` and
    `q_b_proj` share a component and a producer, and belong to different
    wrappers.
    """
    arity = len(form.shared_inputs or ())
    groups: Dict[Tuple, List[Any]] = {}
    for node in nodes:
        if not form.accepts_op(node.op or ""):
            continue
        ids = tuple(_producer_ids(node, edges))
        # The form declares its arity: a node with a different input count is
        # not an instance of the shape, whatever its op. Without this, two
        # sequential linears written with NO declared inputs (MiMo's
        # `visual.merger.mlp.0/.2`) carry the same empty producer key and
        # pass for siblings.
        if len(ids) != arity:
            continue
        # The parent is the one `_dag_matches` confines a binding to —
        # template and parent node, read off the tree — so both pattern
        # forms give one answer to "which module is this node in". A dotted
        # local name (`indexer.wq_b`) declares a submodule of that parent
        # with no node of its own; its leading segments say which.
        scope = scopes[node.uid]
        submodule = node.name.rpartition(".")[0]
        # An alias names the tensor its own call hands it: two calls of one
        # template (Step-3.7's tower on the global views and on the crops)
        # read two tensors under the one name.
        invocation = program.block_of(node.uid) if program is not None else None
        key = (scope, submodule, invocation, ids)
        groups.setdefault(key, []).append(node)

    out: List[Dict[str, Any]] = []
    lower = form.min_count
    upper = form.max_count
    for siblings in groups.values():
        uids = {n.uid for n in siblings}
        if any(p in uids for n in siblings for p, _ in _producers(n, edges)):
            continue                    # one reads another: not siblings
        if len(siblings) < lower:
            continue
        if upper is not None and len(siblings) > upper:
            continue
        ordered = sorted(siblings, key=lambda n: (n.name or "", n.uid))
        out.append({f"parallel.{i}": n for i, n in enumerate(ordered)})
    return out


# -- when a framework runs a kernel ---------------------------------------

@dataclass(frozen=True)
class GraphCondition:
    """The part of a framework's dispatch condition the graph answers.

    A framework binding entry may say WHEN its kernel runs (`when:` in
    `framework_bindings/*.yaml`). The deployment answers some of it — a
    server flag, the activation dtype, the device — and the binding settles
    those before the rewrite. What is left is a fact of the model or of the
    matched subgraph, and this object asks it:

    * `architectures` — the framework's own dispatch key, the model class
      (`Program.architectures`): a kernel only one model class calls.
    * `excluded_architectures` — the same key, the other way round: a kernel
      every model class reaches through a shared layer except the ones that
      replace that layer with their own launches (sglang's DeepSeek-V4 class
      rotates q and the latent KV in two separate kernels and never calls
      the shared rotary embedding on the pair).
    * `head_dim` — the `dim` every matched rotation rotates over.
    * `rotation_covers_head` — each matched rotation's `dim` equals the `dim`
      of the matched node it reads: the rotation spans the whole head the
      norm before it normalised (no `rotary_dim` is passed).
    * `dims` — `(name, values)` pairs: every matched node whose walk bound
      the dim `name` (`Analysis.node_dims`, its own or its frame's) ran it at
      one of `values`, and at least one matched node bound it — a dim no
      member carries answers neither way. The fact a framework's kernel
      choice reads off the layer: a convolution's `dilation` (sglang's
      causal-conv kernel takes none), a DeepSeek-V4 attention layer's
      `compress_ratio` (sglang builds `C4Indexer` for 4 / 128 and
      `DeepseekV41Indexer` for 1 / 2), the step's verify width `n_verify`
      (1 is a step that does not speculate, `Workload.walked_as_number`; a
      kernel compiled for 2-8 verified tokens lists them). A width the walk
      left symbolic is in no list, so the entry does not bind.
    * `phase` — the phase of the walk being rewritten (`Rule.phase`:
      prefill / decode / encode, `dims.PHASES`): a framework that runs one
      subgraph as a chunked kernel in prefill and as one recurrent launch
      in decode binds each launch to its phase. A walk whose rule names no
      phase (the identity rule) admits none.
    * `world_local` — `(world, dim, local_in)`: in the frame of every matched
      node that binds `dim` (at least one must), the walk's degree of rank
      group `world` (`Rule.world_degree`) divides its value, and — when
      `local_in` is not empty — the per-rank value (value / degree) is one
      of `local_in`. Empty `local_in` asks divisibility alone: sglang's
      grouped gated norm runs only while each rank holds whole groups
      (n_groups % tp_size == 0). A set asks the per-rank count a kernel was
      built for: vLLM's Kimi-K3 whole-mixer decode at 12 / 24 / 48 / 96
      local heads. `dim` is a symbol `world` may split
      (`dims.capabilities_of`).
    * `roles` — `{<pattern node>: [<role>, ...]}`: the role the spec declares
      (`Analysis.roles`, `roles.yaml`) for the node this match bound to that
      slot of the pattern is one of these. The same subgraph spelled by two
      modules a framework runs differently is told apart by it, and keyed to
      the node whose role carries the difference rather than to "some member":
      DeepSeek-V4.1 writes its indexer key (`k_norm`, `attention_indexer`) and
      its latent KV (`kv_norm`, `attention` / `attention_swa`) with the same
      norm → rope → re-join, and only those two norms differ — every slice and
      rotation around them is `attention` in both. A question about one named
      node also reads both ways, so each framework states positively which
      chain its kernel runs instead of one of them excluding the other.
    * `readers_eat` — an activation format (`arch/activation_formats/`): the
      tensor the match publishes as its primary output (`io_mapping.output`,
      node and port) has readers, and every one of them eats that format —
      `traffic.reader_format`, the same representations `traffic.derive`
      prices the reader's conversion from. A launch that writes its output
      already quantised, and that the framework picks only when every
      consumer takes that format, says so here (vLLM V4.1's
      `fused_q_kv_rmsnorm_quant`, deepseek_v41/common/ops/query_quant.py).

    * `block_reads_stream` — whether the BLOCK the match sits in declares a
      cross-layer stream read (`BlockGraph.reads`, carried as
      `BlockMeta.stream_reads`): the layer takes a value an earlier layer
      produced, outside its own inputs. It is what tells a SHIFTED sublayer
      schedule from a same-sublayer one where both spell the same arithmetic:
      DeepSeek-V4.1's hyper-connection block collapses the copies with the
      pre-mix the PREVIOUS sublayer published and declares `reads:
      [hc_pre_mix]`, while DeepSeek-V4's and GLM-5.3-Flash's blocks declare
      none and compute theirs in place. The frameworks dispatch on a config
      flag there (`hc_pre_from_prev_sublayer`), which this is the graph's
      statement of; a match with no block occurrence answers neither way.

    * `routed` — whether the match sits inside the dispatch zone, between a
      `permute` and its `combine` (`Program.node_is_routed`): every member
      routed (True) or none (False); a match that straddles the zone answers
      neither. A framework runs the routed experts inside its MoE layer and
      everything else through the module that builds it, so the same
      activation-and-mul is two different launches at the two sites —
      sglang's Kimi-K3 runs `SituAndMul` in its dense and shared MLPs, while
      its MoE runner applies the routed experts' SiTU itself. The fact is the
      graph's topology, not a role: `ffn_moe` names the MoE wrapper, which
      holds a shared expert too wherever the spec does not refine it.

    Empty fields ask nothing. `PROGRAM_FIELDS` are the ones the whole walk
    answers (`admits_program`); every other field asks a match.
    """

    #: The fields `admits_program` asks — the model class and the walk's
    #: phase — as opposed to a match's facts (`admits_match`). The one list
    #: of them: a caller with no match to ask (a primitive row's provider
    #: dispatch, `FrameworkBinding.dispatch_for`) may state only these.
    PROGRAM_FIELDS: ClassVar[Tuple[str, ...]] = (
        "architectures", "excluded_architectures", "phase")

    architectures: Tuple[str, ...] = ()
    excluded_architectures: Tuple[str, ...] = ()
    head_dim: Tuple[int, ...] = ()
    rotation_covers_head: Optional[bool] = None
    dims: Tuple[Tuple[str, Tuple[int, ...]], ...] = ()
    phase: Tuple[str, ...] = ()
    world_local: Optional[Tuple[str, str, Tuple[int, ...]]] = None
    readers_eat: Optional[str] = None
    roles: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    routed: Optional[bool] = None
    block_reads_stream: Optional[bool] = None

    def admits_program(self, program: Any, analysis: Any) -> bool:
        """The questions the whole walk answers: the model class, and the
        phase the walk ran (`analysis.rule.phase`). Asked of every walk the
        rewrite is applied to — a prefill walk and a decode walk are two
        walks, each rewritten on its own (`step_eval.Cell` per phase)."""
        classes = set(program.architectures)
        return ((not self.architectures
                 or bool(set(self.architectures) & classes))
                and not set(self.excluded_architectures) & classes
                and (not self.phase or analysis.rule.phase in self.phase))

    def asks_program(self) -> bool:
        """Whether any of `PROGRAM_FIELDS` is stated."""
        return any(vars(self)[name] for name in self.PROGRAM_FIELDS)

    def admits_match(self, bound: Mapping[str, Any], analysis: Any,
                     output: Optional[Tuple[Any, str]] = None) -> bool:
        """`analysis` is the walk being rewritten. Its `node_dims` give the
        `dim` each node ran at, however the spec bound it — `dim: X`,
        `dims: {dim: X}`, or the enclosing component's frame — read alike
        for the rotation and for the node it reads; its rule gives the
        world degrees (`world_local`). A node grafted after the walk (an
        activation cast) ran at no spec dim and is not asked. `output` is the
        `(member, port)` the match publishes as its primary output
        (`readers_eat`)."""
        node_dims = analysis.node_dims
        members = {node.uid: node for node in bound.values()}
        if self.block_reads_stream is not None:
            blocks = {analysis.program.block_of(uid) for uid in members}
            blocks.discard(None)
            if not blocks:
                return False            # no block: answers neither way
            reads = {bool(analysis.program.blocks[b].meta.stream_reads)
                     for b in blocks}
            if reads != {self.block_reads_stream}:
                return False
        if self.routed is not None and {
                analysis.program.node_is_routed(node)
                for node in members.values()} != {self.routed}:
            return False
        if self.readers_eat is not None and not self._readers_eat(
                output, members, analysis):
            return False
        for name, allowed in self.roles:
            asked = bound.get(name)
            if asked is None:
                raise ValueError(
                    f"a binding asks the role of pattern node {name!r}, "
                    f"which this match does not bind")
            if analysis.roles.values.get(asked.uid) not in allowed:
                return False
        if self.world_local is not None:
            world, dim, local_in = self.world_local
            degree = analysis.rule.world_degree(world)
            values = [int(value_of(node_dims[node][dim]))
                      for node in members.values()
                      if dim in node_dims.get(node, ())]
            if not values or any(v % degree for v in values):
                return False
            if local_in and any(v // degree not in local_in for v in values):
                return False
        for name, values in self.dims:
            ran = [node_dims[node][name] for node in members.values()
                   if name in node_dims.get(node, ())]
            if not ran or any(value not in values for value in ran):
                return False
        for node in members.values():
            spec = OP_REGISTRY.get(node.op)
            if spec is None or not spec.rotary_position_axes:
                continue
            width = node_dims[node].get("dim")
            if self.head_dim and width not in self.head_dim:
                return False
            if self.rotation_covers_head is not None:
                read = [members[ref.src.uid] for ref in node.inputs_obj
                        if ref.src is not None and ref.src.uid in members]
                widths = [node_dims[r].get("dim") for r in read]
                if width is None or not widths or None in widths:
                    return False        # unknown answers neither way
                if all(w == width for w in widths) != self.rotation_covers_head:
                    return False
        return True


    def _readers_eat(self, output: Optional[Tuple[Any, str]],
                     members: Mapping[int, Any], analysis: Any) -> bool:
        """Every walked kernel outside the match that reads the output
        tensor (that node, that port) eats the `readers_eat` format; False
        with no reader (unknown answers neither way)."""
        if output is None:
            return False
        node, port = output
        wanted = ACTIVATION_FORMATS[self.readers_eat]
        edges = Edges(analysis)
        readers = [
            reader for reader in analysis.program.walked_nodes()
            if reader.op and not reader.children
            and reader.uid not in members and edges.covers(reader)
            and any(src is not None and src.uid == node.uid
                    and read_port == port
                    for index in range(len(reader.inputs_obj or ()))
                    for src, read_port in edges.reads(reader, index))]
        return bool(readers) and all(
            reader_format(analysis, reader) == wanted for reader in readers)


def _tree_high_water(program: Any) -> int:
    """The next uid the program's tree would hand out. `SpecIRTree.make_node`
    allocates from this counter, so a fused uid at or above it cannot collide
    with a node the tree holds — which is what lets `materialize_tree` mint
    the tree's copy of a fused node under the uid the rewrite already chose.
    The walked nodes alone are not enough: the tree also holds nodes this
    schedule never walks."""
    return int(program.ir.next_uid)


def _exposed_ports(pattern: Any, bound: Mapping[str, Any]
                   ) -> Dict[int, FrozenSet[str]]:
    """Member uid → the ports of ITS OWN output that the fused node
    re-exports, read off `io_mapping`'s targets (`post: split.post` exposes
    `split`'s `post`). The fused node's port NAME is the mapping's key; this
    is the other half, the origin port a reader outside the match asks for."""
    out: Dict[int, set] = {}
    for target in pattern.io_mapping.values():
        head, _, origin = str(target).partition(".")
        if head == "external":
            continue
        name = pattern.port_node(head, bound)
        if name is None:
            continue
        out.setdefault(bound[name].uid, set()).add(origin or "out")
    return {uid: frozenset(ports) for uid, ports in out.items()}


# -- the transform --------------------------------------------------------

@dataclass(frozen=True)
class Fusion:
    """One fusion scheme, as a `GraphTransform`.

    `closure` is not optional rigour: a match may only absorb a node whose every
    consumer is inside the match or is exposed as a fused output port. The
    naive `add` in `fused_add_rmsnorm` has TWO consumers — the `rmsnorm` beside
    it and the NEXT block's add — and the scheme exposes it as `residual_out`
    for exactly that reason. Absorbing it silently would cut the residual.
    """

    scheme_name: str
    scheme: Any
    #: When the framework runs the kernel, as far as the graph answers it.
    #: None: whenever the pattern matches.
    condition: Optional[GraphCondition] = None
    #: The widest boundary this framework's launch spans (`crosses:` in the
    #: binding entry; `BoundaryView`). None: one component instance, as
    #: every plan matched before boundary-crossing kernels existed.
    crosses: Optional[str] = None

    @classmethod
    def named(cls, scheme_name: str, schemes: Mapping[str, Any],
              condition: Optional[GraphCondition] = None,
              crosses: Optional[str] = None) -> "Fusion":
        # The scheme registry lives in `backend/inference/fusion_plans`;
        # this package must not import it, so the caller passes it in.
        scheme = schemes.get(scheme_name)
        if scheme is None:
            raise KeyError(
                f"unknown fusion scheme {scheme_name!r}; "
                f"declared schemes: {sorted(schemes)}")
        if scheme.output_dtype and scheme_name in OP_REGISTRY:
            # A registered op states its output width once, on the op
            # (`OpSpec.output_dtype`), and the fused node
            # carries that registration; a plan restating it is a second
            # statement of one fact.
            raise ValueError(
                f"fusion scheme {scheme_name!r} states `output_dtype` for a "
                f"registered op; its width is the op's declaration")
        if crosses is not None and crosses not in CROSSES:
            raise ValueError(
                f"fusion scheme {scheme_name!r}: `crosses: {crosses!r}` is "
                f"not a boundary; declared: {list(CROSSES)}")
        return cls(scheme_name, scheme, condition, crosses)

    def rewrite(self, program: Any, analysis: Any,
                local_shapes: Any = None,
                graft: Any = None) -> Tuple[Any, Rewrite]:
        """`analysis` is the walk `program` was evaluated in; a graph
        condition reads the dims each matched node ran at from it.
        `local_shapes` is its per-rank shape layer, which a distributed
        walk's caller materialises (`ViewChains.of`). `graft` is the
        rewrite that made `program` (before any fusion) from the walked
        graph — the serving graph's casts on edges no node writes
        (`inference.CastGraft.rewrite`) — None for none."""
        if (self.condition is not None
                and not self.condition.admits_program(program, analysis)):
            return program, Rewrite()
        pattern = self.scheme.naive_pattern
        # The kernels the walk ran: a stack this program does not walk (an
        # encoder, in the language model's step) has nothing to fuse.
        nodes = [n for n in program.walked_nodes()
                 if n.op and not n.children]
        # A boundary-crossing entry matches over the view instead
        # (`BoundaryView`): wrapper producers resolved to the leaf that
        # writes the port, and the block occurrence as the scope.
        view = (None if self.crosses is None
                else BoundaryView.of(program, nodes))
        if view is not None:
            nodes, scopes, edges = view.leaves, view.scopes, view.edges
        else:
            scopes, edges = _scopes(program), None
        is_parallel = pattern.parallel is not None
        views = (None if is_parallel
                 else ViewChains.of(program, analysis, local_shapes, graft))
        # The seams a binding may cross: a formal a layer reads, written by
        # the iteration before it (`Seams`). A parallel form groups siblings
        # of one iteration and crosses none.
        seams = (Seams({}, {}, {}) if is_parallel
                 else Seams.of(program, analysis, nodes, graft))
        matches = ([_Binding(bound=bound) for bound in _parallel_matches(
                       pattern.parallel, nodes, scopes, program, edges)]
                   if is_parallel
                   else _dag_bindings(pattern, nodes, scopes,
                                      OperandRoles.of(program), edges, views,
                                      seams))
        if self.condition is not None:
            head, _, port = str(pattern.io_mapping.get("output", "")
                                ).partition(".")
            matches = [match for match in matches
                       if self.condition.admits_match(
                           match.bound, analysis,
                           output=_walked_output(pattern, match.bound, head,
                                                 port))]
        if not matches:
            return program, Rewrite()

        consumers = _consumers(nodes, edges)
        for producer, readers in seams.consumers().items():
            consumers.setdefault(producer, []).extend(readers)
        origins: Dict[int, Tuple[int, ...]] = {}
        fused: List[FusedNode] = []
        absorbed: set = set()
        inputs: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # The iterations each node still runs at on its own (`_Runs`), and
        # each launch's (`Rewrite.taken`, `Rewrite.homes`).
        runs = _Runs.of(program)
        taken: Dict[int, Mapping[int, Mapping[Any, Any]]] = {}
        homes: Dict[int, int] = {}
        # A fresh uid past every node of the graph, walked or not: an encode
        # walk's nodes are not the graph's last. It must also clear the
        # TREE's high-water mark: `materialize_tree` asks the tree for each
        # replacement's uid and checks it against the one the rewrite chose,
        # and a tree can hold uids this program does not show — a decode
        # walk's spliced variants leave gaps that `program.nodes()` never
        # reaches. Taking the larger of the two keeps one authority.
        next_uid = max(
            max((n.uid for n in program.nodes()), default=0) + 1,
            _tree_high_water(program))

        for match in matches:
            bound = match.bound
            # The views a quantised member was read through join the launch
            # (`ViewChains`): they are members, priced at nothing as every
            # view is, and the launch writes the tensor they re-index.
            chain = (frozenset() if views is None
                     else views.through(tuple(bound.values())))
            # A view-only wrapper goes with its leaves: every reader of
            # it is in the chain (`TensorReaders.only`), so leaving it
            # would leave an empty component reading the launch.
            inside = {n.uid for n in bound.values()} | set(chain)
            if inside & absorbed:
                continue                # overlapping matches: first wins
            # A parallel fusion re-exports every sibling's output
            # positionally (`output_0..N`, consumers rewired by sort order —
            # the scheme's own contract), so an outside consumer is the
            # NORMAL case, not a cut edge; closure applies to pattern forms
            # whose absorbed intermediates genuinely disappear.
            if not is_parallel and not self._closed(
                    bound, inside, consumers,
                    self._exposed_names(pattern, bound),
                    ports=(_exposed_ports(pattern, bound)
                           if view is not None else None)):
                continue
            took = runs.take(match, tuple(sorted(inside)), seams)
            if took is None:
                continue
            # The home first: every layer that takes a launch's value from
            # one origin (`_first`: its frame, its definitions) takes it
            # from the origin the launch runs with, as `block_of` does.
            members = (took.home, *(uid for uid in sorted(inside)
                                    if uid != took.home))
            member_nodes = tuple(bound.values())
            sink, _exposed = _boundary_outputs(member_nodes, consumers)
            ordered_members = tuple(sorted(member_nodes, key=lambda n: n.uid))
            # A member's SECONDARY output the plan publishes (`post:
            # split.post`), keyed by the member and its port: the main
            # output keeps its place in `output_ports`.
            secondary_ports: Dict[Tuple[int, str], str] = {}
            if is_parallel:
                output_ports = {
                    member.uid: f"output_{index}"
                    for index, member in enumerate(sorted(
                        ordered_members,
                        key=lambda n: (
                            n.display_path or n.name or "",
                            n.uid,
                        ),
                    ))
                }
            else:
                output_ports = {}
                for port_name, target in pattern.io_mapping.items():
                    pattern_name, _, origin_port = str(target).partition(".")
                    if pattern_name == "external":
                        continue
                    pattern_name = pattern.port_node(pattern_name, bound)
                    if pattern_name is None:
                        continue
                    published = "" if port_name == "output" else str(port_name)
                    if origin_port in ("", "out"):
                        output_ports[bound[pattern_name].uid] = published
                    else:
                        secondary_ports[(bound[pattern_name].uid,
                                         origin_port)] = published
            # The widths the plan declares, keyed as `output_ports` names
            # the ports; a declared port this match did not bind is a plan
            # whose topology and width contract disagree.
            output_dtype = {("" if port == "output" else port): dtype
                            for port, dtype in self.scheme.output_dtype.items()}
            missing = sorted(set(output_dtype) - set(output_ports.values())
                             - set(secondary_ports.values()))
            if missing:
                raise ValueError(
                    f"fusion scheme {self.scheme_name!r} declares an "
                    f"output_dtype for port(s) {missing} that this match "
                    f"does not write")
            primary = next(
                (uid for uid, port in output_ports.items() if not port),
                None,
            )
            if primary is not None:
                sink = primary
            named = (sink if sink is None or is_parallel
                     else _named_member(pattern, bound, sink))
            anchor = (
                program.by_uid(named) if named is not None
                else ordered_members[0]
            )
            if is_parallel:
                anchor = sorted(
                    ordered_members,
                    key=lambda n: (
                        n.display_path or n.name or "",
                        n.uid,
                    ),
                )[0]
            sink_node = (
                program.by_uid(named) if named is not None else None
            )
            external = _external_inputs(
                member_nodes, _input_ports(pattern, bound),
                pattern.input_ports(), pattern.shared_ports, edges, chain,
                match.crossed)
            inputs.update(((next_uid, position), (member, index))
                          for position, (member, index, _ref, _ports)
                          in enumerate(external))
            previous = match.previous
            refs = tuple(ref for _m, _i, ref, _p in external)
            operand_ports = tuple(ports for _m, _i, _r, ports in external)
            if previous:
                # A launch across a seam stands in the reading iteration and
                # reads, on the graph, the formal it read there: what the
                # previous iteration's members read is each position's
                # `Rewrite.inputs`, as for any input no node of this graph
                # writes (`cache.Edges.reads`). An edge drawn to the previous
                # iteration's nodes would close a cycle in the template.
                (formal,) = {
                    tuple(program.by_uid(uid).inputs_obj)[index].external_sid
                    for uid, index in match.crossed}
                # Each is a tensor of its own the loop carries across the
                # seam — vLLM's `hidden_states` and `residual` — and enters
                # the drawn layer through the formal (`InputRef.crosses`):
                # that is the edge a graph draws. Its `external_sid` is the
                # tensor's identity, one per carried tensor (one name for
                # two would be one block input of two sizes), an opaque key
                # no reader parses.
                refs = tuple(
                    type(ref)(src=None, port=ref.port,
                              external_sid=(
                                  f"{formal}:{ref.src.display_path or ref.src.name}"
                                  if ref.src is not None
                                  else ref.external_sid),
                              grafted=True, crosses=formal)
                    if member in previous else ref
                    for (member, _i, ref, _p) in external)
            name = anchor.name
            display_path = anchor.display_path or anchor.name or ""
            if previous:
                # Named for the seam it spans: the reading member may keep
                # runs of its own (the norm that opens a stack), and one
                # reader can close two seams — the layer before it in its
                # own template, and the last layer of the template before
                # (DeepSeek-V3's dense → MoE junction) — so the reader's
                # name alone would be two rows' name.
                writers = {program.by_uid(uid).template or ""
                           for uid in previous}
                (writer,) = writers
                suffix = ("__seam" if writer == (anchor.template or "")
                          else f"__seam@{writer or 'model'}")
                name, display_path = name + suffix, display_path + suffix
            node = FusedNode(
                uid=next_uid, op=self.scheme_name, origins=members,
                name=name,
                role=sink_node.role if sink_node is not None else None,
                template=anchor.template or "",
                display_path=display_path,
                stack_name=anchor.stack_name or "",
                component_path=tuple(anchor.component_path or ()),
                component_scope=anchor.component_scope,
                boundary_role=anchor.boundary_role,
                ports={k: (n.uid, "out") for k, n in bound.items()},
                output_ports=output_ports,
                secondary_ports=secondary_ports,
                inputs_obj=refs,
                sink=sink,
                previous_iteration=previous,
                home_uid=took.home,
                operand_ports=operand_ports,
                # Its plan's answer (`NaivePattern.collective`): one fact,
                # which the signatures that measure the launch read too.
                is_comm_op=pattern.collective(),
                in_dispatch_zone=_routed_span(program, member_nodes),
                # The fused op's own output contract (`allocates_output` of
                # its plan): this launch writes a buffer of its own.
                allocates_output=bool(self.scheme.allocates_output),
                merges_siblings=is_parallel,
                output_scale_tile=(
                    (self.scheme.output_scale_tile["rows"],
                     self.scheme.output_scale_tile["groups"])
                    if self.scheme.output_scale_tile else None),
                output_dtype=output_dtype)
            fused.append(node)
            origins[next_uid] = members
            absorbed |= runs.commit(took)
            taken[next_uid] = took.by_member
            homes[next_uid] = took.home
            next_uid += 1

        if not fused:
            return program, Rewrite()
        # `dropped` is not redundant with `origins`: `origins` says where a NEW
        # node's value comes from, `dropped` says which old ones stop existing.
        # Leaving it empty keeps the absorbed nodes in every layer alongside the
        # node that replaced them, which double-counts any consumer that sums
        # leaves.
        rewrite = Rewrite(origins=origins, dropped=frozenset(absorbed),
                          inputs=inputs, taken=taken, homes=homes)
        return (FusedProgram(base=program, fused=tuple(fused),
                             absorbed=frozenset(absorbed), rewrite=rewrite),
                rewrite)

    # -- closure ---------------------------------------------------------

    @staticmethod
    def _exposed_names(pattern: Any,
                       bound: Optional[Mapping[str, Any]] = None
                       ) -> FrozenSet[str]:
        """Pattern-node names whose output the fused op re-exports, so a
        consumer outside the match still has something to read. In one match
        (`bound`), a port naming an absent optional sink exposes the node it
        falls through to (`NaivePattern.port_node`)."""
        out: set = set()
        for target in pattern.io_mapping.values():
            head, _, _ = str(target).partition(".")
            if head == "external":
                continue
            name = head if bound is None else pattern.port_node(head, bound)
            if name is not None:
                out.add(name)
        return frozenset(out)

    @staticmethod
    def _closed(bound: Mapping[str, Any], inside: FrozenSet[int],
                consumers: Mapping[int, List[Tuple[int, str]]],
                exposed: FrozenSet[str],
                ports: Optional[Mapping[int, FrozenSet[str]]] = None) -> bool:
        """Whether every edge leaving the match has somewhere to land.

        `ports` is given for a boundary-crossing match, where one member can
        write several tensors: then the test is per port — an outside reader
        of a port the fused node exposes is fine, one of any other port is a
        cut edge. Without it the test is per member, as it was before a plan
        could cross a component (`hc_split_sinkhorn`'s `pre` is read inside
        `hc_pre` and its `post` / `comb` outside, and the two answers differ)."""
        for pattern_name, node in bound.items():
            if ports is None and pattern_name in exposed:
                continue
            allowed = frozenset() if ports is None else ports.get(node.uid,
                                                                  frozenset())
            for consumer, port in consumers.get(node.uid, ()):
                if consumer in inside:
                    continue
                if ports is None or port not in allowed:
                    return False
        return True


@dataclass(frozen=True)
class Framework:
    """The fusions one serving framework actually has, applied in order.

    A `Fusion` is never applied on its own: which kernels exist is a property of
    the framework, and `framework_bindings/<framework>.yaml` is where that
    is declared — each entry anchored to real source
    (`vllm.v1.attention.backends.flash_attn.FlashAttentionImpl`). The vLLM
    binding's own comment records `gated_act_ffn` being REMOVED because vLLM
    ships no such kernel and "the mega-fusion was hiding intermediate HBM
    materialization that vLLM actually performs" — which is exactly the error
    applying plans from the registry directly would reintroduce.

    Order is the binding's, already sorted by `match_order` at load
    time so a generic scheme cannot preempt a larger pattern that contains it.
    """

    name: str
    fusions: Tuple[Fusion, ...] = ()

    @classmethod
    def of(cls, binding: Any, schemes: Mapping[str, Any]) -> "Framework":
        """From a binding OBJECT the caller already holds, plus the scheme
        registry the caller holds alongside it. Both live in `inference`
        (`framework_bindings` / `fusion_plans`); this package never imports
        the registries, so the caller — pipeline, a test — passes them in.

        `binding.dispatched()` is the binding's answer for its deployment:
        `(scheme, GraphCondition | None)` in specificity-sorted order, an
        entry whose server flag, activation dtype or device the deployment
        does not satisfy already left out. The kernel name and its source
        anchor are evidence about the framework rather than about the graph,
        so only the scheme and the graph's part of the condition come here.
        """
        return cls(binding.framework,
                   tuple(Fusion.named(op, schemes, condition, crosses)
                         for op, condition, crosses in binding.dispatched()))

    def rewrite(self, program: Any, analysis: Any,
                local_shapes: Any = None,
                graft: Any = None) -> Tuple[Any, Rewrite]:
        rewrite = Rewrite()
        for index, fusion in enumerate(self.fusions):
            refuse_tied_preemption(self.name, self.fusions, index, program,
                                   analysis, local_shapes, graft)
            program, step = fusion.rewrite(program, analysis, local_shapes,
                                           graft)
            rewrite = rewrite.then(step)
        return program, rewrite


def fusion_matches(source: Any, program: Any, rewrite: Rewrite,
                   schemes: Mapping[str, Any]) -> Tuple[dict, ...]:
    """Final reporting rows for a framework rewrite.

    Matching and value transport stay UID-keyed. Aggregation ids are rendered
    only here, at the delivery boundary that needs them. `schemes` is the
    caller's registry (see `Fusion.named`).
    """
    rows = []
    for node in program.nodes():
        if not isinstance(node, FusedNode):
            continue
        origins = rewrite.sources(node.uid)
        members = []
        for uid in origins:
            origin = source.by_uid(uid)
            if origin is not None:
                members.append(origin.aggregation_id)
        scheme = schemes.get(node.op)
        pattern = scheme.naive_pattern if scheme is not None else None
        rows.append({
            "fused_op": node.op,
            "fused_uid": node.uid,
            "fused_aggregation_id": node.aggregation_id,
            "matched_uids": tuple(origins),
            "matched_aggregation_ids": tuple(members),
            # Every absorbed member the replacement exposes, primary as
            # well as named secondary ports; renderers use this to retarget
            # pre-fusion refs that now name a child of another compound.
            "absorbed_output_ports": tuple(
                (source.by_uid(uid).aggregation_id, port)
                for uid, port in node.output_ports.items()
            ),
            # A member's secondary output the launch publishes, with the
            # origin port a pre-fusion ref names it by (`split.post`).
            "absorbed_secondary_ports": tuple(
                (source.by_uid(uid).aggregation_id, origin, port)
                for (uid, origin), port in node.secondary_ports.items()
            ),
            "is_parallel": bool(
                pattern is not None and pattern.is_parallel()
            ),
            # A launch across a seam: the members that ran a layer earlier,
            # and those it took only some runs of — each keeps a row of its
            # own beside the launch (`Rewrite.taken`).
            "previous_iteration_aggregation_ids": tuple(
                source.by_uid(uid).aggregation_id
                for uid in sorted(node.previous_iteration)),
            "kept_aggregation_ids": tuple(
                source.by_uid(uid).aggregation_id for uid in origins
                if rewrite.partly(uid)),
        })
    return tuple(rows)


def materialize_tree(source: Any, program: Any, rewrite: Rewrite) -> Any:
    """Render a fused program as a ``SpecIRTree`` for graph delivery.

    This is a structural projection of the rewrite. It creates no evaluated
    result objects and performs no matching of its own.
    """
    import copy

    tree = copy.deepcopy(source.ir)
    if rewrite.is_identity():
        return tree

    def walk():
        def visit(node):
            yield node
            for child in node.children or ():
                yield from visit(child)
        for root in tree.top_level:
            yield from visit(root)
        for _meta, roots in tree.blocks:
            for root in roots:
                yield from visit(root)

    copied = {node.uid: node for node in walk()}
    fused_nodes = [
        node for node in program.nodes() if isinstance(node, FusedNode)
    ]
    replacements: Dict[int, Any] = {}
    absorbed_by: Dict[int, int] = {}
    leaving: set = set()

    def descends_from(node: Any, roots: Sequence[Any]) -> bool:
        while node is not None:
            if any(root.uid == node.uid for root in roots):
                return True
            node = node.parent
        return False

    def ancestors(uid: int) -> List[Any]:
        """The nodes above `uid`, outermost first."""
        out: List[Any] = []
        node = copied[uid].parent
        while node is not None:
            out.append(node)
            node = node.parent
        return out[::-1]

    def root_siblings(origins: set) -> Tuple[list, Any]:
        """Where the launch stands: among the children of its members'
        lowest common ancestor, or — where no node is above them all — in
        the one root list (the model's top level, or a block's roots) they
        all descend from. A launch whose members sit in one component stands
        in it (`FusedNode.component_path` says the same); one over several
        component instances of a layer stands at the block's level, and its
        members leave the wrappers they were in."""
        common = None
        for level in zip(*(ancestors(uid) for uid in origins),
                         strict=False):
            if any(node.uid != level[0].uid for node in level):
                break
            common = level[0]
        if common is not None:
            return common.children, common
        for roots in (tree.top_level, *(roots for _m, roots in tree.blocks)):
            if all(descends_from(copied[uid], roots) for uid in origins):
                return roots, None
        raise ValueError(
            f"fusion origins are not in one root scope: {sorted(origins)!r}"
        )

    for fused in sorted(fused_nodes, key=lambda node: node.uid):
        # A launch across a seam stands among the reading iteration's
        # members; the previous iteration's are another layer's nodes.
        origins = set(rewrite.sources(fused.uid)) - set(
            fused.previous_iteration)
        if not origins:
            continue
        siblings, parent = root_siblings(origins)
        indices = [i for i, node in enumerate(siblings) if node.uid in origins]
        crossing = {siblings[i].uid for i in indices} != origins
        if crossing:
            # A boundary-crossing kernel: the members are spread over the
            # layer's component instances rather than lying side by side.
            # It takes the position of the topmost member's own ancestor in
            # the layer, and every member is removed where it was.
            def ancestor(uid: int) -> Optional[Any]:
                """The member's own ancestor that stands at the layer's
                level, i.e. the first one that IS a block root."""
                node = copied[uid]
                roots = {root.uid for root in siblings}
                while node is not None and node.uid not in roots:
                    node = node.parent
                return node
            tops = {found.uid for uid in origins
                    if (found := ancestor(uid)) is not None}
            indices = [i for i, node in enumerate(siblings)
                       if node.uid in tops]
            if not indices:
                raise ValueError(
                    f"fusion origins are not under one block: "
                    f"{sorted(origins)!r}"
                )
        anchor_uid = fused.sink if fused.sink in origins else min(origins)
        anchor = copied[anchor_uid]
        # A kernel over model-level ops stands in the spine at the last of
        # them; the others are recorded as absorbed there, so the spine's
        # stages still resolve (`SpecIRTree.fused_layers`).
        spine_paths = (sorted(tuple(copied[uid].layer_path)
                              for uid in origins if copied[uid].layer_path)
                       if siblings is tree.top_level else [])
        layer_path = spine_paths[-1] if spine_paths else ()
        for path in spine_paths[:-1]:
            tree.fused_layers[path] = layer_path
        replacement = tree.make_node(
            # The anchor picks the POSITION; the row's identity is the fused
            # node's own (for parallel fusions that is the lexicographically
            # first member, not the sink). Naming the replacement after the
            # anchor gave one fused kernel two aggregation_ids — one on this
            # tree, one on `program` / `fusion_matches` / the timing rows.
            name=fused.name,
            op=fused.op,
            # A fused kernel named after a first-class registration (DSA's
            # reduced `indexer_logits`) carries it; one with no registration
            # (`flash_attention`, `allreduce_fused_rmsnorm`) carries None and
            # its consumers read the span's signature (`FusedNode.is_comm_op`,
            # the fusion scheme) instead.
            declaration=OP_REGISTRY.get(fused.op),
            is_component=False,
            from_custom_op=True,
            role=fused.role,
            template=fused.template,
            stack_name=fused.stack_name,
            component_path=fused.component_path,
            component_scope=fused.component_scope,
            boundary_role=fused.boundary_role,
            dims=dict(anchor.dims or {}),
            inputs=(),
            inputs_obj=(),
            children=[],
            fused_from=fused.op,
            provenance="fusion",
            layer_path=layer_path,
        )
        if replacement.uid != fused.uid:
            raise ValueError(
                f"fusion tree allocated uid {replacement.uid}, expected "
                f"rewrite uid {fused.uid}"
            )
        siblings.insert(min(indices), replacement)
        replacement.parent = parent
        replacements[fused.uid] = replacement
        # A member leaves the tree once a launch took every run of it —
        # after every launch has found its place, since one a launch across
        # a seam took only some runs of (`Rewrite.taken`) stays where it
        # was, and the next launch that takes the rest stands beside it.
        # Each member leaves the component instance it was in. The
        # instances themselves STAY, even one whose every leaf a launch
        # took: a component's cost and bytes are its children's, so an
        # empty one contributes nothing, while removing it would strand
        # every ref that names it — a component reference names the
        # wrapper, which is the whole reason a boundary-crossing view exists.
        for uid in set(rewrite.sources(fused.uid)) & rewrite.dropped:
            absorbed_by[uid] = fused.uid
            leaving.add(uid)

    for uid in leaving:
        holder = copied[uid].parent
        if holder is not None:
            holder.children[:] = [c for c in holder.children
                                  if c.uid != uid]
            continue
        for roots in (tree.top_level, *(roots for _m, roots in tree.blocks)):
            roots[:] = [c for c in roots if c.uid != uid]

    source_by_uid = {node.uid: node for node in program.nodes()}
    live = {node.uid: node for node in walk()}

    def absorbed_port(producer_uid: int, port: str) -> str:
        """The port a ref must name once its producer is absorbed.

        A fusion that publishes only named outputs — `fused_qk_rope`
        maps `q_out`/`k_out` and no main output — leaves the compound
        with no empty-string port. Carrying `port=""` across the rewrite
        points at an output the compound does not have.
        """
        fused_uid = absorbed_by.get(producer_uid)
        if fused_uid is None:
            return port
        published = source_by_uid[fused_uid].published_port(producer_uid, port)
        return port if published is None else published

    def writing_leaf(ref: Any) -> Optional[Tuple[int, str]]:
        """`(leaf uid, port)` a ref that names a COMPONENT resolves to — the
        leaf that writes the port it reads, resolved on the tree as it was
        before the rewrite. A boundary-crossing launch takes leaves out of
        the components they were in, so a surviving consumer that names the
        component has to be rewired to the launch; the component itself is
        no longer a producer of that tensor."""
        node = source_by_uid.get(ref.src.uid)
        if node is None or not node.children:
            return None
        leaf, port = source.output_leaf(node, ref.port or "out")
        return leaf.uid, (port or "out")

    def retarget(ref: Any) -> Optional[Tuple[Any, str]]:
        """`(producer, port)` this ref must name after the rewrite, or None
        when it is unchanged. Either the ref's own producer was absorbed, or
        it names a component whose writing leaf was."""
        if ref.src is None:
            return None
        if ref.src.uid in absorbed_by:
            return (replacements[absorbed_by[ref.src.uid]],
                    absorbed_port(ref.src.uid, ref.port))
        found = writing_leaf(ref)
        if found is None or found[0] not in absorbed_by:
            return None
        leaf_uid, leaf_port = found
        return (replacements[absorbed_by[leaf_uid]],
                absorbed_port(leaf_uid, leaf_port))

    for fused_uid, replacement in replacements.items():
        fused = source_by_uid[fused_uid]
        refs = []
        for ref in fused.inputs_obj or ():
            if ref.src is None:
                refs.append(type(ref)(
                    src=None,
                    port=ref.port,
                    external_sid=ref.external_sid,
                    grafted=ref.grafted,
                    crosses=ref.crosses,
                ))
                continue
            moved = retarget(ref)
            if moved is not None:
                refs.append(type(ref)(src=moved[0], port=moved[1]))
                continue
            target = live.get(ref.src.uid)
            if target is None:
                raise ValueError(
                    f"fused input references missing uid {ref.src.uid}"
                )
            refs.append(type(ref)(src=target, port=ref.port))
        replacement.inputs_obj = tuple(refs)

    # Rewire surviving consumers whose producer was absorbed — directly, or
    # through the component instance they name.
    for node in walk():
        if node.uid in replacements:
            continue
        refs = []
        changed = False
        for ref in node.inputs_obj or ():
            moved = retarget(ref)
            if moved is None:
                refs.append(ref)
                continue
            changed = True
            refs.append(type(ref)(src=moved[0], port=moved[1]))
        if changed:
            node.inputs_obj = tuple(refs)
    return tree


def kept_rows(analysis: Any) -> FrozenSet[int]:
    """The members whose shape row a rewrite keeps even when a kernel
    absorbs them whole (`rebase_shapes(preserve=...)`): what is priced off
    the member's own geometry rather than off the fused node's. A cached
    node (the cache it writes is unchanged), a recurrent state's carrier
    (its per-rank axes come from the carrier's shape), and a lookup (the
    rows it gathers are its output's: `traffic._table_read_fractions`)."""
    return frozenset(cached_node_uids(analysis.program)
                     | state_carrier_uids(analysis)
                     | set(analysis.program.lookups))


def _moves_values(op: Optional[str]) -> bool:
    """Whether a fusion member writes values, or only re-partitions them.

    A view re-indexes (`OpSpec.is_view_op`) and a join re-indexes two
    operands into one frame (`ShapeKind.AXIS_CONCAT`) — both read off the
    op's registration rather than listed here. Neither is free on its own:
    the registry says so for the join in as many words ("`torch.cat` always
    allocates", `ops_registry.py`), and the naive graph is billed for it.
    What makes the region free is being INSIDE the launch — a plan that
    swallows the `slice`/`concat` pair around a partial rotation is saying
    serving does not re-join q, because the rotary kernel writes the lanes
    it rotated back where it found them. That claim belongs to the plans
    that make it, not to fusion in general, which is why a plan whose
    launch writes a buffer of its own says so and this question is never
    asked of it (`_passed_through`). Here the two kinds are only the
    candidates: a region that crossed nothing but views and joins is a
    region no member of the match wrote. An op the registry does not know
    (a custom op, a component) computes until it says otherwise.
    """
    spec = OP_REGISTRY.get(op or "")
    if spec is None:
        return True
    return not (spec.is_view_op or spec.shape_kind is ShapeKind.AXIS_CONCAT)


def _passed_through(fused: Any, members: Sequence[Any],
                    pre: Mapping[int, Any],
                    sources: Sequence[Tuple[int, int]],
                    output_ports: Mapping[int, str]) -> Any:
    """What a fused boundary publishes but its launch does not move.

    The boundary's two sides are wider than the kernel wherever the match
    absorbed a partition: `fused_qk_rope_partial_q` publishes the whole of
    `q` because it took over the `slice` that cut the rope lanes out of it
    and the `concat` that put them back, and the launch rotates only the
    lanes. The difference is read off the match's own topology — which
    branch reached a member that writes values (`_moves_values`) — and
    never off the plan, which carries no cost arithmetic.

    Both sides are stated as a subtraction from the published tensor rather
    than a sum over the live regions, because a region that reached no
    computing member is the thing this knows: a `broadcast` reads its source
    once and publishes it many times, so summing what the members consume
    would bill a repeat the kernel does in registers.

    Element counts come from `pre`, at whatever granularity the caller's
    shape layer holds — the serving path rebases per-rank shapes, so the
    regions it subtracts are per-rank too.
    """
    if fused.allocates_output or fused.output_dtype:
        # A launch that writes a buffer or a format of its OWN cannot be
        # rewriting a tensor in place: every lane of the output is a new
        # buffer's lane, and every lane of the input has to be read to
        # write it. So the region question does not arise.
        #
        # A launch that quantises its output needs no declaration here: the
        # cast is a member (`quantise_activation`), it moves values
        # (`_moves_values`), and a region reaching it is moved. A launch
        # that allocates its output in the dtype it was handed has no member
        # that says so — the declaration is the only thing that can, and it
        # is the plan's to make (`fusion_plans/fused_inv_rope.yaml`).
        return None

    inside = {node.uid: node for node in members}

    def less(whole: Any, regions: Sequence[Any]) -> Any:
        """A published tensor less the regions nobody moved — and the whole
        of it where that subtraction does not leave a region.

        Two readings it refuses. Overlapping regions: `kv_a` and `kv_b` of
        `fused_overlap_compress_norm_rope_store` are two windows over one
        `external.kv`, and nothing here says two regions of a tensor are
        disjoint, so their sum can exceed it. And a view that publishes MORE
        than it reads: `pad_rows` is a view by registration and its output
        is the LARGER tensor, so subtracting it from the input it came from
        would leave negative bytes. Both are the same answer — a launch
        whose passed-through region cannot be established is billed for the
        tensor it was handed, which is the price before any of this.

        Covering the tensor EXACTLY is not one of them, and stays zero: a
        view the match exposes as a port passes the whole of what it was
        handed through, and the naive graph charges nothing for it either
        (`fused_qk_rmsnorm_rope_gate` publishes a slice of its input as the
        gate). Only a sum that OVERSHOOTS says the regions were not a
        partition of the tensor.

        One formula, not a choice this takes (`dim.where`): at a symbolic
        shape whether the regions cover the tensor is the point's."""
        taken = sum((product(dims) for dims in regions[1:]),
                    product(regions[0]))
        span = product(whole)
        return where(sp.Le, (taken, span), span - taken, span)

    def producer(node: Any, index: int) -> Optional[int]:
        refs = tuple(node.inputs_obj or ())
        if index >= len(refs):
            return None
        src = refs[index].src
        return src.uid if src is not None else None

    def source_key(node: Any, index: int) -> Any:
        """The outside tensor an input names, as `_external_inputs` keys it,
        so the readers of one boundary input can be found together."""
        refs = tuple(node.inputs_obj or ())
        if index >= len(refs):
            return None
        ref = refs[index]
        src = ref.src
        return ((src.uid if src is not None else
                 ("alias", ref.external_sid or f"{node.uid}:{index}")),
                ref.port or "out")

    consumers: Dict[int, List[int]] = {}
    for node in members:
        for index in range(len(tuple(node.inputs_obj or ()))):
            found = producer(node, index)
            if found in inside:
                consumers.setdefault(found, []).append(node.uid)

    reached: Dict[int, bool] = {}

    def reaches_compute(uid: int) -> bool:
        """Whether this member's output is consumed, within the match, by a
        member that writes values — directly or past further views and
        joins. A member nothing inside reads passes its region straight to
        the boundary, so it reaches nothing."""
        if uid in reached:
            return reached[uid]
        reached[uid] = False          # a cycle would be a broken match
        answer = any(
            _moves_values(inside[c].op) or reaches_compute(c)
            for c in consumers.get(uid, ()))
        reached[uid] = answer
        return answer

    def region(uid: int, index: int) -> Optional[Any]:
        """A member slot's dims: its output for `index < 0`, else the tensor
        it reads at that input."""
        row = pre.get(uid)
        if row is None:
            return None
        if index < 0:
            return row.out
        return row.ins[index] if index < len(row.ins) else None

    # -- the read side: regions of a boundary input nothing computed from --
    reads: Dict[int, Any] = {}
    for position, (member_uid, index) in enumerate(sources):
        key = source_key(inside.get(member_uid), index) \
            if member_uid in inside else None
        whole = region(member_uid, index)
        if key is None or whole is None:
            continue
        readers = [(node, i) for node in members
                   for i in range(len(tuple(node.inputs_obj or ())))
                   if source_key(node, i) == key]
        if any(_moves_values(node.op) for node, _ in readers):
            # Something inside reads this tensor as it arrives: nothing of
            # it is merely passed through, whatever else also slices it.
            continue
        regions = []
        for node, _i in readers:
            if repeats_input(node.op or "") or reaches_compute(node.uid):
                # A repeat publishes its source many times over, and a live
                # branch is read: neither is a region nobody touched.
                continue
            if len(tuple(node.inputs_obj or ())) != 1:
                # Only a ONE-input view publishes a region OF the tensor it
                # reads; a join publishes its operands side by side, so its
                # output is no slice of any of them to subtract.
                continue
            exposed = region(node.uid, -1)
            if exposed is None or len(exposed) != len(whole):
                # A region in another axis frame than the tensor it came
                # from is not a slice of it; keep the published tensor.
                regions = None
                break
            regions.append(exposed)
        if regions:
            reads[position] = less(whole, regions)

    # -- the write side: regions of an output port nothing wrote ----------
    def unwritten(uid: int, seen: FrozenSet[int]) -> List:
        """The slots of `uid`'s output region no member of the match wrote,
        each one a `(member, input)` whose dims the shapes can price (or
        `(member, -1)` for a member's whole output)."""
        node = inside[uid]
        if _moves_values(node.op) or uid in seen:
            return []
        seen = seen | {uid}
        spec = OP_REGISTRY.get(node.op or "")
        if spec is not None and spec.shape_kind is ShapeKind.AXIS_CONCAT:
            slots: List = []
            for index in range(len(tuple(node.inputs_obj or ()))):
                found = producer(node, index)
                if found is None or found not in inside:
                    # The branch arrived from outside: it is in HBM already,
                    # written by whoever produced it, and this launch leaves
                    # it where it is.
                    slots.append((uid, index))
                    continue
                # Whatever the branch's own producer left unwritten, in its
                # own slots — they sum to this branch's region.
                slots.extend(unwritten(found, seen))
            return slots
        # A view publishes its source's region: whatever wrote that wrote
        # this. A view over the outside wrote nothing.
        found = producer(node, 0)
        if found is None or found not in inside:
            return [(uid, -1)]
        return unwritten(found, seen)

    writes: Dict[str, Any] = {}
    for member_uid, port in output_ports.items():
        if member_uid not in inside:
            continue
        whole = region(member_uid, -1)
        if whole is None:
            continue
        slots = unwritten(member_uid, frozenset())
        if not slots:
            continue
        regions = [region(uid, index) for uid, index in slots]
        if any(dims is None for dims in regions):
            continue
        writes[port] = less(whole, regions)
    # A member's secondary output the launch publishes: the whole of it is
    # that port's tensor, and what nobody moved is read off the member's
    # inputs exactly as for its main output.
    for (member_uid, origin), port in fused.secondary_ports.items():
        if member_uid not in inside:
            continue
        row = pre.get(member_uid)
        whole = None if row is None else (row.ports or {}).get(origin)
        if whole is None:
            continue
        slots = unwritten(member_uid, frozenset())
        if not slots:
            continue
        regions = [region(uid, index) for uid, index in slots]
        if any(dims is None for dims in regions):
            continue
        writes[port] = less(whole, regions)

    return Moved(reads=reads, writes=writes) if (reads or writes) else None


def rebase_shapes(
    source_program: Any,
    shapes: Any,
    program: Any,
    rewrite: Rewrite,
    *,
    preserve: FrozenSet[int] = frozenset(),
) -> Any:
    """Carry a shape layer across a rewrite using graph boundaries.

    An absorbed member whose output the rest of the graph still reads keeps
    its own row. The rewrite does not touch that tensor — it still lands in
    HBM, now as one of the fused kernel's ports — and the outside consumer's
    ref still names the member: `k_pe_broadcast` reads `k_pe_rope` after
    `fused_qk_rope` absorbs it. A walk that follows the ref (the fan-out a
    `head_broadcast` repeats, the cache pedigree) needs the row: without it
    the fan-out falls back to 1 and the broadcast is billed once per head.
    `preserve` adds what the caller knows beyond that — cached nodes a
    kernel absorbs whole. A kept row is looked up, never billed: its bytes
    are the fused node's (`traffic.derive` walks live nodes only).
    """
    if rewrite.is_identity():
        return Layer(program, dict(shapes.values))
    rebased = shapes.rebase(rewrite, program, Shape.combine)
    pre = shapes.values
    boundary: Dict[int, Shape] = {}
    exposed: set = set()
    for fused_uid, member_uids in (rewrite.origins or {}).items():
        members = [source_program.by_uid(u) for u in member_uids]
        members = [m for m in members if m is not None]
        if not members:
            continue
        fused = program.by_uid(fused_uid)
        if not isinstance(fused, FusedNode):
            raise TypeError(f"rewrite uid {fused_uid} is not a fused node")
        # The boundary's input side has ONE source: the `(member, index)` the
        # rewrite recorded per position when it built the fused node's refs
        # (`Rewrite.inputs`). Recomputing it from the members would answer a
        # different question for a boundary-crossing match, whose view made
        # some of those edges internal — so a fused node that reads anything
        # and recorded nothing is a broken rewrite, not a case to guess at.
        rows = rewrite.inputs or {}
        positions = sorted(pos for uid, pos in rows if uid == fused_uid)
        if not positions and (fused.inputs_obj or ()):
            raise ValueError(
                f"fused uid {fused_uid} ({fused.op}) reads "
                f"{len(fused.inputs_obj)} inputs but the rewrite recorded "
                f"none; its shape row and its refs would disagree")
        ins = []
        # The sources the `ins` built below actually carry, by the position
        # they end up at: `_passed_through` prices BY POSITION, and a source
        # whose row is missing is not one of them.
        sources = []
        for member_uid, index in (rows[(fused_uid, pos)] for pos in positions):
            shape = pre.get(member_uid)
            if shape is None or index >= len(shape.ins):
                # One operand per ref and per named port
                # (`FusedNode.inputs_obj`, `operand_ports`): an operand
                # dropped here would hand every later port the tensor
                # before it.
                raise ValueError(
                    f"fused uid {fused_uid} ({fused.op}): operand "
                    f"{len(ins)} reads input {index} of uid {member_uid}, "
                    f"which has no shape row; its refs and ports would "
                    f"name the operands after it")
            ins.append(shape.ins[index])
            sources.append((member_uid, index))
        output_ports = fused.output_ports
        secondary_ports = fused.secondary_ports
        exposed.update(output_ports)
        exposed.update(uid for uid, _origin in secondary_ports)
        primary = next(
            (uid for uid, port in output_ports.items() if not port),
            None,
        )
        primary_shape = pre.get(primary) if primary is not None else None
        primary_out = primary_shape.out if primary_shape is not None else None
        if primary_out is None:
            # The primary may be a member's SECONDARY output
            # (`output: split.post`): then it is that port's tensor.
            primary_out = next(
                ((pre[uid].ports or {}).get(origin)
                 for (uid, origin), port in secondary_ports.items()
                 if not port and uid in pre),
                None)
        ports = {
            port: pre[uid].out
            for uid, port in output_ports.items()
            if port and uid in pre and pre[uid].out is not None
        }
        for (uid, origin), port in secondary_ports.items():
            dims = (pre[uid].ports or {}).get(origin) if uid in pre else None
            if port and dims is not None:
                ports[port] = dims
        boundary[fused_uid] = Shape(
            out=primary_out,
            ins=tuple(ins),
            ports=ports,
            # The tensors above are what the launch is HANDED; a match that
            # absorbed a partition is handed more than it moves, and the
            # difference is the match's own topology (`_passed_through`).
            moved=_passed_through(fused, members, pre, sources,
                                  output_ports),
        )
    values = {**rebased.values, **boundary}
    # An absorbed attention kernel keeps its row as well: its score matrix is
    # the one tensor that says which of the step's queries the fused kernel
    # scores. Under Ulysses the kernel's boundary carries `s_query / cp` on
    # both sides — the query enters split by sequence, the output leaves
    # split by sequence — while the kernel scores every query for its heads
    # (`traffic._query_share`).
    kernels = frozenset(
        uid for uid in rewrite.dropped
        if (member := source_program.by_uid(uid)) is not None
        and attention_kernel(member.op) is not None)
    for uid in (frozenset(preserve) | exposed | kernels) & rewrite.dropped:
        if uid in pre:
            values.setdefault(uid, pre[uid])
    return Layer(program, values)


def union_consts(parts: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    """The function constants a fused kernel takes: those its origins took
    (`Analysis.consts`) — the clamped SwiGLU takes the bound its two `clamp`
    origins clamp to. A launch takes one value per constant, so a symbol
    two origins bound to DIFFERENT values has none here: absent, and a
    consumer that has to compute the kernel declines, rather than being
    handed one origin's bound for both."""
    out: Dict[str, float] = {}
    clash: set = set()
    for part in parts:
        for name, value in part.items():
            if name in out and out[name] != value:
                clash.add(name)
            out.setdefault(name, value)
    return {name: value for name, value in out.items() if name not in clash}


def _last_kv_length(parts: Sequence[Any]) -> Any:
    """A fused launch's output carries its surviving endpoint's KV length."""
    return parts[-1]


def rebase_analysis(analysis: Any, program: Any, rewrite: Rewrite) -> Any:
    """Carry every layer of an analysis across a graph rewrite.

    Each layer states how its own values combine, which is what a single
    synthesised fused result cannot do: cost sums, shape spans.

    Storage and formats are NOT rebased at all — not dropped, not merged. They
    key on the tensors a compute-graph rewrite never touches: a fused QKV still
    reads three weight matrices from disk, each with its own dtype. Merging them
    would misreport the checkpoint; dropping them loses the reads entirely, and
    the fused node's traffic came out with no weight bytes at all.
    """
    from dataclasses import replace as _replace

    if rewrite.is_identity():
        return analysis
    # A cached tensor's geometry is storage-like. The walk declared the cache
    # on a node a kernel rewrite may absorb — fused qk-rope swallows
    # `k_pe_rope` — but the CACHE the fused kernel writes is unchanged: the
    # same rope'd K lands in the same rows. `rebase_shapes` keeps every
    # absorbed member the outside still reads; this adds the cached ones a
    # kernel may absorb whole, or the footprint derivation finds a cache
    # declaration with no row to bill and refuses (correctly) to price the
    # model. A recurrent state's carrier is kept for the same reason: the
    # state's per-rank axes are projected through the carrier's kernel shape.
    shapes = rebase_shapes(
        analysis.program,
        analysis.shapes,
        program,
        rewrite,
        preserve=kept_rows(analysis) & rewrite.dropped,
    )
    own_copies = analysis.own_copies.rebase(rewrite, program, OwnCopies.fused)
    return _replace(
        analysis,
        program=program,
        cost=analysis.cost.rebase(rewrite, program, Cost.combine),
        # How often each node runs, off the rewrite's one statement of it
        # (`Rewrite.taken`): a launch across a layer seam runs once per seam,
        # and a member it took only some runs of keeps the rest.
        instances=analysis.instances.rebase_counts(rewrite, program),
        shapes=shapes,
        roles=analysis.roles.rebase(rewrite, program, lambda parts: parts[0]),
        consts=analysis.consts.rebase(rewrite, program, union_consts),
        kv_lengths=analysis.kv_lengths.rebase(
            rewrite, program, _last_kv_length),
        # A member keeps its own count across the rewrite; a fused node
        # takes its members' when they agree and has none when they do not
        # (`OwnCopies.fused`) — absent, so a reader is refused rather than
        # handed one member's multiplier.
        own_copies=Layer(program, {
            uid: copies for uid, copies in own_copies.values.items()
            if copies is not None}),
        # storage / formats deliberately untouched — see above
        rewrite=analysis.rewrite.then(rewrite),
        # The walk this rewrite started from: what a rewrite merges without
        # changing (the expert pools) is read off its layers.
        walked=analysis.walked if analysis.walked is not None else analysis,
        # The references are the walk's, over the walked program
        # (`Analysis.reference_table`); the rewritten program's nodes hold
        # none of their own.
        input_references=None,
    )
