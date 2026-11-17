"""`Program` — the immutable side of an analysis: the spec, its walked graph, and
the questions you ask about structure rather than about cost.

Two decisions are recorded here.

**The graph is the existing `SpecIRTree`, not a new type.** The designed layer-B
`SpecNode` was "today's SpecIRNode, tightened": fields grouped into `scope` /
`decl` sub-objects, the parallel string channel (`inputs` beside `inputs_obj`,
`dims_raw` beside `dims`) dropped. That is ergonomics, not capability —
`SpecIRNode` already carries a tree-unique `uid`, object-ref inputs, resolved
dims, propagated role, and the dispatch-zone flag, which is everything a walk
needs. Rebuilding it would also mean rebuilding `iter_tree`, and that walker is
where adapter transparency, call-site renames, HF weight naming and multiplicity
accumulation live — a thousand lines of accumulated correctness that is not what
is wrong with the engine. So `Layer` keys on `SpecIRNode.uid` and the builder
stays.

**Walking the IR removes two of the walk's four scope dimensions.** `iter_tree`
resolves `op: <param>` before emitting, so the callee-params frame has nothing
left to do; it precomputes `in_dispatch_zone` per node, so the zone set does too.
The engine needs those stacks because it walks `BlockGraph` — the authored
syntax, with the holes still in it. Walking the resolved graph instead, a frame
carries only the dim binding and the sibling results.

Structural queries live here rather than on the evaluator because they classify
(MHA vs GQA vs MLA, dense vs MoE, which layer index belongs to which assignment)
and classification does not depend on whether dims are ints or symbols. That
dependency is why `SymbolicEngine` has to override four methods to swap numeric
dims back in for the duration of a call: the questions were asked of the wrong
object.

**What the IR does not name, this module does.** Keeping `SpecIRNode` settles
the NODE; it leaves two shapes with no type of their own, and both cost the same
way. `SpecIRTree.blocks` holds a block as a bare `(BlockMeta, roots)` tuple, so
every fact about a template — its layer count, its declared inputs, the dims its
assignment rebinds — had to be a free function unpacking the tuple again. And a
stage of the model is either such a tuple or a node, discriminated by
`isinstance(entry, tuple)` at each of the four places one is handled.

`BlockTemplate` is the first; `Stage` / `OpStage` / `BlockStage` is the second,
and it turns that discrimination into one dispatch. Neither is a new
representation — they wrap what `iter_tree` already produced, and the tree stays
the source of truth.
"""
from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import (
    Any, ClassVar, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence,
    Tuple, Union,
)

from ..ops import (DISPATCH_OP, OpSpec, ProcessorStep, is_dispatch_boundary,
                   is_identity_source, lookup_of)
# `HISTORY_AXIS` is re-exported for the walk, which binds a `read: all`
# value's history axis to it.
from ..schema import HISTORY_AXIS as HISTORY_AXIS, parse_input_ref
from ..spec_tree import (
    STACK_ENTRY, BlockMeta, InputRef, SpecIRNode, SpecIRTree, SpineEntry,
    TreeStream,
    checkpoint_prefix, ir_dim_symbols, ir_to_leaf, iter_leaves, iter_tree,
    require_module,
    scope_prefix,
)
from .binding import Binding
from .copies import copy_factors_of
from .dim import eval_dim_expr, value_of
from .layer import Layer
from .node import Evaluable
from .occurrence import Occurrence, Reference, Source
from .op_ref import OpRef
from .refusal import SpecRefusal
from .storage import StorageKind
from backend.code_trace.memo import memo_property


@dataclass(frozen=True)
class BlockTemplate:
    """A block template invocation: its metadata and the IR body it expanded to.

    `SpecIRTree.blocks` holds these as bare `(BlockMeta, roots)` tuples, which
    is why `is_block` and `entry[1]` appear wherever one is handled. A tuple has
    no methods, so everything true of a template — how many layers use it, what
    its body declares as inputs, which dims the assignment rebinds — had to be
    a free function taking the tuple apart again.
    """

    meta: BlockMeta
    roots: Tuple[SpecIRNode, ...]

    @property
    def name(self) -> str:
        return self.meta.template_name

    @property
    def checkpoint_prefix(self) -> str:
        """The checkpoint address prefix this invocation's tensors live
        under (`spec_tree.checkpoint_prefix` of its stack)."""
        return checkpoint_prefix(self.meta)

    @property
    def count(self) -> int:
        """How many physical layers use this template.

        `or 1` is the wrong guard: it defends against `None` and swallows a
        legitimate ZERO. A template with no layers assigned contributes nothing,
        and counting it once put GLM-5.1 over by exactly the two such blocks it
        has — 19,745,437,440 — and GLM-5.2 by the one it has.

        A property of the TEMPLATE, so the walk's per-node instance count and
        the model total cannot each decide what a missing count means.
        """
        raw = self.meta.count
        return 1 if raw is None else int(raw)

    @property
    def dim_overrides(self) -> Mapping[str, Any]:
        """Per-assignment dim rebindings — `n_kv_heads: n_kv_heads_swa` on a
        hybrid model's sliding-window layers. Evaluating a variant without them
        uses the backbone's dims for a block that does not have them."""
        return self.meta.dim_overrides or {}

    def input_names(self) -> List[str]:
        """The body's declared `inputs:`, for binding what flows in."""
        names = list(self.meta.inputs)
        if not names:
            raise ValueError(
                f"block template {self.name!r} has no declared inputs")
        return names

    def reads_formal(self, name: str) -> bool:
        """Whether the body consumes its formal `name` (`_consumes`). A
        formal declared only so the template's interface matches its
        stack's reads nothing: DeepSeek-V4.1's `dsv41_hc_block` takes
        `input_ids` beside the Engram block that reads it, and DeepSeek-V4's
        block hands `input_ids` to whichever FFN its slot binds, which the
        score-routed one ignores."""
        return _consumes(self.roots, name)

    @property
    def layer_indices(self) -> Tuple[int, ...]:
        """The stack iterations this invocation stands for, resolved: an
        assignment's selected layers, or every iteration of a `template:`
        stack."""
        if self.meta.layer_indices is not None:
            return tuple(self.meta.layer_indices)
        return tuple(range(self.count))

    @property
    def outer_repeat(self) -> int:
        """The resolved `repeat:` of the group enclosing this invocation's
        stack (1 outside a repeated group). `count` already includes it."""
        return self.meta.outer_repeat

    @property
    def visible(self) -> Mapping[str, int]:
        """How many elements of its stack's `read: all` stream each layer
        of this invocation sees (`BlockMeta.visible`)."""
        return dict(self.meta.visible)

    def nodes(self) -> Iterator[SpecIRNode]:
        """Every IR node of the body, depth-first."""
        return _depth_first(self.roots)

    def ffn_type(self) -> Optional[str]:
        """The invocation's primary FFN type: `moe` when it routes tokens to
        experts — a node of it is declared in a dispatch zone
        (`Program.has_dispatch_zone`, asked of one block) — else `dense`
        when it has a dense FFN role, else None. Whether a layer runs a
        dense FFN is a different question (`runs_dense_ffn`): a `moe` block
        may run one beside its experts."""
        found = None
        for node in self.nodes():
            if node.in_dispatch_zone:
                return "moe"
            if node.role == "ffn_dense":
                found = "dense"
        return found

    def runs_dense_ffn(self) -> bool:
        """Whether a layer of this invocation runs the dense FFN: a node
        outside the dispatch zone reads `d_ffn`, the dense FFN width
        (`dims.yaml`). A block can run one beside its experts (LongCat's
        two SwiGLUs per layer); a shared expert reads `d_ffn_shared` and a
        routed one `d_ffn_expert`, so neither is one."""
        return any(not node.in_dispatch_zone and "d_ffn" in ir_dim_symbols(node)
                   for node in self.nodes() if not node.children)

    def attention_call(self) -> Optional[SpecIRNode]:
        """The attention call under this invocation, or None: the first call
        that declares a cache identity (`kv_label` / `attn_variant`, copied
        onto the IR node at emit), not the one whose IR `role` says
        attention — a synthesized wrapper's attention call may carry no
        role while still declaring a family. `eval.cache.identity` builds
        its `Cache` from the same two facts."""
        return next((node for node in self.nodes() if node.children
                     and (node.kv_label is not None
                          or node.attn_variant is not None)), None)

    @property
    def site_call(self) -> Optional[SpecIRNode]:
        """The call an isolated component's stage makes at its site
        (provenance `site`), the template's one root; None for any other
        template."""
        if len(self.roots) == 1 and self.roots[0].provenance == "site":
            return self.roots[0]
        return None

    @property
    def shown(self) -> Tuple[SpecIRNode, ...]:
        """What a report or a drawing of this template lays out: its roots —
        except the call an isolated component's stage makes at its site
        (`site_call`). That call is the stage, not a level of it, so its
        body is shown, as it was when the component itself was the
        stage."""
        call = self.site_call
        return tuple(call.children) if call is not None else self.roots

    def output_roots(self) -> Tuple[SpecIRNode, ...]:
        """Resolve this invocation's declared forward outputs among its roots."""
        declared = tuple(self.meta.outputs)
        if not declared:
            raise ValueError(
                f"block template {self.name!r} has no declared outputs")
        by_name = {node.name: node for node in self.roots}
        missing = [str(name) for name in declared if str(name) not in by_name]
        if missing:
            raise ValueError(
                f"block template {self.name!r} declares unknown output "
                f"root(s) {missing!r}")
        return tuple(by_name[str(name)] for name in declared)


def _depth_first(nodes: Sequence[SpecIRNode]) -> Iterator[SpecIRNode]:
    for n in nodes:
        yield n
        yield from _depth_first(n.children)


@dataclass(frozen=True, kw_only=True)
class Stage(ABC):
    """One step of a stack: a model-layer declaration paired with the IR it
    expanded into.

    Abstract over WHAT it expanded into, because that is the one thing about a
    stage the walk branches on, and a `(BlockMeta, roots)` tuple discriminated
    by `isinstance(entry, tuple)` made the branch appear at every handler
    instead of once.

    `upstream` is the authored `inputs:` (or `in:`), bound to the stage's
    formals by position. Edges between model stages are declared; the
    evaluator does not invent a previous stage. A stage that reads what the
    model receives — token ids, an encoder's image patches — names a model
    input (`Program.model_inputs`).

    `carry` is set on a block stage whose stack carries a stream across its
    block boundary: then formal 0 reads that stream's current value rather
    than `upstream[0]`, which is only its initial value.

    `entry` is the model layer the stage runs, as the spine copied it
    (`SpecIRTree.spine`); `layer_path` is that layer's identity. The stages
    one layer expands into share it.
    """

    entry: SpineEntry
    stack: str = ""                 # "" = the main backbone
    aux_kind: Optional[str] = None
    upstream: Tuple[str, ...] = ()
    carry: Optional["Carry"] = None

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def layer_path(self) -> Tuple[int, ...]:
        return tuple(self.entry.layer_path)

    @property
    @abstractmethod
    def roots(self) -> Sequence[SpecIRNode]:
        """The IR nodes this stage evaluates."""


@dataclass(frozen=True, kw_only=True)
class OpStage(Stage):
    """A top-level op — an embedding, a final norm, an `lm_head`.

    `prelude` holds the nodes a whole-stream read of the op runs before it
    (`Program.lowered`): the crops and the join of a stream it names with
    `axis:`. Empty in a declared program."""

    node: SpecIRNode
    prelude: Tuple[SpecIRNode, ...] = ()

    @property
    def roots(self) -> Sequence[SpecIRNode]:
        return (*self.prelude, self.node)


@dataclass(frozen=True, kw_only=True)
class BlockStage(Stage):
    """A block template, evaluated once and multiplied by its layer count."""

    template: BlockTemplate

    @property
    def roots(self) -> Sequence[SpecIRNode]:
        return self.template.roots


@dataclass(frozen=True)
class Carry:
    """The stream a stack carries across its block boundary
    (`StreamSpec.carries`), as the stages of that stack see it.

    `owner` names the model layer that declares it — qualified by its
    enclosing group, because an MTP head's `layers` and the backbone's
    `layers` are two stacks — so two stacks that both call their stream
    `hidden` hold two values. A carried stream settles its layout at every
    iteration exit.
    """

    owner: str
    stream: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.owner, self.stream)


@dataclass(frozen=True)
class Contribution:
    """One root of the model total, and how many of it there are.

    `count` is 1 for a top-level entry and the template's layer count for a
    block root. It is the multiplicity of the ROOT, not of the node — every
    `repeat:` beneath it is already folded into the walk's own per-node counts,
    which is why applying this twice would be wrong and why it lives on the
    root rather than on every node.

    `block_index` is the root's position in `Program.blocks`, and is None for a
    top-level entry. It is here because `count` is a layer count, and a
    consumer that has to re-count for a PIPELINE STAGE needs to know WHICH
    layers — a property of the block INVOCATION, not of the root.

    It is an ordinal rather than the template's name because a name does not
    identify an invocation: GLM-5.1 declares the same MoE template twice, once
    with 75 layers and once empty, and a consumer keyed by name gave both
    copies the whole stack's share — inflating the model's cache by 1.96x. The
    ordinal is well defined because `SpecTree` appends one `ir.blocks` entry
    per `iter_blocks(resolved)` record, in that order, so index `i` here and
    index `i` there are the same invocation by construction.
    """

    node: SpecIRNode
    count: int
    stack: str
    block_index: Optional[int] = None


@dataclass(frozen=True)
class Stack:
    """A sequence of stages sharing a checkpoint namespace and a cost bucket.
    The main backbone is the one no `group: true` entry opens.

    `is_aux` is that declaration: the stack is not the backbone's index
    space. It is a fact of structure, read off the `group:` that opens the
    stack, and independent of what `aux_kind` names the stack as. A pipeline
    partition cuts the backbone's index space and places an aux stack whole
    (`layout.stages.StagePlacement`); which way it is placed follows from its
    edges, never from its kind."""

    name: str
    aux_kind: Optional[str]
    stages: Tuple[Stage, ...]
    # Where this stack attaches, declared on the `group:` entry itself:
    # DeepSeek's MTP head says `inputs: [norm]`, so it reads the backbone's
    # final hidden state. A vision tower declares nothing, because its input is
    # an image and not a hidden state — which is the whole difference between
    # the two, and it is already written down.
    upstream: Tuple[str, ...] = ()
    is_aux: bool = False
    #: Whether the stack is a modality encoder's (`ArchSpec.encoder_groups`):
    #: opened by an auxiliary group that reads a media model input and names
    #: no backbone tensor.
    encoder: bool = False
    #: The stack whose module this one runs again (its group is a call,
    #: `ModelLayer.call`: a model's videos through the tower its images
    #: run); it holds no weights of its own. None for a stack that owns its weights.
    reruns: Optional[str] = None

    @staticmethod
    def display_name(name: str) -> str:
        """A stack name as a message shows it. The main backbone's name is
        `""` (`Stage.stack`), which prints as nothing; it is `backbone`."""
        return name or "backbone"

    def walk_order(self) -> Tuple[Stage, ...]:
        """The stages in the order the walk evaluates them: as declared,
        except that the variants of one carrying stack are ordered by the
        first iteration each covers after resolution. That is the order the
        carried stream passes through them: each hands its exit to the next.
        """
        out: List[Stage] = []
        placed: set = set()
        for stage in self.stages:
            if stage.carry is None:
                out.append(stage)
                continue
            if stage.carry.owner in placed:
                continue
            placed.add(stage.carry.owner)
            chain = [st for st in self.stages
                     if st.carry is not None
                     and st.carry.owner == stage.carry.owner]
            out.extend(sorted(chain, key=_first_iteration))
        return tuple(out)

    def spanning(self, stage: "BlockStage") -> Optional["BlockStage"]:
        """Another variant of `stage`'s chain that occurs both before and
        after `stage`'s one iteration, if any: the first-to-last span of its
        resolved iterations strictly contains that iteration."""
        (at,) = stage.template.layer_indices
        for other in self.stages:
            if (other is stage or other.carry is None
                    or other.carry.owner != stage.carry.owner):
                continue
            indices = other.template.layer_indices
            if min(indices) < at < max(indices):
                return other
        return None

    @property
    def is_encoder(self) -> bool:
        """An aux stack that starts from its own modality's input — an image,
        a waveform — rather than from the backbone's hidden state.

        Read off the declaration (`ArchSpec.encoder_groups`: the group reads
        a media model input and names no `inputs:`), never off `aux_kind` or
        a name: a module of the language model's step — DeepSeek-V4's
        compressors and indexers, a draft head — is no encoder.
        Its token axis is the modality's own sequence, not the text context,
        so the text model's sequence-splitting deployments (context
        parallelism, sequence parallelism) do not reach it: serving
        frameworks run a multimodal encoder tensor-parallel within its stage
        and nothing else — vLLM through `mm_encoder_tp_mode` on the TP group,
        its prefill context parallelism partitioning only the text batch
        (the encoder path is separate from it), sglang on its whole TP group.
        `schema.model.ModelLayer` holds the declaration to this: an encoder
        group names no `inputs:`, a head group must.
        """
        return self.encoder

    @property
    def reads_hidden_state(self) -> bool:
        """Whether this stack consumes a hidden state the backbone produced.

        The declaration, not the kind. A head — MTP, speculative decoding —
        predicts from the backbone's hidden state and names the node it reads;
        an encoder — vision, audio — takes an image or a waveform and names
        nothing, because there is nothing upstream of it in this graph.

        This used to answer from `aux_kind`, which is a different question:
        `aux_kind` says what the stack IS, and every consumer here wants to know
        what it READS. The two agreed on kind and disagreed on node, so the
        actual attachment point was still synthesised — by searching the
        backbone for its last output at the residual's width. On LongCat that
        search returned `mtp.norm`, the head's own output norm, and the head
        spent every evaluation reading a tensor it produces itself.
        """
        return bool(self.upstream)


def _consumes(nodes: Sequence[SpecIRNode], name: str) -> bool:
    """Whether some op under `nodes` reads `name`, a formal of their scope,
    bound as the walk binds an unresolved ref (`Scope.read`).

    A component call opens a frame whose aliases are what the call hands in
    (`Scope.calling`): each formal of the callee, and the caller-side name
    the tree grafted its reads onto (`InputRef.grafted`). Inside it a
    grafted read of `name` is the formal when the call handed `name` in; a
    read the tree did not graft is a stream, however it is spelled. A call
    that lists `name` and whose body reads nothing from it hands it on
    without reading it."""
    return _frame_reads(nodes, {name: True}, in_call=False)


def _frame_reads(nodes: Sequence[SpecIRNode], aliases: Mapping[str, bool],
                 in_call: bool) -> bool:
    """Whether an op under `nodes` reads an alias marked True in `aliases`
    (the frame's aliases: True for the formal being followed); `in_call`
    as `Scope.in_call`."""
    def carries(ref) -> bool:
        if ref.src is not None or (in_call and not ref.grafted):
            return False
        return aliases.get(ref.external_sid, False)

    for node in nodes:
        refs = node.inputs_obj or ()
        if not node.is_component:
            if any(carries(ref) for ref in refs):
                return True
            continue
        handed = [carries(ref) for ref in refs]
        if not any(handed):
            continue
        inner: Dict[str, bool] = {}
        # As the component frame binds them: per position the callee's
        # formal, then the caller-side name; a later binding wins.
        for formal, ref, value in zip(node.formals, refs, handed,
                                      strict=True):
            inner[formal] = value
            if ref.external_sid:
                inner[ref.external_sid] = value
        if _frame_reads(node.children, inner, in_call=True):
            return True
    return False


@dataclass(frozen=True)
class LayerRead:
    """One read a model layer makes of what another produces, resolved to
    the layers it names by position (`layer_reads`).

    `reader` and `sources` are `layer_path`s; `sources` is empty for a read
    of a model input, and holds more than one layer when one name reaches
    several (a group and the encoder stage it shares its name with).
    `declaration` is where the read is written (`inputs`, `streams`,
    `reads_from`); `stream` names the stream it reads, None for a read of a
    layer's output or a model input. `what` says it in a sentence."""
    reader: Tuple[int, ...]
    sources: Tuple[Tuple[int, ...], ...]
    name: str
    declaration: str
    stream: Optional[str]
    what: str


def layer_reads(ir: SpecIRTree, ir_spec: Any) -> Tuple[LayerRead, ...]:
    """Every read between model layers, each resolved to the layers it
    names by position: a name in `inputs:` in the scope it is read in
    (`ArchSpec.layers_named` — an MTP head's `layers` is its own stage, not
    the backbone's), else the final value of an inner stream of a stack in
    scope, else an appended stream to the stack that appends to it, else a
    model input; an appended stream a block's stack reads (`read: each` /
    `concat`) to that stack; an inner stream's `seed` and a `reads_from`
    that names another layer to that layer. A layer reading its own earlier
    iterations is no read between layers."""
    label: Dict[Tuple[int, ...], str] = {}

    def walk(entries, prefix: str) -> None:
        for entry in entries:
            here = tuple(entry.layer_path)
            label[here] = f"{prefix}/{entry.name}" if prefix else entry.name
            walk(entry.children, label[here])

    walk(ir.spine, "")
    out: List[LayerRead] = []

    def appended(at: Tuple[int, ...], ref: str, declaration: str,
                 who: str) -> bool:
        stream = ir.streams.get(ref)
        if stream is None:
            return False
        source = tuple(stream.layer_path)
        out.append(LayerRead(
            at, (source,), ref, declaration, ref,
            f"{who} reads stream {ref!r}, which {label[source]!r} "
            f"appends to"))
        return True

    def named(at: Tuple[int, ...], ref: str) -> None:
        who = label[at]
        found = ir_spec.layers_named(at, ref)
        if found:
            out.append(LayerRead(at, tuple(found), ref, "inputs", None,
                                 f"{who!r} reads {ref!r}"))
            return
        stack = ir_spec.inner_stream_named(at, ref)
        if stack is not None:
            out.append(LayerRead(
                at, (stack,), ref, "inputs", ref,
                f"{who!r} reads the final value of stream {ref!r}, which "
                f"{label[stack]!r} writes"))
            return
        if appended(at, ref, "inputs", repr(who)):
            return
        if ref in ir_spec.model_inputs:
            out.append(LayerRead(at, (), ref, "inputs", None,
                                 f"{who!r} reads model input {ref!r}"))

    def visit(entries) -> None:
        for entry in entries:
            at = tuple(entry.layer_path)
            for ref in entry.inputs:
                named(at, ref)
            if entry.kind in ("block", "assignments"):
                spine = ir.stack_reads.get(at)
                if spine is not None:
                    for ref in (*spine.each,
                                *(n for n, _axis in spine.concat)):
                        appended(at, ref, "streams", repr(label[at]))
            visit(entry.children)

    visit(ir.spine)
    top_paths = {node.uid: tuple(node.layer_path) for node in ir.top_level}
    for (reader, layer, name), (source, at) in ir.stream_reads.items():
        if at == STACK_ENTRY:
            seed = top_paths[ir.stream_writes[(source, at, name)]]
            out.append(LayerRead(
                reader, (seed,), name, "streams", name,
                f"layer {layer} of {label[reader]!r} reads stream {name!r} "
                f"at its entry, the `seed` {label[seed]!r}"))
        elif source != reader:
            out.append(LayerRead(
                reader, (source,), name, "reads_from", name,
                f"layer {layer} of {label[reader]!r} reads stream {name!r} "
                f"from layer {at} of {label[source]!r}"))
    return tuple(out)


def _dependency_ordered(stacks: Tuple["Stack", ...],
                        ir: SpecIRTree, ir_spec: Any) -> Tuple["Stack", ...]:
    """Stacks in an order where every cross-stack read is already produced.

    A stage's `inputs:` may name a stage of ANOTHER stack (Kimi's decoder
    reads the vision tower's output), a stack may read an appended
    stream another produces (Qwen3-VL's decoder reads the DeepStack
    mergers'), and a reader's `reads_from` may name another stack's layer
    (`SpecIRTree.stream_reads`); evaluating stacks in spec order
    evaluates the whole backbone first and the read resolves to nothing.

    Every read resolves to the layers it names (`layer_reads`). Stable
    topo-sort on the stacks that run those layers — ties keep the spec's
    own order. Stacks that read each other have no order; the cycle is
    refused, naming each read in it.
    """
    # The stack each layer runs in; a group's path, and each enclosing
    # one, is its stages' stack.
    stack_at: Dict[Tuple[int, ...], int] = {}
    for i, st in enumerate(stacks):
        for stage in st.stages:
            path = stage.layer_path
            for depth in range(len(path), 0, -1):
                stack_at.setdefault(path[:depth], i)
    # Per reading stack, each stack it reads and every read that says so
    # (declaration, read): two stacks may be joined by several reads.
    deps: List[Dict[int, List[Tuple[str, str]]]] = [{} for _ in stacks]
    for found in layer_reads(ir, ir_spec):
        i = stack_at.get(found.reader)
        if i is None:
            continue        # a layer no stage runs (fused, or absent)
        owners = sorted({stack_at[path] for path in found.sources
                         if path in stack_at})
        if len(owners) > 1:
            raise SpecRefusal(
                f"{found.what}, which names a layer in each of the stacks "
                f"{[Stack.display_name(stacks[j].name) for j in owners]}",
                declaration="inputs",
                fix=(f"rename one of the layers called {found.name!r}: a "
                     f"name in `inputs:` reads one layer"))
        for j in owners:
            if i != j:
                reads = deps[i].setdefault(j, [])
                if (found.declaration, found.what) not in reads:
                    reads.append((found.declaration, found.what))
    out = _topo_order(stacks, [set(found) for found in deps])
    if out is not None:
        return out
    cycle = _a_cycle(deps)
    steps = [step for i, j in zip(cycle, (*cycle[1:], cycle[0]), strict=True)
             for step in deps[i][j]]
    declarations = [declaration for declaration, _what in steps]
    declaration = ("reads_from" if "reads_from" in declarations
                   else declarations[0])
    names = " -> ".join(Stack.display_name(stacks[i].name)
                        for i in (*cycle, cycle[0]))
    raise SpecRefusal(
        f"stacks {names} read each other, so no order evaluates every "
        f"source before its reader: "
        + "; ".join(what for _declaration, what in steps),
        declaration=declaration,
        fix=(f"break the cycle at one of its reads: in `{declaration}`, "
             f"name a layer of a stack that does not read the reading one"))


def _a_cycle(deps: Sequence[Mapping[int, Sequence[Tuple[str, str]]]]
             ) -> Tuple[int, ...]:
    """One cycle of the dependency graph `deps` (stack -> stacks it reads,
    with the reads), in reading order: each stack reads the next, the last
    reads the first. A cycle through a `reads_from` read when there is
    one, so the refusal names that declaration. Called only when the
    topo-sort found a cycle: stripping every stack that reads none of the
    rest leaves stacks that each read another of them."""
    alive = set(range(len(deps)))
    while True:
        free = {i for i in alive if not set(deps[i]) & alive}
        if not free:
            break
        alive -= free

    def path(start: int, goal: int) -> Optional[List[int]]:
        """The stacks from `start` to `goal` along reads, shortest first."""
        trails = {start: [start]}
        frontier = [start]
        while frontier:
            nxt: List[int] = []
            for i in frontier:
                if i == goal:
                    return trails[i]
                for j in sorted(set(deps[i]) & alive):
                    if j not in trails:
                        trails[j] = [*trails[i], j]
                        nxt.append(j)
            frontier = nxt
        return None

    edges = sorted((i, j) for i in alive for j in set(deps[i]) & alive)
    edges.sort(key=lambda edge: not any(
        declaration == "reads_from"
        for declaration, _what in deps[edge[0]][edge[1]]))
    for i, j in edges:
        back = path(j, i)
        if back is not None:
            return (i, *back[:-1])
    raise AssertionError("no cycle in a graph the topo-sort refused")


def _topo_order(stacks: Tuple["Stack", ...], deps: Sequence[set]
                ) -> Optional[Tuple["Stack", ...]]:
    """`stacks` with each after the ones it depends on, ties in the given
    order; None when two depend on each other."""
    deps = [set(found) for found in deps]
    out: List[Stack] = []
    done: set = set()
    ready = sorted(i for i in range(len(stacks)) if not deps[i])
    while ready:
        i = ready.pop(0)
        if i in done:
            continue
        out.append(stacks[i])
        done.add(i)
        for j in range(len(stacks)):
            if i in deps[j]:
                deps[j].discard(i)
                if not deps[j] and j not in done and j not in ready:
                    ready.append(j)
        ready.sort()
    return tuple(out) if len(out) == len(stacks) else None


def _outermost_readers(roots: Sequence[SpecIRNode],
                       name: str) -> Iterator[SpecIRNode]:
    """The nodes under `roots` that read stream `name`, outermost first: a
    component call that names it binds its body to the same tensor, and
    the body's nodes carry the caller's name for it (`iter_tree` walks a
    formal up to the caller's ref), so only the call is a reading
    position."""
    for node in roots:
        if any(ref.external_sid == name for ref in node.inputs_obj or ()):
            yield node
            continue
        yield from _outermost_readers(node.children, name)


def _first_iteration(stage: Stage) -> int:
    """Where a chain stage first occurs. Only a block stage carries a stream;
    an op stage in a carrying group (an MTP head's logits) is not a link."""
    if not isinstance(stage, BlockStage):
        raise ValueError(
            f"stage {stage.name!r} is in a carrying chain but is not a block")
    return min(stage.template.layer_indices)


@dataclass(frozen=True)
class StreamPairing:
    """One element of an appended stream and the iteration that reads it
    (`Program.stream_pairings`).

    `writer` / `reader` are model layer names and `writer_stack` /
    `reader_stack` the `Stack` each runs in ("" for the backbone).
    `writer_iteration` is the iteration of the appending stack the element
    was appended after (-1: that stack's input). `reader_port` is the formal
    the element binds when the reader names the stream in `inputs:`
    (`inner` false), or the node that reads it when a component does
    (`inner` true); `reader_uid` is the IR node: the reading stage's first
    root, or the reading node. The uid is shared by every iteration of one
    template, which is evaluated once.
    """

    stream: str
    element: int
    writer_stack: str
    writer: str
    writer_iteration: int
    reader_stack: str
    reader: str
    reader_iteration: int
    reader_port: str
    reader_uid: int
    inner: bool


@dataclass(frozen=True)
class WholeStreamPairing:
    """One reader that takes an appended stream's elements at once
    (`Program.whole_stream_pairings`), and how it holds them.

    `representation` is `history` — the elements as one `[B, S, m, D]`
    value along a history axis: the stack's own `read: all` readers, one
    row per derived group, and a model-level `op:` naming the stream with
    no `axis:` — or `concat`: the elements joined along `axis` into one
    tensor, by a later stack's `read: concat` (one row per reading node
    of each template) or a model-level `op:` that declares `axis:`.

    `reader_iterations` are the reading stack's iterations the row stands
    for (empty for a model-level op, which runs once). `visible` is how
    many elements the reader takes, m (all k for a whole read); `sources`
    are the iterations that appended them (`StreamSpec.writers`), in the
    order the reader holds them — the history's, or the join's positions
    — and `kind` says which end of those iterations they were appended at
    (`before` their entry, `after` their exit). `reader_uid` is the
    reading stage's first root for a history, the reading node for a
    join (the op itself at the model level). `through_uid` is the node a
    lowered join's reader reads the stream from (`Program.lowered`): the
    `concat`, or the crop of a single element; None for a history, a
    direct read, or a program not yet lowered.
    """

    stream: str
    writer_stack: str
    writer: str
    reader_stack: str
    reader: str
    reader_iterations: Tuple[int, ...]
    visible: int
    sources: Tuple[int, ...]
    kind: str
    reader_uid: int
    representation: str = "history"
    axis: Optional[int] = None
    through_uid: Optional[int] = None

    def element_ref(self, position: int) -> str:
        """The name a joined read's element `position` goes by in the
        lowered graph (`Program.lowered`): the input the walk binds to
        that element. Not a name any spec can write."""
        return f"{self.stream}[{position}]"


@dataclass(frozen=True)
class Headline:
    """The sizes a model card leads with (`Program.headline`): the dims the
    spec declares at model level, and the backbone's layers.

    `hidden` and `vocab` every language model declares. The rest belong to
    one family each — a KV-head count to grouped attention, `d_ffn` to a
    dense FFN, the LoRA ranks to MLA — and are None where the spec does not
    declare them; none is filled in from another dim. Which attention layers
    read which of them is `Analysis.attention_families`.

    `dense_layers` are the backbone layers whose block runs a dense FFN
    (`BlockTemplate.runs_dense_ffn`), alone or beside its experts, read off
    the layer groups the spec declares — DeepSeek's `first_k_dense_replace`
    prefix and Step's or MiMo's per-layer MoE lists alike. `intermediate` is
    the dense FFN's width: None for a model with no dense FFN layer,
    whatever the config's `intermediate_size` says, because no layer of it
    runs that width."""

    hidden: int
    vocab: int
    layers: int
    dense_layers: Tuple[int, ...]
    heads: Optional[int]
    kv_heads: Optional[int]
    intermediate: Optional[int]
    head_dim: Optional[int]
    kv_lora_rank: Optional[int]
    q_lora_rank: Optional[int]
    qk_rope_dim: Optional[int]


@dataclass(frozen=True)
class ModelInput:
    """A model input as the walk seeds it (`Evaluator._model_input`): the
    spec's declaration (`ModelInputSpec`) and the encoder stack that reads it
    (`Stack.is_encoder`; None for the token ids, which the language model
    reads), whose own view places it.

    An isolated component's stage receives what its call in the model hands
    it (`CallSite.formals`): `shape` is that tensor, every axis an
    expression in the frame's dims, and `kind` and `width` say nothing."""

    name: str
    kind: Optional[str]
    width: Optional[str]
    shape: Optional[Tuple[Any, ...]] = None
    #: A media input's axes: items per request, then one item's extents.
    axes: Tuple[str, ...] = ()
    #: The axes the config fixes (`ModelInputSpec.fixed`), each resolved
    #: once against the config's dims: axis -> number.
    fixed: Mapping[str, Any] = field(default_factory=dict)
    per_item_of: Optional[str] = None
    stack: Optional[str] = None
    #: The least extent one item's axis takes: one resize factor of the
    #: image's `patchify` (`align`) per side, one frame of an audio clip.
    least: Mapping[str, Any] = field(default_factory=dict)
    #: Why no walk reads this input (`ModelInputSpec.unmodeled`); a
    #: request states its item count, and any item is refused.
    unmodeled: Optional[str] = None

    @property
    def encoder(self) -> bool:
        return self.stack is not None

    def media_values(self, media: Mapping[str, Mapping[str, Any]]
                     ) -> Dict[str, Any]:
        """This input's axes as the workload gives them (`Workload.media`):
        exactly its axes the config does not fix, each a whole number; the
        fixed ones from the config. An input the workload does not name
        sends no item this step — its item count is 0."""
        fixed = dict(self.fixed)
        given = media.get(self.name)
        if given is None:
            # No item: its extents are the least an item takes, which the
            # zero count multiplies away (only an item-free fact — a weight,
            # an output's width — survives).
            return {axis: (fixed[axis] if axis in fixed
                           else 0 if axis == self.axes[0]
                           else self.least[axis])
                    for axis in self.axes}
        if not isinstance(given, Mapping):
            raise ValueError(
                f"media[{self.name!r}] is {given!r}; give one value per axis "
                f"of {list(self.axes)}, e.g. "
                f"{ {a: '<n>' for a in self.axes if a not in fixed} }")
        free = [axis for axis in self.axes if axis not in fixed]
        named_fixed = sorted(set(given) & set(fixed))
        if named_fixed:
            raise ValueError(
                f"media[{self.name!r}] gives {named_fixed}, which the config "
                f"fixes ({ {a: fixed[a] for a in named_fixed} }); leave them "
                f"out")
        unknown = sorted(set(given) - set(self.axes))
        missing = [axis for axis in free if axis not in given]
        if unknown or missing:
            raise ValueError(
                f"media[{self.name!r}] gives {sorted(given)}; this input's "
                f"axes are {free}"
                + (f" (missing {missing})" if missing else "")
                + (f" (unknown {unknown})" if unknown else ""))
        out: Dict[str, Any] = {}
        for axis in self.axes:
            if axis in fixed:
                out[axis] = fixed[axis]
                continue
            value = given[axis]
            if isinstance(value, (list, tuple)):
                raise ValueError(
                    f"media[{self.name!r}][{axis!r}] is a list; one input "
                    f"takes one size per step (items of different sizes are "
                    f"not modelled yet)")
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                raise ValueError(
                    f"media[{self.name!r}][{axis!r}] = {value!r}; give a "
                    f"whole number")
            out[axis] = value
        return out


@dataclass(frozen=True)
class Handoff:
    """An encoder output the language model reads (`Program.handoffs`):
    the stage or appended stream `name` of encoder stack `stack`, filling
    (`defines`) or adding onto the prompt positions of media input
    `domain`. A stream hands over one tensor per element."""

    name: str
    stack: str
    domain: str
    defines: bool
    stream: bool


@dataclass(frozen=True)
class HandedTensor:
    """One tensor an encoder hands the language model, per request: `rows`
    rows of `width` values. The language model reads it as `[B, rows,
    width]`; the encode walk at one request is where both numbers come
    from (`transforms.Deployment.handed_over`)."""

    rows: int
    width: Any
    #: Bytes per value: the producing leaf's activation format under the
    #: deployment's precision (`traffic.produced_width`).
    value_bytes: float


@dataclass(frozen=True)
class LookupSite:
    """One leaf's row gather, resolved to the graph (`Program.lookups`)."""
    #: The input port its ids arrive on.
    ids: int
    #: The uid holding the table it gathers from.
    table: int


@dataclass(frozen=True)
class Program:
    """Spec + walked graph. Immutable, shareable, and cheap to hold:
    `iter_tree` is memoised on the ResolvedSpec."""

    resolved: Any                     # ResolvedSpec — `loader` imports this way
    ir: SpecIRTree
    #: Which stacks the walk evaluates. None is the language model — every
    #: stack but the encoders, as prefill and decode run it; an encoder
    #: stack's name is that encoder alone, as the encode phase runs it
    #: (`encoding`); `CHECKPOINT` is every stack, the language model in its
    #: phase and each encoder in its own (`checkpoint`) — what a question
    #: about the whole checkpoint, its tensors and its weights, walks. The
    #: graph is the same; only the schedule differs.
    schedule: Optional[str] = None
    CHECKPOINT: ClassVar[str] = "*"

    @classmethod
    def of(cls, resolved: Any, ir: Optional[SpecIRTree] = None) -> "Program":
        return cls(
            resolved=resolved,
            ir=ir if ir is not None else iter_tree(resolved),
        )

    def encoding(self, stack: str) -> "Program":
        """This program as the encode phase walks it: encoder stack `stack`
        alone, from its own media inputs."""
        encoders = sorted(st.name for st in self.stacks if st.is_encoder)
        if stack not in encoders:
            raise ValueError(
                f"{self.spec.name}: {stack!r} is not an encoder stack of this "
                f"model; its encoders are {encoders}")
        return Program(resolved=self.resolved, ir=self.ir, schedule=stack)

    def checkpoint(self) -> "Program":
        """This program walking every stack, each in its own phase."""
        return Program(resolved=self.resolved, ir=self.ir,
                       schedule=self.CHECKPOINT)

    @property
    def encode_stack(self) -> Optional[str]:
        """The encoder an encode-phase program walks alone, else None."""
        return (None if self.schedule in (None, self.CHECKPOINT)
                else self.schedule)

    @property
    def walks_language_model(self) -> bool:
        return self.schedule in (None, self.CHECKPOINT)

    @property
    def spec(self) -> Any:
        return self.resolved.spec

    def block_label(self, template: BlockTemplate) -> str:
        """The short name `template` goes by (`MLA + MoE FFN`): the one the
        architecture diagram's block cards show, so a block reads the same
        on every page; its full template name stays its identity
        (`BlockMeta.short_name`)."""
        return template.meta.short_name

    @property
    def architectures(self) -> Tuple[str, ...]:
        """The checkpoint's `config.architectures`: the model class a serving
        framework's model registry dispatches on (sglang's
        `Qwen3MoeForCausalLM`)."""
        config = self.resolved.config
        if not isinstance(config, dict):
            return ()
        return tuple(str(n) for n in config.get("architectures") or ())

    def scan_chunk_size(self, fields: Tuple[str, ...]) -> Optional[int]:
        """The Mamba-2 scan chunk the checkpoint config states, at the first
        of `fields` it has — the order a serving framework reads them in,
        which the caller holds and passes as plain names
        (`FrameworkBinding.ssm_chunk_size_from`: vLLM and sglang read
        `mamba_chunk_size`, then `chunk_size`). It is a kernel argument, not
        a graph fact, so no dim binds it. None when the config states none
        of them: no framework fallback stands in. A field that is present
        but not a positive integer is refused, not skipped."""
        config = self.resolved.config
        if not isinstance(config, dict):
            return None
        for name in fields:
            value = config.get(name)
            if value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"the checkpoint config's `{name}` (the Mamba-2 scan "
                    f"chunk) is {value!r}, not a positive integer")
            return int(value)
        return None

    def is_for(self, resolved: Any) -> bool:
        """Whether this program was walked for `resolved`. The eval walk folds
        one shared CLONE (memoised on the ResolvedSpec), so the program's
        `resolved` is never the caller's OBJECT — but the clone keeps the same
        underlying `config`. Identity on config catches a program built for a
        DIFFERENT model without rejecting every walk-produced one. Kept here so
        callers ask the program rather than reach into `resolved.config`."""
        return self.resolved.config is resolved.config

    @memo_property
    def lookups(self) -> Dict[int, "LookupSite"]:
        """Leaf uid → the row gather it makes (`OpSpec.lookup`), resolved
        to this graph: the port its ids arrive on, and the uid holding the
        table — the leaf itself for a lookup into its own weight, else the
        stored parameter on its declared table port. Every reader of the
        fact asks here: the model-input check (the ids port reads ids), the
        storage kind (`as_stored`: a gathered tensor is a TABLE), and the
        traffic law (`traffic._table_read_fractions`: the rows read)."""
        out: Dict[int, LookupSite] = {}
        for node in self.nodes():
            if node.children:
                continue
            lookup = lookup_of(node.declaration)
            if lookup is None:
                continue
            # Both ports are inside the op's arity (`Lookup.check`), and the
            # loader refuses a node wired with fewer inputs than that.
            table = node.uid
            if not lookup.owns_table:
                src = tuple(node.inputs_obj)[lookup.table].src
                if src is None or not is_identity_source(src.op):
                    raise SpecRefusal(
                        f"{self.spec.name}: {node.display_path} ({node.op}) "
                        f"gathers rows from the table on input "
                        f"{lookup.table}, which reads "
                        f"{self._read_by(node, lookup.table)}, not a stored "
                        "parameter",
                        declaration="in",
                        fix=f"pass the table as input {lookup.table} of "
                            f"{node.display_path}: a `parameter_tensor` "
                            "with `shape: [<rows>, <row width>]`",
                        node=node.display_path)
                table = src.uid
            out[node.uid] = LookupSite(ids=lookup.ids, table=table)
        return out

    @memo_property
    def host_offloadable(self) -> FrozenSet[int]:
        """The uids whose stored weight a serving framework can keep in host
        memory (`GraphNode.host_offload`): DeepSeek-V4.1's Engram and
        Qwen's PLE n-gram tables under vLLM's Engram `cpu_offload`. A
        deployment that offloads leaves them out of the GPU's weights. The
        walk refuses the declaration on a node that stores nothing."""
        return frozenset(
            node.uid for node in self.nodes() if node.host_offload is not None)

    @memo_property
    def read_aliases(self) -> Dict[int, Dict[int, Fraction]]:
        """Node uid → the uids whose STORED tensors it reads, when they are
        not its own, each with the share its declared weight is of the one
        stored (1 for a tied LM head; half of `kv_b_proj` for each absorbed
        MLA GEMM at that `module:`). A tied LM head executes a full vocab
        projection every step, but the tensor it streams is the embedding's
        — one stored copy, two readers. Storage stays single (the
        checkpoint's own account); the READ side bills whoever moves the
        bytes.

        The fact is the tie resolved at build (`SpecIRNode.tied_to_uid`).
        Iterated, a value is the uids, in the shape `Rewrite.origins` gives
        a fused node — "my weights are those nodes'" — so traffic consults
        both through one lookup.
        """
        aliases: Dict[int, Dict[int, Fraction]] = {}
        for node in self.nodes():
            if node.tied_to is None:
                continue
            if node.tied_to_uid is None:
                raise ValueError(
                    f"{self.spec.name}: {node.display_path} declares "
                    f"`tied_weight.to: {node.tied_to}` but the tree was never "
                    f"finalized (`finalize_ir_tree` resolves it)")
            mine, held = (
                math.prod(ir_to_leaf(n, "", self.spec.custom_ops or {}).shape)
                for n in (node, self.by_uid(node.tied_to_uid)))
            aliases[node.uid] = {node.tied_to_uid: Fraction(mine, held)}
        return aliases

    @memo_property
    def embeddings_tied(self) -> bool:
        """Whether an input embedding table and a projection share one
        stored weight (HF's `tie_word_embeddings`): some `read_aliases`
        edge joins a lookup table (`lookups`) to a node that is not one.
        An MTP head reading the LM head's weight is a shared weight too,
        but neither end gathers rows, so it does not tie the embeddings."""
        tables = {site.table for site in self.lookups.values()}
        return any(
            (reader in tables) != (target in tables)
            for reader, targets in self.read_aliases.items()
            for target in targets)

    # -- the base dim binding --------------------------------------------

    @memo_property
    def base_binding(self) -> Binding:
        """The spec's dims, as the root frame. Nothing added.

        What an unstated dim means is a property of the spec LANGUAGE, not of
        whoever is reading it, so it is settled in `dim_resolution.resolve_dimensions`
        and arrives already applied. This class used to carry its own copy of
        that table; so does `SpecEngine.__init__`. Three tables meant the same
        model could present a different dim environment depending on which
        consumer asked — and two of the entries in this one were provably dead
        on the whole corpus, because the loader had already bound them.

        None of them are marked lookup-derived. A default that COPIES another
        symbol materialises its value once and thereafter stands alone, so
        sharding it divides it exactly once, which is what lets expert
        parallelism take `n_expert_copies` while the routing denominator
        `n_experts` — never in any policy — stays whole.

        The root frame is TRACKED: every dim is a `Dim` carrying its symbol
        beside its number, and every child bind inherits the wrapping
        (`Binding._identify`). This is what lets a `params_fn`'s arithmetic
        answer `C·s_conv_kernel` and `49152` at once — the projection already
        splits the two (`expr_of` / `value_of`), and a consumer that needs
        the number coerces, so nothing downstream has to pick.
        """
        return Binding(_tracked=True).child_map(dict(self.resolved.dim_values))

    # -- structure ------------------------------------------------------

    @memo_property
    def model_inputs(self) -> Mapping[str, ModelInput]:
        """The spec's `model_inputs:`, each with the encoder stack that reads
        it. `ArchSpec._check_model_inputs` holds every input to at least one
        reader, the token ids to the language model and a media input to the
        encoders; the readers' own declarations are held to it as the walk
        binds them (`check_model_input_reader`).

        An isolated component (`ResolvedSpec.site`) receives its formals,
        as its call in the model hands them over (`CallSite.formals`); that
        model's program held them to their readers. A call in an encoder
        keeps its encoder stack (`isolate_resolved`)."""
        site = self.resolved.site
        if site is not None:
            names = self.spec.get_block(site.component).inputs
            stack = site.stack if site.encoder is not None else None
            return {name: ModelInput(name=name, kind=None, width=None,
                                     stack=stack, shape=formal)
                    for name, formal in zip(names, site.formals, strict=True)}
        out: Dict[str, ModelInput] = {}
        dims = dict(self.resolved.dim_values)
        readers = self.spec.model_input_readers()
        for name, entry in self.spec.model_inputs.items():
            encoders = self.spec.encoder_groups()
            stack = next((group.name for _layer, group in readers[name]
                          if group is not None and group.name in encoders),
                         None)
            least: Dict[str, Any] = {}
            if entry.kind == "audio":
                least = {entry.axes[1]: 1}
            align = self._cut_align(name)
            if align is not None:
                # A video's frames: at least one; its sides, as an
                # image's, whole resize factors.
                least = {axis: (1 if (entry.kind == "video"
                                      and axis == entry.axes[1])
                                else align)
                         for axis in entry.axes[1:]}
            out[name] = ModelInput(
                name=name, kind=entry.kind, width=entry.width,
                axes=tuple(entry.axes),
                fixed={axis: eval_dim_expr(expr, dims)
                       for axis, expr in entry.fixed.items()},
                per_item_of=entry.per_item_of, stack=stack, least=least,
                unmodeled=entry.unmodeled)
        return out

    def _cut_align(self, name: str) -> Optional[Any]:
        """The resize factor (`align`) of the processor's cut reading model
        input `name` (`processor_step: cut`, `patchify`), as the tree bound
        it on that node; None when no cut reads it. Two cuts of one input
        at different factors leave its least item undecided: refused."""
        found = {value_of(node.dims["align"]) for node in self.nodes()
                 if node.processor_step is ProcessorStep.CUT
                 and any(ref.src is None and ref.external_sid == name
                         for ref in node.inputs_obj or ())}
        if len(found) > 1:
            raise ValueError(
                f"model input {name!r} is cut into patches at resize factors "
                f"{sorted(map(str, found))}; one input is cut one way")
        return found.pop() if found else None

    @memo_property
    def walked_inputs(self) -> Mapping[str, ModelInput]:
        """The model inputs this program's walk reads: the token ids for the
        language model, an encoder's own media for that encoder."""
        walked = self.walked_stack_names
        return {name: entry for name, entry in self.model_inputs.items()
                if entry.unmodeled is None
                and (entry.stack in walked if entry.stack is not None
                     else self.walks_language_model)}

    @memo_property
    def handoffs(self) -> Tuple["Handoff", ...]:
        """Every edge from an encoder into the language model, as the spec
        declares it (`ArchSpec.encoder_edges`, `prompt_slots`): the name the
        language model reads, the encoder stack producing it, and which
        prompt positions it fills. The one map the prompt length and the
        handed-over tensors are both read from."""
        out: Dict[str, Handoff] = {}
        for layer, ref, stream, group in self.spec.encoder_edges():
            slots = (layer.streams[stream].prompt_slots if stream is not None
                     else layer.slots_for(ref))
            out.setdefault(ref, Handoff(
                name=ref, stack=group.name, domain=slots.domain,
                defines=slots.defines, stream=ref in self.ir.streams))
        return tuple(out.values())

    def check_model_input_reader(
            self, node: SpecIRNode,
            bound: Mapping[Tuple[int, int], Any],
            arity: Mapping[int, int]) -> None:
        """What `node` declares it takes from a model input, held to what
        the model input declares, as the walk bound its inputs (`bound`:
        `Walk.input_references`, each a reference naming the model input
        it reads, if any; `arity`: how many inputs the walk bound per node —
        none for a source op, whatever edge a caller wired to it).

        A table lookup reads ids: its id input is a model input with no
        width (token ids, audio codebook ids), followed through the
        component formals that pass it. Any other reader that declares the
        rank and row width it takes on that port (a patch projection's
        `in`) takes the model input's own: `model_inputs.width` is what the
        tensor is, and a reader that says otherwise is a wrong edge or a
        wrong width.

        An isolated component's inputs are its call's formals
        (`ResolvedSpec.site`), which the model's own walk held to their
        readers: nothing is checked again."""
        if self.resolved.site is not None:
            return
        inputs = self.model_inputs
        read = arity[node.uid]
        site = self.lookups.get(node.uid)
        if site is not None:
            origin = self._ids_read(node, site.ids, bound, arity)
            if origin is None or inputs[origin].width is not None:
                what = (f"model input {origin!r} of width "
                        f"{inputs[origin].width!r}" if origin is not None
                        else self._read_by(node, site.ids))
                raise SpecRefusal(
                    f"{self.spec.name}: {node.display_path} "
                    f"({node.op}) looks ids up in a table, and its id "
                    f"input reads {what}",
                    declaration="inputs",
                    fix="read a model input of ids with no width — "
                        "`input_ids: {kind: tokens}`, or an audio "
                        "codebook `{kind: audio}` — passed through the "
                        "component formals that reach the lookup",
                    node=node.display_path)
            return
        dims = dict(self.resolved.dim_values)
        for at in range(read):
            origin = bound[(node.uid, at)].model_input
            if origin is None:
                continue
            entry = inputs[origin]
            if entry.kind in ("image", "video"):
                # `patchify` reads the image or video whole; its own
                # contract holds the rank and the sides.
                continue
            takes = self._port_takes(node, at)
            if not takes:
                continue
            rank = 3 if entry.width is not None else 2
            width = (value_of(eval_dim_expr(entry.width, dims))
                     if entry.width is not None else None)
            if any(len(axes) == rank and (
                    width is None or axes[-1] is None
                    or axes[-1] == width)
                   for axes in takes):
                continue
            raise SpecRefusal(
                f"{self.spec.name}: {node.display_path} ({node.op}) "
                f"takes {[['*' if a is None else a for a in axes] for axes in takes]} "
                f"on input {at}, "
                f"which reads model input {origin!r} declared "
                f"{'[B, rows, ' + repr(entry.width) + ' = ' + str(width) + ']' if width is not None else '[B, rows]'}",
                declaration="model_inputs",
                fix=f"write `model_inputs.{origin}.width` as the width "
                    f"{node.display_path} takes in, or read the input "
                    f"that has it",
                node=node.display_path)

    def _ids_read(self, node: SpecIRNode, index: int,
                  bound: Mapping[Tuple[int, int], Any],
                  arity: Mapping[int, int]) -> Optional[str]:
        """The model input of ids input `index` of `node` reads: directly,
        or through a one-input view op that only lays the same ids out
        again (MiMo's `pad_rows`, the processor repeating a clip's last
        frame to a whole group)."""
        if index >= arity.get(node.uid, 0):
            return None
        direct = bound[(node.uid, index)].model_input
        if direct is not None:
            return direct
        refs = tuple(node.inputs_obj or ())
        src = refs[index].src if index < len(refs) else None
        decl = src.declaration if src is not None else None
        if (src is not None and not src.children
                and isinstance(decl, OpSpec) and decl.is_view_op
                and len(tuple(src.inputs_obj or ())) == 1):
            return self._ids_read(src, 0, bound, arity)
        return None

    def _read_by(self, node: SpecIRNode, index: int) -> str:
        """What input `index` of `node` names, for a refusal."""
        refs = tuple(node.inputs_obj or ())
        if index >= len(refs):
            return "nothing"
        ref = refs[index]
        if ref.src is not None:
            return f"the output of {ref.src.display_path!r}"
        return f"{ref.external_sid!r}, which is no model input"

    def _port_takes(self, node: SpecIRNode, index: int
                    ) -> Tuple[Tuple[Any, ...], ...]:
        """Each signature `node`'s op declares for input `index`, as the
        axis values its own dims give; an axis the node binds no value to is
        a type variable (`identity`'s `in`, `B`, `s_ctx`) and stands as
        None. Empty when the op declares none there."""
        decl = node.declaration
        templates = [OpRef.resolve(node.op, self).shape_in_template,
                     *(decl.alt_shape_in or ()
                       if isinstance(decl, OpSpec) else ())]
        bound = {**(node.dims or {}),
                 **({"dim": node.dim} if node.dim is not None else {})}
        out = []
        for template in templates:
            if not template:
                continue
            ports = (template if isinstance(template[0], (list, tuple))
                     else (template,))
            if index >= len(ports):
                continue
            out.append(tuple(
                value_of(bound[str(axis)]) if str(axis) in bound else None
                for axis in ports[index]))
        return tuple(out)

    def declared_stream_names(self) -> FrozenSet[str]:
        """Model-wide stream names the model declared, nested layer groups
        included. A carried stream is local to its stack and is not one."""
        names = set(self.ir.streams)
        for reads in self.ir.stack_reads.values():
            names.update(reads.inner)
        return frozenset(names)

    def component_stream_roles(self, template: BlockTemplate
                               ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Streams this template writes and names as `in:`.

        Reads come from unresolved IR refs whose names are declared
        streams — the same `in:` the walk binds. Writes are
        `stream_out` on this template and on components the IR actually
        invokes (ops already resolved; no second `<param>` parser).
        """
        declared = self.declared_stream_names()
        produces: List[str] = []
        seen_p: set = set()
        for sname, _target in template.meta.stream_out:
            produces.append(sname)
            seen_p.add(sname)
        reads: List[str] = []
        seen_r: set = set()

        def walk(nodes: Sequence[SpecIRNode]) -> None:
            for node in nodes or ():
                for sname, _target in node.stream_out:
                    if sname not in seen_p:
                        seen_p.add(sname)
                        produces.append(sname)
                for ref in node.inputs_obj or ():
                    if ref.src is not None:
                        continue
                    sname = ref.external_sid
                    if sname in declared and sname not in seen_r:
                        seen_r.add(sname)
                        reads.append(sname)
                walk(node.children)

        walk(template.roots)
        return tuple(produces), tuple(reads)

    def output_leaf(
        self,
        node: SpecIRNode,
        port: str = "out",
    ) -> Tuple[SpecIRNode, str]:
        """Resolve a component output to the primitive tensor producing it.

        Component references name the wrapper while execution and liveness
        rows name primitives. Follow the component's declared `out` or named
        output recursively so consumers join those two structural
        views without guessing from traversal order.
        """
        selected = port or "out"
        current = node
        while current.children:
            child_name, selected = self._component_output_ref(
                current, selected
            )
            matches = [child for child in current.children if child.name == child_name]
            if len(matches) != 1:
                raise ValueError(
                    f"{current.display_path}: declared output {child_name!r} "
                    f"resolves to {len(matches)} children"
                )
            current = matches[0]
        return current, selected

    def _component_output_ref(
        self,
        node: SpecIRNode,
        port: str,
    ) -> Tuple[str, str]:
        """Resolve one wrapper port with the schema's input-ref grammar."""
        selected = port or "out"
        if selected == "out":
            reference = str(node.output_names[0])
        else:
            outputs = dict(node.secondary_outputs)
            reference = outputs.get(selected)
            if reference is None:
                raise ValueError(
                    f"{node.display_path}: component has no declared "
                    f"secondary output {selected!r}"
                )
            reference = str(reference)

        children = {child.name: child for child in node.children}
        child_ports = {
            child.op: type("_Ports", (), {
                "secondary_outputs": dict(child.secondary_outputs),
            })()
            for child in node.children
            if child.is_component or child.secondary_outputs
        }
        return parse_input_ref(
            reference,
            children,
            custom_ops=self.spec.custom_ops or {},
            components=child_ports,
        )

    @memo_property
    def stacks(self) -> Tuple["Stack", ...]:
        """The model as a set of STACKS, each an ordered sequence of stages.

        Order is `ir.spine` — the authored interleave
        `embed_tokens → layers → norm → lm_head` plus aux `group:` stacks,
        copied at `iter_tree`. Ops join `ir.top_level` by
        `SpecIRNode.layer_path`; blocks join by `BlockMeta.layer_path`.

        Recovering the order is not enough on its own: the stacks have to stay
        apart. A `group: true` entry with
        an `aux_kind` is a separate stack: an MTP head, a vision tower, an audio
        encoder. Its tensors live under their own checkpoint namespace, its cost
        belongs in its own bucket, and — the part that was silently wrong — its
        first stage does NOT read the backbone's hidden state. Threading one
        flat sequence handed MiMo's vision tower the 4096-wide language hidden
        state as its input, so every block in the tower computed against a
        width that is not its own.

        Blocks are joined to the layer that declares them by
        `BlockMeta.layer_path` — where in `spec.model` the declaration is —
        never by name and never one-apiece: one ModelLayer expands into as
        many BlockMetas as it has `assignments:` entries (and `read: all`
        visibility groups), and every one of them is that layer's. A name is
        not an address: an MTP head's `layers` shares its name with the
        backbone's (MiMo, LongCat), and a group's non-repeated template
        child shares the group's own name (`mtp` in `mtp`), which let it
        claim the templates of its siblings so `mtp.layers` never became a
        stage.
        """
        by_layer = {tuple(n.layer_path): n for n in self.ir.top_level
                    if n.layer_path}
        # What the whole-stream lowering placed ahead of each model-level op
        # (`Program.lowered`): the model root holds them just before it.
        prelude: Dict[Tuple[int, ...], Tuple[SpecIRNode, ...]] = {}
        pending: List[SpecIRNode] = []
        for n in self.ir.top_level:
            if n.provenance == "lowering":
                pending.append(n)
                continue
            if pending:
                prelude[tuple(n.layer_path)] = tuple(pending)
                pending = []
        by_path: Dict[Tuple[int, ...], List[BlockTemplate]] = {}
        for template in self.blocks:
            by_path.setdefault(
                tuple(template.meta.layer_path), []).append(template)

        stacks: Dict[str, List[Stage]] = {}
        entry_upstream: Dict[str, Tuple[str, ...]] = {}
        order: List[str] = []
        # The stacks a `group: true` entry opens: each is an index space of
        # its own (`Stack.is_aux`).
        opened: set = set()

        def _add(stage: Stage) -> None:
            if stage.stack not in stacks:
                stacks[stage.stack] = []
                order.append(stage.stack)
            stacks[stage.stack].append(stage)

        def _carry(entry: SpineEntry,
                   group: Optional[str]) -> Optional[Carry]:
            stream = self.ir.stack_reads[tuple(entry.layer_path)].carried
            if stream is None:
                return None
            return Carry(owner=f"{group}/{entry.name}" if group else entry.name,
                         stream=stream)

        def _walk(entries, stack: str,
                  aux_kind: Optional[str], group: Optional[str] = None,
                  carry: Optional[Carry] = None) -> None:
            for entry in entries:
                declared = tuple(entry.inputs)
                if entry.kind == "group":
                    if tuple(entry.layer_path) in self.ir.absent:
                        continue
                    # A group is a stack of its own: its own namespace, cost
                    # bucket and index space. The schema has every group
                    # declare an `aux_kind` (`ModelLayer`), which names what
                    # the stack is for and decides none of that.
                    inner_stack = entry.name
                    opened.add(inner_stack)
                    if declared:
                        entry_upstream.setdefault(inner_stack, declared)
                    # A repeated group around a block that does not repeat
                    # itself (DeepSeek's `mtp`) declares the carried stream
                    # on the group; its block stages are the chain.
                    _walk(entry.children, inner_stack,
                          entry.aux_kind or aux_kind, group=entry.name,
                          carry=_carry(entry, None))
                    continue
                if entry.kind == "op":
                    path = tuple(entry.layer_path)
                    if path in self.ir.fused_layers:
                        continue        # its fused kernel stands later
                    node = by_layer.get(path)
                    if node is None:
                        raise ValueError(
                            f"model layer {entry.name!r} (spec.model "
                            f"{list(path)}) has no node in the IR tree")
                    _add(OpStage(entry=entry, stack=stack,
                                 aux_kind=aux_kind, upstream=declared,
                                 node=node, prelude=prelude.get(path, ())))
                    continue
                own = _carry(entry, group)
                for template in by_path.get(tuple(entry.layer_path), ()):
                    _add(BlockStage(entry=entry, stack=stack,
                                    aux_kind=aux_kind, upstream=declared,
                                    carry=own if own is not None else carry,
                                    template=template))

        _walk(self.ir.spine, "", None, group=None)
        encoders = self.spec.encoder_groups()
        named = {tuple(entry.layer_path): entry.name
                 for entry in self.ir.spine}
        reruns = {named[path]: named[called]
                  for path, called in self.ir.group_ties.items()}
        return tuple(Stack(
            name=k,
            aux_kind=next((st.aux_kind for st in stacks[k] if st.aux_kind), None),
            stages=tuple(stacks[k]),
            upstream=entry_upstream.get(k, ()),
            is_aux=k in opened,
            encoder=k in encoders,
            reruns=reruns.get(k)) for k in order)

    # -- who reads an appended stream, and which elements ------------------
    #
    # The read rules live here once. The walk binds its values and the
    # occurrences that produced them through them (`Evaluator._element`,
    # `_joined`), and `stream_pairings` /
    # `whole_stream_pairings` are their row views: no table is a second
    # statement of which element a reader takes.

    @memo_property
    def _element_reads(self) -> Mapping[
            Tuple[Tuple[int, ...], str], Tuple[Tuple[int, str, int], ...]]:
        """Per `(layer_path, stream)`, every reading position of an element
        read, `(iteration, port, uid)`: a stack that names the stream in
        `inputs:` (the formal at that position, the stage's first root), or
        a node of its templates that reads it (`read: each`). Every stack
        that declares a `read: each` is here, even with no reading
        iteration: that is a count of zero.

        Checked here, where the positions are found: the readers number
        exactly the stream's k elements, and inner readers are one
        contiguous run. A reader short of an element, or one left over, has
        no element to read."""
        streams = self.ir.streams
        readers: Dict[Tuple[Tuple[int, ...], str],
                      List[Tuple[int, str, int]]] = {}

        def claim(path: Tuple[int, ...], name: str
                  ) -> List[Tuple[int, str, int]]:
            return readers.setdefault((path, name), [])

        for path, reads in self.ir.stack_reads.items():
            for name in reads.each:
                claim(path, name)
        for stack in self.stacks:
            for stage in stack.stages:
                if not isinstance(stage, BlockStage):
                    continue
                template = stage.template
                path = tuple(template.meta.layer_path)
                spine = template.meta.reads
                formals = template.input_names()
                root = template.roots[0].uid
                for position, ref in enumerate(stage.upstream):
                    if ref in streams:
                        claim(path, ref).extend(
                            (i, formals[position], root)
                            for i in template.layer_indices)
                for node in template.nodes():
                    for ref in node.inputs_obj or ():
                        name = ref.external_sid
                        if ref.src is not None or name not in streams:
                            continue
                        if name in (*spine.prefix,
                                    *(n for n, _axis in spine.concat)):
                            # Read whole: `whole_stream_pairings`.
                            continue
                        if name not in spine.each:
                            raise ValueError(
                                f"{self.spec.name}: {node.display_path} "
                                f"reads appended stream {name!r}, but its "
                                f"stack {stage.name!r} does not declare "
                                f"`{name}: {{read: each}}` or `{{read: "
                                f"concat, axis: …}}` (or append to it with "
                                f"`read: all`)")
                        claim(path, name).extend(
                            (i, node.name, node.uid)
                            for i in template.layer_indices)
        out = {}
        for (path, name), reads in readers.items():
            layer_name = self.ir.stack_reads[path].name
            stream = streams[name]
            iterations = sorted({i for i, _, _ in reads})
            if len(iterations) != len(stream.iterations):
                raise ValueError(
                    f"{self.spec.name}: {layer_name!r} reads appended stream "
                    f"{name!r} in {len(iterations)} iteration(s) "
                    f"{iterations}, but it has {len(stream.iterations)} "
                    f"element(s); the j-th reading iteration reads element j")
            if name in self.ir.stack_reads[path].each and iterations and (
                    iterations[-1] - iterations[0] + 1 != len(iterations)):
                raise ValueError(
                    f"{self.spec.name}: {layer_name!r} reads appended stream "
                    f"{name!r} in iterations {iterations}, which are not "
                    f"contiguous; its readers are one run of iterations")
            out[(path, name)] = tuple(sorted(reads))
        return out

    def paired_element(self, name: str, stage: "Stage",
                       iteration: Optional[int]) -> int:
        """The element of appended stream `name` that iteration `iteration`
        of `stage` reads: the j-th reading iteration, ascending after
        resolution, reads element j. A stack evaluates each template once
        whichever element it reads, so this, not the walk, is where element
        j meets iteration j."""
        path = (tuple(stage.template.meta.layer_path)
                if isinstance(stage, BlockStage) else ())
        found = self._element_reads.get((path, name))
        iterations = (sorted({i for i, _, _ in found})
                      if found is not None else [])
        if iteration not in iterations:
            raise ValueError(
                f"{self.spec.name}: {stage.name!r} reads stream {name!r} at "
                f"iteration {iteration}, which pairs with no element")
        return iterations.index(iteration)

    @memo_property
    def stream_pairings(self) -> Tuple["StreamPairing", ...]:
        """Which iteration reads which element of each appended stream, one
        row per reading position and iteration (`paired_element`): the view
        validate's report and a drawing read."""
        streams = self.ir.streams
        stack_of: Dict[Tuple[int, ...], str] = {
            tuple(stage.template.meta.layer_path): stack.name
            for stack in self.stacks
            for stage in stack.stages
            if isinstance(stage, BlockStage)}
        rows: List[StreamPairing] = []
        for (path, name), reads in self._element_reads.items():
            if not reads:
                continue
            stream = streams[name]
            writer_stack = stack_of[stream.layer_path]
            reader_stack = stack_of[path]
            reader = self.ir.stack_reads[path].name
            inner = name in self.ir.stack_reads[path].each
            seen = sorted({i for i, _, _ in reads})
            for i, port, uid in reads:
                j = seen.index(i)
                rows.append(StreamPairing(
                    stream=name, element=j,
                    writer_stack=writer_stack,
                    writer=stream.stack_name,
                    writer_iteration=stream.iterations[j],
                    reader_stack=reader_stack,
                    reader=reader, reader_iteration=i,
                    reader_port=port, reader_uid=uid,
                    inner=inner))
        return tuple(rows)

    @memo_property
    def _concat_reads(self) -> Tuple[Tuple[Stage, int, str, Optional[int]],
                                     ...]:
        """Every reader that joins an appended stream whole, `(stage,
        reading node uid, stream, axis)`: a later stack's `read: concat`,
        read by every node of its templates that names the stream — the
        outermost one, when a component call and the body it binds both
        carry the name — and a model-level `op:` that names the stream with
        `axis:`. A stream with no elements that some node joins is refused:
        the node runs and has nothing to join."""
        streams = self.ir.streams
        out: List[Tuple[Stage, int, str, Optional[int]]] = []
        for stack in self.stacks:
            for stage in stack.stages:
                if isinstance(stage, BlockStage):
                    for name, axis in stage.template.meta.reads.concat:
                        k = len(streams[name].iterations)
                        for node in _outermost_readers(stage.template.roots,
                                                       name):
                            if not k:
                                raise ValueError(
                                    f"{self.spec.name}: {node.display_path} "
                                    f"joins appended stream {name!r} "
                                    f"(`read: concat`), which has no "
                                    f"elements in this config; write the "
                                    f"layers that run it with a variant "
                                    f"that does not read it")
                            out.append((stage, node.uid, name, axis))
                elif stage.node.axis is not None:
                    out.extend((stage, stage.node.uid, name, stage.node.axis)
                               for name in stage.upstream if name in streams)
        return tuple(out)

    def concat_reads(self, stage: Stage
                     ) -> Tuple[Tuple[int, str, Optional[int]], ...]:
        """The joins `stage` reads, `(reading node uid, stream, axis)`."""
        return tuple((uid, name, axis)
                     for reader, uid, name, axis in self._concat_reads
                     if reader is stage)

    def joined_elements(self, stage: Stage) -> Mapping[str, Tuple[str, int]]:
        """The names a stage's joins read their elements by
        (`WholeStreamPairing.element_ref`), each `(stream, element)`, and a
        one-element stream by its own name as well: that read is direct."""
        uids = {node.uid for node in (
            stage.template.nodes() if isinstance(stage, BlockStage)
            else stage.roots)}
        streams = self.ir.streams
        out: Dict[str, Tuple[str, int]] = {}
        for uid, name, _axis in self.concat_reads(stage):
            if uid not in uids:
                continue
            k = len(streams[name].iterations)
            out.update((f"{name}[{j}]", (name, j)) for j in range(k))
            if k == 1:
                out[name] = (name, 0)
        return out

    @memo_property
    def whole_stream_pairings(self) -> Tuple["WholeStreamPairing", ...]:
        """Who reads an appended stream whole, or a prefix of it, and in
        which representation (`WholeStreamPairing`): the row view of the
        read rules. A history is the appending stack's own `read: all`
        readers — every occurrence sees the elements appended strictly
        before its iteration's entry, grouped by what they see into derived
        invocations (`BlockMeta.visible`), one row each — or a model-level
        `op:` naming the stream with no `axis:`, which reads all of it. A
        join is a `_concat_reads` reader. `stream_pairings` is the
        element-by-element view, and neither is the other's.

        The rows are the same before and after `lowered`: a lowered
        reading node keeps the stream's name on the edge that now comes
        from its join, which the row names (`through_uid`).
        """
        streams = self.ir.streams
        stack_of: Dict[Tuple[int, ...], str] = {
            tuple(stage.template.meta.layer_path): stack.name
            for stack in self.stacks
            for stage in stack.stages
            if isinstance(stage, BlockStage)}
        rows: List[WholeStreamPairing] = []

        def row(name: str, stage: Stage, iterations: Tuple[int, ...],
                visible: int, uid: int, axis: Optional[int] = None,
                ) -> WholeStreamPairing:
            stream = streams[name]
            through = None if axis is None else next(
                (ref.src.uid for ref in self.by_uid(uid).inputs_obj or ()
                 if ref.external_sid == name and ref.src is not None), None)
            return WholeStreamPairing(
                stream=name,
                writer_stack=stack_of[stream.layer_path],
                writer=stream.stack_name, reader_stack=stage.stack,
                reader=stage.name, reader_iterations=iterations,
                visible=visible, sources=stream.writers[:visible],
                kind=stream.kind,
                reader_uid=uid,
                representation="history" if axis is None else "concat",
                axis=axis, through_uid=through)

        joins = {id(stage): [] for stage, _, _, _ in self._concat_reads}
        for stage, uid, name, axis in self._concat_reads:
            joins[id(stage)].append((uid, name, axis))
        for stack in self.stacks:
            for stage in stack.stages:
                iterations = (stage.template.layer_indices
                              if isinstance(stage, BlockStage) else ())
                if isinstance(stage, BlockStage):
                    rows.extend(
                        row(name, stage, iterations, m, stage.roots[0].uid)
                        for name, m in stage.template.visible.items())
                elif stage.node.axis is None:
                    rows.extend(
                        row(name, stage, (), len(streams[name].iterations),
                            stage.node.uid)
                        for name in stage.upstream if name in streams)
                rows.extend(
                    row(name, stage, iterations,
                        len(streams[name].iterations), uid, axis)
                    for uid, name, axis in joins.get(id(stage), ()))
        return tuple(rows)

    @memo_property
    def joined_streams(self) -> FrozenSet[str]:
        """The appended streams some reader joins whole (`read: concat`,
        or a model-level `op:` with `axis:`): each element is held from
        the iteration that appends it to the join."""
        return frozenset(row.stream for row in self.whole_stream_pairings
                         if row.representation == "concat")

    def lowered(self) -> "Program":
        """This program with every join a real `concat` in its graph.

        Each concat row of `whole_stream_pairings` becomes nodes on a copy
        of the IR (`_lower_concat_reads`): a `slice` view per element when the
        stream publishes a crop, then one `concat` of the k elements, ahead
        of the reading node, which reads the join. k = 1 reads the element
        (or its crop) directly. The walk then prices, lays out and draws
        the join like any node; a program with nothing to join, or one
        already lowered, is returned as it is.
        """
        if self.is_lowered:
            return self
        rows = [row for row in self.whole_stream_pairings
                if row.representation == "concat"]
        tree = _lower_concat_reads(self.ir, rows, self.ir.streams)
        return Program(resolved=self.resolved, ir=tree,
                       schedule=self.schedule)

    @memo_property
    def is_lowered(self) -> bool:
        """Whether every join the graph needs is a node of it: a reader of
        more than one element, or of a crop, reads the stream from a `src`
        (`_lower_concat_reads`). A read of one uncropped element needs none;
        a program that joins nothing is lowered as it stands."""
        streams = self.ir.streams
        for row in self.whole_stream_pairings:
            if row.representation != "concat" or (
                    row.visible < 2
                    and streams[row.stream].slice is None):
                continue
            if any(ref.external_sid == row.stream and ref.src is None
                   for ref in self.by_uid(row.reader_uid).inputs_obj or ()):
                return False
        return True

    @memo_property
    def role_aliases(self) -> Mapping[str, str]:
        """Slot key → the role the component filling it declares for itself.

        A model writes `components: { attention_swa: swa_gqa_attention }`, and
        every leaf inside inherits `attention_swa` — the SLOT. The component's
        own yaml says `role: attention`. Both are wanted, for different
        questions: cost attribution wants the slot, so a hybrid model's windowed
        layers are their own line, and a quant rule wants the canonical role,
        because `per_role.attention` is written about attention and not about
        one spec's naming of it.

        Without the map the two collapse into whichever the leaf happens to
        carry, and a `per_role.attention: fp8` never reaches the SWA leaves —
        they fall to `default_dtype` and the model comes out under-quantised by
        the whole windowed half of its attention.
        """
        out: Dict[str, str] = {}
        components = self.spec.components or {}
        for comp_name, slot_key in (self.spec.component_roles or {}).items():
            graph = components.get(comp_name)
            declared = graph.role if graph is not None else None
            if declared and declared != slot_key:
                out[slot_key] = declared
        return out

    def custom_op(self, op_name: str) -> Optional[Any]:
        return (self.spec.custom_ops or {}).get(op_name)

    @memo_property
    def has_dispatch_zone(self) -> bool:
        """Whether the model routes tokens to experts: some node is declared
        between a `permute` and its `combine`. The MoE question, asked of the
        graph — `n_experts > 1` is a dim that a dense spec may also bind."""
        return any(node.in_dispatch_zone for node in self.nodes())

    @memo_property
    def attention_mix(self) -> Tuple[str, ...]:
        """The attention / sequence mixers the backbone runs: each call's
        cache identity (`variant or label`), once each, sorted. Auxiliary
        stacks — an MTP head, a vision tower — are not the backbone. More
        than one is a hybrid model (`is_hybrid_attention`); whether their
        caches grow by different laws is a different question
        (`CacheView.mixed_growth_laws`)."""
        backbone = self.included_stack_names()
        found = set()
        for index, template in enumerate(self.blocks):
            if self.stack_of_block(index) not in backbone:
                continue
            for node in template.nodes():
                mixer = node.attn_variant or node.kv_label
                if node.children and mixer:
                    found.add(mixer)
        return tuple(sorted(found))

    @property
    def is_hybrid_attention(self) -> bool:
        return len(self.attention_mix) > 1

    def stack_blocks(self, aux_kind: Optional[str] = None
                     ) -> Tuple[BlockTemplate, ...]:
        """The block invocations of the stacks whose `aux_kind` is
        `aux_kind`, in declaration order; None — the backbone's. A model
        card's layer groups are the backbone's: an MTP head or a vision tower
        is a stack of its own (`Stack.aux_kind`), however its layers are
        written — `assignments:` inside a group make none of them the
        backbone's."""
        kinds = {stack.name: stack.aux_kind for stack in self.stacks}
        return tuple(template for index, template in enumerate(self.blocks)
                     if kinds[self.stack_of_block(index)] == aux_kind)

    @memo_property
    def headline(self) -> Headline:
        """The model card's sizes (`Headline`), read off the model-level
        dims (`base_binding`). A spec that declares no `d_model` or
        `n_vocab` is refused: every language model has both, and no other
        dim stands in for them."""
        dims = self.base_binding

        def declared(name: str) -> Optional[int]:
            raw = dims.get(name)
            value = None if raw is None else value_of(raw)
            return None if value is None else int(value)

        for name, config in (("d_model", "hidden_size"),
                             ("n_vocab", "vocab_size")):
            if declared(name) is None:
                raise SpecRefusal(
                    f"the spec declares no model-level `{name}`",
                    declaration="dims",
                    fix=f"bind `{name}: config.{config}` (or the config "
                        f"field the model names it by) in the spec's "
                        f"`dims:`")
        backbone = self.stack_blocks()
        dense_layers = tuple(sorted(
            index for template in backbone if template.runs_dense_ffn()
            for index in template.layer_indices))
        return Headline(
            hidden=declared("d_model"),
            vocab=declared("n_vocab"),
            layers=sum(template.count for template in backbone),
            dense_layers=dense_layers,
            heads=declared("n_heads"),
            kv_heads=declared("n_kv_heads"),
            intermediate=declared("d_ffn") if dense_layers else None,
            head_dim=declared("d_head"),
            kv_lora_rank=declared("d_kv_lora"),
            q_lora_rank=declared("d_q_lora"),
            qk_rope_dim=declared("d_qk_rope"),
        )

    @staticmethod
    def reads(node: SpecIRNode) -> FrozenSet[str]:
        """The dims `node`'s body reads, in the model's vocabulary
        (`ir_dim_symbols` of every leaf under it, renamed out through each
        enclosing call): what a call's geometry is made of, whatever its
        own names for them."""
        found: set = set()
        for leaf in _depth_first((node,)):
            if not leaf.children:
                found |= ir_dim_symbols(leaf)
        return frozenset(found)

    @memo_property
    def routed_uids(self) -> FrozenSet[int]:
        """Nodes whose work runs on dispatched tokens.

        The zone is declared on the nodes between `permute` and `combine`. A
        component invoked there runs its whole body on those tokens, so the
        body inherits the flag; and a parameter read by a routed node is a
        routed weight — gpt-oss's expert biases hang off the bank as
        `parameter_vector` sources that no dataflow edge from `permute`
        reaches, yet they are streamed once per active expert like the bank.
        """
        routed = {node.uid for node in self.nodes() if self.in_zone(node)}
        for node in self.nodes():
            if node.uid not in routed:
                continue
            for ref in (node.inputs_obj or ()):
                src = ref.src
                if src is not None and is_identity_source(src.op):
                    routed.add(src.uid)
        return frozenset(routed)

    @memo_property
    def dispatch_members(self) -> FrozenSet[int]:
        """The nodes of a MoE block: those running on dispatched tokens
        (`routed_uids`) plus the dispatch boundary ops themselves
        (`OpSpec.is_dispatch_boundary`: `permute` / `combine`), which are
        not inside the zone they bracket."""
        return self.routed_uids | frozenset(
            node.uid for node in self.nodes()
            if is_dispatch_boundary(node.declaration))

    def dispatch_of(self, node: Any) -> Optional[int]:
        """The uid of the dispatch (`permute`) whose routed buffer `node`
        reads or writes — asked of the dispatch itself, a node of its zone,
        its `combine`, or a node grafted beside them (a cast, a collective)
        — None where no body enclosing `node` holds a dispatch.

        A zone is a graph's own (`spec_tree.compute_dispatch_zone`: the
        nodes between a `permute` and the `combine` of one component body),
        so the dispatch is the `permute` of the nearest enclosing body that
        has one, and a body with two is a zone this cannot attribute: it is
        refused rather than guessed. A fused node has no parent chain; its
        caller asks of what it absorbed. Which nodes are ON a routed buffer
        is the access layer's (`LeafAccess.routed_inputs`)."""
        if node is None:
            return None
        if node.op == DISPATCH_OP:
            return node.uid
        scope = node
        while scope is not None:
            siblings = self._scope_of(scope)
            found = [sibling.uid for sibling in siblings
                     if sibling.op == DISPATCH_OP]
            if len(found) > 1:
                raise ValueError(
                    f"{node.display_path}: its body holds {len(found)} "
                    "dispatches, and a zone is one dispatch's")
            if found:
                return found[0]
            scope = scope.parent
        return None

    def _scope_of(self, node: Any) -> Sequence[Any]:
        """The nodes `node` sits beside: its parent's children, or the root
        list (the model's top level, a block template's roots) it is in."""
        if node.parent is not None:
            return node.parent.children
        for roots in (self.ir.top_level,
                      *(roots for _meta, roots in self.ir.blocks)):
            if any(candidate is node for candidate in roots):
                return roots
        return ()

    def in_zone(self, node: Any) -> bool:
        """Whether `node` sits in a dispatch zone: its own flag, else an
        ancestor's. A fused node carries the flag its origins had and has no
        parent chain. Uncached, so a fusion pass may ask it of members before
        the fused nodes it is about to graft exist."""
        while node is not None:
            if node.in_dispatch_zone:
                return True
            node = node.parent
        return False

    def is_routed(self, uid: int) -> bool:
        return uid in self.routed_uids

    def node_is_routed(self, node: Any) -> bool:
        """`is_routed` asked of a node in hand: the cached set first, else
        the node's own flag / ancestors — which is how a fused node grafted
        after `routed_uids` was computed (it is never in that set) answers by
        the flag its origins gave it."""
        return node.uid in self.routed_uids or self.in_zone(node)

    def nodes(self) -> Iterator[SpecIRNode]:
        """Every IR node, top-level entries then block bodies, depth-first."""
        yield from _depth_first(self.ir.top_level)
        for template in self.blocks:
            yield from template.nodes()

    def walked_nodes(self) -> Iterator[SpecIRNode]:
        """`nodes`, of the stacks this program's walk evaluates
        (`walk_stacks`): what an Analysis of it has values for."""
        walked = self.walked_stack_names
        for node in self.nodes():
            if self.stack_of(node) in walked:
                yield node

    @memo_property
    def identity_sources_by_sole_reader(self) -> Mapping[int, Tuple[int, ...]]:
        """Per node, the identity sources it is the ONE reader of — in this
        walk's node order.

        A parameter vector or tensor is no kernel: `traffic` already bills
        its bytes to the consumer that reads it, so a presentation that
        wants one row per launch folds a single-reader source into that
        reader's row. Which reader that is comes from the edges, which are
        this walk's — a consumer asking it re-walks `inputs_obj` and makes
        the same join a second time. A source two nodes read stays its own
        row, and so does one whose only reader is itself."""
        readers: Dict[int, set] = {}
        for node in self.nodes():
            for ref in node.inputs_obj or ():
                if ref.src is not None:
                    readers.setdefault(ref.src.uid, set()).add(node.uid)
        out: Dict[int, List[int]] = {}
        for source in self.nodes():
            seen = readers.get(source.uid) or set()
            if not (is_identity_source(source.op) and len(seen) == 1):
                continue
            reader = next(iter(seen))
            if reader != source.uid:
                out.setdefault(reader, []).append(source.uid)
        return {reader: tuple(sources) for reader, sources in out.items()}

    def language_nodes(self) -> Iterator[SpecIRNode]:
        """`walked_nodes` outside the encoders: the language model's, the
        only nodes that keep a cache for a later step. An encoder runs once
        per media item and keeps nothing, whatever its attention declares.
        An isolated component is walked as its call site's stack, and keeps
        what that stack keeps (`Evaluator`'s isolation)."""
        encoders = (self.encoder_uids if self.resolved.site is None
                    else frozenset())
        for node in self.walked_nodes():
            if node.uid not in encoders:
                yield node

    @memo_property
    def _prefix_by_uid(self) -> Mapping[int, str]:
        """Each node's checkpoint namespace, from the scope it belongs to.

        Asked of the stack via `spec_tree.checkpoint_prefix` rather than
        recomputed here, and of a model-level entry via the root its scope
        declares: two implementations would give the same weight two
        on-disk names.
        """
        def _walk(nodes: Sequence[SpecIRNode], prefix: str,
                  out: Dict[int, str]) -> None:
            for n in nodes:
                out[n.uid] = prefix
                _walk(n.children, prefix, out)

        out: Dict[int, str] = {}
        for n in self.ir.top_level:
            _walk((n,), scope_prefix(n.checkpoint_root), out)
        for template in self.blocks:
            _walk(template.roots, template.checkpoint_prefix, out)
        return out

    def checkpoint_prefix(self, node: SpecIRNode) -> str:
        return self._prefix_by_uid.get(node.uid, "")

    def as_stored(self, node: SpecIRNode, tensors: Sequence[Any]
                  ) -> Tuple[Any, ...]:
        """`tensors` (what `node`'s op declares) as the checkpoint stores
        them: under the suffixes the node declares (`tensor_suffix`), keyed
        by the declared tensor: `weight` is the primary one, even where a
        bare `nn.Parameter` stores it under `""`."""
        require_module(node)
        renamed = node.tensor_suffix
        declared = lambda suffix: suffix or ".weight"
        stray = set(renamed) - {declared(t.suffix) for t in tensors}
        if stray:
            raise ValueError(
                f"{node.display_path!r}: `tensor_suffix` renames "
                f"{sorted(stray)}, which its op does not store")
        # The tensor a lookup gathers rows from — its leading one — is a
        # TABLE, whoever stores it (`lookups`).
        tables = {site.table for site in self.lookups.values()}
        return tuple(
            replace(t, suffix=renamed.get(declared(t.suffix), t.suffix),
                    logical_suffix=renamed.get(declared(t.logical_suffix),
                                               t.logical_suffix),
                    kind=(StorageKind.TABLE
                          if at == 0 and node.uid in tables else t.kind))
            for at, t in enumerate(tensors))

    @memo_property
    def _stack_by_uid(self) -> Mapping[int, str]:
        """Which stack each node belongs to, so cost can be bucketed by a
        group-by instead of by three more fields on the result."""
        def _walk(nodes: Sequence[SpecIRNode], stack: str,
                  out: Dict[int, str]) -> None:
            for n in nodes:
                out[n.uid] = stack
                _walk(n.children, stack, out)

        out: Dict[int, str] = {}
        for stack in self.stacks:
            for stage in stack.stages:
                _walk(stage.roots, stack.name, out)
        # A node grafted at the model root after the stages were declared —
        # the collective layout materializes after an output head — is no
        # stage's root. It belongs to the stack of the node it reads: the
        # vocab all-gather behind an MTP head is the MTP stack's, not the
        # backbone's.
        for n in self.ir.top_level:
            if n.uid in out:
                continue
            producer = next((ref.src for ref in n.inputs_obj or ()
                             if ref.src is not None), None)
            if producer is not None and producer.uid in out:
                _walk((n,), out[producer.uid], out)
        return out

    def stack_of(self, node: SpecIRNode) -> str:
        return self._stack_by_uid.get(node.uid, "")

    @memo_property
    def _block_by_uid(self) -> Mapping[int, int]:
        return {n.uid: index for index, template in enumerate(self.blocks)
                for n in template.nodes()}

    def where(self, uid: int) -> str:
        """Where a node runs, as a message names it: the model layer (stack
        row) and block template of a block node, with the assignment's
        layers when it covers a subset, and the cost stack —
        `'input_layernorm' in layers (template block, stack 'backbone')`.
        A model-level op is named by its path and stack."""
        node = self.by_uid(uid)
        stack = Stack.display_name(self.stack_of(node))
        index = self.block_of(uid)
        if index is None:
            return f"{node.display_path!r} (stack {stack!r})"
        meta = self.blocks[index].meta
        layers = (f" layers {sorted(meta.layer_indices)}"
                  if meta.layer_indices else "")
        return (f"{node.display_path!r} in {meta.stack_name}{layers} "
                f"(template {meta.template_name}, stack {stack!r})")

    def block_index(self, template: BlockTemplate) -> int:
        """The block invocation (`blocks` ordinal) `template` is: the one
        expanded from the same `BlockMeta`."""
        return next(index for index, block in enumerate(self.blocks)
                    if block.meta is template.meta)

    def block_of(self, uid: int) -> Optional[int]:
        """The block invocation (`blocks` ordinal) a node's body belongs to,
        `None` for a model-level op. One invocation is one kind of layer:
        two of its nodes run in the same layer, two invocations' nodes in
        different ones."""
        return self._block_by_uid.get(uid)

    def repeated_in(self, uid: int) -> Tuple[int, ...]:
        """The repeated component instances a node runs inside, outermost
        first; `()` when nothing repeats it.

        `block_of` says which layer a node runs in, and that is not the same
        as how many times it runs there: a component with `repeat: 27` runs
        its body 27 times inside one block invocation, while a sibling of
        that component runs once. Read off the same declarations
        `copy_factors` multiplies (`repeat:` up the parent chain), as
        instances rather than as a product — two different components that
        each repeat 27 times are still two, and a count cannot say so.
        """
        node: Optional[Any] = self.by_uid(uid)
        chain: List[int] = []
        while node is not None:
            if node.repeat is not None and int(node.repeat) > 1:
                chain.append(node.uid)
            node = node.parent
        return tuple(reversed(chain))

    def is_lm_head(self, node: SpecIRNode) -> bool:
        """Whether `node` is the model's LM head: an output head on the
        backbone stack. An auxiliary stack may own an output head of its own
        — an MTP draft's logits — and that head is its stack's cost, not a
        second LM head of the model."""
        return (node.boundary_role == "output_head"
                and self.stack_of(node) == "")

    @memo_property
    def encoder_uids(self) -> FrozenSet[int]:
        """Every node inside an encoder stack (`Stack.is_encoder`)."""
        encoders = {stack.name for stack in self.stacks if stack.is_encoder}
        return frozenset(uid for uid, stack in self._stack_by_uid.items()
                         if stack in encoders)

    def stack_of_block(self, block_index: int) -> str:
        """The stack (cost bucket and checkpoint namespace) owning one block
        invocation.

        Block ordinals identify invocations; template names do not. Resolve
        ownership from the ``BlockStage`` that Program already joined to the
        model structure so parameter-free blocks retain a stack even though
        they have no weight leaf from which a consumer could infer one.
        """
        try:
            template = self.blocks[block_index]
        except IndexError as exc:
            raise KeyError(block_index) from exc
        owners = {
            stack.name
            for stack in self.stacks
            for stage in stack.stages
            if (
                isinstance(stage, BlockStage)
                and stage.template.meta is template.meta
            )
        }
        if len(owners) != 1:
            raise ValueError(
                f"block {block_index} has {len(owners)} Program stack owners: "
                f"{sorted(owners)!r}"
            )
        return next(iter(owners))

    def included_stack_names(self, *, include_aux: bool = False) -> FrozenSet[str]:
        """Stack identities admitted by the model accounting policy.

        Storage, presentation projections and future consumers must select
        the same structural unit. In particular, filtering only block metadata
        misses singleton parameters that live at the top level of an auxiliary
        stack, while filtering leaf names cannot identify nested aux blocks.

        Only the stacks this program walks are admitted: the language
        model's step admits no encoder, whose weights and activations are
        the encode phase's own accounting, whatever `include_aux` says; an
        encode step's encoder is the whole of what it accounts for.
        """
        walked = self.walked_stack_names
        return frozenset(
            stack.name for stack in self.stacks
            if stack.name in walked
            and (include_aux or not stack.is_aux
                 or stack.name == self.encode_stack)
        )

    @memo_property
    def contributions(self) -> Tuple["Contribution", ...]:
        """The model's accounting ROOTS: every node a total sums, with how many
        times the model holds it and which stack it belongs to.

        A model total is not a sum over nodes. It is a sum over roots — a
        component's value already includes its children, so summing everything
        counts the whole tree twice — and each block root counts once per layer
        using its template, because the walk evaluates a template ONCE and
        multiplies rather than walking twenty-eight identical layers.

        Those two rules were written out three times: in `total`, in `by_stack`,
        and again in the traffic test's own model sum. Three copies of a fold is
        three chances to forget the multiplier, and forgetting it is not visible
        in the result — it just makes the model smaller by a factor of `n_layers`
        in a way that still looks like a model.

        What is deliberately NOT here is how a value scales. `Cost` multiplies
        params by the storage count and flops by the active one; `Traffic`
        multiplies every term alike. So this hands out the COUNT and each layer
        applies its own combine — the same reason `Layer.rebase` refuses to
        default one.
        """
        walked = self.walked_stack_names
        out: List[Contribution] = [
            Contribution(node=node, count=1, stack=self.stack_of(node))
            for node in self.ir.top_level
            if self.stack_of(node) in walked
        ]
        for index, template in enumerate(self.blocks):
            out.extend(
                Contribution(
                    node=node, count=template.count,
                    stack=self.stack_of(node), block_index=index)
                for node in template.roots
                if self.stack_of(node) in walked)
        return tuple(out)

    def multiplicity(self, counts: Optional[Mapping[int, int]] = None
                     ) -> "Layer[int]":
        """How many of each node a selection of the model runs: each
        accounting root (`contributions`) at `counts[root uid]` — its own
        count when absent — and every node beneath it alike.

        The one reading of a per-root selection: a pipeline stage's view
        (`layout.stages.StageView`) hands its counts here, and consumers
        receive the resulting `Layer` rather than reconstructing ownership.
        """
        selected = dict(counts or {})
        unknown = set(selected) - {c.node.uid for c in self.contributions}
        if unknown:
            raise ValueError(
                f"counts name roots {sorted(unknown)!r} that are no "
                f"accounting root of the model")
        values: Dict[int, int] = {}
        for contribution in self.contributions:
            source = int(contribution.count)
            kept = int(selected.get(contribution.node.uid, source))
            if not 0 <= kept <= source:
                raise ValueError(
                    f"{contribution.node.display_path!r} keeps {kept} of "
                    f"{source}")
            pending = [contribution.node]
            while pending:
                node = pending.pop()
                previous = values.setdefault(node.uid, kept)
                if previous != kept:
                    raise ValueError(
                        f"node {node.display_path!r} belongs to contributions "
                        f"with counts {previous} and {kept}")
                pending.extend(node.children or ())
        return Layer(self, values)

    # How many copies of each node the model holds is deliberately NOT here.
    # `SpecIRNode.multiplicity` looks like the answer and is only the answer at
    # one binding — the spec's global dims, the only one `iter_tree` has. A rank
    # under expert parallelism holds a fraction of the bank, so the count is a
    # property of an ANALYSIS and not of the program; `Analysis.instances`
    # carries it, resolved by the walk that knows which binding it asked at.

    # -- occurrences: which stage evaluates a node, at which iterations ----
    #
    # The walk evaluates a template once for all its iterations and binds
    # one value per input, and beside it the occurrences that produced the
    # value at each iteration (`Analysis.input_references`).

    @memo_property
    def iteration_space(self) -> Mapping[Tuple[int, ...], Tuple[int, int]]:
        """The iteration space a pipeline partition cuts: the repeated model
        layers of every stack that is not aux (`Stack.is_aux`), concatenated
        in walk order, as `layer_path` → `(offset, depth)`.

        Each such layer's templates cover its iterations exactly once. A
        gap or an overlap names no partition at any `pp`, and the walk asks
        this before it evaluates anything (`Evaluator.__init__`), so it is
        refused at `pp = 1` too."""
        out: Dict[Tuple[int, ...], Tuple[int, int]] = {}
        covered: Dict[Tuple[int, ...], List[int]] = {}
        names: Dict[Tuple[int, ...], str] = {}
        offset = 0
        for stack in self.walk_stacks:
            if stack.is_aux:
                continue
            for stage in stack.walk_order():
                if not isinstance(stage, BlockStage) \
                        or stage.entry.repeat is None:
                    continue
                key = stage.layer_path
                if key not in out:
                    depth = stage.entry.repeat
                    out[key] = (offset, depth)
                    names[key] = stage.name
                    offset += depth
                covered.setdefault(key, []).extend(
                    stage.template.layer_indices)
        for key, indices in covered.items():
            depth = out[key][1]
            if sorted(indices) != list(range(depth)):
                missing = sorted(set(range(depth)) - set(indices))
                twice = sorted({i for i in indices if indices.count(i) > 1})
                found = "; ".join(
                    [*([f"a gap at {missing[:8]}"] if missing else []),
                     *([f"an overlap at {twice[:8]}"] if twice else [])])
                raise ValueError(
                    f"{self.spec.name}: the templates of repeated layer "
                    f"{names[key]!r} cover its iterations 0..{depth - 1} "
                    f"with {found}; each iteration runs exactly one template")
        return out

    @memo_property
    def layer_reads(self) -> Tuple[LayerRead, ...]:
        """Every read between model layers, resolved to the layers it names
        (`layer_reads`): the reads the walk orders its stacks by, and the
        edges a model-level diagram draws."""
        return layer_reads(self.ir, self.spec)

    @memo_property
    def walk_stacks(self) -> Tuple["Stack", ...]:
        """The stacks this program's walk evaluates (`encode_stack`), in the
        order it evaluates them: every cross-stack read after its producer
        (`_dependency_ordered`). The language model reads an encoder's output
        as a handoff (`handoffs`), not by walking the encoder."""
        ordered = _dependency_ordered(self.stacks, self.ir, self.spec)
        if self.resolved.site is not None:
            # An isolated component is one stage, walked as its call site's
            # stack whatever that is, handed what its call hands over
            # (`CallSite.formals`).
            return ordered
        if self.encode_stack is not None:
            # An encoder that runs another's weights (`Stack.reruns`) walks
            # beside that one, which holds them, at no item of its own.
            (own,) = [st for st in ordered if st.name == self.schedule]
            return tuple(st for st in ordered
                         if st.name in (self.schedule, own.reruns))
        language = tuple(st for st in ordered if not st.is_encoder)
        if self.schedule != self.CHECKPOINT:
            return language
        # The language model reads what the encoders hand over, not their
        # walk, so the encoders follow it.
        return language + tuple(st for st in ordered if st.is_encoder)

    @memo_property
    def walked_stack_names(self) -> FrozenSet[str]:
        return frozenset(stack.name for stack in self.walk_stacks)

    @memo_property
    def sparse_dcp_components(self) -> Tuple[SpecIRNode, ...]:
        """Components declaring a sparse DCP runtime contract on this graph."""
        return tuple(node for node in self.walked_nodes() if node.sparse_dcp is not None)

    @memo_property
    def required_runtime_framework(self) -> Optional[str]:
        """Framework required by the runtime contracts this graph declares."""
        frameworks = {node.sparse_dcp.framework for node in self.sparse_dcp_components}
        if len(frameworks) > 1:
            raise ValueError(f"conflicting declared runtime frameworks: {sorted(frameworks)}")
        return next(iter(frameworks), None)

    @memo_property
    def walked_components(self) -> FrozenSet[str]:
        """The components this program's walk invokes, at any depth."""
        out: set = set()
        for stack in self.walk_stacks:
            for stage in stack.stages:
                for node in _depth_first(stage.roots):
                    if node.is_component:
                        out.add(node.op)
        return frozenset(out)

    @memo_property
    def handed_exits(self) -> Mapping[int, str]:
        """A handed-over model-level op, by uid, with its name: a reader
        whose edge the tree resolved to that node by `src` reads the
        handoff. A handed-over block stage is read by its name."""
        return {stage.node.uid: stage.name
                for stack in self.stacks for stage in stack.stages
                if isinstance(stage, OpStage)
                and stage.name in self._handed_over}

    @memo_property
    def _handed_over(self) -> FrozenSet[str]:
        """The names the language model reads from an encoder: what an
        encoder hands over, never an edge into the encoder's own walk, even
        where one walk evaluates both (`CHECKPOINT`)."""
        return frozenset(h.name for h in self.handoffs)

    @memo_property
    def _walk_sequence(self) -> Tuple[Stage, ...]:
        return tuple(stage for stack in self.walk_stacks
                     for stage in stack.walk_order())

    @memo_property
    def _occurrence_frames(self) -> Mapping[
            int, Tuple[int, Optional[SpecIRNode], SpecIRNode]]:
        """uid → (the position in `_walk_sequence` of the stage that
        evaluates it, the component call whose body holds it — None at the
        stage's root —, the stage root above it)."""
        out: Dict[int, Tuple[int, Optional[SpecIRNode], SpecIRNode]] = {}

        def visit(nodes: Sequence[SpecIRNode], position: int,
                  parent: Optional[SpecIRNode],
                  root: Optional[SpecIRNode]) -> None:
            for node in nodes:
                top = node if root is None else root
                out[node.uid] = (position, parent, top)
                visit(node.children, position, node, top)

        for position, stage in enumerate(self._walk_sequence):
            visit(stage.roots, position, None, None)
        return out

    def evaluating_stage(self, uid: int) -> Tuple[Stage, SpecIRNode]:
        """The stage that evaluates node `uid`, and the stage root it lies
        under."""
        position, _parent, root = self._frame(uid)
        return self._walk_sequence[position], root

    def _frame(self, uid: int) -> Tuple[int, Optional[SpecIRNode], SpecIRNode]:
        found = self._occurrence_frames.get(uid)
        if found is None:
            raise ValueError(
                f"{self.spec.name}: uid {uid} is no node a stage evaluates")
        return found

    @staticmethod
    def stage_iterations(stage: Stage) -> Tuple[Optional[int], ...]:
        """The occurrences of a stage's nodes: one per iteration of a block
        stage, one (None) for a model-level op."""
        if isinstance(stage, BlockStage):
            return stage.template.layer_indices
        return (None,)

    def occurrence_layers(self, uid: int) -> Tuple[Optional[int], ...]:
        """The iterations node `uid` occurs at: the layers of the block stage
        that evaluates it, or `(None,)` for a model-level op that runs once.

        One node, one answer, and `occurrence_domains` is this over every
        node — the walk already knows which layers a template covers
        (`BlockTemplate.layer_indices`), and a sideband layer that needs the
        layer a weight sits at (`eval.formats`) reads it here rather than
        carrying a layer of its own."""
        position, _parent, _root = self._frame(uid)
        return self.stage_iterations(self._walk_sequence[position])

    def stack_depth(self, stack: Optional[str]) -> Optional[int]:
        """How many layers `stack` has, as its templates cover them — the
        `total` a `LayerSelection` of that stack resolves against. None for
        a stack of no block stages (nothing to index)."""
        covered = {layer
                   for declared in self.stacks if declared.name == (stack or "")
                   for stage in declared.stages
                   if isinstance(stage, BlockStage)
                   for layer in stage.template.layer_indices}
        return max(covered) + 1 if covered else None

    def occurrence_domains(self) -> Iterator[
            Tuple[int, Tuple[Optional[int], ...]]]:
        """Every node the walk evaluates, with the iterations it occurs at."""
        for uid in self._occurrence_frames:
            yield uid, self.occurrence_layers(uid)

    def declared_reads(self, uid: int) -> Tuple[str, ...]:
        """The streams a component call reads inside a kernel no edge shows
        (`BlockGraph.reads` with no `in:` naming them in its body: GLM's
        shared-indexer attention reuses `dsa_topk`). Each is an input of the
        call, after its `in:` refs."""
        node = self.by_uid(uid)
        if not node.children or not node.stream_reads:
            return ()
        named = {ref.external_sid for inner in _depth_first((node,))
                 for ref in inner.inputs_obj or () if ref.src is None}
        return tuple(name for name in node.stream_reads if name not in named)

    @memo_property
    def _stage_positions(self) -> Mapping[int, int]:
        return {id(stage): position
                for position, stage in enumerate(self._walk_sequence)}

    def walk_position(self, stage: Stage) -> int:
        """Where the walk evaluates `stage`, counted over every stage."""
        return self._stage_positions[id(stage)]

    # -- inner streams: the layer each reader names ------------------------

    def inner_stream_reference(self, template: BlockTemplate,
                               name: str) -> Reference:
        """What inner stream `name` is at each iteration of `template`: the
        occurrence the layer its `reads_from` names writes it at
        (`stream_origin`)."""
        owner = template.meta.iteration_owner
        return Reference(table=tuple(
            (layer, (Source(self.stream_origin(owner, layer, name), "out",
                            "stream"),))
            for layer in template.layer_indices
            if (owner, layer, name) in self.ir.stream_reads))

    def stream_origin(self, stack: Tuple[int, ...], layer: int,
                      name: str) -> Occurrence:
        """The occurrence whose output layer `layer` of `stack` reads as
        inner stream `name`: the node the layer its `reads_from` names
        writes (`SpecIRTree.stream_reads` / `stream_writes`)."""
        at = self.ir.stream_reads[(stack, layer, name)]
        return Occurrence(self.ir.stream_writes[(*at, name)],
                          None if at[1] == STACK_ENTRY else at[1])

    def final_inner_stream(self, at: Tuple[int, ...],
                           name: str) -> Optional[Occurrence]:
        """What a model-level op at `at` that names inner stream `name` in
        `inputs:` reads: the write of the last layer of the declaring stack
        that writes it. None when `name` is no inner stream in scope."""
        stack = self.spec.inner_stream_named(tuple(at), name)
        if stack is None:
            return None
        writes = [(layer, uid) for (owner, layer, n), uid
                  in self.ir.stream_writes.items()
                  if owner == stack and n == name and layer != STACK_ENTRY]
        if not writes:
            raise SpecRefusal(
                f"{self.spec.name}: {name!r} is read by name after its "
                f"stack, and no layer of the stack writes it",
                declaration="inputs",
                fix=(f"have a component of the stack `stream_out` "
                     f"{name!r}, or read another tensor"))
        layer, uid = max(writes)
        return Occurrence(uid, layer)

    def final_inner_streams(self, stage: "Stage") -> Dict[str, Occurrence]:
        """The inner streams model-level op `stage` names in `inputs:`, each
        by the write it reads (`final_inner_stream`). Empty for a block
        stage, whose `inputs:` thread the stack."""
        if not isinstance(stage, OpStage):
            return {}
        out: Dict[str, Occurrence] = {}
        for name in stage.upstream:
            final = self.final_inner_stream(stage.layer_path, name)
            if final is not None:
                out[name] = final
        return out

    def inner_streams_read_after_their_stack(
            self) -> Dict[str, Dict[str, Occurrence]]:
        """Per model-level op that reads an inner stream from OUTSIDE the
        stack declaring it (by `display_path`), the stream names it reads
        and the write each one reads (`final_inner_stream`).

        DeepSeek-V4.1's model-level `hc_exit` reduces the backbone's final
        `hc_pre_mix`. Which write that is, is this walk's answer and no
        reader's to re-derive: the edges are the IR's, the name is resolved
        in the READER's scope (`ArchSpec.inner_stream_named`) and the write
        is the maximum WRITING layer of that stack — the occurrence the
        evaluator bound for this reader. A consumer maps the uid named here
        to whatever it draws around it, and walks no edges of its own."""
        out: Dict[str, Dict[str, Occurrence]] = {}
        for node in _depth_first(self.ir.top_level):
            for ref in node.inputs_obj or ():
                if ref.src is not None or not ref.external_sid:
                    continue
                final = self.final_inner_stream(tuple(node.layer_path),
                                                ref.external_sid)
                if final is not None:
                    out.setdefault(node.display_path, {})[
                        ref.external_sid] = final
        return out

    def inner_stream_layers(self, owner: Tuple[int, ...]) -> Dict[
            str, Dict[int, Tuple[bool, Optional[Union[int, str]]]]]:
        """Per inner stream of the layers `owner` numbers, per layer that
        writes or reads it: whether the layer writes it, and the layer of
        `owner` whose write it reads — or, when the write is not one of
        `owner`'s layers, what it is: the stream's `seed` read at the
        stack's entry by name, another stack's layer as `<stack> layer <i>`;
        None when it reads none (`SpecIRTree.stream_writes` /
        `stream_reads`)."""
        names: Dict[Tuple[int, ...], str] = {}

        def walk(entries) -> None:
            for entry in entries:
                names[tuple(entry.layer_path)] = entry.name
                walk(entry.children)

        walk(self.ir.spine)
        out: Dict[str, Dict[int, Tuple[bool, Optional[Union[int, str]]]]] = {}
        for (stack, layer, name) in self.ir.stream_writes:
            if stack == owner and layer != STACK_ENTRY:
                out.setdefault(name, {})[layer] = (True, None)
        for (stack, layer, name), (source_stack, at) in self.ir.stream_reads.items():
            if stack != owner:
                continue
            if at == STACK_ENTRY:
                origin = self.stream_origin(stack, layer, name)
                source: Union[int, str] = self.by_uid(origin.uid).name
            elif source_stack != owner:
                source = f"{names[source_stack]} layer {at}"
            else:
                source = at
            writes, _ = out.setdefault(name, {}).get(layer, (False, None))
            out[name][layer] = (writes, source)
        return {name: dict(sorted(rows.items())) for name, rows in out.items()}

    def stream_writers(self, template: BlockTemplate
                       ) -> Dict[str, Tuple[SpecIRNode, ...]]:
        """Per inner stream `template` writes, the nodes of its body whose
        output its layers publish as that stream (`SpecIRTree.stream_writes`
        at each of its layers): where in the block the stream is made."""
        owner = template.meta.iteration_owner
        layers = set(template.layer_indices)
        out: Dict[str, Dict[int, None]] = {}
        for (stack, layer, name), uid in self.ir.stream_writes.items():
            if stack == owner and layer in layers:
                out.setdefault(name, {})[uid] = None
        return {name: tuple(self.by_uid(uid) for uid in uids)
                for name, uids in out.items()}

    @memo_property
    def stream_seed_nodes(self) -> FrozenSet[int]:
        """The model-level nodes an inner stream's `seed` names: each
        writes its stream at the stack's entry."""
        return frozenset(uid for (_stack, layer, _name), uid
                         in self.ir.stream_writes.items()
                         if layer == STACK_ENTRY)

    @memo_property
    def read_inner_streams(self) -> FrozenSet[str]:
        """The inner streams some layer reads."""
        return frozenset(name for _stack, _layer, name in self.ir.stream_reads)

    def block_stream_reads(self, block: int) -> Mapping[str, Tuple[int, ...]]:
        """Every inner stream block invocation `block` reads, with the nodes
        whose outputs its layers read as it, in layer order — one per
        distinct producer. The walk holds them to one shape, layout and
        cache law (`Evaluator._stream_value`)."""
        found = self._block_stream_reads.get(block)
        if found is None:
            template = self.blocks[block]
            owner = template.meta.iteration_owner
            layers = set(template.layer_indices)
            rows: Dict[str, Dict[int, None]] = {}
            for (stack, layer, name) in self.ir.stream_reads:
                if stack == owner and layer in layers:
                    rows.setdefault(name, {})
            for name, producers in rows.items():
                for layer in template.layer_indices:
                    if (owner, layer, name) in self.ir.stream_reads:
                        producers[self.stream_origin(owner, layer, name).uid] \
                            = None
            found = {name: tuple(producers)
                     for name, producers in rows.items()}
            self._block_stream_reads[block] = found
        return found

    def declared_stream_reads(self, block: int) -> Mapping[
            str, Tuple[str, Tuple[Tuple[int, Union[int, str]], ...]]]:
        """Per inner stream block invocation `block` reads: its `reads_from`
        value as declared, and each of its layers with the layer whose
        write it reads — or, read at the stack's entry, the name of the
        stream's `seed`."""
        template = self.blocks[block]
        owner = template.meta.iteration_owner
        out: Dict[str, Tuple[str, List[Tuple[int, int]]]] = {}
        for layer in template.layer_indices:
            for name in self.block_stream_reads(block):
                key = (owner, layer, name)
                if key in self.ir.stream_reads:
                    _form, rows = out.setdefault(
                        name, (self.ir.stream_read_forms[key], []))
                    origin = self.stream_origin(*key)
                    rows.append((layer, origin.iteration
                                 if origin.iteration is not None
                                 else self.by_uid(origin.uid).name))
        return {name: (form, tuple(rows))
                for name, (form, rows) in out.items()}

    @memo_property
    def _block_stream_reads(self) -> Dict[int, Mapping[str, Tuple[int, ...]]]:
        return {}

    def stream_distances(self) -> Dict[str, int]:
        """Per inner stream, the most layers between a reader and the layer
        that produced the value it reads, within one stack."""
        out: Dict[str, int] = {}
        for (stack, layer, name), (source, _at) in self.ir.stream_reads.items():
            if source != stack:
                continue
            origin = self.stream_origin(stack, layer, name)
            if origin.iteration is None:
                # The stream's `seed`, read at the stack's entry.
                continue
            out[name] = max(out.get(name, 0), layer - origin.iteration)
        return out

    @memo_property
    def _by_uid(self) -> Mapping[int, SpecIRNode]:
        return {n.uid: n for n in self.nodes()}

    def by_uid(self, uid: int) -> Evaluable:
        return self._by_uid[uid]

    @memo_property
    def blocks(self) -> Tuple[BlockTemplate, ...]:
        """One `BlockTemplate` per block instance, in spec.model order."""
        return tuple(BlockTemplate(meta=meta, roots=tuple(roots))
                     for meta, roots in self.ir.blocks)

    @memo_property
    def copy_factors(self) -> Layer:
        """`Layer[Tuple[CopyFactor, ...]]`: the named counts each
        weight-bearing leaf is repeated by, outermost first.

        On the PROGRAM and not on an `Analysis`, because nothing here depends
        on a binding: `multiplicity` and the declarations that produced it —
        a block invocation's count, each ancestor `repeat:` — are baked when
        the tree is built. `Analysis.copy_factors` delegates here for the
        consumers that hold a walk.

        The domain is `iter_leaves` — this repository's one definition of a
        leaf that puts a tensor on disk — and NOT the nodes a walk stored. A
        walk is per phase and per stage, so its storage layer leaves out the
        stacks it did not evaluate (MiMo-V2.6 walks 41 of its 73 on-disk
        leaves) while a weight breakdown is about the whole checkpoint.

        Total over those leaves, never partial: an entry of `()` means "not
        repeated", so an ABSENT entry means "not a weight leaf" and nothing
        else. A layer that used both spellings for one thing could not
        refuse a leaf repeated 61 times whose factors went missing.
        """
        return Layer(self, {
            leaf.ir_node.uid: copy_factors_of(
                self, self.by_uid(leaf.ir_node.uid))
            for leaf in iter_leaves(self.resolved)})

    def __repr__(self) -> str:
        return f"Program({self.spec.name}, nodes={len(self._by_uid)})"


# -- whole-stream joins, lowered before the walk --------------------------
#
# Whole-stream joins lowered to graph nodes, before the walk.
#
# A `read: concat` of an appended stream — and a model-level `op:` that names
# one with `axis:` — is `torch.cat` of the stream's k elements in the source,
# after a crop of each when the stream publishes one (Llava's `hs[:, 1:]`
# ahead of `torch.cat(..., dim=-1)`). k is known only at resolve, so the spec
# cannot write that `concat`; this pass writes it, on a copy of the IR
# (`Program` is immutable), as real nodes: a `slice` view per element when
# the stream declares `slice`, then one `concat` of the k inputs, ahead of the
# reading node, which then reads the join. They carry fresh uids
# (`SpecIRTree.make_node`) and provenance `lowering`, and the walk prices, lays
# out and draws them like any node: the join reads k elements and writes
# their sum along the axis, a crop moves nothing. k = 1 is a direct read — the
# element, or its crop — with no `concat` and no copy.
#
# Placement. A model-level reader's nodes are top-level siblings just before
# it, and `Program.stacks` hands them to its stage (`OpStage.prelude`). A
# component reader's are siblings just before it in its parent, so they run
# under the count of the template they sit in, once per layer it stands for.
# Every reading node gets a join of its own.
#
# The element inputs are refs by name (`WholeStreamPairing.element_ref`) that
# the walk binds to the elements the writing stack published — a reference
# out of the reader's scope, as every stream read is.

def _lower_concat_reads(ir: SpecIRTree, rows: Sequence[Any],
                        streams: Mapping[str, TreeStream]) -> SpecIRTree:
    """A copy of `ir` with each concat row of `Program.whole_stream_pairings`
    (`rows`) lowered to its crops and join. `streams` is the tree
    stream table, for what each writer publishes."""
    # The op declarations are the registry's own objects, shared by every
    # tree; copying them would give each node a private copy of its op.
    memo = {id(node.declaration): node.declaration for node in _ir_nodes(ir)
            if node.declaration is not None}
    tree = copy.deepcopy(ir, memo)
    by_uid = {node.uid: node for node in _ir_nodes(tree)}
    for row in rows:
        _Join(tree, by_uid[row.reader_uid], row,
              streams[row.stream].slice).lower()
    return tree


def _ir_nodes(tree: SpecIRTree) -> Iterator[SpecIRNode]:
    yield from _depth_first(tree.top_level)
    for _meta, roots in tree.blocks:
        yield from _depth_first(roots)


class _Join:
    """One reading node's join, written into `tree` ahead of it."""

    def __init__(self, tree: SpecIRTree, reader: SpecIRNode, row: Any,
                 crop: Optional[Tuple[int, int]]) -> None:
        self.tree = tree
        self.reader = reader
        self.row = row
        self.crop = crop

    def lower(self) -> None:
        row, reader = self.row, self.reader
        elements = [row.element_ref(j) for j in range(row.visible)]
        refs = [InputRef(src=None, external_sid=name) for name in elements]
        made: List[SpecIRNode] = []
        if self.crop is not None:
            axis, start = self.crop
            made = [self._node(f"{reader.name}_{row.stream}_crop{j}", "slice",
                               [ref], axis=axis,
                               dims={"start": start})
                    for j, ref in enumerate(refs)]
            refs = [InputRef(src=node) for node in made]
        if len(refs) > 1:
            made.append(self._node(f"{reader.name}_{row.stream}_concat",
                                   "concat", refs, axis=row.axis))
        target: Optional[SpecIRNode] = made[-1] if made else None
        siblings = self._siblings()
        at = next(i for i, node in enumerate(siblings) if node is reader)
        siblings[at:at] = made
        if target is not None:
            self._read_from(reader, target)

    def _siblings(self) -> List[SpecIRNode]:
        """The list the reader sits in: its parent's children, or the root
        list of the scope it heads."""
        parent = self.reader.parent
        if parent is not None:
            return parent.children
        for roots in (self.tree.top_level,
                      *(roots for _meta, roots in self.tree.blocks)):
            if any(node is self.reader for node in roots):
                return roots
        raise ValueError(
            f"{self.reader.display_path}: a reading node in no scope of the "
            f"graph")

    def _node(self, name: str, op: str, refs: Sequence[InputRef], *,
              axis: int, dims: Optional[dict] = None) -> SpecIRNode:
        """A crop or join beside the reader: the reader's scope, role and
        dispatch zone, no weight, no count of its own."""
        reader, parent = self.reader, self.reader.parent
        node = self.tree.make_node(
            name=name, op=op, is_component=False,
            role=reader.role, template=reader.template,
            stack_name=reader.stack_name,
            component_path=reader.component_path,
            component_scope=reader.component_scope,
            multiplicity=parent.multiplicity if parent is not None else 1,
            active_multiplicity=(parent.active_multiplicity
                                 if parent is not None else 1),
            in_dispatch_zone=reader.in_dispatch_zone,
            axis=axis,
            dims=dict(dims or {}),
            dims_raw={key: str(value) for key, value in (dims or {}).items()},
            inputs=tuple(ref.external_sid or ref.src.name for ref in refs),
            inputs_obj=tuple(refs),
            is_compute_only=True,
            provenance="lowering")
        node.parent = parent
        return node

    def _read_from(self, node: SpecIRNode, target: SpecIRNode) -> None:
        """Point every ref to the stream in `node`'s subtree at `target`.
        A component body names the caller's tensor by the caller's name,
        so the call and its body move together; the stream's name stays
        on the edge (`InputRef`)."""
        name = self.row.stream
        node.inputs_obj = tuple(
            InputRef(src=target, port="", external_sid=name)
            if ref.external_sid == name else ref
            for ref in node.inputs_obj or ())
        for child in node.children:
            self._read_from(child, target)
