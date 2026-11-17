"""The IR the spec walks into — the data types.

`SpecIRTree` (model root plus per-block node trees) is the canonical
structure every consumer reads: the evaluator's `Program`, the renderer,
the validate matchers. `LeafTensor` is one weight-bearing leaf with its
checkpoint name; `BlockMeta` is one block template's invocation metadata.
Nothing here traverses the authored spec or evaluates cost — `build`
constructs these, `projections` reads them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Tuple

from backend.arch.ops import OP_REGISTRY, ShapeKind
from backend.arch.schema import HISTORY_AXIS
from backend.arch.spec_id_compose import (
    compose_aggregation_id,
    compose_child_spec_id,
)


@dataclass(frozen=True)
class StackReads:
    """Who one model layer reads, copied at enumerate time.

    Keyed by `layer_path` on `SpecIRTree.stack_reads`, and copied onto
    every `BlockMeta` of that path. A group that declares carry (DeepSeek
    `mtp`) has a path and no BlockMeta — tree map only.
    """
    name: str = ""
    each: Tuple[str, ...] = ()
    prefix: Tuple[str, ...] = ()
    concat: Tuple[Tuple[str, int], ...] = ()
    # The inner streams this layer declares (`ModelLayer.inner_streams`).
    inner: Tuple[str, ...] = ()
    carried: Optional[str] = None
    # Isolated-component `shape:` per stream name, copied off YAML so
    # Evaluator does not peek `layer.streams`. Empty for most stacks.
    shapes: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class TreeStream:
    """One appended stream on the expanded tree.

    Re-expression of the loader's `AppendedStream`: writer identity is
    `(stack_name, layer_path)`, not a `ModelLayer`. `iterations` /
    `writers` are copied, not re-resolved (Kimi-K3 keeps sentinel `-1`).
    """
    name: str
    stack_name: str
    layer_path: Tuple[int, ...]
    iterations: Tuple[int, ...]
    writers: Tuple[int, ...]
    from_: str
    slice: Optional[Tuple[int, int]]
    kind: str


#: One layer of an inner stream's stack: the `layer_path` of the model layer
#: whose iterations it counts (`iteration_owner`), and the iteration.
StreamAt = Tuple[Tuple[int, ...], int]

#: The iteration an inner stream's `seed` stands at: the stack's entry,
#: before its first layer — -1, as an appended stream counts it.
STACK_ENTRY = -1


@dataclass(frozen=True)
class SpineEntry:
    """One `spec.model` entry in authored order; group children stay nested.

    The interleave `embed_tokens → layers → norm → lm_head` (plus aux
    `group:` stacks) lives here so `Program.stacks` and rendering do not
    re-walk YAML for order. `inputs` is the model-level attachment, not
    BlockGraph formals. `stack_name` is the Program stack this entry
    lives on (`""` = backbone) or, for a group, the stack it opens.
    `repeat` is the entry's `repeat:` resolved to a count, None when the
    entry declares none. An entry is identified by `layer_path`.
    """
    layer_path: Tuple[int, ...]
    name: str
    kind: Literal["op", "block", "assignments", "group"]
    inputs: Tuple[str, ...] = ()
    stack_name: str = ""
    aux_kind: Optional[str] = None
    group: Optional[str] = None
    children: Tuple["SpineEntry", ...] = ()
    repeat: Optional[int] = None


@dataclass
class BlockMeta:
    """Metadata for one block-template invocation in the spec.

    A `template:` declaration on `spec.model` produces one BlockMeta;
    each entry in an `assignments:` list produces one (so a model with
    different attention/FFN per layer-range yields multiple BlockMetas
    sharing a stack_name); each `template:` sub-layer of a `group:`
    produces one (with `aux_kind` carried through).

    Fields:
      template_name : key into `spec.blocks` to fetch the BlockGraph.
      count         : number of physical layers using this template.
      stack_name    : the spec.model layer's `name` (e.g. "layers",
                      "model.layers", "transformer.encoder.layers", "mtp").
                      Renderer / weight breakdown use this to label
                      block-group cards.
      aux_kind      : when this BlockMeta lives under a `group: true`
                      entry, the group's `aux_kind` field
                      (mtp / vision / audio / speculative). None for
                      main-body blocks.
      group_name    : when this BlockMeta lives under a `group: true`
                      entry, the OUTER group's `name` (e.g. "visual",
                      "audio", "mtp"), which is also its `stack_name`.
                      `group_name` is the aux-bucketing key — every leaf
                      under the same outer group buckets together
                      regardless of which inner stack it lives in.
      layer_indices : for assignment-mode entries, the explicit set of
                      layer indices this assignment covers. None for
                      template-mode entries (contiguous range).
      dim_overrides : per-assignment dim overrides (`assignments[].dims`),
                      applied when walking the BlockGraph.
      dim_overrides_raw : the same rebinding as the spec WROTE it —
                      `n_kv_heads: n_kv_heads_full`, not `n_kv_heads: 1`.
                      An assignment's `dims:` is an invocation binding like
                      any other, but the invocation has no IR node to carry
                      it, so the resolved ints were all that survived and
                      every leaf below reported the template's LOCAL axis
                      name. Identity is not a display detail — it decides
                      which world divides the axis — so gemma-4's full
                      layers claimed the sliding-window `n_kv_heads` and
                      sharded on its extent.
    """
    template_name: str
    count: int
    stack_name: Optional[str]
    aux_kind: Optional[str] = None
    group_name: Optional[str] = None
    # Where this stack's tensors live in the checkpoint, resolved against its
    # scope (`schema.CheckpointAddress`): the stack's `prefix` and `index`
    # when it is indexed, and always the `root` its scope hangs off. Set by
    # `enumerate_blocks` on every meta; consumed by `checkpoint_prefix`.
    checkpoint: Optional[Any] = None
    # For addresses whose index follows the backbone: the main backbone's
    # layer count — the first tail index this stack's tensors ship under
    # (DSv3 `model.layers.61.*` ← n_layers = 61). Resolved in
    # `enumerate_blocks` from the backbone stack's own `repeat:` /
    # `assignments:` binding, never hardcoded. None when no backbone
    # stack exists or its count is unbound — `checkpoint_prefix` then
    # refuses to compose a name.
    backbone_layers: Optional[int] = None
    layer_indices: Optional[Tuple[int, ...]] = None
    # The resolved `repeat:` of the group enclosing this invocation's stack,
    # 1 outside a repeated group. Already multiplied into `count`; kept
    # apart because the product cannot be factored back (Nemotron-3-Ultra's
    # MTP: an outer `repeat: n_mtp_layers` around a two-variant per-layer
    # stack), and the walk of a carried stream needs to know whether the
    # chain runs more than once.
    outer_repeat: int = 1
    #: `count` and `outer_repeat` as the spec WROTE them — `n_layers`, not
    #: `61` — mirroring `repeat_raw` on a node and for the same reason: an
    #: int cannot say which declaration named it, and a consumer that has to
    #: tell a layer loop from an expert bank needs the symbol. None where
    #: there is no symbol to give: a count that came from an assignment's
    #: layer indices, or a declaration that wrote a literal.
    #:
    #: Two fields rather than one, because `count` is their PRODUCT in a
    #: group (`group_repeat * sub_count` — Nemotron-3-Ultra's MTP is an
    #: outer `repeat: n_mtp_layers` around a two-variant per-layer stack).
    #: A single factor would be a number with no declaration behind it.
    count_raw: Optional[str] = None
    outer_repeat_raw: Optional[str] = None
    dim_overrides: Dict[str, Any] = field(default_factory=dict)
    dim_overrides_raw: Dict[str, str] = field(default_factory=dict)
    # Where this invocation is DECLARED, so an editor can change that one
    # declaration and nothing else. `layer_path` indexes `spec.model` —
    # `(i,)` for a top-level layer, `(i, j)` for sub-layer `j` of group `i`;
    # `assignment_index` is the position in that layer's `assignments` (None
    # for a `template:` declaration). A template name does not locate a
    # declaration: two assignments may bind one template (GLM-5.1), and a
    # swap that re-synthesizes to an existing name makes two share one.
    layer_path: Tuple[int, ...] = ()
    #: A call of an earlier template call, or a template call in a call of
    #: a group (`ModelLayer.call`): the called layer's name; every leaf of this call reads
    #: the corresponding leaf of that one (`resolve_tied_weights`).
    tied_to: Optional[str] = None
    #: That earlier call's `layer_path`.
    tied_to_path: Optional[Tuple[int, ...]] = None
    assignment_index: Optional[int] = None
    # The checkpoint address of the call site this invocation stands in
    # for, when the spec is one component isolated out of a model
    # (`ResolvedSpec.site`): `layers.X.attn.` for DeepSeek-V4's CSA
    # attention. None everywhere else — the stack decides the prefix.
    address: Optional[str] = None
    # For a template that reads its stack's `read: all` stream: how many
    # elements of it every layer of this invocation sees, `((stream, m),)`.
    # The declaration's layers are split by it into one invocation per
    # value (`_BlockEnumerator.visibility_groups`), because the history a
    # layer reads is part of its shape, and each binds `Sar` to its m.
    # Empty for every other invocation. With `layer_path` and
    # `assignment_index` it identifies the invocation.
    visible: Tuple[Tuple[str, int], ...] = ()
    # The dims that tell this invocation from another of the same template
    # in the same stack, `((dim, value), …)`: two `per_layer.assign`
    # entries may bind one template at different `dims:` (DeepSeek-V4.1's
    # pure CSA consumers, at compress ratio 2 and 1). Each holds its own
    # layers' weights and computes its own shapes, so each is its own row
    # of every per-node table, and `label` carries what differs. Set by
    # `enumerate_blocks` (`distinguish_by_dims`); empty for an invocation
    # no other one shares a label with.
    distinct_dims: Tuple[Tuple[str, str], ...] = ()
    # This call's formals (`BlockGraph.inputs`, schema default `("x",)`)
    # and declared forward-root names (`BlockGraph.outputs`). Copied at
    # enumerate time; a missing interface is a broken tree.
    inputs: Tuple[str, ...] = ("x",)  # schema default
    outputs: Tuple[str, ...] = ()
    # Per-stack read sets of this invocation's layer_path. Copied at
    # enumerate time so a BlockStage does not dict-lookup the tree map.
    reads: StackReads = field(default_factory=StackReads)
    # Recipe `stream_out` / `reads` / `kv_cached` of this template's
    # BlockGraph, copied at enumerate. Nested copies live on the call
    # SpecIRNode. Not the graph object.
    stream_out: Tuple[Tuple[str, str], ...] = ()
    stream_reads: Tuple[str, ...] = ()
    kv_cached: Tuple[str, ...] = ()
    # The `layer_path` of the model layer whose iterations this invocation
    # runs (`iteration_owner`): its own layer when that repeats, else the
    # repeated group around it (DeepSeek's `mtp`). Inner streams are
    # declared on that layer and its layers are numbered in it.
    iteration_owner: Tuple[int, ...] = ()
    # The short name this template goes by (`MLA + MoE FFN`,
    # `derive_block_label`): every page shows a block by it; the template
    # name stays its identity. Copied at enumerate.
    short_name: str = ""
    # The name of the KIND of block the template is: told apart from the
    # other templates of its stack by the wrapper it calls, not by the
    # components bound into its slots (`distinct_block_labels`,
    # `wrappers_only`). Several templates share it when they differ only in
    # a component they bind (DeepSeek-V4.1's KV-owning and KV-borrowing CSA
    # layers); a wrapper that carries a module of its own (its Engram
    # layers' `dsv41_hc_block_engram`) is another kind. A view that shows
    # the stack by kind groups on it; `short_name` names each member.
    family_name: str = ""
    # What tells this template from the other templates of its kind in its
    # stack: each innermost slot bound differently among them, the
    # component bound there and that component's declared `description`
    # (`block_label.differing_slots`). Empty for a kind of one template.
    distinct_slots: Tuple[Tuple[str, str, str], ...] = ()
    # The wrapper component this template calls and its bindings — the
    # template the loader expanded inline shorthand into, or the one a
    # hand-written one-node adapter binds — None when it is neither.
    wrapper_op: Optional[str] = None
    wrapper_params: Dict[str, str] = field(default_factory=dict)

    @property
    def indexed(self) -> bool:
        """Whether this is a stack of layers the checkpoint indexes — a
        `repeat:` / `per_layer:` stack with an address — rather than a
        one-shot wrapper invoked once (a vision tower, an exit reduce)."""
        return self.checkpoint is not None and self.checkpoint.prefix is not None

    @property
    def label(self) -> str:
        """The name this invocation's nodes carry as their `template`, and
        so the first half of their `aggregation_id`: the template's name,
        and for a derived group the history length its layers see
        (`block_kda_moe[Sar=3]`), and for an invocation in a group the
        stack it runs in (`qwen35_full_dense_block@mtp`; the backbone's
        stack is the unnamed one). Groups of one template read different
        histories, and each stack holds its own weights, so they are
        different rows of every per-node table: with the path, this is the
        node's identity, whichever other stacks bind the template. Two
        declarations binding one template in one stack at different dims
        carry those dims too (`distinct_dims`,
        `dsv41_hc_block__…[compress_ratio=2]`)."""
        name = (self.template_name if self.group_name is None
                else f"{self.template_name}@{self.group_name}")
        if self.visible:
            seen = ",".join(f"{HISTORY_AXIS}={m}" for _stream, m in self.visible)
            name = f"{name}[{seen}]"
        if self.distinct_dims:
            bound = ",".join(f"{dim}={value}" for dim, value in self.distinct_dims)
            name = f"{name}[{bound}]"
        return name

    @property
    def declaration(self) -> Tuple[Tuple[int, ...], Optional[int], str]:
        """The authored declaration this invocation stands for: where it is
        written (`layer_path`, `assignment_index`) and the template it names.
        The derived groups of one declaration share it (`declared_blocks`
        joins them on it)."""
        return (tuple(self.layer_path), self.assignment_index,
                self.template_name)

    def derived_group(self) -> Optional[Dict[str, Any]]:
        """This invocation's identity as a derived group, in the form a
        report carries it: the declaration it was split from, the history
        length its layers see and which layers those are. None for an
        invocation that is its whole declaration.

        A consumer that shows the model per declaration groups on
        `declaration` and orders the groups on `visible`; the `label` suffix
        is a name, not a field to read these back out of."""
        if not self.visible:
            return None
        ((stream, m),) = self.visible
        layer_path, assignment_index, template = self.declaration
        return {
            "base_template": template,
            "stream": stream,
            "axis": HISTORY_AXIS,
            "visible": m,
            "declaration": {"layer_path": list(layer_path),
                            "assignment_index": assignment_index,
                            "base_template": template},
            "stack_name": self.stack_name,
            "layer_indices": list(self.layer_indices or ()),
            "count": self.count,
        }
@dataclass
class LeafTensor:
    tensor_name: str                 # HF on-disk name
    op: str                          # linear / parameter_vector / embedding / rmsnorm / layernorm / per_head_linear
    shape: Tuple[int, ...]           # tensor shape (out, in) for linear, (n_vocab, d_model) for embedding, etc.
    multiplicity: int                # n_layers * n_shared_experts * ... — physical copies on disk (storage count)
    role: Optional[str]              # attention / ffn / ... — propagated from component_roles
    template: str                    # block template name, "" for top-level
    # spec.model layer name this leaf lives under. For block-internal
    # leaves: the parent BlockMeta.stack_name (e.g. "layers",
    # "language_model.model.layers", "mtp"). For top-level non-block
    # leaves: None for the root, group's name when recursing into a
    # `group: true` entry's sub-layers (so MTP-internal embed_tokens /
    # lm_head get stack_name="mtp"). Lets engine bucket leaves into
    # main vs aux totals without re-walking the spec.
    stack_name: Optional[str] = None
    # Component invocation chain from template root → leaf parent. Each
    # entry is (invocation_name, component_op). Lets consumers reconstruct
    # the spec hierarchy without re-walking the spec — e.g. inference
    # weight breakdown groups leaves by this path to build its tree.
    component_path: Tuple[Tuple[str, str], ...] = ()
    # All dim-expression symbols appearing on this leaf's GraphNode
    # (`dims.in`, `dims.out`, `dim`, etc., with arithmetic broken into
    # individual symbols). Consumers intersect this with a parent
    # component's `parallelism.shard_symbols` to decide if the leaf has
    # a TP-shardable axis (e.g. `n_heads` for attention, `d_ffn_expert` for FFN).
    dim_symbols: frozenset = field(default_factory=frozenset)
    # Symbols naming how many COPIES of this leaf exist (`repeat:`, cumulative
    # through ancestors). Separate from `dim_symbols` because a count is not an
    # axis of the tensor.
    count_symbols: frozenset = field(default_factory=frozenset)
    # Optional richer info (filled when easily derivable; not all consumers need it).
    has_bias: bool = False           # linear-only; bias counted separately on disk as <name>.bias
    in_dim: int = 0                  # linear only
    out_dim: int = 0                 # linear only
    from_custom_op: bool = False     # custom-op-fused linear (gpt-oss mxfp4_fused_linear etc.)
    # Every LeafTensor is produced 1:1 from a weight-bearing SpecIRNode
    # by `ir_to_leaf`. Consumers read `lt.display_path` / `lt.spec_id`
    # (both derive from `ir_node.display_path`). Required for engine-
    # produced leaves; test fixtures may also pass `ir_node` directly.
    ir_node: "SpecIRNode" = None

    @property
    def name(self) -> str:
        """Alias for `tensor_name`. Lets materialize / override-matching
        and downstream consumers read `lt.name` interchangeably with
        `lt.tensor_name` — HF on-disk name semantics, no risk of
        confusion with `spec_id` (which is the structural spec
        node id, a different namespace)."""
        return self.tensor_name

    @property
    def display_path(self) -> Optional[str]:
        """Canonical structural path identifying the leaf's position
        in the IR tree. Derived from `ir_node.display_path` — the sole
        source of truth. None for a leaf with no `ir_node` (a test
        fixture built without one); engine-produced leaves always carry it."""
        return None if self.ir_node is None else self.ir_node.display_path

    @property
    def spec_id(self) -> str:
        """Alias for `display_path` at the storage/report boundary.
        Both return the same structural
        path derived from `ir_node`."""
        return self.ir_node.display_path

    @property
    def aggregation_id(self) -> Optional[str]:
        """Pipeline cost-aggregation key derived from `ir_node` —
        mirrors `SpecIRNode.aggregation_id` so leaf-keyed dicts
        (`leaf_divisors`, physical shapes and access rows) stay consistent
        with report and wire-row keyspaces. None for a leaf with no
        `ir_node` (a test fixture built without one)."""
        return None if self.ir_node is None else self.ir_node.aggregation_id


def validate_overrides_against_spec(
    quant_context, logical_tensors: List[LeafTensor],
) -> List[str]:
    """Every literal override selector must name a spec tensor.

    Regex selectors may match nothing (a vision override on a text-only
    checkpoint) and are not errors.
    """
    if quant_context is None or not quant_context.overrides:
        return []
    errors: List[str] = []
    for i, ov in enumerate(quant_context.overrides):
        selector = ov.select
        selected_stack = selector.stack if selector is not None else None
        component = selector.component if selector is not None else None
        literals = selector.tensors if selector is not None else ov.tensors
        if not literals:
            continue
        name_set = {
            lt.name for lt in logical_tensors
            if (selected_stack is None or lt.stack_name == selected_stack)
            and (component is None
                 or component in (op for _name, op in lt.component_path))
        }
        missing = [t for t in literals if t not in name_set]
        if missing:
            errors.append(
                f"quant_context.overrides[{i}]: {missing!r} "
                f"do not match any spec tensor. Available "
                f"(first 10): {sorted(name_set)[:10]}"
            )
    return errors
@dataclass
class InputRef:
    """Object-reference form of a resolved input edge.

    Carries a direct reference to the producing SpecIRNode plus its
    output port name (port is "" for the main output, named for multi-
    output producers). The walker emits InputRefs in nested form (src
    unresolved, `external_sid` holding the nested-form string lookup
    key); `finalize_ir_tree` resolves the in-scope ones by walking
    the per-scope index and sets `src=node`, clearing `external_sid`.

    `src` remains None for external refs (cross-block / forward) that
    the local scope map can't resolve — those keep their string id
    in `external_sid` for HF tensor matching and diagnostics. The
    `SpecIRNode.resolved_inputs` @property serializes by returning
    `src.display_path` when src is set, else `external_sid`.

    A ref that reads a stream whole through the join `Program.lowered`
    put ahead of it keeps the stream's name in `external_sid` beside its
    `src`: the edge is still the read of that stream, which is how
    `Program.whole_stream_pairings` finds the reader in either graph, and
    a component body bound through it resolves the same name.

    A ref to a sibling that resolved `repeat: 0` never reaches the IR: the
    builder reads a step whose op declares what an absent operand stands
    for through to its live operand (`moe_out: add [combine,
    shared_experts]` with no shared expert is `combine`), and refuses any
    other step.
    """
    src: Optional["SpecIRNode"]
    port: str = ""
    external_sid: str = ""
    #: Whether the name was grafted: a component body's read of one of its
    #: own formals, rewritten to what its caller binds that formal to
    #: (`SpecTreeWalker._resolve_input_ref`, step 3). Inside the call the
    #: grafted name is the formal whatever it spells; an unresolved name
    #: that was not grafted is a stream the body reads, which may share the
    #: spelling (`Scope.read`, in a frame `Scope.calling` opened).
    grafted: bool = False


@dataclass
class SpecIRNode:
    """One node in the spec-walked IR tree. Emitted by `iter_tree` —
    every spec graph node (component invocations, weight-bearing leaves,
    AND non-weight ops like add/silu/mul/softmax) appears as a SpecIRNode.

    Where `LeafTensor` is "the on-disk weight tensors", `SpecIRNode` is
    "the spec graph itself, post-walked": renames applied,
    `<param>` op references resolved, renames applied, dim values
    resolved, role propagated, dispatch-zone marked. Both `engine` and
    `_node_build` (planned) consume this — engine adds per-op FLOPs /
    shape computation on top, `_node_build` adds UI decoration.

    A non-weight op (e.g. `add`, `silu`, `slice`) emits a SpecIRNode
    but does NOT produce a LeafTensor. A component invocation emits a
    SpecIRNode WITH children (the sub-graph's body),
    AND the leaves under it produce LeafTensors.
    """
    # Identity ----------------------------------------------------------
    name: str                       # local name in parent's graph
    op: str                         # resolved op (after <param> substitution)
    is_component: bool              # True for component invocations
    # For a model-level entry: where in `spec.model` it is declared, the
    # `SpineEntry.layer_path` it is the node of. A name is not an address —
    # a group's op may share its name with a model-root op — so a stage
    # joins its node by this. Empty below the model level.
    layer_path: Tuple[int, ...] = ()
    # The `nn.Module` the checkpoint has at this node's position, or None
    # when it has none. The three author-facing states of
    # `schema.GraphNode.module` collapse to these two at build time — OWN_NAME
    # has already been resolved to the node's own name — so every consumer
    # downstream asks one question.
    #
    # `None` is the DEFAULT, which is right for the nodes that do not come
    # from a spec at all: a materialized collective and a fused leaf are not
    # in any checkpoint, and only
    # `build_ir_node` has a name to contribute. The default was briefly `""`,
    # which is not an answer to this question — it composes an empty segment
    # and yields `layers.X.mlp.` with a trailing dot.
    module_name: Optional[str] = None
    # The checkpoint writes this repeated node's copies under their own index
    # right after its module segment (`checkpoint: {index: local}`). A repeat
    # without it writes no index.
    checkpoint_indexed: bool = False
    # Where that index goes when it is not after the node's own module: a
    # model-level op storing one copy inside each layer of the stack at this
    # path (`checkpoint: {prefix, path}`).
    checkpoint_index_after: Optional[str] = None
    # For a model-level entry: the checkpoint path its scope hangs off — the
    # model's or its group's `checkpoint.root`, or empty for an op that
    # declares its own `checkpoint.path`. Unused below the model level, where
    # the block stack's address decides.
    checkpoint_root: str = ""
    # The suffixes the op's tensors are stored under, when the checkpoint
    # does not use the op's own (`GraphNode.tensor_suffix`): declared
    # suffix → stored suffix, e.g. `.weight` → `_weight`.
    tensor_suffix: Dict[str, str] = field(default_factory=dict)
    # When a fusion rewrite synthesized this node to replace a matched naive
    # subgraph, the fused op it came from. A field of its own because it is
    # not a naming fact: it used to ride inside the call-site `rename` dict
    # under a `__fused_marker__` key, which meant a node's provenance lived in
    # whatever field happened to be a mutable mapping at the time.
    fused_from: Optional[str] = None
    from_custom_op: bool = False    # True when op resolves to a custom_op declaration
    #: The op's declaration — `OpSpec` for a registered primitive, the
    #: spec's `CustomOpSpec` for a custom op — resolved ONCE when the tree
    #: is built. Consumers (rendering, phases, kernel_bench) read this;
    #: they do not ask the registry by name. None for component
    #: invocations and template / group layers, which have no op of their own.
    declaration: Any = None
    #: The dtypes a fused launch's plan writes its outputs at, by port
    #: (`FusedNode.output_dtype`). Always None on a spec node: a registered
    #: op states its width on its declaration.
    output_dtype: Optional[Dict[str, str]] = None
    #: The quantised activation format (`ActivationFormat`) a grafted cast
    #: writes on its primary output — the conversion it was grafted for
    #: (`CastLaunch.target`), declared whole: element dtype, which elements
    #: share a scale, and the scales' dtype. Read by the traffic derivation
    #: for the width it bills and the format it publishes
    #: (`Traffic.written_formats`). None on every other node; a cast that
    #: lands on a plain dtype (a dequantise) declares `output_dtype` instead.
    output_format: Optional[Any] = None
    # Display-only per-node formula override from the yaml (`formula:` on
    # a graph node) — the call site's concrete shape semantics where the
    # registry's generic formula would hide them (e.g. which axis a
    # position table broadcasts on differs per component). None for
    # synthesized nodes; never read by the engine or pricing.
    local_formula: Optional[str] = None
    # Spec context ------------------------------------------------------
    role: Optional[str] = None      # nearest enclosing component role
    template: str = ""              # block template name; "" for top-level entries
    stack_name: Optional[str] = None
    component_path: Tuple[Tuple[str, str], ...] = ()
    # For a component invocation: the local names its graph DECLARES as its
    # outputs (`ComponentSpec.outputs`), `outputs[0]` first. Empty for a leaf,
    # and for a component synthesized after the walk (a fusion's replacement
    # node) — nobody declared those, so nobody may claim to know.
    #
    # Carried because the alternative is arithmetic on a declared fact: the
    # output is derivable from the dataflow only when exactly one child has no
    # in-scope consumer, and a component is free to end with a `sinks` /
    # `down_proj_bias` parameter vector (gpt-oss), or with a side computation
    # whose result leaves through a cache rather than the return value
    # (DSv4's `compressor` / `indexer`). Then "the free child" is two or three
    # children and the derivation is silently ambiguous.
    output_names: Tuple[str, ...] = ()
    # Nested-recipe facts copied at emit from the same BlockGraph the walk
    # expands. Empty on a leaf. Not the graph object: formals, extra ports,
    # stream interface, cache slots / recurrent states.
    formals: Tuple[str, ...] = ()
    secondary_outputs: Tuple[Tuple[str, str], ...] = ()
    stream_out: Tuple[Tuple[str, str], ...] = ()
    stream_reads: Tuple[str, ...] = ()
    kv_cached: Tuple[str, ...] = ()
    kv_entries: Tuple[Any, ...] = ()
    kv_states: Tuple[Any, ...] = ()
    kv_label: Optional[str] = None
    attn_variant: Optional[str] = None
    sparse_dcp: Any = None
    # The call's own `params:` as the spec writes them — a component call's
    # slot bindings (`compressor: csa2_shared_latent`), nested templates
    # already synthesized to their names at load. The walk consumes them
    # into the params stack; they stay on the node because a call recorded
    # for isolation (`CallSite`) must re-apply them: an isolated call
    # rebuilt without them leaves the component's `<param>` ops unbound.
    call_params: Tuple[Tuple[str, str], ...] = ()
    # Resolved dim context — int values where possible, else raw expr.
    # `dims` carries every key the spec yaml declared on this node;
    # `dim` is the single-dim shorthand resolved value. Consumers that
    # need raw expression strings (symbolic mode) can fetch via
    # `dims_raw` (the original yaml text, unresolved).
    dims: Dict[str, Any] = field(default_factory=dict)
    dims_raw: Dict[str, str] = field(default_factory=dict)
    # The block-level rebinding this node's whole subtree sits under
    # (`BlockMeta.dim_overrides_raw`), carried on the block's ROOT nodes.
    # It is an invocation binding like `dims_raw`, and it is a field of its
    # own only because the invocation it belongs to — the assignment that
    # picked this template for these layers — has no node in the graph.
    block_dims_raw: Dict[str, str] = field(default_factory=dict)
    # The frame the node sits in, as the tree builder resolved it: the
    # model's dims under every enclosing block assignment's and component
    # call's `dims:`. Shared with its siblings (one dict per frame), and
    # empty on a node a pass grafts after the build. Read through `bound`.
    frame: Dict[str, Any] = field(default_factory=dict, repr=False,
                                  compare=False)
    dim: Optional[Any] = None
    dim_raw: Optional[str] = None
    # Explicit n-D shape (parameter_tensor and similar source ops).
    # Each entry is the resolved-int dim value, or the raw expression
    # if resolution failed (mirrors `dim` fallback).
    shape: Optional[Tuple[Any, ...]] = None
    shape_raw: Optional[Tuple[str, ...]] = None
    # Occurrence-level persistent endpoint contract.  Kept as schema objects:
    # they are immutable declarations and are instantiated only after a
    # deployment supplies concrete process groups.
    weight_axes: Optional[Tuple[Any, ...]] = None
    # The checkpoint's axis order for the stored tensor, as a permutation of
    # the op's weight shape (`GraphNode.stored_axes`); None is the op's own
    # order. `weight_axes` lists the axes in this checkpoint order.
    stored_axes: Optional[Tuple[int, ...]] = None
    # The checkpoint's split of the stored tensor into several files' worth of
    # tensors (`GraphNode.stored_split`), with `parts` resolved: `(spec,
    # parts)`. None is one stored tensor.
    stored_split: Optional[Tuple[Any, int]] = None
    # The checkpoint stores this parameter and no node of the modelled graph
    # reads it (`GraphNode.stored_without_graph_reader`, the schema object,
    # with its `kind`); None when a graph node reads it.
    stored_without_graph_reader: Optional[Any] = None
    # A serving framework can keep this stored weight in host memory
    # (`GraphNode.host_offload`: the option and its source); None when it is
    # on the GPU.
    host_offload: Optional[str] = None
    # How the checkpoint stores the stacked expert axis of this node's weights
    # and of every stacked weight under it (`GraphNode.expert_weight`):
    # `unrolled` or `leading_axis`; None where nothing is declared.
    expert_weight: Optional[str] = None
    # Multiplicity ------------------------------------------------------
    # Cumulative storage and per-token-active counts inherited from
    # outer block templates and any ancestor `repeat:` declarations.
    multiplicity: int = 1
    active_multiplicity: int = 1
    # This node's OWN repeat / active (single level, not cumulative).
    repeat: Optional[int] = None
    active: Optional[int] = None
    # The same two as the spec WROTE them, unresolved — `repeat: n_shared_experts`
    # rather than `2`. Mirrors `dim_raw` / `dims_raw` / `shape_raw`, and for
    # the same reason: a consumer holding a different dim binding than the one
    # this walk resolved against has to re-resolve, and it cannot re-resolve an
    # integer (a block assignment rebinding the count, a per-rank frame).
    repeat_raw: Optional[str] = None
    active_raw: Optional[str] = None
    # How those copies are arranged, from the keyword the node declared them
    # with (`GraphNode.chain` vs `GraphNode.repeat`): True when copy i reads
    # copy i−1 and a reader of this node sees only the LAST, False when every
    # copy reads the node's own input and a reader sees ALL of them. False on
    # a node with no copies.
    #
    # The two cost the same (`repeat` copies of a GEMM are `repeat` copies of
    # a GEMM either way) and differ only in the bytes on the edge leaving the
    # node, which is why the distinction had no carrier until there was a
    # keyword to carry: `docs/design.md` §1.2, `docs/known_limitations.md` §5.2.
    chained: bool = False
    # Topology ----------------------------------------------------------
    inputs: Tuple[str, ...] = ()
    children: List["SpecIRNode"] = field(default_factory=list)
    # Engine-flow markers ----------------------------------------------
    # MoE dispatch zone: True for nodes between any `permute` and
    # `combine` op in the same graph. Engine uses this to swap (B, S)
    # for (M, 1) when computing per-expert FLOPs.
    in_dispatch_zone: bool = False
    # Misc spec fields preserved for downstream display / accounting.
    axis: Optional[int] = None       # axis for axis-aware ops
    #: A `patchify`'s attention segment (`ModelLayer.segment`): `item` or
    #: `temporal_patch`; None on every other node.
    segment: Optional[str] = None
    # Full positional permutation for `transpose` / `permute` (e.g.
    # [0, 2, 1, 3] turns (B,s_ctx,n_heads,d_head) →
    # (B,n_heads,s_ctx,d_head)), carried straight from
    # `GraphNode.axes`. Distinct from the single-axis `axis` above.
    #
    # Needed because the per-op DimMap (`axis_propagation.dim_map`) derives a
    # PERMUTE op's dim-correspondence from exactly this list — `transpose`
    # registers `permute_to=None`, so with `axes` absent the DimMap could not
    # be derived for ANY transpose or permute node, and both the shape
    # cross-check and the sharded-axis transfer silently skipped them.
    axes: Optional[List[int]] = None
    has_bias: bool = False           # linear nodes — bias dim resolved to non-zero
    # Object-ref topology (populated by `finalize_ir_tree` at the tail
    # of `iter_tree`, in parallel with the string-keyed `inputs` /
    # `resolved_inputs` / `spec_id` fields). Consumers migrating off
    # string ids should prefer these.
    parent: Optional["SpecIRNode"] = None
    inputs_obj: Tuple[InputRef, ...] = ()
    uid: int = 0
    # HF on-disk weight name (without `.weight` / `.bias` suffix —
    # `compose_tensor_name` appends those). Populated by
    # `populate_weight_names` for every IR node by walking the
    # parent chain through `compose_weight_name`, skipping the links
    # that contribute no segment (so a Llama attention block's
    # inner `q_proj` resolves to `self_attn.q_proj`, not
    # `self_attn.attn.q_proj`).
    #
    # `display_path` and `aggregation_id` are pure-nested over EVERY
    # node, including the ones the checkpoint has no module for.
    # `weight_name` is HF-naming-only.
    weight_name: str = ""
    # Compute-only nodes (no on-disk weight) that disambiguate their
    # local name across components via a `__<component_scope>` suffix.
    # Stored explicitly so display-path derivation doesn't need to
    # parse spec_id strings. Phase A populates by reverse-engineering
    # from spec_id during finalize; Phase B sets them at walker emit.
    component_scope: Optional[str] = None
    is_compute_only: bool = False
    #: `tied_weight.to` of the layer when its `when:` holds — the model
    #: layer whose stored tensor this node reads instead of holding one.
    #: `finalize_ir_tree` resolves the name to `tied_to_uid`, and
    #: `owns_weight` is what the declaration says, not a second flag.
    tied_to: Optional[str] = None
    tied_to_uid: Optional[int] = None
    #: `tied_weight.checkpoint_copy.src` of an active tie — the checkpoint
    #: still ships this node's own tensor, which serving skips. Set only
    #: beside `tied_to`: an inactive tie owns its weight outright.
    checkpoint_copy_src: Optional[str] = None

    @property
    def processor_step(self) -> Optional[str]:
        """The step of a media processor's work this node's op is
        (`OpSpec.processor_step`); None for an op that is none, and for a
        node with no op of its own."""
        decl = self.declaration
        return decl.processor_step if decl is not None else None

    def bound(self, label: str) -> Optional[Any]:
        """What `label` is bound to at this node before any walk: its own
        `dims:` over the frame it sits in (`frame`), or None when neither
        binds it. The walked answer is `Analysis.node_dims`; this one is
        for a reader that runs before the walk does, on a node the tree
        builder made. A node a pass grafted later has no frame, and asking
        it is refused rather than answered None."""
        if not self.frame:
            raise ValueError(
                f"{self.display_path}: a grafted node has no build-time "
                "frame; read the walked frame (`Analysis.node_dims`)")
        value = self.dims.get(label)
        return value if value is not None else self.frame.get(label)

    @property
    def owns_weight(self) -> bool:
        return self.tied_to is None

    @property
    def out_shape(self) -> Optional[Tuple[Any, ...]]:
        """The output shape this node declares (`ShapeNode.out_shape`): its
        `shape:` where its op is a source — its output IS the tensor it
        stores — and None otherwise, a weight-bearing op's `shape:` being
        its weight's."""
        decl = self.declaration
        if (self.shape is None or decl is None
                or ShapeKind(decl.shape_kind) is not ShapeKind.SOURCE):
            return None
        return self.shape

    @property
    def stores_checkpoint_copy(self) -> bool:
        """A tied node whose tensor the checkpoint carries anyway."""
        return self.tied_to is not None and self.checkpoint_copy_src is not None

    boundary_role: Optional[str] = None
    # Origin marker. "spec" for nodes emitted by the yaml walker
    # (default — every yaml-resolved node carries this). "fusion" for
    # nodes synthesized by a framework fusion rewrite.
    # "comm" for nodes synthesized by `parallelism_comm` to represent
    # an injected collective primitive. The field is a structural barrier
    # — validate / materialize / engine all run BEFORE synthesis and
    # expect to only see `"spec"`; downstream phases that see synthesized
    # nodes can discriminate without parsing names or op strings.
    # "site" for the call an isolated component's stage makes at its call
    # site (`ResolvedSpec.site`): the model's invocation of the component,
    # which holds what the component declares on it (its cache).
    # "lowering" for the `slice` / `concat` a whole-stream join runs ahead
    # of its reader (`Program.lowered`, before the walk): real ops of the
    # source (`hs[:, 1:]`, `torch.cat`) that the spec states as a
    # `read: concat` rather than as nodes.
    # "cast" for the `quantise_activation` a quantised matmul's operand
    # needs, and for the op a cache format's stored transform runs (the
    # rotation on the edge into the cache and on the operand its readers
    # multiply it with), grafted between comm and fusion
    # (`inference.materialize_casts`, after the walk).
    provenance: Literal["spec", "fusion", "comm", "site", "lowering",
                        "cast"] = "spec"

    @property
    def resolved_inputs(self) -> Tuple[Tuple[str, str], ...]:
        """Fully-qualified spec_id form of `inputs`: each raw ref
        resolved to the canonical spec_id of the producing node, with
        wrapper-input aliases walked up the scope chain to their
        caller-side source. Single source of truth for "where does
        this input come from" — engine shape lookup, edge construction,
        validation, and HF tensor matching all key off these.

        Derived from `inputs_obj` (the authoritative object-ref form):
        in-scope refs serialize via `ref.src.display_path` (canonical
        structural), out-of-scope refs serialize via the nested
        string `ref.external_sid` that the walker emitted. The pair
        of forms is intentional and matches pre-property behavior —
        cross-block refs that the local scope index can't resolve
        keep their nested key for downstream wrapper-alias / namespace
        handling.

        Structured (spec_id, port) tuples — port is "" for the main
        output and a named secondary port (e.g. "post", "comb") for
        multi-output producers.
        """
        if not self.inputs_obj:
            return ()
        return tuple(
            ((ref.src.display_path if ref.src is not None else ref.external_sid), ref.port)
            for ref in self.inputs_obj
        )

    @property
    def enclosing_components(self) -> Tuple[str, ...]:
        """The components this node sits inside, outermost first: the op
        half of `component_path`. A quant override's `component:` scope
        is asked of this."""
        return tuple(op for _name, op in self.component_path)

    @property
    def local_path(self) -> str:
        """Full local-name chain from the node's scope root (top-level
        model OR a block template root) down to this node, joined by
        ".". Differs from `spec_id` in that it is the HF namespace:
        every level of nesting is preserved. e.g. inside a Llama block,
        `self_attn.attention.softmax` (local_path) vs
        `self_attn.softmax__attention` (spec_id, Phase A still produced
        by the walker).

        Requires `finalize_ir_tree` to have run (parent chain wired).
        At a scope root (`parent is None`), returns the bare `name`.
        """
        chain: List[str] = []
        node: Optional[SpecIRNode] = self
        while node is not None:
            chain.append(node.name)
            node = node.parent
        return ".".join(reversed(chain))

    @property
    def display_path(self) -> str:
        """Primary-folded + scope-suffixed canonical path, derived from
        the parent chain. Phase A: must equal `spec_id` for every node
        produced by `build_ir_tree` — `finalize_ir_tree` populates
        `component_scope` so the suffix matches.

        Phase C: `spec_id` field is deleted; this property becomes the
        sole canonical name source.
        """
        if self.parent is None or self.parent.provenance == "site":
            # Scope root: top-level model entries are emitted with
            # spec_id = layer.name (no compose call,
            # no scope suffix). Block-template roots come from the
            # walker's `_build_ir_subgraph` and ARE composed (parent
            # spec_id = "" at template root). Both cases collapse to
            # bare name + optional scope suffix. So does the body of an
            # isolated component's site call, which the walker scopes as
            # the stage's own (`build_ir_node(at_site=True)`).
            return compose_child_spec_id("", self.name, self.component_scope)
        return compose_child_spec_id(
            self.parent.display_path, self.name, self.component_scope,
        )

    @property
    def aggregation_id(self) -> str:
        """Pipeline cost-aggregation key composed via
        `spec_id_compose.compose_aggregation_id`. Derived live from
        `template` + `display_path` so fusion mutations and comm-node
        grafts (which reparent or synthesize nodes after the initial
        IR build) always see the right value — no stale storage.
        Shared by analysis reports and wire rows."""
        return compose_aggregation_id(self.template, self.display_path)


@dataclass
class SpecIRTree:
    """Output of `iter_tree(resolved)`: model-root and block structure made
    entirely of `SpecIRNode`s, before analysis layers attach derived facts."""
    top_level: List[SpecIRNode] = field(default_factory=list)
    blocks: List[Tuple[Any, List[SpecIRNode]]] = field(default_factory=list)
    # Appended streams re-expressed at construction. Writer identity is
    # `(stack_name, layer_path)`; iterations/writers are the resolved tuples.
    streams: Dict[str, TreeStream] = field(default_factory=dict)
    # Who-reads per `layer_path`, including group-level carry with no BlockMeta.
    stack_reads: Dict[Tuple[int, ...], StackReads] = field(default_factory=dict)
    # Inner streams, by `(stack, iteration, stream)` of a layer (`StreamAt`
    # plus the name): the layer each reading layer's `reads_from` names, the
    # `reads_from` value as declared, and the node each writing layer
    # writes (uid). The one statement of which layer's write a reader
    # reads; `Program.stream_origin` asks it.
    stream_reads: Dict[Tuple[Tuple[int, ...], int, str], StreamAt] = field(
        default_factory=dict)
    stream_read_forms: Dict[Tuple[Tuple[int, ...], int, str], str] = field(
        default_factory=dict)
    stream_writes: Dict[Tuple[Tuple[int, ...], int, str], int] = field(
        default_factory=dict)
    # Authored interleave of ops / blocks / groups. Group children nested.
    spine: Tuple[SpineEntry, ...] = ()
    #: Each call of a group (`ModelLayer.call`): its `layer_path`, and the
    #: called group's.
    group_ties: Dict[Tuple[int, ...], Tuple[int, ...]] = field(
        default_factory=dict)
    # The `layer_path`s of the groups whose `repeat:` resolved to 0: not in
    # the checkpoint, so nothing beneath them is emitted.
    absent: Set[Tuple[int, ...]] = field(default_factory=set)
    #: Model-level ops a fusion absorbed into a kernel that also absorbed a
    #: later one (`eval.fusion.materialize_tree`): each `layer_path` maps to
    #: the path of the spine op whose position the fused node took, the last
    #: absorbed op in spine order. `Program.stacks` stages the fused node
    #: once, there, and these paths not at all.
    fused_layers: Dict[Tuple[int, ...], Tuple[int, ...]] = field(
        default_factory=dict)
    # blocks[i] = (BlockMeta, root_nodes). The root_nodes list is the
    # post-inline body of the template — for `chatglm_block` and
    # similar 1-node alias adapters, the wrapper is collapsed and the
    # body comes from the inlined sub-graph.
    _next_uid: int = 0
    # Monotonic uid allocator for mutation-time node creation.
    # `wire_parents_and_uids` seeds this with `max(existing_uid) + 1`
    # after build_ir_tree. Runtime graph rewrites
    # call `make_node(...)` to get fresh uids without a post-pass rebuild.
    _uid_owner: Optional["SpecIRTree"] = field(
        default=None, repr=False, compare=False
    )
    # A scope view delegates allocation to the complete tree it aliases.
    # Scope-local roots are the mutation boundary; UID ownership is global.

    @property
    def next_uid(self) -> int:
        """The uid the next `alloc_uid` hands out. A rewrite that mints node
        identities of its own (fusion) reads this so the uids it chooses sit
        above every uid this tree holds or will hand out."""
        if self._uid_owner is not None:
            return self._uid_owner.next_uid
        return self._next_uid

    def alloc_uid(self) -> int:
        if self._uid_owner is not None:
            return self._uid_owner.alloc_uid()
        u = self._next_uid
        self._next_uid += 1
        return u

    def make_node(self, **fields: Any) -> SpecIRNode:
        """Create a SpecIRNode with a fresh tree-unique uid. Use this in
        mutation passes instead of constructing SpecIRNode directly so
        the uid invariant is maintained at the point of mutation rather
        than via a post-pass rebuild.

        `parent` is NOT set here — caller pairs it with `insert_child`
        or `replace_span`, which knows the destination list.
        """
        # A node a mutation pass creates (a grafted collective, a fusion
        # replacement) carries its op's registry declaration like a built
        # leaf does; a fused-kernel name with no registration carries None
        # and its consumers read the span's signature instead.
        if "declaration" not in fields and fields.get("op") and not fields.get("is_component"):
            if fields["op"] not in OP_REGISTRY:
                raise ValueError(
                    f"SpecIRTree.make_node: {fields['op']!r} is not a registered "
                    "primitive. A mutation pass that grafts a kernel with no "
                    "registration says so with declaration=None; the tree "
                    "builder refuses such a name the same way."
                )
            fields["declaration"] = OP_REGISTRY[fields["op"]]
        if "uid" in fields:
            raise TypeError(
                "SpecIRTree.make_node: do not pass `uid`; the tree owns "
                "the allocator. Pass other SpecIRNode fields only."
            )
        return SpecIRNode(uid=self.alloc_uid(), **fields)

    def insert_child(
        self,
        parent: SpecIRNode,
        child: SpecIRNode,
        *,
        before: Optional[SpecIRNode] = None,
    ) -> None:
        """Insert `child` into `parent.children`, wiring `child.parent`
        in the same step. `before=None` appends; otherwise inserts
        immediately before `before` (must already be a child of `parent`).
        """
        child.parent = parent
        if before is None:
            parent.children.append(child)
            return
        try:
            idx = parent.children.index(before)
        except ValueError as e:
            raise ValueError(
                f"insert_child: `before` (spec_id={before.display_path!r}) is "
                f"not in parent.children"
            ) from e
        parent.children.insert(idx, child)

    def replace_span(
        self,
        siblings: List[SpecIRNode],
        matched_set: Set[str],
        replacement: SpecIRNode,
        *,
        parent: Optional[SpecIRNode],
    ) -> bool:
        """If `siblings` contains every spec_id in `matched_set`, remove
        them and insert `replacement` at the position of the first match.
        Wires `replacement.parent = parent` in the same step. Returns
        True on hit, False otherwise.

        `parent` is the SpecIRNode that owns `siblings` (or None when
        `siblings` is a top-level / block root list).
        """
        indices = [i for i, n in enumerate(siblings) if n.display_path in matched_set]
        if {siblings[i].display_path for i in indices} != matched_set:
            return False
        if not indices:
            return False
        first_idx = indices[0]
        for i in reversed(indices):
            del siblings[i]
        siblings.insert(first_idx, replacement)
        replacement.parent = parent
        return True
def iter_block_scopes(tree: "SpecIRTree") -> Iterator["SpecIRTree"]:
    """Yield one mini-`SpecIRTree` per scope — the model-root level plus
    each block — where spec_ids are unique WITHIN the scope.

    Any pass that REWRITES the tree (comm injection, kernel fusion, and
    any future SequenceParallel / overlap / dequant rewrite) MUST iterate
    these scopes instead of building a global `spec_id → node` index over
    the whole tree. spec_ids are TEMPLATE-LOCAL: a hybrid model's dense +
    MoE blocks are different templates that share the same attention /
    norm sub-structure, so the SAME spec_id (`self_attn.attn_apply`,
    `post_attention_layernorm`) occurs in multiple block subtrees. A
    global index collapses the duplicates (last-wins) and the pass
    silently processes only ONE block — the others keep no comm / stay
    unfused. (Cost is unaffected: the pipeline keys by spec_id and
    aggregates via count_per_step; this only bites passes that walk the
    per-block tree structure.)

    Each yielded mini-tree's `top_level` IS the real roots list (same
    object), so in-place mutation by the pass propagates to `tree`.
    Fusion / comm are intra-layer, so no legal rewrite spans a scope
    boundary.
    """
    yield SpecIRTree(
        top_level=tree.top_level, blocks=[], _uid_owner=tree
    )
    for _meta, roots in tree.blocks:
        yield SpecIRTree(top_level=roots, blocks=[], _uid_owner=tree)


# Op classification driven by `OP_REGISTRY[op].weight_shape` —
# each weight-bearing primitive declares its on-disk template in
# `ops.py`. spec_tree resolves that template; no per-op membership
# lists here. See `ir_to_leaf` below for the dispatch.
