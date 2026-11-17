"""GraphNode / BlockGraph — the DAG of compute nodes within a transformer block."""
from __future__ import annotations

from typing import (Any, Callable, Dict, List, Literal, Optional,
                    Tuple, Union)

import backend.arch.dims as dim_registry
import backend.arch.roles as roles
from backend.arch.axis_propagation import call_site_dims_error
from backend.arch.ops_base import DISPATCH_OP
from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from .cache import CacheEntry, CacheSpec
from .claims import AxiomSet
from .custom_ops import CustomOpSpec
from .guards import (_check_quant_suffix_in_arch_spec,
                     _check_weight_dim_geometry)
from .model import BoundaryRole, CheckpointAddress
from .markers import (BINDING_ONLY, LOADER_DERIVED, OWN_NAME, _OwnName,
                       reject_retired_graph_node_naming)
from .weights import (StoredSplitSpec, StoredWithoutGraphReaderSpec, WeightAxisSpec,
                      ExpertWeight,
                      check_stored_axes, check_stored_split)



class ParamSpec(BaseModel):
    """Declares a parameter slot on a parameterized compound op.

    A compound op (`components/*.yaml`) can list `params:` whose values are
    other component names supplied by the caller. Inside the component graph,
    a node references its parameter via `op: <param_name>`; at evaluation
    time the engine looks up the actual component name from the caller's
    `params: { param_name: <real_component> }` mapping.

    Example: `pre_norm` declares `attn` and `ffn` params; a model spec calls
    it with `template: pre_norm, params: { attn: mla_attention, ffn: swiglu_ffn }`.
    """
    kind: Literal["component", "dim"] = "component"
    default: Optional[str] = None  # fallback if caller omits the param
    # HF config fields that name this slot's value, when the model's config
    # carries the choice (`output_gate_type` names the GDN norm-gate
    # activation). Looked up at the config's top level and in its text
    # sub-config; when present, the bound op must be the one whose
    # `config_names` contain the value — validate's param-config check.
    # Not serialized when empty, so a spec's dump (and the spec-match
    # fingerprint over it) moves only for the components that declare one.
    config: List[str] = Field(default_factory=list, exclude_if=lambda v: not v)



class GraphNode(BaseModel):
    """A single node in the block computation DAG.

    All tensor operations — including slicing / concatenation / reshape — are
    first-class ops (`op: slice`, `op: concat`, …). Each node reads from a list
    of upstream node names; there is no edge-level slice/concat ref syntax.
    """
    model_config = ConfigDict(extra="forbid")

    op: str = Field(json_schema_extra=BINDING_ONLY)  # op name from OP_REGISTRY, component name, or `<param>`
    inputs: List[str] = Field(default=[], json_schema_extra=BINDING_ONLY)  # upstream node names
    dims: Dict[str, str] = {}                    # dimension overrides, e.g. {d_model: d_model, n_heads: n_heads}
    dim: Optional[str] = None                    # shorthand for single-dim ops (norm, add)
    # Caller-side param bindings when invoking a parameterized compound op.
    # Engine sees component-name strings (e.g. {attn: mla_attention,
    # ffn: swiglu_ffn}). YAML accepts dict-valued entries for nested
    # instantiations like {inner: {template: pre_norm, params: {...}}};
    # the loader resolves those into synthesized wrapper names before the
    # spec is handed to the engine.
    params: Dict[str, Any] = Field(default={}, json_schema_extra=BINDING_ONLY)
    # Copies of this node, in one of the two arrangements a forward puts them
    # in. They differ in what a reader of the node moves, which is a fact no
    # count can carry, so they are two keywords rather than one with a
    # qualifier. At most one of them, and `copies` is the count either way.
    #
    # Expert bank: repeat × component, only `active` are used per token. Every
    # copy reads this node's own `in:`, the copies sit side by side, and a
    # reader sees ALL of them (`out = Σ expert_i(x)`: the bank its `combine`
    # reads, a shared bank the residual `add` reads, and each leaf of a bank
    # whose `repeat:` sits on the leaves instead of on a wrapper).
    #
    # `repeat:` is the count-based modelling of a bank, and the shape rework
    # replaces it: an expert bank becomes the leading axis of one weight and
    # one batched matmul, so every site here is scheduled to go and the
    # scoreboard's `repeat_sites` counter is the ratchet that gets it to zero
    # (`tools/scoreboard.py`). Do not add new ones.
    repeat: Optional[str] = None                 # total copies stored (params multiplier)
    active: Optional[str] = None                 # copies active per token (FLOPs multiplier)
    # Copies that run one after another: copy i reads copy i−1's output and a
    # reader of this node sees only the LAST — `for blk in blocks: x = blk(x)`,
    # a vision or audio tower's block stack, and MiMo's `accumulate` over its
    # speech banks, whose chained value is the accumulator. The node-level
    # counterpart of a chain stack, which says the same thing one level up
    # with `streams: { hidden: { carries: true } }`; a node threads one value
    # and has no stream to name, so the keyword itself says it.
    #
    # The chaining used to be implicit in the engine's handling of `repeat:`
    # and was declared nowhere, which left the copies of a tower and the
    # copies of an expert bank textually identical while a reader of the one
    # moves N tensors and a reader of the other moves one
    # (`docs/design.md` §1.2). Not serialized when absent.
    chain: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    # A step the forward applies only when a config switch is on: a dim
    # symbol that resolves to 0 or non-zero (the spec binds it, e.g.
    # `n_topk_renorm: config.norm_topk_prob ?? "0"`). At 0 the node is not
    # built. A shape-preserving step hands its first input (the one whose
    # shape it keeps) to its readers; any other step must have only
    # switched-off readers, the steps that exist to feed it (a group limit's
    # group score and group top-k beside its mask). It must be a weightless
    # primitive — the spec-tree builder rejects anything else. Distinct from
    # `repeat: 0`, which removes a branch and leaves its readers without
    # that operand.
    # An absent switch is not serialized, so a spec's dump (and the
    # spec-match fingerprint over it) is unchanged unless it uses one.
    when: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    # A logits projection inside a component: an auxiliary stack's own head
    # (DeepSeek-V3 / GLM-4.5 MTP `shared_head.head`, Step `shared_head.output`).
    # Same meaning as the model-level field: serving computes it only at the
    # sampled positions, so analyze splices a `sample` view onto its input.
    # Not serialized when absent.
    boundary_role: Optional[BoundaryRole] = Field(
        default=None, exclude_if=lambda v: v is None)
    # Per-axis op metadata (zero-FLOP view ops).
    #   axis  — single-axis ops (concat, slice, gather, sum_reduce, …).
    #           -1 = last (default for channel/feature axis, the common case).
    #           Set explicitly for non-last axes (seq / batch / head).
    #   axes  — full positional permutation for `transpose`/`permute`. e.g.
    #           [0, 2, 1, 3] turns (B,s_ctx,n_heads,d_head) →
    #           (B,n_heads,s_ctx,d_head). Length must equal
    #           the upstream tensor's rank.
    axis: Optional[int] = None
    axes: Optional[List[int]] = None
    # The shape of the tensor this node stores, as dim expressions evaluated
    # against the spec's dim bindings: a source op's whole tensor
    # (`parameter_tensor`: `shape: [n_hidden_copies, "n_hidden_copies *
    # d_model"]` → `[4, 28672]`), or a weight-bearing op's weight when it
    # carries axes ahead of the op's own — an expert bank's `linear`,
    # `[n_expert_copies, d_ffn_expert, d_model]` (`ops.weight_leading_axes`).
    shape: Optional[List[str]] = None
    # Runtime layout of every axis in this occurrence's stored
    # tensor, aligned with the checkpoint payload axes: the op's declared
    # weight shape, reordered by `stored_axes` when the checkpoint stores
    # the tensor in another axis order. Sibling tensors (bias, scales) reuse
    # only structurally identical stored axes. ``None`` is an explicit
    # coverage gap, never a request to infer placement from a symbol
    # capability.
    weight_axes: Optional[List[WeightAxisSpec]] = None
    # The checkpoint's axis order for this stored tensor, as a permutation
    # of the weight's axes: checkpoint axis i is the weight's axis
    # `stored_axes[i]`, leading axes included. Qwen3-VL-MoE stores its packed
    # experts as [E, in, out] under a `[E, out, in]` `linear_param`, so it
    # writes `[0, 2, 1]`.
    # Absent means the op's own order and is not serialized, so a spec's
    # dump (and its fingerprint) is unchanged unless it uses one.
    stored_axes: Optional[List[int]] = Field(
        default=None, exclude_if=lambda v: v is None)
    # The checkpoint stores this tensor as several tensors cut along one
    # logical axis (`StoredSplitSpec`): Qwen3.8's n-gram table as 128
    # `shard_{i}` row blocks. Absent means one stored tensor and is not
    # serialized.
    stored_split: Optional[StoredSplitSpec] = Field(
        default=None, exclude_if=lambda v: v is None)
    # The checkpoint stores this parameter and no node of the modelled graph
    # reads it (`StoredWithoutGraphReaderSpec`: the forward never reads it,
    # or reads it in a step the graph does not model; why, and the source
    # that shows it). Absent means a graph node reads it; not serialized.
    stored_without_graph_reader: Optional[StoredWithoutGraphReaderSpec] = Field(
        default=None, exclude_if=lambda v: v is None)
    # A serving framework can keep this stored weight in host memory and
    # gather from it there (pinned memory over UVA) rather than on the GPU;
    # the string names the framework option and the source that shows it
    # (vLLM's Engram `cpu_offload`). A deployment that offloads drops it from
    # the GPU's weights (`Program.host_offloadable`). Absent means the weight
    # is on the GPU; not serialized.
    host_offload: Optional[str] = Field(
        default=None, exclude_if=lambda v: v is None)
    # How the checkpoint stores the expert axis (the leading `n_expert_copies`
    # axis their `shape:` declares) of this node's weights and of every
    # weight under it: `unrolled` or `leading_axis`
    # (`schema.weights.ExpertWeight`). Not serialized when absent.
    expert_weight: Optional[ExpertWeight] = Field(
        default=None, exclude_if=lambda v: v is None)
    # The checkpoint's module names for the nodes of the component this node
    # invokes, keyed by their component-internal names. Same grammar as
    # `module` below.
    #
    # Needed because whether a module is there, and what it is called, are
    # facts about a CHECKPOINT, and one library component is shared by
    # checkpoints that disagree. `mtp_block_shared_head` declares its inner
    # block absent — DeepSeek and GLM store an MTP layer flat in the main
    # `layers` index space — while Qwen3.5's head holds
    # `self.layers = nn.ModuleList(...)` at that position (vLLM
    # `qwen3_5_mtp.py`), so its adapter says
    # `inner_modules: { transformer: layers }`.
    #
    #     inner_modules: { qkv: query_key_value, o_proj: dense }
    #
    # `rename:` / `hf_inner:` used to say this by rewriting the node's
    # structural name. That conflated two namespaces: a node called `qkv`
    # in the spec and `query_key_value` on disk is not two nodes, and making
    # the spec adopt the checkpoint's spelling meant every structural id —
    # display_path, aggregation_id, the key a measured row joins on —
    # carried a name the architecture never used. Those keys are refused
    # (`RETIRED_GRAPH_NODE_NAMING`); this field is the only spelling.
    inner_modules: Optional[Dict[str, Optional[str]]] = Field(
        default=None, json_schema_extra=BINDING_ONLY)
    # Optional call-site role override. Lets a graph that inlines its
    # FFN/attention as primitive linears (instead of invoking a roled
    # sub-component) tag the relevant node with `role: ffn_moe` so the
    # engine still classifies its subtree correctly. Takes precedence
    # over the component-declared role.
    role: Optional[str] = None

    @field_validator("dims")
    @classmethod
    def _mask_is_declared_not_bound(cls, v: Dict[str, str]) -> Dict[str, str]:
        # `s_pairs` is the PHASE's output — how many (query, key) pairs the
        # step's tokens fill under a causal mask — or a narrowing op's.
        # Whether the mask exists is the attention's fact, `is_causal`;
        # overriding the pairs stated the second fact through the first.
        if "s_pairs" in (v or {}):
            raise ValueError(
                "`s_pairs` is bound by the phase, not by a node. An "
                "attention that runs without a causal mask declares "
                "`is_causal: \"0\"` on its score, softmax and apply nodes "
                "(and `is_chunked: \"1\"` when its window is fixed groups of "
                "`s_window` positions rather than a sliding band).")
        dim_registry.refuse_written_derived(v, "node `dims:`")
        return v

    @field_validator("role")
    @classmethod
    def _role_is_registered(cls, v: Optional[str]) -> Optional[str]:
        # The vocabulary is `backend/arch/roles.yaml`; a Literal here was a
        # second copy of it, and the two copies had drifted.
        return roles.require_registered(v)
    # The `nn.Module` the CHECKPOINT has at this node's position — which is a
    # different question from what the spec calls the node. A checkpoint path
    # has one segment per module, so this is what the node contributes to it.
    #
    #   absent      the node's own name. True of ~97% of nodes: the two
    #               vocabularies agree and nothing is written down.
    #   `null`      no module here. The spec needed a node to hang a
    #               component call on and the file has nothing at that
    #               position — DeepSeek stores an MTP layer as one more entry
    #               in the main `layers` index space
    #               (`model.layers.61.self_attn.q_a_proj.weight`), so the
    #               inner block's leaves hang directly off the MTP prefix.
    #   `"name"`    the module is called that. ChatGLM's fused QKV projection
    #               is `qkv` in the spec and `query_key_value` on disk.
    #
    # Only naming is affected. The node still evaluates, still costs, still
    # owns its children; it is simply not a level of the checkpoint's path,
    # or is a level under another name.
    #
    # Three states need three values, so "absent" is `OWN_NAME` rather than
    # an absence pydantic cannot tell from a written `null`. Encoding it as
    # the default of an `Optional[str]` and recovering the difference from
    # `model_fields_set` worked and was a trap: `model_copy` does not update
    # that set, so a call-site override to `null` read as "absent" and
    # silently fell back to the node's own name.
    # `Union` rather than `Any` so the sentinel's own pydantic schema
    # engages — an `Any` field serializes unknown runtime types by
    # raising, which is the exact failure the sentinel's serializer
    # exists to prevent.
    module: Union[str, _OwnName, None] = Field(
        default=OWN_NAME, json_schema_extra=BINDING_ONLY)

    @field_validator("module")
    @classmethod
    def _module_is_one_of_three(cls, v: Any) -> Any:
        if v is OWN_NAME or v is None or isinstance(v, str):
            return v
        raise ValueError(
            f"`module` must be a string (the checkpoint's name for this "
            f"position), `null` (no module here), or omitted (the node's own "
            f"name) — got {v!r}")

    # `checkpoint: {index: local}` on a repeated node: the checkpoint writes
    # each copy under its own index right after this node's module segment
    # (`vision_tower.encoder.blocks.<i>.…`). A `repeat:` alone writes no
    # index — DeepSeek-V2 stores its two shared experts as one fused module.
    # Not serialized when absent.
    checkpoint: Optional[CheckpointAddress] = Field(
        default=None, exclude_if=lambda v: v is None)
    # The suffixes this node's tensors are stored under, keyed by the op's
    # declared tensor (`weight`, `bias`): `nn.MultiheadAttention` keeps its
    # fused projection as the bare parameters `in_proj_weight` /
    # `in_proj_bias`, so its `in_proj` linear declares `{weight: _weight,
    # bias: _bias}`. Not serialized when absent.
    tensor_suffix: Optional[Dict[str, str]] = Field(
        default=None, exclude_if=lambda v: v is None)
    # Author opt-out for the quant-metadata-suffix lockdown. Default
    # None means "let the validator decide by name". Set to `false`
    # when an architectural `parameter_vector` has a checkpoint name
    # ending in a registered quant scheme's `metadata_siblings`
    # suffix (e.g. `.scale` colliding with FP4) — declares this is
    # real arch state, not engine-materialized quant metadata.
    quant_metadata: Optional[bool] = None
    # Display-only caption override for the op's registry formula. One
    # registered op can carry different local shape semantics at different
    # call sites — `position_table_add` is a window-axis broadcast inside
    # DSv4's compressor (`unflatten(wgate)[.., p, :] + ape[p, :]`) but a
    # patch-index lookup in a ViT tower — and only the use site knows
    # which one it is. The renderer shows this instead of the registry's
    # generic formula when present; engine, pricing, and spec identity
    # never read it. `exclude=True` keeps it out of the spec dump, so the
    # spec-match fingerprint (and every golden keyed on it) is untouched
    # by a card getting a clearer caption.
    formula: Optional[str] = Field(default=None, exclude=True)
    # Per-input port resolution, populated by SpecLoader after
    # component + custom_op assembly. One `(parent_name, port)`
    # tuple per entry of `inputs`, matching index-for-index. For
    # single-output refs `port == "out"`; for multi-output ops with
    # `<node>.<port>` refs, `port` is the secondary-output name.
    # Engine reads this directly instead of calling `parse_input_ref`
    # at every primitive eval — port disambiguation is a static
    # property of the surrounding graph + op registries, so it's
    # build-time work not runtime work.
    # Empty tuple = not yet resolved (hasn't been through
    # `loader.resolve_block_input_ports` — primarily during spec
    # construction; engine asserts this is filled before use).
    resolved_inputs: Tuple[Tuple[str, str], ...] = Field(
        default=(), json_schema_extra=LOADER_DERIVED)

    @model_validator(mode="before")
    @classmethod
    def _retired_naming(cls, data: Any) -> Any:
        return reject_retired_graph_node_naming(data)

    @field_validator("op")
    @classmethod
    def _nonempty_op(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("graph node op must be a non-empty, trimmed name")
        return value

    @model_validator(mode="after")
    def _contextual_fields(self) -> "GraphNode":
        if self.repeat is not None and self.chain is not None:
            raise ValueError(
                "GraphNode declares both `repeat:` and `chain:`, which say "
                "different things about the same copies — `repeat:` that "
                "every copy reads this node's `in:` and a reader sees all of "
                "them, `chain:` that copy i reads copy i−1 and a reader sees "
                "only the last. Declare the one the forward does")
        if self.active is not None and self.repeat is None:
            raise ValueError(
                "GraphNode.active is only valid with `repeat`: it is the "
                "subset of a bank's copies a token runs, and the copies of a "
                "`chain:` all run")
        if self.boundary_role is not None and (
                self.boundary_role != "output_head" or self.op != "linear"):
            raise ValueError(
                "GraphNode.boundary_role is only `output_head`, on `op: linear`: "
                "a component may own a logits head, but the input embedding "
                "is the model's")
        if self.when is not None:
            if not self.when.strip():
                raise ValueError("GraphNode.when must name a dim symbol")
            if not self.inputs or self.copies is not None:
                raise ValueError(
                    "GraphNode.when is only valid on a node with an input "
                    "and no copies (`repeat:` / `chain:`): an off step hands "
                    "its first input to its readers")
        # The shape of the tensor this node stores: a `parameter_tensor`'s
        # whole tensor, or a weight-bearing op's weight with any leading
        # axes ahead of the op's own (`ops.weight_leading_axes`). An op
        # that stores nothing is refused where it is evaluated.
        if self.shape is not None and (
                not self.shape or any(not axis.strip() for axis in self.shape)):
            raise ValueError(
                "GraphNode.shape must contain non-empty axis expressions")
        if self.axes is not None:
            if self.op not in {"transpose", "permute"}:
                raise ValueError(
                    "GraphNode.axes is only valid on `transpose` or `permute`")
            if not self.axes:
                raise ValueError("GraphNode.axes must be non-empty")
        if self.weight_axes is not None and not self.weight_axes:
            raise ValueError(
                "GraphNode.weight_axes must contain one contract per stored axis")
        check_stored_axes(self.stored_axes, self.weight_axes)
        check_stored_split(self.stored_split, self.expert_weight)
        if self.checkpoint is not None:
            self.checkpoint.only("index", where=f"graph node {self.op!r}")
            if self.checkpoint.index != "local" or self.copies is None:
                raise ValueError(
                    "GraphNode.checkpoint declares `index: local` on a node "
                    "with copies (`repeat:` / `chain:`): where the checkpoint "
                    "indexes them")
        return self

    @field_validator("inputs", mode="before")
    @classmethod
    def normalize_inputs(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(item) for item in v]

    @property
    def copies(self) -> Optional[str]:
        """How many copies of this node there are, whichever keyword says so
        — the expression as written, None when the node runs once.

        The count is the same question for both arrangements (it multiplies
        this node's params and FLOPs and everything under it), so every
        consumer of the count asks here and no consumer re-reads `repeat:`
        in a place a chain would then be missed."""
        return self.chain if self.chain is not None else self.repeat

    @property
    def source_names(self) -> List[str]:
        """The node's input refs as a flat list (a copy of `inputs`)."""
        return list(self.inputs)



def parse_input_ref(
    ref: str,
    graph: Dict[str, "GraphNode"],
    custom_ops: Optional[Dict[str, "CustomOpSpec"]] = None,
    components: Optional[Dict[str, "BlockGraph"]] = None,
) -> Tuple[str, str]:
    """Parse one entry of a node's `inputs:` list into `(node, port)`.

    Single-output ops are referenced as `<node>` and resolve to
    `(<node>, "out")` — the canonical main-output port name.

    Multi-output ops (registered primitives with `secondary_outputs`,
    OR spec-scoped `CustomOpSpec` entries with `secondary_outputs`)
    can be referenced as `<node>.<port>` to pick a secondary output.
    Disambiguation rule: the dotted form is treated as a port reference
    ONLY when `<node>` exists in the same graph AND its op declares
    `<port>` in `secondary_outputs`. Anything else falls back to the
    single-ref interpretation, leaving the dotted string intact as the
    node name.

    `ref` is the raw input string from yaml; `graph` is the surrounding
    `BlockGraph.graph` dict; `custom_ops` is the enclosing ArchSpec's
    `custom_ops` map (or None when calling from a context that doesn't
    have it — e.g. component-level validation, where custom ops are
    spec-scoped and not yet bound). Returns `(node_name, port)`.
    """
    from backend.arch.ops import OP_REGISTRY

    if "." not in ref:
        return (ref, "out")
    head, _, tail = ref.rpartition(".")
    parent = graph.get(head)
    if parent is None:
        return (ref, "out")
    # Check registered primitives first.
    op = OP_REGISTRY.get(parent.op)
    if op is not None and tail in op.secondary_outputs:
        return (head, tail)
    # Then spec-scoped custom_ops (multi-output via CustomOpSpec).
    if custom_ops is not None:
        cop = custom_ops.get(parent.op)
        if cop is not None and tail in cop.secondary_outputs:
            return (head, tail)
    # Then library/inline compounds (multi-output via BlockGraph).
    if components is not None:
        comp = components.get(parent.op)
        if comp is not None and tail in (comp.secondary_outputs or {}):
            return (head, tail)
    return (ref, "out")



def resolve_block_input_ports(
    graph: "BlockGraph",
    custom_ops: Optional[Dict[str, "CustomOpSpec"]] = None,
    components: Optional[Dict[str, "BlockGraph"]] = None,
) -> None:
    """Walk every node in `graph` and populate `node.resolved_inputs`
    via `parse_input_ref`. One-shot precomputation done at
    SpecLoader assembly time, replacing the engine's per-eval
    `parse_input_ref` call. Mutates each `GraphNode` in place.

    Idempotent: running twice with the same context yields the same
    result, so a library component populated at component-load time
    can be re-resolved at model-load time when richer custom_ops /
    components context becomes available.

    Two checks need the same context and run here: a ref that is both a
    sibling's name and a port of another sibling (`split.post` beside a
    `split` whose op outputs `post`) is refused as ambiguous, and the graph
    must have a dependency order (no cycle) under the edges resolved."""
    for name, node in graph.graph.items():
        node.resolved_inputs = tuple(
            parse_input_ref(inp, graph.graph, custom_ops, components)
            for inp in node.inputs
        )
        for ref, (parent, port) in zip(node.inputs, node.resolved_inputs,
                                       strict=True):
            if port != "out" and ref in graph.graph:
                raise ValueError(
                    f"node {name!r} reads {ref!r}, which names both the node "
                    f"{ref!r} and output {port!r} of node {parent!r}. Rename "
                    f"one of them so the edge has one reading.")
    graph.dependency_order()



class SparseDcpSpec(BaseModel):
    """Explicit SGLang sparse MLA runtime graph, independent of model names."""
    backend: Literal["sglang_fi_a2a"] = "sglang_fi_a2a"
    query_projection: str
    absorbed_query: str
    stored_kv_projection: str
    value_projection: str
    score: str
    softmax: str
    apply: str
    indexer_logits: str
    indexer_topk: str

    @property
    def framework(self) -> Literal["sglang"]:
        """The serving framework that owns this backend contract."""
        return "sglang"


class BlockGraph(BaseModel):
    """Computation DAG for a single transformer block.

    A block is also a parameterized compound op when it declares `params:`.
    Inside the graph, nodes reference declared params via `op: <param_name>`;
    callers bind concrete component names through `GraphNode.params` or
    `BlockAssignment.params` / `ModelLayer.params`.
    """
    # The form this component takes at decode, when that is a DIFFERENT graph
    # rather than the same graph at a different sequence length.
    #
    # An MLA attention reads its KV decompressed while prefilling and in latent
    # space while decoding — a genuinely different subgraph, shipped as its own
    # component (`mla_attention_absorbed`). Nothing else in the spec implies the
    # relationship: the twin's NAME ending in `_absorbed` is a convention, not a
    # declaration, and inferring it from the name is how a hardcoded
    # `{"mla_attention": "mla_attention_absorbed", ...}` map ended up in the
    # phase resolver, invisible to the spec and needing an edit per new family.
    #
    # Omitted by every component whose decode form is its prefill form, which is
    # almost all of them.
    decode_variant: Optional[str] = None
    sparse_dcp: Optional[SparseDcpSpec] = None
    # The phases that run `decode_variant` in place of this graph: decode
    # alone unless the component says otherwise. A sparse MLA attention
    # (DSA) also prefills absorbed — its query reads its selected rows of
    # the cached latent, so serving runs the sparse MLA kernel decode uses
    # (vLLM's sparse MLA layer hands every prefill token to `forward_mqa`);
    # a dense MLA prefills expanded. Which form a phase runs is the
    # component's fact, stated here, not read off the ops its graph holds.
    variant_phases: Tuple[Literal["prefill", "decode"], ...] = ("decode",)
    # Input port names (formals). Nearly every block takes one hidden input,
    # so `["x"]` is the default and authoring it is rejected as a repeat; a
    # block that takes more (hc_pre, dsv4_indexer) names all of its ports
    # here. `x` is only the local name of a one-formal component, not a
    # stream. A caller binds these by position: a node's `in:`, a
    # model-level `inputs:`, and — for a stack that carries a stream — the
    # stream's current value on formal 0.
    inputs: List[str] = ["x"]
    graph: Dict[str, GraphNode]                  # name → node
    outputs: List[str]                           # output node names

    def routes_tokens(
        self, resolve: Callable[[str], Optional["BlockGraph"]],
    ) -> bool:
        """Whether this block routes tokens to experts: its graph, or a
        component it calls, holds the dispatch op (`DISPATCH_OP`) — the
        declaration the tree walk marks a dispatch zone from
        (`SpecIRNode.in_dispatch_zone`). `resolve` is the namespace lookup,
        as for `alias_target`."""
        seen: set = set()
        pending: List["BlockGraph"] = [self]
        while pending:
            graph = pending.pop()
            for node in graph.graph.values():
                if node.op == DISPATCH_OP:
                    return True
                callee = resolve(node.op)
                if callee is not None and id(callee) not in seen:
                    seen.add(id(callee))
                    pending.append(callee)
        return False

    def alias_target(
        self, resolve: Callable[[str], Optional["BlockGraph"]],
    ) -> Optional["BlockGraph"]:
        """The component this block merely renames: a one-node graph whose
        op resolves to another BlockGraph (`chatglm_attention` →
        `gqa_fused_qkv_attention`). `resolve` is the namespace lookup — a
        graph does not know the namespace it was parsed into — answering
        None for a primitive op. Such a wrapper inherits cache / kv_label /
        shard_symbols from its target at lookup time, so a check about those
        fields belongs to the target and is not repeated on the wrapper.
        Distinct from `adapted_wrapper`, which asks whether a block is a bare
        `pre_norm` / `post_norm` binding."""
        if len(self.graph) != 1:
            return None
        (node,) = self.graph.values()
        return resolve(node.op)

    def adapted_wrapper(
        self, resolve: Callable[[str], Optional["BlockGraph"]],
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        """`(wrapper_op, params)` if this block is a pure adapter, else None.

        A pure adapter is a one-node graph whose node calls a component
        `roles.yaml` classifies as a wrapper (`role: block` — `pre_norm`,
        `post_norm`, any other layer topology) and binds that component's
        `kind: component` params (attn / ffn), possibly with a rename. It
        contributes no new structure — its content is fully captured by the
        bound components. `resolve` is the namespace lookup `alias_target`
        takes. Read at enumerate for the block's short name and wrapper
        (`BlockMeta.short_name` / `wrapper_op`)."""
        if not self.graph:
            return None
        target = self.alias_target(resolve)
        if target is None or roles.kind_of(target.role) != "wrapper":
            return None
        (node,) = self.graph.values()
        params = node.params or {}
        if not any(name in params for name, slot in (target.params or {}).items()
                   if slot.kind == "component"):
            return None
        return node.op, dict(params)

    def renamed_component(
        self, resolve: Callable[[str], Optional["BlockGraph"]],
    ) -> Optional[str]:
        """The component this block is nothing but a new name for: an
        `alias_target` whose one node binds no params and no dims, so the
        target computes exactly what this block does (`chatglm_attention` →
        `gqa_fused_qkv_attention`). A one-node block that binds something
        (`chatglm_block`: `pre_norm` with `attn` / `ffn`) is not a rename —
        the target alone is not this block. None otherwise."""
        if self.alias_target(resolve) is None:
            return None
        (node,) = self.graph.values()
        if node.params or node.dims:
            return None
        return node.op

    # Multi-output ports for the compound. Maps a wrapper-level port name
    # to an internal child reference (`"<child>"` or `"<child>.<port>"`).
    # Lets a library compound expose more than one output tensor — the
    # main output stays at `outputs[0]`, named ports expose secondary
    # tensors that callers reach via `<wrapper>.<port>` in their `in:` list.
    #
    # Use case: V4-Pro `Block.hc_pre` returns `(reduced, post, comb)`
    # from a single source-level method call; `hc_pre.yaml` mirrors this
    # by declaring `secondary_outputs: { post: split.post, comb: split.comb }`
    # so call sites can read the three outputs the same way the source
    # does (single function call, three return values).
    #
    # Without this, multi-output compounds would force the spec author
    # to hoist the inner multi-output primitive (here: `hc_split_sinkhorn`)
    # back out to the call site, breaking source-parity encapsulation.
    secondary_outputs: Dict[str, str] = {}
    # Param slots accepted by this block when invoked as a compound op.
    # Maps param_name → ParamSpec (kind=component|dim, optional default).
    params: Dict[str, ParamSpec] = {}
    # Set by the loader for synthesized blocks: the underlying compound-op
    # template and the call-site bindings used to instantiate it. These
    # are bookkeeping for the UI label — the graph itself is already
    # fully expanded with no `<param>` placeholders left. Derived, not
    # authored: the loader knows which template it expanded, so a YAML
    # saying so could only agree or lie.
    synth_template: Optional[str] = Field(
        default=None, json_schema_extra=LOADER_DERIVED)
    synth_params: Dict[str, str] = Field(
        default={}, json_schema_extra=LOADER_DERIVED)
    # Slot type this component fills. Answers "what kind of plug-in slot
    # can a model spec wire me into?", NOT "what kind of compute do I
    # do?" — those are different concepts. A KV-compressor sub-module
    # does attention compute, but it's not a complete attention
    # implementation a model can drop in as `attn:`; it has `role: None`
    # because it's a sub-component used inside another component.
    #
    # Allowed values:
    #   attention  — complete attention implementation (Q/K/V projections
    #                + score + output) a model can plug into `attn:`.
    #                Required: `cache:`. `attn_variant` only when the
    #                compute flavor's name differs from the cache family.
    #   ffn_dense  — complete dense FFN (gate/up/down + activation).
    #   ffn_moe    — complete MoE FFN (router + dispatch + experts + combine
    #                + optional shared experts).
    #   router     — MoE routing pipeline (gate / topk / scoring); used
    #                INSIDE an ffn_moe component, but rendered separately
    #                in the slot view.
    #   block      — full transformer block (pre_norm / post_norm /
    #                gemma4_4norm_scaled_block / dsv4_block / …).
    #   aux_block  — auxiliary stack wrapper (MTP head, vision tower wrapper).
    #   None       — sub-component / utility (compressor, indexer,
    #                head_split, head_merge, head_broadcast). Does
    #                attention/FFN compute internally but isn't itself
    #                pluggable as a top-level slot. Leaves under it
    #                inherit role from their nearest non-None ancestor
    #                via spec_tree role propagation.
    #
    # Quant routing (`quant_context.per_role.<role>`) and FLOPs/params
    # attribution use the LEAF role, which is computed via inheritance
    # from this field; see `spec_tree.build_ir_node`.
    role: Optional[str] = None

    @field_validator("role")
    @classmethod
    def _role_is_registered(cls, v: Optional[str]) -> Optional[str]:
        # The vocabulary is `backend/arch/roles.yaml`; a Literal here was a
        # second copy of it, and the two copies had drifted.
        return roles.require_registered(v)
    # One-line description used in the spec-gen prompt manifest. Agents
    # see name+role+description for every component without loading the
    # full graph YAML, then Read the file when they need details. Should
    # capture the distinguishing feature (variant attention scheme,
    # tensor-naming quirk, which models use it) in ~150 chars.
    description: Optional[str] = None

    # Inter-layer STREAMS attached at inner nodes — values that flow forward
    # through the layer stack beside the one the stack carries across its
    # block boundary. The carried stream (`carries: true`, `hidden` in the
    # corpus) reaches a component through its formal 0 and leaves as its
    # output, so a component never names it; `reads` / `stream_out` naming
    # it is refused at load. The streams listed here are the inner ones
    # (auxiliary signals threaded across layers, shared top-k indices,
    # YOCO/CLA shared KV, accumulator sums), whose producer and consumer may
    # live in DIFFERENT variant components selected per-layer by a
    # config_list. A component
    # declares only its INTERFACE to a stream, never the wiring:
    #   • `reads`      — stream names this component consumes; each becomes a
    #                    valid `in:` reference bound to the producer tensor,
    #                    exactly like a named block input.
    #   • `stream_out` — {stream: node} this component PRODUCES; the node's
    #                    value is what this layer writes to the stream.
    # Which layer's write a reader reads is the reader's declaration
    # (`reads_from` on the assignment that selects it), not the component's.
    # The stack (the repeated model group) lists each stream name via
    # `ModelLayer.streams`. Optional `shape` is a publish-gate against the
    # producer node (per-variant dims from `per_layer.assign[].dims`); eval
    # does not seed the edge from it. The cross-layer stream check verifies
    # the `src` anchor occurs in the fetched forward source — a stream is a
    # real topology claim, same truthfulness treatment as an op's
    # source_anchor. Full/shared DSA, YOCO/CLA shared-KV, etc. all
    # express as: producer variant `stream_out`s it, consumer variant `reads`
    # it. Params stay local. If the `stream_out` node is also a cache entry,
    # readers adopt that entry's KV length as `s_kv`; they do not declare
    # the cache. Other streams stay topology (drawable edge).
    reads: List[str] = []
    stream_out: Dict[str, str] = {}

    # What this component keeps between decode steps — see CacheSpec. When
    # set, the inference engine treats it as the source of truth for "which
    # per-token state gets appended to the cache", and the legacy heuristic
    # that infers MLA/GQA/MHA from config dims is bypassed. Examples:
    #     gqa_attention:  {family: GQA, entries: [{node: k_rope, kind: kv},
    #                                             {node: v_split_heads, kind: kv}]}
    #     mla_attention:  {family: MLA, entries: [{node: kv_a_layernorm, kind: latent},
    #                                             {node: k_pe_rope, kind: rope_key}]}
    # Non-attention components (FFN, norm, router, …) leave it unset.
    cache: Optional[CacheSpec] = None
    # Compute-flavor variant within the same `cache.family`. Two attention
    # components can share a KV cache topology (same family) but differ in
    # the *compute graph upstream of the cache* — DSv4-Pro's HCA / CSA /
    # SWA-only all reduce to MLA-style latent KV (family="MLA") but differ
    # in whether they wrap the latent with a non-overlap compressor (HCA),
    # an overlap compressor + sparse indexer (CSA), or nothing (SWA-only).
    # The inference panel only cares about KV bytes (the family is enough),
    # but the architecture view needs this to show the per-layer mix.
    # Set it ONLY when the flavor differs from the family: repeating the
    # family here says nothing, and `attn_display` already falls back to it.
    attn_variant: Optional[str] = None
    # A-class golden anchors — KV facts derived from the modeling source,
    # attached from `claims/components/<identity>.yaml` and judged by
    # validate/_axioms.py at every call in the model's analysis. See AxiomSet.
    # Optional; absent = component not yet anchored.
    axioms: Optional[AxiomSet] = Field(default=None, json_schema_extra=LOADER_DERIVED)

    # ---- Cache views -----------------------------------------------------
    @property
    def open_slots(self) -> List[str]:
        """The component slots only a caller can fill: declared `params:` of
        kind component with no default. A call of another component binds
        none of them."""
        return sorted(name for name, slot in (self.params or {}).items()
                      if slot.kind == "component" and slot.default is None)

    # `cache:` is the declaration; these are the four questions the rest of
    # the codebase asks of it. They are properties, not fields, so there is
    # no second place a cache fact can be written and disagree.

    @property
    def kv_label(self) -> Optional[str]:
        """KV-cache classification tag ("MLA" / "GQA" / "SWA" / …), for
        storage accounting. For the label shown to users, read
        `attn_display` — a DSA model's cache IS MLA-shaped."""
        return self.cache.family if self.cache is not None else None

    @property
    def kv_cached(self) -> List[str]:
        """Graph nodes appended to the cache once per token. Each node's
        resolved `shape_out` (at B=s_ctx=1) is its per-token contribution;
        per-layer bytes = Σ(elements × bpe)."""
        if self.cache is None:
            return []
        return [e.node for e in self.cache.entries if e.node is not None]

    @property
    def kv_states(self) -> List[CacheEntry]:
        """The recurrent states this component keeps, in declaration order —
        linear attention / SSM hold FIXED tensors per request instead of
        appending per token (GatedDeltaNet `[heads, d_v, d_k]` beside its
        causal-conv window; Mamba2 `[heads, d_head, d_state]` beside its
        own). Each entry names the graph node carrying it (`carrier`) and its
        axes (`shape`, resolved in that node's frame). Billed once per
        request with a storage cardinality independent of S: flat
        on the per-token growth chart, real HBM in the KV budget /
        max-batch. A component with any state is a supported KV class even
        with no per-token slots; its `kv_growth` axiom must be
        `constant_state`."""
        if self.cache is None:
            return []
        return [e for e in self.cache.entries if e.state is not None]

    @property
    def attn_display(self) -> Optional[str]:
        """Canonical attention label for DISPLAY — the compute-flavor
        `attn_variant` (e.g. DSA) if set, else the `cache.family` KV-cache
        topology (e.g. MLA). THE single source of truth for the attention
        type shown to users (architecture view, dashboard, badges, block
        labels): every display site must read this, never the family alone.
        The family by itself is reserved for KV-cache shape / cost
        accounting (a DSA model's cache IS MLA-shaped). None only when
        neither is set."""
        return self.attn_variant or self.kv_label

    @model_validator(mode="after")
    def validate_variant_phases(self) -> "BlockGraph":
        """`variant_phases` says which phases run `decode_variant`, so it
        is stated beside one, and decode is among them."""
        if self.variant_phases != ("decode",):
            if not self.decode_variant:
                raise ValueError(
                    "variant_phases names the phases that run "
                    "`decode_variant`, and this component declares none")
            if "decode" not in self.variant_phases:
                raise ValueError(
                    f"variant_phases {list(self.variant_phases)} leaves "
                    f"decode out; `decode_variant` is the decode form")
        return self

    @model_validator(mode="after")
    def validate_cache_declaration(self) -> "BlockGraph":
        """A cache slot names a node in THIS graph, and `attn_variant` says
        something the family does not already say."""
        if self.cache is not None:
            for slot in self.kv_cached:
                # A slot may name a node directly, or reach into a
                # sub-component's port (`indexer.k_norm` where `indexer` is
                # the node) — some graphs also spell the port itself as a
                # dotted node name, so both spellings resolve here.
                if (slot not in self.graph
                        and slot.partition(".")[0] not in self.graph):
                    raise ValueError(
                        f"cache slot {slot!r} is not a node in this "
                        f"component's graph: {sorted(self.graph)}"
                    )
            self._check_entry_names_unique()
            self._check_state_carriers()
        if self.attn_variant is not None and self.attn_variant == self.kv_label:
            raise ValueError(
                f"attn_variant {self.attn_variant!r} repeats cache.family. "
                f"`attn_display` already falls back to the family, so this "
                f"line changes nothing and drifts the moment one is edited. "
                f"Delete it, or set it to the flavor that actually differs."
            )
        return self

    def _check_entry_names_unique(self) -> None:
        """Every cache entry — node slot or state — is named once.

        The name is the key a slot's dtype override, its kind and its cached
        tensor are looked up under (`eval.cache`), so a state `conv`
        declared `dtype: fp32` beside a node slot `conv` would hand the fp32
        to the node slot too, silently."""
        seen: Dict[str, str] = {}
        for entry in self.cache.entries:
            name = entry.name
            kind = "node slot" if entry.node is not None else "state"
            if name in seen:
                raise ValueError(
                    f"cache entry name {name!r} is used twice (a {seen[name]} "
                    f"and a {kind}). A slot's dtype and its cached tensor are "
                    f"looked up by this name, so each entry needs its own: "
                    f"rename the state (e.g. `{name}_state`), or drop the "
                    f"repeated node slot."
                )
            seen[name] = kind

    def _check_state_carriers(self) -> None:
        """Each recurrent state names a node of this graph as its carrier,
        and no node carries two of them.

        The carrier is where the state's per-step read and write are billed
        and drawn, so it has to be a node the walk reaches; two states on one
        carrier would put two tensors behind one kernel with nothing saying
        which read is which."""
        seen_carriers: Dict[str, str] = {}
        for entry in self.kv_states:
            if entry.carrier not in self.graph:
                raise ValueError(
                    f"recurrent state {entry.state!r} names carrier "
                    f"{entry.carrier!r}, which is not a node in this "
                    f"component's graph: {sorted(self.graph)}. Name the node "
                    f"whose kernel reads and rewrites the state each step."
                )
            if entry.carrier in seen_carriers:
                raise ValueError(
                    f"node {entry.carrier!r} is named as the carrier of both "
                    f"{seen_carriers[entry.carrier]!r} and {entry.state!r}. A "
                    f"carrier streams one state; point each state at the "
                    f"kernel that owns it (the scan for the recurrent matrix, "
                    f"each causal conv1d for its own window)."
                )
            seen_carriers[entry.carrier] = entry.state

    def producers_of(self, name: str) -> List[str]:
        """The sibling nodes `name` reads, in `in:` order.

        Each ref is read by `parse_input_ref`, the one grammar for an
        `in:` entry: `split.post` is port `post` of sibling `split` when
        `split`'s op declares that output, and otherwise the node named
        `split.post`. The loader resolves every graph once with the spec's
        custom ops and components in view (`resolve_block_input_ports`)
        and those answers are read here; a graph not yet resolved is read
        with the registry alone; there a port of a sibling whose op the
        registry does not know (a component or custom op) is still read
        from that sibling. Anything that is not a sibling arrives from
        outside and must be declared so: a formal (`inputs:`) or a stream
        (`reads:`). A ref that is neither is refused."""
        from backend.arch.ops import OP_REGISTRY
        node = self.graph[name]
        resolved = len(node.resolved_inputs) == len(node.inputs)
        edges = (node.resolved_inputs if resolved
                 else tuple(parse_input_ref(ref, self.graph)
                            for ref in node.inputs))
        outside = set(self.inputs) | set(self.reads)
        out: List[str] = []
        for ref, (parent, _port) in zip(node.inputs, edges, strict=True):
            if parent not in self.graph and not resolved:
                head = ref.rpartition(".")[0]
                if head in self.graph and self.graph[head].op not in OP_REGISTRY:
                    parent = head
            if parent in self.graph:
                if parent not in out:
                    out.append(parent)
            elif parent not in outside:
                raise ValueError(
                    f"node {name!r} reads {ref!r}, which is no node of this "
                    f"graph, no formal (`inputs: {self.inputs}`) and no "
                    f"stream (`reads: {self.reads}`). Name the producing "
                    f"node, or declare the tensor in `inputs:` or `reads:`.")
        return out

    def dependency_order(self) -> List[str]:
        """The graph's node names with every producer before its readers.

        `graph:` is a mapping, and a mapping's written order is not a fact
        about the model: a spec re-dumped with sorted keys is the same
        spec. Among nodes whose producers are all placed, written order
        decides, so a graph already written producer-first comes back
        unchanged. A cycle has no such order and is refused, naming the
        nodes on it (not the readers it merely blocks)."""
        producers = {name: self.producers_of(name) for name in self.graph}
        placed: set = set()
        order: List[str] = []
        pending = list(self.graph)
        while pending:
            ready = next((name for name in pending
                          if all(p in placed for p in producers[name])), None)
            if ready is None:
                cycle = self._on_a_cycle(pending, producers)
                raise ValueError(
                    f"graph nodes {cycle!r} read each other in a cycle ("
                    + "; ".join(
                        f"{name} reads "
                        f"{[p for p in producers[name] if p in cycle]}"
                        for name in cycle)
                    + "). A forward pass is a DAG: every `in:` must name a "
                    "node that can be computed first.")
            pending.remove(ready)
            placed.add(ready)
            order.append(ready)
        return order

    @staticmethod
    def _on_a_cycle(pending: List[str],
                    producers: Dict[str, List[str]]) -> List[str]:
        """The nodes of `pending` that reach themselves through what they
        read — the cycles, without the nodes downstream of one."""
        def reaches(start: str) -> set:
            seen: set = set()
            stack = [p for p in producers[start] if p in pending]
            while stack:
                n = stack.pop()
                if n not in seen:
                    seen.add(n)
                    stack.extend(p for p in producers[n] if p in pending)
            return seen
        return [name for name in pending if name in reaches(name)]

    def derived_formula(self) -> str:
        """The compound's output expression, composed from the graph itself.

        This is the only formula a compound has — anything hand-written
        would be an abstraction that drifts from the actual graph; this
        one cannot — each node's op is substituted into its consumers, so
        the result IS the graph restated as math. A node read by more than
        one consumer, or through a secondary port, is named once and
        referenced; everything else inlines.
        """
        from collections import Counter
        uses: Counter = Counter()
        for node in self.graph.values():
            for ref in node.source_names:
                uses[parse_input_ref(ref, self.graph)[0]] += 1
        for target in self.secondary_outputs.values():
            uses[parse_input_ref(target, self.graph)[0]] += 1
        port_read = set()
        for node in self.graph.values():
            for ref in node.source_names:
                pname, port = parse_input_ref(ref, self.graph)
                if port:
                    port_read.add(pname)
        for target in self.secondary_outputs.values():
            pname, _port = parse_input_ref(target, self.graph)
            port_read.add(pname)

        statements: List[str] = []
        memo: Dict[str, str] = {}

        def _expr(name: str) -> str:
            if name not in self.graph:
                return name                       # a block input — a leaf name
            if name in memo:
                return memo[name]
            node = self.graph[name]
            args = []
            for ref in node.source_names:
                pname, port = parse_input_ref(ref, self.graph)
                sub = _expr(pname) if pname in self.graph else pname
                if port and port != "out":
                    sub = f"{sub}.{port}"
                args.append(sub)
            body = f"{node.op}({', '.join(args)})"
            if uses[name] > 1 or name in port_read:
                statements.append(f"{name} = {body}")
                memo[name] = name
                return name
            return body

        outs = [(_expr(parse_input_ref(o, self.graph)[0]))
                for o in self.outputs]
        for target in self.secondary_outputs.values():
            pname, port = parse_input_ref(target, self.graph)
            sub = _expr(pname) if pname in self.graph else pname
            outs.append(f"{sub}.{port}" if port and port != "out" else sub)
        lhs = ", ".join(list(self.outputs) + list(self.secondary_outputs))
        rhs = ", ".join(outs)
        statements.append(f"({lhs}) = ({rhs})" if "," in lhs
                          else f"{lhs} = {rhs}")
        return "\n".join(statements)

    @model_validator(mode="after")
    def validate_graph_connectivity(self) -> "BlockGraph":
        """Ensure all node inputs reference existing nodes or block inputs."""
        self._check_node_contracts()
        self._check_outputs_exist()
        self._check_edge_kinds()
        self._check_node_weight_guards()
        self._check_stream_targets()
        return self


    def _check_node_contracts(self) -> None:
        """Per node: zero-input ops take no `in:`, every input ref resolves,
        primitive arity meets `shape_in`, and call-site-local required dims
        are bound."""
        valid_names = (set(self.inputs) | set(self.graph.keys())
                       | set(self.reads))

        # Ops with no upstream tensor — purely stored learnable weights.
        # Declaring `in:` on these is meaningless (the lambda doesn't read
        # it, and the rendered edge would imply a dataflow that doesn't
        # exist). Reject explicitly so silent typos / copy-paste mistakes
        # surface at load time instead of as misleading SVG edges.
        from backend.arch.ops import OP_REGISTRY
        for name, node in self.graph.items():
            spec = OP_REGISTRY.get(node.op)
            dims_error = call_site_dims_error(node.op, node.dims or {})
            if dims_error is not None:
                raise ValueError(f"Node {name!r}: {dims_error}")
            if spec is not None and spec.arity() == 0 and node.inputs:
                raise ValueError(
                    f"Node {name!r} uses zero-input op {node.op!r} but "
                    f"declares `in: {node.inputs!r}`. "
                    f"`{node.op}` is a standalone learnable weight bank "
                    f"with no upstream tensor — drop the `in:` field."
                )
            if spec is not None and spec.axis_required and node.axis is None:
                raise ValueError(
                    f"Node {name!r} uses {node.op!r} without `axis:`. "
                    f"`{node.op}` has no default axis — declare which axis it "
                    f"acts on: `axis: -1` for the last (a head split "
                    f"`[B, S, n_heads*d_head] -> [B, S, n_heads, d_head]`), "
                    f"`axis: 1` for the sequence of a `[B, S, D]` tensor."
                )
            for src in node.source_names:
                # `src` is the raw string from yaml. parse_input_ref strips
                # any trailing `.<port>` suffix when it resolves against
                # a registered multi-output primitive; for single-output
                # references, `parsed[0]` is just the bare ref. Custom-op
                # ports aren't visible here (custom_ops live on ArchSpec,
                # not BlockGraph) — defer port validity to ArchSpec-level
                # validation + engine when the head's op is unknown to
                # OP_REGISTRY (i.e., a component invocation or custom_op).
                # Primitive-port typos (e.g. `splitter.podt`) are still
                # caught: parse_input_ref already disambiguates against
                # OP_REGISTRY and returns a bogus parsed_node when the
                # port doesn't exist on a known primitive.
                from backend.arch.ops import OP_REGISTRY
                parsed_node, _port = parse_input_ref(src, self.graph)
                if parsed_node not in valid_names:
                    if "." in src:
                        head = src.partition(".")[0]
                        head_node = self.graph.get(head)
                        head_op = head_node.op if head_node else None
                        if (head in valid_names
                                and head_op is not None
                                and head_op not in OP_REGISTRY):
                            # Custom-op or component port reference —
                            # defer to ArchSpec-level validation.
                            continue
                    raise ValueError(
                        f"Node {name!r} references unknown input {src!r}. "
                        f"Available: {sorted(valid_names)}"
                    )

            # Required-dims check (call-site-local only): each op
            # declares the dim keys its flops_fn / params_fn read.
            # Spec-global dims (d_model, n_vocab, n_layers, n_heads,
            # d_head, etc.) are bound
            # at the model level and inherited automatically — we
            # can't verify them here without the resolved spec
            # context. So we only enforce the conventional call-
            # site-local keys: `in`, `out`, `dim`. Missing globals
            # surface as later resolution errors with full context.
            #
            # This catches the common typo cases (e.g. forgetting
            # `dims: { in: n_hidden_copies, out: d_hc_mix }` on a
            # `linear`/`linear_param`, or forgetting
            # `dims: { in: n_hidden_copies }` on a `sum_reduce`).
            #
            # Custom-op required_dims are checked at ArchSpec level
            # (`validate_block_definitions`), where the surrounding
            # `custom_ops:` table is bound. BlockGraph runs in
            # isolation and only sees registered primitives.
            from backend.arch.ops import OP_REGISTRY
            _LOCAL_DIMS = {"in", "out", "dim"}
            op_spec = OP_REGISTRY.get(node.op)
            # Primitive-op under-arity check (call-site `in:` length
            # vs op `shape_in` declaration). Without this, an
            # underwired op like `attention_score` with 1 input
            # instead of 2 crashes later inside shape inference
            # with a cryptic `IndexError: list index out of range`.
            #
            # We enforce `actual >= declared` only — never strict
            # equality. Two classes of over-wired-but-legitimate exist
            # in the current op registry:
            #   • variadic shape semantics (add / mul / concat reduce
            #     element-wise / along an axis across all inputs)
            #   • control-flow inputs whose shape isn't read by
            #     the DimMap (e.g. `dispatch` takes `[x, gate]`; the
            #     gate carries the routing decision so the topo sort
            #     places dispatch after routing, but inference only
            #     reads `shape_in[0]`).
            # Catching those requires per-op review; that's separate
            # from this guard, which only protects against the
            # IndexError-in-inference failure mode.
            #
            # Empty `shape_in=()` is the codebase's "untyped, rank
            # flows through" convention for view ops (view_split,
            # view_merge, transpose, broadcast). Genuine zero-input
            # primitives (parameter_vector) are already gated by
            # `_ZERO_INPUT_OPS` above. Skip on empty-template ops.
            if op_spec is not None and op_spec.shape_in:
                expected = op_spec.arity()
                actual = len(node.inputs)
                if actual < expected:
                    raise ValueError(
                        f"Node {name!r} calls op {node.op!r} with "
                        f"{actual} input(s), but it requires at "
                        f"least {expected} (shape_in={op_spec.shape_in!r}). "
                        f"Add the missing operand(s) to the node's "
                        f"`in:` list — under-wiring would crash deep "
                        f"inside shape inference."
                    )
            required_dims = (op_spec.required_dims
                             if op_spec is not None else ())
            if required_dims:
                local_required = [k for k in required_dims
                                  if k in _LOCAL_DIMS]
                if local_required:
                    provided = set(node.dims.keys())
                    if node.dim is not None:
                        provided.add("dim")
                    missing = [k for k in local_required
                               if k not in provided]
                    if missing:
                        raise ValueError(
                            f"Node {name!r} (op={node.op!r}) is missing "
                            f"required dim binding(s): {missing}. "
                            f"Op declares "
                            f"required_dims={list(required_dims)} "
                            f"— bind via `dim:` (single key) or "
                            f"`dims: {{...}}` on the call site."
                        )

    def _check_outputs_exist(self) -> None:
        """A graph declares what it hands its caller, and every declared
        output names a graph node."""
        if not self.outputs:
            raise ValueError(
                "the graph declares no `outputs:`. Name the node whose value "
                "it hands to its caller: `outputs: [<node>]`")
        for out in self.outputs:
            if out not in self.graph:
                raise ValueError(f"Output {out!r} not found in graph nodes")

    def _check_edge_kinds(self) -> None:
        """Every edge's producer output kind equals the consumer's expectation
        at that input position."""
        # Edge-kind check (7.1): for every edge, the producer's declared
        # output kind must equal the consumer's expectation at that input
        # position. Shape says how big a tensor is; kind says what it is —
        # a top-k index list is int32[B,Sq,K]-shaped like an activation,
        # and only the kind stops it being wired where an activation
        # belongs. Component invocations are opaque here (their boundary
        # op is not a primitive); they are checked inside their own
        # BlockGraph when that component loads.
        from backend.arch.ops import OP_REGISTRY as _OPS
        for name, node in self.graph.items():
            consumer_spec = _OPS.get(node.op)
            if consumer_spec is None:
                continue
            for pos, src in enumerate(node.source_names):
                parsed_node, _port = parse_input_ref(src, self.graph)
                if parsed_node not in self.graph:
                    continue        # unknown refs are reported above
                producer_spec = _OPS.get(self.graph[parsed_node].op)
                if producer_spec is None:
                    continue
                p_kind = producer_spec.output_kind
                c_kind = consumer_spec.input_kind(pos)
                if p_kind != c_kind:
                    raise ValueError(
                        f"Edge {parsed_node!r} → {name!r}: kind mismatch "
                        f"({p_kind!r} feeds a slot expecting {c_kind!r}). "
                        f"A {p_kind} edge is not an activation — rewire, "
                        f"or declare the op's kinds if it genuinely "
                        f"consumes one."
                    )

    def _check_node_weight_guards(self) -> None:
        """Quant-suffix lockdown + weight-dim geometry per node. The
        lockdown first: a hand-written scale node is removed outright, which
        its divisor's geometry would only obscure."""
        for name, node in self.graph.items():
            # Quant-metadata-suffix lockdown.
            _check_quant_suffix_in_arch_spec(
                "block_graph", "<unnamed>", name, node.op,
                node.quant_metadata,
            )
            # Geometric weight-dim check (see _check_weight_dim_geometry).
            _check_weight_dim_geometry(
                "block_graph", "<unnamed>", name, node.op, node.dim,
            )

    def _check_stream_targets(self) -> None:
        """Every stream_out target names a graph node, and no stream this
        component reads shares a name with one of its formals."""
        # A formal and a stream of one name would make an `in:` of that name
        # point at two tensors: the caller's bound input and the producer's.
        clash = sorted(set(self.reads) & set(self.inputs))
        if clash:
            raise ValueError(
                f"component `reads:` {clash} also name formal input(s) "
                f"{self.inputs!r}; an `in:` of that name would read two "
                f"tensors. Rename the formal or the stream.")
        # A produced stream names the graph node whose value updates it, so
        # that node has to be in this graph. Catches a typo'd or renamed
        # producer, which would otherwise surface as a stream that silently
        # carries nothing.
        for sname, target_node in self.stream_out.items():
            if target_node not in self.graph:
                raise ValueError(
                    f"stream_out[{sname!r}] = {target_node!r} not found in graph"
                )
        # The reading half of the same interface: a read stream is an input
        # like a formal, so some node's `in:` names it. A `reads:` no node
        # consumes is a dataflow edge the graph claims and does not draw —
        # the reader looks independent of the layer that wrote it.
        consumed = {parse_input_ref(ref, self.graph)[0]
                    for node in self.graph.values() for ref in node.inputs}
        unread = [s for s in self.reads if s not in consumed]
        if unread:
            raise ValueError(
                f"component `reads:` {unread} but no node's `in:` names "
                f"{'it' if len(unread) == 1 else 'them'}; wire the stream "
                f"into the node that consumes it, or drop the read.")
