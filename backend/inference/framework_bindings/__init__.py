"""Framework-binding YAML registry.

Loads `backend/inference/framework_bindings/<framework>.yaml` into a
`framework -> FrameworkBinding` registry. A binding holds the framework's
fusion patterns — the fused-op concepts it runs as one launch, each with
the `source_anchor` that shows it and the `when:` it does it — and the
serving choices a measured kernel's key depends on that the framework, not
the model, makes: the paged-KV page (`kv_block_size`) and where the
framework reads the Mamba-2 scan chunk (`ssm_chunk_size: {from_config: [...]}`,
the checkpoint-config fields it consults, in order). Both are declared in
every binding file, never defaulted.
Which implementation a serving framework dispatches to is a binding fact
(`kernel_dispatches`: provider library plus the provider's COMPLETE resolved
implementation id). Which release measured that implementation, and what
latency it measured, remain the corpus row's identity.

There is one binding per framework. It describes the latest release its facts
were read at, named by `audited_at: {tag, commit}`; re-auditing at a newer
release moves the facts and the tag together, and nothing keeps the old one.

Cross-validation at load time: every `fused_op` referenced by a binding
must exist in `backend/inference/fusion_plans/`. Bindings pointing to
unknown fused_ops are rejected with an explicit error naming the file.

An entry whose kernel the framework runs only under conditions says so in
`when:`, a closed vocabulary checked at load (`DispatchCondition`). The
deployment answers six keys — a server flag, the activation dtype, the
device, the minimum compute capability, the paged-KV page, and the
vocabulary tensor-parallel degree — through
`FrameworkBinding.deployed`; the graph answers the rest
(`eval.fusion.GraphCondition`) during the rewrite. A key outside the
vocabulary is a load error, so a condition cannot sit in the file unread.

Which of the three decides a key — the card, the deployment or the walk —
is declared beside the vocabulary (`DispatchCondition.CARD_KEYS`,
`DEPLOYMENT_KEYS`, `GraphCondition`'s fields) and checked at load
(`_check_dispatch_vocabulary`). The card's share is read off the sheet row
by `card_dispatch_facts` and is what a card's walk class is made of
(`card_class`): two cards it answers alike are dispatched alike, so one
walk of a model serves both.

One level out, the sheet's own columns are classified the same way and for
the same reason (`_check_card_fact_classification`): every `gpu_specs.csv`
column says whether it changes a WALK, changes a PRICE, or is read by
neither, and a column classified nowhere fails the load. A column a walk
reads with nobody having said it is a walk fact leaves `card_class` too
coarse, and then every walk reused inside that class is one card's answer
published as another's.

This `when:` is not the spec's `when:`. A spec node's or tied weight's
`when:` (`arch.schema.graph.GraphNode.when`, `arch.schema.model.TiedWeightSpec`)
is an expression the model config decides (a dim symbol, or a config field
such as `config.tie_word_embeddings`); it says whether the model has a step or
a tie. A binding's `when:` is a mapping over the closed
keys above that the deployment and the graph decide; it says whether the
framework runs its kernel for that step.

A binding says nothing about communication: no pipeline wire, no payload a
cut hands on, no pipeline support. Framework source is a reference for an
implementation, a fusion pattern, a measured kernel's identity, and the
all-reduce dispatch below — nothing else — so a pipeline crossing is priced
the same under every framework, from the graph's dataflow.

`collectives:` is the framework's all-reduce dispatch inside one NVLink
domain: an ordered list of rungs, each an algorithm of
`comm_volume.ALGORITHMS` and the payload it takes (`max_bytes`, and/or
`max_bytes_times_world` over the group) and the smallest group it runs
in (`worlds`, the group sizes its kernel accepts), with a `when:` over the card's
compute capability where the framework's choice depends on it. The first
rung the card admits (its topology, `gpu_specs`) and the payload fits wins;
past them NCCL's schedule runs (`comm_cost.collective_ladder`). Only what
changes the estimate under common deployments is declared — one
representative threshold per algorithm, not the framework's buffer caps,
environment switches or CUDA-graph tables; the binding comments state what
is billed instead.
"""
from __future__ import annotations

import json
from backend.arch.yaml_load import load_yaml
import logging
import os
import re
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import (Any, ClassVar, Dict, FrozenSet, Mapping, NamedTuple,
                    Optional, Tuple, Union)

import yaml as _yaml

from backend.arch import roles as roles_registry
from backend.arch.activation_formats import ACTIVATION_FORMATS
from backend.arch.dims import PHASES, capabilities_of, world_by_label
from backend.arch.dtypes import UnknownDtype, canonical
from backend.arch.eval.fusion import CROSSES, GraphCondition
from backend.arch.eval.formats import LOADER_TRANSFORMS
from backend.arch.quant_schemes import QUANT_SCHEMES
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference import gpu_specs
from backend.inference.gpu_specs import get_gpu, peak_families
from backend.inference.op_signatures import op_bindings
from backend.arch.layout.comm_volume import ALGORITHMS, NCCL
from backend.inference.comm_cost import CollectiveRung


_logger = logging.getLogger(__name__)


def _pattern_specificity(op_name: str) -> int:
    """Larger = more specific = should match before more-generic
    schemes that could consume a substructure of it.

    The matcher iterates `op_implementations` in order and consumes
    matched nodes; if a generic scheme runs first it can preempt a
    larger pattern that contains it as a substructure (e.g. the
    parallel-form `merged_column_parallel_linear` matching the
    gate+up linears inside `gated_act_ffn` before the larger pattern
    gets a chance). Sorting by descending specificity at load time
    makes the right thing happen without any author discipline.

    Specificity scale:
      * Explicit-nodes form: number of required pattern nodes (each a
        required op slot; more nodes = a structurally larger commit). An
        `optional` node is not one: a plan that MAY also take the
        activation cast after its sink (`silu_and_mul`) is no larger than
        its plain pattern, and counting the cast let `fused_add_rmsnorm`
        preempt the `allreduce_fused_rmsnorm` that contains it.
      * Parallel form: 0 (matches N>=2 anonymous siblings of one op
        — the most generic shape we have).
      * Unknown / missing scheme: 0 (defensive — bindings can't
        reference unknown schemes anyway, the load-time check above
        catches that).
    """
    scheme = FUSION_SCHEMES.get(op_name)
    if scheme is None:
        return 0
    pattern = scheme.naive_pattern
    if pattern is None or pattern.is_parallel():
        return 0
    return sum(1 for node in pattern.expanded_nodes().values()
               if not node.optional)


@dataclass(frozen=True)
class ServingDeployment:
    """The deployment facts a binding's `when:` asks: the server flags the
    caller turned on or off, the activation dtype (a serving-dtype name,
    `BF16`), the GPU's device (`gpu_specs.csv` `device`, `cuda`), and the
    request-owned serving facts a primitive kernel dispatch can ask.

    A fact left None is unknown, and a condition on an unknown fact does
    not hold: a kernel is not claimed for a deployment nobody described.
    The activation dtype is the walk's model dtype
    (`ResolvedPrecision.model_dtype`, what the caller passes), never a
    per-role edge width: an `edge_dtypes` entry's own `when:` is answered
    against this deployment, so reading a resolved edge here would make a
    width depend on itself.
    """

    #: The fields `at` fills from a GPU row. They are `CardClass`'s two
    #: halves by construction, and `_check_card_fact_classification` refuses
    #: drift between the two spellings.
    CARD_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
        "device", "compute_capability", "peak_families",
    })

    server_flags: Tuple[Tuple[str, ServerOption], ...] = ()
    activation_dtype: Optional[str] = None
    device: Optional[str] = None
    compute_capability: Optional[str] = None
    #: The paged-KV page this request deploys with. A framework binding's
    #: own default is resolved by the caller that owns the cache-manager
    #: fact; a primitive dispatch condition must not guess it.
    kv_block_size: Optional[int] = None
    #: The vocabulary-parallel tensor-parallel degree (the attention TP
    #: world), or None when the caller did not state a parallel deployment.
    tensor_parallel: Optional[int] = None
    #: The MMA families the described card's SKU row lists peaks for
    #: (`gpu_specs.peak_families`). This is the capability half of a
    #: precision-launch choice: a default candidate needs every declared
    #: family, while an explicitly declared promotion ladder needs one of its
    #: rungs. Empty means either no card was described or the described card
    #: lists no MMA peak; both leave a capability-gated candidate unselected
    #: rather than guessed.
    peak_families: Tuple[str, ...] = ()

    @classmethod
    def of(cls, *, server_flags: Optional[Mapping[str, ServerOption]] = None,
           activation_dtype: Optional[str] = None,
           device: Optional[str] = None,
           compute_capability: Optional[str] = None,
           kv_block_size: Optional[int] = None,
           tensor_parallel: Optional[int] = None,
           peak_families: Tuple[str, ...] = ()) -> "ServingDeployment":
        flags = []
        for name, value in (server_flags or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                flags.append((str(name), int(value)))
            elif isinstance(value, (bool, str)):
                flags.append((str(name), value))
            else:
                raise ValueError(
                    f"server option {name!r} is a boolean, string, or "
                    f"integer, not {value!r}")
        return cls(tuple(sorted(flags)), activation_dtype, device,
                   compute_capability, kv_block_size, tensor_parallel,
                   tuple(peak_families))

    @classmethod
    def at(cls, *, gpu_sku: Optional[str], activation_dtype: Optional[str],
           server_flags: Optional[Mapping[str, ServerOption]] = None,
           kv_block_size: Optional[int] = None,
           tensor_parallel: Optional[int] = None) -> "ServingDeployment":
        """The deployment an estimate request describes: its GPU SKU's
        `device` and `compute_capability` (`gpu_specs.csv`), its activation
        dtype, and the server options it set. A SKU or a dtype the tables do
        not know leaves that fact unknown."""
        gpu = get_gpu(gpu_sku) if gpu_sku else None
        try:
            dtype = (None if activation_dtype is None
                     else canonical(activation_dtype))
        except UnknownDtype:
            dtype = None
        return cls.of(server_flags=server_flags, activation_dtype=dtype,
                      device=gpu.device if gpu is not None else None,
                      compute_capability=(gpu.compute_capability or None)
                      if gpu is not None else None,
                      kv_block_size=kv_block_size,
                      tensor_parallel=tensor_parallel,
                      peak_families=(peak_families(gpu)
                                     if gpu is not None else ()))

    def flag(self, name: str, default: ServerOption) -> ServerOption:
        return dict(self.server_flags).get(name, default)


#: One value of a framework's server selector. Boolean covers a switch;
#: string and integer cover the choices vLLM exposes (for example
#: `linear_backend=auto` or `flash_attn_version=3`).
ServerOption = Union[bool, str, int]


def capability(text: str, where: str = "compute capability"
               ) -> Tuple[int, int]:
    """`"10.0"` -> `(10, 0)`: a CUDA compute capability compared as the
    framework compares it, major then minor."""
    major, dot, minor = str(text).partition(".")
    if not (dot and major.isdigit() and minor.isdigit()):
        raise ValueError(f"{where}: {text!r} is not `major.minor`")
    return int(major), int(minor)


@dataclass(frozen=True)
class ResolvedLaunch:
    """The precision-resolved identity of one per-op launch.

    This is not a second precision resolver: `point_keys` builds it from
    `RowIdentity.leaf_dtype_env`, the same dtype environment that becomes
    the PhysicalKey, after the precision profile and `precision_launches`
    have materialized the walk. Dispatch conditions may read the closed
    semantic facts here (currently whether the whole applicable operand
    environment is plain BF16); they cannot enumerate weight schemes or MMA
    families in a binding file.
    """
    activation_dtype: Optional[str]
    dtype_env: Mapping[str, Any]

    @classmethod
    def from_dtype_env_json(cls, text: str) -> "ResolvedLaunch":
        env = json.loads(text)
        if not isinstance(env, dict):
            raise ValueError("a resolved launch dtype environment is an object")
        activation = env.get("activation")
        if not isinstance(activation, str):
            raise ValueError("a resolved launch states its activation dtype")
        return cls(activation, env)

    @property
    def unquantized_bf16(self) -> bool:
        """Whether every launch operand this key states is plain BF16.

        A launch without a weight, KV, or cache-format slot does not invent
        one; a slot it does state must be `bf16`, and no scale granularity
        or named quantized format may be present.
        """
        operand_slots = ("activation", "weight", "kv", "output")
        shape_slots = ("weight_granularity", "activation_granularity",
                       "output_format", "kv_format", "metadata")
        return (all(self.dtype_env.get(slot) == "bf16"
                    for slot in operand_slots if slot in self.dtype_env)
                and not any(slot in self.dtype_env for slot in shape_slots))


def _roles_by_node(raw: Any, where: str
                   ) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """`when.roles` as `{<pattern node>: [<role>, ...]}`.

    Keyed by the pattern's own node name because that is what carries the
    difference: V4.1's indexer key and its latent KV are the same norm → rope
    → re-join and only the two NORMS have different roles, so "some member is
    an indexer node" cannot tell the chains apart while "the node this match
    bound as the pattern's norm is an indexer node" can — and reads both ways,
    so each framework states which chain its kernel runs. The node name is
    checked against the match (`GraphCondition.admits_match`); the roles are
    checked here, against `roles.yaml`.
    """
    if raw is None:
        return ()
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"{where}: `when.roles` is a mapping from a pattern node to the "
            f"roles its match may be declared in "
            f"(`{{norm: [attention_indexer]}}`); got {raw!r}")
    out = []
    for node, allowed in raw.items():
        if isinstance(allowed, str):
            allowed = [allowed]
        if (not isinstance(node, str) or not isinstance(allowed, list)
                or not allowed
                or not all(isinstance(r, str) for r in allowed)):
            raise ValueError(
                f"{where}: `when.roles[{node!r}]` is a non-empty list of "
                f"role names; got {allowed!r}")
        unknown = [r for r in allowed if not roles_registry.is_registered(r)]
        if unknown:
            raise ValueError(
                f"{where}: `when.roles[{node!r}]` {unknown} are not roles of "
                "roles.yaml")
        out.append((node, tuple(allowed)))
    return tuple(out)


def _reject_shared_patterns(fname: str, ops) -> None:
    """Two plans of one binding over one naive pattern — nodes, edges and
    io_mapping — are refused: which one matches would be the yaml order. A
    launch that writes its output quantised is not such a twin of its plain
    form; it names the cast as a member (`fused_q_kv_rmsnorm_quant`), so
    its pattern is the larger one and specificity tries it first."""
    groups: Dict[str, list] = {}
    for op in ops:
        key = FUSION_SCHEMES[op].naive_pattern.model_dump_json()
        groups.setdefault(key, []).append(op)
    for members in groups.values():
        if len(members) > 1:
            raise ValueError(
                f"framework_bindings/{fname}: {members!r} bind one pattern; "
                "which one matches would be the yaml order")


def _among(fact: Optional[str], allowed: Tuple[str, ...]) -> bool:
    """Whether a deployment fact is one a condition lists; a condition that
    lists none asks nothing, and an unknown fact satisfies any list."""
    return not allowed or (fact is not None and fact.lower() in allowed)


@dataclass(frozen=True)
class DispatchCondition:
    """When a framework runs a bound kernel, as its source decides it.

    The vocabulary is closed (`KEYS`); each key is a fact the analysis
    already owns:

    * `server_flag: {name, default, value}` — a server argument the
      dispatch reads (sglang `enable_fused_qk_norm_rope`, default false); the
      entry holds while the flag is `value` (true when omitted). The caller
      sets it through the API (`framework_options`); unset, the default holds.
      A non-boolean choice is stated the same way: vLLM's audited linear
      path is `linear_backend=auto`, while another request may say
      `flashinfer_cutedsl`.
    * `activation_dtype: [...]` — the deployment's activation dtype.
    * `kv_block_size: [<positive page>, ...]` — the paged-KV page the request
      deploys with. The binding's default is resolved once by the caller
      before the condition is asked; the condition never derives a provider
      implementation suffix from it.
    * `tensor_parallel: [<degree>, ...]` — the degree of the vocabulary/
      attention tensor-parallel world. It scopes a declaration audited only
      on an unsharded deployment; PP/EP alone do not change it.
    * `device: [...]` — the GPU SKU's `device` (`gpu_specs.csv`).
    * `min_compute_capability: "<major>.<minor>"` — the GPU SKU's CUDA
      compute capability (`gpu_specs.csv`) is at least this, the threshold a
      framework's kernel selection tests (vLLM `has_device_capability(100)`
      is `>= (10, 0)`, vllm/platforms/interface.py:531, comparison :549;
      the family test `is_device_capability_family` :577-588; sglang
      `is_sm100_or_newer`, a major of 10 or more).
    * `below_compute_capability: "<major>.<minor>"` — the same fact, below
      this (exclusive): a kernel a framework replaces from some capability on
      (vLLM V4.1's SM100 default attention takes the q rotation out of the
      KV-insert launch, models/deepseek_v41/nvidia/model.py:235-243).
      `unquantized_bf16_launch: true` — a fact of the precision-resolved
      launch, supplied per row after `precision_launches` has materialized
      it. It is one boolean, not a second dtype table: the launch is active
      only when every operand slot it has is plain BF16.
      `excluded_architectures`, `head_dim`,
      `rotation_covers_head`, `dims`, `phase`, `world_local: {world, dim,
      divides: true | local_in: [...]}`, `readers_eat`, `roles`, `routed` — the part the walk answers, asked during the rewrite
      (`eval.fusion.GraphCondition`). `phase` names phases of `dims.PHASES`;
      it is the walk's, not the deployment's, because one deployed binding
      rewrites both of an estimate's walks.
    """
    server_flag: Optional[Tuple[str, ServerOption]] = None
    activation_dtype: Tuple[str, ...] = ()
    kv_block_size: Tuple[int, ...] = ()
    tensor_parallel: Tuple[int, ...] = ()
    device: Tuple[str, ...] = ()
    min_compute_capability: Optional[Tuple[int, int]] = None
    below_compute_capability: Optional[Tuple[int, int]] = None
    unquantized_bf16_launch: Optional[bool] = None
    graph: GraphCondition = field(default_factory=GraphCondition)
    server_flag_value: ServerOption = True

    KEYS = frozenset({"server_flag", "activation_dtype", "device",
                      "kv_block_size", "tensor_parallel",
                      "min_compute_capability", "below_compute_capability",
                      "unquantized_bf16_launch",
                      "architectures",
                      "excluded_architectures", "head_dim",
                      "rotation_covers_head", "dims", "phase",
                      "world_local", "readers_eat", "roles", "routed"})

    #: The keys a CARD decides, each naming the `ServingDeployment` field it
    #: reads — the fields `ServingDeployment.at` fills off the sheet row.
    #: `holds_for_card` is their one test and `card_dispatch_facts` reads a
    #: card through exactly these fields, so the walk class of a card
    #: (`card_class`) is this vocabulary rather than a second statement of
    #: it: a `when:` key over a new card fact enters the class by being
    #: classified here. A key classified in neither this nor
    #: `DEPLOYMENT_KEYS` nor `GraphCondition`'s fields is refused at load
    #: (`_check_dispatch_vocabulary`).
    CARD_KEYS: ClassVar[Mapping[str, str]] = MappingProxyType({
        "device": "device",
        "min_compute_capability": "compute_capability",
        "below_compute_capability": "compute_capability"})
    #: The keys the DEPLOYMENT decides: the caller's server flags and the
    #: walk's activation dtype. Facts of a request, not of a card — two
    #: cards of one walk class answer them alike by construction.
    DEPLOYMENT_KEYS: ClassVar[FrozenSet[str]] = frozenset(
        {"server_flag", "activation_dtype", "kv_block_size",
         "tensor_parallel"})
    #: The keys the precision-resolved LAUNCH decides, supplied per row by
    #: `point_keys` after `precision_launches` has materialized the walk.
    #: They are neither card nor request facts and are never guessed from a
    #: checkpoint id.
    LAUNCH_KEYS: ClassVar[FrozenSet[str]] = frozenset(
        {"unquantized_bf16_launch"})

    def holds_for(self, deployment: ServingDeployment,
                  launch: Optional["ResolvedLaunch"] = None) -> bool:
        """Whether the deployment's facts satisfy every deployment key; an
        unknown fact satisfies none. A supplied `launch` owns the
        precision-resolved activation: `activation_dtype` then reads that
        launch, not the request's model dtype."""
        activation = (deployment.activation_dtype if launch is None
                      else launch.activation_dtype)
        return (self.holds_for_card(deployment)
                and (self.server_flag is None
                or deployment.flag(*self.server_flag)
                    == self.server_flag_value)
                and _among(activation, self.activation_dtype)
                and (self.unquantized_bf16_launch is None
                     or (launch is not None
                         and launch.unquantized_bf16
                         == self.unquantized_bf16_launch))
                and (not self.kv_block_size
                     or (deployment.kv_block_size is not None
                         and deployment.kv_block_size in self.kv_block_size))
                and (not self.tensor_parallel
                     or (deployment.tensor_parallel is not None
                         and deployment.tensor_parallel
                         in self.tensor_parallel)))

    def holds_for_card(self, deployment: ServingDeployment) -> bool:
        """Whether the CARD's facts satisfy this condition's card keys
        (`CARD_KEYS`): the part of a dispatch a card decides — the device
        its kernel needs and the capability window it runs in. The one test
        of them, so `holds_for` and the walk class read the same rule."""
        return (_among(deployment.device, self.device)
                and (self.min_compute_capability is None
                     or (deployment.compute_capability is not None
                         and capability(deployment.compute_capability)
                         >= self.min_compute_capability))
                and (self.below_compute_capability is None
                     or (deployment.compute_capability is not None
                         and capability(deployment.compute_capability)
                         < self.below_compute_capability)))

    def as_dict(self) -> Dict[str, Any]:
        """The condition as the binding file states it (set keys only)."""
        out: Dict[str, Any] = {}
        if self.server_flag is not None:
            out["server_flag"] = {"name": self.server_flag[0],
                                  "default": self.server_flag[1]}
            if self.server_flag_value != self.server_flag[1]:
                out["server_flag"]["value"] = self.server_flag_value
        for key, value in (("architectures", self.graph.architectures),
                           ("excluded_architectures",
                            self.graph.excluded_architectures),
                           ("head_dim", self.graph.head_dim),
                           ("phase", self.graph.phase),
                           ("activation_dtype", self.activation_dtype),
                           ("kv_block_size", self.kv_block_size),
                           ("tensor_parallel", self.tensor_parallel),
                           ("device", self.device)):
            if value:
                out[key] = list(value)
        if self.min_compute_capability is not None:
            out["min_compute_capability"] = "%d.%d" % self.min_compute_capability
        if self.below_compute_capability is not None:
            out["below_compute_capability"] = (
                "%d.%d" % self.below_compute_capability)
        if self.unquantized_bf16_launch is not None:
            out["unquantized_bf16_launch"] = self.unquantized_bf16_launch
        if self.graph.rotation_covers_head is not None:
            out["rotation_covers_head"] = self.graph.rotation_covers_head
        if self.graph.world_local is not None:
            world, dim, local_in = self.graph.world_local
            out["world_local"] = ({"world": world, "dim": dim,
                                   "local_in": list(local_in)} if local_in
                                  else {"world": world, "dim": dim,
                                        "divides": True})
        if self.graph.readers_eat is not None:
            out["readers_eat"] = self.graph.readers_eat
        if self.graph.routed is not None:
            out["routed"] = self.graph.routed
        if self.graph.roles:
            out["roles"] = {node: list(allowed)
                            for node, allowed in self.graph.roles}
        if self.graph.dims:
            out["dims"] = {name: list(values)
                           for name, values in self.graph.dims}
        return out

    @classmethod
    def parse(cls, raw: Any, where: str,
              allowed: frozenset = KEYS) -> "DispatchCondition":
        if not isinstance(raw, dict):
            raise ValueError(f"{where}: `when` must be a mapping")
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(
                f"{where}: `when` keys {unknown} are not in the vocabulary "
                f"{sorted(allowed)}")
        flag = raw.get("server_flag")
        value = True
        if flag is not None:
            def option(item: Any) -> bool:
                return (isinstance(item, (bool, int))
                        or (isinstance(item, str) and bool(item.strip())))
            if (not isinstance(flag, dict)
                    or not {"name", "default"} <= set(flag)
                    or not set(flag) <= {"name", "default", "value"}
                    or not isinstance(flag["name"], str)
                    or (isinstance(flag["name"], str) and not flag["name"])
                    or not option(flag["default"])
                    or not option(flag.get("value", flag["default"]))):
                raise ValueError(
                    f"{where}: `when.server_flag` is "
                    f"{{ name: <server argument>, default: <bool, string, "
                    f"or integer>, value: <that type, default when "
                    f"omitted> }}")
            # Boolean switches preserve Batch 1's spelling: an omitted value
            # means TRUE even when the framework's default is false. A
            # choice-valued selector instead keeps its declared default when
            # the request does not state one.
            value = flag.get(
                "value", True if isinstance(flag["default"], bool)
                else flag["default"])
            flag = (flag["name"], flag["default"])

        def names(key: str) -> Tuple[str, ...]:
            value = raw.get(key) or []
            if not isinstance(value, list) or not all(
                    isinstance(v, str) for v in value):
                raise ValueError(f"{where}: `when.{key}` is a list of names")
            return tuple(value)

        def integers(key: str) -> Tuple[int, ...]:
            value = raw.get(key) or []
            if not isinstance(value, list) or not all(
                    isinstance(v, int) and not isinstance(v, bool)
                    for v in value):
                raise ValueError(f"{where}: `when.{key}` is a list of integers")
            return tuple(value)

        for key in ("kv_block_size", "tensor_parallel"):
            if any(value <= 0 for value in integers(key)):
                raise ValueError(
                    f"{where}: `when.{key}` is a list of positive integers")

        def dims() -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
            value = raw.get("dims") or {}
            if not isinstance(value, dict) or not all(
                    isinstance(name, str) and isinstance(vals, list) and vals
                    and all(isinstance(v, int) and not isinstance(v, bool)
                            for v in vals)
                    for name, vals in value.items()):
                raise ValueError(f"{where}: `when.dims` maps a dim name to a "
                                 "non-empty list of integers")
            return tuple(sorted((name, tuple(vals))
                                for name, vals in value.items()))

        phases = names("phase")
        unknown_phases = sorted(set(phases) - set(PHASES))
        if unknown_phases:
            raise ValueError(
                f"{where}: `when.phase` names {unknown_phases}, which are not "
                f"phases {list(PHASES)}")
        local = raw.get("world_local")
        if local is not None:
            forms = ({"world", "dim", "divides"}, {"world", "dim", "local_in"})
            if (not isinstance(local, dict) or set(local) not in forms
                    or not isinstance(local["world"], str)
                    or not isinstance(local["dim"], str)
                    or ("divides" in local and local["divides"] is not True)
                    or ("local_in" in local and not (
                        isinstance(local["local_in"], list) and local["local_in"]
                        and all(isinstance(v, int) and not isinstance(v, bool)
                                and v > 0 for v in local["local_in"])))):
                raise ValueError(
                    f"{where}: `when.world_local` is {{ world: <world label>, "
                    f"dim: <dim symbol>, divides: true }} or {{ world, dim, "
                    f"local_in: [<positive per-rank counts>] }}")
            if local["world"] not in world_by_label():
                raise ValueError(
                    f"{where}: `when.world_local.world` {local['world']!r} is "
                    f"not a world label {sorted(world_by_label())}")
            if local["world"] not in {
                    w.label for w in capabilities_of(local["dim"])}:
                raise ValueError(
                    f"{where}: `when.world_local.dim` {local['dim']!r} is not "
                    f"a symbol world {local['world']!r} may split (dims.yaml "
                    f"`shardable_by`)")
            local = (local["world"], local["dim"],
                     tuple(local.get("local_in") or ()))
        covers = raw.get("rotation_covers_head")
        if covers is not None and not isinstance(covers, bool):
            raise ValueError(f"{where}: `when.rotation_covers_head` is a bool")
        unquantized = raw.get("unquantized_bf16_launch")
        if unquantized is not None and not isinstance(unquantized, bool):
            raise ValueError(
                f"{where}: `when.unquantized_bf16_launch` is a bool")
        declared_roles = _roles_by_node(raw.get("roles"), where)
        routed = raw.get("routed")
        if routed is not None and not isinstance(routed, bool):
            raise ValueError(f"{where}: `when.routed` is a bool")
        eats = raw.get("readers_eat")
        if eats is not None and eats not in ACTIVATION_FORMATS:
            raise ValueError(
                f"{where}: `when.readers_eat` {eats!r} names no activation "
                f"format (arch/activation_formats/); known: "
                f"{sorted(ACTIVATION_FORMATS)}")
        def bound(key: str) -> Optional[Tuple[int, int]]:
            value = raw.get(key)
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError(f"{where}: `when.{key}` is a `major.minor` "
                                 "string")
            return capability(value, where)

        minimum = bound("min_compute_capability")
        below = bound("below_compute_capability")
        return cls(
            server_flag=flag,
            activation_dtype=tuple(canonical(n).lower()
                                   for n in names("activation_dtype")),
            kv_block_size=integers("kv_block_size"),
            tensor_parallel=integers("tensor_parallel"),
            device=tuple(n.lower() for n in names("device")),
            min_compute_capability=minimum,
            below_compute_capability=below,
            unquantized_bf16_launch=unquantized,
            graph=GraphCondition(architectures=names("architectures"),
                                 excluded_architectures=names(
                                     "excluded_architectures"),
                                 head_dim=integers("head_dim"),
                                 rotation_covers_head=covers,
                                 dims=dims(),
                                 phase=phases,
                                 world_local=local,
                                 readers_eat=eats,
                                 roles=declared_roles,
                                 routed=routed),
            server_flag_value=value)


def _check_dispatch_vocabulary() -> None:
    """Every `when:` key is classified at load: a fact of the CARD
    (`DispatchCondition.CARD_KEYS`, each naming the `ServingDeployment`
    field it reads), of the DEPLOYMENT (`DEPLOYMENT_KEYS`), of the
    precision-resolved launch (`LAUNCH_KEYS`), or of the walk (a field of
    `GraphCondition`). A key in none of them is refused here
    rather than read by `holds_for` alone, because a card fact outside
    `CARD_KEYS` would leave `card_class` — which reads a card through those
    fields — calling two cards one walk class while a framework dispatches
    them differently. Run at import: adding a key to `KEYS` without saying
    whose fact it is fails the load.

    `GraphCondition.PROGRAM_FIELDS` is checked here too, against that
    dataclass's own fields. It is a literal name list, and CLAUDE.md §1.4
    allows one only where something hard-rejects on it at load; this is
    that rejection.
    """
    graph_keys = {f.name for f in fields(GraphCondition)}
    unknown_program = sorted(set(GraphCondition.PROGRAM_FIELDS) - graph_keys)
    if unknown_program:
        raise ValueError(
            f"framework_bindings: GraphCondition.PROGRAM_FIELDS names "
            f"{unknown_program}, which are not fields of GraphCondition. "
            f"That list is what `kernel_dispatches` may ask of a walk with "
            f"no match (`_KERNEL_DISPATCH_WHEN`); a name that is not a "
            f"field answers nothing")
    card_keys = set(DispatchCondition.CARD_KEYS)
    classified = (card_keys | DispatchCondition.DEPLOYMENT_KEYS
                  | DispatchCondition.LAUNCH_KEYS | graph_keys)
    unclassified = sorted(DispatchCondition.KEYS - classified)
    if unclassified:
        raise ValueError(
            f"framework_bindings: `when:` key(s) {unclassified} are in "
            f"DispatchCondition.KEYS and classified nowhere. Say whose fact "
            f"each is: CARD_KEYS (naming the ServingDeployment field it "
            f"reads, so card_class carries it), DEPLOYMENT_KEYS, "
            f"LAUNCH_KEYS, or a field of GraphCondition")
    stray = sorted(classified - DispatchCondition.KEYS)
    if stray:
        raise ValueError(
            f"framework_bindings: {stray} are classified as `when:` keys and "
            f"are not in DispatchCondition.KEYS")
    overlap = sorted((card_keys | DispatchCondition.DEPLOYMENT_KEYS
                      | DispatchCondition.LAUNCH_KEYS) & graph_keys)
    if overlap:
        raise ValueError(
            f"framework_bindings: {overlap} are classified as both a "
            f"deployment fact and a fact of the walk")
    own = {f.name for f in fields(DispatchCondition)}
    not_declared = sorted(
        (card_keys | DispatchCondition.DEPLOYMENT_KEYS
         | DispatchCondition.LAUNCH_KEYS) - own)
    if not_declared:
        raise ValueError(
            f"framework_bindings: {not_declared} are classified as keys "
            f"DispatchCondition answers and are not fields of it")
    deployment_fields = {f.name for f in fields(ServingDeployment)}
    unread = sorted(set(DispatchCondition.CARD_KEYS.values())
                    - deployment_fields)
    if unread:
        raise ValueError(
            f"framework_bindings: CARD_KEYS name {unread}, which are not "
            f"fields of ServingDeployment; a card fact reaches a dispatch "
            f"through one (`ServingDeployment.at` fills them off "
            f"gpu_specs.csv)")


_check_dispatch_vocabulary()


def _card_walk_columns() -> FrozenSet[str]:
    """The `gpu_specs.csv` columns that change a WALK — `CardClass`'s two
    halves, each derived from the declaration it already has rather than
    listed a second time:

    * the columns a framework's dispatch reads of a card. Those are
      `DispatchCondition.CARD_KEYS`' sheet side: each value names the
      `ServingDeployment` field the key reads, and `ServingDeployment.at`
      fills those off the sheet row of the same name (`device`,
      `compute_capability`). `peak_families` is the one that is not a
      column of its own — it IS the peak columns below;
    * the peak columns the sheet lists (`gpu_specs.PEAK_COLUMNS`), because
      a leaf whose dtype has no column is walked weight-only and a
      deployment-chosen dtype without one is refused (`NoPeakForDtype`).

    A card key naming a `ServingDeployment` field that is neither a sheet
    column nor `peak_families` is refused by
    `_check_card_fact_classification`: it would be a card fact the sheet
    has no column to answer with.
    """
    columns = {f.name for f in fields(gpu_specs.GPUSpec)}
    read = set(DispatchCondition.CARD_KEYS.values())
    unknown = sorted(read - columns - {"peak_families"})
    if unknown:
        raise ValueError(
            f"framework_bindings: CARD_KEYS read the ServingDeployment "
            f"field(s) {unknown}, which gpu_specs.csv has no column of and "
            f"which are not `peak_families`. A card fact a dispatch reads "
            f"comes off the sheet row (`ServingDeployment.at`): add the "
            f"column, or read the field the sheet already has")
    return frozenset((read & columns)
                     | set(gpu_specs.PEAK_COLUMNS.values()))


def _check_card_fact_classification() -> None:
    """Every column of `gpu_specs.csv` is classified at load: a fact that
    changes a WALK (`_card_walk_columns`), a fact that changes a PRICE
    (`gpu_specs.PRICE_COLUMNS`), or one nothing walks or prices off
    (`gpu_specs.DESCRIPTIVE_COLUMNS`, each saying why). A column in none of
    the three is refused here.

    The same shape as `_check_dispatch_vocabulary` and for the same reason,
    one level out. That gate makes a `when:` key say whose fact it is so
    that `card_class` cannot call two cards one walk class while a
    framework dispatches them differently. This one makes a sheet COLUMN
    say the same thing, because the split a walk is reused across is
    exactly walk-fact against price-fact: a column added to the sheet and
    read by a walk, with nobody having said it is a walk fact, leaves
    `CardClass` too coarse and every reuse of a walk inside that class
    silently wrong — which is how an RTX-4090 walk came to be published as
    six other cards' answer (`capacity._refuse_unless_one_class`).

    A column may be both (`compute_capability` is: a dispatch reads it and
    a collective rung's window reads it; a peak column is, by value and by
    presence). The third bucket may not overlap either — "descriptive" is
    a statement that nothing reads it, so it cannot also be read. A new
    CSV column with no `GPUSpec` field never gets this far: `_parse_row`
    refuses it where the sheet is read.
    """
    columns = {f.name for f in fields(gpu_specs.GPUSpec)}
    walk = _card_walk_columns()
    card_fields = set(ServingDeployment.CARD_FIELDS)
    expected_card_fields = (
        set(DispatchCondition.CARD_KEYS.values()) & columns) | {"peak_families"}
    if card_fields != expected_card_fields:
        raise ValueError(
            "framework_bindings: ServingDeployment.CARD_FIELDS is "
            f"{sorted(card_fields)}, but the card facts a dispatch can read "
            f"are {sorted(expected_card_fields)}. A serving launch choice "
            "cannot grow a third card fact outside CardClass; add it to "
            "CARD_KEYS and the sheet, or stop reading it")
    price = set(gpu_specs.PRICE_COLUMNS)
    descriptive = set(gpu_specs.DESCRIPTIVE_COLUMNS)
    stray = sorted((walk | price | descriptive) - columns)
    if stray:
        raise ValueError(
            f"framework_bindings: {stray} are classified as gpu_specs.csv "
            f"columns and are not fields of gpu_specs.GPUSpec")
    unclassified = sorted(columns - walk - price - descriptive)
    if unclassified:
        raise ValueError(
            f"gpu_specs.csv column(s) {unclassified} are classified "
            f"nowhere. Say what each changes: a WALK (give it a "
            f"`DispatchCondition.CARD_KEYS` entry, or a peak column the "
            f"sheet's naming law covers), a PRICE (`gpu_specs."
            f"PRICE_COLUMNS`, naming the accessor that reads it), or "
            f"nothing (`gpu_specs.DESCRIPTIVE_COLUMNS`, saying why). A "
            f"column a walk reads with nobody having said so leaves a card "
            f"class too coarse, and every walk reused inside it is then "
            f"one card's answer published as another's")
    read_anyway = sorted(descriptive & (walk | price))
    if read_anyway:
        raise ValueError(
            f"gpu_specs.csv column(s) {read_anyway} are declared "
            f"descriptive-only in gpu_specs.DESCRIPTIVE_COLUMNS and are "
            f"also classified as a walk or a price fact; a column nothing "
            f"reads cannot also be read")


_check_card_fact_classification()


def card_dispatch_facts(gpu: Any) -> Tuple[Tuple[str, Any], ...]:
    """Every fact of a card a framework's `when:` can read, as the dispatch
    reads it: the `ServingDeployment` fields `DispatchCondition.CARD_KEYS`
    names, filled off the sheet row (`ServingDeployment.at`).

    The one statement of which card facts can change a dispatch. A kernel a
    framework runs only on some cards is bound over these and nothing else
    (`holds_for_card`), so two cards that answer them alike are dispatched
    alike by every binding — which is what makes one walk serve both
    (`card_class`). A `when:` key over a further card fact is classified
    into `CARD_KEYS` or refused at load (`_check_dispatch_vocabulary`), and
    classifying it moves the class with it."""
    at = ServingDeployment.at(gpu_sku=gpu.sku, activation_dtype=None)
    return tuple((name, getattr(at, name)) for name in
                 sorted(set(DispatchCondition.CARD_KEYS.values())))


class CardClass(NamedTuple):
    """The card facts that change a walk's STRUCTURE — as opposed to its
    price, which every other card fact (HBM, bandwidths, node size) is.

    A walk's row set and laws move with the card only through:

    * what a framework's dispatch reads of the card (`card_dispatch_facts`:
      the device and the compute capability its `when:` keys test), because
      two cards it answers differently fuse the same graph differently;
    * the peak columns the sheet lists (`gpu_specs.peak_families`) — a leaf
      whose dtype has no column is walked weight-only (activation at the
      edge dtype, no convert term), and a deployment-chosen dtype without a
      column is refused (`NoPeakForDtype`). Both bake into the row, so they
      are walk-time.

    Cards sharing a class walk identically; their prices still differ —
    HBM, bandwidths and `gpus_per_node` are substituted when the walk is
    priced on the card, as the row laws' card symbols already allow
    (`row_roofline.CardForm`).
    """
    #: `(field, value)` per card fact a dispatch reads
    #: (`card_dispatch_facts`).
    dispatch: Tuple[Tuple[str, Any], ...]
    #: The peak families the card lists a column for
    #: (`gpu_specs.peak_families`).
    dtype_peaks: Tuple[str, ...]

    def key(self) -> str:
        """The class as one key part: stable, printable, collision-free."""
        facts = "-".join(f"{name}={value or 'none'}"
                         for name, value in self.dispatch)
        return f"{facts}-peaks={'+'.join(self.dtype_peaks) or 'none'}"


def card_class(gpu: Any) -> CardClass:
    """The walk class of a card (`CardClass`)."""
    return CardClass(card_dispatch_facts(gpu), peak_families(gpu))


@dataclass(frozen=True)
class SourceLocation:
    """One place in the framework's repository an anchor cites: a path from
    the repository root and, when it names lines, the first and last."""
    path: str
    first: Optional[int] = None
    last: Optional[int] = None

    def __str__(self) -> str:
        if self.first is None:
            return self.path
        return self.path + (f":{self.first}" if self.last == self.first
                            else f":{self.first}-{self.last}")


@dataclass(frozen=True)
class SourceAnchor:
    """Where a framework's source shows it runs a fused op as one launch: a
    sentence for a reader, in which every place it cites is written as
    `<path from the repository root>[:<line>[-<line>]]`. The places are
    parsed out once at load (`parse`), so a link to the audited commit is
    built from them (`AuditedAt.cite`) and never by guessing from the text."""
    text: str
    #: The sentence split at the places it cites, in order: plain text and
    #: `SourceLocation`s alternating.
    parts: Tuple[Any, ...] = ()

    #: A cited place: a repository path (at least one directory) with a
    #: source-file extension, optionally followed by `:<line>` or
    #: `:<first>-<last>`.
    _LOCATION = re.compile(
        r"(?P<path>[\w.-]+(?:/[\w.-]+)+\.(?:py|cu|cuh|cpp|cc|c|h|hpp))(?!\w)"
        r"(?::(?P<first>\d+)(?:-(?P<last>\d+))?)?")
    #: What reads as a citation but names no file: a bare `:<line>`, or a
    #: file name with no directory. The location is ambiguous without the
    #: path, so the loader refuses it rather than resolve it.
    _UNANCHORED = re.compile(
        r"(?<![\w./-]):\d+|(?<![\w./-])[\w-]+\.(?:py|cu|cuh|cpp|cc|c|h|hpp)(?!\w)")

    @classmethod
    def parse(cls, text: str, where: str) -> "SourceAnchor":
        parts: list = []
        at = 0
        for m in cls._LOCATION.finditer(text):
            parts.append(text[at:m.start()])
            first = None if m["first"] is None else int(m["first"])
            last = None if m["last"] is None else int(m["last"])
            if first is not None and last is not None and last < first:
                raise ValueError(f"{where}: `source_anchor` cites "
                                 f"{m.group(0)!r}, whose last line is before "
                                 f"its first.")
            parts.append(SourceLocation(
                m["path"], first, first if last is None else last))
            at = m.end()
        parts.append(text[at:])
        loose = [m.group(0) for p in parts if isinstance(p, str)
                 for m in cls._UNANCHORED.finditer(p)]
        if loose:
            raise ValueError(
                f"{where}: `source_anchor` cites {loose} without a path from "
                f"the repository root; write every place as "
                f"`<path>[:<line>[-<line>]]` so it links to the audited "
                f"commit.")
        if not any(isinstance(p, SourceLocation) for p in parts):
            raise ValueError(
                f"{where}: `source_anchor` {text!r} cites no place in the "
                f"framework's source; name at least one "
                f"`<path>[:<line>[-<line>]]`.")
        return cls(text=text, parts=tuple(p for p in parts if p != ""))

    @property
    def locations(self) -> Tuple[SourceLocation, ...]:
        return tuple(p for p in self.parts if isinstance(p, SourceLocation))

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class FusionImplementation:
    """One implementation entry for a fused op — where the framework's source
    shows it runs that span as one launch, and `when` it does (empty:
    whenever the pattern matches).

    `when:` is one condition, or a list of them: the entry holds where ANY
    of them holds. One condition's keys are conjunctive, and a framework
    can take one launch on two paths that no single conjunction states —
    vLLM's Kimi-K3 convolves the packed q|k|v row in one launch at decode
    only, its GLM-5.3 in both phases.

    The fused_op name is the key inside `FrameworkBinding.op_implementations`,
    not duplicated on the impl. A binding may hold several entries for one
    name; their conditions must be provably disjoint at load. They are still
    fusion-span declarations, not provider identities: the graph rewrite sees
    only the fused-op name, while `kernel_dispatches` separately declares the
    provider implementation a measured row carries.
    """
    source_anchor: Optional[SourceAnchor] = None
    #: The alternatives of `when:`, in file order; empty for an entry that
    #: states none.
    whens: Tuple[DispatchCondition, ...] = ()
    #: The widest boundary this framework's launch spans, when it spans more
    #: than one component instance of one block: `component` — any component
    #: under one block occurrence (`eval.fusion.BoundaryView`). Unset: the
    #: launch lives in one component instance, which is every plan that was
    #: written before a kernel crossed one. It is a fact about the framework's
    #: launch, like `source_anchor` and `when`, so it is stated here and not
    #: on the plan: DeepSeek-V4's post+pre is one launch across two
    #: components in vLLM, and sglang's GLM fuses only the in-layer one.
    crosses: Optional[str] = None

    KEYS = frozenset({"source_anchor", "when", "crosses"})

    @property
    def when(self) -> Optional[DispatchCondition]:
        """The entry's one condition (None: unconditioned). An entry with
        alternatives has no single condition, and asking for one is refused
        rather than answered with the first."""
        if len(self.whens) > 1:
            raise ValueError(
                "this entry's `when:` is a list of alternatives; read "
                "`whens`")
        return self.whens[0] if self.whens else None


@dataclass(frozen=True)
class KernelDispatch:
    """One serving framework's audited choice of a provider implementation.

    `impl_id` is not a base name: it is the complete identity a provider
    records for the audited cell, including a parameter suffix such as
    ``@compute=fp32`` or ``@_b16``. `library` is the provider provenance a
    corpus row carries in its ``framework`` column, not the serving framework.

    Batch 1 deliberately answers only facts available without a fusion match:
    deployment/card facts (including the served page and TP degree), model
    architecture, and phase. Match-bound graph conditions are rejected at load
    and belong to a later rewrite-side dispatch, not to a primitive row
    selector. A condition may scope an audited parameterised cell to its page;
    it must not derive the provider's suffix, whose one owner is
    ``Provider.impl_id``.
    """
    op: str
    library: str
    impl_id: str
    source_anchor: SourceAnchor
    whens: Tuple[DispatchCondition, ...] = ()

    KEYS = frozenset({"op", "library", "impl_id", "when", "source_anchor"})


@dataclass(frozen=True)
class DispatchDecision:
    """The closed result of asking a binding which implementation serves an op.

    ``undeclared`` and ``no_active`` are distinct audit facts, but both leave
    latest without a measurement and therefore map to ambiguous coverage.
    Neither is a request to measure a new corpus row.
    """
    state: str
    library: Optional[str] = None
    impl_id: Optional[str] = None
    reason: str = ""
    source_anchor: Optional[SourceAnchor] = None

    DECLARED = "declared"
    UNDECLARED = "undeclared"
    NO_ACTIVE = "no_active"
    AMBIGUOUS = "ambiguous"

    def as_dict(self) -> Dict[str, Any]:
        """Plain data for a serialized point-measurement input."""
        return {"state": self.state, "library": self.library,
                "impl_id": self.impl_id, "reason": self.reason}


@dataclass(frozen=True)
class AuditedAt:
    """The release a binding's facts were read at: the repository (GitHub
    `owner/name`), its tag and commit."""
    tag: str
    commit: str
    repo: str = ""

    @property
    def release(self) -> str:
        """The tag as a package version (`v0.30.0` → `0.30.0`)."""
        return self.tag[1:] if self.tag[:1] in ("v", "V") else self.tag

    def label(self, framework: str) -> str:
        return f"{framework} (audited at {self.tag})"

    def url(self, location: SourceLocation) -> str:
        """The permalink to `location` at the audited commit."""
        url = (f"https://github.com/{self.repo}/blob/{self.commit}/"
               f"{location.path}")
        if location.first is None:
            return url
        return url + (f"#L{location.first}" if location.last == location.first
                      else f"#L{location.first}-L{location.last}")

    def cite(self, anchor: SourceAnchor) -> Dict[str, Any]:
        """`anchor` as a reader sees it: its text, and the same text split
        into plain runs and places, each place with its permalink."""
        return {"text": anchor.text, "segments": [
            {"text": p} if isinstance(p, str)
            else {"text": str(p), "url": self.url(p)}
            for p in anchor.parts]}


@dataclass(frozen=True)
class PrecisionLaunch:
    """One framework-selected treatment of a quantised matmul.

    A scheme says what a checkpoint *allows* its activation operand to be;
    this declaration says what the deployed framework's launch actually does
    with both operands. Entries sharing a scheme/request are ordered. By
    default the launch needs every ``mma_dtypes`` family simultaneously;
    ``promotion_ladder`` is the audited exception on which a declared family
    may be software-promoted to another declared family. In particular
    ``mma_dtypes`` is a hardware-launch fact: software dequantisation can run
    an MXFP4 weight beside a block-FP8 activation on Hopper's FP8/WGMMA path,
    while DeepGEMM's Blackwell path feeds the FP4 MMA. Operand width alone
    cannot choose between them, and neither can a launch name.
    """
    schemes: Tuple[str, ...]
    #: Requested activation formats this entry answers; empty means any.
    requested_activation_formats: Tuple[str, ...]
    #: `None` (input as arrived), `"requested"`, or a named format override.
    activation_format: Optional[str]
    compute_dtype: Optional[str]
    mma_dtypes: Tuple[str, ...]
    #: Whether the audited source says the launch accepts a supported subset
    #: of ``mma_dtypes`` (typically by dequantising one operand in software).
    #: False means every listed family enters the instruction together.
    promotion_ladder: bool = False
    whens: Tuple[DispatchCondition, ...] = ()
    source_anchor: Optional[SourceAnchor] = None
    #: The YAML key, for a selector gap that must name the skipped candidate.
    name: str = ""

    KEYS = frozenset({"schemes", "requested_activation_formats", "launch",
                      "when", "source_anchor"})


@dataclass(frozen=True)
class EdgeDtype:
    """The width the kernel this framework dispatches to REQUIRES a declared
    role's output at.

    The walk prices an edge at the width the checkpoint implies: a BF16 gate
    weight times a BF16 activation writes a BF16 edge. Where the kernel that
    reads that output exists only at a wider one, the framework builds the
    producer to write it, and the wider width is part of what the kernel IS —
    vLLM builds DeepSeek-V4's router `GateLinear(out_dtype=torch.float32)`
    because `can_use_dsv4_topk` requires fp32 logits and `fused_topk_bias`
    falls through to another kernel without them. Without this declaration
    the walk bills a BF16 edge, the cell says `bf16`, and the measurement
    stored under that name is of the fallback kernel, not the one named.

    A REQUIREMENT, not a habit. A framework that merely happens to write a
    role's output wider while the kernel downstream accepts either width
    declares nothing here: we model the better form rather than mirror a
    framework's inefficiency, and the role keeps the width the cell builds.
    That is why the anchor cites two places — where the framework writes the
    width, and where the kernel requires it. One of them alone is not this
    fact.

    This is NOT `traffic.weight_promotion`'s subject. That one covers a
    resident weight held WIDER than its edge, which promotes the dtype the
    matmul COMPUTES in; here the weight and the operand agree and only the
    OUTPUT is wider, so the peak column stays the operand's. It is also not
    a fusion plan's `output_dtype`: a plan's width applies where that plan
    fired, which would put one tensor at two widths depending on whether a
    launch matched. A role's output has one width under a deployment,
    whatever fuses around it.

    Selection is by DECLARED ROLE (`roles.yaml`) plus the facts a binding's
    `when:` already asks — the framework's own dispatch key (the model
    class), its card, its flags. Never a tensor-name pattern and never a
    model id. A role no entry names carries no such requirement and keeps
    the width the checkpoint implies.
    """
    #: The registered roles whose output this entry states the width of.
    roles: Tuple[str, ...]
    #: The canonical dtype the dispatched kernel requires that output at.
    dtype: str
    whens: Tuple[DispatchCondition, ...] = ()
    source_anchor: Optional[SourceAnchor] = None

    KEYS = frozenset({"roles", "dtype", "when", "source_anchor"})


@dataclass(frozen=True)
class WeightLoader:
    """How this framework loads one declared checkpoint scale layout.

    The model spec owns `kind`; the framework owns the transform. Selection
    uses the same deployment and model-class conditions as the other binding
    facts, and the resolved transform crosses into eval as plain data.
    """
    kind: str
    loader: str
    whens: Tuple[DispatchCondition, ...] = ()
    source_anchor: Optional[SourceAnchor] = None

    KEYS = frozenset({"kind", "loader", "when", "source_anchor"})


@dataclass(frozen=True)
class FrameworkBinding:
    """One framework's binding — a frozen, ordered list of fused_op entries
    saying where and when the framework runs each span.

    Stored as a tuple of pairs (not a dict) so the dataclass remains hashable
    / frozen-friendly. An op name can repeat for mutually exclusive
    implementations. Use `implementations_for` to read them all; legacy
    `.get` is only for the one-entry case.

    `deployment` is what the conditional entries are answered against. The
    registry holds every binding undeployed — no flag set, dtype and device
    unknown — and a caller that knows its deployment asks `deployed(...)`.
    """
    framework: str
    audited_at: AuditedAt
    op_implementations: Tuple[Tuple[str, FusionImplementation], ...] = field(
        default_factory=tuple,
    )
    deployment: ServingDeployment = field(default_factory=ServingDeployment)
    #: The paged-KV block size this framework runs by default — the page a
    #: cache manager allocates in, which is a DEPLOYMENT choice and not a
    #: property of the model. A paged attention kernel's time depends on it
    #: (how many pages a query gathers, and how coalesced each gather is),
    #: so it is part of what a measurement measured and part of its key.
    #:
    #: The framework declares it because the framework owns the manager,
    #: and `None` is a DECLARATION too: sglang resolves the page per
    #: attention backend rather than fixing one
    #: (`arg_groups/fields/schedule.py` `page_size … = None`, "resolvable"),
    #: so it has no default to state and a request must say. There is no
    #: code-level number behind either answer. The page is an axis of
    #: every paged kernel's cell (`op_signatures.PAGE_AXIS`), so under a
    #: page nobody declared those kernels have no cell at all, rather than
    #: one keyed on a guessed 16 or the unpaged one.
    #:
    #: Which values a backend can actually serve is the backend's own
    #: constraint (`kernel_bench.providers.attention`, mirrored from each
    #: vLLM backend's `get_supported_kernel_block_sizes()`) and is declined
    #: at the provider.
    kv_block_size: Optional[int] = None
    #: Where this framework reads the chunk length it hands
    #: `mamba_chunk_scan_combined(chunk_size=...)` for a Mamba-2 mixer: the
    #: checkpoint-config fields it consults, in order, the first one present
    #: winning (vLLM `ModelConfig.get_mamba_chunk_size`, sglang's config
    #: classes: `mamba_chunk_size`, then `chunk_size`). The kernel takes the
    #: chunk as an argument, so it is a serving choice rather than a graph
    #: fact: it is not a `dims.yaml` symbol, the spec loader refuses a model
    #: that binds it, and no component's cost reads it. eval reads the value
    #: off the config at these fields (`Program.scan_chunk_size`), and it
    #: keys the `mamba2_scan` cell (`OpSignature.structural_axes`) through
    #: `RowIdentity`. Empty only on a binding built in code without one; a
    #: file must declare it.
    ssm_chunk_size_from: Tuple[str, ...] = ()
    #: The framework's all-reduce dispatch inside one NVLink domain, by
    #: collective op: its rungs in order (`comm_cost.CollectiveRung`; the
    #: module docstring), each with the part of its `when:` the walk answers
    #: (`GraphCondition`: the model class, the framework's own dispatch key)
    #: and the `source_anchor` that shows it. A walk asks for its own
    #: (`collectives_for`).
    collectives: Tuple[Tuple[str, Tuple[Tuple[CollectiveRung, GraphCondition],
                                        ...]], ...] = ()
    collective_anchors: Tuple[SourceAnchor, ...] = ()
    #: Launches this framework selects for a quantised matmul, in file order.
    precision_launches: Tuple[PrecisionLaunch, ...] = ()
    #: Roles whose output the kernel this framework dispatches to requires
    #: at a width the checkpoint does not imply (`EdgeDtype`), in file order.
    #: A walk asks for its own (`edge_dtypes_for`); a role named by none
    #: carries no requirement and keeps its stored width.
    edge_dtypes: Tuple[EdgeDtype, ...] = ()
    #: Checkpoint scale-layout transforms this framework's weight loader runs,
    #: in file order. A walk asks for its own (`weight_loaders_for`).
    weight_loaders: Tuple[WeightLoader, ...] = ()
    #: Audited provider implementations this framework dispatches to, in file
    #: order. Conditions here select AFTER precision/key derivation; they never
    #: restate weight dtype, quant scheme, activation format, or MMA dtype.
    kernel_dispatches: Tuple[KernelDispatch, ...] = ()

    #: The fields of a binding file.
    KEYS = frozenset({"framework", "audited_at", "op_implementations",
                      "kv_block_size", "ssm_chunk_size", "collectives",
                      "precision_launches", "edge_dtypes", "weight_loaders",
                      "kernel_dispatches"})

    def collectives_for(self, program: Any, analysis: Any
                        ) -> Dict[str, Tuple[CollectiveRung, ...]]:
        """The all-reduce dispatch the walk `analysis` of `program` is
        served with: every rung whose condition the walk admits
        (`GraphCondition.admits_program`, the matcher a fusion's `when:`
        is asked through), in order."""
        return {op: tuple(rung for rung, graph in rungs
                          if graph.admits_program(program, analysis))
                for op, rungs in self.collectives}

    def edge_dtypes_for(self, program: Any, analysis: Any
                        ) -> Tuple[Tuple[Tuple[str, ...], str], ...]:
        """`(roles, dtype)` pairs the kernels this deployed framework
        dispatches to on the walk `analysis` of `program` require — the
        output widths they demand, as PLAIN DATA.

        Both halves of each `when:` are answered here, by the same two
        matchers every other binding fact uses: the deployment's
        (`DispatchCondition.holds_for`, so an unknown card claims nothing)
        and the walk's (`GraphCondition.admits_program`, the framework's own
        dispatch key). Those two are also the whole of what a `when:` here
        may ask (`_EDGE_DTYPE_WHEN`), so a declared condition is an answered
        condition. What crosses into `arch/eval` is role names and a dtype
        name, because the walk holds no framework binding (`eval` may not
        import `inference`).
        """
        return tuple(
            (entry.roles, entry.dtype) for entry in self.edge_dtypes
            if not entry.whens
            or any(when.holds_for(self.deployment)
                   and when.graph.admits_program(program, analysis)
                   for when in entry.whens))

    def weight_loaders_for(self, program: Any, analysis: Any
                           ) -> Dict[str, str]:
        """The checkpoint-layout transforms this deployed framework runs.

        Only plain strings cross into eval. An architecture with no declared
        loader has no entry; the checkpoint format and ordinary scale guards
        then remain in force.
        """
        return {
            entry.kind: entry.loader for entry in self.weight_loaders
            if not entry.whens
            or any(when.holds_for(self.deployment)
                   and when.graph.admits_program(program, analysis)
                   for when in entry.whens)
        }

    def implementations_for(self, fused_op: str
                            ) -> Tuple[FusionImplementation, ...]:
        """Every implementation entry for `fused_op`, in binding order."""
        return tuple(impl for op_name, impl in self.op_implementations
                     if op_name == fused_op)

    def get(self, fused_op: str) -> Optional[FusionImplementation]:
        """The sole implementation entry for `fused_op`, if there is one.

        This is the legacy one-implementation accessor. Asking an op with
        multiple entries for “the” entry is ambiguous, so it refuses rather
        than returning file order as a hidden priority.
        """
        impls = self.implementations_for(fused_op)
        if len(impls) > 1:
            raise ValueError(
                f"{fused_op!r} has {len(impls)} implementation entries; read "
                "`implementations_for(fused_op)`")
        return impls[0] if impls else None

    def precision_launches_for(self) -> Tuple[PrecisionLaunch, ...]:
        """The precision launches this deployed framework takes.

        An entry with alternative `when:` conditions is selected when any one
        holds, exactly as a fused-op implementation is. An unknown deployment
        fact makes a condition false: no launch is guessed for a card the
        caller did not name.
        """
        return tuple(launch for launch in self.precision_launches
                     if not launch.whens
                     or any(when.holds_for(self.deployment)
                            for when in launch.whens))

    def unavailable_precision_launches_for(self) -> Tuple[PrecisionLaunch, ...]:
        """Launch candidates this deployment did not satisfy, in file order.

        They remain audit facts, not selectable entries: `ResolvedPrecision`
        records one as a selector gap when its scheme/request matches but a
        required deployment fact (for example an explicit MoE backend) is
        absent.
        """
        return tuple(launch for launch in self.precision_launches
                     if launch.whens and not any(
                         when.holds_for(self.deployment)
                         for when in launch.whens))

    def dispatch_for(self, op: str, program: Any = None,
                     analysis: Any = None,
                     resolved_launch: Optional[ResolvedLaunch] = None
                     ) -> DispatchDecision:
        """Which provider implementation this deployed binding dispatches `op`
        to on the walk described by `program` / `analysis`.

        Exactly one active declaration is a declaration; no entry, no active
        entry, and multiple active entries are explicit non-selections. There
        is no order, recency, speed, corpus row, or base-name fallback.
        `resolved_launch` is the row's precision-resolved identity from
        `point_keys`; a condition that asks a launch fact without one is
        no-active, not a guess from the checkpoint or request dtype.
        """
        entries = tuple(entry for entry in self.kernel_dispatches
                        if entry.op == op)
        if not entries:
            return DispatchDecision(
                DispatchDecision.UNDECLARED, reason="undeclared_kernel_dispatch")
        active = tuple(
            entry for entry in entries
            if not entry.whens or any(
                when.holds_for(self.deployment, resolved_launch)
                and ((program is None
                     and not when.graph.asks_program())
                     or (program is not None
                         and when.graph.admits_program(program, analysis)))
                for when in entry.whens))
        if not active:
            return DispatchDecision(
                DispatchDecision.NO_ACTIVE, reason="no_active_kernel_dispatch")
        if len(active) > 1:
            return DispatchDecision(
                DispatchDecision.AMBIGUOUS, reason="ambiguous_kernel_dispatch")
        entry = active[0]
        return DispatchDecision(
            DispatchDecision.DECLARED, library=entry.library,
            impl_id=entry.impl_id, source_anchor=entry.source_anchor)

    def fused_ops(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(
            op_name for op_name, _impl in self.op_implementations))

    def server_flag_names(self) -> FrozenSet[str]:
        """The server flags this binding's conditions read
        (`DispatchCondition.server_flag`)."""
        return frozenset(when.server_flag[0]
                         for _op, impl in self.op_implementations
                         for when in impl.whens
                         if when.server_flag is not None) | frozenset(
                             when.server_flag[0]
                             for dispatch in self.kernel_dispatches
                             for when in dispatch.whens
                             if when.server_flag is not None) | frozenset(
                                 when.server_flag[0]
                                 for launch in self.precision_launches
                                 for when in launch.whens
                                 if when.server_flag is not None)

    def deployed(self, deployment: ServingDeployment) -> "FrameworkBinding":
        """This binding, answering its conditions against `deployment`. A
        server flag no condition of it reads is refused: the framework has
        no such switch here, and a flag nothing reads would be a deployment
        that differs from another in name only."""
        known = self.server_flag_names()
        unknown = sorted({name for name, _value in deployment.server_flags}
                         - known)
        if unknown:
            raise ValueError(
                f"{self.framework} reads no server flag {unknown}; the flags "
                f"its kernels depend on are {sorted(known)}")
        return replace(self, deployment=deployment)

    def dispatched(self) -> Tuple[
            Tuple[str, Optional[GraphCondition], Optional[str]], ...]:
        """The fused ops this deployment can run, in the binding's order,
        each with the part of its condition the graph answers and the
        boundary its launch spans (`FusionImplementation.crosses`). An entry
        with alternatives is dispatched once per alternative the deployment
        satisfies; the rewrite applies them in turn, and a node one of them
        absorbed is not matched again."""
        out = []
        for op_name, impl in self.op_implementations:
            if not impl.whens:
                out.append((op_name, None, impl.crosses))
            out.extend((op_name, when.graph, impl.crosses)
                       for when in impl.whens
                       if when.holds_for(self.deployment))
        return tuple(out)


def _parse_whens(raw: Any, where: str) -> Tuple[DispatchCondition, ...]:
    """`when:` as its alternatives: absent, one mapping, or a list of two or
    more mappings (a one-item list is one condition, written as such)."""
    if raw is None:
        return ()
    if isinstance(raw, list):
        if len(raw) < 2:
            raise ValueError(
                f"{where}: a `when:` list states alternatives — two or more; "
                f"one condition is written as a mapping")
        return tuple(DispatchCondition.parse(item, f"{where}: when[{i}]")
                     for i, item in enumerate(raw))
    return (DispatchCondition.parse(raw, where),)


def _domains_overlap(left: Tuple[Any, ...], right: Tuple[Any, ...]) -> bool:
    """Whether two closed `when` domains can name the same fact.

    An empty domain asks nothing and therefore overlaps every value. Matching
    is case-insensitive only for name domains; integer domains compare as
    integers.
    """
    if not left or not right:
        return True
    return bool({str(value).lower() for value in left}
                & {str(value).lower() for value in right})


def _capabilities_overlap(left: DispatchCondition,
                          right: DispatchCondition) -> bool:
    """Whether the two inclusive-lower/open-upper capability windows intersect."""
    lower = [capability for capability in (
        left.min_compute_capability, right.min_compute_capability)
        if capability is not None]
    upper = [capability for capability in (
        left.below_compute_capability,
        right.below_compute_capability) if capability is not None]
    return (not lower or not upper or max(lower) < min(upper))


def _architectures_overlap(left: DispatchCondition,
                           right: DispatchCondition) -> bool:
    """Whether one model can satisfy both architecture clauses.

    `excluded_architectures` is a negative domain: two exclusions always have
    an outside model in common, while a positive domain is disjoint only when
    every one of its names is excluded.
    """
    left_positive = set(left.graph.architectures)
    right_positive = set(right.graph.architectures)
    left_excluded = set(left.graph.excluded_architectures)
    right_excluded = set(right.graph.excluded_architectures)
    if left_positive and right_positive:
        return bool((left_positive - left_excluded)
                    & (right_positive - right_excluded))
    if left_positive:
        return bool(left_positive - left_excluded - right_excluded)
    if right_positive:
        return bool(right_positive - right_excluded - left_excluded)
    return True


def _conditions_can_overlap(left: DispatchCondition,
                            right: DispatchCondition) -> bool:
    """A conservative proof over every condition key.

    This is not a graph matcher. It answers only whether a single deployment
    and a single match could satisfy both conjunctions. Facts on different
    keys constrain different questions and therefore overlap; two values on
    one key are disjoint only when the closed vocabulary lets us prove no
    value satisfies both.
    """
    if left.server_flag is not None and right.server_flag is not None:
        if left.server_flag[0] != right.server_flag[0]:
            return True
        if left.server_flag_value != right.server_flag_value:
            return False
    for left_value, right_value in (
            (left.activation_dtype, right.activation_dtype),
            (left.kv_block_size, right.kv_block_size),
            (left.tensor_parallel, right.tensor_parallel),
            (left.device, right.device),
            (left.graph.phase, right.graph.phase),
            (left.graph.head_dim, right.graph.head_dim)):
        if not _domains_overlap(left_value, right_value):
            return False
    if not _capabilities_overlap(left, right):
        return False
    if not _architectures_overlap(left, right):
        return False
    if (left.graph.rotation_covers_head is not None
            and right.graph.rotation_covers_head is not None
            and left.graph.rotation_covers_head
            != right.graph.rotation_covers_head):
        return False
    if (left.graph.readers_eat is not None
            and right.graph.readers_eat is not None
            and left.graph.readers_eat != right.graph.readers_eat):
        return False
    if (left.graph.routed is not None and right.graph.routed is not None
            and left.graph.routed != right.graph.routed):
        return False
    if (left.unquantized_bf16_launch is not None
            and right.unquantized_bf16_launch is not None
            and left.unquantized_bf16_launch
            != right.unquantized_bf16_launch):
        return False
    left_roles = dict(left.graph.roles)
    for node, allowed in right.graph.roles:
        if node in left_roles and not _domains_overlap(
                left_roles[node], allowed):
            return False
    left_dims = dict(left.graph.dims)
    for name, values in right.graph.dims:
        if name in left_dims and not _domains_overlap(left_dims[name], values):
            return False
    if left.graph.world_local is not None and right.graph.world_local is not None:
        left_world, left_dim, left_local = left.graph.world_local
        right_world, right_dim, right_local = right.graph.world_local
        if (left_world == right_world and left_dim == right_dim
                and left_local and right_local
                and not _domains_overlap(left_local, right_local)):
            return False
    return True


def _reject_overlapping_implementations(fname: str,
                                        impls: Tuple[Tuple[
                                            str, FusionImplementation], ...]
                                        ) -> None:
    """Refuse two entries for one fused op whose conditions can both hold.

    Alternatives inside one entry are ORs and may overlap. Entries are
    alternatives at a higher level: without a proof of disjointness, rewrite
    order would silently decide which span wins. Unknown graph facts are
    treated as possibly shared (conservative), never as disjoint.
    """
    by_op: Dict[str, list] = {}
    for index, (op, impl) in enumerate(impls):
        by_op.setdefault(op, []).append((index, impl))
    for op, entries in by_op.items():
        for at, (left_index, left) in enumerate(entries):
            for right_index, right in entries[at + 1:]:
                if any(_conditions_can_overlap(left_when, right_when)
                       for left_when in (left.whens or (DispatchCondition(),))
                       for right_when in (right.whens or (DispatchCondition(),))):
                    raise ValueError(
                        f"framework_bindings/{fname}: op_implementations"
                        f"[{op!r}] entries {left_index} and {right_index} can "
                        "be active together; state provably disjoint `when` "
                        "conditions (or one entry with `when:` alternatives).")


#: What a `kernel_dispatches.when:` may ask: the card's, the deployment's and
#: the precision-resolved launch's keys, and the facts the whole walk answers
#: (`GraphCondition.PROGRAM_FIELDS`) — derived from those declarations, not
#: listed again. Everything else is a fusion-match fact, not something a
#: primitive row selector can answer.
_KERNEL_DISPATCH_WHEN = (frozenset(DispatchCondition.CARD_KEYS)
                         | DispatchCondition.DEPLOYMENT_KEYS
                         | DispatchCondition.LAUNCH_KEYS
                         | frozenset(GraphCondition.PROGRAM_FIELDS))


def _kernel_dispatches(raw: Any, fname: str,
                       known_fused_ops: set) -> Tuple[KernelDispatch, ...]:
    """Load a binding's audited provider dispatches, checked at load time."""
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"framework_bindings/{fname}: `kernel_dispatches` is an ordered "
            f"list, got {type(raw).__name__}.")
    known_ops = set(op_bindings()) | known_fused_ops
    out: list = []
    seen: set = set()
    for i, item in enumerate(raw):
        where = f"framework_bindings/{fname}: kernel_dispatches[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} is a mapping.")
        unknown = sorted(set(item) - KernelDispatch.KEYS)
        if unknown:
            raise ValueError(
                f"{where} has unknown fields {unknown}; an entry is "
                f"{sorted(KernelDispatch.KEYS)}.")
        op = item.get("op")
        library = item.get("library")
        impl_id = item.get("impl_id")
        if not isinstance(op, str) or not op:
            raise ValueError(f"{where} names the engine op in `op`.")
        if op not in known_ops:
            raise ValueError(
                f"{where}: op {op!r} has neither an OpSignature "
                f"(`op_bindings()`) nor a fused-op span.")
        if not isinstance(library, str) or not library:
            raise ValueError(
                f"{where}: `library` names the PROVIDER library whose row is "
                "selected; the serving framework is already this file.")
        if not isinstance(impl_id, str) or not impl_id:
            raise ValueError(
                f"{where}: `impl_id` is the provider's complete resolved "
                "implementation id (including any `@...` suffix).")
        if "when" not in item:
            raise ValueError(
                f"{where}: state the audited `when`; dispatch has no implicit "
                "condition.")
        raw_whens = (item["when"] if isinstance(item["when"], list)
                     else [item["when"]])
        for raw_when in raw_whens:
            if not isinstance(raw_when, dict):
                break
            match_only = sorted(set(raw_when) - _KERNEL_DISPATCH_WHEN)
            if match_only:
                raise ValueError(
                    f"{where}: kernel dispatches answer only deployment, "
                    f"architecture, and phase facts; {match_only} need a "
                    "fusion match (or belong to precision_launches).")
        whens = _parse_whens(item.get("when"), where)
        for when in whens:
            if set(when.as_dict()) - _KERNEL_DISPATCH_WHEN:
                # `as_dict` omits empty fields, so anything left here is a
                # match-bound graph key rather than a row/deployment fact.
                forbidden = sorted(set(when.as_dict()) - _KERNEL_DISPATCH_WHEN)
                raise ValueError(
                    f"{where}: kernel dispatches answer only deployment, "
                    f"architecture, and phase facts; {forbidden} need a fusion "
                    "match (or belong to precision_launches).")
        anchor_raw = item.get("source_anchor")
        if not isinstance(anchor_raw, str) or not anchor_raw:
            raise ValueError(
                f"{where}: needs the audited `source_anchor` that shows THIS "
                "implementation being dispatched.")
        entry = KernelDispatch(
            op=op, library=library, impl_id=impl_id,
            source_anchor=SourceAnchor.parse(anchor_raw, where), whens=whens)
        identity = (entry.op, entry.library, entry.impl_id, entry.whens)
        if identity in seen:
            raise ValueError(f"{where} repeats an earlier dispatch exactly.")
        seen.add(identity)
        out.append(entry)
    return tuple(out)


def _audited_at(raw: Any, fname: str) -> AuditedAt:
    if (not isinstance(raw, dict) or set(raw) != {"repo", "tag", "commit"}
            or not all(isinstance(raw[k], str) and raw[k] for k in raw)
            or raw["repo"].count("/") != 1):
        raise ValueError(
            f"framework_bindings/{fname}: `audited_at` is {{ repo: "
            f"<owner/name on GitHub>, tag: <release tag>, commit: <sha> }}, "
            f"the release the facts were read at.")
    return AuditedAt(tag=raw["tag"], commit=raw["commit"], repo=raw["repo"])


#: The keys of one `collectives:` rung.
_RUNG_KEYS = frozenset({"algorithm", "max_bytes", "max_bytes_times_world",
                        "worlds", "when", "source_anchor"})
#: What a rung's `when:` may ask: the card's compute capability, and the
#: model class the framework dispatches on.
_RUNG_WHEN = frozenset({"min_compute_capability", "below_compute_capability",
                        "architectures"})


def _collectives(raw: Any, fname: str) -> Tuple[
        Tuple[Tuple[str, Tuple[Tuple[CollectiveRung, GraphCondition], ...]], ...],
        Tuple[SourceAnchor, ...]]:
    """A binding's `collectives:` (the module docstring), checked: an
    all-reduce's rungs, each a known algorithm other than NCCL's own (which
    runs past them all), a positive cap, and the anchor that shows it."""
    where = f"framework_bindings/{fname}: collectives"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} maps a collective op to its rungs")
    out, anchors = [], []
    for op, rungs in raw.items():
        if op != "all_reduce":
            raise ValueError(
                f"{where}[{op!r}]: only `all_reduce` has algorithms of its "
                f"own ({sorted(set(ALGORITHMS) - {NCCL})}); every other "
                "collective runs NCCL's schedule")
        if not isinstance(rungs, list) or not rungs:
            raise ValueError(f"{where}[{op!r}] is a non-empty list of rungs")
        parsed = []
        for i, rung in enumerate(rungs):
            at = f"{where}[{op!r}][{i}]"
            if not isinstance(rung, dict) or set(rung) - _RUNG_KEYS:
                raise ValueError(f"{at}: a rung is {sorted(_RUNG_KEYS)}")
            algorithm = rung.get("algorithm")
            if algorithm not in ALGORITHMS or algorithm == NCCL:
                raise ValueError(
                    f"{at}: `algorithm: {algorithm!r}` is not one of "
                    f"{sorted(set(ALGORITHMS) - {NCCL})}")
            caps = {}
            worlds = rung.get("worlds")
            if (not isinstance(worlds, list) or not worlds
                    or not all(isinstance(w, int) and not isinstance(w, bool)
                               and w > 1 for w in worlds)):
                raise ValueError(
                    f"{at}: `worlds` lists the group sizes the framework's "
                    "kernel accepts (integers > 1); a group of any other size "
                    "runs NCCL")
            for key in ("max_bytes", "max_bytes_times_world"):
                v = rung.get(key)
                if v is not None and (not isinstance(v, int)
                                      or isinstance(v, bool) or v <= 0):
                    raise ValueError(f"{at}: `{key}` is a positive integer")
                caps[key] = v
            if not isinstance(rung.get("source_anchor"), str):
                raise ValueError(f"{at}: needs the `source_anchor` that shows it")
            anchors.append(SourceAnchor.parse(rung["source_anchor"], at))
            when = (DispatchCondition.parse(rung["when"], at, _RUNG_WHEN)
                    if rung.get("when") is not None else None)
            cc = lambda t: None if t is None else "%d.%d" % t
            parsed.append((CollectiveRung(
                algorithm, caps["max_bytes"], caps["max_bytes_times_world"],
                cc(when.min_compute_capability) if when else None,
                cc(when.below_compute_capability) if when else None,
                tuple(sorted(set(worlds)))),
                when.graph if when else GraphCondition()))
        out.append((op, tuple(parsed)))
    return tuple(out), tuple(anchors)


def _precision_launches(raw: Any, fname: str) -> Tuple[PrecisionLaunch, ...]:
    """Parse `precision_launches:`, the framework's quantised-matmul dispatch."""
    if not isinstance(raw, dict):
        raise ValueError(f"framework_bindings/{fname}: precision_launches "
                         "must be a mapping")
    out = []
    for name, entry in raw.items():
        where = f"framework_bindings/{fname}: precision_launches[{name!r}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a mapping")
        unknown = sorted(set(entry) - PrecisionLaunch.KEYS)
        if unknown:
            raise ValueError(f"{where} has unknown fields {unknown}")
        schemes = entry.get("schemes")
        if (not isinstance(schemes, list) or not schemes
                or not all(isinstance(s, str) for s in schemes)
                or any(s not in QUANT_SCHEMES for s in schemes)):
            raise ValueError(f"{where}: `schemes` names quant schemes "
                             f"{sorted(QUANT_SCHEMES)}")
        requested = entry.get("requested_activation_formats") or []
        if (not isinstance(requested, list)
                or any(f not in ACTIVATION_FORMATS for f in requested)):
            raise ValueError(
                f"{where}: `requested_activation_formats` names activation "
                f"formats {sorted(ACTIVATION_FORMATS)} (or is omitted for any)")
        launch = entry.get("launch")
        if (not isinstance(launch, dict)
                or not {"activation", "compute_dtype",
                        "mma_dtypes"} <= set(launch)
                or not set(launch) <= {"activation", "compute_dtype",
                                       "mma_dtypes", "promotion_ladder"}):
            raise ValueError(f"{where}: `launch` is {{activation, "
                             "compute_dtype, mma_dtypes[, promotion_ladder]}")
        activation = launch["activation"]
        if activation == "none":
            activation = None
        elif activation != "requested" and activation not in ACTIVATION_FORMATS:
            raise ValueError(f"{where}: launch activation {activation!r} is "
                             "`none`, `requested`, or one of "
                             f"{sorted(ACTIVATION_FORMATS)}")
        try:
            compute = (None if launch["compute_dtype"] is None
                       else canonical(launch["compute_dtype"]))
            families = tuple(canonical(dt) for dt in launch["mma_dtypes"])
        except UnknownDtype as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if not families or (activation is None and compute is None):
            raise ValueError(
                f"{where}: a launch names its MMA families, and a "
                "weight-only launch names the dtype it computes in")
        promotion_ladder = launch.get("promotion_ladder", False)
        if not isinstance(promotion_ladder, bool):
            raise ValueError(
                f"{where}: promotion_ladder is the audited boolean "
                "`true` or `false`")
        anchor = entry.get("source_anchor")
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(f"{where}: needs the `source_anchor` that shows "
                             "the framework selects this launch")
        out.append(PrecisionLaunch(
            name=str(name),
            schemes=tuple(schemes),
            requested_activation_formats=tuple(requested),
            activation_format=activation,
            compute_dtype=compute,
            mma_dtypes=families,
            promotion_ladder=promotion_ladder,
            whens=_parse_whens(entry.get("when"), where),
            source_anchor=SourceAnchor.parse(anchor, where),
        ))
    return tuple(out)


#: What an `edge_dtypes.when:` may ask: exactly the questions
#: `FrameworkBinding.edge_dtypes_for` puts — the deployment's
#: (`DispatchCondition.holds_for`) and the walk's
#: (`GraphCondition.admits_program`). That is the kernel-dispatch vocabulary
#: (same granularity, same two matchers) minus the two keys resolved AFTER a
#: width: the launch's own, and the activation dtype read off a resolved
#: launch. DERIVED from that set rather than listed again, so a key added
#: there is answerable here too or is excluded here on purpose — and so the
#: overlap proof below runs on the same keys the evaluator reads, which is
#: what makes "one role, one width" a proof rather than a wish.
_EDGE_DTYPE_WHEN = _KERNEL_DISPATCH_WHEN - {"activation_dtype",
                                            "unquantized_bf16_launch"}


def _edge_dtypes(raw: Any, fname: str) -> Tuple[EdgeDtype, ...]:
    """Parse `edge_dtypes:`, the widths the kernels this framework
    dispatches to require a declared role's output at (`EdgeDtype`)."""
    if not isinstance(raw, dict):
        raise ValueError(f"framework_bindings/{fname}: edge_dtypes must be "
                         "a mapping")
    out = []
    for name, entry in raw.items():
        where = f"framework_bindings/{fname}: edge_dtypes[{name!r}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a mapping")
        unknown = sorted(set(entry) - EdgeDtype.KEYS)
        if unknown:
            raise ValueError(f"{where} has unknown fields {unknown}")
        roles = entry.get("roles")
        if (not isinstance(roles, list) or not roles
                or not all(isinstance(r, str) for r in roles)):
            raise ValueError(f"{where}: `roles` is a non-empty list of the "
                             "roles whose output this width states")
        # The REGISTERED roles, which is narrower than the role coordinate a
        # precision profile writes (`precision_profile` adds `outside_roles`
        # to the same vocabulary). Narrower on purpose: a width is
        # materialized after the walk, while the edges of no particular role
        # are billed during it (`default_edge_bytes`), so a declaration on
        # `outside_roles` could not be honoured where it is read. Widening
        # this vocabulary means carrying the fact into the walk first.
        unregistered = [r for r in roles
                        if not roles_registry.is_registered(r)]
        if unregistered:
            raise ValueError(f"{where}: `roles` {unregistered} are not roles "
                             "of roles.yaml")
        try:
            dtype = canonical(entry.get("dtype"))
        except (UnknownDtype, TypeError) as exc:
            raise ValueError(f"{where}: `dtype` names the width the kernel "
                             f"downstream requires this output at ({exc})"
                             ) from exc
        raw_when = entry.get("when")
        whens = _parse_whens(raw_when, where)
        # Every key this entry asks, read twice over: as the file spells it,
        # and as the parsed condition states it (`as_dict` omits what is
        # empty, so what remains is a key the full dispatch vocabulary
        # accepts but `edge_dtypes_for` puts to neither matcher).
        asked: set = set()
        for raw_alt in (raw_when if isinstance(raw_when, list) else [raw_when]):
            if isinstance(raw_alt, dict):
                asked |= set(raw_alt)
        for when in whens:
            asked |= set(when.as_dict())
        unanswered = sorted(asked - _EDGE_DTYPE_WHEN)
        if unanswered:
            raise ValueError(
                f"{where}: `when` may ask only what resolving a width "
                f"answers — the deployment's facts and the walk's model "
                f"class and phase; {unanswered} would be read by nobody "
                "(a match-bound graph fact, or a launch fact resolved after "
                "this width) and would hold for every deployment.")
        anchor = entry.get("source_anchor")
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(f"{where}: needs the `source_anchor` that shows "
                             "the framework writing this width AND the kernel "
                             "that requires it")
        parsed = SourceAnchor.parse(anchor, where)
        if sum(isinstance(p, SourceLocation) for p in parsed.parts) < 2:
            raise ValueError(
                f"{where}: `source_anchor` cites one place. A width is "
                "declared because a KERNEL requires it, so cite both: where "
                "the framework writes it, and where the kernel takes only "
                "that width. A framework writing wider than its kernel needs "
                "is an inefficiency we do not follow.")
        out.append(EdgeDtype(
            roles=tuple(roles),
            dtype=dtype,
            whens=whens,
            source_anchor=parsed,
        ))
    # Two entries that can both answer one role would make the width depend
    # on file order. A role is stated once per reachable deployment. The
    # proof is sound because `_EDGE_DTYPE_WHEN` has already confined the
    # conditions to the keys `edge_dtypes_for` actually evaluates.
    for i, left in enumerate(out):
        for right in out[i + 1:]:
            shared = sorted(set(left.roles) & set(right.roles))
            if shared and _when_sets_can_overlap(left.whens, right.whens):
                raise ValueError(
                    f"framework_bindings/{fname}: edge_dtypes state roles "
                    f"{shared} twice under conditions that can both hold; "
                    "one role has one required width per deployment")
    return tuple(out)


def _weight_loaders(raw: Any, fname: str) -> Tuple[WeightLoader, ...]:
    """Parse the framework's audited checkpoint-layout load transforms."""
    if not isinstance(raw, dict):
        raise ValueError(f"framework_bindings/{fname}: weight_loaders must be "
                         "a mapping")
    out = []
    for name, entry in raw.items():
        where = f"framework_bindings/{fname}: weight_loaders[{name!r}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be a mapping")
        unknown = sorted(set(entry) - WeightLoader.KEYS)
        if unknown:
            raise ValueError(f"{where} has unknown fields {unknown}")
        kind = entry.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{where}: `kind` names a checkpoint layout")
        loader = entry.get("loader")
        if loader not in LOADER_TRANSFORMS:
            raise ValueError(
                f"{where}: `loader` must be one of {sorted(LOADER_TRANSFORMS)}")
        raw_when = entry.get("when")
        whens = _parse_whens(raw_when, where)
        asked: set = set()
        for raw_alt in (raw_when if isinstance(raw_when, list) else [raw_when]):
            if isinstance(raw_alt, dict):
                asked |= set(raw_alt)
        for when in whens:
            asked |= set(when.as_dict())
        unanswered = sorted(asked - _EDGE_DTYPE_WHEN)
        if unanswered:
            raise ValueError(
                f"{where}: `when` asks facts loader selection does not answer: "
                f"{unanswered}")
        anchor = entry.get("source_anchor")
        if not isinstance(anchor, str) or not anchor:
            raise ValueError(f"{where}: needs the `source_anchor` that shows "
                             "the framework's loader behavior")
        out.append(WeightLoader(
            kind=kind, loader=loader, whens=whens,
            source_anchor=SourceAnchor.parse(anchor, where)))
    for i, left in enumerate(out):
        for right in out[i + 1:]:
            if (left.kind == right.kind
                    and _when_sets_can_overlap(left.whens, right.whens)):
                raise ValueError(
                    f"framework_bindings/{fname}: weight_loaders state "
                    f"{left.kind!r} twice under conditions that can both hold")
    return tuple(out)


def _when_sets_can_overlap(left: Tuple[DispatchCondition, ...],
                           right: Tuple[DispatchCondition, ...]) -> bool:
    """Whether two `when:` alternative SETS can both hold at once, over the
    same conservative proof one pair takes (`_conditions_can_overlap`). An
    empty set asks nothing and so overlaps everything."""
    if not left or not right:
        return True
    return any(_conditions_can_overlap(a, b) for a in left for b in right)


def _load_framework_bindings_dir(
    dir_path: str,
    known_fused_ops: Optional[set] = None,
) -> Dict[str, FrameworkBinding]:
    """Load every `<framework>.yaml` under `dir_path` into a framework ->
    binding map."""
    if known_fused_ops is None:
        known_fused_ops = set(FUSION_SCHEMES.keys())
    out: Dict[str, FrameworkBinding] = {}
    if not os.path.isdir(dir_path):
        return out
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(dir_path, fname)
        try:
            with open(path) as f:
                data = load_yaml(f) or {}
        except _yaml.YAMLError as e:
            raise ValueError(
                f"framework_bindings/{fname}: YAML parse error: {e}"
            ) from e
        if not isinstance(data, dict):
            raise ValueError(
                f"framework_bindings/{fname}: top-level must be a mapping, "
                f"got {type(data).__name__}."
            )
        unknown_top = sorted(set(data) - FrameworkBinding.KEYS)
        if unknown_top:
            raise ValueError(
                f"framework_bindings/{fname}: unknown fields {unknown_top}; a "
                f"binding is {sorted(FrameworkBinding.KEYS)} (one per framework, no "
                f"version: `audited_at` names the release it was read at).")
        framework = data.get("framework")
        if not framework or not isinstance(framework, str):
            raise ValueError(
                f"framework_bindings/{fname}: missing or non-string "
                f"`framework` field."
            )
        if fname != f"{framework}.yaml":
            raise ValueError(
                f"framework_bindings/{fname}: the file of framework "
                f"{framework!r} is `{framework}.yaml` — one binding per "
                f"framework.")
        audited_at = _audited_at(data.get("audited_at"), fname)
        if "kv_block_size" not in data:
            raise ValueError(
                f"framework_bindings/{fname}: no `kv_block_size`. It is the "
                f"paged-KV page this framework allocates the cache in, and it "
                f"is part of the key of every kernel that reads that cache, so "
                f"it is declared, not defaulted: a positive integer, or an "
                f"explicit `null` for a framework that resolves the page per "
                f"attention backend (sglang) and leaves it to the request.")
        kv_block = data["kv_block_size"]
        if kv_block is not None and (
                not isinstance(kv_block, int) or isinstance(kv_block, bool)
                or kv_block <= 0):
            raise ValueError(
                f"framework_bindings/{fname}: `kv_block_size` is a positive "
                f"integer or `null`, got {kv_block!r}.")
        ssm_chunk = data.get("ssm_chunk_size")
        ssm_from = (ssm_chunk.get("from_config")
                    if isinstance(ssm_chunk, dict)
                    and set(ssm_chunk) == {"from_config"} else None)
        if (not isinstance(ssm_from, list) or not ssm_from
                or not all(isinstance(f, str) and f for f in ssm_from)
                or len(set(ssm_from)) != len(ssm_from)):
            raise ValueError(
                f"framework_bindings/{fname}: `ssm_chunk_size` is "
                f"`{{from_config: [<field>, ...]}}`, the checkpoint-config "
                f"fields this framework reads the chunk length it hands "
                f"`mamba_chunk_scan_combined(chunk_size=...)` from, first "
                f"present wins. It is an axis of the `mamba2_scan` "
                f"measurement cell, so it is declared, not defaulted; got "
                f"{ssm_chunk!r}.")
        op_impls_raw = data.get("op_implementations") or {}
        if not isinstance(op_impls_raw, dict):
            raise ValueError(
                f"framework_bindings/{fname}: `op_implementations` must "
                f"be a mapping, got {type(op_impls_raw).__name__}."
            )
        # Track original yaml declaration position so it survives as
        # the tiebreak among schemes of equal specificity (so authors
        # can still influence order within one bucket).
        impls: list = []
        yaml_position: Dict[int, int] = {}
        impl_index = 0
        for op_name, raw_impls in op_impls_raw.items():
            if op_name not in known_fused_ops:
                raise ValueError(
                    f"framework_bindings/{fname}: op_implementations"
                    f"[{op_name!r}] references a fused_op not defined "
                    f"in backend/inference/fusion_plans/. "
                    f"Known fused_ops: {sorted(known_fused_ops)!r}."
                )
            # One mapping is the original spelling. A non-empty list is the
            # multi-implementation spelling; YAML mapping keys cannot repeat.
            if isinstance(raw_impls, list):
                if not raw_impls:
                    raise ValueError(
                        f"framework_bindings/{fname}: op_implementations"
                        f"[{op_name!r}] has an empty implementation list.")
                impl_datas = raw_impls
            else:
                impl_datas = [raw_impls]
            for impl_number, impl_data in enumerate(impl_datas):
                where = (f"framework_bindings/{fname}: "
                         f"op_implementations[{op_name!r}][{impl_number}]")
                _append_op_implementation(impls, op_name, impl_data, where)
                yaml_position[id(impls[-1])] = impl_index
                impl_index += 1
        _reject_overlapping_implementations(fname, tuple(impls))
        # Sort by specificity (descending) so specific patterns try
        # first; yaml declaration order is the tiebreak inside one
        # specificity bucket so author intent within ties is preserved.
        _reject_shared_patterns(fname, list(dict.fromkeys(
            op for op, _impl in impls)))
        impls.sort(key=lambda entry: (
            -_pattern_specificity(entry[0]),
            yaml_position[id(entry)],
        ))
        collectives, anchors = _collectives(data.get("collectives") or {}, fname)
        precision_launches = _precision_launches(
            data.get("precision_launches") or {}, fname)
        edge_dtypes = _edge_dtypes(data.get("edge_dtypes") or {}, fname)
        weight_loaders = _weight_loaders(
            data.get("weight_loaders") or {}, fname)
        kernel_dispatches = _kernel_dispatches(
            data.get("kernel_dispatches"), fname, known_fused_ops)
        out[framework] = FrameworkBinding(
            framework=framework,
            audited_at=audited_at,
            op_implementations=tuple(impls),
            kv_block_size=(None if kv_block is None else int(kv_block)),
            ssm_chunk_size_from=tuple(ssm_from),
            collectives=collectives,
            collective_anchors=anchors,
            precision_launches=precision_launches,
            edge_dtypes=edge_dtypes,
            weight_loaders=weight_loaders,
            kernel_dispatches=kernel_dispatches,
        )
    return out


def _append_op_implementation(impls: list, op_name: str, impl_data: Any,
                              where: str) -> None:
    """Parse and append one `op_implementations` mapping."""
    if not isinstance(impl_data, dict):
        raise ValueError(
            f"{where} must be a mapping with `source_anchor` and optional "
            "`when`.")
    unknown_fields = sorted(set(impl_data) - FusionImplementation.KEYS)
    if unknown_fields:
        raise ValueError(
            f"{where} has unknown fields {unknown_fields}; an entry is "
            "`source_anchor` and `when` (which kernel runs is a measured "
            "row's identity, not the binding's).")
    if (not isinstance(impl_data.get("source_anchor"), str)
            or not impl_data["source_anchor"]):
        raise ValueError(
            f"{where} needs the `source_anchor` that shows the framework "
            "runs it.")
    crosses = impl_data.get("crosses")
    if crosses is not None and crosses not in CROSSES:
        raise ValueError(
            f"{where}: `crosses: {crosses!r}` is not a boundary; declared: "
            f"{list(CROSSES)}")
    impls.append((op_name, FusionImplementation(
        source_anchor=SourceAnchor.parse(impl_data["source_anchor"], where),
        whens=_parse_whens(impl_data.get("when"), where),
        crosses=crosses,
    )))


_FRAMEWORK_BINDINGS_DIR = os.path.dirname(__file__)
FRAMEWORK_BINDINGS: Dict[str, FrameworkBinding] = (
    _load_framework_bindings_dir(_FRAMEWORK_BINDINGS_DIR)
)


#: The framework an estimate names when no framework binding is to decide
#: it: the engine's own analysis, with no fusion rewrite. Validate's analysis replay runs under it.
THEORY = "theory"

#: The framework a request that names none is answered under. Every default
#: in the API, the bench CLI and the harness reads it.
DEFAULT_FRAMEWORK = "vllm"


def get_binding(framework: str) -> Optional[FrameworkBinding]:
    """The framework's binding, or None when none ships for it."""
    return FRAMEWORK_BINDINGS.get(framework)


def reload_framework_bindings(
    dir_path: Optional[str] = None,
) -> Dict[str, FrameworkBinding]:
    """Reload the registry from `dir_path` (default: this module's directory).

    Test helper — production code reads the module-level
    `FRAMEWORK_BINDINGS` populated at import time.
    """
    global FRAMEWORK_BINDINGS
    FRAMEWORK_BINDINGS = _load_framework_bindings_dir(
        dir_path or _FRAMEWORK_BINDINGS_DIR
    )
    return FRAMEWORK_BINDINGS
