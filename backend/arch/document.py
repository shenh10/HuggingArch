"""The typed authoring schema for one model spec YAML.

`ArchSpec` is what the loader PRODUCES: synthesized block graphs, the
transitive component closure, the role map. `SpecDocument` is what a person
WRITES: which components fill which slots, how dims bind to config, what the
layer stack looks like. The two used to be the same pydantic model, which
meant the fields an author actually types had no schema of their own — they
were read back out of a raw dict with `raw.get(...)` at assembly time, and a
key nobody recognised was simply not read.

That is the failure this file exists to make impossible. Misspell `dims:` as
`dim:` and the old path produced a spec with no dimension bindings and no
complaint; write `strides: 4` in a `per_layer` selector and the generator
matched every layer, because the expander copied a fixed list of selector
names and dropped the rest. `extra="forbid"` on every model here turns both
into an error that names the field.

The document is the input to exactly one lowering (`loader.lower`). Nothing
downstream reads it: consumers see `ArchSpec`, and an author never sees the
assembly product's fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from backend.arch.schema import (
    StreamLayerSet,
    BlockGraph,
    CheckpointAddress,
    CustomOpSpec,
    ModelInputSpec,
    ModelLayer,
    QuantContext,
    ValidationExemptions,
)


class PerLayerAssign(BaseModel):
    """One layer-selecting generator, tagged with the type it produces.

    A generator says WHICH layers; the `map` entry for its `type` says what
    those layers are built from. One or more selectors may be given
    (`range` / `layers` / `stride`+`offset` / `from`+`is` / `from` alone);
    when several appear they are INTERSECTED — the generator selects the
    layers satisfying all of them (e.g. `from: config.compress_ratios, is: "128"` AND
    `range: [0, "n_hash_layers - 1"]` picks HCA layers that are also hash-MoE).
    `rest: true` is the exclusive catch-all and cannot be combined.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    #: Key into the sibling `map`. Free-form — the names are the model's own
    #: vocabulary (`swa`, `full`, `mtp`), not a closed set.
    type: str

    # --- selectors -------------------------------------------------------
    # Indices are dim EXPRESSIONS, not integers: `range: [0, "L-1"]` has to
    # say "up to the last layer" without knowing how many there are. The
    # types mirror `BlockAssignment`'s, which is where they land.
    range: Optional[List[str]] = None
    layers: Optional[List[str]] = None
    stride: Optional[str] = None
    offset: Optional[str] = None
    rest: Optional[bool] = None
    #: `from: config.layer_types` — read the per-layer type list off the
    #: checkpoint config instead of describing the pattern arithmetically.
    from_: Optional[str] = Field(default=None, alias="from")
    #: The value in that list this generator selects on. Without it, the
    #: list `from` names holds the layer numbers themselves
    #: (`fullatt_block_indexes`).
    is_: Optional[str] = Field(default=None, alias="is")
    #: The number that list gives the first layer: 1 for a config that counts
    #: layers from 1 (Qwen3.8's `ple_layer_ids`). Only with `from` alone.
    base: Optional[int] = None
    #: A second config-listed set the selected layers must also lie in
    #: (`{from: config.<list>, is: <value>}` or `{from: config.<list>}` of
    #: layer numbers, optionally `base`): a layer kind that is the product of
    #: two config lists, Qwen3.8's `layer_types` × `ple_layer_ids`.
    within: Optional[StreamLayerSet] = None
    #: In a looped stack: only the executions of the selected stored layers
    #: in this pass (`BlockAssignment.pass_`). Alone, it selects every stored
    #: layer of the loop in that pass.
    pass_: Optional[int] = Field(default=None, alias="pass")

    # --- per-selection overrides, layered over the `map` entry -----------
    dims: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    #: Where the selected layers read the inner streams their components
    #: read (`BlockAssignment.reads_from`).
    reads_from: Optional[Dict[str, Any]] = None

    @field_validator("type")
    @classmethod
    def _valid_type(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError(
                "per_layer assign type must be a non-empty, trimmed map key")
        return value

    @field_validator("range", "layers")
    @classmethod
    def _nonempty_selector_expressions(
        cls, values: Optional[List[str]],
    ) -> Optional[List[str]]:
        if values is not None and any(not value.strip() for value in values):
            raise ValueError("per_layer selector expressions must be non-empty")
        return values

    @field_validator("stride", "offset", "from_", "is_")
    @classmethod
    def _nonempty_scalar_selector(
        cls, value: Optional[str],
    ) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("per_layer selector values must be non-empty")
        return value

    @model_validator(mode="after")
    def _valid_selectors(self) -> "PerLayerAssign":
        positive = {
            "range": self.range is not None,
            "layers": self.layers is not None,
            "stride": self.stride is not None,
            "from": self.from_ is not None or self.is_ is not None,
            "within": self.within is not None,
            "pass": self.pass_ is not None,
        }
        selected = [name for name, present in positive.items() if present]
        if self.rest is not None:
            if selected:
                raise ValueError(
                    "per_layer `rest: true` is the exclusive catch-all; it "
                    "cannot be combined with other selectors; got "
                    + repr(["rest", *selected]))
        elif not selected:
            raise ValueError(
                "per_layer assign requires at least one selector: `range`, "
                "`layers`, `stride` (+ optional `offset`), `rest: true`, "
                "`from` + `is`, `from` alone for a config list of layer "
                "numbers, or `pass` in a looped stack (several are "
                "intersected)")
        if self.range is not None and len(self.range) != 2:
            raise ValueError("per_layer `range` must contain exactly [start, end]")
        if self.layers is not None and not self.layers:
            raise ValueError("per_layer `layers` must be non-empty")
        if self.offset is not None and self.stride is None:
            raise ValueError("per_layer `offset` is only valid with `stride`")
        if self.rest is not None and self.rest is not True:
            raise ValueError("per_layer catch-all selector must be `rest: true`")
        if self.is_ is not None and self.from_ is None:
            raise ValueError(
                "per_layer `is` selects on the config list `from` names; "
                "write `from: config.<list>` beside it")
        if self.base is not None and (self.from_ is None
                                      or self.is_ is not None):
            raise ValueError(
                "per_layer `base` is the number a config list of layer "
                "numbers gives the first layer; write it beside `from` "
                "alone (no `is`)")
        if self.base is not None and self.base < 0:
            raise ValueError("per_layer `base` must be non-negative")
        if self.from_ is not None and not self.from_.startswith("config."):
            raise ValueError("per_layer `from` must be a `config.<path>` reference")
        if self.stride is not None:
            try:
                if int(self.stride) < 1:
                    raise ValueError("per_layer literal `stride` must be positive")
            except ValueError as exc:
                if "must be positive" in str(exc):
                    raise
                # Symbolic dimension expressions are resolved with the model.
        if self.offset is not None:
            try:
                if int(self.offset) < 0:
                    raise ValueError(
                        "per_layer literal `offset` must be non-negative")
            except ValueError as exc:
                if "must be non-negative" in str(exc):
                    raise
        return self


class PerLayerMapEntry(BaseModel):
    """What one layer type is built from.

    Either a `template` (a block, inline or from the library) or the
    `attention` / `ffn` slot shorthand that the loader desugars into the
    spec's block pattern. Both may appear: a hand-written template that takes
    `attn` / `ffn` params gets its bindings this way.
    """

    model_config = ConfigDict(extra="forbid")

    template: Optional[str] = None
    attention: Optional[str] = None
    ffn: Optional[str] = None
    #: A param value may itself be `{template, params}` — a nested compound-op
    #: call — which is why this is not `Dict[str, str]`. See `GraphNode.params`.
    params: Optional[Dict[str, Any]] = None
    dims: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def _has_binding(self) -> "PerLayerMapEntry":
        if not (self.template or self.attention or self.ffn):
            raise ValueError(
                "per_layer map entry must bind `template`, `attention`, or `ffn`")
        return self


class PerLayerBlock(BaseModel):
    """`per_layer: {assign, map}` — the authored surface for per-layer
    template selection.

    Generators and bindings are separated because they vary independently: a
    model reshuffles which layers are sparse far more often than it changes
    what a sparse layer contains, and the two are authored by different facts
    (the config's layer pattern vs. the component library).
    """

    model_config = ConfigDict(extra="forbid")

    assign: List[PerLayerAssign] = Field(default_factory=list)
    map: Dict[str, PerLayerMapEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _every_generator_has_a_binding(self) -> "PerLayerBlock":
        """A `type` with no `map` entry selects layers and builds nothing.

        Unbound, the expansion produced an assignment with no template, and
        the first thing to notice was `ArchSpec`'s template check reporting
        `None` as an unknown template — true, but it points at the assembly
        product rather than at the two lines that disagree.
        """
        unbound = sorted({a.type for a in self.assign if a.type not in self.map})
        if unbound:
            raise ValueError(
                f"per_layer: `assign` tags layer type(s) {unbound} that `map` "
                f"does not bind. Known types: {sorted(self.map)}. Every "
                f"generator names the layers; the map says what they are.")
        if not self.assign:
            raise ValueError("per_layer.assign must contain at least one generator")
        invalid_keys = sorted(
            key for key in self.map if not key.strip() or key != key.strip())
        if invalid_keys:
            raise ValueError(
                f"per_layer.map keys must be non-empty and trimmed: {invalid_keys}")
        if sum(a.rest is True for a in self.assign) > 1:
            raise ValueError("per_layer may declare at most one `rest: true` generator")
        if any(a.rest is True for a in self.assign[:-1]):
            raise ValueError(
                "per_layer `rest: true` must be the final generator; it is the "
                "fallback after all declaration-order selectors")
        used = {a.type for a in self.assign}
        unused = sorted(set(self.map) - used)
        if unused:
            raise ValueError(
                f"per_layer.map contains unused type(s) {unused}; every map row "
                "must be selected by at least one assign generator")
        return self

    def expand(self) -> List[Dict[str, Any]]:
        """Flatten into `BlockAssignment` kwargs — one per generator.

        The map entry supplies the binding, the generator supplies the
        selection, and the generator's own `dims` / `params` win: they are the
        more specific statement about the layers it selects.
        """
        out: List[Dict[str, Any]] = []
        for a in self.assign:
            entry: Dict[str, Any] = self.map[a.type].model_dump(exclude_none=True)
            for sel in ("range", "layers", "stride", "offset", "rest",
                        "dims", "params", "reads_from"):
                val = getattr(a, sel)
                if val is not None:
                    entry[sel] = val
            if a.from_ is not None:
                # `from: config.layer_types ?? absent` declares that the
                # config may lack the list, in which case this generator
                # selects nothing and a sibling takes the layers. Without
                # the marker an absent list is a spec/config error.
                src, _, fallback = a.from_.partition("??")
                src = src.strip()
                fallback = fallback.strip()
                if fallback and fallback != "absent":
                    raise ValueError(
                        f"per_layer.assign[{a.type!r}].from: the only fallback "
                        f"is `?? absent`, got {fallback!r}")
                entry["config_list"] = (
                    src[len("config."):] if src.startswith("config.") else src)
                entry["config_list_optional"] = fallback == "absent"
            if a.is_ is not None:
                entry["equals"] = a.is_
            if a.base is not None:
                entry["config_list_base"] = a.base
            if a.within is not None:
                entry["within"] = a.within
            if a.pass_ is not None:
                entry["pass"] = a.pass_
            out.append(entry)
        return out


class MatchSpec(BaseModel):
    """Checkpoint discriminator for one authored model spec."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    #: Optional as of the id-keyed matching migration (contract B): a base
    #: spec that only exists to be inherited via `base:` declares no
    #: `model_type`, and an identity file discriminates on `model_id` alone.
    #: An empty list is still a mistake (write None / omit it). Retained
    #: alongside the config's own `model_type` for spec-gen context and load-
    #: failure messages; matching itself no longer reads it.
    model_type: Optional[List[str]] = None
    #: The model_ids this spec is the identity of (exact HF ids). `match()`
    #: resolves a request by looking its model_id up in this index — an exact
    #: lookup, so same-family variants (a model vs its Pro/Flash sibling, or
    #: its quant mirrors) each declare their own id and resolve to their own
    #: spec. A base spec that is only inherited declares none.
    model_id: Optional[List[str]] = None

    @field_validator("model_type")
    @classmethod
    def _valid_model_types(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return values
        if not values or any(not value.strip() for value in values):
            raise ValueError("match.model_type must contain non-empty names")
        if len(values) != len(set(values)):
            raise ValueError("match.model_type must not contain duplicates")
        return values

    @field_validator("model_id")
    @classmethod
    def _valid_model_ids(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return values
        if not values or any(not value.strip() for value in values):
            raise ValueError("match.model_id, when present, must contain non-empty patterns")
        if len(values) != len(set(values)):
            raise ValueError("match.model_id must not contain duplicates")
        return values


class SpecDocument(BaseModel):
    """A model spec YAML, exactly.

    Every field here is written by hand; nothing the loader computes appears.
    The reverse holds too — this document is not carried into the assembly
    product, so a consumer cannot reach around `ArchSpec` to re-read what the
    author wrote and reach a different conclusion than the lowering did.
    """

    model_config = ConfigDict(extra="forbid")

    #: Which checkpoints this spec claims, e.g. `{model_type: [qwen3_moe]}`.
    #: Optional as of the id-keyed matching migration (contract B): a base
    #: spec exists only to be inherited (`base:`) and is never matched
    #: directly, so it carries no `match:`. A spec with no `match` is not a
    #: match candidate — it is not an error.
    match: Optional[MatchSpec] = None

    #: Spec this one is derived from — its lineage. A Python table keyed
    #: by spec name is not a fact about the model.
    derived_from: Optional[str] = None

    #: Symbol → binding: `config.hidden_size`, `config.head_dim ?? "d_model // n_heads"`.
    #: Resolved top to bottom, so a later binding may name an earlier symbol.
    dims: Dict[str, str] = Field(default_factory=dict)
    #: Symbols this spec introduces, each a `dims.yaml` row (`kind`,
    #: `shardable_by`, `group`, `meaning`, optional `config` / `derived`),
    #: checked by the constructor dims.yaml is loaded with and promoted into
    #: dims.yaml when the model enters the library. Not `LayerTree.dim_symbols`
    #: (the symbols one layer's tensors are shaped on).
    dim_symbols: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    #: Display facts that are not dims (`ArchSpec.VIEW_KEYS`): key → config
    #: path(s), `??`-chained, kept in their native type by the loader.
    view: Dict[str, str] = Field(default_factory=dict)

    #: The tensors the model receives from outside its graph, by the name a
    #: layer's `inputs:` reads each by: `input_ids: {kind: tokens}`, an
    #: image `pixel_values: {kind: image, axes: [n_img, img_h, img_w]}`.
    model_inputs: Dict[str, ModelInputSpec] = Field(default_factory=dict)

    #: The layer stack, in order. Parsed by spec_parsing rather than by pydantic
    #: because a model layer carries authoring aliases (`in:` → `inputs:`) and
    #: the `per_layer` block above, which lowers to `assignments` before the
    #: `ModelLayer` is built.
    model: List[ModelLayer]

    #: Where the model's entries live in the checkpoint: `{root: model}` for
    #: a transformers `*ForCausalLM` whose layers sit under `model.`. Absent
    #: is the file root.
    checkpoint: Optional[CheckpointAddress] = None

    #: Slot → component name: `{attention: mla_attention, ffn_moe: ...}`.
    #: Slot names are roles, and `roles.yaml` classifies them.
    components: Dict[str, str] = Field(default_factory=dict)

    #: Block graphs written out in full, for topologies the `pre_norm` /
    #: `post_norm` patterns cannot express (vision towers, MTP heads).
    blocks: Dict[str, BlockGraph] = Field(default_factory=dict)

    #: Agent-generated components this spec opts into, by file stem. Invisible
    #: to every other spec, and only for the duration of this one's assembly.
    drafts: List[str] = Field(default_factory=list)

    #: Compute primitives no composition of registered ops can express.
    custom_ops: Dict[str, CustomOpSpec] = Field(default_factory=dict)

    #: Per-role quantization — see docs/mechanisms/quantization.md.
    quant_context: Optional[QuantContext] = None

    #: Narrow validation opt-outs. Every exemption carries a human reason.
    validation_exemptions: ValidationExemptions = Field(
        default_factory=ValidationExemptions)

    @field_validator("dims", mode="before")
    @classmethod
    def _normalize_dim_bindings(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        return {name: str(binding) for name, binding in values.items()}

    @field_validator("model")
    @classmethod
    def _nonempty_model(cls, layers: List[ModelLayer]) -> List[ModelLayer]:
        if not layers:
            raise ValueError("model must contain at least one layer")
        names = [layer.name for layer in layers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"top-level model layer names must be unique: {duplicates}")
        input_boundaries = [
            layer.name for layer in layers
            if layer.boundary_role == "input_embedding"
        ]
        if len(input_boundaries) != 1:
            raise ValueError(
                "an authored model must declare exactly one top-level "
                f"`boundary_role: input_embedding`; got {input_boundaries}")
        output_boundaries = [
            layer.name for layer in layers
            if layer.boundary_role == "output_head"
        ]
        if not output_boundaries:
            raise ValueError(
                "an authored model must declare at least one top-level "
                "`boundary_role: output_head`")
        return layers
