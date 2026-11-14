"""QuantContext — the spec-side quantization context."""
from __future__ import annotations

import re
from typing import (Any, Dict, Iterator, List, Literal, Optional,
                    Tuple)

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

import backend.arch.roles as roles
from backend.arch import cache_families
from backend.arch.cache_formats import CacheFormat
from backend.arch.quant_schemes import QUANT_SCHEMES

from .model import LayerSelection




# ---------------------------------------------------------------------------
# Quantization context — single source of truth for HOW each spec tensor
# materializes on disk.
#
# Two-layer completeness:
#   per_role[<role>]  → method (∈ quant_schemes registry) + scheme-specific
#                       fields (group_size, block_h, block_w, …). Default
#                       coverage for the matching role's tensors with
#                       (out, in) axes: linears, per-head banks, lookup tables.
#   overrides[]       → explicit per-tensor escape hatch for non-linear
#                       and role-overriding cases (embed/lm_head/KV scales).
#
# This module defines the data structure + load-time validation
# (method ∈ scheme registry, dtype ∈ allowed set, override shape).
# Engine fusion (`materialize_tensors`) and the quant-suffix
# parameter_vector lockdown live in their own modules / hooks.
# ---------------------------------------------------------------------------

# safetensors dtype set (what authors can write as `dtype:` in overrides
# or `default_dtype:`). Wider than what HF transformers always supports —
# we allow anything actually serializable to a safetensors file.
_ALLOWED_QUANT_CONTEXT_DTYPES = frozenset({
    "F64", "F32", "F16", "BF16",
    "F8_E4M3", "F8_E5M2", "F8_E8M0FNU",
    "I8", "I16", "I32", "I64",
    "U8", "U16", "U32", "U64",
    # Human-readable aliases — accepted at parse time, not normalized.
    "FP64", "FP32", "FP16", "FP8", "INT8", "INT4", "INT32",
})


# Sentinel `method` value used when a role/override is explicitly NOT
# quantized (e.g. `default: { method: none }`). Bypasses scheme lookup.
_QUANT_METHOD_NONE = "none"



def _valid_quant_regex(location: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        re.compile(value)
    except re.error as exc:
        raise ValueError(f"{location} is not a valid regular expression: {exc}") \
            from exc
    return value



def _valid_quant_shape(
    location: str, values: Optional[List[str]],
) -> Optional[List[str]]:
    if values is not None and (
            not values or any(not value.strip() for value in values)):
        raise ValueError(
            f"{location} must contain non-empty axis expressions when present")
    return values



class BlockScaleLayout(BaseModel):
    """A checkpoint whose complete fused row chunks own independent FP8 grids.

    A loader can merge whole chunks, or gather individual heads from them,
    and requantize a fresh rank-local scale grid. Logical projection segments
    do not define the scale-block boundaries.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tp_interleaved"]
    checkpoint_tp: int = Field(gt=0)
    world: str


class PerRoleEntry(BaseModel):
    """One row of `quant_context.per_role`. `method` routes to a yaml
    under `backend/arch/quant_schemes/`; remaining fields are
    scheme-specific (group_size / block_h / block_w / num_bits / …).
    Schema accepts those open fields, then validates them against the chosen
    scheme's required/optional/default field contract at load time.

    `method` is optional so a role can name a serving dtype without a
    scheme — the same shape `OverrideEntry` already has. A row with no
    method may carry `dtype` and nothing else.
    """
    method: Optional[str] = None
    dtype: Optional[str] = None
    scale_shape: Optional[List[str]] = None
    block_scale_layout: Optional[BlockScaleLayout] = Field(
        default=None, exclude_if=lambda v: v is None)
    # Scheme-specific config fields are passed through verbatim.
    model_config = {"extra": "allow"}

    @field_validator("scale_shape")
    @classmethod
    def _valid_scale_shape(
        cls, values: Optional[List[str]],
    ) -> Optional[List[str]]:
        return _valid_quant_shape("PerRoleEntry.scale_shape", values)

    @field_validator("dtype")
    @classmethod
    def _known_dtype(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if value not in _ALLOWED_QUANT_CONTEXT_DTYPES:
            raise ValueError(
                f"PerRoleEntry.dtype={value!r} is not a recognized "
                f"safetensors dtype. Allowed: "
                f"{sorted(_ALLOWED_QUANT_CONTEXT_DTYPES)}"
            )
        return value



class OverrideSelector(BaseModel):
    """Logical identity selected by a quantization override.

    A checkpoint name is not a tensor identity: an aux stack may deliberately
    share the backbone's checkpoint address, and two components of a hybrid
    stack may both call a projection `self_attn.o_proj`. `stack`, `component`
    and `layers` therefore scope the selection before the literal/regex name
    predicate is evaluated. `component` holds when the node sits inside an
    invocation of that component, at any depth of its `component_path`.

    `layers` is a `LayerSelection` — the same `range:` / `layers:` /
    `stride`+`offset` / `config_list`+`equals` a `BlockAssignment` selects
    layers with, resolved by the same `ResolvedSpec.layers_selected`. It
    exists because a precision is not always a structural fact: Step-5's
    `modules_to_not_convert` names the routed-expert weights of three of its
    91 layers, which are the same role, the same stack, the same component
    and the same graph as the other 88. A checkpoint name cannot say it —
    the eval layer's names carry a literal `X` at every indexed segment — and
    duplicating the component to carry the dtype would make a precision into
    a structural identity, which `quant_context` exists to prevent. Authored
    either as a mapping (`layers: {range: ["88", "n_layers - 1"]}`) or, for
    an explicit set, as the bare list a `BlockAssignment` writes
    (`layers: [88, 89, 90]`).

    The layer numbers are resolved against the stack the row names, the
    backbone when it names none, and matched against the layer the node
    runs at. Dim expressions resolve like any other layer bound.

    A `component`-scoped row may select no tensor that another authored row
    also selects: it is a deliberate narrowing, and first-match order would
    otherwise decide silently between it and a broader row.

    A `layers`-scoped row MAY, and normally does, select tensors another row
    (or a `per_role` rule) also selects: carving some layers out of a rule
    that covers all of them is the whole point, and the two answers do not
    compete because the narrowing is along a dimension the broader row does
    not speak about. Precedence is unchanged — declaration-order first match,
    asked once PER LAYER — so a layer-scoped row goes BEFORE the broader row
    it carves out of, and the existing `quant_override_unused` gate says so
    when it does not.
    """

    model_config = ConfigDict(extra="forbid")

    stack: Optional[str] = None
    component: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    layers: Optional[LayerSelection] = Field(
        default=None, exclude_if=lambda v: v is None)
    tensors: Optional[List[str]] = None
    tensors_re: Optional[str] = None

    @field_validator("layers", mode="before")
    @classmethod
    def _layer_scope(cls, value: Any) -> Any:
        """A bare list is the explicit-index form, spelled exactly as a
        `BlockAssignment` spells it: `layers: [88, 89, 90]`."""
        if isinstance(value, (list, tuple)):
            return {"layers": [str(entry) for entry in value]}
        return value

    @field_validator("tensors_re")
    @classmethod
    def _compiled_regex(cls, value: Optional[str]) -> Optional[str]:
        return _valid_quant_regex("OverrideSelector.tensors_re", value)

    @model_validator(mode="after")
    def _validate_selector(self) -> "OverrideSelector":
        if (self.tensors is None) == (self.tensors_re is None):
            raise ValueError(
                "OverrideSelector: exactly one of `tensors:` or "
                "`tensors_re:` must be set")
        if self.tensors is not None and not self.tensors:
            raise ValueError("OverrideSelector.tensors must be non-empty")
        if self.tensors is not None:
            if any(not tensor.strip() for tensor in self.tensors):
                raise ValueError(
                    "OverrideSelector.tensors must contain non-empty names")
            if len(self.tensors) != len(set(self.tensors)):
                raise ValueError(
                    "OverrideSelector.tensors must not contain duplicates")
        if self.stack is not None and not self.stack.strip():
            raise ValueError("OverrideSelector.stack must be non-empty")
        if self.component is not None and not self.component.strip():
            raise ValueError("OverrideSelector.component must be non-empty")
        if self.layers is not None and not self.layers.selects_layers:
            raise ValueError(
                "OverrideSelector.layers names no layers; write one of "
                "`range:`, `layers:`, `stride:` or `config_list:` (the "
                "selectors a block assignment uses), or drop it")
        return self

    def scopes(self, stack: Optional[str],
               components: Tuple[str, ...]) -> bool:
        """Whether a node in `stack`, inside the component invocations
        `components`, is within this selector's logical scope.

        The layer scope is NOT asked here: resolving it needs the model's
        dims and config, so `FormatPolicy` holds the resolved layer set and
        asks per layer (`FormatPolicy._override_row`)."""
        if self.stack and self.stack != stack:
            return False
        return not self.component or self.component in components



class OverrideEntry(BaseModel):
    """One entry in `quant_context.overrides` — explicit escape hatch
    for tensors not covered by per_role (non-linear, role-overriding,
    or auto-imported from `config.quantization_config.ignore`).

    Existing entries may use one flat `tensors` / `tensors_re` predicate.
    Composite identity predicates live under `select:` so selector fields do
    not mix with the override action and scheme-specific settings.
    """
    select: Optional[OverrideSelector] = None
    tensors: Optional[List[str]] = None
    tensors_re: Optional[str] = None
    dtype: str
    method: Optional[str] = None       # optional; None = unquantized
    # The metadata sibling's SHAPE for these tensors, one expression per axis,
    # when this checkpoint lays it out differently from the scheme's default.
    # Resolved against the weight's own `in`/`out` plus the node's dims.
    #
    # A layout, not a policy — which is why it lives on the model and not in
    # the scheme. MiMo's split K projection can be blocked per head, while
    # its fused QKV checkpoints carry independent grids per interleaved TP
    # chunk. These are exporter layouts; they do not redefine the scheme.
    scale_shape: Optional[List[str]] = None
    block_scale_layout: Optional[BlockScaleLayout] = Field(
        default=None, exclude_if=lambda v: v is None)
    # `auto_imported: true` marks entries auto-translated from
    # `config.quantization_config.ignore`. Used by the regen
    # pipeline so manual entries survive a spec re-generation while
    # auto-imports get re-derived. Authors should leave this False.
    auto_imported: bool = False

    # Scheme-specific config fields pass through verbatim, exactly as they do
    # on `PerRoleEntry`. An override naming a `method` needs that scheme's
    # settings — `block_h` / `block_w` / `group_size` — and pydantic's default
    # is to DROP an unknown field silently. Dropped, the scheme's cardinality
    # cannot resolve, the tensor falls back to the default dtype, and the
    # answer is wrong with nothing said: a `block_h` that never arrived turned
    # an FP8 projection into BF16 and put a model 144 MiB over.
    model_config = {"extra": "allow"}

    @field_validator("tensors_re")
    @classmethod
    def _compiled_regex(cls, value: Optional[str]) -> Optional[str]:
        return _valid_quant_regex("OverrideEntry.tensors_re", value)

    @field_validator("scale_shape")
    @classmethod
    def _valid_scale_shape(
        cls, values: Optional[List[str]],
    ) -> Optional[List[str]]:
        return _valid_quant_shape("OverrideEntry.scale_shape", values)

    @model_validator(mode="after")
    def _validate_shape(self) -> "OverrideEntry":
        has_flat = self.tensors is not None or self.tensors_re is not None
        if self.select is not None and has_flat:
            raise ValueError(
                "OverrideEntry: `select:` and flat `tensors:` / "
                "`tensors_re:` are mutually exclusive")
        if self.select is None:
            if (self.tensors is None) == (self.tensors_re is None):
                raise ValueError(
                    "OverrideEntry: exactly one of `select:`, `tensors:` or "
                    "`tensors_re:` must be set")
            if self.tensors is not None and not self.tensors:
                raise ValueError(
                    "OverrideEntry.tensors must be non-empty if specified.")
            if self.tensors is not None:
                if any(not tensor.strip() for tensor in self.tensors):
                    raise ValueError(
                        "OverrideEntry.tensors must contain non-empty names")
                if len(self.tensors) != len(set(self.tensors)):
                    raise ValueError(
                        "OverrideEntry.tensors must not contain duplicates")
        for scope in ("stack", "component", "layers"):
            if self.model_extra and scope in self.model_extra:
                raise ValueError(
                    f"OverrideEntry: `{scope}:` is a selector; move it under "
                    "`select:` beside `tensors:` / `tensors_re:`")
        if self.dtype not in _ALLOWED_QUANT_CONTEXT_DTYPES:
            raise ValueError(
                f"OverrideEntry.dtype={self.dtype!r} is not a recognized "
                f"safetensors dtype. Allowed: "
                f"{sorted(_ALLOWED_QUANT_CONTEXT_DTYPES)}"
            )
        return self

    def selector_text(self) -> str:
        """This row's selector as the spec author wrote it, for diagnostics."""
        selector = self.select
        if selector is None:
            return (f"tensors_re={self.tensors_re!r}" if self.tensors_re
                    else f"tensors={self.tensors!r}")
        parts = [f"{key}={value!r}" for key, value in (
            ("stack", selector.stack), ("component", selector.component),
            ("layers", (selector.layers.selection_text()
                        if selector.layers is not None else None)),
            ("tensors", selector.tensors), ("tensors_re", selector.tensors_re))
            if value]
        return "select: " + ", ".join(parts)



def _validate_quant_entry_fields(
    location: str,
    method: Optional[str],
    authored_fields: set,
    scale_shape: Optional[List[str]],
) -> None:
    """Validate one open Pydantic row against its selected scheme."""
    if method is None or method == _QUANT_METHOD_NONE:
        if authored_fields or scale_shape is not None:
            extras = sorted(authored_fields | (
                {"scale_shape"} if scale_shape is not None else set()))
            raise ValueError(
                f"{location} has no quantization method but declares scheme "
                f"field(s) {extras}")
        return

    scheme = QUANT_SCHEMES.get(method)
    if scheme is None:
        raise ValueError(
            f"{location}.method={method!r} is not a registered quant scheme; "
            f"known methods: {sorted(QUANT_SCHEMES)}")

    defaults = {name for name, _value in scheme.default_fields}
    allowed = (set(scheme.required_fields) | defaults
               | set(scheme.optional_fields))
    unknown = sorted(authored_fields - allowed)
    if unknown:
        raise ValueError(
            f"{location} declares field(s) {unknown} not accepted by quant "
            f"scheme {method!r}; allowed scheme fields: {sorted(allowed)}")
    missing = sorted(set(scheme.required_fields) - authored_fields - defaults)
    if missing:
        raise ValueError(
            f"{location} uses quant scheme {method!r} but is missing required "
            f"field(s) {missing}")



def _validate_block_scale_entry(location: str, entry: Any) -> None:
    if entry.block_scale_layout is not None and entry.method != "fp8":
        raise ValueError(f"{location}: block_scale_layout requires method: fp8")
    if (entry.block_scale_layout is None
            and any(re.search(r"\bchunks_per_rank\b", axis)
                    for axis in entry.scale_shape or ())):
        raise ValueError(
            f"{location}: scale_shape uses chunks_per_rank without "
            "block_scale_layout")


class QuantContext(BaseModel):
    """Top-level quant configuration on a model spec. Optional — when
    absent, the model is treated as fully unquantized at `BF16` (or
    whatever HF safetensors header says at load time).
    """
    model_config = ConfigDict(extra="forbid")

    per_role: Dict[str, PerRoleEntry] = Field(default_factory=dict)
    overrides: List[OverrideEntry] = Field(default_factory=list)
    default_dtype: str = "BF16"
    #: The storage format this checkpoint's forward keeps a cache slot kind
    #: at, when that is not the model's dtype: slot kind (`cache_families.
    #: yaml` `slot_kinds:`) → a named cache format (`arch/cache_formats/`)
    #: or a spelled-out `{dtype, scale, transform}`. A model fact, so it is
    #: declared per model: components shared by several models (the DSA
    #: indexer of DeepSeek-V3.2 and GLM-5) cache the same slot differently.
    #: It wins over a component entry's own `dtype:` / `format:`.
    cache_formats: Dict[str, Any] = Field(
        default_factory=dict, exclude_if=lambda v: not v)

    @property
    def declared_default_dtype(self) -> Optional[str]:
        """`default_dtype` when the spec writes it, None when it is the
        field's default — a value nobody stated is not a declaration."""
        return (self.default_dtype if "default_dtype" in self.model_fields_set
                else None)

    def scheme_entries(self) -> Iterator[Tuple[str, int, Dict[str, Any]]]:
        """`(method, num_bits, extra)` per declared role, normalized.

        `method: none` (or empty) is unquantized and skipped. `num_bits`
        coerces to int with 0 for junk — the scheme registry's own
        `read_num_bits` is the answer when the entry leaves it unset.
        The one normalization, so the quant badge and the precision
        label cannot disagree on what a per_role row says.
        """
        for entry in (self.per_role or {}).values():
            method = (entry.method or "").strip().lower()
            if not method or method == "none":
                continue
            extra = dict(entry.model_extra or {})
            num_bits = extra.get("num_bits") or 0
            try:
                num_bits = int(num_bits)
            except (TypeError, ValueError):
                num_bits = 0
            yield method, num_bits, extra

    def storage_agreement(self, quant_method: str) -> int:
        """+1 if this context's declared methods agree with a checkpoint
        `quant_method`, -1 if they do not.

        `method: none` is unquantized (`scheme_entries` skips it). Comparison
        is prefix-tolerant: the checkpoint's `fp8` covers a spec's
        `fp8_ue8m0`, and `mxfp4` covers `mxfp4-pack-quantized`. Used only to
        break an equal architecture match, not to price tensors.
        """
        methods = {method for method, _bits, _extra in self.scheme_entries()}
        checkpoint = (quant_method or "").strip().lower()
        if not checkpoint and not methods:
            return 1
        if not checkpoint or not methods:
            return -1
        for method in methods:
            if (method == checkpoint
                    or method.startswith(checkpoint)
                    or checkpoint.startswith(method)):
                return 1
        return -1

    @field_validator("cache_formats")
    @classmethod
    def _cache_formats_are_formats(cls, value):
        for kind, fmt in (value or {}).items():
            if kind not in cache_families.slot_kinds():
                raise ValueError(
                    f"quant_context.cache_formats names {kind!r}, which is "
                    f"not a cache slot kind "
                    f"({sorted(cache_families.slot_kinds())}).")
            cache_families.refuse_storage_format(kind)
            CacheFormat.parse(fmt, f"quant_context.cache_formats.{kind}")
        return value

    @model_validator(mode="after")
    def _validate_default_dtype(self) -> "QuantContext":
        if self.default_dtype not in _ALLOWED_QUANT_CONTEXT_DTYPES:
            raise ValueError(
                f"QuantContext.default_dtype={self.default_dtype!r} is not a "
                f"recognized safetensors dtype. Allowed: "
                f"{sorted(_ALLOWED_QUANT_CONTEXT_DTYPES)}"
            )
        allowed_roles = set(roles.all_roles()) | {"default"}
        unknown_roles = sorted(set(self.per_role) - allowed_roles)
        if unknown_roles:
            raise ValueError(
                f"QuantContext.per_role uses unknown role(s) {unknown_roles}; "
                f"known roles: {sorted(allowed_roles)}")

        for role, entry in self.per_role.items():
            _validate_quant_entry_fields(
                f"quant_context.per_role.{role}", entry.method,
                set(entry.model_extra or {}), entry.scale_shape)
            _validate_block_scale_entry(
                f"quant_context.per_role.{role}", entry)
        for index, entry in enumerate(self.overrides):
            _validate_quant_entry_fields(
                f"quant_context.overrides[{index}]", entry.method,
                set(entry.model_extra or {}), entry.scale_shape)
            _validate_block_scale_entry(
                f"quant_context.overrides[{index}]", entry)
        selectors = []
        for index, entry in enumerate(self.overrides):
            selector = entry.select
            signature = (
                selector.stack if selector is not None else None,
                selector.component if selector is not None else None,
                # The layer scope is part of the identity: two rows that
                # differ only by the layers they speak about are two rules,
                # and that is exactly how a layer-scoped carve-out is written.
                (selector.layers.selection_text()
                 if selector is not None and selector.layers is not None
                 else None),
                tuple(selector.tensors or ()) if selector is not None
                else tuple(entry.tensors or ()),
                selector.tensors_re if selector is not None else entry.tensors_re,
            )
            if signature in selectors:
                earlier = selectors.index(signature)
                raise ValueError(
                    f"quant_context.overrides[{index}] duplicates selector "
                    f"overrides[{earlier}]; declaration-order first-wins would "
                    "make the later row unreachable")
            selectors.append(signature)
        return self

    def known_methods(self) -> set:
        """Helper: union of all `method` values used (per_role + overrides),
        excluding the `none` sentinel. Caller can intersect with the
        scheme registry to validate.
        """
        methods = {e.method for e in self.per_role.values()
                   if e.method != _QUANT_METHOD_NONE}
        methods |= {e.method for e in self.overrides
                    if e.method and e.method != _QUANT_METHOD_NONE}
        return methods
