"""Quantization scheme registry — data-driven from the yaml files in this package.

Each supported HF quant library exposes its own tensor-naming convention
and config layout. Rather than union all names into one flat set, we
register a `QuantScheme` per library and dispatch by `quant_method` from
`config.quantization_config`. Mirrors how transformers uses
`AutoHfQuantizer` to pick a per-method quantizer class.

Adding a new scheme = one YAML file under `arch/quant_schemes/`. No
Python edits.

Public API:
  • `QuantScheme` dataclass — describes one library's on-disk layout +
    badge template + the materialization recipe consumed by the
    engine's `materialize_tensors` fusion pass.
  • `resolve_scheme(config)` → `(scheme, num_bits, unknown_method)`.
    Used by rendering._badges.
  • `SCHEME_NONE` — no sidecars and no packing, for a checkpoint whose
    config names no known quant_method.
  • `is_quantization_metadata(name, scheme)` — classifier for "is this
    tensor a quant scale/index, not a learnable param?"
  • `packed_unpack_factor(name, dtype, num_bits, scheme)` — elements
    per packed unit (e.g. I32 × INT4 → 8).

Each scheme is a single source of truth used by both:
  • the HF-side classifiers (`is_quantization_metadata`,
    `packed_unpack_factor`) which read `metadata_siblings` and
    `format_variants`
  • `eval.formats` which drives on-disk dtype / packing / metadata
    siblings during the evaluator walk
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from backend.arch.activation_formats import ACTIVATION_FORMATS
from backend.arch.dtypes import canonical
from backend.arch.eval.scheme_dialect import eval_predicate


# Tight enum for storage-vs-logical packing ratios. Any other value is
# structurally impossible — bit width must integer-divide storage word
# size. Validated at scheme load.
_ALLOWED_PACKING_FACTORS = frozenset({1, 2, 4, 8, 16})

# Recognized values for `quantization_status` field of compressed-tensors
# and similar libraries. Only `compressed` / `frozen` mean the on-disk
# weights are actually quantized; `initialized` / `calibration` mean
# weights are still in the base dtype.
_KNOWN_QUANT_STATUSES = frozenset(
    {"compressed", "frozen", "initialized", "calibration"}
)


@dataclass(frozen=True)
class FormatVariant:
    """One on-disk layout variant of a quant method.

    Some methods have multiple variants under the same `quant_method`
    name (e.g. compressed-tensors `pack-quantized` vs `dense`). Each
    variant fixes its own packing_factor + storage_dtype + weight tensor
    suffix. The variant is picked by `per_role[<role>].format` at
    materialize time.

    Either `packing_factor` (scalar) OR `packing_factor_per_num_bits`
    (per-num_bits map) is set; same for storage_dtype.
    """
    name: str
    packing_factor: Optional[int] = None
    packing_factor_per_num_bits: Optional[Dict[int, int]] = None
    storage_dtype: Optional[str] = None
    storage_dtype_per_num_bits: Optional[Dict[int, str]] = None
    weight_storage_suffix: str = ".weight"


@dataclass(frozen=True)
class MetadataSibling:
    """A metadata tensor that a method emits alongside each linear
    weight on disk (scale / zero-point / packed-shape / g_idx / ...).

    Cardinality is a SymPy formula in {in, out, group_size, block_h,
    block_w, num_bits, ...}. The engine binds these symbols at
    materialize time from the linear's resolved dims + per_role config.

    `when:` is an optional SymPy boolean predicate over the same symbol
    space — if false, the sibling is NOT emitted (e.g. zero_point only
    when symmetric=false; g_idx only when actorder=true).
    """
    suffix: str
    #: What the sibling IS in the scheme's layout — one of `SIBLING_KINDS`.
    #: Declared, not read off the suffix: the display string borrows the
    #: `scale` tensor's dtype, and which tensor that is used to be answered
    #: by substring-matching the name.
    kind: str
    cardinality: Optional[str] = None                       # SymPy formula
    cardinality_per_strategy: Optional[Dict[str, str]] = None  # strategy → formula
    # The sibling's SHAPE, one expression per axis, when a scalar count is too
    # weak to say what the file holds. The element count is then a projection
    # of it — the product — rather than a second declaration.
    #
    # A count cannot be checked against anything; a shape can be checked
    # against the header's, axis by axis. `kv_a_proj_with_mqa` is [576, 7168]
    # against a 128 block and the file records its scale as [5, 56]; a scalar
    # formula that quietly produced 4 x 56 was wrong by 6,944 bytes across the
    # DeepSeek family and nothing could see it, because a total has no axes.
    #
    # It is also the only form that can express a layout: MiMo blocks the K
    # segment of a QKV projection per HEAD while blocking Q and V flat, in both
    # the fused and the split form, over five checkpoints. That is
    # `n_kv_heads * ceil(d_head / block_h)` — writable as a shape, not as a
    # function of `in` and `out`.
    shape: Optional[List[str]] = None
    shape_per_strategy: Optional[Dict[str, List[str]]] = None
    dtype: Optional[str] = None
    dtype_per_format: Optional[Dict[str, str]] = None       # format_variant → dtype
    #: What the slot HOLDS when that differs from the slot: an MXFP4 block
    #: scale is a UE8M0 exponent stored in a U8. The header and the byte
    #: total read `dtype`; the runtime vocabulary (the calibration lookup's
    #: `metadata_dtype`) reads this. None means the slot type is the type.
    logical_dtype: Optional[str] = None
    when: Optional[str] = None                              # SymPy boolean predicate


@dataclass(frozen=True)
class QuantScheme:
    """Describes how one quant library lays out its tensors on disk."""
    name: str
    # Extract weight `num_bits` from that library's `quantization_config` dict.
    # Return 0 if not declared.
    read_num_bits: Callable = field(default=lambda qc: 0)
    # Architecture-view badge template. May contain `{num_bits}` /
    # `{method}` placeholders, formatted at lookup time. Empty when the
    # scheme is `SCHEME_NONE` or hasn't declared one — caller falls
    # back to uppercased method name.
    badge: str = ""

    # On-disk layout variants. Empty dict when the method has only one
    # variant (use `format_variants["default"]`).
    format_variants: Dict[str, FormatVariant] = field(default_factory=dict)

    # Numeric-format families this scheme can express
    # (e.g. ["int_sym", "int_asym", "fp_e2m1"]). Picked from per_role
    # config via `numeric_format_resolver` rules.
    numeric_format: Tuple[str, ...] = field(default_factory=tuple)

    # Ordered list of (when_predicate, format_name) rules — first
    # matching rule's format is selected. Predicates use per_role config
    # symbols (type, symmetric, num_bits, ...).
    numeric_format_resolver: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    # Scale strategies this method supports
    # (e.g. ["group", "channel", "tensor", "tile_2d"]). Picked from
    # per_role config's `strategy:` field; drives metadata_sibling
    # cardinality formula choice.
    scale_strategy: Tuple[str, ...] = field(default_factory=tuple)

    # Metadata sibling tensors emitted per linear (engine fusion).
    metadata_siblings: Tuple[MetadataSibling, ...] = field(default_factory=tuple)

    @property
    def scale_suffixes(self) -> Tuple[str, ...]:
        """Suffixes of the siblings the scheme declares as its SCALE
        tensors (`kind: scale`) — the answer to "which metadata tensor's
        dtype is the scale's", declared rather than name-matched."""
        return tuple(s.suffix for s in self.metadata_siblings
                     if s.kind == "scale")

    def is_scale_sibling(self, tensor_name: str) -> bool:
        """Whether a checkpoint tensor name is this scheme's scale."""
        return any(tensor_name.endswith(sfx) for sfx in self.scale_suffixes)

    def sibling_kind(self, suffix: str) -> Optional[str]:
        """The declared `kind` of the sibling stored under `suffix`, None
        when this scheme declares no such sibling."""
        for sibling in self.metadata_siblings:
            if sibling.suffix == suffix:
                return sibling.kind
        return None

    # Fields that `model.quant_context.per_role[<role>]` MUST supply
    # for this method. Validated at materialize time.
    required_fields: Tuple[str, ...] = field(default_factory=tuple)

    # Scheme config accepted when it is neither universally required nor
    # supplied by `default_fields`. Keeping this declaration beside the
    # scheme prevents the authored schema from becoming an open bag of every
    # quantizer's knobs.
    optional_fields: Tuple[str, ...] = field(default_factory=tuple)

    # Only these `quantization_status` values mean weights are
    # actually quantized on disk. Empty = don't check.
    valid_status: Tuple[str, ...] = field(default_factory=tuple)

    # Default values for scheme-config fields that per_role doesn't
    # specify. Merged INTO per_role bindings BEFORE predicate /
    # cardinality evaluation. Keeps simple per_role configs short
    # (e.g. user can write `{ method: compressed-tensors, num_bits: 4,
    # group_size: 32, strategy: group, type: int, symmetric: true,
    # format: pack-quantized }` and rely on `dynamic: false` /
    # `actorder: false` defaults from this map). Tuple-of-tuples to
    # keep the dataclass frozen-hashable.
    default_fields: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    # What a checkpoint written under this method may also hold, by its
    # own config: `(field, value, scheme)` — when
    # `quantization_config.<field> == value`, the file also carries
    # `scheme`'s metadata siblings (DeepSeek-V4's `scale_fmt: ue8m0`
    # writes UE8M0 scales at `.scale`, the `fp8_ue8m0` layout). Siblings
    # only: a packed weight is told by the method's own variants.
    also_stores: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)

    # Whether, and how, a DEPLOYMENT may run weights in this scheme when the
    # checkpoint saved them otherwise (a precision profile's weight rule).
    # `deployment_fields` completes the scheme's config for such a weight —
    # a checkpoint states its own fields, a deployment choice has none to
    # state. Empty: the scheme describes checkpoints only.
    deployment_fields: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    # The activation handling a deployment chooses with the scheme:
    # option name → activation format (`backend/arch/activation_formats/`),
    # None for "the kernel takes the input edge as it is" (W4A16).
    # Empty: the scheme offers no choice.
    activation_options: Tuple[Tuple[str, Optional[str]], ...] = field(
        default_factory=tuple)
    #: How a CHECKPOINT written under this scheme treats the activation
    #: operand of the matmuls it covers — the scheme's own `activation:`
    #: declaration, read against the per-role bindings. Empty: the scheme
    #: makes no statement, and the kernel takes the input edge as it
    #: arrives (every weight-only format).
    activation: Optional["ActivationDeclaration"] = None

    def activation_format(self, bindings: Mapping[str, Any]
                          ) -> Optional[str]:
        """The activation format this scheme's matmuls quantise to under
        `bindings` (a per-role entry over the scheme's defaults), None when
        the kernel reads the input edge as it is."""
        if self.activation is None:
            return None
        return self.activation.resolve(bindings, self.name)

    @property
    def deployable(self) -> bool:
        return bool(self.deployment_fields)

    def format_badge(self, num_bits: int, method: str) -> str:
        """Resolve the badge template against the runtime values. Returns
        empty string when no template is declared."""
        if not self.badge:
            return ""
        try:
            return self.badge.format(num_bits=num_bits, method=method)
        except (KeyError, IndexError):
            return self.badge


# --- num_bits resolvers (data-driven via YAML `num_bits:` key) ----------
#
# Each YAML scheme picks ONE of three forms:
#   num_bits: { fixed: N }                       — literal value
#   num_bits: { field_any: [a, b, ...] }         — first present qc.<key>
#   num_bits: { nested_field: [a, "*", b] }      — dotted path with `*` =
#                                                   "first dict-valued
#                                                   child (any key)"
# The resolver builds a callable from the spec at scheme-load time.

def _build_num_bits_resolver(spec: Dict[str, Any]) -> Callable:
    if not isinstance(spec, dict):
        return lambda qc: 0
    if "fixed" in spec:
        v = int(spec["fixed"])
        return lambda qc, _v=v: _v
    if "field_any" in spec:
        keys = list(spec["field_any"])
        def resolve(qc, _keys=keys):
            if not isinstance(qc, dict):
                return 0
            for k in _keys:
                v = qc.get(k)
                if isinstance(v, int) and v > 0:
                    return v
            return 0
        return resolve
    if "nested_field" in spec:
        path = list(spec["nested_field"])
        def resolve(qc, _p=path):
            cur: Any = qc
            for seg in _p:
                if not isinstance(cur, dict):
                    return 0
                if seg == "*":
                    # iterate dict-valued children, return first hit on
                    # the remainder of the path
                    rest = _p[_p.index("*") + 1:]
                    for sub in cur.values():
                        if not isinstance(sub, dict):
                            continue
                        v: Any = sub
                        ok = True
                        for s2 in rest:
                            if not isinstance(v, dict):
                                ok = False
                                break
                            v = v.get(s2)
                        if ok and isinstance(v, int) and v > 0:
                            return v
                    return 0
                cur = cur.get(seg)
            return cur if isinstance(cur, int) and cur > 0 else 0
        return resolve
    return lambda qc: 0


def _parse_format_variants(raw: Dict[str, Any], scheme_name: str) -> Dict[str, FormatVariant]:
    """Parse the `format_variants:` block from a scheme yaml.

    Each entry must declare exactly ONE of (packing_factor,
    packing_factor_per_num_bits) and exactly ONE of (storage_dtype,
    storage_dtype_per_num_bits). Validates `packing_factor` ∈
    `_ALLOWED_PACKING_FACTORS`.
    """
    out: Dict[str, FormatVariant] = {}
    if not isinstance(raw, dict):
        return out
    for variant_name, vraw in raw.items():
        if not isinstance(vraw, dict):
            continue
        pf = vraw.get("packing_factor")
        pf_map = vraw.get("packing_factor_per_num_bits")
        if pf is not None and pf_map is not None:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml format_variants[{variant_name!r}]: "
                f"set EITHER `packing_factor` OR `packing_factor_per_num_bits`, not both."
            )
        if pf is not None and pf not in _ALLOWED_PACKING_FACTORS:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml format_variants[{variant_name!r}]: "
                f"packing_factor={pf} not in {sorted(_ALLOWED_PACKING_FACTORS)}. "
                f"Storage word size must integer-divide bit width."
            )
        if isinstance(pf_map, dict):
            for nb, factor in pf_map.items():
                if factor not in _ALLOWED_PACKING_FACTORS:
                    raise ValueError(
                        f"quant_schemes/{scheme_name}.yaml: "
                        f"packing_factor_per_num_bits[{nb}] = {factor} not in "
                        f"{sorted(_ALLOWED_PACKING_FACTORS)}."
                    )

        sd = vraw.get("storage_dtype")
        sd_map = vraw.get("storage_dtype_per_num_bits")
        if sd is not None and sd_map is not None:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml format_variants[{variant_name!r}]: "
                f"set EITHER `storage_dtype` OR `storage_dtype_per_num_bits`, not both."
            )

        out[variant_name] = FormatVariant(
            name=variant_name,
            packing_factor=pf,
            packing_factor_per_num_bits=(
                {int(k): int(v) for k, v in pf_map.items()}
                if isinstance(pf_map, dict) else None
            ),
            storage_dtype=sd,
            storage_dtype_per_num_bits=(
                {int(k): str(v) for k, v in sd_map.items()}
                if isinstance(sd_map, dict) else None
            ),
            weight_storage_suffix=str(vraw.get("weight_storage_suffix", ".weight")),
        )
    return out


#: What a metadata sibling can be. `scale` and `zero_point` dequantize the
#: weight; `index` is GPTQ's act-order permutation; `shape` is the packed
#: weight's logical shape written beside it; `activation_scale` is a static
#: input scale, quantization metadata that is not about the weight.
SIBLING_KINDS: Tuple[str, ...] = (
    "scale", "zero_point", "index", "shape", "activation_scale")


def _parse_metadata_siblings(raw: Any, scheme_name: str) -> Tuple[MetadataSibling, ...]:
    """Parse the `metadata_siblings:` block. Each entry MUST declare a
    `suffix`, a `kind` from `SIBLING_KINDS`, and EITHER `cardinality` OR
    `cardinality_per_strategy`.
    """
    out: List[MetadataSibling] = []
    if not isinstance(raw, list):
        return tuple(out)
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        suffix = entry.get("suffix")
        if not isinstance(suffix, str) or not suffix:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml metadata_siblings: "
                f"each entry must have a non-empty `suffix:` field."
            )
        card = entry.get("cardinality")
        card_strat = entry.get("cardinality_per_strategy")
        shape = entry.get("shape")
        shape_strat = entry.get("shape_per_strategy")
        # A sibling may declare a SHAPE or a scalar count, and per strategy it
        # may declare one of each — `compressed-tensors` blocks per tile in two
        # dimensions and per group in one, so the tile form is a shape and the
        # group form is still a count. What it may not do is declare the same
        # thing twice.
        if card is None and card_strat is None and shape is None \
                and shape_strat is None:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml metadata_siblings[{suffix!r}]: "
                f"declare a `shape:` / `shape_per_strategy:` (one expression "
                f"per axis, the count is their product) or a `cardinality:` / "
                f"`cardinality_per_strategy:` (a single count)."
            )
        if card is not None and card_strat is not None:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml metadata_siblings[{suffix!r}]: "
                f"set ONE of `cardinality:` / `cardinality_per_strategy:`, not both."
            )
        if shape is not None and shape_strat is not None:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml metadata_siblings[{suffix!r}]: "
                f"set ONE of `shape:` / `shape_per_strategy:`, not both."
            )
        overlap = set(card_strat or ()) & set(shape_strat or ())
        if overlap:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml metadata_siblings[{suffix!r}]: "
                f"strategy {sorted(overlap)} declared as both a shape and a "
                f"count; a shape already determines the count."
            )
        kind = entry.get("kind")
        if kind not in SIBLING_KINDS:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml metadata_siblings[{suffix!r}]: "
                f"`kind:` must be one of {list(SIBLING_KINDS)}, got {kind!r}."
            )
        out.append(MetadataSibling(
            suffix=str(suffix),
            kind=str(kind),
            cardinality=str(card) if card is not None else None,
            cardinality_per_strategy=(
                {str(k): str(v) for k, v in card_strat.items()}
                if isinstance(card_strat, dict) else None
            ),
            shape=([str(x) for x in shape] if isinstance(shape, list) else None),
            shape_per_strategy=(
                {str(k): [str(x) for x in v] for k, v in shape_strat.items()}
                if isinstance(shape_strat, dict) else None
            ),
            dtype=(str(entry["dtype"]) if "dtype" in entry else None),
            logical_dtype=(str(entry["logical_dtype"]) if entry.get("logical_dtype") else None),
            dtype_per_format=(
                {str(k): str(v) for k, v in entry["dtype_per_format"].items()}
                if isinstance(entry.get("dtype_per_format"), dict) else None
            ),
            when=(str(entry["when"]) if entry.get("when") else None),
        ))
    return tuple(out)


def _parse_numeric_format_resolver(raw: Any, scheme_name: str) -> Tuple[Tuple[str, str], ...]:
    """Parse `numeric_format_resolver:` — list of {when, format} rules,
    evaluated in order, first match wins.
    """
    out: List[Tuple[str, str]] = []
    if not isinstance(raw, list):
        return tuple(out)
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        when = entry.get("when")
        fmt = entry.get("format")
        if not when or not fmt:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml numeric_format_resolver: "
                f"each rule requires both `when:` (predicate) and `format:` (target)."
            )
        out.append((str(when), str(fmt)))
    return tuple(out)


def _validate_status_list(raw: Any, scheme_name: str) -> Tuple[str, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"quant_schemes/{scheme_name}.yaml valid_status: must be a list."
        )
    for s in raw:
        if s not in _KNOWN_QUANT_STATUSES:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml valid_status: "
                f"{s!r} not in {sorted(_KNOWN_QUANT_STATUSES)}."
            )
    return tuple(str(s) for s in raw)


def _load_quant_schemes() -> Dict[str, QuantScheme]:
    """Load every YAML in `backend/arch/quant_schemes/`. The directory
    holds one file per `quant_method` known to the analyzer; adding support
    for a new library is one yaml file with no Python changes.

    The file stem IS the method name — it is the registry key, and it is
    what `config.quantization_config.quant_method` (or a spec's
    `quant_context.per_role.method`) is looked up under. So the file is
    named `fp8_ue8m0.yaml` and nothing inside it repeats that.

    Schema:

      num_bits: { fixed | field_any | nested_field }
      badge: "FP8" | "W{num_bits}" | etc.

      format_variants:
        <variant_name>:
          packing_factor: <Literal[1,2,4,8,16]>             # OR
          packing_factor_per_num_bits: { 4: 8, 8: 4 }
          storage_dtype: <safetensors_dtype>                # OR
          storage_dtype_per_num_bits: { 4: I8, 8: I8 }
          weight_storage_suffix: ".weight"                  # default
      numeric_format: [int_sym, int_asym, fp_e2m1, ...]
      numeric_format_resolver:
        - { when: "<predicate>", format: <name> }
      scale_strategy: [group, channel, tensor, tile_2d]
      metadata_siblings:
        - suffix: ".weight_scale"
          when: "<predicate>"                               # optional
          cardinality: "<formula>"                          # OR
          cardinality_per_strategy: { group: "...", ... }
          dtype: F16                                         # OR
          dtype_per_format: { pack-quantized: F16, ... }
      required_fields: [num_bits, group_size, ...]
      optional_fields: [block_h, block_w, ...]                # optional
      valid_status: [compressed, frozen]
      default_fields: { <field>: <default_value>, ... }     # optional
      also_stores:                                          # optional
        - { field: <quantization_config key>, equals: <value>, scheme: <stem> }
    """
    # The yaml files live beside this module: the registry IS this package.
    schemes_dir = os.path.dirname(__file__)
    out: Dict[str, QuantScheme] = {}
    for fname in sorted(os.listdir(schemes_dir)):
        if not fname.endswith(".yaml"):
            continue
        path = os.path.join(schemes_dir, fname)
        with open(path) as f:
            data = load_yaml(f) or {}
        name = fname.removesuffix(".yaml")
        if "name" in data:
            raise ValueError(
                f"quant_schemes/{fname} declares `name:`. The stem names the "
                f"scheme and is the key it registers under; a `name:` that "
                f"disagreed would register it under a key no file matches. "
                f"Delete the line."
            )

        # Materialization recipe parsing — all optional fields.
        format_variants = _parse_format_variants(data.get("format_variants") or {}, name)
        metadata_siblings = _parse_metadata_siblings(data.get("metadata_siblings") or [], name)
        numeric_format_resolver = _parse_numeric_format_resolver(
            data.get("numeric_format_resolver") or [], name,
        )
        valid_status = _validate_status_list(data.get("valid_status"), name)

        # Cross-check: every value in numeric_format must be reachable by
        # at least one resolver rule (else the engine can never select it).
        nf_list = data.get("numeric_format") or []
        nf_targets = {fmt for _, fmt in numeric_format_resolver}
        unreachable = sorted(set(nf_list) - nf_targets) if numeric_format_resolver else []
        if unreachable:
            raise ValueError(
                f"quant_schemes/{name}.yaml: numeric_format values "
                f"{unreachable!r} have no matching numeric_format_resolver rule. "
                f"Either add a rule that emits them, or drop them from "
                f"numeric_format."
            )

        defaults_raw = data.get("default_fields") or {}
        if not isinstance(defaults_raw, dict):
            raise ValueError(
                f"quant_schemes/{name}.yaml: default_fields must be a dict."
            )

        out[name] = QuantScheme(
            name=name,
            read_num_bits=_build_num_bits_resolver(data.get("num_bits") or {}),
            badge=str(data.get("badge") or ""),
            format_variants=format_variants,
            numeric_format=tuple(str(x) for x in nf_list),
            numeric_format_resolver=numeric_format_resolver,
            scale_strategy=tuple(str(x) for x in (data.get("scale_strategy") or [])),
            metadata_siblings=metadata_siblings,
            required_fields=tuple(str(x) for x in (data.get("required_fields") or [])),
            optional_fields=tuple(str(x) for x in (data.get("optional_fields") or [])),
            valid_status=valid_status,
            default_fields=tuple(sorted(defaults_raw.items())),
            also_stores=_parse_also_stores(data.get("also_stores") or [], name),
            deployment_fields=tuple(sorted(
                (data.get("deployment_fields") or {}).items())),
            activation_options=_parse_activation_options(
                data.get("activation_options") or {}, name),
            activation=_parse_activation(data.get("activation"), name),
        )
    for name, scheme in out.items():
        for _field, _value, other in scheme.also_stores:
            if other not in out:
                raise ValueError(
                    f"quant_schemes/{name}.yaml: also_stores names scheme "
                    f"{other!r}, which no file in quant_schemes/ registers.")
    return out


@dataclass(frozen=True)
class ActivationDeclaration:
    """What a checkpoint's own quantiser does to the activation operand.

    `from_config` names the config field that says which treatment the
    checkpoint uses (`activation_scheme` for FineGrainedFP8, whose only HF
    value is `dynamic`); the spec's `quant_context` entry states it, and
    the scheme's `default_fields` carry the value HF defaults to. Each
    treatment maps to ordered `{when, format}` rules over the same bindings
    and the same predicate dialect as `numeric_format_resolver`; `none`
    means the kernel takes the edge as it arrives.

    A treatment the declaration does not cover is a refusal, not a guess:
    pricing a static per-tensor quantiser as a dynamic per-group one would
    be a silent wrong answer.
    """

    #: The config field naming the treatment, None when the scheme has only
    #: one (a repo's reference implementation, not a config switch).
    from_config: Optional[str]
    #: Treatment name -> ordered (when, format-or-None) rules.
    treatments: Tuple[Tuple[str, Tuple[Tuple[str, Optional[str]], ...]], ...]

    def resolve(self, bindings: Mapping[str, Any], scheme_name: str
                ) -> Optional[str]:
        rules = dict(self.treatments)
        if self.from_config is None:
            (treatment, _), = self.treatments[:1] or ((None, ()),)
        else:
            stated = bindings.get(self.from_config)
            treatment = None if stated is None else str(stated)
        if treatment is None:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml declares `activation` "
                f"read off `{self.from_config}`, and this model's "
                f"quant_context entry states no {self.from_config!r}. "
                f"Declare it (one of {sorted(rules)}).")
        if treatment not in rules:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml: {self.from_config}="
                f"{treatment!r} is not one of the treatments it declares "
                f"({sorted(rules)}); add the treatment with the activation "
                f"format its kernel quantises to.")
        for when, fmt in rules[treatment]:
            if bool(eval_predicate(when, dict(bindings))):
                return fmt
        raise ValueError(
            f"quant_schemes/{scheme_name}.yaml: no `activation` rule of "
            f"{self.from_config}={treatment!r} matches this model's "
            f"bindings; the rules must cover every checkpoint the scheme "
            f"describes.")


def _parse_activation(raw: Any, scheme_name: str
                      ) -> Optional[ActivationDeclaration]:
    """Parse `activation:` — `from_config:` plus one list of
    `{when, format}` rules per treatment."""
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"quant_schemes/{scheme_name}.yaml: `activation` must be a "
            f"mapping of `from_config` plus one treatment per key.")
    body = dict(raw)
    from_config = body.pop("from_config", None)
    treatments: List[Tuple[str, Tuple[Tuple[str, Optional[str]], ...]]] = []
    for treatment, rules in body.items():
        if not isinstance(rules, list) or not rules:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml: activation treatment "
                f"{treatment!r} must be a non-empty list of "
                f"{{when, format}} rules.")
        parsed: List[Tuple[str, Optional[str]]] = []
        for rule in rules:
            if not isinstance(rule, dict) or "when" not in rule \
                    or "format" not in rule:
                raise ValueError(
                    f"quant_schemes/{scheme_name}.yaml: each activation "
                    f"rule is {{when, format}}; got {rule!r}.")
            fmt = rule["format"]
            fmt = None if fmt in (None, "none") else str(fmt)
            if fmt is not None and fmt not in ACTIVATION_FORMATS:
                raise ValueError(
                    f"quant_schemes/{scheme_name}.yaml: activation rule "
                    f"names {fmt!r}, which is not an activation format "
                    f"({sorted(ACTIVATION_FORMATS)}).")
            parsed.append((str(rule["when"]), fmt))
        treatments.append((str(treatment), tuple(parsed)))
    if not treatments:
        raise ValueError(
            f"quant_schemes/{scheme_name}.yaml: `activation` declares no "
            f"treatment.")
    if from_config is None and len(treatments) > 1:
        raise ValueError(
            f"quant_schemes/{scheme_name}.yaml: `activation` declares "
            f"several treatments and no `from_config:` to choose between "
            f"them.")
    return ActivationDeclaration(
        from_config=None if from_config is None else str(from_config),
        treatments=tuple(treatments))


def _parse_activation_options(raw: Any, scheme_name: str
                               ) -> Tuple[Tuple[str, Optional[str]], ...]:
    """Parse `activation_options:` — option name → activation format name,
    or `none` for an operand the kernel takes as it arrives."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"quant_schemes/{scheme_name}.yaml: activation_options must be "
            f"a mapping of option name to activation format.")
    out = []
    for option, fmt in raw.items():
        fmt = None if fmt in (None, "none") else str(fmt)
        if fmt is not None and fmt not in ACTIVATION_FORMATS:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml: activation option "
                f"{option!r} names {fmt!r}, which is not an activation "
                f"format ({sorted(ACTIVATION_FORMATS)}).")
        out.append((str(option), fmt))
    return tuple(out)


def _parse_also_stores(raw: Any, scheme_name: str) -> Tuple[Tuple[str, str, str], ...]:
    """Parse `also_stores:` — `{field, equals, scheme}` rows."""
    if not isinstance(raw, list):
        raise ValueError(
            f"quant_schemes/{scheme_name}.yaml: also_stores must be a list.")
    out = []
    for entry in raw:
        if not isinstance(entry, dict) or set(entry) != {"field", "equals", "scheme"}:
            raise ValueError(
                f"quant_schemes/{scheme_name}.yaml: each also_stores row is "
                f"exactly {{field, equals, scheme}}; got {entry!r}.")
        out.append((str(entry["field"]), str(entry["equals"]), str(entry["scheme"])))
    return tuple(out)


QUANT_SCHEMES = _load_quant_schemes()


# The scheme of a checkpoint whose config names no known `quant_method`:
# no sidecars and no packing. Which tensors are quantization metadata, and
# which are packed, are facts of the scheme that wrote the file, so a file
# with no selected scheme has none of either; an unrecognised method is
# reported (`unknown_method`), not guessed from tensor names.
SCHEME_NONE = QuantScheme(name="none")


def _normalize_quant_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Find and normalize the `quantization_config` block from an HF
    config dict. Handles all the in-the-wild shape variations:

      • Top-level: `config.quantization_config` (most quantized LLMs)
      • VLM-nested: `config.text_config.quantization_config`
        (Kimi-K2.5 stores it there; vision_config / language_config
        also checked for completeness)
      • JSON-encoded string instead of nested dict (DeepSeek-V4-Pro
        ships it that way — `quant_method`/`fmt`/`scale_fmt`/etc.
        wrapped in a JSON string literal)

    Returns the resolved dict (possibly empty) — callers can safely
    `.get(...)` without further nesting / shape checks.
    """
    if not isinstance(config, dict):
        return {}
    qc = config.get("quantization_config")
    if not qc:
        for nested_key in ("text_config", "vision_config", "language_config"):
            sub = config.get(nested_key)
            if isinstance(sub, dict) and sub.get("quantization_config"):
                qc = sub["quantization_config"]
                break
    if isinstance(qc, str):
        try:
            import json as _json
            parsed = _json.loads(qc)
            qc = parsed if isinstance(parsed, dict) else {}
        except _json.JSONDecodeError:
            qc = {}
    return qc if isinstance(qc, dict) else {}


def resolve_scheme(config: Optional[Dict[str, Any]]) -> tuple:
    """Pick a QuantScheme + `num_bits` from an HF config dict.

    Returns `(scheme, num_bits, unknown_method)`. The third element is
    `None` when the scheme was selected unambiguously, or the literal
    `quant_method` string when the config declared a method we don't
    recognize and the checkpoint is read under `SCHEME_NONE`. Callers should
    surface that as a warning so the "exact match" comparison between
    spec engine totals and HF safetensors totals doesn't silently
    succeed on a coincidence (the gpt-oss MXFP4 case before this
    scheme was registered: both sides undercounted by the same factor
    and the "exact match" badge lit up on a wrong number).

    The selected scheme carries the metadata siblings of every scheme its
    `also_stores` rows select from this config.
    """
    qc = _normalize_quant_config(config)
    if not qc:
        return SCHEME_NONE, 0, None

    method = str(qc.get("quant_method") or "").strip().lower()
    if method and method not in QUANT_SCHEMES:
        return SCHEME_NONE, 0, method
    scheme = QUANT_SCHEMES.get(method, SCHEME_NONE)
    also = [QUANT_SCHEMES[other] for key, value, other in scheme.also_stores
            if str(qc.get(key)) == value]
    return _with_layouts(scheme, also), scheme.read_num_bits(qc), None


def _with_layouts(scheme: QuantScheme, also) -> QuantScheme:
    """`scheme` carrying the metadata siblings of the `also_stores` layouts
    its config selected, named `<scheme>+<layout>…`."""
    if not also:
        return scheme
    return replace(
        scheme,
        name="+".join([scheme.name, *(s.name for s in also)]),
        metadata_siblings=scheme.metadata_siblings + tuple(
            sib for s in also for sib in s.metadata_siblings))


def scheme_named(name: str) -> QuantScheme:
    """The scheme `resolve_scheme` returned under `name` — what a snapshot
    records of its writer's resolution, read back without the config."""
    if name == SCHEME_NONE.name:
        return SCHEME_NONE
    base, *also = name.split("+")
    return _with_layouts(QUANT_SCHEMES[base], [QUANT_SCHEMES[a] for a in also])


def is_quantization_metadata(name: str, scheme: QuantScheme) -> bool:
    """Return True when a tensor is on-disk quantization metadata (a
    scale / zero-point / packed-shape / column-permute sidecar) that
    should be excluded from HF-comparable `total_params`.

    Driven by `scheme.metadata_siblings` — each entry's `suffix`
    (e.g. `.weight_scale_inv` for fp8, `_scales` for mxfp4,
    `.weight_scale`/`.weight_zero_point`/`.weight_shape` for
    compressed-tensors) classifies a tail as quant metadata.
    """
    return any(name.endswith(sib.suffix) for sib in scheme.metadata_siblings)


def packed_unpack_factor(name: str, dtype: str, num_bits: int,
                          scheme: QuantScheme) -> int:
    """Elements-per-packed-unit factor for `name` under `scheme`.

    Driven by `scheme.format_variants[*].weight_storage_suffix` — find
    the variant whose suffix AND storage dtype match this tensor, then
    apply its `packing_factor` (or `packing_factor_per_num_bits[num_bits]`).
    Returns 1 when no variant declares packing for this name. A variant that
    packs by bit width and has no entry for `num_bits` refuses: the config
    did not state a width this scheme packs at, and 1 would undercount.

    Matching the suffix alone is insufficient for mixed checkpoints: a
    scheme may store packed U8 under `.weight` alongside unquantized BF16
    tensors using that same suffix. The header's dtype distinguishes them;
    the scheme remains the sole owner of the unpacking factor.
    """
    if not scheme.format_variants:
        return 1
    for fv in scheme.format_variants.values():
        if not name.endswith(fv.weight_storage_suffix):
            continue
        if fv.packing_factor == 1:
            continue
        storage_dtype = fv.storage_dtype
        if fv.storage_dtype_per_num_bits:
            storage_dtype = fv.storage_dtype_per_num_bits.get(num_bits)
            if storage_dtype is None:
                if canonical(dtype) not in {
                        canonical(value)
                        for value in fv.storage_dtype_per_num_bits.values()}:
                    continue
                raise ValueError(
                    f"{name}: scheme {scheme.name!r} declares storage dtypes "
                    f"per bit width {sorted(fv.storage_dtype_per_num_bits)}, "
                    f"and the quantization config states num_bits={num_bits}")
        if storage_dtype is None or canonical(dtype) != canonical(storage_dtype):
            continue
        if fv.packing_factor_per_num_bits:
            if num_bits not in fv.packing_factor_per_num_bits:
                raise ValueError(
                    f"{name}: scheme {scheme.name!r} packs "
                    f"`*{fv.weight_storage_suffix}` per bit width "
                    f"{sorted(fv.packing_factor_per_num_bits)}, and the "
                    f"quantization config states num_bits={num_bits}")
            pf = fv.packing_factor_per_num_bits[num_bits]
        else:
            pf = fv.packing_factor
        if pf and pf > 1:
            return pf
        # Variant matches but doesn't pack (e.g. fp8 default `.weight`
        # with packing_factor=1) — keep searching for a better-matching
        # variant before giving up.
    return 1
