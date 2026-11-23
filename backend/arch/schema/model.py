"""ModelLayer / BlockAssignment — full model structure over block templates."""
from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import (Any, Callable, Dict, List, Literal, Optional,
                    Tuple, Union)

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

import backend.arch.dims as dim_registry
import backend.arch.roles as roles

from .weights import (StoredSplitSpec, StoredWithoutGraphReaderSpec, WeightAxisSpec,
                      ExpertWeight, check_stored_axes, check_stored_split)

#: Which model boundary a node is: the embedding the tokens enter through, or
#: a logits projection. Declared once; a model layer and a graph node (an
#: auxiliary stack's own head) both take their value from it.
BoundaryRole = Literal["output_head", "input_embedding"]



#: `i - <k>`: the layer `k` iterations before the reading one, in its stack.
_RELATIVE_LAYER = re.compile(r"^\s*i\s*-\s*(?P<k>.+?)\s*$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_READS_FROM_FIX = (
    "a `reads_from` value is one layer number of the stack — an integer or a "
    "dim expression (`\"20\"`, `\"n_dense_layers - 1\"`), `\"i - k\"` for "
    "the layer k before the reading one, `{stack: <stack>, layer: <number>}` "
    "for another stack's layer, or `{nearest: {from: config.<list>, is: "
    "<value>}}` / `{nearest: {from: config.<list of layer numbers>}}` for "
    "the nearest earlier layer of a set the config lists, or `{map: "
    "{<reading layer>: <source layer>, …}}` for each reading layer's source "
    "written out; a layer that reads the values of several layers at once "
    "reads an appended stream (`from:` + `after` / `before`, read with "
    "`read: concat` or `read: all`). In a looped stack (`loop:`) a layer "
    "number names a stored layer, read at its latest execution before the "
    "reader; `{pass: n}` (alone, or with `nearest:` / `layer:`) reads its "
    "pass-n execution")


def _names_i(text: str) -> bool:
    return "i" in _IDENT.findall(text)


class StreamLayerRef(BaseModel):
    """`{stack, layer}`: the layer of ANOTHER stack whose write an inner
    stream's reader reads. `stack` is that stack's path from the top of
    `model:`, its layer names joined by `/` (`layers`, `mtp/layers`);
    `layer` is one iteration number of it, an integer or a dim expression."""

    model_config = ConfigDict(extra="forbid")

    stack: str
    layer: str

    @field_validator("layer", mode="before")
    @classmethod
    def _layer_number(cls, value: Any) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        if not isinstance(value, str) or not value.strip() \
                or _names_i(value):
            raise ValueError(
                f"`reads_from` {{stack, layer}}: layer {value!r} is not one "
                f"layer number; {_READS_FROM_FIX}")
        return value.strip()

    @field_validator("stack")
    @classmethod
    def _stack_path(cls, value: str) -> str:
        if not value.strip() or any(not part.strip()
                                    for part in value.split("/")):
            raise ValueError(
                f"`reads_from` {{stack, layer}}: stack {value!r} names no "
                f"model layer; write its path from the top of `model:`, "
                f"layer names joined by `/`")
        return value.strip()


class StreamLayerSet(BaseModel):
    """A set of layers of the reading stack, as the config lists them:
    `{from: config.<list>, is: <value>}` — the layers whose entry in a
    per-layer list is `<value>`, the selector `per_layer.assign` uses — or
    `{from: config.<list>}`, a list of layer numbers (one number is a set of
    one; `base: 1` for a list that counts layers from 1). `?? absent` lets
    a config omit the list: no layers. A `per_layer.assign` generator's
    `within:` is one too: the generator selects only layers in the set."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True,
                              serialize_by_alias=True)

    from_: str = Field(alias="from")
    is_: Optional[str] = Field(default=None, alias="is",
                               exclude_if=lambda v: v is None)
    base: Optional[int] = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def _base_numbers_layers(self) -> "StreamLayerSet":
        if self.base is not None and (self.is_ is not None or self.base < 0):
            raise ValueError(
                f"{{from: {self.from_}}}: `base` is the non-negative number a "
                f"list of layer numbers gives the first layer; it goes with "
                f"`from` alone (no `is`)")
        return self

    @field_validator("is_", mode="before")
    @classmethod
    def _value(cls, value: Any) -> Any:
        return None if value is None else str(value)

    @field_validator("from_")
    @classmethod
    def _config_path(cls, value: str) -> str:
        path, _, fallback = value.partition("??")
        if not path.strip().startswith("config.") or (
                fallback.strip() not in ("", "absent")):
            raise ValueError(
                f"`reads_from` {{nearest: {{from: {value!r}}}}} names a "
                f"config list: `config.<path>`, optionally `?? absent`")
        return value.strip()

    @property
    def config_list(self) -> str:
        return self.from_.partition("??")[0].strip()[len("config."):]

    @property
    def config_list_optional(self) -> bool:
        return self.from_.partition("??")[2].strip() == "absent"

    @property
    def equals(self) -> Optional[str]:
        return self.is_

    def __str__(self) -> str:
        return (f"{self.from_} is {self.is_}" if self.is_ is not None
                else self.from_)


class NearestLayer(BaseModel):
    """`{nearest: <set>}`: the nearest layer of `<set>` before the reading
    one, in its stack (`StreamLayerSet`)."""

    model_config = ConfigDict(extra="forbid")

    nearest: StreamLayerSet


class LayerMap(BaseModel):
    """`{map: ...}`: each reading layer's source, written out — the general
    form the others are shorthand for. Either `{<reading layer>: <source>}`
    over exactly the layers the declaration selects, a source being a layer
    number (an integer or a dim expression) or `{stack, layer}`; or
    `config.<path>`, a list of layer numbers aligned with the selected
    layers in ascending order."""

    model_config = ConfigDict(extra="forbid")

    map: Union[str, Dict[str, Union[str, StreamLayerRef]]]

    @field_validator("map", mode="before")
    @classmethod
    def _entries(cls, value: Any) -> Any:
        if isinstance(value, str):
            path, _, fallback = value.partition("??")
            if not path.strip().startswith("config.") or fallback.strip():
                raise ValueError(
                    f"`reads_from` {{map: {value!r}}}: a map is "
                    f"`{{<reading layer>: <source layer>}}` or "
                    f"`config.<path>`, a list of source layers aligned with "
                    f"the selected layers")
            return value.strip()
        if not isinstance(value, dict) or not value:
            raise ValueError(
                "`reads_from` {map: …} maps each reading layer to the layer "
                "whose write it reads")
        out: Dict[str, Any] = {}
        for reader, source in value.items():
            if isinstance(source, int) and not isinstance(source, bool):
                source = str(source)
            if isinstance(source, str) and (not source.strip()
                                            or _names_i(source)):
                raise ValueError(
                    f"`reads_from` {{map: {{{reader}: {source!r}}}}}: a "
                    f"source in a map is one layer number or "
                    f"`{{stack, layer}}`")
            out[str(reader).strip()] = source
        return out

    def __str__(self) -> str:
        if isinstance(self.map, str):
            return f"map {self.map}"
        return "map {" + ", ".join(
            f"{k}: {v.stack} {v.layer}" if isinstance(v, StreamLayerRef)
            else f"{k}: {v}" for k, v in self.map.items()) + "}"


class PassRead(BaseModel):
    """`{pass: n}`, optionally with `nearest:` or `layer:`: in a looped
    stack (`LoopSpec`), the pass-n execution of a stored layer. `{pass: n}`
    alone names the reader's own stored layer; with `nearest` or `layer`
    (a layer number or `"i - k"`) the stored layer that form names. Without
    `pass:` a stored layer's latest execution before the reader is read."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True,
                              serialize_by_alias=True)

    pass_: int = Field(alias="pass")
    nearest: Optional[StreamLayerSet] = Field(
        default=None, exclude_if=lambda v: v is None)
    layer: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)

    @field_validator("layer", mode="before")
    @classmethod
    def _layer_number(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        relative = (_RELATIVE_LAYER.fullmatch(value)
                    if isinstance(value, str) else None)
        if not isinstance(value, str) or not value.strip() or (
                _names_i(value) and (relative is None
                                     or _names_i(relative["k"]))):
            raise ValueError(
                f"`reads_from` {{pass, layer}}: layer {value!r} is not one "
                f"layer number or `\"i - k\"`; {_READS_FROM_FIX}")
        return value.strip()

    @model_validator(mode="after")
    def _one_layer(self) -> "PassRead":
        if self.pass_ < 1:
            raise ValueError(
                f"`reads_from` {{pass: {self.pass_}}}: passes count from 1")
        if self.nearest is not None and self.layer is not None:
            raise ValueError(
                "`reads_from` {pass, …} names one stored layer: `nearest:` "
                "or `layer:`, not both")
        return self

    def __str__(self) -> str:
        where = (f", nearest {self.nearest}" if self.nearest is not None
                 else f", layer {self.layer}" if self.layer is not None
                 else "")
        return f"pass {self.pass_}{where}"


#: One reader declaration's sources, stream name → layer.
ReadsFrom = Dict[str, Union[str, StreamLayerRef, NearestLayer, LayerMap,
                            PassRead]]


def check_reads_from(value: Any) -> Optional[ReadsFrom]:
    """`reads_from:` as written: stream name → one layer number (an integer
    or a dim expression), `i - k`, `{stack, layer}`, `{nearest: <set>}`, or
    `{map: …}`.
    Anything else names no single layer and is refused with what to write
    instead."""
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"`reads_from: {value!r}` maps each inner stream this "
            f"declaration's layers read to the layer whose write they read; "
            f"{_READS_FROM_FIX}")
    out: Dict[str, Any] = {}
    for name, layer in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"`reads_from` keys are stream names, got {name!r}")
        if isinstance(layer, int) and not isinstance(layer, bool):
            layer = str(layer)
        if isinstance(layer, (StreamLayerRef, NearestLayer, LayerMap,
                              PassRead)):
            out[name] = layer
            continue
        if isinstance(layer, dict):
            out[name] = (PassRead(**layer) if "pass" in layer
                         else NearestLayer(**layer) if "nearest" in layer
                         else LayerMap(**layer) if "map" in layer
                         else StreamLayerRef(**layer))
            continue
        if not isinstance(layer, str) or not layer.strip():
            raise ValueError(
                f"`reads_from[{name}]: {layer!r}`: {_READS_FROM_FIX}")
        relative = _RELATIVE_LAYER.fullmatch(layer)
        if _names_i(layer) and (relative is None
                                or _names_i(relative["k"])):
            raise ValueError(
                f"`reads_from[{name}]: {layer!r}`: {_READS_FROM_FIX}")
        out[name] = layer.strip()
    return out


def relative_distance(layer: str) -> Optional[str]:
    """`k` of a `reads_from` value `i - k`, or None for a layer number."""
    found = _RELATIVE_LAYER.fullmatch(layer)
    return found["k"] if found is not None else None


class LayerSelection(BaseModel):
    """A set of layers of one stack, as a spec names it.

    The selection vocabulary, factored out of the one declaration that used
    to own it. `BlockAssignment` says which layers run a block template;
    `quant_context.overrides`' `select.layers` says which layers an override
    row speaks about. Both mean the same thing by `range:` / `layers:` /
    `stride`+`offset` / `config_list`+`equals`, and
    `ResolvedSpec.layers_selected` resolves either against the model's dims
    and config — one spelling, one resolver, so the two cannot drift.

    Every selector present is resolved independently and the results are
    INTERSECTED: a layer is selected only when it satisfies all of them.
    """

    # Layer selection (at least one required by the declarations that use it)
    range: Optional[List[str]] = None       # [start, end] inclusive
    layers: Optional[List[str]] = None      # explicit list of layer indices
    # Strided selection: every `stride`-th layer starting at `offset`
    # (0-based) over the group's `repeat` range. Expresses alternating
    # schedules whose PERIOD is fixed but layer COUNT varies with model
    # size — e.g. Gemma-2 alternates sliding (even) / full (odd) across
    # 26/42/46 layers, which neither an explicit `layers:` list nor a
    # contiguous `range:` can express size-independently. `offset`
    # defaults to 0. Both accept a dim expression or an integer literal.
    stride: Optional[str] = None
    offset: Optional[str] = None

    # Config-list selector: pick layers whose per-layer config-list value
    # equals `equals`. `config_list` names an HF config field holding a
    # per-layer list (e.g. "indexer_types", "mlp_layer_types"). It usually
    # runs length-`repeat`, but a shared schedule array may carry trailing
    # entries for a sibling stack (DeepSeek-V4's `compress_ratios` has one
    # extra for the MTP layer); selection and exhaustiveness clamp to the
    # container's own `range(repeat)`, so those out-of-span entries are the
    # sibling stack's business, not this one's.
    # Whether the field may be ABSENT is declared, not assumed:
    # `config_list_optional` (authored as `from: config.X ?? absent`) lets
    # an absent list select no layers and defer to a sibling `rest:` /
    # `range:` (GLM-5.1 has no indexer_types); without it an absent list is
    # an error, because a misspelt field name looks exactly like one.
    config_list: Optional[str] = None
    config_list_optional: bool = False
    equals: Optional[str] = None
    # With no `equals`, the list holds layer numbers; `config_list_base` is
    # the number it gives the first layer (1 for a 1-based list such as
    # Qwen3.8's `ple_layer_ids`). Left out of the dump while unset.
    config_list_base: Optional[int] = Field(
        default=None, exclude_if=lambda v: v is None)
    # A second config-listed set the selected layers must also lie in: a
    # layer whose kind is the product of two config lists (Qwen3.8's
    # `layer_types` × `ple_layer_ids`).
    within: Optional[StreamLayerSet] = Field(
        default=None, exclude_if=lambda v: v is None)

    @property
    def selects_layers(self) -> bool:
        """Whether any selector is present — a selection that names none
        selects nothing, which every user of this refuses in its own words."""
        return any(value is not None for value in (
            self.range, self.layers, self.stride, self.config_list,
            self.within))

    def selection_text(self) -> str:
        """The selection as the author wrote it, for diagnostics."""
        parts = []
        if self.range:
            parts.append(f"range: [{self.range[0]}, {self.range[1]}]")
        if self.layers:
            parts.append(f"layers: [{', '.join(str(l) for l in self.layers)}]")
        if self.stride is not None:
            parts.append(f"stride: {self.stride}"
                         + (f", offset: {self.offset}"
                            if self.offset is not None else ""))
        if self.config_list is not None:
            parts.append(f"config_list: {self.config_list}"
                         + (f", equals: {self.equals}"
                            if self.equals is not None else ""))
        if self.within is not None:
            parts.append(f"within: {{{self.within}}}")
        return ", ".join(parts)


class BlockAssignment(LayerSelection):
    """Assigns a range of layer indices to a block template.

    Modes:
      • template + params: invoke a parameterized compound op (e.g.
        `template: pre_norm, params: {attn: <name>, ffn: <name>}`).
      • template alone: reference a pre-built block (inline `blocks:`).
      • attention/ffn shortcut: name only the slot this assignment CHANGES.
        The loader fills the rest from the spec's `components:` map — block
        pattern, the other slot, and `norm` when the target accepts it — so
        "these layers, but SWA" is `attention: swa_attention` and nothing
        else. Desugared at parse time, so no consumer sees two shapes.

    The shortcut is not a lesser form of the first mode; it is the one that
    inherits the spec-level defaults, which is why 32 of 38 models use it.
    Spelling it out restates bindings the spec already made, and a restated
    binding is one that can disagree.
    """
    # Layer selection comes from `LayerSelection` (range / layers / stride +
    # offset / config_list + equals / within), so an assignment's authored
    # YAML is unchanged.
    rest: Optional[bool] = None             # all layers not covered by other assignments
    # In a looped stack (`LoopSpec`): only the executions of the selected
    # stored layers in this pass. Without it, every execution of them.
    # Left out of the dump while unset.
    pass_: Optional[int] = Field(default=None, alias="pass",
                                 serialization_alias="pass",
                                 exclude_if=lambda v: v is None)

    # Reference a block template (component name or inline block).
    template: Optional[str] = None

    # Slot shortcut: desugars to template=<spec's components.block> +
    # params={attn, ffn, norm}, each filled from `components:` when omitted.
    attention: Optional[str] = None
    ffn: Optional[str] = None

    # Caller-side param bindings forwarded to the templated component.
    # See GraphNode.params for the dict-valued nested-call semantics.
    params: Dict[str, Any] = {}

    # Per-assignment dimension overrides
    # (e.g. {n_kv_heads: flash_kv_shards} for SWA layers)
    dims: Optional[Dict[str, str]] = None

    # Where the selected layers read each inner stream their components
    # read: stream → the layer of the stack whose write they read (see
    # `StreamSpec`, "Inner"). Left out of the dump while unset.
    reads_from: Optional[ReadsFrom] = Field(
        default=None, exclude_if=lambda v: v is None)

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("reads_from", mode="before")
    @classmethod
    def _reads_from(cls, value: Any) -> Optional[ReadsFrom]:
        return check_reads_from(value)

    @field_validator("pass_")
    @classmethod
    def _pass_counts_from_one(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError(f"`pass: {value}`: passes count from 1")
        return value

    @field_validator("dims")
    @classmethod
    def _writes_no_derived_name(cls, v: Optional[Dict[str, str]]
                                ) -> Optional[Dict[str, str]]:
        dim_registry.refuse_written_derived(v, "block assignment `dims:`")
        return v



class CheckpointIndexAfter(BaseModel):
    """An aux stack whose local index follows another stack's index space."""

    model_config = ConfigDict(extra="forbid")

    after: Literal["backbone"]



class CheckpointAddress(BaseModel):
    """Where a model entry's tensors live in the checkpoint, from the file root.

    One declaration for every place a path is decided, each field for one kind
    of entry:

      * ``prefix`` — an indexed stack: the literal dotted path before the
        layer index. ``index`` says whether the stack owns a local index space
        or appends after the backbone's. Keeping these two facts separate
        avoids an enum whose values mix path spelling with index arithmetic.
        A backbone stack that declares none lives at its own name under the
        scope root, the way an op does.
      * ``root`` — a scope (the model, or a group): the path its entries'
        names hang off. Absent inherits the enclosing scope's; the model's is
        empty unless declared.
      * ``path`` — one op: its module's full path, for a module that does
        not sit under its scope's root (``lm_head`` beside ``model.*``).
      * ``index: local`` on an op or a repeated graph node: the checkpoint
        writes the copy index right after this module's segment
        (``speech_embeddings.<i>``, ``encoder.blocks.<i>``). A repeat with no
        such declaration writes no index.
      * ``prefix`` with ``path`` on a repeated op: one copy under each index
        of the stack at ``prefix``, at the module ``path`` inside it (a
        per-layer buffer the checkpoint serialises inside each layer:
        ``model.layers.<i>.self_attn.rotary_emb``).
    """

    model_config = ConfigDict(extra="forbid")

    # An undeclared field is not serialized, so a spec's dump (and its
    # fingerprint) carries only what the spec wrote.
    prefix: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    index: Optional[Union[Literal["local"], CheckpointIndexAfter]] = Field(
        default=None, exclude_if=lambda v: v is None)
    root: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    path: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)

    @field_validator("prefix", "path")
    @classmethod
    def _valid_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return _dotted_path(value, empty_ok=False)

    @field_validator("root")
    @classmethod
    def _valid_root(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return _dotted_path(value, empty_ok=True)

    @property
    def follows_backbone(self) -> bool:
        """Whether this stack appends to the backbone's index space."""
        return (isinstance(self.index, CheckpointIndexAfter)
                and self.index.after == "backbone")

    def only(self, *fields: str, where: str) -> "CheckpointAddress":
        """This address, refused when it sets a field `where` has no use for."""
        stray = sorted(set(self.model_dump(exclude_none=True)) - set(fields))
        if stray:
            raise ValueError(
                f"{where}: `checkpoint:` declares {stray}; here it takes "
                f"only {sorted(fields)}")
        return self


def _dotted_path(value: str, *, empty_ok: bool) -> str:
    value = value.strip()
    if not value and empty_ok:
        return value
    segments = value.split(".")
    if (not value or any(not segment for segment in segments)
            or "X" in segments):
        raise ValueError(
            "a checkpoint path must be a dotted path without an index "
            "placeholder; `index:` owns the indexed position")
    return value



class CheckpointCopySpec(BaseModel):
    """The checkpoint carries the tied layer's own tensor as a redundant copy.

    The tie still decides compute, parameters and deployed weight bytes: the
    serving loader drops the copy (HunyuanOCR ships `lm_head.weight` beside
    `embed_tokens`, and vLLM skips `lm_head.` when `tie_word_embeddings`). The
    copy counts only where the FILE is the subject — the checkpoint byte
    total, the per-tensor match, the unmatched-tensor gate. `src` is the
    serving-loader line that skips it.
    """

    model_config = ConfigDict(extra="forbid")

    src: str

    @field_validator("src")
    @classmethod
    def _nonempty_src(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "tied_weight.checkpoint_copy.src must cite the serving-loader "
                "line that skips the stored copy (file:line)")
        return value


class TiedWeightSpec(BaseModel):
    """A weight alias: the earlier model layer it reuses and its condition."""

    model_config = ConfigDict(extra="forbid")

    to: str
    when: Optional[str] = None
    # Left out of the dump while unset, so a tie dumps as it did before the
    # field existed.
    checkpoint_copy: Optional[CheckpointCopySpec] = Field(
        default=None, exclude_if=lambda v: v is None)

    @field_validator("to")
    @classmethod
    def _nonempty_target(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tied_weight.to must name an earlier model layer")
        return value

    @field_validator("when")
    @classmethod
    def _valid_condition(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not re.fullmatch(
            r"config(?:\.[A-Za-z_][A-Za-z_0-9]*)+"
            r"(?:\s*\?\?\s*(?:true|false|1|0))?",
            value.strip(), flags=re.IGNORECASE,
        ):
            raise ValueError(
                "tied_weight.when must be exactly `config.<path>` with an "
                "optional `?? true|false|1|0` fallback")
        return value


def resolve_config_bool(config: Any, field_ref: str) -> bool:
    """Evaluate `tied_weight.when` (`config.path` with optional `?? true`).

    A missing key with no fallback is false; `?? true` is how Gemma-2
    records a Python default the published JSON omits.
    """
    if not field_ref:
        return False
    primary, separator, fallback = field_ref.partition("??")
    primary = primary.strip()
    if not primary.startswith("config."):
        return False
    obj: Any = config
    for part in primary[len("config."):].split("."):
        obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
        if obj is None:
            if not separator:
                return False
            return fallback.strip().lower() in {"true", "1"}
    return bool(obj)



_CONFIG_LIST_REF = re.compile(
    r"^config\.(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
    r"(?P<absent>\s*\?\?\s*absent)?$")


#: `after:` / `before:` naming every iteration rather than a list of them.
_EVERY = "every"

#: The dim symbol a `read: all` value's history axis is bound to: a stage
#: reading the prefix of such a stream sees `[B, S, Sar, D]`, and the derived
#: group that stands for its iterations binds `Sar` to the visible length.
#: One per stack (`ModelLayer` refuses a second `read: all`), so one symbol.
HISTORY_AXIS = "Sar"


class StreamSlice(BaseModel):
    """What an appended stream publishes of each value: the value from
    index `start` on along `axis` — `hs[:, 1:]` is `{axis: 1, start: 1}`,
    Llava's CLS crop. The same window the `slice` op takes, written by
    where it starts, because the extent it keeps is the value's own
    extent less `start`, which only the arriving tensor knows."""

    model_config = ConfigDict(extra="forbid")

    axis: int
    start: int

    @field_validator("start")
    @classmethod
    def _positive_start(cls, value: int) -> int:
        if value < 1:
            raise ValueError(
                "stream `slice.start` is where the published window starts; "
                "a window from 0 is the whole value, so write no `slice`")
        return value


#: What a model input is: token ids, an image, a video, or an audio clip.
#: The token ids are read by the language model; a media input is read by an
#: encoder group (`ArchSpec.encoder_groups`), and only there.
ModelInputKind = Literal["tokens", "image", "video", "audio"]

class ModelInputSpec(BaseModel):
    """One tensor the model receives from outside its graph (`model_inputs:`).

    The token ids are `[B, s_ctx]`: one id per prompt position, the step's
    length bound by the phase. A media input declares its own axes, the
    first the number of items per request and the rest one item's extents —
    an image `[n_img, img_h, img_w]` (the size the processor hands the tower,
    after resizing), an audio clip `[n_aud, s_audio]` (frames). The workload
    gives each axis a value (`Workload.media`); an axis the config fixes is
    written in `fixed:` and the workload may not give it. A media input whose
    items belong to the items of another (Step-3.7's crops, several per
    image) names that input in `per_item_of:`.

    The image's pixel channels are a config constant, part of the patch width
    the encoder's `patchify` reader declares; an audio frame's feature width
    is `width` (`[N, s_audio, width]`), or absent when each frame is one id
    (codebook ids, read by an embedding).
    """
    model_config = ConfigDict(extra="forbid")

    kind: ModelInputKind
    #: Each audio frame's feature width, a dim expression: the width the
    #: reading op takes in.
    width: Optional[str] = None
    #: A media input's axes: the item count per request, then one item's
    #: extents. Each name is a dim symbol the workload binds.
    axes: Tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    #: Axes the config fixes, `{axis: dim expression}` (Step-3.7's global
    #: view is always `image_size` square). Never the item count.
    fixed: Dict[str, str] = Field(default_factory=dict,
                                  exclude_if=lambda v: not v)
    #: Another media input whose every item holds this input's items.
    per_item_of: Optional[str] = Field(default=None,
                                       exclude_if=lambda v: v is None)
    #: A tensor the processor hands the model that this spec cannot walk
    #: yet, and why, citing the source: no layer reads it, and a workload
    #: that gives it any item is refused with this reason. It is still
    #: stated with every request (its item count, zero included), so a
    #: request is never taken to carry none of it by default. Only the
    #: items of another input (`per_item_of`) may be unmodeled.
    unmodeled: Optional[str] = Field(default=None,
                                     exclude_if=lambda v: v is None)

    @field_validator("kind", mode="before")
    @classmethod
    def _known_kind(cls, value: Any) -> Any:
        if value == "vision":
            raise ValueError(
                "`kind: vision` is now `kind: image`, declared with its axes "
                "`axes: [n_img, img_h, img_w]` (images per request, then the "
                "processed height and width) and read by a `patchify` node "
                "that cites the processor's patch size")
        return value

    @field_validator("width", mode="before")
    @classmethod
    def _width_as_expression(cls, value: Any) -> Any:
        return None if value is None else str(value)

    @field_validator("fixed", mode="before")
    @classmethod
    def _fixed_as_expressions(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _form(self) -> "ModelInputSpec":
        if self.kind == "tokens":
            if (self.axes or self.width or self.fixed or self.per_item_of
                    or self.unmodeled):
                raise ValueError(
                    "a `kind: tokens` input is the token ids, [B, s_ctx]: it "
                    "declares no `axes`, `width`, `fixed`, `per_item_of` or "
                    "`unmodeled`")
            return self
        # The item count per request, then one item's own axes.
        form = {"image": ("items", "height", "width"),
                "video": ("items", "frames", "height", "width"),
                "audio": ("items", "length")}[self.kind]
        if len(self.axes) != len(form):
            raise ValueError(
                f"a `kind: {self.kind}` input declares `axes:` "
                f"{list(form)} — {len(form)} names, the items per request "
                f"first — and {list(self.axes)} has {len(self.axes)}")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(
                f"`axes: {list(self.axes)}` names an axis twice")
        if self.kind in ("image", "video") and self.width is not None:
            raise ValueError(
                f"an {self.kind} input declares no `width`: its pixel "
                f"channels are part of the patch width its `patchify` reader "
                f"declares")
        if self.unmodeled is not None and (
                self.per_item_of is None or not self.unmodeled.strip()):
            raise ValueError(
                "`unmodeled:` states why the items of another input "
                "(`per_item_of:`) are not walked, citing the source; a "
                "top-level media input is read by its encoder")
        stray = sorted(set(self.fixed) - set(self.axes[1:]))
        if stray:
            raise ValueError(
                f"`fixed:` names {stray}, which are not this input's item "
                f"axes {list(self.axes[1:])}; the item count is the "
                f"workload's")
        return self


class PromptSlots(BaseModel):
    """Which prompt positions an edge from an encoder into the language model
    fills (`prompt_slots:`).

    An image or an audio clip occupies a run of prompt positions — its slots.
    The edge that writes the encoder's rows into those positions DEFINES them
    (`defines: true`): how many positions a request's media fill is that
    edge's rows, an encoder-side fact (`Deployment.media_tokens`) that no
    number of the language model reads — its step is its own text workload.
    An edge that adds further features onto the same positions (Qwen3-VL's
    DeepStack) defines nothing (`defines: false`) and must carry as many rows.
    The op does not say which: a `scatter_rows_add` could define slots, so the
    spec does. `domain` names the media input whose items the slots hold;
    every media input that is not the items of another has exactly one
    defining edge."""
    model_config = ConfigDict(extra="forbid")

    domain: str
    defines: bool
    #: Where the framework fills (or adds onto) these positions.
    source_anchor: str

    @field_validator("domain", "source_anchor")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("`prompt_slots` needs a `domain` and a "
                             "`source_anchor` citing the framework source")
        return value


class StreamSpec(BaseModel):
    """One named stream declared on a repeated stack.

    A stream holds one value per layer that writes it, or the sequence of
    values appended to it. The fields say which, and where it attaches:

    * **Carried** (`carries: true`), at the block boundary. The stream is
      the value the stack hands from one iteration to the next: its initial
      value is the stack's `inputs[0]`, each iteration binds its block's
      formal 0 to the stream's current value, and the block's output becomes
      the new value. Formal 0 is the block template's calling convention,
      not a name. Such a stream is local to its stack: the name means
      something only there, so the backbone, an MTP head and a vision tower
      may each call theirs `hidden`. Outside readers name the stack itself.
    * **Inner** (none of `carries`, `from`, `read`), at inner nodes. A
      component `stream_out`s the stream from one of its nodes and another
      `reads` it (DSA shared top-k, YOCO/CLA shared KV). The reader names
      the layer whose write it reads: the declaration that selects a reading
      layer (`per_layer.assign` entry, or a `template:` stack) writes
      `reads_from: {<stream>: <layer>}` — a layer number of this stack,
      `"i - k"` for the layer k before, `{stack, layer}` for another
      stack's, or `{nearest: <set>}`, the nearest earlier layer of a set the
      config lists (`StreamLayerSet`). That layer writes the stream: a
      component `stream_out`s it from a node. Eval binds a reader to that
      node's tensor.
      The name is local to the stack that declares it: an MTP head may
      declare its own stream of a name the backbone also declares.
      An inner stream with `seed` holds a value before the stack's first
      layer writes it: the output of the model layer `seed` names, read
      by a reader whose `reads_from` names the layer before the first one
      (`"i - 1"` at layer 0) — the stack's entry, as `before: 0` names it
      for an appended stream (DeepSeek-V4.1's `make_identity_pre_mix`
      seeding the hyper-connection mix layer 0 reduces with).
      A model-level `op:` after the stack that names the stream in
      `inputs:` reads its final value: the write of the last layer that
      writes it.
    * **Appended** (`from` + `after` + `index`, or `from` + `before`). The
      stack appends a value to the stream after chosen iterations, or at
      their entry, and the stream is the k-element sequence of those values
      in iteration order (Qwen3-VL's vision blocks 8, 16 and 24 feeding the
      DeepStack mergers; Kimi-K3's layers 0, 12, 24, … appending their
      entry to the AttnRes history). k is known at resolve; a stream with
      k = 0 is not published.
    * **Read all** (`read: all`, on the appending declaration itself). The
      appending stack reads its own stream: every occurrence of a component
      that `reads` the name sees, as one `[B, S, m, D]` value, the m
      elements appended strictly before that iteration's entry — `before: j`
      is appended at iteration j's entry, `after: j` at its exit, so either
      is visible from iteration j + 1 on. An occurrence that sees none
      (m = 0) may not read it. A model-level `op:` that names the stream in
      `inputs:` reads all k elements, the stream's final value.
    * **Read each** (`read: each`), a reference to an appended stream a
      stack reads at an inner node: the readers are the iterations whose
      component `reads` the name, in ascending order after resolution, and
      the j-th reads element j. They number k and are contiguous. A stack
      that reads an appended stream at its block boundary declares nothing
      here: it names the stream in `inputs:`, carries no stream, repeats k
      times, and its iteration j binds element j.
    * **Read concat** (`read: concat`, with `axis`), a reference to an
      appended stream a later stack reads whole at an inner node: every
      occurrence of a component that `reads` the name sees the k elements
      joined along `axis`, in element order — one `concat` of k inputs
      (Llava's multi-layer vision features, EAGLE-3's auxiliary hidden
      states). With k = 1 the element is read as it is, with no copy. A
      model-level `op:` that names the stream in `inputs:` and declares
      `axis:` reads it the same way; without `axis:` it reads the history.

    An appended name is model-wide: it has exactly one producing
    declaration, its `from`, and a `read: each` is a reference to one, not
    a second. An inner name is keyed by its stack and the name.

    Fields:
      carries — attach at the block boundary (see above). A stack has at most
               one. Omitted means false, as for any boolean field. A
               carried stream settles its pending layout at every
               iteration exit.
      from   — what an appended stream appends: the stack's carried stream,
               or `out`, the block's output at each iteration of a stack
               that carries none.
      after  — which iterations it appends after: a list of integers or
               dim expressions (`[2, "n_layers // 2", "n_layers - 3"]`,
               EAGLE-3's default; `//` floors), `config.<path>` naming a
               list in the config (`?? absent`: a config without the key
               appends nothing, k = 0), or `every`.
      before — which iterations it appends at the entry of, instead of
               `after` (not both): the same list forms, `every`, or
               `{every: <dim>}` — iterations 0, N, 2N, … below the stack's
               depth. An index is the iteration number, in [0, n); no
               negatives. The entry of iteration 0 is the stack's input.
               There is no `index` to declare: the entry of iteration j is
               what `hidden_states[j]` would name. Only a carried stream
               has a value at an entry, so `from` is the carried stream.
      index  — what an `after` index counts, declared because the corpus
               uses both: `block` — i is "after iteration i", in [0, n),
               negatives from the end (Qwen3-VL's
               `deepstack_visual_indexes`); `hidden_states` — HF's
               `hidden_states[i]`: 0 is the stack's input, i >= 1 is after
               iteration i-1, in [0, n], negatives from the end (Llava's
               `vision_feature_layer`). `after: every` counts blocks.
      read   — `each`, above: a reference in the reading stack. `all`,
               above: written on the appending declaration, since the
               stack that appends is the one that reads. At most one
               `read: all` per stack; its history axis binds `Sar`.
               `concat`, above: a reference in the reading stack, with
               `axis`.
      axis   — the axis a `read: concat` joins the elements along
               (negative from the end). Only with `read: concat`.
      order  — `declared`: the elements are in the order the explicit
               `after` / `before` list writes them, which need not
               ascend and may repeat an index (HF's
               `vision_feature_layer: [-2, -5, -8, -11, 6]` is joined in
               that order). Only with an explicit list, and only a
               concat reads such a stream: `each`, `all` and a
               model-level history read follow iteration order.
               Omitted, the list must strictly ascend.
      slice  — `{axis, start}`: the stream publishes each value from
               `start` on along `axis` (`hs[:, 1:]`, the CLS crop Llava
               takes before it concatenates). Only a concat reads such a
               stream; the crop is a `slice` view of each element ahead
               of the `concat`.
      seed   — an inner stream's value before the first layer writes it:
               the name of an earlier `op:` layer, as `inputs:` names one.
      shape  — optional publish-gate of an inner stream vs the `stream_out`
               producer node's evaluated shape, e.g.
               `[B, s_query, n_topk_keys]`. Resolved per assignment
               VARIANT: a symbol the variant overrides via
               `per_layer.assign[].dims` takes that variant's value. Eval
               does not seed the edge from this list. Omit it when the
               producer tensor is the only shape that should exist.
      src    — forward-source anchor; same truthfulness treatment as an op's
               source_anchor. Required on an appended stream.
    """

    # `from` is a Python keyword; the field keeps its YAML name in a dump.
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)

    # Left out of the dump while unset, so a stream dumps as it did before
    # the field existed.
    carries: bool = Field(default=False, exclude_if=lambda v: not v)
    from_: Optional[str] = Field(default=None, alias="from",
                                 exclude_if=lambda v: v is None)
    after: Optional[Union[List[Union[int, str]], str, Dict[str, Any]]] = Field(
        default=None, exclude_if=lambda v: v is None)
    before: Optional[Union[List[Union[int, str]], str, Dict[str, Any]]] = Field(
        default=None, exclude_if=lambda v: v is None)
    index: Optional[Literal["block", "hidden_states"]] = Field(
        default=None, exclude_if=lambda v: v is None)
    read: Optional[Literal["each", "all", "concat"]] = Field(
        default=None, exclude_if=lambda v: v is None)
    axis: Optional[int] = Field(default=None, exclude_if=lambda v: v is None)
    order: Optional[Literal["declared"]] = Field(
        default=None, exclude_if=lambda v: v is None)
    slice: Optional[StreamSlice] = Field(
        default=None, exclude_if=lambda v: v is None)
    shape: Optional[List[str]] = None
    seed: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    src: str = ""
    #: On a `read:` entry whose stream an encoder group appends: the prompt
    #: positions its elements land on (`PromptSlots`).
    prompt_slots: Optional[PromptSlots] = Field(
        default=None, exclude_if=lambda v: v is None)

    @field_validator("shape")
    @classmethod
    def _nonempty_shape(
        cls, value: Optional[List[str]],
    ) -> Optional[List[str]]:
        if value is None:
            return value
        if not value or any(not str(axis).strip() for axis in value):
            raise ValueError("stream `shape` must contain non-empty axis expressions")
        return value

    @field_validator("after", "before")
    @classmethod
    def _iteration_selector(cls, value, info):
        field = info.field_name
        if isinstance(value, dict):
            if field == "after":
                raise ValueError(
                    "stream `after: {every: <dim>}` (one iteration in N) is "
                    "not accepted yet; write `every` or name the iterations")
            period = value.get(_EVERY)
            if isinstance(period, int) and not isinstance(period, bool):
                period = str(period)
            if (set(value) != {_EVERY} or not isinstance(period, str)
                    or not period.strip()):
                raise ValueError(
                    "stream `before: {every: <dim>}` names one dim "
                    "expression, the period")
            return {_EVERY: period.strip()}
        if isinstance(value, str):
            value = value.strip()
            if value != _EVERY and not _CONFIG_LIST_REF.fullmatch(value):
                raise ValueError(
                    f"stream `{field}` is a list of integers or dim "
                    f"expressions, `config.<path>` naming a list "
                    f"(optionally `?? absent`) with no arithmetic, or "
                    f"`every`"
                    + (", or `{every: <dim>}`" if field == "before" else ""))
        if isinstance(value, list):
            if any(isinstance(i, str) and not i.strip() for i in value):
                raise ValueError(
                    f"stream `{field}: {value}` has an empty entry")
            if field == "before" and any(
                    isinstance(i, int) and i < 0 for i in value):
                raise ValueError(
                    f"stream `before: {value}` names iteration numbers, "
                    f"which are not negative")
        return value

    @model_validator(mode="after")
    def _one_kind(self) -> "StreamSpec":
        if self.carries and self.shape is not None:
            raise ValueError(
                "a `carries: true` stream has no `stream_out` producer for "
                "`shape` to gate; its value is the block's output")
        if self.before is not None:
            if self.after is not None:
                raise ValueError(
                    "an appended stream appends `after` chosen iterations "
                    "or `before` them, not both")
            if self.index is not None:
                raise ValueError(
                    "`before: j` is the entry of iteration j, which is "
                    "what an index would say; `before` takes no `index`")
            if self.from_ is None:
                raise ValueError(
                    "`before` chooses where an appended stream appends; it "
                    "needs `from`")
        else:
            writer = (self.from_ is not None, self.after is not None,
                      self.index is not None)
            if any(writer) and not all(writer):
                raise ValueError(
                    "an appended stream declares `from`, `after` and "
                    "`index` together, or `from` and `before`")
        if self.read == "all" and not self.appends:
            raise ValueError(
                "`read: all` is written on the appending declaration: the "
                "stack that appends to a stream is the one that reads its "
                "prefix. Another stack reads it element by element "
                "(`read: each`), or a model-level `op:` names it in "
                "`inputs:` for all of it")
        if (self.read == "concat") != (self.axis is not None):
            raise ValueError(
                "`read: concat` joins the elements along one `axis`, and "
                "`axis` belongs to nothing else: write both or neither")
        if (self.order is not None or self.slice is not None) \
                and not self.appends:
            raise ValueError(
                "`order` and `slice` say what an appended stream publishes; "
                "they go on its `from:` declaration")
        if self.order is not None and not (
                isinstance(self.after, list) or isinstance(self.before, list)
                or (isinstance(self.after, str) and self.after != _EVERY)
                or (isinstance(self.before, str)
                    and self.before != _EVERY)):
            raise ValueError(
                "`order: declared` keeps the order an explicit list writes "
                "its indices in; `every` and `{every: <dim>}` write none")
        if self.read == "all" and (self.order is not None
                                   or self.slice is not None):
            raise ValueError(
                "`read: all` sees the history in the order the stack "
                "appended it, whole; `order: declared` and `slice` shape "
                "what a `read: concat` joins")
        if self.seed is not None and (self.carries or self.appends
                                      or self.read is not None):
            raise ValueError(
                "`seed` is the value an inner stream holds before its "
                "first write; a carried stream starts at the stack's "
                "`inputs[0]` and an appended one is a sequence of writes")
        if not (self.appends or self.read is not None):
            return self
        inner = [name for name, set_ in (
            ("carries", self.carries), ("shape", self.shape is not None))
            if set_]
        if self.read in ("each", "concat") and (self.appends or self.src):
            inner.append("from" if self.appends else "src")
        if inner:
            raise ValueError(
                f"stream fields {inner} do not belong on an appended stream "
                f"or a `read:` reference to one")
        if self.appends:
            if not self.src.strip():
                raise ValueError(
                    "an appended stream's `src` quotes the forward line that "
                    "appends to it")
            if self.after == _EVERY and self.index != "block":
                raise ValueError(
                    "`after: every` names iterations, so its `index` is "
                    "`block`")
        return self

    @property
    def appends(self) -> bool:
        """Whether this declaration produces an appended stream."""
        return self.from_ is not None

    def iterations(self, n: int, config: Any,
                   dim: Callable[[str], Any]) -> Tuple[int, ...]:
        """The iteration of an `n`-deep stack after which each element's
        value stands, in order, or -1 for the stack's input: `after: j`
        and `before: j + 1` name one value, iteration j's exit. Refuses
        what names no iteration."""
        if self.before is not None:
            return tuple(j - 1 for j in self._entries(n, config, dim))
        return self._exits(n, config, dim)

    def writers(self, n: int, config: Any,
                dim: Callable[[str], Any]) -> Tuple[int, ...]:
        """The iteration that appends each element, in order: `before: j`
        appends at iteration j's entry, `after: j` at its exit (-1 for
        `hidden_states[0]`, before the first). An element is visible to a
        `read: all` of the stream from the iteration after its writer on;
        `iterations` alone cannot say that, because the entry of j and the
        exit of j - 1 are one value written by two different iterations."""
        if self.before is not None:
            return self._entries(n, config, dim)
        return self._exits(n, config, dim)

    def _entries(self, n: int, config: Any,
                 dim: Callable[[str], Any]) -> Tuple[int, ...]:
        """`before:` resolved to iteration numbers in [0, n). `dim`
        evaluates a dim expression under the model's dims."""
        raw = self.before
        if raw == _EVERY:
            return tuple(range(n))
        if isinstance(raw, dict):
            period = int(dim(raw[_EVERY]))
            if period <= 0:
                raise ValueError(
                    f"stream `before: {{every: {raw[_EVERY]}}}` resolves to "
                    f"{period!r}; its period is a positive integer")
            return tuple(range(0, n, period))
        out = self._listed("before", raw, config, dim)
        bad = [j for j in out if not 0 <= j < n]
        if bad:
            raise ValueError(
                f"stream `before: {list(out)}` names iteration(s) {bad} "
                f"outside a {n}-deep stack, whose iterations are [0, {n})")
        self._check_order("before", raw, out)
        return out

    def _listed(self, field: str, raw: Any, config: Any,
                dim: Callable[[str], Any]) -> Tuple[int, ...]:
        """An explicit `after:` / `before:` list as integers, in the order
        written: a `config.<path>` list, or literals and dim expressions
        (`n_layers // 2` floors; any other expression must resolve to an
        integer)."""
        if isinstance(raw, str):
            return tuple(self._config_list(field, raw, config))
        out: List[int] = []
        for entry in raw:
            if isinstance(entry, int):
                out.append(entry)
                continue
            value = dim(entry)
            if "//" in entry:
                value = math.floor(value)
            if value != int(value):
                raise ValueError(
                    f"stream `{field}` entry {entry!r} resolves to {value}, "
                    f"not an integer")
            out.append(int(value))
        return tuple(out)

    def _check_order(self, field: str, raw: Any,
                     out: Tuple[int, ...]) -> None:
        """Without `order: declared` the elements follow the iterations,
        which strictly ascend; with it, the list's own order stands and may
        repeat an index."""
        if self.order is None and any(b <= a for a, b in pairwise(out)):
            raise ValueError(
                f"stream `{field}: {raw}` must name strictly increasing "
                f"iterations, or declare `order: declared` for a concat "
                f"that joins them in the order written")

    @staticmethod
    def _config_list(field: str, raw: str, config: Any) -> List[int]:
        """The integer list `config.<path>` names — a single integer is a
        list of one; `?? absent` lets a config omit it (no elements)."""
        ref = _CONFIG_LIST_REF.fullmatch(raw)
        value: Any = config
        walked: List[str] = []
        for part in ref["path"].split("."):
            if value is None:
                break
            if not isinstance(value, dict):
                # `?? absent` lets a config omit the key, not give the
                # path a different shape.
                raise ValueError(
                    f"stream `{field}: {raw}`: config.{'.'.join(walked)} "
                    f"is {value!r}, not a mapping")
            walked.append(part)
            value = value.get(part)
        if value is None:
            if ref["absent"]:
                return []
            raise ValueError(
                f"stream `{field}: {raw}`: the config does not set it; "
                f"write `{raw} ?? absent` if a config may omit it")
        if isinstance(value, int) and not isinstance(value, bool):
            # One index is a list of one: HF's `vision_feature_layer: -2`
            # names one hidden state, read as it is.
            value = [value]
        if not isinstance(value, list) or not all(
                isinstance(v, int) and not isinstance(v, bool)
                for v in value):
            raise ValueError(
                f"stream `{field}: {raw}` is {value!r}, not a list of "
                f"integers")
        if field == "before" and any(v < 0 for v in value):
            raise ValueError(
                f"stream `before: {raw}` is {value!r}; iteration numbers "
                f"are not negative")
        return value

    def _exits(self, n: int, config: Any,
               dim: Callable[[str], Any]) -> Tuple[int, ...]:
        """`after:` resolved to the iteration each element follows."""
        raw = self.after
        if raw == _EVERY:
            return tuple(range(n))
        top = n if self.index == "block" else n + 1
        out: List[int] = []
        for i in self._listed("after", raw, config, dim):
            j = i + top if i < 0 else i
            if not 0 <= j < top:
                raise ValueError(
                    f"stream index {i} ({self.index}) is outside a {n}-deep "
                    f"stack, whose {self.index} indices are [0, {top})")
            out.append(j if self.index == "block" else j - 1)
        self._check_order("after", raw, tuple(out))
        return tuple(out)



class LoopGate(BaseModel):
    """What a looped stack runs after each pass: a block template whose
    formal 0 takes the carried value leaving the pass and formal 1 the
    carried value entering it, and whose output is carried on. One stored
    copy per pass, indexed by pass under `checkpoint.prefix`."""

    model_config = ConfigDict(extra="forbid")

    template: str
    checkpoint: CheckpointAddress

    @model_validator(mode="after")
    def _indexed_by_pass(self) -> "LoopGate":
        self.checkpoint.only("prefix", where="loop gate")
        if self.checkpoint.prefix is None:
            raise ValueError(
                "a loop gate stores one copy per pass; its `checkpoint` "
                "declares the `prefix` the pass index follows")
        return self


class LoopSpec(BaseModel):
    """A looped stack: stored layers `layers[0]..layers[1]` (inclusive) run
    `passes` times with the same weights, each pass after the previous one,
    with an optional gate after every pass. `repeat:` still counts STORED
    layers; the stack's iterations are its executions in order."""

    model_config = ConfigDict(extra="forbid")

    layers: List[str]
    passes: str
    gate: Optional[LoopGate] = Field(default=None,
                                     exclude_if=lambda v: v is None)

    @field_validator("layers", mode="before")
    @classmethod
    def _range(cls, value: Any) -> Any:
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"`loop.layers: {value!r}` is the inclusive stored range "
                f"[first, last]")
        return [str(v).strip() for v in value]

    @field_validator("passes", mode="before")
    @classmethod
    def _passes(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, str)) \
                or not str(value).strip():
            raise ValueError(
                f"`loop.passes: {value!r}` is an integer or a dim expression")
        return str(value).strip()


class ModelLayer(BaseModel):
    """A single entry in the model structure (embedding, block group, lm_head, etc.)."""
    model_config = ConfigDict(extra="forbid")

    name: str
    # Upstream model-level edges: the earlier ModelLayer(s), streams or
    # model inputs (`ArchSpec.model_inputs`) this layer reads. Every layer
    # that reads a tensor names it; only a zero-input op (a parameter, a
    # buffer) omits it.
    inputs: Optional[List[str]] = None
    # The stack's streams, name → StreamSpec. At most one `carries: true`:
    # the value handed from iteration to iteration through the block's
    # formal 0 (`hidden` in the corpus). Inner ones hand a value from the
    # layer that writes it to the layers that read it, with producer and
    # consumer in different variant components (DSA full/shared top-k,
    # YOCO/CLA shared KV, …): a component `stream_out`s one (produces it) and
    # another `reads` it (consumes); each reading layer's declaration names
    # the layer it reads in `reads_from`, and eval binds `in:` to the
    # producer tensor. Optional `shape` is a publish-gate against
    # `stream_out`, not the live edge. A cached `stream_out` node also
    # carries its KV length to readers. Appended ones (`from` / `after`) are
    # the sequence of values this stack appends after chosen iterations; a
    # `read: each` entry names one this stack's components read element by
    # element. See StreamSpec and BlockGraph.reads / .stream_out.
    streams: Dict[str, StreamSpec] = {}
    # A `template:` stack whose every layer reads an inner stream: stream →
    # the layer whose write each reads (see `BlockAssignment.reads_from`; a
    # `per_layer:` stack writes it on the `assign` entries). Left out of the
    # dump while unset.
    reads_from: Optional[ReadsFrom] = Field(
        default=None, exclude_if=lambda v: v is None)
    op: Optional[str] = None                     # for simple ops (embedding, norm, linear)
    # Semantic responsibility at the model boundary. Consumers must classify
    # boundary tensors from this declaration rather than module-name
    # conventions such as `lm_head`, `output_layer`, or `head` — or, for the
    # input side, from `op == "embedding"`, which says only that a table is
    # indexed and not WHICH table the model's tokens enter through. A model
    # may carry several: MiMo-V2.5 has `speech_embeddings`, Qwen3.5-MoE has
    # `model.visual.pos_embed`, LongCat has `mtp.embed_tokens`. Exactly one
    # of them is what a tied output head shares its weights with, and
    # traversal order is not evidence about which.
    boundary_role: Optional[BoundaryRole] = None
    # The role of a model-level node no component it calls declares — a
    # model-root parameter such as DeepSeek-V4's `hc_head_fn`. The same
    # field and gate as a graph node's call-site `role:`; quant routing
    # (`quant_context.per_role`) and cost attribution read it.
    role: Optional[str] = None
    dims: Dict[str, str] = {}
    dim: Optional[str] = None                    # shorthand for single-dim ops
    # Axis selector for view-ops at the model top level (broadcast,
    # concat, slice, …). Same semantics as GraphNode.axis: numpy
    # `np.expand_dims`-style position in the OUTPUT for broadcast,
    # last-axis default (-1) otherwise.
    axis: Optional[int] = None
    # Explicit n-D shape for model-level source ops (`parameter_tensor`).
    # List of dim expressions evaluated against the spec's dim
    # bindings. Mirrors GraphNode.shape — used when the model spec
    # declares a top-level n-D nn.Parameter (e.g. V4-Pro's hc_head_fn
    # `[n_hidden_copies, n_hidden_copies·d_model]`).
    shape: Optional[List[str]] = None
    # Same occurrence-level persistent endpoint contract as GraphNode.
    weight_axes: Optional[List[WeightAxisSpec]] = None
    # Same expert-weight checkpoint encoding as GraphNode; not serialized
    # when absent.
    expert_weight: Optional[ExpertWeight] = Field(
        default=None, exclude_if=lambda v: v is None)
    # Same checkpoint axis order as GraphNode.stored_axes; not serialized
    # when absent.
    stored_axes: Optional[List[int]] = Field(
        default=None, exclude_if=lambda v: v is None)
    # Same stored split as GraphNode.stored_split; not serialized when absent.
    stored_split: Optional[StoredSplitSpec] = Field(
        default=None, exclude_if=lambda v: v is None)
    # Same declaration as GraphNode.stored_without_graph_reader; not
    # serialized when absent.
    stored_without_graph_reader: Optional[StoredWithoutGraphReaderSpec] = Field(
        default=None, exclude_if=lambda v: v is None)
    # Same declaration as GraphNode.host_offload; not serialized when absent.
    host_offload: Optional[str] = Field(
        default=None, exclude_if=lambda v: v is None)
    repeat: Optional[str] = None                 # repeat count (expression, e.g. "n_layers")
    template: Optional[str] = None               # block template name (for repeated blocks)
    # Param bindings when `template` references a parameterized compound op.
    # See GraphNode.params for the dict-valued nested-call semantics.
    params: Dict[str, Any] = {}
    assignments: Optional[List[BlockAssignment]] = None  # heterogeneous block assignments
    #: A looped stack (`LoopSpec`). Left out of the dump while unset.
    loop: Optional[LoopSpec] = Field(default=None,
                                     exclude_if=lambda v: v is None)
    tied_weight: Optional[TiedWeightSpec] = None
    #: On a `group:` or `template:` layer the loader wrote from a call site
    #: (`spec_parsing.LayerCall`): the earlier layer whose module this one
    #: runs again, on other inputs and with that layer's weights.
    call: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)
    # Group: named collection of sub-layers (e.g. MTP module)
    group: Optional[bool] = None                 # marks this as a group container
    layers: Optional[List["ModelLayer"]] = None  # sub-layers within the group
    # Primary auxiliary-stack class for stack ownership, accounting and the
    # architecture-view badge. The values are deliberately exclusive at this
    # layer: `speculative` means a non-MTP draft/verification stack; MTP always
    # declares `mtp`. Set on every `group: true` layer so no consumer
    # substring-matches the layer's name.
    aux_kind: Optional[Literal["mtp", "vision", "audio", "speculative"]] = None
    #: On a layer that reads an encoder's output from outside the encoder:
    #: the prompt positions it fills (`PromptSlots`). A layer reading several
    #: encoder outputs (one masked scatter writing every media input's rows,
    #: as vLLM's `_merge_multimodal_embeddings` does) gives each its own,
    #: keyed by the name it reads.
    prompt_slots: Optional[Union[PromptSlots, Dict[str, PromptSlots]]] = Field(
        default=None, exclude_if=lambda v: v is None)
    #: On a `patchify` layer: what one attention segment of the encoder is,
    #: in the processor's `cu_seqlens` sense — a whole item (`item`), or one
    #: temporal patch of it (`temporal_patch`). For an image the two agree.
    segment: Optional[Literal["item", "temporal_patch"]] = Field(
        default=None, exclude_if=lambda v: v is None)
    #: Where in the processor / framework source the facts a media layer
    #: declares come from (`patchify`'s patch size and alignment, `pad_rows`'
    #: padded length, `unpad`'s strip). Required on those layers.
    source_anchor: Optional[str] = Field(
        default=None, exclude_if=lambda v: v is None)

    # WHERE this entry's tensors live in the checkpoint (`CheckpointAddress`).
    # A stack (`template:` / `per_layer:` with `repeat:`) may declare its
    # `prefix` and `index`; one that does not lives at its own name under the
    # scope root. An indexed aux group (`group: true` + `aux_kind`) must
    # declare the `prefix` of its stack, and a second repeated stack inside
    # it declares its own (Qwen3-VL's `model.visual.deepstack_merger_list`
    # beside `model.visual.blocks`). A group may declare the `root` its ops
    # hang off; an op may declare its `path`, or `index: local` when it is
    # repeated and the checkpoint indexes its copies, or `prefix` + `path`
    # when it stores one copy inside each layer of an indexed stack.
    checkpoint: Optional[CheckpointAddress] = None

    # Author opt-out for the quant-metadata-suffix lockdown — see
    # `GraphNode.quant_metadata`. Set on top-level `parameter_vector`
    # layers whose checkpoint name collides with a registered scheme's
    # metadata-sibling suffix.
    quant_metadata: Optional[bool] = None

    @property
    def carried_stream(self) -> Optional[str]:
        """The name of the stream this stack carries across its block
        boundary, or None when it carries none."""
        carried = [name for name, spec in self.streams.items() if spec.carries]
        return carried[0] if carried else None

    @property
    def inner_streams(self) -> Dict[str, StreamSpec]:
        """The streams that attach at inner nodes, produced by a component's
        `stream_out` and read where a reader's `reads_from` says: every one
        but the carried stream and the appended streams and references to
        them."""
        return {name: spec for name, spec in self.streams.items()
                if not (spec.carries or spec.appends or spec.read)}

    @property
    def appended_streams(self) -> Dict[str, StreamSpec]:
        """The appended streams this stack produces (`from` / `after`)."""
        return {name: spec for name, spec in self.streams.items()
                if spec.appends}

    @property
    def read_streams(self) -> Tuple[str, ...]:
        """The appended streams this stack's components read element by
        element (`read: each`)."""
        return tuple(name for name, spec in self.streams.items()
                     if spec.read == "each")

    @property
    def prefix_streams(self) -> Tuple[str, ...]:
        """The appended streams this stack appends to and reads the prefix
        of (`read: all`)."""
        return tuple(name for name, spec in self.streams.items()
                     if spec.read == "all")

    @property
    def concat_streams(self) -> Tuple[str, ...]:
        """The appended streams this stack's components read whole, joined
        along an axis (`read: concat`)."""
        return tuple(name for name, spec in self.streams.items()
                     if spec.read == "concat")

    @property
    def is_parallel_stack(self) -> bool:
        """A repeated stack that carries no stream: its iterations run side
        by side and hand nothing on, so there is no single value its name
        could read. Its outputs leave it only as an appended stream
        (`from: out`).

        The stack is the layer that repeats, as for a carried stream: a
        `template:` / `per_layer:` layer with `repeat:`, or a repeated
        `group:` around blocks that do not repeat themselves (DeepSeek's
        `mtp`). Under a repeated group whose block repeats, the innermost
        repeated layer is the stack and the group is not one."""
        if self.repeat is None or self.carried_stream is not None:
            return False
        if self.group:
            blocks = [sub for sub in self.layers or ()
                      if sub.template is not None or sub.assignments is not None]
            return bool(blocks) and all(sub.repeat is None for sub in blocks)
        return self.template is not None or self.assignments is not None

    @field_validator("role")
    @classmethod
    def _role_is_registered(cls, v: Optional[str]) -> Optional[str]:
        return roles.require_registered(v)

    @field_validator("dims")
    @classmethod
    def _writes_no_derived_name(cls, v: Dict[str, str]) -> Dict[str, str]:
        dim_registry.refuse_written_derived(v, "model layer `dims:`")
        return v

    @field_validator("name", "op")
    @classmethod
    def _nonempty_identity(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("model layer names and ops must be non-empty and trimmed")
        return value

    @field_validator("reads_from", mode="before")
    @classmethod
    def _reads_from(cls, value: Any) -> Optional[ReadsFrom]:
        return check_reads_from(value)

    @field_validator("inputs")
    @classmethod
    def _valid_inputs(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return values
        if any(not value.strip() for value in values):
            raise ValueError("model layer inputs must name non-empty earlier layers")
        if len(values) != len(set(values)):
            raise ValueError("model layer inputs must not contain duplicates")
        return values

    @model_validator(mode="after")
    def _one_layer_form(self) -> "ModelLayer":
        forms = {
            "op": self.op is not None,
            "template": self.template is not None,
            "per_layer": self.assignments is not None,
            "group": self.group is not None,
        }
        selected = [name for name, present in forms.items() if present]
        if len(selected) != 1:
            raise ValueError(
                f"Model layer {self.name!r} requires exactly one of `op`, "
                f"`template`, `per_layer`, or `group: true`; got {selected}")

        form = selected[0]
        contextual = {
            "boundary_role", "dims", "dim", "axis", "shape",
            "weight_axes", "expert_weight", "stored_axes", "stored_split",
            "stored_without_graph_reader", "host_offload", "params",
            "tied_weight", "layers", "aux_kind", "checkpoint", "quant_metadata", "streams",
            "reads_from", "loop",
        }
        allowed = {
            "op": {
                "boundary_role", "dims", "dim", "axis", "shape",
                "weight_axes", "expert_weight", "stored_axes", "stored_split",
                "stored_without_graph_reader", "host_offload", "tied_weight",
                "quant_metadata", "checkpoint",
            },
            "template": {"params", "streams", "checkpoint", "reads_from",
                         "loop"},
            "per_layer": {"streams", "checkpoint", "loop"},
            "group": {"layers", "aux_kind", "checkpoint", "streams"},
        }
        invalid = sorted((self.model_fields_set & contextual) - allowed[form])
        if "reads_from" in invalid and form == "per_layer":
            raise ValueError(
                f"Model layer {self.name!r} declares `reads_from` on a "
                f"`per_layer:` stack; write it on the `per_layer.assign` "
                f"entries whose layers read the stream, one declaration per "
                f"layer")
        if invalid:
            raise ValueError(
                f"Model layer {self.name!r} uses {form} form but declares "
                f"field(s) {invalid} that only belong to another layer form")

        if self.call is not None and form not in ("template", "group"):
            raise ValueError(
                f"Model layer {self.name!r} calls {self.call!r}, a "
                f"`template:` call or a `group:`; it is {form} form")
        if self.dim is not None and self.dims:
            raise ValueError(
                f"Model layer {self.name!r} cannot mix `dim` shorthand with `dims`")
        # As for a graph node: the stored tensor's shape
        # (`ops.weight_leading_axes`).
        if self.shape is not None and (
                not self.shape or any(not axis.strip() for axis in self.shape)):
            raise ValueError(
                "model-layer `shape` must contain non-empty axis expressions")
        if self.weight_axes is not None and not self.weight_axes:
            raise ValueError(
                "model-layer `weight_axes` must contain one contract per stored axis")
        check_stored_axes(self.stored_axes, self.weight_axes)
        check_stored_split(self.stored_split, self.expert_weight)
        if self.boundary_role == "input_embedding" and self.op != "embedding":
            raise ValueError(
                "boundary_role=input_embedding requires `op: embedding`")
        if self.boundary_role == "output_head" and self.op != "linear":
            raise ValueError("boundary_role=output_head requires `op: linear`")
        if self.tied_weight is not None and self.boundary_role != "output_head":
            raise ValueError(
                "tied_weight is only valid on an explicitly declared output_head")
        if self.assignments is not None and self.repeat is None:
            raise ValueError("per_layer model stacks require a `repeat` extent")
        if self.streams and self.repeat is None:
            raise ValueError("model-layer `streams` require a repeated stack")
        carried = [name for name, spec in self.streams.items() if spec.carries]
        if len(carried) > 1:
            raise ValueError(
                f"model layer {self.name!r} declares {len(carried)} "
                f"`carries: true` streams {carried}; a stack hands one value "
                f"from iteration to iteration")
        self._check_appended_streams(form)
        self._check_loop()

        if self.group is not None:
            if self.group is not True:
                raise ValueError("model layer group marker must be `group: true`")
            if not self.layers:
                raise ValueError(
                    f"Model group {self.name!r} must contain non-empty `layers`")
            if self.aux_kind is None:
                raise ValueError(
                    f"Model group {self.name!r} must declare `aux_kind`")
            # What a stack reads decides what it is (`ArchSpec.encoder_groups`,
            # held at the spec, which knows the model inputs): an encoder reads
            # its media input and names no backbone tensor; a head (MTP,
            # speculative) names the backbone tensor it reads in `inputs:`.
            looped = [layer.name for layer in self.layers or []
                      if layer.loop is not None]
            if looped:
                raise ValueError(
                    f"Model group {self.name!r}: {looped} declare `loop:` "
                    f"inside a group; a loop is supported on a top-level "
                    f"stack only")
            child_names = [layer.name for layer in self.layers or []]
            duplicate_children = sorted({
                name for name in child_names if child_names.count(name) > 1
            })
            if duplicate_children:
                raise ValueError(
                    f"Model group {self.name!r} has duplicate child layer "
                    f"name(s) {duplicate_children}")
            block_children = [
                layer for layer in self.layers or []
                if layer.template is not None or layer.assignments is not None
            ]
            indexed_blocks = bool(block_children) and (
                self.repeat is not None
                or any(layer.repeat is not None or layer.assignments is not None
                       for layer in block_children)
            )
            address = self.checkpoint
            if self.call is not None:
                # A call stores nothing: its tensors are the called group's.
                indexed_blocks = False
            if address is not None:
                address.only("prefix", "index", "root",
                             where=f"model group {self.name!r}")
            if indexed_blocks and (address is None or address.prefix is None):
                raise ValueError(
                    f"Indexed aux group {self.name!r} must declare its "
                    "`checkpoint: {prefix, index}` address")
            if not indexed_blocks and address is not None and (
                    address.prefix is not None or address.index is not None):
                raise ValueError(
                    f"Singleton aux group {self.name!r} has no block index "
                    "space, so its `checkpoint` declares no `prefix` / "
                    "`index`; a `root` is all it may say")
        elif self.layers is not None or self.aux_kind is not None:
            raise ValueError(
                f"Model layer {self.name!r}: `layers` and `aux_kind` are "
                "only valid on `group: true`")
        elif self.checkpoint is not None and self.op is not None:
            self.checkpoint.only("path", "index", "prefix",
                                 where=f"model layer {self.name!r}")
            if self.checkpoint.index is not None and (
                    self.checkpoint.index != "local" or self.repeat is None):
                raise ValueError(
                    f"Model layer {self.name!r}: `checkpoint.index: local` "
                    "indexes the copies of a repeated op; it needs `repeat:`")
            if self.checkpoint.prefix is not None and (
                    self.checkpoint.path is None or self.repeat is None
                    or self.checkpoint.index is not None):
                raise ValueError(
                    f"Model layer {self.name!r}: `checkpoint.prefix` on an op "
                    "stores one copy of a repeated op under each index of "
                    "that stack, at the module `path` inside it; it needs "
                    "`path:` and `repeat:`, and no `index:`")
        elif self.checkpoint is not None:
            self.checkpoint.only("prefix", "index",
                                 where=f"model layer {self.name!r}")
            if self.checkpoint.prefix is None or self.repeat is None:
                raise ValueError(
                    f"Model layer {self.name!r}: a stack's `checkpoint` "
                    "declares the `prefix` its `repeat:` index follows")
        if self.tied_weight is not None and self.op is None:
            raise ValueError("tied_weight is only valid on a simple `op` layer")
        return self

    def slots_for(self, ref: str) -> Optional[PromptSlots]:
        """The prompt slots this layer's read of encoder output `ref`
        fills: its one `prompt_slots`, or the entry keyed by `ref`."""
        if isinstance(self.prompt_slots, dict):
            return self.prompt_slots.get(ref)
        return self.prompt_slots

    def _check_loop(self) -> None:
        """A loop belongs to a repeated `template:` / `per_layer:` stack of
        its own (the form check admits no other). It is not a call of
        another layer, and its iterations are executions, so no appended
        stream counts them yet: their iteration selectors would need the
        stored → execution rule, and no model needs it."""
        if self.loop is None:
            return
        if self.repeat is None:
            raise ValueError(
                f"model layer {self.name!r}: `loop:` runs stored layers of a "
                f"repeated stack again; it needs `repeat:`")
        if self.call is not None:
            raise ValueError(
                f"model layer {self.name!r}: a call of {self.call!r} runs "
                f"another layer's weights; it cannot also declare `loop:`")
        appended = [*self.appended_streams, *self.read_streams,
                    *self.concat_streams, *self.prefix_streams]
        if appended:
            raise ValueError(
                f"model layer {self.name!r}: appended streams {appended} on "
                f"a looped stack are not supported; their iteration "
                f"selectors would have to pick among a stored layer's "
                f"executions")

    def _check_appended_streams(self, form: str) -> None:
        """An appended stream, or a `read: each` / `read: concat` of one,
        belongs to the stack whose iterations it counts: a `template:` / `per_layer:` layer
        with `repeat:`. What it appends is that stack's carried stream, or
        `out` when it carries none."""
        declared = [*self.appended_streams, *self.read_streams,
                    *self.concat_streams]
        if declared and form not in ("template", "per_layer"):
            raise ValueError(
                f"model layer {self.name!r} declares appended stream(s) "
                f"{declared} on a {form}; an appended stream counts the "
                f"iterations of a `template:` / `per_layer:` stack, so "
                f"declare it on that stack")
        carried = self.carried_stream
        wanted = carried if carried is not None else "out"
        for name, spec in self.appended_streams.items():
            if spec.from_ != wanted:
                raise ValueError(
                    f"model layer {self.name!r}: appended stream {name!r} "
                    f"appends `from: {spec.from_}`, but this stack "
                    + (f"carries {carried!r}, which is what it appends"
                       if carried is not None else
                       "carries no stream; it appends its block's output, "
                       "`from: out`"))
            if spec.from_ == "out" and spec.before is not None:
                raise ValueError(
                    f"model layer {self.name!r}: appended stream {name!r} "
                    f"appends `from: out` `before` iterations, but a "
                    f"block's output has no value at an iteration's entry; "
                    f"a stack appends its entries from its carried stream")
            if spec.from_ == "out" and spec.index != "block":
                raise ValueError(
                    f"model layer {self.name!r}: appended stream {name!r} "
                    f"appends `from: out`, which has no value before the "
                    f"first iteration; its `index` is `block`")
        if len(self.prefix_streams) > 1:
            raise ValueError(
                f"model layer {self.name!r} reads the prefix of "
                f"{list(self.prefix_streams)} (`read: all`); a stack reads "
                f"at most one, whose history axis binds `{HISTORY_AXIS}`")
