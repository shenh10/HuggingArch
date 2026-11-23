"""
Loader — spec YAML in, ResolvedSpec out.

Responsibilities:
  1. Scan components/ + models/ and parse YAML into ArchSpec objects
     (identity files inherit a base via `base:`, merged before parse)
  2. Resolve a HF model_id to its spec by exact id lookup; on a miss, ask the
     registered spec-miss hook for an overlay directory and look up again
  3. Resolve dimension bindings from config.json via dim_resolution
     (config.hidden_size → 4096, "d_model // n_heads" → 128)
  4. Synthesize block-level structure (`_BlockSynthesis`)
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
from backend.code_trace.memo import memoize
import hashlib
import json
import logging
import os
import re
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, FrozenSet, List, Optional, Sequence, Tuple,
                    TYPE_CHECKING)

if TYPE_CHECKING:
    from backend.hf.header import SafetensorsHeader

logger = logging.getLogger(__name__)


class SpecLoadError(RuntimeError):
    """A spec file exists for the requested model but failed to parse /
    validate. Raised by the loader's refuse-if-failed path so HTTP layers can
    report a SPEC error (and point at the broken file) instead of letting a
    generic RuntimeError surface as an opaque 500. Subclasses RuntimeError so
    existing `except RuntimeError` callers keep working.

    `path` is the offending file when there is exactly one, so a caller can
    ask whose spec it is (`overlay.is_user_spec_path`) instead of reading
    the message."""

    def __init__(self, message: str, *, path: Optional[str] = None):
        super().__init__(message)
        self.path = path


class DraftNameConflict(SpecLoadError):
    """A draft file whose name is already taken — by a formal component or
    by another draft — with a different structure. It belongs to no spec
    until one lists it in `drafts:`, so the loader holds it and raises it
    only for a spec that does: which of the two same-named components that
    spec means is undecidable, and taking either one would be a silent
    substitution."""


def merge_base_documents(parent_doc: Dict[str, Any], child_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the loader's one base-merge rule: child keys replace parents.

    This pure function is the single home for `base:` shallow merge. Both the
    loader and spec-bundle normalization need exactly this semantics; keeping
    it separate makes “a published bundle resolves the same way as the run”
    a property of one implementation, not two implementations that happen to
    resemble each other. Lists and nested mappings are replaced wholesale.
    The caller decides which document is parent/child and whether ``base``
    remains in the parent.
    """
    return {
        **parent_doc,
        **{key: value for key, value in child_doc.items() if key != "base"},
    }


import yaml

import backend.arch.dims as dim_registry
import backend.arch.roles as roles
from backend.arch.block_synthesis import _BlockSynthesis
from backend.arch.component_signature import component_structure_signature
from backend.arch.dtypes import stated_default_dtype
from backend.arch.dim_resolution import (
    CheckpointHeader, config_path_value, language_config, resolve_dimensions,
    resolve_view)
from backend.arch.eval.dim import eval_dim_expr
from backend.arch.eval.refusal import SpecRefusal
from backend.arch.law_text import read_law
from backend.arch.loop_schedule import ExecutionSchedule
from backend.arch.ops_base import reject_divisible_parameter_names
from backend.arch.ops import OP_REGISTRY
from backend.arch.spec_parsing import (
    parse_document,
    _parse_component_document,
    _extract_unresolved_note,
)
from backend.custom_id import CACHE_DIR, is_custom_id, model_path, strip_prefix
from backend.arch.document import SpecDocument
from backend.arch.schema import (
    ArchSpec,
    BlockGraph,
    CustomOpSpec,
    ModelLayer,
)


def _scan_match_field(raw_text: str, field: str) -> List[str]:
    """The values a `match:` header declares for `field`, read straight from
    the file text. The best-effort recovery for a spec whose YAML does not
    parse: `yaml.safe_load` gives nothing, but the identity it claims is still
    printed on the page, so a `model_id: [x, y]` (or scalar `model_id: x`)
    line still names the file for a loud id refusal (contract D).

    The whitespace classes match spaces and tabs only (not the `\\s` that also
    spans newlines) on purpose: a match is confined to one line, so a
    block-style list (`model_id:` then an indented `- x` on the next line) does
    not let the value capture bleed onto the list item below — it recovers
    nothing and the file falls through, rather than claiming a mangled token."""
    m = re.search(rf"\b{re.escape(field)}[ \t]*:[ \t]*(.+)", raw_text)
    if not m:
        return []
    value = m.group(1).strip()
    if value.startswith("["):
        value = value[1:].split("]", 1)[0]
    return [tok.strip().strip("'\"") for tok in value.split(",")
            if tok.strip().strip("'\"")]


class LayerSpanError(ValueError):
    """A `range:` / `layers:` selector or a config list of layer numbers
    names a layer its container does not have."""


@dataclass(frozen=True)
class AppendedStream:
    """An appended stream as resolved: the stack that appends to it (and the
    group around that stack, if any), the iteration whose exit each
    element's value is, in order (-1: the stack's input), and the iteration
    that appends each (`StreamSpec.writers`)."""

    name: str
    writer: ModelLayer
    group: Optional[ModelLayer]
    iterations: Tuple[int, ...]
    writers: Tuple[int, ...]

    def visible(self, iteration: int) -> int:
        """How many elements a `read: all` at `iteration` sees: those
        appended strictly before its entry, by an earlier iteration. The
        writers ascend, so they are the first that many elements."""
        return sum(1 for writer in self.writers if writer < iteration)


def strip_machine_paths(obj: Any) -> Any:
    """Recursively drop `source_path` — the absolute on-disk path a spec/
    component was loaded from. It's machine-specific (`/path/to/checkout/...` locally,
    `/home/runner/...` in CI), NOT structural content, so leaving it in makes
    a fingerprint depend on the checkout location: identical git files then
    hash differently across machines."""
    if isinstance(obj, dict):
        return {k: strip_machine_paths(v) for k, v in obj.items()
                if k != "source_path"}
    if isinstance(obj, list):
        return [strip_machine_paths(v) for v in obj]
    return obj


class ResolvedSpec:
    """An ArchSpec with all dimensions resolved to concrete integer values."""

    def __init__(self, spec: ArchSpec, dim_values: Dict[str, int], config: Dict[str, Any],
                 view_values: Dict[str, Any], site: Any = None,
                 declared_by: Any = None,
                 checkpoint: Optional[CheckpointHeader] = None):
        self.spec = spec
        self.dim_values = dim_values  # {symbol_name: int_value}
        self.config = config
        self.view_values = view_values  # {VIEW_KEYS member: native config value}
        # Where a model invokes the component this spec isolates
        # (`eval.transforms.CallSite`): its checkpoint address, its role and
        # the checkpoint's saved formats. None for a whole model, whose call
        # sites are in its own graph.
        self.site = site
        # The `SpecLoader` that installed this spec's `dim_symbols` in the
        # dim registry. Its layer lives as long as the loader does, and the
        # registry is where a weight's `parts:` asks whether its symbol may
        # be sharded — so a resolution that outlives its loader must hold
        # it, or a spec-declared shardable symbol stops being one halfway
        # through the analysis of the spec that declares it.
        self.declared_by = declared_by
        # The checkpoint's safetensors header this resolution reads
        # (`dim_resolution.CheckpointHeader`): `checkpoint.`-bound dims, and
        # the precision the model ships its unquantised tensors at
        # (`shipped_dtype`). Absent for a resolution with no checkpoint.
        self.checkpoint = (checkpoint if checkpoint is not None
                           else CheckpointHeader.absent(
                               "resolved without a checkpoint"))
        # Memo slots for the spec_tree walks over this resolution
        # (`iter_blocks` / `iter_tree` / `iter_leaves`). Declared here so the
        # walks assign a field the object owns instead of probing for one.
        self._spec_tree_blocks_cache: Any = None
        self._spec_tree_ir_cache: Any = None
        self._spec_tree_cache: Any = None
        self._appended_cache: Optional[Dict[str, AppendedStream]] = None
        # Every call the model makes, with what its analysis hands each
        # (`eval.transforms.CallSite.calls`).
        self._call_sites_cache: Any = None
        # This resolution with its output heads read at the sampled
        # positions (`eval.transforms.splice_sampled_head`), as a one-tuple
        # once asked (its content None: no head to splice): one clone, so
        # every walk of it and every tree built off it share one spec tree.
        self._sampled_head_cache: Optional[Tuple[Any]] = None

    def identity_hash(self) -> str:
        """SHA256 prefix over a stable CONTENT serialization of this
        resolution: the spec and the values its dims bound — the identity
        a calibration row is keyed by (`BenchmarkKey.spec_hash`).

        The spec goes in through Pydantic's `model_dump` (model list,
        blocks, the components dict every referenced block_graph is
        resolved into at load time), so edits to `blocks:`, component refs,
        dim formulas or quant fields invalidate stale calibration rows. The
        dim values and `model_type` go in too, so different bindings of one
        spec yaml hash apart. Machine-specific `source_path` fields are
        stripped (`strip_machine_paths`): a row keyed by this hash must
        match on any machine, not just where it was recorded. A spec that
        cannot serialize raises — a hash blind to structure would be worse.
        """
        parts: Dict[str, Any] = {}
        spec = self.spec
        if spec is not None:
            parts["spec_name"] = spec.name or ""
            # `mode="json"`: round-trippable JSON (tuple→list, datetime→str).
            dump = getattr(spec, "model_dump", None)
            if not callable(dump):
                raise ValueError(
                    f"{spec.name or 'spec'} has no model_dump; its identity "
                    "cannot be hashed structurally")
            parts["spec_content"] = strip_machine_paths(
                dump(mode="json", exclude_none=True))
        parts["dim_values"] = dict(sorted(self.dim_values.items()))
        parts["model_type"] = (self.config or {}).get("model_type", "")
        blob = json.dumps(parts, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def shipped_dtype(self) -> Tuple[Optional[str], Optional[str]]:
        """`(dtype, source)` of the precision this model ships the tensors
        no quant scheme covers at (`dtypes.stated_default_dtype`): the
        spec's declaration, else what the checkpoint header stores them at.
        `config.torch_dtype` only cross-checks it."""
        quant_context = self.spec.quant_context
        return stated_default_dtype(
            quant_context, self.config, self.language_config(),
            self.checkpoint.unquantised_float_dtypes(quant_context))

    def language_config(self) -> Tuple[str, Dict[str, Any]]:
        """Where this config keeps the language model's own fields, as the
        spec's `d_model` binding states it (`dim_resolution.language_config`)."""
        return language_config(self.spec.dims, self.config)

    def appended_streams(self) -> Dict[str, "AppendedStream"]:
        """Each appended stream by name, with the iteration each element is
        appended after (-1: the stack's input). A stream whose `after`
        names no iteration has no elements and is not published."""
        if self._appended_cache is None:
            found: Dict[str, AppendedStream] = {}

            def walk(layers, group) -> None:
                for layer in layers:
                    if layer.layers:
                        walk(layer.layers, layer)
                        continue
                    for name, spec in layer.appended_streams.items():
                        depth = self.get_dim(layer.repeat)
                        found[name] = AppendedStream(
                            name=name, writer=layer, group=group,
                            iterations=spec.iterations(
                                depth, self.config, self.get_dim),
                            writers=spec.writers(
                                depth, self.config, self.get_dim))

            walk(self.spec.model, None)
            self._appended_cache = found
        return self._appended_cache

    def check_appended_streams(self) -> None:
        """What resolve alone can say about who reads an appended stream.

        A stack that names one in `inputs:` reads it at its block boundary:
        its iteration j binds element j, so it repeats once per element
        (`ArchSpec._check_stack_call_arity` has already bound one formal to
        each input). With no elements it must not run at all — binding
        anything else would be the silent fallback the stream exists to
        remove. A group that repeats around the stack that appends or the
        one that reads would copy both together and give element j several
        readers per copy; no pairing is defined for that, so it is refused.
        Readers at an inner node (`read: each`) pair in
        `Program.stream_pairings`, which sees which iterations' components
        read; the appending stack's own `read: all` readers are grouped by
        what they see when its blocks are enumerated
        (`spec_tree.build`). A model-level `op:` reads the whole stream;
        joining it (`axis:`) needs at least one element, since the op runs
        whatever the config says. A component's `read: concat` with no
        elements is asked of its reading positions
        (`Program.whole_stream_pairings`).
        """
        streams = self.appended_streams()

        def walk(layers, group) -> None:
            for layer in layers:
                if layer.layers:
                    walk(layer.layers, layer)
                    continue
                named = [ref for ref in layer.inputs or () if ref in streams]
                involved = (layer.appended_streams or layer.read_streams
                            or layer.concat_streams or named)
                if layer.op is not None:
                    # A model-level op reads the whole stream once, and it
                    # always runs: joined, it has something to join.
                    empty = [ref for ref in named
                             if layer.axis is not None
                             and not streams[ref].iterations]
                    if empty:
                        raise ValueError(
                            f"model-level op {layer.name!r} joins appended "
                            f"stream {empty[0]!r} along `axis: "
                            f"{layer.axis}`, but the stream has no elements "
                            f"in this config; an op that always runs has "
                            f"nothing to join")
                    named = []
                if (involved and group is not None
                        and group.repeat is not None
                        and self.get_dim(group.repeat) != 1):
                    raise ValueError(
                        f"group {group.name!r} repeats {group.repeat!r} "
                        f"around {layer.name!r}, which appends to or reads "
                        f"an appended stream; element-wise pairing is "
                        f"defined only for a group that runs once")
                for ref in named:
                    k = len(streams[ref].iterations)
                    repeat = self.get_dim(layer.repeat)
                    if repeat != k:
                        raise ValueError(
                            f"{layer.name!r} repeats {repeat} times but reads "
                            f"appended stream {ref!r}, which has {k} "
                            f"element(s): iteration j of the reader reads "
                            f"element j")

        walk(self.spec.model, None)

    def with_spec(self, spec: ArchSpec) -> "ResolvedSpec":
        """The same resolution over a rewritten spec: dims, config, view
        values and call site carried over, so a clone cannot drop one."""
        return ResolvedSpec(spec, dict(self.dim_values), self.config,
                            dict(self.view_values), site=self.site,
                            declared_by=self.declared_by,
                            checkpoint=self.checkpoint)

    @property
    def model_type(self) -> str:
        """The checkpoint's declared architecture id — read here, where the
        match against `match.model_type` was made, and nowhere downstream."""
        value = self.config.get("model_type")
        return str(value) if value else "unknown"

    def get_dim(self, expr_str: str) -> int:
        """Evaluate a dimension expression to a concrete integer."""
        return eval_dim_expr(expr_str, self.dim_values)

    def layers_selected(self, assign, total=None) -> set:
        """Resolve a `LayerSelection` to a set of layer indices.

        THE one place a declared set of layers becomes indices. A block
        assignment names the layers that run a template; a
        `quant_context.overrides` row's `select.layers` names the layers it
        speaks about. Both are `LayerSelection`s and both resolve here, so
        the two readings cannot drift.

        Every selector present on the assignment is resolved independently
        and the results are INTERSECTED — a layer is selected only when it
        satisfies all of them. A single selector (the common case) intersects
        to itself, so the historical one-selector behaviour is byte-identical.
        The intersection is what lets a hybrid backbone express an axis
        crossing: DSv4's `from: config.compress_ratios, is: "128"` (which
        layers are HCA) AND `range: [0, "n_hash_layers - 1"]` (which of those
        are also hash-MoE) pick out `HCA ∩ hash` without spelling the indices
        out for one particular layer count.

        `total` (the group's repeat count) is required only by the
        `stride` selector, which enumerates `range(offset, total, stride)`.
        """
        selectors = []
        if assign.config_list is not None:
            selectors.append(self._config_list_indices(
                assign, total, base=assign.config_list_base or 0))
        if assign.within is not None:
            selectors.append(self._config_list_indices(
                assign.within, total, base=assign.within.base or 0))
        if assign.range:
            start = self.get_dim(assign.range[0])
            end = self.get_dim(assign.range[1])
            selectors.append(self._within_span(
                set(range(start, end + 1)), total,
                f"range: [{assign.range[0]}, {assign.range[1]}]"))
        if assign.layers:
            selectors.append(self._within_span(
                {self.get_dim(l) for l in assign.layers}, total,
                f"layers: [{', '.join(str(l) for l in assign.layers)}]"))
        if assign.stride is not None:
            if total is None:
                selectors.append(set())
            else:
                stride = self._coerce_dim_int(assign.stride)
                offset = (self._coerce_dim_int(assign.offset)
                          if assign.offset is not None else 0)
                if not stride or stride < 1:
                    selectors.append(set())
                else:
                    selectors.append(set(range(offset, total, stride)))
        if not selectors:
            return set()
        return set.intersection(*selectors)

    @staticmethod
    def _within_span(indices: set, total, selector: str) -> set:
        """`indices` if every one is a layer of the container, else refuse.

        A `range:` / `layers:` selector, or a config list of layer numbers,
        names layers explicitly, so an index at or past the container's
        `repeat` is not a layer of it: kept, it would be counted as one
        (gpt-oss at `n_layers=12` counted 18 backbone layers). A per-layer
        schedule list (`is:`) may legitimately run long (a shared schedule
        array feeding a sibling stack) and is clamped instead; an explicit
        index has no such reading. `total` None (no `repeat:` on the
        container) leaves nothing to check against.
        """
        if total is None:
            return indices
        outside = sorted(i for i in indices if not 0 <= i < total)
        if outside:
            raise LayerSpanError(
                f"per-layer selector `{selector}` names layer(s) {outside} "
                f"outside this stack's {total} layers (0..{total - 1}).")
        return indices

    def _config_list_indices(self, assign, total=None, base: int = 0) -> set:
        """Layers whose per-layer `config.<list>` entry == `assign.equals`,
        or — with no `equals` — the layer numbers the list holds (one
        number is a list of one), counted from `base`: an inner stream
        reader's `{nearest: {from: config.<list>}}` set (`StreamLayerSet`),
        or a `per_layer` generator's `from` alone.

        Read the raw list directly — `_resolve_config_ref` coerces to
        numeric and would drop a list — through the same dotted-path walk
        `dims:` uses, so `config.text_config.layer_types` reads the nested
        list a multimodal config keeps there. An absent field selects nothing
        ONLY when the selector declared it may be absent
        (`from: "config.X ?? absent"`, GLM-5.1's indexer_types);
        otherwise it is a misspelt field, and says so.
        """
        lst = config_path_value(assign.config_list, self.config)
        if lst is None:
            if not assign.config_list_optional:
                raise ValueError(
                    f"per-layer selector reads config.{assign.config_list}, "
                    f"which this config does not carry. If the list may be "
                    f"absent, declare it: `from: config."
                    f"{assign.config_list} ?? absent`.")
            return set()
        if assign.equals is None:
            numbers = [lst] if isinstance(lst, int) else lst
            if not isinstance(numbers, list) or not all(
                    isinstance(v, int) and not isinstance(v, bool)
                    for v in numbers):
                raise ValueError(
                    f"config.{assign.config_list} is {lst!r}, not a list of "
                    f"layer numbers; a per-layer list selects layers with "
                    f"`is: <value>`")
            below = sorted(v for v in numbers if v < base)
            if below:
                raise ValueError(
                    f"config.{assign.config_list} is {lst!r}: {below} "
                    f"precede its first layer number {base}")
            return self._within_span(
                {v - base for v in numbers}, total,
                f"from: config.{assign.config_list}"
                + (f", base: {base}" if base else ""))
        if not isinstance(lst, list):
            raise ValueError(
                f"config.{assign.config_list} is {type(lst).__name__}, not "
                f"the per-layer list the selector reads")
        # Type-tolerant match: per-layer schedule lists are encoded as
        # strings ("full"/"shared", "sliding_attention"), ints (MiMo
        # hybrid_layer_pattern 0/1, MiniMax attn_type_list), or bools
        # across vendors. Compare stringified so `equals: "1"` matches
        # int 1 and `equals: full` matches "full" — avoids a silent
        # 0-match that would leak all layers into a sibling `rest`.
        want = str(assign.equals)
        idxs = {i for i, v in enumerate(lst) if str(v) == want}
        if total is not None:
            idxs &= set(range(total))
        return idxs

    def execution_schedule(self, container) -> ExecutionSchedule:
        """The executions of a repeated stack in order, with the stored layer
        and pass of each (`loop_schedule`). THE one place a `loop:` is read:
        without one each stored layer runs once and ordinal = stored layer.
        """
        stored = self.get_dim(container.repeat)
        loop = container.loop
        if loop is None:
            return ExecutionSchedule.unlooped(stored)
        where = f"model layer {container.name!r}"
        return ExecutionSchedule.looped(
            stored, self.get_dim(loop.layers[0]), self.get_dim(loop.layers[1]),
            self.get_dim(loop.passes), loop.gate is not None, where)

    def resolve_assignment_groups(self, container, total=None) -> List[Dict[str, Any]]:
        """Canonical per-layer template resolution — THE single place layer→
        template selection lives.

        Turns a container's (`ModelLayer` or group sub-layer) `assignments`
        into ordered groups, each a dict:
          {template, indices (sorted ints), dim_overrides, assign,
           assignment_index}
        (`assignment_index` is the assignment's position in the container's
        own `assignments` list — the declaration an editor changes, which a
        template name cannot pick out once two assignments share one)
        applying declaration-order precedence (a layer is claimed by the
        FIRST matching assignment), a trailing `rest:` filling the remainder,
        and every selector type (range / layers / stride / config_list) via
        `layers_selected`.

        Every consumer — param/FLOPs walk (engine), block enumeration
        (spec_tree), per-class KV (kv_cache), diagram layout, symbolic
        aggregation — resolves through this so the per-layer mapping can
        never diverge across paths (params vs FLOPs vs KV vs graph). Empty
        (count-0) groups are RETAINED so callers that display or skip them
        keep their existing behaviour; each caller decides whether to drop
        them.
        """
        assignments = container.assignments or []
        if total is None:
            rep = container.repeat
            total = self.get_dim(rep) if rep else None
        # In a looped stack a selector names STORED layers and the groups
        # hold the ordinals of their executions (`execution_schedule`): every
        # execution, or with `pass:` that pass's. Precedence runs over
        # executions, so a pass-2 entry first claims pass 2 and the entries
        # after it describe the rest.
        schedule = (self.execution_schedule(container)
                    if container.loop is not None else None)

        def _ovr(a):
            return {k: str(v) for k, v in a.dims.items()} if a.dims else None

        def _executions(assign, stored: set) -> set:
            if schedule is None:
                if assign.pass_ is not None:
                    raise ValueError(
                        f"model layer {container.name!r}: `pass: "
                        f"{assign.pass_}` selects a pass of a loop, and this "
                        f"stack declares no `loop:`")
                return stored
            if assign.pass_ is None:
                return {o for s in stored for o in schedule.ordinals_of(s)}
            if assign.pass_ > schedule.passes():
                raise ValueError(
                    f"model layer {container.name!r}: `pass: {assign.pass_}` "
                    f"names a pass the loop does not run (passes: "
                    f"{schedule.passes()})")
            # `pass:` selects inside the loop: `from: …, is: full, pass: 2`
            # is the loop's full layers in pass 2, and the loop's range is
            # not written a second time. A selection that names layers and
            # none of them loops runs no pass but the first.
            looped = {s for s in (stored if assign.selects_layers
                                  else range(schedule.stored_layers))
                      if schedule.in_loop(s)}
            if not looped:
                raise ValueError(
                    f"model layer {container.name!r}: an assignment with "
                    f"`pass: {assign.pass_}` selects stored layer(s) "
                    f"{sorted(stored)[:8]}, none inside the loop "
                    f"[{schedule.loop[0]}, {schedule.loop[1]}]; only loop "
                    f"layers run more than one pass")
            return {schedule.ordinal(s, assign.pass_) for s in looped}

        covered: set = set()
        rest_assign = None
        groups: List[Dict[str, Any]] = []
        rest_index = None
        for index, assign in enumerate(assignments):
            if assign.rest:
                rest_assign, rest_index = assign, index
                continue
            indices = _executions(assign, self.layers_selected(assign, total))
            indices -= covered            # declaration-order precedence
            covered |= indices
            groups.append({
                "template": assign.template,
                "indices": sorted(indices),
                "dim_overrides": _ovr(assign),
                "assign": assign,
                "assignment_index": index,
            })
        if rest_assign is not None and total is not None:
            every = (set(range(total)) if schedule is None
                     else set(schedule.layer_ordinals()))
            groups.append({
                "template": rest_assign.template,
                "indices": sorted(every - covered),
                "dim_overrides": _ovr(rest_assign),
                "assign": rest_assign,
                "assignment_index": rest_index,
            })
        return groups

    def refuse_literal_rebinding(self, site: str, label: str, raw: Any,
                                 value: Any, *, rebinds: bool = False) -> None:
        """A literal bound to a label named like a model dim, wherever a
        `dims:` map binds one — a graph node, a model-root layer, an
        assignment.

        `rebinds` is an assignment's map, whose whole purpose is to give the
        model's dim another value for the layers it selects (Gemma's full
        layers at `n_kv_heads_full`, a playground group at 128 experts): a
        number different from the model's is that value and passes. A number
        EQUAL to the model's value rebinds nothing and is the dim written
        as a number, refused like anywhere else.

        A binding is identified by its label's name, so `d_model: "256"` is
        an axis whose identity is the model's `d_model` and whose value is
        256. Two values for one identity is refused far downstream, at the
        per-rank projection, as if the engine could not proceed; it is a
        spec choice, and named here where it is made. At the model's own
        value the literal is the model's dim written as a number, which
        stays right only while the config does: it is named here too. A
        literal is any expression that names no dim (`"16 * 160"` as much as
        `"2560"`); an expression over dims carries its own identity. The
        model's top-level `dims:` is where each dim gets its value, and is
        not a rebinding."""
        model = self.dim_values
        if label not in model or read_law(str(raw)).free_symbols:
            return
        binding = f"`dims: {{ {label}: {raw} }}`"
        if rebinds and value != model[label]:
            return
        if value == model[label]:
            raise SpecRefusal(
                f"{site} binds {binding}, the value of the model's dim "
                f"`{label}` written as a number",
                declaration=f"{site} dims.{label}",
                fix=(f"bind the dim the width comes from: `{label}: {label}` "
                     f"when it is the model's {label}, or the dim the forward "
                     f"source reads for it"))
        raise SpecRefusal(
            f"{site} binds {binding}, but `{label}` is the model's dim "
            f"({model[label]}): a literal under that name is read as the "
            f"model's {label} and carries a second value for it",
            declaration=f"{site} dims.{label}",
            fix=(f"when `{label}` is a per-layer value, bind it only where "
                 f"it varies and drop it from the model's top-level `dims:`; "
                 f"otherwise name this width as a dim of its own in the "
                 f"model's `dims:` (registered in dims.yaml or the spec's "
                 f"`dim_symbols:`), read from the config field that states it "
                 f"— `config.<list>.<i>` reads one entry of a per-layer list — "
                 f"and bind `{label}: <that dim>`"))

    def _coerce_dim_int(self, v) -> int:
        """Integer literal as-is, else resolve as a dim expression."""
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except (TypeError, ValueError):
            return int(self.get_dim(v))


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spec registry
# ---------------------------------------------------------------------------

_default_loader_generation = 0


@memoize(maxsize=1)
def get_default_loader() -> "SpecLoader":
    """Module-level cached SpecLoader for routes/CLI that don't customize
    `base_dir`. Profiling /inference/estimate showed ~5.9s/request spent
    in `SpecLoader.__init__._load_all` parsing 339 yaml files — three
    times per request because both the route AND the inference pipeline
    instantiated their own loader. Sharing one instance turns the
    SpecLoader cost into a one-shot startup tax. Tests + callers that
    need an isolated loader (writable component dir, hermetic test
    fixtures) keep using `SpecLoader(...)` directly."""
    return SpecLoader()


def invalidate_default_loader() -> None:
    """Drop the cached loader. Called by `/refresh-specs` (future) or
    after a runtime promotion writes a new yaml file — otherwise the
    next request would still see the pre-write component manifest."""
    global _default_loader_generation
    get_default_loader.cache_clear()
    _default_loader_generation += 1


def default_loader_generation() -> int:
    """How many times this process has dropped its default loader. A
    worker process that answers for this one (`backend.measure.pool.submit`) drops
    its own when the number it last saw moves."""
    return _default_loader_generation


# Spec-miss hook. `match()` answers a config from the tree the loader was
# constructed over. A deployment may hold specs elsewhere — the service layer's
# cross-VM cache of agent-generated specs is one — and registers a callable
# here that, given the model id, materializes that spec into an OVERLAY
# directory (`models/`, `components/drafts/`, `snapshots/`) and returns its
# root, or None for a miss. The loader attaches the overlay and matches again.
# Where the overlay comes from, who may see it and where it lives on disk are
# the hook's business; the loader touches neither the network nor the repo
# tree. Same registration idiom as the anon spec-gen gate.
_spec_miss_hook = None  # Optional[Callable[[str], Optional[str]]]


def set_spec_miss_hook(fn) -> None:
    global _spec_miss_hook
    _spec_miss_hook = fn


def overlay_kwargs(overlay_dir: Optional[str]) -> dict:
    """Build `SpecLoader` overlay kwargs from a per-run/per-user overlay root
    (`<dir>/models` + `<dir>/components/drafts`). Empty dict when `overlay_dir`
    is falsy — so callers can always splat `**overlay_kwargs(x)`."""
    if not overlay_dir:
        return {}
    return {
        "overlay_model_dirs": [os.path.join(overlay_dir, "models")],
        "overlay_draft_dirs": [os.path.join(overlay_dir, "components", "drafts")],
    }


def custom_spec_file(model_id: str, overlay_dir: Optional[str] = None) -> Optional[str]:
    """Path of a custom model's spec: the run overlay identity yaml when the
    agent is writing there, else the cache `spec.yaml`. None when neither
    exists. Overlay wins so generate/refine of the same custom model do not
    share the cache file as their write target."""
    if not is_custom_id(model_id):
        return None
    if overlay_dir:
        models = Path(overlay_dir) / "models"
        safe = model_id.replace("/", "__")
        for name in (f"{safe}.yaml", "spec.yaml"):
            path = models / name
            if path.is_file():
                return str(path)
    cache = model_path(strip_prefix(model_id)) / "spec.yaml"
    return str(cache) if cache.is_file() else None


def arch_fingerprint(spec: ArchSpec) -> str:
    """A canonical hash of a spec's resolved architecture — identity and
    location fields (`name`, `source_path`, `unresolved_note`, `derived_from`)
    excluded, so a rename / re-parent is not a difference. Two specs with the
    same fingerprint resolve to the same architecture. Keys the golden and
    drift tests, where a rename or re-parent must not read as a change."""
    payload = json.dumps(
        spec.model_dump(
            mode="json",
            exclude={"name", "source_path", "unresolved_note", "derived_from"},
        ),
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: The library tier of this checkout. A loader built on another tree (a PR
#: preview, a test fixture) loads its `models/` as that tree's, not the
#: library's.
_LIBRARY_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "models"))


def _withdraw_dim_layers(tokens: Dict[str, int]) -> None:
    """A collected loader's `dim_symbols` layers leave with it."""
    for token in list(tokens.values()):
        dim_registry.withdraw_local(token)
    tokens.clear()


def _bound_dim_names(raw: Any) -> FrozenSet[str]:
    """Every symbol some `dims:` mapping in a spec document binds."""
    found: set = set()
    stack = [raw]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "dims" and isinstance(value, dict):
                    found.update(str(k) for k in value)
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return frozenset(found)


class SpecLoader:
    """Loads shared components + model configs, assembles into ArchSpec."""

    def __init__(self, base_dir: Optional[str] = None, *,
                 overlay_model_dirs: Optional[List[str]] = None,
                 overlay_draft_dirs: Optional[List[str]] = None):
        if base_dir is None:
            base_dir = os.path.dirname(__file__)
        self._components_dir = os.path.join(base_dir, "components")
        self._models_dir = os.path.join(base_dir, "models")
        # Out-of-tree dirs layered on top of the shared tree — lets the
        # per-user serving path load a signed-in user's PRIVATE generated spec
        # WITHOUT materializing it into the shared repo tree. Model specs from
        # these dirs join `_specs`; draft components join `_draft_lib` (still
        # opt-in per spec via `drafts:[...]`). The shared read-only
        # `components/library` is always loaded so a private spec can resolve
        # its built-in component refs.
        self._overlay_model_dirs = [str(d) for d in (overlay_model_dirs or [])]
        self._overlay_draft_dirs = [str(d) for d in (overlay_draft_dirs or [])]
        self._component_lib: Dict[str, BlockGraph] = {}
        #: Model configs that failed to load, kept as first-class state:
        #: `[(path, match_types, error_text)]`. Boot stays available — one
        #: broken spec must not take the server down — but USE refuses:
        #: `match()` raises when the requested model_type is claimed by a
        #: failed spec, instead of silently serving a different match or
        #: None. A component that fails to load is covered transitively —
        #: the spec referencing it fails schema resolution and lands here.
        self._model_load_failures: list = []
        # Component-scoped custom_ops, keyed by component name → {op_name: CustomOpSpec}.
        # Merged into ArchSpec.custom_ops at spec-assembly time, restricted
        # to the components a given spec actually pulls in.
        self._component_custom_ops: Dict[str, Dict[str, "CustomOpSpec"]] = {}
        # Draft components — `components/drafts/*.yaml`. Parsed eagerly but
        # NOT in `_component_lib` by default; only specs that opt in via
        # `drafts: [name1, ...]` see them, and only for the duration of
        # their own load. This keeps agent-generated components from
        # polluting the global namespace until promotion to library/.
        self._draft_lib: Dict[str, BlockGraph] = {}
        self._draft_custom_ops: Dict[str, Dict[str, "CustomOpSpec"]] = {}
        #: Draft namespace provenance. A name may be registered once unless a
        #: later same-named document has the identical conservative structure.
        #: Formal components stay outside `_draft_lib`; their signatures join
        #: the same comparison when a draft tries to reuse their name.
        self._draft_signatures: Dict[str, str] = {}
        self._draft_sources: Dict[str, str] = {}
        # Drafts that failed to parse: name → error string. Surfaced as a
        # spec-load error iff some spec lists the broken draft.
        self._draft_errors: Dict[str, str] = {}
        # Draft names claimed twice with different structures. Raised iff
        # some spec lists the name (`DraftNameConflict`).
        self._draft_conflicts: Dict[str, DraftNameConflict] = {}
        self._specs: Dict[str, ArchSpec] = {}
        # The authored document behind each loaded spec, kept for its
        # `match:` — the one authoring fact whose consumer is the loader
        # itself, so it never crosses into the assembly product.
        self._documents: Dict[str, SpecDocument] = {}
        # custom-model dir name → registered spec name in `_specs`.
        # Populated by `_load_custom_spec_if_any`; consumed by `match()`
        # when caller passes `model_id="custom:<dir>"` to pin a specific
        # registered custom spec instead of letting model_type matching
        # pick from multiple custom specs that share the same model_type.
        self._custom_dir_to_spec: Dict[str, str] = {}
        #: model_id → spec name, built from every loaded spec's
        #: `match.model_id`. The matcher's index: `match()` resolves a request
        #: by looking its model_id up here, with no scoring fallback.
        self._id_index: Dict[str, str] = {}
        #: spec name → the token of the `dim_symbols` layer this loader
        #: installed for it (`dims.LocalSymbols`). The layers live as long as
        #: the loader holds the spec: replaced with it, rolled back with a
        #: failed overlay, and withdrawn when the loader is collected.
        self._dim_tokens: Dict[str, int] = {}
        weakref.finalize(self, _withdraw_dim_layers, self._dim_tokens)
        #: Serializes what mutates a constructed loader: `attach_overlay`
        #: (a spec-miss can attach from two request threads at once) and the
        #: token swaps of the specs it loads.
        self._mutation_lock = threading.RLock()
        self._load_all()
        # Strict only for a pure shared tree — two shared specs claiming one id
        # is a spec-set error CI must catch (contract D). When constructed with
        # a user overlay (the per-request analyze path), a collision is
        # cross-namespace: shared wins with a warning, never a crash (contract
        # E), or a user whose stored spec shares an id with a later shared
        # identity file would 500 on every request.
        self._build_id_index(strict=not self._overlay_model_dirs)

    def _load_all(self) -> None:
        """Load component library, then model configs."""
        # 1. Load shared components from `components/library/` and
        #    `components/adapters/`. Both feed the same flat namespace —
        #    YAML refs like `op: pre_norm` resolve against the union, so
        #    cross-directory references work transparently.
        #    library/ — distinct topology building blocks (any model can
        #               compose). May carry a model-family prefix in their
        #               name as a "first observed in" historical label, but
        #               are not scoped to that family.
        #    adapters/ — pure binding/rename shims around a library
        #               component for a specific model's checkpoint
        #               naming. No new structure of their own.
        for sub in ("library", "adapters"):
            d = os.path.join(self._components_dir, sub)
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if fname.endswith((".yaml", ".yml")):
                    path = os.path.join(d, fname)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            raw = load_yaml(f)
                        name, component = _parse_component_document(raw, path)
                        self._component_lib[name] = component
                        # Component-scoped custom_ops (e.g. GLA's
                        # `gla_simple_recurrence` recurrence kernel) live
                        # at the component YAML's top level. Without
                        # capturing them here they're silently dropped —
                        # the engine and the tooltip formula
                        # renderer both read from a single dict
                        # so collect them by source component for later
                        # merge into the assembled ArchSpec.custom_ops.
                        for cop_name, cop_raw in (raw.get("custom_ops") or {}).items():
                            self._component_custom_ops.setdefault(name, {})[cop_name] = (
                                CustomOpSpec(**cop_raw) if isinstance(cop_raw, dict) else cop_raw
                            )
                    except (OSError, yaml.YAMLError, ValueError) as e:
                        logger.warning(f"Failed to load component {path}: {e}")

        # 1b. Load drafts — agent-generated, spec-scoped components. Parse
        #     eagerly so syntax errors surface, but don't merge into the
        #     global library: only specs that explicitly opt in (via the
        #     top-level `drafts: [name, …]` list) get them, and only for
        #     the duration of that spec's assembly. Parse failures and
        #     name conflicts are deferred to the first spec that lists the
        #     draft — unreferenced broken drafts stay silent, because a
        #     draft file no spec lists is part of no spec.
        self._load_drafts()

        # 1c. Static lint pass: catch components that wrap a sub-component
        #     of a different role under `repeat:` without tagging the
        #     call-site `role:`. Concretely: an `ffn_moe` wrapper invoking
        #     a `ffn_dense` per-expert FFN — without the call-site tag,
        #     leaves under the inner FFN inherit the inner's `ffn_dense`
        #     role, and any model spec's `quant_context.per_role.ffn_moe`
        #     misses every expert leaf. We saw this dominate V4-Pro's
        #     spec-vs-actual diff (1.6T → 815B) until V4-Pro's MoE
        #     wrappers grew explicit `role: ffn_moe` tags. Warning-level
        #     (not error) so existing specs don't suddenly fail to load
        #     while the library completes its migration.
        self._check_role_propagation_in_components()

        # 2. Load model configs (reference components) — shared tree first,
        #    then any per-user overlay dirs (overlay wins on name collision).
        for mdir in [self._models_dir, *self._overlay_model_dirs]:
            if not os.path.isdir(mdir):
                continue
            for fname in sorted(os.listdir(mdir)):
                if fname.endswith((".yaml", ".yml")):
                    path = os.path.join(mdir, fname)
                    try:
                        spec = self.load_model_spec(path)
                        if spec:
                            self._specs[spec.name] = spec
                    except (OSError, yaml.YAMLError, ValueError, SpecLoadError) as e:
                        # SpecLoadError covers a broken `base:` (missing/cyclic
                        # base) in an identity file — a broken spec is recorded,
                        # not a boot crash; the id is refused on request.
                        logger.warning(f"Failed to load model config {path}: {e}")
                        types: list = []
                        model_ids: list = []
                        raw_text = ""
                        try:
                            with open(path, "r", encoding="utf-8") as fh:
                                raw_text = fh.read()
                            raw_doc = load_yaml(raw_text) or {}
                            match_blk = raw_doc.get("match") or {}
                            types = list(match_blk.get("model_type") or [])
                            # A broken identity file still declares its id, so
                            # requesting it can be refused by id (contract D)
                            # instead of silently falling through to the family.
                            # Tolerate a scalar id in a malformed file.
                            mids = match_blk.get("model_id") or []
                            model_ids = mids if isinstance(mids, list) else [mids]
                        except Exception:
                            pass
                        # The structured read fails outright when the YAML does
                        # not parse at all — but the file still names the id it
                        # claims, and dropping it there turns a broken spec into
                        # a silent miss that spec-gen re-runs forever (contract
                        # D). Fall back to a text scan of the `match:` header so
                        # the id is refused loudly even from an unparsable file.
                        if not model_ids and raw_text:
                            types = types or _scan_match_field(raw_text, "model_type")
                            model_ids = _scan_match_field(raw_text, "model_id")
                        self._model_load_failures.append(
                            (path, types, model_ids, str(e)))

    def library_component(self, name: str) -> Optional[BlockGraph]:
        """A component from the global library by name, or None when the
        library declares none of that name. The one public door to the
        library table; callers decide what a miss means for them."""
        return self._component_lib.get(name)

    def decode_form(self, spec_name: str, name: str) -> Optional[BlockGraph]:
        """The component `name` that spec `spec_name`'s decode phase binds in
        (a `decode_variant:`), or None when neither table holds it.

        The library's, or a draft the spec opted into under `drafts:` — the
        same scope a draft has at load. A draft's decode form is a draft
        too (library components never reference drafts), and the spec's
        component closure does not hold it: nothing in the prefill graph
        calls it."""
        found = self._component_lib.get(name)
        if found is not None:
            return found
        document = self._documents.get(spec_name)
        if document is not None and name in document.drafts:
            return self._draft_lib.get(name)
        return None

    def _check_role_propagation_in_components(self) -> None:
        """Warn when a wrapper component invokes another component with
        a different role in copies (`repeat:` / `chain:`, read through
        `GraphNode.copies`) and forgets to set call-site
        `role:` — the pattern that silently misroutes
        `quant_context.per_role` for MoE wrappers around dense FFNs.

        Scans only `library/` + `adapters/` (the global namespace);
        drafts are per-spec scoped and validated when activated. An
        offender refuses the library at load: the corpus has none, and
        a warning nobody reads is how the next one would ship.
        """
        offenders: List[str] = []
        for name, comp in self._component_lib.items():
            if comp.role is None:
                continue
            for nname, gn in comp.graph.items():
                if not gn.copies:
                    continue
                if gn.role:
                    continue
                inner = self._component_lib.get(gn.op)
                if inner is None or inner.role is None:
                    continue
                if inner.role == comp.role:
                    continue
                offenders.append(
                    f"  {name!r} (role={comp.role!r}) "
                    f"→ node {nname!r}: invokes {gn.op!r} "
                    f"(role={inner.role!r}) in copies "
                    f"without `role:` override"
                )
        if offenders:
            raise ValueError(
                "role propagation gap detected: parent component declares "
                "one role but invokes a child component with a different "
                "role in copies (`repeat:` / `chain:`), and the call site does not set "
                "`role:`. Inner leaves will inherit the child's role, so "
                "any `quant_context.per_role.<parent_role>` mapping for "
                "this checkpoint family will not reach those leaves. "
                "Add `role: <parent_role>` to the call site.\n"
                + "\n".join(offenders)
            )

    def _load_drafts(self) -> None:
        """Parse `components/drafts/*.yaml` (+ any per-user overlay draft dirs)
        into `_draft_lib` (separate from `_component_lib`). A spec's `drafts:`
        list is what makes a draft file part of it, so neither kind of broken
        draft stops a load by itself: parse failures are recorded on
        `_draft_errors` and name conflicts on `_draft_conflicts`, each raised
        only when a spec lists that draft."""
        for d in [os.path.join(self._components_dir, "drafts"),
                  *self._overlay_draft_dirs]:
            self._scan_draft_dir(d)

    def _scan_draft_dir(self, d: str) -> None:
        if not os.path.isdir(d):
            return
        for fname in sorted(os.listdir(d)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            path = os.path.join(d, fname)
            stem = Path(path).stem
            previous_source = self._draft_sources.get(stem)
            formal_collision = (
                previous_source is None and stem in self._component_lib
            )
            try:
                with open(path, "r", encoding="utf-8") as f:
                    body = f.read()
                raw = load_yaml(body)
                _, component = _parse_component_document(raw, path)
                signature = component_structure_signature(body, path=path)
            except (OSError, yaml.YAMLError, ValueError) as e:
                if previous_source is not None or formal_collision:
                    conflict = DraftNameConflict(
                        f"draft component {stem!r} from {path} conflicts with "
                        f"the existing component from "
                        f"{previous_source or 'library/'}; rename this draft "
                        "and update the target spec, or reuse the existing "
                        "component", path=path)
                    conflict.__cause__ = e
                    self._draft_conflicts.setdefault(stem, conflict)
                    continue
                # Defer the error: only a spec that actually lists this
                # draft needs to fail. Index by stem (file basename) so
                # the spec's `drafts: [stem]` list resolves even when the
                # YAML's own `name:` field couldn't be parsed.
                self._draft_errors[stem] = f"{e.__class__.__name__}: {e}"
                self._draft_sources[stem] = path
                continue

            formal_source = self._find_formal_component_file(stem)
            if formal_collision and formal_source is None:
                self._draft_conflicts.setdefault(stem, DraftNameConflict(
                    f"draft component {stem!r} from {path} collides with a "
                    "formal component whose source file cannot be located; "
                    "rename this draft and update the target spec", path=path))
                continue
            existing_source = previous_source or (
                str(formal_source) if formal_collision and formal_source else None
            )
            existing_signature = self._draft_signatures.get(stem)
            if existing_signature is None and formal_collision and formal_source:
                existing_signature = component_structure_signature(
                    formal_source.read_text(encoding="utf-8"),
                    path=str(formal_source),
                )
            if existing_source is not None:
                if signature != existing_signature:
                    self._draft_conflicts.setdefault(stem, DraftNameConflict(
                        f"draft component {stem!r} from {path} has the same "
                        f"name as {existing_source} but a different structure; "
                        "rename this draft and update the target spec, or "
                        "reuse the existing component", path=path))
                    continue
                logger.debug(
                    "skipping duplicate draft %s from %s (same structure as %s)",
                    stem, path, existing_source)
                continue

            self._draft_lib[stem] = component
            self._draft_signatures[stem] = signature
            self._draft_sources[stem] = path
            for cop_name, cop_raw in (raw.get("custom_ops") or {}).items():
                self._draft_custom_ops.setdefault(stem, {})[cop_name] = (
                    CustomOpSpec(**cop_raw) if isinstance(cop_raw, dict) else cop_raw
                )

    def _find_formal_component_file(self, stem: str) -> Optional[Path]:
        for tier in ("library", "adapters"):
            for suffix in (".yaml", ".yml"):
                path = Path(self._components_dir) / tier / f"{stem}{suffix}"
                if path.is_file():
                    return path
        return None

    def _activate_drafts(self, listed: List[str], path: str) -> List[str]:
        """Promote spec-listed drafts into `_component_lib` for the
        duration of one spec's assembly. Returns the list of names that
        were promoted, so the caller can pop them out in a finally block.

        Errors deliberately raise here (not warn): if the spec asks for
        a draft that's missing or broken, assembly cannot proceed. That the
        list IS a list of strings is the document's business, not this
        method's — `SpecDocument.drafts` types it.
        """
        promoted: List[str] = []
        for dname in listed:
            if dname in self._draft_conflicts:
                raise self._draft_conflicts[dname]
            if dname in self._component_lib:
                # A draft was promoted to library/ but the spec still
                # lists it — harmless, library wins. Skip silently.
                continue
            if dname in self._draft_errors:
                raise ValueError(
                    f"{path}: draft component {dname!r} failed to load: "
                    f"{self._draft_errors[dname]}"
                )
            if dname not in self._draft_lib:
                raise ValueError(
                    f"{path}: lists draft {dname!r}, but no "
                    f"`components/drafts/{dname}.yaml` exists."
                )
            self._component_lib[dname] = self._draft_lib[dname]
            if dname in self._draft_custom_ops:
                self._component_custom_ops[dname] = self._draft_custom_ops[dname]
            promoted.append(dname)
        return promoted

    def _deactivate_drafts(self, promoted: List[str]) -> None:
        for dname in promoted:
            self._component_lib.pop(dname, None)
            self._component_custom_ops.pop(dname, None)

    def _find_model_file(self, name: str) -> Optional[str]:
        """The `<name>.yaml` (or `.yml`) file backing a base spec, searched in
        the shared models dir then any overlay dirs. Base is referenced by
        spec name, which is the file stem."""
        for mdir in [self._models_dir, *self._overlay_model_dirs]:
            for ext in (".yaml", ".yml"):
                candidate = os.path.join(mdir, name + ext)
                if os.path.isfile(candidate):
                    return candidate
        return None

    def _resolve_base_merge(self, raw: Dict[str, Any], path: str,
                            _chain: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fold a `base:` chain into one flat document. A key the child
        restates replaces the base's wholesale; a key it omits is inherited.
        No list/dict deep-merge — an additive variant (an extra MTP stack)
        restates the whole `model:` (contract B). The `base:` key itself is
        dropped from the result."""
        base = raw.get("base")
        if not base:
            return raw
        chain = _chain or []
        if base in chain:
            raise SpecLoadError(
                f"{path}: cyclic base chain {' -> '.join([*chain, base])}",
                path=path)
        base_path = self._find_model_file(base)
        if base_path is None:
            raise SpecLoadError(
                f"{path}: base spec {base!r} has no `{base}.yaml`", path=path)
        with open(base_path, "r", encoding="utf-8") as f:
            base_raw = load_yaml(f.read())
        if not isinstance(base_raw, dict):
            raise SpecLoadError(
                f"{path}: base spec {base!r} at {base_path} is not a mapping",
                path=path)
        base_flat = self._resolve_base_merge(base_raw, base_path, [*chain, base])
        return merge_base_documents(base_flat, raw)

    def _build_id_index(self, *, strict: bool = True) -> None:
        """Invert every loaded spec's `match.model_id` into `{id → spec name}`.
        Built from the specs, so there is no second table.

        `strict` (boot, shared tree): two specs claiming one id is a spec-set
        error and raises (contract D — caught at load / CI). Not strict
        (runtime overlay / custom load): a collision is CROSS-namespace, so the
        earlier (shared) claim wins with a warning (contract E — shared is
        authoritative, the user's override stays visible via origin=mine); it
        must never raise here, or one cached spec would DoS every match()."""
        index: Dict[str, str] = {}
        for name, document in self._documents.items():
            if document.match is None or not document.match.model_id:
                continue
            for mid in document.match.model_id:
                if mid in index and index[mid] != name:
                    if strict:
                        raise SpecLoadError(
                            f"model_id {mid!r} is claimed by two specs: "
                            f"{index[mid]!r} and {name!r}")
                    logger.warning(
                        "model_id %r claimed by both %r and %r; keeping %r "
                        "(shared wins)", mid, index[mid], name, index[mid])
                    continue
                index[mid] = name
        self._id_index = index

    def load_model_spec(self, path: str) -> Optional[ArchSpec]:
        """One spec YAML → one `ArchSpec`: parse the document, then lower it.

        There is exactly one route from a file to a spec. There used to be
        two — a "thin" assembler and a "legacy full spec" reader — chosen by
        sniffing whether `components:` held strings or sub-graphs, which meant
        a spec that was neither shape fell into the wrong interpreter and
        failed somewhere far from the line that was wrong. The document type
        decides the question once, at the top, by name.
        """
        with open(path, "r", encoding="utf-8") as f:
            source_text = f.read()
        raw = load_yaml(source_text)
        if not isinstance(raw, dict):
            return None

        # An identity file (`base: <family>`) inherits the family spec's
        # top-level keys and overrides the ones it restates. Resolve the
        # inheritance into a flat document BEFORE parsing, so `parse_document`
        # only ever sees a self-contained spec (contract B).
        raw = self._resolve_base_merge(raw, path)

        document = parse_document(raw, path)
        name = Path(path).stem
        with self._mutation_lock:
            self._install_dim_symbols(name, path, document, raw)
            # Activate any spec-scoped drafts BEFORE assembly reads
            # `_component_lib`. Pop them out in finally so a later spec
            # doesn't see the previous spec's drafts.
            promoted: list = []
            try:
                promoted = self._activate_drafts(document.drafts, path)
                spec = self.lower(document, name)
            except Exception:
                # A spec that does not load holds no layer.
                dim_registry.withdraw_local(self._dim_tokens.pop(name, None))
                raise
            finally:
                self._deactivate_drafts(promoted)
        spec.source_path = path
        spec.unresolved_note = _extract_unresolved_note(source_text)
        self._documents[spec.name] = document
        return spec

    def _restore_dim_layers(self, before: Dict[str, int]) -> None:
        """Put the `dim_symbols` layers back as `before` had them: withdraw
        what a rolled-back load installed, and reinstall a held spec's layer
        that load replaced."""
        for name, token in list(self._dim_tokens.items()):
            if before.get(name) != token:
                dim_registry.withdraw_local(self._dim_tokens.pop(name))
        for name in before:
            spec = self._specs.get(name)
            if name not in self._dim_tokens and spec is not None:
                token = dim_registry.declare_local(
                    spec.source_path or name, spec.dim_symbols)
                if token is not None:
                    self._dim_tokens[name] = token

    def _install_dim_symbols(self, name: str, path: str,
                             document: SpecDocument, raw: Dict[str, Any]) -> None:
        """Install the symbols `document` declares (`dim_symbols:`) for as
        long as this loader holds spec `name`, replacing what it held for
        that name before. A library model declares none: its symbols are
        rows of dims.yaml, which is where promotion put them."""
        dim_registry.withdraw_local(self._dim_tokens.pop(name, None))
        declared = document.dim_symbols
        if not declared:
            return
        if os.path.dirname(os.path.abspath(path)) == _LIBRARY_MODELS_DIR:
            raise ValueError(
                f"{path}: a library model declares no `dim_symbols`; its "
                f"symbols {sorted(declared)} are promoted into "
                f"backend/arch/dims.yaml with the model (Open PR does it). "
                f"Move them there and delete the block.")
        bound = _bound_dim_names(raw)
        unvalued = sorted(
            n for n, row in declared.items()
            if n not in bound and not (isinstance(row, dict) and (
                row.get("derived") or row.get("default"))))
        if unvalued:
            raise ValueError(
                f"dim_symbols: {unvalued} declared but never given a value; "
                f"bind each in `dims:` (top level or a block's / assignment's "
                f"`dims:`) or give it `default:`.")
        token = dim_registry.declare_local(path, declared)
        if token is not None:
            self._dim_tokens[name] = token

    def lower(self, document: SpecDocument, name: str) -> ArchSpec:
        """Lower an authored document to the assembly product the engine reads.

        This is the ONLY step between the two representations, and it goes one way:
        `SpecDocument` says which components fill which slots, `ArchSpec`
        carries the block graphs that came out of binding them, and nothing
        downstream re-reads the document to reach its own conclusion.

        Key design point: there is no Python-side "block pattern" factory.
        `pre_norm` / `post_norm` are ordinary parameterized components in
        `components/*.yaml` (declaring `params: {attn, ffn, norm}`), and
        every "block" instantiation is that component with those bindings,
        registered in `blocks` under the name the model layer referenced —
        `block` for a homogeneous backbone, a bindings-derived name for a
        per-assignment override. One construction, one registry.
        """
        comp_refs = document.components

        # Resolve component DAGs from library, track role → name mapping
        components = {}
        component_roles = {}
        for role, cname in comp_refs.items():
            if cname in self._component_lib:
                components[cname] = self._component_lib[cname]
                component_roles[cname] = role

        # Deep-copied: the desugaring below rewrites assignments in place, and
        # the document it came from outlives this call (the loader keeps it to
        # answer `match:`). An input a lowering edits is not an input.
        model_layers = [ml.model_copy(deep=True) for ml in document.model]

        # Inline block definitions — handwritten DAGs for topologies that
        # the standard pre_norm/post_norm pattern can't express (vision
        # towers, MTP heads, multi-branch decoder layers).
        inline_blocks: Dict[str, BlockGraph] = dict(document.blocks)

        synth = _BlockSynthesis(
            self._component_lib, inline_blocks, components, component_roles,
        )
        # A slot names a library component, an inline block, or a registered
        # op (`norm: rmsnorm`); anything else would be dropped from the map
        # above without a word.
        for role, cname in comp_refs.items():
            if (cname not in self._component_lib and cname not in inline_blocks
                    and cname not in OP_REGISTRY):
                raise synth.unknown(cname, f"components.{role}")

        self._desugar_assignment_slots(
            model_layers, comp_refs, inline_blocks, synth)

        self._synthesize_referenced_blocks(
            model_layers, inline_blocks, synth, comp_refs)

        blocks: Dict[str, BlockGraph] = {}

        # Merge synthesized + inline blocks. Derived names carry their
        # bindings, so they can't collide with an inline name; the one
        # author-chosen name, `block`, is only synthesized when no inline
        # block already answers to it.
        synthesized = synth.synthesized
        if synthesized:
            blocks.update(synthesized)
        if inline_blocks:
            blocks.update(inline_blocks)
        if not blocks:
            blocks = None

        self._collect_transitive_components(
            components, component_roles, inline_blocks,
            synth.synthesized, synth, model_layers)
        # Blocks a component's nested binding synthesized join the spec's
        # blocks like every other synthesized one.
        if synth.synthesized:
            blocks = dict(blocks or {})
            for bname, bg in synth.synthesized.items():
                if bname not in inline_blocks:
                    blocks[bname] = bg
            blocks.update(inline_blocks)

        merged_custom_ops = self._merge_custom_ops(
            components, document.custom_ops)
        if document.dim_symbols:
            # A parameter named after one of this spec's own divisible
            # symbols; dims.yaml's are checked wherever the op is built.
            for op_name, declaration in merged_custom_ops.items():
                reject_divisible_parameter_names(
                    op_name, declaration, local=document.dim_symbols)

        self._check_single_input_embedding(model_layers, name)
        self._check_roles_registered(component_roles, name)

        return ArchSpec(
            name=name,
            derived_from=document.derived_from,
            dims=document.dims,
            dim_symbols=document.dim_symbols,
            view=document.view,
            blocks=blocks,
            model=model_layers,
            model_inputs=document.model_inputs,
            checkpoint=document.checkpoint,
            components=components,
            component_roles=component_roles,
            custom_ops=merged_custom_ops,
            quant_context=document.quant_context,
            validation_exemptions=document.validation_exemptions,
        )


    def _desugar_assignment_slots(self, model_layers, comp_refs,
                                 inline_blocks, synth) -> None:
        """Desugar the `attention/ffn` slot shortcut on BlockAssignment into
        `template: <block_pattern>, params: {attn, ffn, norm}`."""
        # Desugar the `attention/ffn` slot shortcut on BlockAssignment into
        # `template: <block_pattern>, params: {attn, ffn, norm}`. Everything
        # the assignment leaves unsaid comes from the spec's `components:`
        # map, so an assignment names only what it CHANGES. The block pattern
        # itself is a YAML compound op (components/pre_norm.yaml).
        default_attn = comp_refs.get("attention", "")
        default_ffn = comp_refs.get("ffn", comp_refs.get("ffn_dense", ""))
        norm_type = comp_refs.get("norm", "rmsnorm")
        block_pattern = comp_refs.get("block", "pre_norm")
        for layer in model_layers:
            if not layer.assignments:
                continue
            for assign in layer.assignments:
                if assign.attention or assign.ffn:
                    attn = assign.attention or default_attn
                    ffn = assign.ffn or default_ffn
                    if not assign.template:
                        assign.template = block_pattern
                    if "attn" not in assign.params:
                        assign.params["attn"] = attn
                    if "ffn" not in assign.params:
                        assign.params["ffn"] = ffn
                    # Only inject `norm` when the target template actually
                    # declares it as a param slot. Hand-written blocks
                    # (e.g. dsv4_block) hardcode rmsnorm internally and
                    # don't accept a `norm` param — auto-injecting pollutes
                    # synth_params and the rendered block label ends up
                    # showing `norm=rmsnorm` for templates that never
                    # consumed it.
                    target_block = (
                        self._component_lib.get(assign.template)
                        or inline_blocks.get(assign.template)
                    )
                    accepts_norm = bool(
                        target_block and "norm" in (target_block.params or {})
                    )
                    if accepts_norm and "norm" not in assign.params:
                        assign.params["norm"] = norm_type
                    # Eager-load the referenced components (transitive deps
                    # walked by the post-assembly BFS below). `norm_type` is
                    # always loadable — even when the target block hardcodes
                    # rmsnorm, the engine resolves the inner `op: rmsnorm`
                    # node against `components`.
                    for cname in (attn, ffn, norm_type):
                        synth.register(cname)
                    assign.attention = None
                    assign.ffn = None

    def _synthesize_referenced_blocks(self, model_layers,
                                    inline_blocks, synth,
                                    comp_refs) -> None:
        """Synthesize block instances for every referenced template,
        including the bare `block` name (the components: slot map
        bound to the block pattern)."""
        def _referenced_templates(layers: List[ModelLayer]) -> set:
            names = set()
            for layer in layers:
                if layer.template:
                    names.add(layer.template)
                for assign in (layer.assignments or []):
                    if assign.template:
                        names.add(assign.template)
                if layer.layers:
                    names |= _referenced_templates(layer.layers)
            return names

        def _walk_layers(layers: List[ModelLayer]) -> None:
            for layer in layers:
                if layer.assignments:
                    for assign in layer.assignments:
                        new_t = synth.maybe_synth(assign.template, assign.params)
                        if new_t and new_t != assign.template:
                            assign.template = new_t
                # ModelLayer's own template (group sub-layer or simple repeat).
                new_t = synth.maybe_synth(layer.template, layer.params)
                if new_t and new_t != layer.template:
                    layer.template = new_t
                # Group containers nest sub-layers.
                if layer.layers:
                    _walk_layers(layer.layers)

        _walk_layers(model_layers)
        # `block` is the one template name with no definition to look up: it
        # IS the `components:` slot map, block_pattern bound to the attention
        # / ffn / norm slots. So a layer that references it gets that binding
        # synthesized the same way every other parameterized reference above
        # was — one construction, one registry, one name.
        #
        # The trigger is the reference. It used to be "this spec has no inline
        # blocks and no assignments", which asks a different question and
        # answers it wrong at both ends: a spec pairing a `template: block`
        # backbone with one inline vision block would have been denied its
        # backbone, and the three specs that route every layer through their
        # own named block still got a default one built — glm4's with `attn:
        # ''` and `ffn: ''`, longcat's naming an ffn that isn't in its
        # component set. Nothing read them, so nothing said so.
        if "block" in _referenced_templates(model_layers) and "block" not in inline_blocks:
            synth.synthesize(
                comp_refs.get("block", "pre_norm"),
                {"attn": comp_refs.get("attention", ""),
                 "ffn": comp_refs.get("ffn", comp_refs.get("ffn_dense", "")),
                 "norm": comp_refs.get("norm", "rmsnorm")},
                as_name="block",
            )

    def _collect_transitive_components(self, components,
                                       component_roles, inline_blocks,
                                       synthesized, synth, model_layers=()):
        """BFS the transitive component deps (node.op and component-typed
        params) into `components`, after every branch contributed. A
        model-level entry that calls a component by `op:` (a pure function
        at the model root, e.g. a logits soft-cap) seeds the walk too."""
        # Walk transitive component deps AFTER every branch (top-level role
        # refs, inline blocks, new-style assignments, legacy moe_block) has
        # contributed. Components can nest — e.g. `moe_ffn_gate_routing_bias`
        # contains `swiglu_ffn`, which contains `linear` primitives. Without
        # this BFS, sub-components fall through to the engine's primitive
        # branch on unknown op names and return 0 params (silent expert-bank
        # underflow).
        #
        # We walk both `node.op` (component-typed call) and `node.params`
        # (component-typed param bindings — strings naming a library
        # component). The latter lets a component file invoke a parameterized
        # template like `{ op: pre_norm, params: { attn: <component> } }` and
        # have the engine still find the bound attention at eval time.
        def _node_component_refs(node) -> List[str]:
            refs: List[str] = []
            op = node.op
            if op and op in self._component_lib:
                refs.append(op)
            refs.extend(pval for pval in (node.params or {}).values()
                        if isinstance(pval, str) and pval in self._component_lib)
            return refs
        pending = list(components.keys())

        def _model_layer_refs(layers):
            for layer in layers or ():
                if layer.op and layer.op in self._component_lib:
                    pending.append(layer.op)
                _model_layer_refs(layer.layers)
        _model_layer_refs(model_layers)
        # A component node may bind a parameterized template with arguments
        # of its own — `params: { attn: { template: X, params: { rope: mrope } } }`
        # — the same nested form a model layer accepts. Its graph is bound
        # into this spec (synthesized under the derived name) rather than
        # left holding a mapping no op lookup can answer.
        for table in (inline_blocks, synthesized):
            for bname, bg in list(table.items()):
                table[bname] = synth.bind_nested_params(bg)
        for bg in list(inline_blocks.values()) + list(synthesized.values()):
            for node in bg.graph.values():
                pending.extend(_node_component_refs(node))
        seen = set()
        walked_synth: set = set()
        while True:
            while pending:
                op = pending.pop()
                if op in seen:
                    continue
                seen.add(op)
                if op not in components:
                    components[op] = self._component_lib[op]
                    _role = synth.role_of(op)
                    if _role is not None:
                        component_roles.setdefault(op, _role)
                components[op] = synth.bind_nested_params(components[op])
                for sub_node in components[op].graph.values():
                    pending.extend(sub_op for sub_op in _node_component_refs(sub_node)
                                   if sub_op not in seen)
            # A binding above may have synthesized a block; walk what it calls.
            fresh = [bname for bname in synthesized if bname not in walked_synth]
            if not fresh:
                break
            for bname in fresh:
                walked_synth.add(bname)
                for node in synthesized[bname].graph.values():
                    pending.extend(_node_component_refs(node))

    def _merge_custom_ops(self, components, own: Dict[str, CustomOpSpec]
                         ) -> Dict[str, CustomOpSpec]:
        """Component-scoped custom_ops merge into the spec's table;
        spec-level entries win name collisions."""
        # Merge component-scoped custom_ops into the spec's custom_ops so
        # the engine and the tooltip-formula renderer both find them.
        # Spec-level entries win on name collision (give the spec author
        # an override path).
        merged_custom_ops: Dict[str, CustomOpSpec] = {}
        for cname in components:
            for cop_name, cop_spec in (self._component_custom_ops.get(cname) or {}).items():
                merged_custom_ops.setdefault(cop_name, cop_spec)
        merged_custom_ops.update(own)
        return merged_custom_ops

    @staticmethod
    def _check_single_input_embedding(model_layers, name) -> None:
        """Exactly one layer may be THE input embedding."""
        embeds = [ml.name for ml in model_layers
                  if ml.boundary_role == "input_embedding"]
        if len(embeds) > 1:
            raise ValueError(
                f"{name}: {len(embeds)} layers "
                f"declare `boundary_role: input_embedding` ({embeds}). The "
                f"model's tokens enter through exactly one table; the others "
                f"— speech banks, vision position embeddings, an MTP head's "
                f"own copy — are embeddings but not THE embedding.")

    @staticmethod
    def _check_roles_registered(component_roles, name) -> None:
        """Every used role must be classified in backend/arch/roles.yaml."""
        unclassified = sorted({r for r in component_roles.values()
                               if r and not roles.is_registered(r)})
        if unclassified:
            raise ValueError(
                f"{name}: role(s) {unclassified} "
                f"that backend/arch/roles.yaml does not classify. A "
                f"`role:` on a component, or a slot name in a model's "
                f"`components:` map, says which component fills a job; what "
                f"that job IS — attention / ffn / norm / wrapper — is a fact "
                f"about the role and belongs in the registry. Unclassified, "
                f"the role reaches every consumer that asks 'is this "
                f"attention?' and is answered no: its FLOPs vanish from the "
                f"attention split, and its weights miss the attention "
                f"sharding rules.")


    def get(self, name: str) -> Optional[ArchSpec]:
        """Get a spec by name."""
        return self._specs.get(name)

    def all_specs(self) -> Dict[str, ArchSpec]:
        """Return all loaded specs."""
        return dict(self._specs)

    def listed_drafts(self, spec_name: str) -> Tuple[str, ...]:
        """The drafts spec `spec_name` opts into (`drafts:`), in the order
        it lists them; empty for a spec this loader did not assemble."""
        document = self._documents.get(spec_name)
        return tuple(document.drafts) if document is not None else ()

    def component_names(self) -> FrozenSet[str]:
        """Every component name a spec can reference by name.

        `library/` + `adapters/`, which share one flat namespace. NOT drafts:
        those are invisible until a spec opts in via `drafts: [...]`, and only
        for its own load — so "a name any spec could use" is exactly this set.

        Public because the question has a real caller outside the loader: a
        node id carries its component name as a `…__variant` suffix, and the
        calibration identity reads that suffix to tell absorbed MLA from dense
        GQA. Asking which names are legitimate is asking the loader, and the
        alternative — a consumer re-globbing the component directories — is a
        second copy of where components live and how they are named."""
        return frozenset(self._component_lib)

    def component(self, name: str) -> BlockGraph:
        """The library component `name` as loaded — the same object a spec
        binds when it names it, decode variants included (they are
        components in their own right; `rebind_decode_variants` swaps them
        in under the prefill slot name at analysis time). Refuses a name
        the library does not have: asking for one is asking the loader,
        not re-globbing the directories."""
        try:
            return self._component_lib[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a library component "
                f"(known: {len(self._component_lib)})") from None

    def draft_components(self) -> Dict[str, BlockGraph]:
        """Every draft this loader scanned, by name — the merged
        `components/drafts/` tree and any overlay draft dir (one run's own
        working copies).

        Not the same question as `component_names()`: a draft is invisible to
        a spec until that spec lists it under `drafts:`, while "what is there
        to reuse instead of writing a new one" covers every draft on disk,
        opted into or not. A draft that failed to parse is absent here, as it
        is absent from a spec that lists it."""
        return dict(self._draft_lib)

    def component_file(self, name: str) -> Optional[str]:
        """The yaml file a component was read from — `library/`, `adapters/`
        or the drafts dir (overlay included) that holds it — or None for a
        name no tier declares. For a reader that wants to open the component
        it was pointed at; where components live stays the loader's fact."""
        formal = self._find_formal_component_file(name)
        if formal is not None:
            return str(formal)
        return self._draft_sources.get(name)

    def synthesize_block(
        self, spec: ArchSpec, template: str, params: Dict[str, Any],
    ) -> Tuple[ArchSpec, str]:
        """Bind `template` with `params` on `spec` the way lowering binds an
        assignment's `template:` + `params:` — same synthesizer, so the same
        name and graph an author writing that binding would have got.

        Returns a new spec and the block's name; `spec` is left as it was.
        `_BlockSynthesis` writes into the maps it is handed, so it is handed
        copies. A binding whose name the spec already holds is that block —
        it is reused, not rebuilt, as lowering reuses a repeated map entry.
        Components bound by the new block are registered (a role the spec
        already gave one stands); their own transitive dependencies are the
        caller's to have injected, as they are for any edit of a lowered spec.
        """
        existing = dict(spec.blocks or {})
        components = dict(spec.components or {})
        component_roles = dict(spec.component_roles or {})
        synth = _BlockSynthesis(
            self._component_lib, existing, components, component_roles)
        synth.synthesized.update(existing)
        name = synth.maybe_synth(template, dict(params))
        created = {k: v for k, v in synth.synthesized.items()
                   if k not in existing}
        for block in created.values():
            for value in block.synth_params.values():
                synth.register(value)
        return spec.model_copy(update={
            "blocks": {**existing, **created},
            "components": components,
            "component_roles": component_roles,
        }), name

    def inject_component(self, spec: ArchSpec, name: str) -> ArchSpec:
        """`spec` with library component `name` bound in the way lowering
        binds the components a spec names: everything it calls,
        transitively (`_collect_transitive_components`), nested template
        bindings synthesized, and the component-scoped custom ops merged
        (`_merge_custom_ops`). What the spec already holds stands — a
        component its lowering resolved keeps that resolution, a role it
        gave one stays, and its own custom ops win a name collision.

        Returns a new spec; `spec` is left as it was."""
        if name not in self._component_lib:
            raise KeyError(
                f"{name!r} is not a library component "
                f"(known: {len(self._component_lib)})")
        components = dict(spec.components or {})
        component_roles = dict(spec.component_roles or {})
        blocks = dict(spec.blocks or {})
        synth = _BlockSynthesis(
            self._component_lib, blocks, components, component_roles)
        synth.register(name)
        self._collect_transitive_components(
            components, component_roles, {}, synth.synthesized, synth)
        return spec.model_copy(update={
            "components": components,
            "component_roles": component_roles,
            "blocks": {**synth.synthesized, **blocks},
            "custom_ops": self._merge_custom_ops(
                components, spec.custom_ops or {}),
        })

    def resolve(
        self, spec: ArchSpec, config: Dict[str, Any], *,
        model_id: Optional[str] = None,
        header: Optional["SafetensorsHeader"] = None,
    ) -> ResolvedSpec:
        """Resolve a spec's dimensions against a model config — and, for a
        dim bound `checkpoint.<tensor>[<axis>]`, against that model's tensor
        header: `header` when the caller has it, else read for `model_id`.
        With neither, such a dim refuses to resolve."""
        checkpoint = CheckpointHeader.of(model_id, header)
        dim_values = resolve_dimensions(
            spec.dims, config, checkpoint, local=spec.dim_symbols)
        resolved = ResolvedSpec(spec, dim_values, config,
                                resolve_view(spec.view, config),
                                declared_by=self, checkpoint=checkpoint)
        resolved.check_appended_streams()
        return resolved

    def resolve_by_name(
        self, name: str, config: Dict[str, Any], *,
        model_id: Optional[str] = None,
        header: Optional["SafetensorsHeader"] = None,
    ) -> Optional[ResolvedSpec]:
        """Convenience: get + resolve in one call."""
        self._refuse_if_load_failed(name=name)
        spec = self.get(name)
        if spec is None:
            return None
        return self.resolve(spec, config, model_id=model_id, header=header)

    def _refuse_if_load_failed(self, *, name: str | None = None,
                               types: Sequence[str] = (),
                               model_id: str | None = None) -> None:
        """Raise with the original load error when a spec that WOULD have
        answered this request is sitting in the failure list.

        A failed spec is a broken answer, not a miss, and both silent
        behaviours are wrong: returning None triggers spec-gen for a model
        that has a spec, and matching some other spec serves the wrong
        architecture. The one piece of information worth having — why the
        file did not parse — is already in hand at boot and gets buried in a
        warning nobody reads.

        Two ways to ask for a spec, so two ways to be owed the refusal: by
        `model_type` (what `match()` asks) and by `name` (what an explicit
        `--spec-name` asks, and what the validator asks after spec-gen
        writes a file). A spec's name is its file stem, so a requested name
        that matches a failed file's stem is that file.
        """
        for path, failed_types, failed_ids, err in self._model_load_failures:
            if name is not None and Path(path).stem == name:
                raise SpecLoadError(
                    f"spec {name!r} exists at {path} but failed to load: {err}",
                    path=str(path))
            if model_id is not None and model_id in failed_ids:
                raise SpecLoadError(
                    f"spec for model_id {model_id!r} exists at {path} "
                    f"but failed to load: {err}", path=str(path))
            if any(t in failed_types for t in types):
                raise SpecLoadError(
                    f"spec for model_type {list(types)!r} exists at "
                    f"{path} but failed to load: {err}", path=str(path))

    def _load_custom_spec_if_any(self) -> None:
        """Dynamic loader: custom models are registered at runtime under
        ~/.huggingarch/custom_models/<name>/spec.yaml. Pick them up on
        each `match()` call so a freshly-registered model works without
        a server restart.

        Also records the `<dir name> → spec name` mapping in
        `_custom_dir_to_spec` so `match(config, model_id="custom:<dir>")`
        can pin to the exact registered spec — multiple custom specs
        often share `model_type` (e.g. several agent-generated KimiV
        variants all declare `moonshot_kimiv_auto`), so the `custom:<dir>`
        pin names the one wanted rather than leaving it ambiguous.
        """
        if not CACHE_DIR.is_dir():
            return
        for sub in CACHE_DIR.iterdir():
            spec_path = sub / "spec.yaml"
            if not spec_path.is_file():
                continue
            try:
                spec = self.load_model_spec(str(spec_path))
                if spec:
                    self._specs[spec.name] = spec
                    self._custom_dir_to_spec[sub.name] = spec.name
            except (OSError, yaml.YAMLError, ValueError, SpecLoadError) as e:
                logger.warning(f"Failed to load custom spec {spec_path}: {e}")
        # Runtime add: a custom id colliding with a shared one keeps shared
        # (contract E) and must never raise — this runs on every match().
        self._build_id_index(strict=False)

    def _spec_by_model_id(self, model_id: str) -> Optional[ArchSpec]:
        """The spec a `custom:<dir>` pin or a registered `match.model_id`
        names, or None. A registered-but-broken id is refused loudly."""
        if model_id.startswith("custom:"):
            spec_name = self._custom_dir_to_spec.get(model_id[len("custom:"):])
            return self._specs.get(spec_name) if spec_name else None
        if model_id in self._id_index:
            self._refuse_if_load_failed(model_id=model_id)
            return self._specs.get(self._id_index[model_id])
        return None

    def spec_for_model_id(self, model_id: Optional[str]) -> Optional[ArchSpec]:
        """The spec this loader serves for `model_id` — the identity half of
        `match()`, which needs no config. Exact lookup, no model_type scoring.

        A registered model_id answers with its own identity spec. An
        unregistered id asks the registered spec-miss hook
        (`set_spec_miss_hook`) for an overlay and looks again once; still
        missing is None. `model_id=None` is always None — matching needs an
        identity. A spec declaring this id that failed to load raises
        `SpecLoadError` (a broken answer, not a miss — contract D)."""
        # Custom specs are registered at runtime; pick them up per call.
        self._load_custom_spec_if_any()
        if not model_id:
            return None
        spec = self._spec_by_model_id(model_id)
        if spec is not None:
            return spec
        # A broken spec never reaches the index, so this check lives here,
        # not in the id-index lookup above.
        self._refuse_if_load_failed(model_id=model_id)
        if _spec_miss_hook is not None:
            overlay = _spec_miss_hook(model_id)
            if overlay and self.attach_overlay(overlay):
                return self._spec_by_model_id(model_id)
        return None

    def overlay_for(self, model_id: Optional[str]) -> Optional[str]:
        """The overlay root this loader serves `model_id`'s spec from, or
        None for a spec of the shared tree (or no spec).

        A stored record's overlay carries the snapshot its spec was judged
        against; a consumer reads the model's config and checkpoint record
        from there (`hf.metadata.fetch_config` / `fetch_metadata` with this
        root), so the page and the verdict compare one spec against one
        record. Asking may attach an overlay through the spec-miss hook,
        as `match` would."""
        spec = self.spec_for_model_id(model_id)
        src = spec.source_path if spec is not None else None
        if not src:
            return None
        for models_dir in self._overlay_model_dirs:
            if os.path.dirname(os.path.abspath(src)) == os.path.abspath(models_dir):
                return os.path.dirname(os.path.abspath(models_dir))
        return None

    def match(
        self,
        config: Dict[str, Any],
        *,
        model_id: Optional[str],
        header: Optional["SafetensorsHeader"] = None,
    ) -> Optional[ResolvedSpec]:
        """Resolve a checkpoint to its spec by model_id: `spec_for_model_id`
        picks the spec, then it is resolved against `config`. None on a miss
        (the caller triggers spec-gen). `header` never selects a
        spec; it is the checkpoint header a `checkpoint.`-bound dim reads,
        passed by a caller that already fetched it (otherwise the header is
        read for `model_id` when such a dim is resolved).
        """
        spec = self.spec_for_model_id(model_id)
        if spec is None:
            return None
        return self.resolve(spec, config, model_id=model_id, header=header)

    def attach_overlay(self, overlay_root: str) -> bool:
        """Load `<root>/components/drafts` and `<root>/models` on top of what
        is already loaded. True iff every model yaml in the overlay loaded;
        otherwise nothing from it is registered and the loader is unchanged,
        so a spec that cannot load is a miss rather than a claim on its
        model_type that would refuse the model on every later request. A
        spec listing a same-name/different-structure draft is not a miss: it
        raises that `DraftNameConflict` after the same full rollback. An
        unlisted one belongs to no spec and is left alone."""
        with self._mutation_lock:
            return self._attach_overlay(str(overlay_root))

    def _attach_overlay(self, root: str) -> bool:
        drafts_dir = os.path.join(root, "components", "drafts")
        models_dir = os.path.join(root, "models")
        # Everything a failed load may have touched, copied deep enough to
        # put back: `load_model_spec` records the document before it knows
        # the spec loads, and `_scan_draft_dir` writes into the per-draft
        # custom-op dicts in place.
        saved = (
            dict(self._draft_lib),
            {name: dict(ops) for name, ops in self._draft_custom_ops.items()},
            dict(self._draft_errors),
            dict(self._draft_conflicts),
            dict(self._draft_signatures),
            dict(self._draft_sources),
            dict(self._documents),
            dict(self._component_lib),
            dict(self._component_custom_ops),
        )
        dim_tokens = dict(self._dim_tokens)
        draft_conflict: DraftNameConflict | None = None
        self._scan_draft_dir(drafts_dir)
        loaded: Dict[str, ArchSpec] = {}
        if os.path.isdir(models_dir):
            for fname in sorted(os.listdir(models_dir)):
                if not fname.endswith((".yaml", ".yml")):
                    continue
                path = os.path.join(models_dir, fname)
                try:
                    spec = self.load_model_spec(path)
                except DraftNameConflict as e:
                    draft_conflict = e
                    break
                except Exception as e:  # an overlay is untrusted input
                    # A cached row is whatever an agent wrote months ago; a
                    # malformed one (a graph written as a list, a bare string
                    # where a node goes) is a miss, the same as a stale one.
                    logger.warning(
                        f"overlay spec {path} did not load ({type(e).__name__}: {e}); "
                        f"treating the overlay as a miss")
                    spec = None
                if spec is None:
                    loaded = {}
                    break
                loaded[spec.name] = spec
        if draft_conflict is not None or not loaded:
            (self._draft_lib, self._draft_custom_ops, self._draft_errors,
             self._draft_conflicts, self._draft_signatures, self._draft_sources,
             self._documents, self._component_lib,
             self._component_custom_ops) = saved
            self._restore_dim_layers(dim_tokens)
            if draft_conflict is not None:
                raise draft_conflict
            return False
        self._specs.update(loaded)
        self._overlay_model_dirs.append(models_dir)
        self._overlay_draft_dirs.append(drafts_dir)
        # Runtime add: overlay ids that collide with shared keep shared
        # (contract E); never raises, so the overlay stays consistently mounted.
        self._build_id_index(strict=False)
        return True
