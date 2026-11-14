"""Document parsing — raw authored YAML (dict) → SpecDocument / BlockGraph /
ModelLayer objects.

Extracted from loader.py, which mixed five concerns; this is the parser alone.
Loading / matching / synthesis stay in loader.py, which imports
parse_document / _parse_component_document / _extract_unresolved_note from
here. Dimension / view resolution lives in dim_resolution.py. One-directional:
this layer depends on document / schema / ops / cache_families, none of which
import it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

import backend.arch.cache_families as cache_families
import backend.arch.roles as roles
from backend.arch.document import PerLayerBlock, SpecDocument
from backend.arch.ops import OP_REGISTRY
from backend.arch.yaml_load import load_yaml
from backend.arch.schema import (
    BlockAssignment,
    BlockGraph,
    CacheSpec,
    ComponentClaims,
    GraphNode,
    ModelLayer,
    ParamSpec,
    reject_retired_graph_node_naming,
)


_GRAPH_NODE_SOURCE_KEYS = frozenset(
    name for name, field in GraphNode.model_fields.items()
    if not (field.json_schema_extra or {}).get("loader_derived")
)

def _parse_graph_node(name: str, raw: Any) -> GraphNode:
    """Parse a graph node from YAML, handling compact dict format."""
    if isinstance(raw, dict):
        # Normalize 'in' → 'inputs' (YAML uses 'in' for readability)
        d = dict(raw)
        # Retired checkpoint-naming keys first, so `rename:` / `hf:` get
        # the replacement field rather than a generic unknown-key list.
        reject_retired_graph_node_naming(d)
        unknown = sorted(set(d) - _GRAPH_NODE_SOURCE_KEYS - {"in"})
        if unknown:
            raise ValueError(
                f"Graph node {name!r} declares unknown or loader-derived "
                f"field(s) {unknown}. Allowed fields: "
                f"{sorted(_GRAPH_NODE_SOURCE_KEYS | {'in'})}")
        if "in" in d and "inputs" in d:
            raise ValueError(
                f"Graph node {name!r} declares both `in` and `inputs`; they "
                "are the same field, so keep one spelling")
        if "in" in d:
            d["inputs"] = d.pop("in")
        if "inputs" not in d:
            d["inputs"] = []
        return GraphNode(**d)
    raise TypeError(f"Invalid graph node {name!r}: expected dict, got {type(raw)}")


_RETIRED_CARRY_KEYS = ("carries", "carry_outputs", "carry_src", "carry_init")

#: The four fields `cache:` replaced. Retired rather than aliased: they were
#: four independent edits to one fact, and the dtype map keyed slots by names
#: that lived in a different list.
_RETIRED_KV_KEYS = ("kv_cached", "kv_cache_dtype", "kv_state_elements",
                    "kv_label")

#: Authoring path that replaced each retired KV key. The loader refusal
#: and the validate findings both name these, so a writer is never told
#: to write a key the loader will refuse.
RETIRED_KV_REPLACEMENTS = {
    "kv_label": "cache.family",
    "kv_cached": "cache.entries",
    "kv_cache_dtype": "cache.entries",
    "kv_state_elements": "cache.entries",
}

#: A component's name is its file stem — that is what `drafts:`, `template:`
#: and the claims sidecar all key on. A `name:` line could only ever repeat
#: the stem or contradict it, and a contradiction would split one component
#: into two identities (claims looked up under one, references resolved
#: under the other).
_RETIRED_IDENTITY_KEYS = ("name",)

#: The input ports a block has when it does not say otherwise. Read off the
#: schema so the default has one home and the gate below cannot drift from it.
_DEFAULT_INPUTS = BlockGraph.model_fields["inputs"].default

#: Every key this function reads. `_parse_block_graph` reads keys one at a
#: time rather than `**raw`, so an unrecognised key is not a pydantic error —
#: it is silence, and the fact the author wrote simply does not exist. The set
#: has to be the READ set exactly: list a key nothing reads and the gate
#: promises a channel that isn't there; omit a key that IS read and a valid
#: component stops loading. `custom_ops` is here because it travels in the
#: same document, consumed by the caller (SpecLoader) rather than by this
#: function.
_BLOCK_GRAPH_SOURCE_KEYS = frozenset({
    "attn_variant", "cache", "custom_ops", "sparse_dcp",
    "decode_variant",
    "description", "formula", "graph", "inputs", "outputs", "params",
    "reads", "role", "secondary_outputs", "stream_out", "variant_phases",
})


def _parse_block_graph(raw: Dict[str, Any],
                       identity: Optional[str] = None) -> BlockGraph:
    """Parse a BlockGraph from raw YAML dict."""
    # The loader reads keys explicitly rather than `**raw`, so a retired key
    # would otherwise be silently dropped and the stream it declares would
    # quietly stop existing. Same treatment as the removed `assignments:`.
    # Named first so a retired key gets the migration note, not `unknown`.
    retired = [k for k in _RETIRED_CARRY_KEYS if k in raw]
    if retired:
        raise ValueError(
            f"Block declares the removed block-local carry syntax "
            f"({', '.join(retired)}). Cross-iteration state is authored as a "
            f"STREAM: the stack declares the stream under `streams: "
            f"{{name: {{shape, src}}}}`, the producing component declares "
            f"`stream_out: {{name: <node>}}`, and the consuming component "
            f"declares `reads: [name]`. See BlockGraph.reads / .stream_out."
        )
    retired_kv = [k for k in _RETIRED_KV_KEYS if k in raw]
    if retired_kv:
        raise ValueError(
            f"Block declares the removed KV field(s) "
            f"({', '.join(retired_kv)}). What a component keeps between "
            f"decode steps is one declaration now:\n"
            f"  cache:\n"
            f"    family: GQA                       # was kv_label\n"
            f"    entries:\n"
            f"      - {{node: k_split_heads, kind: kv}}   # was kv_cached\n"
            f"      - {{node: indexer.k_norm, kind: index_key, dtype: fp8}}"
            f"   # was kv_cache_dtype\n"
            f"      - {{state: recurrent_state, kind: recurrent_state, "
            f"carrier: <node>, shape: [<axis>, ...]}}  # was kv_state_elements\n"
            f"See schema.CacheSpec."
        )
    retired_identity = [k for k in _RETIRED_IDENTITY_KEYS if k in raw]
    if retired_identity:
        raise ValueError(
            f"Block {identity or '<inline>'} declares `name:`. A block is "
            f"named by the thing that holds it — a component by its file "
            f"stem, an inline block by its key under `blocks:` — and that "
            f"is what `template:`, `drafts:` and the claims sidecar key on. "
            f"Delete the line."
        )
    if "axioms" in raw:
        raise ValueError(
            f"Block {identity or '<inline>'} declares `axioms:`. A "
            f"component's claims live in `claims/components/<identity>.yaml` "
            f"beside the component library, and validate judges them against "
            f"the cache each call of the component grows. A sliding window "
            f"is stated in the graph (`sliding_window_kv` over the cached "
            f"stream) and needs no claim. Delete the `axioms:` block."
        )
    unknown = sorted(set(raw) - _BLOCK_GRAPH_SOURCE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown component/block field(s): {unknown}. "
            f"Allowed fields: {sorted(_BLOCK_GRAPH_SOURCE_KEYS)}"
        )
    inputs = raw.get("inputs", list(_DEFAULT_INPUTS))
    outputs = raw.get("outputs", [])
    raw_graph = raw.get("graph", {})
    if not isinstance(raw_graph, dict):
        # A data error in the parser's own channel: every caller handles
        # ValueError (a draft that cannot load, a candidate that cannot be
        # compared), and none should have to know that `.items()` on a list
        # would surface as an AttributeError instead.
        raise ValueError(
            f"Component/block graph must be a mapping of node name → node, "
            f"got {type(raw_graph).__name__}")

    graph = {}
    for node_name, node_data in raw_graph.items():
        graph[node_name] = _parse_graph_node(node_name, node_data)


    # Param slot declarations: parameterized compound op accepts callers
    # binding component names. Shorthand `params: {attn: component}` →
    # ParamSpec(kind="component"); explicit `params: {attn: {kind, default}}`
    # also accepted for forward-compat with dim-typed params.
    params = {}
    raw_params = raw.get("params") or {}
    for pname, pval in raw_params.items():
        if isinstance(pval, dict):
            params[pname] = ParamSpec(**pval)
        elif pval is None:
            params[pname] = ParamSpec(kind="component")
        elif isinstance(pval, str):
            # Shorthand: "rmsnorm" means kind=component, default=rmsnorm.
            params[pname] = ParamSpec(kind="component", default=pval)
        else:
            raise ValueError(f"Invalid param spec for {pname!r}: {pval!r}")

    # Inner streams — a component's interface to the named cross-iteration
    # state threaded across the stack: `reads` it consumes, `stream_out` it
    # produces (stream → producing node).
    reads: List[str] = []
    raw_reads = raw.get("reads") or []
    if isinstance(raw_reads, list):
        reads = [str(n) for n in raw_reads]
    stream_out: Dict[str, str] = {}
    raw_so = raw.get("stream_out") or {}
    if isinstance(raw_so, dict):
        stream_out = {str(k): str(v) for k, v in raw_so.items()}

    # What the component keeps between decode steps. Read explicitly, like
    # everything else here, because the loader doesn't `**raw`. Cache
    # cardinality/windowing is derived from op laws; there is deliberately
    # no component-level `kv_window`.
    cache_raw = raw.get("cache")
    if cache_raw is not None and not isinstance(cache_raw, dict):
        raise ValueError(
            f"cache must be a mapping with `family:` / `entries:`, got "
            f"{type(cache_raw).__name__}"
        )
    cache = CacheSpec(**cache_raw) if cache_raw is not None else None
    if cache is not None and cache.family and not cache_families.is_registered(cache.family):
        raise ValueError(
            f"Component {identity or '<inline>'!r}: cache.family {cache.family!r} "
            f"is not registered in backend/arch/cache_families.yaml "
            f"(known: {sorted(cache_families.all_families())}). A new family is "
            f"a row there with a `kind`, not a new spelling here.")

    role = raw.get("role")
    name_for_err = identity or "<unnamed>"

    # Schema checks for component-authoring fields, asked only of a
    # component that holds a weight leaf of its own. One made of other
    # components — an alias adapter (`BlockGraph.alias_target`), a `pre_norm`
    # binding — inherits cache / kv_label / shard_symbols from what it wraps
    # at lookup time, and `_type_contracts` asks the wrapped target.
    has_direct_weight_leaf = any(
        n.op in OP_REGISTRY and OP_REGISTRY[n.op].has_weight()
        for n in graph.values()
    )
    if (roles.kind_of(role) == "attention" and has_direct_weight_leaf
            and (cache is None or not cache.family)):
        raise ValueError(
            f"Component {name_for_err!r}: role={role!r} (kind attention) requires "
            f"`cache: {{family: ...}}` (short tag like `MLA` / `GQA` / `SWA` "
            f"/ `GLA-Linear` / `GatedDeltaNet` / `ViT`). KV-cache topology, "
            f"weight_breakdown's attention class label, and attention-type "
            f"detection all read it."
        )
    attn_variant = raw.get("attn_variant")
    # secondary_outputs: wrapper-level multi-output port declaration.
    # formula: one-line math summary surfaced on the wrapper UI node.
    secondary_outputs_raw = raw.get("secondary_outputs") or {}
    if not isinstance(secondary_outputs_raw, dict):
        raise ValueError(
            f"secondary_outputs must be a dict mapping port name to "
            f"internal-child ref, got {type(secondary_outputs_raw).__name__}"
        )
    return BlockGraph(inputs=inputs, graph=graph, outputs=outputs,
                      secondary_outputs={str(k): str(v) for k, v in secondary_outputs_raw.items()},
                      params=params,
                      role=role,
                      decode_variant=raw.get("decode_variant"),
                      variant_phases=tuple(raw.get("variant_phases")
                                           or ("decode",)),
                      description=raw.get("description"),
                      formula=raw.get("formula") or "",
                      reads=reads,
                      stream_out=stream_out,
                      cache=cache,
                      attn_variant=attn_variant,
                      sparse_dcp=raw.get("sparse_dcp"))


#: The library's tree: claims about library components live in its
#: `claims/components/`, wherever the spec reaching them was authored.
_LIBRARY_ROOT = Path(__file__).resolve().parent


def _attach_claims(graph: BlockGraph, name: str, root: Path) -> BlockGraph:
    """Attach the claims about the component `name`.

    Validation claims are a sidecar input: they remain outside the authored
    component program, then attach to the BlockGraph because the validator
    reads them there. They are looked up by identity in
    `<tree>/claims/components/<name>.yaml` — first the tree the component was
    authored in (`root`: an overlay's, or the library's), then the library's,
    which states the claims about every component it ships. So an overlay
    model that inherits a library model's inline block through `base:` keeps
    that block's claims, and a component inline in a model is claimed the
    same way as one in its own file.
    """
    for tree in dict.fromkeys((root, _LIBRARY_ROOT)):
        claims_path = tree / "claims" / "components" / f"{name}.yaml"
        if not claims_path.is_file():
            continue
        with claims_path.open("r", encoding="utf-8") as f:
            claims = ComponentClaims(**(load_yaml(f) or {}))
        if claims.subject != name:
            raise ValueError(
                f"{claims_path}: claims subject {claims.subject!r} does not "
                f"match component identity {name!r}"
            )
        graph.axioms = claims.axioms
        return graph
    return graph


def _parse_component_document(raw: Dict[str, Any], path: str) -> Tuple[str, BlockGraph]:
    """Parse one standalone component source into a BlockGraph."""
    name = Path(path).stem
    graph = _parse_block_graph(raw, identity=name)
    return name, _attach_claims(graph, name, Path(path).resolve().parents[2])


class LayerCall:
    """A call site, `{ name, call, inputs, dims? }`: the earlier layer
    `call` names — a `group:` (a model's tower) or a `template:` call
    beside it — run again on `inputs`, with that layer's weights. The HF
    forward does exactly this when it hands a second input to the same
    module (Qwen3-VL's `get_video_features` -> `get_image_features`, Step's
    crops through the vision model), so the spec writes the call, not a
    copy of the module.

    The loader writes the call out as the called layer: the same layers in
    the same order, renamed under the call's name (the called layers'
    shared module path with its last segment the call's name:
    `model.visual.merger` -> `model.video.merger`), each stream the called
    layer exports (`from:`) suffixed `_<call>`, the called layer's outside
    inputs replaced in order by the call's `inputs`, and each dim the call
    rebinds (`dims: {<symbol>: <symbol>}`, a fact of the other input, such
    as the soft tokens a video frame is padded to) rebound. It declares no
    checkpoint: its tensors are the called layer's (`ModelLayer.call`).

    What it may call is an encoder's tower as the corpus runs one: a
    top-level `group:` with an `aux_kind`, or a `template:` call beside it
    in its group, that itself calls nothing and reads no stream or prompt
    position by name (`reads_from`, `prompt_slots`). A rebinding reaches
    the activations only: every layer that holds a weight keeps the called
    one's op, dims, shape and placement, and every `repeat:` its count, so
    each weight read is the weight stored."""

    #: The fields that state a layer's weight: what it is and where it lies.
    WEIGHT_FIELDS = ("op", "dims", "dim", "shape", "weight_axes",
                     "stored_axes", "stored_split", "expert_weight")

    KEYS = frozenset({"name", "call", "inputs", "in", "dims"})

    def __init__(self, raw: Dict[str, Any], target: Dict[str, Any]):
        extra = sorted(set(raw) - self.KEYS)
        if extra:
            raise ValueError(
                f"Layer {raw.get('name')!r} calls {raw['call']!r} and "
                f"declares {extra}; a call states its `name`, `inputs` and "
                f"the dims it rebinds, and runs the called layer as it is")
        self.name = raw["name"]
        self.target = target
        inputs = raw.get("inputs", raw.get("in")) or []
        self.inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        self.rebind = {str(k): str(v) for k, v in (raw.get("dims") or {}).items()}

    def _check_target(self, in_group: bool) -> None:
        """The called layer is a tower this call can run as it is."""
        target = self.target
        shape = ("a `template:` call beside it in its group" if in_group
                 else "a top-level encoder `group:` (with an `aux_kind`)")
        fits = (target.get("template") is not None if in_group
                else bool(target.get("group")) and target.get("aux_kind"))
        if not fits:
            raise ValueError(
                f"Layer {self.name!r} calls {target['name']!r}; a call runs "
                f"{shape} again")
        for layer in self._layers(target):
            named = sorted(k for k in ("call", "reads_from", "prompt_slots")
                           if layer.get(k) is not None)
            if named:
                raise ValueError(
                    f"Layer {self.name!r} calls {target['name']!r}, whose "
                    f"layer {layer['name']!r} declares {named}; a call runs "
                    f"a tower that calls nothing and reads no stream or "
                    f"prompt position by name")

    def _check_same_weights(self, owner: Dict[str, Any],
                            ours: Dict[str, Any]) -> None:
        """Each layer of the call next to the called one it was cloned from:
        a rebinding that reached a weight or a count is refused."""
        if str(owner.get("repeat")) != str(ours.get("repeat")):
            raise ValueError(
                f"Layer {self.name!r} rebinds {sorted(self.rebind)}, which "
                f"changes the count of {owner['name']!r} "
                f"(`repeat: {owner.get('repeat')}` -> {ours.get('repeat')}); "
                f"a call runs the called layers as many times as they run")
        spec = OP_REGISTRY.get(owner.get("op") or "")
        holds = owner.get("op") is not None and (spec is None
                                                 or spec.has_weight())
        if holds:
            theirs = {k: owner.get(k) for k in self.WEIGHT_FIELDS}
            mine = {k: ours.get(k) for k in self.WEIGHT_FIELDS}
            if theirs != mine:
                changed = sorted(k for k in theirs if theirs[k] != mine[k])
                raise ValueError(
                    f"Layer {self.name!r} rebinds {sorted(self.rebind)}, so "
                    f"{ours['name']!r} reads another weight geometry than "
                    f"{owner['name']!r}, whose weight it reads ({changed}: "
                    f"{[theirs[k] for k in changed]} -> "
                    f"{[mine[k] for k in changed]}); rebind only a symbol "
                    f"the other input sets on the activations (rows, items)")
        for sub_owner, sub in zip(owner.get("layers") or [],
                                  ours.get("layers") or [], strict=True):
            self._check_same_weights(sub_owner, sub)

    @staticmethod
    def _layers(layer: Dict[str, Any]):
        yield layer
        for sub in layer.get("layers") or []:
            yield from LayerCall._layers(sub)

    def _names(self) -> Dict[str, str]:
        """Each name the called layer declares, under the call's."""
        inner = list(self._layers(self.target))
        if self.target.get("group"):
            inner = inner[1:]
        paths = [str(layer["name"]).split(".") for layer in inner]
        shared: List[str] = []
        for parts in zip(*paths, strict=False):
            if len(set(parts)) != 1:
                break
            shared.append(parts[0])
        if self.target.get("group") and len(paths) == 1:
            # A group of one layer: its module is the layer's path above it.
            shared = shared[:-1]
        if not shared:
            raise ValueError(
                f"Layer {self.name!r} calls {self.target['name']!r}, whose "
                f"layers share no module path to rename under the call")
        renamed = [*shared[:-1], self.name]
        names = {".".join(parts): ".".join(renamed + parts[len(shared):])
                 for parts in paths}
        names[str(self.target["name"])] = self.name
        for layer in inner:
            for stream, decl in (layer.get("streams") or {}).items():
                if isinstance(decl, dict) and "from" in decl:
                    names[stream] = f"{stream}_{self.name}"
        return names

    def expand(self) -> Dict[str, Any]:
        names = self._names()
        outside: List[str] = []
        for layer in self._layers(self.target):
            refs = layer.get("inputs", layer.get("in")) or []
            for ref in [refs] if isinstance(refs, str) else refs:
                if ref not in names and ref not in outside:
                    outside.append(ref)
        if len(outside) != len(self.inputs):
            raise ValueError(
                f"Layer {self.name!r} calls {self.target['name']!r}, which "
                f"reads {outside}; the call gives {self.inputs}")
        names.update(zip(outside, self.inputs, strict=True))
        unused = set(self.rebind)
        pattern = (re.compile(r"\b(" + "|".join(map(re.escape, self.rebind))
                              + r")\b") if self.rebind else None)

        def rebound(value: Any) -> Any:
            if pattern is None or not isinstance(value, str):
                return value
            unused.difference_update(pattern.findall(value))
            return pattern.sub(lambda m: self.rebind[m.group(1)], value)

        def clone(layer: Dict[str, Any]) -> Dict[str, Any]:
            out = {k: v for k, v in layer.items() if k != "checkpoint"}
            out["name"] = names[str(layer["name"])]
            refs = out.pop("in", None) or out.get("inputs")
            if refs is not None:
                out["inputs"] = [names.get(ref, ref) for ref in (
                    [refs] if isinstance(refs, str) else refs)]
            if isinstance(out.get("streams"), dict):
                out["streams"] = {
                    names.get(stream, stream): (
                        {**decl, "from": names.get(decl["from"], decl["from"])}
                        if isinstance(decl, dict) and "from" in decl else decl)
                    for stream, decl in out["streams"].items()}
            if isinstance(out.get("dims"), dict):
                out["dims"] = {k: rebound(v) for k, v in out["dims"].items()}
            for key in ("dim", "repeat"):
                if key in out:
                    out[key] = rebound(out[key])
            if isinstance(out.get("shape"), list):
                out["shape"] = [rebound(axis) for axis in out["shape"]]
            if out.get("layers"):
                out["layers"] = [clone(sub) for sub in out["layers"]]
            return out

        out = clone(self.target)
        self._check_same_weights(self.target, out)
        if unused:
            raise ValueError(
                f"Layer {self.name!r} rebinds {sorted(unused)}, which "
                f"{self.target['name']!r} does not read")
        out["name"] = self.name
        out["call"] = str(self.target["name"])
        return out

    @classmethod
    def written_out(cls, layers: List[Any],
                    in_group: bool = False) -> List[Any]:
        """`layers` (a model list, or a group's when `in_group`) with each
        call written out as the earlier layer it names, and each group's
        calls inside it."""
        out: List[Any] = []
        for raw in layers:
            if isinstance(raw, dict) and isinstance(raw.get("layers"), list):
                raw = {**raw, "layers": cls.written_out(raw["layers"],
                                                        in_group=True)}
            if isinstance(raw, dict) and "call" in raw:
                earlier = [layer for layer in out if isinstance(layer, dict)
                           and layer.get("name") == raw["call"]]
                if not earlier or "call" in earlier[-1]:
                    raise ValueError(
                        f"Layer {raw.get('name')!r} calls {raw['call']!r}, "
                        f"which names no earlier layer beside it that runs "
                        f"its own weights")
                call = cls(raw, earlier[-1])
                call._check_target(in_group)
                raw = call.expand()
            out.append(raw)
        return out


def _parse_model_layer(raw: Dict[str, Any]) -> ModelLayer:
    """Parse a ModelLayer from raw YAML dict.

    Per-layer template selection is authored as a `per_layer: {assign, map}`
    block — a typed document structure (`document.PerLayerBlock`) — and expanded
    here into the internal `assignments` (BlockAssignment) list that the
    canonical resolver consumes.
    """
    if "per_layer" in raw:
        expanded = PerLayerBlock(**raw["per_layer"]).expand()
        raw = {k: v for k, v in raw.items() if k != "per_layer"}
        raw["assignments"] = [BlockAssignment(**a) for a in expanded]
    elif "assignments" in raw:
        raise ValueError(
            f"Layer {raw.get('name')!r} uses the removed `assignments:` syntax. "
            f"Author per-layer template selection as `per_layer: {{assign, map}}` "
            f"— each `assign` entry is a generator (range/layers/stride/"
            f"`from: config.X, is: v`/rest) tagged with a `type`, and `map` binds "
            f"each type to its template. See `document.PerLayerBlock`."
        )
    # Parse nested layers in groups
    if "layers" in raw and isinstance(raw["layers"], list):
        raw["layers"] = [_parse_model_layer(ml) for ml in raw["layers"]]

    # `in` → `inputs`, the same readability alias graph nodes get. Model
    # layers did not, so a top-level `in:` was dropped: Kimi-K3's
    # `lm_head` reads `norm` and Qwen3.5's vision `merger.gelu` reads
    # `linear_fc1`, and both silently fell back to "whatever layer came
    # before me".
    raw = dict(raw)
    if "in" in raw:
        raw["inputs"] = raw.pop("in")
    if isinstance(raw.get("inputs"), str):
        raw["inputs"] = [raw["inputs"]]

    return ModelLayer(**raw)


def _extract_unresolved_note(source_text: str) -> Optional[str]:
    """Extract an `# UNRESOLVED:` comment block from the top of a spec YAML.

    Spec-gen agents write this marker when they hit the Phase 4 escape
    hatch (4 validate_spec iterations without convergence). Format is a
    block of consecutive `#`-comment lines starting with `# UNRESOLVED:`;
    returned as joined non-`#`-prefixed text. None if no such block.
    """
    lines = source_text.splitlines()
    start = None
    for i, line in enumerate(lines[:50]):  # only scan the first 50 lines
        if re.match(r"#\s*UNRESOLVED\b", line, re.IGNORECASE):
            start = i
            break
    if start is None:
        return None
    collected: List[str] = []
    for line in lines[start:]:
        s = line.strip()
        if not s.startswith("#"):
            break
        collected.append(s.lstrip("#").strip())
    return "\n".join(collected).strip() or None


def parse_document(raw: Dict[str, Any], path: str) -> SpecDocument:
    """Build the typed authoring document from one spec YAML's contents.

    Two of the document's fields carry authoring conveniences pydantic cannot
    see — a block graph's `in:` alias and its role/kv-label rules, a model
    layer's `per_layer` block — so they are parsed first and handed over as
    objects. Everything else goes through `**raw`, which is what makes
    `extra="forbid"` load-bearing: a key this schema does not name is an
    error naming the key, where it used to be a `raw.get` nobody wrote.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML dict, got {type(raw)} in {path}")
    parsed: Dict[str, Any] = dict(raw)
    if "model" in parsed:
        parsed["model"] = [_parse_model_layer(ml) for ml in
                           LayerCall.written_out(parsed["model"] or [])]
    if "blocks" in parsed:
        root = Path(path).resolve().parents[1]
        parsed["blocks"] = {
            name: _attach_claims(
                _parse_block_graph(bg, identity=name),
                name, root)
            for name, bg in (parsed["blocks"] or {}).items()}
    try:
        return SpecDocument(**parsed)
    except ValidationError as e:
        raise ValueError(f"{path}: {e}") from e

