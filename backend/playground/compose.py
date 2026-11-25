"""Whole-model composition editor (playground Phase 2a).

Import a base model, swap library components into its layer groups, adjust
dims, and get the WHOLE-MODEL cost — canonical and byte-for-byte the same as
opening that model in the Inspector, because the numbers come from
`project_model` / `cache_view` on an edited COPY of the base spec (never a
slot-sum). Self-contained: no persistence, no `user:<slug>` id, no jump to
inference (that is Phase 2b).

The editor works on LAYER GROUPS: one per declaration behind the
`iter_blocks` records (`declared_blocks`), identified by its position in that
list (`group_id`). Mostly one record is one declaration, and the group's
ordinal is the one `Program.blocks`, `ModelReport.block_results` and
`CacheView` entries carry; a `read: all` stack's declaration stands for
several records, one per history length its layers see, and its group holds
them all. An edit is a change to the DECLARATION behind a group (the one
assignment / sub-layer / layer its `BlockMeta` points at), after which every
consumer re-resolves the spec as it always does. Shared block templates are never modified: a swap builds a new
template — lowering's own synthesis when the slot is a template parameter, a
copy of the block with the one node changed when it is written into the block
— and repoints that one declaration at it. So two groups that shared a
template (DeepSeek-V3's MoE layers and its MTP layer's inner block) stay
independent.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, List, Optional, Tuple

from backend.hf.metadata import fetch_config, fetch_metadata, stored_without_checkpoint
from backend.arch import dims as dims_mod
from backend.arch import roles as role_reg
from backend.arch.loader import LayerSpanError, ResolvedSpec, get_default_loader, serving_loader
from backend.arch.schema.model import BlockAssignment
from backend.inference.pipeline import NoPeakForDtype
from backend.inference.step_eval import StepCells
from backend.fingerprint import result_version
from backend.measure.cache import KeyedCache, MemoryBackend
from backend.measure.scope import ResultIdentity, of_render
from backend.arch.eval.dim import names_of, value_of
from backend.arch.eval.transforms import DECODE, PREFILL, CallSite, Deployment, binding_of
from backend.arch.eval.program import Program
from backend.arch.eval import capacity as capacity_mod
from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.projection import project_model
from backend.arch.spec_tree import StreamReadRefusal, iter_tree
from backend.arch.spec_tree.build import declared_blocks, iter_blocks
from backend.playground.module_bench import (
    closure_dim_refs, frame_names, inject_component, is_deployment_symbol,
    phase_units, serving_step, step_flops, step_weight_read, StepTraffic,
    snapshot_specs, unbound_dims, walk_dim_names, CONTRACT_ERRORS,
    phase_contract, phase_roofline, call_dto, timing_scope,
)


#: Whole-model costs (`Composition.cost`) by their identity
#: (`Composition.cost_identity`): a request's base side is the same cost
#: whatever the edit, and is computed once. In process memory, as the
#: computed object — nothing round-trips through JSON, so a row reads back
#: with the types it was built with.
_COSTS = KeyedCache("compose_cost", MemoryBackend(max_entries=32),
                    single_flight=True)


def _base_resolved(model_id: str, overlay_dir: Optional[str] = None):
    """The model as the Inspector serves it; `overlay_dir` is a caller's own
    serving overlay (a model they keep, from the service layer)."""
    loader, overlay_dir = serving_loader(model_id, overlay_dir)
    config = fetch_config(model_id, overlay_dir)
    if config is None:
        raise ValueError(f"could not fetch config for {model_id!r}")
    meta = fetch_metadata(model_id, overlay_dir) or {}
    resolved = loader.match(config, model_id=model_id,
                            header=meta.get("header"))
    if not resolved:
        raise ValueError(f"no spec matches {model_id!r}")
    return resolved, meta.get("header"), loader


_swappable_roles_cache: Optional[frozenset] = None


def swappable_roles() -> frozenset:
    """The roles a slot can be filled from: attention / ffn roles that some
    shipped model binds DIRECTLY in a backbone or MTP block template.

    Derived from the resolved corpus (the same snapshot resolutions
    `list_library_components`' provenance reads), not listed: kind `ffn`
    also holds `router` and kind `attention` holds `attention_indexer`,
    which are internal parts of MoE / DSA components and never sit in a block
    slot — offering them as a swap would build a model nobody ships."""
    global _swappable_roles_cache
    if _swappable_roles_cache is None:
        found = set()
        for _mid, resolved in snapshot_specs():
            for group in LayerGroup.all_of(resolved.spec, iter_blocks(resolved)):
                if group.editable:
                    found.update(slot.role for slot in group.slots())
        _swappable_roles_cache = frozenset(r for r in found if r)
    return _swappable_roles_cache


@dataclass(frozen=True)
class Slot:
    """One component reference in a group's block template.

    `path` is `node/node::param`: the node ids from the group template down
    through inner blocks, and — for a wrapper node carrying the component in
    a component-typed param (GLM-4's `pre_norm` with `attn: …`) — the param
    name after `::`. A reference on a node's `op` has no `::` part.
    """

    path: str
    component: str
    role: Optional[str]

    @property
    def kind(self) -> str:
        return role_reg.kind_of(self.role or "")

    @property
    def swappable(self) -> bool:
        """Attention and FFN work is swappable; norms and wrappers have no
        same-role alternative in the library and stay read-only."""
        return self.kind in ("attention", "ffn")

    @property
    def node_id(self) -> str:
        return self.path.rsplit("/", 1)[-1].partition("::")[0]

    @property
    def binding(self) -> str:
        """`op` for a node's op, else the param name that binds it."""
        return self.path.rsplit("/", 1)[-1].partition("::")[2] or "op"

    def candidates(self) -> List[str]:
        """Library components that can fill this slot: a role some shipped
        model binds directly in a block slot, of this slot's kind."""
        loader = get_default_loader()
        roles = swappable_roles()
        return sorted(
            name for name in loader.component_names()
            if (role := loader.component(name).role) in roles
            and role_reg.kind_of(role) == self.kind)


class LayerGroup:
    """One declaration's `iter_blocks` records, as the editor sees them
    (`declared_blocks`)."""

    def __init__(self, spec: Any, group_id: int, meta: Any):
        self.spec = spec
        self.group_id = group_id
        self.meta = meta

    @classmethod
    def all_of(cls, spec: Any, metas: List[Any]) -> List["LayerGroup"]:
        return [cls(spec, i, meta) for i, meta in enumerate(metas)]

    @property
    def template(self) -> str:
        return self.meta.template_name

    @property
    def editable(self) -> bool:
        """Backbone and MTP layers are edited alike; a vision / audio tower
        is one wrapper component — shown, not edited. A looped stack's
        groups are shown, not edited: an edit names layers, and the editor
        has no pass to say which execution of a looped layer it means."""
        return (bool(self.template) and self.meta.aux_kind in (None, "mtp")
                and not self.looped)

    @property
    def looped(self) -> bool:
        """Whether this group runs in a looped stack (`BlockMeta.pass_`,
        `stored_indices`)."""
        return self.meta.stored_indices is not None

    @property
    def address(self) -> Tuple[Tuple[int, ...], Optional[int]]:
        """The declaration behind this group (see `BlockMeta.layer_path`)."""
        return tuple(self.meta.layer_path), self.meta.assignment_index

    def references(self) -> List[Slot]:
        """Every component reference in the group template, descending into
        inner blocks (a node whose op is itself a block of this spec)."""
        spec = self.spec
        blocks = spec.blocks or {}
        components = spec.components or {}
        roles = spec.component_roles or {}
        out: List[Slot] = []

        def walk(name: str, prefix: Tuple[str, ...], seen: frozenset) -> None:
            for node_id, node in spec.get_block(name).graph.items():
                here = "/".join((*prefix, node_id))
                op = node.op
                if isinstance(op, str) and op in blocks and op not in seen:
                    walk(op, (*prefix, node_id), seen | {op})
                elif isinstance(op, str) and op in components:
                    out.append(Slot(here, op, roles.get(op)))
                for key, value in node.params.items():
                    if isinstance(value, str) and value in components:
                        out.append(Slot(f"{here}::{key}", value, roles.get(value)))

        walk(self.template, (), frozenset({self.template}))
        return out

    def slots(self) -> List[Slot]:
        """The swappable references: attention / ffn roles."""
        return [s for s in self.references() if s.swappable]

    def readonly(self) -> List[Slot]:
        """The rest of the block (norms, wrappers) — shown, not swappable."""
        return [s for s in self.references() if not s.swappable]

    def slot(self, path: str) -> Optional[Slot]:
        return next((s for s in self.slots() if s.path == path), None)

    def layers(self) -> Tuple[int, ...]:
        """The layer indices this group covers, in its own stack: the
        resolver's selection for an assignment, the whole stack for a
        `template:` declaration."""
        return self.meta.layer_numbers

    def layers_label(self) -> str:
        """The resolved layers in words: a run is `a–b`, a stride is
        `a, a+k, …, b (every k)`, and what is neither lists its count and
        first few layers. It describes what the resolver selected, not the
        selector that selected it (`selector()` is that)."""
        layers = self.layers()
        runs: List[Tuple[int, ...]] = []
        i, n = 0, len(layers)
        while i < n:
            j = i
            if i + 1 < n:
                step, j = layers[i + 1] - layers[i], i + 1
                while j + 1 < n and layers[j + 1] - layers[j] == step:
                    j += 1
            runs.append(layers[i:j + 1])
            i = j + 1
        if len(runs) > 3:
            head = ", ".join(str(n) for n in layers[:4])
            label = f"{len(layers)} layers: {head}, …"
        else:
            label = " · ".join(self._run_label(run) for run in runs)
        return f"{self.meta.aux_kind.upper()} · {label}" if self.meta.aux_kind else label

    @staticmethod
    def _run_label(run: Tuple[int, ...]) -> str:
        if len(run) == 1:
            return str(run[0])
        step = run[1] - run[0]
        if step == 1:
            return f"{run[0]}–{run[-1]}"
        if len(run) == 2:
            return f"{run[0]}, {run[1]}"
        return f"{run[0]}, {run[1]}, …, {run[-1]} (every {step})"

    def selector(self) -> Dict[str, Any]:
        """How the spec selects these layers, as written (`range`, `stride`,
        `config_list`, `rest`, …) — the provenance of `layers()`."""
        layer_path, index = self.address
        if index is None:
            return {}
        declaration = Composition._layer(self.spec, layer_path).assignments[index]
        return declaration.model_dump(
            include={"range", "layers", "rest", "stride", "offset",
                     "config_list", "equals"}, exclude_none=True)

    def label_of(self, slot: Slot) -> str:
        """What a slot holds, as declared: an attention component's
        `attn_display` (MLA / KDA / SWA …), an FFN's role. The role is the
        component's own (`role:`), not the slot's: Qwen3-MoE plugs
        `moe_ffn` into its `ffn_dense` slot, and its layers are MoE."""
        block = self.spec.get_block(slot.component)
        if slot.kind == "attention":
            return block.attn_display or slot.component
        return block.role or slot.role or slot.component

    def composition(self) -> List[str]:
        """The group's makeup, one label per swappable slot."""
        return [self.label_of(slot) for slot in self.slots()]


class Composition:
    """A base model and an edited copy of its spec.

    Edits are addressed by the BASE's `group_id` and land on the declaration
    behind that group, so they compose in any order and never reach a group
    that merely shares a template. The base resolution is never modified.
    Refused edits are recorded (`refusals()`) and not applied.
    """

    def __init__(self, base: ResolvedSpec, *, model_id: Optional[str] = None,
                 header: Any = None,
                 loader: Any = None):
        self.base = base
        self.model_id = model_id
        self.header = header
        self.loader = loader or get_default_loader()
        self._spec = base.spec
        self._dims: Dict[str, Any] = {}
        self._refusals: List[Dict[str, Any]] = []
        self._swaps: List[Tuple[int, str, str]] = []
        # per-group dim requests as the user wrote them, and what each one
        # came to (the names whose value moved, or why it was refused) —
        # re-derived whenever the global dims move
        self._group_dims: Dict[int, Dict[str, Any]] = {}
        self._group_changes: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._group_refusals: Dict[int, Dict[str, Any]] = {}
        # The stacks declared by a bare `template:` that a per-group dims
        # edit has rewritten as one `rest:` assignment (`_as_assignment`):
        # their group now lives at assignment 0 of the edited copy.
        self._as_assignments: set = set()
        self._resolved: Optional[ResolvedSpec] = None
        self._interface: Optional[Dict[Optional[int], frozenset]] = None
        self._walked: Optional[Tuple[ResolvedSpec, Any]] = None
        self._base_groups = LayerGroup.all_of(
            base.spec, declared_blocks(iter_blocks(base)))

    # -- reading ----------------------------------------------------------

    def groups(self) -> List[LayerGroup]:
        """The BASE's layer groups; `group_id` is what every edit names."""
        return list(self._base_groups)

    def calls_at(self, group: LayerGroup, slot: Slot) -> List[CallSite]:
        """The base's calls at `slot` of `group` (`CallSite.calls`): in each
        block invocation of the group's declaration, the node the slot's
        path names, or — for a slot a wrapper binds through a param — the
        wrapper's call of the component. A module dropped into the slot is
        evaluated at one of them."""
        *heads, last = slot.path.split("/")
        node_id, _, param = last.partition("::")
        prefix = ".".join((*heads, node_id))
        blocks = Program.of(self.base).blocks
        return [
            site for site in CallSite.calls(self.base)
            if binding_of(self.base.spec, site.component, slot.component)
            and site.invocation is not None
            and (tuple(blocks[site.invocation].meta.layer_path),
                 blocks[site.invocation].meta.assignment_index) == group.address
            and (site.path.startswith(f"{prefix}.") if param
                 else site.path == prefix)]

    def bound_with(self, slot: Slot) -> List[Tuple[LayerGroup, Slot]]:
        """Every editable group's slot of the base that holds the same
        component as `slot` — what "all groups bound to X" swaps."""
        return [(group, other)
                for group in self._base_groups if group.editable
                for other in group.slots()
                if other.component == slot.component and other.kind == slot.kind]

    def declaration(self, group_id: int) -> Any:
        """The edited copy's declaration behind `group_id` — the assignment,
        or the layer / sub-layer when it declares a bare `template:`."""
        layer_path, index = self._address(group_id)
        layer = self._layer(self._spec, layer_path)
        return layer if index is None else layer.assignments[index]

    def current(self, group_id: int) -> LayerGroup:
        """`group_id` as it stands in the edited copy (its current template)."""
        meta = replace(self._base_groups[group_id].meta,
                       template_name=self.declaration(group_id).template)
        return LayerGroup(self._spec, group_id, meta)

    def refusals(self) -> List[Dict[str, Any]]:
        out = [*self._refusals, *self._group_refusals.values()]
        try:
            resolved = self.resolved()
        except LayerSpanError as exc:
            out.append({"kind": "layer_span", "group_id": None,
                        "detail": str(exc)})
            return out
        except ValueError as exc:
            # the edited dims do not resolve (an unregistered symbol, an
            # expression over a dim the model lacks)
            out.append({"kind": "dims", "group_id": None, "detail": str(exc)})
            return out
        out.extend({"kind": "dims", "group_id": None, "detail": problem}
                   for name in self._dims
                   if (problem := self.dim_problem(
                       name, resolved.get_dim(name))))
        try:
            # A swapped-in component that takes other tensors than the slot
            # hands over, or leaves slots to its caller: the load-time
            # check, on the edited spec.
            resolved.spec.check_call_bindings()
        except ValueError as exc:
            out.append({"kind": "call", "group_id": None,
                        "detail": str(exc)})
            return out
        try:
            iter_tree(resolved)
        except StreamReadRefusal as exc:
            # A reading layer whose declared source no longer writes the
            # stream (its writer was swapped away), reported on the reading
            # group.
            out.append({"kind": "stream",
                        "group_id": (None if exc.meta is None
                                     else self._group_of(exc.meta)),
                        "detail": f"{exc}. {exc.fix}"})
        except CONTRACT_ERRORS:
            # Building the IR needs every dim bound. A swapped-in
            # component's unbound dims are `unbound_dims()`, not a stream
            # refusal — `compose_groups` asks that next.
            return out
        return out

    def unbound_dims(self) -> List[Dict[str, Any]]:
        """Per edited group, the dims its block template's closure spells
        out that nothing binds — neither the edited model's dims nor the
        group's own — read statically off the spec, not guessed from an
        evaluation error. Each entry names the swapped-in component whose
        closure references them; what no swap accounts for is reported for
        the group (`slot` None)."""
        resolved = self.resolved()
        metas = {self._group_of(meta): meta for meta in iter_blocks(resolved)}
        swaps: Dict[int, List[Tuple[str, str]]] = {}
        for group_id, slot, component in self._swaps:
            swaps.setdefault(group_id, []).append((slot, component))
        out = []
        for group_id, filled in swaps.items():
            meta = metas.get(group_id)
            if meta is None:   # the group has no layers left
                continue
            missing = set(unbound_dims(resolved, meta.dim_overrides,
                                        component=meta.template_name))
            for slot, component in filled:
                mine = missing & closure_dim_refs(resolved, component)
                if mine:
                    out.append({"group_id": group_id, "slot": slot,
                                "component": component,
                                "missing_dims": sorted(mine)})
                    missing -= mine
            if missing:
                out.append({"group_id": group_id, "slot": None,
                            "component": None,
                            "missing_dims": sorted(missing)})
        return out

    def group_dim_changes(self, group_id: int) -> Dict[str, Dict[str, Any]]:
        """What a group's dims came to after its edit: every dim whose value
        in the group differs from the model's, `{name: {"global": g,
        "group": v}}` — the group's declared rebindings, the edited names,
        and every dim the edit moved."""
        return dict(self._group_changes.get(group_id, {}))

    def interface_dims(self, group_id: int) -> frozenset:
        """The dims the tensors crossing this group's boundary are
        expressions in: what it reads from the layer before (or, for an MTP
        group, from the backbone and the shared embedding) and what it hands
        on. A group cannot change them on its own — the residual stream and
        the MTP input contract are the same tensor on both sides.

        Read off the base model's walk, not listed: the names (`frame_names`)
        of the block's output roots and of what its entry nodes take in,
        without the per-request axes. In the corpus that is `d_model`, plus
        the hyper-connection copies on DeepSeek-V4 and the vocabulary on an
        MTP head."""
        if self._interface is None:
            walk = self._checkpoint_walk(self.base)
            names: Dict[Optional[int], frozenset] = {}
            for meta, block in zip(iter_blocks(self.base),
                                   walk.program.blocks, strict=True):
                group = self._group_of(meta)
                names[group] = names.get(group, frozenset()) | (
                    self._boundary_names(walk.program, walk, block))
            self._interface = names
        return self._interface[group_id]

    def local_dims(self, group_id: int) -> Dict[str, Dict[str, Any]]:
        """The dims this group's evaluation depends on, valued as the group
        binds them: `{name: {"value", "global"?, "binding"?, "interface"?}}`
        — `global` when the group's value differs from the model's,
        `binding` the group's own rebinding as written
        (`n_kv_heads ← n_kv_heads_full`), `interface` for a dim only a
        global edit can change, `group` the section `dims.yaml` files the
        symbol under (None for a name the registry does not know). The
        dims come in the registry's section order.

        "Depends on" as for a module card (`_dependency_dims`): the free
        names of the walk (`walk_dim_names`), restricted to this block invocation's
        nodes, plus the static scan of its template's closure. The walk is
        the edited model's own, so the block sees the tensor it really
        receives (DeepSeek-V4's hyper-connection copies do not isolate)."""
        resolved = self.resolved()
        metas = iter_blocks(resolved)
        index = next((i for i, m in enumerate(metas)
                      if self._group_of(m) == group_id), None)
        if index is None:   # the group has no layers left
            return {}
        meta = metas[index]
        if self._walked is None or self._walked[0] is not resolved:
            self._walked = (resolved, self._checkpoint_walk(resolved))
        walk = self._walked[1]
        block = walk.program.blocks[index]
        depends = (closure_dim_refs(resolved, meta.template_name)
                   | walk_dim_names(walk, {n.uid for n in block.nodes()}))
        # What the walk bound: the model's dims, and the frame the block's
        # assignment opened over them (`Analysis.block_frame`).
        model = walk.binding
        values = walk.block_frame(block)
        interface = self.interface_dims(group_id)
        sections = dims_mod.groups()
        rank = {g.slug: i for i, g in enumerate(sections)}
        title = {g.slug: g.title for g in sections}

        def section(name: str) -> Optional[str]:
            sym = dims_mod.get(name)
            return sym.group if sym is not None else None

        out: Dict[str, Dict[str, Any]] = {}
        for name in sorted(depends, key=lambda n: (
                rank.get(section(n), len(rank)), n)):
            value = value_of(values.get(name))
            if (is_deployment_symbol(name) or isinstance(value, bool)
                    or not isinstance(value, (int, float))):
                continue
            row: Dict[str, Any] = {"value": value}
            if value_of(model.get(name)) != value:
                row["global"] = value_of(model.get(name))
            if name in meta.dim_overrides_raw:
                row["binding"] = meta.dim_overrides_raw[name]
            if name in interface:
                row["interface"] = True
            row["group"] = title.get(section(name))
            out[name] = row
        return out

    def resolved(self) -> ResolvedSpec:
        """The edited spec, resolved. Global dim overrides are written into
        the spec's `dims:` and the spec is resolved again, so every dim
        derived from an edited one (`dims.yaml` `derived:`, or an expression
        in the spec) moves with it."""
        if self._resolved is None:
            if self._dims:
                resolved = self._resolve_with(self._dims)
            else:
                resolved = self.base.with_spec(self._spec)
            iter_blocks(resolved)   # a literal layer table out of span raises here
            self._resolved = resolved
        return self._resolved

    # -- editing ----------------------------------------------------------

    def swap(self, group_id: int, slot_path: str, component: str) -> bool:
        """Fill `slot_path` of group `group_id` with library `component`.

        Returns whether the swap was applied; a refused one is recorded."""
        group = self._editable(group_id, slot=slot_path)
        if group is None:
            return False
        live = self.current(group_id)
        slot = live.slot(slot_path)
        if slot is None:
            return self._refuse("unknown_slot", group_id, slot_path,
                                f"template {live.template!r} has no swappable "
                                f"slot {slot_path!r}")
        if component not in self.loader.component_names():
            if slot.component == component:
                return True
            return self._refuse("unknown_component", group_id, slot_path,
                                f"{component!r} is not a library component")
        role = ((self._spec.component_roles or {}).get(component)
                or self.loader.component(component).role)
        if role not in swappable_roles() or role_reg.kind_of(role) != slot.kind:
            return self._refuse(
                "role_kind", group_id, slot_path,
                f"{component!r} fills role {role!r}; slot {slot_path!r} "
                f"takes a {slot.kind} component")
        self._spec = inject_component(
            ResolvedSpec(self._spec, {}, {}, {}), component, self.loader).spec
        template = self._rebind(live.template, slot_path.split("/"), component)
        self._repoint(self._address(group_id), template)
        self._swaps.append((group_id, slot_path, component))
        return True

    def set_dims(self, overrides: Dict[str, Any]) -> None:
        """Global dim overrides (a value may be an expression over dims).
        Each group's own dim edit is derived again against them. A name no
        model can bind is refused and not applied (`dim_problem`)."""
        overrides = dict(overrides or {})
        for name in list(overrides):
            problem = self.dim_problem(name, None)
            if problem:
                self._refuse("dims", None, None, problem)
                del overrides[name]
        self._dims.update(overrides)
        self._resolved = None
        for group_id in self._group_dims:
            self._bind_group_dims(group_id)

    def set_group_dims(self, group_id: int, dims: Dict[str, Any]) -> bool:
        """Rebind dims for one group (a value may be an expression over the
        model's dims). Returns whether the edit was applied.

        The group is resolved again with its edit in force — the model's
        dims, the group's existing rebindings, the edit — so a dim derived
        from an edited one moves inside the group as it would globally
        (`n_experts` → `n_expert_copies`, `n_route_targets`, …). Every name
        whose value then differs from the model's is reported
        (`group_dim_changes`), and every one a spec may write is written into
        the group's assignment `dims:`; a `derived:` name is not, because the
        assignment's frame derives it again (`Binding.frame`). The
        edited names keep the form the user wrote; a rebinding the group
        already has and the edit does not name keeps the form the spec wrote
        (`n_kv_heads: n_kv_heads_full`, which decides the world that shards
        the axis); a derived name is written as its value.

        A group declared by a bare `template:` has no `dims:` of its own; an
        edit that writes any is applied by declaring the stack as the one
        `rest:` assignment it is equivalent to (`_as_assignment`).

        Refused: a change to a dim the group's boundary tensors are
        expressions in (`interface_dims`), and a group whose stack runs
        another layer's weights (`call:`), which a per-group binding would
        silently detach from them."""
        if self._editable(group_id) is None:
            return False
        # A refused edit is recorded and not applied: the group keeps the
        # request (and what it came to) from before this call, so a later
        # edit is not refused again for a key the user never got in.
        previous = self._group_dims.get(group_id)
        changes = self._group_changes.get(group_id)
        self._group_dims[group_id] = {**(previous or {}), **dims}
        if self._bind_group_dims(group_id):
            return True
        self._refusals.append(self._group_refusals.pop(group_id))
        if previous is None:
            del self._group_dims[group_id]
        else:
            self._group_dims[group_id] = previous
        if changes is not None:
            self._group_changes[group_id] = changes
        return False

    def check_skeleton(self, group_id: int, *, template_name: str,
                       count: int, assignment_index: Optional[int]) -> bool:
        """Whether the group the editor was shown is still `group_id` of the
        base: the same template, layer count and assignment. Any mismatch
        means the base spec changed between two requests (a template name
        alone does not tell: one template can back two groups), and the
        edit is refused until the model is loaded again."""
        if not 0 <= group_id < len(self._base_groups):
            return self._refuse("unknown_group", group_id, None,
                                f"no layer group {group_id}")
        meta = self._base_groups[group_id].meta
        now = (meta.template_name, meta.count, meta.assignment_index)
        sent = (template_name, count, assignment_index)
        if now == sent:
            return True
        return self._refuse(
            "stale_skeleton", group_id, None,
            f"group {group_id} is now template {now[0]!r} over {now[1]} "
            f"layers (assignment {now[2]}); the editor sent {sent[0]!r} over "
            f"{sent[1]} layers (assignment {sent[2]}). Reload the model.")

    # -- cost -------------------------------------------------------------

    def cost(self, *, batch: int, input_tokens: int, precision: Any,
             kv_cache_lens: Optional[List[int]] = None,
             gpu_sku: Optional[str] = None) -> Dict[str, Any]:
        """`_cost`, held in `_COSTS` under what it is a function of
        (`cost_identity`): a request reads the base model's cost the last
        one computed, and a compose that edits nothing reads it twice. Each
        caller gets a report of its own, never the held one."""
        identity = self.cost_identity(
            batch=batch, input_tokens=input_tokens, precision=precision,
            kv_cache_lens=kv_cache_lens, gpu_sku=gpu_sku)
        return copy.deepcopy(_COSTS.get_or_compute(
            identity.key, lambda: self._cost(
                batch=batch, input_tokens=input_tokens, precision=precision,
                kv_cache_lens=kv_cache_lens, gpu_sku=gpu_sku)))

    def cost_identity(self, *, batch: int, input_tokens: int,
                      precision: Any,
                      kv_cache_lens: Optional[List[int]] = None,
                      gpu_sku: Optional[str] = None) -> ResultIdentity:
        """What `cost` is a function of: the base's and the edited
        resolution's engine identity (`result_version`: code, shared data,
        each one's spec footprint, config and dims — the edited one carries
        every swap and dim edit, the base one the groups an edited block is
        joined to), the request, and the Kernel Bench scope its times read
        (`timing_scope`; none without a card)."""
        return of_render(
            "compose-cost",
            [result_version(self.base), result_version(self.resolved())],
            {"model_id": self.model_id, "batch": int(batch),
             "input_tokens": int(input_tokens),
             "precision": precision.key,
             "kv_cache_lens": [int(n) for n in kv_cache_lens or ()],
             "gpu_sku": gpu_sku},
            timing_scope(gpu_sku))

    def _cost(self, *, batch: int, input_tokens: int, precision: Any,
              kv_cache_lens: Optional[List[int]] = None,
              gpu_sku: Optional[str] = None) -> Dict[str, Any]:
        """Whole-model totals and per-group rows off the Inspector's path
        (`project_model`, `cache_view`): `project_model` gives the top-level
        params / FLOPs (the declared basis) and the group rows, and nothing
        per phase. A phase's FLOPs are its serving step's rows, the basis
        its bytes are on (see `flops` below for how the two bases differ). KV
        at other lengths is the same `cache_view` evaluated at
        `kv_cache_lens` — not a re-projection. Weights are the stored model,
        scales and buffers included (`capacity.weights`), as module-bench
        prices an entry; each group's and the model level's share comes from
        the same ledger.

        A step's traffic — activation bytes unfused and fused, the decode
        step's KV read, the weights it streams — is the sum of serving's
        per-op rows for that step (`serving_step`), so every layer counts as
        often as the model runs it. Each phase is one serving step: its
        fused rows and, off the same walk, its rows before the rewrite are
        the two sides; the step's walk is the one its timing reads too."""
        resolved = self.resolved()
        # The whole checkpoint, each encoder group walked in its own encode
        # binding with no media (`Deployment.checkpoint`): every group's row
        # and weights, the language model's step for every FLOP.
        report = project_model(resolved, batch=batch, input_tokens=input_tokens,
                               checkpoint=True, precision=precision)
        analysis = report.analysis
        weights = capacity_mod.weights(analysis)
        weights_by_block = weights.by_block(analysis.program)
        cells = StepCells()
        steps = {
            phase: serving_step(
                resolved, batch=batch, input_tokens=input_tokens,
                phase=phase, precision=precision, cells=cells)
            for phase in (PREFILL, DECODE)}
        naive = StepTraffic.of_rows(steps[PREFILL].pre_per_op_timing)
        fused = StepTraffic.of_rows(steps[PREFILL].per_op_timing)
        view = cache_view_from_analysis(analysis)
        kv_by_block = view.storage_bytes_by_block(input_tokens)
        rows = []
        for index, (meta, (template, count, block)) in enumerate(
                zip(iter_blocks(resolved), report.block_results, strict=True)):
            group = LayerGroup(resolved.spec, index, meta)
            rows.append({
                "group_id": self._group_of(meta),
                "block_index": index,
                # Which of its declaration's records this is: the history
                # lengths its layers see (`BlockMeta.visible`), empty when
                # the declaration is one record.
                "visible": dict(meta.visible),
                "template": template,
                "count": int(count),
                "aux_kind": meta.aux_kind,
                "layers": list(group.layers()),
                "layers_label": group.layers_label(),
                "total_params": int(block.total_params),
                "total_active_params": int(block.total_active_params),
                "total_flops": int(block.total_flops),
                "attention_params": int(block.attention_params),
                "attention_flops": int(block.attention_flops),
                "ffn_params": int(block.ffn_params),
                "ffn_flops": int(block.ffn_flops),
                "memory_params": int(block.memory_params),
                "memory_flops": int(block.memory_flops),
                "residual_params": int(block.residual_params),
                "residual_flops": int(block.residual_flops),
                "routed_params": int(block.routed_params),
                "quant_metadata_params": int(block.quant_metadata_params),
                "total_storage_bytes": int(block.total_storage_bytes),
                # None: an aux stack's cache is not in the whole-model KV
                # (Inspector scope), which is not the same as 0.
                "kv_bytes_at_L": (int(kv_by_block.get(index, 0))
                                  if meta.aux_kind is None else None),
                # the numbers above are per layer; these are the group's
                # whole contribution (all `count` layers), the unit of
                # `kv_bytes_at_L` — rows plus the model level sum to the
                # whole-model totals
                "group_totals": {
                    "params": int(count) * int(block.total_params),
                    "active_params": int(count) * int(block.total_active_params),
                    "flops": int(count) * int(block.total_flops),
                    "storage_bytes": int(weights_by_block.get(index, 0)),
                },
            })
        act_naive = naive.act_bytes
        act_fused = fused.act_bytes
        kv_at_L = int(view.storage_bytes_at(input_tokens))
        decode_naive = StepTraffic.of_rows(steps[DECODE].pre_per_op_timing)
        decode_fused = StepTraffic.of_rows(steps[DECODE].per_op_timing)
        # The served step's cache read, as compare mode reads it off the
        # runtime table: unfused, attention's score and apply each read the
        # cache, and the framework's one attention launch reads it once.
        decode_kv_read = decode_fused.kv_reads
        at = phase_contract(batch, input_tokens)
        prefill_weight = step_weight_read(steps[PREFILL].pre_per_op_timing)
        decode_weight = step_weight_read(steps[DECODE].pre_per_op_timing)
        units_scope = "whole model"
        return {
            "params": int(report.total_params),
            "active_params": int(report.total_active_params),
            # The declared structure's prefill FLOPs (`project_model`): what
            # the groups' rows and the model level add up to. A phase's
            # FLOPs below are its serving step's, the basis its bytes are on,
            # and the two differ in three ways: the declared walk computes
            # the logits at every position, a serving step only at the ones
            # it samples; the declared total counts the aux stacks (MTP), a
            # serving step the main stack only; and a serving step counts
            # the activation casts it grafts (Llama-3.2-3B at L=2048 prefill:
            # 13.89 T declared, 12.28 T served — the logits alone).
            "flops": int(report.total_flops),
            "weight_bytes": int(weights.resident),
            "act_bytes_naive": act_naive,
            "act_bytes_fused": act_fused,
            "kv_bytes_at_L": kv_at_L,
            "phases": {
                PREFILL: {
                    "flops": step_flops(steps[PREFILL].per_op_timing),
                    "act_bytes_naive": act_naive,
                    "act_bytes_fused": act_fused,
                    "weight_read_bytes": prefill_weight,
                    # KV write of the whole batch, same as compare-mode
                    # `_phase_row.kv_step_bytes` (per-request storage × B).
                    "kv_step_bytes": kv_at_L * int(batch),
                    "units": phase_units(
                        PREFILL, at[PREFILL], scope=units_scope),
                },
                DECODE: {
                    "flops": step_flops(steps[DECODE].per_op_timing),
                    "act_bytes_naive": decode_naive.act_bytes,
                    "act_bytes_fused": decode_fused.act_bytes,
                    "weight_read_bytes": decode_weight,
                    # Every layer's cache read for the whole batch, so it
                    # sits beside `weight_read_bytes` the way compare-mode's
                    # `kv_step_bytes` does — no extra × batch.
                    "kv_read_bytes": decode_kv_read,
                    "memory_dominant": (
                        "kv" if decode_kv_read > decode_weight else "weights"),
                    "units": phase_units(
                        DECODE, at[DECODE], scope=units_scope),
                },
            },
            "kv_storage_by_cache_len": [
                {"cache_len": n, "storage_bytes": int(view.storage_bytes_at(n))}
                for n in (kv_cache_lens or ())],
            **(self._timings(resolved, batch=batch, input_tokens=input_tokens,
                             precision=precision, gpu_sku=gpu_sku,
                             cells=cells)
               if gpu_sku else {}),
            "rows": rows,
            "aux_params": {k: int(v) for k, v in report.aux_params.items()},
            "model_level": self._model_level(report),
            "model_level_totals": {
                "params": sum(int(n.params) for n in report.model_level_nodes.values()),
                "active_params": sum(int(n.active_params)
                                     for n in report.model_level_nodes.values()),
                "flops": sum(int(n.flops) for n in report.model_level_nodes.values()),
                "storage_bytes": int(weights_by_block.get(None, 0)),
            },
        }

    def _timings(self, resolved: ResolvedSpec, *, batch: int,
                 input_tokens: int, precision: Any, gpu_sku: str,
                 cells: StepCells) -> Dict[str, Any]:
        """`timings` per phase on `gpu_sku`, or — when the card lists no
        peak for the dtype the model computes in — `timings_unavailable`,
        why there is no time (the FLOPs and bytes stand without it)."""
        try:
            return {"timings": {
                phase: phase_roofline(
                    resolved, batch=batch, input_tokens=input_tokens,
                    precision=precision, phase=phase, gpu_sku=gpu_sku,
                    include_aux_timing=False, model_id=self.model_id,
                    cells=cells,
                )
                for phase in (PREFILL, DECODE)
            }}
        except NoPeakForDtype as exc:
            return {"timings_unavailable": str(exc)}

    @staticmethod
    def _model_level(report: Any) -> Dict[str, int]:
        """Embedding / head / the rest of the top-level weights (final norm
        …), so rows + model level sum to the whole-model params."""
        ml_total = sum(int(n.params) for n in report.model_level_nodes.values())
        emb = int(report.embedding_params)
        head = int(report.lm_head_params)
        return {
            "embedding_params": emb,
            "lm_head_params": head,
            "lm_head_flops": int(report.lm_head_flops),
            "final_norm_and_other_params": max(0, ml_total - emb - head),
        }

    # -- internals --------------------------------------------------------

    def _editable(self, group_id: int, slot: Optional[str] = None
                  ) -> Optional[LayerGroup]:
        if not 0 <= group_id < len(self._base_groups):
            self._refuse("unknown_group", group_id, slot,
                         f"no layer group {group_id}")
            return None
        group = self._base_groups[group_id]
        if not group.editable:
            self._refuse("read_only_group", group_id, slot,
                         f"group {group_id} ({group.meta.aux_kind}) is read-only")
            return None
        return group

    def _refuse(self, kind: str, group_id: Optional[int], slot: Optional[str],
                detail: str) -> bool:
        entry = {"kind": kind, "group_id": group_id, "detail": detail}
        if slot is not None:
            entry["slot"] = slot
        self._refusals.append(entry)
        return False

    def _group_of(self, meta: Any) -> Optional[int]:
        """The base `group_id` of a block record of the edited spec."""
        address = (tuple(meta.layer_path), meta.assignment_index)
        if address[0] in self._as_assignments:
            address = (address[0], None)
        return next((g.group_id for g in self._base_groups
                     if g.address == address), None)

    def _resolve_with(self, overrides: Dict[str, Any]) -> ResolvedSpec:
        """The edited spec resolved with `overrides` written into its
        `dims:` as bindings, after the global overrides. A name the spec
        does not bind goes first, so a binding that reads it sees it."""
        spec = self._spec
        written = {**self._dims, **overrides}
        bindings = {k: str(v) for k, v in written.items() if k not in spec.dims}
        bindings.update(spec.dims)
        bindings.update({k: str(v) for k, v in written.items()})
        return self.loader.resolve(
            spec.model_copy(update={"dims": bindings}), self.base.config,
            model_id=self.model_id, header=self.header)

    def dim_problem(self, name: str, value: Any) -> Optional[str]:
        """Why `name = value` is not a dim edit, or None. A dim is a symbol
        dims.yaml or this spec's `dim_symbols:` registers, and it counts
        something, so its value is a whole number, 0 or more (0 is how a
        model says it has none of a thing: `n_shared_experts: 0`). `value`
        None checks the name only."""
        if name not in dims_mod.registered_for(self._spec.dim_symbols):
            return (f"{name!r} is not a dim: no model can bind that name. "
                    f"Use one of the names listed in the group's dims")
        derived = dims_mod.written_derived_error(
            name, "this edit", local=self._spec.dim_symbols)
        if derived:
            return derived
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return (f"{name} = {value}: a dim takes a whole number (or an "
                    f"expression over the model's dims that comes to one)")
        if value != int(value) or value < 0:
            return (f"{name} = {value}: a dim counts something, so it is a "
                    f"whole number, 0 or more")
        return None

    def _bind_group_dims(self, group_id: int) -> bool:
        """Derive the group's dim edit against the current global dims and
        write it into the group's declaration (see `set_group_dims`).

        "Moved" is measured against the group as it stands without the edit
        (the model's dims plus the group's declared rebindings), so an edit
        writes the names it changes and no others."""
        group = self._base_groups[group_id]
        declared_raw = self._declared_dims(group)
        request = self._group_dims[group_id]
        self._group_refusals.pop(group_id, None)
        self._group_changes.pop(group_id, None)

        def refuse(kind: str, detail: str) -> bool:
            self._group_refusals[group_id] = {
                "kind": kind, "group_id": group_id, "detail": detail}
            return False

        try:
            model = self.resolved()
            declared = {k: model.get_dim(v) for k, v in declared_raw.items()}
            edit = {k: model.get_dim(str(v)) for k, v in request.items()}
            problems = [p for k, v in edit.items()
                        if (p := self.dim_problem(k, v))]
            if problems:
                return refuse("dims", "; ".join(problems))
            before = self._resolve_with(declared) if declared else model
            after = self._resolve_with({**declared, **edit})
        except ValueError as exc:
            return refuse("dims", str(exc))
        # The editor's own subject: two resolutions it made of the spec it
        # is editing, compared name by name (no walk binds a written dim
        # that only re-resolution re-reads — see below).
        model_dims, before_dims, after_dims = (
            model.dim_values, before.dim_values, after.dim_values)
        moved = {k: v for k, v in after_dims.items() if before_dims.get(k) != v}
        crossing = sorted(set(moved) & self.interface_dims(group_id))
        if crossing:
            return refuse(
                "interface_dim",
                f"{', '.join(crossing)} is the width of the tensors group "
                f"{group_id} takes in and hands on (the residual stream, or "
                f"an MTP layer's input from the backbone); one group cannot "
                f"change it — edit it globally")
        dims = {**declared_raw, **{k: str(v) for k, v in request.items()}}
        # A `derived:` name is not written: the assignment's frame derives it
        # again from the rebound names (`Binding.frame`), as the walk does
        # everywhere, so writing its value would be a second statement of
        # it. An authorable name that moved — a model-level expression such
        # as `d_ffn_shared: "d_ffn_expert * n_shared_experts"` — is written,
        # because only this re-resolution re-reads that expression.
        for name, value in moved.items():
            row = dims_mod.get(name)
            if row is not None and row.derived:
                continue
            dims.setdefault(name, str(value))
        layer_path, index = self._address(group_id)
        if index is None and dims:
            layer = self._layer(self._spec, layer_path)
            if layer.call is not None:
                return refuse(
                    "group_dims",
                    f"group {group_id} runs the weights of layer "
                    f"{layer.call!r}; its dims are that layer's — edit them "
                    f"there or globally")
            self._as_assignment(layer_path)
            index = 0
        if index is not None:
            self._edit(layer_path, index, {"dims": dims})
        self._group_changes[group_id] = {
            k: {"global": model_dims.get(k), "group": after_dims[k]}
            for k in {**declared, **edit, **moved}
            if k in after_dims and model_dims.get(k) != after_dims[k]}
        return True

    def _address(self, group_id: int) -> Tuple[Tuple[int, ...], Optional[int]]:
        """Where `group_id`'s declaration sits in the edited copy: the base
        address, or assignment 0 once its bare `template:` was rewritten."""
        layer_path, index = self._base_groups[group_id].address
        if index is None and layer_path in self._as_assignments:
            return layer_path, 0
        return layer_path, index

    def _as_assignment(self, layer_path: Tuple[int, ...]) -> None:
        """Declare the bare `template:` stack at `layer_path` as the one
        assignment it is equivalent to — `rest: true` over the same `repeat`,
        carrying its template, params and stream sources — so it has a
        `dims:` to write. The layers, their order and every number stay as
        they were; the corpus gate checks that byte for byte."""
        layer = self._layer(self._spec, layer_path)
        # A stack without `repeat:` is one layer, and has no count for
        # `rest:` to fill: select that layer by number instead (adding a
        # `repeat:` would also give its tensors a layer index).
        selection = ({"rest": True} if layer.repeat is not None
                     else {"range": ["0", "0"]})
        assignment = BlockAssignment(
            **selection, template=layer.template, params=dict(layer.params),
            reads_from=layer.reads_from)
        update = {"template": None, "params": {}, "reads_from": None,
                  "assignments": [assignment]}
        self._as_assignments.add(layer_path)
        self._replace_layer(layer_path, lambda old: old.model_copy(update=update))

    def _declared_dims(self, group: LayerGroup) -> Dict[str, str]:
        """The per-group dims the BASE declares for `group`."""
        layer_path, index = group.address
        if index is None:
            return {}
        declaration = self._layer(self.base.spec, layer_path).assignments[index]
        return {k: str(v) for k, v in (declaration.dims or {}).items()}

    @staticmethod
    def _checkpoint_walk(resolved: ResolvedSpec) -> Any:
        return Deployment(Program.of(resolved)).at(
            batch=1, input_tokens=2).checkpoint()

    @staticmethod
    def _boundary_names(program: Any, analysis: Any, block: Any) -> frozenset:
        """Names (`frame_names`) of what `block`'s entry nodes take in (the nodes
        reading one of its formal inputs, or the residual) and of what its
        output roots hand on, without the per-request axes."""
        formals = set(block.input_names())
        shapes = analysis.shapes.values
        extents: Dict[int, List[Any]] = {}
        for node in block.roots:
            if node.uid in shapes and (not node.inputs or formals & set(node.inputs)):
                for dims in shapes[node.uid].ins:
                    extents.setdefault(node.uid, []).extend(dims or ())
        for node in block.output_roots():
            extents.setdefault(node.uid, []).extend(shapes[node.uid].out or ())
        names = set().union(*(
            frame_names(analysis, uid, {name for extent in found
                                        for name in names_of(extent)})
            for uid, found in extents.items()))
        return frozenset(n for n in names if not is_deployment_symbol(n))

    def _template_param(self, block: Any, node_id: str) -> Optional[str]:
        """The template parameter a synthesized block bound into `node_id`'s
        op (`<attn>` → `attn`), or None when the op is written in the block."""
        if not block.synth_template:
            return None
        spec = self._spec
        template = ((spec.blocks or {}).get(block.synth_template)
                    or (spec.components or {}).get(block.synth_template)
                    or self.loader.library_component(block.synth_template))
        node = template.graph.get(node_id) if template is not None else None
        op = node.op if node is not None else ""
        return op[1:-1] if op.startswith("<") and op.endswith(">") else None

    def _rebind(self, block_name: str, parts: List[str], value: str) -> str:
        """A block like `block_name` with the reference at `parts` set to
        `value`; returns its name. `block_name` itself is left untouched.

        Inner blocks are rebound first and the new inner name is the value
        one level up. A slot that is a template parameter of a synthesized
        block is re-synthesized with that parameter changed (lowering's own
        synthesis: the authored name and graph, reused when it exists); a
        reference written into the block is a copy of the block with that
        one node changed, named `<block>__<node>-<value>`."""
        head, rest = parts[0], parts[1:]
        node_id, _, param = head.partition("::")
        block = self._spec.get_block(block_name)
        node = block.graph[node_id]
        if rest:
            value = self._rebind(node.op, rest, value)
        if not param:
            bound = self._template_param(block, node_id)
            if bound is not None:
                spec, name = self.loader.synthesize_block(
                    self._spec, block.synth_template,
                    {**block.synth_params, bound: value})
                if name != block.synth_template:
                    self._spec = spec
                    return name
            new_node = node.model_copy(update={"op": value})
            tag = f"{node_id}-{value}"
        else:
            new_node = node.model_copy(
                update={"params": {**node.params, param: value}})
            tag = f"{node_id}.{param}-{value}"
        return self._fork(block_name, block, node_id, new_node, tag)

    def _fork(self, block_name: str, block: Any, node_id: str, node: Any,
              tag: str) -> str:
        """Register a copy of `block` with `node_id` replaced, in the table
        `block_name` lives in (a component keeps its role). The copy is not
        the product of any synthesis, so it carries no `synth_*` record."""
        name = f"{block_name}__{tag}"
        spec = self._spec
        in_blocks = bool(spec.blocks) and block_name in spec.blocks
        table = dict((spec.blocks if in_blocks else spec.components) or {})
        if name not in table:
            table[name] = block.model_copy(update={
                "graph": {**block.graph, node_id: node},
                "synth_template": None, "synth_params": {},
            })
        if in_blocks:
            self._spec = spec.model_copy(update={"blocks": table})
        else:
            roles = dict(spec.component_roles or {})
            if block_name in roles:
                roles.setdefault(name, roles[block_name])
            self._spec = spec.model_copy(
                update={"components": table, "component_roles": roles})
        return name

    def _repoint(self, address: Tuple[Tuple[int, ...], Optional[int]],
                 template: str) -> None:
        """Point the one declaration at `address` to `template`. A
        synthesized template also rewrites the declaration's `params` to the
        binding it was synthesized from — lowering leaves an assignment's
        `params` in place, and the reachability walk and the cross-layer
        stream checks read them, so a stale binding would outlive the swap.
        A forked template keeps the declaration's `params` as they were."""
        update: Dict[str, Any] = {"template": template}
        block = self._spec.get_block(template)
        if block.synth_template:
            update["params"] = dict(block.synth_params)
        layer_path, index = address
        # A source declared for a stream the new template does not read
        # described the template it replaced.
        layer = self._layer(self._spec, layer_path)
        declaration = layer if index is None else layer.assignments[index]
        declared = declaration.reads_from
        if declared:
            params = update.get("params", declaration.params)
            read = {stream for _name, graph in self._spec.reachable_graphs(
                        [template, *self._spec._names_in_params(params)])
                    for stream in graph.reads or ()}
            kept = {k: v for k, v in declared.items() if k in read}
            if kept != declared:
                update["reads_from"] = kept or None
        self._edit(layer_path, index, update)

    def _edit(self, layer_path: Tuple[int, ...], index: Optional[int],
              update: Dict[str, Any]) -> None:
        """Copy-on-write update of one declaration in the edited spec."""
        def change(layer: Any) -> Any:
            if index is None:
                return layer.model_copy(update=update)
            assignments = list(layer.assignments)
            assignments[index] = assignments[index].model_copy(update=update)
            return layer.model_copy(update={"assignments": assignments})

        self._replace_layer(layer_path, change)

    def _replace_layer(self, layer_path: Tuple[int, ...], change: Any) -> None:
        """Copy-on-write: the layer at `layer_path` becomes `change(layer)`."""
        model = list(self._spec.model)
        head = layer_path[0]
        if len(layer_path) == 1:
            model[head] = change(model[head])
        else:
            group = model[head]
            subs = list(group.layers)
            subs[layer_path[1]] = change(subs[layer_path[1]])
            model[head] = group.model_copy(update={"layers": subs})
        self._spec = self._spec.model_copy(update={"model": model})
        self._resolved = None

    @staticmethod
    def _layer(spec: Any, layer_path: Tuple[int, ...]) -> Any:
        layer = spec.model[layer_path[0]]
        for j in layer_path[1:]:
            layer = layer.layers[j]
        return layer


_model_cache_memo: Dict[Tuple[Any, ...], Any] = {}


def model_cache_at(base: ResolvedSpec, *, model_id: str, component: str,
                   site: CallSite, dims: Dict[str, Any],
                   loader: Any = None, precision: Any = None) -> Any:
    """The model's own cache of `component`, read off a walk of the model
    with `dims` in force — the overrides an entry's isolated walk ran with
    (the shared ones and the entry's edits), set on the whole model the same
    way (`Composition.set_dims`). The `CacheView` narrowed to the
    cache-holding nodes that call `component`, each call at its own law and
    copies; every stack's cache that persists across decode steps (an MTP
    head's included, an encoder's not).

    A string instead is why there is no such cache to show: the entry takes
    the place of another component (`site.invoked` false), or the model
    does not evaluate with those dims."""
    if not site.invoked:
        return (f"{model_id} does not call {component}: it takes the place "
                f"of the component called at {site.label}, so there is no "
                f"whole model of it")
    key = (model_id, component, site.call, tuple(site.frame),
           tuple(sorted(dims.items())),
           None if precision is None else precision.key)
    if key in _model_cache_memo:
        return _model_cache_memo[key]
    composition = Composition(base, model_id=model_id, loader=loader)
    if dims:
        composition.set_dims(dims)
    try:
        analysis = Deployment(Program.of(composition.resolved()),
                              precision=precision).at(
            batch=1, input_tokens=1).analyze()

    except CONTRACT_ERRORS as exc:
        return f"the model does not evaluate with these dims: {exc}"
    program = analysis.program
    view = cache_view_from_analysis(
        analysis, cache_scope=program.included_stack_names(include_aux=True))

    # Only frame names that distinguish assignments. `CallSite.frame` records
    # written overrides, not a default the assignment specializes (`CSA2`'s
    # ratio-1 consumers have an empty frame but are a distinct parameterization
    # from ratio-2 consumers), so read the selected assignment's effective
    # frame — the one the walk opened for it (`Analysis.block_frame`) — and
    # ignore unrelated sibling rebinds such as an engram's `n_vocab`.
    distinct_names = sorted({
        str(name)
        for _meta, _roots in program.ir.blocks
        for name, _value in (_meta.distinct_dims or ())
    })
    def frame_at(block_index: int) -> Dict[str, str]:
        frame = analysis.block_frame(program.blocks[block_index])
        return {name: str(value_of(frame.get(name)))
                for name in distinct_names}

    wanted_frame = frame_at(site.invocation)

    def outer_component_call(node: Any) -> Optional[Any]:
        while node is not None:
            if binding_of(base.spec, node.op, component):
                return node
            node = node.parent
        return None

    def same_parameterization(entry: Any) -> bool:
        outer = outer_component_call(program.by_uid(entry.cache.component))
        if outer is None:
            return False
        if not 0 <= entry.block_index < len(program.ir.blocks):
            return False
        return frame_at(entry.block_index) == wanted_frame

    out = replace(view, entries=tuple(
        entry for entry in view.entries
        if same_parameterization(entry)))
    if len(_model_cache_memo) >= 64:
        _model_cache_memo.clear()
    _model_cache_memo[key] = out
    return out


def _slot_dto(slot: Slot) -> Dict[str, Any]:
    return {"slot_id": slot.path, "node_id": slot.node_id,
            "binding": slot.binding, "role": slot.role,
            "component": slot.component}


def _group_dto(composition: Composition, group: LayerGroup) -> Dict[str, Any]:
    """One layer group of the skeleton. `template_name` / `count` /
    `assignment_index` are echoed back by the editor with every edit, so the
    server can tell the base changed underneath it (`check_skeleton`)."""
    meta = group.meta
    out: Dict[str, Any] = {
        "group_id": group.group_id,
        "template_name": group.template,
        "count": int(meta.count),
        "assignment_index": meta.assignment_index,
        "stack": meta.stack_name,
        "aux_kind": meta.aux_kind,
        "editable": group.editable,
        "layers": list(group.layers()),
        "layers_label": group.layers_label(),
        "selector": group.selector(),
    }
    if not group.editable:
        return out
    slots = []
    for slot in group.slots():
        peers = composition.bound_with(slot)
        slots.append({
            **_slot_dto(slot),
            "kind": slot.kind,
            "label": group.label_of(slot),
            "candidates": slot.candidates(),
            # The calls a module swapped into this slot can be evaluated at.
            "calls": [call_dto(site)
                      for site in composition.calls_at(group, slot)],
            "bound_with": {
                "slots": [{"group_id": g.group_id, "slot_id": s.path}
                          for g, s in peers],
                "groups": len({g.group_id for g, _ in peers}),
                "layers": sum(int(g.meta.count) for g in
                              {g.group_id: g for g, _ in peers}.values()),
            },
        })
    out.update({
        "composition": group.composition(),
        "slots": slots,
        "readonly": [{"slot_id": s.path, "node_id": s.node_id, "role": s.role,
                      "component": s.component} for s in group.readonly()],
        "local_dims": composition.local_dims(group.group_id),
    })
    return out


def model_skeleton(model_id: str, overlay_dir: Optional[str] = None
                   ) -> Dict[str, Any]:
    """The editor skeleton: every layer group (`groups[]`, see `_group_dto`)
    and the backbone layer count."""
    resolved, header, loader = _base_resolved(model_id, overlay_dir)
    composition = Composition(resolved, model_id=model_id, header=header,
                              loader=loader)
    return {
        "model_id": model_id,
        # Stored as a config and a spec with no weights (a model described
        # in words): its costs are the spec's, unchecked against a file.
        "stored_without_checkpoint": stored_without_checkpoint(model_id),
        "n_layers": sum(g.meta.stored_count for g in composition.groups()
                        if g.meta.aux_kind is None
                        and (g.editable or g.looped)),
        "groups": [_group_dto(composition, g) for g in composition.groups()],
    }


def _totals(cost: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("params", "active_params", "flops",
            "weight_bytes", "kv_bytes_at_L",
            "phases", "kv_storage_by_cache_len")
    out = {k: cost[k] for k in keys}
    for key in ("timings", "timings_unavailable"):
        if key in cost:
            out[key] = cost[key]
    return out


@dataclass(frozen=True)
class PublicGroupRow:
    """The compare-row fields a composed group exposes. `_public_row`
    projects a cost row onto this declaration."""

    group_id: Any
    aux_kind: Any
    count: Any
    layers: Any
    layers_label: Any
    group_totals: Any
    kv_bytes_at_L: Any


def _public_row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {f.name: row[f.name] for f in fields(PublicGroupRow)}


def _group_rows(base: Dict[str, Any], edited: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Base and edited rows side by side, joined on the base `group_id`;
    `changed` when any number differs. A group the edit emptied has no
    edited row; a group the edit brought in (a selector that picks layers
    only after a layer-count change) has no base row."""
    facts = ("count", "total_params", "total_active_params", "total_flops",
             "attention_params", "attention_flops", "ffn_params", "ffn_flops",
             "quant_metadata_params", "total_storage_bytes", "kv_bytes_at_L")
    # A declaration may stand for several records (a `read: all` stack's
    # history lengths); a record is its group and what its layers see.
    def key(row: Dict[str, Any]) -> Tuple[Any, Tuple[Any, ...]]:
        return row["group_id"], tuple(sorted(row["visible"].items()))

    by_group = {key(row): row for row in edited["rows"]
                if row["group_id"] is not None}
    base_keys = {key(row) for row in base["rows"]}
    out = []
    for row in base["rows"]:
        mine = by_group.get(key(row))
        out.append({
            "group_id": row["group_id"], "aux_kind": row["aux_kind"],
            "base": _public_row(row), "edited": _public_row(mine),
            "changed": mine is None or any(row[k] != mine[k] for k in facts),
        })
    out.extend({"group_id": row["group_id"], "aux_kind": row["aux_kind"],
                "base": None, "edited": _public_row(row), "changed": True}
               for row in edited["rows"]
               if row["group_id"] is None or key(row) not in base_keys)
    return out


def compose_groups(*, base_model_id: str, groups: List[Dict[str, Any]],
                   dim_overrides: Optional[Dict[str, Any]] = None,
                   batch: int = 1, input_tokens: int = 2048,
                   precision: Any,
                   kv_cache_lens: Optional[List[int]] = None,
                   gpu_sku: Optional[str] = None,
                   overlay_dir: Optional[str] = None) -> Dict[str, Any]:
    """Whole-model cost of the base with per-group edits.

    `groups` holds one explicit entry per edited group: its `group_id`, the
    skeleton echo (`template_name`, `count`, `assignment_index`), `slots`
    (slot path → library component) and `dims` (the group's own
    rebindings). A swap "in every group bound to X" arrives already
    expanded into one entry per group — nothing spreads implicitly.
    `dim_overrides` are global.

    Anything that does not compose comes back as `errors[]`, each
    `{kind, group_id, slot?, detail}` naming what to fix: a refused edit
    (stale skeleton, unknown slot, role kind, interface dim, stream
    contract, layer span …), a dim the edited groups need and nothing binds
    (`kind: missing_dims`), or an evaluation that still fails
    (`kind: evaluation`)."""
    base, header, loader = _base_resolved(base_model_id, overlay_dir)
    composition = Composition(base, model_id=base_model_id, header=header,
                              loader=loader)
    dim_overrides = dict(dim_overrides or {})
    gpu_sku = (gpu_sku or "").strip() or None
    contract = {"batch": batch, "input_tokens": input_tokens,
                "precision": precision.to_json(), "groups": groups,
                "dim_overrides": dim_overrides,
                "kv_cache_lens": list(kv_cache_lens or []),
                "gpu_sku": gpu_sku}
    current = []
    for entry in groups:
        group_id = entry["group_id"]
        if not composition.check_skeleton(
                group_id, template_name=entry["template_name"],
                count=entry["count"],
                assignment_index=entry["assignment_index"]):
            continue
        current.append(entry)
        for slot, component in (entry.get("slots") or {}).items():
            composition.swap(group_id, slot, component)
    composition.set_dims(dim_overrides)
    for entry in current:
        if entry.get("dims"):
            composition.set_group_dims(entry["group_id"], entry["dims"])

    def failed(errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"success": False, "base_model_id": base_model_id,
                "contract": contract, "errors": errors}

    missing = composition.unbound_dims()
    if missing:
        return failed([{
            "kind": "missing_dims", **e,
            "detail": (f"{e['component'] or 'this group'} needs "
                       f"{', '.join(e['missing_dims'])}, which nothing binds "
                       f"— set them in the group's dims or globally"),
        } for e in missing])
    refusals = composition.refusals()
    if refusals:
        return failed(refusals)
    try:
        cost = composition.cost(batch=batch, input_tokens=input_tokens,
                                precision=precision, kv_cache_lens=kv_cache_lens,
                                gpu_sku=gpu_sku)
    except CONTRACT_ERRORS as exc:
        # Name the edit that led here: the engine's error speaks of the
        # tensor it could not size, not of the dim the user changed.
        edits = {**dim_overrides, **{
            k: v for entry in current for k, v in (entry.get("dims") or {}).items()}}
        edited = ", ".join(f"{k} = {v}" for k, v in edits.items())
        return failed([{"kind": "evaluation", "group_id": None,
                        "detail": (f"the edited model does not evaluate"
                                   f"{f' with {edited}' if edited else ''}: "
                                   f"{exc}")}])
    base_cost = Composition(base, model_id=base_model_id,
                            header=header).cost(
        batch=batch, input_tokens=input_tokens, precision=precision,
        kv_cache_lens=kv_cache_lens, gpu_sku=gpu_sku)
    return {
        "success": True,
        "base_model_id": base_model_id,
        "contract": contract,
        "composed": _totals(cost),
        "base": _totals(base_cost),
        "rows": _group_rows(base_cost, cost),
        "model_level": {
            side: {"totals": c["model_level_totals"]}
            for side, c in (("base", base_cost), ("edited", cost))},
        "groups": [{
            "group_id": entry["group_id"],
            "dim_changes": composition.group_dim_changes(entry["group_id"]),
            "local_dims": composition.local_dims(entry["group_id"]),
        } for entry in current],
        "errors": [],
    }
