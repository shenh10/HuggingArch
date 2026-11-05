"""Physical persistent artifacts derived from one ``Analysis``.

Weight placement, copy ownership and format geometry are joined here, but the
result stays a tuple of typed artifact rows rather than becoming a second
analysis container.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import prod
import operator
from typing import Any, Mapping, Optional, Tuple

import sympy as sp

from backend.arch.eval.refusal import SpecRefusal
from backend.arch.eval import formats as formats_mod
from backend.arch.eval import storage as storage_mod
from backend.arch.eval.capacity import Weights
from backend.arch.eval.dim import algebra, value_of
from backend.arch.eval.layer import Layer

from . import weight_endpoints
from .layout import TensorLayout


ArtifactKey = Tuple[int, int, int, str, Optional[int]]

#: Storage surfaces of one stored tensor: where an artifact lives — the tensor
#: itself, the metadata siblings its saved format declares, or those its
#: resident (deployment) format declares.
STORED = "stored"
SAVED_METADATA = "saved_metadata"
RESIDENT_METADATA = "resident_metadata"


@dataclass(frozen=True)
class PhysicalArtifact:
    """One rank-local persistent artifact with complete byte provenance."""

    node_uid: int
    tensor_index: int
    #: Which layer segment of that tensor (`formats.Representations`): 0 for
    #: every tensor whose node stores it the same way at every layer, which
    #: is all but the ones a layer-scoped `quant_context.overrides` row
    #: carves up. A segment is its own artifact because its format is its
    #: own — a 3-of-91-layer BF16 carve-out holds no scales where the other
    #: 88 layers hold one per block — and its instances are its share.
    segment_index: int
    surface: str
    metadata_index: Optional[int]
    artifact_name: str
    logical_name: str
    kind: storage_mod.StorageKind
    stack: str
    global_shape: Tuple[int, ...]
    runtime_shape: Tuple[int, ...]
    runtime_layout: Optional[TensorLayout]
    metadata_geometry: Optional[formats_mod.MetadataGeometry]
    global_instances: int
    runtime_instances: int
    saved_storage_dtype: Optional[str]
    resident_storage_dtype: Optional[str]
    saved_logical_dtype: Optional[str]
    resident_logical_dtype: Optional[str]
    saved_num_bits: Optional[int]
    resident_num_bits: Optional[int]
    saved_packing_factor: Optional[int]
    resident_packing_factor: Optional[int]
    saved_selected_by: Optional[str]
    resident_selected_by: Optional[str]
    saved_physical_slot_shape: Optional[Tuple[int, ...]]
    resident_physical_slot_shape: Optional[Tuple[int, ...]]
    saved_bytes_per_instance: int
    resident_bytes_per_instance: int
    saved_bytes_per_rank: int
    resident_bytes_per_rank: int
    #: One unsharded copy at the resident width. Cluster total is this
    #: times `global_instances` — not `resident_bytes_per_instance`, which
    #: is already TP-cut.
    resident_bytes_unsharded: int

    @property
    def key(self) -> ArtifactKey:
        return (
            self.node_uid,
            self.tensor_index,
            self.segment_index,
            self.surface,
            self.metadata_index,
        )

    @property
    def is_parameter(self) -> bool:
        return self.kind not in {
            storage_mod.StorageKind.BUFFER,
            storage_mod.StorageKind.METADATA,
        }


@dataclass(frozen=True)
class PhysicalWeightLeaf:
    """One weight-bearing node's complete rank-local resident footprint."""

    node_uid: int
    role: Any
    stack: str
    global_shape: Tuple[int, ...]
    runtime_shape: Tuple[int, ...]
    global_instances: int
    runtime_instances: int
    parameter_elements_per_rank: int
    resident_data_bytes_per_rank: int
    resident_metadata_bytes_per_rank: int

    @property
    def resident_bytes_per_rank(self) -> int:
        return (
            self.resident_data_bytes_per_rank
            + self.resident_metadata_bytes_per_rank
        )

def physical_artifacts(analysis: Any, *, stored_only: bool = False
                       ) -> Tuple[PhysicalArtifact, ...]:
    """Join persistent placement facts for one analysis.

    `stored_only` is for readers of weight-axis geometry alone: it avoids
    projecting scale metadata before a loader has been selected.
    """
    context = weight_endpoints.placement_context(analysis)
    weights = weight_endpoints.instantiate_analysis(analysis, context)
    _require_unique(
        ((row.node_uid, row.tensor_index) for row in weights),
        "weight endpoint",
    )
    return _PhysicalJoin(analysis, weights, stored_only=stored_only).run()



def physical_weight_leaves(
    analysis: Any,
    *,
    artifacts: Optional[Tuple[PhysicalArtifact, ...]] = None,
    layer_multiplicity: Optional[Layer[int]] = None,
) -> Mapping[int, PhysicalWeightLeaf]:
    """Index resident parameter artifacts by node uid."""
    rows = physical_artifacts(analysis) if artifacts is None else artifacts
    if layer_multiplicity is not None:
        rows = artifacts_for_layers(analysis, rows, layer_multiplicity)
    by_uid: dict[int, list[PhysicalArtifact]] = {}
    for row in rows:
        if (
            row.surface == STORED and row.is_parameter
        ) or row.surface == RESIDENT_METADATA:
            by_uid.setdefault(row.node_uid, []).append(row)

    out: dict[int, PhysicalWeightLeaf] = {}
    path_to_uid: dict[tuple, int] = {}
    holder: dict[tuple, str] = {}
    for uid, owned in by_uid.items():
        primaries = sorted(
            (
                row for row in owned
                if row.surface == STORED and row.is_parameter
            ),
            key=lambda row: row.tensor_index,
        )
        if not primaries:
            continue
        anchor = primaries[0]
        # Per TENSOR, summed over its layer segments: a node's stored tensors
        # all have the same instance ownership, and a tensor split across
        # layer segments owns its instances between them (85 + 3 of 88).
        owned_by_tensor: dict[int, Tuple[int, int]] = {}
        for row in primaries:
            held = owned_by_tensor.get(row.tensor_index, (0, 0))
            owned_by_tensor[row.tensor_index] = (
                held[0] + row.global_instances,
                held[1] + row.runtime_instances)
        if len(set(owned_by_tensor.values())) > 1:
            raise ValueError(
                f"weight node uid {uid} has inconsistent instance ownership"
            )
        global_instances, runtime_instances = owned_by_tensor[
            anchor.tensor_index]
        node = analysis.program.by_uid(uid)
        candidate = PhysicalWeightLeaf(
            node_uid=uid,
            role=node.role,
            stack=anchor.stack,
            global_shape=anchor.global_shape,
            runtime_shape=anchor.runtime_shape,
            global_instances=global_instances,
            runtime_instances=runtime_instances,
            parameter_elements_per_rank=sum(
                prod(row.runtime_shape) * row.runtime_instances
                for row in primaries
            ),
            resident_data_bytes_per_rank=sum(
                row.resident_bytes_per_rank for row in primaries
            ),
            resident_metadata_bytes_per_rank=sum(
                row.resident_bytes_per_rank
                for row in owned
                if row.surface == RESIDENT_METADATA
            ),
        )
        path_key = (anchor.stack, node.template, node.display_path)
        previous_uid = path_to_uid.get(path_key)
        if previous_uid is None:
            out[uid] = candidate
            path_to_uid[path_key] = uid
            holder[path_key] = analysis.program.where(uid)
            continue
        previous = out[previous_uid]
        live = [
            row for row in (previous, candidate)
            if row.global_instances > 0
        ]
        if len(live) > 1:
            raise SpecRefusal(
                f"duplicate live physical weight "
                f"{node.template!r} {node.display_path!r}: "
                f"{holder[path_key]} and {analysis.program.where(uid)} both "
                "hold it",
                declaration="template",
                fix=("a weight's id is its stack, block template and path, "
                     "so within one stack a block template is bound by one "
                     "model layer: declare the layers it covers in that "
                     "layer's `repeat:` or `per_layer.assign` (two "
                     "`per_layer.assign` entries of one template are told "
                     "apart only by a dim they bind differently)"),
                node=node.display_path)
        if live and live[0] is candidate:
            del out[previous_uid]
            out[uid] = candidate
            path_to_uid[path_key] = uid
            holder[path_key] = analysis.program.where(uid)
        else:
            out[previous_uid] = live[0] if live else previous
    return out

def resident_weights(artifacts: Tuple[PhysicalArtifact, ...]) -> Weights:
    """Resident parameter, metadata and buffer bytes for these artifacts."""
    parameters = 0
    metadata = 0
    buffers = 0
    by_stack: dict[str, int] = {}
    by_uid: dict[int, Weights] = {}
    for row in artifacts:
        amount = row.resident_bytes_per_rank
        if not amount:
            continue
        node_weights = by_uid.get(row.node_uid, Weights())
        if row.surface == RESIDENT_METADATA:
            metadata += amount
            node_weights = node_weights.add(metadata=amount)
        elif row.surface == STORED:
            dtype = (row.resident_logical_dtype
                     or row.resident_storage_dtype or "").upper()
            if row.is_parameter:
                parameters += amount
                node_weights = node_weights.add(parameters=amount, dtype=dtype)
            else:
                buffers += amount
                node_weights = node_weights.add(buffers=amount, dtype=dtype)
        else:
            continue
        by_uid[row.node_uid] = node_weights
        by_stack[row.stack] = by_stack.get(row.stack, 0) + amount
    return Weights(
        parameters=parameters,
        metadata=metadata,
        buffers=buffers,
        by_stack=by_stack,
        by_uid=by_uid,
    )


def whole_resident(
    artifacts: Tuple[PhysicalArtifact, ...],
    program: Any = None,
    include_aux: bool = True,
) -> float:
    """Whole-model resident bytes — every copy, not one rank.

    `resident_bytes_per_instance` is the rank-local copy (TP already
    cut the shape). Cluster total uses the global shape at the same
    resident width, times `global_instances`. Same surfaces as
    `resident_weights`; unknown stacks raise the way `without_aux` does.
    """
    included = None
    if program is not None:
        declared = program.included_stack_names(include_aux=True)
        unknown = {row.stack for row in artifacts} - declared
        if unknown:
            raise ValueError(
                f"weight ledger names undeclared stack(s) {sorted(unknown)!r}"
            )
        if not include_aux:
            included = program.included_stack_names()
    total = 0.0
    for row in artifacts:
        if included is not None and row.stack not in included:
            continue
        if row.surface not in (
            STORED, RESIDENT_METADATA,
        ):
            continue
        total += row.resident_bytes_unsharded * row.global_instances
    return total


def active_parameters_by_stack(
    analysis: Any,
    artifacts: Tuple[PhysicalArtifact, ...],
) -> dict[str, Any]:
    """Logical parameter elements activated locally, grouped by stack."""
    by_owner: dict[tuple[str, int], int] = {}
    for row in artifacts:
        if row.surface != STORED or not row.is_parameter:
            continue
        elements = prod(row.runtime_shape) * row.runtime_instances
        key = (row.stack, row.node_uid)
        by_owner[key] = by_owner.get(key, 0) + elements

    totals: dict[str, Any] = {}
    for (stack, uid), elements in by_owner.items():
        cost = analysis.cost.values.get(uid)
        if cost is None:
            raise ValueError(f"parameter owner uid {uid} has no cost record")
        params = algebra(value_of(cost.params))
        active = algebra(value_of(cost.active_params))
        if params.free_symbols or active.free_symbols:
            raise ValueError(
                f"parameter owner uid {uid} has ungrounded active count"
            )
        if params == 0:
            if active != 0:
                raise ValueError(
                    f"parameter owner uid {uid} activates {active} of zero "
                    "parameters"
                )
            continue
        totals[stack] = (
            totals.get(stack, sp.Integer(0))
            + sp.Integer(elements) * active / params
        )
    for stack, total in tuple(totals.items()):
        if total.free_symbols:
            raise ValueError(
                f"local active parameter total for stack {stack!r} is ungrounded"
            )
        totals[stack] = int(total) if total.is_integer is True else total
    return totals


def active_parameter_elements(
    analysis: Any,
    artifacts: Tuple[PhysicalArtifact, ...],
    *,
    include_aux: bool = False,
) -> Any:
    """Logical active parameters for the main step, optionally with aux."""
    by_stack = active_parameters_by_stack(analysis, artifacts)
    return sum(by_stack.values(), 0) if include_aux else by_stack.get("", 0)


def artifacts_for_layers(
    analysis: Any,
    artifacts: Tuple[PhysicalArtifact, ...],
    layer_multiplicity: Layer[int],
) -> Tuple[PhysicalArtifact, ...]:
    """Re-count artifacts from an already-resolved layer selection."""
    if layer_multiplicity.program is not analysis.program:
        raise ValueError("artifact layer multiplicity belongs to another Program")
    runtime_by_uid: dict[int, int] = {}
    for row in artifacts:
        previous = runtime_by_uid.setdefault(row.node_uid, row.runtime_instances)
        if previous != row.runtime_instances:
            raise ValueError(
                f"artifact node uid {row.node_uid} has runtime instance counts "
                f"{previous} and {row.runtime_instances}"
            )
    selected_instances = analysis.recount(
        Layer(analysis.program, runtime_by_uid), layer_multiplicity
    )
    projected = []
    for row in artifacts:
        runtime_instances = selected_instances.values[row.node_uid]
        if runtime_instances == row.runtime_instances:
            projected.append(row)
            continue
        projected.append(replace(
            row,
            runtime_instances=runtime_instances,
            saved_bytes_per_rank=row.saved_bytes_per_instance * runtime_instances,
            resident_bytes_per_rank=(
                row.resident_bytes_per_instance * runtime_instances
            ),
        ))
    return tuple(projected)


def _artifact(
    *,
    uid: int,
    tensor_index: int,
    segment_index: int,
    surface: str,
    metadata_index: Optional[int],
    logical: Any,
    runtime_shape: Tuple[int, ...],
    runtime_layout: Optional[TensorLayout],
    geometry: Optional[formats_mod.MetadataGeometry],
    rep: formats_mod.Representation,
    stack: str,
    global_instances: int,
    runtime_instances: int,
) -> PhysicalArtifact:
    saved_format = rep.saved if surface == STORED else None
    resident_format = rep.resident if surface == STORED else None
    saved_dtype = saved_format.storage_dtype if saved_format is not None else None
    resident_dtype = (
        resident_format.storage_dtype if resident_format is not None else None
    )
    saved_logical_dtype = (
        saved_format.logical_dtype if saved_format is not None else None
    )
    resident_logical_dtype = (
        resident_format.logical_dtype if resident_format is not None else None
    )
    saved_bits = saved_format.num_bits if saved_format is not None else None
    resident_bits = resident_format.num_bits if resident_format is not None else None
    saved_packing = (
        saved_format.packing_factor if saved_format is not None else None
    )
    resident_packing = (
        resident_format.packing_factor if resident_format is not None else None
    )
    saved_selected = saved_format.selected_by if saved_format is not None else None
    resident_selected = (
        resident_format.selected_by if resident_format is not None else None
    )

    if surface == SAVED_METADATA:
        saved_dtype = logical.dtype
        saved_logical_dtype = logical.dtype
        saved_bits = _dtype_bits(logical.dtype, logical.name)
        saved_packing = 1
        saved_selected = rep.saved.selected_by
    elif surface == RESIDENT_METADATA:
        resident_dtype = logical.dtype
        resident_logical_dtype = logical.dtype
        resident_bits = _dtype_bits(logical.dtype, logical.name)
        resident_packing = 1
        resident_selected = rep.resident.selected_by

    elements = prod(runtime_shape)
    global_shape = _shape(logical.shape, logical.name)
    saved_per_instance, saved_slots = _surface_bytes(
        logical.name,
        runtime_shape,
        elements,
        saved_dtype,
        saved_packing,
    )
    resident_per_instance, resident_slots = _surface_bytes(
        logical.name,
        runtime_shape,
        elements,
        resident_dtype,
        resident_packing,
    )
    return PhysicalArtifact(
        node_uid=uid,
        tensor_index=tensor_index,
        segment_index=segment_index,
        surface=surface,
        metadata_index=metadata_index,
        artifact_name=logical.name,
        logical_name=logical.logical_name,
        kind=logical.kind,
        stack=stack,
        global_shape=global_shape,
        runtime_shape=runtime_shape,
        runtime_layout=runtime_layout,
        metadata_geometry=geometry,
        global_instances=global_instances,
        runtime_instances=runtime_instances,
        saved_storage_dtype=saved_dtype,
        resident_storage_dtype=resident_dtype,
        saved_logical_dtype=saved_logical_dtype,
        resident_logical_dtype=resident_logical_dtype,
        saved_num_bits=saved_bits,
        resident_num_bits=resident_bits,
        saved_packing_factor=saved_packing,
        resident_packing_factor=resident_packing,
        saved_selected_by=saved_selected,
        resident_selected_by=resident_selected,
        saved_physical_slot_shape=saved_slots,
        resident_physical_slot_shape=resident_slots,
        saved_bytes_per_instance=saved_per_instance,
        resident_bytes_per_instance=resident_per_instance,
        saved_bytes_per_rank=saved_per_instance * runtime_instances,
        resident_bytes_per_rank=resident_per_instance * runtime_instances,
        resident_bytes_unsharded=_surface_bytes(
            logical.name,
            global_shape,
            prod(global_shape),
            resident_dtype,
            1 if resident_packing is None else resident_packing,
        )[0],
    )


def _surface_bytes(
    name: str,
    runtime_shape: Tuple[int, ...],
    elements: int,
    dtype: Optional[str],
    packing: Optional[int],
) -> Tuple[int, Optional[Tuple[int, ...]]]:
    if dtype is None:
        return 0, None
    if packing is None or packing <= 0:
        raise ValueError(f"{name}: invalid packing factor {packing!r}")
    if elements % packing:
        raise ValueError(
            f"{name}: {elements} logical elements do not fill "
            f"{packing}-element storage slots"
        )
    slots = elements // packing
    width = formats_mod.known_dtype_bytes(dtype)
    amount = _integer(
        sp.Integer(slots) * sp.Rational(str(width)),
        f"{name} storage bytes",
    )
    physical_shape = runtime_shape if packing == 1 else None
    return amount, physical_shape


def _dtype_bits(dtype: Optional[str], name: str) -> int:
    if dtype is None:
        raise ValueError(f"{name}: metadata tensor has no storage dtype")
    return _integer(
        sp.Rational(str(formats_mod.known_dtype_bytes(dtype))) * 8,
        f"{name} storage bit width",
    )


def _shape(extents: Any, name: str) -> Tuple[int, ...]:
    return tuple(
        _positive(value_of(extent), f"{name} axis {index}")
        for index, extent in enumerate(extents)
    )


def share_of(instances: int, share: Fraction, label: str) -> int:
    """`instances` times a layer segment's share of them.

    A node's layer count is a factor of its copies, so the product is whole;
    a remainder would mean the segment's layers are not that node's, and
    says so rather than rounding."""
    if share == 1:
        return instances
    scaled = instances * share
    if scaled.denominator != 1:
        raise ValueError(
            f"{label}: a layer segment covering {share} of {instances} "
            f"copies is {scaled}, not a whole number of copies")
    return int(scaled)


def _count(value: Any, label: str) -> int:
    integer = _integer(value_of(value), label)
    if integer < 0:
        raise ValueError(f"{label} must be non-negative, got {integer}")
    return integer


def _positive(value: Any, label: str) -> int:
    integer = _integer(value, label)
    if integer <= 0:
        raise ValueError(f"{label} must be positive, got {integer}")
    return integer


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    expression = algebra(value)
    if expression.free_symbols or expression.is_integer is not True:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    try:
        return int(operator.index(expression))
    except TypeError:
        return int(expression)


def _require_unique(keys: Any, label: str) -> None:
    seen = set()
    duplicates = set()
    for key in keys:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate {label} keys {sorted(duplicates)!r}")




class _PhysicalJoin:
    """One storage ⟂ weight-endpoint join: the index, the consumed set, and
    the artifacts list, so the per-uid / per-tensor / per-surface stages stop
    re-passing them."""

    def __init__(self, analysis, weights, *, stored_only: bool = False):
        self.analysis = analysis
        self.stored_only = stored_only
        self.weight_by_key = {
            (row.node_uid, row.tensor_index): row for row in weights
        }
        self.consumed_weights: set[Tuple[int, int]] = set()
        self.artifacts = []

    def run(self) -> Tuple[PhysicalArtifact, ...]:
        for uid in sorted(self.analysis.storage.values):
            self._node_artifacts(uid)
        missing = set(self.weight_by_key) - self.consumed_weights
        if missing:
            raise ValueError(
                f"weight endpoints do not join storage keys "
                f"{sorted(missing)!r}"
            )
        _require_unique((artifact.key for artifact in self.artifacts),
                        "local artifact")
        return tuple(self.artifacts)

    def _node_artifacts(self, uid) -> None:
        analysis = self.analysis
        node = analysis.program.by_uid(uid)
        if node is None:
            raise ValueError(f"storage layer names unknown node uid {uid}")
        tensors = tuple(analysis.storage.values[uid])
        reps = tuple(analysis.formats.values.get(uid) or ())
        if len(tensors) != len(reps):
            raise ValueError(
                f"{node.display_path}: {len(tensors)} stored tensors but "
                f"{len(reps)} format representations"
            )
        instances = _count(
            analysis.instances.values.get(uid, 1),
            f"{node.display_path} instances",
        )
        stack = analysis.program.stack_of(node)
        for tensor_index, (tensor, reps_of_tensor) in enumerate(
            zip(tensors, reps, strict=True)
        ):
            # One artifact set per layer segment. The placement is the
            # tensor's — every layer of a node holds the same shape and is
            # cut the same way — so the segments share one weight endpoint
            # and differ only in format and in the copies they cover.
            for segment_index, (rep, share) in enumerate(
                reps_of_tensor.with_shares()
            ):
                self._tensor_artifacts(
                    uid, tensor_index, segment_index, tensor, rep, stack,
                    share_of(instances, share,
                             f"{node.display_path} instances"))

    def _tensor_artifacts(self, uid, tensor_index, segment_index, tensor, rep,
                          stack, instances) -> None:
        weight_key = (uid, tensor_index)
        weight = self.weight_by_key.get(weight_key)
        if weight is None:
            raise ValueError(
                f"{tensor.name}: stored tensor has no weight-axis endpoint"
            )
        self.consumed_weights.add(weight_key)
        global_stored_shape = _shape(tensor.shape, tensor.name)
        if weight.global_shape != global_stored_shape:
            raise ValueError(
                f"{tensor.name}: weight endpoint global shape "
                f"{weight.global_shape!r} != {global_stored_shape!r}"
            )
        runtime_stored = replace(tensor, shape=weight.runtime_shape)
        local_symbols = dict(weight.runtime_symbol_extents)
        # A refused load may be projected only for the planner's diagnostic
        # rows; ordinary analyze and capacity reject it before reading them.
        metadata_surfaces = not self.stored_only
        saved_layout = rep.saved.block_scale_layout
        saved_tp = (self.analysis.rule.world_degree(saved_layout.world)
                    if saved_layout is not None else 1)
        # Checkpoint grids belong to complete chunks. A head-gather loader
        # can requantize a finer runtime shard, but that does not give the
        # source scales a contiguous rank-local grid. Keep their global
        # checkpoint representation and omit that undefined physical surface.
        projects_saved = metadata_surfaces and (
            saved_layout is None or saved_layout.checkpoint_tp % saved_tp == 0)
        projected_saved = rep.saved.project_metadata(
            runtime_stored,
            local_symbols=local_symbols,
            runtime_tp=saved_tp,
        ) if rep.saved.metadata and projects_saved else ()
        projected_resident = rep.resident.project_metadata(
            runtime_stored,
            local_symbols=local_symbols,
            runtime_tp=(self.analysis.rule.world_degree(rep.resident.block_scale_layout.world)
                        if rep.resident.block_scale_layout is not None else 1),
        ) if rep.resident.metadata and metadata_surfaces else ()

        surfaces = [(STORED, None, tensor, runtime_stored, None)]
        surfaces.extend(
            (SAVED_METADATA, index, metadata, local,
             rep.saved.metadata_geometry[index])
            for index, (metadata, local) in enumerate(
                zip(rep.saved.metadata if projects_saved else (),
                    projected_saved, strict=True)
            )
        )
        surfaces.extend(
            (RESIDENT_METADATA, index, metadata, local,
             rep.resident.metadata_geometry[index])
            for index, (metadata, local) in enumerate(
                zip(rep.resident.metadata if metadata_surfaces else (),
                    projected_resident, strict=True)
            )
        )
        for surface, metadata_index, logical, local, geometry in surfaces:
            self.artifacts.append(_artifact(
                uid=uid,
                tensor_index=tensor_index,
                segment_index=segment_index,
                surface=surface,
                metadata_index=metadata_index,
                logical=logical,
                runtime_shape=_shape(local.shape, local.name),
                runtime_layout=(weight.runtime_layout
                                if surface == STORED else None),
                geometry=geometry,
                rep=rep,
                stack=stack,
                global_instances=instances,
                runtime_instances=instances,
            ))
