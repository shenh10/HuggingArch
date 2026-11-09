"""`Formats` — how a tensor is stored, as a sideband layer.

Why this is not derived by the walk
-----------------------------------
Shapes, cost and formulas propagate along edges: a node's value follows from
what feeds it. A weight's dtype does not. It follows from the node's ROLE and
the model's `quant_context`, and a tensor fed by a bf16 activation is stored in
FP8 all the same. So `Formats` is aligned to the graph by uid and computed by
lookup — the structural reading of a fact that sits beside the compute graph
rather than propagated through it.

Sideband is not the same as after-the-fact, and the difference is load-bearing:
this layer is consulted DURING the walk, because a weight's metadata footprint
is part of its cost and a component's cost is combined from its children as the
walk descends. Applying it to a finished result reaches no total. `Measured` is
the layer that genuinely applies afterwards, because it overwrites nothing.

That also settles what happens under fusion: a fused QKV still reads three
weight tensors, each with its own dtype and its own bytes on disk, so a rewrite
of the COMPUTE graph does not touch this layer and it needs no `rebase`.

The one join
------------
    Analysis.tensors()  ->  (uid, StoredTensor, Representation, instances)

Deliberately one place. Every consumer that re-derives this join by name
instead is a fork of the same fact — the failure mode is a read path assuming
a uniform FP8 while the storage path uses the native BF16 for the same tensor,
with nothing loud in between.

Per TENSOR, not per node, and the checkpoint is why: a `linear` with a bias puts
two entries on disk, a scheme quantises the weight and leaves the bias at the
native dtype, and the header lists them separately. So this layer holds a TUPLE
per node, index-parallel with the storage layer's.

Two widths, not one
-------------------
`Representation` pairs what the checkpoint SAVED with what the deployment
RUNS AT. The first is declared by `quant_context` and confirmed by the
safetensors header — an equality `checkpoint.py` checks — and the second is a
choice nothing outside the deployment can know. They coincide whenever a model
is served as it was saved, which is most of the time, and diverge whenever it is
not: HF's own `FineGrainedFP8Config` carries a `dequantize` flag.

Metadata is a property of the format, not a node
------------------------------------------------
An FP8 weight implies `(in/block_h) * (out/block_w)` F32 scales; an MXFP4 weight
implies `(in*out)/group_size` U8 block scales. Whether those appear as explicit
graph nodes is an accident of how the checkpoint names things — DeepSeek ships
`*.weight_scale_inv` and gpt-oss ships `*_scales`, both of which the spec then
has to mention, while a scheme whose scales are implicit ships nothing. Either
way the footprint is the same and it is a consequence of the weight's format.

So `quant_metadata_params` is a PROJECTION of this layer, not a second
accounting. `Analysis.iter_checkpoint` flattens implied siblings for
the matcher; nothing else re-derives them.

Two different exclusions, not one
---------------------------------
`_is_quant_meta` also returns True for `rotary_inv_freq_buffer`. That tensor is
not quantisation metadata: it is a precomputed cache that some checkpoints
serialise and others do not, and it is not a parameter for an entirely
different reason. Merging the two is why adding a quant scheme means editing an
engine method.

Here the split is structural: whether a node is quant metadata is asked of the
SCHEMES (they declare their own suffixes), and whether it is a non-learnable
buffer is asked of the OP.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import (Any, Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from .. import roles as roles_mod
from ..dtypes import UnknownDtype
from ..dtypes import dtype_bytes as _dtype_bytes
from .dim import exact_div
from .program import Stack
from .refusal import SpecRefusal
from .scheme_dialect import (eval_formula, eval_predicate, formula_symbols,
                             logical_dtype_label, select_numeric_format)
from .storage import StorageKind
from .values import Extent, product


def known_dtype_bytes(dtype: str) -> float:
    """`arch.dtypes.dtype_bytes` for the callers that price a deployment width
    into every byte downstream: an unknown name is a `ValueError` here, never a
    width. The `None`-returning variant that used to sit beside this one is
    gone; a caller that can tolerate an unknown header dtype catches
    `UnknownDtype` itself and says what it does with it."""
    try:
        return float(_dtype_bytes(dtype))
    except UnknownDtype:
        raise ValueError(f"unknown dtype name {dtype!r} — declare it in "
                         "backend.arch.dtypes or state the width in bytes "
                         "directly") from None


def quant_granularity(quant_context: Any) -> Optional[str]:
    """How this model's weights are blocked, read from `quant_context`.

    Returns `per-tile` / `per-group` / `per-channel` / `per-tensor`, or
    None when nothing is quantized. The most specific entry wins.
    """
    if quant_context is None or not quant_context.per_role:
        return None
    rank = {"per-tile": 4, "per-group": 3, "per-channel": 2, "per-tensor": 1}
    best = "per-tensor"
    found = False
    for entry in quant_context.per_role.values():
        if entry.method in (None, "none"):
            continue
        found = True
        fields = entry.model_dump()
        if fields.get("block_size") or (
            fields.get("block_h") and fields.get("block_w")
        ):
            cand = "per-tile"
        elif fields.get("group_size"):
            cand = "per-group"
        elif (fields.get("strategy") or "").lower() == "channel":
            cand = "per-channel"
        elif (fields.get("strategy") or "").lower() == "group":
            cand = "per-group"
        else:
            cand = "per-channel"
        if rank.get(cand, 0) > rank.get(best, 0):
            best = cand
    return best if found else None


@dataclass(frozen=True)
class MetadataGeometry:
    """The format-declared shape transform for one metadata sibling.

    ``StoredTensor.shape`` is one resolved point. A deployment projection
    needs the declaration that produced it so the same transform can be
    evaluated against the rank-local stored tensor. A scalar cardinality is
    retained for audit, but is deliberately not promoted to a physical shape:
    it says how many slots exist, not which stored axis owns them.
    """

    suffix: str
    axes: Optional[Tuple[str, ...]] = None
    cardinality: Optional[str] = None
    #: How ``in``/``out`` are obtained from the stored payload. ``matrix``
    #: reads the last two axes and keeps any axes before them leading;
    #: ``flatten_leading`` folds a typed HEADWISE bank into checkpoint rows
    #: before the format transform. None means the declaration does not
    #: consume matrix axes.
    stored_transform: Optional[str] = None


#: Every value `TensorFormat.weight_granularity` can take — what a measured
#: row's `weight_granularity` slot is checked against.
WEIGHT_GRANULARITY = re.compile(r"per-tile:\d+x\d+|per-group:\d+|per-channel|per-tensor")

#: The load-time scale-grid transforms eval knows how to materialize. Framework
#: bindings select one; this is the closed vocabulary their loader rejects
#: unknown names against, not a framework registry.
LOADER_TRANSFORMS = frozenset({"whole_chunks", "head_gather"})


def scale_strategy_of(scheme: Any, config: Mapping[str, Any]) -> Optional[str]:
    """The scale strategy a weight's scales follow: the per-role `strategy:`
    when the scheme offers several, else the scheme's only one."""
    return config.get("strategy") or (
        scheme.scale_strategy[0] if scheme.scale_strategy else None)


@dataclass(frozen=True)
class TensorFormat:
    """How one weight-bearing node is stored.

    `packing_factor` is logical elements per stored slot — 2 for MXFP4's FP4
    pairs in a U8, 1 for everything unpacked. Keeping it separate from
    `storage_dtype` is what lets bytes be `params / packing * width` without a
    per-scheme branch.
    """

    storage_dtype: str = "BF16"
    num_bits: int = 16
    packing_factor: int = 1
    metadata: Tuple[Any, ...] = ()   # `StoredTensor`s of kind METADATA
    # Which rule produced this: `override:<pattern>`, `role:<role>`, `default`.
    # Provenance is not decoration — a wrong dtype on one tensor family is
    # otherwise indistinguishable from a wrong dtype everywhere.
    selected_by: str = "default"
    #: The deployment's precision profile chose this format (a weight rule
    #: of `PrecisionProfile`), rather than the checkpoint's own declaration
    #: (`quant_context`). The typed half of `selected_by`'s provenance:
    #: that string is for a reader, and nothing branches on its text.
    chosen_by_profile: bool = False
    # The `FormatVariant` that selected this format, held rather than copied
    # from.
    #
    # `weight_storage_suffix` is already a declared field on it — `.weight` by
    # default, `.weight_packed` for the packed variants — and the variant's own
    # docstring says it "fixes its own packing_factor + storage_dtype + weight
    # tensor suffix". Copying that string onto this class would make a second
    # place the same fact lives, and the two would agree until a variant added a
    # rule the copy did not carry.
    #
    # `storage_dtype` and `packing_factor` ARE fields here and that is not the
    # same thing: those are RESOLVED from the variant against `num_bits`, so
    # what is stored is a computation's answer rather than a declaration's text.
    variant: Any = None
    #: The `QuantScheme` that selected this format, and the per-role bindings it
    #: was resolved against. Held for the same reason `variant` is: `method` and
    #: the logical dtype LABEL are questions the scheme answers — the second
    #: through `numeric_format_resolver`, which is a list of predicates over
    #: exactly these bindings — and copying either out would be a third and
    #: fourth place a declaration lives.
    scheme: Any = None
    bindings: Mapping[str, Any] = field(default_factory=dict)
    #: Shape declarations aligned one-to-one with ``metadata``. Kept on the
    #: resolved format so storage projection does not re-select a quant policy
    #: or replay an evaluator at another deployment.
    metadata_geometry: Tuple[MetadataGeometry, ...] = ()
    #: The activation format a deployment chose to run this weight's matmul
    #: on (`activation_options` of its scheme, `backend/arch/
    #: activation_formats/`), None when the kernel takes its input edge as
    #: it arrives — every checkpoint format, and W4A16.
    activation_format: Optional[str] = None
    block_scale_layout: Any = None
    #: The framework-selected load transform, resolved outside eval and passed
    #: in as plain data. None means the checkpoint format is used unchanged.
    loader: Optional[str] = None
    #: Why the selected loader cannot materialize this rank. Refused formats
    #: publish no rank-local scale surface; estimate entry points reject them.
    load_refusal: Optional[str] = None

    @property
    def method(self) -> Optional[str]:
        """The quant method that covers this tensor, `None` when none does.

        `validate._packing_consistency` asks exactly this and phrases it
        `mt.method is not None`.
        """
        return self.scheme.name if self.scheme else None

    @property
    def logical_dtype(self) -> str:
        """The element type as the ARCHITECTURE sees it, where the slot type is
        what the checkpoint packs it into: `FP4` in a `U8`, `INT4` in an `I32`.

        Equal to `storage_dtype` for everything unpacked, which is most of a
        model. Resolved by the scheme's own two functions rather than a table
        here — they read `numeric_format_resolver` against these bindings, and
        a second implementation of that walk is a second answer.
        """
        if self.scheme is None:
            return self.storage_dtype
        return logical_dtype_label(
            select_numeric_format(self.scheme, dict(self.bindings)),
            int(self.num_bits), self.storage_dtype)

    @property
    def weight_granularity(self) -> Optional[str]:
        """Which weight elements share one scale: `per-tile:<h>x<w>`,
        `per-group:<g>` (along the reduction axis), `per-channel`,
        `per-tensor` — the vocabulary `quant_granularity` speaks, with the
        block size it leaves out. None when the tensor stores no scale.

        A kernel is written for one of these. A 128x128-block FP8 checkpoint
        runs on a block-scaled GEMM; a per-channel one cannot be read by it, so
        the two are different measurements of the same dtype. Read off the
        scheme's own strategy and the bindings its scale shapes resolved with.
        """
        if self.scheme is None or not any(
                m.sibling_kind == "scale" for m in self.metadata):
            return None
        strategy = scale_strategy_of(self.scheme, self.bindings)
        b = self.bindings
        if strategy == "tile_2d":
            return f"per-tile:{int(b['block_h'])}x{int(b['block_w'])}"
        if strategy == "group":
            return f"per-group:{int(b['group_size'])}"
        if strategy in ("channel", "tensor"):
            return f"per-{strategy}"
        raise ValueError(
            f"{self.scheme.name}: scale strategy {strategy!r} has no "
            f"granularity name")

    @property
    def weight_suffix(self) -> str:
        """What the checkpoint calls the weight under this format.

        The SECOND authority on the name, and it does not replace the first: an
        op says whether its weight is `.weight` or a naked parameter, which is a
        fact about the module, while a scheme says whether the bytes on disk are
        that weight or a packed encoding of it. Reading only the op's named
        eleven models' expert weights `...w1.weight` where their checkpoints
        hold `...w1.weight_packed`.
        """
        return str((self.variant.weight_storage_suffix
                    if self.variant is not None else "") or ".weight")

    @property
    def metadata_elements(self) -> Extent:
        total: Extent = 0
        for m in self.metadata:
            total = total + m.elements
        return total

    @property
    def metadata_bytes(self) -> Extent:
        total: Extent = 0
        for m in self.metadata:
            total = total + m.bytes()
        return total

    def weight_bytes_for(self, params: Extent) -> Extent:
        """The tensor's OWN slots, without the metadata that comes with them.

        Separate from `bytes_for` because the two are reported separately
        everywhere they are reported: a checkpoint's `model_storage_bytes` and
        its `quantization_metadata_bytes` are two numbers, and so are `Traffic`'s
        `weight_reads` and `metadata_reads`. Having only the total meant every
        caller that wanted one of them computed the other and subtracted.
        """
        # An unknown storage dtype raises; returning 0 here is how a
        # 49 GB block of scales once weighed nothing.
        width = known_dtype_bytes(self.storage_dtype)
        # `exact_div`, not `/`: the packing factor is how many logical
        # elements share one stored element, so the physical count divides
        # exactly — `copy_checkpoint` already refuses a tensor whose logical
        # elements are not a multiple of its saved packing factor, by name.
        # A bare `/` made the value half a float while the expression half
        # stayed exact (measured: 1229 of these on one corpus pass), so a
        # packed weight's bytes were a number of one kind in the numeric walk
        # and another in the symbolic one.
        return (exact_div(params, self.packing_factor) * width
                if self.packing_factor > 1 else params * width)

    def bytes_for(self, params: Extent) -> Extent:
        """Storage for `params` logical elements, metadata included."""
        return self.weight_bytes_for(params) + self.metadata_bytes

    def for_loader(self, stored: Any, local_stored: Any, *,
                   loader: Optional[str], runtime_tp: int) -> "TensorFormat":
        """The resident format after a declared loader merges FP8 chunks.

        A matching degree loads checkpoint chunks unchanged. Other supported
        degrees merge complete chunks or gather heads, then requantize each
        rank's full fused matrix with the scheme's ordinary block grid.
        """
        contract = self.block_scale_layout
        if contract is None or not loader:
            return self
        selected = replace(self, loader=loader)
        if runtime_tp == contract.checkpoint_tp:
            return selected
        if loader == "whole_chunks" and contract.checkpoint_tp % runtime_tp:
            return replace(selected, load_refusal=(
                f"{stored.name}: the fused weight checkpoint is "
                f"TP={contract.checkpoint_tp}-interleaved; the {loader} "
                f"loader requires attention TP to divide that degree, got "
                f"TP={runtime_tp}"))
        bound = {**self.bindings, **local_stored.as_dims()}
        # Format metadata is the whole deployment's resident storage. The
        # physical projection below evaluates the same geometry at one rank.
        metadata, geometries = [], []
        siblings = {s.suffix: s for s in self.scheme.metadata_siblings}
        for old in self.metadata:
            sibling = siblings[old.logical_suffix]
            axes = sibling.shape or sibling.shape_per_strategy[
                scale_strategy_of(self.scheme, self.bindings)]
            shape = tuple(_cardinality(axis, bound) for axis in axes)
            metadata.append(replace(old, shape=(shape[0] * runtime_tp, *shape[1:])))
            geometries.append(MetadataGeometry(
                suffix=sibling.suffix, axes=tuple(axes), cardinality=None,
                stored_transform=_metadata_stored_transform(stored, axes)))
        return replace(selected, metadata=tuple(metadata),
                       metadata_geometry=tuple(geometries))

    def project_metadata(
        self,
        stored: Any,
        *,
        local_symbols: Optional[Mapping[str, Any]] = None,
        runtime_tp: int = 1,
    ) -> Tuple[Any, ...]:
        """Instantiate metadata beside a projected stored tensor.

        The stored tensor supplies authoritative ``in``/``out``. Additional
        symbols used by an explicitly authored shape must come from the
        persistent-axis projector that owns the stored tensor. Arbitrary node
        bindings are not accepted: they are global evaluator values and would
        silently restore a sharded head or segment count. Cardinality-only
        declarations refuse because flattening their count would invent a
        physical axis order.
        """
        if len(self.metadata_geometry) != len(self.metadata):
            raise ValueError(
                f"{stored.name}: format has {len(self.metadata)} metadata "
                f"tensors but {len(self.metadata_geometry)} geometry declarations"
            )
        projected = dict(local_symbols or {})
        uses_chunks = any(
            geometry.axes is not None
            and any("chunks_per_rank" in axis for axis in geometry.axes)
            for geometry in self.metadata_geometry)
        if uses_chunks:
            if self.block_scale_layout is None:
                raise ValueError(
                    f"{stored.name}: scale geometry uses chunks_per_rank "
                    "without a block_scale_layout")
            checkpoint_tp = self.block_scale_layout.checkpoint_tp
            # A loader-free checkpoint walk is analytical, not a deployability
            # claim. A selected loader that cannot load publishes load_refusal;
            # estimate and capacity reject that analysis before reading it.
            projected["chunks_per_rank"] = (
                checkpoint_tp // runtime_tp
                if checkpoint_tp % runtime_tp == 0 else checkpoint_tp)
        conflicting = sorted(
            name
            for name in set(projected).intersection(self.bindings)
            if projected[name] != self.bindings[name]
        )
        if conflicting:
            raise ValueError(
                f"{stored.name}: local persistent-axis symbols {conflicting!r} "
                "conflict with format configuration"
            )
        bound = {
            **projected,
            **self.bindings,
            **stored.as_dims(),
            "num_bits": self.num_bits,
        }
        out = []
        for metadata, geometry in zip(
            self.metadata, self.metadata_geometry, strict=True
        ):
            if metadata.logical_suffix != geometry.suffix:
                raise ValueError(
                    f"{metadata.name}: metadata geometry suffix "
                    f"{geometry.suffix!r} does not match {metadata.logical_suffix!r}"
                )
            if geometry.axes is None:
                raise ValueError(
                    f"{metadata.name}: metadata declaration provides only "
                    "cardinality, not a physical shape"
                )
            dependencies = frozenset().union(
                *(formula_symbols(axis) for axis in geometry.axes)
            )
            if dependencies.intersection({"in", "out"}):
                if geometry.stored_transform == "matrix":
                    if not stored.kind.stores_out_in or len(stored.shape) < 2:
                        raise ValueError(
                            f"{metadata.name}: matrix metadata transform does not "
                            f"match rank-{len(stored.shape)} {stored.kind.value} "
                            "stored payload"
                        )
                elif geometry.stored_transform == "flatten_leading":
                    if stored.kind is not StorageKind.HEADWISE or len(stored.shape) < 2:
                        raise ValueError(
                            f"{metadata.name}: flatten-leading metadata transform "
                            f"does not match rank-{len(stored.shape)} "
                            f"{stored.kind.value} stored payload"
                        )
                else:
                    raise ValueError(
                        f"{metadata.name}: matrix metadata formula "
                        f"{geometry.axes!r} has no declared leading-axis transform "
                        f"for rank-{len(stored.shape)} stored payload"
                    )
            authoritative = (
                set(self.bindings)
                | set(projected)
                | set(stored.as_dims())
                | {"num_bits"}
            )
            missing = sorted(dependencies - authoritative)
            if missing:
                raise ValueError(
                    f"{metadata.name}: metadata shape symbols {missing!r} have "
                    "no local persistent-axis provenance"
                )
            try:
                shape = tuple(_cardinality(axis, bound) for axis in geometry.axes)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{metadata.name}: cannot project metadata shape "
                    f"{geometry.axes!r} from the local stored tensor"
                ) from exc
            shape = (*_leading(stored), *shape)
            out.append(stored.sibling(
                metadata.logical_suffix,
                shape,
                StorageKind.METADATA,
                metadata.dtype,
                sibling_kind=metadata.sibling_kind,
            ))
        return tuple(out)


NATIVE = TensorFormat()


@dataclass(frozen=True)
class Representation:
    """How one tensor exists in the two places it exists.

    `saved` is the checkpoint. `quant_context` declares it and the safetensors
    header states it, and those two are an EQUALITY to be checked, not two
    numbers to be reconciled — see `checkpoint.py`. Everything about the model
    as an artefact reads this one: disk bytes, parameter totals, whether the
    spec describes the file it claims to.

    `intent` is what the deployment runs at, and `None` means "as saved", which
    is the common case and the right default. The two genuinely differ, in both
    directions, and HF's own configs carry the switch: `FineGrainedFP8Config`
    takes a `dequantize` flag, so a checkpoint saved in FP8 may be served in
    BF16, and a BF16 checkpoint quantised at load is the mirror image.

    Why this is not one dtype with a second opinion
    -----------------------------------------------
    Saved and intent are two QUESTIONS, not two answers to one: `saved` has
    exactly one source, the checkpoint, and `intent` is the precision
    profile's weight table (`precision.WeightIntent`) — sparse, so a tensor
    no rule selects has no intent and runs as saved.

    Both slots are full `TensorFormat`s rather than bare dtype names, because a
    deployment that quantises does not merely narrow the width: it produces
    scales, which are resident bytes the weight did not have on disk. A scalar
    intent dtype cannot express those, and would under-count exactly the
    deployment that motivated having an intent at all.
    """

    saved: TensorFormat
    intent: Optional[TensorFormat] = None
    #: The layers of the node this pairing covers, when its declarations do
    #: not give every layer the same saved format; None — almost always —
    #: when they do, and the pairing covers every layer the node runs at.
    #: Not a second copy of the layer schedule: it is the set a
    #: layer-scoped `quant_context.overrides` row carved out, and the only
    #: thing that makes two pairings of one tensor two different facts.
    layers: Optional[Tuple[int, ...]] = None

    @property
    def resident(self) -> TensorFormat:
        """What is actually in HBM, which is what traffic and capacity read."""
        return self.saved if self.intent is None else self.intent

    def as_saved(self) -> "Representation":
        """This representation as the FILE has it: the saved format, and no
        intent to resolve past it. What a deployment runs the tensor at is
        not a fact about the checkpoint, so the projections that answer for
        the file (`Analysis.checkpoint_tensors`) ask for this."""
        return self if self.intent is None else replace(self, intent=None)

    @property
    def is_converted(self) -> bool:
        return (self.intent is not None
                and self.intent.storage_dtype != self.saved.storage_dtype)

    # Element counts, for the accounting that predates bytes.
    @property
    def metadata_elements(self) -> Extent:
        return self.saved.metadata_elements


@dataclass(frozen=True)
class Representations:
    """Every `Representation` one stored tensor takes across its node's
    layers — one, for almost every tensor there is.

    The plural of the existing concept rather than a concept of its own: the
    walk's `formats` layer stays index-parallel with its `storage` layer (one
    entry per stored tensor, which is what lets a bias not pair with its
    weight's format), and this is that entry.

    More than one segment only where a layer-scoped `quant_context.overrides`
    row gives some of a node's layers another saved format. `saved` /
    `intent` / `resident` then have no single answer, and asking for one
    RAISES rather than returning the majority: a reader that prices one
    format for a tensor the checkpoint stores in two is wrong by the
    minority, and silently. The readers that are right about it either take
    the rows (`Analysis.tensors()` — one row per segment, each with that
    segment's copies) or ask for all of them (`formats`).
    """

    segments: Tuple[Representation, ...]

    @classmethod
    def of(cls, *segments: Representation) -> "Representations":
        if not segments:
            raise ValueError("a stored tensor has at least one representation")
        return cls(tuple(segments))

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)

    @property
    def one(self) -> Representation:
        """The single representation, where there is one."""
        if len(self.segments) != 1:
            raise ValueError(
                f"this tensor is stored at {len(self.segments)} formats across "
                f"its node's layers ("
                + "; ".join(
                    f"{segment.saved.storage_dtype} at layers "
                    f"{list(segment.layers or ())}"
                    for segment in self.segments)
                + "), so it has no single representation. Read the rows "
                  "(`Analysis.tensors()` gives one per layer segment, with "
                  "that segment's copies) or every format (`formats`).")
        return self.segments[0]

    def as_saved(self) -> "Representations":
        """These representations as the FILE has them
        (`Representation.as_saved`), segment for segment: the same tensor
        answered for the checkpoint instead of for the deployment."""
        return Representations.of(
            *(segment.as_saved() for segment in self.segments))

    @property
    def saved(self) -> TensorFormat:
        return self.one.saved

    @property
    def intent(self) -> Optional[TensorFormat]:
        return self.one.intent

    @property
    def resident(self) -> TensorFormat:
        return self.one.resident

    @property
    def is_converted(self) -> bool:
        return any(segment.is_converted for segment in self.segments)

    @property
    def formats(self) -> Tuple[TensorFormat, ...]:
        """Every resident format this tensor is held at, over all its
        layers — for the readers that want them all rather than one."""
        return tuple(segment.resident for segment in self.segments)

    @property
    def weight_suffix(self) -> str:
        """What the checkpoint calls this weight.

        One answer for the tensor, not one per segment: a segment is a set of
        the node's layers storing the same tensor another way, and a NAME is
        not a way of storing it. Two segments whose formats disagree on the
        suffix would be two different checkpoint entries at one address, so
        that is refused rather than resolved — nothing declares which of the
        two a layer's file tensor is called."""
        suffixes = {segment.saved.weight_suffix for segment in self.segments}
        if len(suffixes) > 1:
            raise ValueError(
                f"this tensor's layer segments disagree on what the "
                f"checkpoint calls it ({sorted(suffixes)}). A layer-scoped "
                f"override may give some layers another dtype or scheme; it "
                f"cannot give them another tensor name.")
        return suffixes.pop()

    def with_shares(self) -> Tuple[Tuple[Representation, Fraction], ...]:
        """Each representation and the share of the node's copies it covers.

        The one place that share is worked out. A node's layer count is a
        factor of its copies (the walk seeds the frame with
        `template.count`), so a segment covering k of them covers k/total of
        the copies — `Analysis.tensors()` bills the rows with it and the
        per-rank projection re-applies it to its own count.
        """
        if len(self.segments) == 1:
            return ((self.segments[0], Fraction(1)),)
        total = sum(len(segment.layers or ()) for segment in self.segments)
        if not total:
            raise ValueError("layer-segmented representations cover no layer")
        return tuple((segment, Fraction(len(segment.layers or ()), total))
                     for segment in self.segments)

    @property
    def metadata_elements(self) -> Extent:
        """The metadata elements of one copy, weighted by the share of the
        node's copies each segment covers: a 3-of-91-layer BF16 carve-out
        implies no scales for those three."""
        if len(self.segments) == 1:
            return self.segments[0].metadata_elements
        total = sum(len(segment.layers or ()) for segment in self.segments)
        if not total:
            raise ValueError(
                "layer-segmented representations cover no layer")
        out: Extent = 0
        for segment in self.segments:
            out = out + (segment.metadata_elements
                         * len(segment.layers or ()) / total)
        return out


@dataclass(frozen=True)
class FormatPolicy:
    """`quant_context` as a lookup, with the precedence stated once.

    An override names tensors literally or by regex and wins over everything —
    it exists for the tensors a role rule gets wrong. Otherwise the node's role
    picks a scheme. Otherwise the context's default dtype.

    Resolution is by NODE, never by edge, which is what makes this layer
    sideband; and the same policy answers for a node the graph has not been
    walked yet, which is what lets a weight breakdown be produced without an
    analysis.
    """

    default_dtype: str = "BF16"
    per_role: Mapping[str, Any] = field(default_factory=dict)
    overrides: Sequence[Any] = ()
    # Slot key → the role the component filling it declares. `per_role` is
    # written about roles; a leaf carries the SLOT it sits in, and for a hybrid
    # model those differ. See `Program.role_aliases`.
    role_aliases: Mapping[str, str] = field(default_factory=dict)
    #: One entry per override row that scopes itself to layers
    #: (`select.layers`): the layer indices it speaks about, resolved once
    #: against the model's dims and config by `ResolvedSpec.layers_selected`.
    #: Resolved by the caller because a dim expression and a config list are
    #: the model's facts, not this lookup's. A row with a layer scope and no
    #: entry here is a caller that did not resolve it, and says so.
    layer_scopes: Mapping[int, FrozenSet[int]] = field(default_factory=dict)
    # One frozenset per override row, parallel with `overrides`. Built once
    # so a row naming many tensors is not a linear scan per node. `None` means that row has no literal list (regex, or empty).
    _literal_sets: Tuple[Optional[FrozenSet[str]], ...] = field(
        init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_literal_sets",
            tuple(_override_literals(entry) for entry in self.overrides),
        )
        unresolved = sorted(
            index for index, entry in enumerate(self.overrides)
            if _layer_selection(entry) is not None
            and index not in self.layer_scopes)
        if unresolved:
            raise ValueError(
                f"quant_context.overrides{unresolved} scope themselves to "
                f"layers, and this policy was built without resolving them. "
                f"A layer bound is a dim expression and a config list is the "
                f"model's; build the policy through `Formats.of(program)` (or "
                f"pass `layer_scopes`) so the layers are the model's own.")

    @property
    def is_layer_scoped(self) -> bool:
        """Whether any row speaks about layers. When none does, no
        declaration mentions a layer, so every layer of a node resolves
        identically and the walk asks once — the same answer, not a
        shortcut past one."""
        return bool(self.layer_scopes)

    @classmethod
    def of(cls, quant_context: Any,
           native: Optional[str] = None,
           role_aliases: Optional[Mapping[str, str]] = None,
           layer_scopes: Optional[Mapping[int, FrozenSet[int]]] = None
           ) -> "FormatPolicy":
        """Two declarations, in order of specificity: what the spec says about
        this model's quantisation, then the precision the model ships its
        unquantised tensors at (`ResolvedSpec.shipped_dtype`). With neither
        there is no default to price a tensor at, and none is assumed — a
        literal here is what put gemma-2 out by a factor of two."""
        aliases = dict(role_aliases or {})
        default = (quant_context.declared_default_dtype
                   if quant_context is not None else None) or native
        if default is None:
            raise SpecRefusal(
                "the model states no precision for its unquantised tensors: "
                "no checkpoint header to read, no config `torch_dtype`, no "
                "`quant_context.default_dtype`",
                declaration="quant_context.default_dtype",
                fix="declare `quant_context.default_dtype` in the spec (the "
                    "dtype the checkpoint stores its unquantised weights "
                    "at), or give the config its `torch_dtype`")
        if quant_context is None:
            return cls(default_dtype=default, role_aliases=aliases)
        return cls(
            default_dtype=default,
            per_role=dict(quant_context.per_role),
            overrides=tuple(quant_context.overrides),
            role_aliases=aliases,
            layer_scopes=dict(layer_scopes or {}),
        )

    @property
    def is_native(self) -> bool:
        return not self.per_role and not self.overrides

    # -- selection -------------------------------------------------------

    def unused_overrides(
            self, tensors: Iterable[Tuple[str, Optional[str], Tuple[str, ...],
                                          Sequence[Optional[int]]]]
    ) -> Tuple[int, ...]:
        """Rows of `overrides` that decide the format of none of `tensors`
        (`(checkpoint name, stack, enclosing components, layers)` tuples).

        Such a row is dead: its pattern names no tensor the spec stores, or an
        entirely earlier row always wins first. It still reads like a live
        rule, so a reader believes a dtype the model never gets from it.

        Asked at the tensor's LAYERS, the same grouping the formats come from
        (`rows_by_layer`): a row that scopes itself to layers wins only at
        those, so asking without a layer would call every layer-scoped row
        dead.
        """
        won = set()
        for name, stack, components, layers in tensors:
            indices = [layer for layer in layers if layer is not None]
            groups = self.rows_by_layer(name, stack, components,
                                        indices or [None])
            won.update(index for index in groups if index is not None)
        return tuple(i for i in range(len(self.overrides)) if i not in won)

    def undescribed_overrides(
            self, tensors: Iterable[Tuple[str, Optional[str], Tuple[str, ...],
                                          TensorFormat]]
    ) -> Tuple[Tuple[int, str], ...]:
        """`(row, tensor name)` for every tensor an override row selects with a
        quant `method:` whose scheme does not describe it.

        `tensors` are `(checkpoint name, stack, enclosing components, saved
        format)`. A scheme describes a tensor with stored `(out, in)` axes — a
        matrix, a per-head bank, a lookup table's rows — and resolves its
        metadata from them. Anything else (a vector, a buffer) is stored at
        the default dtype with no metadata, so a row naming a method for it
        claims a format the tensor does not get. A `per_role` entry is not
        asked: it covers a role's weight tensors, and the gains and biases
        sharing that role are outside it by design.
        """
        found = []
        for name, stack, components, fmt in tensors:
            hit = self._override_row(name, stack, components)
            if hit is None:
                continue
            index, entry, _ = hit
            if entry.method not in (None, "none") and fmt.scheme is None:
                found.append((index, name))
        return tuple(found)

    def _override_row(self, tensor_name: str, stack: Optional[str] = None,
                      components: Tuple[str, ...] = (),
                      layer: Optional[int] = None,
                      ) -> Optional[Tuple[int, Any, str]]:
        """The first override row that selects this tensor AT `layer`: its
        index, the entry, and the provenance string `selected_by` records.

        Per layer, because a row may scope itself to layers
        (`OverrideSelector.layers`) and a node's layers then take different
        rows. `layer` None asks the question the way it was always asked —
        of a node with no layer index (a model-level op), where a
        layer-scoped row selects nothing.

        Raises when a `component`-scoped authored row and another authored
        row both select the tensor: which one decides must not rest on
        declaration order."""
        hits = []
        for index, entry in enumerate(self.overrides):
            selector = entry.select
            # A scoped selector speaks about logical identity before
            # inspecting the checkpoint name. Aux and backbone tensors, and
            # two components of one stack, may deliberately share that name.
            if selector is not None and not selector.scopes(stack, components):
                continue
            scoped_layers = self.layer_scopes.get(index)
            if scoped_layers is not None and layer not in scoped_layers:
                continue
            literals = self._literal_sets[index]
            if literals is not None and tensor_name in literals:
                hits.append((index, entry, f"override:{tensor_name}"))
                continue
            pattern = (selector.tensors_re
                       if selector is not None
                       else entry.tensors_re)
            if pattern and re.search(pattern, tensor_name):
                hits.append((index, entry, f"override:{pattern}"))
        if not hits:
            return None
        authored = hits
        scoped = [hit for hit in authored
                  if hit[1].select is not None and hit[1].select.component]
        if scoped and len(authored) > 1:
            first, second = scoped[0], next(
                hit for hit in authored if hit is not scoped[0])
            pair = sorted((first, second), key=lambda hit: hit[0])
            raise ValueError(
                f"quant_context.overrides[{pair[0][0]}] "
                f"({pair[0][1].selector_text()}) and "
                f"overrides[{pair[1][0]}] "
                f"({pair[1][1].selector_text()}) both select "
                f"{tensor_name!r} (stack {Stack.display_name(stack)!r}, inside "
                f"{list(components)}). A component-scoped override must be "
                f"the only row that selects its tensors: narrow the other "
                f"row so the two select disjoint tensors.")
        return hits[0]

    def rows_by_layer(
            self, tensor_name: str, stack: Optional[str],
            components: Tuple[str, ...], layers: Sequence[int],
    ) -> Dict[Optional[int], List[int]]:
        """`{winning override row (None for none) → the layers it wins at}`
        for one stored tensor.

        The one place a tensor's layers are grouped by the row that decides
        them: `layer_segments` asks each group's format once, and
        `unused_overrides` asks which rows win anywhere. Asking per layer and
        grouping after would let the two disagree about which row is live.
        """
        groups: Dict[Optional[int], List[int]] = {}
        for layer in layers:
            hit = self._override_row(tensor_name, stack, components, layer)
            groups.setdefault(None if hit is None else hit[0], []).append(layer)
        return groups

    def layer_segments(self, tensor_name: str, role: Optional[str],
                       stored: Optional[Any] = None,
                       node_dims: Optional[Mapping[str, Any]] = None,
                       stack: Optional[str] = None,
                       components: Tuple[str, ...] = (),
                       layers: Sequence[Optional[int]] = (None,),
                       ) -> Tuple[Tuple[Tuple[int, ...], TensorFormat], ...]:
        """`(layers, format)` for each set of `layers` the declarations give
        this tensor one format, in layer order.

        One entry — covering every layer, with no layer tuple to carry —
        unless a row scopes itself to layers. `for_node` is still the one
        resolution: this groups the layers by the override row that wins at
        each (the only thing a layer can change) and asks it once per group,
        so a 91-layer node costs two resolutions and not 91.
        """
        indices = tuple(layer for layer in layers if layer is not None)
        if not self.is_layer_scoped or not indices:
            return ((indices,
                     self.for_node(tensor_name, role, stored, node_dims,
                                   stack, components)),)
        ordered = sorted(
            self.rows_by_layer(tensor_name, stack, components, indices)
            .values(), key=min)
        return tuple(
            (tuple(group),
             self.for_node(tensor_name, role, stored, node_dims,
                           stack, components, layer=group[0]))
            for group in ordered)

    def for_node(self, tensor_name: str, role: Optional[str],
                 stored: Optional[Any] = None,
                 node_dims: Optional[Mapping[str, Any]] = None,
                 stack: Optional[str] = None,
                 components: Tuple[str, ...] = (),
                 layer: Optional[int] = None,
                 ) -> TensorFormat:
        """The format for one stored tensor, at one layer of its node.

        Takes the `StoredTensor`, not a dims mapping, and that is the point. A
        metadata cardinality formula is written in `in` and `out`, and those
        have to be the WEIGHT's two dims — so they come from the tensor, where
        they are the weight's by construction. Handed a loose dims dict this
        used to resolve against whatever the surrounding scope had bound under
        those names, which is another tensor entirely for a fused expert op, and
        it resolved without complaint: gpt-oss came out 8600x under.

        A tensor with no matrix dims — a norm's gain, a bias — is not what any
        scheme's formula describes, and takes the default dtype.
        """
        dims = stored.as_dims() if stored is not None else {}
        hit = self._override_row(tensor_name, stack, components, layer)
        node_dims = node_dims or {}
        if hit is not None:
            _, entry, why = hit
            method = entry.method
            dtype = entry.dtype or self.default_dtype
            # `method: none` is unquantized here as in `per_role` below.
            if not method or method == "none":
                return TensorFormat(storage_dtype=dtype, num_bits=_bits(dtype),
                                    selected_by=why)
            return self._from_scheme(method, entry, dims, why,
                                     dtype_hint=dtype,
                                     stored=stored,
                                     node_dims=node_dims,
                                     shape_override=entry.scale_shape)

        # The slot first, then the role that slot's component declares, then
        # the roles each refines (`roles.selection_order`). A spec that names
        # a slot `attention_swa` and fills it with a component declaring
        # `role: attention` means both, and `per_role.attention` is written
        # about the second; an `attention_output` leaf takes
        # `per_role.attention` unless the spec names `attention_output`.
        entry = None
        for candidate in roles_mod.selection_order(role, self.role_aliases):
            entry = self.per_role.get(candidate)
            if entry is not None:
                role = candidate
                break
        if entry is None:
            return TensorFormat(storage_dtype=self.default_dtype,
                                num_bits=_bits(self.default_dtype),
                                selected_by="default")
        method = entry.method
        if not method or method == "none":
            dtype = entry.dtype or self.default_dtype
            return TensorFormat(storage_dtype=dtype, num_bits=_bits(dtype),
                                selected_by=f"role:{role}")
        return self._from_scheme(method, entry, dims, f"role:{role}",
                                 stored=stored,
                                 node_dims=node_dims,
                                 shape_override=entry.scale_shape)

    # -- the scheme registry does the describing ---------------------------

    def _from_scheme(self, method: str, entry: Any, dims: Mapping[str, Any],
                     why: str,
                     dtype_hint: Optional[str] = None,
                     stored: Optional[Any] = None,
                     node_dims: Optional[Mapping[str, Any]] = None,
                     shape_override: Optional[Sequence[str]] = None
                     ) -> TensorFormat:
        """Everything here is read from the scheme YAML — packing, storage
        dtype, metadata suffixes and their cardinality formulas. Adding a
        scheme is a file, not a branch.

        Every sibling the scheme declares is implied here, once. A spec cannot
        carry one as a node of its own: `guards._check_quant_suffix_in_arch_spec`
        refuses a quant-suffixed `parameter_vector` at load, so there is no
        second representation to count twice. Should a spec ever be allowed to
        spell a sibling out, the node will declare whose sibling it is and that
        declaration will retire the implied one."""
        scheme = _scheme(method)
        if scheme is None:
            raise ValueError(
                f"{why}: quant method {method!r} is not a registered scheme "
                f"(backend/arch/quant_schemes/); a weight in an unknown "
                f"format has no width to price it at")

        config = _entry_fields(entry, scheme)
        bits = config.get("num_bits")
        if bits is None:
            bits = _scheme_bits(scheme, why)
        bits = int(bits)
        variant = _variant(scheme, config.get("format"))
        storage = _variant_field(variant, "storage_dtype", bits) \
            or dtype_hint or self.default_dtype
        packing = int(_variant_field(variant, "packing_factor", bits) or 1)

        # The node's own dims sit UNDER the stored tensor's, never over them.
        # A sibling shape may need `n_kv_heads` or `d_head` — symbols no weight
        # carries — while `in` and `out` must stay the weight's, which is what
        # the `in * out == params` check below rests on. Layering them this way
        # widens the vocabulary without reopening the hijack: a node dim named
        # `in` cannot displace the stored tensor's.
        bound = {**(node_dims or {}), **config, **dims, "num_bits": bits}
        if entry.block_scale_layout is not None:
            # The unsharded checkpoint is one rank owning every chunk. Rank
            # projection rebinds this symbol to that rank's complete chunks.
            bound["chunks_per_rank"] = entry.block_scale_layout.checkpoint_tp
        # A cardinality formula is written about the WEIGHT — `(in/block_h) *
        # (out/block_w)`, `(in*out)/group_size` — so `in` and `out` have to be
        # the weight's own two dims. Whatever is bound under those names in the
        # surrounding scope need not be: a fused expert op names its dims
        # something else entirely (gpt-oss's `mxfp4_fused_linear` uses
        # `n_indexer_heads` and `Out`), and the surrounding `in`/`out` then
        # belong to some other tensor.
        #
        # `in * out == params` is the check, and it is exact rather than
        # heuristic: the weight has that many elements, so any pair of dims
        # claiming to be its shape must multiply to it. Without it the formula
        # resolves happily against the wrong numbers, which is worse than not
        # resolving — gpt-oss came out 8600x under and nothing said so.
        # A tensor with no matrix dims is not what the formula describes — a
        # norm's gain, a bias vector — and is stored at the default dtype. That
        # is the only failure left: with `in`/`out` coming from the stored
        # tensor they cannot belong to a different tensor, so the mismatch this
        # used to have to check for is no longer expressible.
        if not dims:
            return TensorFormat(
                storage_dtype=self.default_dtype,
                num_bits=_bits(self.default_dtype),
                selected_by=f"{why}(not-a-matrix)")
        strategy = scale_strategy_of(scheme, config)
        metadata = []
        metadata_geometry = []
        for sib in scheme.metadata_siblings:
            # A declared SHAPE wins over a scalar count, and the count is then
            # its product — a projection, not a second declaration. Both forms
            # are read because the schemes are being migrated one at a time and
            # a half-migrated registry must not silently drop a sibling.
            axes = sib.shape
            if axes is None and sib.shape_per_strategy:
                axes = sib.shape_per_strategy.get(strategy)
            if shape_override and "scale" in sib.suffix:
                # This checkpoint lays its scales out differently from the
                # scheme's default. Scoped to the scale sibling by name because
                # that is what the field is called; a zero-point or a shape
                # sidecar is a different tensor with its own rule.
                axes = list(shape_override)
            formula = sib.cardinality
            if formula is None and sib.cardinality_per_strategy:
                formula = sib.cardinality_per_strategy.get(strategy)
            if not axes and not formula:
                continue
            if sib.when and not _predicate(sib.when, bound):
                continue
            dtype = sib.dtype
            if dtype is None and sib.dtype_per_format:
                dtype = sib.dtype_per_format.get(
                    variant.name if variant is not None else "default")
            try:
                resolved = (tuple(_cardinality(a, bound) for a in axes)
                            if axes else None)
                count = (_shape_elements(resolved) if resolved is not None
                         else _cardinality(formula, bound))
            except (KeyError, ValueError, TypeError):
                # The formula names a symbol this node has no binding for —
                # `in` / `out` on an rmsnorm gain or a bias vector. That is not
                # a lookup failure, it is the scheme saying this node is not
                # the weight matrix it describes. Per-role selection is keyed
                # by role alone and cannot see the difference; the scheme's own
                # symbol requirements can.
                return TensorFormat(
                    storage_dtype=self.default_dtype,
                    num_bits=_bits(self.default_dtype),
                    selected_by=f"{why}(not-a-matrix)")
            if stored is None:
                # No tensor, so no module to hang a sibling on. Reached only by
                # a caller that asked for a format without one; naming the
                # sibling off a bare string is what produced two different
                # wrong answers before.
                continue
            shape = tuple(resolved) if resolved else (count,)
            if resolved:
                # A formula tiles ONE matrix — the stored tensor's last two
                # axes (`as_dims`) — and every matrix carries its own grid,
                # so the axes before them lead the sibling as they lead the
                # weight.
                shape = (*_leading(stored), *shape)
            else:
                count = count * _product(_leading(stored))
                shape = (count,)
            metadata.append(stored.sibling(
                sib.suffix, shape,
                StorageKind.METADATA, dtype or "F32", sib.logical_dtype,
                sibling_kind=sib.kind))
            metadata_geometry.append(MetadataGeometry(
                suffix=sib.suffix,
                axes=(tuple(str(axis) for axis in axes) if axes else None),
                cardinality=(str(formula) if formula is not None else None),
                stored_transform=_metadata_stored_transform(stored, axes),
            ))
        return TensorFormat(storage_dtype=storage, num_bits=bits,
                            packing_factor=packing,
                            metadata=tuple(metadata), selected_by=why,
                            variant=variant, scheme=scheme, bindings=config,
                            metadata_geometry=tuple(metadata_geometry),
                            block_scale_layout=entry.block_scale_layout,
                            # What this scheme's own kernel does to the
                            # activation operand (`activation:`). A
                            # deployment's weight rule overrides it with the
                            # option it chose (`WeightIntent.for_node`).
                            activation_format=scheme.activation_format(bound))


# -- the resolver the walk consults ---------------------------------------

@dataclass(frozen=True)
class Formats:
    """Both policies, and the one place their answers are paired.

    A resolver, in the same sense as the op registry: the walk asks it a
    question about a tensor and it answers by lookup. It is not an annotation —
    a weight's metadata footprint is part of its cost, and cost is combined from
    children DURING the walk, so an after-the-fact layer never reaches a total.

    `intent` is the deployment's weight table (`precision.WeightIntent`):
    per tensor, the format a precision profile runs it at, or None — run it
    as saved. Sparse, so a mixed-precision checkpoint keeps every tensor no
    rule selects exactly as it was saved.
    """

    saved: FormatPolicy = field(default_factory=FormatPolicy)
    intent: Optional[Any] = None

    @classmethod
    def of(cls, program: Any, intent: Optional[Any] = None) -> "Formats":
        """Takes the PROGRAM, not the spec, because the saved side needs both of
        the checkpoint's own statements: the spec's `quant_context` for the
        tensors a scheme covers, and `config.torch_dtype` for the ones it does
        not.

        A `Program` and nothing else. This used to accept either through
        `getattr(program, "spec", program)`, which no caller ever used and which
        would have silently taken the `native_dtype` branch to None for a spec —
        landing on the literal BF16 that put gemma-2 out by a factor of two.

        A component isolated out of a model carries that model's call site
        (`ResolvedSpec.site`), and its saved side is that checkpoint's policy
        — the model the tensors come from, not the spec that hosts the walk.
        A site that states no formats keeps the host's.
        """
        site = program.resolved.site
        if site is not None and site.formats is not None:
            return cls(saved=site.formats, intent=intent)
        return cls(
            saved=FormatPolicy.of(
                program.spec.quant_context,
                program.resolved.shipped_dtype()[0],
                role_aliases=program.role_aliases,
                layer_scopes=layer_scopes_of(program)),
            intent=intent)

    def resolve(self, stored: Any, role: Optional[str],
                node_dims: Optional[Mapping[str, Any]] = None,
                stack: Optional[str] = None,
                components: Tuple[str, ...] = (),
                layers: Sequence[Optional[int]] = (None,),
                ) -> Representations:
        """Every `Representation` this tensor takes across `layers` — one,
        unless a layer-scoped override row gives some of them another saved
        format.

        The intent side is asked once per segment too, and asked about the
        same tensor AND the segment's own saved format: a precision profile
        addresses a role and a stack, so every segment gets the same answer
        from the rule tables — but the serving downcast is a statement about
        the dtype this segment is stored at, which is exactly what a
        layer-scoped override row makes differ between segments. Asking per
        segment is also what makes a profile that grows a layer dimension a
        change in one place rather than a second split here.
        """
        segments = self.saved.layer_segments(
            stored.name, role, stored, node_dims, stack, components, layers)
        several = len(segments) > 1
        return Representations.of(*(
            Representation(
                saved=saved,
                intent=(None if self.intent is None else
                        self.intent.for_node(self.saved, stored, role, stack,
                                             node_dims, saved)),
                layers=covered if several else None)
            for covered, saved in segments))


def segments_of(walked_formats: Mapping[int, Any],
                *uids: int) -> Tuple[Representation, ...]:
    """Every layer segment of every tensor the given nodes store.

    The walk's `formats` layer holds one `Representations` per stored tensor,
    index-parallel with `storage`. A reader that asks about formats rather
    than about tensors — which dtype a kernel multiplies by, whether a
    conversion happens — wants them flat, and wants all of them: a tensor
    stored at two formats across its node's layers is two facts, and taking
    one would silently drop the other.
    """
    return tuple(segment
                 for uid in uids
                 for reps in walked_formats.get(uid, ())
                 for segment in reps)


def layer_scopes_of(program: Any) -> Dict[int, FrozenSet[int]]:
    """The layers each layer-scoped `quant_context.overrides` row speaks
    about, resolved against the model's dims and config.

    Resolved here, once per policy, rather than per node: a layer bound is a
    dim expression and a per-layer schedule is a config list, and
    `ResolvedSpec.layers_selected` is the one place either becomes indices —
    the same call a `BlockAssignment` goes through. The stack a row names
    supplies the layer count the selection resolves against, the backbone
    when it names none, since that is what a layer number means unqualified.
    """
    quant_context = program.spec.quant_context
    if quant_context is None:
        return {}
    out: Dict[int, FrozenSet[int]] = {}
    for index, entry in enumerate(quant_context.overrides):
        selection = _layer_selection(entry)
        if selection is None:
            continue
        stack = entry.select.stack or ""
        depth = program.stack_depth(stack)
        selected = program.resolved.layers_selected(selection, depth)
        if not selected:
            raise SpecRefusal(
                f"quant_context.overrides[{index}] "
                f"({entry.selector_text()}) scopes itself to layers, and "
                f"they resolve to none of the "
                f"{Stack.display_name(stack)} stack's layers "
                f"({depth if depth is not None else 'no'} of them)",
                declaration="quant_context.overrides[].select.layers",
                fix="name layers this stack has, or drop the layer scope so "
                    "the row speaks about every layer")
        out[index] = frozenset(selected)
    return out


def _layer_selection(entry: Any) -> Optional[Any]:
    """The `LayerSelection` an override row scopes itself to, or None.

    Read off the declaration's own fields: an `OverrideEntry` has a
    `select`, and a selector has `layers` — both may be None, and neither is
    ever absent, so there is nothing to probe for."""
    selector = entry.select
    return None if selector is None else selector.layers


def _override_literals(entry: Any) -> Optional[FrozenSet[str]]:
    selector = entry.select
    names = (selector.tensors if selector is not None
             else entry.tensors)
    return frozenset(names) if names else None


# -- building the layer ---------------------------------------------------

# -- registry access ------------------------------------------------------

def _scheme(method: str) -> Any:
    from backend.arch.quant_schemes import QUANT_SCHEMES
    return QUANT_SCHEMES.get(method)


def _shape_elements(extents: Sequence[Extent]) -> Extent:
    return product(extents)


def _leading(stored: Any) -> Tuple[Extent, ...]:
    """The axes of a stored matrix before its last two — one matrix per
    index (`StoredTensor.matrix_dims`). Empty for anything that is not a
    `(rows, cols)`-read matrix of rank above two."""
    if not stored.kind.stores_out_in:
        return ()
    return tuple(stored.shape[:-2])


def _product(extents: Sequence[Extent]) -> Extent:
    return product(extents) if extents else 1


def _metadata_stored_transform(
    stored: Any,
    axes: Optional[Sequence[str]],
) -> Optional[str]:
    """Record the typed stored-axis view consumed by a metadata formula."""
    if stored is None or not axes:
        return None
    dependencies = frozenset().union(
        *(formula_symbols(str(axis)) for axis in axes)
    )
    if not dependencies.intersection({"in", "out"}):
        return None
    if stored.kind.stores_out_in and len(stored.shape) >= 2:
        return "matrix"
    if stored.kind is StorageKind.HEADWISE and len(stored.shape) >= 2:
        return "flatten_leading"
    raise ValueError(
        f"{stored.name}: matrix metadata formula has no typed stored-axis "
        f"transform for rank-{len(stored.shape)} {stored.kind.value} storage"
    )


def _cardinality(formula: str, bound: Mapping[str, Any]) -> Extent:
    """A metadata cardinality is NOT written in the spec's dim-expression
    language. It is a restricted arithmetic formula over `in`, `out` and the
    scheme's own config fields, where `in` is a Python keyword that has to be
    rewritten before it can be parsed at all. `scheme_dialect` implements
    exactly that — keyword rewrite plus a whitelisted AST walk — so this defers
    to it rather than growing a second dialect that would drift.
    """
    return eval_formula(formula, dict(bound))


def _predicate(when: str, bound: Mapping[str, Any]) -> bool:
    """A sibling's `when:` — same symbol space, same evaluator."""
    try:
        return bool(eval_predicate(when, dict(bound)))
    except (KeyError, ValueError, TypeError):
        # Unresolvable predicate: emit the sibling. An unstated scale is a
        # silent under-count of storage; a spurious one shows up as a number
        # that does not match the checkpoint.
        return True


def _variant(scheme: Any, name: Optional[str]) -> Any:
    variants = scheme.format_variants or {}
    return variants.get(name or "default") or next(iter(variants.values()), None)


def _variant_field(variant: Any, field_name: str, bits: int) -> Any:
    """A variant states a field either flat or per-num_bits."""
    if variant is None:
        return None
    flat = getattr(variant, field_name, None)
    if flat is not None:
        return flat
    per_bits = getattr(variant, f"{field_name}_per_num_bits", None) or {}
    return per_bits.get(bits)


def _entry_fields(entry: Any, scheme: Any) -> Dict[str, Any]:
    """Scheme defaults, then the per-role config on top."""
    fields: Dict[str, Any] = {k: v for k, v in
                              (scheme.default_fields or ())}
    extra = entry.model_extra or {}
    fields.update(extra)
    for key in ("method", "format", "strategy", "num_bits"):
        value = getattr(entry, key, None)
        if value is not None:
            fields.setdefault(key, value)
    return fields


def _scheme_bits(scheme: Any, why: str) -> int:
    """The scheme's own width, asked with no per-model config — the answer
    when the entry leaves `num_bits` unset.

    A scheme whose width DEPENDS on config (`num_bits: {field_any: …}`) has
    no answer here, and its reader says so with 0; the entry must then state
    the width. Either way a missing width is refused, never priced at a
    guessed one.
    """
    bits = scheme.read_num_bits({})
    if not bits:
        raise ValueError(
            f"{why}: scheme {scheme.name!r} takes its width from the config "
            f"and the quant_context entry states no `num_bits`")
    return int(bits)


def _bits(dtype: str) -> int:
    return int(_dtype_bytes(dtype) * 8)
