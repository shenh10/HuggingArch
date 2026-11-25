"""`OpRef` — one interface for "the thing a node's `op:` names".

A node's op is a registered primitive, a spec-scoped `custom_ops:` declaration, or
a component. The first two both answer the same questions — how many FLOPs, how
many params, what shape comes out, what the secondary ports produce, what goes on
disk, how many cache positions a query touches — and differ only in where the
answers are written: a Python lambda on `OpSpec`, or a dim expression in the
model's YAML.

The engine keeps them as separate branches of `_eval_node_body`, each with its own
shape handling, its own dims merge and its own fallbacks. That is why adding a
custom op means touching the evaluator, and why the custom-op branch drifted into
resolving its formulas against a different dims dict than the primitive branch
uses.

One protocol, two implementations, resolved once at the node. A component is not an
`OpRef` — it has no cost of its own, only children — so it stays a separate case in
the walk rather than being forced into this shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..ops import PhaseBindingMissing, get_op, is_primitive
from ..ops_base import ShapeKind, resolve_shape_template, weight_leading_axes
from ..shape_node import ShapeProbe
from .binding import Definition
from .dim import Dim, equal, require_equal, value_of, varies
from .refusal import SpecRefusal


def _law(positions: Any) -> Any:
    """A position law, in the vocabulary its consumers substitute against.

    The lambda evaluates against whatever `dims` binds — under a tracked
    frame that yields a `Dim`, whose expression half names the op's own
    dims (`Min(s_kv, s_window)`). The law's declared form is the VALUE
    half: every dim resolved except the length the caller bound (the
    `CACHE_LEN` sentinel, itself a sympy expression, survives grounding).
    `positions()` substitutes that sentinel; handing it the `Dim` left
    the law unsubstitutable and `int()` met an unresolved `Min`.
    """
    return value_of(positions) if positions is not None else None


@dataclass(frozen=True)
class KVPositions:
    """What an op says about the cache it touches.

    `read` and `stored` are separate because they disagree on real hardware and
    the disagreement is the interesting case: DSA stores every position and
    reads top-K of them, a sliding window does both at `Min(s_kv, s_window)`,
    and a dense attention does both at `s_kv`. Either may be `None` — an op that describes one
    side and not the other is stating what it knows, and a caller that treats a
    missing side as zero would bill an unbounded cache at nothing.

    `is_block_anchor` says whether this op's answer may stand for the whole
    attention component. An auxiliary full-scan — DSA's `indexer_logits` — has a
    real cardinality for its OWN cached tensor and must not become the layer's
    headline, so it declares itself out.
    """

    read: Optional[Any] = None
    stored: Optional[Any] = None
    is_block_anchor: bool = True
    #: Distinct positions one step reads (`OpSpec.kv_positions_per_step_fn`);
    #: `read` when the op declares no union of its own.
    step: Optional[Any] = None
    #: Live (query, key) pairs the step scores under the op's selection
    #: (`OpSpec.kv_pairs_per_step_fn`); `None` when the phase's dense pairs
    #: already are the op's.
    pairs: Optional[Any] = None
    #: The same pairs as their definition in the node's names
    #: (`binding.Definition`), which the node's `s_pairs` is rebound to.
    named: Optional[Any] = field(default=None, compare=False)
    #: `step` as its definition in the node's names, each name bound to what
    #: the frame bound it to BEFORE the walk narrowed `s_kv` to `step`: the
    #: law a per-rank projection re-evaluates at the rank's own extents
    #: (`layout.physical.rank_law_shape`).
    step_named: Optional[Any] = field(default=None, compare=False)


@dataclass(frozen=True)
class KVLength:
    """How many KV positions a tensor holds, travelling along the edges it
    holds them on.

    `KVPositions` is what an OP says; this is what a TENSOR carries. They are
    the same number for a plain attention chain, and they part company where a
    `kv_stream_concat` joins two streams: the union's length is neither
    operand's and is not `s_kv`, so the score reading it cannot get the number
    from its own law and has to be told.

    `read` is that length at this walk's cache length; `symbolic` is the same
    expression with the cache length held free, for the capacity question that
    asks for the whole curve.

    `dictates` says whether consumers should adopt it as their `s_kv`. A
    `kv_stream_concat` union sets it, and so does a stack stream whose
    producer cached the tensor. Every other law-declaring op records its
    cardinality here so a union can sum it, and stays out of its own
    consumers' way.
    """

    read: Any
    symbolic: Any
    dictates: bool = False
    #: The step's live (query, key) pairs over those positions, at this walk
    #: (`KVPositions.pairs`); a union sums its operands'. `None`: the stream
    #: declares no mask of its own, and its reader keeps the phase's pairs.
    pairs: Optional[Any] = None
    #: Those pairs as their definition (`KVPositions.named`); a union's is
    #: the sum of its streams'.
    named: Optional[Any] = field(default=None, compare=False)
    #: Whether those pairs are a per-query index list its reader gathers
    #: row by row: a `kv_stream_concat` union's, which the source builds by
    #: joining its streams' index lists. A single stream's mask (a window,
    #: every pooled block) read on its own is a band its reader streams.
    listed: bool = False
    #: The node whose op stated this length (`OpRef.kv_positions`), carried
    #: with it through the views and stream publications that hand it on;
    #: None for a union's, which no single node states. Not part of the
    #: length: two layers publishing one stream state equal lengths at
    #: different nodes.
    stated_by: Optional[int] = field(default=None, compare=False)


@dataclass(frozen=True)
class StreamLaw:
    """The cache law an inner stream carries where the model hands it to a
    call: the length the op of one node (`op`, at `node`) states in the
    layer that published the stream, under that node's dims — the ones it
    declares and the ones its frame rebinds, as the model's walk bound them
    there (`dims`, each an expression in the model's dims and the
    workload's symbols) — published as a cache entry's law (`cached`) or as
    the length the node dictates. `op` is None where the stream carries a
    union's length, which no one node states.

    An isolated call is handed the stream with this law, as a call in the
    model is (`Evaluator._isolated_stream`)."""

    stream: str
    op: Optional[str]
    node: str
    dims: Tuple[Tuple[str, Any], ...]
    cached: bool


@dataclass(frozen=True)
class OpRef:
    """A uniform view over a registered primitive or a spec-scoped custom op."""

    name: str
    _spec: Any                       # OpSpec | CustomOpSpec
    _custom: bool
    #: For a join (`axis_concat`), the axis this node joins along: the op's
    #: own `axis_locked`, else the node's `axis:`, else the last. None for
    #: every other op. What tells the joined axis apart in `input_templates`.
    join_axis: Optional[int] = None

    @classmethod
    def resolve(cls, op_name: str, program: Any, *,
                axis: Optional[int] = None) -> Optional["OpRef"]:
        """`None` when the name is a component or unknown — the caller decides
        which of those is an error in its context. `axis` is the node's
        `axis:`, which a join reads (`join_axis`)."""
        custom = program.custom_op(op_name)
        if custom is not None:
            spec, is_custom = custom, True
        elif is_primitive(op_name):
            spec, is_custom = get_op(op_name), False
        else:
            return None
        join_axis = None
        if ShapeKind(spec.shape_kind) is ShapeKind.AXIS_CONCAT:
            join_axis = (spec.axis_locked if spec.axis_locked is not None
                         else axis if axis is not None else -1)
        return cls(op_name, spec, is_custom, join_axis)

    def input_templates(self, shapes: Sequence[Any]) -> Tuple[Tuple[str, ...], ...]:
        """`shape_in` as one template per ARRIVING input, for a join: every
        operand shares the declared labels but the joined axis, which each
        operand names for itself — its length is its own, and the join sums
        them. The declared templates name the joined axis last (`in_a`,
        `in_b`); a join along another axis (`axis: 1`, S + (S - 1) rows)
        moves the operand's own label to that axis and shares the last. An
        axis left of what the template covers extends the template to it.
        Empty for any other op, which reads its declared templates."""
        if self.join_axis is None:
            return ()
        declared = tuple(self._spec.shape_in or ())
        if not declared:
            return ()
        if not all(isinstance(t, (tuple, list)) for t in declared):
            declared = (declared,)
        declared = tuple(tuple(str(x) for x in t) for t in declared)
        base = declared[0]
        out: List[Tuple[str, ...]] = []
        for i, shape in enumerate(shapes):
            rank = len(shape) if shape else 0
            if rank == 0 or not -rank <= self.join_axis < rank:
                out.append(base)
                continue
            at = self.join_axis % rank
            lead = rank - len(base)
            template = list(base)
            if at < lead:
                template = [f"axis{j}" for j in range(at, lead)] + template
                lead = at
            at -= lead
            last = at == len(base) - 1
            template[at] = (declared[i][-1] if last and i < len(declared)
                            else f"{template[at]}_{i}")
            out.append(tuple(template))
        return tuple(out)

    def arity(self) -> int:
        """How many inputs the op declares. Zero means the node is a
        source: nothing arrives along an edge, so nothing is defaulted in."""
        return self._spec.arity()

    @property
    def repeats_last_input(self) -> bool:
        """Whether the op reads its last declared input any number of times
        (`OpSpec.repeats_last_input`); a spec-scoped op never does."""
        return not self._custom and self._spec.repeats_last_input

    @property
    def binding_dims(self) -> Tuple[str, ...]:
        """The model-scope dims the op reads off its binding
        (`OpSpec.binding_dims`); a spec-scoped op reads none."""
        return () if self._custom else self._spec.binding_dims

    # -- cost ------------------------------------------------------------

    def flops(self, dims: Mapping[str, Any], resolve_expr: Callable) -> Any:
        if self._custom:
            return resolve_expr(self._spec.flops or "0", dims)
        # No `dict(dims)`: the lambdas only subscript, and copying a
        # ReadTracking map would mark every dim read.
        return self._spec.flops_fn(dims)

    def params(self, dims: Mapping[str, Any], resolve_expr: Callable) -> Any:
        """How many parameters, and where the number comes from.

        An op that declares its TENSORS has already answered: the parameter
        count is their elements summed, and `params_fn` is that sum written a
        second time. Reading the projection instead of the second copy is what
        makes the two unable to disagree.
        """
        from .storage import declared_elements

        projected = declared_elements(self._spec, dims)
        if projected is not None:
            return projected
        if self._custom:
            counted = resolve_expr(self._spec.params or "0", dims)
        else:
            counted = self._spec.params_fn(dims)
        lead = weight_leading_axes(self._spec, dims)
        if not lead:
            return counted
        # A node that declares its weight's shape stores exactly that
        # tensor, so its parameters are that tensor's elements — which is
        # what the op counted for one matrix of it, or the op counts
        # something its weight is not and the shape cannot be declared.
        core = resolve_shape_template(self._spec.weight_shape, dims)
        one = Dim.of(1)
        for extent in core or ():
            one = one * extent
        if not equal(counted, one):
            raise ValueError(
                f"{self.name}: counts {value_of(counted)} parameters where "
                f"its weight holds {value_of(one)}, so the parameters are not "
                "the weight's elements and its shape cannot be declared")
        total = one
        for extent in lead:
            total = total * extent
        return total

    # -- shape -----------------------------------------------------------

    def shape_out(self, shape_in: Sequence, dims: Mapping[str, Any],
                  axis: Optional[int], axes: Optional[List[int]],
                  resolve_expr: Callable, node: Any = None) -> Optional[Tuple]:
        """Output shape from the DimMap. One map, primitives and custom ops."""
        del resolve_expr
        ins = [tuple(s) if s else () for s in shape_in]
        from backend.arch.axis_propagation import infer_shape
        probe = node
        if probe is None:
            # A declared `shape:` is the output only of a source op, whose
            # output IS the tensor it stores (`ShapeNode.out_shape`).
            listed = (dims.get("_shape") if dims is not None
                      and ShapeKind(self._spec.shape_kind) is ShapeKind.SOURCE
                      else None)
            probe = ShapeProbe.probe(
                axis=axis, axes=axes, out_shape=listed if listed else None)
        inferred = infer_shape(self._spec, probe, ins, dims)
        return tuple(inferred) if inferred is not None else None

    def secondary_ports(self, shape_in: Sequence, dims: Mapping[str, Any],
                        axis: Optional[int],
                        resolve_expr: Callable) -> Dict[str, Tuple]:
        del shape_in, axis
        out: Dict[str, Tuple] = {}
        declared = self._spec.secondary_outputs or {}
        for port, template in declared.items():
            # Templates, same encoding YAML custom ops already use. Binding
            # values are Dims; looking them up keeps identity instead of
            # running a second Python shape function.
            extents = []
            for label in template:
                key = str(label)
                extents.append(dims[key] if key in dims else resolve_expr(key, dims))
            out[port] = tuple(extents)
        return out

    def input_contract_violation(self, shape_in: Sequence,
                                 dims: Mapping[str, Any]) -> Optional[str]:
        """What the arriving tensors break of the op's declared
        `input_contract`, or None. A spec-declared op states its contract in
        its templates alone."""
        lead = weight_leading_axes(self._spec, dims)
        if lead and self._spec.arity():
            # A source's leading axes lead the tensor it stores and emits;
            # only an op that contracts an activation pairs them with one.
            violation = self._batch_violation(shape_in, lead)
            if violation is not None:
                return violation
        if self._custom or self._spec.input_contract is None:
            return None
        return self._spec.input_contract(
            [tuple(s) if s else () for s in shape_in], dims)

    def weight_operand_violation(self, arriving: Mapping[str, Any],
                                 dims: Mapping[str, Any]) -> Optional[str]:
        """What the arriving tensors break of the op's own weight, or None.

        A label the op's stored-weight declaration names (`weight_shape`, or
        a `tensors` entry's shape) and its input template names too is one
        axis seen twice: the extent the node binds it to sizes the weight,
        and the extent `arriving` (`flops.bind_labels` over the tensors that
        arrived) says the input carries must be the same one. A `linear`
        whose `in` is bound to `d_ffn` while a `d_model`-wide tensor reaches
        it walks to a clean output with a weight that cannot multiply what
        it is fed; nothing downstream sees it but the checkpoint's bytes,
        and a model with no checkpoint has none."""
        declared = [*(self._spec.weight_shape or ()),
                    *(label for decl in self._spec.tensors
                      for label in decl.shape)]
        for label in dict.fromkeys(x for x in declared if isinstance(x, str)):
            if label not in arriving or label not in dims:
                continue
            fed, bound = arriving[label], dims[label]
            message = (f"its weight takes {label}={value_of(bound)}, and the "
                       f"input arrives {value_of(fed)} wide on that axis")
            if varies(value_of(fed) - value_of(bound)):
                require_equal(fed, bound, lambda _at: SpecRefusal(
                    f"{self.name}: {message}", declaration="in",
                    fix=f"bind {label} to the dim the input carries, or "
                        "wire the input it was meant to read"))
            elif not equal(fed, bound):
                return message
        return None

    def _batch_violation(self, shape_in: Sequence, lead: Sequence[Any]
                         ) -> Optional[str]:
        """`torch.matmul`'s batch rule for a weight with leading axes: the
        activation's batch axes — every axis before its last two — must END
        in the weight's leading axes, position by position, each the same
        extent (`dim.equal`, over every point a symbolic walk admits).
        Matrix `i` of the weight contracts batch slice `i` of the
        activation and no other."""
        first = next((tuple(s) for s in shape_in if s), ())
        batch = first[:-2]
        if len(batch) < len(lead):
            return (f"its weight has {len(lead)} leading axes "
                    f"{[value_of(x) for x in lead]!r}, and the activation "
                    f"{[value_of(x) for x in first]!r} has only "
                    f"{len(batch)} batch axes to pair them with")
        tail = batch[len(batch) - len(lead):]
        message = (f"its weight's leading axes "
                   f"{[value_of(x) for x in lead]!r} and the activation's "
                   f"batch axes {[value_of(x) for x in tail]!r} differ — "
                   "a batched contraction pairs them position by position")
        for a, b in zip(tail, lead, strict=True):
            if varies(value_of(a) - value_of(b)):
                # Laws in the workload: the pairing is a condition on the
                # point, refused at a point that breaks it.
                require_equal(a, b, lambda _at: SpecRefusal(
                    f"{self.name}: {message}", declaration="in",
                    fix="give the op an activation whose batch axes end in "
                        "its weight's leading axes"))
            elif not equal(a, b):
                return message
        return None

    # -- per-op semantics -------------------------------------------------

    def kv_positions(self, dims: Mapping[str, Any],
                     resolve_expr: Callable) -> Optional["KVPositions"]:
        """How many cache positions one query touches, read and stored.

        The op declares both — a sliding window reads AND stores
        `Min(s_kv, s_window)`, DSA reads `Min(s_kv, n_topk_keys)` but stores
        every `s_kv` so its indexer has
        something to score — and they are asked together because they are one
        declaration about one cache, written beside the `flops_fn` that already
        counts the same positions.

        Evaluated against whatever `dims` binds `s_kv` to, and that is the whole
        of the difference between the two callers. The walk passes its own
        numeric cache length and gets a number, which is what substituting an
        effective length into a shape needs. `eval.cache` passes `Symbol("s_kv")`
        and gets the expression, which is what a capacity question needs — the
        same footprint at every cache length, not one length's answer.

        Asking once is the point. These lambdas were read in two places against
        two dim dicts assembled two ways, and the storage half was read in
        neither of them by the byte derivation that most needed it.
        """
        if self._custom:
            # A YAML custom op declares no KV law: CustomOpSpec (extra="forbid")
            # has none of kv_positions_per_query / kv_storage_positions_per_query
            # / kv_block_anchor, so the reads were always None and this branch
            # always returned None. Its cache cardinality is simply absent.
            return None
        read_fn = self._spec.kv_positions_per_query_fn
        stored_fn = self._spec.kv_storage_positions_per_query_fn
        if read_fn is None and stored_fn is None:
            return None
        read_extent = self._call(read_fn, dims)
        read = _law(read_extent)
        step_fn = self._spec.kv_positions_per_step_fn
        step_extent = (self._call(step_fn, dims) if step_fn is not None
                       else read_extent)
        step = _law(step_extent) if step_fn is not None else read
        pairs = self._call(self._spec.kv_pairs_per_step_fn, dims)
        return KVPositions(
            read=read,
            stored=_law(self._call(stored_fn, dims)),
            is_block_anchor=bool(self._spec.kv_block_anchor),
            step=step,
            pairs=_law(pairs),
            named=(None if pairs is None
                   else Definition.of_extent(pairs, dims)),
            step_named=(None if step_extent is None
                        else Definition.of_extent(step_extent, dims)),
        )

    def cost_pairs(self, dims: Mapping[str, Any]) -> Optional[Any]:
        """The live pairs this op's own matrix holds, for its cost
        (`OpSpec.cost_pairs_fn`), as their definition in `dims`; None for
        an op that declares none."""
        if self._custom or self._spec.cost_pairs_fn is None:
            return None
        return Definition.of_extent(
            self._call(self._spec.cost_pairs_fn, dims), dims)

    def _call(self, fn: Optional[Callable], dims: Mapping[str, Any]) -> Optional[Any]:
        """One position law, evaluated. An op that declares no such law has
        `fn is None`; a declared law that fails to evaluate is a refusal — it
        is not the op saying it has no law."""
        if fn is None:
            return None
        try:
            return fn(dict(dims))
        except PhaseBindingMissing:
            raise           # a binding the caller lacks, named by its own type
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"{self.name}: its kv_positions law failed to evaluate: "
                f"{type(exc).__name__}: {exc}. The law reads a dim this call "
                f"site does not bind — bind it in the component's dims, or use "
                f"an op whose law does not read it.") from exc

    # -- storage ----------------------------------------------------------

    def stored_tensors(self, dims: Mapping[str, Any], params: Any, module: str,
                       resolve_expr: Callable) -> Tuple:
        """What this op puts on disk — a tuple, empty if it owns nothing.

        The fifth question, and it is asked of the op for the same reason the
        other four are: the answer is a property of the operation, and every
        consumer that instead worked it out for itself got a different answer.
        `spec_tree` guesses the weight's dims by symbol name, the quant schemes
        resolve `in`/`out` against the enclosing scope, and `sharding` compares
        `op == "linear"` as a string.

        Plural because a checkpoint is: `q_proj.weight` and `q_proj.bias` are two
        entries with two dtypes, and a scheme that quantises the weight leaves
        the bias alone. Singular made the bias unrepresentable rather than
        wrong, which is the harder kind of gap to see.
        """
        from .storage import from_declaration
        return from_declaration(self._spec, dims, params, module,
                                resolve_expr)

    def refuse_a_shape_it_cannot_store(self, node: Any) -> None:
        """A node's `shape:` is the shape of a tensor it stores — the op's
        `weight_shape` or `tensors` — so on an op that stores none it
        describes nothing, and is refused."""
        stores = bool(self._spec.weight_shape) or bool(self._spec.tensors)
        if node.shape_raw is None or stores:
            return
        raise SpecRefusal(
            f"{node.display_path} ({node.op}) declares `shape:`, the shape "
            "of a tensor it stores, and the op stores none",
            declaration="shape",
            fix="declare `shape:` on the node that stores the tensor",
            node=node.display_path)

    def module_path(self, node: Any, prefix: str) -> str:
        """The on-disk name of what this op stores, under `prefix`.

        Here rather than at the call site because the SUFFIX is the op's
        declaration — `.weight` for an nn.Module-wrapped weight, `""` for a
        naked nn.Parameter — and reaching through `.spec` to get it is the
        caller doing this object's job. Falling back to the node's LOCAL name
        instead is what made every name-keyed override miss: DeepSeek's router
        is `mlp.gate.linear` locally and `layers.X.mlp.gate.weight` on disk, and
        the override pinning it BF16 is written against the latter.
        """
        from .storage import module_path
        return module_path(node, prefix)

    @property
    def shape_in_template(self) -> Tuple:
        if self._custom:
            return tuple(tuple(shape) for shape in self._spec.signature.values())
        return tuple(self._spec.shape_in or ())

    @property
    def shape_out_template(self) -> Tuple:
        if self._custom:
            return ()
        return tuple(self._spec.shape_out or ())

    @property
    def shape_kind(self) -> Any:
        return self._spec.shape_kind

    @property
    def couples_on(self) -> Tuple[str, ...]:
        return tuple(self._spec.couples_on or ())

    @property
    def is_view_op(self) -> bool:
        """Whether this op re-indexes rather than rewrites. A custom op is
        never one — `custom_ops:` declares flops, and an op that declares
        work to do is doing it."""
        if self._custom:
            return False
        return bool(self._spec.is_view_op)

    @property
    def kv_stream_union(self) -> bool:
        """Whether this op joins several KV streams into one tensor — see
        `OpSpec.kv_stream_union`. A custom op is never one: `custom_ops:`
        declares compute in dim expressions and has no way to name the
        cache streams it would be joining."""
        if self._custom:
            return False
        return bool(self._spec.kv_stream_union)

    @property
    def kv_stream_marker(self) -> bool:
        """Whether this op's KV law counts the stream it emits — see
        `OpSpec.kv_stream_marker`. A custom op is never one, for the reason
        it is never a union."""
        if self._custom:
            return False
        return bool(self._spec.kv_stream_marker)

    @property
    def is_buffer(self) -> bool:
        return bool(self._spec.is_buffer)

    @property
    def consts(self) -> Tuple[str, ...]:
        """The constant symbols this op's math takes (`OpSpec.consts`). A
        custom op takes none: `custom_ops:` declares cost in extents and has
        no way to name a function constant."""
        if self._custom:
            return ()
        return tuple(self._spec.consts)

    @property
    def spec(self) -> Any:
        """The underlying declaration, for the derivations that read fields
        both kinds share (`shape_in` as an einsum signature). Read-only by
        convention; the four questions above stay the interface."""
        return self._spec
