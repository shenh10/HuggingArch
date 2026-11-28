"""`flops_by_module` walked off the grafted + fused tree.

The phase table used to walk the logical `analysis.program` — a tree that
knows nothing of parallelism, so collectives and fused kernels had to be
joined and spliced back onto its rows after the fact. The serving pipeline
already materializes the real tree: `AnalysisResult.post_tree` is the spec
IR with every physical collective grafted in place and every fused kernel
standing in for its absorbed siblings. Walking THAT makes comm and fused
rows native: each leaf merges its whole per-op timing row by `uid`, and a
parent's FLOPs are simply its subtree's sum — the one convention that
cannot mix per-call and per-layer frames (a MoE expert leaf fires
`layers × experts` times while its wrapper fires `layers`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple

from backend.arch.eval.analysis import Point
from backend.arch.eval.dim import as_extent, value_of
from backend.arch.eval.program import BlockStage, OpStage, Program
from backend.arch.eval.refusal import node_context
from .row_roofline import ROW, column_names


# Timing-row fields the table never renders, off the producer's one column
# declaration. The whole-row merge keeps the
# leaf's facts complete by default (a deny-list, not the old allow-list: a
# new timing field rides along instead of silently missing), but shipping
# every per-op component field on the module rows too would send the same
# row twice in one response — these stay on `per_op_timing`, which the
# response also carries and the batch sweep rescales from.
_TIMING_DROP = column_names(published_by=ROW, on_module_table=False)


def leaf_columns(leaf: Mapping[str, Any]) -> Dict[str, Any]:
    """A timing row's columns as its table row takes them (`_TIMING_DROP`
    stays on `per_op_timing`)."""
    return {key: value for key, value in leaf.items() if key not in _TIMING_DROP}


@dataclass(frozen=True)
class TableIdentity:
    """What the walk lays over a table row's leaf columns: the row's place in
    the tree (name, op, role, ids, depth) and its counts — the template's
    layer count, the unsharded FLOPs off the full cost layer."""
    name: str
    op: str
    role: str
    aggregation_id: str
    template_name: str
    layer_count: int
    depth: int
    uid: Any
    flops_full: int
    #: The FLOPs as laws, for a reader at another point (`batch_sweep`) and
    #: no response (`served`): `full`, one call's unsharded — one
    #: data-parallel replica's (`replica_cost`) — in
    #: the workload's counts, read at `point`, times `calls`; an untimed
    #: row's `rank`, one call's per rank, the same way.
    flops_law: Mapping[str, Any]

    @classmethod
    def of(cls, row: Mapping[str, Any]) -> "TableIdentity":
        """A table row's identity, read back off the row."""
        return cls(**{f.name: row[f.name] for f in fields(cls)})

    def columns(self) -> Dict[str, Any]:
        # Field by field, not `asdict`: that deep-copies the laws, and a
        # SymPy expression copied is factored again.
        return dict(vars(self))


def served_rows(rows: Any) -> Any:
    """Per-op or module-table rows as a response serves them: without the
    laws they carry for a reader at another point (`roofline_expr`,
    `point_law`, `shape_law`, a table row's `flops_law`, a fused kernel's
    absorbed parts' `shape_out_law`) — expressions, which live in the
    process that walked them."""
    def row_of(row: Mapping[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in row.items()
               if k not in ("roofline_expr", "point_law", "shape_law",
                            "flops_law")}
        badge = out.get("source_badge")
        if badge and badge.get("absorbed"):
            out["source_badge"] = {**badge, "absorbed": [
                {k: v for k, v in part.items() if k != "shape_out_law"}
                for part in badge["absorbed"]]}
        return out

    return [row_of(row) for row in rows] if rows else rows


def served(result: Mapping[str, Any]) -> Dict[str, Any]:
    """An estimate as a response serves it: every per-op row list and every
    module table — each phase's, each encoder stack's — as `served_rows`,
    and a pipeline step without its stages' row laws (`stage_laws`): only
    what a reader of the response shows."""
    def phase(ph: Any) -> Any:
        if not ph:
            return ph
        out = dict(ph)
        if "flops_by_module" in out:
            out["flops_by_module"] = served_rows(out["flops_by_module"])
        if out.get("pipeline_time") is not None:
            out["pipeline_time"] = {k: v for k, v in out["pipeline_time"].items()
                                    if k != "stage_laws"}
        return out

    out = {k: (served_rows(v) if k.startswith("per_op_timing")
               and isinstance(v, list) else v) for k, v in result.items()}
    if out.get("phases"):
        out["phases"] = {name: phase(ph) for name, ph in out["phases"].items()}
    encode = out.get("encode")
    if encode and encode.get("stacks"):
        out["encode"] = {**encode, "stacks": [
            {k: (served_rows(v) if (k == "flops_by_module"
                                    or k.startswith("per_op_timing"))
                 and isinstance(v, list) else v) for k, v in stack.items()}
            for stack in encode["stacks"]]}
    return out


def table_row(leaf: Mapping[str, Any], identity: TableIdentity) -> Dict[str, Any]:
    """A timed table row: its leaf's whole timing row (times, bytes,
    formulas, shapes, comm facts, source badge — `leaf_columns`), the walk's
    identity laid over it, its FLOPs the leaf's. A leaf re-read at another
    batch or card makes its table row the same way (`batch_sweep`)."""
    row = {**leaf_columns(leaf), **identity.columns(),
           "flops_per_rank": leaf["flops_per_rank"]}
    if row.get("shape_out") is None:
        row.update(fused_ports(leaf))
    return row


def fused_ports(leaf: Mapping[str, Any]) -> Dict[str, Any]:
    """The table row of a multi-output fused leaf (merged_column_parallel_linear
    …), from its timing row: the leaf has no primary output, but the fused
    GEMM physically emits ONE tensor, the per-port shapes concatenated on the
    last axis — that concat is the row's `shape_out` and the named ports its
    `shape_outs`. Empty for a leaf with fewer than two ports."""
    names = leaf["point_law"]["ports"]
    shapes = leaf.get("shape_outs") or []
    ports = {name: shape for name, shape in zip(names, shapes, strict=True)
             if shape}
    if len(ports) < 2:
        return {}
    out: Dict[str, Any] = {"shape_outs": ports}
    concat = _concat_last_axis(list(ports.values()))
    if concat is not None:
        out["shape_out"] = concat
    return out


def shape_columns(ins: Any, out: Any, ins_full: Any, out_full: Any
                  ) -> Dict[str, Any]:
    """The shape columns of an untimed table row (a wrapper, plumbing the
    overlay pruned) from its per-rank and unsharded extents — the walk's
    `Dim`s, or a law's values at another batch (`batch_sweep`). The full
    shapes are shown only where the shard differs: a replicated row has
    nothing to explain."""
    def axes(one: Any) -> List[int]:
        return [as_extent(a, rounding="floor") for a in one]

    def tensors(many: Any) -> Any:
        values = [axes(one) for one in many]
        return values[0] if len(values) == 1 else values

    row: Dict[str, Any] = {}
    if ins:
        row["shape_in"] = tensors(ins)
    if out is not None:
        row["shape_out"] = axes(out)
    in_full = tensors(ins_full) if ins_full else None
    shown_out_full = axes(out_full) if out_full is not None else None
    if in_full is not None and in_full != row.get("shape_in"):
        row["shape_in_full"] = in_full
    if shown_out_full is not None and shown_out_full != row.get("shape_out"):
        row["shape_out_full"] = shown_out_full
    return row


def whole_count(n: Any, what: str) -> int:
    """`n` as the number of times something runs — a whole count, never a
    fraction rounded down and never a stand-in for a missing one."""
    if n is None:
        raise ValueError(f"{what}: no count")
    v = value_of(n)
    if isinstance(v, bool) or v != int(v):
        raise ValueError(f"{what} runs {v} times, not a whole count")
    return int(v)


def instance_count(instances: Any, uid: int) -> int:
    """How many times node `uid` runs in the step, off the instances
    layer; a node the layer does not count is an error."""
    if uid not in instances.values:
        raise ValueError(f"uid {uid} has no instance count")
    return whole_count(instances.values[uid], f"uid {uid}")


def _concat_last_axis(shape_list: List[List[int]]) -> Optional[List[int]]:
    """Concat a list of equal-leading-axes shapes on the last axis.

    A column-parallel fused GEMM physically emits one tensor of shape
    `[..., Σ last_axis]`, view-split into per-port slices for free — so
    the concat, not any single port, is what the kernel's output looks
    like. None when the ports don't share leading axes (leave the
    per-port tooltip to carry the breakdown).
    """
    shapes = [list(s) for s in shape_list if s]
    if len(shapes) < 2:
        return None
    head = shapes[0][:-1]
    if any(s[:-1] != head for s in shapes):
        return None
    return [*head, sum(s[-1] for s in shapes)]


def physical_module_rows(
    result: Any,
    per_op_timing: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """The phase's per-op cost table, off `AnalysisResult.post_tree`.

    Row schema matches `phases._flops_rows` (plus `uid` and the merged
    timing fields), so every consumer of `flops_by_module` — the batch
    sweep's rescale, `attach_parent_formulas`, the React fold — reads it
    unchanged. Counts keep the logical table's convention: a row's
    `layer_count` is its enclosing template's multiplicity (the per-layer
    display ladder), while a leaf's FLOPs come from its timing row (own
    invocation count, rounded once). A collective's `layer_count` is its
    own `count_per_step`, as the old splice carried it.
    """
    # The post tree, walked on the analysis's own schedule: an encode step's
    # table is its encoder's stack.
    walked = result.analysis.program
    program = Program(resolved=walked.resolved, ir=result.post_tree,
                      schedule=walked.schedule)
    cost = result.cost
    law_cost = result.cost if result.laws is None else result.laws.cost
    # The unsharded column is one data-parallel replica's
    # (`Analysis.replica_cost`), the unit the rest of the table is per.
    full_cost = (result.replica_cost if result.laws is None
                 else result.laws.replica_cost)
    instances = result.instances
    point = Point(dict(result.point))

    def count_of(uid: int) -> int:
        return instance_count(instances, uid)

    timing = {
        r["uid"]: r for r in (per_op_timing or []) if r.get("uid") is not None
    }
    rows: List[Dict[str, Any]] = []

    def flops_of(layer: Any, uid: int, count: int) -> int:
        law = flops_law(layer, uid)
        return round(float(point.laws(book, (law,))[law]) * count)

    # The laws the rows' FLOPs are read at other points by: the layers
    # before they were read at this result's point.
    book = result.laws.book
    at_point = dict(result.point)

    def flops_law(layer: Any, uid: int) -> str:
        c = layer.values.get(uid)
        return book.write(value_of(c.flops) if c is not None else 0)

    def make_row(node: Any, depth: int, count: int, template: str) -> Dict[str, Any]:
        uid = node.uid
        leaf = timing.get(uid)
        # A leaf merges its whole timing row (`table_row`); an untimed row
        # is its identity, its subtree's FLOPs and its shapes.
        if leaf is not None:
            # full and per-rank share the timing row's own count base: both
            # are (cost of one call) × (the row's count of calls), so their
            # ratio is exactly the cost layer's sharding factor, whatever
            # the count's scope is (per-rank-resident copies under EP>1 or
            # global calls — it cancels). What must never happen is the old
            # bug: full counted at the template's layer count while per-rank
            # counted the fanout calls.
            own_count = whole_count(leaf.get("count_per_step"),
                                    f"timing row uid {uid}")
            flops_full = flops_of(full_cost, uid, own_count)
        else:
            flops_pr = flops_of(law_cost, uid, count)
            flops_full = flops_of(full_cost, uid, count)
        identity = TableIdentity(
            name=node.name, op=node.op, role=node.role or "",
            aggregation_id=node.aggregation_id, template_name=template,
            # A collective's count is its own `count_per_step`.
            layer_count=(own_count if leaf is not None and leaf.get("is_comm")
                         else count),
            depth=depth, uid=uid, flops_full=flops_full,
            flops_law={"full": flops_law(result.laws.replica_cost, uid),
                       "calls": own_count if leaf is not None else count,
                       "point": at_point})
        if leaf is not None:
            return table_row(leaf, identity)
        row: Dict[str, Any] = {
            **identity.columns(), "flops_per_rank": flops_pr,
            "flops_law": {**identity.flops_law,
                          "rank": flops_law(result.laws.cost, uid)}}
        # Untimed rows (wrappers, plumbing the overlay pruned) get no timing
        # shapes; read the uid-keyed layers instead of the old
        # aggregation_id-keyed override, with their laws in the workload
        # beside them so a re-read at another point has them too.
        law = _table_shape_law(uid)
        if law is not None:
            row.update(shape_at(law))
            row["shape_law"] = law
        return row

    def _table_shape_law(uid: int) -> Optional[Dict[str, Any]]:
        """`shape_columns`' four inputs as laws in the workload, off the
        layers before they were read at the result's point; None for a
        result read off no walk's laws."""
        if result.laws is None:
            return None
        shape = result.laws.shapes.values[uid]
        full = result.laws.full_shapes.values.get(uid)
        book = result.laws.book
        return {
            "point": dict(result.point),
            "in": [book.axes(one) for one in shape.ins or ()],
            "out": book.axes(shape.out) if shape.out is not None else None,
            "in_full": [book.axes(one) for one in
                        (full.ins if full is not None else None) or ()],
            "out_full": (book.axes(full.out) if full is not None
                         and full.out is not None else None)}

    def shape_at(law: Mapping[str, Any]) -> Dict[str, Any]:
        """A table row's shape columns at `result.point`, off the laws its
        `shape_law` states — the one road a re-read point takes."""
        axes = tuple(axis for key in ("in", "in_full")
                     for one in law[key] or () for axis in one)
        axes += tuple(axis for key in ("out", "out_full")
                      if law[key] is not None for axis in law[key])
        read = point.laws(book, axes)

        def values(one: Any) -> List[Any]:
            return [read[axis] for axis in one]

        return shape_columns(
            [values(one) for one in law["in"] or ()],
            values(law["out"]) if law["out"] is not None else None,
            [values(one) for one in law["in_full"] or ()],
            values(law["out_full"]) if law["out_full"] is not None else None)

    # A node the walk evaluated as a parent: its cost is its children's. A
    # launch that took every child leaves it empty on the post tree
    # (`materialize_tree` keeps the instance), and its row is then the sum
    # of no children — not its own cost entry, which the launch already
    # bills.
    parents = {node.uid for node in result.pre_program.nodes()
               if node.children}

    def visit(node: Any, depth: int, count: int, template: str) -> Tuple[int, int]:
        with node_context(node.display_path):
            row = make_row(node, depth, count, template)
            rows.append(row)
        sums = [visit(child, depth + 1, count, template)
                for child in node.children or ()]
        if sums or node.uid in parents:
            # A parent's FLOPs are its subtree's sum — never its own cost
            # entry times the template count, which mixes frames for fanout
            # leaves (see module docstring).
            row["flops_per_rank"] = sum(s[0] for s in sums)
            row["flops_full"] = sum(s[1] for s in sums)
            # A parent's FLOPs are its subtree's at every point.
            row.pop("flops_law", None)
        return row["flops_per_rank"], row["flops_full"]

    included = program.included_stack_names()
    visited: set = set()
    for stack in program.stacks:
        if stack.name not in included:
            continue
        for stage in stack.stages:
            if isinstance(stage, BlockStage):
                roots = tuple(stage.template.roots)
                # The template's own multiplicity, not a post-rewrite root's
                # instance count: a seam launch can take one of that root's
                # runs while the block still has every physical layer.
                count = stage.template.count if roots else 0
                header_at = len(rows)
                sums = [visit(root, 1, count, stage.template.name)
                        for root in roots]
                rows.insert(header_at, {
                    "name": stage.template.name,
                    "op": "<template>",
                    "role": "block_template",
                    "aggregation_id": "",
                    "template_name": stage.template.name,
                    # The invocation this header stands for (one per block
                    # group, where a template name may repeat).
                    "template_id": stage.template.meta.label,
                    # The block's short name, as the architecture diagram
                    # names it; `name` keeps the full template name.
                    "template_label": program.block_label(stage.template),
                    "layer_count": count,
                    "depth": 0,
                    "uid": None,
                    "flops_full": sum(s[1] for s in sums),
                    "flops_per_rank": sum(s[0] for s in sums),
                })
            elif isinstance(stage, OpStage):
                node = stage.node
                uid = node.uid
                leaf = timing.get(uid)
                c = cost.values.get(uid)
                # Pure plumbing at depth 0 (no FLOPs, no timing, no shape)
                # is noise; mirror the logical table's skip.
                if (c is None or not c.flops) and leaf is None:
                    continue
                visited.add(uid)
                visit(node, 0, count_of(uid), "")

    # Grafted TOP-LEVEL collectives (embedding all_reduce, vocab all_gather)
    # are siblings in `ir.top_level` but the spine names only authored ops,
    # so no stage covers them. Place each right after its recorded anchor's
    # WHOLE SUBTREE (`comm_meta`'s parent_aggregation_id — where the graft
    # itself says it lives), at the anchor's own depth. Nodes a stage
    # already drew (e.g. a top-level fused kernel that absorbed a
    # collective and took the authored op's OpStage) are not drawn twice.
    insert_after: Dict[str, int] = {}
    for node in program.ir.top_level:
        leaf = timing.get(node.uid)
        if leaf is None or not leaf.get("is_comm") or node.uid in visited:
            continue
        anchor = leaf.get("parent_aggregation_id") or ""
        idx = next(
            (i for i, r in enumerate(rows)
             if r.get("aggregation_id") == anchor),
            None,
        )
        if idx is None:
            raise ValueError(
                f"collective {node.display_path!r} is grafted beside "
                f"{anchor!r}, which the table does not draw")
        row = make_row(node, rows[idx]["depth"], count_of(node.uid), "")
        # Past the anchor's descendants, and past any collective already
        # placed on this anchor (keeps same-anchor comms in tree order).
        j = idx + 1
        while j < len(rows) and int(rows[j].get("depth") or 0) > int(rows[idx].get("depth") or 0):
            j += 1
        j = max(j, insert_after.get(anchor, idx) + 1)
        rows.insert(j, row)
        insert_after[anchor] = j
    return rows
