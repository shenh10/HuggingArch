"""The identity a fused launch carries, below the transform that makes it.

A `FusedNode` is what one launch IS — a uid, an op, the origins it stands
for and its boundary — and nothing about how a plan matched it. It lives
here rather than in `fusion` because the layers a rewrite crosses have to
recognise a launch while `fusion` itself reads them: `traffic` asks whether
the node it is pricing is a launch (and which of its members wrote the
tensor on a port), and `fusion` asks `traffic` what format a reader eats.
With the identity one module below both, neither has to import the other
inside a function body to keep the package loadable (CLAUDE.md §2.7).

`FusedNode` satisfies `node.Identified` and not `node.Evaluable`: it is
created after the walk, out of nodes the walk already evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from backend.arch.ops import attention_kernel


@dataclass(frozen=True)
class FusedNode:
    """One fused op, standing in for the nodes it absorbed.

    Carries no cost and no shape. Those come from the layers, rebased across
    the `Rewrite` — which is the whole point: a node that had to carry its own
    would need the transform to know how each kind of value combines.

    It owns no TENSOR either. Its weights are still the originals' — three
    projections fused into one kernel still read three matrices from disk, each
    with its own dtype — which is why `Formats` keys on storage nodes a
    compute-graph rewrite never touches, and why `traffic.derive` reaches them
    through the `Rewrite` rather than through this node.

    So it satisfies `node.Identified` and not `node.Evaluable`: an identity is
    the whole of what a post-walk correspondence needs a new node to carry.
    """

    uid: int
    op: str
    origins: Tuple[int, ...]
    #: `io_mapping` operand → the origin uid (and port) it resolves to
    ports: Mapping[str, Tuple[int, str]] = field(default_factory=dict)
    #: Origin uid -> output port exposed by the fused kernel. The empty string
    #: is the primary output; named and positional outputs stay explicit.
    output_ports: Mapping[int, str] = field(default_factory=dict)
    #: (Origin uid, origin port) -> published port, for a member whose
    #: SECONDARY output the launch exposes (`post: split.post`). The main
    #: output stays in `output_ports`, so one member may publish several
    #: tensors: `hc_split_sinkhorn` writes `pre` for the reduce beside it
    #: and `post` / `comb` for the post-mix outside, and the Ascend `HcPre`
    #: launch publishes the last two. Empty for a plan whose targets are
    #: all `<node>.out` — every plan before this field existed.
    secondary_ports: Mapping[Tuple[int, str], str] = field(
        default_factory=dict)
    role: Optional[str] = None
    name: str = ""
    template: str = ""
    display_path: str = ""
    stack_name: str = ""
    component_path: Tuple = ()
    component_scope: Optional[str] = None
    boundary_role: Optional[str] = None
    children: Tuple = ()
    #: A fused kernel is a leaf standing for its origins, never a component.
    #: Part of the `Evaluable` protocol, and consumers do read it: a walk
    #: that asks "is this a wrapper whose cost is its children's" gets a
    #: straight no rather than an AttributeError.
    is_component: bool = False
    provenance: str = "fusion"
    #: Whether the plan merges sibling projections of one input into one
    #: launch (`parallel:`, the merged column / QKV plans): the origins'
    #: matrices are then the partitions of ONE weight, concatenated along
    #: the output, rather than weights of their own.
    merges_siblings: bool = False
    #: Whether this launch writes its output into a buffer of its own
    #: rather than rewriting its input in place — its plan's
    #: `allocates_output`. `_passed_through` reads it.
    allocates_output: bool = False
    #: The scale sidecar's storage tile on the quantised output record, as
    #: (rows, groups), when the launch pads it (the plan's
    #: `output_scale_tile`); None when the record is exactly tokens ×
    #: groups. `traffic._record_width` reads it to price the padded
    #: scale bytes rather than the generic per-group law.
    output_scale_tile: Optional[Tuple[int, int]] = None
    #: The dtype this launch writes each output at where its plan states one
    #: (`output_dtype`), keyed like `output_ports`' values: `""` the primary
    #: (MiniMax-M3's fp32 block scores), a name a secondary port (sglang's
    #: fp32 GDN gates). A port not listed: the edge's width, or the
    #: registered op's declaration.
    output_dtype: Mapping[str, str] = field(default_factory=dict)
    #: A fused kernel has no primitive registration: consumers that read a
    #: leaf's `declaration` get None here and turn to the span's signature.
    declaration: Any = None
    #: A fused kernel is a collective when one of the ops it absorbed is —
    #: `allreduce_fused_rmsnorm` moves bytes across ranks exactly as the
    #: `all_reduce` inside it did. Carried from the origins' registrations at
    #: construction, so a consumer asks the node and not a metadata key.
    is_comm_op: bool = False
    #: A fused kernel runs on dispatched tokens when the ops it absorbed
    #: did — carried from the origins at construction (they are siblings,
    #: so they agree), because the node has no parent chain for
    #: `Program.routed_uids` to walk.
    in_dispatch_zone: bool = False
    #: An axis-aware op's authored axis, and a permute's axis order. Always
    #: None on a launch: a fused kernel is a leaf standing for its origins,
    #: and the axes worked on are theirs. Declared for the same reason
    #: `parent` is — so a report reads `node.axis` uniformly over fused and
    #: tree nodes instead of probing for the attribute.
    axis: Optional[int] = None
    axes: Optional[Tuple[int, ...]] = None
    #: Always None: a fused node is chain-terminal, carrying its dispatch-zone
    #: flag from its origins (above) rather than inheriting it. Declared so
    #: `Program.in_zone` walks `node.parent` uniformly over fused and tree
    #: nodes instead of reaching for a getattr default.
    parent: Optional["SpecIRNode"] = None  # noqa: F821 - forward ref only; SpecIRNode lives in spec_tree, kept out of eval (see eval/node.py boundary), and this field is always None
    #: The match's EXTERNAL input refs — each member's inputs minus the edges
    #: that stay inside the match, one entry per distinct outside producer.
    #: Real `Ref` objects to nodes that survive the rewrite, so
    #: `traffic.derive` walks a fused input's producer chain exactly as it
    #: walks any other node's — a broadcast feeding a fusion gets its
    #: fan-out division, absorbed MLA included. Aligned index-for-index
    #: with the boundary `Shape.ins` that `rebase_analysis` installs; both
    #: come from `_external_inputs`, so they cannot drift.
    inputs_obj: Tuple = ()
    #: The member whose output IS the fused op's output — the last, in
    #: program order, of the members the outside still reads. The other
    #: outside-read members become boundary ports: a fused add+rmsnorm
    #: still lands the residual in HBM.
    sink: Optional[int] = None
    #: The members that ran in the PREVIOUS iteration of the layer loop: a
    #: launch matched across a seam (`fusion.Seams`) — the residual add that
    #: closes layer i and the norm that opens layer i+1 — stands in the
    #: reader's layer, its graph inputs the reader's formal, and runs once
    #: per seam (`Rewrite.taken`). Empty for a launch within one iteration.
    previous_iteration: FrozenSet[int] = frozenset()
    #: The origin this launch runs with: its iterations index that origin's
    #: (`Program.occurrence_layers`), and it stands in that origin's block
    #: and repeat scope. For a launch across a seam, the reading layer's
    #: member that reads the formal; otherwise the first origin.
    home_uid: Optional[int] = None
    #: The plan's input ports each of `inputs_obj` fills
    #: (`NaivePattern.input_ports`), aligned index for index, in the plan's
    #: port order: `(k, v)` where one tensor is both (an MLA latent), its
    #: ports in plan order too, and a port no tensor fills (an optional
    #: mask) is absent. Which operand is which is this, not where an
    #: operand stands.
    operand_ports: Tuple[Tuple[str, ...], ...] = ()
    @property
    def home(self) -> int:
        """The origin this launch runs with (`home_uid`; the first origin
        for a launch built without one)."""
        return self.origins[0] if self.home_uid is None else self.home_uid

    def published_port(self, origin: int, port: str = "") -> Optional[str]:
        """The port this launch publishes an origin's output at — `""` the
        primary — or None when it publishes no such tensor. `port` is the
        ORIGIN's port the reader asked for (`""` / `"out"` its main output,
        a name one of its secondary outputs)."""
        if port in ("", "out", None):
            return self.output_ports.get(origin)
        return self.secondary_ports.get((origin, port))

    def publishers(self) -> Tuple[Tuple[int, str, str], ...]:
        """Every `(origin uid, origin port, published port)` this launch
        exposes, main outputs first: the one iteration a consumer that asks
        "which member's tensor is on port X" needs."""
        return (
            *((uid, "", port) for uid, port in self.output_ports.items()),
            *((uid, origin, port)
              for (uid, origin), port in self.secondary_ports.items()),
        )

    @property
    def aggregation_id(self) -> str:
        return (
            f"{self.template}|{self.display_path}"
            if self.template else self.display_path
        )


def score_matrix_owner(node: Any, program: Any, shapes: Any) -> Any:
    """The node whose shape row holds the score matrix `node` works on:
    `node` itself, or for a fused kernel the attention kernel it absorbed,
    whose row the rewrite keeps (`rebase_shapes`) — the kernel's boundary
    is cut by sequence on both sides under Ulysses while the matrix holds
    every query. None for a fused kernel that absorbed no attention kernel."""
    if not isinstance(node, FusedNode):
        return node
    return next((member for member in map(program.by_uid, node.origins)
                 if member.uid in shapes.values
                 and attention_kernel(member.op) is not None), None)
