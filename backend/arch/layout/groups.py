"""ProcessGroupFamily — a world resolved to explicit process-group members.

The layout algebra asks every group question by membership, never by name
or by stride arithmetic: two worlds over the same ranks are `same`, one
whose groups lie inside another's `refines` it, and the partial-overlap
case gets a name (`overlaps`) that lowering must either plan for or reject
aloud. These are the only statements of those relations — the mesh hands
over spans (`mesh.spans`) and nothing else. A family is a few hundred
integers per deployment — smaller than the model graph — and is built
once per deployment and cached by the caller.

Nothing here reads `REGION_AXES` or any mesh table: construction stays
where it lives today (the mesh resolver); this type only holds the result
and answers relations on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Rank = int
Group = Tuple[Rank, ...]


@dataclass(frozen=True, eq=False)
class ProcessGroupFamily:
    """A family of mutually disjoint, identically sized ordered groups.

    `groups` are ordered tuples: member ORDER is the in-group coordinate,
    which `Block.replica_cells` patterns index into. Identical sizing is what lets
    one replica_cells pattern govern every group of the family.

    Equality is MEMBERSHIP: the set of ordered groups. The `id` is
    attribution only and the family listing order carries no meaning —
    layouts over `same()` families must compare equal."""

    id: str
    groups: Tuple[Group, ...]

    def __eq__(self, other) -> bool:
        return (isinstance(other, ProcessGroupFamily)
                and set(self.groups) == set(other.groups))

    def __hash__(self) -> int:
        return hash(frozenset(self.groups))

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("a ProcessGroupFamily holds at least one group")
        size = len(self.groups[0])
        if size == 0:
            raise ValueError("empty group")
        if any(len(g) != size for g in self.groups):
            raise ValueError("groups of one family must be identically sized")
        flat = [r for g in self.groups for r in g]
        if len(set(flat)) != len(flat):
            raise ValueError("groups of one family must be disjoint")

    @property
    def group_size(self) -> int:
        return len(self.groups[0])

    def ranks(self) -> frozenset:
        return frozenset(r for g in self.groups for r in g)

    def group_of(self, rank: Rank) -> Group:
        for g in self.groups:
            if rank in g:
                return g
        raise KeyError(rank)

    def coordinate(self, rank: Rank) -> int:
        return self.group_of(rank).index(rank)


def same(a: ProcessGroupFamily, b: ProcessGroupFamily) -> bool:
    """Every rank sits in the same ordered group under both families."""
    return set(a.groups) == set(b.groups)


def refines(a: ProcessGroupFamily, b: ProcessGroupFamily) -> bool:
    """Partition refinement: every group of `a` lies inside some group of
    `b`, over the SAME rank set — refinement between families of
    different scopes is undefined, not true."""
    if a.ranks() != b.ranks():
        return False
    outers = [frozenset(g) for g in b.groups]
    return all(any(frozenset(g) <= o for o in outers) for g in a.groups)


def joint_components(families, ranks) -> list:
    """Joint scopes: connected components of "shares a group in ANY of the
    families". Pairwise group intersection is NOT the scope — it usually
    degenerates to singletons and lets non-orthogonal interleavings pass."""
    parent = {r: r for r in ranks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fam_ in families:
        for g in fam_.groups:
            members = [r for r in g if r in parent]
            for r in members[1:]:
                parent[find(members[0])] = find(r)
    out: dict = {}
    for r in ranks:
        out.setdefault(find(r), set()).add(r)
    return list(out.values())


def orthogonal(a: ProcessGroupFamily, b: ProcessGroupFamily) -> bool:
    """Within each joint component, rank -> (coord_a, coord_b) is a
    bijection ONTO THE PRODUCT of the coordinate sets present there.
    Distinctness alone is not enough: a family is injective against itself
    yet not orthogonal to itself, and interleaved 2-cycles (A=(0,1)(2,3)..,
    B=(1,2)(3,4)..) produce only two distinct pairs over a six-rank
    component."""
    if a.ranks() != b.ranks():
        return False
    for comp in joint_components((a, b), a.ranks()):
        pairs = [(a.coordinate(r), b.coordinate(r)) for r in comp]
        a_coords = {p[0] for p in pairs}
        b_coords = {p[1] for p in pairs}
        if (len(set(pairs)) != len(pairs)
                or len(pairs) != len(a_coords) * len(b_coords)):
            return False
    return True


def overlaps(a: ProcessGroupFamily, b: ProcessGroupFamily) -> bool:
    """Related but none of the above — lowering must plan or reject."""
    if same(a, b) or refines(a, b) or refines(b, a) or orthogonal(a, b):
        return False
    return bool(a.ranks() & b.ranks())


def cross_replica_family(family: ProcessGroupFamily, replica_cells: Tuple[Tuple[int, ...], ...],
               *, id_suffix: str = "^cross") -> ProcessGroupFamily:
    """The derived family a replicated-cell contraction reduces over.

    For a block distribution with `num_shards` replica cells of size r,
    each parent group yields r derived groups — the i-th takes the i-th
    member of every cell — so a family of m parent groups derives m*r
    groups of size `num_shards`, disjoint and covering every rank. Members
    of one derived group hold pairwise DIFFERENT addends (one per shard),
    so a plain all_reduce over it (N = num_shards) hands every rank the
    complete value: no representative election, no intra-cell multicast,
    no communication event outside the lowering table."""
    num_shards = len(replica_cells)
    r = len(replica_cells[0])
    derived = [tuple(g[cell[i]] for cell in replica_cells)
               for g in family.groups for i in range(r)]
    out = ProcessGroupFamily(id=family.id + id_suffix, groups=tuple(derived))
    assert len(out.groups) == len(family.groups) * r
    assert out.group_size == num_shards
    return out
