"""ExpectedRoutedLoad — the edge evidence of a routed tensor's rows.

Inside a routed zone a tensor is `[n_expert_copies, M, d]`: the expert axis
a static layout cuts over the route group, and `M`, the MEAN rows one expert
receives. How many rows each expert really receives is runtime-determined
and uneven — expressible as no static layout. The quantity lives HERE, on the
edge, never in layout and never in equality, and it is two numbers: the mean,
whose per-rank values sum to the routed total, and the busiest rank's load
(`busiest_rank_load_factor`, from the measured expert-level `ep_load_factor`).
No per-expert distribution is modelled; a kernel-boundary consumer that needs
a padded integer geometry derives it from the same `M` under its own padding
convention — the ceil layering precedent.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import sympy as sp

from backend.arch.eval.dim import exact, require, same_law, value_of

from .comm_volume import RouteFanout
from .groups import ProcessGroupFamily, same
from .layout import Block


@dataclass(frozen=True)
class ExpectedRoutedLoad:
    #: The route group, and the expert collection axis's cut over it. Both
    #: None without expert parallelism: every rank holds every expert, no
    #: exchange happens, and the load is still the mean the rows axis means.
    group: Optional[ProcessGroupFamily]
    expert_shard_axis: Optional[Block]
    #: R_cell = R_global / C. Exact rational value; a `Dim` when the walk
    #: tracked the token axes it came from, so what it moves can be read at
    #: another batch.
    routed_rows_in_cell: Any
    num_experts: int          # E
    load_assumption_id: str
    #: The dispatch's top-k fan-out (`comm_volume.RouteFanout`): how many
    #: DISTINCT destinations a token's picks reach, which is how many copies
    #: of it the exchange sends. Required: a routed tensor's rows ARE one
    #: token's picks, so how those picks collapse onto destinations is part
    #: of what the rows axis means, and the exchange that reads it
    #: (`materialize_rows.exchange_row`) has nowhere else to get it.
    fanout: RouteFanout
    #: rho = max/mean EXPERT load (`Workload.ep_load_factor`), the
    #: number a routing trace measures — independent of how the experts are
    #: placed. What a rank carries is `busiest_rank_load_factor`.
    expert_load_factor: Any = 1
    #: Static row cuts preserved by dispatch may have to be gathered before
    #: expert tensor parallelism. The route measure stays unchanged; this is
    #: the number of physical copies each expert kernel consumes.
    row_replication: int = 1
    #: The rows axis as the dispatch left it (`routed_rows_layout`): what a
    #: gather inside the zone stacks copies of, and what the inverse
    #: dispatch takes the rows back to. None only for evidence built outside
    #: a dispatch (a unit probe); a combine refuses it.
    dispatched_rows: Optional[Any] = None

    def __post_init__(self) -> None:
        if (self.group is None) != (self.expert_shard_axis is None):
            raise ValueError("route group and expert collection cut come "
                             "together: both set (expert parallel) or "
                             "both None (every rank holds every expert)")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if not isinstance(self.fanout, RouteFanout):
            raise ValueError(
                "ExpectedRoutedLoad.fanout is the dispatch's top-k fan-out "
                f"(RouteFanout); got {self.fanout!r}")
        if self.fanout.placed != self.num_experts:
            raise ValueError(
                f"the fan-out places {self.fanout.placed} experts and the "
                f"bank holds {self.num_experts}: one bank, one count")
        if self.group is not None:
            self._check_route_group()
        if self.routed_rows_in_cell < 0:
            raise ValueError("routed_rows_in_cell must be non-negative")
        if not isinstance(self.row_replication, int) \
                or self.row_replication < 1:
            raise ValueError("row_replication must be a positive integer")
        # A condition on the point (`dim.require`), checked where a point is
        # read: the hottest expert carries at least the mean (its declared
        # floor, `Workload.ep_load_factor`) and no more than every row.
        rho = value_of(self.expert_load_factor)
        bounds = sp.And(sp.Ge(exact(rho), 1),
                        sp.Le(exact(rho), self.num_experts))
        require(bounds, lambda at: ValueError(
            f"expert load factor {at(rho)} is outside "
            f"[1, {self.num_experts}]: the hottest of {self.num_experts} "
            "experts carries at least the mean and no more than all of "
            "their rows"))

    def _check_route_group(self) -> None:
        # Conservation precondition: the expert collection axis must be a
        # singleton-cell Block covering exactly the route group — same
        # group, no replication. Replicated-expert deployments break the
        # conservation identity and are rejected at dispatch construction
        # (support, when wanted, is an explicit plan, not a layout bend).
        if self.expert_shard_axis.replication_degree != 1:
            raise ValueError("expert collection axis must be singleton-cell "
                             "(expert replication is a plan, not a layout)")
        if not same(self.expert_shard_axis.group, self.group):
            raise ValueError("expert collection axis must be cut over the "
                             "route group itself — same family, not merely "
                             "the same rank set")
        if self.num_experts % self.group.group_size:
            raise ValueError("experts must divide evenly over the route "
                             "group — a non-divisible expert count is a "
                             "named policy question, not a fraction")

    @property
    def local_experts(self) -> int:
        """Experts one rank holds: E over the route group, or all of them."""
        if self.group is None:
            return self.num_experts
        return self.num_experts // self.group.group_size

    @property
    def expected_rows_per_expert(self) -> Any:
        """M = R_cell / E — exact rational; ceil belongs to the kernel
        boundary (benchmark identity), never folded back into the mean."""
        return (self.routed_rows_in_cell / self.num_experts
                * self.row_replication)

    @property
    def routed_rows_per_expert(self) -> Any:
        """The unreplicated route measure for one expert."""
        return self.routed_rows_in_cell / self.num_experts

    @property
    def busiest_rank_load_factor(self) -> Any:
        """Busiest rank's rows over the mean rank's, from the expert-level rho.

        One expert carries `rho x M`; the rows are conserved, so the other
        `E - 1` share `(E - rho) x M`. The rank holding the hot expert holds
        `k - 1` of the others, `k` = `local_experts`:

            (rho + (k - 1)(E - rho)/(E - 1)) / k

        It is 1 at rho = 1 or k = E (every rank holds every expert: no rank
        is busier), rho at k = 1, and EP at rho = E (the rank holds every
        row). A single hot expert is the least a measured rho implies; hot
        experts sharing a rank would be a placement trace, not this number."""
        e, k = self.num_experts, self.local_experts
        if e == 1 or k == e:
            return sp.Integer(1)
        rho = exact(value_of(self.expert_load_factor))
        return (rho + (k - 1) * (e - rho) / sp.Integer(e - 1)) / k

    @property
    def routed_row_capacity_per_rank(self) -> Any:
        """Rows the busiest rank receives: the mean times its load factor."""
        return (self.expected_rows_on_rank(self.local_experts)
                * self.busiest_rank_load_factor)

    def expected_rows_on_rank(self, local_experts: int) -> Any:
        """Per-rank routed rows = M x local expert count. The token axis's
        routed extent explicitly couples the expert collection axis — not a
        defect, the definition of routing."""
        return self.expected_rows_per_expert * local_experts

    def is_measure_preserving(self) -> bool:
        """Measure preservation: sum of per-rank means over one route group
        equals R_cell. Local experts per rank = E / |group| under the
        singleton-cell precondition."""
        n = self.group.group_size if self.group is not None else 1
        # One law, not one number at some points: the rows a cell routes
        # are conserved at every point or not at all.
        return same_law(self.routed_rows_per_expert * self.local_experts * n,
                        self.routed_rows_in_cell)

    def replicate_rows(self, factor: int) -> "ExpectedRoutedLoad":
        """Record a static row gather without changing the route measure."""
        if not isinstance(factor, int) or factor < 1:
            raise ValueError("row replication factor must be positive")
        return replace(self, row_replication=self.row_replication * factor)

    def as_dispatched(self) -> "ExpectedRoutedLoad":
        """The rows as the dispatch left them: a row gather undone."""
        return replace(self, row_replication=1)
