"""The routed exchange's priced bytes, re-derived by simulating the routing.

The engine prices a MoE dispatch from a closed form: the exchange's total is
how many DISTINCT destination ranks a token's top-k picks reach
(hypergeometric), and the card's placement levels partition that total
across the fabrics (`comm_cost.routed_shares`, `comm_volume.RouteFanout`).
Reading those forms again proves nothing, so this takes the other route: it
draws tokens' top-k picks with numpy, places the experts on the
deployment's REAL member ranks (the mesh family's groups,
`layout.resolver.families_for`), cuts those ranks into nodes by the card's
own `gpus_per_node`, and COUNTS, per token:

  * the distinct destination ranks other than the sender — every one of them
    has to receive the token once, which is the exchange's total;
  * the distinct destination nodes other than the sender's — one RDMA send
    each, which is the inter level;
  * the rest — the destinations in the sender's own node, plus the forwards
    inside each remote node past the rank the RDMA copy landed on, which is
    the intra level.

Nothing here reads `RouteFanout`, `routed_shares` or the walk's
`comm_route_fanout`: the routing counts come from the checkpoint's own
config, the ranks from the mesh and the node size from the GPU sheet. The
counting is by token, not by level, so a term charged to nobody shows up as
a level that is too small — which is how the first form of this model was
caught, with the receiving node's fan-out missing.

Two tests, not one, because the two topologies are two properties and the
default suite runs only the first case of each (`tests/conftest.py`): a
route group inside ONE node, where the whole total is the exchange inside
the node, and one ACROSS nodes, where the partition is the claim. Beside
them the conservation the user sees — the phases of a row add up to the
per-rank bytes that row publishes, which is the number its popover explains.

What this does NOT check is the model's own assumption: the picks are drawn
uniformly from every target, which is what the engine assumes and so what
the simulation assumes too. A group-limited router (DeepSeek's `n_group` /
`topk_group`) concentrates them, and neither side models it — a known,
pessimistic gap written down in `docs/inference_cost_model.md`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.resolver import families_for
from backend.inference import gpu_specs as gs
from backend.inference.comm_cost import CollectiveSignature, Fabric
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.row_roofline import RowRoofline
from tests.snapshots import resolve_snapshot, snapshot_config

#: Tokens drawn per case. At 50k the standard error of the mean distinct
#: destination count is under a tenth of the 1% the comparison allows.
TOKENS = 50_000
#: Drawn in chunks: one `(chunk, targets)` float array at a time.
CHUNK = 10_000
SEED = 20261117

#: What a Hugging Face config calls the routing counts. The oracle's own
#: read of the checkpoint — the engine reads them through the spec's dim
#: bindings, which is the point of asking the config here instead.
_EXPERTS = ("n_routed_experts", "num_experts", "num_local_experts",
            "moe_num_experts")
_PICKS = ("num_experts_per_tok", "moe_topk", "num_experts_per_token")
_RESERVE = ("n_zero_experts", "num_zero_experts")


@dataclass(frozen=True)
class Case:
    """One deployment to check, and what it is here for."""
    model_id: str
    gpu_sku: str
    parallel: Dict[str, Any]
    phase: str = "prefill"
    #: The exchange SIGNATURES this deployment must show, by `(world, has
    #: fan-out)`, sorted. Named so a deployment that stops making the
    #: exchange this case is about fails instead of passing vacuously. A
    #: dispatch and its combine share one signature where they run on one
    #: world at one instance count — the signature is the price key, not the
    #: leg — so the legs are counted apart (`routed_legs`).
    exchanges: Tuple[Tuple[int, bool], ...] = ()
    #: How many ROWS carry a fan-out. A routed layer's dispatch and its
    #: combine are two, and the combine is one of them only while it is an
    #: exchange: with request-owner inclusion the inverse dispatch becomes
    #: an all-reduce (`layout.moe.combine`), which carries no fan-out
    #: (`materialize_comm.PhysicalCollective.__post_init__`) and is priced
    #: flat. Stated per case so which it is, is a fact on the record rather
    #: than an assumption.
    routed_legs: int = 2


#: Route groups that fit inside ONE node: the whole exchange is the one
#: inside the node, and its bytes are the exchange's total.
ONE_NODE = [
    pytest.param(Case("deepseek-ai/DeepSeek-V3", "H100-SXM",
                      dict(ep_moe=8, dp_attention=8),
                      exchanges=((8, True),)),
                 id="dsv3-ep8-one-node"),
    pytest.param(Case("Qwen/Qwen3-30B-A3B", "H100-SXM",
                      dict(ep_moe=8, dp_attention=8),
                      exchanges=((8, True),)),
                 id="qwen3-30b-ep8-one-node"),
    # Context parallelism beside expert parallelism, under the scheme that
    # exchanges heads (`CpScheme.ULYSSES`): those exchanges are not routed
    # and must stay flat, beside the routed ones on the same world.
    pytest.param(Case("moonshotai/Kimi-K2.5", "H100-SXM",
                      dict(cp=8, ep_moe=8, cp_scheme="ulysses"),
                      exchanges=((8, False), (8, False), (8, True))),
                 id="kimi-cp8-ep8-ulysses"),
]

#: Route groups that span nodes: the partition across the two fabrics is
#: the claim, and the sum of the two is still the exchange's total.
CROSS_NODE = [
    # Two nodes of four, because the PCIe card's nodes hold four GPUs: the
    # SAME group of eight that is one level above is two levels here.
    pytest.param(Case("Qwen/Qwen3-30B-A3B", "H100-PCIe",
                      dict(ep_moe=8, dp_attention=8),
                      exchanges=((8, True),)),
                 id="qwen3-30b-pcie-two-nodes"),
    # Four nodes of eight.
    pytest.param(Case("deepseek-ai/DeepSeek-V3", "H100-SXM",
                      dict(ep_moe=32, dp_attention=32),
                      exchanges=((32, True),)),
                 id="dsv3-ep32-four-nodes"),
    # Expert tensor parallelism beneath the route group: the group's eight
    # members are two ranks apart, so four of them share a node.
    pytest.param(Case("deepseek-ai/DeepSeek-V3", "H100-SXM",
                      dict(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2,
                           ep_moe=8, dp_attention=8),
                      exchanges=((8, True),)),
                 id="dsv3-tme2-ep8-two-nodes"),
]


def _find(config: Any, names: Sequence[str]) -> Optional[int]:
    """The first of `names` this config states, nested configs included
    (a multimodal checkpoint keeps the language model's under
    `text_config`)."""
    if isinstance(config, Mapping):
        for name in names:
            value = config.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        for value in config.values():
            found = _find(value, names)
            if found is not None:
                return found
    return None


def routing_counts(model_id: str) -> Tuple[int, int, int]:
    """`(picks, targets, placed)` off the checkpoint's own config."""
    config = snapshot_config(model_id)
    placed = _find(config, _EXPERTS)
    picks = _find(config, _PICKS)
    assert placed and picks, f"{model_id}: config states no routing counts"
    return picks, placed + (_find(config, _RESERVE) or 0), placed


def placement(parallel: ParallelConfig, gpu_sku: str, phase: str
              ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """`(member ranks, their node index)` for the route group, from the
    mesh and the card's node size alone."""
    members = families_for(parallel, phase)["ep_moe"].groups[0]
    per_node = max(gs.gpus_per_node(gs.get_gpu(gpu_sku)), 1)
    nodes = sorted({rank // per_node for rank in members})
    return tuple(members), tuple(nodes.index(r // per_node) for r in members)


def simulated_shares(*, picks: int, targets: int, placed: int,
                     node_of_member: Sequence[int],
                     sender: int = 0) -> Dict[str, float]:
    """`{fabric: bytes per payload byte}`, by counting what the exchange
    really moves for each of `TOKENS` drawn tokens.

    A member holds `placed / members` of the weighted experts, in order, so
    a drawn target names a member or (past the weighted ones) nothing. Per
    token: every distinct destination member other than the sender has to
    receive it, which is the TOTAL; one RDMA send goes to each distinct
    destination node other than the sender's, which is `inter`; the rest is
    NVLink traffic, `intra`.
    """
    members = len(node_of_member)
    per_member = placed // members
    assert per_member * members == placed, "experts do not divide the group"
    nodes = max(node_of_member) + 1
    # target -> member index, or `members` for a target that holds no
    # weights (a zero-compute reserve: it receives nothing).
    where = np.full(targets, members, dtype=np.int64)
    for member in range(members):
        where[member * per_member:(member + 1) * per_member] = member
    # member -> node index, with the no-destination column mapped to the
    # no-node column.
    node_of = np.append(np.asarray(node_of_member, dtype=np.int64), nodes)
    rng = np.random.default_rng(SEED)
    total = inter = 0
    drawn = 0
    while drawn < TOKENS:
        n = min(CHUNK, TOKENS - drawn)
        # `picks` distinct targets per token, uniformly: the smallest
        # `picks` of a row of uniform keys is a uniform subset.
        draw = np.argpartition(rng.random((n, targets)), picks - 1,
                               axis=1)[:, :picks]
        dest = where[draw]
        rows = np.arange(n)[:, None]
        hit_member = np.zeros((n, members + 1), dtype=bool)
        hit_member[rows, dest] = True
        total += int(hit_member[:, :members].sum()
                     - hit_member[:, sender].sum())
        hit_node = np.zeros((n, nodes + 1), dtype=bool)
        hit_node[rows, node_of[dest]] = True
        inter += int(hit_node[:, :nodes].sum()
                     - hit_node[:, node_of_member[sender]].sum())
        drawn += n
    copies = float(Fraction(picks * placed, targets))
    return {"intra": (total - inter) / drawn / copies,
            "inter": inter / drawn / copies}


def priced(case: Case):
    """What the engine prices for this deployment: for each exchange, its
    per-level share of the payload and the bytes its row publishes; and
    every other collective as `(op, carries a fan-out)`."""
    parallel = ParallelConfig(**case.parallel)
    gpu = gs.get_gpu(case.gpu_sku)
    result = analyze(
        resolved_spec=resolve_snapshot(case.model_id), parallel=parallel,
        batch_size=1, avg_input_tokens=1024, avg_output_tokens=0, gpu=gpu,
        phase=case.phase, breakdown=False, comm_latency_s=0.0,
        intra_bw_bytes_per_sec=gs.peak_intra_bw_bytes_per_sec(gpu),
        inter_bw_bytes_per_sec=gs.peak_inter_bw_bytes_per_sec(gpu),
        peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3e12)
    fabric = Fabric.of(gpu)
    # One entry per collective SIGNATURE: a layer's exchange is one
    # collective however many layers run it, and its shares are ratios of
    # its own payload.
    exchanges: Dict[CollectiveSignature, Any] = {}
    others: set = set()
    legs = 0
    for row in result.per_op_timing:
        sig = CollectiveSignature.of(row)
        if sig is None:
            continue
        if sig.route_fanout is not None and int(row["count_per_step"]) > 1:
            # A routed leg of the repeated layers: the dispatch and its
            # combine, one row each.
            legs += 1
        if sig.op != "all_to_all":
            others.add((sig.op, sig.route_fanout is not None))
            continue
        if sig in exchanges:
            continue
        payload = float(row["comm_tensor_bytes"]) * sig.count
        _t, phases = RowRoofline.comm(
            sig, tensor_bytes=float(row["comm_tensor_bytes"]), fabric=fabric)
        exchanges[sig] = ({ph["fabric"]: ph["bytes_per_rank"] / payload
                           for ph in phases},
                          float(row["comm_bytes_per_rank"]) / payload)
    return ([(sig, levels, total)
             for sig, (levels, total) in exchanges.items()],
            sorted(others), legs)


@pytest.fixture(scope="module")
def engine():
    cache: Dict[Any, Any] = {}

    def of(case: Case):
        key = (case.model_id, case.gpu_sku, repr(sorted(case.parallel.items())))
        if key not in cache:
            cache[key] = priced(case)
        return cache[key]
    return of


def check(case: Case, engine, *, nodes: int) -> None:
    """Every exchange this deployment prices, against the simulation."""
    exchanges, others, legs = engine(case)
    assert sorted((sig.world, sig.route_fanout is not None)
                  for sig, _levels, _total in exchanges) == sorted(
                      case.exchanges), (
        f"{case.model_id}: the exchanges this deployment makes are not the "
        f"ones the case names: "
        f"{[(s.op, s.world, s.route_fanout) for s, _l, _t in exchanges]}")
    # The dispatch and its combine, counted: a combine that became an
    # all-reduce is not an exchange and carries no fan-out, so a case where
    # that happens says `routed_legs=1` instead of passing quietly.
    assert legs == case.routed_legs, (
        f"{case.model_id}: {legs} rows carry a fan-out, not "
        f"{case.routed_legs} — the combine either is or is not an exchange "
        "here, and the case has to say which")
    # Only an exchange dedups; a reduction visits every member whatever the
    # routing, so nothing else may carry a fan-out.
    assert [op for op, has in others if has] == [], others
    picks, targets, placed = routing_counts(case.model_id)
    members, node_of_member = placement(
        ParallelConfig(**case.parallel), case.gpu_sku, case.phase)
    assert max(node_of_member) + 1 == nodes, (
        f"{case.model_id} on {case.gpu_sku}: the route group spans "
        f"{max(node_of_member) + 1} nodes, not the {nodes} this case is for "
        f"(members {members})")
    for sig, levels, total in exchanges:
        # Every phase of every exchange adds up to the bytes its own row
        # publishes: the cell and the popover under it are one number.
        assert sum(levels.values()) == pytest.approx(total, rel=1e-9), (
            f"{case.model_id}: the phases of a {sig.op} of {sig.world} sum "
            f"to {sum(levels.values())} and its row publishes {total}")
        if sig.route_fanout is None:
            # Ulysses: every payload row has its own destination, so the
            # flat ring's share and nothing else.
            assert total == pytest.approx((sig.world - 1) / sig.world,
                                          rel=1e-9), (
                f"an unrouted exchange of {sig.world} is not flat")
            continue
        assert (picks, targets, placed) == (
            sig.route_fanout.picks, sig.route_fanout.targets,
            sig.route_fanout.placed), (
            f"{case.model_id}: the walk's fan-out is not the checkpoint's "
            f"routing counts")
        assert len(members) == sig.world, (members, sig.world)
        want = simulated_shares(picks=picks, targets=targets, placed=placed,
                                node_of_member=node_of_member)
        for fabric, share in sorted(levels.items()):
            assert share == pytest.approx(want[fabric], rel=0.01), (
                f"{case.model_id} {fabric}: priced {share} against the "
                f"simulated {want[fabric]}")
        # A level the engine does not price must be nothing to simulate.
        for fabric in set(want) - set(levels):
            assert want[fabric] == pytest.approx(0.0, abs=1e-9), (
                f"{case.model_id}: the simulation moves {want[fabric]} on "
                f"the {fabric} fabric and the engine prices no such level")


@pytest.mark.parametrize("case", ONE_NODE)
def test_a_route_group_inside_one_node(case, engine):
    """The whole exchange happens inside the node, so its one level carries
    the exchange's total — and the dedup is the whole of the saving."""
    check(case, engine, nodes=1)


@pytest.mark.parametrize("case", CROSS_NODE)
def test_a_route_group_across_nodes(case, engine):
    """Two levels: one RDMA send per distinct destination node, and every
    other copy on NVLink. The sum is the same total the one-node case
    pays — a worse topology moves copies to the slower fabric, it does not
    remove them."""
    check(case, engine, nodes=2 if case.gpu_sku == "H100-PCIe"
          else (4 if case.parallel.get("ep_moe") == 32 else 2))


def test_the_simulation_sees_the_dedup_and_the_forwarding():
    """The oracle's own check, on the two things it has to be able to see.

    The dedup: eight picks over eight ranks reach fewer than eight distinct
    destinations, so the simulated total is well under the flat ring, and
    at one pick per token it IS the flat ring.

    The forwarding: with the same group split over two nodes the simulated
    total is UNCHANGED (the copies are the same; only their fabric moves),
    which is the property a model that dropped the receiving node's
    fan-out broke.
    """
    one_node = [0] * 8
    two_nodes = [0] * 4 + [1] * 4
    flat = simulated_shares(picks=1, targets=256, placed=256,
                            node_of_member=one_node)
    assert sum(flat.values()) == pytest.approx(7 / 8, rel=0.01)
    deduped = simulated_shares(picks=8, targets=256, placed=256,
                               node_of_member=one_node)
    assert sum(deduped.values()) == pytest.approx(0.5791, rel=0.01)
    assert sum(deduped.values()) < sum(flat.values()) * 0.7
    split = simulated_shares(picks=8, targets=256, placed=256,
                             node_of_member=two_nodes)
    assert sum(split.values()) == pytest.approx(sum(deduped.values()),
                                                rel=0.01)
    assert split["inter"] > 0.0
    assert split["intra"] < deduped["intra"]
