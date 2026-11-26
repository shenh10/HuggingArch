"""Collective cells measured over a group of processes, one per card.

`kernel_bench/ARCHITECTURE.md`, "How a collective is measured", is the
contract. What is checked here without a GPU:

* a cell whose group needs more cards than the env has, or that the card's
  sheet places across nodes, is refused with the reason, by the one gate
  every provider takes (`Provider.refusals`);
* how many cards a run can drive has one answer for the plan and the run
  (`BoxEnvs.devices`: the cards it is shown, capped by the box's), so a
  single-card queue's plan puts a pair's cells in its gaps;
* each signature's oracle agrees with an exchange simulated rank by rank and
  DISagrees with a launch that skipped the exchange — the failure the old
  single-process floor had.

With one card: the clock meets its group at the points it promised.
With two (`KERNEL_BENCH_MULTI_GPU=1`, both cards idle): a real run of every
provider.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kernel_bench.box import BoxEnvs, BoxProfileError, load_box_profile
from kernel_bench.env import Env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import all_providers, candidates, dependency_manifest, discover

ENVS_DIR = str(ROOT / "kernel_bench" / "envs")
DUAL_4090 = ROOT / "kernel_bench" / "envs" / "boxes" / "rtx4090-dual.toml"
_BF16 = {"activation": "bf16", "output": "bf16"}


def _job(signature: str, dims: dict, phase: str = "decode") -> dict:
    return {"signature": signature, "op": signature, "dims": dims,
            "dtype_env": dict(_BF16), "phase": phase, "model": "test",
            "physical_key": [signature, json.dumps(dims, sort_keys=True),
                             json.dumps(_BF16, sort_keys=True), phase]}


def _inv(signature="all_reduce", ranks=2, stride=1, elems=4096) -> Invocation:
    return Invocation.from_job(_job(signature, {
        "n_comm_elems": elems, "n_comm_ranks": ranks, "n_comm_stride": stride}))


def _provider(kernel_impl: str):
    discover()
    return next(p for p in all_providers() if p.kernel_impl == kernel_impl)


def _env(devices) -> Env:
    return Env("2.9", True, "RTX 4090", "RTX-4090", 1008.0, sm="sm_89",
               libs={"torch": "2.9"}, library="torch", devices=devices)


# --- the gate ---------------------------------------------------------------

def test_a_group_larger_than_the_env_is_refused_and_says_why():
    nccl = _provider("torch_nccl_all_reduce")
    why = list(nccl.refusals(_inv(ranks=4), _env(2)))
    assert why == ["a group of 4 ranks needs 4 cards; this env has 2"]
    assert nccl.accepts(_inv(ranks=2), _env(2))


def test_a_group_the_sheet_places_across_nodes_is_refused():
    # Which node a member is in is `comm_cost.members_per_node`'s answer:
    # the 4090's sheet has 8 per node, so 2 ranks 8 apart are two nodes and
    # 2 ranks 2 apart are one — the same fabric as adjacent cards.
    nccl = _provider("torch_nccl_all_reduce")
    assert list(nccl.refusals(_inv(ranks=2, stride=8), _env(2))) == [
        "a group of 2 ranks at stride 8 spans 2 nodes of RTX-4090; one box "
        "is one node"]
    assert nccl.accepts(_inv(ranks=2, stride=2), _env(2))


def test_an_env_that_states_no_card_count_judges_nothing():
    assert _provider("torch_nccl_all_reduce").accepts(_inv(ranks=512), _env(None))


def test_a_payload_that_does_not_split_is_declined_before_inputs():
    with pytest.raises(NotImplementedError, match="do not split over 2"):
        _provider("torch_nccl_all_gather").impl_id(_inv(ranks=2, elems=4097))
    # The all-reduce hands no rank a share: any payload is a launch.
    assert _provider("torch_nccl_all_reduce").impl_id(
        _inv(ranks=2, elems=4097)) == "torch_nccl_all_reduce"


# --- one answer to "how many cards", for the plan and the run ----------------

def _box(environ=None):
    return BoxEnvs.of(load_box_profile(DUAL_4090), ENVS_DIR,
                      environ=environ if environ is not None else {})


def test_a_single_card_queues_plan_puts_a_pairs_cells_in_its_gaps():
    one = _box({"CUDA_VISIBLE_DEVICES": "1"})
    assert one.devices == 1
    chosen, why = candidates(_inv(ranks=2), one.envs())
    assert not chosen
    assert "needs 2 cards; this env has 1" in why["torch_nccl_all_reduce"]
    both = _box({"CUDA_VISIBLE_DEVICES": "0,1"})
    assert both.devices == 2
    chosen, _ = candidates(_inv(ranks=2), both.envs())
    assert "torch_nccl_all_reduce" in [p.kernel_impl for p in chosen]


def test_the_dual_4090_box_names_a_quad_as_a_gap():
    chosen, why = candidates(_inv(ranks=4), _box().envs())
    assert not chosen
    assert "needs 4 cards; this env has 2" in why["torch_nccl_all_reduce"]


def test_a_run_is_never_shown_more_cards_than_the_box_has():
    assert _box({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}).devices == 2


def test_cuda_reads_the_card_list_up_to_its_first_invalid_entry():
    assert _box({"CUDA_VISIBLE_DEVICES": "-1"}).devices == 0
    assert _box({"CUDA_VISIBLE_DEVICES": "1,-1,0"}).devices == 1


def test_the_two_card_counts_judge_a_plan_differently():
    manifest = dependency_manifest()
    assert (_box({"CUDA_VISIBLE_DEVICES": "1"}).premises(manifest)
            != _box({"CUDA_VISIBLE_DEVICES": "0,1"}).premises(manifest))


def _profile(tmp_path, extra: str):
    path = tmp_path / "box.toml"
    path.write_text(f'name = "dual-rtx4090-test"\n{extra}')
    return load_box_profile(path)


def test_a_box_that_states_no_cards_has_one(tmp_path):
    assert _profile(tmp_path, "").devices == 1


@pytest.mark.parametrize("bad", ["0", '"2"', "true", "-1"])
def test_a_card_count_that_is_not_a_positive_integer_is_refused(tmp_path, bad):
    with pytest.raises(BoxProfileError, match="devices must be a positive integer"):
        _profile(tmp_path, f"devices = {bad}\n")


# --- the oracles ------------------------------------------------------------

torch = pytest.importorskip("torch")


def _simulate(signature: str, data: list) -> "torch.Tensor":
    """Rank 0's result of the exchange, by the collective's definition over
    every rank's buffer — the ranks written out one by one, not the oracle's
    vectorised form."""
    world, n = len(data), data[0].numel()
    k = n // world
    flat = [d.reshape(-1).float() for d in data]
    if signature == "all_reduce":
        out = torch.zeros(n)
        for d in flat:
            out += d
        return out
    if signature == "all_gather":
        return torch.cat([flat[r][r * k:(r + 1) * k] for r in range(world)])
    if signature == "reduce_scatter":
        out = torch.zeros(k)
        for d in flat:
            out += d[:k]
        return out
    if signature == "all_to_all":
        return torch.cat([flat[r][:k] for r in range(world)])
    raise AssertionError(signature)


def _skipped(signature: str, mine: "torch.Tensor", world: int) -> list:
    """What rank 0 could hold if the launch never really exchanged: its own
    buffer, or its own shard copied into every slot."""
    flat = mine.reshape(-1).float()
    k = flat.numel() // world
    own = {"all_reduce": flat, "all_gather": flat,
           "reduce_scatter": flat[:k], "all_to_all": flat}[signature]
    return [own, flat[:k].repeat(world)] if signature == "all_gather" else [own]


@pytest.mark.parametrize("signature", ["all_reduce", "all_gather",
                                       "reduce_scatter", "all_to_all"])
@pytest.mark.parametrize("world", [2, 4])
def test_each_oracle_is_the_exchange_and_not_its_absence(signature, world):
    from kernel_bench.comm import rank_tensor
    from kernel_bench.refs import get_ref
    gen = torch.Generator().manual_seed(7)
    payload = torch.randn(4096, generator=gen)
    inputs = {"payload": payload, "n_comm_ranks": world, "n_comm_stride": 1}
    spec = get_ref(signature)
    assert spec.per_rank == ("payload",)
    expected = spec.reference(inputs)
    simulated = _simulate(signature, [rank_tensor(payload, r, world)
                                      for r in range(world)])
    torch.testing.assert_close(expected, simulated)
    for wrong in _skipped(signature, payload, world):
        assert not torch.allclose(expected, wrong, atol=1e-2), (
            "the oracle agrees with a launch that never exchanged anything")


def test_the_fused_oracle_is_the_norm_of_the_summed_activations():
    from kernel_bench.comm import rank_tensor
    from kernel_bench.refs import get_ref
    gen = torch.Generator().manual_seed(3)
    x, res = torch.randn(4, 256, generator=gen), torch.randn(4, 256, generator=gen)
    inputs = {"x": x, "residual": res, "w": torch.rand(256, generator=gen) + 0.5,
              "eps": 1e-6, "n_comm_ranks": 2}
    spec = get_ref("allreduce_fused_rmsnorm")
    assert spec.per_rank == ("x",)            # the residual is replicated
    expected = spec.reference(inputs)
    norm = get_ref("fused_add_rmsnorm").reference
    torch.testing.assert_close(expected, norm({**inputs, "x": x + rank_tensor(x, 1, 2)}))
    assert not torch.allclose(expected, norm(inputs), atol=1e-2)


@pytest.mark.parametrize("n", [8, 14, 4096, 151936])
def test_every_rank_holds_different_data_and_a_different_shard(n):
    from kernel_bench.comm import rank_tensor
    t = torch.arange(float(n))
    for world in (2, 4):
        assert rank_tensor(t, 0, world) is t
        k = n // world
        copies = [rank_tensor(t, r, world) for r in range(world)]
        assert all(torch.equal(c.sort().values, t) for c in copies)
        shards = {tuple(c[r * k:(r + 1) * k].tolist()) for r, c in enumerate(copies)}
        assert len(shards) == world, "two ranks would send the same shard"
        firsts = {tuple(c[:k].tolist()) for c in copies}
        assert len(firsts) == world, "two ranks would hand rank 0 the same slice"


# --- the clock's meeting points ---------------------------------------------

class _Recorder:
    """A group of one that answers like a slower peer would."""

    def __init__(self):
        self.agreed, self.aligned, self.samples = [], 0, None

    def agree(self, value):
        self.agreed.append(value)
        return value

    def align(self):
        self.aligned += 1

    def slowest(self, samples):
        self.samples = list(samples)
        return [1000.0] * len(samples)


def test_the_clock_meets_its_group_and_reports_the_slowest_rank():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernel_bench.timing import time_callable
    x = torch.randn(1024, device="cuda")
    group = _Recorder()
    stats = time_callable(lambda: x.mul_(1.0), iters=5, group=group)
    # The probe, the host time, and whether every rank holds its L2 sweep.
    assert len(group.agreed) == 3 and group.agreed[2] == 0.0
    assert group.aligned == 3 * 5                    # every window opens on it
    assert len(group.samples) == 5
    assert stats.p50_us == 1000.0                    # the slowest rank's samples


# --- one real run -----------------------------------------------------------

@pytest.mark.skipif(os.environ.get("KERNEL_BENCH_MULTI_GPU") != "1",
                    reason="two idle cards: KERNEL_BENCH_MULTI_GPU=1")
def test_a_pair_measures_every_collective_and_checks_it():
    import dataclasses

    from kernel_bench.comm import Group
    from kernel_bench.driver import run_jobs
    from kernel_bench.env import detect_env
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible cards")
    env = dataclasses.replace(detect_env(), library="torch")
    flat = {"n_comm_ranks": 2, "n_comm_stride": 1}
    jobs = [_job("all_reduce", {"n_comm_elems": 4096, **flat}),
            _job("all_gather", {"n_comm_elems": 1 << 20, **flat}, "prefill"),
            _job("reduce_scatter", {"n_comm_elems": 1 << 20, **flat}, "prefill"),
            _job("all_to_all", {"n_comm_elems": 1 << 20, **flat}, "prefill"),
            _job("allreduce_fused_rmsnorm",
                 {"B": 4, "s_query": 1, "d_model": 4096, "n_comm_ranks": 2})]
    jobs[-1]["dtype_env"] = {"activation": "bf16", "output": "bf16", "weight": "bf16"}
    try:
        res = run_jobs(jobs, env=env, date="2026-11-27", contributor="test")
    finally:
        Group.release()
    rows = {r["kernel_impl"]: r for r in res.rows}
    for impl in ("torch_nccl_all_reduce", "torch_nccl_all_gather",
                 "torch_nccl_reduce_scatter", "torch_nccl_all_to_all",
                 "torch_nccl_allreduce_rmsnorm"):
        assert impl in rows, (impl, res.failures, res.skipped)
        assert rows[impl]["outcome"] in ("", "measured"), rows[impl]
        assert rows[impl]["cross_check"] == "passed", rows[impl]
        assert float(rows[impl]["p50_us"]) > 0
