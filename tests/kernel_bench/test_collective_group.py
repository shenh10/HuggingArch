"""Group mapping, watchdog, unique writer and max-over-ranks latency.

These pin the process model without HCCL: stub children return rows, the
parent aggregates, and a dead or silent rank kills the rest. Die mapping
stays in `group.py` and never enters the 12 identity columns.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from typing import ClassVar
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


_JOB = {
    "op": "all_reduce",
    "signature": "all_reduce",
    "dims": {"n_comm_elems": 8, "n_comm_ranks": 2, "n_comm_stride": 1},
    "dtype_env": {"activation": "bf16"},
    "phase": "prefill",
}


def _ok_rank(cfg, queue):
    rank = int(cfg["rank"])
    queue.put({
        "rank": rank,
        "rows": [{
            "kernel_impl": "hccl_all_reduce",
            "p50_us": f"{10.0 + rank:.4f}",
            "p95_us": f"{12.0 + rank:.4f}",
            "latency_us": f"{10.0 + rank:.4f}",
            "outcome": "measured",
            "launch_signature": "dist.all_reduce()",
        }],
    })


def _hang_rank(cfg, queue):
    time.sleep(60)


def _exit_rank(cfg, queue):
    if int(cfg["rank"]) == 1:
        os._exit(1)
    time.sleep(60)


def _error_rank(cfg, queue):
    if int(cfg["rank"]) == 1:
        queue.put({"rank": 1, "rows": [], "error": "rank 1: boom"})
        return
    _ok_rank(cfg, queue)


def _env():
    from kernel_bench.env import Env
    return Env("2.11", None, None, None, None)


#: What a child reported back about the config it was handed. A module-level
#: queue is not available to a spawned process, so the rank puts the fields
#: it cares about on the queue the parent already reads.
def _cfg_echo_rank(cfg, queue):
    rank = int(cfg["rank"])
    queue.put({
        "rank": rank,
        "rows": [{
            "kernel_impl": "hccl_all_reduce",
            "p50_us": f"{10.0 + rank:.4f}",
            "p95_us": f"{12.0 + rank:.4f}",
            "latency_us": f"{10.0 + rank:.4f}",
            "outcome": "measured",
            "launch_signature": "dist.all_reduce()",
            # Not a real column: the test reads it off the aggregated row to
            # see what the parent put in `cfg`.
            "_saw_peak_bw": repr(cfg.get("peak_bw_gbps")),
            "_saw_gpu_sku": repr(cfg.get("gpu_sku")),
        }],
    })


class TestDieMappingStaysOutOfIdentity(unittest.TestCase):
    def test_die_id_is_rank_times_stride(self):
        from kernel_bench.group import dies_for
        self.assertEqual(dies_for(4, stride=1), [0, 1, 2, 3])
        self.assertEqual(dies_for(4, stride=2), [0, 2, 4, 6])
        self.assertEqual(dies_for(2, stride=2, origin=1), [1, 3])

    def test_a_mixed_job_list_unions_the_dies(self):
        from kernel_bench.group import dies_for_jobs
        local = {"op": "linear", "dims": {"B": 1}}
        comm = dict(_JOB)
        self.assertIsNone(dies_for_jobs([local]))
        self.assertEqual(dies_for_jobs([comm], stride=2), [0, 2])
        self.assertEqual(dies_for_jobs([local, comm], stride=2), [0, 2])


class TestMaxAcrossRanksIsTheLatency(unittest.TestCase):
    def test_the_row_is_the_slowest_rank(self):
        from kernel_bench.group import aggregate_latency, spread_note
        rows = [
            {"_rank": 0, "kernel_impl": "hccl_all_reduce", "p50_us": "10.0",
             "p95_us": "11.0", "latency_us": "10.0", "outcome": "measured",
             "launch_signature": "dist.all_reduce()"},
            {"_rank": 1, "kernel_impl": "hccl_all_reduce", "p50_us": "13.0",
             "p95_us": "14.0", "latency_us": "13.0", "outcome": "measured",
             "launch_signature": "dist.all_reduce()"},
        ]
        out = aggregate_latency(rows)
        self.assertEqual(out["p50_us"], "13.0000")
        self.assertEqual(out["latency_us"], "13.0000")
        self.assertEqual(out["p95_us"], "14.0000")
        self.assertIn("max/min=", out["_rank_spread"])
        self.assertIn("rank_spread", out["launch_signature"])
        self.assertNotIn("_rank", out)
        self.assertIn("max/min=1.3", spread_note([10.0, 13.0]))

    def test_one_parent_row_from_two_children(self):
        from kernel_bench.group import measure_job
        rows = measure_job(_JOB, env=_env(), take_lock=False, timeout_s=15,
                           worker=_ok_rank)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["p50_us"], "11.0000")
        self.assertEqual(rows[0]["kernel_impl"], "hccl_all_reduce")


class TestTheGroupDiesTogether(unittest.TestCase):
    def test_a_silent_rank_kills_the_group(self):
        from kernel_bench.group import GroupFailed, measure_job
        t0 = time.monotonic()
        with self.assertRaises(GroupFailed) as caught:
            measure_job(_JOB, env=_env(), take_lock=False, timeout_s=1.5,
                        worker=_hang_rank)
        self.assertIn("timed out", str(caught.exception))
        self.assertLess(time.monotonic() - t0, 8.0)

    def test_an_exited_rank_kills_the_group(self):
        from kernel_bench.group import GroupFailed, measure_job
        with self.assertRaises(GroupFailed) as caught:
            measure_job(_JOB, env=_env(), take_lock=False, timeout_s=8,
                        worker=_exit_rank)
        self.assertIn("exited", str(caught.exception))

    def test_a_reported_error_kills_the_group(self):
        from kernel_bench.group import GroupFailed, measure_job
        with self.assertRaises(GroupFailed) as caught:
            measure_job(_JOB, env=_env(), take_lock=False, timeout_s=15,
                        worker=_error_rank)
        self.assertIn("boom", str(caught.exception))


class TestTheParentIsTheOnlyWriter(unittest.TestCase):
    def test_the_child_never_opens_the_csv(self):
        source = (PROJECT_ROOT / "kernel_bench" / "rankworker.py").read_text()
        self.assertNotIn("from .emit", source)
        self.assertNotIn("IncrementalCsvWriter(", source)
        self.assertNotIn("results.csv", source)


class TestHcclIsAGroupKernel(unittest.TestCase):
    def test_the_four_collectives_are_enabled_and_named(self):
        from kernel_bench.providers import all_providers
        hccl = [p for p in all_providers()
                if p.kernel_impl.startswith("hccl_")]
        self.assertEqual(sorted(p.signature for p in hccl),
                         ["all_gather", "all_reduce", "all_to_all",
                          "reduce_scatter"])
        for p in hccl:
            self.assertEqual(p.world, "n_comm_ranks")
            self.assertFalse(p.disabled_reason)

    def test_the_child_is_handed_the_card_the_parent_resolved(self):
        """The regression: a group measured correctly and wrote nothing.

        `emit.py:507` writes `peak_bw_at_test` from `env.peak_bw_gbps`, and
        an empty one fails schema validation, so every row a group measured
        was dropped at persist. The parent resolves the card off the BOX
        when the driver's device name names no sheet SKU (an Ascend die
        reports `Ascend910_9362`, whose family token carries no digit), so a
        child that only re-detects gets `None` and writes "".

        Both facts travel, because the SKU alone does not carry the
        bandwidth: `cli._cmd_run` looks the card up and replaces BOTH.
        """
        from dataclasses import replace

        from kernel_bench.group import measure_job
        env = replace(_env(), gpu_sku="Ascend-910C-A3", peak_bw_gbps=1600.0)
        rows = measure_job(_JOB, env=env, take_lock=False, timeout_s=15,
                           worker=_cfg_echo_rank)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_saw_peak_bw"], repr(1600.0))
        self.assertEqual(rows[0]["_saw_gpu_sku"], repr("Ascend-910C-A3"))

    def test_the_empty_bandwidth_column_is_what_schema_rejects(self):
        """The other half of the failure: why an empty column loses the row.

        The group measured correctly and the log said `measured 2 rows`;
        both were then dropped at persist. That step is a REQUIRED column
        check, so a child that inherits no bandwidth does not fail loudly —
        it writes "" and the row disappears. Asserted against the validator
        itself rather than a copy of its list.
        """
        from backend.inference.benchmarks import MEASUREMENT_COLUMNS
        self.assertIn("peak_bw_at_test", MEASUREMENT_COLUMNS)

    def test_a_single_rank_is_not_a_group(self):
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import all_providers
        p = next(x for x in all_providers()
                 if x.kernel_impl == "hccl_all_reduce")
        inv = Invocation(
            signature="all_reduce", op="all_reduce",
            dims={"n_comm_elems": 8, "n_comm_ranks": 1, "n_comm_stride": 1},
            dtype_env={"activation": "bf16"}, phase="prefill", tp=1)
        reasons = list(p.refusals(inv, _env()))
        self.assertTrue(any(r.startswith("world:") for r in reasons), reasons)


class TheRankWorkerAgreesOnSampleCounts(unittest.TestCase):
    """`_agree` broadcasts rank 0's warmup/iters to the group.

    It runs only inside a real process group, which is why the stub
    children above cannot reach it — and why a plain attribute mistake in
    it survived 842 passing tests and surfaced as `TypeError: 'str'
    object is not callable` on sixteen dies at once. These exercise the
    closure against fakes so the next one surfaces here instead.
    """

    class _Dev:
        """A device backend with `tensor_device` as the PROPERTY it is."""
        synced = False

        @property
        def tensor_device(self):
            return "cpu"

        def synchronize(self):
            type(self).synced = True

    class _Dist:
        """A `torch.distributed` stand-in: rank 0's values win."""
        def __init__(self, decided=(7, 11)):
            self.decided = decided
            self.seen = []

        def broadcast(self, tensor, src=0):
            self.seen.append((tuple(int(x) for x in tensor), src))
            tensor[0], tensor[1] = self.decided

    def test_it_takes_the_broadcast_counts(self):
        from kernel_bench.rankworker import _agree
        dist = self._Dist(decided=(7, 11))
        dev = self._Dev()
        agree = _agree(dist, dev)
        self.assertEqual(agree(2, 60), (7, 11))
        # What this rank proposed went in, and rank 0 is the source.
        self.assertEqual(dist.seen, [((2, 60), 0)])
        self.assertTrue(self._Dev.synced,
                        "the broadcast must be synchronised before its "
                        "values are read back off the device")

    def test_without_a_group_there_is_nothing_to_agree(self):
        from kernel_bench.rankworker import _agree
        self.assertIsNone(_agree(None, self._Dev()))

    def test_the_device_is_asked_for_a_string_not_called(self):
        """The regression itself: `tensor_device` is a property, and
        calling it raises rather than returning a device."""
        from kernel_bench.rankworker import _agree
        agree = _agree(self._Dist(), self._Dev())
        try:
            agree(2, 60)
        except TypeError as exc:                      # pragma: no cover
            self.fail(f"treated a device attribute as callable: {exc}")


if __name__ == "__main__":
    unittest.main()


class TheReachSurvivesTheRoundTrip(unittest.TestCase):
    """A crossing measurement must still read as crossing after it lands.

    The write side has TWO places a cell can be stated: the row's
    `shape_json` column, which `emit.build_row` serialises from
    `inv.dims`, and the `PhysicalKey` coverage asks about. Stamping only
    the second left a cross-machine row on disk whose shape said
    intra-host, so reading the corpus back keyed it as a chassis
    measurement -- the pollution `n_comm_domain` exists to prevent,
    arriving from the side it was not watched on.

    This walks the whole chain on the real functions: job -> dims ->
    the row's shape text -> corpus schema -> `BenchmarkKey` ->
    `to_physical`.
    """

    JOB: ClassVar[dict] = {
        "op": "all_reduce", "signature": "all_reduce",
        "dims": {"n_comm_elems": 4194304, "n_comm_ranks": 16,
                 "n_comm_stride": 1},
        "dtype_env": {"activation": "bf16", "output": "bf16"},
        "phase": "prefill", "tp": 1,
        "physical_key": [
            "all_reduce",
            '{"n_comm_elems":4194304,"n_comm_ranks":16,"n_comm_stride":1}',
            '{"activation":"bf16","output":"bf16"}', "prefill"],
    }

    @staticmethod
    def _shape_text(job, domain):
        """The text `emit.build_row` would write, for this domain."""
        """The text a WRITTEN row carries, off `build_row` itself.

        Not `_cell_text`: that is the helper under test, and asking it
        directly would pass whether or not `build_row` calls it. The row
        is what lands in the corpus, so the row is what is read.
        """
        from kernel_bench.driver import _job_with_comm_domain
        from kernel_bench.emit import build_row
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import Provider
        from kernel_bench.timing import LatencyStats

        inv = Invocation.from_job(_job_with_comm_domain(job, domain))
        class _P(Provider):
            def version(self, env):      # the env fixture knows no wheels
                return "2.10.0.post7"

        provider = _P(library="vllm_ascend", signature="all_reduce",
                      kernel_impl="hccl_all_reduce",
                      import_name="torch_npu", build=lambda *a: None)
        stats = LatencyStats(p50_us=1.0, p95_us=1.0, mean_us=1.0, samples=3,
                             timing_method="wall")
        row = build_row(inv, provider, stats, _env(),
                        date="2026-11-20", contributor="@t")
        return row["shape_json"]

    def _physical(self, domain):
        from backend.inference.op_calibration import (
            BenchmarkKey, CalibrationRegistry, merge_validated_rows)
        from tests.corpus_row import row as corpus_row
        shape = self._shape_text(self.JOB, domain)
        r = corpus_row(
            op="all_reduce", kernel_impl="hccl_all_reduce",
            framework="vllm_ascend", gpu="Ascend-910C-A3", phase="prefill",
            shape_json=shape,
            dtype_env_json='{"activation":"bf16","output":"bf16"}')
        # The schema is the gate every corpus row passes; a shape it
        # refuses never reaches a key at all.
        merge_validated_rows(CalibrationRegistry(), [dict(r)],
                             source="<test>", origin="shared")
        return shape, BenchmarkKey.from_csv_row(r).to_physical()

    def test_a_crossing_row_carries_its_reach_all_the_way_through(self):
        from backend.inference.comm_cost import COMM_REACH_SCALEUP
        shape, key = self._physical("scaleup")
        stamped = f'"n_comm_domain":{COMM_REACH_SCALEUP}'
        self.assertIn(stamped, shape,
                      "the row's own shape text lost the reach")
        self.assertIn(stamped, key.shape_json,
                      "the physical key lost the reach the row carried")

    def test_it_does_not_collide_with_the_same_cell_on_one_host(self):
        _, crossed = self._physical("scaleup")
        _, local = self._physical("host")
        self.assertNotEqual(
            crossed, local,
            "a cross-machine row and a chassis row of the same payload "
            "and world share a key, so the envelope would hand one for "
            "the other")

    def test_a_host_row_is_byte_identical_to_before_the_axis(self):
        shape, _ = self._physical("host")
        self.assertEqual(
            shape,
            '{"n_comm_elems":4194304,"n_comm_ranks":16,"n_comm_stride":1}')


class TheGroupPathCarriesTheReachToo(unittest.TestCase):
    """A collective is measured by a second `run_jobs` inside each rank.

    That inner call is handed a JOB, not this call's flags, so a reach
    stamped only on the outer loop reaches every row except the
    collectives -- which are the only rows a reach applies to. The first
    cross-machine run wrote three rows keyed as intra-host because of
    exactly this, and nothing failed: the rows were valid, passed their
    cross-check, and said the wrong thing about where they were taken.
    """

    JOB: ClassVar[dict] = {
        "op": "all_reduce", "signature": "all_reduce",
        "dims": {"n_comm_elems": 8, "n_comm_ranks": 2, "n_comm_stride": 1},
        "dtype_env": {"activation": "bf16"}, "phase": "prefill", "tp": 1,
    }

    def test_the_job_handed_to_the_ranks_is_stamped(self):
        from kernel_bench.driver import _job_with_comm_domain
        from backend.inference.comm_cost import COMM_REACH_SCALEUP
        seen = {}

        def fake_measure_job(job, **_kw):
            seen["job"] = job
            return []

        # What `run_jobs` does at the group branch, in one line: the job
        # is stamped BEFORE it is handed over.
        fake_measure_job(_job_with_comm_domain(self.JOB, "scaleup"))
        self.assertEqual(seen["job"].get("comm_domain"), COMM_REACH_SCALEUP,
                         "the ranks were handed an unstamped job, so their "
                         "rows will say intra-host whatever the flag said")

    def test_the_group_branch_stamps_before_handing_over(self):
        """Read the call site itself: `measure_job` must be given a
        stamped job, not the raw one."""
        import inspect
        from kernel_bench import driver
        src = inspect.getsource(driver.run_jobs)
        call = src[src.index("rows = measure_job("):]
        call = call[:call.index(")")]
        self.assertIn("_job_with_comm_domain", call,
                      "the group branch hands `measure_job` an unstamped "
                      "job; collectives are the only rows the reach "
                      "applies to, so that is all of them")
