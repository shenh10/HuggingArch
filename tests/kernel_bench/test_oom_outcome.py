"""An OOM is a recorded outcome, not a failure to retry.

A cell that does not fit on a card does not fit on the next pass either, so
re-running rediscovers it: the sweep pays for the allocation, the allocator
fragments, and nothing in the corpus can tell "this does not fit" from
"nobody has measured this yet" — which is the distinction the coverage audit
needs most.

So the row is written with `outcome="oom"` and no measurement at all. These
assert the three properties that make that safe: the schema allows the
numbers empty ONLY there, the estimate never indexes such a row (it would
read an empty latency), and the harness's coverage DOES count it (there is
nothing to learn by running it again).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.inference import benchmarks as bm


def _row(**over) -> dict:
    row = {c: "" for c in bm.ALL_COLUMNS}
    row.update(
        model="m/x", spec_hash="h", phase="decode", parallel="tp=1",
        framework="torch", framework_version="2.8", gpu="H100-SXM",
        op="linear", node_spec_id="n",
        shape_json=json.dumps({"B": 1, "s_query": 1, "d_reduce": 8, "d_out": 8}),
        dtype_env_json=json.dumps({"activation": "bf16", "output": "bf16"}),
        kernel_impl="k", schema_version=str(bm.KNOWN_SCHEMA_VERSION),
        contributor="@t", date="2026-10-21")
    row.update(over)
    return row


class TestTheSchemaAllowsNoNumbersOnlyForAnOutcome(unittest.TestCase):
    def test_an_oom_row_needs_no_measurement(self):
        """No numbers — and still a contributor, a date and a `cross_check`
        from the vocabulary: waiving the measurement does not waive the
        provenance, which an earlier draft of this did by returning early."""
        bm.validate_row(_row(outcome="oom", cross_check="unknown"))

    def test_an_oom_row_may_not_carry_one(self):
        """A latency beside `outcome="oom"` is two claims at once, and the
        second one would be read."""
        with self.assertRaises(bm.BenchmarkValidationError) as caught:
            bm.validate_row(_row(outcome="oom", cross_check="unknown",
                                 latency_us="5.0"))
        self.assertIn("carries no measurement", str(caught.exception))

    def test_a_measured_row_still_needs_its_numbers(self):
        with self.assertRaises(bm.BenchmarkValidationError):
            bm.validate_row(_row(outcome="measured"))

    def test_an_empty_outcome_reads_as_measured(self):
        """Every row written before the column has numbers in it — the
        validator required them — so the default is read, not guessed."""
        with self.assertRaises(bm.BenchmarkValidationError):
            bm.validate_row(_row(outcome=""))        # no numbers → refused

    def test_a_refusal_still_needs_its_provenance(self):
        """The early return waived these too, so a third party could upload
        a refusal with no contributor and `date="yesterday"`."""
        with self.assertRaises(bm.BenchmarkValidationError):
            bm.validate_row(_row(outcome="oom"))          # no cross_check

    def test_the_vocabulary_is_closed(self):
        with self.assertRaises(bm.BenchmarkValidationError) as caught:
            bm.validate_row(_row(outcome="oomm", cross_check="unknown"))
        self.assertIn("outcome", str(caught.exception))


def _oom_row() -> dict:
    row = _row(outcome="oom", cross_check="unknown")
    row["origin"] = "shared"
    return row


class TestTheEstimateDoesNotIndexARefusal(unittest.TestCase):
    def test_a_lookup_indexes_no_candidate_for_it(self):
        """Not merely "does not crash": the key must come out with NO
        candidate, because a cell whose only row is a refusal has no
        measured latency to price it with."""
        from backend.inference.op_calibration import (PhysicalKey,
                                                      registry_from_lookup)
        shape = json.dumps({"B": 1, "s_query": 1, "d_reduce": 8, "d_out": 8})
        dtypes = json.dumps({"activation": "bf16", "output": "bf16"})
        key = PhysicalKey(signature="gemm", shape_json=shape,
                          dtype_env_json=dtypes, gpu="H100-SXM", phase="decode")
        registry = registry_from_lookup([key], [[_oom_row()]])
        self.assertEqual(dict(registry.rows_by_key), {})

    def test_the_other_indexing_path_skips_it_too(self):
        """`merge_validated_rows` is the second caller — a request's own
        calibration rows — and it read `float("")` until the skip moved
        into `_index_row` itself."""
        from backend.inference.op_calibration import (CalibrationRegistry,
                                                      merge_validated_rows)
        registry = merge_validated_rows(CalibrationRegistry(), [_oom_row()],
                                        source="test", origin="mine")
        self.assertEqual(dict(registry.rows_by_key), {})


class TestTheRowIsNotCountedAsAMeasurement(unittest.TestCase):
    def test_is_measurement_reads_the_column(self):
        self.assertFalse(bm.is_measurement(_row(outcome="oom")))
        self.assertTrue(bm.is_measurement(_row(outcome="measured")))
        self.assertTrue(bm.is_measurement(_row(outcome="")))

    def test_a_measurement_and_a_refusal_of_one_cell_coexist(self):
        """They share all 12 identity columns BY DESIGN. Keyed on identity
        alone, the upload dropped whichever came second — silently — and the
        harness's own append-only results file stopped parsing."""
        from backend.inference.benchmarks import RowCheck
        check = RowCheck()
        measured = _row(outcome="measured", latency_us="5.0", p50_us="5.0",
                        p95_us="6.0", samples="60", count_per_step="1",
                        achieved_tflops="1.0", achieved_bw_gbps="0.0",
                        peak_tflops_at_test="900.0", peak_bw_at_test="3350.0",
                        cross_check="passed")
        self.assertIsNone(check.reason(measured, 1))
        self.assertIsNone(check.reason(_row(outcome="oom",
                                            cross_check="unknown"), 2))
        self.assertIsNotNone(check.reason(dict(measured), 3))


class TestWhatCountsAsTheCardRefusing(unittest.TestCase):
    def test_a_host_oom_on_a_cpu_box_is_not_a_card_refusing(self):
        from kernel_bench.driver import _is_out_of_memory
        from kernel_bench.env import Env
        cpu = Env(torch_version="2.8.0", device=None,
                  device_name="cpu", gpu_sku="", peak_bw_gbps=0.0)
        self.assertFalse(_is_out_of_memory(RuntimeError("out of memory"), cpu))

    def test_a_wrapped_allocation_failure_still_counts(self):
        from kernel_bench.driver import _is_out_of_memory
        inner = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
        try:
            try:
                raise inner
            except RuntimeError as e:
                raise ValueError("provider wrapped it") from e
        except ValueError as wrapped:
            self.assertTrue(_is_out_of_memory(wrapped))

    def test_an_unrelated_error_is_not_a_refusal(self):
        from kernel_bench.driver import _is_out_of_memory
        self.assertFalse(_is_out_of_memory(RuntimeError("shape mismatch")))


class TestTheReleaseIsJudgedAgainstWhatWasAsked(unittest.TestCase):
    """After an OOM the allocator is emptied and the free memory re-read;
    the card was only crowded if it now has room for the request. The
    cell's own inputs survive the release, so free memory alone said
    "crowded" whenever they were large."""

    _H100 = ("CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a "
             "total capacity of 79.18 GiB of which 38.35 GiB is free.")

    def _fits(self, error, free_gib, total_gib=79.18):
        from unittest import mock
        import torch
        from kernel_bench.driver import _fits_after_release
        from kernel_bench.env import Env
        gpu = Env(torch_version="2.8.0", device="cuda",
                  device_name="H100", gpu_sku="H100-SXM", peak_bw_gbps=0.0)
        info = (int(free_gib * (1 << 30)), int(total_gib * (1 << 30)))
        with mock.patch.object(torch.cuda, "synchronize"), \
                mock.patch.object(torch.cuda, "empty_cache"), \
                mock.patch.object(torch.cuda, "mem_get_info",
                                  return_value=info):
            return _fits_after_release(gpu, error)

    def test_room_short_of_the_request_is_the_cell_not_fitting(self):
        # The H100's 2,621,440-row grouped GEMM: the build asked for a
        # second 40 GiB beside the inputs' 40 GiB, and 38 GiB came back.
        self.assertFalse(self._fits(RuntimeError(self._H100), 38.35))

    def test_room_for_the_request_is_a_crowded_card(self):
        self.assertTrue(self._fits(RuntimeError(self._H100), 41.0))

    def test_the_request_is_read_through_a_wrapper(self):
        from kernel_bench.driver import _requested_bytes
        try:
            try:
                raise RuntimeError(self._H100)
            except RuntimeError as e:
                raise ValueError("provider wrapped it") from e
        except ValueError as wrapped:
            self.assertEqual(_requested_bytes(wrapped), 40 << 30)

    def test_a_failure_that_does_not_say_its_size_keeps_the_card_line(self):
        error = RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
        self.assertTrue(self._fits(error, 38.35))
        self.assertFalse(self._fits(error, 4.0))


def _crowded_run(monkeypatch, fails):
    """One gemm cell through `run_jobs` with a provider whose build runs out
    of memory `fails` times, on a card that has room once released."""
    import kernel_bench.driver as driver
    from kernel_bench.env import Env
    from kernel_bench.providers import BenchCase, KernelSignature, Provider
    from tests.kernel_bench.test_provider_canary import _job
    built = []

    def build(inv, env, inputs):
        built.append(1)
        if len(built) <= fails:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        x, w = inputs["x"], inputs["w"]
        return BenchCase(run=lambda: x @ w.T, to_canonical=lambda o: o)

    provider = Provider(library="torch", signature="gemm",
                        kernel_impl="t_crowded", import_name="torch",
                        build=build, serves=KernelSignature(signature="gemm"))
    monkeypatch.setattr(driver, "applicable", lambda inv, env: [provider])
    monkeypatch.setattr(driver, "_is_out_of_memory", lambda e, env=None: True)
    monkeypatch.setattr(driver, "_fits_after_release", lambda env, e: True)
    rows = []
    result = driver.run_jobs(
        [_job(64)], env=Env("2.11", None, None, None, None,
                            libs={"torch": "2.11"}),
        date="2026-10-26", contributor="@t", on_row=rows.append)
    return built, [bm.outcome_of(r) for r in rows], result


def test_a_crowded_out_of_memory_is_tried_once_more(monkeypatch):
    # The card had room once released: crowded, not too small. The attempt
    # is made again in-process instead of writing an `error` that keeps the
    # cell tried for good (and stops its path when it is a canary).
    built, outcomes, result = _crowded_run(monkeypatch, fails=1)
    assert len(built) == 2
    assert outcomes == ["measured"]
    assert result.disabled == {}


def test_a_second_out_of_memory_is_the_cells_oom(monkeypatch):
    built, outcomes, result = _crowded_run(monkeypatch, fails=5)
    assert len(built) == 2
    assert outcomes == ["oom"]
    assert result.disabled == {}


def _inputs_run(monkeypatch, fails):
    """One gemm cell through `run_jobs` whose INPUTS run out of memory
    `fails` times, each request alone fitting the released card — the
    H100's `mul` at 4M tokens × 5120: a 40 GiB operand beside another
    40 GiB one, on a 79 GiB card."""
    import kernel_bench.driver as driver
    from kernel_bench import refs
    from kernel_bench.env import Env
    from kernel_bench.providers import BenchCase, KernelSignature, Provider
    from tests.kernel_bench.test_provider_canary import _job
    real, made, built = refs.get_ref, [], []

    class Spec:
        def __init__(self, spec):
            self._spec = spec

        def __getattr__(self, name):
            return getattr(self._spec, name)

        def make_inputs(self, inv, device, generator):
            made.append(1)
            if len(made) <= fails:
                raise RuntimeError(
                    "CUDA out of memory. Tried to allocate 40.00 GiB")
            return self._spec.make_inputs(inv, device, generator)

    def build(inv, env, inputs):
        built.append(1)
        x, w = inputs["x"], inputs["w"]
        return BenchCase(run=lambda: x @ w.T, to_canonical=lambda o: o)

    provider = Provider(library="torch", signature="gemm",
                        kernel_impl="t_inputs", import_name="torch",
                        build=build, serves=KernelSignature(signature="gemm"))
    monkeypatch.setattr(refs, "get_ref", lambda sig: Spec(real(sig)))
    monkeypatch.setattr(driver, "applicable", lambda inv, env: [provider])
    monkeypatch.setattr(driver, "_is_out_of_memory", lambda e, env=None: True)
    monkeypatch.setattr(driver, "_fits_after_release", lambda env, e: True)
    rows = []
    result = driver.run_jobs(
        [_job(64)], env=Env("2.11", None, None, None, None,
                            libs={"torch": "2.11"}),
        date="2026-10-29", contributor="@t", on_row=rows.append)
    return made, [bm.outcome_of(r) for r in rows], result


def test_inputs_larger_than_the_card_are_the_cells_oom(monkeypatch):
    # Every allocation alone fits the released card, the cell's operands
    # together do not: built once more, and the second out-of-memory is the
    # cell's `oom`. Deferred with no row, the same cells failed every pass
    # (kda / mul / add / fused_engram_gate / attn_res_combine on the H100).
    made, outcomes, result = _inputs_run(monkeypatch, fails=5)
    assert len(made) == 2
    assert outcomes == ["oom"]
    assert result.failures == []


def test_inputs_on_a_crowded_card_are_built_once_more(monkeypatch):
    made, outcomes, _ = _inputs_run(monkeypatch, fails=1)
    assert len(made) == 2
    assert outcomes == ["measured"]


if __name__ == "__main__":
    unittest.main()
