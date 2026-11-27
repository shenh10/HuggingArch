"""Which driver ran a row, read from the machine and stamped on the row.

A box can measure one env through NVIDIA's forward-compatibility user-mode
driver and another through its own — `kernel_bench/envs/boxes/
h100-sxm-cu128.toml` does exactly that, for three envs and deliberately not
the rest — and before schema 14 nothing on a row said which. These cover the
two reads (`/proc/driver/nvidia/version` for the box's kernel-mode driver,
`/proc/self/maps` for the user-mode driver this process actually loaded) and
the stamping that puts both on the row.

The reads are tested against TEXT rather than the running machine, because
what has to be right is the parsing of forms this box does not have: the
packaged driver's `575.57.08` beside the open module's `595.84`, and a
`libcuda` that came out of a compat directory.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.inference import benchmarks as bm
from kernel_bench.env import (
    Env,
    kernel_mode_driver,
    loaded_user_mode_driver,
)

# The real first line of `/proc/driver/nvidia/version` in both forms seen on
# the boxes this repo measures on. Note the open-module line carries `x86_64`
# BEFORE the version, which is the trap: a pattern of "digits, optionally
# dotted" finds `86` there.
_OPEN_MODULE = (
    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.84  "
    "Release Build  (dvs-builder@U22-I3-AM25-26-2)  Wed Jun 10 21:06:37 UTC 2026\n"
    "GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)\n"
)
_PACKAGED = (
    "NVRM version: NVIDIA UNIX x86_64 Kernel Module  575.57.08  "
    "Fri May 30 02:02:25 UTC 2025\n"
    "GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)\n"
)


class TestKernelModeDriver(unittest.TestCase):
    """The BOX's driver. Same for every env on the machine, and nothing on
    `LD_LIBRARY_PATH` can move it — which is what makes it the baseline the
    user-mode driver is compared against."""

    def test_open_kernel_module_form(self):
        """Excludes a pattern that accepts undotted digits.

        `x86_64` sits ahead of the version on this line, so a `\\d+(\\.\\d+)*`
        search returns `86` — a plausible-looking answer that is not a driver
        and that would read as a substitution against any real user-mode
        driver. Requiring at least one dot is what makes this line parse.
        """
        self.assertEqual(kernel_mode_driver(_OPEN_MODULE), "595.84")

    def test_packaged_module_form(self):
        """The three-part form, which is what the H100 box reports and what
        the substitution case is compared against."""
        self.assertEqual(kernel_mode_driver(_PACKAGED), "575.57.08")

    def test_no_nvrm_line_is_unknown(self):
        """Excludes "read the first line whatever it says": a box with no
        NVIDIA kernel module must answer unknown, not the GCC version."""
        self.assertIsNone(kernel_mode_driver(
            "GCC version:  gcc version 11.4.0\n"))
        self.assertIsNone(kernel_mode_driver(""))

    def test_an_nvrm_line_with_no_version_is_unknown(self):
        self.assertIsNone(kernel_mode_driver("NVRM version: unavailable\n"))

    def test_what_it_returns_passes_the_corpus_validator(self):
        """The parse and the column's accepted shape are one decision. If
        they drift, the harness measures rows the validator then rejects."""
        for text in (_OPEN_MODULE, _PACKAGED):
            row = {"kmd_version": kernel_mode_driver(text), "umd_version": ""}
            bm._check_driver_version(row["kmd_version"], column="kmd_version")


class TestLoadedUserModeDriver(unittest.TestCase):
    """The PROCESS's driver: whichever `libcuda.so.1` the loader resolved.
    Read from the mapping, so an `LD_LIBRARY_PATH` that did not take effect
    reports the system driver — correctly."""

    def test_the_system_driver(self):
        maps = (
            "7f0a00000000-7f0a00001000 r--p 00000000 fd:01 1 "
            "/usr/lib/x86_64-linux-gnu/libcuda.so.595.84\n"
        )
        self.assertEqual(loaded_user_mode_driver(maps), "595.84")

    def test_a_compat_driver_out_of_its_own_directory(self):
        """THE case these columns exist for: the loader resolved
        `libcuda.so.1` to the compat package's newer driver, so the version
        the process ran under is not the box's."""
        maps = (
            "7f0a00000000-7f0a00001000 r--p 00000000 fd:01 1 "
            "/scratch/kb-work/cuda-compat-13.0/compat/libcuda.so.580.95.05\n"
        )
        self.assertEqual(loaded_user_mode_driver(maps), "580.95.05")

    def test_the_cuda_RUNTIME_is_not_the_driver(self):
        """Excludes `base.startswith("libcuda")`, which matches
        `libcudart.so.13.0` and would return `13.0` — a CUDA RUNTIME version
        in a driver column.

        That is the single most damaging wrong answer available here: `13.0`
        beside a real `575.57.08` reads as a driver substitution on a row
        that had none, and every consumer would believe it. The runtime is
        mapped first on purpose, so a lenient prefix test takes it.
        """
        maps = (
            "7f0a00000000-7f0a00001000 r--p 00000000 fd:01 1 "
            "/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/"
            "libcudart.so.13.0\n"
            "7f0a00002000-7f0a00003000 r--p 00000000 fd:01 2 "
            "/usr/lib/x86_64-linux-gnu/libcuda.so.595.84\n"
        )
        self.assertEqual(loaded_user_mode_driver(maps), "595.84")

    def test_an_unversioned_libcuda_is_unknown(self):
        """A `libcuda.so.1` that is itself the real file names no version.
        Unknown, and NOT `"1"` — which is what a pattern allowing undotted
        digits would return, and which would compare unequal to every real
        kernel-mode driver and so flag a substitution on every row.
        """
        maps = (
            "7f0a00000000-7f0a00001000 r--p 00000000 fd:01 1 "
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1\n"
        )
        self.assertIsNone(loaded_user_mode_driver(maps))

    def test_no_libcuda_mapped_is_unknown(self):
        """Excludes any implementation that invents a driver for an env with
        no CUDA — the honest answer is unknown, and the pair then reads as
        unknown rather than as a native configuration."""
        maps = (
            "55a000000000-55a000001000 r-xp 00000000 fd:01 9 /usr/bin/python3.12\n"
            "7f0a00000000-7f0a00021000 rw-p 00000000 00:00 0 \n"
            "7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0 [stack]\n"
        )
        self.assertIsNone(loaded_user_mode_driver(maps))

    def test_what_it_returns_passes_the_corpus_validator(self):
        maps = ("7f0a00000000-7f0a00001000 r--p 00000000 fd:01 1 "
                "/opt/cuda-compat-13.0/compat/libcuda.so.580.95.05\n")
        bm._check_driver_version(loaded_user_mode_driver(maps),
                                 column="umd_version")


class TestTheTwoFormsTogetherReadAsASubstitution(unittest.TestCase):
    """The parsers and the one comparison that reads them, end to end: the
    H100 box's real pair must come out as a substituted driver, and a native
    box's pair must not."""

    def test_the_h100_boxs_compat_pair_is_driver_substituted(self):
        row = {
            "kmd_version": kernel_mode_driver(_PACKAGED),
            "umd_version": loaded_user_mode_driver(
                "7f0a00000000-7f0a00001000 r--p 00000000 fd:01 1 "
                "/scratch/kb-work/cuda-compat-13.0/compat/libcuda.so.580.95.05\n"),
        }
        self.assertEqual(row, {"kmd_version": "575.57.08",
                               "umd_version": "580.95.05"})
        self.assertIs(bm.is_driver_substituted(row), True)

    def test_a_native_box_is_not(self):
        row = {
            "kmd_version": kernel_mode_driver(_OPEN_MODULE),
            "umd_version": loaded_user_mode_driver(
                "7f0a00000000-7f0a00001000 r--p 00000000 fd:01 1 "
                "/usr/lib/x86_64-linux-gnu/libcuda.so.595.84\n"),
        }
        self.assertIs(bm.is_driver_substituted(row), False)


def _env(**over) -> Env:
    base = dict(torch_version="2.13.0+cu130", device="cuda",
                device_name="NVIDIA H100 80GB HBM3", gpu_sku="H100-SXM",
                peak_bw_gbps=3350.0, sm="sm_90",
                libs={"flash_attn": "3.0.0"})
    base.update(over)
    return Env(**base)


class TestTheRowCarriesWhatTheEnvObserved(unittest.TestCase):
    """`build_row` / `build_refusal_row` stamp the pair from the `Env`.

    Both assertions use DISTINCT driver values that appear nowhere else, so
    they fail not only if a builder stops reading `env` but also if it reads
    the wrong one of the two — a swap that an equal-valued fixture could not
    see.
    """

    def _invocation_and_provider(self):
        """A GEMM cell with its key stated, as a planned job carries it, and
        a provider stub. This file measures nothing — the cell is a label, and
        what is under test is which driver reaches the row."""
        import json
        from kernel_bench.invocation import Invocation
        dims = {"B": 1, "s_query": 4, "d_reduce": 8, "d_out": 16}
        dtype_env = {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                     "output": "bf16", "metadata": "fp32"}
        canonical = {"sort_keys": True, "separators": (",", ":")}
        inv = Invocation(
            signature="gemm", op="linear", dims=dims, dtype_env=dtype_env,
            phase="prefill", tp=1, model="m/m", spec_hash="abc123",
            node_spec_id="n", flops_per_rank=1e9, access_bytes_per_rank=1e6,
            physical_key=("gemm", json.dumps(dims, **canonical),
                          json.dumps(dtype_env, **canonical), "prefill"))
        prov = type("P", (), {
            "library": "lib", "kernel_impl": "k", "computes_at": None,
            "actual_computes_at": lambda self, inv: {},
            "version": lambda self, env: "1.0",
            "impl_id": lambda self, i: "k"})()
        return inv, prov

    def _row(self, env):
        from kernel_bench.emit import build_row
        from kernel_bench.timing import LatencyStats
        inv, prov = self._invocation_and_provider()
        return build_row(inv, prov,
                         LatencyStats(p50_us=5.0, p95_us=6.0, mean_us=5.5,
                                      samples=100, timing_method="l2_drain"),
                         env, date="2026-10-22", contributor="@t")

    def test_build_row_stamps_both_drivers(self):
        row = self._row(_env(kmd_version="575.57.08", umd_version="580.95.05"))
        self.assertEqual(row["kmd_version"], "575.57.08")
        self.assertEqual(row["umd_version"], "580.95.05")
        # And the row says, through the one place that reads the pair, that
        # it was measured under a substituted user-mode driver.
        self.assertIs(bm.is_driver_substituted(row), True)

    def test_build_refusal_row_stamps_both_drivers(self):
        """A refusal is a dated observation about (cell, impl, card) too, so
        it carries which driver refused — a cell that OOMs under one driver
        is not evidence about another."""
        from kernel_bench.emit import build_refusal_row
        inv, prov = self._invocation_and_provider()
        env = _env(kmd_version="575.57.08", umd_version="580.95.05")
        row = build_refusal_row(inv, prov, env, outcome="oom",
                                date="2026-10-22", contributor="@t",
                                impl_id="flash_attention_v3",
                                framework_version="3.0.0")
        self.assertEqual(row["kmd_version"], "575.57.08")
        self.assertEqual(row["umd_version"], "580.95.05")

    def test_build_row_records_the_clock_that_timed_it(self):
        """The clock is written by the clock: `time_callable` returns it with
        the latency and the row copies it. A builder that stamped a constant
        would claim the drain for a `flush_l2=False` or CPU timing."""
        from kernel_bench.emit import build_row
        from kernel_bench.timing import LatencyStats
        inv, prov = self._invocation_and_provider()
        for method in ("warm_l2", "l2_drain"):
            row = build_row(inv, prov, LatencyStats(5.0, 6.0, 5.5, 100, method),
                            _env(), date="2026-10-26", contributor="@t")
            self.assertEqual(row["timing_method"], method)
            bm.validate_row(row)

    def test_a_refusal_names_no_clock(self):
        """Nothing was timed on a refusal, so it names no clock — an empty
        column there is not the old clock's mark, it is outcome != measured."""
        from kernel_bench.emit import build_refusal_row
        inv, prov = self._invocation_and_provider()
        row = build_refusal_row(inv, prov, _env(), outcome="oom",
                                date="2026-10-26", contributor="@t",
                                impl_id="k", framework_version="1.0")
        self.assertEqual(row["timing_method"], "")

    def test_an_env_that_observed_nothing_leaves_both_empty(self):
        """Excludes a builder that defaults one driver to the other. Unknown
        must stay unknown: a row claiming a native configuration nobody saw
        is the thing this whole pair exists to prevent."""
        row = self._row(_env(kmd_version=None, umd_version=None))
        self.assertEqual(row["kmd_version"], "")
        self.assertEqual(row["umd_version"], "")
        self.assertIsNone(bm.is_driver_substituted(row))

    def test_a_half_observed_env_is_stamped_as_observed(self):
        """The box's driver read but no `libcuda` mapped (a CPU-only env on a
        GPU box). The row carries what was seen and the pair reads unknown —
        it does not drop the half that WAS observed."""
        row = self._row(_env(kmd_version="595.84", umd_version=None))
        self.assertEqual(row["kmd_version"], "595.84")
        self.assertEqual(row["umd_version"], "")
        self.assertIsNone(bm.is_driver_substituted(row))


class TestOnThisMachine(unittest.TestCase):
    """`detect_env` against the real files, when there are any. Skipped on a
    box with no NVIDIA kernel module, so CI does not need a card."""

    def test_the_box_driver_is_read_and_is_schema_shaped(self):
        if not Path("/proc/driver/nvidia/version").exists():
            self.skipTest("no NVIDIA kernel module on this box")
        from kernel_bench.env import _NVRM_PROC, _read_text
        observed = kernel_mode_driver(_read_text(_NVRM_PROC))
        self.assertIsNotNone(
            observed,
            "this box has an NVIDIA kernel module but its version did not "
            "parse — the form is new and the pattern needs it")
        bm._check_driver_version(observed, column="kmd_version")


if __name__ == "__main__":
    unittest.main()
