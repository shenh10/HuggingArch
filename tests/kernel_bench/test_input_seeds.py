"""A cell's inputs are the same data every time it is measured.

`kernel_bench` drew every input from the process-global RNG, so what a kernel
was measured and cross-checked on depended on the order the jobs happened to
run in. The driver's own notes record what that cost: the same minimal
`mla_sparse_attention` cell flipped its cross-check verdict four times in six
reruns, and `gated_delta_rule`'s relative error wandered between 0.022 and
0.060 — the verdict was the draw, not the kernel.

Now each cell has one seed, derived from its measurement key
(`Invocation.input_seed`), and every input is drawn from an explicit
generator (`refs.input_generator`) handed to the builder. These tests pin the
properties that make a verdict attributable: the data is the cell's, it is
the same on both cards and across reruns, it does not come from (or disturb)
the global RNG, two different cells get different data, every provider of one
cell is measured on one batch of it, and the row says which seed it was.

Tolerances are not touched by any of it: the stability comes from the data
being fixed, not from a wider band, so the band's constants are pinned here
too.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from kernel_bench.invocation import Invocation

#: torch's random constructors. Every one of them takes `generator=`, which
#: is the whole point: a call site without it reads the global RNG.
TORCH_RANDOM_CALLS = ("randn", "rand", "randint", "randperm")

KERNEL_BENCH = PROJECT_ROOT / "kernel_bench"


def _env(gpu_sku: str = "H100-SXM"):
    """A stated env: this file measures nothing, so the card is a label."""
    from kernel_bench.env import Env
    return Env(torch_version="2.9.0", device=None,
               device_name=gpu_sku, gpu_sku=gpu_sku, peak_bw_gbps=3350.0)


def _cell(**over) -> Invocation:
    """A GEMM cell with its key stated, as a planned job carries it."""
    import json
    dims = {"B": 1, "s_query": 4, "d_reduce": 8, "d_out": 16}
    dims.update(over.pop("dims", {}))
    dtype_env = {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                 "output": "bf16", "metadata": "fp32"}
    dtype_env.update(over.pop("dtype_env", {}))
    phase = over.pop("phase", "prefill")
    signature = over.pop("signature", "gemm")
    canonical = {"sort_keys": True, "separators": (",", ":")}
    return Invocation(
        signature=signature, op="linear", dims=dims, dtype_env=dtype_env,
        phase=phase, tp=1,
        physical_key=(signature, json.dumps(dims, **canonical),
                      json.dumps(dtype_env, **canonical), phase),
        **over)


def _inputs(inv: Invocation, device: str = "cpu") -> Dict[str, Any]:
    from kernel_bench.refs import get_ref, input_generator
    return get_ref(inv.signature).make_inputs(inv, device,
                                              input_generator(inv, device))


def _same_bytes(a, b) -> bool:
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


class TestTheDataIsTheCells(unittest.TestCase):
    def test_two_builds_of_one_cell_are_the_same_bytes(self):
        inv = _cell()
        first, second = _inputs(inv), _inputs(inv)
        self.assertEqual(sorted(first), sorted(second))
        for name, value in first.items():
            if hasattr(value, "shape"):
                self.assertTrue(_same_bytes(value, second[name]), name)

    def test_the_seed_does_not_carry_the_card(self):
        """Two runs of one cell on different cards draw the same data — the
        seed's inputs are the four key fields, and none of them is the card
        (`driver.physical_key_for` stamps the card on the ROW)."""
        from kernel_bench.driver import physical_key_for
        inv = _cell()
        keys = [physical_key_for(inv, _env(sku))
                for sku in ("H100-SXM", "B300")]
        self.assertNotEqual(keys[0].gpu, keys[1].gpu)
        self.assertEqual(keys[0].shape_json, keys[1].shape_json)
        # The card changed, the seed did not.
        self.assertEqual(inv.input_seed, _cell().input_seed)

    def test_a_different_cell_is_different_data(self):
        """One minimal change to each key field moves the seed."""
        base = _cell().input_seed
        for name, cell in (
                ("shape", _cell(dims={"d_out": 32})),
                ("dtype_env", _cell(dtype_env={"weight": "fp8_e4m3"})),
                ("phase", _cell(phase="decode")),
                ("signature", _cell(signature="gemm_bias"))):
            with self.subTest(field=name):
                self.assertNotEqual(cell.input_seed, base)

    def test_two_cells_are_built_on_different_data(self):
        """And the generator is the seed's, not a fixed one: two cells whose
        keys differ only in the phase have the same shapes and different
        bytes."""
        a, b = _cell(phase="prefill"), _cell(phase="decode")
        self.assertNotEqual(a.input_seed, b.input_seed)
        first, second = _inputs(a), _inputs(b)
        moved = [name for name, value in first.items()
                 if hasattr(value, "shape")
                 and not _same_bytes(value, second[name])]
        self.assertEqual(sorted(moved), sorted(
            name for name, value in first.items() if hasattr(value, "shape")))

    def test_a_cell_with_no_key_has_no_seed(self):
        """The key is the plan's; a job without one cannot be measured, and
        it cannot be seeded either (rather than seeded off something else)."""
        inv = Invocation(signature="gemm", op="linear", dims={"B": 1},
                         dtype_env={"activation": "bf16"}, phase="prefill",
                         tp=1)
        with self.assertRaises(ValueError):
            _ = inv.input_seed


class TestTheGlobalRngIsNotInvolved(unittest.TestCase):
    def test_building_inputs_leaves_the_global_rng_where_it_was(self):
        import torch
        before = torch.get_rng_state()
        _inputs(_cell())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_the_data_does_not_follow_a_global_seed(self):
        """Seeding the global RNG differently either side of the build gives
        the same inputs — which is what says they came from `gen`."""
        import torch
        inv = _cell()
        torch.manual_seed(12345)
        first = _inputs(inv)
        torch.manual_seed(999)
        second = _inputs(inv)
        for name, value in first.items():
            if hasattr(value, "shape"):
                self.assertTrue(_same_bytes(value, second[name]), name)


def _random_calls_without_generator(path: Path) -> List[Tuple[int, str]]:
    """Call sites in `path` that draw from torch's RNG without naming a
    generator — by AST, so a mention in a docstring is not a call."""
    tree = ast.parse(path.read_text())
    out: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name) else "")
        if name not in TORCH_RANDOM_CALLS:
            continue
        if any(kw.arg == "generator" for kw in node.keywords):
            continue
        out.append((node.lineno, name))
    return out


class TestTheRngBoundaryIsAtTheBuilders(unittest.TestCase):
    """Only `refs` draws data, and only from the cell's generator.

    By call SHAPE rather than by a list of allowed functions: a torch random
    constructor without `generator=` reads the global RNG, whoever calls it.
    """

    def test_no_builder_draws_from_the_global_rng(self):
        bad = _random_calls_without_generator(KERNEL_BENCH / "refs.py")
        self.assertEqual(
            bad, [], f"refs.py draws without the cell's generator at {bad}")

    def test_no_provider_draws_data_at_all(self):
        """A provider arranges, packs and quantizes — all deterministic. One
        that generates its own data would be measured on inputs the
        reference never saw, and the cross-check would be comparing two
        different questions."""
        for path in sorted((KERNEL_BENCH / "providers").glob("*.py")):
            with self.subTest(module=path.name):
                calls = [(line, name) for line, name in
                         _random_calls_without_generator(path)]
                source = path.read_text()
                self.assertEqual(calls, [], f"{path.name} draws data at {calls}")
                self.assertNotIn("manual_seed", source)

    def test_the_scan_sees_a_bare_call(self):
        """The scan itself, on a call it must catch and one it must not."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write("import torch\n"
                     "a = torch.randn(3)\n"
                     "b = torch.randn(3, generator=g)\n")
            tmp = Path(fh.name)
        self.assertEqual(_random_calls_without_generator(tmp), [(2, "randn")])
        tmp.unlink()


class TestEveryBuilderTakesTheGenerator(unittest.TestCase):
    def test_registration_refuses_a_builder_without_it(self):
        from kernel_bench.refs import RefSpec, register_ref
        with self.assertRaises(ValueError) as caught:
            register_ref("test_signature_without_gen",
                         RefSpec(lambda inv, device: {}))
        self.assertIn("gen", str(caught.exception))

    def test_every_registered_builder_has_the_three_parameters(self):
        import inspect
        from kernel_bench.refs import _BUILDER_PARAMS, _REFS
        self.assertTrue(_REFS)
        for signature, spec in _REFS.items():
            with self.subTest(signature=signature):
                self.assertEqual(
                    tuple(inspect.signature(spec.make_inputs).parameters),
                    _BUILDER_PARAMS)


class TestOneBatchOfInputsPerCell:
    """The driver builds a cell's inputs once and measures every provider of
    the cell on them: two kernels timed on different data are not comparable,
    and neither is either of them against the one reference."""

    def test_the_driver_builds_once_and_shares(self, monkeypatch):
        import kernel_bench.driver as drv
        from kernel_bench.providers import BenchCase
        from kernel_bench.timing import LatencyStats

        inv = _cell()
        job = {"op": "linear", "signature": "gemm", "dims": inv.dims,
               "dtype_env": inv.dtype_env, "phase": inv.phase,
               "physical_key": list(inv.physical_key),
               "flops_per_rank": 1.0, "access_bytes_per_rank": 1.0,
               "model": "m/m", "spec_hash": "s", "node_spec_id": "n"}
        built: List[Dict[str, Any]] = []
        seen: List[int] = []
        real = __import__("kernel_bench.refs", fromlist=["x"]).get_ref("gemm")

        class _Spec:
            reference = None

            def make_inputs(self, i, device, gen):
                out = real.make_inputs(i, device, gen)
                built.append(out)
                return out

        def _provider(tag):
            return type("P", (), {
                "library": "lib", "kernel_impl": tag, "signature": "gemm",
                # The kernel computes at the cell's own precision.
                "computes_at": None,
                # A local kernel, not a collective: the driver reads `world`
                # to decide whether a cell needs a process per rank.
                "world": None,
                "actual_computes_at": lambda self, inv: {},
                "weight_group": None, "weight_scale": None,
                "impl_id": lambda self, i: tag,
                "version": lambda self, e: "1.0",
                "convert_inputs": lambda self, i, ins: (seen.append(id(ins))
                                                        or ins),
                "build": lambda self, i, e, ins: BenchCase(
                    run=lambda: None, to_canonical=None),
                "wrap_output": lambda self, i, fn: fn,
            })()

        monkeypatch.setattr(drv, "applicable",
                            lambda inv, env: [_provider("a"), _provider("b")])
        monkeypatch.setattr("kernel_bench.refs.get_ref", lambda sig: _Spec())
        monkeypatch.setattr(drv, "time_callable",
                            lambda fn: LatencyStats(1.0, 1.0, 1.0, 10, "l2_drain"))
        result = drv.run_jobs([job], env=_env(), date="2026-10-13",
                              contributor="@t")
        assert len(built) == 1, "the cell's inputs are built once"
        assert len(seen) == 2 and seen[0] == seen[1], \
            "both providers are handed the same inputs"
        assert len(result.rows) == 2, result.failures


class TestTheRowSaysWhichData(unittest.TestCase):
    def test_the_row_carries_the_seed_and_it_is_the_cells(self):
        from backend.inference import benchmarks as bm
        from kernel_bench.emit import build_row
        from kernel_bench.timing import LatencyStats
        inv = _cell(flops_per_rank=1e9, access_bytes_per_rank=1e6,
                    model="m/m", spec_hash="abc", node_spec_id="n")
        provider = type("P", (), {
            "library": "lib", "kernel_impl": "k", "computes_at": None,
            "actual_computes_at": lambda self, inv: {},
            "version": lambda self, env: "1.0",
            "impl_id": lambda self, i: "k"})()
        row = build_row(inv, provider, LatencyStats(5.0, 5.0, 6.0, 100, "l2_drain"),
                        _env(), date="2026-10-13",
                        contributor="@t", cross_check="passed",
                        rel_err=0.01, cosine=0.9999)
        self.assertEqual(row["input_seed"], str(inv.input_seed))
        self.assertEqual(row["schema_version"], str(bm.KNOWN_SCHEMA_VERSION))
        bm.validate_row(row)
        # Recomputable from the row's own key fields — the column is a
        # function of the identity, which is what makes it checkable.
        recomputed = Invocation(
            signature=row["op"] and inv.signature, op=row["op"],
            dims=inv.dims, dtype_env=inv.dtype_env, phase=row["phase"], tp=1,
            physical_key=(inv.signature, row["shape_json"],
                          row["dtype_env_json"], row["phase"])).input_seed
        self.assertEqual(int(row["input_seed"]), recomputed)

    def test_the_column_refuses_a_number_a_reader_has_already_changed(self):
        from backend.inference import benchmarks as bm
        for bad in ("-1", "1.0", "1e3", "0x10", str(2 ** 64)):
            with self.subTest(value=bad), \
                    self.assertRaises(bm.BenchmarkValidationError):
                bm._check_input_seed(bad)
        bm._check_input_seed("")                      # a pre-v9 row
        bm._check_input_seed(str(2 ** 64 - 1))


class TestTheBandWasNotWidened(unittest.TestCase):
    """A fixed seed is what makes a verdict repeatable; a wider tolerance
    would make failures repeatable too.

    Pinning the band constants is what this used to do, and a change that
    moved no constant walked straight through it: the gate's STATISTIC
    changed, which moved the effective band several-fold while the table
    read the same. So the ratchet holds the DERIVATION instead — each band
    is the largest floor `tools.xcheck_floors` measured for that dtype,
    times one margin — and the floors it was derived from are written down
    beside it. A band can still be changed; it cannot be changed without
    changing a measurement someone can repeat.
    """

    #: The margin every band carries above its floor. One number for both
    #: statistics and every dtype: two margins would be two decisions, and
    #: the first time they differed the gate quietly relaxed on one side
    #: (the worst band was the RMS band x50, three times any measured
    #: floor, while the RMS band sat at 2x).
    MARGIN = 2.5

    #: What `python -m tools.xcheck_floors` measured, as the largest floor
    #: over the signatures it walks: (RMS, worst element). The cell's inputs
    #: are built as the harness builds them and the oracle runs on exactly
    #: those, so this is what the kernel does that the oracle does not —
    #: narrowing its intermediates where the inputs are already in the
    #: dtype, quantising its operands where the provider casts them.
    FLOORS: ClassVar[dict] = {
        "fp16": (0.0004, 0.0050),         # sparse MLA
        "bf16": (0.0031, 0.0402),         # fused MoE / sparse MLA
        "fp8_e4m3": (0.0446, 1.2353),     # fused MoE / GQA
        "fp8_e5m2": (0.0646, 2.7954),     # GQA
    }

    def test_every_band_is_its_measured_floor_times_the_margin(self):
        from kernel_bench.driver import _XCHECK_BANDS
        for dtype, (rms, worst) in self.FLOORS.items():
            with self.subTest(dtype=dtype):
                band = _XCHECK_BANDS[dtype]
                for side, floor in (("typical", rms), ("worst", worst)):
                    want = floor * self.MARGIN
                    got = getattr(band, side)
                    self.assertAlmostEqual(
                        got, want, delta=want * 0.4,
                        msg=f"{dtype}.{side}: {got} is not the measured "
                            f"floor {floor} x{self.MARGIN}")

    def test_fp32_is_an_allowance_and_not_a_floor(self):
        # The narrowing is an identity at fp32, so there is no floor to be a
        # margin above: the pair there allows a kernel to sum in a different
        # order than the oracle, and nothing more.
        from kernel_bench.driver import _XCHECK_BANDS
        band = _XCHECK_BANDS["fp32"]
        self.assertLess(band.typical, min(f[0] for f in self.FLOORS.values()))
        self.assertLess(band.worst, min(f[1] for f in self.FLOORS.values()))

    def test_a_kernel_narrower_than_the_cell_is_still_refused(self):
        """The property the bands exist for, stated as behaviour rather than
        as a number: an fp8 computation of a bf16 cell is outside the bf16
        band. Narrow-for-wide is the 15x step a band can see; the step
        between ADJACENT narrow formats is 2x, and no band with a margin
        above its floor separates those — that one is the recorded kernel
        symbol's to answer."""
        from kernel_bench.driver import _XCHECK_BANDS
        for ran, declared in (("fp8_e4m3", "bf16"), ("fp8_e5m2", "bf16"),
                              ("bf16", "fp16")):
            with self.subTest(ran=ran, declared=declared):
                self.assertGreater(self.FLOORS[ran][0],
                                   _XCHECK_BANDS[declared].typical)


@pytest.mark.skipif("not __import__('torch').cuda.is_available()",
                    reason="a verdict is a GPU kernel's; CPU has no provider")
class TestTheVerdictHoldsAcrossReruns:
    """The two cells the driver's notes name as unstable
    (`mla_sparse_attention`, `gated_delta_rule`), five runs each: one
    outcome, not a distribution.

    Whichever providers this box has, the run is repeatable — the verdicts of
    the rows it measured and the reasons it declined are the same five times,
    and every row carries the cell's seed. At least one of the two cells must
    actually measure, or the property is being asserted over nothing.
    """

    CELLS = ("mla_sparse_attention", "gated_delta_rule")

    def test_five_runs_give_one_outcome(self):
        from kernel_bench.driver import run_jobs
        from kernel_bench.env import detect_env
        from kernel_bench.providers import all_providers
        from kernel_bench.smoke import minimal_dims
        env = detect_env()
        measured = 0
        for signature in self.CELLS:
            provs = [p for p in all_providers() if p.signature == signature
                     and p.available(env)]
            if not provs:
                continue
            dims, _why = minimal_dims(provs[0], "decode")
            if dims is None:
                continue
            inv = _cell(signature=signature, dims=dims, phase="decode")
            job = {"op": signature, "signature": signature, "dims": inv.dims,
                   "dtype_env": inv.dtype_env, "phase": inv.phase,
                   "physical_key": list(inv.physical_key),
                   "flops_per_rank": 1.0, "access_bytes_per_rank": 1.0,
                   "model": "m/m", "spec_hash": "s", "node_spec_id": "n"}
            runs = [run_jobs([dict(job)], env=env, date="2026-10-13",
                             contributor="@t") for _ in range(5)]
            outcomes = {(tuple(sorted((r["kernel_impl"], r["cross_check"],
                                       r["input_seed"]) for r in run.rows)),
                         tuple(sorted(run.failures)),
                         tuple(sorted(run.skipped))) for run in runs}
            assert len(outcomes) == 1, f"{signature}: {outcomes}"
            for run in runs:
                for row in run.rows:
                    assert row["input_seed"] == str(inv.input_seed)
            measured += 1 if runs[0].rows else 0
        assert measured, ("neither cell measured on this box; the "
                          "repeatability claim needs a row")


if __name__ == "__main__":
    unittest.main()
