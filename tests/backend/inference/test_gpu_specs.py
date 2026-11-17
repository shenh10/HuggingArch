"""GPU spec loader + bandwidth-fabric helpers.

Locks the contract that backs the per-op timing pass's intra vs inter
bandwidth split: each SKU exposes a separate NVLink/PCIe (intra-node)
and RDMA/PCIe (inter-node) fabric, with `gpus_per_node` deciding which
one a collective takes based on its world size.

A single intra-node figure would price an EP=64 all_to_all on an
8-GPU/node SKU at 900 GB/s NVLink and under-report comm time by ~3×;
the helpers under test keep the two columns apart and label them.
"""
from __future__ import annotations

import dataclasses
import unittest
import unittest.mock

#: `dataclasses.fields` before the mutation test patches it.
_real_fields = dataclasses.fields

from backend.arch.dtypes import UnknownDtype
from backend.inference.gpu_specs import (
    GPUSpec,
    get_gpu,
    gpus_per_node,
    intra_bw_label,
    inter_bw_label,
    peak_flops_for_dtype,
    peak_inter_bw_bytes_per_sec,
    peak_intra_bw_bytes_per_sec,
    load_gpu_specs,
)


class TestGPUSpecsBandwidthLanes(unittest.TestCase):
    """Per-SKU intra-node + inter-node bandwidth split."""

    def test_h100_sxm_nvlink_and_rdma(self):
        gpu = get_gpu("H100-SXM")
        self.assertIsNotNone(gpu)
        # Intra-node = NVLink 4 (per-direction: 450 GB/s = 900 GB/s
        # bidirectional aggregate halved), inter-node = RDMA CX-7
        # (400 Gbps per-direction = 50 GB/s). gpus_per_node = 8 (HGX).
        self.assertEqual(peak_intra_bw_bytes_per_sec(gpu), 450e9)
        self.assertEqual(peak_inter_bw_bytes_per_sec(gpu), 50e9)
        self.assertEqual(gpus_per_node(gpu), 8)
        self.assertEqual(intra_bw_label(gpu), "NVLink")
        self.assertEqual(inter_bw_label(gpu), "RDMA")

    def test_h100_pcie_falls_back_to_pcie_for_intra(self):
        # PCIe-form-factor SKU has no NVLink — the intra fabric degrades to
        # PCIe Gen5 x16 (64 GB/s per direction). The inter fabric is still
        # RDMA when the spec sheet lists it (H100-PCIe has CX-7 NICs).
        gpu = get_gpu("H100-PCIe")
        self.assertIsNotNone(gpu)
        self.assertEqual(peak_intra_bw_bytes_per_sec(gpu), 64e9)
        self.assertEqual(peak_inter_bw_bytes_per_sec(gpu), 50e9)
        self.assertEqual(gpus_per_node(gpu), 4)
        self.assertEqual(intra_bw_label(gpu), "PCIe")
        self.assertEqual(inter_bw_label(gpu), "RDMA")

    def test_a100_sxm_fabrics(self):
        # A100-40G-SXM: NVLink 3 intra + 200 Gb/s HDR IB inter (DGX/HGX config).
        gpu = get_gpu("A100-40G-SXM")
        self.assertIsNotNone(gpu)
        # NVLink 3 = 600 GB/s bidir agg → 300 GB/s per direction.
        self.assertEqual(peak_intra_bw_bytes_per_sec(gpu), 300e9)
        # 200 Gb/s HDR InfiniBand ÷ 8 = 25 GB/s per direction, per GPU.
        self.assertEqual(peak_inter_bw_bytes_per_sec(gpu), 25e9)

    def test_gb200_nvl72_per_gpu_domain(self):
        # GB200 NVL72: one NVLink domain spans 72 GPUs. A collective with
        # comm_world ≤ 72 stays on NVLink 5 (per-direction = 900 GB/s).
        gpu = get_gpu("GB200")
        self.assertIsNotNone(gpu)
        self.assertEqual(gpus_per_node(gpu), 72)
        self.assertEqual(peak_intra_bw_bytes_per_sec(gpu), 900e9)

    def test_gpus_per_node_refuses_a_missing_column(self):
        """The CSV column is the declaration; there is no "typical HGX"
        number to fall back on (GB200 NVL72 is 72). Every shipped SKU
        declares it, and a missing one raises naming the SKU."""
        with self.assertRaises(ValueError):
            gpus_per_node(None)
        from backend.inference.gpu_specs import load_gpu_specs
        for gpu in load_gpu_specs():
            self.assertGreaterEqual(gpus_per_node(gpu), 1, gpu.sku)

    def test_every_sku_declares_its_device(self):
        """A framework's dispatch can be platform-gated (sglang's `_is_cuda`);
        the SKU sheet says which platform each GPU is, not its arch name."""
        from backend.inference.gpu_specs import load_gpu_specs
        for gpu in load_gpu_specs():
            self.assertEqual(gpu.device, "cuda", gpu.sku)

    def test_every_cuda_sku_states_its_compute_capability(self):
        """A framework selects some kernels by compute capability (vLLM's
        FlashInfer MXFP8 linears: `has_device_capability(100)`); a binding's
        `when.min_compute_capability` reads it off the sheet, so every CUDA SKU
        states one, as `major.minor`."""
        from backend.inference.gpu_specs import load_gpu_specs
        for gpu in load_gpu_specs():
            major, _, minor = gpu.compute_capability.partition(".")
            self.assertTrue(major.isdigit() and minor.isdigit(), gpu.sku)
        self.assertEqual(get_gpu("H100-SXM").compute_capability, "9.0")
        self.assertEqual(get_gpu("B200").compute_capability, "10.0")

    def test_a_cuda_row_without_a_compute_capability_is_refused(self):
        """The loader refuses a CUDA row that leaves the column empty: a new
        SKU without it would silently lose every kernel a binding selects
        by capability."""
        from backend.inference.gpu_specs import check_compute_capability
        from dataclasses import replace
        h100 = get_gpu("H100-SXM")
        check_compute_capability(h100)
        for bad in ("", "10", "ten.0"):
            with self.assertRaisesRegex(ValueError, "compute_capability"):
                check_compute_capability(replace(h100, compute_capability=bad))

    def test_a_row_without_a_memory_bandwidth_is_refused(self):
        """The loader refuses a row that leaves `mem_bw_gbps` empty, and so
        does `peak_hbm_bw` on such a card: every roofline divides a step's
        access bytes by that figure, so a blank read as "no limit" would
        price every byte at zero seconds and the card would read as
        infinitely fast. Every sheet row states one today, which is why
        `peak_hbm_bw` returning None means exactly "no card was chosen"."""
        from dataclasses import replace
        from backend.inference.gpu_specs import (
            NoPeakBandwidth, check_memory_bandwidth, peak_hbm_bw)
        h100 = get_gpu("H100-SXM")
        check_memory_bandwidth(h100)
        for bad in (None, 0.0, float("nan"), float("inf")):
            blank = replace(h100, mem_bw_gbps=bad)
            with self.assertRaisesRegex(NoPeakBandwidth, "mem_bw_gbps"):
                check_memory_bandwidth(blank)
            with self.assertRaisesRegex(NoPeakBandwidth, "H100-SXM"):
                peak_hbm_bw(blank)
        self.assertIsNone(peak_hbm_bw(None))
        for spec in load_gpu_specs():
            self.assertGreater(spec.mem_bw_gbps, 0, spec.sku)

    def test_label_helpers_empty_when_no_bw(self):
        # Defensive: with a hand-built GPU spec missing every BW column,
        # label helpers return "" rather than asserting.
        empty = GPUSpec(
            sku="x", arch="x", device="cuda", form_factor="x",
            fp32_tflops=None, tf32_tflops=None, bf16_tflops=None,
            fp16_tflops=None, int8_tops=None, fp8_tflops=None,
            fp4_tflops=None, nvlink_gbps=None, nvls="no", pcie_gbps=None,
            rdma_gbps=None, mem_gib=None, mem_type="", mem_bw_gbps=None,
            tdp_w=None, gpus_per_node=None,
        )
        self.assertEqual(intra_bw_label(empty), "")
        self.assertEqual(inter_bw_label(empty), "")
        self.assertIsNone(peak_intra_bw_bytes_per_sec(empty))
        self.assertIsNone(peak_inter_bw_bytes_per_sec(empty))


class TestPeakFlopsByName(unittest.TestCase):
    """Peak column choice is a dtype name, not a byte width.

    BF16 and FP16 are both 2 bytes. Mapping width → column collapsed
    them onto BF16, so an FP16 deployment on a SKU with unequal
    columns silently billed the wrong peak.
    """

    def _gpu(self, **peaks) -> GPUSpec:
        kwargs = dict(
            sku="x", arch="x", device="cuda", form_factor="x",
            fp32_tflops=None, tf32_tflops=None,
            bf16_tflops=None, fp16_tflops=None,
            int8_tops=None, fp8_tflops=None, fp4_tflops=None,
            nvlink_gbps=None, nvls="no", pcie_gbps=None, rdma_gbps=None,
            mem_gib=None, mem_type="", mem_bw_gbps=None, tdp_w=None,
        )
        kwargs.update(peaks)
        return GPUSpec(**kwargs)

    def test_bf16_and_fp16_read_different_columns(self):
        gpu = self._gpu(bf16_tflops=100, fp16_tflops=200)
        self.assertEqual(peak_flops_for_dtype(gpu, "BF16"), 100e12)
        self.assertEqual(peak_flops_for_dtype(gpu, "FP16"), 200e12)
        self.assertEqual(peak_flops_for_dtype(gpu, "F16"), 200e12)

    def test_matched_precision_picks_fp4_fp8_bf16(self):
        gpu = self._gpu(fp4_tflops=800, fp8_tflops=400, bf16_tflops=100)
        self.assertEqual(peak_flops_for_dtype(gpu, "FP4"), 800e12)
        self.assertEqual(peak_flops_for_dtype(gpu, "FP8"), 400e12)
        self.assertEqual(peak_flops_for_dtype(gpu, "BF16"), 100e12)

    def test_missing_column_is_none_not_bf16_fallback(self):
        gpu = get_gpu("A100-80G-SXM")
        self.assertIsNotNone(gpu)
        self.assertIsNone(peak_flops_for_dtype(gpu, "FP8"))
        self.assertEqual(peak_flops_for_dtype(gpu, "BF16"), 312e12)

    def test_scale_dtype_has_no_mma_column(self):
        gpu = self._gpu(bf16_tflops=100)
        with self.assertRaises(UnknownDtype):
            peak_flops_for_dtype(gpu, "F8_E8M0")


class TestGPUSpecsLoader(unittest.TestCase):
    """CSV parsing + dataclass shape — guard the column contract."""

    def test_every_sku_has_gpus_per_node(self):
        # Either populated, or None (helper fallback to 8 kicks in).
        # Catches the case where a CSV column drift loses the column.
        for g in load_gpu_specs():
            # gpus_per_node should resolve to a positive int.
            self.assertGreater(gpus_per_node(g), 0)

    def test_every_sku_declares_an_rdma_link(self):
        # A cross-node collective or KV handoff is priced against the RDMA
        # column and nothing else (PCIe is intra-node). A SKU added without
        # it would report no cross-node bandwidth at all, so the column is
        # required, and the number must be a positive per-direction GB/s.
        for g in load_gpu_specs():
            with self.subTest(sku=g.sku):
                self.assertIsNotNone(g.rdma_gbps)
                self.assertGreater(float(g.rdma_gbps), 0.0)

    def test_no_sku_prices_bf16_and_fp16_differently(self):
        """The premise behind serving a float32 checkpoint at ONE dtype.

        vLLM and sglang disagree about which 16-bit float a non-gemma
        float32 checkpoint is loaded at — bfloat16 and float16 respectively
        (`dtypes.served_dtype`) — and the model answers bfloat16 for both.
        That is only unobservable while the two dtypes are the same width
        (they are, by the dtype table) AND read the same GPU peak, which is
        a property of this CSV and was not checked anywhere. A SKU whose
        two columns differ makes the single answer a wrong number on one of
        the two frameworks, and the fix is then a framework dimension on
        the walk, not a new row here.

        `peak_flops_for_dtype` deliberately keeps the two columns apart
        (`TestPeakFlopsByName`); this says the catalog never uses that.
        """
        for g in load_gpu_specs():
            with self.subTest(sku=g.sku):
                self.assertEqual(
                    peak_flops_for_dtype(g, "BF16"),
                    peak_flops_for_dtype(g, "F16"),
                    f"{g.sku} prices BF16 and FP16 differently, so "
                    f"`dtypes.served_dtype` answering one dtype for both "
                    f"frameworks is now observable on this SKU")

    def test_catalog_lists_the_touchstone_skus(self):
        skus = {g.sku for g in load_gpu_specs()}
        # Should include the SKUs the cost model is regularly tested
        # against — touchstones for "did the CSV silently shrink".
        for sku in ("H100-SXM", "H200", "A100-80G-SXM", "B200", "GB200"):
            self.assertIn(sku, skus)


class TestEveryColumnIsClassified(unittest.TestCase):
    """Every column of the sheet says what it changes: a WALK, a PRICE, or
    nothing.

    The gate that holds the one-keyed-walk split up. A walk is reused
    across the cards of one `framework_bindings.CardClass`, so a sheet
    column a walk reads with nobody having declared it a walk fact leaves
    that class too coarse and every reuse inside it publishes one card's
    rows as another's (`capacity._refuse_unless_one_class` names the two
    times that shipped). The classification is checked at load by
    `framework_bindings._check_card_fact_classification`; this test adds a
    column to `GPUSpec` and shows the check goes red for it.
    """

    def _classification(self):
        from backend.inference import framework_bindings as fb
        from backend.inference import gpu_specs as gs
        return (fb, gs, fb._card_walk_columns(), set(gs.PRICE_COLUMNS),
                set(gs.DESCRIPTIVE_COLUMNS))

    def test_the_sheet_is_covered_today(self):
        fb, gs, walk, price, descriptive = self._classification()
        columns = {f.name for f in dataclasses.fields(gs.GPUSpec)}
        self.assertEqual(columns - walk - price - descriptive, set())
        # "descriptive-only" is a claim that nothing reads it.
        self.assertEqual(descriptive & (walk | price), set())
        # The walk half is derived, not listed: the dispatch's own card
        # keys, declared runtime geometry, and the peak columns.
        self.assertEqual(walk, (set(fb.DispatchCondition.CARD_KEYS.values())
                                & columns) | set(gs.PEAK_COLUMNS.values())
                         | fb.ServingDeployment.RUNTIME_CARD_FIELDS)

    def test_a_new_column_fails_at_load(self):
        """Mutation proof: a field added to `GPUSpec` and classified
        nowhere is refused, naming the column and the three buckets. The
        mutation is a real dataclass with the extra column, so it does not
        depend on how the checker imported `dataclasses.fields`."""
        fb, gs, _walk, _price, _descriptive = self._classification()
        mutated = dataclasses.make_dataclass(
            "GPUSpecPlusOne", [("sparsity_tops", "Optional[float]", None)],
            bases=(gs.GPUSpec,))
        with unittest.mock.patch.object(gs, "GPUSpec", mutated), \
                self.assertRaises(ValueError) as caught:
            fb._check_card_fact_classification()
        message = str(caught.exception)
        self.assertIn("sparsity_tops", message)
        self.assertIn("PRICE_COLUMNS", message)
        self.assertIn("DESCRIPTIVE_COLUMNS", message)
        # And it passes again once the new column says what it changes.
        with unittest.mock.patch.object(gs, "GPUSpec", mutated), \
                unittest.mock.patch.object(
                    gs, "PRICE_COLUMNS",
                    frozenset({*gs.PRICE_COLUMNS, "sparsity_tops"})):
            fb._check_card_fact_classification()

    def test_a_descriptive_column_a_peak_starts_reading_fails(self):
        """The other half of the mutation: a column declared
        descriptive-only that a dtype row then declares a peak for. The
        claim "nothing reads it" is refused rather than left standing —
        `tf32_tflops` is the live case (no `arch.dtypes` row declares
        `tf32`, so no peak can select that column)."""
        fb, gs, _walk, _price, _descriptive = self._classification()
        self.assertIn("tf32_tflops", gs.DESCRIPTIVE_COLUMNS)
        self.assertNotIn("tf32", gs.PEAK_COLUMNS)
        with unittest.mock.patch.object(
                gs, "PEAK_COLUMNS",
                {**gs.PEAK_COLUMNS, "tf32": "tf32_tflops"}), \
                self.assertRaises(ValueError) as caught:
            fb._check_card_fact_classification()
        self.assertIn("tf32_tflops", str(caught.exception))
        self.assertIn("descriptive", str(caught.exception))

    def test_a_dispatch_key_the_sheet_cannot_answer_fails(self):
        """A `when:` key classified as a card fact that reads a
        `ServingDeployment` field the sheet has no column for: the card
        class would carry a fact no row states."""
        fb, _gs, _walk, _price, _descriptive = self._classification()
        with unittest.mock.patch.object(
                fb.DispatchCondition, "CARD_KEYS",
                {**fb.DispatchCondition.CARD_KEYS,
                 "min_pcie_generation": "pcie_generation"}), \
                self.assertRaises(ValueError) as caught:
            fb._card_walk_columns()
        self.assertIn("pcie_generation", str(caught.exception))

    def test_a_serving_card_field_cannot_escape_card_class(self):
        """The fourth assertion on the split: what a serving launch reads of a
        card is the dispatch fields plus peak-column existence — no third
        undeclared route into a walk."""
        from backend.inference import framework_bindings as fb

        self.assertEqual(fb.ServingDeployment.CARD_FIELDS, {
            "device", "compute_capability", "peak_families", "sm_count",
        })
        with unittest.mock.patch.object(
                fb.DispatchCondition, "CARD_KEYS",
                {**fb.DispatchCondition.CARD_KEYS, "nvls": "nvls"}), \
                self.assertRaises(ValueError) as caught:
            fb._check_card_fact_classification()
        message = str(caught.exception)
        self.assertIn("ServingDeployment.CARD_FIELDS", message)
        self.assertIn("third card fact outside CardClass", message)


if __name__ == "__main__":
    unittest.main()
