"""An Ascend sheet cell is an upper bound on what the hardware reached, and
its compute cells are the chip's own physics.

A roofline divides by the cell, so a cell BELOW the hardware makes every
consumer read above 100% of peak — the defect that put 1,439 RTX-4090 rows
over peak (`gpu_specs` module docstring). Two facts are pinned here:

* the per-die physics CANN and torch_npu both report for this chip
  (`Ascend910_9362`: 20 cube cores, 8192 FLOP/cycle from a 16x16x16 cube),
  so the compute cells stay derived from cores x rate x clock rather than
  from a vendor figure nobody can check; and
* the throughput a probe actually reached, as a floor under each cell.

The floor is never a replacement for the cell: a kernel reaching 79% of a
cell says nothing about whether the cell is the chip's peak.

Measured 2026-11-14 on an Atlas A3 node, one die, inside the
`quay.io/ascend/vllm-ascend:v0.30.0-a3` container (torch_npu 2.10.0.post7):

    torch.matmul bf16        248.4 TFLOPs   (4096^3, best and median)
    torch.matmul fp32         63.3 TFLOPs   (4096^3; the cube's quarter rate)
    npu_quant_matmul int8    392.5 TOPs     (8192^3)
    copy_ bytes/s           1262 GB/s       (1 GiB)
    HBM capacity              64 GiB        (npu-smi -t memory: 65536 MiB)

The bf16 figure implies 1516 MHz sustained against CANN's 1500 MHz nominal,
so the cell is the rate at the clock the part holds, not at the nominal one.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.gpu_specs import get_gpu

#: The best each probe reached on one die. A cell at or above these is a
#: bound; a cell below one is a sheet the hardware has already beaten.
REACHED = {"bf16_tflops": 248.4, "fp16_tflops": 248.4, "int8_tops": 392.5,
           "mem_bw_gbps": 1262.0, "fp32_tflops": 63.3}

#: The best BUS bandwidth an HCCL collective reached per die, over the links
#: the `scaleup_gbps` cell prices. Bus, not algorithm, bandwidth: the cell is
#: a link rate, and what crosses the link is the ring's share of the payload
#: (`comm_volume.ring_factor`: 2(N-1)/N for an all-reduce, (N-1)/N for the
#: others), so the two are only comparable after that coefficient. Measured
#: 2026-11-15, 128 MiB payloads, torchrun one process per die:
#:
#:   16 dies, every package  186.4 GB/s  (all-gather; all-reduce 176.7)
#:    8 dies, one per package 132.8 GB/s (pure HCCS, no package-internal link)
#:
#: The 8-die figure is the lower one BECAUSE it is pure: a group that takes
#: both dies of a package also rides the package-internal SIO link, which is
#: faster than HCCS and lifts the average (a 0-7 group, four SIO pairs among
#: eight dies, reached 213.2 — above the cell, and not a counter-example to
#: it, because that traffic did not all cross HCCS). `npu-smi info -t topo`
#: is the evidence: dies pair 0-1, 2-3, ... over SIO and reach every other
#: die over HCCS_SW.
REACHED_SCALEUP_GBPS = 186.4
DIE_HBM_GIB = 64.0

#: The one Ascend row the sheet carries. A CloudMatrix384 row stood beside
#: it and was removed: every figure on it was this A3 node's, carried over
#: to a 768-die UB supernode nobody here has measured. A sheet row is
#: priced and walked like any other, so carrying one on borrowed numbers
#: publishes an estimate for hardware no measurement stands behind.
SKU = "Ascend-910C-A3"

#: The per-die physics CANN's platform config (`Ascend910_9362.ini`,
#: `[SoCInfo] cube_core_cnt`, `[AICoreSpec] cube_freq` and the 16x16x16
#: cube) and `torch.npu.get_device_properties` (`cube_core_num`) both state.
CUBE_CORES, FLOPS_PER_CYCLE = 20, 8192
NOMINAL_HZ, SUSTAINED_HZ = 1.500e9, 1.516e9


class TestAscendCellsBoundTheProbe(unittest.TestCase):

    def test_every_compute_and_bandwidth_cell_is_at_least_what_ran(self):
        card = get_gpu(SKU)
        self.assertIsNotNone(card, SKU)
        for column, reached in REACHED.items():
            cell = getattr(card, column)
            self.assertIsNotNone(cell, f"{SKU}.{column}")
            self.assertGreaterEqual(cell, reached, f"{SKU}.{column}")

    def test_the_compute_cell_is_the_chips_physics_at_the_clock_it_holds(self):
        """cores x FLOP/cycle x clock, not a figure from a note.

        At the CANN nominal clock the rate is below what a bf16 GEMM
        reproducibly reached, so the cell is at the sustained clock; a cell
        at the nominal one would be under the hardware."""
        nominal = CUBE_CORES * FLOPS_PER_CYCLE * NOMINAL_HZ / 1e12
        sustained = CUBE_CORES * FLOPS_PER_CYCLE * SUSTAINED_HZ / 1e12
        self.assertLess(nominal, REACHED["bf16_tflops"])
        card = get_gpu(SKU)
        self.assertAlmostEqual(card.bf16_tflops, sustained, places=1)
        self.assertAlmostEqual(card.fp16_tflops, sustained, places=1)
        # The cube's int8 rate is twice its bf16 one; it must exceed 1x,
        # since the int8 measurement exceeds the bf16 cell.
        self.assertAlmostEqual(card.int8_tops, 2 * sustained, places=1)
        self.assertGreater(REACHED["int8_tops"], card.bf16_tflops)

    def test_the_die_states_the_capacity_npu_smi_reports(self):
        self.assertEqual(get_gpu(SKU).mem_gib, DIE_HBM_GIB)

    def test_the_scaleup_cell_bounds_what_a_collective_reached(self):
        """An HCCL collective's bus bandwidth stays under the link rate.

        The direction that matters: a cell BELOW what the fabric reached
        would make every collective row read as faster than its link, the
        same defect the compute cells have. Measured on 16 dies, which is
        the shape a TP16 deployment runs.
        """
        card = get_gpu(SKU)
        self.assertIsNotNone(card.scaleup_gbps)
        self.assertGreaterEqual(card.scaleup_gbps, REACHED_SCALEUP_GBPS)

    def test_no_fp8_or_fp4_peak_is_claimed(self):
        # The probe ran no FP8 or FP4 matmul because the hardware has no
        # such MMA; a cell here would price a launch nothing runs.
        card = get_gpu(SKU)
        self.assertIsNone(card.fp8_tflops)
        self.assertIsNone(card.fp4_tflops)


if __name__ == "__main__":
    unittest.main()
