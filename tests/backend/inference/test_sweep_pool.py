"""The sweeps across the process pool (`backend/measure/pool.py`).

The pool spreads a sweep's cells across worker processes. Correctness rests
on one invariant: a parallel result equals the serial result byte-for-byte
(both equal N single-seq calls). These tests pin that, the result ordering,
and the serial fallback when the pool is unavailable or breaks. The pool's
own behaviour is `tests/backend/measure/test_pool.py`.
"""
from __future__ import annotations

import sys
import unittest
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

from tests.snapshots import resolve_snapshot, snapshot_config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.inference import capacity as _capacity
from backend.measure import pool as _sweep_pool
from backend.inference.capacity import (
    estimate_capacity,
    estimate_capacity_sweep,
)
from backend.inference.parallel_sweep import estimate_parallel_sweep
from backend.inference.parallel_config import ParallelConfig

_MODEL = "meta-llama/Llama-3.2-3B"
_GPU = "H100-SXM"
_SEQS = [1024, 4096, 8192]
_PARITY_KEYS = (
    "max_batch_per_gpu", "max_tokens_per_gpu", "max_prefill_batch_per_gpu",
    "weight_bytes_per_rank", "weight_bytes_total",
)


def _cfg_or_skip():
    # Missing -> skip, present-but-unresolvable -> fail: the helper's contract.
    resolve_snapshot(_MODEL)
    return snapshot_config(_MODEL)


def _base_kwargs(cfg):
    return dict(
        config=cfg, gpu_sku=_GPU, avg_output_tokens=128,
        parallel=ParallelConfig(),
        model_id=_MODEL, selection_mode="theoretical_only",
        )


class TestSweepPoolParallelParity(unittest.TestCase):
    """Parallel results must equal serial single-seq calls, in order.

    Runs the real spawn-based pool (bypassing the pytest gate via the
    explicit `run_parallel` entry). Skipped on boxes too small to build a
    pool — the gate would fall back to serial there anyway."""

    @classmethod
    def setUpClass(cls):
        if _sweep_pool._max_workers() < 2:
            raise unittest.SkipTest("too few cores for a worker pool")
        cls.cfg = _cfg_or_skip()

    @classmethod
    def tearDownClass(cls):
        _sweep_pool.shutdown()

    def test_parallel_matches_serial_in_order(self):
        base = _base_kwargs(self.cfg)
        cells = _capacity.CapacitySweepPlan.of(seq_lens=_SEQS, **base).cells()
        parallel = [seq[0]["sol"] for cell in
                    _sweep_pool.run_parallel(_capacity.walk_sweep_cell, cells)
                    for seq in cell]
        self.assertEqual(len(parallel), len(_SEQS))
        for s, par in zip(_SEQS, parallel, strict=True):
            single = estimate_capacity(avg_input_tokens=s, **base)
            for k in _PARITY_KEYS:
                self.assertEqual(
                    par.get(k), single.get(k),
                    msg=f"seq={s} key={k}: parallel={par.get(k)} single={single.get(k)}",
                )
            # Phase t_sol is the calibrated number that drives throughput —
            # must survive the process boundary identically.
            for phase in ("prefill", "decode"):
                pp = (par.get("phases", {}) or {}).get(phase, {}) or {}
                sp = (single.get("phases", {}) or {}).get(phase, {}) or {}
                self.assertAlmostEqual(
                    float(pp.get("t_sol_s") or 0), float(sp.get("t_sol_s") or 0),
                    delta=1e-12, msg=f"seq={s} {phase}.t_sol_s diverged across pool",
                )

class TestBrokenPoolRecovery(unittest.TestCase):
    """A crashed worker breaks its executor for good. The pool must be
    replaced on the next call, not reused forever."""

    def test_parallel_sweep_serves_serial_when_pool_breaks(self):
        cfg = _cfg_or_skip()
        env = dict(
            config=cfg, gpu_sku="B300", gpu_memory_fraction=0.9,
            num_gpus=8, avg_output_tokens=256, parallel=None,
            selection_mode="theoretical_only", model_id=_MODEL,
        )
        kw = dict(mode="tp_tp", contexts=[1024], gpu_skus=["B300"], base_env=env)
        serial = estimate_parallel_sweep(**kw)

        def _broken(*_a, **_k):
            raise BrokenProcessPool("a worker was killed")

        with mock.patch.object(_sweep_pool, "pool_enabled", lambda: True), \
                mock.patch.object(_sweep_pool, "run_parallel", _broken):
            recovered = estimate_parallel_sweep(**kw)
        self.assertTrue(serial["series"])
        self.assertEqual(recovered, serial)


class TestPoolCellEqualsSerialCell(unittest.TestCase):
    """The pool worker runs one cell of the serial sweep: the whole payload
    must match, not a handful of headline keys."""

    def test_pool_worker_payload_equals_serial_sweep(self):
        model = "Qwen/Qwen2.5-7B"
        resolve_snapshot(model)
        cfg = snapshot_config(model)
        base = {**_base_kwargs(cfg), "model_id": model}
        seqs = [1024, 8192]
        serial = estimate_capacity_sweep(seq_lens=seqs, **base)
        # The sweep's cell as a worker runs it, on a preparation of its own.
        cells = _capacity.CapacitySweepPlan.of(seq_lens=seqs, **base).cells()
        pooled = [seq[0]["sol"] for c in cells
                  for seq in _capacity.walk_sweep_cell(c)]
        self.assertEqual(pooled, serial)


if __name__ == "__main__":
    unittest.main()
