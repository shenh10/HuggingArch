"""A cache that does not grow with length still budgets a batch.

A request's cache is `growth × L + floor` past every window. Mistral-7B-v0.1
windows every layer at 4096, so its growth is zero and the whole cache is
floor: 32 layers × (K + V) × 8 KV heads × 128 × 2 bytes × 4096 positions =
512 MiB a request, however long the request runs. Capacity used to require
growth and refused the model outright ("Could not compute per-token KV
bytes"), which also dropped it from the gallery's performance charts.

What such a cache bounds is the batch — usable bytes over a request's bytes —
not the tokens a request may hold, which no length exhausts (`max_tokens` is
None, not 0). The whole-to-local shard ratio is read off the same law's
leading term: the floor's, when there is no growth.
"""
from __future__ import annotations

import unittest
from functools import lru_cache
from types import SimpleNamespace

from backend.inference.capacity import (
    _cache_budget_terms, estimate_capacity, max_batch_rung)
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import (
    resolve_snapshot, shared_loader, snapshot_config,
)
from backend.inference import capacity as _capacity

MISTRAL = "mistralai__Mistral-7B-v0.1"
# 32 layers × 2 (K, V) × 8 KV heads × 128 head dim × 2 bytes × 4096 window.
_FULL_WINDOW_BYTES = 32 * 2 * 8 * 128 * 2 * 4096


@lru_cache(maxsize=None)
def _estimate(tp: int, input_tokens: int) -> dict:
    return estimate_capacity(
        config=snapshot_config(MISTRAL), gpu_sku="H100-SXM",
        resolved=resolve_snapshot(MISTRAL, shared_loader()),
        selection_mode="theoretical_only",
        avg_input_tokens=input_tokens, avg_output_tokens=1024,
        parallel=ParallelConfig(tp_attention=tp, tp_dense_ffn=tp))


class _View(SimpleNamespace):
    def growth_and_floor(self):
        return self.growth, self.floor

    def static_scale_bytes(self):
        return 0.0


class TestBudgetTermsWithoutGrowth(unittest.TestCase):

    def test_an_all_floor_cache_is_budgeted_and_sharded_by_its_floor(self):
        local = _View(growth=0.0, floor=256.0)
        whole = _View(growth=0.0, floor=1024.0)
        terms = _cache_budget_terms(
            local, whole, ParallelConfig(tp_attention=4, tp_dense_ffn=4))
        self.assertEqual(terms.per_token, 0.0)
        self.assertEqual(terms.constant_state, 256.0)
        self.assertEqual(terms.shard_divisor, 4.0)
        self.assertEqual(terms.replication_factor, 1.0)

    def test_growth_still_decides_the_ratio_when_there_is_growth(self):
        local = _View(growth=10.0, floor=100.0)
        whole = _View(growth=20.0, floor=400.0)
        terms = _cache_budget_terms(
            local, whole, ParallelConfig(tp_attention=2, tp_dense_ffn=2))
        self.assertEqual(terms.shard_divisor, 2.0)

    def test_no_cache_at_all_is_refused(self):
        empty = _View(growth=0.0, floor=0.0)
        with self.assertRaises(ValueError):
            _cache_budget_terms(empty, empty, ParallelConfig())


class TestAllWindowModelEstimates(unittest.TestCase):

    def test_batch_is_bounded_and_tokens_are_not(self):
        r = _estimate(1, 1024)
        self.assertTrue(r["memory"]["fits"])
        self.assertIsNone(r["max_tokens_per_gpu"])
        self.assertIsNone(r["max_tokens_total"])
        # 2048 tokens, inside the window: 2048 of the 4096 positions held.
        per_request = r["kv_cache"]["per_request_kv_bytes"]
        self.assertEqual(per_request, _FULL_WINDOW_BYTES // 2)
        budget = r["memory"]["kv_budget_bytes"]
        # One request's full window is the constant floor the budget holds
        # back before dividing.
        self.assertEqual(r["max_batch_per_gpu"],
                         int((budget - _FULL_WINDOW_BYTES) // per_request))
        self.assertGreater(r["max_batch_per_gpu"], 100)
        # The phase batch is the rung of the exact ceiling
        # (`max_batch_rung`): the max point the panels draw, the exact
        # ceiling a capacity fact (docs/sweep_card_class.md §2.C).
        self.assertEqual(r["phases"]["decode"]["batch"],
                         1 << (r["max_batch_per_gpu"].bit_length() - 1))

    def test_prefills_public_point_is_its_rung_and_its_cap_stays_exact(self):
        r = estimate_capacity(
            config=snapshot_config(MISTRAL), gpu_sku="H100-SXM",
            resolved=resolve_snapshot(MISTRAL, shared_loader()),
            selection_mode="theoretical_only",
            avg_input_tokens=1024, avg_output_tokens=1024,
            prefill_batch_size=-1)
        cap = int(r["max_prefill_batch_per_gpu"])
        self.assertGreater(cap, 1)
        self.assertEqual(r["phases"]["prefill"]["batch"], max_batch_rung(cap))
        self.assertEqual(r["max_prefill_batch_per_gpu"], cap)

    def test_same_class_skus_share_one_overlay_walk(self):
        """The class cache walks the class once; the sibling's SKU, point and
        module layers are substituted, not re-walked."""
        _capacity._CLASS_OVERLAYS.clear()
        original = _capacity.overlay_walk
        calls = []

        def counted(**kwargs):
            calls.append((kwargs["phase"], kwargs["gpu_sku"],
                          kwargs["max_batch"]))
            return original(**kwargs)

        _capacity.overlay_walk = counted
        try:
            estimates = [estimate_capacity(
                config=snapshot_config(MISTRAL), gpu_sku="H100-SXM",
                resolved=resolve_snapshot(MISTRAL, shared_loader()),
                selection_mode="theoretical_only",
                avg_input_tokens=1024, avg_output_tokens=1024),
                estimate_capacity(
                    config=snapshot_config(MISTRAL), gpu_sku="H200",
                    resolved=resolve_snapshot(MISTRAL, shared_loader()),
                    selection_mode="theoretical_only",
                    avg_input_tokens=1024, avg_output_tokens=1024)]
        finally:
            _capacity.overlay_walk = original
        self.assertEqual(calls, [
            ("prefill", "H100-SXM", 1),
            ("decode", "H100-SXM",
             estimates[0]["phases"]["decode"]["batch"])])
        self.assertTrue(estimates)

    def test_the_class_overlay_registry_is_bounded(self):
        """A warm touches every model; finished models' overlays are garbage."""
        _capacity._CLASS_OVERLAYS.clear()
        for index in range(_capacity._CLASS_OVERLAYS_MAX + 10):
            _capacity._remember_class_overlay((index,), index)
        self.assertEqual(len(_capacity._CLASS_OVERLAYS),
                         _capacity._CLASS_OVERLAYS_MAX)
        self.assertEqual(list(_capacity._CLASS_OVERLAYS),
                         [(index,) for index in range(
                             10, _capacity._CLASS_OVERLAYS_MAX + 10)])

    def test_past_the_window_a_request_costs_the_window_and_no_more(self):
        long = _estimate(1, 64 * 1024)
        longer = _estimate(1, 128 * 1024)
        for r in (long, longer):
            self.assertEqual(r["kv_cache"]["per_request_kv_bytes"],
                             _FULL_WINDOW_BYTES)
        self.assertEqual(long["max_batch_per_gpu"],
                         longer["max_batch_per_gpu"])
        self.assertGreater(long["max_batch_per_gpu"], 0)

    def test_the_law_the_budget_uses_is_published(self):
        """`kv_cache` carries the two terms `max_tokens` divides by and
        subtracts, so a reader of the slope does not rebuild it from the
        per-request formula card — whose first term, for this model, is a
        whole window's bytes, not bytes per token."""
        for tp in (1, 2):
            r = _estimate(tp, 64 * 1024)
            kv = r["kv_cache"]
            self.assertEqual(kv["per_token_bytes"], 0.0)
            self.assertEqual(kv["constant_bytes"], _FULL_WINDOW_BYTES / tp)
            # Past the window, growth × L + floor is the request.
            self.assertEqual(kv["per_token_bytes"] * 65 * 1024
                             + kv["constant_bytes"],
                             kv["per_request_kv_bytes"])

    def test_tensor_parallel_divides_the_floor(self):
        one, two = _estimate(1, 1024), _estimate(2, 1024)
        self.assertEqual(one["kv_cache"]["shard_divisor"], 1.0)
        self.assertEqual(two["kv_cache"]["shard_divisor"], 2.0)
        self.assertEqual(two["kv_replication_factor"], 1.0)
        self.assertEqual(two["kv_cache"]["per_request_kv_bytes"] * 2,
                         one["kv_cache"]["per_request_kv_bytes"])
        self.assertGreater(two["max_batch_per_gpu"],
                           one["max_batch_per_gpu"])


GEMMA = "google__gemma-2-9b"


class TestPublishedLawOfAGrowingHybridCache(unittest.TestCase):
    """Gemma-2 alternates 4096-token sliding windows with global layers: the
    windows are the per-request constant, the global layers the growth. The
    two published terms are the ones `max_tokens` is computed from."""

    def test_growth_and_constant_are_the_budget_s_terms(self):
        r = estimate_capacity(
            config=snapshot_config(GEMMA), gpu_sku="H100-SXM",
            resolved=resolve_snapshot(GEMMA, shared_loader()),
            selection_mode="theoretical_only",
            avg_input_tokens=8192, avg_output_tokens=1024)
        kv = r["kv_cache"]
        growth, constant = kv["per_token_bytes"], kv["constant_bytes"]
        self.assertGreater(growth, 0)
        self.assertGreater(constant, 0)
        self.assertEqual(growth * (8192 + 1024) + constant,
                         kv["per_request_kv_bytes"])
        budget = r["memory"]["kv_budget_bytes"]
        self.assertEqual(
            r["max_tokens_per_gpu"],
            int((budget - kv["static_scale_bytes"] - constant) // growth))


if __name__ == "__main__":
    unittest.main()
