"""Model-level capacity golden baseline.

The three production models (MiMo-V2.5, Kimi-K2.5, Step-3.7-Flash) plus
DeepSeek-V3 and Llama-3.2-3B were hand-calibrated against the serving
team's reference doc over a long effort — but those numbers only ever
lived in the doc, never in the suite. This golden pins the calibrated
backend output so any future change that silently shifts the numbers
(weight footprint, capacity, phase t_sol, per-op timing) fails loudly.

What's pinned per (model, seq):
  • weight_bytes_per_rank          — sharded weight footprint / GPU
  • max_batch_per_gpu / max_tokens  — KV-bound capacity headroom
  • phases.{prefill,decode}.t_sol_s — the per-phase roofline folded from
    the graph's leaves (each contributing its corpus time where the
    calibration overlay has a row for it) — i.e. the number that drives
    the throughput curves on the compare page
  • Σ per_op_timing[*].t_sol_s (prefill + decode) — a coarse per-op
    drift catcher: any change to per-op flops/bytes/timing moves it

Determinism: `selection_mode="measured"` exercises the calibration
path (Llama-3.2-3B/H100 hits the committed flash_attention seed; the
others fall back to roofline — no measured seeds shipped). All inputs
(spec snapshots, gpu_specs.csv, the kernel-timing CSV) are committed,
so the output is reproducible.

Regenerate after an INTENTIONAL numeric change:
    REGEN_GOLDEN=1 uv run pytest tests/test_capacity_golden.py
then eyeball the git diff on tests/golden/capacity_baseline.json before
committing — that diff IS the record of what moved and by how much.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "capacity_baseline.json"

# Every case is as shipped, stated (`precision`): the instrument carries no
# precision default of its own. The old per-case weight widths were each the
# shipped precision already (2.0 on BF16 models was a no-op; 2.0 / 0.5 on the
# mixed-precision ones was refused and served as saved).
AS_SHIPPED = {"preset": "as_shipped"}

# (label, snapshot, gpu_sku, precision, parallel) — the
# calibration set: 8× B200 for the production four, single H100 for
# Llama (the model with a shipped measured kernel seed). Parallel knobs
# reproduce the doc-calibration deployments (Kimi FP4 EP8×CP8; MiMo /
# Step / DSv3 TP-sharded on 8 ranks).
_CASES = [
    # Default capacity excludes the complete auxiliary stack, including its
    # model-level vision/audio projections rather than only repeated blocks.
    # That removes 366,744,064 resident bytes per rank here. Capacity rises;
    # decode timing rises slightly too because this baseline evaluates the
    # newly larger derived batch, while the prefill timing surface is unchanged.
    ("MiMo-V2.5", "XiaomiMiMo__MiMo-V2.5.json", "B200", AS_SHIPPED,
     dict(tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4, ep_moe=2,
          cp=1, pp=1, dp_attention=2, dp_replicas=1)),
    # `dp_attention=8`, not `cp=8`. Both spread the cluster's KV over the 8
    # ranks and both used to reach the same capacity here, but only one of
    # them is a deployment: with no attention TP, large-EP MLA serving gives
    # each rank a DISJOINT set of requests (`--enable-dp-attention`), so a
    # rank holds only its own. `cp=8` says something else — that 8 ranks
    # cooperate on ONE request's context — and prefill context parallelism
    # all_gathers the KV before the cache write, so every rank ends up
    # holding the whole thing. It divides no cache and never did; the old
    # reading of it as a cache split is what this case was standing on.
    #
    # The doc's concurrency is therefore `max_batch_total`, not the per-GPU
    # figure: under DP attention the 8 ranks each floor their own budget, and
    # the cluster serves the sum.
    #
    # `act_peak_per_rank` and `max_prefill_batch` moved here and on both
    # DeepSeek-V3 cases — the three MLA models, both seqs, nothing else —
    # when the MLA template went head-major (363d487). The old graph sliced
    # nope/rope out of the FLAT `H*D` projection (all heads' nope ahead of
    # all heads' rope, an index map the head-major checkpoint does not
    # contain) and then paid late `head_split` copies (`k_nope_split`,
    # `v_split_heads`) to repair it; splitting heads at `q_b_proj` /
    # `kv_b_proj` retires those copies, so fewer tensors coexist at the
    # liveness peak. Element totals are unchanged — weights, flops and every
    # timing cell held to the 1e-6 tolerance — only the peak's composition:
    # at 4096 this cell's peak dropped 1,015,545,856 → 898,105,344 (the
    # retired copies' 117,440,512 bytes) and the prefill budget divides into
    # 108 streams instead of 95.
    ("Kimi-K2.5", "moonshotai__Kimi-K2.5.json", "B200", AS_SHIPPED,
     dict(tp_attention=1, tp_dense_ffn=1, tp_moe_expert=1, ep_moe=8,
          cp=1, pp=1, dp_attention=8, dp_replicas=1)),
    # The four timing cells here moved (+13%/+16% prefill, +6.7%/+1.6%
    # decode) when the collective's process group stopped being guessed from
    # the producing component's ROLE and started coming from the WORLD that
    # divides the axis it reduces. This model's MoE block adds its shared
    # expert to the bank's output OUTSIDE the bank, on a node with no role at
    # all, and the guess bottomed out at a group of ONE — a 42-layer
    # 58.7 MB/layer all_reduce priced at zero bytes, since ring cost carries
    # a 2(N-1)/N factor. The graph did not change: the same 12 collectives
    # are injected before and after, and the two the block exit now prices at
    # tp_moe_expert=8 cost per layer exactly what the dense blocks' exit
    # all_reduce costs. Nothing else in the golden set moved — every other
    # model's exit already resolved to a real world.
    ("Step-3.7-Flash", "stepfun-ai__Step-3.7-Flash.json", "B200", AS_SHIPPED,
     dict(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8, ep_moe=1,
          cp=1, pp=1, dp_attention=1, dp_replicas=1)),
    ("DeepSeek-V3", "deepseek-ai__DeepSeek-V3.json", "B200", AS_SHIPPED,
     dict(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8, ep_moe=1,
          cp=1, pp=1, dp_attention=1, dp_replicas=1)),
    ("Llama-3.2-3B", "meta-llama__Llama-3.2-3B.json", "H100-SXM", AS_SHIPPED,
     dict(tp_attention=1, tp_dense_ffn=1, tp_moe_expert=1, ep_moe=1,
          cp=1, pp=1, dp_attention=1, dp_replicas=1)),
    # gpt-oss-20b under pure EP=4 (E=32): the PACKED/fused expert path,
    # whose checkpoint already stores the stacked expert axis
    # (`expert_weight: leading_axis`) where an unrolled bank stores one
    # tensor per expert. Pinned because the two encodings of the same bank
    # must price identically, and this is the one in the corpus that is
    # packed under EP.
    #
    # Both timing cells here moved by ~2e-4 relative when the packed BIASES
    # started naming the expert axis (`n_expert_copies * 2 * d_ffn_expert`
    # rather than `n_experts * ...` — the same number, under the name expert
    # parallelism divides). `weight_per_rank` did not move, and that is the
    # point: storage already divided those biases by `ep` through the
    # routed-expert class while the per-rank SHAPE kept all 32 on every rank.
    # One tensor, two answers; the declaration now gives one.
    #
    # Both timing cells moved again (+5.5%/+3.3% prefill, +0.6%/+0.08% decode)
    # when expert parallelism stopped being spelled on the token axis. This is
    # the deployment where that spelling had nothing to stand on: `ep_moe=4`
    # with every other degree at 1, so no rank holds a token any other rank
    # lacks. Written as a cut of `s_ctx`, the bank's entry met a residual
    # stream cut the same way and derived a routing all_to_all — 212 MB per
    # block variant, exchanging tokens that were never apart, and no return
    # leg at all. Written as a cut of `n_expert_copies` it derives what pure EP
    # actually performs: each rank runs the experts it owns over ALL the
    # tokens, so the bank exits a partial sum and one all_reduce per block
    # variant closes it. Same one collective per MoE layer; the factor is
    # ring's 2(N-1)/N against all_to_all's (N-1)/N, which is the real price of
    # summing rather than exchanging.
    ("gpt-oss-20b", "openai__gpt-oss-20b.json", "H100-SXM", AS_SHIPPED,
     dict(tp_attention=1, tp_dense_ffn=1, tp_moe_expert=1, ep_moe=4,
          cp=1, pp=1, dp_attention=1, dp_replicas=1)),
    # DSv3 under SIMULTANEOUS dp×cp×ep (the combined-parallelism case the
    # dp-only MiMo and cp-only Kimi cases don't jointly cover). ep = dp·cp = 4:
    # the EP all-to-all pools exactly the dp×cp attention-token ranks onto the
    # experts. Pins the invariants that hold only when all three are on:
    #   * M = ⌈B·S·K·dp/E⌉ — dp GATHERS (×2), cp CANCELS (ulysses returns the
    #     residual S/cp-sharded, which the EP a2a dispatches directly), so M is
    #     independent of cp.
    #   * comm layering: the ulysses cp a2a stays world=cp=2 in attention and
    #     dovetails with the MoE EP dispatch/return a2a at world=dp·cp=ep=4 —
    #     two distinct collectives, not fused.
    #   * reduce-after-combine still merges to one [B·S,H] all_reduce (tp_moe=2).
    #
    # The MoE seam is the all_to_all pair — dispatch and combine over the
    # `ep_moe` world of 4, each routed row moved once. The token cut here is
    # two worlds, `cp` {0,2} and `dp_attention` {0,4}, which together tile
    # the expert world `ep_moe` {0,2,4,6} without either equalling it; the
    # dispatch joins the token owner cells with the route groups by
    # membership (`routed_rows_layout`), the join is every rank, and both
    # cuts are absorbed into the exchange. Per token the pair moves exactly what the
    # dp_ep and cp_ep deployments at the same `ep_moe` move
    # (`test_moe_ordered_collectives::
    # test_two_world_token_cut_exchanges_like_one_world`).
    ("DeepSeek-V3-dpcpep", "deepseek-ai__DeepSeek-V3.json", "B200", AS_SHIPPED,
     dict(tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2, ep_moe=4,
          cp=2, pp=1, dp_attention=2, dp_replicas=1)),
]
_SEQS = [4096, 32768]

# Tolerances: integer/byte fields are exact; float seconds carry
# floating-point arithmetic so use a tight relative tolerance.
# `max_batch` is per-GPU and `max_batch_total` is what the cluster serves.
# Both, because DP attention moves them apart: it multiplies the second and
# leaves the first alone, so a case pinned on one of them alone can have its
# deployment re-labelled between replicating and partitioning the request set
# without a single number moving.
_EXACT_KEYS = ("max_batch", "max_batch_total", "max_tokens",
               "max_prefill_batch", "kv_shard_divisor",
               "kv_replication_factor")
_REL_KEYS = ("weight_per_rank", "prefill_t_sol", "decode_t_sol",
             "perop_pre_tmin_sum", "perop_dec_tmin_sum",
             "act_peak_per_rank")
_REL_TOL = 1e-6


def _snapshot_config(filename: str):
    p = SNAPSHOT_DIR / filename
    if not p.exists():
        return None
    return json.loads(p.read_text())["config"]


def _resolved(snapshot: str, cfg):
    """Resolve once so capacity never re-matches (single entry)."""
    from tests.snapshots import shared_loader
    return shared_loader().match(
        cfg, model_id=Path(snapshot).stem.replace("__", "/", 1))


def _measure(snapshot: str, gpu: str, precision: dict, parallel: dict,
             seq: int) -> dict:
    cfg = _snapshot_config(snapshot)
    if cfg is None:
        return None
    r = estimate_capacity(
        precision=precision,
        config=cfg, resolved=_resolved(snapshot, cfg), gpu_sku=gpu, num_gpus=8,
        avg_input_tokens=seq, avg_output_tokens=256,
        parallel=ParallelConfig(**parallel),
        selection_mode="measured",
    )
    wb = r["weight_breakdown"]
    ph = r.get("phases", {})
    pre, dec = ph.get("prefill", {}), ph.get("decode", {})

    def _tmin_sum(rows):
        return sum(float(x.get("t_sol_s") or 0) for x in (rows or []))

    ap = r.get("activation_peak") or {}

    return {
        "calibration_error": r.get("calibration_error"),
        "weight_per_rank": float(wb["total_bytes_per_rank"]),
        "max_batch": r.get("max_batch_per_gpu"),
        "max_batch_total": r.get("max_batch_total"),
        "max_tokens": r.get("max_tokens_per_gpu"),
        # The prefill side of the same memory. `max_batch` is KV-bound and
        # says nothing about activations, so retiring the uniform `peak / tp`
        # divisor moved every TP case's activation peak by 2.5-4.9x and the
        # golden set noticed nothing. Both are pinned now: the peak itself,
        # and the batch the free budget divides into.
        #
        # These cells moved again when activation liveness stopped replaying
        # the shard-on-read evaluator and consumed exact layout physical
        # layers. The old replay treated references to enclosing component
        # outputs as missing leaf producers, silently allocating another
        # model-width external residual for each one; it also priced every
        # genuine external input as `[B,S,d_model]`. The current projection now
        # resolves component primary/secondary ports to their real leaf
        # tensors and prices each declared external from its own physical
        # input shape. Weight, KV, FLOPs and timing cells are unchanged; only
        # the peak live set and the prefill budget derived from it move.
        "act_peak_per_rank": float(ap.get("peak_activation_bytes_per_rank")
                                   or 0.0),
        "max_prefill_batch": r.get("max_prefill_batch_per_gpu"),
        # How the KV budget was ARRIVED at, beside the budget itself. These
        # two are the ratio of the whole model's per-token bytes to this
        # rank's, so a change that moves the cache geometry without moving
        # `max_tokens` — a degree that stops dividing partway, an axis that
        # starts — shows up here rather than in nothing.
        "kv_shard_divisor": (r.get("kv_cache") or {}).get("shard_divisor"),
        "kv_replication_factor": r.get("kv_replication_factor"),
        "prefill_t_sol": float(pre.get("t_sol_s") or 0.0),
        "decode_t_sol": float(dec.get("t_sol_s") or 0.0),
        "perop_pre_tmin_sum": _tmin_sum(r.get("per_op_timing")),
        "perop_dec_tmin_sum": _tmin_sum(r.get("per_op_timing_decode")),
    }


def _measure_all() -> dict:
    out: dict = {}
    for label, snap, gpu, precision, parallel in _CASES:
        for seq in _SEQS:
            m = _measure(snap, gpu, precision, parallel, seq)
            if m is not None:
                out[f"{label}@{seq}"] = m
    return out


def _maybe_regen() -> bool:
    if os.environ.get("REGEN_GOLDEN") != "1":
        return False
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(_measure_all(), indent=2, sort_keys=True) + "\n")
    return True


class TestCapacityGolden(unittest.TestCase):
    """Each (model, seq) cell must reproduce the committed baseline.

    A failure here is the alarm for a numeric drift — investigate WHY
    the number moved before regenerating. Regenerating without
    understanding the diff defeats the purpose."""

    @classmethod
    def setUpClass(cls):
        if _maybe_regen():
            raise unittest.SkipTest("REGEN_GOLDEN=1 — baseline rewritten, not asserted")
        if not GOLDEN_PATH.exists():
            raise unittest.SkipTest(
                f"golden missing — run REGEN_GOLDEN=1 pytest {Path(__file__).name}")
        cls.golden = json.loads(GOLDEN_PATH.read_text())

    def test_all_cells_match_baseline(self):
        for label, snap, gpu, precision, parallel in _CASES:
            if _snapshot_config(snap) is None:
                continue
            for seq in _SEQS:
                key = f"{label}@{seq}"
                self.assertIn(key, self.golden, f"{key} absent from golden")
                want = self.golden[key]
                got = _measure(snap, gpu, precision, parallel, seq)
                self.assertIsNone(
                    got.get("calibration_error"),
                    f"{key}: per-op overlay degraded — the numbers below "
                    f"would be the closed-form fallback, not the pipeline",
                )
                for k in _EXACT_KEYS:
                    self.assertEqual(
                        got[k], want[k],
                        msg=f"{key}.{k}: got {got[k]} want {want[k]} (exact)",
                    )
                for k in _REL_KEYS:
                    self.assertAlmostEqual(
                        got[k], want[k], delta=max(abs(want[k]) * _REL_TOL, 1e-9),
                        msg=f"{key}.{k}: got {got[k]} want {want[k]} "
                            f"(Δ {got[k]-want[k]:+.3e}, drift > {_REL_TOL:g} rel)",
                    )


if __name__ == "__main__":
    unittest.main()
