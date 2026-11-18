"""Per-parallelism comm volume tests: for each parallelism mode, verify the
END-TO-END comm picture through `pipeline.analyze` —

  1. TOPOLOGY  — which collectives appear (op + world),
  2. SHAPE     — the communicated tensor (`shape_out` = producer tensor),
  3. VOLUME    — `comm_bytes_per_rank` matches the closed-form per-rank cost
                 `collective_per_rank_bytes(op, world, Πshape · dtype) · layers`.

This is the user-facing contract (the numbers the dashboard renders), exercised
on real specs: Llama (dense GQA) for TP / SP / CP, DeepSeek-V3 (MoE) for EP.
Layout topology is locked on the public analyze() path; here we pin
the bytes the pipeline actually reports.

Ring-AR per-rank traffic: all_reduce = 2(N-1)/N·T, every other collective
(all_gather / reduce_scatter / all_to_all) = (N-1)/N·T — except a ROUTED
MoE exchange, which sends a token once per distinct destination its top-k
picks land on and so takes its own fan-out's coefficient (the row's
`comm_route_fanout`, `comm_volume.RouteFanout`).
"""
import math
from fractions import Fraction
import os
import sys
import unittest
from typing import Mapping

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from tests.snapshots import resolve_snapshot, shared_loader
from backend.inference.pipeline import analyze
from backend.inference.parallel_config import ParallelConfig
from backend.inference.comm_cost import collective_per_rank_bytes

# The volume models below ship BF16: collectives carry 2-byte activations.
_ACT_DT = 2.0
_B = 1
_S = 2048


# -- the priced step, once per deployment --------------------------------
#
# Each class here pins one mesh's collectives, and several of its tests read
# different rows off the SAME priced step: seven of this file's thirty-four
# steps were a step another test in the file had already priced, and pricing
# one is a second and a half.
#
# So the module keeps the step per deployment and the file's tests share it.
# The rows are read, never written — every test here filters the step's
# `per_op_timing` into a list of its own and asserts on the numbers in it.
# The model is named in the key (the resolution is how the step is reached,
# not what makes two steps the same), and a `ParallelConfig` is a dataclass
# with generated equality but no hash, so the mesh enters it as its `repr`.

_STEPS: dict = {}


def _step(resolved, mid, pc, phase, seq, rho):
    key = (mid, repr(pc), phase, seq, rho)
    if key not in _STEPS:
        _STEPS[key] = analyze(
            resolved_spec=resolved, parallel=pc, batch_size=_B,
            avg_input_tokens=seq, avg_output_tokens=0, ep_load_factor=rho,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=1e12,
            interconnect_bw_bytes_per_sec=1e11, phase=phase, breakdown=False)
    return _STEPS[key]


@pytest.fixture(scope="module", autouse=True)
def _forget_steps():
    """Bound the memo to this module: a module-level dict would
    otherwise hold the steps it kept for the rest of the session,
    and one file's worth resident at a time is the point."""
    yield
    _STEPS.clear()


def _routed_dt(row) -> float:
    """The element width a DeepSeek-V3 MoE exchange moves, counted by hand:
    the dispatch's exchange (`..__on__experts`) carries the tokens quantised
    before the dispatch — one FP8 byte per element and an F32 scale per 128
    channels of each row (`fp8_token_group128`) — and the exchange back
    the experts' BF16 output."""
    if not str(row["aggregation_id"]).endswith("__on__experts"):
        return _ACT_DT
    inner = int((row.get("comm_payload_shape") or row["shape_out"])[-1])
    return 1 + 4 * math.ceil(inner / 128) / inner


def _dedup_coefficient(world: int, fanout: Mapping[str, int]) -> Fraction:
    """A routed exchange's coefficient, written here from the counts.

    Deliberately NOT `comm_volume.ring_factor(op, world, fanout)`: that is
    the implementation, and a test that calls it checks that the row read
    the right function, not that the function is right. So the expectation
    is spelled out — the share whose destination is not this rank, times
    the distinct destination ranks a token's `k` picks reach out of `D`
    (one minus the hypergeometric chance of missing a rank's `E/D`
    targets), over the copies of the token the payload holds.
    """
    d, k = int(world), int(fanout["picks"])
    targets, placed = int(fanout["targets"]), int(fanout["placed"])
    per_rank = placed // d
    reached = d * (1 - Fraction(math.comb(targets - per_rank, k),
                                math.comb(targets, k)))
    return (Fraction(d - 1, d) * reached
            / Fraction(k * placed, targets))


def _expect(op: str, world: int, shape, dt: float = _ACT_DT,
            fanout=None) -> float:
    """Closed-form per-rank bytes for one collective over `shape`·dt.

    `fanout` is the row's own `comm_route_fanout` where it has one: a routed
    MoE exchange sends a token once per distinct destination RANK, so its
    coefficient is `_dedup_coefficient` and not the flat `(N-1)/N`."""
    t = math.prod(int(d) for d in shape) * dt
    if fanout:
        return float(_dedup_coefficient(world, fanout)) * t
    return collective_per_rank_bytes(op, world, t)


class _VolumeBase(unittest.TestCase):
    MODELS = ()

    @classmethod
    def setUpClass(cls):
        cls.loader = shared_loader()
        cls.specs = {}

    def _resolved(self, mid):
        # Missing snapshot → skip; present but unresolvable → failure
        # (tests.snapshots decides, not this file).
        if mid not in self.specs:
            self.specs[mid] = resolve_snapshot(mid, self.loader)
        return self.specs[mid]

    def _comm_rows(self, mid, pc, phase="prefill", seq=_S, rho=1):
        r = _step(self._resolved(mid), mid, pc, phase, seq, rho)
        return [
            row for row in r.per_op_timing
            if int(row.get("comm_world", 1) or 1) > 1
            and row.get("op") in ("all_reduce", "reduce_scatter",
                                  "all_gather", "all_to_all")
        ]

    @staticmethod
    def _split_vocab(rows):
        """Partition comm rows into (region, vocab). The vocab-parallel
        collectives (`embed_tokens.vocab_all_reduce`, `lm_head.vocab_
        all_gather`) are model-level extras grafted by
        `pipeline._augment_vocab_comm` under any `tp_attention>1` config
        (Megatron VocabParallelEmbedding → all_reduce, ParallelLMHead →
        all_gather; keyed on tp_attention, NOT SP-aware). They coexist
        with the per-region block collectives each test pins."""
        region, vocab = [], []
        for r in rows:
            (vocab if ".vocab_" in str(r.get("aggregation_id", ""))
             else region).append(r)
        return region, vocab

    def _assert_vocab_pair(self, vocab, world):
        """The two vocab collectives, present + volume-correct whenever
        tp_attention>1: embed all_reduce([B,S,H]) + lm_head
        all_gather([B,S,V])."""
        ops = sorted(r["op"] for r in vocab)
        self.assertEqual(ops, ["all_gather", "all_reduce"], vocab)
        for r in vocab:
            self._assert_volume(r, op=r["op"], world=world)

    def _assert_volume(self, row, *, op, world, dt=_ACT_DT):
        """A comm row whose per-rank bytes follow the standard shape·dt cost
        (every collective except the head-free decode LSE all_gather)."""
        self.assertEqual(row["op"], op, row)
        self.assertEqual(int(row["comm_world"]), world, row)
        shape = row.get("comm_payload_shape") or row.get("shape_out")
        self.assertTrue(shape, f"comm row missing shape_out: {row}")
        layers = int(row.get("count_per_step", 1) or 1)
        want = _expect(op, world, shape, dt,
                       row.get("comm_route_fanout")) * layers
        got = float(row["comm_bytes_per_rank"])
        self.assertAlmostEqual(
            got, want, delta=max(1.0, want * 1e-6),
            msg=f"{op} w={world} shape={shape} ×{layers}: got {got} want {want}")
        return shape


class TestTensorParallelVolume(_VolumeBase):
    MODELS = ("meta-llama/Llama-2-7b-hf",)

    def test_tp_all_reduce_is_hidden_sized(self):
        # Row-parallel exits (attention o_proj, ffn down_proj) → all_reduce of
        # the FULL hidden activation [B,S,H], per-rank 2(N-1)/N·BSH·dt.
        rows = self._comm_rows("meta-llama/Llama-2-7b-hf",
                               ParallelConfig(tp_attention=8, tp_dense_ffn=8))
        self.assertTrue(rows)
        region, vocab = self._split_vocab(rows)
        # Every per-region (block) collective is a hidden-sized all_reduce.
        self.assertTrue(region)
        self.assertTrue(all(r["op"] == "all_reduce" for r in region), region)
        for r in region:
            shape = self._assert_volume(r, op="all_reduce", world=8)
            # Communicated tensor is the full hidden [B, S, H].
            self.assertEqual(shape[0], _B, shape)
            self.assertEqual(shape[1], _S, shape)
        # Plus the two vocab-parallel model-level collectives.
        self._assert_vocab_pair(vocab, world=8)


class TestSequenceParallelVolume(_VolumeBase):
    MODELS = ("meta-llama/Llama-2-7b-hf",)

    def test_sp_swaps_each_ar_for_ag_plus_rs(self):
        # Megatron SP: each TP all_reduce becomes all_gather (norm→TP entry) +
        # reduce_scatter (TP exit→norm), each (N-1)/N·T — half an AR each, so
        # SP keeps the same total ring traffic but overlaps better.
        rows = self._comm_rows(
            "meta-llama/Llama-2-7b-hf",
            ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                           sequence_parallel=True))
        region, vocab = self._split_vocab(rows)
        # Each per-region TP all_reduce is swapped for AG (entry) + RS (exit).
        ops = sorted({r["op"] for r in region})
        self.assertEqual(ops, ["all_gather", "reduce_scatter"], region)
        for r in region:
            self._assert_volume(r, op=r["op"], world=8)
        # The vocab collectives are keyed on tp_attention, not SP-aware, so
        # they stay all_reduce (embed) + all_gather (lm_head) under SP too.
        self._assert_vocab_pair(vocab, world=8)


class TestContextParallelVolume(_VolumeBase):
    MODELS = (
        "meta-llama/Llama-2-7b-hf",
        "Qwen/Qwen3-30B-A3B",
    )

    def test_prefill_ulysses_a2a_is_hidden_sized(self):
        rows = self._comm_rows("meta-llama/Llama-2-7b-hf",
                               ParallelConfig(cp=2, cp_scheme="ulysses"),
                               phase="prefill")
        plan_rows = [r for r in rows if r["op"] == "all_to_all"]
        self.assertTrue(plan_rows)
        for r in plan_rows:
            shape = self._assert_volume(r, op="all_to_all", world=2)
            # Ulysses exchanges one local Q/K/V/O buffer. Depending on the
            # edge direction that buffer is sequence-sharded or head-sharded;
            # both layouts contain exactly 1/cp of the global tensor. Pin the
            # physical invariant rather than one legal axis ordering.
            self.assertEqual(len(shape), 4, shape)
            self.assertEqual(math.prod(shape), _B * _S * 4096 // 2, shape)

    def test_prefill_pcp_all_gathers_kv(self):
        rows = self._comm_rows(
            "meta-llama/Llama-2-7b-hf",
            ParallelConfig(cp=2, cp_scheme="pcp"), phase="prefill")
        self.assertTrue(all(r["op"] == "all_gather" for r in rows), rows)
        self.assertTrue(rows)
        for r in rows:
            self._assert_volume(r, op="all_gather", world=2)

    def test_prefill_layersplit_matches_pcp_comm(self):
        pcp = self._comm_rows(
            "meta-llama/Llama-2-7b-hf",
            ParallelConfig(cp=2, cp_scheme="pcp"), phase="prefill")
        split = self._comm_rows(
            "meta-llama/Llama-2-7b-hf",
            ParallelConfig(cp=2, cp_scheme="layersplit"), phase="prefill")
        self.assertEqual(
            [(r["op"], r.get("comm_world"), r.get("shape_out"),
              r.get("comm_bytes_per_rank")) for r in split],
            [(r["op"], r.get("comm_world"), r.get("shape_out"),
              r.get("comm_bytes_per_rank")) for r in pcp],
        )

    def test_decode_dcp_combine_and_lse(self):
        # Decode = query all_gather + LSE all_gather + combine reduce_scatter,
        # and it is `dcp` that produces them: PCP replicates the decode stream
        # over its group, so `cp` derives nothing here (pinned in
        # test_parallelism_comm.py). DCP is a subgroup of the attention TP
        # ranks, hence the tp_attention=2 beside it.
        # The query AG + RS follow the standard shape·dt cost; the LSE AG is
        # head-free fp32 ×dcp, so its bytes are MUCH smaller than its shape_out
        # would imply (that's the whole point — it's a tiny statistic).
        pc = ParallelConfig(
            tp_attention=8,
            tp_dense_ffn=8,
            tp_moe_expert=8,
            dcp=2,
        )
        self.assertEqual(pc.validate(), [], pc)
        all_rows = self._comm_rows("Qwen/Qwen3-30B-A3B", pc,
                                   phase="decode")
        rows, vocab = self._split_vocab(all_rows)
        self._assert_vocab_pair(vocab, world=8)
        rows = [r for r in rows if int(r["comm_world"]) == 2]
        # The TP all_reduces coexist with the DCP set; this test pins the DCP
        # three, so drop the world's other collective.
        rows = [r for r in rows if r["op"] != "all_reduce"]
        ops = sorted(r["op"] for r in rows)
        self.assertEqual(ops.count("reduce_scatter"), 1, rows)
        self.assertEqual(ops.count("all_gather"), 2, rows)  # query + LSE
        self.assertEqual(ops.count("all_to_all"), 0, rows)
        rs = next(r for r in rows if r["op"] == "reduce_scatter")
        self._assert_volume(rs, op="reduce_scatter", world=2)
        # Among the two all_gathers, exactly one is the standard query AG and
        # one is the smaller head-free LSE (fails the standard shape·dt cost).
        ags = [r for r in rows if r["op"] == "all_gather"]
        standard, lse = [], []
        for r in ags:
            layers = int(r.get("count_per_step", 1) or 1)
            want = _expect(
                "all_gather", 2,
                r.get("comm_payload_shape") or r["shape_out"],
            ) * layers
            (standard if math.isclose(float(r["comm_bytes_per_rank"]), want,
                                      rel_tol=1e-6) else lse).append(r)
        self.assertEqual(len(standard), 1, f"query AG: {ags}")
        self.assertEqual(len(lse), 1, f"LSE AG: {ags}")
        # LSE is strictly smaller than the query gather (head-free statistic).
        self.assertLess(float(lse[0]["comm_bytes_per_rank"]),
                        float(standard[0]["comm_bytes_per_rank"]), ags)


class TestContextParallelDenseIsDataParallel(_VolumeBase):
    """CP treats the dense FFN as DATA parallelism, not tensor parallelism:
    each cp rank holds the FULL FFN weight and runs on its S/cp token shard.
    So cp→2cp leaves the FFN weight/rank UNCHANGED (replicated) but HALVES the
    FFN flops/tokens per rank. (tp_dense, by contrast, divides the weight.)
    This pins the semantic I got wrong — cp shards tokens via global
    seq-sharding, NOT the dense weight via the region rank-grouping."""

    MODELS = ("meta-llama/Llama-2-7b-hf",)

    def _ffn_down(self, pc):
        resolved = self.specs.get("meta-llama/Llama-2-7b-hf")
        if resolved is None:
            self.skipTest("no llama spec")
        r = analyze(
            resolved_spec=resolved, parallel=pc, batch_size=_B, avg_input_tokens=_S, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=1e12,
            interconnect_bw_bytes_per_sec=1e11, phase="prefill", breakdown=False)
        for x in r.per_op_timing:
            if "down_proj" in (x.get("aggregation_id") or "") \
                    and (x.get("flops_per_rank") or 0) > 0:
                return x
        self.fail("no dense down_proj row")

    def test_cp_replicates_ffn_weight_but_shards_tokens(self):
        a = self._ffn_down(ParallelConfig())            # cp=1
        b = self._ffn_down(ParallelConfig(cp=2))        # cp=2, tp_dense=1
        # weight per rank UNCHANGED (cp does not divide the FFN weight = DP)…
        self.assertEqual(b["weight_bytes_per_rank"], a["weight_bytes_per_rank"])
        # …but flops per rank HALVE (S → S/cp tokens per rank).
        self.assertAlmostEqual(
            b["flops_per_rank"], a["flops_per_rank"] / 2,
            delta=a["flops_per_rank"] * 1e-6)

    def test_tp_dense_divides_the_weight(self):
        # Contrast: tp_dense=2 DOES halve the FFN weight (tensor parallel).
        a = self._ffn_down(ParallelConfig())
        c = self._ffn_down(ParallelConfig(tp_dense_ffn=2, tp_attention=2,
                                          dp_attention=2))
        self.assertAlmostEqual(
            c["weight_bytes_per_rank"], a["weight_bytes_per_rank"] / 2,
            delta=a["weight_bytes_per_rank"] * 1e-6)


class TestExpertParallelVolume(_VolumeBase):
    MODELS = ("deepseek-ai/DeepSeek-V3",)

    def test_ep_routing_a2a_volume(self):
        # EP dispatch + return all_to_all of the routed hidden, per-rank
        # (N-1)/N·T (imbalance factor 1.0 by default → standard cost).
        # DP-attention is what makes it an exchange rather than a reduction:
        # the stream arrives cut by request, the bank wants it cut by expert
        # owner. EP alone moves nothing — see TestExpertParallelComm.
        rows = self._comm_rows("deepseek-ai/DeepSeek-V3",
                               ParallelConfig(ep_moe=8, dp_attention=8),
                               phase="prefill")
        a2a = [r for r in rows if r["op"] == "all_to_all"]
        self.assertTrue(a2a, rows)
        for r in a2a:
            self._assert_volume(r, op="all_to_all", world=8,
                                dt=_routed_dt(r))

    def test_ep_load_factor_scales_a2a_volume(self):
        # max-violate: a hot-expert factor leaves the collective set alone and
        # the routing a2a moves the busiest rank's rows — the mean's times its
        # load, to the next whole row. The factor is per EXPERT: DSv3's
        # 256 experts over EP=8 put 32 on a rank, and one expert at 1.5x the
        # mean lifts its rank by (1.5 + 31 x 254.5/255) / 32, not 1.5.
        base = [r for r in self._comm_rows(
            "deepseek-ai/DeepSeek-V3",
            ParallelConfig(ep_moe=8, dp_attention=8), "prefill")
            if r["op"] == "all_to_all"]
        hot = [r for r in self._comm_rows(
            "deepseek-ai/DeepSeek-V3",
            ParallelConfig(ep_moe=8, dp_attention=8), "prefill", rho=1.5)
            if r["op"] == "all_to_all"]
        self.assertEqual(len(base), len(hot))
        self.assertTrue(base)
        rank = (Fraction(3, 2) + 31 * (256 - Fraction(3, 2)) / 255) / 32
        for b, h in zip(base, hot, strict=True):
            # Whole rows: the busiest buffer is at least the mean's times the
            # rank's load and less than one row more.
            grown = (Fraction(str(h["comm_tensor_bytes"]))
                     / Fraction(str(b["comm_tensor_bytes"])))
            self.assertGreaterEqual(grown, rank)
            self.assertLess(grown - rank, Fraction(1, 1000))
        b_t = sum(float(r["t_comm_s"]) for r in base)
        h_t = sum(float(r["t_comm_s"]) for r in hot)
        self.assertGreater(h_t, b_t)


class TestExpertXTensorParallelVolume(_VolumeBase):
    MODELS = ("deepseek-ai/DeepSeek-V3",)

    def test_ep_x_moe_tp_both_worlds_distinct_volumes(self):
        # EP a2a (world=ep_moe) AND intra-expert TP all_reduce (world via the
        # tp_moe inner_group) coexist, each its own world + standard cost.
        rows = self._comm_rows(
            "deepseek-ai/DeepSeek-V3",
            ParallelConfig(ep_moe=8, tp_moe_expert=4,
                           tp_attention=4, tp_dense_ffn=4, dp_attention=8),
            phase="prefill")
        a2a = [r for r in rows if r["op"] == "all_to_all"]
        ar = [r for r in rows if r["op"] == "all_reduce"]
        self.assertTrue(a2a, rows)
        self.assertTrue(ar, rows)
        for r in a2a:
            self.assertEqual(int(r["comm_world"]), 8, r)
            self._assert_volume(r, op="all_to_all", world=8,
                                dt=_routed_dt(r))
        for r in ar:
            self._assert_volume(r, op="all_reduce", world=int(r["comm_world"]))


class TestCommVolumeInvariantSweep(_VolumeBase):
    """Volume correctness, swept over the combinatorial matrix instead of spot-
    checked. It decomposes into three already-tested facts:
      * TOPOLOGY (which op, which world) — proven by validate_comm_closure;
      * per-rank FORMULA `collective_per_rank_bytes` — unit-tested in
        test_comm_cost;
      * producer SHAPE — engine shape tests.
    The only gluing claim is 'each comm node carries the right shape×world', so
    we assert the per-row identity
        comm_bytes_per_rank == collective_per_rank_bytes(op, world, Πshape·dt)
                               × layers
    for EVERY comm row across (model × TP × SP × EP × CP × scheme × phase). The
    one documented exception is the decode DCP LSE all_gather (head-free fp32),
    which by construction does NOT follow the shape·dt cost — every non-matching
    row must be exactly that."""

    MODELS = (
        "meta-llama/Llama-2-7b-hf",
        "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen3-30B-A3B",
    )

    # (model, ParallelConfig) — valid configs spanning each scheme + combos.
    def _matrix(self):
        L, D = "meta-llama/Llama-2-7b-hf", "deepseek-ai/DeepSeek-V3"
        return [
            (L, ParallelConfig(tp_attention=8, tp_dense_ffn=8)),
            (L, ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                               sequence_parallel=True)),
            (L, ParallelConfig(cp=2)),
            (L, ParallelConfig(cp=2, cp_scheme="pcp")),
            (L, ParallelConfig(cp=2, cp_scheme="layersplit")),
            (L, ParallelConfig(tp_attention=4, cp=2, tp_dense_ffn=4)),
            # dcp alone (re-cuts the TP ranks) and dcp over a PCP column —
            # the two fabrics the subgroup can land on.
            ("Qwen/Qwen3-30B-A3B", ParallelConfig(
                tp_attention=8, tp_dense_ffn=8,
                tp_moe_expert=8, dcp=2)),
            ("Qwen/Qwen3-30B-A3B", ParallelConfig(
                tp_attention=8, tp_dense_ffn=8,
                tp_moe_expert=8, ep_moe=2, cp=2, dcp=2)),
            (D, ParallelConfig(ep_moe=8)),
            (D, ParallelConfig(ep_moe=8, tp_moe_expert=4)),
            (D, ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)),
            (D, ParallelConfig(ep_moe=4, cp=2, tp_attention=2, tp_dense_ffn=2,
                               tp_moe_expert=2, dp_attention=2)),
        ]

    def test_every_comm_row_matches_closed_form(self):
        for mid, pc in self._matrix():
            self.assertEqual(pc.validate(), [], f"{mid} {pc}")
            # A decode step's one token per request is not split by sequence
            # parallelism (the walk refuses it: one token over eight ranks).
            for phase in (("prefill",) if pc.sequence_parallel
                          else ("prefill", "decode")):
                rows = self._comm_rows(mid, pc, phase=phase)
                matched = 0
                for r in rows:
                    layers = int(r.get("count_per_step", 1) or 1)
                    want = _expect(
                        r["op"], int(r["comm_world"]),
                        r.get("comm_payload_shape") or r["shape_out"],
                        dt=float(r.get("comm_payload_dtype_bytes") or _ACT_DT),
                        fanout=r.get("comm_route_fanout"),
                    ) * layers
                    got = float(r["comm_bytes_per_rank"])
                    if math.isclose(got, want, rel_tol=1e-6):
                        matched += 1
                        continue
                    self.fail(
                        f"{mid} {phase}: comm row violates closed form "
                        f"(op={r['op']} world={r['comm_world']} "
                        f"payload={r.get('comm_payload_shape')} "
                        f"got={got} want={want})")
                # Sanity: configs with comm actually exercised the standard
                # path. A decode step under PCP alone has no comm at all — the
                # group runs replicated — so there is nothing to have matched.
                if rows:
                    self.assertGreater(matched, 0, f"{mid} {phase}: no rows")


if __name__ == "__main__":
    unittest.main()
