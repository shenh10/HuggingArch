"""Pipeline driver — identity equivalence + fusion-saving end-to-end.

Locks identity equivalence and the per-op roofline data shape at
the integration level: the full analyze() flow on real reference
models produces the expected pre/post trees, byte sums, and per-op
timing rows.
"""
from __future__ import annotations

import json
import sys
import unittest

from tests.snapshots import resolve_snapshot
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.ops import is_identity_source
from backend.inference.framework_bindings import (
    AuditedAt,
    FrameworkBinding,
    FusionImplementation,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze




def _resolve(model_name):
    return resolve_snapshot(model_name)


def _tree_signature(ir_tree):
    def node_row(node):
        return {
            "name": node.name,
            "path": node.display_path,
            "op": node.op,
            "template": node.template,
            "role": node.role,
            "inputs": list(node.inputs or ()),
            "resolved_inputs": [list(value) for value in node.resolved_inputs],
            "children": [node_row(child) for child in node.children or ()],
        }

    return json.dumps({
        "top_level": [node_row(node) for node in ir_tree.top_level],
        "blocks": [
            (repr(meta), [node_row(node) for node in roots])
            for meta, roots in ir_tree.blocks
        ],
    }, sort_keys=True, default=str)


def _by_sid(program, layer):
    return {
        program.by_uid(uid).aggregation_id: value
        for uid, value in layer.values.items()
    }


def _sum_per_rank(layer):
    return sum(row.per_rank for row in layer.values.values())


def _sum_full(layer):
    return sum(row.full for row in layer.values.values())


def _sum_full_per_step(access, instances):
    return sum(
        row.full * instances.values.get(uid, 1)
        for uid, row in access.values.items()
    )


def _sum_per_rank_per_step(access, instances):
    return sum(
        row.per_rank * instances.values.get(uid, 1)
        for uid, row in access.values.items()
    )


class TestIdentityEquivalence(unittest.TestCase):
    """`framework_binding=None` is a strict no-op.

    `post_tree` is structurally byte-equal to `pre_tree` (canonical
    hash compare); `Σpre.access_bytes_per_rank == Σpost.access_bytes_per_rank`;
    diagnostics is empty.
    """

    MODELS = (
        "meta-llama__Llama-3.2-1B",
        "deepseek-ai__DeepSeek-V3",
        "deepseek-ai__DeepSeek-V2-Lite",
    )

    def _check(self, model_name: str):
        resolved = _resolve(model_name)
        result = analyze(
            resolved,
            parallel=ParallelConfig(),
            framework_binding=None,
            batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            peak_flops_per_sec=1e12, peak_hbm_bw_bytes_per_sec=1e12,
            breakdown=True,
        )
        # Tree structurally unchanged from the one the fusion rewrite
        # started from — the serving tree with its comm and cast grafts
        # (`materialize_comm_tree`, `materialize_casts`), which a binding
        # does not decide.
        self.assertEqual(
            _tree_signature(result.post_tree),
            _tree_signature(result.comm_tree),
            f"{model_name}: post_tree signature drifted under identity binding",
        )
        # Access-bytes sums match under identity.
        self.assertEqual(
            _sum_per_rank(result.pre_access),
            _sum_per_rank(result.access),
            f"{model_name}: Σpre access_bytes != Σpost under identity",
        )
        # Diagnostics empty.
        self.assertEqual(result.fusion_matches, ())
        # Per-op timing non-empty (engine path was taken).
        self.assertGreater(
            len(result.per_op_timing), 0,
            f"{model_name}: per_op_timing list is empty",
        )

    def test_identity_llama_3_2_1b(self):
        self._check("meta-llama__Llama-3.2-1B")

    def test_identity_deepseek_v3(self):
        self._check("deepseek-ai__DeepSeek-V3")

    def test_identity_deepseek_v2_lite(self):
        self._check("deepseek-ai__DeepSeek-V2-Lite")


class TestFlashAttentionSavingThroughPipeline(unittest.TestCase):
    """End-to-end through `pipeline.analyze()`: Σpre - Σpost
    equals the FlashAttention saving formula `4×B×N×S²×act_dtype_bytes`
    per matched layer."""

    def test_llama_3_2_1b_flash_attention_saving(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        B, S = 1, 128
        binding = FrameworkBinding(
            framework="testing",
            audited_at=AuditedAt(tag="v0", commit="0" * 40),
            op_implementations=(
                ("flash_attention", FusionImplementation()),
            ),
        )
        result = analyze(
            resolved,
            parallel=ParallelConfig(),
            framework_binding=binding,
            batch_size=B, avg_input_tokens=S, avg_output_tokens=0,
            breakdown=True,
            )
        n_matches = len(result.fusion_matches)
        self.assertGreater(n_matches, 0, "expected ≥ 1 FlashAttention match")

        # Per-step (full-model) saving = per_layer × num_layers.
        # Llama-3.2-1B has 16 transformer layers; n_matches = 1 because
        # the FlashAttention chain appears once per template (the engine
        # multiplies per-step via block_results count, surfaced as
        # count_per_step in AnalysisResult).
        saved_per_step = (
            _sum_full_per_step(result.pre_access, result.pre_instances)
            - _sum_full_per_step(result.access, result.instances)
        )
        per_layer = 4 * B * 32 * S * S * 2  # N=32, dtype=2
        num_layers = 16  # Llama-3.2-1B
        expected_per_step = per_layer * num_layers
        self.assertEqual(
            saved_per_step, expected_per_step,
            f"Per-step bytes saved {saved_per_step} != per-layer({per_layer}) × "
            f"num_layers({num_layers}) = {expected_per_step}",
        )
        # n_matches is one (template-local match), not 16.
        self.assertEqual(n_matches, 1)


class TestFusionDelta(unittest.TestCase):
    """What fusion changes is the same row exit run on both trees: the rows
    before fusion price their collectives too, and no fusion changes
    nothing."""

    def _analyze(self, binding):
        return analyze(
            _resolve("meta-llama__Llama-3.2-1B"),
            parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2),
            framework_binding=binding, batch_size=1, avg_input_tokens=256, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3e12,
            interconnect_bw_bytes_per_sec=4e11, gpus_per_node=8, breakdown=True)

    def test_pre_fusion_rows_price_the_collectives(self):
        from backend.inference.framework_bindings import (
            DEFAULT_FRAMEWORK, get_binding)
        result = self._analyze(get_binding(DEFAULT_FRAMEWORK))
        self.assertTrue(result.fusion_matches)
        pre_comm = sum(r["t_comm_s"] for r in result.pre_per_op_timing)
        post_comm = sum(r["t_comm_s"] for r in result.per_op_timing)
        self.assertGreater(pre_comm, 0)
        self.assertGreater(post_comm, 0)
        delta = result.fusion_delta()
        self.assertEqual(delta["delta_s"],
                         delta["pre_total_s"] - delta["post_total_s"])

    def test_no_fusion_changes_nothing(self):
        delta = self._analyze(None).fusion_delta()
        self.assertEqual(delta["delta_s"], 0.0)
        self.assertGreater(delta["pre_total_s"], 0)


class TestPerOpTimingShape(unittest.TestCase):
    """per_op_timing carries the documented entry shape and adds
    up consistently with the access-bytes / flops totals."""

    def test_per_op_timing_entry_shape_and_sums(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        result = analyze(
            resolved,
            parallel=ParallelConfig(),
            framework_binding=None,
            batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            peak_flops_per_sec=312e12, peak_hbm_bw_bytes_per_sec=2e12,
            breakdown=True,
        )
        self.assertGreater(len(result.per_op_timing), 0)
        # Each entry has the required keys (M4 adds t_op_s / source /
        # source_badge; R1 adds shape_json / dtype_env_json so the route's
        # calibration lookup can exact-match against CSV identity).
        required_keys = {
            # `aggregation_id` is the canonical frontend join key
            # (`{template}|{display_path}` for block-internal leaves,
            # bare `display_path` at model root). `template_name` stays
            # alongside as a split-out filter convenience.
            "aggregation_id",
            # The one exact join key: aggregation_id is a reporting fact and
            # is not unique across template invocations; the module-tree
            # walk (`module_rows.physical_module_rows`) joins timing onto
            # its rows by uid.
            "uid",
            "template_name", "display_path", "op", "role", "count_per_step",
            # the node's own Invocation dims, so the identity reads spec
            # constants off the row rather than a model-wide table.
            "node_dims",
            # whether the row belongs to a MoE expert block (declared dispatch
            # zone or its permute/combine boundary), for the block aggregation
            # (`Program.dispatch_members`).
            "moe_block_member",
            "flops_per_rank", "access_bytes_per_rank",
            "t_compute_s", "t_memory_s", "t_sol_s",
            # which roofline term binds the row, and its FLOPs per byte.
            "bound", "arithmetic_intensity",
            # and the intensity's own popover (FLOPs over the bytes).
            "arithmetic_intensity_formula",
            # comm rooftop term (parallelism collectives over interconnect).
            "t_comm_s", "comm_bytes_per_rank",
            # the launch floor: a kernel's shortest time, however little it does.
            "t_launch_s",
            # The ordered framework selector's closed outcome. Empty when it
            # chose a launch; the singular value is the report and the list
            # is every candidate skipped, which coverage and the plan read.
            "launch_gap", "launch_gaps",
            # comm-row enrichments: world + chosen fabric bandwidth +
            # human-readable bytes derivation. Empty / 0 for non-comm
            # rows, populated when the row carries a collective.
            "comm_world", "comm_bw_bytes_per_sec",
            "comm_bw_label", "comm_bytes_formula", "t_comm_formula",
            "comm_collective_in_shape", "comm_collective_out_shape",
            "comm_payload_shape", "comm_payload_role",
            "comm_payload_dtype_bytes", "is_comm",
            # A routed exchange's top-k placement fan-out. None for every
            # non-routing collective; the communication identity and cost
            # both read this one published fact.
            "comm_route_fanout",
            "act_dtype", "comm_op", "comm_tensor_bytes",
            "comm_base_stride",
            # The quantised copy of its own operand a matmul's kernel needs
            # when the edge does not carry it, what that conversion is, and
            # the dtype environment the leaf ran in (what a measured row for
            # it is keyed by).
            "convert_bytes_per_rank", "conversion",
            "weight_dtype", "metadata_dtype",
            # How the weight's scales are laid out (per-tensor / per-block
            # …): a measured cell is keyed by it too, since a kernel serves
            # only the scale layout it reads.
            "weight_granularity",
            # The selected launch's input format when it quantises that
            # operand: which elements share a scale, and the dtype carrying
            # those scales.
            "activation_granularity", "activation_scale_dtype",
            # The dtype the leaf writes its primary output at, the cached
            # tensor that output is and its format's registry name (None
            # for an activation): the cell's `output` / `output_format`.
            "output_dtype", "output_cache", "output_format",
            # A hop across a pipeline cut: the cut, the producing occurrence
            # and the occurrences that read it past the cut; None / [] on
            # every other row.
            "cut", "source_occurrence", "sink_occurrences",
            # Hierarchical multi-phase decomposition for cross-node AR
            # (intra-RS + inter-AR-on-shard + intra-AG); empty list on
            # single-phase rows (intra-only or non-comm).
            "comm_phases",
            # The row's law over the workload's counts and the card — the
            # sweep substitutes it; the anchor row's numbers are the same
            # law at the anchor's point — and whether it costs nothing.
            "roofline_expr", "zero_cost",
            # `t_op_s` = effective per-op time; `t_measured_s` is the
            # measured-corpus time sitting beside the theory (None on the
            # theoretical-only path here — no corpus row matches).
            "t_op_s", "t_measured_s", "source", "source_badge",
            # The row's law over the batch and the card, which every
            # re-evaluation substitutes into.
            "shape_json", "dtype_env_json",
            # B: per-leaf formula tooltips for the breakdown.
            "flops_formula", "access_bytes_formula", "t_op_formula",
            # The two roofline terms' own popovers — each divides by the
            # peak its cell was priced at.
            "t_compute_formula", "t_memory_formula",
            # B1: component split for frontend batch-dropdown rescaling.
            "weight_bytes_per_rank", "metadata_bytes_per_rank",
            # Whole-leaf scales, counterpart of `weight_bytes_total_per_rank`.
            # `metadata_bytes_per_rank` is what THIS step read.
            "metadata_bytes_total_per_rank",
            # The write half of the activation term, named. `access_bytes` is
            # `weight + metadata + act_in + act_out`, so without this column
            # the phase's `mem_write_bytes` has nothing to fold from and the
            # read/write split cannot be a partition of the row sum.
            "act_out_bytes_per_rank",
            # And the read half, so all four terms of `access_bytes` are
            # published and none has to be recovered by subtraction.
            "act_in_bytes_per_rank",
            # Of the read half, the cache read: the step's KV read is the
            # sum of this column.
            "kv_read_bytes_per_rank",
            # C: MoE-aware weight saturation.
            "weight_bytes_total_per_rank",
            "weight_bytes_per_token_active_per_rank",
            # What the leaf actually read of its weights on THIS step: the
            # saturation min the engine already evaluated over the leaf's own
            # rows. The two columns above answer the same question at a
            # different batch; only this one answers it at this batch, and
            # without it a consumer re-derives a min over the wrong rows and
            # charges the difference to activation.
            "weight_read_bytes_per_rank",
            "active_ratio_per_token",
            # Shape columns source-of-truth: physical shape layers for compute
            # rows and explicit comm event geometry for collective rows.
            # `shape_in` may be a flat list (single-input op) or
            # list-of-lists (multi-input op). Companion `_full` fields
            # carry the analysis's unsharded global shape — populated only
            # when the sharded per-rank shape diverges, None for
            # replicated leaves under TP/EP = 1.
            # `shape_in_exact` is the same input shape before the axes are
            # rounded to whole tensors — a MoE token axis stays `51/64` there.
            # The grouped-GEMM calibration identity reads it to recover the
            # token count a rounded `1` has already lost.
            "shape_in",
            "shape_in_exact",
            # Each walked axis as its law in the workload, so a sweep
            # re-reads the row's shapes at another point instead of walking
            # the model again.
            "shape_law",
            # The shape columns are this walk's own batch's (a row re-read
            # at another batch says False where it has no law).
            "shape_at_batch",
            # The rest of what the row publishes that moves with the batch,
            # as laws (dims, the FLOPs record, a collective's payload) and
            # the facts that print it (`batch_sweep` reads it at a point).
            "point_law",
            "shape_in_dims",
            "shape_in_full",
            "shape_out",
            # Exact, like `shape_in_exact`: a routed buffer's per-expert rows
            # are a mean and may be a fraction.
            "shape_out_exact",
            "shape_out_dims",
            "shape_out_full",
            # Multi-port outputs: {port: shape} for ops that emit more than
            # one tensor (fused chains); `shape_out` stays the primary output.
            "shape_outs",
            # Dtypes follow the same input/output port structure as the
            # shapes, and weights name each stored operand beside its dtype.
            "input_dtypes", "port_dtypes", "weights",
            # The PARAM's shape, beside the activations': `shape_in` is
            # activations only, so a linear's weight geometry has nowhere else
            # to appear, and kernel_bench's plan joins on it.
            "shape_weight",
            # Its axes by identity (as `shape_in_dims`): which one is an
            # expert axis is asked by identity, not by position.
            "shape_weight_dims",
            # Each matrix partition the kernel multiplies, unsharded and on
            # one rank, with its pack and scale layout (`weight_shards`):
            # what the plan's block-scale shard rule reads.
            "weight_shards",
            # The dispatch's routed launch a leaf on its routed buffer runs
            # (`RoutedLaunch.wire`; None off any routed buffer): the one
            # record the plan's launch geometry reads.
            "routed",
            # Per-op roofline peak (driven by the row's dtype, not the
            # global dominant-weight column).
            "peak_flops_per_sec_for_op",
            # The framework's all-reduce dispatch the row's collective is
            # priced over (`comm_cost.CollectiveRung`), so another card
            # reads the same ladder.
            "comm_rungs",
        }
        from fastapi.encoders import jsonable_encoder
        from backend.inference.module_rows import served
        for row in result.per_op_timing:
            self.assertEqual(
                set(row.keys()), required_keys,
                f"per_op_timing row keys mismatch: {set(row.keys())}",
            )
            # A row goes out on the estimate / parallel-sweep wire as a
            # response serves it (`module_rows.served`: its laws stay in the
            # process) — every value must be JSON-encodable. `node_dims` in
            # particular carries resolved numbers (`as_number`), not sympy
            # `Dim`s.
            jsonable_encoder(served({"per_op_timing": [row]}))
            self.assertEqual(row["t_sol_s"], max(row["t_compute_s"], row["t_memory_s"]))
            # regression-lock: theoretical-only path keeps t_op_s == t_sol_s
            self.assertEqual(row["t_op_s"], row["t_sol_s"])
            self.assertEqual(row["source"], "theoretical_naive")
            self.assertEqual(row["source_badge"], {})
        # Σ per_op_timing[i].access_bytes_per_rank (per-step) ==
        # Σ post_access_bytes[sid].per_rank * count_per_step[sid].
        sum_per_op_access = sum(r["access_bytes_per_rank"] for r in result.per_op_timing)
        sum_post_access_step = _sum_per_rank_per_step(
            result.access, result.instances
        )
        self.assertEqual(
            sum_per_op_access, sum_post_access_step,
            "per_op_timing access_bytes sum disagrees with per-step post_access_bytes",
        )

    def test_per_op_timing_empty_without_peaks(self):
        # When peaks are None, t_compute / t_memory / t_sol stay 0.0
        # (rendered as em-dashes in the UI). The rows still exist;
        # only the timing fields are zero.
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        result = analyze(
            resolved,
            parallel=ParallelConfig(),
            framework_binding=None,
            batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            peak_flops_per_sec=None, peak_hbm_bw_bytes_per_sec=None,
            breakdown=True,
        )
        self.assertGreater(len(result.per_op_timing), 0)
        for row in result.per_op_timing:
            self.assertEqual(row["t_compute_s"], 0.0)
            self.assertEqual(row["t_memory_s"], 0.0)
            self.assertEqual(row["t_sol_s"], 0.0)


class TestTheRowCarriesTheWalksShare(unittest.TestCase):
    """`active_ratio_per_token` is `Analysis.active_share`, read not re-derived.

    The row's share used to be computed here, by dividing a published copy
    count that had the routed fraction multiplied into it. One statement now
    serves both the parameters a token touches and this column, so the column
    has to BE that statement — an equality, not an agreement.
    """

    def test_every_routed_row_reads_the_published_share(self):
        resolved = _resolve("Qwen__Qwen3-30B-A3B")
        result = analyze(
            resolved, parallel=ParallelConfig(), framework_binding=None,
            batch_size=1, avg_input_tokens=256, avg_output_tokens=0,
            peak_flops_per_sec=312e12, peak_hbm_bw_bytes_per_sec=2e12,
            breakdown=True,
        )
        share = result.analysis.active_share.values
        rows = {row["uid"]: row["active_ratio_per_token"]
                for row in result.per_op_timing if row.get("uid") is not None}
        self.assertTrue(rows)
        routed = 0
        for uid, ratio in rows.items():
            if uid not in share:            # a grafted collective: runs whole
                self.assertEqual(ratio, 1.0)
                continue
            self.assertEqual(ratio, float(share[uid]))
            if ratio != 1.0:
                routed += 1
        # The bank's leaves are the rows this is about; a run where none of
        # them is routed would pass on nothing.
        self.assertGreater(routed, 0)


class TestFullStepCoverage(unittest.TestCase):
    """Per-op timing covers EVERY leaf in the full per-step forward (all
    transformer layers via count_per_step, plus model-level leaves like
    `embed_tokens` and `lm_head`). Regression locks Codex's Round-0
    review finding: `pipeline.analyze()` previously emitted
    template-local per-op totals (one layer's worth) and omitted top-
    level leaves entirely.
    """

    def test_per_op_flops_sum_covers_full_per_step_on_reference_models(self):
        # Σ per_op_timing.flops_per_rank must cover engine.total_flops
        # at the full per-step scale on EVERY reference model — not
        # just dense Llama. The Round-1 fix passed Llama (single
        # template, no MoE) but undercounted DSv3 (44%), DSv2-Lite
        # (36%), and Kimi-K2 (31%) because the per-op pass skipped
        # repeated/active component nodes whose `nr.flops` baked in
        # the engine's `_apply_repeat_active` multiplier (MoE expert
        # banks aggregate `E_active` per token; the engine attributes
        # the multiplied flops to the expert component rather than
        # propagating to children).
        from backend.arch.spec_tree import iter_blocks
        from backend.arch.eval.dim import value_of
        models = [
            "meta-llama__Llama-3.2-1B",
            "deepseek-ai__DeepSeek-V3",
            "deepseek-ai__DeepSeek-V2-Lite",
            "moonshotai__Kimi-K2-Base",
        ]
        for model_name in models:
            with self.subTest(model=model_name):
                resolved = _resolve(model_name)
                result = analyze(
                    resolved,
                    parallel=ParallelConfig(),
                    framework_binding=None,
                    batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
                    breakdown=True,
                )
                # per_op_timing now excludes aux modules (MTP / vision /
                # audio / speculative) — they don't run on every step.
                # Subtract aux template flops from engine.total_flops to
                # get the main-trunk expectation.
                aux_templates = {
                    m.template_name for m in iter_blocks(resolved) if m.aux_kind
                }
                included = result.analysis.program.included_stack_names()
                expected_main_flops = round(sum(
                    float(value_of(cost.flops))
                    for stack, cost in result.analysis.by_stack().items()
                    if stack in included
                ))
                sum_per_op_flops = sum(r["flops_per_rank"] for r in result.per_op_timing)
                self.assertGreaterEqual(
                    sum_per_op_flops, expected_main_flops,
                    f"{model_name}: per_op_timing flops sum {sum_per_op_flops} "
                    f"({100*sum_per_op_flops/expected_main_flops:.1f}%) is BELOW "
                    f"main-trunk total {expected_main_flops}; missing "
                    f"{expected_main_flops - sum_per_op_flops} flops",
                )
                # Aux templates must NOT appear in per_op_timing — the
                # phase total is supposed to reflect only what runs every
                # forward step (MTP gates on speculative decoding, vision
                # tower gates on image input).
                aux_rows = [
                    r for r in result.per_op_timing
                    if r.get("template_name") in aux_templates
                ]
                self.assertEqual(
                    aux_rows, [],
                    f"{model_name}: per_op_timing leaked {len(aux_rows)} "
                    f"aux rows from templates {aux_templates}",
                )

    def test_top_level_compute_leaves_get_real_divisor(self):
        """Model-level compute-only leaves (e.g. DSv4's `hc_expand`)
        must get a real divisor in `pre_access_bytes`. Without one,
        `compute_leaf_divisors` (which only walks block roots) plus
        `iter_leaves` (only weight-bearing leaves) would record
        `LeafAccessBytes(0,0,1)` and silently drop activation traffic.
        pipeline.analyze adds an `aux` divisor for any pre_leaf
        compute_leaf_divisors missed.

        Verified by asserting: every leaf with non-zero cost or tensor shape
        has non-zero full traffic in the pre-rewrite access layer."""
        resolved = _resolve("deepseek-ai__DeepSeek-V4-Pro")
        result = analyze(
            resolved,
            parallel=ParallelConfig(),
            framework_binding=None,
            batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            breakdown=True,
        )
        dropped = []
        for uid, ab in result.pre_access.values.items():
            node = result.pre_program.by_uid(uid)
            if node.children:
                continue
            if ab.full > 0:
                continue
            sid = node.aggregation_id
            cost = result.pre_cost.values.get(uid)
            shape = result.pre_shapes.values.get(uid)
            # Zero bytes: only OK if the underlying leaf genuinely has
            # no real traffic. Two legitimate cases:
            #   (a) no params + no shape_in + no shape_out (degenerate)
            #   (b) view-class op: flops=0 AND params=0 AND no metadata
            #       — stride/layout transforms that move zero HBM bytes
            #       in any fused impl. access_bytes.annotate intentionally
            #       zeros these.
            is_view_op = (
                int(cost.flops or 0) == 0
                and int(cost.params or 0) == 0
                and int(cost.quant_metadata_params or 0) == 0
            )
            has_bytes = (
                cost.params > 0
                or bool(shape and shape.ins)
                or bool(shape and shape.out)
            )
            #   (c) identity source: a `parameter_*` whose output IS the
            #       tensor it stores. It carries params and moves nothing;
            #       the read is billed to whoever consumes it.
            if is_identity_source(node.op):
                continue
            if has_bytes and not is_view_op:
                dropped.append((sid, node.op, cost.params, shape.ins, shape.out))
        self.assertEqual(dropped, [], (
            f"non-view leaves dropped to LeafAccessBytes(0,0,*) despite "
            f"carrying real bytes: {dropped[:5]}. compute_leaf_divisors "
            f"missed these — the pipeline.py fallback that adds aux "
            f"divisors for missing entries regressed."
        ))

    def test_top_level_leaves_present_in_access_bytes(self):
        # embed_tokens and lm_head must appear in pre_access_bytes
        # (they fire once per forward step at the model root, outside
        # any block template).
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        result = analyze(
            resolved,
            parallel=ParallelConfig(),
            framework_binding=None,
            batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            breakdown=True,
        )
        access = _by_sid(result.pre_program, result.pre_access)
        counts = _by_sid(result.pre_program, result.pre_instances)
        self.assertIn(
            "embed_tokens", access,
            "embed_tokens missing from pre_access_bytes — top-level leaves "
            "were dropped from the flat node-results dict",
        )
        self.assertIn(
            "lm_head", access,
            "lm_head missing from pre_access_bytes",
        )
        # Also ensure count_per_step has them and = 1 (model-level singletons).
        self.assertEqual(counts.get("embed_tokens"), 1)
        self.assertEqual(counts.get("lm_head"), 1)

    def test_block_internal_count_per_step_matches_layer_count(self):
        # Block-internal leaves get the block's template count as their
        # count_per_step (Llama-3.2-1B = 16 transformer layers).
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        result = analyze(
            resolved,
            parallel=ParallelConfig(),
            framework_binding=None,
            batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            breakdown=True,
        )
        # q_proj is in the transformer block template → count = 16.
        # Llama's template is `block`; pipeline keys leaves by
        # `aggregation_id = <template>|<display_path>` so block-internal
        # nodes are scoped per template.
        counts = _by_sid(result.pre_program, result.pre_instances)
        self.assertEqual(
            counts.get("block|self_attn.q_proj"), 16,
            "block|self_attn.q_proj count_per_step should equal the number of "
            "transformer layers (16 for Llama-3.2-1B)",
        )

    def test_identity_per_step_access_sums_equal(self):
        # Identity equivalence at the FULL PER-STEP scale, not just
        # template-local. Σpre × count == Σpost × count under None binding.
        for model_name in [
            "meta-llama__Llama-3.2-1B",
            "deepseek-ai__DeepSeek-V3",
            "deepseek-ai__DeepSeek-V2-Lite",
        ]:
            with self.subTest(model=model_name):
                resolved = _resolve(model_name)
                result = analyze(
                    resolved,
                    parallel=ParallelConfig(),
                    framework_binding=None,
                    batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
                    breakdown=True,
                )
                pre_step = _sum_full_per_step(
                    result.pre_access, result.pre_instances
                )
                post_step = _sum_full_per_step(
                    result.access, result.instances
                )
                self.assertEqual(
                    pre_step, post_step,
                    f"{model_name}: per-step Σpre {pre_step} != Σpost {post_step} "
                    f"under identity binding",
                )


def _profile(body):
    from backend.arch.eval.precision import PrecisionProfile
    return PrecisionProfile.canonicalize(body)


class TestActivationRuleAccessBytes(unittest.TestCase):
    """A precision profile's activation rule must reach every primitive
    leaf under a roled component. The derivation prices each produced
    tensor at its graph-propagated role's edge dtype
    (`ResolvedPrecision.edge_bytes`) — `iter_tree` puts the enclosing
    component's role on the leaves, so role inheritance is a property of
    the graph itself here.

    The failure mode this guards: a rule that silently no-ops on
    primitive leaves (which are most of the model).
    """

    def test_ffn_moe_f32_activations_widen_only_moe_writes(self):
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        baseline = analyze(resolved, parallel=ParallelConfig(), avg_input_tokens=1, avg_output_tokens=0, breakdown=True)
        with_rule = analyze(
            resolved, parallel=ParallelConfig(),
            precision=_profile({"activations": [
                {"select": {"roles": ["ffn_moe"]},
                 "format": {"dtype": "F32"}}]}), avg_input_tokens=1, avg_output_tokens=0, breakdown=True)
        moe_hit = False
        non_moe_hit = False
        baseline_access = _by_sid(baseline.pre_program, baseline.pre_access)
        rule_access = _by_sid(with_rule.pre_program, with_rule.pre_access)
        for sid, ab_base in baseline_access.items():
            ab_over = rule_access.get(sid)
            if ab_over is None or not (ab_base.act_in_bytes
                                       or ab_base.act_out_bytes):
                continue
            if ab_over.act_dtype_bytes == 4.0:
                if ab_over.output_activation_format is not None:
                    # A write in a quantised record — the cast's declared
                    # output, or an element copy of it (the MoE dispatch
                    # moves the experts' FP8 copy) — whatever the edge
                    # carries. A rule selecting its role widens the
                    # producing leaf's write (asserted on that leaf's own
                    # row), and with it the cast's read — never the record.
                    self.assertEqual(
                        ab_over.act_out_bytes, ab_base.act_out_bytes,
                        msg=f"{sid}: a quantised record's write moved "
                            "under an activation rule",
                    )
                else:
                    self.assertGreater(
                        ab_over.act_out_bytes, ab_base.act_out_bytes,
                        msg=f"{sid}: ffn_moe F32 rule didn't widen the write",
                    )
                if "expert" in sid or "mlp" in sid:
                    moe_hit = True
            else:
                self.assertEqual(
                    ab_over.act_dtype_bytes, ab_base.act_dtype_bytes,
                    msg=f"{sid}: non-moe leaf width moved under "
                        "an ffn_moe-only rule",
                )
                self.assertEqual(
                    ab_over.act_out_bytes, ab_base.act_out_bytes,
                    msg=f"{sid}: non-moe write moved under "
                        "an ffn_moe-only rule",
                )
                non_moe_hit = True
        self.assertTrue(moe_hit, "no moe leaf picked up the activation rule "
                                 "— the profile silently no-oped")
        self.assertTrue(non_moe_hit, "no untouched leaf found — the "
                                     "rule leaked model-wide")

    def test_hybrid_slot_resolves_through_role_alias(self):
        # A hybrid model's windowed layers sit in an `attention_swa` SLOT
        # filled by a component declaring `role: attention`. Rules select
        # roles, so an `attention` rule must reach the SWA leaves through
        # `Program.role_aliases` — on BOTH the activation table and the
        # weight table. Without the alias hop the model comes out with
        # the whole windowed half of its attention unpriced.
        resolved = _resolve("XiaomiMiMo__MiMo-V2-Flash")
        over = analyze(
            resolved, parallel=ParallelConfig(),
            precision=_profile({
                "activations": [{"select": {"roles": ["attention"]},
                                 "format": {"dtype": "F32"}}],
                "weights": [{"select": {"roles": ["attention"]},
                             "format": {"dtype": "F32"}}],
            }), avg_input_tokens=1, avg_output_tokens=0, breakdown=True)
        access = _by_sid(over.pre_program, over.pre_access)
        swa_attn = [ab for sid, ab in access.items()
                    if "swa" in sid.lower() and "self_attn." in sid]
        swa_norm = [ab for sid, ab in access.items()
                    if "swa" in sid.lower() and "layernorm" in sid]
        self.assertTrue(swa_attn, "fixture lost its SWA attention leaves")
        self.assertTrue(swa_norm, "fixture lost its SWA-template norms")
        weighted = 0
        for ab in swa_attn:
            self.assertEqual(ab.act_dtype_bytes, 4.0,
                             "SWA leaf missed the attention activation rule "
                             "— role_aliases not consulted")
            if ab.weight_bytes and ab.params:
                # Effective width (bytes over elements streamed). Leaves
                # that stream no weight — views, and parameter rows whose
                # bytes are billed to their consumer — rightly show 0.
                self.assertEqual(ab.weight_dtype_bytes, 4.0,
                                 "SWA leaf missed the attention weight rule")
                weighted += 1
        self.assertTrue(weighted, "no SWA attention leaf streams a weight")
        for ab in swa_norm:
            self.assertEqual(ab.act_dtype_bytes, 2.0,
                             "norm leaf wrongly caught the attention rule")


class TestWeightRuleAccessBytes(unittest.TestCase):
    """A weight rule must reach per-op HBM traffic (decode mem_read), not
    only the storage ledger: the leaves it selects stream the deployment
    format, scales included; every other leaf is untouched.
    """

    def test_step37_ffn_moe_fp8_drops_moe_access(self):
        resolved = _resolve("stepfun-ai__Step-3.7-Flash")
        parallel = ParallelConfig(
            tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8,
            ep_moe=1, cp=1, dp_replicas=1,
        )
        baseline = analyze(resolved, parallel=parallel, avg_input_tokens=1, avg_output_tokens=0, breakdown=True)
        with_rule = analyze(
            resolved, parallel=parallel,
            precision=_profile({"weights": [
                {"select": {"roles": ["ffn_moe"]},
                 "format": {"scheme": "fp8"}}]}), avg_input_tokens=1, avg_output_tokens=0, breakdown=True)
        moe_hit = False
        non_moe_hit = False
        baseline_access = _by_sid(baseline.pre_program, baseline.pre_access)
        rule_access = _by_sid(with_rule.pre_program, with_rule.pre_access)
        for sid, ab_base in baseline_access.items():
            ab_over = rule_access.get(sid)
            if ab_over is None or ab_base.weight_bytes == 0:
                continue
            if ab_over.weight_bytes != ab_base.weight_bytes:
                # FP8 data is one byte per element; the block-128 scales
                # ride as metadata the leaf also streams.
                self.assertEqual(
                    ab_over.weight_bytes, ab_over.weight_elems,
                    msg=f"{sid}: fp8 data is not elements x 1 byte",
                )
                self.assertGreater(
                    ab_over.metadata_bytes, ab_base.metadata_bytes,
                    msg=f"{sid}: fp8 block scales not streamed",
                )
                self.assertLess(ab_over.full, ab_base.full,
                                msg=f"{sid}: fp8 did not shrink the read")
                self.assertIn("moe.", sid, msg=f"{sid}: not an ffn_moe leaf")
                moe_hit = True
            else:
                self.assertEqual(
                    ab_over.metadata_bytes, ab_base.metadata_bytes,
                    msg=f"{sid}: non-MoE leaf metadata drifted",
                )
                non_moe_hit = True
        self.assertTrue(moe_hit, "no ffn_moe leaves found in fixture")
        self.assertTrue(non_moe_hit, "no non-MoE leaves found in fixture")

    def test_empty_profile_is_as_shipped(self):
        resolved = _resolve("meta-llama__Llama-3.2-3B")
        baseline = analyze(resolved, parallel=ParallelConfig(), avg_input_tokens=1, avg_output_tokens=0, breakdown=True)
        with_empty = analyze(
            resolved, parallel=ParallelConfig(), precision=_profile({}), avg_input_tokens=1, avg_output_tokens=0, breakdown=True)
        baseline_access = _by_sid(baseline.pre_program, baseline.pre_access)
        empty_access = _by_sid(with_empty.pre_program, with_empty.pre_access)
        for sid, ab_base in baseline_access.items():
            ab_empty = empty_access[sid]
            self.assertEqual(
                ab_base.weight_bytes, ab_empty.weight_bytes,
                msg=f"{sid}: empty profile drifted weight_bytes",
            )
            self.assertEqual(
                ab_base.per_rank, ab_empty.per_rank,
                msg=f"{sid}: empty profile drifted per_rank",
            )


if __name__ == "__main__":
    unittest.main()
