"""What the tests keep about the corpus that is NOT a pin, and the scanner
that finds it.

Four syntactic facts make a closed per-model set in a test; the registry test
(`tests/meta/test_corpus_pin_registry.py`) turns each into "register it or go
red":
  * a module walks the corpus (`pytest.mark.regression`, a `.glob` over the
    snapshot directory, `all_specs()`, `SNAPSHOT_CORPUS`, `SNAPSHOT_DIR`,
    `Corpus`) — a pin, a check or a curated set must claim it;
  * a module- or class-level container literal holds a corpus identifier (a
    snapshot stem, file name or model id) — it is a curated set, listed in
    `tests/meta/corpus_literal_manifest.json`;
  * a corpus-walking module keeps a module- or class-level number — it is a
    pin's data or a `Parameter`;
  * a corpus-walking module builds a snapshot identifier from strings — it
    declares whether its keys come from the corpus or from a constant.
The scan is a ratchet over those forms, not a proof: a literal inside a
function body, an identifier read from a file outside `tests/golden/`, or one
computed indirectly in a module that neither walks the corpus nor is marked
is left to review.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from tools.corpus_pins.check import CheckKind, CiOnlyCheck
from tools.corpus_pins.corpus import ROOT, SNAPSHOT_DIR
from tools.corpus_pins.pin import CuratedSet, Parameter

TESTS_DIR = ROOT / "tests"
MANIFEST = TESTS_DIR / "meta" / "corpus_literal_manifest.json"

#: the names whose presence in a module means it walks the corpus
CORPUS_NAMES = ("all_specs", "SNAPSHOT_CORPUS", "SNAPSHOT_DIR", "Corpus")


def corpus_identifiers(snapshot_dir: Path = SNAPSHOT_DIR) -> Set[str]:
    """Every snapshot's stem, file name and model id."""
    out: Set[str] = set()
    for path in snapshot_dir.glob("*.json"):
        out |= {path.stem, path.name, path.stem.replace("__", "/", 1)}
    return out


def module_paths(root: Path = TESTS_DIR) -> List[Path]:
    """Every Python file under `tests/`."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_module_paths(root: Path = TESTS_DIR) -> List[Path]:
    """The test modules (`test_*.py`) under `tests/`; the shared helpers
    beside them are not tests."""
    return [p for p in module_paths(root) if p.name.startswith("test_")]


def module_name(path: Path) -> str:
    """A module's path relative to the repository root (as given, outside it)."""
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _strings(node: ast.AST) -> Iterator[str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


_CONTAINERS = (ast.Dict, ast.Set, ast.List, ast.Tuple, ast.ListComp,
               ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Call)


def _assignments(tree: ast.Module) -> Iterator[Tuple[str, ast.AST]]:
    """(qualified name, value) for every module- and class-level assignment."""
    def visit(body: List[ast.stmt], prefix: str) -> Iterator[Tuple[str, ast.AST]]:
        for stmt in body:
            if isinstance(stmt, ast.ClassDef):
                yield from visit(stmt.body, f"{prefix}{stmt.name}.")
            elif isinstance(stmt, ast.Assign) and stmt.value is not None:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        yield f"{prefix}{target.id}", stmt.value
            elif (isinstance(stmt, ast.AnnAssign) and stmt.value is not None
                  and isinstance(stmt.target, ast.Name)):
                yield f"{prefix}{stmt.target.id}", stmt.value
    yield from visit(tree.body, "")


@dataclass(frozen=True)
class LiteralHit:
    """A container literal that names corpus identifiers."""

    module: str
    name: str
    identifiers: Tuple[str, ...]

    def row(self, kind: str) -> Dict[str, object]:
        return {"module": self.module, "name": self.name,
                "identifiers": list(self.identifiers), "kind": kind}


def literal_hits(paths: List[Path], identifiers: Set[str]) -> List[LiteralHit]:
    hits = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name, value in _assignments(tree):
            if not isinstance(value, _CONTAINERS):
                continue
            found = sorted({s for s in _strings(value) if s in identifiers})
            if found:
                hits.append(LiteralHit(module_name(path), name, tuple(found)))
    return hits


def _is_corpus_mark(node: ast.AST) -> bool:
    return any(isinstance(sub, ast.Attribute) and sub.attr == "regression"
               for sub in ast.walk(node))


def walks_corpus(path: Path) -> bool:
    """Whether a test module walks the corpus, by the syntactic facts above."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for name, value in _assignments(tree):
        if name == "pytestmark" and _is_corpus_mark(value):
            return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in CORPUS_NAMES:
            return True
        if isinstance(node, ast.Attribute) and node.attr in CORPUS_NAMES:
            return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "glob"
                and "snapshot" in ast.unparse(node.func.value).lower()):
            return True
        if isinstance(node, ast.alias) and node.name in CORPUS_NAMES:
            return True
    return False


def numeric_constants(path: Path) -> List[str]:
    """Qualified names of module- and class-level assignments that hold a
    number as data: a literal, or one inside a container literal or a call's
    arguments (`Counter({...: 3})`). An index into an expression
    (`parents[3]`) is not data."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for name, value in _assignments(tree):
        indices = {id(sub.slice) for sub in ast.walk(value) if isinstance(sub, ast.Subscript)}
        indexed = {id(n) for sub in ast.walk(value) if isinstance(sub, ast.Subscript)
                   for n in ast.walk(sub.slice)} | indices
        if any(isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float))
               and not isinstance(sub.value, bool) and id(sub) not in indexed
               for sub in ast.walk(value)):
            out.append(name)
    return out


def dynamic_identifiers(path: Path) -> bool:
    """Whether a module builds snapshot identifiers from strings: a
    `.replace("__", "/")` (stem to model id) or its inverse."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "replace" and len(node.args) >= 2
                and all(isinstance(a, ast.Constant) for a in node.args[:2])
                and {node.args[0].value, node.args[1].value} == {"__", "/"}):
            return True
    return False


def load_manifest(path: Path = MANIFEST) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_rows(hits: List[LiteralHit], kinds: Dict[Tuple[str, str], str]) -> List[Dict[str, object]]:
    return [hit.row(kinds[(hit.module, hit.name)]) for hit in hits]


def render_manifest(rows: List[Dict[str, object]]) -> bytes:
    return (json.dumps(sorted(rows, key=lambda r: (r["module"], r["name"])),
                       indent=2, sort_keys=True) + "\n").encode("utf-8")


# ── the registered sets ──────────────────────────────────────────────────

#: Collections a test keeps for a chosen subset of the corpus, each with the
#: reason it does not grow with the corpus. A new model does not join them.
#: Their container literals are exactly the manifest's rows.
CURATED_OUT_OF_SCOPE: Tuple[CuratedSet, ...] = (
    CuratedSet('tests/backend/inference/test_precision_launch_selection.py',
               '_SGLANG_STEMS',
               'the SGLang-only hybrid KDA/DSA runtime profile whose MXFP4 '
               'launch precision is audited against SGLang; not all models'),
    CuratedSet('tests/backend/arch/layout/test_layout_physical.py', 'LANE_COHORT',
               'the chosen lane cohort, read from its golden',
               artifact='tests/golden/shardv3_lane_snapshots.json'),
    CuratedSet('tests/backend/arch/layout/test_layout_weight_adapter.py', 'LANE_COHORT',
               'the chosen lane cohort, read from its golden',
               artifact='tests/golden/shardv3_lane_snapshots.json'),
    CuratedSet('tests/backend/arch/eval/test_absorbed_latent_read.py', 'CASES', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_serving_measure_law.py', 'LONG_CASES',
               "a MoE, an MTP MoE and a hybrid-attention MoE whose stored measure keys are checked against a direct estimate, not the corpus"),
    CuratedSet('tests/backend/inference/test_pipeline_parallel_compute.py',
               '_PP_EXTRA_TP',
               'the one model whose pp=8 point needs a tensor cut to have a '
               'clock at all: GLM-5.1 holds 173 GiB a rank unsharded and 92 '
               'at tp=2, and a stage admitting no request publishes none'),
    CuratedSet('tests/backend/arch/eval/test_flops_closure.py', '_HELD',
               'the two models whose closure case needs a layout that admits '
               'a request: a card admitting none publishes no per-op rows to '
               'weigh. The small models fit anywhere and are not listed'),
    CuratedSet('tests/backend/arch/eval/test_flops_closure.py', '_STAGED',
               'the same two models under pipeline stages, for the same '
               'reason: the cut that holds a request, not a chosen cohort'),
    CuratedSet('tests/backend/inference/test_attention_kernel_from_cache.py', 'CASES',
               'one model per attention kernel a fused span keys to (MLA '
               'absorbed and gated, GQA, DSA); the kernel is read off the '
               'cache slot, not the model, so the corpus does not join this'),
    CuratedSet('tests/backend/inference/test_precision_launch_selection.py',
               '_DOT_STEMS',
               'the three dot-suffix MXFP4 shapes the ordered selector is '
               'pinned against; their scheme and request come from each '
               'snapshot walk, not from this list'),
    CuratedSet('tests/backend/inference/test_precision_launch_selection.py',
               '_PACK_STEMS',
               'the two pack-quantized MXFP4 shapes whose Hopper/Blackwell '
               'answers are pinned; a new MXFP4 shape joins through its own '
               'selector coverage, not by growing this set'),
    CuratedSet('tests/backend/inference/test_prefill_chunk_sol.py', 'MODEL',
               'the one dense model whose chunked-prefill closure is checked '
               'against itself; the corpus is not asked to grow this set'),
    CuratedSet('tests/kernel_bench/test_rekey_lake.py', 'H100_MASKED_ROWS',
               'one recorded H100 launch shape for each model named by the '
               'activation-granularity review; the migration is licensed by '
               'those rows, so the corpus does not grow this set'),
    CuratedSet('tests/backend/arch/eval/test_replica_cost.py', 'MODELS',
               'a shared-expert bank whose cost is a multiple of its '
               "children's, an MLA MoE and a GQA MoE — the shapes one "
               "replica's cost composes over; not the corpus"),
    CuratedSet('tests/backend/test_prefill_costs_agree.py', 'MODELS',
               'a sparse MLA and a dense one, the two prefill forms; not the '
               'corpus'),
    CuratedSet('tests/backend/arch/test_moe_unbound_defaults.py', 'SHARED_WIDTH',
               "the specs whose shared-width fallback is their HF config class's literal default, each read with its snapshot's config; not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_moe_variant_conservation.py', '_VARIANTS',
               'one model per MoE component the corpus reaches, with its '
               "whole-model FLOPs and params. It does not grow with the "
               'corpus: a new model reusing an existing MoE component adds '
               'no row, and a new COMPONENT is forced in by the module\'s '
               'own coverage assertion, which enumerates from the walk '
               'rather than from this table'),
    CuratedSet('tests/backend/arch/eval/test_moe_variant_conservation.py', '_BOUNDARY',
               'the four models whose dispatch-boundary bytes are written '
               'out with their closed forms in the table\'s own comment, so '
               'the numbers can be checked against a model config without '
               'this engine; a new model does not join them'),
    CuratedSet('tests/backend/arch/eval/test_copy_factors.py', '_REPLACED',
               'one leaf per shape of disagreement between the retired '
               'substring guess and the declared display noun — a vision '
               'tower, an audio encoder, two more towers and a speech bank; '
               'the record of what the field fixed, not a corpus sweep'),
    CuratedSet('tests/backend/arch/layout/test_encoder_parallelism.py', '_ENCODERS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_borrowed_weight_reads.py', 'TestFusionKeepsTheBorrowedWeight.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_batch_sweep_named_dims.py', 'CASES',
               'two sparse MoE deployments and one with a cadence axis — '
               'the shapes whose extents are fractions'),
    CuratedSet('tests/backend/arch/eval/test_eval_symbolic_workload.py',
               'TestTheCacheReadIsAFormula.CASE',
               'one deployment whose decode cache read is billed at some '
               'points and not at others, to assert one walk with no point '
               'answers both sides; the engine writes every such value the '
               'same way'),
    CuratedSet('tests/backend/arch/eval/test_eval_symbolic_workload.py',
               'TestOneWalkIsEveryWalkAcrossTheCorpus.CASES',
               'one deployment per thing that can make a symbolic workload '
               'walk differently — a dense model, a sequence cut, an expert '
               'mesh, a compressed window, a hybrid, and decode; a new model '
               'is held to the equivalence by the default case and the '
               'engine, not by this list'),
    CuratedSet('tests/backend/inference/test_scaling_payload.py', '_GOLDEN',
               'four deployments whose Scaling payload is held to its pinned '
               'bytes point by point — an expert mesh, a dense model, a '
               'sequence cut and a hybrid; not a corpus sweep',
               artifact='tests/golden/scaling_payload.json'),
    CuratedSet('tests/kernel_bench/test_kernel_bench_plan_key.py', 'MODELS',
               'the four models whose every measured job was unreachable, '
               'plus the partly-unreachable one and a dense control'),
    CuratedSet('tests/backend/arch/eval/test_eval_cache.py', 'AGREES', 'KV reconciliation subset with no completeness assertion'),
    CuratedSet('tests/backend/arch/eval/test_eval_fusion.py', 'MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_eval_instances.py', 'TestPerRank.MOE', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_eval_traffic.py', 'TestFusionShowsUpInBytesAndOnlyThere.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_flops_closure.py', 'CLOSURE_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_isolated_cache.py', 'MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_kv_stored_size.py', 'FAST', "the default-run case of the stored-size check: one dense model with a scaled cache format"),
    CuratedSet('tests/backend/inference/test_kv_stored_size.py', 'SWEEP', "the cache families whose stored size carries dynamic scales (latent, index key, compressed, pooled); a chosen cohort, not the corpus"),
    CuratedSet('tests/backend/inference/test_access_formula_states_the_charge.py', 'MODELS', "models whose rows exercise every term the access popover states (views, pooled caches, an expert bank) plus a dense control; a chosen cohort, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_walk_decides_nothing.py', 'MODELS', "one model per kind of walk the workload could steer — dense, a sparse MoE with an absorbed-latent cache, a hybrid with recurrent state, a sliding window with sinks, pooled windows of compressed rows — walked unsettled on every mesh; a chosen cohort, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_view_fanout_bytes.py', 'MODELS', "the models whose fused kernels read broadcast views (indexer head weights, hyper-connection mixes) when a repeating view's reader was first charged the tensor behind it; a chosen cohort, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_pooled_index_cache.py', 'MODELS', "the models whose indexer scores pooled keys at the time the pooled cache took the serving layout; a new one is held to it by its kv_growth claims, not by this list"),
    CuratedSet('tests/backend/arch/eval/test_storage_bytes_closure.py', 'TestBlockStorageClosesOnTheCheckpoint.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/layout/test_cp_prefill_coverage.py', 'CASES', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/layout/test_eval_placement_diff.py', 'PINNED', 'chosen (model, parallel case) placements'),
    CuratedSet('tests/backend/arch/layout/test_layout_propagation_corpus.py', 'MODELS', 'chosen models for per-op propagation'),
    CuratedSet('tests/backend/arch/spec_tree/test_tied_weight_declaration.py', 'TestTheHeadNamesItsEmbedding.TIED', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/test_axis_propagation.py', 'TestIRCarriesPermutation.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/test_forward_nonlinearities.py', 'QWEN_HYBRIDS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/test_declared_identities.py', '_BOUND_ROLE_SNAPSHOTS', "the snapshots whose IR roles changed when a params-bound component took its template's role; no completeness assertion"),
    CuratedSet('tests/backend/arch/test_declared_identities.py', '_MROPE_ATTENTION', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/test_spec_ir_node_refs.py', 'MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/validate/test_curve_axioms.py', '_COHORT', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/eval/test_attention_mask.py', 'VALUE_SCALE_MODELS', "the snapshots whose config carries `attention_value_scale` when the MiMo attentions took their V scale; a new one is held to its V scale by review of the component it binds, not by this list"),
    CuratedSet('tests/backend/arch/validate/test_rotary_positions.py', 'MROPE_MODELS', "the models whose rope parameters carry `mrope_section`, at the time multimodal RoPE became an op; a new one is held to it by the validate gate (`_rotary_positions`), not by this list"),
    CuratedSet('tests/backend/arch/validate/test_validator_binding.py', 'TestHFBindingJoinInvariants._MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/validate/test_validator_golden.py', '_STRICT_MODELS', 'tiered ratchet of ship-gate-clean models; corpus-wide coverage is C4'),
    CuratedSet('tests/backend/arch/validate/test_validator_golden.py', '_WATERMARK_MODELS', 'tiered ratchet with per-model watermarks; corpus-wide coverage is C4'),
    CuratedSet('tests/backend/hf/test_hf_models.py', 'DEFAULT_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_activation_peak_physical.py', '_REFERENCE_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_allreduce_fused_rmsnorm.py', 'TestTPSwitchesBetweenNormFusions.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_capacity_golden.py', '_CASES', 'hand-calibrated deployments; the wide per-model surface is P2',
               artifact='tests/golden/capacity_baseline.json'),
    CuratedSet('tests/backend/inference/test_consistency_invariants.py', '_CASES', 'chosen deployments for the consistency invariants'),
    CuratedSet('tests/backend/inference/test_payload_fields.py', 'CASES', 'two estimates whose published field paths are pinned: a dense model and the widest (DSv4-Flash); the field set does not grow with the corpus',
               artifact='tests/golden/payload_fields.json'),
    CuratedSet('tests/backend/inference/test_decode_context_parallel.py', 'MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_dispatch_zone_membership.py', 'ROUTED', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_dispatch_zone_membership.py', 'DENSE', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_inference_invariants.py', '_CASES', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_combinations.py', '_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_comm_volume.py', 'TestTensorParallelVolume.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_comm_volume.py', 'TestSequenceParallelVolume.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_comm_volume.py', 'TestContextParallelVolume.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_comm_volume.py', 'TestContextParallelDenseIsDataParallel.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_comm_volume.py', 'TestExpertParallelVolume.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_comm_volume.py', 'TestExpertXTensorParallelVolume.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_parallelism_comm_volume.py', 'TestCommVolumeInvariantSweep.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_per_rank_flops_rows.py', 'MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_perturbation_invariants.py', '_CASES', 'chosen deployments for the perturbation invariants'),
    CuratedSet('tests/backend/inference/test_phase_traffic_split.py', 'TestTrafficSplit.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_pipeline.py', 'TestIdentityEquivalence.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_pipeline_parallel_cache.py', '_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_pipeline_parallel_compute.py', '_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_pp_stage_coherence.py', 'CASES', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_pp_stage_coherence.py', 'DISAGREEING', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_row_roofline.py', 'TestAPipelineStepReadElsewhere.CASES', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_scales_follow_weights.py', '_QUANTIZED_MOE', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_sharding.py', 'TestClassIsDeploymentIndependent.MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_sharding.py', 'TestVocabEndpointsAskBoundaryRole.CASES', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/playground/test_compose_corpus.py', '_SWAPS', 'chosen replacement slots'),
    CuratedSet('tests/backend/rendering/test_compute_graph_renderable.py', '_FUSED_AT_LEAST', 'per-model fusion floors for the models that fuse; no completeness assertion'),
    CuratedSet('tests/backend/service/test_module_variants_routes.py', '_ROW', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/arch/test_media_processor.py', 'ENCODER_IMAGES', "the snapshots with an image encoder and a processor configuration, by hand: a model added with one is added here"),
    CuratedSet('tests/backend/test_analyzer.py', '_HEADLINE_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/views/test_diff.py', '_ATTENTION', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/views/test_diff.py', '_LINEAR', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/corpus_row.py', '_BASE', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_compare_frontend.py', '_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_graph_diff_frontend.py', '_CASES', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_pages_smoke.py', '_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_pages_smoke.py', '_FRONTIER', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_pages_smoke.py', '_GALLERY', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_pages_smoke.py', '_CORPUS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_pages_smoke.py', '_CORPUS_ROW', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_pages_smoke.py', '_GRAPH', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_playground_frontend.py', '_COMPONENTS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_playground_frontend.py', '_MODELS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_playground_frontend.py', '_BENCH_RESULTS', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/frontend/test_playground_frontend.py', '_BINDINGS_BY_COMPONENT', "a chosen cohort for this test's behaviour, not the corpus"),
    CuratedSet('tests/backend/inference/test_sweep_parity.py', '_POINT_CASES',
               "one deployment per thing a row's law has to carry at another "
               "point: a dense TP1 baseline, a parameter-shaped node dim and "
               "an every-m-steps cadence (DSv4-Flash), a routed bank (EP8), a "
               "sliding window (gpt-oss). Substitution agreeing with a walk is "
               "a property of the laws, not of a model, so the corpus does not "
               "join this",),
    CuratedSet('tests/backend/api/test_gallery_frontier_class_walk.py', 'CASES',
               "one layout per walk class the Gallery prices: three Hopper "
               "cards of different HBM with an MTP draft and a collective, the "
               "NVL72 card walking for the 8-GPU Blackwell node across a node "
               "boundary, a routed Hopper layout, and the class of one card. "
               "A class walk priced on a card agreeing with that card's own "
               "walk is a property of the pricing, not of a model, so the "
               "corpus does not join this",),
    CuratedSet('tests/backend/api/test_gallery_frontier_class_walk.py', 'LAW_CASES',
               "the law road's oracle: one model whose per-op rows differ by "
               "walk class (DeepSeek-V4-Flash, Blackwell-only activation "
               "casts) on a Hopper and a Blackwell card, and one whose rows do "
               "not (Llama-3.2-3B). A card priced off its own class's law is a "
               "property of the pricing, not of a model, so the corpus does "
               "not join this",),
    CuratedSet('tests/backend/inference/test_step_sum.py', '_CASES',
               "a dense step, a routed one and a pipelined one: the three "
               "shapes a step's one aggregation has to add up. What a step "
               "sums is the engine's arithmetic, not a property of a model",),
    CuratedSet('tests/backend/arch/test_block_label.py', '_MODELS',
               "one model for each block-labelling rule the corpus reaches "
               "(hash- and score-routed MoE, KDA, Mamba, GLM attention, an "
               "engram wrapper and a named LongCat backbone). The label is a "
               "property of what a block calls, not of a model, so a new "
               "model repeating a rule is held to it by the same labeller "
               "rather than by growing this cohort",),
    CuratedSet('tests/kernel_bench/test_plan_keys_are_the_estimates.py',
               'TestTheTwoSidesAskOneKey.MODELS',
               "one BF16, one FP8 and one FP4 checkpoint: the three dtype "
               "environments the write side and the read side can disagree "
               "about. The equality is a property of the one key derivation, "
               "not of a model, so the corpus does not join this — a new "
               "model is held to it by that derivation",),
    CuratedSet('tests/kernel_bench/test_fused_span_providers.py', '_PLANNED',
               "one (model, phase, framework) cell per fused span measured "
               "here, each the deployment whose plan reaches that span: the "
               "GDN step, its gate prep and its gated norm (Qwen3.5-35B-A3B), "
               "the dilated PLE conv (Qwen3.8-Flash-Next), the KDA step "
               "(GLM-5.3-Flash), the whole-width gated norm (Ling-2.6-1T), the "
               "four routers (Hy3 sigmoid, DSv4-Flash sqrt-softplus, LongCat "
               "softmax-with-bias, Qwen3.5 softmax) and partial rope on GQA "
               "and MLA heads (GLM-4.5-Air, DeepSeek-V3). The cell is keyed to "
               "the span, so a new model reaching an already-measured span "
               "adds no row; a new span adds one"),
    CuratedSet('tests/kernel_bench/test_kernel_bench_plan_pp.py', 'SMALL',
               "one small snapshot, chosen because its layer count makes "
               "`stage_layer_ranges` uneven; what a pipeline stage bills is "
               "the placement's arithmetic and not a property of a model, so "
               "the corpus does not join this",
               artifact='tests/golden/kernel_bench_plan_pp1.json'),
    CuratedSet('tests/kernel_bench/test_mxfp4_moe_keying.py', 'QUANTISING',
               "one model per DISTINCT activation declaration against a "
               "4-bit expert weight, which is what decides whether the card "
               "reaches that quantiser: DeepSeek-V4.1-Flash "
               "(`mxfp4-dot-suffix` at `act_group: 32`), DeepSeek-V4-Flash "
               "(the same scheme at the default 128, the entry V4-Pro shares, "
               "so V4-Pro would add no case) and MiMo-V2.6-Pro-RL "
               "(`mxfp4-pack-quantized` with `input_activations: "
               "fp8_dynamic_group32`). The complement is the same file's "
               "weight-only test — gpt-oss and Kimi-K3 declare no quantiser "
               "and are W4A16 on every card. Which activation a launch eats "
               "is a property of that declaration and of the card's peak "
               "columns, not of a model, so the corpus does not join this: a "
               "new model repeating one of these declarations is held to it "
               "by the same resolver",),
    CuratedSet('tests/kernel_bench/test_mxfp4_moe_keying.py',
               'BLACKWELL_UNRESOLVED',
               "of those three, the one whose Blackwell-side launch nobody "
               "has ruled yet: MiMo-V2.6-Pro-RL, whose vLLM entry "
               "(`Fp8Config(store_dtype=\"mxfp4\")` -> `Mxfp4MoEMethod` with "
               "`activation_key=None`) disagrees with the transformers "
               "integration its spec read the FP8 activation off. So no green "
               "assertion claims the SM100 answer; the test asserts instead "
               "that the cell stays a VISIBLE counted gap. A model is in this "
               "set only while its Blackwell declaration is open and leaves "
               "it when someone rules it, so the corpus does not join this "
               "either",),
)

#: Corpus-walking test modules with no pin and no admission check: their
#: invariants are held by the test suite (C7) — the `regression` run, for the
#: modules marked `regression`.
CI_ONLY_CHECKS: Tuple[CiOnlyCheck, ...] = tuple(
    CiOnlyCheck(id="C7", title=f"corpus invariants of {module}",
                kind=CheckKind.CI_ONLY, owner_tests=(module,))
    for module in (
        'tests/backend/arch/eval/test_borrowed_weight_reads.py',
        'tests/backend/arch/eval/test_candidate_only_indexer.py',
        'tests/backend/arch/eval/test_copy_factors.py',
        'tests/backend/arch/eval/test_encode_phase.py',
        'tests/backend/arch/eval/test_encoder_item_formulas.py',
        'tests/backend/arch/eval/test_eval_capacity.py',
        'tests/backend/arch/eval/test_eval_instances.py',
        'tests/backend/arch/eval/test_eval_node.py',
        'tests/backend/arch/eval/test_eval_program.py',
        'tests/backend/arch/eval/test_eval_traffic.py',
        'tests/backend/arch/eval/test_no_per_head_cache_copy.py',
        'tests/backend/arch/eval/test_flops_closure.py',
        'tests/backend/arch/eval/test_formulas_payload_contract.py',
        'tests/backend/arch/eval/test_indexer_fusion.py',
        'tests/backend/arch/eval/test_moe_variant_conservation.py',
        'tests/backend/arch/eval/test_one_evaluation_per_step.py',
        'tests/backend/arch/eval/test_pr1_phase_binding.py',
        'tests/backend/arch/eval/test_program_moe.py',
        'tests/backend/arch/layout/test_eval_placement_diff.py',
        'tests/backend/arch/test_copy_keywords.py',
        'tests/backend/arch/test_declared_identities.py',
        'tests/backend/arch/test_ep_as_count.py',
        'tests/backend/arch/test_group_limit_switch.py',
        'tests/backend/arch/test_eval_axis_kinds.py',
        'tests/backend/arch/test_id_matching_shadow.py',
        'tests/backend/arch/test_loader.py',
        'tests/backend/arch/test_moe_unbound_defaults.py',
        'tests/backend/arch/test_media_processor.py',
        'tests/backend/arch/test_mtp_entry.py',
        'tests/backend/arch/test_mtp_logits.py',
        'tests/backend/arch/test_roles_registry.py',
        'tests/backend/arch/test_spec_id_compose.py',
        'tests/backend/arch/test_spec_ir_node_refs.py',
        'tests/backend/arch/test_spec_tree.py',
        'tests/backend/arch/spec_tree/test_node_bound.py',
        'tests/backend/arch/test_type_contracts.py',
        'tests/backend/inference/test_activation_peak_parallel.py',
        'tests/backend/inference/test_bank_read_rows.py',
        'tests/backend/inference/test_capacity_golden.py',
        'tests/backend/inference/test_decode_context_parallel.py',
        'tests/backend/inference/test_dispatch_zone_membership.py',
        'tests/backend/inference/test_encode_estimate.py',
        'tests/backend/inference/test_moe_dp_gather.py',
        'tests/backend/inference/test_moe_measurement_key.py',
        'tests/backend/inference/test_parallelism_combinations.py',
        'tests/backend/inference/test_pipeline_parallel_cache.py',
        'tests/backend/inference/test_pipeline_parallel_compute.py',
        'tests/backend/inference/test_pp_stage_coherence.py',
        'tests/backend/inference/test_row_roofline.py',
        'tests/backend/inference/test_scales_follow_weights.py',
        'tests/backend/playground/test_compose_corpus.py',
        'tests/backend/rendering/test_compute_graph_renderable.py',
        'tests/backend/rendering/test_render_reconciles.py',
        'tests/backend/rendering/test_rendering.py',
        'tests/backend/test_fingerprint.py',
        'tests/backend/spec_worker/test_prep_validate_reads_snapshot.py',
        'tests/backend/spec_worker/test_spec_overlay.py',
        'tests/frontend/test_model_id_literals.py',
        'tests/meta/test_audit_probes.py',
        'tests/meta/test_corpus_admit_verify.py',
        'tests/meta/test_scoreboard.py',
        'tests/kernel_bench/test_kernel_bench.py',
        'tests/kernel_bench/test_moe_boundary_audit.py',
    ))

_TOL = "a comparison tolerance, not a measured value"
_REQUEST = "the fixed request size the test measures at, not a measured value"
#: Module- and class-level numbers in corpus-walking test modules that are
#: not pinned data.
PARAMETERS: Tuple[Parameter, ...] = (
    Parameter("tests/backend/arch/test_media_processor.py", "RAW", _REQUEST),
    Parameter("tests/backend/inference/test_encode_estimate.py", "IMAGE", _REQUEST),
    Parameter("tests/backend/inference/test_encode_estimate.py", "KIMI_FITS",
              "the fixed mesh that keeps Kimi's oversized checkpoint fittable, not a measured value"),
    Parameter("tests/backend/inference/test_encode_estimate.py", "TestVideos.VIDEO", _REQUEST),
    Parameter("tests/backend/inference/test_sol_surface_golden.py", "REPORT_LIMIT",
              "how many mismatches a failure prints"),
    Parameter("tests/backend/inference/test_sol_surface_golden.py", "pytestmark",
              "the seconds one measurement of the whole corpus is given, a "
              "property of what the surface costs to take, not a model's value"),
    Parameter("tests/backend/inference/test_capacity_golden.py", "_REL_TOL", _TOL),
    Parameter("tests/backend/inference/test_capacity_golden.py", "_SEQS", _REQUEST),
    Parameter("tests/backend/inference/test_activation_peak_parallel.py", "SEQ", _REQUEST),
    Parameter("tests/backend/inference/test_parallelism_combinations.py", "_SEQ", _REQUEST),
    Parameter("tests/backend/inference/test_parallelism_combinations.py", "_MOE_COMBOS",
              "the parallel layouts the test sweeps"),
    Parameter("tests/backend/inference/test_parallelism_combinations.py", "_DENSE_COMBOS",
              "the parallel layouts the test sweeps"),
    Parameter("tests/backend/inference/test_pipeline_parallel_cache.py", "_PP",
              "the stage counts the partition is swept at, finest first"),
    Parameter("tests/backend/playground/test_compose_corpus.py", "L", _REQUEST),
    Parameter("tests/backend/arch/layout/test_eval_placement_diff.py", "CASES",
              "the parallel layouts the pinned placements are taken at"),
    Parameter("tests/backend/arch/test_ep_as_count.py", "TOY_CONFIG",
              "a synthetic spec's config fixture, not a model's value"),
    Parameter("tests/backend/arch/eval/test_one_evaluation_per_step.py", "EXPECTED_WALKS",
              "how many walks one step costs, a property of the engine, not of a model"),
    Parameter("tests/backend/arch/test_spec_tree.py", "TestComponentScopedOverride.SCOPED",
              "a hand-written override fixture"),
    Parameter("tests/backend/spec_worker/test_prep_validate_reads_snapshot.py", "SNAPSHOT",
              "a synthetic snapshot fixture"),
    Parameter("tests/kernel_bench/test_kernel_bench.py", "TestGpuSkuResolution.CASES",
              "GPU device names and the SKU each resolves to, not a model's value"),
    Parameter("tests/kernel_bench/test_plan_keys_are_the_estimates.py", "SEQ", _REQUEST),
    Parameter("tests/kernel_bench/test_plan_keys_are_the_estimates.py", "BATCH", _REQUEST),
    Parameter("tests/kernel_bench/test_kernel_bench_plan_pp.py", "SEQ", _REQUEST),
    Parameter("tests/kernel_bench/test_kernel_bench_plan_pp.py", "BATCH", _REQUEST),
    Parameter("tests/kernel_bench/test_kernel_bench_plan_pp.py", "SMALL_LAYERS",
              "the chosen snapshot's layer count, restated so the expected "
              "stage split is written against `stage_layer_ranges` and not "
              "against whatever the spec happens to say"),
    Parameter("tests/backend/inference/test_prefill_chunk_sol.py", "CHUNK",
              _REQUEST),
    Parameter("tests/backend/inference/test_prefill_chunk_sol.py", "ENDS",
              "the request lengths whose chunked-prefill closure is checked"),
    Parameter("tests/kernel_bench/test_kernel_bench.py",
              "TestSparseLatentAttention.SPEC",
              "the DSA deployment dimensions the identity is projected at"),
    Parameter("tests/kernel_bench/test_kernel_bench.py",
              "TestChunkedPrefillPhysicalRekey.CHUNK",
              _REQUEST),
)

#: Numbers that are a pin's data, read from the pin's file.
PIN_DATA: Tuple[Tuple[str, str, str], ...] = (
    ("tests/backend/arch/layout/test_expert_weight_checkpoint.py",
     "EXPECTED_CHECKPOINT_CORPUS", "P5"),
)

#: Corpus-walking modules that build snapshot identifiers from strings, and
#: where their keys come from: "corpus" (enumerated from the snapshots; a pin
#: or a check claims the module) or "constant" (a curated set's own list).
DYNAMIC_KEY_SOURCES: Dict[str, str] = {
    'tests/backend/arch/eval/test_borrowed_weight_reads.py': 'constant',
    'tests/backend/arch/test_arch_view_aux_inputs.py': 'corpus',
    'tests/backend/arch/eval/test_eval_capacity.py': 'corpus',
    'tests/backend/arch/eval/test_eval_instances.py': 'constant',
    'tests/backend/arch/test_moe_unbound_defaults.py': 'constant',
    'tests/backend/arch/eval/test_eval_node.py': 'corpus',
    'tests/backend/arch/eval/test_eval_program.py': 'corpus',
    'tests/backend/arch/eval/test_eval_traffic.py': 'constant',
    'tests/backend/arch/eval/test_flops_closure.py': 'constant',
    'tests/backend/arch/eval/test_formulas_payload_contract.py': 'corpus',
    'tests/backend/arch/eval/test_indexer_fusion.py': 'corpus',
    'tests/backend/arch/eval/test_one_evaluation_per_step.py': 'corpus',
    'tests/backend/arch/eval/test_stack_attachment.py': 'corpus',
    'tests/backend/arch/layout/test_eval_placement_diff.py': 'constant',
    'tests/backend/arch/test_declared_identities.py': 'corpus',
    'tests/backend/arch/test_eval_axis_kinds.py': 'corpus',
    'tests/backend/arch/test_id_matching_shadow.py': 'corpus',
    'tests/backend/arch/test_index_signature_coverage.py': 'corpus',
    'tests/backend/arch/test_loader.py': 'corpus',
    'tests/backend/arch/test_mtp_logits.py': 'corpus',
    'tests/backend/arch/test_spec_id_compose.py': 'corpus',
    'tests/backend/arch/test_spec_ir_node_refs.py': 'constant',
    'tests/backend/arch/test_spec_tree.py': 'corpus',
    'tests/backend/arch/validate/test_validator_golden.py': 'constant',
    'tests/backend/inference/test_activation_peak_parallel.py': 'corpus',
    'tests/backend/inference/test_capacity_golden.py': 'constant',
    'tests/backend/inference/test_decode_context_parallel.py': 'constant',
    'tests/backend/inference/test_dispatch_zone_membership.py': 'constant',
    'tests/backend/inference/test_moe_dp_gather.py': 'corpus',
    'tests/backend/inference/test_parallelism_combinations.py': 'constant',
    'tests/backend/inference/test_pipeline_parallel_cache.py': 'constant',
    'tests/backend/inference/test_pipeline_parallel_compute.py': 'constant',
    'tests/backend/inference/test_pp_stage_coherence.py': 'constant',
    'tests/backend/inference/test_row_roofline.py': 'corpus',
    'tests/backend/playground/test_compose_corpus.py': 'constant',
    'tests/backend/rendering/test_compute_graph_renderable.py': 'constant',
    'tests/backend/rendering/test_rendering.py': 'corpus',
    'tests/frontend/test_model_id_literals.py': 'corpus',
    'tests/meta/test_audit_probes.py': 'corpus',
}


def find_curated(module: str, name: str) -> Optional[CuratedSet]:
    for entry in CURATED_OUT_OF_SCOPE:
        if entry.module == module and entry.name == name:
            return entry
    return None
