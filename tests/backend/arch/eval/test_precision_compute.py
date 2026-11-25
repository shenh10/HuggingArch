"""What a matmul's own format costs: the column it runs at, the conversion
it pays when the edge does not carry what it eats, and the launch above it
that can write that format instead.

Domain facts only — the design's §2.3 invariants, §3.7 comparison table and
the hardware fallback a framework actually runs — read off real checkpoints
and the real rewrite, not off a fixture of the engine's own shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dataclasses import replace
from functools import lru_cache

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.activation_formats import ACTIVATION_FORMATS
from backend.arch.eval.precision import (AS_SHIPPED, PrecisionError,
                                         PrecisionProfile)
from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.traffic import ElementCopy
from backend.arch.eval.transforms import DECODE, Deployment, Phase
from backend.inference.framework_bindings import FRAMEWORK_BINDINGS
from backend.inference.framework_bindings import (PrecisionLaunch,
                                                  ServingDeployment,
                                                  get_binding)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, shared_program, shared_walk


def _reps(name: str, path: str, profile=None):
    """One leaf's resident formats, by display path."""
    analysis = shared_walk(Deployment(shared_program(name),
                          precision=profile).at(batch=1,
                                                input_tokens=16))
    for uid, reps in analysis.formats.values.items():
        node = analysis.program.by_uid(uid)
        if node is not None and node.display_path == path:
            return analysis.resolved_precision, node, reps
    raise AssertionError(f"{name} has no leaf at {path!r}")


# -- the column a matmul runs at (§3.4) -------------------------------------

def test_an_fp8_checkpoints_matmul_runs_at_the_fp8_column():
    """DeepSeek-V3's config declares `activation_scheme: dynamic`, so its FP8
    kernel quantises the operand and multiplies in FP8 — not at the BF16 its
    input edge carries."""
    precision, node, reps = _reps("deepseek-ai__DeepSeek-V3",
                                  "self_attn.q_b_proj")
    assert precision.edge_dtype(node.role, "") == "BF16"
    assert precision.compute_dtype(node.role, "", reps) == "F8_E4M3"


def test_a_weight_only_checkpoints_matmul_runs_at_its_edge():
    """gpt-oss's MXFP4 experts state that their kernel takes the activation
    as it arrives (`matmul_ogs` on bf16), so the column is the edge's."""
    precision, node, reps = _reps("openai__gpt-oss-20b",
                                  "mlp.experts.gate_up_proj")
    assert precision.required_format(node.role, "", reps) is None
    assert precision.compute_dtype(node.role, "", reps) == "BF16"


def test_the_panels_compute_column_asks_the_matmuls_not_the_stored_tensor():
    """DeepSeek-V3's decode attention runs its projections on FP8 weights
    and its absorbed per-head GEMMs on the BF16 W_UK / W_UV split out of the
    FP8 `kv_b_proj` at load. The panel's compute column is asked of each
    matmul as it holds its weight (`traffic.matmul_operands`): the stored
    `kv_b_proj` is no matmul, and the per-head GEMMs answer BF16 — so the
    attention row names both columns its matmuls run at, on both sides
    (the split is the framework's, whatever the profile)."""
    program = Phase(DECODE).substitute(
        shared_program("deepseek-ai__DeepSeek-V3"))
    # The module's shared walk: this test only reads it.
    analysis = shared_walk(Deployment(program).at(batch=1, input_tokens=16))
    precision = analysis.resolved_precision
    operands = traffic_mod.matmul_operands(
        analysis.program, analysis.formats.values, precision)
    by_path = {analysis.program.by_uid(uid).display_path: (uid, reps)
               for uid, reps in operands.items()}
    assert "self_attn.kv_b_proj" not in by_path
    uid, reps = by_path["self_attn.q_absorbed_nope"]
    node = analysis.program.by_uid(uid)
    assert precision.compute_dtype(node.role, "", reps) == "BF16"
    (row,) = [r for r in analysis.precision_effective()["effective"]
              if (r["category"], r["selector"]) == ("compute", "attention")]
    assert row["effective"] == row["as_shipped"] == "BF16 + FP8 (E4M3)"


def test_a_quantised_activation_does_not_by_itself_assert_a_narrow_mma():
    """The hardware MMA family is a launch fact, not a width inference.

    Humming's Hopper WGMMA path accepts an MXFP4 weight with a block-FP8
    activation and dequantises the weight in software (vLLM
    `fused_humming_moe.py:323-325`). An explicitly declared launch with that
    shape therefore needs the FP8 column alone — even though a narrower
    weight is resident and the activation is quantised.
    """
    precision, node, reps = _reps("openai__gpt-oss-20b",
                                  "mlp.experts.gate_up_proj")
    scheme = reps[0].resident.scheme.name
    humming = precision.with_launches([PrecisionLaunch(
        schemes=(scheme,), requested_activation_formats=(),
        activation_format="fp8_token_group128", compute_dtype="F8_E4M3",
        mma_dtypes=("F8_E4M3",))])
    assert humming.required_format(
        node.role, "", reps) is ACTIVATION_FORMATS["fp8_token_group128"]
    assert humming.compute_dtype(node.role, "", reps) == "F8_E4M3"
    assert humming.mma_dtypes(node.role, "", reps) == ("F8_E4M3",)


def test_mma_family_ladders_are_an_explicit_launch_semantics():
    """The same family tuple means two different launches unless declared.

    A default launch requires F8 and MXFP4 together.  A launch carrying the
    audited `promotion_ladder` fact is selected from a non-empty intersection
    and carries the full ladder to access; pricing, not the selector, then
    names the widest supported rung.
    """
    precision, node, reps = _reps("openai__gpt-oss-20b",
                                  "mlp.experts.gate_up_proj")
    scheme = reps[0].resident.scheme.name

    def launch(*, ladder: bool):
        return PrecisionLaunch(
            schemes=(scheme,),
            requested_activation_formats=(),
            activation_format="fp8_token_group128_ue8m0",
            compute_dtype="F8_E4M3",
            mma_dtypes=("F8_E4M3", "MXFP4"),
            promotion_ladder=ladder)

    default = precision.with_launches(
        [launch(ladder=False)], peak_families=("fp8", "bf16"))
    assert default._launch_for(node.role, "", reps) is None
    assert default.launch_gap(node.role, "", reps) == (
        "missing_card_mma:1:MXFP4")
    assert default.launch_promotion_ladder(node.role, "", reps) is False

    ladder = precision.with_launches(
        [launch(ladder=True)], peak_families=("fp8", "bf16"))
    assert ladder._launch_for(node.role, "", reps) is not None
    assert ladder.mma_dtypes(node.role, "", reps) == ("F8_E4M3", "MXFP4")
    assert ladder.launch_promotion_ladder(node.role, "", reps) is True


# -- the conversion (§3.7) --------------------------------------------------

def test_a_plain_edge_into_a_quantised_operand_is_a_quantise():
    precision, node, reps = _reps("deepseek-ai__DeepSeek-V3",
                                  "self_attn.q_b_proj")
    conversion = precision.conversion(node.role, "", reps)
    assert conversion.kind == "quantise"
    assert conversion.target is ACTIVATION_FORMATS["fp8_token_group128"]
    # The narrow tensor and its scales — priced by the one width law of an
    # activation record: 8 rows of 128 FP8 values, one F32 scale per row.
    from backend.arch.eval.traffic import _format_width
    assert 1024 * _format_width(conversion.target, 1024, 128) == (
        pytest.approx(1024 * 1 + 8 * 4))


def test_an_edge_already_in_the_wanted_format_costs_nothing():
    """What a launch above it wrote in exactly that format needs no change —
    the consumer half of invariant 1."""
    precision, node, reps = _reps("deepseek-ai__DeepSeek-V3",
                                  "self_attn.q_b_proj")
    wanted = precision.required_format(node.role, "", reps)
    assert precision.conversion(node.role, "", reps, edge=wanted) is None


def test_another_quantised_format_on_the_edge_is_a_requantise():
    precision, node, reps = _reps("deepseek-ai__DeepSeek-V3",
                                  "self_attn.q_b_proj")
    other = ACTIVATION_FORMATS["fp8_per_token"]
    conversion = precision.conversion(node.role, "", reps, edge=other)
    assert conversion.kind == "requantise"
    assert conversion.target.name == "fp8_token_group128"


def test_a_quantised_edge_into_a_kernel_that_takes_it_as_it_arrives_is_a_dequantise():
    """gpt-oss's W4A16 expert reads the edge as it is, so a quantised edge is
    dequantised to the model's own dtype first."""
    precision, node, reps = _reps("openai__gpt-oss-20b",
                                  "mlp.experts.gate_up_proj")
    conversion = precision.conversion(
        node.role, "", reps, edge=ACTIVATION_FORMATS["fp8_token_group128"])
    assert conversion.kind == "dequantise"
    assert conversion.target is None
    assert conversion.target_dtype == "BF16"


# -- invariant 3: the launch above may write the format instead -------------

def _conversion_rows(name: str, *, framework, phase="prefill"):
    """`(paid, absorbed)` display paths off the estimate's own access rows.

    A leaf whose kernel needs its operand in another format pays the
    conversion itself (`convert_bytes`) where nothing materialised it, and
    is paid FOR by the cast standing on its operand edge where
    `materialize_casts` grafted one — one format change, one price, the
    payer a node either way. What remains for `absorbed` is a producer that
    wrote the format its consumer reads: a fused launch that took the cast
    in as a member (`quantise_activation` in its plan's pattern)."""
    result = analyze(
        resolve_snapshot(name), parallel=ParallelConfig(), batch_size=1,
        avg_input_tokens=256, avg_output_tokens=0, phase=phase,
        framework_binding=FRAMEWORK_BINDINGS[framework] if framework else None, breakdown=True)
    precision = result.analysis.resolved_precision
    program = result.program
    walked = result.analysis.formats.values
    origins = dict(result.rewrite.origins or {})
    paid, absorbed = [], []
    for uid, row in result.access.values.items():
        node = program.by_uid(uid)
        if node is None or node.children:
            continue
        reps = tuple(rep for member in ((*(origins.get(uid) or ()), uid))
                     for rep in walked.get(member, ()))
        if precision.conversion(node.role, program.stack_of(node),
                                reps) is None:
            continue
        # A cast still standing pays; one a fused launch took in as a member
        # is the launch's write, and the edge's reader is absorbed. A reader
        # of copies of a cast's output (a routed expert reading its
        # dispatch's rows of the tokens quantised before it) is paid for by
        # that cast: the copies hand its record on (`ElementCopy`).
        def made_by_cast(src):
            while src is not None and src.provenance != "cast":
                copy = ElementCopy.of(src, result.full_shapes)
                if copy is None or len(copy.inputs) != 1:
                    return False
                src = src.inputs_obj[copy.inputs[0]].src
            return (src is not None
                    and src.uid not in (result.rewrite.dropped or ()))

        fed_by_cast = any(made_by_cast(ref.src)
                          for ref in (node.inputs_obj or ()))
        (paid if row.convert_bytes_per_rank or fed_by_cast
         else absorbed).append(node.display_path)
    return paid, absorbed


def test_the_conversions_price_is_paid_by_a_node_on_the_edge():
    """One format change, one price, paid by a node standing on the edge:
    the grafted `quantise_activation` (`materialize_casts`), with or without
    a framework binding.

    Under vLLM the activation epilogue (`silu_and_mul`) names the cast as a
    member, so `down_proj`'s conversion is in that launch: its edge reads
    the narrow tensor and nothing pays apart. With no framework there is no
    fusion, and the cast standing on the edge is the payer.
    """
    paid, absorbed = _conversion_rows("deepseek-ai__DeepSeek-V3",
                                      framework="vllm")
    assert "mlp.down_proj" in absorbed
    assert "mlp.down_proj" not in paid
    unfused_paid, unfused_absorbed = _conversion_rows(
        "deepseek-ai__DeepSeek-V3", framework=None)
    assert unfused_absorbed == []
    assert "mlp.down_proj" in unfused_paid


def test_a_tensor_two_kernels_want_differently_is_written_once_as_it_is():
    """A MoE layer's norm feeds the router (which eats bf16) and the experts
    (which eat FP8): the edge keeps its format and the experts' operand is
    quantised apart — once per token, before the dispatch copies the rows to
    the experts — which is the second half of invariant 3."""
    paid, absorbed = _conversion_rows("deepseek-ai__DeepSeek-V3",
                                      framework="vllm")
    assert "mlp.experts.gate_proj" in paid
    assert "mlp.experts.gate_proj" not in absorbed


# -- compute pins (§3.7) ----------------------------------------------------

def _pin(body):
    return PrecisionProfile.canonicalize({"compute": [
        {"select": {"roles": ["ffn_dense"]}, "format": body}]})


def test_a_pin_below_two_bytes_names_the_format_its_kernel_eats():
    with pytest.raises(PrecisionError, match="activation_format"):
        _pin({"dtype": "FP4_E2M1"})


def test_a_pins_format_must_be_the_dtype_it_pins():
    with pytest.raises(PrecisionError, match="not the pinned"):
        _pin({"dtype": "FP4_E2M1",
              "activation_format": "fp8_token_group128"})


def test_a_pin_needs_a_dtype_the_gpu_sheet_has_a_column_for():
    with pytest.raises(PrecisionError, match="no GPU peak column"):
        _pin({"dtype": "F8_E8M0"})


def test_a_pin_does_not_convert_the_weights_it_streams():
    """Pinning BF16 weights to FP8 is refused: a pin changes the column, not
    the checkpoint (§3.7). The fix it names is a weight rule."""
    profile = PrecisionProfile.canonicalize({"compute": [
        {"select": {"roles": ["attention"]},
         "format": {"dtype": "F8_E4M3",
                    "activation_format": "fp8_token_group128"}}]})
    precision, node, reps = _reps("Qwen__Qwen2.5-7B", "self_attn.q_proj",
                                  profile)
    with pytest.raises(PrecisionError, match="does not convert weights"):
        precision.compute_dtype(node.role, "", reps)


def test_a_pin_the_weights_are_narrower_than_is_the_w4a16_case():
    """FP4 weights pinned to BF16: legal, the kernel dequantises them."""
    profile = PrecisionProfile.canonicalize({"compute": [
        {"select": {"roles": ["ffn_moe"]}, "format": {"dtype": "BF16"}}]})
    precision, node, reps = _reps("openai__gpt-oss-20b",
                                  "mlp.experts.gate_up_proj", profile)
    assert precision.compute_dtype(node.role, "", reps) == "BF16"


# -- the card's own fallback (§3.8, and what vLLM runs) ---------------------

@lru_cache(maxsize=None)
def _estimate(name: str, sku: str, profile=AS_SHIPPED):
    from tests.snapshots import snapshot_config
    from backend.inference.capacity import estimate_capacity
    return estimate_capacity(
        config=snapshot_config(name), resolved=resolve_snapshot(name),
        model_id=name.replace("__", "/"), gpu_sku=sku, num_gpus=16,
        # Two stages, so DeepSeek-V3 fits an 80 GB card: a deployment that
        # does not runs no step and publishes no rows.
        parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                tp_moe_expert=1, ep_moe=8, pp=2),
        avg_input_tokens=1024, avg_output_tokens=128, precision=profile,
        selection_mode="theoretical_only")


def test_a_card_without_the_mma_runs_the_checkpoints_matmuls_weight_only():
    """An FP8 checkpoint on a card with no FP8 column is what vLLM serves
    through FP8 Marlin: weights still FP8, activation not quantised, matmul
    at the activation's dtype. So the estimate stands, at the bf16 column,
    and no row pays a conversion."""
    rows = _estimate("deepseek-ai__DeepSeek-V3", "A100-80G-SXM")["per_op_timing"]
    computing = [r for r in rows if r.get("flops_per_rank")]
    assert computing, "the step has matmuls"
    assert {r["act_dtype"] for r in computing} == {"BF16"}
    assert not [r for r in rows if r.get("convert_bytes_per_rank")]
    on_hopper = [r for r in _estimate("deepseek-ai__DeepSeek-V3",
                                      "H100-SXM")["per_op_timing"]
                 if r.get("flops_per_rank")]
    assert "F8_E4M3" in {r["act_dtype"] for r in on_hopper}


def test_a_width_the_deployment_chose_is_refused_where_the_card_has_no_column():
    """The other half of §3.8: NVFP4 W4A4 on a card with no fp4 column is a
    hardware fact, and the point says so instead of borrowing a column."""
    with pytest.raises(ValueError, match="no FP4 peak throughput"):
        _estimate("Qwen__Qwen2.5-7B", "H100-SXM",
                  PrecisionProfile.from_request(
                      {"preset": "w_nvfp4_w4a4_kv_fp8"}))


# -- the measured side takes each leaf's own environment (§4.5) -------------

def test_a_measured_row_is_keyed_by_its_own_leafs_dtypes():
    """An FP8 checkpoint's BF16 norms are looked up as BF16 and its FP8 GEMMs
    as FP8 — the environment each leaf ran in, which is the one the export
    side writes."""
    import json
    rows = _estimate("deepseek-ai__DeepSeek-V3", "H100-SXM")["per_op_timing"]
    envs = {r["aggregation_id"]: json.loads(r["dtype_env_json"]) for r in rows}
    gemm = next(env for sid, env in envs.items()
                if sid.endswith("self_attn.q_b_proj"))
    assert (gemm["weight"], gemm["activation"]) == ("fp8_e4m3", "fp8_e4m3")
    norm = next(env for sid, env in envs.items()
                if sid.endswith("input_layernorm"))
    assert (norm["weight"], norm["activation"]) == ("bf16", "bf16")

def _cast_records(name: str, *, framework, phase="prefill"):
    """Every leaf's `(display_path, Traffic)` where a cast survived, and the
    leaves where a producer absorbed one instead."""
    from backend.arch.eval import traffic as traffic_layer

    result = analyze(
        resolve_snapshot(name), parallel=ParallelConfig(), batch_size=1,
        avg_input_tokens=256, avg_output_tokens=0, phase=phase,
        framework_binding=FRAMEWORK_BINDINGS[framework] if framework else None,
        breakdown=True)
    program = result.program
    layer = traffic_layer.derive(result.analysis, program=program,
                                 rewrite=result.rewrite)
    casts, absorbed = [], []
    walked = result.analysis.formats.values
    origins = dict(result.rewrite.origins or {})
    precision = result.analysis.resolved_precision
    for uid, flow in layer.values.items():
        node = program.by_uid(uid)
        if node is None or node.children:
            continue
        reps = tuple(rep for member in ((*(origins.get(uid) or ()), uid))
                     for rep in walked.get(member, ()))
        if precision.conversion(node.role, program.stack_of(node),
                                reps) is None:
            continue
        (casts if flow.cast is not None else absorbed).append(
            (node.display_path, flow))
    return casts, absorbed


def test_a_surviving_conversion_is_published_as_the_launch_it_is():
    """One format change nobody absorbed is one launch, and the leaf says so.

    `convert_bytes` priced it all along, but as a number inside the
    CONSUMER's traffic — so every reader downstream saw one node where the
    hardware runs two, and a measured row billed the GEMM for both. That is
    why 6,196 rows on this family read 1.7-2.7x the card's bandwidth and
    were refused by the SOL gate. The record is the same decision, made
    once, in a form that can be taken apart — and the serving graph grafts
    the launch it records as a `quantise_activation` node
    (`inference.materialize_casts`, covered in
    `tests/backend/inference/test_materialize_cast.py`). Read here off the
    walk, where the decision is published.
    """
    from backend.arch.eval import traffic as traffic_layer

    result = analyze(
        resolve_snapshot("deepseek-ai__DeepSeek-V3"),
        parallel=ParallelConfig(), batch_size=1,
        avg_input_tokens=256, avg_output_tokens=0, phase="prefill",
        framework_binding=None, breakdown=True)
    layer = traffic_layer.derive(result.analysis)
    casts = []
    for uid, flow in layer.values.items():
        node = result.analysis.program.by_uid(uid)
        if node is not None and not node.children and flow.cast is not None:
            casts.append((node.display_path, flow))
    assert casts, "an fp8 block-scaled checkpoint has conversions to pay"
    for path, flow in casts:
        cast = flow.cast
        assert cast.kind in ("quantise", "requantise", "dequantise"), path
        # The number and the record cannot drift: the record IS the number.
        assert cast.convert_bytes == flow.convert_bytes, path
        # And the split is the two launches: what the cast reads is the
        # operand as the edge holds it, which is on the consumer's own
        # activation read because the consumer is the only node there is.
        assert cast.wide_read > 0, path
        assert cast.narrow_write == cast.narrow_read, path


def test_an_absorbed_conversion_publishes_no_launch():
    """Nothing to launch: the producer wrote the format, so the consumer
    reads it narrow and pays nothing. The record's absence is the same fact
    `convert_bytes == 0` states, and the two are decided together."""
    casts, absorbed = _cast_records("openai__gpt-oss-20b", framework="vllm")
    for path, flow in absorbed:
        assert flow.convert_bytes == 0, path
        assert flow.cast is None, path
    for path, flow in casts:
        assert flow.convert_bytes > 0, path


# -- the width a dispatched kernel requires of an edge ----------------------

def _dsv4_precision(framework: str | None):
    """DeepSeek-V4-Flash's resolved precision as the serving path hands it,
    with and without a deployed framework."""
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    from backend.inference.pipeline import analyze

    binding = (None if framework is None else get_binding(framework).deployed(
        ServingDeployment.at(gpu_sku="B200", activation_dtype="BF16")))
    result = analyze(resolve_snapshot("deepseek-ai__DeepSeek-V4-Flash"),
                     parallel=ParallelConfig(), framework_binding=binding,
                     batch_size=1, avg_input_tokens=128, avg_output_tokens=1,
                     phase="prefill", breakdown=False)
    return result.analysis.resolved_precision


def test_an_undeclared_edge_keeps_the_width_the_checkpoint_implies():
    """The assumption every edge rests on, stated where it is read.

    With no framework deployed no kernel's requirement is in hand, so the
    router's logits are the model's own dtype — which is also what a BF16
    gate weight times a BF16 row produces.
    """
    precision = _dsv4_precision(None)
    assert precision.required_edge_dtypes == ()
    assert precision.edge_dtype("router", "") == "BF16"


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_a_width_a_dispatched_kernel_requires_is_what_the_walk_writes(framework):
    """The top-k kernel each framework dispatches to for DeepSeek-V4 exists
    only at fp32, so vLLM builds the gate `GateLinear(out_dtype=torch.float32)`
    and sglang's `MoEGate.forward` writes fp32 on CUDA to supply it. Both
    bindings declare that requirement on the `router` role, and the walk
    writes the logits edge there."""
    precision = _dsv4_precision(framework)
    assert precision.edge_dtype("router", "") == "F32"
    # Declared per role, not for the whole walk: nothing else moved.
    assert precision.edge_dtype(None, "") == "BF16"
    assert precision.edge_dtype("ffn_moe", "") == "BF16"


def test_a_required_width_does_not_move_the_weight_or_the_column():
    """What the kernel requires is the OUTPUT. The gate's weight stays BF16
    — a stored BF16 weight is not followed into a framework's load-time
    upcast — and so the multiply still runs at the BF16 column, which is
    `traffic.weight_promotion`'s subject and not this one."""
    precision, node, reps = _reps("deepseek-ai__DeepSeek-V4-Flash",
                                  "ffn.gate.weight")
    assert node.role == "router"
    assert {rep.resident.logical_dtype for rep in reps} == {"BF16"}
    required = precision.with_required_edge_dtypes([(("router",), "F32")])
    assert required.edge_dtype("router", "") == "F32"
    assert {rep.resident.logical_dtype for rep in reps} == {"BF16"}
    # The column follows the operand the GEMM streams, which is the BF16
    # hidden row the walk hands it.
    assert required.compute_dtype("router", "", reps, operand="BF16") == "BF16"


def test_a_role_no_entry_names_is_not_walked_up_a_refines_chain():
    """`router_hash_table` refines `router`, and its tensor is a looked-up
    table rather than a written product. A width declared for the parent is
    not inherited onto it: that would state a requirement nobody audited."""
    precision = _dsv4_precision("vllm")
    assert precision.edge_dtype("router", "") == "F32"
    assert precision.edge_dtype("router_hash_table", "") == "BF16"

# -- the precision panel's compute rows -------------------------------------

def _compute_rows(name: str, profile=None):
    analysis = shared_walk(Deployment(shared_program(name),
                                      precision=profile).at(
        batch=1, input_tokens=16))
    return {row["selector"]: row
            for row in analysis.precision_effective()["effective"]
            if row["category"] == "compute"}


def _profile(body):
    return PrecisionProfile.canonicalize(body)


def test_a_checkpoint_promotion_is_not_a_profile_change():
    """MiniMax ships its router gate in F32, so that matmul computes in F32
    as shipped. Both columns read a promotion table of their own side's
    formats, so under the default profile — where saved and resident are the
    same — the router row says FP32 on both sides and nothing changed."""
    rows = _compute_rows("MiniMaxAI__MiniMax-M2.7")
    assert (rows["router"]["as_shipped"], rows["router"]["effective"]) == (
        "FP32", "FP32")
    assert not any(row["changed"] for row in rows.values()), rows


def test_a_profile_that_widens_a_weight_shows_the_widening():
    """Llama-3.2-1B ships BF16, so its matmuls compute in BF16 as shipped.
    A profile holding every weight in F32 promotes those products to F32 —
    the as-shipped column answers for the FILE (saved formats, the shipped
    edge), so the widening is the delta rather than being folded into both
    columns."""
    rows = _compute_rows("meta-llama__Llama-3.2-1B", _profile(
        {"weights": [{"select": {"roles": "all"},
                      "format": {"dtype": "fp32"}}]}))
    assert rows, "a promoted matmul has a compute row"
    for selector, row in rows.items():
        assert (row["as_shipped"], row["effective"], row["changed"]) == (
            "BF16", "FP32", True), (selector, row)


def test_the_as_shipped_compute_column_answers_for_the_file():
    """gemma-2 stores F32 and both frameworks load it BF16. The whole
    panel's as-shipped column answers for the FILE — the weight rows read
    `rep.saved`, the activation row reads `shipped_model_dtype` — so the
    compute column answers for the file too, and the framework's own
    narrowing is the delta on all three. One column, one meaning: reading
    the file here and the served width there would be the same panel giving
    two answers to "as shipped"."""
    analysis = shared_walk(Deployment(shared_program("google__gemma-2-9b")).at(
        batch=1, input_tokens=16))
    rows = {(r["category"], r["selector"]): r
            for r in analysis.precision_effective()["effective"]}
    edge = rows[("activations", "all")]
    assert (edge["as_shipped"], edge["effective"]) == ("FP32", "BF16")
    compute = {sel: r for (cat, sel), r in rows.items() if cat == "compute"}
    assert compute, "an F32 checkpoint's matmuls have a compute row"
    for selector, row in compute.items():
        assert (row["as_shipped"], row["effective"]) == (
            edge["as_shipped"], edge["effective"]), (selector, row)
    # And the table that PRICES the walk is untouched by this: the served
    # narrowing is in the resident format, so nothing is promoted.
    assert not analysis.weight_promotions


def test_a_profile_that_quantises_a_promoted_weight_shows_the_narrowing():
    """The other direction, and the one a single resolution got wrong: an
    FP8 weight profile takes MiniMax's F32 router gate down to the FP8 its
    scheme quantises the activation to. As shipped that gate still multiplies
    in F32, because the FILE holds it in F32."""
    rows = _compute_rows("MiniMaxAI__MiniMax-M2.7", _profile(
        {"weights": [{"select": {"roles": "all"},
                      "format": {"scheme": "fp8"}}]}))
    assert (rows["router"]["as_shipped"], rows["router"]["effective"],
            rows["router"]["changed"]) == ("FP32", "FP8 (E4M3)", True)


def test_the_as_shipped_column_keeps_the_launches_selection_environment():
    """An as-shipped resolution carries the launches AND the card that chose
    them. `_launch_is_supported` reads a resolution with no peak families as
    a cardless fixture and admits every candidate, so a resolution that kept
    the launches but dropped the card answers the as-shipped column for a
    launch this card cannot run. The second opinion is the priced row's own
    dtype: under no profile at all, the file's answer and the deployment's
    answer are the same answer."""
    program = shared_program("deepseek-ai__DeepSeek-V4-Flash")
    deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                      activation_dtype="BF16")
    binding = get_binding("vllm").deployed(deployment)
    walk = shared_walk(Deployment(program).at(batch=1, input_tokens=16))
    served = replace(walk, precision=walk.resolved_precision.with_launches(
        binding.precision_launches_for(program),
        peak_families=deployment.peak_families,
        unavailable_launches=binding.unavailable_precision_launches_for(
            program)))
    assert served.resolved_precision.launches, "the card took a launch"
    assert served.shipped_precision.launch_peak_families == (
        deployment.peak_families), "the as-shipped side kept the card"
    rows = {row["selector"]: row
            for row in served.precision_effective()["effective"]
            if row["category"] == "compute"}
    assert rows, "the launches give the matmuls a compute row"
    for selector, row in rows.items():
        assert (row["as_shipped"], row["changed"]) == (
            row["effective"], False), (selector, row)
