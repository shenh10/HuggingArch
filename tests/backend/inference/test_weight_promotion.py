"""A matmul whose weight is held wider than its activation COMPUTES in the
weight's dtype — and stores at the width its edge declares.

The type promotion of a mixed BF16 x F32 product
(`traffic.weight_promotion`): DeepSeek-V4 stores its hyper-connection mixing
matrices `hc_*_fn` F32, so that multiply runs on the F32 MMA. Before, the
row was timed at the BF16 peak and keyed as a BF16 GEMM, which is why the
measured corpus had an fp32 oracle against a bf16 cell and the launch gate
refused it every pass.

The promotion stops at the arithmetic. Both frameworks' kernels narrow in
registers as they store — vLLM `triton_ihc.py:297-299` and sglang
`hy4_ihc.py:325` allocate the iHC post's output at the sub-block operand's
dtype — so the product keeps its edge's declared width, and the graph needs
no cast node for a narrowing no kernel launches. Writing it wide instead
sent F32 down the whole residual stream after it and cost Hy4's prefill 14%
of a step in tensors the hardware never writes.

So the two halves are pinned apart here: the compute dtype moves (the peak
column, the measurement key's activation slot), the written width does not.

The counter-case: GLM-5.3-Flash stores its `hc_*_fn` BF16, and the walk
keeps it BF16 even though vLLM loads it into an F32 parameter — the
checkpoint's declaration decides, not a framework's load-time choice.
"""
from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.precision import ResolvedPrecision
from backend.arch.eval.program import Program
from backend.arch.precision_profile import AS_SHIPPED
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.framework_bindings import (
    ServingDeployment, get_binding)
from backend.inference.gpu_specs import get_gpu
from backend.inference.parallel_sweep import config_for_degree
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot

GPU = "H100-SXM"


@functools.cache
def _walk(model: str, framework: str = "vllm"):
    # Not a fixture: a failing test's traceback prints its arguments, and
    # rendering a whole walk outlives any timeout.
    #
    # The binding is DEPLOYED, as serving deploys it: the kernels a
    # framework runs only on some cards or some activation dtypes
    # (`when:`) are answered for this deployment, so the plans that bear on
    # a promoted product — Hy4's iHC post — are the ones a served step
    # runs. An undeployed binding silently drops every conditioned plan.
    resolved = resolve_snapshot(model)
    binding = get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku=GPU,
        activation_dtype=ResolvedPrecision(
            AS_SHIPPED, Program.of(resolved)).model_dtype))
    return analyze(resolved, parallel=config_for_degree("tp_tp", 1),
                   framework_binding=binding, batch_size=1,
                   avg_input_tokens=256, avg_output_tokens=0, phase="prefill",
                   breakdown=False, gpu=get_gpu(GPU))


def _width(row, side: str) -> float:
    """Bytes per element the row's one activation `side` tensor was billed
    at (a parameter it reads is a weight term, not an activation read)."""
    shape = row["shape_in"] if side == "in" else row["shape_out"]
    if shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    elements = math.prod(shape) * int(row["count_per_step"])
    return float(row[f"act_{side}_bytes_per_rank"]) / elements


def _facts(walk, row) -> dict:
    """The row's plain facts. Plain values only: an assertion over a row
    itself makes pytest render it whole, laws and all."""
    return {"weight_dtype": row.get("weight_dtype"),
            "output_dtype": row.get("output_dtype"),
            "act_dtype": row.get("act_dtype"),
            "in_width": _width(row, "in"), "out_width": _width(row, "out"),
            "act_in": float(row["act_in_bytes_per_rank"]),
            "act_out": float(row["act_out_bytes_per_rank"]),
            "flops": float(row["flops_per_rank"]),
            "t_compute": float(row["t_compute_s"]),
            "env": json.loads(RowIdentity(walk).dtype_env(row))}


def _rows(model: str, suffix: str, op: Optional[str] = None,
          framework: str = "vllm"):
    """The rows whose id ends `suffix`, of op `op` or of whatever absorbed
    it when `op` is None.

    A deployed binding fuses: DeepSeek-V4's mix GEMM is a `mhc_pre_gemm`
    launch and not a `linear_external_weight` row, and which launch takes
    it differs by framework and phase. The row id is the stable name of the
    place, so a test about a width asks for the place."""
    walk = _walk(model, framework)
    return [_facts(walk, r) for r in walk.per_op_timing
            if str(r.get("aggregation_id")).endswith(suffix)
            and (op is None or r.get("op") == op)]


def _one(model: str, suffix: str, op: Optional[str] = None,
         framework: str = "vllm") -> dict:
    rows = _rows(model, suffix, op, framework)
    assert rows, f"no {op or 'any'} row ending {suffix!r}"
    return rows[0]


V4 = "deepseek-ai/DeepSeek-V4-Flash"


def _mixes() -> dict:
    return _one(V4, "hc_attn_pre.mixes__hc_pre")


def test_an_f32_weight_makes_the_multiply_f32():
    mixes = _mixes()
    weight, compute = mixes["weight_dtype"], mixes["act_dtype"]
    assert (weight, compute) == ("F32", "F32")


def test_the_f32_multiply_still_stores_at_the_edges_width():
    """The half that must NOT move: the product is a bf16 tensor, because
    that is what the edge declares and what the kernel stores."""
    mixes = _mixes()
    written, width = mixes["output_dtype"], mixes["out_width"]
    assert (written, width) == ("BF16", 2.0)


def test_the_reader_of_the_product_reads_it_at_the_edges_width():
    """And so the stream after it is bf16 — no F32 chain, nothing to narrow
    back."""
    split = _one(V4, "hc_attn_pre.split__hc_pre")
    read_width, read = split["in_width"], split["act_in"]
    written = _mixes()["act_out"]
    assert read_width == 2.0
    assert read == written          # the same bytes on both sides of the edge


def test_the_product_is_timed_at_the_cards_f32_peak():
    mixes = _mixes()
    compute = mixes["act_dtype"]
    fp32_peak = float(get_gpu(GPU).fp32_tflops) * 1e12
    expected, timed = mixes["flops"] / fp32_peak, mixes["t_compute"]
    assert compute == "F32"
    assert math.isclose(timed, expected, rel_tol=1e-9)


def test_the_measurement_key_computes_fp32_and_writes_bf16():
    """Both halves in the one place a measurement reads them: the cell runs
    on the fp32 MMA and its output is bf16."""
    env = _mixes()["env"]
    assert env == {"activation": "fp32", "output": "bf16", "weight": "fp32"}


def test_promoted_row_identity_resolves_to_the_torch_fp32_floor():
    """One router gate, end to end: row identity → job → provider identity →
    the launch torch actually builds.

    The walk's promoted cell says both operands are fp32 while the edge still
    says bf16. That must remain distinguishable from an all-fp32 GEMM in the
    provider's identity, yet the launch must not turn the product back into an
    fp32 tensor.
    """
    from backend.inference.op_signatures import op_signature
    from kernel_bench.env import Env
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import Launch, applicable
    from kernel_bench.refs import get_ref, input_generator

    walk = _walk("MiniMaxAI/MiniMax-M2.7")
    row = next(r for r in walk.per_op_timing
               if str(r.get("aggregation_id")).endswith(
                   "block_sparse_moe.gate"))
    identity = RowIdentity(walk)
    op, shape_json = identity.of(row)
    dtype_json = identity.dtype_env(row)
    env = json.loads(dtype_json)
    assert (op, env) == (
        "linear", {"activation": "fp32", "output": "bf16", "weight": "fp32"})

    signature = op_signature(op)
    inv = Invocation.from_job({
        "signature": signature, "op": op, "dims": json.loads(shape_json),
        "dtype_env": env, "phase": "prefill", "physical_key": [
            signature, shape_json, dtype_json, "prefill"]})
    provider_env = Env("2.11", None, None, None, None,
                       libs={"torch": "2.11"})
    providers = applicable(inv, provider_env)
    assert [(p.library, p.kernel_impl, p.impl_id(inv))
            for p in providers] == [
        ("torch", "torch_cublas_linear",
         "torch_cublas_linear@compute=fp32")]
    assert providers[0].actual_computes_at(inv) == {}

    inputs = get_ref(signature).make_inputs(
        inv, "cpu", input_generator(inv, "cpu"))
    case = providers[0].build(inv, provider_env, inputs)
    assert isinstance(case.run, Launch)
    x, matrix, product, out = case.run.args
    assert (str(x.dtype), str(matrix.dtype),
            str(product.dtype), str(out.dtype)) == (
        "torch.float32", "torch.float32", "torch.float32", "torch.bfloat16")
    written = case.run()
    assert written is out
    assert str(written.dtype) == "torch.bfloat16"
    canonical = case.to_canonical(written)
    assert (str(canonical.dtype), tuple(canonical.shape)) == (
        "torch.bfloat16", (256, 256))


def test_the_precision_panel_reads_the_walks_promotion_table():
    """The panel's compute row is the same promoted fact the access row and
    measurement key read, not a second profile-side derivation.

    And the promotion this checkpoint carries is not a change: MiniMax
    ships its router gate in F32, so that gate multiplies in F32 as shipped
    too. The as-shipped column reads the promotion of the SAVED formats
    (`Analysis.shipped_weight_promotions`), which here is the same one, so
    the deployed walk under no profile reports nothing changed."""
    walk = _walk("MiniMaxAI/MiniMax-M2.7")
    [uid] = walk.analysis.weight_promotions
    assert uid in walk.analysis.shipped_weight_promotions
    node = walk.analysis.program.by_uid(uid)
    selector = walk.analysis.resolved_precision.role_value(node.role)
    row = next(r for r in walk.analysis.precision_effective()["effective"]
               if r["category"] == "compute" and r["selector"] == selector)
    assert (row["as_shipped"], row["effective"], row["changed"]) == (
        "FP32", "FP32", False)


def test_an_f32_router_gate_runs_its_multiply_fp32():
    """MiniMax-M2.7 stores its router gate F32, so the gate GEMM is keyed
    on the fp32 MMA — while its logits stay bf16 tensors, which is what the
    top-k that reads them is billed for."""
    model = "MiniMaxAI/MiniMax-M2.7"
    gate = _one(model, "block_sparse_moe.gate")
    topk = _one(model, "moe_router_topk_norm__minimax_m2_decoder_block")
    assert (gate["act_dtype"], gate["output_dtype"]) == ("F32", "BF16")
    assert (topk["in_width"], topk["env"]["activation"]) == (2.0, "bf16")


def test_a_bf16_mixing_matrix_stays_bf16():
    """GLM-5.3-Flash's `hc_*_fn` is BF16 in the checkpoint."""
    rows = _rows("zai-org/GLM-5.3-Flash", "", "linear_external_weight")
    seen = sorted({(r["weight_dtype"], r["output_dtype"],
                    r["env"]["activation"]) for r in rows})
    assert seen == [("BF16", "BF16", "bf16")]


def test_a_checkpoint_served_narrower_than_it_ships_promotes_nothing():
    """The boundary against the served-width rule: gemma-2 stores fp32 and
    both frameworks load it bf16, so `model_dtype` is the served bf16 while
    `shipped_model_dtype` keeps the stored fp32.

    Nothing is promoted there, and the reason is the one that keeps the two
    rules from fighting: this rule compares a weight's RESIDENT format
    against the edge, and the load-time narrowing is in the resident format
    too. A whole model served at bf16 is not "a weight wider than its
    activation" — if this ever fires on gemma-2, the rule has started
    reading the stored width instead of the resident one, and every weight
    of every fp32 checkpoint would promote."""
    walk = _walk("google/gemma-2-9b")
    precision = walk.analysis.resolved_precision
    shipped, served = precision.shipped_model_dtype, precision.model_dtype
    promotions = len(walk.analysis.weight_promotions)
    resident = {rep.resident.logical_dtype
                for reps in walk.analysis.formats.values.values()
                for rep in reps if rep.resident is not None}
    assert (shipped, served) == ("F32", "BF16")
    assert promotions == 0
    assert resident == {"BF16"}
