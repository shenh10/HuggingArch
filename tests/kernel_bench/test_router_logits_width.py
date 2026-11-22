"""A router's logits are measured at the width its kernel requires.

DeepSeek-V4 ships `ffn.gate.weight` BF16 and both frameworks load it BF16, so
nothing about the checkpoint makes its router logits fp32. The top-k kernel
each framework dispatches to does: it exists only at fp32
(`can_use_dsv4_topk`), so vLLM builds the gate
`GateLinear(out_dtype=torch.float32)` and sglang's `MoEGate.forward` writes
fp32 on every CUDA path it takes. Each binding declares that requirement on
the `router` role (`FrameworkBinding.edge_dtypes`).

Before the walk held it, the oracle built fp32 anyway — it had to, or it would
have timed the fallback kernel — and the cell said `bf16`, so an fp32
measurement was filed under a BF16 name. The statements below are read at
three different LINKS of that chain — not three independent derivations, and
saying so would overstate them, since the plan's key and the walk's row both
read `dtype_env["activation"]`:

* the PLAN's key, off the emitted job's `dtype_env`;
* what is MEASURED, off the tensor the reference builder actually allocates —
  a different chain, and the one that was broken: the oracle no longer
  overrides the cell;
* what the WALK bills, off a `pipeline.analyze` row's own `act_dtype`.

What is independent of the engine is the fourth statement, at the end: the
edge's bytes recomputed from the checkpoint's own config.

The hash-routed sibling (`moe_topk_hash_sqrtsoftplus`) stays declined, and for
a reason this change does not touch: its `tid2eid` table is I64 in the
checkpoint, a RESIDENT width rather than a required output width, and I64 has
no corpus token. That decline is asserted here too so the scope stays visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SPAN = "moe_topk_sqrtsoftplus"
HASH_SPAN = "moe_topk_hash_sqrtsoftplus"


def _binding(framework: str, *, declared: bool = True):
    import dataclasses

    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    binding = get_binding(framework).deployed(
        ServingDeployment.at(gpu_sku="B200", activation_dtype="BF16"))
    # Stripping the declaration is how each assertion below is shown to be
    # about the declaration and not about something else the walk already did.
    return binding if declared else dataclasses.replace(binding,
                                                        edge_dtypes=())


def _jobs(framework: str, *, declared: bool = True):
    from kernel_bench.plan import build_jobs

    return build_jobs(
        model_id=MODEL, batch_size=8, seq_len=1024, phase="decode", tp=8,
        framework=framework, binding=_binding(framework, declared=declared),
        avg_output_tokens=512, gpu="B200")


def _keyed_activations(framework: str, span: str, *, declared: bool = True):
    jobs, _dropped = _jobs(framework, declared=declared)
    return {job["dtype_env"].get("activation") for job in jobs
            if job.get("signature") == span}


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_the_plan_keys_the_router_at_the_width_the_framework_writes(framework):
    """Route 1: the key the measurement is filed under."""
    assert _keyed_activations(framework, SPAN) == {"fp32"}


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_without_the_declaration_the_key_is_the_checkpoints_width(framework):
    """The defect this closes, stated as the thing that changed.

    With the binding's `edge_dtypes` stripped the walk is back to pricing the
    width the BF16 gate weight implies, which is the key an fp32 measurement
    used to be stored under.
    """
    assert _keyed_activations(framework, SPAN, declared=False) == {"bf16"}


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_the_oracle_allocates_the_width_the_plan_keyed(framework):
    """Route 2: what is actually measured, read off the allocated tensor.

    Not `inv.dtype` and not the resolver — the tensor the reference builder
    hands a provider. `_moe_topk_inputs` used to force fp32 here regardless of
    the cell; it now reads the operand's declared slot, so this agreeing with
    the key above is a fact about the two rather than a restatement of one.
    """
    import torch

    from kernel_bench.invocation import Invocation
    from kernel_bench.refs import get_ref
    from tests.kernel_bench._inputs import build_inputs

    jobs, _dropped = _jobs(framework)
    job = next(j for j in jobs if j.get("signature") == SPAN)
    inv = Invocation.from_job(job)
    inputs = build_inputs(get_ref(SPAN), inv, device="cpu")
    assert inputs["gating"].dtype is torch.float32
    assert job["dtype_env"]["activation"] == "fp32"


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_the_walk_bills_the_router_edge_at_that_width(framework):
    """Route 3: the step's own rows, through `pipeline.analyze` rather than
    through the plan — the bytes and the peak column the row is judged
    against come from the same `act_dtype` the key was read off.

    The gate GEMM itself is the other half of the statement: its operand is
    still the BF16 hidden row and its weight is still BF16, so what the
    framework widened is the OUTPUT and not the multiply. A row whose
    `act_dtype` had moved would be `traffic.weight_promotion`'s subject, which
    this is not.
    """
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    from tests.snapshots import resolve_snapshot

    snapshot = resolve_snapshot("deepseek-ai__DeepSeek-V4-Flash")
    result = analyze(
        snapshot, parallel=ParallelConfig(),
        framework_binding=_binding(framework),
        batch_size=1, avg_input_tokens=128, avg_output_tokens=1,
        phase="prefill", breakdown=True)
    rows = {}
    for row in result.per_op_timing:
        leaf = str(row.get("aggregation_id", "")).split("|")[-1]
        if leaf.startswith(("ffn.topk_scale", "ffn.gate.weight")):
            rows.setdefault(leaf.split("__")[0], []).append(row)
    routers = rows["ffn.topk_scale"]
    assert routers, "the step has no router span"
    assert {row.get("act_dtype") for row in routers} == {"F32"}
    gates = rows["ffn.gate.weight"]
    assert {row.get("output_dtype") for row in gates} == {"F32"}
    # The multiply is untouched: BF16 operand, BF16 weight, BF16 column.
    assert {row.get("act_dtype") for row in gates} == {"BF16"}
    assert {row.get("weight_dtype") for row in gates} == {"BF16"}
    # And the bytes it moved are exactly the logits at four bytes instead of
    # two — counted from the CONFIG, so the number does not come from the
    # engine it is checking. 128 tokens x n_routed_experts x the MoE layers.
    config = snapshot.config
    experts = int(config["n_routed_experts"])
    moe_layers = int(config["num_hidden_layers"]) - int(
        config.get("first_k_dense_replace") or 0)
    written = sum(row["act_out_bytes_per_rank"] for row in gates)
    assert written == 128 * experts * moe_layers * 4


@pytest.mark.parametrize("framework", ["vllm", "sglang"])
def test_the_hash_routed_sibling_is_still_declined_for_its_resident_table(
        framework):
    """The scope line, asserted rather than described.

    `tid2eid` is I64 on disk and both frameworks allocate it int32, but that
    is the loaded width of a RESIDENT tensor — `Representation.intent`, fixed
    by the walk before any binding is in hand — not the width a role's output
    is written at. So this span still has no corpus token for its own
    environment and is still declined, exactly as
    `docs/inference_cost_model.md` "Stored width vs loaded width" says.
    """
    jobs, dropped = _jobs(framework)
    assert not [j for j in jobs if j.get("signature") == HASH_SPAN]
    reasons = [str(d) for d in dropped if HASH_SPAN in str(d)]
    assert reasons, f"{HASH_SPAN} is neither emitted nor declined"
    assert all("no corpus token" in reason for reason in reasons)


def test_a_cardless_walk_takes_the_requirement_and_no_launch():
    """A required width alone must not pull a launch resolution with it.

    `/inference/compute-graph` and `/fusion` describe a deployment with no
    card. The framework still states the router's requirement there — it is a
    kernel fact, not a card fact — but no launch can be SELECTED without the
    capability table that orders the candidates, so the walk must publish the
    requirement and leave every launch fact as the cardless walk had it. The
    one condition `serving` binds (`apply_launches`) is what keeps those two
    apart: spelled twice, the requirement pushed the gate open and the launch
    resolution followed it through, publishing `missing_deployment_fact:`
    gaps for candidates nothing had been asked to supply.
    """
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    from tests.snapshots import resolve_snapshot, shared_program

    # No card and no platform: the ordering table is absent, so a candidate
    # that needs one is unavailable rather than selected.
    binding = get_binding("vllm").deployed(
        ServingDeployment.of(activation_dtype="BF16"))
    assert binding.deployment.peak_families == (), "a walk with no card"
    assert binding.unavailable_precision_launches_for(
        shared_program("deepseek-ai__DeepSeek-V4-Flash")), (
        "the case only bites where a candidate went unavailable"
    )
    result = analyze(
        resolve_snapshot("deepseek-ai__DeepSeek-V4-Flash"),
        parallel=ParallelConfig(), framework_binding=binding,
        batch_size=1, avg_input_tokens=128, avg_output_tokens=1,
        phase="prefill", breakdown=True)
    precision = result.analysis.resolved_precision
    assert precision.required_edge_dtypes, "the requirement is stated"
    assert precision.edge_dtype("router", "") == "F32"
    assert precision.launches == (), "no card ordered the candidates"
    # This line is the discriminating one, not the one above: before the fix
    # `with_launches([])` stored `()` too, so `launches` read empty either
    # way, and what gave the defect away was the card it had been handed.
    assert precision.launch_peak_families is None
    invented = [row for row in result.per_op_timing
                if row.get("launch_gaps")]
    assert not invented, [
        (row.get("aggregation_id"), row["launch_gaps"]) for row in invented]
