"""Torch floors whose timed launch the launch gate refused on every pass.

Each case is a planned cell (the job's own shapes and dtype environment,
as a sweep carries them) on which the floor's launch used to be refused —
"its largest operand is float32 where the cell declares bfloat16", or "it
measures a part of the cell" — and is run here through the driver itself
(`run_jobs`: inputs, the provider's build, the traced cross-check call and
the gate), stopping before the timing loop. A floor that passes records a
row; one that does not is a failure the driver would log every pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch")

_ACT = {"activation": "bf16", "output": "bf16"}
_KV = {**_ACT, "kv": "bf16"}
_W = {**_ACT, "weight": "bf16"}


def _ports(*names):
    """The plan port each operand fills, as a planned job names them
    (`operand_ports`, from the row): one per operand, in the operands'
    order."""
    return [[name] for name in names]


def _job(signature, op, phase, dims, shape_in, shape_out, shape_weight=None,
         dtype_env=_ACT, **extra):
    import json
    return {"signature": signature, "op": op, "phase": phase, "tp": 1,
            "dims": dims, "shape_in": shape_in, "shape_out": shape_out,
            "shape_weight": shape_weight, "dtype_env": dict(dtype_env),
            "physical_key": [signature, json.dumps(dims, sort_keys=True),
                             json.dumps(dtype_env, sort_keys=True), phase],
            **extra}


# (kernel_impl, the cell it was refused on) — the cells are the ones the
# H100 / RTX 4090 queues logged, shapes as their plans carried them. The
# default suite runs only the first: the rope decode cell, the refusal the
# queues logged most (≈340 a pass) and the cheapest to run.
CASES = [
    # fp32 cos/sin table: the widest operand of a decode rotation.
    ("torch_rope", _job(
        "rope", "fused_qk_rope", "decode",
        {"B": 1, "d_rope": 64, "n_heads": 1, "n_kv_heads": 1, "s_query": 1},
        [[1, 1, 1, 64], [1, 1, 64]], [[1, 1, 1, 64], [1, 1, 64]],
        dtype_env=_KV, operand_ports=_ports("q_pe", "k_pe"))),
    # the passed-through half of each head is read only by the `cat`.
    ("torch_rope_partial", _job(
        "rope_partial", "fused_qk_rope_partial", "prefill",
        {"B": 16, "d_head": 192, "d_rope": 64, "n_heads": 2,
         "n_kv_heads": 1, "s_query": 1024},
        [[16, 2, 1024, 192], [16, 1, 1024, 192]],
        [[16, 2, 1024, 192], [16, 1, 1024, 192]], dtype_env=_KV,
        operand_ports=_ports("q", "k"))),
    # a projection whose output is the widest tensor of the cell.
    ("torch_cublas_linear", _job(
        "gemm", "linear", "prefill",
        {"B": 1, "d_out": 5120, "d_reduce": 432, "s_query": 1024},
        [1, 1024, 432], [1, 1024, 5120], [5120, 432], dtype_env=_W)),
    # an expanding gather out of a small vocabulary.
    ("torch_embedding", _job(
        "embedding", "embedding", "prefill",
        {"B": 1, "d_model": 6144, "n_vocab": 303, "s_query": 1024},
        [1, 1024], [1, 1024, 6144], [303, 6144], dtype_env=_W)),
    # one matmul per expert, each reading one expert's weight.
    ("torch_ragged_matmul", _job(
        "grouped_gemm", "grouped_gemm", "prefill",
        {"n_experts": 8, "n_tokens": 64, "s_expert_rows": 8,
         "d_reduce": 256, "d_out": 512},
        [8, 8, 256], [8, 8, 512], [8, 512, 256], dtype_env=_W,
        kernel_params={"moe_token_distribution": "balanced"})),
]

# A windowed GQA cell. On an RTX 4090 no fused SDPA backend takes it (flash
# takes no mask, memory-efficient no GQA) and SDPA's math path scores in
# fp32; on the CPU flash takes it, so the test states the GPU's answer.
_WINDOWED_GQA = _job(
    "gqa_attention", "flash_attention", "prefill",
    {"B": 1, "d_head": 64, "kv_block_size": 16, "n_heads": 2,
     "n_kv_heads": 1, "s_kv": 256, "s_query": 256, "s_window": 64},
    [[1, 2, 256, 64], [1, 1, 256, 64], [1, 1, 256, 64]],
    [1, 2, 256, 64], dtype_env=_KV, operand_ports=_ports("q", "k", "v"))


@pytest.mark.parametrize("impl, job", CASES, ids=[c[0] for c in CASES])
def test_the_floor_launches_the_cell(impl, job, monkeypatch):
    _assert_launches_the_cell(impl, job, monkeypatch)


# The canary the H100 disabled the softmax router floor on (Qwen3.5,
# 2026-10-26): "float32[1024,256]", the floor's fp32 copy of the logits.
_SOFTMAX_ROUTER = _job(
    "moe_topk_softmax", "moe_topk_softmax", "prefill",
    {"B": 1, "n_experts": 256, "n_experts_active": 8, "s_query": 1024},
    [1, 1024, 256], [1, 1024, 8])


def test_the_softmax_router_floor_launches_the_cell(monkeypatch):
    _assert_launches_the_cell("torch_moe_topk_softmax", _SOFTMAX_ROUTER,
                              monkeypatch)


# The canary the H100 disabled the QSA key-pool floor on (Qwen3.8-Flash-Next,
# 2026-10-27): "float32[256,4,128]", the floor's fp32 copy of the raw keys
# it pools. The pool's mean accumulates in fp32 without one.
_QSA_K_COMPRESS = _job(
    "qsa_index_k_compress_store", "qsa_index_k_compress_store", "prefill",
    {"B": 1, "compress_ratio": 4, "d_indexer_head": 128, "d_rope": 64,
     "kv_block_size": 16, "s_query": 1024, "s_slots": 256},
    [[1, 1024, 128], [128]], [1, 256, 128],
    dtype_env={**_KV, "weight": "bf16"},
    operand_ports=_ports("raw_key", "weight"))


def test_the_qsa_key_pool_floor_launches_the_cell(monkeypatch):
    _assert_launches_the_cell("torch_qsa_index_k_compress_store",
                              _QSA_K_COMPRESS, monkeypatch)


# The canary the H100 disabled the dense indexer floor on, every DSA model
# (2026-10-28): "it measures a part of the cell" — the floor scored all four
# requests' queries against ONE shared key buffer, a quarter of the keys
# the cell bills (each request reads its own cache).
_INDEXER_DECODE = _job(
    "indexer_logits", "indexer_logits", "decode",
    {"B": 4, "d_indexer_head": 128, "kv_block_size": 64,
     "n_indexer_heads": 32, "s_kv": 16512, "s_query": 1},
    [[4, 1, 32], [4, 16512, 128], [4, 1, 4096]], [4, 1, 16512],
    dtype_env={"activation": "bf16", "kv": "bf16", "output": "fp32"},
    operand_ports=_ports("w", "k", "q"))


def test_the_indexer_floor_launches_the_decode_cell(monkeypatch):
    _assert_launches_the_cell("torch_dense_mqa_logits", _INDEXER_DECODE,
                              monkeypatch)


def test_the_indexer_oracle_scores_each_request_against_its_own_keys():
    """The oracle against the definition — the registered op's
    Σ_h w·ReLU(q·k) — request by request in a loop,
    not the batched contraction both the oracle and the floor use. The
    control: a batch scored against request 0's keys (the old shared-key
    form) is a different function, so the loop discriminates."""
    import torch
    from kernel_bench.invocation import Invocation
    from kernel_bench.refs import get_ref, input_generator
    job = _job("indexer_logits", "indexer_logits", "decode",
               {"B": 3, "d_indexer_head": 16, "kv_block_size": 64,
                "n_indexer_heads": 4, "s_kv": 40, "s_query": 2},
               [[3, 2, 4], [3, 40, 16], [3, 2, 64]], [3, 2, 40],
               dtype_env={"activation": "bf16", "kv": "bf16",
                          "output": "fp32"},
               operand_ports=_ports("w", "k", "q"))
    inv = Invocation.from_job(job)
    spec = get_ref(inv.signature)
    inputs = spec.make_inputs(inv, "cpu", input_generator(inv, "cpu"))
    q, k, w = (inputs["q"].float(), inputs["k"].float(),
               inputs["weights"].float())
    assert k.shape == (3, 40, 16)
    rows = []
    for b in range(3):
        for t in range(2):
            i = b * 2 + t
            rows.append(sum(w[i, h] * (k[b] @ q[i, h]).clamp(min=0)
                            for h in range(4)))
    want = torch.stack(rows)
    got = spec.reference(inputs)
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    shared = torch.einsum("thd,jd->thj", q, k[0]).relu()
    assert not torch.allclose(torch.einsum("thj,th->tj", shared, w), want)
    # And the ReLU is the op's (`indexer_logits`' formula): without it the
    # sum is another function.
    linear = torch.stack([sum(w[b * 2 + t, h] * (k[b] @ q[b * 2 + t, h])
                              for h in range(4))
                          for b in range(3) for t in range(2)])
    assert not torch.allclose(linear, want)


# The canary the H100 disabled the torch sink floor on (Hy4, 2026-10-28):
# the floor timed the fp32 oracle, and the job carried the latent value
# broadcast to every head. Shapes as the plan now publishes them: the value
# as stored, one latent row a key.
_SPARSE_SINK_DECODE = _job(
    "mla_sparse_sink_attention", "flash_dsa_sink", "decode",
    {"B": 1, "d_head": 576, "d_v_head": 512, "kv_block_size": 64,
     "n_heads": 64, "n_topk_keys": 1153, "s_kv": 1153, "s_query": 2},
    [[1, 64, 2, 576], [1, 1153, 576], [1, 2, 1153], [64], [1, 1153, 512]],
    [1, 64, 2, 512],
    dtype_env={**_KV, "weight": "bf16"},
    operand_ports=_ports("q", "k", "mask", "sinks", "v"))


def test_the_sparse_sink_floor_launches_the_cell(monkeypatch):
    _assert_launches_the_cell("torch_sparse_mla_sink", _SPARSE_SINK_DECODE,
                              monkeypatch)


# Gated activations whose plan absorbed the cast to the consuming GEMM's
# activation format: the launch writes FP8 elements and scales. No provider
# declared such a cell, so every one of them was left unattempted on the
# H100 (DeepSeek-V4's clamped SwiGLU at per-group:128 / per-group:32 UE8M0,
# Ling-2.6's SwiGLU per-token). DeepSeek-V4's per-group:128 first: the most
# cells, and the one a floor that ignored the cast would fail on.
_QUANT_ACT = {"activation": "bf16", "output": "fp8_e4m3"}
_GATED_QUANT_CASES = [
    ("torch_silu_and_mul_with_clamp_quant", _job(
        "silu_and_mul_with_clamp", "silu_and_mul_with_clamp", "decode",
        {"B": 1, "d_ffn": 256, "s_query": 1},
        [[1, 1, 256], [1, 1, 256]], [1, 1, 256],
        dtype_env={**_QUANT_ACT, "activation_granularity": "per-group:128",
                   "metadata": "ue8m0"},
        kernel_params={"clamp_limit": 10.0},
        operand_ports=_ports("gate", "up"))),
    ("torch_silu_and_mul_with_clamp_quant", _job(
        "silu_and_mul_with_clamp", "silu_and_mul_with_clamp", "decode",
        {"B": 1, "d_ffn": 288, "s_query": 1},
        [[1, 1, 288], [1, 1, 288]], [1, 1, 288],
        dtype_env={**_QUANT_ACT, "activation_granularity": "per-group:32",
                   "metadata": "ue8m0"},
        kernel_params={"clamp_limit": 10.0},
        operand_ports=_ports("gate", "up"))),
    ("torch_silu_and_mul_quant", _job(
        "silu_and_mul", "silu_and_mul", "decode",
        {"B": 2, "d_ffn": 256, "s_query": 1},
        [[2, 1, 256], [2, 1, 256]], [2, 1, 256],
        dtype_env={**_QUANT_ACT, "activation_granularity": "per-token",
                   "metadata": "fp32"},
        operand_ports=_ports("gate", "up"))),
]


@pytest.mark.parametrize("impl, job", _GATED_QUANT_CASES,
                         ids=[f"{c[0]}-{c[1]['dtype_env']['activation_granularity']}"
                              for c in _GATED_QUANT_CASES])
def test_a_gated_activation_floor_writes_the_quantised_record(
        impl, job, monkeypatch):
    _assert_launches_the_cell(impl, job, monkeypatch)


def test_the_quantised_gated_floor_walks_big_cells_in_row_blocks(monkeypatch):
    """Past its working-row budget the floor walks the rows in blocks (the
    H100's 2M-row prefill cells otherwise hold four widened copies of the
    row at once) and writes the record the one-shot launch writes, bit for
    bit (the one-shot record is held to the oracle by the cases above)."""
    import torch
    import kernel_bench.providers.norm_act as norm_act
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    from kernel_bench.refs import get_ref, input_generator
    inv = Invocation.from_job(_job(
        "silu_and_mul_with_clamp", "silu_and_mul_with_clamp", "prefill",
        {"B": 1, "d_ffn": 256, "s_query": 300}, [[1, 300, 256], [1, 300, 256]],
        [1, 300, 256], dtype_env={**_QUANT_ACT, "metadata": "ue8m0",
                                  "activation_granularity": "per-group:128"},
        kernel_params={"clamp_limit": 10.0},
        operand_ports=_ports("gate", "up")))
    spec = get_ref(inv.signature)
    inputs = spec.make_inputs(inv, "cpu", input_generator(inv, "cpu"))
    floor = next(p for p in all_providers()
                 if p.kernel_impl == "torch_silu_and_mul_with_clamp_quant")
    case = floor.build(inv, None, inputs)
    whole = case.to_canonical(case.run())
    # Seven-row blocks: 300 rows is not a multiple, so the tail block runs.
    monkeypatch.setattr(norm_act, "_GATED_QUANT_BLOCK_BYTES", 4 * 4 * 256 * 7)
    case = floor.build(inv, None, inputs)
    blocked = case.to_canonical(case.run())
    assert torch.equal(whole, blocked)


def test_the_quantised_gated_floor_declines_a_row_its_groups_do_not_tile():
    """MiMo-V2.5 plans a 4-wide shard at per-group:128 (and DeepSeek-V4-Pro
    a 192-wide one at 16-way expert TP): no whole group, so the floor
    declines before allocating, as every framework quantiser asserts it."""
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    floor = next(p for p in all_providers()
                 if p.kernel_impl == "torch_silu_and_mul_quant")
    inv = Invocation.from_job(_job(
        "silu_and_mul", "silu_and_mul", "decode",
        {"B": 96, "d_ffn": 4, "s_query": 1}, [[96, 1, 4], [96, 1, 4]],
        [96, 1, 4], dtype_env={**_QUANT_ACT, "metadata": "fp32",
                               "activation_granularity": "per-group:128"},
        operand_ports=_ports("gate", "up")))
    with pytest.raises(NotImplementedError):
        floor.impl_params(inv)


# DeepSeek-R1's FP8 MoE dispatch: the permute moves the quantised rows and
# their scales, and the cell says so on both dtype slots. No provider
# declared it (934 H100 cells never attempted).
_FP8_PERMUTE = _job(
    "moe_permute", "permute", "decode",
    {"B": 1, "d_model": 7168, "n_experts": 1, "n_tokens": 1,
     "s_expert_rows": 1, "s_query": 1},
    [[1, 1, 7168], [1, 1, 8]], [1, 1, 7168],
    dtype_env={"activation": "fp8_e4m3", "output": "fp8_e4m3",
               "metadata": "fp32",
               "activation_granularity": "per-group:128"})


def test_the_quantised_dispatch_floor_launches_the_cell(monkeypatch):
    _assert_launches_the_cell("torch_index_permute_quantised", _FP8_PERMUTE,
                              monkeypatch)


def test_the_quantised_dispatch_declines_a_row_its_groups_do_not_tile():
    """A TP-sharded dispatch width that is not a whole number of the cell's
    scale groups is declined before any input is allocated, the way every
    other launch holding a quantised row declines it — it used to be
    accepted and then raise from the grid inside `build`."""
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    floor = next(p for p in all_providers()
                 if p.kernel_impl == "torch_index_permute_quantised")
    job = dict(_FP8_PERMUTE)
    job["dims"] = {**job["dims"], "d_model": 192}
    inv = Invocation.from_job(job)
    assert floor.serves.accepts(inv)
    with pytest.raises(NotImplementedError):
        floor.impl_params(inv)
    # And the tiling width is the dispatch's own row, so the planned
    # 7168-wide cell is served.
    assert floor.impl_params(Invocation.from_job(_FP8_PERMUTE)) == ""


def test_a_computing_launch_cannot_be_registered_as_moving_a_record():
    """The scope itself refuses at load: "same record in and out" is a move
    only for a launch whose op declares the gather."""
    from kernel_bench.providers.gemm import _moved_record_scope
    assert _moved_record_scope("moe_permute").signature == "moe_permute"
    with pytest.raises(ValueError, match="computes its output"):
        _moved_record_scope("moe_combine")


def test_the_quantised_dispatch_moves_the_record_not_a_widened_row():
    """The launch's own traffic: one byte an element plus four a scale
    group, which is what makes this cell different from the bf16 permute
    beside it. A floor that permuted widened rows would move 2 bytes an
    element and the dequantised rows would still match."""
    import torch
    from kernel_bench import launch_trace
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    from kernel_bench.refs import get_ref, input_generator
    inv = Invocation.from_job(_FP8_PERMUTE)
    spec = get_ref("moe_permute")
    inputs = spec.make_inputs(inv, "cpu", input_generator(inv, "cpu"))
    floor = next(p for p in all_providers()
                 if p.kernel_impl == "torch_index_permute_quantised")
    case = floor.build(inv, None, inputs)
    recording = launch_trace.record()
    pair = recording.run(case.run)
    assert pair[0].dtype == torch.float8_e4m3fn
    assert pair[1].shape == (1, 7168 // 128) and pair[1].dtype == torch.float32
    moved = [c for c in recording.trace.calls if c.moves]
    assert moved and all(
        any(o.itemsize in (1, 4) for o in c.operands) for c in moved)
    assert not any(o.itemsize == 2 for c in moved for o in c.operands)


def test_the_unrenormalised_softmax_floor_meets_the_oracle():
    # The bench's softmax cells are all renormalised; the other branch
    # (DeepSeek-V2-Lite's router) is held to the oracle here.
    import torch
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    from kernel_bench.refs import get_ref, input_generator
    inv = Invocation.from_job(_SOFTMAX_ROUTER)
    spec = get_ref(inv.signature)
    inputs = dict(spec.make_inputs(inv, "cpu", input_generator(inv, "cpu")),
                  renormalize=False, routed_scaling_factor=2.5)
    floor = next(p for p in all_providers()
                 if p.kernel_impl == "torch_moe_topk_softmax")
    case = floor.build(inv, None, inputs)
    got, want = case.to_canonical(case.run()), spec.reference(inputs)
    assert torch.allclose(got.float(), want.float(), rtol=1e-2, atol=1e-4)


def test_a_balanced_grouped_build_packs_its_rows_in_place():
    # Every expert holds `s_expert_rows` rows, so the packed [M, K] the
    # ragged launch takes IS the padded [E, Sq, K] activation: a copy held
    # it twice, and at the H100's 2,621,440 x 8,192 prefill cell that
    # second 40 GiB ran the vLLM Triton build out of memory (2026-10-26).
    # Every grouped build packs through one helper; the torch floor runs it
    # here on the CPU.
    from kernel_bench.env import Env
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    from kernel_bench.refs import get_ref, input_generator
    job = CASES[4][1]
    inv = Invocation.from_job(job)
    assert set(inv.moe_token_counts) == {job["dims"]["s_expert_rows"]}
    inputs = get_ref(inv.signature).make_inputs(
        inv, "cpu", input_generator(inv, "cpu"))
    provider = next(p for p in all_providers()
                    if p.kernel_impl == "torch_ragged_matmul")
    launched = provider.convert_inputs(inv, inputs)
    case = provider.build(inv, Env("2.11", None, None, None, None,
                                   libs={"torch": "2.11"}), launched)
    rows = case.run.args[0]
    assert rows.shape == (sum(inv.moe_token_counts), job["dims"]["d_reduce"])
    assert (rows.untyped_storage().data_ptr()
            == launched["x"].untyped_storage().data_ptr())


def test_a_padded_grouped_build_packs_only_the_routed_rows():
    import torch
    from kernel_bench.providers.gemm import _packed_rows
    x = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
    assert torch.equal(_packed_rows(x, [3, 1, 0, 2]),
                       torch.cat([x[0], x[1, :1], x[3, :2]]))


def test_sdpa_without_a_fused_backend_keeps_the_cells_precision(monkeypatch):
    from kernel_bench.providers import attention
    monkeypatch.setattr(attention, "_fused_sdpa_takes", lambda *a: False)
    _assert_launches_the_cell("torch_sdpa", _WINDOWED_GQA, monkeypatch)


def test_sdpa_times_the_backends_its_probe_accepted(monkeypatch):
    # The H100's answer for the windowed GQA cell: cuDNN takes it once MATH
    # is excluded, so the probe says yes. torch's default order puts MATH
    # ahead of cuDNN, so a timed call made outside the probe's backend set
    # ran the math path and its fp32 scores (Mistral-7B, 2026-10-26:
    # "float32[2,1024,1024]"). The timed call runs with MATH excluded.
    import torch
    from kernel_bench.providers import attention
    monkeypatch.setattr(attention, "_fused_sdpa_takes", lambda *a: True)
    seen = []

    def sdpa(q, k, v, **kwargs):
        seen.append(torch.backends.cuda.math_sdp_enabled())
        return torch.zeros_like(q)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention",
                        sdpa)
    _timed_trace("torch_sdpa", _WINDOWED_GQA)
    assert seen == [False]


# A non-absorbed MLA prefill cell whose score matrix outgrows q (s_kv 256
# against d_head 192): on SDPA's math path the widest buffer is fp32.
_MLA_PREFILL = _job(
    "mla_attention", "mla_attention", "prefill",
    {"B": 1, "d_head": 192, "d_v_head": 128, "n_heads": 2,
     "n_kv_heads": 2, "s_kv": 256, "s_query": 256},
    [[1, 256, 2, 192], [1, 256, 2, 192], [1, 256, 2, 128]],
    [1, 256, 2, 128], dtype_env=_KV)


def test_mla_prefill_without_a_fused_backend_keeps_the_cells_precision(
        monkeypatch):
    from kernel_bench.providers import attention
    monkeypatch.setattr(attention, "_fused_sdpa_takes", lambda *a: False)
    _assert_launches_the_cell("torch_sdpa_mla_prefill", _MLA_PREFILL,
                              monkeypatch)


_MLA_ROPE = _job(
    "rope_partial_q", "fused_qk_rope_partial_q", "prefill",
    {"B": 1, "d_head": 192, "d_rope": 64, "n_heads": 8, "s_query": 64},
    [[1, 8, 64, 192], [1, 64, 64]], [[1, 8, 64, 192], [1, 64, 64]],
    dtype_env=_KV, operand_ports=_ports("q", "k_pe"))


def _timed_trace(impl, job, monkeypatch=None, *, with_inputs=False):
    """What the floor's timed call dispatches, built as the driver builds it
    (and, asked, the cell's canonical inputs it was built from)."""
    from kernel_bench.env import Env
    from kernel_bench.invocation import Invocation
    from kernel_bench.launch_trace import record
    from kernel_bench.providers import all_providers
    from kernel_bench.refs import get_ref, input_generator
    inv = Invocation.from_job(job)
    inputs = get_ref(inv.signature).make_inputs(
        inv, "cpu", input_generator(inv, "cpu"))
    provider = next(p for p in all_providers() if p.kernel_impl == impl)
    launched = provider.convert_inputs(inv, inputs)
    case = provider.build(inv, Env("2.11", None, None, None, None,
                                   libs={"torch": "2.11"}), launched)
    with record() as recorded:
        case.run()
    return (recorded.trace, inputs) if with_inputs else recorded.trace


@pytest.mark.parametrize("impl, job", [("torch_rope", CASES[0][1]),
                                       ("torch_rope_partial_q", _MLA_ROPE)])
def test_a_rope_floor_times_the_rotation_alone(impl, job):
    # No cast inside the timed call (the cos/sin table is cast once, in the
    # build), and every operand at the cell's precision.
    trace = _timed_trace(impl, job)
    assert not [c for c in trace.calls if c.op.startswith("aten._to_copy")]
    assert {o.dtype for c in trace.work for o in c.operands
            if o.dtype in ("float32", "float16", "bfloat16")} == {"bfloat16"}


def _wider_than_the_rotary_tail(trace, job, table_elements):
    """Operands of the timed call holding more than q's rotary tail
    (tokens x n_heads x d_rope), in any layout. The one operand allowed to
    be larger is the cos/sin table, recognised by its own element count
    (max_pos x d_rope, from the cell's inputs) — not by where its extents
    sit, which a whole q laid out [B, N, S, D] or flat would also dodge."""
    dims = job["dims"]
    tail = dims["B"] * dims["s_query"] * dims["n_heads"] * dims["d_rope"]
    return [str(o) for c in trace.work for o in c.operands
            if o.elements > tail and o.elements != table_elements]


def test_the_mla_rope_floor_does_not_rejoin_the_query():
    # Serving rotates q's d_rope tail in place and never re-joins it
    # (vLLM mla.py:227-230); the join the cross-check compares is
    # canonicalisation. So nothing in the timed call holds more than q's
    # rotary tail — not q's untouched part, not a joined head, not q whole.
    trace, inputs = _timed_trace("torch_rope_partial_q", _MLA_ROPE,
                                 with_inputs=True)
    assert trace.work
    table = inputs["cos_sin_cache"].numel()
    assert _wider_than_the_rotary_tail(trace, _MLA_ROPE, table) == []


@pytest.mark.parametrize("shape", [(1, 8, 64, 192), (8, 64, 192),
                                   (64 * 8 * 192,), (64, 8 * 192)])
def test_a_whole_q_operand_is_caught_in_any_layout(shape):
    # The check itself: a re-join that hands q whole to the timed call is
    # caught however its extents are arranged — leading batch, leading
    # heads, or flat.
    from kernel_bench.launch_trace import Call, Operand, Trace
    trace = Trace(calls=[Call("aten.cat", (Operand("bfloat16", shape),))])
    assert _wider_than_the_rotary_tail(trace, _MLA_ROPE,
                                       table_elements=8192 * 64)


def test_the_sdpa_fallback_builds_its_mask_once(monkeypatch):
    # The window mask is position arithmetic on int32; none of it may run
    # inside the timed attention.
    from kernel_bench.providers import attention
    monkeypatch.setattr(attention, "_fused_sdpa_takes", lambda *a: False)
    trace = _timed_trace("torch_sdpa", _WINDOWED_GQA)
    assert trace.work
    assert not [o for c in trace.work for o in c.operands
                if o.dtype == "int64"]
    # The precomputed mask is read once, by the fill; nothing negates or
    # combines masks in the timed call.
    assert not [c for c in trace.work if "bitwise" in c.op]


def test_a_windowed_cell_past_the_dense_max_is_walked_in_query_blocks(
        monkeypatch):
    # Past `_WINDOWED_DENSE_MAX` the dense spelling's [Sq,Lq] distance matrix
    # and bias are the OOM, so the floor asks no fused backend and holds no
    # [Sq,Lq] tensor: the window is walked in query blocks, computing what
    # the dense eager spelling computes. One row a block here, so the block
    # loop — not a single block the size of the cell — is what is checked.
    from kernel_bench import launch_trace
    from kernel_bench.env import Env
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers, attention
    from kernel_bench.refs import get_ref, input_generator, visible_keys
    monkeypatch.setattr(attention, "_WINDOWED_DENSE_MAX", 0)
    monkeypatch.setattr(attention._refs, "GATHER_BLOCK_BYTES", 1)

    def probe(*a):
        raise AssertionError("past the dense max no fused backend is asked")
    monkeypatch.setattr(attention, "_fused_sdpa_takes", probe)
    inv = Invocation.from_job(_WINDOWED_GQA)
    inputs = get_ref(inv.signature).make_inputs(
        inv, "cpu", input_generator(inv, "cpu"))
    provider = next(p for p in all_providers()
                    if p.kernel_impl == "torch_sdpa")
    case = provider.build(inv, Env("2.11", None, None, None, None,
                                   libs={"torch": "2.11"}),
                          provider.convert_inputs(inv, inputs))
    with launch_trace.record() as recorded:
        got = case.run()
    full = inv.dims["s_query"] * inv.dims["s_kv"]
    assert recorded.trace.work
    assert not [o for c in recorded.trace.work for o in c.operands
                if o.elements >= full]
    q, k, v = (inputs[n].transpose(1, 2) for n in ("q", "k", "v"))
    hidden = ~visible_keys(q.shape[2], k.shape[2], q.device, causal=True,
                           s_window=inputs["s_window"])
    want = attention._torch_eager_gqa(q, k, v, hidden)
    assert (got.float() - want.float()).abs().max().item() < 2e-2


# (Sq, Lq, W, budget): queries the last Sq of a longer context (chunked
# prefill / verify), bottom-right aligned — first, because a top-left slip
# shows only here; a decode row whose window is shorter than the context;
# and a square cell. Budgets of a block a row, and of blocks that share an
# interior mask.
@pytest.mark.parametrize("Sq, Lq, W, budget", [
    (7, 19, 4, 1), (1, 13, 5, 1), (3, 11, 6, 1), (40, 40, 6, 4 * 2 * 2 * 120)])
def test_the_windowed_block_walk_is_the_dense_eager_attention(
        Sq, Lq, W, budget, monkeypatch):
    import torch
    from kernel_bench.providers import attention
    from kernel_bench.refs import visible_keys
    monkeypatch.setattr(attention._refs, "GATHER_BLOCK_BYTES", budget)
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, Sq, 8, generator=gen).to(torch.bfloat16)
    k, v = (torch.randn(1, 2, Lq, 8, generator=gen).to(torch.bfloat16)
            for _ in range(2))
    got = attention._torch_eager_windowed_gqa(q, k, v, W)()
    hidden = ~visible_keys(Sq, Lq, q.device, causal=True, s_window=W)
    assert torch.equal(got, attention._torch_eager_gqa(q, k, v, hidden))


def test_a_windowed_cell_under_the_dense_max_keeps_the_dense_path(monkeypatch):
    # The cells the sweeps measured are all under the max: the probe is
    # still asked, and the mask spelling is `_sdpa_causal_args`'s.
    from kernel_bench.providers import attention
    seen = []
    probe = attention._fused_sdpa_takes

    def spied(*a):
        seen.append(a)
        return probe(*a)
    monkeypatch.setattr(attention, "_fused_sdpa_takes", spied)
    _timed_trace("torch_sdpa", _WINDOWED_GQA)
    assert len(seen) == 1
    assert "attn_mask" in seen[0][4]


def test_the_grouped_floor_declines_a_torch_without_grouped_mm():
    # `torch._grouped_mm` runs on every device only from torch 2.9; an
    # older torch declines the provider instead of failing every cell.
    from kernel_bench.env import Env
    from kernel_bench.providers import all_providers
    floor = next(p for p in all_providers()
                 if p.kernel_impl == "torch_ragged_matmul")
    old = Env("2.8.0", None, None, None, None, libs={"torch": "2.8.0"})
    new = Env("2.9.0", None, None, None, None, libs={"torch": "2.9.0"})
    assert not floor.available(old)
    assert floor.available(new)
    # Not a typo in an intersection: the shared torch env may start at 2.5,
    # but this provider narrows its own launch to 2.9, and the reason says
    # both facts.
    assert next(floor.unavailability(old), None) == (
        "torch 2.8.0 is outside >=2.5,>=2.9 "
        "(library floor >=2.5; this kernel's floor >=2.9)")


def test_the_fused_backend_question_is_asked_of_torch():
    import torch
    from kernel_bench.providers.attention import _fused_sdpa_takes
    q = torch.randn(1, 2, 8, 64)
    k = torch.randn(1, 1, 8, 64)
    v = torch.randn(1, 1, 8, 32)
    # Q and V of different widths: no fused backend takes it on any device.
    assert not _fused_sdpa_takes(q, k, v, True, {"is_causal": True})


def _assert_launches_the_cell(impl, job, monkeypatch):
    import kernel_bench.driver as driver
    from kernel_bench.env import Env

    original = driver.applicable
    monkeypatch.setattr(driver, "applicable", lambda inv, env: [
        p for p in original(inv, env) if p.kernel_impl == impl])

    # `**_kw` because this stands in for `timing.time_callable`, whose
    # keyword arguments are the caller's business: the driver passes
    # `align=` for a group's barrier. The stub's job is to raise before
    # any timing happens, which no argument changes.
    def stop_before_timing(run, **_kw):
        raise driver.TooSlowToMeasure("gate passed")
    monkeypatch.setattr(driver, "time_callable", stop_before_timing)

    result = driver.run_jobs(
        [job], env=Env("2.11", None, None, None, None, libs={"torch": "2.11"}),
        date="2026-10-25", contributor="test")
    # The run is stopped right after the launch gate by a "too slow to time"
    # sentinel, which the driver records as a `timeout` attempt (4a8e081a1).
    # That one, carrying the sentinel's text, is the only failure allowed:
    # a gate refusal or a cross-check mismatch would be a second, different one.
    assert len(result.failures) == 1, result.failures
    assert "gate passed" in result.failures[0], result.failures
    assert result.unobserved == []


# The canary the H100 queue refused sglang's rope decode path on, every model
# (2026-10-26): "the launch's largest operand is float32 where the cell
# declares bfloat16 (float32[8192,128])". sglang keeps the cos/sin table fp32
# on CUDA (srt/layers/rotary_embedding/base.py:112-117, v0.5.22) and hands the
# whole of it to `apply_rope_with_cos_sin_cache_inplace` (base.py:398-410), so
# a table padded past the positions the cell reaches outgrew q and k. One head
# each and a full-width rotation is the tightest case: the table then holds
# exactly as many elements as q.
_SGL_ROPE_DECODE = _job(
    "rope", "fused_qk_rope", "decode",
    {"B": 13, "d_rope": 128, "n_heads": 1, "n_kv_heads": 1, "s_query": 1},
    [[13, 1, 1, 128], [13, 1, 128]], [[13, 1, 1, 128], [13, 1, 128]],
    dtype_env=_KV, operand_ports=_ports("q_pe", "k_pe"))

_STUB_ROPE_OPS = {}


def _stub_sglang_rope(monkeypatch, table_first):
    """`sglang.kernels.ops.attention.rope` as a dispatcher-visible op with
    sglang's signature (kernels/ops/attention/rope.py:183-224: keyword `q`,
    `k` as [tokens, heads, rope_dim], an fp32 `[max_position, rope_dim]`
    `cos_sin_cache`, `positions`, `is_neox`) and its in-place semantics —
    the launch the driver traces on the H100, run on the CPU.

    `table_first` hands the launch the table before q and k: at the canary
    the three buffers tie for widest, and which a kernel takes first is its
    own business, not a fact the gate may read (sglang's `run_rope` takes
    q, k, table; another kernel may not)."""
    import types
    import torch
    from kernel_bench.refs import _rope_rotate
    if table_first not in _STUB_ROPE_OPS:
        name = "table_first" if table_first else "q_first"
        args = (["Tensor cos_sin_cache", "Tensor(a!) q", "Tensor(b!) k"]
                if table_first else
                ["Tensor(a!) q", "Tensor(b!) k", "Tensor cos_sin_cache"])

        def rotate(q, k, cos_sin_cache, positions, is_neox):
            assert cos_sin_cache.dtype == torch.float32
            assert q.dim() == 3 and k.dim() == 3
            rd = cos_sin_cache.shape[-1]
            for x in (q, k):
                x.copy_(_rope_rotate(x.float(), positions, cos_sin_cache,
                                     rd, is_neox).to(x.dtype))

        lib = torch.library.Library("kbtest", "FRAGMENT")
        lib.define(f"sgl_apply_rope_inplace_{name}"
                   f"({', '.join(args)}, Tensor positions, bool is_neox) -> ()")
        def table_first_rotate(c, q, k, p, n):
            rotate(q, k, c, p, n)

        impl = table_first_rotate if table_first else rotate
        lib.impl(f"sgl_apply_rope_inplace_{name}", impl, "CompositeExplicitAutograd")
        _STUB_ROPE_OPS[table_first] = (
            lib, getattr(torch.ops.kbtest, f"sgl_apply_rope_inplace_{name}"))
    op = _STUB_ROPE_OPS[table_first][1]

    def apply_rope_with_cos_sin_cache_inplace(*, positions, q, k,
                                              cos_sin_cache, is_neox,
                                              fused_args=None):
        tensors = ((cos_sin_cache, q, k) if table_first
                   else (q, k, cos_sin_cache))
        op(*tensors, positions, is_neox)

    names = ("sglang", "sglang.kernels", "sglang.kernels.ops",
             "sglang.kernels.ops.attention", "sglang.kernels.ops.attention.rope")
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[names[-1]].apply_rope_with_cos_sin_cache_inplace = (
        apply_rope_with_cos_sin_cache_inplace)


# Table first is the order a gate that broke ties by argument order refused,
# so it is the case the default suite runs.
@pytest.mark.parametrize("table_first", [True, False],
                         ids=["table_first", "q_first"])
def test_sglangs_fp32_table_rope_launches_the_cell(monkeypatch, table_first):
    # The whole driver path — the oracle's inputs, the provider's build, the
    # traced call, the launch gate and the cross-check — on the sglang
    # provider: the fp32 table is not the precision the row prices, whatever
    # order the kernel takes its operands in, and the kernel's in-place q/k
    # agree with the oracle.
    import kernel_bench.driver as driver
    from kernel_bench.env import Env
    from kernel_bench.providers import all_providers, rope
    _stub_sglang_rope(monkeypatch, table_first)
    monkeypatch.setattr(rope, "_require_cuda", lambda: None)
    provider = next(p for p in all_providers()
                    if p.kernel_impl == "sgl_apply_rope_cos_sin_cache")
    monkeypatch.setattr(driver, "applicable", lambda inv, env: [provider])

    # `**_kw` because this stands in for `timing.time_callable`, whose
    # keyword arguments are the caller's business: the driver passes
    # `align=` for a group's barrier. The stub's job is to raise before
    # any timing happens, which no argument changes.
    def stop_before_timing(run, **_kw):
        raise driver.TooSlowToMeasure("gate passed")
    monkeypatch.setattr(driver, "time_callable", stop_before_timing)

    result = driver.run_jobs(
        [_SGL_ROPE_DECODE],
        env=Env("2.11", None, None, None, None,
                libs={"torch": "2.11", "sglang": "0.5.22"}),
        date="2026-10-26", contributor="test")
    assert len(result.failures) == 1, result.failures
    assert "gate passed" in result.failures[0], result.failures
    assert result.unobserved == []


# The canary GLM-4.5-Air's sglang partial-rope decode path was disabled on
# (2026-10-26, before the table followed the positions): heads of 128 rotated
# over their first 64 lanes, three of them, so q is the widest bf16 buffer
# only while the fp32 table holds the one row the position reaches.
_SGL_ROPE_PARTIAL_DECODE = _job(
    "rope_partial", "fused_qk_rope_partial", "decode",
    {"B": 1, "d_head": 128, "d_rope": 64, "n_heads": 3, "n_kv_heads": 1,
     "s_query": 1},
    [[1, 3, 1, 128], [1, 1, 1, 128]], [[1, 3, 1, 128], [1, 1, 1, 128]],
    dtype_env=_KV, operand_ports=_ports("q", "k"))


# The canary the H100 disabled it on after that fix (2026-10-27): ONE head of
# 256 lanes, 64 rotated, one token. sglang hands the kernel
# `query.view(T, -1, head_size)[..., :rotary_dim]` (srt/layers/
# rotary_embedding/base.py:397-401, v0.5.22) — a strided view over the whole
# head, as the build does — and with a single row that view's span is only
# the 64 lanes, so the gate read "64 elements where the cell's holds 256".
_SGL_ROPE_PARTIAL_ONE_HEAD = _job(
    "rope_partial", "fused_qk_rope_partial", "decode",
    {"B": 1, "d_head": 256, "d_rope": 64, "n_heads": 1, "n_kv_heads": 1,
     "s_query": 1},
    [[1, 1, 1, 256], [1, 1, 1, 256]], [[1, 1, 1, 256], [1, 1, 1, 256]],
    dtype_env=_KV, operand_ports=_ports("q", "k"))


@pytest.mark.parametrize("job", [_SGL_ROPE_PARTIAL_ONE_HEAD,
                                 _SGL_ROPE_PARTIAL_DECODE],
                         ids=["one_head", "three_heads"])
def test_sglangs_partial_rope_launches_the_cell(monkeypatch, job):
    # The same fp32 table, handed with q/k views of the rotated lanes
    # (`_sgl_rope_partial_build`): the whole driver path on the sglang
    # partial-rope provider.
    import kernel_bench.driver as driver
    from kernel_bench.env import Env
    from kernel_bench.providers import all_providers, rope
    _stub_sglang_rope(monkeypatch, table_first=True)
    monkeypatch.setattr(rope, "_require_cuda", lambda: None)
    provider = next(p for p in all_providers()
                    if p.kernel_impl == "sglang_apply_rope_partial")
    monkeypatch.setattr(driver, "applicable", lambda inv, env: [provider])

    # `**_kw` because this stands in for `timing.time_callable`, whose
    # keyword arguments are the caller's business: the driver passes
    # `align=` for a group's barrier. The stub's job is to raise before
    # any timing happens, which no argument changes.
    def stop_before_timing(run, **_kw):
        raise driver.TooSlowToMeasure("gate passed")
    monkeypatch.setattr(driver, "time_callable", stop_before_timing)

    result = driver.run_jobs(
        [job],
        env=Env("2.11", None, None, None, None,
                libs={"torch": "2.11", "sglang": "0.5.22"}),
        date="2026-10-26", contributor="test")
    assert len(result.failures) == 1, result.failures
    assert "gate passed" in result.failures[0], result.failures
    assert result.unobserved == []


def test_the_rope_table_holds_the_rows_the_positions_reach():
    # Rows past the last position are never gathered; handed to the launch,
    # they are only a buffer the gate reads as the row's precision.
    from kernel_bench.invocation import Invocation
    from kernel_bench.refs import get_ref, input_generator
    inv = Invocation.from_job(_SGL_ROPE_DECODE)
    inputs = get_ref(inv.signature).make_inputs(
        inv, "cpu", input_generator(inv, "cpu"))
    assert inputs["cos_sin_cache"].shape[0] == int(inputs["positions"].max()) + 1


# The canary gpt-oss's vLLM FA3 decode path was disabled on (2026-10-26):
# "cannot import name 'flash_attn_with_kvcache' from 'vllm.vllm_flash_attn'".
_VLLM_FA3_DECODE = _job(
    "gqa_attention", "flash_attention", "decode",
    {"B": 1, "d_head": 64, "kv_block_size": 16, "n_attn_sinks": 1,
     "n_heads": 1, "n_kv_heads": 1, "s_kv": 128, "s_query": 1,
     "s_window": 128},
    [[1, 1, 1, 64], [1, 128, 1, 64], [1, 128, 1, 64]], [1, 1, 1, 64],
    dtype_env={**_KV, "weight": "bf16"}, operand_ports=_ports("q", "k", "v"))


def _stub_vllm_flash_attn(monkeypatch):
    """`vllm.vllm_flash_attn` as v0.32.0 exports it — `flash_attn_varlen_func`
    and no `flash_attn_with_kvcache` (vllm_flash_attn/__init__.py:23-31) —
    with the entry's paged decode written out in torch: keys read through
    `block_table`, `seqused_k` of them, the window, and `s_aux` as one more
    logit a head (flash_attn_interface.py:176-410)."""
    import types
    import torch

    def flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q,
                               max_seqlen_k, cu_seqlens_k=None,
                               seqused_k=None, block_table=None,
                               causal=False, window_size=None, s_aux=None,
                               fa_version=2, **_):
        assert cu_seqlens_k is None and block_table is not None
        assert fa_version == 3
        out = []
        for b in range(block_table.shape[0]):
            n = int(seqused_k[b])
            keys = k[block_table[b]].flatten(0, 1)[:n].float()
            vals = v[block_table[b]].flatten(0, 1)[:n].float()
            rows = q[int(cu_seqlens_q[b]):int(cu_seqlens_q[b + 1])].float()
            group = rows.shape[1] // keys.shape[1]
            keys = keys.repeat_interleave(group, 1)
            vals = vals.repeat_interleave(group, 1)
            scores = torch.einsum("snd,lnd->nsl", rows, keys) * (
                rows.shape[-1] ** -0.5)
            pos = torch.arange(n) - (n - rows.shape[0])
            query = torch.arange(rows.shape[0])[:, None]
            hidden = pos[None, :] > query
            if window_size is not None:
                hidden |= pos[None, :] < query - window_size[0]
            scores = scores.masked_fill(hidden, float("-inf"))
            if s_aux is not None:
                sink = s_aux.float()[:, None, None].expand(
                    -1, rows.shape[0], 1)
                probs = torch.cat([scores, sink], -1).softmax(-1)[..., :-1]
            else:
                probs = scores.softmax(-1)
            out.append(torch.einsum("nsl,lnd->snd", probs, vals).to(q.dtype))
        return torch.cat(out)

    module = types.ModuleType("vllm.vllm_flash_attn")
    module.flash_attn_varlen_func = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", module)


def test_vllms_fa3_decode_is_the_paged_varlen_entry(monkeypatch):
    import kernel_bench.driver as driver
    from kernel_bench.env import Env
    from kernel_bench.providers import all_providers, attention
    _stub_vllm_flash_attn(monkeypatch)
    monkeypatch.setattr(attention, "_require_cuda", lambda: None)
    provider = next(p for p in all_providers()
                    if p.kernel_impl == "vllm_flash_attention_v3_kvcache")
    monkeypatch.setattr(driver, "applicable", lambda inv, env: [provider])

    # `**_kw` because this stands in for `timing.time_callable`, whose
    # keyword arguments are the caller's business: the driver passes
    # `align=` for a group's barrier. The stub's job is to raise before
    # any timing happens, which no argument changes.
    def stop_before_timing(run, **_kw):
        raise driver.TooSlowToMeasure("gate passed")
    monkeypatch.setattr(driver, "time_callable", stop_before_timing)
    result = driver.run_jobs(
        [_VLLM_FA3_DECODE],
        env=Env("2.11", None, None, None, None,
                libs={"torch": "2.11", "vllm": "0.32.0"}),
        date="2026-10-26", contributor="test")
    assert len(result.failures) == 1, result.failures
    assert "gate passed" in result.failures[0], result.failures
