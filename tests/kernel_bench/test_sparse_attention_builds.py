"""The sparse-attention launch paths the H100 disabled on 2026-10-28.

FlashMLA's sparse prefill kernel (`flash_mla_sparse_fwd`) is what vLLM and
sglang run for a bf16 latent cache, with and without Hy4's sink. Every cell
of all four FlashMLA builds failed before a number was taken, for reasons
that live between the cell and the kernel, not in the kernel:

  * the kernel returns its outputs as a LIST (a C++ `std::vector<Tensor>`),
    which a tuple-only test sliced: "list indices must be integers or
    slices, not tuple";
  * the selection was padded to 64 keys where the kernel asserts a multiple
    of 128 (`topk % (2*B_TOPK) == 0`) — every cell whose attended count is
    not a multiple of 128 (GLM-5.3-Flash's 288, Hy4's 576 and 1156);
  * the sink builds reshaped a 512-wide V output to the 576-wide query.

These run each build against a CPU stand-in for the kernel that holds the
cell to the same assertions FlashMLA makes (FlashMLA ba89a34, csrc/api/
sparse_prefill.cpp:125-126, sm90/prefill/sparse/phase1.cuh:577) and computes
what the kernel computes — the selected rows, a -1 or out-of-range index
masked, the rows past `topk_length` skipped, the sink's
`exp(lse) / (exp(lse) + exp(sink))` — so the canonical output is held to the
signature's oracle. Also here: the declarations that close what the
kernels refuse, and the paged layout FlashInfer's SM90 verify accepts.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch")

#: The sparse prefill kernel's top-k tile on SM90 and SM100 (2 * B_TOPK).
_TOPK_TILE = 128
#: Hopper's head multiple for this kernel.
_HEAD_MULTIPLE = 64


def _sparse_prefill_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None,
                        topk_length=None, out=None):
    """`flash_mla_sparse_fwd` on the CPU: its checks, its math, its return."""
    import torch
    s_q, h_q, d_qk = q.shape
    s_kv = kv.shape[0]
    topk = indices.shape[-1]
    if d_qk not in (576, 512):
        raise RuntimeError(f"Invalid d_qk: {d_qk}")
    if d_v != 512:
        raise RuntimeError(f"Invalid d_v{d_v}")
    assert q.dtype == kv.dtype == torch.bfloat16
    assert indices.dtype == torch.int32 and indices.shape[:2] == (s_q, 1)
    assert h_q % _HEAD_MULTIPLE == 0
    if topk % _TOPK_TILE:
        raise RuntimeError(
            "Assertion `params.topk % (2*B_TOPK) == 0` failed")
    if topk_length is not None:
        assert topk_length.dtype == torch.int32 and topk_length.shape == (s_q,)
    o = torch.zeros(s_q, h_q, d_v, dtype=torch.float32)
    lse = torch.full((s_q, h_q), float("-inf"))
    for t in range(s_q):
        reach = topk if topk_length is None else int(topk_length[t])
        idx = indices[t, 0, :reach].long()
        idx = idx[(idx >= 0) & (idx < s_kv)]
        if idx.numel() == 0:
            continue
        rows = kv[idx, 0].float()                             # [n, d_qk]
        scores = q[t].float() @ rows.T * sm_scale            # [h_q, n]
        lse[t] = torch.logsumexp(scores, dim=-1)
        o[t] = scores.softmax(dim=-1) @ rows[:, :d_v]
    if attn_sink is not None:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (h_q,)
        o = o * (1.0 / (1.0 + torch.exp(attn_sink[None, :] - lse))
                 ).unsqueeze(-1)
    return [o.to(torch.bfloat16), lse.clone(), lse]


def _stub_modules(monkeypatch, *dotted):
    for name in dotted:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            mod = ".".join(parts[:i])
            if mod not in sys.modules or i == len(parts):
                monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
        sys.modules[name].flash_mla_sparse_fwd = _sparse_prefill_fwd


def _run_build(kernel_impl, signature, op, dims, dtype_env, monkeypatch):
    """The provider's build on the cell, as the driver drives it, against
    the stand-in; returns (canonical output, the signature's oracle)."""
    from kernel_bench.env import Env
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers, attention
    from kernel_bench.refs import get_ref, input_generator
    monkeypatch.setattr(attention, "_require_cuda", lambda: None)
    _stub_modules(monkeypatch, "flash_mla", "sgl_kernel.flash_mla",
                  "vllm.v1.attention.ops.flashmla")
    import json
    inv = Invocation.from_job({
        "signature": signature, "op": op, "phase": "prefill", "tp": 1,
        "dims": dims, "dtype_env": dtype_env,
        "physical_key": [signature, json.dumps(dims, sort_keys=True),
                         json.dumps(dtype_env, sort_keys=True), "prefill"]})
    spec = get_ref(inv.signature)
    inputs = spec.make_inputs(inv, "cpu", input_generator(inv, "cpu"))
    want = spec.reference(inputs)
    provider = next(p for p in all_providers() if p.kernel_impl == kernel_impl)
    assert provider.serves.accepts(inv)
    launched = provider.convert_inputs(inv, dict(inputs))
    case = provider.build(inv, Env("2.11", None, None, None, None,
                                   libs={"torch": "2.11"}), launched)
    got = provider.wrap_output(inv, case.to_canonical)(case.run())
    return got, want


def _assert_meets_the_oracle(got, want):
    import torch
    assert tuple(got.shape) == tuple(want.shape)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0)
    assert float(cos) > 0.999, float(cos)
    assert torch.allclose(got.float(), want.float(), atol=3e-2, rtol=3e-2)


def test_a_launch_returning_a_list_is_read_at_its_first_tensor():
    # A C++ entry point returning `std::vector<Tensor>` hands Python a list.
    import torch
    from kernel_bench.providers.attention import _primary
    out, lse = torch.zeros(2, 3), torch.ones(2)
    assert _primary([out, lse]) is out
    assert _primary((out, lse)) is out
    assert _primary(out) is out


#: Two requests of three query tokens over 160 cached latent rows, all of
#: them selected: 160 is a multiple of 64 and not of 128, the class every
#: refused cell was in (GLM-5.3-Flash 288, Hy4 576 / 1156).
_DSA_DIMS = {"B": 2, "s_query": 3, "s_kv": 160, "n_topk_keys": 160,
             "n_heads": 2, "d_head": 576, "d_v_head": 512,
             "kv_block_size": 64}
_BF16 = {"activation": "bf16", "kv": "bf16", "output": "bf16"}


def test_flashmla_sparse_hands_the_kernel_its_tile_and_reads_its_list(
        monkeypatch):
    got, want = _run_build("flashmla_sparse", "mla_sparse_attention",
                           "flash_dsa", _DSA_DIMS, _BF16, monkeypatch)
    _assert_meets_the_oracle(got, want)


@pytest.mark.parametrize("kernel_impl", ["sglang_flashmla_sparse_sink",
                                         "vllm_flashmla_sparse_sink_bf16"])
def test_the_sink_builds_hand_the_kernel_its_tile_and_read_v_wide(
        kernel_impl, monkeypatch):
    got, want = _run_build(kernel_impl, "mla_sparse_sink_attention",
                           "flash_dsa_sink", _DSA_DIMS,
                           {**_BF16, "weight": "fp32"}, monkeypatch)
    _assert_meets_the_oracle(got, want)


def test_the_selection_is_tiled_and_its_valid_count_handed_over():
    # Serving hands the kernel `index_topk` columns, -1 past the selection,
    # and the valid count beside them; the cell carries only the attended
    # count, so the width is that count on the entry point's own tile.
    import torch
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers.attention import _flashmla_topk_indices
    inv = Invocation(op="flash_dsa", signature="mla_sparse_attention",
                     phase="decode", dims={"n_topk_keys": 288}, tp=1,
                     dtype_env=_BF16)
    idx = torch.arange(288, dtype=torch.int32).view(1, 1, 288)
    base = torch.zeros(1, 1, 1, dtype=torch.int32)
    for tile, width in ((128, 384), (64, 320)):
        out, valid = _flashmla_topk_indices(inv, idx, base, 1, tile)
        assert out.shape == (1, width)
        assert torch.equal(out[0, :288], idx.view(-1))
        assert (out[0, 288:] == -1).all()
        assert valid.tolist() == [288]


def _named(kernel_impl):
    from kernel_bench.providers import all_providers
    return next(p for p in all_providers() if p.kernel_impl == kernel_impl)


@pytest.mark.parametrize("d_head, d_v_head, served", [
    (256, 256, False), (192, 192, False), (576, 256, False),
    (576, 512, True), (512, 512, True)])
def test_flashmla_sparse_declares_the_widths_it_instantiates(
        d_head, d_v_head, served):
    from kernel_bench.invocation import Invocation
    inv = Invocation(op="flash_dsa", signature="mla_sparse_attention",
                     phase="decode",
                     dims={**_DSA_DIMS, "d_head": d_head,
                           "d_v_head": d_v_head}, tp=1, dtype_env=_BF16)
    assert _named("flashmla_sparse").serves.accepts(inv) is served


@pytest.mark.parametrize("d_rope, served", [
    (192, False), (96, False), (64, True), (128, True), (256, True),
    (512, True)])
def test_flashinfer_rope_serves_the_widths_its_own_kernel_runs(
        d_rope, served):
    from kernel_bench.invocation import Invocation
    inv = Invocation(op="fused_qk_rope", signature="rope", phase="prefill",
                     dims={"B": 1, "s_query": 16, "n_heads": 2,
                           "n_kv_heads": 1, "d_rope": d_rope}, tp=1,
                     dtype_env={"activation": "bf16", "output": "bf16"})
    assert _named("flashinfer_rope").serves.accepts(inv) is served


def test_a_narrower_v_is_paged_as_one_pool_of_k_then_v_rows():
    # FlashInfer's SM90 paged prefill takes K and V at one page stride and
    # one token stride (batch_prefill_sm90.cu:233-237, 0.7.2.post1), and
    # vLLM stores an asymmetric-V cache as one pool of K-then-V rows. Two
    # caches of their own widths were refused on every MiMo verify cell.
    import torch
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers.attention import (
        _FLASHINFER_BLOCK_SIZES, _paged_cache)
    B, Lq, Nk, D, Dv = 2, 37, 2, 192, 128
    inv = Invocation(op="flash_attention", signature="gqa_attention",
                     phase="decode",
                     dims={"B": B, "s_kv": Lq, "s_query": 2, "n_heads": 8,
                           "n_kv_heads": Nk, "d_head": D, "d_v_head": Dv,
                           "kv_block_size": 16}, tp=1, dtype_env=_BF16)
    k = torch.randn(B, Lq, Nk, D, dtype=torch.bfloat16)
    v = torch.randn(B, Lq, Nk, Dv, dtype=torch.bfloat16)
    k_cache, v_cache, _indptr, indices, last, page = _paged_cache(
        inv, k, v, B, _FLASHINFER_BLOCK_SIZES)
    assert k_cache.shape[-1] == D and v_cache.shape[-1] == Dv
    assert k_cache.stride()[:3] == v_cache.stride()[:3]
    table = indices.view(B, -1)
    for b in range(B):
        rows_k = k_cache[table[b]].reshape(-1, Nk, D)[:Lq]
        rows_v = v_cache[table[b]].reshape(-1, Nk, Dv)[:Lq]
        assert torch.equal(rows_k, k[b]) and torch.equal(rows_v, v[b])
    assert int(last[0]) == Lq - (table.shape[1] - 1) * page


@pytest.mark.parametrize("kernel_impl", [
    "sglang_flashmla_sparse_sink", "vllm_flashmla_sparse_sink_bf16",
    "vllm_flashmla_sparse_sink_fp8_e4m3", "flashmla_sparse",
    "flashmla_sparse_fp8"])
def test_flashmlas_sparse_kernels_are_offered_on_hopper_and_up_only(
        kernel_impl):
    """FlashMLA's sparse entry points refuse an Ada card at the first call
    ("only supported on SM90a and SM100f"), which is what the RTX 4090's
    vLLM and sglang sink rows recorded as `error`. Declined there instead,
    so a purged cell is a declared gap, not a canary that fails every
    pass."""
    from kernel_bench.env import Env
    provider = _named(kernel_impl)
    libs = {provider.import_name: "1"}
    ada = Env("2.11", "cuda", None, None, None, sm="sm_89", libs=libs)
    hopper = Env("2.11", "cuda", None, None, None, sm="sm_90", libs=libs)
    assert not provider.available(ada)
    assert provider.available(hopper)
