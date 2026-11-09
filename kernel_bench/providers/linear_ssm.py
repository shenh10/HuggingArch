"""Linear-attention / SSM providers — the kernels vLLM and sglang dispatch.

Four families, each a PAIR of kernels rather than one, because a recurrence has
no single form: a full sequence is processed chunked (matmul-shaped, reusing the
tensor cores), a single step is processed recurrently (memory-bound, one state
read/write). Measuring the chunked kernel at decode shapes measures nothing the
model runs.

  gated_delta_rule  Gated DeltaNet (Qwen3-Next). ONE decay scalar per head.
                    prefill `chunk_gated_delta_rule`
                    decode  `fused_recurrent_gated_delta_rule`
  kda               Kimi Delta Attention (Kimi-K3's linear layers). Decay VECTOR
                    per key channel — a separate kernel, not a flag on the pair
                    above: `chunk_kda` / `fused_recurrent_kda`.
  gla_simple_recurrence  FLA simple_gla / Lightning-2 (Ling-2.6). One constant
                    decay per head, no β. FLA `chunk_simple_gla` /
                    `fused_recurrent_simple_gla`; the serving kernels are
                    vLLM's lightning pair and sglang's `seg_la_fwd`.
  causal_conv1d     the depthwise short conv every one of them runs on q/k/v
                    first. `causal_conv1d_fn` over a packed sequence,
                    `causal_conv1d_update` against a conv-state ring buffer.
  mamba2_scan       Mamba-2 selective scan (Nemotron-H / Zamba2).

Both frameworks VENDOR flash-linear-attention rather than depending on the pip
package (`vllm.third_party.flash_linear_attention.ops` from vLLM v0.30,
`sglang.kernels.ops.attention.fla` from sglang v0.5.21 — the older
`vllm.model_executor.layers.fla.ops` / `sglang.srt.layers.attention.fla` paths
are gone, and importing them made every cell of these kernels fail at build),
so those copies — not upstream fla — are what a deployment actually runs. Each is registered as its own library; three
independent implementations of one recurrence is what the envelope is for.

FlashInfer's `gdn_prefill` is a fourth GDN prefill backend, and on Hopper it is
the DEFAULT one: `ChunkGatedDeltaRule` picks it whenever
`gdn_prefill_backend=auto` and the device is capability 9.0, so on an H100 the
Triton kernel is the fallback, not the kernel.

Layout follows the contract the kernels document and the frameworks call them
with — `[B, T, H, D]`, decay in log space, state `[N, HV, V, K]`. fp32 inputs are
rejected by the kernels themselves ("please use bfloat16"), so bf16 is the floor
for the timed call and only the oracle runs in fp32.
"""
from __future__ import annotations

from ..env import Env
from ..invocation import Invocation
from ..refs import SCAN_MAX_STEPS
from . import TORCH_FLOAT, BenchCase, KernelSignature, Launch, Provider, register
from .attention import _require_cuda, _torch_dtype

# The delta-rule kernels serve bf16 alone, and each says so at its own
# declaration (`activation=("bf16",)`): fp32 is refused by the kernels
# themselves ("please use bfloat16"; `_delta_args` narrows it), and every
# row of the Triton/CUTLASS builds here was measured at bf16 — no fp16
# evidence.


def _declares(fn):
    """The parameter names `fn` declares, or None when it has no readable
    signature (a C entry point). None is "cannot ask", which is not the same
    answer as "declares nothing" — every caller here has to tell them apart.
    """
    import inspect
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def _accepted(fn, **kwargs):
    """Keep only the keywords `fn` actually declares.

    The three copies of this kernel are not the same function: vLLM and sglang
    added `inplace_final_state` to their vendored versions for serving, and
    upstream flash-linear-attention 0.5.2 has no such parameter. Passing it
    upstream is a TypeError; omitting it where it exists changes what is
    measured. So the adapter asks the callee rather than assuming the forks
    agree — the difference between them is a fact to read, not to average over.
    """
    allowed = _declares(fn)
    if allowed is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in allowed}


def _initial_state_args(fn, state, tokens: int = 1):
    """The recurrent state as the callee's own serving call site hands it over.

    `_accepted` settles the keywords the forks spell differently, but this one
    the forks hand over in two different SHAPES, so filtering cannot settle it:

      sglang v0.5.22 takes the whole state POOL plus one slot index per
      sequence. Its GDN backend passes exactly that — `initial_state=
      ssm_states` and `initial_state_indices=cache_indices`, built at
      `srt/layers/attention/linear/kernels/gdn_triton.py:183-184` and handed
      over by the call at `:203-215` (`TritonGDNKernel.extend`, that class at
      `:40`, the chunked prefill) — and `chunk_kda` declares the
      same pair (`kernels/ops/attention/fla/kda.py:1281-1282`). The index is
      NOT optional: both entry points pass it straight through to a `_fwd`
      that declares it without a default (`chunk.py:136` → `:262` → `:44`;
      `kda.py:1274` → `:1155`), and the Triton kernel dereferences it
      unconditionally — `tl.load(initial_state_indices + i_n)` at
      `chunk_delta_h.py:126`, ABOVE the `USE_INITIAL_STATE` guards that
      protect every other state access. So the `= None` default on the public
      entry advertises a path nobody runs: omitting it is a Triton
      CompilationError ("'NoneType' object has no attribute 'type'"), not a
      cheaper call, and serving never omits it.

      vLLM v0.32.0's CHUNK copy has no such parameter at all
      (`third_party/flash_linear_attention/ops/chunk.py:138-152`): its mixer
      gathers the slots itself — `initial_state =
      ssm_state[prefill_state_indices]`, a dense `[N, H, V, K]` at
      `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1599` — and the
      kernel guards every state read on `USE_INITIAL_STATE` (vLLM's OWN copy
      of the h step, `third_party/flash_linear_attention/ops/chunk_delta_h.py`
      :107 and :123 — not the sglang file of that name cited above).
      Upstream fla 0.5.2 is that shape too.
      The two frameworks genuinely differ here; the adapter must not paper
      over it, so it asks which, and there is no list of providers to keep in
      step because the signature already says.

      vLLM v0.32.0's fused_recurrent DECODE copy is a third spelling:
      `ssm_state_indices`, and per TOKEN, not per sequence — the kernel reads
      `ssm_state_indices + i_n * stride_indices_seq + i_t` inside the token
      loop (the INPLACE_FINAL_STATE write of each token's state, spec
      decoding's per-token slots), and that block dereferences it
      unconditionally, so the same None default is the same
      CompilationError ('NoneType' object has no attribute 'type' at
      `tl.load(ssm_state_indices + …)` — the error that kept
      `vllm_fused_recurrent_kda`'s canary down on every pass while the fla /
      sglang / packed-decode rows measured fine). Its index 0 is the padding
      row too (`if state_idx <= 0: return`, NULL_BLOCK_ID). A 1-D `[N]` index
      vector is wrong for `T > 1`: the token loop reads past its end
      (illegal memory access, measured). So the vLLM spelling gets the pool
      plus an `[N, T]` index whose every column names the sequence's slot.

    `B + 1` slots with the sequences at 1..B is not a new cell dimension, it
    is sglang's own pool layout read off the allocator: row 0 is the padding
    row a CUDA-graph padded batch defaults to, so real sequences start at 1
    (`srt/mem_cache/memory_pool.py:398-400` and `:407`
    `free_slots = list(range(1, self._alloc_size))`; the temporal state pool
    at `:748` is `size + 1` deep). Depth beyond the indexed rows costs the
    kernel nothing — it touches only the rows the indices name — which is why
    the decode providers below build the same `B + 1` pool from the same `B`.
    """
    import torch
    declared = _declares(fn)
    if not declared:
        return {"initial_state": state}
    n = state.shape[0]
    if "initial_state_indices" in declared:
        pool = state.new_zeros((n + 1, *state.shape[1:]))
        pool[1:] = state
        return {"initial_state": pool,
                "initial_state_indices": torch.arange(
                    1, n + 1, dtype=torch.int32, device=state.device)}
    if "ssm_state_indices" in declared:
        pool = state.new_zeros((n + 1, *state.shape[1:]))
        pool[1:] = state
        idx = torch.arange(1, n + 1, dtype=torch.int32, device=state.device)
        return {"initial_state": pool,
                "ssm_state_indices": idx[:, None].expand(n, tokens)
                                          .contiguous()}
    return {"initial_state": state}


def _delta_args(inv: Invocation, inputs):
    """Canonical delta-rule inputs at the kernel's dtype. The decay stays fp32:
    it is a log-space quantity the kernels cumsum internally."""
    import torch
    dt = _torch_dtype(inv.dtype)
    if dt is torch.float32:
        dt = torch.bfloat16
    # The recurrent state stays fp32, like the decay above. It is carried across
    # every chunk, so it accumulates where the per-step tensors do not — which
    # is why fla keeps it there and why `chunk_kda` asserts on anything else
    # (`initial_state must be in float32`). Cast to the kernel dtype it was a
    # declaration the kernel refuses, and the cell was lost to an AssertionError.
    return (inputs["q"].to(dt), inputs["k"].to(dt), inputs["v"].to(dt),
            inputs["g"].float(), inputs["beta"].to(dt),
            inputs["initial_state"].float())


# ---------------------------------------------------------------------------
# Delta-rule kernels. Gated DeltaNet and KDA share an entry-point signature —
# `(q, k, v, g, beta, initial_state, …)` — and differ in the rank of `g`, so one
# adapter serves both families and only the import differs.
# ---------------------------------------------------------------------------

def _first(out):
    """The output tensor of an entry point whose return ARITY differs between
    the forks: sglang v0.5.22's `chunk_gated_delta_rule` returns `(o, None, h)`
    (`chunk.py:269`) while its `chunk_kda` returns a bare `o` unless
    `output_intermediate_states` is set (`kda.py:1274` → `chunk_kda_fwd`'s
    `return o` at `kda.py:1271` — NOT the `return o, h` one line above it at
    `:1269`, which is the two-value branch this is distinguishing itself
    from). Indexing `[0]` unconditionally takes a BATCH ROW off the bare
    one and bills the cell for it. Same rule `_fla_gla_build` uses
    below, for the same reason."""
    return out[0] if isinstance(out, tuple) else out


def _chunk_build(entry):
    def build(inv, env, inputs) -> BenchCase:
        _require_cuda()
        fn = entry()
        q, k, v, g, beta, state = _delta_args(inv, inputs)
        # Resolved ONCE, outside the timed call: the state pool
        # `_initial_state_args` may build is a `[B+1, HV, V, K]` allocation
        # plus a copy, and a server allocates its pool at startup, not per
        # launch.
        #
        # `use_qk_l2norm_in_kernel=True` folds the q/k normalization into this
        # kernel, so it is part of the work being measured, and the oracle
        # normalizes too — the cross-check compares the same function rather
        # than a cheaper one. It used to say here that every production call
        # site passes True. That is no longer true of vLLM: at v0.32.0 the
        # norm has moved into the earlier launch (`rearrange_mixed_qkv(
        # apply_l2norm=True)`, qwen_gdn_linear_attn.py:1523) and the chunk
        # call passes `use_qk_l2norm_in_kernel=False` (:1615), while sglang
        # still passes True (gdn_triton.py:212). So the two frameworks now
        # draw this boundary differently, and the vLLM row measured here is
        # one launch WIDER than vLLM's own. Left as it is on purpose: moving
        # it would change what every vLLM row on this signature measures,
        # which is a decision about the span, not a bug in the adapter.
        kwargs = _accepted(
            fn, q=q, k=k, v=v, g=g, beta=beta,
            output_final_state=True, use_qk_l2norm_in_kernel=True,
            **_initial_state_args(fn, state))
        return BenchCase(
            run=lambda: _first(fn(**kwargs)),
            to_canonical=lambda out: out.float(),
        )
    return build


def _recurrent_build(entry):
    def build(inv, env, inputs) -> BenchCase:
        _require_cuda()
        fn = entry()
        q, k, v, g, beta, state = _delta_args(inv, inputs)
        # Production semantics on both knobs: the state is written in place
        # (no extra [N,HV,V,K] allocation per step — 200MB at a large batch),
        # and the q/k norm is inside the kernel. The driver cross-checks
        # BEFORE timing, so the in-place state mutation across repeats
        # changes values, never the shape or the work.
        kwargs = _accepted(
            fn, q=q, k=k, v=v, g=g, beta=beta,
            inplace_final_state=True, use_qk_l2norm_in_kernel=True,
            **_initial_state_args(fn, state, tokens=q.shape[1]))
        return BenchCase(
            run=lambda: _first(fn(**kwargs)),
            to_canonical=lambda out: out.float(),
        )
    return build


def _fla_chunk_gdn():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return chunk_gated_delta_rule


def _fla_recurrent_gdn():
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    return fused_recurrent_gated_delta_rule


def _relocated(name: str, *modules: str):
    """`name` from the first of `modules` that exists — ONE kernel whose
    vendored copy moved between releases. The frameworks' FLA copies moved in vLLM v0.30
    (`model_executor.layers.fla.ops` → `third_party.flash_linear_attention.ops`)
    and sglang v0.5.21 (`srt.layers.attention.fla` → `kernels.ops.attention.fla`);
    the newer path is the release the bindings describe, the older one is what
    a box pinned to an earlier release (the H100 box's vLLM 0.11) imports.
    Neither present is an ImportError, as it always was."""
    import importlib
    for module in modules:
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as e:
            if e.name is not None and module.startswith(e.name):
                continue                      # this location is absent
            raise
        return getattr(mod, name)
    raise ImportError(f"{name} is in none of {modules}")


def _fla(library: str, submodule: str, name: str):
    """`name` from a framework's vendored flash-linear-attention copy.

    Where each framework keeps that copy — the location the audited release
    uses first, the one a box pinned to an earlier release imports second
    (`_relocated`) — is stated here once, for every entry point reached
    through it. A library this does not know raises rather than falling
    through to upstream FLA, which is a different kernel.
    """
    roots = {
        "vllm": ("vllm.third_party.flash_linear_attention.ops",
                 "vllm.model_executor.layers.fla.ops"),
        "sglang": ("sglang.kernels.ops.attention.fla",
                   "sglang.srt.layers.attention.fla"),
    }[library]
    return _relocated(name, *(f"{root}.{submodule}" for root in roots))


def _vllm_chunk_gdn():
    return _fla("vllm", "chunk", "chunk_gated_delta_rule")


def _vllm_recurrent_gdn():
    return _fla("vllm", "fused_recurrent", "fused_recurrent_gated_delta_rule")


def _sglang_chunk_gdn():
    return _fla("sglang", "chunk", "chunk_gated_delta_rule")


def _sglang_recurrent_gdn():
    return _fla("sglang", "fused_recurrent",
                "fused_recurrent_gated_delta_rule")


def _vllm_chunk_kda():
    return _fla("vllm", "kda", "chunk_kda")


def _vllm_recurrent_kda():
    return _fla("vllm", "kda", "fused_recurrent_kda")


def _sglang_chunk_kda():
    return _fla("sglang", "kda", "chunk_kda")


def _sglang_recurrent_kda():
    return _fla("sglang", "kda", "fused_recurrent_kda")


def _pin_fla_triton():
    """Keep upstream FLA on its OWN Triton kernel.

    From flash-linear-attention 0.5, `chunk_kda` auto-dispatches into FlashKDA
    when that package is importable — so a row labelled `fla_chunk_kda` could be
    timing Moonshot's CUTLASS kernel instead, and the corpus would attribute one
    implementation's speed to another. `FLA_FLASH_KDA=0` is FLA's own switch for
    exactly this. FlashKDA is measured on its own row, by its own provider.
    """
    import os
    os.environ.setdefault("FLA_FLASH_KDA", "0")


def _fla_chunk_kda():
    _pin_fla_triton()
    from fla.ops.kda import chunk_kda
    return chunk_kda


def _fla_recurrent_kda():
    _pin_fla_triton()
    from fla.ops.kda import fused_recurrent_kda
    return fused_recurrent_kda


_DELTA_LIBS = (
    # library, import_name, GDN chunk/recurrent, KDA chunk/recurrent.
    # How each of the three gets INSTALLED — including why `fla` is pinned to a
    # git tag — is in `_libraries.py`, once per library rather than once per
    # kernel.
    ("fla", "fla.ops",
     _fla_chunk_gdn, _fla_recurrent_gdn, _fla_chunk_kda, _fla_recurrent_kda),
    ("vllm", "vllm",
     _vllm_chunk_gdn, _vllm_recurrent_gdn, _vllm_chunk_kda, _vllm_recurrent_kda),
    ("sglang", "sglang",
     _sglang_chunk_gdn, _sglang_recurrent_gdn, _sglang_chunk_kda,
     _sglang_recurrent_kda),
)

for _lib, _imp, _gc, _gr, _kc, _kr in _DELTA_LIBS:
    register(Provider(
        library=_lib, signature="gated_delta_rule",
        kernel_impl=f"{_lib}_chunk_gated_delta_rule",
        import_name=_imp,
        serves=KernelSignature(takes_state=True, signature="gated_delta_rule", phases=("prefill",),
                               activation=("bf16",)),
        build=_chunk_build(_gc),
    ))
    register(Provider(
        library=_lib, signature="gated_delta_rule",
        kernel_impl=f"{_lib}_fused_recurrent_gated_delta_rule",
        import_name=_imp,
        serves=KernelSignature(takes_state=True, signature="gated_delta_rule", phases=("decode",),
                               activation=("bf16",)),
        build=_recurrent_build(_gr),
    ))
    register(Provider(
        library=_lib, signature="kda", kernel_impl=f"{_lib}_chunk_kda",
        import_name=_imp,
        serves=KernelSignature(takes_state=True, signature="kda", phases=("prefill",),
                               activation=("bf16",)),
        build=_chunk_build(_kc),
    ))
    register(Provider(
        library=_lib, signature="kda",
        kernel_impl=f"{_lib}_fused_recurrent_kda",
        import_name=_imp,
        serves=KernelSignature(takes_state=True, signature="kda", phases=("decode",),
                               activation=("bf16",)),
        build=_recurrent_build(_kr),
    ))


def _flashinfer_gdn_build(inv, env, inputs) -> BenchCase:
    """FlashInfer GDN prefill — vLLM's default on Hopper. It takes the decay
    ALREADY EXPONENTIATED and fp32 gate/beta/state, packed varlen with the batch
    folded into the token axis, exactly as `gdn_linear_attn`'s
    `fi_chunk_gated_delta_rule` prepares it."""
    _require_cuda()
    import torch
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk

    q, k, v, g, beta, state = _delta_args(inv, inputs)
    B = int(inv.dims["B"])
    T = int(inv.dims["s_query"])
    st = state.float()
    cu = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)

    def _canon(out):
        o = out[0] if isinstance(out, tuple) else out
        return o.float()

    return BenchCase(
        run=lambda: fi_chunk(q=q, k=k, v=v, g=torch.exp(g.float()),
                             beta=beta.float(),
                             initial_state=st, output_final_state=True,
                             cu_seqlens=cu),
        to_canonical=_canon,
    )


register(Provider(
    library="flashinfer", signature="gated_delta_rule",
    kernel_impl="flashinfer_gdn_prefill",
    import_name="flashinfer",
    serves=KernelSignature(takes_state=True, signature="gated_delta_rule", phases=("prefill",),
                           activation=("bf16",)),
    build=_flashinfer_gdn_build,
    launch_inputs=(
        ("q", ((0, 1), 2, 3)),
        ("k", ((0, 1), 2, 3)),
        ("v", ((0, 1), 2, 3)),
        ("g", ((0, 1), 2)),
        ("beta", ((0, 1), 2)),
    ),
    launch_output="v",
))


# ---------------------------------------------------------------------------
# FlashKDA — Moonshot's CUTLASS implementation of the KDA prefill kernel, and
# the reason `kda` needs a provider that is not Triton: it is the vendor's own
# kernel for its own attention, and from flash-linear-attention 0.5 it takes
# over `chunk_kda` transparently. Registered separately so the corpus attributes
# each implementation's speed to itself.
#
#   flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
#                 initial_state=None, final_state=None, cu_seqlens=None)
#
# Its boundary is WIDER than `chunk_kda`'s: the gate is computed inside from the
# pre-activation projection `g [B,T,H,K]` plus `A_log [H]` / `dt_bias [H,K]`,
# where Triton's entry takes the decay already in log space. It also takes beta
# as logits and sigmoids internally, and writes into a caller-owned buffer.
# Fusing the gate adds O(B·T·H·K) elementwise work to an O(B·T·H·K·V)
# recurrence, so the two boundaries are the same measurement to within 1/V.
# SM90+ and CUDA 12.9+, so it does not apply on Ada.
# ---------------------------------------------------------------------------

def _flash_kda_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    import flash_kda

    q, k, v, g_log, beta, state = _delta_args(inv, inputs)
    B, T, H, K = q.shape
    V = v.shape[3]
    dev = q.device
    # Zeroed: an output the harness allocates and the kernel fills in part is
    # compared in whole, so uninitialized memory would enter the check.
    out = torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=dev)

    # Invert the kernel's own gate so it reproduces the canonical decay, and the
    # cross-check compares like with like. With A_log=0 and dt_bias=0 the gate is
    # g = −softplus(g_raw), so g_raw = log(exp(−g_log) − 1). The clamp keeps the
    # inverse finite where the canonical decay approaches zero.
    A_log = torch.zeros(H, dtype=torch.float32, device=dev)
    dt_bias = torch.zeros(H, K, dtype=torch.float32, device=dev)
    target = (-g_log.float()).clamp(min=1e-4)
    g_raw = torch.log(torch.expm1(target)).to(torch.bfloat16)
    # beta arrives already gated (0..1); this kernel sigmoids internally.
    beta_logit = torch.logit(beta.float().clamp(1e-4, 1 - 1e-4)).to(torch.bfloat16)
    scale = float(K) ** -0.5
    # The recurrent state at fp32, as the KDA cell caches and bills it (the
    # components declare it F32); flash_kda takes either width for both
    # buffers (flash_kda/__init__.py `fwd`, "bf16 or fp32"). Narrowed to
    # bf16 the launch moved half the state's bytes the row prices. A copy:
    # the input is shared by every provider of the cell.
    st = state.float().clone()
    final = torch.empty_like(st)

    def _run():
        flash_kda.fwd(q.to(torch.bfloat16), k.to(torch.bfloat16),
                      v.to(torch.bfloat16), g_raw, beta_logit, scale, out,
                      A_log, dt_bias, -5.0,
                      initial_state=st, final_state=final)
        return out

    return BenchCase(run=_run, to_canonical=lambda o: o.float())


register(Provider(
    library="flash_kda", signature="kda", kernel_impl="flash_kda_cutlass_fwd",
    import_name="flash_kda",
    # The kernel pins both head dims to 128; a model outside that is not a
    # miss to explain, it is a shape this implementation does not serve.
    # bf16 only: the build hands the kernel bf16 operands and a bf16 `out`.
    serves=KernelSignature(takes_state=True, signature="kda", phases=("prefill",),
                           activation=("bf16",),
                           enum=(("d_lin_k_head", (128,)),
                                 ("d_lin_v_head", (128,)))),
    build=_flash_kda_build,
))


# ---------------------------------------------------------------------------
# fused_recurrent_gated_delta_rule — the Gated DeltaNet decode step with its
# gates and q/k norms folded in (the fused span of that name). Both frameworks
# launch `fused_recurrent_gated_delta_rule_packed_decode` on the packed
# post-conv projection `mixed_qkv [B, 2·H·K + HV·V]`, raw `a`/`b [B, HV]`,
# `A_log`/`dt_bias [HV]` and the state pool `[slots, HV, V, K]`:
#   vLLM   third_party/flash_linear_attention/ops/fused_recurrent.py:343,
#          called at mamba/gdn/qwen_gdn_linear_attn.py:1766 (v0.32.0)
#   sglang kernels/ops/attention/fla/fused_recurrent.py:269, via
#          srt/layers/attention/linear/kernels/gdn_triton.py `packed_decode`
#          (:46) from srt/layers/attention/linear/gdn_backend.py:868 (v0.5.22)
# One token per sequence is the kernel's contract (`out [B, 1, HV, V]`), so a
# verify window (s_query > 1, speculative decode — vLLM's
# `fused_sigmoid_gating_delta_rule_update` there) is not this provider's.
# The state pool is B + 1 slots with the sequences at 1..B: vLLM skips slot 0
# (its null block). fp32 state, as the graph declares it
# (gated_delta_net_attention.yaml `recurrent_state` dtype F32).
# ---------------------------------------------------------------------------

def _gdn_packed_decode_build(entry):
    def build(inv, env, inputs) -> BenchCase:
        _require_cuda()
        import torch
        fn = entry()
        dt = _torch_dtype(inv.dtype)
        if dt is torch.float32:
            dt = torch.bfloat16
        q, k, v = (inputs[n].to(dt) for n in ("q", "k", "v"))
        B, _T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        mixed_qkv = torch.cat([q.reshape(B, H * K), k.reshape(B, H * K),
                               v.reshape(B, HV * V)], dim=-1).contiguous()
        a = inputs["a"].to(dt).reshape(B, HV).contiguous()
        b = inputs["b"].to(dt).reshape(B, HV).contiguous()
        # The parameters as the checkpoint stores them (the cell's weight
        # dtype); the kernel loads them to fp32 itself.
        A_log = inputs["A_log"].contiguous()
        dt_bias = inputs["dt_bias"].contiguous()
        state = torch.zeros(B + 1, HV, V, K, dtype=torch.float32,
                            device=q.device)
        state[1:] = inputs["initial_state"].float()
        slots = torch.arange(1, B + 1, dtype=torch.int32, device=q.device)
        out = torch.empty(B, 1, HV, V, dtype=dt, device=q.device)

        def run():
            fn(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
               scale=float(K) ** -0.5, initial_state=state, out=out,
               ssm_state_indices=slots, use_qk_l2norm_in_kernel=True)
            return out
        # `out [B, 1, HV, V]` is the canonical `[B, s_query=1, HV, V]`.
        return BenchCase(run=run, to_canonical=lambda o: o.float())
    return build


def _vllm_gdn_packed_decode():
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    return fused_recurrent_gated_delta_rule_packed_decode


def _sglang_gdn_packed_decode():
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    return fused_recurrent_gated_delta_rule_packed_decode


def _torch_gdn_decode_build(inv, env, inputs) -> BenchCase:
    """The span's naive ops, eager: the gates, then the recurrence, carried
    in fp32 like the state it updates (the einsums take one dtype)."""
    from ..refs import _delta_rule_scan, gdn_gates
    q, k, v = (inputs[n].float() for n in ("q", "k", "v"))
    # A copy, outside the timed call: the step writes its state back, as
    # the serving launch writes its slot, and the inputs are shared by every
    # provider of the cell.
    state = inputs["initial_state"].float().clone()

    def run():
        g, beta = gdn_gates(inputs["a"], inputs["b"], inputs["A_log"],
                            inputs["dt_bias"])
        return _delta_rule_scan(q, k, v, g, beta, state, final_state=state)
    return BenchCase(run=run, to_canonical=lambda o: o.float())


_ONE_TOKEN = (("s_query", (1,)),)

for _lib, _imp, _entry in (("vllm", "vllm", _vllm_gdn_packed_decode),
                           ("sglang", "sglang", _sglang_gdn_packed_decode)):
    register(Provider(
        library=_lib, signature="fused_recurrent_gated_delta_rule",
        kernel_impl=f"{_lib}_fused_recurrent_gated_delta_rule_packed_decode",
        import_name=_imp,
        serves=KernelSignature(takes_state=True, signature="fused_recurrent_gated_delta_rule",
                               phases=("decode",), activation=("bf16",),
                               enum=_ONE_TOKEN),
        build=_gdn_packed_decode_build(_entry),
        fused_op="fused_recurrent_gated_delta_rule",
    ))
register(Provider(
    library="torch", signature="fused_recurrent_gated_delta_rule",
    kernel_impl="torch_fused_recurrent_gated_delta_rule",
    import_name="torch",
    serves=KernelSignature(takes_state=True, signature="fused_recurrent_gated_delta_rule",
                           phases=("decode",), activation=TORCH_FLOAT),
    build=_torch_gdn_decode_build,
    fused_op="fused_recurrent_gated_delta_rule",
))


# ---------------------------------------------------------------------------
# fused_gdn_gating — the Gated DeltaNet gate prep alone (the fused span of that
# name): g = -exp(A_log)·softplus(a + dt_bias) and beta = sigmoid(b), both
# written fp32 `[1, T, HV]`. sglang's prefill launch on CUDA
# (kernels/ops/attention/fla/fused_gdn_gating.py:44, kernel :10; called at
# srt/layers/attention/linear/gdn_backend.py:947 / :1349, v0.5.22) on the
# token-major `a`, `b [T, HV]`. vLLM's prefill prep is a larger launch
# (`fused_post_conv_prep`, which the binding does not state), so this span is
# sglang's alone.
# ---------------------------------------------------------------------------

def _sglang_gdn_gating_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import (
        fused_gdn_gating,
    )
    dt = _torch_dtype(inv.dtype)
    a, b = inputs["a"].to(dt).contiguous(), inputs["b"].to(dt).contiguous()
    A_log = inputs["A_log"].contiguous()        # as stored (the cell's weight dtype)
    dt_bias = inputs["dt_bias"].contiguous()
    return BenchCase(
        run=lambda: fused_gdn_gating(A_log, a, b, dt_bias),
        # (g, beta), each `[1, T, HV]` → `[T, 2, HV]`.
        to_canonical=lambda out: torch.stack(
            [out[0][0], out[1][0]], dim=-2).float())


def _torch_gdn_gating_build(inv, env, inputs) -> BenchCase:
    import torch
    from ..refs import gdn_gates
    a = inputs["a"]
    out = torch.empty(*a.shape[:-1], 2, a.shape[-1], dtype=torch.float32,
                      device=a.device)

    def run():
        g, beta = gdn_gates(a, inputs["b"], inputs["A_log"],
                            inputs["dt_bias"])
        out[..., 0, :] = g
        out[..., 1, :] = beta
        return out
    return BenchCase(run=run, to_canonical=lambda o: o.float())


register(Provider(
    library="sglang", signature="fused_gdn_gating",
    kernel_impl="sglang_fused_gdn_gating", import_name="sglang",
    # g and beta are allocated fp32 (fused_gdn_gating.py:57-58), but the
    # cell keys `output` at the activation's dtype: the op's ports are
    # `g_out` / `beta_out`, no primary `output`, so the walk states none.
    # Declared as the cell is keyed until the key carries the ports' fp32 —
    # declaring fp32 here would refuse every cell of the span.
    serves=KernelSignature(signature="fused_gdn_gating", phases=("prefill",),
                           activation=("bf16",)),
    build=_sglang_gdn_gating_build, fused_op="fused_gdn_gating",
))
register(Provider(
    library="torch", signature="fused_gdn_gating",
    kernel_impl="torch_fused_gdn_gating", import_name="torch",
    # Keyed as the sglang launch above is.
    serves=KernelSignature(signature="fused_gdn_gating", phases=("prefill",),
                           activation=TORCH_FLOAT),
    build=_torch_gdn_gating_build, fused_op="fused_gdn_gating",
))


# ---------------------------------------------------------------------------
# fused_kda_gate — the KDA prefill gates in one launch (the fused span of that
# name): the bounded per-channel gate g = lower_bound·sigmoid(exp(A_log)·
# (f + dt_bias)), chunk-cumsummed and scaled to log2, and beta = sigmoid(b),
# both written fp32. sglang's prefill launch on CUDA
# (kernels/ops/attention/fla/kda.py:1059 `kda_gate_chunk_cumsum`, kernel :947;
# called from `chunk_kda_fwd` :1178 when `A_log` is given, in place of the
# plain chunk-local cumsum, v0.5.22). vLLM's KDA prefill is FlashKDA's wider
# prepare launch (the plan's comment), so this span is sglang's alone — as
# `fused_gdn_gating` is.
# ---------------------------------------------------------------------------

def _sglang_kda_gate_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.fla.kda import kda_gate_chunk_cumsum
    from ..refs import KDA_GATE_CHUNK, RCP_LN2
    dt = _torch_dtype(inv.dtype)
    f = inputs["f"].to(dt)
    B, T = f.shape[0], f.shape[1]
    HV = int(inputs["A_log"].shape[0])
    # The kernel takes the gate head-split `[B, T, H, K]` contiguous, beta
    # `[B, T, H]`, the bias flat `[H·K]` — the cell's own layouts.
    g = f.reshape(B, T, HV, f.shape[-1] // HV).contiguous()
    b = inputs["b"].to(dt).contiguous()
    A_log = inputs["A_log"].contiguous()      # as stored (the cell's weight dtype)
    dt_bias = inputs["dt_bias"].contiguous()
    lower_bound = float(inputs["lower_bound"])
    return BenchCase(
        run=Launch(kda_gate_chunk_cumsum, g, A_log=A_log,
                   chunk_size=KDA_GATE_CHUNK, scale=RCP_LN2, dt_bias=dt_bias,
                   lower_bound=lower_bound, beta=b),
        # (gate [B,T,HV,K] fp32, beta [B,T,HV] fp32) → [B, T, HV, K+1].
        to_canonical=lambda out: torch.cat(
            [out[0], out[1].unsqueeze(-1)], dim=-1).float())


def _torch_kda_gate_build(inv, env, inputs) -> BenchCase:
    """The span's naive ops, eager — the launch's own function, as
    `torch_fused_gdn_gating` is the GDN one's."""
    import torch
    from ..refs import kda_gate_chunk_cumsum
    f = inputs["f"]
    # g comes back as f's flat head-key width split [.., HV, K] (HV off
    # A_log, K the remainder); the join adds beta beside the key axis.
    hv = inputs["A_log"].shape[0]
    out = torch.empty(*f.shape[:-1], hv, f.shape[-1] // hv + 1,
                      dtype=torch.float32, device=f.device)

    def run():
        g, beta = kda_gate_chunk_cumsum(f, inputs["b"],
                                        inputs["A_log"], inputs["dt_bias"],
                                        inputs["lower_bound"])
        out[..., :-1] = g
        out[..., -1] = beta
        return out
    return BenchCase(run=run, to_canonical=lambda o: o.float())


register(Provider(
    library="sglang", signature="fused_kda_gate",
    kernel_impl="sglang_kda_gate_chunk_cumsum", import_name="sglang",
    # Gate and beta written fp32 (kda.py:1066, :1112).
    serves=KernelSignature(signature="fused_kda_gate", phases=("prefill",),
                           activation=("bf16",), output=("fp32",)),
    build=_sglang_kda_gate_build, fused_op="fused_kda_gate",
))
register(Provider(
    library="torch", signature="fused_kda_gate",
    kernel_impl="torch_fused_kda_gate", import_name="torch",
    serves=KernelSignature(signature="fused_kda_gate", phases=("prefill",),
                           activation=TORCH_FLOAT, output=("fp32",)),
    build=_torch_kda_gate_build, fused_op="fused_kda_gate",
))


# vLLM's GDN prefill preparation after the causal conv: one Triton launch
# writes q/k/v/g/beta. What serving hands it (v0.32.0):
#   * the conv output TOKEN-major and contiguous. The prefill path transposes
#     the contiguous `[tokens, channels]` mixed_qkv to a channel-last
#     `[channels, tokens]` view for `causal_conv1d_fn`
#     (mamba/gdn/qwen_gdn_linear_attn.py:1448-1461), which allocates its
#     output `torch.empty_like(x)` (mamba/ops/causal_conv1d.py:558) — the
#     same channel-last strides — so the transpose back is `[tokens,
#     channels]` with unit channel stride. The kernel relies on it: it
#     addresses `mixed_qkv[t, c]` as `t * stride(0) + c`, ignoring
#     stride(1) (fused_gdn_prefill_post_conv.py:77-123; :167 "[L, qkv_dim]
#     contiguous"). Handed the transpose of a channel-first buffer
#     instead, it read the conv output across tokens: cosine 0.008 on the
#     H100 (2026-10-26).
#   * b | a as the two halves of one `ba` projection row (:1123), strided
#     views the kernel reads through `stride_a_tok` / `stride_b_tok`.
# ---------------------------------------------------------------------------
def _vllm_gdn_post_conv_build(inv: Invocation, env: Env,
                              inputs) -> BenchCase:
    _require_cuda()
    import torch
    from vllm.third_party.flash_linear_attention.ops import (
        fused_post_conv_prep)

    dt = _torch_dtype(inv.dtype)
    rows = int(inputs["qkv"].shape[0] * inputs["qkv"].shape[1])
    conv_output = inputs["qkv"].to(dt).reshape(rows, -1).contiguous()
    ba = torch.empty(rows, 2 * inputs["a"].shape[-1], dtype=dt,
                     device=conv_output.device)
    ba[:, :inputs["a"].shape[-1]] = inputs["b"].to(dt).reshape(rows, -1)
    ba[:, inputs["a"].shape[-1]:] = inputs["a"].to(dt).reshape(rows, -1)
    b, a = ba.chunk(2, dim=-1)
    # The checkpoint stores A_log fp32 and the kernel loads it that way even
    # when dt_bias (and therefore the row's weight slot) is an activation
    # width; `computes_at` publishes the conversion.
    A_log = inputs["A_log"].float().contiguous()
    dt_bias = inputs["dt_bias"].to(dt).contiguous()
    H, K, V = (int(inputs[n]) for n in ("H", "K", "V"))
    return BenchCase(
        run=lambda: fused_post_conv_prep(
            conv_output, a, b, A_log, dt_bias, H, K, V,
            apply_l2norm=True, output_g_exp=False),
        to_canonical=lambda out: out[0].reshape(
            inputs["qkv"].shape[0], inputs["qkv"].shape[1], H, K),
    )


def _torch_gdn_post_conv_build(inv: Invocation, env: Env,
                               inputs) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    qkv = inputs["qkv"].to(dt)
    a, b = inputs["a"].to(dt), inputs["b"].to(dt)
    A_log, dt_bias = inputs["A_log"].float(), inputs["dt_bias"].to(dt)
    H, K, HV, V = (int(inputs[n]) for n in ("H", "K", "HV", "V"))

    def norm(x):
        return x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    def run():
        # The whole chain inside the timed call: computed at build time, the
        # floor would time returning a tuple.
        q, k, v = qkv.split([H * K, H * K, HV * V], dim=-1)
        q = norm(q.reshape(*q.shape[:-1], H, K))
        k = norm(k.reshape(*k.shape[:-1], H, K))
        v = v.reshape(*v.shape[:-1], HV, V)
        g = -A_log.exp() * torch.nn.functional.softplus(
            a.float() + dt_bias.float())
        beta = torch.sigmoid(b.float())
        return q, k, v, g, beta

    return BenchCase(
        run=run,
        to_canonical=lambda out: out[0].float(),
    )


for _act in ("bf16", "fp16"):
    # Both widths run; only bf16 is reachable here. Every gated-delta-net
    # checkpoint in this corpus (Qwen3-Coder-Next, Qwen3.5-35B-A3B,
    # Qwen3.8-27B / -2.4T-A95B / -Flash-Next) ships bf16, and the corpus's
    # one fp16 model (Llama-2-7b) has no GDN conv.
    _unreached = "" if _act == "bf16" else (
        "no corpus checkpoint runs a gated-delta-net prep in fp16; every "
        "GDN model here ships bf16")
    register(Provider(
        library="vllm", signature="fused_gdn_post_conv_prep",
        kernel_impl=f"vllm_fused_post_conv_prep_{_act}",
        import_name="vllm.third_party.flash_linear_attention.ops",
        serves=KernelSignature(
            signature="fused_gdn_post_conv_prep", activation=(_act,),
            # dt_bias as stored, at the activation width or fp32; the build
            # casts it to the activation and loads A_log fp32 either way.
            weight=(_act, "fp32"),
            output=(_act,), phases=("prefill",),
            secondary_outputs=(("k", _act), ("v", _act),
                               ("g", "fp32"), ("beta", "fp32"))),
        computes_at={"weight": "fp32"},
        unreached_reason=_unreached,
        build=_vllm_gdn_post_conv_build,
        fused_op="fused_gdn_post_conv_prep",
    ))
    register(Provider(
        library="torch", signature="fused_gdn_post_conv_prep",
        kernel_impl=f"torch_fused_post_conv_prep_{_act}",
        import_name="torch",
        serves=KernelSignature(
            signature="fused_gdn_post_conv_prep", activation=(_act,),
            # dt_bias as stored, at the activation width or fp32; the build
            # casts it to the activation and loads A_log fp32 either way.
            weight=(_act, "fp32"),
            output=(_act,), phases=("prefill",),
            secondary_outputs=(("k", _act), ("v", _act),
                               ("g", "fp32"), ("beta", "fp32"))),
        computes_at={"weight": "fp32"},
        unreached_reason=_unreached,
        build=_torch_gdn_post_conv_build,
        fused_op="fused_gdn_post_conv_prep",
    ))


# ---------------------------------------------------------------------------
# fused_recurrent_kda — the KDA decode step with its bounded per-channel gate,
# beta sigmoid and q/k norms folded in (the fused span of that name), as each
# framework runs it for GLM-5.3-Flash (v0.32.0 / v0.5.22):
#   vLLM   `fused_recurrent_kda(compute_gate=True, sigmoid_beta=True,
#          use_qk_l2norm_in_kernel=True, lower_bound)`
#          (models/glm5next/nvidia/ops/third_party/kda/kernels.py:148, kernel
#          fused_recurrent.py; gate :169-177), called varlen — q/k/v
#          `[1, B, H, D]` with `cu_seqlens` — at models/glm5next/common/kda.py
#          :760 for a plain decode step
#   sglang `fused_sigmoid_gating_delta_rule_update(is_kda=True, lower_bound)`
#          (kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:531)
#          through linear/kernels/kda_triton.py `decode` (:125): GLM's gate
#          is bounded (`lower_bound` set) and the Triton KDA backend has no
#          bounded packed decode (`supports_packed_decode_safe_gate` unset),
#          so kda_backend.py:945-947 skips the packed path and :981 calls
#          this kernel
# Both take one token per sequence here. vLLM's wrapper allocates the output
# like `k` and indexes it with v's head/width (kernels.py:76-82), so it serves
# only H == HV and K == V — declared below; GLM is 64 heads of 128 on both.
# fp32 state pool (the graph's recurrent_state is F32), slots 1..B.
# The activations are the model's own — bf16 in every checkpoint that binds
# this span, fp16 the other width the Triton kernels compile for. An fp32
# cell is not a launch either framework makes; it is the torch floor's.
# ---------------------------------------------------------------------------

def _kda_decode_operands(inv, inputs):
    import torch
    dt = _torch_dtype(inv.dtype)
    q, k, v, f, b = (inputs[n].to(dt) for n in ("q", "k", "v", "f", "b"))
    B, _T, _H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    state = torch.zeros(B + 1, HV, V, K, dtype=torch.float32, device=q.device)
    state[1:] = inputs["initial_state"].float()
    slots = torch.arange(1, B + 1, dtype=torch.int32, device=q.device)
    cu = torch.arange(0, B + 1, dtype=torch.int32, device=q.device)
    # The gate bound rides the inputs (`refs.KDA_LOWER_BOUND`), so the
    # kernel and the oracle take one value.
    # Varlen: the B one-token sequences laid end to end, `[1, B, …]`. The
    # raw gate stays flat `[B, 1, HV·K]`, as the fused node hands it over;
    # each provider lays it out the way its framework's call site does.
    return dict(q=q.transpose(0, 1), k=k.transpose(0, 1), v=v.transpose(0, 1),
                f=f, b=b.transpose(0, 1),
                A_log=inputs["A_log"].contiguous(),
                dt_bias=inputs["dt_bias"].contiguous(),
                state=state, slots=slots, cu=cu, B=B, HV=HV, K=K, V=V,
                lower_bound=float(inputs["lower_bound"]))


def _vllm_kda_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        fused_recurrent_kda,
    )
    o = _kda_decode_operands(inv, inputs)
    # vLLM's call site splits the gate into heads before the kernel:
    # `g1.reshape(1, -1, local_num_heads, head_dim)` (glm5next/common/kda.py
    # :477-478), then indexes it `(bos·HV + i_hv)·K + o_k`.
    g = o["f"].reshape(1, o["B"], o["HV"], o["K"])

    def run():
        out, _ = fused_recurrent_kda(
            q=o["q"], k=o["k"], v=o["v"], g=g, beta=o["b"],
            initial_state=o["state"], use_qk_l2norm_in_kernel=True,
            cu_seqlens=o["cu"], ssm_state_indices=o["slots"],
            sigmoid_beta=True, a_log=o["A_log"], g_bias=o["dt_bias"],
            compute_gate=True, lower_bound=o["lower_bound"])
        return out
    # `[1, B, HV, V]` → canonical `[B, 1, HV, V]`.
    return BenchCase(run=run, to_canonical=lambda out: out.transpose(0, 1).float())


def _sglang_kda_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    o = _kda_decode_operands(inv, inputs)
    B, HV, K = o["B"], o["HV"], o["K"]
    # sglang's decode hands the kernel the flat token-major gate `[T, HV·K]`
    # (kda_backend.py:981-993 forwards `a` as projected); the kernel steps it
    # by `a.stride()[-2]`, the token stride of that 2-D layout
    # (fused_sigmoid_gating_recurrent.py:594, :248).
    a = o["f"].reshape(B, HV * K).contiguous()
    b = o["b"].reshape(B, HV).contiguous()

    def run():
        return fused_sigmoid_gating_delta_rule_update(
            A_log=o["A_log"], a=a, dt_bias=o["dt_bias"], softplus_beta=1.0,
            softplus_threshold=20.0, q=o["q"], k=o["k"], v=o["v"], b=b,
            initial_state_source=o["state"], initial_state_indices=o["slots"],
            use_qk_l2norm_in_kernel=True, cu_seqlens=o["cu"], is_kda=True,
            lower_bound=o["lower_bound"])
    return BenchCase(run=run, to_canonical=lambda out: out.transpose(0, 1).float())


def _torch_kda_decode_build(inv, env, inputs) -> BenchCase:
    """The span's naive ops, eager, in fp32 like the state it updates."""
    from ..refs import _delta_rule_scan, kda_bounded_gates
    q, k, v = (inputs[n].float() for n in ("q", "k", "v"))
    # A copy, outside the timed call: the step writes its state back, as
    # the serving launch writes its slot, and the inputs are shared by every
    # provider of the cell.
    state = inputs["initial_state"].float().clone()

    def run():
        g, beta = kda_bounded_gates(inputs["f"], inputs["b"], inputs["A_log"],
                                    inputs["dt_bias"], inputs["lower_bound"])
        return _delta_rule_scan(q, k, v, g, beta, state, final_state=state)
    return BenchCase(run=run, to_canonical=lambda o: o.float())


# H == HV and K == V, as two-way divisibility.
_KDA_SQUARE = (("n_lin_v_heads", "n_lin_k_heads"),
               ("n_lin_k_heads", "n_lin_v_heads"),
               ("d_lin_v_head", "d_lin_k_head"),
               ("d_lin_k_head", "d_lin_v_head"))

register(Provider(
    library="vllm", signature="fused_recurrent_kda",
    kernel_impl="vllm_fused_recurrent_kda_compute_gate",
    import_name="vllm",
    serves=KernelSignature(takes_state=True, signature="fused_recurrent_kda", phases=("decode",),
                           activation=("bf16", "fp16"),
                           enum=_ONE_TOKEN,
                           requires_divisible_dims=_KDA_SQUARE),
    build=_vllm_kda_decode_build,
    fused_op="fused_recurrent_kda",
))
register(Provider(
    library="sglang", signature="fused_recurrent_kda",
    kernel_impl="sglang_fused_sigmoid_gating_delta_rule_update_kda",
    import_name="sglang",
    serves=KernelSignature(takes_state=True, signature="fused_recurrent_kda", phases=("decode",),
                           activation=("bf16", "fp16"),
                           enum=_ONE_TOKEN),
    build=_sglang_kda_decode_build,
    fused_op="fused_recurrent_kda",
))
register(Provider(
    library="torch", signature="fused_recurrent_kda",
    kernel_impl="torch_fused_recurrent_kda",
    import_name="torch",
    serves=KernelSignature(takes_state=True, signature="fused_recurrent_kda", phases=("decode",),
                           activation=TORCH_FLOAT),
    build=_torch_kda_decode_build,
    fused_op="fused_recurrent_kda",
))


# ---------------------------------------------------------------------------
# causal_conv1d — the depthwise short conv WITH its SiLU (the fused span
# `causal_conv1d`), phase-split like the recurrences. The full-sequence form is
# varlen: `x (dim, total_tokens)` with the sequences concatenated left to right
# and `query_start_loc` marking the boundaries, plus a conv-state pool the
# sequences index by slot. The decode form takes one token per sequence and
# rolls that state.
#
# The state pools are `B + 1` slots and the sequences use slots 1..B, as a
# serving deployment does: vLLM reserves slot 0 as its null block
# (`NULL_BLOCK_ID`, skipped by both kernels), so slot 0 cannot carry a
# sequence there. Every state starts at zero — the cell is a fresh prefill,
# and a decode step whose window holds no history — which is also what the
# oracle assumes, so the decode step's single output column is
# cross-checkable against the oracle's last column.
#
# Framework kernels take no dilation (`dilation` enum 1): a dilated conv is
# not their launch, and the torch floor alone measures it.
#
# WHICH DECODE LAUNCH a framework runs is not the same question in both. vLLM
# calls `causal_conv1d_update` from every mixer (qwen_gdn_linear_attn.py:1756,
# kda.py, mamba_mixer2.py at v0.32.0). sglang v0.5.22 splits: the KDA backend
# (kda_backend.py:935 — Kimi-K3, GLM-5.3-Flash), the Mamba-2 mixer
# (mamba/mamba.py:673-693) and the GDN backend's non-handoff path
# (gdn_backend.py:855-863) still call it, but a model whose mixer hands the
# backend its PROJECTION OUTPUTS instead of an unpacked `mixed_qkv` gets a
# wider launch — see `_SGLANG_GDN_FUSED_PROJ_CONV` below.
# ---------------------------------------------------------------------------

def _conv_slots(B: int, device):
    import torch
    return torch.arange(1, B + 1, dtype=torch.int32, device=device)


def _vllm_conv1d_fn_build(inv, env, inputs) -> BenchCase:
    """vLLM `causal_conv1d_fn` as the GDN / KDA / Mamba-2 mixers call it
    (qwen_gdn_linear_attn.py:1448-1451 at v0.32.0): the token-major
    `[B·S, C]` projection TRANSPOSED to a `[C, B·S]` view, state pool
    indexed by `cache_indices`, no prior state.

    The view is the launch, not a detail. The kernel takes `stride_x_dim`
    as a `tl.constexpr` and forms `idx_feats * stride_x_dim` in int32
    (ops/causal_conv1d.py:39, :135, :217); only the TOKEN stride is int64
    (:40). Channel-last, as served, the channel stride is 1 and nothing
    overflows. A contiguous `[C, B·S]` copy makes the channel stride B·S,
    so C·B·S past 2**31 wraps — an illegal address at C=10240, S=262144
    (Qwen3.8-Flash-Next) that poisoned the process — and below that it
    times a memory layout no server hands this kernel."""
    _require_cuda()
    import torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    dt = _torch_dtype(inv.dtype)
    # token-major [B·S, C] (launch_inputs), made dense so its transpose is
    # the channel-last view the framework passes (channel stride 1).
    tokens = inputs["x"].to(dt).contiguous()
    B = int(inv.dims["B"])
    S = int(inv.dims["s_query"])
    C = tokens.shape[1]
    W = int(inputs["W"])
    w = inputs["weight"].to(dt)
    b = inputs["bias"].to(dt)
    conv_states = torch.zeros(B + 1, C, W - 1, dtype=dt, device=tokens.device)
    qsl = torch.arange(0, (B + 1) * S, S, dtype=torch.int32,
                       device=tokens.device)
    slots = _conv_slots(B, tokens.device)
    fresh = torch.zeros(B, dtype=torch.bool, device=tokens.device)
    return BenchCase(
        run=lambda: causal_conv1d_fn(
            tokens.transpose(0, 1), w, b, conv_states, qsl,
            cache_indices=slots, has_initial_state=fresh, activation="silu"),
        # The output is a channel-last [C, B·S] like its input; back to the
        # token-major launch layout, which `launch_output` inverts.
        to_canonical=lambda out: out.transpose(0, 1).float(),
    )


def _sglang_conv1d_fn_build(inv, env, inputs) -> BenchCase:
    """sglang's prefill conv exactly as its GDN backend calls it
    (srt/layers/attention/linear/gdn_backend.py:930 at v0.5.22): the
    token-major `[B·S, C]` projection TRANSPOSED to a `[C, B·S]` view, with
    `seq_lens_cpu`. The wrapper (srt/layers/attention/mamba/causal_conv1d.py
    :43) dispatches on exactly that — a non-unit last stride with
    `seq_lens_cpu` takes the Triton kernel, a contiguous `x` the compiled
    CUDA one — so a contiguous launch here would time the other kernel."""
    _require_cuda()
    import torch
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn,
    )
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)                      # canonical [B, C, S]
    B, C, S = x.shape
    W = int(inputs["W"])
    tokens = x.permute(0, 2, 1).reshape(B * S, C).contiguous()
    w = inputs["weight"].to(dt)
    b = inputs["bias"].to(dt)
    conv_states = torch.zeros(B + 1, C, W - 1, dtype=dt, device=x.device)
    qsl = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=x.device)
    slots = _conv_slots(B, x.device)
    fresh = torch.zeros(B, dtype=torch.bool, device=x.device)
    seq_lens = [S] * B
    return BenchCase(
        run=lambda: causal_conv1d_fn(
            tokens.transpose(0, 1), w, b, activation="silu",
            conv_states=conv_states, has_initial_state=fresh,
            cache_indices=slots, query_start_loc=qsl, seq_lens_cpu=seq_lens),
        # [C, B·S] → [B, C, S]
        to_canonical=lambda out: out.reshape(C, B, S).permute(1, 0, 2).float(),
    )


def _conv1d_update_build(entry):
    def build(inv, env, inputs) -> BenchCase:
        _require_cuda()
        import torch
        fn = entry()
        dt = _torch_dtype(inv.dtype)
        x = inputs["x"].to(dt)
        B, C = x.shape[0], x.shape[1]
        W = int(inputs["W"])
        # ONE token, which is the whole cell: both registrations below
        # declare `s_query = 1` (`_ONE_TOKEN`), and that is the kernel's own
        # assumption rather than this adapter's convenience. Both copies do
        # accept `x (batch, dim, seqlen)` — "single or multiple tokens
        # prediction" — but the multi-token form is the SPECULATIVE one: the
        # effective `state_len = width - 1 + (seqlen - 1)` comes from
        # `num_accept_tokens`, and the state-rolling step assumes
        # `state_len > seqlen` (`causal_conv1d_triton.py:737`). A verify
        # window is therefore a wider launch against a wider state pool with
        # an intermediate window buffer beside it — different bytes, like
        # `_SGLANG_GDN_FUSED_PROJ_CONV` — so `s_query > 1` is a DECLARED GAP
        # here. Taken as `x[:, :, -1]` at any `s_query`, as it was, this
        # timed one token against a row billed for S of them: 733 rows of a
        # conv the harness never ran.
        x1 = x[:, :, -1].contiguous()                       # [B,C]
        state = torch.zeros(B + 1, C, W - 1, dtype=dt, device=x.device)
        w = inputs["weight"].to(dt)
        b = inputs["bias"].to(dt)
        slots = _conv_slots(B, x.device)
        return BenchCase(
            run=Launch(fn, x1, state, w, b, "silu",
                       conv_state_indices=slots),
            # One step against an empty window is the oracle's output at a
            # one-token sequence.
            to_canonical=lambda out: out.unsqueeze(-1).float(),
        )
    return build


def _vllm_conv1d_update():
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    return causal_conv1d_update


def _sglang_conv1d_update():
    # The decode conv sglang's GDN backend imports
    # (srt/layers/attention/linear/gdn_backend.py:7-10, called at :856 at
    # v0.5.22); the KDA backend calls the same one (kda_backend.py:935).
    return _relocated("causal_conv1d_update",
                      "sglang.kernels.ops.mamba.causal_conv1d_triton",
                      "sglang.srt.layers.attention.mamba.causal_conv1d_triton")


def _torch_conv1d_build(inv, env, inputs) -> BenchCase:
    """Depthwise conv through torch — the no-dependency floor, so the short conv
    is measurable on any build, and the prefill form always has a second
    implementation to bound the fused one against. The only implementation
    here of a DILATED conv: the framework kernels take none."""
    import torch.nn.functional as F
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    w = inputs["weight"].to(dt)
    b = inputs["bias"].to(dt)
    dil = int(inputs["dilation"])
    C, W = w.shape
    wc = w.unsqueeze(1)

    def _run():
        return F.silu(F.conv1d(F.pad(x, ((W - 1) * dil, 0)), wc, bias=b,
                               groups=C, dilation=dil))

    return BenchCase(run=_run, to_canonical=lambda out: out.float())


#: Token-major `[B·S, C]`: the projection's own layout, which the framework
#: hands its prefill conv as a transposed (channel-last) view.
_CONV1D_LAUNCH = (("x", ((0, 2), 1)),)
# The framework kernels take no dilation.
_UNDILATED = (("dilation", (1,)),)

#: sglang v0.5.22 serves a decode conv TWO ways, and this build is only one of
#: them. A mixer that hands the backend its projection outputs as a tuple —
#: Qwen3.5 / Qwen3.8-27B through `srt/models/qwen3_5.py:1008-1023`, on by
#: default (`_gdn_decode_fused_proj_conv`, :183-185, `EnvBool(True)` at
#: `srt/environ.py:1873`) — gets `fused_qkvzba_causal_conv1d_update_contiguous`
#: (`kernels/ops/attention/triton_gdn_fused_proj.py:541`, dispatched at
#: `gdn_backend.py:723-801`): ONE launch that reads `qkvz|ba`, unpacks it, runs
#: the conv and writes `mixed_qkv`, `z`, `b`, `a` and the conv state. That is a
#: wider boundary than this span — different bytes, one launch instead of
#: several — so it is a different cell, and no plan draws it yet. Until one
#: does, a Qwen3.5-class sglang decode conv has no measurement route of its own
#: and this row must not be read as one; the two models of that class in the
#: corpus are Qwen3.5-35B-A3B and Qwen3.8-27B. Every other sglang decode conv —
#: KDA (Kimi-K3, GLM-5.3-Flash), Mamba-2 (Nemotron), and the GDN models whose
#: mixer hands an unpacked tensor (Qwen3-Coder-Next via `qwen3_next.py`,
#: Qwen3.8-Flash-Next via `qwen4_exp.py`) — is this kernel.
#:
#: Declared rather than narrowed away: the cell cannot tell the two apart (both
#: are the same B / channels / width), and the models that do run this kernel
#: would lose a correct measurement. `test_sglangs_fused_projection_conv_is_a_
#: named_gap` is what counts it.
_SGLANG_GDN_FUSED_PROJ_CONV = "sglang_gdn_fused_projection_conv"


register(Provider(
    library="vllm", signature="causal_conv1d",
    kernel_impl="vllm_causal_conv1d_fn",
    import_name="vllm",
    serves=KernelSignature(takes_state=True, signature="causal_conv1d", phases=("prefill",),
                           activation=("bf16",), enum=_UNDILATED),
    build=_vllm_conv1d_fn_build,
    launch_inputs=_CONV1D_LAUNCH,
    launch_output="x",
    fused_op="causal_conv1d",
))
register(Provider(
    library="vllm", signature="causal_conv1d",
    kernel_impl="vllm_causal_conv1d_update",
    import_name="vllm",
    # DISABLED, not deleted, so its cells read as a gap rather than as coverage.
    # It fails to COMPILE at every shape probed on the vllm/triton pair the
    # bench box measures with (its env pins vLLM 0.11; 183 failures in a single
    # model) — not anything about the cell. At vLLM v0.32.0 (torch 2.13+cu130)
    # it builds and cross-checks on the same H100 (B=64, C=8192, width 4);
    # enable it when the box's vllm env moves there.
    disabled_reason=("triton CompilationError at causal_conv1d.py:63 "
                     "(\'NoneType\' object has no attribute \'type\') at every "
                     "batch probed on the box's vllm 0.11; builds and "
                     "cross-checks at v0.32.0"),
    serves=KernelSignature(takes_state=True, signature="causal_conv1d", phases=("decode",),
                           activation=("bf16",),
                           enum=_UNDILATED + _ONE_TOKEN),
    build=_conv1d_update_build(_vllm_conv1d_update),
    fused_op="causal_conv1d",
))
register(Provider(
    library="sglang", signature="causal_conv1d",
    kernel_impl="sglang_causal_conv1d_fn",
    import_name="sglang",
    # The Triton launch's grid is (sequences, ceil(longest / BLOCK_M), …)
    # with BLOCK_M = 8 (kernels/ops/mamba/causal_conv1d_triton.py:20,
    # :533-542 at v0.5.22), and a grid's y extent stops at 65,535: a
    # sequence past 8 * 65,535 tokens is "Triton Error [CUDA]: invalid
    # argument" before anything runs. Bisected on an H100 at C = 512:
    # 524,280 runs, 524,281 and 524,288 refuse (GLM-5.3-Flash's 1,048,576-
    # token prefill was the canary). Per sequence, so a ceiling on
    # `s_query` alone — B is the grid's x extent.
    serves=KernelSignature(takes_state=True,
                           signature="causal_conv1d", phases=("prefill",),
                           activation=("bf16",), enum=_UNDILATED,
                           max_dims=(("s_query", 8 * 65535),)),
    build=_sglang_conv1d_fn_build,
    fused_op="causal_conv1d",
))
register(Provider(
    library="sglang", signature="causal_conv1d",
    kernel_impl="sglang_causal_conv1d_update",
    import_name="sglang",
    # The conv state is [B, C, W] and the kernel indexes it with int32, so it
    # faults once the element count crosses 2**31 — `an illegal memory access`,
    # which poisons the context and takes the rest of the model's run with it.
    # That is what killed Qwen3-Coder-Next at B=521,391, C=8192.
    #
    # A PRODUCT, not a per-axis ceiling: bisected on an H100, the boundary sits
    # at exactly B * C * W = 2**31 for C = 1024, 2048, 4096 and 8192 alike —
    # 524,288 / 262,144 / 131,072 / 65,536 all pass and each next integer
    # faults. No combination of per-dim limits describes that.
    serves=KernelSignature(takes_state=True,
        signature="causal_conv1d", phases=("decode",),
        activation=("bf16",), enum=_UNDILATED + _ONE_TOKEN,
        unsupported_features=(_SGLANG_GDN_FUSED_PROJ_CONV,),
        max_product=((("B", "d_conv_channels", "s_conv_kernel"), 2 ** 31),)),
    build=_conv1d_update_build(_sglang_conv1d_update),
    fused_op="causal_conv1d",
))
register(Provider(
    library="torch", signature="causal_conv1d",
    kernel_impl="torch_depthwise_conv1d", import_name="torch",
    serves=KernelSignature(signature="causal_conv1d", phases=("prefill",),
                           activation=TORCH_FLOAT),
    build=_torch_conv1d_build,
    fused_op="causal_conv1d",
))


def _sglang_dilated_conv1d_update_build(inv, env, inputs) -> BenchCase:
    """The decode step of a DILATED short conv, as sglang runs it.

    The framework's `causal_conv1d_update` kernels take no dilation, and
    sglang does not pretend otherwise: for a dilated conv its mixer steps
    the window through `F.conv1d` itself — read the state, append the new
    token, convolve with `dilation=`/`groups=C`, keep the advanced window
    (`srt/models/qwen4_exp.py:1129-1146` at v0.5.22, the `ngram_size`
    short conv of Qwen3.8-Flash-Next). So this cell's launch IS cuDNN's
    grouped dilated conv over the state window plus the step's tokens, not
    a Triton update kernel, and that is what gets timed.

    A step carries the cell's own token count: one token on the plain
    decode path and `draft_token_num` of them when the step verifies a
    draft, where sglang lays the tokens into a `[requests, row_width,
    channels]` buffer and runs ONE conv over `cat(state, tokens)`
    (:1151-1165) — `row_width` outputs from one launch. Timing only the
    last token there made the launch a different shape from the cell
    (every speculative cell's cross-check failed on the output's shape,
    294 on the H100).

    The state write is inside the timed call because it is inside the
    framework's: the next step's window is this call's tail, and a step
    that does not advance the window is not the step a server runs.

    The launch is torch's own `F.conv1d`, a CPU provider env runs it as
    one (the cross-check does); unlike the CUDA-only wrapper bodies this
    build carries no CUDA requirement, and the guard copy-pasted from
    them made every CPU-only CI run red while GPU boxes passed it by
    accident of the machine, not the env.
    """
    import torch
    import torch.nn.functional as F

    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)                                   # [B,C,S]
    B, C, _S = x.shape
    W = int(inputs["W"])
    dil = int(inputs["dilation"])
    tokens = x.shape[-1]                                     # the step's tokens
    w = inputs["weight"].to(dt).unsqueeze(1)                 # [C,1,W], depthwise
    b = inputs["bias"].to(dt)
    # A zero window, so the step's output is the oracle's at this token
    # count — the same ground the undilated update providers stand on.
    window = (W - 1) * dil
    state = torch.zeros(B, C, window, dtype=dt, device=x.device)
    conv_input = torch.empty(B, C, window + tokens, dtype=dt, device=x.device)

    def _run():
        conv_input[..., :window] = state
        conv_input[..., window:] = x
        out = F.silu(F.conv1d(conv_input, w, bias=b, groups=C, dilation=dil))
        state.copy_(conv_input[..., tokens:])
        return out

    return BenchCase(run=_run, to_canonical=lambda out: out.float())


def _dilated_only(inv) -> str:
    """Leave an undilated decode cell to the framework's own update kernel.

    A gate rather than an `enum`, because the set of dilations a checkpoint
    may ship is open (it is the mixer's ngram size) while the one value this
    build must NOT take is single: at `dilation == 1` sglang dispatches the
    Triton `causal_conv1d_update` the providers above measure, and stepping
    the window through `F.conv1d` there would be a launch nobody serves."""
    dil = int(inv.dims["dilation"])
    if dil < 2:
        raise NotImplementedError(
            "undilated decode conv is the framework's Triton update kernel "
            "(sglang_causal_conv1d_update), not an F.conv1d window step")
    return f"dilation={dil}"


register(Provider(
    library="sglang", signature="causal_conv1d",
    kernel_impl="sglang_dilated_conv1d_update", import_name="sglang",
    impl_params=_dilated_only,
    # The complement of the undilated decode providers above: those declare
    # `enum=_UNDILATED` because their kernels have no dilation argument, which
    # left every dilated decode cell with no route at all (220 of them, on
    # Qwen3.8-Flash-Next). Dilation is read from the cell, so one build covers
    # every ngram_size a checkpoint ships.
    # The launch is torch's own F.conv1d at the cell's dtype.
    serves=KernelSignature(takes_state=True, signature="causal_conv1d", phases=("decode",),
                           activation=TORCH_FLOAT),
    build=_sglang_dilated_conv1d_update_build,
    fused_op="causal_conv1d",
))


# ---------------------------------------------------------------------------
# Mamba-2 selective scan
# ---------------------------------------------------------------------------

def _mamba2_build(inv, env, inputs) -> BenchCase:
    """Mamba-2 selective scan. x[B,S,Nmh,Dmh], dt[B,S,Nmh], A[Nmh],
    B/C[B,S,Nmg,Dss], D/dt_bias[Nmh]. API: mamba_ssm ssd_combined
    `mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D, dt_bias)`.

    `chunk_size` is an AXIS of this cell (`ssm_chunk_size`, declared on the
    signature): the launch runs `ceil(s_query / chunk)` tiles and the
    intra-chunk term is quadratic in it, so two chunk lengths are two
    amounts of work and two measurements. It is therefore read from `dims`,
    where the deployment put it (`Program.scan_chunk_size`, read
    off the checkpoint config at the fields the framework binding names),
    and not picked here — a cell whose config states no chunk has no
    identity and the plan drops it before any provider sees it.

    Read before the device is touched, as its resolution always was: what
    the cell says is checkable without a GPU, and a check that needs one
    cannot run where the declarations are checked."""
    chunk = int(inv.dims["ssm_chunk_size"])
    _require_cuda()
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    x, dt, A = inputs["x"], inputs["dt"], inputs["A"]
    B, C = inputs["B"], inputs["C"]
    D, dt_bias = inputs["D"], inputs["dt_bias"]
    # `dt_softplus=True`: the step is softplus(dt + dt_bias), as both
    # servers ask for it (sglang v0.5.22 srt/layers/attention/mamba/
    # mamba.py:638, vLLM's mamba_mixer2 alike) and as the oracle computes
    # it. The entry point defaults to False (mamba_ssm 2.2.6
    # ops/triton/ssd_combined.py:575), which scans with the raw sum — a
    # negative step for a negative bias. Only cells short enough for the
    # O(T) oracle saw it: every prefill cell past its step bound timed the
    # wrong-step launch unchecked.
    #
    # The recurrent state the cell carries, read and written as serving's
    # scan reads and writes it (`initial_states` / `return_final_states`):
    # zeros, as the oracle starts (`refs._ssd_scan`), so the output is the
    # same; the launch now moves the state the row bills.
    import torch
    state = torch.zeros(x.shape[0], x.shape[2], x.shape[3], B.shape[3],
                        dtype=torch.float32, device=x.device)
    return BenchCase(
        run=lambda: mamba_chunk_scan_combined(
            x, dt, A, B, C, chunk_size=chunk, D=D, dt_bias=dt_bias,
            dt_softplus=True, initial_states=state,
            return_final_states=True),
        to_canonical=lambda o: (o[0] if isinstance(o, tuple) else o).float(),
    )


register(Provider(
    library="mamba_ssm", signature="mamba2_scan",
    kernel_impl="mamba_chunk_scan_combined",
    import_name="mamba_ssm",
    # No `impl_params`: the chunk is in the cell's identity now
    # (`ssm_chunk_size` in `dims`), and an `_c{chunk}` suffix on the
    # implementation id would be the same fact written a second time.
    serves=KernelSignature(signature="mamba2_scan", activation=("bf16",),
                           takes_state=True),
    build=_mamba2_build,
))


# ---------------------------------------------------------------------------
# gla_simple_recurrence — Ling 2.x's linear-attention layers (Lightning /
# FLA simple_gla): S_t = exp(-slope_h)·S_{t-1} + kᵀv, o = (q/√d)·S_t.
#
# Every launch here READS the per-sequence fp32 state and WRITES the final
# one, because that is what serving runs and what the cell is billed for (the
# component's `recurrent_state`, carried by `gla_recurrence`). These builds
# used to call FLA with no `initial_state` and `output_final_state=False` and
# the torch floor scanned from zeros: a ~2 MB launch timed against a
# numerator holding the ~128 MB state, which read above HBM peak.
#
# Who serves Ling (`BailingMoeV2_5ForCausalLM`):
#   vLLM v0.32.0  decode  `linear_attention_decode` → `linear_decode_forward_
#                 triton` (mamba/linear/bailing_linear_attn.py:691-703,
#                 minimax_linear_attn.py:49-68, lightning_attn.py:676), state
#                 read and written in place by slot;
#                 prefill `linear_attention_prefill_and_mix` (bailing_linear_
#                 attn.py:85, called :646-658) — a loop over the sequences,
#                 each `MiniMaxText01LinearKernel.jit_linear_forward_prefix`
#                 (minimax_linear_attn.py:124-148) → `lightning_attention`
#                 from the slot's state, final state copied back.
#                 q/k/v reach both fp32 (`qkv.to(torch.float32)`, :558) and q
#                 is scaled by 1/√d BEFORE the launch (`q = q * self.scaling`,
#                 :600-601; `linear_scale` is set for its "minimax" backend,
#                 :406-407), so the scale is applied outside the timed call.
#   sglang v0.5.22 both phases `seg_la_fwd` (kernels/ops/attention/linear/
#                 seg_la.py:682), via `_linear_attention_entry`
#                 (srt/layers/attention/linear/lightning_backend.py:262-301,
#                 from forward_extend :399 and forward_decode :468): Ling's
#                 config names no `linear_backend`, so the default "seg_la"
#                 (:90) is the path. Length per sequence picks the kernel —
#                 `seg_la_d_kernel` at one token, `seg_la_p_kernel` above.
#                 The 1/√d is the kernel's `softmax_scale` (:299).
#   fla 0.5.2     `chunk_simple_gla` / `fused_recurrent_simple_gla` with the
#                 head-wise constant decay as `g_gamma` — the library HF's
#                 modelling code calls; no framework serves it.
# The state pool is `_state_pool`'s B + 1 slots for the two frameworks, which
# both index the state by slot.
# ---------------------------------------------------------------------------

def _state_pool(state):
    """The state as a serving pool: B + 1 slots with the sequences at 1..B,
    and their slot ids (int32). Whose layout that is, and why row 0 is
    left alone, is `_initial_state_args`'s docstring."""
    import torch
    n = state.shape[0]
    pool = state.new_zeros((n + 1, *state.shape[1:]))
    pool[1:] = state
    return pool, torch.arange(1, n + 1, dtype=torch.int32,
                              device=state.device)


def _gla_args(inv, inputs):
    """q/k/v at the kernel dtype, the per-head decay rate (fp32, as every
    kernel loads it), and a fp32 COPY of the state.

    A copy, made here and so outside the timed call: every kernel below
    writes the final state back, and the inputs are shared by every
    provider of the cell — writing into them would hand the next provider a
    starting state the reference never saw.

    q/k/v run at the cell's own dtype. They used to be narrowed to bf16 at
    an fp32 cell, the rule the delta-rule kernels need ("please use
    bfloat16"); these kernels take fp32 (FLA's simple_gla does, and fp32 is
    what both frameworks hand theirs), so narrowing would time a launch
    moving half the bytes the row is billed for."""
    dt = _torch_dtype(inv.dtype)
    return (inputs["q"].to(dt), inputs["k"].to(dt), inputs["v"].to(dt),
            inputs["slope"].float().contiguous(),
            inputs["initial_state"].float().clone())


def _fla_gla_build(entry):
    def build(inv, env, inputs) -> BenchCase:
        _require_cuda()
        fn = entry()
        q, k, v, slope, state = _gla_args(inv, inputs)
        # `g_gamma` is FLA's head-wise data-independent LOG decay [H]
        # (simple_gla/chunk.py:264-267), i.e. -slope. The final state is
        # asked for, as the HF forward asks for it to keep the cache
        # (`output_final_state=use_cache`); FLA has no in-place form, so it
        # writes a fresh `[B, H, K, V]`.
        kwargs = _accepted(fn, q=q, k=k, v=v, g_gamma=-slope,
                           output_final_state=True,
                           **_initial_state_args(fn, state))
        return BenchCase(run=lambda: _first(fn(**kwargs)),
                         to_canonical=lambda o: o.float())
    return build


def _vllm_scaled_q(q):
    """q as vLLM hands it to the launch: already multiplied by 1/√d, a step
    of its own before the call (bailing_linear_attn.py:600-601). Scaled in
    fp32 and rounded once, as serving's fp32 q is."""
    return (q.float() * q.shape[-1] ** -0.5).to(q.dtype)


def _vllm_gla_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
        linear_attention_decode,
    )
    q, k, v, slope, state = _gla_args(inv, inputs)
    B, _T, H, D = q.shape
    # One token per sequence, token-major `[B, H, D]` as the mixer holds it.
    q, k, v = (x.reshape(B, H, D) for x in (_vllm_scaled_q(q), k, v))
    pool, slots = _state_pool(state)

    def run():
        return linear_attention_decode(q, k, v, pool, slope, slots,
                                       block_size=32)
    # `[B, H·D]` back to the canonical `[B, s_query=1, H, D]`.
    return BenchCase(run=run,
                     to_canonical=lambda o: o.reshape(B, 1, H, D).float())


def _vllm_gla_prefill_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.linear.bailing_linear_attn import (
        linear_attention_prefill_and_mix,
    )
    from vllm.model_executor.layers.mamba.linear.minimax_linear_attn import (
        MiniMaxText01LinearKernel,
    )
    q, k, v, slope, state = _gla_args(inv, inputs)
    B, T, H, D = q.shape
    q, k, v = (x.reshape(B * T, H, D) for x in (_vllm_scaled_q(q), k, v))
    pool, slots = _state_pool(state)
    # Only the fields the call site reads (it reads them with `getattr`): a
    # batch of B prefills of T tokens each and no decodes.
    meta = SimpleNamespace(
        num_decodes=0, num_prefills=B, num_decode_tokens=0,
        query_start_loc=torch.arange(0, B * T + 1, T, dtype=torch.int32,
                                     device=q.device))

    def no_decodes(*_args, **_kwargs):
        raise AssertionError("a prefill cell has no decode tokens")

    def run():
        # block_size 256: `self.BLOCK = getattr(config, "block", 256)`
        # (mamba/linear/base.py:36) and Ling's config has no `block`.
        return linear_attention_prefill_and_mix(
            q=q, k=k, v=v, kv_cache=pool, state_indices_tensor=slots,
            attn_metadata=meta, slope_rate=slope, block_size=256,
            decode_fn=no_decodes,
            prefix_fn=MiniMaxText01LinearKernel.jit_linear_forward_prefix)
    return BenchCase(run=run,
                     to_canonical=lambda o: o.reshape(B, T, H, D).float())


def _sglang_seg_la_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch

    from sglang.kernels.ops.attention.linear.seg_la import (
        SegLaMeta,
        seg_la_fwd,
    )
    q, k, v, slope, state = _gla_args(inv, inputs)
    B, T, H, D = q.shape
    q, k, v = (x.reshape(B * T, H, D) for x in (q, k, v))
    pool, slots = _state_pool(state)
    starts = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=q.device)
    # `s_scales` is whether each sequence starts from its state: always at
    # decode (`has_initial_states=torch.ones_like(seq_lens)`,
    # linear_metadata.py:33), a cached prefix at prefill (:57). The cell
    # declares a state the launch reads, so every sequence has one.
    meta = SegLaMeta(batch_size=B, max_q_length=None, q_offsets=starts,
                     s_offsets=slots, q_lengths=starts.diff(),
                     s_scales=torch.ones(B, dtype=torch.bool,
                                         device=q.device))

    def run():
        return seg_la_fwd(q=q, k=k, v=v, s=pool, decay_scales=slope,
                          meta=meta, softmax_scale=D ** -0.5, decouple=True)
    return BenchCase(run=run,
                     to_canonical=lambda o: o.reshape(B, T, H, D).float())


def _fla_chunk_simple_gla():
    from fla.ops.simple_gla import chunk_simple_gla
    return chunk_simple_gla


def _fla_recurrent_simple_gla():
    from fla.ops.simple_gla import fused_recurrent_simple_gla
    return fused_recurrent_simple_gla


register(Provider(
    library="fla", signature="gla_simple_recurrence",
    kernel_impl="fla_chunk_simple_gla",
    import_name="fla.ops",
    # fp32 as well as bf16: simple_gla takes fp32 (`_gla_args`).
    serves=KernelSignature(takes_state=True, signature="gla_simple_recurrence",
                           phases=("prefill",), activation=("bf16", "fp32")),
    build=_fla_gla_build(_fla_chunk_simple_gla),
))


register(Provider(
    library="fla", signature="gla_simple_recurrence",
    kernel_impl="fla_fused_recurrent_simple_gla",
    import_name="fla.ops",
    serves=KernelSignature(takes_state=True, signature="gla_simple_recurrence",
                           phases=("decode",), activation=("bf16", "fp32")),
    build=_fla_gla_build(_fla_recurrent_simple_gla),
))


# vLLM's decode kernel takes ONE token per sequence (`assert k.shape ==
# (B, H, 1, D)`, lightning_attn.py:701; a verify window goes to
# `bailing_linear_attention_decode_spec` instead, bailing_linear_attn.py:
# 671), and loads a whole head row with `tl.arange(0, D)`, which Triton
# allows only for a power of two; the grid splits it in 32s.
register(Provider(
    library="vllm", signature="gla_simple_recurrence",
    kernel_impl="vllm_linear_attention_decode",
    import_name="vllm",
    # fp32 is what serving hands it (bailing_linear_attn.py:558); bf16 measured.
    serves=KernelSignature(takes_state=True, signature="gla_simple_recurrence",
                           phases=("decode",), activation=("bf16", "fp32"),
                           enum=(*_ONE_TOKEN,
                                 ("d_head", (32, 64, 128, 256)))),
    build=_vllm_gla_decode_build,
))


# `lightning_attention` splits the key width in blocks of 128 (64 below
# 128) and asserts it divides (lightning_attn.py:567-568); its kernels load
# the full width with `tl.arange`, a power of two. And it runs fp32 q/k/v
# ONLY: `_fwd_kv_parallel` multiplies k by its fp32 decay and dots the
# product with v (`tl.dot(k_trans * k_decay, v)`), which Triton refuses
# for a bf16 v ("Both operands must be same dtype. Got fp32 and bf16",
# reproduced on the audited source). Serving never meets that because it
# casts q/k/v to fp32 first (bailing_linear_attn.py:558) — so a bf16 cell
# is one this kernel does not serve, not one to upcast and bill at bf16.
#
# Known disagreement with the oracle, left for the cross-check to report
# rather than papered over: from a carried state, this kernel decays it by
# λ^t at the t-th token of the chunk (0-based) where the recurrence — and
# vLLM's own decode kernel, sglang's seg_la and FLA — give λ^(t+1)
# (`q_decay = exp(-s·(off_c·CBLOCK + c_array))`, lightning_attn.py:381).
# Measured on the audited source: 3.6% relative error with a non-zero
# state and decay, 0.15% (fp32 TF32 noise) once the history is decayed by
# λ^t instead, and agreement whenever the state or the decay is zero — which
# a fresh prefill's always is, so serving only diverges on a continued
# (chunked or prefix-cached) prefill.
register(Provider(
    library="vllm", signature="gla_simple_recurrence",
    kernel_impl="vllm_linear_attention_prefill_and_mix",
    import_name="vllm",
    serves=KernelSignature(takes_state=True, signature="gla_simple_recurrence",
                           phases=("prefill",), activation=("fp32",),
                           enum=(("d_head", (64, 128, 256)),)),
    build=_vllm_gla_prefill_build,
))


# One entry point, two kernels: `seg_la_fwd` splits the key width in 128s at
# one token (`seg_la_d_kernel`, seg_la.py:843-855: `HEAD_DIM // 128` blocks,
# zero below 128) and in 32s/64s above it (`seg_la_p_kernel`, :705-716).
# Registered once per phase because the decode side is one token per
# sequence: a verify window is the `caches` (MTP) kernel serving runs there
# (lightning_backend.py:384-410), not this call.
register(Provider(
    library="sglang", signature="gla_simple_recurrence",
    kernel_impl="sglang_seg_la_fwd",
    import_name="sglang",
    serves=KernelSignature(takes_state=True, signature="gla_simple_recurrence",
                           phases=("prefill",), activation=("bf16", "fp32"),
                           divisible=(("d_head", 64),)),
    build=_sglang_seg_la_build,
))
register(Provider(
    library="sglang", signature="gla_simple_recurrence",
    kernel_impl="sglang_seg_la_fwd",
    import_name="sglang",
    serves=KernelSignature(takes_state=True, signature="gla_simple_recurrence",
                           phases=("decode",), activation=("bf16", "fp32"),
                           enum=_ONE_TOKEN,
                           divisible=(("d_head", 128),)),
    build=_sglang_seg_la_build,
))


def _torch_gla_build(inv, env, inputs) -> BenchCase:
    """The recurrence, eager, from the cell's state — and writing the final
    state back into its copy, as the serving launch writes its slot."""
    from ..refs import _gla_scan
    q, k, v, slope, state = _gla_args(inv, inputs)

    def run():
        return _gla_scan(q, k, v, slope, state, final_state=state)
    return BenchCase(run=run, to_canonical=lambda o: o.float())


#: The torch floors below RUN the oracle's own scan (`_delta_rule_scan`,
#: `_ssd_scan`, `_gla_scan`), so they cost what it costs: one Python step per
#: token, every step launching a handful of small kernels. At 262,144 tokens
#: that is 48 seconds for ONE launch — measured, in the queue's own log —
#: against about 8 ms of arithmetic, so the number says how slow an eager
#: scan is and nothing about the hardware. It is also a row the oracle has
#: already declined to check, because it declines past the same step count.
#:
#: So the floor stops where the oracle stops, reading the SAME declaration
#: rather than a second number: one fact about one implementation. Past it a
#: cell has no torch floor, which coverage counts as the gap it is (the
#: framework kernels still measure it — they do the same maths in one fused
#: launch).
_SCAN_FLOOR_CEILING = (("s_query", SCAN_MAX_STEPS),)


register(Provider(
    library="torch", signature="gla_simple_recurrence",
    kernel_impl="torch_gla_simple_recurrence",
    import_name="torch",
    serves=KernelSignature(takes_state=True, signature="gla_simple_recurrence",
                           activation=TORCH_FLOAT,
                           max_dims=_SCAN_FLOOR_CEILING),
    build=_torch_gla_build,
))


def _torch_delta_build(inv, env, inputs) -> BenchCase:
    from ..refs import _delta_rule_scan
    q, k, v, g, beta, state = _delta_args(inv, inputs)
    state = state.clone()       # written back below; the input is shared

    def run():
        return _delta_rule_scan(q, k, v, g, beta, state, final_state=state)
    return BenchCase(run=run, to_canonical=lambda o: o.float())


register(Provider(
    library="torch", signature="gated_delta_rule",
    kernel_impl="torch_gated_delta_rule",
    import_name="torch",
    # Not fp32: `_delta_args` narrows an fp32 cell to bf16.
    serves=KernelSignature(takes_state=True, signature="gated_delta_rule",
                           activation=("bf16", "fp16"),
                           max_dims=_SCAN_FLOOR_CEILING),
    build=_torch_delta_build,
))


register(Provider(
    library="torch", signature="kda",
    kernel_impl="torch_kda",
    import_name="torch",
    serves=KernelSignature(takes_state=True, signature="kda", activation=("bf16", "fp16"),
                           max_dims=_SCAN_FLOOR_CEILING),
    build=_torch_delta_build,
))


def _torch_mamba2_build(inv, env, inputs) -> BenchCase:
    from ..refs import _ssd_scan
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    B = inputs["B"].to(dt)
    C = inputs["C"].to(dt)
    dtp = inputs["dt"].float()
    A = inputs["A"].float()
    D = inputs["D"].float()
    dt_bias = inputs["dt_bias"].float()

    def run():
        return _ssd_scan(x.float(), dtp, A, B.float(), C.float(), D, dt_bias)
    return BenchCase(run=run, to_canonical=lambda o: o.float())


register(Provider(
    library="torch", signature="mamba2_scan",
    kernel_impl="torch_mamba2_scan",
    import_name="torch",
    serves=KernelSignature(signature="mamba2_scan", activation=TORCH_FLOAT,
                           max_dims=_SCAN_FLOOR_CEILING),
    build=_torch_mamba2_build,
))


# ---------------------------------------------------------------------------
# Kimi-K3's KDA decode, two spans (v0.32.0 / v0.5.22):
#
#   fused_recurrent_kda (the per-channel plan binds this cell too — see the
#   signature): vLLM's Kimi-K3 copy `fused_recurrent_kda_packed_decode`
#   (models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py:582, kernel
#   :487), as kda.py:1338-1356 calls it on the packed post-conv q|k|v rows.
#   sglang runs the kernel already registered above for GLM.
#
#   fused_kda_decode, the whole mixer:
#     vLLM    `_custom_ops.fused_kda_decode` (csrc/kimi_k3/
#             fused_kda_decode_kernel.cu, checks :991-1075), the "native"
#             backend `resolve_kda_decode_backend` picks by default
#             (kda.py:286-313), called at kda.py:1012-1028.
#     sglang  `kda_fused_decode` (kernels/ops/attention/kda_fused_decode.py
#             :112), from srt/layers/attention/linear/kda_backend.py:891.
# ---------------------------------------------------------------------------

def _vllm_kda_packed_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    o = _kda_decode_operands(inv, inputs)
    B, HV, K = o["B"], o["HV"], o["K"]
    # The call site's operands: the post-conv q|k|v rows packed per token,
    # the raw gate split into heads `[1, B, H, K]`, beta `[1, B, H]`.
    mixed = torch.cat([o["q"].reshape(B, -1), o["k"].reshape(B, -1),
                       o["v"].reshape(B, -1)], dim=-1).contiguous()
    raw_g = o["f"].reshape(1, B, HV, K).contiguous()
    raw_beta = o["b"].reshape(1, B, HV).contiguous()
    return BenchCase(
        run=lambda: fused_recurrent_kda_packed_decode(
            mixed, raw_g, raw_beta, o["A_log"].float(), o["dt_bias"].float(),
            o["lower_bound"], o["state"], o["slots"])[0],
        to_canonical=lambda out: out.transpose(0, 1).float())


register(Provider(
    library="vllm", signature="fused_recurrent_kda",
    kernel_impl="vllm_fused_recurrent_kda_packed_decode",
    import_name="vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent",
    serves=KernelSignature(takes_state=True, signature="fused_recurrent_kda", phases=("decode",),
                           activation=("bf16", "fp16"),
                           enum=_ONE_TOKEN,
                           # one head count for q|k|v (H == HV), K == V
                           requires_divisible_dims=_KDA_SQUARE),
    build=_vllm_kda_packed_decode_build,
    fused_op="fused_recurrent_kda",
))


def _kda_mixer_operands(inv, inputs):
    """The canonical mixer cell as both whole-mixer kernels take it: bf16
    activations and conv pool, fp32 weights / gate parameters / recurrent
    pool, pools of `B + 1` slots with the sequences in 1..B."""
    import torch
    dt = torch.bfloat16
    q, k, v, f, b, g = (inputs[n].to(dt) for n in ("q", "k", "v", "f", "b", "g"))
    B = int(q.shape[0])
    HV = int(inputs["A_log"].shape[0])
    D = int(inputs["o_norm_w"].shape[0])
    x = torch.cat([q, k, v], dim=-1).reshape(B, -1).contiguous()
    conv = inputs["conv_state"].to(dt)                    # [B, 3·H·D, W−1]
    state = torch.zeros(B + 1, HV, D, D, dtype=torch.float32, device=q.device)
    state[1:] = inputs["initial_state"].float()
    slots = torch.arange(1, B + 1, dtype=torch.int32, device=q.device)
    return dict(x=x, conv=conv, f=f, b=b, g=g, B=B, HV=HV, D=D, state=state,
                slots=slots,
                w=[inputs[n].float() for n in ("q_conv_w", "k_conv_w", "v_conv_w")],
                A_log=inputs["A_log"].float().contiguous(),
                dt_bias=inputs["dt_bias"].float().contiguous(),
                norm_w=inputs["o_norm_w"].float().contiguous(),
                lower_bound=float(inputs["lower_bound"]),
                eps=float(inputs["norm_eps"]))


def _vllm_fused_kda_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    from vllm import _custom_ops as ops
    o = _kda_mixer_operands(inv, inputs)
    B, HV, D = o["B"], o["HV"], o["D"]
    # `decode_conv1d_weight [3, W, H·D]` — each segment's taps transposed
    # (kda.py:505-530) — and the conv pool `[slots, 3·H·D, W−1]` in the DS
    # layout (fused_kda_decode_kernel.cu:1058-1067 takes SD or DS).
    weight = torch.stack([w.transpose(0, 1) for w in o["w"]]).contiguous()
    conv = torch.zeros(B + 1, *o["conv"].shape[1:], dtype=o["conv"].dtype,
                       device=o["conv"].device)
    conv[1:] = o["conv"]
    out = torch.empty(1, B, HV, D, dtype=torch.bfloat16, device=conv.device)
    raw_g = o["f"].reshape(1, B, HV, D).contiguous()
    raw_beta = o["b"].reshape(1, B, HV).contiguous()
    gate = o["g"].reshape(B, HV, D).contiguous()
    return BenchCase(
        run=lambda: ops.fused_kda_decode(
            x=o["x"], weight=weight, bias=None, conv_state=conv, raw_g=raw_g,
            raw_beta=raw_beta, A_log=o["A_log"], dt_bias=o["dt_bias"],
            state_indices=o["slots"], state=o["state"], out=out,
            lower_bound=o["lower_bound"], output_gate=gate,
            norm_weight=o["norm_w"], norm_eps=o["eps"]),
        to_canonical=lambda _o: out.transpose(0, 1).float())


def _sglang_fused_kda_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.kda_fused_decode import kda_fused_decode
    o = _kda_mixer_operands(inv, inputs)
    B, HV, D = o["B"], o["HV"], o["D"]
    # The conv pool `[slots, W−1, 3·H·D]` (SD), the taps `[W, H·D]` per
    # segment, and a zero conv bias (kda_fused_decode.cuh:962-970).
    conv = torch.zeros(B + 1, o["conv"].shape[2], o["conv"].shape[1],
                       dtype=o["conv"].dtype, device=o["conv"].device)
    conv[1:] = o["conv"].transpose(1, 2)
    w_q, w_k, w_v = (w.transpose(0, 1).contiguous() for w in o["w"])
    bias = torch.zeros(o["x"].shape[1], dtype=torch.float32, device=conv.device)
    a = o["f"].reshape(B, HV * D).contiguous()
    b = o["b"].reshape(B, HV).contiguous()
    gate = o["g"].reshape(B, HV * D).contiguous()
    return BenchCase(
        run=lambda: kda_fused_decode(
            o["x"], a, b, conv, w_q, w_k, w_v, bias, o["A_log"], o["dt_bias"],
            gate, o["norm_w"], o["state"], o["slots"], D ** -0.5, o["eps"],
            lower_bound=o["lower_bound"]),
        to_canonical=lambda out: out.transpose(0, 1).float())


def _torch_fused_kda_decode_build(inv, env, inputs) -> BenchCase:
    """The mixer's naive ops, eager, at the cell's precision: activations and
    the conv pool at the activation dtype, the recurrent pool fp32 as the
    graph carries it (`_kda_mixer_operands` hands the kernels the same two
    pools). Both pools are WRITTEN inside the timed call — the convs roll
    one tap and the step stores its new state, traffic the span bills; a
    floor that only computes the output never pays the stores the launch it
    floors performs."""
    import torch
    from ..refs import _delta_rule_scan, kda_bounded_gates
    dt = _torch_dtype(inv.dtype)
    q, k, v, gate = (inputs[n].to(dt) for n in ("q", "k", "v", "g"))
    cq, ck = q.shape[-1], k.shape[-1]
    B, HV = q.shape[0], int(inputs["A_log"].shape[0])
    conv = inputs["conv_state"].to(dt).clone()
    state = inputs["initial_state"].float().clone()
    taps = [inputs[n].float() for n in ("q_conv_w", "k_conv_w", "v_conv_w")]
    heads = lambda x: x.reshape(B, 1, HV, -1)

    def conv_step(x, pool, w):
        """silu(Σ window·w) on the taps plus the new token, the pool rolled
        — the store the fused launch's in-place update pays."""
        window = torch.cat([pool, x.transpose(1, 2)], dim=-1)
        pool.copy_(window[..., 1:])
        acc = (window.float() * w[None]).sum(-1)
        return torch.nn.functional.silu(acc).to(dt)[:, None, :]

    def run():
        qc = conv_step(q, conv[:, :cq], taps[0])
        kc = conv_step(k, conv[:, cq:cq + ck], taps[1])
        vc = conv_step(v, conv[:, cq + ck:], taps[2])
        g, beta = kda_bounded_gates(inputs["f"].to(dt), inputs["b"].to(dt),
                                    inputs["A_log"], inputs["dt_bias"],
                                    inputs["lower_bound"])
        o = _delta_rule_scan(heads(qc), heads(kc), heads(vc), g, beta,
                             state, final_state=state)
        rstd = torch.rsqrt(o.float().square().mean(-1, keepdim=True)
                           + float(inputs["norm_eps"])).to(dt)
        return (o * rstd * inputs["o_norm_w"].to(dt)
                * torch.sigmoid(heads(gate)))
    return BenchCase(run=run, to_canonical=lambda out: out.float())


_KDA_MIXER_GEOMETRY = (("s_query", (1,)), ("d_lin_k_head", (128,)),
                       ("d_lin_v_head", (128,)), ("s_conv_kernel", (4,)))
for _lib, _impl, _imp, _build, _heads in (
    # The head counts each kernel instantiates: vLLM checks 12/24/48/96
    # (fused_kda_decode_kernel.cu:1026-1028); sglang's JIT 3/6/12
    # (kda_fused_decode.py:37).
    ("vllm", "vllm_c_fused_kda_decode", "vllm", _vllm_fused_kda_decode_build,
     (12, 24, 48, 96)),
    ("sglang", "sglang_kda_fused_decode",
     "sglang.kernels.ops.attention.kda_fused_decode",
     _sglang_fused_kda_decode_build, (3, 6, 12)),
):
    register(Provider(
        library=_lib, signature="fused_kda_decode", kernel_impl=_impl,
        import_name=_imp,
        serves=KernelSignature(takes_state=True,
            signature="fused_kda_decode", phases=("decode",),
            activation=("bf16",),
            enum=(*_KDA_MIXER_GEOMETRY, ("n_lin_k_heads", _heads),
                  ("n_lin_v_heads", _heads)),
            requires_divisible_dims=_KDA_SQUARE),
        build=_build, fused_op="fused_kda_decode",
    ))
register(Provider(
    library="torch", signature="fused_kda_decode",
    kernel_impl="torch_fused_kda_decode", import_name="torch",
    # One token per sequence: the convs shift one tap and the step is one
    # recurrence, which is the span the plan prices.
    serves=KernelSignature(takes_state=True, signature="fused_kda_decode", phases=("decode",),
                           activation=TORCH_FLOAT, enum=_ONE_TOKEN),
    build=_torch_fused_kda_decode_build, fused_op="fused_kda_decode",
))
