"""A kernel that no corpus cell reaches says so, where the claim is made.

A provider's `serves` is a promise: the coverage audit counts a cell as
served when some provider claims it, and the API publishes the claim. Three
different facts can leave a registered kernel with no rows, and they want
opposite actions — it must not be measured (`disabled_reason`), the card is
wrong for it (`min_sm`), or it is right and nothing in this corpus asks for
it. Only the first two were declarable, so the third read as "no rows, no
reason declared", which is also how a declaration BUG reads. Nine kernels
sat in that bucket: marlin's weight-only FP8 against a corpus whose FP8
models quantise the activation too, and the fp16/fp32 twins of three
launches every corpus model ships in bf16.

What is tested here is the pair, not the sentence: a kernel that declares
the reason must also still REFUSE the cells the corpus has (otherwise the
reason is stale and the kernel was reachable all along), and a kernel that
declares none must not be one of the known-honest ones. The corpus facts
each reason names are asserted against cells built here, so a declaration
quietly widened to grab the corpus's cells — the "gate wider than the
build" failure the closed declaration exists to prevent — turns this red.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch")

from kernel_bench.env import Env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import Provider, all_providers
from tools.unverified_providers import why


def _env() -> Env:
    """Every library installed and an H100, so only SCOPE is under test."""
    return Env(torch_version=None, device="cuda", device_name=None,
               gpu_sku="H100_SXM", peak_bw_gbps=None, sm="sm_90",
               libs={p.import_name: "0" for p in all_providers()})


def _by_impl(impl: str) -> Provider:
    for p in all_providers():
        if p.kernel_impl == impl:
            return p
    raise AssertionError(f"{impl} is not registered")


def _gemm_cell(weight: str, activation: str, granularity=None) -> Invocation:
    env = {"activation": activation, "weight": weight, "output": activation}
    if granularity is not None:
        env["weight_granularity"] = granularity
        env["metadata"] = "fp32"
    return Invocation.from_job({
        "signature": "gemm", "op": "linear", "phase": "prefill",
        # Divisible by marlin's 128 reduction and 64 output tiles, so the
        # shape is never what refuses: the dtypes are.
        "dims": {"B": 1, "s_query": 64, "d_reduce": 512, "d_out": 512},
        "dtype_env": env})


def _norm_cell(signature: str, dtype: str, **dims) -> Invocation:
    base = {"B": 1, "s_query": 64, "d_model": 512}
    base.update(dims)
    return Invocation.from_job({
        "signature": signature, "op": signature, "phase": "prefill",
        "dims": base,
        "dtype_env": {"activation": dtype, "weight": dtype, "output": dtype}})


class TestTheReasonIsDeclaredWhereTheClaimIs:
    def test_the_audit_reads_the_declared_reason(self):
        """`unverified_providers.why` is the one place the three are read.

        Red before the field existed: `Provider` took no such keyword, and
        marlin's weight-only FP8 printed "no reason declared" beside a
        kernel that is correct and simply unreachable here.
        """
        p = _by_impl("marlin_fp8_w8a16")
        assert p.unreached_reason, "the honest gap is declared on the kernel"
        reason = why(p)
        assert "no cell in this corpus" in reason
        assert "W8A8" in reason
        assert reason != "no reason declared"

    def test_a_kernel_with_no_reason_still_reads_as_a_to_do(self):
        """The field is not a blanket: an undeclared kernel keeps shouting.

        `sglang_softcap_to_float32` is the case — it writes fp32 logits and
        every planned cell of its signature is billed bf16, which is our
        walk's gap to close (`docs/known_limitations.md` §4.3), not an honest absence.
        It must NOT carry a reason.
        """
        p = _by_impl("sglang_softcap_to_float32")
        assert not p.unreached_reason
        assert why(p) == "no reason declared"

    def test_the_reason_travels_into_the_published_record(self):
        """A consumer reading the claim reads the qualification with it."""
        record = _by_impl("marlin_fp8_w8a16").capabilities()
        assert record["weight_dtypes"] == ["fp8_e4m3"]
        assert "W8A8" in record["unreached_reason"]
        assert "unreached_reason" not in (
            _by_impl("marlin_fp4_e2m1_w4a16").capabilities())


class TestTheCorpusFactEachReasonNames:
    def test_weight_only_fp8_is_not_what_this_corpus_runs(self):
        """marlin W8A16 serves its own pair and refuses the corpus's.

        The corpus's FP8 `gemm` cells are W8A8 with per-tile or per-channel
        weight scales (121,100 of them across the plans); marlin's path
        wants a 16-bit activation and one scale per tensor. Both halves are
        asserted, so widening either to reach a corpus cell turns this red.
        """
        p = _by_impl("marlin_fp8_w8a16")
        env = _env()
        assert p.accepts(_gemm_cell("fp8_e4m3", "bf16", "per-tensor"), env), \
            "the kernel still serves the pair it was audited for"
        for granularity in ("per-tile:128x128", "per-tile:32x32",
                            "per-channel"):
            cell = _gemm_cell("fp8_e4m3", "fp8_e4m3", granularity)
            assert not p.accepts(cell, env), granularity
        assert not p.accepts(_gemm_cell("bf16", "bf16"), env)

    @pytest.mark.parametrize("impl,signature,dims", [
        # First, and the one that goes red if the twins are ever widened to
        # the corpus's width: four kernels register off one dtype loop.
        ("sglang_mamba2_rmsnorm_gated_fp16", "mamba2_rmsnorm_gated",
         {"d_ssm_state": 128, "n_ssm_groups": 1, "n_ssm_heads": 8}),
        ("torch_mamba2_rmsnorm_gated_fp32", "mamba2_rmsnorm_gated",
         {"d_ssm_state": 128, "n_ssm_groups": 1, "n_ssm_heads": 8}),
        ("vllm_ple_gate_fp16", "ple_gate", {}),
        ("torch_ple_gate_fp16", "ple_gate", {}),
    ])
    def test_a_dtype_twin_refuses_the_width_the_corpus_ships(
            self, impl, signature, dims):
        """Every corpus model with these mixers ships bf16.

        The twins are the same launch at another width, so they must refuse
        a bf16 cell and accept their own; a twin that accepts bf16 would be
        measured against a launch the model does not make, and its declared
        reason would be false.
        """
        p = _by_impl(impl)
        env = _env()
        own = p.serves.activation[0]
        assert not p.accepts(_norm_cell(signature, "bf16", **dims), env)
        assert p.accepts(_norm_cell(signature, own, **dims), env)
        assert "bf16" in p.unreached_reason
