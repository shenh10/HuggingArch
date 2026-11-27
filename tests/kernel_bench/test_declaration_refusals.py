"""Why a declaration refused a cell, not just that it did.

82 of 181 registered providers have no row in any corpus, and a gate that
answers yes/no cannot say which of them is a bug. A `max_dims` ceiling the
corpus always crosses means the card is wrong for that kernel and the rows
were never going to exist; an `activation` list missing the one dtype the
corpus uses is a declaration to fix and rows that should exist. Those need
opposite actions, and counting which CLAUSE refuses over real planned cells
is what separates them.

The shape matters as much as the answer: `refusals` is a generator and
`accepts` is `next(refusals(...), None) is None`, so there is ONE
implementation of the gate. A predicate beside a diagnostic that re-derives
the same conditions is two readings of one fact, and they drift — which is
the failure this repo keeps finding in other forms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch")

from kernel_bench.env import detect_env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import KernelSignature


#: A gemm declaration's two material slots, declared at the fixture's dtype
#: so a test of another clause sees only that clause.
_BF16 = dict(weight=("bf16",), activation=("bf16",))


def _cell(**over) -> Invocation:
    job = {
        "signature": "gemm", "op": "linear",
        "dims": {"B": 1, "s_query": 64, "d_reduce": 256, "d_out": 256},
        "phase": "prefill",
        "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                      "output": "bf16", "metadata": "fp32"},
    }
    job.update(over)
    return Invocation.from_job(job)


class TestTheGateAndTheDiagnosisAreOneThing:
    def test_accepts_is_exactly_no_refusals(self):
        serves = KernelSignature(signature="gemm", weight=("bf16", "fp8_e4m3"),
                                 activation=("bf16",))
        good, bad = _cell(), _cell(dtype_env={
            "activation": "fp8_e4m3", "weight": "fp8_e4m3", "kv": "bf16",
            "output": "fp8_e4m3", "metadata": "fp32",
            "weight_granularity": "per-tile:128x128"})
        assert list(serves.refusals(good)) == []
        assert serves.accepts(good) is True
        assert list(serves.refusals(bad)) == [
            "activation:fp8_e4m3 is not served (serves bf16)"]
        assert serves.accepts(bad) is False

    def test_a_refusal_names_the_clause_and_the_value(self):
        serves = KernelSignature(signature="gemm", **_BF16,
                                 divisible=(("d_reduce", 128),),
                                 max_dims=(("s_query", 32),))
        said = list(serves.refusals(_cell()))
        # Both clauses refuse this cell, and the tokens say which: 256 % 128
        # is fine, so the divisor is not among them, while s_query 64 > 32 is.
        assert "max_dims:s_query" in said[0] or "max_dims" in said[0]
        assert not any("divisible" in token for token in said)

    def test_each_refusal_names_the_failed_values(self):
        serves = KernelSignature(signature="gemm", **_BF16,
                                 divisible=(("d_reduce", 128),),
                                 max_dims=(("s_query", 32),))
        bad = _cell(dims={"B": 1, "s_query": 64, "d_reduce": 129,
                          "d_out": 256})
        assert list(serves.refusals(bad)) == [
            "divisible:d_reduce=129 is not divisible by 128",
            "max_dims:s_query=64 exceeds 32",
        ]

    def test_an_absent_divisible_dim_does_not_read_as_zero_modulo(self):
        serves = KernelSignature(signature="gemm", **_BF16,
                                 divisible=(("d_reduce", 128),))
        assert list(serves.refusals(_cell(dims={
            "B": 1, "s_query": 64, "d_out": 256}))) == [
            "divisible:d_reduce=0 is not a positive multiple of 128"]

    def test_several_clauses_can_refuse_one_cell(self):
        serves = KernelSignature(signature="gemm", weight=("bf16",),
                                 activation=("fp8_e4m3",),
                                 phases=("decode",))
        said = list(serves.refusals(_cell()))
        assert len(said) == 2
        assert any(token.startswith("activation:") for token in said)
        assert any(token.startswith("phases:") for token in said)

    def test_the_wrong_signature_stops_there(self):
        """The remaining clauses are about another contract.

        Reporting a `gemm` declaration's dtype list against an attention
        cell would count a refusal that means nothing, and the whole point
        of the tally is which clause to go and fix.
        """
        serves = KernelSignature(signature="gemm", weight=("bf16",),
                                 activation=("fp8_e4m3",),
                                 phases=("decode",))
        said = list(serves.refusals(_cell(signature="gqa_attention")))
        assert said == ["signature"]

    def test_it_is_lazy_so_the_gate_still_short_circuits(self):
        import inspect
        serves = KernelSignature(signature="gemm")
        assert inspect.isgenerator(serves.refusals(_cell()))


class TestAProviderRefusesForItsOwnReasons:
    def test_a_disabled_kernel_says_disabled_and_nothing_else(self):
        """Its scope is not the point once it is switched off.

        A disabled provider's cells are a declared gap, and listing which
        of its clauses would also have refused would file them under the
        clause instead of under the disablement.
        """
        from kernel_bench.providers import Provider

        env = detect_env()
        p = Provider(library="torch", signature="gemm",
                     kernel_impl="t", import_name="torch",
                     disabled_reason="under investigation",
                     serves=KernelSignature(signature="gemm",
                                            weight=("fp8_e4m3",),
                                            activation=("fp8_e4m3",)),
                     build=lambda inv, env, inputs: None)
        assert list(p.refusals(_cell(), env)) == ["disabled"]
        assert p.accepts(_cell(), env) is False

    def test_every_registered_provider_answers_both_ways_alike(self):
        """`accepts` and `refusals` cannot disagree on any real declaration.

        The property that makes one the other's definition, asserted over
        the whole registry rather than a fixture — and asked the way the
        only caller asks it (`applicable` filters the signature first,
        because a lambda gate is written for its own cells and
        `moe_gate`'s raises on a GEMM cell rather than returning False).
        """
        from kernel_bench.providers import all_providers

        env = detect_env()
        cell = _cell()
        asked = 0
        for p in all_providers():
            if p.signature != cell.signature:
                continue
            asked += 1
            empty = next(p.refusals(cell, env), None) is None
            assert p.accepts(cell, env) is empty, (
                f"{p.library}:{p.kernel_impl} answers differently")
        assert asked > 5, "the registry should hold several gemm providers"


class TestADtypeSlotBeyondTheFirstClassThree:
    """`dtype_slots` gates a slot that has no field of its own — kv is the
    case: the cache format a launch writes is a permanent fact of the
    launch, so it is refused through the declaration, where smoke, the
    published capability record and the stale-row audit all read it."""

    def test_the_refusal_names_the_slot_and_the_value(self):
        serves = KernelSignature(signature="gemm", **_BF16,
                                 dtype_slots=(("kv", ("fp8_e4m3",)),))
        good = _cell(dtype_env={"activation": "bf16", "weight": "bf16",
                                "kv": "fp8_e4m3"})
        assert list(serves.refusals(good)) == []
        assert list(serves.refusals(_cell())) == [
            "kv:bf16 is not served (serves fp8_e4m3)"]

    def test_a_slot_the_cell_does_not_carry_reads_as_the_activation(self):
        """`Invocation.dtype_for`'s own rule: no kv slot, so the cache is
        billed at the activation's dtype — which is not the record's."""
        serves = KernelSignature(signature="gemm", weight=("bf16",),
                                 activation=("bf16", "fp8_e4m3"),
                                 output=("bf16", "fp8_e4m3"),
                                 dtype_slots=(("kv", ("fp8_e4m3",)),))
        no_kv = _cell(dtype_env={"activation": "bf16", "weight": "bf16"})
        assert list(serves.refusals(no_kv)) == [
            "kv:bf16 is not served (serves fp8_e4m3)"]
        fp8_act = _cell(dtype_env={"activation": "fp8_e4m3", "weight": "bf16"})
        assert list(serves.refusals(fp8_act)) == []

    def test_the_constraint_is_published_with_the_other_dtypes(self):
        import json
        serves = KernelSignature(signature="gemm",
                                 dtype_slots=(("kv", ("fp8_e4m3",)),))
        d = serves.describe()
        json.dumps(d)
        assert d["kv_dtypes"] == ["fp8_e4m3"]

    def test_registration_refuses_a_slot_with_its_own_field(self):
        """`weight`/`activation`/`output` here would be two channels for one
        gate — the drift the declaration exists to remove."""
        from kernel_bench.providers import Provider, register
        p = Provider(library="torch", signature="gemm", kernel_impl="probe",
                     import_name="torch",
                     serves=KernelSignature(
                         signature="gemm",
                         dtype_slots=(("weight", ("bf16",)),)),
                     build=lambda inv, env, inputs: None)
        with pytest.raises(ValueError, match="first-class"):
            register(p)

    def test_registration_refuses_a_token_the_corpus_cannot_price(self):
        from kernel_bench.providers import Provider, register
        p = Provider(library="torch", signature="gemm", kernel_impl="probe",
                     import_name="torch",
                     serves=KernelSignature(
                         signature="gemm",
                         dtype_slots=(("kv", ("fp9_e1",)),)),
                     build=lambda inv, env, inputs: None)
        with pytest.raises(ValueError, match="corpus dtype token"):
            register(p)

    def test_registration_refuses_an_empty_token_list(self):
        from kernel_bench.providers import Provider, register
        p = Provider(library="torch", signature="gemm", kernel_impl="probe",
                     import_name="torch",
                     serves=KernelSignature(
                         signature="gemm", dtype_slots=(("kv", ()),)),
                     build=lambda inv, env, inputs: None)
        with pytest.raises(ValueError, match="empty"):
            register(p)


class TestASecondaryWriteIsDeclared:
    """A launch that writes more than its primary output says so: the
    one-slot `output` cannot state both, and an unpublished write is a
    capability contract that under-describes the measured launch."""

    def test_published_json_safe(self):
        import json
        serves = KernelSignature(signature="gemm", output=("fp8_e4m3",),
                                 secondary_outputs=(("weights", "fp32"),))
        d = serves.describe()
        json.dumps(d)
        assert d["secondary_outputs"] == [{"tensor": "weights",
                                           "dtype": "fp32"}]
        assert "secondary_outputs" not in KernelSignature(
            signature="gemm").describe()

    def test_registration_refuses_a_token_the_corpus_cannot_price(self):
        from kernel_bench.providers import Provider, register
        p = Provider(library="torch", signature="gemm", kernel_impl="probe",
                     import_name="torch",
                     serves=KernelSignature(
                         signature="gemm",
                         secondary_outputs=(("weights", "fp9_e1"),)),
                     build=lambda inv, env, inputs: None)
        with pytest.raises(ValueError, match="corpus dtype token"):
            register(p)


def _attention(**dims) -> Invocation:
    base = {"B": 16, "d_head": 192, "kv_block_size": 16, "n_heads": 64,
            "n_kv_heads": 4, "s_kv": 1152, "s_query": 1}
    base.update(dims)
    return Invocation.from_job({
        "signature": "gqa_attention", "op": "flash_attention", "dims": base,
        "phase": "decode",
        "dtype_env": {"activation": "bf16", "kv": "bf16", "output": "bf16",
                      "metadata": "fp32"}})


def _named(kernel_impl):
    from kernel_bench.providers import all_providers
    return next(p for p in all_providers() if p.kernel_impl == kernel_impl)


class TestTheDeclarationIsClosed:
    """A declaration lists what a kernel serves, and silence serves
    nothing: no "() = any" slot, no list of exclusions. Which slots and axes
    it must speak for is the signature's, not a list beside the kernels."""

    def _probe(self, **serves):
        from kernel_bench.providers import Provider
        return Provider(library="torch", signature="gemm",
                        kernel_impl="probe", import_name="torch",
                        serves=KernelSignature(signature="gemm", **serves),
                        build=lambda inv, env, inputs: None)

    def test_an_undeclared_material_slot_refuses_registration(self):
        # gemm makes `activation` and `weight` material; leaving one empty
        # would serve no cell, and before it served every one.
        from kernel_bench.providers import register
        with pytest.raises(ValueError, match="'weight'"):
            register(self._probe(activation=("bf16",)))

    def test_and_refuses_the_cell_when_asked_unregistered(self):
        serves = KernelSignature(signature="gemm", activation=("bf16",))
        assert list(serves.refusals(_cell())) == [
            "weight:bf16 is not served (serves none)"]

    def test_an_undeclared_output_is_the_activations(self):
        # A kernel that computes in the cell's dtype writes it, so a cell
        # whose output is FP8 (a SwiGLU fused with its cast) is not its own.
        serves = KernelSignature(signature="gemm", **_BF16)
        cast = _cell(dtype_env={"activation": "bf16", "weight": "bf16",
                                "output": "fp8_e4m3"})
        assert list(serves.refusals(cast)) == [
            "output:fp8_e4m3 is not served (serves bf16)"]
        assert serves.writes("bf16") == ("bf16",)

    def test_a_computing_fallback_must_say_what_it_writes(self):
        from kernel_bench.providers import register
        p = self._probe(weight=("fp8_e4m3",), activation=("fp8_e4m3",))
        p.computes_at = {"weight": "bf16", "activation": "bf16"}
        with pytest.raises(ValueError, match="declare the `output`"):
            register(p)

    def test_an_unnamed_format_or_scale_is_the_plain_tensor(self):
        serves = KernelSignature(signature="gemm", **_BF16)
        record = _cell(dtype_env={"activation": "bf16", "weight": "bf16",
                                  "kv": "fp8_e4m3",
                                  "kv_format": "fp8_per_tensor"})
        assert any(token.startswith("kv_format:fp8_per_tensor")
                   for token in serves.refusals(record))
        scaled = _cell(dtype_env={"activation": "bf16", "weight": "bf16",
                                  "activation_granularity": "per-group:128"})
        assert list(serves.refusals(scaled)) == [
            "activation_granularity:per-group:128 is not served "
            "(serves no activation scale)"]

    def test_a_carried_variant_axis_must_be_declared(self):
        # A sink is gqa's optional axis: undeclared, the kernel declines it;
        # the default geometry's axes need no mention.
        bare = KernelSignature(signature="gqa_attention",
                               activation=("bf16",),
                               dtype_slots=(("kv", ("bf16",)),))
        assert list(bare.refusals(_attention())) == []
        assert list(bare.refusals(_attention(n_attn_sinks=1))) == [
            "variant_dims:n_attn_sinks is present but unserved"]
        sinks = KernelSignature(signature="gqa_attention",
                                activation=("bf16",),
                                dtype_slots=(("kv", ("bf16",)),),
                                variant_dims=("n_attn_sinks",))
        assert sinks.accepts(_attention(n_attn_sinks=1))

    @pytest.mark.parametrize("serves,match", [
        (dict(variant_dims=("d_head",)), "not variant axes"),
        (dict(variant_dims=("s_window",), requires_dims=("s_window",)),
         "named twice"),
    ])
    def test_variant_names_are_the_signatures_own_once(self, serves, match):
        from kernel_bench.providers import Provider, register
        p = Provider(library="torch", signature="gqa_attention",
                     kernel_impl="probe", import_name="torch",
                     serves=KernelSignature(
                         signature="gqa_attention", activation=("bf16",),
                         dtype_slots=(("kv", ("bf16",)),), kv_block_sizes=(),
                         **serves),
                     build=lambda inv, env, inputs: None)
        with pytest.raises(ValueError, match=match):
            register(p)


class TestOneHeadWidthForQKAndV:
    """Kernels whose launch takes ONE head width for Q·K and for V decline
    a cell whose V is narrower (`d_v_head`, carried only when it differs).

    FlashInfer's decode wrapper planned MiMo-V2-Flash's 192/128 attention at
    192 for both, read V 192 wide out of a 128-wide cache, and faulted with
    an illegal address that poisoned the process; FA2 refused it by name.
    """

    @pytest.mark.parametrize("kernel_impl", [
        "flashinfer_paged_decode", "flash_attention_v2_kvcache"])
    def test_a_narrower_v_is_declined(self, kernel_impl):
        serves = _named(kernel_impl).serves
        assert any(token.startswith("variant_dims:d_v_head")
                   for token in serves.refusals(_attention(d_v_head=128)))

    @pytest.mark.parametrize("kernel_impl", [
        "flashinfer_paged_decode", "flash_attention_v2_kvcache"])
    def test_an_equal_v_still_runs(self, kernel_impl):
        # 128-wide: inside FA2's ceiling, so only the V clause is under test.
        assert _named(kernel_impl).serves.accepts(_attention(d_head=128))

    def test_the_wrappers_that_take_head_dim_vo_keep_serving_it(self):
        # The paged prefill wrapper (verify) and the ragged prefill wrapper
        # are planned with `head_dim_vo`, so the asymmetric cell is theirs.
        verify = _attention(d_v_head=128, s_query=4)
        assert _named("flashinfer_paged_verify").serves.accepts(verify)
        prefill = Invocation.from_job({
            "signature": "gqa_attention", "op": "flash_attention",
            "dims": {"B": 1, "d_head": 192, "d_v_head": 128, "n_heads": 64,
                     "n_kv_heads": 4, "s_kv": 1024, "s_query": 1024,
                     "kv_block_size": 16},
            "phase": "prefill",
            "dtype_env": {"activation": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"}})
        assert _named("flashinfer_ragged_prefill").serves.accepts(prefill)


class TestAGranularityRefusalNamesTheOperand:
    def test_lings_per_channel_weight_is_refused_as_weight_granularity(self):
        """The first cell Ling-2.6's plan hands `deepgemm_fp8_per_head`.

        Its activation granularity does not match DeepGEMM either. Both
        clauses belong in the diagnosis, but neither may take the other's
        name: the checkpoint's per-channel weight is the declaration's first
        mismatch, and the launch's scale group is the second.
        """
        cell = Invocation.from_job({
            "signature": "per_head_gemm", "op": "per_head_linear",
            "dims": {"B": 1, "d_reduce": 128, "d_out": 512,
                     "n_heads": 64, "s_query": 1},
            "phase": "decode",
            "dtype_env": {"activation": "fp8_e4m3",
                          "activation_granularity": "per-token",
                          "metadata": "bf16", "output": "bf16",
                          "weight": "fp8_e4m3",
                          "weight_granularity": "per-channel"}})
        said = list(_named("deepgemm_fp8_per_head_masked").refusals(
            cell, detect_env()))
        assert said[0].startswith("weight_granularity:per-channel")
        assert said[0].endswith("(serves per-tile:128x128)")
        assert any(token.startswith("activation_granularity:per-token")
                   for token in said)


def _fp8_block_gemm(tile=128, **dims) -> Invocation:
    base = {"B": 1, "s_query": 262144, "d_reduce": 4096, "d_out": 14848}
    base.update(dims)
    return _cell(dims=base, dtype_env={
        "activation": "fp8_e4m3", "weight": "fp8_e4m3", "output": "bf16",
        "metadata": "fp32", "activation_granularity": f"per-group:{tile}",
        "weight_granularity": f"per-tile:{tile}x{tile}"})


class TestTritonBlockFp8IndexesInInt32:
    """vLLM's Triton block-scaled GEMM forms every operand offset in int32,
    so a tensor past 2**31 elements wraps to an illegal address. The 14
    cells that faulted on the 4090 were all past it and none under it."""

    @pytest.mark.parametrize("dims", [
        # The faults, one per operand: C = M x N, then A = M x K.
        {"d_out": 14848, "d_reduce": 4096, "s_query": 262144},
        {"d_out": 512, "d_reduce": 5120, "s_query": 524288},
    ])
    def test_a_tensor_past_int32_offsets_is_declined(self, dims):
        serves = _named("vllm_triton_block_scaled_mm_128").serves
        assert not serves.accepts(_fp8_block_gemm(**dims))

    def test_the_boundary_itself_still_runs(self):
        # M x N == 2**31 exactly: the largest offset is 2**31 - 1.
        serves = _named("vllm_triton_block_scaled_mm_128").serves
        assert serves.accepts(_fp8_block_gemm(
            s_query=2 ** 31 // 8192, d_out=8192, d_reduce=1024))
        assert not serves.accepts(_fp8_block_gemm(
            s_query=2 ** 31 // 8192 + 1, d_out=8192, d_reduce=1024))

    def test_the_32_tile_is_the_same_kernel(self):
        assert not _named("vllm_triton_block_scaled_mm_32").serves.accepts(
            _fp8_block_gemm(tile=32, d_out=32768, d_reduce=1280,
                            s_query=131072))


class TestDeepGemmNeedsHopper:
    """DeepGEMM ships sm90/sm100/sm120 kernels only. In sglang's env it
    imports on an Ada card, JITs, and ptxas refuses `griddepcontrol`; the
    floor keeps it from being tried there at all."""

    @pytest.mark.parametrize("kernel_impl", [
        "deepgemm_fp8_blockwise", "deepgemm_fp8_m_grouped_contiguous",
        "deepgemm_fp8_m_grouped_masked", "deepgemm_fp8_per_head_masked"])
    def test_absent_on_ada_present_on_hopper(self, kernel_impl):
        from kernel_bench.env import Env
        p = _named(kernel_impl)

        def card(sm):
            return Env("2.9", "cuda", "x", "x", 1008.0, sm=sm,
                       libs={"deep_gemm": "2.8.2", "torch": "2.9"})
        assert not p.available(card("sm_89"))
        assert p.available(card("sm_90"))


class TestVllmVendoredFa3IsHoppersOnly:
    """vLLM picks its vendored FA3 only on sm90 (`fa_utils.py:99`) and
    refuses a sink below it (`flash_attn.py:469`); sglang allows FA3 on sm8x,
    so the floor is vLLM's alone."""

    @pytest.mark.parametrize("kernel_impl", [
        "vllm_flash_attention_v3_kvcache", "vllm_flash_attention_v3"])
    def test_vllm_absent_on_ada(self, kernel_impl):
        from kernel_bench.env import Env

        def card(sm):
            return Env("2.9", "cuda", "x", "x", 1008.0, sm=sm,
                       libs={"vllm": "0.32.0", "torch": "2.9"})
        assert not _named(kernel_impl).available(card("sm_89"))
        assert _named(kernel_impl).available(card("sm_90"))

    def test_sglang_keeps_ada(self):
        from kernel_bench.env import Env
        env = Env("2.9", "cuda", "x", "x", 1008.0, sm="sm_89",
                  libs={"sglang": "0.5.22", "torch": "2.9"})
        assert _named("sglang_flash_attention_v3_kvcache").available(env)


def _moe_gate_cell() -> Invocation:
    return Invocation.from_job({
        "signature": "moe_gate", "op": "moe_fused_gate", "phase": "decode",
        "dims": {"B": 2, "s_query": 4, "n_experts": 256,
                 "n_expert_groups": 8, "n_groups_active": 4,
                 "n_experts_active": 8},
        "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                      "output": "bf16", "metadata": "fp32"}})


class TestARetiredKernelIsAbsent:
    """sglang-kernel 0.4.5 retired the AOT `moe_fused_gate`; `sgl_kernel`
    still imports, so only the kernel's own version bound keeps it from being
    tried, cell by cell, in an env holding the audited release (sglang 0.5.22
    with sglang-kernel 0.4.9, on both boxes)."""

    @staticmethod
    def _env(sgl_kernel):
        from kernel_bench.env import Env
        return Env("2.9", "cuda", "x", "x", 1008.0, sm="sm_90",
                   libs={"sgl_kernel": sgl_kernel, "sglang": "0.5.22",
                         "torch": "2.9"})

    def test_absent_where_the_release_removed_it(self):
        from kernel_bench.providers import applicable
        p = _named("sgl_moe_fused_gate")
        env = self._env("0.4.9")
        assert not p.available(env)
        assert list(p.unavailability(env)) == [
            "sgl_kernel 0.4.9 is outside <0.4.5"]
        assert p not in applicable(_moe_gate_cell(), env)

    @pytest.mark.parametrize("sgl_kernel", ["0.4.4", "0.3.17.post1"])
    def test_present_where_the_release_still_carries_it(self, sgl_kernel):
        # Same provider, same cell, only the installed version differs: the
        # gate demonstrably reads the env.
        from kernel_bench.providers import applicable
        p = _named("sgl_moe_fused_gate")
        assert p.available(self._env(sgl_kernel))
        assert p in applicable(_moe_gate_cell(), self._env(sgl_kernel))

    def test_the_rest_of_the_library_is_unaffected(self):
        # The bound is this kernel's, not sglang's: the library's other
        # sgl_kernel-gated kernels still run on 0.4.9.
        from kernel_bench.providers import all_providers
        others = [p for p in all_providers()
                  if p.library == "sglang" and p.import_name == "sgl_kernel"
                  and p.kernel_impl != "sgl_moe_fused_gate" and not p.min_sm]
        assert others
        assert all(p.available(self._env("0.4.9")) for p in others)

    def test_a_malformed_bound_is_refused_at_registration(self):
        from dataclasses import replace

        from packaging.specifiers import InvalidSpecifier

        from kernel_bench.providers import register
        p = replace(_named("sgl_moe_fused_gate"), version_specifier="0.4.5")
        with pytest.raises(InvalidSpecifier):
            register(p)


class TestVllmGroupedTopkFollowsItsEntry:
    """`fused_grouped_topk` moved in vLLM v0.15 from `fused_moe/fused_moe.py`
    to `fused_moe/router/grouped_topk_router.py`, signature unchanged. The
    audited 0.32.0 has only the new path, a box on 0.11 only the old; the
    provider calls whichever the installed release carries."""

    _NEW = "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"
    _OLD = "vllm.model_executor.layers.fused_moe.fused_moe"

    def _install(self, monkeypatch, layout):
        import types

        import torch
        calls = []

        def fused_grouped_topk(*a, **k):
            calls.append(layout)
            g, topk = a[1], a[2]
            return (g.new_zeros(g.shape[0], topk),
                    torch.zeros(g.shape[0], topk, dtype=torch.long))
        for name in [m for m in sys.modules
                     if m == "vllm" or m.startswith("vllm.")]:
            monkeypatch.delitem(sys.modules, name)
        parts = self._OLD.split(".")
        for i in range(1, len(parts)):
            parent = types.ModuleType(".".join(parts[:i]))
            parent.__path__ = []     # a package: an absent child is NotFound
            monkeypatch.setitem(sys.modules, parent.__name__, parent)
        old = types.ModuleType(self._OLD)
        monkeypatch.setitem(sys.modules, self._OLD, old)
        if layout == "0.32":
            new = types.ModuleType(self._NEW)
            new.fused_grouped_topk = fused_grouped_topk
            monkeypatch.setitem(sys.modules, self._NEW, new)
        else:
            old.fused_grouped_topk = fused_grouped_topk
        return calls

    @pytest.mark.parametrize("layout", ["0.32", "0.11"])
    def test_the_installed_release_s_entry_is_called(self, monkeypatch, layout):
        import torch

        from kernel_bench.providers import moe_gate
        monkeypatch.setattr(moe_gate, "_require_cuda", lambda: None)
        calls = self._install(monkeypatch, layout)
        inputs = {"gating": torch.randn(8, 256), "bias": torch.zeros(256),
                  "num_expert_group": 8, "topk_group": 4, "topk": 8}
        case = _named("vllm_moe_c_grouped_topk").build(
            _moe_gate_cell(), None, inputs)
        case.run()
        assert calls == [layout]


class TestTorchSdpaHasNoSinkOperand:
    """The torch SDPA floor said it cannot take an attention sink only by
    raising NotImplementedError in its build, so every gpt-oss sink cell
    was planned for it, and the path's canary disabled it on the H100
    (2026-10-26). The declaration says it, and the plan drops the cell."""

    @staticmethod
    def _gpt_oss_canary(**over):
        dims = {"B": 1, "d_head": 64, "kv_block_size": 16, "n_attn_sinks": 1,
                "n_heads": 1, "n_kv_heads": 1, "s_kv": 1024,
                "s_query": 1024, "s_window": 128}
        dims.update(over)
        dims = {k: v for k, v in dims.items() if v is not None}
        return Invocation.from_job({
            "signature": "gqa_attention", "op": "flash_attention",
            "dims": dims, "phase": "prefill",
            "dtype_env": {"activation": "bf16", "kv": "bf16",
                          "output": "bf16", "weight": "bf16"}})

    def test_a_sink_cell_is_declined_by_the_declaration(self):
        assert any(token.startswith("variant_dims:n_attn_sinks")
                   for token in _named("torch_sdpa").serves.refusals(
                       self._gpt_oss_canary()))

    def test_the_same_cell_without_a_sink_is_still_served(self):
        assert _named("torch_sdpa").serves.accepts(
            self._gpt_oss_canary(n_attn_sinks=None))

    def test_flashinfers_ragged_prefill_declines_a_sink_too(self):
        # Its `run` takes no `sinks` (flashinfer 0.7.2.post1); the gpt-oss
        # canary failed with "unexpected keyword argument 'sinks'".
        serves = _named("flashinfer_ragged_prefill").serves
        assert any(token.startswith("variant_dims:n_attn_sinks")
                   for token in serves.refusals(self._gpt_oss_canary()))
        assert serves.accepts(self._gpt_oss_canary(n_attn_sinks=None))


class TestFlashInfersPagedWrappersDropASink:
    """On the H100 (flashinfer 0.7.2.post1, SM90) the paged decode and the
    paged prefill wrapper accept `sinks=` and return exactly what they return
    without it, so a gpt-oss sink cell was measured as sink-less attention:
    the cross-check saw a scale error (typical 0.012 / 0.024 against the
    bf16 band's 0.008) and the canaries disabled the decode paths
    (2026-10-26). Both decline the sink by declaration."""

    @staticmethod
    def _sink_cell(s_query, n_heads, kv_block_size):
        return Invocation.from_job({
            "signature": "gqa_attention", "op": "flash_attention",
            "dims": {"B": 1, "d_head": 64, "kv_block_size": kv_block_size,
                     "n_attn_sinks": n_heads, "n_heads": n_heads,
                     "n_kv_heads": 1, "s_kv": 128, "s_query": s_query,
                     "s_window": 128},
            "phase": "decode",
            "dtype_env": {"activation": "bf16", "kv": "bf16",
                          "output": "bf16", "weight": "bf16"}})

    @pytest.mark.parametrize("kernel_impl, s_query, n_heads, page", [
        ("flashinfer_paged_decode", 1, 8, 16),
        ("flashinfer_paged_decode", 1, 1, 32),
        ("flashinfer_paged_verify", 4, 8, 16)])
    def test_a_sink_cell_is_declined(self, kernel_impl, s_query, n_heads,
                                     page):
        assert any(token.startswith("variant_dims:n_attn_sinks")
                   for token in _named(kernel_impl).serves.refusals(
                       self._sink_cell(s_query, n_heads, page)))


class TestFlashAttention4TakesHalfAndFp8Only:
    """FA4 asserts its inputs are fp16, bf16, e4m3 or e5m2 (flash-attn-4
    4.0.0b36, flash_attn/cute/interface.py:801-802). Undeclared, the
    Gemma-2 prefill cells — fp32 activation and cache — were handed to the
    GPU, and the path's canary failed on that assertion and disabled the
    path on every H100 pass (gemma-2-27b, gemma-2-9b, 2026-10-26)."""

    @staticmethod
    def _gemma_canary(dtype):
        return Invocation.from_job({
            "signature": "gqa_attention", "op": "flash_attention",
            "dims": {"B": 1, "d_head": 128, "kv_block_size": 16,
                     "n_heads": 1, "n_kv_heads": 1, "s_kv": 1024,
                     "s_query": 1024, "s_window": 4096},
            "phase": "prefill",
            "dtype_env": {"activation": dtype, "kv": dtype,
                          "output": dtype}})

    def test_an_fp32_cell_is_declined_by_the_declaration(self):
        said = list(_named("flash_attention_v4").serves.refusals(
            self._gemma_canary("fp32")))
        assert any(token.startswith("activation:fp32") for token in said)
        assert any(token.startswith("kv:fp32") for token in said)

    def test_its_bf16_twin_is_still_served(self):
        assert _named("flash_attention_v4").serves.accepts(
            self._gemma_canary("bf16"))


class TestTheUnservedAuditReadsAParameterisedIdentity:
    """`tools.unserved_rows` asks the registry about a row by its
    `kernel_impl` column, which holds the `impl_id` — `kernel_impl@params`
    for an implementation that folds deployment parameters in. Matched
    whole, `flashinfer_paged_decode@_b16_tc` named no provider, and every
    refused row of a paged backend was kept as "no longer registered"."""

    def test_a_paged_sink_row_is_refused_not_unregistered(self):
        import json
        from kernel_bench.providers import all_providers
        from tools.unserved_rows import _verdict
        dims = {"B": 1, "d_head": 64, "kv_block_size": 16, "n_attn_sinks": 8,
                "n_heads": 8, "n_kv_heads": 1, "s_kv": 128, "s_query": 1,
                "s_window": 128}
        row = {"framework": "flashinfer",
               "kernel_impl": "flashinfer_paged_decode@_b16_tc",
               "op": "flash_attention", "phase": "decode",
               "shape_json": json.dumps(dims),
               "dtype_env_json": json.dumps({"activation": "bf16",
                                             "kv": "bf16", "output": "bf16"})}
        verdict, why = _verdict(row, list(all_providers()))
        assert verdict == "refused", (verdict, why)
        assert "variant_dims:n_attn_sinks" in why


class TestThePurgeSelectsTheRowsAChangeRevokes:
    """`tools.purge_impl_rows` revokes the rows of an implementation whose
    own bench code changed. Which rows is named on the command line — the
    implementation, and for a change that only alters the launch past a
    threshold, the cells — and a FAILURE goes with the measurements: it
    failed on the old build and counts as tried, so leaving it blocks the
    re-measurement the fix was for."""

    def _row(self, impl, outcome="measured", **dims):
        import json
        shape = {"B": 1, "s_query": 64, "s_kv": 4096, "n_heads": 64,
                 "d_head": 576, "n_topk_keys": 2048}
        shape.update(dims)
        return {"framework": "torch", "kernel_impl": impl, "outcome": outcome,
                "framework_version": "2.8.0+cu128",
                "shape_json": json.dumps(shape)}

    def test_a_failure_of_the_old_build_goes_with_the_measurements(self):
        from tools.purge_impl_rows import _selected
        for outcome in ("measured", "oom", "mismatch", "error"):
            assert _selected(self._row("torch_qsa_sparse_gated", outcome),
                             ["torch_qsa_sparse_gated"], [], None, None)

    def test_only_the_named_implementations_go(self):
        from tools.purge_impl_rows import _selected
        assert not _selected(self._row("flashmla_sparse"),
                             ["torch_gather_sparse_mla"], [], None, None)
        # The column holds the `impl_id`: `kernel_impl@params` for an
        # implementation that folds deployment parameters in.
        assert _selected(self._row("torch_gather_sparse_mla@_blk64"),
                         ["torch_gather_sparse_mla"], [], None, None)

    def test_a_threshold_change_takes_only_the_cells_past_it(self):
        from tools.purge_impl_rows import _selected
        past = "B * n_topk_keys * (4 * d_head + 8 + 8 * n_heads) * s_query > 268435456"
        assert _selected(self._row("torch_gather_sparse_mla",
                                   s_query=131072),
                         ["torch_gather_sparse_mla"], [], None, past)
        assert not _selected(self._row("torch_gather_sparse_mla", s_query=1),
                             ["torch_gather_sparse_mla"], [], None, past)
        # A cell binding no such axis is not the cell the threshold names.
        assert not _selected(
            {"kernel_impl": "torch_gather_sparse_mla", "outcome": "measured",
             "shape_json": '{"B": 1, "s_query": 8}'},
            ["torch_gather_sparse_mla"], [], None, past)


class TestThePurgeRecordKeepsItsColumns:
    """The record is the audit trail: a year out it is the only thing that
    says a gap in the corpus was a cleanup and not a defect. So a row's
    columns may never shift under it.

    They did. `_record` built a `DictWriter` from each source file's own
    header, so rows from a file with a different schema were written in
    THAT file's order under the record's first header — 82 rows of a
    46-column record came out 36 or 37 wide, their `outcome` read as a
    kernel symbol and their `purge_reason` gone (2026-10-29)."""

    def test_a_second_schema_widens_the_record_instead_of_shifting_it(
            self, tmp_path):
        import csv
        from tools.unserved_rows import append_record
        where = str(tmp_path / "bench_purge_test.csv")
        append_record(where, [{"op": "gemm", "kernel_impl": "a",
                               "outcome": "measured", "source_csv": "x",
                               "row_id": "2", "purge_reason": "why"}])
        # A tree whose CSVs carry a column the first did not, and lack one
        # it had: the old writer put `gpu` under `outcome`.
        append_record(where, [{"op": "mla", "kernel_impl": "b", "gpu": "H100",
                               "source_csv": "y", "row_id": "3",
                               "purge_reason": "why"}])
        with open(where, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [r["kernel_impl"] for r in rows] == ["a", "b"]
        assert [r["purge_reason"] for r in rows] == ["why", "why"]
        # Read by name, every row still says what happened to it and where
        # it came from; the column a row did not have is empty, not shifted.
        assert rows[0]["outcome"] == "measured" and rows[1]["outcome"] == ""
        assert rows[0]["gpu"] == "" and rows[1]["gpu"] == "H100"
        assert all(len(r) == len(rows[0]) for r in rows)


class TestEveryResultsTreeIsNamed:
    """The revocation tools find a box's results through one list of trees.
    It used to be a glob, `.cache/h100_results/*/results`, which reads as
    "every box's mirror" and skipped the tree from before the per-box
    subdirectory — one level higher. A purge over the defaults then deleted
    the lake's copy of a row and left the box's (846 indexer rows, found
    2026-10-29), and nothing said a tree had been missed."""

    #: The trees as of 2026-10-29. Here so that dropping one from
    #: `DEFAULT_TREES` turns this red instead of going unnoticed: a purge
    #: cannot report a complete pass over a set that quietly shrank.
    KNOWN = (
        ".cache/h100_results/results",
        ".cache/h100_results/h100-a/results",
        ".cache/h100_results/h100-b/results",
        ".cache/h100_results_fresh/results",
        ".cache/bench4090/results",
        ".cache/prov_rows_pending_upload",
    )

    def test_no_known_tree_is_dropped(self):
        from tools.unserved_rows import DEFAULT_TREES
        assert set(self.KNOWN) <= set(DEFAULT_TREES), (
            set(self.KNOWN) - set(DEFAULT_TREES))
        assert not [t for t in DEFAULT_TREES if "*" in t], (
            "a glob hides a tree it does not match; name them")

    def test_some_here_and_some_missing_is_a_refusal(self, tmp_path,
                                                     monkeypatch):
        from tools.unserved_rows import default_trees
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a").mkdir()
        # None here: another machine, and the caller's own "no result CSVs
        # found" is the right answer.
        assert default_trees(("a/results", "b/results")) == ()
        (tmp_path / "a" / "results").mkdir()
        # One here, one not: a tree moved or a box was renamed, and a run
        # over the rest would report a complete purge of an incomplete set.
        with pytest.raises(SystemExit, match="b/results"):
            default_trees(("a/results", "b/results"))
        (tmp_path / "b" / "results").mkdir(parents=True)
        assert default_trees(("a/results", "b/results")) == (
            "a/results", "b/results")

    def test_every_record_writer_is_the_one_writer(self):
        """One writer, so a column cannot shift in any channel. Three tools
        write records: the unserved audit's file AND lake paths, and the
        duplicate-row dedup. Only the first had been routed — the lake
        path, used the same day, still took its columns from one run and
        dropped the rest with `extrasaction="ignore"`.

        Asked of the code: the only functions that write CSV rows in these
        modules are `append_record` and the ones that rewrite a RESULTS
        file in place (recognisable by the atomic `os.replace` they move it
        into place with); and every default record path is under
        `RECORDS_DIR`.
        """
        import ast
        import inspect
        from tools import dedup_results_csv, purge_impl_rows, unserved_rows
        writers = paths = 0
        for module in (unserved_rows, purge_impl_rows, dedup_results_csv):
            src = inspect.getsource(module)
            for node in ast.walk(ast.parse(src)):
                if not isinstance(node, ast.FunctionDef):
                    continue
                body = ast.get_source_segment(src, node) or ""
                if "bench_purge_{time.strftime" in body:
                    paths += 1
                    assert "RECORDS_DIR" in body, (
                        f"{module.__name__}.{node.name} defaults a record "
                        f"inside a results tree")
                if "csv.DictWriter" in body or "csv.writer" in body:
                    writers += 1
                    assert (node.name == "append_record"
                            or "os.replace(" in body), (
                        f"{module.__name__}.{node.name} writes CSV rows "
                        f"itself; a record goes through append_record, "
                        f"which aligns by name")
        assert paths >= 3 and writers >= 3, (paths, writers)

    def test_the_readers_share_one_record_predicate(self):
        """A record is recognised in one place. `mbu_ceiling_audit` walks
        tree PARENTS recursively, so it cannot use the two-depth
        `result_files`; it shares the predicate instead of spelling the
        name again."""
        from tools import mbu_ceiling_audit, unserved_rows
        assert mbu_ceiling_audit.is_record is unserved_rows.is_record

    def test_a_tree_here_that_nobody_listed_is_a_refusal(self, tmp_path,
                                                         monkeypatch):
        """The direction that bit: a tree that EXISTS and is not in the
        list. Reading the list alone, a purge walks straight past a new
        box's mirror and reports a complete pass — 120,497 rows were left
        behind in one unlisted tree on 2026-10-29, 113,581 of them
        measured. So the trees are also found by shape."""
        from tools.unserved_rows import default_trees, results_trees
        monkeypatch.chdir(tmp_path)
        listed = tmp_path / "cache" / "listed" / "results"
        (listed / "a_model").mkdir(parents=True)
        (listed / "a_model" / "torch.csv").write_text("op,kernel_impl\n")
        assert default_trees(("cache/listed/results",), root="cache") == (
            "cache/listed/results",)
        # A mirror of a box nobody added to the list.
        fresh = tmp_path / "cache" / "newbox" / "results"
        (fresh / "a_model").mkdir(parents=True)
        (fresh / "a_model" / "torch.csv").write_text("op,kernel_impl\n")
        assert "cache/newbox/results" in results_trees("cache")
        with pytest.raises(SystemExit, match="cache/newbox/results"):
            default_trees(("cache/listed/results",), root="cache")
        # Found by its SHAPE, not by its name: a directory of CSVs that are
        # not measurements is not a results tree.
        other = tmp_path / "cache" / "notes" / "model"
        other.mkdir(parents=True)
        (other / "sweep.csv").write_text("model,notes\n")
        assert "cache/notes" not in results_trees("cache")

    def test_a_purge_record_is_not_a_results_file(self, tmp_path,
                                                  monkeypatch):
        """Two guards. A record is never read as results — matched anywhere
        in the name, because records get copied and relabelled
        (`box_bench_purge_…`, `<tree>__bench_purge_…`, `….dryrun.csv`). And
        a tree still holding one STOPS the run: teaching each reader to skip
        them only protects the readers we own, and on 2026-10-29 a purge
        from another checkout, at a commit without the skip, moved 21 rows
        out of a record it had read as results."""
        from tools import unserved_rows as ur
        tree = tmp_path / "results"
        (tree / "model").mkdir(parents=True)
        (tree / "model" / "torch.csv").write_text("op,kernel_impl\n")
        assert ur.result_files((str(tree),)) == [
            str(tree / "model" / "torch.csv")]
        for name in ("bench_purge_2026-10-29_x.csv",
                     "box_bench_purge_2026-10-29_x.csv",
                     "bench_purge_2026-10-29_x.dryrun.csv"):
            record = tree / name
            record.write_text("op,kernel_impl,purge_reason\n")
            assert ur.is_record(str(record))
            with pytest.raises(SystemExit, match="records are inside"):
                ur.result_files((str(tree),))
            # The message says where records belong, so the fix is obvious.
            with pytest.raises(SystemExit, match=ur.RECORDS_DIR):
                ur.result_files((str(tree),))
            record.unlink()
        # Where they belong, they are neither results nor a tree.
        monkeypatch.chdir(tmp_path)
        records = tmp_path / ur.RECORDS_DIR
        records.mkdir(parents=True)
        (records / "bench_purge_2026-10-29_x.csv").write_text(
            "op,kernel_impl\n")
        assert ur.RECORDS_DIR not in ur.results_trees(ur.CACHE_ROOT)


class TestCeilingsTheH100Bisected:
    """Cells the H100 queue's canaries failed for a reason the kernel owns
    at every cell past a boundary, so the declaration states the boundary
    (bisected on the box, 2026-10-28) instead of a canary disabling the
    whole launch path each pass."""

    def test_sglangs_prefill_conv_stops_at_its_grid(self):
        # Grid y = ceil(s_query / 8), and CUDA stops a grid's y at 65,535.
        serves = _named("sglang_causal_conv1d_fn").serves

        def cell(s_query):
            return _cell(signature="causal_conv1d", op="causal_conv1d",
                         dims={"B": 1, "d_conv_channels": 512, "dilation": 1,
                               "s_conv_kernel": 4, "s_query": s_query},
                         dtype_env={"activation": "bf16", "weight": "bf16",
                                    "kv": "bf16", "output": "bf16"})
        assert not serves.accepts(cell(1048576))        # GLM-5.3-Flash
        assert not serves.accepts(cell(8 * 65535 + 1))
        assert serves.accepts(cell(8 * 65535))

    def test_sglangs_gelu_and_mul_takes_whole_vectors_only(self):
        serves = _named("sgl_gelu_and_mul").serves

        def cell(d_ffn):
            return _cell(signature="gelu_and_mul", op="gelu_and_mul",
                         phase="decode",
                         dims={"B": 1, "s_query": 2, "d_ffn": d_ffn},
                         dtype_env={"activation": "bf16", "output": "bf16"})
        assert not serves.accepts(cell(30))     # gemma-4-12B's shard
        assert not serves.accepts(cell(36))     # passes the wrapper, faults
        assert serves.accepts(cell(32))

    @staticmethod
    def _group_cast(B, s_query, scale):
        return _cell(signature="quantise_activation",
                     op="quantise_activation", phase="decode",
                     dims={"B": B, "s_query": s_query, "d_model": 4096},
                     dtype_env={"activation": "bf16", "output": "fp8_e4m3",
                                "metadata": scale,
                                "activation_granularity": "per-group:32"})

    def test_sglangs_pow2_group_quant_offsets_in_int32(self):
        serves = _named("sglang_per_token_group_quant_fp8_ue8m0").serves
        assert not serves.accepts(self._group_cast(13173, 40, "ue8m0"))
        assert not serves.accepts(self._group_cast(524289, 1, "ue8m0"))
        assert serves.accepts(self._group_cast(524288, 1, "ue8m0"))

    def test_the_exact_scale_sibling_is_another_kernel(self):
        # Its JIT kernel ran 524,289 x 4096 cleanly on the same box.
        serves = _named("sglang_per_token_group_quant_fp8").serves
        assert serves.accepts(self._group_cast(13173, 40, "fp32"))


class TestACellThatCarriesAStateNeedsAKernelThatTakesIt:
    """A recurrent state's carrier bills the state's read and write-back
    (`traffic._settle_cache_reads` / `_settle_state_writes`). DeepSeek's
    compressor carries its fp32 `kv_state` on a `mul`, and `torch_mul` timed a
    plain bf16 multiply against those bytes: 107-108% of the H100's
    bandwidth on DeepSeek-V4.1-Flash (2026-10-28). Only a kernel that is
    handed the state and writes it back serves such a cell
    (`KernelSignature.takes_state`, per kernel: under one signature a
    framework's scan takes the state while a floor may start from zeros);
    the rest refuse it, and the plan counts it as a gap."""

    @staticmethod
    def _mul(carries_state):
        job = {"signature": "mul", "op": "mul", "phase": "decode",
               "dims": {"B": 1215, "d_model": 512, "s_query": 6},
               "dtype_env": {"activation": "bf16", "kv": "fp32",
                             "output": "bf16"}}
        if carries_state:
            job["carries_state"] = True
        return Invocation.from_job(job)

    def test_a_plain_multiply_refuses_a_state_carrying_cell(self):
        assert "carries_state" in list(
            _named("torch_mul").serves.refusals(self._mul(True)))

    def test_the_same_multiply_without_a_state_is_served(self):
        assert _named("torch_mul").serves.accepts(self._mul(False))

    @pytest.mark.parametrize("kernel_impl", [
        "torch_depthwise_conv1d", "torch_mamba2_scan",
        "torch_fused_save_compress_norm", "vllm_kpool_compress_and_write_cache"])
    def test_a_kernel_that_keeps_no_state_refuses_the_cell(self, kernel_impl):
        # A conv floor that pads a fresh window, a scan floor that starts
        # from zeros, a compressor floor that pools the cell's own rows, a
        # pool kernel handed candidate windows: none moves the state the
        # cell bills, whatever its signature.
        p = _named(kernel_impl)
        inv = Invocation.from_job({
            "signature": p.signature, "op": p.signature, "phase": "prefill",
            "dims": {}, "dtype_env": {"activation": "bf16"},
            "carries_state": True})
        assert "carries_state" in list(p.serves.refusals(inv))

    @pytest.mark.parametrize("kernel_impl", [
        "vllm_fused_save_compress_norm", "vllm_chunk_gated_delta_rule",
        "vllm_causal_conv1d_fn", "mamba_chunk_scan_combined",
        "deepinfer_compress_pool_prefill"])
    def test_a_kernel_handed_the_state_serves_it(self, kernel_impl):
        assert _named(kernel_impl).serves.takes_state


@pytest.mark.parametrize("kernel_impl, dims", [
    ("torch_fused_recurrent_gated_delta_rule",
     {"B": 2, "n_lin_k_heads": 2, "n_lin_v_heads": 2, "d_lin_k_head": 16,
      "d_lin_v_head": 16, "s_query": 1}),
    ("torch_fused_recurrent_kda",
     {"B": 2, "n_lin_k_heads": 2, "n_lin_v_heads": 2, "d_lin_k_head": 16,
      "d_lin_v_head": 16, "s_query": 1}),
])
def test_a_state_taking_floor_writes_its_state_back(kernel_impl, dims):
    # A floor that declares `takes_state` writes the step's state back, as
    # the serving launch writes its slot — into its own copy, never the
    # shared input another provider of the cell starts from.
    import torch
    from kernel_bench.env import Env
    from kernel_bench.launch_trace import record
    from kernel_bench.refs import get_ref, input_generator
    p = _named(kernel_impl)
    inv = Invocation.from_job({
        "signature": p.signature, "op": p.signature, "phase": "decode",
        "dims": dims, "dtype_env": {"activation": "bf16"},
        "carries_state": True,
        "physical_key": [p.signature, str(sorted(dims.items())), "bf16",
                         "decode"]})
    inputs = get_ref(p.signature).make_inputs(
        inv, "cpu", input_generator(inv, "cpu"))
    before = inputs["initial_state"].clone()
    case = p.build(inv, Env("2.11", None, None, None, None,
                            libs={"torch": "2.11"}),
                   p.convert_inputs(inv, inputs))
    with record() as recorded:
        case.run()
    shape = tuple(before.shape)
    assert [c for c in recorded.trace.calls if c.op.startswith("aten.copy_")
            and c.operands and c.operands[0].shape == shape]
    assert torch.equal(inputs["initial_state"], before)


def test_flashkda_reads_and_writes_the_state_at_its_cached_width(monkeypatch):
    # The KDA cell caches and bills its recurrent state fp32; FlashKDA takes
    # either width, and handed bf16 it moved half the state's bytes.
    import types
    import torch
    from kernel_bench.env import Env
    from kernel_bench.providers import linear_ssm
    from kernel_bench.refs import get_ref, input_generator
    seen = {}

    def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
            initial_state=None, final_state=None, cu_seqlens=None):
        seen.update(initial=initial_state.dtype, final=final_state.dtype)

    monkeypatch.setitem(sys.modules, "flash_kda",
                        types.SimpleNamespace(fwd=fwd))
    monkeypatch.setattr(linear_ssm, "_require_cuda", lambda: None)
    dims = {"B": 1, "n_lin_k_heads": 2, "n_lin_v_heads": 2,
            "d_lin_k_head": 128, "d_lin_v_head": 128, "s_query": 4}
    p = _named("flash_kda_cutlass_fwd")
    inv = Invocation.from_job({
        "signature": "kda", "op": "kda", "phase": "prefill", "dims": dims,
        "dtype_env": {"activation": "bf16"}, "carries_state": True,
        "physical_key": ["kda", str(sorted(dims.items())), "bf16", "prefill"]})
    inputs = get_ref("kda").make_inputs(inv, "cpu", input_generator(inv, "cpu"))
    case = p.build(inv, Env("2.11", None, None, None, None,
                            libs={"torch": "2.11"}),
                   p.convert_inputs(inv, inputs))
    case.run()
    assert seen == {"initial": torch.float32, "final": torch.float32}
