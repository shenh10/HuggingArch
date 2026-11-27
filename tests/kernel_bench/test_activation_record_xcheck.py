"""The cross-check basis for a launch that writes a quantised activation."""

import torch

from kernel_bench.driver import (_xcheck_band, _xcheck_basis,
                                 compared_records)
from kernel_bench.env import Env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import all_providers
from kernel_bench.quant import ActivationCast
from kernel_bench.refs import get_ref, input_generator


def _cell(signature, dims, dtype_env, *, shape_out=None):
    return Invocation(
        signature=signature, op=signature, dims=dims,
        dtype_env=dtype_env, phase="decode", tp=1, shape_out=shape_out,
        physical_key=(signature, "{}", "{}", "decode"))


def _inputs(inv):
    spec = get_ref(inv.signature)
    return spec.make_inputs(inv, "cpu", input_generator(inv, "cpu"))


def _provider(name):
    return next(p for p in all_providers() if p.kernel_impl == name)


def _env():
    return Env("2.13", "cuda", "NVIDIA GeForce RTX 4090", "RTX-4090",
               1008.0, sm="sm_89", libs={"vllm": "0.32.0"})


def _cast():
    return ActivationCast(narrow="fp8_e4m3", wide="bf16", group=128,
                          scale_encoding="pow2", narrows=True)


def _inv():
    # The activation slot is deliberately the narrowest dtype. A cast on the
    # output is a record both sides write, so it must not be charged to the
    # band as though the launch read an FP8 operand.
    return Invocation(
        signature="silu_and_mul", op="silu_and_mul",
        dims={"B": 2, "s_query": 3, "d_ffn": 128},
        dtype_env={"activation": "fp8_e4m3", "weight": "bf16",
                   "kv": "bf16", "output": "fp8_e4m3",
                   "activation_granularity": "per-group:128",
                   "metadata": "ue8m0"},
        phase="decode", tp=1)


def test_an_activation_cast_is_compared_on_the_activation_slot():
    cast = _cast()
    assert compared_records({"cast": cast}) == {"activation": cast}
    assert _xcheck_basis(_inv(), {"cast": cast}) == ("bf16", (cast,))


def test_the_worst_band_reads_the_activation_grid_once():
    cast = _cast()
    plain = _xcheck_band("bf16")
    banded = _xcheck_band("bf16", (cast,))
    assert banded.typical == plain.typical
    assert banded.worst == cast.format.step == 0.125


def test_cached_record_comparisons_are_unchanged():
    records = compared_records({"output_format": "fp8_group128"})
    assert records == {"output": "fp8_group128"}
    band = _xcheck_band("bf16", tuple(records.values()))
    assert band.typical == _xcheck_band("bf16").typical
    assert band.worst == 0.125


def test_plain_norm_and_activation_outputs_stay_plain():
    for signature, dims in (
            ("rmsnorm", {"B": 2, "s_query": 3, "d_model": 128}),
            ("fused_add_rmsnorm", {"B": 2, "s_query": 3, "d_model": 128}),
            ("silu_and_mul", {"B": 2, "s_query": 3, "d_ffn": 128})):
        inv = _cell(signature, dims,
                    {"activation": "bf16", "weight": "bf16",
                     "output": "bf16"})
        inputs = _inputs(inv)
        assert inputs["cast"] is None
        assert compared_records(inputs) == {}


def test_quantised_output_references_read_one_cast():
    cells = (
        ("rmsnorm", {"B": 2, "s_query": 3, "d_model": 128}, None),
        ("fused_add_rmsnorm", {"B": 2, "s_query": 3, "d_model": 128}, None),
        ("silu_and_mul", {"B": 2, "s_query": 3, "d_ffn": 128}, None),
        ("fused_inv_rope",
         {"B": 1, "n_heads": 4, "s_query": 3, "d_head": 64, "d_rope": 16},
         [1, 2, 3, 128]),
    )
    for signature, dims, shape_out in cells:
        inv = _cell(signature, dims,
                    {"activation": "bf16", "weight": "bf16",
                     "output": "fp8_e4m3",
                     "activation_granularity": "per-group:32",
                     "metadata": "ue8m0"}, shape_out=shape_out)
        inputs = _inputs(inv)
        out = get_ref(signature).reference(inputs)
        assert isinstance(inputs["cast"], ActivationCast)
        assert torch.isfinite(out).all()
        assert _xcheck_basis(inv, inputs)[1] == (inputs["cast"],)


def test_real_vllm_providers_claim_only_their_output_grids():
    checks = (
        ("vllm_rms_norm_per_block_quant", "rmsnorm",
         {"B": 2, "s_query": 3, "d_model": 128}, "fp32"),
        ("vllm_fused_add_rms_norm_per_block_quant", "fused_add_rmsnorm",
         {"B": 2, "s_query": 3, "d_model": 128}, "fp32"),
        ("vllm_silu_and_mul_per_block_quant", "silu_and_mul",
         {"B": 2, "s_query": 3, "d_ffn": 128}, "fp32"),
        ("vllm_fused_inv_rope_quant", "fused_inv_rope",
        {"B": 1, "n_heads": 4, "s_query": 3, "d_head": 128,
         "d_rope": 64}, "ue8m0"),
    )
    env = _env()
    for name, signature, dims, metadata in checks:
        provider = _provider(name)
        quant = _cell(signature, dims,
                      {"activation": "bf16", "weight": "bf16",
                       "output": "fp8_e4m3",
                       "activation_granularity": "per-group:128",
                       "metadata": metadata})
        plain = _cell(signature, dims,
                      {"activation": "bf16", "weight": "bf16",
                       "output": "bf16"})
        assert provider.accepts(quant, env), name
        assert not provider.accepts(plain, env), name
