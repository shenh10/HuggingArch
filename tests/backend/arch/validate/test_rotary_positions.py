"""Multimodal RoPE: the rotation that reads three position streams.

Qwen3-VL, Qwen3.5 and Qwen3.8 rotate their text tokens by temporal, height
and width positions (`mrope_section` in the rope parameters). The arithmetic
per rotated element is a rotation's either way, so FLOPs and bytes cannot see
the difference; the kernel can. vLLM builds `MRotaryEmbedding` from
`mrope_section` (layers/rotary_embedding/__init__.py:102-111) and runs q and k
through `triton_mrope` in one launch (mrope.py:374-381); sglang's Qwen3-MoE
attention, which Qwen3-VL-MoE uses (qwen3_vl_moe.py:32), keeps it out of the
fused qk-norm+rope kernel (qwen3_moe.py:515-530).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.fusion import Framework
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.ops import OP_REGISTRY
from backend.arch.validate._rotary_positions import (
    config_position_axes, config_rotation, run_rotary_position_check)
from backend.inference.framework_bindings import (
    FrameworkNotOnDevice, ServingDeployment, get_binding)
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import shared_loader

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
MROPE_MODELS = ("Qwen__Qwen3-VL-235B-A22B-Instruct", "Qwen__Qwen3.5-35B-A3B",
                "Qwen__Qwen3.8-27B")


def _config(stem: str):
    return json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]


def _resolved(stem: str):
    return shared_loader().match(_config(stem), model_id=stem.replace("__", "/", 1))


def _fused_rotations(program: Program, framework: str):
    """The rotation kernels on a BF16 H100 with every server flag the
    binding reads turned on: the widest the framework fuses, in every stack
    of the checkpoint (the language model's and each encoder's)."""
    binding = get_binding(framework)
    binding = binding.deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="BF16",
        server_flags={name: True for name in binding.server_flag_names()}))
    analysis = Deployment(program).at(input_tokens=128).checkpoint()
    fused, _ = Framework.of(binding, FUSION_SCHEMES).rewrite(
        analysis.program, analysis)
    return {n.op for n in fused.walked_nodes()
            if not n.children and "rope" in n.op}


def test_mrope_does_the_work_of_a_rotation():
    """Section layout chooses which cos row a channel reads, nothing more:
    `mrope` and `rope_emb` agree on FLOPs, parameters and shape, 3-D or 4-D."""
    mrope, rope_emb = OP_REGISTRY["mrope"], OP_REGISTRY["rope_emb"]
    dims = {"B": 2, "s_ctx": 16, "dim": 64, "n_heads": 4}
    assert mrope.flops_fn(dims) == rope_emb.flops_fn(dims)
    assert mrope.params_fn(dims) == rope_emb.params_fn(dims) == 0
    assert (mrope.shape_in, mrope.alt_shape_in, mrope.shape_out) == (
        rope_emb.shape_in, rope_emb.alt_shape_in, rope_emb.shape_out)
    assert mrope.rotary_position_axes == 3
    assert OP_REGISTRY["rope"].rotary_position_axes == 1
    assert rope_emb.rotary_position_axes == 1


@pytest.mark.parametrize("stem", MROPE_MODELS)
def test_a_multimodal_rope_model_rotates_its_text_with_mrope(stem):
    """Every text-path rotation of a model whose rope parameters carry
    `mrope_section` is `mrope`; its vision tower's 2-D rotation is not text
    and is `rope_2d`. The spec passes the config check."""
    resolved = _resolved(stem)
    program = Program.of(resolved)
    text = {n.op for n in program.nodes()
            if not n.children and n.uid not in program.encoder_uids
            and OP_REGISTRY.get(n.op) is not None
            and OP_REGISTRY[n.op].rotary_position_axes}
    assert text == {"mrope"}
    assert config_position_axes(resolved.language_config()[1]) == 3
    assert run_rotary_position_check(resolved, _config(stem)) == []


def test_sglang_does_not_fuse_qk_norm_with_multimodal_rope():
    """Qwen3-VL-MoE in sglang: qk-norm runs on its own and the two mrope
    rotations are one `MRotaryEmbedding` launch — `fused_qk_mrope`, not
    `fused_qk_norm_rope`, even with the server flag on. Qwen3-30B-A3B, the
    one-position `Qwen3MoeForCausalLM`, gets the fused kernel; dense
    Qwen3-8B has no fused path in sglang (`Qwen3Attention`) and keeps the
    rope kernel."""
    vl = Program.of(_resolved("Qwen__Qwen3-VL-235B-A22B-Instruct"))
    # the text tower's pair, and the vision tower's two-axis rotations,
    # which no q/k plan fuses (vLLM runs them through `ApplyRotaryEmb`)
    assert _fused_rotations(vl, "sglang") == {"fused_qk_mrope", "rope_2d"}
    assert _fused_rotations(vl, "vllm") == {"fused_qk_mrope", "rope_2d"}
    moe = Program.of(_resolved("Qwen__Qwen3-30B-A3B"))
    assert _fused_rotations(moe, "sglang") == {"fused_qk_norm_rope"}
    assert _fused_rotations(moe, "vllm") == {"fused_qk_rope"}
    dense = Program.of(_resolved("Qwen__Qwen3-8B"))
    assert _fused_rotations(dense, "sglang") == {"fused_qk_rope"}
    assert _fused_rotations(dense, "vllm") == {"fused_qk_rope"}


def test_output_gated_qwen_attention_runs_one_prologue_kernel():
    """Qwen3.5 / Qwen3.8 / Qwen4-Exp: the q|gate split, both per-head norms,
    the partial rotation (one position or mrope) and the gate copy are one
    launch in both frameworks on CUDA (vLLM `fused_qk_rmsnorm_rope_gate`,
    sglang `fused_qk_gemma_rmsnorm_rope_gate`), so no `fused_qk_mrope` and
    no `fused_qk_rope` remains on their attention. The QSA layers of
    Qwen3.8-Flash-Next spell the norm's gain as its own multiply and match
    the second plan. Their indexer rotates its query every step and its
    pooled key once per block, when the block is cached — two rotations at
    different times, not a q/k pair — and neither is a rotary launch of its
    own on CUDA: the query's norm + rope is `qsa_index_q_norm_rope` and, in
    prefill, the key's pool + norm + rope + store is
    `qsa_index_k_compress_store` (vLLM `qsa_prepare`, sglang
    `qsa_index_q_norm_rope_store` / `qsa_index_k_compress_store`).
    Qwen3-Next is vLLM's only: sglang keeps its separate norm and rotary."""
    one_launch = {"fused_qk_rmsnorm_rope_gate"}
    for stem, rest in (("Qwen__Qwen3.5-35B-A3B", {"rope_2d"}),
                       ("Qwen__Qwen3.8-27B", {"rope_2d"}),
                       ("Qwen__Qwen3.8-2.4T-A95B", set())):
        program = Program.of(_resolved(stem))
        for framework in ("vllm", "sglang"):
            assert _fused_rotations(program, framework) == (
                one_launch | rest), (stem, framework)
    flash = Program.of(_resolved("Qwen__Qwen3.8-Flash-Next"))
    for framework in ("vllm", "sglang"):
        assert _fused_rotations(flash, framework) == {
            "fused_qk_rmsnorm_rope_gate_split_gain", "qsa_index_q_norm_rope",
            "rope_2d"}
    coder_next = Program.of(_resolved("Qwen__Qwen3-Coder-Next"))
    assert _fused_rotations(coder_next, "vllm") == one_launch
    # sglang rotates the partial rope lanes in place, so the slices and the
    # re-join around the rotation are that launch's too (287fa757e).
    assert _fused_rotations(coder_next, "sglang") == {"fused_qk_rope_partial"}


def test_the_gated_prologue_kernel_is_cuda_only():
    """The prologue kernel is CUDA's, and two layers now say so.

    The entry still states the device its dispatch asks, so a `when:` that
    lost it goes red here. Above that, vLLM declares the devices it runs on
    at all (`FrameworkBinding.devices`), so a deployment off them is
    refused before any kernel is matched — which is why this no longer
    walks a rocm deployment to see which rotation it takes."""
    entries = get_binding("vllm").implementations_for(
        "fused_qk_rmsnorm_rope_gate")
    assert entries
    for impl in entries:
        assert all(when.device == ("cuda",) for when in impl.whens), impl
    with pytest.raises(FrameworkNotOnDevice):
        get_binding("vllm").deployed(
            ServingDeployment.of(activation_dtype="BF16", device="rocm"))


def test_minimax_m3_norms_and_rotates_q_and_k_in_one_launch():
    """MiniMax-M3: per-head QK norm + partial rotation (64 of 128 dims) is one
    launch in both frameworks — vLLM `fused_minimax_m3_qknorm_rope_kv_insert`,
    sglang `minimax_qknorm_rope`, whose gate asks rotary_dim 64. GLM-4.5 has
    the same attention graph and keeps a separate norm and rotary kernel. No
    snapshot: the spec is resolved from the cached config."""
    from backend.analyzer import fetch_config
    from backend.arch.loader import SpecLoader
    loader = SpecLoader()
    m3 = Program.of(loader.resolve(
        loader.get("minimax_m3"), fetch_config("MiniMaxAI/MiniMax-M3"),
        model_id="MiniMaxAI/MiniMax-M3"))
    glm = Program.of(_resolved("zai-org__GLM-4.5"))
    for framework in ("vllm", "sglang"):
        # the vision tower's rotation stays the plain rope kernel
        assert _fused_rotations(m3, framework) == {
            "fused_qk_norm_partial_rope", "fused_qk_rope"}
        assert "fused_qk_norm_partial_rope" not in _fused_rotations(
            glm, framework)


def test_the_rotation_is_the_same_cost():
    """Moving Qwen3-VL to `mrope` changes no FLOP and no parameter."""
    program = Program.of(_resolved("Qwen__Qwen3-VL-235B-A22B-Instruct"))
    analysis = Deployment(program).at(batch=1, input_tokens=64).analyze()
    flops = {n.uid: analysis.cost.values[n.uid].flops for n in program.nodes()
             if n.op == "mrope" and not n.children}
    dims = analysis.binding
    per_rotation = 3 * 64 * int(dims["d_head"])
    q_heads, kv_heads = int(dims["n_heads"]), int(dims["n_kv_heads"])
    assert sorted(set(int(v) for v in flops.values())) == sorted(
        {per_rotation * q_heads, per_rotation * kv_heads})


def test_config_rotation_follows_the_framework_dispatch():
    """`mrope_section` is three streams under `MRotaryEmbedding`; rope type
    xdrope (HunyuanOCR, four sections) is `XDRotaryEmbedding`, keyed by
    `xdrope_section` whether the config spells its sections that way or HF
    has normalised them to `mrope_section`; a plain rope is one stream."""
    assert config_rotation(
        {"rope_scaling": {"mrope_section": [24, 20, 20]}}
    ) == ("mrope_section", 3)
    assert config_rotation(
        {"rope_parameters": {"rope_type": "xdrope",
                             "xdrope_section": [16, 16, 16, 16]}}
    ) == ("xdrope_section", 4)
    assert config_rotation(
        {"rope_parameters": {"type": "xdrope",
                             "mrope_section": [16, 16, 16, 16]}}
    ) == ("xdrope_section", 4)
    assert config_rotation({"rope_theta": 10000}) == ("", 1)
    hunyuan = _resolved("tencent__HunyuanOCR")
    assert config_position_axes(hunyuan.language_config()[1]) == 4


def test_the_rotation_is_read_from_the_language_models_config():
    """A multimodal config's rope parameters are its language model's: the
    check reads them where the spec binds `d_model`, so a rope mapping at
    the root of a config whose backbone is `text_config` is not the text
    rotation. Both HF spellings stating different rotations is refused."""
    stem = "Qwen__Qwen3-VL-235B-A22B-Instruct"
    config = json.loads(json.dumps(_config(stem)))
    config["rope_scaling"] = {"rope_type": "xdrope",
                              "xdrope_section": [16, 16, 16, 16]}
    assert run_rotary_position_check(_resolved(stem), config) == []
    with pytest.raises(ValueError, match="different rotations"):
        config_rotation({"rope_scaling": {"mrope_section": [24, 20, 20]},
                         "rope_parameters": {"rope_type": "default"}})


def test_hunyuan_ocr_rotates_its_text_with_xdrope():
    """HunyuanOCR's text rotation is `xdrope` and passes the config check;
    under vLLM it is not folded into the text path's q/k rotary kernel
    (`fused_qk_rope`): `XDRotaryEmbedding` runs `ApplyRotaryEmb` per
    tensor, so each node stays its own kernel."""
    stem = "tencent__HunyuanOCR"
    resolved = _resolved(stem)
    program = Program.of(resolved)
    text = {n.op for n in program.nodes()
            if not n.children and n.uid not in program.encoder_uids
            and OP_REGISTRY.get(n.op) is not None
            and OP_REGISTRY[n.op].rotary_position_axes}
    assert text == {"xdrope"}
    assert run_rotary_position_check(resolved, _config(stem)) == []
    assert _fused_rotations(program, "vllm") == {"xdrope"}


def test_xdrope_does_the_work_of_a_rotation():
    """Which position stream a channel section reads changes nothing per
    element: `xdrope` and `rope` agree on FLOPs, parameters and shape."""
    xdrope, rope = OP_REGISTRY["xdrope"], OP_REGISTRY["rope"]
    dims = {"B": 2, "s_ctx": 16, "dim": 128, "n_heads": 4}
    assert xdrope.flops_fn(dims) == rope.flops_fn(dims)
    assert xdrope.params_fn(dims) == rope.params_fn(dims) == 0
    assert (xdrope.shape_in, xdrope.shape_out, xdrope.couples_on) == (
        rope.shape_in, rope.shape_out, rope.couples_on)


def test_a_one_position_rotation_under_xdrope_is_refused_with_its_fix():
    """An xdrope config with a spec that rotates by one position: every text
    rotation is named, and the message says to write `op: xdrope`."""
    resolved = _resolved("Qwen__Qwen3-8B")
    config = {**_config("Qwen__Qwen3-8B"),
              "rope_scaling": {"rope_type": "xdrope", "alpha": 1000.0,
                               "xdrope_section": [16, 16, 16, 16]}}
    findings = run_rotary_position_check(resolved, config)
    assert {f.op for f in findings} == {"rope"}
    assert len(findings) == 2
    assert all("`xdrope_section`" in f.detail and "op: xdrope" in f.detail
               for f in findings)


def test_mrope_under_an_xdrope_config_is_refused():
    """Three streams by `mrope_section` is not the four-stream XD kernel."""
    stem = "Qwen__Qwen3-VL-235B-A22B-Instruct"
    config = json.loads(json.dumps(_config(stem)))
    config["text_config"]["rope_scaling"] = {
        "rope_type": "xdrope", "xdrope_section": [16, 16, 16, 16]}
    findings = run_rotary_position_check(_resolved(stem), config)
    assert findings and {f.op for f in findings} == {"mrope"}
    assert all("op: xdrope" in f.detail for f in findings)


def test_a_one_position_rotation_under_mrope_section_is_refused_with_its_fix():
    """Qwen3-VL's config with a spec that rotates by one position: every text
    rotation is named, and the message says what to write."""
    resolved = _resolved("Qwen__Qwen3-8B")
    config = {**_config("Qwen__Qwen3-8B"),
              "rope_scaling": {"mrope_section": [24, 20, 20]}}
    findings = run_rotary_position_check(resolved, config)
    assert {f.op for f in findings} == {"rope"}
    assert len(findings) == 2
    assert all("op: mrope" in f.detail and "params: { rope: mrope }" in f.detail
               for f in findings)


def test_a_text_model_without_mrope_section_rotates_as_its_hf_class_does():
    """Qwen3.8-2.4T's text-only config carries no `mrope_section`, yet
    `Qwen3_5MoeTextRotaryEmbedding` sections by its default [11, 11, 10] and
    rotates by three streams (modeling_qwen3_5_moe.py:162, :193). A config
    without a section list does not decide the rotation, so the spec's
    `mrope` passes the check. The MTP layer, which transformers drops and the
    spec takes from vLLM, rotates by one position (`rope_emb`)."""
    stem = "Qwen__Qwen3.8-2.4T-A95B"
    resolved = _resolved(stem)
    program = Program.of(resolved)
    text = {n.op for n in program.nodes()
            if not n.children and n.uid not in program.encoder_uids
            and OP_REGISTRY.get(n.op) is not None
            and OP_REGISTRY[n.op].rotary_position_axes}
    assert config_rotation(_config(stem)) == ("", 1)
    assert text == {"mrope", "rope_emb"}
    assert run_rotary_position_check(resolved, _config(stem)) == []


def test_a_vision_tower_rotates_by_its_grid():
    """A vision tower rotates each patch by its (h, w) grid position: vLLM
    builds the cos/sin rows from both axes and applies them with
    `ApplyRotaryEmb` (qwen2_5_vl.py, kimi_k25_vit.py's 2-D table,
    mimo_v2_omni.py). Every encoder rotation of these models is `rope_2d`,
    except MiMo's audio transformer, which rotates by one position per token,
    and none is fused into a q/k kernel under vLLM."""
    expected = {
        "Qwen__Qwen3-VL-235B-A22B-Instruct": {"rope_2d"},
        "Qwen__Qwen3.8-27B": {"rope_2d"},
        "moonshotai__Kimi-K2.5": {"rope_2d"},
        "moonshotai__Kimi-K3": {"rope_2d"},
        "XiaomiMiMo__MiMo-V2.6-Pro-RL": {"rope_2d", "rope"},
    }
    for stem, ops in expected.items():
        program = Program.of(_resolved(stem))
        encoder = {n.op for n in program.nodes()
                   if not n.children and n.uid in program.encoder_uids
                   and OP_REGISTRY.get(n.op) is not None
                   and OP_REGISTRY[n.op].rotary_position_axes}
        assert encoder == ops, stem
        analysis = Deployment(program).at(input_tokens=128).checkpoint()
        fused, _ = Framework.of(get_binding("vllm"),
                                FUSION_SCHEMES).rewrite(analysis.program, analysis)
        assert "rope_2d" in {n.op for n in fused.nodes()
                             if not n.children and n.uid in fused.encoder_uids}, stem


def test_rope_2d_does_the_work_of_a_rotation():
    """Which position stream a channel half reads changes nothing per
    element: `rope_2d` and `rope` agree on FLOPs, parameters and shape."""
    rope_2d, rope = OP_REGISTRY["rope_2d"], OP_REGISTRY["rope"]
    dims = {"B": 2, "s_ctx": 16, "dim": 64, "n_heads": 4}
    assert rope_2d.flops_fn(dims) == rope.flops_fn(dims)
    assert rope_2d.params_fn(dims) == rope.params_fn(dims) == 0
    assert (rope_2d.shape_in, rope_2d.shape_out, rope_2d.couples_on) == (
        rope.shape_in, rope.shape_out, rope.couples_on)
    assert rope_2d.rotary_position_axes == 2


def test_an_encoder_rotation_reading_three_streams_is_refused(monkeypatch):
    """Multimodal RoPE is the text decoder's rotation; an encoder rotation
    reading three streams is named with what to write."""
    monkeypatch.setattr(OP_REGISTRY["rope_2d"], "rotary_position_axes", 3)
    stem = "Qwen__Qwen3-VL-235B-A22B-Instruct"
    findings = run_rotary_position_check(_resolved(stem), _config(stem))
    assert findings and {f.op for f in findings} == {"rope_2d"}
    assert all("encoder rotation" in f.detail and "op: rope_2d" in f.detail
               for f in findings)
