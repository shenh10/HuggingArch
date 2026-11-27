"""A pointwise activation is a cell only where an audited path launches it.

A plan takes its cells from the walk's own per-op rows and DROPS a row whose
op no signature measures (`plan.build_jobs`). For a view or a transpose that
is right — nothing runs. For a pointwise ACTIVATION it can be wrong: the
launch happens, costs bytes, and leaves no cell, so the gap reads as
coverage. That is how Gemma 2's final-logit soft-cap came to have two of its
three launches measured: vLLM runs it as divide / `tanh` / multiply, both
`scale_const` launches had a signature and a floor, and `tanh` had neither.

The opposite error is just as bad and less visible: declaring a cell for an
op no framework launches alone. **The absence of a binding for the span
around an op is OUR coverage gap, not evidence about what a framework
dispatches** — `exp` is the case in point. Both bound frameworks reach the
KDA decay gate inside a WIDER launch (sglang's `fused_kda_gate`, vLLM's
FlashKDA prepare, whose operands the graph has no nodes for), so a standalone
`exp` cell would have recorded a launch nobody makes, and could double-count
work already inside FlashKDA.

So the bar is: an op is measurable here only if some bound framework's
AUDITED source launches it by itself, cited, and the floor is that launch
rather than equivalent math. The tests below hold that bar from three sides —
the op has a signature, the framework it names really absorbs it in no bound
plan, and a provider that is enabled ACCEPTS the cell and matches the oracle.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


#: op -> (the span it sits in, the framework that leaves that span naive, the
#: line in that framework's source which launches the op).
#:
#: The span is named because absorption is per span, not per op: sglang fuses
#: DeepSeek-V4.1's engram gate (`fused_engram_gate`, which all three bindings
#: bind) and leaves Qwen's PLE gate naive, and the same `signed_sqrt`
#: primitive appears in both. A claim keyed on the op alone cannot tell those
#: apart.
AUDITED_STANDALONE = {
    # `logits / soft_cap`, `torch.tanh(logits)`, `logits * soft_cap`.
    "tanh": ("softcap_logits", "vllm",
             "model_executor/layers/logits_processor.py:114-116"),
    # `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`, in the branch that
    # runs when neither PLE fusion applies.
    "signed_sqrt": ("ple_gate", "sglang", "srt/models/qwen4_exp.py:1377"),
}

#: Activation ops that are NOT cells, and what each one lacks. Two different
#: things keep an op off the list above, and the entry says which:
#: no audited standalone launch, or a fact the cell cannot state.
NOT_YET_A_CELL = {
    # No audited launch of its own: sglang fuses it into `fused_kda_gate`,
    # and vLLM's default SM90+ KDA prefill is FlashKDA's prepare launch over
    # the RAW gate (`fusion_plans/fused_kda_gate.yaml`, which cites both),
    # with `chunk_kda_with_fused_gate` as the fallback. Neither dispatches
    # the primitive.
    "exp": "no audited path launches it alone — both frameworks reach it "
           "inside a wider KDA launch",
    # No audited launch of its own either: `relu`'s one live use is the
    # lightning indexer, and `indexer_logits` is bound by all three with no
    # `when:` on vLLM's or sglang's entry, so every indexer in the corpus is
    # a chain the plan matches. A contract would be needed first anyway — its
    # frame is the SCORE MATRIX `[B, n_heads, s_query, s_kv]` (the op's
    # `alt_shape_in`), not rows x trailing feature.
    "relu": "no audited path launches it alone — `indexer_logits` is bound "
            "by all three frameworks",
    # Both bindings fuse the clamped SwiGLU (`silu_and_mul_with_clamp`), and
    # the cell could not state the bound anyway: `clamp` names its constant
    # (`OpSpec.consts`) but no cell carries it, the way
    # `silu_and_mul_with_clamp` carries `clamp_limit` as a kernel param.
    "clamp": "the cell does not carry `clamp_limit` yet",
    # HY-V4's hyper-connection gates add `hc_eps` and no plan absorbs it, so
    # the launch is real — but the op declares no NAME for its constant, so
    # the cell cannot state what is added and no oracle can agree with it.
    "add_const": "the op declares no name for its constant, so no cell can "
                 "state what it adds",
}


def _activation_ops():
    from backend.arch.ops import OP_REGISTRY
    return sorted(name for name, spec in OP_REGISTRY.items()
                  if spec.is_activation)


def _plan_text(fused_op: str) -> str:
    """The plan yaml that declares `fused_op`, as written — its audit
    comments included, which is where the framework citations live."""
    from backend.inference import fusion_plans as FP
    import yaml
    for path in glob.glob(os.path.join(os.path.dirname(FP.__file__), "*.yaml")):
        text = Path(path).read_text(encoding="utf-8")
        if (yaml.safe_load(text) or {}).get("fused_op") == fused_op:
            return text
    raise AssertionError(f"no fusion plan declares {fused_op}")


def _plan_absorbs(fused_op: str, op: str) -> bool:
    """Whether `fused_op`'s naive pattern contains a node running `op`."""
    import yaml
    plan = yaml.safe_load(_plan_text(fused_op)) or {}
    nodes = ((plan.get("naive_pattern") or {}).get("nodes") or {}).values()
    return any(op == node.get("op") or op in (node.get("op_choices") or ())
               for node in nodes)


def _minimal_cell(signature: str):
    """A cell of `signature` as a plan carries one: the unary contract's axes,
    the activation dtype a served deployment runs."""
    import json
    from kernel_bench.invocation import Invocation
    dims = {"B": 2, "s_query": 8, "d_model": 16}
    dtype_env = {"activation": "bf16", "output": "bf16"}
    return Invocation.from_job({
        "signature": signature, "op": signature, "phase": "prefill", "tp": 1,
        "dims": dims, "shape_in": [[2, 8, 16]], "shape_out": [[2, 8, 16]],
        "shape_weight": None, "dtype_env": dict(dtype_env),
        "physical_key": [signature, json.dumps(dims, sort_keys=True),
                         json.dumps(dtype_env, sort_keys=True), "prefill"]})


def test_every_activation_op_is_measurable_or_named():
    from backend.inference.op_signatures import get_signature, op_signature
    unmeasurable = [op for op in _activation_ops()
                    if get_signature(op_signature(op)) is None]
    assert unmeasurable == sorted(NOT_YET_A_CELL), (
        f"activation ops with no signature moved: {unmeasurable}; each is "
        f"either a launch a plan drops or a cell nobody should plan, and the "
        f"entry has to say which")


def test_each_measurable_activation_names_a_span_its_framework_leaves_naive():
    """The registry side of the audited claim, checked three ways.

    An op is a cell because some framework runs it alone inside a span it
    does not fuse. So: the span must really contain the op, the named
    framework must bind no plan for it, and the source file the floor cites
    must be the file that span's own audit cites. None of this can prove the
    framework DISPATCHES the primitive — only reading its source does, which
    is what the anchor records — but it refuses the two ways such a claim
    goes stale: a binding that starts fusing the span, and a citation that
    drifts from the plan it is supposed to come from.
    """
    from backend.inference.op_signatures import get_signature, op_signature
    from backend.inference.framework_bindings import FRAMEWORK_BINDINGS

    measurable = [op for op in _activation_ops()
                  if get_signature(op_signature(op)) is not None]
    # Only the ones this file introduced carry the claim; the long-standing
    # unaries (silu, gelu, sigmoid, …) are the activation of a gated FFN or a
    # router and predate it.
    for op, (span, framework, anchor) in sorted(AUDITED_STANDALONE.items()):
        assert op in measurable, f"{op} names an audited path but has no cell"
        assert framework in FRAMEWORK_BINDINGS, (op, framework)
        assert _plan_absorbs(span, op), (
            f"{span} does not contain a {op} node, so it is not the span "
            f"{framework} leaves naive around it")
        bound = sorted(name for name, b in FRAMEWORK_BINDINGS.items()
                       if span in b.fused_ops())
        assert framework not in bound, (
            f"{op} is declared a standalone launch of {framework} because it "
            f"leaves {span} naive, but {framework} now binds it "
            f"(bound by {bound}); the cell should go or the span is wrong")
        path = anchor.split(":")[0]
        assert path in _plan_text(span), (
            f"{op}'s floor cites {path}, which {span}'s own audit does not "
            f"mention — one of the two is reading a different code path")
    for op, why in sorted(NOT_YET_A_CELL.items()):
        assert op not in measurable, f"{op} has a cell but is named a gap: {why}"


def test_a_measurable_activation_has_a_reference_and_a_floor_that_accepts():
    """"Has a floor" means a provider that is ENABLED and ACCEPTS the cell.

    Sharing a signature name is not enough: `Provider.accepts` is the gate a
    run takes, and it refuses a `disabled_reason` kernel and a declaration
    that does not cover the cell. A floor that passes the name check and
    fails this one is registered and undriveable — it turns "no kernel" into
    "a kernel that records nothing", which is worse because it reads as
    coverage.
    """
    from backend.inference.op_signatures import get_signature, op_signature
    from kernel_bench.env import Env
    from kernel_bench.providers import all_providers
    from kernel_bench.refs import get_ref

    env = Env(None, None, None, "H100-SXM", None)
    missing = {}
    for op in sorted(AUDITED_STANDALONE):
        signature = get_signature(op_signature(op))
        assert signature is not None, op
        inv = _minimal_cell(signature.name)
        accepting = [p.kernel_impl for p in all_providers()
                     if p.signature == signature.name and p.accepts(inv, env)]
        lacks = []
        if get_ref(signature.name) is None:
            lacks.append("no reference: the driver cannot build its inputs")
        if not accepting:
            lacks.append("no enabled provider accepts the cell")
        if lacks:
            missing[op] = lacks
    assert missing == {}, missing


def test_the_floors_run_the_frameworks_lowering_and_match_the_oracle():
    """Layer 3 for the two added here, on the CPU: a pointwise activation
    needs no card, so "this builds, runs and matches the fp32 reference" is
    checkable in the default suite.

    The `signed_sqrt` case is why the signed-zero column is here: at ordinary
    values `copysign(sqrt(|x|), x)` agrees with sglang's
    `sqrt(|x|) * sign(x)`, and random data never separates them, but they are
    different functions at -0.0 and the copysign spelling is three launches
    where the framework runs five.
    """
    torch = pytest.importorskip("torch")
    from kernel_bench.env import Env
    from kernel_bench.providers import all_providers
    from kernel_bench.refs import get_ref
    from tests.kernel_bench._inputs import build_inputs

    env = Env(None, None, None, "H100-SXM", None)
    for signature in sorted(AUDITED_STANDALONE):
        inv = _minimal_cell(signature)
        spec = get_ref(signature)
        inputs = build_inputs(spec, inv, "cpu")
        provider = next(p for p in all_providers()
                        if p.signature == signature and p.accepts(inv, env))
        case = provider.build(inv, env, inputs)
        out = case.to_canonical(case.run()).float()
        ref = spec.reference(inputs)
        assert out.shape == ref.shape, signature
        # bf16 keeps ~3 decimal digits; a wrong function misses by O(1).
        rel = (out - ref).abs().max() / ref.abs().max().clamp_min(1e-9)
        assert rel < 1e-2, (signature, provider.kernel_impl, float(rel))

    # Signed zero, where the two spellings of a signed square root part. The
    # EXPECTED VALUES ARE WRITTEN OUT rather than taken from the reference:
    # comparing the floor against the oracle only shows the two agree, which
    # they would do just as well if both were the wrong lowering.
    zero = torch.tensor([-0.0, 0.0, -4.0, 4.0])
    qwen = torch.tensor([0.0, 0.0, -2.0, 2.0])           # sqrt(|x|) * sign(x)
    floor = next(p for p in all_providers()
                 if p.kernel_impl == "torch_signed_sqrt")
    inv = _minimal_cell("signed_sqrt")
    built = floor.build(inv, env, {"x": zero})
    got = built.to_canonical(built.run())
    assert torch.equal(got, qwen), (got, qwen)
    assert not torch.signbit(got[0]).item(), got
    # …and the oracle is pinned the same way, so neither side can drift to
    # `copysign`, which sends -0.0 to -1e-3 and keeps the sign bit.
    assert torch.equal(get_ref("signed_sqrt").reference({"x": zero}), qwen)


def test_each_span_keeps_its_own_signed_sqrt_lowering():
    """A signed square root has two audited lowerings, one per span, and the
    bug this guards is one span's floor borrowing the other's chain —
    `_ple_floor_chain` timed `copysign` (four elementwise steps) where sglang
    runs `* sign()` (five), under the same `ple_gate` identity as the vLLM
    kernel it is compared against.

    Both the floor and the oracle of each span are checked against an
    expectation recomputed HERE, with the gate spelled out — and, crucially,
    ON A CELL WHERE THE TWO LOWERINGS DISAGREE. On random data they do not:
    `copysign(sqrt(|x|), x)` and `sqrt(|x|) * sign(x)` are the same number
    for every x except ±0, so an all-random probe passes under either chain
    and tests nothing. The zeroed query below drives the dot to exactly +0.0,
    where `* sign()` gives 0 (gate 0.5) and `copysign` gives +1e-3 (gate
    0.50025); the last assertion checks the probe really separates them, so
    this cannot quietly stop discriminating.
    """
    torch = pytest.importorskip("torch")
    from kernel_bench import refs
    from kernel_bench.providers.norm_act import (_engram_floor_chain,
                                                 _ple_floor_chain)

    hc, d, tokens, eps = 2, 4, 3, 1e-6
    g = torch.Generator().manual_seed(7)
    times_sign = (lambda t: t.abs().clamp_min(eps).sqrt() * t.sign())
    copysign = (lambda t: torch.copysign(t.abs().clamp_min(eps).sqrt(), t))

    def rms(y):
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)

    def norm_gain(x, w):
        return (rms(x.unflatten(-1, (hc, d))) * (1 + w)).flatten(-2)

    # ---- the PLE span: sglang runs `* sign()` (srt/models/qwen4_exp.py:1377)
    # The operand layouts the contract declares: activations
    # `[B, s_query, hc*d]`, each gain `[hc, d]`.
    i = {n: torch.randn(1, tokens, hc * d, generator=g) for n in ("x", "key")}
    i.update({n: torch.randn(hc, d, generator=g)
              for n in ("w_key", "w_query", "w_conv")})
    i["value"] = torch.randn(1, tokens, d, generator=g)
    i["eps"] = eps

    def ple_expected(cell, signed_sqrt):
        k = norm_gain(cell["key"], cell["w_key"])
        q = norm_gain(cell["x"], cell["w_query"])
        dot = (k * q).unflatten(-1, (hc, d)).sum(-1) * d ** -0.5
        gate = torch.sigmoid(signed_sqrt(dot))
        gated = (gate.unsqueeze(-1) * cell["value"].unsqueeze(-2)).flatten(-2)
        return gated, norm_gain(gated, cell["w_conv"])

    # A zero query zeroes its RMSNorm, so the dot is exactly +0.0.
    zero_dot = {**i, "x": torch.zeros_like(i["x"])}
    for cell in (i, zero_dot):
        want = ple_expected(cell, times_sign)
        for who, got in (("floor", _ple_floor_chain(cell, torch)),
                         ("oracle", refs.ple_gate(cell))):
            for port, (a, b) in enumerate(zip(got, want, strict=True)):
                assert torch.allclose(a, b, atol=1e-6), (who, port, a, b)
    assert not torch.allclose(ple_expected(zero_dot, times_sign)[0],
                              ple_expected(zero_dot, copysign)[0],
                              atol=1e-6), (
        "the PLE probe no longer separates the two lowerings, so the checks "
        "above would pass under either chain")

    # ---- the engram span: DeepSeek writes `copysign` (dsv41_engram.yaml:130)
    e = {"x": torch.randn(1, tokens, hc * d, generator=g),
         "kv": torch.randn(1, tokens, hc * d + d, generator=g),
         "q_weight": torch.randn(hc, d, generator=g),
         "k_weight": torch.randn(hc, d, generator=g), "eps": eps}

    def engram_expected(cell, signed_sqrt):
        x3 = cell["x"].unflatten(-1, (hc, d))
        key, value = cell["kv"].split([hc * d, d], dim=-1)
        dot = ((rms(x3) * cell["q_weight"])
               * (rms(key.unflatten(-1, (hc, d))) * cell["k_weight"])
               ).sum(-1) * d ** -0.5
        gate = torch.sigmoid(signed_sqrt(dot))
        return (x3 + gate.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)

    engram_zero = {**e, "x": torch.zeros_like(e["x"])}
    for cell in (e, engram_zero):
        want = engram_expected(cell, copysign)
        for who, got in (("floor", _engram_floor_chain(cell, torch)),
                         ("oracle", refs._engram_gate_reference(cell))):
            assert torch.allclose(got, want, atol=1e-6), (who, got, want)
    assert not torch.allclose(engram_expected(engram_zero, copysign),
                              engram_expected(engram_zero, times_sign),
                              atol=1e-6), (
        "the engram probe no longer separates the two lowerings")
