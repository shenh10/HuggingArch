"""`point_row_of`'s factored road against its capture-per-point twin.

The design (docs/point_row_templates.md) factors the single key
derivation into a structure half no point moves (`_RowTemplate`, decided
once per row and deployment) and a values half re-read at every point.
This is the oracle that keeps the factoring honest: the same walk, the
same points, run with the template road and with the capture-per-point
road (today's composition written the long way), must produce the same
`PointRow` at every call, in the same order — and the comparison must be
able to fail, which the mutation test proves.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any
from dataclasses import fields, replace as dc_replace

import pytest

import backend.inference.benchmark_identity as benchmark_identity
import backend.inference.calibration_overlay as calibration_overlay
from backend.api.gallery_frontier import _GALLERY_PRECISION, _layout_law
from backend.inference.calibration_overlay import (
    DeploymentContext, _deployment_key)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.parallel_sweep import class_cards
from tests.snapshots import snapshot_config

#: The corpus the design names, on the H100 walk class the Gallery warms:
#: the simple end (Llama, Qwen3-30B), the warm's time-weighted majors
#: (DeepSeek V4.1-Flash and V3), the model that owns the exception row
#: (Qwen3.8-Flash-Next: a DSA KV pool whose projection refuses at small
#: contexts — 133 keyed calls against 310 refusals on the probe job) and
#: the KDA architecture. The looped-stack family is covered by
#: DeepSeek-V4.1-Flash (no GLM looped model is checked out). Only the
#: first case runs by default; the rest are `regression` by the repo's
#: rule for a parametrised test's non-first cases, and the walks are
#: minutes long — the deliberate trim is this comment, not silence.
CASES = [
    ("Qwen/Qwen3-30B-A3B", ParallelConfig(ep_moe=8, dp_attention=8)),
    ("meta-llama/Llama-3.2-3B", ParallelConfig(tp_attention=2, tp_dense_ffn=2)),
    ("deepseek-ai/DeepSeek-V4.1-Flash",
     ParallelConfig(ep_moe=8, dp_attention=8)),
    ("Qwen/Qwen3.8-Flash-Next", ParallelConfig(ep_moe=8, dp_attention=8)),
    ("deepseek-ai/DeepSeek-V3", ParallelConfig(ep_moe=8, dp_attention=8)),
]
#: Kimi-K3 (the KDA architecture) is not here because it cannot walk this
#: class at all: 1.4 TB of weights are capacity-refused on every Gallery
#: preset (EP8·DP8, TP8, TP1 — and TP16), so the warm never asks its rows
#: and neither does this oracle. Recorded, not silent.


def _walk(model: str, parallel: ParallelConfig):
    snapshot_config(model)
    cards = class_cards("H100-SXM")
    return _layout_law((model, cards, 0.9, _GALLERY_PRECISION.canonical(),
                        parallel.to_dict()))


def _digest(point_row) -> str:
    """One PointRow's whole content, as one digest: every field, order
    included, so two roads that disagree anywhere disagree here."""
    return hashlib.sha256(repr(tuple(
        (f.name, repr(getattr(point_row, f.name)))
        for f in fields(type(point_row)))).encode()).hexdigest()


@pytest.fixture
def point_row_digests(monkeypatch):
    """Every `point_row_of` call's digest, in call order, for one walk —
    with the template road on (`templated`) or off (`plain`: the structure
    half re-captured at every point, the shape memo off — today's
    composition written the long way)."""
    def run(model: str, parallel: ParallelConfig, *, templated: bool):
        # The class overlay cache shares the walk across runs; a fresh
        # run needs a fresh cache so the template capture fires (the
        # mutation test corrupts the CAPTURE, which only happens on
        # the first call for a row — a cached overlay would skip it).
        import backend.inference.capacity as _cap
        _cap._CLASS_OVERLAYS.clear()
        digests: list[str] = []
        stats = Counter()
        orig = calibration_overlay.point_row_of
        def spy(row, identity, deployment, **kw):
            if not templated:
                # The plain road calls the default signature: the private
                # precomputed key is stripped, so `point_row_of` without it
                # is exercised on every call (its parameter must not
                # shadow the module function it falls back to).
                kw.pop("_dkey", None)
            out = orig(row, identity, deployment, **kw)
            digests.append(_digest(out))
            return out
        calibration_overlay.point_row_of = spy
        try:
            if templated:
                row_template = benchmark_identity.RowIdentity.row_template

                def counted_template(self, row, deployment_key):
                    template = row_template(self, row, deployment_key)
                    if (row.get("uid") is not None
                            and str(row.get("aggregation_id") or "")):
                        stats["hits" if template is not None
                              else "captures"] += 1
                    return template

                monkeypatch.setattr(benchmark_identity.RowIdentity,
                                    "row_template", counted_template)
                remember = benchmark_identity.RowIdentity.remember_row_template

                def counted_remember(self, row, deployment_key, template):
                    replaced = remember(self, row, deployment_key, template)
                    if replaced:
                        # A lookup missed a key the store holds: impossible
                        # on the production road, where both use one key.
                        stats["recaptures"] += 1
                    return replaced

                monkeypatch.setattr(benchmark_identity.RowIdentity,
                                    "remember_row_template", counted_remember)
            else:
                monkeypatch.setattr(
                    benchmark_identity.RowIdentity, "row_template",
                    lambda self, row, deployment_key: None)
                monkeypatch.setattr(
                    benchmark_identity, "_memo_key",
                    lambda op, shape_in, args: None)
            record = _walk(model, parallel)
        finally:
            stats = dict(stats)
            calibration_overlay.point_row_of = orig
            monkeypatch.undo()
        return digests, record, stats
    yield run


def _record_without_times(record):
    """The record with wall-clock fields pruned, so two walks of the same
    code compare (everything else is the derivation's own output)."""
    def prune(value):
        if isinstance(value, dict):
            return {key: prune(item) for key, item in value.items()
                    if "wal" not in key and "elapsed" not in key}
        if isinstance(value, list):
            return [prune(item) for item in value]
        return value
    return json.dumps(prune(record), sort_keys=True)


@pytest.mark.parametrize("model,parallel", CASES)
def test_the_factored_road_is_the_capture_per_point_road(
        point_row_digests, model, parallel):
    templated, record_t, stats_t = point_row_digests(
        model, parallel, templated=True)
    plain, record_p, _ = point_row_digests(model, parallel, templated=False)
    assert templated == plain, (
        f"{model}: {len(templated)} vs {len(plain)} PointRow calls; "
        f"first differing call "
        f"{next((i for i, (a, b) in enumerate(zip(templated, plain, strict=False)) if a != b), None)}")
    assert _record_without_times(record_t) == _record_without_times(record_p)
    # The usage gate, exactly: every call either replayed a template or
    # captured one (no third path), no row captured twice (a recapture is
    # a lookup that missed a key the store holds — impossible on this
    # road, counted as a tripwire), and the fold actually happened.
    assert stats_t["captures"] + stats_t["hits"] == len(templated)
    assert stats_t.get("recaptures", 0) == 0
    assert 0 < stats_t["captures"] < stats_t["hits"]
    assert stats_t["hits"] > 5 * stats_t["captures"]


def test_the_comparison_is_not_vacuous(point_row_digests, monkeypatch):
    """A corrupted template must reach the record, or the equality above
    proves nothing (walk-identity R4's lesson: an oracle that cannot be
    made to scream is a report). Two corruptions of the template — a
    literal field and a structure decision — and the memo key's
    composition held where a collision would begin."""
    model, parallel = CASES[0]
    clean, _, _ = point_row_digests(model, parallel, templated=True)

    real = benchmark_identity.RowIdentity.remember_row_template
    def corrupt_dtype(self, row, deployment_key, template):
        # NamedTuple's own replace — dataclasses.replace refuses one, and
        # a corrupt() that raises would end the walk and make the poisoned
        # run differ by truncation, which proves nothing.
        real(self, row, deployment_key, template._replace(
            dtype_env=(template.dtype_env[0] + ' ', template.dtype_env[1])))
    monkeypatch.setattr(benchmark_identity.RowIdentity,
                        "remember_row_template", corrupt_dtype)
    poisoned, _, _ = point_row_digests(model, parallel, templated=True)
    monkeypatch.undo()
    # The scream must be a value difference: the poisoned walk completes
    # the same calls (a crash would differ by truncation, vacuously).
    assert len(poisoned) == len(clean)
    assert poisoned != clean

    def corrupt_op(self, row, deployment_key, template):
        keyed = template.keyed_as
        real(self, row, deployment_key, template._replace(
            keyed_as=keyed._replace(op=keyed.op + "__corrupted")))
    monkeypatch.setattr(benchmark_identity.RowIdentity,
                        "remember_row_template", corrupt_op)
    poisoned, _, _ = point_row_digests(model, parallel, templated=True)
    monkeypatch.undo()
    assert len(poisoned) == len(clean)
    assert poisoned != clean

    # The memo key's composition, held directly: dropping an argument
    # from the key is a collision by construction, so the end-to-end
    # mutation only screams where the model exercises the dimension —
    # asserted at the key instead, where it always does.
    args = dict(shape_out=None, shape_in_exact=None, shape_out_exact=None,
                spec_dims={"d_rope": 64}, comm=None, routed_rows=None,
                operand_ports=None)
    differing = dict(args, spec_dims={"d_rope": 32})
    assert benchmark_identity._memo_key("op", [2, 4], args) != (
        benchmark_identity._memo_key("op", [2, 4], differing))


def test_a_template_is_not_replayed_across_deployments():
    """The template key carries the binding, not six request fields: two
    requests that differ only in what the binding's `when:` reads (server
    flags, activation dtype, device) must not replay one another's
    dispatch — the technical review's blocker, held as a key-level fact
    (the binding is a frozen dataclass, so structure is identity)."""
    import backend.inference.framework_bindings as bindings

    base = bindings.get_binding("vllm")
    flagged = dc_replace(
        base.deployment,
        server_flags=(("attention_backend", "FLASHINFER"),))
    other = dc_replace(base, deployment=base.deployed(flagged))
    a = DeploymentContext(
        framework="vllm", framework_binding=base,
        parallel_signature="ep=8", gpu_sku="H100-SXM", phase="decode")
    b = DeploymentContext(
        framework="vllm", framework_binding=other,
        parallel_signature="ep=8", gpu_sku="H100-SXM", phase="decode")
    assert _deployment_key(a) != _deployment_key(b)
    assert _deployment_key(a) == _deployment_key(
        dc_replace(a, gpu_sku="H100-SXM"))


def test_a_refusal_while_keying_names_the_node(monkeypatch):
    """`of`'s contract through the factored road: a refusal raised while
    the row is keyed carries its node's name (`node_context`), which the
    re-review asked to hold by test rather than by reading the code."""
    from backend.arch.eval.refusal import refusal_node
    captured: dict[str, Any] = {}

    orig = calibration_overlay.point_row_of
    def spy(row, identity, deployment, **kw):
        if "identity" not in captured:
            captured.update(row=row, identity=identity, deployment=deployment,
                            model_id=kw.get("model_id", ""),
                            spec_hash=kw.get("spec_hash", ""))
        return orig(row, identity, deployment, **kw)
    calibration_overlay.point_row_of = spy
    try:
        _walk(*CASES[0])
    finally:
        calibration_overlay.point_row_of = orig
    assert captured, "the walk asked no row"

    def boom(self, row, keyed_as):
        raise ValueError("a refusal raised while keying")
    monkeypatch.setattr(benchmark_identity.RowIdentity, "keyed_shape", boom)
    with pytest.raises(ValueError) as raised:
        orig(captured["row"], captured["identity"], captured["deployment"],
             model_id=captured["model_id"], spec_hash=captured["spec_hash"])
    assert refusal_node(raised.value) is not None
