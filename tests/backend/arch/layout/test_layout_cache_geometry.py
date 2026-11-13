"""Physical cache geometry derived from one Analysis."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import sympy as sp

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.formats import known_dtype_bytes
from backend.arch.eval.dim import Dim
from backend.arch.eval.layer import Layer
from backend.arch.eval.values import Shape
from tests.snapshots import shared_loader
from backend.arch.layout.physical import physical_shapes
from backend.arch.layout.physical_cache import (
    _validate_cache_state_dependencies,
    physical_cache_view,
    whole_model_cache_view,
)
from backend.arch.layout.propagation import propagate_step
from backend.inference.parallel_config import ParallelConfig
from backend.arch.layout.stages import StagePlacement
from backend.inference.step_eval import evaluate_step
from backend.arch.eval.precision import PrecisionProfile


SNAPSHOTS = Path(__file__).resolve().parents[4] / "backend" / "arch" / "snapshots"
TP8 = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)


#: Every per-token cache slot in FP8 with a scale per 128 channels.
_KV_FP8_GROUP128 = PrecisionProfile.canonicalize({"cache": [{
    "select": {"slots": ["kv", "latent", "rope_key", "index_key",
                         "compressed_kv"]},
    "format": "fp8_group128"}]})


def _step(
    snapshot: str,
    *,
    phase: str = "prefill",
    pc: ParallelConfig = TP8,
    precision=None,
):
    payload = json.loads((SNAPSHOTS / snapshot).read_text())
    resolved = shared_loader().match(
        payload["config"], model_id=Path(snapshot).stem.replace("__", "/", 1))
    assert resolved is not None
    return evaluate_step(
        resolved,
        pc,
        phase=phase,
        batch=1,
        input_tokens=4096, output_tokens=0,
        # Prefill: 128 new tokens over 4096 positions, the rest cached.
        s_prefix_cached=4096 - 128 if phase == "prefill" else 0,
        precision=precision,
    )


def test_capacity_adapter_reads_both_views_without_per_rank_replay():
    from backend.inference.capacity import _cache_budget_terms

    step = _step("meta-llama__Llama-3.2-3B.json")
    local = physical_cache_view(step.analysis)
    whole = whole_model_cache_view(step.analysis)

    terms = _cache_budget_terms(local, whole, TP8)

    assert terms.reader is local
    assert terms.per_token > 0
    assert terms.shard_divisor == 8
    assert terms.replication_factor == 1


def test_capacity_adapter_projects_local_and_whole_views_to_the_same_stage():
    from backend.inference.capacity import _cache_budget_terms

    step = _step("meta-llama__Llama-3.2-3B.json")
    # 28 layers at pp = 4: stage 0 holds 7.
    stage = StagePlacement.of(step.analysis, 4).view(0)
    local = physical_cache_view(step.analysis).on(stage)
    whole = whole_model_cache_view(step.analysis).on(stage)

    assert [row.count for row in local.entries] == [7]
    assert [row.count for row in whole.entries] == [7]
    terms = _cache_budget_terms(local, whole, TP8)
    assert terms.shard_divisor == 8
    assert terms.replication_factor == 1


def test_dcp_position_owner_and_local_width_match_the_oracle():
    pc = ParallelConfig(
        tp_attention=4,
        tp_dense_ffn=4,
        tp_moe_expert=2,
        ep_moe=4,
        cp=2,
        dcp=2,
    )
    step = _step("meta-llama__Llama-3.2-3B.json", phase="decode", pc=pc)
    actual = physical_cache_view(propagate_step(step))

    assert {
        tensor.positions_over
        for row in actual.entries
        for tensor in row.cache.tensors
    } == {2}
    assert actual.storage_bytes_at(4096) > 0


def test_cp_prefill_does_not_divide_persistent_cache_positions():
    pc = ParallelConfig(
        tp_attention=4,
        tp_dense_ffn=4,
        tp_moe_expert=1,
        ep_moe=8,
        cp=2,
    )
    step = _step("meta-llama__Llama-3.2-3B.json", pc=pc)
    actual = physical_cache_view(propagate_step(step))

    assert {
        tensor.positions_over
        for row in actual.entries
        for tensor in row.cache.tensors
    } == {1}
    assert actual.storage_bytes_at(4096) > 0


def test_layersplit_occupancy_not_reads():
    """layersplit folds stored layers, not attention reads.

    Occupancy is min(L, ceil(L/cp)+1) / L over the whole view. Read
    traffic stays the full layer count. The whole-model view does not
    fold — it is the capacity ratio's denominator. Sequence positions
    stay whole (that cut is DCP's).
    """
    pcp_pc = ParallelConfig(cp=4, cp_scheme="pcp")
    uly_pc = ParallelConfig(cp=4, cp_scheme="ulysses")
    split_pc = ParallelConfig(cp=4, cp_scheme="layersplit")
    pcp = physical_cache_view(
        _step("meta-llama__Llama-3.2-3B.json", pc=pcp_pc).analysis)
    uly = physical_cache_view(
        _step("meta-llama__Llama-3.2-3B.json", pc=uly_pc).analysis)
    split = physical_cache_view(
        _step("meta-llama__Llama-3.2-3B.json", pc=split_pc).analysis)
    whole = whole_model_cache_view(
        _step("meta-llama__Llama-3.2-3B.json", pc=split_pc).analysis)

    assert pcp.layer_split_over == 1
    assert uly.layer_split_over == 1
    assert split.layer_split_over == 4
    assert whole.layer_split_over == 1
    assert [row.count for row in pcp.entries] == [
        row.count for row in split.entries]
    assert split.entries

    total = sum(row.count for row in split.entries)
    held = min(total, (total + 4 - 1) // 4 + 1)
    scale = held / total
    length = 4096
    assert held <= total
    assert uly.storage_bytes_at(length) == pytest.approx(
        pcp.storage_bytes_at(length))
    assert split.storage_bytes_at(length) == pytest.approx(
        pcp.storage_bytes_at(length) * scale)
    assert split.read_bytes_at(length)[0] == pytest.approx(
        pcp.read_bytes_at(length)[0])
    assert whole.storage_bytes_at(length) == pytest.approx(
        pcp.storage_bytes_at(length))
    split_decode = physical_cache_view(
        _step("meta-llama__Llama-3.2-3B.json", phase="decode",
              pc=split_pc).analysis)
    assert split_decode.layer_split_over == 4
    growth = split.growth_formula
    assert f"{held} / {total}" in growth["substituted"]
    assert growth["result_bytes"] == pytest.approx(split.growth_and_floor()[0])
    assert {
        tensor.positions_over
        for row in split.entries
        for tensor in row.cache.tensors
    } == {1}


def test_layersplit_occupancy_is_global_and_capped():
    from backend.arch.eval.cache_view import CacheClass, CacheView

    class _Layer:
        tensors = ()

        def bytes_at(self, cache_len, read=False):
            return 1.0

    one = CacheView(entries=(CacheClass(_Layer(), 1, None),)).on_layer_split(4)
    assert one.occupancy().held == 1
    assert one.storage_bytes_at(1) == 1.0

    # Same leaf-Σ twice (DSv3 dense+MoE MLA) is one stripe, not hybrid.
    same = CacheView(
        entries=(
            CacheClass(_Layer(), 3, 0),
            CacheClass(_Layer(), 58, 1),
        ),
    ).on_layer_split(4)
    assert same.occupancy().held == 17
    assert same.storage_bytes_at(1) == pytest.approx(17.0)

    class _Grow:
        def __init__(self, stored):
            self.is_growing = True
            self.elements = 1
            self.stored = stored
            self.storage_law = stored
            self.positions_over = 1
            self.position_bytes = 2
            self.resolved_format = "bf16"
            self.recurrent = False

    class _Typed:
        def __init__(self, stored):
            self.tensors = (_Grow(stored),)

        def bytes_at(self, cache_len, width, read=False):
            return 1.0

    hybrid = CacheView(
        entries=(
            CacheClass(_Typed("s_kv"), 12, 0),
            CacheClass(_Typed("Min(s_kv, 4096)"), 24, 1),
        ),
    )
    with pytest.raises(ValueError, match="mixes more than one KV cache type"):
        hybrid.on_layer_split(4)


def test_layersplit_occupancy_after_pipeline_stage():
    step = _step("meta-llama__Llama-3.2-3B.json", pc=ParallelConfig(pp=2))
    view = physical_cache_view(
        step.analysis).on_layer_split(4)
    # A stage that runs 14 of the block's 28 layers.
    staged = view.on(StagePlacement.of(step.analysis).view(0))
    assert sum(entry.count for entry in staged.entries) == 14
    assert staged.occupancy().held == 5


@pytest.mark.parametrize(
    "snapshot",
    (
        "Qwen__Qwen3-Coder-Next.json",
        "inclusionAI__Ling-2.6-1T.json",
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.json",
    ),
)
def test_dcp_divides_cached_positions_but_not_recurrent_state(snapshot):
    dcp = ParallelConfig(
        tp_attention=8,
        tp_dense_ffn=8,
        tp_moe_expert=8,
        ep_moe=2,
        cp=2,
        dcp=2,
    )
    sharded = physical_cache_view(
        propagate_step(_step(snapshot, phase="decode", pc=dcp,
                             precision=_KV_FP8_GROUP128)),
    )

    recurrent = [
        tensor
        for row in sharded.entries
        for tensor in row.cache.tensors
        if tensor.recurrent
    ]
    positioned = [
        tensor
        for row in sharded.entries
        for tensor in row.cache.tensors
        if not tensor.recurrent
    ]
    assert recurrent and positioned
    assert {tensor.positions_over for tensor in recurrent} == {1}
    assert {tensor.positions_over for tensor in positioned} == {2}

    # Each tensor at its own format's width: the profile's FP8 for the
    # per-token slots, the state's own (F32 / the model's) for the rest —
    # plus the group scales the FP8 format stores beside every position,
    # divided over the position owners with the data they scale.
    expected = 0.0
    for row in sharded.entries:
        for tensor in row.cache.tensors:
            dynamic, _static = tensor.scale_overhead()
            expected += (
                (float(tensor.elements) * known_dtype_bytes(tensor.dtype)
                 + dynamic * tensor.resolved_format.scale_bytes)
                * float(tensor.positions(4096))
                * row.count
                / (1 if tensor.recurrent else 2)
            )
    assert sharded.storage_bytes_at(4096) == pytest.approx(expected,
                                                           rel=1e-12)
    growth = sharded.growth_formula
    assert growth["result_bytes"] == sharded.growth_and_floor()[0]
    data, scales = growth["subterms"]
    assert "/ 2" in data["substituted"]
    assert scales["result_bytes"] > 0
    assert "/ 2" in scales["substituted"]
    assert data["result_bytes"] + scales["result_bytes"] == pytest.approx(
        growth["result_bytes"], rel=1e-12)


def test_cache_geometry_never_replays_the_evaluator():
    step = _step("meta-llama__Llama-3.2-3B.json")
    analysis = propagate_step(step)

    view = physical_cache_view(analysis)

    assert view.entries


def test_cache_geometry_rejects_a_foreign_program():
    step = _step("meta-llama__Llama-3.2-3B.json")
    other = _step("openai__gpt-oss-20b.json")
    analysis = propagate_step(step)
    foreign = physical_shapes(propagate_step(other))

    with pytest.raises(ValueError, match="different Program"):
        cache_mod.derive(
            analysis,
            geometry=cache_mod.CacheGeometry(
                foreign,
                analysis.cache_state,
            ),
        )


@pytest.mark.parametrize("values", ({}, {10**9: 1}))
def test_cache_geometry_requires_the_exact_recurrent_state_set(values):
    step = _step(
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.json",
        phase="decode",
    )
    analysis = propagate_step(step)

    with pytest.raises(ValueError, match="every recurrent state exactly once"):
        cache_mod.derive(
            analysis,
            geometry=cache_mod.CacheGeometry(
                physical_shapes(analysis),
                Layer(analysis.program, values),
            ),
        )


@pytest.mark.parametrize(
    ("axes", "binding"),
    (
        (
            ["n_ssm_heads", "d_ssm_state"],
            {
                "n_ssm_heads": Dim(sp.Symbol("n_ssm_heads"), 256),
                "d_ssm_state": Dim(sp.Symbol("d_ssm_state"), 128),
            },
        ),
        (
            ["H", "d_ssm_state"],
            {
                "H": Dim(sp.Symbol("n_ssm_heads"), 256),
                "d_ssm_state": Dim(sp.Symbol("d_ssm_state"), 128),
            },
        ),
    ),
)
def test_live_state_symbol_requires_kernel_axis_provenance(axes, binding):
    step = _step(
        "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.json",
        phase="decode",
    )
    sharding = propagate_step(step).rule
    carrier = SimpleNamespace(display_path="synthetic.recurrence")
    shape = Shape(out=(Dim(sp.Symbol("d_model"), 4096),))

    with pytest.raises(ValueError, match="not carried by the physical kernel"):
        _validate_cache_state_dependencies(
            carrier,
            axes,
            binding,
            shape,
            shape,
            sharding,
        )


def _recurrent(view, name):
    return {
        (int(tensor.elements), tensor.rows, tensor.inner)
        for row in view.entries
        for tensor in row.cache.tensors
        if tensor.recurrent and tensor.name == name
    }


@pytest.mark.parametrize(
    ("pc", "rows"),
    ((ParallelConfig(), 4096), (TP8, 512)),
)
def test_mamba2_state_groups_its_scales_along_d_state(pc, rows):
    """Nemotron-3's SSM state is `[n_ssm_heads, d_ssm_head, d_ssm_state]`
    (vLLM and sglang `mamba2_state_shape` alike): its innermost axis is
    `d_ssm_state` = 128, not the scan output's `d_ssm_head` = 64, and its
    rows are heads × head dim on each rank."""
    step = _step(
        "nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.json",
        phase="decode", pc=pc)

    view = physical_cache_view(step.analysis)

    assert _recurrent(view, "recurrent_state") == {(rows * 128, rows, 128)}


def test_a_state_holds_the_product_of_its_declared_axes():
    """A state's size is its declared shape multiplied out, per rank: DSv4's
    CSA compressor buffers are `[2 * compress_ratio, Cdim_io]` = 8 × 1024
    (sglang `view(bs, coff * ratio, coff * head_dim)`), whole on every rank."""
    step = _step("deepseek-ai__DeepSeek-V4-Flash.json", phase="decode")

    view = physical_cache_view(step.analysis)

    assert (8 * 1024, 8, 1024) in _recurrent(view, "compressor.kv_state")


def test_recurrent_state_marks_the_physical_view_as_tp_sharded():
    step = _step("inclusionAI__Ling-2.6-1T.json", phase="decode")

    view = physical_cache_view(step.analysis)

    assert view.tp_shards_cache


def test_pipeline_stages_recount_one_cache_view_and_conserve_layers():
    pc = ParallelConfig(
        tp_attention=8,
        tp_dense_ffn=8,
        tp_moe_expert=8,
        pp=4,
    )
    step = _step("openai__gpt-oss-20b.json", pc=pc)
    analysis = propagate_step(step)
    view = physical_cache_view(analysis)
    placement = StagePlacement.of(analysis, pc.pp)

    whole = view.storage_bytes_at(4096)
    stages = tuple(
        view.on(placement.view(stage)).storage_bytes_at(4096)
        for stage in range(pc.pp)
    )

    assert sum(stages) == whole


_INNER_REPEAT = """
match: { model_type: [_rep_cache] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  inner:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    graph:
      kv: { op: linear, in: x, dims: { in: d_model, out: d_model } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  prod:
    graph:
      body: { op: inner, in: x, repeat: "2" }
    outputs: [body]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: p, rest: true }
      map:
        p: { template: prod }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""


def _inner_repeat_analysis(pc: ParallelConfig):
    """Three layers, each holding its cached component twice
    (`repeat: "2"` inside the layer template)."""
    import os
    import tempfile

    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.loader import SpecLoader

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(_INNER_REPEAT)
        path = f.name
    loader = SpecLoader()
    try:
        spec = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    resolved = loader.resolve(spec, {
        "torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3, "vocab_size": 16,
        "model_type": "_rep_cache"})
    deployment = Deployment(Program.of(resolved), config=pc)
    return deployment.at(batch=1, input_tokens=16).analyze()


def test_a_cache_under_an_inner_repeat_is_held_once_per_copy():
    """Three layers of two copies hold six caches, not three: the count is
    the walk's `instances`, every enclosing `repeat:` included."""
    analysis = _inner_repeat_analysis(ParallelConfig())
    for view in (whole_model_cache_view(analysis),
                 physical_cache_view(analysis)):
        (entry,) = view.entries
        assert entry.count == 6
        # 6 copies x 8 latent channels x 2 B x 16 positions
        assert view.storage_bytes_at(16) == 6 * 8 * 2 * 16


def test_pipeline_stages_split_the_inner_repeated_cache_exactly():
    """At pp=2 the three layers split 2 + 1; each stage keeps its layers'
    copies of the cache, and the stages sum to the whole model."""
    analysis = _inner_repeat_analysis(ParallelConfig(pp=2))
    view = physical_cache_view(analysis)
    placement = StagePlacement.of(analysis)
    counts = [sum(entry.count for entry in view.on(placement.view(s)).entries)
              for s in range(placement.pp)]
    assert counts == [4, 2]
    assert sum(counts) == sum(entry.count for entry in view.entries)


def test_only_a_deployment_answers_whether_tp_divides_the_cache():
    """The per-rank view measures it off the projected widths — GQA heads
    narrow at tp=8, nothing narrows at tp=1 — and the whole-model view,
    which has no deployment, does not answer."""
    tp8 = _step("meta-llama__Llama-3.2-3B.json").analysis
    tp1 = _step("meta-llama__Llama-3.2-3B.json", pc=ParallelConfig()).analysis
    assert physical_cache_view(tp8).tp_shards_cache is True
    assert physical_cache_view(tp1).tp_shards_cache is False
    assert whole_model_cache_view(tp8).tp_shards_cache is None
