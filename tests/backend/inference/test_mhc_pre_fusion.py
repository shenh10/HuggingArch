"""vLLM-Ascend runs the whole hyper-connection pre-mix as one kernel.

GLM-5.3-Flash's seam opens with seven primitives — flatten the residual
copies, RMS-normalise the flat stream unscaled, project it onto the mix
vector, split that into `pre` / `post` / `comb` with the Sinkhorn
projection on the cross-mix, then weight the copies by `pre` and reduce
them. The plugin runs all seven in `torch.ops._C_ascend.npu_hc_pre_v2`
(profiler `HcPre`): 117.3 us at 1024 tokens on one 910C die, against a
launch floor near 35 us, so the eager chain is not what runs there.

The CUDA bindings cut the same boundary differently — `mhc_pre_gemm` plus
`mhc_pre_big_fuse`, with `hc_split_sinkhorn` left standing between them —
because a launch absorbing the split has to publish TWO of that member's
outputs and the port keying could not say so. It can now
(`FusedNode.secondary_ports`), and this is the plan that needs it.

Falsification targets:
  * the plan exists, the Ascend binding dispatches it on an NPU, and no
    CUDA binding does;
  * GLM's step under the deployed Ascend binding has no pre-mix primitive
    left standing — no `hc_split_sinkhorn`, no unscaled norm, no
    `hc_mix` — while the unfused step has them all;
  * the launch publishes `post` and `comb` off the one split it absorbed,
    and its boundary shape carries both;
  * leaf cost survives the rewrite exactly.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import dim as dim_mod
from backend.arch.eval.fusion import Framework, FusedNode
from backend.arch.eval.transforms import Deployment
from backend.inference.framework_bindings import ServingDeployment, get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import shared_program, shared_walk

MODEL = "zai-org__GLM-5.3-Flash"
#: Leaves of the naive pre-mix no launch may leave standing.
PRE_LEAF_OPS = ("hc_split_sinkhorn", "rmsnorm_unscaled")
#: And of the naive post-mix.
POST_LEAF_OPS = ("hc_mix",)


def _ascend_framework() -> Framework:
    """The Ascend binding as a GLM deployment reaches it: a device, an
    activation dtype and the plugin's page. Without a deployment the
    `when: {device: [npu]}` on these entries holds for nothing and the
    rewrite is the identity — which is the correct answer for a request
    that named no card, and not what this file is about."""
    deployed = get_binding("vllm_ascend").deployed(ServingDeployment.at(
        gpu_sku="Ascend-910C-A3", activation_dtype="bf16",
        kv_block_size=128, tensor_parallel=8))
    return Framework.of(deployed, FUSION_SCHEMES)


def _deployment() -> Deployment:
    return Deployment(shared_program(MODEL)).at(batch=1, input_tokens=128)


def _leaf_totals(analysis) -> tuple:
    flops = params = 0
    for uid, cost in analysis.cost.values.items():
        if getattr(analysis.program.by_uid(uid), "children", None):
            continue
        flops += dim_mod.value_of(cost.flops)
        params += dim_mod.value_of(cost.params)
    return flops, params


@pytest.fixture(scope="module")
def fused():
    return shared_walk(_deployment().fused(_ascend_framework()))


def test_the_plan_is_dispatched_on_an_npu_and_nowhere_else():
    assert "mhc_pre" in FUSION_SCHEMES
    assert "mhc_pre" in {f.scheme_name for f in _ascend_framework().fusions}
    for framework in ("vllm", "sglang"):
        assert "mhc_pre" not in set(get_binding(framework).fused_ops()), (
            f"{framework} cuts this boundary as mhc_pre_gemm + "
            f"mhc_pre_big_fuse, with the split left standing")


def test_every_seam_is_two_launches(fused):
    launches = Counter(n.op for n in fused.program.nodes()
                       if isinstance(n, FusedNode))
    # One pre and one post per sublayer seam, and GLM's walked step has
    # three of them (`hc_sinkhorn_pre_norm_block` x the layer templates).
    assert launches["mhc_pre"] == launches["mhc_post"] > 0


def test_no_pre_mix_primitive_is_left_standing(fused):
    naive = shared_walk(_deployment())
    before = Counter(n.op for n in naive.program.nodes())
    after = Counter(n.op for n in fused.program.nodes())
    for op in PRE_LEAF_OPS + POST_LEAF_OPS:
        assert before[op] > 0, f"{op} is not in the unfused step"
        assert after[op] == 0, (
            f"{op} survived the rewrite: the launch that absorbs the pre-mix "
            f"has to take it")


def test_the_launch_publishes_both_coefficient_sets(fused):
    naive = shared_walk(_deployment())
    splits = {n.uid for n in naive.program.nodes()
              if n.op == "hc_split_sinkhorn"}
    pres = [n for n in fused.program.nodes()
            if isinstance(n, FusedNode) and n.op == "mhc_pre"]
    assert pres
    for launch in pres:
        absorbed = [uid for uid in launch.origins if uid in splits]
        assert len(absorbed) == 1, "one split per pre-mix"
        split = absorbed[0]
        assert launch.secondary_ports == {(split, "post"): "post_out",
                                          (split, "comb"): "comb_out"}
        # `pre` stays inside: the reduce in this very launch consumes it.
        assert launch.published_port(split, "") is None
        shape = fused.shapes.values[launch.uid]
        assert set(shape.ports) == {"post_out", "comb_out"}
        assert shape.ports["post_out"] == naive.shapes.values[split].ports["post"]
        assert shape.ports["comb_out"] == naive.shapes.values[split].ports["comb"]


def test_leaf_cost_survives_the_rewrite():
    d = _deployment()
    assert _leaf_totals(shared_walk(d)) == _leaf_totals(
        shared_walk(d.fused(_ascend_framework())))
