"""CPU storage oracle for SGLang K3.1 DCP; no GPU or full model is allocated.

Executes the pinned source's two replica-allocation methods with a CPU
all-gather stand-in. The stand-in repeats a rank's rows: values are irrelevant
for allocation/alias accounting. This does not measure CUDA allocator reserves,
network traffic, latency, or the complete server resident set.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Optional

import torch

COMMIT = "1b2870d33de7b5f31844fd74e97eea6d7896c53a"


class UnquantizedLinearMethod:
    pass


class DeepseekV2AttentionMLA(torch.nn.Module):
    pass


class CpuGroup:
    world_size = 8
    rank_in_group = 0

    def all_gather(self, tensor, dim):
        return torch.cat([tensor] * self.world_size, dim=dim)


def source_method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    # Only imports of SGLang's two type guards are replaced; all allocation,
    # aliasing, branch and shape operations below execute the actual source.
    method.body = [n for n in method.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def buffers(m):
    return {
        "q_b_proj.weight": m.q_b_proj.weight,
        "kv_b_proj.weight": m.kv_b_proj.weight,
        **{name: getattr(m, name) for name in (
            "w_kc", "w_vc", "q_b_proj_qrep_weight", "w_kc_qrep", "w_vc_qrep", "_dcp_v_weights"
        ) if getattr(m, name, None) is not None},
    }


def unique_bytes(tensors):
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return sum(storages.values())


def topk_owner_oracle(*, context, topk, degree, concentrate_rank=None):
    """An independent exact CPU oracle, not a performance prediction."""
    def ordering(position):
        # Deterministic score descending, position ascending (including ties).
        score = int(position % degree == concentrate_rank) if concentrate_rank is not None else 0
        return (-score, position)

    global_best = sorted(range(context), key=ordering)[:topk]
    local_candidates = [p for rank in range(degree)
                        for p in sorted(range(rank, context, degree), key=ordering)[:topk]]
    merged = sorted(local_candidates, key=ordering)[:topk]
    assert merged == global_best
    return [sum(p % degree == rank for p in merged) for rank in range(degree)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sglang-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.sglang_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError(f"Re-audit source before changing pinned revision: {revision}")
    group = CpuGroup()
    ns = dict(torch=torch, Optional=Optional, logger=logging.getLogger(__name__),
              UnquantizedLinearMethod=UnquantizedLinearMethod,
              DeepseekV2AttentionMLA=DeepseekV2AttentionMLA,
              get_parallel=lambda: SimpleNamespace(dcp_group=group),
              _is_block_fp8=lambda _: False)
    root = args.sglang_source / "python/sglang/srt"
    prepare_q = source_method(root / "model_executor/model_runner.py", "ModelRunner", "_prepare_replicated_q_proj", ns)
    prepare_v = source_method(root / "models/deepseek_common/attention_forward_methods/forward_mla.py",
                              "DeepseekMLAForwardMixin", "_prepare_dcp_v_weights", ns)
    m = DeepseekV2AttentionMLA()
    m.has_q_b_proj = True
    m.q_b_proj = torch.nn.Linear(1536, 12 * 192, bias=False, dtype=torch.bfloat16)
    m.q_b_proj.quant_method = UnquantizedLinearMethod()
    m.kv_b_proj = torch.nn.Linear(512, 12 * 256, bias=False, dtype=torch.bfloat16)
    m.w_scale = 1.0
    checkpoint_local = unique_bytes([m.q_b_proj.weight, m.kv_b_proj.weight])
    # KimiK3ForCausalLM.post_load_weights: these contiguous absorbed weights
    # are distinct allocations; the original kv_b_proj remains resident.
    wk, wv = m.kv_b_proj.weight.unflatten(0, (-1, 256)).split([128, 128], dim=1)
    m.w_kc = wk.transpose(1, 2).contiguous().transpose(1, 2)
    m.w_vc = wv.contiguous().transpose(1, 2)
    m._dcp_v_weights = None
    before_replication = unique_bytes(buffers(m).values())
    prepare_q(SimpleNamespace(model=m))
    after_replication = unique_bytes(buffers(m).values())
    assert m.q_b_proj.weight.untyped_storage().data_ptr() == m.q_b_proj_qrep_weight.untyped_storage().data_ptr()
    prepare_v(m, group)
    after_projection = unique_bytes(buffers(m).values())
    assert m.w_vc_qrep.untyped_storage().data_ptr() != m._dcp_v_weights.untyped_storage().data_ptr()
    # Read the padded push slot declaration from this same pinned source.
    slot_tree = ast.parse((args.sglang_source / "python/sglang/kernels/ops/attention/dcp_push_combine.py").read_text())
    slot_decls = [n for n in slot_tree.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id in {"HEAD_DIM", "SLOT_BYTES_PER_ROW"}]
    slot_ns = {}
    exec(compile(ast.Module(body=slot_decls, type_ignores=[]), "push-slot-declarations", "exec"), slot_ns)
    heads, d_nope, d_latent = m.w_kc_qrep.shape
    d_value = m.w_vc_qrep.shape[-1]
    q_head = m.q_b_proj_qrep_weight.shape[0] // heads
    d_tail = q_head - d_nope
    slot_bytes = slot_ns["SLOT_BYTES_PER_ROW"]
    assert slot_ns["HEAD_DIM"] == d_value
    result = {
        "sglang_commit": revision,
        "scope": "CPU source-method allocation/alias oracle; not GPU allocator or server memory measurement",
        "tp": 8, "dcp": 8, "dsa_layers": 24,
        "bytes_per_layer_per_rank": {
            "checkpoint_qb_and_kvb": checkpoint_local,
            "with_local_absorbed_weights": before_replication,
            "after_prepare_replicated_q_proj": after_replication,
            "after_cuda_v_preprojection": after_projection,
            "local_absorb_extra": before_replication - checkpoint_local,
            "qrep_extra_over_local_absorbed": after_replication - before_replication,
            "cuda_v_preprojection_extra": after_projection - after_replication,
        },
        "bytes_all_24_layers_per_rank": {
            "qrep_extra_over_same_tp8_without_qrep": 24 * (after_replication - before_replication),
            "qrep_and_cuda_v_preprojection_extra_over_local_absorbed": 24 * (after_projection - before_replication),
            "all_extras_over_checkpoint_graph": 24 * (after_projection - checkpoint_local),
        },
        "selection_oracle": {
            "context": 32768, "topk": 2048, "dcp": 8,
            "tie_ordered_valid_slots_per_rank": topk_owner_oracle(context=32768, topk=2048, degree=8),
            "rank0_preferred_valid_slots_per_rank": topk_owner_oracle(context=32768, topk=2048, degree=8, concentrate_rank=0),
            "local_candidates_merge_equals_global_topk": True,
        },
        "communication_geometry_bytes_per_query_per_layer": {
            "scope": "Logical buffers and declared push slots; excludes synchronization and transport overhead; not measured NVLink traffic",
            "full_head_q": heads * (d_latent + d_tail) * 2,
            "unprojected_partial_and_raw_lse": heads * (d_latent * 2 + 4),
            "projected_partial_and_raw_lse": heads * (d_value * 2 + 4),
            "push_slots_all_heads": heads * slot_bytes,
            "push_slots_remote_heads_per_sender": (group.world_size - 1) * (heads // group.world_size) * slot_bytes,
        },
        "q_b_original_shard_aliases_replica": True,
        "cuda_v_preprojection_is_separate_from_w_vc_qrep": True,
        "buffers_after_projection": {name: {"shape": list(t.shape), "logical_bytes": t.numel() * t.element_size(),
                                             "storage_bytes": t.untyped_storage().nbytes()}
                                     for name, t in buffers(m).items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
