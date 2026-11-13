"""GPU oracle for the default SGLang K3.1 cache formats (no model weights).

Run in the SGLang Python environment, not HuggingArch's CPU analysis environment:
  python scripts/probe_kimi_k31_sglang.py --output /persistent/path/probe.json
The output is evidence, never an alternate runtime formula for HuggingArch.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    import torch
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache_separate
    from sglang.srt.configs.mamba_utils import KimiLinearStateShape

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sglang-source', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    n = 16
    latent = torch.randn(n, 512, device='cuda', dtype=torch.bfloat16)
    tail = torch.randn(n, 64, device='cuda', dtype=torch.bfloat16)
    packed, tail_bytes = quantize_k_cache_separate(latent, tail, scale_ue8m0=True)
    torch.cuda.synchronize()
    assert packed.numel() // n == 528
    assert tail_bytes.numel() // n == 128
    scales = packed.view(n, -1)[:, 512:].contiguous().view(torch.float32)
    assert torch.equal(scales.log2(), scales.log2().round())
    assert torch.equal(tail_bytes.contiguous().view(torch.bfloat16).reshape_as(tail), tail)
    from sglang.kernels.ops.kvcache.dsa_fused_store_index_cache import fused_store_index_k_cache
    key = torch.randn(512, 128, device='cuda', dtype=torch.bfloat16)
    slots = torch.arange(512, device='cuda', dtype=torch.int64)
    index_pages = torch.zeros((1, 64 * 132), device='cuda', dtype=torch.uint8)
    # DCP=8, rank 0 owns slots 0,8,...504, packed into one 64-row page.
    fused_store_index_k_cache(key, index_pages, slots, page_size=64, dcp_size=8, dcp_rank=0)
    torch.cuda.synchronize()
    index_scales = index_pages[:, 64 * 128:].contiguous().view(torch.float32).flatten()
    expected_scales = key[::8].float().abs().amax(dim=-1).clamp_min(1e-4) / 448
    torch.testing.assert_close(index_scales, expected_scales)
    assert not torch.equal(index_scales.log2(), index_scales.log2().round())
    state = KimiLinearStateShape.create(tp_world_size=8, num_heads=96, head_dim=128, conv_kernel_size=4)
    assert state.conv == [(3, 4608)]
    assert state.temporal == (12, 128, 128)
    report = {
        'sglang_commit': subprocess.check_output(['git', '-C', str(args.sglang_source), 'rev-parse', 'HEAD'], text=True).strip(),
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'mla_bytes_per_token_per_layer': (packed.numel() + tail_bytes.numel()) // n,
        'latent_bytes': packed.numel() // n,
        'tail_bytes': tail_bytes.numel() // n,
        'scales_physically_float32': True,
        'scales_are_power_of_two': True,
        'tail_is_losslessly_bfloat16': True,
        'index_bytes_per_token_per_layer': index_pages.numel() // 64,
        'index_scales_are_ordinary_float32': True,
        'dcp8_index_writes_only_owned_slots': True,
        'kda_tp8_conv_shape': state.conv,
        'kda_tp8_recurrent_shape': state.temporal,
        'scope': 'Actual GPU cache writer and SGLang state geometry; no full-model throughput claim.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
