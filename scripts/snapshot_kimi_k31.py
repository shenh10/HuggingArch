"""Freeze local K3.1 checkpoint headers and the audited SGLang source.

Run from the repository root with Python 3.13 and the locked dependencies.
Only safetensors headers are read; model tensors are never loaded.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.arch.quant_schemes import scheme_named
from backend.arch.snapshots import write_snapshot
from backend.hf.metadata import metadata_record, scan_local_hf_repo

REVISION = '1b2870d33de7b5f31844fd74e97eea6d7896c53a'
SOURCES = (
    'srt/models/kimi_k3.py',
    'srt/models/kimi_k3_vl.py',
    'srt/models/deepseek_v2.py',
    'srt/configs/kimi_linear.py',
    'srt/configs/kimi_k3.py',
    'srt/configs/mamba_utils.py',
    'srt/layers/attention/dsa/dsa_indexer.py',
    'kernels/ops/quantization/dsa_quant_k_cache.py',
    'srt/arg_groups/model_overrides/kimi_k3.py',
    'srt/layers/attn_residual.py',
    'kernels/jit/csrc/dsa/fused_store_index_cache.cuh',
    'kernels/ops/kvcache/dsa_fused_store_index_cache.py',
    'srt/arg_groups/overrides.py',
    'srt/layers/layer_boundary/residual/attn_bank.py',
    'srt/managers/mm_utils.py',
    'srt/layers/attention/dsa_backend.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sglang-source', type=Path, required=True)
    parser.add_argument('--overlay-dir', type=Path)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    assert config['model_type'] == 'kimi_k31'
    metadata = scan_local_hf_repo(
        args.checkpoint, scheme=scheme_named('mxfp4-pack-quantized'), num_bits=4)
    files = []
    for source in SOURCES:
        path = 'python/sglang/' + source
        # Freeze the audited commit, never unrelated uncommitted edits.
        text = subprocess.check_output(
            ['git', '-C', str(args.sglang_source), 'show', f'{REVISION}:{path}'],
            text=True)
        files.append({'path': 'sglang/' + path, 'text': text})
    processor = json.loads((args.checkpoint / 'preprocessor_config.json').read_text())
    print(write_snapshot(
        'moonshotai/Kimi-K3.1-Pro', config, metadata_record(metadata),
        forward_source={'source': 'sglang', 'revision': REVISION,
                        'paths': [f['path'] for f in files], 'files': files},
        processor={'preprocessor_config.json': processor},
        overlay_dir=str(args.overlay_dir) if args.overlay_dir else None))


if __name__ == '__main__':
    main()
