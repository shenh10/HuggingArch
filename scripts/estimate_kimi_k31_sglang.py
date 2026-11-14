"""Reproduce a TP8 K3.1 SGLang analytical estimate on B300.

This is a theoretical target-only estimate, not the DFLASH recipe's
measured throughput. See docs/kimi-k31-sglang.md for supported boundaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.arch.snapshots import load_snapshot
from backend.inference.capacity import estimate_capacity
from backend.inference.calibration_overlay import _law_to_json
from backend.inference.parallel_config import ParallelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ep', type=int, choices=(1, 2), default=2)
    parser.add_argument('--dcp', type=int, choices=(1, 8), default=1)
    parser.add_argument('--dcp-load-factor', type=float, default=1)
    parser.add_argument('--dcp-failed-owners', type=int, default=0)
    parser.add_argument('--input-tokens', type=int, default=16384)
    parser.add_argument('--output-tokens', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--calibration-rows', type=Path,
                        help='Validated Kernel Bench JSON rows; unmatched operations retain theory.')
    args = parser.parse_args()
    model = 'moonshotai/Kimi-K3.1-Pro'
    snapshot = load_snapshot(model)
    result = estimate_capacity(
        config=snapshot['config'], model_id=model, gpu_sku='B300',
        framework='sglang', num_gpus=8, gpu_memory_fraction=0.85,
        selection_mode='envelope' if args.calibration_rows else 'theoretical_only',
        calibration_rows=json.loads(args.calibration_rows.read_text()) if args.calibration_rows else None,
        parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                tp_moe_expert=8 // args.ep, ep_moe=args.ep, dcp=args.dcp),
        dcp_selected_load_factor=args.dcp_load_factor,
        dcp_failed_owners=args.dcp_failed_owners,
        avg_input_tokens=args.input_tokens,
        avg_output_tokens=args.output_tokens,
        prefill_batch_size=args.batch, decode_batch_size=args.batch,
        prefill_chunk_size=16384,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_law_to_json(result), indent=2, default=str) + '\n')
    summary = {
        'scope': f'Theoretical target-only absorbed MLA, TP8/DCP{args.dcp}; not measured DFLASH serving.',
        'dcp_selected_load_factor': args.dcp_load_factor,
        'dcp_failed_owners': args.dcp_failed_owners,
        'calibration_rows': str(args.calibration_rows) if args.calibration_rows else None,
        'selection_mode': 'envelope' if args.calibration_rows else 'theoretical_only',
        'model_id': model,
        'gpu': result['gpu']['sku'],
        'parallel': result['parallel'],
        'input_tokens': args.input_tokens,
        'output_tokens': args.output_tokens,
        'batch': args.batch,
        'memory': result['memory'],
        'kv_cache': {key: result['kv_cache'][key] for key in (
            'num_layers', 'per_token_bytes', 'constant_bytes',
            'per_request_kv_bytes', 'attention_variant')},
        'phases': {phase: {key: value for key, value in values.items()
                           if not isinstance(value, (dict, list))}
                   for phase, values in result['phases'].items()},
        'assumptions': result['assumptions'],
    }
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(_law_to_json(summary), indent=2, default=str) + '\n')
    print(args.output)
    print(summary_path)


if __name__ == '__main__':
    main()
