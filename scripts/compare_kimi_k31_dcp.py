"""Compare target-only B300 serving samples with explicit sparse DCP scenarios."""
import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--theory-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.measurements.read_text())
    rows = []
    for size in (2048, 16384):
        runs = [r for r in measured['runs'] if r['input'] == size]
        if len(runs) != 3 or any(r['errors'] or r['cached_tokens'] for r in runs):
            raise ValueError('comparison requires three successful uncached repeats per context')
        theory = json.loads((args.theory_dir / f'k31-dcp8-{size}-rho1-f0.summary.json').read_text())
        if theory['parallel']['dcp'] != 8:
            raise ValueError('comparison requires DCP8 theory')
        tpot = mean(r['mean_tpot_ms'] for r in runs)
        sol = 1000 * theory['phases']['decode']['t_sol_s']
        rows.append({
            'input_tokens': size, 'output_tokens': 512, 'concurrency': 1,
            'context_midpoint': theory['phases']['decode']['kv_cache_length'],
            'requests': sum(r['completed'] for r in runs),
            'theory_ms_per_target_step': sol,
            'mean_http_tpot_ms': tpot,
            'repeat_range_http_tpot_ms': [min(r['mean_tpot_ms'] for r in runs), max(r['mean_tpot_ms'] for r in runs)],
            'theory_minus_http_percent': 100 * (sol / tpot - 1),
            'mean_http_ttft_ms': mean(r['mean_ttft_ms'] for r in runs),
        })
    scenarios = []
    for rho, failed in ((1, 0), (8, 0), (1, 8), (8, 8)):
        theory = json.loads((args.theory_dir / f'k31-dcp8-16384-rho{rho}-f{failed}.summary.json').read_text())
        phase = theory['phases']['decode']
        scenarios.append({'input_tokens': 16384, 'rho': rho, 'failed_owners': failed,
                          'target_step_ms': phase['t_sol_s'] * 1000,
                          'flops_per_step': phase['flops_per_step'],
                          'read_bytes_per_step': phase['mem_read_bytes'],
                          'communication_ms': phase['t_comm_s'] * 1000})
    result = {
        'scope': 'Analytical target-step lower bound versus target-only HTTP serving. No fitted multiplier; no DFLASH acceptance conversion.',
        'measurements': str(args.measurements),
        'baseline_assumptions': {'dcp_selected_load_factor': 1, 'dcp_failed_owners': 0, 'ep_load_factor': 1},
        'validation': rows,
        'context_delta_http_tpot_ms': rows[1]['mean_http_tpot_ms'] - rows[0]['mean_http_tpot_ms'],
        'context_delta_theory_ms': rows[1]['theory_ms_per_target_step'] - rows[0]['theory_ms_per_target_step'],
        'scenarios_not_observed_distributions': scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
