# B300 measured results

Raw runs: `results.csv` and per-workload JSONL. Two repeats where available; parenthesized ranges show the minimum–maximum across repeats. Throughput includes prefill and scheduling time. TPOT is (request latency − TTFT)/(output tokens − 1). For native SGLang streaming, each SSE gap is divided by the number of newly emitted tokens and repeated for those tokens; mean/p99 ITL are token-normalized, not raw burst intervals.

| Variant | Input/output | Concurrency | Runs | Output tok/s | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| ep1-breakable | 128/256 | 1 | 2 | 279.30 (279.22–279.37) | 69.22 (69.21–69.23) | 3.32 (3.32–3.32) |
| ep1-breakable | 2048/256 | 1 | 2 | 292.10 (292.10–292.11) | 173.00 (172.89–173.11) | 2.75 (2.75–2.75) |
| ep1-breakable | 2048/512 | 8 | 2 | 1062.72 (1058.26–1067.19) | 448.10 (430.57–465.63) | 6.28 (6.27–6.28) |
| ep1-breakable | 16384/256 | 1 | 2 | 166.85 (166.68–167.02) | 812.38 (811.49–813.27) | 2.82 (2.81–2.83) |
| ep1-breakable | 16384/256 | 8 | 2 | 261.15 (260.85–261.46) | 3738.34 (3737.17–3739.51) | 15.64 (15.63–15.65) |
| ep1-disabled | 128/256 | 1 | 2 | 202.54 (201.60–203.48) | 405.72 (399.32–412.12) | 3.36 (3.35–3.36) |
| ep1-disabled | 2048/256 | 1 | 2 | 230.81 (230.75–230.87) | 365.42 (365.00–365.84) | 2.91 (2.91–2.91) |
| ep1-disabled | 2048/512 | 8 | 2 | 977.35 (961.39–993.31) | 609.99 (570.31–649.67) | 6.64 (6.60–6.68) |
| ep1-disabled | 16384/256 | 1 | 2 | 164.41 (161.92–166.89) | 818.94 (815.05–822.83) | 2.88 (2.80–2.96) |
| ep1-disabled | 16384/256 | 8 | 2 | 260.51 (260.43–260.58) | 3737.82 (3737.78–3737.85) | 15.62 (15.61–15.64) |
| ep2-breakable | 128/256 | 1 | 2 | 288.90 (288.26–289.53) | 70.58 (68.33–72.83) | 3.19 (3.19–3.19) |
| ep2-breakable | 2048/256 | 1 | 2 | 296.86 (296.81–296.92) | 158.26 (158.08–158.44) | 2.75 (2.75–2.75) |
| ep2-breakable | 2048/512 | 8 | 2 | 1128.93 (1093.43–1164.42) | 489.62 (450.05–529.18) | 5.76 (5.63–5.89) |
| ep2-breakable | 16384/256 | 1 | 2 | 169.58 (169.25–169.91) | 789.07 (788.64–789.51) | 2.81 (2.80–2.82) |
| ep2-breakable | 16384/256 | 8 | 2 | 270.17 (269.61–270.74) | 3621.22 (3618.30–3624.15) | 15.04 (15.02–15.05) |
| ep2-disabled | 128/256 | 1 | 2 | 181.59 (156.99–206.19) | 526.93 (408.21–645.65) | 3.56 (3.26–3.85) |
| ep2-disabled | 2048/256 | 1 | 2 | 241.65 (241.47–241.82) | 371.10 (369.28–372.93) | 2.69 (2.69–2.69) |
| ep2-disabled | 2048/512 | 8 | 2 | 1008.36 (993.99–1022.72) | 617.84 (586.47–649.22) | 6.37 (6.31–6.43) |
| ep2-disabled | 16384/256 | 1 | 2 | 169.69 (169.66–169.72) | 790.88 (788.93–792.84) | 2.80 (2.79–2.81) |
| ep2-disabled | 16384/256 | 8 | 2 | 269.99 (269.16–270.83) | 3632.80 (3631.23–3634.37) | 15.05 (15.02–15.08) |

## Relative changes

Percent change in mean metric; negative TTFT/TPOT is faster. Small differences within the repeat range are not conclusive.

| Change | Input/output | Concurrency | Throughput % | TTFT % | TPOT % |
|---|---:|---:|---:|---:|---:|
| prefill graph off, EP2 | 128/256 | 1 | -37.14 | +646.61 | +11.55 |
| prefill graph off, EP2 | 2048/256 | 1 | -18.60 | +134.49 | -2.26 |
| prefill graph off, EP2 | 2048/512 | 8 | -10.68 | +26.19 | +10.58 |
| prefill graph off, EP2 | 16384/256 | 1 | +0.06 | +0.23 | -0.37 |
| prefill graph off, EP2 | 16384/256 | 8 | -0.07 | +0.32 | +0.11 |
| EP2 → EP1, graph on | 128/256 | 1 | -3.32 | -1.92 | +4.00 |
| EP2 → EP1, graph on | 2048/256 | 1 | -1.60 | +9.32 | -0.09 |
| EP2 → EP1, graph on | 2048/512 | 8 | -5.86 | -8.48 | +8.97 |
| EP2 → EP1, graph on | 16384/256 | 1 | -1.61 | +2.95 | +0.20 |
| EP2 → EP1, graph on | 16384/256 | 8 | -3.34 | +3.23 | +4.00 |
| EP2 → EP1, graph off | 128/256 | 1 | +11.54 | -23.00 | -5.63 |
| EP2 → EP1, graph off | 2048/256 | 1 | -4.48 | -1.53 | +8.10 |
| EP2 → EP1, graph off | 2048/512 | 8 | -3.07 | -1.27 | +4.26 |
| EP2 → EP1, graph off | 16384/256 | 1 | -3.11 | +3.55 | +2.94 |
| EP2 → EP1, graph off | 16384/256 | 8 | -3.51 | +2.89 | +3.79 |
| prefill graph off, EP1 | 128/256 | 1 | -27.48 | +486.12 | +1.21 |
| prefill graph off, EP1 | 2048/256 | 1 | -20.98 | +111.22 | +5.75 |
| prefill graph off, EP1 | 2048/512 | 8 | -8.03 | +36.13 | +5.80 |
| prefill graph off, EP1 | 16384/256 | 1 | -1.46 | +0.81 | +2.36 |
| prefill graph off, EP1 | 16384/256 | 8 | -0.25 | -0.01 | -0.09 |

## 128-token startup-drift confirmation

The canonical EP2-off first repeat is unusually slow; it remains above for transparency. Two additional measurements after the other workloads are below. Use these alongside the original repeat range; do not interpret the original EP1-versus-EP2 off comparison as a clean EP advantage.

| Variant / sample | Output tok/s | TTFT ms | TPOT ms |
|---|---:|---:|---:|
| EP2-off in128-out256-c1-r1 | 206.24 | 406.78 | 3.263 |
| EP2-off in128-out256-c1-r2 | 202.64 | 430.58 | 3.257 |

EP1-off versus EP2-off confirmation throughput: -0.93%. These extra runs are not included in the canonical 40-run matrix.
