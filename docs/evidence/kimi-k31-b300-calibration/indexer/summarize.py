import json,dataclasses
from pathlib import Path
from backend.inference.benchmarks import validate_text
from backend.inference.op_calibration import CalibrationRegistry, merge_validated_rows, BenchmarkKey, physical_time_for
p=Path(__file__).resolve().parent
_,train=validate_text((p/'train.csv').read_text());_,held=validate_text((p/'holdout.csv').read_text())
registry=merge_validated_rows(CalibrationRegistry(),train,source=str(p/'train.csv'))
checks=[]
for t,h in zip(train,held):
    assert t['shape_json']==h['shape_json'] and t['input_seed']!=h['input_seed']
    assert t['cross_check']==h['cross_check']=='passed'
    key=BenchmarkKey.from_csv_row(t).to_physical()
    measured=physical_time_for(registry,key,current_peak_tflops_at=lambda _:4500.,current_peak_bw_gbps=8000.)
    assert measured is not None
    checks.append({'shape':json.loads(t['shape_json']),'training_us':float(t['latency_us']),
        'holdout_us':float(h['latency_us']),'holdout_relative_error':float(t['latency_us'])/float(h['latency_us'])-1,
        'overlay_seconds':measured.t_op_s,'input_seed_training':t['input_seed'],'input_seed_holdout':h['input_seed']})
report={'coverage':'6/6 exact physical keys consumable by merge_validated_rows + physical_time_for',
    'timing_method':'l2_drain CUDA events, standard driver; not CUPTI warm-resident serving',
    'deployment':{'tp_attention':8,'dcp':8,'ep_moe':2,'tp_moe_expert':4},
    'checks':checks,'mean_absolute_holdout_error':sum(abs(c['holdout_relative_error']) for c in checks)/len(checks),
    'scope':'heldout input seeds/runs at identical shapes, no unseen-shape or full-model validation',
    'source_commit_sglang':'1b2870d33de7b5f31844fd74e97eea6d7896c53a',
    'runtime':'DeepGEMM0.2.8, Torch2.14.1+cu130, GPU7 B300SM103; package source commit unreported by wheel',
    'shared_host':'Parent pytest on GPU0/CPU; only CUDA events used for latency',
    'setup_excluded':['FP8 quantization','cache packing','page metadata','topk','DCP communication']}
(p/'summary.json').write_text(json.dumps(report,indent=2))
(p/'calibration_rows.json').write_text(json.dumps(train,indent=2))
print(json.dumps(report,indent=2))
