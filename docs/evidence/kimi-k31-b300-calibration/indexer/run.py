import json,sys,dataclasses,time,subprocess
from pathlib import Path
import torch
from kernel_bench.plan import build_jobs
from kernel_bench.driver import run_jobs
from kernel_bench.env import detect_env
from kernel_bench.emit import rows_to_csv
from backend.arch.layout.parallel_config import ParallelConfig
from backend.inference.benchmarks import validate_text
out=Path(__file__).resolve().parent
mode=sys.argv[1]
assert mode in ('train','holdout')
# All shapes planned through production point_keys, never constructed by hand.
# Local contexts include both sides of 64-token page boundaries and serving midpoints.
lengths=[256,257,288,2048,2049,2080]
if (out/'jobs.json').exists():
    jobs=json.loads((out/'jobs.json').read_text())
else:
    jobs=[]
    for length in lengths:
        all_jobs,_=build_jobs(model_id='moonshotai/Kimi-K3.1-Pro',batch_size=1,
            seq_len=length*8-1,avg_output_tokens=2,phase='decode',framework='sglang',
            parallel=ParallelConfig(tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8),
            gpu='B300-SXM',kv_block_sizes=[64])
        candidates=[j for j in all_jobs if j['signature']=='indexer_logits']
        assert candidates, length
        for j in candidates:
            if j['physical_key'] not in [k['physical_key'] for k in jobs]: jobs.append(j)
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
# Driver's canonical training seed retained. Holdout changes only input RNG,
# not key, dimensions, timing method or parameters. No holdout rows enter overlay.
if mode=='holdout':
    import kernel_bench.refs as refs
    def independent_generator(inv,device):
        return torch.Generator(device=device).manual_seed(inv.input_seed ^ 2026111302)
    refs.input_generator=independent_generator
# driver imports input_generator locally from refs.
env=detect_env();env.library='deepgemm'
(out/(mode+'-env.json')).write_text(json.dumps(dataclasses.asdict(env),indent=2))
result=run_jobs(jobs,env=env,date='2026-11-13',contributor='B300 Kimi K3.1 independent kernel validation',
    run_id='kimi-k31-indexer-'+mode,box='B300 GPU7',cross_check=True,log=lambda s:print(s,flush=True))
if mode=='holdout':
    for row in result.rows:
        if row.get('input_seed'):
            row['input_seed']=str(int(row['input_seed']) ^ 2026111302)
csv=rows_to_csv(result.rows)
(out/(mode+'.csv')).write_text(csv)
header,rows=validate_text(csv)
errors=[]
(out/(mode+'-result.json')).write_text(json.dumps({'rows':result.rows,'validation_errors':errors,'failures':result.failures,'refused':result.refused},indent=2))
print('FINAL',len(rows),errors,result.failures,result.refused,flush=True)
