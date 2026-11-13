import json,sys,dataclasses
from pathlib import Path
import torch
from kernel_bench import driver
from kernel_bench.env import detect_env
from kernel_bench.emit import rows_to_csv
from backend.inference.benchmarks import validate_text
p=Path(__file__).resolve().parent;mode=sys.argv[1]
assert mode in ('train','holdout')
all_jobs=json.loads((p/'all-jobs.json').read_text());jobs=[]
for j in all_jobs:
 if any(name in j['node_spec_id'] for name in ('self_attn.q_b_proj','self_attn.q_absorbed_nope','self_attn.v_absorbed')):
  if j['physical_key'] not in [k['physical_key'] for k in jobs]:jobs.append(j)
assert len(jobs)==3
(p/'jobs.json').write_text(json.dumps(jobs,indent=2))
original=driver.applicable
allowed={'torch_cublas_linear','torch_sglang_mla_absorb_q_bmm','torch_sglang_mla_absorb_v_bmm'}
def exact_providers(*args,**kwargs):
 return [x for x in original(*args,**kwargs) if x.kernel_impl in allowed]
driver.applicable=exact_providers
if mode=='holdout':
 import kernel_bench.refs as refs
 refs.input_generator=lambda inv,device:torch.Generator(device=device).manual_seed(inv.input_seed^2026111303)
env=detect_env();env.library='torch'
(p/(mode+'-env.json')).write_text(json.dumps(dataclasses.asdict(env),indent=2))
r=driver.run_jobs(jobs,env=env,date='2026-11-13',contributor='Kimi K3.1 B300 exact BF16 Q and absorbed BMM',run_id='kimi-k31-gemm-'+mode,box='B300 GPU7',log=lambda s:print(s,flush=True))
if mode=='holdout':
 for row in r.rows:
  if row.get('input_seed'):row['input_seed']=str(int(row['input_seed'])^2026111303)
csv=rows_to_csv(r.rows);validate_text(csv)
(p/(mode+'.csv')).write_text(csv)
(p/(mode+'-result.json')).write_text(json.dumps(dataclasses.asdict(r),indent=2))
print('FINAL',len(r.rows),r.failures,r.refused,flush=True)
