import json
from pathlib import Path
from kernel_bench.plan import build_jobs
from backend.arch.layout.parallel_config import ParallelConfig
jobs,dropped=build_jobs(model_id='moonshotai/Kimi-K3.1-Pro',batch_size=1,seq_len=2048,avg_output_tokens=512,phase='decode',framework='sglang',parallel=ParallelConfig(tp_attention=8,tp_dense_ffn=8,tp_moe_expert=4,ep_moe=2,dcp=8),gpu='B300',kv_block_sizes=[64])
p=Path(__file__).resolve().parent
(p/'all-jobs.json').write_text(json.dumps(jobs,indent=2))
for j in jobs:
 if j['signature'] in ('gemm','per_head_gemm','grouped_gemm','fused_moe'):
  print(j['op'],j['node_spec_id'],j['signature'],j['dims'],j['dtype_env'],flush=True)
