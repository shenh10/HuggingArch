# kernel_bench — cross-library operator perf harness

Produces **measured** kernel timings that fill the calibration corpus the
inference estimate consumes. It is the PRODUCER; the schema/contract lives
in `backend/inference/benchmarks.py` (imported here, never forked), and the
measured rows are uploaded to Kernel Bench (`hbench upload`,
`/api/bench/upload`), where estimates look them up.

## The model

The reference for an op is the **measured envelope across many high-perf
libraries** (FA3, FlashInfer, FlashMLA, DeepGEMM, DeepInfer, FLA, vLLM,
sglang …), NOT
any single framework's dispatch choice. Each library that implements an
**op-signature** registers a `Provider`; the driver measures every
applicable provider at a shape; the consumer's envelope picks the fastest.

- **Op signature** (`signatures.py`) — the math contract (what to compute),
  the anchor. Providers register against it.
- **Provider** (`providers/`) — one library's kernel for a signature, with
  a capability predicate (`applies_to`) and an input adapter (`build`).
  Import-guarded: only providers whose library is installed self-register
  as available.
- **Numeric cross-check** (`refs.py` + `driver.py`) — one canonical-input +
  torch-eager reference per SIGNATURE (shared by all its providers). Each
  provider adapts the canonical inputs to its layout and returns a
  `BenchCase(run, to_canonical)`; the driver compares `to_canonical(run())`
  against the reference within a **dtype-scaled tolerance** (fp8 loose, bf16
  mid, fp32 tight) and refuses to record a kernel that disagrees — so
  "fastest" can't be won by a kernel computing the wrong thing.

## Libraries as isolated envs

Kernel libraries are **optional third-party extras, never core deps**, and
their versions conflict — so the unit of deployment is **one isolated venv
per library stack** (`envs/*.txt`). The thin harness is installed into each;
`run` there measures only that env's providers and stamps rows with the
**actually installed** versions. "That env's providers" is `run --library
<lib>` (which `run-all` passes): the providers registered under that library,
the same set its `envs/<lib>.txt` lists — not every provider importable there.
Every env carries torch and several carry flashinfer, so without it each env
re-measured those under its own release. torch is a library like the others:
its floors run in `envs/torch.txt` once, at whatever torch that env resolves:
the box profile's `constraints` pin where it has one (the H100 box's
`torch==2.8.0+cu128`), otherwise the newest build `torch>=2.5` allows (the
4090 box states no pin).

Libraries come in two provisioning modes:

**pip-resolvable** (torch, flash-attn v2, flashinfer, vllm) — each provider
declares its dep once, on the `Provider` (`import_name` + `pip`, e.g.
`pip="flash-attn>=2.6,<3"`). That single declaration is the source of truth:
`env.detect_env` probes exactly those import names, `available()` also checks
the installed version against the spec, and the `envs/*.txt` files are
**generated** from it — `python -m kernel_bench envs` (re)writes them,
`--check` fails CI if a provider's dep changed without regenerating. Never
hand-edit `envs/*.txt`.

**source-build** (FA3-hopper, FlashMLA, DeepGEMM, DeepInfer) — custom builds that aren't
reproducibly pip-installable (GPU-arch-specific, subdirectory builds) and that
you rebuild often, one built env hosting many ops. These are **provisioned out
of band** — a container / manually built venv you maintain — and kernel_bench
only *targets and measures* them, stamping the actually-installed version
(version is an identity axis, so a rebuilt env just accumulates new rows).
Their providers set `source_build=True`: excluded from `envs/*.txt`, never
uv-resolved. Point `run-all` at the interpreter via `--env <lib>=<python>` or
`KERNEL_BENCH_ENV_<LIB>`; unconfigured, they're skipped with a build hint.

Versions differ in perf, so version is a first-class identity axis
(`framework` = library, `framework_version` = its version, `kernel_impl` =
variant). Results only accumulate; the envelope naturally picks the fastest
across versions too.

## Usage

Four commands, nothing in your shell history:

```bash
# ⓪ up — clone → working bench: resolve the box profile, build every pip
#    env (uv resolves + caches it), self-check the torch providers with
#    smoke. Anything it cannot set up is said out loud, not remembered.
hbench up                                     # --dry-run to preview

# ① plan (CPU) — enumerate the jobs a real serving run hits
#    With a box profile (--box, $KERNEL_BENCH_BOX or envs/boxes/<host>.toml)
#    each job is judged against that box's envs on its own card: jobs no
#    kernel there would run go to jobs.no_provider, still counted by report.
hbench sweep deepseek-ai/DeepSeek-V3 -o jobs.jsonl
#    (or `hbench plan MODEL --batch 1,8,64 --seq 2048,8192` for a custom grid)

# ② run (GPU) — measure every provider env THIS PLAN NEEDS in one command:
#    the envs owning a candidate of some job (a plan judged on other
#    premises — another card, an --env, --constraints — is judged again
#    first). `run-all` uses
#    `uv` to resolve+cache each env's deps on the fly (no manual venvs),
#    takes the per-GPU exclusive lock, watches stalls, and writes
#    results/<lib>.csv row by row. Every row carries the batch's run_id.
hbench run-all --jobs jobs.jsonl -o results/ \
    --budget 2h --stall-timeout 15m           # --dry-run to preview

# ③ report — coverage against the corpus: measured cells and who wins each
#    (the envelope), plus the gap list (no provider / declined / pending).
hbench report --jobs jobs.jsonl --results results/

# ③b check — MFU / MBU / allclose on the measured CSVs. Same SOL gate as
#    write, applied after the fact; allclose is the recorded cross-check
#    verdict (the dtype's band, read on the error's RMS and its worst element).
hbench check results/

# …or drive one env yourself (what run-all does under the hood):
PYTHONPATH=. uv run --no-project --with-requirements kernel_bench/envs/fa3.txt \
    -- python -m kernel_bench run --jobs jobs.jsonl -o results/fa3.csv \
    --library fa3 --skip-existing

# ④ upload (optional, any box) — `check` first (MFU / MBU / allclose),
#    then send the CSVs. A file that fails check does not leave the box.
#    Auth = a personal access token minted in the web account page
#    (Access tokens); an admin later promotes a set into the shared corpus.
hbench upload results/ --token "$HUGGINGARCH_TOKEN"   # or --host https://your.deploy
```

Machine-specific execution facts belong in one TOML profile rather than shell
history. `run-all` reads `--box FILE`, `$KERNEL_BENCH_BOX`, or automatically
`kernel_bench/envs/boxes/<hostname>.toml`. The profile can set the uv Python,
constraints file, a command wrapper, source-build interpreters, and environment
variables (including `${PATH}` expansion):

```toml
name = "h100-ngc"
python = "3.11"
constraints = "/opt/kernel-bench/torch-constraints.txt"
command_prefix = ["/opt/kernel-bench/kbrun.sh", "--"]

[env]
OMP_NUM_THREADS = "8"
MKL_NUM_THREADS = "8"
OPENBLAS_NUM_THREADS = "8"
NUMEXPR_NUM_THREADS = "8"
CUDA_HOME = "/usr/local/cuda"
PATH = "/usr/local/cuda/bin:${PATH}"

[providers]
fa3 = "/opt/venvs/fa3/bin/python"
deepgemm = "/opt/venvs/deepgemm/bin/python"
```

`constraints` is the torch build for the envs that do not pin one themselves.
A box can also override one env's requirements file outright, and say what
that one env runs in:

```toml
[env_requirements]
sglang = "/opt/kernel-bench/h100.sglang.txt"   # replaces envs/sglang.txt …
                                               # … and `constraints` with it:
                                               # a file that states one env's
                                               # requirements states them in full

[library_env.sglang]                           # what THIS env runs in, where
LD_LIBRARY_PATH = "/opt/cuda-compat-13.0/compat"   # that differs from [env]
```

`[library_env]` exists for a box with more than one wheel stack. The H100
profile in this directory is the case: its kernel driver tops out at CUDA 12.9,
and the vLLM and sglang releases their bindings are audited at ship extensions
built against CUDA 13, so those three envs run against NVIDIA's
forward-compatibility *user-mode* driver while every other env runs against the
box's own. Box-wide that would re-time the CUDA-12 envs against a driver none
of their rows name; per library it stays a property of the env.

### A box that serves the queue

A box measuring a corpus runs two long-lived commands, and `work_dir` in
its profile is where they keep everything:

```toml
work_dir = "${KERNEL_BENCH_WORK_DIR}"   # jobs/, results/, queue.log, queue.pid
```

```bash
# plan models and hand each plan over as it lands, so the card measures
# the first while the rest are still being planned
hbench enqueue --all                 # or: --missing, or named model ids

# the loop: oldest plan with no `.done`, measured with every OTHER
# model's results as --measured-in, then the next one
hbench queue --measured-in exports/corpus.csv

# hold the card between models — the cluster reclaims a GPU idle for
# twenty minutes, and the queue dying is exactly when it needs covering
hbench keepalive
```

`hbench queue --list` prints what is waiting, `--dry-run` prints the
`run-all` it would issue, and `--once` measures a single model. SIGTERM
stops the loop and the run under it together.

The cross-model `--measured-in` is the reason this is a command and not a
shell loop: the physical key carries no model name, so a cell another
model already measured is this model's cell, and re-measuring it is GPU
time spent twice.

On a target GPU box you don't need a repo checkout — `pip install "huggingarch[bench]"`
gives the `hbench` CLI (the plan engine is torch-free; each provider library is
measured in its own isolated `uv` env, so the box brings its own CUDA-matched torch).

> pip-resolvable envs (fa2, flashinfer, vllm) run via `uv` with no setup.
> Source-build envs (fa3, flashmla, deepgemm) are externally provisioned — build
> them once, then point `run-all` at the interpreter:
>
> ```bash
> python -m kernel_bench run-all --jobs jobs.jsonl -o results/ \
>     --env fa3=/opt/venvs/fa3/bin/python --env deepgemm=/opt/venvs/dg/bin/python
> ```
>
> Unconfigured source-build envs are skipped with a build hint, never fatal.

Each `run` emits validated CSV rows. Name them `<opclass>_<lib>-<ver>_<gpu>.csv`
and upload them to Kernel Bench (`hbench upload`, `/api/bench/upload`).
Before opening the output file, `run` takes a lock keyed by the physical CUDA
UUID and asks `nvidia-smi` for every compute process already resident on that
card. A process from an older nohup/manual run is refused even though it never
knew about the lock; probe failure is printed as a warning rather than being
silently presented as an idle-card guarantee.
`run-all` supervises every provider environment as its own process group. The
wall-clock budget stops an env that exceeds its allocation; the watchdog uses
the flushed output CSV's mtime as progress and terminates a process group whose
file stops advancing. The child publishes its current job before allocation;
after a stall or process crash, `run-all` quarantines that job, restores the
identities already present in the CSV, and continues the remainder in a fresh
CUDA process. Rows already written remain valid partial output, and a resumed
process cannot append duplicate identities merely because its in-memory dedup
set started empty.

## Status

Op coverage (signatures + shape_in→identity projection + providers) — the
full dense-model compute mix:
- **dense attention** (gqa) — FA2/FA3 prefill + decode, FlashInfer ragged
  prefill; FlashMLA (mla) decode staged with the entrypoint + layout TODO.
- **GEMM** — torch/cuBLAS baseline + DeepGEMM (fp8 blockwise). Identity is
  `{B,Sq,K,N}`; N comes from `shape_out`, or (for the fused column-parallel
  linears whose fused node has no single `shape_out`) from the sum of the
  engine's propagated per-port `shape_outs` — real shapes, not a flops
  back-calculation.
- **norm / activation** — rmsnorm, fused_add_rmsnorm, silu_and_mul via vLLM
  custom ops.
- **rope** — rotate Q+K in one launch: torch baseline + flashinfer / vLLM
  (`_custom_ops.rotary_embedding`) / sglang
  (`apply_rope_with_cos_sin_cache_inplace`). Identity `{B,Sq,N,Nk,D}`.
- **fused_qk_norm_rope** — per-head RMSNorm(Q,K)+RoPE over one packed qkv:
  sglang (`sgl_kernel.fused_qk_norm_rope`) + vLLM (`_custom_ops`).
- **moe_gate** — noaux_tc grouped top-k router: torch baseline + vLLM
  (`_moe_C.grouped_topk`) / sglang (`moe_fused_gate`). Discrete ids →
  measured, not numerically gated.
- **indexer_logits** — DSA lightning-indexer logits: torch dense baseline
  (runs anywhere) + DeepGEMM `fp8_paged_mqa_logits` (Blackwell, source-build).

All identities are phase-aware (Sq). Signatures, registry, capability
predicates, per-op shape_in identity, plan/run CLI, emit, physical dedup +
measure-once skip are real and CPU-testable; GPU kernel bodies run in the
matching provider env (`available()` gates them off elsewhere).

Cross-check references are wired for gemm / rmsnorm / fused_add_rmsnorm /
silu_and_mul / gqa_attention / rope / fused_qk_norm_rope / indexer_logits.
Measured-only (no numeric gate, discrete/recurrent output): moe_gate,
linear-attention/SSM (FLA). Not yet covered (staged): embedding,
mla_attention reference + FlashMLA body.

### How an estimate reads the corpus

Both lookups match the **physical key** — `(op-signature, op-relevant shape,
dtype, gpu, phase)`; framework, kernel, model, spec, node and the deployment
(`parallel`) are provenance (`op_calibration.PhysicalKey`). The default
selection is the **envelope**: the fastest measured kernel of any library and
version (`prefer_measured=True`). The framework selection instead takes the
estimate's framework's own kernel at the latest release the corpus has it at
(one `kernel_impl`, or none is quoted), labelled with that release.

### Collectives

`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all` and the
all-reduce + RMSNorm seam are measured through NCCL over a group of
processes, one per card, on the same `run` path as everything else
(`comm.py`; ARCHITECTURE.md, "How a collective is measured"). A box states
its cards (`devices = 2` in its profile) and a run drives the ones it is
shown (`CUDA_VISIBLE_DEVICES=0,1`), which it then locks; a cell whose group
needs more is a gap the plan names. Not covered: groups larger than the box,
the routed MoE exchange (`routed_all_to_all`), and `send_recv`.
