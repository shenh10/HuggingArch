# Contributing to HuggingArch

Thanks for your interest! HuggingArch gets better in two ways: **broader
coverage** (more models) and **higher fidelity** (correct numbers, more
capabilities). The easiest way to add a model is the in-app agent generator —
you rarely need to hand-write YAML. The contributions that move the project
most are **bug fixes, measured benchmarks, and capability extensions**.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

| Contribution | How | Value |
|---|---|---|
| **A new model spec** | Generate it in the Inspector, then **Review → Open PR** from your account (recommended); or hand-write a YAML | ★★☆ |
| **Fix a wrong number or bug** | A spec correction, or an engine fix in `backend/` | ★★★ |
| **Measured kernel benchmarks** | Upload a CSV to Kernel Bench (`hbench upload`, or **My benchmarks**) | ★★★ |
| **Extend a capability** | A new op / component, or a new analysis in `backend/` + `web/` | ★★★ |
| **Verify a model / docs** | `supported_models.txt`, `README*.md`, `docs/` | ★☆☆ |

## Development setup

Requires Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/). **No Supabase
project is needed** — the multi-user layer stays dormant unless `SUPABASE_*` is
set.

```bash
git clone https://github.com/shenh10/HuggingArch.git
cd HuggingArch
uv sync

# Build the web UI (the API serves it from web/dist)
npm ci --prefix web && npm run build --prefix web

# Run the app (API + web UI) at http://localhost:8000
uv run python -m backend.api --host 0.0.0.0 --port 8000

# Code change: run the test suite before you open a PR
uv run pytest tests/
```

A PR that only changes `backend/arch/models/`, `backend/arch/snapshots/`,
and optional `backend/arch/components/drafts/` is a spec contribution.
CI runs `scripts/ci_spec_gate.py` on those — it validates the changed
snapshots — and `python -m tools.corpus_admit verify --changed`, which
measures the changed models against every corpus pin (the closed
per-model tables under `tests/golden/`, registered in `tools/corpus_pins`).

A PR that adds a snapshot is a new model, and a new model owes a key to
every corpus pin, so it runs the full suite. Put the keys in with the
admitter after adding the model's files:

```bash
python -m tools.corpus_admit <model_id> --dry-run   # report only
python -m tools.corpus_admit <model_id>             # write (Python 3.13, locked deps)
```

It writes only the new model's keys and lists what it cannot write: the
decisions a person owns (the model's weight-copy contracts, the reason for a
checkpoint residual) and any corpus check the model fails. How vLLM splits
each stack (D1) it reads from vLLM's own source: it fetches vLLM's files for
the model's architectures, and a stack whose every node vLLM assigns a class
that splits as measured is written with the vLLM commit and each node's
cited line. A node vLLM's source does not settle, or splits differently, is
listed for a person. Every lane runs `python -m tools.corpus_admit verify` (no
missing or orphan pin keys; README counts match the repository).

A change to `backend/arch/schema/` or the retired-field tables in
`backend/arch/spec_parsing.py` must re-load stored private specs:

```bash
PYTHONPATH=. ./.venv/bin/python tools/check_stored_specs.py
```

Default (no `--local-only`) is fail-closed: a DB read error, or any
non-rejected row that the current loader refuses, is a non-zero exit.
CI runs `--local-only` when those paths change.

`web/` is the Vite + React + TypeScript SPA behind every page; `/` and the
other page URLs serve `web/dist/index.html`. CI builds it through
`tests/frontend/test_web_build.py`, and the Docker image builds it in its
first stage.

`scripts/` is the repeatable ops entry (gate, dump, audit). `tools/` is
one-shot probes and migrations.

## Adding a model spec

**Recommended: generate it in the app.** Open the [Inspector](https://huggingarch.com/inspector.html),
enter a model id that isn't supported yet, and let the coding agent discover the
structure. On a hosted deployment you sign in and bring your own agent key
(BYOK), so the generation runs on *your* account and stays private to you until
you contribute it. When the spec looks right, open **My spec generations**,
run **Review**, and once it PASSes, **Open PR**. A maintainer merges after CI.

A PASS also runs the corpus admitter against main, and the record page lists
the corpus rows that are ready to admit and the decisions a maintainer still
owes; how vLLM splits each stack is read from vLLM's source and owed only
where that source does not settle it. **Open PR** runs it again against the main it opens on and puts those
rows in the PR. The PR is a draft while decisions are pending, and it does not
open while the model fails a corpus check or main's own baseline needs a
refresh. A maintainer writes the decisions on the PR branch; **Update PR**
carries them onto the latest main and writes the rows they unlock.

The agent isn't magic — it reads the model's `config.json`, tensor names, and
(when available) its `modeling_*.py`, then composes a spec from existing
components and validates it against the checkpoint. It gets most models right;
for genuinely novel structure a maintainer may refine before merging.

**Advanced: hand-write the YAML.** Architectures are declarative, so a spec is a
YAML, not Python. Add `backend/arch/models/<family>.yaml`, starting from
the closest existing model (the DSL is documented inline; see
[`docs/custom_models.md`](docs/custom_models.md) and [`docs/design.md`](docs/design.md)).
The validator does the hard part — it diffs your spec's parameter count **per
tensor** against the checkpoint's safetensors index:

```bash
python -m backend.inference.validate <hf-model-id>   # aim for valid: true, diff: 0
```

A residual diff is only acceptable if you document *why* (e.g. a tied embedding
the index counts once). Then sanity-check the per-layer breakdown in the
Inspector, and optionally mark it verified by adding the id under the relevant
`## <spec_name>` header in `backend/arch/supported_models.txt`.

## Fixing a wrong number or a bug

The highest-value contribution. Two flavours:

- **A spec is numerically wrong** — the parameter count, KV bytes, or attention
  type is off for a supported model. Reproduce with
  `python -m backend.inference.validate <model-id>` (it points at the tensor
  that disagrees), fix the model or component YAML, and confirm `diff: 0`.
- **The engine itself is wrong** — a FLOPs/memory/shape formula, a parallelism
  or KV-cache derivation, a fusion or calibration bug. These live in `backend/`
  (`arch/`, `inference/`, `rendering/`). Add or update a test that pins the
  correct behavior; a bug fix without a regression test tends to come back.

Include the model id, the config, and expected-vs-actual in the PR (or an issue
first, if you want to discuss the fix).

## Contributing measured benchmarks

The inference estimate is a theoretical roofline until real measurements
calibrate it. Contributing measured per-op kernel timings is one of the most
useful things you can do — and it's data, not code.

1. Export a shape template for a model/config from the **Inference** page, or run
   `huggingarch bench suggest-shapes <model-id> --framework vllm`.
2. Fill the measurement columns from a real run on your hardware, with the
   **same** framework (the op names are post-fusion, so the kernel library
   must match), and put the release you measured on in `framework_version` —
   measurements are versioned, and an estimate labels every number with the
   release it was measured on.
3. Validate and upload it to Kernel Bench:
   ```bash
   huggingarch bench validate <csv>
   hbench upload <csv>
   ```
   The upload runs the same validator. The schema is frozen in
   `backend/inference/BENCHMARKS_CSV.md`.

Uploaded rows are yours: they calibrate your own estimates (calibrate with my
rows) and nobody else's. To share them, contribute the upload from **My
benchmarks**; once an admin promotes it, its rows calibrate every estimate and
surface on the [Kernel Bench](https://huggingarch.com/kernel-corpus.html) page
as a cross-library envelope (fastest implementation per op/shape/GPU).

## Extending a capability

New analysis is welcome — a new primitive op with its FLOPs/params/shape
formulas (`backend/arch/ops.py`), a reusable component
(`backend/arch/components/library/*.yaml`), a fusion pattern, or a new view
in the Inspector / Inference pages. Prefer composing existing components over
adding a primitive; a new primitive changes the op registry and gets closer
review. Open an issue first for anything that touches the op registry or the
public analysis surface, so we can agree on the shape before you build it.

## Coding conventions

- **Lint gate.** CI runs `ruff check .` — the whole repository, `kernel_bench/`,
  `tools/`, `scripts/` and `tests/` included — with the bug-oriented families
  (`F`/`B`/`PLE`/`PLW`/`SIM`/`PERF`/`PIE`/`RUF`): it catches defects, not
  style. Run `uv run ruff check .` locally.
- **Core vs. service boundary.** The OSS analyzer + planner (everything except
  `backend/service/` and the account / admin / sign-in pages in `web/src/pages/`) must **never import the service
  layer**. `tests/meta/test_service_layout.py` enforces this — keep identity / auth /
  BYOK code inside `backend/service/`.
- **Migrations are append-only.** If you touch auth/admin, add a new numbered
  file under `supabase/migrations/` (e.g. `0020_*.sql`); never edit an existing
  one.
  `supabase/init.sql` is the frozen fresh-install baseline for new deployments;
  schema changes still land as numbered migrations and are folded into a new
  baseline only at a deliberate release checkpoint.
- **Match the surrounding code** — comment density, naming, and idiom.

## Pull requests

1. Fork, branch, and open a PR against `main`.
2. **CI must be green** — ruff, the full pytest suite, and the core/service
   boundary check. A maintainer merges after review; the project favours small,
   focused PRs.
3. Describe *what* changed and *why*. For a spec, paste the validator's
   `valid/diff` line; for a bug fix, note the regression test you added.

## Reporting bugs & requesting models

Use the [issue templates](https://github.com/shenh10/HuggingArch/issues/new/choose):
bug report, **new-model request**, or feature request. For a wrong *number*,
include the model id, the config, and what you expected vs. what you saw.

Security issues: please follow [SECURITY.md](SECURITY.md) instead of filing a
public issue.
