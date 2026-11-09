# Deployment — Fly.io + GitHub Actions + custom domain

This walks you through a one-time setup, after which a release ships to
`www.huggingarch.com` with one manual workflow run.

## Architecture

```
GitHub Actions → "deploy" workflow (run manually: workflow_dispatch)
  ├─ smoke job      ← this commit's `tests` run passed (waits for it), then
  │                    ruff + a few-minute smoke subset (web build, app boot,
  │                    core endpoints); the full suite is not run twice
  └─ deploy job     ← needs: smoke
        │
        ▼
flyctl deploy --remote-only --strategy rolling --build-arg HA_GIT_SHA=<sha>
        │
        ▼
Fly builders compile Dockerfile
        │
        ▼
Fly machine in iad region
        │  (HTTPS via auto-issued Let's Encrypt cert)
        ▼
www.huggingarch.com
```

Automatic deploy on every push to `main` is switched off for now; to turn it
back on, restore `push: branches: [main]` under `on:` in
`.github/workflows/deploy.yml`.

The Fly app reads runtime config from `fly.toml` + secrets injected
via `flyctl secrets set`. Deploys are zero-downtime rolling deploys. Schema
changes still need a manual `psql ... -f supabase/migrations/NNNN_*.sql` (we
don't auto-apply migrations on deploy — too easy to ship a destructive one by
accident).

### What the image contains

The `Dockerfile` builds in stages: the web app (`npm ci && npm run build` →
`web/dist`), then the Python environment, then a SOL warm-up
(`python -m backend.warm_caches`, skippable with `--build-arg HA_SKIP_WARM=1`)
that bakes the shared SOL record pool plus the Gallery and Scaling theory
results into the image, outside the volume-mounted cache. The shared pool's
canonical serving walks are produced once per model and card class; Gallery
and Scaling project them rather than republishing parallel walk caches. The
runtime fill joins Kernel Bench measurements but does not re-derive those
theory halves.
The runtime image installs the coding-agent CLIs (Claude
Code, Codex, Kimi) best-effort — a failed install never blocks a build — and
runs the agents as an unprivileged `specagent` user. `HA_GIT_SHA` is passed in
at deploy so the service can report which code it runs (the image has no
`.git`).

`fly.toml` sets the VM (`shared-cpu-1x`, 2 GB), a 3 GB volume
`huggingarch_cache` mounted at `/app/.cache`, rate limits (`ANALYZE_RPM`,
`INFERENCE_RPM`), `SPEC_AGENT_MAX_CONCURRENT`, and the result-cache directories
`HA_GALLERY_CACHE_DIR` / `HA_SWEEP_CACHE_DIR` with the code-trace bodies their
entries name (`HA_CODE_TRACE_DIR`) — on the volume, so they survive a restart.
Nothing is baked into the image for them; at start `scripts/entrypoint.sh`
merges each engine version's published artifact into them
(`backend.release_artifacts`; publish with
`python -m backend.{gallery,sweep}_cache_artifact publish --cache-dir DIR`).
A directory that holds files baked into the image must stay outside the
volume instead, or the mount hides them; `tests/meta/test_volume_dirs.py`
checks every directory a start-time pull writes is under the mount.
Each cache directory is also byte-bounded by one knob
(`HA_GALLERY_CACHE_BYTES` for the gallery's, `HA_SWEEP_CACHE_BYTES` for the
sweeps'; the tiers under a directory split it, oldest entries evicted
first), so a full warm — or the orphaned keys a code edit leaves behind —
cannot fill the volume the lake and the sol pool also live on.
The sol pool's serving-walk records are additionally pinned per process
(`HA_SOL_POOL_PIN_BYTES`): a pinned record is parsed once per process and
survives the pool's hot LRU, bounded by that budget — the machine pays it
once per process that reads walks.

### Continuous integration

| Workflow | Runs on | Does |
|---|---|---|
| `test.yml` | every push to `main`, every PR | A diff that only adds model YAML / snapshots / drafts runs `scripts/ci_spec_gate.py` (validate the changed snapshots); anything else runs ruff, pytest and the core/service boundary check. Both run `tools.corpus_admit verify` against the corpus pins. |
| `regression.yml` | manual | The `regression` tests the default runs exclude (`-m regression`, up to 240 min): the per-model and minutes-long tests, and every parametrised test's non-first cases. |
| `deploy.yml` | manual | Re-runs the suite, then deploys. |
| `update-transformers.yml` | weekly + manual | Opens a PR bumping `transformers`. |

## One-time setup

### 1. Install flyctl

```bash
curl -L https://fly.io/install.sh | sh   # macOS / Linux
# or: brew install flyctl
flyctl auth signup                        # opens browser; pick free plan
```

### 2. Launch the app (only the FIRST time)

```bash
cd /path/to/HuggingArch
flyctl launch --no-deploy
```

When prompted:
- App name: `huggingarch` (or whatever; `fly.toml` has it pre-set)
- Region: `iad` (close to HuggingFace servers; lowest /analyze latency)
- Postgres: **No** (we use Supabase, not Fly Postgres)
- Redis: **No**
- Deploy now: **No** (we'll set secrets first)

Verify `fly.toml` was updated with your app slug.

### 3. Set Supabase secrets

These never enter the Docker image — Fly injects them as env vars at
runtime.

```bash
flyctl secrets set \
  SUPABASE_URL="https://<your-project>.supabase.co" \
  SUPABASE_ANON_KEY="sb_publishable_..." \
  SUPABASE_SERVICE_ROLE_KEY="sb_secret_..."
# only if your project still signs tokens with the legacy HS256 secret:
flyctl secrets set SUPABASE_JWT_SECRET="<legacy-jwt-secret-from-supabase-studio>"
```

Optional secrets, each enabling one feature:

| Secret | Enables |
|---|---|
| `HA_CREDS_ENC_KEY` | users storing their own agent API keys (bring-your-own-key spec generation); without it BYOK is off |
| `HA_GITHUB_APP_ID`, `HA_GITHUB_APP_PRIVATE_KEY` (+ `HA_GITHUB_APP_INSTALLATION_ID`) | opening spec pull requests through a GitHub App |
| `SPEC_AGENT_CLI` | the host's default coding agent (`claude`, `codex`, `kimi`) |

Verify:

```bash
flyctl secrets list
# (values are hidden; only names + last-changed timestamps shown)
```

Initialize the project's SQL schema before the first backend deploy:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/init.sql
```

`init.sql` is the complete fresh-install baseline (migrations 0001–0041 and
the restarted 0000–0026 are folded into it; the next migration is 0027). A
new project runs `init.sql` and then every file currently under
`supabase/migrations/`, in order.

### 4. First deploy

```bash
flyctl deploy --remote-only
```

This builds the Docker image on Fly's remote builders (your laptop
doesn't need Docker), pushes the image to Fly's registry, and starts
one VM in iad. Output ends with the live URL: `https://huggingarch.fly.dev`.

Smoke test:

```bash
curl https://huggingarch.fly.dev/health
# {"status":"ok","version":"1.0.0","spec_jobs_running":0}

curl https://huggingarch.fly.dev/site-config.js
# window.SITE_CONFIG = { supabase: {...}, authEnabled: true };
```

### 5. Custom domain — www.huggingarch.com

Add the cert via flyctl, then point your DNS at the Fly IPs.

```bash
# Step 5a — issue cert (Fly auto-renews via LetsEncrypt)
flyctl certs add www.huggingarch.com
flyctl certs add huggingarch.com   # apex, optional

# Step 5b — discover the IPs Fly gave you
flyctl ips list
#  v4 (anycast)
#  v6 (anycast)

# Step 5c — at your DNS registrar (porkbun / namecheap / cloudflare),
#           add three records pointing to those IPs:
#     A      www      <v4 anycast>
#     AAAA   www      <v6 anycast>
#     A      @        <v4 anycast>     (apex; optional)
#     AAAA   @        <v6 anycast>     (apex; optional)

# Step 5d — wait a few minutes for DNS to propagate, then verify
flyctl certs check www.huggingarch.com
# Status: Issued
```

If you bought your domain from Cloudflare and want their CDN: turn the
proxy "off" (gray cloud) on the A/AAAA records — Fly's anycast already
handles routing, and Cloudflare's proxy + Fly's anycast double-hop
hurts latency.

### 6. GitHub Actions deploy hook

```bash
flyctl auth token         # prints a long-lived token; copy it
```

In the repo on github.com:
- Settings → Secrets and variables → Actions → **New repository secret**
- Name: `FLY_API_TOKEN`
- Value: the token from above

Now running the **deploy** workflow (Actions → deploy → Run workflow) does:
1. Re-runs the test suite (stops the deploy on failure)
2. On success, runs `flyctl deploy --remote-only --strategy rolling`

A rolling deploy means the new VM is started alongside the old, health
check passes, *then* the old gets retired. Zero-downtime.

## Day-2 operations

### View logs

```bash
flyctl logs                  # tail-follow stream
flyctl logs --since 1h       # last hour
```

### SSH into the running container

```bash
flyctl ssh console
# inside: ls /app, cat /app/.cache/huggingarch/hf_meta/*.json, etc.
```

### Roll back to a previous release

```bash
flyctl releases               # list with version numbers + sha
flyctl deploy --image registry.fly.io/huggingarch:<digest>
```

### Bump VM size

Edit `fly.toml`'s `[[vm]]` block, commit, push. Or instantly:

```bash
flyctl scale vm shared-cpu-2x --memory 2048
```

### Add a region

```bash
flyctl regions add nrt        # Tokyo — better for APAC users
flyctl scale count 2          # one VM per region; auto-balanced
```

### Apply a Supabase migration

Migrations are intentionally NOT auto-applied on deploy. Apply
manually after PR merge:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f supabase/migrations/NNNN_xxxx.sql
```

Or paste into Supabase Studio → SQL Editor. Apply a migration before (or
with) the backend that needs it. Some carry a staged rollout — e.g. the
engine-blocked spec records (former migration 0002, now in `init.sql`): deploy, then set
`HA_RECORD_BLOCKED_SPECS=1` once no machine runs the old backend; the steps
are in `supabase/README.md`. Spec-job lock `host_id` is a machine identity
(`FLY_MACHINE_ID` on Fly, or set `HA_MACHINE_ID`), not the container hostname.

### Warm the result caches

The Gallery and Scaling-panel results are cached (`backend/measure/cache.py`:
memory, disk, and optionally Supabase). Warm them outside the server
process, never inside it:

The shared SOL pool (`HA_SOL_POOL_DIR`) is intentionally not one JSON file
per result. Each immutable identity has a small manifest and index; its records
are appended to seekable JSON Lines shards, and the index records each
record's shard, byte range and digest. A reader seeks the records it needs
and promotes only those into a small hot LRU, so a large model's laws can
live on disk without parsing the whole bundle into memory.

The pool is portable across machines for one engine/model-data version: its
records contain no machine identity or measured state. A locally warmed
`HA_SOL_POOL_DIR` can therefore be copied to another host (or baked as an
artifact) only when the deployed source and engine data are the same; there is
no automatic Supabase synchronization for this pool. The request-shaped result
caches below do synchronize when Supabase is configured, but they are not the
canonical law pool.

Gallery and Scaling project the canonical serving laws in-process. The
projection has no durable directory or remote tier beside the request caches;
the compiled law program is memoized by its graph for the process's lifetime.

```bash
flyctl ssh console -C "python -m backend.warm_caches"             # Gallery + Scaling
flyctl ssh console -C "python -m backend.warm_caches --gallery"   # Gallery only
flyctl ssh console -C "python -m backend.warm_caches --sweeps"    # Scaling only
flyctl ssh console -C "python -m backend.warm_caches --landing"   # the landing view only
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `flyctl certs check` stuck "Pending" >5 min | DNS not propagated | `dig www.huggingarch.com` — should resolve to Fly IPs |
| 502 Bad Gateway from app | uvicorn crashed at boot | `flyctl logs` — usually missing required env var |
| "/site-config.js" returns `authEnabled:false` in prod | Forgot `flyctl secrets set` | Re-set the SUPABASE_* secrets |
| Deploy fails with "no FLY_API_TOKEN" | GitHub secret missing | Re-add via repo Settings (above) |
| VM keeps OOM-killing | Many concurrent spec jobs | Lower `SPEC_AGENT_MAX_CONCURRENT`, or raise `[[vm]] memory` |
| Gallery slow right after deploy | Image built with `HA_SKIP_WARM=1`, or a cache dir moved onto the volume | Rebuild with warm-up on; keep `HA_*_CACHE_DIR` under `/app/var` |

## What still needs manual work

These are intentionally NOT automated:
- **Supabase migrations** — admin reviews + applies via psql / Studio
- **Domain DNS records** — owned at your registrar, not in Fly
- **Bootstrapping first admin** — run `scripts/bootstrap_admin.py` once
  after the first user signs up (via SSH or local with prod env)

Anything else (code, config, dependencies, CI) is git → push → run deploy.
