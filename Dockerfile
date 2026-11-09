# HuggingArch — production image
#
# Multi-stage build:
#   1. `builder`  — installs locked deps with uv (≈ 90s, cached by uv.lock)
#   2. `runtime`  — slim, copies in only the runtime venv + source
#
# Python 3.13, the version `requires-python` names and CI installs. uvicorn
# binds to 0.0.0.0 on $PORT (Fly sets PORT=8080 by default; we explicitly
# expose 8000 for any non-Fly host where the default PORT env isn't injected).

# React SPA (`web/`). Built first so the Python image can copy `web/dist`.
FROM node:20-bookworm-slim AS web
WORKDIR /src
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# uv is a single static binary — copy from its image, no apt-get / curl.
COPY --from=ghcr.io/astral-sh/uv:0.11.15 /uv /uvx /usr/local/bin/

# Install runtime deps first using only the lock files. This layer
# is cached as long as pyproject.toml + uv.lock don't change, so
# code-only edits don't re-resolve / re-download dependencies.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source + finalize. Explicit list so we don't accidentally
# bundle .venv / tests / docs (.dockerignore is the primary filter
# but COPY's path also caps the layer).
COPY backend ./backend
COPY web ./web
COPY --from=web /src/dist ./web/dist
RUN test -f /app/web/dist/index.html
COPY supabase ./supabase
COPY scripts ./scripts
COPY kernel_bench ./kernel_bench
COPY README.md LICENSE ./

# Install the project itself (no-op deps; we just want the entrypoint
# wired and the backend package importable).
RUN uv sync --frozen --no-dev

# Pre-compute canonical SOL laws and the Gallery's SOL entries into a path
# the runtime image ships — NOT under /app/.cache, which Fly's volume mounts
# over and would hide. Scaling request answers are runtime cache entries:
# HA_SOL_POOL_ONLY keeps the image to laws, and scripts/entrypoint.sh warms
# the responses once the lake serves. Skip
# with `--build-arg HA_SKIP_WARM=1` for a faster local image that still works
# (first request pays the sweep).
ARG HA_SKIP_WARM=0
ENV HA_GALLERY_CACHE_DIR=/app/var/gallery-cache
ENV HA_SOL_POOL_DIR=/app/var/sol-pool
ENV HA_SOL_POOL_ONLY=1
RUN mkdir -p /app/var/gallery-cache \
    && if [ "$HA_SKIP_WARM" = "1" ]; then \
         echo "[warm] skipped (HA_SKIP_WARM=1)"; \
       else \
         uv run python -m backend.warm_caches --gallery --sweeps; \
       fi


FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/usr/local/bin:/opt/agents/.local/bin:/opt/agents/.kimi-code/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    HUGGINGARCH_HF_META_CACHE_DIR=/app/.cache/huggingarch/hf_meta \
    # BYOK agent hardening: untrusted spec-gen runs drop to this user (distinct
    # from the root backend, so the agent can't read the parent's /proc environ)
    # with HOME here (writable for the CLI's config). Consumed by
    # backend/agents/base.py (_agent_user / _merge_env).
    HA_AGENT_USER=specagent \
    HA_AGENT_HOME=/home/specagent \
    # Per-user spec-gen overlays: absolute roots in agent-writable directories
    # of the /app/.cache volume (chowned to specagent at startup), NOT
    # ~/.huggingarch — the backend runs as root but the agent as specagent, so
    # a `~`-relative root would resolve to different, unwritable homes.
    # Consumed by backend/spec_worker/overlay.py.
    HA_SPEC_RUNS_ROOT=/app/.cache/ha_spec_runs \
    HA_USER_SPECS_ROOT=/app/.cache/ha_user_specs \
    # The builder may bake an image-local Gallery cache. Runtime release
    # state also lives on the persistent volume and is merged from the
    # versioned shared artifact by scripts/entrypoint.sh.
    HA_GALLERY_CACHE_DIR=/app/.cache/huggingarch/gallery-cache \
    HA_SOL_POOL_DIR=/app/var/sol-pool \
    HA_SOL_POOL_ONLY=0 \
    # Nothing is baked for the sweep answers (the builder's warm runs with
    # HA_SOL_POOL_ONLY), so they live on the volume with the trace bodies
    # their entries name; scripts/entrypoint.sh merges the published artifact.
    HA_SWEEP_CACHE_DIR=/app/.cache/huggingarch/sweep-cache \
    HA_CODE_TRACE_DIR=/app/.cache/huggingarch/code-traces \
    # The corpus admitter's per-(base, environment) baseline cache, on the
    # persistent volume so a restart does not re-measure the corpus.
    # Consumed by backend/service/admission.py.
    HA_ADMISSION_CACHE_DIR=/app/.cache/huggingarch/corpus_admit \
    # The Kernel Bench lake (docs/measured_calibration.md §4.1, §4.5): on the
    # volume, in a root-only (0700) directory the agent cannot read or change.
    # Spill (tmp/) and backup staging (staging/) live beside the catalog.
    BENCH_LAKE_CATALOG=/app/.cache/bench_lake/catalog.ducklake \
    BENCH_LAKE_DATA=/app/.cache/bench_lake/data \
    # Production must ship the Vite scaffold. Local/pytest leave this unset
    # so `import app` without a prior `npm run build` still works.
    HA_REQUIRE_WEB_DIST=1

WORKDIR /app

# Copy the pre-built venv + source from the builder stage. Nothing
# else is needed; runtime image stays slim (~150 MB compressed).
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
# The corpus admitter (run by the service after a PASS review and on Open PR)
# writes only when the installed runtime distributions equal what this
# image's uv.lock resolves to, and it reads that resolution with
# `uv export --frozen --no-dev` — the same function CI and a local run use
# (tools/corpus_pins/env.py). So the runtime stage keeps the builder's uv.
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# --- Agent CLIs (for admin/BYOK spec generation on the deployed service) ------
# Installed via each vendor's official native installer (self-contained
# binaries — no Node runtime). HOME=/opt/agents puts them in a SHARED,
# world-readable location (not /root, which `specagent` can't traverse).
# NOTE: verified on the first Fly build — pin flags / install dirs for
# codex+kimi and codex's release URL are confirmed there (claude honors `-s`).
ARG CLAUDE_VERSION=2.1.209
ARG CODEX_VERSION=0.139.0
ARG KIMI_VERSION=0.39.1
# Each agent installs BEST-EFFORT: the codex/kimi installers fetch release
# metadata from the GitHub API, which intermittently 403s the CI builder IP
# (unauthenticated rate limit) and used to fail the whole deploy. So: pass the
# pinned versions (lets an installer skip the "latest" API lookup), retry on
# transient failure, then degrade gracefully — agents power the OPTIONAL admin
# spec-gen feature and are resolved lazily at runtime (shutil.which), so a
# momentarily-unreachable installer must not block the image build / deploy.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && mkdir -p /opt/agents \
    && install_agent() { \
         name="$1"; cmd="$2"; attempt=1; \
         while [ "$attempt" -le 3 ]; do \
           if HOME=/opt/agents \
              CLAUDE_VERSION="${CLAUDE_VERSION}" CODEX_VERSION="${CODEX_VERSION}" \
              KIMI_VERSION="${KIMI_VERSION}" bash -c "$cmd"; then \
             echo "[agents] installed ${name}"; return 0; \
           fi; \
           echo "[agents] ${name} attempt ${attempt} failed; retrying in 15s"; \
           sleep 15; attempt=$((attempt + 1)); \
         done; \
         echo "[agents] WARNING: ${name} unavailable at build time — image ships without it (${name} spec-gen degraded until a later build)"; \
         return 0; \
       } \
    && install_agent claude "curl -fsSL https://claude.ai/install.sh | bash -s ${CLAUDE_VERSION}" \
    && install_agent codex  "curl -fsSL https://chatgpt.com/codex/install.sh | sh" \
    && install_agent kimi   "curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash" \
    && chmod -R a+rX /opt/agents \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Unprivileged user for untrusted agent runs. The backend stays root (so it
# can Popen(user=specagent)); the agent process itself runs as specagent with
# write access limited to the spec-output dirs + its home + logs + its cache
# directories on the volume (chowned at startup, below).
RUN useradd --create-home --home-dir /home/specagent --shell /usr/sbin/nologin specagent \
    && mkdir -p /app/.cache /app/logs \
    && chown -R specagent /home/specagent \
        /app/logs \
        /app/backend/arch/models \
        /app/backend/arch/components/library \
        /app/backend/arch/snapshots

# The commit this image was built from (`deploy.yml` passes
# `--build-arg HA_GIT_SHA=<sha>`). The image carries no `.git`, so this is the
# only way the service can name its code revision
# (`spec_worker.worker._code_revision`); a build without the arg leaves it
# empty and the service reports `unknown`.
ARG HA_GIT_SHA=""
ENV HA_GIT_SHA=$HA_GIT_SHA

EXPOSE 8000

# Two processes, one container (docs/measured_calibration.md §8.1): the HA
# API on $PORT, and the Kernel Bench service on 127.0.0.1:8100 — the one
# process that opens the lake. scripts/entrypoint.sh prepares the volume's
# directories as root (the lake directory root-only), writes a new shared
# secret (0600) for this start, starts both, and exits when either exits
# (Fly restarts the machine).
CMD ["bash", "/app/scripts/entrypoint.sh"]
