#!/usr/bin/env bash
# The image's entrypoint (docs/measured_calibration.md §8.1): the HA API
# and the Kernel Bench service in one container, fail-fast.
#
#  1. As root, before anything serves: the volume's top stays root-owned;
#     only the agent's directories (HF cache, hf_meta, spec runs, user specs,
#     corpus admission) are chowned to specagent; the Kernel Bench lake
#     directory is root-only (0700). Non-fatal in dev where the volume is
#     absent.
#  2. A new shared secret for this container start, in a 0600 file both
#     processes read (HUGGINGARCH_BENCH_SECRET_FILE); the API, its compute
#     workers and the cache warm reach the service at HUGGINGARCH_BENCH_URL.
#  3. Start the service (127.0.0.1:8100, one process: it owns the lake), wait
#     — bounded, BENCH_START_WAIT_S, default 60 s — until it answers /health,
#     then start the API (0.0.0.0:$PORT), so the API's first request finds
#     the service's socket. When either exits, stop the other, reap it and
#     exit with the first one's status: Fly restarts the machine, and the
#     service's gate restores its state at start. SIGTERM / SIGINT are passed
#     to both. No in-container restarts.
#  4. Once the lake serves (bounded, BENCH_SERVING_WAIT_S, default 1800 s),
#     the cache warm runs once at low priority, its own process, outside the
#     API (`python -m backend.warm_caches $HA_WARM_ARGS`; by default every
#     Gallery GPU and every default Scaling request). The image carries
#     canonical laws and the Gallery's SOL views; this materializes Scaling
#     answers and folds measured tracks under the lake's real scope.
#     The warm is not one of the two processes: its exit stops nothing.
#     HA_RUNTIME_WARM=off skips it.
set -uo pipefail

cache="${HA_CACHE_ROOT:-/app/.cache}"
# The canonical SOL law pool is immutable and content-addressed.  Pull a
# complete engine-version artifact onto the persistent volume before any
# process can warm or serve; an absent artifact is not an error, and the
# runtime warm builds the pool instead. A configured artifact that exists but
# cannot be downloaded or verified is an error: silently serving from cold is
# exactly the regression this bootstrap exists to prevent.
python -m backend.sol_pool_artifact pull || exit $?
python -m backend.gallery_cache_artifact pull || exit $?
python -m backend.sweep_cache_artifact pull || exit $?
chown root:root "$cache" && chmod 0755 "$cache"
for d in "${HF_HOME:-}" "${HUGGINGARCH_HF_META_CACHE_DIR:-}" "${HA_SPEC_RUNS_ROOT:-}" \
         "${HA_USER_SPECS_ROOT:-}" "${HA_ADMISSION_CACHE_DIR:-}"; do
  [ -n "$d" ] && mkdir -p "$d" && chown -R specagent "$d"
done
mkdir -p "$cache/bench_lake" && chown root:root "$cache/bench_lake" \
  && chmod 0700 "$cache/bench_lake"

run_dir="${HA_RUN_DIR:-/run/huggingarch}"
mkdir -p "$run_dir" && chmod 0700 "$run_dir"
(umask 077 && python -c 'import secrets; print(secrets.token_hex(32))' > "$run_dir/bench_secret")
export HUGGINGARCH_BENCH_SECRET_FILE="$run_dir/bench_secret"
export HUGGINGARCH_BENCH_PORT="${HUGGINGARCH_BENCH_PORT:-8100}"
export HUGGINGARCH_BENCH_URL="http://127.0.0.1:$HUGGINGARCH_BENCH_PORT"

# The service's /health: `up` when it answers, `serving` when its lake serves.
bench_health() {
  python - "$1" <<'PY'
import json, os, sys, urllib.request
req = urllib.request.Request(os.environ["HUGGINGARCH_BENCH_URL"] + "/health",
                             headers={"X-HA-Internal": open(os.environ["HUGGINGARCH_BENCH_SECRET_FILE"]).read().strip()})
# Never through an environment proxy: the secret stays on loopback.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(req, timeout=2) as r:
        body = json.load(r)
except Exception:
    sys.exit(1)
sys.exit(0 if sys.argv[1] == "up" or (body.get("lake") or {}).get("state") == "serving" else 1)
PY
}

wait_for() {  # $1 = up | serving, $2 = seconds; 0 when reached in time
  local until=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$until" ]; do
    kill -0 "$bench" 2>/dev/null || return 1
    bench_health "$1" && return 0
    sleep 1
  done
  return 1
}

python -m backend.service.bench.server --port "$HUGGINGARCH_BENCH_PORT" &
bench=$!
wait_for up "${BENCH_START_WAIT_S:-60}" \
  || echo "entrypoint: the Kernel Bench service is not answering yet; starting the API" >&2

python -m backend.api --host 0.0.0.0 --port "${PORT:-8000}" &
api=$!

warm_when_serving() {
  if [ "${HA_RUNTIME_WARM:-on}" = off ]; then
    return 0                     # the memory budget leaves no room for it
  fi
  if wait_for serving "${BENCH_SERVING_WAIT_S:-1800}"; then
    if [ "${HA_LANDING_WARM:-on}" != off ]; then
      nice -n 10 python -m backend.warm_caches --landing
    fi
    # shellcheck disable=SC2086
    exec nice -n 10 python -m backend.warm_caches ${HA_WARM_ARGS:-}
  fi
  echo "entrypoint: the lake did not serve in time; no cache warm" >&2
}
warm_when_serving < /dev/null &
warm=$!

stop() { kill -TERM "$bench" "$api" "$warm" 2>/dev/null; }
trap stop TERM INT

wait -n "$bench" "$api"
rc=$?
echo "entrypoint: a process exited ($rc); stopping the other" >&2
stop
wait "$bench" "$api" 2>/dev/null
exit "$rc"
