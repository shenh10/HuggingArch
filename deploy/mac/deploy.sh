#!/usr/bin/env bash
# Build and roll the Mac deployment at the checkout's HEAD, then run the
# post-deploy content smoke against the local port. The same script runs by
# hand and on the self-hosted runner (.github/workflows/deploy.yml,
# deploy-mac): whichever checkout runs it, the compose project (`name:`), its
# volume and the secrets file are this machine's one copy.
#   deploy/mac/deploy.sh            # pull, build, restart
#   NO_PULL=1 deploy/mac/deploy.sh  # build what is checked out (CI does this)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root"
# The machine's secrets, outside every checkout (template: env.example).
export HA_ENV_FILE="${HA_ENV_FILE:-$HOME/.config/huggingarch/mac.env}"
[ -f "$HA_ENV_FILE" ] || { echo "missing $HA_ENV_FILE (template: $here/env.example)" >&2; exit 1; }
[ "${NO_PULL:-0}" = 1 ] || git pull --ff-only
export HA_GIT_SHA="$(git rev-parse HEAD)"
compose=(docker compose -f "$here/docker-compose.yml" --env-file "$HA_ENV_FILE")
"${compose[@]}" build app
"${compose[@]}" up -d
# The engine hash the new image serves, read inside it (no host Python env).
engine="$("${compose[@]}" run --rm --no-deps --entrypoint python app \
  -c 'from backend.fingerprint import engine_version; print(engine_version())')"
python3 tools/post_deploy_smoke.py http://127.0.0.1:8000 \
  --expect-version "$engine" --version-wait 900 --gallery-wait 600
# Each build leaves the previous image dangling; the VM's disk is bounded.
docker image prune -f >/dev/null
