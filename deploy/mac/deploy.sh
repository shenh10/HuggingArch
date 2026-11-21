#!/usr/bin/env bash
# Build and roll the Mac deployment at the checkout's HEAD, then run the
# post-deploy content smoke against the local port.
#   deploy/mac/deploy.sh            # pull main, build, restart
#   NO_PULL=1 deploy/mac/deploy.sh  # build what is checked out
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root"
[ -f "$here/.env" ] || { echo "missing $here/.env (template: env.example)" >&2; exit 1; }
[ "${NO_PULL:-0}" = 1 ] || git pull --ff-only
export HA_GIT_SHA="$(git rev-parse HEAD)"
compose=(docker compose -f "$here/docker-compose.yml")
"${compose[@]}" build app
"${compose[@]}" up -d
# The engine hash the new image serves, read inside it (no host Python env).
engine="$("${compose[@]}" run --rm --no-deps --entrypoint python app \
  -c 'from backend.fingerprint import engine_version; print(engine_version())')"
python3 tools/post_deploy_smoke.py http://127.0.0.1:8000 \
  --expect-version "$engine" --version-wait 900 --gallery-wait 600
