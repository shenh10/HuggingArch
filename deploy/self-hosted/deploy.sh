#!/usr/bin/env bash
# Build and roll the self-hosted deployment at the checkout's HEAD, then run the
# post-deploy content smoke. The deployer (deployer/autodeploy.py) runs it in
# its own clone inside the Docker VM; whichever checkout runs it, the compose
# project (`name:`), its volume and the secrets file are this host's one copy.
#   deploy/self-hosted/deploy.sh            # pull, build, restart
#   NO_PULL=1 deploy/self-hosted/deploy.sh  # build what is checked out
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root"
# The host's secrets (template: env.example); the deployer passes them.
: "${HA_ENV_FILE:?set HA_ENV_FILE to the host secrets file}"
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
# Through the public edge too (Cloudflare -> tunnel -> app) when the host names
# its public URL (HA_PUBLIC_URL): the loopback smoke cannot see a broken
# tunnel or DNS record.
if [ -n "${HA_PUBLIC_URL:-}" ]; then
  python3 tools/post_deploy_smoke.py "$HA_PUBLIC_URL" \
    --expect-version "$engine" --version-wait 300 --gallery-wait 300
fi
# Each build leaves the previous image dangling; the VM's disk is bounded.
docker image prune -f >/dev/null
