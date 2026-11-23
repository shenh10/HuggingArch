# Self-hosted deployment

The production image (`Dockerfile`) in a Docker VM on a self-hosted machine,
reached through a Cloudflare Tunnel. Deploys are pulled, not pushed: a
deployer container inside the same VM watches `main` and rolls a commit once
its release gate passed. Nothing from GitHub reaches the machine, no
repository secret names it, and the machine running the VM executes nothing
of the repository — everything runs inside the VM, which mounts nothing of
the machine. It replaces Fly: run it beside Fly first (read-only Kernel
Bench), then cut over.

What is where:

| | |
|---|---|
| Image | `Dockerfile` (shared with Fly) |
| Machine-independent runtime config (release cache identity and locations, artifact source, rate limits) | `runtime.env` — also what `scripts/release_warm.py` publishes under |
| This host's volume, sizing, edge, machine id | `docker-compose.yml` |
| Secrets | the `huggingarch-secrets` volume in the VM (template `env.example`) |
| Build + restart + content smoke | `deploy.sh` |
| Pull, gate check, deploy; container egress rules | `deployer/` (`deployer.compose.yml`) |
| Proxy for a host that needs one (only with `HTTP(S)_PROXY` set) | `proxy.compose.yml` |
| Release gate | `.github/workflows/deploy.yml`, check run `release-gate` |

## How a commit ships

1. Push to `main`; the `tests` workflow runs the full suite and
   `release-gate` waits for it, then runs the release subset.
2. The deployer sees the new head (`git ls-remote` every 30 s), asks GitHub
   for the head's `release-gate` check run, and waits while it is pending
   (asking again every 2 minutes).
3. Passed: it checks the commit out in its own clone and runs `deploy.sh`
   (build, roll, smoke on the loopback port and on `HA_PUBLIC_URL`). Failed:
   the commit is never deployed; the next head is.
4. A failed deploy is not retried on its own; the previous release keeps
   serving if it still runs. Logs: `docker logs -f huggingarch-deployer-deployer-1`.

## One-time setup

1. **Docker VM** (Docker Engine with compose and buildx; colima on macOS):
   at least 10 GB of memory (Fly's VM was 8 GB; the sizing in
   `docker-compose.yml` is Fly's) and 60 GB of disk. **Mount nothing of the
   machine into it** — the build context goes over the Docker API and the
   data lives in named volumes. With colima:
   ```bash
   mkdir -p ~/.colima-empty
   colima start --arch aarch64 --vm-type vz --cpu 8 --memory 12 --disk 60 \
     --mount ~/.colima-empty     # replaces the default (writable $HOME)
   colima ssh -- ls /Users       # must not list your home
   ```
   The VM and its containers must come back by themselves after a reboot or
   power loss (colima: `brew services start colima`; containers restart with
   Docker), and the machine must not sleep.
2. **Cloudflare Tunnel** (huggingarch.com's DNS is on Cloudflare): Zero
   Trust → Networks → Tunnels → Create (Cloudflared). Copy the token from the
   shown command (`--token eyJ…`). Public hostnames → HTTP `app:8000`.
3. **Secrets**: fill a copy of `env.example` (from the Fly machine:
   `fly ssh console -a huggingarch`, print the variables it names; plus
   `HA_MACHINE`, `CLOUDFLARE_TUNNEL_TOKEN`, `BENCH_LAKE_LEASE=off`,
   `HA_PUBLIC_URL`, optionally `HA_DEPLOY_GITHUB_TOKEN`), then load it into
   the VM and delete the copy:
   ```bash
   docker volume create huggingarch-secrets
   docker run --rm -i -v huggingarch-secrets:/s alpine \
     sh -c 'cat > /s/prod.env && chmod 600 /s/prod.env' < prod.env
   rm prod.env
   ```
   To change a value later, load the whole file again the same way.
4. **Start the deployer** (from any checkout; this is the one command run on
   the machine):
   ```bash
   docker compose -f deploy/self-hosted/deployer.compose.yml up -d --build
   docker logs -f huggingarch-deployer-deployer-1
   ```
   It deploys `main`'s head as soon as that commit's gate has passed. To
   update the deployer itself after a change under `deployer/`, run the same
   command again.

Deploy now, or retry a commit that failed:
```bash
docker exec huggingarch-deployer-deployer-1 autodeploy deploy
```

First start restores the Kernel Bench lake from the backup and pulls the
SOL / gallery / sweep artifacts into the `cache` volume, so it is slower than
later starts. `http://127.0.0.1:8000/health`: the lake reports
`read_only: true` until this host holds the lease.

## Container egress

The deployer keeps the VM's `DOCKER-USER` chain jumping to `HA-EGRESS`, which
drops container traffic to private, link-local and CGNAT ranges — the LAN and
the VM's gateway to the machine's own localhost — and lets
container-to-container and internet traffic through. It applies them at
start (a VM restart empties iptables and restarts the deployer with it) and
again before every deploy; `EGRESS NOT APPLIED` in its log means they are
missing. IPv4 only.

**Behind a proxy.** Where the containers reach the internet only through a
proxy, set the standard `HTTP_PROXY` / `HTTPS_PROXY` (`http://<ipv4>:<port>`)
in the secrets, and `NO_PROXY` for extra names on the compose network.
`deploy.sh` then loads `proxy.compose.yml`, which hands them to the image
build and to the app and tunnel, with loopback and the compose services'
names added to `NO_PROXY`; the deployer lets the proxy's address and port
through ahead of the drops. Base-image pulls are the Docker daemon's own and
follow its proxy settings, not these.

## Running beside Fly

With `BENCH_LAKE_LEASE=off` the restored lake is a read-only copy
(`docs/measured_calibration.md` §4.6): no writes, no data migration, no
backups. It never touches Fly's lake or the backup bucket's contents. Its
measured data is the backup as of the restore; to refresh it, stop the
stack, delete the lake, deploy again:
```bash
docker compose -p huggingarch down
docker run --rm -v huggingarch_cache:/c alpine rm -rf /c/bench_lake
docker exec huggingarch-deployer-deployer-1 autodeploy deploy
```

## Cutover

1. Point `www.huggingarch.com` and `huggingarch.com` at the tunnel: delete
   their DNS records for Fly, add them as the tunnel's public hostnames; set
   `HA_PUBLIC_URL=https://www.huggingarch.com` in the secrets.
2. Stop the writer once its `/health` shows the backup caught up:
   `fly scale count 0 -a huggingarch` (keep the app and volume as rollback).
3. `docker compose -p huggingarch down`, then move the lease from the Fly
   machine to this one (same lake lineage; the Fly machine id is
   `bench_lake_lease.machine`):
   ```sql
   SELECT public.bench_lake_transfer('<fly-machine-id>', '<HA_MACHINE>');
   ```
4. Delete the read-only lake (as above), set `BENCH_LAKE_LEASE=on` in the
   secrets (load them again), `autodeploy deploy`: the lake is restored from
   Fly's last backup and claimed. `/health` must report `read_only: false`
   and the backup running.

Rollback: stop this stack, transfer the lease back, `fly scale count 1`,
point DNS back at Fly.
