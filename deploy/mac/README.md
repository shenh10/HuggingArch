# Self-hosting on an Apple Silicon Mac

The production image (`Dockerfile`) running under colima on a Mac, reached
through a Cloudflare Tunnel, deployed on every push to `main` by a GitHub
self-hosted runner on the same Mac. It is meant to replace Fly: run it beside
Fly first (read-only Kernel Bench), then cut over.

What is where:

| | |
|---|---|
| Image, host-independent runtime config | `Dockerfile` (shared with Fly) |
| This host's volume, sizing, edge, machine id | `docker-compose.yml` |
| Secrets | `~/.config/huggingarch/mac.env` on the Mac (template `env.example`) |
| Build + restart + content smoke | `deploy.sh` (by hand and from CI) |
| Deploy on push | `.github/workflows/deploy.yml`, job `deploy-mac` |

The compose project is named (`name: huggingarch`), so every checkout that
runs `deploy.sh` — yours, the runner's — rolls the same containers and the
same `cache` volume.

## One-time setup

1. **colima** (Docker runtime):
   ```bash
   brew install colima docker docker-compose docker-buildx
   # ~/.docker/config.json: "cliPluginsExtraDirs": ["/opt/homebrew/lib/docker/cli-plugins"]
   colima start --arch aarch64 --vm-type vz --cpu 8 --memory 12 --disk 60
   brew services start colima     # start at login
   ```
   Memory ≥ 10 GB (Fly's VM is 8 GB; the sizing in `docker-compose.yml` is
   Fly's).
2. **Keep the Mac up**: no automatic sleep, start after a power failure
   (`sudo pmset -a sleep 0 disksleep 0 autorestart 1`), and automatic login
   — colima and the runner are login services.
3. **Cloudflare Tunnel** (huggingarch.com's DNS is on Cloudflare): Zero
   Trust → Networks → Tunnels → Create (Cloudflared). Copy the token from the
   shown command (`--token eyJ…`). Public hostname, e.g. `mac.huggingarch.com`
   → HTTP `app:8000`. `www` moves at the cutover.
4. **Secrets**:
   ```bash
   mkdir -p ~/.config/huggingarch
   cp env.example ~/.config/huggingarch/mac.env && chmod 600 ~/.config/huggingarch/mac.env
   ```
   Fill it from the Fly machine (`fly ssh console -a huggingarch`, print the
   variables `env.example` names), plus `HA_MACHINE`,
   `CLOUDFLARE_TUNNEL_TOKEN`, `BENCH_LAKE_LEASE=off`.
5. `NO_PULL=1 ./deploy.sh`

First start restores the Kernel Bench lake from the backup and pulls the
SOL / gallery / sweep artifacts into the `cache` volume, so it is slower than
later starts. `http://127.0.0.1:8000/health`: the lake reports
`read_only: true` while Fly holds the lease.

## Deploy on push (self-hosted runner)

The `deploy-mac` job waits for the same gate as Fly's (`smoke`: the full
suite passed on the commit, plus the release subset), then runs `deploy.sh`
on the runner labelled `huggingarch-mac`. Secrets stay on the Mac; the job
gets none. When the Mac is offline the job queues; Fly's deploy does not
wait for it (separate concurrency groups).

1. **Lock down fork PRs first** — the repository is public, and a pull
   request can edit a workflow to run on any self-hosted runner. GitHub →
   Settings → Actions → General → "Fork pull request workflows from outside
   collaborators" → **Require approval for all outside collaborators**. Never
   approve a fork run that touches `.github/workflows/`.
2. **Register the runner**: GitHub → Settings → Actions → Runners → New
   self-hosted runner → macOS / ARM64. Run the shown download commands in
   `~/actions-runner`, then configure with the label:
   ```bash
   cd ~/actions-runner
   ./config.sh --url https://github.com/shenh10/HuggingArch --token <shown token> \
     --name huggingarch-mac --labels huggingarch-mac --unattended
   ./svc.sh install && ./svc.sh start
   ```
   Run `config.sh` from a shell where `docker ps` works: the runner keeps
   that `PATH` (`~/actions-runner/.path`).
3. Check: the runner shows **Idle** under Settings → Actions → Runners; then
   Actions → deploy → Run workflow (or the next push to `main`) runs
   `deploy-mac`.

By hand, any time: `cd ~/HuggingArch && deploy/mac/deploy.sh`.

## Running beside Fly

With `BENCH_LAKE_LEASE=off` the restored lake is a read-only copy
(`docs/measured_calibration.md` §4.6): no writes, no data migration, no
backups. It never touches Fly's lake or the backup bucket's contents. Its
measured data is the backup as of the restore; to refresh it, stop the
stack, delete the lake, start again:
```bash
docker compose -p huggingarch down
docker run --rm -v huggingarch_cache:/c alpine rm -rf /c/bench_lake
NO_PULL=1 deploy/mac/deploy.sh
```

## Cutover

1. Stop the writer: `fly scale count 0 -a huggingarch` (keep the app and
   volume as rollback). Before that, Fly's `/health` must show its backup
   caught up.
2. On the Mac, refresh the lake from that backup (as above).
3. Move the lease from the Fly machine to this one (same lake lineage):
   ```sql
   SELECT public.bench_lake_transfer('<fly-machine-id>', '<HA_MACHINE>');
   ```
   The Fly machine id is `fly machines list -a huggingarch`'s ID.
4. `BENCH_LAKE_LEASE=on` in `mac.env`; `NO_PULL=1 deploy/mac/deploy.sh`.
   `/health` must report `read_only: false` and the backup running.
5. Point `www.huggingarch.com` at the tunnel (public hostname in the
   tunnel; remove the Fly DNS records).
6. Remove the `deploy` (Fly) job from `.github/workflows/deploy.yml`;
   update the hosting entry in `web/src/pages/Privacy.tsx`.

Rollback before step 5: transfer the lease back, `fly scale count 1`.
