# Self-hosting on an Apple Silicon Mac

The production image (`Dockerfile`) running under Docker Desktop on a Mac,
reached through a Cloudflare Tunnel. It is meant to replace Fly: run it beside
Fly first (read-only Kernel Bench), then cut over.

What is where:

| | |
|---|---|
| Image, host-independent runtime config | `Dockerfile` (shared with Fly) |
| This host's volume, sizing, edge, machine id | `docker-compose.yml` |
| Secrets | `.env` (gitignored; template `env.example`) |
| Build + restart + content smoke | `deploy.sh` |

## One-time setup

1. **Docker Desktop**: Settings → Resources → Memory ≥ 10 GB (Fly's VM is
   8 GB; the sizing in `docker-compose.yml` is Fly's), disk ≥ 40 GB. Enable
   "Start Docker Desktop when you sign in".
2. **Keep the Mac awake**: System Settings → Energy → "Prevent automatic
   sleeping when the display is off", and "Start up automatically after a
   power failure".
3. **Cloudflare Tunnel** (the domain's DNS must be on Cloudflare): Zero Trust
   → Networks → Tunnels → Create (cloudflared). Copy the token. Add a public
   hostname, e.g. `mac.huggingarch.com` → service `http://app:8000`. Use a
   test hostname until the cutover; `www.huggingarch.com` moves at the end.
4. **Secrets**: `cp env.example .env`, then fill it from the Fly machine:
   `fly ssh console -C env` (skip `FLY_*` and what the image sets).
   `BENCH_LAKE_LEASE=off`.
5. `./deploy.sh`

First start restores the Kernel Bench lake from the backup and pulls the
SOL / gallery / sweep artifacts into the `cache` volume, so it is slower than
later starts. Check `http://127.0.0.1:8000/health`: the lake reports
`read_only: true` while Fly holds the lease.

## Running beside Fly

With `BENCH_LAKE_LEASE=off` the restored lake is a read-only copy
(`docs/measured_calibration.md` §4.6): no writes, no data migration, no
backups. It never touches Fly's lake or the backup bucket's contents. Its
measured data is the backup as of the restore; to refresh it, stop the stack,
delete the lake (`docker compose run --rm --no-deps --entrypoint rm app -rf
/app/.cache/bench_lake`), start again.

Deploy a new commit with `./deploy.sh` (pulls `main`, builds, restarts,
smokes). `NO_PULL=1 ./deploy.sh` builds what is checked out.

## Cutover

1. Stop the writer: `fly scale count 0` (keep the app and volume as rollback).
2. Make sure its last writes are backed up: Fly's `/health` reported backup
   lag 0 before it stopped.
3. On the Mac, refresh the lake from that backup (delete it as above).
4. Move the lease from the Fly machine to this one (same lake lineage):
   ```sql
   SELECT public.bench_lake_transfer('<fly-machine-id>', '<HA_MACHINE>');
   ```
   The Fly machine id is `fly machines list`'s ID.
5. Set `BENCH_LAKE_LEASE=on` in `.env`; `docker compose up -d`. `/health`
   must report `read_only: false` and the backup running.
6. Point `www.huggingarch.com` at the tunnel (public hostname in the tunnel,
   remove the Fly DNS records/certs).
7. Turn off `.github/workflows/deploy.yml`'s Fly deploy; update the
   hosting entry in `web/src/pages/Privacy.tsx`.

Rollback before step 6: transfer the lease back, `fly scale count 1`.
