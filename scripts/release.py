#!/usr/bin/env python3
"""The one producer-to-production release path.

A release is not a sequence of facts remembered by an operator.  It is:

1. warm the local release caches;
2. publish the three immutable Supabase artifacts;
3. verify the indexes this deployment will read;
4. push ``HEAD`` to ``main`` and wait for the tests -> release gate;
5. wait for the self-hosted host to deploy it (it pulls main once the gate
   passed) and let the public content smoke fail the release.

The trace provider is the error that motivated this entry point: warm, publish
and deployment must all name the same one.  The production host's runtime
configuration (``deploy/self-hosted/runtime.env``, read through
``tools.deploy_env``) is that single source; this script sets ``HA_CODE_TRACE`` from it before any backend module
is imported in this process or in a child.

Run it from the release checkout:

    uv run python scripts/release.py

Useful stages when diagnosing a producer:

    uv run python scripts/release.py prepare --workers 12
    uv run python scripts/release.py verify
    uv run python scripts/release.py smoke
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.deploy_env import host_env
TRACE_MODES = {"global", "functions"}


def _trace_mode(root: pathlib.Path = ROOT) -> str:
    """The deployment's trace provider, the one fact every producer shares:
    the production host's environment."""
    mode = host_env(root).get("HA_CODE_TRACE")
    if mode not in TRACE_MODES:
        raise ValueError(
            f"the production host's HA_CODE_TRACE must be one of "
            f"{sorted(TRACE_MODES)}, got {mode!r}")
    return str(mode)


def _load_release_env() -> str:
    """Load credentials, then return the deployment's trace provider."""
    load_dotenv(ROOT / ".env", override=False)
    return _trace_mode()


def _child_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    env["HA_CODE_TRACE"] = mode
    return env


def _run(command: list[str], *, mode: str) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=_child_env(mode), check=True)


def _cache_dir(env_name: str, default: str) -> pathlib.Path:
    return pathlib.Path(os.environ.get(env_name, default)).expanduser()


def prepare(workers: int) -> None:
    """Warm and publish SOL, Gallery and Scaling from this exact checkout."""
    mode = _load_release_env()
    home = pathlib.Path.home() / ".cache" / "huggingarch"
    sol = _cache_dir("HA_SOL_POOL_DIR", str(home / "sol-pool"))
    gallery = _cache_dir("HA_GALLERY_CACHE_DIR", str(home / "gallery"))
    sweep = _cache_dir("HA_SWEEP_CACHE_DIR", str(home / "sweep"))

    _run([sys.executable, "-m", "backend.warm_caches",
          "--workers", str(workers)], mode=mode)
    _run([sys.executable, "scripts/publish_sol_pool.py",
          "--pool-dir", str(sol)], mode=mode)
    _run([sys.executable, "scripts/publish_gallery_cache.py",
          "--cache-dir", str(gallery)], mode=mode)
    _run([sys.executable, "scripts/publish_sweep_cache.py",
          "--cache-dir", str(sweep)], mode=mode)
    verify()


def verify() -> None:
    """Every layer this deployment pulls has an index for its cache version."""
    mode = _load_release_env()
    # Imports below install the provider at import time; the environment is
    # already set, so they and the children above compute the same version.
    from backend.fingerprint import cache_engine_version
    from backend.gallery_cache_artifact import GALLERY_CACHE
    from backend.inference.sol_data import walkRecordSchema
    from backend.release_artifacts import (
        artifact_to_pull, bucket, request)
    from backend.supabase_env import configured, service_role
    from backend.sweep_cache_artifact import SWEEP_CACHE

    if not configured():
        raise RuntimeError("Supabase is not configured; cannot publish")
    engine = cache_engine_version()
    base, key = service_role()
    layers = (
        ("SOL", walkRecordSchema),
        ("Gallery", GALLERY_CACHE.prefix),
        ("Scaling", SWEEP_CACHE.prefix),
    )
    for label, prefix in layers:
        selected = artifact_to_pull(
            "", f"{prefix}/{engine}/index.json", engine, base, key, bucket(),
            30.0, request)
        if selected is None:
            raise RuntimeError(
                f"{label} has no {mode} artifact index for {prefix}/{engine}")
        parts = "" if not selected.parts else f" ({len(selected.parts)} parts)"
        print(f"RELEASE INDEX {label}: {selected.path}{parts}", flush=True)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True,
        text=True).stdout.strip()


def _github_repository() -> str:
    remote = _git("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/]([^/]+)/([^/.]+)(?:\.git)?$", remote)
    if match is None:
        raise RuntimeError(f"cannot derive GitHub repository from {remote!r}")
    return f"{match.group(1)}/{match.group(2)}"


def _github_request(url: str) -> dict[str, Any]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get(
        "GITHUB_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _wait_for_release_workflow(
        sha: str, *, timeout_minutes: int) -> None:
    repository = _github_repository()
    url = (f"https://api.github.com/repos/{repository}/actions/runs"
           f"?head_sha={sha}&per_page=100")
    deadline = time.monotonic() + timeout_minutes * 60
    print(f"RELEASE WATCH {repository} {sha}", flush=True)

    while time.monotonic() < deadline:
        try:
            document = _github_request(url)
        except (OSError, ValueError, urllib.error.URLError) as error:
            print(f"  GitHub API error: {error}", flush=True)
            time.sleep(30)
            continue
        runs = {run["name"]: run for run in document.get("workflow_runs", [])}
        tests, deploy = runs.get("tests"), runs.get("release gate")
        if tests is not None:
            print(f"  tests: {tests['status']} {tests['conclusion']}",
                  flush=True)
        if deploy is not None:
            print(f"  deploy: {deploy['status']} {deploy['conclusion']}",
                  flush=True)
        failed = [
            run for run in (tests, deploy)
            if run is not None and run["status"] == "completed"
            and run["conclusion"] not in ("success", "skipped")
        ]
        if failed:
            raise RuntimeError(
                "release workflow failed: "
                + "; ".join(f"{run['name']} {run['conclusion']} {run['html_url']}"
                            for run in failed))
        if (tests is not None and deploy is not None
                and tests["status"] == deploy["status"] == "completed"
                and tests["conclusion"] == deploy["conclusion"] == "success"):
            print("RELEASE WORKFLOW PASS", flush=True)
            return
        time.sleep(60)
    raise TimeoutError(f"release workflow did not finish in {timeout_minutes}m")


def promote(timeout_minutes: int) -> None:
    """Publish ``HEAD`` through the repository's tests -> release gate, then
    wait for the host to serve it."""
    verify()
    subprocess.run(["git", "fetch", "origin"], cwd=ROOT, check=True)
    if _git("status", "--porcelain"):
        raise RuntimeError("release checkout has uncommitted changes")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/main")
    if head != remote:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", remote, head],
            cwd=ROOT, check=True)
        _run(["git", "push", "origin", "HEAD:main"],
             mode=_trace_mode())
    _wait_for_release_workflow(head, timeout_minutes=timeout_minutes)
    # The host pulls main after the gate; the smoke waits for it to serve
    # this commit's engine.
    smoke("https://www.huggingarch.com")


def smoke(base_url: str) -> None:
    """The content smoke the host's deployer runs, against ``base_url``."""
    mode = _load_release_env()
    from backend.fingerprint import engine_version
    _run([
        sys.executable, "tools/post_deploy_smoke.py", base_url,
        "--expect-version", engine_version(),
        "--version-wait", "900", "--gallery-wait", "600",
    ], mode=mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-minutes", type=int, default=120)
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser(
        "prepare", help="warm, publish and verify artifacts")
    prepare_parser.add_argument("--workers", type=int, default=12)

    subparsers.add_parser("verify", help="verify the published indexes")

    promote_parser = subparsers.add_parser(
        "promote", help="push HEAD and wait for tests -> deploy")
    promote_parser.add_argument("--timeout-minutes", type=int, default=120)

    smoke_parser = subparsers.add_parser(
        "smoke", help="run post-deploy content smoke")
    smoke_parser.add_argument(
        "--base-url", default="https://www.huggingarch.com")

    arguments = parser.parse_args(argv)
    if arguments.command in (None, "release"):
        prepare(arguments.workers)
        promote(arguments.timeout_minutes)
    elif arguments.command == "prepare":
        prepare(arguments.workers)
    elif arguments.command == "verify":
        verify()
    elif arguments.command == "promote":
        promote(arguments.timeout_minutes)
    elif arguments.command == "smoke":
        smoke(arguments.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
