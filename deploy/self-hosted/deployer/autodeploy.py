#!/usr/bin/env python3
"""Pull-based deploys of the self-hosted host, from inside its Docker VM.

At start, and again before every deploy, the egress rules: containers reach
the internet, never the LAN or the VM's host (`Egress`). The rules live in the
VM's iptables, which only a VM restart empties — and the deployer restarts
with the VM.

Every `HA_INTERVAL_S` seconds (default 30):

  1. `git ls-remote` names the branch's head (no API quota spent);
  2. a head already deployed, or already failed, is left alone;
  3. the head deploys only when GitHub says its release gate passed — the
     check run `HA_GATE_CHECK` (deploy.yml's job) concluded `success`; a
     pending gate is waited for, a failed one is never deployed;
  4. deploy: check the head out in the deployer's own clone and run
     `deploy/self-hosted/deploy.sh` (build, roll, smoke).

Nothing tells the deployer what to deploy: it reads main and the gate itself,
so no credential anywhere can make it run anything but a gated main commit.

    autodeploy run      the loop (the container's command)
    autodeploy deploy   deploy the head now, even one already deployed or
                        failed (after a secrets change, a lake refresh), once
                        its gate passed

A pending gate is asked about at most every HA_GATE_POLL_S seconds (default
120: 30 requests an hour, inside GitHub's anonymous quota of 60).

Environment: HA_REPO_URL, HA_BRANCH, HA_GATE_CHECK, HA_ENV_FILE (the host's
secrets; this reads HA_DEPLOY_GITHUB_TOKEN, an optional read-only token for
the API quota, HA_PUBLIC_URL, the public smoke's base URL, and the standard
HTTP_PROXY / HTTPS_PROXY when the containers reach the internet through a
proxy), HA_WORK (/work), HA_INTERVAL_S, HA_GATE_POLL_S.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from ipaddress import IPv4Address
from urllib.parse import urlsplit
from pathlib import Path
from typing import Callable

SHA = re.compile(r"^[0-9a-f]{40}$")
#: The proxy variables the containers are given (deploy.sh), in both spellings.
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"), message, flush=True)


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines; one pair of surrounding quotes is stripped."""
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        env[name.strip()] = value
    return env


def github_json(url: str, token: str | None, proxy: str | None = None) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "huggingarch-deployer"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else {}))
    with opener.open(urllib.request.Request(url, headers=headers), timeout=20) as r:
        return json.load(r)


class Egress:
    """The VM's DOCKER-USER chain jumps to HA-EGRESS, which drops container
    traffic to private, link-local and CGNAT ranges (the LAN, the VM's host
    gateway) and lets container-to-container and internet traffic through.
    A proxy the containers are told to use (HTTP_PROXY / HTTPS_PROXY) is
    let through ahead of the drops — that address and port, nothing else."""

    CHAIN = "HA-EGRESS"
    PRIVATE = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
               "169.254.0.0/16", "100.64.0.0/10")

    def __init__(self, run: Callable = subprocess.run):
        self.run = run

    def _binary(self) -> str | None:
        """The iptables whose ruleset Docker's chains are in (nft or legacy)."""
        for binary in ("iptables", "iptables-legacy"):
            if self.run([binary, "-S", "DOCKER-USER"], capture_output=True).returncode == 0:
                return binary
        return None

    @staticmethod
    def destination(proxy: str) -> tuple[str, int]:
        """A proxy URL's (IPv4, port). A name is refused: iptables would
        resolve it once, at insert time."""
        parts = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
        try:
            host = str(IPv4Address(parts.hostname or ""))
            port = parts.port or {"http": 80, "https": 443}[parts.scheme]
        except (ValueError, KeyError) as e:
            raise ValueError(f"a proxy for the egress rules must be http(s)://<ipv4>:<port>, "
                             f"got {proxy!r}") from e
        return host, port

    def rules(self, proxies: tuple[str, ...] = ()) -> list[list[str]]:
        allowed = [["-A", self.CHAIN, "-d", f"{host}/32", "-p", "tcp", "-m", "tcp",
                    "--dport", str(port), "-j", "RETURN"]
                   for host, port in sorted({self.destination(p) for p in proxies})]
        return ([["-A", self.CHAIN, "-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED",
                  "-j", "RETURN"],
                 ["-A", self.CHAIN, "-o", "br+", "-j", "RETURN"],
                 ["-A", self.CHAIN, "-o", "docker0", "-j", "RETURN"]]
                + allowed
                + [["-A", self.CHAIN, "-d", net, "-j", "DROP"] for net in self.PRIVATE]
                + [["-A", self.CHAIN, "-j", "RETURN"]])

    def ensure(self, proxies: tuple[str, ...] = ()) -> bool:
        """Rebuilt only when the chain differs from `rules()`, so checking it
        never opens a window without the drops."""
        ipt = self._binary()
        if ipt is None:
            log("EGRESS NOT APPLIED: no DOCKER-USER chain in iptables or iptables-legacy")
            return False
        listed = self.run([ipt, "-S", self.CHAIN], capture_output=True, text=True)
        rules = self.rules(proxies)
        want = [f"-N {self.CHAIN}"] + [" ".join(rule) for rule in rules]
        if listed.returncode or listed.stdout.split("\n")[:-1] != want:
            if listed.returncode:
                self.run([ipt, "-N", self.CHAIN], check=True, capture_output=True)
            else:
                self.run([ipt, "-F", self.CHAIN], check=True, capture_output=True)
            for rule in rules:
                self.run([ipt, *rule], check=True, capture_output=True)
            log("egress rules (re)applied")
        if self.run([ipt, "-C", "DOCKER-USER", "-j", self.CHAIN], capture_output=True).returncode:
            self.run([ipt, "-I", "DOCKER-USER", "1", "-j", self.CHAIN], check=True,
                     capture_output=True)
        return True


class Deployer:
    def __init__(self, env: dict = os.environ, run: Callable = subprocess.run,
                 api: Callable = github_json, clock: Callable = time.monotonic):
        self.repo_url = env["HA_REPO_URL"]
        self.branch = env.get("HA_BRANCH", "main")
        self.gate_check = env["HA_GATE_CHECK"]
        self.env_file = Path(env["HA_ENV_FILE"])
        self.work = Path(env.get("HA_WORK", "/work"))
        self.interval = float(env.get("HA_INTERVAL_S", "30"))
        self.gate_poll = float(env.get("HA_GATE_POLL_S", "120"))
        match = re.search(r"github\.com[/:]([^/]+/[^/.]+?)(?:\.git)?/?$", self.repo_url)
        if not match:
            raise SystemExit(f"HA_REPO_URL is not a GitHub repository: {self.repo_url}")
        self.slug = match.group(1)
        self.run, self.api, self.clock = run, api, clock
        self.egress = Egress(run)
        self._announced: tuple[str, str] | None = None
        self._pending: tuple[str, float] | None = None    # (sha, when its gate was asked)

    # ---- state: the last deployed and the last failed head

    @property
    def checkout(self) -> Path:
        return self.work / "HuggingArch"

    def state(self) -> dict:
        path = self.work / "state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def record(self, **changes) -> None:
        state = {**self.state(), **changes}
        (self.work / "state.json").write_text(json.dumps(state, indent=1))

    # ---- what to deploy

    def network_env(self) -> dict[str, str]:
        """The deployer's own outbound (git, the GitHub API, the smoke) takes
        the host's proxy too, when it names one; loopback stays direct."""
        secrets = read_env_file(self.env_file)
        proxy = secrets.get("HTTPS_PROXY") or secrets.get("HTTP_PROXY")
        if not proxy:
            return {}
        no_proxy = ",".join(filter(None, ["localhost,127.0.0.1,::1", secrets.get("NO_PROXY")]))
        return {"HTTP_PROXY": secrets.get("HTTP_PROXY") or proxy, "HTTPS_PROXY": proxy,
                "http_proxy": secrets.get("HTTP_PROXY") or proxy, "https_proxy": proxy,
                "NO_PROXY": no_proxy, "no_proxy": no_proxy}

    def head(self) -> str:
        out = self.run(["git", "ls-remote", self.repo_url, f"refs/heads/{self.branch}"],
                       check=True, capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, **self.network_env())).stdout.split()
        if not out or not SHA.match(out[0]):
            raise RuntimeError(f"git ls-remote named no head for {self.branch}: {out!r}")
        return out[0]

    def ensure_egress(self) -> bool:
        secrets = read_env_file(self.env_file)
        return self.egress.ensure(tuple(secrets[k] for k in PROXY_VARIABLES if secrets.get(k)))

    def gate(self, sha: str) -> str:
        """'passed', 'failed' or 'pending' (not run yet, or running)."""
        token = read_env_file(self.env_file).get("HA_DEPLOY_GITHUB_TOKEN") or None
        url = (f"https://api.github.com/repos/{self.slug}/commits/{sha}/check-runs"
               f"?check_name={self.gate_check}&filter=latest")
        runs = self.api(url, token, self.network_env().get("HTTPS_PROXY")).get("check_runs", [])
        if not runs:
            return "pending"
        if any(r.get("status") != "completed" for r in runs):
            return "pending"
        return "passed" if all(r.get("conclusion") == "success" for r in runs) else "failed"

    # ---- doing it

    def deploy(self, sha: str) -> bool:
        net = dict(os.environ, **self.network_env())
        if not (self.checkout / ".git").exists():
            self.run(["git", "clone", "--quiet", self.repo_url, str(self.checkout)], check=True,
                     timeout=900, env=net)
        git = ["git", "-C", str(self.checkout)]
        self.run([*git, "fetch", "--quiet", "origin", self.branch], check=True, timeout=600,
                 env=net)
        self.run([*git, "checkout", "--quiet", "--detach", sha], check=True)
        secrets = read_env_file(self.env_file)
        env = dict(net, NO_PULL="1", HA_ENV_FILE=str(self.env_file),
                   HA_PUBLIC_URL=secrets.get("HA_PUBLIC_URL", ""))
        return self.run(["bash", str(self.checkout / "deploy/self-hosted/deploy.sh")],
                        env=env).returncode == 0

    def _announce(self, sha: str, what: str) -> None:
        if self._announced != (sha, what):
            log(f"{sha[:12]}: {what}")
            self._announced = (sha, what)

    def step(self, again: bool = False) -> str:
        """One pass; returns what happened. `again` deploys the head even
        when it is the deployed or the failed one."""
        sha = self.head()
        state = self.state()
        if sha == state.get("deployed") and not again:
            return "current"
        if sha == state.get("failed") and not again:
            self._announce(sha, "failed before; not retried (autodeploy deploy retries)")
            return "failed-before"
        if self._pending and self._pending[0] == sha and \
                self.clock() - self._pending[1] < self.gate_poll:
            return "gate-pending"
        gate = self.gate(sha)
        self._pending = (sha, self.clock()) if gate == "pending" else None
        if gate != "passed":
            self._announce(sha, f"release gate {gate}")
            return f"gate-{gate}"
        log(f"{sha[:12]}: release gate passed; deploying")
        self.ensure_egress()
        if self.deploy(sha):
            self.record(deployed=sha)
            log(f"{sha[:12]}: deployed")
            return "deployed"
        self.record(failed=sha)
        log(f"{sha[:12]}: DEPLOY FAILED; the previous release keeps serving if it still runs")
        return "deploy-failed"

    def loop(self) -> None:
        log(f"watching {self.slug}@{self.branch}, gate {self.gate_check!r}, "
            f"every {self.interval:g}s")
        self.ensure_egress()
        while True:
            try:
                self.step()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    urllib.error.URLError, OSError,
                    RuntimeError, ValueError) as e:
                log(f"pass failed: {type(e).__name__}: {e}")
            time.sleep(self.interval)


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "run"
    deployer = Deployer()
    if command == "run":
        deployer.loop()
        return 0
    if command == "deploy":
        outcome = deployer.step(again=True)
        log(f"outcome: {outcome}")
        return 0 if outcome == "deployed" else 1
    raise SystemExit(f"unknown command {command!r}: run | deploy")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
