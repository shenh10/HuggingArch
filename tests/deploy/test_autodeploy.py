"""The self-hosted deployer's decisions (deploy/self-hosted/deployer/
autodeploy.py), driven by a fake git, GitHub API and iptables: what it
deploys, when, and what it never deploys."""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOST = ROOT / "deploy" / "self-hosted"
_spec = importlib.util.spec_from_file_location("autodeploy", HOST / "deployer" / "autodeploy.py")
autodeploy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(autodeploy)

A, B = "a" * 40, "b" * 40


class World:
    """main's head, each commit's gate, whether deploy.sh succeeds, and the
    VM's iptables; records what was run."""

    def __init__(self):
        self.head, self.gates, self.deploy_ok = A, {}, True
        self.chains: dict[str, list[str]] = {"DOCKER-USER": []}
        self.deployed: list[str] = []
        self.checked_out = None
        self.iptables: list[list[str]] = []
        self.now, self.api_calls = 0.0, 0
        self.git_env: list[dict] = []

    def run(self, cmd, **kw):
        ok = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == "git" and {"ls-remote", "clone", "fetch"} & set(cmd):
            self.git_env.append(kw.get("env") or {})
        if cmd[:2] == ["git", "ls-remote"]:
            ok.stdout = f"{self.head}\trefs/heads/main\n"
        elif cmd[0] == "git" and "checkout" in cmd:
            self.checked_out = cmd[-1]
        elif cmd[0] == "bash":
            self.deployed.append(self.checked_out)
            ok.returncode = 0 if self.deploy_ok else 1
        elif cmd[0] == "iptables":
            return self._iptables(cmd[1:], ok)
        elif cmd[0] == "iptables-legacy":
            ok.returncode = 1
        return ok

    def _iptables(self, args, ok):
        self.iptables.append(args)
        op, chain = args[0], args[1]
        if op == "-S":
            if chain not in self.chains:
                ok.returncode = 1
            else:
                ok.stdout = "".join(f"{line}\n" for line in [f"-N {chain}", *self.chains[chain]])
        elif op in ("-N", "-F"):
            self.chains[chain] = []
        elif op == "-A":
            self.chains[chain].append(" ".join(args))
        elif op == "-C":
            ok.returncode = 0 if f"-A {chain} {' '.join(args[2:])}" in self.chains[chain] else 1
        elif op == "-I":
            self.chains[chain].insert(0, f"-A {chain} {' '.join(args[3:])}")
        return ok

    def api(self, url, token, proxy=None):
        self.api_calls += 1
        self.api_proxy = proxy
        sha = url.split("/commits/")[1].split("/")[0]
        gate = self.gates.get(sha)
        if gate is None:
            return {"check_runs": []}
        status, conclusion = gate
        return {"check_runs": [{"status": status, "conclusion": conclusion}]}


@pytest.fixture
def world():
    return World()


@pytest.fixture
def deployer(world, tmp_path):
    secrets = tmp_path / "prod.env"
    secrets.write_text("HA_PUBLIC_URL='https://example.test'\n")
    (tmp_path / "work").mkdir()
    env = {"HA_REPO_URL": "https://github.com/o/r.git", "HA_GATE_CHECK": "release-gate",
           "HA_ENV_FILE": str(secrets), "HA_WORK": str(tmp_path / "work")}
    return autodeploy.Deployer(env, run=world.run, api=world.api, clock=lambda: world.now)


def test_deploys_only_a_head_whose_gate_passed_and_only_once(world, deployer):
    assert deployer.step() == "gate-pending"                 # no check run yet
    world.gates[A] = ("in_progress", None)
    world.now += 200
    assert deployer.step() == "gate-pending"
    assert world.deployed == []
    world.gates[A] = ("completed", "success")
    world.now += 200
    assert deployer.step() == "deployed"
    assert world.deployed == [A]
    assert deployer.step() == "current"
    assert world.deployed == [A]


def test_a_failed_gate_is_never_deployed(world, deployer):
    world.gates[A] = ("completed", "failure")
    assert deployer.step() == "gate-failed"
    assert deployer.step(again=True) == "gate-failed"
    assert world.deployed == []


def test_a_failed_deploy_waits_for_an_operator_or_the_next_head(world, deployer):
    world.gates[A] = ("completed", "success")
    world.deploy_ok = False
    assert deployer.step() == "deploy-failed"
    assert deployer.step() == "failed-before"
    assert world.deployed == [A]
    world.deploy_ok = True
    assert deployer.step(again=True) == "deployed"            # `autodeploy deploy`
    world.head, world.gates[B] = B, ("completed", "success")
    assert deployer.step() == "deployed"
    assert world.deployed == [A, A, B]


def test_manual_deploy_rolls_the_current_head_again(world, deployer):
    world.gates[A] = ("completed", "success")
    deployer.step()
    assert deployer.step(again=True) == "deployed"
    assert world.deployed == [A, A]


def test_egress_rules_are_in_place_before_a_deploy_and_untouched_by_idle_passes(world, deployer):
    assert deployer.step() == "gate-pending"
    assert world.iptables == []                                 # an idle pass reads no rules
    world.gates[A] = ("completed", "success")
    world.now += 200
    deployer.step()
    chain = world.chains["HA-EGRESS"]
    for net in autodeploy.Egress.PRIVATE:
        assert f"-A HA-EGRESS -d {net} -j DROP" in chain
    assert world.chains["DOCKER-USER"] == ["-A DOCKER-USER -j HA-EGRESS"]
    world.iptables.clear()
    deployer.step()                                             # current: nothing to do
    assert world.iptables == []


def test_ensure_repairs_lost_rules_and_leaves_intact_ones_alone(world, deployer):
    deployer.egress.ensure()
    chain = list(world.chains["HA-EGRESS"])
    world.iptables.clear()
    deployer.egress.ensure()
    assert not [a for a in world.iptables if a[0] in ("-F", "-N", "-A", "-I")], world.iptables
    world.chains["HA-EGRESS"] = chain[:2]                       # a VM restart lost some
    deployer.egress.ensure()
    assert world.chains["HA-EGRESS"] == chain


def test_the_gate_the_deployer_waits_for_is_a_job_of_the_release_workflow():
    """The deployer reads the check run by name; a renamed job would leave it
    waiting forever on a gate that never reports."""
    compose = yaml.safe_load((HOST / "deployer.compose.yml").read_text())
    gate = compose["services"]["deployer"]["environment"]["HA_GATE_CHECK"]
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text())
    assert gate in workflow["jobs"], (gate, list(workflow["jobs"]))
    assert "main" in workflow[True]["push"]["branches"]


def test_a_pending_gate_is_asked_about_at_most_once_per_poll_interval(world, deployer):
    """30 s passes against a pending gate must not spend the anonymous API
    quota (60 an hour)."""
    for _ in range(4):                                          # two minutes of passes
        assert deployer.step() == "gate-pending"
        world.now += 30
    assert world.api_calls == 1
    world.gates[A] = ("completed", "success")
    world.now += 1
    assert deployer.step() == "deployed"
    assert world.api_calls == 2


def test_the_containers_proxy_is_the_one_private_destination_let_through(world, deployer, tmp_path):
    secrets = tmp_path / "prod.env"
    secrets.write_text(secrets.read_text() + "HTTP_PROXY=http://192.168.5.2:7897\n"
                       "HTTPS_PROXY='http://192.168.5.2:7897'\n")
    deployer.ensure_egress()
    chain = world.chains["HA-EGRESS"]
    allow = "-A HA-EGRESS -d 192.168.5.2/32 -p tcp -m tcp --dport 7897 -j RETURN"
    assert chain.count(allow) == 1                              # one proxy, named twice
    assert chain.index(allow) < chain.index("-A HA-EGRESS -d 192.168.0.0/16 -j DROP")
    world.iptables.clear()
    deployer.ensure_egress()                                    # listed form matches: no rebuild
    assert not [a for a in world.iptables if a[0] in ("-F", "-N", "-A", "-I")]
    secrets.write_text("HTTPS_PROXY=http://proxy.lan:7897\n")   # iptables -d would resolve it once
    with pytest.raises(ValueError, match="ipv4"):
        deployer.ensure_egress()


def test_no_proxy_means_no_allowance(world, deployer):
    deployer.ensure_egress()
    assert not [r for r in world.chains["HA-EGRESS"] if "--dport" in r]


def test_the_proxy_reaches_the_build_and_every_running_container():
    """proxy.compose.yml hands the same variables to the app's build and to
    each service docker-compose.yml runs."""
    proxy = yaml.safe_load((HOST / "proxy.compose.yml").read_text())
    base = yaml.safe_load((HOST / "docker-compose.yml").read_text())
    names = set(proxy["x-proxy"])
    assert {"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"} <= names
    assert set(proxy["services"]["app"]["build"]["args"]) == names
    assert set(proxy["services"]) == set(base["services"])
    for service in proxy["services"].values():
        assert set(service["environment"]) == names


def test_deploy_sh_completes_no_proxy_with_loopback_and_the_compose_services(tmp_path):
    """The real deploy.sh, up to its first build, against a docker stub that
    answers `config --services` and records the rest."""
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "docker").write_text(
        "#!/bin/sh\n"
        'case "$*" in *"config --services"*) printf "app\\ntunnel\\n"; exit 0;; esac\n'
        'echo "$NO_PROXY|$HTTPS_PROXY|$*" >> "$RECORD"; exit 1\n')
    (stub / "docker").chmod(0o755)
    secrets = tmp_path / "prod.env"
    secrets.write_text("HTTP_PROXY=http://192.168.5.2:7897\nNO_PROXY=beszel\n")
    record = tmp_path / "record"
    subprocess.run(["bash", str(HOST / "deploy.sh")], cwd=ROOT, env={
        "PATH": f"{stub}:/usr/bin:/bin", "HA_ENV_FILE": str(secrets), "NO_PULL": "1",
        "RECORD": str(record)}, capture_output=True, check=False)   # the stub fails the build
    no_proxy, https_proxy, args = record.read_text().splitlines()[0].split("|")
    assert no_proxy.split(",") == ["localhost", "127.0.0.1", "::1", "app", "tunnel", "beszel"]
    assert https_proxy == "http://192.168.5.2:7897"
    assert "proxy.compose.yml" in args


def test_the_deployer_reaches_github_through_the_hosts_proxy(world, deployer, tmp_path):
    """Where the host's only reliable route out is its proxy, the deployer's
    own git and API calls take it too — a direct ls-remote stalled for
    minutes on the first trial."""
    (tmp_path / "prod.env").write_text("HTTPS_PROXY=http://192.168.5.2:7897\nNO_PROXY=beszel\n")
    world.gates[A] = ("completed", "success")
    assert deployer.step() == "deployed"
    assert world.api_proxy == "http://192.168.5.2:7897"
    assert len(world.git_env) == 3                              # ls-remote, clone, fetch
    for env in world.git_env:
        assert env.get("HTTPS_PROXY") == "http://192.168.5.2:7897"
        assert env.get("NO_PROXY") == "localhost,127.0.0.1,::1,beszel"


def test_without_a_proxy_the_deployer_goes_direct(world, deployer, monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    world.gates[A] = ("completed", "success")
    deployer.step()
    assert world.api_proxy is None
    assert len(world.git_env) == 3
    assert not [e for e in world.git_env if "HTTPS_PROXY" in e]
