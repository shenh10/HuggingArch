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

    def run(self, cmd, **kw):
        ok = subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
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

    def api(self, url, token):
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
    return autodeploy.Deployer(env, run=world.run, api=world.api)


def test_deploys_only_a_head_whose_gate_passed_and_only_once(world, deployer):
    assert deployer.step() == "gate-pending"                 # no check run yet
    world.gates[A] = ("in_progress", None)
    assert deployer.step() == "gate-pending"
    assert world.deployed == []
    world.gates[A] = ("completed", "success")
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


def test_egress_rules_are_applied_once_and_left_alone_while_intact(world, deployer):
    world.gates[A] = ("completed", "success")
    deployer.step()
    chain = world.chains["HA-EGRESS"]
    for net in autodeploy.Egress.PRIVATE:
        assert f"-A HA-EGRESS -d {net} -j DROP" in chain
    assert world.chains["DOCKER-USER"] == ["-A DOCKER-USER -j HA-EGRESS"]
    world.iptables.clear()
    deployer.step()
    assert not [a for a in world.iptables if a[0] in ("-F", "-N", "-A", "-I")], world.iptables
    world.chains["HA-EGRESS"] = chain[:2]                       # a VM restart lost some
    deployer.step()
    assert world.chains["HA-EGRESS"] == chain


def test_the_gate_the_deployer_waits_for_is_a_job_of_the_release_workflow():
    """The deployer reads the check run by name; a renamed job would leave it
    waiting forever on a gate that never reports."""
    compose = yaml.safe_load((HOST / "deployer.compose.yml").read_text())
    gate = compose["services"]["deployer"]["environment"]["HA_GATE_CHECK"]
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "deploy.yml").read_text())
    assert gate in workflow["jobs"], (gate, list(workflow["jobs"]))
    assert "main" in workflow[True]["push"]["branches"]
