"""A BYOK credential that points its agent at a custom endpoint must name a
model. Blank means the agent CLI's built-in default — Codex's is an OpenAI
model — which another vendor's endpoint rejects only once the run starts
(2026-11-09: a codex credential on api.deepseek.com with no model sent
`gpt-6.1-sol` to DeepSeek). Saving such a credential is refused; a row saved
before the rule is refused at run time, never launched on the CLI default.
"""
from __future__ import annotations

import re

import pytest

from backend.service import agent_credentials as C

_URL = re.compile(r".*user_agent_credentials.*")
_DURL = re.compile(r".*agent_task_defaults.*")


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(C, "_supabase", lambda: ("https://test-project.supabase.co", "svc"))
    monkeypatch.setattr(C.crypto, "decrypt", lambda _ct: "sk-real")


def test_stored_endpoint_without_model_refuses_the_run(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[{
        "agent_cli": "codex", "api_key_enc": "ct", "is_default": False,
        "base_url": "https://api.deepseek.com", "model": None, "task_models": None}])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    with pytest.raises(C.CredentialModelRequired):
        C.resolve_env("u", task="generate")


def test_task_pin_satisfies_the_run(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[{
        "agent_cli": "codex", "api_key_enc": "ct", "is_default": False,
        "base_url": "https://api.deepseek.com", "model": None,
        "task_models": {"generate": "deepseek-v4-pro"}}])
    env = C.resolve_env("u", task="generate")["env"]
    assert env["OPENAI_MODEL"] == "deepseek-v4-pro"


def test_vendor_endpoint_keeps_cli_default(store, httpx_mock):
    httpx_mock.add_response(method="GET", url=_URL, json=[{
        "agent_cli": "codex", "api_key_enc": "ct", "is_default": False,
        "base_url": None, "model": None, "task_models": None}])
    httpx_mock.add_response(method="GET", url=_DURL, json=[])
    assert "OPENAI_MODEL" not in C.resolve_env("u", task="generate")["env"]


@pytest.mark.parametrize("api_key", ["", "sk-new"])  # edit / create
def test_save_endpoint_without_model_is_400(client, user_token, api_key):
    # Rejected before any REST call (no httpx_mock ⇒ a call would error).
    r = client.put("/api/account/agent-credentials", headers=_auth(user_token), json={
        "agent_cli": "codex", "api_key": api_key,
        "base_url": "https://api.deepseek.com", "model": ""})
    assert r.status_code == 400
    assert "model" in r.json()["detail"]
