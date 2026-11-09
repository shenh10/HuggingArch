"""Agent subprocesses receive credentials without inheriting server secrets."""
from __future__ import annotations

import io
import json
import sys
import threading
from unittest import mock

from backend.agents.claude import ClaudeBackend
from backend.agents.codex import CodexBackend
from backend.agents.kimi import KimiBackend


def test_host_agent_env_keeps_only_selected_agent_credentials(monkeypatch):
    monkeypatch.setenv("PATH", "/safe/bin")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-host-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-host-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-leak")
    monkeypatch.setenv("HA_CREDS_ENC_KEY", "must-not-leak-either")

    env = CodexBackend._merge_env(None)

    assert env["PATH"] == "/safe/bin"
    assert env["OPENAI_API_KEY"] == "openai-host-key"
    assert env["CODEX_API_KEY"] == "openai-host-key"
    assert env["OPENAI_BASE_URL"] == "https://openai.example"
    assert "ANTHROPIC_API_KEY" not in env
    assert "SUPABASE_SERVICE_ROLE_KEY" not in env
    assert "HA_CREDS_ENC_KEY" not in env


def test_byok_env_does_not_mix_in_host_credentials(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-key")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "must-not-leak")

    env = ClaudeBackend._merge_env({"ANTHROPIC_API_KEY": "user-key"})

    assert env["ANTHROPIC_API_KEY"] == "user-key"
    assert "SUPABASE_JWT_SECRET" not in env


def _only_gateway_token(monkeypatch):
    """A host that reaches Anthropic through a gateway: a bearer token and
    the gateway's URL, no API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gateway-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example")


def test_host_claude_run_carries_the_gateway_auth_token(monkeypatch):
    from backend.agents.base import host_agent_credentials, host_has_credentials
    _only_gateway_token(monkeypatch)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-leak")

    env = ClaudeBackend._merge_env(None)

    assert env["ANTHROPIC_AUTH_TOKEN"] == "gateway-token"
    assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
    assert "ANTHROPIC_API_KEY" not in env
    assert "SUPABASE_SERVICE_ROLE_KEY" not in env
    assert host_has_credentials("claude")
    assert host_agent_credentials("claude") == {
        "ANTHROPIC_AUTH_TOKEN": "gateway-token",
        "ANTHROPIC_BASE_URL": "https://gateway.example"}
    # Claude's token is claude's alone: no other agent receives it.
    assert "ANTHROPIC_AUTH_TOKEN" not in CodexBackend._merge_env(None)
    assert "ANTHROPIC_AUTH_TOKEN" not in KimiBackend._merge_env(None)


def test_an_endpoint_alone_is_not_a_credential(monkeypatch):
    from backend.agents.base import host_has_credentials
    _only_gateway_token(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-x")
    assert not host_has_credentials("claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert not host_has_credentials("claude")
    assert not host_has_credentials("no-such-cli")


def test_byok_claude_env_does_not_mix_in_the_host_token(monkeypatch):
    _only_gateway_token(monkeypatch)

    env = ClaudeBackend._merge_env({"ANTHROPIC_API_KEY": "user-key"})

    assert env["ANTHROPIC_API_KEY"] == "user-key"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_BASE_URL" not in env


def test_empty_overlay_still_scrubs(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "host-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "must-not-leak")

    env = CodexBackend._merge_env({})

    assert "OPENAI_API_KEY" not in env
    assert "CODEX_API_KEY" not in env
    assert "SUPABASE_SERVICE_ROLE_KEY" not in env


def test_proxy_variables_survive_the_scrub(monkeypatch):
    """Hosts whose only egress is a local proxy must pass the route on —
    withholding these leaves the agent with no route at all (Errno 101)."""
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")

    env = ClaudeBackend._merge_env({})

    assert env["https_proxy"] == "http://127.0.0.1:7890"
    assert env["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert env["no_proxy"] == "localhost,127.0.0.1"


def test_claude_overlay_run_is_path_scoped(tmp_path, monkeypatch):
    from backend.spec_worker import overlay

    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    workspace = overlay.new_run_overlay("user-a", "Qwen/Example")
    fake_proc = mock.Mock()
    fake_proc.stdin = mock.Mock()

    with mock.patch("backend.agents.claude.subprocess.Popen", return_value=fake_proc) as popen:
        ClaudeBackend().spawn(
            "prompt", env={"ANTHROPIC_API_KEY": "user-key"},
            workspace=str(workspace),
        )

    cmd = popen.call_args.args[0]
    kwargs = popen.call_args.kwargs
    assert "bypassPermissions" not in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert "Read,Write,Edit,Bash" in cmd
    assert not any(str(tmp_path / "runs") == arg for arg in cmd)
    assert kwargs["cwd"] == str(workspace.resolve())
    assert kwargs["env"]["HOME"] == str(workspace / ".agent-home")
    assert kwargs["env"]["TMPDIR"] == str(workspace / ".tmp")
    assert kwargs["env"]["HA_SPEC_OVERLAY_DIR"] == str(workspace)
    joined = "\n".join(cmd)
    assert f"Read(/{workspace / 'models'}/**)" in joined
    assert f"Write(/{workspace / 'components' / 'drafts'}/**)" in joined
    assert f"Bash({sys.executable} -m backend.inference.validate *)" in joined
    assert "DESIGN.md" not in joined


def test_claude_tool_log_keeps_full_path():
    events = ClaudeBackend().parse_event({
        "type": "assistant",
        "message": {"content": [{
            "type": "tool_use", "name": "Read",
            "input": {"file_path": "/repo/backend/arch/DESIGN.md"},
        }]},
    })
    assert events[-1]["text"] == "[Read] /repo/backend/arch/DESIGN.md"


def _spawn_overlay_backend(backend, tmp_path, monkeypatch, *, resume=None):
    from backend.spec_worker import overlay

    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    workspace = overlay.new_run_overlay("user-a", "Qwen/Example")
    fake_proc = mock.Mock()
    fake_proc.stdin = mock.Mock()
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return fake_proc

    mod = backend.__class__.__module__
    with mock.patch(f"{mod}.subprocess.Popen", side_effect=fake_popen):
        if resume is None:
            backend.spawn("prompt", env={}, workspace=str(workspace))
        else:
            backend.spawn_resume(resume, "prompt", env={}, workspace=str(workspace))
    return captured, workspace


def test_codex_overlay_run_uses_the_kind_leaf(tmp_path, monkeypatch):
    captured, workspace = _spawn_overlay_backend(CodexBackend(), tmp_path, monkeypatch)
    kwargs = captured["kwargs"]
    leaf = str(workspace.resolve())
    assert kwargs["cwd"] == leaf
    assert captured["cmd"][captured["cmd"].index("-C") + 1] == leaf
    assert kwargs["env"]["HA_SPEC_OVERLAY_DIR"] == leaf
    assert kwargs["env"]["HOME"] == str(workspace / ".agent-home")
    assert kwargs["env"]["TMPDIR"] == str(workspace / ".tmp")


def test_codex_byok_endpoint_is_explicit_for_spawn(tmp_path, monkeypatch):
    """Codex needs provider config overrides; OPENAI_BASE_URL alone is ignored."""
    from backend.spec_worker import overlay

    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    workspace = overlay.new_run_overlay("user-a", "Qwen/Example")
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        proc = mock.Mock()
        proc.stdin = mock.Mock()
        return proc

    with mock.patch("backend.agents.codex.subprocess.Popen", side_effect=fake_popen):
        CodexBackend().spawn(
            "prompt",
            env={
                "OPENAI_API_KEY": "user-key",
                "OPENAI_BASE_URL": "https://cc.example/v1",
                "OPENAI_MODEL": "gpt-user",
            },
            workspace=str(workspace),
        )

    cmd = captured["cmd"]
    env = captured["kwargs"]["env"]
    assert env["CODEX_API_KEY"] == "user-key"
    assert cmd[cmd.index("-m") + 1] == "gpt-user"
    model_provider = cmd.index('model_provider="huggingarch_user"')
    assert cmd[model_provider - 1:model_provider + 1] == [
        "-c", 'model_provider="huggingarch_user"'
    ]
    assert 'model_providers.huggingarch_user.base_url="https://cc.example/v1"' in cmd
    assert 'model_providers.huggingarch_user.wire_api="responses"' in cmd
    assert 'model_providers.huggingarch_user.requires_openai_auth=true' in cmd


def test_runner_starts_codex_and_kimi_with_target_root(tmp_path, monkeypatch):
    """Both CLIs receive target_root, start, and use that overlay leaf as cwd."""
    from backend._agent_runner import runner as R
    from backend.spec_worker import overlay

    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    workspace = overlay.new_run_overlay("user-a", "Qwen/Example", kind="generate")
    leaf = str(workspace.resolve())
    # Host runs: the host holds each agent's credential.
    monkeypatch.setenv("OPENAI_API_KEY", "host-openai")
    monkeypatch.setenv("KIMI_MODEL_API_KEY", "host-kimi")

    for cli, backend_cls in (("codex", CodexBackend), ("kimi", KimiBackend)):
        spawned = {}

        def fake_popen(cmd, **kwargs):
            spawned["cwd"] = kwargs.get("cwd")
            spawned["env"] = kwargs.get("env")
            proc = mock.Mock()
            proc.stdin = mock.Mock()
            proc.pid = 7
            proc.stdout = io.StringIO("")
            proc.stderr = io.StringIO("")
            proc.poll = mock.Mock(return_value=0)
            proc.wait = mock.Mock(return_value=0)
            proc.returncode = 0
            proc.kill = mock.Mock()
            return proc

        with mock.patch(f"{backend_cls.__module__}.subprocess.Popen",
                        side_effect=fake_popen):
            backend = backend_cls()
            monkeypatch.setattr(R, "get_backend",
                                lambda _cli, b=backend: b)
            list(R.stream_agent_core(
                key="m", prep_result={"prompt": "p", "target_root": str(workspace)},
                agent_cli=cli, log_path=str(tmp_path / f"{cli}.log"),
                log_header="h", launch_banner="B", procs_dict={},
                locks_lock=threading.Lock(), release_fn=lambda k: None,
            ))
        assert spawned["cwd"] == leaf
        assert spawned["env"]["HA_SPEC_OVERLAY_DIR"] == leaf


def test_kimi_overlay_run_uses_the_kind_leaf(tmp_path, monkeypatch):
    captured, workspace = _spawn_overlay_backend(KimiBackend(), tmp_path, monkeypatch)
    kwargs = captured["kwargs"]
    leaf = str(workspace.resolve())
    assert kwargs["cwd"] == leaf
    assert kwargs["env"]["HA_SPEC_OVERLAY_DIR"] == leaf
    assert kwargs["env"]["HOME"] == str(workspace / ".agent-home")
    assert kwargs["env"]["TMPDIR"] == str(workspace / ".tmp")
    assert "--work-dir" not in captured["cmd"]


def test_codex_resume_uses_leaf_cwd_without_dash_C(tmp_path, monkeypatch):
    """`codex exec resume` has no `-C`; the kind leaf is Popen cwd + env."""
    captured, workspace = _spawn_overlay_backend(
        CodexBackend(), tmp_path, monkeypatch, resume="thr-1")
    kwargs = captured["kwargs"]
    leaf = str(workspace.resolve())
    cmd = captured["cmd"]
    assert cmd[:3] == ["codex", "exec", "resume"]
    assert "-C" not in cmd
    assert "--cd" not in cmd
    assert "thr-1" in cmd
    assert cmd[-1] == "-"
    assert "--json" in cmd
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert kwargs["cwd"] == leaf
    assert kwargs["env"]["HA_SPEC_OVERLAY_DIR"] == leaf
    assert kwargs["env"]["HOME"] == str(workspace / ".agent-home")
    assert kwargs["env"]["TMPDIR"] == str(workspace / ".tmp")


def test_codex_resume_keeps_user_endpoint_and_model(tmp_path, monkeypatch):
    """Resume follows the same provider/auth config as the first turn."""
    from backend.spec_worker import overlay

    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    workspace = overlay.new_run_overlay("user-a", "Qwen/Example")
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        proc = mock.Mock()
        proc.stdin = mock.Mock()
        return proc

    with mock.patch("backend.agents.codex.subprocess.Popen", side_effect=fake_popen):
        CodexBackend().spawn_resume(
            "thr-1", "prompt",
            env={
                "OPENAI_API_KEY": "user-key",
                "OPENAI_BASE_URL": "https://cc.example",
                "OPENAI_MODEL": "gpt-user",
            },
            workspace=str(workspace),
        )

    cmd = captured["cmd"]
    assert cmd[:3] == ["codex", "exec", "resume"]
    assert captured["kwargs"]["env"]["CODEX_API_KEY"] == "user-key"
    assert 'model_provider="huggingarch_user"' in cmd
    assert 'model_providers.huggingarch_user.base_url="https://cc.example"' in cmd
    assert cmd[cmd.index("-m") + 1] == "gpt-user"


def test_kimi_resume_uses_leaf_cwd_and_session_flag(tmp_path, monkeypatch):
    """Kimi resume is `--session <id>` with cwd on the kind leaf, no work-dir."""
    captured, workspace = _spawn_overlay_backend(
        KimiBackend(), tmp_path, monkeypatch, resume="k-1")
    kwargs = captured["kwargs"]
    leaf = str(workspace.resolve())
    cmd = captured["cmd"]
    assert cmd[0] == "kimi"
    assert "--session" in cmd
    assert cmd[cmd.index("--session") + 1] == "k-1"
    assert "-r" not in cmd
    assert "--work-dir" not in cmd
    assert kwargs["cwd"] == leaf
    assert kwargs["env"]["HA_SPEC_OVERLAY_DIR"] == leaf
    assert kwargs["env"]["HOME"] == str(workspace / ".agent-home")
    assert kwargs["env"]["TMPDIR"] == str(workspace / ".tmp")


class _ScriptedProc:
    """Stdout JSONL fake used to drive stream_agent_core into a repair resume."""

    def __init__(self, stdout_lines, returncode=0):
        self._lines = list(stdout_lines)
        self.stderr = io.StringIO("")
        self.stdin = mock.Mock()
        self.pid = 7
        self.returncode = returncode

    @property
    def stdout(self):
        return self

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _overlay_leaf(tmp_path, monkeypatch):
    from backend.spec_worker import overlay

    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    # Host runs: the host holds each agent's credential.
    monkeypatch.setenv("OPENAI_API_KEY", "host-openai")
    monkeypatch.setenv("KIMI_MODEL_API_KEY", "host-kimi")
    workspace = overlay.new_run_overlay("user-a", "Qwen/Example", kind="generate")
    return workspace, str(workspace.resolve())


def _run_repair_stream(monkeypatch, tmp_path, backend_cls, leaf, scripted_lines):
    """Verdict-downgrade on round 1, success on round 2. `scripted_lines` is
    a list of stdout-line lists, one per Popen."""
    from backend._agent_runner import runner as R

    calls = []
    remaining = list(scripted_lines)

    def fake_popen(cmd, **kwargs):
        calls.append({"cmd": list(cmd), "cwd": kwargs.get("cwd"),
                      "env": kwargs.get("env")})
        return _ScriptedProc(remaining.pop(0))

    verdicts = iter(["generated spec failed validation: bad dim", None])
    with mock.patch(f"{backend_cls.__module__}.subprocess.Popen",
                    side_effect=fake_popen):
        monkeypatch.setattr(R, "get_backend", lambda _cli: backend_cls())
        frames = list(R.stream_agent_core(
            key="m",
            prep_result={"prompt": "p0", "target_root": leaf},
            agent_cli=backend_cls.cli_name,
            log_path=str(tmp_path / f"{backend_cls.cli_name}-repair.log"),
            log_header="h", launch_banner="B", procs_dict={},
            locks_lock=threading.Lock(), release_fn=lambda k: None,
            on_success=lambda *a: next(verdicts),
            repair_prompt_fn=lambda reason, prep: f"FIX: {reason}",
        ))
    events = [json.loads(f[len("data: "):].strip())
              for f in frames if f.startswith("data: ")]
    return calls, events


def test_runner_codex_repair_resume_keeps_leaf_cwd(tmp_path, monkeypatch):
    """thread.started + verdict downgrade must spawn_resume in the kind leaf
    with argv `codex exec resume` accepts (no `-C`)."""
    workspace, leaf = _overlay_leaf(tmp_path, monkeypatch)
    round1 = [
        json.dumps({"type": "thread.started", "thread_id": "thr-1"}) + "\n",
        json.dumps({"type": "turn.completed"}) + "\n",
    ]
    round2 = [json.dumps({"type": "turn.completed"}) + "\n"]
    calls, events = _run_repair_stream(
        monkeypatch, tmp_path, CodexBackend, leaf, [round1, round2])

    assert len(calls) == 2
    first, resume = calls
    assert first["cwd"] == leaf
    assert first["cmd"][first["cmd"].index("-C") + 1] == leaf
    assert resume["cwd"] == leaf
    assert resume["cmd"][:3] == ["codex", "exec", "resume"]
    assert "-C" not in resume["cmd"]
    assert "--cd" not in resume["cmd"]
    assert "thr-1" in resume["cmd"]
    assert resume["cmd"][-1] == "-"
    assert resume["env"]["HA_SPEC_OVERLAY_DIR"] == leaf
    assert resume["env"]["HOME"] == str(workspace / ".agent-home")
    dones = [e for e in events if e.get("type") == "done"]
    assert dones == [{"type": "done", "success": True}]
    assert any("repair round 1/2" in (e.get("text") or "")
               for e in events if e.get("type") == "log")


def test_runner_kimi_repair_resume_keeps_leaf_cwd(tmp_path, monkeypatch):
    """session.resume_hint + verdict downgrade must spawn_resume in the kind
    leaf with `--session` (not `-r`) and no `--work-dir`."""
    workspace, leaf = _overlay_leaf(tmp_path, monkeypatch)
    round1 = [
        json.dumps({"role": "assistant", "content": "ok"}) + "\n",
        json.dumps({"role": "meta", "type": "session.resume_hint",
                    "session_id": "k-1"}) + "\n",
    ]
    round2 = [
        json.dumps({"role": "assistant", "content": "fixed"}) + "\n",
    ]
    calls, events = _run_repair_stream(
        monkeypatch, tmp_path, KimiBackend, leaf, [round1, round2])

    assert len(calls) == 2
    first, resume = calls
    assert first["cwd"] == leaf
    assert "--work-dir" not in first["cmd"]
    assert resume["cwd"] == leaf
    assert resume["cmd"][0] == "kimi"
    assert "--session" in resume["cmd"]
    assert resume["cmd"][resume["cmd"].index("--session") + 1] == "k-1"
    assert "-r" not in resume["cmd"]
    assert "--work-dir" not in resume["cmd"]
    assert resume["env"]["HA_SPEC_OVERLAY_DIR"] == leaf
    assert resume["env"]["HOME"] == str(workspace / ".agent-home")
    dones = [e for e in events if e.get("type") == "done"]
    assert dones == [{"type": "done", "success": True}]
    assert any("repair round 1/2" in (e.get("text") or "")
               for e in events if e.get("type") == "log")


def test_kimi_byok_env_passes_through_as_kimi_model_family(tmp_path, monkeypatch):
    """The BYOK overlay is already named KIMI_MODEL_* (AGENT_ENV_VARS) and
    reaches the child env verbatim."""
    captured = {}
    fake_proc = mock.Mock()
    fake_proc.stdin = mock.Mock()

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return fake_proc

    with mock.patch("backend.agents.kimi.subprocess.Popen", side_effect=fake_popen):
        KimiBackend().spawn(
            "prompt",
            env={"KIMI_MODEL_API_KEY": "sk-u", "KIMI_MODEL_BASE_URL": "https://api.example/v1",
                 "KIMI_MODEL_NAME": "kimi-user-model"},
        )
    env = captured["kwargs"]["env"]
    assert env["KIMI_MODEL_NAME"] == "kimi-user-model"
    assert env["KIMI_MODEL_API_KEY"] == "sk-u"
    assert env["KIMI_MODEL_BASE_URL"] == "https://api.example/v1"


def test_host_run_in_a_private_home_without_credentials_is_refused(monkeypatch, tmp_path):
    """The overlay's HOME carries no login state, so a host run (env None)
    with no credential variable cannot authenticate: refused at spawn, as an
    agent that is not available — not left to the CLI's "please log in"."""
    import pytest
    from backend.agents.base import HostCredentialsMissing
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("backend.spec_worker.overlay.resolve_run_workspace",
                        str)
    with pytest.raises(HostCredentialsMissing):
        ClaudeBackend()._run_cwd_and_env(None, str(tmp_path))
    # A BYOK run carries its own key; the host's absence is irrelevant.
    _, env = ClaudeBackend()._run_cwd_and_env({"ANTHROPIC_API_KEY": "k"}, str(tmp_path))
    assert env["ANTHROPIC_API_KEY"] == "k"
