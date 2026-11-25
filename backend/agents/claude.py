"""Claude Code backend."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from .base import AgentBackend
from .usage import extract_usage_events


class ClaudeBackend(AgentBackend):
    cli_name = "claude"
    display_name = "Claude Code"
    supports_isolated_workspace = True
    supports_resume = True
    # Claude Code ends a failed request on a `result` event with
    # `is_error: true` whose text quotes the API's answer —
    # `API Error: 524 {"type":"https://developers.cloudflare.com/…error-524/"…}`,
    # `API Error: 401 {"type":"error","error":{"type":"authentication_error"…}}`
    # — or names a missing or spent account in its own words.
    unavailable_errors = (
        re.compile(r"API Error: (?P<status>\d{3})\b"),
        re.compile(r"API Error: (?:Connection error|Request timed out)"),
        re.compile(r"Invalid API key|Please run /login|OAuth token has expired"
                   r"|Credit balance is too low|usage limit reached", re.I),
    )
    # A session that outgrew the context window ends on the API's 400
    # (`prompt is too long: 213418 tokens > 200000 maximum`), or, when Claude
    # Code's own compaction of it failed, on its summary of both —
    # `Prompt is too long · automatic compaction failed: API Error: 524 …`.
    context_exhausted_errors = (re.compile(r"prompt is too long", re.I),)

    @staticmethod
    def _isolated_command(workspace: Path) -> list[str]:
        def permission_path(path: Path) -> str:
            # Claude permission rules use //... for absolute filesystem paths.
            return "/" + str(path.resolve())

        models = workspace / "models"
        drafts = workspace / "components" / "drafts"
        snapshots = workspace / "snapshots"
        references = workspace / "backend" / "arch"
        # Match the exact interpreter path embedded in the generated prompt.
        # Resolving a venv symlink here would make the allow rule miss it.
        python = sys.executable
        allowed = [
            f"Read({permission_path(models)}/**)",
            f"Read({permission_path(drafts)}/**)",
            f"Read({permission_path(workspace / 'source')}/**)",
            f"Read({permission_path(references / 'models')}/**)",
            f"Read({permission_path(references / 'components' / 'library')}/**)",
            f"Read({permission_path(references / 'components' / 'adapters')}/**)",
            f"Read({permission_path(references / 'dims.yaml')})",
            f"Write({permission_path(models)}/**)",
            f"Edit({permission_path(models)}/**)",
            f"Write({permission_path(drafts)}/**)",
            f"Edit({permission_path(drafts)}/**)",
            # A custom model described in words: its config is authored in
            # the run's snapshot, which no other run may write (a Hub
            # model's snapshot is its checkpoint record; a registered custom
            # model's config and header live in its own dir, which wins).
            f"Read({permission_path(snapshots)}/custom:*)",
            f"Write({permission_path(snapshots)}/custom:*)",
            f"Edit({permission_path(snapshots)}/custom:*)",
            f"Bash({python} -m backend.inference.validate *)",
            f"Bash({python} -m backend.arch.structure *)",
            # Which existing component already holds a graph the agent is
            # about to write as a draft — the reuse comparison, asked before
            # the spec is built around a copy.
            f"Bash({python} -m backend.arch.component_reuse *)",
        ]
        return [
            "claude", "-p", "--verbose", "--output-format", "stream-json",
            # Session state must persist (in the workspace's private HOME) so
            # the runner can `--resume` the same conversation on a repair
            # round; the overlay GC reaps it with the workspace.
            "--bare", "--safe-mode",
            "--disable-slash-commands", "--strict-mcp-config",
            "--permission-mode", "dontAsk",
            "--tools", "Read,Write,Edit,Bash",
            "--allowedTools", *allowed,
        ]

    def _prepare(self, env: Optional[dict], workspace: Optional[str],
                 resume_id: Optional[str] = None):
        """Command + cwd + env for a claude invocation. `resume_id` re-enters
        an earlier session (repair rounds) instead of starting a fresh one."""
        # Validation is strict: a workspace outside the managed overlay roots
        # raises, never silently falls back to the repo.
        run_cwd, child_env = self._run_cwd_and_env(env, workspace)
        if workspace:
            cmd = self._isolated_command(Path(run_cwd))
        else:
            # Preserve local/self-host behavior until that path also writes
            # through a private overlay. Multi-user runs use the branch above.
            cmd = ["claude", "-p", "--verbose", "--output-format", "stream-json",
                   "--permission-mode", "bypassPermissions"]
        if resume_id:
            cmd = [*cmd, "--resume", resume_id]
        return cmd, run_cwd, child_env

    def _launch(self, cmd: list, prompt: str, run_cwd: str, child_env: dict,
                env: Optional[dict]) -> subprocess.Popen:
        # Why stdin instead of `-p <prompt>`: spec-generation/refine
        # prompts carry the full prompt template + per-model ground truth
        # (config.json, forward source, tensor patterns, etc.) and run
        # tens of KB to a few hundred KB. Passing that as an argv entry
        # blows past ARG_MAX (128 KB single arg on Linux) with `[Errno 7]
        # Argument list too long: 'claude'`. Bare `-p` reads from stdin.
        proc = subprocess.Popen(
            cmd,
            cwd=run_cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
            **self._priv_kwargs(env),
        )
        if proc.stdin is not None:
            proc.stdin.write(prompt)
            proc.stdin.close()
        return proc

    def spawn(self, prompt: str, env: Optional[dict] = None,
              workspace: Optional[str] = None) -> subprocess.Popen:
        cmd, run_cwd, child_env = self._prepare(env, workspace)
        return self._launch(cmd, prompt, run_cwd, child_env, env)

    def spawn_resume(self, session_id: str, prompt: str,
                     env: Optional[dict] = None,
                     workspace: Optional[str] = None) -> subprocess.Popen:
        """Continue the run's own session: verdict-gate errors arrive as a
        follow-up turn with full conversational context, not a cold
        reconstruction."""
        cmd, run_cwd, child_env = self._prepare(env, workspace,
                                                resume_id=session_id)
        return self._launch(cmd, prompt, run_cwd, child_env, env)

    def has_session(self, session_id: str, workspace: str) -> bool:
        """Claude keeps each session's transcript under the run HOME's
        `.claude/projects/<encoded cwd>/<session>.jsonl`."""
        if not session_id or Path(session_id).name != session_id:
            return False
        projects = Path(workspace) / ".agent-home" / ".claude" / "projects"
        if not projects.is_dir():
            return False
        name = f"{session_id}.jsonl"
        return any((project / name).is_file()
                   for project in projects.iterdir() if project.is_dir())

    def extract_session_id(self, evt: dict) -> Optional[str]:
        if evt.get("type") == "system" and evt.get("subtype") == "init":
            sid = evt.get("session_id")
            return str(sid) if sid else None
        return None

    def parse_event(self, evt: dict) -> List[dict]:
        results = extract_usage_events(evt)
        etype = evt.get("type", "")

        if etype == "system":
            sub = evt.get("subtype", "")
            if sub == "init":
                results.append({"type": "log", "text": "Coding agent connected"})
            elif sub == "api_retry":
                attempt = evt.get("attempt", 0)
                results.append({"type": "log", "text": f"API retry #{attempt}..."})

        elif etype == "assistant":
            content = evt.get("message", {}).get("content", [])
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "").strip()
                    if text:
                        results.append(
                            {"type": "log", "text": text, "author": "agent"})
                elif item.get("type") == "tool_use":
                    tool = item.get("name", "")
                    inp = item.get("input", {})
                    if tool == "Bash":
                        cmd = inp.get("command", "")[:120]
                        results.append({"type": "log", "text": f"$ {cmd}"})
                    elif tool in ("Read", "Write", "Edit"):
                        # A basename-only log cannot be audited for path escape.
                        path = inp.get("file_path", "")
                        results.append({"type": "log", "text": f"[{tool}] {path}"})
                    elif tool == "Grep":
                        results.append({"type": "log", "text": f"[Grep] {inp.get('pattern', '')}"})
                    elif tool == "Glob":
                        results.append({"type": "log", "text": f"[Glob] {inp.get('pattern', '')}"})

        elif etype == "result":
            is_error = evt.get("is_error", False)
            if not is_error:
                results.append({"type": "done", "success": True})
            else:
                results.append({"type": "done", "success": False, "message": evt.get("result", "Failed")[:200]})

        return results
