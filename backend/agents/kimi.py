"""Kimi Code backend."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
from typing import List, Optional

from .base import AgentBackend
from .usage import extract_usage_events


class KimiBackend(AgentBackend):
    cli_name = "kimi"
    display_name = "Kimi Code"
    # Resume rides the `session.resume_hint` meta event that print mode emits
    # at end of turn ({"type": "session.resume_hint", "session_id": ...},
    # verified on kimi-code 2.1.1) and `--session <id>` for the follow-up
    # (the same turn keeps earlier context, verified on 2.1.1). The hint's
    # suggested command mentions `-r`, but the documented flag is
    # `-S/--session`; the backend uses `--session`. Older pinned builds
    # without resume_hint never set a session id and the runner cold-starts
    # the repair prompt.
    supports_resume = True
    # Kimi Code exits 1 with `error: failed to run prompt: <reason>` on
    # stderr. A provider's refusal is `provider.<kind>: <status> <text>`
    # (`provider.auth_error: 401 Invalid Authentication`,
    # `provider.auth_error: 403 You've reached your weekly (7-day) usage
    # limit…`). An auth or connection failure is unavailable whatever the
    # status, and so is a CLI with no model to call (`No model configured.
    # Run `kimi` and use /login to sign in…`).
    unavailable_errors = (
        re.compile(r"provider\.\w+: (?P<status>\d{3})\b"),
        re.compile(r"provider\.(?:auth|connection|timeout)_error|No model configured"),
    )
    # A session that outgrew the context window is `context.overflow: …`
    # once Kimi Code's compaction gives up (kimi-code 2.1.1: `Compaction
    # failed to bring the context under the model window after N attempts.`,
    # `Compaction limit exceeded (N)`); a provider's own refusal of it can
    # surface as `provider.api_error: 400 context length exceeded`.
    #: One prompt part file. Below Read's ~100K-character cap with room for
    #: the line-number prefix it adds to every line.
    PROMPT_PART_CHARS = 60_000

    context_exhausted_errors = (
        re.compile(r"\bcontext\.overflow:|context[ _-]?length exceeded", re.I),
    )

    def _launch(self, argv: list, prompt: str, env: Optional[dict],
                workspace: Optional[str]) -> subprocess.Popen:
        run_cwd, child_env = self._run_cwd_and_env(env, workspace)
        part_paths = self._write_prompt_parts(prompt)
        proc = subprocess.Popen(
            self._argv_with_prompt(argv, self._reading_order(part_paths)),
            cwd=run_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
            **self._priv_kwargs(env),
        )
        # The files must outlive the agent's first-turn Reads, and spawn() has
        # no completion hook (the runner owns the process lifetime), so a
        # daemon reaper unlinks them once the process exits. Worst case after
        # a backend crash is a few small files left in the temp dir.
        threading.Thread(
            target=self._reap_prompt_files, args=(proc, part_paths), daemon=True,
        ).start()
        return proc

    @staticmethod
    def _argv_with_prompt(argv: list, reading_order: str) -> list:
        return [arg if arg != "@PROMPT" else reading_order for arg in argv]

    def spawn(self, prompt: str, env: Optional[dict] = None,
              workspace: Optional[str] = None) -> subprocess.Popen:
        # Why a temp file + `@file` reference instead of `-p <prompt>`:
        # spec-generation/refine prompts run tens of KB to a few hundred KB,
        # and a single argv entry caps at 128 KB (MAX_ARG_STRLEN) — past that
        # execve fails with `[Errno 7] Argument list too long`. claude and
        # codex solve this by reading the prompt from stdin, but kimi has no
        # stdin prompt path: bare `-p` errors with "argument missing" and
        # piped stdin is ignored in prompt mode. The only out-of-band channel
        # kimi supports is a file the agent Reads: `-p` carries only the list
        # of part files (`_write_prompt_parts`), and the agent's first turn
        # Reads them and follows their content as the prompt.
        #
        # Flags verified against kimi-code 2.1.1: `-p` alone is already
        # non-interactive with the auto permission policy (no approvals), and
        # `--output-format stream-json` emits one JSON per assistant/tool turn
        # plus a terminal `session.resume_hint` meta event. Do NOT re-add
        # `--print` / `--work-dir` — both are rejected as unknown options
        # (cwd already pins the working directory).
        #
        # Kimi has no path-scoped tool policy. Overlay runs still start: cwd
        # is the kind leaf, HA_SPEC_OVERLAY_DIR points there, and the prompt
        # pins the write surface. Isolation is not claimed.
        return self._launch(
            ["kimi", "-p", "@PROMPT", "--output-format", "stream-json"],
            prompt, env, workspace)

    def spawn_resume(self, session_id: str, prompt: str,
                     env: Optional[dict] = None,
                     workspace: Optional[str] = None) -> subprocess.Popen:
        """`--session <id>` re-enters the same session with the follow-up
        prompt, so a repair round keeps full context. Same spawn shape as
        `spawn` (temp-file prompt, same flags) plus the session handle.

        Workspace is Popen cwd plus HOME/TMPDIR/HA_SPEC_OVERLAY_DIR; this
        CLI has no `--work-dir`. Resume is cwd-scoped, so the kind leaf
        must be cwd on both the first turn and the follow-up.
        """
        return self._launch(
            ["kimi", "-p", "@PROMPT", "--output-format", "stream-json",
             "--session", session_id],
            prompt, env, workspace)

    def extract_session_id(self, evt: dict) -> Optional[str]:
        # kimi-code print mode ends the stream with a meta event:
        # {"role":"meta","type":"session.resume_hint","session_id":"…"}.
        if evt.get("type") == "session.resume_hint":
            sid = evt.get("session_id")
            return str(sid) if sid else None
        return None

    @classmethod
    def _write_prompt_parts(cls, prompt: str) -> List[str]:
        """The prompt, split on line boundaries into files that one Read
        returns whole. Kimi Code's Read stops at ~100K characters (2.1.1)
        and continues only if the model asks for the next offset; a model
        that re-reads without one never reaches the end of a long prompt,
        where the task and the output contract sit (Kimi-K3.1-pro review,
        2026-11-12). Parts the size of one Read leave nothing to page."""
        parts: List[str] = []
        current = ""
        for line in prompt.splitlines(keepends=True):
            if current and len(current) + len(line) > cls.PROMPT_PART_CHARS:
                parts.append(current)
                current = ""
            current += line
        parts.append(current)
        return [cls._write_prompt_file(part) for part in parts]

    @staticmethod
    def _reading_order(part_paths: List[str]) -> str:
        """The `-p` text: the part files, in order. Short enough for argv
        whatever the prompt's size."""
        listing = "\n".join(part_paths)
        return (
            f"Your complete instructions for this run are one document split "
            f"into {len(part_paths)} file(s). Read every file in full, in "
            f"this order, before you do anything else; each file is short "
            f"enough that one Read with just its path returns all of it. "
            f"The task and the required output format are at the end of the "
            f"last file.\n{listing}")

    @staticmethod
    def _write_prompt_file(prompt: str) -> str:
        # Mode 0o644, not NamedTemporaryFile's 0o600: the agent runs as the
        # unprivileged HA_AGENT_USER (a distinct uid) and must be able to read
        # the file.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            prefix="ha-kimi-prompt-", suffix=".txt", delete=False,
        ) as fh:
            fh.write(prompt)
        os.chmod(fh.name, 0o644)
        return fh.name

    @staticmethod
    def _reap_prompt_files(proc: subprocess.Popen, part_paths: List[str]) -> None:
        try:
            proc.wait()
        finally:
            for path in part_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def parse_event(self, evt: dict) -> List[dict]:
        """Each line is {role: assistant|tool, content: [...], tool_calls?: [...]}.
        No explicit 'result' event — stream_agent synthesizes done from exit
        code. `think` items are the model's internal reasoning; skip them to
        keep the UI log compact (matches how we treat Claude's thinking)."""
        results = extract_usage_events(evt)
        role = evt.get("role", "")

        if role == "assistant":
            content = evt.get("content")
            if isinstance(content, str):
                # kimi-code 0.34+ sends plain-string content; older builds
                # sent a list of typed items.
                text = content.strip()
                if text:
                    results.append(
                        {"type": "log", "text": text, "author": "agent"})
                content = []
            for item in content or []:
                if item.get("type") == "text":
                    text = (item.get("text") or "").strip()
                    if text:
                        results.append(
                            {"type": "log", "text": text, "author": "agent"})
            for tc in evt.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if name == "Shell":
                    cmd = (args.get("command") or "")[:120]
                    results.append({"type": "log", "text": f"$ {cmd}"})
                elif name in ("Read", "ReadFile"):
                    path = (args.get("path") or args.get("file_path") or "").split("/")[-1]
                    results.append({"type": "log", "text": f"[Read] {path}"})
                elif name in ("Write", "WriteFile"):
                    path = (args.get("path") or args.get("file_path") or "").split("/")[-1]
                    results.append({"type": "log", "text": f"[Write] {path}"})
                elif name in ("Edit", "EditFile"):
                    path = (args.get("path") or args.get("file_path") or "").split("/")[-1]
                    results.append({"type": "log", "text": f"[Edit] {path}"})
                elif name == "Grep":
                    results.append({"type": "log", "text": f"[Grep] {args.get('pattern', '')}"})
                elif name == "Glob":
                    results.append({"type": "log", "text": f"[Glob] {args.get('pattern', '')}"})
                else:
                    results.append({"type": "log", "text": f"[{name}]"})

        return results
