"""Core SSE driver for agent-CLI workers.

`stream_agent_core` is the single source of truth for:
  • spawning the configured agent backend (claude / codex / kimi)
  • draining its stdout JSON events into normalized SSE frames
  • writing the per-run trace log
  • routing the per-key subprocess into `procs_dict` so `stop()` can kill it
  • calling `release_fn` from a finally block on every termination path

Worker-specific concerns (post-success snapshot writes, Supabase
publishes, etc.) are pushed into the optional `on_success` callback.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import IO, Callable, Generator, Optional

from backend.agents import get_backend
from backend.agents.base import AgentEnd, HostCredentialsMissing

from .scrub import scrub_paths


logger = logging.getLogger(__name__)


# `on_success(key, log_file, started_at, prep_result) -> Optional[str]` —
# fires once when the agent emits a `done` event with `success=True`, or when
# the proc exits with code 0 without a done event. Used by spec_worker to write
# the HF snapshot + publish to the cross-VM Supabase cache. It may also RETURN a
# downgrade reason: a non-empty string means "the agent exited cleanly but did
# not actually produce a valid result", and the runner rewrites the emitted
# `done` to success=False with that message (a clean exit alone is too forgiving
# — e.g. codex's sandbox can fail to launch, so the agent exits 0 having written
# nothing). Return None to keep the success verdict. Best-effort side effects:
# exceptions inside are the caller's problem, the runner does not catch.
# What the hook leaves in `prep_result["terminal"]` (a dict) is added to the
# run's terminal `done` event: the failure's machine fields, or what became of
# the run's stored result.
#
# It also fires on a round its OWN timer stopped while the run's deadline is
# still ahead: the agent was killed, and the hook judges the workspace as the
# round left it. Its side effects (spec publish, a fix's durable write-back)
# therefore also run for a force-killed round whose output passes.
OnSuccess = Callable[[str, IO, float, dict], Optional[str]]


def sse(data: dict) -> str:
    """Format `data` as a single Server-Sent Events frame."""
    return f"data: {json.dumps(data)}\n\n"


def format_usage_log(data: dict) -> str:
    """Format a usage event for the worker log."""
    parts = []
    if data.get("input_tokens") is not None:
        parts.append(f"in={data['input_tokens']:,}")
    if data.get("output_tokens") is not None:
        parts.append(f"out={data['output_tokens']:,}")
    if data.get("cache_creation_input_tokens") is not None:
        parts.append(f"cache_create={data['cache_creation_input_tokens']:,}")
    if data.get("cache_read_input_tokens") is not None:
        parts.append(f"cache_read={data['cache_read_input_tokens']:,}")
    if data.get("total_tokens") is not None:
        parts.append(f"total={data['total_tokens']:,}")
    return "usage: " + ", ".join(parts) if parts else ""


def stream_agent_core(
    *,
    key: str,
    prep_result: dict,
    agent_cli: str,
    log_path: str,
    log_header: str,
    launch_banner: str,
    procs_dict: dict,
    locks_lock: threading.Lock,
    release_fn: Callable[[str], None],
    on_success: Optional[OnSuccess] = None,
    env: Optional[dict] = None,
    hard_cap_sec: Optional[float] = None,
    deadline_at: Optional[float] = None,
    round_budget_sec: Optional[float] = None,
    repair_prompt_fn: Optional[Callable[[str, dict], Optional[str]]] = None,
    max_repair_rounds: int = 2,
    owns: Optional[Callable[[str], bool]] = None,
    timeout_hint: Optional[str] = None,
    round_timeout_hint: Optional[str] = None,
    resume_session: Optional[str] = None,
    append_log: bool = False,
) -> Generator[str, None, None]:
    """Run the configured agent CLI and stream normalized SSE events.

    `prep_result` is what the per-worker `prep_*` functions return — a
    dict with `prompt`, optional `prefetch_logs`, and any extra fields
    the `on_success` hook needs (e.g. spec_worker reads `snapshot_data`).

    `key` is the per-worker concurrency key (model_id or op_name). It is
    used to register the subprocess in `procs_dict` (so `stop(key)` can
    kill it), to pass through to `release_fn`, and to pass to
    `on_success`.

    `owns(key)`, when given, answers whether this run still holds the key. A
    run whose lock was reclaimed while it was stalled must not start an agent
    or register one over the run that replaced it; it is checked before the
    spawn and again, under `locks_lock`, when registering the process.

    The finally block always calls `release_fn(key)` — defense in depth.
    Callers release too (the spec workers' own `finally`, the op-proposal
    route's SSE wrapper), so `release_fn` must be idempotent
    (BoundedSemaphore + pop-with-default).

    Repair rounds: when `on_success` downgrades a clean exit (a verdict-gate
    failure — e.g. the written spec fails validation), `repair_prompt_fn` may
    turn the downgrade reason into a follow-up prompt; the agent is then
    re-spawned IN THE SAME WORKSPACE to fix its own output, and the verdict
    gate runs again. When the backend supports sessions (`supports_resume`),
    the follow-up RESUMES the same conversation — the errors arrive as the
    next turn with full context, not as a cold reconstruction. The
    concurrency slot and log file stay held across rounds, no terminal `done`
    is emitted for a round that gets repaired, and at most
    `max_repair_rounds` repairs happen before the failure is emitted.
    Default (no `repair_prompt_fn`): exactly one round, unchanged behavior.

    `hard_cap_sec` is the wall-clock budget for the whole stream, repairs
    included. A run stopped by it ends on a `done` with `timed_out: True`. The timer starts after the first spawn (Hugging Face prep is
    not charged). Each later round, and a caller that passes `deadline_at`
    (op-continue under the same generate lock), gets the remaining time
    until that deadline — not a fresh full cap. The deadline is written
    onto `prep_result["_run_deadline_at"]` so the holder can reuse it.

    `round_budget_sec` gives each round — the first and every repair round —
    a timer of its own: a round is stopped at
    `min(now + round_budget_sec, overall deadline)`, so one round cannot spend
    the whole run's budget and a repair round still gets a full round when
    the overall deadline allows it. None: a round's only limit is the
    overall deadline.

    A round stopped by its own timer while the overall deadline still has
    time left is judged like a clean exit: `on_success` runs on what the
    agent wrote so far, on every round including the last. A spec that
    passes is the run's success. A downgrade becomes the next repair round
    (same session when the backend supports it) when a repair round is left
    and `repair_prompt_fn` turns it into a prompt — with
    `prep_result["_round_timed_out"]` set so the prompt can say the round
    ran out of time; otherwise the run ends on the timeout `done`. A stop at
    the overall deadline always ends the run unjudged.

    `resume_session` re-enters an earlier session on the FIRST round (a
    continued generation); repair rounds resume their own round's session.
    The current session id is kept on `prep_result["_session_id"]` so the
    holder can record it for a later continuation.

    A failed terminal `done` whose agent did not finish its turn says why,
    as `agent_end` (an `AgentEnd` value the backend classifies from the
    CLI's exit status and error output): `timed_out` when a timer here
    stopped it, else `agent_unavailable`, `killed` or `failed`. No
    `agent_end` means the CLI finished (`ok`) — a verdict downgrade
    included: the spec failing is not the agent failing. The holder decides
    what to do about an unavailable service or a stopped round
    (`spec_worker.failover`).

    `append_log` continues `log_path` instead of starting it: a holder that
    runs the agent again within the same run keeps one log.
    """
    latest_usage = None

    def _duration(seconds: float) -> str:
        if seconds >= 3600 and seconds % 3600 == 0:
            hours = seconds / 3600
            return f"{hours:g} hour" + ("" if hours == 1 else "s")
        if seconds >= 60 and seconds % 60 == 0:
            minutes = seconds / 60
            return f"{minutes:g} minute" + ("" if minutes == 1 else "s")
        return f"{seconds:g} seconds"

    # Which timer ends the current round: its own (`round_budget_sec`) or
    # the run's overall deadline, whichever comes first — and so which one
    # a timeout reports.
    round_timer_binds = [False]

    def _timeout_message() -> str:
        if round_timer_binds[0]:
            msg = (f"An agent round exceeded its budget of "
                   f"{_duration(float(round_budget_sec))} and was stopped")
            hint = round_timeout_hint
        else:
            msg = (f"Agent execution exceeded "
                   f"{_duration(float(hard_cap_sec or 0))} and was stopped")
            hint = timeout_hint
        if hint:
            msg += f". Retry, or raise {hint}"
        return msg

    # Wall-clock checkpoint for the WHOLE stream, repair rounds included:
    # on_success treats artifacts with mtime >= started_at as produced by this
    # run, and a repair round edits the same files the first round wrote.
    started_at = time.time()
    run_deadline_at = deadline_at

    log_file = open(log_path, "a" if append_log else "w")  # noqa: SIM115 - stays open across rounds; closed in the outer finally
    log_file.write(f"{log_header}\n")

    done_seen = [False]

    # Frames leave the process (job buffer → /spec-status polling), so they
    # are scrubbed: a per-run overlay root embeds the deploy home AND, for
    # signed-in runs, the triggering user's UUID. The trace log keeps raw
    # text as the operator's debug artifact; the DB-bound copy uploaded for
    # the run's owner is scrubbed separately at upload time.
    display_root = prep_result.get("target_root")

    def _log_and_yield(data: dict) -> str:
        etype = data.get("type", "?")
        text = data.get("text", data.get("message", ""))
        if etype == "usage":
            text = format_usage_log(data)
        # Always record `done` events even when their text is empty —
        # otherwise a silent exit-0 with no stdout leaves a log that
        # just stops at "=== END PROMPT ===" with no diagnostic line.
        if etype == "done":
            done_seen[0] = True
            if not text:
                text = f"success={data.get('success', False)}"
        if text:
            log_file.write(f"[{etype}] {text}\n")
            log_file.flush()
        out = data
        if etype in ("log", "done"):
            out = dict(data)
            for field in ("text", "message"):
                value = out.get(field)
                if isinstance(value, str) and value:
                    out[field] = scrub_paths(value, display_root)
        return sse(out)

    def _judge_timed_out_round(session_id: Optional[str]):
        """Verdict for a round its own timer stopped while the run's deadline
        still has time: None when the round is not judged (the run's deadline
        bound, or no verdict hook) — the caller emits the timeout `done`.
        Otherwise the outcome of `on_success` on the workspace as the agent
        left it; whether a downgrade is repaired is the caller's decision."""
        if not (round_timer_binds[0] and on_success is not None):
            return None
        if run_deadline_at is not None and time.time() >= run_deadline_at:
            return None
        log_file.write("[round-timeout] judging the workspace as the round left it\n")
        log_file.flush()
        downgrade = on_success(key, log_file, started_at, prep_result)
        return {"downgrade": downgrade, "session_id": session_id,
                "timed_out": True, "message": _timeout_message()}

    def _attempt(prompt: str, resume_session: Optional[str] = None):
        """One spawn→stream→verdict round. Yields SSE frames. When the round
        ends in a verdict DOWNGRADE (agent exited cleanly but on_success
        rejected its output), no terminal `done` is emitted — the round
        returns {"downgrade": reason, "session_id": ...} and the caller
        decides between a repair round and emitting the failure itself. All
        other terminations yield their own done and return None. A judged
        round its own timer stopped returns a downgrade the same way, marked
        `timed_out`.

        `resume_session` (a session id from an earlier round) re-enters that
        conversation when the backend supports it, so a repair round fixes
        the output with full context instead of cold-starting."""
        nonlocal latest_usage, run_deadline_at
        timed_out = threading.Event()
        reaper: Optional[threading.Timer] = None
        proc: Optional[subprocess.Popen] = None
        session_id = resume_session
        try:
            yield _log_and_yield({"type": "log", "text": launch_banner})

            log_file.write(f"\n=== PROMPT ===\n{prompt}\n=== END PROMPT ===\n\n")
            log_file.flush()

            if owns is not None and not owns(key):
                raise RuntimeError(f"{key}: this run was superseded (its lock was reclaimed); not starting the agent")
            backend = get_backend(agent_cli)
            workspace = prep_result.get("target_root")
            # Overlay leaf is always passed when present. Claude enforces a
            # path-scoped tool policy; Codex/Kimi cannot, but they still run
            # with cwd and HA_SPEC_OVERLAY_DIR on that leaf. Refusing them
            # here made signed-in generate/refine fail for those CLIs.
            if resume_session and backend.supports_resume:
                proc = backend.spawn_resume(resume_session, prompt, env=env,
                                            workspace=workspace)
            elif workspace:
                proc = backend.spawn(prompt, env=env, workspace=workspace)
            else:
                proc = backend.spawn(prompt, env=env)
            log_file.write(f"[spawn] pid={proc.pid} cli={agent_cli}\n")
            log_file.flush()
            with locks_lock:
                superseded = owns is not None and not owns(key)
                if not superseded:
                    procs_dict[key] = proc
            if superseded:
                proc.kill()
                raise RuntimeError(f"{key}: this run was superseded while starting; agent stopped")

            # Start the execution cap only after metadata/prompt preparation and
            # subprocess spawn. Large Hugging Face repos can spend several minutes
            # in preparation; charging that time to the coding agent caused healthy
            # runs to be killed early and surface as an opaque exit code -9.
            # The cap is for the WHOLE stream: later repair rounds (and an
            # op-continue that reuses `deadline_at`) inherit the remaining
            # time instead of a fresh full budget.
            now = time.time()
            if run_deadline_at is None and hard_cap_sec is not None and hard_cap_sec > 0:
                run_deadline_at = now + float(hard_cap_sec)
            if run_deadline_at is not None:
                prep_result["_run_deadline_at"] = run_deadline_at
            # This round's own timer, inside the run's deadline.
            round_deadline_at = run_deadline_at
            round_timer_binds[0] = False
            if round_budget_sec is not None and round_budget_sec > 0:
                own = now + float(round_budget_sec)
                if round_deadline_at is None or own < round_deadline_at:
                    round_deadline_at = own
                    round_timer_binds[0] = True
            if round_deadline_at is not None:

                def _reap_agent() -> None:
                    if proc is not None and proc.poll() is None:
                        timed_out.set()
                        proc.kill()

                remaining = round_deadline_at - time.time()
                if remaining <= 0:
                    timed_out.set()
                    if proc.poll() is None:
                        proc.kill()
                else:
                    reaper = threading.Timer(remaining, _reap_agent)
                    reaper.daemon = True
                    reaper.start()

            stderr_lines: list = []

            def _drain_stderr():
                if proc.stderr is None:
                    return
                for raw in proc.stderr:
                    line = raw.rstrip()
                    if line:
                        stderr_lines.append(line)

            stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
            stderr_thread.start()

            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    break
                line = line.rstrip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                    sid = backend.extract_session_id(evt)
                    if sid:
                        session_id = sid
                        prep_result["_session_id"] = sid
                    round_stopped = False
                    for parsed in backend.parse_event(evt):
                        if parsed.get("type") == "usage":
                            if parsed == latest_usage:
                                continue
                            latest_usage = dict(parsed)
                        elif parsed.get("type") == "done" and latest_usage:
                            parsed = {**parsed, "usage": latest_usage}
                        if parsed.get("type") == "done" and timed_out.is_set():
                            # The timer stopped this round: the exit path
                            # below judges and reports it.
                            round_stopped = True
                            break
                        # Run on_success BEFORE emitting the done so its verdict
                        # can intercept a falsely-'success' agent exit. A
                        # downgrade is repairable — defer to the caller.
                        if (parsed.get("type") == "done" and parsed.get("success")
                                and on_success is not None):
                            downgrade = on_success(key, log_file, started_at, prep_result)
                            if downgrade:
                                return {"downgrade": downgrade,
                                        "session_id": session_id}
                            parsed = {**parsed,
                                      **(prep_result.get("terminal") or {})}
                        if (parsed.get("type") == "done"
                                and not parsed.get("success")):
                            # The CLI's own failure event (claude's
                            # `is_error` result, codex's `turn.failed`).
                            parsed = {**parsed, "agent_end": backend.classify_end(
                                returncode=proc.poll(),
                                error_text="\n".join(
                                    [str(parsed.get("message") or ""),
                                     *stderr_lines[-20:]])).value}
                        yield _log_and_yield(parsed)
                        if parsed.get("type") == "done":
                            return None
                    if round_stopped:
                        break
                except json.JSONDecodeError:
                    # Keep non-JSON output for debugging, but don't spam the UI.
                    log_file.write(f"[stdout] {line}\n")
                    log_file.flush()

            proc.wait(timeout=600)
            elapsed = time.time() - started_at
            log_file.write(f"[exit] code={proc.returncode}  elapsed={elapsed:.1f}s\n")
            log_file.flush()
            if timed_out.is_set():
                verdict = _judge_timed_out_round(session_id)
                if verdict is not None and verdict["downgrade"]:
                    return verdict
                if verdict is not None:
                    done_data = {"type": "done", "success": True,
                                 **(prep_result.get("terminal") or {})}
                else:
                    done_data = {
                        "type": "done",
                        "success": False,
                        "message": _timeout_message(),
                        "timed_out": True,
                        "agent_end": AgentEnd.TIMED_OUT.value,
                    }
                if latest_usage:
                    done_data["usage"] = latest_usage
                yield _log_and_yield(done_data)
            elif proc.returncode == 0:
                downgrade = (
                    on_success(key, log_file, started_at, prep_result)
                    if on_success is not None else None
                )
                if downgrade:
                    return {"downgrade": downgrade, "session_id": session_id}
                done_data = {"type": "done", "success": True,
                             **(prep_result.get("terminal") or {})}
                if latest_usage:
                    done_data["usage"] = latest_usage
                yield _log_and_yield(done_data)
            else:
                stderr_thread.join(timeout=5)
                stderr = "\n".join(stderr_lines[-20:])[-1000:]
                done_data = {
                    "type": "done",
                    "success": False,
                    "message": f"Exit code {proc.returncode}: {stderr}",
                    "agent_end": backend.classify_end(
                        returncode=proc.returncode, error_text=stderr).value,
                }
                if latest_usage:
                    done_data["usage"] = latest_usage
                yield _log_and_yield(done_data)
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
            done_data = {"type": "done", "success": False,
                         "message": "Timed out (600s)",
                         "agent_end": AgentEnd.FAILED.value}
            if latest_usage:
                done_data["usage"] = latest_usage
            yield _log_and_yield(done_data)
        except Exception as e:
            # A CLI that cannot be started (not installed, not executable, no
            # credentials to start it with) is an agent that is not available;
            # anything else failed here.
            end = (AgentEnd.AGENT_UNAVAILABLE
                   if proc is None and isinstance(e, (OSError, HostCredentialsMissing))
                   else AgentEnd.FAILED)
            done_data = {"type": "done", "success": False, "message": str(e),
                         "agent_end": end.value}
            if latest_usage:
                done_data["usage"] = latest_usage
            yield _log_and_yield(done_data)
        finally:
            if reaper is not None:
                reaper.cancel()
            # A round that returns on its own `done` or a repairable downgrade
            # leaves the CLI to exit by itself; reap it before the next round
            # spawns into the same workspace (or resumes the same session),
            # and kill it if it has not exited.
            if proc is not None and proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                if proc.poll() is None:
                    proc.kill()
            if proc is not None:
                # A reaped agent is no key's running process: `stop` must not
                # find its handle between this round and the next spawn.
                with locks_lock:
                    if procs_dict.get(key) is proc:
                        del procs_dict[key]
        return None

    try:
        for text in prep_result.get("prefetch_logs", []) or []:
            yield _log_and_yield({"type": "log", "text": text})

        prompt = prep_result.get("prompt", "")
        repair_round = 0
        if resume_session:
            prep_result["_session_id"] = resume_session
        while True:
            may_repair = (repair_prompt_fn is not None
                          and repair_round < max_repair_rounds)
            outcome = yield from _attempt(prompt, resume_session)
            if outcome is None:
                return
            reason = outcome["downgrade"]
            round_timed_out = bool(outcome.get("timed_out"))
            prep_result["_round_timed_out"] = round_timed_out
            # A backend with session support re-enters the SAME conversation
            # next round; others cold-start on the repair prompt.
            resume_session = outcome.get("session_id")
            repair = (
                repair_prompt_fn(reason, prep_result) if may_repair else None
            )
            if not repair:
                if round_timed_out:
                    done_data = {"type": "done", "success": False,
                                 "message": outcome["message"],
                                 "timed_out": True}
                else:
                    done_data = {"type": "done", "success": False,
                                 "message": reason}
                done_data.update(prep_result.get("terminal") or {})
                if round_timed_out:
                    done_data["agent_end"] = AgentEnd.TIMED_OUT.value
                if latest_usage:
                    done_data["usage"] = latest_usage
                yield _log_and_yield(done_data)
                return
            repair_round += 1
            if round_timed_out:
                text = (f"The round reached its time budget — sending the "
                        f"current validation errors back to the agent to "
                        f"continue in place "
                        f"(repair round {repair_round}/{max_repair_rounds})")
            else:
                text = (f"Verdict gate rejected the output — sending the "
                        f"errors back to the agent to fix in place "
                        f"(repair round {repair_round}/{max_repair_rounds})")
            yield _log_and_yield({"type": "log", "text": text})
            prompt = repair
    finally:
        # If no `done` was emitted, the generator was aborted — usually
        # because the SSE client disconnected and Python GC'd the
        # generator before it could run its done path. Leave a breadcrumb
        # so the log file tells the story instead of ending silently.
        if not log_file.closed:
            if not done_seen[0]:
                try:
                    log_file.write(
                        "[finally] stream aborted before done "
                        "(client disconnect or GC)\n"
                    )
                    log_file.flush()
                except OSError:
                    pass
            log_file.close()
        release_fn(key)
