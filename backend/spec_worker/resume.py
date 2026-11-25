"""Continue an earlier generation from the draft its leaf kept.

A generate run that does not succeed (out of time, out of repair rounds, or
cut off by a restart) leaves its draft — any YAML it authored — in its
generate leaf. A continued generation re-opens that leaf
(`overlay.resume_run_overlay`), takes a fresh overall budget, and hands the
agent the draft with the validator's current verdict on it:

  * when the leaf's last run recorded a session (`RunEnd`) for the same
    agent CLI, that session can take another turn (its agent did not fill
    the context window) and the CLI can re-enter it, the first round
    RESUMES the session with the continuation as its next turn;
  * otherwise the agent cold-starts on the original task prompt followed by
    the continuation.

The run is judged like a refine: an untouched draft is judged as it stands
(`seeded_spec`), not as "nothing generated".
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Optional

from backend.agents import get_backend
from backend.agents.base import AgentEnd

from . import worker as _worker
from ._prompts.tasks import _PY
from .overlay import RunEnd, _safe, authored_files, overlay_models_dir


def resumable_session(end: Optional[RunEnd], agent_cli: str,
                      target_root: str) -> Optional[str]:
    """The session the continuation re-enters: the last run's, when it ran
    the same CLI, its agent did not fill it, and that CLI still finds it in
    the leaf.

    A session whose agent filled the context window cannot take another turn
    — re-entering it overflows again — so the continuation cold-starts on the
    task and the draft instead. This is the one place that reading of an
    ending is made; `RunEnd` only records the word."""
    if end is None or not end.session_id or end.agent_cli != agent_cli:
        return None
    if end.agent_end == AgentEnd.CONTEXT_EXHAUSTED.value:
        return None
    backend = get_backend(agent_cli)
    if not backend.has_session(end.session_id, target_root):
        return None
    return end.session_id


def _verdict(model_id: str, target_root: str) -> str:
    """The validator's verdict on the draft as it stands."""
    spec = overlay_models_dir(Path(target_root)) / f"{_safe(model_id)}.yaml"
    if not spec.is_file():
        return (f"The model spec `models/{_safe(model_id)}.yaml` is not "
                "written yet. Complete the components you started and write "
                "it, building on the files above.")
    downgrade, _ = _worker._verify_spec_output(
        model_id, io.StringIO(), 0.0, root=target_root, seeded=True)
    if downgrade is None:
        return ("The draft passes validation as it stands. Re-run the "
                "validator to confirm it, then finish.")
    if downgrade.kind == "validation" and downgrade.errors:
        listed = "\n".join(f"- {error}" for error in downgrade.errors)
        return f"Current validation errors:\n{listed[:6000]}"
    return f"Current verdict: {downgrade.reason[:4000]}"


def continuation_prompt(model_id: str, target_root: str,
                        end: Optional[RunEnd],
                        request: Optional[str] = None) -> str:
    """The turn that picks the draft up: how the last run ended, the files
    it left, and the validator's verdict on them now. `request` is a change
    the user asks for on a model written from a description (a follow-up
    turn): the turn then starts from the model as it stands, whatever the
    last run's ending."""
    files = "\n".join(f"- `{name}`" for name in authored_files(target_root))
    if request is not None:
        return (
            f"# Change request for {model_id}\n\n"
            f"The model as it stands — your last turn, with any edits the "
            f"user made by hand since — is in this workspace:\n{files}\n\n"
            f"{_verdict(model_id, target_root)}\n\n"
            f"The user asks:\n\n> {request.strip()}\n\n"
            "Edit the files in place to make that change. After each edit "
            f"re-run `{_PY} -m backend.inference.validate {model_id}` and "
            "iterate until it reports PASS. Finish with a short message in "
            "the user's language: what you changed, then `Assumed:` with "
            "any value you chose that the user did not give."
        )
    if end is not None and end.timed_out:
        ended = "stopped when it reached its time budget"
    elif end is not None and end.message:
        ended = f"ended with: {end.message[:1000]}"
    else:
        ended = "stopped before it finished"
    return (
        f"# Continuation: your earlier draft of {model_id}\n\n"
        f"An earlier run of this task {ended}. Its files are in this "
        f"workspace:\n{files}\n\n"
        f"{_verdict(model_id, target_root)}\n\n"
        "Continue from these files and edit them in place. After each fix "
        f"re-run `{_PY} -m backend.inference.validate {model_id}` and iterate "
        "until it reports PASS."
    )


def prepare_resume(model_id: str, *, target_root: str, agent_cli: str,
                   base_prep: Callable[[], dict],
                   end: Optional[RunEnd] = None,
                   request: Optional[str] = None) -> dict:
    """Prep for a continued generation. `agent_cli` is the CLI this run
    spawns. `base_prep` is the original task's prep (prompt, snapshot, staged
    source); it runs only for a cold start, which needs the task prompt — a
    resumed session already holds the task, and the leaf already holds the
    staged source and snapshot. `end` is how the run being continued ended;
    by default the leaf's record of its last run (a run continued within
    the same job passes the ending it just saw)."""
    if end is None:
        end = RunEnd.read(target_root)
    session = resumable_session(end, agent_cli, target_root)
    if session:
        prep = {"prompt": continuation_prompt(model_id, target_root, end,
                                              request),
                "target_root": target_root, "prefetch_logs": []}
        note = "Continuing the earlier draft in its agent session"
    else:
        prep = base_prep()
        continuation = continuation_prompt(model_id, target_root, end,
                                           request)
        prep["prompt"] = (f"{(prep.get('prompt') or '').rstrip()}"
                          f"\n\n---\n\n{continuation}")
        note = "Continuing the earlier draft in a new agent session"
    prep.update(resume_session=session, seeded_spec=True)
    prep["prefetch_logs"] = [*(prep.get("prefetch_logs") or []), note]
    return prep
