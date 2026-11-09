"""Agent backend base class.

Each backend subclass implements two methods:
  - spawn(prompt) -> subprocess.Popen        : launch the CLI with the prompt
  - parse_event(evt) -> list[dict]           : convert one stream-json line
                                               into normalized UI events

Normalized events (consumed by stream_agent / the web UI):
  {"type": "log",   "text": str, "author"?: "agent"}
  {"type": "usage", "input_tokens": int, "output_tokens": int, ...}
  {"type": "done",  "success": bool, "message"?: str}

A `log` carrying the model's own prose (what it says, not a tool call it
makes, a connection note or an error) is marked `author: "agent"`. That
text is the agent's answer: the review runner keeps only it as the report.

Backends SHOULD call `extract_usage_events(evt)` first from usage.py so
token-accounting stays consistent across CLIs. They are NOT required to
emit a "done" event — stream_agent synthesizes one from the exit code
if the CLI closes stdout without an explicit terminator.
"""
from __future__ import annotations

import os
import re
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

from .run_user import untrusted_run_user


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Allowlist of environment variables safe to expose to an agent subprocess
# when it runs on behalf of a (potentially untrusted) user. Everything NOT on
# this list — SUPABASE_SERVICE_ROLE_KEY, HA_CREDS_ENC_KEY, SUPABASE_JWT_SECRET,
# FLY_*, and any future secret — is withheld, so a spec_gen run against a
# malicious HF repo (trust_remote_code executes arbitrary model code; the agent
# is told to read the repo's modeling source) finds no secrets in its env to
# exfiltrate. Allowlist, not denylist: a newly-added secret can't leak by
# omission. The user's own key is layered on top by `_merge_env`.
#
# The proxy variables are not credentials — on hosts whose only egress route
# is a local proxy (e.g. 127.0.0.1:7890), withholding them leaves the agent
# with NO route at all (Errno 101), which breaks the spec worker's mandated
# validate loop. Passing them through extends the host's existing egress
# policy to the sandbox instead of inventing a second one.
_SAFE_ENV_ALLOWLIST = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "TZ",
    "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR",
    "HF_HOME", "HUGGINGARCH_HF_META_CACHE_DIR",  # HF cache dirs (not secrets)
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",  # egress route
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
)


class AgentEnvVars(NamedTuple):
    """The environment variables one agent CLI reads its credential, endpoint
    and model from.

    `credentials` lists every variable the CLI itself accepts as its
    credential, the API key first. The host has credentials for the agent
    when any of them is set; an endpoint or a model name alone is not a
    credential. BYOK storage keeps one secret per row and hands it to the run
    as `key` (the first name)."""

    credentials: tuple[str, ...]
    base_url: str
    model: str

    @property
    def key(self) -> str:
        """The API-key variable — the one a stored (BYOK) secret is set as."""
        return self.credentials[0]


# The only credentials a host-managed agent may receive from the server's own
# environment. Keep this mapping in core so both subprocess construction and
# service BYOK storage use the same names. A selected agent receives only its
# own variables.
#
# Claude Code also authenticates with ANTHROPIC_AUTH_TOKEN (sent as a bearer
# token — what a gateway in front of ANTHROPIC_BASE_URL usually issues), so a
# host configured with that token and no API key has claude credentials too.
AGENT_ENV_VARS: dict[str, AgentEnvVars] = {
    "claude": AgentEnvVars(("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
                           "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"),
    "codex": AgentEnvVars(("OPENAI_API_KEY",), "OPENAI_BASE_URL", "OPENAI_MODEL"),
    "kimi": AgentEnvVars(("KIMI_MODEL_API_KEY",), "KIMI_MODEL_BASE_URL",
                         "KIMI_MODEL_NAME"),
}


def build_scrubbed_env(overlay: Optional[dict] = None) -> dict:
    """Minimal allowlist environment (no server secrets) plus an optional
    overlay. Use for ANY subprocess that runs untrusted code — a BYOK agent,
    or loading an arbitrary HF model whose trust_remote_code executes the
    repo's own Python. HOME is redirected to HA_AGENT_HOME when set (the
    unprivileged user's writable home in the container)."""
    base = {k: os.environ[k] for k in _SAFE_ENV_ALLOWLIST if k in os.environ}
    agent_home = os.environ.get("HA_AGENT_HOME")
    if agent_home:
        base["HOME"] = agent_home
    if overlay:
        base.update(overlay)
    return base


def host_agent_credentials(agent_cli: str) -> dict:
    """The selected host agent's own credential variables, from the server's
    environment — never another agent's, and never a server secret."""
    names = AGENT_ENV_VARS.get(agent_cli)
    if names is None:
        return {}
    return {name: os.environ[name]
            for name in (*names.credentials, names.base_url, names.model)
            if name in os.environ}


class HostCredentialsMissing(RuntimeError):
    """A run on the host's own credentials, for an agent whose credential
    variables the server's environment does not hold. In a private agent HOME
    the CLI has no other way to authenticate (no login state, no keychain), so
    the run is refused at spawn instead of the CLI failing on "please log in"."""

    def __init__(self, agent_cli: str):
        names = AGENT_ENV_VARS[agent_cli].credentials
        super().__init__(
            f"This server has no {agent_cli} credentials ({' / '.join(names)} "
            "unset). Add your own agent API key in account settings.")
        self.agent_cli = agent_cli


def host_has_credentials(agent_cli: str) -> bool:
    """Whether the server's environment holds a credential for `agent_cli` —
    any of its credential variables, non-empty. An endpoint or a model name
    alone does not count: the agent could not authenticate with it."""
    names = AGENT_ENV_VARS.get(agent_cli)
    return names is not None and any(os.environ.get(n) for n in names.credentials)


class AgentEnd(str, Enum):
    """Why one agent invocation ended, read from what its CLI reported (exit
    status, its error output, the HTTP status it quotes). The runner puts it
    on every terminal `done` as `agent_end`; a worker tells the agent's
    infrastructure failing from the agent failing at its task by it."""

    #: the CLI finished its turn (whatever the verdict on its output)
    OK = "ok"
    #: the run's own timer stopped it
    TIMED_OUT = "timed_out"
    #: a signal ended it that the run's timer did not send (a stop, the OOM
    #: killer, a restart)
    KILLED = "killed"
    #: the model service refused or failed the request: authentication,
    #: quota, no model configured, a 5xx, no connection — or the CLI itself
    #: could not be started
    AGENT_UNAVAILABLE = "agent_unavailable"
    #: the session's conversation no longer fits the model's context window
    #: and the CLI could not compact it: the service answered, the session
    #: is full
    CONTEXT_EXHAUSTED = "context_exhausted"
    #: the CLI reported a failure that is none of the above
    FAILED = "failed"

    @property
    def reenterable(self) -> bool:
        """Whether the session an invocation ended this way can take another
        turn. A full session cannot: re-entering it overflows again, so a
        continuation starts a new session on the task and the draft."""
        return self is not AgentEnd.CONTEXT_EXHAUSTED


def service_status(status: int) -> bool:
    """Whether an HTTP status a CLI quotes is the service's answer rather
    than a verdict on the request: authentication (401/403), payment (402),
    request timeout (408), rate or quota limit (429), and server or gateway
    errors (5xx — Cloudflare's 524 included)."""
    return status in (401, 402, 403, 408, 429) or 500 <= status < 600


def signalled(returncode: Optional[int]) -> bool:
    """Whether an exit status says a signal ended the process: negative from
    Popen, or 128+N from a CLI that re-raises the signal as its exit code
    (kimi exits 143 on SIGTERM)."""
    return returncode is not None and (returncode < 0 or 128 < returncode <= 128 + 64)


class AgentBackend(ABC):
    """Abstract CLI agent backend."""

    cli_name: str = ""          # matches SPEC_AGENT_CLI value
    display_name: str = ""      # human-readable name (shown in server logs)
    supports_isolated_workspace: bool = False
    # True when the CLI can re-enter a finished session with a follow-up turn
    # (claude: `-p --resume <id>`). The runner prefers this on repair rounds so
    # the agent fixes its output with full conversational context, instead of
    # cold-starting on a reconstructed prompt.
    supports_resume: bool = False
    #: How this CLI reports that its model service is unavailable, as
    #: patterns over its error output (the terminal error event's message
    #: and stderr). A pattern with a `status` group matches only when that
    #: HTTP status is the service's answer (`service_status`); one without
    #: names the condition itself (no credentials, quota, no connection).
    unavailable_errors: Tuple[re.Pattern, ...] = ()
    #: How this CLI reports that the session outgrew the model's context
    #: window (and compacting it failed), as patterns over the same error
    #: output. Checked before `unavailable_errors`: a CLI may quote the
    #: service error its compaction attempt met.
    context_exhausted_errors: Tuple[re.Pattern, ...] = ()

    def reports_context_exhausted(self, error_text: str) -> bool:
        """Whether `error_text` is this CLI saying its session is full."""
        return any(pattern.search(error_text or "")
                   for pattern in self.context_exhausted_errors)

    def reports_unavailable(self, error_text: str) -> bool:
        """Whether `error_text` is this CLI saying its service is unavailable."""
        for pattern in self.unavailable_errors:
            for match in pattern.finditer(error_text or ""):
                status = match.groupdict().get("status")
                if status is None or service_status(int(status)):
                    return True
        return False

    def classify_end(self, *, returncode: Optional[int], error_text: str,
                     timed_out: bool = False) -> AgentEnd:
        """Why an invocation that did not succeed ended. The run's own timer
        first (it acted, whatever the CLI printed while dying); then the
        service's own report, which also explains a CLI stopped while it
        retried a refused request; then a signal; else the CLI failed. A
        full session is the service's report too, and comes before the
        refusal its failed compaction may quote."""
        if timed_out:
            return AgentEnd.TIMED_OUT
        if self.reports_context_exhausted(error_text):
            return AgentEnd.CONTEXT_EXHAUSTED
        if self.reports_unavailable(error_text):
            return AgentEnd.AGENT_UNAVAILABLE
        if signalled(returncode):
            return AgentEnd.KILLED
        return AgentEnd.FAILED

    def extract_session_id(self, evt: dict) -> Optional[str]:
        """Pull the agent session id out of a raw stream event, when the CLI
        emits one (claude: the system/init event). Default: no sessions."""
        return None

    def spawn_resume(self, session_id: str, prompt: str,
                     env: Optional[dict] = None,
                     workspace: Optional[str] = None) -> subprocess.Popen:
        """Re-enter a previous session with `prompt` as the follow-up turn.
        Only called when supports_resume is True."""
        raise NotImplementedError

    def has_session(self, session_id: str, workspace: str) -> bool:
        """Whether `session_id` can be re-entered from `workspace` (whose
        `.agent-home` holds the CLI's state). Default: a resume-capable
        backend is trusted to find its own session."""
        return self.supports_resume and bool(session_id)

    @abstractmethod
    def spawn(self, prompt: str, env: Optional[dict] = None,
              workspace: Optional[str] = None) -> subprocess.Popen:
        """Launch the CLI subprocess fed with `prompt`. Must set stdout/stderr
        to PIPE and text=True so stream_agent can read line-by-line.

        `env` is an optional per-job overlay (e.g. a user's BYOK
        ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL). Pass it through
        `self._merge_env(env)` into Popen. None selects only this backend's
        host credentials; no agent inherits the complete server environment.

        `workspace`, when present, is the current run's private overlay. Every
        backend uses it as cwd and sets HA_SPEC_OVERLAY_DIR; the prompt already
        pins the write surface to that leaf. Backends that cannot enforce a
        path-scoped tool policy still run against the leaf — they do not claim
        isolation they cannot enforce."""

    def _run_cwd_and_env(self, env: Optional[dict],
                         workspace: Optional[str]) -> Tuple[str, dict]:
        """cwd + env for spawn. An overlay leaf becomes cwd, HOME, TMPDIR, and
        HA_SPEC_OVERLAY_DIR. PYTHONPATH stays the repo so in-loop
        validate/structure still import. No path-scoped tool policy is added
        here; Claude layers that on top in `_prepare`."""
        child_env = self._merge_env(env)
        if not workspace:
            return PROJECT_ROOT, child_env
        # The overlay leaf's HOME holds no login state, so a host run (env
        # None) authenticates only through the credential variables.
        if env is None and not host_has_credentials(self.cli_name):
            raise HostCredentialsMissing(self.cli_name)
        from backend.spec_worker.overlay import resolve_run_workspace
        run_cwd = resolve_run_workspace(workspace)
        workspace_path = Path(run_cwd)
        child_env["PYTHONPATH"] = PROJECT_ROOT
        child_env["HA_SPEC_OVERLAY_DIR"] = run_cwd
        child_env["HOME"] = str(workspace_path / ".agent-home")
        child_env["TMPDIR"] = str(workspace_path / ".tmp")
        return run_cwd, child_env

    @classmethod
    def _merge_env(cls, env: Optional[dict]) -> dict:
        """Build the subprocess environment.

        * ``env is None`` is the host/local path. Carry only the selected
          backend's credential, endpoint, and model variables from
          ``os.environ``.
        * An explicit dict is a BYOK or internal overlay and never receives
          the host's own credentials.

        Both paths start from `_SAFE_ENV_ALLOWLIST`; server secrets such as
        SUPABASE_SERVICE_ROLE_KEY and HA_CREDS_ENC_KEY are always withheld.
        An empty dict deliberately means "scrub with no credential overlay".
        """
        overlay = host_agent_credentials(cls.cli_name) if env is None else env
        return build_scrubbed_env(overlay)

    @staticmethod
    def _agent_user(env: Optional[dict]) -> Optional[str]:
        """The OS user to run the agent as — the configured `HA_AGENT_USER`
        whenever set, for BOTH untrusted (BYOK) and trusted (host/admin) runs.

        The agent must never run as root: some CLIs refuse it outright (Claude
        Code: `--dangerously-skip-permissions cannot be used with root/sudo`),
        so the trusted/admin path failed when it inherited the container's root
        uid. Dropping to a DISTINCT user also stops an untrusted agent reading
        the parent's /proc/<pid>/environ (env-scrubbing alone is bypassable);
        environment scrubbing and the uid drop now apply to every agent path.
        None on dev where the var is unset (no drop; backend not root). Requires
        the backend to run as root when set (the container does); Popen(user=…)
        raises otherwise."""
        return untrusted_run_user()

    @classmethod
    def _priv_kwargs(cls, env: Optional[dict]) -> dict:
        """Popen kwargs for the privilege drop — `{"user": <name>}` when
        configured, else `{}` (no-op on dev / trusted path)."""
        user = cls._agent_user(env)
        return {"user": user} if user else {}

    @abstractmethod
    def parse_event(self, evt: dict) -> List[dict]:
        """Convert one parsed JSON event into a list of normalized UI events."""
        raise NotImplementedError
