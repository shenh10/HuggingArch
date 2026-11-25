"""Spec-generation worker — agent CLI integration.

Public API (consumed by api, custom_model, tests):
  • Runtime configuration: configure_runtime, get_agent_cli,
    get_agent_binary, get_agent_display_name, VALID_AGENT_CLIS
  • Concurrency: acquire, release, stop
  • Prompt building: prepare_generate_prompt, prepare_refine_prompt,
    prepare_custom_generate_prompt, build_custom_prompt
  • Execution: allocate_run_overlay, stream_agent
  • Continuing a failed generation: prepare_resume

Internal layout:
  runtime         agent CLI config + per-model locks + path constants
  _metadata       HF model_info / config.json / tensor pattern fetching
  _ground_truth   ground-truth section assembly + snapshot capture
  _stream         build_prefetch_logs (spec-only; sse / usage formatting
                  live in backend._agent_runner.runner)
  _prompts/       prompt-construction subsystem
    system.py     reusable system prompt (template-driven)
    components.py component-library / op-registry / examples readers
    tasks.py      per-task bodies + public build_/prepare_ entry points
    templates/    .md templates (system.md + schema.md)
  worker          stream_agent orchestrator + post-success snapshot writer
  failover        a generate run goes on past an unavailable agent service
                  or a timed-out round
  records         where a run's generation record is kept: `spec_generations`
                  when Supabase is configured, a host-local store otherwise
  record_jobs     starting review / fix on a stored record — shared by the
                  service route and a local, database-less run (imported
                  directly, not re-exported here)
"""
from .runtime import (
    VALID_AGENT_CLIS,
    acquire,
    release,
    stop,
    start_job,
    job_status,
    configure_runtime,
    get_agent_cli,
    get_agent_binary,
    get_agent_display_name,
)
from ._ground_truth import ForwardSourceUnavailable
from ._prompts.tasks import (
    build_custom_prompt,
    prepare_generate_prompt,
    prepare_refine_prompt,
    prepare_custom_generate_prompt,
    prepare_describe_prompt,
)
from .worker import frame_event, generation_job, stream_agent
from .reviewer import prepare_review, stream_review
from .fixer import prepare_fix, stream_fix
from .overlay import allocate_run_overlay
from .resume import prepare_resume
from .failover import Failover

__all__ = [
    "VALID_AGENT_CLIS",
    "Failover",
    "ForwardSourceUnavailable",
    "acquire",
    "allocate_run_overlay",
    "build_custom_prompt",
    "configure_runtime",
    "frame_event",
    "generation_job",
    "get_agent_binary",
    "get_agent_cli",
    "get_agent_display_name",
    "job_status",
    "prepare_custom_generate_prompt",
    "prepare_describe_prompt",
    "prepare_fix",
    "prepare_generate_prompt",
    "prepare_refine_prompt",
    "prepare_resume",
    "prepare_review",
    "release",
    "start_job",
    "stop",
    "stream_agent",
    "stream_fix",
    "stream_review",
]
