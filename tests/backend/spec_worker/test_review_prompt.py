"""Contract facts of the spec review prompt (not its wording).

The review agent's answer is parsed by `reviewer._extract`, so the block the
prompt asks for must round-trip through that parser; every spec-gen task that
self-checks (generate, refine, custom) carries the same forward-alignment
standard the reviewer applies; and the full source staged for citation lands
in the run overlay and is named in every consumer's prompt; forward excerpts are
never staged as source files.
"""
import re

from backend.spec_worker._ground_truth import full_module_sources
from backend.spec_worker._prompts.fix import build_fix_task_body
from backend.spec_worker._prompts.review import build_review_task_body
from backend.spec_worker._prompts.system import forward_alignment_doc
from backend.spec_worker._prompts.tasks import (
    build_generate_task_body,
    build_refine_task_body,
)
from backend.spec_worker.overlay import stage_forward_source
from backend.spec_worker.review_evidence import ReviewEvidence, SnapshotFacts
from backend.spec_worker.reviewer import _extract

_NO_FACTS = SnapshotFacts.empty()


def _review_body(**kw):
    return build_review_task_body("org/model", "a: 1", [], {"valid": True},
                                  facts=_NO_FACTS, **kw)


def _fix_body(**kw):
    return build_fix_task_body("org/model", "a: 1", [], {"valid": True},
                               ReviewEvidence.audit("r", _NO_FACTS), facts=_NO_FACTS, **kw)


def test_requested_report_block_round_trips_through_parser():
    body = _review_body()
    m = re.search(r"===REVIEW REPORT START===.*?===REVIEW REPORT END===", body, re.S)
    assert m, "prompt must show the sentinel-wrapped report block"
    block = m.group(0)
    assert "VERDICT: {verdict}" in block
    for token, parsed in (("PASS", "pass"), ("NO_PASS", "no_pass")):
        assert _extract(block.replace("{verdict}", token))[0] == parsed


def test_every_self_check_shares_the_review_standard():
    doc = forward_alignment_doc()
    assert doc in _review_body()
    assert doc in build_generate_task_body("org/model", {})
    assert doc in build_refine_task_body("org/model", 1, 2, -1, {})


def test_generate_refine_and_fix_pin_the_identity_filename():
    target = "/overlay/models/org__model.yaml"
    assert target in build_generate_task_body(
        "org/model", {}, target_root="/overlay")
    assert target in build_refine_task_body(
        "org/model", 1, 2, -1, {}, target_root="/overlay")
    assert target in _fix_body(target_root="/overlay")


def test_staged_source_is_readable_in_overlay_and_named_in_prompt(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    modeling = pkg / "modeling_x.py"
    modeling.write_text("class X:\n    def forward(self, h):\n        return h\n")
    (pkg / "configuration_x.py").write_text("class XConfig:\n    norm_topk_prob = False\n")
    overlay = tmp_path / "overlay"
    staged = stage_forward_source(
        overlay, [{"path": str(modeling), "text": modeling.read_text()}])
    names = sorted(p.rsplit("/", 1)[-1] for p in staged)
    assert names == ["configuration_x.py", "modeling_x.py"]
    assert (overlay / "source" / "modeling_x.py").read_text() == modeling.read_text()
    for body in (
        _review_body(source_files=staged),
        build_generate_task_body("org/model", {}, source_files=staged),
        build_refine_task_body("org/model", 1, 2, -1, {}, source_files=staged),
        _fix_body(source_files=staged),
    ):
        assert all(p in body for p in staged)


def test_forward_excerpt_cache_is_never_staged_as_source(monkeypatch):
    # The model_info cache holds extracted forward excerpts in a .json, not a
    # module file: the real modeling files are fetched instead.
    import backend.hf.forward_source as fsrc
    excerpt = {"source": "model_info_cache",
               "files": [{"path": "/cache/org__m.json", "text": "# ==== X ===="}]}
    real = [{"path": "/hf/modeling_m.py", "text": "class M: ..."}]
    monkeypatch.setattr(fsrc, "_try_hf_modeling_files", lambda *_: real)
    assert full_module_sources("org/m", {}, excerpt) == real


def test_review_and_fix_commands_are_ones_the_agent_may_run(tmp_path):
    # The isolated claude run allows Bash only for the exact commands in
    # `ClaudeBackend._isolated_command`; a prompt that spells them differently
    # (bare `python`, a `cd` first) gets every call denied.
    from fnmatch import fnmatchcase
    from pathlib import Path

    from backend.agents.claude import ClaudeBackend

    cmd = ClaudeBackend._isolated_command(Path(tmp_path))
    allowed = [a[len("Bash("):-1] for a in cmd if a.startswith("Bash(")]
    assert allowed
    for body in (_review_body(), _fix_body()):
        commands = re.findall(r"`([^`]*-m backend\.(?:arch|inference)\.[^`]*)`", body)
        assert commands
        for c in commands:
            assert any(fnmatchcase(c, rule) for rule in allowed), c
        # ... and every command the sandbox allows is one the prompt names,
        # so a rule added there cannot go unsaid here.
        for rule in allowed:
            assert any(fnmatchcase(c, rule) for c in commands), rule


def test_prompt_shell_description_is_the_sandbox_policy(tmp_path):
    # The prompt once said every other shell command is denied, but under
    # `dontAsk` Claude Code runs read-only commands inside the working
    # directory, and an agent reported its own `ls` as a breach.
    # The description the prompts carry is `shell_rules`, and it has to say
    # what the isolated command's mode and rules do.
    from pathlib import Path

    from backend.agents.claude import ClaudeBackend
    from backend.spec_worker._prompts.tasks import shell_rules

    cmd = ClaudeBackend._isolated_command(Path(tmp_path))
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    rules = shell_rules("org/model")
    assert "only read-only commands" in rules
    assert "inside this working directory run" in rules
    assert "reads outside this working directory, is denied" in rules
    # Writes go through the file tools, only on the fix prompt's edit surface
    # and the config of a custom model described in words (its snapshot).
    writable = {a[a.index("(") + 1:-1] for a in cmd
                if a.startswith(("Write(", "Edit("))}
    root = "/" + str(Path(tmp_path).resolve())
    assert writable == {f"{root}/models/**", f"{root}/components/drafts/**",
                        f"{root}/snapshots/custom:*"}
    for body in (_review_body(), _fix_body()):
        assert rules in body
