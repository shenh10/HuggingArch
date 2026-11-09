"""Shared test fixtures — session-scoped loader and snapshot resolution.

`spec_loader` gives every test a single SpecLoader instance so component /
model registries are only built once per session.

`resolved(model_id)` returns a ResolvedSpec for the given model_id, or
raises pytest.skip if the snapshot file is missing, or pytest.fail if the
file is present but does not resolve. This distinction prevents a missing
file from silently turning a test green, while still allowing corpus tests
to skip models not checked out locally.
"""
from __future__ import annotations

import io
import os
import pickle
import sys
import tempfile

import pytest
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Trace bodies a test session records (`backend.code_trace.provider.TRACES`)
# stay out of the user's cache; set before anything under `backend` imports.
os.environ.setdefault("HA_CODE_TRACE_DIR",
                      tempfile.mkdtemp(prefix="ha-code-traces-"))


class _ParsedYaml:
    """`yaml.safe_load`, parsed once per distinct text in a test process.

    The pure-Python parser is the cost of building a `SpecLoader` (~90% of
    it), and product code a test drives builds its own loaders (`validate`,
    the sweep, the invariant battery) — a test cannot hand those a shared
    one. The same text parses to the same document, so the first parse is
    kept as a pickle and every call gets a fresh copy: a caller that edits
    the document it was handed edits only its own. A stream is read and
    re-offered under its own name, so a parse error reads as before; errors
    are never kept."""

    def __init__(self, parse):
        self._parse = parse
        self._parsed: dict = {}

    def __call__(self, stream):
        if isinstance(stream, (str, bytes)):
            text = source = stream
        else:
            text = stream.read()
            source = io.StringIO(text) if isinstance(text, str) else io.BytesIO(text)
            source.name = getattr(stream, "name", "<file>")
        key = (type(text), text)
        if key in self._parsed:
            return pickle.loads(self._parsed[key])
        document = self._parse(source)
        self._parsed[key] = pickle.dumps(document, protocol=pickle.HIGHEST_PROTOCOL)
        return document


yaml.safe_load = _ParsedYaml(yaml.safe_load)


# -- who shares a walk with whom -----------------------------------------
#
# Analysing a model means resolving it, expanding its component graph,
# building a `Program` and walking it. Measured on DeepSeek-V4-Flash: the
# resolution is ~2 ms, the graph expansion ~140 ms, the walk ~900 ms of
# sympy. Most test files do not sweep the corpus -- they pick one or two
# real models to check one behaviour -- and each of them paid all three
# again.
#
# TWO SCOPES, EACH BECAUSE A MEASUREMENT SAYS SO:
#
# Resolutions and programs are shared for the whole process
# (`corpus._resolution` / `corpus._program`). They are bounded by the corpus
# itself -- 62 models, ~57 MiB measured, no `Analysis` among them -- and the
# reuse is genuinely cross-file: over the 26 converted files, resolutions
# fell 590 -> 220 and graph expansions 751 -> 350, with one file reaching
# zero resolutions because earlier files had already resolved those models.
#
# Those two are COUNTS of operations the walk performed, not timings and not
# keyed on any object's address, so they are reproducible and they are the
# justification for sharing resolutions and programs. Re-verified on this
# tip against its own base, back to back, with `tools/suite_timing.py`.
#
# Walks are shared for one test MODULE and no longer (`walks` below).
#
# WHY MODULE SCOPE, structurally -- this is the reason, and it is not a
# measurement, so it cannot go stale. A pytest fixture reaches a test by
# being named in its signature, and many of the files that share walks are
# `unittest.TestCase` classes whose test methods cannot take one, with
# their walks built by module-level helpers that no fixture argument
# reaches either. That leaves function scope (no reuse at all, so
# pointless), module scope, and session scope -- and session scope is the
# one that would need an eviction policy. Module scope is what is left.
#
# WHAT ACTUALLY BOUNDS THE MEMORY is the CORPUS, not the module. A module
# that sweeps every snapshot -- `test_program_moe.py` does -- holds one
# analysis per model until it ends, so the bound is 62 models, and module
# scope is not what keeps that number small. Measured, sweeping all 62:
# peak RSS 237 MiB with the group holding 62 analyses, against 136 MiB for
# the same sweep dropping each analysis as the pre-branch code did. So this
# RAISED the peak of every corpus-sweeping module it converted, by about
# 100 MiB. (Nothing like 62 x 30 MB: that figure is Kimi-K3, whose graph is
# an order of magnitude bigger than the median model's, so most of the
# corpus costs far less.) Session scope would hold the same 62 for the
# whole run instead of one module, which is the difference that matters on
# a memory-constrained target.
#
# WHY THE WALK SHARING IS WORTH HAVING, and what is NOT claimed for it.
# `test_eval_fusion.py` asks for the same walk of the same model many times
# over and went 230 s -> 75 s (on an idle-ish box; wall clock here swings by
# 3-5x with load, so treat it as an illustration, not a figure). No other
# converted file behaves like that. The justification for the fixture is
# that ONE file is an outlier and the rest are not.
#
# There used to be a per-file table of asks/distinct/repeat-percentages
# here. It is gone on purpose. It was corrected four times in one night --
# a coarse key said 88%, then 72%, then 60% once `asks` was found to
# double-count, and then `distinct` turned out to be understated by `id()`
# recycling and allocator-dependent (one file read 63, 74 and 85 depending
# on how it was run) -- and in all that time nothing about the mechanism it
# described changed, and no decision anyone took depended on which figure
# was right. A number maintained in three places, wrong in all three, that
# nobody acts on is an entity that is not earning its keep (CLAUDE.md
# §1.4); it still has to be correct, and that is the cost.
#
# `tools/suite_timing.py` still measures all of it per file if you need it
# for a specific decision. Read its docstring on what `asks` does and does
# not instrument before quoting it.
#
# Tests are organised by BEHAVIOUR mirroring the source tree (CLAUDE.md
# §5.5), so no test was moved to sit near another that wants the same model.


# -- the default run takes one case of each parametrised test -------------
#
# The default run is the smoke run. A parametrised test runs ONE case there,
# the one whose every parameter is its first value (all `callspec.indices`
# 0); every other case carries `regression`, so `-m regression` runs the
# rest and the two together are the whole suite. Nothing is skipped. This
# is the per-case marking `tests/backend/test_fingerprint.py` already did by
# hand, applied to every parametrised test, so a test file says nothing to
# opt in.
#
# That makes parameter ORDER part of what a test checks, and nothing else
# checks it: the first case is the only one the default run (and the push /
# deploy gate) ever sees, so it must be a case that FAILS when the defect
# the test exists for comes back -- not merely the cheapest. A test whose
# point is a pair (refuse this AND admit that) puts the half most likely to
# regress first, or splits the pair into two tests so each half has a first
# case. Cheapest-first is the tie-break among cases that discriminate
# equally. Get it wrong and the suite stays green while the default run
# stops guarding anything.
#
# The representative is picked by parameter index, not by collection order,
# so every xdist worker (and any reordering plugin) picks the same one. A
# test whose first case is already marked `regression` by hand has no case
# in the default run.
#
# `tests/meta` is exempt: those are the repo's own rule gates, one case per
# file or per declaration, cheap, and a gate that checks one file of N lets
# the other N-1 rot (CLAUDE.md §5.5).


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    meta = PROJECT_ROOT / "tests" / "meta"
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None or item.path.resolve().is_relative_to(meta):
            continue
        if any(callspec.indices.values()):
            item.add_marker(pytest.mark.regression)


@pytest.fixture(scope="module", autouse=True)
def walks():
    """The walks this test module's tests share, gone when it finishes.

    Read out of `sys.modules` rather than imported, so a module that never
    touches the corpus (the frontend's, the service's) does not import
    `backend` just to be handed a group it will not use."""
    corpus = sys.modules.get("tools.corpus_pins.corpus")
    if corpus is None:
        yield None
        return
    group = corpus.WalkGroup("one test module")
    previous = corpus.use_group(group)
    yield group
    corpus.use_group(previous)
    group.clear()


@pytest.fixture(scope="session")
def spec_loader():
    """The process-wide SpecLoader (`tests.snapshots.shared_loader`)."""
    from tests.snapshots import shared_loader
    return shared_loader()


@pytest.fixture(scope="session")
def resolved(spec_loader):
    """Return a function that resolves a model_id or raises skip / fail.

    Usage:
        def test_foo(resolved):
            spec = resolved("meta-llama/Llama-3.2-3B")
            assert spec.dim_values["n_layers"] == 28

    Behavior:
      - Snapshot file missing → pytest.skip
      - Snapshot present but resolution fails → test failure
      - Snapshot resolves → return ResolvedSpec

    One implementation: `tests/snapshots.resolve_snapshot` — the same two
    outcomes every direct caller gets, so "not checked out" and "checked
    out but broken" cannot drift apart between the fixture and the helper.
    """
    from tests.snapshots import resolve_snapshot

    def _resolve(model_id: str):
        return resolve_snapshot(model_id, spec_loader)

    return _resolve


@pytest.fixture(autouse=True)
def _no_host_agent_credentials(monkeypatch):
    """Every test starts on a host that holds no agent credential, as CI's
    runner does: each credential variable the agents declare
    (`AGENT_ENV_VARS[...].credentials`) is unset. Without this a test's
    outcome depends on the developer's shell — a run that a host login lets
    start locally is refused on CI (`host_credentials_missing`). A test whose
    premise is a host that can authenticate asks for `host_credentials`."""
    from backend.agents.base import AGENT_ENV_VARS
    for agent in AGENT_ENV_VARS.values():
        for name in agent.credentials:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def host_credentials(monkeypatch):
    """A host that holds an Anthropic key: runs on the host's credentials
    may start."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host-test")
