"""One answer to "what identifies an estimate's result"
(`backend/measure/scope.py`, docs/measured_calibration.md §7.5):
every cache of an estimate's result takes its key and its remote
eligibility from it. This gate enumerates the places a cache key is built
under `backend/api` and `backend/service`: `make_key` is called only by the
identity module and by caches whose value is not an estimate's result (each
named with what it holds), and a call's arguments are pickled only by
`offload.call_key`, where an estimate request stands for its identity."""
from __future__ import annotations

import ast
import pathlib

from backend.api import sweep_runtime
from backend.api._models import InferenceEstimateSweepRequest, InferenceParallelSweepRequest
from backend.api.offload import call_key
from backend.measure.scope import MeasurementScope, of_request

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Keys of caches that hold no estimate result — (file, function): what the
#: cached value is. Anything else that builds a key builds an estimate's,
#: and must ask `backend.measure.scope`.
NOT_AN_ESTIMATE = {
    ("backend/api/routes/supported_models.py", "_supported_models_payload"): "the supported-model list",
    ("backend/api/routes/gallery.py", "gallery_models"): "the models' deployment-free facts",
    ("backend/api/gallery_overview.py", "_facts_key"): "one model's deployment-free facts",
    ("backend/api/gallery_overview.py", "_model_size_and_date"): "a model's size and date",
    ("backend/api/routes/inference.py", "_default_deployment"): "the default deployment a card opens at",
}


def _calls(name: str):
    """(file, enclosing function, line) of every call to `name` under
    backend/api and backend/service."""
    for sub in ("backend/api", "backend/service"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(ROOT).as_posix()
            for fn in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for node in ast.walk(fn):
                    if not isinstance(node, ast.Call):
                        continue
                    f = node.func
                    called = f.id if isinstance(f, ast.Name) else (
                        f.attr if isinstance(f, ast.Attribute) else None)
                    if called == name:
                        yield rel, fn.name, node.lineno


def test_only_the_identity_builds_an_estimate_key():
    found = {(rel, fn) for rel, fn, _ in _calls("make_key")
             if rel != "backend/measure/scope.py"}
    assert found == set(NOT_AN_ESTIMATE), (
        "a cache key built outside backend/measure/scope.py: an "
        "estimate's result must take its key from ResultIdentity; a cache "
        f"of something else is named in NOT_AN_ESTIMATE. got {sorted(found)}")


def test_a_sweep_step_has_its_own_identity():
    """A local partial-progress artifact is named by the identity module too."""
    from backend.measure.scope import of_sweep_step

    identity = of_sweep_step("theory-key", 3, 4)
    assert identity.kind == "sweep-step"
    assert identity.version == "theory-key"
    assert identity.content == (3, 4)
    assert not identity.reads
    assert not identity.remote_ok


def test_only_the_call_key_pickles_a_call():
    found = {(rel, fn) for rel, fn, _ in _calls("dumps")
             if rel.startswith("backend/api/")}
    pickled = {(rel, fn) for rel, fn in found
               if "pickle" in (ROOT / rel).read_text(encoding="utf-8")}
    assert pickled <= {("backend/api/offload.py", "call_key")}, pickled


def _scoped(cls, owner, snapshot, shared=3, **extra):
    return cls(model_id="m/x", gpu_sku="H100-SXM", **extra).model_copy(update={
        "measurement_scope": MeasurementScope(owner, "lake", snapshot, shared, "r")})


def test_every_surface_reads_the_same_identity(monkeypatch):
    monkeypatch.setattr(sweep_runtime, "resolved_spec_for", lambda m: object())
    monkeypatch.setattr(sweep_runtime, "result_version", lambda r: "rv")
    for kind, cls, extra in (("sweep", InferenceEstimateSweepRequest, {"seq_lens": [1]}),
                             ("parallel", InferenceParallelSweepRequest,
                              {"modes": ["tp_tp"], "contexts": [1024], "gpu_skus": ["H100-SXM"]})):
        a = _scoped(cls, None, 10, **extra)
        # A measured answer is never cached; it joins in flight on its identity.
        assert sweep_runtime.sweep_cache_key(kind, a) is None
        assert of_request(kind, a, None).scope_digest == a.scope().digest
        # A later pin with the same shared rows: one join.
        later = _scoped(cls, None, 11, **extra)
        assert call_key(sweep_runtime._run_sweep, (a,), {}) == call_key(
            sweep_runtime._run_sweep, (later,), {})
        # Another owner: another join.
        mine = _scoped(cls, "alice", 10, **extra)
        assert call_key(sweep_runtime._run_sweep, (a,), {}) != call_key(
            sweep_runtime._run_sweep, (mine,), {})
        # The SOL answer: one entry whoever asks, under whatever scope.
        sol = sweep_runtime.sweep_theory_identity(kind, a, answer=True)
        assert sol.remote_ok
        for r in (a, later, mine):
            assert sweep_runtime.sweep_cache_key(
                kind, r.model_copy(update={"sol_only": True})) == sol.key


def _kind_literals(name: str):
    """(file, function, first argument) of every call to `name` under
    backend/api and backend/service whose first argument is a literal
    string — the `kind` a cache entry is built under."""
    for sub in ("backend/api", "backend/service"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(ROOT).as_posix()
            for fn in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for node in ast.walk(fn):
                    if not isinstance(node, ast.Call):
                        continue
                    f = node.func
                    called = f.id if isinstance(f, ast.Name) else (
                        f.attr if isinstance(f, ast.Attribute) else None)
                    if called != name or not node.args:
                        continue
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        yield rel, fn.name, first.value


def test_only_of_walk_names_a_walk():
    """A WALK has one name (`scope.of_walk`).

    A walk is the expensive half of every published number and it is shared
    across the cards of one class, so two consumers that compose their own
    key over the same facts cannot share one entry — which is exactly why
    the Gallery's walk and the Scaling sweep's walk used to be two 33 GB
    towers of the same work. The hazard is a `kind` string with "walk" in
    it built through one of the other constructors, so that is what this
    refuses: ask `of_walk`, which names the six components a walk is a
    function of.
    """
    offenders = [(rel, fn, kind)
                 for builder in ("of_render", "of_request", "theory_of_request")
                 for rel, fn, kind in _kind_literals(builder)
                 if "walk" in kind.lower()]
    assert offenders == [], (
        "a walk keyed by something other than scope.of_walk: a walk is "
        "named by the facts its ROWS are a function of (model, layout, card "
        f"class, precision, bound pp stage, anchor). got {offenders}")


def test_the_gallery_asks_for_its_walk_by_that_name():
    """The one consumer that already had a keyed, cached, shared walk tier
    reads the shared name, so its entry and a sweep's can be the same
    entry. Pins the two properties the key must have: cards of one walk
    class share it, cards of different classes cannot."""
    from backend.api import gallery_frontier as gf

    stored = gf._Stored("meta-llama/Llama-3.2-3B", "rv1")
    one_class = gf.FrontierView("H100-SXM")
    # H100-SXM and H200 are one walk class: one walk serves both.
    walkKey = lambda view: stored.walk_identity(view).key
    assert walkKey(one_class) == walkKey(gf.FrontierView("H200"))
    # B200 is another (compute capability 10.0): not representable as this
    # entry, rather than merely refused when read.
    assert walkKey(one_class) != walkKey(gf.FrontierView("B200"))
    # Not the floors, and not the card itself.
    floors = one_class._replace(comm_latency_us=99.0, launch_floor_us=99.0)
    assert walkKey(floors) == walkKey(one_class)
