"""Second-campaign ratchets — structural debt, counted so it can only shrink.

These are not number-correctness tests. They pin the shape of the tree:
who may import whom, how many times a display helper is defined, how wide
the capacity entry point is. Going above a baseline is a regression;
coming in below means this file's number must drop in the same commit.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BACKEND = PROJECT_ROOT / "backend"
ARCH = BACKEND / "arch"
INFERENCE = BACKEND / "inference"
MEASURE = BACKEND / "measure"


def _py_files(root: Path):
    for path in root.rglob("*.py"):
        if "/__pycache__/" in str(path):
            continue
        yield path


def _imports_kernel_bench(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "kernel_bench" or alias.name.startswith(
                        "kernel_bench."):
                    return True
        elif (isinstance(node, ast.ImportFrom) and node.module
              and (node.module == "kernel_bench"
                   or node.module.startswith("kernel_bench."))):
            return True
    return False


#: HTTP and Kernel Bench service composition roots may import the harness.
#: Nothing else under `backend/` may — the cost model and the spec tree must not grow a
#: reverse edge. Comments that mention the name are not imports; this
#: walks AST.
KERNEL_BENCH_IMPORTERS = frozenset({
    "backend/api/routes/inference.py",
    "backend/api/routes/fusion.py",
    "backend/service/bench/annotations.py",
    "backend/service/bench/data_rules.py",
    "backend/service/bench/routes.py",
})


def test_arch_and_inference_do_not_import_kernel_bench():
    """`backend.measure` too: the harness imports it (the coverage
    question), never the reverse."""
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for root in (ARCH, INFERENCE, MEASURE)
        for path in _py_files(root)
        if _imports_kernel_bench(path)
    ]
    assert offenders == [], (
        "arch/inference imported kernel_bench; the harness may import "
        f"backend, not the reverse: {offenders}")


def test_only_http_routes_import_kernel_bench():
    found = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _py_files(BACKEND)
        if _imports_kernel_bench(path)
    ]
    assert set(found) == KERNEL_BENCH_IMPORTERS, (
        "kernel_bench importers drifted. HTTP routes are the composition "
        f"root; core must not join them. got {sorted(found)}")


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_providers_do_not_import_driver_or_plan():
    """A provider builds and times one kernel. Driver/plan orchestrate
    jobs; if a provider imports them the two layers become one file."""
    root = PROJECT_ROOT / "kernel_bench" / "providers"
    banned = {"kernel_bench.driver", "kernel_bench.plan", "driver", "plan"}
    offenders = [
        f"{path.name}: {mod}"
        for path in _py_files(root)
        for mod in _imported_modules(path)
        if mod in banned
    ]
    assert offenders == [], offenders


def test_fmt_si_is_defined_once_in_inference():
    found = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _py_files(INFERENCE)
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_fmt_si"
    ]
    assert found == ["backend/inference/phase_formulas.py"], found


def test_phase_formula_builders_live_next_to_the_roofline_not_in_it():
    src = (INFERENCE / "phases.py").read_text()
    for name in ("_flops_formula_for_row", "_access_bytes_formula_for_row",
                 "_t_op_formula_for_row", "_collective_bytes_formula",
                 "phase_roofline_formulas", "attach_parent_formulas",
                 "enrich_module_breakdown"):
        assert f"def {name}" not in src, name
    assert 'formulas["mem_read"]' not in src
    assert 'formulas["t_sol"]' not in src


def test_estimate_capacity_kwonly_does_not_grow():
    tree = ast.parse((INFERENCE / "capacity.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "estimate_capacity":
            n = len(node.args.kwonlyargs)
            # config, gpu_sku, knobs, two seq lengths, three batch overrides.
            assert n == 8, (
                f"estimate_capacity kw-only is {n}, expected 8; "
                "seq-invariant fields belong on CapacityKnobs")
            return
    raise AssertionError("estimate_capacity not found")


def test_estimate_capacity_rejects_knobs_and_flat_together():
    from backend.inference.capacity import CapacityKnobs, estimate_capacity
    with pytest.raises(TypeError, match="not both"):
        estimate_capacity(
            config={}, gpu_sku="H100-SXM",
            knobs=CapacityKnobs(),
            gpu_memory_fraction=0.5)


def test_overlay_orchestration_is_not_in_capacity():
    """Calibration overlay is a stage, not a budget scalar. The helpers
    that build its three dataclasses live next to the stage."""
    src = (INFERENCE / "capacity.py").read_text()
    assert "def _run_calibration_overlay" not in src
    assert "def _resolve_overlay_context" not in src


def test_capacity_formula_popovers_are_not_in_capacity():
    """The estimate's `formulas` object is display, not a budget scalar."""
    src = (INFERENCE / "capacity.py").read_text()
    assert "def _group_kv_nodes" not in src
    assert "def capacity_formulas" not in src


def test_spec_vs_tensors_check_is_not_in_capacity():
    """Bytes-vs-bytes matching is validate's job, not a budget scalar."""
    src = (INFERENCE / "capacity.py").read_text()
    assert "def _spec_vs_tensors_check" not in src
    assert "def spec_vs_tensors_check" not in src


def test_signature_catalog_is_not_in_the_registry_module():
    src = (INFERENCE / "op_signatures.py").read_text()
    assert "register_signature(OpSignature(" not in src, (
        "signature declarations belong in op_signatures_catalog.py")


def test_estimate_capacity_rejects_unknown_flat_field():
    from backend.inference.capacity import estimate_capacity
    with pytest.raises(TypeError, match="unexpected"):
        estimate_capacity(config={}, gpu_sku="H100-SXM", not_a_knob=1)


# ---------------------------------------------------------------------------
# The dependency graph inside `backend`: cycles, and the lazy imports that
# hide them.
# ---------------------------------------------------------------------------


def _module_name(path: Path) -> str:
    parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts
    name = ".".join(parts)
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


# Import cycles and function-body imports are counted by `tools/scoreboard.py`
# and reported by `tests/meta/test_scoreboard.py` against the one declared
# baseline in `scoreboard_baseline.json`.


#: Modules allowed to read a `SUPABASE_*` variable straight from the
#: environment. Nine read the URL and seven the key before this list existed,
#: each with its own idea of whether a trailing slash mattered and whether a
#: URL without a key counted as configured.
SUPABASE_ENV_READERS = {"backend/supabase_env.py"}


def test_supabase_credentials_are_read_in_one_module():
    offenders = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in _py_files(BACKEND)
        if "SUPABASE_" in path.read_text() and "os.environ" in path.read_text()
        and any(f'os.environ.get("{name}"' in path.read_text()
                or f'os.environ["{name}"]' in path.read_text()
                for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
                             "SUPABASE_ANON_KEY", "SUPABASE_JWT_SECRET"))
    )
    assert set(offenders) <= SUPABASE_ENV_READERS, (
        "read the Supabase credentials through backend.service.supabase_env, "
        f"not the environment: {sorted(set(offenders) - SUPABASE_ENV_READERS)}")


#: Modules allowed to parse authored expression text. Two dozen call sites
#: reached for `sympy.sympify` directly, and the first attempt at collapsing
#: them left `arch.schema` with a four-line copy that bound `Min` as an axis
#: — so a law the reader handled raised a TypeError there, and the caller
#: swallowed it as "these two axes differ".
LAW_TEXT_READERS = {"backend/arch/law_text.py"}


def test_authored_expressions_have_one_reader():
    """Both spellings, because the original hole was `sympify`, not
    `parse_expr`: two dozen call sites handed authored text to SymPy's whole
    namespace, where `beta` is a function and `E` is a number."""
    offenders = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in _py_files(BACKEND)
        if any(call in path.read_text()
               for call in ("parse_expr(", "sympify("))
    )
    # `dim.py` coerces an already-evaluated extent with `sympify`; it reads no
    # text, and hands every string to `law_text`.
    allowed = LAW_TEXT_READERS | {"backend/arch/eval/dim.py"}
    assert set(offenders) <= allowed, (
        "read authored expression text through backend.arch.law_text: "
        f"{sorted(set(offenders) - allowed)}")


#: The `custom:` prefix is identity, not registration. Hub metadata and
#: the CLI both need to know whether an id is one of those; they must
#: not import each other to find out.
CUSTOM_ID_MODULE = "backend/custom_id.py"


def test_custom_id_is_defined_in_one_module():
    offenders = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in _py_files(BACKEND)
        if "def is_custom_id" in path.read_text()
        or 'NAME_PREFIX = "custom:"' in path.read_text()
    )
    assert offenders == [CUSTOM_ID_MODULE], (
        "the custom: prefix lives in backend.custom_id: "
        f"{offenders}")
