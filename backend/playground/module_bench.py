"""Isolated single-component benchmark (playground semantics B).

Given a base model (for its resolved dims + config) and a list of library
component names, isolate each component under a fixed input contract
`(batch, input_tokens, dims)` and return its canonical cost numbers. Every
number comes from an existing eval entry point so a module row matches what the
same component contributes inside a whole model:

  * params, flops, weight / KV reads, fused activation bytes -> one phase's
    runtime table (`_runtime_op_table`): the isolated component's serving
    step at the contract `(B, L)` (`serving_step`) — the walk, its
    collectives, its activation casts and the shipped fusions, in the
    serving graph's order — each row its serving row; MoE experts saturate
    at the tokens the step routes. The cells are the table's totals and
    their popovers list its rows. `fused_graph` / `fused_l1` draw that
    prefill rewrite at both graph granularities, with each priced runtime
    row stamped onto its post-fusion graph node.
  * naive activation bytes -> the same step's rows before the rewrite
    (`AnalysisResult.pre_per_op_timing`).
  * weight storage         -> `capacity.weights()` over the isolated `Analysis`:
    parameters, their quant metadata and buffers at the checkpoint's formats.
  * kvcache                -> `of_component()` tensors + `CachedTensor.bytes_at(L)`
    (per-node dtype width handled by the tensor, not by us); the whole
    model's from that model's walk under the entry's dims (`whole_model_kv`).

Every isolation runs at a `CallSite`: one call of the model the entry is
bound to (or of the base model), which hands the component its formals and
its frame's dims, and places its weights (address, role, the checkpoint's
formats — what prices an FP8 checkpoint's weights at FP8). A model that calls
the component once is evaluated at that call; one that calls it more than once
is evaluated at the call the entry names (`call`), and refused until it names
one. A component the model does not call takes the place of the call named.

FLOPs, weight reads and activation bytes are reported per phase, the way the
Inference page reports them: prefill is one forward over `L` tokens per request;
decode is one new token per request against a cache of `L`
(`Deployment.during("decode")`, which also runs the component's declared decode
form). Weight storage is one copy of the block, the same in either phase.

A library component the base model never uses (e.g. MLA on a Llama base) is not
in `spec.components`, so `get_block` would `KeyError`. We inject the component
and its transitive component deps first (mirrors the loader's own BFS).
"""

from __future__ import annotations

from html import escape

from backend.arch.yaml_load import load_yaml
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple

import sympy as sp
import yaml

from backend.measure.auto_sweep import DECLINED, MeasuredTrack, PointMeasure
from backend.hf.metadata import fetch_config, fetch_metadata
from backend.arch import dims as dims_mod
from backend.arch.dim_resolution import _DIM_EXPR_FUNCTIONS
from backend.arch.law_text import read_law
from backend.arch.loader import get_default_loader
from backend.playground import entry_key
from backend.arch.ops import binding_dims
from backend.arch.snapshots import iter_snapshot_ids, load_config
from backend.arch.eval.transforms import (
    DECODE,
    PREFILL,
    CallSite,
    Deployment,
    Phase,
    Workload,
    analyze_isolated,
    isolate_resolved,
    phase_form,
)
from backend.arch.eval.projection import project_block
from backend.arch.block_synthesis import binding_key, binding_of
from backend.arch.eval.program import Program
from backend.arch.eval.binding import STEP_LAWS, Definition, declared_names
from backend.arch.eval.dim import (
    bind_record, bound_formula_text, bound_or_folded, expr_of,
    names_of, value_of, varies,
)
from backend.arch.eval.access import (
    _wrap_expr, params_bytes_formula,
)
from backend.arch.eval.values import CACHE_LEN
from backend.arch.eval.cache import by_call, of_component
from backend.arch.eval.cache_view import Rate
from backend.arch.eval import capacity as capacity_mod
from backend.arch.dtypes import display_name
from backend.inference.defaults import COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import phase_binding
from backend.inference.phase_formulas import STEP_SOL_BOUND_NOTE, STEP_SOL_LAW
from backend.arch.runtime_profile import RuntimeProfile
from backend.inference.step_eval import StepCells, evaluate_step
from backend.inference.pipeline import AnalysisResult, NoPeakForDtype, analyze as serving_analyze
from backend.inference.calibration_overlay import (
    CalibrationOverlay,
    WorkloadShape,
    build_deployment_context,
)
from backend.inference.framework_bindings import (
    AuditedAt,
    FrameworkBinding,
    FusionImplementation,
    ServingDeployment,
)
from backend.inference.gpu_specs import (
    get_gpu,
    peak_hbm_bw,
)
from backend.inference.kv_topology import kv_cache_topology
from backend.inference.batch_sweep import (_measured_row, _measured_table,
                                           measured_selector, step_measure,
                                           terms_by_uid)
from backend.inference.intensity import arithmetic_intensity
from backend.inference.row_roofline import StepSum
from backend.measure import lookup as _measure_lookup
from backend.measure.scope import NONE, MeasurementScope
from backend.inference.phase_formulas import (
    MAX_COMPONENT_TERMS,
    access_display_facts,
    act_in_from_terms,
    act_out_from_terms,
    _finalize_parts,
    _fmt_time,
    _substituted_sum,
    enrich_module_breakdown,
)
from backend.inference.module_rows import physical_module_rows, served_rows
from backend.rendering.builders import build_block_graph_data
from backend.rendering._compute_graph import serialize_post_tree
from backend.rendering.kv_marks import KVMarks
from backend.views.abstraction import Sums
from backend.views.diff import DiffRefused, Entry, GraphDiff
from backend.arch.eval.fusion import (
    Fusion,
)
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.framework_bindings import (
    FRAMEWORK_BINDINGS,
    _pattern_specificity,
)


def shipped_fusion_plans():
    """The fusion schemes at least one SHIPPED framework binding declares
    (vllm, sglang, …) — NOT the raw scheme registry.

    The registry holds plans no deployed framework actually ships (e.g.
    `gated_act_ffn`, which would fold a whole SwiGLU FFN into one leaf and hide
    activations that still land in HBM). The fused view must only elide
    intermediates a real kernel fuses, so it takes the union of what the shipped
    bindings implement (silu_and_mul, merged/QKV linears, flash_attention,
    flash_dsa, …). Ordered most-specific-first, the order bindings apply them
    in; non-matching plans are no-ops, so each variant gets exactly the kernels
    that fit it (none for a recurrent state). Built once.
    """
    names = set()
    for binding in FRAMEWORK_BINDINGS.values():
        names.update(binding.fused_ops())
    # The name settles a tie, so the order is the same in every process
    # rather than the set's.
    ordered = sorted(names, key=lambda n: (-_pattern_specificity(n), n))
    return [Fusion.named(n, FUSION_SCHEMES) for n in ordered]


_FUSION_PLANS = shipped_fusion_plans()

_UNION_BINDING = FrameworkBinding(
    framework="huggingarch-shipped",
    audited_at=AuditedAt(tag="union", commit=""),
    op_implementations=tuple(
        (plan.scheme_name, FusionImplementation())
        for plan in _FUSION_PLANS
    ),
)


def _measurement_scope() -> Any:
    """The Kernel Bench scope a bench reads now (`measure.lookup`); one that
    reads nothing leaves the times SOL. A service that does not answer is the
    request's 503 (`MeasurementsUnavailable`).

    Module rows are deliberately framework-union fusions, so the measured
    fold selects with the envelope strategy: the fastest exact physical-op
    measurement, independent of which serving library shipped it."""
    return _measure_lookup.resolve_scope(reads=True, owner=None)


def timing_scope(gpu_sku: Optional[str]) -> MeasurementScope:
    """The scope a bench's times on `gpu_sku` are read under
    (`_measurement_scope`, which `phase_roofline` folds its rows with), and
    `NONE` with no card: there is no time, so nothing is read. None from
    the scope is `measured_selector`'s spelling of no scope, which is
    `NONE`. What a cached bench result is keyed under
    (`ResultIdentity.scope`)."""
    return (_measurement_scope() if gpu_sku else None) or NONE


#: Failures that mean "the caller's contract is missing/inconsistent for this
#: component" (an unbound dim symbol, a name the library does not have, a shape
#: that does not compose). Surfaced per-component as a repairable error rather
#: than failing the whole request. Anything else propagates as a real bug.
CONTRACT_ERRORS = (KeyError, ValueError, TypeError)

logger = logging.getLogger(__name__)

# --- Missing-dim diagnosis -------------------------------------------------
#
# A component whose dims the base model does not carry (MHA's `d_head` on an
# MLA base, KDA's `d_lin_*` on a dense base) fails deep inside eval — sympy
# leaks "Cannot convert symbols to int" or a flops_fn raises a bare KeyError.
# Both leave the user guessing WHICH dims are missing. The scan below reads
# the injected graph's dim expressions statically and reports every unbound
# name at once — facts only (there are no suggested values; the repair is
# binding the entry to a model that parameterizes the component).

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def is_deployment_symbol(name: str) -> bool:
    """Workload axes the engine pins per-request (`dims.yaml` engine_supplied)."""
    sym = dims_mod.get(name)
    return bool(sym and sym.engine_supplied)


def _is_contract_dim(name: str) -> bool:
    """Not a per-component editable knob.

    Engine-supplied dims (`dims.yaml` `engine_supplied`) are the declaration
    for per-request axes. The rest have no such marker, so they stay
    handwritten:

    - `in` / `out` / `M`: GEMM op formals. dims.yaml uses `d_reduce` / `d_out`;
      these names have no registry entry.
    - `d_model`: residual width. The registry knows the symbol but has no
      interface-width flag.
    - `s_max`: published context length. Model-level bound, not a component
      knob. Group `backbone` also contains real knobs (`n_heads`, `d_ffn`),
      so group membership cannot answer this.
    - `Ob` / `Qb`: bias-presence flags (`derived: "0"`). Not knobs.
    """
    if name in ("in", "out", "M", "d_model", "s_max", "Ob", "Qb"):
        return True
    return is_deployment_symbol(name)


def _expr_dim_idents(text: Any) -> set:
    """Identifier tokens in a dim-expression string, minus known functions."""
    if not isinstance(text, str):
        return set()
    return {tok for tok in _IDENT_RE.findall(text) if tok not in _DIM_EXPR_FUNCTIONS}


def _node_dim_idents(node: Any) -> set:
    """Dim identifiers one graph node's expressions reference. Only the
    value/expression side is scanned: `dims` KEYS are the callee's parameter
    names (bound by the value they map to), and string `params` are component
    bindings, not dims."""
    names = set()
    for value in node.dims.values():
        names |= _expr_dim_idents(value)
    names |= _expr_dim_idents(node.dim)
    for entry in node.shape or []:
        names |= _expr_dim_idents(entry)
    names |= _expr_dim_idents(node.copies)
    names |= _expr_dim_idents(node.active)
    names |= _expr_dim_idents(node.when)
    return names


def unbound_dims(
    host: Any, dim_overrides: Dict[str, Any], component: Optional[str] = None
) -> List[str]:
    """Sorted dim names the graph closure of `component` (a component or a
    block template) references but neither the base model's resolved dims
    nor the caller's overrides bind."""
    bound = set(dim_overrides)
    bound.update(name for name in dims_mod.symbols() if is_deployment_symbol(name))
    return sorted(closure_dim_refs(host, component) - bound - set(host.dim_values))


def closure_dim_refs(host: Any, component: Optional[str] = None) -> set:
    """Dim names the graph closure of `component` — a component or a block
    template — spells out in its expressions, read statically off the spec
    (every host component when `component` is None).

    The closure is what the start reaches through calls: a node whose `op`
    is a component or block, and a wrapper node that binds one through a
    component-typed param (`pre_norm` with `attn: …`). A block template is a
    start like any other — a layer group's own dims (gemma-4's full layers
    rebinding `n_kv_heads`) are only visible from the block that binds them.

    Scope is the start's OWN closure, not every component in the host map —
    an unused sibling (e.g. deepseek_v4's never-instantiated ffn_dense) must
    not lend its dims to the report. A callee's parameters that every call
    site binds (`Cape: "4"` on the compressor node) are satisfied there and
    dropped from THAT callee's references; a pass-through binding
    (`compress_ratio: compress_ratio`) still requires the parent scope's dim,
    which stays referenced via the value side of the call site itself.
    """
    spec = host.spec
    callable_names = set(spec.components or {}) | set(spec.blocks or {})
    refs_by_comp: Dict[str, set] = {}
    # callee → one key-set PER call site. A parameter is "satisfied at the call
    # site" only when EVERY call of that subcomponent binds it (intersection) —
    # with a union, one call site's binding would mute the gap report for the
    # call sites that don't bind it (under-report, the silent direction).
    callsite: Dict[str, List[set]] = {}
    seen: set = set()
    stack = [component] if component else list(spec.components or {})
    while stack:
        cname = stack.pop()
        if cname in seen or cname not in callable_names:
            continue
        seen.add(cname)
        comp = spec.get_block(cname)
        refs: set = set()
        for node in comp.graph.values():
            refs |= _node_dim_idents(node)
            refs |= set(binding_dims(node.op))
            callees = [
                node.op,
                *(v for v in node.params.values() if isinstance(v, str)),
            ]
            for callee in callees:
                if callee in callable_names:
                    callsite.setdefault(callee, []).append(set(node.dims.keys()))
                    stack.append(callee)
        for state in comp.kv_states:
            # Resolved in the carrier's frame: the carrier's own `dims:`
            # keys (a conv's `C`, `kernel_size`) are bound there, not asked
            # of the model.
            carrier = comp.graph.get(state.carrier)
            own = set(carrier.dims) if carrier is not None else set()
            for axis in state.shape:
                refs |= _expr_dim_idents(axis) - own
        refs_by_comp[cname] = refs
    referenced: set = set()
    for cname, refs in refs_by_comp.items():
        sets = callsite.get(cname) or []
        common = set.intersection(*sets) if sets else set()
        referenced |= refs - common
    return referenced


# A gap is reported as FACTS ONLY (which dims nothing binds) — there are no
# suggested values: a component that can't resolve under this base must be
# bound to a model that actually parameterizes it, not patched with a guess.
# The one legitimate source of dims beyond the base's own is a bound entry's
# inline seed (the binding model's real dims at the call it names, its layer
# variant's included — `CallSite.frame`).

_MODELS_DIR = Path(__file__).resolve().parent.parent / "arch" / "models"

_model_docs_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _model_docs() -> Dict[str, Dict[str, Any]]:
    """Every model yaml as `{dims, model_ids, components, variants}`,
    `base:` chains merged. Built once: the directory is git-tracked and static
    for the process."""
    global _model_docs_cache
    if _model_docs_cache is not None:
        return _model_docs_cache
    raw: Dict[str, Dict[str, Any]] = {}
    for path in sorted(_MODELS_DIR.glob("*.yaml")):
        try:
            data = load_yaml(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning(
                "model-doc scan: skipping unreadable model yaml %s: %s", path, e
            )
            continue
        if isinstance(data, dict):
            raw[path.stem] = data

    docs: Dict[str, Dict[str, Any]] = {}

    def merged(stem: str, seen: Tuple[str, ...] = ()) -> Dict[str, Any]:
        if stem in docs:
            return docs[stem]
        if stem in seen or stem not in raw:
            return {"dims": {}, "model_ids": [], "components": {}, "variants": []}
        data = raw[stem]
        # Layer-recipe variants (per_layer assign maps) carry their own
        # component picks and dims — e.g. deepseek_v4 binds compress_ratio=4
        # only on the CSA layers. The flat dims:/components: maps never see
        # them, but provenance needs them.
        variants: List[Dict[str, Any]] = []
        for item in data.get("model") or []:
            per_layer = (item or {}).get("per_layer") or {}
            for variant in (per_layer.get("map") or {}).values():
                if not isinstance(variant, dict):
                    continue
                variants.append(
                    {
                        "components": {
                            k: v
                            for k, v in variant.items()
                            if k != "dims" and isinstance(v, str)
                        },
                        "dims": dict(variant.get("dims") or {}),
                    }
                )
        doc = {
            "dims": dict(data.get("dims") or {}),
            "model_ids": [],
            "components": dict(data.get("components") or {}),
            "variants": variants,
        }
        base = data.get("base")
        if isinstance(base, str):
            parent = merged(base, (*seen, stem))
            doc = {
                "dims": {**parent["dims"], **doc["dims"]},
                "model_ids": list(parent["model_ids"]),
                # role→component map, child overrides parent (the resolved bind).
                "components": {**parent["components"], **doc["components"]},
                "variants": [*parent["variants"], *variants],
            }
        match = data.get("match") or {}
        mids = match.get("model_id") or []
        doc["model_ids"] += [mids] if isinstance(mids, str) else list(mids)
        # NOTE: `match.model_type` is a CANDIDATE condition, not a binding fact
        # (several yamls may claim one model_type — deepseek_v2 vs
        # deepseek_v2_qcomp). Provenance from model_type-matched families comes
        # from the engine's own answer instead: `snapshot_specs()` resolves
        # each installed snapshot and `_component_to_models()` reads the winner
        # spec's component_roles + layer assignments.
        docs[stem] = doc
        return doc

    for stem in raw:
        merged(stem)
    _model_docs_cache = docs
    return docs


_snapshot_specs_cache: Optional[List[Tuple[str, Any]]] = None


def snapshot_specs() -> List[Tuple[str, Any]]:
    """Every installed snapshot as (model_id, resolved spec) — the engine's own
    answer to "which spec does this checkpoint use". Provenance and bindings
    read THIS, never the yamls' match conditions (a `match.model_type` claim is
    a candidate filter; several yamls can claim one model_type, and only
    `loader.match` knows the winner). One match per snapshot, cached for the
    process; snapshots are git-tracked and static within a run."""
    global _snapshot_specs_cache
    if _snapshot_specs_cache is not None:
        return _snapshot_specs_cache
    loader = get_default_loader()
    out: List[Tuple[str, Any]] = []
    for mid in iter_snapshot_ids():
        cfg = load_config(mid)
        if not cfg:
            continue
        try:
            resolved = loader.match(cfg, model_id=mid)
        except Exception as e:  # best-effort; a broken snapshot just has no provenance
            logger.info("snapshot match %s failed: %s", mid, e)
            continue
        if resolved:
            out.append((mid, resolved))
    _snapshot_specs_cache = out
    return out


def _resolved_components(resolved: Any) -> set:
    """Component names a RESOLVED spec actually binds: the role→component map
    (`component_roles`) plus the per-layer assignment params (variant picks
    like dsv4's HCA/CSA recipes, which never appear in the flat map) —
    recursing a nested binding (`{template: …, params: {…}}`), whose inner
    components (V4.1's CSA2 slot forms) are bound exactly as much as the
    wrapper that names them."""
    comps = set((resolved.spec.component_roles or {}).keys())

    def collect(value: Any) -> None:
        if isinstance(value, str):
            comps.add(value)
            return
        if isinstance(value, dict):
            if "template" in value:
                comps.add(str(value["template"]))
            for sub in (value.get("params") or {}).values():
                collect(sub)

    for layer in resolved.spec.model or []:
        for a in layer.assignments or []:
            for v in a.params.values():
                collect(v)
    return comps


_component_to_models_cache: Optional[Dict[str, List[str]]] = None


def _component_to_models() -> Dict[str, List[str]]:
    """Component name → the model ids that bind it. Two honest sources unioned:
    the model yamls' EXPLICIT `match.model_id` lists (authored facts), and the
    installed snapshots resolved through `loader.match` (the engine's verdict,
    including per-layer variant picks). A library component no shipped model
    binds maps to `[]` (honestly: library-only)."""
    global _component_to_models_cache
    if _component_to_models_cache is not None:
        return _component_to_models_cache
    inv: Dict[str, set] = {}
    for doc in _model_docs().values():
        mids = doc.get("model_ids") or []
        if not mids:
            continue
        comps = list((doc.get("components") or {}).values())
        # layer-recipe variants bind components too (dsv4_hca_attention lives
        # only in deepseek_v4's per-layer map)
        for variant in doc.get("variants", []):
            comps.extend(variant["components"].values())
        for comp in comps:
            if isinstance(comp, str):
                inv.setdefault(comp, set()).update(mids)
    for mid, resolved in snapshot_specs():
        for comp in _resolved_components(resolved):
            inv.setdefault(comp, set()).add(mid)
    _component_to_models_cache = {k: sorted(v) for k, v in inv.items()}
    return _component_to_models_cache


_encoder_only_cache: Optional[set] = None


def _encoder_only_components() -> set:
    """Library components the shipped models call only inside an encoder
    stack (`Stack.is_encoder`: a vision or audio tower). The playground
    compares components of the language model's step and leaves these
    out: an encoder call is sized by its media, which a module compare
    does not give. Read off each snapshot's `Program` — the stack every
    call sits in (`stack_of_block`, `stack_of`), each call counted for the
    library component its binding roots at (`binding_of`). A component
    some model calls outside an encoder, or that no snapshot calls, stays.
    Cached for the process, like `snapshot_specs`."""
    global _encoder_only_cache
    if _encoder_only_cache is not None:
        return _encoder_only_cache
    called: set = set()
    outside: set = set()
    names = set(get_default_loader().component_names())
    for _mid, resolved in snapshot_specs():
        program = Program.of(resolved)
        encoders = {stack.name for stack in program.stacks if stack.is_encoder}
        calls = [(template.meta.template_name, program.stack_of_block(index))
                 for index, template in enumerate(program.blocks)
                 if template.count]
        calls += [(node.op, program.stack_of(node))
                  for node in program.nodes() if node.is_component]
        for invoked, stack in calls:
            bound = {name for name in names
                     if binding_of(resolved.spec, invoked, name)}
            called |= bound
            if stack not in encoders:
                outside |= bound
    _encoder_only_cache = called - outside
    return _encoder_only_cache


def _diagnose_contract_error(
    base_resolved: Any,
    name: str,
    exc: Exception,
    dim_overrides: Dict[str, Any],
    loader: Any,
) -> Dict[str, Any]:
    """Turn a contract failure into an honest error entry: which dims are
    missing — facts only, no suggested values (a component that can't resolve
    under this base must be bound to a model that parameterizes it, not
    patched with a guess). Diagnosis itself is best-effort — a component-name
    KeyError re-raises from `inject_component`, leaving `missing_dims` empty
    and the original message intact."""
    missing: List[str] = []
    host = None
    try:
        host = inject_component(base_resolved, name, loader)
        missing = unbound_dims(host, dim_overrides, component=name)
    except CONTRACT_ERRORS:
        missing = []
    if (
        isinstance(exc, KeyError)
        and exc.args
        and isinstance(exc.args[0], str)
        and exc.args[0] not in missing
    ):
        missing = sorted([*missing, exc.args[0]])

    message = str(exc)
    if missing and "Cannot convert symbols" in message:
        message = (
            f"missing dims: {', '.join(missing)} — the base model does not bind them"
        )
    return {
        "component": name,
        "error": message,
        "missing_dims": missing,
    }


def inject_component(resolved: Any, name: str, loader: Any) -> Any:
    """Return a ResolvedSpec whose spec binds `name` and everything it calls,
    so `get_block(name)` resolves — bound by the loader the way lowering binds
    a component a spec names (`SpecLoader.inject_component`): what the host
    already resolved stands, nested template bindings are synthesized, and the
    component's own custom ops come with it.
    """
    if name not in loader.component_names():
        # A block the host declares itself (its `blocks:` or a template its
        # resolution specialized) already resolves, with everything it calls.
        if name in resolved.spec.all_block_names():
            return resolved
        raise KeyError(
            f"{name!r} is neither a library component (known: "
            f"{len(loader.component_names())} components) nor a block of "
            f"{resolved.spec.name}"
        )
    return resolved.with_spec(loader.inject_component(resolved.spec, name))


@dataclass(frozen=True)
class StepTraffic:
    """What one step moves besides the weights: activation reads and writes
    along the graph's edges, and the cache the step reads (`Traffic.kv_reads`,
    already the whole batch)."""

    act_reads: int
    act_writes: int
    kv_reads: int

    @classmethod
    def of_row(cls, row: Dict[str, Any]) -> "StepTraffic":
        """One per-op serving row's traffic, each figure already counting
        every layer that runs it. The read half holds the cache read, which
        `kv_read_bytes_per_rank` names, and the conversion a kernel makes of
        an operand no cast node could be grafted for
        (`convert_bytes_per_rank`, written and read back) — read-side
        traffic, as a serving phase splits its access (`mem_write_bytes` is
        the rows' `act_out` alone)."""
        kv = int(row["kv_read_bytes_per_rank"])
        return cls(
            act_reads=(int(row["act_in_bytes_per_rank"]) - kv
                       + int(row["convert_bytes_per_rank"])),
            act_writes=int(row["act_out_bytes_per_rank"]),
            kv_reads=kv,
        )

    @classmethod
    def of_rows(cls, rows: List[Dict[str, Any]]) -> "StepTraffic":
        """A step's traffic as serving prices it: the sum of its per-op rows
        (`serving_step`, fused or before the rewrite) read one by one
        (`of_row`)."""
        each = [cls.of_row(row) for row in rows]
        return cls(
            act_reads=sum(t.act_reads for t in each),
            act_writes=sum(t.act_writes for t in each),
            kv_reads=sum(t.kv_reads for t in each),
        )

    @property
    def act_bytes(self) -> int:
        return self.act_reads + self.act_writes


def serving_step(
    isolated: Any, *, batch: int, input_tokens: int, phase: str,
    include_aux_timing: bool = False, precision: Any = None,
    cells: Optional[StepCells] = None,
) -> AnalysisResult:
    """One step as serving prices it: `pipeline.analyze` on one rank, the
    phase's binding from `phase_binding`, with every fusion a shipped
    framework binding declares (`_UNION_BINDING`). Fusion is a rewrite of
    the step's one walk, so the step holds both sides of it: its rows fused
    (`per_op_timing`) and the same rows before the rewrite
    (`pre_per_op_timing`), each counting every layer that runs it.

    `cells` are the walks the caller's steps share (`step_eval.StepCells`):
    a caller that also times the step (`phase_roofline`) reads the same
    walk.

    `include_aux_timing` as in `pipeline.analyze`: a whole model prices its
    backbone only, while a benched vision / audio / MTP component IS the
    step, so its own leaves must not be dropped as aux.

    The rows are a step's numbers, which its callers sum: read without the
    breakdown an estimate draws (formula popovers, a collective's phases),
    whose numbers are the same (`pipeline.step_rows`)."""
    parallel = ParallelConfig()
    step = evaluate_step(
        isolated, parallel, phase=phase, batch=batch,
        input_tokens=input_tokens, output_tokens=0, precision=precision,
        cells=cells if cells is not None else StepCells(RuntimeProfile.ARCHITECTURE))
    return serving_analyze(
        isolated,
        parallel=parallel,
        framework_binding=_UNION_BINDING,
        batch_size=batch,
        step_eval=step,
        phase=phase,
        include_aux_timing=include_aux_timing,
        breakdown=False,
    )


def step_flops(rows: List[Dict[str, Any]]) -> int:
    """FLOPs one step runs, from its serving rows: every launch, the
    activation casts serving grafts among them, each counting every layer
    that runs it — the basis the step's bytes are on."""
    return sum(int(row["flops_per_rank"]) for row in rows)


def step_weight_read(rows: List[Dict[str, Any]]) -> int:
    """Weight bytes one step reads, from its serving rows: weights plus
    their quant metadata. A routed expert bank streams only what the step's
    tokens reach — `min(tokens × active share, bank)`,
    `RowRoofline.stored_read` — so a prefill reads essentially every expert
    and a batch-1 decode step only the ones it routes to; a dense weight is
    read whole in both. A weight's quant scales are read in the same
    proportion as the weight."""
    return sum(
        int(row["weight_read_bytes_per_rank"]) + int(row["metadata_bytes_per_rank"])
        for row in rows
    )


def phase_roofline(
    resolved: Any,
    *,
    batch: int,
    input_tokens: int,
    precision: Any,
    phase: str,
    gpu_sku: str,
    include_aux_timing: bool = False,
    model_id: Optional[str] = None,
    cells: Optional[StepCells] = None,
) -> Dict[str, Any]:
    """SOL and strictly-measured time for one phase of a resolved spec.

    The resource columns answer "how much work?"; time needs a hardware
    ceiling. This reuses the serving pipeline's per-op roofline (including the
    framework-union fusions the runtime table shows), then applies the measured
    envelope. `t_measured_s` is emitted only when EVERY runtime row has an exact
    physical measurement; partial and empty coverage stay `None` so a displayed
    total can never be mistaken for a fully measured end-to-end time.

    A whole-model compose prices the backbone (`include_aux_timing=False`);
    an isolated module IS the step, so its own leaves stay in. `cells` as
    in `serving_step`: a caller that read the step's rows already walked it.
    """
    gpu = get_gpu(gpu_sku)
    if gpu is None:
        raise ValueError(f"unknown gpu_sku {gpu_sku!r}")
    peak_bw = peak_hbm_bw(gpu)
    if peak_bw is None:
        raise ValueError(f"{gpu_sku} has no HBM bandwidth column")
    bound = phase_binding(
        phase=phase, avg_input_tokens=input_tokens, avg_output_tokens=0
    )
    parallel = ParallelConfig()
    # The step, walked once under the precision profile: its model dtype
    # decides which kernels the union binding claims and the comparator
    # peak; each leaf's compute peak is still its own column.
    step = evaluate_step(
        resolved, parallel, phase=phase, batch=batch,
        input_tokens=input_tokens, output_tokens=0, precision=precision,
        cells=cells if cells is not None else StepCells(RuntimeProfile.ARCHITECTURE))
    model_dtype = step.analysis.resolved_precision.model_dtype
    binding = _UNION_BINDING.deployed(ServingDeployment.at(
        gpu_sku=gpu_sku, activation_dtype=model_dtype,
    ))
    analysis = serving_analyze(
        resolved,
        parallel=parallel,
        framework_binding=binding,
        batch_size=batch,
        step_eval=step,
        peak_hbm_bw_bytes_per_sec=peak_bw,
        gpu=gpu,
        # The estimate's floors at their defaults: a module priced here and
        # in an estimate pays the same per launch.
        comm_latency_s=COMM_LATENCY_US_DEFAULT * 1e-6,
        launch_floor_s=LAUNCH_FLOOR_US_DEFAULT * 1e-6,
        phase=phase,
        include_aux_timing=include_aux_timing,
        breakdown=True,
    )
    rows = analysis.per_op_timing
    # A measured row is keyed on the weights' dtype as the checkpoint stores
    # them; a component that stores none has no such key, so no row can be
    # its measurement and its time stays SOL only.
    weight_dtype = analysis.analysis.dominant_weight_dtype()
    label = model_id or gpu_sku
    track = MeasuredTrack.unread(len(rows))
    folded = None
    if weight_dtype is not None:
        try:
            workload = WorkloadShape.from_request(
                batch_size=1,
                seq_len=max(1, int(bound["s_query"])),
            )
            deployment = build_deployment_context(
                framework=binding.framework,
                framework_binding=binding,
                parallel=parallel,
                gpu_sku=gpu_sku,
                phase=phase,
            )
            classified = CalibrationOverlay().apply(
                analysis, workload, deployment,
                model_id=label, spec_hash=resolved.identity_hash())
            rows = classified.per_op_timing
            # The step's one measured total, as an estimate's phase states it
            # (`batch_sweep.step_measure`), folded over one lookup.
            measure = step_measure(
                StepSum.of(rows), dict(zip(map(id, rows), classified.points, strict=True)),
                dict.fromkeys(map(id, rows), "per_op_timing"), gpu_sku=gpu_sku,
                framework=binding.framework, binding=binding, written={})
            point = PointMeasure.from_json(measure["measure"])
            select = measured_selector([point], selection_mode="envelope",
                                       measurement_scope=_measurement_scope())
            if select is not None:
                folded = point.fold(select)
                track = folded.track
                rows = [_measured_row(r, folded.rows[k]) for k, r in enumerate(rows)]
        except _measure_lookup.MeasurementsUnavailable:
            raise
        except Exception as exc:
            # A calibration identity failure must not take down the resource
            # bench. SOL still comes from the structural pipeline rows;
            # every row's measurement is declined, and says why.
            logger.warning("timing calibration failed for %s: %s", label, exc)
            reason = f"the measurement identity failed: {exc}"
            labels = [r.get("op") or r.get("aggregation_id") or "" for r in rows]
            rows = [{**r, "t_measured_s": None, "measured_state": DECLINED,
                     "measured_key": name, "measured_reason": reason}
                    for r, name in zip(rows, labels, strict=True)]
            track = MeasuredTrack.of([(DECLINED, name, reason) for name in labels])

    totals = phase_totals(rows, track)
    # The same rows as Inference's per-op tree: walked off the grafted +
    # fused tree, with subtree rollups and parent popovers; each group's
    # measured share is the fold's (`batch_sweep._measured_table`).
    module = {"flops_by_module": physical_module_rows(analysis, rows)}
    enrich_module_breakdown(module)
    if folded is not None:
        module["flops_by_module"] = _measured_table(
            module["flops_by_module"], terms_by_uid(r.get("uid") for r in rows), folded)
    return {
        "gpu_sku": gpu_sku,
        **totals,
        # The same totals as one object, for a caller that reads only them.
        "totals": totals,
        "strategy": "envelope",
        # The full per-op roofline rows (post-calibration, so t_measured_s is
        # in them). The Details table renders these directly; the sums above
        # stay their phase rollup.
        "op_rows": served_rows(rows),
        # The module tree over those rows (Inference's `flops_by_module`).
        "module_rows": served_rows(module["flops_by_module"]),
        # The per-op table's Total row, over those module rows.
        "op_table_total": module["op_table_total"],
        # Display, not the time: the same Σ_leaf max(...) identity the
        # block's per-op table prices, with the largest leaves named.
        "t_sol_formula": _t_sol_sum_formula(
            rows, program=analysis.program, t_sol=totals["t_sol_s"],
            t_compute=totals["t_compute_s"], t_memory=totals["t_memory_s"],
            t_comm=totals["t_comm_s"],
        ),
    }


def phase_totals(rows: List[Dict[str, Any]], measured: MeasuredTrack) -> Dict[str, Any]:
    """A phase's time: SOL summed over its per-op roofline rows, the leaves'
    compute / memory / comm times summed apart (each leaf is priced at its
    max of its four bounds; these say which side the step leans on), and the
    measured total and coverage as the measured fold states them
    (`auto_sweep.MeasuredTrack`: the formula's total only when every row is
    measured) — never re-added here. The rows' FLOPs and HBM access, added
    the way a serving step adds them (`StepSum`), give the phase's
    arithmetic intensity."""
    step = StepSum.of(rows)
    flops = step.total(lambda t: float(t.row.get("flops_per_rank") or 0.0))
    access = step.total(lambda t: float(t.row.get("access_bytes_per_rank") or 0.0))
    return {
        "t_sol_s": sum(float(row.get("t_sol_s") or 0.0) for row in rows),
        "t_measured_s": measured.t_s,
        "t_compute_s": sum(float(row.get("t_compute_s") or 0.0) for row in rows),
        "t_memory_s": sum(float(row.get("t_memory_s") or 0.0) for row in rows),
        "t_comm_s": sum(float(row.get("t_comm_s") or 0.0) for row in rows),
        "measured_rows": measured.measured_rows,
        "total_rows": measured.total_rows,
        "coverage": measured.coverage(),
        "flops": flops,
        "access_bytes": access,
        "arithmetic_intensity": arithmetic_intensity(flops, access),
    }


def _timing_view(
    isolated: Any,
    name: str,
    *,
    batch: int,
    input_tokens: int,
    site: CallSite,
    precision: Any,
    phase: str,
    gpu_sku: str,
    cells: Optional[StepCells] = None,
) -> Dict[str, Any]:
    """SOL / measured time for one phase of an isolated component
    (`isolate_resolved`), off the walk its runtime table read when the
    caller shares it (`cells`)."""
    return phase_roofline(
        isolated,
        batch=batch,
        input_tokens=input_tokens,
        precision=precision,
        phase=phase,
        gpu_sku=gpu_sku,
        include_aux_timing=True,
        model_id=site.model_id or name,
        cells=cells,
    )


def _weight_storage(analysis: Any, site: CallSite, *, bound: bool
                    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One copy of the component's weights as the checkpoint stores them:
    `capacity.weights` (parameters, their quant metadata, buffers) and the
    parameter bytes per dtype (`Analysis.weight_bytes_by_dtype`), priced at
    the call site's checkpoint formats. `precision` names the dtypes largest
    first; `source` says whose checkpoint that is — the bound model's, or,
    for an unbound entry, the base model's quant context. Beside it, the same
    ledger split by node (`Weights.by_uid`) — the parts of `bytes`."""
    ledger = capacity_mod.weights(analysis)
    by_dtype = sorted(analysis.weight_bytes_by_dtype().items(), key=lambda kv: -kv[1])
    names = [display_name(dtype) for dtype, _ in by_dtype]
    precision = (
        "checkpoint precision: "
        + names[0]
        + (f", some {', '.join(names[1:])}" if names[1:] else "")
        if names
        else "no weights"
    )
    model = site.model_id or "the base model"
    note = (
        f"as stored in {model}'s checkpoint"
        if bound
        else f"unbound: priced at {model}'s quant context"
    )
    if not site.invoked:
        note += f", in the place of the component called at {site.label}"
    parts: List[Dict[str, Any]] = []
    for uid, held in ledger.by_uid.items():
        n = int(held.resident)
        if n <= 0:
            continue
        node = analysis.program.by_uid(uid)
        term: Dict[str, Any] = {"label": node.name, "result": n}
        cost = analysis.cost.values.get(uid)
        params = int(cost.params) if cost is not None else 0
        param_bytes = int(held.parameters)
        if cost is not None and params > 0 and param_bytes % params == 0:
            bpe = param_bytes // params
            record = analysis.params_bindings.values.get(uid)
            sym, val = _bound_cost(cost.params, record)
            extra_n = int(held.metadata) + int(held.buffers)
            if bpe != 1:
                if sym:
                    sym = f"({sym}) * {bpe}"
                if val:
                    val = f"({val}) * {bpe}"
            if extra_n:
                if val:
                    val = f"{val} + {extra_n}"
                if sym:
                    sym = f"{sym} + {extra_n}"
            if sym:
                term["symbolic"] = sym
            if val:
                term["substituted"] = val
        parts.append(term)
    return {
        "bytes": int(ledger.resident),
        "parameter_bytes": int(ledger.parameters),
        "metadata_bytes": int(ledger.metadata),
        "buffer_bytes": int(ledger.buffers),
        "by_dtype": [
            {"dtype": dtype, "label": label, "bytes": int(b)}
            for (dtype, b), label in zip(by_dtype, names, strict=True)
        ],
        "precision": precision,
        "source": {
            "model_id": site.model_id,
            # True: the entry is bound, and these are its model's formats.
            # False: an unbound library entry keeps the base model's.
            "bound": bound,
            # Whether that model invokes this component itself, or it takes
            # the place of the component called there (`CallSite.of`).
            "invoked": site.invoked,
            "role": site.role,
            "address": site.address,
            "note": note,
        },
    }, parts


def _storage_law(tensor: Any) -> Dict[str, Any]:
    """Classify a cached tensor's position law from the law itself
    (`cache_view.Rate.of`, the split eval bills KV growth by).

    `growing`   positions per token > 0: dense KV, a compressed pool.
    `windowed`  the length is in the law but adds nothing once it is past
                the window (SWA); the window is the positions it holds.
    `constant`  no length in the law: a recurrent state, one value per
                request.
    """
    if not tensor.is_growing:
        return {"law": "constant"}
    rate = Rate.of(tensor)
    if rate.per_token > 0:
        return {"law": "growing"}
    return {"law": "windowed", "window": int(rate.fixed)}


def _kv_tensor_term(
    tensor: Any, cache_len: int, *, read: bool = False,
    env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One cached tensor's storage identity: what one position occupies
    (`CachedTensor.position_bytes` — its data and the dynamic scales its
    format stores beside it) × the positions law, over the position owners.

    The law keeps `s_kv` free. `cache_len` only substitutes the values line.
    `read=True` uses the tensor's read law (DSA keeps every position and
    reads top-K of what it kept).
    """
    law = tensor.read if read else tensor.stored
    if law is None:
        law = CACHE_LEN
    positions = tensor.positions(cache_len, read=read)
    pos_env = dict(env or {})
    pos_env["s_kv"] = cache_len
    pos_val = _bind_extent(positions, pos_env, expr=expr_of(law),
                           expect=int(value_of(positions) or 0))
    width = tensor.position_bytes
    symbolic = (f"{_wrap_expr(_star(str(_clean_expr(width))))} * "
                f"{_star(str(expr_of(law)))}")
    values = f"{_wrap_expr(_bind_extent(width, env))} * {pos_val}"
    over = int(tensor.positions_over or 1)
    if over != 1:
        symbolic += f" / {over}"
        values += f" / {over}"
    val = int(tensor.bytes_at(cache_len, read=read))
    return {
        "label": tensor.name,
        "symbolic": symbolic,
        "substituted": values,
        "result": val,
    }


def _kv_elements_formula(
    tensor: Any, *, shown: int, per_token: bool, cache_len: int,
    env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Popover for one aligned-tensor cell: the identity of the number it shows.

    A growing / windowed cell is elements per token. A constant cell is
    elements per request (`elements ×` the law at this length). Substituted
    numbers are that cell, so the popover cannot drift from the table.
    """
    el = _star(str(expr_of(tensor.elements)))
    unit = "el/tok" if per_token else "el/req"
    el_val = _bind_extent(tensor.elements, env)
    if per_token:
        return {
            "symbolic": el,
            "substituted": f"{el_val} = {shown:,d} {unit}",
        }
    law = tensor.stored if tensor.stored is not None else CACHE_LEN
    pos_sym = _star(str(expr_of(law)))
    pos_n = int(value_of(tensor.positions(cache_len)) or 0)
    if pos_n == 1:
        return {
            "symbolic": el,
            "substituted": f"{el_val} = {shown:,d} {unit}",
        }
    pos_env = dict(env or {})
    pos_env["s_kv"] = cache_len
    pos_val = _bind_extent(
        tensor.positions(cache_len), pos_env,
        expr=expr_of(law), expect=pos_n)
    return {
        "symbolic": f"{el} * {pos_sym}",
        "substituted": f"{el_val} * {pos_val} = {shown:,d} {unit}",
    }


def _kv_node_dto(
    tensor: Any, *, owner: Optional[str], cache_len: int,
    env: Dict[str, Any], sweep_lens: Tuple[int, ...] = (),
) -> Dict[str, Any]:
    """One cache tensor as the aligned KV table holds it.

    `owner` names the bound slot for a sub-component cache. Those rows explain
    cache marks on the graph without adding their bytes to the parent entry:
    CSA2's shared compressed/index caches are held by source-layer slots, not
    by every consumer layer that reads them.
    """
    store = int(tensor.bytes_at(cache_len, read=False))
    node = {
        "name": tensor.name,
        "dtype": tensor.dtype,
        "growing": bool(tensor.is_growing),
        "elements_per_token": int(tensor.elements) if tensor.is_growing else None,
        "storage_bytes_at_L": store,
        "read_bytes_at_L": int(tensor.bytes_at(cache_len, read=True)),
    }
    if owner is not None:
        node["owner"] = owner
    if not tensor.is_growing:
        node["elements_per_request"] = int(
            tensor.elements * tensor.positions(cache_len))
    node.update(_storage_law(tensor))
    shown = (node["elements_per_token"] if tensor.is_growing
             else node["elements_per_request"])
    node["formula"] = _kv_elements_formula(
        tensor, shown=int(shown), per_token=tensor.is_growing,
        cache_len=cache_len, env=env,
    )
    if owner is not None and sweep_lens:
        node["storage_by_cache_len"] = [
            {
                "cache_len": length,
                "storage_bytes": int(tensor.bytes_at(length, read=False)),
            }
            for length in sweep_lens
        ]
    return node


def _cap_terms(parts: List[Any], *, value) -> tuple:
    """Largest-first, same 20-term ceiling the inference Summary uses."""
    ordered = sorted(parts, key=value, reverse=True)
    if len(ordered) <= MAX_COMPONENT_TERMS:
        return ordered, 0
    return ordered[:MAX_COMPONENT_TERMS], len(ordered) - MAX_COMPONENT_TERMS


def _hidden_note(hidden: int) -> str:
    return f"+ {hidden} smaller terms" if hidden else ""


def _join_notes(*notes: str) -> str:
    return "; ".join(n for n in notes if n)


def _addends_formula(
    terms: List[Dict[str, Any]],
    *,
    total: int,
    unit: str,
    note: str = "",
) -> Optional[Dict[str, Any]]:
    """The one popover shape for a sum of named parts — or None when the
    parts are not the parts of `total`: a popover that does not add up to
    its cell explains nothing, so there is none.

    A term is `{label, result, symbolic?, substituted?, formula?}`:
    `result` is its number, `substituted` its values without the result,
    `formula` a popover the term already has for itself (its own total).
    One part: its identity is the formula (nothing to expand). Several:
    the formula is the parts' IDs; a part that has its own identity
    expands as a subterm. Largest-first, capped at 20; the parts the
    ceiling drops are added as one number, so the line still adds up to
    `total`, the cell's number.
    """
    suffix = f" {unit}" if unit else ""
    if sum(int(t["result"]) for t in terms) != int(total):
        logger.warning("popover parts %s do not add up to %s",
                       [t["label"] for t in terms], total)
        return None
    if not terms:
        return {"symbolic": "0", "substituted": f"= {int(total):,d}{suffix}"}
    shown, hidden = _cap_terms(terms, value=lambda t: int(t["result"]))
    if len(terms) == 1:
        t = shown[0]
        if t.get("formula"):
            return t["formula"]
        values = t.get("substituted")
        return {
            "symbolic": t.get("symbolic") or t["label"],
            "substituted": (f"{values} = " if values else "= ")
            + f"{int(total):,d}{suffix}",
        }
    rest = sum(int(t["result"]) for t in terms) - sum(
        int(t["result"]) for t in shown)
    out: Dict[str, Any] = {
        "symbolic": " + ".join(t["label"] for t in shown),
        "substituted": (
            " + ".join(f"{int(t['result']):,d}" for t in shown)
            + (f" + {rest:,d}" if hidden else "")
            + f" = {int(total):,d}{suffix}"
        ),
    }
    if any(t.get("symbolic") or t.get("formula") for t in shown):
        out["subterms"] = [_subterm(t) for t in shown]
    joined = _join_notes(note, _hidden_note(hidden))
    if joined:
        out["note"] = joined
    return out


def _subterm(term: Dict[str, Any]) -> Dict[str, Any]:
    """One part as a popover subterm: its own popover when it has one."""
    own = term.get("formula") or term
    sub = {"label": term["label"], "result": int(term["result"])}
    for key in ("symbolic", "substituted", "subterms", "note"):
        if own.get(key):
            sub[key] = own[key]
    return sub


def _kv_sum_formula(terms: List[Dict[str, Any]], *, unit: str = "B") -> Dict[str, Any]:
    return _addends_formula(
        terms, total=sum(int(t["result"]) for t in terms), unit=unit)


def _rows_sum_formula(
    rows: List[Dict[str, Any]],
    *,
    value_fn: Any,
    unit: str,
    total: int,
    note: str = "",
    identity: Optional[str] = None,
) -> Dict[str, Any]:
    """Runtime-table kernels as addends of `_addends_formula`; `identity`
    names the row's own popover for the value (`row["formulas"]`)."""
    terms: List[Dict[str, Any]] = []
    for row in rows:
        val = int(value_fn(row) or 0)
        if val <= 0:
            continue
        term: Dict[str, Any] = {"label": row["label"], "result": val}
        if identity is not None:
            own = (row.get("formulas") or {}).get(identity)
            if own:
                term["formula"] = own
        terms.append(term)
    return _addends_formula(terms, total=int(total), unit=unit, note=note)


def _t_sol_sum_formula(
    rows: List[Dict[str, Any]],
    *,
    program: Any,
    t_sol: float,
    t_compute: float,
    t_memory: float,
    t_comm: float,
) -> Dict[str, Any]:
    """t SOL popover: the roofline identity, with the largest leaves named."""
    parts: List[tuple] = []
    for row in rows:
        t = float(row.get("t_sol_s") or 0)
        if t <= 0:
            continue
        parts.append((program.by_uid(row["uid"]).name, t))
    shown, hidden = _finalize_parts(parts)
    bound_note = (
        f"{STEP_SOL_BOUND_NOTE}: Σt_compute={t_compute:.6g}s, "
        f"Σt_memory={t_memory:.6g}s, Σt_comm={t_comm:.6g}s"
    )
    out: Dict[str, Any] = {
        "symbolic": STEP_SOL_LAW,
        "substituted": (
            _substituted_sum(shown, t_sol, _fmt_time) if shown
            else f"= {t_sol:.6g} s"
        ),
        "note": _join_notes(bound_note, _hidden_note(hidden)),
    }
    return out


def _kv_view(
    host: Any,
    name: str,
    isolated: Any,
    *,
    cache_len: int,
    precision: Any,
    dim_overrides: Dict[str, Any],
    site: CallSite,
    sweep_lens: Tuple[int, ...] = (),
) -> Tuple[Dict[str, Any], int]:
    """Per-component kvcache: mechanism + storage family + per-node storage
    law + bytes at L, and the storage at each of `sweep_lens` under the same
    per-tensor laws (a dense tensor grows, a window caps, a recurrent state
    stays flat). The second return is the decode read at L (one request),
    used internally to size decode `kv_step` / `memory_dominant`.

    Every tensor comes from `of_component` + `CachedTensor.bytes_at` (each
    node's own dtype width and `CACHE_LEN` law, so no `elements*dtype` here),
    the recurrent state included — isolation resolves it the way the
    full-model walk does, so serving and this table bill the same state.
    `isolated` is the component's `analyze_isolated` walk.
    """
    env = _walk_env(isolated)
    topo = kv_cache_topology(host, name, dim_overrides, site)
    tensors = of_component(isolated, name)
    program = isolated.program
    component_calls = [
        node for node in program.nodes()
        if node.children and binding_of(program.spec, node.op, name)
    ]
    if len(component_calls) != 1:
        raise ValueError(
            f"isolating {name!r} produced {len(component_calls)} calls to it; "
            "its cache ownership cannot be read")
    component_call = component_calls[0]

    # A parent cache declaration absorbs the sub-component tensors it names
    # (`cache._of_component`); those are already in `tensors` and must not be
    # explained a second time. Direct slot calls are the ownership roots: a
    # slot's own `by_call` entry carries its nested names without also walking
    # the grandchild call and listing the same storage at two ownership levels.
    parent_tensor_nodes = {
        tensor.node for tensor in tensors if tensor.node is not None
    }
    calls_by_uid = by_call(isolated).values
    slot_tensors: List[Tuple[str, Any]] = []
    for child in component_call.children or ():
        if not child.children:
            continue
        entry = calls_by_uid.get(child.uid)
        if entry is None:
            continue
        slot_tensors.extend(
            (child.name, tensor) for tensor in entry.tensors
            if tensor.node not in parent_tensor_nodes
        )

    nodes: List[Dict[str, Any]] = []
    storage_total = 0
    read_total = 0
    per_token = 0
    for tensor in tensors:
        store_b = int(tensor.bytes_at(cache_len, read=False))
        read_b = int(tensor.bytes_at(cache_len, read=True))
        storage_total += store_b
        read_total += read_b
        if tensor.is_growing:
            # One position under the tensor's own law: what each new token
            # adds to this layer's cache.
            per_token += int(tensor.bytes_at(1, read=False))
        nodes.append(_kv_node_dto(
            tensor, owner=None, cache_len=cache_len, env=env,
            sweep_lens=sweep_lens))
    slot_nodes = [
        _kv_node_dto(tensor, owner=owner, cache_len=cache_len, env=env,
                     sweep_lens=sweep_lens)
        for owner, tensor in slot_tensors
    ]

    # Every tensor carries its own law (a recurrent state's is constant), so
    # the sweep is a plain sum: dense grows, a window caps, a state stays flat.
    sweep = []
    for n in sweep_lens:
        terms = [_kv_tensor_term(t, n, env=env) for t in tensors]
        sweep.append({
            "cache_len": n,
            "storage_bytes": sum(int(t["result"]) for t in terms),
            "formula": _kv_sum_formula(terms),
        })
    grow = [_kv_tensor_term(t, 1, env=env) for t in tensors if t.is_growing]
    state = [
        _kv_tensor_term(t, cache_len, env=env)
        for t in tensors if not t.is_growing
    ]
    store_terms = [_kv_tensor_term(t, cache_len, env=env) for t in tensors]
    read_terms = [
        _kv_tensor_term(t, cache_len, read=True, env=env) for t in tensors
    ]

    formulas = {
        "bytes_per_token": _kv_sum_formula(grow),
        "state_bytes": _kv_sum_formula(state),
        "storage": _kv_sum_formula(store_terms),
        "read": _kv_sum_formula(read_terms),
    }
    if slot_nodes:
        note = (
            "counts the component-owned tensors only; slot-owned shared caches "
            "are listed beside the graph tensors and are stored once by their "
            "owning bound slot, not by every layer that reads them"
        )
        for formula in formulas.values():
            if formula is not None:
                formula["note"] = _join_notes(formula.get("note") or "", note)

    return {
        # The mechanism users see (`attn_display`, the same label the
        # compute graph's KV marks carry) and, separately, the family the
        # cache is shaped as: KDA keeps a GatedDeltaNet-shaped state.
        "display": host.spec.get_block(name).attn_display,
        "family": topo.label,
        "storage_bytes_at_L": storage_total,
        # What each new token writes to this layer's cache (growing nodes).
        "bytes_per_token": per_token,
        # State that does not grow with length (a recurrent state): constant
        # per request, part of storage_bytes_at_L, never per-token KV.
        "state_bytes": sum(
            n["storage_bytes_at_L"] for n in nodes if n["law"] == "constant"
        ),
        # Unit contract for the frontend (which must not compute anything):
        # storage/read bytes are ONE layer, ONE request, AT this cache_len;
        # a recurrent state is cache_len-independent (constant per request).
        "unit": f"bytes/layer/request @ cache_len = {cache_len:,} tokens",
        "cache_len": cache_len,
        # Storage per layer per request at each requested context length.
        "storage_by_cache_len": sweep,
        "nodes": nodes,
        "slot_nodes": slot_nodes,
        "formulas": formulas,
    }, read_total


def _block_graph(
    host: Any,
    name: str,
    *,
    batch: int,
    input_tokens: int,
    dim_overrides: Dict[str, Any],
    precision: Any,
    site: CallSite,
) -> Dict[str, Any]:
    """The isolated component's node/edge DAG for the architecture diagram and
    the per-node param/flops breakdown — same builder the Inspector renders with
    (`build_block_graph_data` → BlockElkDag), run on the one-stage isolated spec
    so it carries exactly this component's ops (no embedding/head/other layers).
    """
    isolated = isolate_resolved(host, name, dim_overrides, site)
    blocks, _unmatched = build_block_graph_data(
        isolated, batch_size=batch, seq_len=input_tokens, precision=precision
    )
    entry = blocks.get(isolated.site.block, {})
    # Only the fields the frontend consumes (BlockElkDag nodes/edges + the
    # per-node params/flops the breakdown sums); drop the rest to bound payload.
    return {
        "nodes": entry.get("nodes", []),
        "edges": entry.get("edges", []),
    }


def _fused_row_tooltip(row: Dict[str, Any]) -> str:
    """The runtime row as a fused graph node's hover card."""
    lines = [
        f"kernel: {row['kernel']}",
        f"params: {int(row['params']):,d}",
        f"active params: {int(row['active_params']):,d}",
        f"FLOPs: {int(row['flops']):,d}",
        (
            "HBM: "
            f"activation {int(row['act_read_bytes']) + int(row['act_write_bytes']):,d} B"
            f" + weights {int(row['weight_read_bytes']):,d} B"
            f" + KV {int(row['kv_read_bytes']):,d} B"
            f" = {int(row['total_bytes']):,d} B"
        ),
    ]
    origins = list(row.get("origins") or [])
    if origins:
        lines.insert(1, "absorbed logical ops: " + " + ".join(origins))
    if row.get("arithmetic_intensity") is not None:
        lines.append(
            f"arithmetic intensity: {row['arithmetic_intensity']:,.2f} FLOP/B")
    return "<br/>".join(escape(line) for line in lines)


def _stamp_fused_resources(
    nodes: List[Dict[str, Any]], table: Dict[str, Any],
    live_uids: Optional[Collection[int]] = None,
) -> None:
    """Put every priced runtime row onto its post-fusion graph node.

    The join is an invariant, not a best-effort annotation: a missing graph
    node or a runtime row with no place to draw refuses the whole graph. A
    silent gap here would make “Fused” look cheaper than the table beside it.
    """
    rows: Dict[int, Dict[str, Any]] = {}
    for row in table["rows"]:
        uid = int(row["uid"])
        if uid in rows:
            raise ValueError(f"runtime table repeats priced uid {uid}")
        rows[uid] = row
    expected_uids = set(rows)
    if live_uids is not None:
        expected_uids = {int(uid) for uid in live_uids}
        rows_not_live = set(rows) - expected_uids
        if rows_not_live:
            raise ValueError(
                "runtime table has rows outside the live fused kernels: "
                f"{sorted(rows_not_live)}")
    nodes_by_uid: Dict[int, Dict[str, Any]] = {}

    def collect(items: List[Dict[str, Any]]) -> None:
        for node in items:
            uid = node.get("uid")
            if uid in expected_uids:
                if uid in nodes_by_uid:
                    raise ValueError(
                        f"fused graph has two priced nodes at uid {uid}")
                nodes_by_uid[int(uid)] = node
            collect(node.get("children") or [])

    collect(nodes)
    graph_uids = set(nodes_by_uid)
    rows_without_nodes = set(rows) - graph_uids
    live_without_nodes = expected_uids - graph_uids
    if rows_without_nodes or live_without_nodes:
        raise ValueError(
            "fused graph and runtime table disagree on priced kernels: "
            f"rows without a node {sorted(rows_without_nodes)}, "
            f"live kernels without a node {sorted(live_without_nodes)}")

    for uid, node in nodes_by_uid.items():
        row = rows.get(uid)
        if row is None:
            # A zero-cost live leaf (for example a view) still has to draw,
            # but has no runtime resource row to stamp.
            continue
        node.update({
            "params": int(row["params"]),
            "active_params": int(row["active_params"]),
            "flops": int(row["flops"]),
            "weight_bytes": int(row["weight_read_bytes"]),
            "out_bytes": int(row["act_write_bytes"]),
            "access_bytes": int(row["total_bytes"]),
            "arithmetic_intensity": row["arithmetic_intensity"],
            "priced_runtime": True,
            "hoverHtml": _fused_row_tooltip(row),
        })
        if len(row.get("origins") or []) > 1:
            node["badge"] = f"{len(row['origins'])}→1"

    totals = table["totals"]
    sums = {
        "params": sum(int(row["params"]) for row in rows.values()),
        "active_params": sum(
            int(row["active_params"]) for row in rows.values()),
        "flops": sum(int(row["flops"]) for row in rows.values()),
        "act_read_bytes": sum(
            int(row["act_read_bytes"]) for row in rows.values()),
        "act_write_bytes": sum(
            int(row["act_write_bytes"]) for row in rows.values()),
        "weight_read_bytes": sum(
            int(row["weight_read_bytes"]) for row in rows.values()),
        "kv_read_bytes": sum(
            int(row["kv_read_bytes"]) for row in rows.values()),
        "total_bytes": sum(int(row["total_bytes"]) for row in rows.values()),
    }
    for key, value in sums.items():
        if value != int(totals[key]):
            raise ValueError(
                f"fused graph pricing does not close on {key}: "
                f"{value:,d} != {int(totals[key]):,d}")


def _strip_scope(value: Optional[str], prefix: str) -> Optional[str]:
    """Remove the whole-model serializer's block namespace from a node/ref."""
    if value is None:
        return None
    return value[len(prefix):] if value.startswith(prefix) else value


def _strip_serving_node(node: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    def edge(edge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **edge,
            "id": _strip_scope(edge.get("id"), prefix),
            "src": _strip_scope(edge.get("src"), prefix),
            "dst": _strip_scope(edge.get("dst"), prefix),
        }

    return {
        **node,
        "id": _strip_scope(node.get("id"), prefix),
        "inputs": [
            _strip_scope(ref, prefix) for ref in node.get("inputs") or []
        ],
        "children": [
            _strip_serving_node(child, prefix)
            for child in node.get("children") or []
        ],
        "child_edges": [
            edge(child_edge) for child_edge in node.get("child_edges") or []
        ],
    }


def _validate_scoped_graph(graph: Dict[str, Any]) -> None:
    """Refuse a serializer change that leaves an edge outside its scope."""
    node_ids: set = set()

    def collect_nodes(items: List[Dict[str, Any]]) -> None:
        for node in items:
            node_id = node.get("id")
            if not node_id:
                raise ValueError("fused graph has a node without an id")
            if node_id in node_ids:
                raise ValueError(f"fused graph repeats node id {node_id}")
            node_ids.add(node_id)
            collect_nodes(node.get("children") or [])

    collect_nodes(graph.get("nodes") or [])

    edge_ids: set = set()

    def edge_id(edge: Dict[str, Any]) -> None:
        found = edge.get("id") or f"{edge.get('src')}->{edge.get('dst')}"
        if found in edge_ids:
            raise ValueError(f"fused graph repeats edge id {found}")
        edge_ids.add(found)

    root_ids = {node.get("id") for node in graph.get("nodes") or []}
    for edge in graph.get("edges") or []:
        if edge.get("src") not in root_ids or edge.get("dst") not in root_ids:
            raise ValueError(
                f"fused root edge {edge.get('id')} leaves its component scope")
        edge_id(edge)

    def walk(items: List[Dict[str, Any]]) -> None:
        for node in items:
            child_ids = {
                child.get("id") for child in node.get("children") or []
            }
            for edge in node.get("child_edges") or []:
                if (edge.get("src") not in child_ids
                        or edge.get("dst") not in child_ids):
                    raise ValueError(
                        f"fused child edge {edge.get('id')} leaves "
                        f"the scope of {node.get('id')}")
                edge_id(edge)
            walk(node.get("children") or [])

    walk(graph.get("nodes") or [])


def _unwrap_isolated_serving_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap the serving serializer's one-block shell to Playground's shape.

    `serialize_post_tree` speaks the Inference page's whole-model spine
    (`input -> block_stack -> component`). Reusing it gets fused compounds,
    port-aware edges and KV marks; unwrapping the isolated block avoids two
    placeholder boxes before the component's kernels appear.
    """
    blocks = [
        node for node in graph.get("nodes") or []
        if node.get("op") == "block_stack"
    ]
    if len(blocks) != 1:
        raise ValueError(
            "an isolated component must draw exactly one block stack, got "
            f"{len(blocks)}")
    block = blocks[0]
    children = list(block.get("children") or [])
    inputs = [node for node in children if node.get("op") == "input"]
    bodies = [node for node in children if node.get("op") != "input"]
    if len(bodies) != 1:
        raise ValueError(
            "an isolated component must draw exactly one component body, got "
            f"{len(bodies)}")
    prefix = f"{block.get('id')}/"

    # Normal isolation has one component wrapper whose children are the live
    # kernels. Synthesize the input edges that removing that wrapper fans
    # directly to its entry children.
    if len(bodies) == 1 and bodies[0].get("children"):
        body = bodies[0]
        nodes = [*inputs, *(body.get("children") or [])]
        input_ids = {
            _strip_scope(node.get("id"), prefix) for node in inputs
        }
        edges = [dict(edge) for edge in body.get("child_edges") or []]
        for node in nodes:
            for index, ref in enumerate(node.get("inputs") or []):
                src = _strip_scope(ref, prefix)
                if src not in input_ids:
                    continue
                source = next((
                    candidate for candidate in inputs
                    if _strip_scope(candidate.get("id"), prefix) == src
                ), None)
                ports = node.get("input_ports") or []
                edges.append({
                    "id": (
                        f"{src}->{_strip_scope(node.get('id'), prefix)}:{index}"
                    ),
                    "src": src,
                    "dst": _strip_scope(node.get("id"), prefix),
                    "tensor": "",
                    "shape": (source or {}).get("tensor_out", ""),
                    "port_label": ports[index] if index < len(ports) else "",
                })
    else:
        # A leaf component has no wrapper to unwrap; keep its direct nodes.
        nodes = children
        edges = [dict(edge) for edge in block.get("child_edges") or []]

    def scoped_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **edge,
            "id": _strip_scope(edge.get("id"), prefix),
            "src": _strip_scope(edge.get("src"), prefix),
            "dst": _strip_scope(edge.get("dst"), prefix),
        }

    out = {
        "nodes": [_strip_serving_node(node, prefix) for node in nodes],
        "edges": [scoped_edge(edge) for edge in edges],
    }
    _validate_scoped_graph(out)
    return out


def _fused_l1(graph: Dict[str, Any], table: Dict[str, Any]) -> Dict[str, Any]:
    """Structure boxes over the runtime graph, closed to its priced rows.

    The serving serializer keeps semantic scopes (router, experts, shared
    experts) as compounds. L1 partitions one scope, so this projects each root
    to its runtime aggregate before calling the same abstraction the declared
    view uses. The projection is local to L1: the detailed Fused graph keeps
    every live kernel and fused compound untouched.
    """
    facts = ("flops", "params", "active_params", "weight_bytes", "out_bytes",
             "access_bytes")

    def runtime_total(node: Dict[str, Any]) -> Dict[str, int]:
        values = {fact: 0 for fact in facts}

        def walk(items: List[Dict[str, Any]]) -> None:
            for item in items:
                if item.get("priced_runtime"):
                    for fact in facts:
                        values[fact] += int(item.get(fact) or 0)
                walk(item.get("children") or [])

        walk(node.get("children") or [])
        return values

    projection: List[Dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        # The children stay: L1 reads no fact off them, only which root
        # holds a member a sibling reads (a consumer rewired to a fused
        # kernel inside the router reads the router, as the edge draws it).
        projected = {
            **node,
            **({"priced_runtime": True} if node.get("priced_runtime") else {}),
            **(
                {fact: node[fact] for fact in facts}
                if node.get("priced_runtime") else runtime_total(node)
            ),
        }
        # L1 reads presentation folds beside its own numeric facts. This
        # projection's facts are the runtime rows alone; stale laws and folded
        # presentation from the detailed scope must not be added beside them.
        for key in ("absorbed_params", "flops_law", "flops_symbolic",
                    "params_symbolic"):
            projected.pop(key, None)
        projection.append(projected)

    # The declared structure reconciles stored weight bytes. This is the runtime
    # structure, so its third leg is the prefill walk's weight READ bytes.
    from backend.views.graph_view import l1_of_nodes

    totals = table["totals"]
    view = l1_of_nodes(
        projection,
        Sums(
            flops=int(totals["flops"]),
            params=int(totals["params"]),
            weight_bytes=int(totals["weight_read_bytes"]),
        ),
    )
    view["weight_semantics"] = "weight_read_step"
    return view


def _fused_graph(
    served: AnalysisResult,
    op_table: Dict[str, Any],
) -> Dict[str, Any]:
    """Serialize the serving step that priced `op_table`.

    The table and graph therefore share one program, rewrite, UID space and
    set of fusion matches — the serving step's (`serving_step`), drawn the
    way the inference page draws it. This is deliberately not a second
    serving analysis.
    """
    graph = serialize_post_tree(
        served.post_tree,
        served.analysis.program.resolved,
        # Absorbed subgraphs resolve against the tree fusion operated on:
        # the walk's, with its collectives and activation casts grafted.
        pre_tree=served.comm_tree,
        fusion_matches=served.fusion_matches,
        program=served.program,
        # Playground is single-rank, so the per-rank shapes are the full ones.
        shapes=served.shapes,
        full_shapes=served.full_shapes,
        pre_program=served.pre_program,
        pre_shapes=served.pre_shapes,
        pre_full_shapes=served.pre_full_shapes,
        kv=KVMarks.of_serving(served),
    )
    _stamp_fused_resources(
        graph.get("nodes") or [], op_table,
        live_uids={int(row["uid"]) for row in served.per_op_timing})
    fused = _unwrap_isolated_serving_graph(graph)
    return fused, _fused_l1(fused, op_table)


def _runtime_op_table(
    served: AnalysisResult,
    *,
    batch: int,
    input_tokens: int,
    precision: Any,
    phase: str,
    include_fused_graph: bool = False,
) -> Dict[str, Any]:
    """One phase's live runtime kernels as serving runs them — and the
    phase's numbers: the row's cells (params, FLOPs, the fused activation,
    weight and KV reads) are the `totals` of these rows, and its popovers
    list them, so a cell and its explanation are one set of numbers.

    `served` is the isolated component's serving step (`serving_step`):
    the walk, its collectives, its activation casts and the shipped
    fusions, in the serving graph's order (`serving.rewrite_serving_graph`),
    so an FP8 module's activation casts are launches here exactly where the
    serving graph has them — a fusion that takes a cast in prices it in its
    launch, one that does not leaves it standing as its own row. Each row's
    bytes and FLOPs are its serving row's (`per_op_timing`); its params and
    the popovers' identities are read off the same step's layers. The walk
    stays single-rank — module comparison is canonical/unsharded, not a
    deployment layout. Rows with only traffic (concat, for example) are
    kept: a memory hotspot need not have parameters or arithmetic.
    """
    if include_fused_graph and phase != PREFILL:
        raise ValueError(
            f"a fused graph is the prefill view, not the {phase} walk")
    walk = served.analysis
    pre_program = served.pre_program
    program, rewrite = served.program, served.rewrite
    pre_cost = served.pre_cost
    copy_counts = served.pre_instances
    # The share of itself a node runs per token, as the walk states it
    # (`Analysis.active_share`): the copies that run over the copies held,
    # times the share of one copy a routed leaf runs.
    active_shares = walk.active_share

    # Parameter vectors/tensors are identity sources, not kernels: traffic
    # already bills their bytes to the consumer that reads them. Fold a
    # single-consumer source into that runtime row as well, so the table stays
    # kernel-shaped while its params still close. Which reader a source
    # belongs to is the walk's own join over its edges
    # (`Program.identity_sources_by_sole_reader`); a source with multiple
    # consumers stays its own row rather than being charged to one reader.
    absorbed_by_owner = pre_program.identity_sources_by_sole_reader
    absorbed_sources = {
        source_uid for sources in absorbed_by_owner.values() for source_uid in sources
    }
    env = _walk_env(walk)

    rows: List[Dict[str, Any]] = []
    total_params = 0
    total_active_params = 0
    total_flops = 0
    total_bytes = 0
    total_weight_reads = 0
    total_act_reads = 0
    total_act_writes = 0
    total_kv_reads = 0
    for timed in served.per_op_timing:
        uid = int(timed["uid"])
        if uid in absorbed_sources:
            continue
        node = program.by_uid(uid)
        members = rewrite.sources(uid) or (uid,)
        row_owns = {uid, *members}
        folded_sources = [
            source_uid
            for owner, source_uids in absorbed_by_owner.items()
            if owner in row_owns
            for source_uid in source_uids
        ]
        cost_members = (*members, *folded_sources)
        # Presentation costs are per instance; a fused row sums its
        # pre-fusion members (a grafted cast among them).
        n_params = 0
        n_active = 0
        flop_parts: List[tuple] = []
        param_parts: List[tuple] = []
        for member in cost_members:
            member_cost = pre_cost.values[member]
            count = int(copy_counts.values.get(member, 1))
            # A leaf's cost already holds the share of it a token runs (an
            # expert bank's `n_experts_active / n_route_targets` of its
            # expert axis); what multiplies it is the share of its copies
            # that run, which its enclosing `repeat:`s decide.
            parent = pre_program.by_uid(member).parent
            copies_active = (
                1 if parent is None
                else active_shares.values.get(parent.uid, 1))
            # Multiply before truncating: a routed member's active cost is a
            # fraction that is whole only over the bank.
            member_params = int(member_cost.params * count)
            member_flops = int(member_cost.flops * count)
            n_params += member_params
            n_active += int(member_cost.active_params * count * copies_active)
            member_name = pre_program.by_uid(member).name
            flop_sym, flop_val = _bound_cost(
                member_cost.flops,
                walk.flops_bindings.values.get(member),
                copies=count,
            )
            param_sym, param_val = _bound_cost(
                member_cost.params,
                walk.params_bindings.values.get(member),
                copies=count,
            )
            if member_flops > 0:
                flop_parts.append((member_name, flop_sym, flop_val, member_flops))
            if member_params > 0:
                param_parts.append((member_name, param_sym, param_val, member_params))

        # The step's numbers are the serving row's: FLOPs, the activation
        # read and write, the cache read, and the weights and their scales
        # the step reads (saturated at the rows it routes,
        # `row_roofline.weight_demand`).
        n_flops = int(timed["flops_per_rank"])
        moved = StepTraffic.of_row(timed)
        act_reads, act_writes = moved.act_reads, moved.act_writes
        kv_reads = moved.kv_reads
        weight_reads = step_weight_read([timed])
        n_bytes = act_reads + act_writes + weight_reads + kv_reads
        if n_params <= 0 and n_active <= 0 and n_flops <= 0 and n_bytes <= 0:
            continue

        origins = [
            pre_program.by_uid(source).name for source in rewrite.sources(uid)
        ]
        origins.extend(
            pre_program.by_uid(source_uid).name for source_uid in folded_sources
        )
        byte_parts = _access_style_byte_parts(
            served.access.values[uid],
            count=int(timed["count_per_step"]),
            env=env,
            act_reads=act_reads,
            act_writes=act_writes,
            weight_reads=weight_reads,
            kv_reads=kv_reads,
        )
        rows.append(
            {
                "node_id": f"{node.op}:{uid}",
                "uid": uid,
                "kernel": node.op,
                "label": node.name,
                "origins": origins,
                "params": n_params,
                "active_params": n_active,
                "flops": n_flops,
                "act_read_bytes": act_reads,
                "act_write_bytes": act_writes,
                "weight_read_bytes": weight_reads,
                "kv_read_bytes": kv_reads,
                "total_bytes": n_bytes,
                "arithmetic_intensity": (n_flops / n_bytes) if n_bytes else None,
                "formulas": {
                    "params": _parts_formula(param_parts, unit="", total=n_params),
                    "flops": _parts_formula(flop_parts, unit="FLOPs", total=n_flops),
                    "total_bytes": _parts_formula(
                        byte_parts, unit="B", total=n_bytes),
                    # The phase row's act / weight cells are these terms summed
                    # across kernels; each row carries its own split so those
                    # popovers drill to the same identities (`identity=` in
                    # `_phase_formulas`).
                    "act": _parts_formula(
                        [p for p in byte_parts
                         if p[0] in ("act_read", "act_write")],
                        unit="B", total=act_reads + act_writes),
                    "weight": _parts_formula(
                        [p for p in byte_parts if p[0] == "weights + scales"],
                        unit="B", total=weight_reads),
                    "kv": _parts_formula(
                        [p for p in byte_parts if p[0] == "kv_read"],
                        unit="B", total=kv_reads),
                },
            }
        )
        total_params += n_params
        total_active_params += n_active
        total_flops += n_flops
        total_bytes += n_bytes
        total_weight_reads += weight_reads
        total_act_reads += act_reads
        total_act_writes += act_writes
        total_kv_reads += kv_reads

    for row in rows:
        row["params_share"] = (row["params"] / total_params) if total_params else 0.0
        row["flops_share"] = (row["flops"] / total_flops) if total_flops else 0.0
        row["bytes_share"] = (row["total_bytes"] / total_bytes) if total_bytes else 0.0
    rows.sort(key=lambda r: (-r["flops"], -r["total_bytes"], -r["params"]))
    table = {
        "contract": {
            "phase": phase,
            "batch": batch,
            "input_tokens": input_tokens,
            "fusion": "shipped",
            "parallel": "single_rank_unsharded",
            "activation_dtype": display_name(
                walk.resolved_precision.model_dtype),
            "precision": precision.to_json() if precision is not None
            else None,
        },
        "rows": rows,
        "totals": {
            "params": total_params,
            "active_params": total_active_params,
            "flops": total_flops,
            "act_read_bytes": total_act_reads,
            "act_write_bytes": total_act_writes,
            "weight_read_bytes": total_weight_reads,
            "kv_read_bytes": total_kv_reads,
            "total_bytes": total_bytes,
            "arithmetic_intensity": (total_flops / total_bytes)
            if total_bytes
            else None,
        },
    }
    if include_fused_graph:
        table["fused_graph"], table["fused_l1"] = _fused_graph(served, table)
    return table


def _drawn_as(host: Any, name: str) -> str:
    """The component whose graph stands for `name` in the diagram: `name`
    itself, or — when `name` is only a new name for another component
    (`BlockGraph.renamed_component`; `chatglm_attention` →
    `gqa_fused_qkv_attention`) — the component it renames. The rename adds
    no op of its own, only a level to open before anything shows; the
    structure view is then read off the target's nodes and still has to add
    up to the wrapper's own totals."""
    components = host.spec.components or {}
    graph = components.get(name)
    target = graph.renamed_component(components.get) if graph is not None else None
    return name if target is None else _drawn_as(host, target)


def phase_contract(batch: int, input_tokens: int) -> Dict[str, Dict[str, int]]:
    """What each phase binds for the `(batch, input_tokens)` contract —
    `s_query` new tokens per request against `s_kv` positions — read off
    `Phase.resolve`, the binding the walk itself uses, so a subtitle can say
    what was evaluated without restating the phase rules."""
    workload = Workload(batch=batch, input_tokens=input_tokens)
    out: Dict[str, Dict[str, int]] = {}
    for kind in (PREFILL, DECODE):
        bound = Phase(kind=kind).resolve(workload)
        out[kind] = {
            "batch": batch,
            "s_query": int(bound["s_query"]),
            "s_kv": int(bound["s_kv"]),
        }
    return out


def _tokens(n: int, adjective: str = "") -> str:
    """`2,048 tokens`, `1 new token`."""
    noun = "token" if n == 1 else "tokens"
    return f"{n:,} {adjective + ' ' if adjective else ''}{noun}"


def phase_units(
    phase: str, at: Dict[str, int], scope: str = "per layer"
) -> Dict[str, str]:
    """Unit strings for one phase's numbers, with the values the walk ran at
    written in (never a bare `cache_len` / `(batch, input_tokens)` symbol).
    `scope` is `"per layer"` for a bench row and `"whole model"` for the
    editor; every other phrase (batch, token counts, cache_len) is the same."""
    batch, sq, skv = at["batch"], at["s_query"], at["s_kv"]
    if phase == DECODE:
        step = (
            f"one decode step @ batch {batch} × {_tokens(sq, 'new')}, "
            f"cache_len = {_tokens(skv)}"
        )
    else:
        step = f"one forward @ batch {batch} × {_tokens(sq)}"
    return {
        "flops": f"{scope} · {step}",
        "act_bytes": f"{scope} · {step}",

        # What the step streams of the stored weights: all of a dense
        # weight, the routed share of an expert bank's — each weight's
        # scales read exactly as far as the weight is.
        "weight_read": (
            f"{scope} · {step} · weights + scales "
            f"(routed experts: as far as tokens reach)"
        ),
        # weight_read_bytes vs kv_step_bytes: the step's two memory streams,
        # the KV side summed over every request in the batch.
        "traffic": f"{scope} · {step} · KV × batch",
    }


def _phase_row(
    table: Dict[str, Any],
    naive: StepTraffic,
    *,
    kv_bytes: int,
    phase: str,
    at: Dict[str, int],
    runs_as: str,
    t_sol_formula: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One phase's numbers for a bench row. `kv_bytes` is what the phase does
    to ONE request's cache: prefill writes (stores) it, a decode step reads it;
    internally `kv_step = kv_bytes × batch` is the step-wide figure that sits
    beside `weight_read_bytes` as the other memory stream. Prefill reports
    that as `kv_step_bytes` (KV write / step); decode reports
    `kv_read_bytes` (already the whole batch).

    FLOPs, weight and KV reads and the fused activation bytes are the
    runtime table's totals; the naive activation bytes are the unfused walk
    beside it."""
    kv_step = int(kv_bytes) * int(at["batch"])
    totals = table["totals"]
    traffic = {
        "act_read_bytes": naive.act_reads,
        "act_write_bytes": naive.act_writes,
        "act_read_bytes_fused": totals["act_read_bytes"],
        "act_write_bytes_fused": totals["act_write_bytes"],
        "kv_read_bytes": totals["kv_read_bytes"],
        "weight_read_bytes": totals["weight_read_bytes"],
    }
    fused = traffic["act_read_bytes_fused"] + traffic["act_write_bytes_fused"]
    row = {
        "flops": totals["flops"],
        **traffic,
        "act_bytes_naive": naive.act_bytes,
        "act_bytes_fused": fused,
        # Which of the step's two memory streams is larger.
        "memory_dominant": "kv"
        if kv_step > traffic["weight_read_bytes"]
        else "weights",
        # The component actually evaluated: the declared decode form at decode.
        "runs_as": runs_as,
        # A sentinel. A fusion only removes intermediates from HBM, so fused
        # activation bytes above naive cannot be right: the flag should never
        # fire, and a row that raises it is an accounting bug to report, not a
        # finding. Carried on the row so the UI marks the value instead of
        # plotting it as fact.
        "fused_exceeds_naive": fused > naive.act_bytes,
        "units": phase_units(phase, at),
    }
    if phase == PREFILL:
        row["kv_step_bytes"] = kv_step
    row["formulas"] = _phase_formulas(
        row, phase=phase, at=at, kv_bytes=kv_bytes, op_rows=table["rows"],
    )
    if t_sol_formula:
        row["formulas"]["t_sol"] = t_sol_formula
    return row


def _star(text: str) -> str:
    return (text or "").replace("·", "*")


def _whole(n: Any) -> Any:
    """A whole number as int when the value is integral — so a bound
    formula compared against a traffic Dim's `12582912.0` still matches."""
    if n is None or isinstance(n, bool):
        return n
    try:
        if float(n) == int(n):
            return int(n)
    except (TypeError, ValueError):
        return n
    return n


def _walk_env(analysis: Any) -> Dict[str, Any]:
    """The walk's dim env as name → number, for binding a traffic / KV
    identity the way Inference binds a FLOPs record."""
    binding = analysis.binding
    if binding is None:
        # A walk with no workload binding has no numeric env.
        return {}
    out: Dict[str, Any] = {}
    for name, val in binding.flat().items():
        v = value_of(val)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            out[str(name)] = _whole(v)
    return out


def _clean_expr(extent: Any) -> Any:
    """Identity of an extent with integral dtype widths as ints (`2.0` BF16)."""
    expr = expr_of(extent)
    if not isinstance(expr, sp.Expr):
        return expr
    integral = {
        f: sp.Integer(int(f)) for f in expr.atoms(sp.Float)
        if float(f).is_integer()
    }
    return expr.xreplace(integral) if integral else expr


def _bind_extent(
    extent: Any,
    env: Optional[Dict[str, Any]] = None,
    *,
    record: Optional[Dict[Any, Any]] = None,
    expr: Any = None,
    expect: Any = None,
    scale: Any = 1,
) -> str:
    """Values line of an extent: the symbolic identity with dims bound,
    products kept. `bind_record` of the evaluation when a record is
    given, else the walk env, else the folded number. Scale (copies)
    stays outside the bind.
    """
    identity = _clean_expr(extent) if expr is None else _clean_expr(expr)
    per = _whole(value_of(extent) if expect is None else expect)
    text = None
    if isinstance(identity, sp.Expr) and identity.free_symbols:
        if record:
            text = bind_record(identity, record, per)
        if text is None and env:
            text = bound_formula_text(identity, env, per)
    text = _star(str(text)) if text else ("" if per is None else str(per))
    if scale != 1 and text:
        scaled = f"{scale:g}" if isinstance(scale, float) else str(scale)
        text = f"({text}) * {scaled}"
    return text


def _bound_cost(
    extent: Any, record: Optional[Dict[Any, Any]], *, copies: int = 1,
) -> Tuple[str, str]:
    """Inspector values-line for one fused member: identity, then
    `bound_or_folded` against the walk's substitution record.

    Copies (an expert bank, a repeated block) stay outside the bind so
    the product tree is the per-copy figure the record priced.
    """
    sym = _star(str(expr_of(extent) or ""))
    val = _star(bound_or_folded(extent, record))
    if copies != 1:
        if sym:
            sym = f"({sym}) * {copies}"
        if val:
            val = f"({val}) * {copies}"
    return sym, val


def _access_style_byte_parts(
    acc: Any, *,
    count: int,
    env: Dict[str, Any],
    act_reads: int,
    act_writes: int,
    weight_reads: int,
    kv_reads: int,
) -> List[tuple]:
    """A row's byte terms off its serving access record (`LeafAccess`), the
    terms the inference per-op popover prints (`access_display_facts`): each
    activation edge at the width `traffic` priced it at
    (`phase_formulas.act_in_from_terms` / `act_out_from_terms`) — a GEMM
    that reads a grafted cast's FP8 copy reads it at one byte — the
    conversion a kernel still makes of an operand no cast could be grafted
    for, the weights as `params*dt_w` [+ scales], the cache read as its
    extent. Each term is one launch's; a row run `count` times a step says
    so, and a weight read the step's rows saturate is the row's own read
    (`row_roofline.weight_demand`), not the stored weights."""
    facts = access_display_facts(acc)

    def times(text: str) -> str:
        return text if count == 1 or not text else f"({text}) * {count}"

    parts: List[tuple] = []
    if act_reads > 0:
        # The cache read is its own term below.
        sym, val = act_in_from_terms(
            {**facts, "kv_bytes": 0}, acc.shape_in_vals, acc.shape_out_vals)
        if facts["convert_bytes"]:
            convert = int(facts["convert_bytes"])
            sym = f"{sym} + convert" if sym else "convert"
            val = f"{val} + {convert}" if val else str(convert)
        parts.append(("act_read", times(sym), times(val), act_reads))
    if act_writes > 0:
        sym, val = act_out_from_terms(facts, acc.shape_out_vals)
        parts.append(("act_write", times(sym), times(val), act_writes))
    if weight_reads > 0:
        stored = int(acc.weight_bytes_per_rank + acc.metadata_bytes_per_rank)
        if weight_reads != stored * count:
            w_sym, w_val = "min(rows*w_row, weights)", f"{weight_reads}"
        elif facts["params_sym"]:
            w = params_bytes_formula(
                acc.display.cost, acc.display.bindings,
                dt_w=float(acc.weight_dtype_bytes),
                meta_bytes=acc.metadata_bytes_per_rank)
            w_sym, w_val = times(w["symbolic"]), times(w["substituted"])
        else:
            w_sym, w_val = "weights", times(f"{stored}")
        parts.append(("weights + scales", w_sym, w_val, weight_reads))
    if kv_reads > 0:
        parts.append((
            "kv_read",
            _extent_identity(acc.kv_read_extent_per_rank, count),
            _bind_extent(acc.kv_read_extent_per_rank, env, scale=count),
            kv_reads,
        ))
    return parts


def _extent_identity(extent: Any, scale: Any = 1) -> str:
    """A byte term's raw identity: the traffic Dim's expression × the copies
    the row runs at — `2*B*d_q_lora*s_ctx` for an activation edge, the weight's
    `d_q_lora*n_heads*(d_qk_nope + d_qk_rope) + 9216` for a stored tensor.

    The traffic derivation never lost these identities (`Extent` IS a `Dim`,
    expression and value together); the op table's `int()` reads were what
    threw them away. Integral floats are dtype widths (`2.0` BF16), rendered
    as ints. A term with no identity — a bare number nobody named — returns ""
    and the caller falls back to the term's label, the previous behavior.
    """
    expr = _clean_expr(extent)
    if not isinstance(expr, sp.Expr) or expr.is_number:
        return ""
    sym = _star(str(expr))
    if scale != 1:
        scaled = f"{scale:g}" if isinstance(scale, float) else str(scale)
        sym = f"({sym}) * {scaled}"
    return sym


def _parts_formula(
    parts: List[tuple], *, unit: str, total: int,
) -> Dict[str, Any]:
    """`(name, symbolic, values, val)` addends of `_addends_formula`."""
    terms: List[Dict[str, Any]] = []
    for name, sym, values, val in parts:
        term: Dict[str, Any] = {"label": name, "result": val}
        if sym:
            term["symbolic"] = sym
        if values:
            term["substituted"] = values
        terms.append(term)
    return _addends_formula(terms, total=int(total), unit=unit)


def _phase_formulas(
    row: Dict[str, Any],
    *,
    phase: str,
    at: Dict[str, int],
    kv_bytes: int,
    op_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Formula popovers for a phase row's summary cells.

    Display, not the math: every substituted number is read back off the row
    the cell shows, so the popover can never drift from the table. `*` for
    multiplication, never `·`. The addends are the phase's runtime kernels —
    the walk's leaves at their copy counts, which the Details table closes
    against the cell (`_runtime_op_table`) — each with its own identity.
    """
    kernels = list(op_rows or [])
    formulas: Dict[str, Any] = {
        "flops": _rows_sum_formula(
            kernels,
            value_fn=lambda r: r["flops"],
            unit="FLOPs",
            total=int(row["flops"]),
            identity="flops",
        ),
        "weight_read": _rows_sum_formula(
            kernels,
            value_fn=lambda r: r.get("weight_read_bytes"),
            unit="B",
            total=int(row["weight_read_bytes"]),
            identity="weight",
            note=(
                "a dense weight is read whole; a routed expert bank streams "
                "only the share the step's tokens reach, its scales in the "
                "same proportion"
            ),
        ),
        "act_bytes": _rows_sum_formula(
            kernels,
            value_fn=lambda r: (
                int(r.get("act_read_bytes") or 0)
                + int(r.get("act_write_bytes") or 0)
            ),
            unit="B",
            total=int(row["act_bytes_fused"]),
            identity="act",
        ),
    }
    if phase == PREFILL:
        formulas["kv_step"] = {
            "symbolic": "kv_bytes/request * batch",
            "substituted": (
                f"{int(kv_bytes):,d} * {int(at['batch'])} = "
                f"{row['kv_step_bytes']:,d} B"
            ),
        }
    else:
        formulas["kv_step"] = _rows_sum_formula(
            kernels,
            value_fn=lambda r: r.get("kv_read_bytes"),
            unit="B",
            total=int(row["kv_read_bytes"]),
            identity="kv",
        )
    return formulas


def _bench_formulas(
    table: Dict[str, Any],
    storage: Dict[str, Any],
    weight_parts: List[Dict[str, Any]],
    kv: Dict[str, Any],
) -> Dict[str, Any]:
    """Row-level formula popovers for the summary's stored columns (params,
    active params, weight storage, KV / request).

    Each popover lists the parts its cell is the sum of: params are the
    prefill runtime table's rows (the cells are its `totals`), weight storage
    the ledger's split by node (`_weight_storage`), KV / request the cache
    walk's per-tensor terms.
    """
    totals = table["totals"]
    params, active = totals["params"], totals["active_params"]
    active_formula = _rows_sum_formula(
        table["rows"], value_fn=lambda r: r["active_params"], unit="",
        total=active)
    if active_formula is not None and active != params:
        active_formula["note"] = _join_notes(
            active_formula.get("note") or "",
            f"{params - active:,d} stored params idle per token — a routed "
            "expert bank runs only the experts the token dispatches to",
        )
    weight_formula = _addends_formula(
        weight_parts, total=int(storage["bytes"]), unit="B")
    if weight_formula is not None:
        weight_formula["note"] = _join_notes(
            storage["precision"],
            (
                f"parameters {storage['parameter_bytes']:,d} + "
                f"metadata {storage['metadata_bytes']:,d} + "
                f"buffers {storage['buffer_bytes']:,d}"
            ),
            weight_formula.get("note") or "",
        )
    return {
        "params": _rows_sum_formula(
            table["rows"], value_fn=lambda r: r["params"], unit="",
            total=params, identity="params"),
        "active_params": active_formula,
        "weight_bytes": weight_formula,
        "kv_storage": kv["formulas"]["storage"],
    }


def bench_component(
    base_resolved: Any,
    name: str,
    *,
    batch: int,
    input_tokens: int,
    dim_overrides: Dict[str, Any],
    loader: Any,
    precision: Any,
    kv_cache_lens: Tuple[int, ...] = (),
    site: Optional[CallSite] = None,
    bound: bool = False,
    gpu_sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Canonical cost of one isolated component under the contract.

    `site` is the call it is evaluated at, in the model the weights come
    from (`CallSite.of`); None takes the base model's only call of it.
    `bound` says the site is a bound entry's model rather than the base.
    A vision or audio tower's call is refused (`_compared_calls`)."""
    host = inject_component(base_resolved, name, loader)
    if site is None:
        site = CallSite.of(host, name)
    if site.encoder is not None:
        raise ValueError(
            f"{site.described()} is a call of the {site.encoder} encoder: "
            f"the playground compares the language model's calls only")
    report = project_block(
        host,
        name,
        batch=batch,
        input_tokens=input_tokens,
        dim_overrides=dim_overrides,
        site=site,
        precision=precision,
    )
    isolated = analyze_isolated(
host, name, dim_overrides, site,
                                precision=precision)
    kv, decode_kv = _kv_view(
        host,
        name,
        isolated,
        cache_len=input_tokens,
        precision=precision,
        dim_overrides=dim_overrides,
        site=site,
        sweep_lens=tuple(kv_cache_lens),
    )
    storage, weight_parts = _weight_storage(isolated, site, bound=bound)
    at = phase_contract(batch, input_tokens)
    # A card with no peak for the compute dtype times nothing; the row
    # keeps its FLOPs and bytes and says why it has no time.
    # Each phase is one serving step of the isolated component: the walk,
    # its collectives, its activation casts and the shipped fusions, in the
    # serving graph's order (`serving_step`). Its rows after the rewrite are
    # the runtime table, its rows before it the naive view, and its walk the
    # one the timing reads (`cells`).
    spec = isolate_resolved(host, name, dim_overrides, site)
    cells = StepCells(RuntimeProfile.ARCHITECTURE)
    served = {
        phase: serving_step(
            spec, batch=batch, input_tokens=input_tokens, phase=phase,
            include_aux_timing=True, precision=precision, cells=cells)
        for phase in (PREFILL, DECODE)
    }
    no_time: Optional[str] = None
    try:
        timings = {
            phase: _timing_view(
                spec,
                name,
                batch=batch,
                input_tokens=input_tokens,
                site=site,
                precision=precision,
                phase=phase,
                gpu_sku=gpu_sku,
                cells=cells,
            )
            for phase in (PREFILL, DECODE)
        } if gpu_sku else {}
    except NoPeakForDtype as exc:
        timings, no_time = {}, str(exc)
    # t SOL's popover belongs to the phase row, not the timing dict.
    t_sol_formulas = {
        phase: timing.pop("t_sol_formula")
        for phase, timing in timings.items()
        if timing.get("t_sol_formula")
    }
    graph = _block_graph(
        host,
        _drawn_as(host, name),
        batch=batch,
        input_tokens=input_tokens,
        dim_overrides=dim_overrides,
        precision=precision,
        site=site,
    )
    # The phase's runtime kernels ARE its numbers: every cell of a phase row
    # (and params / active params) is a total of these rows, and its popover
    # lists them, so a cell and its explanation cannot disagree.
    op_tables = {
        phase: _runtime_op_table(
            served[phase], batch=batch, input_tokens=input_tokens,
            precision=precision, phase=phase,
            include_fused_graph=phase == PREFILL)
        for phase in (PREFILL, DECODE)
    }
    fused_graph = op_tables[PREFILL].pop("fused_graph")
    fused_l1 = op_tables[PREFILL].pop("fused_l1")
    totals = op_tables[PREFILL]["totals"]
    # Inner L1: builder does not import views (cycle). Stamp once here.
    # Root L1 stays on the entry, not graph["l1"].
    from backend.views.abstraction import has_anchor
    from backend.views.graph_view import l1_of_nodes, stamp_l1
    stamp_l1(graph["nodes"])
    l1 = None
    if has_anchor(graph["nodes"]):
        # The declared structure closes to the declared walk: the runtime
        # table's FLOPs also count the activation casts serving grafts,
        # which are launches and not nodes of the component.
        l1 = l1_of_nodes(
            graph["nodes"],
            Sums(
                flops=int(report.total_flops),
                params=totals["params"],
                weight_bytes=storage["bytes"],
            ),
        )
        l1["weight_semantics"] = "stored"
    walk = _isolated_walk(host, name, dim_overrides, site)
    depends = _dependency_dims(host, name, walk)
    # The values the displayed formulas substitute: what the evaluation
    # depends on, plus what the L1 box formulas bind beyond it (the
    # placement symbols `M` / `dp_attention` the walk resolves on
    # its own). Everything else the host's nodes resolved (`n_vocab`,
    # `n_layers`, another block's dims) is not this component's.
    shown = depends | _formula_names(
        box["flops_symbolic"] for box in (l1 or {}).get("boxes") or [])
    local = _dim_values_for(walk, shown)
    dims_used = {**local, **{k: v for k, v in report.dims.items()
                              if k in shown}}
    return {
        "component": name,
        "role": host.spec.component_roles.get(name),
        "params": totals["params"],
        "active_params": totals["active_params"],
        # One stored copy of the block — parameters, their scales and
        # buffers at the checkpoint's formats; the same in both phases.
        "weight_bytes": storage["bytes"],
        "weight_storage": storage,
        "units": {
            "weight_bytes": "per layer (one block copy) · stored once",
        },
        "dims_used": dims_used,
        # The component's OWN editable knobs (allowlist), for the editable dims
        # table + seeding — never the host's ambient/contract dims.
        "editable_dims": _component_own_dims(walk, depends),
        # Which shipped models bind this component — the module's provenance, so
        # the comparison shows it is not an anonymous library abstraction.
        "model_ids": _component_to_models().get(name, []),
        "kv": kv,
        # Summary-cell popovers: the composition of every stored number from
        # the engine's primitive terms, substituted with the row's own value.
        "formulas": _bench_formulas(
            op_tables[PREFILL], storage, weight_parts, kv),
        # Prefill and decode side by side. Decode is one new token per request
        # against a cache of input_tokens.
        "phases": {
            PREFILL: _phase_row(
                op_tables[PREFILL],
                StepTraffic.of_rows(served[PREFILL].pre_per_op_timing),
                kv_bytes=kv["storage_bytes_at_L"],
                phase=PREFILL,
                at=at[PREFILL],
                runs_as=phase_form(host.spec.get_block(name), PREFILL) or name,
                t_sol_formula=t_sol_formulas.get(PREFILL),
            ),
            DECODE: _phase_row(
                op_tables[DECODE],
                StepTraffic.of_rows(served[DECODE].pre_per_op_timing),
                kv_bytes=decode_kv,
                phase=DECODE,
                at=at[DECODE],
                runs_as=phase_form(host.spec.get_block(name), DECODE) or name,
                t_sol_formula=t_sol_formulas.get(DECODE),
            ),
        },
        **({"timings": timings} if timings else {}),
        **({"timings_unavailable": no_time} if no_time else {}),
        "block_graph": graph,
        "fused_graph": fused_graph,
        "fused_l1": fused_l1,
        **({"l1": l1} if l1 is not None else {}),
        # The details table's one runtime view: live post-fusion kernels,
        # single-rank, with FLOPs / HBM access / arithmetic intensity side by
        # side. Logical ops folded by a kernel remain visible as origins.
        "op_tables": op_tables,
    }


def list_library_components() -> List[Dict[str, str]]:
    """The swappable library components, each with its role and family, for the
    picker.

    Only components that declare a `role` are top-level swappable slots (a
    `role: None` component is an internal sub-part). `attn_display` is the
    component's canonical display category (`attn_variant` if set — DSA / CSA /
    HCA / KDA — else the KV-cache `kv_label` — MLA / GQA / SWA / GatedDeltaNet /
    …), so the UI groups by a declared category, not a name prefix. `None` when
    the component declares neither (grouped as "Other" by the caller). Named
    `attn_display` (not `family`) so it does not read as `cache.family`. Source
    of truth is the loader's component library — no hand-maintained frontend list.
    A component the models call only in a vision / audio tower is not listed
    (`_encoder_only_components`).
    """
    loader = get_default_loader()
    encoder_only = _encoder_only_components()
    out: List[Dict[str, str]] = []
    for name in sorted(loader.component_names()):
        block = loader.component(name)
        role = block.role
        if role and name not in encoder_only:
            out.append(
                {
                    "name": name,
                    "role": role,
                    "attn_display": block.attn_display,
                    # Declared decode-time absorbed form (e.g. mla_dsa_attention →
                    # mla_dsa_attention_absorbed) — the UI labels that relationship
                    # instead of letting the numbers puzzle the reader.
                    "decode_variant": block.decode_variant,
                    # The component's own spec-authored description — the UI quotes
                    # it (e.g. for why a decode form's FLOPs are higher) instead of
                    # inventing explanations.
                    "description": block.description,
                    # Representative models that bind this component (provenance).
                    "model_ids": _component_to_models().get(name, []),
                }
            )
    return out


def _match_model(model_id: str, loader: Any) -> Any:
    """The spec `loader` resolves for shipped model `model_id` (config and
    checkpoint header fetched for it); ValueError when there is none."""
    overlay_dir = loader.overlay_for(model_id)
    config = fetch_config(model_id, overlay_dir)
    if config is None:
        raise ValueError(f"could not fetch config for {model_id!r}")
    meta = fetch_metadata(model_id, overlay_dir) or {}
    resolved = loader.match(
        config, model_id=model_id, header=meta.get("header")
    )
    if not resolved:
        raise ValueError(f"no spec matches {model_id!r}")
    return resolved


def component_dims_from_model(component: str, model_id: str,
                              call: Optional[str] = None) -> Dict[str, int]:
    """The internal dims `component` uses AS shipped model `model_id`
    parameterizes it at its call `call` (`CallSite.of`: the only call when
    None) — for seeding the comparison's editable per-component dims from a
    real model instead of the arbitrary base."""
    loader = get_default_loader()
    resolved = _match_model(model_id, loader)
    return _dims_at(resolved, component, loader,
                    _call_site(resolved, component, call, model_id, loader))


def _call_site(resolved: Any, component: str, call: Optional[str],
               model_id: str, loader: Any) -> CallSite:
    """`CallSite.of` for shipped model `model_id`, where naming no call also
    takes a model whose calls of `component` all evaluate alike
    (`component_bindings`' `model_parameterizations`): any of them is the
    one, so the first. Calls that differ are still refused until one is
    named."""
    if call is None:
        sites = [site for site in _compared_calls(resolved)
                 if binding_of(resolved.spec, site.component, component)]
        if len(sites) > 1:
            rows = _model_bindings(resolved, model_id, component, loader,
                                   sites)
            if rows[0]["model_parameterizations"] != 1:
                raise ValueError(
                    f"{model_id} uses {component} in {len(sites)} places "
                    f"that are not alike ("
                    + "; ".join(site.label for site in sites)
                    + "). Pick the one this column compares with "
                    f"\"evaluate at which call…\"")
        if sites:
            call = sites[0].call
    return CallSite.of(resolved, component, call=call, model_id=model_id)


def _compared_calls(resolved: Any) -> List[CallSite]:
    """The calls of `resolved` the playground compares: the language
    model's (`CallSite.calls`), not a vision or audio tower's — an encoder
    call is sized by its media, which a module compare does not give
    (`_encoder_only_components`)."""
    return [site for site in CallSite.calls(resolved) if site.encoder is None]


def _dims_at(resolved: Any, component: str, loader: Any,
             site: CallSite) -> Dict[str, int]:
    """The component's own editable dims evaluated at `site` — the bench's
    `editable_dims` reading. The call's frame carries what its layer variant
    and its callers rebind (`CallSite.frame`), so a dim the model sets per
    layer (compress_ratio on V4's CSA layers) is the one that call runs at."""
    host = inject_component(resolved, component, loader)
    walk = _isolated_walk(host, component, {}, site.where_made)
    return _component_own_dims(walk, _dependency_dims(host, component, walk))


def component_calls(resolved: Any, component: str,
                    loader: Any) -> List[Dict[str, Any]]:
    """The calls of `resolved` an entry of `component` can name: the
    component's own calls, or — when the model does not call it — the calls
    of a component in the role it declares, whose place it would take (not
    for a component with slots only its caller fills). A
    block the model runs as a stage lists every other stage of that stack
    after its own calls, each naming the block it runs, so the entry can
    move to another of the model's blocks. What each hands over is shown, so the user can tell the
    calls apart."""
    calls = _compared_calls(resolved)
    own = [site for site in calls
           if binding_of(resolved.spec, site.component, component)]
    # A component whose slots only its caller fills takes no other
    # component's place: that call fills none of them
    # (`BlockGraph.open_slots`, refused by `isolate_resolved`).
    body = inject_component(resolved, component, loader).spec.get_block(
        component)
    if body.open_slots:
        return [call_dto(site) for site in own]
    stacks = {site.stack for site in own if site.is_stage}
    own += [site for site in calls if site.is_stage and site.stack in stacks
            and not binding_of(resolved.spec, site.component, component)]
    if not own:
        role = loader.component(component).role
        own = [site for site in calls if role
               and resolved.spec.get_block(site.component).role == role]
    return [call_dto(site) for site in own]


def call_dto(site: CallSite) -> Dict[str, Any]:
    """A call as the page lists it: its address, where it is in words, the
    component called there and what the call hands over in the phase that
    makes it."""
    return {"call": site.call, "label": site.label,
            "component": site.component,
            "formals": [[str(axis) for axis in formal]
                        for formal in site.handed_where_made]}


_bindings_cache: Dict[str, Dict[str, Any]] = {}


def component_bindings(component: str) -> Dict[str, Any]:
    """Every call of `component` in every shipped model that binds it, one
    row per call: the dims the call resolves to (the single-pick
    parameterization table / `+ compare` source) — a model that calls it in
    several places parameterizes it per call — and how many calls that model
    makes of it (`model_calls`) — and how many of those evaluate differently
    (`model_parameterizations`), so the bench (`_call_site`) and a client
    take a call on their own only when the model's calls all evaluate alike:
    its only call, or one of several that differ in nothing the evaluation
    reads (Kimi-K3's KDA attention, called at 16 layer groups that rebind
    only the AttnRes history `Sar`, which attention never reads; DeepSeek's
    MLA, called in the backbone and in the MTP layer). A call that cannot be evaluated alone
    is still a row, with its reason (`error`) and no dims. A model that does
    not resolve is reported in `failed`, never silently dropped, so an empty
    table reads as "library-only" only when no model binds the component at
    all. Cached per process (model yamls, snapshots and configs are static
    within one run)."""
    if component in _bindings_cache:
        return _bindings_cache[component]
    loader = get_default_loader()
    out: List[Dict[str, Any]] = []
    failed: Dict[str, str] = {}
    for mid in _component_to_models().get(component, []):
        try:
            resolved = _match_model(mid, loader)
            sites = [site for site in _compared_calls(resolved)
                     if binding_of(resolved.spec, site.component, component)]
        except CONTRACT_ERRORS as exc:
            logger.info("binding dims %s from %s failed: %s", component, mid, exc)
            failed[mid] = str(exc)
            continue
        if not sites:
            failed[mid] = f"{mid} does not call {component}"
            continue
        out.extend(_model_bindings(resolved, mid, component, loader, sites))
    result = {"bindings": out, "failed": failed}
    _bindings_cache[component] = result
    return result


def _model_bindings(resolved: Any, mid: str, component: str, loader: Any,
                    sites: List[CallSite]) -> List[Dict[str, Any]]:
    """`component_bindings`' rows of one model: one per call in `sites`
    (the model's calls of `component`), with the dims each resolves to and
    how many of them evaluate differently (`model_parameterizations`)."""
    rows: List[Dict[str, Any]] = []
    stored: List[Any] = []
    seen: List[Dict[str, Any]] = []
    read: set = set()
    for site in sites:
        row = {"model_id": mid, **call_dto(site), "model_calls": len(sites)}
        try:
            host = inject_component(resolved, component, loader)
            walk = _isolated_walk(host, component, {}, site.where_made)
            depends = _dependency_dims(host, component, walk)
            row["dims"] = _component_own_dims(walk, depends)
            read |= depends
            stored.append(_stored_formats(walk))
            seen.append(_dims_seen(walk))
        except CONTRACT_ERRORS as exc:
            logger.info("binding dims %s from %s at %s failed: %s",
                        component, mid, site.call, exc)
            row.update(dims={}, error=str(exc))
            stored.append(None)
            seen.append({})
        rows.append(row)
    # Two calls evaluate alike when the isolation reads the same facts at
    # both (`isolate_resolved`): the formats its weights are stored in
    # (`_stored_formats` — not the checkpoint address they sit at, which
    # names the copy and prices nothing: a backbone layer and an MTP layer
    # stored alike are one parameterization, stored at different dtypes
    # two), its role, the
    # binding they run (`binding_key`: the root template and its full slot
    # bindings — an inline `params:` call and the per-layer map's
    # synthesized name of the same bindings are one parameterization),
    # the streams the stack declares for it (not the stack's name: an MTP
    # layer's attention runs as the backbone's), what is handed over, and
    # the values the component sees (`_dims_seen`: its frame and the call's
    # own `dims:`, by value, not as spelled — an MTP call's `d_head:
    # d_qk_nope` and a backbone frame's `d_head = 128` are one value, and
    # `d_ffn: d_ffn` rebinds nothing), of the dims some call's evaluation
    # reads and of those the call's own `dims:` set. A call that does not
    # run alone is compared on its whole frame and `dims:` as written.
    keys = [
        (formats, site.role,
         binding_key(resolved.spec, site.component, dict(site.params)),
         site.handed,
         tuple(sorted((stream, repr(decl)) for stream, decl
                      in site.stream_reads(resolved).items())),
         "error" in row,
         ((site.binds, site.frame) if "error" in row
          else tuple((name, repr(values.get(name))) for name in sorted(
              read | {name for name, _ in site.binds}))))
        for site, row, formats, values
        in zip(sites, rows, stored, seen, strict=True)]
    # Which of the model's parameterizations each call is (numbered in call
    # order), so a client shows one row per parameterization, not per call.
    alike = {key: n for n, key in enumerate(dict.fromkeys(keys))}
    for row, key in zip(rows, keys, strict=True):
        row["model_parameterizations"] = len(alike)
        row["parameterization"] = alike[key]
    return rows


def _dependency_dims(host: Any, name: str, walk: Any) -> set:
    """Every model dim the evaluation of `name` — a component or a block
    template — depends on, contract dims (`d_model`) included.

    "Depends on" is the names the isolated walk's extents (`walk`, from
    `_isolated_walk`) are expressions
    in (`walk_dim_names`) — the same evaluation — so the router's and
    permute's own reads (`n_experts_active`, `n_route_targets`) count even
    though no node of the component spells them, and the host's ambient
    dims (`n_vocab`, `n_layers`, another block's `n_experts`) do not. A
    name bound to 0 or 1 reads as that number (`frame_names`), so a flag
    the closure spells out (`n_router_bias_params`, a top-1
    `n_experts_active`, gemma-4's `n_kv_heads = 1` on its full layers) is
    added from the static scan of the closure."""
    return closure_dim_refs(host, name) | walk_dim_names(walk)


def _isolated_walk(host: Any, name: str, dim_overrides: Dict[str, Any],
                   site: CallSite) -> Any:
    """The walk of `name` isolated at `site` (`isolate_resolved`), one
    request of two tokens, that `_dependency_dims` reads its dims from."""
    isolated = isolate_resolved(host, name, dim_overrides, site)
    return Deployment(Program.of(isolated), runtime_profile=RuntimeProfile.ARCHITECTURE).at(
        batch=1, input_tokens=2).analyze()


def _dims_seen(walk: Any) -> Dict[str, Any]:
    """The values the component sees in its isolated walk (`walk`, from
    `_isolated_walk`): the frame the walk bound at the call node — the
    model's dims under the call's frame, with the call's own `dims:` over
    them (`Analysis.node_dims`)."""
    (call,) = walk.program.blocks[0].roots
    return {name: value_of(value)
            for name, value in walk.node_dims.values[call.uid].items()}


def _stored_formats(walk: Any) -> Tuple[Any, ...]:
    """How `walk` stores each of its tensors (`Analysis.tensors()`): the
    format's method, dtypes and bytes, and its scale siblings by suffix,
    dtype and shape — everything pricing reads of it, without the module
    path, which names the copy rather than the format."""
    return tuple(
        (row.uid, row.tensor_index, row.segment_index, row.method,
         row.storage_dtype, row.logical_dtype, value_of(row.storage_bytes),
         tuple((meta.suffix, meta.dtype, tuple(meta.shape))
               for meta in row.metadata))
        for row in walk.tensors())


def walk_dim_names(walk: Any, uids: Optional[set] = None) -> set:
    """Dims a walk's costs, stored tensors and shapes are expressions in —
    of the nodes in `uids`, or of every node: the names of each extent's
    identity (`Dim.names`), in the model's names (`frame_names`). A cost is
    read at the leaf that has it; a component adds its own copies."""
    names: Dict[int, set] = {}

    def add(uid: int, *extents: Any) -> None:
        if uids is None or uid in uids:
            for extent in extents:
                names.setdefault(uid, set()).update(names_of(extent))

    for uid, cost in walk.cost.values.items():
        if walk.program.by_uid(uid).children:
            # A component's cost is its children's (`Cost.combine`) times its
            # own copies: the children's names are read in their own frames,
            # where a name the node rebinds (`node_rebinds`) is its own.
            own = walk.own_copies.values.get(uid)
            if own is not None:
                add(uid, own.stored, own.active)
            continue
        add(uid, cost.flops, cost.params)
    for row in walk.tensors():
        add(row.uid, row.storage_bytes, row.count)
    for uid, shape in walk.shapes.values.items():
        for dims in (shape.out, *shape.ins, *shape.ports.values()):
            add(uid, *(dims or ()))
    return set().union(*(frame_names(walk, uid, found)
                         for uid, found in names.items()))


def frame_names(walk: Any, uid: int, names: Any) -> set:
    """`names`, read at node `uid` of `walk`, in the names of the walk's own
    frame, kept where it binds them to anything but 0 or 1.

    Each frame from the node out to its block puts a name it binds in the
    names its declaration reads (`declared_names`): what the node's op
    rebinds (`node_rebinds`: a narrowed `s_kv`, the live pairs `s_pairs` by
    their definition), the node's own `dims:` and each enclosing call's — so a draft pool's `n_route_targets` is the
    `n_dspark_experts` its call rebinds `n_experts` to. A block
    assignment's rebinding is the block's own vocabulary and stays as it
    is named there (`n_kv_heads` on a sliding-window layer, shown with the
    rebinding beside it). A name no frame declares and the walk's frame
    does not bind is a node's own, valued by the engine; a name bound to 0
    or 1 — a flag, a count of one (`B` at one request) — reads as the
    number. The phase's derived extents read as the laws it declares
    (`STEP_LAWS`)."""
    program = walk.program
    node = program.by_uid(uid)
    names = set(names)
    # The node's op rebinds in its own frame first (`node_rebinds`): the
    # narrowed `s_kv`, and the live pairs `s_pairs` as their definition.
    rebound = {name: value.law if isinstance(value, Definition) else value
               for name, value in
               (walk.node_rebinds.values.get(uid) or {}).items()}
    names = declared_names(names, rebound, None)
    at = node
    while at is not None:
        if at is node or at.is_component:
            binds = dict(at.dims or {})
            written = dict(at.dims_raw or {})
            if at.dim is not None or at.dim_raw is not None:
                binds["dim"] = at.dim
                if at.dim_raw is not None:
                    written["dim"] = at.dim_raw
            names = declared_names(
                names, binds, written,
                set(walk.node_dims.values.get(at.uid) or ())
                if at.is_component else None)
        at = at.parent
    # The phase's derived step extents read as the laws it declares them
    # to be (`STEP_LAWS`): the pairs and the keys before the step are the
    # step's lengths.
    names = declared_names(names, {}, None, rederived=STEP_LAWS)
    frame = walk.binding.flat()

    def named(name: str) -> bool:
        value = value_of(frame[name])
        return varies(value) or value not in (0, 1)

    return {name for name in names if name in frame and named(name)}


def _component_own_dims(walk: Any, depends: set) -> Dict[str, Any]:
    """The component's OWN editable knobs: the dims its evaluation depends
    on (`_dependency_dims`), valued as the isolated walk (`walk`) binds
    them at its root — where an edit of them lands — minus the shared
    contract and deployment symbols. Seed and the editable table share
    it."""
    model = walk.program.base_binding
    out: Dict[str, Any] = {}
    for dim in sorted(depends):
        value = value_of(model.get(dim))
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or _is_contract_dim(dim)
        ):
            continue
        out[dim] = value
    return out


def _formula_names(texts: Any) -> set:
    """Dim names the authored formula texts are expressions in."""
    names: set = set()
    for text in texts:
        if text:
            names |= {str(s) for s in read_law(text).free_symbols}
    return names


def _dim_values_for(walk: Any, names: Collection[str]) -> Dict[str, int]:
    """Whole values the isolated walk bound for exactly those `names`.

    A block frame does not carry a leaf's own flags: `bias` is bound on the
    linear that has it, while the box formula over that leaf still names it.
    Include a name only when every leaf that binds it gives the same whole
    value — two leaves disagreeing is a second law, not a global value.
    """
    if not names:
        return {}
    wanted = set(names)
    seen: Dict[str, set] = {}
    for dims in walk.node_dims.values.values():
        for name in wanted & set(dims):
            try:
                value = int(value_of(dims[name]))
            except (TypeError, ValueError):
                continue
            seen.setdefault(name, set()).add(value)
    return {name: next(iter(values)) for name, values in seen.items()
            if len(values) == 1}


def _diff_entry(base_resolved: Any, row: Dict[str, Any], loader: Any) -> Entry:
    """One benched row as a graph-diff entry: its drawn graph and L1 view,
    plus what only the spec holds — the custom ops its nodes may name and
    the drawn component's declared interface."""
    # The graph is drawn where it was benched: in the caller's host when one
    # was given (drop-in), else in the entry's own model.
    host_res = (
        base_resolved if base_resolved is not None
        else _match_model(row["model_id"], loader)
    )
    host = inject_component(host_res, row["component"], loader)
    graph = host.spec.get_block(_drawn_as(host, row["component"]))
    return Entry(
        key=row["entry"],
        block_graph=row["block_graph"],
        l1=row.get("l1") or {},
        dims=row["dims_used"],
        inputs=graph.inputs,
        outputs=graph.outputs,
        reads=graph.reads,
    )


def graph_diff(
    base_resolved: Any,
    results: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    loader: Any,
) -> Dict[str, Any]:
    """The compute-graph diff of every entry (`backend.views.diff`).

    Per entry, not per component: two bindings of one component are two
    entries. Drawn only when every entry benched — a diff covering part of
    the selection would read as the whole — and refused, with the reason,
    when the diff itself declines."""
    if errors:
        return GraphDiff.refused(
            f"{len(errors)} of {len(errors) + len(results)} entries did not bench: "
            + ", ".join(e["entry"] for e in errors)
        )
    try:
        return GraphDiff.of(
            [_diff_entry(base_resolved, r, loader) for r in results]
        ).to_dict()
    except DiffRefused as exc:
        return GraphDiff.refused(str(exc))


def whole_model_kv(kv: Dict[str, Any], cache: Any, *, model_id: str,
                   site: CallSite, precision: Any) -> Dict[str, Any]:
    """The entry's model's own cache of the component (`cache`, a
    `CacheView` narrowed to its cache-holding calls, or the reason there is
    none), written onto the entry's KV sweep: each point's
    `model_storage_bytes` is the sum of the model's calls at their own laws
    and copies — `model_formula` lists them — and the returned
    `model_layers` says how many copies hold it.

    The model is the one the entry describes, so its call must hold, per
    copy, exactly the cache the entry's own row reports; where it does not,
    there is no whole-model number for the entry (`refused`), never one of
    a different model beside it."""
    def refused(reason: str) -> Dict[str, Any]:
        return {"model_id": model_id, "layers": None, "refused": reason}

    if isinstance(cache, str):
        return refused(cache)
    if not cache.entries:
        for point in kv["storage_by_cache_len"]:
            point["model_storage_bytes"] = 0
        return {"model_id": model_id, "layers": 0, "refused": None}
    program = cache.source_analysis.program
    view = cache
    own = [entry for entry in view.entries
           if entry.block_index == site.invocation
           and (
               program.by_uid(entry.cache.component).display_path == site.path
               or program.by_uid(
                   entry.cache.component
               ).display_path.startswith(site.path + ".")
           )]
    if not own:
        return refused(f"{model_id}'s cache has no node at {site.label}")
    one_copy = replace(
        view, entries=tuple(entry._replace(count=1) for entry in own))

    def slot_storage_at(length: int) -> int:
        total = 0
        for node in kv.get("slot_nodes") or []:
            points = node.get("storage_by_cache_len") or []
            total += sum(
                int(point["storage_bytes"]) for point in points
                if int(point["cache_len"]) == length)
        return total

    def component_layer(entry: Any) -> Optional[Tuple[int, int]]:
        node = program.by_uid(entry.cache.component)
        while node is not None:
            if binding_of(program.spec, node.op, site.component):
                return entry.block_index, node.uid
            node = node.parent
        return None

    layer_counts: Dict[Tuple[int, int], int] = {}
    for entry in view.entries:
        identity = component_layer(entry)
        if identity is not None:
            layer_counts.setdefault(identity, int(entry.count))
    sums = []
    for point in kv["storage_by_cache_len"]:
        n = point["cache_len"]
        held = int(one_copy.storage_bytes_at(n))
        shown = int(point["storage_bytes"]) + slot_storage_at(n)
        if held != shown:
            return refused(
                f"{model_id} holds {held:,d} B per copy at {site.label} for "
                f"{n:,d} tokens, the entry {shown:,d} B: with "
                "these dims the model does not run that call as the entry does")

        # One term per assignment/component call: a source CSA2 layer has its
        # window beside the caches its bound slots own, but it is still one
        # layer in the model's cache budget.
        grouped: Dict[Tuple[int, str, int], int] = {}
        for entry in view.entries:
            key = (entry.block_index, site.path, int(entry.count))
            grouped[key] = grouped.get(key, 0) + int(
                replace(view, entries=(entry,)).storage_bytes_at(n))
        terms = [
            {"label": f"{block}/{path} × {count}", "result": result}
            for (block, path, count), result in grouped.items()
        ]
        sums.append(terms)
    for point, terms in zip(kv["storage_by_cache_len"], sums, strict=True):
        point["model_storage_bytes"] = sum(t["result"] for t in terms)
        point["model_formula"] = _addends_formula(
            terms, total=point["model_storage_bytes"], unit="B")
    return {"model_id": model_id,
            "layers": sum(layer_counts.values()),
            "refused": None}


class EntryRefused(Exception):
    """A compare entry that cannot be evaluated; `error` is its row of the
    response's `errors[]`."""

    def __init__(self, error: Dict[str, Any]):
        super().__init__(error["error"])
        self.error = error


@dataclass(frozen=True)
class BenchEntry:
    """One compare entry, resolved: the model its weights come from
    (`owner`, named `weights_from`), the call it is evaluated at, the calls
    it could name, and its own dims (`own`: explicit edits over the
    binding's seed). `seed` is the binding's dims when this request seeded
    them, else None."""

    key: str
    name: str
    model_id: Optional[str]
    weights_from: str
    owner: Any
    site: CallSite
    calls: List[Dict[str, Any]]
    own: Dict[str, Any]
    seed: Optional[Dict[str, Any]]


def resolve_entry(entry: Dict[str, Any], *, base: Any, host_id: Optional[str],
                  loader: Any) -> BenchEntry:
    """Resolve one compare entry `{name, model_id?, call?, dims?, seeded?,
    label?}` (see `module_bench`), or raise `EntryRefused`."""
    name = entry["name"]
    model_id = entry.get("model_id")
    call = entry.get("call")
    # The entry's identity: an explicit label (a saved variant's key) or
    # the component×binding×call triple.
    key = entry.get("label") or entry_key(name, model_id, call)
    entry_host_id = model_id or host_id
    if not entry_host_id:
        raise EntryRefused(
            {
                "component": name,
                "entry": key,
                "model_id": model_id,
                "error": "bind this component to a shipped model",
                "missing_dims": [],
            }
        )
    # The weights are a checkpoint's: a bound entry's model's, priced
    # where that model calls the component at its formats. A caller's
    # host keeps drop-in behavior; without one, the binding IS the host.
    # The call the entry is evaluated at is that model's, chosen the same
    # way whether the model came as the entry's binding or as the host.
    weights_from = entry_host_id
    calls: List[Dict[str, Any]] = []
    try:
        owner = _match_model(weights_from, loader) if model_id else base
        calls = component_calls(owner, name, loader)
        site = _call_site(owner, name, call, weights_from, loader)
    except CONTRACT_ERRORS as exc:
        raise EntryRefused(
            {
                "component": name,
                "entry": key,
                "model_id": model_id,
                "error": str(exc),
                "missing_dims": [],
                "calls": calls,
            }
        ) from None
    # Shared dims, then this entry's own dims: explicit edits win; dims the
    # user did not set are seeded from the bound model (if any); a per-entry
    # override can never touch a shared contract dim.
    own = {
        k: v
        for k, v in (entry.get("dims") or {}).items()
        if not _is_contract_dim(k)
    }
    seed_dims: Optional[Dict[str, Any]] = None
    if model_id and not entry.get("seeded"):
        # First bench of a bound entry: seed dims the user has not set from
        # the binding model. Once the client adopts the echo it marks the
        # entry seeded, so a dim the user deleted is not resurrected here.
        # A FAILED seed is an error, not a fallback: benching on with base
        # dims while labeling the column `@ model` would be a disguised
        # guess — exactly what this endpoint promises never to do.
        missing_seed = set(own.keys())
        try:
            seed = _dims_at(owner, name, loader, site)
        except CONTRACT_ERRORS as exc:
            logger.info("entry seed %s from %s failed: %s", name, model_id, exc)
            raise EntryRefused(
                {
                    "component": name,
                    "entry": key,
                    "model_id": model_id,
                    "error": f"cannot resolve {name} as {model_id} parameterizes it: {exc}",
                    "missing_dims": [],
                    "calls": calls,
                    "call": call_dto(site),
                }
            ) from None
        own.update(
            {
                k: v
                for k, v in seed.items()
                if k not in missing_seed and not _is_contract_dim(k)
            }
        )
        seed_dims = {
            k: v for k, v in seed.items() if not _is_contract_dim(k)
        }
    return BenchEntry(key=key, name=name, model_id=model_id,
                      weights_from=weights_from, owner=owner, site=site,
                      calls=calls, own=own, seed=seed_dims)


def module_bench(
    *,
    base_model_id: Optional[str] = None,
    components: Optional[List[str]] = None,
    batch: int,
    input_tokens: int,
    d_model: Optional[int] = None,
    dim_overrides: Optional[Dict[str, Any]] = None,
    per_component_dims: Optional[Dict[str, Dict[str, Any]]] = None,
    entries: Optional[List[Dict[str, Any]]] = None,
    precision: Any,
    kv_cache_lens: Optional[List[int]] = None,
    gpu_sku: Optional[str] = None,
    model_cache: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Bench each compare entry at its own binding, each with its OWN dims.

    An entry is `{name, model_id?, call?, dims?}`: `model_id` is the host
    model — dims the entry does not set explicitly are seeded from that
    model's binding; `dims` (user edits) always win. `call` names the call of
    that model the entry is evaluated at (`CallSite.of`): a model that calls
    the component once, or at calls that all evaluate alike, needs none
    (`_call_site`); one whose calls differ — or a base it is dropped into,
    which does not call it — is refused until the entry names one, and every row and refusal lists the calls it can name
    (`calls`). Compare UI sends no `base_model_id`. Callers that still pass
    one keep drop-in-into-host behavior (tests, editor slot-diff). A missing
    host on an entry is an error, not a silent default.

    `d_model` (if given) overrides hidden size across entries. No hidden
    sizing policy: a dim a component needs that nothing supplies surfaces
    as a per-entry error.

    `model_cache` (`compose.model_cache_at`, injected by the caller: the
    edited model is the editor's) gives each entry its model's whole cache
    of the component under the dims the entry ran with (`whole_model_kv`);
    without it the entries carry no whole-model KV.
    """
    if not entries:
        entries = [
            {
                "name": n,
                "model_id": None,
                "dims": dict((per_component_dims or {}).get(n) or {}),
            }
            for n in (components or [])
        ]
    if not entries:
        raise ValueError("entries or components must be a non-empty list")
    loader = get_default_loader()
    host_id = (base_model_id or "").strip() or None
    base = _match_model(host_id, loader) if host_id else None

    overrides: Dict[str, Any] = dict(dim_overrides or {})
    if d_model is not None:
        overrides.setdefault("d_model", d_model)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    applied_per_entry: Dict[str, Dict[str, Any]] = {}
    seed_per_entry: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        try:
            ready = resolve_entry(entry, base=base, host_id=host_id, loader=loader)
        except EntryRefused as exc:
            errors.append(exc.error)
            continue
        name, model_id, key = ready.name, ready.model_id, ready.key
        owner, site, calls, own = ready.owner, ready.site, ready.calls, ready.own
        weights_from = ready.weights_from
        if ready.seed is not None:
            seed_per_entry[key] = ready.seed
        if own:
            applied_per_entry[key] = own
        comp_ov = {**overrides, **own}
        host = None
        try:
            host = base if base is not None else owner
            row = bench_component(
                host,
                name,
                batch=batch,
                input_tokens=input_tokens,
                dim_overrides=comp_ov,
                loader=loader,
                precision=precision,
                kv_cache_lens=tuple(kv_cache_lens or ()),
                site=site,
                bound=bool(model_id),
                gpu_sku=gpu_sku,
            )
            row["entry"] = key
            row["model_id"] = model_id
            row["call"] = call_dto(site)
            row["calls"] = calls
            row["kv"]["model_layers"] = None
            if model_cache is not None:
                # The entry's model with the dims its isolated walk ran
                # with: the shared ones, and its own where they differ from
                # what the model binds at the call (its edits; a seed equal
                # to the call's frame is no edit).
                frame = site.dims_in(owner)
                edits = {k: v for k, v in own.items() if frame.get(k) != v}
                row["kv"]["model_layers"] = whole_model_kv(
                    row["kv"],
                    model_cache(owner, model_id=weights_from,
                                component=site.component,
                                site=site, dims={**overrides, **edits},
                                loader=loader, precision=precision),
                    model_id=weights_from, site=site, precision=precision)
            results.append(row)
        except CONTRACT_ERRORS as exc:
            probe = host if host is not None else base
            if probe is None:
                err = {
                    "component": name,
                    "error": str(exc),
                    "missing_dims": [],
                }
            else:
                err = _diagnose_contract_error(probe, name, exc, comp_ov, loader)
            err["entry"] = key
            err["model_id"] = model_id
            err["calls"] = calls
            # The call it was refused at.
            err["call"] = call_dto(site)
            errors.append(err)

    out: Dict[str, Any] = {
        "base_model_id": host_id or (results[0].get("model_id") if results else "") or "",
        "contract": {
            "batch": batch,
            "input_tokens": input_tokens,
            # What each phase evaluated: s_query new tokens per request
            # against s_kv cached positions.
            "phases": phase_contract(batch, input_tokens),
            # Echo what actually took effect (contract dims filtered out, seeds
            # folded in), not the raw request — so the DTO can't claim an
            # override that was dropped, and the client can adopt the seed.
            "per_component_dims": applied_per_entry,
            # The binding's own dims for each entry seeded this request,
            # before its explicit dims won — the baseline its edits are
            # measured against (a shared link carries only the edits).
            "seed_dims": seed_per_entry,
        },
        "results": results,
        "errors": errors,
    }
    if len(entries) >= 2:
        out["graph_diff"] = graph_diff(base, results, errors, loader)
    return out
