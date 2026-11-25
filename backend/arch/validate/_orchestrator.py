"""Top-level `validate(model_id, spec_name)` entry.

Composes the 4 sub-systems:
  1. Resolve spec from config (or explicit name).
  2. Run the check modules: structural forward-source hints stay FYI
     warnings; shape / type-contract / cross-layer findings are promoted
     into `errors[]` by `_gating_errors` (the publish gate).
  3. Compute per-tensor diff via the matching module.
  4. Return a single dict the CLI / test / API consumers all read.

This is the public `validate` symbol — it stays importable as
`backend.arch.validate.validate` (via the package __init__).
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Sequence


from ._finding import Finding
from ._matching import (
    CheckpointJoin,
    CheckpointUnavailable,
    _classify_ratio,
    _flatten_block_nodes,
    _get_safetensors_data,
    _unmatched_report_leaves,
    match_checkpoint_to_hf,
)
from ._shapes import _run_shape_consistency_check, run_dim_map_consistency_check
from ._structural import _run_structural_checks
from ._cache_writes import run_cache_write_check
from ._model_card import run_model_card_check
from ._reuse import run_reuse_check
from ._selection_reads import run_selection_read_check
from ._type_contracts import run_type_contract_checks
from ._module_exec import run_module_exec_checks
from ._axioms import run_axiom_checks
from ._weight_ledger import run_weight_ledger_check
from ._analysable import ENGINE_KIND, run_analysable_check
from ._tied_copy import run_tied_copy_check
from ._unread_parameters import run_unread_parameter_check
from ._checkpoint_names import run_checkpoint_name_check
from ._rotary_positions import run_rotary_position_check
from ._attention_masks import run_attention_mask_check
from ._cross_layer_channels import cross_layer_anchor_notes, run_cross_layer_checks
from ._schedule_exhaustiveness import run_layer_schedule_check
from ._ablation import parse_disabled, lever_disabled, SOURCE_DIFF_KINDS
from backend.arch.eval.checkpoint import Reconciliation
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.custom_id import is_custom_id


@dataclass(frozen=True)
class GateOutcome:
    """What an injected gate returns: its findings, and a report the result
    carries under `analysis_replay` (what it ran, for the review report)."""
    findings: List[Finding]
    report: Dict[str, Any] = field(default_factory=dict)


#: An injected gate: `(resolved, config, model_result) -> GateOutcome`.
DownstreamGate = Callable[[Any, Any, Any], GateOutcome]


class ConfigUnavailable(RuntimeError):
    """No config could be read for the model — snapshot, custom cache and
    the Hub all failed. Carries the model id and the last cause."""


class SpecEvalError(Exception):
    """Evaluating this spec failed at a named stage. The message already
    names the spec, the stage, and (when known) the node or field so a
    writer can act on it. Infrastructure failures never become this."""


# Config / checkpoint / Hub / import / resource failures are deployment
# problems. They must not become `errors[]` the spec writer is asked to fix.
_INFRASTRUCTURE = (
    ConfigUnavailable, CheckpointUnavailable, OSError, ImportError, MemoryError,
)


def _format_eval_error(exc: BaseException, *, spec: str, stage: str) -> str:
    """One `errors[]` line: which spec, which stage, which node/field, why.

    A bare `KeyError: 'in'` is not actionable; wrapping it as
    `evaluating spec 'x' during projection field 'in': KeyError: 'in'` is.
    """
    from backend.arch.loader import SpecLoadError
    node = exc.path if isinstance(exc, SpecLoadError) else None
    field = None
    if isinstance(exc, KeyError) and exc.args:
        key = exc.args[0]
        if isinstance(key, str) and key.isidentifier():
            field = key
    loc = [f"evaluating spec `{spec}` during {stage}"]
    if node:
        loc.append(f"at node `{node}`")
    if field is not None:
        loc.append(f"field `{field}`")
    text = str(exc)
    name = type(exc).__name__
    reason = text if text.startswith(name) else f"{name}: {text}"
    return f"{' '.join(loc)}: {reason}"


def _run_stage(stage: str, spec: str, fn):
    """Run one eval stage. Infrastructure re-raises; anything else becomes
    a `SpecEvalError` naming this spec and stage."""
    try:
        return fn()
    except _INFRASTRUCTURE:
        raise
    except SpecEvalError:
        raise
    except Exception as exc:
        raise SpecEvalError(
            _format_eval_error(exc, spec=spec, stage=stage)
        ) from exc


def _fetch_config(model_id: str, overlay_dir=None):
    """Config for the model; `ConfigUnavailable` when it cannot be fetched.

    `hf.metadata.fetch_config`, the order every consumer reads in (custom
    dir, snapshot — shared, then the overlay's — then the Hub), so offline
    validate judges the same config and checkpoint record the rest of the
    pipeline sees.
    """
    from backend.hf.metadata import fetch_config
    config = fetch_config(model_id, overlay_dir)
    if config is None:
        raise ConfigUnavailable(f"Cannot fetch config for {model_id}")
    return config


def _resolve_spec(model_id, spec_name, config, overlay_dir, header=None):
    """Load and resolve the spec for this config; None when nothing matches.

    Custom models load the spec directly from their own directory,
    bypassing match() which would compete with all specs sharing the same
    model_type (e.g. another custom model's stale spec with more dims could
    win despite having wrong dim bindings).
    """
    from backend.arch.loader import SpecLoader, custom_spec_file, overlay_kwargs

    loader = SpecLoader(**overlay_kwargs(overlay_dir))
    if spec_name:
        return loader.resolve_by_name(spec_name, config, model_id=model_id,
                                      header=header)
    if is_custom_id(model_id):
        spec_path = custom_spec_file(model_id, overlay_dir)
        if spec_path:
            spec = loader.load_model_spec(spec_path)
            return loader.resolve(spec, config, model_id=model_id,
                                  header=header) if spec else None
        return None
    return loader.match(
        config, model_id=model_id, header=header,
    )


def _spec_block_breakdown(model_result):
    """Per-block-template params breakdown for the report."""
    spec_blocks = []
    for tmpl_name, count, br in model_result.block_results:
        leaves = _flatten_block_nodes(br)
        spec_blocks.append({
            "template": tmpl_name,
            "count": count,
            "params_per_block": br.total_params,
            "params_total": br.total_params * count,
            "nodes": [{"name": n["name"], "params": n["params"]} for n in leaves],
        })
    return spec_blocks


def _stream_pairings(resolved) -> list:
    """`Program.stream_pairings`, one dict per element and reader."""
    return [asdict(row) for row in Program.of(resolved).stream_pairings]


def _whole_stream_pairings(resolved) -> list:
    """`Program.whole_stream_pairings`, one dict per reader of a whole
    stream or its prefix: each derived group of a `read: all` stack, each
    node that joins a stream (`read: concat`), and a model-level op
    reading a whole stream, as a history or joined."""
    return [asdict(row)
            for row in Program.of(resolved).whole_stream_pairings]


def _storage_coverage(model_result, hf_data):
    """Per-tensor physical-element match + both sides' unmatched lists.

    Checkpoint names, packing, dtype, and multiplicity all come from
    materialization; display-node suffixes are not a storage identity.

    The spec-side coverage comes off the SAME model_result everything else
    uses: the header tensors no stored tensor is named come from the one
    name join (`CheckpointJoin`), and `_unmatched_report_leaves` finds
    report leaves with params that no row covers (the refine loop's rename
    targets).
    """
    matched = match_checkpoint_to_hf(model_result.analysis, hf_data["header"])
    unmatched_spec_nodes = _unmatched_report_leaves(
        model_result.block_results, model_result.analysis.tensor_index(),
    )
    unmatched_hf = CheckpointJoin.of(
        model_result.analysis, hf_data["header"]).unmatched_header
    return matched, unmatched_spec_nodes, unmatched_hf


def _guarded(label, fn, *args, **kwargs):
    """Run one check module. A crash becomes a `check_crashed` Finding in
    the same list the caller would have stored successes in, so gating
    buckets still fail the publish gate and Layer-2 diagnostics stay
    warnings. Infrastructure (OSError, import, config/checkpoint) re-raises
    so it cannot be filed as a spec finding."""
    try:
        return fn(*args, **kwargs)
    except _INFRASTRUCTURE:
        raise
    except Exception as exc:
        print(f"(warning: {label} failed: {exc})", file=sys.stderr)
        return [Finding(
            kind="check_crashed",
            detail=f"{label} raised {type(exc).__name__}: {exc}",
        )]


def _finding_text(w: Finding) -> str:
    """The human text of a finding — its one text field."""
    return w.detail


class _Checks:
    """Every check module's findings for one validate() run, in run order."""

    def __init__(self, model_id, resolved, config, model_result, hf_data):
        #: Every check this run ran, and the ones it had nothing to read for
        #: (a model stored without a checkpoint), by label — the verdict's
        #: own account of what it judged (`checks` in the result).
        self.ran: List[str] = []
        self.skipped: List[str] = []
        #: Checks that raised (a `check_crashed` finding): they ran and
        #: judged nothing, whatever their bucket does with the finding.
        self.crashed: List[str] = []
        # Structural: compare each block template's forward() source against
        # its spec block graph. Catches topology bugs that params alone miss
        # (missing residual adds, reordered norms, etc.). Purely best-effort
        # — quiet when forward source isn't available.
        self.structural_warnings = self._run(
            "structural check", _run_structural_checks, model_id, resolved)
        # Cache-write quantisation: a cache has no checkpoint bytes to diff,
        # so the forward's own writes are the witness — a write it quantises
        # first wants a cache slot stored at that format. Hints, never a
        # gate. See _cache_writes.py.
        self.structural_warnings += self._run(
            "cache write check", run_cache_write_check, model_id, resolved)

        # Shape consistency: for every primitive op in every used block
        # template, the analysis-derived input shape (from its incoming edge)
        # must match the op's declared shape_in template (resolved against
        # current dim values). Catches geometric bugs that params alone miss
        # (e.g. concat'ing tensors whose ranks disagree, view_split on a flat
        # dim that isn't n_heads·d_head, missing broadcast before a 4-D
        # concat).
        self.shape_mismatches = self._run(
            "shape check", _run_shape_consistency_check, resolved,
            model_result.block_results)

        # Per-op DimMap consistency: every primitive op's input→output dim
        # correspondence (a `from` copies its input dim, a `split`'s parts
        # multiply back, a `merge` equals the product) must hold against the
        # resolved shapes. Generalises the shape check's few multi-input
        # families to EVERY mapped op — so a reshape on a flat dim that isn't
        # n_heads·d_head, or a node whose resolved input doesn't fit the op
        # (a mis-wired edge / missing reshape), surfaces here. Positional, so
        # a concretized head axis is checked the same. Best-effort.
        self.shape_mismatches += self._run(
            "dim-map consistency check", run_dim_map_consistency_check, resolved,
            model_result.block_results)

        # Type contracts: declared KV/attention semantics (kv_label /
        # kv_cached / kv_window) are internally consistent — the non-weight
        # attributes the bytes diff is blind to.
        self.type_contract_warnings = self._run(
            "type-contract checks", run_type_contract_checks, model_id, resolved)

        # A score that reads through a selection (`select` operand) reads a
        # bounded number of positions; every cached tensor its keys come
        # from must say so in its read law. Graph-internal, FP-free over the
        # corpus; gates under the `axiom` lever. See _selection_reads.py.
        self.selection_read_errors = self._run(
            "selection-read check", run_selection_read_check,
            model_result.analysis)

        # Drafts that restate one another or a library component: the
        # shared structure belongs in one component bound per call. Warns,
        # never gates. See _reuse.py.
        self.reuse_warnings = self._run("reuse check", run_reuse_check, resolved)

        # The analysis's parameter counts against the ones the model card
        # states: the checkpoint anchors cannot see how a token uses what is
        # stored. Warns, never gates. See _model_card.py.
        self.model_card_warnings = self._run(
            "model-card check", run_model_card_check, model_id, model_result)

        # Module execution: run each block template through the engine at
        # basic configs (B=1/2, S) — does the spec actually COMPUTE? Catches
        # a broken module (crash / non-finite / non-batch-linear) before
        # publish, especially newly-authored ones the static checks can't vet.
        self.module_exec_warnings = self._run(
            "module-exec checks", run_module_exec_checks, resolved)

        # Soft-fidelity facts the numeric anchors are blind to (both WARN,
        # never gate): the FFN activation vs the config's declared
        # `hidden_act`, and whether forward() source was even available to
        # model the block topology from. See _activation.py.
        def _activation():
            from ._activation import run_activation_check
            return run_activation_check(resolved, config)

        def _forward_source():
            from ._activation import run_forward_source_completeness_check
            return run_forward_source_completeness_check(model_id, resolved)

        self.activation_warnings = self._run("activation check", _activation)

        # A param slot whose value the config names (`ParamSpec.config`, the
        # GDN norm gate's `output_gate_type`) is bound to that op. Grounded
        # in a config value, like the schedule check; gates. See
        # _activation.py.
        def _param_config():
            from ._activation import run_param_config_check
            return run_param_config_check(resolved, config)

        self.param_config_errors = self._run(
            "param-config check", _param_config)

        # The text rotation against the section list the config's rope
        # parameters declare (`mrope_section`, `xdrope_section`). Same
        # arithmetic either way, so no numeric anchor sees it. See
        # _rotary_positions.py.
        def _rotary():
            return run_rotary_position_check(resolved, config)

        self.rotary_position_errors = self._run(
            "rotary-position check", _rotary)
        # The mask each attention declares (`is_causal` / `is_chunked`):
        # stated inside an encoder stack, one mask per attention, a
        # bidirectional band symmetric. See _attention_masks.py.
        self.attention_mask_errors = self._run(
            "attention-mask check", run_attention_mask_check, resolved)
        # Stream anchors no recorded source could check are reported here,
        # not gated: the spec cannot supply the missing source.
        self.forward_source_warnings = self._run(
            "forward-source completeness check", _forward_source) + self._run(
            "cross-layer anchor notes", cross_layer_anchor_notes,
            model_id, resolved)

        # Axiom checks: each component's declared A-class anchors (the
        # source-derived single-layer shape/compute/KV facts) must match the
        # engine's computed values in this model's dim context. A violation
        # means the spec drifted from the locked source fact.
        self.axiom_warnings = self._run(
            "axiom checks", run_axiom_checks, model_id, resolved)

        # Per-layer schedule exhaustiveness: when a spec drives a per-layer
        # template from a `config_list`, every value that list actually takes
        # must be handled (a `rest:` fallback can't silently absorb an
        # unmodelled schedule value). Grounded in the config VALUE, opt-in,
        # not name-matching — the old name-based config-coverage heuristic was
        # removed (faithfulness to the model is the inspector agent's job,
        # grounded in forward() source, not a config field-name regex). See
        # _schedule_exhaustiveness.py.
        self.coverage_errors = self._run(
            "schedule-exhaustiveness check", run_layer_schedule_check, resolved)

        # Cross-layer streams: a declared inner stream (state threaded
        # across layers beside the stack's carried one) must anchor to the forward
        # source and have a producer and a consumer (truthfulness, hard).
        # Source-derived → dropped with the `source` lever. Anchors with no
        # recorded source to check are noted with the forward-source warnings
        # above. See _cross_layer_channels.py.
        self.cross_layer_warnings = self._run(
            "cross-layer stream check", run_cross_layer_checks, model_id, resolved)

        # Packing / quant-scheme consistency: decompose the byte product back
        # into (stored element count, stored dtype) and anchor each
        # independently against the safetensors header. Catches a compensating
        # packing error (2× logical count × ½ dtype width) that the aggregate
        # byte sum — and the non-gating total_params diagnostic — would miss.
        # Also: every quant override row decides some stored tensor's format
        # (a dead row changes no byte, so the anchor cannot see it), and a
        # checkpoint whose quant_method no scheme is named for is described
        # by the spec's quant_context methods. Every role holds one stored
        # precision per stack, in prefill and decode form alike (precision is
        # chosen by role, so a role the checkpoint splits is refused with the
        # finer role to declare), and a checkpoint tensor resolves to one
        # format in both forms. See _packing_consistency.py.
        def _packing():
            from backend.arch.validate._packing_consistency import (
                run_packing_consistency_check, run_phase_format_check,
                run_quant_method_check, run_quant_override_check,
                run_quant_role_check,
            )
            # The decode form of the same checkpoint walk: components with a
            # `decode_variant` swap in their twin, reading the same tensors.
            decode = Deployment(Program.of(resolved)).at(
                batch=1, input_tokens=1).during("decode").checkpoint()
            against_file = self._skip("packing against the checkpoint") if hf_data is None else (
                run_packing_consistency_check(
                    model_result.analysis, hf_data["header"])
                + run_quant_method_check(
                    resolved.spec.quant_context,
                    hf_data.get("quant_method_unknown")))
            return (against_file
                    + run_quant_override_check(model_result.analysis)
                    + run_quant_role_check(model_result.analysis, decode)
                    + run_phase_format_check(model_result.analysis, decode))

        self.packing_errors = self._run("packing-consistency check", _packing)

        # The per-rank weight ledger every drawn page prices its nodes off
        # (`capacity.weights` over the layout): a stored weight with no
        # declared `weight_axes` cannot be placed, and the byte anchor above
        # never builds it. See _weight_ledger.py.
        self.weight_ledger_errors = self._run(
            "weight-ledger check", run_weight_ledger_check,
            model_result.analysis)

        # The analysability checks that need no deployment: every op the
        # walk reaches has a layout transition rule, and the per-rank weight
        # join builds at one GPU. The deployment replay is an injected gate
        # (`downstream_gates`). See _analysable.py.
        self.analysable_errors = self._run(
            "analysability check", run_analysable_check,
            resolved, model_result.analysis,
            ledger_built=not self.weight_ledger_errors)

        # A tie against the checkpoint's names: `tied_weight.checkpoint_copy`
        # is declared exactly when the file still ships the tied layer's own
        # tensor. See _tied_copy.py.
        self.tied_copy_errors = self._skip("tied-copy check") if hf_data is None else self._run(
            "tied-copy check", run_tied_copy_check,
            model_result.analysis, hf_data.get("header"))

        # A stored parameter no edge reads: a lookup priced as a gather
        # while the graph that uses it reads something else, or a dropped
        # edge. The byte anchor cannot see either. A parameter no graph node
        # reads — the forward never does, or does in a step the graph does
        # not model — says so (`stored_without_graph_reader`, with its
        # kind), and is held to it. See _unread_parameters.py.
        self.unread_parameter_errors = self._run(
            "unread-parameter check", run_unread_parameter_check,
            model_result.analysis)

        # Every tensor name the spec stores is a header name: the other
        # direction of the name join the coverage gate reads. See
        # _checkpoint_names.py.
        self.checkpoint_name_errors = self._skip("checkpoint-name check") if hf_data is None else self._run(
            "checkpoint-name check", run_checkpoint_name_check,
            model_result.analysis, hf_data.get("header"))

    def _run(self, label, fn, *args, **kwargs):
        self.ran.append(label)
        found = _guarded(label, fn, *args, **kwargs)
        if any(isinstance(f, Finding) and f.kind == "check_crashed"
               for f in found or ()):
            self.crashed.append(label)
        return found

    def _skip(self, label) -> list:
        self.skipped.append(label)
        return []


def _format_unmatched_hf(unmatched_hf):
    """Checkpoint tensors no spec node claimed, as report rows + total."""
    rows = []
    total = 0
    for pattern, info in sorted(unmatched_hf.items()):
        rows.append({
            "pattern": pattern,
            "shape": info["shape"],
            "dtype": info["dtype"],
            "count": info["count"],
            "params": info["params"],
        })
        total += info["params"]
    return rows, total


def _format_param_mismatches(matched):
    """Matched-but-different rows + ratio hint. Ratio = spec/hf; a cluster
    of identical ratios across nodes usually means a single dim binding is
    off by that factor."""
    out = []
    for m in matched:
        if m["diff"] == 0:
            continue
        hf = m["hf_params"]
        ratio = round(m["spec_params"] / hf, 4) if hf > 0 else None
        out.append({
            "spec_node": m["spec_node"],
            "hf_module": m["hf_module"],
            "spec_params": m["spec_params"],
            "hf_params": hf,
            "diff": m["diff"],
            "ratio": ratio,
            "ratio_hint": _classify_ratio(ratio),
        })
    return out


def _spec_storage_bytes(analysis) -> int:
    """The spec's side of `model_storage_bytes`: every tensor the header does
    not file as a quant sidecar — the weights, and the buffers the spec
    declares the checkpoint serialises."""
    return analysis.checkpoint_param_bytes() + analysis.checkpoint_buffer_bytes()


def _gating_errors(checks, hf_data, model_result, unmatched_hf_list,
                   param_mismatches, off) -> list:
    """The publish gate. Each contribution rides its ablation lever so a
    disabled lever leaves `errors[]` empty; all guards are unconditional in
    production."""
    errors = []
    if not off("schedule"):
        errors.extend(_finding_text(w) for w in checks.coverage_errors)
        errors.extend(_finding_text(w) for w in checks.param_config_errors)
    if not off("structural"):
        errors.extend(_finding_text(w) for w in checks.rotary_position_errors)
        errors.extend(_finding_text(w) for w in checks.attention_mask_errors)
        errors.extend(_finding_text(w) for w in checks.unread_parameter_errors)
    if not off("packing"):
        errors.extend(_finding_text(w) for w in checks.packing_errors)
    if not off("weight"):
        errors.extend(_finding_text(w) for w in checks.weight_ledger_errors)
        errors.extend(_finding_text(w) for w in checks.tied_copy_errors)
        errors.extend(_finding_text(w) for w in checks.checkpoint_name_errors)

    if not off("analysis"):
        errors.extend(_finding_text(w) for w in checks.analysable_errors)

    # Shape/dataflow consistency is deterministic for the op contracts that
    # emit it. A graph whose tensors do not compose must not publish merely
    # because its checkpoint byte total happens to match.
    if not off("shape"):
        errors.extend(
            f"{m.kind} in "
            f"[{m.template or '?'}] {m.node or '?'} "
            f"({m.op or '?'}): {m.detail}"
            if isinstance(m, Finding) else str(m)
            for m in checks.shape_mismatches
        )

    # Promote the precise, diagnostic-confirmed-FP-free checks to gating:
    #   • axiom_warnings — a source-locked component axiom (indexer/MLA/CSA
    #     KV-growth / shape facts) drifted from the value the engine computes;
    #     a violation means the spec left the locked source fact.
    #   • every KV/attention type contract. Linear/SSM components legitimately
    #     append no per-token KV, but must declare their fixed recurrent state;
    #     transparent adapters are removed by _type_contracts before this gate.
    if not off("axiom"):
        errors.extend(
            _finding_text(w)
            for w in checks.axiom_warnings
            if not (isinstance(w, Finding)
                    and w.kind == "swa_cache_unbounded"
                    and off("swa_window"))
        )
        errors.extend(_finding_text(w) for w in checks.type_contract_warnings)
        errors.extend(_finding_text(w) for w in checks.selection_read_errors)

    # Cross-layer findings test the streams the spec declares: an anchor the
    # source lacks, a stream nobody produces or reads, a shape that disagrees.
    if not off("source"):
        errors.extend(_finding_text(w) for w in checks.cross_layer_warnings)

    # Primary: every safetensors tensor (shape × dtype × count) maps onto
    # a spec leaf and agrees in physical elements. The byte *sum* below
    # cannot see a swapped submodule that still totals the same.
    if not off("weight") and unmatched_hf_list:
        errors.append(
            f"Unmatched HF tensors: {len(unmatched_hf_list)} patterns, "
            f"{sum(u['params'] for u in unmatched_hf_list):,} params not "
            f"mapped to any spec node "
            f"(first: {[u['pattern'] for u in unmatched_hf_list[:5]]})"
        )
    if not off("weight") and param_mismatches:
        first = param_mismatches[0]
        errors.append(
            f"Per-tensor mismatches: {len(param_mismatches)} patterns "
            f"(first: {first.get('spec_node')} spec={first.get('spec_params'):,} "
            f"hf={first.get('hf_params'):,})"
        )

    # Secondary: totals as a checksum on that join.
    if not off("weight") and hf_data is not None:
        spec_storage_bytes = _spec_storage_bytes(model_result.analysis)
        actual_storage_bytes = hf_data["model_storage_bytes"]
        bytes_diff = actual_storage_bytes - spec_storage_bytes
        bytes_diff_pct = (
            bytes_diff / actual_storage_bytes * 100 if actual_storage_bytes > 0 else 0
        )
        if actual_storage_bytes == 0:
            errors.append("Could not fetch model_storage_bytes from safetensors")
        elif bytes_diff != 0:
            errors.append(
                f"Model storage bytes: spec={spec_storage_bytes:,} "
                f"actual={actual_storage_bytes:,} diff={bytes_diff:+,} "
                f"({bytes_diff_pct:+.4f}%)"
            )
        # The file's own byte total (the shard headers' `data_offsets`), the
        # one figure no dtype table reconstructs: every byte the spec says
        # the checkpoint holds, weights, scales and buffers, against it.
        recon = Reconciliation.of(model_result.analysis, hf_data)
        if recon.is_anchored and int(recon.measured_delta) != 0:
            errors.append(
                f"Checkpoint bytes: spec={int(recon.total):,} "
                f"file={int(recon.measured_total):,} "
                f"diff={-int(recon.measured_delta):+,} (the shard headers' "
                f"own byte spans)")
    return errors


def _ablate_warnings(checks, weight_fields, off):
    """Drop each disabled lever's warning field too, so the spec-gen agent's
    stderr diagnostics lose it (not just the gating errors). Inert in
    production (no levers disabled).

    `weight_fields` is the mutable (unmatched_hf_list, unmatched_hf_total,
    unmatched_spec_nodes, param_mismatches) tuple; structural_warnings is
    partitioned by kind — source-diff kinds belong to lever `source`, the
    rest to `structural`.
    """
    unmatched_hf_list, unmatched_hf_total, unmatched_spec_nodes, param_mismatches = (
        weight_fields)
    if off("weight"):
        unmatched_hf_list, unmatched_hf_total, unmatched_spec_nodes = [], 0, []
        param_mismatches = []
    if off("shape"):
        checks.shape_mismatches = []
    if off("axiom"):
        checks.axiom_warnings = []
        checks.selection_read_errors = []
        checks.type_contract_warnings = []
    if off("source"):
        checks.cross_layer_warnings = []
    if off("structural"):
        checks.module_exec_warnings = []
        checks.rotary_position_errors = []
        checks.attention_mask_errors = []
        checks.unread_parameter_errors = []
    if off("schedule"):
        checks.param_config_errors = []
    if off("packing"):
        checks.packing_errors = []
    if off("analysis"):
        checks.analysable_errors = []
    if off("structural") or off("source"):
        def _keep(w):
            # source-diff kinds → lever `source`; everything else → `structural`
            if w.kind in SOURCE_DIFF_KINDS:
                return not off("source")
            return not off("structural")
        checks.structural_warnings = [w for w in checks.structural_warnings
                                      if _keep(w)]
    return unmatched_hf_list, unmatched_hf_total, unmatched_spec_nodes, param_mismatches


def _headline_blocks(model_result, hf_data, off):
    """The spec/actual/diff headline blocks.

    `model_storage_bytes` is the weight-tensor headline (bytes-vs-bytes,
    mixed-precision safe). `quantization_metadata_bytes` is the scale
    bucket against the same header split; `checkpoint_storage_bytes` is
    every safetensors tensor. `total_params` is retained for backward compat
    with consumers (UI, agent prompt's Phase-4 references) but flagged as a
    diagnostic — the HF side depends on `resolve_scheme` picking one
    quant_method correctly, so this number is only reliable for
    single-precision checkpoints.

    The aggregate byte/param headline is itself the strongest `weight`
    signal (spec-vs-checkpoint total gap). When the weight lever is ablated
    it must be withheld too, or "naked" still leaks the diff a strong
    reasoner can close by arithmetic — inert in production.
    """
    spec_storage_bytes = _spec_storage_bytes(model_result.analysis)
    actual_storage_bytes = hf_data["model_storage_bytes"]
    bytes_diff = actual_storage_bytes - spec_storage_bytes
    bytes_diff_pct = (
        bytes_diff / actual_storage_bytes * 100 if actual_storage_bytes > 0 else 0
    )
    spec_total = model_result.total_params
    # The header counts every non-sidecar tensor's elements; the ones the
    # spec declares as buffers (a serialised `inv_freq`) are not parameters.
    actual_total = hf_data["total_params"] - sum(
        hf_data["header"].select(name).elements
        for name in {row.name for row in model_result.analysis.checkpoint_tensors()
                     if row.is_buffer})
    # The checkpoint counts the redundant copy of a tied weight
    # (`tied_weight.checkpoint_copy`); the model's parameters do not. The
    # copy is reported beside `spec` and joins the checkpoint side of the diff.
    copy_params = sum(
        int(value_of(row.elements)) * int(value_of(row.count))
        for row in model_result.analysis.checkpoint_copy_rows()
        if not (row.is_metadata or row.is_buffer))
    # Logical-params diff: driven by HF-side `packed_unpack_factor()` —
    # accurate when the model uses a single quant_method declared in
    # `config.quantization_config`, incorrect for vendors that split methods
    # across multiple top-level config fields (DSv4's `expert_dtype`).
    diff = actual_total - spec_total - copy_params
    diff_pct = diff / actual_total * 100 if actual_total > 0 else 0

    storage_block = {"spec": spec_storage_bytes}
    params_block = {"spec": spec_total}
    if copy_params:
        params_block["checkpoint_copy_params"] = copy_params
    recon = Reconciliation.of(model_result.analysis, hf_data)
    spec_meta = int(recon.metadata_bytes)
    spec_all = int(recon.total)
    metadata_block = {"spec": spec_meta}
    all_block = {"spec": spec_all}
    if not off("weight"):
        storage_block.update({"actual": actual_storage_bytes, "diff": bytes_diff,
                              "diff_pct": round(bytes_diff_pct, 2)})
        params_block.update({"actual": actual_total, "diff": diff,
                             "diff_pct": round(diff_pct, 2)})
        hf_meta_b = int(recon.header_metadata_bytes)
        hf_all = int(recon.header_total)
        # `actual − spec`: the negation of Reconciliation's `spec − header`.
        meta_diff = -int(recon.metadata_delta)
        all_diff = -int(recon.delta)
        metadata_block.update({
            "actual": hf_meta_b, "diff": meta_diff,
            "diff_pct": round(meta_diff / hf_meta_b * 100, 2) if hf_meta_b else 0,
        })
        all_block.update({
            "actual": hf_all, "diff": all_diff,
            "diff_pct": round(all_diff / hf_all * 100, 2) if hf_all else 0,
        })
        if recon.measured_total is not None:
            measured = int(recon.measured_total)
            all_block["measured"] = measured
            all_block["measured_diff"] = -int(recon.measured_delta)
    return storage_block, params_block, metadata_block, all_block


def validate(model_id: str, spec_name: str | None = None,
             overlay_dir: str | None = None, *,
             downstream_gates: Sequence[DownstreamGate] = ()) -> dict:
    """Validate spec params against actual model params with per-tensor diff.

    `overlay_dir` (a per-run overlay root) layers `<dir>/models` +
    `<dir>/components/drafts` onto the shared tree so the agent can validate an
    in-progress PRIVATE spec that isn't in the shared repo tree.

    Two results leave this boundary:

    * A spec rejection — loader / schema / projection / coverage / checks
      — as `{"valid": False, "errors": [...]}` with stage / spec / node
      context, which spec-gen's verdict gate hands to a repair round.
    * An infrastructure failure — config fetch or checkpoint metadata —
      as a raised `ConfigUnavailable` / `CheckpointUnavailable` / `OSError`.
      The worker records those as crash, never as a "fix your spec" prompt.

    `downstream_gates` are checks this package cannot import (the analysis
    replay lives in `backend.inference`): each is called as
    `gate(resolved, config, model_result)` and returns findings. They run
    last, and only when every other gate passed; `analysis` in the result
    says whether they ran (`not_requested` / `not_run` / `passed` /
    `refused`).
    """
    from backend.arch.snapshots import default_overlay_dir

    # The agent's in-loop `validate` runs without --overlay-dir; let it find its
    # per-run overlay via the env the run injected (explicit arg still wins).
    overlay_dir = default_overlay_dir(overlay_dir)

    # Guard-ablation switch (HA_ABLATE_GUARDS). Empty in production → inert.
    _disabled = parse_disabled()
    _off = lambda lever: lever_disabled(_disabled, lever)

    # 1. Checkpoint anchor, then config. Both are infrastructure: a down Hub
    # or an empty safetensors fetch is not a broken spec.
    # A model stored without a checkpoint (`stored_without_checkpoint`: a
    # custom model described in words, or registered without weights) has
    # nothing to compare against — `hf_data` None, and every check that
    # reads only the spec still gates. Any other model's missing checkpoint
    # is a failed fetch.
    from backend.hf.metadata import stored_without_checkpoint
    hf_data = (None if stored_without_checkpoint(model_id)
               else _get_safetensors_data(model_id, overlay_dir))
    if hf_data is not None and "model_storage_bytes" not in hf_data:
        raise CheckpointUnavailable(
            f"Cannot read checkpoint metadata for {model_id}: "
            "safetensors fetch returned no model_storage_bytes"
        )
    config = _fetch_config(model_id, overlay_dir)

    return _evaluate_spec(
        model_id, spec_name, config, overlay_dir, hf_data, _off,
        downstream_gates=downstream_gates,
    )


def _downstream(errors, gates, resolved, config, model_result, spec, off):
    """Run the injected gates when every other gate passed. Returns the
    findings, what happened — a gate skipped because earlier gates failed is
    `not_run`, never a pass — and the gates' reports."""
    if not gates or off("analysis"):
        return [], "not_requested", {}
    if errors:
        return [], "not_run", {}
    findings, report = [], {}
    for gate in gates:
        outcome = _run_stage(
            "analysis replay", spec,
            lambda gate=gate: gate(resolved, config, model_result))
        findings += outcome.findings
        report.update(outcome.report)
    return findings, ("refused" if findings else "passed"), report


def _evaluate_spec(model_id, spec_name, config, overlay_dir, hf_data, _off,
                   *, downstream_gates=()) -> dict:
    """Resolve, project, cover, check this spec. Rejections become
    `errors[]`; infrastructure re-raises."""
    spec = spec_name or model_id
    try:
        resolved = _run_stage("resolve", spec, lambda: _resolve_spec(
            model_id, spec_name, config, overlay_dir,
            header=(hf_data or {}).get("header"),
        ))
        if not resolved:
            return {
                "valid": False,
                "errors": [f"No spec matched for {model_id}" + (f" (tried: {spec_name})" if spec_name else "")],
            }

        spec = resolved.spec.name

        # Unstated dims (Qb, Ob, …) are filled once in `resolve_dimensions`.

        # Compute spec params through the eval data model. The compatibility
        # projection is the temporary boundary for validator response fields; the
        # accounting itself comes from Analysis layers.
        from backend.arch.eval.projection import project_model

        model_result = _run_stage("projection", spec, lambda: project_model(resolved, checkpoint=True))

        spec_blocks = _spec_block_breakdown(model_result)
        stream_pairings = _run_stage(
            "projection", spec, lambda: _stream_pairings(resolved))
        whole_stream_pairings = _run_stage(
            "projection", spec, lambda: _whole_stream_pairings(resolved))

        checks = _run_stage(
            "checks", spec,
            lambda: _Checks(model_id, resolved, config, model_result, hf_data),
        )

        if hf_data is not None:
            matched, unmatched_spec_nodes, unmatched_hf = _run_stage(
                "coverage", spec,
                lambda: _storage_coverage(model_result, hf_data))
        else:
            matched, unmatched_spec_nodes, unmatched_hf = [], [], {}
            checks._skip("checkpoint coverage")
            checks._skip("checkpoint bytes")
        unmatched_hf_list, unmatched_hf_total = _format_unmatched_hf(unmatched_hf)
        param_mismatches = _format_param_mismatches(matched)

        errors = _gating_errors(checks, hf_data, model_result, unmatched_hf_list,
                                param_mismatches, _off)
        unmatched_hf_list, unmatched_hf_total, unmatched_spec_nodes, param_mismatches = (
            _ablate_warnings(
                checks,
                (unmatched_hf_list, unmatched_hf_total, unmatched_spec_nodes,
                 param_mismatches),
                _off))
        storage_block, params_block, metadata_block, all_block = _headline_blocks(
            model_result, hf_data, _off) if hf_data is not None else (
            {"spec": _spec_storage_bytes(model_result.analysis)},
            {"spec": model_result.total_params}, {}, {})
        replayed, analysis, replay_report = _downstream(
            errors, downstream_gates, resolved, config, model_result, spec,
            _off)
        errors.extend(_finding_text(w) for w in replayed)
        analysis_findings = list(checks.analysable_errors) + replayed
        engine_texts = {f.detail for f in analysis_findings
                        if f.kind == ENGINE_KIND}

        return {
            "valid": len(errors) == 0,
            "spec_name": resolved.spec.name,
            # `compared`: judged against the model's checkpoint; `absent`: a
            # custom model stored without one, so the checks against the
            # file (coverage, bytes, names, packing, tied copies) had
            # nothing to read and every other check gated alone.
            "checkpoint": "compared" if hf_data is not None else "absent",
            # What this verdict judged: the checks it ran, and the ones that
            # compare against a checkpoint it did not have.
            "checks": {"ran": checks.ran, "skipped": checks.skipped,
                       "crashed": checks.crashed},
            "dims": resolved.dim_values,
            # Canonical headline diff — bytes-vs-bytes (mixed-precision safe).
            "model_storage_bytes": storage_block,
            "quantization_metadata_bytes": metadata_block,
            "checkpoint_storage_bytes": all_block,
            # Logical-params diff retained for backward compat with consumers
            # (UI, agent prompt's Phase-4 references) but flagged as a
            # diagnostic — HF side depends on `resolve_scheme` picking one
            # quant_method correctly, so this number is only reliable for
            # single-precision checkpoints.
            "total_params": params_block,
            "spec_blocks": spec_blocks,
            # Which iteration reads which element of each appended stream.
            "stream_pairings": stream_pairings,
            # Who reads a stream whole or its prefix, as a history or
            # joined (`read: all`, `read: concat`, a model-level op).
            "whole_stream_pairings": whole_stream_pairings,
            "unmatched_hf_tensors": unmatched_hf_list,
            "unmatched_hf_params": unmatched_hf_total,
            "unmatched_spec_nodes": unmatched_spec_nodes,
            "param_mismatches": param_mismatches,
            "structural_warnings": [f.as_dict()
                                    for f in checks.structural_warnings],
            "shape_mismatches": [f.as_dict() for f in checks.shape_mismatches],
            "type_contract_warnings": [f.as_dict()
                                       for f in checks.type_contract_warnings],
            "selection_read_errors": [f.as_dict()
                                      for f in checks.selection_read_errors],
            "reuse_warnings": [f.as_dict() for f in checks.reuse_warnings],
            "model_card_warnings": [f.as_dict()
                                    for f in checks.model_card_warnings],
            "cross_layer_warnings": [f.as_dict()
                                     for f in checks.cross_layer_warnings],
            "module_exec_warnings": [f.as_dict()
                                     for f in checks.module_exec_warnings],
            "activation_warnings": [f.as_dict()
                                    for f in checks.activation_warnings],
            "rotary_position_errors": [f.as_dict()
                                       for f in checks.rotary_position_errors],
            "attention_mask_errors": [f.as_dict()
                                      for f in checks.attention_mask_errors],
            "forward_source_warnings": [f.as_dict()
                                        for f in checks.forward_source_warnings],
            "axiom_warnings": [f.as_dict() for f in checks.axiom_warnings],
            "packing_mismatches": [_finding_text(w) for w in checks.packing_errors],
            # Whether the analysis replay ran, and every analysis refusal
            # (dispatch, weight identity, replay) with its kind and node.
            "analysis": analysis,
            "analysis_replay": replay_report,
            "analysis_refusals": [f.as_dict() for f in analysis_findings],
            # Every error is the engine's to resolve: no spec edit can make
            # this spec valid, and the spec-gen loop stops repairing.
            "blocked_by_engine": bool(errors) and all(
                e in engine_texts for e in errors),
            "errors": errors,
        }
    except SpecEvalError as exc:
        return {"valid": False, "errors": [str(exc)]}
    except _INFRASTRUCTURE:
        raise
    except Exception as exc:
        return {"valid": False, "errors": [
            _format_eval_error(exc, spec=spec, stage="evaluate")
        ]}
