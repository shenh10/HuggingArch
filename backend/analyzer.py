"""
Analyzer — Main entry point for model analysis.

Two paths:
  1. Spec exists → pure config + engine computation (no torch, instant)
  2. No spec → return partial result, frontend can trigger spec generation
"""
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
import re

from backend.arch.dtypes import dtype_bytes as dtype_bytes_of
from backend.arch.eval import capacity as capacity_mod
from backend.arch.eval.checkpoint import Reconciliation
from backend.arch.eval.dim import as_number, value_of
from backend.arch.eval.projection.result import ModelReport, NodeReport
from backend.arch.eval.projection.formulas import encoder_item_formulas
from backend.arch.eval.transforms import CallSite
from backend.custom_id import is_custom_id
from backend.arch.loader import SpecLoader, overlay_kwargs, get_default_loader
from backend.hf.metadata import HUB_UNREACHABLE, fetch_config, fetch_metadata
from backend.arch.snapshots import load_model_str as _load_snapshot_model_str
from backend.hf.forward_source import model_info_cache_path
from backend.hf.task_gate import unsupported_task_refusal
from backend.views.composition import ModelComposition, root_kind
from backend.inference.declaration_groups import declaration_key

def _extent_int(value, default=None):
    """Integer extent from a Binding/Dim value; `default` when absent."""
    v = value_of(value)
    return default if v is None else int(v)

logger = logging.getLogger(__name__)


#: The key under which `analyze_model` hands the resolved spec it analysed
#: to the response builder; a payload never carries it.
RESOLVED_KEY = "resolved"


class NoSpecMatch(ValueError):
    """No spec — library, local or generated cache — matches the model: the
    one refusal a spec generation answers. Routes report it as `no_spec`,
    so a page offers generation on this fact, not on the message."""


def analyze_model(model_id: str = "", batch_size: int = 1, seq_len: int = 2048,
                  dtype: str = "float16", trust_remote_code: bool = False,
                  overlay_dir: Optional[str] = None,
                  **kwargs) -> Dict[str, Any]:
    """
    Analyze a HuggingFace model. No torch needed if spec exists.

    `overlay_dir` (a per-user serving overlay, set by the service layer) layers
    a user's PRIVATE generated spec + its draft components on the shared tree,
    so a signed-in user sees their own spec for a not-yet-merged model. Repo
    precedence is preserved by the service endpoint checking the shared loader
    first; None → the shared singleton loader (public / anonymous path).

    `trust_remote_code` defaults to **False**: the only place it matters is
    the no-spec skeleton fetch (`_fetch_model_str` → `extract_model_info`),
    which executes an arbitrary repo's modeling code. The public `/analyze`
    endpoint MUST keep this False — otherwise anyone can RCE the server by
    pointing it at a malicious unsupported model. Trusted callers (the local
    CLI) opt in explicitly.
    """

    # 2. Try to match spec. Pass `model_id` so `custom:<dir>` ids pin
    # to the named registered spec instead of falling into score-based
    # model_type matching (which can collide across agent-generated
    # custom specs sharing a model_type).
    if overlay_dir:
        loader = SpecLoader(**overlay_kwargs(overlay_dir))
    else:
        loader = get_default_loader()
    # The config, checkpoint and print(model) tree all come from the record
    # the spec is served from: the shared snapshot, else the overlay's (a
    # stored record carries the snapshot its spec was judged against).
    overlay_dir = overlay_dir or loader.overlay_for(model_id)
    config = fetch_config(model_id, overlay_dir)
    if config is None:
        return {"success": False, "error": explain_config_fetch_failure(model_id)}
    hf_meta = fetch_metadata(model_id, overlay_dir)
    resolved = loader.match(
        config, model_id=model_id,
        header=hf_meta.get("header"),
    )

    if resolved:
        # Fast path: pure computation from spec (still fetches the print(model)
        # skeleton via _fetch_model_str, hence trust_remote_code flows here).
        return _analyze_from_spec(model_id, config, resolved, hf_meta,
                                  batch_size, seq_len, dtype,
                                  trust_remote_code=trust_remote_code,
                                  overlay_dir=overlay_dir)
    # No spec, no Analysis. The config-only path that used to answer here
    # guessed MLA from `kv_lora_rank` and MoE from `n_routed_experts` — a
    # second alias table the corpus had already retired — and the page had
    # no banner for it.
    # But first: a readable config on an out-of-scope task (a diffusion
    # repo can ship a root config.json) will never match a spec — say WHY
    # instead of offering spec generation for it.
    unsupported = unsupported_task_refusal(model_id)
    if unsupported:
        raise ValueError(unsupported)
    raise NoSpecMatch(
        f"No spec matches {model_id!r}. Add a spec in backend/arch/models/ "
        "or run /generate-spec first."
    )


def classify_config_fetch_failure(model_id: str,
                                  cause: Optional[BaseException] = None) -> tuple[str, str]:
    """Failure-path classifier for `fetch_config` returning None.

    Returns `(kind, message)`. `fetch_config`'s None contract is load-bearing
    (many callers use it as "skip when offline"), so the reason cannot ride
    its return value. Pass `cause` when the caller already has the failed
    download exception so this does not re-probe. Without `cause`, re-probe
    the HF download — the exception type says which failure it was.

    Catch order is load-bearing: GatedRepoError ⊂ RepositoryNotFoundError,
    and LocalEntryNotFoundError ⊂ EntryNotFoundError.

    kind is one of: custom, invalid_id, gated, not_found, no_config,
    unreachable, network, transient, unknown.
    """
    if is_custom_id(model_id):
        return ("custom",
                f"Custom model '{model_id}' is not registered on this server "
                f"— register it first, or check the id.")
    if cause is None:
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=model_id, filename="config.json")
            # The retry succeeded — the first failure was transient.
            return ("transient",
                    f"Could not fetch config for '{model_id}' on the first "
                    f"attempt; a retry just worked, so try again.")
        except Exception as exc:  # classified below by type
            cause = exc
    from huggingface_hub import errors as hf_errors
    if isinstance(cause, hf_errors.HFValidationError):
        return ("invalid_id",
                f"'{model_id}' is not a valid Hugging Face repo id — use "
                f"'owner/name' made of letters, digits, '-', '_' or '.'.")
    if isinstance(cause, hf_errors.GatedRepoError):
        return ("gated",
                f"'{model_id}' is a gated repo on Hugging Face — accept its "
                f"license on the model page and configure an HF token first.")
    if isinstance(cause, hf_errors.RepositoryNotFoundError):
        return ("not_found",
                f"Model '{model_id}' does not exist on Hugging Face — check "
                f"the id for typos. (A private repo looks the same without "
                f"an HF token.)")
    if isinstance(cause, hf_errors.LocalEntryNotFoundError):
        return ("unreachable",
                f"Hugging Face is unreachable from the server and "
                f"'{model_id}' is not in the local cache — retry shortly.")
    if isinstance(cause, hf_errors.EntryNotFoundError):
        # A non-Transformers repo is the common no-config cause — name the
        # hub task when the gate can see it, instead of the generic words.
        precise = unsupported_task_refusal(model_id)
        if precise:
            return ("no_config", precise)
        return ("no_config",
                f"'{model_id}' has no config.json on Hugging Face — it may "
                f"not be a Transformers model repo.")
    if isinstance(cause, (OSError, *HUB_UNREACHABLE)):  # reset, DNS, timeout, …
        return ("network",
                f"Could not reach Hugging Face to fetch '{model_id}' — "
                f"the server's network/proxy looks down; retry shortly.")
    return ("unknown",
            f"Could not fetch config for '{model_id}': "
            f"{type(cause).__name__}: {cause}")


def explain_config_fetch_failure(model_id: str) -> str:
    """User-facing reason `fetch_config` returned None. Same probe and
    wording as `classify_config_fetch_failure`; callers that only display
    the message keep this name."""
    return classify_config_fetch_failure(model_id)[1]



def _fetch_model_str(model_id: str, trust_remote_code: bool = False,
                     overlay_dir: Optional[str] = None) -> str:
    """Best-effort `print(model)` output via meta-tensor skeleton.

    Strategy:
      1. Git snapshot (backend/arch/snapshots/) — offline, authoritative.
      2. User disk cache (~/.cache/huggingarch/model_info/<id>.json) — free.
      3. Build skeleton with init_empty_weights — a few seconds, no GPU.
      4. Fail silently → return "".
    """

    # Custom models don't have HF model classes
    if is_custom_id(model_id):
        return ""

    # 1. Git snapshot
    ms = _load_snapshot_model_str(model_id, overlay_dir)
    if ms:
        return ms

    # 2. User disk cache
    cache = model_info_cache_path(model_id)
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            ms = data.get("model_str", "")
            if ms:
                return ms
        except (OSError, json.JSONDecodeError):
            pass

    # 3. Build skeleton — torch / HF model class loading.
    # Broad `Exception` is intentional: `extract_model_info` runs
    # trust_remote_code modeling files written by arbitrary model
    # authors (DeepSeek, GLM, MiMo, …). We've seen ValueError (unknown
    # model_type), AttributeError (`ChatGLMConfig` accessing missing
    # attrs), KeyError (custom code's "default"), RuntimeError, OSError —
    # the surface is genuinely unbounded. "" is the documented fallback
    # ("skeleton unavailable, agent must work from config"); narrowing
    # would require chasing each new model release's failure mode.
    try:
        # Lazy: model_info runs `sys.path.insert(0, repo_root)` at module body,
        # a process-global side effect that must stay on the skeleton-fetch path
        # rather than firing for every analyzer/config importer.
        from backend.hf.model_info import extract_model_info
        info = extract_model_info(model_id, trust_remote_code=trust_remote_code)
        ms = info.get("model_str", "")
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        except OSError:
            pass
        return ms
    except Exception as e:
        logger.debug(f"model_str unavailable for {model_id}: {e}")
        return ""




def _analyze_from_spec(model_id: str, config: dict, resolved, hf_meta: dict,
                       batch_size: int, seq_len: int, dtype: str,
                       trust_remote_code: bool = False,
                       overlay_dir: Optional[str] = None) -> Dict[str, Any]:
    """Full analysis from spec — no torch for the numbers, but the optional
    print(model) skeleton (`_fetch_model_str`) can execute trust_remote_code;
    kept False for anonymous /analyze."""

    # Unstated dims (Qb, Ob, …) are filled once in `resolve_dimensions`.

    # Lazy: `project_model` is a PEP 562 lazy re-export from the projection
    # package; binding it here (not at module top) keeps `transforms`/`layout`
    # off every analyzer importer's import path — they load only on this
    # spec-analysis path.
    from backend.arch.eval.projection import project_model
    model_result = project_model(
        resolved, batch=batch_size, input_tokens=seq_len, checkpoint=True,
    )

    # Headline sizes: the dims the spec declares at model level.
    analysis = model_result.analysis
    program = analysis.program
    headline = program.headline
    H, V, L = headline.hidden, headline.vocab, headline.layers
    I = headline.intermediate

    # The routing numbers, as this walk bound them (None for dense).
    moe = analysis.moe

    # Attention / FFN kinds of the backbone's layer groups, from this
    # invocation's IR + eval.cache.
    layer_groups_meta = []
    for template in program.stack_blocks():
        call = template.attention_call()
        layer_groups_meta.append({
            "count": template.count,
            # Stored layers it holds (a looped stack's later pass and its
            # gate hold none) and the pass it runs in.
            "stored_count": template.stored_count,
            "pass": template.meta.pass_,
            "template": template.name,
            "attn_type": call.kv_label if call is not None else None,
            "attn_display": (
                (call.attn_variant or call.kv_label) if call is not None else None),
            "ffn_type": template.ffn_type(),
            # The layers as the checkpoint numbers them (stored layers).
            "layer_indices": template.meta.layer_numbers,
        })
    ffn_type = next((cfg.get("ffn_type") for cfg in layer_groups_meta
                     if cfg.get("ffn_type")), None)
    if any(cfg.get("ffn_type") == "moe" for cfg in layer_groups_meta):
        ffn_type = "moe"
    num_moe_layers = sum(cfg["stored_count"] for cfg in layer_groups_meta
                         if cfg.get("ffn_type") == "moe")
    # The backbone layers that run a dense FFN, alone or beside their
    # experts (LongCat runs one in every MoE layer): the one published fact
    # (`Headline.dense_layers`), not a count of blocks whose primary FFN
    # type is dense.
    num_dense_layers = len(headline.dense_layers)
    # Layers whose block makes an attention / sequence-mixer call (its cache
    # identity, `BlockTemplate.attention_call`): Nemotron's MoE layers make
    # none.
    num_attention_layers = sum(cfg["stored_count"]
                               for cfg in layer_groups_meta
                               if cfg.get("attn_type"))

    total_flops = model_result.total_flops
    attn_flops = sum(c * br.attention_flops for _, c, br in model_result.block_results)
    ffn_flops = sum(c * br.ffn_flops for _, c, br in model_result.block_results)

    # The weights' formats, from the one weight join (`Analysis.tensors()`):
    # FP4 / INT4 where the checkpoint packs them into I8 / U8 / I32
    # containers — the container is storage, not the format.
    weight_formats = sorted(analysis.weight_bytes_by_dtype().items(),
                            key=lambda kv: -kv[1])
    weight_dtype = analysis.dominant_weight_dtype()
    dtype_bytes = analysis.dominant_weight_dtype_bytes()

    # Two byte figures, for two questions. `param_bytes` is the parameters
    # alone, as the FILE holds them (`checkpoint_tensors`, saved formats):
    # it is what the checkpoint header's parameter bytes are compared
    # against below (`storage_diff`), not a second source for the headline. The weight total is everything stored
    # — parameters, their quantisation scales and buffers — read off the
    # weight ledger the drawn nodes' `weight_bytes` come from, so the
    # headline is the sum of the per-op figures on the same page.
    param_bytes = model_result.analysis.checkpoint_param_bytes()
    weights = capacity_mod.weights(model_result.analysis)
    total_weight_bytes = int(weights.resident)

    # Build per-component quant-metadata breakdown (so the UI can show how it's computed)
    quant_meta_breakdown = _collect_quant_meta_breakdown(model_result)
    scale_dtype = analysis.dominant_weight_scale_dtype()
    meta_formats = [{"dtype": name, "bytes": int(n)}
                    for name, n in sorted(
                        analysis.metadata_bytes_by_dtype().items(),
                        key=lambda kv: -kv[1]) if n]

    # Split total/active params into main backbone vs auxiliary groups (MTP, …).
    aux_params = dict(model_result.aux_params)
    aux_active_params = dict(model_result.aux_active_params)
    aux_flops = dict(model_result.aux_flops)
    aux_prefill_active = dict(model_result.aux_prefill_active_params)
    main_flops = total_flops - sum(aux_flops.values())

    summary = {
        "model_id": model_id,
        "model_type": resolved.model_type,
        # Every stack the checkpoint holds: main model, MTP heads, encoders.
        "total_params": model_result.total_params,
        "total_active_params": model_result.total_active_params,
        # The main model alone (no MTP, no encoder): the headline every page
        # leads with, the figure a model card and the gallery state.
        "main_total_params": model_result.main_params,
        "main_active_params": model_result.main_active_params,
        # Per prompt token in prefill (long prompt): only the layers that
        # must run on every position. `*_active_params` is per decoded token.
        "total_decode_active_params": model_result.total_decode_active_params,
        "total_prefill_active_params": model_result.total_prefill_active_params,
        "prefill_full_from": model_result.prefill_full_from,
        "main_prefill_active_params": model_result.main_prefill_active_params,
        "aux_prefill_active_params": aux_prefill_active,
        # Beside the main-model headline, so the main stack's tables only.
        "lookup_table_params": model_result.main_lookup_table_params,
        "main_flops": main_flops,
        "aux_params": aux_params,                         # per-group breakdown
        "aux_active_params": aux_active_params,
        "aux_flops": aux_flops,
        "quant_metadata_params": model_result.quant_metadata_params,
        "quant_metadata_breakdown": quant_meta_breakdown,
        # The scales' most common dtype, for the precision label.
        "quant_metadata_scale_dtype": scale_dtype,
        # The metadata's stored bytes, each tensor at its own width (the
        # one join, `iter_checkpoint`): the total, and per dtype, largest
        # first — a checkpoint can mix 1-byte ue8m0 and 4-byte F32 scales.
        "quant_metadata_bytes": sum(row["bytes"] for row in meta_formats),
        "quant_metadata_formats": meta_formats,
        # Deprecated, kept for existing API clients; the Inspector does not
        # read it. There is no single per-scale width: read
        # `quant_metadata_formats`. It is the width of the metadata's one
        # dtype when it has exactly one, and null when the dtypes are mixed.
        "quant_metadata_bytes_per_scale": (
            dtype_bytes_of(meta_formats[0]["dtype"])
            if len(meta_formats) == 1 else None),
        "param_bytes": param_bytes,
        "param_size_gb": param_bytes / (1024 ** 3),
        # Tied weights the checkpoint also stores as a copy of their own
        # (HunyuanOCR's lm_head beside its embedding): on disk, so in the
        # checkpoint size and `param_bytes`, and skipped at load, so not in
        # `weight_bytes`.
        "checkpoint_copies": [
            {"tensor": row.name,
             "bytes": int(value_of(row.storage_bytes * row.count))}
            for row in analysis.checkpoint_copy_rows()],
        "weight_bytes": total_weight_bytes,
        "dtype_bytes": dtype_bytes,
        "weight_dtype": weight_dtype,
        # Every weight format with its stored bytes, largest first.
        "weight_formats": [{"dtype": name, "bytes": int(n)}
                           for name, n in weight_formats if n],
        # The quant methods covering the weights, largest first: which
        # scheme the model mostly is (the quant badge), by the same join.
        "weight_methods": [{"method": name, "bytes": int(n)}
                           for name, n in sorted(
                               analysis.weight_bytes_by_method().items(),
                               key=lambda kv: -kv[1]) if n],
        "total_flops": total_flops,
        "total_flops_gflops": total_flops / 1e9,
        "attention_flops": attn_flops,
        "attention_flops_gflops": attn_flops / 1e9,
        "ffn_flops": ffn_flops,
        "ffn_flops_gflops": ffn_flops / 1e9,
        "num_layers": L,
        "num_attention_layers": num_attention_layers,
        "num_dense_ffn_layers": num_dense_layers,
        "num_moe_layers": num_moe_layers,
        "hidden_size": H,
        "intermediate_size": I,
        "vocab_size": V,
        # Structure-faithful embedding split from the engine's per-node
        # accounting — NOT `vocab_size * hidden_size`, which assumes a single
        # untied [V,H] table and misreads tied weights, resized/padded vocabs,
        # and multi-embedding models (e.g. MTP `mtp.embed_tokens`).
        "embedding_params": model_result.embedding_params,
        # The rest of the main model: the breakdown under the headline sums
        # to `main_total_params`, not to the checkpoint (MTP heads and
        # encoders are their own rows).
        "non_embedding_params": (model_result.main_params
                                 - model_result.embedding_params
                                 - model_result.main_lookup_table_params),
        "attention_info": {
            **_build_attention_info(headline, layer_groups_meta, L,
                                    program.is_hybrid_attention),
            # Each family's geometry: the dims its attention call reads, at
            # the values the walk bound there.
            "families": [family.as_dict()
                         for family in analysis.attention_families],
        },
        # Whether the model routes tokens to experts — its graph declares a
        # dispatch zone; the template's `ffn_type` reads the same zone.
        "moe_enabled": program.has_dispatch_zone,
        "ffn_info": {
            "ffn_type": ffn_type,
            "intermediate_size": I,
            "num_experts": moe.n_experts if moe else 0,
            "top_k": moe.n_experts_active if moe else 0,
            "num_shared_experts": moe.n_shared_experts if moe else 0,
            "expert_intermediate_size": (
                moe.d_ffn_expert if moe and moe.d_ffn_expert else None),
        },
    }

    # Use already-fetched HF metadata for cross-check.
    # Bytes-vs-bytes is the only unit whose accounting holds for mixed-
    # precision checkpoints (FP4 routed experts + FP8 attention + BF16
    # norms in one model). `packed_unpack_factor` on the HF side picks
    # one quant_method globally, so HF-derived "logical params"
    # (`hf_params` / `hf_checkpoint_params_total` / `quant_metadata_params`)
    # are unreliable for vendors that split methods across config fields
    # (DSv4's `expert_dtype` outside `quantization_config`). bytes-vs-
    # bytes sidesteps this entirely: spec walks `quant_context.per_role`
    # per-leaf via materialize, HF sums per-tensor `numel × dtype_bytes`
    # from safetensors headers. The legacy logical-params HF-side fields
    # are intentionally NOT exposed in summary anymore — keeping them
    # would silently mislead consumers and the only honest descriptor at
    # this layer is bytes.
    hf_total_params = hf_meta.get("total_params", 0)
    if hf_total_params > 0:
        summary["hf_model_storage_bytes"] = hf_meta.get("model_storage_bytes", 0)
        summary["hf_quantization_metadata_bytes"] = hf_meta.get("quantization_metadata_bytes", 0)
        summary["hf_checkpoint_storage_bytes"] = hf_meta.get("checkpoint_storage_bytes", 0)
        summary["hf_header"] = hf_meta.get("header")

        # Spec storage bytes via materialize. Mirrors `validate`'s
        # storage-bytes path (orchestrator.py).
        actual_storage_bytes = summary.get("hf_model_storage_bytes", 0)
        # The header files a serialised buffer with the weights (it is no
        # quant sidecar), so the spec's side carries its declared buffers.
        spec_storage_bytes = (
            param_bytes + model_result.analysis.checkpoint_buffer_bytes())
        summary["spec_storage_bytes"] = spec_storage_bytes
        if actual_storage_bytes > 0:
            storage_diff = actual_storage_bytes - spec_storage_bytes
            summary["storage_diff"] = storage_diff
            summary["storage_diff_pct"] = (
                storage_diff / actual_storage_bytes * 100
            )
        else:
            summary["storage_diff"] = 0
            summary["storage_diff_pct"] = 0.0

        # Secondary checksums on the per-tensor join (validate's primary
        # gate is unmatched tensors + per-pattern element agreement).
        # `storage_diff` stays the weight column; metadata and all-tensor
        # columns catch a wrong scale width that weights cannot see.
        recon = Reconciliation.of(model_result.analysis, hf_meta, model_id)
        spec_all = int(recon.total)
        hf_meta_bytes = int(recon.header_metadata_bytes)
        hf_all = int(recon.header_total)
        # The UI reads `actual − spec`; Reconciliation's deltas are
        # `spec − header`, so each column is its negation.
        meta_diff = -int(recon.metadata_delta)
        all_diff = -int(recon.delta)
        summary["spec_metadata_bytes"] = int(recon.metadata_bytes)
        summary["spec_checkpoint_bytes"] = spec_all
        summary["metadata_storage_diff"] = meta_diff
        summary["metadata_storage_diff_pct"] = (
            meta_diff / hf_meta_bytes * 100 if hf_meta_bytes else 0.0
        )
        summary["all_tensor_diff"] = all_diff
        summary["all_tensor_diff_pct"] = (
            all_diff / hf_all * 100 if hf_all else 0.0
        )
        summary["checkpoint_anchored"] = recon.is_anchored
        if recon.measured_total is not None:
            measured = int(recon.measured_total)
            measured_diff = -int(recon.measured_delta)
            summary["hf_index_total_size"] = measured
            summary["measured_tensor_diff"] = measured_diff
            summary["measured_tensor_diff_pct"] = (
                measured_diff / measured * 100 if measured else 0.0
            )

    # Build layer groups from spec
    layer_groups = _build_layer_groups(model_result, resolved, batch_size, seq_len)
    representative = _representative_template(model_result)
    composition = ModelComposition(model_result, weights).payload()

    # Weights with and without the encoders: an encoder's are its own
    # accounting, and a surface that totals the language model (Compare)
    # shows the language model's unless asked to include the towers.
    encoders = {stack.name for stack in program.stacks if stack.is_encoder}
    language = composition["totals"]["language_model"]
    summary["encoder_params"] = {
        name: n for name, n in aux_params.items() if name in encoders}
    summary["encoder_active_params"] = {
        name: n for name, n in aux_active_params.items() if name in encoders}
    # Each encoder's modality, as its stack declares it: the key a Compare
    # card groups encoders by.
    summary["encoder_stacks"] = {
        stack.name: stack.aux_kind for stack in program.stacks
        if stack.is_encoder}
    # An encoder's FLOPs are per media item, in its own encode phase: an
    # expression in the item's axes, which this page binds no size to. The
    # step's aux FLOPs are the language model's aux stacks (MTP) alone.
    summary["aux_flops"] = {
        name: n for name, n in aux_flops.items() if name not in encoders}
    summary["encoder_flops_per_item"] = encoder_item_formulas(resolved)
    summary["weight_totals"] = {
        "lm": {
            "weight_bytes": language["storage_bytes"],
            "param_bytes": analysis.checkpoint_param_bytes(
                without_encoders=True),
            "storage_by_dtype": language["storage_by_dtype"],
        },
        "with_encoders": {
            "weight_bytes": total_weight_bytes,
            "param_bytes": param_bytes,
            "storage_by_dtype": composition["totals"]["storage_by_dtype"],
        },
    }

    # Surface the spec's own "this spec is partial" marker to callers.
    # When the spec-gen agent hits the Phase 4 escape hatch, it writes a
    # `# UNRESOLVED:` comment block at the top of the YAML; the loader
    # picks it up into `spec.unresolved_note`. The UI shows this as a
    # warning banner so users know the spec is partial / needs review.
    summary["spec_unresolved_note"] = resolved.spec.unresolved_note

    return {
        "success": True,
        "model_id": model_id,
        "summary": summary,
        "layer_groups": layer_groups,
        "representative_template": representative,
        "composition": composition,
        "config": config,
        "raw_config_json": config,
        "model_str": _fetch_model_str(model_id, trust_remote_code=trust_remote_code,
                                      overlay_dir=overlay_dir),
        "spec_matched": True,
        "spec_name": resolved.spec.name,
        "spec_unresolved_note": summary["spec_unresolved_note"],
        # The spec this analysis ran on, for the response builder's views
        # (compute graph, formulas, labels): one match per request. Not
        # part of the payload (`RESOLVED_KEY`).
        RESOLVED_KEY: resolved,
    }


def _build_attention_info(headline, layer_groups_meta: list, L: int,
                          is_hybrid: bool) -> dict:
    """Build the `summary.attention_info` dict.

    Adds two extra fields when the model has heterogeneous attention
    across layers (Qwen3-Next: 3× linear-attn + 1× full-attn rotation;
    GPT-OSS: alternating SWA / full):

      `layer_types`             — per-layer attn_type strings (length L)
      `full_attention_interval` — repeat period of the rare attn type

    Display comes from `Cache.variant or label` per `BlockMeta`; hybrid
    `layer_types` from `BlockMeta.layer_indices`. Whether the model is
    hybrid is `is_hybrid` (`Program.is_hybrid_attention`), the verdict the
    badge reads; the display labels only order the line.
    """
    # Headline attention type from the canonical display label (`attn_display`
    # = attn_variant or kv_label, so DSA reads as DSA not plain MLA). When the
    # backbone MIXES flavors across layers (gpt-oss SWA ⊕ GQA; GLM DSA-indexer
    # ⊕ shared-MLA), the distinct labels compose in first-appearance order
    # ("DSA + MLA"). A label line, not the hybrid verdict — that is
    # `Program.is_hybrid_attention`, the backbone's mixers asked once.
    _seen: list = []
    for cfg in layer_groups_meta:
        _d = cfg.get("attn_display") or cfg.get("attn_type")
        if _d and _d not in _seen:
            _seen.append(_d)
    _display_type = " + ".join(_seen)
    # The cache families (`cache.family`, the closed vocabulary of
    # `arch/cache_families.yaml`) the backbone's attention runs, in the same
    # first-appearance order: what a page keys a style on, where the display
    # line above is only for reading.
    _families: list = []
    for cfg in layer_groups_meta:
        _f = cfg.get("attn_type")
        if _f and _f not in _families:
            _families.append(_f)
    info = {
        # As the labels spell it (GatedDeltaNet, Mamba2): every page shows
        # this string, and no page re-cases it.
        "attention_type": _display_type or None,
        "attention_families": _families,
        "num_heads": headline.heads,
        "num_kv_heads": headline.kv_heads,
        "head_dim": headline.head_dim,
        "kv_lora_rank": headline.kv_lora_rank,
        "q_lora_rank": headline.q_lora_rank,
        "qk_rope_dim": headline.qk_rope_dim,
    }

    # A hybrid backbone (`Program.is_hybrid_attention`, the verdict the
    # badge reads) gets a per-layer breakdown: gpt-oss
    # SWA ⊕ GQA, GLM DSA-indexer ⊕ shared-MLA, Qwen3-Next linear ⊕ full.
    # A layer whose display label is missing withholds the breakdown below
    # rather than answering with a second verdict.
    if not is_hybrid:
        return info

    layer_types: list = [None] * L
    for cfg in layer_groups_meta:
        # Per stored layer: a looped stack's later pass and its gate hold no
        # layer (`stored_count` 0) and would overwrite the layer's own.
        if not cfg["stored_count"]:
            continue
        attn = cfg.get("attn_display")
        for idx in cfg.get("layer_indices") or ():
            if 0 <= idx < L:
                layer_types[idx] = attn

    if any(t is None for t in layer_types):
        return info

    info["layer_types"] = layer_types

    # Compute the cycle interval as the gap between successive
    # appearances of the rare attn type. Robust to either ordering
    # (rare-first or rare-last layer).
    from collections import Counter
    counts = Counter(layer_types)
    if len(counts) >= 2:
        rare_type = counts.most_common()[-1][0]
        rare_indices = [i for i, t in enumerate(layer_types) if t == rare_type]
        if len(rare_indices) >= 2:
            info["full_attention_interval"] = rare_indices[1] - rare_indices[0]

    return info


def _scale_declarations(rep) -> list:
    """One entry per metadata sibling declared by `rep.saved` (fp8 scale_inv,
    zero point, …): the symbolic axes with the format's config bindings
    substituted (`block_h` → 128; `in`/`out` stay symbolic), plus the resolved
    shape and dtype. `leaf.params_symbolic` is the WEIGHT's formula; this is
    the scale tensor's own."""
    saved = rep.saved
    if not saved.metadata_geometry or len(saved.metadata_geometry) != len(saved.metadata):
        return []
    out = []
    for meta, geo in zip(saved.metadata, saved.metadata_geometry, strict=True):
        out.append({
            "suffix": geo.suffix,
            "kind": meta.sibling_kind,
            "axes": ([_substitute_bindings(axis, saved.bindings)
                      for axis in geo.axes] if geo.axes else None),
            "shape": [as_number(dim) for dim in meta.shape],
            "dtype": meta.dtype,
        })
    return out


def _substitute_bindings(formula: str, bindings) -> str:
    """Numeric config constants in a metadata axis formula read as their
    values — `(out + block_h - 1) // block_h` → `(out + 128 - 1) // 128`."""
    for key in sorted(bindings, key=len, reverse=True):
        value = bindings[key]
        if not isinstance(value, (int, float)):
            continue
        formula = re.sub(rf"\b{re.escape(key)}\b", str(value), formula)
    return formula


def _collect_quant_meta_breakdown(model_result) -> list:
    """Walk the per-block node tree and surface every quant-metadata leaf (fp8 scale_inv etc.)
    with its formula and resolved param count — so the UI can show how the total is derived."""
    items: list = []
    seen_templates: set = set()

    # uid -> Representation, from the one join (`Analysis.tensors()`), so a
    # leaf can quote its scale's own declaration. A node stores several
    # tensors; the format carrying `metadata_geometry` is the one wanted.
    # Built once up front — template dedup below must not hide a uid.
    reps_by_uid: Dict[int, Any] = {}
    for row in model_result.analysis.tensors():
        if row.rep is None:
            reps_by_uid.setdefault(row.uid, None)
            continue
        current = reps_by_uid.get(row.uid)
        if current is None or (not current.saved.metadata_geometry
                               and row.rep.saved.metadata_geometry):
            reps_by_uid[row.uid] = row.rep

    def _walk(node: NodeReport, path: str):
        if node.quant_metadata_params == 0:
            return
        if node.children:
            for child in node.children:
                _walk(child, f"{path}.{child.name}" if path else child.name)
            return
        # leaf quant-meta node
        leaf = {
            "path": path,
            "op": node.op,
            "params": int(node.quant_metadata_params),
            "params_symbolic": node.params_symbolic,
            "params_values": node.params_values,
        }
        # The scale tensor's own formula and shape. Absent when the leaf has
        # no format declaration to quote — e.g. gpt-oss, whose scales are
        # explicit graph nodes rather than format metadata.
        ir_node = node.ir_node
        rep = reps_by_uid.get(ir_node.uid) if ir_node is not None else None
        if rep is not None:
            scales = _scale_declarations(rep)
            if scales:
                leaf["scales"] = scales
        items.append(leaf)

    for template_name, count, br in model_result.block_results:
        key = (template_name,)
        if key in seen_templates or br.quant_metadata_params == 0:
            continue
        seen_templates.add(key)
        block_items: list = []
        for top_node in br.nodes:
            before = len(items)
            _walk(top_node, top_node.name)
            block_items.extend(items[before:])
        if block_items:
            items = items[:-len(block_items)] if block_items else items
            items.append({
                "template": template_name,
                "layer_count": int(count),
                "per_layer_params": int(br.quant_metadata_params),
                "total_params": int(count * br.quant_metadata_params),
                "leaves": block_items,
            })
    return items


def _representative_template(model_result: ModelReport) -> Optional[Dict[str, Any]]:
    """The main-path block_results item with the largest `count`, and the
    basis it was picked on, so Compare can state it.

    Main path is `aux_kind is None`. `basis` is `most_layers` for a unique
    winner. A tie names no block (`name` None, `basis` `tie`) and lists the
    tied blocks, so Compare cannot invent a representative. `call` is the
    block's stage call (`CallSite.stage_call`): Playground's module
    comparison evaluates the block there. `dims` is what the block's
    frame binds (`BlockReport.dims`). Analyzer computes this once;
    `/analyze` and `/compare` only read the published field.
    """
    aux_kinds = list(model_result.block_aux_kinds or [])
    derived = list(model_result.block_derived_groups or [])
    # One entry per declaration: the groups a `read: all` stack splits one
    # declaration into (Kimi-K3's `block_kda_moe`, 61 layers in 8 history
    # lengths) are one block, as the composition table counts them. Its
    # bucket is its largest group's.
    by_decl: Dict[Any, List[Any]] = {}
    for i, (tmpl_name, count, _br) in enumerate(model_result.block_results):
        aux = aux_kinds[i] if i < len(aux_kinds) else None
        if aux is not None:
            continue
        group = derived[i] if i < len(derived) else None
        key = declaration_key(group) if group is not None else i
        entry = by_decl.setdefault(key, [i, tmpl_name, 0, -1])
        if int(count) > entry[3]:
            entry[0], entry[3] = i, int(count)
        entry[2] += int(count)
    main: List[Tuple[int, str, int]] = [
        (i, name, total) for i, name, total, _largest in by_decl.values()]
    if not main:
        return None
    max_count = max(c for _i, _n, c in main)
    winners = [
        {"name": name, "bucket_id": i, "call": CallSite.stage_call(i),
         # The values its frame binds: what a compare overlay states a
         # differing dim by.
         "dims": model_result.block_results[i][2].dims}
        for i, name, c in main if c == max_count
    ]
    n_layers = sum(c for _i, _n, c in main)
    if len(winners) != 1:
        return {"name": None, "basis": "tie", "count": max_count,
                "n_layers": n_layers, "tied": winners}
    return {**winners[0], "basis": "most_layers", "count": max_count,
            "n_layers": n_layers}


def _build_layer_groups(model_result: ModelReport, _resolved, batch_size, seq_len) -> list:
    """Build layer_groups from spec engine results.

    `total_memory_bytes` is each group's own checkpoint bytes
    (`BlockReport.total_storage_bytes` / `NodeReport.storage_bytes`): a
    block's tensors at their own widths, so an MXFP4 expert block and a
    BF16 attention block are not both priced at the model's average.
    """

    groups = []

    analysis = model_result.analysis
    H = analysis.program.headline.hidden

    def _root_group(node, layer_type: str, block_index: int, name: str) -> dict:
        """A model-root node rendered exactly like a transformer block: every
        figure read off the node the engine actually evaluated (params / flops /
        real shapes), not reconstructed from config dims."""
        return {
            "name": name,
            "layer_type": layer_type,
            "count": 1,
            "total_params": node.params,
            "total_active_params": node.active_params,
            # `as_number` because a node's flops is EXACT and exact is not
            # always whole — a dispatch-zone leaf carries the fractional `M`
            # average as a `Fraction`, which `jsonable_encoder` refuses. A
            # model root is never in a dispatch zone, so this reads `int` in
            # practice; it is written this way so it stays true if one ever is.
            "total_flops": as_number(node.flops),
            "total_memory_bytes": node.storage_bytes,
            "block_index": block_index,
            "input_shape": node.shape_in,
            "output_shape": node.shape_out,
            "sub_layers": [],
            "attention_params": 0,
            "ffn_params": 0,
        }

    # Model-root nodes that precede the blocks (embedding tables). Emitted from
    # `model_level_nodes` — projected from the same Analysis — so a
    # model with no embedding simply yields no card, and one with several
    # (e.g. MTP `mtp.embed_tokens`) yields one each.
    root_nodes = model_result.model_level_nodes or {}
    for name, node in root_nodes.items():
        kind, label = root_kind(name, node)
        if kind == "embedding":
            # The main table gets the familiar label; secondary embeddings
            # (MiMo `speech_embeddings`, MTP `mtp.embed_tokens`) keep their own
            # spec name so they aren't all mislabelled "Token Embedding".
            # Which one is main is the spec's `boundary_role` to say. It used
            # to be `name == "embed_tokens"`, so the nine models that call
            # theirs something else — `backbone.embeddings`, `word_embeddings`,
            # `transformer.embedding.word_embeddings`, DSv4's `embed` — showed
            # a raw spec path where every other model showed a label.
            groups.append(_root_group(node, kind, -1, label))

    # Transformer blocks — one layer_groups entry per layer, tagged with
    # the engine's block_results INDEX as `bucket_id`. Two layers share
    # bucket_id iff the engine emitted them from the same block_results
    # entry, i.e. same template AND same dim_overrides — the unambiguous
    # definition of structural identity. Template name alone is NOT
    # sufficient (an `assignments:` spec can reuse a template with
    # different `dims:` overrides, producing different computed shapes).
    # Layer position (block_index) is the layer the invocation covers in its
    # own stack (`ModelReport.layer_positions`), so vision / text don't share
    # a numbering.
    stack_names = list(model_result.block_stack_names) if model_result.block_stack_names else []
    aux_kinds = list(model_result.block_aux_kinds) if model_result.block_aux_kinds else []
    derived_groups = list(model_result.block_derived_groups)
    positions = model_result.layer_positions()
    passes = model_result.layer_passes()
    binding = analysis.binding
    B = (_extent_int(binding.get("B"), batch_size) if binding is not None
         else batch_size)
    S = (_extent_int(binding.get("s_ctx"), seq_len) if binding is not None
         else seq_len)
    block_shape = [B, S, H] if H else None

    for i, (tmpl_name, _count, br) in enumerate(model_result.block_results):
        stack_name = stack_names[i] if i < len(stack_names) else None
        aux_kind = aux_kinds[i] if i < len(aux_kinds) else None
        layer_positions = positions[i]
        first_shape = br.nodes[0].shape_in if br.nodes and br.nodes[0].shape_in else block_shape
        last_shape = br.nodes[-1].shape_out if br.nodes and br.nodes[-1].shape_out else block_shape
        pass_ = passes[i]
        for pos in layer_positions:
            bucket_id = i
            groups.append({
                "name": (f"Loop gate {pos + 1}" if pass_ == 0
                         else f"Transformer Block {pos}"
                         + (f" · pass {pass_}" if pass_ > 1 else "")),
                "layer_type": "transformer_block",
                "count": 1,
                "total_params": br.total_params,
                "total_active_params": br.total_active_params,
                "total_flops": br.total_flops,
                "total_memory_bytes": br.total_storage_bytes,
                "block_index": pos,
                "bucket_id": bucket_id,
                "stack_name": stack_name,
                "aux_kind": aux_kind,
                "template": tmpl_name,
                "input_shape": first_shape,
                "output_shape": last_shape,
                "sub_layers": [],
                "attention_params": br.attention_params,
                "attention_flops": br.attention_flops,
                "ffn_params": br.ffn_params,
                "ffn_flops": br.ffn_flops,
                # Everything in the block that is neither attention nor FFN:
                # norms, residual adds, routers. Published here so the page
                # reads it rather than subtracting and clamping at zero.
                "other_flops": br.total_flops - br.attention_flops - br.ffn_flops,
                # A derived group's declaration and history length, so the
                # page shows its layers under the declaration they share.
                **({"derived_group": derived_groups[i]}
                   if derived_groups[i] is not None else {}),
            })

    # Model-root nodes that follow the blocks: the final norm and the lm_head
    # projection. Same treatment as the embedding above — read off the engine's
    # nodes, so the final norm (previously invisible to the UI despite carrying
    # real params + flops) now renders alongside the head.
    for name, node in root_nodes.items():
        kind, label = root_kind(name, node)
        if kind == "embedding":
            continue                      # emitted before the blocks
        if kind == "lm_head":
            groups.append(_root_group(node, kind, -2, label))
        elif node.params or node.flops:
            # Any other model-root leaf that carries weight or compute (final
            # norm, root parameter tensors) — no special-casing, and it keeps
            # its spec name so a new root op shows up honestly instead of being
            # dropped. The block-stack entry is inherently excluded: it is a
            # 0-param / 0-flop placeholder whose real figures live in
            # `block_results`, already emitted above.
            groups.append(_root_group(node, kind, -2, label))

    return groups
