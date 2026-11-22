"""Pydantic request/response models + MODALITY_REGISTRY.

Kept as plain data so any route module can import them without a
circular dependency on the FastAPI `app`. New endpoints add their
request model here, then declare a route in `routes/<domain>.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union, get_args

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from backend.arch import dims as _dims
from backend.arch.eval.precision import (PRESETS, PrecisionError,
                                         PrecisionProfile)
from backend.arch.schema.model import ModelLayer
from backend.arch.spec_migrate import symbol_renames
from backend.inference.calibration_overlay import looks_up
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK
from backend.inference.defaults import COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT
from backend.measure.scope import NONE as NO_MEASUREMENTS, MeasurementScope
from backend.inference.frontier import CloudPhase, LineAxis, ViewForm
from backend.inference.sweep_grid import ROUTINE_SEQ_LENS
from backend.inference.parallel_sweep import ALL_MODES, CloudDim, _resolve_modes
from backend.playground import entry_key


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

#: The precision knobs a precision profile replaced. A request carrying any
#: of them is refused by name — never translated: a width does not say a
#: format, and a deployment-wide KV dtype does not say which slots.
PRECISION_FIELDS_RETIRED = (
    "act_dtype_bytes", "act_dtype_per_role", "kv_dtype",
    "kv_quant_granularity", "kv_scale_dtype", "weight_dtype_bytes",
    "weight_quant_granularity", "weight_scale_dtype",
    "weight_dtype_per_role", "compute_dtype_bytes")


def refuse_retired_precision(data: Any, *, takes_precision: bool) -> Any:
    """Refuse a retired precision field by name, with the field to send."""
    if not isinstance(data, dict):
        return data
    named = [k for k in PRECISION_FIELDS_RETIRED if k in data]
    if not takes_precision and "precision" in data:
        named.append("precision")
    if named:
        fix = (f"send `precision`: {{\"preset\": <one of {sorted(PRESETS)}>}} "
               f"or {{\"canonical\": <profile from POST "
               f"/precision/canonicalize>}}" if takes_precision else
               "this endpoint always shows every model as shipped; drop them")
        what = ("retired with precision profiles" if takes_precision
                else "not taken here")
        raise ValueError(
            f"{', '.join(f'`{k}`' for k in named)} {what}: {fix}")
    return data


def canonical_precision(value: Optional[Dict[str, Any]]
                        ) -> Optional[Dict[str, Any]]:
    """A request's `precision`, validated and written in its one spelling
    (`{"canonical": ...}`): a preset and the rules it stands for are the
    same profile and the same cache key."""
    if value is None:
        return None
    try:
        profile = PrecisionProfile.from_request(value)
    except PrecisionError as exc:
        raise ValueError("precision: " + "; ".join(
            e["message"] for e in exc.errors)) from None
    return {"canonical": profile.canonical()}


def precision_from_query(text: Optional[str]) -> PrecisionProfile:
    """A GET endpoint's `precision` query parameter: the same body a POST
    carries, as JSON text (`{"preset": "kv_fp8"}`); absent is as shipped.
    A malformed or refused profile is a ValueError with the reasons."""
    if not text:
        return PrecisionProfile.from_request(None)
    import json
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"precision is JSON ({exc.msg})") from None
    try:
        return PrecisionProfile.from_request(body)
    except PrecisionError as exc:
        raise ValueError("precision: " + "; ".join(
            e["message"] for e in exc.errors)) from None


class PrecisionRequest(BaseModel):

    """The `precision` field every costing request carries: a precision
    profile as `{"preset": <id>}` or `{"canonical": <profile>}`; absent is
    as shipped."""
    precision: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _retired_precision_fields(cls, data: Any) -> Any:
        return refuse_retired_precision(data, takes_precision=True)

    @field_validator("precision")
    @classmethod
    def _canonical_precision(cls, v):
        return canonical_precision(v)

    def precision_profile(self) -> PrecisionProfile:
        return PrecisionProfile.from_request(self.precision)

class PrecisionEffectiveRequest(PrecisionRequest):
    """`/precision/effective`: one model under one precision profile."""
    model_id: str


class AnalyzeRequest(PrecisionRequest):
    model_id: str = ""
    batch_size: int = 1
    seq_len: int = 1
    dtype: str = "float16"
    # Draw `block_graph` for the model rather than for this batch and length:
    # workload axes stay symbolic in the shape labels, the KV marks carry no
    # per-request / per-step figure, and the cache is priced at the dtype the
    # model states (`build_block_graph_data(workload_symbolic=...)`). The
    # Inspector asks for it.
    symbolic_shapes: bool = False


class ComposeGroupEdit(BaseModel):
    """One layer group's edit. `group_id` is the group's position in the
    skeleton's `groups[]`; `template_name` / `count` / `assignment_index`
    echo what the skeleton said about it, so an edit made against a base
    that has since changed is refused instead of landing on another group.
    `slots`: slot path (`node/node::param`) -> library component. `dims`:
    the group's own dim rebindings."""
    group_id: int
    template_name: str
    count: int
    assignment_index: Optional[int]
    slots: Dict[str, str] = {}
    dims: Dict[str, int] = {}


class ComposeModelRequest(PrecisionRequest):
    """Playground Phase 2a: whole-model composition, canonical whole-model
    cost. Public, no auth; no persistence (that is Phase 2b).

    `groups` edits layer groups, one explicit entry per group;
    `dim_overrides` are global. Empty `groups` is a no-edit compose."""
    base_model_id: str
    groups: List[ComposeGroupEdit]
    dim_overrides: Dict[str, int] = {}
    batch_size: int = 1
    seq_len: int = 2048
    #: GPU whose roofline times each phase. Editor compose stays unsharded;
    #: this only selects the hardware ceiling, same as module-bench.
    #: Omitted: resource columns only, no SOL / measured time.
    gpu_sku: Optional[str] = None
    #: Context lengths to report whole-model KV storage at (the Scaling
    #: sweep's `SWEEP_SEQ_LENS`). Empty: no sweep, same default as module-bench.
    kv_cache_lens: List[int] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _one_request_shape(self) -> "ComposeModelRequest":
        if any(n < 1 for n in self.kv_cache_lens):
            raise ValueError("kv_cache_lens must be positive")
        ids = [g.group_id for g in self.groups]
        if len(set(ids)) != len(ids):
            raise ValueError("one entry per group_id")
        return self


class ModuleBenchEntry(BaseModel):
    """One compare entry: a library component, optionally bound to a shipped
    model (the dims seed + provenance label) plus the user's own dim overrides.
    The same component may appear several times with different bindings — that
    is the component × bound-model comparison.

    `seeded` tells the backend the client already carries the binding's seed in
    `dims` (echoed back after the first bench), so a dim the user then deleted
    stays deleted instead of being silently re-seeded.

    `label` overrides the entry identity used for results and the echo —
    non-model entries (a saved custom variant) need a key that is not
    `name@model_id`.

    `call` names the call of the entry's model (or of the base it is dropped
    into) the entry is evaluated at — one of the `calls` every row and
    refusal lists. A model that calls the component once needs none; one
    that calls it more than once, or a base that does not call it, is
    refused until the entry names one."""
    name: str
    model_id: Optional[str] = None
    call: Optional[str] = None
    dims: Dict[str, int] = {}
    seeded: bool = False
    label: Optional[str] = None


class ModuleBenchRequest(PrecisionRequest):
    """Playground semantics B: isolate each named library component under one
    shared contract and return its canonical flops / bytes / kvcache. Public,
    no auth — reuses the same eval path as `/analyze`."""
    #: Optional drop-in host. Compare UI omits it: each entry benches in the
    #: model that binds it. Tests and the editor still pass a host.
    base_model_id: Optional[str] = None
    #: Legacy flat pick list — expanded into unbound entries when `entries` is
    #: empty. New clients send `entries`.
    components: List[str] = []
    #: The compare set. Entries with a `model_id` but missing dims are seeded
    #: inline from that model's binding (explicit `dims` always win).
    entries: List[ModuleBenchEntry] = []
    batch_size: int = 1
    seq_len: int = 2048
    #: Optional hidden-size override for the shared contract. Every other
    #: internal dim defaults to the base model's resolved value and can be
    #: overridden explicitly via `dim_overrides`; a dim a component needs that
    #: the base lacks surfaces as a per-component error, never a silent default.
    d_model: Optional[int] = None
    dim_overrides: Dict[str, int] = {}
    #: Per-component internal-dim overrides (component name → {dim: value}),
    #: merged over `dim_overrides` for that component only — so one module bound
    #: by several models can be compared at each model's real parameterization,
    #: seeded from `/api/playground/component-dims` and then edited. d_model
    #: stays shared (the drop-in interface). Legacy companion of `components`;
    #: `entries[].dims` is the per-entry form.
    per_component_dims: Dict[str, Dict[str, int]] = {}
    #: GPU whose roofline (and measured-corpus rows) time each phase. Module
    #: comparison itself stays unsharded; this only selects the hardware ceiling.
    gpu_sku: str = "B200"
    #: Context lengths to report each entry's KV storage at (the Scaling
    #: sweep's `SWEEP_SEQ_LENS`), priced through each cached tensor's own law.
    kv_cache_lens: List[int] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _nonempty(self) -> "ModuleBenchRequest":
        if any(n < 1 for n in self.kv_cache_lens):
            raise ValueError("kv_cache_lens must be positive")
        if self.entries and self.components:
            raise ValueError("send entries OR components, not both "
                             "(entries win silently is not a contract)")
        if not self.entries and not self.components:
            raise ValueError("entries or components must be a non-empty list")
        # Dedupe on the EFFECTIVE identity (label, or the bench's own entry
        # key: name@model_id#call), not the raw fields — a label that spells
        # another entry's key must collide with it, not pass silently, and
        # two calls of one model's block are two columns.
        keys = [e.label or entry_key(e.name, e.model_id, e.call)
                for e in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate entries (same effective key)")
        for e in self.entries:
            if e.label is not None and (not e.label.strip() or "@" in e.label):
                raise ValueError("entry labels must be non-empty and must not contain '@'")
        if len(self.entries) > 8 or len(self.components) > 8:
            raise ValueError("at most 8 entries per bench")  # each is a full eval
        return self


class ModuleLatencySweepRequest(ModuleBenchRequest):
    """The bench's entries and contract, timed in both phases at each of
    `lens` (prefill: a prompt of that many tokens; decode: one step against
    that many cached positions) instead of at `seq_len` alone."""
    lens: List[int] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _positive_lens(self) -> "ModuleLatencySweepRequest":
        if any(n < 1 for n in self.lens):
            raise ValueError("lens must be positive")
        return self


#: Why a request naming the old verify window is refused, in every place
#: that accepts speculative decoding: the estimate body and the GET
#: endpoints' query.
SPECULATIVE_WINDOW_RETIRED = (
    "`speculative_window` is retired: send "
    "`speculative_draft_tokens` = K, the number of draft tokens "
    "per step (the old window minus 1; the verify covers K + 1 "
    "positions)")


#: The most rows a request may carry in `calibration_rows` (422 beyond).
CALIBRATION_ROWS_MAX = 20_000

def measurement_only(default: Any, **kw: Any) -> Any:
    """A request field that decides only how the measured track is read
    (which rows, which selection), never the theory: a sweep's theory half is
    keyed without it (`measure.scope.theory_of_request` reads the mark)."""
    return Field(default, json_schema_extra={"measurement_only": True}, **kw)


class InferenceEstimateRequest(PrecisionRequest):
    model_id: str = ""
    # weight_bytes_total: if omitted, the endpoint fetches it via hf_metadata
    # (snapshot or HF).
    weight_bytes_total: Optional[float] = None
    gpu_sku: str
    dcp_selected_load_factor: float = Field(1.0, ge=1.0, allow_inf_nan=False)
    dcp_failed_owners: int = Field(0, ge=0)
    gpu_memory_fraction: float = 0.9
    # The GPU count is the parallel config's rank count. Optional: a stated
    # count that disagrees with the mesh is refused, never priced.
    num_gpus: Optional[int] = None
    avg_input_tokens: int = 1024
    avg_output_tokens: int = 1024
    # ParallelConfig knobs (all default to 1 = no parallelism)
    parallel: Optional[dict] = None
    # Workload knobs — surface prefix caching + speculative decoding
    # to the inference page. Defaults reproduce plain prefill /
    # autoregressive decode (no prefix, K=0, α=1).
    prefix_cached_tokens: int = 0
    # K: draft tokens per decode step. The target verifies them in one
    # forward over K + 1 positions and keeps 1 + α·K tokens per sequence
    # (`inference/phases.Speculation`). 0 = plain autoregressive decode.
    speculative_draft_tokens: int = Field(0, ge=0)
    # α ∈ (0, 1], as the gallery route declares it; only meaningful when K > 0
    acceptance_rate: float = Field(1.0, gt=0.0, le=1.0)
    # Count the draft's own forwards in the decode step, priced off the
    # model's MTP stack (`inference/speculative_draft.py`). None = the verify
    # forward alone. "recursive" (EAGLE): K one-token forwards;
    # "parallel" (DFlash): one forward over K tokens. A model with no
    # MTP stack is refused with the reason.
    speculative_draft: Optional[Literal["recursive", "parallel"]] = None
    # Aux module weight accounting. The language model's aux groups (MTP,
    # tagged `aux_kind` in the spec) are dropped from per_op_timing
    # FLOPs by default (text-only serving anchor). This toggle aligns
    # the weight breakdown to the same default: False = backbone-only
    # weight (matches the FLOPs path + published backbone-only figures);
    # True = reserve the aux module weights in HBM too. This toggle is
    # residency only; the draft head's per-step reads are counted by
    # `speculative_draft`, not by this. An encoder (vision / audio) is in
    # neither position: its weights are the encode phase's own accounting.
    include_aux_weights: bool = False
    # Keep the weights the spec declares a framework can hold in host memory
    # (`host_offload`: DeepSeek-V4.1's Engram and Qwen's PLE n-gram tables)
    # there rather than on the GPU, as vLLM does by default (Engram
    # `cpu_offload`, `VLLM_PLE_CPU_OFFLOAD=1`). They then leave the per-GPU
    # weights and the cache gets their room. Residency only; a model that
    # declares none is unaffected.
    host_offload: bool = True
    # One link traversal's latency (µs; a collective pays it once per hop
    # its algorithm waits on, `comm_volume.ALGORITHMS`) and the shortest one
    # kernel launch takes (µs): the two time floors a small step pays
    # whatever it moves. Estimates until Kernel Bench
    # measures them; the defaults are `inference/defaults.py`'s.
    comm_latency_us: float = Field(COMM_LATENCY_US_DEFAULT, ge=0.0, le=1000.0)
    launch_floor_us: float = Field(LAUNCH_FLOOR_US_DEFAULT, ge=0.0, le=1000.0)
    # The media each request carries, by model input name — the axes of the
    # input the config does not fix (`Workload.media`):
    # `{"pixel_values": {"n_img": 1, "img_h": 672, "img_w": 672}}`, sides
    # the processed size (`/inference/media-inputs` lists each input's axes
    # and the resize factor a side is a multiple of). Only the encode phase
    # reads it: the response's `encode` prices each encoder over it, and no
    # prefill / decode number moves with it. An input not named carries no
    # item. An input carrying items of different sizes gives a list of
    # groups, one size each (`[{"n_img": 1, ...}, {"n_img": 2, ...}]`).
    media: Dict[str, Union[Dict[str, int], List[Dict[str, int]]]] = {}
    # Requests in one encode step; None is the prefill batch.
    encode_batch_size: Optional[int] = Field(None, ge=1)
    # Per-phase batch overrides. None → backend defaults: prefill=1
    # (one-shot serving anchor), decode=max_decode_batch (KV-bound).
    # Threaded into estimate_phases + the per-phase calibration overlay
    # anchor so per-op rows reflect the requested batch.
    prefill_batch_size: Optional[int] = None
    decode_batch_size: Optional[int] = None
    # Chunked-prefill token budget (vLLM `max_num_batched_tokens`,
    # SGLang `chunked_prefill_size`). Bounds the prefill activation
    # footprint to min(seq, chunk) so `max_prefill_batch` plateaus past
    # the chunk instead of decaying ∝ seq. Defaults to 8192 — the modern
    # serving default (vLLM V1 / SGLang). Set to 0 / null for un-chunked
    # (one-shot full-seq prefill).
    prefill_chunk_size: Optional[int] = 8192
    # Calibration preference — single toggle. `True` = measured where the
    # corpus has any kernel for an op's physical cell (the fastest of them,
    # `envelope`), theoretical roofline elsewhere.
    # `False` = pure theoretical roofline floor.
    # Internal `selection_mode` is derived from this; legacy callers
    # passing `selection_mode` directly still work via back-compat
    # translation in `_back_compat_selection_mode`.
    prefer_measured: bool = measurement_only(True)
    # Explicit lookup mode: `latest` (the binding's exact declared provider
    # implementation at its latest measured release), `envelope` (fastest
    # kernel across libraries), `theoretical_only`, or `kernel_mix`. The retired
    # `framework_version` / `best_of` / `measured` spellings read as `latest`.
    # New clients leave this empty and use `prefer_measured`.
    selection_mode: Optional[str] = measurement_only(None)
    # The framework whose binding the estimate runs under; None → the default
    # framework (`DEFAULT_FRAMEWORK`). Bindings are not versioned, so a
    # request carrying the retired `framework_version` field has it ignored.
    framework: Optional[str] = None
    # Server options set on the framework, e.g.
    # `{"enable_fused_qk_norm_rope": true}` for sglang or
    # `{"linear_backend": "auto"}` for vLLM. An option left unset keeps the
    # framework's default; a binding entry's `when.server_flag` reads it.
    framework_options: Optional[Dict[str, Union[bool, str, int]]] = None
    # The paged-KV page the deployment allocates the cache in. A DEPLOYMENT
    # choice, not a model fact: a paged attention kernel's time depends on
    # how many pages a query gathers and how coalesced each gather is, so it
    # is part of the key of every kernel that reads the cache, and the
    # estimate looks rows up at the page it is deploying with.
    #
    # None takes the framework binding's own default (vLLM 16). Which values
    # a backend can actually serve is the backend's constraint (FlashInfer
    # [16,32,64], FlashMLA [64], CUTLASS MLA [128]); a page nothing measured
    # simply finds no row, which is the honest answer rather than a
    # substituted one. `/inference/frameworks` surfaces each binding's
    # default so a client can show it.
    kv_block_size: Optional[int] = Field(None, gt=0, le=4096)
    # Debug-only kernel override map. Lets a power user pin
    # `{"flash_attention": "fa_v3_quirky"}` to force a specific
    # `kernel_impl` in the lookup tuple, or `"library:impl_id"` when the
    # implementation id alone is shared across provider libraries. Not a
    # production knob —
    # normal users leave it empty and let the framework binding's
    # `op_implementations` drive kernel choice.
    kernel_mix: Optional[dict] = measurement_only(None)
    # Calibrate with the caller's own measured rows too: the server resolves
    # the caller's session to their user id (the owner of the scope below)
    # before anything is computed — 401 without a valid session, 501 when
    # no service layer is installed. The client never names an owner.
    calibrate_with_mine: bool = measurement_only(False)
    # Request-supplied measured calibration rows (each a dict keyed by the
    # CSV columns), for headless callers; the core re-validates and overlays
    # them on the looked-up rows for THIS request only. Bounded here: a
    # corpus belongs in Kernel Bench (`calibrate_with_mine`), not in a body.
    calibration_rows: Optional[list] = measurement_only(
        None, max_length=CALIBRATION_ROWS_MAX)
    # Internal: what the measured numbers are read under
    # (`measure.scope.MeasurementScope`: the owner, the lake, its snapshot,
    # the annotation rules), set by the server before the request is
    # offloaded, joined or keyed (`_access.with_measurement_scope`). Not
    # JSON: excluded from every dump (the cache keys add its `key`
    # themselves) and refused when a body spells it.
    measurement_scope: SkipJsonSchema[Optional[MeasurementScope]] = Field(None, exclude=True)
    # Disaggregated serving: an optional SECOND (decode) deployment. When
    # present, prefill/TTFT + prefill throughput come from THIS request's
    # top-level deployment (the prefill/shared deployment), while decode/TPOT
    # and KV capacity (max_tokens/max_batch) come from a second
    # `estimate_capacity` run driven by this block. A partial dict of
    # `{parallel, gpu_sku, num_gpus}`. An omitted `gpu_sku` falls back to the
    # top-level request; `parallel` is the decode mesh whole, and its GPU
    # count is that mesh's rank count (a stated `num_gpus` is only checked
    # against it, and the prefill's stated count does not carry over).
    # Absent → single deployment (byte-identical to today).
    decode: Optional[dict] = None
    # Disagg KV-handoff link bandwidth (GB/s) for the prefill→decode KV cache
    # transfer. Only used when `decode` is set. None → default to the decode
    # GPU's cross-node RDMA bandwidth (`gpu.rdma_gbps`), the realistic
    # cross-node disaggregation link; pass a value to model NVLink (same node)
    # or a slower fabric.
    kv_transfer_bw_gbps: Optional[float] = None

    def resolved_selection_mode(self) -> str:
        """Translate the user-facing `prefer_measured` toggle (plus
        any debug `selection_mode` override + `kernel_mix` presence)
        into the internal mode string the capacity layer understands.

        Resolution rule:
          1. Explicit `selection_mode` wins (back-compat for callers
             that still pass the legacy enum).
          2. `kernel_mix` populated → "kernel_mix" (caller obviously
             wants kernel pinning).
          3. `prefer_measured=True` → "envelope" (the fastest measured
             kernel of any library at each physical op).
          4. Default → "theoretical_only".
        """
        if self.selection_mode:
            return self.selection_mode
        if self.kernel_mix:
            return "kernel_mix"
        return "envelope" if self.prefer_measured else "theoretical_only"

    def reads_measurements(self) -> bool:
        """Whether this request's answer reads any measurement: its mode looks
        measurements up (`calibration_overlay.looks_up`). A sweep asked for
        its SOL alone does not (`SweepRequest.sol_only`)."""
        return looks_up(self.resolved_selection_mode())

    @model_validator(mode="before")
    @classmethod
    def _measurement_scope_is_the_servers(cls, data: Any) -> Any:
        """`measurement_scope` is set by the server (the owner from the
        caller's verified session); a body naming one (another owner, a
        snapshot) is refused, never read."""
        if isinstance(data, dict) and "measurement_scope" in data:
            raise ValueError("measurement_scope is set by the server; send "
                             "calibrate_with_mine instead")
        return data

    def scope(self) -> MeasurementScope:
        """The scope this request reads under (`NONE` before the server set
        one: a caller that bypasses the API reads no measurements)."""
        return self.measurement_scope or NO_MEASUREMENTS

    @model_validator(mode="before")
    @classmethod
    def _speculative_window_is_retired(cls, data: Any) -> Any:
        """`speculative_window` was the verify window (draft tokens + 1). It
        is refused by name rather than ignored, so a caller still sending it
        does not silently get autoregressive decode."""
        if isinstance(data, dict) and "speculative_window" in data:
            raise ValueError(SPECULATIVE_WINDOW_RETIRED)
        return data

    @model_validator(mode="after")
    def _kernel_mix_only_in_kernel_mix_mode(self) -> "InferenceEstimateRequest":
        """`kernel_mix` and `selection_mode="kernel_mix"` go together.
        If `selection_mode` was explicitly set to something else but
        `kernel_mix` is populated, reject — the user's intent is
        ambiguous.

        Bare `kernel_mix` (no explicit selection_mode) is fine — the
        resolver translates it to `kernel_mix` mode automatically.
        """
        if (
            self.kernel_mix
            and self.selection_mode is not None
            and self.selection_mode != "kernel_mix"
        ):
            raise ValueError(
                f"kernel_mix is only valid when selection_mode='kernel_mix' "
                f"(got selection_mode={self.selection_mode!r}). "
                f"Drop the explicit selection_mode OR drop kernel_mix."
            )
        return self


class ProcessedMediaRequest(BaseModel):
    """Raw images to the estimate's processed `media`: per image input, its
    items axis and the raw `height` / `width` in pixels
    (`{"pixel_values": {"n_img": 1, "height": 1080, "width": 1920}}`)."""
    model_id: str
    raw: Dict[str, Dict[str, int]]


class ParallelShapeRequest(BaseModel):
    """A deployment as the Inference page edits it (`ParallelConfig` fields),
    and the model it runs: a mesh region the model has no node in (the
    expert region of a dense model) spans no ranks."""
    parallel: Dict[str, Any] = {}
    model_id: Optional[str] = None


class BatchPoint(BaseModel):
    """One (prefill, decode) batch point of a sweep's batch axis."""
    key: str
    prefill_batch_size: Optional[int] = None
    decode_batch_size: Optional[int] = None


class SweepRequest(InferenceEstimateRequest):
    """A Scaling sweep's request: the SOL answer and the measured answer of
    the same body are two requests (docs/auto_sweep.md §7.5).
    `sol_only` asks for the SOL answer alone — the sweep's theory half
    assembled, no measurement read, so no Kernel Bench state reaches it; it
    is cached on the theory's identity, and its cells still state their rows'
    keys when the mode looks measurements up, so the measured request that
    follows reads the stored theory half and makes only the lookup and the
    fold."""
    sol_only: bool = measurement_only(False)

    def reads_measurements(self) -> bool:
        return not self.sol_only and super().reads_measurements()


class InferenceEstimateSweepRequest(SweepRequest):
    """Multi-seq variant of `InferenceEstimateRequest`. `seq_lens` is
    the sweep axis (one call per element instead of one HTTP request
    per element). `avg_output_tokens` stays at request scope — held
    constant across the sweep. The inherited `avg_input_tokens` field
    is ignored by the route; clients should leave it at its default.
    """
    seq_lens: List[int]
    # Optional BATCH sweep axis. Each entry {key, prefill_batch_size,
    # decode_batch_size} (sentinels as the scalar fields: None=auto, -1=cap,
    # N=pinned). The preparation is batch-invariant, so the (seq × batch) grid
    # amortises ONE preparation — no per-batch HTTP fanout. Response is then keyed
    # by batch: {"results_by_batch": {<key>: [<per-seq>…]}}.
    batch_grid: Optional[List[BatchPoint]] = None
    # Optional parallel PRESET: {"mode": "dp_ep"|"cp_ep"|"tp_ep"|"tp_tp",
    # "degree": N, "ep_load_factor": ρ (optional, default 1.0)} — resolved
    # server-side by `config_for_degree` so the frontend never carries its own
    # copy of the mode table. Ignored when `parallel` is set explicitly.
    parallel_preset: Optional[Dict[str, Any]] = None

    @field_validator("seq_lens")
    @classmethod
    def _seq_lens_non_empty_non_negative(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("seq_lens must be non-empty")
        if any(s < 0 for s in v):
            raise ValueError("seq_lens entries must be ≥ 0")
        return v

    @field_validator("parallel_preset")
    @classmethod
    def _preset_load_factor_canonical(
            cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The preset is part of the cache key, so its ρ is stored in one
        spelling: a float ≥ 1, and absent when it is 1.0 — an omitted ρ and an
        explicit 1.0 are the same deployment and share one entry."""
        if not v or "ep_load_factor" not in v:
            return v
        rho = float(v["ep_load_factor"])
        if not rho >= 1.0:
            raise ValueError(
                f"parallel_preset.ep_load_factor must be ≥ 1.0 (max/mean "
                f"expert load), got {v['ep_load_factor']!r}")
        out = {k: x for k, x in v.items() if k != "ep_load_factor"}
        if rho != 1.0:
            out["ep_load_factor"] = rho
        return out


class InferenceParallelSweepRequest(SweepRequest):
    """Deployment-scale sweep for the Scaling panel: hold model + context +
    batch fixed, sweep the parallelism DEGREE (EP/TP) — one series per GPU
    type — reporting per-GPU throughput. The inherited `parallel` / `gpu_sku`
    / `num_gpus` are ignored (each point builds its own config); `gpu_skus`
    is the GPU series.

    `mode` is the legacy single-axis form. `modes` batches several axes
    (DP-EP / CP-EP / TP-EP / TP) into one pool pass. At least one of the
    two is required; if both are set, `modes` wins.
    """
    mode: Optional[str] = None                 # dp_ep | cp_ep | tp_ep | tp_tp
    modes: Optional[List[str]] = None
    #: The prompt lengths every layout is priced at (one walk per card
    #: class, the rest by substitution): the throughput cloud's `context`
    #: coordinate and each chart point's context.
    contexts: List[int]
    batch: Optional[int] = None                # per-GPU batch; None = each config's max
    gpu_skus: Optional[List[str]] = None       # GPU series; None = curated default
    gpu_sku: str = ""                          # inherited-but-ignored (each point sets its own)
    # Expert-routing imbalance (max/mean expert load, ≥ 1) every point runs
    # under. `parallel` is per-point and ignored, so the one deployment field
    # no topology determines is carried here; it is not per-point, so it is
    # part of the sweep's cache key.
    ep_load_factor: float = Field(1.0, ge=1.0)

    @field_validator("contexts")
    @classmethod
    def _contexts_on_the_routine_grid(cls, v: List[int]) -> List[int]:
        """A subset of the routine context grid (`ROUTINE_SEQ_LENS`): the
        work a request asks is bounded, and every context is one a canonical
        serving record carries (`serving_projection.servingProjectionContexts`),
        so the sweep can project rather than walk."""
        if not v:
            raise ValueError("contexts must be non-empty")
        off = [c for c in v if c not in ROUTINE_SEQ_LENS]
        if off:
            raise ValueError(f"contexts must be on the routine grid "
                             f"{ROUTINE_SEQ_LENS}; got {off}")
        if len(set(v)) != len(v):
            raise ValueError("contexts must not repeat a length")
        return v

    @model_validator(mode="after")
    def _modes_resolved(self) -> "InferenceParallelSweepRequest":
        try:
            self.modes = _resolve_modes(self.mode, self.modes)
        except ValueError as e:
            raise ValueError(str(e)) from e
        unknown = [m for m in self.modes if m not in ALL_MODES]
        if unknown:
            raise ValueError(f"mode must be one of {ALL_MODES}, got {unknown[0]!r}")
        return self


class ParallelFrontierRequest(BaseModel):
    """A view of a parallel sweep's throughput point cloud
    (`parallel_sweep.cloud_view`): the sweep as the Scaling panel posts it
    (its answer's cache entry — a `sol_only` body reads the SOL answer, a
    measured one the measured answer), the phase drawn, the dimensions held
    to a set of values, the ones each drawn line is one value of, and the
    form — each line its Pareto frontier (`frontier`), or its points along
    `x` (`line`: GPU count, per-GPU decode batch or context). A frontier
    may also ask for the envelope over coarser groups of its lines
    (`envelope_by`: each card's front over all its deployments, drawn over
    the deployments' own lines).

    Which (phase, form) pairs the cloud can actually answer is the cloud's
    own answer, published with every view (`frontier.view_forms`); this
    model validates only the vocabulary and the form/x pairing."""
    sweep: InferenceParallelSweepRequest
    phase: CloudPhase = "decode"
    fix: Dict[CloudDim, List[Union[int, str]]] = Field(default_factory=dict)
    group_by: List[CloudDim] = Field(default_factory=lambda: ["gpu_sku"])
    form: ViewForm = "frontier"
    x: Optional[LineAxis] = None
    envelope_by: Optional[List[CloudDim]] = None

    @model_validator(mode="after")
    def _x_matches_form(self) -> "ParallelFrontierRequest":
        if self.form == "line" and self.x is None:
            raise ValueError("a line view names its X axis (x)")
        if self.form == "frontier" and self.x is not None:
            raise ValueError("a frontier view has no X axis; drop x")
        if self.form == "line" and self.envelope_by is not None:
            raise ValueError("an envelope is a frontier's; a line view has "
                             "none — drop envelope_by")
        return self


class CompareFrontierRequest(BaseModel):
    """The Compare page's decode Pareto fronts: its rows' estimate sweeps,
    as the page posts them (and so their answers), in row order."""
    sweeps: List[InferenceEstimateSweepRequest] = Field(min_length=1, max_length=16)


class SweepShapesRequest(PrecisionRequest):
    """Export the FULL Scaling-panel sweep as measurable shapes for one
    (model, gpu): every deployment cell the panel plots — the routine context
    grid × batch grid × parallel modes/degrees — VRAM-gated on that GPU,
    expanded to the deduped per-op job list. This is the single source both the
    Scaling panel's export button and kernel_bench consume, so the measured
    corpus covers exactly what the panel shows (no drift). Thin wrapper over
    `kernel_bench.sweep.build_sweep_jobs`.
    """
    model_id: str
    gpu: str = "RTX-4090"
    seq_lens: Optional[List[int]] = None   # default → the routine 9-point grid
    #: Routing skews to plan beside the track grid. `1.0` is the balanced
    #: routine corpus; a value above it also plans the busiest routed rank's
    #: launch geometry (`Workload.ep_load_factor`).
    ep_load_factors: List[float] = Field(default_factory=lambda: [1.0])
    framework: str = DEFAULT_FRAMEWORK

    @field_validator("model_id")
    @classmethod
    def _model_id_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_id required")
        return v.strip()

    @field_validator("ep_load_factors")
    @classmethod
    def _ep_load_factors_ge_one(cls, v: List[float]) -> List[float]:
        if not v or any(rho < 1 for rho in v):
            raise ValueError("ep_load_factors must be non-empty and >= 1.0")
        return v


class BenchPlanJobsRequest(PrecisionRequest):
    """Export ONE step as Kernel Bench jobs — the step the Inference page is
    showing, as the same `jobs.jsonl` `hbench plan` writes.

    Every field here is an input of `kernel_bench.plan.build_jobs_over_points`
    under the name that builder reads it by; the route passes them and adds
    nothing. The workload fields are the estimate's own
    (`/inference/compute-graph` takes the same four), so the shapes a job is
    keyed at are the shapes the per-op table showed: the prompt, how much of
    it the step found cached, the drafted tokens the verify forward covers
    (K → `s_verify = K + 1`, `phases.Speculation`) and the batch the phase
    ran at.

    `phase="encode"` plans the encoders over `media` instead, each in its own
    encode step; a point's prompt, prefix and verify width say nothing there.
    """
    model_id: str
    phase: Literal["prefill", "decode", "encode"] = "prefill"
    #: The deployment whole (`ParallelConfig.from_dict` + `cp_scheme`), not a
    #: flattened TP degree: it decides the per-RANK shape of every job.
    parallel: Optional[Dict[str, Any]] = None
    framework: str = DEFAULT_FRAMEWORK
    #: The card the displayed estimate ran on. A queue plan is a box plan:
    #: framework dispatch can make its dtype environment card-dependent.
    gpu_sku: str
    batch_size: int = Field(1, ge=1)
    avg_input_tokens: int = Field(1, ge=1)
    avg_output_tokens: int = Field(0, ge=0)
    prefix_cached_tokens: int = Field(0, ge=0)
    speculative_draft_tokens: int = Field(0, ge=0)
    #: The media each request carries in an encode step, by model input name
    #: (`Workload.media`) — the estimate's own `encode.media`.
    media: Dict[str, Dict[str, int]] = {}

    @field_validator("model_id")
    @classmethod
    def _model_id_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_id required")
        return v.strip()

    @field_validator("gpu_sku")
    @classmethod
    def _gpu_sku_required(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("gpu_sku required")
        return v.strip()


class CompareRequest(BaseModel):
    # 2..5 models compared side by side. (Was a fixed model_a/model_b pair.)
    models: List[str]
    batch_size: int = 1
    seq_len: int = 1
    # Each model's graph drawn for the model (`AnalyzeRequest.symbolic_shapes`).
    symbolic_shapes: bool = False

    @model_validator(mode="before")
    @classmethod
    def _retired_precision_fields(cls, data: Any) -> Any:
        """Compare shows every model as shipped: no precision field."""
        return refuse_retired_precision(data, takes_precision=False)

    @field_validator("models")
    @classmethod
    def _check_models(cls, v):
        cleaned = [m.strip() for m in (v or []) if m and m.strip()]
        if not (2 <= len(cleaned) <= 5):
            raise ValueError("compare requires between 2 and 5 models")
        return cleaned


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    spec_jobs_running: int = 0
    # Reports route modules register (`_app.health_detail`), by name; a
    # failing one is listed with its error and never fails the check.
    detail: dict = {}


class GenerateSpecRequest(BaseModel):
    model_id: str
    # Continue the draft an earlier failed generation left, instead of
    # starting over (`spec_worker.prepare_resume`).
    resume: bool = False
    # Start over even though an unfinished run kept a draft; without it such
    # a request is refused with 409 `draft_kept`.
    discard_draft: bool = False


class StopSpecRequest(BaseModel):
    """POST /api/spec/stop (and the body shape the frontend sends). `kind`
    is a closed vocabulary so a refine stop cannot be silently treated as
    generate. Review/fix jobs are keyed by `record_id`, not model_id."""
    model_id: str = ""
    kind: Literal["generate", "refine", "review", "fix"] = "generate"
    record_id: Optional[str] = None


class RefineSpecRequest(BaseModel):
    """Drives /refine-spec — sent by the frontend when an analyze pass
    found a non-zero spec-vs-actual diff worth fixing.

    Diff is ALWAYS in storage bytes (not logical params). Logical-params
    diff is misleading on mixed-precision checkpoints (FP4 + FP8 in one
    model, e.g. DeepSeek-V4-Pro): HF-side `packed_unpack_factor` picks
    one quant_method globally and silently misroutes the others. The
    `validate` orchestrator and the `/analyze` dashboard agree on bytes
    as the canonical comparison axis; this request body matches.
    """
    model_id: str
    spec_storage_bytes: int
    hf_storage_bytes: int
    storage_diff: int


class ProposeOpRequest(BaseModel):
    """Drives /propose-op — kicks off the op-proposal worker on a gap
    statement previously dropped by the spec worker.

    Either provide `gap_path` (path to a .gap.json under
    `logs/op_proposals/`, repo-relative or absolute) OR `gap` inline.
    `op_name` is the proposed snake_case identifier — used as the
    per-op concurrency key and the proposal artifact directory name.

    `chain_review`: when True (the default), the SSE stream continues
    past the proposer's `done` into the op_workers.review pipeline
    (reviewer agent → rule check → optional auto-merge). When False, the
    stream stops after the proposer finishes — a human picks up from
    there.
    """
    op_name: str
    gap_path: Optional[str] = None
    gap: Optional[Dict] = None
    chain_review: bool = True


class ReviewOpRequest(BaseModel):
    """Drives /review-op — runs the op-review worker on an artifact
    directory the proposer already wrote.

    `op_name` is the artifact directory name under
    `backend/arch/ops_proposals/<op_name>/`.

    `auto_merge`: when True (the default), if the reviewer's verdict is
    `auto_merge` AND the mechanical rule gate passes, the route calls
    `backend.arch.promote_op_proposal.run_promote` to graft +
    pytest + revert-on-fail. When False, the route writes review.json
    and stops — useful for dry-run / debugging.
    """
    op_name: str
    auto_merge: bool = True


# ---------------------------------------------------------------------------
# Modality registry
#
# Maps an encoder's modality — the `aux_kind` its stack declares; which
# stacks are encoders is the spec's structure (`ArchSpec.encoder_groups`) —
# to its display metadata: the Compare tab's "Modality" cards, one per
# modality any compared model encodes. The dims listed below drive the
# side-by-side compare diff.
#
# The labels are display and are declared nowhere else; the two sets of
# NAMES are not, so `_checked_modalities` resolves both against their
# declarations at import. Listing them here instead of asking had already
# drifted twice: a `speech` modality no spec can declare (`aux_kind` has no
# such word), whose `Is` row would have shown the shared-expert FFN width
# under "Speech FFN dim", and audio `Mela` / `Stra` rows naming dims the
# registry no longer has, which `_dim_get` answered None for on every
# model — a card row that silently never rendered.
# ---------------------------------------------------------------------------
def _checked_modalities(entries: Dict[str, dict]) -> Dict[str, dict]:
    """`entries`, each keyed by a declared modality and naming real dims.

    A modality is one of `ModelLayer.aux_kind`'s words (the loader refuses
    a spec writing another), and a row's key is a registered dim symbol —
    after the renames `_dim_get` follows (`spec_migrate.symbol_renames`),
    since that is the lookup the card actually makes.
    """
    modalities = {k for k in get_args(get_args(
        ModelLayer.model_fields["aux_kind"].annotation)[0])
        if isinstance(k, str)}
    renames = symbol_renames()
    for modality, meta in entries.items():
        if modality not in modalities:
            raise ValueError(
                f"modality {modality!r} is not an `aux_kind` any stack can "
                f"declare ({sorted(modalities)}), so no compared model can "
                f"ever show its card")
        unknown = sorted(key for key, _label in meta["dims"]
                         if not _dims.is_registered(renames.get(key, key)))
        if unknown:
            raise ValueError(
                f"the {modality!r} card names dim(s) {unknown}, which "
                f"backend/arch/dims.yaml does not register; a row keyed by "
                f"one reads None on every model")
    return entries


MODALITY_REGISTRY = _checked_modalities({
    "vision": {
        "label": "Vision Tower",
        "icon": "vision",
        "dims": [
            ("n_vis_layers",  "Vision layers"),
            ("d_vis_model",  "Vision hidden"),
            ("n_vis_heads",  "Vision heads"),
            ("d_vis_head", "Vision head dim"),
            ("d_vis_ffn",  "Vision FFN dim"),
            ("s_patch",  "Patch size"),
            ("PeH", "Pos-embed H"),
            ("PeW", "Pos-embed W"),
            ("d_proj_hidden",  "Projector mm_hidden"),
            ("n_spatial_merge",  "Spatial merge"),
            ("d_proj_in",  "Projector inner dim"),
        ],
    },
    "audio": {
        "label": "Audio Tower",
        "icon": "audio",
        "dims": [
            ("n_audio_layers",  "Audio layers"),
            ("d_audio_model",  "Audio hidden"),
            ("n_audio_heads",  "Audio heads"),
            ("d_audio_head", "Audio head dim"),
            ("d_audio_ffn",  "Audio FFN dim"),
        ],
    },
})
