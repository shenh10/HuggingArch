import { formatGiB, formatNumber, fmtTime } from "../format";
import type { DeclarationRow, DerivedGroup } from "../declarations";
import { FormulaHover } from "../FormulaHover";
import type { FormulaDef, FormulaFields } from "../tooltip";
import type { PrecisionReport } from "../precision";
import type { MeasuredCoverage } from "../measured";
import type { CitedAnchor } from "../SourceAnchor";

/** The four bounds of the row roofline (`row_roofline.BOUNDS`). */
export type RowBound = "compute" | "memory" | "comm" | "launch";

export type Formula = { symbolic?: string; substituted?: string; note?: string; defs?: FormulaDef[] };

/** A per-op table's Total row, as the backend summed it over the same rows
 *  the table draws (`phase_formulas.table_total`). `t_sol_s` adds the rows'
 *  t_op (the column the phase's own t_sol sums). The measured total is
 *  not here: it is the phase's one measured formula, all or nothing. */
export type OpTableTotal = {
  flops_per_rank: number;
  access_bytes_per_rank: number;
  /** Null when a row does not publish its collective bytes at this batch:
   *  the column then has no total. */
  comm_bytes_per_rank: number | null;
  t_sol_s: number | null;
};

export type PhaseMetrics = {
  batch?: number;
  /** Total row of this phase's per-op table (`flops_by_module`). */
  op_table_total?: OpTableTotal;
  seq_effective?: number;
  kv_cache_length?: number;
  flops_per_step?: number;
  mem_read_bytes?: number;
  mem_write_bytes?: number;
  /** Null when the step moves no bytes. */
  arithmetic_intensity?: number | null;
  t_compute_s?: number;
  t_memory_s?: number;
  t_comm_s?: number;
  t_sol_s?: number | null;
  /** `t_sol_s` split by the term that set each row's time (compute, memory,
   *  comm, launch); the parts add up to `t_sol_s`. */
  t_sol_by_bound?: Record<RowBound, number> | null;
  /** Each part of `t_sol_by_bound` as a fraction of `t_sol_s` (backend). */
  t_sol_by_bound_share?: Record<RowBound, number> | null;
  /** The measured step time, shown only at complete coverage
   *  (`measured.ts`); never the SOL time. */
  t_measured_s?: number | null;
  /** The partial track: each measured row at its measured time, every other
   *  row at its own SOL time (`Folded.partial`). A blend, not a measurement;
   *  shown only when the reader asks for it, always beside its coverage. */
  t_partial_s?: number | null;
  /** The part of `t_partial_s` that is measured, by time (0–1). */
  partial_measured_share?: number | null;
  /** How many of the step's per-op rows are measured, of how many, and the
   *  keys of the rest by state (docs/auto_sweep.md §6). */
  coverage?: MeasuredCoverage | null;
  /** The same counts as the backend sends them before `coverage`. */
  measured_rows?: number;
  total_rows?: number;
  time_unpublished_reason?: string;
  /** pp>1: the sums the times were published from — one microbatch's
   *  latency, Σ per-stage roofline + inter-stage hops (draft included),
   *  pipeline bubble not modelled. Absent at pp=1. */
  pipeline_time?: {
    t_compute?: number;
    t_memory?: number;
    t_comm?: number;
    t_sol?: number;
    stages?: number;
    /** The bound stage the per-rank account (and this table's rows)
     *  describes. */
    stage?: number;
    per_stage?: number[];
  };
  note?: string;
  binding?: {
    s_prefix_cached?: number;
    /** The verify forward's width, K + 1. */
    s_verify?: number;
    /** K: draft tokens per step. */
    speculative_draft_tokens?: number;
    acceptance_rate?: number;
    mla_absorbed?: boolean;
  };
  formulas?: {
    flops?: Formula;
    mem_read?: Formula;
    mem_write?: Formula;
    arithmetic_intensity?: Formula;
    t_sol?: Formula;
  };
  throughput?: {
    /** Decode: the tokens one request outputs per step (1 + α·K). */
    output_tokens_per_request_step?: number;
    sol?: { per_gpu?: number; cluster?: number; formula?: Formula;
            /** Decode: one request's time per output token — the step over
             *  1 + α·K under speculative decoding, the step itself without. */
            time_per_output_token_s?: number };
    measured?: { per_gpu?: number; cluster?: number; formula?: Formula };
    partial?: { per_gpu?: number; cluster?: number; formula?: Formula;
                time_per_output_token_s?: number };
  };
  /** The speculative draft's forwards, when the request counted them. Every
   *  total above already includes them; this is the draft's own term. */
  speculative_draft?: SpeculativeDraftTerm;
  /** Fold of this step's Program walk. The per-op table paints this tree. */
  flops_by_module?: ModuleFoldRow[];
};

export type ModuleFoldRow = {
  name?: string;
  /** The node's path in its block (the backend's `display_path`). */
  display_path?: string;
  op?: string;
  role?: string;
  aggregation_id?: string;
  template_name?: string;
  /** A template header row's block invocation (`BlockMeta.label`). */
  template_id?: string;
  /** A template header row's short name (`MLA + MoE FFN`), as the
   *  architecture diagram names the block (`Program.block_label`). */
  template_label?: string;
  layer_count?: number;
  depth?: number;
  shape_in?: unknown;
  shape_out?: unknown;
  flops_per_rank?: number;
  flops_full?: number;
  t_op_s?: number;
  /** The leaf's own roofline time (its merged timing row). */
  t_sol_s?: number;
  t_op_agg?: number;
  t_compute_s?: number;
  t_memory_s?: number;
  t_comm_s?: number;
  /** The term that set the row's t_op (backend); null on a parent row. */
  bound?: RowBound | null;
  /** The row's FLOPs over its access bytes (backend); null when it moves none. */
  arithmetic_intensity?: number | null;
  arithmetic_intensity_formula?: FormulaFields | null;
  /** Subtree rollups of the roofline's component columns (compute,
   *  memory, comm) + measured time, so
   *  aggregator rows show what their descendants exercise. */
  t_compute_agg?: number;
  t_memory_agg?: number;
  t_comm_agg?: number;
  t_measured_agg?: number;
  /** The row's own measured time, under any coverage; 0 is a value
   *  (a zero-cost row), absent / null is no measurement. */
  t_measured_s?: number | null;
  measured_state?: string;
  measured_key?: string;
  /** Closed on zero/unplaced rows; a declined row carries its existing
   * human-readable refusal reason. */
  measured_reason?: string;
  /** A measured leaf's MFU / MBU (`PerOpRow.measured_mfu`). */
  measured_mfu?: number | null;
  measured_mbu?: number | null;
  access_bytes_per_rank?: number;
  access_bytes_agg?: number;
  source?: string;
  source_badge?: SourceBadge;
  flops_formula?: FormulaFields;
  t_op_formula?: FormulaFields;
  t_compute_formula?: FormulaFields;
  t_memory_formula?: FormulaFields;
  access_bytes_formula?: FormulaFields;
  /** Spliced collective row (backend `enrich_module_breakdown`): a bare
   *  all_reduce / all_gather inserted after its parent module. */
  is_comm?: boolean;
  comm_bytes_per_rank?: number;
  comm_bytes_agg?: number;
  comm_bytes_formula?: FormulaFields;
  t_comm_formula?: FormulaFields | null;
  comm_world?: number;
  comm_bw_label?: string;
  comm_bw_bytes_per_sec?: number;
  comm_phases?: CommPhase[];
  /** Output shapes of a multi-output leaf — a list of shape lists from
   *  per-op timing, or a port-name → shape map from the shape layer. */
  shape_outs?: unknown[] | Record<string, unknown>;
  /** Unsharded global shapes, when TP/EP makes per-rank differ. */
  shape_in_full?: unknown;
  shape_out_full?: unknown;
  /** A hop across a pipeline cut (see PerOpRow). */
  cut?: number | null;
  source_occurrence?: Occurrence | null;
  sink_occurrences?: Occurrence[];
};

/** One fabric hop of a hierarchical collective (e.g. intra RS + inter AR). */
export type CommPhase = {
  name?: string;
  fabric?: string;
  fabric_label?: string;
  world?: number;
  bytes_per_rank?: number;
  bw_bytes_per_sec?: number;
  t_phase_s?: number;
  symbolic?: string;
  substituted?: string;
};

export type SpeculativeDraftTerm = {
  style: "recursive" | "parallel";
  stacks: string[];
  draft_tokens: number;
  forwards: number;
  tokens_per_forward: number;
  flops_per_step: number;
  total_bytes: number;
  t_sol_s: number | null;
  t_measured_s: number | null;
  /** One forward's share: the draft table's measured total. */
  per_forward?: { t_measured_s?: number | null };
  t_sol_share: number | null;
  formula?: Formula;
  /** Total row of the draft's own per-op table (one forward's rows). */
  op_table_total?: OpTableTotal;
};

export type AbsorbedOp = { name?: string; op?: string; shape_out?: number[] };

export type SourceBadge = {
  fused_op?: string;
  absorbed?: AbsorbedOp[];
  kernel_impl?: string;
  framework?: string;
  /** A measured row's own release. */
  framework_version?: string;
  /** Every release the framework's kernel was measured on, newest first. */
  framework_versions_measured?: string;
  /** The release a framework binding's facts were read at. */
  framework_audited_at?: string;
  selection_mode?: string;
  /** The user-mode NVIDIA driver that executed the measured kernel. Absent on
   *  rows measured before the corpus recorded a driver (schema 14). */
  driver?: string;
  /** Present ONLY when the row's user-mode driver was not the box's own, and
   *  then it is the box's own kernel-mode driver — the one that did NOT run
   *  this kernel. Says only that: the pair carries no direction for the
   *  substitution (an older overlay is a substitution too, and is not
   *  "forward" anything) and no support verdict for it, which depends on the
   *  specific pairing and is not on the row. The comparison is made once, in
   *  the backend (`benchmarks.is_driver_substituted`); absence here means
   *  native or unknown, which `driver` itself distinguishes. */
  driver_substituted_for?: string;
  source_anchor?: CitedAnchor;
};

export type PerOpRow = {
  spec_id?: string;
  aggregation_id?: string;
  op?: string;
  name?: string;
  display_path?: string;
  shape_in?: unknown;
  shape_out?: unknown;
  flops_per_rank?: number;
  access_bytes_per_rank?: number;
  t_compute_s?: number;
  t_memory_s?: number;
  t_comm_s?: number;
  t_sol_s?: number;
  t_op_s?: number;
  /** The term that set the row's t_op (backend); null where none did. */
  bound?: RowBound | null;
  /** The row's FLOPs over its access bytes (backend); null when it moves none. */
  arithmetic_intensity?: number | null;
  arithmetic_intensity_formula?: FormulaFields | null;
  /** The row's own measured time, under any coverage; 0 is a value
   *  (a zero-cost row), absent / null is no measurement. */
  t_measured_s?: number | null;
  measured_state?: string;
  measured_key?: string;
  /** Closed on zero/unplaced rows; a declined row carries its existing
   * human-readable refusal reason. */
  measured_reason?: string;
  /** The selected measurement against the card it ran on (achieved over
   *  the peak recorded at test); null where that peak is missing. */
  measured_mfu?: number | null;
  measured_mbu?: number | null;
  source?: string;
  source_badge?: SourceBadge;
  flops_formula?: FormulaFields;
  t_op_formula?: FormulaFields;
  t_compute_formula?: FormulaFields;
  t_memory_formula?: FormulaFields;
  access_bytes_formula?: FormulaFields;
  /** Collective row data — set on spliced comm rows (see ModuleFoldRow). */
  is_comm?: boolean;
  comm_bytes_per_rank?: number;
  comm_bytes_agg?: number;
  comm_bytes_formula?: FormulaFields;
  /** The backend's `t_comm` popover for a collective row. */
  t_comm_formula?: FormulaFields | null;
  comm_world?: number;
  comm_bw_label?: string;
  comm_bw_bytes_per_sec?: number;
  comm_phases?: CommPhase[];
  shape_outs?: unknown[] | Record<string, unknown>;
  shape_in_full?: unknown;
  shape_out_full?: unknown;
  derived_group?: DerivedGroup;
  /** A hop across a pipeline cut: sent by stage `cut`, from the occurrence
   * that produced the value to the ones that read it past the cut. */
  cut?: number | null;
  source_occurrence?: Occurrence | null;
  sink_occurrences?: Occurrence[];
};

export type Occurrence = { node: string; iteration: number | null };

export type WeightNode = {
  name?: string;
  op?: string;
  class_?: string;
  is_component?: boolean;
  children?: WeightNode[];
  bytes_full?: number;
  bytes_per_rank?: number;
  data_bytes_full?: number;
  scale_bytes_full?: number;
  divisor?: number;
  shard_symbolic?: string;
  shard_substituted?: string;
  leaf_symbolic?: string;
  leaf_substituted?: string;
  leaf_cnt?: number;
  leaf_note?: string;
  leaf_copy_factors?: { n: number; unit: string }[];
  /** How the rank's share reads: `÷N`, a fraction, or `(mixed)`. */
  shard_label?: string;
  params?: number;
  weight_shape?: string;
  weight_dtype?: string;
};

export type WeightBreakdown = {
  weight_dtype?: string;
  deployment_total_bytes?: number;
  /** The deployment total's weight-data and scale-metadata addends, as the
   *  backend folded them off the leaves (the rest is unattributed scale). */
  deployment_data_bytes?: number;
  deployment_scale_bytes?: number;
  published_checkpoint_bytes?: number;
  /** Stacks the model stores that this deployment does not load. */
  left_out_stacks?: Array<{ stack: string; aux_kind?: string | null; encoder?: boolean; bytes: number }>;
  total_bytes_per_rank?: number;
  spec_name?: string;
  groups?: WeightNode[];
  model_level?: WeightNode[];
};

/** One phase's theoretical step time before and after fusion; `delta_s` is
 *  `pre - post`, negative when the fused graph is slower. */
export type FusionDelta = { pre_total_s: number; post_total_s: number; delta_s: number };

/** One `deployment_assumptions` entry: `knob` is the request field that
 *  replaces the assumption with an input; `text` is the sentence to show. */
export type DeploymentAssumption = {
  id: string;
  knob: string | null;
  value: number | null;
  effect: string;
  text: string;
};

/** The media each request carries, by model input, axis → size. */
export type Media = Record<string, Record<string, number>>;

/** One encoder's encode step (`inference/encode_phase.estimate_encode`).
 *  Every number is the backend's; none of it is in the language model's
 *  prefill / decode figures. Timing fields are null for an encoder whose
 *  inputs the request's media did not name (it ran no item). */
export type EncodeStack = {
  stack: string;
  aux_kind: string | null;
  /** The encoder whose weights this one runs (a model's videos through its
   *  image tower); it holds none of its own. */
  reruns_weights_of: string | null;
  inputs: Media;
  /** Prompt positions one request's media fill, per media input (the
   *  defining edge's rows) — the encoder's report; the language model's
   *  prompt does not include them. */
  handoff_tokens_per_request: Record<string, number>;
  items: number;
  media_rows: Record<string, {
    segments: number;
    rows_per_segment: number;
    valid_rows_per_segment: number;
  }>;
  flops_per_rank: number | null;
  bytes_per_rank: number | null;
  t_compute: number | null;
  t_memory: number | null;
  t_comm: number | null;
  t_sol: number | null;
  t_measured: number | null;
  t_sol_per_item: number | null;
  items_per_s: number | null;
  per_op_timing: PerOpRow[];
  per_op_timing_by_declaration?: DeclarationRow[];
  flops_by_module: ModuleFoldRow[];
  /** Total row of the encoder's per-op table; absent when it runs no item. */
  op_table_total?: OpTableTotal;
  vs_naive_delta: FusionDelta | null;
  /** The encoder's resident weights on one rank, whatever the media. */
  weight_bytes_per_rank: number;
  /** The encode step's activation peak on one rank; null when it runs no
   *  item. */
  activation_peak: {
    bytes_per_rank: number;
    worst_block: string | null;
    by_block: Record<string, number>;
  } | null;
};

/** The encoder outputs the framework caches for the language model
 *  (vLLM's encoder cache budget); `tokens` / `bytes_per_rank` are null when
 *  the scheduler states no token budget. */
export type EncoderOutputCache = {
  scheduler_tokens: number | null;
  max_tokens_per_item: number;
  tokens: number | null;
  bytes_per_token: number;
  bytes_per_rank: number | null;
};

/** The estimate's encode phase: each encoder over the request's media;
 *  `subtotal` (several encoders) is the encoders' total alone. */
export type EncodeResult = {
  batch: number;
  media: Media;
  media_tokens: Record<string, number>;
  stacks: EncodeStack[];
  output_cache: EncoderOutputCache;
  subtotal?: {
    weight_bytes_per_rank: number;
    flops_per_rank: number | null;
    bytes_per_rank: number | null;
    t_compute: number | null;
    t_memory: number | null;
    t_comm: number | null;
    t_sol: number | null;
  };
};

/** One deployment's memory budget: what `MemBudget` draws. */
export type DeploymentBudget = Pick<Estimate,
  "cluster_size" | "memory" | "weight_breakdown" | "gpu_memory_fraction" | "formulas">;

export type Estimate = {
  cluster_size?: number;
  num_gpus?: number;
  gpu?: { sku?: string };
  parallel_warnings?: string[];
  shape_warnings?: string[];
  /** The model's published context length; null when it publishes none. */
  max_context_tokens?: number | null;
  /** Whether this request's prompt runs past it. */
  beyond_max_context?: boolean;
  memory?: {
    total_bytes?: number;
    weight_bytes_per_gpu?: number;
    /** The weights a framework can keep in host memory (Engram / PLE
     *  n-gram tables) on this GPU's rank, and whether they are kept there
     *  — then they are not in `weight_bytes_per_gpu`. */
    host_offload_bytes_per_gpu?: number;
    host_offload?: boolean;
    kv_budget_bytes?: number;
    weight_deficit_bytes?: number;
    usable_bytes?: number;
    fits?: boolean;
  };
  kv_cache?: {
    attention_variant?: string;
    attention_type?: string;
    is_hybrid_attention?: boolean;
    num_layers?: number;
    per_request_kv_bytes?: number;
    max_batch_total?: number;
    max_batch_per_gpu?: number;
    effective_storage_tokens_per_request?: number;
    growth_formula?: { symbolic?: string; substituted?: string };
    /** Per-request cache that does not grow with the request (a recurrent
     *  or conv state), per GPU: in `per_request_kv_bytes`, not in
     *  `growth_formula`. */
    constant_bytes?: number;
  };
  phases?: {
    prefill?: PhaseMetrics;
    decode?: PhaseMetrics;
  };
  per_op_timing?: PerOpRow[];
  per_op_timing_decode?: PerOpRow[];
  /** One speculative-draft forward's rows; the decode phase's
   *  `speculative_draft.forwards` says how many the step runs. */
  per_op_timing_draft?: PerOpRow[];
  per_op_timing_by_declaration?: DeclarationRow[];
  per_op_timing_decode_by_declaration?: DeclarationRow[];
  vs_naive_delta?: { prefill: FusionDelta; decode: FusionDelta };
  weight_breakdown?: WeightBreakdown;
  gpu_memory_fraction?: number;
  max_batch_total?: number;
  max_batch_per_gpu?: number;
  /** Why `max_batch_per_gpu` is 0 on this card; null when it serves. */
  infeasibility_reason?: string | null;
  max_decode_batch_per_gpu?: number;
  max_prefill_batch?: number;
  max_prefill_batch_per_gpu?: number;
  /** null: the cache does not grow with length, so no token count fills it. */
  max_tokens_per_gpu?: number | null;
  activation_peak?: { peak_activation_bytes_per_rank?: number };
  avg_input_tokens?: number;
  avg_output_tokens?: number;
  roles_present?: string[];
  /** Whether the model routes tokens to experts (its graph declares a dispatch zone). */
  moe_enabled?: boolean;
  /** What the estimate assumed that its inputs did not determine
   *  (`inference/assumptions.deployment_assumptions`). */
  assumptions?: DeploymentAssumption[];
  formulas?: {
    mem_total?: FormulaFields;
    usable?: FormulaFields;
    deployment?: FormulaFields;
    kv_budget?: FormulaFields;
    max_tokens?: FormulaFields;
    max_tokens_total?: FormulaFields;
    max_batch?: FormulaFields;
    max_batch_total?: FormulaFields;
    max_prefill_batch?: FormulaFields;
    per_request_kv?: FormulaFields;
    weight_per_rank?: FormulaFields;
    weight_deficit?: FormulaFields;
  };
  /** The precision profile the estimate ran under and what it changed. */
  precision?: PrecisionReport;
  /** The encode phase; null for a model with no encoder. */
  encode?: EncodeResult | null;
  disagg?: boolean;
  /** Each cluster's own budget when disaggregated: the stitched fields above
   *  mix the prefill cluster's mesh and weights with the decode cluster's
   *  memory. */
  prefill_deployment?: DeploymentBudget;
  decode_deployment?: DeploymentBudget;
  disagg_handoff?: {
    kv_transfer?: {
      per_request_kv_bytes?: number;
      link_bw_gbps?: number;
      handoff_latency_s?: number;
    };
    rate_match?: {
      decode_to_prefill_time_ratio?: number;
      prefill_time_per_request_s?: number;
      decode_time_per_request_s?: number;
    };
  };
};

function fmtLatency(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s >= 1) return s.toFixed(2) + " s";
  if (s >= 1e-3) return (s * 1e3).toFixed(2) + " ms";
  return (s * 1e6).toFixed(1) + " µs";
}

export function ResultSummary({ result }: { result: Estimate }) {
  const kv = result.kv_cache || {};
  const mem = result.memory || {};
  const formulas = result.formulas || {};
  const reqLen = (result.avg_input_tokens || 0) + (result.avg_output_tokens || 0);
  // No output token, no decode step: nothing below has a decode batch or time.
  const noDecode = result.avg_output_tokens === 0;
  const perReq = kv.per_request_kv_bytes || 0;
  const stateBytes = kv.constant_bytes || 0;
  // One request's time per output token: the step under plain decode, the
  // step over 1 + α·K under speculative decoding (the backend's figure).
  const decodeTpot = result.phases?.decode?.throughput?.sol?.time_per_output_token_s;
  const storeTok = kv.effective_storage_tokens_per_request || reqLen;
  // Per GPU, as the backend names them; absent shows "—", never a total.
  const maxPrefill = result.max_prefill_batch_per_gpu;
  const maxDecode = result.max_decode_batch_per_gpu;

  // A request longer than the model's published context: the formulas do
  // not stop there, so the page says the numbers run past the model.
  const contextWarnings = result.beyond_max_context && result.max_context_tokens
    ? [`The prompt is ${(result.avg_input_tokens || 0).toLocaleString("en-US")} tokens, past the model's `
      + `published max context of ${result.max_context_tokens.toLocaleString("en-US")}; `
      + "every number below extrapolates beyond what the model serves."]
    : [];
  const warnings = [
    ...contextWarnings, ...(result.parallel_warnings || []), ...(result.shape_warnings || []),
  ];

  return (
    <div className="space-y-6 mb-6">
      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <ul>
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}
      {result.disagg && result.disagg_handoff && (() => {
        const kv = result.disagg_handoff.kv_transfer || {};
        const rm = result.disagg_handoff.rate_match || {};
        const ratio = rm.decode_to_prefill_time_ratio;
        return (
          <div className="rounded-lg border border-sky-300 bg-sky-50 px-4 py-3 text-sm text-sky-900">
            <div className="font-semibold mb-1">Prefill → decode handoff</div>
            <ul className="list-disc list-inside space-y-0.5">
              <li>
                KV transfer: <span className="font-mono">{formatGiB(kv.per_request_kv_bytes || 0)}</span> / request
                {" "}over <span className="font-mono">{(kv.link_bw_gbps || 0).toFixed(0)} GB/s</span> link
                {" "}→ <span className="font-mono">{fmtLatency(kv.handoff_latency_s)}</span> latency
              </li>
              <li>
                Per-request time: prefill <span className="font-mono">{fmtLatency(rm.prefill_time_per_request_s)}</span>,
                {" "}decode <span className="font-mono">{fmtLatency(rm.decode_time_per_request_s)}</span>
                {" "}→ ratio <span className="font-mono">{ratio != null && Number.isFinite(ratio) ? ratio.toFixed(1) + "×" : "—"}</span>
                {" "}<span className="text-sky-700">(decode:prefill replica upper bound)</span>
              </li>
            </ul>
          </div>
        );
      })()}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="text-xs text-gray-400 uppercase tracking-wide">Attention</div>
          <div id="infer-attn-type" className="text-lg font-semibold text-gray-900">{kv.attention_variant || kv.attention_type || "—"}</div>
          {/* The growth formula covers the caches that grow per token; a
              hybrid's recurrent state is one per request beside it, so it is
              named here rather than left out of the card. */}
          <div id="infer-attn-dims" className="text-xs text-gray-500 mt-1">
            <span className="text-gray-400">per token: </span>
            <FormulaHover value={kv.growth_formula?.symbolic || "—"} formula={kv.growth_formula} />
          </div>
          {stateBytes > 0 ? (
            <div id="infer-attn-state" className="text-xs text-gray-500 mt-0.5">
              <span className="text-gray-400">+ per request: </span>
              <FormulaHover value={`fixed state ${formatGiB(stateBytes)}`} formula={formulas.per_request_kv} />
            </div>
          ) : null}
        </div>
        <div className="card p-4">
          <div className="text-xs text-gray-400 uppercase tracking-wide">Per-request KV / GPU</div>
          <div id="infer-per-request-kv" className="text-lg font-semibold text-gray-900 font-mono">
            <FormulaHover value={formatGiB(perReq)} formula={formulas.per_request_kv} />
          </div>
          <div id="infer-per-request-kv-sub" className="text-xs text-gray-500 mt-1 font-mono">
            stores {formatNumber(storeTok)} tok{stateBytes > 0 ? <> + {formatGiB(stateBytes)} state</> : null} → {formatGiB(perReq)}
          </div>
        </div>
        <div className="card p-4">
          <div className="text-xs text-gray-400 uppercase tracking-wide">KV budget / GPU</div>
          <div id="infer-kv-budget" className="text-lg font-semibold text-gray-900 font-mono">
            <FormulaHover value={formatGiB(mem.kv_budget_bytes || 0)} formula={formulas.kv_budget} />
          </div>
          <div id="infer-fits" className="text-xs mt-1">
            {mem.fits
              ? <span className="text-green-600">✓ Weights fit</span>
              : <span className="text-red-600">{(mem.weight_deficit_bytes || 0) > 0
                ? <>Short by <FormulaHover value={formatGiB(mem.weight_deficit_bytes || 0)} formula={formulas.weight_deficit} /></>
                : "✗ Weights exceed usable memory"}</span>}
          </div>
        </div>
        <div className="card p-4">
          <div className="text-xs text-gray-400 uppercase tracking-wide"
            title="Requests one DP rank holds: the TP × PP × CP group of GPUs that serves the same requests. The cluster holds this × DP-attention × DP replicas.">
            Max batch / DP rank
          </div>
          <div id="infer-max-prefill-card" className="mt-1 flex items-baseline gap-2">
            <span className="text-10 text-gray-400 uppercase tracking-wide w-12 shrink-0">prefill</span>
            <span id="infer-max-prefill" className="text-lg font-semibold text-purple-700 font-mono">
              <FormulaHover value={formatNumber(maxPrefill)} formula={formulas.max_prefill_batch} cls="text-purple-700" />
            </span>
            <span id="infer-max-prefill-sub" className="text-10 text-gray-500 font-mono">activation-bound</span>
          </div>
          <div id="infer-max-batch-card" className="mt-1 flex items-baseline gap-2">
            <span className="text-10 text-gray-400 uppercase tracking-wide w-12 shrink-0">decode</span>
            <span id="infer-max-batch" className="text-lg font-semibold text-amber-600 font-mono">
              {noDecode ? "—" : <FormulaHover value={formatNumber(maxDecode)} formula={formulas.max_batch} cls="text-amber-600" />}
            </span>
            <span id="infer-max-batch-sub" className="text-10 text-gray-500 font-mono">{noDecode ? "no output tokens" : `KV-bound @ ${reqLen} tok/req`}</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-orange-50 rounded-lg p-3">
          <div className="text-xs text-gray-500">Weights / GPU</div>
          <div className="mt-1 text-sm font-semibold text-gray-900 font-mono">
            <FormulaHover value={formatGiB(mem.weight_bytes_per_gpu || 0)} formula={formulas.weight_per_rank} />
          </div>
        </div>
        <div className="bg-orange-50 rounded-lg p-3">
          <div className="text-xs text-gray-500">Prefill TTFT (SOL)</div>
          <div className="mt-1 text-sm font-semibold text-gray-900 font-mono">
            <FormulaHover value={fmtTime(result.phases?.prefill?.t_sol_s)} formula={result.phases?.prefill?.formulas?.t_sol} />
          </div>
        </div>
        <div className="bg-orange-50 rounded-lg p-3">
          <div className="text-xs text-gray-500">Decode TPOT (SOL)</div>
          <div className="mt-1 text-sm font-semibold text-gray-900 font-mono">
            {noDecode
              ? <span id="infer-tpot-none" className="text-gray-400 font-normal">— no output tokens</span>
              : <FormulaHover value={fmtTime(decodeTpot ?? result.phases?.decode?.t_sol_s)} formula={result.phases?.decode?.formulas?.t_sol} />}
          </div>
          {decodeTpot != null && result.phases?.decode?.t_sol_s != null
            && Math.abs(decodeTpot - result.phases.decode.t_sol_s) > 1e-9 ? (
            <div id="infer-tpot-step" className="mt-0.5 text-11 text-gray-500"
              title="A speculative step verifies the draft tokens and keeps the accepted ones: each request gets 1 + α·K tokens per step, so a token takes the step over that.">
              step {fmtTime(result.phases.decode.t_sol_s)} · {result.phases.decode.throughput?.output_tokens_per_request_step?.toFixed(2)} tok/step per request
            </div>
          ) : null}
        </div>
        <div className="bg-orange-50 rounded-lg p-3">
          <div className="text-xs text-gray-500">Usable HBM / GPU</div>
          <div className="mt-1 text-sm font-semibold text-gray-900 font-mono">
            <FormulaHover value={formatGiB(mem.usable_bytes || 0)} formula={formulas.usable} />
          </div>
        </div>
      </div>
    </div>
  );
}
