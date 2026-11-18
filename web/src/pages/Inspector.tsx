import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { ApiError, fetchJson, isRequestableModelId, stillComputingNote } from "../api";
import { agentLabel, useAgentBackends } from "../agents";
import { authContext, trackAnalyticsEvent } from "../auth";
import { DataTable, tbl, type ColumnDef } from "../DataTable";
import { visibleSpan } from "../declarations";
import { compressRanges, dimMeaning, formatFlops, formatGiB, formatNumber, setDimGlossary } from "../format";
import { ArchGraph, type ArchGraphEdge, type ArchGraphNode } from "./ArchGraph";
import { StreamMap, VARIANT_COLORS, type StreamRow } from "./StreamMap";
import { type BlockGraphEntry } from "./BlockDag";
import { GraphShell, type GraphDiffData } from "./Structure";
import { playgroundCompareHref, type PlaygroundPick } from "../url";
import {
  fetchSpecJson,
  formatUsage,
  jobMessage,
  latestGeneration,
  pollSpecJob,
  shouldAutoFix,
  shouldAutoGenerate,
  shouldOfferRefine,
  specUrls,
  stopSpecJob,
  type GenerationRow,
  type SpecJobResult,
  type SpecStart,
  type SpecStatus,
} from "../specGen";
import type { EngineGap } from "../specStatus";
import { EngineBlockedBanner } from "../EngineGaps";
import { filterAndSortModels, ModelOptions, useSupportedModels } from "../supportedModels";
import { currentHref, readLastInspectorHref, rememberInspectorHref } from "../lastVisited";
import { FormulaHover } from "../FormulaHover";
import type { FormulaFields } from "../tooltip";

// The "Try:" quick-pick is fetched from `/inspector/examples` (the latest
// ≥100B models in the corpus, computed server-side so it never drifts). This
// is only the offline fallback shown until that fetch returns / if it fails.
const EXAMPLES_FALLBACK = [
  "zai-org/GLM-5.2",
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
  "stepfun-ai/Step-3.7-Flash",
  "inclusionAI/Ling-2.6-1T",
] as const;

type Tab = "dashboard" | "diagram" | "modules" | "raw" | "forward";

type QuantMetaScale = {
  suffix?: string;
  kind?: string | null;
  axes?: string[] | null;
  shape?: (number | string)[];
  dtype?: string | null;
};
type QuantMetaLeaf = {
  path?: string;
  params?: number;
  params_symbolic?: string | null;
  params_values?: string | null;
  scales?: QuantMetaScale[];
};
type QuantMetaTpl = {
  template?: string;
  layer_count?: number;
  per_layer_params?: number;
  total_params?: number;
  leaves?: QuantMetaLeaf[];
};

type Dashboard = {
  total_params?: number;
  total_active_params?: number;
  total_params_formatted?: string;
  total_active_params_formatted?: string;
  /** Per prompt token in prefill, for a long prompt: only the layers that
   *  must run on every position. `total_active_params` is per decoded token. */
  total_prefill_active_params?: number | null;
  /** The prompt length past which `total_prefill_active_params` is what
   *  each further prompt token touches; 0 when every layer runs on every
   *  prompt position. */
  prefill_full_from?: number | null;
  /** Per decoded token: the language model's layers, no encoder. */
  total_decode_active_params?: number | null;
  main_prefill_active_params?: number | null;
  /** Lookup tables other than the input embedding, read a few rows a token. */
  lookup_table_params?: number | null;
  embedding_params?: number;
  non_embedding_params?: number;
  flops_per_token?: number;
  flops_per_token_g?: string;
  total_flops?: number;
  attention_flops?: number;
  ffn_flops?: number;
  attention_type?: string | null;
  /** The cache families behind `attention_type`, first appearance first. */
  attention_families?: string[];
  num_layers?: number;
  hidden_size?: number;
  moe_enabled?: boolean;
  moe_layers?: number;
  moe_active_experts?: number;
  moe_total_experts?: number;
  main_total_params?: number;
  main_active_params?: number;
  aux_params?: Record<string, number>;
  aux_active_params?: Record<string, number>;
  aux_total_params?: number;
  encoder_params?: Record<string, number>;
  encoder_total_params?: number;
  /** Each encoder stack's modality (`aux_kind`). */
  encoder_stacks?: Record<string, string | null>;
  /** The FLOPs one media item costs each encoder in its encode phase, as an
   *  expression in the item's own axes (`variables`); no size is bound. */
  encoder_flops_per_item?: Record<string, {
    aux_kind: string | null;
    inputs: string[];
    variables: string[];
    symbolic: string;
    per_item: string;
    per_item_readable?: string;
    latex: string;
    description: string;
  }>;
  /** Weight totals over the language model (`lm`) and over the whole
   *  checkpoint, encoder towers included (`with_encoders`). */
  weight_totals?: Record<WeightScope, WeightTotals> | null;
  weight_bytes?: number;
  param_bytes?: number;
  /** Tied weights the checkpoint also stores as their own copy: in the
   *  checkpoint size and `param_bytes`, skipped at load (`weight_bytes`). */
  checkpoint_copies?: CheckpointCopy[];
  quant_metadata_params?: number;
  /** The scales' most common dtype. */
  quant_metadata_scale_dtype?: string | null;
  /** The metadata's stored bytes, each tensor at its own width: the total
   *  and per dtype, largest first. */
  quant_metadata_bytes?: number;
  quant_metadata_formats?: { dtype: string; bytes: number }[];
  quant_metadata_breakdown?: QuantMetaTpl[];
  hf_model_storage_bytes?: number;
  hf_quantization_metadata_bytes?: number;
  hf_checkpoint_storage_bytes?: number;
  spec_storage_bytes?: number;
  spec_metadata_bytes?: number;
  spec_checkpoint_bytes?: number;
  storage_diff?: number;
  storage_diff_pct?: number;
  metadata_storage_diff?: number;
  metadata_storage_diff_pct?: number;
  all_tensor_diff?: number;
  all_tensor_diff_pct?: number;
  checkpoint_anchored?: boolean;
  hf_index_total_size?: number | null;
  measured_tensor_diff?: number | null;
  measured_tensor_diff_pct?: number | null;
  weight_dtype?: string | null;
  dtype_bytes?: number | null;
  vocab_size?: number;
  spec_unresolved_note?: string | null;
};

type ModelConfigSwa = {
  num_heads?: number;
  num_kv_heads?: number;
  head_dim?: number;
  v_head_dim?: number;
  rope_theta?: string | null;
};
type AttnFact = { symbol: string; value: number; config?: string | null };
type AttnFamily = { name: string; layers: number; dims: AttnFact[] };
type ModelConfigAttn = {
  type?: string | null;
  families?: AttnFamily[];
  model_dims?: AttnFact[];
  rope_theta?: string | number;
  rope_scaling?: string;
  sliding_window?: number;
  swa?: ModelConfigSwa;
};
type ModelConfigFfn = {
  type?: string;
  intermediate_size?: number;
  activation?: string;
  num_experts?: number;
  top_k?: number;
  shared_experts?: number;
  moe_intermediate_size?: number;
  dense_layers?: number[];
  mtp_layers?: number;
};
type ModelConfigNorm = { type?: string; eps?: unknown };
type ConfigDim = { field?: string; symbol?: string; value?: number };
type BlockTypeRow = {
  template?: string;
  label?: string;
  count?: number;
  attention_type?: string | null;
  ffn_type?: string | null;
};
type ModelConfig = {
  model_type?: string;
  model_class?: string | null;
  total_params?: number;
  total_active_params?: number | null;
  precision?: string;
  tied_weight?: boolean | null;
  max_position_embeddings?: number;
  attention?: ModelConfigAttn;
  ffn?: ModelConfigFfn;
  norm?: ModelConfigNorm;
  config_dims?: ConfigDim[];
  block_types?: BlockTypeRow[];
};
type RawSummary = {
  vocab_size?: number;
  hf_model_storage_bytes?: number;
  param_bytes?: number;
};
/** A dtype a row's cached tensors are priced at, "declared" by the spec's
 *  cache entries or the checkpoint's "native" one. */
type KvDtype = { dtype: string; source: "declared" | "native" };

/** One row of the Dashboard's model composition (`backend/views/composition.py`):
 *  a block declaration (derived groups joined back under it) or a model-level
 *  node. Every figure is the row's whole share over its `layers`. */
type CompositionRow = {
  key: string;
  kind: "embedding" | "block" | "model_root" | "lm_head";
  label: string;
  template?: string | null;
  templates?: string[];
  stack?: string | null;
  aux_kind?: string | null;
  layer_indices: number[];
  layers: number;
  params: { attention: number; ffn: number; experts: number; other: number; total: number };
  active_params: number;
  storage_bytes: number;
  storage_by_dtype: Array<{ dtype: string; bytes: number }>;
  formulas?: {
    attention?: FormulaFields;
    ffn?: FormulaFields;
    experts?: FormulaFields;
    other?: FormulaFields;
    total?: FormulaFields;
    active?: FormulaFields;
    storage?: FormulaFields;
  };
  kv: {
    per_token_bytes: number;
    constant_bytes: number;
    growth_formula?: FormulaFields | null;
    constant_formula?: FormulaFields | null;
    dtypes: KvDtype[];
  } | null;
  histogram?: Array<{ visible: number; layers: number; layer_indices: number[] }>;
  axis?: string;
  stream?: string;
};
type Composition = {
  rows: CompositionRow[];
  totals: {
    params: number;
    active_params: number;
    storage_bytes: number;
    kv_per_token_bytes: number | null;
    kv_constant_bytes: number | null;
    kv_growth_formula?: FormulaFields | null;
    kv_constant_formula?: FormulaFields | null;
    formulas?: {
      params?: FormulaFields;
      active?: FormulaFields;
      storage?: FormulaFields;
    };
    storage_by_dtype: Array<{ dtype: string; bytes: number }>;
    /** Every dtype the KV figures are priced at, and where it is stated. */
    kv_dtypes?: KvDtype[];
  };
  /** What an undeclared cached tensor is priced at — the model's stated
   *  default precision; null when it states none (the KV columns are then
   *  empty). */
  kv_native_dtype: string | null;
  /** "quant_context", "config", or "config.text_config" (a multimodal
   *  config's language model). */
  kv_native_source: string | null;
};

type ArchLayer = {
  id?: string;
  name?: string;
  type?: string;
  detail?: string;
  expandable?: boolean;
  template?: string;
  cycle_blocks?: ArchLayer[];
  // A kind card's templates: the blocks of one kind that bind different
  // components, each with its own layers.
  variants?: ArchLayer[];
  // A variant: the slots bound differently from the card's other variants,
  // each with the component bound there and its declared description.
  slots?: { slot: string; component: string; description: string }[];
  // The layer indices the card stands for, and the inner streams they
  // write and read, layer by layer.
  layers?: number[];
  stream_map?: StreamRow[];
  stream_produces?: string[];
  stream_reads?: string[];
  stream_windows?: Record<string, number | string>;
  // Per inner stream the card's layers read: the `reads_from` they declare
  // and, per reading layer, the layer whose write it reads.
  stream_sources?: Record<string, { declared: string; reads: [number, number | string][] }>;
};
type ArchSection = {
  id?: string;
  name?: string;
  repeat?: number;
  sub_layers?: ArchLayer[];
};
type ArchView = {
  model_name?: string;
  subtitle?: string;
  badges?: string[];
  model_inputs?: { id: string; name: string }[];
  layers?: ArchLayer[];
  extra_sections?: ArchSection[];
  // Who reads whom between the cards, by id; null when the view has no
  // spec to read it off and the cards stand in order.
  edges?: ArchGraphEdge[] | null;
  summary_table?: Record<string, string>;
};
type BlockGraph = Record<string, BlockGraphEntry>;

type ForwardSource = {
  source?: string;
  files?: Array<{ path?: string; text?: string }>;
  truncated?: boolean;
};

type AnalyzeData = {
  success?: boolean;
  message?: string;
  model_id?: string;
  composition?: Composition | null;
  analysis_time_ms?: number;
  model_info?: { family?: string };
  dashboard?: Dashboard;
  block_graph?: BlockGraph;
  representative_template?: RepresentativeTemplate | null;
  formulas?: unknown;
  architecture_view?: ArchView;
  model_config?: ModelConfig;
  raw_data?: { summary?: RawSummary };
  model_str?: string;
  config_json?: unknown;
  config_processed?: unknown;
  forward_source?: ForwardSource;
  served_from?: string;
  // The status of the caller's own record the analysis was served from; an
  // engine-blocked record also carries where validate met each gap.
  record_status?: string;
  engine_gaps?: EngineGap[];
  // An analysis that refused because the engine lacks a capability.
  error_kind?: string;
  engine_gap?: { capability: string; node?: string | null };
  /** A refusal the spec writer resolves: the repair, for Refine / Agent Fix. */
  errors?: string[];
  /** Agent Fix can open the saved record on these refusals. */
  fixable?: boolean;
  record_id?: string;
  /** No spec matches the model (no library spec, no record of the caller's). */
  no_spec?: boolean;
  // false when the library spec answered a signed-in user: a private Refine
  // of it is refused (it would be superseded the moment it landed).
  refine_offered?: boolean;
  spec_unresolved_note?: string | null;
};

type BannerKind = "none" | "missing" | "fail" | "ready" | "success" | "spec_ready" | "blocked" | "record_failed";
type SpecBanner = {
  kind: BannerKind;
  /** `blocked`: the engine refusals, and whether the spec is the caller's
   *  served record, the gap this analysis itself hit, and a record that did
   *  not save. */
  gaps?: EngineGap[];
  served?: boolean;
  hit?: { capability: string; node?: string | null };
  notSaved?: boolean;
  /** `record_failed`: the served record's status, and the repair a
   *  spec-owned refusal names. */
  recordStatus?: string;
  /** `record_failed`: the analysis's `error_kind` (`RunErrorKind`). */
  errorKind?: string;
  errors?: string[];
  modelId?: string;
  message?: string;
  record?: GenerationRow | null;
  needSignIn?: boolean;
  authEnabled?: boolean;
  /** The failed generation left a draft that can be continued. */
  resumable?: boolean;
};

/** What differs between the Inspector's spec-agent jobs (generate a
 *  missing spec / refine a mismatched one / Fix a saved spec that does
 *  not load); `runSpecJob` does the rest. */
type SpecJob = {
  kind: "generate" | "refine" | "fix";
  mid: string;
  /** Required when `kind` is `fix`. */
  recordId?: string;
  /** "generation" / "refinement", for the reattach status line. */
  noun: string;
  start: (urls: ReturnType<typeof specUrls>) => { url: string; body: Record<string, unknown> };
  successStatus: string;
  failStatus: string;
  onOpen?: () => void;
  onExpired: () => void;
  onSuccess?: () => void;
  /** `signInRequired`: the run could not go on without a signed-in
   *  session (`SpecJobResult.signInRequired`). */
  onFail: (message: string, resumable: boolean, signInRequired: boolean) => void;
  /** The run ended blocked by the analysis engine: its spec is correct. */
  onBlocked: (result: SpecJobResult) => void;
  /** The start request answered without starting a job. */
  onNotStarted: (
    data: SpecStart,
    finish: (success: boolean, message: string, resumable?: boolean,
             result?: SpecJobResult) => Promise<void>,
  ) => Promise<void>;
  onConnectionError: (msg: string) => void;
  connectionErrorCloseMs: number;
};

type CompareDiffRow = {
  label?: string; key?: string; kind?: string; slot?: string; values?: unknown[]; same?: boolean;
  /** A weight total's row carries the whole-checkpoint figures beside the
   *  language model's (`values`); the page shows one of the two. */
  values_with_encoders?: unknown[]; same_with_encoders?: boolean;
};
type WeightScope = "lm" | "with_encoders";
type WeightTotals = {
  weight_bytes?: number;
  param_bytes?: number;
  storage_by_dtype?: Array<{ dtype: string; bytes: number }>;
};

/** The weight totals a compare column shows: the backend's figures for the
 *  chosen scope. */
function ccWeights(dash: Dashboard | undefined, scope: WeightScope): WeightTotals {
  return (dash?.weight_totals && dash.weight_totals[scope]) || {};
}
type CompareModality = {
  name?: string;
  label?: string;
  present?: boolean[];
  rows?: CompareDiffRow[];
};
type CompareDiff = {
  metrics?: CompareDiffRow[];
  components?: CompareDiffRow[];
  dims?: CompareDiffRow[];
  modalities?: CompareModality[];
};
type CompareResp = {
  success?: boolean;
  message?: string;
  errors?: Array<string | null>;
  results?: Array<AnalyzeData | null>;
  specs?: Array<string | null>;
  diff?: CompareDiff;
  graph_diff?: GraphDiffData;
};

/** The expensive Inspector result survives top-nav route changes in this JS
 * session. URL state remains authoritative: a direct link to another model
 * does not consume a snapshot belonging to the previous one. */
type InspectorMemory = {
  href: string;
  modelId: string;
  shownId: string;
  mode: "single" | "compare";
  tab: Tab;
  error: string;
  data: AnalyzeData | null;
  cmpModels: string[];
  cmpStatus: string;
  cmp: CompareResp | null;
  includeEncoders: boolean;
  banner: SpecBanner;
};

let inspectorMemory: InspectorMemory | null = null;

// `batch`, `seq` and `dtype` are not read: they are keys so a link carrying
// them has them cleared on the next write.
const HASH_KEYS = ["model", "batch", "seq", "dtype", "sub", "m", "view"] as const;
const TABS = new Set<Tab>(["dashboard", "diagram", "modules", "raw", "forward"]);

/** Slot index colors — vanilla compare.js `_CC_COLORS`. */
const CC_COLORS = ["#4e79a7", "#f28e2c", "#e15759", "#59a14f", "#af7aa1"] as const;

/** vanilla `_ccModelName`: result.model_id || spec || `Model ${i+1}`. */
function ccModelName(
  result: AnalyzeData | null | undefined,
  spec: string | null | undefined,
  i: number,
): string {
  return (result && result.model_id) || spec || `Model ${i + 1}`;
}

/** What every headline Total / Active counts. */
const MAIN_MODEL_TITLE = "The main model: no MTP head, no encoder tower — what the model card and the gallery count.";

/** vanilla header badges: attention_type, MoE, formatNumber(main_total_params); falsy dropped. */
function ccHeaderBadges(dash: Dashboard | undefined): string[] {
  const d = dash || {};
  const total = dash?.main_total_params;
  const params = total == null || !Number.isFinite(total) ? "" : formatNumber(total);
  return [d.attention_type, d.moe_enabled ? "MoE" : null, params].filter((x): x is string => Boolean(x));
}

/** vanilla `_ccShortName`: last `/` segment of `_ccModelName`. */
/** Backend `analyzer._representative_template`: the main-path block Compare
 *  draws for a model, the basis it was picked on and the call Playground
 *  evaluates it at; a tie names no block and lists the tied ones. */
type RepresentativeBlock = { name: string; bucket_id: number; call: string };
type RepresentativeTemplate = {
  name?: string | null;
  basis?: "most_layers" | "tie";
  count?: number;
  n_layers?: number;
  bucket_id?: number;
  call?: string;
  tied?: RepresentativeBlock[];
};

// Playground benches at most 8 columns.
const PLAYGROUND_COLUMN_CAP = 8;

/** The Playground columns of a comparison: each model's representative block
 *  at its call, or every tied block of a model that has no single one. */
function representativePicks(results: (AnalyzeData | null | undefined)[]): { picks: PlaygroundPick[]; dropped: PlaygroundPick[] } {
  const picks: PlaygroundPick[] = [];
  for (const r of results) {
    const rt = r?.representative_template;
    if (!r?.model_id || !rt) continue;
    const blocks = rt.name && rt.call != null ? [{ name: rt.name, call: rt.call }] : rt.tied || [];
    for (const b of blocks) picks.push({ name: b.name, model_id: r.model_id, call: b.call });
  }
  return { picks: picks.slice(0, PLAYGROUND_COLUMN_CAP), dropped: picks.slice(PLAYGROUND_COLUMN_CAP) };
}

/** One model's line under "Representative Block Layer": which block is drawn
 *  and why, as the backend states it. */
function RepresentativeBasis({ result, spec, i }: {
  result: AnalyzeData | null | undefined; spec: string | null | undefined; i: number;
}) {
  const rt = result?.representative_template;
  const who = (
    <span className="font-semibold" style={{ color: CC_COLORS[i % CC_COLORS.length] }}>
      {ccShortName(result, spec, i)}
    </span>
  );
  if (!rt) return <li data-rep-basis="none">{who}: no main-path block</li>;
  if (rt.basis === "tie") {
    return (
      <li data-rep-basis="tie">
        {who}: no single representative block — {(rt.tied || []).length} blocks tie
        at {rt.count} of {rt.n_layers} layers
        ({(rt.tied || []).map((b) => <span key={b.call} className="font-mono">{b.name}</span>)
          .reduce<ReactNode[]>((acc, el, k) => (k ? [...acc, ", ", el] : [el]), [])})
      </li>
    );
  }
  return (
    <li data-rep-basis={rt.basis}>
      {who}: representative block <span className="font-mono break-all">{rt.name}</span>
      {" "}(most layers, {rt.count} of {rt.n_layers})
    </li>
  );
}

function ccShortName(
  result: AnalyzeData | null | undefined,
  spec: string | null | undefined,
  i: number,
): string {
  const full = ccModelName(result, spec, i);
  return full.includes("/") ? full.split("/").pop() || full : full;
}

/** vanilla `compareValueFmt` in compare.js. */
function compareValueFmt(kind: string | undefined, v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (kind === "params") return formatNumber(typeof v === "number" ? v : Number(v));
  if (kind === "flops") return formatFlops(typeof v === "number" ? v : Number(v));
  if (kind === "bytes") {
    const n = typeof v === "number" ? v : Number(v);
    return formatGiB(n);
  }
  if (kind === "number" && typeof v === "number") return v.toLocaleString();
  if (kind === "bytes_by_dtype" && Array.isArray(v)) {
    return (v as Array<{ dtype: string; bytes: number }>)
      .map((d) => `${d.dtype} ${formatGiB(d.bytes)}`).join(" · ");
  }
  return String(v);
}

type DiffRow = { values?: unknown[]; same?: boolean; kind?: string; label: ReactNode };

function DiffMatrixTable({
  title,
  rows,
  firstColLabel,
  results,
  specs,
  cellFmt,
}: {
  title: string;
  rows: DiffRow[];
  firstColLabel: string;
  results: Array<AnalyzeData | null>;
  specs: Array<string | null | undefined>;
  cellFmt: (row: DiffRow, v: unknown) => ReactNode;
}) {
  const columns: ColumnDef<DiffRow>[] = [
    {
      key: "label", wrap: true,
      header: firstColLabel,
      className: "text-gray-600 text-sm",
      render: (row) => row.label,
      sortValue: null,
    },
    ...results.map((r, i): ColumnDef<DiffRow> => ({
      key: `v${i}`,
      header: (
        <span
          className="whitespace-nowrap"
          style={{ color: CC_COLORS[i % CC_COLORS.length] }}
        >
          {ccShortName(r, specs[i], i)}
        </span>
      ),
      className: "font-mono text-sm text-gray-900 whitespace-nowrap",
      render: (row) => cellFmt(row, (row.values || [])[i]),
      sortValue: null,
    })),
  ];
  if (!rows.length) return null;
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">{title}</div>
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(_row, i) => i}
        framed={false}
        rowClassName={(row) => (row.same ? "opacity-50" : "bg-amber-50/40")}
      />
    </div>
  );
}

/** vanilla `renderCompareDiff`. */
function CompareDiffContent({
  diff,
  results,
  specs,
  scope,
}: {
  diff: CompareDiff;
  results: Array<AnalyzeData | null>;
  specs: Array<string | null | undefined>;
  scope: WeightScope;
}) {
  return (
    <>
      <DiffMatrixTable
        title="Metrics"
        firstColLabel=""
        results={results}
        specs={specs}
        rows={(diff.metrics || []).map((m) => (
          scope === "with_encoders" && m.values_with_encoders
            ? { ...m, values: m.values_with_encoders, same: m.same_with_encoders, label: m.label || "" }
            : { ...m, label: m.label || "" }))}
        cellFmt={(row, v) => compareValueFmt(row.kind, v)}
      />
      <DiffMatrixTable
        title="Component Calls"
        firstColLabel="Role"
        results={results}
        specs={specs}
        rows={(diff.components || []).map((c) => ({ ...c, label: c.slot || "" }))}
        cellFmt={(_row, v) => (Array.isArray(v) && v.length ? (
          <div className="space-y-0.5">
            {v.map((call, k) => <div key={k} className="font-mono text-xs whitespace-normal break-all">{String(call)}</div>)}
          </div>
        ) : "—")}
      />
      <DiffMatrixTable
        title="Dim Bindings (differing)"
        firstColLabel="Symbol"
        results={results}
        specs={specs}
        rows={(diff.dims || []).map((d) => ({
          ...d,
          label: (
            <>
              <span className="font-mono" title={dimMeaning(d.key) || undefined}>{d.key}</span>
              <span className="hidden sm:inline text-gray-400 ml-2" title={dimMeaning(d.key) || undefined}>{dimHeadline(d.key)}</span>
            </>
          ),
        }))}
        cellFmt={(_row, v) => (v === null || v === undefined ? "—" : String(v))}
      />
      {(diff.modalities || []).map((m, mi) => {
        const presence = (m.present || [])
          .map((p, i) => (p ? (
            <span key={i} style={{ color: CC_COLORS[i % CC_COLORS.length] }}>
              {ccShortName(results[i], specs[i], i)}
            </span>
          ) : null))
          .filter(Boolean);
        const presenceNodes: ReactNode[] = [];
        presence.forEach((node, i) => {
          if (i) presenceNodes.push(", ");
          presenceNodes.push(node);
        });
        return (
          <div key={m.name || m.label || mi} className="border border-amber-100 bg-amber-50/30 rounded-lg p-3">
            <div className="text-xs uppercase tracking-wide text-amber-700 mb-2">{m.label}</div>
            <div className="text-sm text-gray-600 mb-2">
              {m.label} present in: {presenceNodes.length ? presenceNodes : "(none)"}
            </div>
            <DiffMatrixTable
              title={`${m.label} dims`}
              firstColLabel="Dim"
              results={results}
              specs={specs}
              rows={(m.rows || []).map((r) => ({ ...r, label: r.label || "" }))}
              cellFmt={(_row, v) => (v === null || v === undefined ? "—" : String(v))}
            />
          </div>
        );
      })}
    </>
  );
}

/** One model's Compare column: Architecture / Parameters / Stored weights /
 *  KV cache — the same scenario-free quantities as its composition. */
function CompareDashCard({ title, rows }: { title: string; rows: Array<[string, string]> }) {
  return (
    <div className="card p-4">
      <h4 className="font-semibold text-sm mb-2">{title}</h4>
      <div className="space-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between text-sm">
            <span className="text-gray-500">{k}</span>
            <span className="font-mono text-gray-900">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function CompareDashboard({ data, scope }: { data: AnalyzeData | null | undefined; scope: WeightScope }) {
  const d: Dashboard = (data && data.dashboard) || {};
  const comp = data?.composition;
  const w = ccWeights(d, scope);
  const archRows: Array<[string, string]> = [
    ["Attention", d.attention_type || "—"],
    ["Layers", d.num_layers != null ? String(d.num_layers) : "—"],
    ["Hidden size", d.hidden_size != null ? String(d.hidden_size) : "—"],
  ];
  if (d.moe_enabled) {
    const tot = d.moe_total_experts;
    const act = d.moe_active_experts;
    const mly = d.moe_layers;
    archRows.push(["MoE experts", tot != null && act != null ? `${act} / ${tot} active` : "—"]);
    archRows.push([
      "MoE layers",
      mly != null && d.num_layers != null ? `${mly} / ${d.num_layers}` : mly != null ? String(mly) : "—",
    ]);
  }
  return (
    <>
      <CompareDashCard title="Architecture" rows={archRows} />
      <CompareDashCard
        title="Parameters"
        rows={[
          ["Total", formatNumber(d.main_total_params)],
          ["Active (decode)", formatNumber(d.main_active_params)],
          ["+ MTP", formatNumber(d.aux_total_params || 0)],
          ["+ Encoders", formatNumber(d.encoder_total_params || 0)],
          ["Quant metadata", formatGiB(d.hf_quantization_metadata_bytes || 0)],
          [
            "Diff %",
            d.storage_diff_pct !== undefined && d.storage_diff_pct !== null
              ? `${d.storage_diff_pct.toFixed(4)}%`
              : "—",
          ],
        ]}
      />
      <CompareDashCard
        title="Stored weights"
        rows={[
          ["Weight bytes", formatGiB(w.weight_bytes || 0)],
          // On disk: a tied weight's stored copy counts here, not in the
          // weight bytes above, which are what serving loads.
          ["Param bytes (on disk)", formatGiB(w.param_bytes || 0)],
          ["HF param bytes", formatGiB(d.hf_model_storage_bytes || 0)],
          ...(w.storage_by_dtype || []).map(
            (x): [string, string] => [x.dtype, formatGiB(x.bytes)]),
        ]}
      />
      <CompareDashCard
        title="KV cache"
        rows={comp && comp.totals.kv_per_token_bytes != null ? [
          ["Per token", formatGiB(comp.totals.kv_per_token_bytes)],
          ["Constant / sequence", formatGiB(comp.totals.kv_constant_bytes || 0)],
          ["Dtype", (comp.totals.kv_dtypes || []).map((d) => d.dtype).join(" · ") || "—"],
        ] : [["Per token", "—"]]}
      />
    </>
  );
}

type CheckpointCopy = { tensor: string; bytes: number };

/** Why the checkpoint holds more bytes than the model loads: the tied
 *  weights it stores a second time. Nothing when it stores none. */
function CheckpointCopiesNote({ copies }: { copies?: CheckpointCopy[] }) {
  if (!copies || !copies.length) return null;
  const total = copies.reduce((n, c) => n + c.bytes, 0);
  return (
    <div className="text-10 text-gray-400 font-normal"
      title="The config ties these weights to another, so serving loads one table; the checkpoint still stores the copy.">
      incl. {formatGiB(total)} tied {copies.length === 1 ? "copy" : "copies"} ({copies.map((c) => c.tensor).join(", ")}), not loaded
    </div>
  );
}

function readInspectHash(): Record<string, string> {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const out: Record<string, string> = {};
  for (const k of HASH_KEYS) {
    const v = params.get(k);
    if (v) out[k] = v;
  }
  return out;
}

function writeInspectHash(h: Record<string, string>) {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  for (const k of HASH_KEYS) {
    if (h[k]) params.set(k, h[k]);
    else params.delete(k);
  }
  const next = params.toString();
  const hash = next ? `#${next}` : "";
  if (hash !== window.location.hash) {
    history.replaceState(null, "", `${window.location.pathname}${hash}`);
  }
  rememberInspectorHref(currentHref());
}

/** Same text as vanilla `renderQuantMetaBreakdown` in dashboard.js. */
function quantMetaBreakdownText(breakdown: QuantMetaTpl[], total: number,
  totalBytes: number, formats: { dtype: string; bytes: number }[]): string {
  const lines: string[] = [];
  for (const tpl of breakdown) {
    lines.push(`${tpl.template} × ${tpl.layer_count} layers → ${formatNumber(tpl.total_params)} scales`);
    lines.push(`  per-layer = ${formatNumber(tpl.per_layer_params)}`);
    for (const leaf of tpl.leaves || []) {
      // leaf.params counts the scale tensors; params_symbolic/values describe
      // the WEIGHT the scales are derived from — never join them with "=".
      const sym = leaf.params_symbolic;
      const val = leaf.params_values;
      const weight = sym ? ` (weight ${sym}${val && val !== sym ? ` = ${val}` : ""})` : "";
      // The scale tensor's own declaration: symbolic axes (config constants
      // substituted, in/out symbolic) and its resolved shape, per sibling.
      const scaleDesc = (leaf.scales || [])
        .map((s) => {
          const axes = (s.axes || []).join(" × ");
          const shape = (s.shape || []).join("×");
          const eq = axes && shape ? `${axes} = ${shape}` : axes || shape;
          return eq ? ` = ${eq}` : "";
        })
        .join("");
      lines.push(`    ${leaf.path}: ${formatNumber(leaf.params)} scales${scaleDesc}${weight}`);
    }
  }
  lines.push("");
  // The bytes are the backend's sum, each tensor at its own width — a
  // checkpoint can mix scale dtypes, so no count × one width here.
  const only = formats.length === 1 ? ` (${formats[0].dtype})` : "";
  lines.push(
    totalBytes
      ? `TOTAL = ${formatNumber(total)} scales${only} = ${formatGiB(totalBytes)}`
      : `TOTAL = ${formatNumber(total)}`,
  );
  if (formats.length > 1) {
    for (const f of formats) lines.push(`  ${f.dtype}: ${formatGiB(f.bytes)}`);
  }
  return lines.join("\n");
}

/** The attention badge's colour, keyed on the cache families the model's
 *  attention runs (`attention_families`, `arch/cache_families.yaml`) — a DSA
 *  model declares MLA — never on its display name. The first family with a
 *  colour of its own wins; any other reads as MHA's. */
const ATTN_FAMILY_CLASS: Record<string, string> = {
  MLA: "attention-mla",
  GQA: "attention-gqa",
};

function attnClass(families: string[] | null | undefined): string {
  return (families || []).map((f) => ATTN_FAMILY_CLASS[f]).find(Boolean) || "attention-mha";
}

/** vanilla `analyze.js` paints `#generate-status` / `#refine-status` via innerHTML spans. */
type OverlayStatusTone = "green" | "red" | "red600" | "orange" | "blue" | "violet";

const OVERLAY_STATUS_CLASS: Record<OverlayStatusTone, string> = {
  violet: "text-violet-700 font-medium",
  green: "text-green-600 font-medium",
  red: "text-red-500",
  red600: "text-red-600",
  orange: "text-orange-500",
  blue: "text-blue-500",
};

function AgentChip({ cli, model }: { cli?: string; model?: string }) {
  const { agents } = useAgentBackends();
  if (!cli && !model) return null;
  return (
    <div className="text-xs text-gray-500 text-right shrink-0 leading-snug">
      {cli ? <div>{agentLabel(agents, cli)}</div> : null}
      {model ? <div className="font-mono text-11 text-gray-400 max-w-[14rem] truncate" title={model}>{model}</div> : null}
    </div>
  );
}

function OverlayStatusSlot({ id, text, tone }: { id: string; text: string; tone: OverlayStatusTone | "" }) {
  return (
    <div id={id} className="text-sm text-gray-400">
      {text ? (tone ? <span className={OVERLAY_STATUS_CLASS[tone]}>{text}</span> : text) : null}
    </div>
  );
}

/** vanilla inspector.html Parameters / Computation / Memory Traffic icon heads. */
function ResultCardHead({
  title,
  iconBg,
  iconFg,
  path,
}: {
  title: string;
  iconBg: string;
  iconFg: string;
  path: string;
}) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <div className={`w-10 h-10 ${iconBg} rounded-lg flex items-center justify-center`}>
        <svg className={`w-6 h-6 ${iconFg}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={path} />
        </svg>
      </div>
      <h3 className="font-semibold text-gray-900">{title}</h3>
    </div>
  );
}

export function Inspector() {
  const restored = useRef(
    inspectorMemory?.href === currentHref() ? inspectorMemory : null,
  ).current;
  const [modelId, setModelId] = useState(restored?.modelId || "deepseek-ai/DeepSeek-V3");
  const { models, loading: modelsLoading } = useSupportedModels();
  const [examples, setExamples] = useState<readonly string[]>(EXAMPLES_FALLBACK);
  const [modelDropOpen, setModelDropOpen] = useState(false);
  const modelDropRef = useRef<HTMLDivElement>(null);
  const [cmpDropOpen, setCmpDropOpen] = useState<number | null>(null);
  const cmpListRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<"single" | "compare">(restored?.mode || "single");
  const [tab, setTab] = useState<Tab>(restored?.tab || "dashboard");
  const [loading, setLoading] = useState(false);
  const [loadDetail, setLoadDetail] = useState("Loading configuration...");
  const [error, setError] = useState(restored?.error || "");
  const [data, setData] = useState<AnalyzeData | null>(restored?.data || null);
  const [hydrated, setHydrated] = useState(false);
  // The model the single view analyzed (or is analyzing). The link names this,
  // not the text being typed into the input.
  const [shownId, setShownId] = useState(restored?.shownId || "");
  const [cmpModels, setCmpModels] = useState(restored?.cmpModels || [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3.2",
  ]);
  const [cmpStatus, setCmpStatus] = useState(restored?.cmpStatus || "");
  const [cmp, setCmp] = useState<CompareResp | null>(restored?.cmp || null);
  // Compare totals the language models' weights; the towers are added only
  // when asked (an encoder's weights are its own accounting).
  const [includeEncoders, setIncludeEncoders] = useState(restored?.includeEncoders || false);
  const weightScope: WeightScope = includeEncoders ? "with_encoders" : "lm";
  const [banner, setBanner] = useState<SpecBanner>(restored?.banner || { kind: "none" });
  /** Model whose kept draft is being chosen over (continue vs start over)
    * before a fresh generation is allowed to start. */
  const [resumePrompt, setResumePrompt] = useState<string | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [overlayKind, setOverlayKind] = useState<"generate" | "refine" | "fix">("generate");
  const [genLog, setGenLog] = useState("Connecting to coding agent...\n");
  const [genUsage, setGenUsage] = useState("");
  const [genStatus, setGenStatus] = useState("");
  const [genStatusTone, setGenStatusTone] = useState<OverlayStatusTone | "">("");
  const [genAgent, setGenAgent] = useState<{ cli?: string; model?: string }>({});
  const [stopDisabled, setStopDisabled] = useState(false);
  const [copied, setCopied] = useState("");
  const [quantMetaOpen, setQuantMetaOpen] = useState(false);
  const overlayOpen = useRef(false);
  const stopFlag = useRef(false);
  const autoGenFor = useRef("");
  const autoFixFor = useRef("");
  // The hash this mount arrived with. Hydration and the one-shot auto-run read
  // it, not the hash the page writes back once state changes.
  // The hash this page follows: read at mount, and again when the user
  // edits it or navigates back (`hashchange`; the page's own writes are
  // `replaceState`, which fires none). `appliedHash` is it once the form
  // state it sets has rendered, which is when it is run.
  const [inboundHash, setInboundHash] = useState(readInspectHash);
  const [appliedHash, setAppliedHash] = useState<Record<string, string> | null>(null);
  const skipRestoredRun = useRef(Boolean(
    restored && (restored.mode === "compare" ? restored.cmp : restored.data),
  ));
  const memorySnapshot = useRef<InspectorMemory | null>(null);
  memorySnapshot.current = loading ? null : {
    href: readLastInspectorHref(), modelId, shownId, mode, tab, error, data,
    cmpModels, cmpStatus, cmp, includeEncoders, banner,
  };
  useEffect(() => () => {
    inspectorMemory = memorySnapshot.current
      ? { ...memorySnapshot.current, href: readLastInspectorHref() }
      : null;
  }, []);
  useEffect(() => {
    const onHash = () => setInboundHash(readInspectHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  // Latest analyze / compare request; a response from an older one is dropped.
  const analyzeSeq = useRef(0);
  const compareSeq = useRef(0);

  // Leaving the page ends any spec-job polling this mount started.
  useEffect(() => () => { stopFlag.current = true; }, []);

  useEffect(() => {
    const h = inboundHash;
    if (h.model) {
      setModelId(h.model);
      setShownId(h.model);
    }
    if (h.sub === "compare") {
      setMode("compare");
      const list = (h.m || "").split("|").filter(Boolean);
      if (list.length >= 2) setCmpModels(list.slice(0, 5));
    } else if (h.model) {
      setMode("single");
    }
    if (h.view && TABS.has(h.view as Tab)) setTab(h.view as Tab);
    setHydrated(true);
    setAppliedHash(h);
  }, [inboundHash]);

  useEffect(() => {
    fetchJson<{ items?: Record<string, string> }>("/dim-glossary")
      .then((j) => {
        if (j && j.items && typeof j.items === "object") setDimGlossary(j.items);
      })
      .catch(() => { /* dim tooltips disappear; non-blocking */ });
    fetchJson<{ examples?: string[] }>("/inspector/examples")
      .then((j) => {
        if (Array.isArray(j.examples) && j.examples.length) setExamples(j.examples);
      })
      .catch(() => { /* keep the fallback list */ });
  }, []);

  useEffect(() => {
    if (!modelDropOpen) return;
    const onDoc = (e: MouseEvent) => {
      const el = modelDropRef.current;
      if (el && !el.contains(e.target as Node)) setModelDropOpen(false);
    };
    document.addEventListener("click", onDoc);
    return () => document.removeEventListener("click", onDoc);
  }, [modelDropOpen]);

  useEffect(() => {
    if (mode !== "compare") setCmpDropOpen(null);
  }, [mode]);

  useEffect(() => {
    if (cmpDropOpen === null) return;
    const onDoc = (e: MouseEvent) => {
      const el = cmpListRef.current;
      if (el && !el.contains(e.target as Node)) setCmpDropOpen(null);
    };
    document.addEventListener("click", onDoc);
    return () => document.removeEventListener("click", onDoc);
  }, [cmpDropOpen]);

  useEffect(() => {
    if (!resumePrompt) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setResumePrompt(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [resumePrompt]);

  useEffect(() => {
    setQuantMetaOpen(false);
  }, [data]);

  useEffect(() => {
    if (!hydrated) return;
    writeInspectHash({
      // A compare link names its models in `m`; the single view's model
      // would be a stale extra.
      model: mode === "compare" ? "" : shownId,
      sub: mode === "compare" ? "compare" : "",
      m: mode === "compare" ? cmpModels.filter((m) => m.trim()).join("|") : "",
      view: tab === "dashboard" ? "" : tab,
    });
  }, [hydrated, shownId, mode, cmpModels, tab]);

  const runRef = useRef<(id?: string) => Promise<void>>(async () => {});
  const runCompareRef = useRef<() => Promise<void>>(async () => {});

  /** Hide the job overlay after `ms`, then run `then`. */
  const closeOverlayAfter = useCallback((ms: number, then?: () => void) => {
    setTimeout(() => {
      overlayOpen.current = false;
      setOverlay(false);
      then?.();
    }, ms);
  }, []);

  /** One spec-agent job behind the log overlay: reattach to a job already
   *  running for the model, else start one, then poll it to the end and
   *  re-analyze on success. */
  const runSpecJob = useCallback(async (job: SpecJob) => {
    const { mid } = job;
    if (!mid || overlayOpen.current) return;
    overlayOpen.current = true;
    stopFlag.current = false;
    setStopDisabled(false);
    setOverlayKind(job.kind);
    setOverlay(true);
    setGenLog("Connecting to coding agent...\n");
    setGenUsage("");
    setGenAgent({});
    setGenStatus("");
    setGenStatusTone("");
    job.onOpen?.();

    const ctx = await authContext();
    if (ctx.expired) {
      setGenStatus("Session expired — please sign in again.");
      setGenStatusTone("red");
      job.onExpired();
      closeOverlayAfter(6000);
      return;
    }

    const urls = specUrls(ctx.loggedIn);
    const finish = async (success: boolean, message: string, resumable = false,
                          result?: SpecJobResult) => {
      if (!success && result?.errorKind === "blocked_by_engine") {
        // The spec is correct and waits on the engine: not an error card.
        setGenStatus(message || "Blocked by the analysis engine.");
        setGenStatusTone("violet");
        job.onBlocked(result);
        // A kept record (or a newer valid one) is what analyze serves now.
        const kept = ["recorded", "stale"].includes(result.record?.outcome || "");
        closeOverlayAfter(5000, kept ? () => void runRef.current(mid) : undefined);
        return;
      }
      if (success) {
        setGenStatus(job.successStatus);
        setGenStatusTone("green");
        job.onSuccess?.();
        closeOverlayAfter(1000, () => void runRef.current(mid));
      } else {
        setGenStatus(message || job.failStatus);
        setGenStatusTone("red");
        job.onFail(message, resumable, !!result?.signInRequired);
        closeOverlayAfter(5000);
      }
    };
    const poll = () => pollSpecJob(mid, {
      authed: ctx.loggedIn,
      kind: job.kind,
      recordId: job.recordId,
      onLog: (line) => setGenLog((prev) => prev + line + "\n"),
      onUsage: (acc) => setGenUsage(formatUsage(acc)),
      onAgent: (cli, model) => setGenAgent({ cli, model }),
      stopped: () => stopFlag.current,
    });
    const liveStatusUrl = job.kind === "fix"
      ? `/api/spec/fix-status?record_id=${encodeURIComponent(job.recordId || "")}`
      : `${urls.status}?model_id=${encodeURIComponent(mid)}&kind=${encodeURIComponent(job.kind)}`;

    try {
      try {
        const s = await fetchSpecJson<{ state?: string }>(liveStatusUrl);
        if (s && s.state === "running") {
          setGenStatus(`Reattaching to in-progress ${job.noun}…`);
          setGenStatusTone("");
          const result = await poll();
          await finish(result.success, result.message, result.resumable, result);
          return;
        }
      } catch { /* no in-flight job */ }

      const start = job.start(urls);
      const data = await fetchSpecJson<SpecStart>(start.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(start.body ?? {}),
      });
      if (data.status === "already_running") {
        setGenStatus(`Reattaching to in-progress ${job.noun}…`);
        setGenStatusTone("");
        const result = await poll();
        await finish(result.success, result.message, result.resumable, result);
        return;
      }
      if (data.status !== "started") {
        await job.onNotStarted(data, finish);
        return;
      }
      const result = await poll();
      await finish(result.success, result.message, result.resumable, result);
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
        setGenStatus("Session expired — please sign in again.");
        setGenStatusTone("red");
        job.onExpired();
        closeOverlayAfter(6000);
        return;
      }
      if (e instanceof ApiError && e.status >= 400 && e.status < 500) {
        const data = { ...e.body, message: e.message } as SpecStart;
        await job.onNotStarted(data, finish);
        return;
      }
      const msg = e instanceof Error ? e.message : String(e);
      setGenStatus(`Connection error: ${msg}`);
      setGenStatusTone("red");
      job.onConnectionError(msg);
      closeOverlayAfter(job.connectionErrorCloseMs);
    }
  }, [closeOverlayAfter]);

  /** Before a fresh generation starts: does its leaf keep an unfinished
   *  draft? The server refuses a plain new run over one (409 `draft_kept`),
   *  so the choice belongs here — in a dialog — not in the failure banner
   *  after the start "errored". Returns "prompt" when that dialog is now
   *  open; "start" when there is nothing to choose over, or the check itself
   *  failed (the POST's own 409 still asks then, as the race fallback). */
  const checkKeptDraft = useCallback(async (mid: string): Promise<"start" | "prompt"> => {
    const ctx = await authContext();
    if (ctx.expired) return "start"; // runSpecJob reports the expiry
    try {
      const urls = specUrls(ctx.loggedIn);
      const s = await fetchSpecJson<SpecStatus>(
        `${urls.status}?model_id=${encodeURIComponent(mid)}&kind=generate`,
      );
      if (s?.resumable) {
        setResumePrompt(mid);
        return "prompt";
      }
    } catch { /* uncheckable → start; the POST's 409 asks */ }
    return "start";
  }, []);

  /** Generate a spec. `"resume"` continues the draft an unfinished generation
   *  kept; `"discard"` starts over and discards it. A plain `"new"` asks over
   *  a kept draft before anything starts (the server would otherwise refuse
   *  it with 409 `draft_kept`). */
  const generateSpec = useCallback(async (id?: string, start: "new" | "resume" | "discard" = "new") => {
    const mid = (id ?? modelId).trim();
    if (start === "new" && !overlayOpen.current && (await checkKeptDraft(mid)) === "prompt") {
      return;
    }
    await runSpecJob({
      kind: "generate",
      mid,
      noun: "generation",
      start: (urls) => ({
        url: urls.generate,
        body: { model_id: mid, resume: start === "resume", discard_draft: start === "discard" },
      }),
      successStatus: "Spec generated successfully. Re-analyzing...",
      failStatus: "Generation failed",
      onOpen: () => setBanner({ kind: "missing", modelId: mid }),
      onExpired: () => setBanner({
        kind: "fail",
        modelId: mid,
        message: "Your session expired. Please sign in again, then retry generation.",
        needSignIn: true,
      }),
      onSuccess: () => setBanner({ kind: "success", modelId: mid }),
      onBlocked: (result) => setBanner({
        kind: "blocked",
        modelId: mid,
        gaps: result.engineGaps || [],
        notSaved: result.record?.outcome === "failed",
      }),
      onFail: (message, resumable, signInRequired) => setBanner({
        kind: "fail",
        modelId: mid,
        message,
        needSignIn: signInRequired,
        resumable,
      }),
      onNotStarted: async (data, finish) => {
        if (!data.success) {
          const message = jobMessage(data, "Generation failed");
          await finish(false, message, !!data.resumable, {
            success: false,
            message,
            signInRequired: data.status === "disabled",
          });
          return;
        }
        setGenStatus("Spec ready. Re-analyzing...");
        setGenStatusTone("green");
        setBanner({ kind: "spec_ready", modelId: mid, message: data.spec_name });
        closeOverlayAfter(1000, () => void runRef.current(mid));
      },
      onConnectionError: (msg) => setBanner({
        kind: "fail",
        modelId: mid,
        message: msg,
      }),
      connectionErrorCloseMs: 5000,
    });
  }, [modelId, runSpecJob, closeOverlayAfter, checkKeptDraft]);

  const refineSpec = useCallback(async (mid: string, specBytes: number, hfBytes: number, storageDiff: number) => {
    await runSpecJob({
      kind: "refine",
      mid,
      noun: "refinement",
      start: (urls) => ({
        url: urls.refine,
        body: {
          model_id: mid,
          spec_storage_bytes: specBytes,
          hf_storage_bytes: hfBytes,
          storage_diff: storageDiff,
        },
      }),
      successStatus: "Spec refined successfully. Re-analyzing...",
      failStatus: "Refinement failed",
      onExpired: () => setError("Your session expired. Please sign in again, then retry refinement."),
      onFail: (message) => setError(message || "Spec refinement failed"),
      onBlocked: (result) => setBanner({
        kind: "blocked",
        modelId: mid,
        gaps: result.engineGaps || [],
        notSaved: result.record?.outcome === "failed",
      }),
      onNotStarted: async (data, finish) => {
        if (data.status === "overloaded") {
          setGenStatus("Server busy — too many concurrent tasks. Try again later.");
          setGenStatusTone("orange");
          setError("Spec refinement queued: server busy (too many concurrent tasks). Try again later.");
          closeOverlayAfter(3000);
        } else if (data.status === "duplicate") {
          setGenStatus("Already analyzing this model. Please wait.");
          setGenStatusTone("blue");
          setError("Spec refinement already in progress for this model.");
          closeOverlayAfter(3000);
        } else {
          await finish(false, jobMessage(data, "Spec refinement failed"));
        }
      },
      onConnectionError: (msg) => setError(`Connection error during spec refinement: ${msg}`),
      connectionErrorCloseMs: 3000,
    });
  }, [runSpecJob, closeOverlayAfter]);

  const fixSpec = useCallback(async (recordId: string, mid: string) => {
    await runSpecJob({
      kind: "fix",
      mid,
      recordId,
      noun: "fix",
      start: () => ({
        url: `/api/account/generations/${encodeURIComponent(recordId)}/fix`,
        body: {},
      }),
      successStatus: "Spec repaired. Re-analyzing...",
      failStatus: "Agent Fix failed",
      onOpen: () => setBanner({ kind: "record_failed", modelId: mid }),
      onExpired: () => setBanner({
        kind: "record_failed",
        modelId: mid,
        message: "Your session expired. Please sign in again, then retry Agent Fix.",
        needSignIn: true,
      }),
      onFail: (message) => setBanner({
        kind: "record_failed",
        modelId: mid,
        message: message || "Agent Fix failed",
      }),
      onBlocked: (result) => setBanner({
        kind: "blocked",
        modelId: mid,
        served: true,
        gaps: result.engineGaps || [],
        notSaved: result.record?.outcome === "failed",
      }),
      onNotStarted: async (data, finish) => {
        await finish(false, jobMessage(data, "Agent Fix could not start"));
      },
      onConnectionError: (msg) => setBanner({
        kind: "record_failed",
        modelId: mid,
        message: msg,
      }),
      connectionErrorCloseMs: 5000,
    });
  }, [runSpecJob]);

  const run = useCallback(async (id?: string) => {
    const mid = (id ?? modelId).trim();
    if (!mid) {
      setError("Model ID required.");
      return;
    }
    if (!isRequestableModelId(mid)) {
      setError("Model ID must be owner/name, or a registered custom:name id.");
      return;
    }
    // The link follows the model being analyzed (the hash effect writes it).
    setShownId(mid);
    const my = ++analyzeSeq.current;
    const stale = () => my !== analyzeSeq.current;
    setLoading(true);
    setError("");
    setData(null);
    setBanner((b) => (b.kind === "ready" && b.modelId === mid ? b : { kind: "none" }));
    setLoadDetail(`Loading config for ${mid}...`);
    const t0 = performance.now();
    try {
      const ctx = await authContext();
      const urls = specUrls(ctx.loggedIn);
      const body = await fetchJson<AnalyzeData>(urls.analyze, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_id: mid,
          // The graph drawn for the model, not for a batch and length.
          symbolic_shapes: true,
        }),
      }, (info) => { if (!stale()) setLoadDetail(stillComputingNote(info)); });
      if (stale()) return;
      if (!body.success && body.served_from === "user_generation") {
        // The caller's own spec did not analyse. It is never regenerated
        // by itself: an engine gap shows where the engine falls short; a
        // repairable schema refusal starts Agent Fix; anything else shows
        // the repair for a click.
        setData({ model_id: mid, success: false, message: body.message, dashboard: {} });
        setLoading(false);
        if (body.error_kind === "blocked_by_engine" || body.record_status === "blocked_by_engine") {
          setBanner({ kind: "blocked", modelId: mid, served: true, gaps: body.engine_gaps || [],
                      hit: body.engine_gap, recordStatus: body.record_status });
        } else {
          setBanner({ kind: "record_failed", modelId: mid, message: body.message,
                      errorKind: body.error_kind || undefined,
                      errors: body.errors, recordStatus: body.record_status });
          if (shouldAutoFix(body) && autoFixFor.current !== body.record_id) {
            autoFixFor.current = body.record_id || "";
            void fixSpec(body.record_id as string, mid);
          }
        }
        return;
      }
      if (!body.success) {
        const msg = body.message || "Analysis failed";
        // No spec at all (no library spec, no record of the caller's): the
        // one case the Inspector generates one by itself.
        if (shouldAutoGenerate(body)) {
          setData({ model_id: mid, success: false, message: msg, dashboard: {} });
          setLoading(false);
          setBanner({ kind: "missing", modelId: mid });
          if (autoGenFor.current !== mid) {
            autoGenFor.current = mid;
            void generateSpec(mid);
          }
          return;
        }
        throw new Error(msg);
      }
      // The result carries the model it was computed for; everything below the
      // form reads `data.model_id`, not the live input.
      setData({ ...body, model_id: body.model_id || mid });
      setLoading(false);
      if (shouldAutoGenerate(body)) {
        setBanner({ kind: "missing", modelId: mid });
        if (autoGenFor.current !== mid) {
          autoGenFor.current = mid;
          void generateSpec(mid);
        }
      } else if (body.record_status === "blocked_by_engine") {
        setBanner({ kind: "blocked", modelId: mid, served: true, gaps: body.engine_gaps || [],
                    recordStatus: body.record_status });
      } else if (body.served_from === "user_generation") {
        const record = ctx.loggedIn ? await latestGeneration(mid) : null;
        if (stale()) return;
        setBanner({
          kind: "ready",
          modelId: mid,
          message: "This spec is private to you. Review it and open a PR from your account.",
          record,
          needSignIn: ctx.authEnabled && !ctx.loggedIn,
          authEnabled: ctx.authEnabled,
        });
      }
      trackAnalyticsEvent("analyze", mid, Math.round(performance.now() - t0));
    } catch (e) {
      if (stale()) return;
      setLoading(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [modelId, generateSpec, fixSpec]);

  runRef.current = run;

  useEffect(() => {
    if (!appliedHash) return;
    if (skipRestoredRun.current) {
      skipRestoredRun.current = false;
      return;
    }
    const h = appliedHash;
    // vanilla registerTrigger('inspector-compare'): sub=compare needs a|b, not analyze.
    if (h.sub === "compare") {
      const models = (h.m || "").split("|").map((s) => s.trim()).filter(Boolean);
      if (models.length >= 2) void runCompareRef.current();
      return;
    }
    if (h.model) void run(h.model);
    // once per hash followed, after its form state has rendered
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appliedHash]);

  const filteredModels = useMemo(() => filterAndSortModels(models, modelId), [models, modelId]);
  const dash = data?.dashboard || {};
  const mainTotal = dash.main_total_params || 0;
  const mainActive = dash.main_active_params || 0;
  // A prompt must be able to get past `prefill_full_from`: when that length
  // is beyond the model's max context (an all-window model whose layers'
  // windows stack past it), the row would describe prompts it never takes.
  const maxContext = data?.model_config?.max_position_embeddings;
  const showPrefillActive = dash.main_prefill_active_params != null
    && (dash.prefill_full_from || 0) > 0 && dash.main_prefill_active_params < mainActive
    && !(maxContext && (dash.prefill_full_from || 0) >= maxContext);
  const hasSpec = !!((dash?.total_params || 0) > 0 && data?.block_graph && typeof data.block_graph === "object" && Object.keys(data.block_graph as object).length);
  const arch = data?.architecture_view;
  const showRefine = shouldOfferRefine(dash) && data?.refine_offered !== false;

  const copyText = (text: string, ok: string) => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(ok);
      window.setTimeout(() => setCopied(""), 2000);
    }).catch(() => setCopied("Copy failed"));
  };


  /** A composition row on the Architecture Diagram: switch to that tab and
   *  bring the first box drawn for one of the row's templates into view. */
  const openInDiagram = useCallback((templates: string[]) => {
    setTab("diagram");
    if (!templates.length) return;
    window.setTimeout(() => {
      const box = Array.from(document.querySelectorAll<HTMLElement>("#arch-view-container [data-arch-template]"))
        .find((el) => templates.includes(el.dataset.archTemplate || ""));
      if (!box) return;
      box.scrollIntoView({ behavior: "smooth", block: "center" });
      box.style.outline = "2px solid #f59e0b";
      window.setTimeout(() => { box.style.outline = ""; }, 1600);
    }, 0);
  }, []);

  const runCompare = async () => {
    const ids = cmpModels.map((m) => m.trim()).filter(Boolean);
    if (ids.length < 2) {
      setCmpStatus("At least 2 models required.");
      return;
    }
    const invalidAt = ids.findIndex((id) => !isRequestableModelId(id));
    if (invalidAt >= 0) {
      setCmpStatus(`Model #${invalidAt + 1} must be owner/name, or a registered custom:name id.`);
      return;
    }
    // vanilla runCompare: updateShareableUrl() after the min-2 check, before Analyzing
    writeInspectHash({
      // A compare link names its models in `m`; the single view's model
      // would be a stale extra.
      model: mode === "compare" ? "" : shownId,
      sub: mode === "compare" ? "compare" : "",
      m: mode === "compare" ? cmpModels.filter((m) => m.trim()).join("|") : "",
      view: tab === "dashboard" ? "" : tab,
    });
    const my = ++compareSeq.current;
    setCmpStatus(`Analyzing ${ids.length} models…`);
    setCmp(null);
    try {
      const body = await fetchJson<CompareResp>("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models: ids, symbolic_shapes: true }),
      });
      if (my !== compareSeq.current) return;
      if (!body.success) {
        const errs = (body.errors || [])
          .map((e, i) => (e ? `#${i + 1}: ${e}` : null))
          .filter(Boolean)
          .join("; ");
        setCmpStatus(`Failed: ${body.message || ""} ${errs}`.trim());
        return;
      }
      setCmp(body);
      setCmpStatus("");
    } catch (e) {
      if (my !== compareSeq.current) return;
      const ex = e as { message?: string };
      setCmpStatus(`Error: ${ex.message || e}`);
    }
  };
  runCompareRef.current = runCompare;

  return (
    <main className="w-full page-shell py-4 sm:py-8 flex-1">
      <div id="inspect-mode-single" className={mode === "single" ? undefined : "hidden"}>
          <section className="card p-4 sm:p-6 mb-8">
            <h2 className="text-xl font-semibold mb-1">Model Configuration</h2>
            <p className="text-sm text-gray-500 mb-4">Enter a HuggingFace model id to analyze its architecture.</p>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model ID</label>
            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1" ref={modelDropRef}>
                <input
                  type="text"
                  id="model-id"
                  autoComplete="off"
                  data-action="filter-supported-models"
                  className="w-full pr-10 px-4 py-3 text-base border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500"
                  value={modelId}
                  onChange={(e) => { setModelId(e.target.value); setModelDropOpen(true); }}
                  onFocus={() => setModelDropOpen(true)}
                  onKeyPress={(e) => { if (e.key === "Enter") void run(); }}
                  placeholder="e.g., meta-llama/Llama-2-7b-hf"
                />
                <button
                  type="button"
                  id="model-id-caret"
                  title="Show supported models"
                  data-action="toggle-supported-models"
                  className="absolute inset-y-0 right-0 w-10 flex items-center justify-center text-gray-400 hover:text-amber-600 transition"
                  onClick={(e) => { e.preventDefault(); setModelDropOpen((o) => !o); }}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                </button>
                <div
                  id="model-id-dropdown"
                  className={`${modelDropOpen ? "" : "hidden "}absolute left-0 right-0 top-full mt-1 max-h-72 overflow-y-auto bg-white border border-gray-200 rounded-lg shadow-lg z-20`}
                >
                  <ModelOptions
                    items={filteredModels}
                    action="select-supported-model"
                    loading={modelsLoading}
                    onSelect={(id) => { setModelId(id); setModelDropOpen(false); }}
                  />
                </div>
              </div>
              <button
                type="button"
                data-action="analyze"
                className="bg-amber-500 text-white px-6 py-3 rounded-lg hover:bg-amber-600 transition-colors flex items-center justify-center gap-2 shrink-0"
                onClick={() => void run()}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                </svg>
                Analyze Model
              </button>
            </div>
            <div className="mt-3 flex items-center gap-3 flex-wrap text-sm">
              <span className="text-gray-400">Try:</span>
              {examples.map((m) => (
                <button
                  key={m}
                  type="button"
                  data-action="load-example"
                  data-model={m}
                  className="text-amber-600 hover:text-amber-700"
                  onClick={() => { setModelId(m); }}
                >
                  {m.split("/").pop()}
                </button>
              ))}
              <div className="flex-1" />
              <button
                type="button"
                id="inspect-to-compare"
                className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg text-gray-700 hover:bg-amber-50 hover:border-amber-300"
                title="Diff this model against a second one side by side."
                onClick={() => {
                  const m = modelId.trim();
                  setCmpModels((ms) => {
                    const next = [...(ms.length ? ms : [""])];
                    next[0] = m;
                    while (next.length < 2) next.push("");
                    return next;
                  });
                  setMode("compare");
                }}
              >
                + Compare with another model
              </button>
            </div>
          </section>

          <div id="loading" className={`${loading ? "" : "hidden "}text-center py-12`}>
            <div className="loading-spinner mx-auto mb-4"></div>
            <p className="text-gray-600">Analyzing model architecture...</p>
            <p className="text-sm text-gray-500" id="loading-detail">{loadDetail}</p>
          </div>

          {banner.kind === "ready" && !loading && (
            <SpecBannerCard banner={banner} />
          )}
          {banner.kind === "blocked" && !loading && (
            <BlockedSpecBanner banner={banner} />
          )}
          {banner.kind === "record_failed" && !loading && (
            <RecordFailedBanner banner={banner} />
          )}

          {/* Outside the results: a model with no spec has no dashboard, and
              this banner is the whole answer for it. */}
          {data && !loading && (
              <div
                id="no-spec-banner"
                className={
                  banner.kind === "fail"
                    ? "mb-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center justify-between"
                    : banner.kind === "success" || banner.kind === "spec_ready"
                      ? "mb-4 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center justify-between"
                      : `${hasSpec || banner.kind === "ready" || banner.kind === "blocked" || banner.kind === "record_failed" ? "hidden " : ""}mb-4 bg-amber-50 border border-amber-200 rounded-lg p-4 flex items-center justify-between`
                }
              >
                {banner.kind === "fail" ? (
                  <div className="flex items-start justify-between w-full gap-4 min-w-0">
                    <div className="text-red-800 min-w-0">
                      <span className="font-medium">Generation failed:</span>
                      <span className="text-red-600 text-sm ml-2 break-words">{banner.message || "Unknown error"}</span>
                    </div>
                    {banner.needSignIn ? (
                      <a href={`/auth/login.html?next=${encodeURIComponent(window.location.pathname + window.location.hash)}`}
                        data-action="sign-in-to-generate"
                        className="shrink-0 px-4 py-1.5 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 transition">
                        Sign in
                      </a>
                    ) : (
                    <div className="flex items-center gap-2 shrink-0">
                      {banner.resumable ? (
                        <button
                          type="button"
                          data-action="resume-spec-generation"
                          className="px-4 py-1.5 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 transition"
                          onClick={() => void generateSpec(banner.modelId, "resume")}
                        >
                          Continue last generation
                        </button>
                      ) : null}
                      <button
                        type="button"
                        data-action="trigger-spec-generation"
                        className={banner.resumable
                          ? "px-4 py-1.5 bg-white text-red-700 border border-red-300 text-sm rounded-lg hover:bg-red-50 transition"
                          : "px-4 py-1.5 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 transition"}
                        onClick={() => void generateSpec(banner.modelId, banner.resumable ? "discard" : "new")}
                      >
                        {banner.resumable ? "Start over" : "Retry"}
                      </button>
                    </div>
                    )}
                  </div>
                ) : banner.kind === "spec_ready" ? (
                  <div className="text-green-800">
                    <span className="font-medium">Spec ready: {banner.message}</span>
                    <span className="text-green-600 text-sm ml-2">— Re-analyzing...</span>
                  </div>
                ) : (
                  <>
                    <div>
                      <span className="text-amber-800 font-medium">No architecture spec found for </span>
                      <code className="model-id-slot text-amber-900 font-mono text-sm bg-amber-100 px-1 rounded">{data?.model_id}</code>
                      <span className="text-amber-600 text-sm ml-2">— FLOPs and structure details unavailable</span>
                    </div>
                    <button
                      type="button"
                      data-action="trigger-spec-generation"
                      className="px-4 py-1.5 bg-amber-600 text-white text-sm rounded-lg hover:bg-amber-700 transition"
                      onClick={() => void generateSpec()}
                    >
                      Generate Spec
                    </button>
                  </>
                )}
              </div>
          )}
          {/* An analysis that did not run (no spec yet, or the caller's spec
              refused) has no dashboard: its banner above is the whole answer. */}
          <div id="results" className={`${data && data.success !== false && !loading ? "" : "hidden "}fade-in`}>
              {hasSpec && dash && data && <UnresolvedBanner data={data} />}
              <div className="card p-4 sm:p-6 mb-6">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <h2 id="result-model-id" className="text-2xl font-bold text-gray-900 break-words">{data?.model_id}</h2>
                    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mt-2">
                      <span id="result-family" className="px-3 py-1 bg-gray-100 rounded-full text-sm text-gray-600">{data?.model_info?.family}</span>
                      <span id="result-attention" className={`px-3 py-1 rounded-full text-sm font-medium ${attnClass(dash.attention_families)}`}>{dash.attention_type || "—"}</span>
                      <span id="result-moe" className={`${dash.moe_enabled ? "" : "hidden "}px-3 py-1 bg-orange-100 text-orange-700 rounded-full text-sm font-medium`}>MoE Enabled</span>
                    </div>
                  </div>
                  <div className="text-right text-sm text-gray-500">
                    Analysis time: <span id="analysis-time">{data?.analysis_time_ms?.toFixed(0)}</span>ms
                  </div>
                </div>
              </div>

              <div className={`mb-8 grid grid-cols-1 gap-6 ${dash.moe_enabled ? "lg:grid-cols-[minmax(0,1fr)_minmax(16rem,20rem)] lg:items-start" : ""}`}>
                <div className="card p-4 sm:p-6">
                  <ResultCardHead
                    title="Parameters"
                    iconBg="bg-blue-100"
                    iconFg="text-blue-600"
                    path="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
                  />
                  {/* Total / Active are the main model's — no MTP head, no
                      encoder — as the gallery, compare and model cards state
                      them; what else the checkpoint holds is listed below. */}
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600" title={MAIN_MODEL_TITLE}>Total</span>
                    <span id="total-params" className="text-2xl font-bold text-gray-900">{formatNumber(dash.main_total_params || 0)}</span>
                  </div>
                  <div id="active-params-row" className={`${mainActive < mainTotal || showPrefillActive ? "" : "hidden "}flex justify-between items-center`}>
                    <span className="text-gray-600" title={`Parameters one decoded token touches in the main model: every layer it passes, and the head. ${MAIN_MODEL_TITLE}`}>Active (decode)</span>
                    <span id="active-params" className="text-xl font-bold text-orange-700">{formatNumber(mainActive)}</span>
                  </div>
                  {/* Shown only when some layer runs on the prompt's last positions
                      alone (`prefill_full_from > 0`). The head always does — it
                      samples one position — so a model differing by its head
                      alone would show a second row with nothing to say. */}
                  {showPrefillActive ? (
                    <div id="prefill-active-params-row" className="flex justify-between items-center">
                      <span className="text-gray-600" title="Parameters of the main model each further prompt token touches in prefill, once the prompt is past the stated length: the layers that must run on every prompt position. Layers that keep only a window run on the prompt's last positions, and the head samples only the last one, so neither grows with the prompt and neither is counted here. This is the floor an engine could reach; engines commonly run every layer on every position.">
                        Active per prompt token (prefill)
                        {dash.prefill_full_from ? <span className="text-xs text-gray-400 ml-1">per token past {dash.prefill_full_from.toLocaleString()} prompt tokens</span> : null}
                      </span>
                      <span id="prefill-active-params" className="text-lg font-semibold text-orange-600">{formatNumber(dash.main_prefill_active_params || 0)}</span>
                    </div>
                  ) : null}
                  <div id="main-aux-rows" className={`${Object.values(dash.aux_params || {}).some((n) => n > 0) ? "" : "hidden "}space-y-1 pt-1 border-t border-dashed border-gray-200`}>
                    <div id="aux-rows" className="space-y-0.5">
                      {Object.keys(dash.aux_params || {}).filter((name) => ((dash.aux_params || {})[name] || 0) > 0).map((name) => {
                        const p = (dash.aux_params || {})[name] || 0;
                        const a = (dash.aux_active_params || {})[name] || 0;
                        const perItem = (dash.encoder_flops_per_item || {})[name];
                        const isEncoder = name in (dash.encoder_params || {});
                        return (
                          <div key={name} className="text-xs">
                            <div className="flex justify-between">
                              <span className="text-gray-400" title={isEncoder
                                ? "An encoder tower: runs in its own encode phase, not counted in Total / Active"
                                  + (perItem ? `\n\nFLOPs per item${perItem.variables.length ? ` (${perItem.variables.join(", ")})` : ""}: ${perItem.per_item_readable || perItem.per_item}` : "")
                                : "An auxiliary stack (MTP / speculative head): not counted in Total / Active"}>+ {name}{isEncoder ? " encoder" : ""}</span>
                              <span className="mono text-gray-500">
                                {formatNumber(p)}
                                {a > 0 && a < p ? (
                                  <>
                                    {" "}
                                    <span className="text-gray-400">({formatNumber(a)} active)</span>
                                  </>
                                ) : null}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500" title="Every parameter the checkpoint stores: the main model, MTP heads and encoders">Checkpoint total</span>
                      <span id="checkpoint-total-params" className="mono text-gray-800">{dash.total_params_formatted}</span>
                    </div>
                  </div>
                  <div className="h-px bg-gray-200" />
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Embeddings</span>
                    <span id="embedding-params" className="mono text-gray-700">{formatNumber(dash.embedding_params)}</span>
                  </div>
                  {dash.lookup_table_params ? (
                    <div id="lookup-table-params-row" className="flex justify-between text-sm">
                      <span className="text-gray-500" title="Tables a token gathers a few rows from (n-gram memories, hashed embeddings), other than the input embedding">Lookup tables</span>
                      <span id="lookup-table-params" className="mono text-gray-700">{formatNumber(dash.lookup_table_params)}</span>
                    </div>
                  ) : null}
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Transformer</span>
                    <span id="transformer-params" className="mono text-gray-700">{formatNumber(dash.non_embedding_params)}</span>
                  </div>
                  <div id="hf-params-row" className={`${(dash.hf_model_storage_bytes || 0) > 0 ? "" : "hidden "}mt-2 pt-2 border-t border-gray-100`}>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400" title="The checkpoint's parameter tensors, without the quantisation scales listed below">Weights (no scales)</span>
                      <span id="hf-params" className="mono text-gray-500">{formatGiB(dash.hf_model_storage_bytes || 0)}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Quant Metadata
                        <button
                          id="quant-meta-toggle"
                          type="button"
                          data-action="toggle-quant-meta"
                          className={`${(dash.quant_metadata_breakdown || []).length > 0 && (dash.quant_metadata_params || 0) !== 0 ? "" : "hidden "}ml-1 text-blue-500 hover:text-blue-700`}
                          onClick={() => setQuantMetaOpen((v) => !v)}
                        >
                          {quantMetaOpen ? "[hide]" : "[show]"}
                        </button>
                      </span>
                      <span id="hf-quant-metadata" className="mono text-gray-500">{formatGiB(dash.hf_quantization_metadata_bytes || 0)}</span>
                    </div>
                    <div
                      id="quant-meta-breakdown"
                      className={`${quantMetaOpen && (dash.quant_metadata_breakdown || []).length > 0 && (dash.quant_metadata_params || 0) !== 0 ? "" : "hidden "}mt-1 p-2 bg-gray-50 rounded text-xs font-mono text-gray-600 whitespace-pre-wrap`}
                    >
                      {(dash.quant_metadata_breakdown || []).length > 0 && (dash.quant_metadata_params || 0) !== 0
                        ? quantMetaBreakdownText(dash.quant_metadata_breakdown || [], dash.quant_metadata_params || 0, dash.quant_metadata_bytes || 0, dash.quant_metadata_formats || [])
                        : ""}
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Checkpoint Size</span>
                      <span id="hf-checkpoint-total" className="mono text-gray-500">{formatGiB(dash.hf_checkpoint_storage_bytes || ((dash.hf_model_storage_bytes || 0) + (dash.hf_quantization_metadata_bytes || 0)))}</span>
                    </div>
                    <div id="hf-checkpoint-copies" className="text-right"><CheckpointCopiesNote copies={dash.checkpoint_copies} /></div>
                    <div className="mt-2 pt-2 border-t border-gray-100 text-10 uppercase tracking-wide text-gray-300">Checksums</div>
                    <div id="params-diff-row" className="flex justify-between text-xs">
                      <span className="text-gray-400">Weights</span>
                      <span className="flex items-center gap-2">
                        <span
                          id="params-diff"
                          className={`mono ${Math.abs(dash.storage_diff || 0) === 0 || Math.abs(dash.storage_diff_pct || 0) < 0.5 ? "text-green-500" : "text-orange-500"}`}
                        >
                          {Math.abs(dash.storage_diff || 0) > 0
                            ? `${(dash.storage_diff || 0) > 0 ? "+" : ((dash.storage_diff || 0) < 0 ? "-" : "")}${formatGiB(Math.abs(dash.storage_diff || 0))} (${(dash.storage_diff_pct || 0) > 0 ? "+" : ""}${(dash.storage_diff_pct || 0).toFixed(2)}%)`
                            : "exact match"}
                        </span>
                        <button
                          id="custom-refine-btn"
                          type="button"
                          className={`${showRefine ? "" : "hidden "}px-2 py-0.5 text-xs bg-orange-50 text-orange-600 rounded hover:bg-orange-100 transition-colors`}
                          onClick={() => void refineSpec(
                            data?.model_id || modelId,
                            dash.spec_storage_bytes || 0,
                            dash.hf_model_storage_bytes || 0,
                            dash.storage_diff || 0,
                          )}
                        >
                          Refine
                        </button>
                      </span>
                    </div>
                    <div id="metadata-diff-row" className="flex justify-between text-xs">
                      <span className="text-gray-400">Metadata</span>
                      <span
                        id="metadata-diff"
                        className={`mono ${Math.abs(dash.metadata_storage_diff || 0) === 0 || Math.abs(dash.metadata_storage_diff_pct || 0) < 0.5 ? "text-green-500" : "text-orange-500"}`}
                      >
                        {Math.abs(dash.metadata_storage_diff || 0) > 0
                          ? `${(dash.metadata_storage_diff || 0) > 0 ? "+" : ((dash.metadata_storage_diff || 0) < 0 ? "-" : "")}${formatGiB(Math.abs(dash.metadata_storage_diff || 0))} (${(dash.metadata_storage_diff_pct || 0) > 0 ? "+" : ""}${(dash.metadata_storage_diff_pct || 0).toFixed(2)}%)`
                          : "exact match"}
                      </span>
                    </div>
                    <div id="all-tensor-diff-row" className="flex justify-between text-xs">
                      <span className="text-gray-400">All tensors</span>
                      <span
                        id="all-tensor-diff"
                        className={`mono ${Math.abs(dash.all_tensor_diff || 0) === 0 || Math.abs(dash.all_tensor_diff_pct || 0) < 0.5 ? "text-green-500" : "text-orange-500"}`}
                      >
                        {Math.abs(dash.all_tensor_diff || 0) > 0
                          ? `${(dash.all_tensor_diff || 0) > 0 ? "+" : ((dash.all_tensor_diff || 0) < 0 ? "-" : "")}${formatGiB(Math.abs(dash.all_tensor_diff || 0))} (${(dash.all_tensor_diff_pct || 0) > 0 ? "+" : ""}${(dash.all_tensor_diff_pct || 0).toFixed(2)}%)`
                          : "exact match"}
                      </span>
                    </div>
                    {dash.checkpoint_anchored && dash.hf_index_total_size ? (
                      <div id="measured-diff-row" className="flex justify-between text-xs">
                        <span className="text-gray-400">Index total</span>
                        <span
                          id="measured-diff"
                          className={`mono ${Math.abs(dash.measured_tensor_diff || 0) === 0 || Math.abs(dash.measured_tensor_diff_pct || 0) < 0.5 ? "text-green-500" : "text-orange-500"}`}
                        >
                          {Math.abs(dash.measured_tensor_diff || 0) > 0
                            ? `${(dash.measured_tensor_diff || 0) > 0 ? "+" : ((dash.measured_tensor_diff || 0) < 0 ? "-" : "")}${formatGiB(Math.abs(dash.measured_tensor_diff || 0))} (${(dash.measured_tensor_diff_pct || 0) > 0 ? "+" : ""}${(dash.measured_tensor_diff_pct || 0).toFixed(2)}%)`
                            : "exact match"}
                        </span>
                      </div>
                    ) : null}
                  </div>
                </div>
                <div
                  id="moe-section"
                  className={`${dash.moe_enabled ? "" : "hidden "}card p-4 sm:p-6 bg-gradient-to-b from-orange-50 to-amber-50 border border-orange-200`}
                >
                  <h3 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
                    Mixture of Experts (MoE)
                  </h3>
                  <div className="grid grid-cols-1 gap-3">
                    <div className="metric-card">
                      <div className="text-sm text-gray-600">MoE Layers</div>
                      <div id="moe-layers" className="text-2xl font-bold text-orange-700">{dash.moe_layers}</div>
                    </div>
                    <div className="metric-card">
                      <div className="text-sm text-gray-600">Active Experts / Token</div>
                      <div id="moe-active-experts" className="text-2xl font-bold text-orange-700">{dash.moe_active_experts}</div>
                    </div>
                    <div className="metric-card">
                      <div className="text-sm text-gray-600">Total Experts</div>
                      <div id="moe-total-experts" className="text-2xl font-bold text-orange-700">{dash.moe_total_experts}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="card mb-8">
                <div className="border-b border-gray-200">
                  <nav className="flex gap-4 lg:gap-8 px-4 lg:px-6 overflow-x-auto">
                    <button
                      type="button"
                      data-action="switch-tab"
                      data-tab="dashboard"
                      id="tab-dashboard"
                      className={tab === "dashboard" ? "tab-active py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0" : "py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0 text-gray-500 hover:text-gray-700"}
                      onClick={() => setTab("dashboard")}
                    >
                      Dashboard
                    </button>
                    <button
                      type="button"
                      data-action="switch-tab"
                      data-tab="diagram"
                      id="tab-diagram"
                      className={tab === "diagram" ? "tab-active py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0" : "py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0 text-gray-500 hover:text-gray-700"}
                      onClick={() => setTab("diagram")}
                    >
                      Architecture Diagram
                    </button>
                    <button
                      type="button"
                      data-action="switch-tab"
                      data-tab="modules"
                      id="tab-modules"
                      className={tab === "modules" ? "tab-active py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0" : "py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0 text-gray-500 hover:text-gray-700"}
                      onClick={() => setTab("modules")}
                    >
                      Raw Modules
                    </button>
                    <button
                      type="button"
                      data-action="switch-tab"
                      data-tab="raw"
                      id="tab-raw"
                      className={tab === "raw" ? "tab-active py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0" : "py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0 text-gray-500 hover:text-gray-700"}
                      onClick={() => setTab("raw")}
                    >
                      Raw Config
                    </button>
                    <button
                      type="button"
                      data-action="switch-tab"
                      data-tab="forward"
                      id="tab-forward"
                      className={tab === "forward" ? "tab-active py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0" : "py-4 px-2 font-medium text-sm whitespace-nowrap shrink-0 text-gray-500 hover:text-gray-700"}
                      onClick={() => setTab("forward")}
                    >
                      Forward Source
                    </button>
                  </nav>
                </div>
                <div className="p-6">
                  <div id="content-dashboard" className={tab === "dashboard" ? undefined : "hidden"}>
                    {data ? <DashboardStats data={data} /> : <div id="dashboard-stats" className="space-y-6" />}
                    {data?.composition ? (
                      <CompositionTable data={data} composition={data.composition} onOpen={openInDiagram} />
                    ) : null}
                  </div>
                  <div id="content-diagram" className={tab === "diagram" ? undefined : "hidden"}>
                    <ArchitectureView arch={arch} blockGraph={data?.block_graph} />
                  </div>
                  <div id="content-modules" className={`${tab === "modules" ? "" : "hidden "}space-y-6`}>
                    <div className="border border-gray-200 rounded-lg">
                      <div className="flex justify-between items-center px-4 py-3 bg-gray-50 border-b border-gray-200 rounded-t-lg">
                        <h3 className="font-semibold text-gray-900">Raw PyTorch Modules (print(model))</h3>
                        <button
                          type="button"
                          data-action="copy-model-str"
                          className="text-amber-600 hover:text-amber-700 text-sm"
                          onClick={() => {
                            if (!data) return;
                            if (data.model_str) copyText(data.model_str, "Model structure copied to clipboard!");
                          }}
                        >
                          Copy to Clipboard
                        </button>
                      </div>
                      <RawModulesPrint modelStr={data?.model_str} />
                    </div>
                  </div>
                  <div id="content-raw" className={`${tab === "raw" ? "" : "hidden "}space-y-8`}>
                    <div className="border border-gray-200 rounded-lg">
                      <div className="flex justify-between items-center px-4 py-3 bg-gray-50 border-b border-gray-200 rounded-t-lg">
                        <h3 className="font-semibold text-gray-900">Raw Config (Original config.json)</h3>
                        <button
                          type="button"
                          data-action="copy-raw-config"
                          className="text-amber-600 hover:text-amber-700 text-sm"
                          onClick={() => {
                            if (data) copyText(JSON.stringify(data.config_json, null, 2), "Raw config copied to clipboard!");
                          }}
                        >
                          Copy to Clipboard
                        </button>
                      </div>
                      <RawConfigPre id="raw-config-container" text={data ? JSON.stringify(data.config_json, null, 2) : ""} className="bg-slate-50 text-emerald-800 p-4 overflow-x-auto mono text-sm m-0" />
                    </div>
                    <div className="border border-gray-200 rounded-lg">
                      <div className="flex justify-between items-center px-4 py-3 bg-gray-50 border-b border-gray-200 rounded-t-lg">
                        <h3 className="font-semibold text-gray-900">Processed Config (with defaults)</h3>
                        <button
                          type="button"
                          data-action="copy-processed-config"
                          className="text-amber-600 hover:text-amber-700 text-sm"
                          onClick={() => {
                            if (data) copyText(JSON.stringify(data.config_processed, null, 2), "Processed config copied to clipboard!");
                          }}
                        >
                          Copy to Clipboard
                        </button>
                      </div>
                      <RawConfigPre id="processed-config-container" text={data ? JSON.stringify(data.config_processed, null, 2) : ""} className="bg-slate-50 text-sky-800 p-4 overflow-x-auto mono text-sm m-0" />
                    </div>
                  </div>
                  <div id="content-forward" className={`${tab === "forward" ? "" : "hidden "}space-y-4`}>
                    <ForwardSourcePanel fs={data?.forward_source} />
                  </div>
                </div>
              </div>
          </div>

          <div id="error" className={`${error ? "" : "hidden "}card p-6 bg-red-50 border border-red-200`}>
            <div className="flex items-center gap-3">
              <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <div>
                <h3 className="font-semibold text-red-900">Analysis Failed</h3>
                <p id="error-message" className="text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
      </div>

      <div id="inspect-mode-compare" className={mode === "compare" ? undefined : "hidden"}>
        <section className="card p-4 sm:p-6 mb-6">
          <h2 className="text-xl font-semibold mb-1">Compare Models</h2>
          <p className="text-xs text-gray-500 mb-4">2–5 models side by side.</p>
          <div className="mb-2 flex items-baseline justify-between">
            <label className="text-sm font-medium text-gray-700">Models <span className="text-xs text-gray-400 ml-2">(2 – 5)</span></label>
            <button
              type="button"
              id="compare-add-model"
              className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg text-gray-700 hover:bg-amber-50 hover:border-amber-300"
              disabled={cmpModels.length >= 5}
              onClick={() => setCmpModels((ms) => ms.length < 5 ? [...ms, ""] : ms)}
            >
              + Add model
            </button>
          </div>
          <div id="compare-model-list" className="space-y-2" ref={cmpListRef}>
            {cmpModels.map((m, i) => {
              const items = filterAndSortModels(models, m);
              return (
              <div key={i} className="flex gap-2 items-center">
                <span className="w-6 text-xs text-center">
                  <span className="font-mono" style={{ color: CC_COLORS[i % CC_COLORS.length] }}>{i + 1}</span>
                </span>
                <div className="relative flex-1">
                  <input
                    type="text"
                    id={`compare-model-${i}`}
                    autoComplete="off"
                    data-action="cc-slot-input"
                    data-index={i}
                    className="w-full pr-9 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-amber-400 focus:border-amber-500"
                    placeholder="e.g., meta-llama/Llama-3.3-70B-Instruct"
                    value={m}
                    onChange={(e) => {
                      const v = e.target.value;
                      setCmpModels((ms) => ms.map((x, j) => (j === i ? v : x)));
                      setCmpDropOpen(i);
                    }}
                    onFocus={() => setCmpDropOpen(i)}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 w-9 flex items-center justify-center text-gray-400 hover:text-amber-600"
                    data-action="cc-toggle-dropdown"
                    data-index={i}
                    title="Show supported models"
                    onClick={(e) => { e.preventDefault(); setCmpDropOpen((o) => (o === i ? null : i)); }}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" /></svg>
                  </button>
                  <div
                    id={`compare-model-dropdown-${i}`}
                    className={`${cmpDropOpen === i ? "" : "hidden "}cc-row-dropdown absolute left-0 right-0 top-full mt-1 max-h-64 overflow-y-auto bg-white border border-gray-200 rounded-lg shadow-lg z-20`}
                  >
                    <ModelOptions
                      items={items}
                      action="cc-select-model"
                      index={i}
                      loading={modelsLoading}
                      onSelect={(id) => {
                        setCmpModels((ms) => ms.map((x, j) => (j === i ? id : x)));
                        setCmpDropOpen(null);
                      }}
                    />
                  </div>
                </div>
                <button
                  type="button"
                  className={`text-xs text-gray-400 hover:text-red-600 px-2 py-1 ${cmpModels.length > 2 ? "" : "invisible"}`}
                  data-action="cc-remove-slot"
                  data-index={i}
                  title="Remove model"
                  onClick={() => {
                    if (cmpModels.length <= 2) return;
                    setCmpModels((ms) => ms.filter((_, j) => j !== i));
                    setCmpDropOpen(null);
                  }}
                >
                  ✕
                </button>
              </div>
              );
            })}
          </div>
          <div className="mt-4 flex flex-wrap gap-3 items-center">
            <button
              type="button"
              data-action="run-compare"
              className="px-5 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg font-medium transition"
              onClick={() => void runCompare()}
            >
              Compare
            </button>
            <span id="compare-status" className="text-sm text-gray-500">{cmpStatus}</span>
            <label className="text-xs text-gray-600 flex items-center gap-1.5 whitespace-nowrap" htmlFor="compare-include-encoders">
              <input
                id="compare-include-encoders"
                type="checkbox"
                className="accent-amber-500"
                checked={includeEncoders}
                onChange={(e) => setIncludeEncoders(e.target.checked)}
              />
              Include encoder weights
            </label>
            <div className="flex-1" />
            <label className="text-xs text-gray-500 whitespace-nowrap" htmlFor="inspect-to-single">Open one in single view:</label>
            <select
              id="inspect-to-single"
              className="max-w-full text-xs px-2 py-1.5 border border-gray-300 rounded-lg bg-white text-gray-700 hover:bg-gray-50"
              title="Pick any model to deep-dive in the single-model inspector."
              defaultValue=""
              onChange={(e) => {
                const v = e.target.value.trim();
                e.target.value = "";
                if (!v) return;
                setModelId(v);
                setMode("single");
                // The single view shows what it analyzed; a pick that is not
                // the model already shown is analyzed now.
                if (v !== data?.model_id) void run(v);
              }}
            >
              <option value="">‹ Back to single…</option>
              {cmpModels.map((m, i) => {
                const id = m.trim();
                return id ? <option key={`${i}-${id}`} value={id}>{id}</option> : null;
              })}
            </select>
          </div>
        </section>
        <div id="compare-results" className={`${cmp?.success ? "" : "hidden "}fade-in`}>
          <div id="compare-headers" className="flex gap-4 overflow-x-auto pb-2 mb-6">
            {(cmp?.results || []).map((r, i) => {
              const badges = ccHeaderBadges(r?.dashboard);
              return (
                <div
                  key={i}
                  className="card p-4 min-w-[240px] flex-shrink-0 border-t-4"
                  style={{ borderTopColor: CC_COLORS[i % CC_COLORS.length] }}
                >
                  <div className="text-xs text-gray-400 uppercase tracking-wide">Model {i + 1}</div>
                  <div className="text-sm font-semibold text-gray-900 break-all">{ccModelName(r, cmp?.specs?.[i], i)}</div>
                  <div className="flex flex-wrap gap-1 mt-1 text-xs text-gray-500">
                    {badges.map((x) => (
                      <span key={x} className="px-2 py-0.5 bg-gray-100 rounded">{x}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
          <section className="card p-5 mb-6">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h8m-8 6h16" /></svg>
              Key Differences
            </h3>
            <div id="compare-diff-content" className="space-y-4">
              <CompareDiffContent diff={cmp?.diff || {}} results={cmp?.results || []} specs={cmp?.specs || []} scope={weightScope} />
            </div>
          </section>
          <section className="card p-5 mb-6">
            <h3 className="font-semibold text-gray-900 mb-3">Dashboards</h3>
            <div id="compare-dashboards" className="flex gap-6 overflow-x-auto pb-2">
              {(cmp?.results || []).map((r, i) => (
                <div key={i} className="min-w-[300px] flex-1 basis-0">
                  <div
                    className="text-xs font-semibold mb-2 break-all"
                    style={{ color: CC_COLORS[i % CC_COLORS.length] }}
                  >
                    {ccShortName(r, cmp?.specs?.[i], i)}
                  </div>
                  <div id={`compare-dashboard-${i}`} className="space-y-3">
                    <CompareDashboard data={r} scope={weightScope} />
                  </div>
                </div>
              ))}
            </div>
          </section>
          <section className="card p-5">
            <div className="flex flex-wrap items-start justify-between gap-2 mb-2">
              <h3 className="font-semibold text-gray-900">Representative Block Layer</h3>
              {(() => {
                const { picks, dropped } = representativePicks(cmp?.results || []);
                return picks.length ? (
                  <div className="flex flex-col items-end gap-0.5">
                    <a id="compare-blocks-playground" href={playgroundCompareHref(picks)}
                      title="Open Playground's module comparison with these blocks; switch any column to another block of its model there"
                      className="text-xs px-2.5 py-1 rounded-full border border-amber-500 text-amber-700 hover:bg-amber-50">
                      Compare these blocks in Playground →
                    </a>
                    {dropped.length ? (
                      <span data-rep-dropped className="text-11 text-gray-500"
                        title={dropped.map((p) => `${p.name} @ ${p.model_id}`).join("\n")}>
                        Playground compares {PLAYGROUND_COLUMN_CAP} columns at most: {dropped.length} more
                        block{dropped.length === 1 ? "" : "s"} left out (add them there)
                      </span>
                    ) : null}
                  </div>
                ) : null;
              })()}
            </div>
            <ul id="compare-blocks-basis" className="text-xs text-gray-600 space-y-0.5 mb-3">
              {(cmp?.results || []).map((r, i) => (
                <RepresentativeBasis key={i} result={r} spec={cmp?.specs?.[i]} i={i} />
              ))}
            </ul>
            <div id="compare-blocks" className="min-w-0">
              <GraphShell
                rows={(cmp?.results || []).map((r, i) => {
                  const rt = r?.representative_template;
                  const graph = rt?.name && r?.block_graph ? r.block_graph[rt.name] : undefined;
                  const layers = (rt?.count != null && rt?.n_layers != null)
                    ? ` · ${rt.count} of ${rt.n_layers} layers` : "";
                  return {
                    label: ccShortName(r, cmp?.specs?.[i], i) + layers,
                    nodes: graph?.nodes || [],
                    edges: graph?.edges || [],
                  };
                })}
                diff={cmp?.graph_diff}
                colors={[...CC_COLORS]}
              />
            </div>
          </section>
        </div>
      </div>

      {copied ? <div className="fixed bottom-4 right-4 z-40 bg-gray-900 text-white text-xs px-3 py-2 rounded-lg">{copied}</div> : null}

      <div id="generate-overlay" className={`${overlay && overlayKind === "generate" ? "" : "hidden "}fixed inset-0 bg-white/80 z-50 flex items-center justify-center`}>
        <div className="bg-white shadow-xl rounded-xl p-6 max-w-lg w-full mx-4">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="loading-spinner" style={{ width: 24, height: 24, borderWidth: 3 }} />
              <div className="min-w-0">
                <div className="font-semibold text-lg">Spec Generation</div>
                <div className="text-gray-400 text-xs truncate">{data?.model_id || modelId}</div>
              </div>
            </div>
            <AgentChip cli={genAgent.cli} model={genAgent.model} />
          </div>
          <div className="text-sm mb-3 text-amber-700">No architecture spec matched. Discovering a new structure via coding agent.</div>
          <div id="generate-log" className="bg-gray-900 text-green-400 text-xs font-mono p-3 rounded-lg h-48 overflow-y-auto whitespace-pre-wrap">{genLog}</div>
          <div id="generate-usage" className="mt-3 text-xs text-gray-500">{genUsage}</div>
          <div className="mt-3 flex items-center justify-between">
            <OverlayStatusSlot id="generate-status" text={genStatus} tone={genStatusTone} />
            <button
              id="generate-stop-btn"
              type="button"
              data-action="stop-spec-generation"
              data-model={data?.model_id || modelId}
              disabled={stopDisabled}
              className="px-3 py-1 text-xs bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              onClick={() => {
                setStopDisabled(true);
                stopFlag.current = true;
                const mid = (data?.model_id || modelId).trim();
                void authContext().then((ctx) => stopSpecJob(mid, ctx.loggedIn, "generate")).then(() => {
                  setGenStatus("Stopped by user");
                  setGenStatusTone("orange");
                  setTimeout(() => {
                    overlayOpen.current = false;
                    setOverlay(false);
                    setBanner((b) => (b.kind !== "fail" ? { kind: "missing" } : b));
                  }, 1500);
                }).catch((e) => {
                  stopFlag.current = false;
                  setStopDisabled(false);
                  setGenStatus(`Stop failed: ${e instanceof Error ? e.message : String(e)}`);
                  setGenStatusTone("red600");
                });
              }}
            >
              Stop
            </button>
          </div>
        </div>
      </div>

      <div id="refine-overlay" className={`${overlay && overlayKind === "refine" ? "" : "hidden "}fixed inset-0 bg-white/80 z-50 flex items-center justify-center`}>
        <div className="bg-white shadow-xl rounded-xl p-6 max-w-lg w-full mx-4">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="loading-spinner" style={{ width: 24, height: 24, borderWidth: 3 }} />
              <div className="min-w-0">
                <div className="font-semibold text-lg">Spec Refinement</div>
                <div className="text-gray-400 text-xs truncate">{data?.model_id || modelId}</div>
              </div>
            </div>
            <AgentChip cli={genAgent.cli} model={genAgent.model} />
          </div>
          <div className="flex justify-between text-sm mb-3">
            <span className="text-gray-500">Spec: {formatGiB(dash?.spec_storage_bytes || 0)}</span>
            <span className="text-gray-500">HF: {formatGiB(dash?.hf_model_storage_bytes || 0)}</span>
            <span className="text-orange-500 font-medium">diff: {(dash?.storage_diff || 0) > 0 ? "+" : (dash?.storage_diff || 0) < 0 ? "-" : ""}{formatGiB(Math.abs(dash?.storage_diff || 0))}</span>
          </div>
          <div id="refine-log" className="bg-gray-900 text-green-400 text-xs font-mono p-3 rounded-lg h-48 overflow-y-auto whitespace-pre-wrap">{genLog}</div>
          <div id="refine-usage" className="mt-3 text-xs text-gray-500">{genUsage}</div>
          <div className="mt-3 flex items-center justify-between">
            <OverlayStatusSlot id="refine-status" text={genStatus} tone={genStatusTone} />
            <button
              id="refine-stop-btn"
              type="button"
              data-action="stop-spec-refinement"
              data-model={data?.model_id || modelId}
              disabled={stopDisabled}
              className="px-3 py-1 text-xs bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              onClick={() => {
                setStopDisabled(true);
                stopFlag.current = true;
                const mid = (data?.model_id || modelId).trim();
                void authContext().then((ctx) => stopSpecJob(mid, ctx.loggedIn, "refine")).then(() => {
                  setGenStatus("Stopped by user");
                  setGenStatusTone("orange");
                  setTimeout(() => {
                    overlayOpen.current = false;
                    setOverlay(false);
                  }, 1500);
                }).catch((e) => {
                  stopFlag.current = false;
                  setStopDisabled(false);
                  setGenStatus(`Stop failed: ${e instanceof Error ? e.message : String(e)}`);
                  setGenStatusTone("red600");
                });
              }}
            >
              Stop
            </button>
          </div>
        </div>
      </div>

      <div id="fix-overlay" className={`${overlay && overlayKind === "fix" ? "" : "hidden "}fixed inset-0 bg-white/80 z-50 flex items-center justify-center`}>
        <div className="bg-white shadow-xl rounded-xl p-6 max-w-lg w-full mx-4">
          <div className="flex items-start justify-between gap-3 mb-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="loading-spinner" style={{ width: 24, height: 24, borderWidth: 3 }} />
              <div className="min-w-0">
                <div className="font-semibold text-lg">Agent Fix</div>
                <div className="text-gray-400 text-xs truncate">{data?.model_id || modelId}</div>
              </div>
            </div>
            <AgentChip cli={genAgent.cli} model={genAgent.model} />
          </div>
          <div className="text-sm mb-3 text-indigo-700">Saved spec needs a schema declaration. An agent is writing it.</div>
          <div id="fix-log" className="bg-gray-900 text-green-400 text-xs font-mono p-3 rounded-lg h-48 overflow-y-auto whitespace-pre-wrap">{genLog}</div>
          <div id="fix-usage" className="mt-3 text-xs text-gray-500">{genUsage}</div>
          <div className="mt-3 flex items-center justify-between">
            <OverlayStatusSlot id="fix-status" text={genStatus} tone={genStatusTone} />
            <button
              id="fix-stop-btn"
              type="button"
              data-action="stop-spec-fix"
              data-model={data?.model_id || modelId}
              disabled={stopDisabled}
              className="px-3 py-1 text-xs bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              onClick={() => {
                setStopDisabled(true);
                stopFlag.current = true;
                const mid = (data?.model_id || modelId).trim();
                void authContext().then((ctx) => stopSpecJob(mid, ctx.loggedIn, "fix", autoFixFor.current)).then(() => {
                  setGenStatus("Stopped by user");
                  setGenStatusTone("orange");
                  setTimeout(() => {
                    overlayOpen.current = false;
                    setOverlay(false);
                  }, 1500);
                }).catch((e) => {
                  stopFlag.current = false;
                  setStopDisabled(false);
                  setGenStatus(`Stop failed: ${e instanceof Error ? e.message : String(e)}`);
                  setGenStatusTone("red600");
                });
              }}
            >
              Stop
            </button>
          </div>
        </div>
      </div>

      {resumePrompt ? (
        <div
          id="resume-draft-dialog"
          className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4"
          onClick={() => setResumePrompt(null)}
        >
          <div
            className="bg-white rounded-xl shadow-xl p-6 max-w-md w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="font-semibold text-lg mb-2">Continue the last generation?</div>
            <div className="text-sm text-gray-600">
              An earlier generation of{" "}
              <code className="font-mono text-xs bg-gray-100 px-1 rounded break-all">{resumePrompt}</code>{" "}
              stopped before it finished and kept its draft. Continue where it stopped, or start
              over and discard the draft.
            </div>
            <div className="mt-5 flex flex-wrap justify-end gap-2">
              <button
                type="button"
                data-action="cancel-spec-generation"
                className="px-4 py-2 text-sm text-gray-500 hover:text-gray-700 transition"
                onClick={() => setResumePrompt(null)}
              >
                Cancel
              </button>
              <button
                type="button"
                data-action="restart-spec-generation"
                className="px-4 py-2 bg-white text-amber-700 border border-amber-300 text-sm rounded-lg hover:bg-amber-50 transition"
                onClick={() => {
                  const mid = resumePrompt;
                  setResumePrompt(null);
                  void generateSpec(mid, "discard");
                }}
              >
                Start over
              </button>
              <button
                type="button"
                data-action="resume-spec-generation"
                className="px-4 py-2 bg-amber-600 text-white text-sm rounded-lg hover:bg-amber-700 transition"
                onClick={() => {
                  const mid = resumePrompt;
                  setResumePrompt(null);
                  void generateSpec(mid, "resume");
                }}
              >
                Continue last generation
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

/** vanilla `toggleArchPanel` — lazy ELK on first open via panel → template. */
function ArchExpandDag({
  panelId,
  template,
  blockGraph,
  extraBtn,
  open,
  onToggle,
}: {
  panelId: string;
  template?: string;
  blockGraph: BlockGraph | undefined;
  extraBtn?: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  const [rendered, setRendered] = useState(false);
  const graph = template && blockGraph ? blockGraph[template] : null;
  return (
    <>
      <button
        type="button"
        className={`arch-expand-btn${open ? " is-open" : ""}`}
        data-action="toggle-arch-panel"
        data-panel={panelId}
        aria-expanded={open}
        aria-controls={panelId}
        aria-label={open ? "Collapse computation" : "Expand computation"}
        title={open ? "Collapse computation" : "Expand computation"}
        style={extraBtn ? { fontSize: 10, width: 20, height: 20 } : undefined}
        onClick={() => {
          if (!open) setRendered(true);
          onToggle();
        }}
      >
        {open ? "−" : "+"}
      </button>
      <div id={panelId} className={`arch-detail-panel${open ? " open" : ""}`}>
        <div
          id={`${panelId}-dag`}
          className="dag-container"
          style={{ padding: extraBtn ? 12 : 16 }}
        >
          {rendered && blockGraph ? (
            graph ? (
              <GraphShell rows={[{
                label: template || "",
                nodes: graph.nodes || [],
                edges: graph.edges || [],
              }]} />
            ) : (
              <div className="text-gray-500 text-sm p-4">No block graph available for this layer.</div>
            )
          ) : null}
        </div>
      </div>
    </>
  );
}

/** Which variant (by position) runs each layer, for a stream map's band;
 *  undefined when the card has no variants. */
function variantBands(variants: ArchLayer[]): Record<number, number> | undefined {
  if (!variants.length) return undefined;
  const out: Record<number, number> = {};
  variants.forEach((v, i) => (v.layers || []).forEach((l) => { out[l] = i; }));
  return out;
}

function ArchLayerBox({
  layer,
  panelId,
  blockGraph,
  variant,
  swatch,
}: {
  layer: ArchLayer;
  panelId: string;
  blockGraph: BlockGraph | undefined;
  // The colour a kind card's stream map draws this variant's layers in.
  swatch?: string;
  // `member`: one template of a kind card — its name, its layers and which
  // streams it writes or reuses; where each read comes from is a tooltip.
  variant: "main" | "cycle" | "member" | "extra" | "extra-cycle";
}) {
  const [open, setOpen] = useState(false);
  const variants = layer.variants || [];
  const prod = layer.stream_produces || [];
  const reads = layer.stream_reads || [];
  const onStream = prod.length || reads.length;
  const extra = variant === "extra";
  const cycleInner = variant === "cycle" || variant === "member" || variant === "extra-cycle";
  const style: CSSProperties | undefined = cycleInner
    ? {
        margin: 0,
        maxWidth: "none",
        padding: "10px 16px",
        ...(variant !== "extra-cycle" && onStream ? { borderLeft: "2px solid #f59e0b" } : {}),
        ...(variant === "member" ? { padding: "6px 36px 6px 10px", textAlign: "left" } : {}),
      }
    : extra
      ? { borderStyle: "dashed", opacity: 0.85, padding: "8px 16px" }
      : undefined;
  return (
    <div
      className={`arch-layer type-${layer.type}${open ? " is-expanded" : ""}`}
      style={style}
      data-arch-template={layer.template || undefined}
    >
      <div className={variant === "member" ? undefined : "text-center"}>
        {variant === "member" ? (
          <div
            title={Object.entries(layer.stream_sources || {}).map(([s, src]) =>
              `reads ${s} from ${src.declared}: ${src.reads.map(([at, from]) => `${at}←${from}`).join(", ")}`).join("\n") || undefined}
          >
            <div className="text-xs font-medium" style={{ overflowWrap: "anywhere" }} title={layer.name}>
              {swatch ? (
                <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: swatch, marginRight: 6 }} />
              ) : null}
              {layer.name}
            </div>
            {(layer.slots || []).map((s) => (
              <div key={s.slot} className="text-11 arch-variant-slot" title={s.description || undefined} data-variant-slot={s.slot}>
                <span className="font-medium text-gray-700 mono">{s.slot} = {s.component}</span>
                {s.description ? <div className="text-gray-500">{s.description}</div> : null}
              </div>
            ))}
            <div className="text-10 text-gray-500" style={{ marginTop: 2 }}>
              {layer.detail}
              {prod.length ? <span style={{ color: "#d97706", marginLeft: 8 }}>▲ writes {prod.join(", ")}</span> : null}
              {reads.length ? <span style={{ color: "#b45309", marginLeft: 8 }}>↳ reuses {reads.join(", ")}</span> : null}
            </div>
          </div>
        ) : variant === "cycle" ? (
          <div className="font-semibold text-sm">
            {swatch ? (
              <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: swatch, marginRight: 6 }} />
            ) : null}
            {layer.name}
            {prod.map((s) => (
              <span
                key={`w-${s}`}
                className="text-10 px-1.5 py-0.5 rounded"
                style={{ background: "rgba(245,158,11,0.20)", color: "#d97706", marginLeft: 6, fontWeight: 600 }}
              >
                ▲ writes {s}
              </span>
            ))}
            {reads.map((s) => (
              <span
                key={`r-${s}`}
                className="text-10 px-1.5 py-0.5 rounded"
                style={{ background: "rgba(245,158,11,0.10)", color: "#d97706", marginLeft: 6 }}
              >
                ↳ reuses {s}
              </span>
            ))}
            {Object.entries(layer.stream_sources || {}).map(([s, src]) => (
              <div
                key={`src-${s}`}
                className="text-10 font-normal"
                style={{ color: "#d97706" }}
                data-stream-source={s}
              >
                ↳ reads {s} from {src.declared}:{" "}
                {src.reads.slice(0, 6).map(([at, from]) => `${at}←${from}`).join(", ")}
                {src.reads.length > 6 ? ", …" : ""}
              </div>
            ))}
          </div>
        ) : extra ? (
          <div className="text-xs font-medium">{layer.name}</div>
        ) : (
          <div className="font-semibold text-sm">{layer.name}</div>
        )}
        {layer.detail && variant !== "member" ? (
          extra
            ? <div className="text-xs text-gray-500">{layer.detail}</div>
            : <div className="text-xs text-gray-500 mt-1">{layer.detail}</div>
        ) : null}
      </div>
      {layer.stream_map && layer.layers ? (
        <StreamMap layers={layer.layers} rows={layer.stream_map} bands={variantBands(variants)} />
      ) : null}
      {variants.length ? (
        <div className="arch-variants" style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
          {variants.map((v, vi) => (
            <ArchLayerBox
              key={`${v.template}-${vi}`}
              layer={v}
              panelId={`${panelId}-v${vi}`}
              blockGraph={blockGraph}
              variant="member"
              swatch={VARIANT_COLORS[vi % VARIANT_COLORS.length]}
            />
          ))}
        </div>
      ) : null}
      {layer.expandable && !variants.length ? (
        <ArchExpandDag
          panelId={panelId}
          template={layer.template}
          blockGraph={blockGraph}
          extraBtn={extra}
          open={open}
          onToggle={() => setOpen((o) => !o)}
        />
      ) : null}
    </div>
  );
}

function ArchCycleContainer({
  layer,
  indexPrefix,
  blockGraph,
  extra,
}: {
  layer: ArchLayer;
  indexPrefix: string;
  blockGraph: BlockGraph | undefined;
  extra?: boolean;
}) {
  const cycleStreams = [...new Set((layer.cycle_blocks || []).flatMap(
    (s) => [...(s.stream_produces || []), ...(s.stream_reads || [])],
  ))];
  const wins: Record<string, number | string> = {};
  for (const s of layer.cycle_blocks || []) Object.assign(wins, s.stream_windows || {});
  return (
    <div
      className="arch-cycle-container"
      style={{
        border: "1px dashed rgba(124,58,237,0.45)",
        borderRadius: 14,
        padding: "18px 14px 14px",
        background: "rgba(124,58,237,0.03)",
        position: "relative",
        ...(extra ? { opacity: 0.9 } : {}),
      }}
    >
      <div
        style={{
          position: "absolute",
          top: -10,
          left: "50%",
          transform: "translateX(-50%)",
          background: "#ffffff",
          padding: "0 12px",
          fontSize: 11,
          color: "#7c3aed",
          letterSpacing: 0.5,
          fontWeight: 600,
        }}
      >
        {(layer.name || "").toUpperCase()}
      </div>
      {layer.detail ? (
        <div className="text-center text-11 text-gray-500" style={{ margin: "2px 0 10px" }}>{layer.detail}</div>
      ) : null}
      {!extra && cycleStreams.length ? (
        <div className="text-center text-10" style={{ color: "#d97706", margin: "-4px 0 8px" }}>
          ⟳ shared stream: {cycleStreams.map((s, i) => (
            <Fragment key={s}>
              {i ? " · " : null}
              {s}
              {wins[s] ? <span className="text-gray-500"> (≤{wins[s]} layers back)</span> : null}
            </Fragment>
          ))}{" "}
          <span className="text-gray-500">(computed once, reused across the cycle)</span>
        </div>
      ) : null}
      {layer.stream_map && layer.layers ? (
        <div style={{ margin: "0 0 10px" }}>
          <StreamMap layers={layer.layers} rows={layer.stream_map} bands={variantBands(layer.cycle_blocks || [])} />
        </div>
      ) : null}
      {(layer.cycle_blocks || []).map((sub, si) => (
        <Fragment key={`${sub.name}-${si}`}>
          <ArchLayerBox
            layer={sub}
            panelId={`${indexPrefix}-${si}`}
            blockGraph={blockGraph}
            variant={extra ? "extra-cycle" : "cycle"}
            swatch={layer.stream_map ? VARIANT_COLORS[si % VARIANT_COLORS.length] : undefined}
          />
          {si < (layer.cycle_blocks || []).length - 1 ? (
            <div className="arch-connector" style={{ height: 14 }}></div>
          ) : null}
        </Fragment>
      ))}
    </div>
  );
}

/** An auxiliary module (`group:`) as one box of the overview: its layers
 *  in order, each its own card. */
function ArchSectionBox({ section, blockGraph }: { section: ArchSection; blockGraph: BlockGraph | undefined }) {
  const key = (section.id || section.name || "").replace(/[^\w-]/g, "");
  const subLayers = section.sub_layers || [];
  return (
    <div className="arch-section" style={{ border: "1px dashed #cbd5e1", borderRadius: 14, padding: "10px 12px", background: "#fcfcfd" }}>
      <div className="text-center text-sm font-semibold mb-2" style={{ color: "#64748b" }}>
        {section.name} <span className="text-xs font-normal">×{section.repeat}</span>
      </div>
      {subLayers.map((sl, si) => (
        <Fragment key={`${sl.name}-${si}`}>
          {sl.type === "cycle_container" && Array.isArray(sl.cycle_blocks) ? (
            <ArchCycleContainer layer={sl} indexPrefix={`arch-extra-cycle-${key}-${si}`} blockGraph={blockGraph} extra />
          ) : (
            <ArchLayerBox layer={sl} panelId={`arch-extra-${key}-${si}`} blockGraph={blockGraph} variant="extra" />
          )}
          {si < subLayers.length - 1 ? <div className="arch-connector" style={{ height: 12 }}></div> : null}
        </Fragment>
      ))}
    </div>
  );
}

/** vanilla `renderRawModules` print(model) into `#modules-container`. */
function RawModulesPrint({ modelStr }: { modelStr?: string }) {
  return (
    <div className="max-h-96 overflow-y-auto">
      <pre id="modules-container" className="bg-slate-50 text-slate-800 p-4 overflow-x-auto mono text-sm m-0">
        {modelStr ? modelStr : (
          <div className="text-gray-500 whitespace-normal font-sans">
            No print(model) for this model: the installed transformers could not build it (its modeling code is
            custom or newer). The Architecture Diagram and Forward Source tabs describe its structure.
          </div>
        )}
      </pre>
    </div>
  );
}

/** vanilla analyze.js fill of `#raw-config-container` / `#processed-config-container`. */
function RawConfigPre({ id, text, className }: { id: string; text?: string; className: string }) {
  return (
    <div className="max-h-96 overflow-y-auto">
      <pre id={id} className={className}>{text}</pre>
    </div>
  );
}

/** A dim's meaning cut to its opening phrase, for a table cell: the glossary
 *  text is written for spec authors too (examples, code paths), and the full
 *  text stays on hover. */
function dimHeadline(symbol: string | null | undefined): string {
  const meaning = dimMeaning(symbol);
  return meaning.split(/\.\s|\s\(|:\s|\s—\s|;\s/)[0].replace(/\.$/, "");
}

/** vanilla `renderForwardSource` into `#forward-source-meta` / `#forward-source-files`. */
function ForwardSourcePanel({ fs }: { fs?: ForwardSource }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  useEffect(() => {
    if (expanded === null) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setExpanded(null); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [expanded]);
  const sourceLabels: Record<string, string> = {
    custom_model: "custom model",
    local_transformers: "installed transformers package",
    model_info_cache: "model_info cache",
    hf_modeling_files: "HF Hub download",
  };
  const files = (fs && Array.isArray(fs.files)) ? fs.files : [];
  if (!files.length) {
    return (
      <div className="space-y-4">
        <div id="forward-source-meta" className="text-xs text-gray-500 mono">no source available</div>
        <div id="forward-source-files" className="space-y-4" />
      </div>
    );
  }
  const label = sourceLabels[fs?.source || ""] || fs?.source || "unknown";
  const truncTag = fs?.truncated ? " (some files truncated)" : "";
  const trimPath = (p: string | undefined) => {
    if (!p) return "";
    const s = String(p);
    const markers = [
      "/site-packages/",
      "/.cache/huggingface/hub/",
      "/.cache/huggingarch/",
      "/modules/transformers_modules/",
      "/.huggingarch/",
      "/backend/arch/",
    ];
    for (const m of markers) {
      const i = s.indexOf(m);
      if (i >= 0) return "…" + s.slice(i);
    }
    return s;
  };
  const copyForwardSourceFile = (idx: number) => {
    const f = fs?.files?.[idx];
    if (f && f.text) void navigator.clipboard.writeText(f.text);
  };
  return (
    <div className="space-y-4">
      <div id="forward-source-meta" className="text-xs text-gray-500 mono">
        <span className="text-gray-400">via</span> {label}{truncTag} · {files.length} file{files.length === 1 ? "" : "s"}
      </div>
      <div id="forward-source-files" className="space-y-4">
        {files.map((f, i) => {
          const fname = (f.path || "").split("/").pop() || `file ${i + 1}`;
          const trimmed = trimPath(f.path);
          return (
            <div key={i} className="border border-gray-200 rounded-lg">
              <div className="flex justify-between items-center px-4 py-3 bg-gray-50 border-b border-gray-200 rounded-t-lg">
                <div className="flex flex-col min-w-0">
                  <h3 className="font-semibold text-gray-900 truncate">{fname}</h3>
                  <div className="text-10 text-gray-400 mono truncate">{trimmed}</div>
                </div>
                <div className="flex items-center gap-3 shrink-0 ml-4">
                  <button
                    type="button"
                    data-action="expand-forward-source"
                    data-index={i}
                    title="Expand to fullscreen"
                    className="text-gray-400 hover:text-gray-700"
                    onClick={() => setExpanded(i)}
                  >
                    <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M9.5 2.5h4v4M13.5 2.5 9 7M6.5 13.5h-4v-4M2.5 13.5 7 9" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    data-action="copy-forward-source"
                    data-index={i}
                    className="text-amber-600 hover:text-amber-700 text-sm whitespace-nowrap"
                    onClick={() => copyForwardSourceFile(i)}
                  >
                    Copy
                  </button>
                </div>
              </div>
              <div className="max-h-96 overflow-y-auto">
                <pre className="bg-slate-50 text-slate-800 border border-slate-200 rounded-lg p-4 overflow-x-auto mono text-xs m-0">{f.text || ""}</pre>
              </div>
            </div>
          );
        })}
      </div>
      {expanded !== null && files[expanded] ? (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
             onClick={() => setExpanded(null)}>
          <div className="bg-white rounded-lg shadow-2xl w-[95vw] max-w-6xl max-h-[90vh] flex flex-col"
               onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between items-center px-4 py-3 bg-gray-50 border-b border-gray-200 rounded-t-lg">
              <div className="flex flex-col min-w-0">
                <h3 className="font-semibold text-gray-900 truncate">
                  {(files[expanded].path || "").split("/").pop() || `file ${expanded + 1}`}
                </h3>
                <div className="text-10 text-gray-400 mono truncate">{trimPath(files[expanded].path)}</div>
              </div>
              <div className="flex items-center gap-3 shrink-0 ml-4">
                <button
                  type="button"
                  className="text-amber-600 hover:text-amber-700 text-sm whitespace-nowrap"
                  onClick={() => copyForwardSourceFile(expanded)}
                >
                  Copy
                </button>
                <button
                  type="button"
                  title="Close"
                  className="text-gray-400 hover:text-gray-700 text-lg leading-none"
                  onClick={() => setExpanded(null)}
                >
                  ×
                </button>
              </div>
            </div>
            <div className="overflow-auto flex-1">
              <pre className="bg-slate-50 text-slate-800 p-4 mono text-xs m-0 min-h-full">{files[expanded].text || ""}</pre>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

/** vanilla `renderUnresolvedBanner` — amber warning above the numbers. */
function UnresolvedBanner({ data }: { data: AnalyzeData }) {
  const note = data.dashboard?.spec_unresolved_note || data.spec_unresolved_note;
  if (!note) return null;
  return (
    <div id="unresolved-banner" className="mb-4 p-4 rounded-lg border border-amber-400 bg-amber-50">
      <div className="flex items-start gap-3">
        <div className="text-2xl">⚠️</div>
        <div className="flex-1">
          <div className="font-semibold text-amber-900 mb-1">Spec marked UNRESOLVED by generator</div>
          <div className="text-sm text-amber-800 mb-2">
            The spec-generation agent hit its iteration limit without reaching
            diff=0. Numbers below are computed from the partial spec and may be
            wrong. A human should review the following note and finish the spec.
          </div>
          <pre className="text-xs bg-white border border-amber-200 rounded p-3 whitespace-pre-wrap text-amber-900 max-h-48 overflow-auto">{note}</pre>
        </div>
      </div>
    </div>
  );
}

/** vanilla dashboard.js `row` — skip null/undefined/''/0. */
/** One attention dim: the config field it is bound from (else the dim
 *  symbol), the symbol beside it, and its value. */
function attnFactRow(d: AttnFact): ReactNode {
  return (
    <div key={d.symbol} className="flex justify-between py-1.5 border-b border-gray-50">
      <span className="text-gray-500 text-sm font-mono">
        {d.config || d.symbol}
        {d.config ? <span className="text-gray-300"> ({d.symbol})</span> : null}
      </span>
      <span className="font-mono text-sm text-gray-800">{d.value.toLocaleString()}</span>
    </div>
  );
}

/** Sorted layer indices as runs: [0, 1, 2, 5] → "0–2, 5". */
function layerRanges(indices: number[]): string {
  const runs: string[] = [];
  let start = indices[0];
  for (let i = 1; i <= indices.length; i++) {
    if (i < indices.length && indices[i] === indices[i - 1] + 1) continue;
    const end = indices[i - 1];
    runs.push(end === start ? String(start) : `${start}–${end}`);
    start = indices[i];
  }
  return runs.join(", ");
}

function dashRow(label: string, value: ReactNode): ReactNode {
  if (value === undefined || value === null || value === "" || value === 0) return null;
  const display = typeof value === "number" ? value.toLocaleString() : value;
  return (
    <div className="flex justify-between py-1.5 border-b border-gray-50">
      <span className="text-gray-500 text-sm">{label}</span>
      <span className="font-mono text-sm text-gray-800">{display}</span>
    </div>
  );
}

/** vanilla `renderDashboardStats` — four cards into `#dashboard-stats`. */
function DashboardStats({ data }: { data: AnalyzeData }) {
  const dash = data.dashboard || {};
  const mc = data.model_config || {};
  const attn = mc.attention || {};
  const ffn = mc.ffn || {};
  const norm = mc.norm || {};
  const configDims = mc.config_dims || [];
  const blockTypes = mc.block_types || [];
  // The breakdown is the main model's (MTP heads and encoders are their own
  // rows), so its shares are of the main-model headline.
  const total = dash.main_total_params;
  const embedding = dash.embedding_params;
  const nonEmbedding = dash.non_embedding_params;

  // Every dim the spec binds from a config field, less the ones the
  // Attention card already lists by symbol.
  const attnSymbols = new Set([
    ...(attn.families || []).flatMap((fam) => fam.dims.map((d) => d.symbol)),
    ...(attn.model_dims || []).map((d) => d.symbol),
  ]);
  const dimRows = configDims
    .filter((item) => item && !attnSymbols.has(item.symbol || ""))
    .filter((item) => item && (item.value || 0) > 0);

  const modelBytes = dash.hf_model_storage_bytes || 0;
  const quantBytes = dash.hf_quantization_metadata_bytes || 0;
  const hfCkpt = (() => {
    if (!modelBytes) return null;
    const tot = modelBytes + quantBytes;
    const tip = quantBytes > 0
      ? `Weights ${formatGiB(modelBytes)} + quant scales/shapes ${formatGiB(quantBytes)}`
      : "Sum of per-tensor (dtype × shape) from safetensors headers";
    return dashRow(
      "HF Checkpoint Size",
      <span title={tip} style={{ textDecoration: "underline dotted" }}>{formatGiB(tot)}</span>,
    );
  })();

  const weightBytes = dash.weight_bytes || 0;
  const ckpt = dash.hf_checkpoint_storage_bytes || 0;
  const weights = dash.hf_model_storage_bytes || 0;
  const metaBytes = ckpt - weights;
  const lookup = dash.lookup_table_params || 0;
  const paramShares: Array<[string, number, string]> = [
    ["Embedding", embedding as number, "bg-blue-500"],
    ...(lookup ? [["Lookup tables", lookup, "bg-purple-500"] as [string, number, string]] : []),
    ["Transformer", nonEmbedding as number, "bg-green-500"],
  ];

  return (
    <div id="dashboard-stats" className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card p-4 sm:p-6">
          <h3 className="font-semibold text-gray-900 mb-3">Model Overview</h3>
          {dashRow("Model Type",
            mc.model_type && mc.model_type.toLowerCase() !== "unknown" ? mc.model_type.toUpperCase() : mc.model_class || "—")}
          {dashRow("Total Parameters", <span title={MAIN_MODEL_TITLE}>{formatNumber(dash.main_total_params || 0)}</span>)}
          {dashRow("Layers", dash.num_layers)}
          {dashRow("Hidden Size (H)", dash.hidden_size)}
          {mc.total_active_params ? dashRow("Active Parameters (decode)", <span title={MAIN_MODEL_TITLE}>{formatNumber(mc.total_active_params)}</span>) : null}
          {hfCkpt}
          {dashRow("Precision", mc.precision)}
          {dashRow("Max Context", mc.max_position_embeddings)}
          {dashRow(
            "Tie Embeddings",
            mc.tied_weight !== undefined && mc.tied_weight !== null ? (mc.tied_weight ? "Yes" : "No") : null,
          )}
          {dimRows.length ? (
            <div className="mt-3 pt-2 border-t border-gray-100">
              <div className="text-xs text-gray-400 mb-1">Dims bound from the config</div>
              {dimRows.map((item) => (
                <div key={`${item.field}-${item.symbol}`} className="flex justify-between gap-2 py-1 border-b border-gray-50 text-sm">
                  {/* A long config path wraps; the value keeps its width. */}
                  <span className="text-gray-500 font-mono min-w-0 break-all">{item.field} <span className="text-gray-300">({item.symbol})</span></span>
                  <span className="font-mono text-gray-800 shrink-0">{Number(item.value).toLocaleString()}</span>
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="card p-4 sm:p-6">
          <h3 className="font-semibold text-gray-900 mb-3">Attention</h3>
          {blockTypes.length > 0 ? blockTypes.map((bt, i) => {
            const tmpl = bt.template || "block";
            const headerLabel = bt.label || tmpl;
            return (
              <div key={`${tmpl}-${i}`} className={`mb-3${blockTypes.length > 1 ? " pb-3 border-b border-gray-100" : ""}`}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-xs font-medium text-gray-400">{headerLabel}</span>
                  <span className="text-xs text-gray-400">×{bt.count} layers</span>
                </div>
                {dashRow("Attention", bt.attention_type)}
                {dashRow("FFN", bt.ffn_type === "moe" ? "MoE" : bt.ffn_type)}
              </div>
            );
          }) : (
            <>
              {dashRow("Attention", attn.type)}
              {dashRow("FFN", ffn.type === "moe" ? "MoE" : ffn.type)}
            </>
          )}
          {(attn.families || []).map((fam, i) => (
            <div key={`${fam.name}-${i}`} className="mt-2">
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-medium text-gray-400">{fam.name}</span>
                <span className="text-xs text-gray-400">×{fam.layers} layers</span>
              </div>
              {fam.dims.map((d) => attnFactRow(d))}
            </div>
          ))}
          {attn.model_dims && attn.model_dims.length > 0 ? (
            <div className="mt-2">
              <div className="text-xs text-gray-400 mb-1">Declared, read by no attention layer</div>
              {attn.model_dims.map((d) => attnFactRow(d))}
            </div>
          ) : null}
          {dashRow("RoPE θ", attn.rope_theta)}
          {dashRow("RoPE Scaling", attn.rope_scaling)}
          {dashRow("Sliding Window", attn.sliding_window)}
          {attn.swa ? (
            <>
              <div className="h-px bg-gray-100 my-2"></div>
              <div className="text-xs text-gray-400 mb-1">SWA Variant</div>
              {dashRow("SWA Heads", attn.swa.num_heads)}
              {dashRow("SWA KV Heads", attn.swa.num_kv_heads)}
              {dashRow("SWA Head Dim", attn.swa.head_dim)}
              {dashRow("SWA V Head Dim", attn.swa.v_head_dim)}
              {dashRow("SWA RoPE θ", attn.swa.rope_theta)}
            </>
          ) : null}
        </div>

        <div className="card p-4 sm:p-6">
          <h3 className="font-semibold text-gray-900 mb-3">FFN & Norm</h3>
          {dashRow("Intermediate Size (I)",
            ffn.dense_layers && ffn.dense_layers.length === 0 ? "N/A (no dense FFN layer)" : ffn.intermediate_size)}
          {dashRow("Activation", ffn.activation)}
          {ffn.type === "moe" ? dashRow("Routed Experts", ffn.num_experts) : null}
          {ffn.type === "moe" ? dashRow("Active (Top-K)", ffn.top_k) : null}
          {ffn.type === "moe" ? dashRow("Shared Experts", ffn.shared_experts) : null}
          {ffn.type === "moe" ? dashRow("MoE FFN Size", ffn.moe_intermediate_size) : null}
          {ffn.type === "moe" && ffn.dense_layers && ffn.dense_layers.length > 0
            ? dashRow("Layers with a dense FFN",
                `${ffn.dense_layers.length} (${ffn.dense_layers.length === 1 ? "layer" : "layers"} ${layerRanges(ffn.dense_layers)})`)
            : null}
          {dashRow("MTP Layers", ffn.mtp_layers)}
          <div className="h-px bg-gray-100 my-2"></div>
          {dashRow("Norm", norm.type)}
        </div>

        <div className="card p-4 sm:p-6">
          <h3 className="font-semibold text-gray-900 mb-3">Parameters</h3>
          <div className="space-y-3">
            {paramShares.map(([label, n, bar]) => {
              const pct = (n / (total as number)) * 100;
              return (
                <div key={label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{label}</span>
                    <span className="font-mono">{formatNumber(n)} ({pct.toFixed(1)}%)</span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div className={`h-full ${bar}`} style={{ width: `${pct}%` }}></div>
                  </div>
                </div>
              );
            })}
            <div className="text-sm flex justify-between pt-2 border-t border-gray-100">
              <span className="text-gray-500" title="Parameters, their quantisation scales and buffers — the per-op weight bytes, summed">Weights + scales</span>
              <span className="font-mono">{formatGiB(weightBytes)} <span className="text-gray-400">({dash.weight_dtype ?? "—"})</span></span>
            </div>
            {ckpt ? (
              <div className="text-sm flex justify-between">
                <span className="text-gray-500">Checkpoint Size (full)</span>
                <span className="font-mono text-right">{formatGiB(ckpt)}{metaBytes > 0 ? <span className="text-gray-400"> (incl. {formatGiB(metaBytes)} quant metadata)</span> : null}
                  <CheckpointCopiesNote copies={dash.checkpoint_copies} />
                </span>
              </div>
            ) : null}
          </div>
          {blockTypes.length > 1 ? (
            <div className="mt-3 pt-2 border-t border-gray-100">
              <div className="text-xs text-gray-400 mb-1">Block Types (heterogeneous)</div>
              {blockTypes.map((bt, i) => (
                <div key={`${bt.template}-${i}`} className="flex justify-between py-1 text-sm">
                  <span className="text-gray-500">{bt.label || bt.template}</span>
                  <span className="font-mono text-gray-700">
                    ×{bt.count} · {[bt.attention_type, bt.ffn_type === "moe" ? "MoE" : bt.ffn_type ? `${bt.ffn_type} FFN` : null]
                      .filter(Boolean).join(" + ") || "no attention or FFN"}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

/** Section label of an auxiliary stack's rows, from the spec's `aux_kind`. */
const AUX_LABELS: Record<string, string> = {
  vision: "Vision Tower",
  audio: "Audio Tower",
  mtp: "MTP / Speculative Head",
  speculative: "MTP / Speculative Head",
};

/** A composition amount with its ledger-backed derivation on hover. */
function compFormulaCell(v: number, formula?: FormulaFields,
                         fmt: (n: number) => string = formatNumber): ReactNode {
  if (!v) return "—";
  const shown = fmt(v);
  return <FormulaHover value={shown} formula={formula ? {
    ...formula, result: shown,
  } : undefined} />;
}

function withFormulaResult(formula: FormulaFields | undefined,
                           result: string): FormulaFields | undefined {
  return formula ? { ...formula, result } : undefined;
}

/** The KV cell: bytes each token adds, and/or the state held per sequence. */
function KvCell({ kv }: { kv: {
  per_token_bytes: number;
  constant_bytes: number;
  growth_formula?: FormulaFields | null;
  constant_formula?: FormulaFields | null;
  dtypes?: KvDtype[];
} | null }) {
  if (!kv) return <span className="text-gray-300" title="Not part of the serving KV cache, or the model states no dtype to price it at">—</span>;
  const parts: ReactNode[] = [];
  if (kv.per_token_bytes) parts.push(
    <div key="t"><FormulaHover
      value={`${formatGiB(kv.per_token_bytes)} / token`}
      formula={kv.growth_formula || undefined}
    /></div>,
  );
  if (kv.constant_bytes) {
    parts.push(
      <div key="c" title="Held per sequence whatever its length: a recurrent state, or a sliding window's cache once the sequence is past the window">
        <FormulaHover
          value={`${formatGiB(kv.constant_bytes)} constant / sequence`}
          formula={kv.constant_formula || undefined}
        />
      </div>,
    );
  }
  if (parts.length && kv.dtypes && kv.dtypes.length) {
    parts.push(
      <div key="d" className="text-10 text-gray-400" title="declared: the spec's cache entry states it; native: the model's stated default precision">
        {kv.dtypes.map((d) => `${d.dtype} (${d.source})`).join(" · ")}
      </div>,
    );
  }
  return parts.length ? <>{parts}</> : <span className="text-gray-300">—</span>;
}

/** The Dashboard's model composition: one row per block declaration and per
 *  model-level node, every figure fixed by the spec and checkpoint
 *  (`data.composition`, `backend/views/composition.py`). The totals row is the
 *  dashboard's own Total / Active Parameters and weight bytes. */
function CompositionTable({
  data,
  composition,
  onOpen,
}: {
  data: AnalyzeData;
  composition: Composition;
  onOpen: (templates: string[]) => void;
}) {
  const dash = data.dashboard || {};
  const rows = composition.rows;
  const hasExperts = rows.some((r) => r.params.experts > 0);
  const hasActive = Number(dash.total_active_params) < Number(dash.total_params);
  const label = (r: CompositionRow) => r.label;
  const numCol = (
    key: string,
    header: string,
    render: (r: CompositionRow) => ReactNode,
    sortValue: (r: CompositionRow) => number | null,
    className?: string,
  ): ColumnDef<CompositionRow> => ({
    key,
    header,
    align: "right",
    className: ["font-mono whitespace-nowrap", className ?? ""].filter(Boolean).join(" "),
    render,
    sortValue,
  });
  const columns: ColumnDef<CompositionRow>[] = [
    {
      key: "block", wrap: true,
      header: "Block",
      // Long block names wrap inside a bounded column, so the figure
      // columns stay in view.
      className: "min-w-[12rem] max-w-[22rem]",
      sortValue: (r) => label(r),
      render: (r) => (
        <>
          <div className="font-medium text-gray-900 break-words">{label(r)}</div>
          {r.aux_kind ? (
            <div className="text-10 text-gray-400">{AUX_LABELS[r.aux_kind] || r.aux_kind}</div>
          ) : null}
          {r.histogram && r.histogram.length ? (
            <div className="text-10 text-amber-800" title={r.histogram.map((h) => `${h.visible}: layers ${compressRanges(h.layer_indices)}`).join("\n")}>
              {r.histogram.length} groups by history: a layer reads {visibleSpan(r.histogram)} earlier
              {r.stream ? ` ${r.stream}` : ""} entries ({r.axis})
            </div>
          ) : null}
        </>
      ),
    },
    {
      key: "layers",
      header: "Layers",
      className: "font-mono text-gray-500",
      render: (r) => (r.kind === "block" ? compressRanges(r.layer_indices) : "—"),
      sortValue: (r) => (r.kind === "block" ? r.layer_indices[0] : null),
    },
    numCol("count", "Count", (r) => (r.kind === "block" ? r.layers : "—"), (r) => (r.kind === "block" ? r.layers : null)),
    numCol("attention", "Attention", (r) => compFormulaCell(r.params.attention, r.formulas?.attention), (r) => r.params.attention || null),
    numCol("ffn", "FFN", (r) => compFormulaCell(r.params.ffn, r.formulas?.ffn), (r) => r.params.ffn || null),
    ...(hasExperts ? [numCol("experts", "Experts", (r) => compFormulaCell(r.params.experts, r.formulas?.experts), (r) => r.params.experts || null)] : []),
    numCol("other", "Other", (r) => compFormulaCell(r.params.other, r.formulas?.other), (r) => r.params.other || null),
    numCol("params", "Params", (r) => compFormulaCell(r.params.total, r.formulas?.total), (r) => r.params.total, "text-gray-900"),
    ...(hasActive ? [numCol("active", "Active", (r) => compFormulaCell(r.active_params, r.formulas?.active), (r) => r.active_params, "text-orange-700")] : []),
    {
      key: "stored", wrap: true,
      header: "Stored",
      className: "font-mono",
      sortValue: (r) => r.storage_bytes,
      render: (r) => (
        <>
          <div className="text-gray-900 whitespace-nowrap">
            {compFormulaCell(r.storage_bytes, r.formulas?.storage, formatGiB)}
          </div>
          {r.storage_by_dtype.length > 1 || (r.storage_by_dtype[0] && r.storage_by_dtype[0].dtype !== dash.weight_dtype) ? (
            // One line per width, so a mixed row does not widen the table.
            <div className="text-10 text-gray-400">
              {r.storage_by_dtype.map((d) => <div key={d.dtype} className="whitespace-nowrap">{d.dtype} {formatGiB(d.bytes)}</div>)}
            </div>
          ) : null}
        </>
      ),
    },
    {
      key: "kv", wrap: true,
      header: (
        <span title={composition.kv_native_dtype
          ? `dtype per spec, else the model default (${composition.kv_native_dtype} · ${composition.kv_native_source})`
          : "The model states no dtype, so the cache is not priced"}>
          KV cache
        </span>
      ),
      className: "font-mono",
      render: (r) => <KvCell kv={r.kv} />,
      sortValue: (r) => r.kv?.per_token_bytes || null,
    },
  ];
  const footTd = `${tbl.cell} text-right font-mono tabular-nums whitespace-nowrap`;
  return (
    <div id="model-composition" className="card p-4 sm:p-6 mt-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-3">
        <h3 className="font-semibold text-gray-900">Model composition</h3>
        <a
          id="composition-to-inference"
          href={`/inference.html#model=${encodeURIComponent(data.model_id || "")}`}
          className="text-xs text-amber-600 hover:text-amber-700"
        >
          See time &amp; FLOPs in Inference →
        </a>
      </div>
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(r) => r.key}
        onRowClick={(r) => onOpen(r.templates || [])}
        rowProps={(r) => ({
          "data-composition-row": r.kind,
          "data-layers": String(r.layers),
          "data-params": String(r.params.total),
          "data-active": String(r.active_params),
          "data-bytes": String(r.storage_bytes),
          "data-groups": r.histogram ? String(r.histogram.length) : undefined,
          title: "Show on the Architecture Diagram",
        })}
        footer={
          <tr className="border-t-2 border-gray-200 font-semibold" data-composition-totals="">
            <td className={tbl.cell} title="Every row above: the main model, MTP heads and encoders. Bytes are the weights as loaded; a tied weight the checkpoint stores twice counts once (Checkpoint Size above includes the copy).">Model total</td>
            <td className={tbl.cell}></td>
            <td className={footTd}>{dash.num_layers || "—"}</td>
            <td className={footTd} colSpan={hasExperts ? 4 : 3}></td>
            <td className={`${footTd} text-gray-900`} id="composition-total-params">
              <FormulaHover value={dash.total_params_formatted || formatNumber(composition.totals.params)} formula={withFormulaResult(composition.totals.formulas?.params, dash.total_params_formatted || formatNumber(composition.totals.params))} />
            </td>
            {hasActive ? <td className={`${footTd} text-orange-700`} id="composition-total-active">
              <FormulaHover value={dash.total_active_params_formatted || formatNumber(composition.totals.active_params)} formula={withFormulaResult(composition.totals.formulas?.active, dash.total_active_params_formatted || formatNumber(composition.totals.active_params))} />
            </td> : null}
            <td className={`${tbl.wrapCell} font-mono`}>
              <div id="composition-total-bytes" title="Weights as loaded">
                <FormulaHover value={formatGiB(dash.weight_bytes || 0)} formula={withFormulaResult(composition.totals.formulas?.storage, formatGiB(dash.weight_bytes || 0))} />
              </div>
              <div className="text-10 text-gray-400 font-normal">
                {composition.totals.storage_by_dtype.map((d) => <div key={d.dtype} className="whitespace-nowrap">{d.dtype} {formatGiB(d.bytes)}</div>)}
              </div>
            </td>
            <td className={`${tbl.wrapCell} font-mono`}>
              <KvCell kv={composition.totals.kv_per_token_bytes == null ? null : {
                per_token_bytes: composition.totals.kv_per_token_bytes,
                constant_bytes: composition.totals.kv_constant_bytes || 0,
                growth_formula: composition.totals.kv_growth_formula,
                constant_formula: composition.totals.kv_constant_formula,
              }} />
            </td>
          </tr>
        }
      />
      <p className="mt-2 text-11 text-gray-400">
        Each row is its whole share across its layers. FFN excludes routed experts (router and shared experts stay in FFN); Other is norms and model-level tables. Click a row to find it on the Architecture Diagram.
      </p>
    </div>
  );
}

/** vanilla `renderArchitectureView` — main path + cycle + aux + summary. */
function ArchitectureView({
  arch,
  blockGraph,
}: {
  arch: ArchView | undefined;
  blockGraph: BlockGraph | undefined;
}) {
  if (!arch || !arch.layers) {
    return (
      <div id="arch-view-container">
        <div className="text-gray-500 text-center py-8">No architecture view available</div>
      </div>
    );
  }

  const layers = arch.layers;
  const sections = arch.extra_sections || [];
  const table = arch.summary_table || {};
  const keys = Object.keys(table);
  const nodes: ArchGraphNode[] = [
    ...(arch.model_inputs || []).map((m): ArchGraphNode => ({
      id: m.id, kind: "input", content: m.name,
    })),
    ...layers.map((layer, i): ArchGraphNode => ({
      id: layer.id || `card-${i}`,
      kind: "card",
      wide: !!layer.stream_map,
      content: layer.type === "cycle_container" && Array.isArray(layer.cycle_blocks) ? (
        <ArchCycleContainer layer={layer} indexPrefix={`arch-cycle-${i}`} blockGraph={blockGraph} />
      ) : (
        <ArchLayerBox layer={layer} panelId={`arch-panel-${i}`} blockGraph={blockGraph} variant="main" />
      ),
    })),
    ...sections.map((section, k): ArchGraphNode => ({
      id: section.id || `section-${k}`,
      kind: "section",
      content: <ArchSectionBox section={section} blockGraph={blockGraph} />,
    })),
  ];
  // Without a spec there is no topology to read: the cards stand in order.
  const edges: ArchGraphEdge[] = arch.edges || layers.slice(1).map((layer, i) => ({
    from: layers[i].id || `card-${i}`, to: layer.id || `card-${i + 1}`, kind: "data",
  }));

  return (
    <div id="arch-view-container">
      <div className="arch-card">
        <div className="text-center mb-6">
          <div className="flex flex-wrap items-center justify-center gap-2 mb-2">
            <span className="text-lg font-semibold">Model Architecture</span>
            {(arch.badges || []).map((badge) => (
              <span key={badge} className="arch-badge">{badge}</span>
            ))}
          </div>
          <div className="text-2xl font-bold mb-1">{arch.model_name || "Model"}</div>
          <div className="text-sm text-gray-500">{arch.subtitle || ""}</div>
        </div>

        <ArchGraph nodes={nodes} edges={edges} />

        {keys.length > 0 ? (
          <div className="arch-summary-table">
            {keys.map((key) => (
              <div key={key} className="arch-summary-cell">
                <div className="label">{key}</div>
                <div className="value">{table[key]}</div>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

/** A spec the analysis engine blocks: the capabilities it lacks and where
 *  validate met each. Served from the caller's record, the analysis below
 *  is of every deployment the engine supports. */
function BlockedSpecBanner({ banner }: { banner: SpecBanner }) {
  const title = !banner.served
    ? "The spec is complete, but the analysis engine does not yet support"
    : banner.recordStatus === "blocked_by_engine"
      ? "Your spec is engine-blocked: it states the model correctly, but the analysis engine does not yet support"
      : "Your spec is valid, but this analysis needs a capability the analysis engine does not yet support";
  return (
    <EngineBlockedBanner
      title={`${title}:`}
      gaps={banner.gaps}
      note={banner.hit ? (
        <>This analysis needs it too: <span className="font-medium">{banner.hit.capability}</span>
          {banner.hit.node ? <> at <code className="font-mono text-xs">{banner.hit.node}</code></> : null}.</>
      ) : banner.notSaved ? "The blocked spec could not be saved to your generations; it remains in this run's workspace." : null}
    >
      <a href="/account/workspace.html#generations" className="px-4 py-1.5 bg-violet-700 text-white text-sm rounded-lg hover:bg-violet-800">
        Your generations
      </a>
    </EngineBlockedBanner>
  );
}

/** The caller's own spec did not analyse, for a reason other than an engine
 *  gap. A repairable schema refusal starts Agent Fix by itself; this card
 *  stays while that runs, and after an unrepairable refusal. */
function RecordFailedBanner({ banner }: { banner: SpecBanner }) {
  // A spec that only needs declarations is answered with the card's own
  // heading as its message; its refusals ride in `errors[]`
  // (`normalize_error_payload`), so the line under the heading would only
  // repeat it.
  const schema = banner.errorKind === "spec_needs_declaration";
  return (
    <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start justify-between gap-4" data-record-failed>
      <div className="text-red-800 min-w-0">
        <div className="font-medium">Schema error</div>
        {!schema && banner.message ? (
          <div className="text-sm text-red-700 mt-1 break-words">{banner.message}</div>
        ) : null}
      </div>
      <a href="/account/workspace.html#generations" className="px-4 py-1.5 bg-red-600 text-white text-sm rounded-lg hover:bg-red-700 shrink-0">
        Review &amp; fix
      </a>
    </div>
  );
}

function SpecBannerCard({ banner }: { banner: SpecBanner }) {
  return (
    <div className="mb-4 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center justify-between gap-4">
      <div className="text-green-800">
        <span className="font-medium">Spec generated</span>
        <span className="text-green-700 text-sm ml-2">{banner.message || "— Review it and open a PR from your account."}</span>
      </div>
      {banner.needSignIn ? (
        <a href="/auth/login.html" className="px-4 py-1.5 bg-green-700 text-white text-sm rounded-lg hover:bg-green-800 shrink-0">Sign in</a>
      ) : banner.record?.pr_url ? (
        <a href={banner.record.pr_url} target="_blank" rel="noopener" className="px-4 py-1.5 bg-green-700 text-white text-sm rounded-lg shrink-0">View PR</a>
      ) : banner.authEnabled ? (
        <a href="/account/workspace.html#generations" className="px-4 py-1.5 bg-green-700 text-white text-sm rounded-lg hover:bg-green-800 shrink-0">Review & open PR</a>
      ) : null}
    </div>
  );
}
