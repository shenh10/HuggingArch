/** Sweep/compare hash. Keys match url-state.js (`sub`/`m`/`s`), not DOM ids. */

import type { Parallel } from "./parallel";
import { withoutRetiredPrecision } from "./precision";
import type { Estimate, PhaseMetrics } from "./pages/ResultSummary";

export const SWEEP_MIN_ROWS = 2;
export const SWEEP_MAX_ROWS = 8;

export type SweepCfg = {
  gpu_sku?: string;
  parallel?: Partial<Parallel>;
  [extra: string]: unknown;
};

export type SweepRow = {
  model_id: string;
  cfg: SweepCfg | null;
};

export type SweepHash = {
  sub: string;
  m: string;
  s: string;
};

type SweepPayload = {
  template?: SweepCfg | null;
  cfgs?: unknown;
  share?: unknown;
};

function b64UrlEncode(s: string): string {
  const bytes = new TextEncoder().encode(s);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function b64UrlDecode(s: string): string {
  let t = String(s).replace(/-/g, "+").replace(/_/g, "/");
  while (t.length % 4) t += "=";
  const bin = atob(t);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder().decode(bytes);
}

export function parseModels(raw: string): string[] {
  return String(raw || "").split("|").map((x) => x.trim());
}

export function decodeState(raw: string): SweepPayload | null {
  if (!raw) return null;
  try {
    const payload = JSON.parse(b64UrlDecode(raw)) as unknown;
    if (!payload || typeof payload !== "object") return null;
    return payload as SweepPayload;
  } catch {
    return null;
  }
}

/** Drafted tokens for a custom speculative K saved before K counted drafted
 *  tokens. Those links stored the verify window (drafted tokens + 1) — the
 *  Inference hash as `spec_k`, a shared config as `speculative_K` — and load
 *  as window − 1, so an old link prices the same step it did. */
export function draftTokensFromLegacyWindow(raw: unknown): number | undefined {
  const w = Number(raw);
  return raw !== "" && raw != null && Number.isFinite(w) ? Math.max(0, Math.round(w) - 1) : undefined;
}

function migrateSpeculativeK(cfg: SweepCfg): SweepCfg {
  if (!("speculative_K" in cfg)) return cfg;
  const { speculative_K: window, ...rest } = cfg;
  if (rest.speculative_draft_tokens === undefined) {
    const k = draftTokensFromLegacyWindow(window);
    if (k !== undefined) rest.speculative_draft_tokens = k;
  }
  return rest;
}

/** Said atop Compare: it prices every model at its shipped precision. */
export const COMPARE_AS_SHIPPED_NOTE = "Compare shows every model as shipped.";

/** Said when a Compare link carried precision settings, which were dropped. */
export const COMPARE_LEGACY_PRECISION_NOTICE =
  "Compare shows every model as shipped; this link's precision settings were not applied.";

/** Per-row cfg from `cfgs[i]`. Does not copy template onto later rows.
 *  Compare prices every model as shipped: a link's precision fields are
 *  dropped, and `droppedPrecision` says whether there were any. */
export function hydrateSweep(m: string, s: string): {
  rows: SweepRow[]; template: SweepCfg | null; droppedPrecision: boolean;
} {
  let models = parseModels(m);
  while (models.length < SWEEP_MIN_ROWS) models.push("");
  if (models.length > SWEEP_MAX_ROWS) models = models.slice(0, SWEEP_MAX_ROWS);

  let droppedPrecision = false;
  const asCfg = (v: unknown): SweepCfg | null => {
    if (!v || typeof v !== "object") return null;
    const kept = withoutRetiredPrecision(v as SweepCfg);
    if (kept.dropped) droppedPrecision = true;
    return migrateSpeculativeK(kept.cfg);
  };
  const payload = decodeState(s);
  const template = asCfg(payload?.template ?? null);
  let cfgs: (SweepCfg | null)[] = [];
  if (payload && Array.isArray(payload.cfgs)) {
    cfgs = payload.cfgs.map(asCfg);
  } else if (payload?.share && template) {
    cfgs = models.map(() => template);
  }

  const rows = models.map((model_id, i) => ({
    model_id,
    cfg: cfgs[i] ?? null,
  }));
  return { rows, template, droppedPrecision };
}

export function encodeSweep(rows: SweepRow[], template: SweepCfg | null): SweepHash {
  const payload = { template, cfgs: rows.map((r) => r.cfg) };
  return { sub: "sweep", m: rows.map((r) => r.model_id).join("|"), s: b64UrlEncode(JSON.stringify(payload)) };
}

export function readSweepParams(hash: string): Partial<SweepHash> {
  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const out: Partial<SweepHash> = {};
  const sub = params.get("sub");
  if (sub) out.sub = sub;
  const m = params.get("m");
  if (m != null && m !== "") out.m = m;
  const s = params.get("s");
  if (s != null && s !== "") out.s = s;
  return out;
}

/** Vanilla `p` / `hydrateIcPhase`. Decode is default — elide it. Unknown values stay decode. */
export type SweepPhase = "prefill" | "decode";

export function readSweepPhase(hash: string): SweepPhase {
  const p = new URLSearchParams(hash.replace(/^#/, "")).get("p");
  return p === "prefill" ? "prefill" : "decode";
}

/** Write `p=prefill`; drop `p` for decode or when not in sweep. Leaves other keys. */
export function mergeSweepPhase(existingHash: string, phase: SweepPhase | null): string {
  const params = new URLSearchParams(existingHash.replace(/^#/, ""));
  if (phase === "prefill") params.set("p", "prefill");
  else params.delete("p");
  const next = params.toString();
  return next ? `#${next}` : "";
}

/** Only set/delete `sub`/`m`/`s`. Leaves `p`/`dec` and other unowned keys. */
export function mergeSweep(existingHash: string, sweep: SweepHash | null): string {
  const params = new URLSearchParams(existingHash.replace(/^#/, ""));
  if (!sweep) {
    params.delete("sub");
    params.delete("m");
    params.delete("s");
  } else {
    if (sweep.sub) params.set("sub", sweep.sub);
    else params.delete("sub");
    if (sweep.m) params.set("m", sweep.m);
    else params.delete("m");
    if (sweep.s) params.set("s", sweep.s);
    else params.delete("s");
  }
  const next = params.toString();
  return next ? `#${next}` : "";
}

/** Same section keys as vanilla `IC_SECTIONS`. Push copies template → each row. */
export const SWEEP_SECTION_KEYS = {
  gpu: ["gpu_sku", "gpu_memory_fraction", "comm_latency_us", "launch_floor_us"],
  parallel: ["tie_tp", "parallel", "cp_scheme", "ep_load_factor"],
  workload: [
    "prefix_cached_tokens",
    "prefill_chunk_size",
    "speculative_mode", "speculative_draft_tokens", "acceptance_rate",
    "speculative_draft", "include_aux_weights", "host_offload",
  ],
} as const;

export type SweepSection = keyof typeof SWEEP_SECTION_KEYS;

export function applyTemplateSection(
  rows: SweepRow[],
  template: SweepCfg | null,
  section: SweepSection,
  skipIdx?: ReadonlySet<number>,
): { rows: SweepRow[]; pushed: number; skipped: number } {
  if (!template) return { rows, pushed: 0, skipped: 0 };
  const keys = SWEEP_SECTION_KEYS[section];
  let pushed = 0;
  let skipped = 0;
  const next = rows.map((row, i) => {
    if (skipIdx?.has(i)) {
      skipped += 1;
      return row;
    }
    const cfg: SweepCfg = { ...(row.cfg || {}) };
    for (const k of keys) (cfg as Record<string, unknown>)[k] = template[k];
    pushed += 1;
    return { ...row, cfg };
  });
  return { rows: next, pushed, skipped };
}

export function patchRow(
  row: SweepRow,
  patch: { model_id?: string; gpu_sku?: string; ep_moe?: number },
): SweepRow {
  const next: SweepRow = { model_id: patch.model_id ?? row.model_id, cfg: row.cfg };
  const needsCfg = patch.gpu_sku !== undefined || patch.ep_moe !== undefined;
  if (!needsCfg) return next;
  const cfg: SweepCfg = { ...(row.cfg || {}) };
  if (patch.gpu_sku !== undefined) cfg.gpu_sku = patch.gpu_sku;
  if (patch.ep_moe !== undefined) {
    const prev = cfg.parallel && typeof cfg.parallel === "object" ? cfg.parallel : {};
    cfg.parallel = { ...prev, ep_moe: patch.ep_moe };
  }
  next.cfg = cfg;
  return next;
}

/** Same 9 seq points as vanilla `IC_SEQ_POINTS`. Bodies still come from `row.cfg`. */
export const SWEEP_SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 131072, 262144, 1048576];

/** The serving grid: prefill stays at one request while decode concurrency
 * sweeps. Paired-phase batch requests remain a measurement-campaign shape. */
export type SweepBatchPoint = {
  key: string;
  prefill_batch_size: number;
  decode_batch_size: number | null;
};

export const SWEEP_BATCH_GRID: SweepBatchPoint[] = [
  { key: "b1", prefill_batch_size: 1, decode_batch_size: 1 },
  { key: "b16", prefill_batch_size: 1, decode_batch_size: 16 },
  { key: "b64", prefill_batch_size: 1, decode_batch_size: 64 },
  { key: "max", prefill_batch_size: 1, decode_batch_size: null },
];

/** What the Scaling panel's Context × Batch body asks of the grid
 *  (`sweep_grid.SCALING_PANEL_BATCH_GRID`): one request (prefill vs context)
 *  and each card's ceiling (KV storage). Its decode chart is a view of the
 *  parallel sweep's cloud; Compare keeps the whole grid. */
export const SCALING_PANEL_BATCH_GRID: SweepBatchPoint[] = SWEEP_BATCH_GRID.filter(
  (p) => p.decode_batch_size === 1 || p.decode_batch_size === null);

/** The routine grid's decode length (`sweep_grid.ROUTINE_OUTPUT_TOKENS`); also the
 *  Inference page's seeded `outTok`. */
export const SWEEP_OUTPUT_TOKENS = 256;

/** vLLM V1 / SGLang default chunked-prefill token budget. 0 = unchunked. */
export const DEFAULT_PREFILL_CHUNK_SIZE = 8192;

/** Speculative-decoding presets: K drafted tokens per step and acceptance
 *  rate α. The target verifies K + 1 positions (the last accepted token plus
 *  the drafts); K = 0 is plain autoregressive decode. Mirrors
 *  `backend.inference.sweep_grid.SPEC_PRESETS` (the presets Kernel Bench
 *  plans), pinned by tests/backend/inference/test_sweep_grid.py; `custom`
 *  seeds the free-form input and is no preset. */
export const SPEC_PRESETS: Record<string, { K: number; alpha: number; label: string }> = {
  none: { K: 0, alpha: 1, label: "None (plain AR)" },
  mtp: { K: 1, alpha: 0.85, label: "MTP (K=1 draft token, α=0.85)" },
  medusa: { K: 4, alpha: 0.6, label: "Medusa (K=4 draft tokens, α=0.6)" },
  eagle: { K: 5, alpha: 0.7, label: "EAGLE (K=5 draft tokens, α=0.7)" },
  custom: { K: 4, alpha: 0.65, label: "Custom…" },
};

/** The /inference/estimate request fields a deployment config determines. */
export type EstimateCfgFields = {
  gpu_sku?: string;
  gpu_memory_fraction?: number;
  /** One link traversal's latency and the shortest one kernel
   *  launch takes, µs (the server's defaults when absent). */
  comm_latency_us?: number;
  launch_floor_us?: number;
  parallel?: Partial<Parallel> & { cp_scheme?: string; ep_load_factor?: number };
  prefix_cached_tokens?: number;
  /** Tokens one prefill forward carries (vLLM `max_num_batched_tokens`,
   *  SGLang `chunked_prefill_size`). 0 = unchunked full-seq prefill. */
  prefill_chunk_size?: number;
  speculative_draft_tokens?: number;
  acceptance_rate?: number;
  speculative_draft?: SpecDraftStyle;
  include_aux_weights?: boolean;
  host_offload?: boolean;
};

/** How the draft produces its tokens when its cost is counted: EAGLE-style
 *  recursive single-token forwards, or one DFlash-style parallel forward. */
export type SpecDraftStyle = "recursive" | "parallel";

export function specDraftStyle(v: unknown): SpecDraftStyle | undefined {
  return v === "recursive" || v === "parallel" ? v : undefined;
}

export type SweepRequestBody = EstimateCfgFields & {
  model_id: string;
  gpu_sku: string;
  seq_lens: number[];
  batch_grid: SweepBatchPoint[];
  avg_output_tokens: number;
};

/** An `/inference/estimate-sweep` response: `results_by_batch[batch key][seq index]`. */
export type SweepResponse = { results_by_batch?: Record<string, Estimate[]> };

/** One compared row: the request sent, and its response or error. */
export type SweepHit = {
  body: SweepRequestBody;
  /** The body as posted (with the caller's calibration flag), so a view of
   *  the sweep reads the same answer. */
  sent?: SweepRequestBody & { calibrate_with_mine?: true };
  result?: SweepResponse;
  error?: string;
};

/** Compare-chart row colours, by row index. */
export const SWEEP_COLORS = [
  "#4e79a7", "#f28e2c", "#e15759", "#76b7b2",
  "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
];

/** A phase's tok/s per GPU on one track as the backend reported it, so rows
 *  on different GPU counts compare per GPU; null where that track has no
 *  value (measured: nothing was measured). */
export function phaseTokPerSec(phase: PhaseMetrics | undefined, source = "sol"): number | null {
  if (!phase || !phase.batch || phase.batch <= 0) return null;
  const t = phase.throughput || {};
  const src = source === "measured" ? t.measured : t.sol;
  const v = src?.per_gpu;
  return typeof v === "number" ? v : null;
}

/** A phase's step time in seconds on one track as the backend reported it;
 *  null when absent or not positive (measured: nothing was measured). */
export function phaseSeconds(phase: PhaseMetrics | undefined, source: string): number | null {
  if (!phase) return null;
  const t = source === "measured" ? phase.t_measured_s : phase.t_sol_s;
  return typeof t === "number" && t > 0 ? t : null;
}

/** Same dim order as vanilla `IC_DIM_ORDER`. Common dims → caption; varying → legend. */
export const SWEEP_DIM_ORDER = ["model", "gpu", "ngpu", "tp", "ep", "cp", "dp", "pp"] as const;
const SWEEP_PAR_DIMS = ["tp", "ep", "cp", "dp", "pp"] as const;

type DimTok = { disp: string | null; val: string | number | null };

function posInt(v: unknown, d: number): number {
  return typeof v === "number" && Number.isFinite(v) && v > 0 ? v : d;
}

/** The GPUs a compared row's deployment occupies, as its estimate says
 *  (`cluster_size`); null until a result arrives. */
function clusterOf(hit: SweepHit): number | null {
  for (const row of Object.values(hit.result?.results_by_batch || {})) {
    for (const est of row || []) {
      if (est?.cluster_size != null) return est.cluster_size;
    }
  }
  return null;
}

function groupDims(hit: SweepHit): Record<(typeof SWEEP_DIM_ORDER)[number], DimTok> {
  const g = hit.body;
  const p = g.parallel || {};
  const tpA = posInt(p.tp_attention, 1);
  const par: Parallel = {
    tp_attention: tpA,
    tp_dense_ffn: posInt(p.tp_dense_ffn, 1),
    tp_moe_expert: posInt(p.tp_moe_expert, 1),
    ep_moe: posInt(p.ep_moe, 1),
    cp: posInt(p.cp, 1),
    pp: posInt(p.pp, 1),
    dp_attention: posInt(p.dp_attention, 1),
    dp_replicas: posInt(p.dp_replicas, 1),
  };
  const world = clusterOf(hit);
  const shortModel = (g.model_id || "").split("/").pop() || g.model_id || "?";
  return {
    model: { disp: shortModel, val: g.model_id },
    gpu: { disp: g.gpu_sku || "?", val: g.gpu_sku || "?" },
    ngpu: { disp: world == null ? null : `${world}×GPU`, val: world },
    tp: { disp: `TP${par.tp_attention}`, val: par.tp_attention },
    ep: { disp: `EP${par.ep_moe}`, val: par.ep_moe },
    cp: { disp: `CP${par.cp}`, val: par.cp },
    dp: { disp: `DP${par.dp_attention}`, val: par.dp_attention },
    pp: { disp: `PP${par.pp}`, val: par.pp },
  };
}

/** Factor groups into a shared caption and per-row legend labels. Payload only; no SKU default. */
export function groupLabels(hits: SweepHit[]): { labels: string[]; common: string[] } {
  const dims = hits.map(groupDims);
  const varying: (typeof SWEEP_DIM_ORDER)[number][] = [];
  const commonKeys: (typeof SWEEP_DIM_ORDER)[number][] = [];
  for (const k of SWEEP_DIM_ORDER) {
    const vals = new Set(dims.map((d) => String(d[k] ? d[k].val : null)));
    (vals.size > 1 ? varying : commonKeys).push(k);
  }
  const labels = dims.map((d, i) => {
    const toks = varying.map((k) => d[k] && d[k].disp).filter(Boolean);
    return toks.length ? toks.join(" · ") : `#${i + 1}`;
  });
  const d0 = dims[0] || ({} as Record<string, DimTok>);
  const caption: string[] = [];
  for (const k of commonKeys) {
    if ((SWEEP_PAR_DIMS as readonly string[]).includes(k)) continue;
    if (d0[k] && d0[k].disp) caption.push(d0[k].disp);
  }
  const commonPar = SWEEP_PAR_DIMS.filter((k) => commonKeys.includes(k))
    .map((k) => d0[k] && d0[k].disp)
    .filter((x): x is string => Boolean(x));
  if (commonPar.length) caption.push(commonPar.join("/"));
  return { labels, common: caption };
}

/** Same fields as vanilla `icDeployFromPayload`. Payload only; no SKU default. */
export type SweepDeploy = {
  gpu: string;
  world: number | null;
  ep: number;
  dpAttn: number;
  tpAttn: number;
  cp: number;
  pp: number;
  label: string;
};

export function deployFromPayload(hit: SweepHit): SweepDeploy {
  const g = hit.body;
  const p = g.parallel || {};
  const tpA = posInt(p.tp_attention, 1);
  const par: Parallel = {
    tp_attention: tpA,
    tp_dense_ffn: posInt(p.tp_dense_ffn, 1),
    tp_moe_expert: posInt(p.tp_moe_expert, 1),
    ep_moe: posInt(p.ep_moe, 1),
    cp: posInt(p.cp, 1),
    pp: posInt(p.pp, 1),
    dp_attention: posInt(p.dp_attention, 1),
    dp_replicas: posInt(p.dp_replicas, 1),
  };
  const world = clusterOf(hit);
  const gpu = g.gpu_sku || "?";
  const parLab =
    `TP${par.tp_attention}/EP${par.ep_moe}/CP${par.cp}/DP${par.dp_attention}` +
    (par.pp > 1 ? `/PP${par.pp}` : "") +
    (par.dp_replicas > 1 ? `×${par.dp_replicas}` : "");
  return {
    gpu,
    world,
    ep: par.ep_moe,
    dpAttn: par.dp_attention,
    tpAttn: par.tp_attention,
    cp: par.cp,
    pp: par.pp,
    label: `${gpu} · ${parLab} · ${world ?? "—"} GPU`,
  };
}

function strField(v: unknown): string | undefined {
  return typeof v === "string" && v !== "" ? v : undefined;
}

function numFieldVal(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/** Request fields for one deployment config. The single-model estimate and
 *  every compare row build their body through this one function, so a row
 *  that received the template by "push" posts what the single-model run posts
 *  for the same config. Absent keys are omitted (the server default applies). */
export function estimateCfgFields(cfg: SweepCfg): EstimateCfgFields {
  const body: EstimateCfgFields = {};
  const gpu = strField(cfg.gpu_sku)?.trim();
  if (gpu) body.gpu_sku = gpu;
  const frac = numFieldVal(cfg.gpu_memory_fraction);
  if (frac !== undefined) body.gpu_memory_fraction = frac;
  const commLatency = numFieldVal(cfg.comm_latency_us);
  if (commLatency !== undefined) body.comm_latency_us = commLatency;
  const launchFloor = numFieldVal(cfg.launch_floor_us);
  if (launchFloor !== undefined) body.launch_floor_us = launchFloor;
  if (cfg.parallel && typeof cfg.parallel === "object") {
    body.parallel = { ...cfg.parallel };
  }
  const cpScheme = strField(cfg.cp_scheme);
  if (cpScheme) body.parallel = { ...(body.parallel || {}), cp_scheme: cpScheme };
  const rho = numFieldVal(cfg.ep_load_factor);
  if (rho !== undefined) body.parallel = { ...(body.parallel || {}), ep_load_factor: rho };
  const prefix = numFieldVal(cfg.prefix_cached_tokens);
  if (prefix !== undefined) body.prefix_cached_tokens = prefix;
  const chunk = numFieldVal(cfg.prefill_chunk_size);
  if (chunk !== undefined) body.prefill_chunk_size = chunk;
  const specMode = strField(cfg.speculative_mode);
  if (specMode) {
    const preset = SPEC_PRESETS[specMode] || SPEC_PRESETS.none;
    const k = specMode === "custom" ? numFieldVal(cfg.speculative_draft_tokens) : preset.K;
    const alpha = specMode === "custom" ? numFieldVal(cfg.acceptance_rate) : preset.alpha;
    if (k !== undefined) body.speculative_draft_tokens = k;
    if (alpha !== undefined) body.acceptance_rate = alpha;
  }
  const draft = specDraftStyle(cfg.speculative_draft);
  if (draft) body.speculative_draft = draft;
  if (typeof cfg.include_aux_weights === "boolean") {
    body.include_aux_weights = cfg.include_aux_weights;
  }
  if (typeof cfg.host_offload === "boolean") {
    body.host_offload = cfg.host_offload;
  }
  return body;
}

/** One POST /inference/estimate-sweep body per filled row. Missing GPU is an error, not a default SKU. */
export function sweepRequestBodies(
  rows: SweepRow[],
): { bodies: SweepRequestBody[] } | { error: string } {
  const bodies: SweepRequestBody[] = [];
  for (const [i, row] of rows.entries()) {
    const model_id = row.model_id.trim();
    if (!model_id) continue;
    const gpu_sku = strField(row.cfg?.gpu_sku)?.trim() || "";
    if (!gpu_sku) return { error: `Row ${i + 1} (${model_id}): pick a GPU.` };
    bodies.push({
      ...estimateCfgFields(row.cfg || {}),
      model_id,
      gpu_sku,
      seq_lens: SWEEP_SEQ_LENS,
      batch_grid: SWEEP_BATCH_GRID.map((p) => ({ ...p })),
      avg_output_tokens: SWEEP_OUTPUT_TOKENS,
    });
  }
  if (bodies.length < SWEEP_MIN_ROWS) {
    return { error: `At least ${SWEEP_MIN_ROWS} models required.` };
  }
  return { bodies };
}
