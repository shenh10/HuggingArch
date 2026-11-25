// Playground — module comparison (semantics B).
//
// Pick a base model + several same-role library components (e.g. MLA vs DSA vs
// GQA) and compare their canonical flops / weight-bytes / activation-bytes /
// kvcache under ONE fixed contract (batch, seq_len, hidden size). Every number
// comes from the backend `/api/playground/module-bench` endpoint — this page
// computes nothing, it only renders. Comparison is a DROP-IN replacement cost
// under one hidden size, not an architecture ranking; the caveat is stated in
// the UI and each variant's resolved internal dims are shown beside it.
// There is no run button: any change to the picks or the contract recomputes
// automatically (debounced).
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, fetchJson } from "../api";
import { authContext } from "../auth";
import { DataTable, tbl, type ColumnDef } from "../DataTable";
import { promptDialog } from "../Dialog";
import { AXIS, HaChart } from "../charts/HaChart";
import { ENTRY_PALETTE, type BlockNode, type BlockEdge } from "./BlockDag";
import { PerOpTable } from "./PhaseTable";
import { GraphDiffView, GraphShell, type DiffRow, type GraphDiffData, type L1Data } from "./Structure";
import { JumpBar } from "./pg/JumpBar";
import { LatencySection } from "./pg/LatencySweep";
import { PicksBar } from "./pg/PicksBar";
import { DescribeDrawer, DescribeForm, useMyModels } from "./pg/DescribeModel";
import { Section } from "./pg/Section";
import { SummaryTable, type MetricCol, type SummaryRow } from "./pg/SummaryTable";
import type { ModuleFoldRow, OpTableTotal, PerOpRow } from "./ResultSummary";
import { formulaHTML, type FormulaFields } from "../tooltip";
import { tipHandlers } from "../Tip";
import { binaryUnit, fmtSeqLabel, formatGiB, formatCount, formatDim, formatSI, formatTokens, setDimGlossary, dimMeaning } from "../format";
import { SWEEP_SEQ_LENS } from "../compare";
import { playgroundSearch, readPlaygroundState, type PlaygroundPick } from "../url";
import { AS_SHIPPED, PREDATES_PROFILES_NOTICE, precisionBody,
         type PrecisionReport } from "../precision";
import { PrecisionPresets } from "../PrecisionPresets";
import { DimSymbol } from "../DimSymbol";
import { gpuOptions, skuOf, type GpuItem } from "../gpu";
import type { EChartsOption } from "echarts";

type ComponentInfo = { name: string; role: string; attn_display?: string | null;
                       decode_variant?: string | null; description?: string | null;
                       model_ids?: string[] };
type ComponentsResponse = { success: boolean; components: ComponentInfo[] };
type GpusResponse = { items?: GpuItem[] };

type KVNode = {
  name: string;
  dtype: string;
  law: "growing" | "windowed" | "constant";
  window?: number;
  growing: boolean;
  elements_per_token: number | null;
  elements_per_request?: number;
  storage_bytes_at_L: number;
  read_bytes_at_L: number;
  formula?: FormulaFields;
  /** Bound slot that owns this cache; absent on the component's own tensor. */
  owner?: string;
  /** Slot-cache storage at the KV sweep's lengths; component-owned storage
   * stays in the parent `storage_by_cache_len` terms. */
  storage_by_cache_len?: { cache_len: number; storage_bytes: number }[];
};
type KV = {
  // the mechanism users see (the component's attn_display: KDA, DSA, MLA …)
  display: string | null;
  // the family the cache is shaped as (KDA keeps a GatedDeltaNet-shaped state)
  family: string | null;
  supported: boolean;
  unit: string;
  cache_len: number;
  elements_per_token: number;
  storage_bytes_at_L: number;
  read_bytes_at_L: number;
  // what each new token writes to this layer's cache (growing nodes)
  bytes_per_token: number;
  // constant per-request state (e.g. a recurrent state) — not per-token KV
  state_bytes: number;
  // storage per layer per request at each requested context length, and the
  // entry's model's own cache of this component at that length (its walk:
  // every cache-holding call at its own law)
  storage_by_cache_len: {
    cache_len: number; storage_bytes: number; model_storage_bytes?: number;
    formula?: FormulaFields; model_formula?: FormulaFields | null;
  }[];
  formulas?: { bytes_per_token?: FormulaFields; state_bytes?: FormulaFields };
  // how many layers of the entry's model hold this cache (0: not used there)
  // and why the entry has no whole model (`layers` null): it takes another
  // component's place, or the model does not run its call as it does
  model_layers?: { model_id: string; layers: number | null; refused?: string | null } | null;
  nodes: KVNode[];
  /** Graph-visible caches owned by bound sub-components. They explain the
   * graph's KV marks but stay outside this parent component's per-layer total. */
  slot_nodes?: KVNode[];
};
type PhaseKey = "prefill" | "decode";
type PhaseTiming = {
  gpu_sku: string;
  t_sol_s: number;
  t_measured_s: number | null;
  measured_rows: number;
  total_rows: number;
  strategy: string;
  // The per-op roofline rows the phase sums are rolled up from (post
  // calibration); the Details table renders these directly.
  op_rows?: PerOpRow[];
  /** The module tree over `op_rows`, as Inference's `flops_by_module`. */
  module_rows?: ModuleFoldRow[];
  /** The per-op table's Total row, over `module_rows`. */
  op_table_total?: OpTableTotal;
};
// One phase's numbers for a row (backend `phases.<phase>`). `kv_bytes` is what
// the phase does to the cache: prefill writes it, a decode step reads it.
// `runs_as` is the component actually evaluated — its declared decode form at
// decode, and at prefill for a sparse (top-k) attention.
type PhaseNumbers = {
  flops: number;
  act_read_bytes: number;
  act_write_bytes: number;
  act_read_bytes_fused: number;
  act_write_bytes_fused: number;
  // Backend totals for activation read + write. The summary table reads the
  // fused total; it does not add the split fields.
  act_bytes_naive: number;
  act_bytes_fused: number;
  kv_bytes: number;
  // the step's two memory streams: the weights it reads (serving's basis:
  // a routed expert bank only as far as the step's tokens reach) and the KV
  // bytes of every request in the batch; which one is larger
  weight_read_bytes: number;
  kv_step_bytes: number;
  // decode KV read for the whole step (decision 4). Prefill is 0.
  kv_read_bytes: number;
  memory_dominant: "kv" | "weights";
  // the backend's sentinel for a fused activation value above naive — a
  // fusion only removes intermediates, so it should never be set; a row that
  // sets it is an accounting bug to report
  fused_exceeds_naive: boolean;
  runs_as: string;
  units: { flops: string; act_bytes: string; kv_bytes: string; traffic: string; weight_read: string };
  // Summary-cell formula popovers: the number's composition from the engine's
  // primitive terms, substituted with the exact value the cell shows.
  formulas?: {
    flops?: FormulaFields;
    t_sol?: FormulaFields;
    weight_read?: FormulaFields;
    kv_step?: FormulaFields;
    act_bytes?: FormulaFields;
  };
};
// One stored copy of the block at the checkpoint's formats (backend
// `weight_storage`): the precision in words and whose checkpoint it is — a
// bound entry's model, or the base model for an unbound entry.
type WeightStorage = {
  bytes: number;
  parameter_bytes: number;
  metadata_bytes: number;
  buffer_bytes: number;
  by_dtype: { dtype: string; label: string; bytes: number }[];
  precision: string;
  source: { model_id: string | null; bound: boolean; invoked: boolean;
            role: string | null; address: string; note: string };
};
type RuntimeOpContract = {
  phase: PhaseKey;
  batch: number;
  input_tokens: number;
  fusion: "shipped";
  parallel: "single_rank_unsharded";
  /** The model's activation dtype as the backend states it; null on a
   *  table the editor synthesises with no rows. */
  activation_dtype: string | null;
  /** The precision profile the table was walked under. */
  precision: PrecisionReport | null;
};
type RuntimeOpRow = {
  node_id: string;
  uid?: number;
  kernel: string;
  label: string;
  origins: string[];
  params: number;
  active_params: number;
  flops: number;
  act_read_bytes: number;
  act_write_bytes: number;
  weight_read_bytes: number;
  kv_read_bytes: number;
  total_bytes: number;
  arithmetic_intensity: number | null;
  params_share: number;
  flops_share: number;
  bytes_share: number;
  formulas?: {
    params?: FormulaFields;
    flops?: FormulaFields;
    total_bytes?: FormulaFields;
    act?: FormulaFields;
    weight?: FormulaFields;
  };
};
type RuntimeOpTable = {
  contract: RuntimeOpContract;
  // The phase's numbers are these rows' totals (the summary cells read them).
  rows: RuntimeOpRow[];
  totals: {
    params: number;
    active_params: number;
    flops: number;
    total_bytes: number;
    arithmetic_intensity: number | null;
  };
};
// One call of a component in a model (`CallSite`): where it is, in words,
// and what the model hands the component there (prefill shapes, as
// expressions). A model that calls the component in several places is
// evaluated at the call the entry names.
type CallInfo = { call: string; label: string; component: string; formals: string[][] };

type BenchRow = {
  component: string;
  // Entry identity: bare component name when unbound, `name@model_id` when
  // bound — the same component bound to two models is two columns — and
  // `#call` after it when the entry names its call.
  entry?: string;
  model_id?: string | null;
  // The call this row was evaluated at, and every call the entry can name.
  call?: CallInfo;
  calls?: CallInfo[];
  role: string | null;
  flops: number;
  params: number;
  active_params: number;
  weight_bytes: number;
  weight_storage: WeightStorage;
  act_read_bytes: number;
  act_write_bytes: number;
  act_read_bytes_fused: number;
  act_write_bytes_fused: number;
  units: Record<string, string>;
  dims_used: Record<string, number>;
  editable_dims?: Record<string, number>;
  model_ids?: string[];
  kv: KV;
  phases: Record<PhaseKey, PhaseNumbers>;
  // Stored-column formula popovers (params / active / weight / KV storage).
  formulas?: {
    params?: FormulaFields;
    active_params?: FormulaFields;
    weight_bytes?: FormulaFields;
    kv_storage?: FormulaFields;
  };
  timings?: Record<PhaseKey, PhaseTiming>;
  /** Why there is no `timings`: the card lists no peak for the dtype the
   *  step computes in. FLOPs and bytes stand without it. */
  timings_unavailable?: string;
  block_graph?: { nodes: BlockNode[]; edges: BlockEdge[] };
  // The same prefill graph after the shipped-kernel rewrite. Resource rows in
  // `op_tables` are stamped onto matching post-fusion nodes.
  fused_graph?: { nodes: BlockNode[]; edges: BlockEdge[] };
  // Structure boxes over `fused_graph`, closed to the same runtime rows.
  fused_l1?: L1Data;
  // Structure view over block_graph (boxes by operand role), or why not.
  l1?: L1Data;
  // Backend-owned live-kernel rows after shipped fusions. One table per
  // phase; the UI never combines unfused and fused rows.
  op_tables: Record<PhaseKey, RuntimeOpTable>;
};
// One compare entry: a component, optionally bound to a shipped model. `dims`
// carries the user's edits plus the adopted binding seed; `seedDims` is the
// customization baseline (the "edited" marker compares against it); `seeded`
// tells the backend the seed is already in `dims` (so a cleared dim stays
// cleared). `label` overrides the entry's identity end-to-end — saved custom
// variants ride it (`name~my:<variant>`) so they never collide with the bare
// component name in results or the echo.
type Entry = {
  key: string; // name, name@model_id[#call], or name~my:<variant>
  name: string;
  model_id: string | null;
  // The call of the model the entry is evaluated at; null until the user
  // names one when the model calls the component more than once.
  call: string | null;
  // The calls the entry can name, as the last response listed them.
  calls?: CallInfo[];
  dims: Record<string, number>;
  seedDims?: Record<string, number>;
  seeded: boolean;
  label?: string;
};

// A saved "my module" variant (backend row) — a named custom parameterization
// of a component, usable as a compare entry.
type ModuleVariant = {
  id: string; component: string; name: string;
  dims: Record<string, number>;
  base_model_id?: string | null; is_public?: boolean; updated_at?: string;
};
const variantKey = (name: string, variantName: string) => `${name}~my:${variantName}`;
// Display label for any identity string (entry keys and response rows).
const entryLabel = (key: string) => key.includes("~my:") ? key.replace("~my:", " · ") : key;
const entryKey = (name: string, modelId: string | null, call: string | null = null) => {
  const key = modelId ? `${name}@${modelId}` : name;
  return call ? `${key}#${call}` : key;
};
// The call a column bound to `modelId` starts at: the model's first call of
// the component (its first parameterization) — when its calls differ, the
// column's call picker moves it to another. Null when the model lists none.
const firstCall = (rows: BindingRow[], modelId: string | null) =>
  rows.find((b) => b.model_id === modelId)?.call ?? null;
// A call as the picker lists it: where it is, then what it hands over.
const callText = (c: CallInfo) =>
  `${c.label} · ${c.formals.map((f) => `[${f.join(", ")}]`).join(" ")}`;

// Canonical signature of a dims map (sorted by KEY) — pick-time dedup, twin
// detection and the edited baseline all compare through this one helper.
const dimsSig = (d: Record<string, number>) =>
  JSON.stringify(Object.entries(d).sort(([a], [b]) => a.localeCompare(b)));
type BenchResponse = {
  success: boolean;
  message?: string;
  base_model_id: string;
  contract: { batch: number; input_tokens: number; dim_overrides: Record<string, number>;
              per_component_dims?: Record<string, Record<string, number>>;
              seed_dims?: Record<string, Record<string, number>>;
              // what each phase evaluated: s_query new tokens/request against s_kv
              phases: Record<PhaseKey, { batch: number; s_query: number; s_kv: number }> };
  // The compute-graph diff of every entry (two or more), or why there is none.
  graph_diff?: GraphDiffData;
  results: BenchRow[];
  errors: {
    component: string;
    entry?: string;
    model_id?: string | null;
    error: string;
    missing_dims?: string[];
    // The calls the entry can name — listed when it must name one.
    calls?: CallInfo[];
    // The call it was refused at, when it got that far.
    call?: CallInfo;
  }[];
};
// component-dims without a model_id: every call of the component in every
// shipped model that binds it, with the dims each resolves to, plus bindings
// that failed to resolve (with reasons) — a failed binding is reported, never
// silently dropped.
// `model_calls` is how many calls the model makes of the component, and
// `model_parameterizations` how many of them evaluate differently — the only
// fact a column may take its call from on its own (exactly one: the calls
// all evaluate alike, so any of them is the column's). A call that cannot be
// evaluated alone is still a row, with its reason and no dims.
type BindingRow = CallInfo & { model_id: string; dims: Record<string, number>;
                               model_calls: number; model_parameterizations: number;
                               /** Which of the model's parameterizations this call is. */
                               parameterization: number;
                               error?: string };
type BindingsResponse = { success: boolean; bindings: BindingRow[];
                          failed?: Record<string, string> };
type BindingsPack = {
  bindings: BindingRow[];
  failed: Record<string, string>;
};

const PALETTE = ENTRY_PALETTE;

// The mechanism an entry runs, then — when it differs — the family its cache
// is shaped as: "KDA · state shaped as GatedDeltaNet", "DSA · cache shaped as
// MLA", plain "MLA". Both names are the backend's; "state" is said of a cache
// that is only a flat recurrent state, as in kvShape.
function kvName(kv: KV, none: string): string {
  const name = kv.display ?? kv.family;
  if (!name) return none;
  if (!kv.family || kv.family === name) return name;
  const held = kv.state_bytes > 0 && kv.bytes_per_token === 0 ? "state" : "cache";
  return `${name} · ${held} shaped as ${kv.family}`;
}

// How an entry's cache grows with length, named from its nodes' laws: a
// recurrent state is flat, a window caps, a dense cache grows.
function KvNum({ bytes, formula }: {
  bytes: number; formula?: FormulaFields | null;
}) {
  const text = formatGiB(bytes);
  const f = formula;
  if (!f?.symbolic) return <>{text}</>;
  return (
    <span className="underline decoration-dotted" {...tipHandlers(formulaHTML(f))}>
      {text}
    </span>
  );
}

// Elements count in the aligned tensors fold: same dotted-underline popover
// as KvNum, over the identity of the number the cell shows.
function KvEl({ n }: { n: KVNode }) {
  const text = n.elements_per_token != null
    ? formatCount(n.elements_per_token)
    : `${formatCount(n.elements_per_request ?? 0)}/req`;
  if (!n.formula?.symbolic) return <>{text}</>;
  return (
    <span className="underline decoration-dotted" {...tipHandlers(formulaHTML(n.formula))}>
      {text}
    </span>
  );
}

function kvShape(kv: KV): string {
  if (kv.state_bytes > 0 && kv.bytes_per_token === 0) return "state · flat";
  const laws = Array.from(new Set(kv.nodes.map(kvLaw)));
  return laws.join(" + ");
}

type KvScope = "layer" | "model";

// KV storage per request across the Scaling sweep's context lengths: rows are
// entries, columns are lengths, every cell the backend's number (per layer, or
// the whole model's — per layer × the layers of the entry's model that hold
// this cache). The smallest cell of each column is marked.
function KvSweep({ resp, colorOf, wholeModel }: {
  resp: BenchResponse; colorOf: (r: BenchRow) => string; wholeModel?: boolean;
}) {
  const [scope, setScope] = useState<KvScope>(wholeModel ? "model" : "layer");
  const rows = resp.results.filter((r) => r.kv.storage_by_cache_len?.length);
  if (rows.length === 0) return null;
  const lens = rows[0].kv.storage_by_cache_len.map((p) => p.cache_len);
  const extras = !wholeModel;
  const valueOf = (r: BenchRow, i: number) => {
    const p = r.kv.storage_by_cache_len[i];
    if (wholeModel) return p.storage_bytes;
    return scope === "model" ? (p.model_storage_bytes ?? 0) : p.storage_bytes;
  };
  const usable = (r: BenchRow) => wholeModel || scope === "layer" || (r.kv.model_layers?.layers ?? 0) > 0;
  const minOf = lens.map((_, i) => Math.min(...rows.filter(usable).map((r) => valueOf(r, i))));
  const btn = (k: KvScope) =>
    "px-2 py-1 text-xs " + (scope === k ? "bg-gray-700 text-white" : "bg-white text-gray-600 hover:bg-gray-50");
  const seen = new Map<string, number>();
  const dash = (r: BenchRow) => {
    const k = seen.get(r.component) ?? 0;
    seen.set(r.component, k + 1);
    return (["solid", "dashed", "dotted"] as const)[k % 3];
  };
  // Plotted in one binary unit for the whole chart, so the axis steps are
  // round numbers of it (1, 2, 4 GiB), not bytes cut into fifths
  // ("954 MiB", "763 MiB", and "1 GiB" twice).
  const unit = binaryUnit(Math.max(0, ...rows.filter(usable).flatMap((r) => lens.map((_, i) => valueOf(r, i)))));
  const option: EChartsOption = {
    grid: { left: 72, right: 16, top: 40, bottom: 40 },
    tooltip: { trigger: "axis", order: "valueDesc", valueFormatter: (v) => formatGiB(Number(v) * unit.scale) },
    legend: { top: 0, textStyle: { fontSize: 9 } },
    xAxis: { type: "log", logBase: 2, min: Math.max(1, lens[0] || 1),
             name: "cache_len (tokens)", nameLocation: "middle", nameGap: 24, ...AXIS,
             axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => fmtSeqLabel(v) },
             axisPointer: { label: { formatter: ({ value }) => fmtSeqLabel(Number(value)) } } },
    yAxis: { type: "value", ...AXIS, axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => (v === 0 ? "0" : `${+v.toPrecision(4)} ${unit.unit}`) } },
    series: rows.filter(usable).map((r) => ({
      type: "line" as const,
      name: rowLabel(r),
      data: lens.map((n, i) => [n, valueOf(r, i) / unit.scale]),
      lineStyle: { color: colorOf(r), type: dash(r), width: 2 },
      itemStyle: { color: colorOf(r) },
      symbolSize: 5,
    })),
  };
  const columns: ColumnDef<BenchRow>[] = [
    {
      key: "entry",
      header: "entry",
      className: "whitespace-nowrap",
      sortValue: (r) => rowLabel(r),
      render: (r) => (
        <>
          <span className="inline-block w-2.5 h-2.5 rounded-sm mr-1.5 align-middle" style={{ background: colorOf(r) }} />
          <span className="font-mono text-gray-800">{rowLabel(r)}</span>
          {extras && <div className="pg-kv-mech text-11 text-gray-500 pl-4">{kvName(r.kv, "no-cache")}</div>}
        </>
      ),
    },
  ];
  if (extras) {
    columns.push({
      key: "shape",
      header: "shape",
      className: "text-gray-500 whitespace-nowrap",
      render: (r) => kvShape(r.kv),
      sortValue: (r) => kvShape(r.kv),
    });
    if (scope === "model") {
      columns.push({
        key: "layers",
        header: "layers",
        align: "right",
        className: "text-gray-500 whitespace-nowrap",
        cellProps: (r) => ({
          title: r.kv.model_layers?.refused
            ?? (r.kv.model_layers ? `layers of ${r.kv.model_layers.model_id} that hold this cache` : undefined),
        }),
        render: (r) => (!r.kv.model_layers || r.kv.model_layers.layers == null
          ? "no whole model"
          : r.kv.model_layers.layers > 0
            ? `× ${r.kv.model_layers.layers}`
            : `not in ${shortModel(r.kv.model_layers.model_id)}`),
        sortValue: (r) => r.kv.model_layers?.layers,
      });
    }
    columns.push({
      key: "per-token",
      header: "per token",
      align: "right",
      className: "font-mono whitespace-nowrap text-gray-800",
      cellProps: (r) => ({ "data-value": r.kv.bytes_per_token }),
      render: (r) => <KvNum bytes={r.kv.bytes_per_token} formula={r.kv.formulas?.bytes_per_token} />,
      sortValue: (r) => r.kv.bytes_per_token,
    });
    columns.push({
      key: "state",
      header: "state",
      align: "right",
      className: "font-mono whitespace-nowrap text-gray-800",
      cellProps: (r) => ({ "data-value": r.kv.state_bytes }),
      render: (r) => <KvNum bytes={r.kv.state_bytes} formula={r.kv.formulas?.state_bytes} />,
      sortValue: (r) => r.kv.state_bytes,
    });
  }
  lens.forEach((n, i) => {
    const atL = n === resp.contract.input_tokens;
    const bestOf = (r: BenchRow) => usable(r) && valueOf(r, i) === minOf[i] && rows.filter(usable).length > 1;
    columns.push({
      key: `len-${n}`,
      header: <span title={`cache_len = ${n.toLocaleString("en-US")} tokens`}>{fmtSeqLabel(n)}</span>,
      align: "right",
      headerClassName: atL ? "pg-kv-at-L bg-amber-50 text-amber-800" : undefined,
      className: (r) =>
        `pg-kv-cell font-mono whitespace-nowrap ${atL ? "pg-kv-at-L bg-amber-50" : ""} ${bestOf(r) ? "pg-kv-min text-emerald-700 font-semibold" : "text-gray-800"}`,
      cellProps: (r) => ({ title: bestOf(r) ? "smallest at this length" : undefined }),
      render: (r) => (usable(r)
        ? <KvNum
            bytes={valueOf(r, i)}
            formula={scope === "model" && !wholeModel
              ? r.kv.storage_by_cache_len[i]?.model_formula
              : r.kv.storage_by_cache_len[i]?.formula}
          />
        : "—"),
      sortValue: (r) => (usable(r) ? valueOf(r, i) : null),
    });
  });
  return (
    <div id="pg-kv-sweep" className="mt-5">
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <div className="text-sm font-medium text-gray-700">
          KV storage per request vs context length{" "}
          <span className="text-xs font-normal text-gray-400">
            {wholeModel
              ? "· whole model"
              : scope === "layer"
                ? "· per layer · each cache slot as the model ships it"
                : "· whole model: the entry model's own cache of this component, every call at its own law"}
          </span>
        </div>
        {!wholeModel && (
        <div className="inline-flex rounded border border-gray-200 overflow-hidden" role="tablist" aria-label="KV scope">
          <button id="pg-kv-scope-layer" type="button" className={btn("layer")} onClick={() => setScope("layer")}>per layer</button>
          <button id="pg-kv-scope-model" type="button" className={btn("model")} onClick={() => setScope("model")}>× layers (whole model)</button>
        </div>
        )}
      </div>
      <div className="mt-2">
        <DataTable columns={columns} rows={rows} rowKey={(r) => r.entry ?? r.component} framed={false} />
        <div className="text-11 text-gray-400 mt-1">green = smallest in its column</div>
      </div>
      <div className="mt-3">
        <HaChart option={option} height={260} />
      </div>
    </div>
  );
}

// KV sweep (independent of phase) plus the per-tensor fold. Totals live in
// the summary table; this section is the length axis and the tensor list.
function KvSection({ resp, colorOf, decodeOf, descOf, serving }: {
  resp: BenchResponse; colorOf: (r: BenchRow) => string;
  decodeOf: Map<string, string>; descOf: Map<string, string>;
  serving: ReactNode;
}) {
  return (
    <Section id="pg-kv" scope="independent of phase" title="KV" lead={serving}>
      <KvSweep resp={resp} colorOf={colorOf} />
      <KvTensors results={resp.results} colorOf={colorOf} decodeOf={decodeOf} descOf={descOf} />
    </Section>
  );
}

/** B / L / GPU — the serving contract. Shared state, rendered on the
 *  sections that consume it (summary, KV). Structure / params /
 *  weight bytes do not. Precision is the page's profile switch. */
function ServingFields({ batch, seqLen, gpuSku, gpus, onBatch, onSeqLen, onGpuSku }: {
  batch: number; seqLen: number;
  gpuSku?: string; gpus?: GpuItem[];
  onBatch: (n: number) => void; onSeqLen: (n: number) => void;
  onGpuSku?: (sku: string) => void;
}) {
  return (
    <div className="pg-serving flex flex-wrap items-end gap-x-4 gap-y-2 mb-3" aria-label="Serving contract">
      <label className="block">
        <span className="block text-xs font-medium text-gray-600 mb-0.5">Batch (B)</span>
        <input type="number" min={1} value={batch}
          onChange={(e) => onBatch(Math.max(1, Number(e.target.value)))}
          className="pg-serving-batch w-24 text-sm border border-gray-300 rounded-lg px-2 py-1 bg-white" />
      </label>
      <label className="block">
        <span className="block text-xs font-medium text-gray-600 mb-0.5">Sequence length (L)</span>
        <input type="number" min={1} value={seqLen}
          onChange={(e) => onSeqLen(Math.max(1, Number(e.target.value)))}
          className="pg-serving-seq w-28 text-sm border border-gray-300 rounded-lg px-2 py-1 bg-white" />
      </label>
      {gpuSku !== undefined && onGpuSku ? (
        <label className="block">
          <span className="block text-xs font-medium text-gray-600 mb-0.5">GPU</span>
          <select value={gpuSku} onChange={(e) => onGpuSku(e.target.value)}
            className="pg-serving-gpu text-sm border border-gray-300 rounded-lg px-2 py-1 bg-white">
            {gpuOptions(gpus || [], gpuSku).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>
      ) : null}
    </div>
  );
}

// Color key for one response: entry → palette index by its position in the
// result rows (the order the backend's graph_diff.entries shares). Bars and the
// diff graph read the same map, so an entry is one color everywhere on the
// page — two bindings of one component are two colours — and colors track the
// response on screen, not a newer pick.
const entryOf = (r: BenchRow) => r.entry ?? r.component;
function respColorIdx(resp: BenchResponse): Map<string, number> {
  const m = new Map<string, number>();
  for (const r of resp.results) if (!m.has(entryOf(r))) m.set(entryOf(r), m.size % PALETTE.length);
  return m;
}

// The comparison contract a fresh page starts on; a shared URL names only
// the fields that differ from it.
const pageDefaults = {
  role: "attention", batch: 1, seq: 4096, prec: AS_SHIPPED, gpu: "B200",
  phase: "prefill" as PhaseKey,
};
const unlessDefault = <K extends keyof typeof pageDefaults>(k: K, v: (typeof pageDefaults)[K]) =>
  v === pageDefaults[k] ? undefined : v;

export function Playground() {
  // The comparison the page was opened with (`?pick=…&batch=…`, written back
  // on every change so the URL shares it; Compare's representative blocks
  // link here too): its columns are benched on arrival. The catalog opens on
  // the link's `role`; a link without one opens on the role of what it
  // compares (its first pick's), once the library is loaded.
  const [opened] = useState(() => readPlaygroundState(window.location.search));
  const [components, setComponents] = useState<ComponentInfo[]>([]);
  const [componentsLoaded, setComponentsLoaded] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [role, setRole] = useState(opened.role ?? pageDefaults.role);
  const roleFromPicks = useRef(!opened.role && opened.picks.length > 0);
  useEffect(() => {
    if (!roleFromPicks.current || !components.length) return;
    roleFromPicks.current = false;
    const picked = components.find((c) => c.name === opened.picks[0].name);
    if (picked) setRole(picked.role);
  }, [components, opened]);
  // The compare set: component entries bound to a shipped model.
  // A chip pick adds one entry per DISTINCT binding parameterization (identical
  // dims collapse into one column); a component nothing binds is an error.
  // A labeled column (a my module) is its dims; a bound one is its binding
  // plus the edits the link carries, seeded on the first run.
  const [entries, setEntries] = useState<Entry[]>(() => opened.picks.map((p) => {
    const dims = { ...p.dims };
    return p.label
      ? { key: p.label, name: p.name, model_id: p.model_id, call: p.call, dims,
          seedDims: { ...dims }, seeded: true, label: p.label }
      : { key: entryKey(p.name, p.model_id, p.call), name: p.name, model_id: p.model_id,
          call: p.call, dims, seeded: false };
  }));
  // Structural selection changes stage here. Contract/dim edits stay debounced,
  // but picking several modules must not launch (and cancel) a bench per chip.
  const [selectionDirty, setSelectionDirty] = useState(false);
  const [batch, setBatch] = useState(opened.batch ?? pageDefaults.batch);
  const [seqLen, setSeqLen] = useState(opened.seq ?? pageDefaults.seq);
  // The page's precision profile (preset id), shared by both modes and kept
  // in the URL as `prec=`; as shipped by default.
  const [precPreset, setPrecPreset] = useState(opened.prec ?? pageDefaults.prec);
  const [precisionFromLink] = useState(() => opened.prec !== undefined);
  const [precNotice, setPrecNotice] = useState<string | null>(
    opened.retiredPrecision ? PREDATES_PROFILES_NOTICE : null);
  const [gpuSku, setGpuSku] = useState(opened.gpu ?? pageDefaults.gpu);
  const [gpus, setGpus] = useState<GpuItem[]>([]);
  const [newModuleFor, setNewModuleFor] = useState<string | null>(null);
  const [resp, setResp] = useState<BenchResponse | null>(null);
  const [status, setStatus] = useState("");
  const [statusKind, setStatusKind] = useState<"idle" | "ok" | "err">("idle");
  const [busy, setBusy] = useState(false);
  const [gapNote, setGapNote] = useState(opened.malformed.length
    ? `Ignored link columns that do not parse: ${opened.malformed.join(", ")}` : "");
  const [mode, setMode] = useState<"compare" | "editor">("compare");
  // Which phase the metric charts show (both come back in every response).
  const [chartPhase, setChartPhase] = useState<PhaseKey>(
    opened.phase === "decode" ? "decode" : pageDefaults.phase);
  // The chip catalog (role / families / bindings) tucks away after the first
  // results so the structure section stays on the first screen. The compact
  // picks bar stays visible. + Add (`#pg-add-pick`) opens the catalog again.
  const [catalogOpen, setCatalogOpen] = useState(true);
  const hadResults = useRef(false);
  const picksRef = useRef<HTMLDivElement>(null);
  // Bound models: every shipped binding of each selected component, with the
  // dims each resolves to (null = loading). Mixed architectures share one
  // table, pivoted so a model is one row and each component a column group.
  // Library components only: a block a model declares itself (a column opened
  // from Compare) is that model's alone, bound to nothing else.
  const uniqueNames = useMemo(() => {
    const library = new Set(components.map((c) => c.name));
    const names: string[] = [];
    for (const e of entries) if (library.has(e.name) && !names.includes(e.name)) names.push(e.name);
    return names;
  }, [entries, components]);
  const namesKey = uniqueNames.join("\0");
  const [bindingsByName, setBindingsByName] = useState<Record<string, BindingsPack | null>>({});

  useEffect(() => {
    const names = namesKey.length === 0 ? [] : namesKey.split("\0");
    if (names.length === 0) { setBindingsByName({}); return; }
    let dead = false;
    setBindingsByName((prev) => {
      const next: Record<string, BindingsPack | null> = {};
      for (const n of names) next[n] = n in prev ? prev[n] : null;
      return next;
    });
    for (const name of names) {
      fetchJson<BindingsResponse>(`/api/playground/component-dims?component=${encodeURIComponent(name)}`)
        .then((d) => {
          if (dead) return;
          setBindingsByName((prev) => ({
            ...prev,
            [name]: d.success
              ? { bindings: d.bindings || [], failed: d.failed || {} }
              : { bindings: [], failed: {} },
          }));
        })
        .catch(() => {
          if (!dead) setBindingsByName((prev) => ({ ...prev, [name]: { bindings: [], failed: {} } }));
        });
    }
    return () => { dead = true; };
  }, [namesKey]);

  // "My modules" per selected component — saved custom parameterizations,
  // usable as compare entries. Fetched lazily; `signedIn` (a live session
  // per /api/auth/me) gates the ☆ save button too, so anonymous users
  // never get bounced to login and lose the workspace they were about to save.
  // Auth is independent of the selection: mixed architectures still save.
  const [variantsByName, setVariantsByName] = useState<Record<string, ModuleVariant[]>>({});
  const [signedIn, setSignedIn] = useState(false);
  const [variantsReload, setVariantsReload] = useState(0);
  useEffect(() => {
    const names = namesKey.length === 0 ? [] : namesKey.split("\0");
    let dead = false;
    void (async () => {
      const ctx = await authContext();
      if (dead) return;
      const authed = ctx.authEnabled && ctx.loggedIn;
      setSignedIn(authed);
      if (!authed || names.length === 0) { setVariantsByName({}); return; }
      const next: Record<string, ModuleVariant[]> = {};
      await Promise.all(names.map(async (name) => {
        try {
          const d = await fetchJson<{ success: boolean; variants?: ModuleVariant[] }>(
            `/api/account/modules?component=${encodeURIComponent(name)}`);
          next[name] = d.success ? d.variants || [] : [];
        } catch { next[name] = []; }
      }));
      if (!dead) setVariantsByName(next);
    })();
    return () => { dead = true; };
  }, [namesKey, variantsReload]);

  // Save an edited column as "my module": the column's component + its
  // RESOLVED full own dims (editable_dims — the same allowlisted set the
  // table shows), so the variant is a complete parameterization, not a
  // fragment the next base would silently fill in differently. The button
  // only renders when signed in and settled (not mid-bench).
  const saveVariant = useCallback(async (name: string, entryName: string, dims: Record<string, number>) => {
    try {
      const d = await fetchJson<{ success: boolean; message?: string }>("/api/account/modules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ component: entryName, name, dims,
                               base_model_id: entriesRef.current.find((e) => e.name === entryName)?.model_id || null }),
      });
      if (!d.success) throw new Error(d.message || "save failed");
      setGapNote(`Saved as my module “${name}” — it appears under Bound models → mine`);
      setVariantsReload((v) => v + 1);
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
        setGapNote("Sign in to save modules — your selection is kept here, nothing was saved.");
        return;
      }
      setGapNote(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, []);

  useEffect(() => {
    fetchJson<GpusResponse>("/inference/gpus")
      .then((d) => {
        const items = (d.items || []).filter((g) => skuOf(g));
        const skus = items.map(skuOf);
        setGpus(items);
        setGpuSku((current) => current && skus.includes(current) ? current : skus[0] || "");
      })
      .catch(() => setGpus([]));
  }, []);

  // Add a saved variant as a compare entry — dims from the DB row (the
  // variant's whole point: data, not guesses). `label` carries its identity
  // through the bench so it never collides with the bare component name.
  const addVariant = useCallback((v: ModuleVariant) => {
    setGapNote("");
    setSelectionDirty(true);
    setEntries((prev) => {
      const key = variantKey(v.component, v.name);
      if (prev.some((e) => e.key === key) || prev.length >= 8) return prev; // bench cap
      return [...prev, { key, name: v.component, model_id: v.base_model_id || null,
                         call: null, dims: { ...v.dims }, seedDims: { ...v.dims },
                         seeded: true, label: key }];
    });
  }, []);

  // "New module" modal: add the freshly customized parameterization as its own
  // seeded column (seedDims = dims, so it isn't flagged "edited" against
  // itself), then optionally persist it to "My modules". The bench takes raw
  // dims keyed off `label`, so an unsaved module compares identically — saving
  // only makes it reusable.
  const createModule = useCallback(async (
    component: string, name: string, dims: Record<string, number>, save: boolean,
  ) => {
    const key = variantKey(component, name);
    setGapNote("");
    setSelectionDirty(true);
    setEntries((prev) => {
      if (prev.some((e) => e.key === key) || prev.length >= 8) return prev; // bench cap
      const column = prev.find((e) => e.name === component && e.model_id);
      const rows = bindingsByName[component]?.bindings ?? [];
      const host = column?.model_id ?? rows[0]?.model_id ?? null;
      const call = column ? column.call : firstCall(rows, host);
      return [...prev, { key, name: component, model_id: host, call, dims: { ...dims },
                         seedDims: { ...dims }, seeded: true, label: key }];
    });
    if (save) await saveVariant(name, component, dims);
  }, [saveVariant, bindingsByName]);

  // The "base dims" binding option is offered only when the base model itself
  // binds the component (base ∈ its model_ids) — then it is an ordinary
  // binding on the same seed path. No probe endpoint second-guessing the eval.
  const canUseBase = useCallback((_name: string) => false, []);

  useEffect(() => {
    fetchJson<ComponentsResponse>("/api/playground/components")
      .then((d) => {
        if (!d.success || !d.components?.length) throw new Error("no components returned");
        setComponents(d.components);
        setComponentsLoaded(true);
      })
      .catch((e) => {
        setComponents([]);
        setComponentsLoaded(true);
        setStatus(`Failed to load components: ${e instanceof Error ? e.message : String(e)}`);
        setStatusKind("err");
      });
    fetchJson<{ items?: { id?: string; model_id?: string }[]; model_ids?: string[] }>("/supported-models")
      .then((d) => setModels(d.model_ids || (d.items || []).map((i) => i.id || i.model_id || "").filter(Boolean)))
      .catch((e) => {
        setModels([]);
        setStatus(`Failed to load model list: ${e instanceof Error ? e.message : String(e)}`);
        setStatusKind("err");
      });
    // Dim glossary — feeds the semantic labels on the Gaps card's missing-dim
    // chips and the editable internal-dims rows (a raw `Cdim_io` or
    // `n_route_targets` means nothing without it).
    fetchJson<{ items?: Record<string, string> }>("/dim-glossary")
      .then((j) => {
        if (j && j.items && typeof j.items === "object") setDimGlossary(j.items);
      })
      .catch(() => { /* dim notes disappear; non-blocking */ });
  }, []);

  const roles = useMemo(() => Array.from(new Set(components.map((c) => c.role))).sort(), [components]);
  // The graphs' rows, rebuilt per response rather than per render, so a
  // re-render that brings no new response (a phase click) does not re-lay
  // the graphs out.
  const diffRows = useMemo(() => (resp?.results ?? []).map((r) => ({
    label: rowLabel(r), nodes: r.block_graph?.nodes ?? [], edges: r.block_graph?.edges ?? [], l1: r.l1,
    fused: r.fused_graph,
    fusedL1: r.fused_l1,
  })), [resp]);
  // component → the component it is the declared decode-time (absorbed) form of
  const decodeOf = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of components) if (c.decode_variant) m.set(c.decode_variant, c.name);
    return m;
  }, [components]);
  // spec-authored description per component (the decode-form tooltip quotes it
  // instead of the UI inventing reasons)
  const descOf = useMemo(() => {
    const m = new Map<string, string>();
    for (const c of components) if (c.description) m.set(c.name, c.description);
    return m;
  }, [components]);
  // THE color key: entry → palette index by position IN THIS RESPONSE's rows
  // (the order the backend's graph_diff.entries shares), so bars and the diff
  // graph color an entry identically — derived from the response on screen,
  // not from a newer pick still in flight.
  // (respColorIdx is a module-level helper below.)
  const roleComponents = useMemo(() => components.filter((c) => c.role === role), [components, role]);
  // Group the role's components by their declared family (attn_display), so the
  // picker reads as MLA / GQA / SWA / DSA / … instead of one flat wall. Sorted
  // by group, undeclared families collected under "Other".
  const familyGroups = useMemo(() => {
    const by = new Map<string, string[]>();
    for (const c of roleComponents) {
      const fam = c.attn_display || "Other";
      (by.get(fam) ?? by.set(fam, []).get(fam)!).push(c.name);
    }
    return Array.from(by.entries()).sort((a, b) => {
      if (a[0] === "Other") return 1;
      if (b[0] === "Other") return -1;
      return a[0].localeCompare(b[0]);
    });
  }, [roleComponents]);
  // Only show group headers when there's real grouping. A role with no declared
  // categories (block / ffn / router) would otherwise render a lone "Other · N"
  // header that reads like an error — flatten to plain chips instead.
  const showGroupHeaders = familyGroups.length > 1;

  // Picker presentation state: families start collapsed (60+ attention chips
  // are unreadable flat) and a substring filter narrows/auto-expands groups.
  // `entries` itself is untouched — collapse/filter is display-only.
  const [openFamilies, setOpenFamilies] = useState<Set<string>>(new Set());
  const [pickerFilter, setPickerFilter] = useState("");
  const filterLower = pickerFilter.trim().toLowerCase();
  const visibleGroups = useMemo(
    () => familyGroups
      .map(([family, names]) => ({
        family,
        names,
        shown: filterLower ? names.filter((n) => n.toLowerCase().includes(filterLower)) : names,
        selected: names.reduce((s, n) => s + (entries.some((e) => e.name === n) ? 1 : 0), 0),
      }))
      .filter((g) => !filterLower || g.shown.length > 0),
    [familyGroups, filterLower, entries],
  );

  const toggleFamily = useCallback((family: string) => {
    setOpenFamilies((prev) => {
      const next = new Set(prev);
      next.has(family) ? next.delete(family) : next.add(family);
      return next;
    });
  }, []);

  // A chip picks the component as its BINDINGS matrix: one entry per DISTINCT
  // parameterization across the shipped models that bind it (identical dims
  // collapse into one column labeled by a representative model — V3.2 and
  // V3.2-Exp parameterize a module identically, so two columns would be noise,
  // not information). Every column then carries a known model provenance; a
  // component nothing binds falls back to the unbound entry (base dims, and a
  // facts-only gap card if it can't resolve). Clicking again removes all its
  // entries — or cancels the in-flight pick.
  //
  // The pick is async (fetch bindings first), so it carries an epoch, like
  // runSeq for bench: a role switch or clear bumps `pickEpoch` and stale
  // in-flight picks are dropped on landing; `pendingPicks` marks the chip as
  // loading and makes a second click a cancel.
  const pickEpoch = useRef(0);
  const [pendingPicks, setPendingPicks] = useState<Set<string>>(new Set());
  const applySelection = useCallback(() => {
    if (!selectionDirty || entries.length === 0 || pendingPicks.size > 0) return;
    setCatalogOpen(false);
    setSelectionDirty(false);
  }, [selectionDirty, entries.length, pendingPicks.size]);
  const toggle = useCallback((name: string) => {
    setGapNote(""); // selection structure changed — any note is stale
    if (pendingPicks.has(name)) { // second click while loading = cancel
      pickEpoch.current++;
      setPendingPicks((prev) => { const n = new Set(prev); n.delete(name); return n; });
      return;
    }
    if (entriesRef.current.some((e) => e.name === name)) {
      setSelectionDirty(true);
      setEntries((prev) => prev.filter((e) => e.name !== name));
      return;
    }
    const epoch = pickEpoch.current;
    setPendingPicks((prev) => new Set(prev).add(name));
    void (async () => {
      let rows: BindingRow[] = [];
      try {
        const d = await fetchJson<BindingsResponse>(`/api/playground/component-dims?component=${encodeURIComponent(name)}`);
        if (d.success) rows = d.bindings || [];
      } catch { /* fall through to the unbound entry */ }
      setPendingPicks((prev) => { const n = new Set(prev); n.delete(name); return n; });
      if (epoch !== pickEpoch.current) return; // cancelled / role switched away
      setSelectionDirty(true);
      // One column per distinct parameterization across the models, each at
      // the first call that makes it (with its dims): a model whose calls
      // evaluate differently brings one column per parameterization, never
      // a column with no call. Identical dims of different models collapse
      // into one column; one model's parameterizations stay apart even at
      // equal dims — the backend split them on something else (a stored
      // format, a bound slot).
      const broughtBy = new Map<string, string>(); // dims signature → model
      const variants: { model_id: string; call: string; dims: Record<string, number> }[] = [];
      for (const b of rows) {
        if (rows.find((x) => x.model_id === b.model_id
            && x.parameterization === b.parameterization) !== b) continue;
        const sig = dimsSig(b.dims);
        const by = broughtBy.get(sig);
        if (by !== undefined && by !== b.model_id) continue;
        broughtBy.set(sig, b.model_id);
        variants.push({ model_id: b.model_id, call: b.call, dims: b.dims });
      }
      // Decided against the entries on screen now, outside the state
      // updater: the note is read right after, and an updater runs later.
      const prev = entriesRef.current;
      if (prev.some((e) => e.name === name)) return; // toggled meanwhile
      const room = Math.max(8 - prev.length, 0); // backend caps a bench at 8 entries
      const add: Entry[] = variants.slice(0, room).map((b) => Object.keys(b.dims).length
        ? { key: entryKey(name, b.model_id, b.call), name, model_id: b.model_id, call: b.call,
            dims: { ...b.dims }, seedDims: { ...b.dims }, seeded: true }
        : { key: entryKey(name, b.model_id, b.call), name, model_id: b.model_id, call: b.call,
            dims: {}, seeded: false });
      let capNote = "";
      if (variants.length === 0) {
        capNote = `${name}: no shipped model binds this component`;
      } else if (room === 0) {
        capNote = `${name} not added: the bench holds 8 columns and is full — remove a column first`;
      } else if (variants.length > room) {
        capNote = `${name}: ${variants.length} distinct parameterizations, showing ${add.length} (the bench holds 8 columns)`;
      }
      if (add.length) {
        entriesRef.current = [...prev, ...add];
        setEntries((cur) => (cur.some((e) => e.name === name) ? cur : [...cur, ...add]));
      }
      if (capNote) setGapNote(capNote);
    })();
  }, [pendingPicks]);

  const removeEntry = useCallback((key: string) => {
    setGapNote("");
    setSelectionDirty(true);
    setEntries((prev) => prev.filter((e) => e.key !== key));
  }, []);

  // The bindings table's "+ compare": add this model's parameterization of the
  // component at one of its calls as its own entry. `dims` IS the binding's
  // seed (and the customization baseline for the "edited" marker) — the entry
  // is marked seeded so the backend won't re-seed over later edits. A null
  // `call` (a model calling it in several places) waits for the user's pick.
  const addBinding = useCallback((name: string, modelId: string, call: string | null,
                                  dims: Record<string, number>) => {
    setGapNote("");
    setSelectionDirty(true);
    setEntries((prev) => {
      const key = entryKey(name, modelId, call);
      if (prev.some((e) => e.key === key) || prev.length >= 8) return prev; // bench cap
      return [...prev, call
        ? { key, name, model_id: modelId, call, dims: { ...dims },
            seedDims: { ...dims }, seeded: true }
        : { key, name, model_id: modelId, call: null, dims: {}, seeded: false }];
    });
  }, []);

  // Name the call an entry is evaluated at ("evaluate at which call"): a
  // fresh key, dims reset and seeded from that call on the next run, the
  // listed calls kept so the choice stays visible. A saved variant keeps its
  // own key and dims — they are the user's, not the call's.
  // When the model calls the entry's component itself, its listed calls are
  // that model's own (a block's include every layer stage of its stack), so
  // naming one evaluates the component that call runs — the column moves to
  // that block. Otherwise the entry takes the place of what the call runs.
  const pickCall = useCallback((key: string, call: string) => {
    setGapNote("");
    setEntries((prev) => {
      const cur = prev.find((e) => e.key === key);
      if (!cur || cur.call === call) return prev;
      if (cur.label) return prev.map((e) => e.key === key ? { ...e, call } : e);
      const listed = cur.calls ?? [];
      const picked = listed.find((c) => c.call === call);
      const name = picked && listed.some((c) => c.component === cur.name) ? picked.component : cur.name;
      const newKey = entryKey(name, cur.model_id, call);
      if (prev.some((e) => e.key === newKey)) return prev;
      return prev.map((e) => e.key === key
        ? { key: newKey, name, model_id: e.model_id, call, calls: e.calls,
            dims: {}, seeded: false }
        : e);
    });
  }, []);

  // Rebind an entry to another shipped model at its first call: fresh key,
  // dims reset, seeded back to false so the backend seeds from the new
  // binding on the next run.
  const rebindEntry = useCallback((key: string, modelId: string | null) => {
    setGapNote("");
    setSelectionDirty(true);
    setEntries((prev) => {
      const cur = prev.find((e) => e.key === key);
      if (!cur) return prev;
      const call = firstCall(bindingsByName[cur.name]?.bindings ?? [], modelId);
      const newKey = entryKey(cur.name, modelId, call);
      if (newKey !== key && prev.some((e) => e.key === newKey)) return prev;
      return prev.map((e) => e.key === key
        ? { key: newKey, name: e.name, model_id: modelId, call, dims: {}, seeded: false }
        : e);
    });
  }, [bindingsByName]);

  // Contract/dim changes re-post after a short debounce. Structural selection
  // changes first stage behind `selectionDirty` and wait for the sticky Benchmark
  // button, so several picks become one request instead of a cancel/restart storm.
  // `runSeq` drops stale responses (bumped BOTH at run start and whenever the
  // effect reschedules, so an in-flight response can't land after the user has
  // already moved on).
  const runSeq = useRef(0);
  const entriesRef = useRef(entries);
  useEffect(() => { entriesRef.current = entries; }, [entries]);
  // Canonical request body (seeded flag excluded — it gates re-seeding, not the
  // numbers). `lastPosted` lets echo adoption skip a redundant repost.
  const lastPosted = useRef("");
  const canonBody = useCallback((list: Entry[]) => JSON.stringify({
    entries: list.map((e) => ({ name: e.name, model_id: e.model_id, call: e.call, dims: e.dims, label: e.label ?? null })),
    batch_size: batch,
    seq_len: seqLen,
    ...precisionBody(precPreset),
    gpu_sku: gpuSku,
    // KV storage is also priced at every length the Scaling sweep uses.
    kv_cache_lens: SWEEP_SEQ_LENS,
  }), [batch, seqLen, precPreset, gpuSku]);
  const bodyJson = canonBody(entries);
  // The latency sweep extends the bench request the page settled on: the
  // same entries (seeded as the bench echoed them) and contract, timed at
  // every sweep length. Null while the bench is not in sync with the picks.
  const latencyBody = resp && !busy && bodyJson === lastPosted.current
    ? JSON.stringify({
        ...JSON.parse(bodyJson),
        kv_cache_lens: [],
        // L only marks a column here; the lengths are the sweep's.
        seq_len: undefined,
        lens: SWEEP_SEQ_LENS,
        entries: entries.map((e) => ({
          name: e.name, model_id: e.model_id, call: e.call, dims: e.dims, seeded: e.seeded,
          label: e.label ?? null,
        })),
      })
    : null;

  const run = useCallback(async (sentJson: string) => {
    if (entries.length === 0) return;
    const seq = ++runSeq.current;
    lastPosted.current = sentJson;
    setBusy(true);
    setStatus("Benchmarking…");
    setStatusKind("idle");
    // NOTE: gapNote (bench-cap note) is NOT reset here — it must survive the
    // rerun it triggered. It is cleared when the selection's STRUCTURE changes
    // (toggle/remove/rebind/addBinding/role switch) or entries empty.
    try {
      const data = await fetchJson<BenchResponse>("/api/playground/module-bench", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...JSON.parse(sentJson),
          entries: entries.map((e) => ({
            name: e.name, model_id: e.model_id, call: e.call, dims: e.dims, seeded: e.seeded,
            label: e.label ?? null,
          })),
        }),
      });
      if (seq !== runSeq.current) return; // superseded — a newer run owns the screen
      if (!data.success) {
        setStatus(`Failed: ${data.message || "unknown error"}`);
        setStatusKind("err");
        return;
      }
      setResp(data);
      // Adopt the echoed dims (what actually took effect, binding seeds folded
      // in) into the entries — a bound entry's first response turns its inline
      // seed into visible, editable dims, records the customization baseline
      // (`seedDims`, which the "edited" marker compares against), and marks it
      // seeded so later clears stick. The adopt writes back exactly what the
      // backend already used, so it syncs `lastPosted` instead of triggering
      // a redundant repost.
      const applied = data.contract.per_component_dims || {};
      // The binding's own dims for entries seeded this run: the baseline
      // their edits (a link's, or none) are measured against.
      const seeds = data.contract.seed_dims || {};
      // The calls each entry can name, as this response lists them — kept on
      // the entry so the call picker shows them; not part of the request.
      const listed = new Map<string, CallInfo[]>();
      for (const r of data.results) if (r.calls) listed.set(entryOf(r), r.calls);
      for (const er of data.errors) if (er.calls && er.entry) listed.set(er.entry, er.calls);
      const prevE = entriesRef.current;
      let adopted = false;
      const nextE = prevE.map((e0) => {
        const calls = listed.get(e0.key);
        const e = calls && JSON.stringify(calls) !== JSON.stringify(e0.calls ?? null)
          ? { ...e0, calls } : e0;
        if (e !== e0) adopted = true;
        const a = applied[e.key];
        if (!a) return e;
        const same = Object.keys(a).length === Object.keys(e.dims).length
          && Object.entries(a).every(([k, v]) => e.dims[k] === v);
        if (same && (e.seeded || !e.model_id)) return e;
        adopted = true;
        return { ...e, dims: { ...a }, seedDims: { ...(seeds[e.key] ?? a) },
                 seeded: e.seeded || !!e.model_id };
      });
      if (adopted) {
        entriesRef.current = nextE;
        lastPosted.current = canonBody(nextE); // backend already used these — no repost
        setEntries(nextE);
      }
      const n = data.results.length;
      const g = data.errors.length;
      setStatus(`Done. ${n} component${n === 1 ? "" : "s"} benched${g ? `, ${g} gap${g === 1 ? "" : "s"}` : ""}.`);
      setStatusKind("ok");
    } catch (e) {
      if (seq !== runSeq.current) return;
      setStatus(`Failed: ${e instanceof Error ? e.message : String(e)}`);
      setStatusKind("err");
    } finally {
      if (seq === runSeq.current) setBusy(false);
    }
  }, [entries, batch, seqLen, precPreset, gpuSku, canonBody]);

  useEffect(() => {
    if (entries.length === 0) {
      runSeq.current++; // invalidate any in-flight run, then clear the stale view
      setSelectionDirty(false);
      setResp(null);
      setBusy(false);
      setStatus("");
      setStatusKind("idle");
      setGapNote("");
      return;
    }
    if (selectionDirty) {
      runSeq.current++; // an old run no longer owns the staged selection
      setBusy(false);
      setStatusKind("idle");
      setStatus(pendingPicks.size > 0
        ? "Resolving bindings…"
        : `${entries.length} column${entries.length === 1 ? "" : "s"} staged — Benchmark to run.`);
      return;
    }
    if (bodyJson === lastPosted.current) return; // adopt already synced this state
    const t = setTimeout(() => { void run(bodyJson); }, 450);
    return () => {
      clearTimeout(t);
      runSeq.current++; // rescheduled: any in-flight response is now stale
    };
  }, [bodyJson, run, entries.length, selectionDirty, pendingPicks.size]);

  // The URL is the comparison on screen, so it can be shared: every column
  // with the dims it sets itself (a bound column's edits to its binding, the
  // whole parameterization of a labeled or unbound one), and each contract
  // field that is not the page default.
  useEffect(() => {
    const picks: PlaygroundPick[] = entries.map((e) => {
      const base = e.label || !e.model_id ? undefined : e.seedDims;
      return {
        name: e.name, model_id: e.model_id, call: e.call, label: e.label ?? null,
        dims: base ? Object.fromEntries(Object.entries(e.dims).filter(([k, v]) => base[k] !== v)) : e.dims,
      };
    });
    const search = playgroundSearch({
      picks,
      role: unlessDefault("role", role),
      batch: unlessDefault("batch", batch),
      seq: unlessDefault("seq", seqLen),
      prec: unlessDefault("prec", precPreset),
      gpu: unlessDefault("gpu", gpuSku),
      phase: unlessDefault("phase", chartPhase),
    });
    if (search !== window.location.search) {
      history.replaceState(null, "", `${window.location.pathname}${search}${window.location.hash}`);
    }
  }, [entries, role, batch, seqLen, precPreset, gpuSku, chartPhase]);

  // Edit a single dim of one entry; auto-run recomputes. Editing a bound
  // entry's dims makes it a CUSTOMIZATION of that binding (the "edited" marker
  // compares against `seedDims`); a seeded entry's backend never re-seeds, so
  // a cleared dim stays cleared.
  const editEntryDim = useCallback((key: string, dim: string, value: number | null) => {
    setEntries((prev) => prev.map((e) => {
      if (e.key !== key) return e;
      const dims = { ...e.dims };
      if (value == null) delete dims[dim]; else dims[dim] = value;
      return { ...e, dims };
    }));
  }, []);

  useEffect(() => {
    const has = !!(resp && resp.results.length > 0);
    // Applying a staged selection closes the catalog explicitly. A late response
    // must not yank the catalog shut while the user is staging more picks.
    if (!has) setCatalogOpen(true);
    hadResults.current = has;
  }, [resp]);

  useLayoutEffect(() => {
    const el = picksRef.current;
    if (!el) {
      document.documentElement.style.removeProperty("--pg-picks-h");
      return;
    }
    const apply = () => {
      document.documentElement.style.setProperty(
        "--pg-picks-h", `${el.getBoundingClientRect().height}px`);
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => {
      ro.disconnect();
      document.documentElement.style.removeProperty("--pg-picks-h");
    };
  }, [entries.length, catalogOpen, mode]);

  const statusClass = statusKind === "ok" ? "text-sm text-green-600"
    : statusKind === "err" ? "text-sm text-red-500" : "text-sm text-gray-500";
  // The staged-selection state and the Benchmark button. They live at the
  // catalog's bottom; with the catalog closed (results on screen) a change
  // staged from the picks bar or the details still needs them, so they show
  // in their own row under the picks.
  const runRow = (
    <div className="flex items-center gap-3">
      {busy && (
        <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-amber-500 border-t-transparent animate-spin" />
      )}
      <span className={statusClass}>{status}</span>
      {gapNote && <span className="text-xs text-amber-600">{gapNote}</span>}
      {selectionDirty && entries.length > 0 && (
        <button
          id="pg-run-picks"
          type="button"
          onClick={applySelection}
          disabled={pendingPicks.size > 0}
          title={pendingPicks.size > 0
            ? "Still resolving the selected components' bindings"
            : "Benchmark every staged column in one request"}
          className={`text-xs px-2.5 py-1 rounded-full font-medium transition ${
            pendingPicks.size > 0
              ? "border border-gray-200 bg-gray-50 text-gray-400 cursor-default"
              : "border border-amber-500 bg-amber-500 text-white hover:bg-amber-600"
          }`}
        >
          {pendingPicks.size > 0
            ? "Resolving…"
            : `Benchmark ${entries.length} ${entries.length === 1 ? "column" : "columns"}`}
        </button>
      )}
      {!busy && entries.length > 0 && (
        <span className="text-xs text-gray-400">
          {selectionDirty
            ? "selection changes stage — Benchmark submits them together"
            : "contract and dim changes recompute automatically"}
        </span>
      )}
    </div>
  );

  return (
    <main className="w-full page-shell py-4 sm:py-8 flex-1 min-w-0">
      <div className="flex gap-2 mb-4 border-b border-gray-200">
        {(["compare", "editor"] as const).map((m) => (
          <button key={m} type="button" onClick={() => setMode(m)}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition ${mode === m
              ? "border-amber-500 text-amber-700" : "border-transparent text-gray-500 hover:text-gray-800"}`}>
            {m === "compare" ? "Module comparison" : "Model editor"}
          </button>
        ))}
      </div>

      <div className="mb-4">
        <PrecisionPresets idPrefix="pg" value={precPreset} notice={precNotice}
          onChange={(id) => { setPrecPreset(id); setPrecNotice(null); }} />
      </div>

      {mode === "editor" && <ModelEditor models={models} components={components}
        gpuSku={gpuSku} gpus={gpus} onGpuSku={setGpuSku}
        precision={precPreset} precisionFromLink={precisionFromLink}
        onDraftPrecision={setPrecPreset}
        onLegacyDraftPrecision={() => setPrecNotice(PREDATES_PROFILES_NOTICE)} />}

      {mode === "compare" && <>
      <h1 className="text-2xl font-bold text-gray-900 mb-1">Playground — Module Comparison</h1>
      <p className="text-sm text-gray-500 mb-4 max-w-3xl">
        Compare attention / FFN variants as they are parameterized in the models that ship them.
        Each column is one component × binding. Not an architecture ranking.
      </p>

      {entries.length > 0 && (
        <PicksBar
          barRef={picksRef}
          items={entries.map((e) => {
            const mids = components.find((c) => c.name === e.name)?.model_ids ?? [];
            const models = mids.map((id) => ({ id, label: shortModel(id) }));
            if (e.model_id && !models.some((m) => m.id === e.model_id)) {
              models.unshift({ id: e.model_id, label: shortModel(e.model_id) });
            }
            const mine = e.key.includes("~my:");
            return {
              key: e.key,
              name: mine ? entryLabel(e.key) : e.name,
              modelId: e.model_id,
              modelLabel: mine ? null : (e.model_id ? shortModel(e.model_id) : null),
              models,
              canRebind: !mine && models.length > 0,
              call: e.call,
              calls: (e.calls ?? []).map((c) => ({
                call: c.call,
                label: c.component === e.name ? callText(c) : `${c.component} · ${callText(c)}`,
              })),
            };
          })}
          onRebind={rebindEntry}
          onPickCall={pickCall}
          onRemove={removeEntry}
          onAdd={() => setCatalogOpen((v) => !v)}
        />
      )}

      {!(catalogOpen || entries.length === 0) && selectionDirty && (
        <div id="pg-staged" className="card px-4 py-3 mb-6">{runRow}</div>
      )}

      {(catalogOpen || entries.length === 0) && (
      <section id="pg-catalog" className="card p-4 sm:p-6 mb-6">
        <div>
          <div className="flex flex-wrap items-center gap-3 mb-2">
            <label className="block text-sm font-medium text-gray-700">Role</label>
            <select id="pg-role" value={role}
              onChange={(e) => {
                setRole(e.target.value);
                pickEpoch.current++;
                setPendingPicks(new Set());
                setSelectionDirty(false);
                setEntries([]);
                setOpenFamilies(new Set());
                setPickerFilter("");
              }}
              className="text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white">
              {roles.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
            <span className="text-xs text-gray-400">{componentsLoaded ? `${roleComponents.length} components` : "…"}</span>
            <input value={pickerFilter} onChange={(e) => setPickerFilter(e.target.value)}
              placeholder="Filter components…"
              className="ml-auto w-full sm:w-48 text-xs border border-gray-300 rounded-lg px-2.5 py-1.5 bg-white" />
          </div>
          <div id="pg-groups">
            {showGroupHeaders && (
              <div className="flex flex-wrap gap-1.5 mb-2">
                {visibleGroups.map(({ family, selected }) => {
                  const open = filterLower !== "" || openFamilies.has(family);
                  return (
                    <button key={family} type="button" onClick={() => toggleFamily(family)}
                      className={`uppercase tracking-wide text-11 font-semibold px-2.5 py-1 rounded-full border transition ${
                        open ? "border-gray-400 bg-gray-100 text-gray-800" : "border-gray-300 bg-white text-gray-500 hover:text-gray-700"}`}>
                      {family}{selected > 0 ? ` · ${selected}` : ""}
                    </button>
                  );
                })}
              </div>
            )}
            {visibleGroups.map(({ family, shown }) => {
              const open = !showGroupHeaders || filterLower !== "" || openFamilies.has(family);
              if (!open) return null;
              return (
                <div key={family} className="flex flex-wrap gap-2 mb-2">
                  {shown.map((name) => (
                    <button key={name} type="button" onClick={() => toggle(name)}
                      className={`text-xs px-2.5 py-1 rounded-full border transition ${pendingPicks.has(name)
                        ? "border-amber-300 bg-amber-50/50 text-amber-400 animate-pulse"
                        : entries.some((e) => e.name === name)
                        ? "border-amber-500 bg-amber-50 text-amber-800"
                        : "border-gray-300 bg-white text-gray-600 hover:border-gray-400"}`}>
                      {name}{pendingPicks.has(name) ? " …" : ""}
                    </button>
                  ))}
                </div>
              );
            })}
            {!componentsLoaded ? (
              <div className="text-xs text-gray-400">Loading components…</div>
            ) : visibleGroups.length === 0 && (
              <div className="text-xs text-gray-400">No components match “{pickerFilter}”.</div>
            )}
          </div>
        </div>

        <div className="mt-4">{runRow}</div>
      </section>
      )}

      {uniqueNames.length > 0 && (
        <section id="pg-bound-models" className="card p-4 sm:p-6 mb-6">
          <BindingsTable
            blocks={uniqueNames.map((name) => ({
              name,
              bindings: bindingsByName[name] ?? null,
              myVariants: signedIn ? (variantsByName[name] ?? []) : null,
              addedSigs: new Map(entries.filter((e) => e.name === name)
                .map((e) => [dimsSig(e.seedDims ?? e.dims),
                             e.model_id ? shortModel(e.model_id)
                             : e.key.includes("~my:") ? entryLabel(e.key) : "custom"] as const)),
            }))}
            addedKeys={new Set(entries.map((e) => e.key))}
            capReached={entries.length >= 8}
            onAdd={addBinding} onRemove={removeEntry} onAddVariant={addVariant} />
          <div className="mt-4">
            <button type="button"
              disabled={entries.length >= 8}
              title={entries.length >= 8 ? "The comparison is full (8 entries max)."
                : "Seed from a base structure, tweak dims, compare as its own column."}
              onClick={() => setNewModuleFor(uniqueNames[0])}
              className="text-xs px-3 py-1.5 rounded-lg border border-amber-300 text-amber-700 font-medium hover:bg-amber-50 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent">
              + New module…
            </button>
          </div>
        </section>
      )}

      {resp && (
        <div className={busy ? "opacity-60 transition" : "transition"}>
          {resp.errors.length > 0 && (
            <GapsPanel errors={resp.errors}
              modelIdsOf={(name) => components.find((c) => c.name === name)?.model_ids ?? []}
              onRebind={rebindEntry} onPickCall={pickCall} />
          )}

          {resp.results.length > 0 && (
            <>
              <div className="text-xs text-gray-500 mb-3">
                {resp.results[0]?.dims_used?.d_model ? <>d_model=<span className="font-mono text-gray-700">{formatDim(resp.results[0].dims_used.d_model)}</span> — </> : null}
                each column is one component × binding; internal dims are editable below.
              </div>
              <JumpBar present={new Set(["pg-structure", "pg-summary", "pg-kv",
                "pg-latency", "pg-details"])} />
              <Section
                id="pg-structure"
                scope="FLOPs: one prefill forward"
                title={resp.results.length === 1 ? (
                  <>
                    Architecture · <span className="font-mono">{resp.results[0].component}</span>
                    {" "}<span className="text-sm font-normal text-gray-400">
                      {resp.results[0].model_id
                        ? `as ${shortModel(resp.results[0].model_id)} parameterizes it`
                        : "unbound"}
                    </span>
                  </>
                ) : (
                  <>
                    Compute-graph diff{" "}
                    <span className="text-sm font-normal text-gray-400">
                      {resp.results.length} entries · nodes paired from declarations, not names
                    </span>
                  </>
                )}
              >
                <div id="pg-graph-diff" className="min-w-0">
                  <GraphShell
                    rows={diffRows}
                    diff={resp.graph_diff}
                    colors={resp.results.map((r) => PALETTE[respColorIdx(resp).get(entryOf(r)) ?? 0])}
                  />
                </div>
              </Section>

              <Section id="pg-summary" scope="both phases side by side" title="Summary" lead={
                <ServingFields batch={batch} seqLen={seqLen}
                  gpuSku={gpuSku} gpus={gpus} onGpuSku={setGpuSku}
                  onBatch={setBatch} onSeqLen={setSeqLen} />
              }>
                <SummaryTable
                  rows={resp.results.map((r) => {
                    const edited = entries.some((e) => e.key === entryOf(r) && isCustomized(e));
                    const form = decodeOf.get(r.component);
                    return {
                      ...r,
                      key: entryOf(r),
                      label: rowLabel(r),
                      color: PALETTE[respColorIdx(resp).get(entryOf(r)) ?? 0],
                      marks: [edited ? "edited" : null,
                              form ? `decode form of ${form}` : null]
                        .filter(Boolean).join(" · ") || undefined,
                      aside: <ProvenanceChip ids={r.model_ids} />,
                    };
                  })}
                  cols={resp.results.some((r) => r.timings?.prefill)
                    ? SUMMARY_COLS
                    : RESOURCE_SUMMARY_COLS}
                  footnotes={summaryFootnotes(resp.results[0], resp.results)}
                />
              </Section>

              <KvSection resp={{ ...resp, results: resp.results }} colorOf={(r) => PALETTE[respColorIdx(resp).get(entryOf(r)) ?? 0]}
                decodeOf={decodeOf} descOf={descOf} serving={
                  <ServingFields batch={batch} seqLen={seqLen}
                    gpuSku={gpuSku} gpus={gpus} onGpuSku={setGpuSku}
                    onBatch={setBatch} onSeqLen={setSeqLen} />
                } />
              <LatencySection body={latencyBody} atLen={seqLen}
                styleOf={(key) => {
                  const r = resp.results.find((x) => entryOf(x) === key);
                  return r && { label: rowLabel(r), color: PALETTE[respColorIdx(resp).get(key) ?? 0] };
                }}
                phase={chartPhase} onPhase={setChartPhase} serving={
                  <ServingFields batch={batch} seqLen={seqLen}
                    gpuSku={gpuSku} gpus={gpus} onGpuSku={setGpuSku}
                    onBatch={setBatch} onSeqLen={setSeqLen} />
                } />
              <DetailsSection results={resp.results} baseModelId={resp.base_model_id}
                entryDims={Object.fromEntries(entries.map((e) => [e.key, e.dims]))}
                seedDims={Object.fromEntries(entries.map((e) => [e.key, e.seedDims]))}
                editedKeys={new Set(entries.filter(isCustomized).map((e) => e.key))}
                canUseBase={canUseBase} decodeOf={decodeOf} descOf={descOf}
                onSaveVariant={saveVariant} signedIn={signedIn}
                savingReady={!busy && bodyJson === lastPosted.current}
                onRebind={rebindEntry} onEditDim={editEntryDim}
                colorOf={(r) => PALETTE[respColorIdx(resp).get(entryOf(r)) ?? 0]}
                phase={chartPhase} onPhase={setChartPhase}
                gpuSku={gpuSku} gpus={gpus} onGpuSku={setGpuSku} />
            </>
          )}
        </div>
      )}
      {newModuleFor && (
        <NewModuleModal component={newModuleFor} components={uniqueNames}
          entries={entries}
          bindingsByName={Object.fromEntries(uniqueNames.map((n) => [n, bindingsByName[n]?.bindings ?? []]))}
          signedIn={signedIn}
          onClose={() => setNewModuleFor(null)}
          onCreate={(c, n, d, s) => void createModule(c, n, d, s)} />
      )}
      </>}
    </main>
  );
}

// Key-value editor for dim overrides — replaces the raw JSON box so users never
// need to know the exact internal dim names or JSON syntax up front. `value`
// (numbers only) is the single source of truth; the add row validates before
// committing (empty or duplicate names, and non-numeric values, are rejected).
// Why a typed dim value is not one, or null. A dim counts something: a whole
// number, 0 or more (0 is how a model says it has none of a thing). Whether
// the NAME is a dim is the server's answer (a spec may register its own).
function dimValueProblem(raw: string): string | null {
  if (raw.trim() === "") return "enter a value";
  const n = Number(raw);
  if (!Number.isFinite(n)) return "not a number";
  if (!Number.isInteger(n)) return "must be a whole number";
  if (n < 0) return "cannot be negative";
  return null;
}

function DimOverridesEditor({ value, onChange }: {
  value: Record<string, number>;
  onChange: (next: Record<string, number>) => void;
}) {
  const [newKey, setNewKey] = useState("");
  const [newVal, setNewVal] = useState("");
  // a row's typed text while it is not a valid value (not sent)
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const entries = Object.entries(value).sort(([a], [b]) => a.localeCompare(b));
  const key = newKey.trim();
  const num = Number(newVal);
  const keyProblem = key === "" ? null
    : key in value ? `${key} is already set above`
    : /^[A-Za-z_][A-Za-z0-9_]*$/.test(key) ? null : "a dim name is letters, digits and _";
  const valProblem = newVal.trim() === "" ? null : dimValueProblem(newVal);
  const canAdd = key !== "" && !keyProblem && newVal.trim() !== "" && !valProblem;
  const add = () => {
    if (!canAdd) return;
    onChange({ ...value, [key]: num });
    setNewKey("");
    setNewVal("");
  };
  return (
    <div className="space-y-1.5">
      {entries.length === 0 && <div className="text-xs text-gray-400">No overrides.</div>}
      {entries.map(([k, v]) => (
        <div key={k} className="flex items-center gap-2">
          <span className="text-xs font-mono text-gray-700 w-40 shrink-0 truncate" title={k}>{k}</span>
          <input type="number" min={0} step={1} value={drafts[k] ?? v}
            onChange={(e) => {
              const raw = e.target.value;
              if (dimValueProblem(raw)) { setDrafts((d) => ({ ...d, [k]: raw })); return; }
              setDrafts((d) => { const n = { ...d }; delete n[k]; return n; });
              onChange({ ...value, [k]: Number(raw) });
            }}
            className={`w-32 text-xs border rounded-lg px-2 py-1 bg-white ${k in drafts ? "border-red-400" : "border-gray-300"}`} />
          <button type="button" title="Remove"
            onClick={() => {
              const next = { ...value }; delete next[k]; onChange(next);
              setDrafts((d) => { const n = { ...d }; delete n[k]; return n; });
            }}
            className="text-xs text-gray-400 hover:text-red-500 px-1">×</button>
          {k in drafts && <span className="pg-dim-invalid text-xs text-red-600">{dimValueProblem(drafts[k])} — not applied</span>}
        </div>
      ))}
      <div className="flex flex-wrap items-center gap-2">
        <input value={newKey} onChange={(e) => setNewKey(e.target.value)} placeholder="dim name"
          className="w-40 text-xs border border-gray-300 rounded-lg px-2 py-1 bg-white font-mono" />
        <input type="number" min={0} step={1} value={newVal} onChange={(e) => setNewVal(e.target.value)} placeholder="value"
          className="w-32 text-xs border border-gray-300 rounded-lg px-2 py-1 bg-white" />
        <button type="button" onClick={add} disabled={!canAdd}
          className="text-xs px-2.5 py-1 rounded-full border border-gray-300 bg-white text-gray-600 transition hover:border-gray-400 disabled:opacity-40 disabled:hover:border-gray-300">
          + Add dim
        </button>
        {(keyProblem || valProblem) && (
          <span className="pg-dim-invalid text-xs text-red-600">{keyProblem ?? `value ${valProblem}`}</span>
        )}
      </div>
    </div>
  );
}

// "New module" modal: seed a fresh compare column from a base parameterization
// (a current column or a shipped binding of this component), tweak its dims,
// name it, and drop it into the comparison — optionally persisted to
// "My modules" (signed-in only). All base dims here are the component's OWN
// editable knobs (contract dims never reach these lists), so every row is
// editable without second-guessing the shared contract.
function NewModuleModal({ component, components, entries, bindingsByName, signedIn, onClose, onCreate }: {
  component: string;
  components: string[];
  entries: Entry[];
  bindingsByName: Record<string, BindingRow[]>;
  signedIn: boolean;
  onClose: () => void;
  onCreate: (component: string, name: string, dims: Record<string, number>, save: boolean) => void;
}) {
  const [comp, setComp] = useState(component);
  const bindings = (bindingsByName[comp] ?? []).filter((b) => !b.error);
  const bases = [
    ...entries.filter((e) => e.name === comp).map((e) => ({ id: `entry:${e.key}`, label: `column · ${entryLabel(e.key)}`, dims: e.dims })),
    ...bindings.map((b) => ({ id: `binding:${b.model_id}#${b.call}`, label: `binding · ${shortModel(b.model_id)} · ${b.label}`, dims: b.dims })),
  ];
  const [baseId, setBaseId] = useState(bases[0]?.id ?? "");
  const [dims, setDims] = useState<Record<string, number>>({ ...(bases[0]?.dims ?? {}) });
  const [name, setName] = useState("");
  const [save, setSave] = useState(true);
  const [err, setErr] = useState("");
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const pickComponent = (next: string) => {
    setComp(next);
    const nextBindings = (bindingsByName[next] ?? []).filter((b) => !b.error);
    const nextBases = [
      ...entries.filter((e) => e.name === next).map((e) => ({ id: `entry:${e.key}`, dims: e.dims })),
      ...nextBindings.map((b) => ({ id: `binding:${b.model_id}#${b.call}`, dims: b.dims })),
    ];
    setBaseId(nextBases[0]?.id ?? "");
    setDims({ ...(nextBases[0]?.dims ?? {}) });
  };
  const pickBase = (id: string) => {
    setBaseId(id);
    setDims({ ...(bases.find((b) => b.id === id)?.dims ?? {}) });
  };
  const trimmed = name.trim();
  // The name rides the entry's label (`<component>~my:<name>`), which the
  // backend's bench contract forbids '@' in — and embedding the "~my:"
  // separator itself would corrupt entry identity/display. Reject both here,
  // at the only input surface that can produce such a label.
  const badName = trimmed.includes("@") || trimmed.includes("~my:");
  const create = () => {
    if (!trimmed) { setErr("Name the module."); return; }
    if (badName) { setErr('The name can\'t contain "@" or "~my:" — both are structural in entry keys.'); return; }
    if (entries.some((e) => e.key === variantKey(comp, trimmed))) {
      setErr(`"${trimmed}" is already in the comparison.`); return;
    }
    const vals = Object.values(dims);
    if (vals.length === 0) { setErr("Add at least one dim."); return; }
    if (vals.some((v) => !Number.isInteger(v) || v <= 0)) {
      setErr("Dims must be positive integers."); return;
    }
    onCreate(comp, trimmed, dims, save && signedIn);
    onClose();
  };
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/30" />
      <div className="relative bg-white rounded-xl shadow-xl w-full max-w-lg mx-4 p-5" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-sm font-semibold text-gray-900 mb-3">
          New comparison module{components.length === 1 ? <> — <span className="font-mono">{comp}</span></> : null}
        </h2>
        {components.length > 1 && (
          <>
            <label className="block text-xs font-medium text-gray-700 mb-1">Component</label>
            <select value={comp} onChange={(e) => pickComponent(e.target.value)}
              className="w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white mb-3 font-mono">
              {components.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </>
        )}
        <label className="block text-xs font-medium text-gray-700 mb-1">Base structure</label>
        <select value={baseId} onChange={(e) => pickBase(e.target.value)}
          className="w-full text-sm border border-gray-300 rounded-lg px-2 py-1.5 bg-white mb-3">
          {bases.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
        </select>
        <label className="block text-xs font-medium text-gray-700 mb-1">Hyperparameters (dims)</label>
        <div className="max-h-56 overflow-y-auto mb-3">
          <DimOverridesEditor value={dims} onChange={setDims} />
        </div>
        <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. mla-fat-kv"
          className="w-full text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white mb-1" />
        {badName && <div className="text-xs text-red-600 mb-2">The name can't contain "@" or "~my:".</div>}
        <div className="mb-2" />
        {signedIn && (
          <label className="flex items-center gap-2 text-xs text-gray-600 mb-3 select-none">
            <input type="checkbox" checked={save} onChange={(e) => setSave(e.target.checked)}
              className="rounded border-gray-300" />
            Save to My modules (reusable under Bound models → mine)
          </label>
        )}
        {err && <div className="text-xs text-red-600 mb-2">{err}</div>}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose}
            className="text-xs px-3 py-1.5 rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50">
            Cancel
          </button>
          <button type="button" onClick={create} disabled={!trimmed || badName}
            className="text-xs px-3 py-1.5 rounded-lg bg-amber-500 text-white font-medium hover:bg-amber-600 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed">
            Create & compare
          </button>
        </div>
      </div>
    </div>
  );
}

// Gaps panel: an entry that can't resolve under the current base. Facts only —
// no guessed values are ever filled in; the repair is BINDING the column to a
// model that actually parameterizes the component (select right here), or
// removing the entry.
function GapsPanel({ errors, modelIdsOf, onRebind, onPickCall }: {
  errors: BenchResponse["errors"];
  modelIdsOf: (name: string) => string[];
  onRebind: (key: string, modelId: string) => void;
  onPickCall: (key: string, call: string) => void;
}) {
  return (
    <section id="pg-gaps" className="card p-4 mb-6 border-l-4 border-amber-400">
      <h2 className="text-sm font-semibold text-gray-900 mb-1">Gaps ({errors.length})</h2>
      <p className="text-xs text-gray-500 mb-2">
        These entries can't resolve under the current base — nothing is guessed or
        auto-filled. Bind the column to a model that parameterizes the component,
        or remove it.
      </p>
      <ul className="text-xs text-gray-600 space-y-2">
        {errors.map((e) => {
          const key = e.entry ?? e.component;
          const mids = modelIdsOf(e.component);
          return (
            <li key={key}>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-gray-800">{key}</span>
                {": "}<span className="text-gray-500">{e.error}</span>
                {(e.calls?.length ?? 0) > 0 && (
                  <select value="" data-call-pick={key}
                    title="The model calls this component in more than one place (or the base does not call it): name the call to evaluate it at"
                    onChange={(ev) => { if (ev.target.value) onPickCall(key, ev.target.value); }}
                    className="text-11 border border-amber-300 rounded px-1.5 py-0.5 bg-white text-amber-700 max-w-[28rem]">
                    <option value="">evaluate at which call…</option>
                    {(e.calls || []).map((c) => (
                      <option key={c.call} value={c.call}>{callText(c)}</option>
                    ))}
                  </select>
                )}
                {mids.length > 0 && (
                  <select value="" title="Bind this column to a shipped model"
                    onChange={(ev) => { if (ev.target.value) onRebind(key, ev.target.value); }}
                    className="text-11 border border-amber-300 rounded px-1.5 py-0.5 bg-white text-amber-700">
                    <option value="">bind to…</option>
                    {mids.filter((mid) => mid !== e.model_id).map((mid) => (
                      <option key={mid} value={mid}>{shortModel(mid)}</option>
                    ))}
                  </select>
                )}
              </div>
              {(e.missing_dims?.length ?? 0) > 0 && (
                <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
                  {(e.missing_dims || []).map((d) => (
                    <DimSymbol key={d} symbol={d} meaning={dimMeaning(d)}
                      className="font-mono text-gray-500" />
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// "deepseek-ai/DeepSeek-V3.2" → "DeepSeek-V3.2"
const shortModel = (id: string) => (id.includes("/") ? id.split("/").pop()! : id);

// Column/axis label for one bench row: the entry identity — `name @ model`
// for bindings, `name · <variant>` for saved variants, the bare name otherwise
// — and the call it was evaluated at when the model makes more than one.
const rowLabel = (r: BenchRow) => {
  const base = r.model_id ? `${r.component} @ ${shortModel(r.model_id)}` : entryLabel(r.entry ?? r.component);
  return r.call && (r.calls?.length ?? 0) > 1 ? `${base} · ${r.call.label}` : base;
};

// An entry whose dims diverge from its binding's seed — a customization of
// that parameterization (the column gets the "edited" marker). An unbound
// entry has no seed baseline: any dims at all are the user's customization.
const isCustomized = (e: Entry) => e.seedDims
  ? dimsSig(e.dims) !== dimsSig(e.seedDims)
  : Object.keys(e.dims).length > 0;

// Which shipped models bind this component — so a swapped-in library module is
// not an anonymous abstraction. Empty → a library-only variant, said plainly.
function ProvenanceChip({ ids }: { ids?: string[] }) {
  if (!ids || ids.length === 0)
    return <span className="text-11 text-gray-400 italic">library-only</span>;
  const shown = ids.slice(0, 2).map(shortModel);
  const extra = ids.length - shown.length;
  return (
    <span className="text-11 text-gray-500" title={ids.join(", ")}>
      from {shown.join(", ")}{extra > 0 ? ` +${extra}` : ""}
    </span>
  );
}

type BenchSummary = BenchRow & SummaryRow;
const timingCoverage = (r: BenchRow, phase: PhaseKey) => {
  const t = r.timings?.[phase];
  return t ? `${t.measured_rows}/${t.total_rows} measured` : null;
};
const SUMMARY_COLS: MetricCol<BenchSummary>[] = [
  { key: "params", label: "Params", kind: "count", group: "stored",
    get: (r) => r.params,
    formula: (r) => r.formulas?.params },
  { key: "active_params", label: "Active params", kind: "count", group: "stored",
    get: (r) => r.active_params,
    formula: (r) => r.formulas?.active_params },
  { key: "weight_bytes", label: "Weight storage", kind: "bytes", group: "stored",
    get: (r) => r.weight_bytes,
    formula: (r) => r.formulas?.weight_bytes,
    note: (r) => {
      const ws = r.weight_storage;
      if (!ws?.precision) return null;
      // Bound rows: whose checkpoint it is reads off the row label already.
      const prec = ws.precision.replace(/^checkpoint precision: /, "").replace(/, some /g, ", ");
      return ws.source.bound ? prec : `${prec} · ${ws.source.note}`;
    } },
  { key: "kv_storage", label: "KV / request @L", kind: "bytes", group: "stored",
    get: (r) => r.kv.storage_bytes_at_L,
    formula: (r) => r.formulas?.kv_storage },
  { key: "prefill_flops", label: "FLOPs", kind: "flops", group: "prefill",
    get: (r) => r.phases.prefill.flops,
    formula: (r) => r.phases.prefill.formulas?.flops },
  { key: "prefill_t_sol", label: "t SOL", kind: "time", group: "prefill",
    get: (r) => r.timings?.prefill.t_sol_s,
    note: (r) => (r.timings_unavailable ? "not timed — see below" : null),
    formula: (r) => r.phases.prefill.formulas?.t_sol },
  { key: "prefill_weight_read", label: "Weight read", kind: "bytes", group: "prefill",
    get: (r) => r.phases.prefill.weight_read_bytes,
    formula: (r) => r.phases.prefill.formulas?.weight_read },
  { key: "prefill_kv_write", label: "KV write / step", kind: "bytes", group: "prefill",
    get: (r) => r.phases.prefill.kv_step_bytes,
    formula: (r) => r.phases.prefill.formulas?.kv_step },
  { key: "prefill_act_fused", label: "Act (fused)", kind: "bytes", group: "prefill",
    get: (r) => r.phases.prefill.act_bytes_fused,
    formula: (r) => r.phases.prefill.formulas?.act_bytes },
  { key: "prefill_t_measured", label: "t measured", kind: "time", group: "prefill",
    get: (r) => r.timings?.prefill.t_measured_s,
    note: (r) => timingCoverage(r, "prefill") },
  { key: "decode_flops", label: "FLOPs", kind: "flops", group: "decode",
    get: (r) => r.phases.decode.flops,
    formula: (r) => r.phases.decode.formulas?.flops },
  { key: "decode_t_sol", label: "t SOL", kind: "time", group: "decode",
    get: (r) => r.timings?.decode.t_sol_s,
    note: (r) => (r.timings_unavailable ? "not timed — see below" : null),
    formula: (r) => r.phases.decode.formulas?.t_sol },
  { key: "decode_weight_read", label: "Weight read", kind: "bytes", group: "decode",
    get: (r) => r.phases.decode.weight_read_bytes,
    formula: (r) => r.phases.decode.formulas?.weight_read },
  { key: "decode_kv_read", label: "KV read / step", kind: "bytes", group: "decode",
    get: (r) => r.phases.decode.kv_read_bytes,
    formula: (r) => r.phases.decode.formulas?.kv_step,
    mark: (r) => (r.phases.decode.memory_dominant === "kv" ? "dominant" : null) },
  { key: "decode_act_fused", label: "Act (fused)", kind: "bytes", group: "decode",
    get: (r) => r.phases.decode.act_bytes_fused,
    formula: (r) => r.phases.decode.formulas?.act_bytes },
  { key: "decode_t_measured", label: "t measured", kind: "time", group: "decode",
    get: (r) => r.timings?.decode.t_measured_s,
    note: (r) => timingCoverage(r, "decode") },
];
const RESOURCE_SUMMARY_COLS = SUMMARY_COLS.filter(
  (c) => !c.key.endsWith("_t_sol") && !c.key.endsWith("_t_measured"),
);

// Every unit string of a phase opens with the same contract ("per layer ·
// one forward @ batch 1 × 4,096 tokens"). Say it once, then what each column
// adds to it, named by the column. The strings are the backend's; this only
// removes the repetition.
function phaseFootnote(parts: [string, string][]): string {
  const texts = parts.map(([, t]) => t).filter(Boolean);
  if (!texts.length) return "";
  const first = texts[0];
  const cuts = [first.length];
  for (let i = first.lastIndexOf(" · "); i > 0; i = first.lastIndexOf(" · ", i - 1)) cuts.push(i);
  const fits = (head: string) => texts.every((t) => t.startsWith(head)
    && (t.length === head.length || t.startsWith(" · ", head.length) || t.startsWith(",", head.length)));
  const n = cuts.find((c) => fits(first.slice(0, c)));
  if (n == null) return [...new Set(texts)].join(" · ");
  const head = first.slice(0, n);
  const tails = parts.flatMap(([label, t]) => {
    const tail = t.slice(n).replace(/^( · |,\s*)/, "");
    return t && tail ? [`${label}: ${tail}`] : [];
  });
  return tails.length ? `${head} — ${tails.join("; ")}` : head;
}

function summaryFootnotes(r: BenchRow, all: BenchRow[] = [r]): Partial<Record<"stored" | "prefill" | "decode", string>> {
  const uniq = (...ss: string[]) => [...new Set(ss.filter(Boolean))].join(" · ");
  const pre = r.phases.prefill.units;
  const dec = r.phases.decode.units;
  // why a row has no time, once per distinct reason, naming its rows
  const untimed = new Map<string, string[]>();
  for (const row of all) {
    if (!row.timings_unavailable) continue;
    untimed.set(row.timings_unavailable, [...(untimed.get(row.timings_unavailable) ?? []), row.entry ?? row.component]);
  }
  const noTime = [...untimed].map(([why, who]) => `not timed (${who.join(", ")}): ${why}`).join(" · ");
  const time = (phase: PhaseKey) => {
    const t = all.find((row) => row.timings?.[phase])?.timings?.[phase];
    return [t
      ? `${t.gpu_sku} · t measured shown only at full coverage (${t.measured_rows}/${t.total_rows} measured)`
      : "", noTime].filter(Boolean).join(" — ");
  };
  return {
    stored: uniq(r.units.weight_bytes, r.kv.unit),
    prefill: [phaseFootnote([["FLOPs", pre.flops], ["Weight read", pre.weight_read],
                             ["KV write / step", pre.traffic], ["Act", pre.act_bytes]]),
              time("prefill")].filter(Boolean).join(" — "),
    decode: [phaseFootnote([["FLOPs", dec.flops], ["Weight read", dec.weight_read],
                            ["KV read / step", dec.traffic], ["Act", dec.act_bytes]]),
             time("decode")].filter(Boolean).join(" — "),
  };
}

// A row "varies" when its columns don't all agree — those are the
// reason-to-look rows, so their values get a highlight tint. A MISSING cell
// counts as a value of its own ("∅"): a dim/op that exists in only one column
// is exactly the signal this highlight exists for. Cells compared by their
// VISIBLE content (structure included), not one scalar.
const rowVaries = (vals: (number | string | null | undefined)[]) =>
  new Set(vals.map((v) => (v == null ? "∅" : v))).size > 1;

// One cell of the aligned kvcache table: elements + dtype + law, one renderer
// shared by the aligned table and the Detail card.
const kvCellKey = (n: KVNode) =>
  `${n.elements_per_token ?? n.elements_per_request ?? "∅"}|${n.elements_per_token != null ? "tok" : "req"}|${n.dtype}|${n.law}|${n.window ?? ""}|${n.owner ?? ""}`;
const kvLaw = (n: KVNode) => (n.law === "windowed" && n.window ? `windowed (${n.window})` : n.law);
const kvRowId = (n: KVNode) => n.owner ? `slot:${n.owner}.${n.name}` : n.name;

const colKey = (r: BenchRow) => r.entry ?? r.component;

type EntryHeadOpts = {
  results: BenchRow[];
  baseModelId?: string;
  editedKeys?: Set<string>;
  canUseBase?: (component: string) => boolean;
  decodeOf: Map<string, string>;
  descOf: Map<string, string>;
  colorOf?: (r: BenchRow) => string;
  onSaveVariant?: (name: string, component: string, dims: Record<string, number>) => void;
  signedIn?: boolean;
  savingReady?: boolean;
  onRebind?: (key: string, modelId: string | null) => void;
  withRebind?: boolean;
};

function EntryColumnHead({ r, opts }: { r: BenchRow; opts: EntryHeadOpts }) {
  const { editedKeys, canUseBase, decodeOf, descOf,
          colorOf, onSaveVariant, signedIn, savingReady, onRebind, withRebind } = opts;
  return (
    <>
      <div className="font-mono text-gray-800">
        {colorOf && (
          <span className="inline-block w-2.5 h-2.5 rounded-sm mr-1.5 align-middle"
            style={{ background: colorOf(r) }} />
        )}
        {r.component}
      </div>
      <div className="font-normal text-11 text-amber-700">
        {r.model_id ? `@ ${shortModel(r.model_id)}` : r.entry?.includes("~my:") ? entryLabel(r.entry) : "custom"}
        {editedKeys?.has(colKey(r)) && (
          <>
            <span className="text-sky-600" title="customized — dims edited away from the binding's seed"> · edited</span>
            {signedIn && savingReady && onSaveVariant && (
              <button type="button"
                onClick={async () => {
                  const name = await promptDialog({ title: "Save this customization as my module",
                    message: "Name it. Saving under a name you already use overwrites that module.",
                    confirmLabel: "Save" });
                  if (name?.trim()) onSaveVariant(name.trim(), r.component, r.editable_dims ?? {});
                }}
                title="save this column's full resolved parameterization as one of my modules"
                className="ml-1 text-10 px-1.5 py-0.5 rounded border border-sky-300 text-sky-700 hover:bg-sky-50 transition">
                ☆ save
              </button>
            )}
          </>
        )}
      </div>
      {decodeOf.has(r.component) && (
        <div className="font-normal text-10 text-gray-400"
          title={`Declared decode-time form of ${decodeOf.get(r.component)}.${descOf.get(r.component) ? ` ${descOf.get(r.component)}` : ""}`}>
          decode form of {decodeOf.get(r.component)}
        </div>
      )}
      <div className="font-normal"><ProvenanceChip ids={r.model_ids} /></div>
      {withRebind && onRebind && canUseBase && !r.entry?.includes("~my:") && (r.model_ids?.length ?? 0) > 0 && (
        <select
          className="mt-1 text-11 border border-gray-200 rounded px-1 py-0.5 bg-white text-gray-600 font-normal"
          value={r.model_id ?? r.model_ids![0]}
          onChange={(e) => onRebind(colKey(r), e.target.value)}
          title="Bind this column to a shipped model — its dims become that model's parameterization (edits reset)">
          {r.model_ids!.map((mid) => (
            <option key={mid} value={mid}>{shortModel(mid)}</option>
          ))}
        </select>
      )}
    </>
  );
}

function KvTensors({ results, colorOf, decodeOf, descOf }: {
  results: BenchRow[];
  colorOf: (r: BenchRow) => string;
  decodeOf: Map<string, string>;
  descOf: Map<string, string>;
}) {
  const cols = results;
  const kvTensorIds = Array.from(
    new Set(cols.flatMap((r) => [
      ...(r.kv.nodes || []).map(kvRowId),
      ...(r.kv.slot_nodes || []).map(kvRowId),
    ]))
  ).sort();
  if (kvTensorIds.length === 0) return null;
  const head: EntryHeadOpts = { results: cols, decodeOf, descOf, colorOf };
  type KvTensorRow = { id: string; tn: string; cells: (KVNode | undefined)[]; varies: boolean };
  const tensorRows: KvTensorRow[] = kvTensorIds.map((id) => {
    const cells = cols.map((r) => [
      ...(r.kv.nodes || []), ...(r.kv.slot_nodes || []),
    ].find((n) => kvRowId(n) === id));
    const tn = cells.find(Boolean)?.name || id;
    return { id, tn, cells, varies: rowVaries(cells.map((n) => (n ? kvCellKey(n) : null))) };
  });
  const columns: ColumnDef<KvTensorRow>[] = [
    {
      key: "tensor",
      header: " ",
      className: (row) => `font-sans ${row.varies ? "text-amber-700 font-medium" : "text-gray-500"}`,
      render: (row) => row.tn,
      sortValue: (row) => row.id,
    },
    ...cols.map((r, c): ColumnDef<KvTensorRow> => ({
      key: colKey(r),
      header: <EntryColumnHead r={r} opts={head} />,
      align: "right",
      headerClassName: "align-bottom",
      className: (row) => `font-mono text-gray-800 ${row.varies ? "bg-amber-50" : ""}`,
      sortValue: (row) => row.cells[c]?.elements_per_token,
      render: (row) => {
        const n = row.cells[c];
        return n ? (
          <>
            <KvEl n={n} />
            <span className="text-gray-400 font-sans">
              {" "}{n.dtype} · {kvLaw(n)}
              {n.owner ? ` · slot: ${n.owner}` : ""}
            </span>
          </>
        ) : <span className="text-gray-300">—</span>;
      },
    })),
  ];
  return (
    <details className="mt-4 text-xs">
      <summary className="text-gray-500 cursor-pointer">
        kvcache tensors — elements per token (or per request) · dtype · law
        (aligned by tensor name; “slot: …” rows are graph-visible caches owned
        by bound sub-components, outside this component’s per-layer total)
      </summary>
      <DataTable className="mt-1" columns={columns} rows={tensorRows} rowKey={(row) => row.id} framed={false} />
    </details>
  );
}

/** One phase's kernels as the rows Inference's PerOpTable paints. The
 *  roofline rows the backend priced on the selected GPU
 *  (`timings[phase].op_rows`) are the same numbers under that table's
 *  column names — shapes, times, bound, measured included — and pass
 *  through untouched; the resource-only runtime table
 *  (`op_tables[phase].rows`) is the fallback when no GPU priced the
 *  phase. */
function opRows(row: BenchRow, phase: PhaseKey): PerOpRow[] {
  const roofline = row.timings?.[phase]?.op_rows;
  if (roofline && roofline.length > 0) return roofline;
  return (row.op_tables?.[phase]?.rows ?? []).map((op) => ({
    aggregation_id: op.node_id,
    name: op.label,
    op: op.kernel,
    flops_per_rank: op.flops,
    access_bytes_per_rank: op.total_bytes,
    arithmetic_intensity: op.arithmetic_intensity,
    flops_formula: op.formulas?.flops,
    access_bytes_formula: op.formulas?.total_bytes,
  }));
}

/** The Total row for `opRows`: the roofline table's own total, or — for the
 *  resource-only fallback — that table's backend totals under the same
 *  column names (no time: nothing priced it). */
function opTotal(row: BenchRow, phase: PhaseKey): OpTableTotal | undefined {
  const timing = row.timings?.[phase];
  if (timing?.op_rows && timing.op_rows.length > 0) return timing.op_table_total;
  const totals = row.op_tables?.[phase]?.totals;
  if (!totals) return undefined;
  return {
    flops_per_rank: totals.flops,
    access_bytes_per_rank: totals.total_bytes,
    comm_bytes_per_rank: 0,
    t_sol_s: null,
  };
}

function OpTable({ results, colorOf, phase, onPhase, gpuSku, gpus, onGpuSku }: {
  results: BenchRow[]; colorOf: (r: BenchRow) => string;
  phase: PhaseKey; onPhase: (p: PhaseKey) => void;
  gpuSku: string; gpus: GpuItem[]; onGpuSku: (s: string) => void;
}) {
  const rowsOf = (r: BenchRow, p: PhaseKey) => opRows(r, p).length;
  if (!results.some((r) => rowsOf(r, "prefill") + rowsOf(r, "decode") > 0)) return null;
  const shown = results.filter((r) => rowsOf(r, phase) > 0);
  const first = shown[0]?.op_tables[phase];
  const tabCls = (k: PhaseKey) =>
    "px-2 py-1 text-11 font-medium " + (phase === k
      ? "bg-amber-500 text-white"
      : "bg-white text-gray-600 hover:bg-gray-50");
  const swapped = Array.from(new Map(results
    .filter((r) => r.phases[phase].runs_as !== r.component)
    .map((r) => [r.component, r.phases[phase].runs_as])).entries());
  return (
    <div id="pg-op-table" className="mt-4 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="text-gray-500">
          {first && (
            <span className="text-gray-400">
              {" "}· B={first.contract.batch} · L={formatTokens(first.contract.input_tokens)}
              {first.contract.activation_dtype ? ` · act ${first.contract.activation_dtype}` : ""}
              {first.contract.precision ? ` · ${first.contract.precision.label}` : ""}
              {" "}· roofline on {gpuSku}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <label className="inline-flex items-center gap-1 text-11 text-gray-500">
            GPU
            <select value={gpuSku} onChange={(e) => onGpuSku(e.target.value)}
              className="pg-serving-gpu text-xs border border-gray-300 rounded-lg px-1.5 py-1 bg-white">
              {gpuOptions(gpus, gpuSku).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </label>
          <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden" role="tablist" aria-label="Runtime table phase">
            <button type="button" className={tabCls("prefill")} role="tab"
              aria-selected={phase === "prefill"} onClick={() => onPhase("prefill")}>Prefill</button>
            <button type="button" className={tabCls("decode")} role="tab"
              aria-selected={phase === "decode"} onClick={() => onPhase("decode")}>Decode</button>
          </div>
        </div>
      </div>
      {swapped.length > 0 && (
        <div className="text-xs text-gray-400 mb-2">
          {swapped.map(([c, twin]) => (
            <div key={c}>{phase} runs <span className="font-mono">{c}</span> as its declared decode form <span className="font-mono">{twin}</span></div>
          ))}
        </div>
      )}
      {shown.map((row) => (
        <div key={colKey(row)} className="mt-3" data-group-header={colKey(row)}>
          <div className="flex items-center gap-1.5 mb-1 text-gray-600">
            <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: colorOf(row) }} />
            <span className="font-mono">{rowLabel(row)}</span>
          </div>
          <PerOpTable
            rows={opRows(row, phase)}
            moduleRows={row.timings?.[phase]?.module_rows}
            total={opTotal(row, phase)}
            byDeclaration={[]}
            phase={phase}
            id={`pg-op-${colKey(row)}`}
            caption={rowLabel(row)}
            measured={row.timings?.[phase]?.t_measured_s ?? null}
          />
        </div>
      ))}
    </div>
  );
}

function DimsTable({ results, baseModelId, entryDims, seedDims, editedKeys, canUseBase, decodeOf, descOf, onSaveVariant, signedIn, savingReady, onRebind, onEditDim }: {
  results: BenchRow[]; baseModelId: string;
  entryDims: Record<string, Record<string, number>>;
  seedDims: Record<string, Record<string, number> | undefined>;
  editedKeys: Set<string>;
  canUseBase: (component: string) => boolean;
  decodeOf: Map<string, string>;
  descOf: Map<string, string>;
  onSaveVariant: (name: string, component: string, dims: Record<string, number>) => void;
  signedIn: boolean;
  savingReady: boolean;
  onRebind: (key: string, modelId: string | null) => void;
  onEditDim: (key: string, dim: string, value: number | null) => void;
}) {
  const cols = results;
  const dimKeys = Array.from(
    new Set(cols.flatMap((r) => Object.keys(r.editable_dims || {})))
  ).sort();
  if (dimKeys.length === 0) return null;
  const head: EntryHeadOpts = {
    results: cols, baseModelId, editedKeys, canUseBase, decodeOf, descOf,
    onSaveVariant, signedIn, savingReady, onRebind, withRebind: true,
  };
  type DimRow = { k: string; varies: boolean };
  const dimRows: DimRow[] = dimKeys.map((k) => ({
    k,
    varies: rowVaries(cols.map((r) => (r.editable_dims || {})[k])),
  }));
  const columns: ColumnDef<DimRow>[] = [
    {
      key: "dim",
      header: " ",
      className: (row) => `pg-dim-name font-sans ${row.varies ? "text-amber-700 font-medium" : "text-gray-500"}`,
      render: (row) => <DimSymbol symbol={row.k} meaning={dimMeaning(row.k)} className="" />,
      sortValue: (row) => row.k,
    },
    ...cols.map((r): ColumnDef<DimRow> => ({
      key: colKey(r),
      header: <EntryColumnHead r={r} opts={head} />,
      align: "right",
      headerClassName: "align-bottom",
      className: (row) => `font-mono text-gray-800 ${row.varies ? "bg-amber-50" : ""}`,
      // The header holds the column's own controls (☆ save, the rebind
      // <select>); a sort toggle would wrap them in a second button.
      sortValue: null,
      render: (row) => {
        const k = row.k;
        const v = (r.editable_dims || {})[k];
        const seed = seedDims[colKey(r)];
        const cur = entryDims[colKey(r)]?.[k];
        const customized = seed ? (cur != null && seed[k] !== cur) : cur != null;
        return v != null ? (
          <input type="number" defaultValue={v} key={`${colKey(r)}:${k}:${v}`}
            onBlur={(e) => {
              const t = e.target.value.trim();
              const n = Number(t);
              if (t === "") {
                if (seed?.[k] != null && cur !== seed[k]) onEditDim(colKey(r), k, seed[k]);
                else if (cur != null) onEditDim(colKey(r), k, null);
                return;
              }
              if (Number.isFinite(n) && n !== v) onEditDim(colKey(r), k, n);
            }}
            className={`w-20 text-right bg-transparent border-b ${customized
              ? "border-amber-400 text-amber-700"
              : "border-transparent hover:border-gray-300"} focus:border-gray-500 focus:outline-none`}
            title={customized
              ? "customized — blank + blur to restore the binding's seed"
              : "edit to customize this column"} />
        ) : <span className="text-gray-300">—</span>;
      },
    })),
  ];
  return (
    <details className="mt-2 text-xs" open={editedKeys.size > 0 || undefined}>
      <summary className="text-gray-500 cursor-pointer">
        internal dims · editable per component (why the metrics differ)
      </summary>
      <DataTable className="mt-1" columns={columns} rows={dimRows} rowKey={(row) => row.k} framed={false} />
    </details>
  );
}

function DetailsSection({ results, baseModelId, entryDims, seedDims, editedKeys, canUseBase, decodeOf, descOf, onSaveVariant, signedIn, savingReady, onRebind, onEditDim, colorOf, phase, onPhase, gpuSku, gpus, onGpuSku }: {
  results: BenchRow[]; baseModelId: string;
  entryDims: Record<string, Record<string, number>>;
  seedDims: Record<string, Record<string, number> | undefined>;
  editedKeys: Set<string>;
  canUseBase: (component: string) => boolean;
  decodeOf: Map<string, string>;
  descOf: Map<string, string>;
  onSaveVariant: (name: string, component: string, dims: Record<string, number>) => void;
  signedIn: boolean;
  savingReady: boolean;
  onRebind: (key: string, modelId: string | null) => void;
  onEditDim: (key: string, dim: string, value: number | null) => void;
  colorOf: (r: BenchRow) => string;
  phase: PhaseKey; onPhase: (p: PhaseKey) => void;
  gpuSku: string; gpus: GpuItem[]; onGpuSku: (s: string) => void;
}) {
  return (
    <details id="pg-details" className="group/details card p-4 sm:p-6 mb-6 min-w-0 scroll-mt-[calc(3.5rem+var(--pg-picks-h,0px)+2.75rem)]">
      <summary className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 cursor-pointer list-none [&::-webkit-details-marker]:hidden">
        <h2 className="text-lg font-semibold text-gray-900 flex items-baseline gap-2">
          <span aria-hidden="true" className="inline-block text-gray-400 text-sm transition-transform group-open/details:rotate-90">▶</span>
          Details
          <span className="text-xs font-normal text-gray-500 group-open/details:hidden">click to expand</span>
        </h2>
        <span className="text-11 text-gray-400 min-w-0">dims independent · runtime table follows phase · roofline priced on the selected GPU</span>
      </summary>
      <DimsTable results={results} baseModelId={baseModelId} entryDims={entryDims} seedDims={seedDims}
        editedKeys={editedKeys} canUseBase={canUseBase} decodeOf={decodeOf} descOf={descOf}
        onSaveVariant={onSaveVariant} signedIn={signedIn} savingReady={savingReady}
        onRebind={onRebind} onEditDim={onEditDim} />
      <OpTable results={results} colorOf={colorOf} phase={phase} onPhase={onPhase}
        gpuSku={gpuSku} gpus={gpus} onGpuSku={onGpuSku} />
    </details>
  );
}

// Bindings table: selected component(s) as EACH shipped model parameterizes
// them (the "who is who" of their dims). One component keeps model × dims.
// Mixed picks pivot: rows are models, columns group each component, so the
// same checkpoint's parameterization of every architecture sits on one row.
// "+ compare" turns one binding into its own compare entry. Three honest
// states: loading / library-only / table, with bindings that failed to resolve
// listed with their reason (a failed binding is reported, never silently dropped).
function BindingToggle({ added, twin, capReached, onToggle }: {
  added: boolean;
  twin?: string;
  capReached: boolean;
  onToggle: () => void;
}) {
  return (
    <button type="button"
      disabled={!!twin || (!added && capReached)}
      onClick={onToggle}
      title={added ? "remove this column"
        : twin ? `identical dims to the ${twin} column`
        : capReached ? "a bench compares at most 8 columns — remove one first"
        : "add this parameterization as its own column"}
      className={`text-11 px-2 py-0.5 rounded border whitespace-nowrap transition ${added
        ? "border-amber-500 bg-amber-50 text-amber-800 hover:bg-amber-100"
        : twin || capReached
          ? "border-gray-200 text-gray-400 cursor-default"
          : "border-amber-400 text-amber-700 hover:bg-amber-50"}`}>
      {added ? "✓ added ×" : twin ? `≡ ${twin}` : "+ compare"}
    </button>
  );
}

type BindingsBlock = {
  name: string;
  bindings: BindingsPack | null;
  myVariants: ModuleVariant[] | null;
  // dims-signature → label of the entry already covering it for THIS
  // component: a twin binding (identical parameterization) is not addable.
  addedSigs: Map<string, string>;
};

const dimsOf = (block: BindingsBlock) =>
  Array.from(new Set((block.bindings?.bindings ?? []).flatMap((r) => Object.keys(r.dims)))).sort();

function BindingsTable({ blocks, addedKeys, capReached, onAdd, onRemove, onAddVariant }: {
  blocks: BindingsBlock[];
  addedKeys: Set<string>;
  capReached: boolean; // backend caps a bench at 8 entries
  onAdd: (name: string, modelId: string, call: string | null, dims: Record<string, number>) => void;
  onRemove: (key: string) => void;
  onAddVariant: (v: ModuleVariant) => void;
}) {
  const showComponent = blocks.length > 1;
  const pending = blocks.filter((b) => b.bindings === null);
  const loaded = blocks.filter((b) => b.bindings !== null);
  const allRows = loaded.flatMap((b) => b.bindings!.bindings);
  const failed = loaded.flatMap((b) =>
    Object.entries(b.bindings!.failed).map(([mid, reason]) => ({ name: b.name, mid, reason })));
  const mine = loaded.flatMap((b) => (b.myVariants ?? []).map((v) => ({ name: b.name, v })));
  const titleName = blocks.length === 1 ? blocks[0].name : null;

  if (loaded.length === 0) {
    return (
      <div>
        <h3 className="text-sm font-semibold text-gray-900">
          Bound models{titleName ? <> — <span className="font-mono">{titleName}</span></> : null}
        </h3>
        <div className="mt-2 text-xs text-gray-400">Resolving bound models…</div>
      </div>
    );
  }

  const empty = allRows.length === 0 && failed.length === 0 && mine.length === 0 && pending.length === 0;
  if (empty) {
    // Library-only: say so instead of vanishing the whole panel —
    // appearing for one pick and disappearing for the next reads as a bug.
    return (
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">
          Bound models — {titleName
            ? <span className="font-mono">{titleName}</span>
            : "selected components"} as each shipped model parameterizes it
        </h3>
        <div className="text-xs text-gray-400">no shipped model binds {titleName ? "this component" : "these components"} (library-only)</div>
      </div>
    );
  }

  // One section per component; each parameterization a model uses is one
  // row (its first call, with how many calls evaluate alike). A dim two or more
  // sections carry has one column in every section (most widely carried
  // first), so it lines up across architectures — a section without it
  // leaves that column blank; each section's own dims share the columns
  // after those. Each section's header row names its own dims.
  const sections = blocks.map((block) => {
    const all = block.bindings?.bindings ?? [];
    const rows = all.filter((b, i) => all.findIndex((x) =>
      x.model_id === b.model_id && x.parameterization === b.parameterization) === i);
    const alikeCalls = (b: BindingRow) => all.filter((x) =>
      x.model_id === b.model_id && x.parameterization === b.parameterization).length;
    return { block, rows, dims: dimsOf(block), alikeCalls };
  });
  const carriers = new Map<string, number>();
  for (const s of sections) for (const k of s.dims) carriers.set(k, (carriers.get(k) ?? 0) + 1);
  const shared = [...carriers].filter(([, n]) => n > 1)
    .sort(([a, n], [b, m]) => m - n || a.localeCompare(b)).map(([k]) => k);
  const columnsOf = (dims: string[]): (string | null)[] => [
    ...shared.map((k) => (dims.includes(k) ? k : null)),
    ...dims.filter((k) => !shared.includes(k)),
  ];
  const width = Math.max(0, ...sections.map((s) => columnsOf(s.dims).length));

  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-900 mb-1">
        Bound models — {titleName
          ? <span className="font-mono">{titleName}</span>
          : "selected components"} as each shipped model parameterizes it
      </h3>
      <div className="text-xs text-gray-400 mb-2">
        Each row is one parameterization a shipped model uses; dims that differ between rows are
        darker. <span className="text-amber-700">+ compare</span> adds it as a bench column.
      </div>
      <div className={tbl.wrap}>
        <table className="w-full text-xs">
          {sections.map(({ block, rows, dims, alikeCalls }) => {
            const cols = columnsOf(dims);
            const pad = width - cols.length;
            const varies = new Set(dims.filter((k) =>
              new Set(rows.filter((b) => !b.error).map((b) => b.dims[k])).size > 1));
            return (
              <tbody key={block.name} data-component={block.name}
                className="border-t border-gray-200 first:border-t-0">
                <tr className="bg-gray-50 text-11 text-gray-500">
                  <th className="px-3 py-2 text-left font-medium whitespace-nowrap">
                    {showComponent
                      ? <span className="font-mono text-xs font-semibold text-gray-800">{block.name}</span>
                      : "model"}
                  </th>
                  {cols.map((k, i) => (
                    <th key={k ?? `-${i}`} className="px-3 py-2 text-right font-mono font-normal whitespace-nowrap">{k}</th>
                  ))}
                  {pad > 0 && <th colSpan={pad} />}
                  <th />
                </tr>
                {block.bindings === null && (
                  <tr><td colSpan={width + 2} className="px-3 py-2 text-gray-400">Resolving…</td></tr>
                )}
                {rows.map((b, i) => {
                  const key = entryKey(block.name, b.model_id, b.call);
                  const added = addedKeys.has(key);
                  const twin = added ? undefined : block.addedSigs.get(dimsSig(b.dims));
                  const firstOfModel = i === 0 || rows[i - 1].model_id !== b.model_id;
                  return (
                    <tr key={`${b.model_id}#${b.call}`} data-call={b.call}
                      className={`${firstOfModel ? "border-t border-gray-100" : ""} ${added ? "bg-amber-50/60" : "hover:bg-gray-50"}`}>
                      <td className="px-3 py-1.5 whitespace-nowrap" title={`${b.model_id}\n${callText(b)}`}>
                        {firstOfModel && <div className="text-gray-800 font-medium">{shortModel(b.model_id)}</div>}
                        <div className="text-11 text-gray-400">
                          {alikeCalls(b) > 1 ? `${b.label} · ${alikeCalls(b)} calls alike` : b.label}
                        </div>
                      </td>
                      {b.error ? (
                        <td colSpan={width} data-call-refused={b.call} title={b.error}
                          className="px-3 py-1.5 text-right text-gray-400 italic">
                          not evaluable alone
                        </td>
                      ) : (
                        <>
                          {cols.map((k, i) => k === null ? <td key={`-${i}`} /> : (
                            <td key={k} className={`px-3 py-1.5 text-right font-mono tabular-nums ${
                              b.dims[k] == null ? "text-gray-300"
                                : varies.has(k) ? "text-gray-900 font-medium" : "text-gray-400"}`}>
                              {b.dims[k] != null ? formatDim(b.dims[k]) : "—"}
                            </td>
                          ))}
                          {pad > 0 && <td colSpan={pad} />}
                        </>
                      )}
                      <td className="px-3 py-1.5 text-right w-px">
                        {!b.error && (
                          <BindingToggle added={added} twin={twin} capReached={capReached}
                            onToggle={() => added ? onRemove(key) : onAdd(block.name, b.model_id, b.call, b.dims)} />
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            );
          })}
        </table>
      </div>
      {allRows.length === 0 && failed.length > 0 && (
        <div className="text-xs text-amber-600 mt-1">
          {failed.length} model{failed.length === 1 ? "" : "s"} bind {titleName ? "this component" : "these components"}, but none resolved.
        </div>
      )}
      {mine.length > 0 && (
        <div className="mt-3">
          <div className="text-xs text-gray-500 mb-1">mine — your saved parameterizations</div>
          <div className="flex flex-wrap gap-2">
            {mine.map(({ name, v }) => {
              const key = variantKey(v.component, v.name);
              const added = addedKeys.has(key);
              return (
                <button key={v.id} type="button"
                  disabled={!added && capReached}
                  onClick={() => added ? onRemove(key) : onAddVariant(v)}
                  title={added ? "remove this column"
                    : `${Object.entries(v.dims).map(([k, x]) => `${k}=${x}`).join(", ")}${v.base_model_id ? ` — tuned under ${v.base_model_id}` : ""}`}
                  className={`text-xs px-2.5 py-1 rounded-full border transition ${added
                    ? "border-sky-500 bg-sky-50 text-sky-800"
                    : capReached
                      ? "border-gray-200 text-gray-400 cursor-default"
                      : "border-sky-300 text-sky-700 hover:bg-sky-50"}`}>
                  {added
                    ? `✓ my:${v.name} ×`
                    : showComponent ? `my:${name} · ${v.name}` : `my:${v.name}`}
                </button>
              );
            })}
          </div>
        </div>
      )}
      {failed.length > 0 && (
        <details className="mt-2 text-xs">
          <summary className="text-gray-400 cursor-pointer">
            {failed.length} binding{failed.length === 1 ? "" : "s"} failed to resolve
          </summary>
          <ul className="mt-1 space-y-0.5 text-gray-500">
            {failed.map(({ name, mid, reason }) => (
              <li key={`${name}@${mid}`}>
                {showComponent && <span className="font-mono">{name}</span>}
                {showComponent ? " · " : null}
                <span className="font-mono">{shortModel(mid)}</span>: {reason}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

// ---- Model editor (Phase 2a.2): edit a model's layer groups, edited vs base ----
//
// The backend lists the model's layer groups (one per block template in use:
// DeepSeek-V3's dense layers, its MoE layers, its MTP layer), each with its
// layers, its slots and their candidates, and the dims it depends on. The
// user swaps a slot in one group — or in every group bound to the same
// component, expanded here into one explicit entry per group — and edits
// dims globally or per group. Every number, label and refusal comes from
// `/api/playground/model-skeleton` and `/api/playground/compose-model`.
type LocalDim = { value: number; global?: number; binding?: string; interface?: boolean;
                  group?: string | null };
type SkelSlot = {
  slot_id: string; node_id: string; binding: string; role: string; component: string;
  kind: string; label: string; candidates: string[];
  // The base's calls at this slot: a swapped-in module is evaluated at one.
  calls: CallInfo[];
  bound_with: { slots: { group_id: number; slot_id: string }[]; groups: number; layers: number };
};
type SkelGroup = {
  group_id: number; template_name: string; count: number; assignment_index: number | null;
  stack: string | null; aux_kind: string | null; editable: boolean;
  layers: number[]; layers_label: string; selector: Record<string, unknown>;
  composition?: string[]; slots?: SkelSlot[];
  readonly?: { slot_id: string; node_id: string; role: string; component: string }[];
  local_dims?: Record<string, LocalDim>;
};
type ModelLevel = { embedding_params: number; lm_head_params: number; lm_head_flops: number; final_norm_and_other_params: number };
type Skeleton = {
  success: boolean; message?: string; model_id?: string; n_layers?: number; groups?: SkelGroup[];
  /** Stored as a config and a spec with no weights (a model described in words). */
  stored_without_checkpoint?: boolean;
  /** `user_generation` when the caller's own record answered (the account route). */
  served_from?: string;
};
type EditorPhase = {
  flops: number;
  act_bytes_naive: number;
  act_bytes_fused: number;
  weight_read_bytes: number;
  kv_step_bytes?: number;
  kv_read_bytes?: number;
  memory_dominant?: "kv" | "weights";
  units: { flops: string; act_bytes: string; traffic: string; weight_read: string; kv_bytes?: string };
};
type WholeCost = {
  params: number; active_params: number; flops: number; weight_bytes: number;
  kv_bytes_at_L: number; act_bytes_naive: number; act_bytes_fused: number;
  phases?: { prefill: EditorPhase; decode: EditorPhase };
  kv_storage_by_cache_len?: { cache_len: number; storage_bytes: number }[];
  timings?: Record<PhaseKey, PhaseTiming>;
  /** Why there is no `timings`: the card lists no peak for the dtype the
   *  step computes in. FLOPs and bytes stand without it. */
  timings_unavailable?: string;
};
type CostRow = {
  group_id: number | null; template: string; count: number; aux_kind: string | null;
  layers: number[]; layers_label: string;
  total_params: number; total_active_params: number; total_flops: number;
  total_storage_bytes: number; kv_bytes_at_L: number | null;
  // the group's whole contribution (all `count` layers), the unit of kv_bytes_at_L
  group_totals: { params: number; active_params: number; flops: number; storage_bytes: number };
};
type Totals = { params: number; active_params: number; flops: number; storage_bytes: number };
type ComposeError = {
  kind: string; group_id: number | null; slot?: string | null; detail: string;
  component?: string | null; missing_dims?: string[];
};
type ComposeRow = { group_id: number | null; aux_kind: string | null; base: CostRow | null; edited: CostRow | null; changed: boolean };
type ComposeResp = {
  success: boolean; message?: string;
  composed?: WholeCost; base?: WholeCost;
  rows?: ComposeRow[];
  model_level?: { base: ModelLevel & { totals: Totals }; edited: ModelLevel & { totals: Totals } };
  groups?: { group_id: number; dim_changes: Record<string, { global: number; group: number }>; local_dims: Record<string, LocalDim> }[];
  errors?: ComposeError[];
};

const EMPTY_UNITS = { flops: "", act_bytes: "", kv_bytes: "", traffic: "", weight_read: "", weight_bytes: "" };
const EMPTY_WEIGHT: WeightStorage = {
  bytes: 0, parameter_bytes: 0, metadata_bytes: 0, buffer_bytes: 0, by_dtype: [],
  precision: "", source: { model_id: null, bound: false, invoked: false, role: null, address: "", note: "" },
};

/** Batch 5a target state on a successful `composed`/`base`. Missing any of
 *  these is an explicit error — do not pad from the old whole-model fields. */
function wholeHasTargetState(w: WholeCost | undefined): boolean {
  return !!w && !!w.phases?.prefill && !!w.phases?.decode && Array.isArray(w.kv_storage_by_cache_len);
}
function composeHasTargetState(resp: ComposeResp): boolean {
  return wholeHasTargetState(resp.base) && wholeHasTargetState(resp.composed);
}

function padEditorPhase(p: EditorPhase): PhaseNumbers {
  return {
    flops: p.flops,
    act_read_bytes: 0, act_write_bytes: 0, act_read_bytes_fused: 0, act_write_bytes_fused: 0,
    act_bytes_naive: p.act_bytes_naive,
    act_bytes_fused: p.act_bytes_fused,
    kv_bytes: 0,
    weight_read_bytes: p.weight_read_bytes,
    kv_step_bytes: p.kv_step_bytes ?? 0,
    kv_read_bytes: p.kv_read_bytes ?? 0,
    memory_dominant: p.memory_dominant ?? "weights",
    fused_exceeds_naive: false,
    runs_as: "",
    units: {
      flops: p.units.flops,
      act_bytes: p.units.act_bytes,
      kv_bytes: p.units.kv_bytes ?? "",
      traffic: p.units.traffic,
      weight_read: p.units.weight_read,
    },
  };
}

/** Project the editor's whole-model `composed`/`base` onto the compare-mode
 *  row shape so SummaryTable and KvSweep are the same components. */
function editorAsBench(resp: ComposeResp, baseModel: string, batch: number, seqLen: number):
    (Omit<BenchResponse, "results"> & { results: BenchRow[] }) | null {
  if (!composeHasTargetState(resp) || !resp.base || !resp.composed) return null;
  const emptyRuntimeTable = (phase: PhaseKey): RuntimeOpTable => ({
    contract: { phase, batch, input_tokens: seqLen, fusion: "shipped",
                parallel: "single_rank_unsharded", activation_dtype: null,
                precision: null },
    rows: [],
    totals: { params: 0, active_params: 0, flops: 0, total_bytes: 0,
              arithmetic_intensity: null },
  });
  const asRow = (key: string, w: WholeCost): BenchRow => ({
    component: key, entry: key, model_id: null, role: null,
    flops: w.flops, params: w.params, active_params: w.active_params, weight_bytes: w.weight_bytes,
    weight_storage: { ...EMPTY_WEIGHT, bytes: w.weight_bytes, parameter_bytes: w.weight_bytes },
    act_read_bytes: 0, act_write_bytes: 0, act_read_bytes_fused: 0, act_write_bytes_fused: 0,
    units: EMPTY_UNITS, dims_used: {},
    kv: {
      display: null, family: null, supported: true, unit: "",
      cache_len: seqLen, elements_per_token: 0,
      storage_bytes_at_L: w.kv_bytes_at_L, read_bytes_at_L: 0,
      bytes_per_token: 0, state_bytes: 0,
      storage_by_cache_len: w.kv_storage_by_cache_len!.map((p) => ({
        cache_len: p.cache_len, storage_bytes: p.storage_bytes, model_storage_bytes: p.storage_bytes,
      })),
      model_layers: { model_id: baseModel, layers: 1, refused: null },
      nodes: [],
    },
    phases: {
      prefill: { ...padEditorPhase(w.phases!.prefill), runs_as: key },
      decode: { ...padEditorPhase(w.phases!.decode), runs_as: key },
    },
    timings: w.timings,
    timings_unavailable: w.timings_unavailable,
    op_tables: { prefill: emptyRuntimeTable("prefill"),
                 decode: emptyRuntimeTable("decode") },
  });
  return {
    success: true, base_model_id: baseModel,
    contract: {
      batch, input_tokens: seqLen, dim_overrides: {},
      phases: {
        prefill: { batch, s_query: seqLen, s_kv: seqLen },
        decode: { batch, s_query: 1, s_kv: seqLen },
      },
    },
    results: [asRow("base", resp.base), asRow("edited", resp.composed)],
    errors: [],
  };
}

// Options for a slot's swap dropdown: the slot's candidates (the backend's,
// by role), grouped by declared family (attn_display) as optgroups, and ALWAYS
// including the current component — so the select never silently shows the
// wrong current value.
function slotOptionGroups(components: ComponentInfo[], candidates: string[], current: string): [string, string[]][] {
  const names = candidates.includes(current) ? candidates : [current, ...candidates];
  const family = new Map(components.map((c) => [c.name, c.attn_display || "Other"]));
  const by = new Map<string, string[]>();
  for (const n of names) {
    const fam = family.get(n) || "Other";
    (by.get(fam) ?? by.set(fam, []).get(fam)!).push(n);
  }
  return Array.from(by.entries()).sort((a, b) => a[0] === "Other" ? 1 : b[0] === "Other" ? -1 : a[0].localeCompare(b[0]));
}

// Structure edits and the precision preset. A v2 draft carried a KV dtype
// from before precision profiles: its structure edits are read, its dtype is
// not (the page says so), and the next save writes v3.
const DRAFT_KEY = "pg-editor-draft-v3";
const V2_DRAFT_KEY = "pg-editor-draft-v2";
// The Phase 2a.1 draft named slots, not groups; it is not migrated. Its key
// is dropped the first time the editor opens, with a one-time notice.
const OLD_DRAFT_KEY = "pg-editor-draft";

const groupColor = (g: number) => PALETTE[g % PALETTE.length];
const selectorText = (s: Record<string, unknown>) =>
  Object.entries(s).map(([k, v]) => `${k}: ${Array.isArray(v) ? `[${v.join(", ")}]` : String(v)}`).join(" · ");

// Backbone groups by their first layer, then MTP, then the read-only towers
// (display order only; every request names the backend's group_id).
function displayOrder(groups: SkelGroup[]): SkelGroup[] {
  const rank = (g: SkelGroup) => !g.editable ? 2 : g.aux_kind ? 1 : 0;
  return [...groups].sort((a, b) => rank(a) - rank(b) || (a.layers[0] ?? 0) - (b.layers[0] ?? 0) || a.group_id - b.group_id);
}

// One cell per backbone layer, coloured by the group that holds it.
function LayerStrip({ groups }: { groups: { group_id: number | null; layers: number[]; label: string }[] }) {
  const owner = new Map<number, { group_id: number | null; label: string }>();
  for (const g of groups) for (const i of g.layers) owner.set(i, g);
  const cells = Array.from(owner.keys()).sort((a, b) => a - b);
  return (
    <div id="pg-layer-strip" className="flex w-full h-3 rounded overflow-hidden border border-gray-200" aria-label="layers by group">
      {cells.map((i) => {
        const g = owner.get(i)!;
        return <div key={i} data-layer={i} data-group={g.group_id ?? ""} className="flex-1 min-w-0"
          title={`layer ${i} · group ${g.group_id ?? "new"} · ${g.label}`}
          style={{ background: g.group_id == null ? "#9ca3af" : groupColor(g.group_id) }} />;
      })}
    </div>
  );
}

/** A frame that scrolls sideways shows a soft shadow on the side with more
 *  to see; the white covers scroll with the content and hide it at the ends
 *  (`local` vs `scroll` attachment). Rows with their own background draw
 *  over it, the rest show it. */
const SCROLL_EDGE_SHADOWS = {
  background: [
    "linear-gradient(to right, #fff 30%, rgb(255 255 255 / 0)) left center / 40px 100% no-repeat local",
    "linear-gradient(to left, #fff 30%, rgb(255 255 255 / 0)) right center / 40px 100% no-repeat local",
    "radial-gradient(farthest-side at 0 50%, rgb(0 0 0 / 0.16), transparent) left center / 14px 100% no-repeat scroll",
    "radial-gradient(farthest-side at 100% 50%, rgb(0 0 0 / 0.16), transparent) right center / 14px 100% no-repeat scroll",
    "#fff",
  ].join(", "),
} as const;

function ErrorLine({ e, causedBy }: { e: ComposeError; causedBy?: number[] }) {
  return (
    <div className="pg-editor-error text-xs text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1 break-words">
      {causedBy && <span className="pg-error-cause text-red-600">caused by the edit on group{causedBy.length > 1 ? "s" : ""} {causedBy.join(", ")}: </span>}
      <span className="font-mono font-semibold">{e.kind}</span>
      {e.slot ? <span className="font-mono"> · {e.slot}</span> : null}
      <span> — {e.detail}</span>
      {e.missing_dims && e.missing_dims.length > 0 && (
        <span className="ml-1">[{e.missing_dims.map((d, i) => (
          <span key={d}>{i > 0 ? ", " : ""}<DimSymbol symbol={d} meaning={dimMeaning(d)} className="font-mono" /></span>
        ))}]</span>
      )}
    </div>
  );
}

// A group's own dims: the dims its evaluation depends on, valued as the group
// binds them, under the section dims.yaml files each symbol in (the backend
// sends them in that order). An interface dim (the residual width) is
// global-only.
const UNREGISTERED = "not in the dims registry";

function GroupDimsPanel({ group, value, onChange, changes }: {
  group: SkelGroup; value: Record<string, number>;
  onChange: (next: Record<string, number>) => void;
  changes?: Record<string, { global: number; group: number }>;
}) {
  const local = group.local_dims || {};
  const extra = Object.fromEntries(Object.entries(value).filter(([k]) => !(k in local)));
  // a dim's typed text while it is not a valid value (not sent)
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const set = (k: string, raw: string, base: number) => {
    const problem = raw.trim() === "" ? null : dimValueProblem(raw);
    setDrafts((d) => { const n = { ...d }; if (problem) n[k] = raw; else delete n[k]; return n; });
    if (problem) return;
    const next = { ...value };
    const n = Number(raw);
    if (raw.trim() === "" || n === base) delete next[k]; else next[k] = n;
    onChange(next);
  };
  const buckets = new Map<string, [string, (typeof local)[string]][]>();
  for (const [k, d] of Object.entries(local)) {
    const b = d.group ?? UNREGISTERED;
    const list = buckets.get(b) || [];
    list.push([k, d]);
    buckets.set(b, list);
  }
  const shownBuckets = Array.from(buckets.keys());
  const dimRow = (k: string, d: (typeof local)[string]) => (
    <div key={k} className="flex items-center gap-2 min-w-0">
      <span className="text-xs w-40 shrink-0 truncate">
        <DimSymbol symbol={k} meaning={dimMeaning(k)}
          binding={d.binding ? `${k} ← ${d.binding} = ${d.value}` : undefined} className="font-mono text-gray-700" />
      </span>
      {d.interface
        ? <span className="text-xs text-gray-400" title="the width of the tensors between layers: edit it globally">{formatDim(d.value)} · global only</span>
        : <input type="number" min={0} step={1} value={drafts[k] ?? value[k] ?? d.value}
            onChange={(e) => set(k, e.target.value, d.value)}
            className={`w-28 text-xs border rounded px-2 py-0.5 bg-white ${k in drafts ? "border-red-400" : k in value ? "border-amber-400" : "border-gray-300"}`} />}
      {d.global !== undefined && <span className="text-10 text-gray-400">model: {formatDim(d.global)}</span>}
      {k in drafts && <span className="pg-dim-invalid text-10 text-red-600">{dimValueProblem(drafts[k])} — not applied</span>}
    </div>
  );
  return (
    <div className="pg-group-dims space-y-3">
      {shownBuckets.map((b) => (
        <div key={b}>
          {shownBuckets.length > 1 && (
            <div className="text-10 font-semibold uppercase tracking-wide text-gray-400 mb-1">{b}</div>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-4 gap-y-1">
            {(buckets.get(b) || []).map(([k, d]) => dimRow(k, d))}
          </div>
        </div>
      ))}
      <div>
        <div className="text-11 text-gray-500 mb-1">other dims for this group (e.g. ones a swapped-in component needs)</div>
        <DimOverridesEditor value={extra}
          onChange={(next) => onChange({ ...Object.fromEntries(Object.entries(value).filter(([k]) => k in local)), ...next })} />
      </div>
      {changes && <DimChanges changes={changes} />}
    </div>
  );
}

// What a group's dims came to after its edit (`group_dim_changes`): each dim
// whose value in the group differs from the model's, the edited ones and the
// ones derived from them.
function DimChanges({ changes }: { changes: Record<string, { global: number; group: number }> }) {
  if (Object.keys(changes).length === 0) return null;
  return (
    <div className="pg-dim-changes text-xs text-gray-600">
      <span className="text-gray-500">in this group after the edit: </span>
      {Object.entries(changes).map(([k, c], i) => (
        <span key={k}>{i > 0 ? " · " : ""}<DimSymbol symbol={k} meaning={dimMeaning(k)} className="font-mono" /> {formatDim(c.global)} → <span className="font-semibold">{formatDim(c.group)}</span></span>
      ))}
    </div>
  );
}

const GROUP_TABLE_COLS = 9;

function groupCostCells(row: ComposeRow | undefined) {
  const b = row?.base, e = row?.edited;
  const changed = !!row?.changed;
  const cell = (col: string, get: (c: CostRow) => number, fmt: (n: number) => string) => (
    <td key={col} className={`pg-g-num ${tbl.wrapCell} min-w-0 text-right tabular-nums sm:whitespace-nowrap`} data-col={col} data-edited={e ? get(e) : 0}>
      {b ? fmt(get(b)) : "—"}{changed ? <> → <span className="font-semibold">{e ? fmt(get(e)) : "—"}</span></> : null}
    </td>
  );
  const kv = (c: CostRow | null | undefined) => !c ? "—" : c.kv_bytes_at_L == null ? "not in whole-model KV" : formatGiB(c.kv_bytes_at_L);
  return (
    <>
      {cell("params", (c) => c.group_totals.params, formatCount)}
      {cell("flops", (c) => c.group_totals.flops, (n) => formatSI(n, "FLOPs"))}
      {cell("weight_bytes", (c) => c.group_totals.storage_bytes, formatGiB)}
      <td className={`pg-g-num ${tbl.wrapCell} min-w-0 text-right tabular-nums sm:whitespace-nowrap`} data-col="kv" data-edited={e?.kv_bytes_at_L ?? 0}>
        {kv(b)}{changed ? <> → <span className="font-semibold">{kv(e)}</span></> : null}
      </td>
    </>
  );
}

function emptyCostCells() {
  return (
    <>
      <td className={`pg-g-num ${tbl.cell}`} data-col="params" data-edited={0} />
      <td className={`pg-g-num ${tbl.cell}`} data-col="flops" data-edited={0} />
      <td className={`pg-g-num ${tbl.cell}`} data-col="weight_bytes" data-edited={0} />
      <td className={`pg-g-num ${tbl.cell}`} data-col="kv" data-edited={0} />
    </>
  );
}

/** Decision 8: a swapped slot expands to the compare-mode graph diff of the
 *  old component (bound to the editor's base) vs the new one, both evaluated
 *  at the base's call at this slot — the user names which one when the
 *  slot's declaration runs as several calls. Fetches `/module-bench` only
 *  while open. */
function SlotDiff({ oldName, newName, dims, calls, baseModel, batch, seqLen, precision, gpuSku }: {
  oldName: string; newName: string; dims: Record<string, number>; calls: CallInfo[];
  baseModel: string; batch: number; seqLen: number; precision: string; gpuSku: string;
}) {
  const [open, setOpen] = useState(false);
  const [gd, setGd] = useState<GraphDiffData | null>(null);
  const [rows, setRows] = useState<DiffRow[] | null>(null);
  const [err, setErr] = useState("");
  const [picked, setPicked] = useState<string | null>(null);
  const call = calls.length === 1 ? calls[0].call : picked;
  const dimsJson = JSON.stringify(dims);
  useEffect(() => {
    if (!open || !call) return;
    let dead = false;
    setErr(""); setGd(null); setRows(null);
    void (async () => {
      try {
        const data = await fetchJson<BenchResponse>("/api/playground/module-bench", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            base_model_id: baseModel,
            entries: [
              { name: oldName, model_id: baseModel, call },
              { name: newName, call, dims: JSON.parse(dimsJson) as Record<string, number> },
            ],
            batch_size: batch, seq_len: seqLen, ...precisionBody(precision), gpu_sku: gpuSku,
          }),
        });
        if (dead) return;
        setGd(data.graph_diff ?? { available: false, reason: "no graph_diff in response" });
        setRows((data.results ?? []).map((r) => ({
          label: rowLabel(r), nodes: r.block_graph?.nodes ?? [], edges: r.block_graph?.edges ?? [], l1: r.l1,
          fused: r.fused_graph,
          fusedL1: r.fused_l1,
        })));
      } catch (e) {
        if (!dead) setErr(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { dead = true; };
  }, [open, call, oldName, newName, dimsJson, baseModel, batch, seqLen, precision, gpuSku]);
  return (
    <details className="pg-slot-diff mt-1" data-slot-diff={`${oldName}->${newName}`}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="text-xs text-gray-600 cursor-pointer">old vs new</summary>
      {calls.length === 0 && (
        <div className="text-xs text-red-700 mt-1">the base makes no call at this slot to evaluate the swap at</div>
      )}
      {calls.length > 1 && (
        <select data-call-pick={`${oldName}->${newName}`} value={picked ?? ""}
          onChange={(e) => { setPicked(e.target.value || null); setGd(null); setRows(null); }}
          title="This slot runs as several calls in the base: name the call to evaluate the swap at"
          className="text-xs border border-amber-300 rounded px-1.5 py-0.5 bg-white text-amber-700 mt-1 max-w-[28rem]">
          <option value="">evaluate at which call…</option>
          {calls.map((c) => <option key={c.call} value={c.call}>{callText(c)}</option>)}
        </select>
      )}
      {err && <div className="text-xs text-red-700 mt-1">{err}</div>}
      {gd && rows && (
        <div className="pg-slot-diff-view min-w-0 mt-2">
          <GraphDiffView gd={gd} rows={rows} />
        </div>
      )}
    </details>
  );
}

function ModelEditor({ models, components, gpuSku, gpus, onGpuSku, precision, precisionFromLink,
  onDraftPrecision, onLegacyDraftPrecision }: {
  models: string[]; components: ComponentInfo[];
  gpuSku: string; gpus: GpuItem[]; onGpuSku: (sku: string) => void;
  /** The page's precision preset id. */
  precision: string;
  /** Whether the page URL named a preset; a link's preset wins over a draft's. */
  precisionFromLink: boolean;
  onDraftPrecision: (presetId: string) => void;
  /** A draft from before precision profiles carried a KV dtype that was not applied. */
  onLegacyDraftPrecision: () => void;
}) {
  const [baseModel, setBaseModel] = useState("meta-llama/Llama-3.2-3B");
  const [skel, setSkel] = useState<Skeleton | null>(null);
  // Where the model comes from: the library, a description in words (the
  // agent writes it as the caller's own custom model), or one they keep.
  const [source, setSource] = useState<"library" | "describe" | "mine">("library");
  const [signedIn, setSignedIn] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [firstTurn, setFirstTurn] = useState<{ name: string; description: string; base?: string; seq: number } | null>(null);
  const [myReload, setMyReload] = useState(0);
  const myModels = useMyModels(signedIn, myReload);
  // A signed-in caller's requests go to the account routes, which answer
  // with the caller's own model where they keep one and as the library
  // otherwise; refs, since the loaders are stable callbacks.
  const playgroundRoute = useRef("/api/playground");
  useEffect(() => {
    void authContext().then((ctx) => {
      const authed = ctx.authEnabled && ctx.loggedIn;
      playgroundRoute.current = authed ? "/api/account/playground" : "/api/playground";
      setSignedIn(authed);
    });
  }, []);
  // group_id → slot path → component
  const [swaps, setSwaps] = useState<Record<number, Record<string, string>>>({});
  // `${group_id}|${slot}` set when the swap applies to every group bound to the component
  const [scopeAll, setScopeAll] = useState<Record<string, boolean>>({});
  const [groupDims, setGroupDims] = useState<Record<number, Record<string, number>>>({});
  const [openDims, setOpenDims] = useState<Record<number, boolean>>({});
  const [dims, setDims] = useState<Record<string, number>>({});
  const [batch, setBatch] = useState(1);
  const [seqLen, setSeqLen] = useState(2048);
  const [resp, setResp] = useState<ComposeResp | null>(null);
  const [status, setStatus] = useState("");
  const [statusKind, setStatusKind] = useState<"idle" | "ok" | "err">("idle");
  const [oldDraftNotice, setOldDraftNotice] = useState(false);
  // the groups the last compose request edited: an error landing elsewhere
  // (a stream reader of an edited producer) names them as its cause
  const [sentGroups, setSentGroups] = useState<number[]>([]);
  const [composeState, setComposeState] = useState<"idle" | "pending" | "done" | "error">("idle");
  const [composeErrors, setComposeErrors] = useState<ComposeError[]>([]);
  const [renderedSeq, setRenderedSeq] = useState(0);
  const sendSeq = useRef(0);
  const epoch = useRef(0);
  const lastPosted = useRef("");

  const loadSkeleton = useCallback(async (typed: string, keepEdits = false) => {
    const model = typed.trim();
    if (!model) { setStatus("Enter a model id (e.g. meta-llama/Llama-3.2-3B) to load."); setStatusKind("err"); return; }
    setStatus("Loading model…"); setStatusKind("idle"); setSkel(null); setResp(null);
    lastPosted.current = ""; sendSeq.current = 0; epoch.current++;
    setRenderedSeq(0); setComposeState("idle");
    if (!keepEdits) { setSwaps({}); setScopeAll({}); setGroupDims({}); setOpenDims({}); }
    try {
      const s = await fetchJson<Skeleton>(`${playgroundRoute.current}/model-skeleton?model_id=${encodeURIComponent(model)}`);
      setSkel(s);
      setStatus(s.success ? "Loaded." : `Failed: ${s.message}`);
      setStatusKind(s.success ? "ok" : "err");
    } catch (e) {
      setStatus(`Failed: ${e instanceof Error ? e.message : String(e)}`); setStatusKind("err");
    }
  }, []);

  // restore the draft once; an old-format draft is dropped with a notice
  useEffect(() => {
    if (localStorage.getItem(OLD_DRAFT_KEY) !== null) {
      localStorage.removeItem(OLD_DRAFT_KEY);
      setOldDraftNotice(true);
    }
    try {
      let d = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
      if (!d) {
        d = JSON.parse(localStorage.getItem(V2_DRAFT_KEY) || "null");
        localStorage.removeItem(V2_DRAFT_KEY);
        if (d && d.kvDtype) onLegacyDraftPrecision();
        if (d) delete d.kvDtype;
      }
      if (d && d.baseModel) {
        setBaseModel(d.baseModel);
        if (d.swaps && typeof d.swaps === "object") setSwaps(d.swaps);
        if (d.scopeAll && typeof d.scopeAll === "object") setScopeAll(d.scopeAll);
        if (d.groupDims && typeof d.groupDims === "object") setGroupDims(d.groupDims);
        if (d.dims && typeof d.dims === "object") setDims(d.dims);
        if (d.batch) setBatch(d.batch); if (d.seqLen) setSeqLen(d.seqLen);
        if (typeof d.precision === "string" && d.precision && !precisionFromLink && precision === AS_SHIPPED) {
          onDraftPrecision(d.precision);
        }
        if (Object.keys(d.swaps || {}).length || Object.keys(d.groupDims || {}).length) loadSkeleton(d.baseModel, true);
      }
    } catch { /* no draft */ }
    // restore once, on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadSkeleton]);
  useEffect(() => {
    localStorage.setItem(DRAFT_KEY, JSON.stringify({ baseModel, swaps, scopeAll, groupDims, dims, batch, seqLen, precision }));
  }, [baseModel, swaps, scopeAll, groupDims, dims, batch, seqLen, precision]);

  const groups = useMemo(() => skel?.groups ?? [], [skel]);
  const byId = useMemo(() => new Map(groups.map((g) => [g.group_id, g])), [groups]);

  // Set one slot; with the scope switch on, every slot bound to the same
  // component — each becomes its own explicit entry. Back to the base
  // component = no edit.
  const pick = useCallback((g: SkelGroup, s: SkelSlot, value: string) => {
    const targets = scopeAll[`${g.group_id}|${s.slot_id}`] ? s.bound_with.slots : [{ group_id: g.group_id, slot_id: s.slot_id }];
    setSwaps((prev) => {
      const next: Record<number, Record<string, string>> = { ...prev };
      for (const t of targets) {
        const slots = { ...(next[t.group_id] || {}) };
        if (value === s.component) delete slots[t.slot_id]; else slots[t.slot_id] = value;
        if (Object.keys(slots).length) next[t.group_id] = slots; else delete next[t.group_id];
      }
      return next;
    });
  }, [scopeAll]);

  // The scope switch expands (or withdraws) the slot's CURRENT value over
  // every slot bound to the same component, so a swap made before switching
  // reaches them too, and switching back leaves only this group's own swap.
  // Every peer's switch follows, so the table reads the same from any row.
  const scope = useCallback((g: SkelGroup, s: SkelSlot, all: boolean) => {
    const value = swaps[g.group_id]?.[s.slot_id] ?? s.component;
    setScopeAll((prev) => {
      const next = { ...prev };
      for (const t of s.bound_with.slots) next[`${t.group_id}|${t.slot_id}`] = all;
      return next;
    });
    setSwaps((prev) => {
      const next: Record<number, Record<string, string>> = { ...prev };
      for (const t of s.bound_with.slots) {
        if (t.group_id === g.group_id && t.slot_id === s.slot_id) continue;
        const slots = { ...(next[t.group_id] || {}) };
        if (all && value !== s.component) slots[t.slot_id] = value;
        else if (!all && slots[t.slot_id] === value) delete slots[t.slot_id];
        else if (all) delete slots[t.slot_id];
        if (Object.keys(slots).length) next[t.group_id] = slots; else delete next[t.group_id];
      }
      return next;
    });
  }, [swaps]);

  const requestGroups = useMemo(() => {
    const ids = new Set([...Object.keys(swaps), ...Object.keys(groupDims)].map(Number));
    return Array.from(ids).sort((a, b) => a - b).flatMap((id) => {
      const g = byId.get(id);
      if (!g) return [];
      const slots = swaps[id] || {};
      const gd = groupDims[id] || {};
      if (!Object.keys(slots).length && !Object.keys(gd).length) return [];
      return [{ group_id: id, template_name: g.template_name, count: g.count, assignment_index: g.assignment_index, slots, dims: gd }];
    });
  }, [swaps, groupDims, byId]);

  // Decision 6: the input is a Load draft. Compose and slot-diff bind to the
  // skeleton that is actually on screen (`skel.model_id`), not the typed id.
  const loadedModelId = skel?.success ? (skel.model_id ?? "") : "";
  const bodyJson = JSON.stringify({
    base_model_id: loadedModelId, groups: requestGroups, dim_overrides: { ...dims },
    batch_size: batch, seq_len: seqLen, ...precisionBody(precision), gpu_sku: gpuSku,
    kv_cache_lens: SWEEP_SEQ_LENS,
  });

  const runCompose = useCallback(async (sentJson: string) => {
    const seq = ++sendSeq.current;
    const myEpoch = epoch.current;
    lastPosted.current = sentJson;
    setComposeState("pending");
    setStatus("Composing…"); setStatusKind("idle");
    try {
      const r = await fetchJson<ComposeResp>(`${playgroundRoute.current}/compose-model`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: sentJson,
      });
      if (myEpoch !== epoch.current) return;
      setSentGroups(JSON.parse(sentJson).groups.map((g: { group_id: number }) => g.group_id));
      setRenderedSeq(seq);
      if (!r.success) {
        // the last composed numbers are not this edit's: drop them, so no
        // table shows an "edited" row for an edit that did not compose
        setResp(null);
        const n = r.errors?.length ?? 0;
        setStatus(n ? `Not composed — ${n} thing${n > 1 ? "s" : ""} to fix (shown on the groups).` : `Failed: ${r.message}`);
        setStatusKind("err"); setComposeState("error");
        setComposeErrors(r.errors ?? []);
        return;
      }
      if (!composeHasTargetState(r)) {
        setResp(null);
        setStatus("Failed: compose response is missing phases or kv_storage_by_cache_len");
        setStatusKind("err"); setComposeState("error");
        return;
      }
      setResp(r);
      setComposeErrors([]);
      setStatus("Done."); setStatusKind("ok"); setComposeState("done");
    } catch (e) {
      if (myEpoch !== epoch.current) return;
      setResp(null);
      setStatus(`Failed: ${e instanceof Error ? e.message : String(e)}`);
      setStatusKind("err"); setComposeState("error"); setRenderedSeq(seq);
    }
  }, []);

  useEffect(() => {
    if (!skel?.success) return;
    if (bodyJson === lastPosted.current) return;
    setComposeState("pending");
    const t = setTimeout(() => { void runCompose(bodyJson); }, 450);
    return () => { clearTimeout(t); epoch.current++; };
  }, [bodyJson, skel, runCompose]);

  const statusClass = statusKind === "ok" ? "text-sm text-green-600" : statusKind === "err" ? "text-sm text-red-500" : "text-sm text-gray-500";
  // a model the caller keeps and the agent writes: their own record, stored
  // without a checkpoint (a described model, not a generated Hub spec)
  const ownLoaded = !!skel?.success && skel.served_from === "user_generation" && !!skel.stored_without_checkpoint;
  // the editor's unsaved changes on a model the caller keeps, as words the
  // next agent turn carries (writing them into the spec is the agent's)
  const pendingEdits = [
    ...requestGroups.map((g) => {
      const label = byId.get(g.group_id)?.layers_label ?? `group ${g.group_id}`;
      const parts = [
        ...Object.entries(g.slots).map(([slot, comp]) => `${slot} → ${comp}`),
        ...Object.entries(g.dims).map(([k, v]) => `${k} = ${v}`),
      ];
      return `layers ${label}: ${parts.join(", ")}`;
    }),
    ...(Object.keys(dims).length ? [`every layer: ${Object.entries(dims).map(([k, v]) => `${k} = ${v}`).join(", ")}`] : []),
  ].join("; ");
  const errors = composeErrors.length ? composeErrors : (resp?.errors ?? []);
  const errorsOf = (id: number | null) => errors.filter((e) => e.group_id === id);
  const changesOf = (id: number) => resp?.groups?.find((g) => g.group_id === id)?.dim_changes;
  const edited = new Set(requestGroups.map((g) => g.group_id));
  // the strip: the edited regrouping once composed, the base's before
  const stripFromResp = resp?.success
    ? (resp.rows ?? []).filter((r) => r.edited && !r.aux_kind).map((r) => ({ group_id: r.group_id, layers: r.edited!.layers, label: r.edited!.layers_label }))
    : [];
  const stripGroups = stripFromResp.length
    ? stripFromResp
    : groups.filter((g) => g.editable && !g.aux_kind).map((g) => ({ group_id: g.group_id, layers: g.layers, label: g.layers_label }));
  const costOf = (id: number | null) => (resp?.success ? resp.rows?.find((r) => r.group_id === id) : undefined);
  const extraRows = resp?.success ? (resp.rows ?? []).filter((r) => r.group_id == null) : [];
  const bench = resp?.success ? editorAsBench(resp, loadedModelId, batch, seqLen) : null;
  const editorColorOf = (r: BenchRow) => PALETTE[r.entry === "edited" ? 1 : 0];

  return (
    <>
      <h1 className="text-2xl font-bold text-gray-900 mb-1">Playground — Model Editor</h1>
      <p className="text-sm text-gray-500 mb-4 max-w-3xl">
        Import a model — from the library, or described in words — swap the attention / FFN of any layer group and adjust dims, and see the
        <span className="font-medium"> whole-model</span> cost — the same numbers the Inspector shows for that model.
        Hybrid stacks are edited group by group; an MTP layer is a group of its own.
      </p>
      {oldDraftNotice && (
        <div id="pg-old-draft-notice" className="mb-4 text-sm text-gray-700 border-l-4 border-amber-400 bg-amber-50 pl-3 pr-2 py-2 flex items-start gap-2">
          <span className="flex-1">The editor now works on layer groups. A draft saved by the earlier editor is no longer read — load the model and redo the edit.</span>
          <button type="button" onClick={() => setOldDraftNotice(false)} className="text-gray-400 hover:text-gray-700 px-1" title="Dismiss">×</button>
        </div>
      )}

      <div className={drawerOpen ? "grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_24rem] gap-5 items-start" : ""}>
      <div className="min-w-0">
      <section className="card p-4 sm:p-6 mb-6">
        <div id="pg-source" className="inline-flex rounded-lg border border-gray-300 p-0.5 text-sm mb-4">
          {([["library", "From the library"], ["describe", "Describe in words"], ["mine", `My models (${myModels.length})`]] as const).map(([k, label]) => (
            <button key={k} type="button" data-source={k} onClick={() => { setSource(k); if (k === "describe") setDrawerOpen(true); }}
              className={`px-3 py-1 rounded-md ${source === k ? "bg-gray-800 text-white" : "text-gray-500 hover:text-gray-800"}`}>{label}</button>
          ))}
        </div>
        {source === "library" && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Base model</label>
            <input list="pg-models" value={baseModel} onChange={(e) => setBaseModel(e.target.value)}
              className="w-full text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white" />
          </div>
        )}
        {source === "mine" && (
          myModels.length
            ? <div className="flex flex-wrap gap-2">{myModels.map((m) => (
                <button key={m} type="button" onClick={() => { setBaseModel(m); void loadSkeleton(m); }}
                  className={`px-2.5 py-1 rounded-lg border text-sm font-mono ${loadedModelId === m ? "border-gray-800" : "border-gray-300 hover:border-gray-500"}`}>{m}</button>
              ))}</div>
            : <p className="text-sm text-gray-500">{signedIn ? "No models yet — describe one in words." : "Sign in to keep models of your own."}</p>
        )}
        {source === "describe" && (
          <DescribeForm models={models} signedIn={signedIn} busy={false}
            onStart={(name, description, base) => {
              setDrawerOpen(true);
              setFirstTurn({ name, description, base, seq: Date.now() });
            }} />
        )}
        <datalist id="pg-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          {source === "library" && (
            <button id="pg-load" type="button" onClick={() => loadSkeleton(baseModel)} disabled={!baseModel.trim()}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-800 text-white rounded-lg text-sm font-medium transition disabled:opacity-40 disabled:cursor-not-allowed">Load model</button>
          )}
          {ownLoaded && <span className="font-mono text-sm font-semibold">{loadedModelId}</span>}
          {skel?.success && skel.stored_without_checkpoint && (
              <span id="pg-hypothetical" className="text-xs px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200"
                title="Written from a description: structure and costs come from the spec, but there is no checkpoint to check the weights against.">
                Hypothetical · not checked against a checkpoint</span>
          )}
          {ownLoaded && !drawerOpen && <button type="button" onClick={() => setDrawerOpen(true)} className="text-xs text-gray-600 underline">Agent &amp; files</button>}
          <span className={statusClass}>{status}</span>
        </div>
      </section>

      {skel?.success && groups.length > 0 && (
        <>
          {bench && <JumpBar present={new Set(["pg-structure", "pg-summary",
            ...((bench.results[0]?.kv.storage_by_cache_len?.length ?? 0) > 0 ? ["pg-kv"] : [])])} />}
          <Section id="pg-structure" scope="FLOPs: one prefill forward" title={
            <>Layer groups <span className="text-sm font-normal text-gray-400">· {skel.n_layers} layers</span></>
          }>
            <div id="pg-editor-groups">
              <LayerStrip groups={stripGroups} />
              {errorsOf(null).length > 0 && (
                <div className="mt-2 space-y-1">{errorsOf(null).map((e, i) => <ErrorLine key={i} e={e} />)}</div>
              )}
              {composeState === "error" && errors.length === 0 && (
                <div className="pg-editor-error mt-2 text-xs text-red-700">{status}</div>
              )}
              <div className="overflow-x-auto mt-2 border border-gray-100 rounded min-w-0 @container" style={SCROLL_EDGE_SHADOWS} id="pg-group-table-box">
                <table id="pg-group-table" className={tbl.table}
                  data-compose-state={composeState === "idle" ? undefined : composeState}
                  data-compose-seq={renderedSeq || undefined}>
                  <thead className={tbl.thead}>
                    <tr>
                      <th className={tbl.headCell}>layers</th>
                      <th className={`${tbl.headCell} text-right`}>×</th>
                      <th className={tbl.headCell}>makeup</th>
                      <th className={tbl.headCell}>slots</th>
                      <th className={tbl.headCell}>group dims</th>
                      <th className={`pg-g-num ${tbl.headCell} text-right`}>params (group total)</th>
                      <th className={`pg-g-num ${tbl.headCell} text-right`}
                        title={"The declared structure's prefill FLOPs, not the Summary's served step: "
                          + "the declared walk computes logits at every position (a step at the sampled one), "
                          + "counts aux stacks such as MTP (a step the main stack only), "
                          + "and grafts no activation casts (a step counts them)."}>
                        FLOPs (group total, declared)</th>
                      <th className={`pg-g-num ${tbl.headCell} text-right`}>weight bytes (group total)</th>
                      <th className={`pg-g-num ${tbl.headCell} text-right`}>KV @L (group total)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayOrder(groups).map((g) => (
                      <GroupRows key={g.group_id} g={g} components={components}
                        swaps={swaps[g.group_id] || {}} scopeAll={scopeAll} edited={edited.has(g.group_id)}
                        onPick={pick}
                        onScope={(slot, all) => scope(g, slot, all)}
                        dimsOpen={!!openDims[g.group_id]}
                        onToggleDims={() => setOpenDims((o) => ({ ...o, [g.group_id]: !o[g.group_id] }))}
                        dims={groupDims[g.group_id] || {}}
                        onDims={(next) => setGroupDims((d) => {
                          const out = { ...d };
                          if (Object.keys(next).length) out[g.group_id] = next; else delete out[g.group_id];
                          return out;
                        })}
                        changes={changesOf(g.group_id)} errors={errorsOf(g.group_id)}
                        causedBy={sentGroups.length && !sentGroups.includes(g.group_id) ? sentGroups : undefined}
                        cost={costOf(g.group_id)}
                        swap={{ baseModel: loadedModelId, batch, seqLen, precision, gpuSku }} />
                    ))}
                    {extraRows.map((r, i) => (
                      <tr key={`new-${i}`} data-group="" className="pg-result-row pg-row-changed bg-amber-50">
                        <td className={tbl.cell}>
                          <span className="inline-block w-2 h-2 rounded-sm mr-1 align-middle" style={{ background: "#9ca3af" }} />
                          new{r.aux_kind ? ` · ${r.aux_kind}` : ""}
                        </td>
                        <td className={`${tbl.cell} font-sans`} colSpan={2}>{(r.edited ?? r.base)!.layers_label}</td>
                        <td /><td />
                        {groupCostCells(r)}
                      </tr>
                    ))}
                    {resp?.success && resp.model_level && (
                      <tr className="pg-result-model-level text-gray-500">
                        <td className={`${tbl.cell} font-sans`}>model-level</td>
                        <td className={`${tbl.cell} font-sans`} colSpan={2}>embedding · head · final norm</td>
                        <td /><td />
                        <td className={`pg-g-num ${tbl.cell} text-right tabular-nums`} data-col="params" data-edited={resp.model_level.edited.totals.params}>{formatCount(resp.model_level.edited.totals.params)}</td>
                        <td className={`pg-g-num ${tbl.cell} text-right tabular-nums`} data-col="flops" data-edited={resp.model_level.edited.totals.flops}>{formatSI(resp.model_level.edited.totals.flops, "FLOPs")}</td>
                        <td className={`pg-g-num ${tbl.cell} text-right tabular-nums`} data-col="weight_bytes" data-edited={resp.model_level.edited.totals.storage_bytes}>{formatGiB(resp.model_level.edited.totals.storage_bytes)}</td>
                        <td className={`pg-g-num ${tbl.cell}`} />
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              <div className="mt-3">
                <label className="block text-xs font-medium text-gray-700 mb-1">global dim overrides — every group that does not rebind the dim itself</label>
                <DimOverridesEditor value={dims} onChange={setDims} />
              </div>
            </div>
          </Section>
          {bench && (
            <>
              <Section id="pg-summary" scope="both phases side by side" title="Summary" lead={
                <ServingFields batch={batch} seqLen={seqLen}
                  gpuSku={gpuSku} gpus={gpus} onGpuSku={onGpuSku}
                  onBatch={setBatch} onSeqLen={setSeqLen} />
              }>
                <SummaryTable
                  rows={bench.results.map((r) => ({
                    ...r, key: r.entry ?? r.component, label: r.entry ?? r.component,
                    color: PALETTE[r.entry === "edited" ? 1 : 0],
                  }))}
                  cols={bench.results.some((r) => r.timings?.prefill)
                    ? SUMMARY_COLS
                    : RESOURCE_SUMMARY_COLS}
                  footnotes={summaryFootnotes(bench.results[0], bench.results)}
                />
              </Section>
              {(bench.results[0]?.kv.storage_by_cache_len?.length ?? 0) > 0 && (
                <Section id="pg-kv" scope="independent of phase" title="KV" lead={
                  <ServingFields batch={batch} seqLen={seqLen}
                    gpuSku={gpuSku} gpus={gpus} onGpuSku={onGpuSku}
                    onBatch={setBatch} onSeqLen={setSeqLen} />
                }>
                  <KvSweep resp={bench} colorOf={editorColorOf} wholeModel />
                </Section>
              )}
            </>
          )}
        </>
      )}
      </div>
      {drawerOpen && (
        <DescribeDrawer modelId={ownLoaded ? loadedModelId : (firstTurn ? `custom:${firstTurn.name}` : null)}
          firstTurn={firstTurn} pendingEdits={ownLoaded ? pendingEdits : ""}
          onModel={(id) => { setBaseModel(id); setMyReload((n) => n + 1); void loadSkeleton(id); }}
          onClose={() => setDrawerOpen(false)} />
      )}
      </div>
    </>
  );
}

// One layer group: its row in the table, and — expanded — its dims panel.
function GroupRows({ g, components, swaps, scopeAll, edited, onPick, onScope, dimsOpen, onToggleDims, dims, onDims, changes, errors, causedBy, cost, swap }: {
  g: SkelGroup; components: ComponentInfo[]; swaps: Record<string, string>;
  scopeAll: Record<string, boolean>; edited: boolean;
  onPick: (g: SkelGroup, s: SkelSlot, value: string) => void;
  onScope: (slot: SkelSlot, all: boolean) => void;
  dimsOpen: boolean; onToggleDims: () => void;
  dims: Record<string, number>; onDims: (next: Record<string, number>) => void;
  changes?: Record<string, { global: number; group: number }>; errors: ComposeError[];
  causedBy?: number[];
  cost?: ComposeRow;
  swap: { baseModel: string; batch: number; seqLen: number; precision: string; gpuSku: string };
}) {
  const layersTitle = `layers ${g.layers.join(", ")}${Object.keys(g.selector).length ? `\nselected by ${selectorText(g.selector)}` : ""}\ntemplate ${g.template_name}`;
  const swapped = (g.slots || []).filter((s) => s.slot_id in swaps && swaps[s.slot_id] !== s.component);
  return (
    <>
      <tr data-group={g.group_id}
        className={`pg-group-row ${cost ? "pg-result-row" : ""} ${tbl.row} align-top ${edited || cost?.changed ? "bg-amber-50" : ""} ${cost?.changed ? "pg-row-changed" : ""} ${g.editable ? "" : "text-gray-400"}`}>
        <td className={`${tbl.cell} whitespace-nowrap`} title={layersTitle}>
          <span className="inline-block w-2.5 h-2.5 rounded-sm mr-1.5 align-middle"
            style={{ background: g.editable ? groupColor(g.group_id) : "#d1d5db" }} />
          {g.aux_kind === "mtp" && <span className="text-10 font-semibold text-violet-700 bg-violet-50 rounded px-1 mr-1">MTP</span>}
          <span className={g.editable ? "text-gray-800" : ""}>{g.layers_label}</span>
          {!g.editable && <span className="ml-1">(read-only)</span>}
        </td>
        <td className={`${tbl.cell} text-right font-mono tabular-nums`}>{g.count}</td>
        <td className={`${tbl.cell} whitespace-nowrap`}>{g.editable ? (g.composition || []).join(" · ") : <span className="font-mono">{g.template_name}</span>}</td>
        <td className={`pg-g-slots ${tbl.wrapCell}`}>
          {(g.slots || []).map((s) => {
            const key = `${g.group_id}|${s.slot_id}`;
            const all = !!scopeAll[key];
            return (
              <div key={s.slot_id} className="flex flex-wrap items-center gap-1.5 mb-1">
                <span className="text-gray-500 font-mono">{s.slot_id}</span>
                <select data-slot={s.slot_id} value={swaps[s.slot_id] ?? s.component}
                  onChange={(e) => onPick(g, s, e.target.value)}
                  className={`text-xs border rounded px-1.5 py-1 bg-white font-mono max-w-[16rem] ${s.slot_id in swaps ? "border-amber-400" : "border-gray-300"}`}>
                  {slotOptionGroups(components, s.candidates, s.component).map(([fam, names]) => (
                    <optgroup key={fam} label={fam}>
                      {names.map((name) => <option key={name} value={name}>{name}</option>)}
                    </optgroup>
                  ))}
                </select>
                {s.bound_with.groups > 1 && (
                  <label className="inline-flex items-center gap-1 text-11 text-gray-600 shrink-0"
                    title={`apply this swap only here, or to all ${s.bound_with.groups} groups (${s.bound_with.layers} layers) whose ${s.slot_id} is ${s.component}`}>
                    <input type="checkbox" data-scope={s.slot_id} checked={all}
                      onChange={(e) => onScope(s, e.target.checked)}
                      className="rounded border-gray-300" />
                    all {s.bound_with.groups} groups
                  </label>
                )}
              </div>
            );
          })}
          {(g.readonly || []).length > 0 && (
            <div className="text-11 text-gray-400" title="no library alternative — read-only">
              {(g.readonly || []).map((r) => r.node_id).join(" · ")} (read-only)
            </div>
          )}
        </td>
        <td className={`${tbl.cell} whitespace-nowrap`}>
          {g.editable && (
            <button type="button" onClick={onToggleDims} className="pg-dims-toggle text-xs text-gray-600 hover:text-gray-900">
              {dimsOpen ? "▾" : "▸"} {Object.keys(dims).length ? `${Object.keys(dims).length} edited` : "dims"}
            </button>
          )}
        </td>
        {cost ? groupCostCells(cost) : emptyCostCells()}
      </tr>
      {errors.length > 0 && (
        <tr data-group-errors={g.group_id}><td colSpan={GROUP_TABLE_COLS} className="px-2 pb-1.5">
          {/* Held to the visible width of the scrolling frame, so a long
              message wraps where the reader can see it, not past the edge. */}
          <div className="sticky left-2 max-w-[calc(100cqw-1rem)] space-y-1">{errors.map((e, i) => <ErrorLine key={i} e={e} causedBy={causedBy} />)}</div>
        </td></tr>
      )}
      {g.editable && (dimsOpen || (changes && Object.keys(changes).length > 0)) && (
        <tr data-group-dims={g.group_id}><td colSpan={GROUP_TABLE_COLS} className="px-2 pb-2 bg-gray-50">
          {dimsOpen
            ? <GroupDimsPanel group={g} value={dims} onChange={onDims} changes={changes} />
            : <DimChanges changes={changes!} />}
        </td></tr>
      )}
      {swapped.map((s) => (
        <tr key={`diff-${s.slot_id}`} data-group-diff={g.group_id}>
          <td colSpan={GROUP_TABLE_COLS} className="px-2 pb-2">
            <SlotDiff oldName={s.component} newName={swaps[s.slot_id]} dims={dims} calls={s.calls}
              baseModel={swap.baseModel} batch={swap.batch} seqLen={swap.seqLen}
              precision={swap.precision} gpuSku={swap.gpuSku} />
          </td>
        </tr>
      ))}
    </>
  );
}
