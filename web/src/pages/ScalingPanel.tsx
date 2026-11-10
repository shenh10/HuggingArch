import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { EChartsOption, EChartsType } from "echarts";
import { AXIS, HaChart, interactiveBits } from "../charts/HaChart";
import { ApiError, fetchJson, isAbort, stillComputingNote } from "../api";
import { ComputeProgress } from "../ProgressBar";
import { fetchSweepCached, isTheory, readSweepCache, solOnly } from "../sweepCache";
import { binaryUnit, fmtCompact, fmtSeqLabel, formatGiB } from "../format";
import { legendRows, useWidth } from "../charts/useWidth";
import { SPEC_PRESETS, SWEEP_BATCH_GRID, SWEEP_COLORS, SWEEP_OUTPUT_TOKENS, SWEEP_SEQ_LENS } from "../compare";
import { precisionBody } from "../precision";
import {
  PAR_SWEEP_MODES,
  parallelOf,
  type Parallel,
  type ParSweepMode,
} from "../parallel";
import { gpuOptionLabel, skuOf } from "../gpu";
import type { Calibration, GpuItem } from "./PlanCard";
import { FrontierChart, type FrontierPt, type FrontierSeries } from "./GalleryCharts";
import { coverageText, isComplete, phaseCoverage, stateCounts, type MeasuredCoverage, type MeasuredPhase } from "../measured";

/** Viridis re-sampled at 9 stops for the sweep's 9 contexts; paired with a
 *  distinct line symbol per series so adjacent stops stay distinguishable. */
const VIRIDIS = [
  "#440154", "#482475", "#414487", "#355f8d",
  "#2a788e", "#22a884", "#7ad151", "#bddf26", "#fde725",
];
const SYMBOLS = ["circle", "rect", "triangle", "diamond", "roundRect", "pin", "arrow"];

const MODE_LABEL: Record<ParSweepMode, string> = {
  dp_ep: "DP-EP",
  cp_ep: "CP-EP",
  tp_ep: "TP-EP",
  tp_tp: "TP",
};

/** Default of the Context × Batch GPU-count dropdown, independent of Plan;
 *  8 matches `AUDIT_PARALLEL` (EP8/DP8 on 8 GPUs). KV storage reads the same
 *  GPU / parallel / degree. */
const BATCH_PAR_DEGREE = 8;

const DEFAULT_SWEEP_GPU = "B200";

/** The panel's context dropdown and GPU-count ladder, as the backend declares
 *  them (`sweep_grid.SCALING_CONTEXTS` / `SCALING_DEGREES` via
 *  `/inference/defaults`) — the same values `warm --sweeps` pre-computes. */
type ScalingGrid = { contexts: number[]; degrees: number[] };

type Phase = MeasuredPhase & {
  t_sol_s?: number | null;
  flops_per_step?: number;
  mem_read_bytes?: number;
  arithmetic_intensity?: number;
  throughput?: number | { sol?: { per_gpu?: number }; measured?: { per_gpu?: number } };
};
type SweepEst = {
  max_batch_per_gpu?: number;
  avg_input_tokens?: number;
  avg_output_tokens?: number;
  /** The model's published context length; null when it publishes none. */
  max_context_tokens?: number | null;
  /** This point's context runs past it. */
  beyond_max_context?: boolean;
  kv_cache?: {
    /** One request's cache on one GPU of this deployment. */
    per_request_kv_bytes?: number;
    effective_storage_tokens_per_request?: number;
    shard_note?: string;
  };
  memory?: {
    total_bytes?: number;
    weight_bytes_per_gpu?: number;
    kv_budget_bytes?: number;
    weight_deficit_bytes?: number;
    fits?: boolean;
  };
  phases?: { prefill?: Phase; decode?: Phase };
  /** The deployment these numbers are for, as the backend resolved it. */
  parallel?: Partial<Parallel>;
};
type SweepResp = { results_by_batch?: Record<string, SweepEst[]>; results?: SweepEst[] };

type LineSeries = {
  key: string;
  /** Identity used when a click applies the point to Plan (GPU sku / ctx). */
  applyKey: string;
  label: string;
  color: string;
  symbol?: string;
  dashed?: boolean;
  /** `v` is NaN where this track has no value: a measured line breaks there.
   *  `cov` (on a SOL point) is the measured coverage where the measured
   *  point is not drawn, for the hover. */
  points: Array<{ x: number; v: number; b?: number | null; t?: number | null; cov?: string }>;
};

type ParPoint = {
  x: number;
  /** The deployment this point is (`ParallelConfig.to_dict`). */
  parallel?: Partial<Parallel>;
  /** Axes the layout splits fewer ways than asked; their other ranks hold copies. */
  warnings?: string[];
  feasible?: boolean;
  /** Backend's reason this card cannot serve at this degree (feasible false). */
  reason?: string | null;
  /** Backend's error when this layout could not be estimated at all. */
  error?: string;
  prefill_tps_per_gpu?: number | null;
  decode_tps_per_gpu?: number | null;
  prefill_tps_per_gpu_measured?: number | null;
  decode_tps_per_gpu_measured?: number | null;
  /** Hover facts beside a throughput point: the batch the phase ran at and
    * its step time (t follows the SOL/measured source toggle). */
  prefill_batch_per_gpu?: number | null;
  decode_batch_per_gpu?: number | null;
  prefill_t_sol_s?: number | null;
  decode_t_sol_s?: number | null;
  prefill_t_measured_s?: number | null;
  decode_t_measured_s?: number | null;
  /** Each phase's measured coverage: the measured rate and time are drawn
   *  only where it is complete. */
  prefill_measured_coverage?: MeasuredCoverage | null;
  decode_measured_coverage?: MeasuredCoverage | null;
  /** The request at this context runs past the model's published context. */
  beyond_max_context?: boolean;
  max_context_tokens?: number | null;
};
/** One point of the sweep's decode cloud: a layout of a mode on a card at
 *  one per-GPU decode batch (`parallel_sweep.decode_point`). */
type ParFrontierPt = FrontierPt & {
  gpu_sku: string; mode: ParSweepMode; x: number; concurrent_requests: number; parallel: Partial<Parallel>;
};
type ParGpuSeries = { gpu_sku?: string; points?: ParPoint[] };
/** What a decode point is held at or grouped by (`frontier.POINT_DIMS`). */
type CloudDim =
  | "gpu_sku" | "mode" | "num_gpus" | "context" | "batch_key"
  | "draft_tokens" | "acceptance" | "ep_load_factor";
type CloudValue = string | number;
/** A frontier view of the cloud (`/inference/parallel-sweep/frontier`). */
type CloudView = {
  series: { group: Partial<Record<CloudDim, CloudValue>>; frontier: ParFrontierPt[] }[];
  values: Record<CloudDim, CloudValue[]>;
};
// `unavailable` is set when EVERY point of this mode failed to lay out (e.g. CP
// over a recurrent mixer) — the backend
// collapses the series and carries the reason, so the UI says "not applicable",
// NOT the "weight over budget" message an empty series would otherwise trigger.
/** Degrees the layout refused (each with its reason), and the bound past
 *  which the sweep does not enumerate — two different answers. */
type ParLadder = {
  refused?: { x: number; reason: string }[];
  beyond_cap?: { cap: number } | null;
};
type ParModeBlob = { moe_enabled?: boolean; series?: ParGpuSeries[]; unavailable?: string; ladder?: ParLadder };
type ParResp = {
  moe_enabled?: boolean;
  by_mode?: Record<string, ParModeBlob>;
  mode?: string;
  series?: ParGpuSeries[];
};

type LegendSelected = Record<string, boolean>;

export type ScalingApply = { gpu?: string; seq?: number; batch?: number; par?: Parallel };

/** What the layout said about the degrees the chart does not draw: each
 *  refused degree with its reason, the degrees drawn that replicate, and the
 *  bound past which nothing was enumerated. */
function LadderNotes({ blob, unit }: { blob: ParModeBlob; unit: string }) {
  const refused = blob.ladder?.refused || [];
  const cap = blob.ladder?.beyond_cap;
  const replicated = new Map<number, string[]>();
  for (const s of blob.series || []) {
    for (const p of s.points || []) {
      if (p.warnings?.length) replicated.set(p.x, p.warnings);
    }
  }
  // A card none of whose points could be priced (it lists no peak for the
  // format a step computes in) draws no line: say why. The reason names the card.
  const unpriced = (blob.series || []).filter((s) =>
    (s.points || []).length > 0 && (s.points || []).every((p) => p.error));
  if (!refused.length && !cap && !replicated.size && !unpriced.length) return null;
  return (
    <div className="mt-2 text-11 text-gray-500 space-y-0.5">
      {unpriced.map((s) => (
        <div key={`unpriced-${s.gpu_sku}`} data-ladder="unpriced" data-gpu={s.gpu_sku}>
          {s.gpu_sku} not drawn: {s.points?.[0]?.error}
        </div>
      ))}
      {refused.map((r) => (
        <div key={`refused-${r.x}`} data-ladder="refused" data-x={r.x}>
          {unit}{r.x}: this deployment does not hold — {r.reason}
        </div>
      ))}
      {[...replicated.entries()].sort((a, b) => a[0] - b[0]).map(([x, ws]) => (
        <div key={`replicated-${x}`} data-ladder="replicated" data-x={x} className="text-amber-700">
          {unit}{x}: {ws.join(" ")}
        </div>
      ))}
      {cap ? (
        <div data-ladder="beyond-cap">Not swept past {cap.cap} GPUs (cap {cap.cap}).</div>
      ) : null}
    </div>
  );
}

/** One legend item per context / GPU. SOL and its measured twin share the
 *  name — same colour, solid vs dashed — so a click hides both. */
function legendNames(series: LineSeries[]): string[] {
  const names: string[] = [];
  const seen = new Set<string>();
  for (const s of series) {
    if (seen.has(s.label)) continue;
    seen.add(s.label);
    names.push(s.label);
  }
  return names;
}

function legendSelEqual(a: LegendSelected | undefined, b: LegendSelected): boolean {
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b)]);
  for (const k of keys) {
    if ((a?.[k] ?? true) !== (b[k] ?? true)) return false;
  }
  return true;
}

/** GPU counts the parallel sweep marked feasible on this SKU (max-decode
 *  batch > 0), within the backend's ladder. Empty when the sweep has not
 *  arrived, or this GPU has no series. */
function feasibleDegrees(blob: ParModeBlob | null | undefined, gpuSku: string, ladder: number[]): number[] {
  if (!blob?.series?.length) return [];
  const match = blob.series.find((s) => s.gpu_sku === gpuSku);
  if (!match) return [];
  const xs = new Set<number>();
  for (const p of match.points || []) {
    if (p.feasible && ladder.includes(p.x)) xs.add(p.x);
  }
  return [...xs].sort((a, b) => a - b);
}

/** The GPU counts to offer. `refused` is null while there is something to
 *  offer; when the parallel sweep says nothing fits on this SKU it is the
 *  backend's reason at the largest degree it tried. */
function degreeOptions(
  blob: ParModeBlob | null | undefined, gpuSku: string, ladder: number[],
): { options: number[]; refused: string | null } {
  const match = blob?.series?.find((s) => s.gpu_sku === gpuSku);
  if (!match) return { options: ladder, refused: null };
  const options = feasibleDegrees(blob, gpuSku, ladder);
  if (options.length) return { options, refused: null };
  const last = [...(match.points || [])].reverse().find((p) => p.reason || p.error);
  return { options, refused: last?.reason || last?.error || "" };
}

function snapDegree(current: number, allowed: number[]): number {
  if (allowed.includes(current)) return current;
  if (allowed.includes(BATCH_PAR_DEGREE)) return BATCH_PAR_DEGREE;
  let best = allowed[0] ?? current;
  for (const n of allowed) {
    if (Math.abs(n - current) < Math.abs(best - current)) best = n;
  }
  return best;
}

/** The charts' one count reading (`format.fmtCompact`): 1.5K, 120K, 3M. */
function fmtNum(v: number): string {
  return fmtCompact(v);
}

/** A step time for a tooltip: ms while that reads best, else seconds. */
function fmtT(s: number): string {
  const ms = s * 1000;
  if (ms < 1) return `${ms.toFixed(2)} ms`;
  if (ms < 1000) return `${ms.toFixed(ms < 100 ? 1 : 0)} ms`;
  return `${s.toFixed(2)} s`;
}

function throughputValue(est: SweepEst | undefined, phase: "prefill" | "decode", source: "sol" | "measured"): number {
  if (!est) return NaN;
  const p = est.phases?.[phase] || {};
  // A measured rate is all or nothing: none unless every op is measured.
  if (source === "measured" && !isComplete(phaseCoverage(p))) return NaN;
  const tp = p.throughput;
  if (tp && typeof tp === "object") {
    const src = source === "measured" ? tp.measured : tp.sol;
    const v = src?.per_gpu;
    return typeof v === "number" && v > 0 ? v : NaN;
  }
  return source === "sol" && typeof tp === "number" && tp > 0 ? tp : NaN;
}

/** Where a sweep's measured answer is, beside its SOL answer on screen:
 *  being read (`progress` while the server says how far it got), or
 *  (`failed`) why it did not load — the SOL curves stay either way. */
type MeasuredNote = { text: string; failed?: boolean };

function MeasuredCheck({
  id, checked, onChange, note,
}: {
  id: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  note?: MeasuredNote | null;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <label className="inline-flex items-center gap-1.5 text-sm text-gray-700 cursor-pointer whitespace-nowrap">
        <input
          id={id}
          type="checkbox"
          className="accent-gray-700"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        Measured
      </label>
      {note ? (
        <span
          data-measured-note
          className={`text-11 max-w-xs truncate ${note.failed ? "text-red-500" : "text-gray-400"}`}
          title={note.text}
        >
          {note.text}
        </span>
      ) : null}
    </span>
  );
}

/** The hover text for a measured point that is not drawn: its coverage,
 *  or "" when there is nothing to say (no measured value was asked for). */
function coverageNote(cov: MeasuredCoverage | null | undefined): string {
  const counts = stateCounts(cov);
  return `measured: — (${coverageText(cov)}${counts ? `; ${counts}` : ""})`;
}

/** Said under a chart pair when the measured track is on. */
function MeasuredRule() {
  return (
    <div data-measured-rule className="mt-1 text-11 text-gray-500">
      Measured (dashed) points are drawn only where every op of the step is measured;
      elsewhere the hover gives the coverage.
    </div>
  );
}

/** Whether the estimates at context index `si` run past the model's
 *  published context (the backend's `beyond_max_context`; the request is
 *  the same at every batch). */
function pastContext(by: Record<string, SweepEst[]>, si: number): boolean {
  return Object.values(by).some((rows) => rows?.[si]?.beyond_max_context);
}


/** The measured answer's note while it is read, and when it failed. */
const READING_MEASURED: MeasuredNote = { text: "reading measurements…" };
function measuredFailed(e: unknown): MeasuredNote {
  return { text: `measured unavailable: ${e instanceof Error ? e.message : String(e)}`, failed: true };
}

/** The model's published context length as the estimates carry it. */
function maxContextOf(rows: Array<SweepEst | ParPoint | undefined>): number | null {
  for (const r of rows) {
    if (typeof r?.max_context_tokens === "number") return r.max_context_tokens;
  }
  return null;
}

/** Contexts whose requests run past the model's published context: still
 *  drawn, and said under the chart as extrapolation, never silent. */
function ContextLimitNote({ limit, past }: { limit: number | null; past: number[] }) {
  if (!past.length) return null;
  const at = limit ? ` (${limit.toLocaleString("en-US")} tokens)` : "";
  return (
    <div data-context-limit className="mt-1 text-11 text-amber-700">
      {past.map((x) => fmtSeqLabel(x)).join(", ")} {past.length > 1 ? "are" : "is"} past the
      model's published max context{at}; those points extrapolate beyond what the model serves.
    </div>
  );
}

/** One line per context length (X = decode batch; prefill stays at one
 * request). SOL is always present; measured is an optional same-colour dashed
 * twin that reuses the SOL legend name. */
function batchSeries(
  by: Record<string, SweepEst[]>,
  showMeasured: boolean,
): { pre: LineSeries[]; dec: LineSeries[] } {
  const pre: LineSeries[] = [];
  const dec: LineSeries[] = [];
  SWEEP_SEQ_LENS.forEach((seqLen, si) => {
    const color = VIRIDIS[si % VIRIDIS.length];
    const symbol = SYMBOLS[si % SYMBOLS.length];
    const label = fmtSeqLabel(seqLen);
    const applyKey = String(seqLen);
    const push = (phase: "prefill" | "decode", into: LineSeries[]) => {
      const solPts: LineSeries["points"] = [];
      const measPts: LineSeries["points"] = [];
      for (const bp of SWEEP_BATCH_GRID) {
        const est = by[bp.key]?.[si];
        const B = bp.key === "max"
          ? (est?.max_batch_per_gpu || 0)
          : (bp.decode_batch_size || bp.prefill_batch_size || 1);
        if (B <= 0) continue;
        const sv = throughputValue(est, phase, "sol");
        if (!Number.isFinite(sv)) continue;
        // Beside every SOL point; NaN where not every op was measured (a
        // gap), and the SOL point's hover says the coverage instead.
        const mv = showMeasured ? throughputValue(est, phase, "measured") : NaN;
        solPts.push({
          x: B, v: sv,
          ...(showMeasured && !Number.isFinite(mv) ? { cov: coverageNote(phaseCoverage(est?.phases?.[phase])) } : {}),
        });
        if (showMeasured) measPts.push({ x: B, v: mv });
      }
      into.push({
        key: applyKey, applyKey, label, color, symbol, points: solPts,
      });
      if (showMeasured && measPts.some((p) => Number.isFinite(p.v))) {
        into.push({
          key: `${applyKey}:meas`, applyKey, label, color, symbol, dashed: true, points: measPts,
        });
      }
    };
    push("prefill", pre);
    push("decode", dec);
  });
  return { pre, dec };
}

function SweepChart({
  title,
  series,
  xTitle,
  xUnit,
  yTitle,
  valueUnit,
  onPoint,
  legendSel,
  onLegendSel,
}: {
  title: string;
  series: LineSeries[];
  xTitle: string;
  /** Tooltip X heading, e.g. "batch" or "EP" → "batch = 64". */
  xUnit?: string;
  yTitle?: string;
  /** Appended to tooltip values, e.g. " tok/s". */
  valueUnit?: string;
  onPoint?: (s: LineSeries, p: { x: number; v: number }) => void;
  /** Shared legend selection across a chart pair: toggling a series in one
   *  chart hides it in its sibling too. */
  legendSel?: LegendSelected;
  onLegendSel?: (sel: LegendSelected) => void;
}) {
  // The zr click handler below registers once per chart instance, but the
  // series / axis it must read change every render (mode switch, source
  // toggle, legend) — it reads this always-current ref, not its closure.
  const [box, width] = useWidth<HTMLDivElement>();
  const live = useRef<{
    series: LineSeries[];
    xs: number[];
    hidden?: Record<string, boolean>;
    onPoint?: (s: LineSeries, p: { x: number; v: number }) => void;
    skipPlotClick?: boolean;
  }>({
    series: [],
    xs: [],
  });
  const onInstance = useCallback((chart: EChartsType) => {
    const zr = chart.getZr();
    zr.on("click", (e) => {
      const { series: ss, xs: axis, hidden, onPoint: fn, skipPlotClick } = live.current;
      const [ox, oy] = [e.offsetX, e.offsetY];
      // Legend / toolbox / axis-margin clicks are not chart clicks.
      if (skipPlotClick || !fn || !axis.length || !chart.containPixel("grid", [ox, oy])) return;
      let best: { s: LineSeries; p: { x: number; v: number }; d: number } | null = null;
      for (const s of ss) {
        if (hidden?.[s.label] === false) continue; // legend-toggled-off is not clickable
        for (const p of s.points) {
          if (!Number.isFinite(p.v)) continue;
          const idx = axis.indexOf(p.x);
          if (idx < 0) continue;
          const pix = chart.convertToPixel({ gridIndex: 0 }, [idx, p.v]);
          if (!Array.isArray(pix)) continue;
          const d = (pix[0] - ox) ** 2 + (pix[1] - oy) ** 2;
          if (!best || d < best.d) best = { s, p, d };
        }
      }
      if (best) fn(best.s, best.p);
    });
    zr.on("mousemove", (e) => {
      if (!live.current.onPoint) return;
      const inside = chart.containPixel("grid", [e.offsetX, e.offsetY]);
      zr.setCursorStyle(inside ? "pointer" : "default");
    });
  }, []);
  const events = useMemo(() => {
    if (!onLegendSel) return undefined;
    return {
      legendselectchanged: (raw: unknown) => {
        const p = raw as { selected?: LegendSelected };
        if (!p.selected) return;
        live.current.skipPlotClick = true;
        const next = p.selected;
        if (legendSelEqual(live.current.hidden, next)) {
          window.setTimeout(() => { live.current.skipPlotClick = false; }, 0);
          return;
        }
        // setOption must not run inside ECharts' legend dispatch — React
        // flushing here rebuilds the chart (notMerge) mid-click and blows up.
        window.setTimeout(() => {
          live.current.skipPlotClick = false;
          onLegendSel(next);
        }, 0);
      },
    };
  }, [onLegendSel]);
  const pts = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.v) && p.x > 0);
  if (!pts.length) {
    return (
      <div className="flex-1 min-w-[280px]">
        <div className="text-xs font-medium text-gray-700 mb-1">{title}</div>
        <div className="text-xs text-gray-400 py-6 text-center">no curve</div>
      </div>
    );
  }
  // The sweep is a discrete set of configs (1, 2, 4, 8, …), so a category axis
  // labels every measured point — a log axis only labels the decades and the
  // points in between are unreadable.
  const xs = [...new Set(pts.map((p) => p.x))].sort((a, b) => a - b);
  live.current.series = series;
  live.current.xs = xs;
  live.current.hidden = legendSel;
  live.current.onPoint = onPoint;
  // The legend sits under the x-axis name, never across the top where it
  // ran into the toolbox and the y-axis name on a phone: the plot gives up
  // one row of height per row the legend wraps into at this width.
  const names = legendNames(series);
  const rows = legendRows(names, width, { iconPx: 16, gapPx: 14, charPx: 6.6 });
  const legendH = rows * 18;
  const height = 260 + legendH;
  const option: EChartsOption = {
    color: series.map((s) => s.color),
    ...interactiveBits({ name: `${title} vs ${xTitle}` }),
    grid: { left: 56, right: 20, top: 36, bottom: 44 + legendH },
    legend: {
      show: true,
      type: "plain",
      bottom: 0,
      left: "center",
      width: "92%",
      itemGap: 14,
      itemWidth: 16,
      itemHeight: 3,
      icon: "roundRect",
      textStyle: { fontSize: 11, color: "#6b7280", lineHeight: 14 },
      data: names,
      selected: legendSel,
    },
    tooltip: {
      trigger: "axis",
      formatter: (raw: unknown) => {
        const items = (Array.isArray(raw) ? raw : [raw]) as Array<{
          seriesName?: string;
          seriesIndex?: number;
          value?: (number | string)[];
          marker?: string;
        }>;
        if (!items.length) return "";
        const xv = items[0].value?.[0];
        const head = `<div style="font-weight:600;margin-bottom:4px">${xUnit ? `${xUnit} = ` : ""}${xv ?? ""}</div>`;
        const rows = items
          .filter((it) => it.value?.[1] != null)
          .map((it) => {
            const s = series[it.seriesIndex ?? -1];
            const v = Number(it.value?.[1]);
            const xv = Number(it.value?.[0]);
            const p = s?.points.find((q) => q.x === xv);
            const name = s
              ? (s.dashed ? `${s.label} · measured` : s.label)
              : (it.seriesName ?? "");
            const facts = p && p.b != null && p.t != null && p.t > 0
              ? ` <span style="color:#9ca3af">(B/GPU=${p.b} · t=${fmtT(p.t)})</span>`
              : "";
            const cov = p?.cov ? `<div style="color:#9ca3af;padding-left:14px">${p.cov}</div>` : "";
            return `<div>${it.marker ?? ""}${name}: <b>${fmtNum(v)}${valueUnit || ""}</b>${facts}</div>${cov}`;
          })
          .join("");
        return head + rows;
      },
    },
    xAxis: {
      type: "category",
      data: xs.map(String),
      boundaryGap: false,
      name: xTitle,
      nameLocation: "middle",
      nameGap: 22,
      ...AXIS,
      axisLabel: { ...AXIS.axisLabel, interval: 0 },
    },
    yAxis: {
      type: "value",
      name: yTitle,
      ...AXIS,
      axisLabel: { ...AXIS.axisLabel, formatter: (v: string | number) => fmtNum(Number(v)) },
    },
    series: series.map((s) => ({
      type: "line" as const,
      name: s.label,
      // A point with no value is a null: the line breaks there, it is not
      // bridged and not filled from another track.
      data: s.points.filter((p) => p.x > 0).map((p) => [String(p.x), Number.isFinite(p.v) ? p.v : null]),
      symbol: s.symbol || "circle",
      showSymbol: true,
      symbolSize: 5,
      connectNulls: false,
      lineStyle: { width: 2, type: s.dashed ? "dashed" : "solid", color: s.color },
      itemStyle: { color: s.color },
      emphasis: { focus: "series" as const },
    })),
  };
  // Clicks are handled at the zr level (nearest point anywhere in the grid),
  // not as series clicks — a 5px symbol is too small a target to discover.
  return (
    <div ref={box} className="flex-1 min-w-[280px]">
      <div className="text-xs font-medium text-gray-700 mb-1">{title}</div>
      <HaChart
        option={option}
        height={height}
        onEvents={events}
        onInstance={onInstance}
      />
    </div>
  );
}

/** What the parallel-sweep panel says instead of its curves: a wait, or
 *  (`failed`) why the sweep did not load. `progress` is how far the server's
 *  work got, while its 503s say so. */
type SweepStatus = { text: string; failed?: boolean; progress?: { done: number; total: number } };

function ParallelSection({
  ctx,
  onCtx,
  contexts,
  byMode,
  status,
  measuredNote,
  showMeasured,
  onShowMeasured,
  hideEp,
  onApply,
}: {
  ctx: number;
  onCtx: (n: number) => void;
  contexts: number[];
  byMode: Record<string, ParModeBlob> | null;
  status: SweepStatus | null;
  measuredNote: MeasuredNote | null;
  showMeasured: boolean;
  onShowMeasured: (v: boolean) => void;
  hideEp: boolean;
  onApply?: (p: ScalingApply) => void;
}) {
  const [mode, setMode] = useState<ParSweepMode>("dp_ep");
  const [legendSel, setLegendSel] = useState<LegendSelected>({});
  const [applied, setApplied] = useState("");

  const disabledReason = (m: ParSweepMode): string => {
    if (m !== "tp_tp") {
      const blob = byMode?.[m];
      if (hideEp || (blob != null && !blob.moe_enabled)) return "Not MoE — no EP axis";
      if (blob?.unavailable) return blob.unavailable;
    } else if (byMode?.[m]?.unavailable) {
      return byMode[m].unavailable as string;
    }
    return "";
  };

  // Keep the active tab on an applicable mode: a model/GPU switch can disable
  // the current one (non-MoE hides EP modes, an unshardable layout collapses
  // a mode), in which case fall through to the first enabled tab.
  useEffect(() => {
    if (!disabledReason(mode)) return;
    const next = PAR_SWEEP_MODES.find((m) => !disabledReason(m));
    if (next && next !== mode) setMode(next);
  }, [byMode, hideEp, mode]);

  const blob = byMode?.[mode] || null;

  let content: ReactNode;
  if (status?.failed) {
    content = <div className="text-xs text-red-500 text-center py-6">{status.text}</div>;
  } else if (!blob) {
    content = status
      ? <div className="py-6 flex justify-center text-xs text-gray-400"><ComputeProgress compact label={status.text} progress={status.progress} /></div>
      : <div className="text-xs text-red-500 text-center py-6">{MODE_LABEL[mode]}: sweep load failed</div>;
  } else if (blob.unavailable) {
    // Every layout of this mode was invalid for this model (e.g. CP over a
    // recurrent mixer) — say so, don't misreport it as OOM. The tab
    // is normally disabled in this state; this is the fallback panel.
    content = (
      <div className="text-xs text-gray-400 text-center py-6" title={blob.unavailable}>
        {MODE_LABEL[mode]} does not apply to this model.
      </div>
    );
  } else {
    const num = (a: unknown): number => (typeof a === "number" && a > 0 ? a : NaN);
    const build = (phase: "prefill" | "decode"): LineSeries[] =>
      (blob.series || []).flatMap((s, idx) => {
        const gpu = s.gpu_sku || `gpu-${idx}`;
        const solKey = `${phase}_tps_per_gpu` as keyof ParPoint;
        const measKey = `${phase}_tps_per_gpu_measured` as keyof ParPoint;
        const bKey = `${phase}_batch_per_gpu` as keyof ParPoint;
        const tSolKey = `${phase}_t_sol_s` as keyof ParPoint;
        const tMeasKey = `${phase}_t_measured_s` as keyof ParPoint;
        const covKey = `${phase}_measured_coverage` as const;
        const color = SWEEP_COLORS[idx % SWEEP_COLORS.length];
        // The measured track sits beside every SOL point; NaN where not
        // every op was measured, so its line breaks there and the SOL
        // point's hover gives the coverage.
        const fed = (s.points || []).filter((p) => p.feasible && num(p[solKey]) > 0);
        const complete = (p: ParPoint) => isComplete(p[covKey]);
        const ptsOf = (vKey: keyof ParPoint, tKey: keyof ParPoint, measured: boolean) =>
          fed.map((p) => ({
            x: p.x,
            v: measured && !complete(p) ? NaN : num(p[vKey]),
            b: p[bKey] as number | null,
            t: measured && !complete(p) ? null : p[tKey] as number | null,
            ...(!measured && showMeasured && !complete(p) ? { cov: coverageNote(p[covKey]) } : {}),
          }));
        const solPts = ptsOf(solKey, tSolKey, false);
        const out: LineSeries[] = [{
          key: gpu, applyKey: gpu, label: gpu, color, points: solPts,
        }];
        if (showMeasured) {
          const measPts = ptsOf(measKey, tMeasKey, true);
          if (measPts.some((p) => Number.isFinite(p.v))) {
            out.push({
              key: `${gpu}:meas`, applyKey: gpu, label: gpu,
              color, dashed: true, points: measPts,
            });
          }
        }
        return out;
      });
    const pre = build("prefill");
    const dec = build("decode");
    if (pre.every((s) => !s.points.length) && dec.every((s) => !s.points.length)) {
      // Still computing: the cells that landed so far fit nothing, the rest
      // may — not a verdict yet.
      content = status
        ? <div className="py-6 flex justify-center"><ComputeProgress compact label={status.text} progress={status.progress} /></div>
        : (
          <div className="text-xs text-amber-700 text-center py-8">
            No GPU × degree fits (weights over budget).
          </div>
        );
    } else {
      const xTitle = mode === "tp_tp" ? "TP degree (= #GPUs)" : "EP degree (= #GPUs)";
      const xUnit = mode === "tp_tp" ? "TP" : "EP";
      const click = (gpu: string, x: number) => {
        const point = blob.series?.find((s) => s.gpu_sku === gpu)?.points?.find((p) => p.x === x);
        if (point?.parallel) {
          setApplied(`Applied: ${gpu} · ${MODE_LABEL[mode]} ${x}`);
          onApply?.({ gpu, seq: ctx, par: parallelOf(point.parallel) });
        }
      };
      content = (
        <div className="flex gap-4 flex-wrap">
          <SweepChart
            title="Prefill"
            xTitle={xTitle}
            xUnit={xUnit}
            yTitle="tok/s / GPU"
            valueUnit=" tok/s"
            series={pre}
            onPoint={onApply ? (s, p) => click(s.applyKey, p.x) : undefined}
            legendSel={legendSel}
            onLegendSel={setLegendSel}
          />
          <SweepChart
            title="Decode"
            xTitle={xTitle}
            xUnit={xUnit}
            yTitle="tok/s / GPU"
            valueUnit=" tok/s"
            series={dec}
            onPoint={onApply ? (s, p) => click(s.applyKey, p.x) : undefined}
            legendSel={legendSel}
            onLegendSel={setLegendSel}
          />
        </div>
      );
    }
    // One context for every point: if its request runs past the model's
    // published context, every point here extrapolates — said above them.
    const allPts = (blob.series || []).flatMap((s) => s.points || []);
    const past = allPts.some((p) => p.beyond_max_context);
    const limit = maxContextOf(allPts);
    const pastNote = past ? (
      <div data-context-limit className="mb-2 text-11 text-amber-700">
        ctx {fmtSeqLabel(ctx)} is past the model's published max context
        {limit ? ` (${limit.toLocaleString("en-US")} tokens)` : ""}; these throughputs
        extrapolate beyond what the model serves.
      </div>
    ) : null;
    content = <>{pastNote}{content}{showMeasured ? <MeasuredRule /> : null}<LadderNotes blob={blob} unit={mode === "tp_tp" ? "TP" : "EP"} /></>;
  }
  // Drawn from the cells that landed while the rest compute: a small note
  // in the corner says so, not a bar over the chart.
  if (blob && status && !status.failed) {
    content = (
      <>
        <div data-sweep-partial className="flex justify-end mb-1 text-11 text-gray-400">
          <ComputeProgress compact label="computing" progress={status.progress} unit="cells" />
        </div>
        {content}
      </>
    );
  }

  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 flex-wrap mb-3">
        <div className="text-sm font-medium text-gray-800">Parallelism</div>
        <label className="inline-flex items-center gap-1.5 text-xs text-gray-500">
          ctx
          <select
            id="infer-scaling-ctx"
            className="text-sm border border-gray-200 rounded-lg px-2 py-1.5 bg-white text-gray-800"
            disabled={!contexts.length}
            value={ctx}
            onChange={(e) => onCtx(Number(e.target.value))}
          >
            {contexts.map((c) => <option key={c} value={c}>{fmtSeqLabel(c)}</option>)}
          </select>
        </label>
        <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-0.5 text-xs">
          {PAR_SWEEP_MODES.map((m) => {
            const reason = disabledReason(m);
            const active = m === mode;
            return (
              <button
                key={m}
                id={`infer-scaling-mode-${m}`}
                type="button"
                disabled={!!reason}
                title={reason || undefined}
                onClick={() => setMode(m)}
                className={
                  "px-2.5 py-1 rounded-md transition-colors " +
                  (reason
                    ? "text-gray-300 cursor-not-allowed"
                    : active
                      ? "bg-white shadow-sm text-gray-900 font-medium"
                      : "text-gray-500 hover:text-gray-800")
                }
              >
                {MODE_LABEL[m]}
              </button>
            );
          })}
        </div>
        <MeasuredCheck id="infer-par-measured" checked={showMeasured} onChange={onShowMeasured}
          note={showMeasured ? measuredNote : null} />
        {applied ? <div className="text-11 text-green-600">{applied}</div> : null}
      </div>
      {content}
    </div>
  );
}

/** How a mode's frontier is drawn beside the other modes' of the same card. */
const MODE_LINE: Record<ParSweepMode, { lineType: "solid" | "dashed" | "dotted"; symbol: string }> = {
  dp_ep: { lineType: "solid", symbol: "circle" },
  cp_ep: { lineType: "dashed", symbol: "rect" },
  tp_ep: { lineType: "dotted", symbol: "triangle" },
  tp_tp: { lineType: "solid", symbol: "diamond" },
};

/** The dimensions a line of the frontier chart can be one value of, beside
 *  the card (a line never mixes cards); with none picked, each card's line
 *  is its best over every layout. */
const LINE_DIMS: CloudDim[] = ["mode", "num_gpus", "batch_key"];

const LINE_TYPES = ["solid", "dashed", "dotted"] as const;

/** One cloud dimension's value, as a label. */
function cloudLabel(d: CloudDim, v: CloudValue): string {
  if (d === "mode") return MODE_LABEL[v as ParSweepMode] ?? String(v);
  if (d === "num_gpus") return `${v} GPU${v === 1 ? "" : "s"}`;
  if (d === "batch_key") return v === "max" ? "max batch" : `batch ${String(v).replace(/^b/, "")}`;
  return String(v);
}

/** A dimension's multi-select: a dropdown of its values, each a checkbox;
 *  none ticked admits every value. */
function MultiPick({
  id, label, options, picked, text, onChange,
}: {
  id: string;
  label: string;
  options: CloudValue[];
  picked: CloudValue[];
  text: (v: CloudValue) => string;
  onChange: (next: CloudValue[]) => void;
}) {
  const summary = picked.length ? picked.map(text).join(", ") : "any";
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-gray-500">
      {label}
      <details id={id} className="relative">
        <summary className="list-none cursor-pointer text-xs border rounded px-1 py-0.5 bg-white text-gray-800 max-w-[12rem] truncate"
          title={summary}>
          {summary} ▾
        </summary>
        <div className="absolute z-20 mt-1 min-w-full bg-white border rounded shadow-sm py-1">
          {options.map((v) => {
            const on = picked.some((x) => x === v);
            return (
              <label key={String(v)} className="flex items-center gap-1.5 px-2 py-0.5 whitespace-nowrap cursor-pointer hover:bg-gray-50 text-gray-700">
                <input
                  type="checkbox"
                  className="accent-gray-700"
                  value={String(v)}
                  checked={on}
                  onChange={() => onChange(on ? picked.filter((x) => x !== v) : [...picked, v])}
                />
                {text(v)}
              </label>
            );
          })}
          {picked.length ? (
            <button type="button" className="block w-full text-left px-2 py-0.5 text-gray-400 hover:text-gray-700"
              onClick={() => onChange([])}>
              clear
            </button>
          ) : null}
        </div>
      </details>
    </span>
  );
}

/** Each hold control's label; the controls are the dimensions the view
 *  answers with, in its order. */
const HOLD_LABEL: Record<CloudDim, string> = {
  gpu_sku: "GPU",
  mode: "Mode",
  num_gpus: "GPUs",
  context: "Context",
  batch_key: "Batch",
  draft_tokens: "Draft tokens",
  acceptance: "Acceptance",
  ep_load_factor: "EP load",
};

/** Decode Pareto frontier at the sweep's context: tokens/s one request sees
 *  against tokens/s per GPU, a view of the sweep's decode point cloud (every
 *  layout at every per-GPU decode batch) the backend draws: the dimensions
 *  held, the ones each line is one value of, each line its front. A click
 *  applies the point. */
function ParetoSection({
  ctx,
  sweepBody,
  gpus,
  onApply,
}: {
  ctx: number;
  /** The parallel sweep's body, as posted: the view reads its cache entry. */
  sweepBody: Record<string, unknown>;
  gpus: string[];
  onApply?: (p: ScalingApply) => void;
}) {
  const [lineDims, setLineDims] = useState<CloudDim[]>([]);
  const [fix, setFix] = useState<Partial<Record<CloudDim, CloudValue[]>>>({});
  // The view with the request it answers: one drawn under other inputs
  // (another context, other holds) is not shown, so a click never applies a
  // point of the previous sweep.
  const [answer, setAnswer] = useState<{ key: string; view: CloudView } | null>(null);
  const [failed, setFailed] = useState("");
  const [applied, setApplied] = useState("");
  const by: CloudDim[] = ["gpu_sku", ...LINE_DIMS.filter((d) => lineDims.includes(d))];
  const sweepKey = JSON.stringify(sweepBody);
  // A new sweep holds nothing: its cloud may not have the held values.
  useEffect(() => { setFix({}); }, [sweepKey]);
  const reqBody = JSON.stringify({ sweep: sweepBody, fix, group_by: by });
  const view = answer?.key === reqBody ? answer.view : null;
  useEffect(() => {
    const ctrl = new AbortController();
    setFailed("");
    fetchJson<CloudView>("/inference/parallel-sweep/frontier", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: reqBody,
      signal: ctrl.signal,
    })
      .then((v) => setAnswer({ key: reqBody, view: v }))
      .catch((e) => {
        if (isAbort(e)) return;
        setFailed(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, [reqBody]);
  const colorOf = new Map(gpus.map((g, i) => [g, SWEEP_COLORS[i % SWEEP_COLORS.length]]));
  // Cards tell lines apart by colour; with one card on the chart, the
  // other dimension takes the colour instead.
  const cards = new Set((view?.series || []).map((s) => s.group.gpu_sku));
  const rest = by.filter((d) => d !== "gpu_sku");
  const restKeys = [...new Set((view?.series || []).map((s) => rest.map((d) => s.group[d]).join("·")))];
  const series: FrontierSeries<ParFrontierPt>[] = (view?.series || []).map((s, i) => {
    const restKey = rest.map((d) => s.group[d]).join("·");
    const ri = restKeys.indexOf(restKey);
    const gpu = String(s.group.gpu_sku ?? "");
    const mode = s.group.mode as ParSweepMode | undefined;
    return {
      key: `${gpu}:${restKey}:${i}`,
      label: by.map((d) => cloudLabel(d, s.group[d] ?? "")).join(" · "),
      color: cards.size > 1
        ? colorOf.get(gpu) || SWEEP_COLORS[0]
        : SWEEP_COLORS[ri % SWEEP_COLORS.length],
      points: s.frontier,
      ...(cards.size > 1 && rest.length
        ? (rest.length === 1 && mode ? MODE_LINE[mode] : { lineType: LINE_TYPES[ri % LINE_TYPES.length] })
        : {}),
    };
  });
  const pick = (_s: FrontierSeries<ParFrontierPt>, p: ParFrontierPt) => {
    setApplied(`Applied: ${p.gpu_sku} · ${p.layout} · batch ${p.batch}`);
    onApply?.({ gpu: p.gpu_sku, seq: ctx, batch: p.batch ?? undefined, par: parallelOf(p.parallel) });
  };
  // The controls stay while a new view loads: the last answer's dimensions.
  const dims = Object.keys((view || answer?.view)?.values || {}) as CloudDim[];
  const holds = dims.map((d) => ({ d, label: HOLD_LABEL[d] ?? d }));
  const held = holds.filter((h) => fix[h.d]?.length)
    .map((h) => (fix[h.d] as CloudValue[]).map((v) => cloudLabel(h.d, v)).join(", ")).join(" · ");
  const options = (d: CloudDim) => (view || answer?.view)?.values[d] || [];
  const specK = Number(sweepBody.speculative_draft_tokens) || 0;
  const specNote = specK > 0
    ? ` Speculative decoding, K = ${specK} at α = ${sweepBody.acceptance_rate}: ${sweepBody.speculative_draft
      ? "each step runs the K MTP draft forwards and the verify over K + 1 positions"
      : "each step is the verify over K + 1 positions alone (no draft forward on the GPU)"}, keeping 1 + α·K tokens per request.`
    : "";
  return (
    <FrontierChart<ParFrontierPt>
      bare
      title={`Decode Pareto frontier — ctx ${fmtSeqLabel(ctx)}${held ? ` · ${held}` : ""}`}
      subtitle={`Every layout of the Parallelism sweep at every per-GPU decode batch, kept to the held values; each line keeps the points no other point of it beats on both axes, up and to the right is better. Throughput is per GPU.${specNote} SOL only.${onApply ? " Click a point to apply it." : ""}`}
      controls={
        <div className="flex items-center gap-3 flex-wrap">
          {applied ? <span className="text-11 text-green-600">{applied}</span> : null}
          {holds.map(({ d, label }) => (
            <MultiPick
              key={d}
              id={`infer-pareto-fix-${d}`}
              label={label}
              options={options(d)}
              picked={fix[d] || []}
              text={(v) => cloudLabel(d, v)}
              onChange={(vs) => setFix((f) => {
                const next = { ...f };
                if (vs.length) next[d] = vs; else delete next[d];
                return next;
              })}
            />
          ))}
          <MultiPick
            id="infer-pareto-lines"
            label="Lines: GPU ×"
            options={LINE_DIMS}
            picked={lineDims}
            text={(d) => HOLD_LABEL[d as CloudDim]}
            onChange={(ds) => setLineDims(ds as CloudDim[])}
          />
        </div>
      }
      series={series}
      height={420}
      empty={failed ? `frontier load failed: ${failed}` : view ? "No layout serves these held values on these GPUs." : "Loading…"}
      onPoint={onApply ? pick : undefined}
    />
  );
}

/** Requests that fit on one GPU: the estimate's `max_batch_per_gpu`, the
 *  figure the Summary shows (static KV scales and every other reservation
 *  already taken out), or a gap when the estimate names none. */
function kvMaxRequests(est: SweepEst | undefined): number {
  const n = est?.max_batch_per_gpu;
  return typeof n === "number" && Number.isFinite(n) ? n : NaN;
}

function kvRequestHover(est: SweepEst | undefined, perReq: number | undefined): string {
  const w = est?.memory?.weight_bytes_per_gpu;
  const budget = est?.memory?.kv_budget_bytes;
  const parts: string[] = [];
  if (typeof w === "number") parts.push(`weights ${formatGiB(w)}`);
  if (est?.memory?.fits === false) {
    const short = est.memory.weight_deficit_bytes;
    parts.push(typeof short === "number" && short > 0
      ? `over HBM by ${formatGiB(short)}`
      : "weights over HBM");
  } else if (typeof budget === "number") {
    parts.push(`KV budget ${formatGiB(budget)}`);
  }
  if (typeof perReq === "number") parts.push(`request needs ${formatGiB(perReq)}`);
  return parts.join(" · ");
}

function sweepSeqResults(body: SweepResp): SweepEst[] | null {
  if (body.results?.length) return body.results;
  const by = body.results_by_batch;
  if (!by) return null;
  return by.max || by.b1 || Object.values(by)[0] || null;
}

type KvPoint = { x: number; v: number; extra?: string };

function KvChart({
  title,
  points,
  yTitle,
  yFormat,
  yScale = 1,
  tipFormat,
}: {
  title: string;
  points: KvPoint[];
  yTitle: string;
  /** Axis tick label of a plotted (scaled) value. */
  yFormat: (v: number) => string;
  /** Values are plotted divided by this, so the axis picks round ticks in
   *  the display unit. */
  yScale?: number;
  /** Hover value of a point's own (unscaled) value. */
  tipFormat?: (v: number) => string;
}) {
  const pts = points.filter((p) => Number.isFinite(p.v) && p.x > 0);
  if (!pts.length) {
    return (
      <div className="flex-1 min-w-[280px]">
        <div className="text-xs font-medium text-gray-700 mb-1">{title}</div>
        <div className="text-xs text-gray-400 py-6 text-center">no curve</div>
      </div>
    );
  }
  const xs = [...new Set(pts.map((p) => p.x))].sort((a, b) => a - b);
  const fmtTip = tipFormat || yFormat;
  const option: EChartsOption = {
    color: [SWEEP_COLORS[0]],
    ...interactiveBits({ name: title }),
    grid: { left: 64, right: 20, top: 28, bottom: 40 },
    tooltip: {
      trigger: "axis",
      formatter: (raw: unknown) => {
        const items = (Array.isArray(raw) ? raw : [raw]) as Array<{
          value?: (number | string)[];
          marker?: string;
        }>;
        if (!items.length) return "";
        const label = String(items[0].value?.[0] ?? "");
        const p = pts.find((q) => fmtSeqLabel(q.x) === label);
        if (!p) return "";
        const head = `<div style="font-weight:600;margin-bottom:4px">ctx = ${label}</div>`;
        const row = `<div>${items[0].marker ?? ""}${title}: <b>${fmtTip(p.v)}</b></div>`;
        const extra = p.extra
          ? `<div style="color:#6b7280;margin-top:2px">${p.extra}</div>`
          : "";
        return head + row + extra;
      },
    },
    xAxis: {
      type: "category",
      data: xs.map((x) => fmtSeqLabel(x)),
      boundaryGap: false,
      name: "ctx len",
      nameLocation: "middle",
      nameGap: 22,
      ...AXIS,
      axisLabel: { ...AXIS.axisLabel, interval: 0 },
    },
    yAxis: {
      type: "value",
      name: yTitle,
      ...AXIS,
      axisLabel: { ...AXIS.axisLabel, formatter: (v: string | number) => yFormat(Number(v)) },
    },
    series: [{
      type: "line" as const,
      name: title,
      data: pts.map((p) => [fmtSeqLabel(p.x), p.v / yScale]),
      showSymbol: true,
      symbolSize: 6,
      connectNulls: true,
      lineStyle: { width: 2 },
    }],
  };
  return (
    <div className="flex-1 min-w-[280px]">
      <div className="text-xs font-medium text-gray-700 mb-1">{title}</div>
      <HaChart option={option} height={260} />
    </div>
  );
}

/** What one request's cache point is: the positions it stores (the context
 *  plus the output tokens) and how the deployment divides it. */
function kvStoredHover(est: SweepEst | undefined): string {
  const parts: string[] = [];
  const stored = est?.kv_cache?.effective_storage_tokens_per_request;
  const inTok = est?.avg_input_tokens;
  const outTok = est?.avg_output_tokens;
  if (typeof stored === "number") {
    const req = typeof inTok === "number" && typeof outTok === "number"
      ? ` of ${inTok.toLocaleString("en-US")} context + ${outTok.toLocaleString("en-US")} output`
      : "";
    parts.push(`stores ${stored.toLocaleString("en-US")} positions${req}`);
  }
  if (est?.kv_cache?.shard_note) parts.push(est.kv_cache.shard_note);
  return parts.join(" · ");
}

function KvStorageSection({ rows }: { rows: SweepEst[] | null }) {
  const kvPts: KvPoint[] = [];
  const reqPts: KvPoint[] = [];
  const past: number[] = [];
  if (rows) {
    SWEEP_SEQ_LENS.forEach((seqLen, i) => {
      const est = rows[i];
      if (est?.beyond_max_context) past.push(seqLen);
      const perReq = est?.kv_cache?.per_request_kv_bytes;
      const maxB = kvMaxRequests(est);
      if (typeof perReq === "number" && Number.isFinite(perReq)) {
        kvPts.push({ x: seqLen, v: perReq, extra: kvStoredHover(est) });
      }
      if (Number.isFinite(maxB)) {
        reqPts.push({ x: seqLen, v: maxB, extra: kvRequestHover(est, perReq) });
      }
    });
  }
  // `per_request_kv_bytes` is one request's cache on ONE GPU of the
  // deployment above (the Plan's "Per-request KV / GPU"), not the whole
  // request's cache the Gallery ranks — so the chart says per GPU.
  const kvUnit = binaryUnit(Math.max(...kvPts.map((p) => p.v), 0));

  return (
    <div id="infer-scaling-kv" className="mt-6">
      <div className="text-sm font-medium text-gray-800 mb-3">KV storage</div>
      <div className="flex gap-4 flex-wrap">
        <KvChart
          title="KV per request, on one GPU"
          points={kvPts}
          yTitle="KV / request / GPU"
          yScale={kvUnit.scale}
          yFormat={(v) => (v === 0 ? "0" : `${fmtNum(v)} ${kvUnit.unit}`)}
          tipFormat={formatGiB}
        />
        <KvChart
          title="Max requests / GPU"
          points={reqPts}
          yTitle="max requests"
          yFormat={fmtNum}
        />
      </div>
      <ContextLimitNote limit={maxContextOf(rows || [])} past={past} />
    </div>
  );
}

export function ScalingPanel(props: {
  modelId: string;
  gpuSku: string;
  gpus: GpuItem[];
  /** The Inference page's precision preset id; every sweep carries it. */
  precision: string;
  frac: number;
  /** The two time floors (µs); null leaves them to the server's default. */
  commLatencyUs?: number | null;
  launchFloorUs?: number | null;
  outTok?: number;
  selectionMode?: string;
  framework?: string;
  /** The Inference page's paged-KV page choice, as a string; "" = not chosen.
   *  Inherited rather than chosen again here: the page is part of every
   *  cache-reading kernel's measurement key, so a sweep must read rows at the
   *  page the deployment on screen runs, and a second control could only make
   *  the two disagree. */
  kvBlock?: string;
  /** Whether the model routes tokens to experts (the estimate's `moe_enabled`). */
  moeEnabled?: boolean;
  /** ρ = max/mean expert load every swept deployment runs under; null when
   *  the typed value is invalid (the sweeps then wait, they do not assume 1). */
  epLoadFactor?: number | null;
  /** The caller's own measured rows, as the estimate reads them: signed in,
   *  every sweep asks the server to calibrate with them too; while not yet
   *  known the sweeps wait rather than post an uncalibrated sweep first. */
  calibration?: Calibration;
  /** A calibrated sweep answered 401: the session lapsed, read it again. */
  onSessionLapsed?: () => void;
  onApply?: (p: ScalingApply) => void;
}) {
  const [ovGpu, setOvGpu] = useState(DEFAULT_SWEEP_GPU);
  const [ovPar, setOvPar] = useState<ParSweepMode>("dp_ep");
  const [ovDegree, setOvDegree] = useState(BATCH_PAR_DEGREE);
  const [ctx, setCtx] = useState(4096);
  const [showBatchMeas, setShowBatchMeas] = useState(false);
  // Speculative decoding for every sweep of the panel (Parallelism, its
  // Pareto frontier, Context × Batch): a preset's K drafts verified per step
  // at acceptance α. "none" sends neither field, so the default bodies (and
  // their warm cache keys) are plain decode. `countDraft` adds the draft's
  // own forwards, priced off the model's MTP stack (`speculative_draft`);
  // off, the step is the verify alone — a drafter that runs no forward on the
  // GPU (n-gram / prompt lookup), the one form a model with no MTP stack has.
  const [specMode, setSpecMode] = useState("none");
  const [countDraft, setCountDraft] = useState(false);
  const [draftAvail, setDraftAvail] = useState<{ key: string; available: boolean; reason: string | null } | null>(null);
  const [showParMeas, setShowParMeas] = useState(false);
  const [status, setStatus] = useState<SweepStatus | null>({ text: "Pick a model and GPU." });
  // The (batch × context) sweep as the backend returned it. Measured only
  // chooses what to read out of it, so toggling it never re-posts.
  const [batchResults, setBatchResults] = useState<Record<string, SweepEst[]> | null>(null);
  const { pre, dec } = useMemo(
    () => (batchResults ? batchSeries(batchResults, showBatchMeas) : { pre: [], dec: [] }),
    [batchResults, showBatchMeas],
  );
  const kvRows = useMemo(
    () => (batchResults ? sweepSeqResults({ results_by_batch: batchResults }) : null),
    [batchResults],
  );
  const [batchLegend, setBatchLegend] = useState<LegendSelected>({});
  const [parStatus, setParStatus] = useState<SweepStatus | null>({ text: "Loading…" });
  const [batchMeasNote, setBatchMeasNote] = useState<MeasuredNote | null>(null);
  const [parMeasNote, setParMeasNote] = useState<MeasuredNote | null>(null);
  const [byMode, setByMode] = useState<Record<string, ParModeBlob> | null>(null);
  // The parallel sweep's last answer for this model, for the GPU counts the
  // Context × Batch sweep may take: kept while another context's parallel
  // sweep is computed, so that sweep never falls back to a count nobody chose.
  const [degreeAnswer, setDegreeAnswer] = useState<{ model: string; byMode: Record<string, ParModeBlob> } | null>(null);
  // The parallel sweep's body the charts show: the frontier view reads its
  // SOL answer's cache entry.
  const [parBody, setParBody] = useState<Record<string, unknown> | null>(null);
  const [appliedNote, setAppliedNote] = useState("");
  const seq = useRef(0);
  const parSeq = useRef(0);
  // The body each chart now shows: a re-render with the same body (another
  // prop of the page moved) keeps it, rather than asking again.
  const shownBody = useRef("");
  const shownParBody = useRef("");
  // The estimate's answer: the model routes tokens to experts.
  const hideEp = props.moeEnabled === false;
  // ρ is sent only when it can move a number: this model's estimate says it
  // routes and ρ > 1. At 1 the body (and so the warm cache key) is the
  // balanced one. A ρ > 1 whose model has no answer yet waits for it rather
  // than posting a balanced sweep and re-posting; an invalid ρ waits too.
  const rhoWanted = props.epLoadFactor ?? 1;
  const rho = props.moeEnabled === true && rhoWanted !== null && rhoWanted > 1 ? rhoWanted : undefined;
  const rhoHold = props.epLoadFactor === null
    ? "MoE load imbalance ρ is invalid — fix it in Plan Inference."
    : rhoWanted !== null && rhoWanted > 1 && props.moeEnabled === undefined
      ? "Waiting for this model's estimate (ρ is set; Estimate to apply it)."
      : "";
  // Calibrated with the caller's rows: the body says so. The in-tab cache
  // holds theory sweeps only (`sweepCache`): a measured sweep is always asked
  // of the server, so an upload elsewhere is never served from old curves.
  const calibrateMine = props.calibration?.state === "on";
  const spec = SPEC_PRESETS[specMode] ?? SPEC_PRESETS.none;
  // Whether this model can count a draft at the preset's K: the refusal the
  // estimate would raise (`/inference/speculative-draft`), asked per model and K.
  useEffect(() => {
    const id = props.modelId.trim();
    if (!id || spec.K <= 0) return;
    const ctrl = new AbortController();
    const key = `${id}|${spec.K}`;
    fetchJson<{ available?: boolean; unavailable_reason?: string | null }>(
      `/inference/speculative-draft?model_id=${encodeURIComponent(id)}`
        + `&speculative_draft_tokens=${encodeURIComponent(String(spec.K))}`,
      { signal: ctrl.signal },
    )
      .then((j) => setDraftAvail({ key, available: !!j.available, reason: j.unavailable_reason || null }))
      .catch((e) => {
        if (isAbort(e)) return;
        setDraftAvail({ key, available: false, reason: e instanceof Error ? e.message : String(e) });
      });
    return () => ctrl.abort();
  }, [props.modelId, spec.K]);
  // The answer for this model and K; null while it is asked (an answer for
  // another model or K is no answer here).
  const draft = draftAvail?.key === `${props.modelId.trim()}|${spec.K}` ? draftAvail : null;
  // The speculative fields every sweep body carries; none under plain decode.
  const specBody = useMemo(() => (spec.K > 0 ? {
    speculative_draft_tokens: spec.K,
    acceptance_rate: spec.alpha,
    ...(countDraft && draft?.available ? { speculative_draft: "recursive" } : {}),
  } : {}), [spec.K, spec.alpha, countDraft, draft]);
  // Asked to count the draft before this model's answer is known: the sweeps
  // wait, rather than post a verify-only sweep first and the drafted one after.
  const draftHold = spec.K > 0 && countDraft && !draft ? "Checking whether this model has a draft graph…" : "";
  const sweepHold = rhoHold || draftHold || (props.calibration?.state === "loading" ? "Reading your calibration rows…" : "");
  const lapsed = (e: unknown) => {
    if (calibrateMine && e instanceof ApiError && e.status === 401) props.onSessionLapsed?.();
  };
  const skuOpts = props.gpus.map(skuOf).filter(Boolean);
  const [grid, setGrid] = useState<ScalingGrid | null>(null);
  const [gridError, setGridError] = useState("");
  const degreeModes = degreeAnswer?.model === props.modelId.trim() ? degreeAnswer.byMode : null;
  const { options: batchDegrees, refused: noDegreeFits } = useMemo(
    () => degreeOptions(degreeModes?.[ovPar], ovGpu, grid?.degrees || []),
    [degreeModes, ovPar, ovGpu, grid],
  );
  // The GPU counts are not known yet: this model has no parallel answer, and
  // its parallel sweep can run (a GPU to run it on) and has not failed.
  const degreesPending = degreeModes === null && !!props.modelId.trim() && !!props.gpuSku
    && !parStatus?.failed;
  const batchDegree = snapDegree(ovDegree, batchDegrees);

  useEffect(() => {
    fetchJson<{ scaling_contexts?: number[]; scaling_degrees?: number[] }>("/inference/defaults")
      .then((d) => {
        if (Array.isArray(d.scaling_contexts) && Array.isArray(d.scaling_degrees)) {
          setGrid({ contexts: d.scaling_contexts, degrees: d.scaling_degrees });
        } else {
          setGridError("Scaling grid missing from /inference/defaults");
        }
      })
      .catch((e) => setGridError(`Failed to load the scaling grid: ${e instanceof Error ? e.message : String(e)}`));
  }, []);

  useEffect(() => {
    if (hideEp && ovPar !== "tp_tp") setOvPar("tp_tp");
  }, [hideEp, ovPar]);

  useEffect(() => {
    if (!skuOpts.length) return;
    if (!skuOpts.includes(ovGpu)) {
      setOvGpu(skuOpts.includes(DEFAULT_SWEEP_GPU) ? DEFAULT_SWEEP_GPU : skuOpts[0]);
    }
  }, [skuOpts, ovGpu]);

  useEffect(() => {
    // A measured answer still being read belongs to the body this run
    // replaces; this run says its own.
    setBatchMeasNote(null);
    const id = props.modelId.trim();
    const gpu = ovGpu;
    if (!id || !gpu) {
      { shownBody.current = ""; setBatchResults(null); }
      setStatus({ text: "Pick a model and GPU." });
      return;
    }
    if (sweepHold) {
      ++seq.current;
      setStatus({ text: sweepHold });
      return;
    }
    if (!grid) {
      // A held request still supersedes the one whose cleanup just aborted it.
      ++seq.current;
      setStatus({ text: gridError || "Drawing…" });
      return;
    }
    if (degreesPending) {
      // A sweep at a guessed GPU count is a sweep nobody chose: wait for the
      // counts this card serves.
      ++seq.current;
      setStatus({ text: "Waiting for the GPU counts this card serves…" });
      return;
    }
    if (noDegreeFits !== null) {
      // The parallel sweep found no GPU count on this card that serves the
      // model in this layout: say so, with its reason, and draw nothing.
      ++seq.current;
      { shownBody.current = ""; setBatchResults(null); }
      setStatus({ text: `No GPU count fits ${MODE_LABEL[ovPar]} on ${gpu}${noDegreeFits ? `: ${noDegreeFits}` : "."}` });
      return;
    }
    const body = {
      model_id: id,
      gpu_sku: gpu,
      seq_lens: SWEEP_SEQ_LENS,
      batch_grid: SWEEP_BATCH_GRID.map((p) => ({ ...p })),
      avg_output_tokens: SWEEP_OUTPUT_TOKENS,
      parallel_preset: { mode: ovPar, degree: batchDegree, ...(rho ? { ep_load_factor: rho } : {}) },
      ...precisionBody(props.precision),
      gpu_memory_fraction: props.frac,
      ...(props.commLatencyUs == null ? {} : { comm_latency_us: props.commLatencyUs }),
      ...(props.launchFloorUs == null ? {} : { launch_floor_us: props.launchFloorUs }),
      selection_mode: props.selectionMode || "envelope",
      ...(props.framework?.trim() ? { framework: props.framework.trim() } : {}),
      ...(props.kvBlock ? { kv_block_size: Number(props.kvBlock) } : {}),
      ...specBody,
      ...(calibrateMine ? { calibrate_with_mine: true } : {}),
    };
    // The measured answer is read only while it is drawn: unticked, the
    // charts read the SOL answer alone and nothing is asked of Kernel Bench.
    const wantMeasured = showBatchMeas && !isTheory(body);
    const bodyKey = JSON.stringify(body) + (wantMeasured ? "|measured" : "");
    if (bodyKey === shownBody.current) return;
    const path = "/inference/estimate-sweep";
    const sol = solOnly(body);
    const show = (resp: SweepResp): boolean => {
      const by = resp.results_by_batch;
      if (!by) return false;
      setBatchResults(by);
      setStatus(null);
      return true;
    };
    const missing = new Error("Sweep response missing results_by_batch");
    shownBody.current = "";
    const my = ++seq.current;
    const ctrl = new AbortController();
    // The SOL answer drawn, the measured answer is read live and drawn over
    // it; a measured answer that does not come leaves the SOL curves.
    const measure = () => {
      if (!wantMeasured) {
        shownBody.current = bodyKey;
        setBatchMeasNote(null);
        return;
      }
      setBatchMeasNote(READING_MEASURED);
      fetchSweepCached<SweepResp>(path, body, { signal: ctrl.signal },
        (info) => { if (my === seq.current) setBatchMeasNote({ text: stillComputingNote(info) }); })
        .then((resp) => {
          if (my !== seq.current) return;
          if (!show(resp)) {
            setBatchMeasNote(measuredFailed(missing));
            return;
          }
          shownBody.current = bodyKey;
          setBatchMeasNote(null);
        })
        .catch((e) => {
          if (my !== seq.current || isAbort(e)) return;
          lapsed(e);
          setBatchMeasNote(measuredFailed(e));
        });
    };
    const hit = readSweepCache<SweepResp>(path, sol);
    if (hit && show(hit)) {
      measure();
      return () => ctrl.abort();
    }
    setStatus({ text: "Drawing…" });
    setBatchMeasNote(null);
    fetchSweepCached<SweepResp>(path, sol, { signal: ctrl.signal },
      (info) => { if (my === seq.current) setStatus({ text: stillComputingNote(info), progress: info.progress }); })
      .then((resp) => {
        if (my !== seq.current) return;
        if (show(resp)) {
          measure();
          return;
        }
        { shownBody.current = ""; setBatchResults(null); }
        setStatus({ text: missing.message, failed: true });
      })
      .catch((e) => {
        if (my !== seq.current || isAbort(e)) return;
        { shownBody.current = ""; setBatchResults(null); }
        setStatus({ text: e instanceof Error ? e.message : String(e), failed: true });
      });
    return () => ctrl.abort();
  }, [props.modelId, ovGpu, ovPar, batchDegree, noDegreeFits, degreesPending, grid, gridError, props.precision, props.frac, props.commLatencyUs, props.launchFloorUs, props.selectionMode, props.framework, props.kvBlock, rho, sweepHold, specBody, calibrateMine, showBatchMeas]);

  useEffect(() => {
    setParMeasNote(null);
    const id = props.modelId.trim();
    if (!id || !props.gpuSku) {
      { shownParBody.current = ""; setByMode(null); }
      setParStatus(null);
      return;
    }
    if (sweepHold) {
      ++parSeq.current;
      setParStatus({ text: sweepHold });
      return;
    }
    const body = {
      model_id: id,
      gpu_sku: props.gpuSku,
      context_len: ctx,
      modes: [...PAR_SWEEP_MODES],
      ...precisionBody(props.precision),
      gpu_memory_fraction: props.frac,
      ...(props.commLatencyUs == null ? {} : { comm_latency_us: props.commLatencyUs }),
      ...(props.launchFloorUs == null ? {} : { launch_floor_us: props.launchFloorUs }),
      avg_output_tokens: props.outTok ?? SWEEP_OUTPUT_TOKENS,
      selection_mode: props.selectionMode || "envelope",
      ...(props.framework?.trim() ? { framework: props.framework.trim() } : {}),
      ...(props.kvBlock ? { kv_block_size: Number(props.kvBlock) } : {}),
      ...(rho ? { ep_load_factor: rho } : {}),
      ...specBody,
      ...(calibrateMine ? { calibrate_with_mine: true } : {}),
    };
    const wantMeasured = showParMeas && !isTheory(body);
    const bodyKey = JSON.stringify(body) + (wantMeasured ? "|measured" : "");
    if (bodyKey === shownParBody.current) return;
    const path = "/inference/estimate-parallel-sweep";
    const sol = solOnly(body);
    const show = (resp: ParResp) => {
      const mapped = resp.by_mode || (resp.mode ? { [resp.mode]: resp } : {});
      setByMode(mapped);
      setDegreeAnswer({ model: id, byMode: mapped });
      setParBody(body);
      setParStatus(null);
    };
    shownParBody.current = "";
    const my = ++parSeq.current;
    const ctrl = new AbortController();
    // As the batch sweep: the SOL answer, then the measured one over it.
    const measure = () => {
      if (!wantMeasured) {
        shownParBody.current = bodyKey;
        setParMeasNote(null);
        return;
      }
      setParMeasNote(READING_MEASURED);
      fetchSweepCached<ParResp>(path, body, { signal: ctrl.signal },
        (info) => { if (my === parSeq.current) setParMeasNote({ text: stillComputingNote(info) }); })
        .then((resp) => {
          if (my !== parSeq.current) return;
          show(resp);
          shownParBody.current = bodyKey;
          setParMeasNote(null);
        })
        .catch((e) => {
          if (my !== parSeq.current || isAbort(e)) return;
          lapsed(e);
          setParMeasNote(measuredFailed(e));
        });
    };
    const hit = readSweepCache<ParResp>(path, sol);
    if (hit) {
      show(hit);
      measure();
      return () => ctrl.abort();
    }
    // The curves on screen are the previous inputs'; drop them so a slow
    // sweep never shows them under the new model or GPU.
    { shownParBody.current = ""; setByMode(null); }
    setParStatus({ text: "Loading…" });
    setParMeasNote(null);
    fetchSweepCached<ParResp>(path, sol, { signal: ctrl.signal },
      (info) => {
        if (my !== parSeq.current) return;
        // The cells finished so far, assembled by the server: drawn now,
        // replaced by the whole answer when it lands.
        if (info.partial?.head) {
          const part = info.partial.head as ParResp;
          setByMode(part.by_mode || (part.mode ? { [part.mode]: part } : {}));
        }
        setParStatus({ text: stillComputingNote(info), progress: info.progress });
      })
      .then((resp) => {
        if (my !== parSeq.current) return;
        show(resp);
        measure();
      })
      .catch((e) => {
        if (my !== parSeq.current || isAbort(e)) return;
        { shownParBody.current = ""; setByMode(null); }
        setParStatus({ text: `sweep load failed: ${e instanceof Error ? e.message : String(e)}`, failed: true });
      });
    return () => ctrl.abort();
  }, [props.modelId, props.gpuSku, ctx, props.precision, props.frac, props.commLatencyUs, props.launchFloorUs, props.outTok, props.selectionMode, props.framework, props.kvBlock, rho, sweepHold, specBody, calibrateMine, showParMeas]);

  // The deployment the curves were drawn for, as the backend resolved it.
  const batchEst = batchResults ? Object.values(batchResults).flat().find((r) => r?.parallel) : undefined;
  const batchPar = batchEst?.parallel ? parallelOf(batchEst.parallel) : undefined;
  // Contexts whose requests run past the model's published context.
  const batchPast = batchResults
    ? SWEEP_SEQ_LENS.filter((_, si) => pastContext(batchResults, si))
    : [];
  const batchLimit = batchResults ? maxContextOf(Object.values(batchResults).flat()) : null;

  return (
    <section id="infer-scaling" className="card p-4 sm:p-6 mb-6">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
        <h2 className="text-lg font-semibold text-gray-900">Scaling</h2>
        <div className="flex items-center gap-3 flex-wrap">
          <label
            className="inline-flex items-center gap-1.5 text-xs text-gray-500"
            title="Decode, every sweep below: the target verifies K draft tokens in one forward over K + 1 positions and keeps 1 + α·K tokens per step."
          >
            Spec decode
            <select
              id="infer-scaling-spec"
              className="text-sm border border-gray-200 rounded-lg px-2 py-1.5 bg-white text-gray-800"
              value={specMode}
              onChange={(e) => setSpecMode(e.target.value)}
            >
              {Object.entries(SPEC_PRESETS).filter(([v]) => v !== "custom").map(([v, p]) => (
                <option key={v} value={v}>{p.label}</option>
              ))}
            </select>
          </label>
          {spec.K > 0 ? (
            <label
              className={`inline-flex items-center gap-1.5 text-xs ${draft?.available ? "text-gray-500" : "text-gray-400"}`}
              title={draft?.available
                ? "On: each step also runs the K draft forwards of the model's MTP stack (one token each). Off: the verify alone — a drafter that runs no forward on the GPU (n-gram / prompt lookup)."
                : `Only the verify is counted: ${draft?.reason ?? "checking whether this model has a draft graph…"}`}
            >
              <input
                id="infer-scaling-spec-draft"
                type="checkbox"
                className="accent-amber-500"
                disabled={!draft?.available}
                checked={!!draft?.available && countDraft}
                onChange={(e) => setCountDraft(e.target.checked)}
              />
              Count draft cost (MTP stack)
            </label>
          ) : null}
          {appliedNote ? <div className="text-xs text-green-600">{appliedNote}</div> : null}
        </div>
      </div>
      <ParallelSection
        ctx={ctx}
        onCtx={setCtx}
        contexts={grid?.contexts || []}
        byMode={byMode}
        status={parStatus}
        measuredNote={parMeasNote}
        showMeasured={showParMeas}
        onShowMeasured={setShowParMeas}
        hideEp={hideEp}
        onApply={props.onApply}
      />
      {byMode && parBody && !parStatus ? (
        <ParetoSection ctx={ctx} sweepBody={parBody || {}} onApply={props.onApply}
          gpus={(Object.values(byMode).find((m) => m.series?.length)?.series || []).map((s) => s.gpu_sku || "")} />
      ) : null}
      <div>
        <div className="flex items-center gap-3 flex-wrap mb-3">
          <div className="text-sm font-medium text-gray-800">Context × Batch</div>
          <label className="inline-flex items-center gap-1.5 text-xs text-gray-500 max-w-full min-w-0">
            GPU
            <select
              id="infer-scaling-gpu"
              className="min-w-0 max-w-full text-sm border border-gray-200 rounded-lg px-2 py-1.5 bg-white text-gray-800"
              value={skuOpts.includes(ovGpu) ? ovGpu : (skuOpts[0] || ovGpu)}
              onChange={(e) => setOvGpu(e.target.value)}
            >
              {skuOpts.length
                ? props.gpus.filter((g) => skuOf(g)).map((g) => <option key={skuOf(g)} value={skuOf(g)}>{gpuOptionLabel(g)}</option>)
                : <option value={ovGpu}>{ovGpu}</option>}
            </select>
          </label>
          <label className="inline-flex items-center gap-1.5 text-xs text-gray-500">
            Parallel
            <select
              id="infer-scaling-par"
              className="text-sm border border-gray-200 rounded-lg px-2 py-1.5 bg-white text-gray-800"
              value={ovPar}
              onChange={(e) => setOvPar(e.target.value as ParSweepMode)}
            >
              {PAR_SWEEP_MODES.filter((m) => m === "tp_tp" || !hideEp).map((m) => (
                <option key={m} value={m}>{MODE_LABEL[m]}</option>
              ))}
            </select>
          </label>
          <label className="inline-flex items-center gap-1.5 text-xs text-gray-500">
            GPUs
            <select
              id="infer-scaling-ngpu"
              className="text-sm border border-gray-200 rounded-lg px-2 py-1.5 bg-white text-gray-800"
              disabled={!grid}
              value={batchDegree}
              onChange={(e) => setOvDegree(Number(e.target.value))}
            >
              {batchDegrees.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <MeasuredCheck id="infer-scaling-measured" checked={showBatchMeas} onChange={setShowBatchMeas}
            note={showBatchMeas ? batchMeasNote : null} />
        </div>
        {status ? (
          // Reserve the batch row's height so the parallel section above keeps
          // its position while this (independent) sweep is still loading —
          // otherwise the two sweeps finishing out of order make the page jump.
          <div className="text-xs text-gray-400 text-center py-4 mb-2">
            {status.progress
              ? <ComputeProgress compact label={status.text} progress={status.progress} />
              : status.text}
          </div>
        ) : (
          <div className="flex gap-4 flex-wrap">
            <SweepChart
              title="Prefill (B=1)"
              series={pre}
              xTitle="decode batch"
              xUnit="batch"
              yTitle="tok/s"
              valueUnit=" tok/s"
              onPoint={props.onApply ? (s, p) => {
                setAppliedNote(`Applied: ctx ${fmtSeqLabel(Number(s.applyKey))} · batch ${p.x}`);
                props.onApply?.({
                  gpu: ovGpu || undefined,
                  seq: Number(s.applyKey) || undefined,
                  batch: p.x,
                  par: batchPar,
                });
              } : undefined}
              legendSel={batchLegend}
              onLegendSel={setBatchLegend}
            />
            <SweepChart
              title="Decode"
              series={dec}
              xTitle="decode batch"
              xUnit="batch"
              yTitle="tok/s"
              valueUnit=" tok/s"
              onPoint={props.onApply ? (s, p) => {
                setAppliedNote(`Applied: ctx ${fmtSeqLabel(Number(s.applyKey))} · batch ${p.x}`);
                props.onApply?.({
                  gpu: ovGpu || undefined,
                  seq: Number(s.applyKey) || undefined,
                  batch: p.x,
                  par: batchPar,
                });
              } : undefined}
              legendSel={batchLegend}
              onLegendSel={setBatchLegend}
            />
          </div>
        )}
        {!status ? <ContextLimitNote limit={batchLimit} past={batchPast} /> : null}
        {!status && showBatchMeas ? <MeasuredRule /> : null}
        {!status ? <KvStorageSection rows={kvRows} /> : null}
      </div>
    </section>
  );
}
