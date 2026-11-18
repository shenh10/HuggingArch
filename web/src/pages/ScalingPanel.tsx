import { useCallback, useEffect, useMemo, useRef, useState,
  type ChangeEvent, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from "react";
import type { EChartsOption, EChartsType } from "echarts";
import { AXIS, HaChart, interactiveBits } from "../charts/HaChart";
import { ApiError, fetchJson, isAbort, stillComputingNote } from "../api";
import { ComputeProgress } from "../ProgressBar";
import { fetchSweepCached, isTheory, readSweepCache, solOnly } from "../sweepCache";
import { binaryUnit, fmtCompact, fmtSeqLabel, formatGiB } from "../format";
import { legendRows, useWidth } from "../charts/useWidth";
import { DEFAULT_PREFILL_CHUNK_SIZE, SCALING_PANEL_BATCH_GRID, SPEC_PRESETS, SWEEP_COLORS, SWEEP_OUTPUT_TOKENS, SWEEP_SEQ_LENS } from "../compare";
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
import { coverageText, isComplete, stateCounts, type MeasuredCoverage, type MeasuredPhase } from "../measured";

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

/** The GPU-count ladder KV storage may be read at, as the backend declares
 *  it (`sweep_grid.SCALING_DEGREES` via `/inference/defaults`) — the same
 *  values `warm --sweeps` pre-computes. The contexts are not declared here:
 *  they are the ones the cloud priced, read off its own answer. */
type ScalingGrid = { degrees: number[] };

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
  points: Array<{
    x: number; v: number; b?: number | null; t?: number | null; cov?: string;
    /** A line of hover text naming what the point is (its layout, context). */
    note?: string;
    /** The cloud point a click applies. */
    src?: unknown;
  }>;
};

type ParPoint = {
  x: number;
  /** The context this point is priced at (the sweep prices every one). */
  context?: number;
  /** The deployment this point is (`ParallelConfig.to_dict`). */
  parallel?: Partial<Parallel>;
  /** Axes the layout splits fewer ways than asked; their other ranks hold copies. */
  warnings?: string[];
  feasible?: boolean;
  /** Backend's reason this card cannot serve at this degree (feasible false). */
  reason?: string | null;
  /** Backend's error when this layout could not be estimated at all. */
  error?: string;
  /** The request at this context runs past the model's published context. */
  beyond_max_context?: boolean;
  max_context_tokens?: number | null;
};
/** One point of the sweep's throughput cloud: a layout of a mode on a card
 *  at one context and per-GPU decode batch, carrying BOTH phases — the
 *  decode rung it is and the prefill of the same deployment, which runs one
 *  request whatever the rung (`point_cloud.parallel_records`). */
type ParFrontierPt = FrontierPt & {
  gpu_sku: string; mode: ParSweepMode; x: number; context: number; batch_key: string;
  concurrent_requests: number; parallel: Partial<Parallel>; beyond_max_context?: boolean;
  prefill_tok_s_per_user?: number | null;
  prefill_tok_s_per_gpu?: number | null;
  prefill_tok_s_per_gpu_measured?: number | null;
  prefill_batch?: number | null;
  prefill_t_s?: number | null;
  prefill_t_measured_s?: number | null;
  prefill_coverage?: MeasuredCoverage | null;
};
type ParGpuSeries = { gpu_sku?: string; points?: ParPoint[] };
/** What a decode point is held at or grouped by (`frontier.POINT_DIMS`). */
type CloudDim =
  | "gpu_sku" | "mode" | "num_gpus" | "context" | "batch_key"
  | "draft_tokens" | "acceptance" | "ep_load_factor";
type CloudValue = string | number;
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

/** One note of a ladder as the layout said it: which mode it is about and
 *  the degrees of that mode it holds for. */
type LadderRow = { mode: string; xs: number[]; lines: string[] };
/** A sentence said once, with every mode and degree it holds for. */
type LadderGroup = { modes: { mode: string; xs: number[] }[]; lines: string[] };

/** Notes whose sentence reads exactly the same are one row: the sentence is
 *  said once, with every mode and degree it holds for. Nothing here shortens
 *  or re-words the backend's text (§2.7) — rows group by equality alone. */
function byNote(rows: LadderRow[]): LadderGroup[] {
  const out = new Map<string, { modes: Map<string, number[]>; lines: string[] }>();
  for (const r of rows) {
    const key = r.lines.join("\u0000");
    let at = out.get(key);
    if (!at) {
      at = { modes: new Map(), lines: r.lines };
      out.set(key, at);
    }
    at.modes.set(r.mode, [...(at.modes.get(r.mode) || []), ...r.xs]);
  }
  return [...out.values()].map((g) => ({
    lines: g.lines,
    modes: [...g.modes.entries()].map(([mode, xs]) => ({
      mode, xs: [...new Set(xs)].sort((a, b) => a - b),
    })),
  }));
}

/** How a run of degrees reads: the ends of the ladder past two of them. */
function degreeRange(unit: string, xs: number[]): string {
  if (xs.length > 2) return `${unit}${xs[0]}–${unit}${xs[xs.length - 1]}`;
  return xs.map((x) => `${unit}${x}`).join(", ");
}

/** The axis a mode's ladder counts in: three of the four modes scale the
 *  expert axis, so a degree of theirs is an EP degree. */
function ladderUnit(mode: string): string {
  return mode === "tp_tp" ? "TP" : "EP";
}

/** A mode's degrees, named so a reader knows whose ladder refused: the mode
 *  then its degrees, except where the ladder's own unit already names the
 *  mode (TP's) and the prefix would read twice. */
function modeDegrees(mode: string, xs: number[]): string {
  const unit = ladderUnit(mode);
  const label = MODE_LABEL[mode as ParSweepMode] ?? mode;
  const ds = degreeRange(unit, xs);
  return label === unit ? ds : `${label} ${ds}`;
}

/** What the layout said about the deployments the cloud does not contain —
 *  for every parallel mode at once, since the cloud spans them all: a mode
 *  no layout of this model admits, each refused degree with its reason, the
 *  degrees drawn that replicate an axis, cards no point could be priced on,
 *  and the bound past which nothing was enumerated.
 *
 *  Collapsed by default to one line that counts them and names the degrees:
 *  the reasons are a paragraph apiece and a ladder refuses six of them at a
 *  time, which buried the charts under a wall of amber. The detail is one
 *  click away and unabridged. */
function LadderNotes({ byMode }: { byMode: Record<string, ParModeBlob> | null }) {
  const refusedRows: LadderRow[] = [];
  const replicatedRows: LadderRow[] = [];
  const unavailable: { mode: string; reason: string }[] = [];
  const unpriced: { gpu: string; error: string }[] = [];
  const caps = new Set<number>();
  for (const [mode, blob] of Object.entries(byMode || {})) {
    // An EP mode of a model the estimate says routes no tokens to experts is
    // not a deployment this model has — its ladder's notes are about an axis
    // that does not exist here, so they are not the reader's news.
    if (mode !== "tp_tp" && blob.moe_enabled === false) continue;
    if (blob.unavailable) unavailable.push({ mode, reason: blob.unavailable });
    for (const r of blob.ladder?.refused || []) {
      refusedRows.push({ mode, xs: [r.x], lines: [r.reason] });
    }
    const seen = new Map<number, string[]>();
    for (const s of blob.series || []) {
      for (const p of s.points || []) {
        if (p.warnings?.length) seen.set(p.x, p.warnings);
      }
      // A card none of whose points could be priced (it lists no peak for the
      // format a step computes in) draws no line: say why, naming the card.
      if ((s.points || []).length && (s.points || []).every((p) => p.error)) {
        unpriced.push({ gpu: s.gpu_sku || "", error: s.points?.[0]?.error || "" });
      }
    }
    for (const [x, lines] of seen) replicatedRows.push({ mode, xs: [x], lines });
    const cap = blob.ladder?.beyond_cap?.cap;
    if (cap) caps.add(cap);
  }
  const refused = byNote(refusedRows);
  const replicated = byNote(replicatedRows);
  // One card is said once however many modes could not price it.
  const cards = [...new Map(unpriced.map((u) => [`${u.gpu}\u0000${u.error}`, u])).values()];
  const counted = (n: number, what: string) => `${n} ${what}${n === 1 ? "" : "s"}`;
  const pairs = (groups: LadderGroup[]) =>
    groups.flatMap((g) => g.modes.flatMap((m) => m.xs)).length;
  const degrees = (groups: LadderGroup[], said: (n: number) => string) => {
    const n = pairs(groups);
    if (!n) return "";
    const per = new Map<string, number[]>();
    for (const g of groups) {
      for (const m of g.modes) per.set(m.mode, [...(per.get(m.mode) || []), ...m.xs]);
    }
    const named = [...per.entries()].map(([mode, xs]) =>
      modeDegrees(mode, [...new Set(xs)].sort((a, b) => a - b))).join(", ");
    return `${counted(n, "degree")} ${said(n)} (${named})`;
  };
  const headline = [
    unavailable.length
      ? `${counted(unavailable.length, "mode")} ${unavailable.length === 1 ? "does" : "do"}`
        + ` not apply (${unavailable.map((u) =>
          MODE_LABEL[u.mode as ParSweepMode] ?? u.mode).join(", ")})`
      : "",
    cards.length ? `${counted(cards.length, "card")} not drawn` : "",
    degrees(refused, () => "did not hold"),
    degrees(replicated, (n) => `replicate${n === 1 ? "s" : ""} an axis`),
  ].filter(Boolean).join(" · ");
  if (!headline && !caps.size) return null;
  return (
    <div className="mt-2 text-11 text-gray-500 space-y-0.5">
      {headline ? (
        <details data-ladder-notes>
          <summary className="cursor-pointer select-none marker:text-gray-300 hover:text-gray-700">
            {headline} — show why
          </summary>
          <div className="mt-1 space-y-1 pl-4">
            {unavailable.map((u) => (
              <div key={`unavailable-${u.mode}`} data-ladder="unavailable" data-mode={u.mode}>
                {MODE_LABEL[u.mode as ParSweepMode] ?? u.mode} does not apply to this
                model — {u.reason}
              </div>
            ))}
            {cards.map((u) => (
              <div key={`unpriced-${u.gpu}`} data-ladder="unpriced" data-gpu={u.gpu}>
                {u.gpu} not drawn: {u.error}
              </div>
            ))}
            {refused.map((g) => (
              <div key={`refused-${g.lines.join("")}`} data-ladder="refused"
                data-x={g.modes[0]?.xs[0]}
                data-xs={g.modes.flatMap((m) => m.xs).join(",")}
                data-modes={g.modes.map((m) => m.mode).join(",")}>
                {g.modes.map((m) => modeDegrees(m.mode, m.xs)).join(" · ")}: this deployment
                does not hold — {g.lines.join(" ")}
              </div>
            ))}
            {replicated.map((g) => (
              <div key={`replicated-${g.lines.join("")}`} data-ladder="replicated"
                data-x={g.modes[0]?.xs[0]}
                data-xs={g.modes.flatMap((m) => m.xs).join(",")}
                data-modes={g.modes.map((m) => m.mode).join(",")}
                className="text-amber-700">
                {g.modes.map((m) => modeDegrees(m.mode, m.xs)).join(" · ")}: {g.lines.join(" ")}
              </div>
            ))}
          </div>
        </details>
      ) : null}
      {[...caps].sort((a, b) => a - b).map((cap) => (
        <div key={`cap-${cap}`} data-ladder="beyond-cap">
          Not swept past {cap} GPUs (cap {cap}).
        </div>
      ))}
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

/** GPU counts the parallel sweep marked feasible on this SKU at the contexts
 *  on screen (max-decode batch > 0), within the backend's ladder — every
 *  context the sweep priced when the reader holds none. Empty when the sweep
 *  has not arrived, or this GPU has no series. */
function feasibleDegrees(blob: ParModeBlob | null | undefined, gpuSku: string, ladder: number[], ctxs: number[]): number[] {
  if (!blob?.series?.length) return [];
  const match = blob.series.find((s) => s.gpu_sku === gpuSku);
  if (!match) return [];
  const at = (p: ParPoint) => !ctxs.length || (p.context != null && ctxs.includes(p.context));
  const xs = new Set<number>();
  for (const p of match.points || []) {
    if (at(p) && p.feasible && ladder.includes(p.x)) xs.add(p.x);
  }
  return [...xs].sort((a, b) => a - b);
}

/** The GPU counts to offer. `refused` is null while there is something to
 *  offer; when the parallel sweep says nothing fits on this SKU it is the
 *  backend's reason at the largest degree it tried, and a card the sweep
 *  did not price is said as such — never the whole ladder offered blind. */
function degreeOptions(
  blob: ParModeBlob | null | undefined, gpuSku: string, ladder: number[], ctxs: number[],
): { options: number[]; refused: string | null } {
  const match = blob?.series?.find((s) => s.gpu_sku === gpuSku);
  if (!match) return { options: [], refused: `the parallel sweep did not price ${gpuSku}` };
  const options = feasibleDegrees(blob, gpuSku, ladder, ctxs);
  if (options.length) return { options, refused: null };
  const at = (p: ParPoint) => !ctxs.length || (p.context != null && ctxs.includes(p.context));
  const last = [...(match.points || [])].reverse().find((p) => at(p) && (p.reason || p.error));
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

/** Where a sweep's measured answer is, beside its SOL answer on screen:
 *  being read (`progress` while the server says how far it got), or
 *  (`failed`) why it did not load — the SOL curves stay either way. */
type MeasuredNote = { text: string; failed?: boolean };

function MeasuredCheck({
  id, checked, onChange, note, disabled,
}: {
  id: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  note?: MeasuredNote | null;
  /** Why a measured track cannot be read here; the box is off while it is
   *  said, so no chart claims a measurement nobody asked for. */
  disabled?: string;
}) {
  return (
    <span className="inline-flex items-center gap-2" title={disabled || undefined}>
      <label className={"inline-flex items-center gap-1.5 h-7 text-xs whitespace-nowrap "
        + (disabled ? "text-gray-400 cursor-default" : "text-gray-700 cursor-pointer")}>
        <input
          id={id}
          type="checkbox"
          className="accent-gray-700"
          disabled={!!disabled}
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

function SweepChart({
  title,
  series,
  xTitle,
  xUnit,
  xLabel = String,
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
  /** How an X value reads on the axis and in the hover (a context: "32K"). */
  xLabel?: (x: number) => string;
  yTitle?: string;
  /** Appended to tooltip values, e.g. " tok/s". */
  valueUnit?: string;
  onPoint?: (s: LineSeries, p: LineSeries["points"][number]) => void;
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
    onPoint?: (s: LineSeries, p: LineSeries["points"][number]) => void;
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
      let best: { s: LineSeries; p: LineSeries["points"][number]; d: number } | null = null;
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
  // A legend label's width at its 11px font runs past 6.6px a character
  // for capitals and digits ("H100-SXM · TP-EP"): an estimate short of the
  // drawn rows let the legend climb into the axis name.
  const rows = legendRows(names, width, { iconPx: 16, gapPx: 14, charPx: 7.4 });
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
        const head = `<div style="font-weight:600;margin-bottom:4px">${xUnit ? `${xUnit} = ` : ""}${xv == null ? "" : xLabel(Number(xv))}</div>`;
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
            const note = p?.note ? `<div style="color:#9ca3af;padding-left:14px">${p.note}</div>` : "";
            return `<div>${it.marker ?? ""}${name}: <b>${fmtNum(v)}${valueUnit || ""}</b>${facts}</div>${note}${cov}`;
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
      axisLabel: { ...AXIS.axisLabel, interval: 0, formatter: (v: string) => xLabel(Number(v)) },
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

/** What a sweep of this panel says instead of its curves, and which of
 *  three things the reader is looking at:
 *
 *  - `computing`: the server is doing this work now. `progress` is how far
 *    it got, while its 503s say so, and the cold-start sentence belongs
 *    here and nowhere else.
 *  - `waiting`: an input this sweep needs is still arriving. Nothing for
 *    the reader to do; an answer is coming.
 *  - `verdict`: there will be no answer until the reader acts (no model
 *    picked, no GPU count fits, the sweep failed).
 *
 *  A box whose answer is still coming draws a skeleton; a verdict is said
 *  in a line. Every status says which it is, so neither is guessed from
 *  its wording. */
type SweepStatus = {
  text: string;
  kind: "computing" | "waiting" | "verdict";
  failed?: boolean;
  progress?: { done: number; total: number };
};

/** What each sweep of the panel computes, said where the reader waits for
 *  it: this names the request the panel posted, it states no new fact. */
const CLOUD_WORK = "the deployment sweep — every card × context × parallel mode";
const KV_WORK = "the KV storage sweep — one request's cache at every context";

/** Why the first visit for a model is the slow one; said beside whichever
 *  sweep is computing, never as a figure the server has not given. */
const COLD_NOTE = "First visit for a model computes them all — nothing is cached"
  + " yet, so this takes minutes.";

/** The status of a sweep the server is computing, with however far it got. */
function computingStatus(work: string, progress?: { done: number; total: number }): SweepStatus {
  return { text: `Computing ${work}`, kind: "computing", ...(progress ? { progress } : {}) };
}

/** An answer is still coming — this sweep waits on an input, not on the
 *  reader. Its box draws the shape of the answer meanwhile. */
function waitingStatus(text: string): SweepStatus {
  return { text, kind: "waiting" };
}

/** There will be no answer until the reader acts; `failed` where a request
 *  is what did not answer. */
function verdictStatus(text: string, failed?: boolean): SweepStatus {
  return { text, kind: "verdict", ...(failed ? { failed: true } : {}) };
}

/** Whether what this box will hold is still on its way. */
function stillComing(status: SweepStatus | null): boolean {
  return !!status && status.kind !== "verdict";
}

/** The panel's one status line: whichever sweep is being waited on, the
 *  server's own count while it reports one, and the cold-start sentence
 *  while the work is the server's. One line for the whole panel — neither
 *  block prints its own copy of a sentence this one already says. */
function PanelStatus({ status }: { status: SweepStatus }) {
  return (
    <div
      data-panel-status
      className="mb-3 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border
        border-gray-100 bg-gray-50 px-2.5 py-1.5 text-11 text-gray-500"
    >
      {status.progress
        ? <ComputeProgress compact label={status.text} progress={status.progress} unit="cells" />
        : <span className={status.failed ? "text-red-500" : ""}>{status.text}</span>}
      {status.kind === "computing" ? <span className="text-gray-400">{COLD_NOTE}</span> : null}
    </div>
  );
}

/** The box a chart will occupy while its answer is computed: the axis frame
 *  it will have, three of its gridlines and a sheen across the plot area, at
 *  the height the chart itself is given — so the answer lands without moving
 *  the page, and nothing here reads as a number. */
function ChartSkeleton({ height }: { height: number }) {
  return (
    <div data-chart-skeleton aria-hidden className="pl-12 pr-4 pt-3 pb-8" style={{ height }}>
      <div className="relative h-full border-l border-b border-gray-200">
        {[25, 50, 75].map((p) => (
          <div key={p} className="absolute inset-x-0 border-t border-gray-100"
            style={{ top: `${p}%` }} />
        ))}
        <div className="skeleton absolute inset-x-2 bottom-1 top-6 rounded" />
      </div>
    </div>
  );
}

/** The controls' row while the cloud has not answered: the pickers it will
 *  hold are the cloud's own dimensions, so their boxes stand in for them
 *  rather than this page naming dimensions it has not been told about. */
function ControlsSkeleton() {
  return (
    <span className="flex items-center gap-1.5" data-controls-skeleton aria-hidden>
      {[0, 1, 2].map((i) => (
        <span key={i} className="skeleton h-3.5 w-20 rounded" />
      ))}
    </span>
  );
}

/** How a mode's frontier is drawn beside the other modes' of the same card. */
const MODE_LINE: Record<ParSweepMode, { lineType: "solid" | "dashed" | "dotted"; symbol: string }> = {
  dp_ep: { lineType: "solid", symbol: "circle" },
  cp_ep: { lineType: "dashed", symbol: "rect" },
  tp_ep: { lineType: "dotted", symbol: "triangle" },
  tp_tp: { lineType: "solid", symbol: "diamond" },
};

const LINE_TYPES = ["solid", "dashed", "dotted"] as const;

/** One cloud dimension's value, as a label. */
function cloudLabel(d: CloudDim, v: CloudValue): string {
  if (d === "mode") return v === "tp1" ? "1 GPU" : MODE_LABEL[v as ParSweepMode] ?? String(v);
  if (d === "num_gpus") return `${v} GPU${v === 1 ? "" : "s"}`;
  if (d === "batch_key") return v === "max" ? "max batch" : `batch ${String(v).replace(/^b/, "")}`;
  if (d === "context") return `ctx ${fmtSeqLabel(Number(v))}`;
  return String(v);
}

/** The one shell every control of these sections wears: a LINE OF TEXT —
 *  the control's name in grey, its value in near-black, one caret — with
 *  no box of its own, whether the control holds several values
 *  (`MultiPick`) or exactly one (`OnePick`). Five boxed controls read as a
 *  form to fill in; the same five as text read as a toolbar, and what a
 *  reader looks for (the values) is what carries the ink. The row's hairline
 *  dividers do the separating a border used to. */
const PICK_SHELL = "group inline-flex items-center gap-1 h-7 min-w-0 rounded"
  + " transition-colors";
const PICK_QUIET = "";
/** The trigger of a control that holds something: its value in amber, the
 *  only tint in the row, so what is held is what stands out. */
const PICK_HELD = "";
/** Open: a wash of grey behind the words (the ring is the padding a
 *  borderless control has none of), and the caret turned over. */
const PICK_OPEN = "bg-gray-50 ring-4 ring-gray-50";
const PICK_CAPTION = "text-xs text-gray-400 leading-none whitespace-nowrap";
const PICK_VALUE = "flex items-center gap-1 min-w-0 text-xs leading-tight font-medium"
  + " text-gray-900 group-hover:underline underline-offset-4 decoration-gray-300";
/** A row of these controls: they wrap, a hairline stands between two of
 *  them, and a popover hangs below one. */
const PICK_ROW = "flex items-center gap-x-3 gap-y-1 flex-wrap"
  + " [&>*+*]:border-l [&>*+*]:border-gray-200 [&>*+*]:pl-3";
/** The row's segmented switch (one of a few forms / modes), at the pickers'
 *  height: one choice at a time, so it is buttons and not a dropdown. */
const SEGMENTED = "inline-flex items-center h-7 rounded-lg border border-gray-200 bg-gray-50 p-0.5 text-xs";
const SEGMENT = "px-2.5 py-1 rounded-md transition-colors ";

/** The caret every control of the row shows (the native select's own is
 *  turned off, so one shape points one way). */
function Caret({ up = false }: { up?: boolean }) {
  return (
    <svg className={`w-3 h-3 shrink-0 text-gray-300 group-hover:text-gray-500 transition-transform ${up ? "rotate-180" : ""}`}
      viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M3 4.5 6 7.5 9 4.5" stroke="currentColor" strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

/** A single-valued dropdown in the multi-selects' shell: still a native
 *  <select> — these parameterise one estimate sweep, so one value is the
 *  whole answer — wearing the trigger the holds beside it wear. */
function OnePick({
  id, label, value, title, disabled, onChange, className = "", children,
}: {
  id: string;
  label: string;
  value: string | number;
  title?: string;
  disabled?: boolean;
  onChange: (e: ChangeEvent<HTMLSelectElement>) => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <label
      title={title}
      className={`${PICK_SHELL} ${PICK_QUIET} cursor-pointer max-w-full`
        + " focus-within:ring-2 focus-within:ring-amber-400/30 " + className}
    >
      <span className={PICK_CAPTION}>{label}:</span>
      <select
        id={id}
        // `field-sizing-content`: a native select is as wide as its widest
        // option, which in a text toolbar leaves the caret stranded a word
        // away from the value. Where the browser does not have it (Safari)
        // the control is merely wider, never broken.
        className="appearance-none field-sizing-content bg-transparent border-0 p-0 min-w-0
          max-w-full truncate text-xs leading-tight font-medium text-gray-900 cursor-pointer
          focus:outline-none group-hover:underline underline-offset-4 decoration-gray-300
          disabled:cursor-default disabled:no-underline disabled:text-gray-400"
        disabled={disabled}
        value={value}
        onChange={onChange}
      >
        {children}
      </select>
      <Caret />
    </label>
  );
}

/** The popover every multi-valued control of these sections hangs below
 *  its trigger: a `<details>` whose summary IS the trigger, closing on
 *  Escape or a click outside, and hanging from the trigger's right edge
 *  when it would otherwise run past the viewport (a phone, or the row's
 *  last control). */
function PickPopover({
  id, label, held, title, trigger, open, onOpen, onOpened, onKeyDown,
  panelClass = "", children,
}: {
  id: string;
  label: string;
  /** The trigger wears the held colour (amber) when it holds something. */
  held: boolean;
  title?: string;
  /** What the trigger reads, inside the shell. */
  trigger: ReactNode;
  open: boolean;
  onOpen: (v: boolean) => void;
  /** The panel, once it is on screen and measured: focus, scroll. */
  onOpened?: (panel: HTMLElement) => void;
  onKeyDown?: (e: ReactKeyboardEvent<HTMLElement>) => void;
  panelClass?: string;
  children: ReactNode;
}) {
  const [flip, setFlip] = useState(false);
  const root = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (!open) {
      setFlip(false);
      return;
    }
    // Dismissed on the CLICK outside, not its mousedown: closing this
    // popover can move the row under the pointer (the trigger's width
    // follows what it holds, so the controls after it shift), and a
    // control that moves between a press and its release never receives
    // the click at all — the first click outside would be spent on the
    // dismissal.
    const onOutside = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) onOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      onOpen(false);
      root.current?.querySelector<HTMLElement>("summary")?.focus();
    };
    document.addEventListener("click", onOutside);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onOutside);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onOpen]);
  // Measured on the way open: the panel's own width against the viewport.
  useEffect(() => {
    if (!open) return;
    const el = root.current;
    const panel = el?.querySelector<HTMLElement>("[data-pick-panel]");
    if (!el || !panel) return;
    setFlip(el.getBoundingClientRect().left + panel.offsetWidth > window.innerWidth - 8);
    onOpened?.(panel);
    // Once per opening: it focuses and scrolls, which a re-render must
    // not repeat under the reader's hands.
  }, [open]);
  return (
    <details
      id={id}
      ref={root}
      className="relative max-w-full"
      open={open}
      onKeyDown={onKeyDown}
    >
      <summary
        // Opened by this handler, not by the browser's own toggling: a
        // re-render between the native toggle and its (async) `toggle`
        // event — another popover closing on the same click, a view
        // landing — put the element's `open` back and swallowed the
        // click. One owner of the state, so that cannot happen.
        onClick={(e) => {
          e.preventDefault();
          onOpen(!open);
        }}
        aria-expanded={open}
        aria-haspopup="listbox"
        title={title}
        className={`${PICK_SHELL} list-none select-none cursor-pointer max-w-[15rem]`
          + " [&::-webkit-details-marker]:hidden"
          + " focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400/50 "
          + (open ? PICK_OPEN : held ? PICK_HELD : PICK_QUIET)}
      >
        {trigger}
        <Caret up={open} />
      </summary>
      {open ? (
        <div
          data-pick-panel
          role="group"
          aria-label={label}
          className={"absolute z-30 mt-1 w-max min-w-[11rem] max-w-[min(18rem,calc(100vw-2rem))]"
            + " rounded-xl border border-gray-200 bg-white shadow-lg "
            + (flip ? "right-0 " : "left-0 ") + (panelClass || "overflow-hidden")}
        >
          {children}
        </div>
      ) : null}
    </details>
  );
}

/** A dimension's multi-select: a popover of its values, each a checkbox;
 *  none ticked admits every value (no hold). The trigger says how many
 *  values are held — as chips with a × apiece — so several of them are
 *  plain without opening it; the popover filters when the dimension has
 *  many values (every GPU), and closes on Escape or a click outside. */
function MultiPick({
  id, label, options, picked, text, onChange, emptyText = "any",
}: {
  id: string;
  label: string;
  options: CloudValue[];
  picked: CloudValue[];
  text: (v: CloudValue) => string;
  onChange: (next: CloudValue[]) => void;
  /** What the trigger reads with nothing ticked; "any" (no hold) by default. */
  emptyText?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const search = useRef<HTMLInputElement>(null);
  useEffect(() => { if (!open) setQuery(""); }, [open]);
  const held = picked.length ? picked.map(text).join(", ") : emptyText;
  const q = query.trim().toLowerCase();
  const shown = q ? options.filter((v) => text(v).toLowerCase().includes(q)) : options;
  const chips = picked.slice(0, 2);
  const more = picked.length - chips.length;
  const toggle = (v: CloudValue, on: boolean) =>
    onChange(on ? picked.filter((x) => x !== v) : [...picked, v]);
  // Held values are the control's text, each with a faint × that drops it
  // — no chip, no box inside a box: the value IS the control's value, and
  // the amber says this one is held.
  const trigger = (
    <>
      <span className={PICK_CAPTION}>{label}:</span>
      <span className={PICK_VALUE}>
        {picked.length ? (
          <>
            {chips.map((v, i) => (
              <span key={String(v)} className="inline-flex items-center max-w-[8rem] text-amber-800">
                <span className="truncate">{text(v)}</span>
                <button
                  type="button"
                  tabIndex={-1}
                  aria-label={`Remove ${text(v)}`}
                  className="px-0.5 leading-none text-amber-300 hover:text-amber-700"
                  onClick={(e) => {
                    // A click inside a <summary> would toggle the popover.
                    e.preventDefault();
                    e.stopPropagation();
                    onChange(picked.filter((x) => x !== v));
                  }}
                >
                  ×
                </button>
                {i < chips.length - 1 || more > 0 ? "," : ""}
              </span>
            ))}
            {more > 0 ? <span className="text-amber-800">+{more}</span> : null}
          </>
        ) : <span className="font-normal text-gray-400 truncate">{emptyText}</span>}
      </span>
    </>
  );
  return (
    <PickPopover
      id={id}
      label={label}
      held={!!picked.length}
      title={`${label}: ${held}`}
      trigger={trigger}
      open={open}
      onOpen={setOpen}
      onOpened={(panel) => {
        if (options.length > 8) search.current?.focus();
        // A long list opens on what it holds, not on its first row.
        panel.querySelector<HTMLElement>("input:checked")
          ?.closest("label")?.scrollIntoView({ block: "nearest" });
      }}
      onKeyDown={(e) => {
        // The arrows walk the rows from wherever the focus is — the
        // trigger (which they open), the filter box, or another row.
        if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
        e.preventDefault();
        if (!open) {
          setOpen(true);
          return;
        }
        const boxes = [...(e.currentTarget.querySelectorAll<HTMLInputElement>(
          "[data-pick-panel] input[type=checkbox]") || [])];
        if (!boxes.length) return;
        const at = boxes.indexOf(document.activeElement as HTMLInputElement);
        const next = e.key === "ArrowDown"
          ? (at + 1) % boxes.length
          : (at <= 0 ? boxes.length - 1 : at - 1);
        boxes[next].focus();
      }}
    >
      {options.length > 8 ? (
        <div className="p-1.5 border-b border-gray-100">
          <input
            ref={search}
            type="text"
            aria-label={`Filter ${label}`}
            placeholder="Filter…"
            className="w-full px-2 py-1 text-11 border border-gray-200 rounded-md
              focus:outline-none focus:border-amber-400"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
      ) : null}
      {options.length > 3 ? (
        <div className="flex items-center justify-between gap-2 px-2.5 py-1.5
          border-b border-gray-100 text-10 text-gray-500">
          <span>{picked.length ? `${picked.length} of ${options.length} held` : emptyText}</span>
          <span className="flex items-center gap-1.5">
            <button type="button" data-pick-all className="hover:text-amber-700"
              onClick={() => onChange([...options])}>
              Select all
            </button>
            <span className="text-gray-200">|</span>
            <button type="button" data-pick-clear className="hover:text-amber-700"
              onClick={() => onChange([])}>
              Clear
            </button>
          </span>
        </div>
      ) : null}
      <div className="max-h-56 overflow-y-auto py-1">
        {shown.map((v) => {
          const on = picked.some((x) => x === v);
          return (
            <label
              key={String(v)}
              className={"flex items-center gap-2 px-2.5 py-1.5 text-11 cursor-pointer "
                + (on ? "bg-amber-50/70 text-amber-900" : "text-gray-700 hover:bg-gray-50")}
            >
              <input
                type="checkbox"
                className="w-3.5 h-3.5 shrink-0 accent-amber-500"
                value={String(v)}
                checked={on}
                onChange={() => toggle(v, on)}
              />
              <span className="truncate">{text(v)}</span>
            </label>
          );
        })}
        {!shown.length ? (
          <div className="px-2.5 py-2 text-11 text-gray-400">no match</div>
        ) : null}
      </div>
      {options.length <= 3 && picked.length ? (
        <button type="button" data-pick-clear
          className="block w-full border-t border-gray-100 px-2.5 py-1.5 text-left
            text-10 text-gray-500 hover:text-amber-700"
          onClick={() => onChange([])}>
          Clear (any)
        </button>
      ) : null}
    </PickPopover>
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

/** The dimensions this cloud answers with ONE value, said in a line under
 *  the row: there is nothing to pick, so they are no control — but a
 *  reader comparing two sweeps still has to know which value the cloud
 *  priced. A line, not a disclosure and not a dead "any" box: a dimension
 *  the cloud gives a CHOICE on is always a control of the row itself. */
function FixedNote({
  dims, values,
}: {
  /** The dimensions this cloud has one value for, in the cloud's order. */
  dims: CloudDim[];
  values: Record<string, CloudValue[]> | undefined;
}) {
  if (!dims.length) return null;
  return (
    <div data-fixed-note className="mt-1 text-10 leading-relaxed text-gray-400">
      One value in this cloud: {dims.map((d) =>
        `${HOLD_LABEL[d] ?? d} ${cloudLabel(d, (values?.[d] || [])[0])}`).join(" · ")}
    </div>
  );
}

/** The one chart area's two controls, orthogonal: which PHASE of the one
 *  cloud is drawn, and in which FORM. The form keys are the backend's own
 *  (`frontier.FORM_KEYS`): the Pareto front, or a line along one of its X
 *  axes — so a form is the request, not a second table translating one. */
type CloudPhase = "prefill" | "decode";
type FormKey = "frontier" | "num_gpus" | "batch" | "context";

const PHASES: { phase: CloudPhase; label: string }[] = [
  { phase: "prefill", label: "Prefill" },
  { phase: "decode", label: "Decode" },
];

const FORMS: { form: FormKey; label: string }[] = [
  { form: "frontier", label: "Pareto" },
  { form: "num_gpus", label: "vs GPUs" },
  { form: "context", label: "vs context" },
  { form: "batch", label: "vs batch" },
];

/** What a form asks the server for: the front needs no X, every other form
 *  is a line along the axis it is named after. */
function formView(form: FormKey): { form: "line" | "frontier"; x?: FormKey } {
  return form === "frontier" ? { form: "frontier" } : { form: "line", x: form };
}

type Holds = Partial<Record<CloudDim, CloudValue[]>>;

/** Either view's answer: per group its points (a front, or a line along X),
 *  each control's values, the dimensions the cloud declares — what a line
 *  may be one value of — the phase drawn with the fields its axes and
 *  coverage are on, and which (phase, form) pairs this cloud can answer at
 *  all, with the reason where it cannot (`frontier.view_forms`). */
type CloudView = {
  series: {
    group: Partial<Record<CloudDim, CloudValue>>;
    frontier?: ParFrontierPt[];
    points?: ParFrontierPt[];
  }[];
  values: Record<CloudDim, CloudValue[]>;
  dims: CloudDim[];
  phase: CloudPhase;
  axes: { sol: [string, string]; measured?: [string, string];
          partial?: [string, string]; coverage: string };
  forms: Record<CloudPhase, Partial<Record<FormKey, string | null>>>;
};

/** The GPU count the vs-batch and vs-context forms hold first: the KV
 *  storage default where it serves. */
const CLOUD_BATCH_DEGREE = BATCH_PAR_DEGREE;

/** What the cloud serves, read once per sweep: the max-batch lines per card
 *  × mode along the GPU count (`baseRequest`). Every form's defaults — the
 *  context among them, since this section's hold is the panel's one context
 *  control — are read off it, and this request is a function of the sweep
 *  alone, so no view's request depends on that view's own answer: a default
 *  read off the answer it shapes never settles. */
type CloudBase = { served: Map<string, Map<string, number[]>>; view: CloudView };

function baseRequest(sweepBody: Record<string, unknown>) {
  return {
    sweep: sweepBody, fix: { batch_key: ["max"] },
    group_by: ["gpu_sku", "mode"], form: "line", x: "num_gpus",
  };
}

function cloudBase(view: CloudView): CloudBase {
  const served = new Map<string, Map<string, number[]>>();
  for (const s of view.series) {
    const ns = (s.points || []).map((p) => p.num_gpus);
    if (!ns.length) continue;
    const card = String(s.group.gpu_sku);
    if (!served.has(card)) served.set(card, new Map());
    served.get(card)?.set(String(s.group.mode), ns);
  }
  return { served, view };
}

/** The context the panel opens on where the reader holds none: the routine
 *  4K request, which every card serves, else the shortest context the cloud
 *  priced. The hold IS the panel's context control, so this is the panel's
 *  default context — read off what the cloud serves, so a context this
 *  model was not priced at is never held. */
const DEFAULT_HOLD_CONTEXT = 4096;

function defaultContext(base: CloudBase): number | null {
  const cs = (base.view.values?.context || []).map(Number)
    .filter((n) => Number.isFinite(n)).sort((a, b) => a - b);
  if (!cs.length) return null;
  return cs.includes(DEFAULT_HOLD_CONTEXT) ? DEFAULT_HOLD_CONTEXT : cs[0];
}

/** A form's holds where the reader has set none, read off what the cloud
 *  serves (`CloudBase`), never off another section's pickers: a value the
 *  cloud does not have would draw an empty chart with a false reason.
 *  Pareto and vs GPUs hold the default context (Pareto also Plan's card,
 *  so one card's layouts stand against each other); the forms whose X is
 *  the context do not hold it, and pin a card, a mode and a GPU count that
 *  serve instead. */
function formDefaults(form: FormKey, ctx: number | null, planGpu: string,
                      base: CloudBase): Holds {
  const modeServing = (modes: Map<string, number[]> | undefined) =>
    PAR_SWEEP_MODES.find((m) => modes?.has(m)) ?? [...(modes?.keys() || [])][0];
  // The card a form holds when the reader has not picked one: Plan's, so
  // the section opens on the deployment the page is about.
  const planOr = () => (base.served.has(planGpu) ? planGpu : [...base.served.keys()][0]);
  // One card's layouts against each other — every mode its own line (the
  // lines default reads `cardsHeld === 1`), which is what the frontier is
  // asked for. Other cards are added back by clearing the GPU hold.
  const atCtx = ctx == null ? {} : { context: [ctx] };
  if (form === "frontier") {
    const gpu = planOr();
    return { ...atCtx, ...(gpu ? { gpu_sku: [gpu] } : {}) };
  }
  if (form === "num_gpus") {
    const any = new Map<string, number[]>();
    for (const modes of base.served.values()) for (const [m, ns] of modes) any.set(m, ns);
    const mode = modeServing(any);
    return { ...atCtx, batch_key: ["max"], ...(mode ? { mode: [mode] } : {}) };
  }
  const gpu = planOr();
  const modes = gpu ? base.served.get(gpu) : undefined;
  const mode = modeServing(modes);
  const ns = (mode && modes?.get(mode)) || [];
  const n = ns.includes(CLOUD_BATCH_DEGREE) ? CLOUD_BATCH_DEGREE : Math.min(...ns);
  return {
    ...(gpu ? { gpu_sku: [gpu] } : {}),
    ...(mode ? { mode: [mode] } : {}),
    ...(Number.isFinite(n) ? { num_gpus: [n] } : {}),
    // Along the context, one rung per card so the line is one deployment's
    // ceiling at each length rather than its whole ladder.
    ...(form === "context" ? { batch_key: ["max"] } : {}),
  };
}

/** Which CLOUD a sweep body is of: everything about it except which answer
 *  of that cloud was asked for — the contexts a measured read narrows to,
 *  and `sol_only`. Those two pick the answer, not the deployments it is
 *  about, so a reader's holds (and the controls they are set on) survive
 *  the measured answer landing on the very same points. */
function cloudOf(sweepBody: Record<string, unknown> | null): string {
  if (!sweepBody) return "";
  const rest = { ...sweepBody };
  delete rest.contexts;
  delete rest.sol_only;
  return JSON.stringify(rest);
}

async function postView(body: unknown, signal: AbortSignal): Promise<CloudView> {
  return fetchJson<CloudView>("/inference/parallel-sweep/frontier", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

/** How a (phase, form) reads under the chart: what it plots and why that
 *  pair of axes is the comparison. */
const SUBTITLE: Record<CloudPhase, Record<"frontier" | "line", string>> = {
  decode: {
    frontier: "Every layout of the parallel sweep at every per-GPU decode batch, kept to the held values; each line keeps the points no other point of it beats on both axes, up and to the right is better — faster for one request AND more tokens per GPU.",
    line: "Decode tokens/s per GPU, every layout of the parallel sweep kept to the held values; where several points of a line share an X, the line keeps the best (the hover names it). One GPU is the first point of every mode's line.",
  },
  prefill: {
    frontier: "Every layout of the parallel sweep, kept to the held values, at one request: how fast that request's prompt is read (its tokens over its time to first token) against what each GPU contributes. The knob along a line is the deployment — more GPUs cut the time to first token and cost per-GPU efficiency — so up and to the right is better, as decode's is.",
    line: "Prefill tokens/s per GPU at one request, every layout of the parallel sweep kept to the held values; where several points of a line share an X, the line keeps the best (the hover names it). One GPU is the first point of every mode's line.",
  },
};

/** What prefill here is priced under, said where it is read: this panel
 *  surveys deployments at the DEFAULT serving schedule and sends no
 *  `prefill_chunk_size`, so Plan's prefill-chunk box does not move these
 *  curves (and every reader shares one warm cache entry). The budget comes
 *  from the shared constant, so the sentence cannot drift from the body. */
const PREFILL_SCHEDULE_NOTE =
  ` Prefill is priced at the default chunked-prefill budget (${DEFAULT_PREFILL_CHUNK_SIZE.toLocaleString("en-US")} tokens per`
  + " forward), whatever Plan's prefill-chunk box says.";

const AXIS_TITLE: Record<CloudPhase, { x: string; y: string }> = {
  decode: { x: "Interactivity (tok/s per user)", y: "Output throughput (tok/s per GPU)" },
  prefill: { x: "Prompt read speed (tok/s per request)", y: "Prefill throughput (tok/s per GPU)" },
};

/** Throughput: EVERY chart of it — prefill and decode, front and lines — is
 *  a view of the parallel sweep's one point cloud (every layout from one
 *  GPU, on every card, at every context and per-GPU decode batch, each
 *  point carrying both phases). The phase and form switches pick the view;
 *  the holds and the lines are shared across both and survive a switch. A
 *  hold or the lines the reader has not set take the form's default, read
 *  off what the cloud serves (`formDefaults`). A (phase, form) pair the
 *  cloud cannot answer is disabled with the cloud's own reason
 *  (`view.forms`) — never a rule this page keeps. A click applies the point.
 *
 *  The Context hold is the panel's ONE context control: the sweep prices
 *  every routine context in one request, the holds pick which of them is on
 *  screen, and whoever else needs "the context the reader is looking at" —
 *  the measured answer, KV storage's GPU counts — reads it from here
 *  (`onHeldContexts`) rather than keeping a second picker.
 *
 *  While the sweep is still being computed the section keeps its shape: the
 *  title, the switches and the controls' row stand (disabled, since there is
 *  nothing yet to switch between), and the chart's box is a skeleton of the
 *  size the chart will be. */
function ThroughputSection({
  sweepBody,
  byMode,
  gpus,
  planGpu,
  showMeasured,
  onShowMeasured,
  measuredNote,
  onHeldContexts,
  onApply,
}: {
  /** The parallel sweep's body whose answer every view reads: the SOL one,
   *  or — with Measured ticked — the measured one the panel also posts, so
   *  the prefill track it folds is on these very points. Null until that
   *  sweep has answered: the section then draws its skeleton and asks for
   *  no view. */
  sweepBody: Record<string, unknown> | null;
  /** The sweep's per-mode answer, for what the layout refused (`LadderNotes`). */
  byMode: Record<string, ParModeBlob> | null;
  gpus: string[];
  /** Plan's card: the one a form holds first, if the cloud serves it. */
  planGpu: string;
  showMeasured: boolean;
  onShowMeasured: (v: boolean) => void;
  measuredNote: MeasuredNote | null;
  /** The contexts this section holds, for the panel's other readers of it. */
  onHeldContexts: (ctxs: number[]) => void;
  onApply?: (p: ScalingApply) => void;
}) {
  const [phase, setPhase] = useState<CloudPhase>("decode");
  const [form, setForm] = useState<FormKey>("frontier");
  // The reader's own holds and lines; a dimension absent here takes the
  // form's default.
  const [picked, setPicked] = useState<Holds>({});
  const [pickedLines, setPickedLines] = useState<CloudDim[] | null>(null);
  const [baseAnswer, setBaseAnswer] = useState<
    { key: string; cloud: string; base: CloudBase } | null>(null);
  const [answer, setAnswer] = useState<{ key: string; view: CloudView } | null>(null);
  const [failed, setFailed] = useState("");
  const [applied, setApplied] = useState("");
  // A new CLOUD holds nothing the reader picked: it may not have it. The
  // same cloud's measured answer is not a new cloud (`cloudOf`).
  const sweepKey = cloudOf(sweepBody);
  useEffect(() => { setPicked({}); setPickedLines(null); }, [sweepKey]);
  // What the cloud serves: asked once per sweep.
  const baseKey = sweepBody ? JSON.stringify(baseRequest(sweepBody)) : "";
  const fresh = baseKey && baseAnswer?.key === baseKey ? baseAnswer.base : null;
  // While the next probe of the SAME cloud is in flight the last answer
  // stands: ticking Measured asks for the same deployments with their
  // measured values folded in, and the controls — and what they hold —
  // must not empty under the reader's hand and come back. Only the REQUEST
  // below waits for the fresh answer; everything on screen reads this one.
  const base = fresh ?? (sweepKey && baseAnswer?.cloud === sweepKey ? baseAnswer.base : null);
  useEffect(() => {
    if (!baseKey) return;
    const ctrl = new AbortController();
    setFailed("");
    postView(JSON.parse(baseKey), ctrl.signal)
      .then((v) => setBaseAnswer({ key: baseKey, cloud: sweepKey, base: cloudBase(v) }))
      .catch((e) => {
        if (isAbort(e)) return;
        setFailed(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, [baseKey, sweepKey]);
  // Which (phase, form) pairs this cloud answers, as the cloud says. The
  // chosen form falls through to the first the phase can answer — decided
  // here, while the request is built, so a switch asks one view and not two.
  const refusals = base?.view.forms;
  const refusedFor = (f: FormKey): string => refusals?.[phase]?.[f] || "";
  const drawnForm: FormKey = !refusals || !refusedFor(form)
    ? form : (FORMS.map((f) => f.form).find((f) => !refusedFor(f)) ?? form);
  const fix: Holds = base ? formDefaults(drawnForm, defaultContext(base), planGpu, base) : {};
  for (const [d, vs] of Object.entries(picked) as [CloudDim, CloudValue[]][]) {
    if (vs.length) fix[d] = vs; else delete fix[d];
  }
  // The panel's one context answer: what this section holds. Published once
  // the cloud has answered — before that the hold is not a fact yet, and a
  // reader of it (the measured sweep, KV storage's GPU counts) waits.
  const heldCtxKey = base ? JSON.stringify((fix.context || []).map(Number)) : "";
  useEffect(() => {
    if (!heldCtxKey) return;
    onHeldContexts(JSON.parse(heldCtxKey) as number[]);
  }, [heldCtxKey, onHeldContexts]);
  // A measurement is keyed to the context it ran at, so the measured answer
  // is read at ONE context: the held one. Where the hold is not a single
  // context (the forms whose X is the context), Measured says so rather than
  // the panel inventing a context — the way a form the cloud cannot answer
  // disables itself.
  const measuredReason = !base
    ? "Waiting for the deployment sweep."
    : (fix.context?.length ?? 0) === 1
      ? ""
      : "Hold one context to read measured rows: a measurement is keyed to the"
        + " context it ran at.";
  const measuredOn = showMeasured && !measuredReason;
  // The values each control offers: the last answer's, so a control does
  // not empty while the next view is in flight; before the first answer,
  // what the cloud serves at this context (`base`), so the first paint is
  // not a row of empty boxes.
  const lastValues = (answer?.view ?? base?.view)?.values;
  const cardsHeld = fix.gpu_sku?.length ?? base?.served.size ?? gpus.length;
  const lineDims: CloudDim[] = pickedLines ?? (drawnForm === "batch"
    ? ["context"] : drawnForm === "frontier" && cardsHeld === 1 ? ["mode"] : []);
  // What a line may be one value of: the cloud's own dimensions beside the
  // card (a line never mixes cards), each with more than one value here.
  const cloudDims = base?.view.dims || [];
  const lineOptions = cloudDims.filter((d) => d !== "gpu_sku"
    && ((lastValues?.[d]?.length ?? 0) > 1 || lineDims.includes(d)));
  const by: CloudDim[] = ["gpu_sku", ...lineDims.filter((d) => d !== "gpu_sku")];
  // No view is asked before its defaults are known: the request is one
  // function of the sweep, the context, the phase, the form and the
  // reader's picks.
  const reqBody = fresh
    ? JSON.stringify({ sweep: sweepBody, phase, fix, group_by: by,
                       ...formView(drawnForm) }) : "";
  const view = reqBody && answer?.key === reqBody ? answer.view : null;
  useEffect(() => {
    if (!reqBody) return;
    const ctrl = new AbortController();
    setFailed("");
    postView(JSON.parse(reqBody), ctrl.signal)
      .then((v) => setAnswer({ key: reqBody, view: v }))
      .catch((e) => {
        if (isAbort(e)) return;
        setFailed(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, [reqBody]);
  const colorOf = new Map(gpus.map((g, i) => [g, SWEEP_COLORS[i % SWEEP_COLORS.length]]));
  // More cards than colours: each further round of the palette takes
  // another symbol (a dashed line is the measured track), so no two
  // cards' lines look alike.
  const roundOf = new Map(gpus.map((g, i) => [g, Math.floor(i / SWEEP_COLORS.length)]));
  // Each line's group and its points, whichever form drew them; a group of
  // absent points alone draws nothing and names no legend entry.
  const groups = (view?.series || [])
    .map((s) => ({ group: s.group, points: s.frontier ?? s.points ?? [] }))
    .filter((s) => s.points.length);
  // Cards tell lines apart by colour; with one card on the chart, the
  // other dimension takes the colour instead (contexts their own scale).
  const cards = new Set(groups.map((s) => s.group.gpu_sku));
  const rest = by.filter((d) => d !== "gpu_sku");
  const restKeys = [...new Set(groups.map((s) => rest.map((d) => s.group[d]).join("·")))];
  const style = (group: Partial<Record<CloudDim, CloudValue>>, i: number) => {
    const ri = restKeys.indexOf(rest.map((d) => group[d]).join("·"));
    const mode = group.mode as ParSweepMode | undefined;
    const ci = rest.length === 1 && rest[0] === "context"
      ? SWEEP_SEQ_LENS.indexOf(Number(group.context)) : -1;
    return {
      key: `${group.gpu_sku}:${rest.map((d) => group[d]).join("·")}:${i}`,
      label: by.map((d) => cloudLabel(d, group[d] ?? "")).join(" · "),
      color: cards.size > 1
        ? colorOf.get(String(group.gpu_sku ?? "")) || SWEEP_COLORS[0]
        : ci >= 0 ? VIRIDIS[ci % VIRIDIS.length] : SWEEP_COLORS[ri % SWEEP_COLORS.length],
      mode, ri,
    };
  };
  const apply = (p: ParFrontierPt) => {
    // The deployment and context the point is, at the batch the DRAWN phase
    // ran: a decode rung is the point's own batch, prefill is the one
    // request every prefill point of this cloud prices.
    const batch = (phase === "decode" ? p.batch : p.prefill_batch) ?? undefined;
    setApplied(`Applied: ${p.gpu_sku} · ${p.layout} · ctx ${fmtSeqLabel(p.context)} · batch ${batch ?? "—"}`);
    onApply?.({ gpu: p.gpu_sku, seq: p.context, batch, par: parallelOf(p.parallel) });
  };
  const dims = cloudDims;
  // The title says every held value, the context among them — this section's
  // holds are the whole of what is on screen.
  const held = dims.filter((d) => fix[d]?.length)
    .map((d) => (fix[d] as CloudValue[]).map((v) => cloudLabel(d, v)).join(", ")).join(" · ");
  const specK = Number(sweepBody?.speculative_draft_tokens) || 0;
  const specNote = phase === "decode" && specK > 0
    ? ` Speculative decoding, K = ${specK} at α = ${sweepBody?.acceptance_rate}: ${sweepBody?.speculative_draft
      ? "each step runs the K MTP draft forwards and the verify over K + 1 positions"
      : "each step is the verify over K + 1 positions alone (no draft forward on the GPU)"}, keeping 1 + α·K tokens per request.`
    : "";
  const drawn = groups.flatMap((s) => s.points);
  const past = drawn.some((p) => p.beyond_max_context);
  // Nothing to switch between until the cloud has answered: the switches
  // stand where they will be, greyed, rather than vanishing and shoving the
  // chart up when the answer lands.
  const pending = !base && !failed;
  const waiting = pending ? "Waiting for the deployment sweep." : "";
  const segment = (active: boolean, reason: string) => SEGMENT + (reason
    ? "text-gray-300 cursor-not-allowed"
    : active
      ? "bg-white shadow-sm text-gray-900 font-medium"
      : "text-gray-500 hover:text-gray-800");
  // Which dimensions are controls of the row — read off the cloud's own
  // answer (`values`), never a list this page keeps. Every dimension the
  // cloud gives a CHOICE on is a control, held or not: nothing a reader
  // can change sits behind a disclosure, because one click costs more
  // than the two boxes it saves. A dimension the cloud offers ONE value
  // for is nothing to pick (the speculative knobs of a plain decode
  // sweep): it is said in the line under the row, not rendered as a dead
  // "any" box. Let a sweep make those dimensions multi-valued and they
  // join the row by the same rule.
  const offered = (d: CloudDim) => lastValues?.[d]?.length ?? 0;
  const rowDims = dims.filter((d) => offered(d) > 1);
  const fixedDims = dims.filter((d) => offered(d) === 1);
  const hold = (d: CloudDim) => (
    <MultiPick
      key={d}
      id={`infer-throughput-fix-${d}`}
      label={HOLD_LABEL[d] ?? d}
      options={lastValues?.[d] || []}
      picked={fix[d] || []}
      text={(v) => cloudLabel(d, v)}
      onChange={(vs) => setPicked((f) => ({ ...f, [d]: vs }))}
    />
  );
  // The phase and the form sit with the chart's title, not among the
  // holds: they are what is drawn, the holds are what it is kept to.
  const switches = (
    <div className="flex items-center gap-1.5 flex-wrap">
      <div className={SEGMENTED}>
        {PHASES.map(({ phase: ph, label }) => {
          // A phase no point of this cloud prices is not a view of it; the
          // cloud says so for every form, so its Pareto's reason stands for
          // the phase.
          const reason = refusals?.[ph]?.frontier || waiting;
          return (
            <button
              key={ph}
              id={`infer-throughput-phase-${ph}`}
              type="button"
              disabled={!!reason}
              title={reason || undefined}
              onClick={() => setPhase(ph)}
              className={segment(ph === phase, reason)}
            >
              {label}
            </button>
          );
        })}
      </div>
      <div className={SEGMENTED}>
        {FORMS.map(({ form: f, label }) => {
          const reason = refusedFor(f) || waiting;
          return (
            <button
              key={f}
              id={`infer-throughput-form-${f}`}
              type="button"
              disabled={!!reason}
              title={reason || undefined}
              onClick={() => setForm(f)}
              className={segment(f === drawnForm, reason)}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
  const controls = (
    <div className={PICK_ROW}>
      {pending ? <ControlsSkeleton /> : (
        <>
          {rowDims.map(hold)}
          {lineOptions.length ? (
            <MultiPick
              id="infer-throughput-lines"
              label="Lines"
              emptyText="GPU only"
              options={lineOptions}
              picked={lineDims}
              text={(d) => HOLD_LABEL[d as CloudDim] ?? String(d)}
              onChange={(ds) => setPickedLines(ds as CloudDim[])}
            />
          ) : null}
        </>
      )}
      <span title={measuredReason
        || "Prefill only: a decode rung of the cloud keys no measurement, so decode is SOL."}>
        <MeasuredCheck id="infer-throughput-measured" checked={measuredOn}
          disabled={measuredReason} onChange={onShowMeasured}
          note={measuredOn ? measuredNote : null} />
      </span>
      {applied ? <span className="text-11 text-green-600">{applied}</span> : null}
    </div>
  );
  const subtitle = SUBTITLE[phase][drawnForm === "frontier" ? "frontier" : "line"]
    + (phase === "prefill" ? PREFILL_SCHEDULE_NOTE : "") + specNote + ` SOL${measuredOn ? ", measured beside it where every op of the step is measured" : " only"}.`
    + (onApply ? " Click a point to apply it." : "");
  const empty = failed ? `view load failed: ${failed}` : view ? "No layout serves these held values on these GPUs." : "Loading…";
  const pastNote = past ? (
    <div data-context-limit className="mt-1 text-11 text-amber-700">
      Some points are past the model's published max context; they extrapolate beyond what the model serves.
    </div>
  ) : null;
  const title = `${phase === "decode" ? "Decode" : "Prefill"}${held ? ` · ${held}` : ""}`;
  // One header for every form: what is drawn (the title and the two
  // switches on one line), why that is the comparison, and what the view
  // is kept to.
  const header = (
    <>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 mb-1.5">
        <div className="text-sm font-medium text-gray-800 min-w-0 truncate">{title}</div>
        {switches}
      </div>
      <p className="text-10 leading-snug text-gray-400 mb-2 max-w-[90ch]">{subtitle}</p>
      <div className="mb-2">
        {controls}
        {pending ? null : <FixedNote dims={fixedDims} values={lastValues} />}
      </div>
    </>
  );
  // The fields this view is drawn from, as the backend named them: the page
  // restates no part of the point's vocabulary. While a view is in flight
  // the last answer's names stand — the series drawn are that answer's, so
  // the chart's shape does not change under the controls and a popover the
  // reader has open stays open.
  const axes = (view ?? answer?.view ?? base?.view)?.axes;
  const covOf = (p: ParFrontierPt) =>
    (p as unknown as Record<string, unknown>)[axes?.coverage || ""] as MeasuredCoverage | null;
  const numAt = (p: ParFrontierPt, field: string | undefined) => {
    const v = field ? (p as unknown as Record<string, unknown>)[field] : null;
    return typeof v === "number" && Number.isFinite(v) ? v : NaN;
  };
  // Only the chart area below the header changes with the view: the header
  // is rendered once, outside these branches. A header re-created inside a
  // branch is a header REMOUNTED whenever the branch changes — which, with
  // a skeleton for a view in flight, is every time a reader holds a value:
  // the popover under their hand would close and the control move.
  let chart: ReactNode;
  // Being computed, or a view of it in flight: the box the chart will fill,
  // at the height it will have. A sentence in white space is not a chart,
  // and a box that appears when the answer does moves the page.
  if (pending || (!failed && !view)) {
    chart = <ChartSkeleton height={drawnForm === "frontier" ? 420 : 300} />;
  } else if (!axes) {
    chart = <div className="text-xs text-gray-400 text-center py-6">{empty}</div>;
  } else if (drawnForm === "frontier") {
    const series: FrontierSeries<ParFrontierPt>[] = groups.map((s, i) => {
      const st = style(s.group, i);
      return {
        key: st.key, label: st.label, color: st.color, points: s.points,
        ...(measuredOn
          ? { measured: s.points.filter((p) => isComplete(covOf(p))) } : {}),
        ...(cards.size > 1 && (roundOf.get(String(s.group.gpu_sku)) ?? 0) > 0
          ? { symbol: SYMBOLS[(roundOf.get(String(s.group.gpu_sku)) ?? 0) % SYMBOLS.length] } : {}),
        ...(cards.size > 1 && rest.length
          ? (rest.length === 1 && st.mode && MODE_LINE[st.mode] ? MODE_LINE[st.mode] : { lineType: LINE_TYPES[st.ri % LINE_TYPES.length] })
          : {}),
      };
    });
    chart = (
      <>
        <FrontierChart<ParFrontierPt>
          bare
          title=""
          series={series}
          axes={{ sol: axes.sol, measured: measuredOn ? axes.measured : null,
                  partial: null }}
          xTitle={AXIS_TITLE[phase].x}
          yTitle={AXIS_TITLE[phase].y}
          hover={(p) => (phase === "decode"
            ? `batch ${p.batch ?? "—"} / GPU${p.concurrent_requests != null ? ` · ${p.concurrent_requests} concurrent` : ""} · TPOT ${(p.tpot_s * 1000).toPrecision(3)} ms`
            : `one request · ctx ${fmtSeqLabel(p.context)} · TTFT ${p.prefill_t_s != null ? fmtT(p.prefill_t_s) : "—"}`)}
          height={420}
          empty={empty}
          onPoint={onApply ? (_s, p) => apply(p) : undefined}
        />
      </>
    );
  } else {
    const x = drawnForm;
    const [, gpuField] = axes.sol;
    const measField = measuredOn ? axes.measured?.[1] : undefined;
    const tField = phase === "decode" ? "tpot_s" : "prefill_t_s";
    const bField = phase === "decode" ? "batch" : "prefill_batch";
    const series: LineSeries[] = groups.flatMap((s, i) => {
      const st = style(s.group, i);
      const shape = {
        key: st.key, applyKey: st.key, label: st.label, color: st.color,
        ...(cards.size > 1 && (roundOf.get(String(s.group.gpu_sku)) ?? 0) > 0
          ? { symbol: SYMBOLS[(roundOf.get(String(s.group.gpu_sku)) ?? 0) % SYMBOLS.length] }
          : {}),
        ...(cards.size > 1 && st.mode && MODE_LINE[st.mode] && rest.length === 1
          ? { symbol: MODE_LINE[st.mode].symbol } : {}),
      };
      const note = (p: ParFrontierPt) =>
        `${p.num_gpus} GPU${p.num_gpus === 1 ? "" : "s"} · ${p.layout} · ctx ${fmtSeqLabel(p.context)}`;
      const sol: LineSeries = {
        ...shape,
        points: s.points.map((p) => ({
          x: Number(p[x as keyof ParFrontierPt] as unknown as number),
          v: numAt(p, gpuField),
          b: numAt(p, bField) || null,
          t: numAt(p, tField) || null,
          src: p,
          note: note(p),
          // Beside every SOL point: where the measured track is asked for
          // and this step is not fully measured, the hover says what is
          // missing instead of a number.
          ...(measField && !isComplete(covOf(p))
            ? { cov: coverageNote(covOf(p)) } : {}),
        })),
      };
      if (!measField) return [sol];
      const meas: LineSeries["points"] = s.points.map((p) => ({
        x: Number(p[x as keyof ParFrontierPt] as unknown as number),
        v: isComplete(covOf(p)) ? numAt(p, measField) : NaN,
        b: numAt(p, bField) || null,
        t: isComplete(covOf(p)) ? numAt(p, "prefill_t_measured_s") || null : null,
      }));
      if (!meas.some((p) => Number.isFinite(p.v))) return [sol];
      return [sol, { ...shape, key: `${shape.key}:meas`, dashed: true, points: meas }];
    });
    const xName: Record<Exclude<FormKey, "frontier">, string> = {
      num_gpus: "GPUs", batch: "decode batch / GPU", context: "context",
    };
    chart = (
      <div>
        {series.length ? (
          <SweepChart
            title={`${phase === "decode" ? "Decode" : "Prefill"} vs ${xName[x]}`}
            series={series}
            xTitle={xName[x]}
            xUnit={x === "num_gpus" ? "GPUs" : x === "batch" ? "batch" : "ctx"}
            {...(x === "context" ? { xLabel: fmtSeqLabel } : {})}
            yTitle="tok/s / GPU"
            valueUnit=" tok/s"
            onPoint={onApply ? (_s, p) => { if (p.src) apply(p.src as ParFrontierPt); } : undefined}
          />
        ) : <div className="text-xs text-gray-400 text-center py-6">{empty}</div>}
      </div>
    );
  }
  return (
    // No bottom margin of its own: the chart inside carries one, and the
    // rule under it is KV storage's — the panel is one card, not three
    // stacked fragments with a band of nothing between them.
    <div id="infer-throughput">
      {header}{chart}{pastNote}{measuredOn ? <MeasuredRule /> : null}
      <LadderNotes byMode={byMode} />
    </div>
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

/** The two KV charts' boxes while that sweep is computed: the same pair of
 *  cells, each the size its chart will be. */
function KvSkeleton() {
  return (
    <div className="flex gap-4 flex-wrap" data-kv-skeleton>
      {[0, 1].map((i) => (
        <div key={i} className="flex-1 min-w-[280px]">
          <div className="skeleton h-3 w-36 rounded mb-1" />
          <ChartSkeleton height={260} />
        </div>
      ))}
    </div>
  );
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
    <div id="infer-scaling-kv">
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
  // The contexts on screen, as the one control for them says: the Throughput
  // section's Context hold (`onHeldContexts`). Null until that section has a
  // cloud to hold anything of — a reader of it waits rather than assume one.
  // Empty is an answer too: the forms whose X is the context hold none.
  const [heldCtxs, setHeldCtxs] = useState<number[] | null>(null);
  const onHeldContexts = useCallback((next: number[]) => {
    setHeldCtxs((prev) => (prev && prev.length === next.length
      && prev.every((v, i) => v === next[i]) ? prev : next));
  }, []);
  // Measurements are keyed to the context they ran at, so the measured answer
  // is asked for ONE context: the held one, where exactly one is held.
  const measuredCtx = heldCtxs?.length === 1 ? heldCtxs[0] : null;
  // Speculative decoding for every sweep of the panel (the deployment cloud
  // and Context × Batch): a preset's K drafts verified per step
  // at acceptance α. "none" sends neither field, so the default bodies (and
  // their warm cache keys) are plain decode. `countDraft` adds the draft's
  // own forwards, priced off the model's MTP stack (`speculative_draft`);
  // off, the step is the verify alone — a drafter that runs no forward on the
  // GPU (n-gram / prompt lookup), the one form a model with no MTP stack has.
  const [specMode, setSpecMode] = useState("none");
  const [countDraft, setCountDraft] = useState(false);
  const [draftAvail, setDraftAvail] = useState<{ key: string; available: boolean; reason: string | null } | null>(null);
  const [showParMeas, setShowParMeas] = useState(false);
  const [status, setStatus] = useState<SweepStatus | null>(verdictStatus("Pick a model and GPU."));
  // The (batch × context) sweep as the backend returned it: KV storage's
  // one source. Its throughputs are not read here — prefill and decode
  // alike are views of the parallel sweep's one cloud.
  const [batchResults, setBatchResults] = useState<Record<string, SweepEst[]> | null>(null);
  const kvRows = useMemo(
    () => (batchResults ? sweepSeqResults({ results_by_batch: batchResults }) : null),
    [batchResults],
  );
  const [parStatus, setParStatus] = useState<SweepStatus | null>(computingStatus(CLOUD_WORK));
  const [parMeasNote, setParMeasNote] = useState<MeasuredNote | null>(null);
  const [byMode, setByMode] = useState<Record<string, ParModeBlob> | null>(null);
  // The parallel sweep's last answer for this model, for the GPU counts the
  // Context × Batch sweep may take: kept while another context's parallel
  // sweep is computed, so that sweep never falls back to a count nobody chose.
  const [degreeAnswer, setDegreeAnswer] = useState<{ model: string; byMode: Record<string, ParModeBlob> } | null>(null);
  // The parallel sweep body whose answer is on screen: the throughput
  // views read that same cache entry.
  const [parBody, setParBody] = useState<Record<string, unknown> | null>(null);
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
  // What holds BOTH sweeps of the panel, and which kind of hold it is: an
  // invalid ρ needs the reader (a verdict), the two lookups are on their
  // way (a wait). Said once, on the panel's one status line.
  const sweepHold: SweepStatus | null = rhoHold ? verdictStatus(rhoHold)
    : draftHold ? waitingStatus(draftHold)
      : props.calibration?.state === "loading"
        ? waitingStatus("Reading your calibration rows…") : null;
  const holdKey = sweepHold?.text || "";
  const lapsed = (e: unknown) => {
    if (calibrateMine && e instanceof ApiError && e.status === 401) props.onSessionLapsed?.();
  };
  const [grid, setGrid] = useState<ScalingGrid | null>(null);
  const [gridError, setGridError] = useState("");
  const degreeModes = degreeAnswer?.model === props.modelId.trim() ? degreeAnswer.byMode : null;
  // The cards Context × Batch offers: the ones the parallel sweep priced,
  // whose GPU counts it knows — none until it has answered.
  const swept = new Set(Object.values(degreeModes || {})
    .flatMap((m) => (m.series || []).map((s) => s.gpu_sku || "")));
  const sweptGpus = props.gpus.filter((g) => swept.has(skuOf(g)));
  const skuOpts = sweptGpus.map(skuOf).filter(Boolean);
  const ctxKey = JSON.stringify(heldCtxs ?? []);
  const { options: batchDegrees, refused: noDegreeFits } = useMemo(
    () => degreeOptions(degreeModes?.[ovPar], ovGpu, grid?.degrees || [],
                        JSON.parse(ctxKey) as number[]),
    [degreeModes, ovPar, ovGpu, grid, ctxKey],
  );
  // The GPU counts are not known yet: this model has no parallel answer, or
  // no context is on screen yet to read the counts at — and its parallel
  // sweep can run (a GPU to run it on) and has not failed.
  const degreesPending = (degreeModes === null || heldCtxs === null)
    && !!props.modelId.trim() && !!props.gpuSku && !parStatus?.failed;
  const batchDegree = snapDegree(ovDegree, batchDegrees);

  useEffect(() => {
    fetchJson<{ scaling_degrees?: number[] }>("/inference/defaults")
      .then((d) => {
        if (Array.isArray(d.scaling_degrees)) {
          setGrid({ degrees: d.scaling_degrees });
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
    const id = props.modelId.trim();
    const gpu = ovGpu;
    if (!id || !gpu) {
      { shownBody.current = ""; setBatchResults(null); }
      setStatus(verdictStatus("Pick a model and GPU."));
      return;
    }
    if (sweepHold) {
      ++seq.current;
      setStatus(sweepHold);
      return;
    }
    if (!grid) {
      // A held request still supersedes the one whose cleanup just aborted it.
      ++seq.current;
      setStatus(gridError ? verdictStatus(gridError, true) : computingStatus(KV_WORK));
      return;
    }
    if (degreesPending) {
      // A sweep at a guessed GPU count is a sweep nobody chose: wait for the
      // counts this card serves.
      ++seq.current;
      setStatus(waitingStatus("Waiting for the GPU counts this card serves…"));
      return;
    }
    if (noDegreeFits !== null) {
      // The parallel sweep found no GPU count on this card that serves the
      // model in this layout: say so, with its reason, and draw nothing.
      ++seq.current;
      { shownBody.current = ""; setBatchResults(null); }
      setStatus(verdictStatus(
        `No GPU count fits ${MODE_LABEL[ovPar]} on ${gpu}${noDegreeFits ? `: ${noDegreeFits}` : "."}`));
      return;
    }
    const body = {
      model_id: id,
      gpu_sku: gpu,
      seq_lens: SWEEP_SEQ_LENS,
      // The rungs this section still draws: one request (prefill) and the
      // ceiling (KV storage); its decode is the Decode section's.
      batch_grid: SCALING_PANEL_BATCH_GRID.map((p) => ({ ...p })),
      // One output length per panel: Plan's, as the parallel sweep's.
      avg_output_tokens: props.outTok ?? SWEEP_OUTPUT_TOKENS,
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
    // KV storage is a capacity fact, not a timing: this sweep is read as
    // its SOL answer alone and nothing is ever asked of Kernel Bench for
    // it. The measured track lives where a step time is drawn — the
    // throughput cloud above.
    const bodyKey = JSON.stringify(body);
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
    const hit = readSweepCache<SweepResp>(path, sol);
    if (hit && show(hit)) {
      shownBody.current = bodyKey;
      return () => ctrl.abort();
    }
    setStatus(computingStatus(KV_WORK));
    fetchSweepCached<SweepResp>(path, sol, { signal: ctrl.signal },
      (info) => { if (my === seq.current) setStatus(computingStatus(KV_WORK, info.progress)); })
      .then((resp) => {
        if (my !== seq.current) return;
        if (show(resp)) {
          shownBody.current = bodyKey;
          return;
        }
        { shownBody.current = ""; setBatchResults(null); }
        setStatus(verdictStatus(missing.message, true));
      })
      .catch((e) => {
        if (my !== seq.current || isAbort(e)) return;
        { shownBody.current = ""; setBatchResults(null); }
        setStatus(verdictStatus(e instanceof Error ? e.message : String(e), true));
      });
    return () => ctrl.abort();
  }, [props.modelId, ovGpu, ovPar, batchDegree, noDegreeFits, degreesPending, grid, gridError, props.precision, props.frac, props.commLatencyUs, props.launchFloorUs, props.outTok, props.selectionMode, props.framework, props.kvBlock, rho, holdKey, specBody, calibrateMine]);

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
      setParStatus(sweepHold);
      return;
    }

    // The SOL answer prices every context, so holding another one re-asks
    // nothing; the measured answer is read live, so it asks only for the
    // context the Throughput section holds (`measuredCtx`).
    const base = {
      model_id: id,
      gpu_sku: props.gpuSku,
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
    const all = { ...base, contexts: SWEEP_SEQ_LENS };
    const body = { ...base, contexts: measuredCtx == null ? [] : [measuredCtx] };
    // No context held is not a context to read measurements at: the box says
    // so (`measuredReason`) and nothing is asked.
    const wantMeasured = showParMeas && measuredCtx != null && !isTheory(body);
    const bodyKey = JSON.stringify(all) + (wantMeasured ? `|measured|${measuredCtx}` : "");
    if (bodyKey === shownParBody.current) return;
    const path = "/inference/estimate-parallel-sweep";
    const sol = solOnly(all);
    // `read` is the sweep body whose answer this response IS: the views
    // read that one, so the measured answer's prefill track — folded into
    // the very points of its cloud — is what the charts draw once it lands.
    const show = (resp: ParResp, read: Record<string, unknown>) => {
      const mapped = resp.by_mode || (resp.mode ? { [resp.mode]: resp } : {});
      setByMode(mapped);
      setDegreeAnswer({ model: id, byMode: mapped });
      setParBody(read);
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
          show(resp, body);
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
      show(hit, sol);
      measure();
      return () => ctrl.abort();
    }
    // The curves on screen are the previous inputs'; drop them so a slow
    // sweep never shows them under the new model or GPU.
    { shownParBody.current = ""; setByMode(null); }
    setParStatus(computingStatus(CLOUD_WORK));
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
        setParStatus(computingStatus(CLOUD_WORK, info.progress));
      })
      .then((resp) => {
        if (my !== parSeq.current) return;
        show(resp, sol);
        measure();
      })
      .catch((e) => {
        if (my !== parSeq.current || isAbort(e)) return;
        { shownParBody.current = ""; setByMode(null); }
        setParStatus(verdictStatus(
          `sweep load failed: ${e instanceof Error ? e.message : String(e)}`, true));
      });
    return () => ctrl.abort();
  }, [props.modelId, props.gpuSku, measuredCtx, props.precision, props.frac, props.commLatencyUs, props.launchFloorUs, props.outTok, props.selectionMode, props.framework, props.kvBlock, rho, holdKey, specBody, calibrateMine, showParMeas]);

  // The panel's status line, under its header: the deployment sweep's — the
  // one every chart of the Throughput block waits on, and the one that
  // carries what holds BOTH sweeps (an invalid ρ, calibration rows still
  // being read).
  const line = parStatus;
  // KV storage's own status sits with KV storage, where its charts are —
  // but never when it would repeat the sentence already on the line above.
  // The same hold, printed by each block in its own empty space, is what
  // made this panel read as two broken sections.
  const kvNote = status && status.text !== line?.text ? status : null;
  return (
    <section id="infer-scaling" className="card p-4 sm:p-6 mb-6">
      <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
        <h2 className="text-lg font-semibold text-gray-900">Scaling</h2>
        <div className="flex items-center gap-3 flex-wrap">
          <OnePick
            id="infer-scaling-spec"
            label="Spec decode"
            title="Decode, every sweep below: the target verifies K draft tokens in one forward over K + 1 positions and keeps 1 + α·K tokens per step."
            value={specMode}
            onChange={(e) => setSpecMode(e.target.value)}
          >
            {Object.entries(SPEC_PRESETS).filter(([v]) => v !== "custom").map(([v, p]) => (
              <option key={v} value={v}>{p.label}</option>
            ))}
          </OnePick>
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
        </div>
      </div>
      {line ? <PanelStatus status={line} /> : null}
      <ThroughputSection
        sweepBody={byMode && parBody && !parStatus ? parBody : null}
        // What the layout refused, from the cells that have landed: the
        // server's own assembly of a sweep still computing already says so,
        // and these notes only ever grow as the rest arrive.
        byMode={byMode}
        onApply={props.onApply}
        planGpu={props.gpuSku}
        showMeasured={showParMeas}
        onShowMeasured={setShowParMeas}
        measuredNote={parMeasNote}
        onHeldContexts={onHeldContexts}
        gpus={(Object.values(byMode || {}).find((m) => m.series?.length)?.series || [])
          .map((s) => s.gpu_sku || "")}
      />
      <div className="border-t border-gray-100 pt-4">
        <div className={`${PICK_ROW} mb-3`}>
          <div className="text-sm font-medium text-gray-800 mr-1">KV storage</div>
          <OnePick
            id="infer-scaling-gpu"
            label="GPU"
            className="max-w-[18rem]"
            value={skuOpts.includes(ovGpu) ? ovGpu : (skuOpts[0] || ovGpu)}
            onChange={(e) => setOvGpu(e.target.value)}
          >
            {skuOpts.length
              ? sweptGpus.map((g) => <option key={skuOf(g)} value={skuOf(g)}>{gpuOptionLabel(g)}</option>)
              : <option value={ovGpu}>{ovGpu}</option>}
          </OnePick>
          <OnePick
            id="infer-scaling-par"
            label="Parallel"
            value={ovPar}
            onChange={(e) => setOvPar(e.target.value as ParSweepMode)}
          >
            {PAR_SWEEP_MODES.filter((m) => m === "tp_tp" || !hideEp).map((m) => (
              <option key={m} value={m}>{MODE_LABEL[m]}</option>
            ))}
          </OnePick>
          <OnePick
            id="infer-scaling-ngpu"
            label="GPUs"
            // No counts to offer yet (the ladder or the sweep that filters it
            // has not answered) or none that serve: a dash and no dropdown,
            // never an empty box over a value nobody chose.
            disabled={!grid || !batchDegrees.length}
            title={!grid ? "Reading the GPU-count ladder"
              : degreesPending ? "Waiting for the GPU counts this card serves"
                : noDegreeFits !== null
                  ? `No GPU count fits ${MODE_LABEL[ovPar]} on ${ovGpu}`
                  : undefined}
            value={batchDegrees.length ? batchDegree : ""}
            onChange={(e) => setOvDegree(Number(e.target.value))}
          >
            {batchDegrees.length
              ? batchDegrees.map((n) => <option key={n} value={n}>{n}</option>)
              : <option value="">—</option>}
          </OnePick>
        </div>
        {kvNote ? (
          <div data-kv-note
            className={`mb-2 text-11 ${kvNote.failed ? "text-red-500"
              : kvNote.kind === "verdict" ? "text-amber-700" : "text-gray-500"}`}>
            {kvNote.progress
              ? <ComputeProgress compact label={kvNote.text} progress={kvNote.progress} unit="cells" />
              : kvNote.text}
          </div>
        ) : null}
        {!status ? <KvStorageSection rows={kvRows} />
          // Its answer is still coming — being computed, or waiting on the
          // sweep that says which GPU counts serve: the boxes the two
          // charts will fill, at their height, so the sections above keep
          // their position while this (independent) sweep lands. A verdict
          // draws no box: its line above is the whole answer.
          : stillComing(status) ? <KvSkeleton /> : null}
      </div>
    </section>
  );
}
