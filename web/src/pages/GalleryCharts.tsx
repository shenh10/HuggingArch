import type { EChartsOption, EChartsType } from "echarts";
import { useEffect, useMemo, useRef, type ReactNode } from "react";
import { binaryUnit, escapeHtml, fmtCompact, formatGiB } from "../format";
import { AXIS, HaChart, interactiveBits } from "../charts/HaChart";
import { formulaHTML, type FormulaFields } from "../tooltip";
import { makeLineEndLabelLayout, makeScatterLabelLayout } from "../charts/labelLayout";
import { legendRows, useWidth } from "../charts/useWidth";

/** Twenty hues no two of which read as the same line: Tableau 10, then ten
 *  more chosen apart from it (the old second half was category10, the same
 *  ten hues again a shade off — line 11 looked like line 1). A chart draws
 *  at most `LINE_MAX` = 20 lines, so a colour names one line. */
export const G_PALETTE = [
  "#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f",
  "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ac",
  "#1d3557", "#c51b7d", "#00a6a6", "#8c2d04", "#6a3d9a",
  "#7fbc41", "#b8860b", "#2f4f4f", "#a6cee3", "#525252",
];

export type SeriesPt = {
  x: number;
  y: number;
  label?: string;
  tip?: string;
  attention?: string[];
  /** Point diameter in px. The caller encodes a third quantity as area. */
  size?: number;
  /** Full model id for tooltips; on-chart labels stay short. */
  full?: string;
};
export type LineSeries = {
  key: string;
  label: string;
  color: string;
  points: SeriesPt[];
  /** Full model id for tooltips; legend and end labels stay short. */
  full?: string;
};

export function shortName(modelId: string): string {
  const i = modelId.lastIndexOf("/");
  return i >= 0 ? modelId.slice(i + 1) : modelId;
}

export function parseReleaseDate(s: string | null | undefined): number | null {
  if (!s) return null;
  const m = s.match(/^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?/);
  if (!m) return null;
  const y = Number(m[1]), mo = m[2] ? Number(m[2]) : 1, d = m[3] ? Number(m[3]) : 1;
  if (mo < 1 || mo > 12 || d < 1 || d > 31) return null;
  return y + (mo - 1) / 12 + (d - 1) / 365;
}


export function fmtYear(n: number): string {
  return String(Math.round(n));
}

/** Fractional year (parseReleaseDate) as "YYYY-MM" — the release timeline's
 *  tick label; a bare year is too coarse once several models share one. */
export function fmtYearMonth(n: number): string {
  const y = Math.floor(n);
  const mo = Math.min(12, Math.round((n - y) * 12) + 1);
  return `${y}-${String(mo).padStart(2, "0")}`;
}

/** Break long model names at hyphens so legend/point labels wrap instead of
 *  paginating (`1/3`) or colliding. Short names stay one line. */
export function wrapChartLabel(name: string, max = 14): string {
  if (!name || name.length <= max) return name;
  const parts: string[] = [];
  let rest = name;
  while (rest.length > max) {
    // Break at any separator — hyphen, dash or the " · " before params — so
    // labels like "GLM–5.1 · 41.78B" never split mid-number.
    let cut = Math.max(
      rest.lastIndexOf("-", max), rest.lastIndexOf("–", max),
      rest.lastIndexOf("—", max), rest.lastIndexOf(" ", max),
    );
    if (cut < Math.floor(max / 3)) cut = max;
    parts.push(rest.slice(0, cut));
    rest = "-–— ".includes(rest.charAt(cut)) ? rest.slice(cut + 1) : rest.slice(cut);
  }
  if (rest) parts.push(rest);
  return parts.join("\n");
}

function labelName(p: { data?: unknown; name?: string }): string {
  const d = p.data;
  if (d && typeof d === "object" && "name" in d && typeof (d as { name: unknown }).name === "string") {
    return wrapChartLabel((d as { name: string }).name);
  }
  return wrapChartLabel(p.name || "");
}

function attentionBadges(labels: string[] | undefined): string {
  return (labels || []).map((label) => {
    const text = escapeHtml(label);
    return `<span style="display:inline-block;margin-left:5px;padding:0 5px;border:1px solid #fcd34d;border-radius:9999px;background:#fffbeb;color:#92400e;font-size:10px;font-weight:600;line-height:15px;vertical-align:1px" title="Attention family">${text}</span>`;
  }).join("");
}

function legendOpts(names?: string[]) {
  return {
    type: "plain" as const,
    bottom: 0,
    left: "center" as const,
    width: "92%",
    itemGap: 10,
    itemWidth: 10,
    itemHeight: 8,
    textStyle: { fontSize: 10, lineHeight: 13 },
    formatter: (name: string) => wrapChartLabel(name),
    ...(names ? { data: names } : {}),
  };
}

function axis(
  type: "log" | "value",
  name: string,
  fmt?: (n: number) => string,
  nameGap = 36,
) {
  return {
    type,
    name,
    nameLocation: "middle" as const,
    nameGap,
    ...AXIS,
    axisLabel: {
      ...AXIS.axisLabel,
      formatter: fmt ? (v: string | number) => fmt(Number(v)) : undefined,
    },
  };
}

export function ScatterChart({
  title, subtitle, series, xTitle, yTitle, xLog, yLog, xFmt, yFmt, xInterval, height = 420, empty = "no points",
  note,
}: {
  title: string;
  subtitle?: string;
  series: LineSeries[];
  xTitle: string;
  yTitle: string;
  xLog?: boolean;
  yLog?: boolean;
  xFmt?: (n: number) => string;
  yFmt?: (n: number) => string;
  /** Fixed X tick step in axis units (e.g. 0.25 = quarterly on the
   *  fractional-year release axis); unset lets ECharts pick. */
  xInterval?: number;
  height?: number;
  /** Shown instead of the chart when there is nothing to plot. */
  empty?: ReactNode;
  /** Under the chart: what it could not place (e.g. models with no date). */
  note?: ReactNode;
}) {
  const [box, width] = useWidth<HTMLDivElement>();
  const xf = xFmt || fmtCompact;
  const yf = yFmt || fmtCompact;
  const pts = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && p.x > 0 && p.y > 0);
  // Every bubble's box, from the last layout pass, so a label keeps off the
  // bubbles laid out after it too (`makeScatterLabelLayout`). Started afresh
  // for another set of points or size; once drawn, the chart lays its labels
  // out once more with every bubble known.
  const layoutKey = `${width}x${height}|${series.map((s) => `${s.label}:${s.points.map((p) => `${p.x},${p.y},${p.size ?? ""}`).join(";")}`).join("|")}`;
  const bubbles = useMemo(() => new Map<string, { x1: number; y1: number; x2: number; y2: number }>(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [layoutKey]);
  const chartRef = useRef<EChartsType | null>(null);
  useEffect(() => {
    const id = requestAnimationFrame(() => chartRef.current?.resize());
    return () => cancelAnimationFrame(id);
  }, [layoutKey]);
  if (!pts.length) {
    return (
      <section className="card p-5 mb-6">
        <h2 className="text-base font-semibold text-gray-800 mb-1">{title}</h2>
        {subtitle ? <p className="text-xs text-gray-400 mb-3">{subtitle}</p> : null}
        <div className="text-xs text-gray-400 py-10 text-center">{empty}</div>
        {note}
      </section>
    );
  }
  const grid = { left: 72, right: 24, top: 36, bottom: 96 };
  // Labels stay inside the plot: one at the right edge goes to the left of
  // its bubble rather than off the card.
  const scatterLayout = makeScatterLabelLayout({
    top: grid.top, bottom: height - grid.bottom,
    ...(width ? { left: grid.left, right: width - grid.right } : {}),
  }, bubbles);
  // Room past the outermost bubbles, so none is cut in half by the plot edge.
  const span = (lo: number, hi: number, log?: boolean) => log
    ? { min: (v: { min: number }) => v.min / 1.6, max: (v: { max: number }) => v.max * 1.6 }
    : { min: (v: { min: number; max: number }) => v.min - (v.max - v.min) * 0.04 - (lo === hi ? 1 : 0),
        max: (v: { min: number; max: number }) => v.max + (v.max - v.min) * 0.04 + (lo === hi ? 1 : 0) };
  const xsAll = pts.map((p) => p.x);
  const ysAll = pts.map((p) => p.y);
  const option: EChartsOption = {
    color: series.map((s) => s.color),
    ...interactiveBits({ name: title }),
    grid,
    tooltip: {
      trigger: "item",
      formatter: (p: unknown) => {
        const d = p as { seriesName?: string; data?: { value?: number[]; name?: string; tip?: string; attention?: string[]; full?: string } };
        const v = d.data?.value || [];
        const name = d.data?.full || d.data?.name || d.seriesName || "";
        return `${escapeHtml(name)}${attentionBadges(d.data?.attention)}<br/>${escapeHtml(
          d.data?.tip || `${xf(v[0] || 0)} · ${yf(v[1] || 0)}`,
        )}`;
      },
    },
    legend: legendOpts(series.map((s) => s.label)),
    // Linear X (release year) must not start at 0 — that pins every 2020s
    // model to the right edge; each axis runs a little past its data. The
    // padded ends are not round values, so they carry no label of their own
    // (a log axis cut at the data used to label its floor "30.53B").
    xAxis: {
      ...axis(xLog ? "log" : "value", xTitle, xf, 28),
      ...(xLog ? {} : { scale: true }),
      ...span(Math.min(...xsAll), Math.max(...xsAll), xLog),
      ...(xInterval ? { interval: xInterval } : {}),
      axisLabel: { ...AXIS.axisLabel, formatter: (v: string | number) => xf(Number(v)), showMinLabel: false, showMaxLabel: false },
    },
    yAxis: {
      ...axis(yLog ? "log" : "value", yTitle, yf, 52),
      ...span(Math.min(...ysAll), Math.max(...ysAll), yLog),
      axisLabel: { ...AXIS.axisLabel, formatter: (v: string | number) => yf(Number(v)), showMinLabel: false, showMaxLabel: false },
    },
    series: [...series]
      .sort((a, b) => Math.max(0, ...b.points.map((p) => p.x)) - Math.max(0, ...a.points.map((p) => p.x)))
      .map((s) => ({
      type: "scatter" as const,
      name: s.label,
      data: s.points
        .filter((p) => p.x > 0 && p.y > 0)
        .slice()
        .sort((a, b) => b.x - a.x)
        .map((p) => ({
          value: [p.x, p.y],
          name: p.label || s.label,
          full: p.full,
          tip: p.tip,
          attention: p.attention,
          symbolSize: p.size,
        })),
      symbolSize: 8,
      itemStyle: { color: s.color },
      label: {
        show: true,
        formatter: (p: unknown) => labelName(p as { data?: unknown; name?: string }),
        fontSize: 9,
        lineHeight: 12,
        color: "#4b5563",
        position: "right",
      },
      labelLine: { show: true, lineStyle: { width: 0.6, color: "rgba(107,114,128,0.45)" } },
      labelLayout: scatterLayout,
    })),
  };
  return (
    <section className="card p-5 mb-6">
      <h2 className="text-base font-semibold text-gray-800 mb-1">{title}</h2>
      {subtitle ? <p className="text-xs text-gray-400 mb-3">{subtitle}</p> : null}
      <div ref={box}>
        <HaChart option={option} height={height} onInstance={(c) => { chartRef.current = c; }} />
      </div>
      {note}
    </section>
  );
}

export type KvBarRow = {
  key: string;
  label: string;
  /** Per-request KV at 256K context, as its formula (null = no data);
   *  the bar is its `result_bytes`. */
  f256k: FormulaFields | null;
  /** Per-request KV at 1M context, as its formula (null = no data). */
  f1M: FormulaFields | null;
  /** Full model id for the tooltip; row labels stay short. */
  full?: string;
  /** The element dtypes the cache is stored in, most bytes first — shown
   *  under the name when given. */
  dtypes?: string[];
};

/** A row's bytes at 256K / 1M (null = no data). */
export const v256k = (r: KvBarRow) => r.f256k?.result_bytes ?? null;
export const v1M = (r: KvBarRow) => r.f1M?.result_bytes ?? null;

/** Leaderboard-style horizontal bar chart of per-request KV cache at the two
 *  longest gallery contexts. One row per model, grouped bars for 256K / 1M.
 *  Linear X, sorted small → large: bar length stays proportional to cache
 *  size, so the GQA-vs-MLA gap reads off the chart directly. */
export function KvBarChart({
  title, subtitle, rows, empty = "no data", controls,
}: {
  title: string;
  subtitle?: string;
  /** Extra header controls (the precision switcher), right of the title. */
  controls?: ReactNode;
  /** Shown instead of the chart when there are no rows. */
  empty?: ReactNode;
  /** Pre-sorted by the caller: primary 1M asc, secondary 256K asc. */
  rows: KvBarRow[];
}) {
  const head = (
    <div className="flex flex-wrap items-baseline justify-between gap-2 mb-1">
      <h2 className="text-base font-semibold text-gray-800">{title}</h2>
      {controls}
    </div>
  );
  if (!rows.length) {
    return (
      <section className="card p-5 mb-6">
        {head}
        {subtitle ? <p className="text-xs text-gray-400 mb-3">{subtitle}</p> : null}
        <div className="text-xs text-gray-400 py-10 text-center">{empty}</div>
      </section>
    );
  }
  const height = Math.max(320, 110 + rows.length * 46);
  // Ratio anchor: the leader (first row after the small → large sort) at
  // 1M — the primary sort key. Every 1M bar label carries `×N` vs it.
  const leader1M = rows.map(v1M).find((v): v is number => (v ?? 0) > 0) ?? null;
  const fmtRatio = (v: number) => {
    const r = v / (leader1M as number);
    return `×${r >= 100 ? r.toFixed(0) : r.toFixed(1)}`;
  };
  const seriesFor = (name: string, pick: (r: KvBarRow) => FormulaFields | null, color: string, ratio = false) => ({
    type: "bar" as const,
    name,
    data: rows.map((r) => {
      const v = pick(r)?.result_bytes;
      return v == null ? null : v / unit.scale;
    }),
    barMaxWidth: 14,
    itemStyle: { color, borderRadius: [0, 3, 3, 0] },
    label: {
      show: true,
      position: "right" as const,
      fontSize: 9,
      color: "#4b5563",
      formatter: (p: unknown) => {
        const scaled = (p as { value?: unknown }).value;
        if (typeof scaled !== "number" || !Number.isFinite(scaled)) return "";
        const v = scaled * unit.scale;
        const base = formatGiB(v);
        return ratio && leader1M ? `${base} · ${fmtRatio(v)}` : base;
      },
    },
  });
  // Bars are plotted in one binary unit so the axis picks round ticks in it
  // (16 GiB, 32 GiB); labels and tooltips read the raw bytes.
  const unit = binaryUnit(Math.max(0, ...rows.flatMap((r) => [v256k(r) ?? 0, v1M(r) ?? 0])));
  const option: EChartsOption = {
    ...interactiveBits({ name: title }),
    grid: { left: 8, right: 118, top: 34, bottom: 46, containLabel: true },
    // One bar, one tooltip: its value, then the formula it is.
    tooltip: {
      trigger: "item",
      confine: true,
      extraCssText: "max-width: 560px; white-space: normal;",
      formatter: (item: unknown) => {
        const c = item as { seriesIndex?: number; seriesName?: string; marker?: string; dataIndex?: number };
        const r = rows[c.dataIndex ?? -1];
        if (!r) return "";
        const f = c.seriesIndex === 1 ? r.f1M : r.f256k;
        const v = f?.result_bytes;
        if (v == null) return "";
        const base = `${c.marker || ""}${escapeHtml(c.seriesName || "")}: ${formatGiB(v)}`;
        const head = c.seriesIndex === 1 && leader1M ? `${base} · ${fmtRatio(v)}` : base;
        return `<div class="text-xs"><div class="font-semibold">${escapeHtml(r.full || r.label)}</div>`
          + `<div class="mb-2">${head}</div>${formulaHTML(f as FormulaFields)}</div>`;
      },
    },
    legend: {
      top: 0,
      right: 8,
      itemWidth: 10,
      itemHeight: 8,
      textStyle: { fontSize: 10 },
    },
    xAxis: {
      type: "value",
      name: "Per-request KV cache",
      nameLocation: "middle",
      nameGap: 30,
      ...AXIS,
      axisLabel: {
        ...AXIS.axisLabel,
        formatter: (v: string | number) => (Number(v) === 0 ? "0" : `${fmtCompact(Number(v))} ${unit.unit}`),
      },
    },
    yAxis: {
      type: "category",
      data: rows.map((r) => r.label),
      inverse: true,
      ...AXIS,
      axisLabel: {
        ...AXIS.axisLabel, fontSize: 10, color: "#374151", lineHeight: 14,
        formatter: (label: string, i: number) => {
          const dt = rows[i]?.dtypes;
          return dt?.length ? `${label}\n{dt|${dt.join(" + ")}}` : label;
        },
        rich: { dt: { fontSize: 9, lineHeight: 12, color: "#9ca3af", align: "right" } },
      },
    },
    series: [
      seriesFor("256K context", (r) => r.f256k, "#76b7b2"),
      seriesFor("1M context", (r) => r.f1M, "#4e79a7", true),
    ],
  };
  return (
    <section className="card p-5 mb-6">
      {head}
      {subtitle ? <p className="text-xs text-gray-400 mb-3">{subtitle}</p> : null}
      <HaChart option={option} height={height} />
    </section>
  );
}

/** One point of a decode Pareto frontier (`/gallery/frontier`). */
export type FrontierPt = {
  tok_s_per_user: number;
  tok_s_per_gpu: number;
  tpot_s: number;
  batch: number | null;
  layout: string;
  num_gpus: number;
  /** Requests the cluster decodes at once. */
  concurrent_requests?: number;
  tok_s_per_user_measured?: number | null;
  tok_s_per_gpu_measured?: number | null;
  tpot_measured_s?: number | null;
  /** The partial track: measured ops measured, the rest at SOL, and the
   *  share of the step's time that is measured (0–1). */
  tok_s_per_user_partial?: number | null;
  tok_s_per_gpu_partial?: number | null;
  tpot_partial_s?: number | null;
  partial_measured_share?: number | null;
};

export type FrontierSeries<P extends FrontierPt = FrontierPt> = {
  key: string;
  label: string;
  color: string;
  full?: string;
  points: P[];
  measured?: P[];
  /** The partial-measured frontier (opt-in), drawn dotted. */
  partial?: P[];
  /** Line style, for series that share a colour (one card under several
   *  layouts). */
  lineType?: "solid" | "dashed" | "dotted";
  symbol?: string;
};

/** Decode Pareto frontier: tokens/s one request sees (x) against tokens/s
 *  each GPU outputs (y), one line per model through its frontier points.
 *  Log on both axes — twenty models span decades on each. */
export function FrontierChart<P extends FrontierPt = FrontierPt>({
  title, subtitle, series, controls, height = 480, empty = "no curve", stale, footer,
  onPoint, bare,
}: {
  title: string;
  subtitle?: string;
  series: FrontierSeries<P>[];
  controls?: ReactNode;
  height?: number;
  empty?: ReactNode;
  stale?: ReactNode;
  /** Drawn under the chart in the same card, only when a curve is: it
   *  reads the chart's controls, so it shares the chart's stale note. */
  footer?: ReactNode;
  /** A click on a frontier point. */
  onPoint?: (s: FrontierSeries<P>, p: P) => void;
  /** Drawn inside another card: no card of its own, a smaller heading. */
  bare?: boolean;
}) {
  const [box, width] = useWidth<HTMLDivElement>();
  const drawn = series.filter((s) => s.points.length);
  const head = (
    <div className="flex flex-wrap items-baseline justify-between gap-2 mb-1">
      {bare
        ? <div className="text-sm font-medium text-gray-800">{title}</div>
        : <h2 className="text-base font-semibold text-gray-800">{title}</h2>}
      {controls}
    </div>
  );
  const shell = bare ? "mb-6" : "card p-5 mb-6";
  const staleNote = stale
    ? <div className="g-stale-note text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1 mb-2">{stale}</div>
    : null;
  if (!drawn.length) {
    return (
      <section className={shell}>
        {head}
        {subtitle ? <p className="text-xs text-gray-400 mb-3">{subtitle}</p> : null}
        {staleNote}
        <div className="text-xs text-gray-400 py-10 text-center">{empty}</div>
      </section>
    );
  }
  // The legend sits under the x-axis name: the plot gives up a row of
  // height per row past the ones the default margin holds.
  const wrapped = drawn.map((s) => wrapChartLabel(s.label).split("\n"));
  const rowH = wrapped.some((ls) => ls.length > 1) ? 28 : 16;
  const rows = legendRows(wrapped.map((ls) => ls.reduce((a, b) => (b.length > a.length ? b : a))),
    width, { iconPx: 10, gapPx: 10, charPx: 6 });
  const extra = Math.max(0, 56 + rows * rowH - 108);
  const chartH = height + extra;
  const grid = { left: 72, right: 24, top: 28, bottom: 108 + extra, containLabel: false };
  const endLayout = makeLineEndLabelLayout({
    top: grid.top, bottom: chartH - grid.bottom,
    ...(width ? { left: grid.left, right: width - grid.right } : {}),
  });
  const perUser = (n: number) => `${fmtCompact(n)} tok/s/user`;
  const perGpu = (n: number) => `${fmtCompact(n)} tok/s/GPU`;
  // Linear axes from zero: both rates are non-negative, and a shared zero
  // keeps the gaps between curves proportional to the rates themselves.
  const linearSpan = {
    min: 0,
    max: (v: { max: number }) => v.max * 1.05,
  };
  // The first `drawn.length` series are the SOL frontiers, in order.
  const events = onPoint ? {
    click: (raw: unknown) => {
      const it = raw as { seriesIndex?: number; data?: { pt?: P } };
      const s = it.seriesIndex != null ? drawn[it.seriesIndex] : undefined;
      if (s && it.data?.pt) onPoint(s, it.data.pt);
    },
  } : undefined;
  const option: EChartsOption = {
    color: drawn.map((s) => s.color),
    ...interactiveBits({ name: title }),
    grid,
    tooltip: {
      trigger: "item",
      formatter: (item: unknown) => {
        const it = item as { seriesIndex?: number; seriesName?: string; data?: { pt?: P } };
        // A measured or partial line is named as its model's SOL line.
        const s = (it.seriesIndex != null ? drawn[it.seriesIndex] : undefined)
          ?? drawn.find((d) => d.label === it.seriesName);
        const p = it.data?.pt;
        if (!s || !p) return "";
        const lines = [
          `<b>${escapeHtml(s.full || s.label)}</b>`,
          escapeHtml(`${p.num_gpus}×GPU · ${p.layout}`),
          `batch ${p.batch ?? "—"} / GPU${p.concurrent_requests != null ? ` · ${p.concurrent_requests} concurrent` : ""} · TPOT ${(p.tpot_s * 1000).toPrecision(3)} ms`,
          `${perUser(p.tok_s_per_user)} · ${perGpu(p.tok_s_per_gpu)}`,
        ];
        if (p.tok_s_per_user_measured != null && p.tok_s_per_gpu_measured != null) {
          lines.push(`measured ${perUser(p.tok_s_per_user_measured)} · ${perGpu(p.tok_s_per_gpu_measured)}`);
        }
        if (p.tok_s_per_user_partial != null && p.tok_s_per_gpu_partial != null) {
          const share = p.partial_measured_share;
          lines.push(`partial measured ${perUser(p.tok_s_per_user_partial)} · ${perGpu(p.tok_s_per_gpu_partial)}`
            + (share != null ? ` (${(share * 100).toFixed(0)}% of time measured)` : ""));
        }
        return lines.join("<br/>");
      },
    },
    legend: legendOpts(drawn.map((s) => s.label)),
    xAxis: {
      ...axis("value", "Interactivity (tok/s per user)", fmtCompact, 28), ...linearSpan,
      axisLabel: { ...AXIS.axisLabel, formatter: (v: string | number) => fmtCompact(Number(v)), showMinLabel: true, showMaxLabel: false },
    },
    yAxis: {
      ...axis("value", "Output throughput (tok/s per GPU)", fmtCompact, 52), ...linearSpan,
      axisLabel: { ...AXIS.axisLabel, formatter: (v: string | number) => fmtCompact(Number(v)), showMinLabel: true, showMaxLabel: false },
    },
    series: [
      ...drawn.map((s) => ({
        type: "line" as const,
        name: s.label,
        // The best layout is named where it changes along the curve;
        // hovering a curve names every point.
        data: [...s.points]
          .sort((a, b) => a.tok_s_per_user - b.tok_s_per_user)
          .map((p, i, sorted) => ({
            value: [p.tok_s_per_user, p.tok_s_per_gpu],
            pt: p,
            label: { show: i === 0 || sorted[i - 1].layout !== p.layout },
          })),
        showSymbol: true,
        symbol: s.symbol || "emptyCircle",
        symbolSize: 5,
        lineStyle: { width: 1.6, color: s.color, type: s.lineType || "solid" },
        itemStyle: { color: s.color },
        label: {
          formatter: (x: unknown) => (x as { data?: { pt?: FrontierPt } }).data?.pt?.layout ?? "",
          fontSize: 9,
          color: s.color,
          position: "top" as const,
        },
        labelLayout: { hideOverlap: true },
        emphasis: { focus: "series" as const, label: { show: true } },
      })),
      ...drawn.flatMap((s) => {
        const pts = (s.measured || []).filter(
          (p) => p.tok_s_per_user_measured != null && p.tok_s_per_gpu_measured != null,
        );
        if (!pts.length) return [];
        return [{
          type: "line" as const,
          name: s.label,
          data: [...pts]
            .sort((a, b) => (a.tok_s_per_user_measured ?? 0) - (b.tok_s_per_user_measured ?? 0))
            .map((p) => ({
              value: [p.tok_s_per_user_measured, p.tok_s_per_gpu_measured],
              pt: p,
            })),
          showSymbol: true,
          symbolSize: 5,
          lineStyle: { width: 1.4, type: "dashed" as const, color: s.color },
          itemStyle: { color: s.color },
        }];
      }),
      ...drawn.flatMap((s) => {
        const pts = (s.partial || []).filter(
          (p) => p.tok_s_per_user_partial != null && p.tok_s_per_gpu_partial != null,
        );
        if (!pts.length) return [];
        return [{
          type: "line" as const,
          name: s.label,
          data: [...pts]
            .sort((a, b) => (a.tok_s_per_user_partial ?? 0) - (b.tok_s_per_user_partial ?? 0))
            .map((p) => ({
              value: [p.tok_s_per_user_partial, p.tok_s_per_gpu_partial],
              pt: p,
            })),
          showSymbol: true,
          symbol: "triangle",
          symbolSize: 5,
          lineStyle: { width: 1.4, type: "dotted" as const, color: s.color },
          itemStyle: { color: s.color },
        }];
      }),
      // The name rides the most interactive end, as LineChart's end labels.
      ...drawn.map((s) => {
        const last = s.points.reduce((a, b) => (b.tok_s_per_user > a.tok_s_per_user ? b : a));
        return {
          type: "scatter" as const,
          name: s.label,
          data: [{ value: [last.tok_s_per_user, last.tok_s_per_gpu], name: s.label, itemStyle: { color: s.color } }],
          symbolSize: 6,
          label: {
            show: true,
            formatter: (p: unknown) => labelName(p as { data?: unknown; name?: string }),
            fontSize: 9,
            lineHeight: 12,
            color: "#4b5563",
            position: "right" as const,
          },
          labelLine: { show: false },
          labelLayout: endLayout,
          tooltip: { show: false },
          silent: true,
          clip: true,
          legendHoverLink: false,
        };
      }),
    ],
  };
  return (
    <section className={shell}>
      {head}
      {subtitle ? <p className="text-xs text-gray-400 mb-3">{subtitle}</p> : null}
      {staleNote}
      <div ref={box} className={stale ? "opacity-40 transition-opacity" : "transition-opacity"} aria-busy={stale ? true : undefined}>
        <HaChart option={option} height={chartH} onEvents={events} />
        {footer}
      </div>
    </section>
  );
}
