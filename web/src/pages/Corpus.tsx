import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { EChartsOption } from "echarts";
import { AXIS, HaChart, interactiveBits } from "../charts/HaChart";
import { legendRows, useWidth } from "../charts/useWidth";
import { useNearViewport } from "../charts/useNearViewport";
import { fetchJson } from "../api";
import { DataTable, type ColumnDef, type SortState } from "../DataTable";
import { BenchOp } from "../BenchOp";
import { fmtCompact } from "../format";

type Row = {
  id?: string;
  op?: string;
  signature?: string;
  framework?: string;
  framework_version?: string;
  kernel_impl?: string;
  gpu?: string;
  phase?: string;
  parallel?: string;
  shape_json?: string | Record<string, unknown>;
  /** The shape projected to the op's identifying axes — the envelope's. */
  physical_shape?: string;
  dtype_env_json?: string | Record<string, unknown>;
  /** The precision the KERNEL computed at, when not the cell's — a load-time
   *  dequantising path runs bf16 on an fp8 checkpoint. Empty: no fallback. */
  measured_dtype_env_json?: string | Record<string, unknown>;
  /** The host-side call that was timed, with its operands' shapes and
   *  dtypes, rendered where it was made. Empty on a provider that still
   *  returns a bare closure. */
  launch_signature?: string;
  model?: string;
  node_spec_id?: string;
  /** The node's name without its block template (the backend's). */
  node_label?: string;
  latency_us?: number;
  achieved_tflops?: number;
  achieved_bw_gbps?: number;
  peak_tflops_at_test?: number;
  peak_bw_at_test?: number;
  /** Achieved over the peaks recorded at test time, and FLOPs per byte
   *  moved — the backend's, null when a peak or the bandwidth is missing. */
  mfu?: number | null;
  mbu?: number | null;
  arith_intensity?: number | null;
  /** The backend's verdict against the recorded peaks: `ok`, or which
   *  ratio is over 1 (`mfu`, `mbu`, `mfu+mbu` — a suspect measurement). */
  roof_status?: string | null;
  samples?: number;
  date?: string;
  origin?: string;
  upload_id?: string;
  run_id?: string;
  /** Which clock timed the row (`benchmarks.TIMING_METHODS`). Null on a
   *  measurement timed before the clock recorded itself — see `PRE_DRAIN`. */
  timing_method?: string | null;
};

/** One physical op: the fastest measurement of it and what it beat. */
type Candidate = {
  id?: string; framework?: string; kernel_impl?: string; catalog_kernel?: string | null;
  latency_us?: number; model?: string;
};
/** What identifies a physical op — the estimator's lookup key. */
type EnvelopeKey = {
  signature: string; physical_shape: string; dtype_env_json: unknown;
  gpu: string; phase: string;
};
type EnvelopeItem = {
  winner: Row;
  n_candidates: number;
  /** How much slower the runner-up is, as a fraction; null without one. */
  gain: number | null;
  candidates: Candidate[];
  models: string[];
  /** Every candidate's catalog kernel. */
  kernels: string[];
  key: EnvelopeKey;
};

/** One page of one view, as /api/bench/browse answers it. */
type Browse<T> = {
  total?: number;
  items?: T[];
  facets?: Partial<Record<"model" | "signature" | "gpu" | "framework" | "dtype" | "phase" | "kernel", string[]>>;
  /** Workload: the batches measured, the contexts at the chosen batch (0 —
   *  ops without a context axis — last), the rows that name no batch, and
   *  the rows read for the chosen workload. */
  batches?: number[];
  contexts?: number[];
  unplaced?: number;
  candidates?: number;
  total_latency_us?: number;
  ops?: OpCurves[];
  /** Coverage: rows per registered kernel by library, and per framework
   *  the rows no registered kernel claims. */
  libraries?: Record<string, Record<string, number>>;
  unclassified?: Record<string, number>;
};

type View = "envelope" | "rows" | "workload" | "providers";
type Cover = "" | "measured" | "unmeasured" | "undeclared";

type Caps = {
  signature?: string;
  weight_dtypes?: string[];
  activation_dtypes?: string[];
  phases?: string[];
  shape_constraints?: string[];
  unsupported_features?: string[];
  disabled_reason?: string;
  // Why no cell of this corpus reaches a kernel whose declaration is right
  // (`Provider.unreached_reason`): the claim above is a promise, and an
  // unreachable one reads as coverage unless the reason is shown beside it.
  unreached_reason?: string;
};

type CatalogKernel = {
  kernel_impl: string;
  signature: string;
  fused_op?: string;
  min_sm?: number | null;
  capabilities: Caps | null;
};

type CatalogLib = {
  library: string;
  import_names?: string[];
  provisioning?: string;
  pip?: string;
  build_hint?: string;
  kernels: CatalogKernel[];
};

type Catalog = {
  libraries: CatalogLib[];
  kernel_count?: number;
  declared_count?: number;
};

const PAGE_SIZE = 50;
const VIEWS: View[] = ["envelope", "rows", "workload", "providers"];



function dtypeLabel(caps: Caps | null): string {
  if (!caps) return "—";
  const a = caps.activation_dtypes || [];
  const w = caps.weight_dtypes || [];
  if (!a.length && !w.length) return "—";
  if (!w.length || a.join() === w.join()) return a.join(", ") || w.join(", ");
  return `a ${a.join(", ")} · w ${w.join(", ")}`;
}

function phaseLabel(caps: Caps | null): string {
  if (!caps) return "—";
  return caps.phases?.length ? caps.phases.join(", ") : "any";
}

const RUNTIME_LIMIT_LABELS: Record<string, string> = {
  sm90_page_table: "SM90 paged KV / page_table unsupported",
};

function runtimeLimitLabel(token: string): string {
  return RUNTIME_LIMIT_LABELS[token] ?? token;
}

function capHasDtype(caps: Caps | null, dtype: string): boolean {
  if (!caps) return false;
  return (caps.activation_dtypes || []).includes(dtype)
    || (caps.weight_dtypes || []).includes(dtype);
}

function capHasPhase(caps: Caps | null, phase: string): boolean {
  if (!caps) return false;
  return !caps.phases?.length || caps.phases.includes(phase);
}


function asObj(j: unknown): Record<string, unknown> {
  if (j && typeof j === "object") return j as Record<string, unknown>;
  try { return JSON.parse(String(j || "{}")); } catch { return {}; }
}
/** A latency, or "–" when the row has none. A recorded refusal
 *  (`outcome="oom"`: the card could not hold the cell) carries no latency,
 *  and `Number(null)` is 0 — which drew the hardest cell on the page as the
 *  fastest thing on it. */
function fmtLatency(v: unknown): string {
  return v == null || v === "" ? "–" : Number(v).toFixed(1);
}
/** The note a latency carries when the row names no clock: it was timed
 *  before the harness billed a launch's own L2 write-back. Kept, not
 *  re-measured (the user's call); marked here and filtered nowhere. */
const PRE_DRAIN = "Measured before the L2 drain: a small kernel's latency may be off, "
  + "either way, by up to one L2 write-back (~17 µs on H100, ~82 µs on RTX 4090).";
/** A row's latency, with `PRE_DRAIN` on a measurement that names no clock. */
function latencyCell(r: Row): ReactNode {
  const text = fmtLatency(r.latency_us);
  if (r.latency_us == null || r.timing_method) return text;
  return <span className="text-amber-700" title={PRE_DRAIN}>{text}<sup>†</sup></span>;
}
function actDtype(j: unknown): string {
  return String(asObj(j).activation || "?");
}
/** `fp8_e4m3→bf16` when the kernel fell back, else "". The cell's precision
 *  is what the checkpoint ships; this says what actually multiplied. */
function dtypeFallback(cell: unknown, measured: unknown): string {
  const m = asObj(measured);
  if (!Object.keys(m).length) return "";
  const was = String(asObj(cell).activation || "?");
  return `${was}→${String(m.activation || m.weight || "?")}`;
}
function shortShape(j: unknown): string {
  return Object.entries(asObj(j)).map(([k, v]) => `${k}=${v}`).join(" ");
}
// The launch as a column: the op it dispatched and the operands it was
// handed, which is what says a row measured ITS cell rather than a slice of
// one or a cast of one. The full text (every operand, every scalar, and how
// many launches a composite baseline really is) stays on hover — these run
// to 200 characters and the table has to stay readable.
function shortLaunch(s?: string): string {
  const text = String(s || "");
  if (!text) return "";
  const many = text.match(/^(\d+) launches: /);
  const first = many ? text.slice(many[0].length) : text;
  const call = first.split(";")[0];
  const name = call.split("(")[0].replace(/^aten\./, "");
  const operands = (call.match(/\(([^)]*)\)/)?.[1] || "")
    .split(", ")
    .filter((p) => /\[/.test(p))
    .map((p) => p.replace(/float(32|16)/, "f$1").replace(/bfloat16/, "bf16")
                 .replace(/float8_e(4m3fn|5m2)/, "fp8"))
    .join(" ");
  return `${many ? `${many[1]}x ` : ""}${name} ${operands}`.trim();
}
function modelShort(m?: string): string {
  return String(m || "").split("/").pop() || "—";
}
function fmtPct(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x) || x <= 0) return "–";
  return x < 0.001 ? "<0.1%" : `${(x * 100).toFixed(x < 0.1 ? 1 : 0)}%`;
}
function fmtRate(x: number | null | undefined, digits: number): string {
  if (x == null || !(Number(x) > 0)) return "–";
  const v = Number(x);
  return v < 1 ? v.toPrecision(2) : v.toFixed(digits); // a small rate keeps two digits, not "0.0"
}
// Utilization: high is green; over 1 (faster than the peak) is suspect.
const utilClass = (x: number | null | undefined) =>
  x == null ? "text-gray-400" : x > 1 ? "text-amber-700 font-medium" : x >= 0.6 ? "text-green-600 font-medium" : "text-gray-500";

// The server pages these tables and sorts them: a sortable column's `key`
// is the sort the server is asked for (see bench_browse); the header only
// offers the toggle. A column the server does not sort by has none.
const serverSorted = () => null;

/** The measured row against its GPU — achieved rates and their share of the
 *  peaks — as the same four columns in every view. */
function rateColumns<T>(rowOf: (t: T) => Row): ColumnDef<T>[] {
  return [
    { key: "tflops", header: "TFLOP/s", align: "right", className: "text-gray-500", render: (t) => fmtRate(rowOf(t).achieved_tflops, 1), sortValue: serverSorted },
    { key: "gbps", header: "GB/s", align: "right", className: "text-gray-500", render: (t) => fmtRate(rowOf(t).achieved_bw_gbps, 0), sortValue: serverSorted },
    { key: "mfu", header: "MFU", align: "right", className: (t) => utilClass(rowOf(t).mfu), render: (t) => fmtPct(rowOf(t).mfu), sortValue: serverSorted },
    { key: "mbu", header: "MBU", align: "right", className: (t) => utilClass(rowOf(t).mbu), render: (t) => fmtPct(rowOf(t).mbu), sortValue: serverSorted },
  ];
}

/** A context as the workload picker names it: 0 is the ops that have no
 *  context axis (decode norms, GEMMs), shown with every context. */
function contextLabel(c: string): string {
  return c === "0" ? "no context axis" : c;
}

function readHash(): Record<string, string> {
  const p = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const out: Record<string, string> = {};
  p.forEach((v, k) => { if (v) out[k] = v; });
  return out;
}

function writeHash(h: Record<string, string>) {
  const p = new URLSearchParams();
  Object.entries(h).forEach(([k, v]) => { if (v) p.set(k, v); });
  const next = p.toString() ? `#${p.toString()}` : "";
  if (next !== window.location.hash) history.replaceState(null, "", `${window.location.pathname}${next}`);
}

function uniq(vals: Array<string | undefined>): string[] {
  return [...new Set(vals.filter((v): v is string => !!v))].sort();
}

/** One page of one view from /api/bench/browse — re-asked when the query
 *  changes (a superseded answer is dropped). `null` query: nothing to ask. */
function useBrowse<T>(query: Record<string, string | number | undefined> | null): {
  data: Browse<T> | null; error: string; loading: boolean;
} {
  const qs = query == null ? null : new URLSearchParams(
    Object.entries(query).filter(([, v]) => v !== undefined && v !== "").map(([k, v]) => [k, String(v)]),
  ).toString();
  const [state, setState] = useState<{ qs: string | null; data: Browse<T> | null; error: string }>(
    { qs: null, data: null, error: "" });
  useEffect(() => {
    if (qs == null) return;
    const ctl = new AbortController();
    const timer = setTimeout(() => {
      fetchJson<Browse<T>>(`/api/bench/browse?${qs}`, { signal: ctl.signal }).then(
        (data) => setState({ qs, data, error: "" }),
        (e: unknown) => {
          if (ctl.signal.aborted) return;
          setState({ qs, data: null, error: e instanceof Error ? e.message : String(e) });
        },
      );
    }, 120);
    return () => { clearTimeout(timer); ctl.abort(); };
  }, [qs]);
  const current = state.qs === qs;
  return { data: current ? state.data : null, error: current ? state.error : "", loading: qs != null && !current };
}

/** One envelope on a curve: its x (tokens per call), its fastest kernel's time and
 *  rates. */
type CurvePoint = {
  x: number; shape: string; framework?: string; kernel?: string; model?: string;
  latency_us?: number; tflops?: number; gbps?: number; mfu?: number | null; mbu?: number | null;
  /** The phase of the run it came from. */
  phase?: string;
};
/** One op shape with its token axes taken out, at one GPU and dtype, across
 *  phases: the same kernel problem (a GEMM's n, k) at several token counts
 *  — `axis` says which: B×s_query where the kernel folds them into rows,
 *  B×s_kv for attention (batch × context), B where the kernel keeps them
 *  apart, a grouped expert GEMM's n_tokens —
 *  ordered by them.
 *  The deployment that cut the shape is not part of it: one card runs the
 *  kernel the same way whichever parallel config produced the shape. */
type Curve = {
  gpu: string; dtype_env_json: unknown;
  /** What the x is, as the backend names it: an axis, or a product `a×b`. */
  axis: string;
  /** Where the curve turns compute-bound (the backend's `_saturation`). */
  saturation?: Saturation;
  shape: Record<string, unknown>; points: CurvePoint[];
  /** The lines through the points, as indices: one factor of x moving, the
   *  phase and the others held — two launches at one x are never joined.
   *  A point on no line is a line alone. */
  lines: number[][];
};
/** Where a curve's intensity reaches the card's ridge, in its tokens:
 *  `crossed` at `tokens`; `compute` from the first point; `memory` at every
 *  point; `no_flops` — no FLOP rate recorded. */
type Saturation = { state: "crossed" | "compute" | "memory" | "no_flops"; tokens?: number };
type OpCurves = {
  signature: string;
  /** More envelopes in view than the database reads for one op. */
  too_many?: boolean;
  points?: number;
  curves_total?: number;
  /** Envelopes not drawn: their shape has no token axis its signature's
   *  contract lays out, or their signature has no contract. */
  unplaced?: number;
  curves?: Curve[];
};

/** The roof a point is under: compute when its MFU is the larger of the
 *  two ratios, memory otherwise — achieved / min(peak FLOP/s, intensity ×
 *  peak BW) is whichever of them the binding roof divides. */
function boundOf(p: CurvePoint): "compute" | "memory" {
  const f = Number(p.mfu ?? NaN), m = Number(p.mbu ?? NaN);
  return (Number.isFinite(f) ? f : -1) >= (Number.isFinite(m) ? m : -1) ? "compute" : "memory";
}

/** The two utilizations, each drawn on its own chart. */
const CURVE_METRICS = [
  { key: "mfu", label: "MFU", empty: "no MFU recorded — this op's rows record no FLOP rate" },
  { key: "mbu", label: "MBU", empty: "no MBU recorded" },
] as const;

/** Colours for the kernels that won an op's points, in order of how many
 *  points each won (the validated categorical order, never cycled); the
 *  rest share `OTHER_KERNEL`. */
const KERNEL_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d"];
const OTHER_KERNEL = { name: "other kernels", color: "#6b7280" };

/** Each winning kernel of an op's curves, keyed by name, with its colour:
 *  the same in every GPU · dtype row and both charts of the op. */
function kernelColors(curves: Curve[]): (kernel: string | undefined) => { name: string; color: string } {
  const counts = new Map<string, number>();
  for (const c of curves) for (const p of c.points) {
    const k = p.kernel || "?";
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([k]) => k);
  const named = new Map(ranked.slice(0, KERNEL_COLORS.length).map((k, i) => [k, KERNEL_COLORS[i]]));
  return (kernel) => {
    const k = kernel || "?";
    const color = named.get(k);
    return color ? { name: k, color } : OTHER_KERNEL;
  };
}

/** A point's x as the axes it multiplies, from its own shape
 *  (`B=256 s_kv=4K`): equal products from different splits are different
 *  launches. Empty for an x of one axis. */
function splitOf(c: Curve, p: CurvePoint): string {
  const names = c.axis.split("×");
  if (names.length < 2) return "";
  const shape = asObj(p.shape);
  return names.map((n) => `${n}=${fmtCompact(Number(shape[n]))}`).join(" ");
}

/** A facet of an op's curves: the GPU and the dtype environment they ran
 *  at — the activation dtype, and every other slot whose dtype differs
 *  between the curves in view, so two environments never share a name. */
function curveGrouping(curves: Curve[]): (c: Curve) => string {
  const envs = curves.map((c) => asObj(c.dtype_env_json));
  const slots = uniq(envs.flatMap((e) => Object.keys(e))).filter((slot) =>
    slot !== "activation" && new Set(envs.map((e) => String(e[slot] ?? ""))).size > 1);
  return (c) => {
    const env = asObj(c.dtype_env_json);
    const rest = slots.map((slot) => `${slot} ${String(env[slot] ?? "–")}`);
    return [c.gpu, String(env.activation || "?"), ...rest].join(" · ");
  };
}

/** One GPU · dtype row of an op's curves: MFU and MBU charts over the same
 *  curves, and the curve the user pinned. A click on a curve pins it on
 *  both charts (the rest fade) and lists its points below; a click on a row,
 *  or ←/→, moves the selected point, which both charts ring; a second click
 *  on the curve, Esc or × unpins. */
function CurveFacet({ op, facet, curves, axes, initiallyOpen, kernelOf }: {
  op: string; facet: string; curves: Curve[]; axes: string[]; initiallyOpen: boolean;
  kernelOf: (kernel: string | undefined) => { name: string; color: string };
}) {
  // Rows past the first are drawn when opened: an op over every GPU and
  // dtype is otherwise a wall of charts.
  const [open, setOpen] = useState(initiallyOpen);
  // Both charts are drawn at the same width; one measures for the pair.
  const [box, width] = useWidth<HTMLDivElement>();
  const [pin, setPin] = useState<{ curve: number; point: number } | null>(null);
  const [nearRef, near] = useNearViewport();
  // Another answer (filters, scope) is other curves: nothing stays pinned.
  useEffect(() => { setPin(null); }, [curves]);
  const pinned = pin ? curves[pin.curve] ?? null : null;
  const satLineOf = (c: Curve) => {
    const sat = c.saturation;
    return sat?.state === "crossed" ? `predicted saturation ≈ ${fmtCompact(Math.round(sat.tokens ?? 0))} tokens`
      : sat?.state === "compute" ? "compute-bound from the first point"
      : sat?.state === "memory" ? "memory-bound at every point measured" : "no FLOP rate";
  };
  const step = (d: number) => setPin((p) => {
    if (!p || !curves[p.curve]) return p;
    const n = curves[p.curve].points.length;
    return { ...p, point: Math.min(n - 1, Math.max(0, p.point + d)) };
  });
  const onClick = (params: unknown) => {
    const q = params as { seriesIndex?: number; data?: { pi?: number } };
    // The curves are the first series; the kernels' legend keys follow.
    if (q.seriesIndex == null || q.seriesIndex >= curves.length) return;
    const curve = q.seriesIndex;
    const point = q.data?.pi ?? -1;
    setPin((p) => (p && p.curve === curve && (point < 0 || p.point === point) ? null
      : { curve, point: point >= 0 ? point : 0 }));
  };
  const chart = (metric: (typeof CURVE_METRICS)[number], width: number) => {
    const series = curves.map((c, i) => {
      const rest = shortShape(c.shape);
      const satLine = satLineOf(c);
      const isPinned = pin?.curve === i;
      const faded = pin != null && !isPinned;
      const tipOf = (p: CurvePoint) => [
        `<b>${op}</b> · ${c.axis}=${p.x}${splitOf(c, p) ? ` (${splitOf(c, p)})` : ""}`,
        `${rest}`,
        `${facet} · ${p.phase || "?"}`,
        `${p.framework || "?"} · ${p.kernel || "?"} · ${modelShort(p.model)}`,
        `${fmtRate(p.latency_us, 1)} µs · ${fmtRate(p.tflops, 1)} TFLOP/s · ${fmtRate(p.gbps, 0)} GB/s`,
        `MFU ${fmtPct(p.mfu)} · MBU ${fmtPct(p.mbu)} — ${boundOf(p)}-bound`,
        satLine,
      ].join("<br/>");
      const item = (pi: number): Array<Record<string, unknown>> => {
        const p = c.points[pi];
        const v = Number(p[metric.key] ?? NaN);
        if (!Number.isFinite(v)) return [];
        const selected = isPinned && pin?.point === pi;
        // The point's colour is the kernel that won it; its shape the roof
        // it is under — compute-bound a diamond, memory-bound a circle.
        return [{ value: [p.x, v], pi, symbol: boundOf(p) === "compute" ? "diamond" : "circle", tip: tipOf(p),
                  itemStyle: { color: kernelOf(p.kernel).color, opacity: faded ? 0.15 : 1,
                               ...(selected ? { borderColor: "#111827", borderWidth: 2.5 } : {}) },
                  ...(selected ? { symbolSize: 15 } : {}) }];
      };
      // One series per curve, its lines apart: a null ends a line.
      const data: Array<Record<string, unknown>> = c.lines.flatMap((line) => {
        const pts = line.flatMap(item);
        return pts.length ? [...pts, { value: [(pts[0].value as number[])[0], null] }] : [];
      });
      return {
        type: "line" as const,
        // Stable across a pin, so the update restyles the series in place.
        id: `${c.axis}|${JSON.stringify(c.shape)}|${i}`,
        // A line only says which points are one shape: grey, dark when pinned.
        name: "",
        color: isPinned ? "#111827" : "#9ca3af",
        lineStyle: { width: isPinned ? 2.5 : 1, opacity: faded ? 0.08 : isPinned ? 1 : 0.7 },
        symbolSize: 7,
        z: isPinned ? 10 : 2,
        // A shape measured at one token count is a point, not a line.
        showSymbol: true,
        connectNulls: false,
        emphasis: { focus: "series" as const, lineStyle: { width: 2.5, opacity: 1 } },
        ...(i === 0 ? { markLine: {
          silent: true, symbol: "none", label: { show: false },
          lineStyle: { color: "#9ca3af", type: "dashed" as const, width: 1 },
          data: [{ yAxis: 1 }],
        } } : {}),
        data,
      };
    });
    if (!series.some((sr) => sr.data.length)) return (
      <div className="text-xs text-gray-400 text-center flex items-center justify-center" style={{ height: 300 }}>
        {metric.empty}
      </div>
    );
    // The legend keys the kernels: an empty series per kernel in this chart.
    const kernels = uniq(curves.flatMap((c) => c.points.filter((p) => Number.isFinite(Number(p[metric.key] ?? NaN)))
      .map((p) => kernelOf(p.kernel).name)));
    const legendKeys = kernels.map((k) => ({
      type: "scatter" as const, id: `kernel|${k}`, name: k, data: [] as never[],
      color: k === OTHER_KERNEL.name ? OTHER_KERNEL.color : kernelOf(k).color,
    }));
    // Room under the axis for the legend's rows, at the width this chart is
    // actually drawn (a phone draws one chart per row, a desktop two).
    const rows = legendRows(kernels, width, { charPx: 6 });
    const option: EChartsOption = {
      ...interactiveBits({ name: `${op} ${facet} ${metric.label} by ${axes.join(" · ")}` }),
      grid: { left: 48, right: 16, top: 32, bottom: 46 + 18 * rows },
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const d = (params as { data?: { tip?: string } }).data;
          return d?.tip || "";
        },
      },
      legend: {
        type: "plain", bottom: 0, left: "center", width: "92%",
        itemGap: 12, itemWidth: 10, itemHeight: 8, textStyle: { fontSize: 10 },
        data: kernels, selectedMode: false,
      },
      xAxis: { type: "log", logBase: 2, min: "dataMin", max: "dataMax", name: axes.join(" · "), nameLocation: "middle", nameGap: 24, ...AXIS,
               // The axis ends at the data, so an end can fall between
               // powers of two (2.828…); a batch reads as a whole count.
               axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => fmtCompact(Math.round(v)) } },
      yAxis: {
        type: "value", min: 0, max: (v: { max: number }) => Math.max(1, Math.ceil(v.max * 10) / 10),
        name: metric.label, nameLocation: "middle", nameGap: 34, ...AXIS,
        axisLabel: { ...AXIS.axisLabel, formatter: (v: number) => `${Math.round(v * 100)}%` },
      },
      series: [...series, ...legendKeys],
    };
    // Merged, not replaced: a pin restyles the same series in place.
    return <HaChart option={option} height={282 + 18 * rows} onEvents={{ click: onClick }} merge />;
  };
  return (
    <div data-facet={facet} data-open={open ? "1" : "0"} ref={nearRef}>
      <button type="button" className="curve-facet-toggle text-11 text-gray-500 hover:text-gray-800 mb-1"
              onClick={() => { setOpen((v) => !v); setPin(null); }}>
        {open ? "▾" : "▸"} {facet} · {curves.length.toLocaleString()} shape{curves.length === 1 ? "" : "s"}
        {open ? <span className="ml-2 text-gray-400">· click a curve to pin it</span> : null}
      </button>
      {open ? (<>
      {/* Off screen, no option is built: an open facet per op is dozens. */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3" style={near ? undefined : { minHeight: 300 }}>
        {near && CURVE_METRICS.map((metric) => (
          <div key={metric.key} data-metric={metric.key} ref={metric.key === "mfu" ? box : undefined}>
            {chart(metric, width)}
          </div>
        ))}
      </div>
      {pinned && pin ? (
        <div
          className="curve-pin mt-2 border border-gray-200 rounded-lg p-3 bg-gray-50 outline-none focus:ring-2 focus:ring-blue-200"
          tabIndex={0}
          ref={(el) => el?.focus({ preventScroll: true })}
          onKeyDown={(e) => {
            if (e.key === "ArrowRight" || e.key === "ArrowDown") { step(1); e.preventDefault(); }
            else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { step(-1); e.preventDefault(); }
            else if (e.key === "Escape") setPin(null);
          }}
        >
          <div className="flex items-baseline justify-between gap-2 mb-2">
            <div className="text-xs text-gray-700">
              <span className="font-medium">{shortShape(pinned.shape) || "—"}</span>
              <span className="ml-2 text-gray-500">{facet} · {satLineOf(pinned)} · ←/→ to step, Esc to unpin</span>
            </div>
            <button type="button" className="curve-unpin text-gray-400 hover:text-gray-700 text-sm" onClick={() => setPin(null)}>×</button>
          </div>
          <div className="max-h-72 overflow-y-auto">
          <table className="w-full text-11">
            <thead className="text-gray-400 sticky top-0 bg-gray-50">
              <tr>
                <th className="text-right font-normal pr-3">{pinned.axis}</th>
                {pinned.axis.includes("×") ? <th className="text-left font-normal pr-3">split</th> : null}
                <th className="text-left font-normal pr-3">phase</th>
                <th className="text-left font-normal pr-3">kernel</th>
                <th className="text-left font-normal pr-3">model</th>
                <th className="text-right font-normal pr-3">µs</th>
                <th className="text-right font-normal pr-3">TFLOP/s</th>
                <th className="text-right font-normal pr-3">GB/s</th>
                <th className="text-right font-normal pr-3">MFU</th>
                <th className="text-right font-normal">MBU</th>
              </tr>
            </thead>
            <tbody>
              {pinned.points.map((p, pi) => (
                <tr
                  key={pi}
                  className={`cursor-pointer ${pi === pin.point ? "bg-blue-100 text-gray-900" : "text-gray-600 hover:bg-gray-100"}`}
                  ref={pi === pin.point ? (el) => el?.scrollIntoView({ block: "nearest" }) : undefined}
                  onClick={() => setPin({ ...pin, point: pi })}
                >
                  <td className="text-right pr-3 tabular-nums">{fmtCompact(p.x)}</td>
                  {pinned.axis.includes("×") ? <td className="pr-3 tabular-nums">{splitOf(pinned, p)}</td> : null}
                  <td className="pr-3">{p.phase || "—"}</td>
                  <td className="pr-3">
                    <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ background: kernelOf(p.kernel).color }} />
                    {p.framework || "?"} · {p.kernel || "?"}
                  </td>
                  <td className="pr-3">{modelShort(p.model)}</td>
                  <td className="text-right pr-3 tabular-nums">{fmtRate(p.latency_us, 1)}</td>
                  <td className="text-right pr-3 tabular-nums">{fmtRate(p.tflops, 1)}</td>
                  <td className="text-right pr-3 tabular-nums">{fmtRate(p.gbps, 0)}</td>
                  <td className={`text-right pr-3 tabular-nums ${utilClass(p.mfu)}`}>{fmtPct(p.mfu)}</td>
                  <td className={`text-right tabular-nums ${utilClass(p.mbu)}`}>{fmtPct(p.mbu)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </div>
      ) : null}
      </>) : null}
    </div>
  );
}

/** The curves of the ops in view: the curves view lists the ops (a
 *  signature each), and each op asks for its own curves when it comes near
 *  the viewport — one answer with every op's curves was tens of megabytes. */
function BatchCurves({ ops, query, empty }: {
  ops: OpCurves[]; query: Record<string, string | number | undefined>; empty: string;
}) {
  if (!ops.length) return <div className="text-center text-gray-400 text-sm py-8">{empty}</div>;
  return (
    <div id="curve-charts" className="space-y-8">
      {ops.map((o) => <OpCurveSection key={o.signature} signature={o.signature} query={query} />)}
    </div>
  );
}

/** One op's curves, asked (the curves view with the op's `signature`) once
 *  the section is near the viewport. */
function OpCurveSection({ signature, query }: {
  signature: string; query: Record<string, string | number | undefined>;
}) {
  const [nearRef, near] = useNearViewport();
  const got = useBrowse<never>(near ? { ...query, view: "curves", signature } : null);
  const o: OpCurves | null = got.data ? (got.data.ops?.[0] ?? { signature, curves: [] }) : null;
  const curves = o?.curves || [];
  const head = (
    <div className="text-xs font-medium text-gray-700 mb-1">
      {signature}
      {!o || o.too_many ? null : (
        <span className="ml-2 text-gray-400 font-normal">
          {curves.length < (o.curves_total || 0)
            ? `${curves.length.toLocaleString()} of ${(o.curves_total || 0).toLocaleString()} shapes (the longest curves)`
            : `${curves.length.toLocaleString()} shape${curves.length === 1 ? "" : "s"}`}
          {o.unplaced ? ` · ${o.unplaced.toLocaleString()} not drawn (no token axis, or no registered signature)` : ""}
        </span>
      )}
    </div>
  );
  // Until its answer lands an op holds a chart row's height, so only the
  // ops near the viewport ask at once.
  if (!o || o.too_many || !curves.length) return (
    <div data-op={signature} ref={nearRef} style={o ? undefined : { minHeight: 300 }}>
      {head}
      <div className="text-xs text-gray-400 py-6 text-center">
        {!o ? (got.error || "Loading…")
          : o.too_many ? "more than 50,000 op shapes — narrow the filters to draw this op"
          : o.unplaced ? "no op shape has a token axis its registered signature declares" : "no op shape has a measured utilization (MFU or MBU)"}
      </div>
    </div>
  );
  // Facets, the one with the most curves first (and open).
  const facetOf = curveGrouping(curves);
  const facets = uniq(curves.map(facetOf)).sort((a, b) =>
    curves.filter((c) => facetOf(c) === b).length - curves.filter((c) => facetOf(c) === a).length || a.localeCompare(b));
  const axes = uniq(curves.map((c) => c.axis));
  const kernelOf = kernelColors(curves);
  return (
    <div data-op={signature} ref={nearRef}>
      {head}
      <div className="space-y-4">
        {facets.map((facet, i) => (
          <CurveFacet key={facet} op={signature} facet={facet} axes={axes} initiallyOpen={i === 0} kernelOf={kernelOf}
                      curves={curves.filter((c) => facetOf(c) === facet)} />
        ))}
      </div>
    </div>
  );
}

export function Corpus() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogErr, setCatalogErr] = useState("");
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [view, setView] = useState<View>("envelope");
  const [model, setModel] = useState("");
  const [signature, setSignature] = useState("");
  const [gpu, setGpu] = useState("");
  const [framework, setFramework] = useState("");
  const [dtype, setDtype] = useState("");
  const [phase, setPhase] = useState("");
  const [source, setSource] = useState("shared");
  // "over": only rows faster than a peak they recorded; "ok": only the rest.
  const [check, setCheck] = useState<"" | "over" | "ok">("");
  const [batch, setBatch] = useState("");
  const [context, setContext] = useState("");
  const [upload, setUpload] = useState("");
  const [runId, setRunId] = useState("");
  // Whose batch `run` / `upload` names: an admin reviewing a nomination.
  const [owner, setOwner] = useState("");
  const [kernel, setKernel] = useState("");
  // One envelope's rows: its key, as JSON (the rows view's op-shape scope).
  const [envKey, setEnvKey] = useState("");
  const [cover, setCover] = useState<Cover>("");
  const [curvesHidden, setCurvesHidden] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [page, setPage] = useState(0);
  // The column the table is sorted by (each view's own order when none);
  // a new sort starts on the first page.
  const [sort, setSortState] = useState<SortState>(null);
  const setSort = (next: SortState) => { setSortState(next); setPage(0); };

  useEffect(() => {
    const h = readHash();
    if (VIEWS.includes(h.view as View)) setView(h.view as View);
    if (h.model) setModel(h.model);
    if (h.signature) setSignature(h.signature);
    if (h.gpu) setGpu(h.gpu);
    if (h.framework) setFramework(h.framework);
    if (h.dtype) setDtype(h.dtype);
    if (h.phase) setPhase(h.phase);
    if (h.source === "all") setSource("");
    else if (h.source) setSource(h.source);
    if (h.check === "over" || h.check === "ok") setCheck(h.check);
    if (h.upload) setUpload(h.upload);
    if (h.run) setRunId(h.run);
    if (h.owner) setOwner(h.owner);
    if (h.batch) setBatch(h.batch);
    if (h.context) setContext(h.context);
    if (h.kernel) setKernel(h.kernel);
    if (h.key) setEnvKey(h.key);
    if (h.cover === "measured" || h.cover === "unmeasured" || h.cover === "undeclared") setCover(h.cover);
    setHydrated(true);
    let cancelled = false;
    void fetchJson<Catalog>("/api/bench/providers").then(
      (j) => {
        if (cancelled) return;
        setCatalog({ libraries: j.libraries || [], kernel_count: j.kernel_count, declared_count: j.declared_count });
        setCatalogErr("");
        setCatalogLoading(false);
      },
      (e: unknown) => {
        if (cancelled) return;
        setCatalogErr(`Failed to load provider catalog: ${e instanceof Error ? e.message : String(e)}`);
        setCatalogLoading(false);
      },
    );
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    writeHash({ view, model, signature, gpu, framework, dtype, phase, source: source || "all", check, upload, run: runId, owner, batch, context, kernel, key: envKey, cover });
  }, [hydrated, view, model, signature, gpu, framework, dtype, phase, source, check, upload, runId, owner, batch, context, kernel, envKey, cover]);

  useEffect(() => { setSortState(null); }, [view]);
  useEffect(() => { setPage(0); }, [view, model, signature, gpu, framework, dtype, phase, source, check, upload, runId, batch, context, kernel, envKey, cover]);

  // Envelope and All rows share this filter set and both fetch one page.
  const scope = { source, run: runId, upload, owner, model, signature, gpu, framework, dtype, phase, kernel, check };
  const workloadScoped = !!(model && phase);
  const tableAsked = view === "envelope" || view === "rows" || (view === "workload" && workloadScoped);
  const table = useBrowse<Row & EnvelopeItem>(hydrated && tableAsked ? {
    ...scope, view, sort: sort?.key, dir: sort?.dir, page, page_size: PAGE_SIZE,
    ...(view === "rows" && envKey ? { key: envKey } : {}),
    ...(view === "workload" ? { batch: batch || undefined, context: batch ? context || undefined : undefined } : {}),
  } : null);
  // The workload picker starts at the smallest batch and its smallest real
  // context (the server lists them so).
  useEffect(() => {
    if (view !== "workload" || !table.data) return;
    const bs = table.data.batches || [];
    if (!batch && bs.length) setBatch(String(bs[0]));
    const cs = table.data.contexts || [];
    if (batch && !context && cs.length) setContext(String(cs[0]));
  }, [view, table.data, batch, context]);
  // The curves' scope: the ops it lists, and each op's curves under it.
  const curvesQuery = { ...scope, ...(view === "rows" && envKey ? { key: envKey } : {}) };
  const curves = useBrowse<never>(hydrated && (view === "envelope" || view === "rows") && !curvesHidden
    ? { ...curvesQuery, view: "curves" } : null);
  const coverage = useBrowse<never>(hydrated && view === "providers"
    ? { source, run: runId, upload, owner, gpu, dtype, phase, view: "coverage" } : null);
  // A view that lists nothing until it is scoped still needs the dropdowns
  // that scope it: their values come from one envelope entry's facets.
  const unscopedFacets = useBrowse<never>(hydrated && !tableAsked && view !== "providers"
    ? { ...scope, view: "envelope", page: 0, page_size: 1 } : null);
  const shown = view === "providers" ? coverage : table;
  const facets = shown.data?.facets || table.data?.facets || coverage.data?.facets || unscopedFacets.data?.facets || {};
  const noneMatch = "No measurements match.";

  const libraries = catalog?.libraries || [];
  // The active selection is unioned in so a select never renders blank while
  // its filter still applies (e.g. a hash value with no matching rows).
  // Providers also lists the catalog, so an unmeasured kernel is findable.
  const models = uniq([...(facets.model || []), model]);
  const sigs = uniq([
    ...(facets.signature || []),
    ...(view === "providers" ? libraries.flatMap((l) => l.kernels.map((k) => k.signature)) : []),
    signature,
  ]);
  const gpus = uniq([...(facets.gpu || []), gpu]);
  const fws = uniq([
    ...(facets.framework || []),
    ...(view === "providers" ? libraries.map((l) => l.library) : []),
    framework,
  ]);
  const dtypes = uniq([
    ...(facets.dtype || []),
    ...libraries.flatMap((l) => l.kernels.flatMap((k) => [
      ...(k.capabilities?.activation_dtypes || []),
      ...(k.capabilities?.weight_dtypes || []),
    ])),
    dtype,
  ]);
  const batches = (table.data?.batches || []).map(String);
  const contexts = (table.data?.contexts || []).map(String);

  const tabCls = (k: View) =>
    `px-4 py-1.5 text-sm ${view === k ? "bg-amber-500 text-white" : "bg-white text-gray-700"}`;

  const total = table.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const loadingTable = table.loading && !table.data;
  let count = "";
  let tableEl: ReactNode = null;

  if (view === "envelope") {
    const items = (table.data?.items || []) as EnvelopeItem[];
    count = table.data ? `${total.toLocaleString()} physical ops` : "";
    // Leads with the winner's model — the one the model filter matches —
    // then counts the other models sharing this physical op.
    const modelLabel = (e: EnvelopeItem) => {
      const ms = e.models || [];
      const lead = e.winner.model || ms[0];
      if (!lead) return "—";
      const others = ms.filter((m) => m !== lead).length;
      return others ? `${modelShort(lead)} +${others}` : modelShort(lead);
    };
    const cols: ColumnDef<EnvelopeItem>[] = [
      { key: "model", className: "text-gray-700 text-11", render: modelLabel, cellProps: (e) => ({ title: (e.models || []).join(", ") }), sortValue: null },
      { key: "op", className: "text-gray-900", render: ({ winner }) => winner.signature, sortValue: null },
      { key: "gpu", render: ({ winner }) => winner.gpu, sortValue: null },
      { key: "phase", render: ({ winner }) => winner.phase, sortValue: null },
      { key: "shape", wrap: true, className: "font-mono text-11 text-gray-500", render: ({ winner }) => shortShape(winner.physical_shape ?? winner.shape_json), sortValue: null },
      { key: "dtype", render: ({ winner }) => actDtype(winner.dtype_env_json), sortValue: null },
      {
        key: "winner", sortValue: null,
        render: ({ winner }) => (
          <>
            <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-800 text-11">{winner.framework}</span>
            {winner.origin === "mine" ? <span title="your upload" className="text-green-700"> ★</span> : null}
          </>
        ),
      },
      { key: "latency", header: "winner µs", align: "right", className: "text-gray-900 font-medium", render: ({ winner }) => latencyCell(winner), sortValue: serverSorted },
      ...rateColumns(({ winner }: EnvelopeItem) => winner),
      { key: "gain", header: "vs 2nd", align: "right", className: "text-green-700", render: ({ gain }) => (gain == null ? "—" : `${(gain * 100).toFixed(0)}% faster`), sortValue: serverSorted },
      {
        key: "candidates", align: "right", wrap: true, className: "font-mono text-11 text-gray-500",
        render: (e) => (
          <button type="button" data-open-rows className="text-blue-600 hover:underline text-right" title="List this op shape's rows"
            onClick={() => { setEnvKey(JSON.stringify(e.key)); setView("rows"); }}>
            {e.candidates.slice(0, 3).map((x) => `${x.framework}/${x.kernel_impl} ${Number(x.latency_us || 0).toFixed(1)}µs`).join("  ·  ")
              + (e.n_candidates > 3 ? `  · +${e.n_candidates - 3} more` : "")}
          </button>
        ),
        sortValue: serverSorted,
      },
    ];
    tableEl = (
      <DataTable
        columns={cols}
        rows={items}
        sort={sort}
        onSort={setSort}
        rowKey={(e) => e.winner.id || ""}
        framed={false}
        tableClassName="text-sm"
        loading={loadingTable}
        empty={loadingTable ? "Loading…" : noneMatch}
      />
    );
  } else if (view === "rows") {
    const items = (table.data?.items || []) as Row[];
    count = table.data ? `${total.toLocaleString()} rows` : "";
    const cols: ColumnDef<Row>[] = [
      { key: "model", className: "text-gray-700 text-11", render: (r) => modelShort(r.model), sortValue: serverSorted },
      {
        key: "source", sortValue: null,
        render: (r) => r.origin === "mine"
          ? <span className="px-1.5 py-0.5 rounded bg-green-100 text-green-800 text-10">mine</span>
          : <span className="px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 text-10">shared</span>,
      },
      { key: "layer", className: "text-gray-400 font-mono text-11", render: (r) => r.node_label || "—", cellProps: (r) => ({ title: r.node_spec_id }), sortValue: serverSorted },
      { key: "op", className: "text-gray-900", render: (r) => <BenchOp signature={r.signature} op={r.op} />, sortValue: serverSorted },
      { key: "framework", render: (r) => r.framework, sortValue: serverSorted },
      { key: "version", className: "text-gray-500", render: (r) => r.framework_version, sortValue: serverSorted },
      { key: "kernel", className: "text-gray-500 font-mono text-11", render: (r) => r.kernel_impl, cellProps: (r) => ({ title: r.launch_signature || undefined }), sortValue: serverSorted },
      // What was actually timed. A row whose rate looks impossible is read
      // here first: the operands' dtypes and extents are what say whether
      // the launch is the cell's at all, and a composite baseline shows how
      // many launches it is against a fused kernel's one.
      {
        key: "launch", wrap: true, className: "font-mono text-11 text-gray-500",
        render: (r) => shortLaunch(r.launch_signature) || "—",
        cellProps: (r) => ({ title: r.launch_signature || undefined }),
        sortValue: serverSorted,
      },
      { key: "gpu", render: (r) => r.gpu, sortValue: serverSorted },
      { key: "phase", render: (r) => r.phase, sortValue: serverSorted },
      { key: "shape", wrap: true, className: "font-mono text-11 text-gray-500", render: (r) => shortShape(r.shape_json), sortValue: serverSorted },
      {
        key: "dtype", sortValue: serverSorted,
        render: (r) => {
          const fallback = dtypeFallback(r.dtype_env_json, r.measured_dtype_env_json);
          return fallback
            ? <span className="text-amber-700" title={`kernel computed at ${fallback.split("→")[1]}, not the cell's precision`}>{fallback}</span>
            : actDtype(r.dtype_env_json);
        },
      },
      { key: "latency", header: "latency µs", align: "right", className: "text-gray-900", render: (r) => latencyCell(r), sortValue: serverSorted },
      ...rateColumns((r: Row) => r),
      { key: "ai", header: "AI", align: "right", className: "text-gray-400", render: (r) => fmtRate(r.arith_intensity, 1), sortValue: serverSorted },
      { key: "samples", align: "right", className: "text-gray-400", render: (r) => r.samples ?? "–", sortValue: serverSorted },
      { key: "date", className: "text-gray-400", render: (r) => r.date, sortValue: serverSorted },
    ];
    tableEl = (
      <DataTable
        columns={cols}
        rows={items}
        sort={sort}
        onSort={setSort}
        rowKey={(r) => r.id || ""}
        framed={false}
        tableClassName="text-sm"
        loading={loadingTable}
        empty={loadingTable ? "Loading…" : noneMatch}
      />
    );
  } else if (view === "workload") {
    const items = (table.data?.items || []) as Row[];
    const cols: ColumnDef<Row>[] = [
      { key: "layer", className: "text-gray-400 font-mono text-11", render: (r) => r.node_label || "—", cellProps: (r) => ({ title: r.node_spec_id }), sortValue: serverSorted },
      {
        key: "op", sortValue: serverSorted,
        className: "text-gray-900",
        render: (r) => (
          <>
            <BenchOp signature={r.signature} op={r.op} />{" "}
            {r.origin === "mine"
              ? <span className="px-1.5 py-0.5 rounded bg-green-100 text-green-800 text-10">mine</span>
              : <span className="px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 text-10">shared</span>}
          </>
        ),
      },
      { key: "shape", wrap: true, className: "font-mono text-11 text-gray-500", render: (r) => shortShape(r.shape_json), sortValue: serverSorted },
      { key: "dtype", render: (r) => actDtype(r.dtype_env_json), sortValue: null },
      { key: "kernel", className: "text-gray-500 font-mono text-11", render: (r) => r.kernel_impl, sortValue: serverSorted },
      { key: "latency", header: "latency µs", align: "right", className: "text-gray-900", render: (r) => latencyCell(r), sortValue: serverSorted },
      ...rateColumns((r: Row) => r),
    ];
    const asked = workloadScoped && !!batch && !!context;
    if (asked && table.data && total) {
      count = `${total.toLocaleString()} op shapes measured at this workload · Σ ${Number(table.data.total_latency_us || 0).toFixed(1)} µs`;
    }
    tableEl = (
      <DataTable
        columns={cols}
        rows={items}
        sort={sort}
        onSort={setSort}
        rowKey={(r) => r.id || ""}
        framed={false}
        tableClassName="text-sm"
        empty={!workloadScoped ? "Pick a model and a phase to see its measured ops by workload."
          : !asked ? (loadingTable ? "Loading…" : "No batch of this model and phase is measured.")
            : loadingTable ? "Loading…" : "No measurements for this workload."}
      />
    );
  }

  const measuredRows = coverage.data?.libraries || {};
  const unclassified = Object.entries(coverage.data?.unclassified || {}).filter(([, n]) => n > 0);
  type KernelHit = { lib: CatalogLib; k: CatalogKernel; cells: number; status: Exclude<Cover, ""> };
  const providerKernels: KernelHit[] = [];
  for (const lib of libraries) {
    if (framework && lib.library !== framework) continue;
    for (const k of lib.kernels) {
      if (signature && k.signature !== signature) continue;
      if (dtype && !capHasDtype(k.capabilities, dtype)) continue;
      if (phase && !capHasPhase(k.capabilities, phase)) continue;
      const n = measuredRows[lib.library]?.[k.kernel_impl] || 0;
      const st: KernelHit["status"] = k.capabilities == null ? "undeclared" : n > 0 ? "measured" : "unmeasured";
      if (cover && st !== cover) continue;
      providerKernels.push({ lib, k, cells: n, status: st });
    }
  }
  if (view === "providers") {
    const m = providerKernels.filter((x) => x.status === "measured").length;
    const u = providerKernels.filter((x) => x.status === "unmeasured").length;
    const d = providerKernels.filter((x) => x.status === "undeclared").length;
    count = `${providerKernels.length} kernels · ${m} measured · ${u} unmeasured · ${d} undeclared`;
  }
  const envKeyLabel = (() => {
    if (!envKey) return "";
    try {
      const k = JSON.parse(envKey) as EnvelopeKey;
      return `${k.signature} ${shortShape(k.physical_shape)} · ${k.gpu} · ${k.phase}`;
    } catch { return "one op shape"; }
  })();

  const libsShown: CatalogLib[] = [];
  for (const hit of providerKernels) {
    if (!libsShown.includes(hit.lib)) libsShown.push(hit.lib);
  }

  const openKernel = (lib: string, k: CatalogKernel) => {
    setKernel(k.kernel_impl);
    setSignature(k.signature);
    setFramework(lib);
    setCover("");
    setView("envelope");
  };

  const curveOps = useMemo(() => curves.data?.ops || [], [curves.data]);
  const error = shown.error;

  return (
    <main className="w-full page-shell py-6 sm:py-8 space-y-5 flex-1">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Kernel Bench</h1>
        <p className="mt-1 text-sm text-gray-500">
          Shared kernel timings, the per-op envelope, and the provider catalog.
        </p>
        {error ? <p id="browse-error" className="mt-1 text-xs text-red-600">{error}</p> : null}
        {view === "workload" && table.data?.unplaced ? (
          <p id="workload-unplaced" className="mt-1 text-xs text-gray-500">
            {table.data.unplaced.toLocaleString()} measurements of this model and phase name no batch (e.g. grouped GEMM, measured per expert) and are not listed by workload; the envelope and rows views have them.
          </p>
        ) : null}
        {view === "providers" && catalogErr ? <p className="mt-1 text-xs text-red-600">{catalogErr}</p> : null}
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <div className="inline-flex max-w-full rounded-lg border border-gray-200 overflow-x-auto">
          <button type="button" id="tab-envelope" className={tabCls("envelope")} onClick={() => setView("envelope")}>Envelope</button>
          <button type="button" id="tab-rows" className={tabCls("rows")} onClick={() => { setEnvKey(""); setView("rows"); }}>All rows</button>
          <button type="button" id="tab-workload" className={tabCls("workload")} onClick={() => setView("workload")}>By workload</button>
          <button type="button" id="tab-providers" className={tabCls("providers")} onClick={() => setView("providers")}>Providers</button>
        </div>
        {view !== "providers" ? (
          <select id="f-model" className="max-w-full px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={model} onChange={(e) => { setModel(e.target.value); setBatch(""); setContext(""); }}>
            <option value="">All models</option>
            {models.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        ) : null}
        <select id="f-signature" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={signature} onChange={(e) => setSignature(e.target.value)}>
          <option value="">All ops</option>
          {sigs.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <select id="f-gpu" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={gpu} onChange={(e) => setGpu(e.target.value)}>
          <option value="">All GPUs</option>
          {gpus.map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
        <select id="f-framework" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={framework} onChange={(e) => setFramework(e.target.value)}>
          <option value="">All frameworks</option>
          {fws.map((f) => <option key={f} value={f}>{f}</option>)}
        </select>
        <select id="f-dtype" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={dtype} onChange={(e) => setDtype(e.target.value)}>
          <option value="">All dtypes</option>
          {dtypes.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <select id="f-phase" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={phase} onChange={(e) => { setPhase(e.target.value); setBatch(""); setContext(""); }}>
          <option value="">prefill+decode</option>
          <option value="prefill">prefill</option>
          <option value="decode">decode</option>
        </select>
        <select id="f-source" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={source} onChange={(e) => setSource(e.target.value)}>
          <option value="">All sources</option>
          <option value="shared">Shared</option>
          <option value="mine">Mine</option>
        </select>
        <select id="f-check" title="Measurements faster than a peak they recorded (MFU or MBU over 100%) are suspect"
          className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={check}
          onChange={(e) => setCheck(e.target.value as "" | "over" | "ok")}>
          <option value="">All checks</option>
          <option value="over">Over peak only</option>
          <option value="ok">Within peak only</option>
        </select>
        {view === "providers" ? (
          <select id="f-cover" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={cover} onChange={(e) => setCover(e.target.value as Cover)}>
            <option value="">All coverage</option>
            <option value="measured">measured</option>
            <option value="unmeasured">unmeasured</option>
            <option value="undeclared">undeclared</option>
          </select>
        ) : null}
        {envKey && view === "rows" ? (
          <button type="button" id="f-key-clear" className="px-2 py-1 text-11 rounded border border-amber-200 bg-amber-50 text-amber-800" onClick={() => setEnvKey("")}>
            {envKeyLabel} ×
          </button>
        ) : null}
        {kernel ? (
          <button type="button" id="f-kernel-clear" className="px-2 py-1 text-11 rounded border border-amber-200 bg-amber-50 text-amber-800" onClick={() => setKernel("")}>
            {kernel} ×
          </button>
        ) : null}
        {view === "workload" && (
          <span id="wl-controls" className="inline-flex items-center gap-2">
            <select id="f-batch" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={batch} onChange={(e) => { setBatch(e.target.value); setContext(""); }}>
              <option value="">batch</option>
              {[...batches, ...(batch && !batches.includes(batch) ? [batch] : [])].map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
            <select id="f-context" className="px-2 py-1.5 text-sm border border-gray-300 rounded bg-white" value={context} onChange={(e) => setContext(e.target.value)}>
              <option value="">context</option>
              {[...contexts, ...(context && !contexts.includes(context) ? [context] : [])].map((c) => <option key={c} value={c}>{contextLabel(c)}</option>)}
            </select>
          </span>
        )}
        <span id="count" className="text-xs text-gray-400 ml-auto">{count}</span>
      </div>

      {view === "providers" ? (
        <section id="providers-section" className="space-y-4">
          {unclassified.length ? (
            <p id="coverage-unclassified" className="text-xs text-gray-500">
              Rows whose kernel no registered provider names:{" "}
              {unclassified.map(([fw, n]) => `${fw || "—"} ${n.toLocaleString()}`).join(" · ")}
            </p>
          ) : null}
          {catalogLoading || coverage.loading ? (
            <p className="text-sm text-gray-400 py-8 text-center">Loading…</p>
          ) : catalogErr && !libraries.length ? (
            <p className="text-sm text-gray-400 py-8 text-center">{catalogErr}</p>
          ) : !libsShown.length ? (
            <p className="text-sm text-gray-400 py-8 text-center">No providers match these filters.</p>
          ) : libsShown.map((lib) => {
            const hits = providerKernels.filter((x) => x.lib === lib);
            const cols: ColumnDef<KernelHit>[] = [
              {
                key: "kernel",
                className: "font-mono text-11",
                sortValue: ({ k }) => k.kernel_impl,
                render: ({ k }) => (
                  <>
                    <button
                      type="button"
                      className="text-blue-600 hover:underline text-left"
                      onClick={() => openKernel(lib.library, k)}
                    >
                      {k.kernel_impl}
                    </button>
                    {k.fused_op ? <span className="ml-1 text-10 text-gray-400">{k.fused_op}</span> : null}
                  </>
                ),
              },
              { key: "signature", className: "font-mono text-11 text-gray-700", render: ({ k }) => k.signature, sortValue: ({ k }) => k.signature },
              { key: "dtypes", className: "text-11 text-gray-600", render: ({ k }) => dtypeLabel(k.capabilities), sortValue: ({ k }) => dtypeLabel(k.capabilities) },
              { key: "phases", className: "text-11 text-gray-600", render: ({ k }) => phaseLabel(k.capabilities), sortValue: ({ k }) => phaseLabel(k.capabilities) },
              {
                key: "constraints", wrap: true,
                className: "text-11 text-gray-400 font-mono",
                render: ({ k }) => (k.capabilities?.shape_constraints || []).join("; ") || "—",
                cellProps: ({ k }) => ({ title: (k.capabilities?.shape_constraints || []).join("; ") }),
                sortValue: ({ k }) => (k.capabilities?.shape_constraints || []).join("; "),
              },
              {
                key: "runtime limits", wrap: true,
                className: "text-11 text-amber-700",
                render: ({ k }) => (k.capabilities?.unsupported_features || []).map(runtimeLimitLabel).join("; ") || "—",
                cellProps: ({ k }) => ({ title: (k.capabilities?.unsupported_features || []).map(runtimeLimitLabel).join("; ") }),
                sortValue: ({ k }) => (k.capabilities?.unsupported_features || []).map(runtimeLimitLabel).join("; "),
              },
              { key: "min SM", align: "right", className: "text-11 text-gray-600", render: ({ k }) => k.min_sm ?? "—", sortValue: ({ k }) => k.min_sm },
              {
                key: "coverage", wrap: true,
                sortValue: ({ status }) => status,
                render: ({ k, cells: n, status }) => (
                  <>
                    <span className={`px-1.5 py-0.5 rounded text-10 ${
                      status === "measured" ? "bg-green-100 text-green-800"
                        : status === "undeclared" ? "bg-gray-100 text-gray-600"
                          : "bg-amber-100 text-amber-800"
                    }`}>{status}</span>
                    {status === "measured" ? <span className="ml-1 text-11 text-gray-500">{n.toLocaleString()} rows</span> : null}
                    {k.capabilities?.disabled_reason
                      ? <span className="ml-1 text-10 text-red-600">{k.capabilities.disabled_reason}</span>
                      : null}
                    {k.capabilities?.unreached_reason
                      ? <span className="ml-1 text-10 text-gray-500">{k.capabilities.unreached_reason}</span>
                      : null}
                  </>
                ),
              },
            ];
            return (
              <div key={lib.library} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                <header className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-gray-100 bg-gray-50">
                  <span className="font-mono text-sm text-gray-900">{lib.library}</span>
                  {lib.provisioning === "source_build"
                    ? <span className="px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 text-10">source-build</span>
                    : <span className="px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 text-10">pip</span>}
                  {lib.pip ? <span className="text-11 text-gray-500 font-mono">{lib.pip}</span> : null}
                  {lib.build_hint ? <span className="text-11 text-gray-500">{lib.build_hint}</span> : null}
                  {lib.import_names?.length
                    ? <span className="text-11 text-gray-400 font-mono ml-auto">import {lib.import_names.join(", ")}</span>
                    : null}
                </header>
                <DataTable
                  columns={cols}
                  rows={hits}
                  rowKey={({ k }) => k.kernel_impl}
                  framed={false}
                  rowClassName={({ k }) => (kernel === k.kernel_impl ? "bg-amber-50" : undefined)}
                  rowProps={({ k }) => ({ id: `kernel-${lib.library}-${k.kernel_impl}` })}
                />
              </div>
            );
          })}
        </section>
      ) : (
        <section id="table-section" className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-x-auto">
          {tableEl}
          {total > PAGE_SIZE ? (
            <div id="table-pager" className="flex items-center justify-end gap-3 px-3 py-2 text-xs text-gray-500 border-t border-gray-100">
              <span>{(page * PAGE_SIZE + 1).toLocaleString()}–{Math.min(total, (page + 1) * PAGE_SIZE).toLocaleString()} / {total.toLocaleString()}</span>
              <button
                type="button"
                className="px-2 py-1 border border-gray-200 rounded bg-white disabled:opacity-40"
                disabled={page <= 0}
                onClick={() => setPage((n) => Math.max(0, n - 1))}
              >
                Prev
              </button>
              <button
                type="button"
                className="px-2 py-1 border border-gray-200 rounded bg-white disabled:opacity-40"
                disabled={page >= pages - 1}
                onClick={() => setPage((n) => n + 1)}
              >
                Next
              </button>
            </div>
          ) : null}
        </section>
      )}

      {view !== "workload" && view !== "providers" && (
        <section id="curves-section" className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 sm:p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-x-2 mb-1">
            <h2 className="text-sm font-semibold text-gray-800">Utilization by tokens</h2>
            <button type="button" id="curves-toggle" className="text-11 text-gray-500 hover:text-amber-600" onClick={() => setCurvesHidden((v) => !v)}>
              {curvesHidden ? "show" : "hide"}
            </button>
          </div>
          <p className="text-11 text-gray-500 mb-3">
            Per op, one row per GPU · dtype (one roofline each), MFU on the left and MBU on the right. Each point is an op
            shape&apos;s fastest kernel; a ◆ is compute-bound, a ● memory-bound (whichever of MFU and MBU is larger). The x is the
            tokens per call, B×s_query, where the kernel folds them into one row axis (GEMM, norms, activations; in decode the
            batch); attention runs along B×s_kv, the KV tokens a call reads (in decode, batch × context), one curve per s_query —
            sparse attention, which reads only its top-k keys, runs along B; other kernels that keep B and s_query apart run
            along B; a grouped
            expert GEMM runs along its routed tokens, n_tokens. A line joins one kernel problem (a GEMM&apos;s n, k) across token
            counts, whichever phase it ran in or deployment cut the shape (shapes are per rank). A point&apos;s colour is the kernel
            that won it (an op&apos;s most frequent winners get a colour each, the rest are grey); the line is grey. The predicted
            saturation — the tokens where the line turns compute-bound — is in each point&apos;s tooltip and the pinned table. Click a line to pin it and step through its points. The row with the most shapes is open; click a
            row&apos;s title to open the others.
          </p>
          {curvesHidden ? null : curves.data ? <BatchCurves ops={curveOps} query={curvesQuery} empty={noneMatch} /> : (
            <p className="text-xs text-gray-400 py-6 text-center">{curves.error || "Loading…"}</p>
          )}
        </section>
      )}
    </main>
  );
}
