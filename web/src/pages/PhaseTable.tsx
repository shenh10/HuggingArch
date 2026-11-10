import { useEffect, useRef, useState, type ReactNode } from "react";
import { fetchJson } from "../api";
import { DataTable, tbl, type ColumnDef } from "../DataTable";
import { compressRanges, escapeHtml, formatBytes, formatFlops, fmtTime } from "../format";
import { BOUND_PILL, BOUND_TEXT } from "../bound";
import { FormulaHover, HOVER_HINT } from "../FormulaHover";
import { tipHandlers } from "../Tip";
import { formulaHTML, type FormulaFields } from "../tooltip";
import { BenchJobsExport } from "./BenchExport";
import { BlockElkDag, type BlockEdge, type BlockNode } from "./BlockDag";
import {
  compactFusedSubgraph,
  computeGraphQuery,
  findFusedCompound,
  nodesById,
  graphFailureMessage,
  type GraphResp,
  type GraphTarget,
} from "./ComputeGraph";
import { visibleSpan, type DeclarationRow } from "../declarations";
import { foldDepth, foldKey, hasFoldChildren, visibleFold } from "../moduleFold";
import { completeTotal, coverageText, fmtMeasured, groupMeasured, partialCoverageText, phaseCoverage, rowMeasured, stateCounts, stateKeysText } from "../measured";
import type { AbsorbedOp, Estimate, FusionDelta, ModuleFoldRow, Occurrence, OpTableTotal, PerOpRow, PhaseMetrics, RowBound, SourceBadge, SpeculativeDraftTerm } from "./ResultSummary";
import { SourceAnchor } from "../SourceAnchor";

/** The step a per-op table describes: a language-model phase, or an
 *  encoder's encode step. */
export type PhaseKey = "prefill" | "decode" | "encode";

/** Shape cell in the app's `[1,1,7168]` form (the compute graph's
 *  `tensor_out` convention), several tensors joined by `+`. */
/** A shape cell's tensors, each kept whole: the line may break between
 *  tensors (after `+`, around `→`), never inside one, so a fused op's
 *  three inputs stack instead of widening the column to one long line. */
function tensorList(v: unknown): string[] {
  return Array.isArray(v) && v.length && Array.isArray(v[0]) ? v.map((x) => fmtShape(x)) : [fmtShape(v)];
}

function ShapeFlow({ from, to }: { from: unknown; to?: unknown }) {
  const side = (v: unknown) =>
    tensorList(v).map((t, i) => (
      <span key={i}>{i > 0 ? <span className="text-gray-400">+ </span> : null}<span className="whitespace-nowrap">{t}</span>{" "}</span>
    ));
  return (
    <>
      {side(from)}
      {to !== undefined ? <><span className="text-gray-400">→ </span>{side(to)}</> : null}
    </>
  );
}

function fmtShape(v: unknown): string {
  if (v == null) return "—";
  if (Array.isArray(v)) {
    if (v.length && Array.isArray(v[0])) {
      return v.map((x) => fmtShape(x)).join("+");
    }
    return `[${v.map((x) => String(x)).join(",")}]`;
  }
  return String(v);
}

function shapeStr(m: PhaseMetrics): string {
  const parts = [
    `${m.batch ?? 0} req × ${m.seq_effective ?? 0} tok`,
    `cache ${(m.kv_cache_length || 0).toFixed(0)}`,
  ];
  const b = m.binding || {};
  if ((b.s_prefix_cached || 0) > 0) {
    parts.push(`prefix ${b.s_prefix_cached}`);
  }
  if ((b.speculative_draft_tokens || 0) > 0) {
    parts.push(`K=${b.speculative_draft_tokens} draft tok α=${(b.acceptance_rate || 1).toFixed(2)}`);
  }
  if (b.mla_absorbed) parts.push("absorbed");
  return parts.join(" · ");
}

function tokPerSec(m: PhaseMetrics, track: "sol" | "partial" = "sol"): string {
  const t = m.throughput?.[track];
  const v = t?.per_gpu;
  return v != null && Number.isFinite(v) ? `${v.toFixed(0)} tok/s` : "—";
}

/** Plain-text tip through the shared popover (pre-wrap), not the native title. */
const textTip = (s?: string) =>
  tipHandlers(s ? `<div style="white-space:pre-wrap">${escapeHtml(s)}</div>` : undefined);

function foldAsPerOp(row: ModuleFoldRow): PerOpRow {
  return {
    name: row.name,
    op: row.op,
    aggregation_id: row.aggregation_id,
    display_path: row.display_path,
    shape_in: row.shape_in,
    shape_out: row.shape_out,
    flops_per_rank: row.flops_per_rank,
    access_bytes_per_rank: row.access_bytes_agg,
    t_compute_s: row.t_compute_s,
    t_memory_s: row.t_memory_s,
    t_comm_s: row.t_comm_s,
    t_sol_s: row.t_sol_s,
    t_op_s: row.t_op_s,
    bound: row.bound,
    arithmetic_intensity: row.arithmetic_intensity,
    arithmetic_intensity_formula: row.arithmetic_intensity_formula,
    t_measured_s: row.t_measured_s,
    measured_mfu: row.measured_mfu,
    measured_mbu: row.measured_mbu,
    source: row.source,
    source_badge: row.source_badge,
    flops_formula: row.flops_formula,
    t_op_formula: row.t_op_formula,
    t_compute_formula: row.t_compute_formula,
    t_memory_formula: row.t_memory_formula,
    access_bytes_formula: row.access_bytes_formula,
    is_comm: row.is_comm,
    comm_bytes_per_rank: row.comm_bytes_agg,
    comm_bytes_formula: row.comm_bytes_formula,
    t_comm_formula: row.t_comm_formula,
    comm_world: row.comm_world,
    comm_bw_label: row.comm_bw_label,
    comm_bw_bytes_per_sec: row.comm_bw_bytes_per_sec,
    comm_phases: row.comm_phases,
    shape_outs: row.shape_outs,
    shape_in_full: row.shape_in_full,
    shape_out_full: row.shape_out_full,
    cut: row.cut,
    source_occurrence: row.source_occurrence,
    sink_occurrences: row.sink_occurrences,
  };
}

/** The vanilla "(x.xx%)" subscript under a value — share of the phase total.
 *  Sub-0.01% shares stay hidden: "(0.00%)" is noise. */
function Share({ part, total, bar }: { part: number | null | undefined; total: number; bar?: boolean }) {
  if (part == null || !Number.isFinite(part) || !(total > 0)) return null;
  const pct = (part / total) * 100;
  if (pct < 0.01) return null;
  if (bar) {
    // The step's time column: the share is also drawn, so where the time
    // goes reads down the column without reading the numbers.
    return (
      <div className="mt-0.5 flex items-center justify-end gap-1.5 text-10 text-gray-400 font-normal font-sans">
        <span className="relative inline-block h-1 w-12 overflow-hidden rounded-full bg-blue-100">
          <span className="absolute inset-y-0 left-0 rounded-full bg-blue-500" style={{ width: `${Math.min(100, pct)}%` }} />
        </span>
        {pct.toFixed(2)}%
      </div>
    );
  }
  return (
    <div className="text-10 text-gray-400 font-normal font-sans">
      ({pct.toFixed(2)}%)
    </div>
  );
}

/** The site's cell (`tbl.cell`: one line, top-aligned, so a value with a
 *  share line under it keeps the row's values on one baseline) in 11px
 *  tabular numbers; an unexercised term collapses to a gray dash instead of
 *  shouting zeros. The op column is the one that wraps (`tbl.wrapCell`). */
const NUM_CELL = `${tbl.cell} text-right text-11 font-mono tabular-nums`;
const NUM = `${NUM_CELL} text-gray-700`;
const DIM_NUM = `${NUM_CELL} text-gray-300`;
const SHAPE = `${tbl.flowCell} text-11 font-mono`;

const BOUNDS: RowBound[] = ["compute", "memory", "comm", "launch"];

/** A per-op table's work and theory column totals. The measured total is not
 *  among them: it is the backend's one measured formula (`measured`), never a
 *  sum of the rows. */
type Totals = { f: number; m: number; c: number; t: number };

/** The term that set the row's t_op, as the backend labelled it; a row with
 *  no winning term (a parent, a sum, an op that takes no time) is a dash. */
function BoundCell({ bound }: { bound?: RowBound | null }) {
  if (!bound) return <td className={DIM_NUM} data-bound="">—</td>;
  return (
    <td className={NUM_CELL} data-bound={bound}>
      <span className={`inline-block rounded-full px-1.5 text-10 leading-4 font-sans font-medium ${BOUND_PILL[bound]}`}>{bound}</span>
    </td>
  );
}

/** FLOPs per access byte, as the backend divided them. */
function AICell({ ai, formula }: { ai?: number | null; formula?: FormulaFields | null }) {
  if (ai == null || !Number.isFinite(ai)) return <td className={DIM_NUM} data-ai="">—</td>;
  const text = ai >= 100 ? ai.toFixed(0) : ai >= 10 ? ai.toFixed(1) : ai.toFixed(2);
  return <td className={NUM} data-ai={String(ai)}><FormulaHover value={text} formula={formula || undefined} /></td>;
}

/** `compute 1.2 ms (8%) · memory …`: the backend's t_SOL split by bound and
 *  each part's share of t_SOL. */
function boundSplit(active: PhaseMetrics): string {
  const by = active.t_sol_by_bound;
  const share = active.t_sol_by_bound_share;
  if (!by || !share) return "—";
  const pct = (f: number) => (f > 0 && f < 0.01 ? "<1%" : `${(f * 100).toFixed(0)}%`);
  return BOUNDS.map((b) => `${b} ${fmtTime(by[b])} (${pct(share[b])})`).join(" · ");
}

/** Rewrite a formula for the per-layer ladder (vanilla `withQtyPerLayer` /
 *  `withPerLayer`): the cell shows ONE invocation, the popover reads
 *  `per layer × N layers = row total`. Identity when the row is unstacked. */
function withPerLayer(
  f: FormulaFields | null | undefined,
  agg: number | null | undefined,
  div: number,
  fmtFn: (n: number) => string,
): FormulaFields | undefined {
  if (!f || div <= 1 || agg == null) return f || undefined;
  return { ...f, per_layer_result: fmtFn(agg / div), cnt: div, cnt_unit: "layers", result: fmtFn(agg) };
}

/** The AI popover under the per-layer ladder: the row's FLOPs and access
 *  cells show one invocation, so the division reads those same per-layer
 *  operands; the × N layers cancels and the ratio is the row's. */
function withPerLayerRatio(
  f: FormulaFields | null | undefined,
  flops: number,
  bytes: number,
  div: number,
): FormulaFields | undefined {
  if (!f || div <= 1) return f || undefined;
  const n = (v: number) => Math.round(v / div).toLocaleString("en-US");
  return { ...f, substituted: `${n(flops)} / ${n(bytes)}`, note: `per layer — × ${div} layers cancels in the ratio` };
}

/** Two-line quantity cell (vanilla `qtyPctCell`): value on top with the
 *  formula popover, `(xx%)` of the phase total underneath; qty <= 0 → dash. */
function QtyCell({ qty, display, formula, total }: {
  qty: number | null | undefined;
  display: string;
  formula?: FormulaFields;
  total: number;
}) {
  if (qty == null || !(qty > 0)) return <td className={DIM_NUM}>—</td>;
  return (
    <td className={NUM}>
      <div className="leading-tight"><FormulaHover value={display} formula={formula} /></div>
      <Share part={qty} total={total} />
    </td>
  );
}

/** Single-line time cell (vanilla `timeCell`): <= 0 collapses to a dash. */
function TimeCell({ t, formula, cls }: {
  t: number | null | undefined;
  formula?: FormulaFields;
  cls?: string;
}) {
  if (t == null || !(t > 0)) return <td className={DIM_NUM}>—</td>;
  return (
    <td className={`${NUM_CELL} ${cls || "text-gray-700"}`}>
      <FormulaHover value={fmtTime(t)} formula={formula} cls={cls} />
    </td>
  );
}

function LeafOpRow({ row, target, phase, depth = 0, perLayer = 1, totals, measured }: {
  row: PerOpRow;
  target?: GraphTarget;
  phase: PhaseKey;
  depth?: number;
  /** Enclosing template's instance count: the row's values are that many
   *  invocations rolled up; show ONE invocation's cost (the vanilla
   *  per-layer ladder), the template header keeps the rollup. */
  perLayer?: number;
  totals?: Totals;
  /** The table's measured total (`PerOpTable.measured`), each share's base. */
  measured?: number | null;
}) {
  const div = perLayer > 1 ? perLayer : 1;
  const per = (v: number | null | undefined) => (v == null ? v : v / div);
  const shapeText = row.shape_in != null && row.shape_out != null && row.shape_in !== row.shape_out
    ? <ShapeFlow from={row.shape_in} to={row.shape_out} />
    : <ShapeFlow from={row.shape_in ?? row.shape_out} />;
  // Shape hover: the unsharded global form when TP/EP shards the per-rank
  // shape the cell shows, plus the outputs of a multi-output leaf. The
  // backend ships shape_outs in two forms: a list of shape lists (per-op
  // timing) or a port-name → shape map (the shape layer on fused leaves).
  const inFull = row.shape_in_full != null ? fmtShape(row.shape_in_full) : "";
  const outFull = row.shape_out_full != null ? fmtShape(row.shape_out_full) : "";
  const outs = row.shape_outs;
  const fmtOut = (sh: unknown) => (Array.isArray(sh) ? `[${sh.join(",")}]` : String(sh));
  const outsNote = Array.isArray(outs)
    ? outs.length ? `outputs: ${outs.map(fmtOut).join(" + ")}` : ""
    : outs && typeof outs === "object" && Object.keys(outs).length
      ? `outputs: ${Object.entries(outs).map(([p, sh]) => `${p}=${fmtOut(sh)}`).join(", ")}`
      : "";
  const shapeNote = [
    inFull || outFull ? `unsharded: ${inFull || "—"} → ${outFull || "—"}` : "",
    outsNote,
  ].filter(Boolean).join("\n");
  const commF = row.is_comm ? row.t_comm_formula : null;
  const flops = row.flops_per_rank || 0;
  const mem = row.access_bytes_per_rank || 0;
  // A row that does not PUBLISH its collective bytes is not a row that
  // moves zero of them: the backend leaves the figure null where it has no
  // law to state it at this batch (`batch_sweep`), and the cell says so.
  const comm = row.comm_bytes_per_rank ?? null;
  // Every priced row carries its own roofline terms and t_sol (collectives
  // included: `phases` prices them with the same roofline as any leaf).
  const tOp = row.t_sol_s;
  const tComm = row.t_comm_s;
  return (
    <tr
      className={tbl.row}
      data-leaf-row=""
      data-aggregation-id={row.aggregation_id || undefined}
      data-per-layer={div > 1 ? String(div) : undefined}
    >
      <td className={tbl.wrapCell}>
        <div className="flex items-start" style={{ paddingLeft: depth * 16 }}>
          {/* Same-width placeholder as the parents' chevron, so every row's
           *  name shares one left edge at its depth (vanilla chevron rule). */}
          <span className="inline-block h-5 w-5 shrink-0" />
          <span className={`ml-1 min-w-0 font-mono ${row.is_comm ? "text-sky-800" : "text-gray-700"}`}>
            {/* OpLabel stays for comm rows too: it carries the fused-op
             *  popover and the pipeline-hop (cut) label. */}
            <OpLabel row={row} target={target} phase={phase} />
            {row.is_comm ? (
              <>
                <span className={`ml-2 inline-block rounded-full px-1.5 font-sans text-10 font-medium leading-4 ${BOUND_PILL.comm}`}>
                  comm{row.comm_bw_label ? ` · ${row.comm_bw_label}` : ""}
                </span>
              </>
            ) : null}
          </span>
        </div>
      </td>
      <td
        className={`${SHAPE} text-gray-500`}
        {...(shapeNote ? textTip(shapeNote) : {})}
      >
        {shapeNote ? <span className={HOVER_HINT}>{shapeText}</span> : shapeText}
      </td>
      <QtyCell qty={flops} display={formatFlops(per(flops) || 0)} formula={withPerLayer(row.flops_formula, flops, div, formatFlops)} total={totals?.f || 0} />
      <QtyCell qty={mem} display={formatBytes(per(mem) || 0)} formula={withPerLayer(row.access_bytes_formula, mem, div, formatBytes)} total={totals?.m || 0} />
      {comm === null ? (
        <td className="px-2 py-0.5 text-right font-mono text-gray-400"
            title="not published at this batch — the collective's byte figure has no law to be re-read from, so the row carries none rather than the walked batch's value">
          —
        </td>
      ) : (
        <QtyCell qty={comm} display={formatBytes(per(comm) || 0)} formula={withPerLayer(row.comm_bytes_formula, comm, div, formatBytes)} total={totals?.c || 0} />
      )}
      <TimeCell t={per(row.t_compute_s)} formula={withPerLayer(row.t_compute_formula, row.t_compute_s, div, fmtTime)} />
      <TimeCell t={per(row.t_memory_s)} formula={withPerLayer(row.t_memory_formula, row.t_memory_s, div, fmtTime)} />
      <TimeCell t={per(tComm)} formula={withPerLayer(commF, tComm, div, fmtTime)} />
      <BoundCell bound={row.bound} />
      <AICell ai={row.arithmetic_intensity} formula={withPerLayerRatio(row.arithmetic_intensity_formula, flops, mem, div)} />
      {tOp != null && tOp > 0 ? (
        <td className={NUM}>
          <div className="leading-tight">
            <FormulaHover value={fmtTime(per(tOp))} formula={withPerLayer(commF || row.t_op_formula, tOp, div, fmtTime)} cls="text-blue-700" />
          </div>
          <Share part={tOp} total={totals?.t || 0} bar />
        </td>
      ) : (
        <td className={DIM_NUM}>—</td>
      )}
      <MeasuredCell row={div > 1 ? { ...row, t_measured_s: per(row.t_measured_s) ?? null } : row} total={measured ?? undefined} part={row.t_measured_s} />
    </tr>
  );
}

export function PhaseTable({ result, target }: { result: Estimate; target: GraphTarget }) {
  const [picked, setTab] = useState<"prefill" | "decode">("decode");
  const [partial, setPartial] = useState(false);
  // No output token, no decode step (the TPOT card says the same): only the
  // prefill phase has a table.
  const noDecode = result.avg_output_tokens === 0;
  const tab = noDecode ? "prefill" : picked;
  const pre = result.phases?.prefill || {};
  const dec = result.phases?.decode || {};
  if ((pre.batch || 0) === 0 && (dec.batch || 0) === 0) {
    return (
      <section className="card p-5 mb-6">
        <h3 className="font-semibold text-gray-900 mb-2">
          Prefill vs Decode <span className="text-xs text-gray-400 font-normal">per-step, per-rank</span>
        </h3>
        <div className="text-sm text-gray-500" id="infer-phase-no-step">
          No step runs here, so there is nothing per step to show
          {pre.time_unpublished_reason || dec.time_unpublished_reason
            ? `: ${pre.time_unpublished_reason || dec.time_unpublished_reason}`
            : " — adjust parallelism or fraction."}
        </div>
      </section>
    );
  }
  const active = tab === "prefill" ? pre : dec;
  // A deployment that admits no request publishes no step (a disaggregated
  // pair can have one side without): its tab says why, not a row of zeros.
  const noStep = active.batch == null;
  const rows = tab === "prefill" ? (result.per_op_timing || []) : (result.per_op_timing_decode || []);
  const byDeclaration = (tab === "prefill"
    ? result.per_op_timing_by_declaration
    : result.per_op_timing_decode_by_declaration) || [];
  const tabCls = (k: "prefill" | "decode") =>
    "px-3 py-1.5 text-xs font-medium " + (tab === k ? "bg-amber-500 text-white" : "bg-white text-gray-600 hover:bg-gray-50");

  const metrics: PhaseMetric[] = [
    { label: "Workload shape", value: shapeStr(active), cls: "text-gray-600" },
    { label: "FLOPs / step", value: formatFlops(active.flops_per_step || 0), formula: active.formulas?.flops },
    { label: "Memory read / step", value: formatBytes(active.mem_read_bytes || 0), formula: active.formulas?.mem_read },
    { label: "Memory write / step", value: formatBytes(active.mem_write_bytes || 0), formula: active.formulas?.mem_write },
    { label: "Arithmetic intensity", value: active.arithmetic_intensity == null ? "—" : `${active.arithmetic_intensity.toFixed(1)} FLOPs/B`, cls: "text-amber-700", formula: active.formulas?.arithmetic_intensity },
    {
      label: "t_SOL by bound (each op's t_op under the term that set it)",
      value: boundSplit(active),
      cls: "text-blue-700",
    },
    {
      label: "t_SOL (Speed-of-Light) = Σ leaf max(t_compute, t_memory, t_comm, t_launch)",
      value: active.t_sol_s == null ? "—" : fmtTime(active.t_sol_s),
      cls: "text-blue-700 font-semibold",
      formula: active.formulas?.t_sol,
    },
    measuredMetric(active),
    ...(partial ? [partialMetric(active)] : []),
    {
      label: "Throughput (per-rank equivalent = cluster_total / n_gpus)",
      value: tokPerSec(active),
      cls: "text-gray-600",
      formula: active.throughput?.sol?.formula,
    },
    ...(partial ? [{
      label: "Throughput, partial measured",
      value: tokPerSec(active, "partial"),
      cls: "text-teal-700",
      formula: active.throughput?.partial?.formula,
    }] : []),
  ];
  const metricCols: ColumnDef<PhaseMetric>[] = [
    { key: "label", wrap: true, header: "", className: "text-gray-600", sortValue: null },
    {
      key: "value",
      header: tab === "prefill" ? "Prefill" : "Decode",
      align: "right",
      sortValue: null,
      className: (r) => `font-mono ${r.cls || "text-gray-900"}`,
      cellProps: (r) => ({
        ...(r.formula ? tipHandlers(formulaHTML(r.formula)) : r.tip ? textTip(r.tip) : {}),
        style: r.formula || r.tip ? { textDecoration: "underline dotted" } : undefined,
        ...(r.label.startsWith("t_measured") ? { "data-measured-total": "" } : {}),
        ...(r.label.startsWith("t_partial") ? { "data-partial-total": "" } : {}),
      }),
    },
  ];

  return (
    <section className="card p-5 mb-6">
      <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
        <h3 className="font-semibold text-gray-900">
          Prefill vs Decode <span className="text-xs text-gray-400 font-normal">per-step, per-rank</span>
        </h3>
        <div id="infer-phase-source" className="text-xs text-gray-500">FLOPs from spec-engine primitive sum</div>
      </div>
      <div className="flex items-center gap-3 flex-wrap mb-3">
        <div className="inline-flex rounded-lg border border-gray-200 overflow-hidden" role="tablist" aria-label="Phase">
          <button id="infer-phase-tab-prefill" type="button" className={tabCls("prefill")} role="tab" aria-selected={tab === "prefill"} onClick={() => setTab("prefill")}>
            Prefill
          </button>
          <button id="infer-phase-tab-decode" type="button" className={`${tabCls("decode")}${noDecode ? " opacity-50 cursor-not-allowed" : ""}`} role="tab" aria-selected={tab === "decode"}
            disabled={noDecode} title={noDecode ? "No output tokens: there is no decode step" : undefined} onClick={() => setTab("decode")}>
            Decode
          </button>
        </div>
        <span className="text-xs text-gray-400">
          Prefill B=1 (activation). Decode B=max (KV).
        </span>
        <label id="infer-partial-toggle" className="flex items-center gap-1.5 text-xs text-gray-600"
          title="Partial measured: each op Kernel Bench has measured at its measured time, every other op at its SOL time, added up as the step adds them. Shown with how much of the step is real measurement.">
          <input type="checkbox" checked={partial} onChange={(e) => setPartial(e.target.checked)} />
          Partial measured
        </label>
        <div className="flex-1" />
        {/* This step as measurable jobs. One per STEP, not per table: the
         *  draft breakdown below is part of the decode step, and the
         *  decode export already carries its ops. */}
        {noStep ? null : <BenchJobsExport target={target} phase={tab} />}
      </div>
      {noStep ? (
        <div className="text-sm text-gray-500" id="infer-phase-no-step">
          No {tab} step runs here, so there is nothing per step to show
          {active.time_unpublished_reason ? `: ${active.time_unpublished_reason}` : "."}
        </div>
      ) : (<>
      <div className="text-11 text-amber-700 mb-2">
        ⚠ <span className="font-medium">No-overlap roofline.</span> t_SOL sums per-leaf t_op = max(t_compute, t_memory, t_comm, t_launch) — kernels assumed serial, no compute/comm overlap; one link latency on every fabric.
      </div>
      <div id="infer-phase-table" className="overflow-x-auto">
        <DataTable
          columns={metricCols}
          rows={metrics}
          rowKey={(r) => r.label}
          framed={false}
          tableClassName="text-sm"
          rowProps={() => ({ "data-metric": "" })}
          footer={tab === "decode" && active.speculative_draft ? <DraftRow draft={active.speculative_draft} /> : null}
        />
      </div>
      <PerOpTable
        key={tab}
        rows={rows}
        moduleRows={active.flops_by_module}
        total={active.op_table_total}
        byDeclaration={byDeclaration}
        target={target}
        phase={tab}
        measured={completeTotal(active.t_measured_s, phaseCoverage(active))}
      />
      {tab === "decode" && active.speculative_draft && (result.per_op_timing_draft || []).length > 0 && (
        <PerOpTable
          key="draft"
          id="infer-phase-draft-breakdown"
          caption={`Speculative draft — one forward of ${active.speculative_draft.forwards} per step (${active.speculative_draft.stacks.join(", ")} stack, ${active.speculative_draft.tokens_per_forward} tok/seq)`}
          rows={result.per_op_timing_draft || []}
          total={active.speculative_draft.op_table_total}
          byDeclaration={[]}
          target={target}
          phase={tab}
          measured={active.speculative_draft.per_forward?.t_measured_s ?? null}
        />
      )}
      <FusionSavings delta={result.vs_naive_delta?.[tab]} />
      </>)}
    </section>
  );
}

/** What fusion changes in this phase's theoretical step time, as the backend
 *  priced both trees: saves when positive, costs when negative. */
function FusionSavings({ delta }: { delta?: FusionDelta }) {
  if (!delta || delta.delta_s === 0) return null;
  const saves = delta.delta_s > 0;
  return (
    <div id="infer-fusion-delta" className={`mt-2 text-11 ${saves ? "text-green-700" : "text-red-700"}`}>
      fusion {saves ? "saves" : "costs"} {fmtTime(Math.abs(delta.delta_s))} vs pre-fusion roofline
    </div>
  );
}

/** The draft's own term of the decode step, as the backend priced it: its
 *  time, which every total above already includes, and its share of the step. */
function DraftRow({ draft }: { draft: SpeculativeDraftTerm }) {
  const style = draft.style === "recursive" ? "recursive draft" : "parallel draft";
  const share = draft.t_sol_share == null ? "" : ` · ${(draft.t_sol_share * 100).toFixed(1)}% of step`;
  return (
    <tr className={tbl.row} data-metric="" id="infer-phase-draft">
      <td className={`${tbl.wrapCell} text-gray-600`}>
        Speculative draft ({style}: {draft.forwards} × {draft.tokens_per_forward} tok/seq) — included above
      </td>
      <td
        className={`${tbl.cell} text-right font-mono tabular-nums text-purple-700`}
        {...tipHandlers(draft.formula ? formulaHTML(draft.formula) : undefined)}
        style={draft.formula ? { textDecoration: "underline dotted" } : undefined}
      >
        {draft.t_sol_s == null ? "—" : fmtTime(draft.t_sol_s)}{share}
      </td>
    </tr>
  );
}

type PhaseMetric = { label: string; value: string; cls?: string; formula?: FormulaFields; tip?: string };

/** The step's measured time, all or nothing: the total only when every op is
 *  measured, otherwise a dash, the coverage and the unmeasured keys by state
 *  (in the tip) — never a partial sum. */
function measuredMetric(active: PhaseMetrics): PhaseMetric {
  const cov = phaseCoverage(active);
  const t = completeTotal(active.t_measured_s, cov);
  const counts = stateCounts(cov);
  return {
    label: "t_measured (shown only when every op is measured)",
    value: t != null ? fmtMeasured(t) : `— (${coverageText(cov)}${counts ? `; ${counts}` : ""})`,
    cls: t != null ? "text-green-700 font-semibold" : "text-gray-500",
    tip: t != null ? undefined : stateKeysText(cov) || undefined,
  };
}

/** The partial track (opt-in): measured ops measured, the rest at SOL — a
 *  blend the backend adds up (`t_partial_s`), named as one and shown with
 *  its real-value coverage, never as `t_measured`. */
function partialMetric(active: PhaseMetrics): PhaseMetric {
  const cov = phaseCoverage(active);
  const t = active.t_partial_s;
  const coverage = partialCoverageText(active.partial_measured_share, cov);
  return {
    label: "t_partial (measured ops measured, the rest at SOL)",
    value: t == null ? "— (no measured fold)" : `${fmtTime(t)}${coverage ? ` (${coverage})` : ""}`,
    cls: t == null ? "text-gray-500" : "text-teal-700 font-semibold",
    tip: t == null ? "No partial total: the estimate did not read Kernel Bench, or the step has no SOL time." : stateKeysText(cov) || undefined,
  };
}

const perOpCols: ColumnDef<unknown>[] = [
  { key: "op", header: "Module / op", wrap: true, headerClassName: "min-w-[13rem]", sortValue: null },
  { key: "shape", header: "Shape", wrap: true, headerClassName: "min-w-[9rem]", sortValue: null },
  // A quantity and its roofline term name themselves in their bound's colour
  // (the bound pills' palette): FLOPs and t_compute are compute's.
  { key: "flops", header: "FLOPs", align: "right", headerClassName: BOUND_TEXT.compute, sortValue: null },
  { key: "memory", header: "Memory", align: "right", headerClassName: BOUND_TEXT.memory, sortValue: null },
  { key: "comm", header: "Comm", align: "right", headerClassName: BOUND_TEXT.comm, sortValue: null },
  { key: "t_compute", header: "t_compute", align: "right", headerClassName: BOUND_TEXT.compute, sortValue: null },
  { key: "t_memory", header: "t_memory", align: "right", headerClassName: BOUND_TEXT.memory, sortValue: null },
  { key: "t_comm", header: "t_comm", align: "right", headerClassName: BOUND_TEXT.comm, sortValue: null },
  { key: "bound", header: "bound", align: "right", sortValue: null },
  { key: "ai", header: "AI", align: "right", sortValue: null },
  { key: "t_sol", header: "t_sol", align: "right", headerClassName: "text-blue-700", sortValue: null },
  { key: "t_measured", header: "t_measured", align: "right", sortValue: null },
];

export function PerOpTable({
  rows, moduleRows, total, byDeclaration, target, phase, id = "infer-phase-breakdown", caption, measured,
}: {
  rows: PerOpRow[];
  moduleRows?: ModuleFoldRow[];
  /** The table's Total row as the backend summed it over these rows; no
   *  Total row (and no shares) without it. */
  total?: OpTableTotal;
  byDeclaration: DeclarationRow[];
  /** The deployment the rows were priced for, which a fused row's absorbed
   *  subgraph is fetched from; omitted for rows of no model deployment. */
  target?: GraphTarget;
  phase: PhaseKey;
  id?: string;
  caption?: string;
  /** The table's measured total as the backend stated it (the one measured
   *  formula, null unless every term is measured) — read, never re-added. */
  measured: number | null;
}) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const foldRows = moduleRows || [];
  // Every row opens unfolded; the reader folds what they do not need.
  const [folded, setFolded] = useState<Set<string>>(new Set());
  // Fold state survives batch-size re-renders; it resets when the model, the
  // phase, or the parallel layout changes (vanilla keyed on tp/ep/pp), not
  // when the row count shifts under a new batch. Adjusted during render (not
  // in an effect) so the new dataset never paints one frame under the old
  // dataset's fold keys.
  const dataset = `${target?.modelId ?? id}|${phase}|${JSON.stringify(target?.par)}`;
  const [prevDataset, setPrevDataset] = useState(dataset);
  if (prevDataset !== dataset) {
    setPrevDataset(dataset);
    setFolded(new Set());
  }
  const foldedKeys = folded;
  if (!rows.length && !foldRows.length) return null;
  const parentAt = new Map<number, number>();
  const declHidden = new Set<number>();
  byDeclaration.forEach((p, k) => {
    parentAt.set(p.children[0], k);
    p.children.forEach((i) => declHidden.add(i));
  });
  const toggle = (k: number) => setOpen((prev) => {
    const next = new Set(prev);
    if (next.has(k)) next.delete(k);
    else next.add(k);
    return next;
  });
  const shownDecl = rows.length - declHidden.size + byDeclaration.length;
  const body: ReactNode[] = [];
  const tree = visibleFold(foldRows, foldedKeys);
  // The table's work and theory totals are the backend's (`op_table_total`),
  // one sum for the fold and the flat view alike. Shares read the RAW rollup
  // against them, even where the cell shows one invocation.
  const totals = shareBase(total);
  const setAllOpen = (value: boolean) => {
    if (value) setFolded(new Set());
    else {
      const next = new Set<string>();
      foldRows.forEach((row, i) => {
        if (hasFoldChildren(foldRows, i)) next.add(foldKey(row));
      });
      setFolded(next);
    }
  };
  const toggleFold = (key: string) => {
    const next = new Set(foldedKeys);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setFolded(next);
  };

  tree.forEach(({ row, kids }) => {
    const key = foldKey(row);
    const d = foldDepth(row);
    // Per-layer divisor: the backend's `layer_count` on every row — a row
    // inside a template carries the template's count (one cost per call, so
    // total / count is one layer's), a collective its own count_per_step, a
    // depth-0 op its multiplicity.
    const perLayer = row.layer_count || 1;
    const per = foldAsPerOp(row);
    if (kids) {
      const isHeader = row.op === "<template>";
      // Three tiers (vanilla): template headers read as section titles,
      // mid-tree aggregators go dim, rows with their own timing stay plain.
      const weight = isHeader
        ? "font-sans text-xs font-semibold text-gray-900"
        : (row.t_op_s || 0) > 0
          ? "font-normal text-gray-700"
          : "font-medium text-gray-500";
      // Vanilla's per-layer ladder: only the template header keeps the ×N
      // rollup; every other row inside the template shows ONE layer's cost.
      const div = !isHeader && perLayer > 1 ? perLayer : 1;
      const perN = (v: number | null | undefined) => (v == null ? v : v / div);
      const flops = row.flops_per_rank || 0;
      // A parent shows its subtree: the backend's rollups
      // (`enrich_module_breakdown`), on every fold it ships.
      const mem = row.access_bytes_agg || 0;
      // Null when a row under it does not publish its collective bytes:
      // the backend rollup goes unpublished with it (`_rollup_column`).
      const comm = row.comm_bytes_agg ?? null;
      const tCompute = row.t_compute_agg;
      const tMemory = row.t_memory_agg;
      const tComm = row.t_comm_agg;
      const tOp = row.t_op_agg;
      // The backend's share of the measured formula under the group: blank
      // unless every term under it is measured.
      const tMeas = groupMeasured(row);
      body.push(
        <tr
          key={key}
          className={`${tbl.row} ${isHeader ? "bg-amber-50/70 shadow-[inset_3px_0_0_var(--color-amber-400)]" : "bg-gray-50/70"}`}
          data-fold-row=""
          data-depth={String(d)}
          data-name={row.name || ""}
          data-children={kids ? "1" : "0"}
          data-flops={String(row.flops_per_rank || 0)}
        >
          <td className={tbl.wrapCell}>
            <div className="flex items-start" style={{ paddingLeft: d * 16 }}>
              <button
                type="button"
                className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md text-gray-400 hover:bg-white hover:text-gray-700"
                aria-expanded={!foldedKeys.has(key)}
                data-module-toggle=""
                onClick={() => toggleFold(key)}
              >
                <svg className={`h-3 w-3 transition-transform ${foldedKeys.has(key) ? "" : "rotate-90"}`} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M7 5l5 5-5 5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              <span className={`ml-1 ${isHeader ? "" : "font-mono"} ${weight}`} {...(isHeader ? textTip(row.name) : {})}>
                {isHeader ? row.template_label || row.name : row.name}
                {isHeader && row.layer_count && row.layer_count > 1 ? (
                  <span className="ml-2 inline-block whitespace-nowrap rounded-full bg-amber-100 px-2 font-sans text-10 font-medium leading-4 text-amber-800">{`× ${row.layer_count} layers`}</span>
                ) : null}
                {row.op && row.op !== row.name && row.op !== "<template>" ? (
                  <span className="ml-1.5 text-10 font-normal text-gray-400">{row.op}</span>
                ) : null}
              </span>
            </div>
          </td>
          <td className={`${SHAPE} text-gray-300`}>—</td>
          <QtyCell qty={flops} display={formatFlops(perN(flops) || 0)} formula={withPerLayer(row.flops_formula, flops, div, formatFlops)} total={totals.f} />
          <QtyCell qty={mem} display={formatBytes(perN(mem) || 0)} formula={withPerLayer(row.access_bytes_formula, mem, div, formatBytes)} total={totals.m} />
          {comm === null
            ? <td className="px-2 py-0.5 text-right font-mono text-gray-400"
                  title="not published at this batch — this collective's payload is not its typed flow, so there is no law to read its per-rank volume from">—</td>
            : <QtyCell qty={comm} display={formatBytes(perN(comm) || 0)} formula={withPerLayer(row.comm_bytes_formula, comm, div, formatBytes)} total={totals.c} />}
          <TimeCell t={perN(tCompute)} formula={withPerLayer(row.t_compute_formula, tCompute, div, fmtTime)} />
          <TimeCell t={perN(tMemory)} formula={withPerLayer(row.t_memory_formula, tMemory, div, fmtTime)} />
          <TimeCell t={perN(tComm)} />
          <BoundCell bound={row.bound} />
          <AICell ai={row.arithmetic_intensity} formula={withPerLayerRatio(row.arithmetic_intensity_formula, flops, mem, div)} />
          {tOp != null && tOp > 0 ? (
            <td className={`${NUM} font-semibold`}>
              <div className="leading-tight"><FormulaHover value={fmtTime(perN(tOp))} formula={withPerLayer(row.t_op_formula, tOp, div, fmtTime)} cls="text-blue-700" /></div>
              <Share part={tOp} total={totals.t} bar />
            </td>
          ) : (
            <td className={DIM_NUM}>—</td>
          )}
          <td className={`${NUM_CELL} ${tMeas != null ? "text-green-700" : "text-gray-300"}`} data-measured-cell="">
            <div className="leading-tight">{fmtMeasured(perN(tMeas))}</div>
            {tMeas != null && measured ? <Share part={tMeas} total={measured} /> : null}
          </td>
        </tr>,
      );
    } else {
      body.push(
        <LeafOpRow
          key={key}
          row={per}
          target={target}
          phase={phase}
          depth={d}
          perLayer={perLayer}
          totals={totals}
          measured={measured}
        />,
      );
    }
  });
  if (!tree.length) {
    rows.forEach((r, i) => {
      const k = parentAt.get(i);
      if (k !== undefined) {
        const p = byDeclaration[k];
        body.push(
          <DeclarationOpRow key={`decl-${k}`} parent={p} first={r} open={open.has(k)} onToggle={() => toggle(k)} target={target} phase={phase} />,
        );
        if (open.has(k)) {
          p.children.forEach((c) => body.push(<GroupOpRow key={`decl-${k}-${c}`} row={rows[c]} />));
        }
        return;
      }
      if (declHidden.has(i)) return;
      body.push(
        <LeafOpRow
          key={`leaf-${r.spec_id || r.aggregation_id || r.op || i}`}
          row={r}
          target={target}
          phase={phase}
          measured={measured}
        />,
      );
    });
  }
  const leafCount = tree.filter((t) => !t.kids).length || rows.length;
  return (
    <div id={id} className="mt-4">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
        <div className="text-xs font-semibold text-gray-700">
          {caption ? `${caption} · ${leafCount} leaves` : tree.length
            ? `Per-op — ${tree.length} rows · ${leafCount} leaves`
            : byDeclaration.length
            ? `Per-op — ${shownDecl} rows per declaration · ${rows.length} leaves`
            : `Per-op — ${rows.length} leaves`}
        </div>
        {/* Gate on ALL fold rows, not the visible ones: after Collapse all
         *  only the depth-0 parents remain visible, and on a single-template
         *  model that is one row — the controls must survive their own use. */}
        {foldRows.some((_, i) => hasFoldChildren(foldRows, i)) ? (
          <div className="text-xs text-gray-400 select-none">
            <button
              type="button"
              className="text-gray-500 hover:text-amber-700"
              data-module-toggle-all=""
              onClick={() => setAllOpen(false)}
            >
              Collapse all
            </button>
            <span className="mx-1">·</span>
            <button
              type="button"
              className="text-gray-500 hover:text-amber-700"
              data-module-expand-all=""
              onClick={() => setAllOpen(true)}
            >
              Expand all
            </button>
          </div>
        ) : null}
      </div>
      {tree.some((t) => t.row.op === "<template>" && (t.row.layer_count || 1) > 1) ? (
        <div className="text-11 text-gray-500 mb-2" data-per-layer-note="">
          A block row (amber, <span className="font-medium">× N layers</span>) totals all its layers. Rows inside it
          show one layer; their % is that op&apos;s share of the whole step, all layers included.
        </div>
      ) : null}
      <DataTable
        columns={perOpCols}
        rows={[]}
        rowKey={() => ""}
        headerGroups={[
          { label: "Op", span: 2 },
          { label: "Work", span: 3 },
          { label: "Roofline terms", span: 3 },
          { label: "Bound", span: 2 },
          { label: "Result", span: 2 },
        ]}
        renderBody={() => body}
        footer={total && (tree.length || rows.length) ? <TotalRow total={total} measured={measured} /> : null}
      />
    </div>
  );
}

/** The backend's table total as the denominators a row's share reads. */
function shareBase(total?: OpTableTotal): Totals {
  return {
    f: total?.flops_per_rank ?? 0,
    m: total?.access_bytes_per_rank ?? 0,
    c: total?.comm_bytes_per_rank ?? 0,
    t: total?.t_sol_s ?? 0,
  };
}

/** Bottom Total row (vanilla): the backend's `op_table_total` for the
 *  quantity and t_sol columns, and the table's measured total as the backend
 *  stated it (`measured`, the one measured formula). The roofline-term columns (compute, memory, comm) deliberately
 *  stay blank — leaf t_op = max(terms), so their column sums would misread as
 *  wall time; the cells carry a tip explaining why. */
function TotalRow({ total, measured }: { total: OpTableTotal; measured: number | null }) {
  const totals = shareBase(total);
  const totalQty = (qty: number, display: string, label = "(100%)") =>
    qty > 0 ? (
      <td className={NUM_CELL}>
        <div className="leading-tight">{display}</div>
        <div className="text-10 text-gray-500 leading-tight font-normal">{label}</div>
      </td>
    ) : (
      <td className={`${NUM_CELL} text-gray-400`}>—</td>
    );
  const noTermTotal = (
    <td
      className={DIM_NUM}
      {...textTip("No Total: leaf t_op = max(t_compute, t_memory, t_comm, t_launch), so the column sums of the terms are not addends of t_op. The Total t_sol cell already reflects the max-per-leaf rollup.")}
      style={{ textDecoration: "underline dotted" }}
    >
      ·
    </td>
  );
  return (
    <tr className={tbl.footRow} data-total-row="">
      <td className={tbl.cell}>Total</td>
      <td />
      {totalQty(totals.f, formatFlops(totals.f))}
      {totalQty(totals.m, formatBytes(totals.m))}
      {total.comm_bytes_per_rank == null
        ? <td className="px-2 py-0.5 text-right font-mono text-gray-400"
              title="not published at this batch — at least one collective row carries no byte figure, so this column has no total">—</td>
        : totalQty(totals.c, formatBytes(totals.c))}
      {noTermTotal}
      {noTermTotal}
      {noTermTotal}
      <BoundCell />
      <AICell />
      {totalQty(totals.t, fmtTime(totals.t))}
      {/* All or nothing: the measured total only when every row is
       *  measured; a partial sum is not the phase's time. */}
      {measured != null ? (
        <td className={NUM_CELL} data-measured-total="">
          <div className="leading-tight">{fmtMeasured(measured)}</div>
          <div className="text-10 text-gray-500 leading-tight font-normal">(100%)</div>
        </td>
      ) : (
        <td className={`${NUM_CELL} text-gray-400`} data-measured-total="" {...textTip("No measured total: not every op of the step is measured.")}>—</td>
      )}
    </tr>
  );
}

/** One op of a declaration split into derived groups: the backend's sums over
 *  its groups, the `m -> layers` histogram, and a toggle for the groups. */
function DeclarationOpRow({
  parent, first, open, onToggle, target, phase,
}: {
  parent: DeclarationRow;
  first: PerOpRow;
  open: boolean;
  onToggle: () => void;
  target?: GraphTarget;
  phase: PhaseKey;
}) {
  const d = parent.declaration;
  const hist = parent.histogram
    .map((h) => `${d.axis}=${h.visible}: ${h.layers} layer${h.layers === 1 ? "" : "s"} (${compressRanges(h.layer_indices)})`)
    .join("\n");
  return (
    <tr
      className={`${tbl.row} bg-amber-50/40`}
      data-declaration-row=""
      data-children={parent.children.join(",")}
      data-flops={String(parent.flops_per_rank)}
      data-bytes={String(parent.access_bytes_per_rank)}
      data-t-sol={String(parent.t_sol_s)}
      data-layers={String(parent.layers)}
    >
      <td className={`${tbl.wrapCell} text-gray-800 font-mono`}>
        <button
          type="button"
          className="mr-1 text-gray-400 hover:text-gray-700"
          aria-expanded={open}
          aria-label="Toggle derived groups"
          data-declaration-toggle=""
          onClick={onToggle}
        >
          {open ? "▾" : "▸"}
        </button>
        <OpLabel row={first} target={target} phase={phase} />
        <span className="ml-2 text-10 text-gray-400" {...textTip(hist)}>
          {d.base_template} · {parent.layers} layers · {d.axis} {visibleSpan(parent.histogram)}
        </span>
      </td>
      <td className={`${SHAPE} text-gray-500`}>
        {parent.shape_in != null || parent.shape_out != null
          ? <ShapeFlow from={parent.shape_in ?? parent.shape_out} />
          : <span className="text-gray-400" {...textTip(`varies with ${d.axis}`)}>varies with {d.axis}</span>}
      </td>
      <td className={NUM}>{formatFlops(parent.flops_per_rank)}</td>
      <td className={NUM}>{formatBytes(parent.access_bytes_per_rank || 0)}</td>
      <td className={DIM_NUM}>—</td>
      <td className={NUM}>{fmtTime(parent.t_compute_s)}</td>
      <td className={NUM}>{fmtTime(parent.t_memory_s)}</td>
      <td className={NUM}>{fmtTime(parent.t_comm_s)}</td>
      <BoundCell />
      <AICell ai={parent.arithmetic_intensity} formula={parent.arithmetic_intensity_formula} />
      <td className={`${NUM_CELL} text-blue-700`}>{fmtTime(parent.t_sol_s)}</td>
      <td className={`${NUM_CELL} text-green-700`}>{fmtMeasured(parent.t_measured_s)}</td>
    </tr>
  );
}

/** One derived group's row under its declaration: which history length, which
 *  layers, and the group's own numbers. */
function GroupOpRow({ row }: { row: PerOpRow }) {
  const g = row.derived_group;
  return (
    <tr
      className={`${tbl.row} text-gray-600`}
      data-declaration-child=""
      data-visible={g ? String(g.visible) : ""}
      data-flops={String(row.flops_per_rank ?? 0)}
      data-bytes={String(row.access_bytes_per_rank ?? 0)}
      data-t-sol={String(row.t_sol_s ?? 0)}
    >
      <td className={`${tbl.wrapCell} pl-9 font-mono`}>
        {g ? (
          <>
            {g.axis}={g.visible}
            <span className="ml-2 text-10 text-gray-400">
              {g.count} layer{g.count === 1 ? "" : "s"}: {compressRanges(g.layer_indices)}
            </span>
          </>
        ) : (row.op || "—")}
      </td>
      <td className={`${SHAPE} text-gray-500`}><ShapeFlow from={row.shape_in ?? row.shape_out} /></td>
      <td className={NUM}>{formatFlops(row.flops_per_rank)}</td>
      <td className={NUM}>{formatBytes(row.access_bytes_per_rank || 0)}</td>
      <td className={`${NUM_CELL} ${(row.comm_bytes_per_rank || 0) > 0 ? "text-gray-700" : "text-gray-300"}`}>
        {(row.comm_bytes_per_rank || 0) > 0 ? formatBytes(row.comm_bytes_per_rank || 0) : "—"}
      </td>
      <td className={NUM}><FormulaHover value={fmtTime(row.t_compute_s)} formula={row.t_compute_formula} /></td>
      <td className={NUM}><FormulaHover value={fmtTime(row.t_memory_s)} formula={row.t_memory_formula} /></td>
      <td className={NUM}>{fmtTime(row.t_comm_s)}</td>
      <BoundCell bound={row.bound} />
      <AICell ai={row.arithmetic_intensity} formula={row.arithmetic_intensity_formula} />
      <td className={`${NUM_CELL} text-blue-700`}>{fmtTime(row.t_sol_s)}</td>
      <MeasuredCell row={row} />
    </tr>
  );
}

/** The measured time and the release it was measured on. Measurements are
 *  versioned and framework bindings are not: the lookup takes the framework's
 *  kernel at the latest release the corpus measured it on (or, in envelope
 *  mode, the fastest kernel of any library), and the cell names that release
 *  so an older measurement never reads as a newer one. */
function MeasuredCell({ row, total, part }: { row: PerOpRow; total?: number; part?: number | null }) {
  const b = row.source_badge;
  const measured = (row.source || "").startsWith("measured") && b?.framework;
  const made = measured ? `${b.framework} ${b.framework_version || ""}`.trim() : "";
  const title = measured
    ? [made + (b.kernel_impl ? ` → ${b.kernel_impl}` : ""),
       b.framework_versions_measured ? `measured on: ${b.framework_versions_measured}` : "",
       b.selection_mode ? `selection: ${b.selection_mode}` : "",
       // Which driver produced the number, and — when it was not the box's
       // own — which driver it was not. Stated as the observation and no
       // further: the row carries no direction for the substitution and no
       // support verdict for it, so neither is claimed here. The backend
       // makes the comparison; this only shows it.
       b.driver ? `driver: ${b.driver}` : "",
       b.driver_substituted_for
         ? `substituted for this box's own ${b.driver_substituted_for}`
         : ""].filter(Boolean).join("\n")
    : undefined;
  // 0 is a value: a zero-cost row is measured at 0.
  const has = rowMeasured(row.t_measured_s) != null;
  return (
    <td
      className={`${NUM_CELL} ${has ? "text-green-700" : "text-gray-300"}`}
      data-measured-cell=""
      {...textTip(title)}
    >
      <div className="leading-tight">
        {fmtMeasured(row.t_measured_s)}
        {made ? <span className="ml-1 text-10 text-gray-400" data-measured-on="">{made}</span> : null}
      </div>
      {measured ? <Utilization mfu={row.measured_mfu} mbu={row.measured_mbu} /> : null}
      {has && total ? <Share part={part ?? row.t_measured_s} total={total} /> : null}
    </td>
  );
}

/** The measured kernel's MFU and MBU, as the backend stated them. */
function Utilization({ mfu, mbu }: { mfu?: number | null; mbu?: number | null }) {
  const pct = (v?: number | null) => (v == null ? "—" : `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`);
  return (
    <div className="text-10 leading-tight text-gray-500" data-measured-util=""
      {...textTip("Of the measured kernel, on the card it ran on: achieved FLOP/s and HBM bytes/s over the peaks recorded at test")}>
      MFU {pct(mfu)} · MBU {pct(mbu)}
    </div>
  );
}

function occurrenceLabel(o: Occurrence): string {
  return o.iteration === null || o.iteration === undefined
    ? o.node : `${o.node}[${o.iteration}]`;
}

/** `out[17] → input_layernorm[18] (cut 0)`: what a pipeline hop moves. */
function crossingLabel(row: PerOpRow): string {
  if (row.cut === null || row.cut === undefined || !row.source_occurrence) return "";
  const sinks = Array.from(new Set((row.sink_occurrences || []).map(occurrenceLabel)));
  const shown = sinks.slice(0, 2).join(", ") + (sinks.length > 2 ? ` +${sinks.length - 2}` : "");
  return `${occurrenceLabel(row.source_occurrence)} → ${shown} (cut ${row.cut})`;
}

function fusedBadge(row: PerOpRow): SourceBadge | null {
  if (row.source !== "theoretical_fused") return null;
  return row.source_badge || null;
}

function OpLabel({ row, target, phase }: { row: PerOpRow; target?: GraphTarget; phase: PhaseKey }) {
  const [open, setOpen] = useState(false);
  const hideTimer = useRef<number | null>(null);
  const badge = fusedBadge(row);
  const fusedOp = badge?.fused_op;
  const show = () => {
    if (hideTimer.current) window.clearTimeout(hideTimer.current);
    hideTimer.current = null;
    setOpen(true);
  };
  const hide = () => {
    if (hideTimer.current) window.clearTimeout(hideTimer.current);
    hideTimer.current = window.setTimeout(() => setOpen(false), 180);
  };
  if (!fusedOp) {
    const name = row.name || row.display_path || row.op || row.spec_id || "—";
    const op = row.op && row.op !== name ? row.op : "";
    const crossing = crossingLabel(row);
    return (
      <>
        {name}
        {op ? <span className="ml-1.5 text-10 text-gray-400">{op}</span> : null}
        {crossing ? <span className="text-10 text-gray-500" data-crossing=""> {crossing}</span> : null}
      </>
    );
  }
  const absorbed = badge?.absorbed || [];
  const fw = badge?.framework
    ? (badge.framework_audited_at
      ? `${badge.framework} (audited at ${badge.framework_audited_at})`
      : badge.framework)
    : "";
  return (
    <span className="relative">
      <button
        type="button"
        className="inline-flex items-center gap-1 rounded-md bg-indigo-50 px-1.5 text-indigo-700 ring-1 ring-inset ring-indigo-200 hover:bg-indigo-100"
        data-fusion-pop=""
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
      >
        {fusedOp}
      </button>
      <span className="ml-1.5 font-sans text-10 text-indigo-400">
        {absorbed.length > 0 ? `${absorbed.length} ops fused` : `fused from ${row.op || "?"}`}
      </span>
      {open && (
        <div
          id="fusion-popover"
          className="absolute z-50 left-0 top-full mt-1 w-[28rem] max-w-[min(520px,calc(100vw-32px))] max-h-[70vh] overflow-x-hidden overflow-y-auto bg-white border border-gray-200 rounded-xl shadow-2xl text-left"
          onMouseEnter={show}
          onMouseLeave={hide}
        >
          <div className="px-4 py-2.5 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-semibold font-mono text-blue-800">{fusedOp}</span>
              <span className="text-10 text-gray-500">{absorbed.length} ops → 1 fused op</span>
            </div>
            {fw ? (
              <div className="mt-0.5 text-10 text-gray-500 font-mono">
                {fw}{badge?.source_anchor ? <> · <SourceAnchor anchor={badge.source_anchor} /></> : null}
              </div>
            ) : null}
          </div>
          <div className="px-4 py-3">
            <div className="text-10 text-gray-500 uppercase tracking-wide mb-1.5">Absorbed subgraph</div>
            <FusedSubgraph fusedOp={fusedOp} target={target} phase={phase} absorbed={absorbed} />
            <div className="text-10 text-gray-500 uppercase tracking-wide mt-3 mb-1.5">Shape evolution</div>
            {absorbed.length ? (
              <DataTable
                framed={false}
                rows={absorbed}
                rowKey={(a, i) => a.name || a.op || i}
                columns={[
                  {
                    key: "node",
                    header: "node",
                    className: "font-mono text-gray-800 max-w-[10rem] truncate",
                    render: (a) => a.name || "—",
                    sortValue: null,
                    cellProps: (a) => ({ title: a.name }),
                  },
                  {
                    key: "op",
                    header: "op",
                    className: "text-10 text-gray-500 font-mono",
                    render: (a) => a.op || "",
                    sortValue: null,
                  },
                  {
                    key: "shape_out",
                    header: "shape_out",
                    className: "font-mono text-10 text-indigo-700",
                    render: (a) => (a.shape_out ? `[${a.shape_out.join(", ")}]` : "—"),
                    sortValue: null,
                  },
                ]}
              />
            ) : (
              <p className="text-11 text-gray-400">No absorbed ops.</p>
            )}
            <p className="mt-3 text-10 text-gray-400 leading-snug">
              Intermediate shapes stay inside the fused kernel.
            </p>
          </div>
        </div>
      )}
    </span>
  );
}

function FusedSubgraph({
  fusedOp, target, phase, absorbed,
}: {
  fusedOp: string;
  target?: GraphTarget;
  phase: PhaseKey;
  absorbed: AbsorbedOp[];
}) {
  const [msg, setMsg] = useState("Loading…");
  const [graph, setGraph] = useState<{ nodes: BlockNode[]; edges: BlockEdge[] }>({ nodes: [], edges: [] });
  useEffect(() => {
    if (!target?.modelId.trim()) {
      setMsg("Missing model / TP context.");
      setGraph({ nodes: [], edges: [] });
      return;
    }
    const qs = computeGraphQuery(target, phase);
    let cancelled = false;
    fetchJson<GraphResp>(`/inference/compute-graph?${qs}`)
      .then((data) => {
        if (cancelled) return;
        const compound = findFusedCompound(data.graph?.nodes || [], fusedOp);
        if (!compound?.children?.length) {
          setGraph({ nodes: [], edges: [] });
          setMsg(`No fused subgraph for ${fusedOp}.`);
          return;
        }
        setGraph(compactFusedSubgraph(compound, nodesById(data.graph?.nodes || []), absorbed));
        setMsg("");
      })
      .catch((e) => {
        if (!cancelled) {
          setGraph({ nodes: [], edges: [] });
          setMsg(graphFailureMessage(e, "Failed"));
        }
      });
    return () => { cancelled = true; };
  }, [fusedOp, target, phase, absorbed]);

  if (msg) return <div id="fusion-popover-dag" className="rounded-lg p-4 text-xs text-gray-400">{msg}</div>;
  return (
    <div id="fusion-popover-dag" className="dag-container p-2 overflow-hidden">
      <BlockElkDag nodes={graph.nodes} edges={graph.edges} legend={false} fit />
    </div>
  );
}
