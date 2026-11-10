// The measured track's response contract and its one reading on the page
// (docs/auto_sweep.md §6).
//
// Two rules, decided:
//   - An end-to-end measured total (a step time, a throughput, a chart
//     point) is shown only when every op of the step is measured. Otherwise
//     the page shows coverage — "N / M ops measured" and the keys that are
//     not — and never a partial or mixed number under the measured name.
//     The one exception is opt-in and separately named: the partial track
//     (`t_partial_s`, a checkbox on the Inference and Gallery panels), the
//     step with every measured row at its measured time and every other row
//     at its SOL time, always shown with how much of it is real.
//   - A per-op row shows its own measured time whenever it has one, under
//     any coverage. 0 is a value (a zero-cost row is measured at 0).
import { fmtTime } from "./format";

/** Why a row has no measured time; each row is in exactly one state
 *  (`measured` is the fifth, counted in `MeasuredCoverage.measured`). */
export type CoverageStates = {
  /** A declared exact identity exists, but the corpus has no row for it. */
  missing?: string[];
  /** `latest` has no unique declared dispatch. */
  ambiguous?: string[];
  /** No signature or no shape: no key. */
  unplaced?: string[];
  /** The key could not be stated exactly; each carries its reason. */
  declined?: { key: string; reason: string }[];
};

/** Why each unplaced row has no key; parallel to `states.unplaced`, which
 * remains an opaque list of op names. */
export type UnplacedDetail = {
  op: string;
  aggregation_id: string;
  reason: "missing_signature" | "missing_projection" | "pipeline_stage";
};

/** How many of a step's per-op rows are measured, of how many (rows, not
 *  launches), and the keys of the rest by state. */
export type MeasuredCoverage = {
  measured: number;
  total: number;
  states?: CoverageStates;
  /** Optional because older responses predate structured unplaced accounting. */
  unplaced_details?: UnplacedDetail[];
  /** Rows per non-measured state (`states` names each key once). */
  rows_by_state?: Partial<Record<(typeof STATE_ORDER)[number], number>>;
};

/** The measured half of a phase as an estimate (and each estimate-sweep
 *  point) carries it. `measured_rows` / `total_rows` are what the backend
 *  sends before `coverage` exists; `coverage` wins when both are there. */
export type MeasuredPhase = {
  /** The measured step time; present only when coverage is complete. */
  t_measured_s?: number | null;
  coverage?: MeasuredCoverage | null;
  measured_rows?: number;
  total_rows?: number;
};

/** The phase's coverage, or null when the response says nothing about it. */
export function phaseCoverage(p: MeasuredPhase | null | undefined): MeasuredCoverage | null {
  if (!p) return null;
  if (p.coverage) return p.coverage;
  if (typeof p.total_rows === "number" && p.total_rows > 0) {
    return { measured: p.measured_rows ?? 0, total: p.total_rows };
  }
  return null;
}

/** Every row measured. Unknown coverage is not complete. */
export function isComplete(cov: MeasuredCoverage | null | undefined): boolean {
  return !!cov && cov.total > 0 && cov.measured === cov.total;
}

/** A total under the all-or-nothing rule: the value when it is a number and
 *  coverage is complete, else null. */
export function completeTotal(v: number | null | undefined, cov: MeasuredCoverage | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 && isComplete(cov) ? v : null;
}

/** A per-op row's own measured time: 0 is kept, absent and null are not. */
export function rowMeasured(v: number | null | undefined): number | null {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
}

/** A measured time for a cell: an explicit 0 for a row measured at 0, a
 *  dash for no value. */
export function fmtMeasured(v: number | null | undefined): string {
  const r = rowMeasured(v);
  if (r == null) return "—";
  return r === 0 ? "0" : fmtTime(r);
}

/** "12 / 40 ops measured", or "coverage not reported" when unknown. */
export function coverageText(cov: MeasuredCoverage | null | undefined): string {
  return cov ? `${cov.measured} / ${cov.total} ops measured` : "coverage not reported";
}

/** The non-measured states, in the order the page lists them. */
export const STATE_ORDER = ["missing", "ambiguous", "unplaced", "declined"] as const;

/** What each non-measured state means, as a tooltip says it. */
export const STATE_MEANING: Record<(typeof STATE_ORDER)[number], string> = {
  missing: "a key exists and Kernel Bench has no row for it: measure it",
  ambiguous: "rows exist, but more than one kernel of the framework; latest cannot pick one",
  unplaced: "no measurement signature or no measurable shape: no key",
  declined: "the key could not be stated exactly",
};

/** "missing 6 ops (3 keys) · unplaced 2": the rows per non-measured state,
 *  so they add up to the ops not measured, and the distinct keys behind them
 *  when fewer — one missing key repeated over layers is one thing to
 *  measure. */
export function stateCounts(cov: MeasuredCoverage | null | undefined): string {
  const st = cov?.states;
  if (!st) return "";
  return STATE_ORDER
    .map((k) => [k, st[k]?.length ?? 0, cov?.rows_by_state?.[k]] as const)
    .filter(([, n]) => n > 0)
    .map(([k, n, rows]) => (rows != null && rows !== n
      ? `${k} ${rows} ops (${n} ${k === "unplaced" ? "op name" : "key"}${n === 1 ? "" : "s"})`
      : `${k} ${n}`))
    .join(" · ");
}

/** The keys per state, one state per paragraph, for a tooltip; a long list
 *  is cut at `limit` with the remainder counted. */
export function stateKeysText(cov: MeasuredCoverage | null | undefined, limit = 12): string {
  const st = cov?.states;
  if (!st) return "";
  const cut = (keys: string[]) =>
    keys.slice(0, limit).join("\n") + (keys.length > limit ? `\n… +${keys.length - limit} more` : "");
  const parts: string[] = [];
  for (const k of STATE_ORDER) {
    const keys = k === "declined"
      ? (st.declined || []).map((d) => `${d.key} — ${d.reason}`)
      : (st[k] || []);
    if (keys.length) parts.push(`${k} (${keys.length}):\n${cut(keys)}`);
  }
  return parts.join("\n\n");
}

/** The measured value of a module-fold row as the backend stated it: a
 *  group's `t_measured_agg` (its share of the one measured formula, null
 *  unless every term under it is measured), a leaf's own time. Never a sum
 *  made here. */
export function groupMeasured(row: { t_measured_s?: number | null; t_measured_agg?: number | null }): number | null {
  return rowMeasured(row.t_measured_agg !== undefined ? row.t_measured_agg : row.t_measured_s);
}

/** The partial track's real-value coverage: "63% of time measured · 12 / 40
 *  ops measured" — the time share the backend states (`partial_measured_
 *  share`) and the row count. */
export function partialCoverageText(share: number | null | undefined,
                                    cov: MeasuredCoverage | null | undefined): string {
  const pct = typeof share === "number" && Number.isFinite(share)
    ? `${(share * 100).toFixed(share > 0 && share < 0.01 ? 1 : 0)}% of time measured` : "";
  return [pct, cov ? coverageText(cov) : ""].filter(Boolean).join(" · ");
}
