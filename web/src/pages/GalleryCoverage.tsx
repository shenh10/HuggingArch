// How much of each drawn model's decode frontier Kernel Bench has measured,
// at the frontier chart's GPU and context tab: drawn inside that chart's
// card, under it. The numbers are the server's
// (`gallery_frontier.frontier_coverage`); this file only draws them.
import { useState } from "react";
import { STATE_MEANING, STATE_ORDER } from "../measured";

/** One unmeasured key and how many frontier points need it. */
export type NeededKey = { key: string; reason?: string; points: number };

/** One context's coverage of a model's SOL frontier: its points, how many
 *  are fully measured, the best point's rows and the median share
 *  measured (null with no point), and per non-measured state how many
 *  distinct keys are in it and the most needed of them. */
export type FrontierCoverage = {
  points: number;
  complete: number;
  best: { measured: number; total: number } | null;
  median: number | null;
  states?: Partial<Record<(typeof STATE_ORDER)[number], { keys: number; top: NeededKey[] }>>;
};

/** A frontier entry's measured track: whether Kernel Bench rows were read
 *  (`reads`), the server reads none (`none`), the lake could not be read
 *  (`unavailable`), or our fold has an engine error (`engine_error`);
 *  coverage by context only when read. */
export type FrontierMeasuredTrack = {
  state: "reads" | "none" | "unavailable" | "engine_error";
  error?: string;
  coverage?: Record<string, FrontierCoverage>;
};

export type CoverageRow = {
  modelId: string;
  label: string;
  color: string;
  track?: FrontierMeasuredTrack;
};

/** A share as a whole percent, rounded down: 100% only when every op is. */
function pct(x: number): string {
  return `${Math.floor(x * 100)}%`;
}

function corpusHref(modelId: string, gpuSku: string): string {
  return `/kernel-corpus.html#${new URLSearchParams({ view: "workload", model: modelId, gpu: gpuSku, phase: "decode" })}`;
}

function CoverageDetail({ cov, modelId, gpuSku }: { cov: FrontierCoverage; modelId: string; gpuSku: string }) {
  return (
    <div className="g-cov-detail mt-2 mb-1 ml-5 text-11 text-gray-600">
      {STATE_ORDER.map((st) => {
        const s = cov.states?.[st];
        if (!s) return null;
        return (
          <div key={st} className="mb-2">
            <div title={STATE_MEANING[st]}>
              <span className="font-medium text-gray-700">{st}</span> — {s.keys} key{s.keys === 1 ? "" : "s"}
              {s.top.length < s.keys ? `, the ${s.top.length} most frontier points need:` : ":"}
            </div>
            <ul className="ml-3">
              {s.top.map((k) => (
                <li key={`${k.key}\u0000${k.reason ?? ""}`} className="font-mono break-all">
                  {k.key}{k.reason ? <span className="font-sans text-gray-400"> — {k.reason}</span> : null}
                  <span className="font-sans text-gray-400"> · {k.points} of {cov.points} point{cov.points === 1 ? "" : "s"}</span>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
      <a className="text-blue-700 hover:underline" href={corpusHref(modelId, gpuSku)}>
        This model's decode rows on {gpuSku} in Kernel Bench →
      </a>
    </div>
  );
}

/** Why no model has coverage to show, or null when some do. */
function whyNone(rows: CoverageRow[]): string | null {
  if (rows.some((r) => r.track?.state === "reads")) return null;
  if (rows.some((r) => r.track?.state === "unavailable")) return "The Kernel Bench lake could not be read; coverage is unknown, not zero.";
  if (rows.some((r) => r.track?.state === "engine_error")) return "A measured-fold engine error occurred; SOL is shown while the error is investigated.";
  if (rows.some((r) => r.track?.state === "none")) return "This server reads no Kernel Bench measurements.";
  return null;
}

export function CoverageTable({
  rows, ctx, gpuSku,
}: {
  rows: CoverageRow[];
  /** The chart's context tab, a `seq_lens` entry as a string. */
  ctx: string;
  gpuSku: string;
}) {
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const reason = whyNone(rows);
  const covOf = (r: CoverageRow) => r.track?.coverage?.[ctx] ?? null;
  // Most covered first: median share, then best point's share; models with
  // no coverage at this context last.
  const share = (c: FrontierCoverage | null) => (c?.median ?? -1);
  const bestShare = (c: FrontierCoverage | null) => (c?.best && c.best.total ? c.best.measured / c.best.total : -1);
  const sorted = [...rows].sort((a, b) => (share(covOf(b)) - share(covOf(a)))
    || (bestShare(covOf(b)) - bestShare(covOf(a))) || a.label.localeCompare(b.label));
  return (
    <div id="g-coverage" className="mt-4 pt-4 border-t border-gray-100">
      <h3 className="text-sm font-semibold text-gray-800">Kernel Bench coverage</h3>
      <p className="text-xs text-gray-400 mb-3">
        At the GPU and context above, for each drawn model, how many of the decode step's ops Kernel Bench has measured, over the points on its SOL frontier: the bar is the median point's share, "best" the most covered point's ops; a point is drawn measured only when every op is. Expand a model for the kernels still unmeasured, most needed first.
      </p>
      {reason ? (
        <div id="g-coverage-none" className="text-xs text-gray-500 py-4">{reason}</div>
      ) : (
        <ul>
          {sorted.map((r) => {
            const cov = covOf(r);
            const isOpen = !!open[r.modelId];
            const hasDetail = !!cov && Object.keys(cov.states ?? {}).length > 0;
            return (
              <li key={r.modelId} className="g-cov-row py-1 border-b border-gray-100 last:border-0" data-model={r.modelId}>
                <div className="flex items-center gap-3 text-xs">
                  <button type="button" disabled={!hasDetail}
                    className="w-4 text-gray-500 disabled:opacity-0"
                    aria-expanded={isOpen} aria-label={`unmeasured kernels of ${r.modelId}`}
                    onClick={() => setOpen((o) => ({ ...o, [r.modelId]: !o[r.modelId] }))}>
                    {isOpen ? "▾" : "▸"}
                  </button>
                  <span className="w-48 truncate text-gray-800" title={r.modelId}>
                    <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ background: r.color }} />
                    {r.label}
                  </span>
                  {r.track?.state !== "reads" ? (
                    <span
                      className={r.track?.state === "engine_error" ? "text-red-600" : "text-gray-400"}
                      title={r.track?.error}
                    >
                      {r.track?.state === "unavailable" ? "lake unreadable" : r.track?.state === "engine_error" ? "measured-fold error" : "not read"}
                    </span>
                  ) : !cov || !cov.points || cov.median == null ? (
                    <span className="text-gray-400">no frontier at this context</span>
                  ) : (
                    <>
                      <span className="relative w-40 h-2.5 bg-gray-100 rounded overflow-hidden" title={`median point: ${pct(cov.median)} of its ops measured`}>
                        <span className="absolute inset-y-0 left-0 rounded" style={{ width: pct(cov.median), background: r.color }} />
                      </span>
                      <span className="g-cov-median w-10 text-right tabular-nums text-gray-700">{pct(cov.median)}</span>
                      <span className="text-gray-500 tabular-nums">best {cov.best ? `${cov.best.measured} / ${cov.best.total}` : "–"} ops</span>
                      <span className="g-cov-complete text-gray-500 tabular-nums">{cov.complete} / {cov.points} points fully measured</span>
                    </>
                  )}
                </div>
                {isOpen && cov ? <CoverageDetail cov={cov} modelId={r.modelId} gpuSku={gpuSku} /> : null}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
