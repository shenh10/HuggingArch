import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { fetchJson, isAbort, type StillComputing } from "../api";
import { ComputeProgress } from "../ProgressBar";
import axes from "../gallery/axes.json";
import { fmtCompact, fmtSeqLabel } from "../format";
import { gpuOptions, skuOf, type GpuItem } from "../gpu";
import {
  FrontierChart,
  G_PALETTE,
  KvBarChart,
  ScatterChart,
  fmtYearMonth,
  parseReleaseDate,
  shortName,
  v1M,
  v256k,
  type FrontierPt,
  type FrontierSeries,
  type KvBarRow,
  type LineSeries,
} from "./GalleryCharts";
import { CoverageTable, type FrontierMeasuredTrack } from "./GalleryCoverage";
import { AS_SHIPPED } from "../precision";
import type { FormulaFields } from "../tooltip";
import { GALLERY_HASH_KEYS } from "../galleryHash";

type GpusResponse = { items?: GpuItem[] };

/** Several names as one label, their shared stem said once:
 *  `DeepSeek-R1 / V3 / V3.1`. The stem stops at a `-`, so a version is
 *  never cut in half. */
function jointName(names: string[]): string {
  if (names.length < 2) return names[0] ?? "";
  let stem = names[0];
  for (const n of names) while (!n.startsWith(stem)) stem = stem.slice(0, -1);
  stem = stem.slice(0, stem.lastIndexOf("-") + 1);
  return [names[0], ...names.slice(1).map((n) => n.slice(stem.length))].join(" / ");
}

type GalleryModel = {
  model_id: string;
  family?: string;
  attention_families?: string[];
  total_params?: number | null;
  active_params?: number | null;
  is_moe?: boolean | null;
  release_date?: string | null;
  error?: string | null;
  /** By leaderboard precision (`kv_precisions` id), then context: the
   *  cache as its formula (bytes are `result_bytes`). */
  kv_per_request?: Record<string, Record<string, FormulaFields>>;
  /** By leaderboard precision: what the cache is, as one digest — equal
   *  digests store the same cache. */
  kv_cache_identity?: Record<string, string>;
  /** By leaderboard precision: the cache's element dtypes, most bytes first. */
  kv_dtypes?: Record<string, string[]>;
};

/** `/gallery/models`: each model's own facts, no GPU or parallelism. */
type ModelFacts = { models?: GalleryModel[]; kv_title?: string; kv_precisions?: KvPrecision[]; errors_count?: number };

/** A precision the KV leaderboard can draw: a preset that sets only the cache. */
type KvPrecision = { id: string; label: string };

/** The frontier request for the selected GPU. */
type PerfState =
  | { phase: "loading"; progress?: { done: number; total: number }; jobsDone?: number }
  | { phase: "error"; message: string }
  | { phase: "ok" };

/** `/gallery/frontier`: each model's decode Pareto frontier by context, on
 *  one GPU type, over every layout up to `config.max_gpus` GPUs. */
type FrontierModel = {
  model_id: string;
  frontier?: Record<string, FrontierPt[]>;
  frontier_measured?: Record<string, FrontierPt[]>;
  /** The partial-measured frontier: measured ops measured, the rest at SOL. */
  frontier_partial?: Record<string, FrontierPt[]>;
  layouts?: { layout: string; points?: number; refused?: string; error?: string }[];
  error?: string | null;
  /** Why the speculative view draws nothing for this model (no MTP draft). */
  draft_unavailable?: string | null;
  /** Under a speculative view: each layout whose step at that K failed to
   *  price, and why (its points are missing from the curve). */
  draft_errors?: { layout: string; draft_tokens: number; error: string }[];
  /** Whether Kernel Bench rows were read for this view, and how much of
   *  each context's SOL frontier they cover. */
  measured_track?: FrontierMeasuredTrack;
};
type FrontierPayload = {
  config?: { gpu_sku?: string; max_gpus?: number; output_tokens?: number };
  seq_lens?: number[];
  models?: FrontierModel[];
};
/** The two time floors a frontier is priced under (µs): one link
 *  traversal's latency and the shortest one kernel launch takes. An absent
 *  one is the server's default. */
type Floors = { comm_latency_us?: number; launch_floor_us?: number };
/** Speculative decoding with the model's MTP draft: K drafted tokens per
 *  step kept at acceptance rate α (`axes.draft_tokens` /
 *  `axes.acceptance_rates`). Null is plain decode. */
type Draft = { k: number; alpha: number } | null;
/** A loaded frontier and what it was asked under (`key`: GPU, floors and
 *  draft). */
type FrontierView = { gpuSku: string; key: string; draft: Draft; data: FrontierPayload };

function frontierKey(gpuSku: string, floors: Floors, draft: Draft): string {
  return JSON.stringify([gpuSku, floors.comm_latency_us ?? null, floors.launch_floor_us ?? null,
    draft?.k ?? null, draft?.alpha ?? null]);
}

/** A non-negative number typed into a floor box, or undefined. */
function floorValue(text: string): number | undefined {
  const v = Number(text);
  return text.trim() !== "" && Number.isFinite(v) && v >= 0 ? v : undefined;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

const LINE_MAX = 20;

function seriesByFamily(
  models: GalleryModel[],
  point: (m: GalleryModel) => { x: number; y: number; label?: string; tip?: string; attention?: string[] } | null,
  famColor: Record<string, string>,
): LineSeries[] {
  const byFam: Record<string, LineSeries> = {};
  for (const m of models) {
    const p = point(m);
    if (!p) continue;
    const fam = m.family || "other";
    (byFam[fam] ||= { key: fam, label: fam, color: famColor[fam] || "#888", points: [] }).points.push(p);
  }
  return Object.keys(byFam).sort().map((k) => byFam[k]);
}

function Tabs({
  options, value, onChange, idPrefix,
}: {
  options: { key: string; label: string }[];
  value: string;
  onChange: (k: string) => void;
  idPrefix: string;
}) {
  return (
    <div className="inline-flex rounded-lg border border-gray-300 overflow-hidden text-xs">
      {options.map((o) => (
        <button
          key={o.key}
          id={`${idPrefix}-${o.key}`}
          type="button"
          className={`px-3 py-1.5 ${o.key === value ? "bg-gray-900 text-white" : "bg-white text-gray-600 hover:bg-gray-50"}`}
          onClick={() => onChange(o.key)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** The view a Gallery link carries: the frontier it shows — its card, its
 *  decode context, its KV precision and its draft (`k`, `alpha`; absent is
 *  plain decode). Written as the view changes (`replaceState`), read once
 *  at load; a value the page does not offer falls back to its default. */
type GalleryHash = Partial<Record<(typeof GALLERY_HASH_KEYS)[number], string>>;

function readGalleryHash(): GalleryHash {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const out: GalleryHash = {};
  for (const k of GALLERY_HASH_KEYS) {
    const v = params.get(k);
    if (v) out[k] = v;
  }
  return out;
}

function writeGalleryHash(h: GalleryHash) {
  const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  for (const k of GALLERY_HASH_KEYS) {
    if (h[k]) params.set(k, h[k]!);
    else params.delete(k);
  }
  const next = params.toString();
  const hash = next ? `#${next}` : "";
  if (hash !== window.location.hash) {
    history.replaceState(null, "", `${window.location.pathname}${window.location.search}${hash}`);
  }
}

/** `raw` as one of `options` (a number), or `fallback`. */
function pickNumber(raw: string | undefined, options: readonly number[], fallback: number): number {
  const n = Number(raw);
  return raw !== undefined && options.includes(n) ? n : fallback;
}

export function Gallery() {
  const [inbound] = useState(readGalleryHash);
  const [gpuSku, setGpuSku] = useState("");
  const [gpus, setGpus] = useState<GpuItem[]>([]);
  const [gpusLoading, setGpusLoading] = useState(true);
  const [gpuError, setGpuError] = useState<string | null>(null);
  // The models' own facts (timeline, filters, model list, KV) — served
  // without a sweep, so they show while the decode frontier computes.
  const [facts, setFacts] = useState<ModelFacts | null>(null);
  const [factsError, setFactsError] = useState<string | null>(null);
  const [factsTry, setFactsTry] = useState(0);
  // How far the server got computing the model facts, while it says so.
  const [factsProgress, setFactsProgress] = useState<{ done: number; total: number } | undefined>(undefined);
  const [fType, setFType] = useState("all");
  const [totMin, setTotMin] = useState("30");
  const [totMax, setTotMax] = useState("");
  const [actMin, setActMin] = useState("");
  const [actMax, setActMax] = useState("");
  // Performance panel: the per-model checkbox pick (null = default
  // top-by-params selection).
  // The decode frontier's context tab (a `seq_lens` entry, as a string),
  // and the last frontier that loaded, kept (and marked stale) while
  // another GPU computes or fails.
  const [decCtx, setDecCtx] = useState(inbound.ctx && /^\d+$/.test(inbound.ctx) ? inbound.ctx : "1024");
  const [front, setFront] = useState<FrontierView | null>(null);
  const [frontState, setFrontState] = useState<PerfState>({ phase: "loading" });
  const [frontTry, setFrontTry] = useState(0);
  const [kvPrec, setKvPrec] = useState(inbound.kv || AS_SHIPPED);
  const [picked, setPicked] = useState<Record<string, boolean> | null>(null);
  // The models the frontier should price first: those on the chart (the
  // picked ones, or the default top-by-params). Sent with every retry, so a
  // model picked while the view computes jumps the server's queue.
  const priorityRef = useRef<string[]>([]);
  // The floors the frontier is asked under: the server's defaults
  // (`/inference/defaults`) until edited; null until they are known. An
  // edit applies on blur / Enter, not per keystroke — each point's step is
  // cached as a function of both floors, so a new value is a read of the
  // same cached entries, not a new walk.
  const [floors, setFloors] = useState<Floors | null>(null);
  const [floorDraft, setFloorDraft] = useState({ comm: "", launch: "" });
  // Speculative decoding: off, or K and α from the page's pickers. The
  // entry the warm builds holds every point's step at every K (one walk of
  // each layout, every K read off it), so a K or α is a read of the same
  // cached entry as the plain frontier.
  const [partialOn, setPartialOn] = useState(false);
  const [specOn, setSpecOn] = useState(inbound.k !== undefined
    && axes.draft_tokens.includes(Number(inbound.k)));
  const [specK, setSpecK] = useState(pickNumber(inbound.k, axes.draft_tokens, axes.draft_tokens[0]));
  const [specAlpha, setSpecAlpha] = useState(
    pickNumber(inbound.alpha, axes.acceptance_rates, axes.acceptance_default));
  const draft: Draft = useMemo(
    () => (specOn ? { k: specK, alpha: specAlpha } : null), [specOn, specK, specAlpha]);

  useEffect(() => {
    fetchJson<{ comm_latency_us?: number; launch_floor_us?: number }>("/inference/defaults")
      .then((d) => {
        setFloors({ comm_latency_us: d.comm_latency_us, launch_floor_us: d.launch_floor_us });
        setFloorDraft({ comm: String(d.comm_latency_us ?? ""), launch: String(d.launch_floor_us ?? "") });
      })
      // Without them the server's own defaults price the frontier.
      .catch(() => setFloors({}));
  }, []);

  const applyFloors = () => {
    const next: Floors = {
      comm_latency_us: floorValue(floorDraft.comm),
      launch_floor_us: floorValue(floorDraft.launch),
    };
    if (floors && frontierKey("", next, null) === frontierKey("", floors, null)) return;
    setFloors(next);
  };

  useEffect(() => {
    fetchJson<GpusResponse>("/inference/gpus")
      .then((j) => {
        // The gallery's GPUs are its own series (`axes.gpus`); the frontier
        // route refuses another.
        const items = (j.items || []).filter((g) => axes.gpus.includes(skuOf(g)));
        const skus = items.map(skuOf);
        setGpus(items);
        const landing = axes.landing_gpu;
        if (inbound.gpu && skus.includes(inbound.gpu)) setGpuSku(inbound.gpu);
        else if (skus.includes(landing)) setGpuSku(landing);
        else if (skus[0]) setGpuSku(skus[0]);
        else setGpuSku("");
        setGpuError(skus.length ? null : "the GPU list is empty");
      })
      .catch((e) => {
        setGpus([]);
        setGpuSku("");
        setGpuError(errorText(e));
      })
      .finally(() => setGpusLoading(false));
  }, []);

  // The view on screen, in the link — once the card is known, so a link is
  // never rewritten to the empty card of a page still loading.
  useEffect(() => {
    if (!gpuSku) return;
    writeGalleryHash({
      gpu: gpuSku, ctx: decCtx,
      kv: kvPrec === AS_SHIPPED ? undefined : kvPrec,
      ...(specOn ? { k: String(specK), alpha: String(specAlpha) } : {}),
    });
  }, [gpuSku, decCtx, kvPrec, specOn, specK, specAlpha]);

  useEffect(() => {
    const ctrl = new AbortController();
    setFactsError(null);
    fetchJson<ModelFacts>("/gallery/models", { signal: ctrl.signal },
      (info: StillComputing) => setFactsProgress(info.progress))
      .then((body) => setFacts(body))
      .catch((e) => {
        if (!isAbort(e)) setFactsError(errorText(e));
      });
    return () => ctrl.abort();
  }, [factsTry]);

  // The decode frontier spans every layout it prices, so it follows the GPU
  // alone. No deadline of our own: `fetchJson` keeps asking while the
  // server says it is still computing, and gives up after its window.
  useEffect(() => {
    if (!gpuSku || floors === null) return;
    const ctrl = new AbortController();
    setFrontState({ phase: "loading" });
    const query: Record<string, string> = { gpu_sku: gpuSku };
    if (floors.comm_latency_us !== undefined) query.comm_latency_us = String(floors.comm_latency_us);
    if (floors.launch_floor_us !== undefined) query.launch_floor_us = String(floors.launch_floor_us);
    if (draft) {
      query.draft_tokens = String(draft.k);
      query.acceptance_rate = String(draft.alpha);
    }
    const key = frontierKey(gpuSku, floors, draft);
    // The models the server published so far for this view, drawn as they
    // land; the cursor tells the server which it need not send again.
    const cursor = { tag: 0, count: 0 };
    let arrived: FrontierModel[] = [];
    const url = () => {
      const q = new URLSearchParams(query);
      if (priorityRef.current.length) q.set("priority", priorityRef.current.join(","));
      if (cursor.tag) {
        q.set("have_tag", String(cursor.tag));
        q.set("have", String(cursor.count));
      }
      return `/gallery/frontier?${q}`;
    };
    fetchJson<FrontierPayload>(url, { signal: ctrl.signal }, (info: StillComputing) => {
        const part = info.partial;
        if (part) {
          if (part.tag !== cursor.tag) arrived = [];
          const updates = part.items as FrontierModel[];
          // A measured update replaces the model's SOL partial by id; items
          // are a stream of current values, not append-only rows.
          const byId = new Map(arrived.map((m) => [m.model_id, m]));
          updates.forEach((m) => byId.set(m.model_id, m));
          arrived = [...byId.values()];
          cursor.tag = part.tag;
          cursor.count = part.count;
          if (part.head) {
            setFront({ gpuSku, key, draft, data: { ...(part.head as FrontierPayload), models: arrived } });
          }
        }
        setFrontState({ phase: "loading", progress: info.progress, jobsDone: info.jobsDone });
      })
      .then((body) => {
        setFront({ gpuSku, key, draft, data: body });
        setFrontState({ phase: "ok" });
      })
      .catch((e) => {
        // An abort is this effect's cleanup: a newer GPU replaced it.
        if (isAbort(e)) return;
        setFrontState({ phase: "error", message: errorText(e) });
      });
    return () => ctrl.abort();
  }, [gpuSku, floors, draft, frontTry]);

  const factModels = facts?.models;
  const allOk = useMemo(() => (factModels || []).filter((m) => !m.error), [factModels]);
  const famColor = useMemo(() => {
    const fams = [...new Set(allOk.map((m) => m.family || "other"))].sort();
    const out: Record<string, string> = {};
    fams.forEach((f, i) => { out[f] = G_PALETTE[i % G_PALETTE.length]; });
    return out;
  }, [allOk]);

  const filtered = useMemo(() => {
    const nB = (s: string) => {
      const v = parseFloat(s);
      return Number.isFinite(v) ? v * 1e9 : null;
    };
    const tMin = nB(totMin), tMax = nB(totMax), aMin = nB(actMin), aMax = nB(actMax);
    return allOk.filter((m) => {
      if (fType === "moe" && m.is_moe !== true) return false;
      if (fType === "dense" && m.is_moe === true) return false;
      const tot = m.total_params || 0, act = m.active_params || 0;
      if (tMin != null && tot < tMin) return false;
      if (tMax != null && tot > tMax) return false;
      if (aMin != null && act < aMin) return false;
      if (aMax != null && act > aMax) return false;
      return true;
    });
  }, [allOk, fType, totMin, totMax, actMin, actMax]);

  // Biggest-first order drives both the default Performance selection and
  // the checkbox bar.
  const sortedFiltered = useMemo(
    () => [...filtered].sort((a, b) => (b.total_params || 0) - (a.total_params || 0)),
    [filtered],
  );

  const perfModels = useMemo(() => {
    if (picked == null) return sortedFiltered.slice(0, LINE_MAX);
    return sortedFiltered.filter((m) => picked[m.model_id]);
  }, [sortedFiltered, picked]);
  const perfSet = useMemo(() => new Set(perfModels.map((m) => m.model_id)), [perfModels]);
  useEffect(() => {
    priorityRef.current = perfModels.map((m) => m.model_id);
  }, [perfModels]);
  // One colour per plotted line: the palette is dealt over the checked
  // models (sorted by id, so a model keeps its colour while others toggle
  // around it as far as the order allows), not over all sixty — dealt over
  // all, twenty plotted lines repeated colours. Unchecked chips are grey.
  const modelColor = useMemo(() => {
    const out: Record<string, string> = {};
    perfModels.map((m) => m.model_id).sort().forEach((id, i) => {
      out[id] = G_PALETTE[i % G_PALETTE.length];
    });
    return out;
  }, [perfModels]);

  const toggleModel = (id: string) => {
    const cur = new Set(perfSet);
    if (cur.has(id)) cur.delete(id);
    else cur.add(id);
    const next: Record<string, boolean> = {};
    cur.forEach((x) => { next[x] = true; });
    setPicked(next);
  };

  // Vendor two-level pick: groups in biggest-model-first order (the family's
  // largest member's rank), models inside a group keep the params-desc order.
  const byFamily = useMemo(() => {
    const groups = new Map<string, typeof sortedFiltered>();
    sortedFiltered.forEach((m) => {
      const fam = m.family || "other";
      if (!groups.has(fam)) groups.set(fam, []);
      groups.get(fam)!.push(m);
    });
    return [...groups.entries()].map(([fam, models]) => ({ fam, models }));
  }, [sortedFiltered]);

  const toggleFamily = (ids: string[]) => {
    const cur = new Set(perfSet);
    if (ids.every((id) => cur.has(id))) {
      ids.forEach((id) => cur.delete(id));
      // Same invariant as the per-model guard: at least one model stays on.
      if (!cur.size && ids.length) cur.add(ids[0]);
    } else {
      ids.forEach((id) => cur.add(id));
    }
    const next: Record<string, boolean> = {};
    cur.forEach((x) => { next[x] = true; });
    setPicked(next);
  };

  const maxActive = Math.max(...filtered.map((m) => m.active_params || 0), 1);
  // symbolSize is a diameter; keeping it proportional to sqrt(active) makes
  // bubble AREA proportional to activated parameters, the honest 2-D encoding.
  const activeBubble = (active: number) => 7 + 21 * Math.sqrt(active / maxActive);

  const timeline = seriesByFamily(filtered, (m) => {
    const x = parseReleaseDate(m.release_date);
    const total = m.total_params || 0;
    const active = m.active_params || 0;
    if (x == null || !total || !active) return null;
    return {
      x,
      y: total,
      // The name only: the bubble sits at TOTAL params, so an active count
      // beside it read as that height. Active is the bubble's area and the
      // tooltip's.
      label: shortName(m.model_id),
      tip: `${m.release_date} · ${fmtCompact(total)} total · ${fmtCompact(active)} active · ${(
        (active / total) * 100
      ).toFixed(1)}% active`,
      attention: m.attention_families,
      size: activeBubble(active),
      full: m.model_id,
    };
  }, famColor);

  // Models the timeline cannot place: no release date on record.
  const undated = filtered.filter((m) => parseReleaseDate(m.release_date) == null);

  // KV leaderboard rows — the checked (Performance) models with a value at
  // either context, sorted small → large: primarily by 1M, then by 256K.
  // Checkpoints of one family whose caches are the same
  // (`kv_cache_identity`, eval's) are one row: the chart cannot tell them
  // apart, so it names them together instead of drawing one bar per name.
  const kvRows: KvBarRow[] = useMemo(() => {
    const groups = new Map<string, GalleryModel[]>();
    for (const m of perfModels) {
      const identity = m.kv_cache_identity?.[kvPrec];
      const key = identity ? `${m.family ?? ""}\u0000${identity}` : m.model_id;
      groups.set(key, [...(groups.get(key) ?? []), m]);
    }
    return [...groups.values()]
      .map((ms) => {
        const m = ms[0];
        return {
          key: m.model_id,
          label: jointName(ms.map((x) => shortName(x.model_id))),
          full: ms.map((x) => x.model_id).join(", "),
          f256k: m.kv_per_request?.[kvPrec]?.["262144"] ?? null,
          f1M: m.kv_per_request?.[kvPrec]?.["1048576"] ?? null,
          // As shipped, each model's own cache dtype; under a preset every
          // row is the preset's, which the switcher already says.
          dtypes: kvPrec === AS_SHIPPED ? m.kv_dtypes?.[kvPrec] : undefined,
        };
      })
      .filter((r) => (v1M(r) ?? 0) > 0 || (v256k(r) ?? 0) > 0)
      .sort((a, b) => ((v1M(a) ?? Infinity) - (v1M(b) ?? Infinity)) || ((v256k(a) ?? Infinity) - (v256k(b) ?? Infinity)));
  }, [perfModels, kvPrec]);
  const kvPrecisions = facts?.kv_precisions ?? [];

  const frontById = useMemo(
    () => new Map((front?.data.models || []).map((m) => [m.model_id, m])),
    [front],
  );
  // Under a speculative view, models with no MTP draft are not drawn: they
  // are listed under the chart instead.
  const noDraft = useMemo(() => (front?.draft == null ? new Set<string>() : new Set(
    (front.data.models || []).filter((m) => m.draft_unavailable).map((m) => m.model_id))), [front]);
  const frontierSeries: FrontierSeries[] = useMemo(() => perfModels
    .filter((m) => !noDraft.has(m.model_id)).map((m) => ({
    key: m.model_id,
    label: shortName(m.model_id),
    full: m.model_id,
    color: modelColor[m.model_id] || "#888",
    points: frontById.get(m.model_id)?.frontier?.[decCtx] || [],
    measured: frontById.get(m.model_id)?.frontier_measured?.[decCtx] || [],
    partial: partialOn ? frontById.get(m.model_id)?.frontier_partial?.[decCtx] || [] : [],
  })).filter((s) => s.points.length), [perfModels, modelColor, frontById, decCtx, noDraft, partialOn]);

  // What the page says while a part of it is not there yet, or failed.
  const retryFacts = (
    <button id="g-facts-retry" type="button" className="ml-2 px-2 py-0.5 border border-gray-300 rounded text-gray-700 bg-white hover:bg-gray-50" onClick={() => setFactsTry((n) => n + 1)}>Retry</button>
  );
  // A cold page has two computes in flight — the model facts and the decode
  // frontier — but draws only one progress bar: the banner under the title.
  // Every waiting card says so in plain text instead. A bar per card meant
  // bars of different widths sliding out of step all over the page.
  const factsComputing = !factModels && !factsError;
  const factsEmpty = factsError && !factModels
    ? <span>Could not load the model list: {factsError}{retryFacts}</span>
    : (factModels ? undefined : <span id="g-models-waiting">Loading models…</span>);

  const frontSeq = front?.data.seq_lens || [];
  const decodeTabs = frontSeq.map((x) => ({ key: String(x), label: fmtSeqLabel(x) }));
  // A link's context or KV precision this page does not offer reads as the
  // default, once the offered ones are known.
  useEffect(() => {
    if (frontSeq.length && !frontSeq.map(String).includes(decCtx)) {
      setDecCtx(frontSeq.map(String).includes("1024") ? "1024" : String(frontSeq[0]));
    }
  }, [frontSeq, decCtx]);
  useEffect(() => {
    if (kvPrecisions.length && !kvPrecisions.some((p) => p.id === kvPrec)) setKvPrec(AS_SHIPPED);
  }, [kvPrecisions, kvPrec]);
  const frontLoading = frontState.phase === "loading";
  const frontProgress = frontState.phase === "loading" ? frontState.progress : undefined;
  const frontJobs = frontState.phase === "loading" ? frontState.jobsDone : undefined;
  const frontWaitingText = `Computing the decode frontier on ${gpuSku}…`;
  const frontWaiting = frontLoading
    ? <span id="g-front-waiting">{frontWaitingText}</span>
    : null;
  // The server's progress counts models; between two of them the compute
  // jobs it finished say the work is moving (a model is dozens of layouts).
  const frontNote = "Every model is priced under every layout; a GPU not yet computed can take hours, and each model is kept once computed — leaving the page stops the computation, and coming back resumes from the models already kept."
    + (frontJobs ? ` ${frontJobs.toLocaleString()} compute jobs finished so far.` : "");
  // The frontier is the long pole, so the one bar tracks it while both run;
  // the model list gets the bar only if the frontier lands first.
  const galleryComputing: { label: string; progress?: { done: number; total: number }; unit?: string; note?: string } | null =
    frontLoading
      ? {
          label: frontWaitingText,
          progress: frontProgress,
          unit: "models",
          note: factsComputing ? `${frontNote} The model list is loading alongside it.` : frontNote,
        }
      : factsComputing
        ? { label: "Loading models", progress: factsProgress, unit: "models" }
        : null;
  const frontFailed = frontState.phase === "error"
    ? <span>Could not load the decode frontier on {gpuSku}: {frontState.message}
        <button id="g-front-retry" type="button" className="ml-2 px-2 py-0.5 border border-gray-300 rounded text-gray-700 bg-white hover:bg-gray-50" onClick={() => setFrontTry((n) => n + 1)}>Retry</button></span>
    : null;
  const frontStale = front != null && floors !== null
    && front.key !== frontierKey(gpuSku, floors, draft);
  // Checked models that draw no frontier at this context: every layout
  // refused or failed, or none serves this context.
  // While the view computes, a model not yet arrived is pending, not missing.
  const frontMissing = front == null ? [] : perfModels
    .filter((m) => !frontLoading || frontById.has(m.model_id))
    .filter((m) => !noDraft.has(m.model_id))
    .filter((m) => !(frontById.get(m.model_id)?.frontier?.[decCtx] || []).length)
    .map((m) => shortName(m.model_id));
  const frontNoDraft = perfModels.filter((m) => noDraft.has(m.model_id)).map((m) => shortName(m.model_id));
  // Checked models whose speculative step failed under some layout: those
  // layouts' points are missing at this K, and the server says why.
  const frontDraftErrors = front?.draft == null ? [] : perfModels
    .flatMap((m) => (frontById.get(m.model_id)?.draft_errors || [])
      .map((e) => `${shortName(m.model_id)} ${e.layout}: ${e.error}`));
  const frontDraft = front?.draft ?? null;
  const maxGpus = front?.data.config?.max_gpus;
  const decodeChart = (
    <FrontierChart
      empty={frontFailed ?? frontWaiting
        ?? (front != null && !frontStale ? "no model serves this context on this GPU" : undefined)}
      stale={frontStale ? <span>Showing {front?.gpuSku} — {frontFailed ?? (frontState.phase === "loading" ? frontWaitingText : null)}</span> : undefined}
      title={`Decode Pareto frontier — ${fmtSeqLabel(Number(decCtx))} context · ${front?.gpuSku ?? gpuSku}`}
      subtitle={`Each point is the best layout${maxGpus ? ` of up to ${maxGpus} GPUs` : ""} (TP / EP / DP / CP) at one batch per GPU, named where it changes (hover a curve for every point); up and to the right is better. ${frontDraft
        ? ` Speculative decoding with the model's MTP draft, K = ${frontDraft.k} at acceptance α = ${frontDraft.alpha}: each step counts the K draft forwards and the verify over K + 1 positions, and keeps 1 + α·K tokens per request. SOL only — Kernel Bench measures the plain step.`
        : ` Solid is SOL; dashed is measured where Kernel Bench covers the step.${partialOn
          ? " Dotted is partial measured: measured ops at their measured time, the rest at SOL — hover a point for the share of its time that is measured." : ""}`}`}
      controls={<Tabs options={decodeTabs} value={decCtx} onChange={setDecCtx} idPrefix="g-dec-tab" />}
      series={frontierSeries}
      height={500}
      footer={
        <CoverageTable
          rows={frontierSeries.map((s) => ({ modelId: s.key, label: s.label, color: s.color,
            track: frontById.get(s.key)?.measured_track }))}
          ctx={decCtx}
          gpuSku={front?.gpuSku ?? gpuSku}
          speculative={front?.draft != null}
        />
      }
    />
  );
  const status: ReactNode = gpuError
    ? `Failed to load GPUs: ${gpuError}`
    : frontLoading
      ? `Computing ${gpuSku}${frontProgress ? ` — ${frontProgress.done} / ${frontProgress.total} models` : ""}…`
      : frontState.phase === "error"
        ? "Failed — see the chart below"
        : (() => {
            const models = front?.data.models || [];
            const failed = models.filter((m) => m.error).map((m) => shortName(m.model_id));
            let msg = `${models.length - failed.length}/${models.length} models · ${front?.data.config?.gpu_sku ?? gpuSku}`;
            if (failed.length) msg += ` · ${failed.length} failed: ${failed.join(", ")}`;
            return msg;
          })();
  const frontMissingNote = frontMissing.length ? (
    <p id="g-front-missing" className="-mt-3 mb-6 px-1 text-xs text-gray-500">
      No frontier at {fmtSeqLabel(Number(decCtx))}: {frontMissing.join(", ")}
    </p>
  ) : null;
  const frontNoDraftNote = frontNoDraft.length ? (
    <p id="g-front-no-draft" className="-mt-3 mb-6 px-1 text-xs text-gray-500">
      Not drawn with speculative decoding (no MTP draft): {frontNoDraft.join(", ")}
    </p>
  ) : null;
  const frontDraftErrorNote = frontDraftErrors.length ? (
    <p id="g-front-draft-errors" className="-mt-3 mb-6 px-1 text-xs text-amber-800">
      Speculative step failed to price, those layouts not drawn: {frontDraftErrors.join("; ")}
    </p>
  ) : null;
  const frontFootnote = (
    <p id="g-front-footnote" className="-mt-3 mb-6 px-1 text-xs text-gray-400">
      SOL is a speed-of-light roofline: each kernel at max(compute, memory, comm, launch), kernels serial with no compute/comm overlap, and one link latency for every fabric (NVLink, PCIe and RDMA alike).
    </p>
  );

  return (
    <main className="w-full page-shell py-4 sm:py-8 flex-1">
      <section className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Model Gallery</h1>
      </section>

      {galleryComputing ? (
        // A corner note, not a banner: the page draws each model as it
        // lands, so what is still computing is a footnote to what is there.
        <div id="g-computing" role="status" title={galleryComputing.note}
          className="fixed bottom-4 right-4 z-40 max-w-xs rounded-md border border-amber-200 bg-white/95 px-3 py-2 text-xs text-gray-600 shadow-sm">
          <ComputeProgress
            compact
            label={galleryComputing.label}
            progress={galleryComputing.progress}
            unit={galleryComputing.unit}
          />
        </div>
      ) : null}

      <ScatterChart
        empty={factsEmpty}
        title="Model scale over time — total size and activated params"
        subtitle="Area = activated params. Colour = family."
        series={timeline}
        xTitle="Release date"
        yTitle="Total parameters (log)"
        yLog
        xFmt={fmtYearMonth}
        xInterval={0.25}
        yFmt={fmtCompact}
        height={560}
        note={undated.length ? (
          <p id="g-undated" className="mt-2 text-11 text-gray-500">
            Not placed — no release date on record: {undated.map((m) => shortName(m.model_id)).join(", ")}.
          </p>
        ) : null}
      />

      <section className="card p-4 mb-6">
        <div className="text-xs font-semibold text-gray-700 mb-2">
          Filter models
        </div>
        <div className="flex flex-wrap items-end gap-4">
          <label className="block text-xs">
            <span className="font-medium text-gray-600">Type</span>
            <select id="g-f-type" className="mt-1 block text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white" value={fType} onChange={(e) => setFType(e.target.value)}>
              <option value="all">All</option>
              <option value="moe">MoE only</option>
              <option value="dense">Dense only</option>
            </select>
          </label>
          <label className="block text-xs">
            <span className="font-medium text-gray-600">Total params (B)</span>
            <div className="mt-1 flex items-center gap-1">
              <input id="g-f-tot-min" type="number" min={0} step={1} placeholder="min" className="text-sm border border-gray-300 rounded-lg px-2 py-1.5 w-20" value={totMin} onChange={(e) => setTotMin(e.target.value)} />
              <span className="text-gray-400">–</span>
              <input id="g-f-tot-max" type="number" min={0} step={1} placeholder="max" className="text-sm border border-gray-300 rounded-lg px-2 py-1.5 w-20" value={totMax} onChange={(e) => setTotMax(e.target.value)} />
            </div>
          </label>
          <label className="block text-xs">
            <span className="font-medium text-gray-600">Activated params (B)</span>
            <div className="mt-1 flex items-center gap-1">
              <input id="g-f-act-min" type="number" min={0} step={0.5} placeholder="min" className="text-sm border border-gray-300 rounded-lg px-2 py-1.5 w-20" value={actMin} onChange={(e) => setActMin(e.target.value)} />
              <span className="text-gray-400">–</span>
              <input id="g-f-act-max" type="number" min={0} step={0.5} placeholder="max" className="text-sm border border-gray-300 rounded-lg px-2 py-1.5 w-20" value={actMax} onChange={(e) => setActMax(e.target.value)} />
            </div>
          </label>
          <button id="g-f-reset" type="button" className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg text-gray-600 hover:bg-gray-50" onClick={() => {
            setTotMax(""); setActMin(""); setActMax(""); setTotMin("30"); setFType("all");
          }}>Reset</button>
          <div className="flex-1" />
          <div id="g-f-count" className="text-xs text-gray-500">{factModels ? `${filtered.length} / ${allOk.length} models shown` : "Loading models…"}</div>
        </div>
      </section>

      <section className="card p-4 mb-6">
        <div className="flex flex-wrap items-center gap-2 mb-2 text-xs">
          <span className="font-semibold text-gray-700">Models</span>
          <span className="font-normal text-gray-400">— the KV and decode charts plot the checked ones</span>
          <span id="g-pick-count" className="text-gray-400">{factModels ? `${perfModels.length} / ${filtered.length} in the charts` : "Loading models…"}</span>
          <button id="g-pick-all" type="button" className="px-2 py-0.5 border border-gray-300 rounded text-gray-600 hover:bg-gray-50" onClick={() => {
            const next: Record<string, boolean> = {};
            filtered.forEach((m) => { next[m.model_id] = true; });
            setPicked(next);
          }}>All</button>
          <button id="g-pick-default" type="button" className="px-2 py-0.5 border border-gray-300 rounded text-gray-600 hover:bg-gray-50" onClick={() => setPicked(null)}>Default</button>
        </div>
        <div className="max-h-64 overflow-y-auto divide-y divide-gray-100">
          {byFamily.map(({ fam, models }) => {
            const ids = models.map((m) => m.model_id);
            const on = ids.filter((id) => perfSet.has(id)).length;
            return (
              <div key={fam} className="flex items-start gap-3 py-2">
                <label className="group shrink-0 w-40 inline-flex items-center gap-2 pt-1 text-xs cursor-pointer select-none">
                  <input
                    id={`g-pick-fam-${fam}`}
                    type="checkbox"
                    className="accent-gray-700"
                    checked={on === ids.length && ids.length > 0}
                    ref={(el) => { if (el) el.indeterminate = on > 0 && on < ids.length; }}
                    onChange={() => toggleFamily(ids)}
                  />
                  {/* The family's colour on the timeline above; the chips
                      beside it carry each checked model's line colour. */}
                  <span
                    className="inline-block w-2 h-2 rounded-sm shrink-0"
                    title="Colour on the timeline"
                    style={{ background: famColor[fam] || "#888" }}
                  />
                  <span className="font-semibold text-gray-800 truncate group-hover:text-gray-950">{fam}</span>
                  <span className="ml-auto px-1.5 rounded-full bg-gray-100 text-10 tabular-nums text-gray-500">{on}/{ids.length}</span>
                </label>
                <div className="flex flex-wrap gap-1.5 min-w-0">
                  {models.map((m) => {
                    const checked = perfSet.has(m.model_id);
                    // At least one model stays on — an empty pick is
                    // a dead chart.
                    const locked = checked && perfModels.length <= 1;
                    return (
                      <label
                        key={m.model_id}
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs transition-colors select-none has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-gray-400 ${
                          checked
                            ? "bg-white border-gray-300 text-gray-800 shadow-sm"
                            : "bg-gray-50 border-transparent text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                        } ${locked ? "cursor-not-allowed" : "cursor-pointer"}`}
                      >
                        <input
                          type="checkbox"
                          className="sr-only"
                          checked={checked}
                          disabled={locked}
                          onChange={() => toggleModel(m.model_id)}
                        />
                        <span
                          className={`inline-block w-2 h-2 rounded-full ${checked ? "" : "opacity-30"}`}
                          style={{ background: modelColor[m.model_id] || "#9ca3af" }}
                        />
                        {shortName(m.model_id)}
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
          {!sortedFiltered.length && (
            <span className="text-xs text-gray-400">{factModels ? "no models pass the filter" : factsEmpty}</span>
          )}
        </div>
      </section>

      <KvBarChart
        title={facts?.kv_title ?? ""}
        controls={kvPrecisions.length > 1
          ? <Tabs options={kvPrecisions.map((p) => ({ key: p.id, label: p.label }))} value={kvPrec} onChange={setKvPrec} idPrefix="g-kv-prec" />
          : null}
        empty={factsEmpty}
        subtitle="256K / 1M context."
        rows={kvRows}
      />

      <section className="card p-4 mb-6">
        <div className="text-xs font-semibold text-gray-700 mb-2">
          Performance
        </div>
        <div className="flex flex-wrap items-end gap-4">
          <label className="block text-xs max-w-full">
            <span className="font-medium text-gray-600">GPU</span>
            {gpus.length ? (
              <select id="g-gpu" className="mt-1 block max-w-full text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white min-w-[10rem]" value={gpuSku} onChange={(e) => setGpuSku(e.target.value)}>
                {gpuOptions(gpus).map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            ) : (
              <select id="g-gpu" className="mt-1 block max-w-full text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white min-w-[10rem]" value="" disabled>
                <option value="">{gpusLoading ? "Loading GPUs…" : "No GPUs loaded"}</option>
              </select>
)}
          </label>
          <label className="block text-xs" title="One link traversal's latency, the same on every link. A collective pays it once per hop its schedule waits on: an all-reduce of N ranks 2·log2 N, a reduce-scatter / all-gather / all-to-all log2 N. Applies on Enter or leaving the box.">
            <span className="font-medium text-gray-600">Link latency (µs)</span>
            <input id="g-comm-latency" type="number" min={0} step={0.5}
              className="mt-1 block w-28 text-sm border border-gray-300 rounded-lg px-3 py-1.5"
              value={floorDraft.comm}
              onChange={(e) => setFloorDraft((d) => ({ ...d, comm: e.target.value }))}
              onBlur={applyFloors}
              onKeyDown={(e) => { if (e.key === "Enter") applyFloors(); }} />
          </label>
          <label className="block text-xs" title="The shortest one kernel launch takes, however little it does. Applies on Enter or leaving the box.">
            <span className="font-medium text-gray-600">Kernel launch floor (µs)</span>
            <input id="g-launch-floor" type="number" min={0} step={0.5}
              className="mt-1 block w-28 text-sm border border-gray-300 rounded-lg px-3 py-1.5"
              value={floorDraft.launch}
              onChange={(e) => setFloorDraft((d) => ({ ...d, launch: e.target.value }))}
              onBlur={applyFloors}
              onKeyDown={(e) => { if (e.key === "Enter") applyFloors(); }} />
          </label>
          <label className="flex items-center gap-1.5 text-xs pb-2" title="Partial measured: each op Kernel Bench has measured at its measured time, every other op at its SOL time. Drawn dotted; each point says how much of its time is real measurement. Plain decode only.">
            <input id="g-partial-on" type="checkbox" checked={partialOn} onChange={(e) => setPartialOn(e.target.checked)} />
            <span className="font-medium text-gray-600">Partial measured</span>
          </label>
          <label className="flex items-center gap-1.5 text-xs pb-2" title="Speculative decoding with the model's own MTP draft: each step runs K one-token draft forwards and one verify over K + 1 positions, and keeps 1 + α·K tokens per request. Only models with an MTP draft are drawn; Kernel Bench's measured track is off.">
            <input id="g-spec-on" type="checkbox" checked={specOn} onChange={(e) => setSpecOn(e.target.checked)} />
            <span className="font-medium text-gray-600">Speculative decoding (MTP)</span>
          </label>
          {specOn && (
            <>
              <label className="block text-xs" title="Draft tokens per step (K): the draft's forwards, and the verify's width K + 1.">
                <span className="font-medium text-gray-600">Draft tokens K</span>
                <select id="g-spec-k" className="mt-1 block text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white"
                  value={specK} onChange={(e) => setSpecK(Number(e.target.value))}>
                  {axes.draft_tokens.map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </label>
              <label className="block text-xs" title="Acceptance rate α: the mean fraction of the K drafts the verify keeps. It moves only the tokens a step keeps (1 + α·K), not the step's time.">
                <span className="font-medium text-gray-600">Acceptance α</span>
                <select id="g-spec-alpha" className="mt-1 block text-sm border border-gray-300 rounded-lg px-3 py-1.5 bg-white"
                  value={specAlpha} onChange={(e) => setSpecAlpha(Number(e.target.value))}>
                  {axes.acceptance_rates.map((a) => <option key={a} value={a}>{a}</option>)}
                </select>
              </label>
            </>
          )}
          <div className="flex-1" />
          <div id="g-status" className="text-xs text-gray-500">{status}</div>
        </div>
      </section>

      {decodeChart}
      {frontFootnote}
      {frontMissingNote}
      {frontNoDraftNote}
      {frontDraftErrorNote}
    </main>
  );
}
