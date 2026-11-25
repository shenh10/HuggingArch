/** Inspector spec-generation: poll / retry. Session cookies via auth.ts. */

import { ApiError } from "./api";
import { refreshSession } from "./auth";
import { formatNumber } from "./format";
import { generationBadge, type EngineGap } from "./specStatus";

/** What became of a run's generation record (`RecordOutcome`). */
export type RecordResult = { outcome?: string; id?: string; reason?: string };

/** `resumable`: a failed generation left a draft that `resume: true`
 *  continues. `errorKind` / `engineGaps` / `record`: the run's terminal
 *  machine fields (a blocked run: `blocked_by_engine` and its gaps). */
export type SpecJobResult = {
  success: boolean;
  message: string;
  resumable?: boolean;
  errorKind?: string;
  engineGaps?: EngineGap[];
  record?: RecordResult;
  /** What the verdict judged: the checks it ran, and the ones it had no
   *  checkpoint for (`validate`'s `checks`). */
  checks?: VerdictChecks;
  /** The run could not go on without a signed-in session: the session
   *  lapsed mid-poll, or the deployment serves generation to signed-in
   *  users only (start `status: "disabled"`). */
  signInRequired?: boolean;
};

export type VerdictChecks = { ran: string[]; skipped: string[]; crashed?: string[] };

export type SpecStart = {
  /** `started` / `already_running` / `cached`, or why no job started
   *  (`disabled`: this deployment generates for signed-in users only). */
  status?: string;
  success?: boolean;
  message?: string;
  detail?: unknown;
  spec_name?: string;
  /** 409 `draft_kept`: an unfinished generation kept a draft. */
  resumable?: boolean;
};

export type SpecEvent = {
  type?: string;
  text?: string;
  /** `agent` on a log line the agent itself wrote (its prose), absent on
   *  tool calls and harness lines. */
  author?: string;
  success?: boolean;
  message?: string;
  resumable?: boolean;
  error_kind?: string;
  engine_gaps?: EngineGap[];
  record?: RecordResult;
  checks?: VerdictChecks;
  usage?: UsageDelta;
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
};

export type SpecStatus = {
  state?: string;
  events?: SpecEvent[];
  next_cursor?: number;
  success?: boolean;
  message?: string;
  /** Seconds the server's run may hold its lock; the poll budget. */
  poll_budget_sec?: number;
  /** No run is live and the caller's generate workspace keeps a draft. */
  resumable?: boolean;
  agent_cli?: string;
  agent_model?: string;
  /** A finished run read back from its log row (after a restart). */
  error_kind?: string | null;
  engine_gaps?: EngineGap[] | null;
};

export type UsageAcc = {
  input: number;
  output: number;
  cacheCreate: number;
  cacheRead: number;
  turns: number;
};

export type UsageDelta = {
  input_tokens?: number;
  output_tokens?: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
};

export type GenerationRow = {
  id?: string;
  model_id?: string;
  status?: string;
  pr_url?: string | null;
};

export function specUrls(authed: boolean) {
  return {
    analyze: authed ? "/api/spec/analyze" : "/analyze",
    generate: authed ? "/api/spec/generate" : "/generate-spec",
    refine: authed ? "/api/spec/refine" : "/refine-spec",
    status: authed
      ? "/api/spec/status"
      : "/spec-status",
    stop: authed ? "/api/spec/stop" : "/stop-spec",
  };
}

/** The fields of an analyze answer that decide what the Inspector does
 *  about a spec it could not show. */
export type AnalyzeVerdict = {
  success?: boolean;
  /** No spec matches the model: no library spec and no record of the caller's. */
  no_spec?: boolean;
  served_from?: string;
  dashboard?: { total_params?: number } | null;
  block_graph?: unknown;
  /** Agent Fix can open the saved record on these refusals. */
  fixable?: boolean;
  record_id?: string;
};

/** Whether the Inspector starts a spec generation by itself: only when the
 *  model has no spec at all. A spec the caller already has
 *  (`served_from: user_generation`) is never regenerated automatically,
 *  whatever its analysis did — an engine gap, a spec error or a crash is
 *  shown, and a new run is the user's click. */
export function shouldAutoGenerate(body: AnalyzeVerdict): boolean {
  if (body.served_from === "user_generation") return false;
  if (!body.success) return body.no_spec === true;
  const graph = body.block_graph;
  const emptyGraph = !graph || (typeof graph === "object" && !Object.keys(graph as object).length);
  return (body.dashboard?.total_params || 0) === 0 || emptyGraph;
}

/** Whether the Inspector starts Agent Fix by itself: the caller's saved
 *  spec does not load, and every refusal names a declaration Fix can write.
 *  An engine gap, a retired syntax, or a record without an id is shown. */
export function shouldAutoFix(body: AnalyzeVerdict): boolean {
  return body.served_from === "user_generation"
    && body.success === false
    && body.fixable === true
    && !!body.record_id;
}

export function emptyUsage(): UsageAcc {
  return { input: 0, output: 0, cacheCreate: 0, cacheRead: 0, turns: 0 };
}

export function addUsage(acc: UsageAcc, u: UsageDelta | undefined): UsageAcc {
  if (!u) return acc;
  return {
    input: acc.input + (u.input_tokens || 0),
    output: acc.output + (u.output_tokens || 0),
    cacheCreate: acc.cacheCreate + (u.cache_creation_input_tokens || 0),
    cacheRead: acc.cacheRead + (u.cache_read_input_tokens || 0),
    turns: acc.turns + 1,
  };
}

export function formatUsage(acc: UsageAcc): string {
  const grand = acc.input + acc.output + acc.cacheCreate + acc.cacheRead;
  if (!grand && !acc.turns) return "";
  const turn = acc.turns > 0 ? ` (${acc.turns} turn${acc.turns === 1 ? "" : "s"})` : "";
  return `Token usage${turn}: new input ${formatNumber(acc.input)} · output ${formatNumber(acc.output)} · cache create ${formatNumber(acc.cacheCreate)} · cache read ${formatNumber(acc.cacheRead)} · total ${formatNumber(grand)}`;
}

export function jobMessage(data: { message?: string; detail?: unknown } | null | undefined, fallback: string): string {
  if (!data) return fallback;
  if (typeof data.message === "string" && data.message) return data.message;
  if (typeof data.detail === "string" && data.detail) return data.detail;
  return fallback;
}

/** The poll budget until the server states its own (`poll_budget_sec`, the
 *  spec lock TTL, raised by the server's knobs): the default TTL, three
 *  60-min agent rounds (the first + 2 repairs) + 10 min validate + 15 min
 *  grace. */
export const SPEC_POLL_BUDGET_MS = (3 * 60 + 10 + 15) * 60 * 1000;
export const SPEC_POLL_INTERVAL_MS = 2000;
export const SPEC_POLL_MAX_INTERVAL_MS = 30_000;
export const SPEC_AUTH_EXPIRED_MESSAGE =
  "Sign-in expired. The job is still running in the background; sign in again and reopen this page to reconnect.";
export const SPEC_INTERRUPTED_MESSAGE =
  "Interrupted by a server restart. You can retry.";

/** JSON fetch for spec-gen. GET retries 502–504 once. HTTP ≥400 throws. */
export async function fetchSpecJson<T>(url: string, init?: RequestInit): Promise<T> {
  const method = (init?.method || "GET").toUpperCase();
  let response: Response;
  try {
    response = await fetch(url, init);
    if ((method === "GET" || method === "HEAD") && response.status >= 502 && response.status <= 504) {
      await new Promise((r) => setTimeout(r, 800));
      response = await fetch(url, init);
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(`Cannot reach the backend (${msg}).`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    let body: Record<string, unknown> = {};
    try {
      const parsed: unknown = await response.json();
      if (parsed && typeof parsed === "object") body = parsed as Record<string, unknown>;
      if (typeof body.message === "string" && body.message) detail = body.message;
      else if (typeof body.detail === "string") detail = body.detail;
    } catch { /* keep status */ }
    throw new ApiError(detail, response.status, body);
  }
  if (contentType && !contentType.includes("application/json")) {
    throw new Error(`Backend unreachable — HTTP ${response.status} instead of JSON.`);
  }
  return response.json() as Promise<T>;
}

/** Consecutive failed status polls before polling gives up. */
const MAX_POLL_FAILURES = 30;

const LIVE_STATES = new Set(["running", "done", "idle", "interrupted"]);

export type SpecJobKind = "generate" | "refine" | "fix";

/** Whether Inspector should offer the Refine button. Analyze never POSTs refine. */
export function shouldOfferRefine(dash: {
  hf_model_storage_bytes?: number;
  storage_diff?: number;
} | null | undefined): boolean {
  return !!dash && (dash.hf_model_storage_bytes || 0) > 0 && Math.abs(dash.storage_diff || 0) > 0;
}

export async function pollSpecJob(
  modelId: string,
  opts: {
    /** Signed-in session → the authed endpoint set; cookies ride along. */
    authed?: boolean;
    kind?: SpecJobKind;
    /** Required when `kind` is `fix`: jobs are keyed by record, not model. */
    recordId?: string;
    onLog?: (line: string, author?: string) => void;
    onUsage?: (acc: UsageAcc) => void;
    onAgent?: (cli?: string, model?: string) => void;
    stopped?: () => boolean;
    now?: () => number;
    sleep?: (ms: number) => Promise<void>;
    budgetMs?: number;
  },
): Promise<SpecJobResult> {
  const sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
  const now = opts.now ?? Date.now;
  const startedAt = now();
  let deadline = startedAt + (opts.budgetMs ?? SPEC_POLL_BUDGET_MS);
  const kind = opts.kind ?? "generate";
  const urls = specUrls(!!opts.authed);
  let cursor = 0;
  let success = false;
  let message = "";
  let resumable = false;
  let terminal: Pick<SpecJobResult, "errorKind" | "engineGaps" | "record" | "checks"> = {};
  let usage = emptyUsage();
  let failures = 0;
  let interval = SPEC_POLL_INTERVAL_MS;

  while (true) {
    if (opts.stopped?.()) return { success: false, message: "Stopped by user" };
    if (now() >= deadline) {
      return { success: false, message: "Timed out waiting for the job. You can retry." };
    }
    const statusUrl = kind === "fix"
      ? `/api/spec/fix-status?record_id=${encodeURIComponent(opts.recordId || "")}&cursor=${cursor}`
      : `${urls.status}?model_id=${encodeURIComponent(modelId)}&kind=${encodeURIComponent(kind)}&cursor=${cursor}`;
    let data: SpecStatus;
    try {
      data = await fetchSpecJson<SpecStatus>(statusUrl);
      failures = 0;
      interval = SPEC_POLL_INTERVAL_MS;
    } catch (e) {
      if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
        if (!(await refreshSession())) {
          return { success: false, message: SPEC_AUTH_EXPIRED_MESSAGE, signInRequired: true };
        }
        try {
          data = await fetchSpecJson<SpecStatus>(statusUrl);
          failures = 0;
          interval = SPEC_POLL_INTERVAL_MS;
        } catch (e2) {
          if (e2 instanceof ApiError && (e2.status === 401 || e2.status === 403)) {
            return { success: false, message: SPEC_AUTH_EXPIRED_MESSAGE, signInRequired: true };
          }
          failures += 1;
          if (failures >= MAX_POLL_FAILURES) {
            const why = e2 instanceof Error ? e2.message : String(e2);
            return { success: false, message: `Lost contact with the backend while polling the job: ${why}` };
          }
          interval = Math.min(interval * 2, SPEC_POLL_MAX_INTERVAL_MS);
          await sleep(interval);
          continue;
        }
      } else {
        failures += 1;
        if (failures >= MAX_POLL_FAILURES) {
          const why = e instanceof Error ? e.message : String(e);
          return { success: false, message: `Lost contact with the backend while polling the job: ${why}` };
        }
        interval = Math.min(interval * 2, SPEC_POLL_MAX_INTERVAL_MS);
        await sleep(interval);
        continue;
      }
    }
    if (data.agent_cli || data.agent_model) {
      opts.onAgent?.(data.agent_cli, data.agent_model);
    }
    if (opts.budgetMs === undefined && typeof data.poll_budget_sec === "number" && data.poll_budget_sec > 0) {
      deadline = startedAt + data.poll_budget_sec * 1000;
    }
    if (data.state === "idle") {
      return {
        success: false,
        message: "Job not found (backend may have restarted). You can retry.",
        resumable: !!data.resumable,
      };
    }
    if (data.state === "interrupted") {
      return {
        success: false,
        message: data.message || SPEC_INTERRUPTED_MESSAGE,
        resumable: !!data.resumable,
      };
    }
    if (data.state && !LIVE_STATES.has(data.state)) {
      return { success: false, message: `Unexpected job state: ${data.state}` };
    }
    if (!data.state) {
      return { success: false, message: "Unexpected job state: missing" };
    }
    for (const evt of data.events || []) {
      if (evt.type === "log") opts.onLog?.(evt.text || "", evt.author);
      else if (evt.type === "usage") {
        usage = addUsage(usage, evt);
        opts.onUsage?.(usage);
      } else if (evt.type === "done") {
        success = !!evt.success;
        message = evt.message || "";
        resumable = !!evt.resumable;
        terminal = {
          errorKind: evt.error_kind || undefined,
          engineGaps: evt.engine_gaps || undefined,
          record: evt.record || undefined,
          checks: evt.checks || undefined,
        };
        if (evt.usage) {
          usage = addUsage(usage, evt.usage);
          opts.onUsage?.(usage);
        }
      }
    }
    if (typeof data.next_cursor === "number") cursor = data.next_cursor;
    if (data.state === "done") {
      if (data.success != null) success = !!data.success;
      if (data.message) message = data.message;
      if (!terminal.errorKind && data.error_kind) {
        terminal = { ...terminal, errorKind: data.error_kind, engineGaps: data.engine_gaps || undefined };
      }
      return { success, message, resumable: !success && (resumable || !!data.resumable), ...terminal };
    }
    await sleep(interval);
  }
}

export async function stopSpecJob(
  modelId: string,
  authed: boolean,
  kind: SpecJobKind = "generate",
  recordId?: string,
): Promise<void> {
  const urls = specUrls(authed);
  const send = () => fetch(urls.stop, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(kind === "fix"
      ? { kind, record_id: recordId }
      : { model_id: modelId, kind }),
  });
  let r = await send();
  if ((r.status === 401 || r.status === 403) && await refreshSession()) r = await send();
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try {
      detail = jobMessage(await r.json() as { message?: string; detail?: unknown }, detail);
    } catch { /* keep the status */ }
    throw new Error(detail);
  }
}

export async function latestGeneration(modelId: string): Promise<GenerationRow | null> {
  try {
    const data = await fetchSpecJson<{ items?: GenerationRow[] }>("/api/account/generations");
    const items = (data.items || []).filter((g) => g.model_id === modelId && g.id);
    return items[0] || null;
  } catch {
    return null;
  }
}


/** Display label for a spec_generations status. The stored `contributed` means
 *  the author opened the PR; a `contributed` row with no PR is a leftover of the
 *  retired Contribute button. `prUrl` undefined = the payload doesn't carry it. */
export function specStatusLabel(status: unknown, prUrl?: unknown): string {
  if (status !== "contributed") return generationBadge(status).label;
  return prUrl === undefined || prUrl ? "PR open" : "no PR";
}
