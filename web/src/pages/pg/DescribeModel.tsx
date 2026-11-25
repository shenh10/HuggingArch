// A model described in words: the agent writes it as the caller's own
// custom model (`/api/spec/describe`), the model editor loads it like any
// other, and the user refines it by further turns or by editing its files.
// The conversation is kept per model in this browser; the model itself is
// the caller's stored record.

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "../../api";
import { Spinner, fmtElapsed } from "../../AgentJob";
import { Markdown } from "../../Markdown";
import { fetchSpecJson, pollSpecJob, type VerdictChecks } from "../../specGen";

type Turn = {
  /** What the user asked: the description on the first turn. */
  ask: string;
  /** The agent's own prose, in order. */
  said: string[];
  /** Tool calls and harness lines. */
  steps: string[];
  state: "running" | "ok" | "err";
  message?: string;
  /** What the turn's verdict judged (`validate`'s `checks`). */
  checks?: VerdictChecks;
  startedAt: number;
  endedAt?: number;
};

type Conversation = { description: string; base?: string; turns: Turn[] };

const KEY = "pg-describe:";
const loadConversation = (modelId: string): Conversation | null => {
  try { return JSON.parse(localStorage.getItem(KEY + modelId) || "null"); } catch { return null; }
};
const saveConversation = (modelId: string, c: Conversation) =>
  localStorage.setItem(KEY + modelId, JSON.stringify(c));

/** The agent's closing message: its last prose block. */
const lastWords = (t: Turn) => t.said[t.said.length - 1] ?? "";

function TurnView({ t, now }: { t: Turn; now: number }) {
  const [open, setOpen] = useState(false);
  const elapsed = (t.endedAt ?? now) - t.startedAt;
  return (
    <div className="space-y-2">
      <div className="ml-6 bg-gray-100 rounded-lg px-3 py-2 text-sm text-gray-800 whitespace-pre-wrap">{t.ask}</div>
      <button type="button" onClick={() => setOpen(!open)}
        className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1.5">
        {t.state === "running" && <Spinner />}
        <span>{open ? "▾" : "▸"} {t.steps.length} steps · {fmtElapsed(elapsed)}</span>
      </button>
      {open && (
        <pre className="text-[11px] leading-4 text-gray-500 bg-gray-50 border border-gray-100 rounded p-2 max-h-56 overflow-auto whitespace-pre-wrap">
          {t.steps.join("\n")}
        </pre>
      )}
      {lastWords(t) && <Markdown text={lastWords(t)} className="text-sm text-gray-800" />}
      {t.state === "ok" && t.checks && <ChecksView checks={t.checks} />}
      {t.state === "err" && <div className="text-xs text-red-600 whitespace-pre-wrap">{t.message}</div>}
    </div>
  );
}

/** What the verdict that passed judged: the checks it ran (a turn ends
 *  well only when every gate among them passed), the ones it had no
 *  checkpoint for, and any that crashed and so judged nothing. */
function ChecksView({ checks }: { checks: VerdictChecks }) {
  const [open, setOpen] = useState(false);
  const crashed = new Set(checks.crashed ?? []);
  return (
    <div className="border border-gray-200 rounded-lg px-3 py-2 text-xs">
      <button type="button" onClick={() => setOpen(!open)} className="font-medium text-gray-800 text-left">
        <span className="text-green-600 mr-1.5">✓</span>Passed validation · {checks.ran.length} checks ran
        {checks.skipped.length > 0 && <span className="text-gray-400 font-normal"> · {checks.skipped.length} skipped (no checkpoint)</span>}
        {crashed.size > 0 && <span className="text-amber-700 font-normal"> · {crashed.size} crashed</span>}
        <span className="text-gray-400 ml-1">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <ul className="mt-1 space-y-0.5">
          {checks.ran.map((c) => crashed.has(c)
            ? <li key={c} className="text-amber-700"><span className="mr-1.5">!</span>{c} — crashed, judged nothing</li>
            : <li key={c}><span className="text-gray-400 mr-1.5">•</span>{c}</li>)}
          {checks.skipped.map((c) => <li key={c} className="text-gray-400"><span className="mr-1.5">–</span>{c} — skipped, no checkpoint</li>)}
        </ul>
      )}
    </div>
  );
}

function FilesTab({ modelId, reload, onSaved }: { modelId: string; reload: number; onSaved: () => void }) {
  const [spec, setSpec] = useState("");
  const [config, setConfig] = useState("");
  const [loaded, setLoaded] = useState({ spec: "", config: "", revision: null as number | null });
  const [errors, setErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const name = modelId.replace(/^custom:/, "");

  useEffect(() => {
    let dead = false;
    void (async () => {
      try {
        const d = await fetchSpecJson<{ spec_yaml: string; config: Record<string, unknown>; revision?: number }>(
          `/api/spec/describe/files?name=${encodeURIComponent(name)}`);
        if (dead) return;
        const c = JSON.stringify(d.config, null, 2);
        setSpec(d.spec_yaml); setConfig(c); setErrors([]); setNote("");
        setLoaded({ spec: d.spec_yaml, config: c, revision: d.revision ?? null });
      } catch (e) { if (!dead) setNote(e instanceof Error ? e.message : String(e)); }
    })();
    return () => { dead = true; };
  }, [name, reload]);

  const dirty = spec !== loaded.spec || config !== loaded.config;
  const save = async () => {
    let parsed: unknown;
    try { parsed = JSON.parse(config); } catch (e) {
      setErrors([`config.json is not valid JSON: ${e instanceof Error ? e.message : String(e)}`]); return;
    }
    setBusy(true); setErrors([]); setNote("Validating…");
    try {
      const r = await fetchSpecJson<{ success: boolean; errors?: string[]; message?: string }>(
        "/api/spec/describe/save", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, spec_yaml: spec, config: parsed, revision: loaded.revision }) });
      if (r.success) {
        setNote("Saved — re-analysing."); onSaved();
      } else {
        setErrors(r.errors?.length ? r.errors : [r.message || "Not saved."]); setNote("Not saved — the stored model stands.");
      }
    } catch (e) {
      setErrors([e instanceof Error ? e.message : String(e)]); setNote("");
    } finally { setBusy(false); }
  };

  const box = "w-full font-mono text-[11px] leading-4 border border-gray-200 rounded-lg p-2 bg-gray-50";
  return (
    <div className="space-y-2 text-sm">
      <label className="block text-xs font-medium text-gray-600">config.json{config !== loaded.config && " ●"}</label>
      <textarea className={`${box} h-40`} value={config} onChange={(e) => setConfig(e.target.value)} spellCheck={false} />
      <label className="block text-xs font-medium text-gray-600">spec.yaml{spec !== loaded.spec && " ●"}</label>
      <textarea className={`${box} h-72`} value={spec} onChange={(e) => setSpec(e.target.value)} spellCheck={false} />
      {errors.length > 0 && (
        <div className="border border-red-200 bg-red-50 rounded-lg p-2 text-xs text-gray-800 space-y-1 max-h-40 overflow-auto">
          <div className="font-medium text-red-700">{errors.length} problem{errors.length > 1 ? "s" : ""}</div>
          {errors.map((e, i) => <div key={i} className="whitespace-pre-wrap">{e}</div>)}
        </div>
      )}
      <div className="flex items-center gap-2">
        <button type="button" disabled={!dirty || busy} onClick={() => void save()}
          className="px-3 py-1.5 bg-gray-800 text-white rounded-lg text-xs disabled:opacity-40">Save &amp; re-analyse</button>
        <button type="button" disabled={!dirty || busy} onClick={() => { setSpec(loaded.spec); setConfig(loaded.config); setErrors([]); setNote(""); }}
          className="px-3 py-1.5 border border-gray-300 rounded-lg text-xs text-gray-700 disabled:opacity-40">Revert</button>
        <span className="text-xs text-gray-500">{note}</span>
      </div>
    </div>
  );
}

/** The agent drawer. `modelId`: the described model on screen, null before
 *  the first turn. `pendingEdits`: the editor's unsaved swaps and dim
 *  changes, as words the next turn carries. */
export function DescribeDrawer({ modelId, firstTurn, pendingEdits, onModel, onClose }: {
  modelId: string | null;
  /** A first turn the source card started: name, description, base. */
  firstTurn: { name: string; description: string; base?: string; seq: number } | null;
  pendingEdits: string;
  onModel: (modelId: string) => void;
  onClose: () => void;
}) {
  const [conv, setConv] = useState<Conversation | null>(null);
  const [tab, setTab] = useState<"chat" | "files">("chat");
  const [message, setMessage] = useState("");
  const [now, setNow] = useState(Date.now());
  const [filesReload, setFilesReload] = useState(0);
  const stopped = useRef(false);
  // the model the drawer shows: a turn on another model keeps writing its
  // own conversation, and stops drawing into this one
  const shown = useRef<string | null>(modelId);
  const running = conv?.turns.some((t) => t.state === "running") ?? false;

  useEffect(() => {
    shown.current = modelId;
    setConv(modelId ? loadConversation(modelId) : null);
  }, [modelId]);
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);

  const runTurn = useCallback(async (id: string, base: Conversation, ask: string, body: Record<string, unknown>) => {
    const turn: Turn = { ask, said: [], steps: [], state: "running", startedAt: Date.now() };
    let c: Conversation = { ...base, turns: [...base.turns, turn] };
    const put = (t: Turn) => {
      c = { ...c, turns: [...c.turns.slice(0, -1), t] };
      saveConversation(id, c);
      if (shown.current === id) setConv(c);
    };
    put(turn);
    let t = turn;
    try {
      const start = await fetchSpecJson<{ status?: string; message?: string }>("/api/spec/describe", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (start.status !== "started") {
        put({ ...t, state: "err", message: start.message || `Not started (${start.status}).`, endedAt: Date.now() });
        return;
      }
      stopped.current = false;
      const result = await pollSpecJob(id, {
        authed: true, kind: "generate", stopped: () => stopped.current,
        onLog: (line, author) => {
          t = author === "agent" ? { ...t, said: [...t.said, line] } : { ...t, steps: [...t.steps, line] };
          put(t);
        },
      });
      put({ ...t, state: result.success ? "ok" : "err", message: result.message,
        checks: result.checks, endedAt: Date.now() });
      if (result.success) { setFilesReload((n) => n + 1); onModel(id); }
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
      put({ ...t, state: "err", message: msg, endedAt: Date.now() });
    }
  }, [onModel]);

  // a first turn the source card asked for
  const startedSeq = useRef(0);
  useEffect(() => {
    if (!firstTurn || firstTurn.seq === startedSeq.current) return;
    startedSeq.current = firstTurn.seq;
    const id = `custom:${firstTurn.name}`;
    const c: Conversation = { description: firstTurn.description, base: firstTurn.base, turns: [] };
    shown.current = id; setConv(c); setTab("chat");
    void runTurn(id, c, firstTurn.description, {
      name: firstTurn.name, description: firstTurn.description,
      base_model_id: firstTurn.base || null });
  }, [firstTurn, runTurn]);

  const send = () => {
    if (!modelId || !message.trim()) return;
    const ask = pendingEdits ? `${message.trim()}\n\nAlso apply my edits in the editor: ${pendingEdits}` : message.trim();
    // the description and base are the stored model's (the server reads
    // them from its record); this browser keeps only the conversation
    const base = loadConversation(modelId) ?? { description: "", turns: [] };
    setMessage("");
    void runTurn(modelId, base, ask, { name: modelId.replace(/^custom:/, ""), message: ask });
  };

  return (
    <aside id="pg-describe-drawer" className="card flex flex-col min-h-[32rem] max-h-[calc(100vh-6rem)] sticky top-4">
      <div className="px-4 py-3 border-b border-gray-100 flex items-center gap-3 text-sm">
        <span className="font-semibold text-gray-900">Agent</span>
        {(["chat", "files"] as const).map((k) => (
          <button key={k} type="button" disabled={k === "files" && !modelId} onClick={() => setTab(k)}
            className={`text-xs ${tab === k ? "text-gray-900 font-medium" : "text-gray-400 hover:text-gray-700"} disabled:opacity-40`}>
            {k === "chat" ? "Conversation" : "Files"}
          </button>
        ))}
        {running && <button type="button" onClick={() => { stopped.current = true; }} className="ml-auto text-xs text-red-600">Stop watching</button>}
        <button type="button" onClick={onClose} title="Close" className={`${running ? "" : "ml-auto"} text-gray-400 hover:text-gray-700`}>×</button>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {tab === "files" && modelId
          ? <FilesTab modelId={modelId} reload={filesReload} onSaved={() => { setFilesReload((n) => n + 1); onModel(modelId); }} />
          : conv?.turns.length
            ? conv.turns.map((t, i) => <TurnView key={i} t={t} now={now} />)
            : <p className="text-sm text-gray-400">Describe a model in the box on the left to start.</p>}
      </div>
      {tab === "chat" && modelId && (
        <div className="p-3 border-t border-gray-100 space-y-2">
          {pendingEdits && (
            <div className="text-xs text-blue-700 bg-blue-50 border border-blue-200 rounded px-2 py-1">
              Your unsaved editor changes ({pendingEdits}) will be sent with the message.
            </div>
          )}
          <div className="flex gap-2">
            <input value={message} onChange={(e) => setMessage(e.target.value)} disabled={running}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask for a change…" className="flex-1 text-sm border border-gray-300 rounded-lg px-3 py-2" />
            <button type="button" onClick={send} disabled={running || !message.trim()}
              className="px-3 py-2 bg-gray-800 text-white rounded-lg text-sm disabled:opacity-40">Send</button>
          </div>
        </div>
      )}
    </aside>
  );
}

/** The source card's "Describe in words" form. */
export function DescribeForm({ models, signedIn, busy, onStart }: {
  models: string[];
  signedIn: boolean;
  busy: boolean;
  onStart: (name: string, description: string, base?: string) => void;
}) {
  const [description, setDescription] = useState("");
  const [name, setName] = useState("");
  const [base, setBase] = useState("");
  const examples = [
    "Llama-3-8B but with MLA attention (kv_lora 512)",
    "Qwen3-30B-A3B with 256 experts, top-8",
    "a 7B hybrid: 3 linear-attention layers to 1 GQA layer",
  ];
  const nameOk = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name);
  if (!signedIn) {
    return <p className="text-sm text-gray-600">Sign in to describe a model — the agent runs on your own API key (account settings).</p>;
  }
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_16rem] gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">Describe the model</label>
        <textarea id="pg-describe-text" value={description} onChange={(e) => setDescription(e.target.value)}
          className="w-full h-24 text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white"
          placeholder="e.g. a 48-layer MoE, hidden 4096, MLA attention with kv_lora 512, 128 experts top-8, every 4th layer full attention and the rest sliding window 4096, FP8 weights" />
        <div className="flex flex-wrap gap-2 mt-2 text-xs">
          <span className="text-gray-400">Try:</span>
          {examples.map((x) => (
            <button key={x} type="button" onClick={() => setDescription(x)}
              className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 hover:bg-gray-200">{x}</button>
          ))}
        </div>
      </div>
      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
          <div className="flex items-center border border-gray-300 rounded-lg bg-white text-sm">
            <span className="pl-3 text-gray-400 font-mono">custom:</span>
            <input id="pg-describe-name" value={name} onChange={(e) => setName(e.target.value)}
              className="flex-1 min-w-0 px-1 py-2 font-mono outline-none rounded-r-lg" placeholder="my-model" />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Start from <span className="text-gray-400 font-normal">(optional)</span></label>
          <input list="pg-describe-models" value={base} onChange={(e) => setBase(e.target.value)} placeholder="none — compose from the library"
            className="w-full text-sm border border-gray-300 rounded-lg px-3 py-2 bg-white" />
          <datalist id="pg-describe-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
        </div>
        <button id="pg-describe-start" type="button" disabled={busy || !nameOk || !description.trim()}
          onClick={() => onStart(name.trim(), description.trim(), base.trim() || undefined)}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-800 text-white rounded-lg text-sm font-medium disabled:opacity-40">
          {busy ? "Generating…" : "Generate"}
        </button>
      </div>
    </div>
  );
}

/** The caller's described models, for the "My models" source. */
export function useMyModels(signedIn: boolean, reload: number): string[] {
  const [ids, setIds] = useState<string[]>([]);
  useEffect(() => {
    if (!signedIn) { setIds([]); return; }
    let dead = false;
    fetchSpecJson<{ models: { model_id: string }[] }>("/api/spec/describe/mine")
      .then((d) => { if (!dead) setIds(d.models.map((m) => m.model_id)); })
      .catch(() => { if (!dead) setIds([]); });
    return () => { dead = true; };
  }, [signedIn, reload]);
  return ids;
}
