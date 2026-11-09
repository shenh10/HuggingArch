import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiError, fetchJson } from "../api";
import { agentLabel, useAgentBackends, type AgentBackends } from "../agents";
import { loadAuth, refreshSession, signOut, type AuthState } from "../auth";
import { ComboBox } from "../ComboBox";
import { AccountNav, ACCOUNT_PANELS, WORKSPACE_PANELS } from "../AccountNav";
import { alertError, confirmDialog } from "../Dialog";

type Cred = {
  agent_cli: string;
  name?: string | null;
  is_default?: boolean;
  model?: string | null;
  base_url?: string | null;
  task_models?: Record<string, string>;
};

// An agent run a credential can pin its own model for (backend AgentTask).
type AgentTask = { key: string; label: string };

type TokenRow = {
  id: string;
  name?: string;
  token_prefix?: string;
  last_used_at?: string | null;
  revoked_at?: string | null;
};

type ProfilePanel = (typeof ACCOUNT_PANELS)[number];

function credName(c: { name?: string | null }) {
  return (c.name || "").trim();
}

/** Routing / React key: `agent_cli` or `agent_cli:name`. */
function credId(c: { agent_cli: string; name?: string | null }) {
  const n = credName(c);
  return n ? `${c.agent_cli}:${n}` : c.agent_cli;
}

/** Card title: the stored name, or the agent label when name is blank. */
function credLabel(c: { agent_cli: string; name?: string | null }, agents: AgentBackends | null) {
  return credName(c) || agentLabel(agents, c.agent_cli);
}

/** "A", "A or B", "A, B, or C". */
function orList(names: string[]): string {
  if (names.length <= 2) return names.join(" or ");
  return `${names.slice(0, -1).join(", ")}, or ${names[names.length - 1]}`;
}

function credNameQuery(name?: string | null) {
  const n = (name || "").trim();
  return n ? `?name=${encodeURIComponent(n)}` : "";
}

const INPUT = "mt-1 w-full px-3 py-2 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:border-amber-500";
const CARD = "bg-white border border-gray-200 rounded-2xl p-6 sm:p-7 shadow-sm";
const BTN = "px-4 py-2 text-sm font-medium rounded-lg bg-amber-500 text-white hover:bg-amber-600 disabled:opacity-60";

function panelFromHash(): ProfilePanel {
  const h = (window.location.hash || "").replace(/^#/, "");
  return (ACCOUNT_PANELS as readonly string[]).includes(h) ? (h as ProfilePanel) : "profile";
}

function loginRedirect(): void {
  const next = encodeURIComponent(window.location.pathname + window.location.search + window.location.hash);
  window.location.href = `/auth/login.html?next=${next}`;
}

async function accountJson<T>(method: string, path: string, body?: unknown): Promise<T> {
  const send = () => fetchJson<T>(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  try {
    return await send();
  } catch (e) {
    if (e instanceof ApiError && (e.status === 401 || e.status === 403)) {
      if (await refreshSession()) return await send();
      loginRedirect();
      throw new Error("Not signed in");
    }
    throw e;
  }
}

function AvatarFace({ url, text, size }: { url: string; text: string; size: string }) {
  const [failed, setFailed] = useState(false);
  const initial = (text || "?").trim().slice(0, 1).toUpperCase() || "?";
  if (!url || failed) {
    return (
      <span className={`${size} rounded-full bg-amber-100 text-amber-700 inline-flex items-center justify-center font-semibold`}>
        {initial}
      </span>
    );
  }
  return (
    <img
      src={url}
      alt=""
      referrerPolicy="no-referrer"
      className={`${size} rounded-full object-cover`}
      onError={() => setFailed(true)}
    />
  );
}

const AVATAR_MAX_DIM = 512;

async function resizeImageBlob(file: File): Promise<Blob> {
  if (file.type === "image/gif") return file;
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result));
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
  const img = await new Promise<HTMLImageElement>((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = dataUrl;
  });
  const longest = Math.max(img.width, img.height);
  if (longest <= AVATAR_MAX_DIM) return file;
  const scale = AVATAR_MAX_DIM / longest;
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d")?.drawImage(img, 0, 0, w, h);
  const mime = file.type === "image/png" ? "image/png" : "image/jpeg";
  const quality = mime === "image/jpeg" ? 0.9 : undefined;
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, mime, quality));
  return blob || file;
}

function taskModelSummary(cred: Cred, tasks: AgentTask[], siteDefaults: Record<string, string>) {
  const chosen = cred.task_models || {};
  const setCount = tasks.filter((t) => chosen[t.key]).length;
  const effectiveDefaults = tasks.map(
    (t) => siteDefaults[t.key] || cred.model || "endpoint default",
  );
  const uniqueDefaults = [...new Set(effectiveDefaults)];
  const collapsedDefault = uniqueDefaults.length === 1
    ? uniqueDefaults[0]
    : "varies by task";
  return { collapsedDefault, setCount };
}

/** Collapsed one-line view of per-task models. Edit expands the grid. */
function TaskModelSummary({ cred, tasks, siteDefaults }: {
  cred: Cred; tasks: AgentTask[]; siteDefaults: Record<string, string>;
}) {
  const { collapsedDefault, setCount } = taskModelSummary(cred, tasks, siteDefaults);
  return (
    <div className="mt-2 border-t border-gray-100 pt-2">
      <div className="rounded-lg border border-gray-200 bg-gray-50/60 px-2.5 py-2">
        <div className="text-10 font-semibold uppercase tracking-wide text-gray-400">
          Task models
        </div>
        <div className="min-w-0 truncate text-xs font-medium text-gray-800" title={collapsedDefault}>
          {collapsedDefault}
          <span className="ml-1.5 font-normal text-gray-400">
            {setCount ? `· ${setCount} pinned` : "· all tasks use this default"}
          </span>
        </div>
      </div>
    </div>
  );
}

function modelOptions(models: string[], extra?: string | null) {
  const extras = extra && !models.includes(extra) ? [extra] : [];
  return [
    ...extras.map((m) => ({ value: m, label: m })),
    ...models.map((m) => ({ value: m, label: m })),
  ];
}

function TaskModelGrid({
  tasks, chosen, models, siteDefaults, credModel, onChange, onDefaultChange,
}: {
  tasks: AgentTask[];
  chosen: Record<string, string>;
  models: string[];
  siteDefaults: Record<string, string>;
  credModel?: string | null;
  onChange: (task: string, model: string) => void;
  onDefaultChange?: (model: string) => void;
}) {
  return (
    <div className="mt-2 grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1.5 max-w-md">
      {onDefaultChange ? (
        <label className="contents text-xs text-gray-600">
          <span>Default</span>
          <ComboBox
            className="w-full"
            allowCustom
            value={credModel || ""}
            placeholder="endpoint default"
            options={[
              { value: "", label: "endpoint default" },
              ...modelOptions(models, credModel),
            ]}
            onChange={onDefaultChange}
          />
        </label>
      ) : null}
      {tasks.map((t) => {
        const value = chosen[t.key] || "";
        const fallback = siteDefaults[t.key] || credModel || "endpoint default";
        return (
          <label key={t.key} className="contents text-xs text-gray-600">
            <span>{t.label}</span>
            <ComboBox
              className="w-full"
              allowCustom
              value={value}
              placeholder={`default — ${fallback}`}
              options={[
                { value: "", label: `default — ${fallback}` },
                ...modelOptions(models, value),
              ]}
              onChange={(v) => onChange(t.key, v)}
            />
          </label>
        );
      })}
    </div>
  );
}

/** Add-key form, or the in-card editor that Edit expands in place
 *  (endpoint, key, default model, and per-task models together). */
function AgentCredForm({
  mode, cred, agents, onSaved, onCancel, tasks, siteDefaults, namesEnabled,
}: {
  mode: "add" | "edit";
  /** The agents the server registers; the add form opens once it is read. */
  agents: AgentBackends | null;
  cred?: Cred;
  onSaved: () => void;
  onCancel?: () => void;
  tasks?: AgentTask[];
  siteDefaults?: Record<string, string>;
  namesEnabled?: boolean;
}) {
  const [agent, setAgent] = useState(cred?.agent_cli || agents?.default || "");
  const [credNameValue, setCredNameValue] = useState(credName(cred || {}));
  const [base, setBase] = useState(cred?.base_url || "");
  const [key, setKey] = useState("");
  const [model, setModel] = useState(cred?.model || "");
  const [isDefault, setIsDefault] = useState(!!cred?.is_default);
  const [taskModels, setTaskModels] = useState<Record<string, string>>(
    () => ({ ...(cred?.task_models || {}) }),
  );
  const [status, setStatus] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [modelStatus, setModelStatus] = useState("");
  const modelLoadSeq = useRef(0);

  const loadModels = useCallback(async () => {
    const seq = ++modelLoadSeq.current;
    setModelStatus("↻ loading…");
    try {
      const r = await accountJson<{ models?: string[] }>("POST", "/api/account/agent-credentials/models", {
        agent_cli: agent,
        base_url: base.trim() || null,
        api_key: key.trim() || null,
        name: credNameValue || null,
      });
      if (seq !== modelLoadSeq.current) return;
      const list = r.models || [];
      setModels(list);
      setModelStatus(list.length ? `${list.length} models from endpoint` : "no catalog — type a model id");
    } catch (e) {
      if (seq !== modelLoadSeq.current) return;
      const msg = e instanceof Error ? e.message : String(e);
      if (e instanceof ApiError && e.reason === "key_required") {
        setModels([]);
        setModelStatus("enter a key to list models");
      } else {
        setModelStatus(msg.replace(/^Could not load models:\s*/i, "").slice(0, 48));
      }
    }
  }, [agent, base, key, credNameValue]);

  useEffect(() => {
    const t = window.setTimeout(() => void loadModels(), 300);
    return () => clearTimeout(t);
  }, [loadModels]);

  // A custom endpoint serves its own models; the agent CLI's built-in default
  // is the vendor's, so the server rejects an endpoint without a model.
  const needsModel = !!base.trim() && !model.trim();

  const save = async () => {
    if (needsModel) {
      setStatus("Pick a default model: a custom endpoint does not serve the agent's built-in default.");
      return;
    }
    setStatus("Saving…");
    try {
      await accountJson("PUT", "/api/account/agent-credentials", {
        agent_cli: agent,
        name: credNameValue,
        api_key: key.trim(),
        base_url: base.trim() || null,
        model: model.trim() || null,
        is_default: isDefault,
        ...(tasks && tasks.length > 0 ? { task_models: taskModels } : {}),
      });
      setKey("");
      if (mode === "add") {
        setModel("");
        setBase("");
        setCredNameValue("");
        setIsDefault(false);
      }
      setStatus("Saved.");
      onSaved();
    } catch (e) {
      setStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  return (
    <div className={mode === "edit" ? "mt-3 border-t border-gray-100 pt-3" : "mt-3"}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {mode === "add" && (
          <label className="block text-xs font-medium text-gray-600">
            Agent
            <select className={INPUT} value={agent} onChange={(e) => setAgent(e.target.value)}>
              {(agents?.agents || []).map((a) => (
                <option key={a.cli} value={a.cli}>{a.label}</option>
              ))}
            </select>
          </label>
        )}
        <label className="block text-xs font-medium text-gray-600">
          Name (optional)
          <input
            className={INPUT}
            placeholder={agentLabel(agents, agent)}
            value={credNameValue}
            disabled={mode === "edit" || !namesEnabled}
            onChange={(e) => setCredNameValue(e.target.value)}
          />
          <p className="text-xs text-gray-400 mt-1">
            {mode === "edit"
              ? "The name identifies this key; leave it as-is."
              : namesEnabled
                ? "Blank uses the agent name. A name lets you keep several keys for the same agent."
                : "Blank uses the agent name. Named configs need the name column (migration 0000)."}
          </p>
        </label>
        <label className="block text-xs font-medium text-gray-600 sm:col-span-2">
          Endpoint (optional)
          <input className={INPUT} placeholder="https://api.anthropic.com" value={base} onChange={(e) => setBase(e.target.value)} onBlur={() => void loadModels()} />
        </label>
      </div>
      <label className="block text-xs font-medium text-gray-600 mt-3">
        API key
        <input type="password" autoComplete="off" className={INPUT} placeholder={mode === "edit" ? "leave blank to keep the current key" : "sk-…"} value={key} onChange={(e) => setKey(e.target.value)} onBlur={() => void loadModels()} />
        {mode === "edit" ? (
          <p className="text-xs text-gray-400 mt-1">Leave blank to keep the current key, or paste a new one to replace it.</p>
        ) : null}
      </label>
      <div className="mt-3">
        <label className="flex items-center gap-1.5 text-xs font-medium text-gray-600 mb-1">
          Models
          <span className="ml-auto font-normal text-11 text-gray-400">{modelStatus}</span>
        </label>
        <TaskModelGrid
          tasks={tasks || []}
          chosen={taskModels}
          models={models}
          siteDefaults={siteDefaults || {}}
          credModel={model}
          onDefaultChange={setModel}
          onChange={(task, value) => setTaskModels((prev) => {
            const next = { ...prev };
            if (value) next[task] = value;
            else delete next[task];
            return next;
          })}
        />
        <p className={`text-xs mt-1 ${needsModel ? "text-red-600" : "text-gray-400"}`}>
          {needsModel
            ? "A custom endpoint needs a default model — the agent's built-in default is the vendor's own model."
            : "Blank default uses the agent's built-in model (vendor endpoint only). A task left on default follows the row above, or the site default."}
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
        <label className="inline-flex items-center gap-2 text-sm text-gray-700">
          <input type="checkbox" className="rounded border-gray-300" checked={isDefault} onChange={(e) => setIsDefault(e.target.checked)} />
          Use as my default agent
        </label>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 min-w-0">{status}</span>
          {onCancel && (
            <button type="button" className="px-3 py-2 text-sm border border-gray-300 rounded-lg" onClick={onCancel}>Cancel</button>
          )}
          <button type="button" className={BTN} disabled={needsModel} onClick={() => void save()}>{mode === "edit" ? "Update" : "Save key"}</button>
        </div>
      </div>
    </div>
  );
}

export function AccountProfile() {
  const nav = useNavigate();
  const [auth, setAuth] = useState<AuthState | null>(null);
  const [panel, setPanel] = useState<ProfilePanel>(panelFromHash);
  const [username, setUsername] = useState("");
  const [fullName, setFullName] = useState("");
  const [bio, setBio] = useState("");
  const [github, setGithub] = useState("");
  const [saveStatus, setSaveStatus] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState(false);
  const [avatarUrl, setAvatarUrl] = useState("");
  const [avatarHint, setAvatarHint] = useState("");
  const [avatarStatus, setAvatarStatus] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const [creds, setCreds] = useState<Cred[]>([]);
  const [agentTasks, setAgentTasks] = useState<AgentTask[]>([]);
  const [taskModelsEnabled, setTaskModelsEnabled] = useState(false);
  const [taskDefaults, setTaskDefaults] = useState<Record<string, Record<string, string>>>({});
  const [taskAgents, setTaskAgents] = useState<Record<string, string>>({});
  const [routingStatus, setRoutingStatus] = useState("");
  const [credTest, setCredTest] = useState<Record<string, { ok: boolean; text: string } | "testing">>({});

  const testCred = async (c: Cred) => {
    const id = credId(c);
    setCredTest((m) => ({ ...m, [id]: "testing" }));
    try {
      const r = await accountJson<{ ok: boolean; models?: number; latency_ms?: number; detail?: string; generation?: { ok: boolean; model?: string | null; detail?: string } }>(
        "POST", `/api/account/agent-credentials/${encodeURIComponent(c.agent_cli)}/test${credNameQuery(c.name)}`);
      const gen = r.generation;
      const genText = gen
        ? gen.ok
          ? `generation ✓ ${gen.model || ""}`.trim()
          : `generation ✗ ${gen.detail || "failed"}`
        : "";
      setCredTest((m) => ({
        ...m,
        [id]: {
          ok: r.ok && (!gen || gen.ok),
          text: r.ok
            ? `Connected · ${r.models} models · ${((r.latency_ms || 0) / 1000).toFixed(1)}s${genText ? ` · ${genText}` : ""}`
            : r.detail || "connection failed",
        },
      }));
    } catch (e) {
      setCredTest((m) => ({ ...m, [id]: { ok: false, text: e instanceof Error ? e.message : String(e) } }));
    }
  };
  const [byokEnabled, setByokEnabled] = useState(true);
  const [credErr, setCredErr] = useState("");
  const [credsLoaded, setCredsLoaded] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const { agents, error: agentsErr } = useAgentBackends();
  const [namesEnabled, setNamesEnabled] = useState(false);

  const [tokens, setTokens] = useState<TokenRow[]>([]);
  const [tokErr, setTokErr] = useState("");
  const [tokLoaded, setTokLoaded] = useState(false);
  const [tokName, setTokName] = useState("");
  const [tokNew, setTokNew] = useState("");
  const [tokCopied, setTokCopied] = useState(false);

  useEffect(() => {
    document.title = "Profile · HuggingArch";
    return () => {
      document.title = "HuggingArch";
    };
  }, []);

  useEffect(() => {
    void loadAuth().then((a) => {
      if (!a.authEnabled) {
        nav("/", { replace: true });
        return;
      }
      if (!a.user) {
        loginRedirect();
        return;
      }
      setAuth(a);
      setUsername(a.profile?.username || "");
      setFullName(a.profile?.full_name || "");
      setBio(a.profile?.bio || "");
      setGithub(a.profile?.github_handle || "");
      const url = a.profile?.avatar_url || a.user.user_metadata?.avatar_url || "";
      setAvatarUrl(url);
      setAvatarHint(url ? "Click avatar to upload a new image." : "No avatar — click to upload one.");
    });
  }, [nav]);

  useEffect(() => {
    const onHash = () => setPanel(panelFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const want = `#${panel}`;
    if (window.location.hash !== want) history.replaceState(null, "", want);
  }, [panel]);

  // Deep links to the workspace page's panels land here when the hash
  // predates the split (the hash never reaches the server) — forward them.
  useEffect(() => {
    const h = (window.location.hash || "").replace(/^#/, "");
    if ((WORKSPACE_PANELS as readonly string[]).includes(h)) {
      nav(`/account/workspace.html#${h}`, { replace: true });
    }
  }, [nav]);

  const loadCreds = useCallback(async () => {
    try {
      const data = await accountJson<{
        items?: Cred[]; byok_enabled?: boolean; agent_tasks?: AgentTask[]; task_models_enabled?: boolean;
        names_enabled?: boolean;
        task_defaults?: Record<string, Record<string, string>>;
        task_agents?: Record<string, string>;
      }>("GET", "/api/account/agent-credentials");
      setByokEnabled(data.byok_enabled !== false);
      setCreds(data.items || []);
      setAgentTasks(data.agent_tasks || []);
      setTaskModelsEnabled(!!data.task_models_enabled);
      setNamesEnabled(!!data.names_enabled);
      setTaskDefaults(data.task_defaults || {});
      setTaskAgents(data.task_agents || {});
      setCredErr("");
    } catch (e) {
      setCredErr(e instanceof Error ? e.message : String(e));
    } finally {
      setCredsLoaded(true);
    }
  }, []);

  const saveRouting = async (task: string, agentCli: string) => {
    setRoutingStatus("Saving…");
    try {
      await accountJson("PUT", "/api/account/agent-task-routing", { task, agent_cli: agentCli || null });
      setRoutingStatus("");
      void loadCreds();
    } catch (e) {
      setRoutingStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  const loadTokens = useCallback(async () => {    try {
      const data = await accountJson<{ tokens?: TokenRow[] }>("GET", "/api/account/tokens");
      setTokens(data.tokens || []);
      setTokErr("");
    } catch (e) {
      setTokErr(e instanceof Error ? e.message : String(e));
    } finally {
      setTokLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!auth?.user) return;
    void loadCreds();
    void loadTokens();
  }, [auth, loadCreds, loadTokens]);

  const email = auth?.user?.email || "";
  const name = auth?.profile?.username || auth?.profile?.full_name || email || "Account";

  const saveProfile = async () => {
    setSaving(true);
    setSaveStatus("");
    setSaveOk(false);
    try {
      await accountJson("PATCH", "/api/account/profile", {
        username: username.trim() || null,
        full_name: fullName.trim() || null,
        bio: bio.trim() || null,
        github_handle: github.trim().replace(/^@/, "") || null,
      });
      setSaveOk(true);
      window.setTimeout(() => setSaveOk(false), 1600);
    } catch (e) {
      setSaveStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
    } finally {
      setSaving(false);
    }
  };

  const onAvatarFile = async (raw: File | undefined) => {
    if (!raw) return;
    const ALLOWED = ["image/png", "image/jpeg", "image/webp", "image/gif"];
    if (!ALLOWED.includes(raw.type)) {
      setAvatarStatus("Unsupported type " + raw.type);
      return;
    }
    if (raw.size > 5 * 1024 * 1024) {
      setAvatarStatus("File too large (max 5 MB before resize)");
      return;
    }
    setAvatarStatus("Resizing…");
    let blob: Blob;
    try {
      blob = await resizeImageBlob(raw);
    } catch {
      setAvatarStatus("Failed to read image");
      return;
    }
    setAvatarStatus("Uploading…");
    try {
      const fd = new FormData();
      fd.append("file", blob, raw.name || "avatar.png");
      const send = () => fetch("/api/account/avatar", {
        method: "POST",
        body: fd,
      });
      let r = await send();
      if (r.status === 401 && await refreshSession()) r = await send();
      if (!r.ok) {
        const detail = ((await r.json().catch(() => ({}))) as { detail?: string }).detail || r.statusText;
        throw new Error(detail);
      }
      const { avatar_url } = (await r.json()) as { avatar_url: string };
      const bust = avatar_url + (avatar_url.includes("?") ? "&" : "?") + "t=" + Date.now();
      setAvatarUrl(bust);
      setAvatarHint("Updated just now.");
      setAvatarStatus("Saved.");
      window.setTimeout(() => setAvatarStatus(""), 2500);
    } catch (e) {
      setAvatarStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  const removeAvatar = async () => {
    if (!(await confirmDialog({ title: "Remove your avatar?", confirmLabel: "Remove", danger: true }))) return;
    setAvatarStatus("Removing…");
    try {
      const send = () => fetch("/api/account/avatar", {
        method: "DELETE",
      });
      let r = await send();
      if (r.status === 401 && await refreshSession()) r = await send();
      if (!r.ok) {
        const detail = ((await r.json().catch(() => ({}))) as { detail?: string }).detail || r.statusText;
        throw new Error(detail);
      }
      setAvatarUrl("");
      setAvatarHint("No avatar — click to upload one.");
      setAvatarStatus("Removed.");
      window.setTimeout(() => setAvatarStatus(""), 2500);
    } catch (e) {
      setAvatarStatus("Failed: " + (e instanceof Error ? e.message : String(e)));
    }
  };

  if (!auth?.user) {
    return <p className="p-8 text-sm text-gray-500">Loading account…</p>;
  }

  return (
    <div className="bg-gray-50 min-h-screen" data-account-page="profile">
      <header className="sticky top-0 z-30 bg-white/90 backdrop-blur border-b border-gray-200">
        <div className="page-shell h-14 flex items-center justify-between">
          <Link to="/" className="inline-flex items-center gap-2 font-semibold text-gray-900 hover:text-amber-600">
            <img src="/static/logo.png" alt="HuggingArch" className="w-7 h-7 rounded-lg" />
            HuggingArch
          </Link>
          <button
            type="button"
            className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
            onClick={() => void signOut().then(() => { window.location.href = "/"; })}
          >
            Sign out
          </button>
        </div>
      </header>

      <div className="page-shell py-8 flex flex-col md:flex-row gap-6">
        <aside className="md:w-60 shrink-0">
          <div className="flex items-center gap-2.5 px-2 pb-3 mb-2 border-b border-gray-100">
            <AvatarFace url={avatarUrl} text={name} size="w-9 h-9 text-sm" />
            <div className="min-w-0">
              <div className="text-sm font-semibold text-gray-900 truncate">{name}</div>
              <div className="text-xs text-gray-500 truncate mt-0.5">{email}</div>
            </div>
          </div>
          <AccountNav
            active={panel}
            isAdmin={auth.isAdmin}
            localPanels={ACCOUNT_PANELS}
            onPanelSelect={(p) => {
              if ((ACCOUNT_PANELS as readonly string[]).includes(p)) setPanel(p as ProfilePanel);
            }}
          />
        </aside>

        <main className="flex-1 min-w-0">
          {panel === "profile" && (
            <section className={CARD}>
              <p className="text-10 font-semibold uppercase tracking-wider text-amber-700 mb-1">Account</p>
              <div className="flex items-center gap-4 mb-6">
                <button
                  type="button"
                  className="shrink-0 group relative rounded-full overflow-hidden focus:outline-none focus:ring-2 focus:ring-amber-400"
                  title="Click to change avatar"
                  onClick={() => fileRef.current?.click()}
                >
                  <AvatarFace url={avatarUrl} text={username || email} size="w-16 h-16 text-2xl" />
                  <span className="hidden group-hover:flex absolute inset-0 bg-black/40 text-white text-10 uppercase tracking-wide items-center justify-center rounded-full">Change</span>
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/gif"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    e.target.value = "";
                    void onAvatarFile(f);
                  }}
                />
                <div className="flex-1">
                  <h2 className="text-lg font-semibold text-gray-900">Profile</h2>
                  <p className="text-xs text-gray-500">{avatarHint}</p>
                  <div className="mt-2 flex items-center gap-3 text-xs">
                    {avatarUrl ? (
                      <button type="button" className="text-red-600 hover:underline" onClick={() => void removeAvatar()}>
                        Remove avatar
                      </button>
                    ) : null}
                    <span className="text-gray-500">{avatarStatus}</span>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className="block text-xs font-medium text-gray-600">
                  Email
                  <input disabled className={`${INPUT} bg-gray-50`} value={email} />
                </label>
                <label className="block text-xs font-medium text-gray-600">
                  Username
                  <input className={INPUT} value={username} onChange={(e) => setUsername(e.target.value)} />
                </label>
                <label className="block text-xs font-medium text-gray-600">
                  Full name
                  <input className={INPUT} value={fullName} onChange={(e) => setFullName(e.target.value)} />
                </label>
                <label className="block text-xs font-medium text-gray-600 md:col-span-2">
                  Bio
                  <textarea rows={3} className={INPUT} value={bio} onChange={(e) => setBio(e.target.value)} />
                </label>
                <label className="block text-xs font-medium text-gray-600 md:col-span-2">
                  GitHub handle
                  <input className={INPUT} placeholder="your-github-username" value={github} onChange={(e) => setGithub(e.target.value)} />
                  <p className="text-xs text-gray-400 mt-1">Optional — when you contribute a generated spec, the PR credits you with a linkable Co-authored-by.</p>
                </label>
              </div>
              <div className="mt-4 flex items-center justify-end gap-3">
                <span className="text-xs text-red-600">{saveStatus}</span>
                <button type="button" disabled={saving} className={saveOk ? "px-4 py-2 text-sm font-medium rounded-lg bg-green-500 text-white" : BTN} onClick={() => void saveProfile()}>
                  {saving ? "Saving…" : saveOk ? "✓ Saved" : "Save"}
                </button>
              </div>
            </section>
          )}

          {panel === "credentials" && (
            <section className={CARD}>
              <p className="text-10 font-semibold uppercase tracking-wider text-amber-700 mb-1">
                Spec generation · outbound
              </p>
              <div className="flex items-center justify-between mb-1">
                <h2 className="text-lg font-semibold text-gray-900">Model provider keys</h2>
                {!byokEnabled && (
                  <span className="text-xs px-2 py-1 rounded-full bg-gray-100 text-gray-500">Disabled on this server</span>
                )}
              </div>
              <p className="text-sm text-gray-500 mb-4 max-w-xl leading-relaxed">
                Keys you give to HuggingArch so it can call{" "}
                {agents ? orList(agents.agents.map((x) => x.label)) : "a coding agent"} on
                your behalf when generating specs. They are encrypted at rest and
                never shown again after saving.
              </p>
              <div className="space-y-2 mb-4">
                {credErr ? <div className="text-sm text-red-600">Could not load: {credErr}</div> : null}
                {agentsErr ? <div className="text-sm text-red-600">Could not load the agent list: {agentsErr}</div> : null}
                {!credErr && !credsLoaded ? <div className="text-sm text-gray-400">Loading…</div> : null}
                {!credErr && credsLoaded && creds.length === 0 ? <div className="text-sm text-gray-400">No agent keys configured yet.</div> : null}
                {creds.length > 0 && agentTasks.length > 0 ? (
                  <div className="border border-gray-100 rounded-lg px-3 py-2 bg-gray-50/60">
                    <div className="flex flex-wrap items-center gap-2 text-xs mb-1.5">
                      <span className="font-medium text-gray-700">Task routing</span>
                      <span className="text-gray-400">— which agent runs each task; unset tasks use the default agent</span>
                      {routingStatus ? <span className="text-gray-400">{routingStatus}</span> : null}
                    </div>
                    <div className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1.5 max-w-md">
                      {agentTasks.map((t) => (
                        <label key={t.key} className="contents text-xs text-gray-600">
                          <span>{t.label}</span>
                          <ComboBox
                            className="w-full"
                            value={taskAgents[t.key] || ""}
                            placeholder="Default agent"
                            options={[
                              { value: "", label: "Default agent" },
                              ...creds.map((c) => ({
                                value: credId(c),
                                label: credName(c)
                                  ? `${credLabel(c, agents)} (${agentLabel(agents, c.agent_cli)})`
                                  : credLabel(c, agents),
                              })),
                            ]}
                            onChange={(v) => void saveRouting(t.key, v)}
                          />
                        </label>
                      ))}
                    </div>
                  </div>
                ) : null}
                {creds.map((c) => {
                  const id = credId(c);
                  return (
                  <div key={id} className={`border rounded-lg px-3 py-2 ${editingId === id ? "border-amber-300" : "border-gray-200"}`}>
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm min-w-0">
                        <span className="font-medium text-gray-800">{credLabel(c, agents)}</span>
                        {credName(c) ? (
                          <span className="ml-2 text-xs text-gray-400">{agentLabel(agents, c.agent_cli)}</span>
                        ) : null}
                        {c.is_default ? <span className="ml-2 text-11 px-2 py-0.5 rounded-full bg-amber-100 text-amber-700">default</span> : null}
                        <span className="ml-2 text-xs text-gray-400">key set · {c.model || "default model"} · {c.base_url || "default endpoint"}</span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          type="button"
                          title="Test the stored key against its endpoint"
                          className="inline-flex items-center gap-1 text-xs px-2 py-1 border border-gray-200 text-gray-700 hover:bg-gray-50 rounded disabled:opacity-50"
                          disabled={credTest[id] === "testing"}
                          onClick={() => void testCred(c)}
                        >
                          <svg className={`w-3 h-3 ${credTest[id] === "testing" ? "animate-pulse" : ""}`} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
                            <path d="M1.5 8.5a7 7 0 0 1 9 0" /><path d="M3.5 10.2a4 4 0 0 1 5 0" /><circle cx="6" cy="11" r="0.6" fill="currentColor" stroke="none" />
                          </svg>
                          {credTest[id] === "testing" ? "Testing…" : "Test"}
                        </button>
                        {!c.is_default && (
                          <button
                            type="button"
                            className="text-xs px-2 py-1 border border-gray-200 text-gray-700 hover:bg-amber-50 hover:border-amber-300 rounded"
                            onClick={() => void accountJson("POST", `/api/account/agent-credentials/${encodeURIComponent(c.agent_cli)}/default${credNameQuery(c.name)}`).then(loadCreds).catch(alertError)}
                          >
                            Set as default
                          </button>
                        )}
                        <button
                          type="button"
                          className={`text-xs px-2 py-1 border rounded ${editingId === id ? "border-amber-300 bg-amber-50 text-amber-800" : "border-gray-200 text-gray-700 hover:bg-gray-50"}`}
                          aria-expanded={editingId === id}
                          onClick={() => {
                            setAdding(false);
                            setEditingId((cur) => cur === id ? null : id);
                          }}
                        >
                          {editingId === id ? "Cancel" : "Edit"}
                        </button>
                        <button
                          type="button"
                          className="text-xs px-2 py-1 border border-red-200 text-red-700 hover:bg-red-50 rounded"
                          onClick={async () => {
                            if (!(await confirmDialog({ title: `Remove your ${credLabel(c, agents)} key?`,
                              confirmLabel: "Remove", danger: true }))) return;
                            void accountJson("DELETE", `/api/account/agent-credentials/${encodeURIComponent(c.agent_cli)}${credNameQuery(c.name)}`).then(() => {
                              setEditingId((cur) => cur === id ? null : cur);
                              return loadCreds();
                            }).catch(alertError);
                          }}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                    {(() => {
                      const t = credTest[id];
                      if (!t || t === "testing") return null;
                      return (
                        <p className={`text-11 mt-1 ${t.ok ? "text-green-600" : "text-red-600"}`}>
                          {t.ok ? "✓ " : "✗ "}{t.text}
                        </p>
                      );
                    })()}
                    {editingId === id ? (
                      <AgentCredForm
                        mode="edit"
                        cred={c}
                        agents={agents}
                        namesEnabled={namesEnabled}
                        tasks={taskModelsEnabled ? agentTasks : []}
                        siteDefaults={taskDefaults[c.agent_cli] || {}}
                        onSaved={() => { setEditingId(null); void loadCreds(); }}
                        onCancel={() => setEditingId(null)}
                      />
                    ) : taskModelsEnabled && agentTasks.length > 0 ? (
                      <TaskModelSummary
                        cred={c}
                        tasks={agentTasks}
                        siteDefaults={taskDefaults[c.agent_cli] || {}}
                      />
                    ) : null}
                  </div>
                  );
                })}
                {adding ? (
                  <div className="border border-amber-300 rounded-lg px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium text-gray-800">New key</span>
                      <button
                        type="button"
                        className="text-xs px-2 py-1 border border-amber-300 bg-amber-50 text-amber-800 rounded"
                        onClick={() => setAdding(false)}
                      >
                        Cancel
                      </button>
                    </div>
                    <AgentCredForm
                      mode="add"
                      agents={agents}
                      namesEnabled={namesEnabled}
                      tasks={taskModelsEnabled ? agentTasks : []}
                      onSaved={() => { setAdding(false); void loadCreds(); }}
                      onCancel={() => setAdding(false)}
                    />
                  </div>
                ) : (
                  <button
                    type="button"
                    className="w-full text-sm px-3 py-2 border border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-amber-300 hover:text-amber-700 hover:bg-amber-50/40 disabled:opacity-50"
                    disabled={!agents}
                    onClick={() => { setEditingId(null); setAdding(true); }}
                  >
                    Add a key
                  </button>
                )}
              </div>
            </section>
          )}

          {panel === "credentials" && (
            <section className={CARD}>
              <p className="text-10 font-semibold uppercase tracking-wider text-amber-700 mb-1">
                Benchmark upload · inbound
              </p>
              <h2 className="text-lg font-semibold text-gray-900 mb-1">
                HuggingArch access tokens
              </h2>
              <p className="text-sm text-gray-500 mb-3 max-w-xl leading-relaxed">
                Tokens HuggingArch issues to your GPU box. <code className="font-mono bg-gray-100 px-1 rounded">hbench</code>
                {" "}uses one to upload benchmark results as you — it cannot read
                your provider keys or start spec generation. Shown once — store it safely.
              </p>
              <div className="flex items-center gap-2 mb-3">
                <input className={`${INPUT} mt-0 flex-1`} placeholder="Name (e.g. rtx4090-box)" value={tokName} onChange={(e) => setTokName(e.target.value)} />
                <button
                  type="button"
                  className={`${BTN} shrink-0`}
                  onClick={() => {
                    void (async () => {
                      try {
                        const issued = await accountJson<{ token?: string }>("POST", "/api/account/tokens", { name: tokName.trim() });
                        setTokNew(issued.token || "");
                        setTokName("");
                        void loadTokens();
                      } catch (e) {
                        void alertError(e);
                      }
                    })();
                  }}
                >
                  Generate token
                </button>
              </div>
              {tokNew ? (
                <div className="mb-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <div className="text-xs text-amber-800 mb-1 font-medium">Copy this now — it won&apos;t be shown again:</div>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 font-mono text-[12px] text-gray-900 bg-white border border-amber-200 rounded px-2 py-1 break-all">{tokNew}</code>
                    <button
                      type="button"
                      className="px-2 py-1 text-xs border border-amber-300 rounded hover:bg-amber-100 shrink-0"
                      onClick={() => {
                        void navigator.clipboard?.writeText(tokNew);
                        setTokCopied(true);
                        window.setTimeout(() => setTokCopied(false), 1500);
                      }}
                    >
                      {tokCopied ? "Copied" : "Copy"}
                    </button>
                  </div>
                </div>
              ) : null}
              <div className="space-y-2">
                {tokErr ? <div className="text-sm text-red-600">Could not load: {tokErr}</div> : null}
                {!tokErr && !tokLoaded ? <div className="text-sm text-gray-400">Loading…</div> : null}
                {!tokErr && tokLoaded && tokens.length === 0 ? <div className="text-sm text-gray-400">No tokens yet.</div> : null}
                {tokens.map((t) => (
                  <div key={t.id} className={`flex items-center justify-between border border-gray-200 rounded-lg px-3 py-2 ${t.revoked_at ? "opacity-50" : ""}`}>
                    <div className="text-sm min-w-0">
                      <span className="font-medium text-gray-800">{t.name || "(unnamed)"}</span>
                      <span className="ml-2 font-mono text-11 text-gray-400">{t.token_prefix || ""}…</span>
                      <span className="ml-2 text-xs text-gray-400">{t.last_used_at ? new Date(t.last_used_at).toLocaleString() : "never used"}</span>
                      {t.revoked_at ? <span className="ml-2 text-11 text-red-500">revoked</span> : null}
                    </div>
                    {!t.revoked_at && (
                      <button
                        type="button"
                        className="text-xs px-2 py-1 text-red-600 hover:bg-red-50 rounded shrink-0"
                        onClick={async () => {
                          if (!(await confirmDialog({ title: "Revoke this token?",
                            message: "Any box using it will stop uploading.", confirmLabel: "Revoke", danger: true }))) return;
                          void accountJson("DELETE", `/api/account/tokens/${encodeURIComponent(t.id)}`).then(loadTokens).catch(alertError);
                        }}
                      >
                        Revoke
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}
