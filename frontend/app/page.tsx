"use client";

import { Fragment, useEffect, useMemo, useRef, useState } from "react";

type Backend = "postgresql" | "mysql" | "duckdb";
type Page = "analysis" | "database" | "knowledge" | "model" | "runs";
type Model = "deepseek-v4-pro" | "deepseek-v4-flash";
type Profile = { id: string; label: string; description: string; backend: Backend; host: string; port: number; username: string; database: string; password?: string; password_saved?: boolean; duckdb_path: string; knowledge_root: string };
type KnowledgeSummary = { path: string; card_count?: number; types?: Record<string, number>; error?: string };
type ApiState = { active: Profile; profiles: Profile[]; model: Model; models: Model[]; knowledge: KnowledgeSummary; model_configured: boolean };
type SqlResult = { columns?: string[]; rows?: Record<string, unknown>[]; returned_rows?: number; truncated?: boolean; status?: string; error_type?: string; message?: string };
type ChatResponse = { run_id: string; status: "success" | "incomplete" | "canceled"; thread_id: string; model: Model; latency_ms: number; answer: string; tool_counts: Record<string, number>; sql_queries: { tool_call_id: string; sql: string; result?: SqlResult }[]; result_preview?: SqlResult | null; knowledge_view?: { knowledge_view_mode?: string } | null };
type ToolCallView = { name: string; arguments: Record<string, unknown> };
type LlmRoundView = { number: number; content: string; toolCalls: ToolCallView[] };
type ChatStreamEvent = { type: "started" | "round" | "progress" | "final" | "error"; run_id: string; thread_id?: string; message?: string; round?: number; content?: string; tool_calls?: ToolCallView[]; response?: ChatResponse };
type ChatItem = { id: string; role: "user" | "assistant"; content: string; details?: ChatResponse };
type RunEvent = { run_id: string; created_at: string; thread_id: string; model: Model; status: string; latency_ms: number; tool_counts: Record<string, number>; sql_count: number };
type Notice = { tone: "success" | "error" | "info"; text: string };

const emptyProfile = (): Profile => ({ id: `profile-${Date.now()}`, label: "新配置方案", description: "", backend: "postgresql", host: "", port: 5432, username: "", database: "", password: "", duckdb_path: "", knowledge_root: "" });
const pageNames: Record<Page, string> = { analysis: "数据分析", database: "数据源", knowledge: "Knowledge", model: "模型设置", runs: "运行记录" };

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...options, headers: { "Content-Type": "application/json", ...(options?.headers || {}) } });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "请求失败");
  return payload as T;
}

async function readJsonLines(response: Response, onEvent: (event: ChatStreamEvent) => void) {
  if (!response.body) throw new Error("浏览器没有收到可读取的响应流。");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line) as ChatStreamEvent);
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as ChatStreamEvent);
}

function backendName(backend: Backend) { return { postgresql: "PostgreSQL", mysql: "MySQL", duckdb: "DuckDB" }[backend]; }
function modelName(model: Model) { return model === "deepseek-v4-pro" ? "DeepSeek V4 Pro" : "DeepSeek V4 Flash"; }
function runStatusName(status: string) { return { success: "完成", incomplete: "未完成", canceled: "已停止", error: "失败" }[status] || status; }

function InlineText({ value }: { value: string }) {
  const parts = value.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return <>{parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    return <Fragment key={index}>{part}</Fragment>;
  })}</>;
}

function AnswerBody({ content }: { content: string }) {
  const lines = content.replace(/\r/g, "").split("\n");
  const blocks: React.ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (line.trim().startsWith("```")) {
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) code.push(lines[index++]);
      index += 1;
      blocks.push(<pre className="answer-code" key={`code-${index}`}><code>{code.join("\n")}</code></pre>);
      continue;
    }
    if (line.includes("|") && index + 1 < lines.length && /^\s*\|?\s*:?-{3}/.test(lines[index + 1])) {
      const parse = (row: string) => row.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
      const header = parse(line);
      index += 2;
      const rows: string[][] = [];
      while (index < lines.length && lines[index].includes("|")) rows.push(parse(lines[index++]));
      blocks.push(<div className="table-scroll" key={`table-${index}`}><table><thead><tr>{header.map((cell, cellIndex) => <th key={cellIndex}><InlineText value={cell} /></th>)}</tr></thead><tbody>{rows.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}><InlineText value={cell} /></td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    if (/^#{1,4}\s+/.test(line)) { blocks.push(<h3 key={`heading-${index}`}><InlineText value={line.replace(/^#{1,4}\s+/, "")} /></h3>); index += 1; continue; }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) items.push(lines[index++].replace(/^\s*[-*]\s+/, ""));
      blocks.push(<ul key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}><InlineText value={item} /></li>)}</ul>);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) items.push(lines[index++].replace(/^\s*\d+\.\s+/, ""));
      blocks.push(<ol key={`ordered-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}><InlineText value={item} /></li>)}</ol>);
      continue;
    }
    blocks.push(line.trim() ? <p key={`line-${index}`}><InlineText value={line} /></p> : <div className="answer-gap" key={`gap-${index}`} />);
    index += 1;
  }
  return <div className="answer-body">{blocks}</div>;
}

function ResultTable({ result }: { result?: SqlResult | null }) {
  if (!result) return <div className="result-message error">未收到这条 SQL 对应的 Tool Result。</div>;
  if (["error", "denied", "rejected"].includes(result.status || "") || result.error_type) return <div className="result-message error">{result.message || `SQL 执行失败：${result.error_type || result.status}`}</div>;
  if (!result.columns?.length) return <div className="result-message">{result.message || "Tool Result 没有可展示的结果列。"}</div>;
  const rows = result.rows || [];
  return <div className="result-block"><div className="result-heading"><strong>该 SQL 的查询结果</strong><span>{result.returned_rows ?? rows.length} 行{result.truncated ? " · 已截断" : ""}</span></div><div className="table-scroll"><table><thead><tr>{result.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{rows.length ? rows.map((row, rowIndex) => <tr key={rowIndex}>{result.columns!.map((column) => <td key={column}>{String(row[column] ?? "NULL")}</td>)}</tr>) : <tr><td colSpan={result.columns.length}>结果为空</td></tr>}</tbody></table></div></div>;
}

const toolNames: Record<string, string> = {
  browse_knowledge: "浏览 Knowledge 目录",
  search_knowledge: "搜索 Knowledge",
  read_knowledge: "读取 KnowledgeCard",
  execute_readonly_sql: "执行只读 SQL",
};

function ToolCallCard({ call }: { call: ToolCallView }) {
  const sql = typeof call.arguments.sql === "string" ? call.arguments.sql : "";
  return <div className="round-tool-call"><div className="round-tool-heading"><strong>{toolNames[call.name] || call.name}</strong><code>{call.name}</code></div><pre><code>{sql || JSON.stringify(call.arguments, null, 2)}</code></pre></div>;
}

export default function Home() {
  const [page, setPage] = useState<Page>("analysis");
  const [state, setState] = useState<ApiState | null>(null);
  const [form, setForm] = useState<Profile>(emptyProfile());
  const [model, setModel] = useState<Model>("deepseek-v4-pro");
  const [modelApiKey, setModelApiKey] = useState("");
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [question, setQuestion] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [currentRound, setCurrentRound] = useState<LlmRoundView | null>(null);
  const [roundStatus, setRoundStatus] = useState("");
  const [stopRequested, setStopRequested] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState<Notice | null>(null);
  const [runs, setRuns] = useState<RunEvent[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const selectedProfile = useMemo(() => state?.profiles.find((profile) => profile.id === form.id), [state, form.id]);

  const loadState = async (preferredId?: string) => {
    const next = await api<ApiState>("/api/state");
    setState(next); setModel(next.model);
    const preferred = next.profiles.find((profile) => profile.id === preferredId);
    setForm({ ...(preferred || next.active), password: "" });
  };
  const loadRuns = async () => setRuns((await api<{ runs: RunEvent[] }>("/api/runs")).runs);
  useEffect(() => { loadState().catch((error) => setNotice({ tone: "error", text: error.message })); }, []);
  useEffect(() => { if (page === "runs") loadRuns().catch(() => undefined); }, [page]);
  const update = <K extends keyof Profile>(key: K, value: Profile[K]) => setForm((current) => ({ ...current, [key]: value }));

  const runAction = async (name: string, action: () => Promise<{ message?: string; details?: unknown }>) => {
    setBusy(name);
    try { const result = await action(); setNotice({ tone: "success", text: result.message || "操作成功" }); return result; }
    catch (error) { setNotice({ tone: "error", text: error instanceof Error ? error.message : "操作失败" }); return null; }
    finally { setBusy(""); }
  };

  const sendQuestion = async () => {
    const text = question.trim();
    if (!text || chatBusy) return;
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", content: text }]);
    setQuestion(""); setChatBusy(true); setStopRequested(false); setCurrentRound(null); setRoundStatus("正在启动分析…"); setActiveRunId(null); activeRunIdRef.current = null;
    try {
      const response = await fetch("/api/chat/stream", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, thread_id: threadId, model }) });
      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "请求失败");
      }
      let finalReceived = false;
      await readJsonLines(response, (event) => {
        if (event.type === "started") {
          activeRunIdRef.current = event.run_id;
          setActiveRunId(event.run_id);
          if (event.thread_id) setThreadId(event.thread_id);
          setRoundStatus("任务已启动，正在等待模型输出…");
        } else if (event.type === "round") {
          setCurrentRound({ number: event.round || 1, content: event.content || "", toolCalls: event.tool_calls || [] });
          setRoundStatus(event.message || "本轮模型输出已生成。");
        } else if (event.type === "progress" && event.message) {
          setRoundStatus(event.message);
        } else if (event.type === "final" && event.response) {
          finalReceived = true;
          setThreadId(event.response.thread_id);
          setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: event.response!.answer || "分析完成，但没有生成可展示的回答。", details: event.response }]);
        } else if (event.type === "error") {
          throw new Error(event.message || "分析执行失败。");
        }
      });
      if (!finalReceived) throw new Error("响应流已结束，但没有收到最终结果。");
    } catch (error) {
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: error instanceof Error ? error.message : "分析执行失败。" }]);
    } finally { activeRunIdRef.current = null; setActiveRunId(null); setStopRequested(false); setCurrentRound(null); setRoundStatus(""); setChatBusy(false); }
  };

  const stopRun = async () => {
    const runId = activeRunIdRef.current;
    if (!runId || stopRequested) return;
    setStopRequested(true);
    setRoundStatus("已请求停止，正在等待当前步骤安全结束…");
    try {
      const result = await api<{ status: string; message: string }>(`/api/runs/${runId}/cancel`, { method: "POST" });
      if (result.status === "not_running") setNotice({ tone: "info", text: result.message });
    } catch (error) {
      setStopRequested(false);
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "停止任务失败。" });
    }
  };

  const newConversation = () => { if (chatBusy) return; setThreadId(crypto.randomUUID()); setMessages([]); setQuestion(""); setCurrentRound(null); setRoundStatus(""); };
  const saveAndApply = async () => { const result = await runAction("save", () => api("/api/save-and-apply", { method: "POST", body: JSON.stringify(form) })); if (result) { await loadState(form.id); newConversation(); } };
  const saveModelSettings = async () => { const result = await runAction("model", () => api("/api/model-settings", { method: "POST", body: JSON.stringify({ model, api_key: modelApiKey }) })); if (result) { setModelApiKey(""); await loadState(form.id); } };
  const uploadKnowledge = async (file?: File) => {
    if (!file) return;
    setBusy("upload");
    try {
      const response = await fetch("/api/import-knowledge", { method: "POST", headers: { "Content-Type": "application/zip", "X-Knowledge-Name": encodeURIComponent(file.name.replace(/\.zip$/i, "")) }, body: file });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "导入失败");
      update("knowledge_root", payload.details.path); setNotice({ tone: "success", text: payload.message });
    } catch (error) { setNotice({ tone: "error", text: error instanceof Error ? error.message : "导入失败" }); }
    finally { setBusy(""); if (fileInput.current) fileInput.current.value = ""; }
  };

  if (!state) return <main className="loading-shell">正在加载 DataAgent…</main>;

  return <div className="product-shell">
    <aside className="main-nav"><div className="brand"><span>DA</span><strong>DataAgent</strong></div><nav>
      <button className={page === "analysis" ? "active" : ""} onClick={() => setPage("analysis")}><span>⌁</span>分析</button>
      <button className={page === "database" ? "active" : ""} onClick={() => setPage("database")}><span>▤</span>数据源</button>
      <button className={page === "knowledge" ? "active" : ""} onClick={() => setPage("knowledge")}><span>◇</span>Knowledge</button>
      <button className={page === "model" ? "active" : ""} onClick={() => setPage("model")}><span>◉</span>模型</button>
      <button className={page === "runs" ? "active" : ""} onClick={() => setPage("runs")}><span>≡</span>运行</button>
    </nav><div className={`nav-status ${state.model_configured ? "" : "warning"}`}><span />{state.model_configured ? "服务已连接" : "模型密钥未配置"}</div></aside>

    <main className="main-area"><header className="app-header"><div><h1>{pageNames[page]}</h1><p>{backendName(state.active.backend)} · {state.active.database || "本地文件"}</p></div><div className="header-actions">
      <label className="model-select"><span>模型</span><select disabled={chatBusy} value={model} onChange={(event) => setModel(event.target.value as Model)}>{state.models.map((item) => <option key={item} value={item}>{modelName(item)}</option>)}</select></label>
      {page === "analysis" && <button className="button secondary" disabled={chatBusy} onClick={newConversation}>新会话</button>}
    </div></header>

    {page === "analysis" && <section className="analysis-page"><div className="conversation">
      {messages.length === 0 ? <div className="empty-state"><h2>开始一次数据分析</h2><p>输入业务问题。Agent 会查找已配置的 Knowledge，执行只读 SQL，再返回结果。</p></div> : messages.map((message) => <article className={`message ${message.role}`} key={message.id}>
        <div className="message-label">{message.role === "user" ? "你" : "DataAgent"}</div>{message.role === "assistant" ? <AnswerBody content={message.content} /> : <p>{message.content}</p>}
        {message.details && <details className="run-details"><summary>查看 SQL 与运行信息</summary><div className="metric-row"><span>{(message.details.latency_ms / 1000).toFixed(1)} 秒</span><span>{message.details.sql_queries.length} 次 SQL</span><span>{message.details.knowledge_view?.knowledge_view_mode || "-"} View</span>{message.details.status === "incomplete" && <span className="warning-text">未完成</span>}</div>{message.details.sql_queries.map((query, queryIndex) => <div className="sql-card" key={query.tool_call_id || queryIndex}><div>SQL {queryIndex + 1}</div><pre><code>{query.sql}</code></pre><ResultTable result={query.result} /></div>)}</details>}
      </article>)}{chatBusy && <article className="message assistant pending"><div className="message-label">{currentRound ? `DataAgent · 第 ${currentRound.number} 轮` : "DataAgent"}</div>{currentRound ? <div className="current-round">{currentRound.content ? <><div className="round-section-title">模型本轮输出</div><AnswerBody content={currentRound.content} /></> : <p className="round-empty">本轮没有文本输出，模型直接发起了 Tool Call。</p>}{currentRound.toolCalls.length > 0 && <div className="round-tools"><div className="round-section-title">本轮 Tool Call</div>{currentRound.toolCalls.map((call, index) => <ToolCallCard call={call} key={`${call.name}-${index}`} />)}</div>}</div> : <p className="live-activity">正在等待第一轮模型输出…</p>}<div className="round-status"><span />{roundStatus || "正在分析…"}</div>{stopRequested && <small>停止将在当前模型或工具调用结束后的安全位置生效。</small>}</article>}
    </div><div className="composer"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuestion(); } }} placeholder="输入数据分析问题，Enter 发送，Shift + Enter 换行" /><button className={`send-button ${chatBusy ? "stop-button" : ""}`} disabled={chatBusy ? !activeRunId || stopRequested : !question.trim()} onClick={chatBusy ? stopRun : sendQuestion}>{chatBusy ? (stopRequested ? "正在停止" : activeRunId ? "停止" : "启动中") : "发送"}</button></div></section>}

    {page === "database" && <section className="content-page two-column-page"><aside className="profile-panel"><div className="panel-heading"><h2>配置方案</h2><button onClick={() => setForm(emptyProfile())}>新建</button></div>{state.profiles.map((profile) => <button className={`profile-item ${profile.id === form.id ? "active" : ""}`} key={profile.id} onClick={() => setForm({ ...profile, password: "" })}><strong>{profile.label}</strong><span>{backendName(profile.backend)} · {profile.database || "本地文件"}</span></button>)}</aside>
      <div className="settings-panel"><div className="panel-heading"><div><h2>{selectedProfile ? "编辑数据源" : "新数据源"}</h2><p>账号应只具备读取权限。</p></div><span className="apply-note">保存后立即生效</span></div><div className="form-grid">
        <label><span>方案名称</span><input value={form.label} onChange={(event) => update("label", event.target.value)} /></label><label><span>方案 ID</span><input disabled={Boolean(selectedProfile)} value={form.id} onChange={(event) => update("id", event.target.value.toLowerCase())} /></label>
        <label className="wide"><span>数据库类型</span><select value={form.backend} onChange={(event) => { const backend = event.target.value as Backend; setForm((current) => ({ ...current, backend, port: backend === "mysql" ? 3306 : backend === "postgresql" ? 5432 : 0 })); }}><option value="postgresql">PostgreSQL</option><option value="mysql">MySQL</option><option value="duckdb">DuckDB</option></select></label>
        {form.backend === "duckdb" ? <label className="wide"><span>DuckDB 文件</span><input value={form.duckdb_path} onChange={(event) => update("duckdb_path", event.target.value)} /></label> : <><label><span>Host</span><input value={form.host} onChange={(event) => update("host", event.target.value)} placeholder="host.docker.internal" /></label><label><span>Port</span><input type="number" value={form.port || ""} onChange={(event) => update("port", Number(event.target.value))} /></label><label><span>只读用户名</span><input value={form.username} onChange={(event) => update("username", event.target.value)} /></label><label><span>密码 {form.password_saved ? "（已保存）" : ""}</span><input type="password" value={form.password || ""} onChange={(event) => update("password", event.target.value)} placeholder={form.password_saved ? "留空继续使用" : ""} /></label><label className="wide"><span>数据库名称</span><input value={form.database} onChange={(event) => update("database", event.target.value)} placeholder="例如：cold_chain_pharma_compliance" /><small>填写 PostgreSQL 或 MySQL 中实际存在的数据库名称。</small></label></>}
      </div><div className="form-actions"><button className="button secondary" disabled={Boolean(busy)} onClick={() => runAction("test", () => api("/api/test-database", { method: "POST", body: JSON.stringify(form) }))}>测试连接</button><button className="button primary" disabled={Boolean(busy)} onClick={saveAndApply}>保存并应用</button></div></div>
    </section>}

    {page === "knowledge" && <section className="content-page knowledge-page"><div className="summary-card"><div><h2>当前 Knowledge</h2><p>{state.knowledge.path}</p></div><div className="summary-number"><strong>{state.knowledge.card_count ?? "-"}</strong><span>KnowledgeCards</span></div></div><div className="knowledge-grid">
      <div className="settings-panel"><div className="panel-heading"><div><h2>导入 Knowledge</h2><p>上传符合 KnowledgeCard 结构的 ZIP 包。</p></div></div><input ref={fileInput} type="file" accept=".zip,application/zip" hidden onChange={(event) => uploadKnowledge(event.target.files?.[0])} /><button className="upload-area" disabled={busy === "upload"} onClick={() => fileInput.current?.click()}><strong>{busy === "upload" ? "正在校验…" : "选择 ZIP 文件"}</strong><span>导入后先校验，不会立即替换当前配置</span></button></div>
      <div className="settings-panel"><div className="panel-heading"><div><h2>Knowledge 路径</h2><p>本地运行时也可以直接指定目录。</p></div></div><label><span>目录</span><input value={form.knowledge_root} onChange={(event) => update("knowledge_root", event.target.value)} /></label><div className="form-actions"><button className="button secondary" disabled={Boolean(busy)} onClick={() => runAction("validate", () => api("/api/validate-knowledge", { method: "POST", body: JSON.stringify({ knowledge_root: form.knowledge_root }) }))}>验证</button><button className="button primary" disabled={Boolean(busy)} onClick={saveAndApply}>保存并应用</button></div></div>
    </div>{state.knowledge.types && <div className="type-list">{Object.entries(state.knowledge.types).map(([type, count]) => <div key={type}><span>{type}</span><strong>{count}</strong></div>)}</div>}</section>}

    {page === "model" && <section className="content-page model-page"><div className="settings-panel"><div className="panel-heading"><div><h2>DeepSeek 模型</h2><p>当前仅支持 DeepSeek V4 Pro 与 DeepSeek V4 Flash。</p></div><span className={`config-status ${state.model_configured ? "configured" : ""}`}>{state.model_configured ? "API Key 已配置" : "API Key 未配置"}</span></div><div className="form-grid">
      <label className="wide"><span>模型</span><select value={model} onChange={(event) => setModel(event.target.value as Model)}>{state.models.map((item) => <option key={item} value={item}>{modelName(item)}</option>)}</select></label>
      <label className="wide"><span>模型 API Key</span><input type="password" autoComplete="new-password" value={modelApiKey} onChange={(event) => setModelApiKey(event.target.value)} placeholder={state.model_configured ? "留空继续使用已保存的 API Key" : "输入 DeepSeek API Key"} /><small>密钥只保存在本机，页面不会读取或回显完整内容。</small></label>
    </div><div className="form-actions"><button className="button primary" disabled={Boolean(busy)} onClick={saveModelSettings}>保存模型配置</button></div></div></section>}

    {page === "runs" && <section className="content-page runs-page"><div className="panel-heading"><div><h2>最近 50 次运行</h2><p>仅保留当前服务进程中的精简指标，不保存问题或答案。</p></div><button className="button secondary" onClick={loadRuns}>刷新</button></div>{runs.length ? <div className="table-scroll"><table><thead><tr><th>时间</th><th>模型</th><th>状态</th><th>耗时</th><th>SQL</th><th>Thread</th></tr></thead><tbody>{runs.map((run) => <tr key={run.run_id}><td>{new Date(run.created_at).toLocaleString("zh-CN")}</td><td>{modelName(run.model)}</td><td><span className={`run-status ${run.status}`}>{runStatusName(run.status)}</span></td><td>{(run.latency_ms / 1000).toFixed(1)}s</td><td>{run.sql_count}</td><td className="mono">{run.thread_id.slice(0, 8)}</td></tr>)}</tbody></table></div> : <div className="empty-list">当前还没有运行记录。</div>}</section>}
    {notice && <div className={`toast ${notice.tone}`}><span>{notice.text}</span><button onClick={() => setNotice(null)}>×</button></div>}
  </main></div>;
}
