"use client";

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react";
import DatabaseExplorer from "./DatabaseExplorer";
import KnowledgeGraph, { type LiveKnowledgeTrace } from "./KnowledgeGraph";
import KnowledgeWorkspace from "./KnowledgeWorkspace";
import { fetchForUser, fetchJsonForUser } from "./api-client";
import ContextWindowMeter, { type ContextUsage } from "./ContextWindowMeter";
import DatasourceConversationPicker, { LEGACY_SCOPE, conversationListPath, filterConversationScope, hasMultipleKnowledgeBases, knowledgeProfiles, mergeConversationSources, sameConversationBinding, type ConversationBinding, type ConversationSource } from "./DatasourceConversationPicker";

type Backend = "postgresql" | "mysql" | "duckdb";
type Page = "analysis" | "database" | "knowledge" | "model";
type Model = "deepseek-v4-pro" | "deepseek-v4-flash";
type Profile = { id: string; label: string; description: string; backend: Backend; host: string; port: number; username: string; database: string; password?: string; password_saved?: boolean; duckdb_path: string; knowledge_root: string; binding?: ConversationBinding | null };
type Account = { login_id: string; user_id: string; display_name: string; avatar: string; workspace_id: string; workspace_name: string; resources_ready: boolean; role: "admin" | "analyst" | "viewer"; role_label: string; permissions: string[] };
type KnowledgeSummary = { path: string; card_count?: number; types?: Record<string, number>; error?: string };
type ApiState = { active: Profile; profiles: Profile[]; model: Model; models: Model[]; model_context_windows?: Partial<Record<Model, number | null>>; knowledge: KnowledgeSummary; model_configured: boolean; workspace: Account };
type AccountsResponse = { demo_mode: boolean; current: Account; accounts: Account[] };
type SqlResult = { columns?: string[]; rows?: Record<string, unknown>[]; returned_rows?: number; truncated?: boolean; status?: string; error_type?: string; message?: string };
type ArtifactView = { id: string; kind: "report"; title: string; preview_url: string };
type ChatResponse = { request_id: string; status: "success" | "paused" | "canceled"; thread_id: string; model: Model; latency_ms: number; answer: string; tool_counts: Record<string, number>; sql_queries: { tool_call_id: string; sql: string; result?: SqlResult }[]; result_preview?: SqlResult | null; knowledge_view?: { knowledge_view_mode?: string } | null; artifacts?: ArtifactView[]; context_usage?: ContextUsage | null };
type ToolCallView = { name: string; arguments: Record<string, unknown> };
type LlmRoundView = { number: number; content: string; toolCalls: ToolCallView[] };
type ChatStreamEvent = { type: "started" | "round" | "progress" | "knowledge_trace" | "final" | "error"; request_id: string; thread_id?: string; message?: string; round?: number; content?: string; tool_calls?: ToolCallView[]; response?: ChatResponse; action?: "open" | "close"; stage?: string; mode?: string; active_ids?: string[]; context_usage?: ContextUsage | null };
type ChatItem = { id: string; role: "user" | "assistant"; content: string; details?: ChatResponse };
type ConversationSummaryPayload = { thread_id: string; title: string; custom_title: boolean; created_at: string; updated_at: string; binding: ConversationBinding | null };
type ConversationMessagePayload = { id: number; role: "user" | "assistant"; content: string; details: ChatResponse | null; created_at: string };
type ConversationDetailPayload = ConversationSummaryPayload & { messages: ConversationMessagePayload[]; can_continue: boolean };
type ConversationListResponse = { conversations: ConversationSummaryPayload[]; data_sources: ConversationSource[]; legacy_count: number; active_binding: ConversationBinding | null };
type RenameConversationResponse = { conversation: ConversationDetailPayload };
type ConversationHistoryItem = { threadId: string; title: string; customTitle: boolean; binding: ConversationBinding | null };
type Notice = { tone: "success" | "error" | "info"; text: string };

const emptyProfile = (): Profile => ({ id: `profile-${Date.now()}`, label: "新配置方案", description: "", backend: "postgresql", host: "", port: 5432, username: "", database: "", password: "", duckdb_path: "", knowledge_root: "" });
const pageNames: Record<Page, string> = { analysis: "数据分析", database: "数据源", knowledge: "知识库", model: "模型设置" };
const DEFAULT_DEV_USER = "admin-a";

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
function conversationSummary(payload: ConversationSummaryPayload): ConversationHistoryItem {
  return { threadId: payload.thread_id, title: payload.title, customTitle: payload.custom_title, binding: payload.binding };
}
function conversationMessages(payload: ConversationDetailPayload): ChatItem[] {
  return payload.messages.map((message) => ({
    id: String(message.id),
    role: message.role,
    content: message.content,
    details: message.details ?? undefined,
  }));
}
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

function ToolCallCard({ call }: { call: ToolCallView }) {
  const sql = typeof call.arguments.sql === "string" ? call.arguments.sql : "";
  const title = typeof call.arguments.title === "string" ? call.arguments.title : "";
  const visualTool = ["create_metric_cards", "create_chart", "compose_dashboard", "export_report"].includes(call.name);
  return <div className="round-tool-call"><div className="round-tool-heading"><code>{call.name}</code></div><pre><code>{sql || (visualTool ? title || "生成可视化" : JSON.stringify(call.arguments, null, 2))}</code></pre></div>;
}

function ArtifactPreview({ artifact }: { artifact: ArtifactView }) {
  return <section className="artifact-preview"><header><h3>{artifact.title}</h3><a href={artifact.preview_url} target="_blank" rel="noreferrer">新窗口打开</a></header><iframe title={artifact.title} src={artifact.preview_url} sandbox="allow-scripts" /></section>;
}

export default function Home() {
  const [devUser, setDevUser] = useState(DEFAULT_DEV_USER);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [page, setPage] = useState<Page>("analysis");
  const [state, setState] = useState<ApiState | null>(null);
  const [form, setForm] = useState<Profile>(emptyProfile());
  const [model, setModel] = useState<Model>("deepseek-v4-pro");
  const [modelApiKey, setModelApiKey] = useState("");
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [conversationHistory, setConversationHistory] = useState<ConversationHistoryItem[]>([]);
  const [conversationMenu, setConversationMenu] = useState<string | null>(null);
  const [conversationBusy, setConversationBusy] = useState(true);
  const [historySources, setHistorySources] = useState<ConversationSource[]>([]);
  const [historyScope, setHistoryScope] = useState("");
  const [legacyCount, setLegacyCount] = useState(0);
  const [threadBinding, setThreadBinding] = useState<ConversationBinding | null>(null);
  const [conversationCanContinue, setConversationCanContinue] = useState(false);
  const [configurationReady, setConfigurationReady] = useState(true);
  const [knowledgeMenuOpen, setKnowledgeMenuOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [activeRequestId, setActiveRequestId] = useState<string | null>(null);
  const [currentRound, setCurrentRound] = useState<LlmRoundView | null>(null);
  const [contextUsage, setContextUsage] = useState<ContextUsage | null>(null);
  const [roundStatus, setRoundStatus] = useState("");
  const [stopRequested, setStopRequested] = useState(false);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState<Notice | null>(null);
  const [runtimeRevision, setRuntimeRevision] = useState(0);
  const [knowledgeDirty, setKnowledgeDirty] = useState(false);
  const [knowledgeSaving, setKnowledgeSaving] = useState(false);
  const [liveKnowledgeTrace, setLiveKnowledgeTrace] = useState<LiveKnowledgeTrace | null>(null);
  const [liveKnowledgeClosing, setLiveKnowledgeClosing] = useState(false);
  const [liveKnowledgeMinimized, setLiveKnowledgeMinimized] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const activeRequestIdRef = useRef<string | null>(null);
  const liveKnowledgeTraceRef = useRef<LiveKnowledgeTrace | null>(null);
  const liveKnowledgeCloseTimer = useRef<number | null>(null);
  const historyScopeRef = useRef("");
  const scopeGeneration = useRef(0);
  const historyRequestGeneration = useRef(0);
  const selectedProfile = useMemo(() => state?.profiles.find((profile) => profile.id === form.id), [state, form.id]);
  const canChat = state?.workspace.permissions.includes("chat:run") ?? false;
  const canConfigure = state?.workspace.permissions.includes("config:write") ?? false;
  const canWriteConversations = state?.workspace.permissions.includes("conversation:write") ?? false;
  const resourcesReady = state?.workspace.resources_ready ?? false;
  const dataSources = useMemo(() => mergeConversationSources(state?.profiles || [], historySources, state?.active.binding), [state, historySources]);
  const displayedSource = dataSources.find((source) => source.data_source_id === historyScope);
  const knowledgeChoices = useMemo(() => knowledgeProfiles(state?.profiles || [], historyScope), [state, historyScope]);
  const showKnowledgeLabels = useMemo(() => hasMultipleKnowledgeBases(state?.profiles || [], conversationHistory, historyScope), [state, conversationHistory, historyScope]);
  const originalProfile = state?.profiles.find((profile) => sameConversationBinding(profile.binding, threadBinding));
  const sourceIsActive = Boolean(historyScope && historyScope === state?.active.binding?.data_source_id);
  const canContinue = configurationReady && conversationCanContinue && sourceIsActive && sameConversationBinding(threadBinding, state?.active.binding);
  const readOnlyReason = historyScope === LEGACY_SCOPE ? "未归属会话 · 只读" : !configurationReady ? "配置已更新，请刷新页面" : "原数据源或知识库未连接 · 只读";
  const switchBusy = chatBusy || conversationBusy || Boolean(busy);
  const chatUnavailableReason = !canChat
    ? "当前账号为只读角色，不能发起分析。"
    : !resourcesReady
      ? "当前工作空间尚未配置独立数据库、模型和知识库。"
      : "";
  const api = <T,>(path: string, options?: RequestInit, requestUser = devUser) => fetchJsonForUser<T>(path, requestUser, options);
  const leaveKnowledge = () => {
    if (knowledgeSaving) { setNotice({ tone: "info", text: "知识正在保存，请稍候。" }); return false; }
    if (page === "knowledge" && knowledgeDirty && !window.confirm("放弃尚未保存的知识修改？")) return false;
    setKnowledgeDirty(false);
    return true;
  };
  const navigatePage = (next: Page) => { if (next !== page && !leaveKnowledge()) return; setPage(next); };
  const confirmAnalysisChange = () => {
    if (switchBusy) return false;
    if (question.trim() && !window.confirm("放弃尚未发送的问题？")) return false;
    return leaveKnowledge();
  };

  const showLiveKnowledge = (trace: LiveKnowledgeTrace) => {
    if (liveKnowledgeCloseTimer.current !== null) window.clearTimeout(liveKnowledgeCloseTimer.current);
    liveKnowledgeCloseTimer.current = null;
    const wasClosed = liveKnowledgeTraceRef.current === null;
    liveKnowledgeTraceRef.current = trace;
    setLiveKnowledgeTrace(trace);
    setLiveKnowledgeClosing(false);
    if (wasClosed) setLiveKnowledgeMinimized(false);
  };

  const hideLiveKnowledge = useCallback((immediate = false) => {
    if (immediate) {
      if (liveKnowledgeCloseTimer.current !== null) window.clearTimeout(liveKnowledgeCloseTimer.current);
      liveKnowledgeCloseTimer.current = null;
      liveKnowledgeTraceRef.current = null;
      setLiveKnowledgeTrace(null);
      setLiveKnowledgeClosing(false);
      setLiveKnowledgeMinimized(false);
      return;
    }
    if (liveKnowledgeCloseTimer.current !== null) return;
    if (!liveKnowledgeTraceRef.current) return;
    setLiveKnowledgeClosing(true);
    liveKnowledgeCloseTimer.current = window.setTimeout(() => {
      liveKnowledgeCloseTimer.current = null;
      liveKnowledgeTraceRef.current = null;
      setLiveKnowledgeTrace(null);
      setLiveKnowledgeClosing(false);
      setLiveKnowledgeMinimized(false);
    }, 320);
  }, []);

  useEffect(() => () => {
    if (liveKnowledgeCloseTimer.current !== null) window.clearTimeout(liveKnowledgeCloseTimer.current);
  }, []);

  const loadState = async (preferredId?: string, requestUser = devUser) => {
    const generation = scopeGeneration.current;
    const next = await api<ApiState>("/api/page_configuration", undefined, requestUser);
    if (generation !== scopeGeneration.current) return next;
    setState(next); setModel(next.model);
    const preferred = next.profiles.find((profile) => profile.id === preferredId);
    setForm({ ...(preferred || next.active), password: "" });
    return next;
  };
  const showConversationList = (response: ConversationListResponse) => {
    setConversationHistory(filterConversationScope(response.conversations, historyScopeRef.current).map(conversationSummary));
    setHistorySources(response.data_sources);
    setLegacyCount(response.legacy_count);
  };
  const loadConversations = async (requestUser = devUser, scope = historyScopeRef.current) => {
    const generation = scopeGeneration.current;
    const requestGeneration = ++historyRequestGeneration.current;
    const response = await api<ConversationListResponse>(conversationListPath(scope), { cache: "no-store" }, requestUser);
    if (generation === scopeGeneration.current && requestGeneration === historyRequestGeneration.current) showConversationList(response);
    return response;
  };
  const resetAnalysis = useCallback((binding: ConversationBinding | null) => {
    setThreadId(crypto.randomUUID());
    setMessages([]);
    setThreadBinding(binding);
    setConversationCanContinue(Boolean(binding));
    setContextUsage(null);
    setQuestion("");
    setCurrentRound(null);
    setRoundStatus("");
    setConversationMenu(null);
    setKnowledgeMenuOpen(false);
    hideLiveKnowledge(true);
  }, [hideLiveKnowledge]);
  const showScope = (scope: string) => {
    historyScopeRef.current = scope;
    setHistoryScope(scope);
  };
  const showConfiguration = (next: ApiState, preferredId?: string) => {
    setState(next);
    setModel(next.model);
    setForm({ ...(next.profiles.find((profile) => profile.id === preferredId) || next.active), password: "" });
    setConfigurationReady(true);
  };
  const showSavedConversation = (saved: ConversationDetailPayload) => {
    setThreadId(saved.thread_id);
    setMessages(conversationMessages(saved));
    setThreadBinding(saved.binding);
    setConversationCanContinue(saved.can_continue);
    const latestAnswer = [...saved.messages].reverse().find((message) => message.role === "assistant");
    setContextUsage(latestAnswer?.details?.context_usage ?? null);
    setQuestion("");
    setCurrentRound(null);
    setRoundStatus("");
    setKnowledgeMenuOpen(false);
    hideLiveKnowledge(true);
  };
  useEffect(() => {
    const generation = ++scopeGeneration.current;
    Promise.all([
      fetchJsonForUser<AccountsResponse>("/api/accounts", DEFAULT_DEV_USER, { cache: "no-store" }),
      fetchJsonForUser<ApiState>("/api/page_configuration", DEFAULT_DEV_USER),
      fetchJsonForUser<ConversationListResponse>("/api/conversations", DEFAULT_DEV_USER, { cache: "no-store" }),
    ])
      .then(([accountResponse, nextState, conversationResponse]) => {
        if (generation !== scopeGeneration.current) return;
        setAccounts(accountResponse.accounts);
        setDevUser(accountResponse.current.login_id);
        showConfiguration(nextState);
        showScope(nextState.active.binding?.data_source_id || "");
        showConversationList(conversationResponse);
        resetAnalysis(nextState.active.binding || null);
      })
      .catch((error) => { if (generation === scopeGeneration.current) setNotice({ tone: "error", text: error.message }); })
      .finally(() => { if (generation === scopeGeneration.current) setConversationBusy(false); });
    return () => { scopeGeneration.current += 1; };
  }, [resetAnalysis]);
  const update = <K extends keyof Profile>(key: K, value: Profile[K]) => setForm((current) => ({ ...current, [key]: value }));

  const runAction = async (name: string, action: () => Promise<{ message?: string; details?: unknown }>) => {
    setBusy(name);
    try { const result = await action(); setNotice({ tone: "success", text: result.message || "操作成功" }); return result; }
    catch (error) { setNotice({ tone: "error", text: error instanceof Error ? error.message : "操作失败" }); return null; }
    finally { setBusy(""); }
  };

  const sendQuestion = async (suggestedText?: string) => {
    const text = (suggestedText ?? question).trim();
    if (!text || switchBusy || !canChat || !resourcesReady || !canContinue || !threadBinding) return;
    setConversationHistory((current) => {
      if (current.some((conversation) => conversation.threadId === threadId)) return current;
      const oneLineQuestion = text.replace(/\s+/g, " ");
      const title = `${oneLineQuestion.slice(0, 18)}${oneLineQuestion.length > 18 ? "…" : ""}`;
      return [{ threadId, title, customTitle: false, binding: threadBinding }, ...current];
    });
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", content: text }]);
    hideLiveKnowledge(true); setQuestion(""); setChatBusy(true); setStopRequested(false); setCurrentRound(null); setRoundStatus("正在分析现有信息并决定下一步…"); setActiveRequestId(null); activeRequestIdRef.current = null;
    try {
      const response = await fetchForUser("/api/chat/stream", devUser, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, thread_id: threadId, model, data_source_id: threadBinding.data_source_id, knowledge_base_id: threadBinding.knowledge_base_id }) });
      if (!response.ok) {
        if (response.status === 409) setConversationCanContinue(false);
        const payload = await response.json();
        throw new Error(payload.detail || "请求失败");
      }
      let finalReceived = false;
      await readJsonLines(response, (event) => {
        if (event.type === "started") {
          activeRequestIdRef.current = event.request_id;
          setActiveRequestId(event.request_id);
          if (event.thread_id) setThreadId(event.thread_id);
          setRoundStatus("正在分析现有信息并决定下一步…");
        } else if (event.type === "round") {
          setContextUsage(event.context_usage ?? null);
          setCurrentRound({ number: event.round || 1, content: event.content || "", toolCalls: event.tool_calls || [] });
          setRoundStatus(event.message || "本轮模型输出已生成。");
        } else if (event.type === "progress" && event.message) {
          setRoundStatus(event.message);
        } else if (event.type === "knowledge_trace") {
          if (event.action === "open") {
            showLiveKnowledge({
              stage: event.stage,
              mode: event.mode,
              message: event.message,
              activeIds: event.active_ids || [],
            });
          } else {
            hideLiveKnowledge();
          }
        } else if (event.type === "final" && event.response) {
          hideLiveKnowledge();
          finalReceived = true;
          setContextUsage(event.response.context_usage ?? null);
          setThreadId(event.response.thread_id);
          setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: event.response!.answer || (event.response!.artifacts?.length ? "报告已生成。" : "分析完成，但没有生成可展示的回答。"), details: event.response }]);
        } else if (event.type === "error") {
          hideLiveKnowledge();
          throw new Error(event.message || "分析执行失败。");
        }
      });
      if (!finalReceived) throw new Error("响应流已结束，但没有收到最终结果。");
    } catch (error) {
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: error instanceof Error ? error.message : "分析执行失败。" }]);
    } finally {
      hideLiveKnowledge(); activeRequestIdRef.current = null; setActiveRequestId(null); setStopRequested(false); setCurrentRound(null); setRoundStatus(""); setChatBusy(false);
      void loadConversations().catch((error) => console.error("会话列表刷新失败", error));
    }
  };

  const stopRun = async () => {
    const requestId = activeRequestIdRef.current;
    if (!requestId || stopRequested) return;
    setStopRequested(true);
    setRoundStatus("已请求停止，正在等待当前步骤安全结束…");
    try {
      const result = await api<{ status: string; message: string }>(`/api/runs/${requestId}/cancel`, { method: "POST" });
      if (result.status === "not_running") setNotice({ tone: "info", text: result.message });
    } catch (error) {
      setStopRequested(false);
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "停止任务失败。" });
    }
  };

  const newConversation = () => {
    if (!sourceIsActive || !configurationReady || !confirmAnalysisChange()) return;
    scopeGeneration.current += 1;
    resetAnalysis(state?.active.binding || null);
    setPage("analysis");
  };
  const applyProfile = async (profile: Profile, resumeThreadId?: string) => {
    if (!profile.binding || !canConfigure || !confirmAnalysisChange()) return;
    const generation = ++scopeGeneration.current;
    setBusy("source-switch");
    setConversationBusy(true);
    setKnowledgeMenuOpen(false);
    let applied = false;
    try {
      await api("/api/apply-profile", { method: "POST", body: JSON.stringify({ profile_id: profile.id }) });
      applied = true;
      const next = await api<ApiState>("/api/page_configuration", { cache: "no-store" });
      const scope = next.active.binding?.data_source_id || "";
      const [list, saved] = await Promise.all([
        api<ConversationListResponse>(conversationListPath(scope), { cache: "no-store" }),
        resumeThreadId ? api<ConversationDetailPayload>(`/api/conversations/${encodeURIComponent(resumeThreadId)}`, { cache: "no-store" }) : Promise.resolve(null),
      ]);
      if (generation !== scopeGeneration.current) return;
      showConfiguration(next);
      showScope(scope);
      showConversationList(list);
      resetAnalysis(next.active.binding || null);
      if (saved) showSavedConversation(saved);
      setRuntimeRevision((current) => current + 1);
      setPage("analysis");
      setNotice(null);
    } catch (error) {
      if (generation !== scopeGeneration.current) return;
      if (applied) setConfigurationReady(false);
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "切换数据源失败。" });
    } finally {
      if (generation === scopeGeneration.current) { setBusy(""); setConversationBusy(false); }
    }
  };
  const switchHistorySource = async (scope: string) => {
    if (scope === historyScope || switchBusy) return;
    const profile = state?.profiles.find((item) => item.binding?.data_source_id === scope);
    if (scope !== state?.active.binding?.data_source_id && profile && canConfigure) {
      await applyProfile(profile);
      return;
    }
    if (!confirmAnalysisChange()) return;
    const generation = ++scopeGeneration.current;
    setConversationBusy(true);
    try {
      const list = await api<ConversationListResponse>(conversationListPath(scope), { cache: "no-store" });
      if (generation !== scopeGeneration.current) return;
      showScope(scope);
      showConversationList(list);
      resetAnalysis(scope === state?.active.binding?.data_source_id ? state?.active.binding || null : null);
      setPage("analysis");
    } catch (error) {
      if (generation === scopeGeneration.current) setNotice({ tone: "error", text: error instanceof Error ? error.message : "读取会话失败。" });
    } finally {
      if (generation === scopeGeneration.current) setConversationBusy(false);
    }
  };
  const switchDevAccount = async (account: Account) => {
    if (account.login_id === devUser || !confirmAnalysisChange()) return;
    const generation = ++scopeGeneration.current;
    setAccountMenuOpen(false);
    setConversationBusy(true);
    setBusy("account-switch");
    try {
      const [next, list] = await Promise.all([
        api<ApiState>("/api/page_configuration", { cache: "no-store" }, account.login_id),
        api<ConversationListResponse>("/api/conversations", { cache: "no-store" }, account.login_id),
      ]);
      if (generation !== scopeGeneration.current) return;
      setDevUser(account.login_id);
      showConfiguration(next);
      showScope(next.active.binding?.data_source_id || "");
      showConversationList(list);
      resetAnalysis(next.active.binding || null);
      setPage("analysis");
      setNotice(null);
      setRuntimeRevision((current) => current + 1);
    } catch (error) {
      if (generation === scopeGeneration.current) setNotice({ tone: "error", text: error instanceof Error ? error.message : "切换账号失败。" });
    } finally {
      if (generation === scopeGeneration.current) { setBusy(""); setConversationBusy(false); }
    }
  };
  const openConversation = async (conversation: ConversationHistoryItem) => {
    if (!confirmAnalysisChange()) return;
    setConversationMenu(null);
    setPage("analysis");
    if (conversation.threadId === threadId) return;
    const generation = ++scopeGeneration.current;
    setConversationBusy(true);
    try {
      const savedConversation = await api<ConversationDetailPayload>(`/api/conversations/${encodeURIComponent(conversation.threadId)}`, { cache: "no-store" });
      if (generation !== scopeGeneration.current) return;
      showSavedConversation(savedConversation);
    } catch (error) {
      if (generation === scopeGeneration.current) setNotice({ tone: "error", text: error instanceof Error ? error.message : "读取会话失败。" });
    } finally {
      if (generation === scopeGeneration.current) setConversationBusy(false);
    }
  };
  const renameConversation = async (conversation: ConversationHistoryItem) => {
    if (switchBusy) return;
    const nextTitle = window.prompt("重命名分析", conversation.title)?.trim();
    if (!nextTitle) return;
    setConversationMenu(null);
    setConversationBusy(true);
    try {
      const response = await api<RenameConversationResponse>(`/api/conversations/${encodeURIComponent(conversation.threadId)}`, { method: "PATCH", body: JSON.stringify({ title: nextTitle }) });
      const renamedConversation = conversationSummary(response.conversation);
      setConversationHistory((current) => current.map((item) => item.threadId === conversation.threadId ? renamedConversation : item));
    } catch (error) {
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "重命名失败。" });
    } finally {
      setConversationBusy(false);
    }
  };
  const deleteConversation = async (conversation: ConversationHistoryItem) => {
    if (switchBusy) return;
    setConversationMenu(null);
    setConversationBusy(true);
    try {
      await api(`/api/conversations/${encodeURIComponent(conversation.threadId)}`, { method: "DELETE" });
      setConversationHistory((current) => current.filter((item) => item.threadId !== conversation.threadId));
      if (conversation.threadId === threadId) {
        resetAnalysis(sourceIsActive ? state?.active.binding || null : null);
      }
      await loadConversations();
    } catch (error) {
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "删除会话失败。" });
    } finally {
      setConversationBusy(false);
    }
  };
  const saveAndApply = async () => {
    if (!canConfigure || !confirmAnalysisChange()) return;
    const generation = ++scopeGeneration.current;
    setBusy("save");
    setConversationBusy(true);
    let applied = false;
    try {
      await api("/api/save-and-apply", { method: "POST", body: JSON.stringify(form) });
      applied = true;
      const next = await api<ApiState>("/api/page_configuration", { cache: "no-store" });
      const scope = next.active.binding?.data_source_id || "";
      const list = await api<ConversationListResponse>(conversationListPath(scope), { cache: "no-store" });
      if (generation !== scopeGeneration.current) return;
      showConfiguration(next, form.id);
      showScope(scope);
      showConversationList(list);
      resetAnalysis(next.active.binding || null);
      setRuntimeRevision((current) => current + 1);
      setPage("analysis");
      setNotice({ tone: "success", text: "配置已应用" });
    } catch (error) {
      if (generation !== scopeGeneration.current) return;
      if (applied) setConfigurationReady(false);
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "保存配置失败。" });
    } finally {
      if (generation === scopeGeneration.current) { setBusy(""); setConversationBusy(false); }
    }
  };
  const refreshKnowledgeAfterSave = async () => {
    setRuntimeRevision((current) => current + 1);
    const generation = scopeGeneration.current;
    try {
      const next = await loadState();
      if (generation !== scopeGeneration.current) return;
      if (!sameConversationBinding(next.active.binding, state?.active.binding)) {
        scopeGeneration.current += 1;
        showScope(next.active.binding?.data_source_id || "");
        resetAnalysis(next.active.binding || null);
        await loadConversations();
      }
    } catch (error) {
      if (generation === scopeGeneration.current) setNotice({ tone: "error", text: error instanceof Error ? error.message : "刷新知识库失败。" });
    }
  };
  const deleteProfile = async () => {
    if (!canConfigure || !selectedProfile || selectedProfile.id === state?.active.id) return;
    if (!window.confirm(`删除数据源配置“${selectedProfile.label}”？`)) return;
    const result = await runAction("delete-profile", () => api(`/api/profiles/${encodeURIComponent(selectedProfile.id)}`, { method: "DELETE" }));
    if (result) await loadState();
  };
  const saveModelSettings = async () => { if (!canConfigure) return; const result = await runAction("model", () => api("/api/model-settings", { method: "POST", body: JSON.stringify({ model, api_key: modelApiKey }) })); if (result) { setModelApiKey(""); await loadState(form.id); } };
  const uploadKnowledge = async (file?: File) => {
    if (!file || !canConfigure) return;
    setBusy("upload");
    try {
      const response = await fetchForUser("/api/import-knowledge", devUser, { method: "POST", headers: { "Content-Type": "application/zip", "X-Knowledge-Name": encodeURIComponent(file.name.replace(/\.zip$/i, "")) }, body: file });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "导入失败");
      update("knowledge_root", payload.details.path); setNotice({ tone: "success", text: payload.message });
    } catch (error) { setNotice({ tone: "error", text: error instanceof Error ? error.message : "导入失败" }); }
    finally { setBusy(""); if (fileInput.current) fileInput.current.value = ""; }
  };

  if (!state) return <main className="loading-shell">正在加载 DataAgent…</main>;

  return <div className="product-shell">
    <aside className="main-nav">
      <div className="brand"><strong>DataAgent</strong></div>
      <nav aria-label="主导航">
        <button className="new-analysis" title={chatUnavailableReason || (!sourceIsActive ? "选择可用数据源后开始分析" : "开始新的数据分析")} disabled={switchBusy || !canChat || !resourcesReady || !sourceIsActive || !configurationReady} onClick={newConversation}>
          <span aria-hidden="true" className="nav-icon nav-icon-new-analysis" />
          <span className="nav-label">开始新分析</span>
        </button>
        <button className={page === "database" ? "active" : ""} onClick={() => navigatePage("database")}>
          <span aria-hidden="true" className="nav-icon nav-icon-database" />
          <span className="nav-label">数据源</span>
        </button>
        <button className={page === "knowledge" ? "active" : ""} onClick={() => navigatePage("knowledge")}>
          <span aria-hidden="true" className="nav-icon nav-icon-knowledge" />
          <span className="nav-label">知识库</span>
        </button>
        <button className={page === "model" ? "active" : ""} onClick={() => navigatePage("model")}>
          <span aria-hidden="true" className="nav-icon nav-icon-model" />
          <span className="nav-label">模型</span>
        </button>
      </nav>
      <div className="sidebar-source"><DatasourceConversationPicker sources={dataSources} value={historyScope} legacyCount={legacyCount} disabled={switchBusy} onChange={switchHistorySource} /></div>
      <div className="conversation-history">
        <div className="conversation-section-label">最近</div>
        {conversationHistory.map((conversation) => <div className={`conversation-item ${conversation.threadId === threadId ? "active" : ""}`} onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setConversationMenu(null); }} key={conversation.threadId}>
          <button type="button" className="conversation-open" title={conversation.title} disabled={switchBusy} onClick={() => openConversation(conversation)}><span>{conversation.title}</span>{showKnowledgeLabels && conversation.binding && <small>{conversation.binding.knowledge_base_name}</small>}</button>
          <button type="button" className="conversation-more" disabled={chatBusy || conversationBusy || !canWriteConversations} aria-label={`管理“${conversation.title}”`} onClick={() => setConversationMenu((current) => current === conversation.threadId ? null : conversation.threadId)}><svg aria-hidden="true" viewBox="0 0 18 6"><circle cx="3" cy="3" r="1.5" /><circle cx="9" cy="3" r="1.5" /><circle cx="15" cy="3" r="1.5" /></svg></button>
          {conversationMenu === conversation.threadId && <div className="conversation-menu" role="menu">
            <button type="button" role="menuitem" onClick={() => renameConversation(conversation)}><svg aria-hidden="true" viewBox="0 0 18 18"><path d="m11.5 4.5 2 2M4 14l3-.7 7.4-7.4a1.4 1.4 0 0 0-2-2L5 11.3Z" /></svg><span>重命名</span></button>
            <button type="button" role="menuitem" className="delete" onClick={() => deleteConversation(conversation)}><svg aria-hidden="true" viewBox="0 0 18 18"><path d="M3 5h12M7 5V3.5h4V5M5.5 5l.7 10h5.6l.7-10M8 8v4M10 8v4" /></svg><span>删除</span></button>
          </div>}
        </div>)}
        {!conversationBusy && !conversationHistory.length && <div className="conversation-empty">暂无会话</div>}
      </div>
      <div className="account-panel" onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setAccountMenuOpen(false); }}>
        {accountMenuOpen && <div className="account-menu" role="menu" aria-label="切换模拟账号">
          <div className="account-menu-label">模拟账号</div>
          {accounts.map((account) => <button type="button" role="menuitem" className={account.login_id === devUser ? "active" : ""} key={account.login_id} onClick={() => switchDevAccount(account)}>
            <span className="account-option-avatar" aria-hidden="true">{account.avatar}</span>
            <span className="account-option-copy"><strong>{account.display_name}</strong><small>{account.role_label} · {account.workspace_name}</small></span>
            <span className="account-option-check" aria-hidden="true">{account.login_id === devUser ? "✓" : ""}</span>
          </button>)}
        </div>}
        <button className="account-shell" type="button" aria-label="当前账号" aria-haspopup="menu" aria-expanded={accountMenuOpen} disabled={chatBusy || conversationBusy || Boolean(busy)} onClick={() => setAccountMenuOpen((current) => !current)}>
          <span className="account-avatar-wrap">
            <span className="account-avatar" aria-hidden="true">{state.workspace.avatar}</span>
            <span className={`account-online ${state.workspace.resources_ready && state.model_configured ? "" : "warning"}`} title={state.workspace.resources_ready ? "工作空间资源已连接" : "工作空间资源尚未配置"} />
          </span>
          <span className="account-copy"><strong>{state.workspace.display_name}</strong><span>{state.workspace.role_label} · {state.workspace.workspace_name}</span></span>
          <svg aria-hidden="true" className="account-more" viewBox="0 0 18 6"><circle cx="3" cy="3" r="1.4" /><circle cx="9" cy="3" r="1.4" /><circle cx="15" cy="3" r="1.4" /></svg>
        </button>
      </div>
    </aside>

    <main className="main-area"><header className="app-header"><div className="app-header-title"><h1>{page === "analysis" ? historyScope === LEGACY_SCOPE ? "未归属会话" : threadBinding ? `${backendName(threadBinding.backend)} · ${threadBinding.database || threadBinding.data_source_name}` : displayedSource ? `${backendName(displayedSource.backend)} · ${displayedSource.database || displayedSource.data_source_name}` : "数据分析" : pageNames[page]}</h1>{page !== "analysis" && <p>{backendName(state.active.backend)} · {state.active.database || "本地文件"}</p>}</div>
      {page === "analysis" && <div className="analysis-context-controls">
        <div className="mobile-source"><DatasourceConversationPicker sources={dataSources} value={historyScope} legacyCount={legacyCount} disabled={switchBusy} onChange={switchHistorySource} /></div>
        <div className="conversation-knowledge" role="group" aria-label="会话知识库" onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setKnowledgeMenuOpen(false); }}>
          <button type="button" className="knowledge-context-trigger" aria-haspopup="menu" aria-expanded={knowledgeMenuOpen} disabled={switchBusy} title={threadBinding?.knowledge_base_name || "未绑定知识库"} onClick={() => setKnowledgeMenuOpen((current) => !current)}><svg aria-hidden="true" viewBox="0 0 20 20"><path d="M10 5C7 3 4 3 2 4v12c2-1 5-1 8 1 3-2 6-2 8-1V4c-2-1-5-1-8 1Zm0 0v12" /></svg><span>知识库</span><svg aria-hidden="true" viewBox="0 0 16 16"><path d="m4 6 4 4 4-4" /></svg></button>
          {knowledgeMenuOpen && <div className="knowledge-context-menu" role="menu" tabIndex={-1} aria-label="会话知识库" onKeyDown={(event) => { if (event.key === "Escape") { setKnowledgeMenuOpen(false); event.currentTarget.parentElement?.querySelector<HTMLButtonElement>(".knowledge-context-trigger")?.focus(); } }}>
            <div className="knowledge-context-current"><strong>{threadBinding?.knowledge_base_name || "未绑定知识库"}</strong>{threadBinding && <span title={threadBinding.knowledge_root}>{threadBinding.knowledge_root}</span>}</div>
            {knowledgeChoices.map((profile) => <button key={profile.id} role="menuitem" type="button" disabled={!canConfigure || sameConversationBinding(profile.binding, threadBinding) && canContinue} onClick={() => applyProfile(profile)}><span>{profile.binding!.knowledge_base_name}</span>{sameConversationBinding(profile.binding, threadBinding) && <span aria-hidden="true">✓</span>}</button>)}
          </div>}
        </div>
      </div>}
    </header>

    {page === "analysis" && <section className="analysis-page"><div className="conversation">
      {messages.length === 0 ? <div className="empty-state">{chatUnavailableReason && <h2>{chatUnavailableReason}</h2>}</div> : messages.map((message) => <article className={`message ${message.role}`} key={message.id}>
        {message.role === "assistant" ? <><AnswerBody content={message.content} />{message.details?.artifacts?.map((artifact) => <ArtifactPreview artifact={artifact} key={artifact.id} />)}</> : <p>{message.content}</p>}
        {message.details && <details className="run-details"><summary>查看 SQL 与运行信息</summary><div className="metric-row"><span>{(message.details.latency_ms / 1000).toFixed(1)} 秒</span><span>{message.details.sql_queries.length} 次 SQL</span><span>{message.details.knowledge_view?.knowledge_view_mode || "-"} View</span>{message.details.status === "paused" && <span className="warning-text">已暂停</span>}</div>{message.details.sql_queries.map((query, queryIndex) => <div className="sql-card" key={query.tool_call_id || queryIndex}><div>SQL {queryIndex + 1}</div><pre><code>{query.sql}</code></pre><ResultTable result={query.result} /></div>)}</details>}
      </article>)}{chatBusy && <article className="message assistant pending"><div className="round-status"><span className="round-status-dot" /><span className="round-status-text">{roundStatus || "正在分析现有信息并决定下一步…"}</span></div>{currentRound && <div className="current-round">{currentRound.content && <AnswerBody content={currentRound.content} />}{currentRound.toolCalls.length > 0 && <div className="round-tools">{currentRound.toolCalls.map((call, index) => <ToolCallCard call={call} key={`${call.name}-${index}`} />)}</div>}</div>}{stopRequested && <small>停止将在当前模型或工具调用结束后的安全位置生效。</small>}</article>}
    </div>{liveKnowledgeTrace && !liveKnowledgeMinimized && <div className={`live-knowledge-overlay ${liveKnowledgeClosing ? "closing" : ""}`} aria-live="polite"><div className="live-knowledge-stage"><button className="live-knowledge-minimize" type="button" onClick={() => setLiveKnowledgeMinimized(true)} aria-label="收起知识库导航">×</button><KnowledgeGraph key={`live-knowledge-${devUser}`} revision={runtimeRevision} devUser={devUser} live liveTrace={liveKnowledgeTrace} /></div></div>}{liveKnowledgeTrace && liveKnowledgeMinimized && <button className="live-knowledge-reopen" type="button" onClick={() => setLiveKnowledgeMinimized(false)}><span />查看知识库导航</button>}
    {!canContinue && !chatBusy ? <div className="conversation-readonly" role="status"><span>{readOnlyReason}</span>{originalProfile && canConfigure && <button type="button" disabled={switchBusy} onClick={() => applyProfile(originalProfile, messages.length ? threadId : undefined)}>使用原配置</button>}</div> : <div className="composer" title={chatUnavailableReason}>
      <textarea value={question} disabled={conversationBusy || Boolean(busy) || !canChat || !resourcesReady || !canContinue} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendQuestion(); } }} placeholder={chatUnavailableReason || "询问 DataAgent"} />
      <ContextWindowMeter usage={conversationBusy ? null : contextUsage} model={model} capacity={state.model_context_windows?.[model]} />
      <button className={`send-button ${chatBusy ? "stop-button" : ""}`} aria-label={chatBusy ? "停止分析" : "发送"} disabled={conversationBusy || Boolean(busy) || !canChat || !resourcesReady || (chatBusy ? !activeRequestId || stopRequested : !question.trim() || !canContinue)} onClick={chatBusy ? stopRun : () => sendQuestion()}>{chatBusy ? <span className="stop-symbol" /> : <svg aria-hidden="true" viewBox="0 0 20 20"><path d="M10 15V5M6 9l4-4 4 4" /></svg>}</button>
    </div>}</section>}

    {page === "database" && <section className="content-page database-page"><aside className="profile-panel"><div className="panel-heading"><h2>配置方案</h2><button disabled={!canConfigure} onClick={() => setForm(emptyProfile())}>新建</button></div>{state.profiles.map((profile) => <button className={`profile-item ${profile.id === form.id ? "active" : ""}`} key={profile.id} onClick={() => setForm({ ...profile, password: "" })}><strong>{profile.label}</strong><span>{backendName(profile.backend)} · {profile.database || "本地文件"}</span></button>)}</aside>
      <div className="database-config-column"><div className="settings-panel"><div className="panel-heading"><div><h2>{selectedProfile ? "编辑数据源" : "新数据源"}</h2><p>账号应只具备读取权限。</p></div><span className="apply-note">保存后立即生效</span></div><div className="form-grid">
        <label><span>方案名称</span><input value={form.label} onChange={(event) => update("label", event.target.value)} /></label><label><span>方案 ID</span><input disabled={Boolean(selectedProfile)} value={form.id} onChange={(event) => update("id", event.target.value.toLowerCase())} /></label>
        <label className="wide"><span>数据库类型</span><select value={form.backend} onChange={(event) => { const backend = event.target.value as Backend; setForm((current) => ({ ...current, backend, port: backend === "mysql" ? 3306 : backend === "postgresql" ? 5432 : 0 })); }}><option value="postgresql">PostgreSQL</option><option value="mysql">MySQL</option><option value="duckdb">DuckDB</option></select></label>
        {form.backend === "duckdb" ? <label className="wide"><span>DuckDB 文件</span><input value={form.duckdb_path} onChange={(event) => update("duckdb_path", event.target.value)} /></label> : <><label><span>Host</span><input value={form.host} onChange={(event) => update("host", event.target.value)} placeholder="host.docker.internal" /></label><label><span>Port</span><input type="number" value={form.port || ""} onChange={(event) => update("port", Number(event.target.value))} /></label><label><span>只读用户名</span><input value={form.username} onChange={(event) => update("username", event.target.value)} /></label><label><span>密码 {form.password_saved ? "（已保存）" : ""}</span><input type="password" value={form.password || ""} onChange={(event) => update("password", event.target.value)} placeholder={form.password_saved ? "留空继续使用" : ""} /></label><label className="wide"><span>数据库名称</span><input value={form.database} onChange={(event) => update("database", event.target.value)} placeholder="例如：cold_chain_pharma_compliance" /><small>填写 PostgreSQL 或 MySQL 中实际存在的数据库名称。</small></label></>}
      </div><div className="form-actions">{selectedProfile && <button className="button danger" title={selectedProfile.id === state.active.id ? "当前生效配置不能删除" : "删除数据源配置"} disabled={Boolean(busy) || !canConfigure || selectedProfile.id === state.active.id} onClick={deleteProfile}>删除配置</button>}<button className="button secondary" disabled={Boolean(busy) || !canConfigure} onClick={() => runAction("test", () => api("/api/test-database", { method: "POST", body: JSON.stringify(form) }))}>测试连接</button><button className="button primary" disabled={Boolean(busy) || !canConfigure} onClick={saveAndApply}>保存并应用</button></div></div><div className="connection-overview"><div><span>数据库</span><strong>{backendName(form.backend)}</strong></div><div><span>连接地址</span><strong>{form.backend === "duckdb" ? "本地文件" : `${form.host || "-"}:${form.port || "-"}`}</strong></div><div><span>访问账号</span><strong>{form.username || "-"}</strong></div><div><span>访问模式</span><strong>只读事务</strong></div></div></div>
      <DatabaseExplorer key={`database-${devUser}`} revision={runtimeRevision} devUser={devUser} />
    </section>}

    {page === "knowledge" && <KnowledgeWorkspace key={`knowledge-${devUser}`} revision={runtimeRevision} devUser={devUser} canEdit={canConfigure} resourcesReady={resourcesReady} databaseName={state.active.database} onDirtyChange={setKnowledgeDirty} onSavingChange={setKnowledgeSaving} onSaved={() => void refreshKnowledgeAfterSave()} settings={<>
      <div className="settings-panel"><div className="panel-heading"><h2>知识库路径</h2></div><label><span>目录</span><input value={form.knowledge_root} disabled={!canConfigure} onChange={(event) => update("knowledge_root", event.target.value)} /></label><div className="form-actions"><button className="button secondary" disabled={Boolean(busy) || !canConfigure} onClick={() => runAction("validate", () => api("/api/validate-knowledge", { method: "POST", body: JSON.stringify({ knowledge_root: form.knowledge_root }) }))}>验证</button><button className="button primary" disabled={Boolean(busy) || !canConfigure} onClick={saveAndApply}>保存并应用</button></div></div>
      <div className="settings-panel"><div className="panel-heading"><h2>导入知识库</h2></div><input ref={fileInput} type="file" accept=".zip,application/zip" hidden onChange={(event) => uploadKnowledge(event.target.files?.[0])} /><button className="upload-area" disabled={busy === "upload" || !canConfigure} onClick={() => fileInput.current?.click()}><strong>{busy === "upload" ? "正在校验…" : "选择 ZIP 文件"}</strong></button></div>
      <p className="kw-settings-total">{state.knowledge.card_count ?? 0} 项知识</p>
      {state.knowledge.types && <div className="type-list">{Object.entries(state.knowledge.types).map(([type, count]) => <div key={type}><span>{({ column: "字段", database: "数据库", glossary_term: "术语", metric: "指标", relationship: "关系", table: "数据表" } as Record<string, string>)[type] || type}</span><strong>{count}</strong></div>)}</div>}
    </>} />}

    {page === "model" && <section className="content-page model-page"><div className="settings-panel"><div className="panel-heading"><div><h2>DeepSeek 模型</h2><p>当前仅支持 DeepSeek V4 Pro 与 DeepSeek V4 Flash。</p></div><span className={`config-status ${state.model_configured ? "configured" : ""}`}>{state.model_configured ? "API Key 已配置" : "API Key 未配置"}</span></div><div className="form-grid">
      <label className="wide"><span>模型</span><select value={model} onChange={(event) => setModel(event.target.value as Model)}>{state.models.map((item) => <option key={item} value={item}>{modelName(item)}</option>)}</select></label>
      <label className="wide"><span>模型 API Key</span><input type="password" autoComplete="new-password" value={modelApiKey} onChange={(event) => setModelApiKey(event.target.value)} placeholder={state.model_configured ? "留空继续使用已保存的 API Key" : "输入 DeepSeek API Key"} /><small>密钥只保存在本机，页面不会读取或回显完整内容。</small></label>
    </div><div className="form-actions"><button className="button primary" disabled={Boolean(busy) || !canConfigure} onClick={saveModelSettings}>保存模型配置</button></div></div></section>}

    {notice && <div className={`toast ${notice.tone}`}><span>{notice.text}</span><button onClick={() => setNotice(null)}>×</button></div>}
  </main></div>;
}
