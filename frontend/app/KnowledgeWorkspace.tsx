"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import KnowledgeGraph, { type GraphNode, type GraphPayload } from "./KnowledgeGraph";
import { fetchJsonForUser } from "./api-client";
import "./knowledge-workspace.css";

type Relation = { source_id: string; target_id: string; relation: string };
type Evidence = { source_id?: string; locator?: string; result_summary?: string; evidence_type?: string };
type KnowledgeCard = {
  knowledge_id: string; knowledge_type: string; title: string; summary: string;
  database_id: string; revision: number; aliases: string[]; definition: string;
  source_file: string; evidence_refs: Evidence[]; relations: Relation[];
  status: string; payload: Record<string, unknown>;
};
type CardResponse = { card: KnowledgeCard; message?: string };
type CardDraft = { title: string; summary: string; aliases: string; definition: string; knowledge_type: "metric" | "glossary_term"; database_id: string };
type WorkspaceProps = {
  revision: number; devUser: string; canEdit: boolean; resourcesReady: boolean;
  databaseName: string; settings: ReactNode; onSaved: () => void;
  onDirtyChange: (dirty: boolean) => void;
  onSavingChange: (saving: boolean) => void;
};
type IconName = "close" | "add" | "edit" | "settings" | "book" | "database" | "chevron" | "link" | "arrow" | "check";
const typeLabels: Record<string, string> = { database: "数据库", table: "数据表", column: "字段", relationship: "关系", metric: "指标", glossary_term: "术语", query_recipe: "查询方案" };
const relationLabels: Record<string, string> = { related_to: "相关", sourced_from: "依据", requires: "依赖", contains: "包含", belongs_to: "归属", describes: "描述", default_time_field: "时间字段", applies_to: "适用于", from_table: "来源表", to_table: "目标表" };
const incomingLabels: Record<string, string> = { related_to: "相关", sourced_from: "作为依据", requires: "被依赖", contains: "归属", belongs_to: "包含", describes: "被描述", default_time_field: "作为时间字段", applies_to: "适用此项", from_table: "出发关系", to_table: "到达关系" };
const isSemantic = (type: string) => type === "metric" || type === "glossary_term";
const emptyDraft = (databaseId = ""): CardDraft => ({ title: "", summary: "", aliases: "", definition: "", knowledge_type: "glossary_term", database_id: databaseId });
const draftFrom = (card: KnowledgeCard): CardDraft => ({ title: card.title, summary: card.summary, aliases: card.aliases.join(", "), definition: card.definition || "", knowledge_type: card.knowledge_type === "metric" ? "metric" : "glossary_term", database_id: card.database_id });

function Icon({ name }: { name: IconName }) {
  const shapes: Record<IconName, ReactNode> = {
    close: <path d="m6 6 12 12M18 6 6 18" />,
    add: <path d="M12 5v14M5 12h14" />,
    edit: <><path d="m14 5 5 5M4 20l5-1 11-11a2.8 2.8 0 0 0-4-4L5 15Z" /><path d="M13 20h7" /></>,
    settings: <><path d="M4 7h6m4 0h6M4 17h10m4 0h2" /><circle cx="12" cy="7" r="2" /><circle cx="16" cy="17" r="2" /></>,
    book: <><path d="M12 5v15M3 4h5a4 4 0 0 1 4 2 4 4 0 0 1 4-2h5v15h-5a5 5 0 0 0-4 1 5 5 0 0 0-4-1H3Z" /></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" /></>,
    chevron: <path d="m9 5 7 7-7 7" />,
    link: <><path d="m10 13 4-4m-5 8-1 1a4 4 0 0 1-6-6l4-4a4 4 0 0 1 6 0m0 8a4 4 0 0 0 6 0l4-4a4 4 0 0 0-6-6l-1 1" /></>,
    arrow: <path d="M5 12h14m-6-6 6 6-6 6" />,
    check: <path d="m5 12 4 4L19 6" />,
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" className="kw-icon">{shapes[name]}</svg>;
}

function SettingsDialog({ open, close, children }: { open: boolean; close: () => void; children: ReactNode }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (open && !dialog.current?.open) dialog.current?.showModal();
    else if (!open && dialog.current?.open) dialog.current?.close();
  }, [open]);
  return <dialog ref={dialog} className="kw-settings-dialog" aria-labelledby="kw-settings-title" onCancel={close} onClose={close}>
    <header><h2 id="kw-settings-title">知识库设置</h2><button type="button" className="kw-icon-button" onClick={close} aria-label="关闭知识库设置"><Icon name="close" /></button></header>
    <div className="kw-settings-body">{children}</div>
  </dialog>;
}

function NewKnowledgeDialog({ children, close, saving }: { children: ReactNode; close: () => void; saving: boolean }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const drag = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const moveTo = (left: number, top: number) => {
    const element = dialog.current;
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    element.style.left = `${Math.max(8, Math.min(left, window.innerWidth - bounds.width - 8))}px`;
    element.style.top = `${Math.max(8, Math.min(top, window.innerHeight - bounds.height - 8))}px`;
    element.style.transform = "none";
  };
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    element.show();
    const graph = element.parentElement?.querySelector(".kw-graph-region")?.getBoundingClientRect();
    const bounds = element.getBoundingClientRect();
    moveTo(graph ? graph.left + (graph.width - bounds.width) / 2 : (window.innerWidth - bounds.width) / 2, Math.max(80, (window.innerHeight - bounds.height) / 2));
    element.querySelector<HTMLInputElement>('input[name="title"]')?.focus();
    const clamp = () => { const rect = element.getBoundingClientRect(); moveTo(rect.left, rect.top); };
    window.addEventListener("resize", clamp);
    return () => { window.removeEventListener("resize", clamp); element.close(); previousFocus?.focus(); };
  }, []);
  return <dialog ref={dialog} className="kw-new-dialog" aria-labelledby="kw-new-title" onKeyDown={(event) => { if (event.key === "Escape") { event.preventDefault(); if (!saving) close(); } }}>
    <header><button type="button" className="kw-drag-handle" aria-label="移动新增知识窗口" title="拖动标题栏移动；方向键微调位置"
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        const bounds = dialog.current!.getBoundingClientRect();
        drag.current = { x: event.clientX, y: event.clientY, left: bounds.left, top: bounds.top };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => { const start = drag.current; if (start) moveTo(start.left + event.clientX - start.x, start.top + event.clientY - start.y); }}
      onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
        event.preventDefault(); const bounds = dialog.current!.getBoundingClientRect();
        moveTo(bounds.left + (event.key === "ArrowRight" ? 16 : event.key === "ArrowLeft" ? -16 : 0), bounds.top + (event.key === "ArrowDown" ? 16 : event.key === "ArrowUp" ? -16 : 0));
      }}>
      <span id="kw-new-title">新增知识</span></button><button className="kw-icon-button" type="button" aria-label="关闭新增知识" disabled={saving} onClick={close}><Icon name="close" /></button>
    </header>
    {children}
  </dialog>;
}

function CardFields({ draft, setDraft, creating, semantic, databases }: { draft: CardDraft; setDraft: (next: CardDraft) => void; creating: boolean; semantic: boolean; databases: { id: string; name: string }[] }) {
  const update = (key: keyof CardDraft, value: string) => setDraft({ ...draft, [key]: value });
  return <>
    {creating && <div className="kw-form-pair"><label>类型<select name="knowledge_type" value={draft.knowledge_type} onChange={(event) => update("knowledge_type", event.target.value)}><option value="glossary_term">术语</option><option value="metric">指标</option></select></label><label>数据源<select name="database_id" value={draft.database_id} onChange={(event) => update("database_id", event.target.value)} required><option value="" disabled>选择数据源</option>{databases.map((database) => <option value={database.id} key={database.id}>{database.name}</option>)}</select></label></div>}
    <label>名称<input name="title" value={draft.title} onChange={(event) => update("title", event.target.value)} required maxLength={200} /></label>
    <label>别名<input name="aliases" value={draft.aliases} onChange={(event) => update("aliases", event.target.value)} title="多个别名以逗号分隔" maxLength={2000} /></label>
    <label>摘要<textarea name="summary" className="kw-summary-input" value={draft.summary} onChange={(event) => update("summary", event.target.value)} required maxLength={4000} /></label>
    {(creating || semantic) && <label>定义<textarea name="definition" className="kw-definition-input" value={draft.definition} onChange={(event) => update("definition", event.target.value)} required maxLength={20000} /></label>}
  </>;
}

export default function KnowledgeWorkspace({ revision, devUser, canEdit, resourcesReady, databaseName, settings, onSaved, onDirtyChange, onSavingChange }: WorkspaceProps) {
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [card, setCard] = useState<KnowledgeCard | null>(null);
  const [loadedKey, setLoadedKey] = useState("");
  const [loadFailure, setLoadFailure] = useState({ key: "", message: "" });
  const [error, setError] = useState("");
  const [saved, setSaved] = useState("");
  const [editing, setEditing] = useState(false);
  const [creating, setCreating] = useState(false);
  const [draft, setDraft] = useState<CardDraft>(emptyDraft);
  const [saving, setSaving] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [relationOpen, setRelationOpen] = useState(false);
  const [relationType, setRelationType] = useState("related_to");
  const [relationTarget, setRelationTarget] = useState("");
  const [targetQuery, setTargetQuery] = useState("");
  const [reload, setReload] = useState(0);
  const saveRef = useRef(false);
  const requestKey = `${devUser}:${selectedId}:${revision}:${reload}`;
  const loadError = loadFailure.key === requestKey ? loadFailure.message : "";
  const loading = Boolean(selectedId && !creating && loadedKey !== requestKey && !loadError);
  const baseline = card ? draftFrom(card) : emptyDraft(draft.database_id);
  const dirty = (creating && Boolean(draft.title || draft.summary || draft.aliases || draft.definition)) || (editing && JSON.stringify(draft) !== JSON.stringify(baseline)) || (relationOpen && Boolean(relationTarget));
  const titleById = useMemo(() => new Map(graph?.nodes.map((node) => [node.knowledge_id, node.title]) || []), [graph]);
  const databases = useMemo(() => {
    const known = (graph?.database_ids || []).map((id) => ({ id, name: (graph?.nodes || []).find((node) => node.knowledge_type === "database" && node.database_id === id)?.title || id }));
    if (card?.database_id && !known.some((database) => database.id === card.database_id)) known.push({ id: card.database_id, name: card.database_id });
    return known;
  }, [graph, card]);

  useEffect(() => { onDirtyChange(dirty); return () => onDirtyChange(false); }, [dirty, onDirtyChange]);
  useEffect(() => { onSavingChange(saving); return () => onSavingChange(false); }, [saving, onSavingChange]);
  useEffect(() => {
    if (!dirty) return;
    const beforeUnload = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [dirty]);
  useEffect(() => {
    if (!saved) return;
    const timer = window.setTimeout(() => setSaved(""), 3500);
    return () => window.clearTimeout(timer);
  }, [saved]);
  useEffect(() => {
    if (!selectedId || creating) return;
    const controller = new AbortController();
    fetchJsonForUser<CardResponse>(`/api/knowledge-cards/${encodeURIComponent(selectedId)}`, devUser, { cache: "no-store", signal: controller.signal })
      .then((result) => { if (!controller.signal.aborted) { setCard(result.card); setLoadedKey(requestKey); setLoadFailure({ key: requestKey, message: "" }); } })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setLoadFailure({ key: requestKey, message: reason instanceof Error ? reason.message : "读取知识失败" }); });
    return () => controller.abort();
  }, [selectedId, creating, requestKey, devUser]);

  const mayLeave = useCallback(() => !saveRef.current && (!dirty || window.confirm("放弃尚未保存的修改？")), [dirty]);
  const selectNode = useCallback((node: GraphNode | null) => {
    if (node?.knowledge_id === selectedId && !creating) return;
    if (!mayLeave()) return;
    setSelectedId(node?.knowledge_id || null); setCard(null); setEditing(false); setCreating(false);
    setRelationOpen(false); setRelationTarget(""); setError(""); setSaved("");
  }, [selectedId, creating, mayLeave]);
  const receiveGraph = useCallback((next: GraphPayload) => {
    setGraph(next);
  }, []);

  const createNode = () => {
    if (!canEdit || !resourcesReady || !mayLeave()) return;
    setDraft(emptyDraft(card?.database_id || databases[0]?.id || "")); setEditing(false); setCreating(true);
    setRelationOpen(false); setRelationTarget(""); setError(""); setSaved("");
  };
  const beginEdit = () => { if (card && canEdit && mayLeave()) { setDraft(draftFrom(card)); setCreating(false); setEditing(true); setRelationOpen(false); setRelationTarget(""); setError(""); } };
  const cancelEdit = () => { if (!mayLeave()) return; setEditing(false); setCreating(false); setRelationOpen(false); setRelationTarget(""); setError(""); };
  const reloadCard = () => { if (!mayLeave()) return; setEditing(false); setRelationOpen(false); setRelationTarget(""); setError(""); setReload((value) => value + 1); };
  const saveCard = async (event: React.FormEvent) => {
    event.preventDefault();
    if (saveRef.current || !canEdit || !draft.title.trim() || !draft.summary.trim()) return;
    saveRef.current = true; setSaving(true); setError("");
    const payload: Record<string, unknown> = { title: draft.title.trim(), summary: draft.summary.trim(), aliases: draft.aliases.split(/[,，\n]/).map((alias) => alias.trim()).filter(Boolean) };
    if (creating || isSemantic(card?.knowledge_type || "")) payload.definition = draft.definition.trim();
    if (creating) Object.assign(payload, { knowledge_type: draft.knowledge_type, database_id: draft.database_id });
    else payload.expected_revision = card?.revision;
    try {
      const response = await fetchJsonForUser<CardResponse>(creating ? "/api/knowledge-cards" : `/api/knowledge-cards/${encodeURIComponent(card!.knowledge_id)}`, devUser, { method: creating ? "POST" : "PATCH", body: JSON.stringify(payload) });
      setCard(response.card); setSelectedId(response.card.knowledge_id); setCreating(false); setEditing(false); setSaved("已保存"); onSaved();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "保存失败"); }
    finally { saveRef.current = false; setSaving(false); }
  };
  const saveRelation = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!card || !relationTarget || saveRef.current || !canEdit) return;
    saveRef.current = true; setSaving(true); setError("");
    try {
      const response = await fetchJsonForUser<CardResponse>("/api/knowledge-relations", devUser, { method: "POST", body: JSON.stringify({ source_id: card.knowledge_id, target_id: relationTarget, relation: relationType, expected_revision: card.revision }) });
      setCard(response.card); setRelationOpen(false); setRelationTarget(""); setTargetQuery(""); setSaved("已添加关联"); onSaved();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "关联失败"); }
    finally { saveRef.current = false; setSaving(false); }
  };
  const openRelated = (id: string) => selectNode(graph?.nodes.find((node) => node.knowledge_id === id) || null);
  const availableTargets = (graph?.nodes || []).filter((node) => node.knowledge_id !== card?.knowledge_id && node.database_id === card?.database_id && !(card?.relations || []).some((relation) => relation.source_id === card?.knowledge_id && relation.target_id === node.knowledge_id && relation.relation === relationType) && (!targetQuery.trim() || `${node.title} ${node.knowledge_id}`.toLowerCase().includes(targetQuery.trim().toLowerCase())));
  const inspectorVisible = Boolean(selectedId);
  const editableDefinition = creating || isSemantic(card?.knowledge_type || "");

  return <section className={`knowledge-editor-workspace ${inspectorVisible ? "with-inspector" : ""}`} aria-label="知识工作区">
    <div className="kw-graph-region"><KnowledgeGraph revision={revision} devUser={devUser} selectedKnowledgeId={selectedId} onSelectNode={selectNode} onGraphLoaded={receiveGraph} toolbarActions={<>
      {canEdit && <button className="kw-new-button" type="button" onClick={createNode} disabled={!resourcesReady || saving}><Icon name="add" /><span>新增</span></button>}
      <button className="kw-icon-button" type="button" title="知识库设置" aria-label="知识库设置" disabled={saving} onClick={() => { if (!mayLeave()) return; setEditing(false); setCreating(false); setRelationOpen(false); setRelationTarget(""); setError(""); setSettingsOpen(true); }}><Icon name="settings" /></button>
    </>} /></div>
    {inspectorVisible && <aside className="kw-inspector" aria-label="知识卡片">
      <div className="kw-inspector-bar"><span>{typeLabels[card?.knowledge_type || ""] || "知识"}</span><button className="kw-icon-button" type="button" aria-label="关闭知识卡片" onClick={() => selectNode(null)} disabled={saving}><Icon name="close" /></button></div>
      {loading && !creating ? <div className="kw-loading" role="status">正在读取…</div> : loadError && !creating ? <div className="kw-load-error" role="alert"><p>{loadError}</p><button className="kw-secondary-button" type="button" onClick={() => setReload((value) => value + 1)}>重试</button></div> : editing ? <form className="kw-edit-form" onSubmit={saveCard} aria-label="编辑知识">
        <fieldset disabled={saving}><CardFields draft={draft} setDraft={setDraft} creating={false} semantic={editableDefinition} databases={databases} /></fieldset>
        {error && <div className="kw-error" role="alert">{error}<button type="button" className="kw-reload-button" onClick={reloadCard} disabled={saving}>重新读取</button></div>}
        <footer className="kw-form-footer"><span>{`v${card?.revision}`}</span><div><button className="kw-secondary-button" type="button" onClick={cancelEdit} disabled={saving}>取消</button><button className="kw-primary-button" type="submit" disabled={saving || !dirty}>{saving ? "保存中…" : "保存"}</button></div></footer>
      </form> : card && <>
        <div className="kw-card-heading"><h2>{card.title}</h2>{card.aliases.length > 0 && <p className="kw-aliases">{card.aliases.filter((alias) => alias !== card.title).join(" · ")}</p>}</div>
        <section className="kw-card-section"><h3>{isSemantic(card.knowledge_type) ? "定义" : "摘要"}</h3><p className="kw-definition">{card.summary || card.definition}</p>{isSemantic(card.knowledge_type) && card.definition && card.summary !== card.definition && <details className="kw-more"><summary>完整定义</summary><p>{card.definition}</p></details>}</section>
        <section className="kw-card-section"><div className="kw-section-heading"><h3>关联</h3>{canEdit && <button className="kw-icon-button" type="button" aria-label="添加关联" title="添加关联" onClick={() => { if (!mayLeave()) return; setRelationOpen(true); setRelationType("related_to"); setRelationTarget(""); setTargetQuery(""); setError(""); }}><Icon name="add" /></button>}</div>
          {relationOpen ? <form className="kw-relation-form" aria-label="添加关联" onSubmit={saveRelation}><fieldset disabled={saving}>
            <label>关系<select name="relation" value={relationType} onChange={(event) => { setRelationType(event.target.value); setRelationTarget(""); }}><option value="related_to">相关</option>{card.knowledge_type === "metric" && <option value="sourced_from">依据</option>}{card.knowledge_type === "query_recipe" && <option value="requires">依赖</option>}</select></label>
            <label>目标节点<input aria-label="搜索目标节点" placeholder="搜索名称" value={targetQuery} onChange={(event) => setTargetQuery(event.target.value)} /><select name="target_id" aria-label="目标节点" value={relationTarget} onChange={(event) => setRelationTarget(event.target.value)} required><option value="">选择节点</option>{availableTargets.map((node) => <option value={node.knowledge_id} key={node.knowledge_id}>{node.title}</option>)}</select></label>
            <div className="kw-relation-direction"><span>{card.title}</span><Icon name="arrow" /><span>{titleById.get(relationTarget) || "目标节点"}</span></div>
          </fieldset>{error && <div className="kw-error" role="alert">{error}<button type="button" className="kw-reload-button" onClick={reloadCard} disabled={saving}>重新读取</button></div>}<div className="kw-form-actions"><button className="kw-secondary-button" type="button" disabled={saving} onClick={cancelEdit}>取消</button><button className="kw-primary-button" type="submit" disabled={saving || !relationTarget}>{saving ? "保存中…" : "添加"}</button></div></form> : null}
          {!card.relations.length && !relationOpen && <p className="kw-empty-relations">暂无关联</p>}
          <div className="kw-relations">{card.relations.map((relation) => { const outgoing = relation.source_id === card.knowledge_id; const target = outgoing ? relation.target_id : relation.source_id; const targetNode = graph?.nodes.find((node) => node.knowledge_id === target); return <button type="button" className="kw-relation-row" key={`${relation.source_id}/${relation.relation}/${relation.target_id}`} onClick={() => openRelated(target)}><i aria-hidden="true" className={`kw-node-dot kw-node-${targetNode?.knowledge_type || "glossary_term"}`} /><span>{titleById.get(target) || target}</span><small>{(outgoing ? relationLabels : incomingLabels)[relation.relation] || relation.relation}</small><Icon name="chevron" /></button>; })}</div>
        </section>
        <section className="kw-card-section"><h3>生效数据源</h3><div className="kw-source-row"><Icon name="database" /><span>{card.database_id || databaseName}</span></div></section>
        <section className="kw-card-section"><h3>来源</h3><details className="kw-source-details"><summary><Icon name="book" /><span>{card.source_file}</span><Icon name="chevron" /></summary><div>{card.evidence_refs.length ? card.evidence_refs.map((evidence, index) => <div className="kw-evidence-item" key={index}>{evidence.locator && <p>{evidence.locator}</p>}{evidence.result_summary && <p>{evidence.result_summary}</p>}{evidence.source_id && <small>{evidence.source_id}</small>}</div>) : <p>手动维护</p>}</div></details></section>
        <details className="kw-metadata"><summary>更多属性</summary><dl><dt>知识标识</dt><dd>{card.knowledge_id}</dd><dt>状态</dt><dd>{{ DRAFT: "草稿", VALIDATED: "已验证", REVIEWED: "已审核" }[card.status] || card.status}</dd></dl><pre>{JSON.stringify(card.payload, null, 2)}</pre></details>
        <footer className="kw-card-footer"><span>v{card.revision}</span>{canEdit && <button type="button" className="kw-edit-button" onClick={beginEdit}><Icon name="edit" />编辑</button>}</footer>
      </>}
    </aside>}
    {creating && <NewKnowledgeDialog close={cancelEdit} saving={saving}>
      <form className="kw-edit-form kw-create-form" onSubmit={saveCard} aria-label="新增知识">
        <fieldset disabled={saving}><CardFields draft={draft} setDraft={setDraft} creating semantic databases={databases} /></fieldset>
        {error && <div className="kw-error" role="alert">{error}</div>}
        <footer className="kw-form-footer"><div><button className="kw-secondary-button" type="button" onClick={cancelEdit} disabled={saving}>取消</button><button className="kw-primary-button" type="submit" disabled={saving}>{saving ? "保存中…" : "保存"}</button></div></footer>
      </form>
    </NewKnowledgeDialog>}
    <SettingsDialog open={settingsOpen} close={() => setSettingsOpen(false)}>{settings}</SettingsDialog>
    {saved && <div className="kw-save-status" role="status"><Icon name="check" />{saved}</div>}
  </section>;
}
