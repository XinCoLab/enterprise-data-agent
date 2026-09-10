export type ConversationBinding = {
  data_source_id: string;
  data_source_name: string;
  knowledge_base_id: string;
  knowledge_base_name: string;
  knowledge_root: string;
  profile_id: string;
  backend: "postgresql" | "mysql" | "duckdb";
  database: string;
};
export type ConversationSource = Pick<ConversationBinding, "data_source_id" | "data_source_name" | "backend" | "database">;
type BoundProfile = { binding?: ConversationBinding | null };
export const LEGACY_SCOPE = "legacy";

export function sameConversationBinding(left?: ConversationBinding | null, right?: ConversationBinding | null) {
  return Boolean(left && right && left.data_source_id === right.data_source_id && left.knowledge_base_id === right.knowledge_base_id);
}

export function conversationListPath(scope?: string) {
  if (scope === LEGACY_SCOPE) return "/api/conversations?legacy=true";
  return scope ? `/api/conversations?data_source_id=${encodeURIComponent(scope)}` : "/api/conversations";
}

export function mergeConversationSources(profiles: BoundProfile[], historical: ConversationSource[], active?: ConversationBinding | null) {
  const sources = new Map<string, ConversationSource>();
  const configuredSources = new Set<string>();
  for (const source of historical) sources.set(source.data_source_id, source);
  for (const profile of profiles) {
    if (profile.binding && !configuredSources.has(profile.binding.data_source_id)) {
      sources.set(profile.binding.data_source_id, profile.binding);
      configuredSources.add(profile.binding.data_source_id);
    }
  }
  if (active) sources.set(active.data_source_id, active);
  return [...sources.values()];
}

export function filterConversationScope<T extends BoundProfile>(conversations: T[], scope: string) {
  return conversations.filter((conversation) => scope === LEGACY_SCOPE
    ? conversation.binding == null
    : Boolean(scope && conversation.binding?.data_source_id === scope));
}

export function hasMultipleKnowledgeBases(profiles: BoundProfile[], conversations: BoundProfile[], sourceId: string) {
  return new Set([...profiles, ...conversations]
    .filter((item) => item.binding?.data_source_id === sourceId)
    .map((item) => item.binding!.knowledge_base_id)).size > 1;
}

export function knowledgeProfiles<T extends BoundProfile>(profiles: T[], sourceId: string) {
  const seen = new Set<string>();
  return profiles.filter((profile) => {
    const binding = profile.binding;
    if (!binding || binding.data_source_id !== sourceId || seen.has(binding.knowledge_base_id)) return false;
    seen.add(binding.knowledge_base_id);
    return true;
  });
}

export default function DatasourceConversationPicker({ sources, value, legacyCount, disabled, onChange }: {
  sources: ConversationSource[];
  value: string;
  legacyCount: number;
  disabled: boolean;
  onChange: (sourceId: string) => void;
}) {
  return <label className="conversation-source-picker" title="切换数据源">
    <svg aria-hidden="true" viewBox="0 0 20 20"><ellipse cx="10" cy="4.5" rx="6" ry="2.5" /><path d="M4 4.5v11c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5v-11M4 10c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5" /></svg>
    <select aria-label="切换数据源" value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
      {!value && <option value="">未配置数据源</option>}
      {sources.map((source) => <option key={source.data_source_id} value={source.data_source_id}>{source.data_source_name}</option>)}
      {(legacyCount > 0 || value === LEGACY_SCOPE) && <option value={LEGACY_SCOPE}>未归属会话{legacyCount > 0 ? ` (${legacyCount})` : ""}</option>}
    </select>
    <svg className="source-chevron" aria-hidden="true" viewBox="0 0 16 16"><path d="m4 6 4 4 4-4" /></svg>
  </label>;
}
