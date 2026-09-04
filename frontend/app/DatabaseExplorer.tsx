"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchJsonForUser } from "./api-client";

type DatabaseColumn = { name: string; data_type: string; nullable: boolean };
type DatabaseTable = { schema: string; name: string; kind: string; columns: DatabaseColumn[] };
type SchemaPayload = {
  backend: string;
  database: string;
  host: string;
  port: number;
  username: string;
  table_count: number;
  column_count: number;
  tables: DatabaseTable[];
};

export default function DatabaseExplorer({ revision, devUser }: { revision: number; devUser: string }) {
  const [payload, setPayload] = useState<SchemaPayload | null>(null);
  const [selectedName, setSelectedName] = useState("");
  const [query, setQuery] = useState("");
  const [refreshIndex, setRefreshIndex] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchJsonForUser<SchemaPayload>("/api/database-schema", devUser, { cache: "no-store" })
      .then((result) => {
        setError("");
        setPayload(result);
        setSelectedName((current) => current && result.tables.some((table) => `${table.schema}.${table.name}` === current)
          ? current
          : result.tables[0] ? `${result.tables[0].schema}.${result.tables[0].name}` : "");
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "数据库结构加载失败"))
      .finally(() => setLoading(false));
  }, [devUser, refreshIndex, revision]);

  const filteredTables = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return payload?.tables || [];
    return (payload?.tables || []).filter((table) => `${table.schema}.${table.name}`.toLowerCase().includes(normalized));
  }, [payload, query]);

  const selectedTable = payload?.tables.find((table) => `${table.schema}.${table.name}` === selectedName) || null;

  const refreshSchema = () => {
    setLoading(true);
    setError("");
    setPayload(null);
    setRefreshIndex((current) => current + 1);
  };

  return <section className="database-explorer">
    <div className="database-explorer-heading">
      <div><h2>Schema Explorer</h2></div>
      <button type="button" onClick={refreshSchema} disabled={loading}>刷新</button>
    </div>
    {loading && !payload ? <div className="database-explorer-state">正在读取数据库结构…</div> : error ? <div className="database-explorer-state error">{error}</div> : payload && <>
      <div className="database-stats">
        <div><strong>{payload.table_count}</strong><span>Tables</span></div>
        <div><strong>{payload.column_count}</strong><span>Columns</span></div>
        <div><strong>{payload.username}</strong><span>Current User</span></div>
      </div>
      <label className="database-table-search"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="查找表" /></label>
      <div className="database-explorer-body">
        <div className="database-table-list">
          {filteredTables.map((table) => {
            const fullName = `${table.schema}.${table.name}`;
            return <button type="button" className={selectedName === fullName ? "active" : ""} key={fullName} onClick={() => setSelectedName(fullName)}>
              <span><i />{table.name}</span><small>{table.columns.length}</small>
            </button>;
          })}
          {!filteredTables.length && <div className="database-empty">没有匹配的表</div>}
        </div>
        <div className="database-column-panel">
          {selectedTable ? <><div className="database-selected-table"><span>{selectedTable.schema}</span><strong>{selectedTable.name}</strong><small>{selectedTable.columns.length} 个字段</small></div><div className="database-column-list">
            {selectedTable.columns.map((column) => <div key={column.name}><strong>{column.name}</strong><code>{column.data_type}</code><span>{column.nullable ? "NULL" : "NOT NULL"}</span></div>)}
          </div></> : <div className="database-empty">选择一张表查看字段</div>}
        </div>
      </div>
    </>}
  </section>;
}
