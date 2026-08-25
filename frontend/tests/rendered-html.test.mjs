import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  const request = new Request("http://localhost/", {
    headers: { accept: "text/html" },
  });
  return typeof worker === "function"
    ? worker(request)
    : worker.fetch(
        request,
        { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
        { waitUntil() {}, passThroughOnException() {} },
      );
}

test("renders the DataAgent product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /<title>DataAgent<\/title>/i);
  assert.match(html, /正在加载 DataAgent/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/);
});

test("keeps database configuration domain-neutral", async () => {
  const source = await readFile(
    new URL("../app/page.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /<option value="mysql">MySQL<\/option>/);
  assert.match(source, /database: ""/);
  assert.match(source, />数据库名称</);
  assert.match(source, /deleteProfile/);
  assert.match(source, />删除配置</);
  assert.match(source, /填写 PostgreSQL 或 MySQL 中实际存在的数据库名称/);
});

test("exposes analysis, knowledge and model controls", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(source, /deepseek-v4-pro/);
  assert.match(source, /deepseek-v4-flash/);
  assert.match(source, />模型</);
  assert.match(source, /模型 API Key/);
  assert.match(source, /保存后立即生效/);
  assert.doesNotMatch(source, /保存后重启生效/);
  assert.match(source, />Knowledge</);
  assert.match(source, /开始新分析/);
  assert.match(source, /conversation-history/);
  assert.match(source, /setPage\("analysis"\)/);
  assert.doesNotMatch(source, /page === "analysis" && <div className="conversation-history"/);
  assert.match(source, /renameConversation/);
  assert.match(source, /deleteConversation/);
  assert.match(source, />重命名</);
  assert.match(source, />删除</);
  assert.match(source, /placeholder="询问 DataAgent"/);
  assert.match(source, /className="stop-symbol"/);
  assert.doesNotMatch(source, /message\.role === "user" \? "你" : "DataAgent"/);
  assert.doesNotMatch(source, />运行</);
  assert.doesNotMatch(source, /输入业务问题。Agent 会查找/);
  assert.doesNotMatch(source, /输入数据分析问题，Enter 发送/);
  assert.doesNotMatch(source, /模型本轮输出/);
  assert.doesNotMatch(source, /本轮 Tool Call/);
});

test("renders the interactive Knowledge graph from runtime data", async () => {
  const pageSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const graphSource = await readFile(new URL("../app/KnowledgeGraph.tsx", import.meta.url), "utf8");

  assert.match(pageSource, /<KnowledgeGraph revision=\{runtimeRevision\} \/>/);
  assert.match(pageSource, /setRuntimeRevision/);
  assert.match(graphSource, /\/api\/knowledge-graph/);
  assert.match(graphSource, /cache: "no-store"/);
  assert.match(graphSource, /<canvas/);
  assert.match(graphSource, /pointerdown/);
  assert.match(graphSource, /wheel/);
});

test("shows Knowledge navigation only while runtime events request it", async () => {
  const pageSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const graphSource = await readFile(new URL("../app/KnowledgeGraph.tsx", import.meta.url), "utf8");

  assert.match(pageSource, /event\.type === "knowledge_trace"/);
  assert.match(pageSource, /event\.action === "open"/);
  assert.match(pageSource, /live-knowledge-overlay/);
  assert.match(graphSource, /activeIds/);
  assert.match(graphSource, /frontierRefs/);
  assert.match(graphSource, /lineDashOffset/);
});

test("shows the active database schema beside its connection settings", async () => {
  const pageSource = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const explorerSource = await readFile(new URL("../app/DatabaseExplorer.tsx", import.meta.url), "utf8");

  assert.match(pageSource, /<DatabaseExplorer revision=\{runtimeRevision\} \/>/);
  assert.match(explorerSource, /\/api\/database-schema/);
  assert.match(explorerSource, /Schema Explorer/);
  assert.match(explorerSource, /column\.data_type/);
});

test("streams Agent progress and exposes cooperative cancellation", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(source, /\/api\/chat\/stream/);
  assert.match(source, /\/api\/runs\/\$\{runId\}\/cancel/);
  assert.match(source, /停止将在当前模型或工具调用结束后的安全位置生效/);
  assert.match(source, /currentRound/);
  assert.doesNotMatch(source, /本轮没有文本输出/);
  assert.doesNotMatch(source, /模型本轮输出/);
  assert.doesNotMatch(source, /本轮 Tool Call/);
  assert.doesNotMatch(source, /activityLog/);
});

test("keeps every SQL beside its own Tool Result", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(source, /<ResultTable result=\{query\.result\} \/>/);
  assert.doesNotMatch(source, /<ResultTable result=\{message\.details\.result_preview\} \/>/);
});
