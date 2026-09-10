import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const source = await readFile(new URL("../app/DatasourceConversationPicker.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2017 } });
const module = { exports: {} };
new Function("require", "module", "exports", compiled.outputText)(createRequire(import.meta.url), module, module.exports);
const { default: Picker, sameConversationBinding, mergeConversationSources, knowledgeProfiles, conversationListPath, filterConversationScope, hasMultipleKnowledgeBases } = module.exports;
const binding = (data_source_id, knowledge_base_id, name = "虚拟偶像") => ({ data_source_id, knowledge_base_id, data_source_name: name, knowledge_base_name: "运营知识库", knowledge_root: "C:/knowledge", profile_id: `profile-${knowledge_base_id}`, backend: "postgresql", database: data_source_id });

test("conversation binding requires both source and knowledge identity, never only a profile label", () => {
  const current = binding("idol", "kb-a");
  assert.equal(sameConversationBinding(current, { ...current, profile_id: "renamed" }), true);
  assert.equal(sameConversationBinding(current, binding("cold", "kb-a")), false);
  assert.equal(sameConversationBinding(current, binding("idol", "kb-b")), false);
  assert.equal(sameConversationBinding(null, null), false);
  assert.equal(sameConversationBinding(current, null), false);
});

test("source groups retain archived sources and merge multiple knowledge profiles into one source", () => {
  const groups = mergeConversationSources([{ binding: binding("idol", "kb-a") }, { binding: binding("idol", "kb-b") }], [binding("cold", "kb-c", "冷链")], binding("idol", "kb-a"));
  assert.deepEqual(groups.map((item) => item.data_source_id), ["cold", "idol"]);
  const choices = knowledgeProfiles([{ binding: binding("idol", "kb-a") }, { binding: binding("idol", "kb-b") }, { binding: binding("idol", "kb-a") }, { binding: binding("cold", "kb-c") }, { binding: null }], "idol");
  assert.deepEqual(choices.map((profile) => profile.binding.knowledge_base_id), ["kb-a", "kb-b"]);
});

test("source labels prefer the first configured profile, then the active binding", () => {
  const profiles = [{ binding: binding("idol", "kb-a", "虚拟偶像") }, { binding: binding("idol", "kb-b", "另一知识库方案") }];
  const historical = [binding("idol", "kb-old", "旧名称")];
  assert.equal(mergeConversationSources(profiles, historical)[0].data_source_name, "虚拟偶像");
  assert.equal(mergeConversationSources(profiles, historical, binding("idol", "kb-b", "当前方案"))[0].data_source_name, "当前方案");
});

test("a stale or mixed list never leaks a different source or legacy conversation", () => {
  const conversations = [{ id: 1, binding: binding("idol", "kb-a") }, { id: 2, binding: binding("cold", "kb-c") }, { id: 3, binding: null }];
  assert.deepEqual(filterConversationScope(conversations, "idol").map((item) => item.id), [1]);
  assert.deepEqual(filterConversationScope(conversations, "cold").map((item) => item.id), [2]);
  assert.deepEqual(filterConversationScope(conversations, "legacy").map((item) => item.id), [3]);
  assert.deepEqual(filterConversationScope(conversations, ""), []);
});

test("knowledge labels still distinguish a deleted knowledge profile in saved conversations", () => {
  const profiles = [{ binding: binding("idol", "kb-current") }];
  assert.equal(hasMultipleKnowledgeBases(profiles, [{ binding: binding("idol", "kb-deleted") }], "idol"), true);
  assert.equal(hasMultipleKnowledgeBases(profiles, [{ binding: binding("cold", "kb-other") }], "idol"), false);
  assert.equal(hasMultipleKnowledgeBases(profiles, [{ binding: null }], "idol"), false);
});

test("legacy requests stay distinct from active source requests", () => {
  assert.equal(conversationListPath("legacy"), "/api/conversations?legacy=true");
  assert.equal(conversationListPath("db/a & b"), "/api/conversations?data_source_id=db%2Fa%20%26%20b");
  assert.equal(conversationListPath(), "/api/conversations");
});

test("renders one keyboard accessible source selector without explanatory copy", () => {
  const html = renderToStaticMarkup(createElement(Picker, { sources: [binding("idol", "kb-a"), binding("cold", "kb-b", "冷链")], value: "idol", legacyCount: 5, disabled: false, onChange() {} }));
  assert.match(html, /aria-label="切换数据源"/);
  assert.match(html, /value="idol" selected=""/);
  assert.match(html, /未归属会话 \(5\)/);
  assert.doesNotMatch(html, /<(?:small|p)(?:\s|>)|点击|你可以/);
});

test("page protects history scope changes, submits identities and retains read-only legacy access", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /data_source_id: threadBinding\.data_source_id, knowledge_base_id: threadBinding\.knowledge_base_id/);
  assert.match(page, /generation !== scopeGeneration\.current/);
  assert.match(page, /requestGeneration === historyRequestGeneration\.current/);
  assert.match(page, /filterConversationScope\(response\.conversations, historyScopeRef\.current\)/);
  assert.match(page, /放弃尚未发送的问题/);
  assert.match(page, /未归属会话 · 只读/);
  assert.match(page, /className="empty-state">\{chatUnavailableReason && <h2>\{chatUnavailableReason\}<\/h2>\}/);
  assert.doesNotMatch(page, /开始一次数据分析/);
  assert.match(page, /使用原配置/);
  assert.match(page, /showSavedConversation\(savedConversation\)/);
  assert.match(page, /if \(response\.status === 409\) setConversationCanContinue\(false\)/);
  assert.doesNotMatch(page, /page === "analysis" \? `\$\{backendName\(state\.active\.backend\)/);
});
