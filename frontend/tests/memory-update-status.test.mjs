import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const source = await readFile(new URL("../app/MemoryUpdateStatus.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS } });
const module = { exports: {} };
new Function("require", "module", "exports", compiled.outputText)(createRequire(import.meta.url), module, module.exports);
const render = (...statuses) => renderToStaticMarkup(createElement(module.exports.default, {
  updates: statuses.map((status, index) => ({ tool_call_id: `call-${index}`, status })),
}));

test("ordinary replies have no memory badge", () => assert.equal(render(), ""));

test("submitted, failed and unknown writes never display a success badge", () => {
  for (const [status, label] of Object.entries({ updating: "正在更新记忆", pending: "记忆已提交，待确认", failed: "记忆更新失败", unknown: "记忆更新待确认" })) {
    const html = render(status);
    assert.ok(html.includes(label));
    assert.ok(!html.includes("记忆已更新"));
  }
});

test("completed write uses the icon and accessible status label", () => {
  const html = render("succeeded");
  assert.match(html, /记忆已更新/);
  assert.match(html, /memory-update-icon/);
  assert.match(html, /role="status"/);
});

test("no new entries is distinct from a successful write", () => {
  assert.match(render("unchanged"), /本次未新增记忆/);
  assert.ok(!render("unchanged").includes("记忆已更新"));
  assert.match(render("unchanged", "succeeded"), /记忆已更新/);
});

test("multiple writes cannot hide a pending or failed write behind success", () => {
  assert.ok(!render("succeeded", "pending").includes("记忆已更新"));
  assert.match(render("succeeded", "failed"), /记忆未全部更新成功/);
  assert.match(render("succeeded", "succeeded"), /记忆已更新/);
});
