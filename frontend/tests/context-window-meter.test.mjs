import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const source = await readFile(new URL("../app/ContextWindowMeter.tsx", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS } });
const module = { exports: {} };
new Function("require", "module", "exports", compiled.outputText)(createRequire(import.meta.url), module, module.exports);
const Meter = module.exports.default;
const model = "deepseek-v4-pro";
const render = (usage, extra = {}) => renderToStaticMarkup(createElement(Meter, { usage, model, capacity: 1_048_576, ...extra }));
const usage = (input_tokens, context_window = 1_048_576) => ({ input_tokens, context_window, model, source: "provider" });

test("renders a provider-measured circular meter with an accessible tooltip", () => {
  const html = render(usage(262_144));
  assert.match(html, /role="progressbar"/);
  assert.match(html, /aria-valuenow="25"/);
  assert.match(html, /stroke-dasharray="25 100"/);
  assert.match(html, /262,144 \/ 1,048,576/);
  assert.match(html, /最近一次模型输入/);
  assert.match(html, /role="tooltip"/);
});

test("missing or mismatched usage is unknown, not zero or a previous model's percentage", () => {
  for (const snapshot of [null, { ...usage(200), input_tokens: null, source: "unavailable" }, { ...usage(200), model: "deepseek-v4-flash" }, usage(-1), usage(NaN)]) {
    const html = render(snapshot);
    assert.match(html, /context-meter unknown/);
    assert.doesNotMatch(html, /aria-valuenow=/);
    assert.match(html, /暂无用量/);
  }
});

test("handles tiny usage, zero, warning and overflow without wrapping the ring", () => {
  assert.match(render(usage(500)), /&lt;1%/);
  assert.match(render(usage(0)), /aria-valuenow="0"/);
  assert.match(render(usage(850_000, 1_000_000)), /context-meter warning/);
  const overflow = render(usage(1_100_000, 1_000_000));
  assert.match(overflow, /context-meter critical/);
  assert.match(overflow, /aria-valuenow="100"/);
  assert.match(overflow, /stroke-dasharray="100 100"/);
  assert.match(overflow, />110%<\/span>/);
});

test("unknown capacity never produces a made-up percentage", () => {
  const html = render({ ...usage(200), context_window: null }, { capacity: null });
  assert.doesNotMatch(html, /aria-valuenow=/);
});
