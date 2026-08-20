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
});

test("exposes analysis, knowledge and model controls", async () => {
  const source = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(source, /deepseek-v4-pro/);
  assert.match(source, /deepseek-v4-flash/);
  assert.match(source, />Knowledge</);
  assert.match(source, />运行</);
});
