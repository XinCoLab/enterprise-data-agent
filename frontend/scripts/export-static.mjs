import { mkdir, writeFile } from "node:fs/promises";

const workerUrl = new URL("../dist/server/index.js", import.meta.url);
workerUrl.searchParams.set("export", `${Date.now()}`);
const { default: worker } = await import(workerUrl.href);
const request = new Request("http://localhost/", {
  headers: { accept: "text/html" },
});
const response =
  typeof worker === "function"
    ? await worker(request)
    : await worker.fetch(
        request,
        { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
        { waitUntil() {}, passThroughOnException() {} },
      );

if (!response.ok) {
  throw new Error(`Static export failed with HTTP ${response.status}`);
}

const output = new URL("../dist/client/index.html", import.meta.url);
await mkdir(new URL("../dist/client/", import.meta.url), { recursive: true });
await writeFile(output, await response.text(), "utf8");
