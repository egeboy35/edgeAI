// Where a server-held API key is allowed to be sent.
//
// `resolveBackend` takes the base URL from the request when one is given, and
// the key from the server's environment when the request does not carry one.
// These pin down that the two halves are never combined: a caller that
// redirects the base URL has to bring its own key.
//
// Node's built-in runner, so the app gains no dependency:
//
//     node --test edgeLLM/nextjs-nemotron-app/lib/providers.test.mjs
//
// A scratch home directory is set before the import because providers.js reads
// ~/.env.local once, at first call, and the real one must not leak in here.

import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const scratchHome = mkdtempSync(join(tmpdir(), "edgeai-providers-"));
writeFileSync(join(scratchHome, ".env.local"), "");
process.env.HOME = scratchHome;
process.env.USERPROFILE = scratchHome;

const { resolveBackend } = await import("./providers.js");

const SERVER_KEY = "nvapi-server-held-key";
const ELSEWHERE = "http://somewhere-else.example/v1";
const NVIDIA = "https://integrate.api.nvidia.com/v1";

function withEnv(vars, fn) {
  const saved = {};
  for (const [k, v] of Object.entries(vars)) {
    saved[k] = process.env[k];
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  try {
    return fn();
  } finally {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  }
}

const cloud = (extra = {}) =>
  withEnv({ NVIDIA_API_KEY: SERVER_KEY, NVIDIA_BASE_URL: undefined, ...extra.env },
    () => resolveBackend("nvidia", extra.options || {}));

// ------------------------------------------------------- the ordinary path
test("with no override the server key goes to the configured endpoint", () => {
  const b = cloud();
  assert.equal(b.baseUrl, NVIDIA);
  assert.equal(b.apiKey, SERVER_KEY);
});

test("an environment base URL is still honoured with the server key", () => {
  const b = cloud({ env: { NVIDIA_BASE_URL: "https://proxy.internal/v1" } });
  assert.equal(b.baseUrl, "https://proxy.internal/v1");
  assert.equal(b.apiKey, SERVER_KEY);
});

test("a request naming the configured endpoint still gets the server key", () => {
  const b = cloud({ options: { baseUrl: NVIDIA } });
  assert.equal(b.apiKey, SERVER_KEY);
});

// --------------------------------------------------------- the redirection
test("a redirected base URL does not receive the server key", () => {
  const b = cloud({ options: { baseUrl: ELSEWHERE } });
  assert.equal(b.baseUrl, ELSEWHERE);
  assert.notEqual(b.apiKey, SERVER_KEY);
  assert.equal(b.apiKey, "EMPTY");
});

// --------------------------------------------- the flag the route reads
// These two are the only ones asserting on the fields this change adds; the
// behavioural tests above deliberately do not, so that running them against
// the previous resolver fails on behaviour rather than on a missing field.
test("a withheld key is flagged, with the endpoint it was measured against", () => {
  const b = cloud({ options: { baseUrl: ELSEWHERE } });
  assert.equal(b.keyWithheld, true);
  assert.equal(b.serverBaseUrl, NVIDIA);
  assert.equal(b.keyEnv, "NVIDIA_API_KEY");
});

test("nothing is flagged when the key was legitimately applied or absent", () => {
  assert.equal(cloud().keyWithheld, false);
  assert.equal(cloud({ options: { baseUrl: NVIDIA } }).keyWithheld, false);
  assert.equal(cloud({ options: { baseUrl: ELSEWHERE, apiKey: "sk-own" } }).keyWithheld, false);
  assert.equal(
    withEnv({ LLAMA_BASE_URL: undefined }, () =>
      resolveBackend("llama", { baseUrl: ELSEWHERE })).keyWithheld,
    false
  );
});

test("whitespace around a redirected URL does not slip the key through", () => {
  const b = cloud({ options: { baseUrl: `  ${ELSEWHERE}  ` } });
  assert.equal(b.apiKey, "EMPTY");
});

test("a redirected URL under an env-configured endpoint is still a redirect", () => {
  const b = cloud({
    env: { NVIDIA_BASE_URL: "https://proxy.internal/v1" },
    options: { baseUrl: ELSEWHERE },
  });
  assert.equal(b.apiKey, "EMPTY");
});

// ------------------------------------------------- callers with their own key
test("a caller that brings its own key may point anywhere", () => {
  const b = cloud({ options: { baseUrl: ELSEWHERE, apiKey: "sk-callers-own" } });
  assert.equal(b.baseUrl, ELSEWHERE);
  assert.equal(b.apiKey, "sk-callers-own");
});

// ------------------------------------------------- backends without a key
test("a key-less backend is unaffected by a base URL override", () => {
  const b = withEnv({ LLAMA_BASE_URL: undefined }, () =>
    resolveBackend("llama", { baseUrl: "http://192.168.1.10:8000/v1" }));
  assert.equal(b.baseUrl, "http://192.168.1.10:8000/v1");
  assert.equal(b.keyEnv, null);
});

// ---------------------------------------------------- the custom backend
test("the custom backend gets its key when it points at its configured URL", () => {
  const b = withEnv(
    { CUSTOM_BASE_URL: "http://192.168.1.10:8000/v1", CUSTOM_API_KEY: "sk-lab" },
    () => resolveBackend("custom", { baseUrl: "http://192.168.1.10:8000/v1" }));
  assert.equal(b.apiKey, "sk-lab");
});

test("the custom backend withholds its key when pointed somewhere else", () => {
  const b = withEnv(
    { CUSTOM_BASE_URL: "http://192.168.1.10:8000/v1", CUSTOM_API_KEY: "sk-lab" },
    () => resolveBackend("custom", { baseUrl: ELSEWHERE }));
  assert.equal(b.apiKey, "EMPTY");
});

test("the custom backend still works with a key supplied per request", () => {
  const b = withEnv({ CUSTOM_BASE_URL: undefined, CUSTOM_API_KEY: undefined }, () =>
    resolveBackend("custom", { baseUrl: ELSEWHERE, apiKey: "sk-typed-in-the-ui" }));
  assert.equal(b.baseUrl, ELSEWHERE);
  assert.equal(b.apiKey, "sk-typed-in-the-ui");
});

// ------------------------------------------------------------- unchanged
test("an unknown backend id still falls back to nvidia", () => {
  const b = cloud({ options: {} });
  assert.equal(withEnv({ NVIDIA_API_KEY: SERVER_KEY }, () =>
    resolveBackend("no-such-backend", {}).id), "no-such-backend");
  assert.equal(b.name, "NVIDIA Build");
});

test("the model still comes from the request, then env, then the default", () => {
  const b = cloud({ options: { model: "some/model" } });
  assert.equal(b.model, "some/model");
  const c = withEnv({ NVIDIA_API_KEY: SERVER_KEY, AGENT_MODEL: "env/model" }, () =>
    resolveBackend("nvidia", {}));
  assert.equal(c.model, "env/model");
});
