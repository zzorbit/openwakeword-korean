"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/review.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

async function screen(queue, search, saved = {}, config = {}) {
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1], { textContent: "", innerHTML: "", children: [], append(node) { this.children.push(node); } }]));
  const calls = [], writes = [];
  nodes.set("koww-page-config", { textContent: JSON.stringify(config) });
  const existing = JSON.stringify(saved);
  const context = vm.createContext({
    document: { getElementById: id => nodes.get(id), createElement: () => ({}) },
    URLSearchParams, location: { search },
    localStorage: { getItem(key) { assert.equal(key, "koww:" + queue.queue_sha256); return existing; }, setItem(...args) { writes.push(args); } },
    fetch: async (url, options) => { calls.push({ url, options }); return { ok: true, json: async () => options?.method === "POST" ? { state: "saved" } : queue }; }
  });
  vm.runInContext(script, context);
  for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve));
  return { nodes, calls, writes };
}

(async () => {
  const queue = { wakeword: "보라야", purpose: "합성 목소리가 자연스럽고 서로 다른지 확인하세요.", queue_sha256: "fixed-queue", items: [
    { id: "one", text: "오늘은 창밖으로 따뜻한 햇살이 들어옵니다.", audio: "/audio/0" },
    { id: "two", text: "오늘은 창밖으로 따뜻한 햇살이 들어옵니다.", audio: "/audio/1" }
  ] };
  const voices = await screen(queue, "?project=bora&queue=references", { one: { decision: "discard", listened: true } });
  assert.equal(voices.nodes.get("review-heading").textContent, "합성 목소리 확인");
  assert.match(voices.nodes.get("purpose").textContent, /‘보라야’를 생성/);
  assert.match(voices.nodes.get("purpose").textContent, /호출어 학습 음원으로 쓰지 않습니다/);
  assert.match(voices.nodes.get("expected").textContent, /^샘플 문장: 오늘은/);
  assert.match(voices.nodes.get("criteria").innerHTML, /목소리가 자연스럽/);
  assert.equal(voices.nodes.get("state").textContent, "discard");
  assert.deepEqual(voices.writes, []);
  assert.deepEqual(voices.calls, [{ url: "/queue", options: undefined }]);
  assert.equal(voices.nodes.get("items").children.length, 2);
  voices.nodes.get("save").onclick();
  assert.match(voices.nodes.get("saved").textContent, /아직 1개/);
  assert.equal(voices.calls.length, 1); // A wording update never saves unheard answers.

  const pilot = await screen({ ...queue, kind: "pilot", purpose: "호출어 확인" }, "?queue=pilot");
  assert.equal(pilot.nodes.get("purpose").textContent, "호출어 확인");
  assert.match(pilot.nodes.get("expected").textContent, /^기대 문장:/);
  const content = await screen({ ...queue, policy_review: true, policy_text: "묶음 사용 동의" }, "?queue=content");
  assert.equal(content.nodes.get("policy").hidden, false);
  assert.equal(content.nodes.get("policy-text").textContent, "묶음 사용 동의");
  assert.match(content.nodes.get("criteria").innerHTML, /안내에 맞고/);
  const heard = { one: { decision: "keep", listened: true }, two: { decision: "keep", listened: true } };
  const config = { queue_url: "/review/queue?project=bora&queue=pilot", save_url: "/review/save?project=bora&queue=pilot", token: "studio-review-token", saved_message: "저장 완료 · 다음 단계를 준비합니다." };
  const nested = await screen(queue, "?queue=pilot", heard, config);
  await nested.nodes.get("save").onclick();
  assert.equal(nested.calls[0].url, config.queue_url);
  assert.equal(nested.calls[1].url, config.save_url);
  assert.equal(nested.calls[1].options.headers["X-Koww-Token"], config.token);
  assert.deepEqual(JSON.parse(nested.calls[1].options.body), heard);
  assert.equal(nested.nodes.get("saved").textContent, config.saved_message);
  const standalone = await screen(queue, "?queue=pilot", heard);
  await standalone.nodes.get("save").onclick();
  assert.equal(standalone.calls[1].url, "/save");
  assert.equal(standalone.calls[1].options.headers["X-Koww-Token"], undefined);
  assert.deepEqual(JSON.parse(standalone.calls[1].options.body), heard);
  console.log("Review UI: voice purpose, saved decisions, unheard gate and other review types passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
