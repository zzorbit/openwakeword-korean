"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/reference_preview.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

async function screen(mode, items, { error, ok = true } = {}) {
  const nodes = new Map();
  class Element {
    constructor() { this.value = ""; this.textContent = ""; this.children = []; this.handlers = {}; this.attributes = {}; this.played = 0; this.paused = 0; }
    append(node) { this.children.push(node); }
    addEventListener(event, callback) { this.handlers[event] = callback; }
    setAttribute(key, value) { this.attributes[key] = value; }
    play() { this.played++; return Promise.resolve(); }
    pause() { this.paused++; }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], new Element());
  const calls = [], messages = {}, document = { title: "기준 목소리 듣기", getElementById: id => nodes.get(id), createElement: () => new Element() };
  const context = vm.createContext({ document, URL, location: new URL(`http://127.0.0.1:8793/${mode}/?project=bora`),
    fetch: async (url, options) => { calls.push({ url, options }); return { ok, json: async () => ({ items, error }) }; } });
  context.window = context; context.parent = {};
  context.addEventListener = (name, callback) => { messages[name] = callback; };
  vm.runInContext(script, context);
  for (let index = 0; index < 8; index++) await new Promise(resolve => setImmediate(resolve));
  return { nodes, calls, document, messages, parent: context.parent };
}

(async () => {
  assert.doesNotMatch(html, /<form|localStorage|listened|decision|whole_pool_policy_approved/);
  const items = [
    { id: "hard1", text: "돌아와", voice_family_id: "train_01", audio: "/hard-preview/audio/0?project=bora" },
    { id: "hard2", text: "기다려", voice_family_id: "train_02", audio: "/hard-preview/audio/1?project=bora" }
  ];
  const hard = await screen("hard-preview", items);
  assert.equal(hard.document.title, "유사어 전체 듣기");
  assert.equal(hard.nodes.get("voice").attributes["aria-label"], "들을 유사어 음원");
  assert.equal(hard.nodes.get("voice").children[0].textContent, "1. 돌아와 · train_01");
  assert.equal(hard.nodes.get("audio").src, items[0].audio);
  assert.equal(hard.nodes.get("previous").disabled, true);
  hard.nodes.get("continuous").handlers.click();
  assert.equal(hard.nodes.get("audio").played, 1);
  hard.nodes.get("audio").handlers.ended();
  assert.equal(hard.nodes.get("audio").src, items[1].audio);
  assert.equal(hard.nodes.get("audio").played, 2);
  assert.equal(hard.nodes.get("next").disabled, true);
  hard.nodes.get("audio").handlers.ended();
  assert.equal(hard.nodes.get("continuous").textContent, "연속 듣기");
  assert.deepEqual(hard.calls, [{ url: "/hard-preview/queue?project=bora", options: undefined }]);
  const reference = await screen("reference-preview", [{ id: "train1", group: "학습용", text: "오늘은", audio: "/reference-preview/audio/0?project=bora" }]);
  assert.equal(reference.document.title, "기준 목소리 듣기");
  assert.equal(reference.nodes.get("voice").children[0].textContent, "1. 학습용 · train1");
  assert.deepEqual(reference.calls, [{ url: "/reference-preview/queue?project=bora", options: undefined }]);
  const paused = await screen("hard-preview", items);
  paused.nodes.get("next").handlers.click();
  paused.nodes.get("continuous").handlers.click();
  paused.messages.message({origin: "https://other.example", source: paused.parent, data: {type: "koww:pause"}});
  paused.messages.message({origin: "http://127.0.0.1:8793", source: {}, data: {type: "koww:pause"}});
  assert.equal(paused.nodes.get("audio").paused, 0);
  paused.messages.message({origin: "http://127.0.0.1:8793", source: paused.parent, data: {type: "koww:pause"}});
  assert.equal(paused.nodes.get("audio").paused, 1);
  assert.equal(paused.nodes.get("audio").src, items[1].audio);
  assert.equal(paused.nodes.get("voice").value, "1");
  assert.equal(paused.nodes.get("continuous").textContent, "연속 듣기");
  assert.equal(paused.calls.length, 1);
  for (const audio of ["https://example.com/private.wav", "javascript:alert(1)", "", null]) {
    const invalid = await screen("hard-preview", [{ ...items[0], audio }]);
    assert.match(invalid.nodes.get("error").textContent, /음원 경로가 올바르지 않습니다/);
    assert.equal(invalid.nodes.get("audio").src, undefined);
  }
  const empty = await screen("hard-preview", []);
  assert.equal(empty.nodes.get("error").textContent, "준비된 유사어 음원이 없습니다.");
  const missing = await screen("hard-preview", null);
  assert.equal(missing.nodes.get("error").textContent, "음원 목록이 올바르지 않습니다.");
  const failed = await screen("hard-preview", [], { ok: false, error: "준비 자료가 없습니다." });
  assert.equal(failed.nodes.get("error").textContent, "준비 자료가 없습니다.");
  const unsafeText = await screen("hard-preview", [{ ...items[0], text: "<img src=x onerror=bad()>" }]);
  assert.equal(unsafeText.nodes.get("sentence").textContent, "<img src=x onerror=bad()>");
  console.log("Read-only preview UI: hard/reference playback, no consent writes, source validation and errors passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
