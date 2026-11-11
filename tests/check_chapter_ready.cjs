"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/listening_player.js"), "utf8");
const tick = () => new Promise(resolve => setImmediate(resolve));

function page(href = "http://127.0.0.1:8793/reference-preview/?project=bora", {fonts, animationFrames = true, timers = true, standalone = false} = {}) {
  const frames = [], messages = [], listeners = new Map();
  const context = vm.createContext({URL, location: new URL(href), document: {fonts: fonts ? {ready: fonts} : undefined}});
  context.window = context;
  context.parent = standalone ? context : {postMessage: (message, origin) => messages.push({message: JSON.parse(JSON.stringify(message)), origin})};
  context.addEventListener = (event, callback) => listeners.set(event, [...(listeners.get(event) || []), callback]);
  if (animationFrames) context.requestAnimationFrame = callback => frames.push(callback);
  if (timers) context.setTimeout = callback => frames.push(callback);
  vm.runInContext(source, context);
  const emit = (name, event = {}) => { for (const callback of listeners.get(name) || []) callback(event); };
  const request = (id, overrides = {}, envelope = {}) => emit("message", {
    source: context.parent, origin: context.location.origin,
    data: {type: "koww:chapter-request", project: "bora", request: id, url: href, ...overrides}, ...envelope,
  });
  return {context, ready: context.KowwChapterReady, messages, frames, request, emit,
    paint: async () => { await tick(); const next = frames.shift(); assert.ok(next, "a render turn is scheduled"); next(); await tick(); },
    settle: async () => { for (let count = 0; count < 8; count++) { await tick(); frames.shift()?.(); } await tick(); },
  };
}

(async () => {
  let fontsReady;
  const delayed = page(undefined, {fonts: new Promise(resolve => {fontsReady = resolve;})});
  assert.deepEqual(Object.keys(delayed.context.KowwListening).sort(), ["closePickers", "createPicker", "enhance"], "the existing playback APIs and shared-picker closing API remain public");
  delayed.request(1); await tick(); assert.deepEqual(delayed.messages, [], "pending queue data never reports ready");
  delayed.ready.ready(); await tick(); assert.equal(delayed.frames.length, 0, "font layout settles before render turns");
  fontsReady(); await delayed.paint(); assert.deepEqual(delayed.messages, [], "one render turn is insufficient");
  await delayed.paint();
  assert.deepEqual(delayed.messages, [{message: {type: "koww:chapter-ready", project: "bora", request: 1,
    url: "http://127.0.0.1:8793/reference-preview/?project=bora", status: "ready"}, origin: "http://127.0.0.1:8793"}]);

  const trusted = page(); trusted.ready.ready();
  for (const overrides of [{project: "other"}, {request: 0}, {request: 1.5}, {request: Number.MAX_SAFE_INTEGER + 1},
    {url: "/reference-preview/?project=bora"}, {url: "http://127.0.0.1:8793/reference-preview/?project=other"}, {url: "https://other.example/reference-preview/?project=bora"}]) trusted.request(1, overrides);
  trusted.request(1, {}, {source: {}}); trusted.request(1, {}, {origin: "https://other.example"});
  await trusted.settle(); assert.deepEqual(trusted.messages, [], "foreign or mismatched requests are ignored");
  trusted.request(2, {url: "http://127.0.0.1:8793/reference-preview/?project=bora#section"});
  await trusted.settle(); assert.equal(trusted.messages.at(-1).message.request, 2, "hashes do not distinguish a document");

  const latest = page(); latest.request(3); latest.request(4); latest.ready.ready(); await latest.settle();
  assert.deepEqual(latest.messages.map(row => row.message.request), [4], "only the latest retained request completes");
  latest.request(5); await latest.paint(); latest.ready.loading(); await latest.settle();
  assert.equal(latest.messages.length, 1, "loading invalidates an already scheduled ready response");
  latest.ready.fail("  목록을 다시 확인하세요.  "); await latest.settle();
  assert.deepEqual(latest.messages.at(-1).message, {type: "koww:chapter-ready", project: "bora", request: 5,
    url: "http://127.0.0.1:8793/reference-preview/?project=bora", status: "error", message: "목록을 다시 확인하세요."});
  latest.ready.loading(); latest.request(6); latest.ready.ready(); await latest.paint(); latest.emit("pagehide"); await latest.settle();
  assert.equal(latest.messages.length, 2, "a departed document cannot finish a parent transition");
  latest.emit("pageshow"); latest.request(7); await latest.settle(); assert.equal(latest.messages.at(-1).message.request, 7);

  for (const suffix of ["/results/bora/", "/mic/bora/"]) {
    const child = page("http://127.0.0.1:8793" + suffix); child.request(8); child.ready.ready(); await child.settle();
    assert.equal(child.messages.at(-1).message.project, "bora", "embedded result/mic routes identify their project");
  }
  const fallback = page(undefined, {animationFrames: false}); fallback.request(9); fallback.ready.ready(); await fallback.settle();
  assert.equal(fallback.messages.at(-1).message.status, "ready", "non-browser harnesses can settle without RAF");
  const immediate = page(undefined, {animationFrames: false, timers: false}); immediate.request(10); immediate.ready.ready(); await tick(); await tick();
  assert.equal(immediate.messages.at(-1).message.status, "ready");
  const standalone = page(undefined, {standalone: true}); standalone.request(11); standalone.ready.ready(); await standalone.settle(); assert.deepEqual(standalone.messages, []);
  console.log("Chapter rendered-readiness: trusted requests, deferred data/fonts/paint, retries, cancellation and route identity passed");
})().catch(error => {console.error(error); process.exitCode = 1;});
