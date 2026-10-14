"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const resources = path.resolve(__dirname, "../korean_openwakeword/resources");
const tests = [];
async function test(name, task) { await task(); tests.push(name); }
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(resolve => setImmediate(resolve)); };

function worklet() {
  const messages = [];
  let Processor;
  const context = vm.createContext({
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage(message, transfer = []) { messages.push(structuredClone(message, { transfer })); } }; } },
    registerProcessor(name, klass) { assert.equal(name, "koww-microphone"); Processor = klass; }
  });
  vm.runInContext(fs.readFileSync(path.join(resources, "microphone_worklet.js"), "utf8"), context);
  return { context, messages, instance: new Processor() };
}

function browser() {
  const elements = new Map(), listeners = {}, mediaListeners = {}, contexts = [], workers = [], calls = [], streams = [];
  let clock = 0, nextTimer = 0;
  const timers = new Map(), intervals = new Map();
  class Element {
    constructor(id = "") { this.id = id; this.textContent = ""; this.children = []; this.handlers = {}; this.disabled = false; this.value = ""; this.className = ""; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, handler) { this.handlers[name] = handler; }
    get options() { return this.children; }
  }
  class Option extends Element { constructor(text, value) { super(); this.textContent = text; this.value = value; } }
  const state = { deferredPermission: null, deferredStart: null, failFrame: false, holdFrame: false, audioRate: 16000, permissionCalls: 0, constraints: null };
  function makeStream() {
    const track = { label: "모의 마이크", stopped: false, onended: null, onmute: null, stop() { this.stopped = true; }, getSettings() { return { deviceId: "mic1", sampleRate: 48000, channelCount: 2, echoCancellation: false, noiseSuppression: false, autoGainControl: false }; } };
    const stream = { track, getTracks: () => [track], getAudioTracks: () => [track] };
    streams.push(stream); return stream;
  }
  const devices = [{ kind: "audioinput", deviceId: "mic1", label: "모의 마이크" }];
  const document = { hidden: false, title: "", getElementById(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); }, createElement: () => new Element(), addEventListener(name, handler) { listeners[name] = handler; } };
  const mediaDevices = { enumerateDevices: async () => devices, getUserMedia(constraints) { state.permissionCalls++; state.constraints = constraints; if (state.deferredPermission) return state.deferredPermission.promise; return Promise.resolve(makeStream()); }, addEventListener(name, handler) { mediaListeners[name] = handler; } };
  class Node { constructor() { this.disconnected = false; } connect() {} disconnect() { this.disconnected = true; } }
  class Context {
    constructor() { this.sampleRate = state.audioRate; this.state = "suspended"; this.destination = {}; this.audioWorklet = { addModule: async url => { assert.equal(url, "/microphone_worklet.js"); } }; contexts.push(this); }
    createMediaStreamSource() { return new Node(); }
    createGain() { const node = new Node(); node.gain = { value: 1 }; return node; }
    async resume() { this.state = "running"; }
    async close() { this.state = "closed"; }
  }
  class Worklet extends Node { constructor() { super(); this.port = { onmessage: null }; workers.push(this); } }
  const info = { wakeword: "보라야", sample_rate: 16000, frame_samples: 1280, warmup_s: 2, threshold: .986, token: "test-local-token", models: [{ id: "head1", label: "모델 1", kind: "single" }, { id: "_mean", label: "평균 앙상블", kind: "ensemble" }] };
  const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
  const fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (url === "/api/info") return json(info);
    assert.equal(options.headers["X-Koww-Token"], info.token);
    if (url === "/api/start") {
      if (state.deferredStart) return state.deferredStart.promise;
      return json({ session_id: `session${calls.filter(call => call.url === "/api/start").length}`, models: info.models });
    }
    if (url === "/api/stop") return json({ stopped: true });
    if (url === "/api/frame") {
      assert.equal(options.body.byteLength, 2560);
      if (state.failFrame) return json({ error: "모의 서버 오류" }, 500);
      if (state.holdFrame) return new Promise((_, reject) => options.signal.addEventListener("abort", () => { const error = new Error("aborted"); error.name = "AbortError"; reject(error); }, { once: true }));
      const sequence = Number(options.headers["X-Koww-Sequence"]);
      return json({ session_id: options.headers["X-Koww-Session"], sequence, time_s: (sequence + 1) * .08, eligible: sequence >= 24, rms: .02, peak: .3, responses: info.models.map(model => ({ ...model, score: .2, max_score: .3, event: false, events: 0, last_event_s: null })) });
    }
    throw new Error(`Unexpected URL ${url}`);
  };
  const context = vm.createContext({ document, navigator: { mediaDevices }, fetch, Option, AudioContext: Context, AudioWorkletNode: Worklet, AbortController, performance: { now: () => clock }, setTimeout(callback, duration) { const id = ++nextTimer; timers.set(id, { callback, duration }); return id; }, clearTimeout(id) { timers.delete(id); }, setInterval(callback, duration) { const id = ++nextTimer; intervals.set(id, { callback, duration }); return id; }, clearInterval(id) { intervals.delete(id); }, console });
  context.window = context;
  context.isSecureContext = true;
  context.addEventListener = (name, handler) => { listeners[name] = handler; };
  vm.runInContext(fs.readFileSync(path.join(resources, "microphone.js"), "utf8"), context);
  return { state, elements, listeners, mediaListeners, contexts, workers, calls, streams, intervals, devices, makeStream, info,
    click(id) { return elements.get(id).handlers.click(); },
    frame() { const worker = workers.at(-1); const pcm = vm.runInContext("new ArrayBuffer(2560)", context); worker.port.onmessage?.({ data: { type: "frame", pcm } }); },
    tick(ms) { clock += ms; for (const timer of [...intervals.values()]) timer.callback(); },
    hidden() { document.hidden = true; listeners.visibilitychange(); },
    response: json,
  };
}
function deferred() { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }

(async () => {
  await test("PCM16 rounding matches half-to-even and signed endpoints", () => {
    const result = worklet();
    const values = [1, -1, .5, -.5, .5 / 32768, 1.5 / 32768, 2.5 / 32768, -.5 / 32768, -1.5 / 32768, -2.5 / 32768];
    const expected = [32767, -32768, 16384, -16384, 0, 2, 2, 0, -2, -2];
    for (let i = 0; i < values.length; i++) assert.equal(vm.runInContext(`pcm16Value(${values[i]})`, result.context), expected[i]);
  });
  await test("10 render quanta form one exact 1280-sample little-endian mono frame; output is silent", () => {
    const { instance, messages } = worklet();
    for (let quantum = 0; quantum < 10; quantum++) {
      const output = new Float32Array(128).fill(.8);
      instance.process([[new Float32Array(128).fill(.5), new Float32Array(128)]], [[output]]);
      assert.ok(output.every(value => value === 0));
      assert.equal(messages.length, quantum === 9 ? 1 : 0);
    }
    const view = new DataView(messages[0].pcm);
    assert.equal(view.byteLength, 2560);
    for (let i = 0; i < 1280; i++) assert.equal(view.getInt16(i * 2, true), 8192);
  });
  await test("nonfinite, out-of-range and uneven channel inputs stop worklet without fabricating audio", () => {
    for (const channels of [[new Float32Array([NaN])], [new Float32Array([1.1])], [new Float32Array(2), new Float32Array(1)]]) {
      const { instance, messages } = worklet();
      instance.process([channels], []); instance.process([[new Float32Array(1280)]], []);
      assert.equal(messages.length, 1); assert.equal(messages[0].type, "error");
    }
  });
  await test("initial page load never requests microphone permission", async () => {
    const b = browser(); await flush();
    assert.equal(b.state.permissionCalls, 0); assert.equal(b.elements.get("start").disabled, false);
    assert.equal(b.elements.get("wakeword").textContent, "‘보라야’");
  });
  await test("start submits requested DSP settings; frames are authenticated, sequential, and stop releases capture", async () => {
    const b = browser(); await flush(); await b.click("start");
    assert.equal(b.state.constraints.audio.noiseSuppression.ideal, false);
    assert.equal(b.state.constraints.audio.echoCancellation.ideal, false);
    assert.equal(b.state.constraints.audio.autoGainControl.ideal, false);
    assert.equal(b.contexts[0].sampleRate, 16000);
    b.frame(); b.frame(); await flush();
    assert.deepEqual(b.calls.filter(call => call.url === "/api/frame").map(call => call.options.headers["X-Koww-Sequence"]), ["0", "1"]);
    assert.equal(b.elements.get("elapsed").textContent, "0.2초");
    b.click("stop"); await flush();
    assert.equal(b.streams[0].track.stopped, true); assert.equal(b.contexts[0].state, "closed");
    assert.equal(b.intervals.size, 0); assert.equal(b.workers[0].port.onmessage, null);
    assert.equal(b.calls.filter(call => call.url === "/api/stop").length, 1);
  });
  await test("stop while permission is pending disposes the late stream and never opens server session", async () => {
    const b = browser(); await flush(); b.state.deferredPermission = deferred();
    const start = b.click("start"); b.click("stop");
    const stream = b.makeStream(); b.state.deferredPermission.resolve(stream);
    await start; await flush();
    assert.equal(stream.track.stopped, true); assert.equal(b.calls.some(call => call.url === "/api/start"), false);
  });
  await test("stop while start response is pending closes the late server session", async () => {
    const b = browser(); await flush(); b.state.deferredStart = deferred();
    const start = b.click("start"); await flush(); b.click("stop");
    b.state.deferredStart.resolve(b.response({ session_id: "late-session", models: b.info.models }));
    await start; await flush();
    assert.equal(b.streams[0].track.stopped, true);
    assert.equal(JSON.parse(b.calls.find(call => call.url === "/api/stop").options.body).session_id, "late-session");
  });
  await test("an unsupported actual AudioContext sample rate is rejected before server start", async () => {
    const b = browser(); await flush(); b.state.audioRate = 48000;
    await b.click("start"); await flush();
    assert.equal(b.streams[0].track.stopped, true); assert.equal(b.calls.some(call => call.url === "/api/start"), false);
    assert.match(b.elements.get("statusText").textContent, /16 kHz/);
  });
  await test("backlog aborts rather than dropping or reusing audio", async () => {
    const b = browser(); await flush(); await b.click("start"); b.state.holdFrame = true;
    for (let i = 0; i < 13; i++) b.frame(); await flush();
    assert.equal(b.streams[0].track.stopped, true);
    assert.match(b.elements.get("statusText").textContent, /入力速度|입력 속도/);
    assert.equal(b.calls.filter(call => call.url === "/api/frame").length, 1);
  });
  await test("track mute, track ended, hidden page, and context interruption all stop capture", async () => {
    for (const cause of ["mute", "ended", "hidden", "context"]) {
      const b = browser(); await flush(); await b.click("start");
      if (cause === "mute") b.streams[0].track.onmute();
      if (cause === "ended") b.streams[0].track.onended();
      if (cause === "hidden") b.hidden();
      if (cause === "context") { b.contexts[0].state = "suspended"; b.contexts[0].onstatechange(); }
      await flush(); assert.equal(b.streams[0].track.stopped, true); assert.equal(b.intervals.size, 0);
    }
  });
  await test("three seconds without worklet audio stops an otherwise live stream", async () => {
    const b = browser(); await flush(); await b.click("start");
    b.tick(3001); await flush();
    assert.equal(b.streams[0].track.stopped, true); assert.match(b.elements.get("statusText").textContent, /3초/);
  });
  await test("valid silence frames keep the input watchdog alive", async () => {
    const b = browser(); await flush(); await b.click("start");
    b.tick(2900); b.frame(); await flush(); b.tick(2900);
    assert.equal(b.streams[0].track.stopped, false); b.click("stop"); await flush();
  });
  await test("backend errors release microphone and retain the actual error", async () => {
    const b = browser(); await flush(); await b.click("start"); b.state.failFrame = true;
    b.frame(); await flush();
    assert.equal(b.streams[0].track.stopped, true); assert.equal(b.elements.get("statusText").textContent, "모의 서버 오류");
  });
  await test("a new start uses a fresh session and resets sequence and metrics", async () => {
    const b = browser(); await flush(); await b.click("start"); b.frame(); await flush(); b.click("stop"); await flush();
    await b.click("start"); b.frame(); await flush();
    const frames = b.calls.filter(call => call.url === "/api/frame");
    assert.equal(frames[0].options.headers["X-Koww-Sequence"], "0"); assert.equal(frames[1].options.headers["X-Koww-Sequence"], "0");
    assert.notEqual(frames[0].options.headers["X-Koww-Session"], frames[1].options.headers["X-Koww-Session"]);
    b.click("stop"); await flush();
  });
  const files = ["microphone.html", "microphone.css", "microphone.js", "microphone_worklet.js"];
  const crypto = require("node:crypto");
  const report = { state: "passed", tests: tests.length, names: tests, actual_microphone_opened: false, actual_browser_interaction: false, files: Object.fromEntries(files.map(name => [name, crypto.createHash("sha256").update(fs.readFileSync(path.join(resources, name))).digest("hex")])) };
  if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify(report, null, 2) + "\n");
  process.stdout.write(JSON.stringify(report, null, 2) + "\n");
})().catch(error => { console.error(error); process.exitCode = 1; });
