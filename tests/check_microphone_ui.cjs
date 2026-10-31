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

function browser(config = {}) {
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
  const state = { deferredPermission: null, deferredStart: null, deferredInfo: null, startRuntime: null, failInfo: false, permissionError: null, enumerationError: null, hideDeviceNames: false, enumerations: [], failFrame: false, holdFrame: false, audioRate: 16000, permissionCalls: 0, constraints: null, ...config };
  function makeStream() {
    const track = { label: "모의 마이크", stopped: false, onended: null, onmute: null, stop() { this.stopped = true; }, getSettings() { return { deviceId: "mic1", sampleRate: 48000, channelCount: 2, echoCancellation: false, noiseSuppression: false, autoGainControl: false }; } };
    const tracks = [track];
    const stream = { track, tracks, getTracks: () => tracks, getAudioTracks: () => tracks };
    streams.push(stream); return stream;
  }
  const devices = [{ kind: "audioinput", deviceId: "mic1", label: "모의 마이크" }];
  const document = { hidden: false, title: "", getElementById(id) { if (!elements.has(id)) elements.set(id, new Element(id)); return elements.get(id); }, createElement: () => new Element(), addEventListener(name, handler) { listeners[name] = handler; } };
  const pageConfig = config.pageConfig || {};
  document.getElementById("koww-page-config").textContent = JSON.stringify(pageConfig);
  const mediaDevices = { enumerateDevices: async () => {
    if (state.enumerationError) throw state.enumerationError;
    const active = streams.some(stream => !stream.track.stopped);
    state.enumerations.push({ active });
    return devices.map(device => ({ ...device, label: state.hideDeviceNames && !active ? "" : device.label }));
  }, getUserMedia(constraints) { state.permissionCalls++; state.constraints = constraints; if (state.permissionError) return Promise.reject(state.permissionError); if (state.deferredPermission) return state.deferredPermission.promise; return Promise.resolve(makeStream()); }, addEventListener(name, handler) { mediaListeners[name] = handler; } };
  class Node { constructor() { this.disconnected = false; } connect() {} disconnect() { this.disconnected = true; } }
  class Context {
    constructor() { this.sampleRate = state.audioRate; this.state = "suspended"; this.destination = {}; this.audioWorklet = { addModule: async url => { assert.equal(url, pageConfig.worklet_url || "/microphone_worklet.js"); } }; contexts.push(this); }
    createMediaStreamSource() { return new Node(); }
    createGain() { const node = new Node(); node.gain = { value: 1 }; return node; }
    async resume() { this.state = "running"; }
    async close() { this.state = "closed"; }
  }
  class Worklet extends Node { constructor() { super(); this.port = { onmessage: null }; workers.push(this); } }
  const defaultRuntime = { threshold: .9862500429153442, consecutive_frames: 2, ...config.defaultRuntime };
  const info = { wakeword: "보라야", sample_rate: 16000, frame_samples: 1280, warmup_s: 2, ...defaultRuntime, default_runtime: { ...defaultRuntime }, token: "test-local-token", models: [
    { id: "head1", label: "모델 1", kind: "single" }, { id: "head2", label: "모델 2", kind: "single" },
    { id: "head3", label: "모델 3", kind: "single" }, { id: "_mean", label: "평균 앙상블", kind: "ensemble" }
  ] };
  if (config.recommendedRuntimes) info.recommended_runtimes = config.recommendedRuntimes;
  if (config.recommendations) info.runtime_recommendations = config.recommendations;
  const json = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
  const fetch = async (url, options = {}) => {
    calls.push({ url, options });
    if (pageConfig.api_base) { assert.ok(url.startsWith(pageConfig.api_base + "/api/")); url = url.slice(pageConfig.api_base.length); }
    if (url === "/api/info") {
      if (state.failInfo) return json({ error: "모의 모델 연결 오류" }, 500);
      if (state.deferredInfo) return new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => { const error = new Error("aborted"); error.name = "AbortError"; reject(error); }, { once: true });
        state.deferredInfo.promise.then(resolve, reject);
      });
      return json(info);
    }
    assert.equal(options.headers["X-Koww-Token"], info.token);
    if (url === "/api/start") {
      if (state.deferredStart) return state.deferredStart.promise;
      if (JSON.parse(options.body).mode === "recommended") return json({ session_id: "recommended-session", models: info.models,
        runtimes: state.startRuntimes || info.recommended_runtimes });
      return json({ session_id: `session${calls.filter(call => call.url === "/api/start").length}`, models: info.models,
        runtime: state.startRuntime || { ...defaultRuntime, ...JSON.parse(options.body) } });
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
  return { state, elements, listeners, mediaListeners, contexts, workers, calls, streams, timers, intervals, devices, makeStream, info,
    click(id) { return elements.get(id).handlers.click({ type: "click", target: elements.get(id) }); },
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
  await test("all four models are visible and a readable default retains its exact original threshold on start", async () => {
    const b = browser(); await flush();
    assert.equal(b.elements.get("models").children.length, 4);
    assert.equal(b.elements.get("trial-threshold").value, "0.98625");
    assert.equal(b.elements.get("trial-consecutive").value, "2");
    assert.equal(b.elements.get("trial-threshold").disabled, false);
    await b.click("start"); await flush();
    const payload = JSON.parse(b.calls.find(call => call.url === "/api/start").options.body);
    assert.deepEqual(payload, { threshold: b.info.threshold, consecutive_frames: 2 });
    assert.equal(b.elements.get("models").children.length, 4);
    assert.equal(b.elements.get("trial-threshold").disabled, true);
    assert.match(b.elements.get("threshold").textContent, /0\.98625.*2회/);
    b.click("stop"); await flush();
  });
  const recommendations = {head1: {threshold: .6, consecutive_frames: 1}, head2: {threshold: .7, consecutive_frames: 2},
    head3: {threshold: .8, consecutive_frames: 3}, _mean: {threshold: .9, consecutive_frames: 2}};
  await test("complete per-model recommendations are the initial microphone mode without hiding any model", async () => {
    const b = browser({defaultRuntime: {threshold: .5, consecutive_frames: 1}, recommendedRuntimes: recommendations}); await flush();
    assert.equal(b.elements.get("trial-mode").value, "recommended");
    assert.equal(b.elements.get("trial-threshold").disabled, true);
    assert.equal(b.elements.get("trial-mode").disabled, false);
    const rows = b.elements.get("models").children;
    assert.equal(rows.length, 4);
    assert.deepEqual(rows.map(row => row.children.slice(5).map(cell => cell.textContent)), [["0.6", "1회"], ["0.7", "2회"], ["0.8", "3회"], ["0.9", "2회"]]);
    await b.click("start"); await flush();
    assert.deepEqual(JSON.parse(b.calls.find(call => call.url === "/api/start").options.body), {mode: "recommended"});
    assert.equal(b.elements.get("trial-mode").disabled, true);
    assert.equal(b.elements.get("threshold").textContent, "모델별 추천값 적용");
    b.frame(); await flush(); assert.equal(b.elements.get("models").children.length, 4);
    b.click("stop"); await flush();
  });
  await test("reset leaves recommendation mode and restores the project's exact starting settings", async () => {
    const b = browser({defaultRuntime: {threshold: .5, consecutive_frames: 1}, recommendedRuntimes: recommendations}); await flush();
    b.click("reset-detection");
    assert.equal(b.elements.get("trial-mode").value, "manual");
    assert.equal(b.elements.get("trial-threshold").value, "0.5");
    assert.equal(b.elements.get("trial-consecutive").value, "1");
    assert.equal(b.elements.get("trial-threshold").disabled, false);
    await b.click("start"); await flush();
    assert.deepEqual(JSON.parse(b.calls.find(call => call.url === "/api/start").options.body), {threshold: .5, consecutive_frames: 1});
    b.click("stop"); await flush();
  });
  await test("manual and per-model trial modes can switch without saving or opening a microphone", async () => {
    const b = browser({recommendedRuntimes: recommendations}); await flush();
    b.elements.get("trial-mode").value = "manual"; b.elements.get("trial-mode").handlers.change();
    assert.equal(b.elements.get("trial-threshold").disabled, false);
    assert.equal(b.elements.get("recommendationHint").hidden, true);
    b.elements.get("trial-mode").value = "recommended"; b.elements.get("trial-mode").handlers.change();
    assert.equal(b.elements.get("trial-threshold").disabled, true);
    assert.equal(b.elements.get("recommendationHint").hidden, false);
    assert.equal(b.state.permissionCalls, 0);
    assert.equal(b.calls.length, 1);
  });
  await test("incomplete or invalid recommendations never partially apply to some microphone models", async () => {
    for (const value of [{head1: recommendations.head1}, {...recommendations, head2: {threshold: 1, consecutive_frames: 2}}, {...recommendations, _mean: {threshold: .6, consecutive_frames: 0}}]) {
      const b = browser({recommendedRuntimes: value}); await flush();
      assert.equal(b.elements.get("trial-mode").value, "manual");
      assert.equal(b.elements.get("trial-mode").disabled, true);
      assert.equal(b.elements.get("trial-threshold").disabled, false);
    }
  });
  await test("a recommendation changed by the server is rejected before microphone frames are sent", async () => {
    const b = browser({recommendedRuntimes: recommendations, startRuntimes: {...recommendations, head2: {threshold: .8, consecutive_frames: 2}}}); await flush();
    await b.click("start"); await flush();
    assert.match(b.elements.get("statusText").textContent, /시험 감지 설정을 확인할 수 없습니다/);
    assert.equal(b.calls.some(call => call.url === "/api/frame"), false);
    assert.equal(b.streams[0].track.stopped, true);
    assert.equal(b.calls.some(call => call.url === "/api/stop"), true);
  });
  await test("target-miss and cache-only recommendations remain explicitly qualified", async () => {
    const b = browser({recommendedRuntimes: recommendations, recommendations: {candidates: [{id: "head1", passed: false, deployment_backend_verified: false}]}}); await flush();
    assert.match(b.elements.get("recommendationHint").textContent, /목표 미달/);
    assert.match(b.elements.get("recommendationHint").textContent, /저장 점수 기준/);
  });
  await test("a model without a protected recommendation clearly keeps its original gate", async () => {
    const original = {threshold: .9862500429153442, consecutive_frames: 2};
    const b = browser({recommendedRuntimes: {...recommendations, head2: original}, recommendations: {candidates: [
      {id: "head1", recommendation_available: true, deployment_backend_verified: false},
      {id: "head2", recommendation_available: false, original_runtime: original, passed: false, deployment_backend_verified: false}
    ]}}); await flush();
    assert.equal(b.elements.get("trial-mode").value, "recommended");
    assert.match(b.elements.get("recommendationHint").textContent, /모델 2: 추천 기준을 만족하는 조합이 없어 기존 설정을 유지/);
    assert.doesNotMatch(b.elements.get("recommendationHint").textContent, /가장 나은 값을 사용/);
    assert.equal(b.elements.get("threshold").textContent, "모델별 추천·기존값 적용");
    await b.click("start"); await flush();
    assert.equal(b.elements.get("models").children.length, 4);
    assert.equal(b.elements.get("models").children[1].children[5].textContent, "0.98625");
    b.click("stop"); await flush();
  });
  await test("near-one defaults stay valid on display, reset and explicit trial start", async () => {
    for (const threshold of [.9999999, .9999999999999999]) {
      const b = browser({ defaultRuntime: { threshold } }); await flush();
      assert.equal(b.elements.get("trial-threshold").value, String(threshold));
      b.elements.get("trial-threshold").value = "0.9";
      b.elements.get("trial-threshold").handlers.input();
      b.click("reset-detection");
      assert.equal(b.elements.get("trial-threshold").value, String(threshold));
      await b.click("start"); await flush();
      assert.equal(JSON.parse(b.calls.find(call => call.url === "/api/start").options.body).threshold, threshold);
      assert.equal(b.elements.get("models").children.length, 4);
      b.click("stop"); await flush();
    }
  });
  await test("trial settings are sent only on explicit start and the server's applied settings label the responses", async () => {
    const b = browser(); await flush(); const original = JSON.stringify(b.info);
    b.elements.get("trial-threshold").value = "0.9";
    b.elements.get("trial-threshold").handlers.input();
    b.elements.get("trial-consecutive").value = "3";
    assert.equal(b.calls.length, 1); assert.equal(b.state.permissionCalls, 0);
    await b.click("start"); await flush();
    assert.deepEqual(JSON.parse(b.calls.find(call => call.url === "/api/start").options.body), { threshold: .9, consecutive_frames: 3 });
    assert.match(b.elements.get("threshold").textContent, /0\.9.*3회/);
    assert.equal(JSON.stringify(b.info), original);
    b.frame(); await flush(); assert.equal(b.elements.get("models").children.length, 4);
    b.click("stop"); await flush();
    assert.equal(b.elements.get("trial-threshold").disabled, false);
    assert.equal(b.elements.get("trial-consecutive").disabled, false);
  });
  await test("invalid trial settings never open the microphone or start a server session", async () => {
    for (const [id, values] of [["trial-threshold", ["", "0", "1", "NaN", "Infinity"]], ["trial-consecutive", ["", "0", "11", "1.5"]]]) {
      for (const value of values) {
        const b = browser(); await flush(); b.elements.get(id).value = value;
        await b.click("start"); await flush();
        assert.equal(b.state.permissionCalls, 0); assert.equal(b.calls.length, 1);
        assert.equal(b.elements.get("statusHeading").textContent, "감지 설정을 확인하세요");
        assert.equal(b.elements.get(id).disabled, false);
      }
    }
  });
  await test("settings stay frozen while permission is pending and reset cannot change a running trial", async () => {
    const b = browser(); await flush(); b.state.deferredPermission = deferred();
    const started = b.click("start");
    for (const id of ["trial-threshold", "trial-consecutive", "reset-detection"]) assert.equal(b.elements.get(id).disabled, true);
    b.elements.get("trial-threshold").value = "0.8"; b.click("reset-detection");
    assert.equal(b.elements.get("trial-threshold").value, "0.8");
    b.state.deferredPermission.resolve(b.makeStream()); await started; await flush();
    assert.equal(JSON.parse(b.calls.find(call => call.url === "/api/start").options.body).threshold, b.info.threshold);
    b.click("stop"); await flush();
  });
  await test("reset restores exact defaults for the next trial without starting or saving anything", async () => {
    const b = browser(); await flush();
    b.elements.get("trial-threshold").value = "0.9"; b.elements.get("trial-threshold").handlers.input();
    b.elements.get("trial-consecutive").value = "5";
    b.click("reset-detection");
    assert.equal(b.elements.get("trial-threshold").value, "0.98625");
    assert.equal(b.elements.get("trial-consecutive").value, "2");
    assert.equal(b.calls.length, 1); assert.equal(b.state.permissionCalls, 0);
    await b.click("start"); await flush();
    assert.equal(JSON.parse(b.calls.find(call => call.url === "/api/start").options.body).threshold, b.info.threshold);
    b.click("stop"); await flush();
  });
  await test("an explicit edit of the rounded value uses that value rather than silently restoring the full default", async () => {
    const b = browser(); await flush(); b.elements.get("trial-threshold").handlers.input();
    await b.click("start"); await flush();
    assert.equal(JSON.parse(b.calls.find(call => call.url === "/api/start").options.body).threshold, .98625);
    b.click("stop"); await flush();
  });
  await test("a server that does not apply requested trial settings is stopped without sending microphone frames", async () => {
    const b = browser({ startRuntime: { threshold: .5, consecutive_frames: 1 } }); await flush();
    await b.click("start"); await flush();
    assert.match(b.elements.get("statusText").textContent, /시험 감지 설정을 확인할 수 없습니다/);
    assert.equal(b.streams[0].track.stopped, true);
    assert.equal(b.calls.filter(call => call.url === "/api/stop").length, 1);
    assert.equal(b.calls.some(call => call.url === "/api/frame"), false);
  });
  await test("model connection recovery fills defaults even while a separate microphone permission probe is pending", async () => {
    const b = browser({ failInfo: true }); await flush();
    b.state.deferredPermission = deferred(); const connecting = b.click("connect");
    b.state.failInfo = false; await b.click("retry"); await flush();
    assert.equal(b.elements.get("trial-threshold").value, "0.98625");
    assert.equal(b.elements.get("trial-threshold").disabled, true);
    b.state.deferredPermission.resolve(b.makeStream()); await connecting; await flush();
    assert.equal(b.elements.get("trial-threshold").disabled, false);
    assert.equal(b.calls.some(call => call.url === "/api/start"), false);
  });
  await test("nested Studio config routes info, start, PCM frames, stop and worklet without rewriting JavaScript", async () => {
    const prefix = "/mic/bora", b = browser({ pageConfig: { api_base: prefix, worklet_url: prefix + "/microphone_worklet.js" } });
    await flush(); await b.click("start"); await flush();
    b.frame(); await flush(); await b.click("stop"); await flush();
    assert.deepEqual(b.calls.map(call => call.url), [prefix + "/api/info", prefix + "/api/start", prefix + "/api/frame", prefix + "/api/stop"]);
    const frame = b.calls.find(call => call.url.endsWith("/frame"));
    assert.equal(frame.options.body.byteLength, 2560);
    assert.equal(frame.options.headers["X-Koww-Token"], b.info.token);
    assert.equal(frame.options.headers["X-Koww-Sequence"], "0");
    assert.equal(b.streams.every(stream => stream.track.stopped), true);
  });
  await test("explicit microphone connection reveals device names while capture is active then releases every probe track", async () => {
    const b = browser({ hideDeviceNames: true }); await flush();
    assert.match(b.elements.get("microphone").options[1].textContent, /권한 허용 후/);
    b.state.deferredPermission = deferred();
    const connecting = b.click("connect"), stream = b.makeStream();
    const extra = { stopped: false, stop() { this.stopped = true; } };
    stream.tracks.push(extra); b.state.deferredPermission.resolve(stream);
    await connecting; await flush();
    assert.equal(b.state.permissionCalls, 1); assert.equal(b.state.constraints.audio, true);
    assert.equal(b.state.enumerations.at(-1).active, true);
    assert.equal(b.elements.get("microphone").options[1].textContent, "모의 마이크");
    assert.equal(b.streams.every(stream => stream.track.stopped), true);
    assert.equal(extra.stopped, true);
    assert.equal(b.calls.some(call => call.url === "/api/start"), false);
    assert.equal(b.elements.get("start").disabled, false);
  });
  await test("device discovery remains available after a model connection failure and retry recovers without reloading", async () => {
    const b = browser({ failInfo: true }); await flush();
    assert.equal(b.elements.get("connection").textContent, "모델 연결 실패");
    assert.match(b.elements.get("statusText").textContent, /모의 모델 연결 오류/);
    assert.equal(b.elements.get("retry").hidden, false); assert.equal(b.elements.get("start").disabled, true);
    assert.equal(b.elements.get("connect").disabled, false);
    await b.click("connect"); await flush();
    assert.equal(b.streams[0].track.stopped, true);
    assert.equal(b.elements.get("connection").textContent, "모델 연결 실패");
    b.state.failInfo = false; await b.click("retry"); await flush();
    assert.equal(b.elements.get("retry").hidden, true); assert.equal(b.elements.get("start").disabled, false);
    assert.equal(b.calls.filter(call => call.url === "/api/info").length, 2);
  });
  await test("cold model initialization has a 30-second limit and rejects duplicate retry requests", async () => {
    const pending = deferred(), b = browser({ deferredInfo: pending }); await flush();
    const timeout = [...b.timers.values()].find(timer => timer.duration === 30000);
    assert.ok(timeout); assert.equal(b.elements.get("retry").disabled, true);
    await b.click("retry"); assert.equal(b.calls.filter(call => call.url === "/api/info").length, 1);
    timeout.callback(); await flush();
    assert.match(b.elements.get("statusText").textContent, /30초/);
    assert.equal(b.elements.get("connection").textContent, "모델 연결 실패");
    assert.equal(b.elements.get("retry").disabled, false); assert.equal(b.state.permissionCalls, 0);
    b.state.deferredInfo = null; await b.click("retry"); await flush();
    pending.resolve(b.response(b.info)); await flush();
    assert.equal(b.elements.get("start").disabled, false);
  });
  await test("permission denial is a device error and does not invalidate the ready model", async () => {
    const error = new Error("permission denied"); error.name = "NotAllowedError";
    const b = browser({ permissionError: error }); await flush(); await b.click("connect");
    assert.match(b.elements.get("deviceHint").textContent, /마이크 권한이 거부/);
    assert.equal(b.elements.get("connection").textContent, "모델 준비됨");
    assert.equal(b.elements.get("start").disabled, false); assert.equal(b.streams.length, 0);
  });
  await test("a failed device enumeration releases the permission probe and permits another connection", async () => {
    const b = browser(); await flush(); b.state.enumerationError = new Error("장치 목록 오류");
    await b.click("connect"); await flush();
    assert.equal(b.streams[0].track.stopped, true);
    assert.match(b.elements.get("deviceHint").textContent, /장치 목록 오류/);
    assert.equal(b.elements.get("connect").disabled, false);
    b.state.enumerationError = null; await b.click("connect"); await flush();
    assert.equal(b.elements.get("microphone").options[1].textContent, "모의 마이크");
    assert.equal(b.streams.every(stream => stream.track.stopped), true);
  });
  await test("click and devicechange event objects do not suppress ordinary device refresh", async () => {
    const b = browser(); await flush();
    b.devices.push({ kind: "audioinput", deviceId: "mic2", label: "두 번째 마이크" });
    await b.click("refresh");
    assert.equal(b.elements.get("microphone").options[2].textContent, "두 번째 마이크");
    b.devices.pop(); b.mediaListeners.devicechange({ type: "devicechange" }); await flush();
    assert.equal(b.elements.get("microphone").options.length, 2);
  });
  await test("cancelling a permission probe disposes a late stream without changing the cancellation hint", async () => {
    const b = browser(); await flush(); b.state.deferredPermission = deferred();
    const connecting = b.click("connect");
    assert.equal(b.elements.get("stop").disabled, false); assert.equal(b.elements.get("start").disabled, true);
    assert.match(b.elements.get("deviceHint").textContent, /허용/);
    b.click("stop"); const stream = b.makeStream(); b.state.deferredPermission.resolve(stream);
    await connecting; await flush();
    assert.equal(stream.track.stopped, true); assert.match(b.elements.get("deviceHint").textContent, /취소/);
    assert.equal(b.calls.some(call => call.url === "/api/start"), false);
  });
  await test("leaving the page cancels device discovery and releases a delayed microphone", async () => {
    const b = browser(); await flush(); b.state.deferredPermission = deferred();
    const connecting = b.click("connect"); b.listeners.pagehide();
    const stream = b.makeStream(); b.state.deferredPermission.resolve(stream);
    await connecting; await flush(); assert.equal(stream.track.stopped, true);
    assert.equal(b.elements.get("stop").disabled, true);
  });
  await test("a cancelled late permission response cannot clear a newer device connection", async () => {
    const b = browser(); await flush();
    const oldPermission = deferred(), newPermission = deferred();
    b.state.deferredPermission = oldPermission; const oldConnection = b.click("connect"); b.click("stop");
    b.state.deferredPermission = newPermission; const newConnection = b.click("connect");
    const oldStream = b.makeStream(); oldPermission.resolve(oldStream); await oldConnection; await flush();
    assert.equal(oldStream.track.stopped, true); assert.equal(b.elements.get("connect").disabled, true);
    assert.match(b.elements.get("deviceHint").textContent, /허용/);
    const newStream = b.makeStream(); newPermission.resolve(newStream); await newConnection; await flush();
    assert.equal(newStream.track.stopped, true); assert.equal(b.elements.get("start").disabled, false);
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
