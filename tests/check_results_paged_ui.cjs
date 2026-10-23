"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_paged.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_paged.html"), "utf8");
const flush = async () => { for (let count = 0; count < 8; count++) await new Promise(resolve => setImmediate(resolve)); };
const tests = [];
async function test(name, run) { await run(); tests.push(name); }
function browser(tuples, {deferPages = false, pageFailure = false, indexPath = "viewer-index.json", report = {wakeword: "보라야", models: []}} = {}) {
  const elements = new Map(), calls = [], deferred = new Map();
  class Element {
    constructor(tag = "div") { this.tagName = tag; this.children = []; this.value = ""; this.textContent = ""; this.disabled = false; this.src = ""; this.plays = 0; this.pauses = 0; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    get options() { return this.children; }
    pause() { this.pauses++; }
    async play() { this.plays++; }
    removeAttribute(name) { this[name] = ""; }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1], new Element());
  for (const id of ["summary", "conditions", "responses"]) elements.get(id).tBodies = [new Element("tbody")];
  for (const value of ["none", "moderate", "strong"]) { const option = new Element("option"); option.value = value; elements.get("reverb").append(option); }
  elements.get("reverb").value = "none"; elements.get("category").value = "all";
  const row = (tuple, sourceIndex) => ({sample_id: tuple[0], path: `audio/${sourceIndex}.wav`, sha256: `sha${sourceIndex}`, kind: tuple[1], category: tuple[2], snr_db: tuple[3], reverb_condition: tuple[4], original_id: tuple[5], original_order: tuple[6], duration_s: tuple[7], calls: tuple[1] === "positive" ? [{start_s: 2, end_s: 3}] : [], noise_eval_label: "시험 배경", results: {mean: {success: true, false_activations: 0, max_score: 0.999, clean_score_drop: 0.01, latency_s: 0.08}}});
  const page = number => ({start: number * 128, clips: tuples.slice(number * 128, (number + 1) * 128).map((tuple, offset) => row(tuple, number * 128 + offset))});
  const context = vm.createContext({document: {getElementById: id => elements.get(id), createElement: tag => new Element(tag)}, console, evaluationBackgroundTypeText: clip => clip.noise_eval_label || "", fetch: async url => {
    calls.push(url);
    if (url === indexPath) return {ok: true, json: async () => ({schema_version: 1, scope_text: "정확한 평가 범위", noise_coverage_text: "전체 음원 사용", clips: tuples})};
    const number = Number(url.match(/page-(\d+)\.json/)[1]);
    if (pageFailure) return {ok: false, status: 503};
    if (deferPages) return new Promise(resolve => deferred.set(number, () => resolve({ok: true, json: async () => page(number)})));
    return {ok: true, json: async () => page(number)};
  }});
  context.window = context; context.location = new URL("http://127.0.0.1/results/bora/"); context.URL = URL; context.KOWW_RESULTS_TEST = true;
  context.KOWW_RESULTS = {report, viewer: {schema_version: 1, index_path: indexPath, page_size: 128}};
  vm.runInContext(script, context);
  return {elements, calls, deferred, helper: context.KowwResults};
}
const tuple = (index, overrides = {}) => { const value = [`clip_${index}`, "positive", ["environment", "music", "competing_speech"][index % 3], 10, "none", `original_${Math.floor(index / 3)}`, Math.floor(index / 3) + 1, 4]; for (const [key, field] of Object.entries(overrides)) value[Number(key)] = field; return value; };

(async () => {
  await test("112137 clips load one detail page and keep selectors bounded", async () => {
    const tuples = Array.from({length: 112137}, (_, index) => tuple(index));
    for (let index = 110000; index < tuples.length; index++) tuples[index] = tuple(index, {1: "negative", 2: "environment"});
    const b = browser(tuples); await b.helper.boot();
    assert.equal(b.helper.state().queueLength, 110000);
    assert.deepEqual(b.calls, ["viewer-index.json", "http://127.0.0.1/results/bora/clips/page-000000.json"]);
    assert.equal(b.elements.get("item").children.length, 100);
    assert.equal(b.elements.get("extra").children.length, 100);
    assert.equal(b.elements.get("original").children.length, 0);
    assert.equal(b.elements.get("scope").textContent, "정확한 평가 범위");
  });
  await test("continuous playback crosses detail and selector pages without skipping audio", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    await b.elements.get("start").onclick(); await b.helper.move(127, true); await b.elements.get("audio").onended();
    assert.equal(b.helper.state().index, 128); assert.equal(b.elements.get("audio").src, "audio/128.wav");
    assert.equal(b.elements.get("item").children.length, 100);
    assert.equal(b.elements.get("item").value, "128");
    assert.equal(b.elements.get("hash").textContent, "SHA256 sha128");
  });
  await test("only two detail pages are retained with LRU eviction", async () => {
    const b = browser(Array.from({length: 600}, (_, index) => tuple(index))); await b.helper.boot();
    await b.helper.detail(256); await b.helper.detail(128);
    assert.deepEqual(Array.from(b.helper.state().cachePages), [2, 1]);
    await b.helper.detail(0); assert.equal(b.calls.filter(url => url.endsWith("clips/page-000000.json")).length, 2);
    assert.ok(b.helper.state().cachePages.length <= 2);
  });
  await test("concurrent requests for the same page are deduplicated", async () => {
    const b = browser(Array.from({length: 300}, (_, index) => tuple(index)), {deferPages: true});
    const boot = b.helper.boot(); await flush(); b.deferred.get(0)(); await boot;
    const first = b.helper.detail(128), second = b.helper.detail(129); await flush();
    assert.equal(b.calls.filter(url => url.endsWith("clips/page-000001.json")).length, 1);
    b.deferred.get(1)(); await Promise.all([first, second]);
  });
  await test("late page responses cannot replace a newer navigation", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index)), {deferPages: true});
    const boot = b.helper.boot(); await flush(); b.deferred.get(0)(); await boot;
    const old = b.helper.move(128); const newer = b.helper.move(256); await flush();
    b.deferred.get(2)(); await newer; b.deferred.get(1)(); await old;
    assert.equal(b.elements.get("audio").src, "audio/256.wav");
    assert.equal(b.elements.get("clipTitle").textContent, "clip_256 · 음악·전자음");
  });
  await test("filter changes reject stale audio, and all selected conditions remain available", async () => {
    const tuples = Array.from({length: 400}, (_, index) => tuple(index));
    tuples[256][3] = 5; tuples[256][4] = "strong";
    const b = browser(tuples, {deferPages: true}); const boot = b.helper.boot(); await flush(); b.deferred.get(0)(); await boot;
    const old = b.helper.move(128); await flush();
    b.elements.get("snr").value = "5"; b.elements.get("reverb").value = "strong";
    const changed = b.helper.rebuild(); await flush(); b.deferred.get(2)(); await changed; b.deferred.get(1)(); await old;
    assert.equal(b.helper.state().queueLength, 1); assert.equal(b.elements.get("audio").src, "audio/256.wav");
  });
  await test("original number jump navigates the complete queue", async () => {
    const b = browser(Array.from({length: 3000}, (_, index) => tuple(index))); await b.helper.boot();
    b.elements.get("original").value = "900"; await b.elements.get("original-go").onclick();
    assert.equal(b.helper.state().index, 2697); assert.equal(b.elements.get("audio").src, "audio/2697.wav");
    assert.ok(b.elements.get("item").children.length <= 100);
  });
  await test("Clean and negative browsing is paginated and loads only selected detail", async () => {
    const tuples = Array.from({length: 500}, (_, index) => tuple(index, {1: index > 0 ? "negative" : "positive"}));
    const b = browser(tuples); await b.helper.boot(); const before = b.calls.length;
    b.elements.get("extra-next").onclick(); assert.equal(b.calls.length, before);
    assert.equal(b.elements.get("extra").children.length, 100);
    b.elements.get("extra").value = "130"; await b.elements.get("extra").onchange();
    assert.equal(b.helper.state().source, "extra"); assert.equal(b.elements.get("audio").src, "audio/131.wav");
    assert.equal(b.elements.get("responses").tBodies[0].children[0].children[1].textContent, "오호출 0회");
  });
  await test("detail failure stays a recoverable inline error", async () => {
    const b = browser([tuple(0)], {pageFailure: true}); await b.helper.boot();
    assert.match(b.elements.get("playState").textContent, /503/); assert.equal(b.helper.state().continuous, false);
  });
  await test("completed continuous playback stops at the final selected clip", async () => {
    const b = browser([tuple(0), tuple(1)]); await b.helper.boot(); await b.elements.get("start").onclick();
    await b.elements.get("audio").onended(); await b.elements.get("audio").onended();
    assert.equal(b.helper.state().index, 1); assert.equal(b.helper.state().continuous, false);
    assert.equal(b.elements.get("playState").textContent, "선택한 목록을 모두 들었습니다.");
  });
  await test("versioned presentation folder resolves pages beside its own index", async () => {
    const b = browser([tuple(0)], {indexPath: "viewer_v1_fix/viewer-index.json"}); await b.helper.boot();
    assert.deepEqual(b.calls, ["viewer_v1_fix/viewer-index.json", "http://127.0.0.1/results/bora/viewer_v1_fix/clips/page-000000.json"]);
    assert.equal(b.elements.get("audio").src, "audio/0.wav");
  });
  await test("reported model counts and conditional statistics remain unchanged", async () => {
    const report = {wakeword: "보라야", selected_candidate: "member2", models: [{id: "member2", rank: 1, kind: "single", primary: {successes: 42, calls: 45, recall: 42 / 45, latency_p95_s: 0.2}, clean: {recall: 1}, noise_cell_mean: 0.9, negative: {events: 3, hours: 30, fa_per_hour: 0.1}, conditions: {"environment:10": {successes: 10, calls: 12, recall: 10 / 12}}, reverb_conditions: {strong: {conditions: {"music:5": {successes: 7, calls: 8, recall: 7 / 8}}}}}], candidate_adoptions: {member2: {passed: false, reasons: ["FA/hour target failed or unmeasured"]}}};
    const b = browser([tuple(0)], {report}); await b.helper.boot();
    const summary = b.elements.get("summary").tBodies[0].children[0].children;
    assert.equal(summary[1].textContent, "member2"); assert.equal(summary[2].textContent, "42/45 · 93.33%");
    assert.equal(summary[5].textContent, "3회 / 30.00h · 0.100/h"); assert.equal(summary[7].textContent, "미달");
    assert.equal(b.elements.get("conditions").tBodies[0].children.length, 2);
    assert.equal(b.elements.get("conditions").tBodies[0].children[1].children[1].textContent, "강한 실내 울림 · music:5");
  });
  console.log(`${tests.length} paged result UI tests passed`);
})().catch(error => { console.error(error); process.exitCode = 1; });
