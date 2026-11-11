"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_paged.js"), "utf8");
const staticScript = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results.js"), "utf8");
const presentation = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_ui.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_paged.html"), "utf8");
const flush = async () => { for (let count = 0; count < 8; count++) await new Promise(resolve => setImmediate(resolve)); };
const tests = [];
async function test(name, run) { await run(); tests.push(name); }
function browser(tuples, {deferPages = false, pageFailure = false, indexFailure = false, invalidIndex = false, summaryFailure = false, autoBoot = false, readinessHook = true, staticViewer = false, viewportHeight, playback, enhance, createPicker, indexPath = "viewer-index.json", report = {wakeword: "보라야", models: []}} = {}) {
  const elements = new Map(), calls = [], deferred = new Map(), listeners = {}, parentWindow = {}, readiness = [];
  class Element {
    constructor(tag = "div") { this.tagName = tag; this.children = []; this.value = ""; this.textContent = ""; this.disabled = false; this.src = ""; this.plays = 0; this.pauses = 0; this.attributes = {}; this.paused = true; this.ended = false; this.volume = 1; this.playbackRate = 1; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    get options() { return this.children; }
    get src() { return this._src; }
    set src(value) { this._src = value; this.currentTime = 0; this.ended = false; this.paused = true; }
    pause() { this.pauses++; this.paused = true; this.onpause?.(); }
    play() { this.plays++; this.paused = false; this.ended = false; this.onplay?.(); return playback ? playback(this.plays) : Promise.resolve(); }
    removeAttribute(name) { this[name] = ""; }
    setAttribute(name, value) { this.attributes[name] = value; }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1], new Element());
  for (const id of ["summary", "conditions", "responses", "scores", "recommendations", "targetComparison"]) elements.get(id).tBodies = [new Element("tbody")];
  for (const value of ["none", "moderate", "strong"]) { const option = new Element("option"); option.value = value; elements.get("reverb").append(option); }
  elements.get("reverb").value = "none"; elements.get("category").value = "all";
  const row = (tuple, sourceIndex) => ({sample_id: tuple[0], path: `audio/${sourceIndex}.wav`, sha256: `sha${sourceIndex}`, kind: tuple[1], category: tuple[2], snr_db: tuple[3], reverb_condition: tuple[4], original_id: tuple[5], original_order: tuple[6], duration_s: tuple[7], calls: tuple[1] === "positive" ? [{start_s: 2, end_s: 3}] : [], noise_eval_label: "시험 배경", results: {mean: {success: true, false_activations: 0, max_score: 0.999, clean_score_drop: 0.01, latency_s: 0.08}}});
  const page = number => ({start: number * 128, clips: tuples.slice(number * 128, (number + 1) * 128).map((tuple, offset) => row(tuple, number * 128 + offset))});
  const context = vm.createContext({document: {getElementById: id => elements.get(id), createElement: tag => new Element(tag)}, console, evaluationScopeText: () => "시험 범위", evaluationNoiseCoverageText: () => "시험 배경", evaluationBackgroundTypeText: clip => clip.noise_eval_label || "", fetch: async url => {
    calls.push(url);
    if (url === indexPath) return {ok: !indexFailure, status: indexFailure ? 503 : 200, json: async () => ({schema_version: invalidIndex ? 99 : 1, scope_text: "정확한 평가 범위", noise_coverage_text: "전체 음원 사용", clips: tuples})};
    const number = Number(url.match(/page-(\d+)\.json/)[1]);
    if (pageFailure) return {ok: false, status: 503};
    if (deferPages) return new Promise(resolve => deferred.set(number, () => resolve({ok: true, json: async () => page(number)})));
    return {ok: true, json: async () => page(number)};
  }});
  context.window = context; context.location = new URL("http://127.0.0.1/results/bora/"); context.URL = URL; context.KOWW_RESULTS_TEST = !autoBoot;
  if (readinessHook) context.KowwChapterReady = {
    loading: () => readiness.push({status: "loading"}),
    ready: () => readiness.push({status: "ready", scope: elements.get("scope").textContent, title: elements.get("clipTitle").textContent, source: elements.get("audio").src, listCount: elements.get("item").children.length}),
    fail: message => readiness.push({status: "failed", message}),
  };
  if (enhance) context.KowwListening = {enhance, createPicker};
  elements.get("resume").parentElement = {after() {}};
  elements.get("results-panel-audio").classList = {add() {}};
  context.innerHeight = viewportHeight;
  context.parent = parentWindow; context.addEventListener = (name, listener) => { listeners[name] = listener; };
  context.KOWW_RESULTS = {report, clips: staticViewer ? tuples.map(row) : undefined, viewer: {schema_version: 1, index_path: indexPath, page_size: 128}};
  vm.runInContext(presentation, context);
  if (summaryFailure) context.KowwResultsUI.renderSummary = () => { throw new Error("결과 요약 오류"); };
  vm.runInContext(staticViewer ? staticScript : script, context);
  return {elements, calls, deferred, readiness, helper: context.KowwResults,
    resize(height) { context.innerHeight = height; listeners.resize?.(); },
    message: extra => listeners.message({origin: context.location.origin, source: parentWindow, data: {type: "koww:pause"}, ...extra})};
}
const tuple = (index, overrides = {}) => { const value = [`clip_${index}`, "positive", ["environment", "music", "competing_speech"][index % 3], 10, "none", `original_${Math.floor(index / 3)}`, Math.floor(index / 3) + 1, 4]; for (const [key, field] of Object.entries(overrides)) value[Number(key)] = field; return value; };

(async () => {
  await test("paged chapter readiness waits for the initial detail, filters and list without playing audio", async () => {
    const b = browser([tuple(0), tuple(1)], {deferPages: true});
    const boot = b.helper.boot(); await flush();
    assert.deepEqual(b.readiness, [{status: "loading"}], "metadata alone cannot finish the chapter transition");
    assert.equal(b.elements.get("audio").plays, 0);
    assert.equal(b.elements.get("item").children.length, 2);
    b.deferred.get(0)(); await boot;
    assert.deepEqual(b.readiness.map(value => value.status), ["loading", "ready"]);
    assert.equal(b.readiness[1].scope, "정확한 평가 범위");
    assert.equal(b.readiness[1].source, "audio/0.wav");
    assert.equal(b.readiness[1].title, "음성 1 · 환경소음");
    assert.equal(b.readiness[1].listCount, 2); assert.equal(b.elements.get("audio").plays, 0);
    await b.helper.move(1); assert.equal(b.readiness.length, 2, "later detail browsing does not restart chapter readiness");
  });
  await test("an empty initial results filter is ready without fetching a detail page", async () => {
    const b = browser([tuple(0, {1: "negative"})]); await b.helper.boot();
    assert.deepEqual(b.readiness.map(value => value.status), ["loading", "ready"]);
    assert.equal(b.calls.length, 1); assert.equal(b.elements.get("audio").plays, 0);
    assert.match(b.readiness[1].title, /오디오가 없습니다/);
  });
  await test("a cancelled first detail cannot announce a completed results chapter", async () => {
    const b = browser([tuple(0)], {deferPages: true});
    const boot = b.helper.boot(); await flush();
    b.message(); b.deferred.get(0)(); await boot;
    assert.deepEqual(b.readiness.map(value => value.status), ["loading", "failed"]);
    assert.equal(b.elements.get("audio").plays, 0); assert.notEqual(b.elements.get("audio").src, "audio/0.wav");
  });
  await test("initial index and detail errors fail chapter readiness and keep standalone inline errors", async () => {
    for (const failure of [{indexFailure: true}, {invalidIndex: true}, {pageFailure: true}, {summaryFailure: true}]) {
      const b = browser([tuple(0)], {...failure, autoBoot: true}); await flush();
      assert.deepEqual(b.readiness.map(value => value.status), ["loading", "failed"]);
      assert.ok(b.readiness[1].message); assert.ok(b.elements.get("playState").textContent);
      assert.equal(b.elements.get("audio").plays, 0);
    }
  });
  await test("static results signal readiness after initial rendering and report synchronous boot failures", async () => {
    const b = browser([tuple(0), tuple(1)], {staticViewer: true});
    assert.deepEqual(b.readiness.map(value => value.status), ["loading", "ready"]);
    assert.equal(b.readiness[1].source, "audio/0.wav"); assert.equal(b.readiness[1].listCount, 2);
    assert.equal(b.elements.get("audio").plays, 0); assert.equal(b.calls.length, 0);
    const failed = browser([tuple(0)], {staticViewer: true, summaryFailure: true});
    assert.deepEqual(failed.readiness.map(value => value.status), ["loading", "failed"]);
    assert.equal(failed.readiness[1].message, "결과 요약 오류"); assert.equal(failed.elements.get("playState").textContent, "결과 요약 오류");
  });
  await test("portable static and paged viewers still boot without the optional chapter protocol", async () => {
    for (const staticViewer of [false, true]) {
      const b = browser([tuple(0)], {staticViewer, readinessHook: false});
      if (!staticViewer) await b.helper.boot();
      assert.equal(b.elements.get("audio").src, "audio/0.wav"); assert.equal(b.elements.get("audio").plays, 0);
    }
  });
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
  await test("dashboard navigation keeps continuous audio and the current queue position", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    b.elements.get("results-tab-audio").onclick();
    await b.elements.get("start").onclick(); await b.helper.move(127, true);
    const source = b.elements.get("audio").src, pauses = b.elements.get("audio").pauses, calls = b.calls.length;
    b.elements.get("results-tab-models").onclick(); b.elements.get("model-tab-targets").onclick();
    b.elements.get("results-tab-conditions").onclick(); b.elements.get("results-tab-audio").onclick();
    assert.equal(b.helper.state().continuous, true); assert.equal(b.helper.state().index, 127);
    assert.equal(b.elements.get("audio").src, source); assert.equal(b.elements.get("audio").pauses, pauses);
    assert.equal(b.calls.length, calls);
    await b.elements.get("audio").onended(); assert.equal(b.helper.state().index, 128);
  });
  await test("compact listening subviews preserve actual playback, filters and the current result", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    await b.elements.get("start").onclick(); await b.helper.move(127, true);
    const source = b.elements.get("audio").src, pauses = b.elements.get("audio").pauses, calls = b.calls.length;
    b.elements.get("audio-view-reactions").onclick();
    assert.equal(b.elements.get("results-panel-audio").attributes["data-audio-view"], "reactions");
    b.elements.get("audio-view-settings").onclick(); b.elements.get("audio-view-player").onclick();
    assert.equal(b.elements.get("audio-view-player").attributes["aria-pressed"], "true");
    assert.equal(b.helper.state().continuous, true); assert.equal(b.helper.state().index, 127);
    assert.equal(b.elements.get("audio").src, source); assert.equal(b.elements.get("audio").pauses, pauses); assert.equal(b.calls.length, calls);
    await b.elements.get("audio").onended(); assert.equal(b.helper.state().index, 128);
  });
  await test("optional audio control popovers keep the user's choice across resize without resetting playback", async () => {
    const b = browser(Array.from({length: 20}, (_, index) => tuple(index)), {viewportHeight: 404}); await b.helper.boot();
    assert.equal(b.elements.get("audio-filter-details").open, false);
    assert.equal(b.elements.get("clip-select-details").open, false);
    await b.elements.get("start").onclick(); await b.helper.move(3, true);
    const source = b.elements.get("audio").src, calls = b.calls.length;
    b.elements.get("audio-filter-details").open = true; b.resize(444);
    assert.equal(b.elements.get("audio-filter-details").open, true);
    b.resize(600); assert.equal(b.elements.get("clip-select-details").open, false);
    b.resize(404); assert.equal(b.elements.get("audio-filter-details").open, true);
    assert.equal(b.helper.state().continuous, true); assert.equal(b.helper.state().index, 3);
    assert.equal(b.elements.get("audio").src, source); assert.equal(b.calls.length, calls);
  });
  await test("only a same-origin exact parent can pause audio without resetting its queue or filters", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    await b.elements.get("start").onclick(); await b.helper.move(127, true);
    const audio = b.elements.get("audio"), source = audio.src, pauses = audio.pauses, calls = b.calls.length;
    const filters = ["snr", "category", "reverb"].map(id => b.elements.get(id).value);
    b.message({origin: "https://unrelated.example"});
    b.message({source: {}});
    b.message({data: {type: "something-else"}});
    b.message({data: null});
    assert.equal(audio.pauses, pauses); assert.equal(b.helper.state().continuous, true);
    b.message();
    assert.equal(audio.pauses, pauses + 1); assert.equal(b.helper.state().continuous, false);
    assert.equal(b.helper.state().index, 127); assert.equal(audio.src, source);
    assert.deepEqual(["snr", "category", "reverb"].map(id => b.elements.get(id).value), filters);
    assert.equal(b.calls.length, calls);
    await audio.onended(); assert.equal(b.helper.state().index, 127);
    await b.elements.get("resume").onclick(); assert.equal(b.helper.state().continuous, true);
    assert.equal(b.helper.state().index, 127);
  });
  await test("static exports also retain the current clip when their parent pauses continuous listening", async () => {
    const b = browser(Array.from({length: 9}, (_, index) => tuple(index)), {staticViewer: true});
    await b.elements.get("start").onclick();
    b.elements.get("item").value = "2"; b.elements.get("item").onchange(); await flush();
    const audio = b.elements.get("audio"), source = audio.src, pauses = audio.pauses;
    b.message({origin: "https://unrelated.example"}); assert.equal(audio.pauses, pauses);
    b.message(); assert.equal(audio.pauses, pauses + 1);
    audio.onended(); assert.equal(audio.src, source); assert.equal(b.elements.get("item").value, "2");
    await b.elements.get("resume").onclick(); assert.equal(audio.src, source);
    audio.onended(); await flush(); assert.notEqual(audio.src, source); assert.equal(b.elements.get("item").value, "3");
    assert.equal(b.calls.length, 0);
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
    assert.equal(b.elements.get("clipTitle").textContent, "음성 86 · 음악·전자음");
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
    assert.equal(b.elements.get("responses").tBodies[0].children[0].children[1].textContent, "오작동 0회");
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
    const report = {wakeword: "보라야", selected_candidate: "member2", candidate_definitions: [{id: "member1", kind: "single"}, {id: "member2", kind: "single"}], models: [{id: "member2", rank: 1, kind: "single", primary: {successes: 42, calls: 45, recall: 42 / 45, latency_p95_s: 0.2}, clean: {recall: 1}, noise_cell_mean: 0.9, negative: {events: 3, hours: 30, fa_per_hour: 0.1}, conditions: {"environment:10": {successes: 10, calls: 12, recall: 10 / 12}}, reverb_conditions: {strong: {conditions: {"music:5": {successes: 7, calls: 8, recall: 7 / 8}}}}}], candidate_adoptions: {member2: {passed: false, reasons: ["FA/hour target failed or unmeasured"]}}};
    const frozen = JSON.stringify(report);
    const b = browser([tuple(0)], {report}); await b.helper.boot();
    const summary = b.elements.get("summary").tBodies[0].children[0].children;
    assert.equal(summary[1].textContent, "모델 2"); assert.equal(summary[2].textContent, "93.33%\n42 / 45회");
    assert.equal(summary[5].textContent, "0.100회\n3회 / 30.00시간"); assert.equal(summary[7].textContent, "미달");
    assert.equal(summary[6].textContent, "0.20초");
    assert.equal(b.elements.get("conditions").tBodies[0].children.length, 3);
    assert.equal(b.elements.get("conditions").tBodies[0].children[0].children[3].textContent, "83.33%\n10 / 12회");
    b.elements.get("condition-reverb").value = "strong"; b.elements.get("condition-reverb").onchange();
    assert.equal(b.elements.get("conditions").tBodies[0].children[1].children[4].textContent, "87.50%\n7 / 8회");
    assert.equal(JSON.stringify(report), frozen);
  });
  for (const staticViewer of [false, true]) {
    const mode = staticViewer ? "static" : "paged";
    await test(`${mode} shared picker browses bounded pages without changing audio and selects the correct clip`, async () => {
      const picker = {
        mounts: 0, closed: 0,
        setItems(entries, index, label) { this.entries = entries; this.mounts++; this.setSelected(index, label); },
        setSelected(index, label) { this.index = index; this.label = label; },
        close() { this.closed++; },
      };
      let choose, transport;
      const items = Array.from({length: 321}, (_, i) => tuple(i));
      items[320][2] = "clean";
      const b = browser(items, {staticViewer,
        createPicker: options => { choose = options.onSelect; return picker; },
        enhance: (audio, controls, options) => { transport = options; },
      });
      if (b.helper) await b.helper.boot();
      assert.equal(transport.volume, b.elements.get("result-volume"));
      assert.equal(picker.entries.length, 100);
      assert.equal(picker.index, 0);
      const audio = b.elements.get("audio"), calls = b.calls.length;
      audio.currentTime = 2.25;
      b.elements.get("clip-list-next").onclick();
      assert.equal(picker.index, -1);
      assert.match(picker.entries[0].fullLabel, /^101번째 음원/);
      assert.equal(audio.src, "audio/0.wav");
      assert.equal(audio.currentTime, 2.25);
      assert.equal(b.calls.length, calls);
      await choose(2);
      assert.equal(audio.src, "audio/102.wav");
      assert.equal(picker.index, 2);
      await choose(3);
      assert.equal(picker.mounts, 2, "selecting within a page keeps its options in place");
      b.elements.get("extra").value = staticViewer ? "clip_320" : "0";
      await b.elements.get("extra").onchange();
      assert.equal(audio.src, "audio/320.wav");
      assert.equal(picker.index, -1);
      assert.match(picker.label, /조용한 환경/);
      assert.equal(b.elements.get("queue").textContent, "추가 음원");
      b.elements.get("results-tab-conditions").onclick();
      assert.equal(picker.closed, 1);
    });
    await test(`${mode} C-style list stays at five global rows and browsing pages never fetches or changes playback`, async () => {
      let mounted = 0;
      const b = browser(Array.from({length: 320}, (_, i) => tuple(i)), {staticViewer, enhance: () => { mounted++; }});
      if (b.helper) await b.helper.boot();
      assert.equal(mounted, 1);
      const list = b.elements.get("clip-list"), audio = b.elements.get("audio"); audio.currentTime = 2.25;
      assert.equal(list.children.length, 5); assert.equal(list.children[0].attributes["data-number"], "1");
      assert.equal(list.children[0].attributes["aria-current"], "true");
      assert.ok(list.children[0].attributes["aria-label"].includes("음성"));
      const source = audio.src, calls = b.calls.length;
      b.elements.get("clip-list-next").onclick();
      assert.equal(list.children.length, 5); assert.equal(list.children[0].attributes["data-number"], "6");
      assert.equal(b.elements.get("clip-list-position").textContent, "2 / 64");
      assert.equal(audio.src, source); assert.equal(audio.currentTime, 2.25); assert.equal(audio.plays, 0); assert.equal(b.calls.length, calls);
      await list.children[0].onclick();
      assert.equal(audio.src, "audio/5.wav"); assert.equal(list.children[0].attributes["aria-current"], "true");
      b.elements.get("item").value = "299"; await b.elements.get("item").onchange();
      assert.equal(audio.src, "audio/299.wav"); assert.equal(list.children.length, 5);
      assert.equal(list.children[0].attributes["data-number"], "296"); assert.equal(list.children[4].attributes["data-number"], "300");
      assert.equal(list.children[4].attributes["aria-current"], "true"); assert.equal(mounted, 1);
      if (b.helper) assert.ok(b.elements.get("item").children.length <= 100);
    });
    await test(`${mode} transport toggles actual playback and resumes the current position without new source or gain`, async () => {
      const b = browser([tuple(0), tuple(1)], {staticViewer}); if (b.helper) await b.helper.boot();
      const audio = b.elements.get("audio"), toggle = b.elements.get("resume");
      audio.currentTime = 2.5; audio.volume = .37;
      const source = audio.src, calls = b.calls.length;
      assert.equal(toggle.textContent, "재생");
      await toggle.onclick();
      assert.equal(toggle.textContent, "일시정지"); assert.equal(toggle.attributes["aria-pressed"], "true");
      assert.equal(audio.currentTime, 2.5); assert.equal(audio.src, source); assert.equal(b.calls.length, calls);
      await toggle.onclick(); assert.equal(audio.paused, true); assert.equal(toggle.textContent, "재생");
      await toggle.onclick(); assert.equal(audio.currentTime, 2.5); assert.equal(audio.src, source);
      assert.equal(audio.volume, .37); assert.equal(audio.playbackRate, 1);
      b.message(); await toggle.onclick(); assert.equal(audio.currentTime, 2.5);
      audio.pause(); assert.equal(toggle.textContent, "재생");
      await audio.play(); assert.equal(toggle.textContent, "일시정지");
      audio.onerror(); assert.equal(toggle.textContent, "재생"); assert.equal(toggle.attributes["aria-pressed"], "false");
    });
    await test(`${mode} transport plays the selected extra without jumping back to the main queue`, async () => {
      const b = browser([tuple(0), tuple(1, {1: "negative"})], {staticViewer}); if (b.helper) await b.helper.boot();
      b.elements.get("extra").value = staticViewer ? "clip_1" : "0";
      await b.elements.get("extra").onchange();
      const audio = b.elements.get("audio"), source = audio.src;
      audio.currentTime = 1.25;
      await b.elements.get("resume").onclick();
      assert.equal(audio.src, source); assert.equal(source, "audio/1.wav"); assert.equal(audio.currentTime, 1.25);
      await audio.onended(); assert.equal(audio.src, source); assert.equal(b.elements.get("resume").textContent, "재생");
    });
    await test(`${mode} cancelled pending autoplay cannot restart audio or override the stopped transport`, async () => {
      let finishPlay;
      const b = browser([tuple(0), tuple(1)], {staticViewer, playback: () => new Promise(resolve => { finishPlay = resolve; })});
      if (b.helper) await b.helper.boot();
      const pendingPlay = b.elements.get("resume").onclick();
      b.message(); finishPlay(); await pendingPlay;
      assert.equal(b.elements.get("audio").paused, true); assert.equal(b.elements.get("resume").textContent, "재생");
      const source = b.elements.get("audio").src;
      await b.elements.get("audio").onended(); assert.equal(b.elements.get("audio").src, source);
    });
  }
  await test("cancelled pending detail is not allowed to autoplay after the parent closes the player", async () => {
    const b = browser(Array.from({length: 300}, (_, i) => tuple(i)), {deferPages: true});
    const boot = b.helper.boot(); await flush(); b.deferred.get(0)(); await boot;
    const old = b.helper.move(128, true); await flush();
    const plays = b.elements.get("audio").plays;
    b.message(); b.deferred.get(1)(); await old;
    assert.equal(b.elements.get("audio").plays, plays); assert.equal(b.elements.get("resume").textContent, "재생");
  });
  console.log(`${tests.length} paged result UI tests passed`);
})().catch(error => { console.error(error); process.exitCode = 1; });
