"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_paged.js"), "utf8");
const staticScript = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results.js"), "utf8");
const presentation = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_ui.js"), "utf8");
const pagedHtml = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_paged.html"), "utf8");
const staticHtml = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results.html"), "utf8");
const flush = async () => { for (let count = 0; count < 8; count++) await new Promise(resolve => setImmediate(resolve)); };
const tests = [];
async function test(name, run) { await run(); tests.push(name); }
function browser(tuples, {deferPages = false, pageFailure = false, indexFailure = false, invalidIndex = false, summaryFailure = false, autoBoot = false, readinessHook = true, staticViewer = false, playback, indexPath = "viewer-index.json", report = {wakeword: "보라야", models: []}} = {}) {
  const elements = new Map(), calls = [], deferred = new Map(), listeners = {}, parentWindow = {}, readiness = [];
  const picker = {folder: null, index: -1};
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
  const html = staticViewer ? staticHtml : pagedHtml;
  for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1], new Element());
  for (const id of ["storySummary", "conditions", "responses"]) elements.get(id).tBodies = [new Element("tbody")];
  const row = (tuple, sourceIndex) => ({sample_id: tuple[0], path: `audio/${sourceIndex}.wav`, sha256: `sha${sourceIndex}`, kind: tuple[1], category: tuple[2], snr_db: tuple[3], reverb_condition: tuple[4], original_id: tuple[5], original_order: tuple[6], duration_s: tuple[7], calls: tuple[1] === "positive" ? [{start_s: 2, end_s: 3}] : [], noise_eval_label: "시험 배경", results: {mean: {success: true, false_activations: 0, max_score: 0.999, clean_score_drop: 0.01, latency_s: 0.08}}});
  const page = number => ({start: number * 128, clips: tuples.slice(number * 128, (number + 1) * 128).map((tuple, offset) => row(tuple, number * 128 + offset))});
  const context = vm.createContext({document: {getElementById: id => elements.get(id), createElement: tag => new Element(tag)}, console, fetch: async url => {
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
    ready: () => readiness.push({status: "ready", title: elements.get("clipTitle").textContent, source: elements.get("audio").src,
      listCount: picker.folder?.clips.length ?? 0}),
    fail: message => readiness.push({status: "failed", message}),
  };
  elements.get("resume").parentElement = {after() {}};
  elements.get("results-panel-audio").classList = {add() {}};
  context.parent = parentWindow; context.addEventListener = (name, listener) => { listeners[name] = listener; };
  context.KOWW_RESULTS = {report, clips: staticViewer ? tuples.map(row) : undefined, viewer: {schema_version: 1, index_path: indexPath, page_size: 128}};
  vm.runInContext(presentation, context);
  context.KowwResultsUI.mountClipPicker = (library, choose, labelAt, pageSize = 100) => {
    Object.assign(picker, {library, choose, labelAt, pageSize});
    return (folder, index) => Object.assign(picker, {folder, index});
  };
  if (summaryFailure) context.KowwResultsUI.renderSummary = () => { throw new Error("결과 요약 오류"); };
  vm.runInContext(staticViewer ? staticScript : script, context);
  return {elements, calls, deferred, readiness, picker, helper: context.KowwResults,
    message: extra => listeners.message({origin: context.location.origin, source: parentWindow, data: {type: "koww:pause"}, ...extra})};
}
const tuple = (index, overrides = {}) => { const value = [`clip_${index}`, "positive", "environment", 10, "none", `original_${Math.floor(index / 3)}`, Math.floor(index / 3) + 1, 4]; for (const [key, field] of Object.entries(overrides)) value[Number(key)] = field; return value; };
const allLeaves = folders => Array.from(folders).flatMap(folder => folder.folders ? allLeaves(folder.folders) : [folder]);
const folderFor = (viewer, sourceIndex) => allLeaves(viewer.picker.library.folders).find(folder => folder.clips.includes(sourceIndex));
const chooseClip = (viewer, sourceIndex) => {
  const folder = folderFor(viewer, sourceIndex);
  assert.ok(folder, `no folder found for clip ${sourceIndex}`);
  return viewer.picker.choose(folder, folder.clips.indexOf(sourceIndex));
};

(async () => {
  await test("paged chapter readiness waits for the initial detail and folder without playing audio", async () => {
    const b = browser([tuple(0), tuple(1)], {deferPages: true});
    const boot = b.helper.boot(); await flush();
    assert.deepEqual(b.readiness, [{status: "loading"}], "metadata alone cannot finish the chapter transition");
    assert.equal(b.elements.get("audio").plays, 0);
    assert.equal(b.picker.library.defaultFolder.clips.length, 2);
    assert.equal(b.elements.get("playState").textContent, "");
    b.deferred.get(0)(); await boot;
    assert.deepEqual(b.readiness.map(value => value.status), ["loading", "ready"]);
    assert.equal(b.readiness[1].source, "audio/0.wav");
    assert.equal(b.readiness[1].title, "음성 1 · 환경소음");
    assert.equal(b.readiness[1].listCount, 2); assert.equal(b.elements.get("audio").plays, 0);
    assert.equal(b.elements.get("playState").textContent, "");
    await b.helper.move(1); assert.equal(b.readiness.length, 2, "later detail browsing does not restart chapter readiness");
  });
  await test("an empty result is ready without fetching a detail page", async () => {
    const b = browser([]); await b.helper.boot();
    assert.deepEqual(b.readiness.map(value => value.status), ["loading", "ready"]);
    assert.equal(b.calls.length, 1); assert.equal(b.elements.get("audio").plays, 0);
    assert.match(b.readiness[1].title, /오디오가 없습니다/);
  });
  await test("a report with only negative recordings opens the first recorded clip", async () => {
    const b = browser([tuple(0, {1: "negative"})]); await b.helper.boot();
    assert.deepEqual(b.readiness.map(value => value.status), ["loading", "ready"]);
    assert.equal(b.readiness[1].source, "audio/0.wav");
    assert.match(b.readiness[1].title, /호출어 없는 음원/);
    assert.equal(b.picker.folder.continuous, false);
    assert.equal(b.elements.get("audio").plays, 0);
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
      assert.equal(b.elements.has("noiseCoverage"), false);
      assert.equal(b.elements.has("condition-clean"), false);
    }
  });
  await test("112137 clips load only the first detail page and remain reachable from folders", async () => {
    const tuples = Array.from({length: 112137}, (_, index) => tuple(index));
    for (let index = 110000; index < tuples.length; index++) tuples[index] = tuple(index, {1: "negative", 2: "environment"});
    const b = browser(tuples); await b.helper.boot();
    assert.equal(b.helper.state().queueLength, 110000);
    assert.deepEqual(b.calls, ["viewer-index.json", "http://127.0.0.1/results/bora/clips/page-000000.json"]);
    assert.equal(b.picker.pageSize, 100);
    assert.equal(allLeaves(b.picker.library.folders).reduce((total, folder) => total + folder.clips.length, 0), tuples.length);
  });
  await test("continuous playback crosses detail and selector pages without skipping audio", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    await b.elements.get("start").onclick(); await b.helper.move(127, true); await b.elements.get("audio").onended();
    assert.equal(b.helper.state().index, 128); assert.equal(b.elements.get("audio").src, "audio/128.wav");
    assert.equal(b.picker.index, 128);
    assert.equal(b.elements.get("clipTitle").textContent, "음성 43 · 환경소음");
  });
  await test("dashboard navigation keeps continuous audio and the current queue position", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    b.elements.get("results-tab-audio").onclick();
    await b.elements.get("start").onclick(); await b.helper.move(127, true);
    const source = b.elements.get("audio").src, pauses = b.elements.get("audio").pauses, calls = b.calls.length;
    b.elements.get("results-tab-models").onclick();
    b.elements.get("results-tab-conditions").onclick(); b.elements.get("results-tab-audio").onclick();
    assert.equal(b.helper.state().continuous, true); assert.equal(b.helper.state().index, 127);
    assert.equal(b.elements.get("audio").src, source); assert.equal(b.elements.get("audio").pauses, pauses);
    assert.equal(b.calls.length, calls);
    await b.elements.get("audio").onended(); assert.equal(b.helper.state().index, 128);
  });
  await test("compact listening subviews preserve actual playback and the current folder", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    await b.elements.get("start").onclick(); await b.helper.move(127, true);
    const source = b.elements.get("audio").src, pauses = b.elements.get("audio").pauses, calls = b.calls.length;
    b.elements.get("audio-view-reactions").onclick();
    assert.equal(b.elements.get("results-panel-audio").attributes["data-audio-view"], "reactions");
    b.elements.get("audio-view-player").onclick();
    assert.equal(b.elements.get("audio-view-player").attributes["aria-pressed"], "true");
    assert.equal(b.helper.state().continuous, true); assert.equal(b.helper.state().index, 127);
    assert.equal(b.elements.get("audio").src, source); assert.equal(b.elements.get("audio").pauses, pauses); assert.equal(b.calls.length, calls);
    await b.elements.get("audio").onended(); assert.equal(b.helper.state().index, 128);
  });
  await test("only a same-origin exact parent can pause audio without resetting the active folder", async () => {
    const b = browser(Array.from({length: 400}, (_, index) => tuple(index))); await b.helper.boot();
    await b.elements.get("start").onclick(); await b.helper.move(127, true);
    const audio = b.elements.get("audio"), source = audio.src, pauses = audio.pauses, calls = b.calls.length;
    const folder = b.picker.folder;
    b.message({origin: "https://unrelated.example"});
    b.message({source: {}});
    b.message({data: {type: "something-else"}});
    b.message({data: null});
    assert.equal(audio.pauses, pauses); assert.equal(b.helper.state().continuous, true);
    b.message();
    assert.equal(audio.pauses, pauses + 1); assert.equal(b.helper.state().continuous, false);
    assert.equal(b.helper.state().index, 127); assert.equal(audio.src, source);
    assert.equal(b.picker.folder, folder);
    assert.equal(b.calls.length, calls);
    await audio.onended(); assert.equal(b.helper.state().index, 127);
    await b.elements.get("resume").onclick(); assert.equal(b.helper.state().continuous, true);
    assert.equal(b.helper.state().index, 127);
  });
  await test("static exports also retain the current clip when their parent pauses continuous listening", async () => {
    const b = browser(Array.from({length: 9}, (_, index) => tuple(index)), {staticViewer: true});
    await chooseClip(b, 2); await b.elements.get("resume").onclick();
    const audio = b.elements.get("audio"), source = audio.src, pauses = audio.pauses;
    b.message({origin: "https://unrelated.example"}); assert.equal(audio.pauses, pauses);
    b.message(); assert.equal(audio.pauses, pauses + 1);
    audio.onended(); assert.equal(audio.src, source); assert.equal(b.picker.index, 2);
    await b.elements.get("resume").onclick(); assert.equal(audio.src, source);
    audio.onended(); await flush(); assert.notEqual(audio.src, source); assert.equal(b.picker.index, 3);
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
    assert.equal(b.elements.get("clipTitle").textContent, "음성 86 · 환경소음");
  });
  await test("choosing a different folder ignores a stale audio response and stops autoplay", async () => {
    const tuples = Array.from({length: 400}, (_, index) => tuple(index));
    tuples[256][3] = 5; tuples[256][4] = "strong";
    const b = browser(tuples, {deferPages: true}); const boot = b.helper.boot(); await flush(); b.deferred.get(0)(); await boot;
    await b.elements.get("resume").onclick();
    const old = b.helper.move(128, true); await flush();
    const changed = chooseClip(b, 256); await flush(); b.deferred.get(2)(); await changed; b.deferred.get(1)(); await old;
    assert.equal(b.helper.state().queueLength, 1); assert.equal(b.elements.get("audio").src, "audio/256.wav");
    assert.equal(b.picker.folder, folderFor(b, 256));
    assert.equal(b.helper.state().continuous, false);
    assert.equal(b.elements.get("audio").paused, true);
  });
  await test("selecting a negative clip requests only its detail page", async () => {
    const tuples = Array.from({length: 500}, (_, index) => tuple(index, {1: index > 0 ? "negative" : "positive"}));
    const b = browser(tuples); await b.helper.boot(); const before = b.calls.length;
    assert.equal(folderFor(b, 131).clips.length, 499);
    await chooseClip(b, 131);
    assert.deepEqual(b.calls.slice(before), ["http://127.0.0.1/results/bora/clips/page-000001.json"]);
    assert.equal(b.picker.folder, folderFor(b, 131));
    assert.equal(b.elements.get("audio").src, "audio/131.wav");
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
    assert.equal(b.elements.get("playState").textContent, "");
  });
  await test("versioned presentation folder resolves pages beside its own index", async () => {
    const b = browser([tuple(0)], {indexPath: "viewer_v1_fix/viewer-index.json"}); await b.helper.boot();
    assert.deepEqual(b.calls, ["viewer_v1_fix/viewer-index.json", "http://127.0.0.1/results/bora/viewer_v1_fix/clips/page-000000.json"]);
    assert.equal(b.elements.get("audio").src, "audio/0.wav");
  });
  await test("reported model results and measured noise conditions remain available", async () => {
    const report = {wakeword: "보라야", selected_candidate: "member2", candidate_definitions: [{id: "member1", kind: "single"}, {id: "member2", kind: "single"}], models: [{id: "member2", rank: 1, kind: "single", primary: {successes: 42, calls: 45, recall: 42 / 45, latency_p95_s: 0.2}, clean: {recall: 1}, noise_cell_mean: 0.9, negative: {events: 3, hours: 30, fa_per_hour: 0.1}, conditions: {"environment:10": {successes: 10, calls: 12, recall: 10 / 12}}, reverb_conditions: {strong: {conditions: {"music:5": {successes: 7, calls: 8, recall: 7 / 8}}}}}], candidate_adoptions: {member2: {passed: false, reasons: ["FA/hour target failed or unmeasured"]}}};
    const frozen = JSON.stringify(report);
    const b = browser([tuple(0)], {report}); await b.helper.boot();
    const comparison = b.elements.get("storySummary").tBodies[0].children[0].children;
    assert.equal(comparison[0].textContent, "모델 2");
    assert.equal(comparison[1].textContent, "93.33%");
    assert.equal(comparison[5].textContent, "0.1회");
    assert.match(comparison[5].title, /3회.*30/);
    assert.equal(comparison[6].textContent, "0.20초");
    assert.equal(comparison.length, 8);
    assert.equal(b.elements.get("conditions").tBodies[0].children.length, 3);
    assert.equal(b.elements.get("conditions").tBodies[0].children[0].children[3].textContent, "83.33%\n10 / 12회");
    b.elements.get("condition-reverb").value = "strong"; b.elements.get("condition-reverb").onchange();
    assert.equal(b.elements.get("conditions").tBodies[0].children[1].children[4].textContent, "87.50%\n7 / 8회");
    assert.equal(JSON.stringify(report), frozen);
  });
  for (const staticViewer of [false, true]) {
    const mode = staticViewer ? "static" : "paged";
    await test(`${mode} picker selects the visible clip and switches folders without playing`, async () => {
      const items = Array.from({length: 321}, (_, i) => tuple(i, i === 320 ? {2: "clean"} : {}));
      const b = browser(items, {staticViewer});
      if (b.helper) await b.helper.boot();
      const audio = b.elements.get("audio"), initial = b.picker.folder;
      assert.equal(b.picker.pageSize, 100);
      assert.equal(initial.clips.length, 320);
      assert.equal(b.picker.index, 0);
      assert.match(b.picker.labelAt(102), /음성 35 · 환경소음/);
      await chooseClip(b, 102);
      assert.equal(audio.src, "audio/102.wav"); assert.equal(b.picker.index, 102);
      assert.equal(b.picker.folder, initial);
      await chooseClip(b, 320);
      assert.equal(audio.src, "audio/320.wav"); assert.equal(b.picker.index, 0);
      assert.equal(b.picker.folder, folderFor(b, 320));
      assert.match(b.picker.labelAt(320), /조용한 환경/);
      assert.equal(audio.plays, 0);
    });
    await test(`${mode} transport toggles actual playback and resumes the current position without new source or gain`, async () => {
      const b = browser([tuple(0), tuple(1)], {staticViewer}); if (b.helper) await b.helper.boot();
      const audio = b.elements.get("audio"), toggle = b.elements.get("resume"), notice = b.elements.get("playState");
      audio.currentTime = 2.5; audio.volume = .37;
      const source = audio.src, calls = b.calls.length;
      assert.equal(toggle.textContent, "재생");
      assert.equal(notice.textContent, "");
      await toggle.onclick();
      assert.equal(toggle.textContent, "일시정지"); assert.equal(toggle.attributes["aria-pressed"], "true");
      assert.equal(audio.currentTime, 2.5); assert.equal(audio.src, source); assert.equal(b.calls.length, calls);
      assert.equal(notice.textContent, "");
      await toggle.onclick(); assert.equal(audio.paused, true); assert.equal(toggle.textContent, "재생");
      assert.equal(notice.textContent, "");
      await toggle.onclick(); assert.equal(audio.currentTime, 2.5); assert.equal(audio.src, source);
      assert.equal(audio.volume, .37); assert.equal(audio.playbackRate, 1);
      b.message(); await toggle.onclick(); assert.equal(audio.currentTime, 2.5);
      audio.pause(); assert.equal(toggle.textContent, "재생");
      await audio.play(); assert.equal(toggle.textContent, "일시정지");
      assert.equal(notice.textContent, "");
      audio.onerror(); assert.equal(toggle.textContent, "재생"); assert.equal(toggle.attributes["aria-pressed"], "false");
      assert.match(notice.textContent, /오디오를 불러올 수 없습니다/);
      await b.elements.get("next").onclick();
      assert.equal(audio.src, "audio/1.wav"); assert.equal(notice.textContent, "");
      await toggle.onclick(); assert.equal(notice.textContent, "");
    });
    await test(`${mode} transport plays a negative clip once and never advances into a different folder`, async () => {
      const b = browser([tuple(0), tuple(1, {1: "negative"}), tuple(2, {1: "negative"})], {staticViewer});
      if (b.helper) await b.helper.boot();
      await chooseClip(b, 1);
      const audio = b.elements.get("audio"), source = audio.src;
      assert.equal(b.elements.get("playState").textContent, "");
      audio.currentTime = 1.25;
      await b.elements.get("resume").onclick();
      assert.equal(audio.src, source); assert.equal(source, "audio/1.wav"); assert.equal(audio.currentTime, 1.25);
      await audio.onended(); assert.equal(audio.src, source); assert.equal(b.elements.get("resume").textContent, "재생");
      assert.equal(b.elements.get("playState").textContent, "");
    });
    await test(`${mode} keeps model reactions available for another folder`, async () => {
      const b = browser([tuple(0), tuple(1, {1: "negative"})], {staticViewer}); if (b.helper) await b.helper.boot();
      for (const id of ["original", "original-go", "clipInfo", "scores", "sample-id", "download", "hash"]) {
        assert.equal(b.elements.has(id), false);
      }
      const reaction = () => b.elements.get("responses").tBodies[0].children[0].children.map(cell => cell.textContent);
      assert.deepEqual(reaction().slice(1), ["감지", "0.08초"]);
      await chooseClip(b, 1);
      b.elements.get("audio-view-reactions").onclick();
      assert.deepEqual(reaction().slice(1), ["오작동 0회", "—"]);
      assert.equal(b.elements.get("audio").src, "audio/1.wav");
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
