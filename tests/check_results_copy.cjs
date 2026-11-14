"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_ui.js"), "utf8");
const cases = [];
function test(name, run) { run(); cases.push(name); }
test("both frozen result layouts support the shared picker and playback controls", () => {
  for (const name of ["results.html", "results_paged.html"]) {
    const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources", name), "utf8");
    assert.match(html, /results\.css[\s\S]*audio_listening\.css/);
    for (const className of ["listen-player", "listen-heading", "listen-title", "listen-position", "listen-audio", "listen-controls", "listen-select"]) assert.ok(html.includes(className), name + " " + className);
    assert.match(html, /id="start"[^>]*hidden/); assert.match(html, /id="pause"[^>]*hidden/);
    const controls = html.match(/<div class="playback-controls listen-controls">([\s\S]*?)<\/div>/)[1];
    assert.deepEqual([...controls.matchAll(/id="([^"]+)"/g)].map(match => match[1]), ["prev", "resume", "next", "result-volume"]);
    assert.match(html, /id="result-clip-picker-button"[^>]*aria-controls="result-clip-picker-list"/);
    assert.match(html, /id="item" hidden class="listen-select" aria-label=/);
    for (const id of ["clip-list", "clip-list-prev", "clip-list-next", "clip-list-position"]) assert.ok(html.includes(`id="${id}"`));
    assert.match(html, /class="listen-workspace results-listening-workspace"/);
    assert.match(html, /src="listening_player\.js"[\s\S]*src="results(?:_paged)?\.js"/);
    assert.doesNotMatch(html, /data-value="(?:keep|discard|review)"/);
  }
  const css = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results.css"), "utf8");
  assert.match(css, /data-audio-view="reactions"\] \.audio-player\.listen-player\{display:contents/);
  assert.match(css, /data-audio-view="settings"\] \.playback-controls\{display:none/);
});
test("report navigation omits the repeated evaluation scope and keeps the four results tabs", () => {
  for (const name of ["results.html", "results_paged.html"]) {
    const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources", name), "utf8");
    const navigation = html.match(/<nav class="dashboard-tabs"[^>]*>([\s\S]*?)<\/nav>/)[1];
    assert.deepEqual([...navigation.matchAll(/id="results-tab-([^"]+)"/g)].map(match => match[1]), ["models", "conditions", "audio", "criteria"]);
    assert.doesNotMatch(html, /평가 정보|scope-disclosure|id="scope"/);
    assert.match(html, /id="noiseCoverage"/);
  }
});
test("both report layouts put the comparison metrics in the models panel", () => {
  for (const name of ["results.html", "results_paged.html"]) {
    const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources", name), "utf8");
    const models = html.slice(html.indexOf('id="results-panel-models"'), html.indexOf('id="results-panel-conditions"'));
    for (const id of ["overview-source", "overview-model", "storySummary", "storySummaryMetrics"]) {
      assert.match(models, new RegExp(`id="${id}"`));
    }
    assert.doesNotMatch(html, /result-full-details|상세 결과|model-tab-|targetComparison|id="summary"|id="recommendations"/);
    assert.deepEqual([...models.matchAll(/<th scope="col"[^>]*>([\s\S]*?)<\/th>/g)]
      .map(match => match[1].replace(/<[^>]+>/g, "").trim()), [
      "모델", "종합 감지율", "소음·울림 없음", "강한 소음·울림", "최약 조건", "오호출/시간", "반응 시간", "추천 감지 설정", "평가 목표"
    ]);
    assert.match(models, /aria-describedby="overview-latency-note"/);
    assert.equal([...models.matchAll(/id="overview-latency-note"/g)].length, 1);
    assert.deepEqual([...models.matchAll(/<button[^>]+data-overview-help="([^"]+)"/g)].map(match => match[1]), ["hard", "worst"]);
    assert.match(html, /id="overview-help"[^>]*popover="auto"[^>]*role="dialog"[^>]*aria-labelledby="overview-help-title"/);
    for (const id of ["overview-help", "overview-help-content", "overview-help-close"]) {
      assert.equal([...html.matchAll(new RegExp(`id="${id}"`, "g"))].length, 1);
    }
  }
});
const fullText = element => [element.textContent, ...element.children.map(fullText)].filter(Boolean).join(" ");
function browser({dashboard = false} = {}) {
  class Element {
    constructor(tag = "div") { this.tagName = tag; this.children = []; this.value = ""; this.textContent = ""; this.disabled = false; this.attributes = {}; this.dataset = {}; this.style = {}; this.listeners = new Map(); this.open = false; }
    append(...nodes) {
      for (const node of nodes) node.parentElement = this;
      this.children.push(...nodes);
    }
    replaceChildren(...nodes) { this.children = []; this.textContent = ""; this.append(...nodes); }
    get previousElementSibling() {
      const items = this.parentElement?.children || [];
      return items[items.indexOf(this) - 1] ?? null;
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name]; }
    focus() { this.focused = true; }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    dispatch(name, details = {}) {
      const event = {target: this, currentTarget: this, preventDefault() {}, ...details};
      this[`on${name}`]?.(event);
      this.listeners.get(name)?.(event);
    }
    getBoundingClientRect() { return {top: 0, bottom: 20, left: 0, right: 20, width: 20, height: 20}; }
    matches(selector) { return selector === ":popover-open" && this.open; }
    showPopover({source} = {}) { this.open = true; this.popoverSource = source; this.dispatch("toggle", {newState: "open"}); }
    hidePopover() { this.open = false; this.dispatch("toggle", {newState: "closed"}); }
  }
  const elements = new Map();
  for (const id of ["responses", "scores", "condition-model", "condition-reverb", "condition-clean", "conditions", "storySummary", "overview-source", "overview-model", "overview-settings-label", "storySummaryMetrics", "overview-note", "searchCriteria", "searchCriteriaNote", "evaluationCriteria", "condition-snr", "conditionMetrics", "overview-help", "overview-help-title", "overview-help-content", "overview-help-close"]) {
    elements.set(id, new Element());
  }
  const headers = ["hard", "worst"].map(key => {
    const element = new Element("button"); element.dataset.overviewHelp = key; return element;
  });
  for (const id of ["storySummary", "responses", "scores", "conditions"]) elements.get(id).tBodies = [new Element("tbody")];
  elements.get("condition-snr").value = "10";
  if (dashboard) {
    for (const name of ["models", "conditions", "audio", "criteria"])
      for (const kind of ["tab", "panel"]) elements.set(`results-${kind}-${name}`, new Element());
    elements.set("audio", Object.assign(new Element("audio"), {src: "unchanged.wav", currentTime: 2.5}));
  }
  const context = vm.createContext({document: {getElementById: id => elements.get(id), createElement: tag => new Element(tag),
    querySelectorAll: selector => selector === "[data-overview-help]" ? headers : []}});
  context.window = context; context.innerWidth = 1024; context.innerHeight = 768;
  vm.runInContext(source, context);
  return {elements, headers, ui: context.KowwResultsUI,
    rows: id => elements.get(id).tBodies[0].children.map(row => row.children.map(cell => cell.textContent))};
}
function sample() {
  const stats = (successes, calls) => ({successes, calls, recall: calls ? successes / calls : null});
  return {
    selected_candidate: "mean",
    deployment_candidate: {id: "mean", members: [{path: "a.onnx"}, {path: "b.onnx"}, {path: "c.onnx"}]},
    candidate_definitions: [
      {id: "member1_seed1", kind: "single"}, {id: "member2_seed2", kind: "single"}, {id: "member3_seed3", kind: "single"},
      {id: "mean", kind: "ensemble", members: [{path: "a.onnx"}, {path: "b.onnx"}, {path: "c.onnx"}]}
    ],
    models: [
      {id: "member2_seed2", rank: 1, kind: "single", primary: {...stats(42, 45), latency_p95_s: 0.2}, clean: stats(3, 3), noise_cell_mean: 0.9, negative: {events: 3, hours: 30, fa_per_hour: 0.1}, conditions: {"environment:10": stats(10, 12)}},
      {id: "member1_seed1", rank: 2, kind: "single", primary: {...stats(41, 45), latency_p95_s: 0.18}, clean: stats(3, 3), noise_cell_mean: 0.88, negative: {events: 1, hours: 30, fa_per_hour: 1 / 30}, conditions: {"music:5": stats(7, 8)}},
      {id: "member3_seed3", rank: 2, kind: "single", primary: stats(41, 45), clean: stats(0, 0), noise_cell_mean: null, negative: {events: null, hours: 0, fa_per_hour: null}, conditions: {}},
      {id: "mean", kind: "ensemble", primary: {...stats(43, 45), latency_p95_s: 0.16}, clean: stats(3, 3), noise_cell_mean: 0.92, negative: {events: 1, hours: 30, fa_per_hour: 1 / 30}, conditions: {"environment:20.0": stats(12, 12), "environment:15": stats(11, 12), "environment:10": stats(10, 12), "environment:5": stats(9, 12), "environment:0": stats(0, 12), "music:10": stats(0, 0)}, reverb_conditions: {strong: {clean: stats(2, 3), conditions: {"music:5": stats(7, 8)}}}}
    ],
    candidate_adoptions: {
      member2_seed2: {passed: false, reasons: ["FA/hour target failed or unmeasured", "environment SNR5 below target or unmeasured"]},
      member1_seed1: {passed: true}, member3_seed3: {passed: false, reasons: ["Clean recall below target or unmeasured"]}, mean: {passed: true}
    }
  };
}
test("model names follow declared identities rather than ranked position", () => {
  const {ui} = browser(), report = sample();
  assert.equal(ui.modelLabel(report, "member2_seed2"), "모델 2");
  assert.equal(ui.modelLabel(report, "mean"), "앙상블");
  const collision = {candidate_definitions: [{id: "mean", kind: "single"}, {id: "_mean", kind: "ensemble"}], models: []};
  assert.equal(ui.modelLabel(collision, "mean"), "모델 1");
  assert.equal(ui.modelLabel(collision, "_mean"), "앙상블");
});
test("dashboard tabs switch one panel without replacing audio or resetting the comparison source", () => {
  const {ui, elements} = browser({dashboard: true});
  const audio = elements.get("audio");
  const report = sample();
  report.runtime_recommendations = {schema_version: 1, candidates: [{
    id: "mean", runtime: {threshold: .8, consecutive_frames: 2}, metrics: report.models[3]
  }]};
  ui.renderSummary(report);
  const source = elements.get("overview-source");
  assert.equal(source.value, "recommended");
  source.value = "baseline"; source.onchange();
  assert.equal(elements.get("results-panel-models").hidden, false);
  assert.equal(elements.get("results-panel-conditions").hidden, true);
  assert.equal(elements.get("results-panel-audio").hidden, true);
  elements.get("results-tab-audio").onclick();
  assert.equal(elements.get("results-panel-models").hidden, true);
  assert.equal(elements.get("results-panel-audio").hidden, false);
  assert.equal(elements.get("results-tab-audio").attributes["aria-selected"], "true");
  elements.get("results-tab-models").onclick();
  assert.equal(elements.get("results-panel-models").hidden, false);
  assert.equal(source.value, "baseline");
  assert.equal(elements.get("audio"), audio);
  assert.equal(audio.src, "unchanged.wav"); assert.equal(audio.currentTime, 2.5);
});
test("dashboard tabs support keyboard navigation", () => {
  const {ui, elements} = browser({dashboard: true});
  let prevented = false;
  elements.get("results-tab-models").onkeydown({key: "ArrowLeft", preventDefault() { prevented = true; }});
  assert.equal(prevented, true); assert.equal(ui.dashboardState().view, "criteria");
  assert.equal(elements.get("results-tab-criteria").focused, true);
  elements.get("results-tab-models").onkeydown({key: "End", preventDefault() {}});
  assert.equal(ui.dashboardState().view, "criteria");
  assert.equal(elements.get("results-tab-criteria").tabIndex, 0);
  assert.equal(elements.get("results-tab-models").tabIndex, -1);
});
test("legacy reports label stable model numbers without changing source order", () => {
  const {ui} = browser(), report = {models: [{id: "member10", kind: "single"}, {id: "member2", kind: "single"}, {id: "mean", kind: "ensemble"}]};
  assert.equal(ui.modelLabel(report, "member2"), "모델 1");
  assert.equal(ui.modelLabel(report, "member10"), "모델 2");
  assert.equal(report.models[0].id, "member10");
});
test("plain labels describe background, relative noise level and reverberation", () => {
  const {ui} = browser();
  assert.equal(ui.noiseLabel("competing_speech"), "다른 사람 말소리");
  assert.equal(ui.noiseLabel("clean"), "조용한 환경");
  assert.equal(ui.snrLabel(0), "목소리와 같은 크기 · SNR 0 dB");
  assert.equal(ui.snrLabel(20), "소음 아주 작음 · SNR 20 dB");
  assert.equal(ui.snrLabel(null), "소음 크기 · SNR — dB");
  assert.equal(ui.reverbLabel("strong"), "강한 울림");
});
test("evaluation comparison preserves saved statistics and decisions", () => {
  const {ui, rows, elements} = browser(), report = sample(), before = JSON.stringify(report);
  ui.renderSummary(report);
  const model = elements.get("storySummary").tBodies[0].children[1].children;
  assert.equal(elements.get("overview-source").value, "baseline");
  assert.deepEqual(rows("storySummary")[1].slice(0, 7), ["모델 2", "93.33%", "100.00%", "결과 없음", "결과 없음", "0.1회", "0.20초"]);
  assert.equal(rows("storySummary")[1][8], "미달");
  assert.equal(rows("storySummary")[3][8], "충족");
  assert.match(model[5].title, /3회.*30/);
  assert.equal(model[6].children.length, 0);
  assert.equal(JSON.stringify(report), before);
});
test("hourly false alarms trim only unnecessary trailing zeros", () => {
  const {ui, rows} = browser(), report = sample();
  report.models[0].negative.fa_per_hour = 1.200001;
  report.models[1].negative.fa_per_hour = 1 / 30;
  report.models[2].negative.fa_per_hour = 0;
  report.models[3].negative.fa_per_hour = 0.101;
  ui.renderSummary(report);
  assert.deepEqual(rows("storySummary").map(row => row[5]), ["0.033회", "1.2회", "0회", "0.101회"]);
});
test("runtime model and evaluation selection are not conflated", () => {
  const {ui, elements} = browser(), report = sample();
  report.selected_candidate = "member2_seed2";
  ui.renderSummary(report); ui.renderConditions(report);
  assert.equal(elements.get("overview-model").value, "mean");
  assert.equal(elements.get("condition-model").value, "member2_seed2");
});
test("criteria show frozen original goals without repeating measured noise results", () => {
  const {ui, rows, elements} = browser(), report = sample();
  report.adoption_criteria = {schema_version: 1, clean_recall: 0.98, snr10_recall: 0.95, snr5_recall: 0.9, fa_per_hour: 0.1, latency_p95_s: 0.5, minimum_negative_hours: 20};
  const before = JSON.stringify(report);
  ui.renderSummary(report);
  const text = node => [node.textContent, ...node.children.map(text)].join(" ");
  const criteria = text(elements.get("evaluationCriteria"));
  assert.match(criteria, /98(?:\.00)?%/);
  assert.match(criteria, /95(?:\.00)?%/);
  assert.match(criteria, /90(?:\.00)?%/);
  assert.match(criteria, /0\.1회/);
  assert.match(criteria, /0\.5초/);
  assert.match(criteria, /20시간/);
  assert.doesNotMatch(criteria, /83\.33%|75\.00%/);
  assert.equal(rows("storySummary")[1][8], "미달");
  assert.equal(JSON.stringify(report), before);
});
test("the comparison uses recorded decisions at threshold boundaries", () => {
  const {ui, rows} = browser(), report = sample();
  report.adoption_criteria = {schema_version: 1, clean_recall: 1, snr10_recall: 10 / 12, snr5_recall: 0.9, fa_per_hour: 0.1, latency_p95_s: 0.2, minimum_negative_hours: 30};
  ui.renderSummary(report);
  assert.equal(rows("storySummary")[1][8], "미달");
  report.candidate_adoptions.member2_seed2.passed = true; ui.renderSummary(report);
  assert.equal(rows("storySummary")[1][8], "충족");
});
test("legacy or invalid criteria never silently borrow current default targets", () => {
  const {ui, rows, elements} = browser(), report = sample();
  for (const criteria of [undefined, {}, {schema_version: 1, clean_recall: 0.98}, {schema_version: 999}, {schema_version: 1, clean_recall: 2, snr10_recall: 0.95, snr5_recall: 0.9, fa_per_hour: 0.1, latency_p95_s: 0.5, minimum_negative_hours: 20}]) {
    report.adoption_criteria = criteria; ui.renderSummary(report);
    const target = elements.get("evaluationCriteria");
    assert.ok(target.hidden || target.children.length === 0);
    assert.equal(rows("storySummary")[1][8], "미달");
  }
});
test("conditions show actual cells in fixed noise order including zero successes", () => {
  const {ui, rows, elements} = browser(), report = sample(), before = JSON.stringify(report);
  ui.renderConditions(report);
  assert.equal(elements.get("condition-model").value, "mean");
  assert.deepEqual(rows("conditions")[0], ["환경소음", "100.00%\n12 / 12회", "91.67%\n11 / 12회", "83.33%\n10 / 12회", "75.00%\n9 / 12회", "0.00%\n0 / 12회"]);
  assert.equal(rows("conditions")[1][3], "시험 안 함");
  assert.equal(rows("conditions")[2][1], "결과 없음");
  assert.equal(elements.get("condition-clean").textContent, "배경 소음 없이: 100.00%\n3 / 3회");
  assert.equal(JSON.stringify(report), before);
});
test("room selector switches to the recorded room condition without recomputing means", () => {
  const {ui, rows, elements} = browser(), report = sample();
  ui.renderConditions(report);
  elements.get("condition-reverb").value = "strong"; elements.get("condition-reverb").onchange();
  assert.equal(rows("conditions")[1][4], "87.50%\n7 / 8회");
  assert.equal(rows("conditions")[0][3], "결과 없음");
  assert.equal(elements.get("condition-clean").textContent, "배경 소음 없이: 66.67%\n2 / 3회");
  elements.get("condition-model").value = "member2_seed2"; elements.get("condition-model").onchange();
  assert.equal(elements.get("condition-reverb").value, "none");
  assert.equal(rows("conditions")[0][3], "83.33%\n10 / 12회");
});
test("missing results stay distinct from measured zero and explicit empty trials", () => {
  const {ui, rows, elements} = browser(), report = sample();
  delete report.candidate_adoptions.member3_seed3;
  ui.renderSummary(report);
  assert.equal(rows("storySummary")[2][2], "시험 안 함");
  assert.equal(rows("storySummary")[2][3], "결과 없음");
  assert.equal(rows("storySummary")[2][5], "결과 없음");
  assert.equal(rows("storySummary")[2][8], "기록 없음");
  ui.renderConditions({models: []});
  assert.equal(elements.get("condition-model").disabled, true);
});
test("audio reactions expose detection and time while retaining separate numeric details", () => {
  const {ui, rows} = browser(), report = sample();
  const clip = {kind: "positive", results: {member2_seed2: {success: false, max_score: 0.97, clean_score_drop: -0.02, latency_s: null}, mean: {success: true, max_score: 0.99991, clean_score_drop: 0, latency_s: 0.08}}};
  const before = JSON.stringify(clip); ui.renderResponses(report, clip);
  assert.deepEqual(rows("responses"), [["모델 2", "놓침", "—"], ["앙상블", "감지", "0.08초"]]);
  assert.deepEqual(rows("scores"), [["모델 2", "0.9700", "-0.0200"], ["앙상블", "0.9999", "0.0000"]]);
  assert.equal(JSON.stringify(clip), before);
  ui.renderResponses(report, {kind: "positive", results: {mean: {success: true, latency_s: -0.08}}});
  assert.deepEqual(rows("responses"), [["앙상블", "감지", "0.08초 전"]]);
  ui.renderResponses(report, {kind: "negative", results: {mean: {false_activations: 0, max_score: 0.2}}});
  assert.deepEqual(rows("responses"), [["앙상블", "오작동 0회", "—"]]);
});
test("audio titles avoid internal filenames and do not claim a negative clip contains a call", () => {
  const {ui} = browser();
  assert.equal(ui.clipLabel({kind: "positive", category: "environment", original_order: 12, sample_id: "opaque-file"}), "음성 12 · 환경소음");
  assert.equal(ui.clipLabel({kind: "negative", category: "music", original_order: 12}), "호출어 없는 음원 · 음악·전자음");
});
test("missing, invalid and cache-only recommendations are presented honestly", () => {
  const {ui, rows, elements} = browser(), report = sample(), model = report.models[0];
  ui.renderSummary(report);
  assert.equal(elements.get("overview-source").value, "baseline");
  report.runtime_recommendations = {schema_version: 1, candidates: [{
    id: model.id, runtime: {threshold: .9999999, consecutive_frames: 2}, metrics: model,
    adoption: {passed: false}, deployment_backend_verified: false
  }]};
  ui.renderSummary(report);
  elements.get("overview-source").value = "recommended"; elements.get("overview-source").onchange();
  assert.deepEqual(rows("storySummary")[1].slice(7), ["0.9999999 · 2회", "미달"]);
  assert.match(elements.get("overview-note").textContent, /저장.*점수/);
  report.runtime_recommendations.candidates[0].runtime.threshold = 1; ui.renderSummary(report);
  assert.equal(elements.get("overview-source").value, "baseline");
});
test("one comparison shows measured settings, weakest conditions and goal status in both layouts", () => {
  const {ui, elements, rows} = browser(), report = sample();
  report.runtime = {threshold: .5, consecutive_frames: 1};
  Object.assign(report.models[0], {clean: {recall: .999}, negative: {fa_per_hour: 9.399818}});
  const hard = (cell_mean, worst_recall) => ({complete: true, cell_mean, worst_recall,
    conditions: {"strong:competing_speech:5": {recall: worst_recall, reverb_condition: "strong", category: "competing_speech", snr_db: 5}}});
  report.runtime_recommendations = {schema_version: 1, recommendation_criteria: {fa_per_hour: 1}, adoption_criteria: {fa_per_hour: .1}, candidates: [
    {id: "mean", recommendation_available: true, runtime: {threshold: .97, consecutive_frames: 3}, passed: false,
      adoption: {passed: false, reasons: ["FA/hour target failed or unmeasured"]},
      metrics: {clean: {recall: .995}, hard_conditions: hard(.73, .5), negative: {fa_per_hour: .9}, primary: {recall: .81, latency_p95_s: -.02}}},
    {id: "member2_seed2", recommendation_available: false, recommendation_reasons: ["FA/hour target failed or unmeasured"],
      runtime: {threshold: .99, consecutive_frames: 5}, adoption: {passed: false, reasons: ["Latency target failed or unmeasured"]},
      metrics: {clean: {recall: .9855}, hard_conditions: hard(.62785, 0), negative: {fa_per_hour: 1.199976889}, primary: {recall: .69, latency_p95_s: .44}},
      baseline: {runtime: report.runtime, metrics: {hard_conditions: hard(.783125, .5), clean: {recall: .8}, negative: {fa_per_hour: 8}}}},
    {id: "member3_seed3", recommendation_available: false, runtime: {threshold: .9, consecutive_frames: 2}, metrics: null}
  ]};
  const before = JSON.stringify(report); ui.renderSummary(report);
  assert.equal(elements.get("overview-source").value, "recommended");
  assert.deepEqual(rows("storySummary")[3], ["앙상블", "81.00%", "99.50%", "73.00%", "50.00%", "0.9회", "0.02초 전", "0.97 · 3회", "미달"]);
  assert.deepEqual(rows("storySummary")[1].slice(1), ["69.00%", "98.55%", "62.79%", "0.00%", "1.2회", "0.44초", "0.99 · 5회", "미달"]);
  assert.equal(elements.get("overview-settings-label").textContent, "추천 감지 설정");
  for (const index of [0, 2]) {
    assert.deepEqual(rows("storySummary")[index].slice(1, 7), Array(6).fill("결과 없음"));
    assert.equal(rows("storySummary")[index][8], "기록 없음");
  }
  const status = "추천 기준 미달 · 오호출 한도 1회/시간";
  const modelCells = () => elements.get("storySummary").tBodies[0].children[1].children;
  const details = cell => cell.children.map(node => node.textContent).join(" ");
  assert.equal(modelCells()[0].children.find(node => node.className === "recommendation-status")?.textContent, status);
  assert.equal(modelCells()[4].children.length, 0);
  assert.equal(modelCells()[8].title, "반응 시간");
  const mobileMetrics = () => elements.get("storySummaryMetrics").children.map(item =>
    [fullText(item.children[0]), item.children[1].textContent]);
  assert.deepEqual(mobileMetrics()[2], ["강한 소음·울림", "73.00%"]);
  elements.get("overview-model").value = "member2_seed2"; elements.get("overview-model").onchange();
  assert.deepEqual(mobileMetrics(), [
    ["추천 상태", status], ["종합 감지율", "69.00%"], ["소음·울림 없음", "98.55%"], ["강한 소음·울림", "62.79%"],
    ["최약 조건", "0.00%"], ["오호출/시간", "1.2회"], ["반응 시간", "0.44초"],
    ["추천 감지 설정", "0.99 · 5회"], ["평가 목표", "미달"]
  ]);
  assert.equal(elements.get("storySummaryMetrics").children[0].className, "overview-status-row");
  assert.equal(elements.get("storySummaryMetrics").children[4].children[1].children.length, 0);
  elements.get("overview-source").value = "baseline"; elements.get("overview-source").onchange();
  assert.deepEqual(rows("storySummary")[1], ["모델 2", "93.33%", "99.90%", "78.31%", "50.00%", "9.4회", "0.20초", "0.5 · 1회", "미달"]);
  assert.equal(elements.get("overview-settings-label").textContent, "평가 당시 감지 설정");
  assert.deepEqual(mobileMetrics()[6], ["평가 당시 감지 설정", "0.5 · 1회"]);
  assert.equal(mobileMetrics()[0][0], "종합 감지율");
  assert.equal(JSON.stringify(report), before);

  report.runtime_recommendations.candidates[1].baseline.runtime = {...report.runtime, consecutive_frames: 2};
  ui.renderSummary(report);
  assert.deepEqual(rows("storySummary")[1].slice(3, 5), ["결과 없음", "결과 없음"]);
  elements.get("overview-source").value = "recommended"; elements.get("overview-source").onchange();
  const candidate = report.runtime_recommendations.candidates[1];
  candidate.metrics.hard_conditions.complete = false;
  candidate.recommendation_reasons = ["Hard noise/reverberation conditions incomplete"];
  ui.renderSummary(report);
  assert.deepEqual(rows("storySummary")[1].slice(3, 5), ["자료 부족", "자료 부족"]);
  assert.equal(modelCells()[3].children.length, 0);
  assert.equal(modelCells()[4].children.length, 0);
  assert.match(details(modelCells()[0]), /강소음·울림 시험 자료 부족/);
});
test("metric help opens only from headings and describes the archived comparison policy", () => {
  const {ui, elements, headers} = browser(), report = sample();
  report.runtime = {threshold: .5, consecutive_frames: 1};
  const model = report.models.find(row => row.id === "mean");
  model.hard_conditions = {complete: true, cell_mean: .8, worst_recall: .6};
  report.runtime_recommendations = {schema_version: 1,
    search_policy: {hard_conditions: [
      {reverb_condition: "strong", category: "environment", snr_db: 5},
      {reverb_condition: "moderate", category: "clean", snr_db: null}
    ]},
    candidates: [{
      id: model.id, runtime: {threshold: .9, consecutive_frames: 3}, recommendation_available: true,
      metrics: {noise_cell_mean: .92, hard_conditions: {complete: true, cell_mean: .7, worst_recall: .64, conditions: [
      {reverb_condition: "strong", category: "competing_speech", snr_db: 5, recall: .64},
      {reverb_condition: "moderate", category: "music", snr_db: 10, recall: .64}
    ]}}
  }]};
  const saved = JSON.stringify(report);
  ui.renderSummary(report);

  const dialog = elements.get("overview-help");
  const label = () => fullText(elements.get("overview-help-title"));
  const content = () => fullText(elements.get("overview-help-content"));
  const close = () => elements.get("overview-help-close").dispatch("click");
  const desktopRow = () => elements.get("storySummary").tBodies[0].children[3].children;
  const mobileRow = name => elements.get("storySummaryMetrics").children.find(row => fullText(row.children[0]) === name);
  const mobileCell = name => mobileRow(name).children[1];
  function checkPlainValues(hard, worst) {
    for (const [cell, expected] of [[desktopRow()[3], hard], [desktopRow()[4], worst],
      [mobileCell("강한 소음·울림"), hard], [mobileCell("최약 조건"), worst]]) {
      assert.equal(cell.textContent, expected);
      assert.equal(cell.children.length, 0);
      assert.equal(cell.title, undefined);
      assert.equal(cell.attributes["aria-haspopup"], undefined);
      cell.dispatch("click");
      assert.equal(dialog.open, false);
    }
  }
  checkPlainValues("70.00%", "64.00%");

  headers[0].dispatch("click");
  assert.equal(dialog.open, true);
  assert.match(label(), /강한 소음·울림/);
  assert.match(content(), /평균/);
  assert.match(content(), /2개.*환경소음.*SNR 5 dB/s);
  assert.doesNotMatch(content(), /70\.00%|92\.00%|앙상블/);
  const hardHelp = content();
  close();
  assert.equal(dialog.open, false);
  assert.equal(headers[0].attributes["aria-expanded"], "false");
  assert.equal(headers[0].focused, true);

  headers[1].dispatch("click");
  assert.equal(dialog.open, true);
  assert.match(label(), /최약 조건/);
  assert.match(content(), /낮/);
  assert.match(content(), /모델.*(다르|다를)/);
  assert.doesNotMatch(content(), /64\.00%|전자음|5 dB/);
  const worstHelp = content();
  close();

  const compactLabel = mobileRow("최약 조건").children[0].children.find(node => node.tagName === "button");
  assert.ok(compactLabel);
  compactLabel.dispatch("click");
  assert.equal(dialog.open, true);
  assert.equal(content(), worstHelp);
  const source = elements.get("overview-source");
  source.value = "baseline"; source.onchange();
  assert.equal(dialog.open, false);
  assert.equal(compactLabel.attributes["aria-expanded"], "false");
  checkPlainValues("80.00%", "60.00%");
  const compactHard = mobileRow("강한 소음·울림").children[0].children.find(node => node.tagName === "button");
  compactHard.dispatch("click");
  assert.equal(dialog.open, true);
  assert.equal(content(), hardHelp);
  elements.get("overview-model").value = "member2_seed2"; elements.get("overview-model").onchange();
  assert.equal(dialog.open, false);
  assert.equal(compactHard.attributes["aria-expanded"], "false");
  assert.equal(JSON.stringify(report), saved);
});
test("baseline aggregates only the recorded cells named by the archived policy", () => {
  const {ui, elements, rows} = browser(), report = sample();
  report.runtime = {threshold: .5, consecutive_frames: 1};
  const model = report.models.find(row => row.id === "mean");
  report.runtime_recommendations = {schema_version: 1, search_policy: {hard_conditions: [
    {reverb_condition: "none", category: "environment", snr_db: 5},
    {reverb_condition: "strong", category: "clean", snr_db: null}
  ]}, candidates: [{id: model.id, runtime: {threshold: .9, consecutive_frames: 3}, metrics: {clean: model.clean},
    baseline: {runtime: {threshold: .5, consecutive_frames: 2}, metrics: {hard_conditions: {complete: true, cell_mean: .99, worst_recall: .99}}}}]};
  const before = JSON.stringify(report);
  ui.renderSummary(report);
  assert.deepEqual(rows("storySummary")[3].slice(3, 5), ["결과 없음", "결과 없음"]);
  const source = elements.get("overview-source");
  source.value = "baseline"; source.onchange();
  assert.deepEqual(rows("storySummary")[3].slice(3, 5), ["70.83%", "66.67%"]);
  assert.equal(JSON.stringify(report), before);
  delete model.reverb_conditions.strong.clean;
  ui.renderSummary(report);
  assert.deepEqual(rows("storySummary")[3].slice(3, 5), ["자료 부족", "자료 부족"]);
});
test("story criteria disclose the recorded recommendation policy instead of inventing defaults for historical reports", () => {
  const {ui, elements} = browser(), report = sample();
  ui.renderSummary(report);
  assert.equal(elements.get("searchCriteria").children[0].children[1].textContent, "기록 없음");
  report.runtime_recommendations = {schema_version: 1, search_policy: {threshold_step: .01, consecutive_frames: [1, 2, 3, 4, 5, 6]},
    recommendation_criteria: {clean_recall: .98, fa_per_hour: 1, latency_p95_s: .5}, candidates: []};
  ui.renderSummary(report);
  const cells = elements.get("searchCriteria").children.map(row => row.children[1].textContent);
  assert.deepEqual(cells.slice(0, 5), ["0.01", "1~6회 비교", "98.00% 이상", "시간당 1회 이하", "95%가 0.5초 이내"]);
});
test("compact condition rows use the same stored cells as the full table for each noise selection", () => {
  const {ui, elements} = browser(), report = sample(), before = JSON.stringify(report);
  ui.renderConditions(report);
  assert.equal(elements.get("conditionMetrics").children[0].children[1].textContent, "83.33%\n10 / 12회");
  elements.get("condition-snr").value = "0"; elements.get("condition-snr").onchange();
  assert.equal(elements.get("conditionMetrics").children[0].children[1].textContent, "0.00%\n0 / 12회");
  assert.equal(JSON.stringify(report), before);
});
console.log(`${cases.length} result presentation tests passed`);
