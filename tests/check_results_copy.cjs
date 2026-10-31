"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/results_ui.js"), "utf8");
const cases = [];
function test(name, run) { run(); cases.push(name); }
function browser() {
  class Element {
    constructor(tag = "div") { this.tagName = tag; this.children = []; this.value = ""; this.textContent = ""; this.disabled = false; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
  }
  const elements = new Map();
  for (const id of ["summary", "candidateStatus", "candidateReasons", "denominators", "criteriaSummary", "criteriaNote", "targetControls", "target-model", "targetComparison", "responses", "scores", "condition-model", "condition-reverb", "condition-clean", "conditions", "runtimeRecommendations", "recommendations", "recommendationScope", "recommendationNote", "baselineRuntime", "recommendationFailures", "recommendationReasons"]) elements.set(id, new Element());
  for (const id of ["summary", "responses", "scores", "conditions", "targetComparison", "recommendations"]) elements.get(id).tBodies = [new Element("tbody")];
  const context = vm.createContext({document: {getElementById: id => elements.get(id), createElement: tag => new Element(tag)}});
  context.window = context; vm.runInContext(source, context);
  return {elements, ui: context.KowwResultsUI, rows: id => elements.get(id).tBodies[0].children.map(row => row.children.map(cell => cell.textContent))};
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
test("summary preserves rank, counts, rates, latency and recorded decisions", () => {
  const {ui, rows, elements} = browser(), report = sample(), before = JSON.stringify(report);
  ui.renderSummary(report);
  assert.deepEqual(rows("summary")[0], ["1", "모델 2", "93.33%\n42 / 45회", "100.00%\n3 / 3회", "90.00%", "0.100회\n3회 / 30.00시간", "0.20초", "미달"]);
  assert.equal(rows("summary")[2][0], "2");
  assert.equal(rows("summary")[3][0], "—");
  assert.equal(rows("summary")[3][7], "충족");
  assert.equal(elements.get("candidateStatus").textContent, "평가 당시 실행 모델: 앙상블 · 모델 3개 점수 평균");
  assert.match(elements.get("summary").tBodies[0].children[0].children[6].title, /95%/);
  assert.equal(elements.get("candidateReasons").children[0].textContent, "모델 2: 시간당 오작동 횟수 · 환경소음 성공률 · SNR 5 dB");
  assert.equal(JSON.stringify(report), before);
});
test("runtime model and evaluation selection are not conflated", () => {
  const {ui, elements} = browser(), report = sample();
  report.selected_candidate = "member2_seed2";
  ui.renderSummary(report); ui.renderConditions(report);
  assert.equal(elements.get("candidateStatus").textContent, "평가 당시 실행 모델: 앙상블 · 모델 3개 점수 평균");
  assert.equal(elements.get("condition-model").value, "member2_seed2");
});
test("criteria show frozen targets and compare exact measurements for the selected model", () => {
  const {ui, rows, elements} = browser(), report = sample();
  report.adoption_criteria = {schema_version: 1, clean_recall: 0.98, snr10_recall: 0.95, snr5_recall: 0.9, fa_per_hour: 0.1, latency_p95_s: 0.5, minimum_negative_hours: 20};
  const before = JSON.stringify(report);
  ui.renderSummary(report);
  assert.equal(elements.get("criteriaSummary").textContent, "목표 기준: 조용할 때 98.00% 이상 · 소음 중간(SNR 10 dB) 95.00% 이상 · 소음 큼(SNR 5 dB) 90.00% 이상 · 시간당 오작동 0.1회 이하 · 반응 시간 0.5초 이내");
  assert.match(elements.get("criteriaNote").textContent, /울림을 추가하지 않은.*각각.*20시간 이상.*모든 기준/);
  assert.equal(elements.get("target-model").value, "mean");
  assert.equal(elements.get("candidateReasons").children.length, 0);
  assert.equal(rows("targetComparison").length, 10);
  assert.deepEqual(rows("targetComparison")[0], ["조용한 환경 성공률", "98.00% 이상", "100.00%", "충족"]);
  assert.deepEqual(rows("targetComparison")[1], ["환경소음 · 소음 중간 (SNR 10 dB)", "95.00% 이상", "83.33%", "미달"]);
  assert.deepEqual(rows("targetComparison")[2], ["환경소음 · 소음 큼 (SNR 5 dB)", "90.00% 이상", "75.00%", "미달"]);
  assert.deepEqual(rows("targetComparison")[7], ["시간당 오작동", "0.1회 이하", "0.033회", "충족"]);
  assert.deepEqual(rows("targetComparison")[8], ["반응 시간 · 감지된 호출의 95%", "0.5초 이하", "0.16초", "충족"]);
  assert.deepEqual(rows("targetComparison")[9], ["오작동 시험 시간", "20시간 이상", "30시간", "충족"]);
  elements.get("target-model").value = "member3_seed3"; elements.get("target-model").onchange();
  assert.deepEqual(rows("targetComparison")[0], ["조용한 환경 성공률", "98.00% 이상", "결과 없음", "결과 없음"]);
  assert.equal(rows("targetComparison")[9][3], "미달");
  assert.equal(JSON.stringify(report), before);
});
test("target comparisons retain boundary inclusivity without changing recorded decisions", () => {
  const {ui, rows, elements} = browser(), report = sample();
  report.adoption_criteria = {schema_version: 1, clean_recall: 1, snr10_recall: 10 / 12, snr5_recall: 0.9, fa_per_hour: 0.1, latency_p95_s: 0.2, minimum_negative_hours: 30};
  ui.renderSummary(report);
  elements.get("target-model").value = "member2_seed2"; elements.get("target-model").onchange();
  for (const index of [0, 1, 7, 8, 9]) assert.equal(rows("targetComparison")[index][3], "충족");
  assert.equal(rows("summary")[0][7], "미달");
});
test("legacy or invalid criteria never silently borrow current default targets", () => {
  const {ui, rows, elements} = browser(), report = sample();
  for (const criteria of [undefined, {}, {schema_version: 1, clean_recall: 0.98}, {schema_version: 999}, {schema_version: 1, clean_recall: 2, snr10_recall: 0.95, snr5_recall: 0.9, fa_per_hour: 0.1, latency_p95_s: 0.5, minimum_negative_hours: 20}]) {
    report.adoption_criteria = criteria; ui.renderSummary(report);
    assert.equal(elements.get("criteriaSummary").textContent, "평가 당시 목표 기준이 기록되지 않아 상세 비교를 표시할 수 없습니다.");
    assert.equal(elements.get("targetControls").hidden, true);
    assert.equal(elements.get("targetComparison").hidden, true);
    assert.equal(rows("targetComparison").length, 0);
    assert.equal(rows("summary")[0][7], "미달");
  }
});
test("rounding cannot disguise a measurement slightly outside its target", () => {
  const {ui, rows, elements} = browser(), report = sample();
  report.adoption_criteria = {schema_version: 1, clean_recall: 0.98, snr10_recall: 0.95, snr5_recall: 0.9, fa_per_hour: 0.1, latency_p95_s: 0.5, minimum_negative_hours: 20};
  const model = report.models.find(value => value.id === "mean");
  model.clean.recall = 0.979999; model.negative.fa_per_hour = 0.1000004;
  model.primary.latency_p95_s = 0.500001; model.negative.hours = 19.9999;
  ui.renderSummary(report);
  assert.deepEqual(rows("targetComparison")[0].slice(1), ["98.00% 이상", "97.9999%", "미달"]);
  assert.deepEqual(rows("targetComparison")[7].slice(1), ["0.1회 이하", "0.1000004회", "미달"]);
  assert.deepEqual(rows("targetComparison")[8].slice(1), ["0.5초 이하", "0.500001초", "미달"]);
  assert.deepEqual(rows("targetComparison")[9].slice(1), ["20시간 이상", "19.9999시간", "미달"]);
  model.negative.fa_per_hour = 0.1; ui.renderSummary(report);
  assert.deepEqual(rows("targetComparison")[7].slice(1), ["0.1회 이하", "0.1회", "충족"]);
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
  assert.equal(rows("summary")[2][3], "시험 안 함");
  assert.equal(rows("summary")[2][4], "결과 없음");
  assert.equal(rows("summary")[2][5], "결과 없음");
  assert.equal(rows("summary")[2][7], "결과 없음");
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
test("recommendations show independently chosen settings and separate metrics without changing original results", () => {
  const {ui, rows, elements} = browser(), report = sample();
  report.runtime = {threshold: .5, consecutive_frames: 1};
  report.runtime_recommendations = {schema_version: 1, candidates: report.models.map((model, index) => ({id: model.id,
    runtime: {threshold: .8 + index * .01, consecutive_frames: index + 1}, recommendation_available: true,
    metrics: {...model, noise_cell_mean: .95, hard_conditions: {complete: true, cell_mean: .85, worst_recall: .7,
      conditions: {"strong:environment:5": {recall: .7, reverb_condition: "strong", category: "environment", snr_db: 5}}}},
    passed: index !== 0, deployment_backend_verified: true}))};
  const before = JSON.stringify(report); ui.renderSummary(report);
  assert.equal(elements.get("runtimeRecommendations").hidden, false);
  assert.equal(rows("recommendations").length, 4);
  assert.deepEqual(rows("recommendations")[0], ["모델 2", "추천", "0.8", "1회", "100.00%", "85.00%", "70.00%", "0.100회", "미달"]);
  assert.equal(elements.get("recommendations").tBodies[0].children[0].children[6].title, "강한 울림 · 환경소음 · SNR 5 dB");
  assert.equal(rows("summary")[0][4], "90.00%");
  assert.match(elements.get("recommendationScope").textContent, /강한 소음과 울림/);
  assert.doesNotMatch(elements.get("recommendationNote").textContent, /추천 기준을 만족하는 조합이 없는/);
  assert.match(elements.get("baselineRuntime").textContent, /감지 기준 0.5 · 연속 1회/);
  assert.equal(JSON.stringify(report), before);
});
test("missing, invalid and cache-only recommendation evidence is never shown as confirmed success", () => {
  const {ui, rows, elements} = browser(), report = sample();
  ui.renderSummary(report);
  assert.equal(elements.get("runtimeRecommendations").hidden, true);
  assert.equal(rows("recommendations").length, 0);
  const metric = report.models[0];
  report.runtime_recommendations = {schema_version: 1, candidates: [{id: metric.id, runtime: {threshold: .9999999, consecutive_frames: 2}, metrics: metric, adoption: {passed: false}, deployment_backend_verified: false}]};
  ui.renderSummary(report);
  assert.equal(rows("recommendations")[0][2], "0.9999999");
  assert.equal(rows("recommendations")[0][5], "결과 없음");
  assert.equal(rows("recommendations")[0][8], "미달");
  assert.match(elements.get("recommendationNote").textContent, /저장 점수 기준/);
  report.runtime_recommendations.candidates[0].runtime.threshold = 1; ui.renderSummary(report);
  assert.equal(elements.get("runtimeRecommendations").hidden, true);
  assert.equal(elements.get("baselineRuntime").hidden, true);
});

test("a failed protection check is a tradeoff, never a recommendation or automatic microphone gate", () => {
  const {ui, rows, elements} = browser(), report = sample(), model = report.models[0];
  report.runtime_recommendations = {schema_version: 1, candidates: [{id: model.id,
    runtime: {threshold: .999, consecutive_frames: 4}, original_runtime: {threshold: .5, consecutive_frames: 1},
    recommendation_available: false, recommendation_reasons: ["FA/hour target failed or unmeasured"],
    metrics: {...model, hard_conditions: {complete: true, cell_mean: .8, worst_recall: .6}}, passed: false}]};
  const before = JSON.stringify(report); ui.renderSummary(report);
  assert.equal(rows("recommendations")[0][1], "절충 후보");
  assert.equal(rows("recommendations")[0][5], "80.00%");
  assert.match(elements.get("recommendationNote").textContent, /마이크에는 그 모델의 기존 설정을 유지/);
  assert.match(elements.get("recommendationReasons").children[0].textContent, /추천 없음: 시간당 오작동 횟수/);
  assert.equal(JSON.stringify(report), before);
  report.runtime_recommendations.candidates[0].metrics.hard_conditions = {complete: false, cell_mean: null, worst_recall: null};
  report.runtime_recommendations.candidates[0].recommendation_reasons = ["Hard noise/reverberation conditions incomplete"];
  ui.renderSummary(report);
  assert.equal(rows("recommendations")[0][5], "자료 부족");
  assert.match(elements.get("recommendationReasons").children[0].textContent, /강소음·울림 시험 자료 부족/);
});

test("new selection limits are shown separately from the preserved original evaluation goals", () => {
  const {ui, rows, elements} = browser(), report = sample(), model = report.models[0];
  report.runtime_recommendations = {schema_version: 1, method: "joint_cached_hard_conditions_v3",
    search_policy: {threshold_step: .01}, recommendation_criteria: {fa_per_hour: 1},
    adoption_criteria: {fa_per_hour: .1}, candidates: [{id: model.id,
      runtime: {threshold: .99, consecutive_frames: 2}, recommendation_available: true,
      metrics: {...model, negative: {...model.negative, fa_per_hour: .8}, hard_conditions: {complete: true, cell_mean: .84, worst_recall: .66}},
      passed: false, adoption: {passed: false, reasons: ["FA/hour target failed or unmeasured"]}}]};
  const before = JSON.stringify(report); ui.renderSummary(report);
  assert.equal(rows("recommendations")[0][1], "추천");
  assert.equal(rows("recommendations")[0][8], "미달");
  assert.match(elements.get("recommendationScope").textContent, /감지 기준 0.01 간격 · 시간당 오작동 1회 이하/);
  assert.match(elements.get("recommendationNote").textContent, /기존 평가 목표의 오작동 한도는 시간당 0.1회/);
  assert.doesNotMatch(elements.get("recommendationNote").textContent, /추천 기준을 만족하는 조합이 없는/);
  assert.equal(JSON.stringify(report), before);
  report.runtime_recommendations.recommendation_criteria.fa_per_hour = 2;
  ui.renderSummary(report);
  assert.match(elements.get("recommendationScope").textContent, /시간당 오작동 2회 이하/);
});
test("recommended-setting failures describe that setting rather than the preserved baseline", () => {
  const {ui, elements, rows} = browser(), report = sample(), model = report.models[0];
  report.runtime_recommendations = {schema_version: 1, candidates: [{id: model.id,
    runtime: {threshold: .8, consecutive_frames: 2}, metrics: model, passed: false,
    adoption: {passed: false, reasons: ["Latency target failed or unmeasured", "music SNR10 below target or unmeasured"]}}]};
  const before = JSON.stringify(report); ui.renderSummary(report);
  assert.equal(elements.get("recommendationFailures").hidden, false);
  assert.equal(elements.get("recommendationReasons").children[0].textContent, "모델 2: 반응 시간 · 음악·전자음 성공률 · SNR 10 dB");
  assert.equal(elements.get("candidateReasons").children[0].textContent, "모델 2: 시간당 오작동 횟수 · 환경소음 성공률 · SNR 5 dB");
  assert.equal(rows("summary")[0][4], "90.00%");
  assert.equal(JSON.stringify(report), before);
  report.runtime_recommendations.candidates[0].passed = true; ui.renderSummary(report);
  assert.equal(elements.get("recommendationFailures").hidden, true);
  assert.equal(elements.get("recommendationReasons").children.length, 0);
});
console.log(`${cases.length} result presentation tests passed`);
