"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio.html"), "utf8");
const flush = async () => { for (let index = 0; index < 12; index++) await new Promise(resolve => setImmediate(resolve)); };
const tests = [];
async function test(name, action) { await action(); tests.push(name); }

function browser({ noBoot = false, initialState, recipeError = false, recipeOverrides = {} } = {}) {
  const elements = new Map(), calls = [], intervals = [];
  class Element {
    constructor(tag = "div") { this.tagName = tag; this.value = ""; this.textContent = ""; this.children = []; this.handlers = {}; this.attributes = {}; this.hidden = false; this.disabled = false; this.open = false; this.srcWrites = 0; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, handler) { this.handlers[name] = handler; }
    setAttribute(name, value) { if (name === "src") this.srcWrites++; this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name] ?? null; }
    removeAttribute(name) { delete this.attributes[name]; }
    focus() { this.focused = true; }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) elements.set(match[1], new Element());
  const state = { value: initialState || { projects: [], token: "local-token", workspace: "F:\\test" } };
  const doc = { hidden: false, getElementById: id => elements.get(id), createElement: tag => new Element(tag), addEventListener() {} };
  const json = (body, status = 200) => ({ ok: status < 400, status, text: async () => JSON.stringify(body) });
  const presets = [{ id: "check", label: "동작 확인용", originals: 1000, rounds: 1, updates: 10000, sources: [{ name: "<img src=x onerror=bad()>", role: "비호출 자료", url: "javascript:alert(1)", bytes: 1024 }], ...recipeOverrides }, { id: "trial", originals: 20000, rounds: 3, updates: 20000 }];
  const fetch = async (url, options) => {
    calls.push({ url, options });
    if (url === "/api/state") return json(state.value);
    if (url === "/api/recipes") return recipeError ? json({ error: "recipe failure" }, 500) : json({ presets });
    if (url === "/api/projects" && options.method === "POST") {
      const payload = JSON.parse(options.body);
      const project = { id: "bora", wakeword: payload.wakeword, status: "ready", preset: payload.preset, stage: "data" };
      state.value.projects.push(project); state.value.active_project_id = project.id;
      return json(project);
    }
    if (/\/api\/projects\/bora\/(start|resume|pause|input)$/.test(url)) return json({ ok: true });
    if (url.endsWith("/log")) return json({ text: "ready" });
    throw new Error("Unexpected route " + url);
  };
  const context = vm.createContext({ document: doc, fetch, URL, AbortController, console, setTimeout, clearTimeout, setInterval: callback => { intervals.push(callback); return intervals.length; }, clearInterval() {} });
  context.window = context; context.location = new URL("http://127.0.0.1:8800/"); context.KOWW_STUDIO_TEST = noBoot; context.addEventListener = () => {};
  vm.runInContext(script, context);
  return { elements, calls, state, intervals, helpers: context.KowwStudio, click: id => elements.get(id).handlers.click(), submit: id => elements.get(id).handlers.submit({ preventDefault() {} }), tick: async () => { for (const tick of intervals) await tick(); } };
}

(async () => {
  await test("budget request preserves selected values and does not invent overrides", () => {
    const { helpers } = browser({ noBoot: true });
    const result = helpers.requestBody({ wakeword: " 보라야 ", directory: "", advanced: { positive_originals: "", generic_hours: "" } }, "check");
    assert.deepEqual(JSON.parse(JSON.stringify(result)), { wakeword: "보라야", preset: "check" });
    const custom = helpers.requestBody({ wakeword: "보라야", directory: "F:\\bora", advanced: { positive_originals: "500", augmentation_rounds: "3", korean_hours: "1.5" } }, "trial");
    assert.equal(custom.directory, "F:\\bora"); assert.equal(custom.advanced.positive_originals, 500); assert.equal(custom.advanced.korean_hours, 1.5);
    for (const value of ["-1", "1.5", "Infinity"]) assert.throws(() => helpers.requestBody({ wakeword: "보라야", directory: "", advanced: { positive_originals: value } }, "check"));
  });
  await test("unknown progress and ETA stay unknown, and units are formatted separately", () => {
    const { helpers } = browser({ noBoot: true });
    assert.equal(helpers.progressValue({}), null); assert.equal(helpers.progressValue({ progress: { done: 0, total: 0 } }), null);
    assert.equal(helpers.progressValue({ progress: { done: null, total: 100 } }), null);
    assert.equal(helpers.progressValue({ progress: { done: 25, total: 100, unit: "개" } }).percent, 25);
    assert.equal(helpers.eta(null), ""); assert.equal(helpers.eta([3600, 5400]), "1시간–1시간 30분");
    assert.equal(helpers.bytes(1073741824), "1 GiB"); assert.equal(helpers.duration(0), "0초");
  });
  await test("result/review frames only accept same-origin URLs", () => {
    const { helpers } = browser({ noBoot: true });
    assert.equal(helpers.localUrl("/review/?project=bora&queue=pilot"), "/review/?project=bora&queue=pilot");
    assert.equal(helpers.localUrl("https://example.org/tracker"), ""); assert.equal(helpers.localUrl("javascript:alert(1)"), "");
    assert.equal(helpers.webUrl("javascript:alert(1)"), ""); assert.equal(helpers.webUrl("https://example.org/data"), "https://example.org/data");
  });
  await test("initial load only checks state/recipes; source labels stay text and creation enables after recipes", async () => {
    const b = browser(); await flush();
    assert.deepEqual(b.calls.map(call => call.url), ["/api/state", "/api/recipes"]);
    assert.equal(b.elements.get("create").disabled, false);
    const source = b.elements.get("source-preview").children[0].children[1].children[1].children[0];
    assert.equal(source.tagName, "span"); assert.equal(source.textContent, "<img src=x onerror=bad()>");
  });
  await test("training sources and independent evaluation summaries are clearly separated", async () => {
    const b = browser({ recipeOverrides: { sources: [{ id: "generic", role: "generic_features", name: "일반 특징" }, { id: "dns_fan", role: "environment_smoke", split: "train" }], evaluation_sources: [{ dataset_id: "esc50_held_out_v2", name: "다양한 환경 소음", source_url: "https://example.org/noise", file_count: 24, type_count: 12, total_bytes: 1024, license: "CC-BY-NC-3.0", individual_licenses: ["CC0-1.0"] }] } }); await flush();
    const groups = b.elements.get("source-preview").children;
    assert.equal(groups.length, 2); assert.equal(groups[0].children[0].textContent, "학습용 자료");
    assert.equal(groups[1].children[0].textContent, "성능 평가용 배경 음원");
    assert.match(groups[1].children[1].textContent, /호출어와 섞어.*배경만 재생해.*학습에 넣지 않습니다/);
    assert.equal(groups[0].children[2].children[0].textContent, "환경음 증강");
    const evaluation = groups[1].children[2];
    assert.equal(evaluation.children[1].children[0].textContent, "다양한 환경 소음");
    assert.equal(evaluation.children[0].textContent, "시험용 배경 음원");
    assert.equal(evaluation.children[1].children[1].textContent, "종류 12종 · 음원 24개");
    assert.match(evaluation.children[2].textContent, /1 KiB.*CC-BY-NC-3.0/);
    assert.equal(evaluation.children[2].children[0].children[0].textContent, "개별 음원 이용 조건");
    assert.match(groups[1].children.at(-1).textContent, /준비 예정.*실제 사용한 종류/);
    assert.match(groups[1].children.at(-1).textContent, /개발 평가.*Final/);
  });
  await test("evaluation descriptions distinguish Korean field speech from read speech without inventing counts", async () => {
    const b = browser({ recipeOverrides: { sources: [], evaluation_sources: [
      { id: "field", role: "held_out_korean_speech", name: "한국어 현장 대화·안내방송", description: "원출처에서 한국어로 설명된 현장 음원입니다. 검수 후 사용합니다.", file_count: 6, type_count: 5, eval_types: ["korean_market_conversation", "korean_transit_announcement"], usage_description: "호출어 혼합과 배경 단독 시험" },
      { id: "read", role: "held_out_korean_speech", name: "Zeroth 한국어 낭독 · 별도 화자", description: "학습용과 다른 화자를 분리합니다.", eval_types: ["korean_read_speech"] }
    ] } }); await flush();
    const group = b.elements.get("source-preview").children[0], field = group.children[2], read = group.children[3];
    assert.equal(field.children[0].textContent, "겹치는 한국어 음성");
    assert.equal(field.children[1].children[0].textContent, "한국어 현장 대화·안내방송");
    assert.equal(field.children[1].children[1].textContent, "원출처에서 한국어로 설명된 현장 음원입니다. 검수 후 사용합니다.");
    assert.equal(field.children[1].children[2].textContent, "종류 5종 · 음원 6개");
    assert.equal(field.children[1].children[3].textContent, "시장 대화 · 교통 안내방송");
    assert.equal(field.children[1].children[4].textContent, "호출어 혼합과 배경 단독 시험");
    assert.equal(read.children[1].children[2].textContent, "한국어 낭독");
    assert.equal(read.children[2].textContent, "");
    assert.equal(read.children[1].children.some(child => /음원.*개/.test(child.textContent)), false);
  });
  await test("all training budgets advertise one shared full evaluation independently from training amount", async () => {
    const policy = { id: "common_development_v1", validation_references: 10, validation_originals: 5000, snrs: [20, 15, 10, 5, 0], positive_sessions: 80000 };
    const b = browser({ recipeOverrides: { evaluation_policy: policy, evaluation_sources: [{ name: "공통 배경 음원", file_count: 200 }] } }); await flush();
    const group = b.elements.get("source-preview").children[1];
    const notice = group.children.find(child => child.className?.includes("common-evaluation"));
    assert.match(notice.textContent, /세 학습 규모 모두 같은 공통 평가/);
    assert.match(notice.textContent, /5,000개.*20\/15\/10\/5\/0.*80,000개/);
    assert.match(group.children.at(-1).textContent, /학습·공통 평가에서 쓰지 않은.*확정 후/);
    assert.match(group.children.at(-1).textContent, /장시간 오호출 검증/);
  });
  await test("large background-type lists show eight examples and fold the remaining types", async () => {
    const types = ["rain", "wind", "dog", "cat", "coughing", "laughing", "clapping", "snoring", "music_jazz", "music_techno", "music_hiphop"];
    const b = browser({ recipeOverrides: { sources: [], evaluation_sources: [{ name: "공통 평가 자료", eval_types: types }] } }); await flush();
    const info = b.elements.get("source-preview").children[0].children[2].children[1];
    const preview = info.children.find(child => child.className?.includes("source-types"));
    assert.match(preview.textContent, /비 · 바람 · 개 짖음.*외 3종/);
    assert.equal(preview.textContent.includes("재즈"), false);
    const folded = info.children.find(child => child.tagName === "details");
    assert.equal(folded.children[0].textContent, "나머지 3종 보기");
    assert.equal(folded.children[1].textContent, "재즈 · 테크노 · 힙합");
  });
  await test("result view shows measured scope and missing coverage without implying Final approval", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "completed", results: { url: "/results/bora/", scope: "비호출 실측 0.25시간 · 독립 최종 평가 미완료", common_evaluation_assessment: { coverage_targets_passed: false, shortages: ["연속 비호출 평가 0.25/20시간", "평가 음색 계열 2/10개"] } } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.match(b.elements.get("results-note").textContent, /0.25시간.*최종 평가 미완료/);
    assert.equal(b.elements.get("evaluation-coverage").hidden, false);
    assert.match(b.elements.get("evaluation-coverage-summary").textContent, /미충족.*2개/);
    assert.equal(b.elements.get("evaluation-shortages").children.length, 2);
  });
  await test("create-and-start needs one form submission and sends the local token", async () => {
    const b = browser(); await flush(); b.elements.get("wakeword").value = "보라야";
    await b.submit("create-form");
    const mutations = b.calls.filter(call => call.options.method === "POST");
    assert.deepEqual(mutations.map(call => call.url), ["/api/projects", "/api/projects/bora/start"]);
    assert.deepEqual(JSON.parse(mutations[0].options.body), { wakeword: "보라야", preset: "check" });
    assert.ok(mutations.every(call => call.options.headers["X-Koww-Token"] === "local-token"));
    assert.equal(b.elements.get("setup").hidden, true); assert.equal(b.elements.get("project").hidden, false);
  });
  await test("state polling preserves an active review iframe; starting work hides and unloads it", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_review", stage: "pilot", review: { queue_id: "pilot1", url: "/review/?project=bora&queue=pilot1" } };
    const b = browser({ initialState: { projects: [project], active_project_id: "bora" } }); await flush();
    assert.equal(b.elements.get("review-section").hidden, false); assert.equal(b.elements.get("start").hidden, true);
    assert.equal(b.elements.get("review-frame").srcWrites, 1); await b.tick(); await b.tick();
    assert.equal(b.elements.get("review-frame").srcWrites, 1);
    project.status = "running"; await b.tick();
    assert.equal(b.elements.get("review-section").hidden, true); assert.equal(b.elements.get("review-frame").getAttribute("src"), null);
  });
  await test("human input is explicit; polling does not overwrite the text being edited", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_input", stage: "pilot", input: { kind: "hard_phrases", phrases: ["돌아와"] } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("input-value").value, "돌아와"); b.elements.get("input-value").value = "돌아와\n불러줘";
    await b.tick(); assert.equal(b.elements.get("input-value").value, "돌아와\n불러줘");
    await b.submit("input-form"); const request = b.calls.find(call => call.url.endsWith("/input"));
    assert.deepEqual(JSON.parse(request.options.body), { phrases: ["돌아와", "불러줘"] });
  });
  await test("backend's detailed stages map to broad stages without presenting stage count as work percent", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "running", stage: { id: "features", label: "소음 증강" }, stages: [{ id: "frontend", status: "completed" }, { id: "data_download", status: "completed" }, { id: "features", status: "running" }, { id: "training", status: "pending" }], progress: { done: 13, total: 19, unit: "완료 단계" }, elapsed_seconds: 3600, current_stage_elapsed_seconds: 30 };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.match(b.elements.get("stages").children[0].className, /done/);
    assert.match(b.elements.get("stages").children[4].className, /current/);
    assert.equal(b.elements.get("progress").hidden, true); assert.equal(b.elements.get("progress-percent").textContent, "");
    assert.equal(b.elements.get("progress-label").textContent, "전체 절차 13 / 19단계 완료");
    assert.equal(b.elements.get("elapsed").textContent, "이 작업 경과 30초 · 전체 경과 1시간");
    assert.match(b.elements.get("eta").textContent, /측정한 뒤/);
  });
  await test("dependency setup asks for CPU/GPU without mislabeling it as source admission", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_input", stage: "frontend", input: { kind: "dependency_setup", recommended_backend: "cu128" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("input-value").hidden, true); assert.equal(b.elements.get("input-value").required, false);
    assert.equal(b.elements.get("dependency-backend").hidden, false); assert.equal(b.elements.get("dependency-backend").value, "cu128");
    assert.equal(b.elements.get("input-submit").textContent, "선택한 도구 설치하고 계속");
    await b.submit("input-form"); const request = b.calls.find(call => call.url.endsWith("/input"));
    assert.deepEqual(JSON.parse(request.options.body), { dependency_backend: "cu128" });
  });
  await test("diagnostic and generation policy holds never ask for unrelated admission paths", async () => {
    for (const kind of ["diagnostic", "generation_policy", "unknown"]) {
      const project = { id: "bora", wakeword: "보라야", status: "waiting_input", input: { kind, description: "생성 품질을 확인하세요." } };
      const b = browser({ initialState: { projects: [project] } }); await flush();
      assert.equal(b.elements.get("input-section").hidden, false); assert.equal(b.elements.get("input-form").hidden, true);
      assert.equal(b.elements.get("input-description").textContent, "생성 품질을 확인하세요.");
      await b.submit("input-form"); assert.equal(b.calls.some(call => call.url.endsWith("/input")), false);
    }
  });
  await test("additional raw sources are submitted for preparation; existing receipt connection stays optional", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_input", input: { kind: "material_admissions", remaining_hours: 12 } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("source-fields").hidden, false); assert.equal(b.elements.get("input-value").hidden, true);
    assert.equal(b.elements.get("source-hours").value, "12"); b.elements.get("source-directory").value = "F:\\public-data\\korean";
    await b.submit("input-form");
    let request = b.calls.filter(call => call.url.endsWith("/input")).at(-1);
    assert.deepEqual(JSON.parse(request.options.body), { source: { directory: "F:\\public-data\\korean", dataset: "zeroth", pool: "korean_negative", hours: 12 } });
    b.elements.get("material-mode").value = "admissions"; b.elements.get("material-mode").handlers.change();
    assert.equal(b.elements.get("source-directory").required, false); assert.equal(b.elements.get("input-value").required, true);
    b.elements.get("input-value").value = "F:\\admitted\\receipt.json";
    await b.submit("input-form"); request = b.calls.filter(call => call.url.endsWith("/input")).at(-1);
    assert.deepEqual(JSON.parse(request.options.body), { admissions: ["F:\\admitted\\receipt.json"] });
  });
  await test("RIR preparation uses the whole pool without an invented hour budget; changes apply immediately", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_input", input: { kind: "material_admissions" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    b.elements.get("source-directory").value = "F:\\public-data\\rir";
    b.elements.get("source-dataset").value = "mit_rir"; b.elements.get("source-dataset").handlers.change();
    assert.equal(b.elements.get("source-pool").value, "rir");
    assert.equal(b.elements.get("source-hours").required, false); assert.equal(b.elements.get("source-hours-label").hidden, true);
    assert.equal(b.elements.get("source-rir-note").hidden, false);
    b.elements.get("source-hours").value = "";
    await b.submit("input-form"); const request = b.calls.filter(call => call.url.endsWith("/input")).at(-1);
    assert.deepEqual(JSON.parse(request.options.body), { source: { directory: "F:\\public-data\\rir", dataset: "mit_rir", pool: "rir" } });
    b.elements.get("source-pool").value = "noise"; b.elements.get("source-pool").handlers.change();
    assert.equal(b.elements.get("source-hours").required, true); assert.equal(b.elements.get("source-hours-label").hidden, false);
    assert.equal(b.elements.get("source-rir-note").hidden, true);
    b.elements.get("source-dataset").value = "mit_rir"; b.elements.get("source-dataset").handlers.change();
    assert.equal(b.elements.get("source-hours").required, false);
  });
  await test("switching to new setup releases embedded audio and missing recipes never enables creation", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_review", review: { queue_id: "pilot", url: "/review/?project=bora&queue=pilot" } };
    const b = browser({ initialState: { projects: [project] } }); await flush(); await b.click("new-project");
    assert.equal(b.elements.get("setup").hidden, false); assert.equal(b.elements.get("review-frame").getAttribute("src"), null);
    const failed = browser({ recipeError: true }); await flush();
    assert.equal(failed.elements.get("create").disabled, true); assert.match(failed.elements.get("message").textContent, /recipe failure/);
  });
  console.log(`Studio UI: ${tests.length} tests passed`);
  for (const name of tests) console.log("  ✓ " + name);
})().catch(error => { console.error(error); process.exitCode = 1; });
