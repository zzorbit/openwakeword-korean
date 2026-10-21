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

function browser({ noBoot = false, initialState, recipeError = false, recipeOverrides = {}, recipeList } = {}) {
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
  const doc = { hidden: false, getElementById: id => elements.get(id), createElement: tag => new Element(tag), createElementNS: (_namespace, tag) => new Element(tag), addEventListener() {} };
  const json = (body, status = 200) => ({ ok: status < 400, status, text: async () => JSON.stringify(body) });
  const presets = recipeList || [{ id: "check", label: "동작 확인용", originals: 1000, rounds: 1, updates: 10000, sources: [{ name: "<img src=x onerror=bad()>", role: "비호출 자료", url: "javascript:alert(1)", bytes: 1024 }], ...recipeOverrides }, { id: "trial", originals: 20000, rounds: 3, updates: 20000 }];
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

const descendants = node => [node, ...(node.children || []).flatMap(descendants)];
const withClass = (node, name) => descendants(node).filter(item => (item.className || "").split(/\s+/).includes(name));
const textOf = node => [node.textContent || "", ...(node.children || []).map(textOf)].join(" ").trim();
const materialRows = group => withClass(group, "material");
const material = (group, name) => materialRows(group).find(row => withClass(row, "material-name")[0]?.textContent === name);
const sourceEntries = row => withClass(row, "source-item");
function assertMaterialDisclosures(node) {
  const rows = materialRows(node);
  assert.ok(rows.length);
  for (const row of rows) {
    const button = withClass(row, "material-button")[0], panel = withClass(row, "source-panel")[0];
    assert.ok(button); assert.equal(button.tagName, "button"); assert.equal(button.type, "button");
    assert.equal(button.getAttribute("aria-controls"), panel.id);
    assert.equal(button.getAttribute("aria-expanded"), String(!panel.hidden));
    assert.equal(typeof button.handlers.click, "function");
    assert.equal(row.hidden, false);
    assert.equal(panel.getAttribute("role"), "region");
    assert.ok(panel.getAttribute("aria-label"));
    assert.equal(descendants(row).filter(child => child.tagName === "button").length, 1);
    assert.equal(descendants(row).some(child => ["details", "summary"].includes(child.tagName)), false);
  }
  assert.ok(rows.filter(row => !withClass(row, "source-panel")[0].hidden).length <= 1);
}

function commonRecipe(overrides = {}) {
  const catalog = JSON.parse(fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/evaluation_materials_v1.json"), "utf8"));
  const archives = Object.fromEntries(catalog.archives.map(row => [row.id, row]));
  const buckets = { korean_conversation: 10, korean_interview: 5, korean_read: 3, other_speech: 4, music: 3, environment: 4, electronic: 1 };
  const evaluation_sources = Object.entries(buckets).map(([key, target_hours]) => ({
    id: `common_30h_${key}`, dataset_id: key, name: key, role: "held_out_evaluation_material", split: "validation", target_hours,
    source_url: catalog.buckets[key].source_url, description: catalog.buckets[key].description,
    total_bytes: catalog.buckets[key].archives.reduce((sum, id) => sum + archives[id].size_bytes, 0), shared_archive_bytes_not_additive: true
  }));
  evaluation_sources.push({ id: "openslr28_common_rir_v1", name: "공통 평가용 공간 울림", file_count: 80, room_count: 20, simulated_room_count: 20, size_bytes: 1311166223 });
  return {
    id: "check", label: "동작 확인용", originals: 1000, rounds: 1, updates: 10000, hard_originals: 100,
    generic_hours: 2000, korean_hours: 1, train_references: 4,
    automatic_target_hours: { generic_nominal: 2000, korean_train: 1 },
    assets: [
      { id: "generic", role: "generic_features", size_bytes: 17280000128 },
      { id: "zeroth", role: "korean_negative", size_bytes: 10339720618 },
      { id: "audioset", role: "noise_train", size_bytes: 968540160 },
      { id: "audioset_segments", role: "provider_metadata", size_bytes: 1211931 },
      { id: "audioset_ontology", role: "provider_metadata", size_bytes: 342780 },
      { id: "fma", role: "noise_music_train", size_bytes: 7679594875 },
      { id: "fma_metadata", role: "provider_metadata", size_bytes: 358412441 }
    ],
    additional_download: { id: "mit_rir", files: 270, size_bytes: null },
    evaluation_policy: { id: "common_development_v4", validation_references: 30, validation_originals: 5000,
      snrs: [20, 15, 10, 5, 0], positive_sessions: 112000, dry_positive_sessions: 80000,
      reverb_originals: 1000, reverb_positive_sessions: 32000, rir_rooms: 20, negative_hours_by_bucket: buckets },
    evaluation_sources, common_evaluation_catalog: catalog, known_download_bytes: 48123456789,
    ...overrides
  };
}

function commercialRecipe() {
  const component = (id, pool, name, description, target_hours, sourceName, source_url) => ({
    id, pool, name, description, target_hours,
    sources: [{ id, name: sourceName, source_url, availability: "public_source_requires_preparation" }]
  });
  return commonRecipe({
    id: "production", generic_hours: 7000, korean_hours: 3000,
    noise_material_policy: "reuse_negative_waveforms",
    training_negative_plan: {
      id: "commercial_negative_v1", version: 1, generic_hours: 7000, korean_hours: 3000, total_hours: 10000,
      components: [
        component("mixed", "generic_negative", "생활·환경 혼합 음원", "말소리·음악·환경음", 3200, "AudioSet", "https://research.google.com/audioset/"),
        component("multilingual", "generic_negative", "다국어 발화", "여러 언어의 낭독 음성", 2600, "MLS", "https://www.openslr.org/94/"),
        component("web_speech", "generic_negative", "대화·해설 음성", "한국어 외 언어의 웹 음성", 900, "YODAS2", "https://huggingface.co/datasets/espnet/yodas2"),
        component("music", "generic_negative", "음악", "이용조건을 확인한 음원", 300, "FMA", "https://github.com/mdeff/fma"),
        component("korean_web", "korean_negative", "한국어 웹 음성", "대화·해설·강의·낭독", 3000, "YODAS2 한국어", "https://huggingface.co/datasets/espnet/yodas2")
      ]
    }
  });
}

function commercialRecipeV2() {
  const recipe = commercialRecipe();
  recipe.training_negative_plan.id = "commercial_negative_v2";
  recipe.training_negative_plan.version = 2;
  recipe.training_negative_plan.components = recipe.training_negative_plan.components.filter(item => item.id !== "web_speech");
  const [mixed, multilingual, music] = recipe.training_negative_plan.components;
  mixed.target_hours = 3000;
  multilingual.target_hours = 3000;
  music.target_hours = 1000;
  music.name = "음악·보컬·연주곡";
  music.description = "원곡 길이의 음악 음원";
  music.sources[0].id = "fma_full";
  music.sources[0].name = "FMA full";
  return recipe;
}

function commercialRecipeV3() {
  const recipe = commercialRecipeV2();
  recipe.training_negative_plan.id = "commercial_negative_v3";
  recipe.training_negative_plan.version = 3;
  recipe.training_negative_plan.components.find(item => item.id === "music").sources[0].download_mode = "selective";
  return recipe;
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
  await test("initial load checks state/recipes only and untrusted source labels remain text", async () => {
    const b = browser(); await flush();
    assert.deepEqual(b.calls.map(call => call.url), ["/api/state", "/api/recipes"]);
    assert.equal(b.elements.get("create").disabled, false);
    const preview = b.elements.get("source-preview");
    const label = descendants(preview).find(node => node.textContent === "<img src=x onerror=bad()>");
    assert.ok(label); assert.notEqual(label.tagName, "a");
    assert.equal(descendants(preview).some(node => node.tagName === "img"), false);
  });
  await test("materials use the approved list hierarchy with short consistent titles", async () => {
    const b = browser({ recipeList: [commonRecipe()] }); await flush();
    const preview = b.elements.get("source-preview"), groups = preview.children;
    assert.equal(groups.length, 2);
    assert.equal(descendants(groups[0]).find(node => node.tagName === "h3").textContent, "학습용 자료");
    assert.equal(descendants(groups[1]).find(node => node.tagName === "h3").textContent, "성능 평가용 자료");
    assert.deepEqual(materialRows(groups[0]).map(row => withClass(row, "material-name")[0].textContent),
      ["호출어 음성", "유사어 음성", "일반 비호출 자료", "한국어 비호출 자료", "소음 증강 음원", "잔향 자료"]);
    assert.deepEqual(materialRows(groups[1]).map(row => withClass(row, "material-name")[0].textContent),
      ["호출어 음성", "일반 비호출 자료", "한국어 비호출 자료", "소음 증강 음원", "잔향 자료"]);
    for (const row of materialRows(preview)) {
      assert.ok(withClass(row, "material-description")[0].textContent.trim());
      assert.ok(withClass(row, "material-amount")[0].textContent.trim());
      assert.equal(withClass(row, "source-panel")[0].hidden, true);
      for (const source of sourceEntries(row)) {
        assert.ok(withClass(source, "source-name")[0].textContent.trim());
        assert.ok(withClass(source, "source-content")[0].textContent.trim());
      }
    }
    const all = textOf(preview);
    for (const removed of ["각 항목을 눌러", "준비 예정 내역", "유형별 시간은 제공자가 공개하지 않음", "원본 음원은 포함하지 않음", "추가 준비량 없음", "합성 음색 4계열, WAV 크기는 생성 후 확인"]) assert.equal(all.includes(removed), false, removed);
    assert.equal(descendants(preview).some(node => ["table", "select"].includes(node.tagName)), false);
  });
  await test("one accessible material disclosure stays open across both groups and rerenders", async () => {
    const b = browser({ recipeList: [commonRecipe()] }); await flush();
    const preview = b.elements.get("source-preview"), rows = materialRows(preview);
    assert.equal(rows.length, 11);
    assertMaterialDisclosures(preview);
    const button = row => withClass(row, "material-button")[0], panel = row => withClass(row, "source-panel")[0];
    assert.equal(rows.filter(row => !panel(row).hidden).length, 0);
    for (const row of rows) {
      button(row).handlers.click();
      assert.equal(button(row).getAttribute("aria-expanded"), "true");
      assert.equal(panel(row).hidden, false);
      assert.equal(rows.filter(item => !panel(item).hidden).length, 1);
      assertMaterialDisclosures(preview);
    }
    await b.tick();
    assertMaterialDisclosures(b.elements.get("source-preview"));
    b.elements.get("adv-originals").value = "1234";
    b.elements.get("adv-originals").handlers.input();
    const rerendered = b.elements.get("source-preview");
    assertMaterialDisclosures(rerendered);
    const open = materialRows(rerendered).filter(row => !panel(row).hidden);
    assert.equal(open.length, 1);
    assert.equal(panel(open[0]).id, panel(rows.at(-1)).id);
    button(open[0]).handlers.click();
    assert.equal(materialRows(rerendered).filter(row => !panel(row).hidden).length, 0);
  });
  await test("selected scale and advanced targets update counts without claiming automatic supply", async () => {
    const specs = [
      ["check", 1000, 100, 2000, 1],
      ["trial", 20000, 1000, 2000, 2000 / 3],
      ["production", 100000, 2000, 7500, 2500]
    ];
    const recipes = specs.map(([id, originals, hard_originals, generic_hours, korean_hours]) => commonRecipe({ id, originals, hard_originals, generic_hours, korean_hours }));
    const b = browser({ recipeList: recipes }); await flush();
    const amount = (group, name) => withClass(material(group, name), "material-amount")[0].textContent;
    for (let index = 0; index < specs.length; index++) {
      const radio = descendants(b.elements.get("presets").children[index]).find(node => node.tagName === "input");
      radio.handlers.change();
      const groups = b.elements.get("source-preview").children;
      assertMaterialDisclosures(b.elements.get("source-preview"));
      assert.equal(amount(groups[0], "호출어 음성"), specs[index][1].toLocaleString("ko-KR") + "개");
      assert.equal(amount(groups[0], "유사어 음성"), specs[index][2].toLocaleString("ko-KR") + "개");
      assert.equal(amount(groups[0], "일반 비호출 자료"), specs[index][3].toLocaleString("ko-KR") + "시간");
      assert.equal(amount(groups[0], "한국어 비호출 자료"), specs[index][4].toLocaleString("ko-KR", { maximumFractionDigits: 1 }) + "시간");
      assert.equal(amount(groups[1], "호출어 음성"), "5,000개");
      assert.equal(amount(groups[1], "일반 비호출 자료"), "12시간");
      assert.equal(amount(groups[1], "한국어 비호출 자료"), "18시간");
      const genericSources = textOf(withClass(material(groups[0], "일반 비호출 자료"), "source-panel")[0]);
      const koreanSources = textOf(withClass(material(groups[0], "한국어 비호출 자료"), "source-panel")[0]);
      assert.match(genericSources, /2,000시간/);
      assert.match(koreanSources, /1시간/);
      if (index === 2) {
        assert.equal(genericSources.includes("7,500시간"), false);
        assert.equal(koreanSources.includes("2,500시간"), false);
        assert.match(genericSources, /추가 자료.*5,500시간/);
        assert.match(koreanSources, /추가 자료.*2,499시간/);
      }
    }
    b.elements.get("adv-originals").value = "4321";
    b.elements.get("adv-originals").handlers.input();
    b.elements.get("adv-generic").value = "3210";
    b.elements.get("adv-generic").handlers.input();
    b.elements.get("adv-korean").value = "12";
    b.elements.get("adv-korean").handlers.input();
    let training = b.elements.get("source-preview").children[0];
    assertMaterialDisclosures(b.elements.get("source-preview"));
    assert.equal(amount(training, "호출어 음성"), "4,321개");
    assert.equal(amount(training, "일반 비호출 자료"), "3,210시간");
    assert.equal(amount(training, "한국어 비호출 자료"), "12시간");
    for (const id of ["adv-originals", "adv-generic", "adv-korean"]) { b.elements.get(id).value = ""; b.elements.get(id).handlers.input(); }
    training = b.elements.get("source-preview").children[0];
    assertMaterialDisclosures(b.elements.get("source-preview"));
    assert.equal(amount(training, "호출어 음성"), "100,000개");
    assert.equal(amount(training, "일반 비호출 자료"), "7,500시간");
  });
  await test("commercial raw-source plan shows separate preparation targets without adding legacy features", async () => {
    const b = browser({ recipeList: [commercialRecipe()] }); await flush();
    const preview = b.elements.get("source-preview"), training = preview.children[0];
    assertMaterialDisclosures(preview);
    const generic = material(training, "일반 비호출 자료"), korean = material(training, "한국어 비호출 자료");
    assert.equal(withClass(generic, "material-amount")[0].textContent, "7,000시간");
    assert.equal(withClass(korean, "material-amount")[0].textContent, "3,000시간");
    assert.deepEqual(sourceEntries(generic).map(row => withClass(row, "source-name")[0].textContent), ["AudioSet", "MLS", "YODAS2", "FMA"]);
    assert.deepEqual(sourceEntries(generic).map(row => withClass(row, "source-quantity")[0].textContent), ["준비 목표 3,200시간", "준비 목표 2,600시간", "준비 목표 900시간", "준비 목표 300시간"]);
    assert.deepEqual(sourceEntries(korean).map(row => withClass(row, "source-name")[0].textContent), ["YODAS2 한국어"]);
    assert.equal(withClass(sourceEntries(korean)[0], "source-quantity")[0].textContent, "준비 목표 3,000시간");
    for (const row of [...sourceEntries(generic), ...sourceEntries(korean)]) {
      const name = withClass(row, "source-name")[0];
      assert.equal(name.tagName, "a"); assert.match(name.href, /^https:\/\//);
      assert.equal(withClass(row, "file-size").length, 0);
    }
    for (const excluded of ["openWakeWord", "Zeroth", "2,000시간", "추가 자료", "준비 완료"]) assert.equal(textOf(generic).includes(excluded) || textOf(korean).includes(excluded), false, excluded);
    assert.match(textOf(korean), /대화·해설·강의·낭독/);
    assert.equal(withClass(material(training, "소음 증강 음원"), "material-amount")[0].textContent, "재사용");
    assert.equal(withClass(material(preview.children[1], "일반 비호출 자료"), "material-amount")[0].textContent, "12시간");
  });
  await test("commercial component targets follow custom pool hours while keeping the same source plan", async () => {
    const b = browser({ recipeList: [commercialRecipe()] }); await flush();
    b.elements.get("adv-generic").value = "3500"; b.elements.get("adv-generic").handlers.input();
    b.elements.get("adv-korean").value = "1500"; b.elements.get("adv-korean").handlers.input();
    const training = b.elements.get("source-preview").children[0];
    const generic = material(training, "일반 비호출 자료"), korean = material(training, "한국어 비호출 자료");
    assert.equal(withClass(generic, "material-amount")[0].textContent, "3,500시간");
    assert.deepEqual(sourceEntries(generic).map(row => withClass(row, "source-quantity")[0].textContent), ["준비 목표 1,600시간", "준비 목표 1,300시간", "준비 목표 450시간", "준비 목표 150시간"]);
    assert.equal(withClass(sourceEntries(korean)[0], "source-quantity")[0].textContent, "준비 목표 1,500시간");
    for (const id of ["adv-generic", "adv-korean"]) { b.elements.get(id).value = ""; b.elements.get(id).handlers.input(); }
    const restored = b.elements.get("source-preview").children[0];
    assert.equal(withClass(material(restored, "일반 비호출 자료"), "material-amount")[0].textContent, "7,000시간");
    assert.equal(withClass(material(restored, "한국어 비호출 자료"), "material-amount")[0].textContent, "3,000시간");
  });
  await test("commercial v2 shows AudioSet, MLS and full-length FMA without foreign web speech or legacy feature hours", async () => {
    const b = browser({ recipeList: [commercialRecipeV2()] }); await flush();
    const preview = b.elements.get("source-preview"), training = preview.children[0];
    assertMaterialDisclosures(preview);
    const generic = material(training, "일반 비호출 자료"), korean = material(training, "한국어 비호출 자료");
    assert.equal(withClass(generic, "material-amount")[0].textContent, "7,000시간");
    assert.equal(withClass(korean, "material-amount")[0].textContent, "3,000시간");
    assert.deepEqual(sourceEntries(generic).map(row => withClass(row, "source-name")[0].textContent), ["AudioSet", "MLS", "FMA full"]);
    assert.deepEqual(sourceEntries(generic).map(row => withClass(row, "source-quantity")[0].textContent), ["준비 목표 3,000시간", "준비 목표 3,000시간", "준비 목표 1,000시간"]);
    assert.deepEqual(sourceEntries(korean).map(row => withClass(row, "source-name")[0].textContent), ["YODAS2 한국어"]);
    assert.match(textOf(sourceEntries(generic)[2]), /원곡 길이의 음악 음원/);
    for (const excluded of ["YODAS2", "900시간", "300시간", "30초", "openWakeWord", "준비 완료"]) assert.equal(textOf(generic).includes(excluded), false, excluded);
    assert.equal(withClass(material(training, "소음 증강 음원"), "material-amount")[0].textContent, "재사용");
    assert.equal(withClass(material(preview.children[1], "소음 증강 음원"), "material-amount")[0].textContent, "30시간");
  });
  await test("commercial v2 scales its three Generic sources independently of the Korean pool", async () => {
    const b = browser({ recipeList: [commercialRecipeV2()] }); await flush();
    b.elements.get("adv-generic").value = "3500"; b.elements.get("adv-generic").handlers.input();
    b.elements.get("adv-korean").value = "1500"; b.elements.get("adv-korean").handlers.input();
    let training = b.elements.get("source-preview").children[0];
    assert.deepEqual(sourceEntries(material(training, "일반 비호출 자료")).map(row => withClass(row, "source-quantity")[0].textContent), ["준비 목표 1,500시간", "준비 목표 1,500시간", "준비 목표 500시간"]);
    assert.equal(withClass(sourceEntries(material(training, "한국어 비호출 자료"))[0], "source-quantity")[0].textContent, "준비 목표 1,500시간");
    for (const id of ["adv-generic", "adv-korean"]) { b.elements.get(id).value = ""; b.elements.get(id).handlers.input(); }
    training = b.elements.get("source-preview").children[0];
    assert.deepEqual(sourceEntries(material(training, "일반 비호출 자료")).map(row => withClass(row, "source-quantity")[0].textContent), ["준비 목표 3,000시간", "준비 목표 3,000시간", "준비 목표 1,000시간"]);
    assert.equal(withClass(material(training, "한국어 비호출 자료"), "material-amount")[0].textContent, "3,000시간");
  });
  await test("commercial v3 describes selective original MP3 downloads without changing the data budget", async () => {
    const b = browser({ recipeList: [commercialRecipeV3()] }); await flush();
    const preview = b.elements.get("source-preview"), training = preview.children[0];
    assertMaterialDisclosures(preview);
    const generic = material(training, "일반 비호출 자료"), korean = material(training, "한국어 비호출 자료");
    assert.deepEqual(sourceEntries(generic).map(row => withClass(row, "source-name")[0].textContent), ["AudioSet", "MLS", "FMA full"]);
    assert.deepEqual(sourceEntries(generic).map(row => withClass(row, "source-quantity")[0].textContent), ["준비 목표 3,000시간", "준비 목표 3,000시간", "준비 목표 1,000시간"]);
    assert.match(textOf(sourceEntries(generic)[2]), /필요한 원곡만 선택 다운로드.*원본 MP3 보존/);
    for (const text of ["879", "900 GiB", "준비 완료", "원곡 길이의 음악 음원"]) assert.equal(textOf(sourceEntries(generic)[2]).includes(text), false, text);
    assert.equal(withClass(korean, "material-amount")[0].textContent, "3,000시간");
    assert.equal(withClass(material(training, "소음 증강 음원"), "material-amount")[0].textContent, "재사용");
  });
  await test("commercial v3 keeps selective download wording while custom hours are changed and restored", async () => {
    const b = browser({ recipeList: [commercialRecipeV3()] }); await flush();
    b.elements.get("adv-generic").value = "3500"; b.elements.get("adv-generic").handlers.input();
    const music = sourceEntries(material(b.elements.get("source-preview").children[0], "일반 비호출 자료"))[2];
    assert.equal(withClass(music, "source-quantity")[0].textContent, "준비 목표 500시간");
    assert.match(textOf(music), /선택 다운로드.*원본 MP3 보존/);
    b.elements.get("adv-generic").value = ""; b.elements.get("adv-generic").handlers.input();
    const restored = sourceEntries(material(b.elements.get("source-preview").children[0], "일반 비호출 자료"))[2];
    assert.equal(withClass(restored, "source-quantity")[0].textContent, "준비 목표 1,000시간");
    assert.match(textOf(restored), /선택 다운로드.*원본 MP3 보존/);
  });
  await test("new trial projects show automatically selected Korean sources without a manual shortage or legacy Zeroth", async () => {
    const recipe = commonRecipe({ id: "trial", korean_hours: 2000 / 3, material_acquisition_policy: "automatic_v1",
      automatic_source_plan: { version: 1, enabled: true, components: [{ id: "korean", provider: "yodas2", pool: "korean_negative", target_hours: 2000 / 3 }] } });
    recipe.assets = recipe.assets.filter(item => item.id !== "zeroth");
    const b = browser({ recipeList: [recipe] }); await flush();
    const training = b.elements.get("source-preview").children[0], korean = material(training, "한국어 비호출 자료");
    assert.equal(withClass(korean, "material-description")[0].textContent, "한국어 대화 · 낭독 등 음성");
    assert.equal(withClass(korean, "material-amount")[0].textContent, "666.7시간");
    assert.deepEqual(sourceEntries(korean).map(row => withClass(row, "source-name")[0].textContent), ["YODAS2 한국어"]);
    assert.match(textOf(korean), /목표량 자동 선정/);
    assert.equal(textOf(training).includes("직접 준비할"), false);
    assert.equal(textOf(training).includes("Zeroth"), false);
    assert.match(textOf(material(training, "소음 증강 음원")), /학습용 한국어 원본 음원 재사용/);
    assert.match(textOf(material(training, "일반 비호출 자료")), /2,000시간/);
    assert.equal(b.elements.get("source-plan-note").hidden, false);
    assert.match(b.elements.get("source-plan-note").textContent, /추가 다운로드 용량은 목록 확정 후/);
    b.elements.get("adv-korean").value = "100"; b.elements.get("adv-korean").handlers.input();
    assert.equal(withClass(sourceEntries(material(b.elements.get("source-preview").children[0], "한국어 비호출 자료"))[0], "source-quantity")[0].textContent, "준비 목표 100시간");
  });
  await test("automatic production omits unused legacy downloads and never presents a planned corpus as admitted", async () => {
    const recipe = commercialRecipeV3();
    recipe.material_acquisition_policy = "automatic_v1";
    recipe.automatic_source_plan = { version: 1, enabled: true, components: recipe.training_negative_plan.components.map(item => ({ ...item, provider: item.sources[0].id })) };
    recipe.assets = recipe.assets.filter(item => !["generic", "zeroth", "audioset", "fma", "fma_metadata", "audioset_segments", "audioset_ontology"].includes(item.id));
    const b = browser({ recipeList: [recipe] }); await flush();
    const training = b.elements.get("source-preview").children[0];
    assert.match(textOf(training), /AudioSet.*MLS.*FMA full/);
    assert.equal(textOf(training).includes("openWakeWord"), false);
    assert.equal(textOf(training).includes("Zeroth"), false);
    assert.equal(textOf(training).includes("준비 완료"), false);
    assert.equal(textOf(training).includes("검수 승인"), false);
    assert.equal(b.calls.some(call => /download|prepare/.test(call.url)), false);
  });
  await test("historical bounded recipes retain explicit manual shortages and no automatic-acquisition claim", async () => {
    const b = browser({ recipeList: [commonRecipe({ id: "trial", korean_hours: 2000 / 3 })] }); await flush();
    assert.match(textOf(material(b.elements.get("source-preview").children[0], "한국어 비호출 자료")), /직접 준비할/);
    assert.equal(b.elements.get("source-plan-note").hidden, true);
  });
  await test("minimum storage uses known payloads and updates with custom N, rounds and source hours", async () => {
    const recipe = commercialRecipeV3();
    Object.assign(recipe, { originals: 100000, rounds: 5, material_acquisition_policy: "automatic_v1",
      automatic_source_plan: { version: 1, enabled: true, components: [{ pool: "generic_negative" }, { pool: "korean_negative" }] },
      automatic_space_plan: { base_archive_bytes: 1000, base_working_bytes: 2000, future_buffer_bytes: 3000, estimated_original_archive_bytes: 99 * 1024 ** 4 } });
    const before = JSON.stringify(recipe);
    const b = browser({ recipeList: [recipe], initialState: { projects: [], system: { free_bytes: 1000 } } }); await flush();
    const expected = (generic, korean, originals, rounds) => 6000 + Math.ceil((generic + korean) * 3600 * 32000)
      + Math.ceil((generic + korean) * 3600 / 2) * 6144 + Math.ceil(originals * rounds * 6144);
    assert.equal(b.helpers.minimumStorage(recipe), expected(7000, 3000, 100000, 5));
    assert.equal(b.helpers.bytes(1024 ** 4), "1 TiB");
    assert.match(b.elements.get("source-plan-note").textContent, /최소 저장 공간 .*TiB \+ 원본·압축파일 추가/);
    assert.match(b.elements.get("source-plan-note").textContent, /현재 여유 공간보다 큽니다/);
    b.elements.get("adv-generic").value = "3500"; b.elements.get("adv-generic").handlers.input();
    b.elements.get("adv-korean").value = "1500"; b.elements.get("adv-korean").handlers.input();
    b.elements.get("adv-originals").value = "1000"; b.elements.get("adv-originals").handlers.input();
    b.elements.get("adv-rounds").value = "1"; b.elements.get("adv-rounds").handlers.input();
    assert.match(b.elements.get("source-plan-note").textContent, new RegExp(b.helpers.bytes(expected(3500, 1500, 1000, 1)).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
    assert.equal(JSON.stringify(recipe), before);
    assert.equal(b.helpers.minimumStorage(commonRecipe()), null);
  });
  await test("compact storage shows retained-source estimate and one temporary archive budget with custom scaling", async () => {
    const recipe = commercialRecipeV3();
    Object.assign(recipe, { originals: 100000, rounds: 5, material_acquisition_policy: "automatic_v2",
      automatic_source_plan: { version: 2, enabled: true, components: recipe.training_negative_plan.components.map(item => ({ ...item,
        provider: { mixed: "audioset_unbalanced", multilingual: "mls", korean_web: "yodas2" }[item.id] || item.sources[0].id })) },
      automatic_space_plan: { base_archive_bytes: 1000, base_working_bytes: 2000, future_buffer_bytes: 3000,
        transient_archive_bytes: 8 * 1024 ** 3, estimated_original_archive_bytes: 99 * 1024 ** 4 } });
    const original = JSON.stringify(recipe);
    const b = browser({ recipeList: [recipe], initialState: { projects: [], system: { free_bytes: 2 * 1024 ** 4 } } }); await flush();
    const estimate = b.helpers.storageEstimate(recipe);
    assert.equal(estimate.retained, 2585000000000);
    assert.equal(estimate.minimum, 6000 + 8 * 1024 ** 3 + 1152000000000 + 110592000000 + 3072000000);
    assert.equal(estimate.total, estimate.minimum + 2585000000000);
    assert.match(b.elements.get("source-plan-note").textContent, /예상 필요 공간 .*묶음 임시 8 GiB 예산 포함/);
    assert.match(b.elements.get("source-plan-note").textContent, /원본 제외 최소/);
    const small = { genericHours: 3500, koreanHours: 1500, originals: 1000, rounds: 1 };
    assert.equal(b.helpers.storageEstimate(recipe, small).retained, 1292500000000);
    for (const [id, value] of [["adv-generic", "3500"], ["adv-korean", "1500"], ["adv-originals", "1000"], ["adv-rounds", "1"]]) {
      b.elements.get(id).value = value; b.elements.get(id).handlers.input();
    }
    assert.ok(b.elements.get("source-plan-note").textContent.includes(b.helpers.bytes(b.helpers.storageEstimate(recipe, small).total)));
    assert.equal(JSON.stringify(recipe), original);
    assert.equal(b.helpers.storageEstimate({ ...recipe, material_acquisition_policy: "automatic_v1" }), null);
    assert.equal(b.helpers.storageEstimate({ ...recipe, automatic_space_plan: { ...recipe.automatic_space_plan, transient_archive_bytes: undefined } }), null);
    assert.equal(b.helpers.storageEstimate({ ...recipe, automatic_source_plan: { components: [{ provider: "unknown", pool: "generic_negative", target_hours: 1 }] } }), null);
  });
  await test("compact trial still selects YODAS automatically and does not change the legacy source display", async () => {
    const recipe = commonRecipe({ id: "trial", korean_hours: 2000 / 3, material_acquisition_policy: "automatic_v2",
      automatic_source_plan: { version: 2, enabled: true, components: [{ id: "yodas2", provider: "yodas2", pool: "korean_negative", target_hours: 2000 / 3 }] } });
    recipe.assets = recipe.assets.filter(item => item.id !== "zeroth");
    const b = browser({ recipeList: [recipe] }); await flush();
    const training = b.elements.get("source-preview").children[0];
    assert.match(textOf(material(training, "한국어 비호출 자료")), /YODAS2 한국어.*목표량 자동 선정/);
    assert.equal(textOf(training).includes("직접 준비할"), false);
    assert.equal(textOf(training).includes("Zeroth"), false);
    assert.equal(b.elements.get("source-plan-note").hidden, false);
  });
  await test("compact check omits empty temporary and duplicate minimum amounts", async () => {
    const recipe = commonRecipe({ id: "check", originals: 1000, rounds: 1, generic_hours: 2000, korean_hours: 1,
      material_acquisition_policy: "automatic_v2",
      automatic_source_plan: { version: 2, enabled: true, components: [] },
      automatic_space_plan: { base_archive_bytes: 57880940221, base_working_bytes: 16106127360,
        future_buffer_bytes: 8589934592, transient_archive_bytes: 0 } });
    const b = browser({ recipeList: [recipe] }); await flush();
    const note = b.elements.get("source-plan-note").textContent;
    assert.match(note, /예상 필요 공간 76\.9 GiB/);
    assert.equal(note.includes("묶음 임시"), false);
    assert.equal(note.includes("원본 제외 최소"), false);
    assert.match(note, /실제 용량은 선정 후 확인합니다/);
  });
  await test("training source coverage separates targets, measured candidates, admission and nominal legacy features", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "running", training_materials: {
      targets: { generic_negative: 7000, korean_negative: 3000 }, candidate: { generic_negative: 24.5, korean_negative: 0 }, admitted: null, generic_publisher_nominal_hours: null
    }, progress: { done: 2.5, total: 10, unit: "audio_hours", label: "한국어 원본 검사" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("training-material-coverage").hidden, false);
    assert.match(textOf(b.elements.get("training-material-coverage-list")), /기술 검사 24.5시간 · 검수 승인 확인 전 \/ 목표 7,000시간/);
    assert.match(textOf(b.elements.get("training-material-coverage-list")), /기술 검사 0시간 · 검수 승인 확인 전 \/ 목표 3,000시간/);
    assert.match(b.elements.get("progress-label").textContent, /2.5 \/ 10시간/);
    assert.match(b.elements.get("eta").textContent, /처리 속도를 측정한 뒤/);
    project.training_materials = { targets: { generic_negative: 2000, korean_negative: 1 }, candidate: { generic_negative: 0, korean_negative: 1.02 }, admitted: { generic_negative: 0, korean_negative: 1.01 }, generic_publisher_nominal_hours: 2000 };
    await b.tick();
    assert.match(textOf(b.elements.get("training-material-coverage-list")), /공개 특징 명목 2,000시간 \/ 목표 2,000시간/);
    assert.match(textOf(b.elements.get("training-material-coverage-list")), /기술 검사 1.02시간 · 검수 승인 1.01시간/);
    delete project.training_materials; await b.tick();
    assert.equal(b.elements.get("training-material-coverage").hidden, true);
  });
  await test("common evaluation groups each real source once and keeps archive bytes with quantity", async () => {
    const b = browser({ recipeList: [commonRecipe()] }); await flush();
    const group = b.elements.get("source-preview").children[1];
    const generic = material(group, "일반 비호출 자료"), korean = material(group, "한국어 비호출 자료"), noise = material(group, "소음 증강 음원");
    assert.equal(sourceEntries(generic).length, 2);
    assert.equal(sourceEntries(korean).length, 2);
    assert.equal(sourceEntries(noise).length, 4);
    const names = sourceEntries(noise).map(row => withClass(row, "source-name")[0].textContent);
    assert.equal(new Set(names).size, 4);
    const quantities = sourceEntries(noise).map(row => withClass(row, "source-quantity")[0]);
    assert.deepEqual(quantities.map(node => node.textContent), ["15시간", "7시간", "5시간", "3시간"]);
    for (const quantity of quantities) assert.equal(withClass(quantity, "file-size").length, 1);
    assert.match(textOf(korean), /대화 10시간.*인터뷰 5시간|인터뷰 5시간.*대화 10시간/);
    assert.match(textOf(korean), /낭독 3시간/);
    assert.equal(withClass(korean, "file-size").filter(node => node.textContent.includes("2.59 GB")).length, 1);
    assert.equal(textOf(korean).includes("5.18 GB"), false);
    assert.match(textOf(noise), /30시간/);
  });
  await test("training augmentation uses real source budgets and does not invent missing durations", async () => {
    const b = browser({ recipeList: [commonRecipe()] }); await flush();
    const group = b.elements.get("source-preview").children[0], rows = sourceEntries(material(group, "소음 증강 음원"));
    assert.equal(rows.length, 3);
    const audio = rows.find(row => /AudioSet/.test(withClass(row, "source-name")[0].textContent));
    assert.ok(audio); assert.match(withClass(audio, "source-quantity")[0].textContent, /묶음/);
    assert.equal(/시간|분|초/.test(withClass(audio, "source-quantity")[0].textContent), false);
    assert.match(withClass(audio, "file-size")[0].textContent, /970\.1 MB/);
    const music = rows.find(row => /FMA/.test(withClass(row, "source-name")[0].textContent));
    assert.match(textOf(music), /1시간/); assert.match(textOf(music), /120/);
    assert.match(withClass(music, "file-size")[0].textContent, /8\.04 GB/);
    assert.equal(/DNS|팬 작동 소리|팬 소음/.test(textOf(group)), false);
    assert.equal(withClass(material(group, "소음 증강 음원"), "material-amount")[0].textContent, "3종");
    const rir = material(group, "잔향 자료"); assert.match(textOf(rir), /270개/);
    assert.equal(withClass(rir, "file-size").length, 0);
  });
  await test("commercial augmentation reuses Negative waveforms without new archives or invented hours", async () => {
    const recipes = [
      commonRecipe(),
      commonRecipe({ id: "full", generic_hours: 7500, korean_hours: 2500, noise_material_policy: "reuse_negative_waveforms" }),
      commonRecipe({ id: "trial", originals: 20000, rounds: 3 }),
      commonRecipe({ id: "production", generic_hours: 7500, korean_hours: 2500 })
    ];
    const b = browser({ recipeList: recipes }); await flush();
    const select = index => descendants(b.elements.get("presets").children[index]).find(node => node.tagName === "input").handlers.change();
    for (const index of [0, 1, 2, 1, 3]) {
      select(index);
      const preview = b.elements.get("source-preview"), group = preview.children[0], noise = material(group, "소음 증강 음원");
      assertMaterialDisclosures(preview);
      assert.equal(materialRows(group).length, 6);
      if (index === 1) {
        assert.equal(withClass(noise, "material-amount")[0].textContent, "재사용");
        assert.match(withClass(noise, "material-description")[0].textContent, /비호출 원본 음원 재사용/);
        const rows = sourceEntries(noise);
        assert.deepEqual(rows.map(row => withClass(row, "source-name")[0].textContent), ["일반 비호출 자료", "한국어 비호출 자료"]);
        assert.deepEqual(rows.map(row => withClass(row, "source-quantity")[0].textContent), ["원본 재사용", "원본 재사용"]);
        assert.equal(withClass(noise, "file-size").length, 0);
        assert.equal(/AudioSet|FMA|시간|GB|MB/.test(textOf(noise)), false);
        assert.equal(textOf(group).includes("provider_metadata"), false);
      } else {
        assert.equal(withClass(noise, "material-amount")[0].textContent, "3종");
        assert.equal(sourceEntries(noise).length, 3);
        assert.match(textOf(noise), /AudioSet/);
        assert.match(textOf(noise), /FMA/);
      }
      assert.equal(withClass(material(preview.children[1], "소음 증강 음원"), "material-amount")[0].textContent, "30시간");
    }
  });
  await test("retired fan test clips stay hidden in legacy preview payloads", async () => {
    for (const collection of ["assets", "sources"]) {
      const recipe = commonRecipe();
      recipe[collection] = [...recipe.assets, { id: "dns_fan", name: "DNS-Challenge", description: "팬 작동 소리", role: "environment_smoke", split: "train", size_bytes: 320044 }];
      const b = browser({ recipeList: [recipe] }); await flush();
      const group = b.elements.get("source-preview").children[0];
      assert.equal(withClass(material(group, "소음 증강 음원"), "material-amount")[0].textContent, "3종");
      assert.equal(/dns_fan|DNS|팬 작동 소리|팬 소음/.test(textOf(group)), false);
      assert.equal(sourceEntries(material(group, "소음 증강 음원")).some(row => /AudioSet/.test(textOf(row))), true);
    }
  });
  await test("versioned policy summaries retain historical evaluation totals and available sources", async () => {
    for (const [version, total] of [[1, 80000], [2, 100000], [3, 112000], [4, 112000]]) {
      const recipe = commonRecipe();
      recipe.evaluation_policy.id = `common_development_v${version}`;
      recipe.evaluation_policy.positive_sessions = total;
      recipe.evaluation_policy.reverb_originals = version === 2 ? 625 : 1000;
      recipe.evaluation_sources.push({ id: "old_extra", name: "기존 독립 배경", role: "held_out_evaluation_noise", file_count: 193 });
      const b = browser({ recipeList: [recipe] }); await flush();
      const group = b.elements.get("source-preview").children[1], all = textOf(group);
      assert.match(all, new RegExp(total.toLocaleString("ko-KR") + "회"));
      if (version < 4) assert.match(all, /기존 독립 배경/);
      else assert.equal(all.includes("기존 독립 배경"), false);
      if (total !== 112000) assert.equal(all.includes("112,000회"), false);
    }
  });
  await test("partial and unknown metadata stays explicit without invented time, counts or bytes", async () => {
    const b = browser({ recipeOverrides: { originals: undefined, hard_originals: undefined, sources: [], evaluation_sources: [
      { id: "custom", name: "직접 연결한 자료", role: "held_out_evaluation_noise", description: "다른 조건의 독립 자료", source_url: "javascript:alert(1)" }
    ] } }); await flush();
    const preview = b.elements.get("source-preview"), all = textOf(preview);
    assert.match(all, /직접 연결한 자료/);
    assert.equal(/NaN|undefined|Infinity/.test(all), false);
    const row = withClass(preview, "source-item").find(item => /직접 연결한 자료/.test(textOf(item)));
    assert.ok(row); assert.equal(withClass(row, "file-size").length, 0);
    assert.equal(/0시간|0개|0 B/.test(textOf(row)), false);
    assert.equal(descendants(row).some(node => node.tagName === "a"), false);
  });
  await test("result view shows measured scope and missing coverage without implying Final approval", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "completed", results: { url: "/results/bora/", scope: "비호출 실측 0.25시간 · 독립 최종 평가 미완료", common_evaluation_assessment: { coverage_targets_passed: false, shortages: ["연속 비호출 평가 0.25/20시간", "평가 음색 계열 2/10개"] } } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.match(b.elements.get("results-note").textContent, /0.25시간.*최종 평가 미완료/);
    assert.equal(b.elements.get("evaluation-coverage").hidden, false);
    assert.match(b.elements.get("evaluation-coverage-summary").textContent, /미충족.*2개/);
    assert.equal(b.elements.get("evaluation-shortages").children.length, 2);
  });
  await test("v4 material progress separates core quotas, startup margin and eligible time without inventing admission", async () => {
    const project = {id:"bora",wakeword:"보라야",status:"waiting_review",stage:"data_download",evaluation_materials:{
      candidate:{target_hours_by_bucket:{korean_read:3},actual_hours_by_bucket:{korean_read:3.08},core_coverage:{actual_hours_by_bucket:{korean_read:3}}},
      admitted:{target_hours_by_bucket:{korean_read:3},actual_hours_by_bucket:{korean_read:3.08},core_coverage:{actual_hours_by_bucket:{korean_read:3}}},
      duration:{candidate:{core_hours:30,initialization_margin_hours:.08,eligible_hours:30.01},admitted:{core_hours:30,initialization_margin_hours:.08,eligible_hours:30.01}},scope:"기본·보충·채점 가능 시간 구분"}};
    const b = browser({initialState:{projects:[project],active_project_id:"bora"}}); await flush();
    const lines = b.elements.get("material-coverage-list").children;
    assert.equal(lines[0].textContent,"한국어 낭독: 기술 검사 3.00h · 검수 승인 3.00h / 3h");
    assert.match(lines[1].textContent,/기술 검사: 기본 30.000h · 초기화 보충 4.800분 · 채점 가능 30.010h/);
    assert.match(lines[2].textContent,/검수 승인: 기본 30.000h · 초기화 보충 4.800분 · 채점 가능 30.010h/);
    project.evaluation_materials.admitted = undefined;
    project.evaluation_materials.duration = {candidate:{core_hours:30,initialization_margin_hours:.08}};
    await b.tick();
    const waiting = b.elements.get("material-coverage-list").children;
    assert.equal(waiting.length,2);
    assert.match(waiting[1].textContent,/기술 검사: 기본 30.000h.*채점 가능 확인 전/);
    assert.equal(waiting.some(line=>line.className === "material-duration" && /검수 승인/.test(line.textContent)),false);
    project.evaluation_materials = {candidate:{target_hours_by_bucket:{korean_read:3},actual_hours_by_bucket:{korean_read:3.08}},scope:"기존 v3 결과"};
    await b.tick();
    const legacy = b.elements.get("material-coverage-list").children;
    assert.equal(legacy.length,1);
    assert.match(legacy[0].textContent,/기술 검사 3.08h/);
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
  await test("MLS and YODAS2 raw sources require their download receipt and select the expected default pool", async () => {
    for (const [dataset, pool] of [["mls", "generic_negative"], ["yodas2", "korean_negative"]]) {
      assert.ok(html.includes(`value="${dataset}"`));
      const project = { id: "bora", wakeword: "보라야", status: "waiting_input", input: { kind: "material_admissions", remaining_hours: 12 } };
      const b = browser({ initialState: { projects: [project] } }); await flush();
      b.elements.get("source-dataset").value = dataset; b.elements.get("source-dataset").handlers.change();
      b.elements.get("source-directory").value = "F:\\public-data\\audio";
      assert.equal(b.elements.get("source-pool").value, pool);
      assert.equal(b.elements.get("source-format-note").hidden, false);
      assert.equal(b.elements.get("source-metadata-details").open, true);
      assert.equal(b.elements.get("source-metadata").required, true);
      assert.equal(b.elements.get("source-metadata").placeholder, "download_receipt.json 경로");
      await b.submit("input-form");
      assert.equal(b.calls.some(call => call.url.endsWith("/input")), false);
      assert.match(b.elements.get("message").textContent, /download_receipt\.json/);
      b.elements.get("source-metadata").value = "F:\\public-data\\download_receipt.json";
      await b.submit("input-form");
      const request = b.calls.filter(call => call.url.endsWith("/input")).at(-1);
      assert.deepEqual(JSON.parse(request.options.body), { source: { directory: "F:\\public-data\\audio", dataset, pool, hours: 12, metadata: "F:\\public-data\\download_receipt.json" } });
    }
  });
  await test("prefilled raw formats keep an explicit pool and metadata while legacy lists stay optional", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_input", input: { kind: "material_admissions", source: { dataset: "mls", directory: "F:\\mls", hours: 2, metadata: "F:\\mls\\download_receipt.json" } } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("source-pool").value, "generic_negative");
    assert.equal(b.elements.get("source-metadata-details").open, true);
    assert.equal(b.elements.get("source-metadata").value, "F:\\mls\\download_receipt.json");
    b.elements.get("source-dataset").value = "yodas2"; b.elements.get("source-dataset").handlers.change();
    b.elements.get("source-pool").value = "generic_negative"; b.elements.get("source-pool").handlers.change();
    await b.submit("input-form");
    const request = b.calls.filter(call => call.url.endsWith("/input")).at(-1);
    assert.equal(JSON.parse(request.options.body).source.pool, "generic_negative");
    b.elements.get("source-dataset").value = "manifest"; b.elements.get("source-dataset").handlers.change();
    assert.equal(b.elements.get("source-format-note").hidden, true);
    assert.equal(b.elements.get("source-metadata").required, false);
    assert.equal(b.elements.get("source-metadata").placeholder, "해당 자료의 목록 파일");
    b.elements.get("material-mode").value = "admissions"; b.elements.get("material-mode").handlers.change();
    assert.equal(b.elements.get("source-metadata").required, false);
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
    assert.equal(b.elements.get("project-switch").hidden, false);
    assert.equal(b.elements.get("setup").hidden, false); assert.equal(b.elements.get("review-frame").getAttribute("src"), null);
    const failed = browser({ recipeError: true }); await flush();
    assert.equal(failed.elements.get("project-switch").hidden, true);
    assert.equal(failed.elements.get("create").disabled, true); assert.match(failed.elements.get("message").textContent, /recipe failure/);
  });
  console.log(`Studio UI: ${tests.length} tests passed`);
  for (const name of tests) console.log("  ✓ " + name);
})().catch(error => { console.error(error); process.exitCode = 1; });
