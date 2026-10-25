"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const script = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio.js"), "utf8");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio.html"), "utf8");
const flush = async () => { for (let index = 0; index < 12; index++) await new Promise(resolve => setImmediate(resolve)); };
const locationSettled = async () => { await new Promise(resolve => setTimeout(resolve, 230)); await flush(); };
const tests = [];
async function test(name, action) { await action(); tests.push(name); }

function browser({ noBoot = false, initialState, recipeError = false, recipeOverrides = {}, recipeList, recipeDirectory = "F:\\test\\models", locationResponse, createdDirectory, logResponse = "ready", logError = false } = {}) {
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
    if (url === "/api/state") { if (state.fetchError) throw state.fetchError; if (state.httpError) return json(state.httpError.body, state.httpError.status); return json(state.value); }
    if (url === "/api/recipes") return recipeError ? json({ error: "recipe failure" }, 500) : json({ presets, default_directory: recipeDirectory });
    if (url === "/api/model-location" && options.method === "POST") {
      const payload = JSON.parse(options.body);
      return json(locationResponse ? await locationResponse(payload) : { directory: recipeDirectory + "\\model_0001" });
    }
    if (url === "/api/projects" && options.method === "POST") {
      const payload = JSON.parse(options.body);
      const project = { id: "bora", wakeword: payload.wakeword, status: "ready", preset: payload.preset, directory: createdDirectory || recipeDirectory + "\\model_0001", stage: "data", error: null };
      state.value.projects.push(project); state.value.active_project_id = project.id;
      return json(project);
    }
    if (/\/api\/projects\/bora\/(start|resume|pause|input|migrate)$/.test(url)) return json({ ok: true });
    if (url.endsWith("/log")) { if (logError) throw new Error("optional log unavailable"); return json({ log: logResponse }); }
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
const downloadFile = (id, extra = {}) => ({ id, label: id, group: id, group_label: id, state: "waiting", done: 0, total: 1024, network_bytes: 0, bytes_per_second: null, eta_seconds: null, ...extra });
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
    generic_hours: 2000, korean_hours: 1, train_references: 20,
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

function storageBudgetRecipe(overrides = {}) {
  const variables = { negative_hours: 5000, negative_feature_rows: 9000000,
    positive_feature_rows: 500000, tts_slots: 112024, reference_count: 140, hard_originals: 2000 };
  const row = (id, label, classification, terms, detail) => ({ id, label, classification, terms, detail,
    bytes: Math.ceil(Object.entries(terms).reduce((sum, [key, value]) => sum + value * (key === "constant" ? 1 : variables[key]), 0)) });
  const rows = [
    row("archives", "고정 공개 자료", "published", { constant: 1000 }, "제공자가 공개한 파일 크기"),
    row("negative_audio", "비호출 변환 음원", "reference", { negative_hours: 115200000 }, "16 kHz PCM 비교 기준"),
    row("negative_features", "비호출 특징", "calculated", { negative_feature_rows: 6144 }, "명목 2초 window 기준"),
    row("positive_features", "호출어 특징", "calculated", { positive_feature_rows: 6144 }, "원본 수 × 증강 횟수"),
    row("tts", "TTS 음원", "reference", { tts_slots: 32000 }, "음원당 비교 기준"),
    row("references", "참조 음색", "measured", { reference_count: 1200 }, "측정 파일의 비교값"),
    row("hard_features", "유사어 특징", "calculated", { hard_originals: 6144 }, "고정 유사어 수"),
    row("zero", "추가 사본", "calculated", { constant: 0 }, "추가 사본 없음"),
    { id: "metadata", label: "목록·검사 기록", bytes: null, classification: "unresolved", detail: "실제 음원 수 확정 후 측정" }
  ];
  return commonRecipe({ id: "production", originals: 100000, rounds: 5, generic_hours: 3500, korean_hours: 1500,
    material_acquisition_policy: "automatic_v3",
    automatic_source_plan: { components: [{ pool: "generic_negative" }, { pool: "korean_negative" }] },
    // This old forecast is intentionally invalid: a new ledger must never fall back to it.
    automatic_space_plan: { base_archive_bytes: 999999999999 },
    storage_budget: { version: 1, complete: false, variables, rows,
      reference_subtotal_bytes: rows.reduce((sum, item) => sum + (item.bytes ?? 0), 0),
      total_required_bytes: null, peak_required_bytes: null,
      assumptions: ["비교 합계에는 조건부 참고값이 포함됩니다."],
      unknowns: [{ id: "metadata", label: "목록·검사 기록", detail: "실제 음원 수 확정 후 측정" },
        { id: "refill", label: "자료 보충", detail: "품질 검사 결과에 따라 달라짐" }],
      removed_legacy_allowance_bytes: 33285996544 }, ...overrides });
}

(async () => {
  await test("budget request preserves selected values and does not invent overrides", () => {
    const { helpers } = browser({ noBoot: true });
    const result = helpers.requestBody({ wakeword: " 보라야 ", advanced: { positive_originals: "", generic_hours: "" } }, "check");
    assert.deepEqual(JSON.parse(JSON.stringify(result)), { wakeword: "보라야", preset: "check" });
    const custom = helpers.requestBody({ wakeword: "보라야", directory: "F:\\bora", advanced: { positive_originals: "500", augmentation_rounds: "3", korean_hours: "1.5" } }, "trial");
    assert.equal(Object.hasOwn(custom, "directory"), false); assert.equal(custom.advanced.positive_originals, 500); assert.equal(custom.advanced.korean_hours, 1.5);
    for (const value of ["-1", "1.5", "Infinity"]) assert.throws(() => helpers.requestBody({ wakeword: "보라야", advanced: { positive_originals: value } }, "check"));
  });
  await test("new model storage previews the full server-provided read-only location without a directory override", async () => {
    const b = browser({ recipeDirectory: "F:\\workspace\\models" }); await flush();
    assert.equal(b.elements.has("directory"), false);
    assert.equal(b.elements.has("storage-note"), false);
    assert.match(b.elements.get("storage-directory").textContent, /호출어를 입력하면/);
    assert.equal(b.elements.get("create").disabled, true);
    assert.doesNotMatch(html, /(?:id|name)="directory"|새 프로젝트/);
    assert.doesNotMatch(script, /\$\("directory"\)|form\.directory/);
    b.elements.get("wakeword").value = "보라야";
    b.elements.get("wakeword").handlers.input();
    assert.equal(b.elements.get("create").disabled, true);
    await locationSettled();
    assert.equal(b.elements.get("storage-directory").textContent, "F:\\workspace\\models\\model_0001");
    assert.equal(b.elements.get("create").disabled, false);
    await b.submit("create-form");
    const request = b.calls.find(call => call.url === "/api/projects" && call.options.method === "POST");
    assert.deepEqual(JSON.parse(request.options.body), { wakeword: "보라야", preset: "check" });
    const preview = b.calls.find(call => call.url === "/api/model-location");
    assert.equal(preview.options.headers["X-Koww-Token"], "local-token");
    assert.deepEqual(JSON.parse(preview.options.body), JSON.parse(request.options.body));
    assert.equal(b.elements.get("project-directory").textContent, "F:\\workspace\\models\\model_0001");
  });
  await test("existing model locations remain visible verbatim and storage labels are escaped as text", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "ready", directory: "F:\\old-workspace\\projects\\bora" };
    const label = "F:\\<img src=x onerror=bad()>\\models";
    const b = browser({ initialState: { projects: [project], active_project_id: "bora" }, locationResponse: async () => ({ directory: label }) }); await flush();
    assert.equal(b.elements.get("project-directory").textContent, project.directory);
    await b.click("new-project");
    b.elements.get("wakeword").value = "보라야"; b.elements.get("wakeword").handlers.input();
    await locationSettled();
    assert.equal(b.elements.get("storage-directory").textContent, label);
    assert.equal(b.elements.get("storage-directory").children.length, 0);
    assert.match(textOf(b.elements.get("project-select")), /새 모델/);
  });
  await test("location previews debounce edits, follow every workload choice and do not repeat on polling", async () => {
    const b = browser(); await flush();
    for (const word of ["보", "보라", "보라야"]) { b.elements.get("wakeword").value = word; b.elements.get("wakeword").handlers.input(); }
    await locationSettled();
    assert.equal(b.calls.filter(call => call.url === "/api/model-location").length, 1);
    for (const [id, key, value] of [["adv-originals", "positive_originals", "1234"], ["adv-rounds", "augmentation_rounds", "3"], ["adv-updates", "updates", "20000"], ["adv-generic", "generic_hours", "2001"], ["adv-korean", "korean_hours", "2"]]) {
      b.elements.get(id).value = value; b.elements.get(id).handlers.input();
      assert.equal(b.elements.get("create").disabled, true);
      await locationSettled();
      const last = b.calls.filter(call => call.url === "/api/model-location").at(-1);
      assert.equal(JSON.parse(last.options.body).advanced[key], Number(value));
    }
    const trialRadio = descendants(b.elements.get("presets")).find(node => node.tagName === "input" && node.value === "trial");
    trialRadio.handlers.change(); await locationSettled();
    assert.equal(JSON.parse(b.calls.filter(call => call.url === "/api/model-location").at(-1).options.body).preset, "trial");
    const count = b.calls.filter(call => call.url === "/api/model-location").length;
    await b.tick(); await b.tick(); await locationSettled();
    assert.equal(b.calls.filter(call => call.url === "/api/model-location").length, count);
    assert.equal(b.calls.some(call => /\/api\/projects.*(?:start|resume)/.test(call.url)), false);
  });
  await test("late preview responses cannot replace the current path or re-enable creation after input is cleared", async () => {
    const pending = [];
    const b = browser({ locationResponse: payload => new Promise(resolve => pending.push({ payload, resolve })) }); await flush();
    b.elements.get("wakeword").value = "보라야"; b.elements.get("wakeword").handlers.input(); await locationSettled();
    b.elements.get("wakeword").value = "다음아"; b.elements.get("wakeword").handlers.input(); await locationSettled();
    assert.equal(pending.length, 2);
    pending[1].resolve({ directory: "F:\\test\\models\\model_0002" }); await flush();
    assert.equal(b.elements.get("create").disabled, false);
    pending[0].resolve({ directory: "F:\\test\\models\\model_0001" }); await flush();
    assert.equal(b.elements.get("storage-directory").textContent, "F:\\test\\models\\model_0002");
    b.elements.get("adv-updates").value = "30000"; b.elements.get("adv-updates").handlers.input(); await locationSettled();
    assert.equal(b.elements.get("create").disabled, true);
    b.elements.get("wakeword").value = ""; b.elements.get("wakeword").handlers.input();
    pending[2].resolve({ directory: "F:\\test\\models\\model_0003" }); await flush();
    assert.match(b.elements.get("storage-directory").textContent, /호출어를 입력하면/);
    assert.equal(b.elements.get("create").disabled, true);
  });
  await test("failed, incomplete and stale choices cannot create a model before a valid location is displayed", async () => {
    let result = { directory: "" };
    const b = browser({ locationResponse: async () => result }); await flush();
    b.elements.get("wakeword").value = "보라야"; b.elements.get("wakeword").handlers.input(); await locationSettled();
    assert.match(b.elements.get("storage-directory").textContent, /확인 실패/);
    assert.equal(b.elements.get("create").disabled, true);
    result = { directory: "F:\\test\\models\\model_0001" };
    b.elements.get("adv-updates").value = "-1"; b.elements.get("adv-updates").handlers.input();
    await locationSettled();
    assert.match(b.elements.get("storage-directory").textContent, /입력한 값을/);
    b.elements.get("adv-updates").value = "10000"; await b.submit("create-form");
    assert.equal(b.calls.some(call => call.url === "/api/projects"), false);
    await locationSettled(); assert.equal(b.elements.get("create").disabled, false);
  });
  await test("creation displays the authoritative returned path if another model occupied the previewed slot", async () => {
    const b = browser({ createdDirectory: "F:\\test\\models\\model_0002" }); await flush();
    b.elements.get("wakeword").value = "보라야"; b.elements.get("wakeword").handlers.input(); await locationSettled();
    assert.equal(b.elements.get("storage-directory").textContent, "F:\\test\\models\\model_0001");
    await b.submit("create-form");
    assert.equal(b.elements.get("project-directory").textContent, "F:\\test\\models\\model_0002");
  });
  await test("unknown progress and ETA stay unknown, and units are formatted separately", () => {
    const { helpers } = browser({ noBoot: true });
    assert.equal(helpers.progressValue({}), null); assert.equal(helpers.progressValue({ progress: { done: 0, total: 0 } }), null);
    assert.equal(helpers.progressValue({ progress: { done: null, total: 100 } }), null);
    assert.equal(helpers.progressValue({ progress: { done: 25, total: 100, unit: "개" } }).percent, 25);
    assert.equal(helpers.eta(null), ""); assert.equal(helpers.eta([3600, 5400]), "1시간–1시간 30분");
    assert.equal(helpers.bytes(1073741824), "1 GiB"); assert.equal(helpers.duration(0), "0초");
  });
  await test("audited storage scales explicit terms without changing unknowns, fixed slots or the recipe", () => {
    const { helpers } = browser({ noBoot: true }), recipe = storageBudgetRecipe(), before = JSON.stringify(recipe);
    const base = helpers.storageBudget(recipe);
    assert.equal(base.reference_subtotal_bytes, recipe.storage_budget.reference_subtotal_bytes);
    assert.equal(base.total_required_bytes, null); assert.equal(base.peak_required_bytes, null);
    assert.equal(base.rows.find(row => row.id === "metadata").bytes, null);
    const updated = helpers.storageBudget(recipe, { genericHours: 1000.001, koreanHours: 200, originals: 20000, rounds: 3 });
    assert.equal(updated.variables.negative_hours, 1200.001);
    assert.equal(updated.variables.negative_feature_rows, Math.ceil(1200.001 * 1800));
    assert.equal(updated.variables.positive_feature_rows, 60000);
    assert.equal(updated.variables.tts_slots, 32024);
    for (const key of ["reference_count", "hard_originals"]) assert.equal(updated.variables[key], base.variables[key]);
    assert.equal(updated.reference_subtotal_bytes, updated.rows.reduce((sum, row) => sum + (row.bytes ?? 0), 0));
    assert.equal(updated.rows.find(row => row.id === "archives").bytes, 1000);
    assert.equal(JSON.stringify(recipe), before);
  });
  await test("audited storage keeps feature-only pools outside raw audio calculations and permits measured zero", () => {
    const { helpers } = browser({ noBoot: true }), recipe = storageBudgetRecipe();
    recipe.automatic_source_plan.components = [{ pool: "korean_negative" }];
    const budget = helpers.storageBudget(recipe, { genericHours: 9999, koreanHours: 0 });
    assert.equal(budget.variables.negative_hours, 0); assert.equal(budget.variables.negative_feature_rows, 0);
    assert.equal(budget.rows.find(row => row.id === "negative_audio").bytes, 0);
    assert.equal(budget.rows.find(row => row.id === "zero").bytes, 0);
    recipe.automatic_source_plan.components = [];
    assert.equal(helpers.storageBudget(recipe).variables.negative_hours, 0);
  });
  await test("audited storage rejects absent, null and malformed numeric fields rather than silently treating them as zero", () => {
    const { helpers } = browser({ noBoot: true });
    const mutations = [
      recipe => { delete recipe.storage_budget.variables.negative_hours; },
      recipe => { recipe.storage_budget.variables.negative_hours = null; },
      recipe => { recipe.storage_budget.reference_subtotal_bytes = null; },
      recipe => { delete recipe.storage_budget.rows[0].bytes; },
      recipe => { recipe.storage_budget.rows[0].bytes = null; },
      recipe => { recipe.storage_budget.rows[0].terms.constant = null; },
      recipe => { recipe.storage_budget.rows[0].terms.constant = "1000"; },
      recipe => { recipe.storage_budget.rows[0].bytes += 10; },
      recipe => { recipe.storage_budget.rows[0].terms.other = 1; },
      recipe => { recipe.storage_budget.rows.at(-1).terms = false; },
      recipe => { recipe.storage_budget.total_required_bytes = 0; },
      recipe => { delete recipe.storage_budget.peak_required_bytes; }
    ];
    for (const mutate of mutations) { const recipe = storageBudgetRecipe(); mutate(recipe); assert.equal(helpers.storageBudget(recipe), null); }
    assert.equal(helpers.storageBudget(storageBudgetRecipe(), { genericHours: null }), null);
    assert.equal(helpers.storageBudget(storageBudgetRecipe(), { originals: 1.5 }), null);
    assert.equal(helpers.storageBudget({}), null);
  });
  await test("new storage summary says the full requirement is unknown and presents evidence in a separate disclosure", async () => {
    const recipe = storageBudgetRecipe(), b = browser({ recipeList: [recipe] }); await flush();
    const note = b.elements.get("source-plan-note"), details = b.elements.get("storage-budget-details"), content = b.elements.get("storage-budget-content");
    assert.equal(note.hidden, false); assert.equal(details.hidden, false);
    assert.match(note.textContent, /^전체 필요 공간: 미확정\. 계산 가능한 항목의 비교 합계 /);
    assert.equal(/최소 저장 공간|예상 필요 공간|WAV 기준 참고 예산/.test(note.textContent), false);
    assert.match(textOf(content), /항목.*용량.*구분.*근거/);
    assert.match(textOf(content), /실제 FLAC 저장량이나 최대 필요 공간이 아닙니다/);
    assert.match(textOf(content), /추가 사본 0 B 형식 계산/);
    assert.match(textOf(content), /목록·검사 기록 미확정 미확정 실제 음원 수 확정 후 측정/);
    assert.match(textOf(content), /자료 보충 미확정 미확정 품질 검사 결과에 따라 달라짐/);
    assert.equal(descendants(content).filter(node => node.textContent === "목록·검사 기록").length, 1);
    details.open = true;
    b.elements.get("adv-originals").value = "20000"; b.elements.get("adv-originals").handlers.input();
    const estimate = b.helpers.storageBudget(recipe, { originals: 20000 });
    assert.ok(note.textContent.includes(b.helpers.bytes(estimate.reference_subtotal_bytes)));
    assert.equal(details.open, true);
    assert.equal(b.calls.some(call => /download|prepare/.test(call.url)), false);
  });
  await test("storage evidence uses text nodes for untrusted labels and malformed ledgers never fall back to legacy totals", async () => {
    const recipe = storageBudgetRecipe();
    recipe.storage_budget.rows[0].label = "<img src=x onerror=bad()>";
    recipe.storage_budget.rows[0].detail = "<script>bad()</script>";
    recipe.storage_budget.assumptions.push("<iframe src=bad>");
    recipe.storage_budget.unknowns[1].detail = "<svg onload=bad()>";
    const b = browser({ recipeList: [recipe] }); await flush();
    const content = b.elements.get("storage-budget-content");
    for (const value of ["<img src=x onerror=bad()>", "<script>bad()</script>", "<iframe src=bad>", "<svg onload=bad()>"]) assert.ok(textOf(content).includes(value));
    assert.equal(descendants(content).some(node => ["img", "script", "iframe", "svg"].includes(node.tagName)), false);
    for (const storage_budget of [null, {}, { version: 1 }]) {
      const invalid = browser({ recipeList: [storageBudgetRecipe({ storage_budget })] }); await flush();
      assert.equal(invalid.elements.get("source-plan-note").textContent, "전체 필요 공간: 미확정. 용량 계산 근거를 확인할 수 없습니다.");
      assert.equal(invalid.elements.get("storage-budget-details").hidden, true);
      assert.equal(invalid.elements.get("storage-budget-content").children.length, 0);
    }
  });
  await test("switching back to a historical recipe clears the new storage evidence and retains legacy behavior", async () => {
    const b = browser({ recipeList: [storageBudgetRecipe(), commonRecipe()] }); await flush();
    // The default selected recipe is check; choose production, then return.
    const radios = descendants(b.elements.get("presets")).filter(node => node.tagName === "input");
    radios.find(node => node.value === "production").handlers.change();
    assert.equal(b.elements.get("storage-budget-details").hidden, false);
    radios.find(node => node.value === "check").handlers.change();
    assert.equal(b.elements.get("storage-budget-details").hidden, true);
    assert.equal(b.elements.get("storage-budget-content").children.length, 0);
    assert.equal(b.elements.get("source-plan-note").hidden, true);
  });
  if (process.env.KOWW_STUDIO_STORAGE_FIXTURE) await test("actual Studio recipes storage evidence agrees with backend arithmetic and renders every evidence row", async () => {
    // Optional integration input is produced by the backend without acquiring
    // audio or touching training/Final state; ordinary UI tests stay standalone.
    const fixture = JSON.parse(fs.readFileSync(process.env.KOWW_STUDIO_STORAGE_FIXTURE, "utf8"));
    const b = browser({ recipeList: [fixture.recipe] }); await flush();
    for (const item of fixture.cases) {
      const actual = b.helpers.storageBudget(fixture.recipe, item.selected);
      assert.ok(actual);
      assert.equal(actual.reference_subtotal_bytes, item.expected.reference_subtotal_bytes);
      assert.deepEqual(JSON.parse(JSON.stringify(actual.variables)), item.expected.variables);
      assert.deepEqual(JSON.parse(JSON.stringify(actual.rows.map(row => [row.id, row.bytes]))), item.expected.rows.map(row => [row.id, row.bytes]));
      for (const [key, id] of [["originals", "adv-originals"], ["rounds", "adv-rounds"], ["genericHours", "adv-generic"], ["koreanHours", "adv-korean"]]) {
        b.elements.get(id).value = Object.hasOwn(item.selected, key) ? String(item.selected[key]) : "";
      }
      b.elements.get("adv-originals").handlers.input();
      assert.ok(b.elements.get("source-plan-note").textContent.includes(b.helpers.bytes(item.expected.reference_subtotal_bytes)));
      const content = b.elements.get("storage-budget-content");
      assert.equal(descendants(content).filter(node => node.tagName === "tbody")[0].children.length, item.expected.rows.length);
      for (const row of item.expected.rows) assert.ok(textOf(content).includes(row.label), row.label);
      assert.equal(item.expected.total_required_bytes, null); assert.equal(item.expected.peak_required_bytes, null);
    }
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
    assert.equal(b.elements.get("create").disabled, true);
    const preview = b.elements.get("source-preview");
    const label = descendants(preview).find(node => node.textContent === "<img src=x onerror=bad()>");
    assert.ok(label); assert.notEqual(label.tagName, "a");
    assert.equal(descendants(preview).some(node => node.tagName === "img"), false);
  });
  await test("each training scale shows its reference voice count while common evaluation keeps 30 voices", async () => {
    const recipes = [["check", 20, 1000], ["trial", 100, 20000], ["production", 500, 100000]]
      .map(([id, train_references, originals]) => commonRecipe({ id, train_references, originals }));
    const b = browser({ recipeList: recipes }); await flush();
    for (const [index, recipe] of recipes.entries()) {
      const preset = b.elements.get("presets").children[index];
      assert.match(textOf(preset), new RegExp(`학습 기준 목소리 ${recipe.train_references}개`));
      descendants(preset).find(node => node.tagName === "input").handlers.change();
      const groups = b.elements.get("source-preview").children;
      assert.match(textOf(material(groups[0], "호출어 음성")), new RegExp(`기준 목소리 ${recipe.train_references}개`));
      assert.match(textOf(material(groups[1], "호출어 음성")), /30가지 음색 계열/);
    }
    assert.equal(b.calls.some(call => /\/api\/projects(?:\/|$)/.test(call.url) && call.options?.method === "POST"), false);
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
    const recipe = commonRecipe({ id: "trial", korean_hours: 2000 / 3, material_acquisition_policy: "automatic_v3",
      automatic_source_plan: { version: 3, enabled: true, components: [{ id: "korean", provider: "yodas2", pool: "korean_negative", target_hours: 2000 / 3 }] } });
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
    recipe.material_acquisition_policy = "automatic_v3";
    recipe.automatic_source_plan = { version: 3, enabled: true, components: recipe.training_negative_plan.components.map(item => ({ ...item, provider: item.sources[0].id })) };
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
  await test("current trial selects YODAS automatically and preserves fixed source details", async () => {
    const recipe = commonRecipe({ id: "trial", korean_hours: 2000 / 3, material_acquisition_policy: "automatic_v3",
      automatic_source_plan: { version: 3, enabled: true, components: [{ id: "yodas2", provider: "yodas2", pool: "korean_negative", target_hours: 2000 / 3 }] } });
    recipe.assets = recipe.assets.filter(item => item.id !== "zeroth");
    const b = browser({ recipeList: [recipe] }); await flush();
    const training = b.elements.get("source-preview").children[0];
    assert.match(textOf(material(training, "한국어 비호출 자료")), /YODAS2 한국어.*목표량 자동 선정/);
    assert.equal(textOf(training).includes("직접 준비할"), false);
    assert.equal(textOf(training).includes("Zeroth"), false);
    assert.equal(b.elements.get("source-plan-note").hidden, false);
  });
  await test("new setup does not calculate or display obsolete v1/v2 storage forecasts", async () => {
    for (const policy of ["automatic_v1", "automatic_v2"]) {
      const recipe = commonRecipe({ material_acquisition_policy: policy,
        automatic_source_plan: { version: Number(policy.at(-1)), enabled: true, components: [] },
        automatic_space_plan: { base_archive_bytes: 1, base_working_bytes: 2, future_buffer_bytes: 3, transient_archive_bytes: 0 } });
      const b = browser({ recipeList: [recipe] }); await flush();
      assert.equal(b.helpers.storageEstimate(recipe), null);
      assert.equal(b.elements.get("source-plan-note").hidden, true);
      assert.equal(b.calls.some(call => /download|prepare/.test(call.url)), false);
    }
  });
  await test("canonical FLAC storage shows a WAV reference budget without inventing a compression ratio or a minimum", async () => {
    const recipe = commercialRecipeV3();
    Object.assign(recipe, { originals: 100000, rounds: 5, material_acquisition_policy: "automatic_v3",
      automatic_source_plan: { version: 3, enabled: true, components: recipe.training_negative_plan.components.map(item => ({ ...item,
        provider: { mixed: "audioset_unbalanced", multilingual: "mls", korean_web: "yodas2" }[item.id] || item.sources[0].id })) },
      automatic_space_plan: { base_archive_bytes: 1000, base_working_bytes: 2000, future_buffer_bytes: 3000,
        transient_source_working_bytes: 8 * 1024 ** 3, transient_source_bytes: 125000000000,
        transient_archive_bytes: 0, canonical_flac_estimated_bytes: null, estimated_retained_source_bytes: 0 } });
    const before = JSON.stringify(recipe);
    const b = browser({ recipeList: [recipe], initialState: { projects: [], system: { free_bytes: 10 ** 12 } } }); await flush();
    const estimate = b.helpers.storageEstimate(recipe), note = b.elements.get("source-plan-note").textContent;
    assert.equal(b.helpers.minimumStorage, undefined);
    assert.equal(estimate.basis, "wav_reference");
    assert.equal(estimate.retained, 0);
    assert.equal(estimate.reference, 1152000000000);
    assert.equal(estimate.known, 113664000000);
    assert.equal(estimate.transient, 125000000000);
    assert.equal(estimate.total, 1390664006000);
    assert.match(note, /WAV 기준 참고 예산 .*임시/);
    assert.match(note, /무손실 FLAC.*실제 용량은 변환 후/);
    assert.match(note, /여유가 참고 예산보다 적습니다/);
    assert.match(note, /다운로드량은 같으며 원본 재처리 시 다시 받습니다/);
    assert.equal(note.includes("최소"), false);
    assert.equal(note.includes("예상 필요 공간"), false);
    const half = b.helpers.storageEstimate(recipe, { genericHours: 3500, koreanHours: 1500, originals: 1000, rounds: 1 });
    assert.equal(half.reference, 576000000000);
    assert.equal(half.transient, 62500000000);
    for (const [id, value] of [["adv-generic", "3500"], ["adv-korean", "1500"], ["adv-originals", "1000"], ["adv-rounds", "1"]]) {
      b.elements.get(id).value = value; b.elements.get(id).handlers.input();
    }
    assert.ok(b.elements.get("source-plan-note").textContent.includes(b.helpers.bytes(half.total)));
    assert.equal(JSON.stringify(recipe), before);
    assert.equal(b.helpers.storageEstimate({ ...recipe, automatic_space_plan: { ...recipe.automatic_space_plan, transient_source_working_bytes: undefined } }), null);
  });
  await test("5000-hour production recipe renders the halved sources and keeps the positive workload", async () => {
    const recipe = commercialRecipeV3();
    recipe.generic_hours = 3500; recipe.korean_hours = 1500;
    recipe.training_negative_plan.generic_hours = 3500;
    recipe.training_negative_plan.korean_hours = 1500;
    recipe.training_negative_plan.total_hours = 5000;
    for (const item of recipe.training_negative_plan.components) item.target_hours /= 2;
    Object.assign(recipe, { originals: 100000, rounds: 5, updates: 100000,
      material_acquisition_policy: "automatic_v3",
      automatic_source_plan: { version: 3, enabled: true, components: recipe.training_negative_plan.components.map(item => ({ ...item,
        provider: { mixed: "audioset_unbalanced", multilingual: "mls", korean_web: "yodas2" }[item.id] || item.sources[0].id })) },
      automatic_space_plan: { base_archive_bytes: 31592837906, base_working_bytes: 16106127360,
        future_buffer_bytes: 8 * 1024 ** 3, transient_source_working_bytes: 8 * 1024 ** 3,
        transient_archive_bytes: 0, canonical_flac_estimated_bytes: null, estimated_retained_source_bytes: 0 } });
    const b = browser({ recipeList: [recipe] }); await flush();
    const estimate = b.helpers.storageEstimate(recipe);
    assert.equal(estimate.reference, 576000000000);
    assert.equal(estimate.known, 58368000000);
    assert.equal(estimate.transient, 62500000000);
    assert.equal(estimate.total, 753156899858);
    assert.equal(b.elements.get("adv-generic").placeholder, "3500");
    assert.equal(b.elements.get("adv-korean").placeholder, "1500");
    assert.match(b.elements.get("source-plan-note").textContent, /WAV 기준 참고 예산/);
  });
  await test("per-track FMA replaces only the batch scratch forecast and preserves old estimates", async () => {
    const recipe = commercialRecipeV3();
    Object.assign(recipe, { originals: 100000, rounds: 5, material_acquisition_policy: "automatic_v3",
      automatic_source_plan: { components: recipe.training_negative_plan.components.map(item => ({ ...item,
        provider: { mixed: "audioset_unbalanced", multilingual: "mls", korean_web: "yodas2" }[item.id] || item.sources[0].id })) },
      automatic_space_plan: { base_archive_bytes: 1000, base_working_bytes: 2000, future_buffer_bytes: 3000,
        transient_source_working_bytes: 8 * 1024 ** 3 } });
    const b = browser({ recipeList: [recipe] }); await flush();
    const legacy = b.helpers.storageEstimate(recipe);
    const current = { ...recipe, fma_acquisition_policy: "per_track_flac_prefetch_v1" };
    const streamed = b.helpers.storageEstimate(current);
    assert.equal(legacy.transient, 125000000000);
    assert.equal(streamed.transient, 8 * 1024 ** 3);
    for (const field of ["reference", "known", "retained"]) assert.equal(streamed[field], legacy[field]);
    assert.equal(legacy.total - streamed.total, legacy.transient - streamed.transient);
    assert.equal(b.helpers.storageEstimate({ ...current, fma_acquisition_policy: "unknown" }), null);
    assert.deepEqual(b.helpers.storageEstimate({ ...recipe, automatic_space_plan: {
      ...recipe.automatic_space_plan, fma_acquisition_policy: "per_track_flac_prefetch_v1" } }), streamed);
  });
  await test("canonical trial keeps source working space and automatic YODAS coverage", async () => {
    const recipe = commonRecipe({ id: "trial", korean_hours: 2000 / 3, material_acquisition_policy: "automatic_v3",
      automatic_source_plan: { version: 3, enabled: true, components: [{ id: "yodas2", provider: "yodas2", pool: "korean_negative", target_hours: 2000 / 3 }] },
      automatic_space_plan: { base_archive_bytes: 1000, base_working_bytes: 2000, future_buffer_bytes: 3000,
        transient_source_working_bytes: 8 * 1024 ** 3, transient_source_bytes: 8 * 1024 ** 3, transient_archive_bytes: 0 } });
    recipe.assets = recipe.assets.filter(item => item.id !== "zeroth");
    const b = browser({ recipeList: [recipe] }); await flush();
    assert.equal(b.helpers.storageEstimate(recipe).transient, 8 * 1024 ** 3);
    assert.match(textOf(material(b.elements.get("source-preview").children[0], "한국어 비호출 자료")), /YODAS2 한국어.*목표량 자동 선정/);
    assert.match(b.elements.get("source-plan-note").textContent, /WAV 기준 참고 예산/);
  });
  await test("canonical storage progress reports measured FLAC bytes separately and leaves unknown sizes unknown", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "running", training_materials: {
      targets: { generic_negative: 7000, korean_negative: 3000 },
      storage: { storage_policy: "canonical_flac_only_v1", canonical_storage_bytes: 1024 ** 3, source_storage_bytes: 99 * 1024 ** 3 }
    } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    const text = () => textOf(b.elements.get("training-material-coverage-list"));
    assert.match(text(), /학습용 FLAC 실제 보관량: 1 GiB/);
    assert.equal(text().includes("99 GiB"), false);
    project.training_materials.storage.canonical_storage_bytes = null; await b.tick();
    assert.equal(text().includes("FLAC 실제 보관량"), false);
    project.training_materials.storage.canonical_storage_bytes = 0; await b.tick();
    assert.match(text(), /FLAC 실제 보관량: 0 B/);
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
    for (const [version, total] of [[1, 80000], [2, 100000], [3, 112000], [4, 112000], [5, 48000]]) {
      const recipe = commonRecipe();
      recipe.evaluation_policy.id = `common_development_v${version}`;
      recipe.evaluation_policy.positive_sessions = total;
      recipe.evaluation_policy.reverb_originals = version === 5 ? 500 : version === 2 ? 625 : 1000;
      if (version === 5) Object.assign(recipe.evaluation_policy, { validation_originals: 2000, dry_positive_sessions: 32000, reverb_positive_sessions: 16000 });
      recipe.evaluation_sources.push({ id: "old_extra", name: "기존 독립 배경", role: "held_out_evaluation_noise", file_count: 193 });
      const b = browser({ recipeList: [recipe] }); await flush();
      const group = b.elements.get("source-preview").children[1], all = textOf(group);
      assert.match(all, new RegExp(total.toLocaleString("ko-KR") + "회"));
      if (version < 4) assert.match(all, /기존 독립 배경/);
      else assert.equal(all.includes("기존 독립 배경"), false);
      if (total !== 112000) assert.equal(all.includes("112,000회"), false);
      assert.equal(withClass(material(group, "호출어 음성"), "material-amount")[0].textContent, version === 5 ? "2,000개" : "5,000개");
    }
  });
  await test("all three workload scales share the 48000-call policy and unchanged common backgrounds", async () => {
    const policy = { ...commonRecipe().evaluation_policy, id: "common_development_v5", validation_originals: 2000,
      dry_positive_sessions: 32000, reverb_originals: 500, reverb_positive_sessions: 16000,
      positive_sessions: 48000, evaluation_materials_recipe: 6, evaluation_source_scope: "common_30h_only" };
    const recipes = [["check", 1000], ["trial", 20000], ["production", 100000]].map(([id, originals]) => commonRecipe({ id, originals, evaluation_policy: policy }));
    for (const recipe of recipes) recipe.evaluation_sources.push({ id: "old_extra", name: "기존 독립 배경", role: "held_out_evaluation_noise", file_count: 193 });
    const b = browser({ recipeList: recipes }); await flush();
    const radios = descendants(b.elements.get("presets")).filter(node => node.tagName === "input");
    assert.equal(radios.length, 3);
    for (const [index, radio] of radios.entries()) {
      radio.handlers.change();
      const [training, evaluation] = b.elements.get("source-preview").children;
      assert.equal(withClass(material(training, "호출어 음성"), "material-amount")[0].textContent, `${recipes[index].originals.toLocaleString("ko-KR")}개`);
      assert.equal(withClass(material(evaluation, "호출어 음성"), "material-amount")[0].textContent, "2,000개");
      assert.match(withClass(material(evaluation, "호출어 음성"), "material-description")[0].textContent, /30가지 음색/);
      assert.equal(withClass(material(evaluation, "소음 증강 음원"), "material-amount")[0].textContent, "30시간");
      assert.equal(withClass(material(evaluation, "잔향 자료"), "material-amount")[0].textContent, "80개");
      assert.match(textOf(evaluation), /가상 공간 20개/);
      assert.match(textOf(evaluation), /48,000회/);
      assert.match(textOf(evaluation), /30시간 이상/);
      assert.doesNotMatch(textOf(evaluation), /기존 독립 배경|112,000회/);
    }
    assert.equal(b.calls.some(call => /\/api\/projects.*(?:start|resume)/.test(call.url)), false);
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
  await test("result view uses plain language and retains the actual missing-data list", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "completed", results: { url: "/results/bora/", scope: "비호출 실측 0.25시간 · 독립 최종 평가 미완료", common_evaluation_assessment: { coverage_targets_passed: false, shortages: ["연속 비호출 평가 0.25/20시간", "평가 음색 계열 2/10개"] } } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("results-note").textContent, "같은 시험 오디오에서 모델별 성능을 비교한 결과입니다.");
    assert.equal(b.elements.get("resume-note").hidden, true);
    assert.match(b.elements.get("current-description").textContent, /모델 성능을 비교/);
    assert.equal(b.elements.get("evaluation-coverage").hidden, false);
    assert.match(b.elements.get("evaluation-coverage-summary").textContent, /확인 필요.*2개/);
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
    b.elements.get("wakeword").handlers.input(); await locationSettled();
    await b.submit("create-form");
    const mutations = b.calls.filter(call => call.options.method === "POST" && call.url !== "/api/model-location");
    assert.deepEqual(mutations.map(call => call.url), ["/api/projects", "/api/projects/bora/start"]);
    assert.deepEqual(JSON.parse(mutations[0].options.body), { wakeword: "보라야", preset: "check" });
    assert.ok(mutations.every(call => call.options.headers["X-Koww-Token"] === "local-token"));
    assert.equal(b.elements.get("setup").hidden, true); assert.equal(b.elements.get("project").hidden, false);
    assert.equal(b.elements.get("connection").hidden, true);
    assert.equal(b.elements.get("project-error").hidden, true);
  });
  await test("optional reference listening loads only on request and polling preserves playback without writing decisions", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "running", stage: "pilot",
      reference_preview: { url: "/reference-preview/?project=bora", count: 50, train: 20, validation: 30 } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    const details = b.elements.get("reference-preview"), frame = b.elements.get("reference-preview-frame");
    assert.match(html, /<summary>기준 목소리 듣기<\/summary>/);
    assert.doesNotMatch(html, /기준 목소리 듣기\s*\(선택\)/);
    assert.doesNotMatch(html.match(/<iframe id="reference-preview-frame"[^>]*>/)[0], /\ssrc=/);
    assert.equal(details.hidden, false); assert.equal(details.open, false);
    assert.equal(frame.getAttribute("src"), null); assert.equal(frame.srcWrites, 0);
    assert.equal(b.elements.get("reference-preview-counts").textContent, "학습용 20개 · 공통 평가용 30개");
    details.open = true; details.handlers.toggle();
    assert.equal(frame.getAttribute("src"), "/reference-preview/?project=bora");
    await b.tick(); await b.tick();
    assert.equal(details.open, true); assert.equal(frame.srcWrites, 1);
    project.stage = "training"; await b.tick();
    assert.equal(details.hidden, false); assert.equal(frame.srcWrites, 1);
    details.open = false; details.handlers.toggle();
    assert.equal(frame.getAttribute("src"), null);
    assert.equal(b.calls.some(call => call.options.method === "POST" && call.url !== "/api/model-location"), false);
  });
  await test("optional voice preview closes on model changes and rejects unavailable or external previews", async () => {
    const project = { id: "bora", status: "complete", reference_preview: { url: "/reference-preview/?project=bora", count: 50 } };
    const other = { id: "other", status: "running", reference_preview: { url: "/reference-preview/?project=other", count: 10 } };
    const b = browser({ initialState: { projects: [project, other], active_project_id: "bora" } }); await flush();
    const details = b.elements.get("reference-preview"), frame = b.elements.get("reference-preview-frame");
    details.open = true; details.handlers.toggle();
    b.elements.get("project-select").value = "other"; b.elements.get("project-select").handlers.change();
    assert.equal(details.open, false); assert.equal(frame.getAttribute("src"), null);
    details.open = true; details.handlers.toggle();
    assert.equal(frame.getAttribute("src"), "/reference-preview/?project=other");
    await b.click("new-project");
    assert.equal(details.hidden, true); assert.equal(details.open, false); assert.equal(frame.getAttribute("src"), null);
    other.reference_preview.url = "https://example.com/reference-preview/";
    b.elements.get("project-select").value = "other"; b.elements.get("project-select").handlers.change(); await b.tick();
    assert.equal(details.hidden, true); assert.equal(frame.getAttribute("src"), null);
    other.reference_preview = { url: "/reference-preview/?project=other", count: 0 }; await b.tick();
    assert.equal(details.hidden, true);
  });
  await test("automatic reference checking respects the task label and optional listening never replaces pilot review", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "running",
      stage: { id: "references_review", kind: "task", label: "기준 목소리 자동 검사" }, stage_label: "기존 목소리 확인",
      reference_preview: { url: "/reference-preview/?project=bora", count: 50, train: 20, validation: 30 } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("current-title").textContent, "기준 목소리 자동 검사");
    assert.equal(b.elements.get("review-section").hidden, true);
    project.status = "waiting_review"; project.stage = { id: "pilot_review", kind: "review", label: "호출어 발음 확인" };
    project.review = { queue_id: "pilot", title: "호출어 발음 확인", url: "/review/?project=bora&queue=pilot" }; await b.tick();
    const details = b.elements.get("reference-preview"); details.open = true; details.handlers.toggle();
    assert.equal(b.elements.get("review-section").hidden, false);
    assert.equal(b.elements.get("review-title").textContent, "호출어 발음 확인");
    assert.equal(b.elements.get("start").hidden, true);
    assert.equal(b.elements.get("review-frame").getAttribute("src"), "/review/?project=bora&queue=pilot");
    await b.tick();
    assert.equal(b.elements.get("review-frame").srcWrites, 1);
    assert.equal(b.elements.get("reference-preview-frame").srcWrites, 1);
    assert.equal(b.calls.some(call => call.options.method === "POST" && call.url !== "/api/model-location"), false);
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
  await test("synthetic voice stages explain voice checking without changing the saved workflow or iframe", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_review", stage: { id: "references_review", label: "합성 음색 듣기" }, stage_label: "합성 음색 듣기", review: { queue_id: "references", title: "음원 확인", url: "/review/?project=bora&queue=references" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("current-title").textContent, "합성 목소리 확인");
    assert.equal(b.elements.get("review-title").textContent, "합성 목소리 확인");
    assert.match(b.elements.get("review-description").textContent, /호출어 발음은 다음 단계/);
    assert.match(textOf(b.elements.get("stages")), /합성 목소리/);
    assert.equal(project.stage_label, "합성 음색 듣기");
    await b.tick(); assert.equal(b.elements.get("review-frame").srcWrites, 1);
    project.stage = { id: "references", label: "학습 · 평가용 합성 음색 생성" }; project.status = "running";
    await b.tick(); assert.equal(b.elements.get("current-title").textContent, "합성 목소리 만들기");
  });
  await test("voice review shows saved workflow counts without replacing the review iframe or applying new defaults", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_review", stage: "references_review", reference_counts: { train: 4, validation: 30 }, review: { queue_id: "references", url: "/review/?project=bora&queue=references" } };
    const b = browser({ recipeList: [commonRecipe()], initialState: { projects: [project] } }); await flush();
    assert.match(b.elements.get("review-description").textContent, /학습용 4개 · 공통 평가용 30개 · 총 34개/);
    const before = b.elements.get("review-frame").srcWrites;
    project.reference_counts = { train: 20, validation: 30 }; await b.tick();
    assert.match(b.elements.get("review-description").textContent, /학습용 20개 · 공통 평가용 30개 · 총 50개/);
    assert.equal(b.elements.get("review-frame").srcWrites, before);
    assert.equal(b.calls.some(call => /\/api\/projects(?:\/|$)/.test(call.url) && call.options?.method === "POST"), false);
  });
  await test("voice design previews render supplied scale distributions rather than a fixed voice budget", async () => {
    const design = count => ({ policy: "balanced_v1", counts: {
      train: { count, gender: { man: count / 2, woman: count / 2 }, age: { child:count/5, teenager:count/5, "young adult":count/5, "middle-aged":count/5, "older adult":count/5 }, timbre: { light: Math.ceil(count/2), warm: Math.floor(count/2) }, style: { neutral: count/4, friendly: count/4, urgent: count/4, projected: count/4 }, vocal_effort: { soft: Math.ceil(count/3), normal: Math.floor(count/3), strong: count-Math.ceil(count/3)-Math.floor(count/3) } },
      validation: { count: 30, gender: { man: 15, woman: 15 }, age: { child:6, teenager:6, "young adult":6, "middle-aged":6, "older adult":6 }, timbre: { light: 15, warm: 15 }, style: { neutral: 8, friendly: 8, urgent: 7, projected: 7 }, vocal_effort: { soft: 10, normal: 10, strong: 10 } }
    } });
    const recipes = [["check",20],["trial",100],["production",500]].map(([id,count]) => commonRecipe({ id, reference_design: design(count) }));
    const b = browser({ recipeList: recipes }); await flush();
    for (const [index, count] of [20,100,500].entries()) {
      if (index) { const radio = descendants(b.elements.get("presets").children[index]).find(node => node.tagName === "input"); await radio.handlers.change(); }
      assert.equal(b.elements.get("setup-voices").hidden, false);
      assert.equal(b.elements.get("setup-voices-train").textContent, `학습용 ${count}개`);
      assert.equal(b.elements.get("setup-voices-validation").textContent, "공통 평가용 30개");
      const rows = b.elements.get("setup-voices-rows").children;
      assert.equal(rows.length, 5);
      assert.match(textOf(rows[0]), new RegExp(`남성 ${count/2}개 · 여성 ${count/2}개`));
      assert.match(textOf(rows[1]), /어린이/); assert.match(textOf(rows[1]), /청소년/);
      assert.match(textOf(rows[1]), /중년/); assert.match(textOf(rows[1]), /고령/);
      assert.match(textOf(rows[4]), /보통/); assert.match(textOf(rows[4]), /강하게/);
    }
  });
  await test("legacy and supplemental voice review distributions remain factual and preserve iframe identity", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_review", stage: "references_review", reference_counts: { train: 20, validation: 30 },
      reference_design: { policy: "legacy", counts: { train: { count:20, gender:{man:10,woman:10}, age:{"young adult":20}, timbre:{light:20}, style:{neutral:20}, vocal_effort:{soft:20} }, validation:{count:30,gender:{man:15,woman:15},age:{adult:20,"young adult":10},timbre:{warm:30},style:{friendly:30},vocal_effort:{soft:22,strong:8}} } },
      review: { queue_id: "references_added_fixture", url: "/review/?project=bora&queue=references_added_fixture" } };
    const b = browser({ initialState: { projects:[project] } }); await flush();
    assert.equal(b.elements.get("review-voices").hidden, false);
    assert.equal(b.elements.get("review-title").textContent, "합성 목소리 확인");
    assert.match(textOf(b.elements.get("review-voices-rows")), /성인\(연령 미지정\) 20개/);
    assert.match(textOf(b.elements.get("review-voices-rows").children[4]), /부드럽게 20개/);
    const writes = b.elements.get("review-frame").srcWrites;
    await b.tick(); assert.equal(b.elements.get("review-frame").srcWrites,writes);
    project.reference_design = null; await b.tick(); assert.equal(b.elements.get("review-voices").hidden,true);
    assert.equal(b.calls.some(call => /\/api\/projects(?:\/|$)/.test(call.url) && call.options?.method === "POST"),false);
  });
  await test("same wakeword revisions use their model folders in the selector without changing IDs", async () => {
    const b = browser({ initialState: { active_project_id:"new", projects:[
      {id:"old",wakeword:"언년아",directory:"F:\\test\\models\\model_0001",status:"paused",stage:"references_review"},
      {id:"new",wakeword:"언년아",directory:"F:\\test\\models\\model_0002",status:"paused",stage:"references"}
    ] } }); await flush();
    const options = b.elements.get("project-select").children;
    assert.deepEqual(options.map(option => [option.textContent,option.value]), [["언년아 · model_0001","old"],["언년아 · model_0002","new"]]);
    assert.equal(b.elements.get("project-select").value,"new");
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
    assert.match(b.elements.get("stages").children[5].className, /current/);
    assert.equal(b.elements.get("progress").hidden, true); assert.equal(b.elements.get("progress-percent").textContent, "");
    assert.equal(b.elements.get("progress-label").textContent, "전체 절차 13 / 19단계 완료");
    assert.equal(b.elements.get("elapsed").textContent, "이 작업 경과 30초 · 전체 경과 1시간");
    assert.match(b.elements.get("eta").textContent, /측정한 뒤/);
  });
  await test("nine visible groups separate positive generation and hard-negative preparation while keeping all nineteen workflow stages", async () => {
    const workflow = ["frontend", "data_download", "data_review", "data_connect", "references", "references_review",
      "pilot", "pilot_review", "positives", "hard_phrases", "hard_generation", "hard_review", "hard_connect",
      "features", "training", "package", "development", "evaluation", "results"];
    const groupIndexes = [0,0,0,0,1,1,2,2,3,4,4,4,4,5,6,6,7,7,8];
    const labels = ["자료 준비", "합성 목소리", "호출어 확인", "호출어 생성", "유사어 준비", "증강·특징 추출", "모델 학습", "성능 평가", "결과·시험"];
    const project = { id: "bora", wakeword: "보라야", status: "running", stage: "frontend",
      progress: { done: 0, total: 19, unit: "완료 단계" }, settings: { frozen_policy: "unchanged" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    for (const [index, id] of workflow.entries()) {
      project.stage = id; project.status = /review|phrases/.test(id) ? "waiting_review" : "running";
      project.stages = workflow.map((name, position) => ({ id: name, status: position < index ? "completed" : position === index ? project.status : "pending" }));
      project.progress.done = index;
      await b.tick();
      const groups = b.elements.get("stages").children;
      assert.equal(groups.length, 9);
      assert.deepEqual(groups.map(group => group.children[1].textContent.replace(/^✓ /, "")), labels);
      assert.equal(groups.filter(group => /\bcurrent\b/.test(group.className)).length, 1);
      assert.equal(groups[groupIndexes[index]].getAttribute("aria-current"), "step");
      assert.equal(groups.filter(group => /\bdone\b/.test(group.className)).length, groupIndexes[index]);
      assert.equal(b.elements.get("progress-label").textContent, `전체 절차 ${index} / 19단계 완료`);
      assert.equal(b.elements.get("progress-percent").textContent, "");
      assert.deepEqual(project.stages.map(stage => stage.id), workflow);
      assert.equal(project.settings.frozen_policy, "unchanged");
    }
    project.status = "completed"; project.stage = "completed"; project.progress.done = 19;
    project.stages.forEach(stage => { stage.status = "completed"; }); await b.tick();
    const completed = b.elements.get("stages").children;
    assert.equal(completed.filter(group => /\bdone\b/.test(group.className)).length, 9);
    assert.equal(completed.filter(group => /\bcurrent\b/.test(group.className)).length, 0);
    assert.equal(b.calls.some(call => call.options.method === "POST" && call.url !== "/api/model-location"), false);
  });
  await test("generation counters distinguish attempts and file checks from human listening", async () => {
    const project = { id: "bora", status: "running", stage: "positives",
      counts: { tts_generated: 21, technical_passed: 20, reviewed: 0, admitted: 0 } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    const counts = b.elements.get("counts");
    assert.equal(counts.hidden, false);
    assert.deepEqual(counts.children.map(textOf), ["생성 시도 21", "파일 검사 통과 20", "사람 청취 0", "학습 사용 판정 0"]);
    project.counts = [{ id: "positive_generated", value: 22 }]; await b.tick();
    assert.equal(textOf(counts), "생성 음원 22");
    assert.equal(b.calls.some(call => call.options.method === "POST" && call.url !== "/api/model-location"), false);
  });
  await test("legacy generation alias stays in the positive group and unknown stages invent no completion", async () => {
    const project = { id: "bora", status: "running", stage: "generate" };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    let groups = b.elements.get("stages").children;
    assert.equal(groups[3].getAttribute("aria-current"), "step");
    assert.equal(groups[3].children[1].textContent, "호출어 생성");
    assert.equal(groups[4].children[1].textContent, "유사어 준비");
    assert.doesNotMatch(groups[4].className, /\bcurrent\b|\bdone\b/);
    project.stage = "unknown"; await b.tick(); groups = b.elements.get("stages").children;
    assert.equal(groups.length, 9);
    assert.equal(groups.filter(group => /\bcurrent\b|\bdone\b/.test(group.className)).length, 0);
  });
  await test("dependency setup asks for CPU/GPU without mislabeling it as source admission", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_input", stage: "frontend", error: null, input: { kind: "dependency_setup", recommended_backend: "cu128" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("connection").hidden, true);
    assert.equal(b.elements.get("project-error").hidden, true);
    assert.equal(b.elements.get("input-section").hidden, false);
    assert.equal(b.elements.get("input-form").hidden, false);
    assert.equal(b.elements.get("input-value").hidden, true); assert.equal(b.elements.get("input-value").required, false);
    assert.equal(b.elements.get("dependency-backend").hidden, false); assert.equal(b.elements.get("dependency-backend").value, "cu128");
    assert.equal(b.elements.get("input-submit").textContent, "선택한 도구 설치하고 계속");
    await b.submit("input-form"); const request = b.calls.find(call => call.url.endsWith("/input"));
    assert.deepEqual(JSON.parse(request.options.body), { dependency_backend: "cu128" });
  });
  await test("null and missing errors render normally while string and object errors remain visible", async () => {
    const cases = [
      { value: null, visible: false, text: "" },
      { value: undefined, visible: false, text: "" },
      { value: "설치 실패", visible: true, text: "설치 실패" },
      { value: { message: "도구 설치 실패", code: "setup_failed" }, visible: true, text: "도구 설치 실패" },
      { value: { code: "setup_failed" }, visible: true, text: '{"code":"setup_failed"}' },
    ];
    for (const entry of cases) {
      const project = { id: "bora", wakeword: "보라야", status: "waiting_input", stage: "frontend", error: entry.value, input: { kind: "dependency_setup", recommended_backend: "cu128" } };
      const b = browser({ initialState: { projects: [project] } }); await flush();
      assert.equal(b.elements.get("connection").hidden, true, `error=${JSON.stringify(entry.value)}`);
      assert.equal(b.elements.get("project-error").hidden, !entry.visible);
      assert.equal(b.elements.get("project-error").textContent, entry.text);
      assert.equal(b.elements.get("input-section").hidden, false);
      assert.equal(b.elements.get("dependency-backend").value, "cu128");
    }
  });
  await test("a null current stage does not prevent rendering dependency input", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_input", stage: null, error: null, stages: [{ id: "frontend", status: "pending" }], input: { kind: "dependency_setup", recommended_backend: "cu128" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("connection").hidden, true);
    assert.equal(b.elements.get("project-title").textContent, "‘보라야’");
    assert.ok(b.elements.get("stages").children.length > 0);
    assert.equal(b.elements.get("input-section").hidden, false);
    assert.equal(b.elements.get("dependency-backend").value, "cu128");
  });
  await test("fetch failures are connection errors and recover on the next successful poll", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "ready", stage: "data", error: null };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    b.state.fetchError = new Error("network failure"); await b.tick();
    assert.equal(b.elements.get("connection").hidden, false);
    assert.equal(b.elements.get("connection").className, "notice error");
    assert.equal(b.elements.get("connection").getAttribute("role"), "alert");
    assert.match(b.elements.get("connection").textContent, /서버에 연결하지 못했습니다/);
    assert.match(b.elements.get("connection").textContent, /network failure/);
    b.state.fetchError = null; await b.tick();
    assert.equal(b.elements.get("connection").hidden, true);
    assert.equal(b.elements.get("project-title").textContent, "‘보라야’");
  });
  await test("a timed-out state poll keeps the last project and iframe while automatic polling recovers", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_review", stage: "pilot", error: null,
      progress: { done: 2, total: 24, unit: "files" },
      review: { queue_id: "pilot1", url: "/review/?project=bora&queue=pilot1" } };
    const b = browser({ initialState: { projects: [project], active_project_id: "bora" } }); await flush();
    const title = b.elements.get("project-title").textContent;
    const progress = b.elements.get("progress-label").textContent;
    const frame = b.elements.get("review-frame"), source = frame.getAttribute("src"), writes = frame.srcWrites;
    const aborted = new Error("The operation was aborted"); aborted.name = "AbortError";
    b.state.fetchError = aborted; await b.tick(); await b.tick();
    assert.equal(b.elements.get("connection").hidden, false);
    assert.equal(b.elements.get("connection").className, "notice");
    assert.equal(b.elements.get("connection").getAttribute("role"), "status");
    assert.match(b.elements.get("connection").textContent, /진행 상태 확인이 늦어지고 있습니다/);
    assert.match(b.elements.get("connection").textContent, /마지막으로 받은 상태.*자동으로 다시 확인/);
    assert.doesNotMatch(b.elements.get("connection").textContent, /서버에 연결하지 못했습니다|서버를 실행한 창/);
    assert.equal(b.elements.get("project").hidden, false);
    assert.equal(b.elements.get("project-title").textContent, title);
    assert.equal(b.elements.get("progress-label").textContent, progress);
    assert.equal(b.elements.get("review-section").hidden, false);
    assert.equal(frame.getAttribute("src"), source); assert.equal(frame.srcWrites, writes);
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
    b.state.fetchError = null; project.progress.done = 4; await b.tick();
    assert.equal(b.elements.get("connection").hidden, true);
    assert.equal(b.elements.get("project-title").textContent, title);
    assert.notEqual(b.elements.get("progress-label").textContent, progress);
    assert.equal(frame.getAttribute("src"), source); assert.equal(frame.srcWrites, writes);
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
  });
  await test("HTTP 503 progress-read failures preserve the last snapshot, gauge and iframe until polling recovers", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "waiting_review", stage: "pilot", error: null,
      progress: { done: 2, total: 24, unit: "files" },
      review: { queue_id: "pilot1", url: "/review/?project=bora&queue=pilot1" } };
    const b = browser({ initialState: { projects: [project], active_project_id: "bora" } }); await flush();
    const title = b.elements.get("project-title").textContent, label = b.elements.get("progress-label").textContent;
    const gauge = b.elements.get("progress"), percent = b.elements.get("progress-percent").textContent, value = gauge.value;
    const frame = b.elements.get("review-frame"), source = frame.getAttribute("src"), writes = frame.srcWrites;
    b.state.httpError = { status: 503, body: { error: "진행 상태를 읽는 중입니다.", code: "progress_read_pending" } };
    for (let attempt = 0; attempt < 4; attempt++) {
      await b.tick();
      assert.equal(b.elements.get("connection").hidden, false);
      assert.equal(b.elements.get("connection").className, "notice");
      assert.equal(b.elements.get("connection").getAttribute("role"), "status");
      assert.equal(b.elements.get("connection").textContent, "진행 상태 파일을 읽지 못했습니다. 마지막으로 받은 상태를 표시하며 자동으로 다시 확인합니다.");
      assert.equal(b.elements.get("project").hidden, false);
      assert.equal(b.elements.get("project-title").textContent, title);
      assert.equal(b.elements.get("progress-label").textContent, label);
      assert.equal(b.elements.get("progress-percent").textContent, percent);
      assert.equal(gauge.value, value);
      assert.equal(b.elements.get("review-section").hidden, false);
      assert.equal(frame.getAttribute("src"), source); assert.equal(frame.srcWrites, writes);
    }
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
    b.state.httpError = null; project.progress.done = 4; await b.tick();
    assert.equal(b.elements.get("connection").hidden, true);
    assert.notEqual(b.elements.get("progress-label").textContent, label);
    assert.equal(gauge.value, 4 / 24 * 100);
    assert.equal(frame.getAttribute("src"), source); assert.equal(frame.srcWrites, writes);
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
  });
  await test("HTTP permission errors and unrelated failures remain ordinary errors rather than progress-read notices", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "running", stage: "data_download", progress: { done: 10, total: 100 } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    for (const [status, code] of [[403, "forbidden"], [403, "progress_read_pending"], [503, "other_failure"], [503, { value: "progress_read_pending" }]]) {
      b.state.httpError = { status, body: { error: "접근할 수 없습니다.", code } }; await b.tick();
      assert.equal(b.elements.get("connection").hidden, false);
      assert.equal(b.elements.get("connection").className, "notice error");
      assert.equal(b.elements.get("connection").getAttribute("role"), "alert");
      assert.match(b.elements.get("connection").textContent, /접근할 수 없습니다/);
      assert.doesNotMatch(b.elements.get("connection").textContent, /진행 상태 파일을 읽지 못했습니다/);
    }
    b.state.httpError = null; await b.tick();
    assert.equal(b.elements.get("connection").hidden, true);
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
  });
  await test("render failures are screen errors rather than false server connection errors", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "ready", stage: "data", error: null };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    const title = b.elements.get("project-title");
    Object.defineProperty(title, "textContent", { configurable: true, set() { throw new Error("render failure"); } });
    await b.tick();
    assert.equal(b.elements.get("connection").hidden, false);
    assert.match(b.elements.get("connection").textContent, /화면.*(?:표시|그리)/);
    assert.match(b.elements.get("connection").textContent, /render failure/);
    assert.doesNotMatch(b.elements.get("connection").textContent, /서버에 연결하지 못했습니다/);
    Object.defineProperty(title, "textContent", { configurable: true, writable: true, value: "" });
    await b.tick();
    assert.equal(b.elements.get("connection").hidden, true);
    assert.equal(title.textContent, "‘보라야’");
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
  await test("running evaluation consumes structured progress and fetches logs only on request", async () => {
    const project = { id: "bora", wakeword: "보라야", status: "running", stage: { id: "evaluation" }, progress: { done: 80, total: 100, unit: "개 음원", phase: "scoring", label: "성능 평가 · GPU" } };
    const b = browser({ initialState: { projects: [project] }, logResponse: "Evaluate development: 80/100 (CUDA batch=128, CPU rechecked=8)\n" });
    await flush();
    assert.equal(b.elements.get("log-details").open, false);
    assert.equal(b.elements.get("progress").hidden, false); assert.equal(b.elements.get("progress").value, 80);
    assert.equal(b.elements.get("progress-percent").textContent, "80.0%");
    assert.match(b.elements.get("progress-label").textContent, /80 \/ 100/);
    const before = b.calls.filter(call => call.url.endsWith("/log")).length;
    assert.equal(before, 0);
    b.elements.get("log-details").open = true;
    await b.tick();
    assert.equal(b.calls.filter(call => call.url.endsWith("/log")).length, before + 1);
    assert.equal(b.elements.get("connection").hidden, true);
  });
  await test("evaluation validation does not poll logs or invent a score; manual log errors stay isolated", async () => {
    const project = { id: "bora", status: "running", stage: { id: "evaluation" }, progress: { done: null, total: null, phase: "validation", label: "평가 음원 검증 중" } };
    const b = browser({ initialState: { projects: [project] }, logError: true }); await flush();
    assert.equal(b.elements.get("progress-section").hidden, true);
    assert.match(b.elements.get("eta").textContent, /무결성과 형식/);
    assert.equal(b.elements.get("connection").hidden, true);
    await b.tick();
    assert.equal(b.calls.filter(call => call.url.endsWith("/log")).length, 0);
    b.elements.get("log-details").open = true; await b.tick();
    assert.match(b.elements.get("log").textContent, /optional log unavailable/);
    assert.equal(b.elements.get("connection").hidden, true);
  });
  await test("a server evaluation progress record needs no extra closed-log request", async () => {
    const project = { id: "bora", status: "running", stage: { id: "evaluation" }, progress: { done: 100, total: 100, unit: "개 음원", phase: "finalizing" } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("progress").value, 100);
    assert.match(b.elements.get("current-description").textContent, /통계와 결과 파일/);
    assert.equal(b.elements.get("project-status").textContent, "진행 중");
    assert.equal(b.calls.filter(call => call.url.endsWith("/log")).length, 0);
  });
  await test("completed timing stays explicit and separates processing from waits", async () => {
    const project = {id:"bora",wakeword:"보라야",status:"completed",timing:{total_seconds:7200,active_seconds:1200,review_wait_seconds:1800,paused_seconds:600,unclassified_seconds:3600,legacy_estimate:true}};
    const b=browser({initialState:{projects:[project]}});await flush();
    const value=b.elements.get("elapsed").textContent;
    assert.match(value,/전체 2시간/);assert.match(value,/실제 처리 20분/);assert.match(value,/사용자 확인 대기 30분/);assert.match(value,/일시정지 10분/);assert.match(value,/기타 경과 1시간/);
  });
  await test("a compatible update needs an explicit action and hides ordinary resume", async () => {
    const project={id:"bora",status:"paused",migration:{state:"available",reason:"완료 자료를 보존하고 업데이트할 수 있습니다."}};
    const b=browser({initialState:{projects:[project]}});await flush();
    assert.equal(b.elements.get("migration-note").hidden,false);assert.equal(b.elements.get("migrate").hidden,false);assert.equal(b.elements.get("resume").hidden,true);
    assert.equal(b.calls.some(call=>call.url.endsWith("/migrate")),false);await b.click("migrate");await flush();assert.ok(b.calls.some(call=>call.url.endsWith("/migrate")&&call.options.method==="POST"));
  });
  await test("result selection tab opens an independent page and survives polling", async () => {
    const project={id:"bora",status:"completed",results_url:"/results/bora/"};
    const b=browser({initialState:{projects:[project]}});await flush();b.click("analysis-tab");await flush();
    assert.equal(b.elements.get("result-frame").getAttribute("src"),"/analysis/bora/");await b.tick();
    assert.equal(b.elements.get("result-frame").getAttribute("src"),"/analysis/bora/");assert.equal(b.calls.some(call=>call.options.method==="POST"),false);
  });
  await test("all required download groups are visible while hundreds of RIR files expand only on demand", async () => {
    const rirs = Array.from({ length: 270 }, (_, index) => downloadFile(`rir-${index}`, { group: "mit", group_label: "MIT 잔향 자료", label: `room_${index}.wav`, total: 1024,
      ...(index === 0 ? { state: "downloading", done: 512, network_bytes: 256, bytes_per_second: 128 } : {}) }));
    const files = [downloadFile("generic", { label: "일반 비호출 자료", total: 4096 }),
      downloadFile("common", { label: "공통 평가 배경", state: "verifying", done: 2048, total: 2048 }), ...rirs];
    const project = { id: "bora", wakeword: "보라야", status: "running", stage: "data_download", progress: { done: 1, total: 4, unit: "files", files } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.equal(b.elements.get("downloads-section").hidden, false);
    assert.equal(b.elements.get("downloads-count").textContent, "0 / 272개 준비 완료");
    const groups = b.elements.get("downloads-list").children;
    assert.equal(groups.length, 3);
    assert.match(textOf(groups[0]), /일반 비호출 자료.*대기.*0 B \/ 4 KiB/);
    assert.match(textOf(groups[1]), /공통 평가 배경.*검증/);
    assert.match(textOf(groups[2]), /MIT 잔향 자료 · 0 \/ 270개 완료.*다운로드/);
    const details = withClass(groups[2], "download-details")[0], list = withClass(groups[2], "download-files")[0];
    assert.equal(list.children.length, 0);
    details.open = true; details.handlers.toggle();
    assert.equal(list.children.length, 270);
    assert.match(textOf(list.children[0]), /room_0.wav.*다운로드.*512 B \/ 1 KiB.*128 B\/초/);
    assert.match(textOf(list.children[269]), /room_269.wav.*대기/);
    assert.equal(b.elements.get("progress").value, 25);
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
  });
  await test("download rows preserve nodes and disclosure state while quantities and statuses change", async () => {
    const files = [downloadFile("a", { group: "pair", group_label: "자료 묶음", state: "downloading", done: 256, network_bytes: 256, bytes_per_second: 128 }),
      downloadFile("b", { group: "pair", group_label: "자료 묶음" })];
    const project = { id: "bora", status: "running", stage: "data_download", progress: { phase: "download_many", files } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    const group = b.elements.get("downloads-list").children[0], details = withClass(group, "download-details")[0];
    const list = withClass(group, "download-files")[0]; details.open = true; details.handlers.toggle();
    const child = list.children[0], bar = descendants(child).find(node => node.tagName === "progress");
    files[0].done = 768; files[0].network_bytes = 768; files[1].state = "verifying";
    await b.tick(); await b.tick();
    assert.equal(b.elements.get("downloads-list").children[0], group);
    assert.equal(withClass(group, "download-details")[0], details); assert.equal(details.open, true);
    assert.equal(list.children[0], child); assert.equal(bar.value, 75);
    assert.match(textOf(list.children[1]), /검증/);
    files[0].state = "complete"; files[0].done = 1024; files[1].state = "reused"; files[1].done = 1024;
    project.progress.phase = "preprocessing"; await b.tick();
    assert.equal(b.elements.get("downloads-section").hidden, false);
    assert.equal(b.elements.get("downloads-count").textContent, "2 / 2개 준비 완료");
    assert.match(textOf(list.children[0]), /완료/); assert.match(textOf(list.children[1]), /기존 파일/);
    assert.equal(withClass(child, "download-speed")[0].textContent, "");
    b.state.httpError = { status: 503, body: { code: "progress_read_pending", error: "read pending" } }; await b.tick();
    assert.equal(b.elements.get("downloads-list").children[0], group); assert.equal(details.open, true);
    b.state.httpError = null; await b.tick();
    assert.equal(b.elements.get("connection").hidden, true); assert.equal(list.children[0], child);
  });
  await test("cached, retrying, failed and unknown-size downloads never invent network rates or percentages", async () => {
    const files = [downloadFile("cached", { state: "reused", done: 1024, network_bytes: 0, bytes_per_second: 99999 }),
      downloadFile("resumed", { state: "downloading", done: 900, network_bytes: 0, bytes_per_second: 99999 }),
      downloadFile("retry", { state: "retrying", done: 100, network_bytes: 100, bytes_per_second: 50, attempt: 2 }),
      downloadFile("failure", { state: "failed", error: "<img src=x onerror=bad()>" }),
      downloadFile("unknown", { state: "downloading", done: 512, total: null, network_bytes: 512, bytes_per_second: null }),
      downloadFile("malformed", { state: "unrecognized", done: null, total: null, network_bytes: null })];
    const project = { id: "bora", status: "running", stage: "data_download", progress: { files } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    const groups = b.elements.get("downloads-list").children;
    assert.match(textOf(groups[0]), /기존 파일/); assert.match(textOf(groups[2]), /재시도/); assert.match(textOf(groups[3]), /실패/);
    assert.match(textOf(groups[4]), /512 B · 전체 용량 확인 전/); assert.match(textOf(groups[5]), /상태 확인 중/);
    assert.equal(descendants(groups[4]).find(node => node.tagName === "progress").hidden, true);
    assert.equal(descendants(groups[5]).find(node => node.tagName === "progress").hidden, true);
    for (const group of groups) assert.equal(withClass(group, "download-speed")[0].textContent, "");
    assert.match(withClass(groups[3], "download-error")[0].textContent, /<img src=x onerror=bad\(\)>/);
    assert.equal(descendants(b.elements.get("downloads-list")).some(node => ["img", "script"].includes(node.tagName)), false);
    assert.doesNotMatch(textOf(b.elements.get("downloads-list")), /NaN|Infinity|undefined/);
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
  });
  await test("deferred hours and file counts use their own units while mixed groups never sum byte percentages", async () => {
    const files = [downloadFile("hours", { label: "한국어 비호출", unit: "hours", done: 1.5, total: 3, state: "downloading", network_bytes: 100, bytes_per_second: 50 }),
      downloadFile("files", { label: "음악 음원", unit: "files", done: 2, total: 5, state: "downloading", network_bytes: 100, bytes_per_second: 50 }),
      downloadFile("mixed-bytes", { group: "mixed", group_label: "여러 단위", state: "complete", done: 1024 }),
      downloadFile("mixed-hours", { group: "mixed", group_label: "여러 단위", unit: "hours", done: 10, total: 20, state: "downloading" }),
      downloadFile("unknown-hours", { unit: "hours", done: 5, total: null }),
      downloadFile("unknown-unit", { unit: "seconds", done: 2, total: 4 })];
    const project = { id: "bora", status: "running", stage: "data_download", progress: { files } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    const groups = b.elements.get("downloads-list").children;
    assert.equal(withClass(groups[0], "download-amount")[0].textContent, "1.5시간 / 3시간");
    assert.equal(descendants(groups[0]).find(node => node.tagName === "progress").value, 50);
    assert.equal(withClass(groups[1], "download-amount")[0].textContent, "2개 / 5개");
    assert.equal(descendants(groups[1]).find(node => node.tagName === "progress").value, 40);
    assert.match(textOf(groups[2]), /여러 단위 · 1 \/ 2개 완료/);
    assert.equal(withClass(groups[2], "download-amount")[0].textContent, "");
    assert.equal(descendants(groups[2]).find(node => node.tagName === "progress").hidden, true);
    assert.equal(withClass(groups[3], "download-amount")[0].textContent, "5시간 · 전체 분량 확인 전");
    assert.equal(descendants(groups[3]).find(node => node.tagName === "progress").hidden, true);
    assert.equal(withClass(groups[4], "download-amount")[0].textContent, "진행량 확인 전");
    assert.equal(descendants(groups[4]).find(node => node.tagName === "progress").hidden, true);
    for (const group of groups) assert.equal(withClass(group, "download-speed")[0].textContent, "");
    const details = withClass(groups[2], "download-details")[0]; details.open = true; details.handlers.toggle();
    assert.match(textOf(groups[2]), /10시간 \/ 20시간/);
  });
  await test("legacy progress hides the file list and switching models releases old download rows", async () => {
    const first = { id: "bora", status: "running", stage: "data_download", progress: { files: [downloadFile("old")] } };
    const second = { id: "other", status: "running", stage: "data_download", progress: { done: 1, total: 2, unit: "files" } };
    const b = browser({ initialState: { projects: [first, second], active_project_id: "bora" } }); await flush();
    assert.equal(b.elements.get("downloads-section").hidden, false);
    b.elements.get("project-select").value = "other"; b.elements.get("project-select").handlers.change();
    assert.equal(b.elements.get("downloads-section").hidden, true); assert.equal(b.elements.get("downloads-list").children.length, 0);
    assert.equal(b.elements.get("progress").value, 50);
    second.progress.files = [null, { label: "no ID" }]; await b.tick();
    assert.equal(b.elements.get("downloads-section").hidden, true);
    second.progress.files = [downloadFile("new", { label: "새 자료" })]; await b.tick();
    assert.equal(b.elements.get("downloads-list").children.length, 1);
    assert.match(textOf(b.elements.get("downloads-list")), /새 자료/); assert.doesNotMatch(textOf(b.elements.get("downloads-list")), /old/);
  });
  await test("download speed and retry information describe the current operation", async () => {
    const project={id:"bora",status:"running",stage:{id:"data_download"},progress:{done:100,total:1000,unit:"bytes",phase:"download",bytes_per_second:1048576},eta_seconds:90};
    const b=browser({initialState:{projects:[project]}});await flush();assert.match(b.elements.get("eta").textContent,/1 MiB\/초/);assert.match(b.elements.get("eta").textContent,/이 작업 예상 남은 시간/);
    project.progress={done:100,total:1000,unit:"bytes",phase:"retry",attempt:2,retry_limit:3};await b.tick();assert.match(b.elements.get("eta").textContent,/연결 재시도 2 \/ 3/);
  });
  await test("the evaluation RIR archive uses its purpose label without renaming source records", async () => {
    const row = downloadFile("common:rir_archive", { label: "rirs_noises.zip", group: "common_rir",
      group_label: "평가용 공간 울림 · OpenSLR RIR", state: "downloading", done: 256, network_bytes: 256, bytes_per_second: 128 });
    const project = { id: "bora", status: "running", stage: "data_download", progress: { files: [row] } };
    const b = browser({ initialState: { projects: [project] } }); await flush();
    assert.match(textOf(b.elements.get("downloads-list")), /평가용 공간 울림 · OpenSLR RIR/);
    assert.doesNotMatch(textOf(b.elements.get("downloads-list")), /rirs_noises\.zip/);
    assert.equal(row.label, "rirs_noises.zip"); assert.equal(row.id, "common:rir_archive");
    row.state = "complete"; row.done = row.total; await b.tick();
    assert.match(textOf(b.elements.get("downloads-list")), /평가용 공간 울림 · OpenSLR RIR/);
    assert.equal(b.calls.some(call => call.options.method === "POST"), false);
  });
  console.log(`Studio UI: ${tests.length} tests passed`);
  for (const name of tests) console.log("  ✓ " + name);
})().catch(error => { console.error(error); process.exitCode = 1; });
