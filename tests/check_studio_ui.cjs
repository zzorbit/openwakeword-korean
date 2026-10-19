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
    additional_download: { id: "mit_rir", files: 271, size_bytes: null },
    evaluation_policy: { id: "common_development_v4", validation_references: 30, validation_originals: 5000,
      snrs: [20, 15, 10, 5, 0], positive_sessions: 112000, dry_positive_sessions: 80000,
      reverb_originals: 1000, reverb_positive_sessions: 32000, rir_rooms: 20, negative_hours_by_bucket: buckets },
    evaluation_sources, common_evaluation_catalog: catalog, known_download_bytes: 48123456789,
    ...overrides
  };
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
      assert.equal(amount(groups[0], "한국어 비호출 자료"), specs[index][4].toLocaleString("ko-KR") + "시간");
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
    const rir = material(group, "잔향 자료"); assert.match(textOf(rir), /271개/);
    assert.equal(withClass(rir, "file-size").length, 0);
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
