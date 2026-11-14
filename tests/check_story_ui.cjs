"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio_story.js"), "utf8");
const listeningSource = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/listening_player.js"), "utf8");
const textOf = item => [item.textContent, ...item.children.map(textOf)].join(" ");
const historyRows = root => root.querySelectorAll(".story-history-row");
const historyRowLabel = row => row.querySelector(".story-history-line")?.children[0]?.textContent;
const historyRowByLabel = (root, label) => historyRows(root).find(row => historyRowLabel(row) === label);
const hasActiveDownloadSpinner = root => root.querySelectorAll(".story-progress-indicator")
  .some(icon => !icon.hidden && icon.dataset.status === "active");
const commonToolPackages = ["qwen-tts==0.1.1", "onnx>=1.16", "openwakeword==0.6.0"];
const evaluationToolPackages = ["onnxruntime-gpu==1.31.0", "nvidia-cublas==13.8.0.4", "nvidia-cuda-runtime==13.4.92", "nvidia-cudnn-cu13==9.27.0.42"];
const toolSetup = {
  recommended_backend: "cu128",
  packages: {
    cpu: ["torch==2.8.0+cpu", "torchaudio==2.8.0+cpu", ...commonToolPackages],
    cu128: ["torch==2.8.0+cu128", "torchaudio==2.8.0+cu128", ...commonToolPackages, ...evaluationToolPackages],
  },
  evaluation_runtime_packages: evaluationToolPackages,
  download_bytes: { cu128: 3123456789 },
};
function assertToolPreview(installs, device, capacityLabel) {
  assert.equal(installs.hidden, false);
  const text = textOf(installs);
  assert.equal(installs.querySelector("h2")?.textContent, "연산 도구");
  assert.match(text, device);
  assert.match(text, /PyTorch 2\.8\.0/);
  assert.match(text, /TorchAudio/);
  assert.doesNotMatch(text, /공통|GPU 평가|Qwen TTS|ONNX|openWakeWord/i);
  assert.doesNotMatch(text, /==|>=|nvidia-cublas|nvidia-cuda-runtime|nvidia-cudnn/i);
  const row = installs.querySelector(".story-material-row"), amount = row.querySelector("dd");
  assert.deepEqual(amount.children.map(node => [node.tagName, node.textContent]),
    [["strong", "2개"], ["span", capacityLabel]]);
}

// A DOM adapter exercises navigation and real-state gates without audio,
// network, an installed browser, or changes to a stored workflow.
function page(initialProject = null, options = {}) {
  const registry = new Map(), timers = [], frames = [], calls = [], creationOptions = [], viewModes = [];
  let clock = 0;
  const flush = () => new Promise(resolve => setImmediate(resolve));
  function matches(element, selector) {
    if (selector.startsWith("#")) return element.id === selector.slice(1);
    const [tag, className] = selector.split(".");
    return (!tag || element.tagName === tag) && (!className || element.className.split(/\s+/).includes(className));
  }
  class Element {
    constructor(tag = "div") {
      this.tagName = tag; this.children = []; this.parentElement = null; this.className = "";
      this.handlers = {}; this.attributes = {}; this.dataset = {}; this.style = { setProperty(name, value) { this[name] = String(value); } };
      this.hidden = false; this.disabled = false; this.textContent = ""; this.value = ""; this.open = false;
      if (tag === "audio") { this.paused = true; this.ended = false; this.playCalls = 0; this.loadCalls = 0; }
      this.classList = {
        contains: name => this.className.split(/\s+/).includes(name),
        add: name => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), name])].join(" "); },
        remove: (...names) => { this.className = this.className.split(/\s+/).filter(value => !names.includes(value)).join(" "); },
        toggle: (name, force) => { if (force) this.classList.add(name); else this.classList.remove(name); },
      };
    }
    set id(value) { this._id = value; registry.set(value, this); }
    get id() { return this._id || ""; }
    get firstElementChild() { return this.children[0] || null; }
    contains(target) { return this === target || this.children.some(child => child.contains(target)); }
    append(...children) {
      for (const item of children) {
        const child = typeof item === "string" ? Object.assign(new Element("#text"), { textContent: item }) : item;
        if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child);
        child.parentElement = this; this.children.push(child);
      }
    }
    prepend(child) { this.append(child); this.children = [child, ...this.children.filter(item => item !== child)]; }
    insertBefore(child, next) {
      this.append(child);
      this.children.pop();
      this.children.splice(this.children.indexOf(next), 0, child);
    }
    replaceChildren(...children) { for (const child of this.children) child.parentElement = null; this.children = []; this.append(...children); }
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this); this.parentElement = null; }
    querySelectorAll(selector) { return this.children.flatMap(child => [...(matches(child, selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    matches(selector) { return selector === ":focus-visible" ? this.focusVisible === true : matches(this, selector); }
    addEventListener(name, fn) { (this.handlers[name] ||= []).push(fn); }
    getAnimations() { return options.animations?.(this) || []; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    getAttribute(name) { return this.attributes[name] ?? null; }
    focus() { this.focused = true; }
    showModal() { this.open = true; }
    close() { this.open = false; this.fire("close"); }
    pause() { const wasPlaying = this.paused === false; this.paused = true; if (wasPlaying) this.fire("pause"); }
    load() { this.loadCalls = (this.loadCalls || 0) + 1; this.paused = true; this.ended = false; }
    play() {
      this.playCalls++;
      const perform = () => { this.paused = false; this.ended = false; return this.fire("play"); };
      return options.playMedia ? options.playMedia(this, perform) : perform();
    }
    async fire(name, event = {}) { await Promise.all((this.handlers[name] || []).map(fn => fn(event))); await flush(); }
  }
  const body = new Element("body"), legacy = new Element("main"); legacy.className = "dashboard-shell"; body.append(legacy);
  const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio.html"), "utf8");
  for (const match of html.matchAll(/id="([^"]+)"/g)) { const item = new Element(); item.id = match[1]; legacy.append(item); }
  body.append(registry.get("studio-notices"));
  for (const id of ["connection", "message", "project-error", "review-error", "studio-notice-action"])
    registry.get("studio-notices").append(registry.get(id));
  const currentCard = new Element(); currentCard.className = "current-card"; currentCard.append(registry.get("current-title")); legacy.append(currentCard);
  registry.get("migration-note").append(registry.get("migration-reason"), registry.get("migrate"));
  currentCard.append(registry.get("migration-note"));
  const storage = new Element(); storage.className = "storage-location"; storage.append(registry.get("storage-directory")); legacy.append(storage);
  const get = id => {
    const found = registry.get(id);
    for (let parent = found; parent; parent = parent.parentElement) if (parent === body) return found;
    return null;
  };
  get("wakeword").value = "보라야"; get("create").disabled = false;
  const documentHandlers = {};
  const document = { body, hidden: false, createElement: tag => new Element(tag), getElementById: get, querySelector: selector => body.querySelector(selector), title: "", addEventListener: (name, fn) => { (documentHandlers[name] ||= []).push(fn); } };
  if (options.fonts) document.fonts = options.fonts;
  const state = { project: initialProject, recipes: [], selectedPreset: "check", recipeReady: true, busy: false,
    dependencySetup: options.dependencySetup || { recommended_backend: "cpu", packages: { cpu: [], cu128: [] } } };
  let renderedPhraseInput;
  const windowHandlers = {};
  const chapterRequests = [];
  const context = vm.createContext({ document, console, URL, location: { origin: "http://studio.local", href: "http://studio.local/" }, innerWidth: options.width || 1200, innerHeight: options.height || 900, matchMedia: query => ({ matches: options.matchMedia?.(query) ?? false }),
    setTimeout: (callback, delay) => { const timer = { callback, delay, at: clock + delay }; timers.push(timer); return timer; },
    clearTimeout: timer => { if (timer) timer.canceled = true; } });
  context.window = context; context.addEventListener = (name, fn) => { (windowHandlers[name] ||= []).push(fn); };
  if (options.paintFrames === "manual") context.requestAnimationFrame = callback => frames.push(callback);
  else if (options.paintFrames) context.requestAnimationFrame = callback => setImmediate(callback);
  context.getComputedStyle = element => ({ animationDuration: "0s", animationDelay: "0s", transitionDuration: "0s", transitionDelay: "0s", ...options.getComputedStyle?.(element) });
  for (const id of ["reference-preview-frame", "review-frame", "result-frame"]) {
    const frame = get(id);
    frame.contentWindow = { postMessage(data, origin) {
      if (data.type === "koww:reference-status" && data.request_layout && state.project?.reference_preview?.candidate_selection && !options.deferReferenceLayout) {
        const send = type => { for (const fn of windowHandlers.message || []) fn({ origin, source: frame.contentWindow,
          data: { type, project: data.project, mode: "candidate_pool_v2", height: 410 } }); };
        setImmediate(() => { send("koww:reference-layout"); send("koww:reference-height"); });
      }
      if (data.type !== "koww:chapter-request") return;
      chapterRequests.push({ frame, data, origin });
      if (options.chapterReady === false) return;
      setImmediate(() => { for (const fn of windowHandlers.message || []) fn({ origin, source: frame.contentWindow,
        data: { ...data, type: "koww:chapter-ready", status: "ready" } }); });
    } };
  }
  if (options.compactListening) vm.runInContext(listeningSource, context);
  vm.runInContext(source, context);
  let noticeAction;
  const actionButton = get("studio-notice-action"); actionButton.hidden = true;
  actionButton.addEventListener("click", () => noticeAction?.run());
  const ui = context.KowwStory.create({
    state: () => state,
    materials: recipe => options.materials?.(recipe) || ({ training: [], evaluation: [] }),
    formatBytes: value => `${value}B`,
    wakewordError: value => options.wakewordError?.(value) || (String(value).trim() ? "" : "호출어를 입력하세요."),
    mediaUrl: value => typeof value === "string" && value.startsWith("/") && !value.startsWith("//") ? value : "",
    create: input => { state.busy = true; calls.push("create"); creationOptions.push(JSON.parse(JSON.stringify(input || {}))); return options.create?.(state); },
    action: (action, payload, projectId) => { state.busy = true; calls.push(action); return options.action?.(action, state, payload && JSON.parse(JSON.stringify(payload)), projectId); },
    input: (choice, projectId) => { calls.push("input"); return options.input?.(state, choice && JSON.parse(JSON.stringify(choice)), projectId); }, view: (view, mode) => {
      calls.push("view:" + view); viewModes.push({ view, mode });
      const project = state.project;
      const frameSource = (id, url) => {
        const frame = get(id); if (frame.getAttribute("src") === url && frame.fakeLoadedUrl === url) return;
        frame.setAttribute("src", url);
        setImmediate(() => { if (get("story-panel")?.inert && frame.fakeLoadedUrl !== url) { frame.fakeLoadedUrl = url; frame.fire("load"); } });
      };
      if (view === "review" && mode === "reference" && project?.reference_preview?.url) frameSource("reference-preview-frame", project.reference_preview.url);
      if (view === "review" && mode === "required" && project?.review?.url) frameSource("review-frame", project.review.url);
      if (["results", "mic"].includes(view) && (project?.results?.url || project?.results_url)) frameSource("result-frame", view === "mic" ? project.results?.mic_url || `/mic/${project.id}/` : project.results?.url || project.results_url);
      const phraseInput = project?.status === "waiting_input" && project.input?.kind === "hard_phrases" ? project.input : null;
      if (phraseInput && phraseInput !== renderedPhraseInput && options.mountPhraseFields !== false) {
        const values = phraseInput.phrases || phraseInput.suggestions || [];
        const fields = (values.length ? values : [""]).map(value => Object.assign(new Element("input"), { value }));
        get("hard-phrase-fields").replaceChildren(...fields);
        renderedPhraseInput = phraseInput;
      }
      options.view?.(view, mode, state);
    },
    pause: frame => calls.push("pause:" + frame.id), preview: kind => calls.push("preview:" + kind),
    refresh: () => { calls.push("refresh"); return options.refresh?.(state); },
    prepareRecipes: () => options.prepareRecipes?.(state),
    logs: () => calls.push("logs"), installLog: options.installLog,
    message: (text, error, extra) => {
      calls.push("message:" + text);
      noticeAction = extra?.action;
      actionButton.hidden = !noticeAction;
    },
    noticeActionVisible: action => action === noticeAction && !actionButton.hidden,
    clearNoticeAction: action => { if (action === noticeAction) { noticeAction = null; actionButton.hidden = true; } },
    reviewNotice: text => { const notice = get("review-error"); notice.textContent = text; notice.hidden = !text; },
    projectNotice: text => { const notice = get("project-error"); notice.textContent = text; notice.hidden = !text; },
    projectErrorText: error => typeof error === "string" ? error : error?.message || "",
  });
  const shell = get("story-shell"), actions = shell.querySelector(".story-actions");
  const next = actions.children[2], operation = actions.children[1], back = actions.children[0];
  async function advanceFrame() {
    const pending = frames.splice(0);
    for (const callback of pending) callback(clock);
    await flush();
    return pending.length > 0;
  }
  function advanceTimer() { const timer = timers.filter(timer => !timer.canceled && !timer.ran).sort((a, b) => a.at - b.at)[0]; if (!timer) return false; clock = Math.max(clock, timer.at); timer.ran = true; timer.callback(); return true; }
  async function settle() { let advanced = false; await flush(); while (advanceTimer()) { advanced = true; await flush(); } return advanced; }
  ui.render();
  return { api: context.KowwStory, ui, shell, get, state, calls, timers, chapterRequests, creationOptions, viewModes, next, operation, back, settle, advanceTimer, advanceFrame, flush, now: () => clock, elapse: value => { clock += value; }, windowEvent: async (name, event) => { for (const fn of windowHandlers[name] || []) fn(event); await flush(); }, documentEvent: async (name, event) => { for (const fn of documentHandlers[name] || []) fn(event); await flush(); }, element: tag => new Element(tag), visibility: async hidden => { document.hidden = hidden; await Promise.all((documentHandlers.visibilitychange || []).map(fn => fn())); }, ticks: shell.querySelectorAll(".story-index-step") };
}

async function main() {
  let passed = 0;
  async function advanceSetup(p, ...steps) {
    for (const step of steps) {
      await p.next.fire("click"); await p.settle();
      assert.equal(p.shell.dataset.step, step);
    }
  }
  async function test(name, fn) {
    if (process.env.KOWW_STORY_TEST_FILTER && !new RegExp(process.env.KOWW_STORY_TEST_FILTER).test(name)) return;
    await fn(); console.log("  ✓ " + name); passed++;
  }
  await test("hover magnification follows the pointer while the current chapter stays semantic only", async () => {
    const p = page({ id: "index-hover", wakeword: "보라야", status: "completed", results: { url: "/results/" } });
    const visible = p.ticks.filter(tick => !tick.hidden), active = visible.find(tick => tick.getAttribute("aria-current") === "step");
    const distances = () => visible.map(tick => tick.dataset.distance);
    assert.ok(active);assert.ok(distances().every(value => value === "3"), "resting index bars all use the small baseline");
    await visible[3].fire("pointerenter", { pointerType: "mouse" });
    assert.deepEqual(distances().slice(0, 7), ["3", "2", "1", "0", "1", "2", "3"]);
    assert.equal(active.getAttribute("aria-current"), "step", "hover does not change the semantic current chapter");
    assert.equal(visible[3].getAttribute("aria-current"), "false");
    p.ui.render();
    assert.equal(visible[3].dataset.distance, "0", "background polling preserves the hovered center");
    await visible[3].fire("pointerleave");assert.ok(distances().every(value => value === "3"));
    await visible[4].fire("pointerenter", { pointerType: "mouse" });
    await visible[4].parentElement.fire("pointerleave");assert.ok(distances().every(value => value === "3"));
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("hover magnification supports visible keyboard focus without sticking after mouse focus or blur", async () => {
    const p = page({ id: "index-focus", wakeword: "보라야", status: "completed", results: { url: "/results/" } });
    const visible=p.ticks.filter(tick=>!tick.hidden), target=visible[2];
    target.focusVisible=true;await target.fire("focus");
    assert.equal(target.dataset.distance,"0");assert.equal(visible[1].dataset.distance,"1");assert.equal(visible[0].dataset.distance,"2");
    await visible[5].fire("pointerenter",{pointerType:"mouse"});assert.equal(visible[5].dataset.distance,"0");
    await visible[5].fire("pointerleave");assert.equal(target.dataset.distance,"0","visible keyboard focus remains usable after a pointer leaves");
    await target.fire("blur");target.focusVisible=false;assert.ok(visible.every(tick=>tick.dataset.distance==="3"));
    await target.fire("pointerenter",{pointerType:"mouse"});await target.fire("pointerdown");await target.fire("focus");await target.fire("pointerleave");
    assert.ok(visible.every(tick=>tick.dataset.distance==="3"),"mouse focus alone never leaves an index bar expanded");
    await target.fire("pointerenter",{pointerType:"touch"});assert.ok(visible.every(tick=>tick.dataset.distance==="3"),"touch does not create sticky hover state");
  });
  await test("hover magnification clears for navigation, disabled state, hidden controls and model exchange", async () => {
    const p=page({id:"index-reset",wakeword:"보라야",status:"completed",results:{url:"/results/"}});
    const visible=()=>p.ticks.filter(tick=>!tick.hidden), baseline=()=>assert.ok(visible().every(tick=>tick.dataset.distance==="3"));
    await visible()[1].fire("pointerenter",{pointerType:"mouse"});assert.equal(visible()[1].dataset.distance,"0");
    await visible()[1].fire("click");baseline();
    await visible()[4].fire("pointerenter",{pointerType:"mouse"});baseline();
    await p.settle();assert.equal(p.shell.dataset.step,"tools");baseline();
    await visible()[3].fire("pointerenter",{pointerType:"mouse"});p.state.busy=true;p.ui.render();baseline();
    await visible()[3].fire("pointerenter",{pointerType:"mouse"});baseline();
    p.state.busy=false;p.ui.render();await visible()[3].fire("pointerenter",{pointerType:"mouse"});
    await p.windowEvent("blur",{});baseline();
    await visible()[3].fire("pointerenter",{pointerType:"mouse"});await p.visibility(true);baseline();
    await p.visibility(false);await visible()[3].fire("pointerenter",{pointerType:"mouse"});
    p.state.project={id:"earlier-index",wakeword:"다른이름",status:"running",stage_id:"references",progress:{}};p.ui.render();baseline();
    const unavailable=p.ticks.find(tick=>tick.hidden);assert.ok(unavailable);await unavailable.fire("pointerenter",{pointerType:"mouse"});baseline();
    assert.equal(unavailable.disabled,true);
  });
  await test("backend current stage determines unlocked frontier; pending later stages unlock nothing", () => {
    const p = page();
    assert.equal(p.api.frontier({ status: "running", stage_id: "frontend" }), 4);
    assert.equal(p.api.stageStep({ status: "running", stage_id: "frontend" }), "downloads");
    assert.equal(p.api.stageStep({ status: "waiting_input", stage_id: "frontend", dependency_setup_selected: true,
      input: { kind: "dependency_setup" } }), "tools");
    assert.equal(p.api.frontier({ status: "running", stage_id: "data_download" }), 4);
    assert.equal(p.api.frontier({ status: "running", stage_id: "references", stages: [{ id: "evaluation", status: "pending" }] }), 5);
    assert.equal(p.api.frontier({ status: "waiting_review", stage: { id: "hard_review" } }), 11);
    assert.equal(p.api.frontier({ status: "running", stage_id: "unknown" }), 4);
    assert.equal(p.api.frontier({ status: "completed", results: { url: "/results/" } }), 15);
    assert.equal(p.api.frontier({ status: "completed", results: { url: "/results/", mic_url: "/microphone/" } }), 16);
  });
  await test("setup index keeps reached screens when returning to the first screen", async () => {
    const p = page(), visible = () => p.ticks.filter(tick => !tick.hidden).length;
    assert.equal(p.ticks[0].parentElement.hidden, true);
    await p.next.fire("click"); await p.settle();
    assert.equal(visible(), 2); assert.equal(p.ticks[2].hidden, true);
    await p.ticks[0].fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(p.ticks[0].parentElement.hidden, false);
    assert.equal(visible(), 2); assert.equal(p.ticks[0].getAttribute("aria-current"), "step");
    await p.next.fire("click"); await p.settle();
    await p.next.fire("click"); await p.settle();
    assert.equal(visible(), 3);
    await p.ticks[0].fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(visible(), 3);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("saved progress stays indexed while reviewing earlier screens and grows with actual backend progress", async () => {
    const project = { id: "model", wakeword: "보라야", status: "waiting_review", stage_id: "pilot_review", progress: {} };
    const p = page(project), visible = () => p.ticks.filter(tick => !tick.hidden).length;
    assert.equal(p.shell.dataset.step, "pronunciation"); assert.equal(visible(), 8);
    assert.ok(p.ticks.slice(0, 8).every(tick => tick.dataset.distance === "3"));
    assert.equal(p.next.disabled, true); assert.equal(p.ticks[8].hidden, true); assert.equal(p.ticks[8].disabled, true);
    await p.ticks[8].fire("click"); assert.equal(p.timers.length, 0);
    await p.ticks[0].fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(p.ticks[0].parentElement.hidden, false);
    assert.equal(visible(), 8); assert.equal(p.ticks[0].getAttribute("aria-current"), "step");
    assert.equal(p.ticks[7].getAttribute("aria-current"), "false");
    assert.ok(p.ticks.slice(0, 8).every(tick => tick.dataset.distance === "3"));
    project.stage_id = "hard_review"; p.ui.render();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(visible(), 12);
    assert.equal(p.ticks[11].hidden, false); assert.equal(p.ticks[12].hidden, true);
    const timerCount = p.timers.length;
    await p.ticks[12].fire("click"); assert.equal(p.timers.length, timerCount);
    assert.equal(p.shell.dataset.step, "entry");
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("index extent resets to the selected model and a fresh setup without leaking later progress", async () => {
    const p = page({ id: "finished", wakeword: "보라야", status: "completed", results: { url: "/results/" } });
    assert.equal(p.ticks.filter(tick => !tick.hidden).length, 16);
    p.state.project = { id: "earlier", wakeword: "다른이름", status: "running", stage_id: "references", progress: {} };
    p.ui.render();
    assert.equal(p.shell.dataset.step, "voices"); assert.equal(p.ticks.filter(tick => !tick.hidden).length, 6);
    assert.equal(p.ticks[6].hidden, true);
    p.state.project = null; p.ui.render();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(p.ticks[0].parentElement.hidden, true);
    await p.next.fire("click"); await p.settle();
    assert.equal(p.ticks.filter(tick => !tick.hidden).length, 2); assert.equal(p.ticks[2].hidden, true);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("chapter definitions keep tool preparation before downloads and voice listening after voice preparation", async () => {
    const p = page(), { definitions } = p.api;
    assert.equal(definitions.length, 19);
    assert.ok(definitions.every(([, title, subtitle]) => title && !title.endsWith(".") && subtitle.endsWith(".")));
    assert.equal(p.get("story-title").textContent, "호출어 입력");
    assert.equal(p.get("story-title").getAttribute("aria-describedby"), "story-subtitle");
    assert.equal(p.get("story-subtitle").textContent, "‘보라야’와 같은 한국어 호출어를 입력해 주세요.");
    await advanceSetup(p, "tools", "scale");
    assert.equal(p.get("story-title").textContent, "학습 규모 선택");
    assert.equal(p.get("story-subtitle").textContent, "사용 목적과 환경에 맞는 학습 규모를 선택하세요.");
    assert.deepEqual(Array.from(definitions.slice(0, 5), ([id]) => id), ["entry", "tools", "scale", "materials", "downloads"]);
    assert.equal(definitions[1][1], "학습 도구 준비");
    const voices = definitions.findIndex(([id]) => id === "voices");
    assert.equal(definitions[voices + 1][0], "voice-listening"); assert.equal(definitions[voices + 2][0], "call-pilot-generation");
  });
  await test("history pagination fits tall narrow, short and desktop column layouts without shrinking text", () => {
    const { pageCapacity } = page().api;
    const fits = (items, available, height, columns, gap) => {
      const capacity = pageCapacity(items, available, height, columns, gap);
      const rows = Math.ceil(Math.min(items, capacity) / columns);
      const used = rows * height + Math.max(0, rows - 1) * gap + (capacity < items ? 44 : 0);
      assert.ok(used <= available, `${used} exceeds ${available}`); return capacity;
    };
    assert.equal(fits(9, 734, 128, 1, 12), 5, "319×950 single-column history paginates");
    assert.equal(fits(9, 330, 80, 1, 12), 3, "600×540 short history paginates");
    assert.equal(fits(9, 600, 80, 2, 4), 9, "1280×800 desktop keeps the complete two-column list");
    assert.equal(fits(20, 330, 44, 1, 10), 5, "phrase fields reserve pager space");
    assert.equal(fits(6, 734, 44, 1, 10), 6, "tall narrow phrase fields all fit when appropriate");
  });
  await test("the complete material list keeps capacity labels and remains read-only until confirmation", async () => {
    const materials = {
      training: [
        { name: "호출어 음성", amount: "1,000개", downloadLabel: "생성 예정", sources: [{ name: "Qwen TTS", downloadReuse: "생성 예정" }] },
        { name: "일반 비호출 자료", amount: "2,000시간", downloadLabel: "다운로드 1KiB", sources: [{ name: "고정 자료", downloadBytes: 1024 }] },
        { name: "한국어 비호출 자료", amount: "1시간", downloadLabel: "다운로드 약 2KiB", sources: [{ name: "자동 선정", downloadBytes: 2048, downloadEstimated: true }] },
        { name: "소음 증강 음원", amount: "원본 재사용", downloadLabel: "비호출 자료 재사용", sources: [{ name: "학습 음원", downloadReuse: "비호출 자료 재사용" }] },
        { name: "추가 자료", amount: "확인 후 결정", sources: [{ name: "추가 원본" }] },
        { name: "잔향 자료", amount: "270개", downloadLabel: "다운로드 3KiB", sources: [{ name: "MIT RIR", downloadBytes: 3072 }] },
      ],
      evaluation: Array.from({ length: 5 }, (_, index) => ({ name: `평가 자료 ${index + 1}`, amount: "2,000개", downloadLabel: "생성 예정", sources: [{ name: "Qwen TTS", downloadReuse: "생성 예정" }] })),
      downloadTotals: { training: { bytes: 6144, unknown: true, label: "약 6KiB + 미정" }, evaluation: { bytes: 0, unknown: false, label: "별도 다운로드 없음" } },
    };
    const p = page(null, { materials: () => materials, dependencySetup: toolSetup, create: state => {
      state.project = { id: "capacity-model", wakeword: "보라야", preset: "check", status: "ready", stage_id: "frontend", progress: {},
        dependency_setup_selected: true, navigation_checkpoint: { policy: "explicit_story_v1", step: "downloads", state: "awaiting_start" } };
      state.busy = false;
    } });
    p.state.recipes = [{ id: "check" }];
    await advanceSetup(p, "tools");
    assert.equal(p.get("story-tool-picker").hidden, false);
    assert.equal(p.get("story-tool-cu128").checked, true);
    await advanceSetup(p, "scale", "materials");
    assertToolPreview(p.get("story-tool-materials"), /NVIDIA GPU\s*·\s*CUDA 12\.8/, "다운로드 약 3123456789B");
    const columns = p.shell.querySelector(".story-material-columns"), training = () => columns.children[0];
    assert.deepEqual(p.shell.querySelectorAll(".story-material-total").map(node => node.textContent), ["약 6KiB + 미정", "별도 다운로드 없음"]);
    assert.equal(p.shell.querySelector(".story-material-total").getAttribute("aria-label"), "학습용 자료 합산 다운로드 용량: 약 6KiB + 미정");
    assert.equal(p.shell.querySelector(".story-material-note"), null, "the footer does not repeat the capacity explanation");
    const capacities = () => training().querySelectorAll(".story-material-capacity");
    training().querySelectorAll(".story-material-row").forEach((row, index) => {
      const title = row.querySelector("dt"), amount = row.querySelector("dd"), item = materials.training[index];
      assert.deepEqual(title.children.map(node => node.textContent), [item.name, item.sources.map(source => source.name).join(" · ")], "each material title keeps its source immediately below it");
      assert.equal(amount.children.length, 2, "the right column contains quantity and capacity only");
      assert.equal(amount.children[0].tagName, "strong"); assert.equal(amount.children[0].textContent, item.amount);
      assert.equal(amount.children[1], row.querySelector(".story-material-capacity"));
    });
    assert.deepEqual(capacities().map(node => node.textContent), ["생성 예정", "다운로드 1KiB", "다운로드 약 2KiB", "비호출 자료 재사용", "다운로드 용량 미확정", "다운로드 3KiB"]);
    assert.equal(capacities()[0].title, "Qwen TTS: 생성 예정"); assert.equal(capacities()[1].title, "고정 자료: 1024B");
    assert.equal(capacities()[2].title, "자동 선정: 약 2048B"); assert.equal(capacities()[3].title, "학습 음원: 비호출 자료 재사용");
    assert.equal(capacities()[4].title, "추가 원본: 용량 미확정");
    assert.equal(columns.children[1].querySelector(".story-material-capacity").textContent, "생성 예정");
    assert.equal(columns.children[0].hidden, false); assert.equal(columns.children[1].hidden, false);
    assert.equal(columns.children[1].querySelectorAll(".story-material-row").length, 5);
    assert.equal(p.shell.querySelectorAll("button").some(button => ["앞 자료", "뒤 자료"].includes(button.getAttribute("aria-label"))), false, "material rows have no pagination controls");
    p.ui.render(); await p.windowEvent("resize"); assert.equal(capacities().length, 6, "render and resize keep every row");
    assert.equal(p.calls.some(call => ["create", "start", "resume", "pause", "input"].includes(call)), false);
    const creation = p.next.fire("click"); await p.settle(); await creation;
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.deepEqual(p.creationOptions, [{ navigationCheckpoints: true, dependencyBackend: "cu128" }]);
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.next.disabled, true);
    const downloadStart = p.operation;
    assert.equal(downloadStart.textContent, "설치·다운로드 시작"); assert.equal(downloadStart.parentElement.hidden, false); assert.equal(downloadStart.disabled, false);
    assert.equal(p.calls.some(call => ["start", "resume", "input"].includes(call)), false);
  });
  await test("small material screens switch groups without slicing rows or starting work", async () => {
    const rows = (prefix, length) => Array.from({ length }, (_, index) => ({ name: `${prefix} ${index + 1}`, amount: "1시간", downloadLabel: index ? "다운로드 약 1GiB" : "다운로드 용량 미확정", description: "직접 준비할 자료", sources: index ? [{ name: "선정 자료" }] : [] }));
    let narrow = true;
    const p = page(null, { width: 319, height: 420, matchMedia: query => query.includes("max-height:700px") || narrow && query.includes("max-width:760px"),
      materials: () => ({ training: rows("학습", 6), evaluation: rows("평가", 5) }) });
    p.state.recipes = [{ id: "check" }];
    await advanceSetup(p, "tools", "scale", "materials");
    const columns = p.shell.querySelector(".story-material-columns"), tabs = p.shell.querySelectorAll(".story-material-tab");
    assert.equal(columns.children[0].hidden, false); assert.equal(columns.children[1].hidden, true);
    assert.equal(columns.children[0].querySelectorAll(".story-material-row").length, 6);
    const first = columns.children[0].querySelector(".story-material-row");
    assert.equal(first.querySelector("dt").children[1].textContent, "직접 준비할 자료", "source-less items retain their description under the title on narrow screens");
    assert.equal(first.querySelector("dd").children.length, 2);
    await tabs[1].fire("click");
    assert.equal(columns.children[0].hidden, true); assert.equal(columns.children[1].hidden, false);
    assert.equal(columns.children[1].querySelectorAll(".story-material-row").length, 5);
    assert.equal(tabs[1].getAttribute("aria-selected"), "true");
    await tabs[1].fire("keydown", { key: "Home", preventDefault() {} }); await p.windowEvent("resize");
    assert.equal(columns.children[0].hidden, false); assert.equal(columns.children[0].querySelectorAll(".story-material-row").length, 6);
    narrow = false; await p.windowEvent("resize");
    assert.ok(columns.children.every(section => !section.hidden), "both columns remain visible in a wide, short window");
    assert.equal(p.calls.some(call => ["create", "start", "resume", "pause", "input"].includes(call)), false);
  });
  await test("new setup advances locally and creates only once after material confirmation", async () => {
    const p = page();
    assert.equal(p.shell.dataset.step, "entry");
    assert.equal(p.ticks[0].parentElement.hidden, true);
    await p.next.fire("click");
    assert.equal(p.calls.includes("create"), false);
    assert.equal(p.timers[0].delay, 800);
    await p.next.fire("click");
    assert.equal(p.timers.length, 1, "transition blocks a second click");
    await p.settle(); assert.equal(p.shell.dataset.step, "tools");
    const cpu = p.get("story-tool-cpu"), gpu = p.get("story-tool-cu128");
    assert.equal(cpu.checked, true);
    cpu.checked = false; gpu.checked = true;
    await gpu.fire("change", { target: gpu });
    await advanceSetup(p, "scale");
    await p.back.fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "tools"); assert.equal(gpu.checked, true);
    await advanceSetup(p, "scale", "materials");
    const creation = p.next.fire("click"); await p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.deepEqual(p.creationOptions, [{ navigationCheckpoints: true, dependencyBackend: "cu128" }]);
    assert.equal(p.calls.includes("start"), false, "model creation does not start downloads");
    assert.equal(p.timers.at(-1).delay, 800); await p.settle(); await creation;
  });
  await test("saved setup confirms only the final CPU/GPU choice each time and installs only when started", async () => {
    const project = { id: "reselect-tools", status: "ready", stage_id: "frontend", preset: "check", dependency_backend: "cu128",
      dependency_backend_editable: true, navigation_checkpoint: { policy: "explicit_story_v1", step: "downloads", state: "awaiting_start" } };
    const sent = [], p = page(project, { dependencySetup: toolSetup, input: (_state, choice, id) => {
      assert.equal(id, project.id); sent.push(choice);
      project.dependency_backend = choice.dependency_backend;
      project.dependency_install_pending = true;
      project.status = "paused";
      p.ui.render();
      return true;
    }, action: (action, _state, payload, id) => {
      assert.equal(action, "start"); assert.equal(id, project.id);
      assert.deepEqual(payload, { expected_dependency_backend: "cu128" });
    } });
    p.state.recipes = [{ id: "check" }];
    const change = p.shell.querySelectorAll("button").find(button => button.textContent === "연산 장치 변경");
    const cpu = p.get("story-tool-cpu"), gpu = p.get("story-tool-cu128");
    async function choose(radio) { cpu.checked = radio === cpu; gpu.checked = radio === gpu; await radio.fire("change"); }
    assert.match(textOf(change.parentElement), /NVIDIA GPU/);
    await change.fire("click"); await p.settle();
    assert.equal(p.get("story-tool-picker").disabled, false);
    for (const radio of [cpu, gpu, cpu]) { await choose(radio); p.ui.render(); assert.equal(radio.checked, true); }
    assert.deepEqual(sent, [], "changing radio choices does not write every intermediate choice");
    await advanceSetup(p, "scale", "materials", "downloads");
    assert.match(textOf(change.parentElement), /CPU/);
    await change.fire("click"); await p.settle();
    for (const radio of [gpu, cpu, gpu]) { await choose(radio); p.ui.render(); assert.equal(radio.checked, true); }
    await p.ticks.find(tick => tick.dataset.step === "downloads").fire("click"); await p.settle();
    assert.match(textOf(change.parentElement), /NVIDIA GPU/);
    assert.deepEqual(sent, [{ dependency_backend: "cpu", defer_dependency_install: true },
      { dependency_backend: "cu128", defer_dependency_install: true }]);
    assert.equal(p.calls.some(call => ["start", "resume"].includes(call)), false);
    await p.operation.fire("click");
    assert.equal(p.calls.filter(call => call === "start").length, 1);
  });
  await test("evaluation resumes only after the last chosen device is stored and confirmed", async () => {
    for (const confirm of [false, true]) {
      const project = { id: "reselect-evaluation", status: "error", stage_id: "evaluation", error: "GPU 오류",
        evaluation_backend: "cuda", evaluation_backend_editable: true };
      let save, sent;
      const p = page(project, { input: (_state, choice, id) => {
        assert.equal(id, project.id); sent = choice;
        return new Promise(resolve => { save = () => { if (confirm) project.evaluation_backend = choice.evaluation_backend; resolve(true); }; });
      }, action: (action, _state, payload, id) => {
        assert.equal(action, "resume"); assert.equal(id, project.id); assert.equal(project.evaluation_backend, "cpu");
        assert.deepEqual(payload, { expected_evaluation_backend: "cpu" });
      } });
      const cpu = p.get("story-evaluation-cpu"), gpu = p.get("story-evaluation-cuda");
      assert.equal(gpu.checked, true);
      for (const radio of [cpu, gpu, cpu]) {
        cpu.checked = radio === cpu; gpu.checked = radio === gpu;
        await radio.fire("change"); p.ui.render(); assert.equal(radio.checked, true);
      }
      const begin = p.operation.fire("click"); await p.flush();
      assert.deepEqual(sent, { evaluation_backend: "cpu" });
      await p.operation.fire("click");
      assert.equal(p.calls.filter(call => call === "input").length, 1);
      assert.equal(p.calls.includes("resume"), false);
      save(); await begin;
      assert.equal(p.calls.filter(call => call === "resume").length, confirm ? 1 : 0);
      if (!confirm) assert.equal(cpu.checked, true, "the unsaved selection can still be retried");
    }
  });
  await test("fast creation runs behind the visible notice and keeps its full 800 ms hold", async () => {
    const p = page(null, { create: state => {
      assert.equal(p.shell.querySelector(".story-loader").hidden, false);
      state.project = { id: "new-model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: {} };
      state.busy = false;
    } });
    await advanceSetup(p, "tools", "scale", "materials");
    const creation = p.next.fire("click");
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.flush();
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    p.ui.render(); assert.equal(p.shell.dataset.step, "downloads", "the destination is prepared behind the notice");
    assert.equal(p.get("story-panel").inert, true, "preparing the destination cannot reveal it early");
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.next.fire("click"); assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.equal(p.timers[3].delay, 800); await p.settle(); await creation;
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
  });
  await test("creation failure restores the material screen and controls after the notice hold without retrying", async () => {
    const p = page(null, { create: state => { state.busy = false; return Promise.reject(new Error("creation unavailable")); } });
    await advanceSetup(p, "tools", "scale", "materials");
    const creation = p.next.fire("click");
    await p.flush();
    p.ui.render(); assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.settle(); await creation;
    assert.equal(p.shell.dataset.step, "materials");
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
    assert.equal(p.get("story-panel").inert, false); assert.equal(p.next.disabled, false);
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.ok(p.calls.includes("message:creation unavailable"));
  });
  await test("empty wakeword never advances or creates and backward transition uses the same minimum", async () => {
    const p = page(); p.get("wakeword").value = "";
    await p.next.fire("click"); assert.equal(p.timers.length, 0); assert.equal(p.get("story-wakeword-error").hidden, false);
    p.get("wakeword").value = "보라야";
    await p.next.fire("click"); await p.settle(); await p.back.fire("click");
    assert.equal(p.timers[1].delay, 800); await p.settle(); assert.equal(p.shell.dataset.step, "entry");
  });
  await test("Next and Previous display the notice before painting a destination chapter", async () => {
    const p = page(null, { paintFrames: "manual" }), loader = p.shell.querySelector(".story-loader");
    for (const [control, source, destination] of [[p.next, "entry", "tools"], [p.back, "tools", "entry"]]) {
      await control.fire("click");
      assert.equal(loader.hidden, false);
      assert.equal(p.shell.dataset.step, source);
      assert.equal(p.get("story-panel").inert, true);
      assert.equal(await p.advanceFrame(), true);
      assert.equal(p.shell.dataset.step, source, "a frame callback alone has not painted the notice");
      assert.equal(await p.advanceFrame(), true);
      assert.equal(p.shell.dataset.step, destination, "the destination can be laid out after the notice has painted");
      assert.equal(loader.hidden, false);
      while (await p.advanceFrame()) {}
      await p.settle();
      assert.equal(loader.hidden, true);
      assert.equal(p.get("story-panel").inert, false);
    }
  });
  function motion(element, duration) {
    let resolve, reject;
    const animation = { effect: { target: element, getComputedTiming: () => ({ endTime: duration }) },
      currentTime: 0, playbackRate: 1, playState: "running", finished: new Promise((yes, no) => { resolve = yes; reject = no; }),
      finish() { this.playState = "finished"; this.currentTime = duration; resolve(); },
      cancel() { this.playState = "idle"; reject(new Error("Animation canceled")); } };
    return animation;
  }
  await test("chapter exit and loading entry start together and keep the source until its fade completes", async () => {
    let leaving;
    const p = page(null, { animations: element => {
      if (element.id !== "story-panel" || !element.classList.contains("is-leaving")) return [];
      leaving = motion(element, 150); return [leaving];
    } });
    const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
    for (const [control, source, destination, cancel] of [[p.next, "entry", "tools", false], [p.back, "tools", "entry", true]]) {
      const start = p.now();
      await control.fire("click");
      assert.equal(p.shell.dataset.step, source);
      assert.equal(panel.inert, true);
      assert.equal(loader.hidden, false);
      assert.equal(loader.classList.contains("is-entering"), true);
      p.elapse(150);
      if (cancel) leaving.cancel(); else leaving.finish();
      await p.flush();
      assert.equal(p.shell.dataset.step, destination);
      assert.equal(panel.classList.contains("is-leaving"), true);
      assert.equal(panel.classList.contains("is-concealed"), cancel);
      await p.settle();
      assert.equal(p.now() - start, 800, "the exit runs within the existing loading hold");
      assert.equal(loader.hidden, true);
      assert.equal(loader.classList.contains("is-entering"), false);
      assert.equal(panel.classList.contains("is-concealed"), false);
      assert.equal(panel.inert, false);
    }
  });
  await test("both directions hold the visible notice for 800 ms and wait for its conceal animation", async () => {
    const phases = [];
    const p = page(null, { getComputedStyle: () => ({ animationDuration: "240ms" }), animations: element => {
      if (!element.classList.contains("is-finishing")) return [];
      const animation = motion(element, 240); phases.push(animation);
      return [animation, motion({}, Infinity)]; // An infinite child effect cannot hold the parent transition.
    } });
    const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
    for (const [control, after] of [[p.next, "tools"], [p.back, "entry"]]) {
      const start = p.now();
      await control.fire("click");
      assert.equal(loader.hidden, false, "the loader opens without waiting for the source panel");
      assert.equal(panel.inert, true); assert.equal(panel.className.includes("is-leaving"), true);
      assert.equal(loader.className.includes("is-finishing"), false);
      const hold = p.timers.filter(timer => !timer.canceled && !timer.ran).at(-1);
      assert.equal(hold.delay, 800);
      assert.equal(hold.at - start, 800);
      p.elapse(799); await p.flush();
      assert.equal(loader.className.includes("is-finishing"), false, "the notice remains fully visible for the whole hold");
      p.advanceTimer(); await p.flush();
      assert.equal(p.now() - start, 800);
      assert.equal(loader.className.includes("is-finishing"), true);
      assert.equal(phases.at(-1).effect.target, loader);
      assert.equal(p.shell.dataset.step, after, "the destination is already laid out behind the loader fade-out");
      assert.equal(panel.inert, true);
      await control.fire("click");
      p.elapse(240); phases.at(-1).finish(); await p.flush();
      assert.equal(p.shell.dataset.step, after); assert.equal(loader.hidden, true);
      assert.equal(loader.className.includes("is-finishing"), false); assert.equal(panel.inert, false);
      assert.equal(p.now() - start, 1040);
    }
  });
  await test("slow animation completion is never replaced by the nominal CSS duration", async () => {
    const phases = [];
    const p = page(null, { getComputedStyle: () => ({ animationDuration: "240ms" }), animations: element => {
      if (!element.classList.contains("is-finishing")) return [];
      const animation = motion(element, 800); phases.push(animation); return [animation];
    } });
    await p.next.fire("click");
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    assert.equal(p.get("story-panel").inert, true);
    assert.equal(p.timers.filter(timer => !timer.canceled && !timer.ran).at(-1).delay, 800);
    p.advanceTimer(); await p.flush(); p.elapse(700);
    assert.equal(p.get("story-panel").inert, true, "the prepared destination stays concealed until exit completion");
    phases[0].finish(); await p.flush();
    assert.equal(p.shell.dataset.step, "tools"); assert.equal(p.now(), 1500);
  });
  await test("no-motion and reduced-motion paths keep the full notice hold without a completion event", async () => {
    const p = page(null, { getComputedStyle: () => ({ animationDuration: "0s", transitionDuration: "0s" }) });
    await p.next.fire("click"); assert.equal(p.get("story-panel").inert, true);
    assert.equal(p.timers.at(-1).delay, 800);
    await p.settle(); assert.equal(p.now(), 800); assert.equal(p.shell.dataset.step, "tools");
  });
  await test("a missing finite completion signal is recovered by finishing the notice exit", async () => {
    const phases = [];
    const p = page(null, { animations: element => {
      if (!element.classList.contains("is-finishing")) return [];
      const animation = motion(element, 200); phases.push(animation); return [animation];
    } });
    await p.next.fire("click");
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    p.advanceTimer(); await p.flush();
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    p.advanceTimer(); await p.flush();
    assert.equal(phases[0].playState, "finished", "watchdog recovery reaches the actual final animation state first");
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
  });
  await test("a slow model response holds the loader and never repeats model creation", async () => {
    let resolveCreate;
    const p = page(null, { create: state => new Promise(resolve => { resolveCreate = () => {
      state.project = { id: "created", status: "ready", stage_id: "data_download" }; state.busy = false; resolve();
    }; }) });
    await advanceSetup(p, "tools", "scale", "materials");
    const creation = p.next.fire("click"); await p.flush(); p.advanceTimer(); await p.flush(); p.elapse(3000);
    assert.equal(p.shell.dataset.step, "materials"); assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.next.fire("click"); assert.equal(p.calls.filter(value => value === "create").length, 1);
    resolveCreate(); await creation;
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.calls.includes("start"), false);
  });
  await test("model changes and page hiding cancel stale animations and queued watchdogs", async () => {
    for (const cancel of ["model", "pagehide"]) {
      const phases = [];
      const p = page(null, { animations: element => {
        if (!element.classList.contains("is-finishing")) return [];
        const animation = motion(element, 200); phases.push(animation); return [animation];
      } });
      await p.next.fire("click"); p.advanceTimer(); await p.flush(); const staleWatchdog = p.timers.at(-1).callback;
      if (cancel === "model") { p.state.project = { id: "different-model", status: "running", stage_id: "features" }; p.ui.render(); }
      else await p.windowEvent("pagehide");
      const expected = p.shell.dataset.step;
      staleWatchdog(); await p.flush();
      assert.equal(phases[0].playState, "running", "an already-queued old watchdog cannot force an old animation to finish");
      phases[0].finish(); await p.flush(); await p.settle();
      assert.equal(p.shell.dataset.step, expected); assert.equal(p.get("story-panel").inert, false);
      assert.equal(p.shell.querySelector(".story-loader").hidden, true);
    }
  });
  await test("a canceled motion restores controls without an unhandled finished rejection", async () => {
    const phases = [];
    const p = page(null, { animations: element => {
      if (!element.classList.contains("is-finishing")) return [];
      const animation = motion(element, 200); phases.push(animation); return [animation];
    } });
    await p.next.fire("click"); p.advanceTimer(); await p.flush(); phases[0].cancel(); await p.flush();
    assert.equal(p.shell.dataset.step, "tools"); assert.equal(p.get("story-panel").inert, false);
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
  });
  await test("page hiding releases a pending model-creation transition without waiting for its network response", async () => {
    const p = page(null, { create: () => new Promise(() => {}) });
    await advanceSetup(p, "tools", "scale", "materials");
    const creation = p.next.fire("click"); await p.flush();
    await p.windowEvent("pagehide"); await creation;
    assert.equal(p.get("story-panel").inert, false); assert.equal(p.shell.querySelector(".story-loader").hidden, true);
    assert.equal(p.calls.filter(value => value === "create").length, 1);
  });
  await test("running downloads never become complete from a display-only percent or timer", async () => {
    const project = { id: "model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: { done: 100, total: 100, files: [] } };
    const p = page(project);
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true);
    assert.equal(p.ticks[5].hidden, true); assert.equal(p.timers.length, 0);
    await p.next.fire("click"); assert.equal(p.calls.includes("pause"), false);
    assert.equal(p.operation.textContent, "일시정지"); await p.operation.fire("click"); assert.deepEqual(p.calls.filter(call => call === "pause"), ["pause"]);
    assert.equal(p.calls.includes("start"), false);
    assert.equal(p.shell.dataset.step, "downloads");
  });
  await test("real backend advance follows once while polling preserves an explicit historical view", async () => {
    const project = { id: "model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: {} };
    const p = page(project);
    project.stage_id = "references"; p.ui.render(); await p.flush(); assert.equal(p.timers.length, 1);
    p.ui.render(); assert.equal(p.timers.length, 1); await p.settle(); assert.equal(p.shell.dataset.step, "voices");
    await p.back.fire("click"); await p.settle(); assert.equal(p.shell.dataset.step, "downloads");
    project.stage_id = "pilot"; p.ui.render(); assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(p.next.textContent, "다음");
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
    p.ui.render(); assert.equal(p.shell.dataset.step, "downloads");
  });
  const checkpointProject = (state = "awaiting_start", status = "ready") => ({
    id: "explicit", wakeword: "보라야", status, stage_id: "frontend", progress: {},
    download_checkpoint: { policy: "explicit_download_v1", state },
    stages: ["frontend", "data_download", "data_review", "data_connect", "references"].map(id => ({ id, status: "pending" })),
  });
  const navigationGroups = [
    ["downloads", ["frontend", "data_download", "data_review", "data_connect"]],
    ["voices", ["references", "references_review"]], ["pronunciation", ["pilot", "pilot_review"]],
    ["positive-generation", ["positives"]], ["phrases", ["hard_phrases"]],
    ["hard-generation", ["hard_generation"]], ["hard-review", ["hard_review", "hard_connect"]],
    ["features", ["features"]], ["training", ["training", "package"]],
    ["evaluation", ["development", "evaluation"]], ["results", ["results"]],
  ];
  const navigationProject = step => {
    const position = navigationGroups.findIndex(([id]) => id === step);
    const complete = navigationGroups.slice(0, position + 1).flatMap(([, stages]) => stages);
    const stages = navigationGroups.flatMap(([, stages]) => stages).map(id => ({ id, status: complete.includes(id) ? "completed" : "pending" }));
    return { id: "navigated", wakeword: "보라야", preset: "check", status: "paused", stage_id: stages[complete.length].id,
      navigation_checkpoint: { policy: "explicit_story_v1", state: "awaiting_next", step }, stages, progress: {},
      reference_counts: { train: 20, validation: 30 }, reference_preview: { count: 50, train: 20, validation: 30, url: "/reference-preview/" } };
  };
  await test("automatic and explicit forward transitions keep one cover across advanced worker snapshots", async () => {
    for (const via of ["poll", "next"]) {
      const project = navigationProject("hard-review");
      if (via === "poll") Object.assign(project, { stage_id: "hard_connect", status: "running",
        navigation_checkpoint: { policy: "explicit_story_v1", step: "hard-review", state: "running" } });
      const beginFeatures = () => {
        Object.assign(project, { stage_id: "features", status: "running",
          navigation_checkpoint: { policy: "explicit_story_v1", step: "features", state: "running" } });
        p.ui.render();
      };
      const p = page(project, { paintFrames: "manual", action: (action, state) => {
        assert.equal(action, "resume");
        state.busy = false;
        beginFeatures();
      } });
      const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
      assert.equal(p.shell.dataset.step, "hard-review");
      if (via === "poll") beginFeatures();
      else await p.next.fire("click");
      await p.advanceFrame(); await p.advanceFrame();
      assert.equal(p.shell.dataset.step, "features");
      while (await p.advanceFrame()) {}
      Object.assign(project, { stage_id: "training", progress: { done: 0, total: 2000, unit: "학습 업데이트" },
        navigation_checkpoint: { policy: "explicit_story_v1", step: "training", state: "running" } });
      p.ui.render();
      p.advanceTimer(); await p.flush();
      while (await p.advanceFrame()) {}
      assert.equal(p.shell.dataset.step, "training");
      assert.equal(loader.hidden, true);
      assert.equal(panel.inert, false, "the new chapter finishes under the original mask");
      assert.equal(p.shell.querySelector(".story-voice-counts").querySelector("progress").max, 2000);
      assert.equal(p.calls.filter(call => call === "resume").length, via === "next" ? 1 : 0);
    }
  });
  await test("every explicit story boundary waits for proven completion and next releases the next real group once", async () => {
    for (const [step] of navigationGroups.slice(0, -1).filter(([step]) => !["voices", "hard-generation"].includes(step))) {
      const project = navigationProject(step), target = navigationGroups[navigationGroups.findIndex(([id]) => id === step) + 1][0];
      const p = page(project, { action: (_action, state) => {
        state.project.navigation_checkpoint = { policy: "explicit_story_v1", state: "running", step: target };
        state.project.status = target === "phrases" ? "waiting_input" : "running";
        if (target === "phrases") state.project.input = { kind: "hard_phrases", suggestions: [] };
        p.ui.render();
        assert.equal(p.shell.dataset.step, step, "a busy release response cannot commit the next screen early");
        assert.equal(p.shell.querySelector(".story-loader").hidden, false);
        state.busy = false; p.ui.render();
      } });
      assert.equal(p.shell.dataset.step, step); assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, false);
      assert.equal(p.operation.hidden, true); p.ui.render(); assert.equal(p.calls.includes("resume"), false);
      await p.next.fire("click"); await p.next.fire("click");
      assert.equal(p.calls.filter(call => call === "resume").length, 1); assert.equal(p.timers.length, 1);
      assert.equal(p.timers[0].delay, 800); await p.settle(); assert.equal(p.shell.dataset.step, target);
      assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, target !== "phrases");
    }
  });
  await test("a slow Next request starts after notice paint and remains covered until the destination is ready", async () => {
    for (const failed of [false, true]) {
      const project = navigationProject("downloads");
      let finish;
      const p = page(project, { paintFrames: "manual", action: (action, state) => {
        assert.equal(action, "resume");
        assert.equal(p.shell.querySelector(".story-loader").hidden, false);
        return new Promise(resolve => {
          finish = () => {
            state.busy = false;
            if (!failed) {
              project.navigation_checkpoint = { policy: "explicit_story_v1", state: "running", step: "voices" };
              project.status = "running";
              p.ui.render();
            }
            resolve(!failed);
          };
        });
      } });
      const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
      await p.next.fire("click");
      assert.equal(loader.hidden, false);
      assert.equal(p.calls.includes("resume"), false);
      assert.equal(await p.advanceFrame(), true);
      assert.equal(p.calls.includes("resume"), false);
      assert.equal(await p.advanceFrame(), true);
      await p.next.fire("click");
      assert.equal(p.calls.filter(call => call === "resume").length, 1);
      assert.equal(p.advanceTimer(), true);
      await p.flush();
      assert.equal(p.shell.dataset.step, "downloads");
      assert.equal(loader.hidden, false, "the display minimum cannot uncover an unfinished request");
      assert.equal(panel.inert, true);
      finish();
      await p.flush();
      if (!failed) {
        assert.equal(p.shell.dataset.step, "voices");
        assert.equal(loader.hidden, false, "the response alone cannot uncover the destination before paint");
        assert.equal(await p.advanceFrame(), true);
        assert.equal(loader.hidden, false);
        assert.equal(await p.advanceFrame(), true);
      }
      await p.settle();
      assert.equal(p.shell.dataset.step, failed ? "downloads" : "voices");
      assert.equal(loader.hidden, true);
      assert.equal(panel.inert, false);
      assert.equal(p.next.disabled, !failed);
    }
  });
  await test("voice preparation and optional listening are separate pages and only listening next starts pilot generation", async () => {
    const project = navigationProject("voices"), p = page(project, { action: (_action, state) => {
      state.project.navigation_checkpoint = { policy: "explicit_story_v1", state: "running", step: "pronunciation" };
      state.project.status = "running"; state.busy = false; p.ui.render();
    } });
    const preview = p.get("reference-preview"), voiceCounts = p.shell.querySelector(".story-voice-counts");
    assert.equal(p.shell.dataset.step, "voices"); assert.equal(preview.parentElement.hidden, true);
    assert.ok(voiceCounts.children[0].children[1].textContent.includes("20 / 20"));
    assert.ok(voiceCounts.children[1].children[1].textContent.includes("30 / 30"));
    assert.equal(p.next.disabled, false); await p.next.fire("click");
    assert.equal(p.calls.includes("resume"), false); await p.settle();
    assert.equal(p.shell.dataset.step, "voice-listening"); assert.equal(preview.parentElement.hidden, false); assert.equal(preview.open, true);
    assert.equal(voiceCounts.hidden, true);
    assert.equal(p.shell.querySelector(".story-listening-status").hidden, true);
    assert.equal(p.shell.querySelector(".story-listening-status").textContent, "");
    assert.equal(p.shell.querySelector(".story-listening-note"), null);
    assert.ok(p.viewModes.some(({ view, mode }) => view === "review" && mode === "reference"));
    assert.equal(p.operation.hidden, true); assert.equal(p.next.disabled, false);
    p.ui.render(); assert.equal(preview.open, true, "polling keeps the optional listening document open");
    await p.back.fire("click"); await p.settle(); assert.equal(preview.open, false);
    assert.ok(p.calls.includes("pause:reference-preview-frame")); assert.equal(p.calls.includes("resume"), false);
    await p.next.fire("click"); await p.settle(); await p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "resume").length, 1, "no audio must be played or approved to continue");
    assert.equal(p.calls.includes("input"), false); await p.settle(); assert.equal(p.shell.dataset.step, "pronunciation");
  });
  await test("candidate review layout accepts only a matching project and trusted reference frame", async () => {
    const project = navigationProject("voices"), p = page(project);
    await p.next.fire("click"); await p.settle();
    const wrapper = p.shell.querySelector(".story-voice-listening");
    const source = {}, frame = p.get("reference-preview-frame"); frame.contentWindow = source;
    const message = { origin: "http://studio.local", source, data: { type: "koww:reference-layout", project: project.id, mode: "candidate_pool_v2" } };
    for (const invalid of [
      { ...message, origin: "https://other.invalid" }, { ...message, source: {} },
      { ...message, data: { ...message.data, project: "another-project" } },
      { ...message, data: { ...message.data, mode: "unknown" } },
    ]) await p.windowEvent("message", invalid);
    assert.equal(wrapper.classList.contains("candidate-review"), false);
    assert.equal(p.shell.classList.contains("candidate-reference-review"), false);
    await p.windowEvent("message", message);
    assert.equal(wrapper.classList.contains("candidate-review"), true);
    assert.equal(p.shell.classList.contains("candidate-reference-review"), true);
    await p.windowEvent("message", { ...message, data: { ...message.data, mode: "legacy" } });
    assert.equal(wrapper.classList.contains("candidate-review"), false);
    assert.equal(p.shell.classList.contains("candidate-reference-review"), false);
  });
  await test("candidate review height follows its content only for the current trusted frame", async () => {
    const project = navigationProject("voices"), p = page(project);
    await p.next.fire("click"); await p.settle();
    const source = {}, frame = p.get("reference-preview-frame"); frame.contentWindow = source;
    const send = data => p.windowEvent("message", { origin: "http://studio.local", source, data: { project: project.id, ...data } });
    const height = value => ({ type: "koww:reference-height", mode: "candidate_pool_v2", height: value });
    await send(height(500));
    assert.equal(p.shell.style["--reference-content-height"], "", "a height message cannot turn legacy review into a candidate chapter");
    await send({ type: "koww:reference-layout", mode: "candidate_pool_v2" });
    await send(height(500)); assert.equal(p.shell.style["--reference-content-height"], "500px");
    await send(height(2400)); assert.equal(p.shell.style["--reference-content-height"], "2400px", "an open selection list expands the chapter instead of creating an inner scroll box");
    await send(height(280)); assert.equal(p.shell.style["--reference-content-height"], "280px", "folding the list can shrink the frame again");
    for (const invalid of [
      { origin: "https://other.invalid", source, data: { project: project.id, ...height(999) } },
      { origin: "http://studio.local", source: {}, data: { project: project.id, ...height(999) } },
      { origin: "http://studio.local", source, data: { project: "another-project", ...height(999) } },
      ...[0, -1, 2.5, NaN, Infinity, 100001].map(value => ({ origin: "http://studio.local", source, data: { project: project.id, ...height(value) } })),
    ]) await p.windowEvent("message", invalid);
    assert.equal(p.shell.style["--reference-content-height"], "280px");
    await frame.fire("load"); assert.equal(p.shell.style["--reference-content-height"], "", "a replaced document cannot inherit a previous expanded list height");
    await send({ type: "koww:reference-layout", mode: "candidate_pool_v2" }); await send(height(600));
    await send({ type: "koww:reference-layout", mode: "legacy" });
    assert.equal(p.shell.style["--reference-content-height"], "");
    assert.equal(p.calls.some(value => ["create", "start", "resume", "input"].includes(value)), false, "layout measurements cannot alter work or decisions");
  });
  await test("only the active reference frame can expand its picker and navigation clears the expansion", async () => {
    const p = page(navigationProject("voices"));
    await p.next.fire("click"); await p.settle();
    const wrapper = p.shell.querySelector(".story-voice-listening"), posted = [], sequence = [];
    const source = { postMessage: (data, origin) => posted.push({ data, origin }) };
    const frame = p.get("reference-preview-frame"); frame.contentWindow = source;
    frame.getBoundingClientRect = () => { assert.equal(wrapper.className.includes("is-selecting"), true); sequence.push("layout"); return { height: 360 }; };
    source.postMessage = (data, origin) => { if (data.type === "koww:voice-picker-ready") sequence.push("ready"); posted.push({ data, origin }); };
    const message = { origin: "http://studio.local", source, data: { type: "koww:voice-picker", open: true, request: 1 } };
    await p.windowEvent("message", { ...message, origin: "https://other.invalid" });
    await p.windowEvent("message", { ...message, source: {} });
    await p.windowEvent("message", { ...message, data: { type: "koww:voice-picker", open: "true" } });
    for (const request of [undefined, 0, -1, 1.5, "1"]) await p.windowEvent("message", { ...message, data: { ...message.data, request } });
    assert.equal(wrapper.className.includes("is-selecting"), false);
    assert.equal(posted.length, 0);
    await p.windowEvent("message", message);
    assert.equal(wrapper.className.includes("is-selecting"), true);
    assert.deepEqual(sequence, ["layout", "ready"], "the host expands and resolves iframe layout before allowing the popup to appear");
    assert.equal(posted[0].data.type, "koww:voice-picker-ready"); assert.equal(posted[0].data.request, 1);
    assert.equal(posted[0].origin, "http://studio.local");
    p.ui.render(); assert.equal(wrapper.className.includes("is-selecting"), true);
    await p.windowEvent("message", { ...message, data: { ...message.data, open: false } });
    assert.equal(wrapper.className.includes("is-selecting"), false);
    await p.windowEvent("message", message);
    await p.windowEvent("pointerdown");
    assert.equal(wrapper.className.includes("is-selecting"), false);
    assert.equal(posted.length, 3); assert.equal(posted.at(-1).data.type, "koww:voice-picker-close");
    assert.equal(posted.at(-1).origin, "http://studio.local");
    await p.windowEvent("message", message);
    await p.back.fire("click"); await p.settle();
    assert.equal(wrapper.className.includes("is-selecting"), false);
    await p.windowEvent("message", message);
    assert.equal(wrapper.className.includes("is-selecting"), false);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("story next never becomes a task control and pending work pauses and resumes only through the central button", async () => {
    const project = navigationProject("downloads");
    project.navigation_checkpoint = { policy: "explicit_story_v1", state: "running", step: "voices" }; project.status = "running";
    const p = page(project); assert.equal(p.shell.dataset.step, "voices");
    assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true); assert.equal(p.operation.textContent, "일시정지");
    await p.next.fire("click"); assert.equal(p.calls.includes("pause"), false);
    await p.operation.fire("click"); await p.operation.fire("click"); assert.equal(p.calls.filter(call => call === "pause").length, 1);
    p.state.busy = false; project.status = "paused"; p.ui.render();
    assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true); assert.equal(p.operation.textContent, "재개");
    await p.next.fire("click"); assert.equal(p.calls.includes("resume"), false);
    await p.operation.fire("click"); assert.equal(p.calls.filter(call => call === "resume").length, 1);
  });
  await test("story checkpoints never unlock unfinished prefixes, unsaved review/input, migration or a running worker", async () => {
    for (const variant of ["missing", "failed", "review", "input", "migration", "running"]) {
      const project = navigationProject("voices");
      if (variant === "missing") project.stages = project.stages.filter(stage => stage.id !== "references_review");
      else if (variant === "failed") project.stages.find(stage => stage.id === "data_connect").status = "failed";
      else if (variant === "running") project.status = "running";
      else project[variant] = { state: "available", kind: "required" };
      const p = page(project);
      assert.equal(p.next.disabled, true, variant + " must block next");
      await p.next.fire("click"); await p.operation.fire("click");
      assert.equal(p.calls.some(call => ["start", "resume", "pause", "input"].includes(call)), false);
      if (["missing", "failed"].includes(variant)) assert.equal(p.api.frontier(project), 5);
    }
  });
  await test("unfinished results work retains its status and central recovery controls before the result page exists", async () => {
    for (const marker of [false, true]) for (const status of ["ready", "running", "paused", "error", "interrupted"]) {
      const project = { id: "results-worker", status, stage_id: "results", progress: {} };
      if (marker) project.results = { url: "/previous-results/" };
      if (marker) project.navigation_checkpoint = { policy: "explicit_story_v1", step: "results", state: "running" };
      const p = page(project);
      assert.equal(p.shell.dataset.step, "results"); assert.equal(p.shell.querySelector(".story-live-summary").hidden, true);
      assert.equal(p.get("story-title").textContent, "학습 결과");
      assert.equal(p.get("story-subtitle").textContent, "학습 결과를 정리하고 모델별 감지 설정을 비교합니다.");
      assert.equal(p.get("results-section").hidden, true, "a stale report URL does not replace active result preparation");
      assert.equal(p.viewModes.at(-1).view, "overview");
      assert.equal(p.operation.hidden, false); assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true);
      await p.next.fire("click"); assert.equal(p.calls.some(call => ["start", "pause", "resume"].includes(call)), false);
      await p.operation.fire("click");
      assert.equal(p.calls.includes(status === "running" ? "pause" : status === "ready" ? "start" : "resume"), true);
    }
  });
  await test("legacy frontend work starts in downloads and retains pause and resume controls", async () => {
    const project = checkpointProject(), p = page(project), primary = p.operation;
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(primary.hidden, false); assert.equal(primary.textContent, "설치·다운로드 시작");
    assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true);
    await p.next.fire("click"); assert.equal(p.calls.includes("start"), false);
    await primary.fire("click"); await primary.fire("click"); assert.equal(p.calls.filter(call => call === "start").length, 1);
    p.state.busy = false; project.status = "running"; project.stage_id = "data_download"; project.download_checkpoint.state = "running";
    project.progress = { done: 100, total: 100 }; p.ui.render(); await p.settle();
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(primary.hidden, false); assert.equal(p.operation.textContent, "일시정지"); assert.equal(p.next.disabled, true);
    await p.next.fire("click"); assert.equal(p.calls.includes("pause"), false);
    await p.operation.fire("click"); assert.equal(p.calls.filter(call => call === "pause").length, 1);
    p.state.busy = false; project.status = "paused"; p.ui.render();
    assert.equal(primary.hidden, false); assert.equal(p.operation.textContent, "재개"); await p.operation.fire("click");
    assert.equal(p.calls.filter(call => call === "resume").length, 1);
    assert.equal(p.shell.dataset.step, "downloads");
  });
  await test("live download density uses measured space without hiding rows or changing progress and operation gates", async () => {
    for (const [width, height, available, rowHeights, expectedDensity, overflow] of [
      [1280, 800, 460, { normal: 80, compact: 64, tight: 48 }, "normal", false],
      [1280, 800, 460, { normal: 120, compact: 80, tight: 48 }, "compact", false],
      [600, 540, 205, { normal: 100, compact: 50, tight: 20 }, "tight", false],
      [319, 950, 518, { normal: 128, compact: 80, tight: 64 }, "tight", true],
    ]) {
      const project = checkpointProject("running", "running"); project.stage_id = "data_download";
      project.progress = { done: 17, total: 100, files: Array.from({ length: 9 }, (_, i) => ({
        id: `asset-${i}`, group: `group-${i}`, label: `자료 ${i + 1}`, state: "downloading", done: 17, total: 100, unit: "bytes",
      })) };
      const p = page(project, { width, height, getComputedStyle: () => ({ rowGap: "0px" }) });
      const history = p.shell.querySelector(".story-history"), list = history.querySelector(".story-history-rows");
      p.shell.querySelector(".story-stage").getBoundingClientRect = () => ({ height: available });
      p.shell.querySelector(".story-panel-heading").getBoundingClientRect = () => ({ height: 0 });
      p.shell.querySelector(".story-actions").getBoundingClientRect = () => ({ height: 0 });
      const rows = [...list.children];
      rows.forEach(row => { row.getBoundingClientRect = () => ({ height: rowHeights[list.dataset.density] }); });
      project.progress.files[0].done = 18; p.ui.render();
      const captionHeight = p.get("story-download-activity").hidden ? 0 : 22;
      assert.equal(list.style["--download-list-height"], `${available - captionHeight}px`, "only a visible work caption reserves space outside the download rows");
      assert.equal(list.dataset.density, expectedDensity);
      assert.equal(list.dataset.overflow, String(overflow), "unavoidable overflow is reported without concealing any asset");
      assert.equal(rows.filter(row => /^자료 \d+$/.test(historyRowLabel(row))).length, 9);
      assert.ok(rows.every(row => !row.hidden));
      assert.equal(history.querySelector(".story-pagination").hidden, true);
      assert.equal(p.next.disabled, true); assert.equal(p.operation.textContent, "일시정지");
      p.ui.render(); assert.deepEqual(list.children, rows, "polling preserves every displayed download row");
      assert.equal(project.progress.done, 17); assert.equal(project.progress.files[0].done, 18);
      assert.equal(p.calls.some(call => ["start", "pause", "resume"].includes(call)), false);
      p.state.project = null; p.ui.render();
      assert.equal(p.shell.querySelector(".story-live").hidden, true, "new setup does not expose the retained download contents");
    }
  });
  await test("parallel download gauges retain one completed compute-tools row", () => {
    const project = checkpointProject("running", "running");
    project.stage_id = "data_download";
    project.stages[0].status = "completed";
    project.dependency_download_bytes = 2 * 1024 ** 3;
    project.download_setup_items = ["torch", "torchaudio", "openwakeword"].map(id => ({ id, label: id, state: "complete" }));
    project.download_files = [
      { id: "a", label: "자료 A", state: "downloading", done: 2500, total: 10000, unit: "bytes" },
      { id: "b", label: "자료 B", state: "downloading", done: 750, total: 1000, unit: "bytes" },
      { id: "c", label: "자료 C", state: "waiting", done: 0, total: 500, unit: "bytes" },
    ];
    const p = page(project);
    const history = p.shell.querySelector(".story-history"), list = history.querySelector(".story-history-rows");
    const rows = ["자료 A", "자료 B", "자료 C"].map(label => historyRowByLabel(list, label));
    assert.equal(list.children.length, 4);
    const setup = historyRowByLabel(list, "연산 도구 · PyTorch · TorchAudio");
    assert.equal(setup.querySelector(".story-history-percent").textContent, "100%");
    assert.equal(setup.querySelector(".story-history-note-detail").textContent, "2 GiB / 2 GiB");
    assert.equal(setup.querySelector(".story-progress-indicator").dataset.status, "complete");
    assert.deepEqual(rows.map(row => row.querySelector(".story-history-percent").textContent), ["25%", "75%", "0%"]);
    assert.deepEqual(rows.map(row => row.querySelector("progress").value), [2500, 750, 0]);
    assert.equal(history.querySelector(".story-pagination").hidden, true);
    project.download_files[0].done = 5000;
    p.ui.render();
    assert.deepEqual(["자료 A", "자료 B", "자료 C"].map(label => historyRowByLabel(list, label)), rows);
    assert.deepEqual(rows.map(row => row.querySelector(".story-history-percent").textContent), ["50%", "75%", "0%"]);
  });
  await test("CPU and GPU packages share one gauge based on verified install counts", () => {
    for (const build of ["cpu", "cu128"]) {
      const project = checkpointProject();
      project.dependency_backend = build;
      const sizeKnown = build === "cu128";
      if (sizeKnown) project.dependency_download_bytes = 2 * 1024 ** 3;
      const torch = { id: "torch", label: "torch==2.8.0+" + build, state: "waiting" };
      const audio = { id: "torchaudio", label: "TorchAudio", state: "waiting" };
      project.download_setup_items = [torch, audio, { id: "openwakeword", label: "openwakeword==0.6.0", state: "waiting" }];
      project.download_files = [{ id: "a", label: "자료 A", state: "waiting", done: 0, total: 500, unit: "bytes" }];
      const p = page(project), list = p.shell.querySelector(".story-history-rows");
      assert.equal(list.children.length, 2);
      const row = historyRowByLabel(list, "연산 도구 · PyTorch · TorchAudio"), bar = row.querySelector("progress");
      assert.doesNotMatch(textOf(row), /CPU|GPU|CUDA/);
      const capacity = row.querySelector(".story-history-note-detail");
      assert.equal(capacity.textContent, sizeKnown ? "0 B / 2 GiB" : "전체 용량 확인 전");
      project.status = "running";
      project.download_checkpoint.state = "running";
      torch.state = "installing";
      p.ui.render();
      assert.equal(capacity.textContent, sizeKnown ? "— / 2 GiB" : "전체 용량 확인 전");
      assert.equal(bar.hidden, false);
      assert.equal(bar.getAttribute("value"), null);
      assert.equal(row.querySelector(".story-history-percent").textContent, "");
      assert.equal(row.querySelector(".story-progress-indicator").dataset.status, "active");
      assert.equal(historyRowByLabel(list, "자료 A").querySelector("progress").value, 0);
      torch.state = "complete"; audio.state = "installing"; p.ui.render();
      assert.equal(historyRowByLabel(list, "연산 도구 · PyTorch · TorchAudio"), row);
      assert.equal(bar.max, 2);
      assert.equal(bar.value, 1);
      assert.equal(row.querySelector(".story-history-percent").textContent, "50%");
      audio.state = "complete"; p.ui.render();
      assert.equal(bar.value, 2);
      assert.equal(row.querySelector(".story-history-percent").textContent, "100%");
    }
  });
  await test("a shared package failure preserves verified compute installs without a running spinner", async () => {
    for (const hasSetupRows of [false, true]) {
      const project = checkpointProject("running", "error");
      project.stage_id = "frontend";
      project.error = { message: "", type: "MemoryError", stage: "frontend" };
      project.dependency_packages = ["torch==2.8.0+cu128", "torchaudio==2.8.0+cu128", "qwen-tts==0.1.1"];
      project.evaluation_runtime_packages = ["onnxruntime-gpu==1.31.0"];
      if (hasSetupRows) project.download_setup_items = [
        { id: "environment-check", label: "학습·평가 실행 환경 확인", state: "waiting" },
        ...[...project.dependency_packages, ...project.evaluation_runtime_packages]
          .map(label => ({ id: label.split("==")[0], label, state: label.startsWith("torch==") ? "installing" : "waiting" })),
      ];
      let requests = 0;
      const p = page(project, { installLog: async () => {
        requests++;
        return { log: "Requirement already satisfied: torch>=2.0.0 in site-packages (2.8.0+cu128)\nSuccessfully installed onnxruntime-gpu-1.31.0\nMemoryError\n" };
      } });
      await p.flush(); p.ui.render();
      const history = p.shell.querySelector(".story-history");
      assert.equal(history.hidden, false);
      const row = historyRowByLabel(history, "연산 도구 · PyTorch · TorchAudio");
      assert.match(textOf(row), /1 \/ 2개/);
      assert.equal(row.dataset.failed, "false");
      assert.equal(history.querySelectorAll(".story-history-row").length, 1);
      assert.equal(hasActiveDownloadSpinner(history), false);
      assert.equal(requests, 1);
    }
  });
  await test("download stays at 100% while quality inspection reports its own state", () => {
    const project = checkpointProject("running", "running");
    project.stage_id = "data_download";
    project.download_files = [{ id: "a", group: "generic_negative", label: "일반 비호출 자료", state: "complete", done: 100, total: 100, unit: "bytes" }];
    const p = page(project);
    const row = historyRowByLabel(p.shell.querySelector(".story-history"), "일반 비호출 자료");
    const percent = row.querySelector(".story-history-percent"), quality = row.querySelector(".story-quality-label");
    assert.equal(percent.textContent, "100%");
    assert.equal(quality.textContent, "품질 검사 대기");
    assert.equal(hasActiveDownloadSpinner(row), false);
    project.download_quality = { state: "checking", done: 7, total: 10 };
    p.ui.render();
    assert.equal(percent.textContent, "100%");
    assert.equal(quality.textContent, "품질 검사 중");
    assert.equal(hasActiveDownloadSpinner(row), true);
    project.status = "error"; project.error = "자료 검사 실패";
    p.ui.render();
    assert.equal(percent.textContent, "100%");
    assert.equal(quality.textContent, "품질 검사 미완료");
    assert.equal(hasActiveDownloadSpinner(row), false);
    assert.equal(p.get("story-download-activity").hidden, true);
  });
  await test("download checkpoint holds a completed data phase until next explicitly releases it and follows a busy response once", async () => {
    const project = checkpointProject("awaiting_next", "paused"); project.stage_id = "references";
    for (const record of project.stages) if (record.id !== "references") record.status = "completed";
    const p = page(project, { action: (_action, state) => {
      state.project.download_checkpoint.state = "released"; state.project.status = "running";
      p.ui.render(); assert.equal(p.shell.dataset.step, "downloads");
      assert.equal(p.shell.querySelector(".story-loader").hidden, false);
      state.busy = false; p.ui.render();
    } });
    const primary = p.operation;
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.api.frontier(project), 4);
    assert.equal(p.next.disabled, false); assert.equal(primary.hidden, true);
    p.ui.render(); assert.equal(p.timers.length, 0, "awaiting-next polling never starts voice generation or navigation");
    assert.equal(p.calls.includes("resume"), false);
    await p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "resume").length, 1); assert.equal(p.timers.length, 1);
    assert.equal(p.timers[0].delay, 800); p.ui.render(); assert.equal(p.timers.length, 1);
    await p.settle(); assert.equal(p.shell.dataset.step, "voices");
  });
  await test("explicit download controls never bypass pending human input, review, migration or an unproven completed phase", async () => {
    for (const key of ["input", "review", "migration"]) {
      const project = checkpointProject("running", "waiting_input"); project[key] = { kind: "required", state: "available" };
      const p = page(project), primary = p.operation;
      await primary.fire("click"); await p.next.fire("click");
      assert.equal(p.next.disabled, true); assert.equal(p.calls.some(call => ["start", "pause", "resume", "migrate"].includes(call)), false);
    }
    const project = checkpointProject("awaiting_next", "paused"); project.stage_id = "references";
    const p = page(project); assert.equal(p.next.disabled, true); await p.next.fire("click");
    assert.equal(p.calls.includes("resume"), false, "a marker alone cannot prove material preparation completed");
  });
  await test("awaiting-next checkpoint keeps the existing migration control visible outside hidden live progress", async () => {
    const project = checkpointProject("awaiting_next", "paused"); project.stage_id = "references";
    for (const record of project.stages) if (record.id !== "references") record.status = "completed";
    project.migration = { state: "available", reason: "화면 업데이트를 적용하세요." };
    const p = page(project), migration = p.get("migration-note"), primary = p.operation;
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.shell.querySelector(".story-live-summary").hidden, true);
    assert.equal(migration.hidden, false); assert.equal(migration.parentElement.className, "story-live");
    assert.equal(p.get("migrate").parentElement, migration, "the original migration button stays connected and exposed");
    assert.equal(p.next.disabled, true); assert.equal(primary.hidden, true);
    await p.next.fire("click"); await primary.fire("click");
    assert.equal(p.calls.some(call => ["start", "resume", "input"].includes(call)), false);
    project.migration = null; p.ui.render();
    assert.equal(migration.hidden, true); assert.equal(p.next.disabled, false);
    project.review = { url: "/review/" }; p.ui.render(); assert.equal(p.next.disabled, true);
    await p.next.fire("click"); assert.equal(p.calls.includes("resume"), false, "human review remains a separate gate after migration");
  });
  await test("review/input gates cannot be bypassed by the main next button", async () => {
    const p = page({ id: "model", status: "waiting_review", stage_id: "pilot_review", review: { url: "/review/" } });
    assert.equal(p.shell.dataset.step, "pronunciation"); assert.equal(p.next.disabled, true);
    await p.next.fire("click"); assert.equal(p.calls.includes("input"), false); assert.equal(p.timers.length, 0);
  });
  const editablePhrasesProject = () => ({ id: "phrase-edit", wakeword: "가영아", status: "waiting_input", stage_id: "hard_phrases",
    input: { kind: "hard_phrases", suggestions: ["가연아", "나영아", "다영아", "가은아", "하영아", "가현아", "가영이", "나연아"] } });
  await test("phrase input shares the saved grid and removes zero-basis inner layout and redundant save controls", () => {
    const p = page(editablePhrasesProject());
    assert.equal(p.shell.dataset.step, "phrases");
    assert.equal(p.shell.querySelector(".story-live-summary").hidden, true);
    assert.equal(p.get("hard-phrase-fields").classList.contains("story-phrase-grid"), true);
    const css = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio_story.css"), "utf8");
    assert.match(css, /\.story-phrase-grid[^\n]*\{[^}]*grid-template-columns:repeat\(2,minmax\(0,1fr\)\)[^}]*max-width:560px/);
    assert.match(css, /\.story-phrase-grid input[^\n]*\{[^}]*height:44px/);
    assert.match(css, /\[data-step="phrases"\][^\n]+#input-section\[data-input-kind="hard_phrases"\]\{[^}]*height:auto;flex:none;overflow:visible/);
    assert.match(css, /\[data-step="phrases"\][^\n]+#input-form\{display:block;height:auto;flex:none;overflow:visible/);
    assert.match(css, /\[data-step="phrases"\][^\n]+\.input-content\{height:auto;overflow:visible/);
    assert.match(css, /\[data-step="phrases"\][^\n]+#input-submit\{display:none\}/);
    assert.match(css, /\[data-step="phrases"\][^\n]+:is\(#input-label,#hard-phrase-rules\)\{[^}]*clip:rect\(0,0,0,0\)/);
  });
  await test("phrase input footer next submits only the current live form and waits once without an extra resume", async () => {
    const project = editablePhrasesProject(); let finish;
    const p = page(project, { paintFrames: "manual", input: () => new Promise(resolve => { finish = resolve; }) });
    const draft = p.element("input"); draft.value = "직접수정한문구"; p.get("hard-phrase-fields").append(draft);
    assert.equal(p.next.disabled, false);
    const pending = p.next.fire("click"); await p.flush();
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    assert.equal(p.get("story-panel").inert, true);
    assert.equal(p.calls.includes("input"), false);
    assert.equal(await p.advanceFrame(), true);
    assert.equal(p.calls.includes("input"), false);
    assert.equal(await p.advanceFrame(), true);
    await p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "input").length, 1);
    assert.equal(p.shell.dataset.step, "phrases"); assert.equal(draft.value, "직접수정한문구");
    project.status = "running"; project.process_alive = true; delete project.input;
    finish(); await pending; await p.flush();
    while (await p.advanceFrame()) {}
    await p.settle();
    assert.equal(p.calls.some(call => ["start", "resume"].includes(call)), false);
    assert.equal(p.get("hard-phrase-fields").children.at(-1), draft);
  });
  await test("phrase input validation and save errors keep the live fields available for a deliberate retry", async () => {
    let fail = false;
    const p = page(editablePhrasesProject(), { input: () => { if (fail) throw new Error("저장 연결 오류"); } });
    const draft = p.element("input"); draft.value = "직접수정"; draft.focus(); p.get("hard-phrase-fields").append(draft);
    await p.next.fire("click"); await p.settle(); // Validation can return without a POST or state change.
    assert.equal(p.shell.dataset.step, "phrases"); assert.equal(p.next.disabled, false);
    fail = true; await p.next.fire("click"); await p.settle();
    assert.ok(p.calls.includes("message:저장 연결 오류")); assert.equal(p.next.disabled, false);
    assert.equal(p.get("hard-phrase-fields").children.at(-1), draft); assert.equal(draft.value, "직접수정"); assert.equal(draft.focused, true);
    assert.equal(p.calls.some(call => ["start", "resume"].includes(call)), false);
  });
  await test("phrase input next never submits off-route diagnostics reviews migration or an active worker", async () => {
    for (const changes of [{ stage_id: "pilot_review" }, { status: "paused" }, { input: { kind: "diagnostic" } },
      { review: { url: "/review/" } }, { migration: { state: "available" } }, { process_alive: true }]) {
      const p = page(Object.assign(editablePhrasesProject(), changes));
      await p.next.fire("click"); assert.equal(p.calls.includes("input"), false);
      assert.equal(p.calls.some(call => ["start", "resume"].includes(call)), false);
    }
  });
  await test("completed result and microphone navigation pause child media without changing training", async () => {
    const p = page({ id: "model", wakeword: "보라야", status: "completed", results: { url: "/results/", mic_url: "/microphone/" } });
    assert.equal(p.shell.dataset.step, "results");
    await p.next.fire("click"); await p.settle(); assert.equal(p.shell.dataset.step, "microphone");
    assert.ok(p.calls.includes("view:mic")); assert.equal(p.next.hidden, true);
    await p.back.fire("click"); await p.settle(); assert.equal(p.shell.dataset.step, "results");
    assert.ok(p.calls.includes("pause:result-frame"));
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("every reachable chapter completes a covered Previous and Next in a saved workflow", async () => {
    const project = { id: "all-chapters", wakeword: "보라야", status: "completed",
      settings: { pilot_review_policy: "one_per_voice_before_and_after_generation_v4" },
      results: { url: "/results/all-chapters/", mic_url: "/mic/all-chapters/" } };
    const p = page(project, { paintFrames: "manual" });
    const chapters = p.api.definitions.map(([step, title]) => [step, title]);
    const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
    const move = async (button, index) => {
      const [step, title] = chapters[index];
      const source = p.shell.dataset.step;
      assert.equal(button.disabled, false, source);
      await button.fire("click");
      assert.equal(p.shell.dataset.step, source);
      assert.equal(loader.hidden, false);
      assert.equal(panel.inert, true);
      await p.advanceFrame(); await p.advanceFrame();
      for (let i = 0; i < 8; i++) { await p.advanceFrame(); await p.flush(); }
      await p.settle();
      assert.equal(p.shell.dataset.step, step);
      assert.equal(p.get("story-title").textContent, title);
      assert.equal(loader.hidden, true);
      assert.equal(panel.inert, false);
    };
    await move(p.next, chapters.length - 1);
    for (let index = chapters.length - 2; index >= 0; index--) await move(p.back, index);
    for (let index = 1; index < chapters.length; index++) await move(p.next, index);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("every ordinary workflow destination waits for paint on refresh and automatic movement", async () => {
    const settings = { pilot_review_policy: "one_per_voice_before_and_after_generation_v4", hard_review_policy: "one_per_phrase_v1" };
    const base = { id: "all-active-chapters", wakeword: "보라야", settings, status: "running", process_alive: true };
    const destinations = [
      ["downloads", { stage_id: "data_download" }],
      ["voices", { stage_id: "references" }],
      ["call-pilot-generation", { stage_id: "pilot" }],
      ["positive-generation", { stage_id: "positives" }],
      ["phrases", { stage_id: "hard_phrases", status: "waiting_input", process_alive: false,
        input: { kind: "hard_phrases", phrases: ["보라빛"] } }],
      ["hard-generation", { stage_id: "hard_generation" }],
      ["features", { stage_id: "features" }],
      ["training", { stage_id: "training" }],
      ["evaluation", { stage_id: "evaluation" }],
      ["results", { stage_id: "results" }],
    ];
    const opened = p => {
      assert.equal(p.get("story-panel").inert, false);
      assert.equal(p.shell.querySelector(".story-loader").hidden, true);
      assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
    };
    for (const [step, changes] of [["entry", null], ...destinations]) {
      const p = page(changes ? { ...base, ...changes } : null, { paintFrames: "manual" });
      let reveal;
      const minimum = new Promise(resolve => { reveal = resolve; });
      const prepared = p.ui.prepareInitialView(minimum);
      await p.flush();
      assert.equal(p.get("story-panel").inert, true, step);
      while (await p.advanceFrame()) {}
      assert.equal(p.get("story-panel").inert, true, step);
      reveal();
      assert.equal(await prepared, true, step);
      assert.equal(p.shell.dataset.step, step);
      opened(p);
    }
    const p = page({ ...base, ...destinations[0][1] }, { paintFrames: "manual" });
    for (const [step, changes] of destinations.slice(1)) {
      p.state.project = { ...base, ...changes };
      p.ui.render();
      assert.equal(p.get("story-panel").inert, true, step);
      assert.equal(p.shell.querySelector(".story-loader").hidden, false, step);
      while (await p.advanceFrame()) {}
      await p.settle();
      assert.equal(p.shell.dataset.step, step);
      opened(p);
    }
  });
  await test("saved model entry preserves the recorded word without editing or overwriting a new-model draft", async () => {
    const model = { id: "model", wakeword: "보라야", status: "completed", results: { url: "/results/" } };
    const p = page(model);
    p.get("wakeword").value = "새호출어";
    await p.ticks[0].fire("click"); await p.settle();
    const saved = p.get("story-saved-wakeword");
    assert.equal(p.shell.dataset.step, "entry");
    assert.equal(saved.parentElement.hidden, false);
    assert.equal(saved.value, "보라야"); assert.equal(saved.readOnly, true);
    assert.equal(saved.disabled, false, "read-only text keeps its normal appearance and can be selected");
    assert.equal(p.get("wakeword").parentElement.hidden, true);
    assert.equal(p.shell.querySelector(".story-history").hidden, true);
    p.ui.render(); assert.equal(saved.value, "보라야");
    assert.equal(p.get("wakeword").value, "새호출어");
    p.state.project = { ...model, id: "other", wakeword: "다른이름" }; p.ui.render();
    await p.ticks[0].fire("click"); await p.settle(); assert.equal(saved.value, "다른이름");
    p.state.project = null; p.ui.render();
    assert.equal(p.get("wakeword").parentElement.hidden, false);
    assert.equal(p.get("wakeword").value, "새호출어");
    assert.notEqual(p.get("wakeword").readOnly, true);
    assert.equal(saved.parentElement.hidden, true);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  const savedStages = () => ["frontend", "data_download", "data_review", "data_connect", "references", "references_review", "pilot", "pilot_review", "positives", "hard_phrases", "hard_generation", "hard_review", "hard_connect", "features", "training", "package", "development", "evaluation", "results"].map(id => ({ id, status: "completed" }));
  const historyModel = views => ({ id: "history", wakeword: "보라야", preset: "production", status: "completed", stages: savedStages(), completed_stage_views: views, results: { url: "/results/" } });
  await test("phrase input saved six-phrase view keeps its original read-only values and shared presentation", async () => {
    const phrases = ["언니야", "언제나", "원년아", "작년아", "언니가", "언년이"];
    const project = historyModel({ phrases: { completed: true, phrases } }), original = JSON.stringify(project);
    const p = page(project);
    await p.ticks.find(tick => tick.dataset.step === "phrases").fire("click"); await p.settle();
    const grid = p.shell.querySelector(".story-saved-phrases");
    assert.ok(grid); assert.equal(grid.classList.contains("story-phrase-grid"), true);
    assert.deepEqual(grid.querySelectorAll("input").map(field => field.value), phrases);
    assert.ok(grid.querySelectorAll("input").every(field => field.readOnly));
    assert.equal(JSON.stringify(project), original);
    assert.equal(p.calls.some(call => ["input", "start", "resume", "create"].includes(call)), false);
  });
  await test("a saved dependency prompt previews the chosen packages and waits for explicit install and download", async () => {
    const project = checkpointProject("running", "running");
    delete project.download_checkpoint;
    project.preset = "check";
    project.navigation_checkpoint = { policy: "explicit_story_v1", step: "downloads", state: "running" };
    project.download_files = ["a", "b"].map(id => ({ id, label: `자료 ${id}`, state: "waiting", done: 0, total: 100, unit: "bytes" }));
    let finishSave;
    const p = page(project, { input: async state => {
      state.busy = true; p.ui.render();
      const saved = await new Promise(resolve => { finishSave = resolve; });
      state.busy = false;
      return saved;
    } });
    p.state.recipes = [{ id: "check" }];
    const history = p.shell.querySelector(".story-history"), input = p.get("input-section");
    const activity = p.get("story-download-activity");
    const tools = p.ticks.find(tick => tick.dataset.step === "tools"), downloads = p.ticks.find(tick => tick.dataset.step === "downloads");
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(downloads.getAttribute("aria-current"), "step");
    assert.equal(history.hidden, false);
    assert.equal(historyRowByLabel(history, "자료 a").querySelector(".story-history-percent").textContent, "0%");
    assert.equal(input.hidden, true);
    assert.equal(activity.hidden, false);
    assert.match(textOf(activity), /실행환경/);

    project.status = "waiting_input";
    project.input = { kind: "dependency_setup", recommended_backend: "cu128" };
    project.dependency_setup = toolSetup;
    p.ui.render(); await p.settle();
    assert.equal(p.shell.dataset.step, "tools");
    assert.equal(p.get("story-title").textContent, "학습 도구 준비");
    assert.equal(tools.getAttribute("aria-current"), "step");
    assert.equal(downloads.disabled, true);
    assert.equal(input.hidden, false);
    assert.equal(activity.hidden, true);
    assert.equal(history.hidden, true, "planned rows cannot overlap the CPU/GPU form");
    assert.equal(p.next.textContent, "다음");
    assert.equal(p.next.disabled, false);
    assert.equal(p.operation.hidden, true);

    p.get("dependency-cpu").checked = true;
    p.get("dependency-cu128").checked = false;
    await advanceSetup(p, "scale", "materials");
    assert.equal(input.hidden, true);
    assert.equal(p.operation.hidden, true);
    assert.equal(p.calls.some(call => ["input", "start", "resume"].includes(call)), false);
    const installs = p.get("story-tool-materials");
    assertToolPreview(installs, /\bCPU\b/, "다운로드 용량 미확정");
    assert.doesNotMatch(textOf(installs), /NVIDIA GPU|CUDA 12\.8/);

    const confirmation = p.next.fire("click"); await p.flush();
    assert.equal(p.get("story-panel").inert, true);
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "input").length, 1);

    project.status = "paused";
    delete project.input;
    project.dependency_backend = "cpu";
    project.dependency_setup_selected = true;
    project.dependency_install_pending = true;
    finishSave(true); await p.settle(); await confirmation; await p.settle();
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(p.operation.textContent, "설치·다운로드 시작");
    assert.equal(p.operation.hidden, false);
    assert.equal(p.next.disabled, true);
    assert.equal(p.calls.some(call => ["start", "resume"].includes(call)), false);

    await p.operation.fire("click"); await p.operation.fire("click");
    assert.equal(p.calls.filter(call => call === "start").length, 1);
    assert.equal(p.calls.includes("resume"), false);

    project.status = "running";
    project.dependency_install_pending = false;
    project.progress = { phase: "evaluation_environment", label: "GPU 평가 실행환경 설치·확인", phase_elapsed_seconds: 74.8 };
    p.state.busy = false; p.ui.render();
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(p.next.textContent, "다음");
    assert.equal(input.hidden, true);
    assert.equal(history.hidden, false);
    assert.equal(activity.querySelector("i").dataset.status, "active");
    assert.match(textOf(activity), /GPU 평가 실행환경 설치·확인 · 1분 14초 경과/);

    project.stages[0].status = "completed";
    project.stage_id = "data_download";
    project.progress = {};
    project.download_files[0] = { ...project.download_files[0], state: "downloading", done: 35 };
    p.ui.render(); await p.settle();
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(downloads.getAttribute("aria-current"), "step");
    assert.equal(input.hidden, true);
    assert.equal(history.hidden, false);
    assert.match(textOf(history), /자료 a.*35%/);
    assert.match(textOf(history), /자료 b/);

    await tools.fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "tools");
    assert.equal(input.hidden, true);
    assert.equal(history.hidden, true);
    assert.equal(p.get("story-tool-picker").hidden, false);
    assert.equal(p.get("story-tool-cpu").checked, true);
    assert.equal(p.get("story-tool-picker").disabled, true);
    await downloads.fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "downloads");
    assert.match(textOf(history), /35%/);
    assert.equal(p.calls.filter(call => call === "input").length, 1);
    assert.equal(p.calls.filter(call => call === "start").length, 1);
    assert.equal(p.calls.some(call => ["resume", "pause"].includes(call)), false);
  });
  await test("cached download verification uses its own measured gauge without displaying transferred bytes as completion", () => {
    const project=checkpointProject("running","running");project.stage_id="data_download";
    project.download_files=[{id:"asset-a",group:"group-a",label:"자료 A",state:"verifying",done:1000,total:1000,unit:"bytes"},
      {id:"asset-b",group:"group-b",label:"자료 B",state:"verifying",done:2000,total:2000,unit:"bytes"}];
    project.progress={stage:"data_download",current_asset:"asset-a",phase:"hash",label:"원본 파일 검증",current_event:{phase:"hash",done:7,total:100},files:project.download_files};
    const p=page(project),rows=["자료 A", "자료 B"].map(label => historyRowByLabel(p.shell, label));
    assert.match(textOf(rows[0]),/0 \/ 1개/);assert.match(textOf(rows[0]),/7%/);assert.doesNotMatch(textOf(rows[0]),/100%/);
    assert.equal(rows[0].querySelector("progress").max,100);assert.equal(rows[0].querySelector("progress").value,7);
    assert.equal(rows[1].querySelector("progress").hidden,false);assert.equal(rows[1].querySelector("progress").getAttribute("value"),null);
    assert.equal(rows[1].querySelector(".story-history-percent").textContent,"");
    assert.match(rows[0].querySelector("progress").getAttribute("aria-label"),/무결성 검사/);
    project.download_files[0].state="reused";p.ui.render();assert.match(textOf(rows[0]),/1 \/ 1개/);
    assert.equal(rows[0].querySelector(".story-progress-indicator").dataset.status,"complete");
  });
  await test("parallel downloads retain each active file hash gauge until verification or the attempt changes", () => {
    const project = checkpointProject("running", "running");
    project.stage_id = "data_download";
    const zeroth = { id: "base:zeroth", group: "zeroth", label: "Zeroth", state: "verifying", done: 100, total: 100, unit: "bytes" };
    const musan = { id: "common:musan", group: "musan", label: "MUSAN", state: "downloading", done: 30, total: 100, unit: "bytes" };
    project.download_files = [zeroth, musan];
    const hashEvent = (done, taskId = "attempt-one") => ({ stage: "data_download", task_id: taskId, current_asset: zeroth.id,
      current_event: { phase: "hash", done, total: 100 } });
    const downloadEvent = (taskId = "attempt-one") => ({ stage: "data_download", task_id: taskId, current_asset: musan.id,
      current_event: { phase: "download", done: musan.done, total: musan.total } });
    project.progress = hashEvent(18);
    const p = page(project);
    const row = historyRowByLabel(p.shell, "Zeroth");
    const bar = row.querySelector("progress"), percent = row.querySelector(".story-history-percent");
    assert.equal(percent.textContent, "18%");
    assert.equal(bar.value, 18);

    musan.done = 32; project.progress = downloadEvent(); p.ui.render();
    assert.equal(historyRowByLabel(p.shell, "Zeroth"), row);
    assert.equal(percent.textContent, "18%");
    assert.equal(bar.value, 18);
    assert.equal(historyRowByLabel(p.shell, "MUSAN").querySelector(".story-history-percent").textContent, "32%");

    project.progress = hashEvent(22); p.ui.render();
    assert.equal(percent.textContent, "22%");
    assert.equal(bar.value, 22);
    zeroth.state = "retrying"; project.progress = downloadEvent(); p.ui.render();
    zeroth.state = "verifying"; p.ui.render();
    assert.equal(percent.textContent, "");
    assert.equal(bar.getAttribute("value"), null);

    project.progress = hashEvent(24); p.ui.render();
    assert.equal(bar.value, 24);
    project.progress = downloadEvent("attempt-two"); p.ui.render();
    assert.equal(percent.textContent, "");
    assert.equal(bar.getAttribute("value"), null);
  });
  await test("parallel verification in one group never borrows another file's hash percentage",()=>{
    const project=checkpointProject("running","running");project.stage_id="data_download";
    project.progress={current_asset:"one",current_event:{phase:"hash",done:80,total:100},files:["one","two"].map(id=>({id,group:"shared",label:"묶음",state:"verifying",done:100,total:100,unit:"bytes"}))};
    const p=page(project),row=historyRowByLabel(p.shell, "묶음");
    assert.equal(row.querySelector(".story-history-percent").textContent,"");assert.equal(row.querySelector("progress").getAttribute("value"),null);
    assert.match(textOf(row),/0 \/ 2개/);assert.equal(p.calls.includes("resume"),false);
  });
  await test("live transfers show per-asset progress without a redundant download caption", () => {
    const project = checkpointProject("running", "running");
    project.stage_id = "data_download";
    project.progress = { phase: "download_many", label: "자료 다운로드 · 자동 1개 · 4/283개 확인", done: 4, total: 283 };
    project.download_files = [{ id: "mit_rir", label: "MIT RIR", state: "downloading", done: 30, total: 100, unit: "bytes" }];
    const p = page(project), caption = p.get("story-download-activity");
    assert.equal(caption.hidden, true);
    const row = historyRowByLabel(p.shell, "MIT RIR");
    assert.equal(row.querySelector(".story-history-percent").textContent, "30%");
    project.progress = { phase: "extract", label: "다운로드한 자료 압축 해제" };
    p.ui.render();
    assert.equal(caption.hidden, false);
    assert.match(textOf(caption), /다운로드한 자료 압축 해제/);
  });
  await test("download rows retain one stable current-work caption through extraction and connection",()=>{
    const css = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio_story.css"), "utf8");
    assert.match(css, /#story-download-activity\{text-align:center\}/);
    const project=checkpointProject("running","running");project.stage_id="data_download";
    project.download_files=[{id:"a",label:"파일",state:"waiting",unit:"bytes",done:0,total:100}];
    const p=page(project),caption=p.get("story-download-activity");
    assert.equal(caption.hidden,false);assert.match(textOf(caption),/자료 파일/);assert.equal(caption.querySelector("i").dataset.status,"active");
    project.progress={stage:"data_download",phase:"extract",label:"기존 추출 파일 검증",done:123,total:22835,unit:"files"};p.ui.render();
    assert.match(textOf(caption),/기존 추출 파일 검증.*123 \/ 22,835개/);assert.equal(caption.style.whiteSpace,"nowrap");
    project.stage_id="data_connect";project.progress={done:3,total:20,unit:"완료 단계"};p.ui.render();
    assert.match(textOf(caption),/자료를 연결/);assert.doesNotMatch(textOf(caption),/3 \/ 20/);
    project.progress={label:"공통 평가 원본 준비",done:12115.651,total:14400,unit:"seconds"};p.ui.render();
    assert.match(textOf(caption),/3시간 21분 55초 \/ 4시간/);
    project.progress={label:"자료 변환·분리",done:1,total:3,unit:"datasets"};p.ui.render();
    assert.match(textOf(caption),/1 \/ 3개 자료/);
    project.status="failed";project.error="자료 검증 실패";p.ui.render();assert.equal(caption.hidden,true);
  });
  await test("one phrase next carries successful input through the exact completed boundary and destination readiness",async()=>{
    for(const fast of [false,true]){
      const project=navigationProject("phrases");Object.assign(project,{status:"waiting_input",stage_id:"hard_phrases",input:{kind:"hard_phrases",suggestions:[]}});
      project.navigation_checkpoint.state="running";project.stages.find(s=>s.id==="hard_phrases").status="pending";
      let p,finishSave,resumes=0;
      const complete=()=>{Object.assign(project,{status:"paused",stage_id:"hard_generation",process_alive:false});delete project.input;
        project.navigation_checkpoint={policy:"explicit_story_v1",step:"phrases",state:"awaiting_next"};project.stages.find(s=>s.id==="hard_phrases").status="completed";};
      p=page(project,{input:()=>new Promise(resolve=>{finishSave=resolve;}),action:(action,state)=>{assert.equal(action,"resume");resumes++;
        project.navigation_checkpoint={policy:"explicit_story_v1",step:"hard-generation",state:"running"};project.status="running";project.process_alive=true;state.busy=false;p.ui.render();return true;}});
      const waiting=p.next.fire("click");await p.flush();await p.next.fire("click");assert.equal(p.calls.filter(c=>c==="input").length,1);
      if(fast)complete();else{project.status="running";project.process_alive=true;delete project.input;}
      finishSave(true);await waiting;await p.flush();
      if(!fast){assert.equal(resumes,0);assert.equal(p.get("story-panel").inert,true);complete();p.ui.render();await p.flush();}
      await p.settle();assert.equal(resumes,1);assert.equal(p.shell.dataset.step,"hard-generation");assert.equal(p.get("story-panel").inert,false);
      p.ui.render();await p.flush();assert.equal(resumes,1,"polling cannot replay the released boundary");
    }
  });
  await test("Previous waits for a generation pause and opens the saved phrases for editing", async () => {
    for (const [stage, restoredStatus] of [["hard_generation", "paused"], ["hard_pilot", "paused"], ["hard_generation", "waiting_input"]]) {
      const project = navigationProject("phrases"), sample = stage === "hard_pilot";
      const previous = sample ? "hard-review" : "hard-generation";
      const checkpoint = { policy: "explicit_story_v1", step: previous, state: "running" };
      Object.assign(project, { status: "pausing", stage_id: stage, process_alive: true, navigation_checkpoint: checkpoint });
      if (sample) project.settings = { hard_review_policy: "sample_before_generation_v2" };
      const p = page(project, { action: (action, state) => {
        assert.equal(action, "edit_hard_phrases");
        Object.assign(project, { status: "waiting_input", stage_id: "hard_phrases", can_edit_hard_phrases: false,
          navigation_checkpoint: { ...checkpoint, step: "phrases" },
          input: { kind: "hard_phrases", phrases: ["보아야", "나나야"], editing_paused_generation: true } });
        state.busy = false; p.ui.render(); return true;
      }, input: () => {
        Object.assign(project, { status: restoredStatus, stage_id: stage, navigation_checkpoint: checkpoint, can_edit_hard_phrases: true });
        delete project.input; p.ui.render(); return true;
      } });
      assert.equal(p.shell.dataset.step, previous);
      assert.equal(p.back.disabled, true);
      Object.assign(project, { status: "paused", process_alive: false, can_edit_hard_phrases: true });
      p.ui.render(); assert.equal(p.back.disabled, false);
      await p.back.fire("click"); await p.settle();
      assert.equal(p.shell.dataset.step, "phrases");
      assert.equal(p.next.disabled, false);
      assert.deepEqual(p.get("hard-phrase-fields").children.map(field => field.value), ["보아야", "나나야"]);
      assert.deepEqual(p.calls.filter(call => call === "edit_hard_phrases" || call === "resume"), ["edit_hard_phrases"]);
      await p.next.fire("click"); await p.settle();
      assert.equal(p.shell.dataset.step, previous);
      if (restoredStatus === "paused") assert.equal(p.operation.textContent, "재개");
      assert.equal(p.calls.includes("resume"), false, "saving the same phrases preserves the pause");
    }
  });
  await test("phrase saved intent cannot release a changed project hidden document or unrelated checkpoint stage",async()=>{
    for(const changed of ["project","hidden","stage","failure"]){
      const project=navigationProject("phrases");Object.assign(project,{status:"waiting_input",stage_id:"hard_phrases",input:{kind:"hard_phrases",suggestions:[]}});
      project.navigation_checkpoint.state="running";project.stages.find(s=>s.id==="hard_phrases").status="pending";
      let resolveSave;const p=page(project,{input:()=>new Promise(resolve=>{resolveSave=resolve;})});
      const pending=p.next.fire("click");await p.flush();
      if(changed==="project"){p.state.project={id:"other",status:"running",stage_id:"features"};p.ui.render();}
      else if(changed==="hidden")await p.visibility(true);
      else if(changed==="stage"){project.status="paused";project.stage_id="features";delete project.input;project.navigation_checkpoint.state="awaiting_next";project.stages.find(s=>s.id==="hard_phrases").status="completed";}
      resolveSave(changed!=="failure");await pending;await p.settle();assert.equal(p.calls.includes("resume"),false,changed);
      assert.equal(p.get("story-panel").inert,false,changed+" restores usable presentation");
    }
  });
  await test("numeric evaluation preparation stays active while its completed file count is being finalized",()=>{
    const project={id:"prepare",status:"running",stage_id:"development",progress:{phase:"preparation_finalizing",done:48000,total:48000,unit:"개 음원"}};
    const p=page(project),rows=p.shell.querySelector(".story-voice-counts");
    assert.match(textOf(rows),/평가 자료 등록·검증/);assert.match(textOf(rows),/48,000 \/ 48,000개 음원/);
    assert.equal(rows.querySelector(".story-progress-indicator").dataset.status,"active");assert.equal(p.next.disabled,true);
  });
  async function visit(p, step) {
    const tick = p.ticks.find(tick => tick.dataset.step === step);
    assert.ok(tick, `the current workflow includes ${step}`);
    await tick.fire("click"); await p.settle();
  }
  await test("saved chapters request preset metadata only when the scale or materials view needs it", async () => {
    let request, reads = 0;
    const p = page(historyModel({}), { prepareRecipes: () => {
      reads++;
      return new Promise(resolve => { request = () => { p.state.recipeReady = true; resolve(); }; });
    } });
    assert.equal(await p.ui.prepareInitialView(), true);
    assert.equal(reads, 0);
    for (const step of ["scale", "materials"]) {
      p.state.recipeReady = false;
      await p.ticks.find(tick => tick.dataset.step === step).fire("click"); await p.flush();
      assert.equal(p.shell.dataset.step, step);
      assert.equal(p.get("story-panel").inert, true);
      assert.ok(request);
      request(); request = null; await p.settle();
      assert.equal(p.get("story-panel").inert, false);
    }
    assert.equal(reads, 2);
  });
  await test("saved scale uses separate disabled cards and never overwrites the next model's chosen preset", async () => {
    const p = page(historyModel({})); p.state.selectedPreset = "trial";
    await visit(p, "scale");
    const group = p.get("story-saved-presets"), radios = group.querySelectorAll("input");
    assert.equal(group.parentElement.hidden, false); assert.equal(radios.length, 3);
    assert.ok(group.children.every(card => card.classList.contains("story-choice-card") && card.children[0].type === "radio"));
    assert.equal(radios.filter(radio => radio.checked).length, 1);
    assert.equal(radios.find(radio => radio.checked).value, "production");
    assert.ok(radios.every(radio => radio.disabled)); assert.equal(p.state.selectedPreset, "trial");
    p.ui.render(); assert.equal(p.state.selectedPreset, "trial");
    p.state.project = null; p.ui.render(); await advanceSetup(p, "tools", "scale");
    assert.equal(p.shell.dataset.step, "scale"); assert.equal(group.parentElement.hidden, true);
    assert.equal(p.state.selectedPreset, "trial");
  });
  await test("work chapters use the same rows and preserve recorded quantities", async () => {
    const p = page(historyModel({ "positive-generation": { completed: true, rows: [
      { label: "학습용 호출어", done: 1000, total: 1000, unit: "개" },
      { label: "공통 평가용 호출어", done: 2000, total: 2000, unit: "개" },
    ] }, "hard-generation": { completed: true, rows: [
      { label: "유사어 음성", done: 120, total: 120, unit: "개" },
    ] } }));
    p.state.project.settings = { originals: 100000 };
    await visit(p, "positive-generation");
    const counts = p.shell.querySelector(".story-voice-counts"), rows = counts.querySelectorAll(".story-voice-row");
    assert.equal(counts.hidden, false); assert.equal(rows.length, 2);
    assert.ok(textOf(counts).includes("1,000 / 1,000개"));
    assert.ok(!textOf(counts).includes("100,000"));
    assert.ok(rows.every(row => row.querySelector("progress").value === row.querySelector("progress").max));
    await visit(p, "hard-generation");
    assert.equal(counts.querySelectorAll(".story-voice-row").length, 1);
    assert.ok(textOf(counts).includes("120 / 120개"));
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
    for (const [stage, label] of [["positives", "호출어 음성"], ["hard_generation", "유사어 음성"],
      ["features", "특징 추출"], ["training", "모델 학습"], ["evaluation", "성능 평가"]]) {
      const active = page({ id: stage, status: "running", stage_id: stage, progress: { unit: "개", done: 13, total: 20 } });
      const activeRows = active.shell.querySelector(".story-voice-counts");
      assert.equal(activeRows.hidden, false);
      assert.ok(textOf(activeRows).includes(label));
      assert.ok(textOf(activeRows).includes("13 / 20개"));
      assert.equal(activeRows.querySelector("progress").value, 13);
    }
  });
  await test("evaluation explains planned calls and separates recorded non-call sessions", async () => {
    const policy = {
      validation_originals: 2000, noise_sections: 1, snrs: [20, 15, 10, 5, 0],
      positive_sessions: 48000, dry_positive_sessions: 32000,
      reverb_originals: 500, reverb_positive_sessions: 16000, reverb_groups: ["moderate", "strong"],
      development_negative_target_hours: [30, 30],
    };
    const project = historyModel({ evaluation: { completed: true, rows: [
      { label: "성능 평가 음원", done: 48137, total: 48137 },
    ] } });
    project.evaluation_policy = policy;
    project.results.negative_hours = 30.00058;
    project.results.common_evaluation_assessment = { reverberation: { condition_counts: [
      { samples: 32000 }, { samples: 8000 }, { samples: 8000 },
    ] } };
    const p = page(project), details = p.shell.querySelector(".story-evaluation-plan");
    await visit(p, "evaluation");
    assert.equal(details.hidden, false);
    assert.match(textOf(details), /호출 감지 시험 48,000회/);
    assert.match(textOf(details), /2,000개 × 16조건 = 32,000회/);
    assert.match(textOf(details), /500개 × 16조건 × 2강도 = 16,000회/);
    assert.match(textOf(details), /약 30시간/);
    assert.match(textOf(details), /오호출 시험 137세션/);
    await visit(p, "training");
    assert.equal(details.hidden, true);

    const earlier = page({ id: "earlier", status: "running", stage_id: "evaluation",
      evaluation_policy: { ...policy, validation_originals: 5000, positive_sessions: 112000,
        dry_positive_sessions: 80000, reverb_originals: 1000, reverb_positive_sessions: 32000 },
      progress: { done: 10, total: 112137, unit: "개" },
    });
    const earlierDetails = textOf(earlier.shell.querySelector(".story-evaluation-plan"));
    assert.match(earlierDetails, /호출 감지 시험 계획 112,000회/);
    assert.doesNotMatch(earlierDetails, /48,000|137세션/);
  });
  await test("whole hard listening is available only from representative review with generated clips", async () => {
    const project = historyModel({ phrases: { completed: true, phrases: ["언제나"] } });
    project.hard_preview = { url: "/hard-preview/?project=history", count: 20 };
    const p = page(project);
    const actions = p.shell.querySelector(".story-listen-actions");
    for (const step of ["phrases", "hard-generation", "hard-review"]) {
      await visit(p, step);
      assert.equal(p.shell.dataset.layout, "standard");
      assert.equal(actions.hidden, step !== "hard-review");
    }
    await actions.children[0].fire("click");
    assert.ok(p.calls.includes("preview:hard"));
    project.hard_preview.count = 0; p.ui.render();
    assert.equal(actions.hidden, true);
  });
  await test("missing quantities and uncompleted backend stages never receive invented 100 percent", async () => {
    const p = page(historyModel({ features: { completed: true, rows: [{ label: "호출어 특징", done: null, total: 500000, unit: "rows" }] } }));
    await visit(p, "features");
    const history = p.shell.querySelector(".story-history");
    assert.equal(history.querySelectorAll("progress").length, 0);
    assert.ok(!textOf(history).includes("100%"));
    p.state.project.stages.find(stage => stage.id === "features").status = "failed";
    p.state.project.completed_stage_views.features.rows[0].done = 500000; p.ui.render();
    assert.equal(history.querySelectorAll("progress").length, 0);
    assert.ok(textOf(history).includes("저장된 세부 내역이 없습니다"));
  });
  await test("historical downloads show receipt files and known bytes without inventing network rates", async () => {
    const files = [
      { id: "archive", label: "음악", group: "music", group_label: "소음 증강 음악", state: "complete", unit: "bytes", done: 2048, total: 2048 },
      { id: "rir", label: "학습용 잔향", group: "rir", state: "reused", unit: "bytes", done: 1024, total: 1024 },
    ];
    const p = page(historyModel({ downloads: { completed: true, progress: { files } } }));
    await visit(p, "downloads");
    const history = p.shell.querySelector(".story-history");
    assert.ok(textOf(history).includes("소음 증강 음악")); assert.ok(textOf(history).includes("2 KiB"));
    assert.ok(historyRowByLabel(history, "소음 증강 음악").querySelector("progress"));
    assert.ok(historyRowByLabel(history, "학습용 잔향").querySelector("progress"));
    assert.ok(!textOf(history).includes("/초"));
    assert.equal(p.next.textContent, "다음");
  });
  await test("automatic historical source downloads keep measured hours, partial progress and unknown quantities", async () => {
    const files = [
      { id: "automatic:yodas", label: "한국어 비호출 · YODAS2", group: "automatic:korean", state: "complete", unit: "hours", done: 1500.25, total: 1500 },
      { id: "automatic:mls", label: "다국어 낭독 · MLS", group: "automatic:generic", state: "unknown", unit: "hours", done: 750, total: 1500 },
      { id: "automatic:missing", label: "추가 음원", group: "automatic:unknown", state: "unknown", unit: "hours", done: null, total: 500 },
    ];
    const p = page(historyModel({ downloads: { completed: true, progress: { files } } }));
    await visit(p, "downloads");
    const history = p.shell.querySelector(".story-history"), rows = files.map(file => historyRowByLabel(history, file.label));
    assert.ok(textOf(rows[0]).includes("1,500.25 / 1,500시간"));
    assert.ok(textOf(rows[1]).includes("750 / 1,500시간"));
    assert.ok(textOf(rows[1]).includes("50%"));
    assert.ok(textOf(rows[2]).includes("— / 500시간"));
    assert.equal(rows[2].querySelector("progress").hidden, true);
    assert.ok(!textOf(history).includes("1 / 1개"));
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("saved phrases and representative review decisions remain read-only and listening makes no approval request", async () => {
    const p = page(historyModel({
      phrases: { completed: true, phrases: ["언니야", "언제나"] },
      "hard-review": { completed: true, review: { items: [
        { id: "one", text: "언니야", audio: "/history-stage/audio/0?project=history&step=hard-review", decision: "keep", listened: true },
        { id: "two", text: "언제나", audio: "https://outside.invalid/audio.wav", decision: "discard", listened: false },
      ] } },
    }));
    await visit(p, "phrases");
    const phrases = p.shell.querySelector(".story-saved-phrases").querySelectorAll("input");
    assert.deepEqual(phrases.map(field => field.value), ["언니야", "언제나"]);
    assert.ok(phrases.every(field => field.readOnly));
    await visit(p, "hard-review");
    const review = p.shell.querySelector(".story-saved-review"), choices = review.querySelectorAll("input"), audio = review.querySelector("audio"), select = review.querySelector("select");
    assert.equal(review.hidden, false); assert.equal(choices.find(choice => choice.checked).value, "keep");
    assert.ok(choices.every(choice => choice.disabled));
    assert.ok(audio.getAttribute("src").startsWith("/history-stage/audio/"));
    select.value = "1"; await select.fire("change");
    assert.equal(choices.find(choice => choice.checked).value, "discard");
    assert.equal(audio.getAttribute("src"), null, "foreign history media is rejected");
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  const savedListeningView = () => ({ "hard-review": { completed: true, review: { items: [
    { id: "one", text: "언니야", audio: "/history-stage/audio/0?project=history&step=hard-review", decision: "keep", listened: true },
    { id: "two", text: "언제나", audio: "/history-stage/audio/1?project=history&step=hard-review", decision: "discard", listened: false },
  ] } } });
  await test("completed pronunciation and hard reviews use the compact picker with read-only verdicts", async () => {
    const view = savedListeningView()["hard-review"];
    const model = historyModel({ pronunciation: view, "hard-review": view });
    const original = JSON.stringify(view);
    const p = page(model, { compactListening: true }); await visit(p, "pronunciation");
    const review = p.shell.querySelector(".story-saved-review");
    const picker = review.querySelector(".voice-picker-button"), popup = review.querySelector(".voice-picker-popup");
    const player = review.querySelector(".listen-player"), decisions = player.querySelector(".listen-decisions");
    picker.getBoundingClientRect = () => ({ left: 80, top: 200, bottom: 240, width: 600 });
    assert.equal(review.querySelector(".listen-sidebar").hidden, true);
    assert.equal(player.querySelector(".listen-volume").hidden, false);
    await picker.fire("click");
    assert.equal(popup.hidden, false);
    const option = popup.querySelectorAll(".voice-picker-option")[1];
    assert.equal(option.querySelector(".voice-picker-badge").textContent, "부적합");
    await option.fire("click");
    assert.equal(popup.hidden, true);
    assert.equal(player.querySelector(".listen-position").textContent, "2 / 2");
    assert.equal(decisions.querySelectorAll("input").find(input => input.checked).value, "discard");
    assert.equal(decisions.hidden, true, "completed call review uses saved badges instead of inert verdict buttons");
    assert.ok(decisions.querySelectorAll("input").every(input => input.disabled));
    assert.equal(player.querySelector("audio").playCalls, 0);
    await visit(p, "hard-review");
    assert.equal(decisions.hidden, false, "legacy hard history retains its read-only verdicts");
    await picker.fire("click");
    const hardOption = popup.querySelectorAll(".voice-picker-option")[0];
    assert.equal(hardOption.querySelector(".voice-picker-group").textContent, "언니야");
    await hardOption.fire("click");
    assert.equal(player.querySelector(".listen-position").textContent, "1 / 2");
    assert.equal(decisions.querySelectorAll("input").find(input => input.checked).value, "keep");
    assert.equal(JSON.stringify(view), original);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("saved review advances to the next clip and stops at the end without changing decisions", async () => {
    const model = historyModel(savedListeningView()), original = JSON.stringify(model.completed_stage_views);
    const p = page(model); await visit(p, "hard-review");
    const player = p.shell.querySelector(".listen-player"), audio = player.querySelector(".listen-audio"), select = player.querySelector(".listen-select");
    const [previous, toggle, next] = player.querySelector(".listen-controls").children;
    assert.equal(previous.disabled, true); assert.equal(next.disabled, false); assert.equal(toggle.textContent, "재생");
    await toggle.fire("click"); assert.equal(audio.paused, false); assert.equal(toggle.textContent, "일시정지");
    const loads = audio.loadCalls; p.ui.render(); p.ui.render(); assert.equal(audio.loadCalls, loads, "polling preserves playback");
    audio.paused = true; audio.ended = true; await audio.fire("pause"); await audio.fire("ended");
    assert.equal(audio.paused, false); assert.ok(audio.getAttribute("src").includes("/1?"));
    assert.equal(previous.disabled, false); assert.equal(next.disabled, true);
    audio.paused = true; audio.ended = true; await audio.fire("pause"); await audio.fire("ended");
    assert.ok(audio.getAttribute("src").includes("/1?")); assert.equal(audio.paused, true); assert.equal(toggle.textContent, "재생");
    audio.ended = true; await audio.fire("ended"); assert.ok(audio.getAttribute("src").includes("/1?"), "ended cannot restart after user pause");
    await next.fire("click"); assert.ok(audio.getAttribute("src").includes("/1?"), "manual next is bounded at the last item");
    select.value = "0"; await select.fire("change"); assert.ok(audio.getAttribute("src").includes("/0?")); assert.equal(audio.paused, true, "paused selection stays paused");
    await audio.play(); assert.equal(toggle.textContent, "일시정지"); audio.pause(); assert.equal(toggle.textContent, "재생");
    assert.equal(JSON.stringify(model.completed_stage_views), original);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("saved review late play is stopped after navigation or model exchange and media sources are released", async () => {
    for (const change of ["navigation", "model"]) {
      let finishPlay;
      const p = page(historyModel(savedListeningView()), { playMedia: (_audio, perform) => new Promise(resolve => { finishPlay = async () => { await perform(); resolve(); }; }) });
      await visit(p, "hard-review");
      const player = p.shell.querySelector(".listen-player"), audio = player.querySelector(".listen-audio"), toggle = player.querySelector(".listen-controls").children[1];
      const starting = toggle.fire("click");
      if (change === "navigation") { await p.back.fire("click"); await p.settle(); }
      else { p.state.project = { ...historyModel(savedListeningView()), id: "other-model" }; p.ui.render(); }
      assert.equal(audio.getAttribute("src"), null); assert.equal(audio.paused, true);
      await finishPlay(); await starting;
      assert.equal(audio.paused, true, "a delayed play completion cannot restart hidden audio");
      assert.equal(audio.getAttribute("src"), null);
      assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
    }
  });
  await test("saved review tab hiding releases media and explicit play restores it without autoplay on return", async () => {
    const p = page(historyModel(savedListeningView())); await visit(p, "hard-review");
    const player = p.shell.querySelector(".listen-player"), audio = player.querySelector(".listen-audio"), toggle = player.querySelector(".listen-controls").children[1];
    await toggle.fire("click"); assert.equal(audio.paused, false);
    await p.visibility(true); assert.equal(audio.paused, true); assert.equal(audio.getAttribute("src"), null);
    await p.visibility(false); p.ui.render(); assert.equal(audio.getAttribute("src"), null); assert.equal(toggle.textContent, "재생");
    assert.equal(toggle.disabled, false); await toggle.fire("click");
    assert.ok(audio.getAttribute("src")); assert.equal(audio.paused, false);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("completed review uses a five-item list with saved verdict badges and preserves playing selection across pages", async () => {
    const items = Array.from({ length: 12 }, (_, i) => ({ id: "saved-" + i, text: "문구 " + (i + 1),
      audio: `/history-stage/audio/${i}?project=history&step=hard-review`, decision: i % 2 ? "discard" : "keep", listened: i % 2 === 0 }));
    const model = historyModel({ "hard-review": { completed: true, review: { items } } }), original = JSON.stringify(model.completed_stage_views);
    const p = page(model); await visit(p, "hard-review");
    const review = p.shell.querySelector(".story-saved-review"), workspace = review.querySelector(".listen-workspace");
    const rows = workspace.querySelectorAll(".voice-row"), [pageBack, pageText, pageNext] = workspace.querySelector(".listen-list-pager").children;
    const player = workspace.querySelector(".listen-player"), audio = player.querySelector("audio"), toggle = player.querySelector(".listen-controls").children[1];
    assert.equal(rows.length, 12); assert.equal(rows.filter(row => !row.hidden).length, 5);
    assert.equal(pageText.textContent, "1 / 3"); assert.equal(pageBack.disabled, true);
    assert.equal(rows[0].querySelector(".listen-badge").textContent, "정상");
    assert.equal(rows[1].querySelector(".listen-badge").textContent, "제외");
    assert.equal(rows[6].dataset.number, "7"); assert.ok(rows[6].getAttribute("aria-label").includes("음원 7 · 문구 7 · 정상"));
    assert.equal(player.querySelector("select").hidden, true);
    assert.equal(review.querySelector(".listen-completion").textContent, "검수한 음원과 저장된 판정을 확인할 수 있습니다.");
    await toggle.fire("click"); const playCount = audio.playCalls;
    await pageNext.fire("click");
    assert.equal(pageText.textContent, "2 / 3"); assert.equal(rows.filter(row => !row.hidden).length, 5);
    assert.ok(audio.getAttribute("src").includes("/0?")); assert.equal(audio.playCalls, playCount, "paging only browses the list");
    await rows[6].fire("click"); assert.ok(audio.getAttribute("src").includes("/6?")); assert.equal(audio.paused, false);
    assert.equal(rows[6].getAttribute("aria-current"), "true");
    p.ui.render(); assert.equal(workspace.querySelectorAll(".voice-row")[6], rows[6]);
    assert.equal(pageText.textContent, "2 / 3");
    await pageNext.fire("click"); assert.equal(pageText.textContent, "3 / 3"); assert.equal(rows.filter(row => !row.hidden).length, 2);
    assert.equal(pageNext.disabled, true); await pageBack.fire("click"); assert.equal(pageText.textContent, "2 / 3");
    assert.equal(JSON.stringify(model.completed_stage_views), original);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  const preflightProject=()=>{
    const project=navigationProject('voices');project.settings={pilot_review_policy:'one_per_voice_before_and_after_generation_v4'};
    project.stage_id='pilot_review';project.status='waiting_review';project.process_alive=false;
    project.navigation_checkpoint={policy:'explicit_story_v1',step:'call-pilot-review',state:'running'};
    project.review={kind:'pilot',queue_sha256:'current-preflight-queue',url:'/review/?project=navigated&queue=pilot'};
    project.stages.find(stage=>stage.id==='pilot').status='completed';
    return project;
  };
  async function wirePreflight(p,project){
    const frame=p.get('review-frame'),posted=[],queue=project.review.queue_sha256;let request;
    const source={postMessage(data){posted.push(data);if(data.type==='koww:preflight-status')request=data.request;}};
    frame.contentWindow=source;p.ui.render();await frame.fire('load');frame.fakeLoadedUrl=frame.getAttribute('src');
    const send=data=>p.windowEvent('message',{origin:'http://studio.local',source,data:{project:project.id,queue_sha256:queue,request,...data}});
    return {frame,source,posted,send,get request(){return request;}};
  }
  await test('pilot transition waits for an absent queue, then for the matching frame, list and geometry',async()=>{
    const project=preflightProject(),queue=project.review;
    Object.assign(project,{stage_id:'pilot',status:'running',process_alive:true});delete project.review;
    const p=page(project);assert.equal(p.shell.dataset.step,'call-pilot-generation');
    project.stage_id='pilot_review';p.ui.render();await p.flush();p.advanceTimer();await p.flush();
    assert.equal(p.get('story-panel').inert,true);
    assert.equal(p.shell.querySelector('.story-loader').hidden,false);
    assert.equal(p.get('review-section').hidden,true);
    Object.assign(project,{status:'waiting_review',process_alive:false,review:{...queue,queue_id:'hard'}});
    p.ui.render();await p.flush();assert.equal(p.get('story-panel').inert,true);
    Object.assign(project,{status:'waiting_review',process_alive:false,review:queue});
    const wire=await wirePreflight(p,project);await p.flush();
    const chapter=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1);assert.ok(chapter);
    await wire.send({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});
    await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false,queue_sha256:'stale'});
    await wire.send({type:'koww:preflight-height',height:170});
    assert.equal(p.get('story-panel').inert,true);
    await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false});
    await p.settle();assert.equal(p.get('story-panel').inert,false);assert.equal(p.get('review-section').hidden,false);
    assert.equal(p.next.disabled,true);
    assert.equal(p.calls.some(call=>['start','resume','input'].includes(call)),false);
  });
  await test('preflight picker reserves its baseline and expands only after trusted current-frame requests',async()=>{
    for(const height of [453,900]){
      const project=preflightProject(),p=page(project,{height}),wire=await wirePreflight(p,project),section=p.get('review-section');
      section.getBoundingClientRect=()=>({top:height===453?174:315,height:170});
      const layouts=[];wire.frame.getBoundingClientRect=()=>{const open=p.shell.classList.contains('preflight-picker-open');layouts.push(open);return {height:open?500:170};};
      const post=wire.source.postMessage;wire.source.postMessage=data=>{if(data.type==='koww:preflight-picker-closed'){
        assert.equal(p.shell.classList.contains('preflight-picker-open'),false);assert.equal(layouts.at(-1),false,'the baseline is applied before the close acknowledgement is sent');
      }post(data);};
      await wire.send({type:'koww:preflight-height',height:170});const before=p.calls.length;
      const message={type:'koww:preflight-picker',open:true,picker_request:1};
      for(const invalid of [{origin:'https://wrong',source:wire.source,data:{project:project.id,queue_sha256:'current-preflight-queue',request:wire.request,...message}},
        {origin:'http://studio.local',source:{},data:{project:project.id,queue_sha256:'current-preflight-queue',request:wire.request,...message}}])await p.windowEvent('message',invalid);
      for(const change of [{project:'other'},{queue_sha256:'old'},{request:wire.request-1},{picker_request:0},{picker_request:'1'},{open:'true'}])await wire.send({...message,...change});
      assert.equal(p.shell.classList.contains('preflight-picker-open'),false);await wire.send(message);
      const expanded=parseFloat(p.shell.style['--preflight-picker-height']);assert.ok(expanded>300&&expanded<=720);
      assert.equal(p.shell.style['--preflight-content-height'],'170px');assert.deepEqual(layouts,[true]);
      assert.deepEqual(JSON.parse(JSON.stringify(wire.posted.at(-1))),{type:'koww:preflight-picker-ready',project:project.id,queue_sha256:'current-preflight-queue',request:wire.request,picker_request:1});
      await wire.send({type:'koww:preflight-height',height:expanded});assert.equal(p.shell.style['--preflight-content-height'],'170px','overlay resize cannot feed its height into the reserved body');
      await wire.send({...message,picker_request:2});await wire.send({...message,open:false,picker_request:1});assert.equal(p.shell.classList.contains('preflight-picker-open'),true,'an old close cannot collapse a newer popup');
      await wire.send({...message,open:false,picker_request:2});assert.equal(p.shell.classList.contains('preflight-picker-open'),false);
      assert.equal(wire.posted.at(-1).type,'koww:preflight-picker-closed');assert.equal(wire.posted.at(-1).picker_request,2);
      assert.equal(p.shell.style['--preflight-content-height'],'170px');assert.equal(p.calls.length,before,'open and close only resize; they never pause audio or mutate the project');
      await wire.send(message);await wire.frame.fire('load');assert.equal(p.shell.classList.contains('preflight-picker-open'),false);
      await wire.send(message);await p.back.fire('click');assert.equal(p.shell.classList.contains('preflight-picker-open'),false,'navigation collapses the overlay immediately');await p.settle();
    }
  });
  await test('preflight close acknowledges an already-applied parent pause and never collapses a newer popup',async()=>{
    const project=preflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-height',height:170});await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});
    await wire.send({type:'koww:preflight-picker',open:true,picker_request:1});await p.next.fire('click');
    assert.equal(p.shell.classList.contains('preflight-picker-open'),false,'the parent pause can apply before the child receives its outside-close gesture');
    await wire.send({type:'koww:preflight-picker',open:false,picker_request:1});
    assert.equal(wire.posted.at(-1).type,'koww:preflight-picker-closed');
    await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:true});await p.settle();
    await wire.send({type:'koww:preflight-picker',open:true,picker_request:2});
    const count=wire.posted.length;await wire.send({type:'koww:preflight-picker',open:false,picker_request:1});
    assert.equal(p.shell.classList.contains('preflight-picker-open'),true);assert.equal(wire.posted.length,count,'old close cannot receive an acknowledgement for the newer open');
    await wire.send({type:'koww:preflight-picker',open:false,picker_request:2});assert.equal(wire.posted.at(-1).picker_request,2);
  });
  await test('preflight closing a retained hidden frame still receives its applied-close acknowledgement',async()=>{
    const project=preflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-picker',open:true,picker_request:1});await p.back.fire('click');await p.settle();
    assert.equal(p.shell.dataset.step,'call-pilot-generation');assert.equal(p.shell.classList.contains('preflight-picker-open'),false);
    await wire.send({type:'koww:preflight-picker',open:false,picker_request:1});
    assert.equal(wire.posted.at(-1).type,'koww:preflight-picker-closed','navigation does not strand the retained child in an expanded closing state');
    assert.equal(p.calls.includes('resume'),false);
  });
  function finishPreflightProcessing(project){
    project.status='paused';project.stage_id='positives';project.process_alive=false;delete project.review;
    project.navigation_checkpoint={policy:'explicit_story_v1',step:'call-pilot-review',state:'awaiting_next'};
    project.stages.find(stage=>stage.id==='pilot_review').status='completed';
  }
  await test('preflight one next carries finalization through processing and its explicit chapter release without showing history',async()=>{
    for(const fast of [false,true]){
      const project=preflightProject();let p,resumes=0;
      p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(action,state)=>{
        assert.equal(action,'resume');resumes++;
        if(resumes===1){
          state.project.status='running';state.project.process_alive=true;
          if(fast)finishPreflightProcessing(state.project);
        }else{
          assert.equal(state.project.navigation_checkpoint.step,'call-pilot-review');
          state.project.status='running';state.project.stage_id='positives';state.project.process_alive=true;
          state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'positive-generation',state:'running'};
        }
        state.busy=false;p.ui.render();
      }});
      const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      assert.equal(p.get('story-panel').inert,true);await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();
      if(!fast){assert.equal(resumes,1);assert.equal(p.shell.querySelector('.story-loader').hidden,false);assert.equal(p.get('story-panel').inert,true);
        finishPreflightProcessing(project);p.ui.render();await p.flush();}
      assert.equal(resumes,2);assert.equal(p.get('story-panel').inert,true,'the intermediate saved-review page stays behind the same transition');
      await p.settle();assert.equal(p.shell.dataset.step,'positive-generation');assert.equal(p.get('story-panel').inert,false);assert.equal(p.shell.querySelector('.story-loader').hidden,true);
      p.ui.render();await p.flush();assert.equal(resumes,2,'polling never repeats either completed action');
    }
  });
  await test('preflight own finalization survives a committed-state poll before the retained child response arrives',async()=>{
    const project=preflightProject();let p,resumes=0;
    p=page(project,{action:(_action,state)=>{resumes++;state.project.status='running';state.project.process_alive=true;
      if(resumes===2){state.project.stage_id='positives';state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'positive-generation',state:'running'};}
      state.busy=false;p.ui.render();}});
    const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    assert.equal(p.ui.retainReviewFrame(project),true);
    project.status='paused';delete project.review;p.ui.render();await p.flush();
    assert.equal(p.ui.retainReviewFrame(project),true,'the committed input poll retains the exact source until its pending POST can reply');
    assert.equal(resumes,0,'a changed poll alone is not a substitute for the finalization response');
    assert.equal(p.ui.retainReviewFrame({...project,id:'other'}),false);assert.equal(p.ui.retainReviewFrame({...project,review:{queue_sha256:'other'}}),false);
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();assert.equal(resumes,1);
    assert.equal(p.ui.retainReviewFrame(project),false,'the iframe can be cleared as soon as the acknowledged action enters processing');
    finishPreflightProcessing(project);p.ui.render();await p.flush();assert.equal(resumes,2);await p.settle();
    assert.equal(p.shell.dataset.step,'positive-generation');assert.equal(p.get('story-panel').inert,false);
  });
  await test('preflight successful resume RPCs wait for fresh polls and never replay while the snapshot remains paused',async()=>{
    const project=preflightProject();let p,resumes=0;
    p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{resumes++;state.busy=false;p.ui.render();return true;}});
    const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();
    p.ui.render();p.ui.render();await p.flush();assert.equal(resumes,1);assert.equal(p.get('story-panel').inert,true);assert.equal(p.calls.some(value=>value.startsWith('message:')),false);
    finishPreflightProcessing(project);p.ui.render();await p.flush();assert.equal(resumes,2);
    p.ui.render();p.ui.render();await p.flush();assert.equal(resumes,2,'the boundary release is not retried against an old poll');
    project.status='running';project.stage_id='positives';project.process_alive=true;project.navigation_checkpoint={policy:'explicit_story_v1',step:'positive-generation',state:'running'};
    p.ui.render();await p.settle();assert.equal(p.shell.dataset.step,'positive-generation');assert.equal(resumes,2);
  });
  await test('preflight retained finalization errors restore the review without waiting for the watchdog',async()=>{
    const project=preflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    project.status='paused';delete project.review;p.ui.render();assert.equal(p.ui.retainReviewFrame(project),true);
    await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:true});await p.settle();
    assert.equal(p.calls.includes('resume'),false);assert.equal(p.get('story-panel').inert,false);assert.equal(p.ui.retainReviewFrame(project),false);
    assert.ok(p.now()<45000,'a captured error response cancels immediately instead of waiting for the long worker timeout');
  });
  await test('preflight failed resume RPC cancels its intent instead of continuing to generation',async()=>{
    const project=preflightProject();const p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{state.busy=false;return false;}});
    const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();
    assert.equal(p.calls.filter(value=>value==='resume').length,1);assert.equal(p.get('story-panel').inert,false);assert.equal(p.shell.dataset.step,'call-pilot-review');
    assert.equal(p.calls.some(value=>value.startsWith('message:')),true);finishPreflightProcessing(project);p.ui.render();await p.flush();
    assert.equal(p.calls.filter(value=>value==='resume').length,1,'a later checkpoint does not revive a failed intent');
  });
  await test('preflight stale completed checkpoint cannot resume an unrelated generation stage',async()=>{
    for(const stage of ['hard_generation','training','evaluation']){
      const project=preflightProject(),p=page(project,{refresh:state=>{finishPreflightProcessing(state.project);state.project.stage_id=stage;}});
      const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();
      assert.equal(p.calls.includes('resume'),false,stage+' is outside the explicit next route');assert.equal(p.get('story-panel').inert,false);
    }
  });
  await test('preflight automatic worker and retry results keep the same next intent within their authorized route',async()=>{
    for(const workflow of ['none','retry']){
      const project=preflightProject();let p;
      p=page(project,{refresh:state=>{delete state.project.review;state.project.status='running';state.project.process_alive=true;
        if(workflow==='retry'){state.project.stage_id='pilot';state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'call-pilot-generation',state:'running'};}},
        action:(_action,state)=>{state.project.status='running';state.project.stage_id='positives';state.project.process_alive=true;
          state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'positive-generation',state:'running'};state.busy=false;p.ui.render();}});
      const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await wire.send({type:'koww:preflight-finalized',workflow_action:workflow});await p.flush();assert.equal(p.calls.includes('resume'),false,'finalization never restarts an already running worker');
      if(workflow==='none'){finishPreflightProcessing(project);p.ui.render();await p.flush();assert.equal(p.calls.filter(value=>value==='resume').length,1);}
      await p.settle();assert.equal(p.shell.dataset.step,workflow==='retry'?'call-pilot-generation':'positive-generation');
      assert.equal(p.calls.filter(value=>value==='resume').length,workflow==='retry'?0:1,'unsuitable voices only route to sample regeneration');
    }
  });
  await test('preflight retry rejects blocked or off-route changed queues before finishing its next intent',async()=>{
    for(const variant of ['failed','error','input','migration','positive_review']){
      const project=preflightProject(),p=page(project,{refresh:state=>{
        const changed=state.project;changed.status='paused';changed.process_alive=false;
        changed.review={kind:variant==='positive_review'?'positive':'pilot',queue_sha256:'changed-queue',url:'/review/?project=navigated&queue=changed'};
        if(variant==='positive_review'){changed.stage_id='positive_review';changed.navigation_checkpoint={policy:'explicit_story_v1',step:'positive-generation',state:'running'};}
        else if(variant==='failed')changed.status='failed';
        else if(variant==='error')changed.error='generation error';
        else if(variant==='input')changed.input={kind:'generation_policy'};
        else changed.migration={reason:'configuration changed'};
      }});
      const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await wire.send({type:'koww:preflight-finalized',workflow_action:'retry'});await p.settle();
      assert.equal(p.calls.includes('resume'),false,variant+' never starts another job from the retry route');
      assert.equal(p.get('story-panel').inert,false);assert.equal(p.calls.some(value=>value.startsWith('message:')),true,variant+' cancels instead of marking its target ready');
    }
  });
  await test('preflight finalization timeout clears its intent and rejects later acknowledgements or completed polls',async()=>{
    const project=preflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');await p.settle();
    assert.equal(p.calls.includes('resume'),false);assert.equal(p.get('story-panel').inert,false);assert.equal(p.ui.retainReviewFrame(project),false);
    assert.ok(p.now()>=45000);assert.equal(p.calls.some(value=>value.startsWith('message:')),true);
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});finishPreflightProcessing(project);p.ui.render();await p.flush();
    assert.equal(p.calls.includes('resume'),false,'a timed-out explicit next cannot be revived by a late response or poll');
  });
  await test('preflight fast cached positive result accepts the destination queue without treating it as stale review input',async()=>{
    const project=preflightProject();let p;
    p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{
      state.project.status='paused';state.project.stage_id='positive_review';state.project.process_alive=false;
      state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'positive-generation',state:'awaiting_next'};
      state.project.review={kind:'positive',queue_sha256:'new-positive-queue',url:'/review/?project=navigated&queue=positive'};
      state.project.stages.find(stage=>stage.id==='pilot_review').status='completed';state.project.stages.find(stage=>stage.id==='positives').status='completed';
      state.busy=false;p.ui.render();
    }});
    const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();
    assert.equal(p.shell.dataset.step,'positive-generation');assert.equal(p.calls.filter(value=>value==='resume').length,1);
    assert.equal(p.calls.some(value=>value.startsWith('message:')),false,'a real destination queue is not reported as a stale preflight error');
  });
  await test('preflight release never starts from ordinary polls and cancels an explicit intent after failure or project exchange',async()=>{
    const untouched=preflightProject(),plain=page(untouched);finishPreflightProcessing(untouched);plain.ui.render();await plain.flush();
    assert.equal(plain.calls.includes('resume'),false,'a reloaded completed boundary still requires a real next click');
    assert.equal(plain.next.disabled,false);await plain.next.fire('click');assert.equal(plain.calls.filter(value=>value==='resume').length,1,'the existing paused boundary still releases from one explicit next click');
    for(const change of ['failure','project','queue','reload']){
      const project=preflightProject(),p=page(project),wire=await wirePreflight(p,project);
      await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      if(change==='failure'){project.status='failed';project.error='worker stopped';p.ui.render();}
      else if(change==='project'){p.state.project={...project,id:'other-model'};p.ui.render();}
      else if(change==='queue'){project.review={...project.review,queue_sha256:'new-queue'};p.ui.render();}
      else await wire.frame.fire('load');
      await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();
      assert.equal(p.calls.includes('resume'),false,change+' cannot reuse an old finalization');assert.equal(p.get('story-panel').inert,false);
    }
  });
  await test('preflight footer next waits for saved drafts and sends explicit finalization only',async()=>{
    const project=preflightProject(),p=page(project),frame=p.get('review-frame'),posted=[];
    frame.contentWindow={postMessage:data=>posted.push(data)};p.ui.render();assert.equal(p.shell.dataset.step,'call-pilot-review');assert.equal(p.next.disabled,true);
    await frame.fire('load');frame.fakeLoadedUrl=frame.getAttribute('src');const request=posted.at(-1).request;
    assert.match(p.get('story-subtitle').textContent,/호출어 샘플을 연속으로 듣고, 문제가 있는 음원만 부적합으로 표시하세요\.\n재생된 음원은 자동으로 적합 판정하며, 부적합한 음원만 모아 다시 생성합니다\./);
    const message={origin:'http://studio.local',source:frame.contentWindow,data:{type:'koww:preflight-state',project:project.id,queue_sha256:project.review.queue_sha256,request,ready:true,pending:false,error:false}};
    for(const invalid of [{...message,origin:'https://other.invalid'},{...message,source:{}},{...message,data:{...message.data,project:'other'}},
      {...message,data:{...message.data,queue_sha256:'old'}},{...message,data:{...message.data,request:request-1}},{...message,data:{...message.data,pending:'false'}}])await p.windowEvent('message',invalid);
    assert.equal(p.next.disabled,true);await p.next.fire('click');assert.equal(posted.filter(row=>row.type==='koww:preflight-finalize').length,0);
    await p.windowEvent('message',{...message,data:{...message.data,pending:true}});assert.equal(p.next.disabled,true);
    await p.windowEvent('message',message);assert.equal(p.next.disabled,false);
    await p.next.fire('click');await p.next.fire('click');assert.equal(posted.filter(row=>row.type==='koww:preflight-finalize').length,1);
    assert.equal(p.next.disabled,true);assert.equal(p.calls.includes('resume'),false,'playing or clicking next does not bypass child finalize');
    await p.windowEvent('message',{...message,data:{...message.data,type:'koww:preflight-state',ready:false,pending:false,error:true}});
    await p.settle();
    await p.windowEvent('message',message);assert.equal(p.next.disabled,false,'a recovered failed finalize can be retried explicitly');
    await p.windowEvent('message',{...message,data:{type:'koww:preflight-height',project:project.id,queue_sha256:project.review.queue_sha256,request,height:174}});
    assert.equal(p.shell.style['--preflight-content-height'],'174px');assert.equal(p.shell.classList.contains('preflight-draft-review'),true);
    frame.contentWindow={postMessage:data=>posted.push(data)};p.ui.render();assert.equal(p.next.disabled,true,'frame exchange discards old readiness');
  });
  await test('preflight finalization resumes only explicit completed paused input and never duplicates retry generation',async()=>{
    for(const action of ['resume','none','retry']){
      const project=preflightProject(),p=page(project,{refresh:state=>{
        if(action==='resume'){state.project.status='paused';delete state.project.review;}
        else state.project.status='running';
      }}),frame=p.get('review-frame');let request;frame.contentWindow={postMessage:data=>{if(data.type==='koww:preflight-status')request=data.request;}};p.ui.render();await frame.fire('load');frame.fakeLoadedUrl=frame.getAttribute('src');
      const send=data=>p.windowEvent('message',{origin:'http://studio.local',source:frame.contentWindow,data:{project:project.id,queue_sha256:'current-preflight-queue',request,...data}});
      await send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await send({type:'koww:preflight-finalized',workflow_action:action});await p.flush();
      assert.equal(p.calls.filter(call=>call==='refresh').length,1);assert.equal(p.calls.filter(call=>call==='resume').length,action==='resume'?1:0);
      await send({type:'koww:preflight-finalized',workflow_action:action});assert.equal(p.calls.filter(call=>call==='refresh').length,1,'duplicate completion messages are ignored');
    }
  });
  await test('preflight finalized input waits for a real refreshed state before its one resume',async()=>{
    const project=preflightProject(),p=page(project),frame=p.get('review-frame');let request;
    frame.contentWindow={postMessage:data=>{if(data.type==='koww:preflight-status')request=data.request;}};p.ui.render();await frame.fire('load');frame.fakeLoadedUrl=frame.getAttribute('src');
    const send=data=>p.windowEvent('message',{origin:'http://studio.local',source:frame.contentWindow,data:{project:project.id,queue_sha256:'current-preflight-queue',request,...data}});
    await send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await send({type:'koww:preflight-finalized',workflow_action:'resume'});assert.equal(p.calls.includes('resume'),false,'an in-progress refresh may still expose the old review');
    project.status='paused';delete project.review;p.ui.render();await p.flush();assert.equal(p.calls.filter(call=>call==='resume').length,1);
    p.state.busy=false;p.ui.render();await p.flush();assert.equal(p.calls.filter(call=>call==='resume').length,1);
  });
  await test('preflight completed history preserves the gender age and category ordinal label',async()=>{
    const label='공통 평가용 · 남성 · 중년 · 2번';
    const project=historyModel({'call-pilot-review':{completed:true,review:{autosave_policy:'play_started_default_keep_v1',items:[
      {id:'voice',text:'보라야',label,group:'공통 평가용',voice_number:16,decision:'keep',audio_url:'/history/audio/voice',playback_started:true,playback_completed:false}
    ]}}});project.settings={pilot_review_policy:'one_per_voice_before_and_after_generation_v4'};
    const p=page(project,{compactListening:true});await visit(p,'call-pilot-review');
    const picker=p.shell.querySelector('.voice-picker-label');assert.equal(picker.textContent,label);
    assert.deepEqual(p.shell.querySelector('.story-saved-review-decisions').querySelectorAll('span').map(span=>span.textContent),['적합','부적합']);
    assert.equal(p.shell.querySelector('.story-saved-review-decisions').hidden,true,'completed preflight history uses the existing saved badge instead of two inactive choice controls');
    assert.equal(p.shell.querySelector('.voice-picker-badge').textContent,'적합');
  });
  const postflightProject=()=>{
    const project=preflightProject();project.stage_id='positive_review';
    project.navigation_checkpoint={policy:'explicit_story_v1',step:'pronunciation',state:'running'};
    project.review={kind:'positive',queue_sha256:'current-postflight-queue',url:'/review/?project=navigated&queue=positive'};
    for(const stage of project.stages)if(['pilot','pilot_review','positives'].includes(stage.id))stage.status='completed';
    project.stages.push({id:'positive_review',status:'pending'});return project;
  };
  function finishPostflightProcessing(project){
    project.status='paused';project.stage_id='hard_phrases';project.process_alive=false;delete project.review;
    project.navigation_checkpoint={policy:'explicit_story_v1',step:'pronunciation',state:'awaiting_next'};
    project.stages.find(stage=>stage.id==='positive_review').status='completed';
  }
  await test('postflight next uses only its trusted completed-listening state and shares the compact popup layout',async()=>{
    const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
    assert.equal(p.shell.dataset.step,'pronunciation');assert.equal(p.next.disabled,true);
    assert.match(p.get('story-subtitle').textContent,/끝까지 들은 음원은 적합/);
    const ready={type:'koww:preflight-state',ready:true,pending:false,error:false};
    for(const invalid of [{queue_sha256:'current-preflight-queue'},{request:wire.request-1},{project:'other'},{pending:true}])await wire.send({...ready,...invalid});
    assert.equal(p.next.disabled,true);await wire.send(ready);assert.equal(p.next.disabled,false);
    await wire.send({type:'koww:preflight-height',height:170});
    assert.equal(p.shell.style['--preflight-content-height'],'170px');assert.equal(p.shell.classList.contains('preflight-draft-review'),true);
    await wire.send({type:'koww:preflight-picker',open:true,picker_request:1});assert.equal(p.shell.classList.contains('preflight-picker-open'),true);
    assert.equal(wire.posted.at(-1).type,'koww:preflight-picker-ready');
    await wire.send({type:'koww:preflight-picker',open:false,picker_request:1});assert.equal(p.shell.classList.contains('preflight-picker-open'),false);
    assert.equal(p.calls.includes('resume'),false,'readiness and popup changes never submit or advance');
  });
  await test('postflight one next retains the saving frame then admits the pool and releases its boundary into the phrase input',async()=>{
    for(const fast of [false,true]){
      const project=postflightProject();let p,resumes=0;
      p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{
        resumes++;
        if(resumes===1){state.project.status='running';state.project.process_alive=true;if(fast)finishPostflightProcessing(state.project);}
        else{assert.equal(state.project.navigation_checkpoint.step,'pronunciation');
          state.project.status='waiting_input';state.project.process_alive=false;state.project.stage_id='hard_phrases';
          state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'phrases',state:'running'};
          state.project.input={kind:'hard_phrases',suggestions:[]};}
        state.busy=false;p.ui.render();
      }});
      const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});
      await p.next.fire('click');await p.next.fire('click');assert.equal(wire.posted.filter(row=>row.type==='koww:preflight-finalize').length,1);
      assert.match(textOf(p.shell.querySelector('.story-loader')), /다음 단계로 이동 중/);
      project.status='paused';delete project.review;p.ui.render();assert.equal(p.ui.retainReviewFrame(project),true);
      assert.equal(resumes,0,'a poll before the save response does not authorize admission');
      await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();
      if(!fast){assert.equal(resumes,1);assert.equal(p.get('story-panel').inert,true);finishPostflightProcessing(project);p.ui.render();await p.flush();}
      await p.settle();assert.equal(resumes,2);assert.equal(p.shell.dataset.step,'phrases');assert.equal(p.get('story-panel').inert,false);
      assert.equal(p.shell.querySelector('.story-loader').hidden,true);assert.equal(p.calls.some(call=>call.startsWith('message:')),false);
      p.ui.render();await p.flush();assert.equal(resumes,2,'fresh polling never repeats the admitted source or released boundary');
    }
  });
  await test('postflight save failure restores the current review and allows an explicit fresh next',async()=>{
    const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:true});await p.settle();
    assert.equal(p.shell.dataset.step,'pronunciation');assert.equal(p.get('story-panel').inert,false);assert.equal(p.calls.includes('resume'),false);
    assert.equal(p.next.disabled,true);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});assert.equal(p.next.disabled,false);
    await p.next.fire('click');assert.equal(wire.posted.filter(row=>row.type==='koww:preflight-finalize').length,2,'only another explicit click can retry submission');
  });
  async function finishHeldReturn(p,wire){
    await p.flush();const chapter=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1);assert.ok(chapter);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});
    await wire.send({type:'koww:preflight-height',height:194});
    await wire.send({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});await p.settle();
  }
  await test('postflight saved unsuitable hold returns benignly to the same review without approving or resuming',async()=>{
    const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-held',held_count:1});await finishHeldReturn(p,wire);
    assert.equal(p.shell.dataset.step,'pronunciation');assert.equal(p.get('story-panel').inert,false);assert.equal(p.shell.querySelector('.story-loader').hidden,true);
    assert.equal(p.calls.some(call=>call.startsWith('message:')),false,'a saved approval hold is not shown as a failed write');
    assert.equal(p.calls.some(call=>['resume','start','input','refresh'].includes(call)),false,'expected hold performs no admission or generation action');
    assert.equal(p.next.disabled,false,'the user can inspect the selected unsuitable voice and explicitly try Next again');
    await p.next.fire('click');await wire.send({type:'koww:preflight-held',held_count:1});await finishHeldReturn(p,wire);
    assert.equal(wire.posted.filter(row=>row.type==='koww:preflight-finalize').length,2);assert.equal(p.calls.includes('resume'),false);
  });
  await test('postflight held response accepts only the bound current finalizing request and rejects stale or preflight messages',async()=>{
    const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    const message={type:'koww:preflight-held',held_count:1};
    for(const invalid of [{project:'other'},{queue_sha256:'old'},{request:wire.request-1},{held_count:0},{held_count:'1'}]){
      await wire.send({...message,...invalid});assert.equal(p.get('story-panel').inert,true);
    }
    for(const envelope of [{origin:'https://elsewhere',source:wire.source},{origin:'http://studio.local',source:{}}]){
      await p.windowEvent('message',{...envelope,data:{...message,project:project.id,queue_sha256:project.review.queue_sha256,request:wire.request}});
      assert.equal(p.get('story-panel').inert,true);
    }
    await wire.send(message);await finishHeldReturn(p,wire);assert.equal(p.calls.some(call=>call.startsWith('message:')),false);
    const before=wire.posted.length;await wire.send(message);await p.flush();assert.equal(wire.posted.length,before,'a held message without an explicit finalization cannot start a new transition');
    const pre=preflightProject(),prePage=page(pre),preWire=await wirePreflight(prePage,pre);
    await preWire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await prePage.next.fire('click');
    await preWire.send(message);assert.equal(prePage.get('story-panel').inert,true,'post hold cannot replace a preflight regeneration result');
    await preWire.send({type:'koww:preflight-state',ready:false,pending:false,error:true});await prePage.settle();assert.equal(prePage.calls.includes('resume'),false);
  });
  await test('postflight expired held responses cannot return a changed project queue or document as a valid saved hold',async()=>{
    for(const change of ['timeout','project','queue','url']){
      const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
      await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      if(change==='timeout')await p.settle();
      else if(change==='project'){p.state.project={...project,id:'other-model'};p.ui.render();}
      else if(change==='queue'){project.review={...project.review,queue_sha256:'new-positive-queue'};p.ui.render();}
      else{wire.frame.setAttribute('src','/review/?project=navigated&queue=replaced');p.ui.render();}
      await wire.send({type:'koww:preflight-held',held_count:1});await p.settle();
      assert.equal(p.calls.includes('resume'),false);assert.equal(p.get('story-panel').inert,false);
    }
  });
  await test('postflight expanded picker stays above the actual footer while preserving its reserved body center',async()=>{
    const project=postflightProject(),p=page(project,{height:453}),wire=await wirePreflight(p,project);
    p.get('review-section').getBoundingClientRect=()=>({top:174,height:170});
    p.shell.querySelector('.story-actions').getBoundingClientRect=()=>({top:397,height:44});
    await wire.send({type:'koww:preflight-height',height:170});await wire.send({type:'koww:preflight-picker',open:true,picker_request:1});
    const height=parseFloat(p.shell.style['--preflight-picker-height']),center=174+170/2;
    assert.equal(height,260);assert.ok(center+height/2<=397-8,'the expanded iframe cannot cover footer hit targets or require a footer layer above its popup');
    assert.equal(p.shell.style['--preflight-content-height'],'170px','the natural player position remains reserved while only the popup surface shrinks');
    await wire.send({type:'koww:preflight-picker',open:false,picker_request:1});assert.equal(p.shell.classList.contains('preflight-picker-open'),false);
    assert.equal(p.calls.some(call=>['start','resume','input'].includes(call)),false);
  });
  await test('postflight keeps the transition until the destination phrase form is actually available',async()=>{
    const project=postflightProject();let p,resumes=0;
    p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{
      resumes++;state.project.status='running';state.project.process_alive=true;
      if(resumes===2){state.project.stage_id='hard_phrases';state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'phrases',state:'running'};}
      state.busy=false;p.ui.render();
    }});
    const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();
    finishPostflightProcessing(project);p.ui.render();await p.flush();assert.equal(resumes,2);
    assert.equal(project.stage_id,'hard_phrases');assert.equal(p.shell.dataset.step,'pronunciation');
    assert.equal(p.get('story-panel').inert,true);assert.equal(p.shell.querySelector('.story-loader').hidden,false);
    project.status='waiting_input';project.process_alive=false;project.input={kind:'hard_phrases',suggestions:[]};p.ui.render();await p.settle();
    assert.equal(p.shell.dataset.step,'phrases');assert.equal(p.get('story-panel').inert,false);assert.equal(p.calls.some(call=>call.startsWith('message:')),false);
  });
  await test("Next from a saved pronunciation review waits for the phrase form to mount", async () => {
    const project = postflightProject();
    finishPostflightProcessing(project);
    const p = page(project, { mountPhraseFields: false, action: (action, state) => {
      assert.equal(action, "resume");
      project.navigation_checkpoint = { policy: "explicit_story_v1", step: "phrases", state: "running" };
      project.status = "running"; project.process_alive = true; state.busy = false; p.ui.render();
    } });
    const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
    assert.equal(p.shell.dataset.step, "pronunciation"); assert.equal(p.next.disabled, false);
    await p.next.fire("click"); p.advanceTimer(); await p.flush();
    assert.equal(p.shell.dataset.step, "phrases"); assert.equal(loader.hidden, false);
    assert.equal(panel.classList.contains("is-leaving"), true);
    project.status = "waiting_input"; project.process_alive = false;
    project.input = { kind: "hard_phrases", suggestions: ["보나야"] };
    p.ui.render(); await p.flush();
    assert.equal(loader.hidden, false, "an input payload alone cannot reveal an empty form");
    const field = p.element("input"); field.value = "보나야"; p.get("hard-phrase-fields").append(field);
    p.ui.render(); await p.settle();
    assert.equal(loader.hidden, true); assert.equal(panel.inert, false);
    assert.equal(p.get("input-section").hidden, false); assert.equal(p.next.disabled, false);
    assert.equal(p.get("hard-phrase-fields").firstElementChild, field);
    assert.equal(p.calls.filter(call => call === "resume").length, 1);
  });
  await test('postflight save acknowledgement releases its timeout while a real corpus admission worker remains active',async()=>{
    const project=postflightProject();let p,resumes=0;
    p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{
      resumes++;
      if(resumes===1){state.project.status='running';state.project.process_alive=true;}
      else{state.project.status='waiting_input';state.project.stage_id='hard_phrases';state.project.process_alive=false;
        state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'phrases',state:'running'};state.project.input={kind:'hard_phrases',suggestions:[]};}
      state.busy=false;p.ui.render();
    }});
    const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();assert.equal(resumes,1);
    p.timers.push({callback:()=>{},at:p.now()+120000,delay:120000});while(p.now()<120000)assert.equal(p.advanceTimer(),true);await p.flush();
    project.progress={done:100,total:3000};p.ui.render();await p.flush();
    assert.equal(p.get('story-panel').inert,true);assert.equal(p.shell.querySelector('.story-loader').hidden,false);
    assert.equal(p.calls.some(call=>call.startsWith('message:')),false,'legitimate admitted-pool verification is not a save timeout');
    finishPostflightProcessing(project);p.ui.render();await p.settle();assert.equal(resumes,2);assert.equal(p.shell.dataset.step,'phrases');
  });
  await test('postflight failed admission cancels the transition without releasing a later stale completed boundary',async()=>{
    const project=postflightProject();let p,resumes=0;
    p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{
      resumes++;state.project.status='running';state.project.process_alive=true;state.busy=false;p.ui.render();
    }});
    const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();assert.equal(resumes,1);
    project.status='interrupted';project.process_alive=false;project.error='admission worker stopped';p.ui.render();await p.settle();
    assert.equal(p.get('story-panel').inert,false);assert.equal(p.shell.querySelector('.story-loader').hidden,true);
    assert.equal(p.calls.some(call=>call.startsWith('message:')),true);
    delete project.error;finishPostflightProcessing(project);p.ui.render();await p.flush();assert.equal(resumes,1,'polling cannot revive a failed admission intent');
  });
  await test('postflight never accepts a regeneration acknowledgement or a preflight route in place of approval',async()=>{
    for(const workflow of ['retry','none']){
      const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
      await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await wire.send({type:'koww:preflight-finalized',workflow_action:workflow});await p.settle();
      assert.equal(p.calls.includes('resume'),false);assert.equal(p.get('story-panel').inert,false);assert.equal(p.shell.dataset.step,'pronunciation');
    }
    const project=postflightProject(),p=page(project,{refresh:state=>{state.project.stage_id='pilot_review';state.project.status='paused';delete state.project.review;}}),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();
    assert.equal(p.calls.includes('resume'),false,'post approval never resumes a preflight stage');assert.equal(p.get('story-panel').inert,false);
  });
  await test('postflight destination input accepts only the actual phrase form and keeps diagnostic blockers visible',async()=>{
    for(const input of [{kind:'diagnostic'},{kind:'generation_policy'}]){
      const project=postflightProject(),p=page(project,{refresh:state=>{
        state.project.status='waiting_input';state.project.stage_id='hard_phrases';state.project.process_alive=false;delete state.project.review;
        state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'phrases',state:'running'};state.project.input=input;
      }}),wire=await wirePreflight(p,project);
      await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();
      assert.equal(p.calls.includes('resume'),false);assert.equal(p.get('story-panel').inert,false);
      assert.equal(p.calls.some(call=>call.startsWith('message:')),true,'a non-phrase input cancels the explicit route');
    }
  });
  await test('postflight ordinary polls and stale sources cannot revive an expired or replaced next request',async()=>{
    const untouched=postflightProject(),plain=page(untouched);finishPostflightProcessing(untouched);plain.ui.render();await plain.flush();
    assert.equal(plain.calls.includes('resume'),false,'ordinary polling of a completed post boundary still waits for the user');
    for(const change of ['failure','project','queue','reload','timeout']){
      const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
      await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      if(change==='failure'){project.status='failed';project.error='admission failed';p.ui.render();}
      else if(change==='project'){p.state.project={...project,id:'other-model'};p.ui.render();}
      else if(change==='queue'){project.review={...project.review,queue_sha256:'new-positive-queue'};p.ui.render();}
      else if(change==='reload')await wire.frame.fire('load');
      else await p.settle();
      await wire.send({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();
      assert.equal(p.calls.includes('resume'),false,change+' cannot reuse a previous approval');assert.equal(p.get('story-panel').inert,false);
    }
  });
  await test('postflight completed history uses saved badges without old inactive decision buttons',async()=>{
    const project=historyModel({pronunciation:{completed:true,review:{items:[{id:'voice',text:'보라야',group:'학습용',voice_number:1,label:'학습용 · 남성 · 어린이 · 1번',decision:'keep',listened:true,audio_url:'/history/audio/voice'}]}}});
    project.settings={pilot_review_policy:'one_per_voice_before_and_after_generation_v4'};
    project.stages.push({id:'positive_review',status:'completed'});
    const p=page(project,{compactListening:true});await visit(p,'pronunciation');
    assert.equal(p.shell.querySelector('.story-saved-review-decisions').hidden,true);
    assert.equal(p.shell.querySelector('.voice-picker-label').textContent,'학습용 · 남성 · 어린이 · 1번');
    assert.equal(p.shell.querySelector('.voice-picker-badge').textContent,'적합');
    assert.equal(p.calls.some(call=>['start','resume','input'].includes(call)),false);
  });
  const repairProgress=()=>({state:'running',source_queue_sha256:'current-postflight-queue',done:0,total:50,unit:'개',
    label:'부적합 목소리 호출어 다시 생성',family_count:1,recheck_count:1,retained_count:49,round:1});
  function startRepair(project,state='running'){
    delete project.review;project.status=state;project.process_alive=state==='running';project.postgeneration_repair={...repairProgress(),state};
  }
  await test('postflight repair acknowledgement reveals only the authorized worker progress without another resume RPC',async()=>{
    for(const state of ['running','paused','failed']){
      const project=postflightProject();let p;
      p=page(project,{refresh:row=>{startRepair(row.project,state);if(state==='failed')row.project.error='repair worker failed';}});
      const wire=await wirePreflight(p,project);await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await wire.send({type:'koww:preflight-finalized',workflow_action:'repair',repair_pending:state==='paused'});await p.settle();
      assert.equal(p.shell.dataset.step,'pronunciation');assert.equal(p.get('story-panel').inert,false);
      assert.equal(p.shell.classList.contains('post-repair-generating'),true);assert.equal(p.next.disabled,true);
      assert.match(p.get('story-subtitle').textContent,/적합 판정 49개는 보존/);
      const counts=p.shell.querySelector('.story-voice-counts');assert.equal(counts.hidden,false);assert.equal(counts.children.length,1);
      assert.equal(counts.children[0].children[1].textContent,'0 / 50개');
      assert.equal(p.operation.textContent,state==='running'?'일시정지':'재개');
      assert.equal(p.calls.includes('resume'),false,'the save endpoint already schedules the repair or leaves an explicit resume control');
      assert.equal(p.calls.some(call=>call.startsWith('message:')),false);
    }
  });
  await test('postflight repair progress on reload does not approve audio and waits for the new pending review to be ready',async()=>{
    const project=postflightProject();startRepair(project);const p=page(project,{chapterReady:false});
    assert.equal(p.calls.includes('resume'),false);assert.equal(p.shell.classList.contains('post-repair-generating'),true);
    const frame=p.get('review-frame'),posted=[];let request;
    const source={postMessage:message=>{posted.push(message);if(message.type==='koww:preflight-status')request=message.request;}};frame.contentWindow=source;
    project.status='waiting_review';project.process_alive=false;project.postgeneration_repair.state='waiting_review';project.postgeneration_repair.done=50;
    project.review={kind:'positive',queue_sha256:'repaired-queue',url:'/review/?project=navigated&queue=positive&rev=repaired'};
    p.ui.render();await p.flush();await frame.fire('load');frame.fakeLoadedUrl=frame.getAttribute('src');await p.flush();
    assert.equal(p.get('story-panel').inert,true,'the new queue cannot replace the generation view before its content is ready');
    p.timers.push({callback:()=>{},at:p.now()+22000,delay:22000});
    while(p.now()<22000){assert.equal(p.advanceTimer(),true);await p.flush();}
    assert.equal(p.get('story-panel').inert,true,'a slow repaired queue can still become ready after the generic timeout');
    const chapter=posted.filter(row=>row.type==='koww:chapter-request').at(-1);assert.ok(chapter);
    const send=data=>p.windowEvent('message',{origin:'http://studio.local',source,data:{project:project.id,queue_sha256:'repaired-queue',request,...data}});
    await send({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});
    assert.equal(p.get('story-panel').inert,true);
    await send({type:'koww:preflight-height',height:170});await send({type:'koww:preflight-state',ready:false,pending:false,error:false,retained_count:49,review_count:1});
    await p.settle();assert.equal(p.get('story-panel').inert,false);assert.equal(p.get('review-section').hidden,false);
    assert.equal(p.shell.classList.contains('post-repair-generating'),false);assert.equal(p.next.disabled,true);
    assert.match(p.get('story-subtitle').textContent,/49개.*다시 생성한 1개/);
    assert.equal(p.calls.some(call=>['start','resume','input'].includes(call)),false);
  });
  await test('postflight repair reveal failure keeps progress concealed until an explicit local reopen',async()=>{
    const project=postflightProject();startRepair(project);const p=page(project,{chapterReady:false});
    project.status='waiting_review';project.process_alive=false;project.postgeneration_repair.state='waiting_review';project.postgeneration_repair.done=50;
    project.review={kind:'positive',queue_sha256:'repaired-queue',url:'/review/?project=navigated&queue=positive&rev=repaired'};
    p.ui.render();await p.settle();assert.equal(p.get('story-panel').inert,false);assert.equal(p.get('review-section').hidden,true);
    assert.equal(p.shell.classList.contains('post-repair-generating'),true);assert.equal(p.operation.hidden,true);
    const retry=p.get('studio-notice-action');
    assert.equal(retry.hidden,false);assert.equal(retry.focused,true);
    assert.ok(p.calls.includes('message:다시 생성한 음원은 저장됐습니다. 검수 화면이 열리지 않으면 ‘다시 열기’를 누르세요.'));
    const attempts=p.chapterRequests.length;p.ui.render();await p.flush();assert.equal(p.chapterRequests.length,attempts,'an error cannot trigger automatic reload loops');
    const stale=p.chapterRequests.at(-1),frame=p.get('review-frame'),posted=[];
    const source={postMessage:message=>posted.push(message)};frame.contentWindow=source;
    p.ui.render();await frame.fire('load');
    await retry.fire('click');assert.equal(p.get('story-panel').inert,true);assert.equal(retry.hidden,true);
    assert.equal(posted.some(row=>row.type==='koww:chapter-request'),false,'reopening must wait for the new load at the same URL');
    await replyChapter(p,stale);
    assert.equal(p.get('story-panel').inert,true);
    await frame.fire('load');
    const chapter=posted.filter(row=>row.type==='koww:chapter-request').at(-1);
    assert.ok(chapter.request>stale.data.request);
    const request=posted.filter(row=>row.type==='koww:preflight-status').at(-1).request;
    const send=data=>p.windowEvent('message',{origin:'http://studio.local',source,data:{project:project.id,queue_sha256:'repaired-queue',request,...data}});
    await send({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});
    await send({type:'koww:preflight-height',height:170});await send({type:'koww:preflight-state',ready:false,pending:false,error:false});
    await p.settle();
    assert.equal(p.get('story-panel').inert,false);assert.equal(p.get('review-section').hidden,false);
    assert.equal(p.calls.some(call=>['start','resume','input','load_review'].includes(call)),false,'reopening the review is local navigation, never a new generation request');
  });
  await test('postflight repair acknowledgement rejects an unrelated plan and cannot be used by preflight',async()=>{
    const project=postflightProject(),p=page(project,{refresh:row=>{startRepair(row.project);row.project.postgeneration_repair.source_queue_sha256='unrelated-source';}}),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.send({type:'koww:preflight-finalized',workflow_action:'repair',repair_pending:false});await p.settle();
    assert.equal(p.calls.includes('resume'),false);assert.equal(p.calls.some(call=>call.startsWith('message:')),true);
    const before=preflightProject(),old=page(before),pre=await wirePreflight(old,before);
    await pre.send({type:'koww:preflight-state',ready:true,pending:false,error:false});await old.next.fire('click');
    await pre.send({type:'koww:preflight-finalized',workflow_action:'repair',repair_pending:false});await old.settle();assert.equal(old.calls.includes('resume'),false);
  });
  await test('initial readiness waits for fresh call geometry packets and fonts without an approval or leaving animation',async()=>{
    for(const make of [preflightProject,postflightProject]){
      let releaseFonts;const project=make(),p=page(project,{fonts:{ready:new Promise(resolve=>{releaseFonts=resolve;})},paintFrames:true}),wire=await wirePreflight(p,project);
      let result;const prepared=p.ui.prepareInitialView().then(value=>{result=value;return value;});await p.flush();
      assert.equal(p.get('story-panel').classList.contains('is-leaving'),false,'the opaque boot mask owns concealment; preparation has no departing transform');
      assert.equal(p.get('review-frame').getAttribute('loading'),'eager');
      const chapter=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1);assert.ok(chapter);
      await wire.send({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});
      assert.equal(result,undefined,'queue readiness alone does not prove that the parent has accepted the child height');
      await wire.send({type:'koww:preflight-height',height:170,request:wire.request-1});
      await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false,queue_sha256:'old'});assert.equal(result,undefined);
      await wire.send({type:'koww:preflight-height',height:170});assert.equal(result,undefined,'height alone is not a current loaded state packet');
      await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false});assert.equal(result,undefined,'font settlement remains part of initial layout readiness');
      releaseFonts();await prepared;
      assert.equal(result,true);assert.equal(p.get('story-panel').classList.contains('is-leaving'),false);assert.equal(p.get('story-panel').inert,false);
      assert.equal(p.next.disabled,true,'unjudged audio is allowed to display but never becomes approved by loading');
      assert.equal(p.calls.some(call=>['start','resume','input'].includes(call)),false);
    }
  });
  await test('initial readiness invalidates old geometry and chapter responses after a same-frame reload',async()=>{
    const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
    let result;const prepared=p.ui.prepareInitialView().then(value=>{result=value;return value;});await p.flush();
    const oldChapter=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1),oldRequest=wire.request;
    await wire.send({type:'koww:preflight-height',height:170});await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false});
    await wire.frame.fire('load');const fresh=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1);assert.notEqual(fresh.request,oldChapter.request);
    await wire.send({type:'koww:chapter-ready',request:oldChapter.request,url:oldChapter.url,status:'ready'});
    await wire.send({type:'koww:preflight-height',height:170,request:oldRequest});
    await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false,request:oldRequest});assert.equal(result,undefined);
    await wire.send({type:'koww:chapter-ready',request:fresh.request,url:fresh.url,status:'ready'});assert.equal(result,undefined);
    await wire.send({type:'koww:preflight-height',height:170});await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false});
    await prepared;assert.equal(result,true);assert.equal(p.calls.includes('resume'),false);
  });
  await test("initial readiness rechecks a changed frame before the minimum hold ends", async () => {
    const project = { id: "finished", status: "completed", wakeword: "보라야", results: { url: "/results/finished/" } };
    const p = page(project, { chapterReady: false });
    let finishMinimum, result;
    const minimum = new Promise(resolve => { finishMinimum = resolve; });
    const prepared = p.ui.prepareInitialView(minimum).then(value => { result = value; });
    const frame = p.get("result-frame");
    await frame.fire("load");
    await replyChapter(p);
    assert.equal(result, undefined);
    await p.windowEvent("pagehide", { persisted: true });
    assert.equal(p.get("story-panel").inert, true);
    project.results.url = "/results/finished/?updated=1"; p.ui.render(); await p.flush();
    finishMinimum(); await p.flush();
    assert.equal(result, undefined);
    const updated = p.chapterRequests.at(-1);
    assert.match(updated.data.url, /updated=1$/);
    await replyChapter(p, updated); await prepared;
    assert.equal(result, true); assert.equal(p.get("story-panel").inert, false);
  });
  await test("refresh follows a newer workflow chapter before revealing the first page", async () => {
    const project = { id: "refresh-advances", wakeword: "보라야", status: "running", stage_id: "positives",
      settings: { pilot_review_policy: "one_per_voice_before_and_after_generation_v4" } };
    const p = page(project, { chapterReady: false, paintFrames: "manual" });
    let finishMinimum, result;
    const minimum = new Promise(resolve => { finishMinimum = resolve; });
    const prepared = p.ui.prepareInitialView(minimum).then(value => {
      result = { value, step: p.shell.dataset.step, inert: p.get("story-panel").inert };
    });
    p.ui.render();
    await p.flush();
    while (await p.advanceFrame()) {}
    assert.equal(result, undefined);
    project.stage_id = "hard_phrases";
    project.status = "waiting_input";
    project.input = { kind: "hard_phrases", phrases: ["보라빛"] };
    p.ui.render();
    finishMinimum();
    await p.flush();
    assert.equal(result, undefined, "the first chapter's paint cannot release the later prompt");
    while (await p.advanceFrame()) {}
    await prepared;
    assert.deepEqual(result, { value: true, step: "phrases", inert: false });
    assert.equal(p.shell.dataset.step, "phrases");
    const fields = p.get("hard-phrase-fields");
    assert.equal(fields.firstElementChild.value, "보라빛");
    assert.equal(p.get("hard-phrase-editor").hidden, false);
    assert.equal(p.calls.some(call => ["start", "resume", "input"].includes(call)), false);
  });
  await test('initial readiness fails rather than revealing incomplete geometry after timeout or project exchange',async()=>{
    for(const change of ['timeout','project']){
      const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
      let failure;const prepared=p.ui.prepareInitialView().catch(error=>{failure=error;});await p.flush();
      if(change==='timeout')await p.settle();else{p.state.project={id:'other',status:'running',stage_id:'features'};p.ui.render();await p.flush();}
      await prepared;assert.ok(failure);assert.equal(p.get('story-panel').classList.contains('is-leaving'),false);assert.equal(p.get('story-panel').inert,false);
      assert.equal(p.calls.some(call=>['start','resume','input'].includes(call)),false);
    }
  });
  await test('chapter readiness also waits for fresh call state and height after the child queue is ready',async()=>{
    const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-height',height:170});await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false});
    const previous=wire.request;await p.back.fire('click');await p.settle();assert.equal(p.shell.dataset.step,'positive-generation');
    await p.next.fire('click');await p.flush();const chapter=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1);
    assert.ok(wire.request>previous);await wire.send({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});
    p.advanceTimer();await p.flush();assert.equal(p.shell.querySelector('.story-loader').hidden,false);
    await wire.send({type:'koww:preflight-height',height:170,request:previous});
    await wire.send({type:'koww:preflight-state',ready:false,pending:false,error:false});assert.equal(p.shell.querySelector('.story-loader').hidden,false);
    await wire.send({type:'koww:preflight-height',height:170});await p.settle();
    assert.equal(p.shell.dataset.step,'pronunciation');assert.equal(p.shell.querySelector('.story-loader').hidden,true);
    assert.equal(p.next.disabled,true);assert.equal(p.calls.includes('resume'),false);
  });
  await test("candidate category next explicitly approves each ready group before releasing the chapter", async () => {
    const project=navigationProject("voices");
    project.reference_preview={...project.reference_preview,candidate_selection:true,review_required:true,review_complete:false};
    let p;
    p=page(project,{paintFrames:"manual",action:(_action,state)=>{
      state.project.navigation_checkpoint={policy:"explicit_story_v1",step:"pronunciation",state:"running"};
      state.project.status="running";state.project.process_alive=true;state.busy=false;p.ui.render();return true;
    }});
    const frame=p.get("reference-preview-frame"),posted=[];
    const source={postMessage:(data,origin)=>posted.push({data:JSON.parse(JSON.stringify(data)),origin})};
    frame.contentWindow=source;frame.setAttribute("src",project.reference_preview.url);
    frame.fakeLoadedUrl=project.reference_preview.url;p.ui.render();
    const send=data=>p.windowEvent("message",{origin:"http://studio.local",source,data:{project:project.id,...data}});
    assert.equal(p.next.disabled,true);assert.equal(p.shell.dataset.step,"voice-listening");
    await send({type:"koww:reference-layout",mode:"candidate_pool_v2"});
    await send({type:"koww:reference-group",group:"train:man:child",ready:false});assert.equal(p.next.disabled,true);
    await send({type:"koww:reference-group",group:"train:man:child",ready:true});assert.equal(p.next.disabled,false);
    await p.next.fire("click");await p.next.fire("click");
    assert.deepEqual(posted,[{data:{type:"koww:reference-approve",project:project.id,group:"train:man:child",request:1},origin:"http://studio.local"}]);
    assert.equal(p.next.disabled,true,"pending approval blocks another next click");assert.equal(p.calls.includes("resume"),false);
    for(const invalid of [
      {origin:"https://other.invalid",source,data:{type:"koww:reference-group",project:project.id,group:"train:man:child",ready:true}},
      {origin:"http://studio.local",source:{},data:{type:"koww:reference-group",project:project.id,group:"train:man:child",ready:true}},
      ...[{project:"another-project"},{group:""},{group:3},{group:" train:man:child"},{group:"bad\nkey"},{ready:"true"}].map(change=>({
        origin:"http://studio.local",source,data:{type:"koww:reference-group",project:project.id,group:"train:man:child",ready:true,...change}})),
    ])await p.windowEvent("message",invalid);
    assert.equal(p.next.disabled,true,"malformed or unrelated group messages cannot release pending approval");
    await send({type:"koww:reference-group",group:"train:man:child",ready:false});assert.equal(p.next.disabled,true);
    await send({type:"koww:reference-group",group:"train:man:child",ready:true});
    assert.equal(p.next.disabled,true,"a readiness broadcast does not acknowledge this save");
    await send({type:"koww:reference-approved",group:"train:man:child",request:1,success:false,complete:false});
    assert.equal(p.next.disabled,false,"a failed save can restore explicit retry");
    await p.next.fire("click");assert.equal(posted.length,2);
    await send({type:"koww:reference-group",group:"train:man:teenager",ready:false});assert.equal(p.next.disabled,true);
    await send({type:"koww:reference-group",group:"train:man:teenager",ready:true,completes_review:true});
    await send({type:"koww:reference-approved",group:"train:man:child",request:2,success:true,complete:false});
    await p.next.fire("click");
    const loader=p.shell.querySelector(".story-loader");
    assert.equal(loader.hidden,false);
    assert.equal(posted.length,2,"the final approval waits until the notice has painted");
    assert.equal(await p.advanceFrame(),true);assert.equal(posted.length,2);
    assert.equal(await p.advanceFrame(),true);
    assert.equal(posted.at(-1).data.group,"train:man:teenager");assert.equal(p.calls.includes("resume"),false,"changing categories never resumes model generation");
    await send({type:"koww:reference-review",complete:true});assert.equal(p.get("story-panel").inert,true);
    assert.equal(p.calls.includes("resume"),false,"an unrelated completion broadcast never releases the boundary");
    await send({type:"koww:reference-approved",group:"train:man:teenager",request:2,success:true,complete:true});
    assert.equal(p.calls.includes("resume"),false,"an old approval acknowledgement is ignored");
    await send({type:"koww:reference-approved",group:"train:man:teenager",request:3,success:true,complete:true});
    while(await p.advanceFrame()){}
    await p.settle();assert.equal(p.calls.filter(call=>call==="resume").length,1);
    assert.equal(p.shell.dataset.step,"pronunciation","the same final Next both approves and releases its completed source boundary");
  });
  await test("candidate category next resets on frame reload exchange and model change", async () => {
    const project=navigationProject("voices");project.reference_preview={...project.reference_preview,candidate_selection:true,review_required:true,review_complete:false};
    const p=page(project),frame=p.get("reference-preview-frame"),posted=[];
    const source={postMessage:(data,origin)=>posted.push({data:JSON.parse(JSON.stringify(data)),origin})};
    frame.contentWindow=source;frame.setAttribute("src","/reference-preview/?project="+project.id);p.ui.render();
    const ready=(sender=source,id=project.id)=>p.windowEvent("message",{origin:"http://studio.local",source:sender,data:{type:"koww:reference-group",project:id,group:"train:man:child",ready:true}});
    await ready();assert.equal(p.next.disabled,false);
    await frame.fire("load");assert.equal(p.next.disabled,true);
    assert.deepEqual(posted.at(-1),{data:{type:"koww:reference-status",project:project.id,request_layout:true},origin:"http://studio.local"},"reload requests fresh child readiness");
    await ready();assert.equal(p.next.disabled,false);
    const replacement={postMessage:(data,origin)=>posted.push({data:JSON.parse(JSON.stringify(data)),origin})};
    frame.contentWindow=replacement;frame.setAttribute("src","/reference-preview/?project="+project.id+"&reload=1");p.ui.render();assert.equal(p.next.disabled,true);
    await ready();assert.equal(p.next.disabled,true,"the previous document cannot revive its readiness");
    await ready(replacement);assert.equal(p.next.disabled,false);
    p.state.project={...project,id:"different-model",reference_preview:{...project.reference_preview}};p.ui.render();assert.equal(p.next.disabled,true);
    await ready(replacement);assert.equal(p.next.disabled,true,"a previous project's status cannot release the new model");
    await ready(replacement,"different-model");assert.equal(p.next.disabled,false);
    await p.windowEvent("message",{origin:"http://studio.local",source:replacement,data:{type:"koww:reference-layout",project:"different-model",mode:"legacy"}});
    assert.equal(p.next.disabled,true,"legacy mode clears candidate-only readiness without bypassing its saved-review gate");
  });
  await test("a final reference approval request cannot be revived after timeout frame exchange model change or page hiding",async()=>{
    for(const variant of ["timeout","frame","project","hidden","failed","wrong-request","back","pagehide"]){
      const project=navigationProject("voices");project.reference_preview={...project.reference_preview,candidate_selection:true,review_required:true,review_complete:false};
      const p=page(project),frame=p.get("reference-preview-frame"),posted=[],source={postMessage:data=>posted.push(data)};
      frame.contentWindow=source;frame.setAttribute("src","/reference-preview/?project="+project.id);p.ui.render();
      const send=data=>p.windowEvent("message",{origin:"http://studio.local",source,data:{project:project.id,...data}});
      await send({type:"koww:reference-group",group:"validation:woman:older adult",ready:true});await p.next.fire("click");
      const request=posted.at(-1).request,ack={type:"koww:reference-approved",group:"validation:woman:older adult",request,success:true,complete:true};
      if(variant==="timeout")await p.settle();
      if(variant==="frame"){frame.setAttribute("src","/reference-preview/?project="+project.id+"&reload=1");p.ui.render();}
      if(variant==="project"){p.state.project={id:"other",status:"running",stage_id:"features"};p.ui.render();}
      if(variant==="hidden")await p.visibility(true);
      if(variant==="back"){await p.back.fire("click");await p.settle();}
      if(variant==="pagehide")await p.windowEvent("pagehide",{});
      if(variant==="failed")await send({...ack,success:false,complete:false});
      if(variant==="wrong-request"){
        await send({...ack,request:request-1});assert.equal(p.next.disabled,true);
        await send({...ack,group:"another-group"});assert.equal(p.next.disabled,true);
        await send({...ack,success:false,complete:false});
      }
      await send(ack);await p.settle();assert.equal(p.calls.includes("resume"),false,variant+" cannot release a source boundary");
    }
  });
  await test("candidate category next requires the actual completed voice boundary even if a child reports ready", async () => {
    for(const variant of ["busy","running","review","migration","incomplete"]){
      const project=navigationProject("voices");project.reference_preview={...project.reference_preview,candidate_selection:true,review_required:true,review_complete:false};
      if(variant==="running")project.status="running";
      if(variant==="review")project.review={url:"/review/"};
      if(variant==="migration")project.migration={reason:"업데이트 필요"};
      if(variant==="incomplete")project.stages.find(stage=>stage.id==="references_review").status="pending";
      const p=page(project),frame=p.get("reference-preview-frame"),posted=[];
      const source={postMessage:data=>posted.push(data)};frame.contentWindow=source;p.state.busy=variant==="busy";p.ui.render();
      await p.windowEvent("message",{origin:"http://studio.local",source,data:{type:"koww:reference-group",project:project.id,group:"train:man:child",ready:true}});
      assert.equal(p.next.disabled,true,variant+" must not enable category approval");await p.next.fire("click");
      assert.equal(posted.length,0);assert.equal(p.calls.includes("resume"),false);
    }
  });
  const preparationProject=(preparation={})=>{
    const project=navigationProject("voices");
    project.reference_preview={...project.reference_preview,candidate_selection:true,review_required:true,review_complete:false,selected_count:2,approved_groups:["train:man:child"]};
    project.reference_preparation={policy:"initial_candidate_batches_v1",required:true,status:"pending",ready:false,operation:"helper",can_start:true,can_pause:false,can_resume:false,
      rows:[{split:"train",label:"학습용 목소리 후보",done:15,total:50,active:false,completed:false},
        {split:"validation",label:"공통 평가용 목소리 후보",done:5,total:50,active:false,completed:false}],
      done_groups:4,total_groups:20,reused_candidates:20,generated_candidates:0,error:null,...preparation};
    return project;
  };
  await test("reference preparation restores the chapter with explicit fifty-candidate rows and blocks early review", async()=>{
    for(const target of [20,100,500]){
      const project=preparationProject();project.reference_counts.train=target;const original=JSON.stringify(project.reference_preview);
      const p=page(project),rows=p.shell.querySelector(".story-voice-counts").children;
      assert.equal(p.shell.dataset.step,"voices");assert.equal(p.get("story-title").textContent,"기준 목소리 생성");
      assert.equal(p.shell.classList.contains("merged-reference-review"),false);assert.equal(p.shell.dataset.history,"false");
      assert.equal(rows.length,2);assert.equal(rows[0].children[1].textContent,"15 / 50개");assert.equal(rows[1].children[1].textContent,"5 / 50개");
      assert.equal(p.next.disabled,true);assert.equal(p.operation.textContent,"생성 시작");assert.equal(p.get("reference-preview").open,false);
      const review=p.ticks.find(tick=>tick.dataset.step==="voice-listening");assert.ok(review);assert.equal(review.disabled,true);
      await review.fire("click");await p.next.fire("click");assert.equal(p.shell.dataset.step,"voices");
      assert.equal(p.calls.some(call=>["prepare_start","prepare_resume","start","resume","input"].includes(call)),false,"render and blocked navigation never start GPU work");
      assert.equal(JSON.stringify(project.reference_preview),original,"existing selected/approved reference records remain unchanged");
    }
    const unready=preparationProject({status:"completed",can_start:false});
    unready.reference_preparation.rows.forEach(row=>Object.assign(row,{done:50,completed:true}));
    const waiting=page(unready);assert.equal(waiting.next.disabled,true,"two full gauges and old completed stages cannot replace the preparation-ready receipt");
    assert.equal(waiting.get("reference-preview").open,false);
  });
  await test("reference preparation redirects an existing candidate review back to missing preparation", async()=>{
    const project=preparationProject(),preparation=project.reference_preparation;delete project.reference_preparation;
    const p=page(project);assert.equal(p.shell.dataset.step,"voice-listening");
    project.reference_preparation=preparation;p.ui.render();await p.settle();
    assert.equal(p.shell.dataset.step,"voices");assert.equal(p.next.disabled,true);assert.equal(p.get("reference-preview").open,false);
    assert.equal(project.reference_preview.selected_count,2);assert.deepEqual(project.reference_preview.approved_groups,["train:man:child"]);
    assert.equal(p.calls.some(call=>["prepare_start","prepare_resume","resume","input"].includes(call)),false);
  });
  await test("reference preparation helper uses explicit start pause resume and ready next only opens review", async()=>{
    const project=preparationProject();let p;
    p=page(project,{action:(action,state)=>{
      const prep=state.project.reference_preparation;
      if(action==="prepare_start"||action==="prepare_resume")Object.assign(prep,{status:"running",can_start:false,can_pause:true,can_resume:false});
      if(action==="prepare_pause")Object.assign(prep,{status:"pausing",can_pause:false});
      if(action==="resume"){state.project.status="running";state.project.navigation_checkpoint={policy:"explicit_story_v1",step:"pronunciation",state:"running"};}
      state.busy=false;p.ui.render();
    }});
    await p.operation.fire("click");assert.equal(p.calls.filter(call=>call==="prepare_start").length,1);assert.equal(p.operation.textContent,"일시정지");
    project.reference_preparation.rows[0].active=true;project.reference_preparation.rows[0].done=25;p.ui.render();
    const rows=p.shell.querySelector(".story-voice-counts").children;
    assert.equal(rows[0].children[2].dataset.active,"true");assert.equal(rows[0].querySelector(".story-progress-indicator").dataset.status,"active");
    assert.equal(rows[1].children[2].dataset.active,"false");
    await p.operation.fire("click");assert.equal(p.calls.filter(call=>call==="prepare_pause").length,1);assert.equal(p.operation.textContent,"정지 대기");assert.equal(p.operation.disabled,true);
    await p.operation.fire("click");assert.equal(p.calls.filter(call=>call==="prepare_pause").length,1);
    Object.assign(project.reference_preparation,{status:"paused",can_resume:true});p.ui.render();assert.equal(p.operation.textContent,"재개");
    await p.operation.fire("click");assert.equal(p.calls.filter(call=>call==="prepare_resume").length,1);
    Object.assign(project.reference_preparation,{status:"completed",ready:true,can_pause:false,can_resume:false});
    project.reference_preparation.rows.forEach(row=>Object.assign(row,{done:50,active:false,completed:true}));p.ui.render();
    assert.equal(p.shell.dataset.step,"voices");assert.equal(p.next.disabled,false);assert.equal(p.operation.hidden,true);
    assert.equal(rows[0].querySelector(".story-progress-indicator").dataset.status,"complete");assert.equal(rows[0].children[2].dataset.active,"false");
    const actionsBefore=p.calls.filter(call=>["resume","prepare_start","prepare_resume","prepare_pause","input"].includes(call));
    await p.next.fire("click");await p.settle();assert.equal(p.shell.dataset.step,"voice-listening");assert.equal(p.get("reference-preview").open,true);
    assert.deepEqual(p.calls.filter(call=>["resume","prepare_start","prepare_resume","prepare_pause","input"].includes(call)),actionsBefore,"preparation next is presentation-only");
    assert.equal(p.next.disabled,true,"the review still requires explicit candidate selections");
    const frame=p.get("reference-preview-frame"),posted=[],source={postMessage:(data,origin)=>posted.push({data:JSON.parse(JSON.stringify(data)),origin})};
    frame.contentWindow=source;p.ui.render();
    await p.windowEvent("message",{origin:"http://studio.local",source,data:{type:"koww:reference-group",project:project.id,group:"train:man:child",ready:true}});
    assert.equal(p.next.disabled,false);await p.next.fire("click");
    assert.equal(posted.at(-1).data.type,"koww:reference-approve");assert.equal(p.calls.includes("resume"),false,"a prepared model still approves categories before bulk synthesis");
    await p.windowEvent("message",{origin:"http://studio.local",source,data:{type:"koww:reference-review",project:project.id,complete:true}});
    assert.equal(p.calls.includes("resume"),false);
    await p.windowEvent("message",{origin:"http://studio.local",source,data:{type:"koww:reference-approved",project:project.id,
      group:"train:man:child",request:posted.at(-1).data.request,success:true,complete:true}});
    await p.settle();assert.equal(p.calls.filter(call=>call==="resume").length,1,"the bound final approval releases backend generation without another click");
  });
  await test("reference preparation stage retains normal pause and download release while helper errors remain visible", async()=>{
    const project=navigationProject("downloads");project.reference_preview={...project.reference_preview,candidate_selection:true,review_required:false};
    project.reference_preparation={...preparationProject().reference_preparation,operation:"stage",can_start:false,
      rows:[{split:"train",label:"학습용 목소리 후보",done:0,total:50,active:false,completed:false},{split:"validation",label:"공통 평가용 목소리 후보",done:0,total:50,active:false,completed:false}]};
    const p=page(project);assert.equal(p.shell.dataset.step,"downloads");assert.equal(p.next.disabled,false);
    await p.next.fire("click");assert.equal(p.calls.filter(call=>call==="resume").length,1,"the download boundary still explicitly starts normal preparation stages");
    p.state.busy=false;project.status="running";project.stage_id="references";project.navigation_checkpoint={policy:"explicit_story_v1",step:"voices",state:"running"};
    project.reference_preparation.status="running";project.reference_preparation.rows[0].active=true;p.ui.render();await p.settle();
    assert.equal(p.shell.dataset.step,"voices");assert.equal(p.operation.textContent,"일시정지");await p.operation.fire("click");assert.equal(p.calls.filter(call=>call==="pause").length,1);
    const failed=page(preparationProject({status:"failed",can_start:false,can_resume:true,error:"후보 준비 실패"}));
    assert.equal(failed.get("project-error").hidden,false);assert.equal(failed.get("project-error").textContent,"후보 준비 실패");
    assert.equal(failed.shell.querySelector(".story-live-summary").hidden,false);assert.equal(failed.operation.textContent,"재개");
    failed.state.project.reference_preparation.error=null;failed.state.project.reference_preparation.status="paused";failed.ui.render();assert.equal(failed.get("project-error").hidden,true);
  });
  await test("reference preparation keeps its policy chapter after review and completion while legacy stays read only",async()=>{
    const project=preparationProject({required:false,status:"completed",ready:true,can_start:false});project.status="completed";project.results={url:"/results/"};
    project.reference_preparation.rows.forEach(row=>Object.assign(row,{done:50,completed:true}));
    const original=JSON.stringify(project),p=page(project);assert.equal(p.shell.dataset.step,"results");
    assert.equal(p.shell.classList.contains("separate-reference-preparation"),true);
    assert.equal(p.ticks.some(tick=>tick.dataset.step==="voices"),true,"the enabled policy keeps preparation in completed history");
    await p.ticks.find(tick=>tick.dataset.step==="voices").fire("click");await p.settle();
    const rows=p.shell.querySelector(".story-voice-counts").children;
    assert.equal(p.shell.dataset.step,"voices");assert.equal(rows[0].children[1].textContent,"50 / 50개");assert.equal(rows[1].children[1].textContent,"50 / 50개");
    assert.equal(p.operation.hidden,true,"completed preparation remains read-only history");
    assert.equal(JSON.stringify(project),original);assert.equal(p.calls.some(call=>["prepare_start","prepare_resume","resume","input"].includes(call)),false);
    const prepared=preparationProject({status:"completed",ready:true,can_start:false});prepared.reference_preparation.rows.forEach(row=>Object.assign(row,{done:50,completed:true}));
    const reviewed=page(prepared);await reviewed.next.fire("click");await reviewed.settle();assert.equal(reviewed.shell.dataset.step,"voice-listening");
    const indexOrder=reviewed.ticks.map(tick=>tick.dataset.step);prepared.reference_preparation.required=false;prepared.reference_preview.review_complete=true;reviewed.ui.render();
    assert.equal(reviewed.shell.dataset.step,"voice-listening");assert.deepEqual(reviewed.ticks.map(tick=>tick.dataset.step),indexOrder,"freezing review does not remove or reorder the preparation chapter");
    assert.equal(reviewed.calls.includes("resume"),false);
    const settingsPolicy=preparationProject();delete settingsPolicy.reference_preparation.policy;settingsPolicy.settings={reference_preparation_policy:"initial_candidate_batches_v1"};
    assert.equal(page(settingsPolicy).shell.classList.contains("separate-reference-preparation"),true,"new project settings also retain the preparation policy");
    const old=preparationProject();delete old.reference_preparation;old.status="completed";old.results={url:"/results/"};
    assert.equal(page(old).ticks.some(tick=>tick.dataset.step==="voices"),false,"completed legacy models without the new policy keep their original merged order");
    const legacy=page(navigationProject("voices"));assert.equal(legacy.shell.dataset.step,"voices");assert.equal(legacy.shell.querySelector(".story-voice-counts").children[0].children[1].textContent,"20 / 20개");
  });
  await test("merged reference chapter replaces generation and preserves the review gate", async () => {
    const project = navigationProject("voices");
    project.reference_preview = { ...project.reference_preview, candidate_selection: true, review_required: true, review_complete: false };
    const p = page(project);
    assert.equal(p.shell.dataset.step, "voice-listening");
    assert.equal(p.get("story-title").textContent, "기준 목소리 검수");
    assert.equal(p.shell.classList.contains("merged-reference-review"), true);
    assert.equal(p.ticks.some(tick => tick.dataset.step === "voices"), false, "generation is not a second chapter");
    assert.equal(p.next.disabled, true);
    project.reference_preview.review_complete = true; p.ui.render();
    assert.equal(p.next.disabled, false);
    await p.next.fire("click"); assert.equal(p.calls.filter(call => call === "resume").length, 1);
  });
  await test("merged reference chapter starts after download and shows zero-audio preparation without another start", async () => {
    const project = navigationProject("downloads");
    project.settings = { reference_selection_policy: "candidate_pool_v3", pilot_review_policy: "one_per_voice_before_and_after_generation_v4" };
    project.reference_preview = { url: "/reference-preview/?project=navigated", count: 0, candidate_selection: true, review_required: false };
    const p = page(project, { action: (_action, state) => {
      state.project.navigation_checkpoint = { policy: "explicit_story_v1", step: "voices", state: "running" };
      state.project.stage_id = "references"; state.project.status = "running";
      state.busy = false; p.ui.render();
    } });
    await p.next.fire("click"); await p.settle();
    assert.equal(p.calls.filter(call => call === "resume").length, 1);
    assert.equal(p.shell.dataset.step, "voice-listening");
    assert.equal(p.calls.includes("start"), false);
    assert.equal(p.get("reference-preview").open, true);
    assert.equal(p.next.disabled, true);
    assert.equal(p.viewModes.at(-1).mode, "reference");
  });
  await test("settings dropdown contains existing project controls and preserves their actions", async () => {
    const p = page(navigationProject("voices"));
    const menu = p.get("story-settings-menu"), toggle = p.get("story-settings-button");
    assert.equal(menu.hidden, true); assert.equal(toggle.getAttribute("aria-expanded"), "false");
    assert.equal(menu.contains(p.get("project-select")), true);
    assert.equal(menu.contains(p.get("new-project")), true);
    const notices = p.get("studio-notices");
    assert.equal(notices.querySelector(".story-toolbar").children.length, 1);
    for (const id of ["connection", "message", "project-error", "review-error", "studio-notice-action"])
      assert.equal(notices.contains(p.get(id)), true);
    await toggle.fire("click"); assert.equal(menu.hidden, false); assert.equal(toggle.getAttribute("aria-expanded"), "true");
    let switched = 0, created = 0;
    p.get("project-select").addEventListener("change", () => switched++);
    await p.get("project-select").fire("change"); assert.equal(switched, 1); assert.equal(menu.hidden, true); assert.equal(toggle.focused, true);
    await toggle.fire("click"); p.get("new-project").addEventListener("click", () => created++);
    await p.get("new-project").fire("click"); assert.equal(created, 1); assert.equal(menu.hidden, true);
    await toggle.fire("click"); await p.get("story-settings-details").fire("click");
    assert.equal(menu.hidden, true); assert.equal(p.shell.querySelectorAll("dialog").some(dialog => dialog.open), true);
    await toggle.fire("click"); await p.get("story-settings-log").fire("click");
    assert.equal(menu.hidden, true); assert.equal(p.calls.filter(call => call === "logs").length, 1);
  });
  await test("settings dropdown closes outside and on Escape while keeping keyboard access", async () => {
    const p = page(), menu = p.get("story-settings-menu"), toggle = p.get("story-settings-button"), container = toggle.parentElement;
    await toggle.fire("click"); await p.documentEvent("pointerdown", { target: p.get("project-select") }); assert.equal(menu.hidden, false);
    await p.documentEvent("pointerdown", { target: p.shell }); assert.equal(menu.hidden, true);
    let prevented = false;
    await container.fire("keydown", { key: "ArrowDown", target: toggle, preventDefault() { prevented = true; } });
    assert.equal(prevented, true); assert.equal(menu.hidden, false);
    await container.fire("keydown", { key: "Escape", preventDefault() {} });
    assert.equal(menu.hidden, true); assert.equal(toggle.focused, true); assert.equal(toggle.getAttribute("aria-expanded"), "false");
    await toggle.fire("click"); await container.fire("focusout", { relatedTarget: p.shell }); assert.equal(menu.hidden, true);
    await toggle.fire("click"); await p.windowEvent("blur"); assert.equal(menu.hidden, true, "leaving the window or focusing an embedded player dismisses the menu");
  });
  await test("top strip accepts errors only from the active review and clears each cause independently", async () => {
    const project = postflightProject(), p = page(project);
    const frame = p.get("review-frame"), strip = p.get("review-error");
    const url = new URL(frame.getAttribute("src"), "http://studio.local");
    const notice = {type: "koww:review-notice", project: project.id, queue_sha256: project.review.queue_sha256,
      url: url.pathname + url.search, channel: "playback", message: "음원 재생 실패"};
    const send = (changes = {}, envelope = {}) => p.windowEvent("message", {origin: "http://studio.local",
      source: frame.contentWindow, data: {...notice, ...changes}, ...envelope});
    for (const [changes, envelope] of [[{}, {origin:"https://wrong.invalid"}], [{}, {source:{}}],
      [{queue_sha256:"stale"}, {}], [{url:notice.url + "&stale=1"}, {}], [{request:-1}, {}]]) await send(changes, envelope);
    assert.equal(strip.textContent, "");
    await send(); assert.equal(strip.textContent, "음원 재생 실패");
    await send({channel:"save",message:"판정 저장 실패"}); assert.equal(strip.textContent, "판정 저장 실패");
    await send({channel:"save",message:""}); assert.equal(strip.textContent, "음원 재생 실패");
    await frame.fire("load"); assert.equal(strip.hidden, true);
    await send(); assert.equal(strip.hidden, false);
    await p.back.fire("click"); await p.settle(); assert.equal(strip.hidden, true);
    await send(); assert.equal(strip.hidden, true);
  });
  await test("download single view keeps every live row visible at narrow and short heights", () => {
    for (const [width, height] of [[319, 540], [600, 540], [1280, 800]]) {
      const project = checkpointProject("running", "running"); project.stage_id = "data_download";
      project.progress = { files: Array.from({ length: 9 }, (_, i) => ({ id: `asset-${i}`, group: `group-${i}`, label: `자료 ${i + 1}`, state: "downloading", done: i + 1, total: 100, unit: "bytes" })) };
      const p = page(project, { width, height });
      const history = p.shell.querySelector(".story-history"), list = history.querySelector(".story-history-rows"), pager = history.querySelector(".story-pagination");
      assert.equal(list.children.filter(row => /^자료 \d+$/.test(historyRowLabel(row))).length, 9);
      assert.equal(list.children.every(row => !row.hidden), true);
      assert.equal(pager.hidden, true); assert.equal(p.next.disabled, true);
      assert.ok(["normal", "compact", "tight"].includes(list.dataset.density));
      const first = historyRowByLabel(list, "자료 1");
      project.progress.files[0].done = 57; p.ui.render();
      assert.equal(historyRowByLabel(list, "자료 1"), first, "progress polling preserves the existing rows");
      assert.equal(list.children.every(row => !row.hidden), true); assert.equal(pager.hidden, true);
      assert.equal(project.progress.files[0].done, 57);
      assert.equal(p.calls.some(call => ["start", "resume", "pause", "input"].includes(call)), false);
    }
  });
  await test("download single view also keeps historical rows on one continuous list", async () => {
    const project = navigationProject("voices");
    project.completed_stage_views = { downloads: { completed: true, rows: Array.from({ length: 9 }, (_, i) => ({ label: `완료 자료 ${i + 1}`, done: 100, total: 100, unit: "files" })) } };
    const p = page(project, { width: 360, height: 540 });
    await p.ticks.find(tick => tick.dataset.step === "downloads").fire("click"); await p.settle();
    const history = p.shell.querySelector(".story-history"), list = history.querySelector(".story-history-rows");
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(list.children.filter(row => /^완료 자료 \d+$/.test(historyRowLabel(row))).length, 9);
    assert.equal(list.children.every(row => !row.hidden), true);
    assert.equal(history.querySelector(".story-pagination").hidden, true);
    assert.equal(p.calls.some(call => ["start", "resume", "pause", "input"].includes(call)), false);
  });
  async function pendingReferenceChapter() {
    const project = preparationProject({ status: "completed", ready: true, can_start: false });
    project.reference_preparation.rows.forEach(row => Object.assign(row, { done: 50, completed: true }));
    const p = page(project, { chapterReady: false });
    await p.ticks.find(tick => tick.dataset.step === "voices").fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "voices");
    await p.next.fire("click"); await p.flush(); await p.flush();
    assert.equal(p.shell.dataset.step, "voice-listening");
    assert.ok(p.chapterRequests.length, "the target document receives a rendered-readiness request");
    return p;
  }
  async function replyChapter(p, request = p.chapterRequests.at(-1), changes = {}, envelope = {}) {
    await p.windowEvent("message", { origin: "http://studio.local", source: request.frame.contentWindow,
      data: { ...request.data, type: "koww:chapter-ready", status: "ready", ...changes }, ...envelope });
    await p.flush();
  }
  await test("candidate review waits for the measured current layout even after the document is ready", async () => {
    const project = preparationProject({ status: "completed", ready: true, can_start: false });
    project.reference_preparation.rows.forEach(row => Object.assign(row, { done: 50, completed: true }));
    project.reference_preview = { ...project.reference_preview, candidate_selection: true, review_required: true };
    const p = page(project, { chapterReady: false, deferReferenceLayout: true });
    await p.ticks.find(tick => tick.dataset.step === "voices").fire("click"); await p.settle();
    await p.next.fire("click"); await p.flush();
    const loader = p.shell.querySelector(".story-loader"), frame = p.get("reference-preview-frame");
    p.advanceTimer(); await p.flush();
    await replyChapter(p);
    assert.equal(loader.hidden, false, "the parent still needs the candidate frame height");
    const send = (data, envelope = {}) => p.windowEvent("message", {
      origin: "http://studio.local", source: frame.contentWindow, data: { project: project.id, ...data }, ...envelope,
    });
    await send({ type: "koww:reference-layout", mode: "candidate_pool_v2" });
    await send({ type: "koww:reference-height", mode: "candidate_pool_v2", height: 410 }, { source: {} });
    assert.equal(loader.hidden, false, "another frame cannot settle the candidate geometry");
    await send({ type: "koww:reference-height", mode: "candidate_pool_v2", height: 410 });
    assert.equal(p.shell.style["--reference-content-height"], "410px");
    assert.equal(loader.hidden, true);
    assert.equal(p.next.disabled, true, "sizing a review does not approve its candidate group");
  });
  await test("chapter readiness keeps the destination concealed beyond the notice minimum until real content arrives", async () => {
    const p = await pendingReferenceChapter(), loader = p.shell.querySelector(".story-loader");
    p.advanceTimer(); await p.flush();
    assert.ok(p.now() >= 800); assert.equal(loader.hidden, false); assert.equal(p.get("story-panel").inert, true);
    const frame = p.get("reference-preview-frame");
    await p.windowEvent("message", { origin: "http://studio.local", source: frame.contentWindow,
      data: { type: "koww:reference-group", project: p.state.project.id, group: "train:man:child", ready: true } });
    assert.equal(loader.hidden, false, "quota-ready is not view-ready");
    await replyChapter(p); assert.equal(loader.hidden, true); assert.equal(p.get("story-panel").inert, false);
    assert.equal(p.shell.dataset.step, "voice-listening");
    assert.equal(p.calls.some(call => ["start", "resume", "input", "prepare_start", "prepare_resume"].includes(call)), false);
  });
  await test("chapter readiness rejects stale origin source project url and nonce then accepts the current loaded document", async () => {
    const p = await pendingReferenceChapter(), old = p.chapterRequests.at(-1), loader = p.shell.querySelector(".story-loader");
    p.advanceTimer(); await p.flush();
    for (const [changes, envelope] of [
      [{}, { origin: "https://other.invalid" }], [{}, { source: {} }], [{ project: "another" }, {}],
      [{ request: old.data.request - 1 }, {}], [{ url: old.data.url + "&old=1" }, {}],
    ]) { await replyChapter(p, old, changes, envelope); assert.equal(loader.hidden, false); }
    await old.frame.fire("load"); const fresh = p.chapterRequests.at(-1);
    assert.notEqual(fresh.data.request, old.data.request);
    await replyChapter(p, old); assert.equal(loader.hidden, false, "same-URL old-document replies cannot finish a reload");
    await replyChapter(p, fresh); assert.equal(loader.hidden, true);
  });
  await test("chapter readiness failure and timeout return to the source and allow an explicit fresh retry", async () => {
    for (const failure of ["response", "timeout"]) {
      const p = await pendingReferenceChapter(), old = p.chapterRequests.at(-1);
      p.advanceTimer(); await p.flush();
      if (failure === "response") await replyChapter(p, old, { status: "error", message: "목록을 불러오지 못했습니다." });
      else { p.advanceTimer(); await p.flush(); await p.flush(); }
      assert.equal(p.shell.dataset.step, "voices"); assert.equal(p.shell.querySelector(".story-loader").hidden, true);
      assert.equal(p.get("story-panel").inert, false); assert.ok(p.calls.some(call => call.startsWith("message:")));
      await p.next.fire("click"); await p.flush(); const fresh = p.chapterRequests.at(-1);
      assert.ok(fresh.data.request > old.data.request);
      await replyChapter(p, old); assert.equal(p.shell.querySelector(".story-loader").hidden, false);
      await replyChapter(p, fresh); await p.settle(); assert.equal(p.shell.dataset.step, "voice-listening");
    }
  });
  await test("chapter readiness revokes a fast result if its document reloads during the minimum hold", async () => {
    for (const failedReload of [false, true]) {
      const p = await pendingReferenceChapter(), first = p.chapterRequests.at(-1), loader = p.shell.querySelector(".story-loader");
      await replyChapter(p, first); assert.equal(loader.hidden, false, "the minimum hold remains even for a fast view");
      await first.frame.fire("load"); p.advanceTimer(); await p.flush(); await p.flush();
      const fresh = p.chapterRequests.at(-1); assert.ok(fresh.data.request > first.data.request);
      assert.equal(loader.hidden, false, "the completed hold cannot reveal a reloading destination");
      await replyChapter(p, first); assert.equal(loader.hidden, false);
      await replyChapter(p, fresh, failedReload ? { status: "error", message: "다시 불러오지 못했습니다." } : {});
      assert.equal(loader.hidden, true); assert.equal(p.shell.dataset.step, failedReload ? "voices" : "voice-listening");
    }
  });
  await test("chapter readiness cancellation releases pending waits without revealing a stale destination", async () => {
    for (const kind of ["pagehide", "project"]) {
      const p = await pendingReferenceChapter(), request = p.chapterRequests.at(-1);
      if (kind === "pagehide") await p.windowEvent("pagehide");
      else { p.state.project = { id: "different", status: "running", stage_id: "features" }; p.ui.render(); }
      await replyChapter(p, request); await p.settle();
      assert.equal(p.shell.querySelector(".story-loader").hidden, true); assert.equal(p.get("story-panel").inert, false);
      if (kind === "project") assert.equal(p.shell.dataset.step, "features");
    }
  });
  await test("chapter readiness covers result and microphone destinations with their own document and timeout budget", async () => {
    const project = { id: "finished", status: "completed", wakeword: "보라야", results: { url: "/results/finished/", mic_url: "/mic/finished/" } };
    for (const target of ["results", "microphone"]) {
      const p = page(project, { chapterReady: false });
      await p.ticks.find(tick => tick.dataset.step === "training").fire("click"); await p.settle();
      await p.ticks.find(tick => tick.dataset.step === target).fire("click"); await p.flush(); await p.flush();
      assert.equal(p.shell.querySelector(".story-loader").hidden, false); const request = p.chapterRequests.at(-1);
      assert.equal(request.data.url, "http://studio.local/" + (target === "results" ? "results" : "mic") + "/finished/");
      assert.ok(p.timers.some(timer => !timer.canceled && timer.delay === (target === "microphone" ? 45000 : 20000)));
      await replyChapter(p); await p.settle(); assert.equal(p.shell.dataset.step, target);
      assert.equal(p.shell.querySelector(".story-loader").hidden, true); assert.equal(p.calls.includes("start"), false);
    }
  });
  await test("chapter readiness waits for a result frame that appears during the destination paint", async () => {
    const project = navigationProject("evaluation");
    const p = page(project, { chapterReady: false, paintFrames: "manual", action: (_action, state) => {
      project.status = "running";
      project.navigation_checkpoint = { policy: "explicit_story_v1", step: "results", state: "running" };
      state.busy = false; p.ui.render();
    } });
    const loader = p.shell.querySelector(".story-loader");
    await p.next.fire("click"); await p.advanceFrame(); await p.advanceFrame();
    assert.equal(p.shell.dataset.step, "results");
    assert.equal(p.chapterRequests.length, 0);
    project.status = "completed"; project.results = { url: "/results/navigated/" };
    p.ui.render(); await p.flush();
    while (await p.advanceFrame()) {}
    p.advanceTimer(); await p.flush();
    assert.equal(loader.hidden, false);
    const request = p.chapterRequests.at(-1);
    assert.equal(request.frame.id, "result-frame");
    await replyChapter(p, request);
    while (await p.advanceFrame()) {}
    await p.settle();
    assert.equal(loader.hidden, true); assert.equal(p.get("story-panel").inert, false);
  });
  await test("finished results repurpose the current preparation chapter behind a cover until the report is ready", async () => {
    const project = { id: "result-published", status: "running", stage_id: "results", process_alive: true };
    const p = page(project, { chapterReady: false });
    const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
    assert.equal(p.shell.dataset.step, "results");
    assert.equal(loader.hidden, true, "ongoing result generation has its own visible progress");
    Object.assign(project, { status: "completed", process_alive: false, results: { url: "/results/result-published/" } });
    p.ui.render(); await p.flush();
    assert.equal(loader.hidden, false);
    p.advanceTimer(); await p.flush();
    assert.equal(panel.inert, true);
    const frameRequest = p.chapterRequests.at(-1);
    assert.equal(frameRequest?.frame.id, "result-frame");
    await replyChapter(p, frameRequest);
    await p.settle();
    assert.equal(loader.hidden, true);
    assert.equal(p.get("results-section").hidden, false);
    assert.equal(panel.inert, false);
    assert.equal(p.calls.some(call => ["start", "resume", "input"].includes(call)), false);
  });
  await test("chapter readiness also waits for required pronunciation review content", async () => {
    const project = { id: "reviewing", status: "waiting_review", stage_id: "pilot", wakeword: "보라야",
      review: { url: "/review/?project=reviewing&queue=pilot" } };
    const p = page(project, { chapterReady: false });
    await p.ticks.find(tick => tick.dataset.step === "scale").fire("click"); await p.settle();
    await p.ticks.find(tick => tick.dataset.step === "pronunciation").fire("click"); await p.flush(); await p.flush();
    const request = p.chapterRequests.at(-1); assert.equal(request.frame.id, "review-frame");
    p.advanceTimer(); await p.flush(); assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await replyChapter(p); assert.equal(p.shell.querySelector(".story-loader").hidden, true);
    assert.equal(p.shell.dataset.step, "pronunciation"); assert.equal(p.calls.includes("input"), false);
  });
  await test("legacy pronunciation stays covered until the pilot queue is mounted and rendered", async () => {
    for (const [via, source, policy] of [
      ["refresh", "pilot_review", "one_per_voice_after_generation_v3"],
      ["poll", "positives", "one_per_voice_after_generation_v3"],
      ["same chapter", "pilot", undefined],
    ]) {
      const project = { id: "legacy-pronunciation", wakeword: "보라야", status: "running", process_alive: true,
        stage_id: source, settings: { pilot_review_policy: policy } };
      const p = page(project, { chapterReady: false });
      const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
      let initial, prepared;
      if (via === "refresh") initial = p.ui.prepareInitialView().then(value => { prepared = value; });
      else { project.stage_id = "pilot_review"; p.ui.render(); }
      await p.flush();
      if (via !== "refresh") { p.advanceTimer(); await p.flush(); }
      assert.equal(p.shell.dataset.step, "pronunciation", via);
      assert.equal(panel.inert, true, "the chapter stays covered while the worker publishes its queue");
      assert.equal(p.chapterRequests.length, 0);
      Object.assign(project, { status: "waiting_review", process_alive: false,
        review: { queue_id: "pilot", queue_sha256: "legacy-pilot-queue", url: "/review/?project=legacy-pronunciation&queue=pilot" } });
      p.ui.render(); await p.flush();
      assert.equal(panel.inert, true, "a newly mounted iframe still has to render");
      assert.equal(p.chapterRequests.at(-1)?.frame.id, "review-frame");
      await replyChapter(p);
      await initial;
      if (via === "refresh") assert.equal(prepared, true);
      assert.equal(p.get("review-section").hidden, false);
      assert.equal(panel.inert, false);
      assert.equal(loader.hidden, true);
      assert.equal(p.calls.some(call => ["start", "resume", "input"].includes(call)), false);
    }
  });
  await test("chapter readiness waits for fonts and paint on ordinary chapters without starting work", async () => {
    let release;
    const p = page(null, { fonts: { ready: new Promise(resolve => { release = resolve; }) }, paintFrames: "manual" });
    await p.next.fire("click"); await p.advanceFrame(); await p.advanceFrame(); p.advanceTimer(); await p.flush();
    assert.equal(p.shell.dataset.step, "tools"); assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    assert.equal(p.get("story-panel").inert, true);
    release(); await p.flush(); while (await p.advanceFrame()) {}
    assert.equal(p.shell.querySelector(".story-loader").hidden, true); assert.equal(p.get("story-panel").inert, false);
    await p.back.fire("click"); await p.flush(); assert.equal(p.get("story-panel").inert, true);
    while (await p.advanceFrame()) {}
    await p.settle();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(p.calls.includes("create"), false);
  });
  await test("chapter readiness font timeout restores controls without waiting on the failed font again", async () => {
    const p = page(null, { fonts: { ready: new Promise(() => {}) }, paintFrames: "manual" });
    await p.next.fire("click"); await p.advanceFrame(); await p.advanceFrame();
    p.advanceTimer(); await p.flush(); p.advanceTimer();
    for (let i = 0; i < 8; i++) await p.flush();
    assert.equal(p.shell.querySelector(".story-loader").hidden, true); assert.equal(p.get("story-panel").inert, false);
    assert.equal(p.shell.dataset.step, "entry"); assert.ok(p.calls.some(call => call.startsWith("message:")));
  });
  const hardProject=sampleFirst=>{
    const project=navigationProject('hard-generation');project.status='waiting_review';project.stage_id='hard_review';project.process_alive=false;
    project.settings={hard_review_policy:sampleFirst?'sample_before_generation_v2':'one_per_phrase_v1'};
    project.navigation_checkpoint={policy:'explicit_story_v1',step:'hard-review',state:'running'};
    project.review={queue_id:'hard',queue_sha256:'current-hard-queue',url:'/review/?project=navigated&queue=hard'};
    if(sampleFirst){project.stages.find(row=>row.id==='hard_generation').status='pending';project.stages.push({id:'hard_pilot',status:'completed'});}
    return project;
  };
  async function wireHard(p,project){const wire=await wirePreflight(p,project);wire.sendHard=data=>wire.send({phase:'hard',...data});return wire;}
  await test("bulk hard review stays covered until its current queue and player layout have painted", async () => {
    for (const via of ["next", "poll"]) {
      const project = navigationProject("hard-generation");
      project.settings = { hard_review_policy: "one_per_phrase_v1" };
      const queue = { queue_id: "hard", queue_sha256: "current-hard-queue", url: "/review/?project=navigated&queue=hard" };
      const enterReview = () => Object.assign(project, { stage_id: "hard_review", status: "ready",
        process_alive: false, navigation_checkpoint: { policy: "explicit_story_v1", step: "hard-review", state: "running" } });
      if (via === "poll") Object.assign(project, { stage_id: "hard_generation", status: "running",
        process_alive: true, navigation_checkpoint: { policy: "explicit_story_v1", step: "hard-generation", state: "running" } });
      const p = page(project, { paintFrames: "manual", action: (action, state) => {
        assert.equal(action, "resume");
        state.busy = false;
        enterReview();
        p.ui.render();
      } });
      const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
      assert.equal(p.shell.dataset.step, "hard-generation");
      if (via === "next") await p.next.fire("click");
      else { enterReview(); p.ui.render(); }
      await p.advanceFrame();
      await p.advanceFrame();
      await p.flush();
      assert.equal(p.shell.dataset.step, "hard-review");
      assert.equal(p.advanceTimer(), true);
      await p.flush();
      while (await p.advanceFrame()) {}
      assert.equal(loader.hidden, false, "the notice minimum cannot reveal a review with no queue");
      assert.equal(panel.inert, true);
      assert.equal(p.get("review-section").hidden, true);

      project.status = "running"; project.process_alive = true; p.ui.render();
      assert.equal(panel.inert, true);
      Object.assign(project, { status: "waiting_review", process_alive: false,
        review: { ...queue, queue_id: "pilot", queue_sha256: "previous-queue", url: "/review/?project=navigated&queue=pilot" } });
      p.ui.render(); await p.flush();
      while (await p.advanceFrame()) {}
      assert.equal(loader.hidden, false, "a previous chapter's queue cannot reveal the hard review");
      project.review = queue;
      const wire = await wireHard(p, project);
      const chapter = wire.posted.filter(row => row.type === "koww:chapter-request").at(-1);
      assert.ok(chapter, "the destination iframe must acknowledge its current document");
      await wire.sendHard({ type: "koww:chapter-ready", request: chapter.request - 1, url: chapter.url, status: "ready" });
      await wire.sendHard({ type: "koww:preflight-height", height: 170 });
      await wire.sendHard({ type: "koww:preflight-state", ready: false, pending: false, error: false, queue_sha256: "previous-queue" });
      assert.equal(loader.hidden, false);
      await wire.sendHard({ type: "koww:chapter-ready", request: chapter.request, url: chapter.url, status: "ready" });
      assert.equal(loader.hidden, false, "a frame acknowledgement without current player state must not reveal the chapter");
      await wire.sendHard({ type: "koww:preflight-state", ready: false, pending: false, error: false });
      assert.equal(loader.hidden, false, "the matched layout must paint before removing the notice");
      while (await p.advanceFrame()) {}
      assert.equal(loader.hidden, true);
      assert.equal(panel.inert, false);
      assert.equal(p.get("review-section").hidden, false);
      assert.equal(p.next.disabled, true, "loading the player does not approve audio");
      assert.equal(p.calls.filter(call => call === "resume").length, via === "next" ? 1 : 0);
    }
  });
  await test("hard review offers local recovery after a timeout for both generation paths", async () => {
    for (const sampleFirst of [false, true]) {
      const project = hardProject(sampleFirst), queue = project.review;
      Object.assign(project, { stage_id: sampleFirst ? "hard_pilot" : "hard_generation", status: "running", process_alive: true });
      delete project.review;
      const p = page(project);
      assert.equal(p.shell.dataset.step, sampleFirst ? "hard-review" : "hard-generation");
      project.stage_id = "hard_review"; p.ui.render(); await p.flush(); p.advanceTimer(); await p.flush();
      assert.equal(p.get("story-panel").inert, true); assert.equal(p.get("review-section").hidden, true);
      await p.settle();
      assert.equal(p.shell.dataset.step, "hard-review");
      assert.equal(p.get("story-panel").inert, false); assert.equal(p.get("review-section").hidden, true);
      const retry = p.get("studio-notice-action");
      assert.equal(retry.hidden, false); assert.equal(p.operation.hidden, true); assert.equal(p.next.disabled, true);
      assert.equal(p.calls.filter(call => call.startsWith("message:")).length, 1);
      Object.assign(project, { status: "waiting_review", process_alive: false, review: queue });
      const wire = await wireHard(p, project);
      assert.equal(p.get("review-section").hidden, true); assert.equal(retry.hidden, false);
      await retry.fire("click"); await p.flush();
      assert.equal(wire.posted.some(row => row.type === "koww:chapter-request"), false, "a retry cannot accept the old frame before its new load");
      await wire.frame.fire("load"); await p.flush();
      const chapter = wire.posted.filter(row => row.type === "koww:chapter-request").at(-1); assert.ok(chapter);
      await wire.sendHard({ type: "koww:chapter-ready", request: chapter.request, url: chapter.url, status: "ready" });
      await wire.sendHard({ type: "koww:preflight-height", height: 170 });
      await wire.sendHard({ type: "koww:preflight-state", ready: false, pending: false, error: false });
      await p.settle(); assert.equal(p.get("story-panel").inert, false); assert.equal(p.get("review-section").hidden, false);
      assert.equal(p.next.disabled, true);
      assert.equal(p.calls.some(call => ["start", "resume", "input", "load_hard_review"].includes(call)), false);
    }
  });
  function finishHardProcessing(project,sampleFirst){
    project.status='paused';project.process_alive=false;delete project.review;
    project.stage_id=sampleFirst?'hard_generation':'features';project.navigation_checkpoint={policy:'explicit_story_v1',step:'hard-review',state:'awaiting_next'};
    project.stages.find(row=>row.id==='hard_review').status='completed';
    if(!sampleFirst)project.stages.find(row=>row.id==='hard_connect').status='completed';
  }
  await test('hard compact review requires its explicit phase and trusted state before enabling Next',async()=>{
    const project=hardProject(false),p=page(project),wire=await wireHard(p,project);
    assert.equal(p.shell.dataset.step,'hard-review');assert.equal(p.next.disabled,true);
    assert.equal(wire.posted.find(row=>row.type==='koww:preflight-status').phase,'hard');
    assert.equal(p.shell.querySelector('.story-live-summary').hidden,true);
    assert.match(p.get('story-subtitle').textContent,/다음을 누르면 저장하고 적합한 유사어를 학습에 사용합니다/);
    const ready={type:'koww:preflight-state',ready:true,pending:false,error:false};
    for(const changes of [{phase:undefined},{phase:'post'},{project:'other'},{queue_sha256:'old'},{request:wire.request-1},{pending:true}]){
      await wire.sendHard({...ready,...changes});assert.equal(p.next.disabled,true);
    }
    await wire.sendHard(ready);assert.equal(p.next.disabled,false);
    await wire.sendHard({type:'koww:preflight-height',height:170});assert.equal(p.shell.style['--preflight-content-height'],'170px');
    await wire.sendHard({type:'koww:preflight-picker',open:true,picker_request:1});assert.equal(wire.posted.at(-1).phase,'hard');
    await wire.sendHard({type:'koww:preflight-picker',open:false,picker_request:1});assert.equal(wire.posted.at(-1).type,'koww:preflight-picker-closed');assert.equal(wire.posted.at(-1).phase,'hard');
    assert.equal(p.calls.includes('resume'),false);
  });
  await test('hard compact Next saves once then waits for connector and the correct next policy chapter',async()=>{
    for(const sampleFirst of [false,true]){
      const project=hardProject(sampleFirst),target=sampleFirst?'hard-generation':'features';let p,resumes=0;
      p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(_action,state)=>{
        resumes++;state.project.status='running';state.project.process_alive=true;
        if(resumes===2){state.project.stage_id=sampleFirst?'hard_generation':'features';state.project.navigation_checkpoint={policy:'explicit_story_v1',step:target,state:'running'};}
        state.busy=false;p.ui.render();
      }});
      const wire=await wireHard(p,project);await wire.sendHard({type:'koww:preflight-state',ready:true,pending:false,error:false});
      await p.next.fire('click');await p.next.fire('click');assert.equal(wire.posted.filter(row=>row.type==='koww:preflight-finalize').length,1);
      assert.equal(wire.posted.find(row=>row.type==='koww:preflight-finalize').phase,'hard');
      project.status='paused';delete project.review;p.ui.render();assert.equal(p.ui.retainReviewFrame(project),true);
      await wire.sendHard({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();assert.equal(resumes,1);
      if(!sampleFirst){project.stage_id='hard_connect';project.status='running';project.process_alive=true;p.ui.render();await p.flush();
        assert.equal(p.get('story-panel').inert,true);assert.equal(p.calls.some(call=>call.startsWith('message:')),false);}
      finishHardProcessing(project,sampleFirst);p.ui.render();await p.settle();
      assert.equal(resumes,2);assert.equal(p.shell.dataset.step,target);assert.equal(p.get('story-panel').inert,false);
      assert.equal(p.calls.some(call=>call.startsWith('message:')),false);p.ui.render();await p.flush();assert.equal(resumes,2);
    }
  });
  await test('hard compact all unsuitable hold returns to the same review without a failure banner or resume',async()=>{
    const project=hardProject(false),p=page(project),wire=await wireHard(p,project);
    await wire.sendHard({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
    await wire.sendHard({type:'koww:preflight-held',held_count:3});await p.flush();
    const chapter=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1);assert.ok(chapter);
    await wire.sendHard({type:'koww:preflight-state',ready:true,pending:false,error:false});await wire.sendHard({type:'koww:preflight-height',height:194});
    await wire.sendHard({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});await p.settle();
    assert.equal(p.shell.dataset.step,'hard-review');assert.equal(p.get('story-panel').inert,false);assert.equal(p.calls.includes('resume'),false);
    assert.equal(p.calls.some(call=>call.startsWith('message:')),false);assert.equal(p.next.disabled,false);
  });
  await test('hard compact cannot use positive repair retry or automatic-worker acknowledgements',async()=>{
    for(const workflow_action of ['repair','retry','none']){
      const project=hardProject(false),p=page(project),wire=await wireHard(p,project);
      await wire.sendHard({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      await wire.sendHard({type:'koww:preflight-finalized',workflow_action,repair_pending:true});await p.settle();
      assert.equal(p.calls.includes('resume'),false);assert.equal(p.get('story-panel').inert,false);
    }
    const project=postflightProject(),p=page(project),wire=await wirePreflight(p,project);
    await wire.send({type:'koww:preflight-state',phase:'hard',ready:true,pending:false,error:false});assert.equal(p.next.disabled,true);
  });
  await test('hard compact initial readiness waits for actual loaded geometry without judging or saving audio',async()=>{
    const project=hardProject(false),p=page(project),wire=await wireHard(p,project);let prepared;
    const waiting=p.ui.prepareInitialView().then(value=>{prepared=value;});await p.flush();
    const chapter=wire.posted.filter(row=>row.type==='koww:chapter-request').at(-1);
    await wire.sendHard({type:'koww:chapter-ready',request:chapter.request,url:chapter.url,status:'ready'});assert.equal(prepared,undefined);
    await wire.sendHard({type:'koww:preflight-state',ready:false,pending:false,error:false});assert.equal(prepared,undefined);
    await wire.sendHard({type:'koww:preflight-height',height:170});await waiting;
    assert.equal(prepared,true);assert.equal(p.next.disabled,true);assert.equal(p.get('story-panel').classList.contains('is-leaving'),false);
    assert.equal(p.calls.some(call=>['start','resume','input'].includes(call)),false);
  });
  await test('hard compact stale errors and other queue gates cannot approve a replacement document',async()=>{
    for(const change of ['queue','project','reload','failure']){
      const project=hardProject(false),p=page(project),wire=await wireHard(p,project);
      await wire.sendHard({type:'koww:preflight-state',ready:true,pending:false,error:false});await p.next.fire('click');
      if(change==='queue'){project.review={...project.review,queue_sha256:'other-hard-queue'};p.ui.render();}
      else if(change==='project'){p.state.project={...project,id:'other'};p.ui.render();}
      else if(change==='reload')await wire.frame.fire('load');
      else{project.status='failed';project.error='hard connection failed';p.ui.render();}
      await wire.sendHard({type:'koww:preflight-finalized',workflow_action:'resume'});await p.settle();assert.equal(p.calls.includes('resume'),false);
    }
    const project=hardProject(false);project.review.queue_id='content';
    const p=page(project);assert.equal(p.next.disabled,true);await p.next.fire('click');assert.equal(p.calls.includes('resume'),false);
  });
  await test('hard connection leaves the movement overlay while the audit runs and continues exactly once',async()=>{
    const project=hardProject(false);let p,resumes=0;
    p=page(project,{refresh:state=>{state.project.status='paused';delete state.project.review;},action:(action,state)=>{
      if(action==='resume'){resumes++;state.project.status='running';state.project.process_alive=true;
        if(resumes===2){state.project.stage_id='features';state.project.navigation_checkpoint={policy:'explicit_story_v1',step:'features',state:'running'};}}
      state.busy=false;p.ui.render();
    }});
    const wire=await wireHard(p,project);await wire.sendHard({type:'koww:preflight-state',ready:true,pending:false,error:false});
    await p.next.fire('click');await wire.sendHard({type:'koww:preflight-finalized',workflow_action:'resume'});await p.flush();
    project.stage_id='hard_connect';project.status='running';project.process_alive=true;p.ui.render();await p.settle();
    assert.equal(p.get('story-panel').inert,false);assert.equal(p.shell.querySelector('.story-loader').hidden,true);
    assert.equal(p.shell.dataset.step,'hard-review');assert.equal(p.shell.classList.contains('hard-connecting'),true);
    assert.equal(p.shell.querySelector('.story-live-summary').hidden,true);assert.equal(p.operation.textContent,'일시정지');
    const row=p.shell.querySelector('.story-voice-counts').children[0];assert.equal(row.children[1].textContent,'확인 중');
    assert.equal(row.children[2].hidden,false);assert.equal(row.children[2].getAttribute('value'),null);
    finishHardProcessing(project,false);p.ui.render();await p.settle();assert.equal(resumes,2);assert.equal(p.shell.dataset.step,'features');
    p.ui.render();await p.flush();assert.equal(resumes,2);
  });
  await test('hard connection reload restores real work without inventing a Next or restarting review',async()=>{
    const project=hardProject(false);delete project.review;project.stage_id='hard_connect';project.status='running';project.process_alive=true;
    project.stages.find(row=>row.id==='hard_review').status='completed';const p=page(project);
    assert.equal(p.shell.dataset.step,'hard-review');assert.equal(p.shell.classList.contains('hard-connecting'),true);
    assert.equal(p.shell.querySelector('.story-loader').hidden,true);assert.equal(p.calls.includes('resume'),false);
    finishHardProcessing(project,false);p.ui.render();await p.flush();assert.equal(p.calls.includes('resume'),false);assert.equal(p.next.disabled,false);
  });
  await test('feature preparation keeps a real indeterminate work row before per-work counts arrive',async()=>{
    for(const [status,processAlive,active,state] of [['running',true,true,'확인 중'],['pausing',true,true,'확인 중'],['paused',false,false,'일시정지'],['running',false,false,'대기 중']]){
      const project={id:'features-prepare',status,process_alive:processAlive,stage_id:'features',progress:{done:14,total:20,unit:'완료 단계'}};
      const original=JSON.stringify(project),p=page(project),counts=p.shell.querySelector('.story-voice-counts');
      assert.equal(p.shell.dataset.step,'features');assert.equal(counts.hidden,false);assert.equal(counts.children.length,1);
      const row=counts.children[0],bar=row.querySelector('progress'),indicator=row.querySelector('.story-progress-indicator');
      assert.equal(row.querySelector('strong').dataset.label,'자료 확인 및 특징 추출 준비');assert.equal(row.children[1].textContent,state);
      assert.equal(bar.hidden,false);assert.equal(bar.getAttribute('value'),null);assert.equal(bar.getAttribute('max'),null);
      assert.equal(bar.dataset.active,String(active));assert.equal(indicator.dataset.status,active?'active':'');assert.equal(indicator.hidden,!active);
      if(status==='pausing'){assert.equal(p.operation.textContent,'정지 대기');assert.equal(p.operation.disabled,true);}
      assert.ok(!textOf(row).includes('14 / 20'),'chapter completion counters never become feature progress');
      assert.equal(p.next.disabled,true);p.ui.render();await p.next.fire('click');await p.flush();
      assert.equal(counts.children[0],row);assert.equal(JSON.stringify(project),original);
      assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false,'displaying preparation never launches or resumes work');
    }
  });
  await test('feature preparation switches to actual counts without rebuilding rows or inventing completion',()=>{
    const project={id:'features-progress',status:'running',process_alive:true,stage_id:'features'};
    const p=page(project),counts=p.shell.querySelector('.story-voice-counts'),row=counts.children[0],bar=row.querySelector('progress');
    assert.equal(counts.hidden,false);assert.equal(bar.hidden,false);assert.equal(bar.getAttribute('value'),null);
    project.progress={label:'positive',done:0,total:3000,unit:'rows'};p.ui.render();
    assert.equal(counts.children[0],row);assert.equal(row.querySelector('strong').dataset.label,'호출어 음성');
    assert.equal(bar.value,0);assert.equal(bar.max,3000);assert.equal(row.children[1].textContent,'0 / 3,000행');
    project.progress.done=7;p.ui.render();assert.equal(bar.value,7);assert.equal(row.children[1].textContent,'7 / 3,000행');
    assert.equal(row.querySelector('.story-progress-indicator').dataset.status,'active');
    assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
    project.progress={done:14,total:20,unit:'완료 단계'};p.ui.render();
    assert.equal(row.querySelector('strong').dataset.label,'자료 확인 및 특징 추출 준비');
    assert.equal(bar.getAttribute('value'),null);assert.equal(row.children[1].textContent,'확인 중');
  });
  await test('feature preparation fallback never labels unrelated stages or a completed feature checkpoint as active work',()=>{
    for(const stage of ['training']){
      const p=page({id:stage,status:'running',stage_id:stage,progress:{done:14,total:20,unit:'완료 단계'}});
      assert.ok(!textOf(p.shell.querySelector('.story-voice-counts')).includes('자료 확인 및 특징 추출 준비'));
      assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
    }
    const p=page({id:'features-finished',status:'paused',stage_id:'features',stages:[{id:'features',status:'completed'}],progress:{done:14,total:20,unit:'완료 단계'}});
    assert.ok(!textOf(p.shell.querySelector('.story-voice-counts')).includes('자료 확인 및 특징 추출 준비'));
  });
  await test('evaluation preparation and validation keep the shared gauge before numeric scoring exists',async()=>{
    for(const [stage,progress,label,status] of [
      ['development',{done:17,total:20,unit:'완료 단계'},'평가 자료 준비','준비 중'],
      ['evaluation',{},'성능 평가 준비','준비 중'],
      ['evaluation',{done:null,total:null,unit:'개 음원',phase:'validation',label:'평가 음원 검증 중'},'평가 음원 검증','검증 중'],
      ['evaluation',{done:null,total:null,unit:'개 음원',phase:'finalizing',label:'성능 평가 · GPU'},'평가 결과 정리','정리 중'],
      ['evaluation',{done:null,total:null,phase:'runtime_recommendation',label:'모델별 추천 감지 설정 비교'},'모델별 추천 감지 설정 비교','비교 중'],
      ['evaluation',{done:null,total:null,phase:'scoring',label:'성능 평가 · CPU'},'성능 평가 · CPU','평가 중'],
    ]){
      const project={id:'evaluation-prepare',status:'running',process_alive:true,stage_id:stage,progress};
      const original=JSON.stringify(project),p=page(project),counts=p.shell.querySelector('.story-voice-counts'),row=counts.children[0];
      assert.equal(p.shell.dataset.step,'evaluation');assert.equal(counts.hidden,false);assert.equal(counts.children.length,1);
      assert.equal(row.querySelector('strong').dataset.label,label);assert.equal(row.children[1].textContent,status);
      const bar=row.querySelector('progress'),indicator=row.querySelector('.story-progress-indicator');
      assert.equal(bar.hidden,false);assert.equal(bar.getAttribute('value'),null);assert.equal(bar.getAttribute('max'),null);
      assert.equal(bar.dataset.active,'true');assert.equal(indicator.dataset.status,'active');assert.equal(indicator.hidden,false);
      assert.ok(!textOf(row).includes('17 / 20'),'chapter counts cannot become evaluation session progress');
      p.ui.render();await p.next.fire('click');await p.flush();assert.equal(counts.children[0],row);
      assert.equal(JSON.stringify(project),original);assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
    }
  });
  await test('evaluation phases update the same work row through real scoring and result finalization',()=>{
    const project={id:'evaluation-phases',status:'running',process_alive:true,stage_id:'development',progress:{done:17,total:20,unit:'완료 단계'}};
    const p=page(project),counts=p.shell.querySelector('.story-voice-counts'),row=counts.children[0],bar=row.querySelector('progress');
    project.stage_id='evaluation';project.progress={done:null,total:null,unit:'개 음원',phase:'validation'};p.ui.render();
    assert.equal(counts.children[0],row);assert.equal(row.querySelector('strong').dataset.label,'평가 음원 검증');
    project.progress={done:0,total:48,unit:'개 음원',phase:'scoring',label:'성능 평가 · GPU'};p.ui.render();
    assert.equal(counts.children[0],row);assert.equal(row.querySelector('strong').dataset.label,'성능 평가 · GPU');
    assert.equal(bar.value,0);assert.equal(bar.max,48);assert.equal(row.children[1].textContent,'0 / 48개 음원');
    project.progress.done=48;project.progress.phase='finalizing';p.ui.render();
    assert.equal(row.querySelector('strong').dataset.label,'평가 결과 정리');assert.equal(bar.value,48);
    assert.equal(row.querySelector('.story-progress-indicator').dataset.status,'active','all sessions scored is not chapter completion');
    project.progress.done=null;project.progress.total=null;p.ui.render();
    assert.equal(counts.children[0],row);assert.equal(bar.hidden,false);assert.equal(bar.getAttribute('value'),null);assert.equal(row.children[1].textContent,'정리 중');
    assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
  });
  await test('evaluation preparation preserves real pause and stopped-worker states without starting anything',async()=>{
    for(const [status,alive,active,label] of [['pausing',true,true,'검증 중'],['paused',false,false,'일시정지'],['running',false,false,'대기 중']]){
      const p=page({id:'evaluation-paused',status,process_alive:alive,stage_id:'evaluation',progress:{done:null,total:null,phase:'validation'}});
      const row=p.shell.querySelector('.story-voice-counts').children[0];assert.equal(row.children[1].textContent,label);
      assert.equal(row.querySelector('progress').dataset.active,String(active));assert.equal(row.querySelector('.story-progress-indicator').hidden,!active);
      await p.next.fire('click');p.ui.render();assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
    }
  });
  await test('results preparation uses the shared centered work row until the actual report exists',async()=>{
    for(const [status,alive,active,state] of [['running',true,true,'정리 중'],['pausing',true,true,'정리 중'],['paused',false,false,'일시정지'],['ready',false,false,'대기 중']]){
      const project={id:'results-preparation',status,process_alive:alive,stage_id:'results',progress:{done:19,total:20,unit:'완료 단계'},results:{url:'/older-results/'}};
      const original=JSON.stringify(project),p=page(project),counts=p.shell.querySelector('.story-voice-counts'),row=counts.children[0];
      assert.equal(p.shell.classList.contains('preparing-results'),true);assert.equal(counts.hidden,false);assert.equal(counts.children.length,1);
      assert.equal(row.querySelector('strong').dataset.label,'학습 결과 정리');assert.equal(row.children[1].textContent,state);
      const bar=row.querySelector('progress'),indicator=row.querySelector('.story-progress-indicator');
      assert.equal(bar.hidden,false);assert.equal(bar.getAttribute('value'),null);assert.equal(bar.getAttribute('max'),null);
      assert.equal(bar.dataset.active,String(active));assert.equal(indicator.hidden,!active);
      assert.equal(p.shell.querySelector('.story-live-summary').hidden,true);assert.equal(p.get('results-section').hidden,true);
      assert.equal(p.viewModes.at(-1).view,'overview');assert.equal(p.next.disabled,true);
      p.ui.render();await p.next.fire('click');await p.flush();assert.equal(counts.children[0],row);
      assert.equal(JSON.stringify(project),original);assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
    }
  });
  await test('results preparation hands off to the completed wide iframe and never hides a real error',async()=>{
    const project={id:'results-handoff',status:'running',process_alive:true,stage_id:'results'};
    const p=page(project),frame=p.get('result-frame');assert.equal(p.get('results-section').hidden,true);
    project.error='보고서 정리를 확인하세요.';p.ui.render();assert.equal(p.shell.querySelector('.story-live-summary').hidden,false);
    delete project.error;project.status='completed';project.process_alive=false;project.results={url:'/results/current/',mic_url:'/mic/current/'};p.ui.render();
    await p.settle();
    assert.equal(p.shell.classList.contains('preparing-results'),false);assert.equal(p.shell.querySelector('.story-voice-counts').hidden,true);
    assert.equal(p.get('results-section').hidden,false);assert.equal(frame.getAttribute('src'),'/results/current/');assert.equal(p.viewModes.at(-1).view,'results');
    const css=fs.readFileSync(path.join(__dirname,'../korean_openwakeword/resources/studio_story.css'),'utf8');
    assert.match(css,/\.story-shell\.preparing-results\[data-step="results"\] :is\(\.story-content,\.story-live\)\{flex:0 1 auto;overflow:visible\}/);
    assert.match(css,/\.story-shell:is\(\[data-step="results"\],\[data-step="microphone"\]\) \.story-panel\{gap:12px;width:min\(1060px,100%\)\}/);
    assert.match(css,/iframe#result-frame\{width:100%;height:0;min-height:0;max-height:100%;border:0;border-radius:0;margin:0;flex:1 1 0;background:transparent\}/);
    assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
  });
  await test('new wakewords share the same workflow-driven chapter layout and pending work components',async()=>{
    const settings={pilot_review_policy:'one_per_voice_before_and_after_generation_v4',hard_review_policy:'one_per_phrase_v1'};
    for(const word of ['새봄아','나래야']) for(const [stage,step] of [['pilot','call-pilot-generation'],['positives','positive-generation'],
      ['hard_generation','hard-generation'],['features','features'],['package','training'],['development','evaluation'],['results','results']]){
      const project={id:`new-${word}-${stage}`,wakeword:word,status:'running',process_alive:true,stage_id:stage,settings,
        progress:{done:3,total:20,unit:'완료 단계'}};
      const p=page(project);assert.equal(p.shell.dataset.step,step);assert.equal(p.ui.ownsDownloadRows,true);
      const counts=p.shell.querySelector('.story-voice-counts');assert.equal(counts.hidden,false);assert.equal(counts.children.length,1);
      const bar=counts.querySelector('progress');assert.equal(bar.hidden,false);assert.equal(bar.getAttribute('value'),null);
      assert.equal(counts.querySelector('.story-progress-indicator').dataset.status,'active');
      await p.next.fire('click');assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
    }
  });
  await test('training preparation uses only the explicit backend phase and preserves genuine zero update counts',()=>{
    const project={id:'training-new-word',wakeword:'새봄아',status:'running',process_alive:true,stage_id:'training',progress:{done:0,total:30000,unit:'학습 업데이트'}};
    const p=page(project),counts=p.shell.querySelector('.story-voice-counts'),row=counts.children[0],bar=row.querySelector('progress');
    assert.equal(row.querySelector('strong').dataset.label,'모델 학습');assert.equal(row.children[1].textContent,'0 / 30,000회');assert.equal(bar.value,0);
    for(const phase of ['feature_validation','device_selection','model_initialization','gpu_cache_loading']){
      project.training_preparation={kind:'training_preparation_v1',phase,label:'학습 시작 준비',task_id:'current-task'};p.ui.render();
      assert.equal(counts.children[0],row);assert.equal(row.querySelector('strong').dataset.label,'학습 시작 준비');
      assert.equal(bar.getAttribute('value'),null);assert.equal(row.children[1].textContent,'준비 중');
    }
    project.training_preparation={kind:'training_preparation_v1',phase:'gpu_cache_loading',label:'학습 자료 적재',done:1024,total:4096,unit:'bytes'};p.ui.render();
    assert.equal(row.children[1].textContent,'1 KiB / 4 KiB');assert.equal(bar.value,1024);assert.equal(bar.max,4096);
    project.training_preparation={kind:'studio_task_progress_v1',phase:'feature_cache_verification',label:'기존 특징 배열 무결성·형식 검사',done:0,total:17179869184,unit:'bytes'};p.ui.render();
    assert.equal(row.querySelector('strong').dataset.label,'기존 특징 배열 무결성·형식 검사');
    assert.equal(row.children[1].textContent,'0 B / 16 GiB');assert.equal(bar.value,0);
    project.training_preparation={kind:'studio_task_progress_v1',phase:'training_finalizing',label:'학습 모델 내보내기·검증'};p.ui.render();
    assert.equal(row.children[1].textContent,'정리 중');assert.equal(bar.getAttribute('value'),null);
    delete project.training_preparation;project.progress.done=1;p.ui.render();
    assert.equal(counts.children[0],row);assert.equal(row.querySelector('strong').dataset.label,'모델 학습');assert.equal(bar.value,1);assert.equal(bar.max,30000);
    assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
  });
  await test('generation shows owned preparation and finalization inline with the actual file counter',()=>{
    for(const stage of ['pilot','positives','hard_generation']){
      const project={id:'phase-word',status:'running',process_alive:true,stage_id:stage,settings:{pilot_review_policy:'one_per_voice_before_and_after_generation_v4'},
        progress:{done:0,total:50,unit:'개',phase:'tts_model_loading',eta_eligible:false}};
      const p=page(project),row=p.shell.querySelector('.story-voice-counts').children[0],detail=row.querySelector('.story-generation-summary');
      assert.equal(detail.hidden,false);assert.equal(detail.textContent,'음성 생성 모델 준비 중');assert.equal(row.children[1].textContent,'0 / 50개');
      assert.equal(p.get('eta').hidden,true);
      project.progress={...project.progress,phase:'tts_generation',done:9,eta_eligible:true};p.ui.render();
      assert.equal(detail.hidden,true);assert.equal(row.children[1].textContent,'9 / 50개');assert.equal(p.get('eta').hidden,false);
      project.progress={...project.progress,phase:'tts_finalizing',done:50,eta_eligible:false,phase_elapsed_seconds:14,
        activity:{phase:'tts_finalizing',label:'검수 재생 목록 검증 중'}};p.ui.render();
      assert.equal(detail.hidden,false);assert.equal(detail.textContent,'검수 재생 목록 검증 중 · 14초 경과');
      assert.equal(row.children[1].textContent,'50 / 50개');assert.equal(p.get('eta').hidden,true);
      assert.equal(p.shell.querySelector('.story-voice-counts').children[0],row);
    }
  });
  await test("synthesis chapters keep the same progress elements through startup, pause, and a quality hold", () => {
    const settings = { pilot_review_policy: "one_per_voice_before_and_after_generation_v4", hard_samples_per_phrase: 20 };
    for (const [stage, label] of [["pilot", "호출어 샘플"], ["positives", "호출어 음성"], ["hard_generation", "유사어 음성"]]) {
      const project = { id: stage, stage_id: stage, status: "starting", process_alive: true, settings,
        progress: { done: 4, total: 16, unit: "완료 단계" } };
      const p = page(project), rows = p.shell.querySelector(".story-voice-counts"), row = rows.children[0];
      const title = row.querySelector("strong"), bar = row.querySelector("progress");
      const indicator = title.querySelector(".story-progress-indicator");
      const detail = title.querySelector(".story-generation-summary");
      assert.equal(p.shell.classList.contains("synthesis-chapter"), true);
      assert.equal(title.dataset.label, label);
      assert.deepEqual(title.children.slice(1), [detail, indicator]);
      assert.equal(bar.getAttribute("value"), null);
      assert.equal(row.children[1].textContent, "준비 중");
      assert.equal(indicator.dataset.status, "active");
      Object.assign(project, { status: "paused", process_alive: false, progress: { done: 19, total: 20, unit: "개" } });
      p.ui.render();
      assert.equal(rows.children[0], row);
      assert.equal(row.querySelector("progress"), bar);
      assert.equal(indicator.hidden, true);
      assert.equal(bar.value, 19);
      assert.equal(row.children[1].textContent, "19 / 20개");
      if (stage === "hard_generation") assert.equal(detail.textContent, "유사어 1종 · 문구당 20개 · 총 20개");
      Object.assign(project, { status: "waiting_input", input: { kind: "generation_policy" } });
      p.ui.render();
      assert.equal(rows.children[0], row);
      assert.equal(row.children[1].textContent, "19 / 20개 통과");
      assert.equal(p.next.disabled, true);
    }
  });
  await test('pilot quality hold preserves accepted progress and retries only on an explicit click', async () => {
    const input = { kind: "generation_policy", retry_action: "retry_generation", accepted_count: 19, target_count: 50, failed_count: 1 };
    const project = { id: "retry-pilot", status: "waiting_input", process_alive: true, stage_id: "pilot", input,
      settings: { pilot_review_policy: "one_per_voice_before_and_after_generation_v4" } };
    const p = page(project);
    const row = p.shell.querySelector(".story-voice-counts").children[0], bar = row.querySelector("progress");
    assert.equal(row.children[1].textContent, "19 / 50개 통과");
    assert.equal(bar.value, 19); assert.equal(bar.max, 50);
    assert.equal(row.querySelector(".story-progress-indicator").hidden, true);
    assert.equal(p.next.disabled, true);
    assert.equal(p.operation.hidden, false);
    assert.equal(p.operation.textContent, "실패한 샘플 다시 생성");
    assert.equal(p.operation.disabled, true, "a still-exiting worker cannot be retried");
    await p.operation.fire("click");
    project.process_alive = false; p.ui.render();
    assert.equal(p.operation.disabled, false);
    assert.ok(!p.calls.includes("retry_generation"), "polling and rendering never submit a retry");
    await p.next.fire("click");
    assert.ok(!p.calls.includes("resume") && !p.calls.includes("retry_generation"), "Next does not bypass the quality hold");
    await p.operation.fire("click"); await p.operation.fire("click");
    assert.equal(p.calls.filter(action => action === "retry_generation").length, 1);
    for (const unsupported of [{ ...project, input: { kind: "generation_policy" } }, { ...project, stage_id: "positives" }]) {
      const other = page(unsupported);
      assert.equal(other.operation.hidden, true, "other generation-policy holds do not advertise the pilot action");
      await other.operation.fire("click");
      assert.ok(!other.calls.includes("retry_generation"));
    }
  });
  await test("hard synthesis quality holds expose targeted recovery without advancing or restarting on render", async () => {
    for (const stage of ["hard_pilot", "hard_generation"]) {
      const project = { id: stage, stage_id: stage, status: "waiting_input", process_alive: false,
        can_edit_hard_phrases: true, settings: { hard_review_policy: "sample_before_generation_v2" },
        input: { kind: "generation_policy", retry_action: "retry_generation", accepted_count: 19, target_count: 20, failed_count: 1 } };
      const p = page(project, { action: (_action, state) => { state.busy = false; return true; } });
      const row = p.shell.querySelector(".story-voice-counts").children[0];
      assert.equal(p.shell.classList.contains("synthesis-chapter"), true);
      assert.equal(row.children[1].textContent, "19 / 20개 통과");
      assert.equal(row.querySelector(".story-progress-indicator").hidden, true);
      assert.equal(p.operation.textContent, stage === "hard_pilot" ? "실패한 샘플 다시 생성" : "실패한 유사어 다시 생성");
      assert.equal(p.operation.hidden, false);
      assert.equal(p.next.disabled, true);
      await p.next.fire("click");
      assert.equal(p.calls.some(action => ["retry_generation", "resume", "input"].includes(action)), false);
      await p.operation.fire("click");
      assert.equal(p.calls.filter(action => action === "retry_generation").length, 1);
      delete project.input.retry_action;
      p.ui.render();
      assert.equal(p.operation.hidden, true);
      assert.equal(p.back.disabled, false);
    }
  });
  await test('training preparation rejects unsupported records and old download toggles have no dead callback',async()=>{
    const project={id:'training-records',status:'paused',process_alive:false,stage_id:'training',progress:{done:0,total:30000,unit:'학습 업데이트'},
      training_preparation:{kind:'unrelated',phase:'gpu_cache_loading',label:'잘못된 준비'}};
    const p=page(project),counts=p.shell.querySelector('.story-voice-counts');assert.equal(counts.children[0].querySelector('strong').dataset.label,'모델 학습');
    project.training_preparation={kind:'training_preparation_v1',phase:'model_initialization',label:'모델 준비'};p.ui.render();
    assert.equal(counts.children[0].children[1].textContent,'일시정지');assert.equal(counts.querySelector('.story-progress-indicator').hidden,true);
    await p.get('downloads-section').fire('toggle',{target:{tagName:'DETAILS'}});
    assert.equal(p.calls.some(call=>['start','resume','pause','input'].includes(call)),false);
    const css=fs.readFileSync(path.join(__dirname,'../korean_openwakeword/resources/studio_story.css'),'utf8');
    assert.doesNotMatch(source,/paginateDownloads/);assert.doesNotMatch(css,/\.story-hard-connection/);
  });
  console.log(`Story UI: ${passed} tests passed`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
