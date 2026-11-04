"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio_story.js"), "utf8");

// A DOM adapter exercises navigation and real-state gates without audio,
// network, an installed browser, or changes to a stored workflow.
function page(initialProject = null, options = {}) {
  const registry = new Map(), timers = [], calls = [], creationOptions = [], viewModes = [];
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
      this.handlers = {}; this.attributes = {}; this.dataset = {}; this.style = { setProperty() {} };
      this.hidden = false; this.disabled = false; this.textContent = ""; this.value = ""; this.open = false;
      if (tag === "audio") { this.paused = true; this.ended = false; this.playCalls = 0; this.loadCalls = 0; }
      this.classList = {
        contains: name => this.className.split(/\s+/).includes(name),
        add: name => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), name])].join(" "); },
        remove: name => { this.className = this.className.split(/\s+/).filter(value => value !== name).join(" "); },
        toggle: (name, force) => { if (force) this.classList.add(name); else this.classList.remove(name); },
      };
    }
    set id(value) { this._id = value; registry.set(value, this); }
    get id() { return this._id || ""; }
    get firstElementChild() { return this.children[0] || null; }
    append(...children) {
      for (const child of children) {
        if (child.parentElement) child.parentElement.children = child.parentElement.children.filter(item => item !== child);
        child.parentElement = this; this.children.push(child);
      }
    }
    prepend(child) { this.append(child); this.children = [child, ...this.children.filter(item => item !== child)]; }
    replaceChildren(...children) { for (const child of this.children) child.parentElement = null; this.children = []; this.append(...children); }
    remove() { if (this.parentElement) this.parentElement.children = this.parentElement.children.filter(child => child !== this); this.parentElement = null; }
    querySelectorAll(selector) { return this.children.flatMap(child => [...(matches(child, selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
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
  const state = { project: initialProject, recipes: [], selectedPreset: "check", recipeReady: true, busy: false };
  const windowHandlers = {};
  const context = vm.createContext({ document, console, location: { origin: "http://studio.local" }, innerWidth: options.width || 1200, innerHeight: options.height || 900, matchMedia: () => ({ matches: false }),
    setTimeout: (callback, delay) => { const timer = { callback, delay, at: clock + delay }; timers.push(timer); return timer; },
    clearTimeout: timer => { if (timer) timer.canceled = true; } });
  context.window = context; context.addEventListener = (name, fn) => { (windowHandlers[name] ||= []).push(fn); };
  context.getComputedStyle = element => ({ animationDuration: "0s", animationDelay: "0s", transitionDuration: "0s", transitionDelay: "0s", ...options.getComputedStyle?.(element) });
  vm.runInContext(source, context);
  const ui = context.KowwStory.create({
    state: () => state,
    materials: () => ({ training: [], evaluation: [] }),
    mediaUrl: value => typeof value === "string" && value.startsWith("/") && !value.startsWith("//") ? value : "",
    create: input => { state.busy = true; calls.push("create"); creationOptions.push(JSON.parse(JSON.stringify(input || {}))); return options.create?.(state); },
    action: action => { state.busy = true; calls.push(action); return options.action?.(action, state); },
    input: () => calls.push("input"), view: (view, mode) => { calls.push("view:" + view); viewModes.push({ view, mode }); },
    pause: frame => calls.push("pause:" + frame.id), preview: kind => calls.push("preview:" + kind),
    logs: () => calls.push("logs"), message: text => calls.push("message:" + text),
  });
  const shell = get("story-shell"), actions = shell.querySelector(".story-actions");
  const next = actions.children[2], operation = actions.children[1], back = actions.children[0];
  function advanceTimer() { const timer = timers.filter(timer => !timer.canceled && !timer.ran).sort((a, b) => a.at - b.at)[0]; if (!timer) return false; clock = Math.max(clock, timer.at); timer.ran = true; timer.callback(); return true; }
  async function settle() { let advanced = false; await flush(); while (advanceTimer()) { advanced = true; await flush(); } return advanced; }
  ui.render();
  return { api: context.KowwStory, ui, shell, get, state, calls, timers, creationOptions, viewModes, next, operation, back, settle, advanceTimer, flush, now: () => clock, elapse: value => { clock += value; }, windowEvent: async (name, event) => { for (const fn of windowHandlers[name] || []) fn(event); await flush(); }, element: tag => new Element(tag), visibility: async hidden => { document.hidden = hidden; await Promise.all((documentHandlers.visibilitychange || []).map(fn => fn())); }, ticks: shell.querySelectorAll(".story-index-step") };
}

async function main() {
  let passed = 0;
  async function test(name, fn) { await fn(); console.log("  ✓ " + name); passed++; }
  await test("backend current stage determines unlocked frontier; pending later stages unlock nothing", () => {
    const p = page();
    assert.equal(p.api.frontier({ status: "running", stage_id: "references", stages: [{ id: "evaluation", status: "pending" }] }), 4);
    assert.equal(p.api.frontier({ status: "waiting_review", stage: { id: "hard_review" } }), 10);
    assert.equal(p.api.frontier({ status: "running", stage_id: "unknown" }), 3);
    assert.equal(p.api.frontier({ status: "completed", results: { url: "/results/" } }), 14);
    assert.equal(p.api.frontier({ status: "completed", results: { url: "/results/", mic_url: "/microphone/" } }), 15);
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
    assert.equal(p.shell.dataset.step, "pronunciation"); assert.equal(visible(), 7);
    assert.equal(p.next.disabled, true); assert.equal(p.ticks[7].hidden, true); assert.equal(p.ticks[7].disabled, true);
    await p.ticks[7].fire("click"); assert.equal(p.timers.length, 0);
    await p.ticks[0].fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(p.ticks[0].parentElement.hidden, false);
    assert.equal(visible(), 7); assert.equal(p.ticks[0].getAttribute("aria-current"), "step");
    assert.equal(p.ticks[6].getAttribute("aria-current"), "false");
    project.stage_id = "hard_review"; p.ui.render();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(visible(), 11);
    assert.equal(p.ticks[10].hidden, false); assert.equal(p.ticks[11].hidden, true);
    const timerCount = p.timers.length;
    await p.ticks[11].fire("click"); assert.equal(p.timers.length, timerCount);
    assert.equal(p.shell.dataset.step, "entry");
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("index extent resets to the selected model and a fresh setup without leaking later progress", async () => {
    const p = page({ id: "finished", wakeword: "보라야", status: "completed", results: { url: "/results/" } });
    assert.equal(p.ticks.filter(tick => !tick.hidden).length, 15);
    p.state.project = { id: "earlier", wakeword: "다른이름", status: "running", stage_id: "references", progress: {} };
    p.ui.render();
    assert.equal(p.shell.dataset.step, "voices"); assert.equal(p.ticks.filter(tick => !tick.hidden).length, 5);
    assert.equal(p.ticks[5].hidden, true);
    p.state.project = null; p.ui.render();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(p.ticks[0].parentElement.hidden, true);
    await p.next.fire("click"); await p.settle();
    assert.equal(p.ticks.filter(tick => !tick.hidden).length, 2); assert.equal(p.ticks[2].hidden, true);
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("sixteen single-role pages keep action titles punctuated and voice listening separate from preparation", () => {
    const definitions = page().api.definitions;
    assert.equal(definitions.length, 16); assert.ok(definitions.every(([, , title]) => title.endsWith(".")));
    const voices = definitions.findIndex(([id]) => id === "voices");
    assert.equal(definitions[voices + 1][0], "voice-listening"); assert.equal(definitions[voices + 2][0], "pronunciation");
  });
  await test("history pagination fits tall narrow, short and desktop column layouts without shrinking text", () => {
    const { pageCapacity } = page().api;
    const fits = (items, available, height, columns, gap) => {
      const capacity = pageCapacity(items, available, height, columns, gap);
      const rows = Math.ceil(Math.min(items, capacity) / columns);
      const used = rows * height + Math.max(0, rows - 1) * gap + (capacity < items ? 44 : 0);
      assert.ok(used <= available, `${used} exceeds ${available}`); return capacity;
    };
    assert.equal(fits(9, 734, 128, 1, 12), 5, "319×950 single-column download list paginates");
    assert.equal(fits(9, 330, 80, 1, 12), 3, "600×540 short download list paginates");
    assert.equal(fits(9, 600, 80, 2, 4), 9, "1280×800 desktop keeps the complete two-column list");
    assert.equal(fits(20, 330, 44, 1, 10), 5, "phrase fields reserve pager space");
    assert.equal(fits(6, 734, 44, 1, 10), 6, "tall narrow phrase fields all fit when appropriate");
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
    await p.settle(); assert.equal(p.shell.dataset.step, "scale");
    await p.next.fire("click"); await p.settle(); assert.equal(p.shell.dataset.step, "materials");
    const creation = p.next.fire("click"); await p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.deepEqual(p.creationOptions, [{ navigationCheckpoints: true }]);
    assert.equal(p.calls.includes("start"), false, "model creation does not start downloads");
    assert.equal(p.timers[2].delay, 800); await p.settle(); await creation;
  });
  await test("fast creation starts immediately while the visible notice keeps its full 800 ms hold", async () => {
    const p = page(null, { create: state => {
      state.project = { id: "new-model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: {} };
      state.busy = false;
    } });
    await p.next.fire("click"); await p.settle(); await p.next.fire("click"); await p.settle();
    const creation = p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    await p.flush();
    p.ui.render(); assert.equal(p.shell.dataset.step, "materials", "poll cannot reveal a fast response early");
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.next.fire("click"); assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.equal(p.timers[2].delay, 800); await p.settle(); await creation;
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
  });
  await test("creation failure restores the material screen and controls after the notice hold without retrying", async () => {
    const p = page(null, { create: state => { state.busy = false; return Promise.reject(new Error("creation unavailable")); } });
    await p.next.fire("click"); await p.settle(); await p.next.fire("click"); await p.settle();
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
    await p.next.fire("click"); assert.equal(p.timers.length, 0); assert.ok(p.calls.some(call => call.startsWith("message:")));
    p.get("wakeword").value = "보라야";
    await p.next.fire("click"); await p.settle(); await p.back.fire("click");
    assert.equal(p.timers[1].delay, 800); await p.settle(); assert.equal(p.shell.dataset.step, "entry");
  });
  function motion(element, duration) {
    let resolve, reject;
    const animation = { effect: { target: element, getComputedTiming: () => ({ endTime: duration }) },
      currentTime: 0, playbackRate: 1, playState: "running", finished: new Promise((yes, no) => { resolve = yes; reject = no; }),
      finish() { this.playState = "finished"; this.currentTime = duration; resolve(); },
      cancel() { this.playState = "idle"; reject(new Error("Animation canceled")); } };
    return animation;
  }
  await test("both directions hold the fully visible notice for 800 ms after real reveal completion", async () => {
    const phases = [];
    const p = page(null, { getComputedStyle: () => ({ animationDuration: "240ms" }), animations: element => {
      const animation = motion(element, element.id === "story-panel" ? 200 : 240); phases.push(animation);
      return [animation, motion({}, Infinity)]; // An infinite child effect cannot hold the parent transition.
    } });
    const loader = p.shell.querySelector(".story-loader"), panel = p.get("story-panel");
    for (const [control, before, after] of [[p.next, "entry", "scale"], [p.back, "scale", "entry"]]) {
      const start = p.now();
      await control.fire("click");
      assert.equal(loader.hidden, true, "the loader is not shown before the panel actually finishes");
      assert.equal(panel.inert, true); assert.equal(panel.className.includes("is-leaving"), true);
      p.elapse(200); phases.at(-1).finish(); await p.flush();
      assert.equal(loader.hidden, false); assert.equal(phases.length % 3, 2);
      p.elapse(240); phases.at(-1).finish(); await p.flush();
      assert.equal(loader.className.includes("is-finishing"), false);
      const hold = p.timers.filter(timer => !timer.canceled && !timer.ran).at(-1);
      assert.equal(hold.delay, 800, "the full hold starts only after the reveal finishes");
      assert.equal(hold.at - start, 1240);
      p.elapse(799); await p.flush();
      assert.equal(loader.className.includes("is-finishing"), false, "the notice remains fully visible for the whole hold");
      p.advanceTimer(); await p.flush();
      assert.equal(p.now() - start, 1240);
      assert.equal(loader.className.includes("is-finishing"), true);
      assert.equal(p.shell.dataset.step, before, "the old panel remains hidden during the loader fade-out");
      assert.equal(panel.inert, true);
      await control.fire("click");
      p.elapse(240); phases.at(-1).finish(); await p.flush();
      assert.equal(p.shell.dataset.step, after); assert.equal(loader.hidden, true);
      assert.equal(loader.className.includes("is-finishing"), false); assert.equal(panel.inert, false);
      assert.equal(p.now() - start, 1480);
    }
  });
  await test("slow animation completion is never replaced by the nominal CSS duration", async () => {
    const phases = [];
    const p = page(null, { getComputedStyle: () => ({ animationDuration: "240ms" }), animations: element => {
      const animation = motion(element, 800); phases.push(animation); return [animation];
    } });
    await p.next.fire("click"); p.elapse(700);
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
    phases[0].finish(); await p.flush(); p.elapse(700);
    assert.equal(p.shell.dataset.step, "entry");
    phases[1].finish(); await p.flush();
    assert.equal(p.timers.filter(timer => !timer.canceled && !timer.ran).at(-1).delay, 800);
    p.advanceTimer(); await p.flush(); p.elapse(700);
    assert.equal(p.shell.dataset.step, "entry", "the destination waits for the exit completion too");
    phases[2].finish(); await p.flush();
    assert.equal(p.shell.dataset.step, "scale"); assert.equal(p.now(), 2900);
  });
  await test("no-motion and reduced-motion paths keep the full notice hold without a completion event", async () => {
    const p = page(null, { getComputedStyle: () => ({ animationDuration: "0s", transitionDuration: "0s" }) });
    await p.next.fire("click"); assert.equal(p.shell.dataset.step, "entry");
    assert.equal(p.timers.at(-1).delay, 800);
    await p.settle(); assert.equal(p.now(), 800); assert.equal(p.shell.dataset.step, "scale");
  });
  await test("a missing finite completion signal is recovered by finishing that motion before advancing", async () => {
    const phases = [];
    const p = page(null, { animations: element => { const animation = motion(element, 200); phases.push(animation); return [animation]; } });
    await p.next.fire("click");
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
    p.advanceTimer(); await p.flush();
    assert.equal(phases[0].playState, "finished", "watchdog recovery reaches the actual final animation state first");
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.windowEvent("pagehide");
  });
  await test("a slow model response holds the loader and never repeats model creation", async () => {
    let resolveCreate;
    const p = page(null, { create: state => new Promise(resolve => { resolveCreate = () => {
      state.project = { id: "created", status: "ready", stage_id: "data_download" }; state.busy = false; resolve();
    }; }) });
    await p.next.fire("click"); await p.settle(); await p.next.fire("click"); await p.settle();
    const creation = p.next.fire("click"); await p.flush(); p.advanceTimer(); await p.flush(); p.elapse(3000);
    assert.equal(p.shell.dataset.step, "materials"); assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.next.fire("click"); assert.equal(p.calls.filter(value => value === "create").length, 1);
    resolveCreate(); await creation;
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.calls.includes("start"), false);
  });
  await test("model changes and page hiding cancel stale animations and queued watchdogs", async () => {
    for (const cancel of ["model", "pagehide"]) {
      const phases = [];
      const p = page(null, { animations: element => { const animation = motion(element, 200); phases.push(animation); return [animation]; } });
      await p.next.fire("click"); const staleWatchdog = p.timers[0].callback;
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
    const p = page(null, { animations: element => { const animation = motion(element, 200); phases.push(animation); return [animation]; } });
    await p.next.fire("click"); phases[0].cancel(); await p.flush();
    assert.equal(p.shell.dataset.step, "entry"); assert.equal(p.get("story-panel").inert, false);
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
  });
  await test("page hiding releases a pending model-creation transition without waiting for its network response", async () => {
    const p = page(null, { create: () => new Promise(() => {}) });
    await p.next.fire("click"); await p.settle(); await p.next.fire("click"); await p.settle();
    const creation = p.next.fire("click"); await p.flush();
    await p.windowEvent("pagehide"); await creation;
    assert.equal(p.get("story-panel").inert, false); assert.equal(p.shell.querySelector(".story-loader").hidden, true);
    assert.equal(p.calls.filter(value => value === "create").length, 1);
  });
  await test("running downloads never become complete from a display-only percent or timer", async () => {
    const project = { id: "model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: { done: 100, total: 100, files: [] } };
    const p = page(project);
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true);
    assert.equal(p.ticks[4].hidden, true); assert.equal(p.timers.length, 0);
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
    return { id: "navigated", wakeword: "언년아", preset: "check", status: "paused", stage_id: stages[complete.length].id,
      navigation_checkpoint: { policy: "explicit_story_v1", state: "awaiting_next", step }, stages, progress: {},
      reference_counts: { train: 20, validation: 30 }, reference_preview: { count: 50, train: 20, validation: 30, url: "/reference-preview/" } };
  };
  await test("every explicit story boundary waits for proven completion and next releases the next real group once", async () => {
    for (const [step] of navigationGroups.slice(0, -1).filter(([step]) => step !== "voices")) {
      const project = navigationProject(step), target = navigationGroups[navigationGroups.findIndex(([id]) => id === step) + 1][0];
      const p = page(project, { action: (_action, state) => {
        state.project.navigation_checkpoint = { policy: "explicit_story_v1", state: "running", step: target };
        state.project.status = "running"; p.ui.render();
        assert.equal(p.timers.length, 0, "a busy release response cannot reveal the next screen early");
        state.busy = false; p.ui.render();
      } });
      assert.equal(p.shell.dataset.step, step); assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, false);
      assert.equal(p.operation.hidden, true); p.ui.render(); assert.equal(p.calls.includes("resume"), false);
      await p.next.fire("click"); await p.next.fire("click");
      assert.equal(p.calls.filter(call => call === "resume").length, 1); assert.equal(p.timers.length, 1);
      assert.equal(p.timers[0].delay, 800); await p.settle(); assert.equal(p.shell.dataset.step, target);
      assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true);
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
      if (["missing", "failed"].includes(variant)) assert.equal(p.api.frontier(project), 4);
    }
  });
  await test("unfinished results work retains its status and central recovery controls before the result page exists", async () => {
    for (const marker of [false, true]) for (const status of ["ready", "running", "paused", "error", "interrupted"]) {
      const project = { id: "results-worker", status, stage_id: "results", progress: {} };
      if (marker) project.results = { url: "/previous-results/" };
      if (marker) project.navigation_checkpoint = { policy: "explicit_story_v1", step: "results", state: "running" };
      const p = page(project);
      assert.equal(p.shell.dataset.step, "results"); assert.equal(p.shell.querySelector(".story-live-summary").hidden, false);
      assert.equal(p.get("story-title").textContent, "학습 결과를 준비합니다.");
      assert.equal(p.get("results-section").hidden, true, "a stale report URL does not replace active result preparation");
      assert.equal(p.viewModes.at(-1).view, "overview");
      assert.equal(p.operation.hidden, false); assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true);
      await p.next.fire("click"); assert.equal(p.calls.some(call => ["start", "pause", "resume"].includes(call)), false);
      await p.operation.fire("click");
      assert.equal(p.calls.includes(status === "running" ? "pause" : status === "ready" ? "start" : "resume"), true);
    }
  });
  await test("explicit download start, pause and resume use a separate operation button while footer next stays gated", async () => {
    const project = checkpointProject(), p = page(project), primary = p.shell.querySelector(".story-download-primary");
    assert.equal(primary.parentElement.hidden, false); assert.equal(primary.textContent, "다운로드 시작");
    assert.equal(p.next.textContent, "다음"); assert.equal(p.next.disabled, true);
    await p.next.fire("click"); assert.equal(p.calls.includes("start"), false);
    await primary.fire("click"); await primary.fire("click"); assert.equal(p.calls.filter(call => call === "start").length, 1);
    p.state.busy = false; project.status = "running"; project.stage_id = "data_download"; project.download_checkpoint.state = "running";
    project.progress = { done: 100, total: 100 }; p.ui.render();
    assert.equal(primary.parentElement.hidden, true); assert.equal(p.operation.textContent, "일시정지"); assert.equal(p.next.disabled, true);
    await p.next.fire("click"); assert.equal(p.calls.includes("pause"), false);
    await p.operation.fire("click"); assert.equal(p.calls.filter(call => call === "pause").length, 1);
    p.state.busy = false; project.status = "paused"; p.ui.render();
    assert.equal(primary.parentElement.hidden, true); assert.equal(p.operation.textContent, "재개"); await p.operation.fire("click");
    assert.equal(p.calls.filter(call => call === "resume").length, 1);
    assert.equal(p.shell.dataset.step, "downloads");
  });
  await test("live download pages use allocated height and measured rows while preserving real progress and operation gates", async () => {
    for (const [width, height, available, rowHeight, columns, liveGap, expected] of [
      [319, 950, 518, 128, 1, 12, 3],
      [600, 540, 205, 76, 1, 8, 2],
      [1280, 800, 460, 80, 2, 18, 9],
    ]) {
      const project = checkpointProject("running", "running"); project.stage_id = "data_download";
      project.progress = { done: 17, total: 100, files: [{ id: "real", done: 17, total: 100 }] };
      const p = page(project, { width, height, getComputedStyle: item => ({
        gridTemplateColumns: item.id === "downloads-list" ? Array(columns).fill("200px").join(" ") : "none",
        rowGap: item.className === "story-live" ? `${liveGap}px` : "0px",
      }) });
      const list = p.get("downloads-list"), section = p.get("downloads-section");
      section.getBoundingClientRect = () => ({ top: 0, bottom: available + 36, height: available + 36 });
      list.getBoundingClientRect = () => ({ top: 36, bottom: available + 36, height: available });
      const groups = Array.from({ length: 9 }, (_, i) => {
        const group = p.element("li"); group.className = "download-group"; group.textContent = `자료 ${i + 1}`;
        group.getBoundingClientRect = () => ({ height: group.hidden ? 0 : rowHeight }); return group;
      });
      list.append(...groups); p.ui.render();
      assert.equal(p.shell.dataset.downloadList, "true", "the live panel requests a definite available height");
      const visible = groups.filter(group => !group.hidden), pager = p.shell.querySelector(".story-live").querySelector(".story-pagination");
      assert.equal(visible.length, expected, `${width}×${height} uses the allocated list space`);
      const used = Math.ceil(visible.length / columns) * rowHeight + (pager.hidden ? 0 : 32 + liveGap);
      assert.ok(used <= available, `${width}×${height}: ${used} exceeds ${available}`);
      assert.equal(pager.hidden, expected === groups.length); assert.equal(p.next.disabled, true);
      if (!pager.hidden) {
        await pager.children[2].fire("click");
        assert.equal(groups[0].hidden, true); assert.equal(groups[expected].hidden, false);
        p.ui.render(); assert.equal(groups[expected].hidden, false, "polling keeps the selected download page");
      }
      assert.equal(project.progress.done, 17); assert.equal(p.calls.some(call => ["start", "pause", "resume"].includes(call)), false);
      p.state.project = null; p.ui.render();
      assert.equal(p.shell.dataset.downloadList, "false", "new setup does not inherit the full-height download layout");
    }
  });
  await test("download checkpoint holds a completed data phase until next explicitly releases it and follows a busy response once", async () => {
    const project = checkpointProject("awaiting_next", "paused"); project.stage_id = "references";
    for (const record of project.stages) if (record.id !== "references") record.status = "completed";
    const p = page(project, { action: (_action, state) => {
      state.project.download_checkpoint.state = "released"; state.project.status = "running";
      p.ui.render(); assert.equal(p.timers.length, 0, "busy state cannot start a transition early");
      state.busy = false; p.ui.render();
    } });
    const primary = p.shell.querySelector(".story-download-primary");
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.api.frontier(project), 3);
    assert.equal(p.next.disabled, false); assert.equal(primary.parentElement.hidden, true);
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
      const p = page(project), primary = p.shell.querySelector(".story-download-primary");
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
    const p = page(project), migration = p.get("migration-note"), primary = p.shell.querySelector(".story-download-primary");
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.shell.querySelector(".story-live-summary").hidden, true);
    assert.equal(migration.hidden, false); assert.equal(migration.parentElement.className, "story-live");
    assert.equal(p.get("migrate").parentElement, migration, "the original migration button stays connected and exposed");
    assert.equal(p.next.disabled, true); assert.equal(primary.parentElement.hidden, true);
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
  await test("completed result and microphone navigation pause child media without changing training", async () => {
    const p = page({ id: "model", wakeword: "보라야", status: "completed", results: { url: "/results/", mic_url: "/microphone/" } });
    assert.equal(p.shell.dataset.step, "results");
    await p.next.fire("click"); await p.settle(); assert.equal(p.shell.dataset.step, "microphone");
    assert.ok(p.calls.includes("view:mic")); assert.equal(p.next.hidden, true);
    await p.back.fire("click"); await p.settle(); assert.equal(p.shell.dataset.step, "results");
    assert.ok(p.calls.includes("pause:result-frame"));
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  await test("saved model entry preserves the recorded word without editing or overwriting a new-model draft", async () => {
    const model = { id: "model", wakeword: "언년아", status: "completed", results: { url: "/results/" } };
    const p = page(model);
    p.get("wakeword").value = "새호출어";
    await p.ticks[0].fire("click"); await p.settle();
    const saved = p.get("story-saved-wakeword");
    assert.equal(p.shell.dataset.step, "entry");
    assert.equal(saved.parentElement.hidden, false);
    assert.equal(saved.value, "언년아"); assert.equal(saved.readOnly, true);
    assert.equal(saved.disabled, false, "read-only text keeps its normal appearance and can be selected");
    assert.equal(p.get("wakeword").parentElement.hidden, true);
    assert.equal(p.shell.querySelector(".story-history").hidden, true);
    p.ui.render(); assert.equal(saved.value, "언년아");
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
  const historyModel = views => ({ id: "history", wakeword: "언년아", preset: "production", status: "completed", stages: savedStages(), completed_stage_views: views, results: { url: "/results/" } });
  const textOf = item => [item.textContent, ...item.children.map(textOf)].join(" ");
  async function visit(p, step) { const position = p.api.definitions.findIndex(([id]) => id === step); await p.ticks[position].fire("click"); await p.settle(); }
  await test("saved scale uses separate disabled cards and never overwrites the next model's chosen preset", async () => {
    const p = page(historyModel({})); p.state.selectedPreset = "trial";
    await visit(p, "scale");
    const group = p.get("story-saved-presets"), radios = group.querySelectorAll("input");
    assert.equal(group.parentElement.hidden, false); assert.equal(radios.length, 3);
    assert.equal(radios.filter(radio => radio.checked).length, 1);
    assert.equal(radios.find(radio => radio.checked).value, "production");
    assert.ok(radios.every(radio => radio.disabled)); assert.equal(p.state.selectedPreset, "trial");
    p.ui.render(); assert.equal(p.state.selectedPreset, "trial");
    p.state.project = null; p.ui.render(); await p.next.fire("click"); await p.settle();
    assert.equal(p.shell.dataset.step, "scale"); assert.equal(group.parentElement.hidden, true);
    assert.equal(p.state.selectedPreset, "trial");
  });
  await test("completed counts come from saved rows rather than new preset targets and retain full progress", async () => {
    const p = page(historyModel({ "positive-generation": { completed: true, rows: [
      { label: "학습용 호출어", done: 1000, total: 1000, unit: "개" },
      { label: "공통 평가용 호출어", done: 2000, total: 2000, unit: "개" },
    ] } }));
    p.state.project.settings = { originals: 100000 };
    await visit(p, "positive-generation");
    const history = p.shell.querySelector(".story-history"), rows = history.querySelectorAll(".story-history-row");
    assert.equal(history.hidden, false); assert.equal(rows.length, 2);
    assert.ok(textOf(history).includes("1,000 / 1,000개"));
    assert.ok(!textOf(history).includes("100,000"));
    assert.ok(rows.every(row => row.querySelector("progress").value === row.querySelector("progress").max));
    assert.ok(!textOf(history).includes("이 단계를 마쳤습니다"));
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
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
    assert.equal(history.querySelectorAll("progress").length, 2);
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
    const history = p.shell.querySelector(".story-history"), rows = history.querySelectorAll(".story-history-row");
    assert.equal(rows.length, 3);
    assert.ok(textOf(rows[0]).includes("1,500.25 / 1,500시간"));
    assert.ok(textOf(rows[1]).includes("750 / 1,500시간"));
    assert.ok(textOf(rows[1]).includes("50%"));
    assert.ok(textOf(rows[2]).includes("— / 500시간"));
    assert.equal(rows[2].querySelectorAll("progress").length, 0);
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
  await test("saved review transport preserves playing selection, loops on end, pauses explicitly and retains saved decisions", async () => {
    const model = historyModel(savedListeningView()), original = JSON.stringify(model.completed_stage_views);
    const p = page(model); await visit(p, "hard-review");
    const player = p.shell.querySelector(".listen-player"), audio = player.querySelector(".listen-audio"), select = player.querySelector(".listen-select");
    const [previous, toggle, next] = player.querySelector(".listen-controls").children;
    assert.equal(previous.disabled, true); assert.equal(next.disabled, false); assert.equal(toggle.textContent, "재생");
    await toggle.fire("click"); assert.equal(audio.paused, false); assert.equal(toggle.textContent, "일시정지");
    const loads = audio.loadCalls; p.ui.render(); p.ui.render(); assert.equal(audio.loadCalls, loads, "polling preserves playback");
    select.value = "1"; await select.fire("change");
    assert.equal(audio.paused, false); assert.ok(audio.getAttribute("src").includes("/1?"));
    assert.equal(previous.disabled, false); assert.equal(next.disabled, true);
    audio.paused = true; audio.ended = true; await audio.fire("pause"); await audio.fire("ended");
    assert.ok(audio.getAttribute("src").includes("/0?")); assert.equal(audio.paused, false, "last item loops to the first while playing");
    await toggle.fire("click"); assert.equal(audio.paused, true); assert.equal(toggle.textContent, "재생");
    await next.fire("click"); assert.ok(audio.getAttribute("src").includes("/1?")); assert.equal(audio.paused, true, "paused selection stays paused");
    audio.ended = true; await audio.fire("ended"); assert.ok(audio.getAttribute("src").includes("/1?"), "ended cannot restart after user pause");
    await next.fire("click"); assert.ok(audio.getAttribute("src").includes("/1?"), "manual next is bounded at the last item");
    await previous.fire("click"); assert.ok(audio.getAttribute("src").includes("/0?"));
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
  console.log(`Story UI: ${passed} tests passed`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
