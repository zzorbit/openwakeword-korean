"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio_story.js"), "utf8");

// A DOM adapter exercises navigation and real-state gates without audio,
// network, an installed browser, or changes to a stored workflow.
function page(initialProject = null, options = {}) {
  const registry = new Map(), timers = [], calls = [];
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
      this.classList = {
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
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    getAttribute(name) { return this.attributes[name] ?? null; }
    focus() { this.focused = true; }
    showModal() { this.open = true; }
    close() { this.open = false; this.fire("close"); }
    fire(name, event = {}) { return Promise.all((this.handlers[name] || []).map(fn => fn(event))); }
  }
  const body = new Element("body"), legacy = new Element("main"); legacy.className = "dashboard-shell"; body.append(legacy);
  const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/studio.html"), "utf8");
  for (const match of html.matchAll(/id="([^"]+)"/g)) { const item = new Element(); item.id = match[1]; legacy.append(item); }
  const currentCard = new Element(); currentCard.className = "current-card"; currentCard.append(registry.get("current-title")); legacy.append(currentCard);
  const storage = new Element(); storage.className = "storage-location"; storage.append(registry.get("storage-directory")); legacy.append(storage);
  const get = id => {
    const found = registry.get(id);
    for (let parent = found; parent; parent = parent.parentElement) if (parent === body) return found;
    return null;
  };
  get("wakeword").value = "보라야"; get("create").disabled = false;
  const document = { body, createElement: tag => new Element(tag), getElementById: get, querySelector: selector => body.querySelector(selector), title: "" };
  const state = { project: initialProject, recipes: [], selectedPreset: "check", recipeReady: true, busy: false };
  const context = vm.createContext({ document, console, innerWidth: 1200, innerHeight: 900, matchMedia: () => ({ matches: false }),
    setTimeout: (callback, delay) => { const timer = { callback, delay }; timers.push(timer); return timer; },
    clearTimeout: timer => { if (timer) timer.canceled = true; } });
  context.window = context; context.addEventListener = () => {};
  vm.runInContext(source, context);
  const ui = context.KowwStory.create({
    state: () => state,
    materials: () => ({ training: [], evaluation: [] }),
    create: () => { state.busy = true; calls.push("create"); return options.create?.(state); },
    action: action => { state.busy = true; calls.push(action); },
    input: () => calls.push("input"), view: view => calls.push("view:" + view),
    pause: frame => calls.push("pause:" + frame.id), preview: kind => calls.push("preview:" + kind),
    logs: () => calls.push("logs"), message: text => calls.push("message:" + text),
  });
  const shell = get("story-shell"), actions = shell.querySelector(".story-actions");
  const next = actions.children[1], back = actions.children[0];
  function settle() { const timer = timers.find(timer => !timer.canceled && !timer.ran); if (!timer) return false; timer.ran = true; timer.callback(); return true; }
  ui.render();
  return { api: context.KowwStory, ui, shell, get, state, calls, timers, next, back, settle, ticks: shell.querySelectorAll(".story-index-step") };
}

async function main() {
  let passed = 0;
  async function test(name, fn) { await fn(); console.log("  ✓ " + name); passed++; }
  await test("backend current stage determines unlocked frontier; pending later stages unlock nothing", () => {
    const p = page();
    assert.equal(p.api.frontier({ status: "running", stage_id: "references", stages: [{ id: "evaluation", status: "pending" }] }), 4);
    assert.equal(p.api.frontier({ status: "waiting_review", stage: { id: "hard_review" } }), 9);
    assert.equal(p.api.frontier({ status: "running", stage_id: "unknown" }), 3);
    assert.equal(p.api.frontier({ status: "completed", results: { url: "/results/" } }), 13);
    assert.equal(p.api.frontier({ status: "completed", results: { url: "/results/", mic_url: "/microphone/" } }), 14);
  });
  await test("new setup advances locally and creates only once after material confirmation", async () => {
    const p = page();
    assert.equal(p.shell.dataset.step, "entry");
    assert.equal(p.ticks[0].parentElement.hidden, true);
    await p.next.fire("click");
    assert.equal(p.calls.includes("create"), false);
    assert.equal(p.timers[0].delay, 1500);
    await p.next.fire("click");
    assert.equal(p.timers.length, 1, "transition blocks a second click");
    p.settle(); assert.equal(p.shell.dataset.step, "scale");
    await p.next.fire("click"); p.settle(); assert.equal(p.shell.dataset.step, "materials");
    const creation = p.next.fire("click"); await p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.equal(p.timers[2].delay, 1500); p.settle(); await creation;
  });
  await test("fast creation starts immediately while successful screen reveal waits at least 1500 ms", async () => {
    const p = page(null, { create: state => {
      state.project = { id: "new-model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: {} };
      state.busy = false;
    } });
    await p.next.fire("click"); p.settle(); await p.next.fire("click"); p.settle();
    const creation = p.next.fire("click");
    assert.equal(p.calls.filter(call => call === "create").length, 1);
    p.ui.render(); assert.equal(p.shell.dataset.step, "materials", "poll cannot reveal a fast response early");
    assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    await p.next.fire("click"); assert.equal(p.calls.filter(call => call === "create").length, 1);
    assert.equal(p.timers[2].delay, 1500); p.settle(); await creation;
    assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(p.shell.querySelector(".story-loader").hidden, true);
  });
  await test("creation failure restores the material screen and controls after the minimum without retrying", async () => {
    const p = page(null, { create: state => { state.busy = false; return Promise.reject(new Error("creation unavailable")); } });
    await p.next.fire("click"); p.settle(); await p.next.fire("click"); p.settle();
    const creation = p.next.fire("click");
    p.ui.render(); assert.equal(p.shell.querySelector(".story-loader").hidden, false);
    p.settle(); await creation;
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
    await p.next.fire("click"); p.settle(); await p.back.fire("click");
    assert.equal(p.timers[1].delay, 1500); p.settle(); assert.equal(p.shell.dataset.step, "entry");
  });
  await test("running downloads never become complete from a display-only percent or timer", async () => {
    const project = { id: "model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: { done: 100, total: 100, files: [] } };
    const p = page(project);
    assert.equal(p.shell.dataset.step, "downloads"); assert.equal(p.next.textContent, "일시정지");
    assert.equal(p.ticks[4].hidden, true); assert.equal(p.timers.length, 0);
    await p.next.fire("click"); assert.deepEqual(p.calls.filter(call => call === "pause"), ["pause"]);
    assert.equal(p.calls.includes("start"), false);
    assert.equal(p.shell.dataset.step, "downloads");
  });
  await test("real backend advance follows once while polling preserves an explicit historical view", async () => {
    const project = { id: "model", wakeword: "보라야", status: "running", stage_id: "data_download", progress: {} };
    const p = page(project);
    project.stage_id = "references"; p.ui.render(); assert.equal(p.timers.length, 1);
    p.ui.render(); assert.equal(p.timers.length, 1); p.settle(); assert.equal(p.shell.dataset.step, "voices");
    await p.back.fire("click"); p.settle(); assert.equal(p.shell.dataset.step, "downloads");
    project.stage_id = "pilot"; p.ui.render(); assert.equal(p.shell.dataset.step, "downloads");
    assert.equal(p.next.textContent, "다음");
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
    p.ui.render(); assert.equal(p.shell.dataset.step, "downloads");
  });
  await test("review/input gates cannot be bypassed by the main next button", async () => {
    const p = page({ id: "model", status: "waiting_review", stage_id: "pilot_review", review: { url: "/review/" } });
    assert.equal(p.shell.dataset.step, "pronunciation"); assert.equal(p.next.disabled, true);
    await p.next.fire("click"); assert.equal(p.calls.includes("input"), false); assert.equal(p.timers.length, 0);
  });
  await test("completed result and microphone navigation pause child media without changing training", async () => {
    const p = page({ id: "model", wakeword: "보라야", status: "completed", results: { url: "/results/", mic_url: "/microphone/" } });
    assert.equal(p.shell.dataset.step, "results");
    await p.next.fire("click"); p.settle(); assert.equal(p.shell.dataset.step, "microphone");
    assert.ok(p.calls.includes("view:mic")); assert.equal(p.next.hidden, true);
    await p.back.fire("click"); p.settle(); assert.equal(p.shell.dataset.step, "results");
    assert.ok(p.calls.includes("pause:result-frame"));
    assert.equal(p.calls.some(call => ["create", "start", "resume", "input"].includes(call)), false);
  });
  console.log(`Story UI: ${passed} tests passed`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
