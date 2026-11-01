"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/listening_player.js"), "utf8");

// Exercise the shipped helper without a network, audio device, or browser.
function screen() {
  class Element {
    constructor(tag) {
      this.tagName = tag; this.children = []; this.attributes = {}; this.handlers = {};
      this.className = ""; this.textContent = ""; this.value = "";
      this.classList = { add: (...names) => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), ...names])].join(" "); } };
    }
    append(...nodes) { this.children.push(...nodes); }
    insertBefore(node, anchor) {
      const index = this.children.indexOf(anchor);
      assert.notEqual(index, -1, "the replacement uses an existing transport anchor");
      this.children.splice(index, 0, node);
    }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    getAttribute(name) { return this.attributes[name] ?? null; }
    addEventListener(name, fn) { (this.handlers[name] ||= []).push(fn); }
    fire(name) { this["on" + name]?.(); for (const fn of this.handlers[name] || []) fn(); }
  }
  const context = vm.createContext({ document: { createElement: tag => new Element(tag) } });
  context.window = context;
  vm.runInContext(source, context);
  const audio = new Element("audio"), controls = new Element("div");
  const previous = new Element("button"), play = new Element("button"), next = new Element("button");
  controls.append(previous, play, next);
  Object.assign(audio, { controls: true, duration: 120, currentTime: 30, src: "/unchanged.wav", volume: 0.7, playbackRate: 1.25, muted: false });
  return { enhance: context.KowwListening.enhance, audio, controls, previous, play, next };
}

let passed = 0;
function test(label, run) { run(); passed++; console.log("  ✓ " + label); }

test("invalid mounts preserve native controls; valid replacement mounts once before hiding them", () => {
  const p = screen();
  assert.equal(p.enhance(p.audio, p.controls, { play: p.play, previous: p.previous }), null);
  assert.equal(p.audio.controls, true); assert.equal(p.controls.children.length, 3);
  let native = true, hiddenAfterReplacement = false;
  Object.defineProperty(p.audio, "controls", { get: () => native, set(value) {
    if (!value) hiddenAfterReplacement = p.controls.children.length === 6;
    native = value;
  } });
  const handle = p.enhance(p.audio, p.controls, p);
  assert.equal(hiddenAfterReplacement, true); assert.equal(p.audio.controls, false);
  assert.equal(p.controls.children[2], handle.seek);
  assert.equal(p.controls.children.at(-1), handle.mute);
  const listenerCount = Object.values(p.audio.handlers).reduce((sum, listeners) => sum + listeners.length, 0);
  assert.equal(p.enhance(p.audio, p.controls, p), handle);
  assert.equal(p.controls.children.length, 6);
  assert.equal(Object.values(p.audio.handlers).reduce((sum, listeners) => sum + listeners.length, 0), listenerCount);
});

test("enhancement leaves the review controller's native events and transport actions authoritative", () => {
  const p = screen(), calls = [];
  const onplay = () => calls.push("play"), onended = () => calls.push("ended"), click = () => calls.push("click");
  p.audio.onplay = onplay; p.audio.onended = onended; p.play.onclick = click;
  p.previous.onclick = click; p.next.onclick = click;
  p.enhance(p.audio, p.controls, p);
  assert.equal(p.audio.onplay, onplay); assert.equal(p.audio.onended, onended);
  assert.equal(p.play.onclick, click); assert.equal(p.previous.onclick, click); assert.equal(p.next.onclick, click);
  assert.deepEqual(calls, [], "mounting never plays, approves, or completes a recording");
  p.audio.fire("play"); p.audio.fire("ended"); p.play.fire("click");
  assert.deepEqual(calls, ["play", "ended", "click"]);
});

test("seeking changes only currentTime and updates the public position without altering the audio input", () => {
  const p = screen(), before = { src: p.audio.src, volume: p.audio.volume, rate: p.audio.playbackRate, muted: p.audio.muted };
  const { seek } = p.enhance(p.audio, p.controls, p);
  assert.equal(seek.disabled, false); assert.equal(seek.value, "250");
  seek.value = "750"; seek.fire("input");
  assert.equal(p.audio.currentTime, 90);
  assert.equal(seek.getAttribute("aria-valuetext"), "1:30 / 2:00");
  assert.deepEqual({ src: p.audio.src, volume: p.audio.volume, rate: p.audio.playbackRate, muted: p.audio.muted }, before);
});

test("missing or invalid duration disables seeking and cannot create a playback position", () => {
  for (const duration of [NaN, 0, Infinity]) {
    const p = screen(); p.audio.duration = duration; p.audio.currentTime = 0;
    const { seek } = p.enhance(p.audio, p.controls, p);
    assert.equal(seek.disabled, true); assert.equal(seek.value, "0");
    seek.value = "900"; seek.fire("input");
    assert.equal(p.audio.currentTime, 0); assert.equal(seek.value, "0");
    assert.equal(seek.getAttribute("aria-valuetext"), "0:00 / 0:00");
  }
});

test("mute exposes its real toggle state and never changes volume, playback rate, or source", () => {
  const p = screen(), { mute } = p.enhance(p.audio, p.controls, p);
  assert.equal(mute.getAttribute("aria-label"), "음소거"); assert.equal(mute.getAttribute("aria-pressed"), "false");
  mute.fire("click");
  assert.equal(p.audio.muted, true); assert.equal(mute.getAttribute("aria-label"), "소리 켜기"); assert.equal(mute.getAttribute("aria-pressed"), "true");
  assert.equal(p.audio.volume, 0.7); assert.equal(p.audio.playbackRate, 1.25); assert.equal(p.audio.src, "/unchanged.wav");
  mute.fire("click"); assert.equal(p.audio.muted, false);
  p.audio.muted = true; p.audio.fire("volumechange");
  assert.equal(mute.getAttribute("aria-pressed"), "true", "external mute changes update the displayed state");
});

test("metadata, time, duration, and emptied events keep elapsed time and the seek position synchronized", () => {
  const p = screen(), { seek } = p.enhance(p.audio, p.controls, p);
  const time = p.controls.children.find(node => node.className === "listen-time");
  p.audio.duration = 183; p.audio.currentTime = 61.9; p.audio.fire("loadedmetadata");
  assert.equal(time.textContent, "1:01 / 3:03");
  p.audio.currentTime = 183; p.audio.fire("timeupdate"); assert.equal(seek.value, "1000");
  p.audio.duration = 366; p.audio.fire("durationchange"); assert.equal(seek.value, "500");
  p.audio.duration = NaN; p.audio.currentTime = 0; p.audio.fire("emptied");
  assert.equal(time.textContent, "0:00 / 0:00"); assert.equal(seek.disabled, true);
});

console.log(`Listening player UI: ${passed} tests passed`);
