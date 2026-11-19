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
      this.style = { setProperty(name, value) { this[name] = String(value); }, getPropertyValue(name) { return this[name] ?? ""; } };
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
    removeAttribute(name) { delete this.attributes[name]; }
    replaceChildren(...nodes) { this.children = nodes; }
    getBoundingClientRect() { return {left: 8, width: 280, top: 40, bottom: 80}; }
    focus() { this.focused = true; }
    addEventListener(name, fn) { (this.handlers[name] ||= []).push(fn); }
    fire(name) { this["on" + name]?.(); for (const fn of this.handlers[name] || []) fn(); }
  }
  const context = vm.createContext({ document: { createElement: tag => new Element(tag) } });
  context.window = context;
  context.innerHeight = 500; context.innerWidth = 768;
  vm.runInContext(source, context);
  const audio = new Element("audio"), controls = new Element("div");
  const previous = new Element("button"), play = new Element("button"), next = new Element("button");
  controls.append(previous, play, next);
  Object.assign(audio, { controls: true, duration: 120, currentTime: 30, src: "/unchanged.wav", volume: 0.7, playbackRate: 1.25, muted: false });
  return {...context.KowwListening, audio, controls, previous, play, next, Element};
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
  assert.equal(seek.style.getPropertyValue("--listen-progress"), "25%");
  seek.value = "750"; seek.fire("input");
  assert.equal(p.audio.currentTime, 90);
  assert.equal(seek.getAttribute("aria-valuetext"), "1:30 / 2:00");
  assert.equal(seek.style.getPropertyValue("--listen-progress"), "75%");
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

test("painted progress stays within the track for invalid times without changing the audio input", () => {
  const p = screen(), { seek, sync } = p.enhance(p.audio, p.controls, p);
  for (const [duration, currentTime, expected] of [
    [120, -30, "0%"], [120, 0, "0%"], [120, 240, "100%"],
    [120, NaN, "0%"], [120, Infinity, "0%"], [120, -Infinity, "0%"],
    [0, 30, "0%"], [-1, 30, "0%"], [NaN, 30, "0%"], [Infinity, 30, "0%"]
  ]) {
    Object.assign(p.audio, { duration, currentTime });
    const before = { duration, currentTime, src: p.audio.src, volume: p.audio.volume, rate: p.audio.playbackRate, muted: p.audio.muted };
    sync();
    assert.equal(seek.style.getPropertyValue("--listen-progress"), expected);
    assert.deepEqual({ duration: p.audio.duration, currentTime: p.audio.currentTime, src: p.audio.src,
      volume: p.audio.volume, rate: p.audio.playbackRate, muted: p.audio.muted }, before);
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
  assert.equal(seek.style.getPropertyValue("--listen-progress"), "100%");
  p.audio.duration = 366; p.audio.fire("durationchange"); assert.equal(seek.value, "500");
  assert.equal(seek.style.getPropertyValue("--listen-progress"), "50%");
  p.audio.duration = NaN; p.audio.currentTime = 0; p.audio.fire("emptied");
  assert.equal(time.textContent, "0:00 / 0:00"); assert.equal(seek.disabled, true);
  assert.equal(seek.style.getPropertyValue("--listen-progress"), "0%");
});

test("folder traversal matches the stable playback order and retains original items when selecting", () => {
  const p = screen();
  const samples = [
    {id: "eval", split: "validation", gender: "남성", age_group: "어린이", voice_number: 1},
    {id: "senior", split: "train", gender: "남성", age_group: "노년", voice_number: 1},
    {id: "child2", split: "train", gender: "남성", age_group: "어린이", voice_number: 2},
    {id: "child1", split: "train", gender: "남성", age_group: "어린이", voice_number: 1},
  ];
  const playlist = p.buildReviewPlaylist(samples);
  assert.deepEqual(Array.from(playlist.items, item => item.id), ["child1", "child2", "senior", "eval"]);
  assert.deepEqual(samples.map(item => item.id), ["eval", "senior", "child2", "child1"]);
  const [container, popup, list, button, label] = Array.from({length: 5}, () => new p.Element("div"));
  let selected = null;
  const picker = p.createFolderPicker({container, button, label, popup, list, folders: playlist.folders,
    describe: item => ({label: item.id}), onSelect: item => { selected = item; }});
  picker.setSelected(samples[0]);
  button.fire("click");
  assert.equal(list.children[0].getAttribute("aria-label"), "폴더 · 학습용 · 3개");
  list.children[0].fire("click");
  assert.equal(selected, null);
  list.children[1].fire("click");
  list.children[1].fire("click");
  assert.equal(list.getAttribute("aria-label"), "학습용 › 남성 › 어린이");
  list.children[1].fire("click");
  assert.equal(selected, samples[3]);
  assert.equal(popup.hidden, true);
  const nextQueue = p.buildReviewPlaylist([{id: "other", text: "다른 말"}], {similar: true});
  picker.refresh(nextQueue.folders);
  assert.equal(label.textContent, "음원 없음");
  button.fire("click");
  assert.equal(list.children[0].getAttribute("aria-label"), "폴더 · 다른 말 · 1개");
});

test("a large folder opens in compact ranges without changing global clip order", () => {
  const p = screen();
  const samples = Array.from({length: 205}, (_, index) => ({split: "train", voice_number: 205 - index}));
  const playlist = p.buildReviewPlaylist(samples, {folderSize: 100});
  const folder = playlist.folders[0];
  assert.equal(folder.count, 205);
  assert.deepEqual(Array.from(folder.folders, child => child.count), [100, 100, 5]);
  assert.equal(folder.items.length, 0);
  assert.deepEqual(Array.from(folder.folders.flatMap(child => child.items)), samples.toReversed());
  assert.deepEqual(Array.from(playlist.items), samples.toReversed());
});

console.log(`Listening player UI: ${passed} tests passed`);
