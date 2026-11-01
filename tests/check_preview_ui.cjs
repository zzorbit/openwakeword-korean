"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/reference_preview.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

async function screen(mode, items, { error, ok = true, playback } = {}) {
  const nodes = new Map();
  class Element {
    constructor() { this.value = ""; this.textContent = ""; this.children = []; this.handlers = {}; this.attributes = {}; this.played = 0; this.pauses = 0; this.paused = true; this.ended = false; }
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    addEventListener(event, callback) { this.handlers[event] = callback; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    get src() { return this.attributes.src; }
    set src(value) { this.attributes.src=value; this.ended=false; }
    load() { this.ended=false; }
    play() { this.played++; this.paused=false; this.handlers.play?.(); return playback ? playback(this.played) : Promise.resolve(); }
    pause() { this.pauses++; this.paused=true; this.handlers.pause?.(); }
    finish() { this.ended=true; this.paused=true; this.handlers.pause?.(); this.handlers.ended?.(); }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], new Element());
  const calls = [], messages = {}, documentEvents = {}, document = { title: "기준 목소리 듣기", hidden:false, addEventListener:(name,callback)=>{documentEvents[name]=callback;}, getElementById: id => nodes.get(id), createElement: () => new Element() };
  const context = vm.createContext({ document, URL, location: new URL(`http://127.0.0.1:8793/${mode}/?project=bora`),
    fetch: async (url, options) => { calls.push({ url, options }); return { ok, json: async () => ({ items, error }) }; } });
  context.window = context; context.parent = {};
  context.addEventListener = (name, callback) => { messages[name] = callback; };
  vm.runInContext(script, context);
  for (let index = 0; index < 8; index++) await new Promise(resolve => setImmediate(resolve));
  return { nodes, calls, document, documentEvents, messages, parent: context.parent };
}

(async () => {
  assert.match(html, /href="\/audio_listening\.css"/);
  assert.ok(html.indexOf('</style>') < html.indexOf('href="/audio_listening.css"'));
  assert.match(html, /class="listen-workspace"/);
  assert.match(html, /class="listen-sidebar"/);
  assert.match(html, /class="listen-player"/);
  assert.match(html, /듣기는 선택 사항이며 따로 승인할 필요가 없습니다/);
  assert.ok(html.indexOf('class="listen-heading"') < html.indexOf('id="audio"'));
  assert.ok(html.indexOf('id="audio"') < html.indexOf('class="controls listen-controls"'));
  assert.ok(html.indexOf('class="controls listen-controls"') < html.indexOf('id="voice"'));
  assert.doesNotMatch(html, /<form|localStorage|listened|decision|whole_pool_policy_approved/);
  const items = [
    { id: "hard1", text: "돌아와", voice_family_id: "train_01", audio: "/hard-preview/audio/0?project=bora" },
    { id: "hard2", text: "기다려", voice_family_id: "train_02", audio: "/hard-preview/audio/1?project=bora" }
  ];
  const hard = await screen("hard-preview", items);
  assert.equal(hard.document.title, "유사어 전체 듣기");
  assert.equal(hard.nodes.get("voice").attributes["aria-label"], "들을 유사어 음원");
  assert.equal(hard.nodes.get("voice").children[0].textContent, "돌아와 · 목소리 1");
  assert.equal(hard.nodes.get("audio").src, items[0].audio);
  assert.equal(hard.nodes.get("previous").disabled, true);
  assert.equal(hard.nodes.get("listening-list").children.length,2);
  assert.equal(hard.nodes.get("listening-list").children[0].attributes['aria-current'],'true');
  assert.equal(hard.nodes.get("listening-list").children[0].attributes['data-number'],'1');
  assert.equal(hard.nodes.get("listening-list").children[0].attributes['aria-label'],'돌아와 · 목소리 1 · 미청취');
  assert.equal(hard.nodes.get("listening-list").children[0].children[1].textContent,'미청취');
  // Native audio play has the same continuous behavior as the visible play button.
  await hard.nodes.get("audio").play();
  assert.equal(hard.nodes.get("audio").played, 1);
  hard.nodes.get("audio").finish();
  assert.equal(hard.nodes.get("audio").src, items[1].audio);
  assert.equal(hard.nodes.get("audio").played, 2);
  assert.equal(hard.nodes.get("listening-list").children[0].children[1].textContent,'들음');
  assert.equal(hard.nodes.get("next").disabled, true);
  hard.nodes.get("audio").finish();
  assert.equal(hard.nodes.get("audio").src, items[0].audio);
  assert.equal(hard.nodes.get("audio").played, 3);
  assert.equal(hard.nodes.get("continuous").textContent, "일시정지");
  hard.nodes.get("audio").pause();
  hard.nodes.get("audio").handlers.ended();
  assert.equal(hard.nodes.get("audio").played, 3);
  assert.equal(hard.nodes.get("continuous").textContent, "재생");
  assert.deepEqual(hard.calls, [{ url: "/hard-preview/queue?project=bora", options: undefined }]);
  const reference = await screen("reference-preview", [{ id: "train1", group: "학습용", text: "오늘은", audio: "/reference-preview/audio/0?project=bora" }]);
  assert.equal(reference.document.title, "기준 목소리 듣기");
  assert.equal(reference.nodes.get("voice").children[0].textContent, "학습용 · 목소리 1");
  assert.equal(reference.nodes.get("audio").played, 0); // Opening is never autoplay.
  assert.deepEqual(reference.calls, [{ url: "/reference-preview/queue?project=bora", options: undefined }]);
  const many=await screen('reference-preview',Array.from({length:7},(_,i)=>({id:`voice-${i}`,group:'학습용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`})));
  assert.equal(many.nodes.get('listening-list').children.length,5);
  many.nodes.get('list-next').handlers.click();
  assert.equal(many.nodes.get('listening-list').children.length,2);
  assert.equal(many.nodes.get('voice').value,'0','browsing a list page never selects or plays an audio');
  many.nodes.get('listening-list').children[1].handlers.click();
  assert.equal(many.nodes.get('voice').value,'6');
  assert.equal(many.nodes.get('audio').played,0,'selecting a paused recording stays paused');
  assert.equal(many.calls.length,1,'list pagination makes no approval or extra data request');
  const paused = await screen("hard-preview", items);
  paused.nodes.get("next").handlers.click();
  paused.nodes.get("continuous").handlers.click();
  const pausesBefore=paused.nodes.get("audio").pauses;
  paused.messages.message({origin: "https://other.example", source: paused.parent, data: {type: "koww:pause"}});
  paused.messages.message({origin: "http://127.0.0.1:8793", source: {}, data: {type: "koww:pause"}});
  assert.equal(paused.nodes.get("audio").pauses, pausesBefore);
  paused.messages.message({origin: "http://127.0.0.1:8793", source: paused.parent, data: {type: "koww:pause"}});
  assert.equal(paused.nodes.get("audio").pauses, pausesBefore+1);
  assert.equal(paused.nodes.get("audio").src, undefined);
  assert.equal(paused.nodes.get("voice").value, "1");
  assert.equal(paused.nodes.get("continuous").textContent, "재생");
  assert.equal(paused.calls.length, 1);
  paused.nodes.get("continuous").handlers.click();
  assert.equal(paused.nodes.get("audio").src,items[1].audio);
  paused.document.hidden=true;paused.documentEvents.visibilitychange();
  assert.equal(paused.nodes.get("audio").paused,true);
  assert.equal(paused.nodes.get("audio").src,undefined);
  const playsBefore=paused.nodes.get("audio").played;
  paused.nodes.get("audio").handlers.ended();
  assert.equal(paused.nodes.get("audio").played,playsBefore);

  // A cancelled play request must not overwrite a later successful selection.
  let rejectOld;
  const pending=await screen("hard-preview",items,{playback:count=>count===1?new Promise((_,reject)=>{rejectOld=reject;}):Promise.resolve()});
  pending.nodes.get("continuous").handlers.click();
  pending.nodes.get("next").handlers.click();
  rejectOld(new Error("old request was interrupted"));
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(pending.nodes.get("audio").src,items[1].audio);
  assert.equal(pending.nodes.get("error").textContent,"");
  assert.equal(pending.nodes.get("continuous").textContent,"일시정지");
  const failedPlayback=await screen("hard-preview",items,{playback:()=>Promise.reject(new Error("device failed"))});
  failedPlayback.nodes.get("continuous").handlers.click();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(failedPlayback.nodes.get("continuous").textContent,"재생");
  assert.match(failedPlayback.nodes.get("error").textContent,/재생하지 못했습니다/);
  failedPlayback.nodes.get("audio").finish();assert.equal(failedPlayback.nodes.get("audio").played,1);
  const mediaError=await screen("hard-preview",items);
  mediaError.nodes.get("continuous").handlers.click();mediaError.nodes.get("audio").handlers.error();
  assert.equal(mediaError.nodes.get("continuous").textContent,"재생");
  mediaError.nodes.get("audio").finish();assert.equal(mediaError.nodes.get("audio").played,1);
  for (const audio of ["https://example.com/private.wav", "javascript:alert(1)", "", null]) {
    const invalid = await screen("hard-preview", [{ ...items[0], audio }]);
    assert.match(invalid.nodes.get("error").textContent, /음원 경로가 올바르지 않습니다/);
    assert.equal(invalid.nodes.get("audio").src, undefined);
  }
  const empty = await screen("hard-preview", []);
  assert.equal(empty.nodes.get("error").textContent, "준비된 유사어 음원이 없습니다.");
  const missing = await screen("hard-preview", null);
  assert.equal(missing.nodes.get("error").textContent, "음원 목록이 올바르지 않습니다.");
  const failed = await screen("hard-preview", [], { ok: false, error: "준비 자료가 없습니다." });
  assert.equal(failed.nodes.get("error").textContent, "준비 자료가 없습니다.");
  const unsafeText = await screen("hard-preview", [{ ...items[0], text: "<img src=x onerror=bad()>" }]);
  assert.equal(unsafeText.nodes.get("sentence").textContent, "<img src=x onerror=bad()>");
  console.log("Read-only preview UI: hard/reference playback, no consent writes, source validation and errors passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
