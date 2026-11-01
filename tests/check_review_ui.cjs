"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/review.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

async function screen(queue, search, saved = {}, config = {}, { playback } = {}) {
  const element=()=>({ textContent: "", innerHTML: "", children: [], attributes:{}, setAttribute(key,value){this.attributes[key]=value;}, removeAttribute(key){if(key==='src')delete this.src;else delete this.attributes[key];}, append(...nodes) { this.children.push(...nodes); },replaceChildren(...nodes){this.children=nodes;} });
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1],element()]));
  const calls = [], writes = [], listeners = {}, documentEvents={}, audioEffects = {pauses: 0, plays: 0};
  const audio=nodes.get("audio");audio.paused=true;audio.ended=false;
  audio.pause = () => { audioEffects.pauses++;audio.paused=true;audio.onpause?.(); };
  audio.play = async () => {audioEffects.plays++;audio.ended=false;audio.paused=false;audio.onplay?.();if(playback)return playback(audioEffects.plays);};
  audio.load = () => {audio.ended=false;};
  nodes.set("koww-page-config", { textContent: JSON.stringify(config) });
  const existing = JSON.stringify(saved);
  const context = vm.createContext({
    document: { hidden:false, addEventListener:(name,callback)=>{documentEvents[name]=callback;}, getElementById: id => nodes.get(id), createElement: element },
    URL, URLSearchParams, location: new URL("http://127.0.0.1:8794/" + search),
    localStorage: { getItem(key) { assert.equal(key, "koww:" + queue.queue_sha256); return existing; }, setItem(...args) { writes.push(args); } },
    fetch: async (url, options) => { calls.push({ url, options }); return { ok: true, json: async () => options?.method === "POST" ? { state: "saved" } : queue }; }
  });
  context.window = context;
  context.parent = {};
  context.addEventListener = (type, handler) => { listeners[type] = handler; };
  vm.runInContext(script, context);
  for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve));
  return { nodes, calls, writes, listeners, documentEvents, document:context.document, audioEffects, parent: context.parent };
}

(async () => {
  assert.match(html, /href="\/audio_listening\.css"/);
  assert.ok(html.indexOf('</style>') < html.indexOf('href="/audio_listening.css"'));
  assert.match(html, /class="review-player listen-player"/);
  assert.match(html, /class="listen-workspace"/);
  assert.match(html, /class="listen-sidebar"/);
  assert.match(html, /끝까지 들으면 정상으로 표시됩니다.*제외·다시 확인.*결과를 저장/);
  assert.match(html, /id="save">검수 결과 저장/);
  assert.match(html, /id="pause"[^>]*hidden/);
  assert.ok(html.indexOf('class="listen-heading"') < html.indexOf('id="audio"'));
  assert.ok(html.indexOf('id="audio"') < html.indexOf('class="buttons listen-controls"'));
  assert.ok(html.indexOf('class="buttons listen-controls"') < html.indexOf('id="items"'));
  const queue = { wakeword: "보라야", purpose: "합성 목소리가 자연스럽고 서로 다른지 확인하세요.", queue_sha256: "fixed-queue", items: [
    { id: "one", text: "오늘은 창밖으로 따뜻한 햇살이 들어옵니다.", audio: "/audio/0" },
    { id: "two", text: "오늘은 창밖으로 따뜻한 햇살이 들어옵니다.", audio: "/audio/1" }
  ] };
  const voices = await screen(queue, "?project=bora&queue=references", { one: { decision: "discard", listened: true } });
  assert.equal(voices.nodes.get("review-heading").textContent, "합성 목소리 확인");
  assert.match(voices.nodes.get("purpose").textContent, /‘보라야’를 생성/);
  assert.match(voices.nodes.get("purpose").textContent, /호출어 학습 음원으로 쓰지 않습니다/);
  assert.match(voices.nodes.get("expected").textContent, /^샘플 문장: 오늘은/);
  assert.match(voices.nodes.get("criteria").innerHTML, /목소리가 자연스럽/);
  assert.equal(voices.nodes.get("state").textContent, "제외");
  assert.match(voices.nodes.get('review-progress').textContent,/판정한 음원 1 \/ 2개 · 미판정 1개 · 다시 확인 0개/);
  assert.equal(voices.nodes.get('listening-list').children[0].children[1].textContent,'제외');
  assert.equal(voices.nodes.get('listening-list').children[0].attributes['data-number'],'1');
  assert.equal(voices.nodes.get('listening-list').children[0].attributes['aria-label'],'목소리 1 · 제외');
  assert.deepEqual(voices.writes, []);
  assert.deepEqual(voices.calls, [{ url: "/queue", options: undefined }]);
  assert.equal(voices.nodes.get("items").children.length, 2);
  voices.nodes.get("save").onclick();
  assert.match(voices.nodes.get("saved").textContent, /아직 1개/);
  assert.equal(voices.calls.length, 1); // A wording update never saves unheard answers.

  const pilot = await screen({ ...queue, kind: "pilot", purpose: "호출어 확인" }, "?queue=pilot");
  assert.equal(pilot.nodes.get("purpose").textContent, "호출어 확인");
  assert.match(pilot.nodes.get("expected").textContent, /^기대 문장:/);
  const content = await screen({ ...queue, policy_review: true, policy_text: "묶음 사용 동의" }, "?queue=content");
  assert.equal(content.nodes.get("policy").hidden, false);
  assert.equal(content.nodes.get("policy-text").textContent, "묶음 사용 동의");
  assert.match(content.nodes.get("criteria").innerHTML, /안내에 맞고/);
  const heard = { one: { decision: "keep", listened: true }, two: { decision: "keep", listened: true } };
  const config = { queue_url: "/review/queue?project=bora&queue=pilot", save_url: "/review/save?project=bora&queue=pilot", token: "studio-review-token", saved_message: "저장 완료 · 다음 단계를 준비합니다." };
  const nested = await screen(queue, "?queue=pilot", heard, config);
  await nested.nodes.get("save").onclick();
  assert.equal(nested.calls[0].url, config.queue_url);
  assert.equal(nested.calls[1].url, config.save_url);
  assert.equal(nested.calls[1].options.headers["X-Koww-Token"], config.token);
  assert.deepEqual(JSON.parse(nested.calls[1].options.body), heard);
  assert.equal(nested.nodes.get("saved").textContent, config.saved_message);
  const standalone = await screen(queue, "?queue=pilot", heard);
  await standalone.nodes.get("save").onclick();
  assert.equal(standalone.calls[1].url, "/save");
  assert.equal(standalone.calls[1].options.headers["X-Koww-Token"], undefined);
  assert.deepEqual(JSON.parse(standalone.calls[1].options.body), heard);
  const paused = await screen(queue, "?queue=pilot");
  paused.nodes.get("items").value = "1";
  paused.nodes.get("items").onchange();
  const title = paused.nodes.get("title").textContent, source = paused.nodes.get("audio").src;
  const pausesBefore=paused.audioEffects.pauses;
  paused.listeners.message({origin: "https://elsewhere.example", source: paused.parent, data: {type: "koww:pause"}});
  paused.listeners.message({origin: "http://127.0.0.1:8794", source: {}, data: {type: "koww:pause"}});
  paused.listeners.message({origin: "http://127.0.0.1:8794", source: paused.parent, data: {type: "other"}});
  assert.equal(paused.audioEffects.pauses, pausesBefore);
  paused.listeners.message({origin: "http://127.0.0.1:8794", source: paused.parent, data: {type: "koww:pause"}});
  assert.equal(paused.audioEffects.pauses, pausesBefore+1);
  assert.equal(paused.nodes.get("title").textContent, title);
  assert.equal(paused.nodes.get("audio").src, undefined);
  assert.equal(paused.writes.length, 0);
  assert.equal(paused.calls.length, 1);
  // Playback and page visibility, not drawing or navigation, supply listening evidence.
  const heardOnly=await screen({...queue,kind:'pilot'},'?queue=pilot');
  heardOnly.nodes.get('audio').onended();
  assert.equal(heardOnly.writes.length,0);
  await heardOnly.nodes.get('audio').play();
  heardOnly.nodes.get('audio').ended=true;heardOnly.nodes.get('audio').onended();
  assert.deepEqual(JSON.parse(heardOnly.writes.at(-1)[1]),{one:{decision:'keep',listened:true}});
  assert.equal(heardOnly.nodes.get('listening-list').children[0].children[1].textContent,'정상');
  heardOnly.nodes.get('next').onclick();
  await heardOnly.nodes.get('audio').play();
  heardOnly.document.hidden=true;heardOnly.documentEvents.visibilitychange();
  const writesBefore=heardOnly.writes.length;
  heardOnly.nodes.get('audio').ended=true;heardOnly.nodes.get('audio').onended();
  assert.equal(heardOnly.writes.length,writesBefore);
  assert.equal(heardOnly.nodes.get('audio').src,undefined);
  assert.equal(heardOnly.calls.length,1); // Listening never submits an approval.
  const continuousReview=await screen({...queue,kind:'pilot'},'?queue=pilot');
  await continuousReview.nodes.get('start').onclick();
  continuousReview.nodes.get('audio').ended=true;continuousReview.nodes.get('audio').onended();
  assert.equal(continuousReview.nodes.get('items').value,1);
  continuousReview.nodes.get('audio').ended=true;continuousReview.nodes.get('audio').onended();
  assert.equal(continuousReview.nodes.get('items').value,1); // Required review stops at the end, rather than replaying approvals.
  assert.equal(continuousReview.nodes.get('start').textContent,'재생');
  assert.equal(continuousReview.nodes.get('start').attributes['aria-pressed'],'false');
  assert.equal(continuousReview.calls.length,1);
  assert.match(continuousReview.nodes.get('review-progress').textContent,/판정한 음원 2 \/ 2개 · 미판정 0개/);
  const many=await screen({...queue,kind:'pilot',items:Array.from({length:7},(_,i)=>({id:`voice-${i}`,text:'보라야',audio:`/audio/${i}`}))},'?queue=pilot');
  assert.equal(many.nodes.get('listening-list').children.length,5);
  many.nodes.get('list-next').onclick();
  assert.equal(many.nodes.get('listening-list').children.length,2);
  assert.equal(many.nodes.get('items').value,0,'list paging leaves the current audio and decisions untouched');
  many.nodes.get('listening-list').children[1].onclick();
  assert.equal(many.nodes.get('items').value,6);
  assert.equal(many.audioEffects.plays,0);assert.equal(many.writes.length,0);
  many.nodes.get('review').onclick();
  assert.match(many.nodes.get('review-progress').textContent,/판정한 음원 1 \/ 7개 · 미판정 6개 · 다시 확인 1개/);
  assert.equal(many.nodes.get('listening-list').children[1].children[1].textContent,'다시 확인');
  const recheck=await screen(queue,'?queue=pilot',{...heard,two:{decision:'review',listened:true}},config);
  await recheck.nodes.get('save').onclick();
  assert.equal(recheck.calls.length,2,'recheck decisions still reach the existing backend hold/save policy');
  assert.match(recheck.nodes.get('saved').textContent,/문제 항목을 확인/);
  const toggled=await screen({...queue,kind:'pilot'},'?queue=pilot');
  await toggled.nodes.get('start').onclick();
  assert.equal(toggled.nodes.get('start').textContent,'일시정지');
  toggled.nodes.get('audio').currentTime=0.8;
  await toggled.nodes.get('start').onclick();
  assert.equal(toggled.nodes.get('audio').paused,true);
  assert.equal(toggled.nodes.get('start').textContent,'재생');
  await toggled.nodes.get('start').onclick();
  assert.equal(toggled.nodes.get('audio').currentTime,0.8,'play resumes without reloading the selected recording');
  assert.equal(toggled.writes.length,0,'starting and pausing never records a listening decision');
  const policy=await screen({...queue,policy_review:true},'?queue=content',heard);
  await policy.nodes.get('save').onclick();
  assert.equal(policy.calls.length,1,'unchecked source policy never sends approval');
  policy.nodes.get('policy-approved').checked=true;await policy.nodes.get('save').onclick();
  assert.equal(JSON.parse(policy.calls[1].options.body).policy_approved,true);
  const failed=await screen({...queue,kind:'pilot'},'?queue=pilot',{}, {},{playback:()=>Promise.reject(new Error('device failed'))});
  await failed.nodes.get('start').onclick();
  assert.match(failed.nodes.get('state').textContent,/재생하지 못했습니다/);
  assert.equal(failed.nodes.get('start').textContent,'재생');
  failed.nodes.get('audio').ended=true;failed.nodes.get('audio').onended();
  assert.equal(failed.writes.length,0,'failed playback supplies no listening evidence');
  const mediaError=await screen({...queue,kind:'pilot'},'?queue=pilot');
  await mediaError.nodes.get('start').onclick();mediaError.nodes.get('audio').onerror();
  assert.equal(mediaError.nodes.get('start').textContent,'재생');
  mediaError.nodes.get('audio').ended=true;mediaError.nodes.get('audio').onended();
  assert.equal(mediaError.writes.length,0);
  let rejectOld;
  const pending=await screen({...queue,kind:'pilot'},'?queue=pilot',{}, {},{playback:count=>count===1?new Promise((_,reject)=>{rejectOld=reject;}):Promise.resolve()});
  pending.nodes.get('start').onclick();await pending.nodes.get('next').onclick();
  rejectOld(new Error('old request interrupted'));await new Promise(resolve=>setImmediate(resolve));
  assert.equal(pending.nodes.get('audio').src,'/audio/1');
  assert.equal(pending.nodes.get('start').textContent,'일시정지');
  for(const audio of ['https://elsewhere.example/private.wav','javascript:alert(1)','',null]){
    const invalid=await screen({...queue,items:[{...queue.items[0],audio}]},'?queue=pilot');
    assert.match(invalid.nodes.get('saved').textContent,/음원 경로가 올바르지 않습니다/);
    assert.equal(invalid.nodes.get('audio').src,undefined);
    assert.equal(invalid.writes.length,0);
  }
  const outside=await screen(queue,'?queue=pilot',heard,{...config,save_url:'https://elsewhere.example/save'});
  await outside.nodes.get('save').onclick();assert.equal(outside.calls.length,1,'token cannot be sent to a foreign origin');
  const foreignQueue=await screen(queue,'?queue=pilot',{}, {...config,queue_url:'https://elsewhere.example/queue'});
  assert.equal(foreignQueue.calls.length,0);
  console.log("Review UI: voice purpose, saved decisions, unheard gate and other review types passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
