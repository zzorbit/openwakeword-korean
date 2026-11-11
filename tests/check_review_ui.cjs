"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/review.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

async function screen(queue, search, saved = {}, config = {}, { playback, sharedControls = false, queueGate, queueError,respond,actualPlaying=true,mainHeight=160,embedded=true } = {}) {
  const element=()=>({ textContent: "", innerHTML: "", children: [], attributes:{}, setAttribute(key,value){this.attributes[key]=value;}, removeAttribute(key){if(key==='src')delete this.src;else delete this.attributes[key];}, append(...nodes) { this.children.push(...nodes); },replaceChildren(...nodes){this.children=nodes;},insertBefore(node,before){this.children.splice(this.children.indexOf(before),0,node);} });
  const nodes = new Map([...html.matchAll(/id="([^"]+)"/g)].map(match => [match[1],element()]));
  const calls = [], writes = [], readiness = [], parentMessages=[], listeners = {}, documentEvents={}, resizeObservers=[],audioEffects = {pauses: 0, plays: 0};
  const classes = new Set(), enhanced = [], pickers = [];
  const audio=nodes.get("audio");audio.paused=true;audio.ended=false;audio.currentTime=0;
  audio.duration=1;audio.played={length:1,start:()=>0,end:()=>1};
  audio.pause = () => { audioEffects.pauses++;audio.paused=true;audio.onpause?.(); };
  audio.play = async () => {audioEffects.plays++;audio.ended=false;audio.paused=false;audio.onplay?.();if(playback)await playback(audioEffects.plays);if(typeof actualPlaying==='function'?actualPlaying(audioEffects.plays):actualPlaying)audio.onplaying?.({timeStamp:1000+audioEffects.plays});};
  audio.load = () => {audio.ended=false;};
  nodes.set("koww-page-config", { textContent: JSON.stringify(config) });
  const existing = JSON.stringify(saved);
  const context = vm.createContext({
    document: { hidden:false, documentElement:{classList:{add(...values){for(const value of values)classes.add(value);},remove(...values){for(const value of values)classes.delete(value);}}}, addEventListener:(name,callback)=>{documentEvents[name]=callback;}, getElementById: id => nodes.get(id), createElement: element,querySelector:()=>({getBoundingClientRect:()=>({height:typeof mainHeight==='function'?mainHeight():mainHeight})}) },
    ResizeObserver:class{constructor(callback){this.callback=callback;resizeObservers.push(this);}observe(){}},
    URL, URLSearchParams, performance:{now:()=>100}, location: new URL("http://127.0.0.1:8794/" + search),
    localStorage: { getItem(key) { assert.equal(key, "koww:" + (queue.review_phase?queue.review_phase+':':'') + queue.queue_sha256); return existing; }, setItem(...args) { writes.push(args); } },
    fetch: async (url, options) => { calls.push({ url, options }); if(queueGate)await queueGate;if(queueError)throw Error(queueError);const result=respond?await respond(url,options):{payload:options?.method === "POST" ? { state: "saved" } : queue};return { ok:result.ok!==false,json:async()=>JSON.parse(JSON.stringify(result.payload)),text:async()=>result.payload?.error||'저장 실패' }; }
  });
  context.window = context;
  context.parent = embedded?{postMessage:(message,origin)=>parentMessages.push({message:JSON.parse(JSON.stringify(message)),origin})}:context;
  context.KowwChapterReady={loading:()=>readiness.push({status:'loading'}),ready:()=>readiness.push({status:'ready',title:nodes.get('title').textContent,source:audio.src}),fail:message=>readiness.push({status:'error',message})};
  if(sharedControls)context.KowwListening={
    enhance(audio,controls,options){enhanced.push({audio,controls,options});},
    createPicker(options){
      const picker={
        options,items:[],selected:null,closed:false,badgeWrites:[],revealed:[],open:false,
        setItems(items){this.items=items;},setSelected(index){this.selected=index;},
        setBadge(index,text){this.badgeWrites.push({index,text});this.items[index].badge=text;},
        reveal(request){this.revealed.push(request);},show(request=1){this.open=true;this.request=request;options.onOpenChange?.(true,request);},
        close(){this.closed=true;if(this.open){this.open=false;options.onOpenChange?.(false,this.request);}},
      };
      pickers.push(picker);
      return picker;
    },
  };
  context.addEventListener = (type, handler) => { listeners[type] = handler; };
  vm.runInContext(script, context);
  for (let i = 0; i < 5; i++) await new Promise(resolve => setImmediate(resolve));
  return { nodes, calls, writes, readiness, listeners, documentEvents, document:context.document, audioEffects, parent: context.parent, classes, enhanced, pickers,parentMessages,resizeObservers,flush:async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));} };
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
  assert.deepEqual(voices.readiness,[{status:'loading'},{status:'ready',title:'1 / 2',source:'/audio/0'}],'readiness follows initial queue validation and visible player setup');
  let releaseQueue;const queueGate=new Promise(resolve=>{releaseQueue=resolve;});
  const pendingQueue=await screen(queue,'?queue=references',{}, {},{queueGate});
  assert.deepEqual(pendingQueue.readiness,[{status:'loading'}]);assert.equal(pendingQueue.audioEffects.plays,0);
  releaseQueue();for(let count=0;count<8;count++)await new Promise(resolve=>setImmediate(resolve));
  assert.equal(pendingQueue.readiness.at(-1).status,'ready');assert.equal(pendingQueue.writes.length,0);
  const failedQueue=await screen(queue,'?queue=references',{}, {},{queueError:'목록 연결 실패'});
  assert.equal(failedQueue.readiness.at(-1).status,'error');assert.equal(failedQueue.readiness.at(-1).message,failedQueue.nodes.get('saved').textContent);
  assert.equal(failedQueue.audioEffects.plays,0);assert.equal(failedQueue.writes.length,0);
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
  assert.equal(pilot.nodes.get("review").hidden, true);
  assert.doesNotMatch(pilot.nodes.get("review-instruction").textContent, /다시 확인/);
  const content = await screen({ ...queue, policy_review: true, policy_text: "묶음 사용 동의" }, "?queue=content");
  assert.equal(content.nodes.get("policy").hidden, false);
  assert.equal(content.nodes.get("policy-text").textContent, "묶음 사용 동의");
  assert.match(content.nodes.get("criteria").innerHTML, /안내에 맞고/);
  assert.equal(content.nodes.get("review").hidden, false);
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
  const skipped=await screen({...queue,kind:'pilot'},'?queue=pilot');
  skipped.nodes.get('audio').played={length:1,start:()=>.5,end:()=>1};
  await skipped.nodes.get('audio').play();skipped.nodes.get('audio').ended=true;skipped.nodes.get('audio').onended();
  assert.equal(skipped.writes.length,0,'a completed event without the full played range never marks a clip heard');
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
  many.nodes.get('discard').onclick();
  assert.match(many.nodes.get('review-progress').textContent,/^판정한 음원 1 \/ 7개 · 미판정 6개$/);
  assert.equal(many.nodes.get('listening-list').children[1].children[1].textContent,'제외');
  const shared = {sharedControls:true};
  const fullQueue = {...queue,items:Array.from({length:7},(_,i)=>({id:`voice-${i}`,text:'보라야',audio:`/audio/${i}`}))};
  const compact=await screen(fullQueue,'?queue=pilot',{'voice-0':{decision:'keep',listened:true}}, {}, shared);
  const picker=compact.pickers[0];
  assert.ok(compact.classes.has('listen-voice-compact'));
  assert.equal(compact.nodes.get('review-sidebar').hidden,true);
  assert.equal(compact.nodes.get('review-picker').hidden,false);
  assert.equal(compact.nodes.get('review-volume').hidden,false);
  assert.equal(compact.enhanced[0].options.volume,compact.nodes.get('review-volume'));
  assert.equal(compact.enhanced[0].options.previous,undefined);
  assert.equal(compact.enhanced[0].options.next,undefined);
  assert.equal(picker.items.length,7,'all pronunciation samples are reachable in the dropdown');
  assert.equal(picker.items[0].badge,'정상');
  assert.equal(picker.selected,0);
  picker.options.onSelect(6);
  assert.equal(picker.selected,6);
  assert.equal(compact.nodes.get('title').textContent,'7 / 7');
  assert.equal(compact.nodes.get('audio').src,'/audio/6');
  assert.equal(compact.audioEffects.plays,0);
  assert.equal(compact.writes.length,0,'selecting a recording never supplies listening evidence');
  compact.nodes.get('discard').onclick();
  assert.equal(picker.items[6].badge,'제외');
  assert.equal(compact.nodes.get('discard').attributes['aria-pressed'],'true');
  picker.options.onSelect(5);
  await compact.nodes.get('start').onclick();
  compact.nodes.get('audio').ended=true;
  compact.nodes.get('audio').onended();
  assert.equal(picker.items[5].badge,'정상');
  assert.equal(picker.selected,6,'continuous playback updates the dropdown selection');
  compact.nodes.get('audio').ended=true;
  compact.nodes.get('audio').onended();
  assert.equal(picker.items[6].badge,'제외','finishing a clip preserves its explicit exclusion');
  assert.equal(compact.calls.length,1,'continuous playback does not submit the review');
  compact.listeners.message({origin:'http://127.0.0.1:8794',source:compact.parent,data:{type:'koww:pause'}});
  assert.equal(picker.closed,true,'leaving the review closes its dropdown');
  const byKind=await screen({...fullQueue,kind:'pilot'},'',{}, {},shared);
  assert.equal(byKind.pickers.length,1);
  const similarOnly=await screen(fullQueue,'?queue=content',{}, {},shared);
  assert.equal(similarOnly.pickers.length,0,'matching the wakeword alone does not change another review layout');
  assert.equal(similarOnly.enhanced[0].options.previous,similarOnly.nodes.get('prev'));
  const hardQueue={
    ...queue,
    policy_review:true,
    policy_text:'문구별 대표 음원을 듣고 학습에 사용하는 데 동의합니다.',
    items:[
      {...queue.items[0],text:'보라색'},
      {...queue.items[1],text:'보랏빛'},
    ],
  };
  const hard=await screen(hardQueue,'?queue=hard',{}, {},shared);
  const hardPicker=hard.pickers[0];
  assert.ok(hard.classes.has('listen-voice-compact'));
  assert.equal(hardPicker.options.list.attributes['aria-label'],'확인할 유사어 음원');
  assert.deepEqual(Array.from(hardPicker.items,item=>item.label),['보라색','보랏빛']);
  assert.equal(hardPicker.items[0].fullLabel,'유사어 음원 1 · 보라색');
  assert.equal(hard.nodes.get('policy').hidden,false);
  assert.equal(hard.nodes.get('review').hidden,true);
  assert.equal(hard.nodes.get('policy-text').textContent,hardQueue.policy_text);
  hardPicker.options.onSelect(1);
  assert.equal(hard.nodes.get('title').textContent,'2 / 2');
  assert.equal(hard.nodes.get('audio').src,'/audio/1');
  assert.equal(hard.writes.length,0,'choosing a phrase does not mark it as heard');
  hardPicker.options.onSelect(0);
  await hard.nodes.get('start').onclick();
  hard.nodes.get('audio').ended=true;
  hard.nodes.get('audio').onended();
  assert.equal(hardPicker.items[0].badge,'정상');
  assert.equal(hardPicker.selected,1);
  hard.nodes.get('audio').ended=true;
  hard.nodes.get('audio').onended();
  assert.equal(hardPicker.items[1].badge,'정상');
  await hard.nodes.get('save').onclick();
  assert.equal(hard.calls.length,1,'the compact hard review still requires explicit consent');
  hard.nodes.get('policy-approved').checked=true;
  await hard.nodes.get('save').onclick();
  assert.deepEqual(JSON.parse(hard.calls[1].options.body),{
    decisions:heard,
    policy_approved:true,
  });
  const hardByKind=await screen({...hardQueue,kind:'hard'},'',{}, {},shared);
  assert.equal(hardByKind.pickers.length,1);
  const policyOnly=await screen(hardQueue,'?queue=content',{}, {},shared);
  assert.equal(policyOnly.pickers.length,0);
  assert.equal(policyOnly.nodes.get('policy').hidden,false);
  const recheck=await screen(queue,'?queue=pilot',{...heard,two:{decision:'review',listened:true}},config);
  assert.match(recheck.nodes.get('review-progress').textContent,/미판정 1개 · 판정 보류 1개/);
  await recheck.nodes.get('save').onclick();
  assert.equal(recheck.calls.length,1,'a legacy hold requires an explicit decision before saving');
  recheck.nodes.get('next').onclick();
  assert.match(recheck.nodes.get('state').textContent,/정상 또는 제외/);
  recheck.nodes.get('keep').onclick();
  await recheck.nodes.get('save').onclick();
  assert.deepEqual(JSON.parse(recheck.calls[1].options.body),heard);
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
  const preflightQueue=()=>({...queue,kind:'pilot',review_phase:'pre',autosave_policy:'play_started_default_keep_v1',draft_revision:0,draft_decisions:{},
    items:queue.items.map((row,i)=>({...row,audio_sha256:String(i+1).repeat(64),label:`${i?'공통 평가용 · 여성 · 노년':'학습용 · 남성 · 어린이'} · 1번`}))});
  const preConfig={...config,queue_url:'/review/queue?project=bora&queue=pilot',save_url:'/review/save?project=bora&queue=pilot'};
  function draftServer(state,actions){return (url,options)=>{
    if(!options)return {payload:state};
    const body=JSON.parse(options.body);actions.push(body);assert.equal(body.queue_sha256,state.queue_sha256);assert.equal(body.revision,state.draft_revision);
    if(body.action==='finalize')return {payload:{review_finalized:true,workflow_action:'resume',state:'input_submitted'}};
    assert.equal(body.action,'autosave');assert.equal(body.audio_sha256,state.items.find(row=>row.id===body.sample_id).audio_sha256);
    if(body.decision==='clear')delete state.draft_decisions[body.sample_id];
    else{if(body.decision==='keep')assert.equal(body.playback_started,true);state.draft_decisions[body.sample_id]={decision:body.decision,
      playback_started:body.playback_started,playback_completed:body.playback_completed,listened:body.playback_completed};}
    state.draft_revision++;return {payload:{...state,state:'draft_saved'}};
  };}
  const preMessage=(view,type,overrides={})=>view.listeners.message({origin:'http://127.0.0.1:8794',source:view.parent,
    data:{type,project:'bora',queue_sha256:'fixed-queue',request:7,...overrides}});
  const finishPreflight=(view,timeStamp=1200)=>{const audio=view.nodes.get('audio');audio.currentTime=1;audio.paused=true;audio.ended=true;audio.onended({timeStamp});};
  const preState=preflightQueue(),preActions=[],continuousPre=await screen(preState,'?project=bora&queue=pilot',heard,preConfig,{...shared,respond:draftServer(preState,preActions)});
  preMessage(continuousPre,'koww:preflight-status');
  assert.equal(continuousPre.nodes.get('keep').hidden,true);assert.equal(continuousPre.nodes.get('discard').textContent,'부적합');assert.equal(continuousPre.nodes.get('save').hidden,true);
  assert.equal(continuousPre.writes.length,0);assert.deepEqual(Array.from(continuousPre.pickers[0].items,row=>row.label),preState.items.map(row=>row.label));
  assert.equal(continuousPre.pickers[0].items[0].badge,'미청취');assert.equal(continuousPre.pickers[0].items[1].badge,'미청취');
  await continuousPre.nodes.get('start').onclick();await continuousPre.flush();
  assert.equal(preActions.length,1);assert.equal(preActions[0].decision,'keep');assert.equal(preActions[0].playback_started,true);assert.equal(preActions[0].playback_completed,false);
  assert.equal(preState.draft_decisions.one.listened,false);assert.equal(continuousPre.pickers[0].items[0].badge,'적합');assert.equal(continuousPre.pickers[0].items[1].badge,'미청취');
  const sourceBefore=continuousPre.nodes.get('audio').src,pausesBeforeChoice=continuousPre.audioEffects.pauses,playsBeforeChoice=continuousPre.audioEffects.plays;
  continuousPre.nodes.get('audio').currentTime=.5;continuousPre.nodes.get('discard').onpointerdown();continuousPre.nodes.get('discard').onclick({detail:1});await continuousPre.flush();
  assert.equal(preState.draft_decisions.one.decision,'discard');assert.equal(continuousPre.nodes.get('audio').src,sourceBefore);assert.equal(continuousPre.nodes.get('audio').currentTime,.5);
  assert.equal(continuousPre.audioEffects.pauses,pausesBeforeChoice);assert.equal(continuousPre.audioEffects.plays,playsBeforeChoice,'unsuitable toggle never stops or advances audio');
  continuousPre.nodes.get('discard').onpointerdown();continuousPre.nodes.get('discard').onclick({detail:1});await continuousPre.flush();assert.equal(preState.draft_decisions.one.decision,'keep');
  finishPreflight(continuousPre);await continuousPre.flush();
  assert.equal(continuousPre.nodes.get('audio').src,'/audio/1');assert.equal(continuousPre.audioEffects.plays,2);assert.equal(preState.draft_decisions.one.playback_completed,true);
  assert.equal(preState.draft_decisions.two.playback_started,true);assert.equal(preState.draft_decisions.two.playback_completed,false);
  finishPreflight(continuousPre,1400);await continuousPre.flush();assert.equal(continuousPre.nodes.get('start').textContent,'재생');assert.equal(continuousPre.audioEffects.plays,2);
  assert.equal(continuousPre.nodes.get('audio').src,'/audio/1');assert.equal(preActions.filter(row=>row.action==='finalize').length,0,'last ended stops without wrapping or finalizing');
  assert.equal(continuousPre.parentMessages.filter(row=>row.message.type==='koww:preflight-state').at(-1).message.ready,true);
  for(const invalid of [{project:'other'},{queue_sha256:'old'},{request:6}])preMessage(continuousPre,'koww:preflight-finalize',invalid);
  await continuousPre.flush();assert.equal(preActions.filter(row=>row.action==='finalize').length,0);preMessage(continuousPre,'koww:preflight-finalize');await continuousPre.flush();
  assert.equal(preActions.filter(row=>row.action==='finalize').length,1);preMessage(continuousPre,'koww:preflight-finalize');await continuousPre.flush();assert.equal(preActions.filter(row=>row.action==='finalize').length,1);
  const unheardState=preflightQueue(),unheardActions=[],unheard=await screen(unheardState,'?project=bora&queue=pilot',{},preConfig,{...shared,respond:draftServer(unheardState,unheardActions)});
  unheard.nodes.get('discard').onpointerdown();unheard.nodes.get('discard').onclick({detail:1});await unheard.flush();assert.equal(unheardActions[0].decision,'discard');assert.equal(unheardActions[0].playback_started,false);
  assert.equal(unheard.nodes.get('audio').src,'/audio/0');assert.equal(unheard.audioEffects.plays,0);unheard.nodes.get('discard').onclick({detail:1});await unheard.flush();
  assert.equal(unheardActions.at(-1).decision,'clear');assert.equal(unheardState.draft_decisions.one,undefined);assert.equal(unheard.pickers[0].items[0].badge,'미청취');
  unheard.nodes.get('discard').onclick({detail:1});await unheard.flush();await unheard.nodes.get('start').onclick();await unheard.flush();assert.equal(unheardState.draft_decisions.one.decision,'discard','actual playing respects an existing unsuitable disposition');
  const raceState=preflightQueue(),raceActions=[],race=await screen(raceState,'?project=bora&queue=pilot',{},preConfig,{...shared,respond:draftServer(raceState,raceActions)});
  await race.nodes.get('start').onclick();await race.flush();race.nodes.get('discard').onpointerdown();finishPreflight(race);await race.flush();
  const raceCount=raceActions.length;race.nodes.get('discard').onclick({detail:1});await race.flush();assert.equal(raceActions.length,raceCount,'pointerdown on the finished row cannot reject the next row');
  assert.equal(raceState.draft_decisions.two.decision,'keep');race.nodes.get('discard').onpointerdown();race.nodes.get('discard').onclick({detail:2});await race.flush();assert.equal(raceActions.length,raceCount,'native double-click cannot toggle the next row');
  let prevented=false;race.nodes.get('discard').onkeydown({key:'Enter',repeat:true,preventDefault(){prevented=true;}});assert.equal(prevented,true);
  race.nodes.get('pause').onclick();race.pickers[0].options.onSelect(0);await race.nodes.get('start').onclick();await race.flush();
  race.nodes.get('discard').onkeydown({key:' ',repeat:false,preventDefault(){}});finishPreflight(race,1400);await race.flush();const keyboardCount=raceActions.length;
  race.nodes.get('discard').onclick({detail:0});await race.flush();assert.equal(raceActions.length,keyboardCount,'a held keyboard gesture remains bound to its original row');
  let releaseDraft;const delayedState=preflightQueue(),delayedActions=[],applyDelayed=draftServer(delayedState,delayedActions);
  const delayed=await screen(delayedState,'?project=bora&queue=pilot',{},preConfig,{...shared,respond:(url,options)=>options&&delayedActions.length===0?
    new Promise(resolve=>{releaseDraft=()=>resolve(applyDelayed(url,options));}):applyDelayed(url,options)});
  preMessage(delayed,'koww:preflight-status');await delayed.nodes.get('start').onclick();finishPreflight(delayed);await delayed.flush();
  assert.equal(delayed.audioEffects.plays,2,'ended starts the next clip before the previous draft ACK');assert.equal(delayedActions.length,0);assert.equal(delayed.pickers[0].items[0].badge,'저장 중');
  finishPreflight(delayed,1400);preMessage(delayed,'koww:preflight-finalize');await delayed.flush();assert.equal(delayedActions.length,0,'Next waits for serialized pending drafts');
  releaseDraft();await delayed.flush();assert.deepEqual(delayedActions.map(row=>row.action),['autosave','autosave','autosave','finalize']);
  assert.equal(delayedState.draft_decisions.one.playback_completed,true);assert.equal(delayedState.draft_decisions.two.playback_completed,true);
  assert.equal(delayed.pickers[0].items[0].badge,'적합');assert.equal(delayed.pickers[0].items[1].badge,'적합');
  const pausedState=preflightQueue(),pausedActions=[],pausedPre=await screen(pausedState,'?project=bora&queue=pilot',{},preConfig,{...shared,respond:draftServer(pausedState,pausedActions)});
  await pausedPre.nodes.get('start').onclick();await pausedPre.flush();pausedPre.nodes.get('audio').currentTime=.4;pausedPre.nodes.get('start').onclick();
  assert.equal(pausedPre.nodes.get('audio').paused,true);const pausedPlays=pausedPre.audioEffects.plays;finishPreflight(pausedPre);await pausedPre.flush();assert.equal(pausedPre.audioEffects.plays,pausedPlays);
  await pausedPre.nodes.get('start').onclick();await pausedPre.flush();pausedPre.document.hidden=true;pausedPre.documentEvents.visibilitychange();finishPreflight(pausedPre);await pausedPre.flush();
  assert.equal(pausedPre.nodes.get('audio').src,undefined);assert.equal(pausedState.draft_decisions.two,undefined);const hiddenPlays=pausedPre.audioEffects.plays;
  await pausedPre.nodes.get('start').onclick();assert.equal(pausedPre.audioEffects.plays,hiddenPlays);pausedPre.document.hidden=false;
  pausedPre.listeners.message({origin:'http://127.0.0.1:8794',source:pausedPre.parent,data:{type:'koww:pause'}});assert.equal(pausedPre.nodes.get('audio').paused,true);
  let rejectPreflightOld;const lateState=preflightQueue(),lateActions=[],late=await screen(lateState,'?project=bora&queue=pilot',{},preConfig,{...shared,
    playback:count=>count===1?new Promise((_,reject)=>{rejectPreflightOld=reject;}):Promise.resolve(),respond:draftServer(lateState,lateActions)});
  const oldPlay=late.nodes.get('start').onclick();late.nodes.get('audio').onplaying({timeStamp:1000});late.pickers[0].options.onSelect(1);await late.flush();rejectPreflightOld(Error('old play failed'));await oldPlay;await late.flush();
  assert.equal(late.nodes.get('audio').src,'/audio/1');assert.equal(late.nodes.get('audio').paused,false);const lateCount=lateActions.length;
  late.nodes.get('audio').onpause();late.nodes.get('audio').currentSrc='/audio/0';late.nodes.get('audio').onplaying({timeStamp:0});finishPreflight(late,0);await late.flush();
  late.nodes.get('audio').currentSrc='/audio/1';late.nodes.get('audio').onplaying({timeStamp:0});finishPreflight(late,0);await late.flush();
  assert.equal(lateActions.length,lateCount);assert.equal(lateState.draft_decisions.two.playback_completed,false,'old source/timestamp completion cannot contaminate the next row');
  let failSent;const failedState=preflightQueue(),failedActions=[],applyFailed=draftServer(failedState,failedActions);let failOnce=true;
  const failedSave=await screen(failedState,'?project=bora&queue=pilot',{},preConfig,{...shared,respond:(url,options)=>{
    if(options&&failOnce){failOnce=false;return new Promise((_,reject)=>{failSent=()=>reject(Error('save failed'));});}return applyFailed(url,options);
  }});
  await failedSave.nodes.get('start').onclick();finishPreflight(failedSave);await failedSave.flush();assert.equal(failedSave.audioEffects.plays,2);failSent();await failedSave.flush();
  assert.equal(failedSave.nodes.get('audio').paused,true);assert.equal(failedSave.nodes.get('start').disabled,true);assert.equal(failedSave.nodes.get('preflight-retry').hidden,false);
  assert.notEqual(failedSave.pickers[0].items[0].badge,'적합');const blockedPlays=failedSave.audioEffects.plays;await failedSave.nodes.get('start').onclick();assert.equal(failedSave.audioEffects.plays,blockedPlays);
  await failedSave.nodes.get('audio').play();assert.equal(failedSave.nodes.get('audio').paused,true,'native play is stopped while save recovery is blocked');
  await failedSave.nodes.get('preflight-retry').onclick();await failedSave.flush();assert.equal(failedSave.nodes.get('start').disabled,false);assert.equal(failedSave.nodes.get('audio').paused,true,'retry never restarts audio on its own');
  assert.equal(failedState.draft_decisions.one.playback_completed,true);assert.equal(failedState.draft_decisions.two.playback_started,true);assert.equal(failedState.draft_decisions.two.playback_completed,false);
  let releaseFailedNext;const failedNextState=preflightQueue(),failedNextActions=[],applyFailedNext=draftServer(failedNextState,failedNextActions);
  const failedNext=await screen(failedNextState,'?project=bora&queue=pilot',{},preConfig,{...shared,playback:count=>count===2?Promise.reject(Error('next failed')):Promise.resolve(),
    respond:(url,options)=>options&&failedNextActions.length===0?new Promise(resolve=>{releaseFailedNext=()=>resolve(applyFailedNext(url,options));}):applyFailedNext(url,options)});
  await failedNext.nodes.get('start').onclick();finishPreflight(failedNext);await failedNext.flush();assert.match(failedNext.nodes.get('state').textContent,/재생하지 못했습니다/);
  releaseFailedNext();await failedNext.flush();assert.match(failedNext.nodes.get('state').textContent,/재생하지 못했습니다/);assert.equal(failedNext.nodes.get('discard').disabled,false);
  failedNext.nodes.get('discard').onclick({detail:1});await failedNext.flush();assert.equal(failedNextState.draft_decisions.two.decision,'discard');assert.equal(failedNextState.draft_decisions.two.playback_started,false);
  let releaseInitial;const initialGate=new Promise(resolve=>{releaseInitial=resolve;});const handshake=await screen(preflightQueue(),'?project=bora&queue=pilot',{},preConfig,{...shared,queueGate:initialGate});
  preMessage(handshake,'koww:preflight-status',{request:19});releaseInitial();await handshake.flush();assert.equal(handshake.parentMessages.filter(row=>row.message.type==='koww:preflight-state').at(-1).message.request,19);
  const notPlaying=await screen(preflightQueue(),'?project=bora&queue=pilot',{},preConfig,{...shared,actualPlaying:false});await notPlaying.nodes.get('start').onclick();await notPlaying.flush();assert.equal(notPlaying.calls.length,1);
  const failedPre=await screen(preflightQueue(),'?project=bora&queue=pilot',{},preConfig,{...shared,playback:()=>Promise.reject(Error('device failure'))});await failedPre.nodes.get('start').onclick();await failedPre.flush();assert.equal(failedPre.calls.length,1);
  let staleReads=0;const staleState=preflightQueue(),stale=await screen(staleState,'?project=bora&queue=pilot',{},preConfig,{...shared,respond:(_url,options)=>{
    if(options)throw Error('lost reply');return {payload:++staleReads===1?staleState:{...staleState,queue_sha256:'different-queue'}};
  }});await stale.nodes.get('start').onclick();await stale.flush();assert.notEqual(stale.pickers[0].items[0].badge,'적합');assert.equal(stale.nodes.get('preflight-retry').hidden,false);assert.equal(stale.calls.filter(row=>row.options).length,1);
  assert.match(html,/html\.preflight-review,html\.preflight-review body\{height:auto;overflow:visible\}/);
  let overlayBase=170,overlayView;const overlayState=preflightQueue(),overlayActions=[];
  overlayView=await screen(overlayState,'?project=bora&queue=pilot',{},preConfig,{...shared,mainHeight:()=>overlayView?.classes.has('preflight-picker-open')?405:overlayBase,
    respond:draftServer(overlayState,overlayActions)});
  preMessage(overlayView,'koww:preflight-status');const overlayPicker=overlayView.pickers[0];
  const beforeOverlay={source:overlayView.nodes.get('audio').src,plays:overlayView.audioEffects.plays,pauses:overlayView.audioEffects.pauses,index:overlayPicker.selected,calls:overlayView.calls.length};
  overlayPicker.show(3);assert.equal(overlayView.classes.has('preflight-picker-open'),true);
  const openMessage=overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-picker').at(-1).message;
  assert.deepEqual(openMessage,{type:'koww:preflight-picker',project:'bora',queue_sha256:'fixed-queue',request:7,picker_request:3,open:true});assert.equal(overlayPicker.revealed.length,0);
  const heightCount=overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-height').length;
  for(let i=0;i<4;i++)overlayView.resizeObservers.forEach(observer=>observer.callback());await overlayView.flush();
  assert.equal(overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-height').length,heightCount,'expanded main height cannot feed back into the reserved natural body height');
  for(const invalid of [{project:'other'},{queue_sha256:'stale'},{request:6},{picker_request:2}])preMessage(overlayView,'koww:preflight-picker-ready',{picker_request:3,...invalid});
  assert.equal(overlayPicker.revealed.length,0);preMessage(overlayView,'koww:preflight-picker-ready',{picker_request:3});assert.deepEqual(overlayPicker.revealed,[3]);
  assert.deepEqual({source:overlayView.nodes.get('audio').src,plays:overlayView.audioEffects.plays,pauses:overlayView.audioEffects.pauses,index:overlayPicker.selected,calls:overlayView.calls.length},beforeOverlay,'opening the list leaves playback, selection and backend requests unchanged');
  overlayBase=190;overlayPicker.close();assert.equal(overlayView.classes.has('preflight-picker-open'),true,
    'logical close hides the list but keeps the centered surface until the parent has restored the iframe');
  assert.equal(overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-picker').at(-1).message.open,false);
  const closingHeightCount=overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-height').length;
  for(let i=0;i<4;i++)overlayView.resizeObservers.forEach(observer=>observer.callback());await overlayView.flush();
  assert.equal(overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-height').length,closingHeightCount,
    'the expanded closing surface cannot leak its temporary height into the natural baseline');
  for(const invalid of [{project:'other'},{queue_sha256:'stale'},{request:6},{picker_request:2}]){
    preMessage(overlayView,'koww:preflight-picker-closed',{picker_request:3,...invalid});
    assert.equal(overlayView.classes.has('preflight-picker-open'),true);
  }
  overlayView.listeners.message({origin:'https://other.invalid',source:overlayView.parent,data:{type:'koww:preflight-picker-closed',project:'bora',queue_sha256:'fixed-queue',request:7,picker_request:3}});
  assert.equal(overlayView.classes.has('preflight-picker-open'),true);
  preMessage(overlayView,'koww:preflight-picker-closed',{picker_request:3});assert.equal(overlayView.classes.has('preflight-picker-open'),false);
  assert.equal(overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-height').at(-1).message.height,190,'the applied-close acknowledgement restores actual natural content height');
  preMessage(overlayView,'koww:preflight-picker-ready',{picker_request:3});assert.deepEqual(overlayPicker.revealed,[3],'a late ready cannot reopen a closed popup');
  await overlayView.nodes.get('start').onclick();await overlayView.flush();const outsidePauses=overlayView.audioEffects.pauses;
  overlayPicker.show(4);overlayPicker.close();assert.equal(overlayView.audioEffects.pauses,outsidePauses,'outside close never pauses audio');
  overlayPicker.show(5);preMessage(overlayView,'koww:preflight-picker-closed',{picker_request:4});
  assert.equal(overlayView.classes.has('preflight-picker-open'),true,'reopening supersedes an in-flight close acknowledgement');
  overlayPicker.close();preMessage(overlayView,'koww:preflight-picker-closed',{picker_request:5});assert.equal(overlayView.classes.has('preflight-picker-open'),false);
  assert.equal(overlayView.parentMessages.filter(row=>row.message.type==='koww:preflight-height').at(-1).message.height,190,'reopening during a close preserves the natural height instead of measuring the expanded surface');
  overlayPicker.show(6);overlayView.listeners.message({origin:'http://127.0.0.1:8794',source:overlayView.parent,data:{type:'koww:pause'}});
  assert.equal(overlayView.classes.has('preflight-picker-open'),true);preMessage(overlayView,'koww:preflight-picker-closed',{picker_request:6});
  assert.equal(overlayView.classes.has('preflight-picker-open'),false);preMessage(overlayView,'koww:preflight-picker-ready',{picker_request:6});assert.deepEqual(overlayPicker.revealed,[3]);
  overlayPicker.show(7);overlayPicker.close();preMessage(overlayView,'koww:preflight-status',{request:8});
  assert.equal(overlayView.classes.has('preflight-picker-open'),false,'a new parent frame epoch resets a superseded close after the parent has restored its baseline');
  preMessage(overlayView,'koww:preflight-picker-closed',{picker_request:7});assert.equal(overlayView.classes.has('preflight-picker-open'),false);
  assert.equal(hard.pickers[0].options.onOpenChange,undefined,'hard list opening keeps the existing immediate behavior');
  assert.match(html,/html\.preflight-review\.preflight-picker-open main\{height:100dvh;min-height:0;justify-content:center\}/);
  assert.match(html,/\.preflight-review main\{height:auto;min-height:0;overflow:visible\}/);
  // An expanded iframe is a temporary popup surface, not a taller player.
  // The legacy top-aligned compact selector has greater specificity than the
  // shared call-review centering selector, so it must exclude both phases.
  assert.match(html,/html\.listening-embedded\.compact-review:not\(\.preflight-review\):not\(\.compact-call-review\) main\{justify-content:flex-start\}/);
  assert.doesNotMatch(html,/html\.listening-embedded\.compact-review:not\(\.preflight-review\) main\{justify-content:flex-start\}/);
  assert.match(html,/html\.compact-call-review\.preflight-picker-open main\{height:100dvh;min-height:0;justify-content:center\}/);
  assert.match(html,/html\.listening-embedded\.compact-call-review main\{gap:10px\}/,
    'the main gap remains the same before and after crossing the iframe-height media breakpoint');
  assert.match(html,/html\.listening-embedded\.compact-call-review :is\(\.review-footer,\.review-save\)\{gap:6px\}/,
    'error/retry footer spacing must also remain independent of the popup surface height');
  // Post-generation review shares the compact bridge without weakening the
  // existing completed-listening admission or using the preflight draft API.
  const postQueue=()=>({...queue,kind:'positive',review_phase:'post',items:queue.items.map((row,i)=>({...row,
    group:i?'공통 평가용':'학습용',voice_number:1}))});
  const postConfig={...config,queue_url:'/review/queue?project=bora&queue=positive',save_url:'/review/save?project=bora&queue=positive'};
  const postState=(view)=>view.parentMessages.filter(row=>row.message.type==='koww:preflight-state').at(-1).message;
  const postPosts=(view)=>view.calls.filter(row=>row.options?.method==='POST');
  const postResponse=(state,handler)=>async(url,options)=>options?handler(JSON.parse(options.body)):{payload:state};
  const completePost=async(view)=>{await view.nodes.get('start').onclick();finishPreflight(view,1400);await view.flush();finishPreflight(view,1600);await view.flush();};
  const postData=postQueue(),post=await screen(postData,'?project=bora&queue=positive',{},postConfig,{...shared,
    respond:postResponse(postData,body=>({payload:{state:'saved',policy_approved:true}}))});
  const labelledPostData=postQueue();
  labelledPostData.items[0].label='학습용 · 남성 · 어린이 · 1번';
  labelledPostData.items[1].label='공통 평가용 · 여성 · 노년 · 1번';
  const labelledPost=await screen(labelledPostData,'?project=bora&queue=positive',{},postConfig,shared);
  assert.deepEqual(Array.from(labelledPost.pickers[0].items,row=>row.label),labelledPostData.items.map(row=>row.label));
  assert.equal(labelledPost.nodes.get('items').children[1].textContent,labelledPostData.items[1].label);
  assert.equal(labelledPost.nodes.get('audio').src,labelledPostData.items[0].audio,'display metadata never chooses another audio');
  assert.equal(postPosts(labelledPost).length,0,'showing detailed labels neither judges nor saves a voice');
  preMessage(post,'koww:preflight-status');
  assert.equal(post.classes.has('compact-call-review'),true);assert.equal(post.classes.has('preflight-review'),false);
  assert.equal(post.nodes.get('keep').hidden,true);assert.equal(post.nodes.get('discard').textContent,'부적합');assert.equal(post.nodes.get('save').hidden,true);
  assert.equal(post.nodes.get('state').hidden,true);assert.equal(post.nodes.get('review-heading').textContent,'호출어 발음 검수');
  assert.deepEqual(Array.from(post.pickers[0].items,row=>row.label),['학습용 · 1','공통 평가용 · 1']);
  assert.equal(postState(post).ready,false);assert.equal(postPosts(post).length,0);
  await post.nodes.get('start').onclick();
  assert.equal(post.writes.length,0,'post playback start does not grant completed listening or an automatic keep');
  assert.equal(postState(post).ready,false);
  const postSource=post.nodes.get('audio').src,postPauses=post.audioEffects.pauses;
  post.nodes.get('discard').onpointerdown();post.nodes.get('discard').onclick({detail:1});
  assert.deepEqual(JSON.parse(post.writes.at(-1)[1]),{one:{decision:'discard',listened:false}});
  assert.equal(post.nodes.get('audio').src,postSource);assert.equal(post.audioEffects.pauses,postPauses);
  post.nodes.get('discard').onpointerdown();post.nodes.get('discard').onclick({detail:1});
  assert.deepEqual(JSON.parse(post.writes.at(-1)[1]),{},'unheard unsuitable can be cleared without inventing a keep');
  finishPreflight(post,1400);await post.flush();
  assert.deepEqual(JSON.parse(post.writes.at(-1)[1]),{one:{decision:'keep',listened:true}});
  assert.equal(post.nodes.get('audio').src,'/audio/1');assert.equal(post.audioEffects.plays,2);
  finishPreflight(post,1600);await post.flush();assert.equal(postState(post).ready,true);
  assert.equal(postPosts(post).length,0,'finishing all post clips never submits the corpus on its own');
  post.pickers[0].show(11);
  assert.equal(post.parentMessages.filter(row=>row.message.type==='koww:preflight-picker').at(-1).message.open,true);
  preMessage(post,'koww:preflight-picker-ready',{picker_request:10});assert.deepEqual(post.pickers[0].revealed,[]);
  preMessage(post,'koww:preflight-picker-ready',{picker_request:11});assert.deepEqual(post.pickers[0].revealed,[11]);post.pickers[0].close();
  assert.equal(post.classes.has('preflight-picker-open'),true);preMessage(post,'koww:preflight-picker-closed',{picker_request:11});assert.equal(post.classes.has('preflight-picker-open'),false);
  for(const invalid of [{project:'other'},{queue_sha256:'old'},{request:6}])preMessage(post,'koww:preflight-finalize',invalid);
  for(const invalid of [{origin:'https://elsewhere.example',source:post.parent},{origin:'http://127.0.0.1:8794',source:{}}])post.listeners.message({...invalid,data:{type:'koww:preflight-finalize',project:'bora',queue_sha256:'fixed-queue',request:7}});
  await post.flush();assert.equal(postPosts(post).length,0);
  preMessage(post,'koww:preflight-finalize');preMessage(post,'koww:preflight-finalize');await post.flush();
  assert.equal(postPosts(post).length,1);assert.deepEqual(JSON.parse(postPosts(post)[0].options.body),{decisions:heard,queue_sha256:'fixed-queue'});
  assert.equal(postPosts(post)[0].options.headers['X-Koww-Token'],postConfig.token);
  assert.equal(post.parentMessages.filter(row=>row.message.type==='koww:preflight-finalized').length,1);
  assert.equal(post.parentMessages.filter(row=>row.message.type==='koww:preflight-finalized').at(-1).message.workflow_action,'resume');
  assert.equal(post.nodes.get('start').disabled,true);assert.equal(post.nodes.get('discard').disabled,true);
  preMessage(post,'koww:preflight-finalize');await post.flush();assert.equal(postPosts(post).length,1);
  const heldData=postQueue(),heldPost=await screen(heldData,'?project=bora&queue=positive',{...heard,two:{decision:'discard',listened:true}},postConfig,{...shared,
    respond:postResponse(heldData,body=>({payload:{state:body.decisions.two.decision==='discard'?'pilot_exceptions_need_policy_fix':'saved',policy_approved:body.decisions.two.decision==='keep',kept:body.decisions.two.decision==='keep'?2:1,total:2}}))});
  preMessage(heldPost,'koww:preflight-status');assert.equal(postState(heldPost).ready,true);
  preMessage(heldPost,'koww:preflight-finalize');await heldPost.flush();
  assert.equal(postState(heldPost).error,false);assert.equal(postState(heldPost).ready,true);assert.equal(heldPost.nodes.get('preflight-retry').hidden,true);
  assert.equal(heldPost.nodes.get('discard').disabled,false);assert.equal(heldPost.nodes.get('discard').attributes['aria-pressed'],'true');
  assert.equal(heldPost.nodes.get('audio').src,heldData.items[1].audio);assert.equal(heldPost.pickers[0].selected,1);
  assert.match(heldPost.nodes.get('state').textContent,/부적합 1개.*판정을 확인/);assert.equal(heldPost.nodes.get('saved').textContent,'');
  assert.equal(heldPost.parentMessages.filter(row=>row.message.type==='koww:preflight-held').length,1);
  assert.equal(heldPost.writes.length,0,'focusing a held voice never changes a human disposition');assert.equal(heldPost.audioEffects.plays,0);
  assert.equal(heldPost.parentMessages.some(row=>row.message.type==='koww:preflight-finalized'),false);
  heldPost.pickers[0].options.onSelect(1);heldPost.nodes.get('discard').onpointerdown();heldPost.nodes.get('discard').onclick({detail:1});
  assert.deepEqual(JSON.parse(heldPost.writes.at(-1)[1]),heard);assert.equal(postState(heldPost).error,false);assert.equal(postState(heldPost).ready,true);
  preMessage(heldPost,'koww:preflight-finalize');await heldPost.flush();assert.equal(postPosts(heldPost).length,2);assert.equal(heldPost.parentMessages.filter(row=>row.message.type==='koww:preflight-finalized').length,1);
  const fiftyData={...postQueue(),items:Array.from({length:50},(_,i)=>({...queue.items[0],id:`voice-${i+1}`,audio:`/audio/held-${i+1}`,group:i<20?'학습용':'공통 평가용',voice_number:i+1}))};
  const fiftyAnswers=Object.fromEntries(fiftyData.items.map((row,i)=>[row.id,{decision:i===37?'discard':'keep',listened:true}]));
  const fifty=await screen(fiftyData,'?project=bora&queue=positive',fiftyAnswers,postConfig,{...shared,respond:postResponse(fiftyData,body=>{
    const kept=Object.values(body.decisions).filter(answer=>answer.decision==='keep').length;
    return {payload:{state:kept===50?'saved':'pilot_exceptions_need_policy_fix',policy_approved:kept===50,kept,total:50}};
  })});
  preMessage(fifty,'koww:preflight-status');preMessage(fifty,'koww:preflight-finalize');await fifty.flush();
  assert.deepEqual(JSON.parse(postPosts(fifty)[0].options.body).decisions,fiftyAnswers);
  assert.equal(fifty.pickers[0].selected,37);assert.equal(fifty.nodes.get('audio').src,fiftyData.items[37].audio);
  assert.equal(fifty.nodes.get('discard').attributes['aria-pressed'],'true');assert.equal(fifty.writes.length,0);assert.equal(fifty.audioEffects.plays,0);
  assert.equal(postState(fifty).error,false);assert.match(fifty.nodes.get('state').textContent,/부적합 1개/);
  preMessage(fifty,'koww:preflight-finalize');await fifty.flush();assert.equal(postPosts(fifty).length,2);
  assert.equal(fifty.parentMessages.filter(row=>row.message.type==='koww:preflight-held').length,2);assert.equal(fifty.writes.length,0,'repeating Next preserves the saved explicit rejection');
  fifty.nodes.get('discard').onpointerdown();fifty.nodes.get('discard').onclick({detail:1});
  const explicitlyCorrected=JSON.parse(fifty.writes.at(-1)[1]);assert.equal(explicitlyCorrected['voice-38'].decision,'keep');
  assert.ok(Object.values(explicitlyCorrected).every(answer=>answer.listened===true));
  preMessage(fifty,'koww:preflight-finalize');await fifty.flush();assert.equal(postPosts(fifty).length,3);
  assert.equal(fifty.parentMessages.filter(row=>row.message.type==='koww:preflight-finalized').length,1);
  for(const response of [
    {state:'pilot_exceptions_need_policy_fix',policy_approved:false,kept:1,total:2},
    {state:'unknown',policy_approved:false,kept:1,total:2},
    {state:'saved'},
  ]){
    const invalid=await screen(postQueue(),'?project=bora&queue=positive',heard,postConfig,{...shared,respond:postResponse(postQueue(),()=>({payload:response}))});
    preMessage(invalid,'koww:preflight-status');preMessage(invalid,'koww:preflight-finalize');await invalid.flush();
    assert.equal(postState(invalid).error,true);assert.equal(invalid.parentMessages.some(row=>row.message.type==='koww:preflight-held'),false,
      'an unknown response or a false approval without submitted negative decisions is not an expected hold');
  }
  for(const response of [
    {state:'unknown',policy_approved:false,kept:1,total:2},
    {state:'pilot_exceptions_need_policy_fix',policy_approved:false,kept:2,total:2},
    {state:'saved',policy_approved:true,kept:2,total:2},
  ]){
    const invalid=await screen(postQueue(),'?project=bora&queue=positive',{...heard,two:{decision:'discard',listened:true}},postConfig,
      {...shared,respond:postResponse(postQueue(),()=>({payload:response}))});
    preMessage(invalid,'koww:preflight-status');preMessage(invalid,'koww:preflight-finalize');await invalid.flush();
    assert.equal(postState(invalid).error,true);assert.equal(invalid.parentMessages.some(row=>['koww:preflight-held','koww:preflight-finalized'].includes(row.message.type)),false,
      'contradictory server counts, unknown hold states, or approval with an explicit negative are protocol failures');
    assert.equal(invalid.writes.length,0);
  }
  let releaseHeld;const staleHeldData=postQueue(),staleHeld=await screen(staleHeldData,'?project=bora&queue=positive',
    {...heard,two:{decision:'discard',listened:true}},postConfig,{...shared,respond:postResponse(staleHeldData,()=>new Promise(resolve=>{releaseHeld=resolve;}))});
  preMessage(staleHeld,'koww:preflight-status');preMessage(staleHeld,'koww:preflight-finalize');await staleHeld.flush();
  preMessage(staleHeld,'koww:preflight-status',{request:8});releaseHeld({payload:{state:'pilot_exceptions_need_policy_fix',policy_approved:false,kept:1,total:2}});await staleHeld.flush();
  assert.equal(staleHeld.parentMessages.some(row=>row.message.type==='koww:preflight-held'),false);
  assert.equal(staleHeld.pickers[0].selected,0,'an obsolete hold response cannot move the current selection');assert.equal(staleHeld.writes.length,0);
  let failPost=true;const errorData=postQueue(),errorPost=await screen(errorData,'?project=bora&queue=positive',heard,postConfig,{...shared,
    respond:postResponse(errorData,()=>failPost?{ok:false,payload:{error:'일시적인 저장 오류'}}:{payload:{state:'saved',policy_approved:true}})});
  preMessage(errorPost,'koww:preflight-status');preMessage(errorPost,'koww:preflight-finalize');await errorPost.flush();
  assert.equal(postState(errorPost).error,true);assert.match(errorPost.nodes.get('saved').textContent,/일시적인 저장 오류/);
  failPost=false;await errorPost.nodes.get('preflight-retry').onclick();await errorPost.flush();
  assert.equal(postPosts(errorPost).length,1,'retry control restores a fresh Next without posting on its own');assert.equal(postState(errorPost).ready,true);
  preMessage(errorPost,'koww:preflight-finalize');await errorPost.flush();assert.equal(postPosts(errorPost).length,2);
  let releasePost;const latePostData=postQueue(),latePost=await screen(latePostData,'?project=bora&queue=positive',heard,postConfig,{...shared,
    respond:postResponse(latePostData,()=>new Promise(resolve=>{releasePost=resolve;}))});
  preMessage(latePost,'koww:preflight-status');preMessage(latePost,'koww:preflight-finalize');await latePost.flush();
  assert.equal(latePost.nodes.get('discard').disabled,true);latePost.pickers[0].options.onSelect(1);assert.equal(latePost.nodes.get('audio').src,'/audio/0');
  preMessage(latePost,'koww:preflight-status',{request:8});releasePost({payload:{state:'saved',policy_approved:true}});await latePost.flush();
  assert.equal(latePost.parentMessages.some(row=>row.message.type==='koww:preflight-finalized'),false,'a changed parent request cannot receive a late save ACK');
  assert.equal(postState(latePost).request,8);
  const skipPost=await screen(postQueue(),'?project=bora&queue=positive',{},postConfig,shared);preMessage(skipPost,'koww:preflight-status');
  skipPost.nodes.get('audio').played={length:1,start:()=>.5,end:()=>1};await skipPost.nodes.get('start').onclick();finishPreflight(skipPost,1400);await skipPost.flush();
  assert.equal(skipPost.writes.length,0);assert.equal(postState(skipPost).ready,false);assert.equal(skipPost.nodes.get('state').hidden,false);assert.match(skipPost.nodes.get('state').textContent,/건너뛴 구간/);
  const notPlayedPost=await screen(postQueue(),'?project=bora&queue=positive',{},postConfig,{...shared,actualPlaying:false});await notPlayedPost.nodes.get('start').onclick();finishPreflight(notPlayedPost,1400);await notPlayedPost.flush();assert.equal(notPlayedPost.writes.length,0);
  const failedPostAudio=await screen(postQueue(),'?project=bora&queue=positive',{},postConfig,{...shared,playback:()=>Promise.reject(Error('device failed'))});await failedPostAudio.nodes.get('start').onclick();
  assert.equal(failedPostAudio.nodes.get('state').hidden,false);assert.match(failedPostAudio.nodes.get('state').textContent,/재생하지 못했습니다/);assert.equal(failedPostAudio.writes.length,0);
  const gesturePost=await screen(postQueue(),'?project=bora&queue=positive',{},postConfig,shared);await gesturePost.nodes.get('start').onclick();gesturePost.nodes.get('discard').onpointerdown();finishPreflight(gesturePost,1400);await gesturePost.flush();
  const gestureWrites=gesturePost.writes.length;gesturePost.nodes.get('discard').onclick({detail:1});assert.equal(gesturePost.writes.length,gestureWrites,'a gesture for the previous clip cannot reject the next clip');
  gesturePost.nodes.get('discard').onclick({detail:2});assert.equal(gesturePost.writes.length,gestureWrites,'a repeated double click cannot reject the next clip');
  const nativePostData=postQueue(),nativePost=await screen(nativePostData,'?project=bora&queue=positive',{},postConfig,{...shared,embedded:false,
    respond:postResponse(nativePostData,()=>({payload:{state:'saved',policy_approved:true}}))});
  assert.equal(nativePost.nodes.get('save').hidden,false);await completePost(nativePost);assert.equal(nativePost.parentMessages.length,0);await nativePost.nodes.get('save').onclick();
  assert.equal(postPosts(nativePost).length,1);assert.deepEqual(JSON.parse(postPosts(nativePost)[0].options.body),{decisions:heard,queue_sha256:'fixed-queue'});
  assert.equal(continuousPre.nodes.get('review-heading').textContent,'호출어 발음 사전 검수');
  assert.match(html,/\.compact-call-review :is\(\.review-guidance,#review-instruction,#review-progress,#expected\)\{display:none\}/);
  const repairedQueue=()=>{
    const items=Array.from({length:50},(_,i)=>({id:`retained-${i+1}`,text:'가영아',group:i<20?'학습용':'공통 평가용',
      voice_number:i<20?i+1:i-19,audio:`/audio/repair-${i+1}`,
      audio_sha256:crypto.createHash('sha256').update(`repair-audio-${i+1}`).digest('hex')}));
    items[49].id='new-representative';
    return {kind:'positive',review_phase:'post',wakeword:'가영아',queue_sha256:'fixed-queue',items,
      retained_decisions:Object.fromEntries(items.slice(0,49).map(row=>[row.id,{decision:'keep',listened:true,audio_sha256:row.audio_sha256}])),
      focus_sample_ids:[items[49].id]};
  };
  const repairedData=repairedQueue(),repaired=await screen(repairedData,'?project=bora&queue=positive',{
    'retained-1':{decision:'discard',listened:true},'unknown-old':{decision:'keep',listened:true}},postConfig,{...shared,
    respond:postResponse(repairedData,()=>({payload:{state:'saved',policy_approved:true,kept:50,total:50}}))});
  preMessage(repaired,'koww:preflight-status');
  assert.equal(repaired.pickers[0].items.length,1,'only the new representative is exposed for replay or editing');
  assert.equal(repaired.nodes.get('items').children.length,1);assert.equal(repaired.nodes.get('title').textContent,'1 / 1');
  assert.equal(repaired.nodes.get('audio').src,repairedData.items[49].audio);assert.equal(repaired.audioEffects.plays,0);
  assert.equal(postState(repaired).retained_count,49);assert.equal(postState(repaired).review_count,1);assert.equal(postState(repaired).ready,false);
  repaired.pickers[0].options.onSelect(49);assert.equal(repaired.nodes.get('audio').src,repairedData.items[49].audio,'hidden old rows cannot be selected through a stale list index');
  await repaired.nodes.get('start').onclick();assert.equal(repaired.writes.length,0,'retained facts do not grant listening to the new clip');
  finishPreflight(repaired,1400);await repaired.flush();assert.equal(postState(repaired).ready,true);
  assert.deepEqual(JSON.parse(repaired.writes.at(-1)[1]),{'new-representative':{decision:'keep',listened:true}},'local draft contains only the newly reviewed clip');
  preMessage(repaired,'koww:preflight-finalize');await repaired.flush();
  const repairedSubmission=JSON.parse(postPosts(repaired)[0].options.body);
  assert.equal(Object.keys(repairedSubmission.decisions).length,50);
  assert.deepEqual(repairedSubmission.decisions['retained-1'],{decision:'keep',listened:true},'server-bound retained approval overrides a stale local edit');
  assert.deepEqual(repairedSubmission.decisions['new-representative'],{decision:'keep',listened:true});
  assert.equal(repaired.parentMessages.filter(row=>row.message.type==='koww:preflight-finalized').at(-1).message.workflow_action,'resume');
  const repairAgainData=repairedQueue(),repairAgain=await screen(repairAgainData,'?project=bora&queue=positive',{'new-representative':{decision:'discard',listened:true}},postConfig,{...shared,
    respond:postResponse(repairAgainData,()=>({payload:{state:'postgeneration_repair_started',review_finalized:true,workflow_action:'repair',policy_approved:false,kept:49,total:50}}))});
  preMessage(repairAgain,'koww:preflight-status');preMessage(repairAgain,'koww:preflight-finalize');preMessage(repairAgain,'koww:preflight-finalize');await repairAgain.flush();
  assert.equal(postPosts(repairAgain).length,1);
  assert.equal(repairAgain.parentMessages.filter(row=>row.message.type==='koww:preflight-finalized').at(-1).message.workflow_action,'repair');
  assert.equal(repairAgain.nodes.get('start').disabled,true);assert.equal(repairAgain.nodes.get('discard').disabled,true);
  assert.equal(repairAgain.parentMessages.some(row=>row.message.type==='koww:preflight-held'),false);
  for(const mutate of [
    d=>{d.retained_decisions['retained-1'].audio_sha256='f'.repeat(64);},
    d=>{d.retained_decisions['retained-1'].decision='discard';},
    d=>{d.retained_decisions['retained-1'].listened=false;},
    d=>{d.retained_decisions.unknown={decision:'keep',listened:true,audio_sha256:'a'.repeat(64)};},
    d=>{d.focus_sample_ids.push(d.focus_sample_ids[0]);},
    d=>{d.focus_sample_ids=['retained-1'];},
    d=>{d.focus_sample_ids=['unknown'];},
    d=>{delete d.focus_sample_ids;},
    d=>{d.focus_sample_ids=[];},
    d=>{d.retained_decisions=null;},
  ]){
    const invalid=repairedQueue();mutate(invalid);const view=await screen(invalid,'?project=bora&queue=positive',{},postConfig,shared);
    assert.equal(view.readiness.at(-1).status,'error','invalid or unbound carry evidence cannot reveal a review');
    assert.equal(view.audioEffects.plays,0);assert.equal(postPosts(view).length,0);
  }
  console.log("Review UI: voice purpose, saved decisions, unheard gate and other review types passed");
  // Hard review has its own pool-admission acknowledgement. Deliberate Next
  // confirms use; playback never enters positive regeneration or admission.
  {
  const hardQueue=()=>({...queue,policy_review:true,purpose:'학습에 사용할 유사어 음성을 확인하세요.',policy_text:'적합한 유사어 자료 묶음을 학습에 사용하는 데 동의합니다.',
    items:queue.items.map((row,i)=>({...row,text:i?'언제나':'언니야'}))});
  const hardConfig={...config,queue_url:'/review/queue?project=bora&queue=hard',save_url:'/review/save?project=bora&queue=hard'};
  const hardMessage=(view,type,changes={})=>preMessage(view,type,{phase:'hard',...changes});
  const hardSuccess=(kept=1)=>({state:'review_saved',kept,total:2,admission_pending:true});
  const hardData=hardQueue(),hard=await screen(hardData,'?project=bora&queue=hard',{},hardConfig,{...shared,
    respond:postResponse(hardData,()=>({payload:hardSuccess()}))});
  assert.equal(hard.classes.has('compact-call-review'),true);assert.equal(hard.nodes.get('review-heading').textContent,'유사어 발음 검수');
  assert.equal(hard.nodes.get('keep').hidden,true);assert.equal(hard.nodes.get('discard').textContent,'부적합');assert.equal(hard.nodes.get('save').hidden,true);
  assert.equal(hard.nodes.get('policy').hidden,true);assert.notEqual(hard.nodes.get('policy-approved').checked,true);
  assert.equal(hard.pickers[0].items[0].label,'언니야');assert.equal(hard.nodes.get('review-volume').hidden,false);
  hardMessage(hard,'koww:preflight-status');assert.equal(postState(hard).phase,'hard');assert.equal(postState(hard).ready,false);
  await hard.nodes.get('start').onclick();assert.equal(hard.writes.length,0,'starting hard playback grants neither listening nor consent');
  hard.nodes.get('discard').onpointerdown();hard.nodes.get('discard').onclick({detail:1});
  assert.deepEqual(JSON.parse(hard.writes.at(-1)[1]),{one:{decision:'discard',listened:false}});
  hard.nodes.get('discard').onpointerdown();hard.nodes.get('discard').onclick({detail:1});assert.deepEqual(JSON.parse(hard.writes.at(-1)[1]),{});
  finishPreflight(hard,1400);await hard.flush();finishPreflight(hard,1600);await hard.flush();
  assert.equal(postState(hard).ready,true,'completed listening enables a deliberate Next without approving the bundle');
  hard.nodes.get('discard').onclick();
  assert.equal(postState(hard).ready,true);assert.notEqual(hard.nodes.get('policy-approved').checked,true);
  assert.equal(postPosts(hard).length,0,'listening and local judgments never save or approve the pool');
  hard.pickers[0].show(23);assert.equal(hard.parentMessages.filter(row=>row.message.type==='koww:preflight-picker').at(-1).message.phase,'hard');
  hardMessage(hard,'koww:preflight-picker-ready',{picker_request:23});assert.deepEqual(hard.pickers[0].revealed,[23]);hard.pickers[0].close();
  hardMessage(hard,'koww:preflight-picker-closed',{picker_request:23});assert.equal(hard.classes.has('preflight-picker-open'),false);
  assert.equal(hard.parentMessages.filter(row=>row.message.type==='koww:preflight-height').at(-1).message.phase,'hard');
  preMessage(hard,'koww:preflight-finalize');hardMessage(hard,'koww:preflight-finalize',{phase:'post'});
  for(const changes of [{project:'other'},{queue_sha256:'old'},{request:6}])hardMessage(hard,'koww:preflight-finalize',changes);
  await hard.flush();assert.equal(postPosts(hard).length,0);
  hardMessage(hard,'koww:preflight-finalize');hardMessage(hard,'koww:preflight-finalize');await hard.flush();
  assert.equal(postPosts(hard).length,1);assert.deepEqual(JSON.parse(postPosts(hard)[0].options.body),{
    decisions:{one:{decision:'keep',listened:true},two:{decision:'discard',listened:true}},queue_sha256:'fixed-queue',policy_approved:true});
  assert.equal(hard.calls.filter(row=>!row.options).length,2,'hard finalization rechecks the current queue before writing');
  const hardAck=hard.parentMessages.filter(row=>row.message.type==='koww:preflight-finalized').at(-1).message;
  assert.equal(hardAck.phase,'hard');assert.equal(hardAck.workflow_action,'resume');assert.notEqual(hard.nodes.get('policy-approved').checked,true,'Next does not fake a checkbox action');
  hardMessage(hard,'koww:preflight-finalize');await hard.flush();assert.equal(postPosts(hard).length,1);
  const awaitingNextData=hardQueue(),awaitingNext=await screen(awaitingNextData,'?project=bora&queue=hard',heard,hardConfig,{...shared,
    respond:postResponse(awaitingNextData,()=>({payload:hardSuccess(2)}))});
  hardMessage(awaitingNext,'koww:preflight-status');await awaitingNext.flush();
  assert.equal(postState(awaitingNext).ready,true);assert.equal(postPosts(awaitingNext).length,0);
  assert.notEqual(awaitingNext.nodes.get('policy-approved').checked,true);
  hardMessage(awaitingNext,'koww:preflight-finalize');await awaitingNext.flush();assert.equal(postPosts(awaitingNext).length,1);
  const nativeHardData=hardQueue(),nativeHard=await screen(nativeHardData,'?project=bora&queue=hard',heard,hardConfig,{...shared,embedded:false,
    respond:postResponse(nativeHardData,()=>({payload:hardSuccess(2)}))});
  assert.equal(nativeHard.nodes.get('save').hidden,false);assert.equal(nativeHard.nodes.get('save').textContent,'다음');
  assert.equal(postPosts(nativeHard).length,0);assert.notEqual(nativeHard.nodes.get('policy-approved').checked,true);
  await nativeHard.nodes.get('save').onclick();assert.equal(postPosts(nativeHard).length,1);
  assert.equal(JSON.parse(postPosts(nativeHard)[0].options.body).policy_approved,true);assert.notEqual(nativeHard.nodes.get('policy-approved').checked,true);
  const allBad={one:{decision:'discard',listened:true},two:{decision:'discard',listened:true}},allBadData=hardQueue();
  const allBadHard=await screen(allBadData,'?project=bora&queue=hard',allBad,hardConfig,{...shared,respond:postResponse(allBadData,()=>({payload:hardSuccess(0)}))});
  hardMessage(allBadHard,'koww:preflight-status');hardMessage(allBadHard,'koww:preflight-finalize');await allBadHard.flush();
  assert.equal(allBadHard.parentMessages.some(row=>row.message.type==='koww:preflight-finalized'),false);
  assert.equal(allBadHard.parentMessages.filter(row=>row.message.type==='koww:preflight-held').at(-1).message.phase,'hard');
  assert.match(allBadHard.nodes.get('state').textContent,/적합한 유사어를 하나 이상/);
  assert.deepEqual(JSON.parse(postPosts(allBadHard)[0].options.body).decisions,allBad,'a zero-kept hold preserves every negative decision');
  for(const changed of [state=>{state.queue_sha256='changed';},state=>{state.items[0].id='changed';},state=>{state.items[0].audio='/audio/replaced';}]){
    const state=hardQueue();let reads=0;
    const view=await screen(state,'?project=bora&queue=hard',heard,hardConfig,{...shared,respond:async(url,options)=>{
      if(options)return{payload:hardSuccess(2)};const payload=JSON.parse(JSON.stringify(state));if(++reads>1)changed(payload);return{payload};}});
    hardMessage(view,'koww:preflight-status');hardMessage(view,'koww:preflight-finalize');await view.flush();
    assert.equal(postPosts(view).length,0,'a changed hard source cannot be admitted');assert.equal(postState(view).error,true);
  }
  for(const payload of [{state:'saved',policy_approved:true},{...hardSuccess(),total:3},{...hardSuccess(),kept:2},{...hardSuccess(),admission_pending:false}]){
    const state=hardQueue(),view=await screen(state,'?project=bora&queue=hard',{...heard,two:{decision:'discard',listened:true}},hardConfig,{...shared,respond:postResponse(state,()=>({payload}))});
    hardMessage(view,'koww:preflight-status');hardMessage(view,'koww:preflight-finalize');await view.flush();
    assert.equal(postState(view).error,true);assert.equal(view.parentMessages.some(row=>['koww:preflight-finalized','koww:preflight-held'].includes(row.message.type)),false);
  }
  const incompleteHard=await screen(hardQueue(),'?project=bora&queue=hard',{},hardConfig,shared);
  hardMessage(incompleteHard,'koww:preflight-status');
  incompleteHard.nodes.get('audio').played={length:1,start:()=>.5,end:()=>1};await incompleteHard.nodes.get('start').onclick();finishPreflight(incompleteHard,1400);await incompleteHard.flush();
  assert.equal(incompleteHard.writes.length,0);assert.equal(postState(incompleteHard).ready,false);
  console.log('Hard compact review: explicit Next admission, no checkbox exception, factual full playback, same-source recheck, isolated pool ACK and zero-kept hold passed');
  }
  console.log('Preflight continuous review: factual playing defaults, unsuitable toggle, ACK-independent relay, gesture/source guards, recovery and finalization passed');
  console.log('Post compact review: strict completed listening, factual local draft, trusted Next finalization, hold/error retry, stale/duplicate guards and standalone fallback passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
