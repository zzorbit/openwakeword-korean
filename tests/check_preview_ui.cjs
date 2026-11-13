"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/reference_preview.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const css = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/audio_listening.css"), "utf8");
const playerScript = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/listening_player.js"), "utf8");

async function screen(mode, items, { error, ok = true, playback, height = 144, width = 560, buttonRect = {top:22,bottom:62,left:8,width:460},autoReady=true,standalone=false,queue,config,payload,respond,workspaceHeight,candidateAreaHeight,bodyPadding={top:0,bottom:0} } = {}) {
  const nodes = new Map(), created = [], pickerCreations = [];
  class Element {
    constructor() { this.hidden=false; this.value = ""; this.textContent = ""; this.children = []; this.handlers = {}; this.attributes = {}; this.style = {setProperty(name,value){this[name]=String(value);},getPropertyValue(name){return this[name]??"";}}; this.scrollTop = 0; this.scrollHeight = 156; this.clientHeight = 68; this.offsetTop = 0; this.offsetHeight = 36; this.currentTime = 0; this.controls = true; this.played = 0; this.pauses = 0; this.paused = true; this.ended = false; this.duration = NaN; this.volume = 1; this.playbackRate = 1; this.muted = false; this.className = ""; this.classList = { add: (...names) => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), ...names])].join(" "); } }; }
    append(...nodes) { for(const node of nodes){node.mountTemplate?.();if(node.parentElement){const siblings=node.parentElement.children;siblings.splice(siblings.indexOf(node),1);}node.parentMoves=(node.parentMoves||0)+1;node.parentElement=this;node.offsetTop=this.children.length*36+4;this.children.push(node);} }
    replaceChildren(...nodes) { for(const child of this.children)child.parentElement=null;this.children=[];this.append(...nodes); }
    remove() { if(this.parentElement){const siblings=this.parentElement.children;siblings.splice(siblings.indexOf(this),1);this.parentElement=null;} }
    insertBefore(node, anchor) { assert.ok(this.children.includes(anchor),'shared transport inserts beside the existing next button');if(node===anchor)return;if(node.parentElement){const siblings=node.parentElement.children;siblings.splice(siblings.indexOf(node),1);}node.parentElement=this;this.children.splice(this.children.indexOf(anchor),0,node); }
    addEventListener(event, callback) { const previous=this.handlers[event];this.handlers[event]=previous?(...args)=>{previous(...args);callback(...args);}:callback; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    contains(node) { return node===this || this.children.some(child=>child.contains(node)); }
    focus() { document.activeElement=this; }
    getBoundingClientRect() { return {height:buttonRect.bottom-buttonRect.top,...buttonRect}; }
    get src() { return this.attributes.src; }
    set src(value) { this.attributes.src=value; this.ended=false;this.currentTime=0; }
    load() { this.ended=false; }
    play() { this.playCalls=(this.playCalls||0)+1;if(typeof this.played==='number')this.played=this.playCalls; if(this.ended)this.currentTime=0;this.ended=false;this.paused=false; this.handlers.play?.(); return playback ? playback(this.playCalls) : Promise.resolve(); }
    pause() { this.pauses++; this.paused=true; this.handlers.pause?.(); }
    finish() { this.ended=true; this.paused=true; this.handlers.pause?.(); this.handlers.ended?.(); }
    change(checked=!this.checked) { this.checked=checked;return this.handlers.change?.(); }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], new Element());
  for(const id of nodes.keys()){
    const tag=html.match(new RegExp(`<[^>]+\\bid="${id}"[^>]*>`))?.[0];
    if(tag)nodes.get(id).hidden=/\bhidden(?:\s|>)/.test(tag);
  }
  if(workspaceHeight!==undefined)nodes.get('reference-workspace').getBoundingClientRect=()=>({height:typeof workspaceHeight==='function'?workspaceHeight(nodes):workspaceHeight});
  if(candidateAreaHeight!==undefined)nodes.get('candidate-panel').getBoundingClientRect=()=>({height:typeof candidateAreaHeight==='function'?candidateAreaHeight(nodes):candidateAreaHeight});
  if(config){const node=new Element();node.textContent=JSON.stringify(config);nodes.set('koww-page-config',node);}
  nodes.get('voice-picker').append(nodes.get('voice-picker-button'),nodes.get('voice-picker-popup'));
  nodes.get('voice-picker-popup').append(nodes.get('voice-picker-list'));
  nodes.get('voice-picker-button').append(nodes.get('voice-picker-label'));
  nodes.get('candidate-empty').append(nodes.get('candidate-status-label'),nodes.get('candidate-job-icon'));
  nodes.get('candidate-progress-slot').append(nodes.get('candidate-progress'));
  nodes.get('candidate-progress').append(nodes.get('candidate-progress-fill'));
  nodes.get('voice-picker-button').setAttribute('aria-label','들을 목소리');
  nodes.get('voice-picker-button').disabled=true;
  const playbackControls=new Element(),calls = [], enhancements = [], parentMessages = [], layoutMessages = [], heightMessages=[],frameMessages=[],resizeObservers=[], messages = {}, documentEvents = {}, document = { title: html.match(/<title>([^<]+)<\/title>/)[1], hidden:false,body:new Element(), addEventListener:(name,callback)=>{documentEvents[name]=callback;}, getElementById: id => nodes.get(id)||created.find(node=>node.id===id)||null, createElement: () => {const node=new Element();created.push(node);return node;},querySelector:selector=>selector==='.listen-controls'&&nodes.has('continuous')?playbackControls:null };
  playbackControls.append(nodes.get('continuous'));
  nodes.get('reference-volume').hidden=true;nodes.get('audio').controls=false;
  nodes.get('reference-playback').append(playbackControls);
  const templateHtml=html.match(/<template id="reference-listening-template">([\s\S]*?)<\/template>/)[1];
  const templateNodes=new Map([...templateHtml.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],nodes.get(match[1])]));
  for(const id of templateNodes.keys())nodes.delete(id);
  nodes.get('reference-listening-template').content={cloneNode(){const fragment=new Element();fragment.mountTemplate=()=>{for(const [id,node] of templateNodes)nodes.set(id,node);};return fragment;}};
  const timers=new Map();let timerId=0;
  const context = vm.createContext({ document, URL, innerHeight:height,innerWidth:width,location: new URL(`http://127.0.0.1:8793/${mode}/?project=bora`),
    setTimeout:(callback,delay)=>{const id=++timerId;timers.set(id,{callback,delay});return id;},clearTimeout:id=>timers.delete(id),
    requestAnimationFrame:callback=>setImmediate(callback),cancelAnimationFrame:handle=>clearImmediate(handle),
    getComputedStyle:element=>element===document.body?{paddingTop:`${bodyPadding.top}px`,paddingBottom:`${bodyPadding.bottom}px`}:{},
    ResizeObserver:class{constructor(callback){this.callback=callback;resizeObservers.push(this);}observe(node){this.node=node;}disconnect(){this.disconnected=true;}},
    fetch: async (url, options) => { calls.push({ url, options });if(queue)await queue;const response=respond?await respond(url,options,calls.length):{ok,payload:payload||{items,error}};return { ok:response.ok!==false,status:response.status??(response.ok===false?400:200), json: async () => JSON.parse(JSON.stringify(response.payload)) }; } });
  context.window = context; context.parent = standalone?context:{location:{origin:context.location.origin},postMessage:(message,origin)=>{
    const entry={message:JSON.parse(JSON.stringify(message)),origin};frameMessages.push(entry);
    (message.type==='koww:reference-layout'?layoutMessages:message.type==='koww:reference-height'?heightMessages:parentMessages).push(entry);
    if(autoReady&&message.type==='koww:voice-picker'&&message.open)messages.message({origin,source:context.parent,data:{type:'koww:voice-picker-ready',request:message.request}});
  }};
  context.addEventListener = (name, callback) => { const previous=messages[name];messages[name]=previous?event=>{previous(event);callback(event);}:callback; };
  vm.runInContext(playerScript,context);
  const { enhance, createPicker } = context.KowwListening;
  context.KowwListening={createPicker:options=>{pickerCreations.push(options);return createPicker(options);},enhance:(audio,controls,options)=>{
    const handle=enhance(audio,controls,options);enhancements.push({audio,controls,options,handle});return handle;
  }};
  vm.runInContext(script, context);
  for (let index = 0; index < 8; index++) await new Promise(resolve => setImmediate(resolve));
  return { nodes, calls, enhancements, pickerCreations, playbackControls, document, documentEvents, messages, parentMessages,layoutMessages,heightMessages,frameMessages,resizeObservers,notifyLayout:()=>resizeObservers.forEach(observer=>observer.callback()),parent: context.parent,timers,flush:async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));},poll:async()=>{const [id,timer]=timers.entries().next().value||[];assert.ok(timer,'a status poll is scheduled');timers.delete(id);await timer.callback();}, ready:request=>messages.message({origin:'http://127.0.0.1:8793',source:context.parent,data:{type:'koww:voice-picker-ready',request}}),resize:(nextHeight,nextRect)=>{context.innerHeight=nextHeight;if(nextRect)Object.assign(buttonRect,nextRect);messages.resize();} };
}

const referenceNotices=view=>view.frameMessages.filter(entry=>entry.message.type==='koww:reference-notice');

(async () => {
  assert.match(html, /href="\/audio_listening\.css"/);
  assert.match(html, /class="listen-workspace"/);
  assert.match(html, /id="reference-workspace"[^>]*\bhidden\b/,'initial HTML hides playback before scripts and the first queue response');
  assert.match(html, /#reference-workspace\[hidden\][^{]*\{display:none!important\}/,'the initial guard wins over shared workspace layout CSS');
  assert.doesNotMatch(html, /class="listen-sidebar"|id="(?:previous|next)"/);
  assert.match(html, /class="listen-player"/);
  assert.doesNotMatch(html, /듣기는 선택 사항이며 따로 승인할 필요가 없습니다/);
  assert.match(html,/<audio id="audio"[^>]* hidden>/,'the initial document contains only a plain hidden audio engine');
  assert.match(html,/#audio\[hidden\][^{]*\{display:none!important\}/,'the plain engine cannot become a blank audio row through shared display rules');
  const listTemplate=html.match(/<template id="reference-listening-template">([\s\S]*?)<\/template>/)[1];
  assert.match(listTemplate,/id="voice"/);assert.match(listTemplate,/id="reference-playback"/);assert.match(listTemplate,/id="reference-group-actions"/);
  assert.doesNotMatch(html, /<form|localStorage|listened|whole_pool_policy_approved/);
  assert.doesNotMatch(html, /reference-more-|reference-playback-actions|play-single|play-auto/);
  assert.match(css,/\.listen-voice-compact \.listen-heading\{[^}]*gap:2px/);
  assert.match(css,/\.voice-picker-popup\{[^}]*overflow:hidden/,'the rounded popup clips its inner native scrollbar');
  assert.match(css,/\.voice-picker-list\{[^}]*overflow-y:auto/,'the existing listbox remains the keyboard-scrollable element');
  assert.match(css,/\.voice-picker-button\{[^}]*color:#e9edf2[^}]*transition:background-color \.2s ease/);
  assert.match(css,/\.voice-picker-option\{[^}]*display:flex[^}]*color:#e9edf2/,'voice numbers follow their labels instead of filling a distant grid column');
  assert.doesNotMatch(css.match(/\.voice-picker-option\{([^}]*)\}/)[1],/transition/,'list keyboard and pointer highlights remain immediate');
  assert.match(css,/\.voice-picker-option\[data-active="true"\]\{[^}]*color:#e9edf2/,'hover keeps the same foreground color');
  const items = [
    { id: "hard1", text: "돌아와", voice_family_id: "train_01", audio: "/hard-preview/audio/0?project=bora" },
    { id: "hard2", text: "기다려", voice_family_id: "train_02", audio: "/hard-preview/audio/1?project=bora" }
  ];
  const hard = await screen("hard-preview", items, {autoReady:false});
  assert.equal(hard.document.title, "유사어 전체 듣기");
  assert.equal(hard.nodes.get("voice").attributes["aria-label"], "들을 유사어 음원");
  assert.equal(hard.nodes.get("voice").hidden, true);
  assert.equal(hard.nodes.get("voice").children[0].textContent, "돌아와 · 목소리 1");
  assert.equal(hard.nodes.get("voice-picker-label").textContent, "돌아와 · 목소리 1");
  assert.equal(hard.nodes.get("position").textContent, "1 / 2");
  const hardPicker = hard.nodes.get("voice-picker-button"), hardList = hard.nodes.get("voice-picker-list");
  hardPicker.handlers.click();
  assert.equal(hardList.hidden, false, "the modal picker opens without a parent layout acknowledgement");
  assert.deepEqual(hard.parentMessages, []);
  assert.equal(hardList.children.length, 2);
  assert.equal(hardList.children[0].attributes["aria-selected"], "true");
  assert.equal(hardList.children[0].attributes["aria-label"], "돌아와 · 목소리 1");
  assert.equal(hardList.children[0].children.some(child => child.className === "voice-picker-badge"), false);
  hardPicker.handlers.click();
  assert.equal(hard.enhancements.length,1);assert.equal(hard.enhancements[0].controls,hard.playbackControls);
  assert.equal(hard.playbackControls.children.at(-1), hard.nodes.get("reference-volume"));
  assert.equal(hard.nodes.get("reference-volume").hidden, false);assert.equal(hard.nodes.get("reference-volume").disabled, false);
  assert.equal(hard.nodes.get('audio').controls,false);
  assert.equal(hard.nodes.get('reference-playback').hidden,false,'whole-corpus listening retains its visible player bar');
  assert.equal(hard.nodes.get('candidate-volume').hidden,true);
  assert.equal(hard.nodes.get("audio").src, items[0].audio);
  assert.equal(hard.nodes.get("audio").played, 0, "opening the modal does not play the first clip");
  // Native audio play has the same continuous behavior as the visible play button.
  await hard.nodes.get("audio").play();
  assert.equal(hard.nodes.get("audio").played, 1);
  hard.nodes.get("audio").finish();
  assert.equal(hard.nodes.get("audio").src, items[1].audio);
  assert.equal(hard.nodes.get("audio").played, 2);
  assert.equal(hardList.children[0].children.at(-1).textContent, "들음");
  assert.equal(hard.nodes.get("position").textContent, "2 / 2");
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
  assert.equal(reference.document.title, "기준 목소리 검수");
  assert.equal(reference.nodes.get("voice").children[0].textContent, "학습용 · 목소리 1");
  assert.equal(reference.nodes.get("voice").hidden, true);
  assert.equal(reference.nodes.get("voice-picker").hidden, false);
  assert.equal(reference.nodes.get("voice-picker-label").textContent, "학습용 · 1");
  assert.equal(reference.nodes.get("voice-picker-button").attributes['aria-label'], "들을 목소리: 학습용 · 목소리 1");
  assert.equal(reference.nodes.get("voice-picker-list").hidden, true);
  assert.equal(reference.nodes.get("reference-volume").hidden, false);
  assert.equal(reference.enhancements.length,1,'reference listening uses the shared custom player again');
  assert.equal(reference.enhancements[0].audio,reference.nodes.get('audio'));assert.equal(reference.enhancements[0].controls,reference.playbackControls);
  assert.equal(reference.enhancements[0].options.play,reference.nodes.get('continuous'));
  assert.equal(reference.nodes.get('audio').controls,false);
  assert.equal(reference.nodes.get('reference-playback').hidden,false,'legacy reference listening retains its visible player bar');
  assert.equal(reference.nodes.get('candidate-volume').hidden,true);
  assert.equal(reference.nodes.get("audio").played, 0); // Opening is never autoplay.
  assert.deepEqual(reference.calls, [{ url: "/reference-preview/queue?project=bora", options: undefined }]);
  const many=await screen('reference-preview',Array.from({length:7},(_,i)=>({id:`voice-${i}`,group:'학습용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`})));
  assert.equal(many.nodes.get('voice').children.length,7);
  many.nodes.get('voice').value='6';many.nodes.get('voice').handlers.change();
  assert.equal(many.nodes.get('voice').value,'6');
  assert.equal(many.nodes.get('audio').played,0,'selecting a paused recording stays paused');
  assert.equal(many.nodes.get('position').textContent,'7 / 7');
  many.nodes.get('continuous').handlers.click();many.nodes.get('audio').finish();
  assert.equal(many.nodes.get('voice').value,'6','reference listening stops at the final voice');
  assert.equal(many.nodes.get('audio').played,1);assert.equal(many.nodes.get('continuous').textContent,'재생');
  assert.equal(many.calls.length,1,'voice selection and playback make no approval or extra data request');
  const modes=await screen('reference-preview',Array.from({length:3},(_,i)=>({id:`mode-${i}`,group:'학습용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`})));
  const modeAudio=modes.nodes.get('audio'),modeBar=modes.nodes.get('continuous');
  modeBar.handlers.click();assert.equal(modeAudio.played,1);assert.equal(modeBar.textContent,'일시정지');
  modeAudio.currentTime=0.75;modeBar.handlers.click();assert.equal(modeAudio.paused,true);
  modeBar.handlers.click();assert.equal(modeAudio.played,2);assert.equal(modeAudio.currentTime,0.75,'main play resumes its paused position');
  modeAudio.finish();
  assert.equal(modes.nodes.get('voice').value,'1');assert.equal(modeAudio.played,3);
  assert.equal(modes.nodes.get('position').textContent,'2 / 3');assert.equal(modeBar.textContent,'일시정지');
  modeBar.handlers.click();assert.equal(modeAudio.paused,true);
  modeAudio.handlers.play(); // A queued play event cannot undo a newer pause.
  modeAudio.finish();
  assert.equal(modes.nodes.get('voice').value,'1');assert.equal(modeAudio.played,3);assert.equal(modeBar.textContent,'재생');
  modes.nodes.get('voice-picker-button').handlers.click();modes.nodes.get('voice-picker-list').children[2].handlers.click();
  assert.equal(modes.nodes.get('voice').value,'2');assert.equal(modeAudio.played,3,'selecting while paused never starts playback');
  modeBar.handlers.click();modes.nodes.get('voice-picker-button').handlers.click();modes.nodes.get('voice-picker-list').children[0].handlers.click();
  assert.equal(modeAudio.played,5);assert.equal(modeBar.textContent,'일시정지');
  modeAudio.finish();assert.equal(modes.nodes.get('voice').value,'1');assert.equal(modeAudio.played,6);
  modeAudio.finish();assert.equal(modes.nodes.get('voice').value,'2');assert.equal(modeAudio.played,7);
  modeAudio.finish();assert.equal(modes.nodes.get('voice').value,'2');assert.equal(modeAudio.played,7);assert.equal(modeBar.textContent,'재생');
  modeBar.handlers.click();
  modes.document.hidden=true;modes.documentEvents.visibilitychange();assert.equal(modeAudio.paused,true);assert.equal(modeAudio.src,undefined);
  assert.equal(modeBar.textContent,'재생');
  modeAudio.finish();assert.equal(modeAudio.played,8);
  modes.document.hidden=false;modeBar.handlers.click();
  modes.messages.message({origin:'http://127.0.0.1:8793',source:modes.parent,data:{type:'koww:pause'}});
  assert.equal(modeAudio.paused,true);assert.equal(modeBar.attributes['aria-pressed'],'false');
  modeAudio.finish();assert.equal(modeAudio.played,9);
  assert.equal(modes.calls.length,1,'mode selection and lifecycle controls are read-only');
  // Mount the shipped shared player once: layout changes must keep its real
  // seek/mute controls wired to the same reference transport and picker.
  const compact=await screen('reference-preview',Array.from({length:3},(_,i)=>({id:`compact-${i}`,group:'학습용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`})));
  const compactAudio=compact.nodes.get('audio'),transport=compact.enhancements[0].handle;
  assert.ok(transport);assert.equal(compactAudio.controls,false);assert.equal(compact.enhancements.length,1);
  assert.ok(compact.playbackControls.children.includes(transport.seek));
  assert.equal(transport.mute.id,'reference-mute');assert.equal(transport.mute.parentElement,compact.playbackControls);
  assert.equal(compact.playbackControls.children.at(-1),compact.nodes.get('reference-volume'));
  compactAudio.duration=4;compactAudio.currentTime=1;compactAudio.handlers.loadedmetadata();
  transport.seek.value='750';transport.seek.handlers.input();
  assert.equal(compactAudio.currentTime,3);assert.equal(transport.seek.attributes['aria-valuetext'],'0:03 / 0:04');
  assert.equal(transport.seek.style.getPropertyValue('--listen-progress'),'75%');
  assert.equal(compactAudio.played,0,'seeking a paused voice never starts playback');
  transport.mute.handlers.click();assert.equal(compactAudio.muted,true);assert.equal(transport.mute.attributes['aria-pressed'],'true');
  compact.nodes.get('continuous').handlers.click();assert.equal(compactAudio.played,1);
  compactAudio.finish();assert.equal(compact.nodes.get('voice').value,'1');assert.equal(compactAudio.played,2);
  compact.nodes.get('continuous').handlers.click();assert.equal(compactAudio.paused,true);
  const compactButton=compact.nodes.get('voice-picker-button'),compactList=compact.nodes.get('voice-picker-list');
  const compactKey=(element,key)=>element.handlers.keydown({key,preventDefault(){}});
  compactKey(compactButton,'ArrowDown');assert.equal(compactList.hidden,false);
  compactKey(compactList,'End');compactKey(compactList,'Enter');
  assert.equal(compact.nodes.get('voice').value,'2');assert.equal(compactList.hidden,true);assert.equal(compact.document.activeElement,compactButton);
  assert.equal(compactAudio.played,2,'keyboard selection after pause remains paused');
  transport.mute.handlers.click();assert.equal(compactAudio.muted,false);
  assert.equal(compact.calls.length,1,'the shared compact player never writes listening decisions or reloads the queue');
  const volumeScreen=await screen('reference-preview',[{id:'volume-0',group:'학습용',text:'오늘은',audio:'/reference-preview/audio/0?project=bora'}]);
  const volumeAudio=volumeScreen.nodes.get('audio'),volumeControl=volumeScreen.nodes.get('reference-volume');
  const volumeMute=volumeScreen.enhancements[0].handle.mute;
  const volumeTag=html.match(/<input\b[^>]*\bid="reference-volume"[^>]*>/)[0];
  for(const attribute of ['min="0"','max="100"','step="1"'])assert.ok(volumeTag.includes(attribute));
  assert.equal(volumeControl.value,'100');assert.equal(volumeControl.getAttribute('aria-valuetext'),'100%');
  assert.equal(volumeControl.style.getPropertyValue('--listen-progress'),'100%');
  volumeAudio.currentTime=1.25;
  const mediaState=()=>({src:volumeAudio.src,currentTime:volumeAudio.currentTime,played:volumeAudio.played,pauses:volumeAudio.pauses,paused:volumeAudio.paused,ended:volumeAudio.ended,rate:volumeAudio.playbackRate});
  for(const playing of [false,true]){
    if(playing)volumeScreen.nodes.get('continuous').handlers.click();
    for(const value of [35,0,100]){
      const before=mediaState();volumeControl.value=String(value);volumeControl.handlers.input();volumeAudio.handlers.volumechange();
      assert.equal(volumeAudio.volume,value/100);assert.equal(volumeControl.value,String(value));assert.equal(volumeControl.getAttribute('aria-valuetext'),`${value}%`);
      assert.equal(volumeControl.style.getPropertyValue('--listen-progress'),`${value}%`,'the volume fill follows the selected volume');
      assert.deepEqual(mediaState(),before,'volume input changes neither paused nor active playback');
    }
  }
  const beforeVolumeChange=mediaState();volumeAudio.volume=0.37;volumeAudio.handlers.volumechange();
  assert.equal(volumeControl.value,'37');assert.equal(volumeControl.getAttribute('aria-valuetext'),'37%');assert.match(volumeMute.getAttribute('title'),/37%/);
  assert.equal(volumeControl.style.getPropertyValue('--listen-progress'),'37%','external volume changes refresh the painted fill');
  volumeMute.handlers.click();volumeAudio.handlers.volumechange();
  assert.equal(volumeAudio.muted,true);assert.equal(volumeAudio.volume,0.37);assert.equal(volumeMute.attributes['aria-pressed'],'true');
  assert.equal(volumeMute.getAttribute('title'),'소리 켜기');assert.equal(volumeMute.getAttribute('data-muted'),'true');
  volumeMute.handlers.click();volumeAudio.handlers.volumechange();
  assert.equal(volumeAudio.muted,false);assert.equal(volumeAudio.volume,0.37);assert.equal(volumeMute.attributes['aria-pressed'],'false');
  assert.equal(volumeControl.value,'37');assert.equal(volumeControl.getAttribute('aria-valuetext'),'37%');assert.match(volumeMute.getAttribute('title'),/37%/);assert.equal(volumeMute.getAttribute('data-muted'),'false');
  volumeAudio.muted=true;volumeAudio.handlers.volumechange();assert.equal(volumeMute.attributes['aria-pressed'],'true');
  assert.equal(volumeMute.getAttribute('title'),'소리 켜기');
  volumeAudio.muted=false;volumeAudio.handlers.volumechange();assert.equal(volumeMute.attributes['aria-pressed'],'false');
  assert.deepEqual(mediaState(),beforeVolumeChange,'external volume changes and mute restoration preserve the active audio');
  volumeAudio.muted=true;volumeControl.value='25';volumeControl.handlers.input();volumeAudio.handlers.volumechange();
  assert.equal(volumeAudio.muted,false);assert.equal(volumeAudio.volume,0.25,'raising volume unmutes the inline player');
  assert.deepEqual(volumeScreen.parentMessages,[],'inline volume and mute never request iframe expansion');
  assert.equal(volumeScreen.calls.length,1,'volume controls do not autoplay, reload the queue, or save listening decisions');
  const positiveRows=Array.from({length:101},(_,i)=>{
    const group=i<50?'학습용':'공통 평가용',number=i<50?i+1:i-49;
    const gender=i%2?'여성':'남성',age_group=i%2?'노년':'어린이';
    return {id:`positive-${i}`,group,number,voice_number:17,gender,age_group,
      label:`${group} · ${gender} · ${age_group} · ${number}번`,text:'가영아',audio:`/positive-preview/audio/${i}?project=bora`};
  });
  const positiveLabels=await screen('positive-preview',[],{respond:async(url,options)=>{
    assert.equal(options,undefined,'entire-list listening remains read only');
    const start=Number(new URL(url,'http://127.0.0.1:8793').searchParams.get('start'));
    return {payload:{kind:'positive',start,count:positiveRows.length,items:positiveRows.slice(start,start+100)}};
  }});
  assert.equal(positiveLabels.nodes.get('voice-picker-label').textContent,positiveRows[0].label);
  assert.equal(positiveLabels.nodes.get('voice').children[50].textContent,positiveRows[50].label);
  positiveLabels.nodes.get('voice-picker-button').handlers.click();
  positiveLabels.nodes.get('voice-picker-list').children[50].handlers.click();
  assert.equal(positiveLabels.nodes.get('audio').src,positiveRows[50].audio);
  assert.equal(positiveLabels.nodes.get('voice-picker-label').textContent,positiveRows[50].label);
  assert.equal(positiveLabels.nodes.get('position').textContent,'51 / 101');
  positiveLabels.nodes.get('listen-page-forward').handlers.click();await positiveLabels.flush();
  assert.equal(positiveLabels.nodes.get('voice-picker-label').textContent,positiveRows[100].label);
  assert.equal(positiveLabels.nodes.get('audio').src,positiveRows[100].audio);
  assert.equal(positiveLabels.nodes.get('position').textContent,'101 / 101');
  assert.equal(positiveLabels.nodes.get('voice').children.length,1,'the last page retains its real audio count');
  assert.equal(positiveLabels.nodes.get('audio').played,0,'changing labels or pages does not start playback');
  const choiceItems=Array.from({length:50},(_,i)=>({id:`choice-${i}`,group:i<20?'학습용':'공통 평가용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`}));
  const choices=await screen('reference-preview',choiceItems);
  const pickerButton=choices.nodes.get('voice-picker-button'),pickerList=choices.nodes.get('voice-picker-list'),pickerPopup=choices.nodes.get('voice-picker-popup');
  const key=(element,value)=>{let prevented=false;element.handlers.keydown({key:value,preventDefault:()=>{prevented=true;}});return prevented;};
  pickerButton.handlers.click();
  assert.equal(pickerList.hidden,false);assert.equal(pickerPopup.hidden,false);assert.equal(pickerButton.attributes['aria-expanded'],'true');
  assert.equal(pickerList.parentElement,pickerPopup,'the scrollable listbox stays inside the rounded clipping boundary');
  assert.equal(choices.document.activeElement,pickerList);
  assert.equal(pickerList.children.length,50);
  assert.deepEqual(pickerList.children[0].children.map(node=>node.textContent),['학습용','1']);assert.equal(pickerList.children[0].attributes['aria-label'],'학습용 · 목소리 1');
  assert.deepEqual(pickerList.children[19].children.map(node=>node.textContent),['학습용','20']);
  assert.deepEqual(pickerList.children[20].children.map(node=>node.textContent),['공통 평가용','1']);
  assert.deepEqual(pickerList.children[49].children.map(node=>node.textContent),['공통 평가용','30']);assert.equal(pickerList.children[49].attributes['aria-label'],'공통 평가용 · 목소리 30');
  assert.equal(pickerPopup.style.maxHeight,'75px','popup fits the available iframe and overlaps the selected control border');
  assert.equal(pickerList.style.maxHeight,'73px','the inner scrollbar reserves both wrapper border pixels');
  assert.equal(pickerPopup.style.top,'61px');assert.equal(pickerPopup.style.left,'8px');assert.equal(pickerPopup.style.width,'460px');
  assert.equal(pickerPopup.attributes['data-side'],'down');
  assert.ok(parseFloat(pickerPopup.style.top)+parseFloat(pickerPopup.style.maxHeight)<=144-8);
  assert.deepEqual(choices.parentMessages,[{message:{type:'koww:voice-picker',open:true,request:1},origin:'http://127.0.0.1:8793'}]);
  choices.resize(300,{top:100,bottom:140});
  assert.equal(pickerPopup.style.maxHeight,'153px');assert.equal(pickerList.style.maxHeight,'151px');
  assert.ok(parseFloat(pickerPopup.style.top)+parseFloat(pickerPopup.style.maxHeight)<=300-8);
  choices.resize(700,{top:100,bottom:140});assert.equal(pickerPopup.style.maxHeight,'360px');assert.equal(pickerList.style.maxHeight,'358px');
  choices.resize(300,{top:240,bottom:280});
  assert.equal(pickerPopup.style.top,'8px','picker opens upward when space below is smaller');
  assert.equal(pickerPopup.style.maxHeight,'233px');assert.equal(pickerList.style.maxHeight,'231px');assert.equal(pickerPopup.attributes['data-side'],'up');
  assert.equal(key(pickerList,'End'),true);assert.equal(pickerList.attributes['aria-activedescendant'],'voice-option-49');
  assert.ok(pickerList.scrollTop>=pickerList.children[49].offsetTop+pickerList.children[49].offsetHeight-pickerList.clientHeight,'End scrolls the inner listbox to reveal its final row');
  key(pickerList,'Home');key(pickerList,'ArrowDown');key(pickerList,'Enter');
  assert.equal(choices.nodes.get('voice').value,'1');assert.equal(choices.nodes.get('position').textContent,'2 / 50');
  assert.equal(choices.nodes.get('voice-picker-label').textContent,'학습용 · 2');assert.equal(pickerList.hidden,true);assert.equal(pickerPopup.hidden,true);
  assert.equal(choices.document.activeElement,pickerButton);assert.equal(choices.nodes.get('audio').played,0,'choosing a voice does not autoplay');
  assert.deepEqual(choices.parentMessages.at(-1),{message:{type:'koww:voice-picker',open:false,request:1},origin:'http://127.0.0.1:8793'});
  key(pickerButton,' ');key(pickerList,'ArrowUp');key(pickerList,' ');
  assert.equal(choices.nodes.get('voice').value,'0');assert.equal(pickerList.hidden,true);
  key(pickerButton,'ArrowDown');key(pickerList,'Escape');
  assert.equal(pickerList.hidden,true);assert.equal(choices.document.activeElement,pickerButton);
  pickerButton.handlers.click();choices.documentEvents.pointerdown({target:pickerList.children[3]});
  assert.equal(pickerList.hidden,false,'clicking inside the picker does not dismiss it before selection');
  pickerList.children[3].handlers.click();assert.equal(choices.nodes.get('voice').value,'3');assert.equal(pickerList.hidden,true);
  pickerButton.handlers.click();choices.documentEvents.pointerdown({target:choices.nodes.get('continuous')});assert.equal(pickerList.hidden,true);
  pickerButton.handlers.click();choices.messages.message({origin:'http://127.0.0.1:8793',source:choices.parent,data:{type:'koww:pause'}});
  assert.equal(pickerList.hidden,true);assert.equal(choices.nodes.get('audio').src,undefined);
  pickerButton.handlers.click();choices.document.hidden=true;choices.documentEvents.visibilitychange();assert.equal(pickerList.hidden,true);
  choices.document.hidden=false;pickerButton.handlers.click();choices.messages.pagehide();assert.equal(pickerList.hidden,true);
  assert.equal(choices.calls.length,1,'picker interactions do not request approvals or reload the queue');
  pickerButton.handlers.click();pickerList.children[49].handlers.click();
  assert.equal(choices.nodes.get('voice').value,'49');assert.equal(choices.nodes.get('position').textContent,'50 / 50');
  assert.equal(choices.nodes.get('voice-picker-label').textContent,'공통 평가용 · 30');assert.equal(pickerList.children[49].attributes['aria-selected'],'true');
  assert.equal(pickerList.children[49].children[1].textContent,'30','the selected voice number stays separate from a possibly truncated group name');
  const awaiting=await screen('reference-preview',choiceItems,{autoReady:false});
  const awaitingButton=awaiting.nodes.get('voice-picker-button'),awaitingList=awaiting.nodes.get('voice-picker-list'),awaitingPopup=awaiting.nodes.get('voice-picker-popup');
  awaitingButton.focus();key(awaitingButton,'End');
  assert.equal(awaitingButton.attributes['aria-expanded'],'true');assert.equal(awaitingList.hidden,true);assert.equal(awaitingPopup.hidden,true);
  assert.equal(awaiting.document.activeElement,awaitingButton,'the list does not take focus while the parent layout is pending');
  assert.equal(awaitingPopup.style.top,undefined);
  awaiting.resize(300,{top:100,bottom:140});assert.equal(awaitingPopup.style.top,undefined,'resize before acknowledgment cannot show or position the small pending popup');
  awaiting.messages.message({origin:'https://other.example',source:awaiting.parent,data:{type:'koww:voice-picker-ready',request:1}});
  awaiting.messages.message({origin:'http://127.0.0.1:8793',source:{},data:{type:'koww:voice-picker-ready',request:1}});
  for(const request of [0,-1,1.5,'1',undefined])awaiting.ready(request);
  assert.equal(awaitingList.hidden,true,'only an authentic acknowledgment for this open request can show the list');
  awaiting.ready(1);assert.equal(awaitingList.hidden,false);assert.equal(awaitingPopup.hidden,false);assert.equal(awaitingPopup.style.maxHeight,'153px');assert.equal(awaitingList.style.maxHeight,'151px');
  assert.equal(awaitingList.attributes['aria-activedescendant'],'voice-option-49','End while pending is preserved by the ready acknowledgment');
  assert.equal(awaiting.document.activeElement,awaitingList);assert.equal(awaiting.nodes.get('audio').played,0);
  key(awaitingList,'Escape');awaiting.ready(1);assert.equal(awaitingList.hidden,true,'late acknowledgments cannot reopen a closed picker');
  key(awaitingButton,'Home');assert.equal(awaitingList.hidden,true);
  assert.equal(awaiting.parentMessages.at(-1).message.request,2);
  awaiting.ready(1);assert.equal(awaitingList.hidden,true,'a prior open acknowledgment cannot satisfy a newer open request');
  awaiting.ready(2);assert.equal(awaitingList.hidden,false);assert.equal(awaitingList.attributes['aria-activedescendant'],'voice-option-0');
  key(awaitingList,'Escape');awaitingButton.handlers.click();key(awaitingButton,'Escape');awaiting.ready(3);
  assert.equal(awaitingList.hidden,true,'Escape can cancel a pending request before the list receives focus');
  awaitingButton.handlers.click();awaiting.messages.message({origin:'http://127.0.0.1:8793',source:awaiting.parent,data:{type:'koww:voice-picker-close'}});awaiting.ready(4);
  assert.equal(awaitingList.hidden,true);
  awaitingButton.handlers.click();awaiting.messages.message({origin:'http://127.0.0.1:8793',source:awaiting.parent,data:{type:'koww:pause'}});awaiting.ready(5);
  assert.equal(awaitingList.hidden,true);assert.equal(awaiting.nodes.get('audio').src,undefined);
  awaitingButton.handlers.click();awaiting.document.hidden=true;awaiting.documentEvents.visibilitychange();awaiting.document.hidden=false;awaiting.ready(6);
  assert.equal(awaitingList.hidden,true);
  awaitingButton.handlers.click();key(awaitingButton,'Tab');awaiting.ready(7);assert.equal(awaitingList.hidden,true);
  const standalone=await screen('reference-preview',choiceItems,{standalone:true});
  standalone.nodes.get('voice-picker-button').handlers.click();assert.equal(standalone.nodes.get('voice-picker-list').hidden,false);
  assert.deepEqual(standalone.parentMessages,[],'standalone listening opens directly without waiting for a parent');
  let resolveQueue;const queueGate=new Promise(resolve=>{resolveQueue=resolve;});
  const loading=await screen('reference-preview',choiceItems,{autoReady:false,queue:queueGate});
  loading.ready(1);assert.equal(loading.document.getElementById('voice-picker-button'),null,'no list control exists before the queue chooses a layout');
  assert.deepEqual(loading.parentMessages,[]);assert.equal(loading.enhancements.length,0);assert.equal(loading.pickerCreations.length,0);
  resolveQueue();for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));
  loading.nodes.get('voice-picker-button').handlers.click();assert.equal(loading.nodes.get('voice-picker-list').hidden,true);
  loading.ready(1);assert.equal(loading.nodes.get('voice-picker-list').hidden,false);assert.equal(loading.calls.length,1);
  const parentClose=await screen('reference-preview',choiceItems);
  parentClose.nodes.get('continuous').handlers.click();parentClose.nodes.get('voice-picker-button').handlers.click();
  parentClose.messages.message({origin:'https://other.example',source:parentClose.parent,data:{type:'koww:voice-picker-close'}});
  parentClose.messages.message({origin:'http://127.0.0.1:8793',source:{},data:{type:'koww:voice-picker-close'}});
  assert.equal(parentClose.nodes.get('voice-picker-list').hidden,false,'foreign messages cannot close the reference picker');
  parentClose.messages.message({origin:'http://127.0.0.1:8793',source:parentClose.parent,data:{type:'koww:voice-picker-close'}});
  assert.equal(parentClose.nodes.get('voice-picker-list').hidden,true);
  assert.equal(parentClose.nodes.get('audio').paused,false,'clicking the parent outside the picker does not interrupt playback');
  assert.equal(parentClose.nodes.get('audio').played,1);
  assert.deepEqual(hard.parentMessages,[],'hard-example playback never requests reference-picker expansion');
  const selectHardClip = (preview, index) => {
    preview.nodes.get("voice-picker-button").handlers.click();
    preview.nodes.get("voice-picker-list").children[index].handlers.click();
  };
  const repeated=await screen('hard-preview',Array.from({length:7},(_,i)=>({id:`hard-${i}`,text:'돌아와',audio:`/hard-preview/audio/${i}?project=bora`})));
  selectHardClip(repeated, 6);
  assert.equal(repeated.nodes.get('voice').value,'6');
  assert.equal(repeated.nodes.get('voice-picker-label').textContent,'돌아와 · 목소리 7');
  assert.equal(repeated.nodes.get('audio').src,'/hard-preview/audio/6?project=bora');
  assert.equal(repeated.nodes.get('audio').played,0);
  assert.equal(repeated.calls.length,1);
  const paused = await screen("hard-preview", items);
  selectHardClip(paused, 1);
  paused.nodes.get("continuous").handlers.click();
  paused.nodes.get("voice-picker-button").handlers.click();
  const pausesBefore=paused.nodes.get("audio").pauses;
  paused.messages.message({origin: "https://other.example", source: paused.parent, data: {type: "koww:pause"}});
  paused.messages.message({origin: "http://127.0.0.1:8793", source: {}, data: {type: "koww:pause"}});
  assert.equal(paused.nodes.get("audio").pauses, pausesBefore);
  paused.messages.message({origin: "http://127.0.0.1:8793", source: paused.parent, data: {type: "koww:pause"}});
  assert.equal(paused.nodes.get("audio").pauses, pausesBefore+1);
  assert.equal(paused.nodes.get("voice-picker-list").hidden, true);
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
  selectHardClip(pending, 1);
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
  assert.equal(failed.nodes.get('reference-workspace').hidden,true,'first-load failures cannot expose the default legacy player');
  assert.equal(failed.nodes.get('reference-startup').hidden,false);assert.equal(failed.nodes.get('reference-startup').attributes['role'],'alert');
  assert.equal(failed.nodes.get('reference-startup-status').textContent,'준비 자료가 없습니다.');assert.equal(failed.nodes.get('reference-startup-retry').hidden,false);
  const unsafeText = await screen("hard-preview", [{ ...items[0], text: "<img src=x onerror=bad()>" }]);
  assert.equal(unsafeText.nodes.get("voice-picker-label").textContent, "<img src=x onerror=bad()> · 목소리 1");
  // The candidate-pool contract deliberately separates hearing, selecting,
  // requesting more candidates and approving a quota. Card buttons control one audio source.
  const candidateConfig={save_url:'/reference-preview/review?project=bora',token:'test-token'};
  function assertGeneratingStatus(view){
    assert.equal(view.nodes.get('candidate-more').hidden,true,'the bordered right button is absent while creating');
    assert.equal(view.nodes.get('candidate-more').disabled,true);
    assert.equal(view.nodes.get('candidate-actions').hidden,false,'the empty right column stays reserved');
    assert.equal(view.nodes.get('candidate-empty').hidden,false,'progress stays visible even when an older batch is still on screen');
    assert.equal(view.nodes.get('candidate-status-label').textContent,'추가 기준 목소리 후보를 생성하고 있습니다...');
    assert.equal(view.nodes.get('candidate-job-icon').hidden,false);
    assert.equal(view.nodes.get('candidate-job-icon').attributes['data-state'],'running');
    assert.equal(view.nodes.get('candidate-progress-slot').hidden,false);
    assert.equal(view.nodes.get('candidate-progress').hidden,false,'the actual gauge appears in the same synchronous update as the message and spinner');
    assert.equal(view.nodes.get('candidate-empty').children[1],view.nodes.get('candidate-job-icon'),'the single spinner follows the central progress text');
  }
  const voiceOptions=JSON.parse(fs.readFileSync(path.join(__dirname,'../korean_openwakeword/resources/voice_design_options.json'),'utf8'));
  const candidateSettings=index=>voiceOptions.fields.map(field=>({id:field.id,label:field.label,
    value:voiceOptions.ages.child.controls[field.id][index%voiceOptions.ages.child.controls[field.id].length].label}));
  const nodeText=node=>`${node.textContent||''}${node.children.map(nodeText).join(' ')}`;
  function assertCandidateMetadata(body,item,displayNumber=item.number){
    const text=nodeText(body);
    const settings=body.children.find(child=>child.className==='candidate-settings');
    assert.ok(settings,'candidate settings have a visible text container');
    assert.equal(settings.children.length,item.voice_settings.length,'every actual selected option is shown');
    assert.notEqual(settings.hidden,true,'the five selected options are visible by default');
    assert.match(text,new RegExp(`후보 ${displayNumber}(?:\\D|$)`));
    assert.match(text,/시드/);assert.ok(text.includes(String(item.seed))||text.includes(item.seed.toLocaleString('ko-KR')),'the actual generation seed is shown');
    for(const setting of item.voice_settings){
      const visibleLabel={'성대 울림':'울림','말하는 속도':'속도'}[setting.label]||setting.label;
      assert.ok(text.includes(visibleLabel),`the ${setting.id} setting is labelled`);
      assert.ok(text.includes(setting.value),`the selected ${setting.id} value is shown`);
    }
    assert.doesNotMatch(text,/성대 울림|말하는 속도/,'candidate cards use the shorter display labels');
    assert.notEqual(body.hidden,true,'candidate generation settings are visible without expanding a panel');
  }
  const categories=Array.from({length:20},(_,index)=>({key:`${index<10?'train':'validation'}-${index%10}`,
    label:`${index%10<5?'남성':'여성'} ${['어린이','청소년','청년','중년','노년'][index%5]}`,
    split:index<10?'train':'validation',target:index<10?2:3,selected_count:0,approved:false}));
  const candidates=(group,start=1)=>Array.from({length:5},(_,index)=>({id:`${group}-candidate-${start+index}`,sha256:`hash-${group}-${start+index}`,
    audio:`/reference-preview/audio/${group}-${start+index}?project=bora`,text:'기준 문장',number:start+index,group_key:group,heard:false,selected:false,
    seed:start===1&&index===0?0:7000+start+index,voice_settings:candidateSettings(index)}));
  const state={mode:'candidate_pool_v2',groups:structuredClone(categories),active_group:categories[0].key,editable:true,review_complete:false,
    items:candidates(categories[0].key),accepted:[],job:null};
  const initialCandidateState=structuredClone(state);
  const requestChapter=(preview,request)=>preview.messages.message({origin:'http://127.0.0.1:8793',source:preview.parent,
    data:{type:'koww:chapter-request',project:'bora',request,url:'http://127.0.0.1:8793/reference-preview/?project=bora'}});
  const chapterReplies=preview=>preview.parentMessages.filter(row=>row.message.type==='koww:chapter-ready').map(row=>row.message);
  for(const mode of ['candidate','legacy']){
    let releaseQueue;const queue=new Promise(resolve=>{releaseQueue=resolve;});
    const delayed=await screen('reference-preview',mode==='legacy'?choiceItems:[],{queue,
      ...(mode==='candidate'?{config:candidateConfig,payload:initialCandidateState}:{}),workspaceHeight:244});
    assert.equal(delayed.nodes.get('reference-workspace').hidden,true,`${mode} playback stays hidden during the first GET`);
    assert.equal(delayed.nodes.get('reference-startup').hidden,false);assert.equal(delayed.nodes.get('reference-startup').attributes['aria-busy'],'true');
    assert.equal(delayed.nodes.get('reference-startup-retry').hidden,true);assert.equal(delayed.layoutMessages.length,0);assert.equal(delayed.heightMessages.length,0);
    requestChapter(delayed,73);await delayed.flush();assert.deepEqual(chapterReplies(delayed),[],'the parent transition remains pending while the initial queue is loading');
    releaseQueue();await delayed.flush();
    assert.equal(delayed.nodes.get('reference-workspace').hidden,false);assert.equal(delayed.nodes.get('reference-startup').hidden,true);
    assert.equal(delayed.document.getElementById('reference-playback')!==null,mode==='legacy','only the list layout constructs its player bar');
    assert.equal(delayed.enhancements.length,mode==='legacy'?1:0);assert.equal(delayed.pickerCreations.length,mode==='legacy'?1:0);
    assert.equal(delayed.layoutMessages.at(-1).message.mode,mode==='candidate'?'candidate_pool_v2':'legacy');
    assert.equal(delayed.nodes.get('candidate-list').children.length,mode==='candidate'?5:0);
    if(mode==='candidate')assert.equal(delayed.heightMessages.at(-1).message.height,244,'the revealed candidate workspace reports its natural height');
    assert.equal(delayed.calls.length,1);assert.equal(delayed.nodes.get('audio').playCalls,undefined,'revealing a layout never starts playback');
    assert.equal(chapterReplies(delayed).at(-1).status,'ready');assert.equal(chapterReplies(delayed).at(-1).request,73);
  }
  for(const failure of ['http','malformed','invalid-mode','invalid-history','throw']){
    let reads=0;
    const retry=await screen('reference-preview',[],{config:candidateConfig,respond:async()=>{
      if(++reads===1){
        if(failure==='throw')throw Error('연결 실패');
        if(failure==='http')return {ok:false,payload:{error:'목록 연결 실패'}};
        if(failure==='invalid-mode')return {payload:{...initialCandidateState,mode:'unsupported-layout'}};
        if(failure==='invalid-history')return {payload:{...initialCandidateState,batches:[{id:'bad',number:1,count:2}]}};
        return {payload:{...initialCandidateState,groups:[]}};
      }
      return {payload:initialCandidateState};
    }});
    assert.equal(retry.nodes.get('reference-workspace').hidden,true);assert.equal(retry.nodes.get('reference-startup-retry').hidden,false);
    assert.equal(retry.document.getElementById('voice'),null,'failed first payload never mounts legacy controls');
    assert.equal(retry.document.getElementById('reference-mute'),null,'failed first payload never mounts candidate controls');
    assert.equal(retry.enhancements.length,0);assert.equal(retry.pickerCreations.length,0);
    assert.ok(retry.nodes.get('reference-startup-status').textContent);assert.equal(retry.nodes.get('reference-startup').attributes['aria-busy'],'false');
    requestChapter(retry,74);await retry.flush();assert.equal(chapterReplies(retry).at(-1).status,'error');
    assert.equal(chapterReplies(retry).at(-1).message,retry.nodes.get('reference-startup-status').textContent,'the rendered retry surface is a terminal error state');
    await retry.nodes.get('reference-startup-retry').handlers.click();await retry.flush();
    assert.equal(retry.nodes.get('reference-workspace').hidden,false);assert.equal(retry.nodes.get('reference-startup').hidden,true);
    assert.equal(retry.nodes.get('candidate-list').children.length,5);assert.equal(retry.document.getElementById('reference-playback'),null);
    assert.equal(retry.enhancements.length,0);assert.equal(retry.pickerCreations.length,0);
    assert.equal(retry.calls.length,2);assert.ok(retry.calls.every(call=>!call.options),'initial retry makes read-only GETs only');
    assert.deepEqual(chapterReplies(retry).map(reply=>reply.status),['error','ready'],'retry keeps the parent request and reports only the resulting rendered state');
  }
  {
    const optionalState=structuredClone(initialCandidateState),firstBatch='optional-first',secondBatch='optional-second';
    optionalState.items=candidates(optionalState.active_group,66).map(item=>({...item,batch_id:secondBatch}));
    Object.assign(optionalState,{batches:[{id:firstBatch,number:1,count:5},{id:secondBatch,number:2,count:5}],view_batch:secondBatch,current_batch:secondBatch});
    let releaseFirstToggle,postCount=0;
    const optional=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
      if(options){
        const action=JSON.parse(options.body);assert.equal(action.action,'toggle','manual selection never adds a synthetic hear action');
        const row=optionalState.items.find(item=>item.id===action.id);assert.ok(row);assert.equal(action.sha256,row.sha256);
        if(++postCount===1)await new Promise(resolve=>{releaseFirstToggle=resolve;});
        optionalState.accepted=optionalState.accepted.some(item=>item.id===row.id)?optionalState.accepted.filter(item=>item.id!==row.id):[...optionalState.accepted,{...row,selected:true}];
      }
      return {payload:optionalState};
    }});
    const cards=optional.nodes.get('candidate-list').children,checkbox=index=>cards[index].children[3].children[0];
    const writes=()=>optional.calls.filter(call=>call.options).map(call=>JSON.parse(call.options.body));
    assert.ok(cards.every(card=>card.children[3].children[0].disabled===false));
    assert.equal(checkbox(0).attributes['aria-label'],'후보 6 선택');assert.equal(cards[0].children[3].title,'');
    checkbox(0).focus();checkbox(0).change(true);
    assert.equal(writes().length,1);assert.deepEqual(writes()[0],{action:'toggle',id:optionalState.items[0].id,sha256:optionalState.items[0].sha256});
    assert.equal(checkbox(0).checked,true);assert.equal(cards[0].attributes['data-selected'],'true');
    assert.equal(optional.nodes.get('reference-group-count').textContent,'1 / 2 선택','selection and count change before the first POST is acknowledged');
    checkbox(1).change(true);assert.equal(writes().length,1,'rapid selections share one serialized writer');
    assert.equal(checkbox(1).checked,true);assert.equal(optional.nodes.get('reference-group-count').textContent,'2 / 2 선택');
    assert.equal(optional.nodes.get('reference-group-next').disabled,true,'a projected full quota cannot approve before all writes are acknowledged');
    assert.equal(optional.document.activeElement,checkbox(0));
    releaseFirstToggle();await optional.flush();
    assert.equal(optional.nodes.get('candidate-list').children[0],cards[0]);assert.equal(optional.document.activeElement,checkbox(0));
    assert.equal(checkbox(0).checked,true);assert.equal(checkbox(0).attributes['aria-label'],'후보 6 선택 해제');
    assert.equal(optional.nodes.get('reference-group-count').textContent,'2 / 2 선택');assert.equal(optional.nodes.get('reference-group-next').disabled,false);
    assert.equal(checkbox(2).disabled,true,'the exact quota applies equally to unheard candidates');
    const beforeBlocked=writes().length;checkbox(2).change(true);await optional.flush();assert.equal(writes().length,beforeBlocked);
    checkbox(0).change(false);await optional.flush();assert.equal(checkbox(0).checked,false);assert.equal(checkbox(2).disabled,false);
    checkbox(2).change(true);await optional.flush();assert.equal(optionalState.accepted.length,2);
    assert.ok(optionalState.items.every(item=>item.heard===false));assert.ok(optionalState.accepted.every(item=>item.heard===false),'selection does not falsify listening records');
    assert.equal(optional.nodes.get('audio').playCalls,undefined,'filling and changing the quota requires no audio playback');
    assert.ok(writes().every(action=>action.action==='toggle'));
    assert.deepEqual(writes().map(action=>action.id),[optionalState.items[0].id,optionalState.items[1].id,optionalState.items[0].id,optionalState.items[2].id]);
    cards[0].children[0].handlers.click();assert.equal(cards[0].attributes['data-playing'],'true');
    assert.equal(cards[0].children[0].attributes['aria-pressed'],'true');assert.equal(cards[0].children[3].children[1].textContent,'선택','playing keeps its dedicated controls and never replaces selection text');
    cards[0].children[0].handlers.click();assert.equal(cards[0].children[3].children[1].textContent,'선택');
    for(const variation of ['running','approved','read-only']){
      const locked=structuredClone(initialCandidateState);
      if(variation==='running')locked.job={status:'running',group:locked.active_group,completed:0,total:5};
      if(variation==='approved')locked.groups[0].approved=true;
      if(variation==='read-only')locked.editable=false;
      const guarded=await screen('reference-preview',[],{config:candidateConfig,payload:locked});
      const choice=guarded.nodes.get('candidate-list').children[0].children[3].children[0];assert.equal(choice.disabled,true);
      choice.change(true);await guarded.flush();assert.equal(guarded.calls.filter(call=>call.options).length,0,`${variation} still blocks selection`);
    }
  }
  async function toggleFixture({selected=[],oldView=false}={}){
    const saved=structuredClone(initialCandidateState),group=saved.active_group;
    const banks=new Map([['toggle-old',candidates(group)],['toggle-current',candidates(group,66)]]);
    for(const [batch,rows] of banks)rows.forEach(row=>{row.batch_id=batch;});
    const initialView=oldView?'toggle-old':'toggle-current',rows=banks.get(initialView);
    saved.accepted=selected.map(index=>({...rows[index],selected:true}));
    const control={pending:[],failReads:false,readPayload:null,maxConcurrent:0,concurrent:0};
    const payload=view=>({...saved,items:banks.get(view),batches:[...banks.keys()].map((id,index)=>({id,number:index+1,count:5})),current_batch:'toggle-current',view_batch:view});
    const ui=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
      const view=new URL(url,'http://127.0.0.1:8793').searchParams.get('batch')||initialView;
      if(!options){if(control.failReads)return {ok:false,status:503,payload:{error:'상태 확인 실패'}};return {payload:control.readPayload||payload(view)};}
      const action=JSON.parse(options.body);
      if(action.action==='more'){
        assert.equal(action.group,group);assert.equal(control.concurrent,0,'generation waits for selection saves');
        saved.job={status:'running',group,completed:0,total:5};return {payload:payload(view)};
      }
      assert.equal(action.action,'toggle');
      const row=banks.get(view).find(row=>row.id===action.id&&row.sha256===action.sha256);assert.ok(row,'writes retain the candidate revision and viewed batch');
      control.concurrent++;control.maxConcurrent=Math.max(control.maxConcurrent,control.concurrent);
      return await new Promise((resolve,reject)=>control.pending.push({action,finish({commit=true,failure='',reply}={}){
        if(commit){
          const already=saved.accepted.some(item=>item.id===row.id);
          if(!already)assert.ok(saved.accepted.length<saved.groups[0].target,'the serialized writer must free a full server quota before adding another candidate');
          saved.accepted=already?saved.accepted.filter(item=>item.id!==row.id):[...saved.accepted,{...row,selected:true}];
        }
        control.concurrent--;
        if(failure==='network')reject(new Error('응답 유실'));
        else resolve(failure==='http'?{ok:false,status:503,payload:{error:'선택 저장 실패'}}:{payload:reply||payload(view)});
      }}));
    }});
    return {ui,saved,control,rows,cards:()=>ui.nodes.get('candidate-list').children,
      checkbox:index=>ui.nodes.get('candidate-list').children[index].children[3].children[0],
      writes:()=>ui.calls.filter(call=>call.options).map(call=>JSON.parse(call.options.body)),
      finish:async options=>{assert.ok(control.pending.length,'one serialized request is pending');control.pending.shift().finish(options);await ui.flush();}};
  }
  {
    const f=await toggleFixture(),first=f.checkbox(0);first.focus();
    first.change(true);assert.equal(first.checked,true);assert.equal(f.ui.nodes.get('reference-group-count').textContent,'1 / 2 선택');
    first.change(false);first.change(true);f.checkbox(1).change(true);f.checkbox(1).change(false);
    assert.equal(f.writes().length,1);assert.equal(first.checked,true);assert.equal(f.checkbox(1).checked,false);
    await f.finish();assert.equal(f.writes().length,1,'repeated desired states coalesce instead of replaying redundant toggles');
    assert.equal(f.saved.accepted.length,1);assert.equal(f.ui.document.activeElement,first);assert.equal(f.checkbox(0),first);
    assert.equal(f.control.maxConcurrent,1);assert.equal(f.ui.nodes.get('audio').playCalls,undefined);assert.ok(f.saved.accepted.every(row=>!row.heard));
  }
  {
    const f=await toggleFixture(),first=f.checkbox(0),audio=f.ui.nodes.get('audio');
    const play=f.cards().map(card=>card.children[0]),more=f.ui.nodes.get('candidate-more');
    play[0].handlers.click();audio.currentTime=.4;first.focus();
    for(const selected of [true,false]){
      first.change(selected);
      assert.ok(play.every(button=>!button.disabled),'saving a choice leaves every play button available');
      assert.equal(more.disabled,false,'saving below quota does not dim the generation button');
      assert.equal(play[0].attributes['aria-pressed'],'true');assert.equal(audio.paused,false);
      await f.finish();
      assert.equal(audio.currentTime,.4,'saving does not restart the playing clip');
      assert.equal(audio.playCalls,1);assert.equal(f.ui.document.activeElement,first);
    }
    assert.deepEqual(f.writes().map(action=>action.action),['toggle','toggle'],'generation always requires a click');
    play[0].handlers.click();first.change(true);
    play[1].handlers.click();assert.equal(audio.src,f.rows[1].audio,'play can switch clips during a choice save');
    await f.finish();assert.equal(play[1].attributes['aria-pressed'],'true');assert.equal(audio.paused,false);
  }
  {
    const f=await toggleFixture(),more=f.ui.nodes.get('candidate-more');
    f.checkbox(0).change(true);
    more.handlers.click();more.handlers.click();
    f.checkbox(0).change(false);
    assert.equal(more.hidden,false,'a queued click keeps the controls visible while the selection saves');
    assert.deepEqual(f.writes().map(action=>action.action),['toggle']);
    await f.finish();
    assert.deepEqual(f.writes().map(action=>action.action),['toggle','toggle'],'a queued generation waits for the entire selection writer');
    await f.finish();
    assert.deepEqual(f.writes().map(action=>action.action),['toggle','toggle','more'],'one generation starts after every selection is acknowledged');
    assertGeneratingStatus(f.ui);
  }
  for(const reason of ['quota','save failure','scope change']){
    const f=await toggleFixture({selected:reason==='scope change'?[0]:[]}),more=f.ui.nodes.get('candidate-more');
    f.checkbox(0).change(reason!=='scope change');
    more.handlers.click();
    if(reason==='quota'){
      f.checkbox(1).change(true);await f.finish();await f.finish();
    }else if(reason==='save failure'){
      await f.finish({commit:false,failure:'http'});
    }else{
      const different=structuredClone(initialCandidateState),nextGroup=different.groups[1].key;
      Object.assign(different,{active_group:nextGroup,items:candidates(nextGroup),accepted:[]});
      await f.finish({reply:different});
    }
    assert.ok(f.writes().every(action=>action.action==='toggle'),`${reason} cancels a queued generation`);
  }
  {
    const f=await toggleFixture({selected:[0,1]});
    f.checkbox(0).change(false);f.checkbox(2).change(true);
    assert.equal(f.checkbox(0).checked,false);assert.equal(f.checkbox(2).checked,true);
    assert.equal(f.ui.nodes.get('reference-group-count').textContent,'2 / 2 선택');assert.equal(f.ui.nodes.get('reference-group-next').disabled,true);
    assert.equal(f.checkbox(3).disabled,true);assert.equal(f.writes().length,1);
    await f.finish();assert.equal(f.writes().length,2);assert.equal(f.writes()[0].id,f.rows[0].id);assert.equal(f.writes()[1].id,f.rows[2].id);
    assert.equal(f.ui.nodes.get('reference-group-next').disabled,true,'the second projected selection cannot approve while it is still unsaved');
    await f.finish();assert.equal(f.control.maxConcurrent,1);assert.equal(f.ui.nodes.get('reference-group-next').disabled,false);
    assert.deepEqual(f.saved.accepted.map(row=>row.id),[f.rows[1].id,f.rows[2].id]);
  }
  for(const failure of ['http','network','malformed','unchanged-ack']){
    const f=await toggleFixture({oldView:true}),first=f.checkbox(0);first.focus();first.change(true);f.checkbox(1).change(true);
    const before=f.ui.calls.length;
    await f.finish({commit:failure==='network',failure:['http','network'].includes(failure)?failure:'',
      ...(failure==='malformed'?{reply:{items:[]}}:{}),...(failure==='unchanged-ack'?{reply:{...structuredClone(initialCandidateState),accepted:[]}}:{})});
    assert.equal(f.writes().length,1,'uncertain or invalid toggle replies never cause a blind toggle retry');
    assert.ok(f.ui.calls.slice(before).some(call=>!call.options),'a failed save is reconciled with an authoritative GET');
    assert.equal(new URL(f.ui.calls.at(-1).url,'http://127.0.0.1:8793').searchParams.get('batch'),'toggle-old','recovery preserves the viewed history batch');
    assert.equal(f.checkbox(0).checked,failure==='network');assert.equal(f.checkbox(1).checked,false,'all unsent intents roll back after a failed save');
    assert.equal(f.ui.nodes.get('reference-group-count').textContent,failure==='network'?'1 / 2 선택':'0 / 2 선택');
    assert.equal(f.ui.nodes.get('reference-group-next').disabled,true);assert.equal(f.ui.document.activeElement,first);
    assert.equal(f.checkbox(0),first);assert.ok(f.ui.nodes.get('error').textContent);assert.equal(f.ui.nodes.get('audio').playCalls,undefined);
  }
  {
    const f=await toggleFixture();f.checkbox(0).change(true);f.checkbox(1).change(true);f.control.failReads=true;
    await f.finish({failure:'network'});
    assert.equal(f.writes().length,1);assert.ok(f.cards().every(card=>card.children[3].children[0].disabled));
    assert.equal(f.ui.nodes.get('reference-group-next').disabled,true);assert.equal(f.ui.nodes.get('candidate-more').disabled,true);assert.equal(f.ui.nodes.get('candidate-previous').disabled,true);
    assert.equal(f.ui.nodes.get('candidate-retry-actions').hidden,false);assert.equal(f.ui.nodes.get('error').textContent,'');
    assert.equal(referenceNotices(f.ui).at(-1).message.channel,'save');assert.match(referenceNotices(f.ui).at(-1).message.message,/상태 다시 확인/);
    f.checkbox(2).change(true);assert.equal(f.writes().length,1);
    f.control.failReads=false;await f.ui.nodes.get('candidate-status-retry').handlers.click();await f.ui.flush();
    assert.equal(referenceNotices(f.ui).at(-1).message.message,'','recovery clears only its save alert');
    assert.equal(f.writes().length,1);assert.equal(f.checkbox(0).checked,true);assert.equal(f.checkbox(1).checked,false);
    assert.equal(f.ui.nodes.get('candidate-retry-actions').hidden,true);assert.equal(f.checkbox(2).disabled,false);
  }
  for(const reason of ['pagehide','pause']){
    const f=await toggleFixture();f.checkbox(0).change(true);f.checkbox(1).change(true);
    const readCount=f.ui.calls.filter(call=>!call.options).length;
    if(reason==='pagehide'){f.ui.messages.pagehide();f.ui.messages.pageshow();}
    else{
      f.ui.messages.message({origin:'http://127.0.0.1:8793',source:f.ui.parent,data:{type:'koww:pause'}});
      f.ui.messages.message({origin:'http://127.0.0.1:8793',source:f.ui.parent,data:{type:'koww:reference-status',project:'bora'}});
    }
    await f.ui.flush();assert.equal(f.ui.calls.filter(call=>!call.options).length,readCount,'navigation recovery waits for the already sent POST before GET');
    await f.finish();assert.equal(f.writes().length,1,'navigation discards unsent intents without pretending they were saved');
    assert.equal(f.ui.calls.filter(call=>!call.options).length,readCount+1);assert.equal(f.checkbox(0).checked,true);assert.equal(f.checkbox(1).checked,false);
    assert.ok(f.ui.nodes.get('error').textContent);assert.equal(f.ui.nodes.get('reference-group-next').disabled,true);
  }
  {
    const f=await toggleFixture();f.checkbox(0).change(true);f.checkbox(1).change(true);
    const different=structuredClone(initialCandidateState),nextGroup=different.groups[1].key;
    Object.assign(different,{active_group:nextGroup,items:candidates(nextGroup),accepted:[]});
    f.control.readPayload=different;await f.finish({reply:different});
    assert.equal(f.writes().length,1,'old-group intents cannot write into a new category');
    assert.equal(f.ui.nodes.get('reference-group-count').textContent,'0 / 2 선택');assert.ok(f.cards().every(card=>!card.children[3].children[0].checked));
  }
  let finishApproval,finishNextGeneration;
  const candidateRespond=async(url,options)=>{
    if(options){
      assert.equal(options.headers['X-Koww-Token'],'test-token');
      const action=JSON.parse(options.body),row=[...state.items,...state.accepted].find(item=>item.id===action.id);
      if(action.action==='hear'){assert.equal(row.sha256,action.sha256);row.heard=true;}
      if(action.action==='toggle'){
        assert.equal(row.sha256,action.sha256,'selection always refers to the original candidate revision');
        if(state.accepted.some(item=>item.id===row.id))state.accepted=state.accepted.filter(item=>item.id!==row.id);
        else state.accepted.push({...row,selected:true});
        row.selected=state.accepted.some(item=>item.id===row.id);
      }
      if(action.action==='more'){
        if(state.active_group===categories[1].key)await new Promise(resolve=>{finishNextGeneration=resolve;});
        state.job={status:'running',group:state.active_group,completed:0,total:5};
      }
      if(action.action==='approve'){
        const group=state.groups.find(item=>item.key===state.active_group);assert.equal(state.accepted.length,group.target);
        await new Promise(resolve=>{finishApproval=resolve;});
        group.approved=true;state.active_group=state.groups[state.groups.indexOf(group)+1].key;state.accepted=[];state.items=[];state.job=null;
      }
    }
    return {payload:state};
  };
  const candidate=await screen('reference-preview',[],{config:candidateConfig,respond:candidateRespond,
    candidateAreaHeight:nodes=>nodes.get('candidate-list').children.length?246:0});
  const rowList=candidate.nodes.get('candidate-list'),audioCandidate=candidate.nodes.get('audio');
  const candidateRequests=()=>candidate.calls.filter(call=>call.options).map(call=>JSON.parse(call.options.body));
  const headerVolume=candidate.nodes.get('candidate-volume'),candidateVolume=candidate.nodes.get('reference-volume'),candidateMute=candidate.document.getElementById('reference-mute');
  for(const id of ['voice','voice-picker','continuous','position','reference-playback','reference-voices','reference-group-actions','reference-replay','reference-replace','reference-regenerate','listen-page'])assert.equal(candidate.document.getElementById(id),null,`candidate mode never mounts ${id}`);
  assert.equal(candidate.enhancements.length,0,'candidate mode never creates a seek bar or time transport');assert.equal(candidate.pickerCreations.length,0,'candidate mode never initializes a list picker');
  assert.equal(audioCandidate.hidden,true);assert.equal(audioCandidate.controls,false);
  assert.equal(headerVolume.hidden,false);assert.deepEqual(headerVolume.children,[candidateMute,candidateVolume],'the candidate header contains only the existing mute and volume controls');
  candidateVolume.value='35';candidateVolume.handlers.input();assert.equal(audioCandidate.volume,.35);
  candidateMute.handlers.click();assert.equal(audioCandidate.muted,true);
  candidateVolume.value='70';candidateVolume.handlers.input();assert.equal(audioCandidate.volume,.7);assert.equal(audioCandidate.muted,false);
  assert.equal(audioCandidate.played,0);assert.equal(candidateRequests().length,0,'header volume changes neither play nor save candidate decisions');
  assert.equal(rowList.children.length,5);assert.equal(candidate.nodes.get('reference-group-name').textContent,'학습용 · 남성 어린이');
  assert.ok(rowList.children.every(card=>card.className.split(' ').includes('candidate-card')),'the current five sources are individual cards');
  assert.ok(rowList.children.every(card=>card.attributes.role==='group'),'each card has an accessible candidate group');
  state.items.forEach((item,index)=>assertCandidateMetadata(rowList.children[index].children[1],item));
  assert.match(nodeText(rowList.children[0].children[1]),/시드\s*[:·]?\s*0(?:\D|$)/,'a valid zero seed is never treated as missing');
  assert.equal(candidate.nodes.get('reference-group-count').textContent,'0 / 2 선택');
  assert.equal(candidate.nodes.get('reference-group-count').attributes['data-complete'],'false');
  assert.equal(candidate.nodes.get('reference-group-next').disabled,true);
  assert.equal(candidate.nodes.get('reference-group-nav').hidden,true,'embedded candidate review uses the chapter footer instead of another category navigation row');
  assert.equal(candidate.nodes.get('reference-status').hidden,true,'the redundant category counter is hidden');
  assert.equal(candidate.nodes.get('candidate-more').disabled,false,'zero selections can request another batch');
  assert.equal(candidate.nodes.get('candidate-more').hidden,false);assert.equal(candidate.nodes.get('candidate-job-icon').hidden,true);
  assert.equal(candidate.nodes.get('candidate-more-label').textContent,'+');
  assert.equal(candidate.nodes.get('candidate-more').attributes['aria-label'],'기준 목소리 후보 더 뽑기');
  assert.equal(rowList.children[0].children[3].children[0].disabled,false,'unheard candidates can be selected without playing audio');
  assert.equal(rowList.children[0].children[3].children[1].textContent,'선택');
  assert.equal(rowList.children[0].children[2].hidden,true);assert.equal(rowList.children[0].children[2].textContent,'','candidate cards have no separate white completion check');
  assert.equal(audioCandidate.played,0);assert.equal(candidateRequests().length,0,'opening an existing batch neither plays nor selects');
  assert.equal(candidate.layoutMessages[0].message.mode,'candidate_pool_v2');
  const groupReports=()=>candidate.parentMessages.filter(({message})=>message.type==='koww:reference-group').map(({message})=>message);
  const sendApproval=(overrides={})=>candidate.messages.message({origin:'http://127.0.0.1:8793',source:candidate.parent,
    data:{type:'koww:reference-approve',project:'bora',group:state.active_group},...overrides});
  assert.equal(groupReports().at(-1).ready,false);
  sendApproval();await candidate.flush();assert.equal(candidateRequests().length,0,'the chapter footer cannot approve a category before its quota is filled');
  const reportsBeforeStatus=groupReports().length;
  candidate.messages.message({origin:'http://127.0.0.1:8793',source:candidate.parent,data:{type:'koww:reference-status',project:'wrong-project'}});
  assert.equal(groupReports().length,reportsBeforeStatus);
  candidate.messages.message({origin:'http://127.0.0.1:8793',source:candidate.parent,data:{type:'koww:reference-status',project:'bora'}});
  assert.equal(groupReports().length,reportsBeforeStatus+1,'a parent frame reload can request fresh category readiness');
  async function hearCandidate(index){
    const row=rowList.children[index];row.children[0].handlers.click();
    assert.equal(row.attributes['data-playing'],'true','the currently playing candidate is highlighted');
    assert.equal(row.children[3].children[1].textContent,'선택','playback state never replaces the selection label');
    audioCandidate.duration=1;audioCandidate.played={length:1,start:()=>0,end:()=>1};audioCandidate.currentTime=1;
    audioCandidate.finish();await candidate.flush();
  }
  await hearCandidate(0);
  assert.equal(audioCandidate.playCalls,1,'finishing a candidate does not start the next candidate');
  assert.equal(candidateRequests()[0].action,'hear');assert.equal(state.accepted.length,0,'hearing never accepts a candidate automatically');
  const firstCheckbox=rowList.children[0].children[3].children[0];assert.equal(firstCheckbox.disabled,false);
  firstCheckbox.focus();firstCheckbox.change(true);await candidate.flush();
  assert.equal(rowList.children[0].children[3].children[0],firstCheckbox,'saving retains the same checkbox node');
  assert.equal(firstCheckbox.checked,true);assert.equal(rowList.children[0].children[2].hidden,true,'removing the completion icon does not remove the selected checkbox state');
  assert.equal(candidate.document.activeElement,firstCheckbox);assert.equal(candidate.nodes.get('reference-group-count').textContent,'1 / 2 선택');
  assert.equal(candidate.nodes.get('candidate-accepted-list'),undefined,'selected voices are counted without a separate visible tray');
  assert.equal(candidate.nodes.get('reference-group-count').attributes['data-complete'],'false');
  candidate.nodes.get('candidate-more').handlers.click();assertGeneratingStatus(candidate);assert.equal(rowList.children.length,5);await candidate.flush();
  assertGeneratingStatus(candidate);
  assert.equal(candidateRequests().at(-1).action,'more');assert.equal(candidate.nodes.get('candidate-more').disabled,true);
  assert.equal(candidate.nodes.get('candidate-job-icon').attributes['data-state'],'running');
  const unchangedRow=rowList.children[0],source=audioCandidate.src;audioCandidate.currentTime=.4;
  const volumeMoves=candidateVolume.parentMoves,muteMoves=candidateMute.parentMoves;
  state.job.completed=2;await candidate.poll();
  assert.equal(rowList.children[0],unchangedRow,'status polling preserves row nodes');
  assertCandidateMetadata(rowList.children[0].children[1],state.items[0]);
  assert.equal(audioCandidate.src,source);assert.equal(audioCandidate.currentTime,.4,'status polling does not reload the selected clip');
  assert.equal(candidate.document.activeElement,firstCheckbox);
  assert.equal(candidateVolume.parentMoves,volumeMoves);assert.equal(candidateMute.parentMoves,muteMoves,'status polling never detaches the header volume controls');
  assert.equal(candidate.nodes.get('candidate-progress').attributes['aria-valuenow'],'2');
  assert.equal(candidate.layoutMessages.length,1,'a status poll does not repeatedly resize the parent');
  state.items=candidates(state.active_group,6);state.job=null;await candidate.poll();
  assert.equal(candidate.nodes.get('candidate-more').hidden,false);assert.equal(candidate.nodes.get('candidate-job-icon').hidden,true);assert.equal(candidate.nodes.get('candidate-empty').hidden,true);
  assert.equal(rowList.children.length,5);assertCandidateMetadata(rowList.children[0].children[1],state.items[0],1);
  assert.equal(candidate.nodes.get('reference-group-count').textContent,'1 / 2 선택','accepted candidates survive a new batch');
  assert.equal(state.accepted[0].number,1,'an earlier batch selection is retained in the total without rendering a separate list');
  await hearCandidate(0);
  rowList.children[0].children[3].children[0].change(true);await candidate.flush();
  assert.equal(state.accepted.length,2);assert.equal(candidate.nodes.get('reference-group-next').disabled,false);
  assert.equal(candidate.nodes.get('reference-group-count').attributes['data-complete'],'true');
  assert.match(candidate.nodes.get('reference-group-count').attributes['aria-label'],/선택 완료/);
  assert.doesNotMatch(candidate.nodes.get('reference-group-count').attributes['aria-label'],/검수 완료|확정/);
  assert.equal(candidateRequests().filter(action=>action.action==='approve').length,0,'a full count is not an approval');
  assert.equal(groupReports().at(-1).ready,true,'only an exact selected quota enables the chapter next button');
  assert.equal(candidate.nodes.get('candidate-more').disabled,true,'a full category cannot create an unneeded batch');
  await hearCandidate(1);
  assert.equal(rowList.children[1].children[3].children[0].disabled,true,'the remaining quota is enforced within the batch');
  const selectedCardCheckbox=rowList.children[0].children[3].children[0];
  selectedCardCheckbox.change(false);await candidate.flush();assert.equal(state.accepted.length,1,'selected cards can be unselected without a separate tray');
  assert.equal(candidate.nodes.get('reference-group-count').attributes['data-complete'],'false');
  rowList.children[1].children[3].children[0].change(true);await candidate.flush();assert.equal(state.accepted.length,2);
  const writesBeforeApproval=candidateRequests().length;
  for(const invalid of [
    {origin:'https://other.invalid'},{source:{}},
    {data:{type:'koww:reference-approve',project:'another-project',group:state.active_group}},
    {data:{type:'koww:reference-approve',project:'bora',group:'wrong-group'}},
  ])sendApproval(invalid);
  await candidate.flush();assert.equal(candidateRequests().length,writesBeforeApproval,'only the exact parent, project and active group can approve');
  const approvedCandidateGroup=state.active_group;
  const deckDuringApproval=candidate.nodes.get('candidate-deck'),panelDuringApproval=candidate.nodes.get('candidate-panel');
  const deckVisibilityChanges=[];
  let deckHidden=deckDuringApproval.hidden;
  Object.defineProperty(deckDuringApproval,'hidden',{
    get:()=>deckHidden,
    set:hidden=>{if(hidden!==deckHidden)deckVisibilityChanges.push(hidden);deckHidden=hidden;},
  });
  sendApproval({data:{type:'koww:reference-approve',project:'bora',group:approvedCandidateGroup,request:17}});await candidate.flush();
  assert.equal(state.groups[0].approved,false,'the approval response is still pending');
  assert.equal(deckDuringApproval.hidden,true,'approving replaces the old cards instead of dimming their controls');
  assert.equal(deckDuringApproval.inert,true,'the hidden controls cannot accept keyboard input during approval');
  assert.equal(candidate.nodes.get('candidate-empty').hidden,false,'the approval shows a status while the server saves');
  assert.equal(candidate.nodes.get('candidate-status-label').textContent,'선택한 기준 목소리를 확정하고 있습니다...');
  assert.equal(candidate.nodes.get('candidate-job-icon').attributes['data-state'],'running');
  assert.equal(candidate.nodes.get('candidate-progress-slot').hidden,false,'approval reserves the progress row for a stable handoff');
  assert.equal(candidate.nodes.get('candidate-progress').hidden,true,'approval does not show invented progress');
  assert.equal(panelDuringApproval.style.getPropertyValue('--candidate-work-height'),'246px');
  finishApproval();await candidate.flush();
  assert.equal(state.groups[0].approved,true);assert.equal(state.active_group,categories[1].key);
  assert.equal(candidate.nodes.get('reference-group-name').textContent,'학습용 · 남성 청소년');
  assert.equal(candidateRequests().at(-1).action,'more','the newly confirmed next category starts its first batch once');
  assert.equal(candidateRequests().filter(action=>action.action==='approve').length,1);
  assertGeneratingStatus(candidate);
  assert.deepEqual(deckVisibilityChanges,[true],'the empty category never shows its deck between approval and generation');
  assert.equal(panelDuringApproval.style.getPropertyValue('--candidate-work-height'),'246px','the handoff retains the old deck height');
  assert.deepEqual(candidate.parentMessages.filter(({message})=>message.type==='koww:reference-approved').at(-1).message,
    {type:'koww:reference-approved',project:'bora',group:approvedCandidateGroup,request:17,success:true,complete:false},
    'approval acknowledgement is bound to its original group and request after the next group is rendered');
  assert.equal(groupReports().at(-1).group,categories[1].key);assert.equal(groupReports().at(-1).ready,false);
  finishNextGeneration();await candidate.flush();
  assert.deepEqual(deckVisibilityChanges,[true],'the deck stays hidden while the first batch is running');
  state.items=candidates(state.active_group);state.job=null;await candidate.poll();
  assert.deepEqual(deckVisibilityChanges,[true,false],'the deck returns once with the next group’s actual cards');
  assert.equal(deckDuringApproval.inert,false);
  assert.equal(rowList.children.length,5);
  assert.equal(candidate.nodes.get('candidate-empty').hidden,true);
  const emptyState={...structuredClone(state),items:[],accepted:[],job:null};
  let initialMore=0;
  const initial=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
    if(options){initialMore++;emptyState.job={status:'failed',message:'생성 실패',group:emptyState.active_group};}
    return {payload:emptyState};
  }});
  assert.equal(initialMore,1,'an empty category requests its initial batch exactly once');
  assert.equal(initial.nodes.get('error').textContent,'');assert.equal(initial.timers.size,0,'a failed job does not loop or auto-retry');
  assert.deepEqual(referenceNotices(initial).at(-1),{origin:'http://127.0.0.1:8793',message:{type:'koww:reference-notice',
    project:'bora',url:'/reference-preview/?project=bora',channel:'generation',message:'생성 실패'}});
  const initialNoticeCount=referenceNotices(initial).length;
  initial.messages.message({origin:'http://127.0.0.1:8793',source:initial.parent,data:{type:'koww:reference-status',project:'bora'}});
  assert.equal(referenceNotices(initial).length,initialNoticeCount+1,'the parent can recover a notice sent before its frame load event');
  assert.equal(initial.nodes.get('candidate-more').disabled,false,'a failure leaves an explicit retry available');
  initial.nodes.get('candidate-more').handlers.click();await initial.flush();assert.equal(initialMore,2);
  const preparingState={...structuredClone(initialCandidateState),items:[],accepted:[],editable:false,
    job:{status:'running',phase:'preparing',completed:0,total:5,message:'목소리 분류를 준비하고 있습니다.'}};
  const preparing=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
    if(options)preparingState.job={status:'running',group:preparingState.active_group,completed:0,total:5};
    return {payload:preparingState};
  }});
  assert.equal(preparing.calls.length,1,'opening the merged chapter only polls while classification metadata is preparing');
  assert.equal(preparing.nodes.get('candidate-more').disabled,true);
  assert.equal(preparing.nodes.get('candidate-more-label').hidden,true);
  assert.equal(preparing.nodes.get('candidate-more').attributes['aria-label'],'목소리 분류 준비 중');
  assertGeneratingStatus(preparing);
  assert.equal(preparing.nodes.get('candidate-progress').hidden,false,'classification preparation shows the gauge immediately without a synthetic candidate percentage');
  assert.equal(preparing.nodes.get('candidate-progress').attributes['data-indeterminate'],'true');
  assert.equal(preparing.nodes.get('candidate-progress').getAttribute('aria-valuenow'),null);
  assert.equal(preparing.nodes.get('candidate-progress').attributes['aria-valuetext'],'목소리 분류 준비 중');
  assert.equal(preparing.nodes.get('candidate-progress-fill').style.width,'25%');
  assert.equal(preparing.nodes.get('candidate-progress-slot').hidden,false,'classification preparation reserves the future progress row without inventing a percentage');
  assert.equal(preparing.nodes.get('candidate-job-icon').attributes['data-state'],'running');
  assert.equal(preparing.nodes.get('candidate-previous').hidden,true);assert.equal(preparing.nodes.get('candidate-previous').disabled,true);
  preparingState.editable=true;preparingState.job=null;await preparing.poll();await preparing.flush();
  assert.equal(preparing.calls.filter(call=>call.options).length,1,'the first candidate batch starts once after preparation becomes editable');
  assert.equal(JSON.parse(preparing.calls.find(call=>call.options).options.body).action,'more');
  assert.equal(preparing.nodes.get('candidate-progress').hidden,false,'the real candidate generation phase shows its own progress');
  assert.equal(preparing.nodes.get('candidate-progress-slot').hidden,false,'the same reserved row remains when real candidate progress becomes available');
  let pollAvailable=false,pollReads=0;
  const interruptedState={...structuredClone(initialCandidateState),job:{status:'running',group:categories[0].key,completed:1,total:5}};
  const interrupted=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
    assert.equal(options,undefined,'checking an existing job never starts another GPU task');
    pollReads++;if(pollReads>1&&!pollAvailable)return {ok:false,payload:{error:'상태 연결 끊김'}};
    return {payload:interruptedState};
  }});
  await interrupted.poll();await interrupted.poll();assert.deepEqual(referenceNotices(interrupted),[],'short-lived polling errors do not surface');
  await interrupted.poll();
  assert.equal(interrupted.timers.size,0,'repeated queue failures stop polling instead of retrying forever');
  assert.equal(interrupted.nodes.get('candidate-retry-actions').hidden,false);
  assert.equal(interrupted.nodes.get('error').textContent,'');
  assert.equal(referenceNotices(interrupted).at(-1).message.channel,'load');
  assert.equal(referenceNotices(interrupted).at(-1).message.message,'상태 연결 끊김');
  pollAvailable=true;interrupted.nodes.get('candidate-status-retry').handlers.click();await interrupted.flush();
  assert.equal(referenceNotices(interrupted).at(-1).message.message,'');
  assert.equal(interrupted.nodes.get('candidate-retry-actions').hidden,true);assert.equal(interrupted.timers.size,1,'manual status retry resumes the existing job');
  const incomplete=await screen('reference-preview',[],{config:candidateConfig,payload:initialCandidateState});
  const incompleteAudio=incomplete.nodes.get('audio');incomplete.nodes.get('candidate-list').children[0].children[0].handlers.click();
  incompleteAudio.duration=1;incompleteAudio.played={length:1,start:()=>.5,end:()=>1};incompleteAudio.finish();await incomplete.flush();
  assert.equal(incomplete.calls.length,1,'seeking to the end cannot mark a candidate heard');
  assert.equal(incomplete.nodes.get('candidate-list').children[0].children[3].children[0].disabled,false,'partial listening does not block explicit selection');
  assert.equal(incomplete.nodes.get('error').textContent,'처음부터 끝까지 들은 경우에만 청취 완료로 기록합니다.');
  assert.deepEqual(referenceNotices(incomplete),[],'a skipped-interval hint remains beside its audio');
  const failedCandidateAudio=await screen('reference-preview',[],{config:candidateConfig,payload:initialCandidateState,
    playback:count=>count===1?Promise.reject(Error('device failed')):Promise.resolve()});
  failedCandidateAudio.nodes.get('candidate-list').children[0].children[0].handlers.click();await failedCandidateAudio.flush();
  assert.equal(failedCandidateAudio.nodes.get('error').textContent,'');
  assert.equal(referenceNotices(failedCandidateAudio).at(-1).message.channel,'playback');
  assert.match(referenceNotices(failedCandidateAudio).at(-1).message.message,/재생하지 못했습니다/);
  failedCandidateAudio.nodes.get('candidate-list').children[0].children[0].handlers.click();await failedCandidateAudio.flush();
  failedCandidateAudio.nodes.get('audio').handlers.playing();
  assert.equal(referenceNotices(failedCandidateAudio).at(-1).message.message,'');
  const standaloneCandidate=await screen('reference-preview',[],{config:candidateConfig,payload:initialCandidateState,standalone:true,
    playback:()=>Promise.reject(Error('device failed'))});
  standaloneCandidate.nodes.get('candidate-list').children[0].children[0].handlers.click();await standaloneCandidate.flush();
  assert.match(standaloneCandidate.nodes.get('error').textContent,/재생하지 못했습니다/);
  assert.deepEqual(referenceNotices(standaloneCandidate),[]);
  const endedRaceState=structuredClone(initialCandidateState);
  const endedRace=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
    if(options){const action=JSON.parse(options.body);assert.equal(action.action,'hear');endedRaceState.items.find(item=>item.id===action.id).heard=true;}
    return {payload:endedRaceState};
  }});
  const raceAudio=endedRace.nodes.get('audio'),raceCard=endedRace.nodes.get('candidate-list').children[0];
  raceCard.children[0].handlers.click();raceAudio.duration=1;raceAudio.currentTime=1;raceAudio.played={length:1,start:()=>0,end:()=>1};
  raceAudio.paused=true;raceAudio.ended=false;raceAudio.handlers.pause();raceAudio.ended=true;raceAudio.handlers.ended();await endedRace.flush();
  assert.equal(endedRace.calls.filter(call=>call.options).length,1,'a natural pause event just before ended cannot drop a fully heard candidate');
  assert.equal(raceCard.children[3].children[0].disabled,false);assert.equal(raceCard.children[3].children[1].textContent,'선택');
  assert.equal(raceCard.children[2].hidden,true);assert.equal(raceCard.children[2].textContent,'');
  assert.equal(endedRaceState.accepted.length,0);assert.equal(raceAudio.playCalls,1,'natural completion never selects or relays');
  {
    const pendingHearState=structuredClone(initialCandidateState);
    const [heardRow,oldChoice,newChoice]=pendingHearState.items;
    oldChoice.selected=true;pendingHearState.accepted=[oldChoice];
    let releaseHear;
    const pendingHear=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
      if(!options)return {payload:pendingHearState};
      const action=JSON.parse(options.body);
      const row=pendingHearState.items.find(item=>item.id===action.id&&item.sha256===action.sha256);
      assert.ok(row);
      if(action.action==='hear'){
        assert.equal(row,heardRow);
        row.heard=true;
        // Freeze the server reply before either subsequent selection is saved.
        const reply=structuredClone(pendingHearState);
        await new Promise(resolve=>{releaseHear=resolve;});
        return {payload:reply};
      }
      assert.equal(action.action,'toggle');
      row.selected=!row.selected;
      pendingHearState.accepted=pendingHearState.items.filter(item=>item.selected);
      return {payload:pendingHearState};
    }});
    const cards=pendingHear.nodes.get('candidate-list').children;
    const choice=index=>cards[index].children[3].children[0];
    const audio=pendingHear.nodes.get('audio');
    function finishFirstClip(){
      cards[0].children[0].handlers.click();
      audio.duration=1;audio.currentTime=1;audio.played={length:1,start:()=>0,end:()=>1};
      audio.finish();
    }
    finishFirstClip();
    assert.ok(cards.every((_,index)=>!choice(index).disabled),'natural playback completion never disables candidate selections');
    await pendingHear.flush();
    assert.equal(typeof releaseHear,'function','a completed clip still saves a factual listening record');
    assert.ok(cards.every((_,index)=>!choice(index).disabled),'candidate selections remain available throughout the pending listening save');
    choice(1).change(false);choice(2).change(true);
    assert.equal(choice(1).checked,false,'deselection is immediately visible during the listening save');
    assert.equal(choice(2).checked,true,'selection is immediately visible during the listening save');
    await pendingHear.flush();
    releaseHear();await pendingHear.flush();
    assert.deepEqual(pendingHearState.accepted.map(item=>item.id),[newChoice.id]);
    assert.equal(choice(1).checked,false,'the earlier listening reply cannot restore a deselected candidate');
    assert.equal(choice(2).checked,true,'the earlier listening reply cannot discard the newer choice');
    assert.equal(pendingHear.nodes.get('candidate-list').children[2],cards[2],'saving hearing and selection preserves the candidate cards');
    const writes=()=>pendingHear.calls.filter(call=>call.options).map(call=>JSON.parse(call.options.body));
    assert.deepEqual(writes().filter(action=>action.action==='toggle').map(action=>action.id),[oldChoice.id,newChoice.id]);
    finishFirstClip();await pendingHear.flush();
    assert.equal(writes().filter(action=>action.action==='hear').length,1,'the acknowledged listening record is retained after the selection changes');
  }
  const stoppedRace=await screen('reference-preview',[],{config:candidateConfig,payload:initialCandidateState});
  const stoppedAudio=stoppedRace.nodes.get('audio'),stoppedPlay=stoppedRace.nodes.get('candidate-list').children[0].children[0];
  stoppedPlay.handlers.click();stoppedAudio.duration=1;stoppedAudio.currentTime=1;stoppedAudio.played={length:1,start:()=>0,end:()=>1};
  stoppedPlay.handlers.click();stoppedAudio.ended=true;stoppedAudio.handlers.ended();await stoppedRace.flush();
  assert.equal(stoppedRace.calls.length,1,'explicit user stop invalidates completion intent even if a stale ended event arrives');
  const approvalState=structuredClone(initialCandidateState);
  approvalState.items.slice(0,2).forEach(item=>{item.heard=true;item.selected=true;});approvalState.accepted=approvalState.items.slice(0,2);
  const failedApproval=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>options?
    {ok:false,payload:{error:'검수 저장 실패'}}:{payload:approvalState}});
  const cardsBeforeFailure=[...failedApproval.nodes.get('candidate-list').children];
  failedApproval.messages.message({origin:'http://127.0.0.1:8793',source:failedApproval.parent,
    data:{type:'koww:reference-approve',project:'bora',group:approvalState.active_group,request:19}});await failedApproval.flush();
  const failureReports=failedApproval.parentMessages.filter(({message})=>message.type==='koww:reference-group').map(({message})=>message.ready);
  assert.deepEqual(failureReports.slice(-2),[false,true],'a rejected approval restores explicit next-button retry instead of leaving it pending');
  assert.equal(failedApproval.nodes.get('candidate-deck').hidden,false);
  assert.deepEqual(failedApproval.nodes.get('candidate-list').children,cardsBeforeFailure,'a rejected approval restores the same selected cards');
  assert.equal(failedApproval.nodes.get('error').textContent,'');
  assert.equal(referenceNotices(failedApproval).at(-1).message.channel,'save');
  assert.equal(referenceNotices(failedApproval).at(-1).message.message,'검수 저장 실패');
  assert.deepEqual(failedApproval.parentMessages.filter(({message})=>message.type==='koww:reference-approved').at(-1).message,
    {type:'koww:reference-approved',project:'bora',group:approvalState.active_group,request:19,success:false,complete:false});
  const finalApprovalState=structuredClone(approvalState);
  finalApprovalState.groups=finalApprovalState.groups.slice(0,1);
  const finalApproval=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
    if(options){assert.equal(JSON.parse(options.body).action,'approve');finalApprovalState.review_complete=true;finalApprovalState.groups[0].approved=true;}
    return {payload:finalApprovalState};
  }});
  finalApproval.messages.message({origin:'http://127.0.0.1:8793',source:finalApproval.parent,
    data:{type:'koww:reference-approve',project:'bora',group:finalApprovalState.active_group,request:23}});await finalApproval.flush();
  assert.deepEqual(finalApproval.parentMessages.filter(({message})=>message.type==='koww:reference-approved').at(-1).message,
    {type:'koww:reference-approved',project:'bora',group:finalApprovalState.active_group,request:23,success:true,complete:true},
    'only a successful save acknowledges completion for the same approval request');
  const directCandidate=await screen('reference-preview',[],{config:candidateConfig,payload:approvalState,standalone:true});
  assert.equal(directCandidate.nodes.get('reference-group-nav').hidden,false,'a directly opened candidate page still has a usable next control');
  assert.equal(directCandidate.nodes.get('reference-group-next').textContent,'다음');
  directCandidate.nodes.get('reference-group-next').handlers.click();await directCandidate.flush();
  assert.equal(JSON.parse(directCandidate.calls.find(call=>call.options).options.body).action,'approve');
  const legacyGrouped=await screen('reference-preview',[],{config:candidateConfig,payload:{
    items:[{id:'legacy-1',sha256:'legacy-hash',audio:'/reference-preview/audio/0?project=bora',text:'기준 문장',group:'학습용',group_key:'legacy',voice_number:1,heard:true}],
    groups:[{key:'legacy',label:'학습용 · 남성 · 어린이',approved:false}],editable:true,review_complete:false}});
  assert.equal(legacyGrouped.nodes.get('reference-group-nav').hidden,false,'legacy grouped review keeps its original category navigation');
  assert.equal(legacyGrouped.nodes.get('reference-playback').hidden,false);
  assert.equal(legacyGrouped.nodes.get('reference-group-count').attributes['data-complete'],undefined,'legacy group counters retain their original meaning');
  const historyState=structuredClone(initialCandidateState),historyBatches=new Map([
    ['batch-one',candidates(historyState.active_group)],['batch-two',candidates(historyState.active_group,6)],
  ]);
  let latestHistoryBatch='batch-two',historyJob=null,gpuRequests=0;
  historyBatches.get('batch-two')[0].heard=true;historyState.accepted=[{...historyBatches.get('batch-two')[0],selected:true}];
  const historyPayload=view=>({...historyState,items:historyBatches.get(view)||[],job:historyJob,
    batches:[...historyBatches].map(([id,rows],index)=>({id,number:index+1,count:rows.length})),current_batch:latestHistoryBatch,view_batch:view});
  const historyScreen=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
    const request=new URL(url,'http://127.0.0.1:8793'),context=request.searchParams.get('batch');
    if(options){
      const action=JSON.parse(options.body),row=[...historyBatches.values()].flat().find(item=>item.id===action.id);
      if(action.action==='hear'){assert.ok(context);row.heard=true;}
      if(action.action==='toggle'){
        assert.ok(context,'selection writes carry only response-view context');
        if(historyState.accepted.some(item=>item.id===row.id))historyState.accepted=historyState.accepted.filter(item=>item.id!==row.id);
        else historyState.accepted.push({...row,selected:true});
      }
      if(action.action==='more'){
        assert.equal(context,null,'a new batch never carries an old history-view parameter');gpuRequests++;
        latestHistoryBatch=`batch-new-${gpuRequests}`;historyJob={status:'running',group:historyState.active_group,completed:0,total:5};
      }
    }
    const payload=historyPayload(context||latestHistoryBatch);
    payload.items=payload.items.map(item=>({...item,selected:historyState.accepted.some(value=>value.id===item.id)}));
    return {payload};
  }});
  const historyRows=()=>historyScreen.nodes.get('candidate-list').children;
  const historyWrites=()=>historyScreen.calls.filter(call=>call.options).map(call=>JSON.parse(call.options.body));
  assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 6');
  assert.equal(historyScreen.nodes.get('candidate-previous').disabled,false);assert.equal(historyScreen.nodes.get('candidate-more-label').textContent,'+');
  historyRows()[0].children[0].handlers.click();assert.equal(historyScreen.nodes.get('audio').paused,false);
  const historySnapshot=JSON.stringify(historyState.accepted);
  historyScreen.nodes.get('candidate-previous').handlers.click();await historyScreen.flush();
  assert.equal(historyScreen.nodes.get('audio').paused,true,'history navigation explicitly pauses the previous recording');
  assert.equal(historyWrites().length,0);assert.equal(gpuRequests,0,'browsing old candidates never starts a generation');
  assert.equal(JSON.stringify(historyState.accepted),historySnapshot);assert.equal(historyScreen.nodes.get('candidate-previous').disabled,true);
  assert.equal(historyScreen.nodes.get('candidate-more-label').textContent,'›');assert.equal(historyScreen.nodes.get('candidate-more').attributes['aria-label'],'다음 후보 5개');
  historyRows()[0].children[0].handlers.click();const historyAudio=historyScreen.nodes.get('audio');
  historyAudio.duration=1;historyAudio.currentTime=1;historyAudio.played={length:1,start:()=>0,end:()=>1};historyAudio.finish();
  assert.equal(historyScreen.nodes.get('candidate-previous').hidden,false,'hearing-status saves do not flicker the history arrow');await historyScreen.flush();
  assert.equal(new URL(historyScreen.calls.find(call=>call.options).url,'http://127.0.0.1:8793').searchParams.get('batch'),'batch-one');
  historyRows()[0].children[3].children[0].change(true);assert.equal(historyScreen.nodes.get('candidate-previous').hidden,false,'selection saves do not flicker the history arrow');await historyScreen.flush();
  assert.equal(historyState.accepted.length,2);assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 1','selecting an old candidate preserves its viewed batch');
  assert.equal(historyScreen.nodes.get('candidate-more').disabled,false,'a full quota still allows browsing already generated later candidates');
  const writesBeforeForward=historyWrites().length;historyScreen.nodes.get('candidate-more').handlers.click();await historyScreen.flush();
  assert.equal(historyWrites().length,writesBeforeForward);assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 6');
  assert.equal(historyScreen.nodes.get('candidate-more').disabled,true,'only the latest plus is constrained by the generation quota');
  historyScreen.nodes.get('candidate-previous').handlers.click();await historyScreen.flush();
  assert.equal(historyRows()[0].children[3].children[0].checked,true,'returning to an earlier batch restores its selected checkbox');
  historyRows()[0].children[3].children[0].change(false);await historyScreen.flush();
  assert.equal(historyState.accepted.length,1);assert.equal(historyRows()[0].children[3].children[0].checked,false,'an earlier selection can be removed through its history card');
  assert.equal(historyScreen.nodes.get('reference-group-count').attributes['data-complete'],'false');
  historyScreen.nodes.get('candidate-more').handlers.click();await historyScreen.flush();
  assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 6');
  historyScreen.nodes.get('candidate-more').handlers.click();assert.equal(historyScreen.nodes.get('candidate-previous').hidden,true,'more hides the previous arrow immediately, before its POST returns');
  assert.equal(historyScreen.nodes.get('candidate-previous').disabled,true);assert.equal(historyScreen.nodes.get('candidate-previous-actions').hidden,false,'the fixed previous-arrow column stays reserved');
  assert.equal(historyScreen.nodes.get('candidate-panel').attributes['aria-busy'],'true');await historyScreen.flush();assert.equal(gpuRequests,1);
  historyBatches.set(latestHistoryBatch,candidates(historyState.active_group,11));historyJob=null;await historyScreen.poll();
  assert.equal(historyScreen.nodes.get('candidate-previous').hidden,false,'completed generation restores history navigation');
  assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 11','a new generation follows its latest completed batch');
  historyScreen.nodes.get('candidate-more').handlers.click();await historyScreen.flush();assert.equal(gpuRequests,2);
  const callsDuringGeneration=historyScreen.calls.length;historyScreen.nodes.get('candidate-previous').handlers.click();await historyScreen.flush();
  assert.equal(historyScreen.calls.length,callsDuringGeneration,'a hidden previous arrow cannot be invoked while generating');
  historyBatches.set(latestHistoryBatch,candidates(historyState.active_group,16));historyJob=null;await historyScreen.poll();
  assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 16');
  historyScreen.nodes.get('candidate-previous').handlers.click();await historyScreen.flush();
  assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 11','completed history remains available after the temporary generation lock');
  assert.equal(historyScreen.nodes.get('candidate-more-label').textContent,'›');
  historyScreen.nodes.get('candidate-more').handlers.click();await historyScreen.flush();assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 16');
  assert.equal(gpuRequests,2,'all previous/next history moves are read-only GETs');
  historyScreen.nodes.get('candidate-more').handlers.click();await historyScreen.flush();assert.equal(gpuRequests,3);
  historyJob={status:'failed',group:historyState.active_group,message:'후보 생성 실패'};await historyScreen.poll();
  assert.equal(historyScreen.nodes.get('candidate-progress-slot').hidden,true,'a failed running job removes its reserved progress row');
  assert.equal(historyScreen.nodes.get('candidate-more').hidden,false);assert.equal(historyScreen.nodes.get('candidate-job-icon').hidden,true,'a failed job restores the action instead of leaving a progress spinner');
  assert.equal(historyScreen.nodes.get('candidate-previous').hidden,false);assert.equal(historyScreen.nodes.get('candidate-previous').disabled,false,'a failed pending batch does not permanently hide completed history');
  historyScreen.nodes.get('candidate-previous').handlers.click();await historyScreen.flush();assert.equal(historyRows()[0].children[1].children[0].children[0].textContent,'후보 16');
  for(const failure of ['http','throw']){
    let finishMore;
    const failedMore=await screen('reference-preview',[],{config:candidateConfig,candidateAreaHeight:246,respond:async(url,options)=>{
      if(options)return await new Promise((resolve,reject)=>{finishMore=()=>failure==='http'?resolve({ok:false,payload:{error:'요청 실패'}}):reject(new Error('연결 실패'));});
      return {payload:{...initialCandidateState,batches:[{id:'old',number:1,count:5},{id:'current',number:2,count:5}],current_batch:'current',view_batch:'current',
        job:{status:'completed',group:initialCandidateState.active_group,completed:5,total:5}}};
    }});
    const previous=failedMore.nodes.get('candidate-previous');assert.equal(previous.disabled,false);
    assert.equal(failedMore.nodes.get('candidate-progress-slot').hidden,true);
    const originalCards=[...failedMore.nodes.get('candidate-list').children];
    failedMore.nodes.get('candidate-more').handlers.click();assert.equal(previous.hidden,true);assert.equal(previous.disabled,true);
    assert.equal(failedMore.nodes.get('candidate-deck').hidden,true,'pending POST atomically replaces cards instead of adding a status row below them');
    assert.equal(failedMore.nodes.get('candidate-deck').inert,true);assert.equal(failedMore.nodes.get('candidate-deck').attributes['aria-hidden'],'true');
    assert.equal(failedMore.nodes.get('candidate-panel').style.getPropertyValue('--candidate-work-height'),'246px');
    assert.equal(failedMore.nodes.get('candidate-progress-slot').hidden,false,'the progress row is reserved synchronously before the More request returns');
    assert.equal(failedMore.nodes.get('candidate-progress').hidden,false,'the gauge is visible before the delayed POST has returned');
    assert.equal(failedMore.nodes.get('candidate-progress').attributes['data-indeterminate'],'true');
    assert.equal(failedMore.nodes.get('candidate-progress').getAttribute('aria-valuenow'),null,'an old completed 5/5 job never becomes a new request\'s fake 100%');
    assert.equal(failedMore.nodes.get('candidate-progress').attributes['aria-valuetext'],'생성 요청 처리 중');
    assert.equal(failedMore.nodes.get('candidate-progress-fill').style.width,'25%','the immediate pending gauge has a visible moving segment, not an empty 0% fill');
    assertGeneratingStatus(failedMore);
    assert.equal(failedMore.nodes.get('candidate-job-icon').attributes['data-state'],'running');
    finishMore();await failedMore.flush();assert.equal(previous.hidden,false);assert.equal(previous.disabled,false);
    assert.equal(failedMore.nodes.get('candidate-more').hidden,false);assert.equal(failedMore.nodes.get('candidate-job-icon').hidden,true);
    assert.equal(failedMore.nodes.get('candidate-panel').attributes['aria-busy'],'false');
    assert.equal(failedMore.nodes.get('candidate-progress-slot').hidden,true,'both HTTP and network failures restore the original card layout without a blank progress row');
    assert.equal(failedMore.nodes.get('candidate-progress').hidden,true);
    assert.equal(failedMore.nodes.get('candidate-deck').hidden,false);assert.equal(failedMore.nodes.get('candidate-deck').inert,false);
    assert.deepEqual(failedMore.nodes.get('candidate-list').children,originalCards,'a rejected POST restores the original card instances and their selection state');
    assert.equal(failedMore.nodes.get('candidate-panel').style.getPropertyValue('--candidate-work-height'),'');
  }
  {
    const atomic=structuredClone(initialCandidateState);atomic.items[0].heard=true;atomic.accepted=[{...atomic.items[0],selected:true}];
    let phase='cards',finishPost;
    const response=()=>({payload:{...atomic,items:phase==='cards'?atomic.items:phase==='done'?candidates(atomic.active_group,6):[],
      job:['queued','preparing','running'].includes(phase)?{status:phase==='preparing'?'running':phase,...(phase==='preparing'?{phase:'preparing'}:{}),group:atomic.active_group,completed:phase==='running'?2:0,total:5}:null}});
    const atomicScreen=await screen('reference-preview',[],{config:candidateConfig,
      candidateAreaHeight:nodes=>nodes.get('candidate-list').children.length?246:0,
      workspaceHeight:nodes=>{
        const panel=nodes.get('candidate-panel'),busy=panel.attributes['aria-busy']==='true';
        const content=busy?parseFloat(panel.style.getPropertyValue('--candidate-work-height'))||232:
          (nodes.get('candidate-list').children.length?246:0)+(nodes.get('candidate-empty').hidden?0:46);
        return 40+content;
      },respond:async(url,options)=>{
        if(options){assert.equal(JSON.parse(options.body).action,'more');return await new Promise(resolve=>{finishPost=()=>{phase='queued';resolve(response());};});}
        return response();
      }});
    const deck=atomicScreen.nodes.get('candidate-deck'),panel=atomicScreen.nodes.get('candidate-panel');
    const slot=atomicScreen.nodes.get('candidate-progress-slot'),bar=atomicScreen.nodes.get('candidate-progress');
    assert.equal(bar.parentElement,slot);assert.deepEqual(slot.children,[bar]);
    assert.equal(slot.hidden,true,'the normal card deck has no reserved progress row');
    assert.equal(deck.hidden,false);assert.equal(atomicScreen.heightMessages.at(-1).message.height,286);
    atomicScreen.nodes.get('candidate-more').handlers.click();
    assertGeneratingStatus(atomicScreen);
    assert.equal(bar.hidden,false);assert.equal(bar.attributes['data-indeterminate'],'true');
    assert.equal(bar.getAttribute('aria-valuenow'),null);assert.equal(bar.attributes['aria-valuetext'],'생성 요청 처리 중');
    assert.equal(atomicScreen.nodes.get('candidate-progress-fill').style.width,'25%');
    await atomicScreen.flush();
    assert.equal(deck.hidden,true);assert.equal(panel.attributes['aria-busy'],'true');assertGeneratingStatus(atomicScreen);
    assert.equal(slot.hidden,false);assert.equal(bar.hidden,false,'pending POST shows the same gauge while its measured progress is unknown');
    assert.equal(atomicScreen.nodes.get('candidate-list').children.length,5,'pending POST keeps existing cards intact for failure recovery');
    assert.equal(atomicScreen.nodes.get('candidate-list').children[0].children[3].children[0].checked,true);
    assert.equal(panel.style.getPropertyValue('--candidate-work-height'),'246px');
    finishPost();await atomicScreen.flush();
    assert.equal(deck.hidden,true);assert.equal(atomicScreen.nodes.get('candidate-list').children.length,0);
    assert.equal(slot.hidden,false);assert.equal(bar.hidden,false,'queued progress appears inside the row already reserved by the pending POST');
    assert.equal(bar.attributes['data-indeterminate'],'false');assert.equal(bar.getAttribute('aria-valuenow'),'0');
    assert.equal(bar.getAttribute('aria-valuetext'),null);assert.equal(atomicScreen.nodes.get('candidate-progress-fill').style.width,'0%');
    assert.equal(panel.style.getPropertyValue('--candidate-work-height'),'246px','queued empty state retains the same card-area height');
    phase='preparing';await atomicScreen.poll();await atomicScreen.flush();
    assert.equal(slot.hidden,false);assert.equal(bar.hidden,false,'a metadata-preparation phase keeps the visible gauge while withholding candidate percentages');
    assert.equal(bar.attributes['data-indeterminate'],'true');assert.equal(bar.getAttribute('aria-valuenow'),null);
    assert.equal(bar.attributes['aria-valuetext'],'목소리 분류 준비 중');assert.equal(atomicScreen.nodes.get('candidate-progress-fill').style.width,'25%');
    assert.equal(panel.style.getPropertyValue('--candidate-work-height'),'246px');
    phase='running';await atomicScreen.poll();await atomicScreen.flush();
    assert.equal(slot.hidden,false);assert.equal(bar.hidden,false,'actual generation progress reuses the same row again');
    assert.equal(bar.attributes['data-indeterminate'],'false');assert.equal(bar.getAttribute('aria-valuetext'),null);
    assert.equal(atomicScreen.nodes.get('candidate-progress-fill').style.width,'40%');
    assert.equal(deck.hidden,true);assert.equal(atomicScreen.nodes.get('candidate-progress').attributes['aria-valuenow'],'2');
    assert.equal(panel.style.getPropertyValue('--candidate-work-height'),'246px');
    assert.ok(atomicScreen.heightMessages.every(entry=>entry.message.height===286),'pending, queued and running states never report a transient cards-plus-status height');
    phase='done';await atomicScreen.poll();await atomicScreen.flush();
    assert.equal(deck.hidden,false);assert.equal(panel.attributes['aria-busy'],'false');
    assert.equal(slot.hidden,true);assert.equal(bar.hidden,true,'completion removes the generation-only row');
    assert.equal(atomicScreen.nodes.get('candidate-empty').hidden,true);assert.equal(atomicScreen.nodes.get('candidate-list').children.length,5);
    assert.equal(panel.style.getPropertyValue('--candidate-work-height'),'');
    assert.equal(atomicScreen.nodes.get('reference-group-count').textContent,'1 / 2 선택','generation never changes the accepted quota');
    assert.ok(atomicScreen.heightMessages.every(entry=>entry.message.height===286));
    assert.equal(atomicScreen.calls.filter(call=>call.options).length,1);assert.equal(atomicScreen.nodes.get('audio').playCalls,undefined);
  }
  assert.match(html,/\.candidate-deck\[hidden\]\{display:none\}/);
  assert.doesNotMatch(html,/\.candidate-deck:not\(\[hidden\]\)[^{]*\{[^}]*animation\s*:\s*(?!none\b)[\w-]+/,'revealing the next deck does not fade its controls in from transparent');
  assert.match(html,/<div id="candidate-progress-slot" class="candidate-progress-slot" hidden>\s*<div id="candidate-progress"[^>]*hidden><span id="candidate-progress-fill"[^>]*><\/span><\/div>\s*<\/div>/,'the optional progress bar lives inside one initially hidden reservation row');
  const progressSlotCss=html.match(/\.candidate-progress-slot\{([^}]*)\}/)?.[1];
  assert.ok(progressSlotCss);assert.match(progressSlotCss,/height:9px/);assert.match(progressSlotCss,/padding-top:6px/);assert.match(progressSlotCss,/flex:none/);
  const progressBarCss=html.match(/\.candidate-progress\{([^}]*)\}/)?.[1];
  assert.match(progressBarCss,/height:3px/);assert.doesNotMatch(progressBarCss,/margin-top/,'the 6px gap belongs to the persistent slot, so toggling the bar cannot move the centered status text');
  assert.match(html,/\.candidate-progress\[data-indeterminate="true"\] span\{[^}]*animation:candidate-progress-wait 3s[^}]*alternate/,'unknown progress uses a moving segment so the gauge is visible immediately');
  assert.match(html,/@media\(prefers-reduced-motion:reduce\)\{[^\n]*\.candidate-progress\[data-indeterminate="true"\] span\{animation:none\}/,'reduced motion keeps the segment visible without animation');
  for(const job of [
    {status:'queued'}, {status:'running'}, {status:'running',completed:null,total:5},
    {status:'running',completed:-1,total:5}, {status:'running',completed:6,total:5}, {status:'running',completed:0,total:0},
  ]){
    const unknown=await screen('reference-preview',[],{config:candidateConfig,payload:{...initialCandidateState,job:{group:initialCandidateState.active_group,...job}}});
    assertGeneratingStatus(unknown);
    const bar=unknown.nodes.get('candidate-progress');assert.equal(bar.attributes['data-indeterminate'],'true','missing or invalid measurements cannot fabricate a completed count');
    assert.equal(bar.getAttribute('aria-valuenow'),null);assert.equal(unknown.nodes.get('candidate-progress-fill').style.width,'25%');
    assert.ok(bar.getAttribute('aria-valuetext'));assert.equal(unknown.calls.filter(call=>call.options).length,0,'showing an existing unmeasured job never starts another generation');
  }
  assert.match(html,/\.candidate-panel\[aria-busy="true"\]:not\(\[hidden\]\)\{[^}]*justify-content:center[^}]*min-height:var\(--candidate-work-height,232px\)/,'the progress message replaces the deck inside its reserved height');
  const queuedGeneration=await screen('reference-preview',[],{config:candidateConfig,payload:{...initialCandidateState,
    batches:[{id:'old',number:1,count:5},{id:'current',number:2,count:5}],current_batch:'current',view_batch:'current',
    job:{status:'queued',group:initialCandidateState.active_group,completed:0,total:5}}});
  assert.equal(queuedGeneration.nodes.get('candidate-previous').hidden,true);assert.equal(queuedGeneration.nodes.get('candidate-previous').disabled,true);
  assertGeneratingStatus(queuedGeneration);
  const cardPlayback=await screen('reference-preview',[],{config:candidateConfig,payload:initialCandidateState});
  const playbackCards=cardPlayback.nodes.get('candidate-list').children,cardAudio=cardPlayback.nodes.get('audio');
  const liveCounter=cardPlayback.nodes.get('reference-group-count');let countTextValue=liveCounter.textContent,countTextWrites=0,countAttributeWrites=0;
  Object.defineProperty(liveCounter,'textContent',{get:()=>countTextValue,set:value=>{countTextWrites++;countTextValue=value;}});
  const setCountAttribute=liveCounter.setAttribute.bind(liveCounter);liveCounter.setAttribute=(name,value)=>{countAttributeWrites++;setCountAttribute(name,value);};
  playbackCards[0].children[0].handlers.click();assert.equal(cardAudio.playCalls,1);
  assert.equal(playbackCards[0].children[0].attributes['aria-pressed'],'true');
  cardAudio.currentTime=.4;playbackCards[0].children[0].handlers.click();
  assert.equal(cardAudio.paused,true);assert.equal(playbackCards[0].children[0].attributes['aria-pressed'],'false');
  playbackCards[0].children[0].handlers.click();assert.equal(cardAudio.playCalls,2);assert.equal(cardAudio.currentTime,.4,'the card play button resumes a paused clip');
  playbackCards[1].children[0].handlers.click();assert.equal(cardAudio.playCalls,3);
  assert.equal(cardAudio.src,initialCandidateState.items[1].audio);assert.equal(playbackCards[0].attributes['data-playing'],'false');
  assert.equal(playbackCards[1].attributes['data-playing'],'true','switching cards moves the one active audio source');
  playbackCards[1].children[0].handlers.click();cardAudio.finish();await cardPlayback.flush();
  assert.equal(cardAudio.playCalls,3,'a paused candidate cannot auto-play the next one');
  assert.equal(cardPlayback.calls.length,1,'play/pause and switching cards neither select candidates nor mark them heard');
  assert.equal(countTextWrites,0);assert.equal(countAttributeWrites,0,'play/pause never rewrites the unchanged live selection counter');
  const missingMetadataState=structuredClone(initialCandidateState);
  delete missingMetadataState.items[0].voice_settings;delete missingMetadataState.items[0].seed;
  missingMetadataState.items[1].voice_settings=[];missingMetadataState.items[1].seed=null;
  const missingMetadata=await screen('reference-preview',[],{config:candidateConfig,payload:missingMetadataState});
  for(const row of missingMetadata.nodes.get('candidate-list').children.slice(0,2)){
    const text=nodeText(row.children[1]);
    assert.match(text,/설정 정보 없음/);assert.match(text,/시드 정보 없음/);
    assert.doesNotMatch(text,/undefined|null|시드\s*[:·]?\s*0(?:\D|$)/,'missing metadata is not replaced with an invented value');
  }
  const completedState={...structuredClone(state),groups:categories.map(group=>({...group,approved:true})),active_group:categories[19].key,
    review_complete:true,items:[],accepted:candidates(categories[19].key).slice(0,3).map(item=>({...item,heard:true,selected:true})),job:null};
  const completed=await screen('reference-preview',[],{config:candidateConfig,payload:completedState});
  assert.equal(completed.nodes.get('reference-group-name').textContent,'공통 평가용 · 여성 노년');
  assert.equal(completed.nodes.get('reference-group-count').textContent,'3 / 3 선택');
  assert.equal(completed.nodes.get('candidate-job-icon').attributes['data-state'],'complete');
  assert.equal(completed.nodes.get('candidate-more').disabled,true);assert.equal(completed.nodes.get('reference-group-next').disabled,true);
  assert.ok(completed.parentMessages.some(({message})=>message.type==='koww:reference-review'&&message.complete===true));
  const naturalState=structuredClone(initialCandidateState);
  naturalState.groups[0].target=50;
  naturalState.accepted=Array.from({length:10},(_,index)=>candidates(naturalState.active_group,index*5+1)).flat().map(item=>({...item,heard:true,selected:true}));
  naturalState.items=naturalState.accepted.slice(-5);
  let workspaceSize=260.2;
  const natural=await screen('reference-preview',[],{config:candidateConfig,payload:naturalState,workspaceHeight:()=>workspaceSize,bodyPadding:{top:4,bottom:6}});
  assert.equal(natural.nodes.get('candidate-accepted-list'),undefined,'fifty accepted voices do not add fifty hidden or visible tray rows');
  assert.equal(natural.nodes.get('candidate-list').children.length,5);
  assert.equal(natural.nodes.get('reference-group-count').textContent,'50 / 50 선택');
  assert.equal(natural.nodes.get('reference-group-count').attributes['data-complete'],'true');
  assert.deepEqual(natural.heightMessages.at(-1),{message:{type:'koww:reference-height',project:'bora',mode:'candidate_pool_v2',height:271},origin:'http://127.0.0.1:8793'},'natural height includes body padding exactly once');
  assert.ok(natural.frameMessages.findIndex(entry=>entry.message.type==='koww:reference-layout')<natural.frameMessages.findIndex(entry=>entry.message.type==='koww:reference-height'),'the parent receives candidate layout before its measured height');
  assert.equal(natural.resizeObservers[0].node,natural.nodes.get('reference-workspace'),'the visible workspace, including errors, is observed');
  const heightCount=natural.heightMessages.length;natural.resize(900);natural.notifyLayout();natural.notifyLayout();await natural.flush();
  assert.equal(natural.heightMessages.length,heightCount,'a parent viewport-height change cannot cause a measurement feedback loop');
  workspaceSize=360.4;natural.notifyLayout();await natural.flush();
  assert.equal(natural.heightMessages.at(-1).message.height,371,'visible card wrapping still reports its natural content height');
  const expandedCount=natural.heightMessages.length;natural.resize(5000);await natural.flush();assert.equal(natural.heightMessages.length,expandedCount);
  workspaceSize=420.4;natural.nodes.get('error').textContent='연결 상태를 확인하고 있습니다.';natural.notifyLayout();await natural.flush();
  assert.equal(natural.heightMessages.at(-1).message.height,431,'changes to the observed error region also resize the parent');
  workspaceSize=260.2;natural.nodes.get('error').textContent='';natural.notifyLayout();await natural.flush();
  assert.equal(natural.heightMessages.at(-1).message.height,271,'shorter visible content shrinks the reported height again');
  const beforeHandshake=natural.heightMessages.length;
  natural.messages.message({origin:'https://other.invalid',source:natural.parent,data:{type:'koww:reference-status',project:'bora'}});
  natural.messages.message({origin:'http://127.0.0.1:8793',source:natural.parent,data:{type:'koww:reference-status',project:'wrong'}});await natural.flush();
  assert.equal(natural.heightMessages.length,beforeHandshake);
  natural.messages.message({origin:'http://127.0.0.1:8793',source:natural.parent,data:{type:'koww:reference-status',project:'bora'}});await natural.flush();
  assert.equal(natural.heightMessages.length,beforeHandshake+1,'a trusted iframe-load handshake forces a height report even when unchanged');
  const beforePageHide=natural.heightMessages.length;natural.messages.pagehide();natural.notifyLayout();await natural.flush();
  assert.equal(natural.heightMessages.length,beforePageHide,'a document being replaced cannot send stale size updates');
  natural.messages.pageshow();await natural.flush();assert.equal(natural.heightMessages.length,beforePageHide+1,'restoring the same document reports its current natural height again');
  assert.equal(natural.calls.length,1,'expansion, measurements and handshakes never mutate selections or start GPU work');
  legacyGrouped.notifyLayout();await legacyGrouped.flush();assert.deepEqual(legacyGrouped.heightMessages,[],'legacy previews keep their existing iframe sizing behavior');
  assert.deepEqual(directCandidate.heightMessages,[],'a standalone candidate page does not send iframe-size messages');
  assert.doesNotMatch(html,/id="candidate-accepted(?:-list|-title)?"/,'the previous selected-voices disclosure is removed');
  assert.match(html,/animation:candidate-spin 3s linear infinite/);assert.equal((html.match(/<audio /g)||[]).length,1);
  assert.equal((html.match(/id="candidate-job-icon"/g)||[]).length,1,'there is exactly one candidate generation spinner');
  const studioCss=fs.readFileSync(path.join(__dirname,'../korean_openwakeword/resources/studio_story.css'),'utf8');
  const cssProperty=(rule,name)=>rule.match(new RegExp(`(?:^|;)\\s*${name}\\s*:\\s*([^;]+)`))?.[1]?.trim();
  const spinner=html.match(/\.candidate-job-icon\[data-state="running"\]::before\{([^}]+)\}/)[1];
  const downloadSpinner=studioCss.match(/\.story-progress-indicator\[data-status="active"\][^{]*\{([^}]+)\}/)[1];
  const downloadSize=studioCss.match(/\.story-progress-indicator,[^{]+\{([^}]+)\}/)[1];
  for(const property of ['border','border-right-color','border-radius'])assert.equal(cssProperty(spinner,property),cssProperty(downloadSpinner,property),'candidate generation shares the download spinner shape/color');
  for(const property of ['width','height','box-sizing'])assert.equal(cssProperty(spinner,property),cssProperty(downloadSize,property));
  assert.equal(cssProperty(spinner,'animation').split(/\s+/).slice(1).join(' '),cssProperty(downloadSpinner,'animation').split(/\s+/).slice(1).join(' '));
  const playRule=html.match(/\.candidate-row \.candidate-play\{([^}]+)\}/)[1],cardRule=html.match(/\.candidate-row\{([^}]+)\}/)[1];
  const luminance=color=>{const channel=color.match(/[a-f0-9]{2}/gi).map(value=>parseInt(value,16)/255).map(value=>value<=.04045?value/12.92:((value+.055)/1.055)**2.4);return .2126*channel[0]+.7152*channel[1]+.0722*channel[2];};
  const contrast=(a,b)=>{const values=[luminance(a),luminance(b)].sort((x,y)=>y-x);return (values[0]+.05)/(values[1]+.05);};
  assert.ok(contrast(cssProperty(playRule,'background'),cssProperty(cardRule,'background'))>=3,'the play button is distinct against the dark card');
  assert.ok(contrast(cssProperty(playRule,'color'),cssProperty(playRule,'background'))>=4.5,'the play/pause symbol remains legible on its neutral button');
  const displayGroup=categories[0].key,displayBatchOne='display-one',displayBatchTwo='display-two';
  const displayOriginal=candidates(displayGroup).map(row=>({...row,batch_id:displayBatchOne}));
  const displayAdditional=candidates(displayGroup,66).map(row=>({...row,batch_id:displayBatchTwo}));
  const displayState={...structuredClone(initialCandidateState),items:displayAdditional,accepted:[{...displayOriginal[0],heard:true,selected:true}],
    batches:[{id:displayBatchOne,number:1,count:5},{id:displayBatchTwo,number:2,count:5}],view_batch:displayBatchTwo,current_batch:displayBatchTwo};
  const displaySnapshot=JSON.stringify(displayState);
  const displayScreen=await screen('reference-preview',[],{config:candidateConfig,respond:async url=>{
    const batch=new URL(url,'http://127.0.0.1:8793').searchParams.get('batch')||displayBatchTwo;
    return {payload:{...displayState,view_batch:batch,items:batch===displayBatchOne?displayOriginal:displayAdditional}};
  }});
  const displayCards=preview=>preview.nodes.get('candidate-list').children;
  const displayLabels=preview=>displayCards(preview).map(card=>card.children[1].children[0].children[0].textContent);
  assert.deepEqual(displayLabels(displayScreen),['후보 6','후보 7','후보 8','후보 9','후보 10'],'failed generation reservations do not consume visible candidate numbers');
  for(let i=0;i<5;i++){
    assert.equal(displayCards(displayScreen)[i].attributes['aria-label'],`후보 ${i+6}`);
    assert.match(displayCards(displayScreen)[i].children[0].attributes['aria-label'],new RegExp(`후보 ${i+6} `));
    assertCandidateMetadata(displayCards(displayScreen)[i].children[1],displayAdditional[i],i+6);
  }
  displayScreen.nodes.get('candidate-previous').handlers.click();await displayScreen.flush();
  assert.deepEqual(displayLabels(displayScreen),['후보 1','후보 2','후보 3','후보 4','후보 5']);
  assert.equal(displayScreen.nodes.get('candidate-more-label').textContent,'›');
  displayScreen.nodes.get('candidate-more').handlers.click();await displayScreen.flush();
  assert.deepEqual(displayLabels(displayScreen),['후보 6','후보 7','후보 8','후보 9','후보 10']);
  assert.equal(JSON.stringify(displayState),displaySnapshot,'display numbering never edits original numbers, seeds, selection IDs or sources');
  assert.equal(displayScreen.calls.filter(call=>call.options).length,0,'history renumbering makes no writes or generation requests');
  for(const group of [categories[1].key,categories[10].key]){
    const payload={...structuredClone(initialCandidateState),active_group:group,items:candidates(group,101).map(row=>({...row,batch_id:'first'})),
      batches:[{id:'first',number:1,count:5}],view_batch:'first',current_batch:'first'};
    const separate=await screen('reference-preview',[],{config:candidateConfig,payload});
    assert.deepEqual(displayLabels(separate),['후보 1','후보 2','후보 3','후보 4','후보 5'],'each age and train/evaluation category starts independently');
  }
  const enrichedState={...structuredClone(displayState),accepted:[],batches:[]};
  const enriched=await screen('reference-preview',[],{config:candidateConfig,respond:async(url,options)=>{
    if(options){
      const action=JSON.parse(options.body);
      assert.deepEqual(action,{action:'toggle',id:displayAdditional[0].id,sha256:displayAdditional[0].sha256},'display labels never replace persisted candidate identifiers');
      enrichedState.batches=displayState.batches;enrichedState.accepted=[{...enrichedState.items[0],selected:true}];
    }
    return {payload:enrichedState};
  }});
  const enrichedCard=displayCards(enriched)[0],enrichedCheckbox=enrichedCard.children[3].children[0];
  assert.equal(displayLabels(enriched)[0],'후보 1');enrichedCheckbox.focus();enrichedCheckbox.change(true);await enriched.flush();
  assert.equal(displayCards(enriched)[0],enrichedCard);assert.equal(enriched.document.activeElement,enrichedCheckbox);
  assert.equal(displayLabels(enriched)[0],'후보 6');assert.equal(enrichedCard.attributes['aria-label'],'후보 6','a reused card updates its heading and accessibility label together');
  console.log("Read-only preview UI: hard/reference playback, no consent writes, source validation and errors passed");
  console.log('Candidate review UI: optional listening, explicit selection, visible settings/actual seeds, factual full-heard records, quota, group transition, polling and retries passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
