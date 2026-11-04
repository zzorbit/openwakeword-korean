"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const html = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/reference_preview.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const playerScript = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/listening_player.js"), "utf8");

async function screen(mode, items, { error, ok = true, playback, height = 144, width = 560, buttonRect = {top:22,bottom:62,left:8,width:460},autoReady=true,standalone=false,queue,actualPlayer=false } = {}) {
  const nodes = new Map();
  class Element {
    constructor() { this.value = ""; this.textContent = ""; this.children = []; this.handlers = {}; this.attributes = {}; this.style = {setProperty(name,value){this[name]=String(value);},getPropertyValue(name){return this[name]??"";}}; this.scrollTop = 0; this.scrollHeight = 156; this.clientHeight = 68; this.offsetTop = 0; this.offsetHeight = 36; this.currentTime = 0; this.controls = true; this.played = 0; this.pauses = 0; this.paused = true; this.ended = false; this.duration = NaN; this.muted = false; this.className = ""; this.classList = { add: (...names) => { this.className = [...new Set([...this.className.split(/\s+/).filter(Boolean), ...names])].join(" "); } }; }
    append(...nodes) { for(const node of nodes){if(node.parentElement){const siblings=node.parentElement.children;siblings.splice(siblings.indexOf(node),1);}node.parentElement=this;node.offsetTop=this.children.length*36+4;this.children.push(node);} }
    replaceChildren(...nodes) { for(const child of this.children)child.parentElement=null;this.children=[];this.append(...nodes); }
    insertBefore(node, anchor) { assert.ok(this.children.includes(anchor),'shared transport inserts beside the existing next button');if(node===anchor)return;if(node.parentElement){const siblings=node.parentElement.children;siblings.splice(siblings.indexOf(node),1);}node.parentElement=this;this.children.splice(this.children.indexOf(anchor),0,node); }
    addEventListener(event, callback) { this.handlers[event] = callback; }
    setAttribute(key, value) { this.attributes[key] = value; }
    getAttribute(key) { return this.attributes[key] ?? null; }
    removeAttribute(key) { delete this.attributes[key]; }
    contains(node) { return node===this || this.children.some(child=>child.contains(node)); }
    focus() { document.activeElement=this; }
    getBoundingClientRect() { return {height:buttonRect.bottom-buttonRect.top,...buttonRect}; }
    get src() { return this.attributes.src; }
    set src(value) { this.attributes.src=value; this.ended=false;this.currentTime=0; }
    load() { this.ended=false; }
    play() { this.played++; if(this.ended)this.currentTime=0;this.ended=false;this.paused=false; this.handlers.play?.(); return playback ? playback(this.played) : Promise.resolve(); }
    pause() { this.pauses++; this.paused=true; this.handlers.pause?.(); }
    finish() { this.ended=true; this.paused=true; this.handlers.pause?.(); this.handlers.ended?.(); }
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], new Element());
  nodes.get('voice-picker').append(nodes.get('voice-picker-button'),nodes.get('voice-picker-list'));
  nodes.get('voice-picker-button').append(nodes.get('voice-picker-label'));
  const playbackControls=new Element(),calls = [], enhancements = [], parentMessages = [], messages = {}, documentEvents = {}, document = { title: "기준 목소리 듣기", hidden:false, addEventListener:(name,callback)=>{documentEvents[name]=callback;}, getElementById: id => nodes.get(id), createElement: () => new Element(),querySelector:selector=>selector==='.listen-controls'?playbackControls:null };
  playbackControls.append(nodes.get('previous'),nodes.get('continuous'),nodes.get('next'));
  nodes.get('reference-playback-actions').append(nodes.get('play-single'),nodes.get('play-auto'));
  if(nodes.has('reference-more-panel'))nodes.get('reference-more-panel').append(nodes.get('reference-more-transport'),nodes.get('reference-playback-actions'));
  const context = vm.createContext({ document, URL, innerHeight:height,innerWidth:width,location: new URL(`http://127.0.0.1:8793/${mode}/?project=bora`),
    fetch: async (url, options) => { calls.push({ url, options });if(queue)await queue;return { ok, json: async () => ({ items, error }) }; } });
  context.window = context; context.parent = standalone?context:{postMessage:(message,origin)=>{
    parentMessages.push({message:JSON.parse(JSON.stringify(message)),origin});
    if(autoReady&&message.type==='koww:voice-picker'&&message.open)messages.message({origin,source:context.parent,data:{type:'koww:voice-picker-ready',request:message.request}});
  }};
  if(actualPlayer)vm.runInContext(playerScript,context);
  const enhance=actualPlayer?context.KowwListening.enhance:null;
  context.KowwListening={enhance:(audio,controls,options)=>{const handle=enhance?enhance(audio,controls,options):{seek:new Element(),mute:new Element()};enhancements.push({audio,controls,options,handle});if(!enhance)audio.controls=false;return handle;}};
  context.addEventListener = (name, callback) => { messages[name] = callback; };
  vm.runInContext(script, context);
  for (let index = 0; index < 8; index++) await new Promise(resolve => setImmediate(resolve));
  return { nodes, calls, enhancements, playbackControls, document, documentEvents, messages, parentMessages, parent: context.parent, ready:request=>messages.message({origin:'http://127.0.0.1:8793',source:context.parent,data:{type:'koww:voice-picker-ready',request}}),resize:(nextHeight,nextRect)=>{context.innerHeight=nextHeight;if(nextRect)Object.assign(buttonRect,nextRect);messages.resize();} };
}

(async () => {
  assert.match(html, /href="\/audio_listening\.css"/);
  assert.ok(html.indexOf('href="/audio_listening.css"') < html.indexOf('html.reference-listening .listen-workspace'));
  assert.match(html, /class="listen-workspace"/);
  assert.match(html, /id="listening-sidebar" class="listen-sidebar"/);
  assert.match(html, /class="listen-player"/);
  assert.doesNotMatch(html, /듣기는 선택 사항이며 따로 승인할 필요가 없습니다/);
  assert.ok(html.indexOf('class="listen-heading"') < html.indexOf('id="audio"'));
  assert.ok(html.indexOf('id="audio"') < html.indexOf('class="controls listen-controls"'));
  assert.ok(html.indexOf('id="voice"') < html.indexOf('id="audio"'));
  assert.doesNotMatch(html, /<form|localStorage|listened|decision|whole_pool_policy_approved/);
  const items = [
    { id: "hard1", text: "돌아와", voice_family_id: "train_01", audio: "/hard-preview/audio/0?project=bora" },
    { id: "hard2", text: "기다려", voice_family_id: "train_02", audio: "/hard-preview/audio/1?project=bora" }
  ];
  const hard = await screen("hard-preview", items);
  assert.equal(hard.document.title, "유사어 전체 듣기");
  assert.equal(hard.nodes.get("voice").attributes["aria-label"], "들을 유사어 음원");
  assert.equal(hard.nodes.get("listening-sidebar").hidden, false);
  assert.equal(hard.nodes.get("voice").hidden, true);
  assert.equal(hard.nodes.get("sentence").hidden, false);
  assert.equal(hard.nodes.get("voice").children[0].textContent, "돌아와 · 목소리 1");
  assert.equal(hard.enhancements.length,1);assert.equal(hard.enhancements[0].controls,hard.playbackControls);
  assert.equal(hard.nodes.get('audio').controls,false);
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
  assert.equal(reference.nodes.get("listening-sidebar").hidden, true);
  assert.equal(reference.nodes.get("voice").hidden, true);
  assert.equal(reference.nodes.get("voice-picker").hidden, false);
  assert.equal(reference.nodes.get("voice-picker-label").textContent, "학습용 · 1");
  assert.equal(reference.nodes.get("voice-picker-button").attributes['aria-label'], "들을 목소리: 학습용 · 목소리 1");
  assert.equal(reference.nodes.get("voice-picker-list").hidden, true);
  assert.equal(reference.nodes.get("sentence").hidden, true);
  assert.equal(reference.nodes.get("listening-list").children.length, 0);
  assert.equal(reference.nodes.get("reference-playback-actions").hidden, false);
  assert.equal(reference.enhancements.length,1,'reference listening uses the shared custom player again');
  assert.equal(reference.enhancements[0].audio,reference.nodes.get('audio'));assert.equal(reference.enhancements[0].controls,reference.playbackControls);
  assert.equal(reference.enhancements[0].options.play,reference.nodes.get('continuous'));
  assert.equal(reference.enhancements[0].options.previous,reference.nodes.get('previous'));assert.equal(reference.enhancements[0].options.next,reference.nodes.get('next'));
  assert.equal(reference.nodes.get('audio').controls,false);
  assert.equal(reference.nodes.get("audio").played, 0); // Opening is never autoplay.
  assert.deepEqual(reference.calls, [{ url: "/reference-preview/queue?project=bora", options: undefined }]);
  const many=await screen('reference-preview',Array.from({length:7},(_,i)=>({id:`voice-${i}`,group:'학습용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`})));
  assert.equal(many.nodes.get('voice').children.length,7);
  many.nodes.get('voice').value='6';many.nodes.get('voice').handlers.change();
  assert.equal(many.nodes.get('voice').value,'6');
  assert.equal(many.nodes.get('audio').played,0,'selecting a paused recording stays paused');
  assert.equal(many.nodes.get('position').textContent,'7 / 7');
  many.nodes.get('play-single').handlers.click();many.nodes.get('audio').finish();
  assert.equal(many.nodes.get('voice').value,'6','reference bar play defaults to the current voice only');
  assert.equal(many.nodes.get('audio').played,1);assert.equal(many.nodes.get('continuous').textContent,'재생');
  many.nodes.get('play-auto').handlers.click();many.nodes.get('audio').finish();
  assert.equal(many.nodes.get('voice').value,'0','automatic listening loops after the last selected voice');
  assert.equal(many.nodes.get('audio').played,3);assert.equal(many.nodes.get('play-auto').textContent,'일시정지');
  assert.equal(many.calls.length,1,'voice selection and playback make no approval or extra data request');
  const modes=await screen('reference-preview',Array.from({length:3},(_,i)=>({id:`mode-${i}`,group:'학습용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`})));
  const modeAudio=modes.nodes.get('audio'),single=modes.nodes.get('play-single'),automatic=modes.nodes.get('play-auto'),modeBar=modes.nodes.get('continuous');
  single.handlers.click();assert.equal(modeAudio.played,1);assert.equal(single.textContent,'일시정지');assert.equal(automatic.textContent,'자동 재생');
  modeAudio.currentTime=0.75;automatic.handlers.click();
  assert.equal(modeAudio.played,1,'switching the active playback mode does not restart audio');assert.equal(modeAudio.currentTime,0.75);
  assert.equal(automatic.textContent,'일시정지');assert.equal(single.textContent,'재생');
  automatic.handlers.click();assert.equal(modeAudio.paused,true);assert.equal(modeBar.textContent,'재생');
  automatic.handlers.click();assert.equal(modeAudio.played,2);assert.equal(modeAudio.currentTime,0.75);assert.equal(automatic.textContent,'일시정지','automatic playback resumes from its paused position');
  single.handlers.click();assert.equal(modeAudio.played,2);assert.equal(modeAudio.currentTime,0.75);
  modeAudio.finish();assert.equal(modes.nodes.get('voice').value,'0');assert.equal(modeAudio.played,2);assert.equal(single.textContent,'재생');
  single.handlers.click();modes.nodes.get('next').handlers.click();
  assert.equal(modes.nodes.get('voice').value,'1');assert.equal(modeAudio.played,4);
  modeAudio.finish();assert.equal(modes.nodes.get('voice').value,'1','a new selection retains single-current mode');assert.equal(modeAudio.played,4);
  modes.nodes.get('voice-picker-button').handlers.click();modes.nodes.get('voice-picker-list').children[2].handlers.click();
  assert.equal(modes.nodes.get('voice').value,'2');assert.equal(modeAudio.played,4,'selecting while paused never starts playback');
  automatic.handlers.click();modes.nodes.get('voice-picker-button').handlers.click();modes.nodes.get('voice-picker-list').children[0].handlers.click();
  assert.equal(modeAudio.played,6);assert.equal(automatic.textContent,'일시정지');
  modeAudio.finish();assert.equal(modes.nodes.get('voice').value,'1');assert.equal(modeAudio.played,7,'selection during automatic playback keeps continuous advancement');
  modes.document.hidden=true;modes.documentEvents.visibilitychange();assert.equal(modeAudio.paused,true);assert.equal(modeAudio.src,undefined);
  assert.equal(single.textContent,'재생');assert.equal(automatic.textContent,'자동 재생');
  modeAudio.handlers.ended();assert.equal(modeAudio.played,7);
  modes.document.hidden=false;automatic.handlers.click();
  modes.messages.message({origin:'http://127.0.0.1:8793',source:modes.parent,data:{type:'koww:pause'}});
  assert.equal(modeAudio.paused,true);assert.equal(automatic.textContent,'자동 재생');assert.equal(single.attributes['aria-pressed'],'false');assert.equal(automatic.attributes['aria-pressed'],'false');
  assert.equal(modes.calls.length,1,'mode selection and lifecycle controls are read-only');
  // Mount the shipped shared player once: layout changes must keep its real
  // seek/mute controls wired to the same reference transport and picker.
  const compact=await screen('reference-preview',Array.from({length:3},(_,i)=>({id:`compact-${i}`,group:'학습용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`})),{actualPlayer:true});
  const compactAudio=compact.nodes.get('audio'),transport=compact.enhancements[0].handle;
  assert.ok(transport);assert.equal(compactAudio.controls,false);assert.equal(compact.enhancements.length,1);
  assert.ok(compact.playbackControls.children.includes(transport.seek));
  const compactMore=compact.nodes.get('reference-more-button'),compactPanel=compact.nodes.get('reference-more-panel');
  assert.equal(transport.mute.parentElement,compact.nodes.get('reference-more-transport'));
  assert.equal(compact.nodes.get('previous').parentElement,transport.mute.parentElement);assert.equal(compact.nodes.get('next').parentElement,transport.mute.parentElement);
  compactAudio.duration=4;compactAudio.currentTime=1;compactAudio.handlers.loadedmetadata();
  transport.seek.value='750';transport.seek.handlers.input();
  assert.equal(compactAudio.currentTime,3);assert.equal(transport.seek.attributes['aria-valuetext'],'0:03 / 0:04');
  assert.equal(transport.seek.style.getPropertyValue('--listen-progress'),'75%');
  assert.equal(compactAudio.played,0,'seeking a paused voice never starts playback');
  compactMore.handlers.click();assert.equal(compactPanel.hidden,false);assert.equal(compact.document.activeElement,compactPanel);
  transport.mute.handlers.click();assert.equal(compactAudio.muted,true);assert.equal(transport.mute.attributes['aria-pressed'],'true');
  compact.nodes.get('play-single').handlers.click();assert.equal(compactAudio.played,1);assert.equal(compactPanel.hidden,true);
  compactMore.handlers.click();
  compact.nodes.get('next').handlers.click();assert.equal(compact.nodes.get('voice').value,'1');assert.equal(compactAudio.played,2);
  compact.nodes.get('previous').handlers.click();assert.equal(compact.nodes.get('voice').value,'0');assert.equal(compactAudio.played,3);
  assert.equal(compactAudio.muted,true,'transport selection preserves mute state');
  compact.nodes.get('play-auto').handlers.click();assert.equal(compactAudio.played,3,'mode switching preserves the active play request');assert.equal(compactPanel.hidden,true);
  compactAudio.finish();assert.equal(compact.nodes.get('voice').value,'1');assert.equal(compactAudio.played,4);
  compact.nodes.get('continuous').handlers.click();assert.equal(compactAudio.paused,true);
  const compactButton=compact.nodes.get('voice-picker-button'),compactList=compact.nodes.get('voice-picker-list');
  const compactKey=(element,key)=>element.handlers.keydown({key,preventDefault(){}});
  compactKey(compactButton,'ArrowDown');assert.equal(compactList.hidden,false);
  compactKey(compactList,'End');compactKey(compactList,'Enter');
  assert.equal(compact.nodes.get('voice').value,'2');assert.equal(compactList.hidden,true);assert.equal(compact.document.activeElement,compactButton);
  assert.equal(compactAudio.played,4,'keyboard selection after pause remains paused');
  transport.mute.handlers.click();assert.equal(compactAudio.muted,false);
  assert.equal(compact.calls.length,1,'the shared compact player never writes listening decisions or reloads the queue');
  const moreAwaiting=await screen('reference-preview',[{id:'more-0',group:'학습용',text:'오늘은',audio:'/reference-preview/audio/0?project=bora'}],{autoReady:false});
  const moreButton=moreAwaiting.nodes.get('reference-more-button'),morePanel=moreAwaiting.nodes.get('reference-more-panel');
  const morePicker=moreAwaiting.nodes.get('voice-picker-button'),moreList=moreAwaiting.nodes.get('voice-picker-list');
  moreButton.handlers.click();assert.equal(morePanel.hidden,true);assert.equal(moreButton.attributes['aria-expanded'],'true');
  const moreRequest=()=>moreAwaiting.parentMessages.filter(item=>item.message.open).at(-1).message.request;
  const firstMore=moreRequest();moreAwaiting.ready(firstMore);assert.equal(morePanel.hidden,false);
  morePicker.handlers.click();assert.equal(morePanel.hidden,true);assert.equal(moreList.hidden,true);
  const pickerRequest=moreRequest();assert.ok(pickerRequest>firstMore);
  moreAwaiting.ready(firstMore);assert.equal(morePanel.hidden,true);assert.equal(moreList.hidden,true);
  moreAwaiting.ready(pickerRequest);assert.equal(moreList.hidden,false);
  moreButton.handlers.click();assert.equal(moreList.hidden,true);assert.equal(morePanel.hidden,true);
  const lastMore=moreRequest();moreAwaiting.ready(pickerRequest);assert.equal(morePanel.hidden,true);
  moreAwaiting.ready(lastMore);assert.equal(morePanel.hidden,false);
  compactKey(morePanel,'Escape');assert.equal(morePanel.hidden,true);assert.equal(moreAwaiting.document.activeElement,moreButton);
  moreAwaiting.ready(lastMore);assert.equal(morePanel.hidden,true,'late ready cannot reopen a closed more panel');
  moreButton.handlers.click();moreAwaiting.ready(moreRequest());
  moreAwaiting.documentEvents.pointerdown({target:moreAwaiting.nodes.get('audio')});assert.equal(morePanel.hidden,true);
  moreButton.handlers.click();moreAwaiting.ready(moreRequest());
  moreAwaiting.messages.message({origin:'http://127.0.0.1:8793',source:moreAwaiting.parent,data:{type:'koww:pause'}});
  assert.equal(morePanel.hidden,true);assert.equal(moreButton.attributes['aria-expanded'],'false');
  moreButton.handlers.click();const tabRequest=moreRequest();compactKey(moreButton,'Tab');moreAwaiting.ready(tabRequest);
  assert.equal(morePanel.hidden,true,'Tab cancels a pending more request before a late ready can take focus');
  assert.equal(moreButton.attributes['aria-expanded'],'false');
  moreButton.handlers.click();moreAwaiting.ready(moreRequest());
  morePanel.handlers.focusout({relatedTarget:null});assert.equal(morePanel.hidden,true,'leaving the iframe releases the expanded more panel');
  assert.equal(moreAwaiting.calls.length,1,'both overlays share layout messages without a data request');
  const choiceItems=Array.from({length:50},(_,i)=>({id:`choice-${i}`,group:i<20?'학습용':'공통 평가용',text:'오늘은',audio:`/reference-preview/audio/${i}?project=bora`}));
  const choices=await screen('reference-preview',choiceItems);
  const pickerButton=choices.nodes.get('voice-picker-button'),pickerList=choices.nodes.get('voice-picker-list');
  const key=(element,value)=>{let prevented=false;element.handlers.keydown({key:value,preventDefault:()=>{prevented=true;}});return prevented;};
  pickerButton.handlers.click();
  assert.equal(pickerList.hidden,false);assert.equal(pickerButton.attributes['aria-expanded'],'true');
  assert.equal(choices.document.activeElement,pickerList);
  assert.equal(pickerList.children.length,50);
  assert.deepEqual(pickerList.children[0].children.map(node=>node.textContent),['학습용','1']);assert.equal(pickerList.children[0].attributes['aria-label'],'학습용 · 목소리 1');
  assert.deepEqual(pickerList.children[19].children.map(node=>node.textContent),['학습용','20']);
  assert.deepEqual(pickerList.children[20].children.map(node=>node.textContent),['공통 평가용','1']);
  assert.deepEqual(pickerList.children[49].children.map(node=>node.textContent),['공통 평가용','30']);assert.equal(pickerList.children[49].attributes['aria-label'],'공통 평가용 · 목소리 30');
  assert.equal(pickerList.style.maxHeight,'75px','popup fits the available iframe and overlaps the selected control border');
  assert.equal(pickerList.style.top,'61px');assert.equal(pickerList.style.left,'8px');assert.equal(pickerList.style.width,'460px');
  assert.equal(pickerList.attributes['data-side'],'down');
  assert.ok(parseFloat(pickerList.style.top)+parseFloat(pickerList.style.maxHeight)<=144-8);
  assert.deepEqual(choices.parentMessages,[{message:{type:'koww:voice-picker',open:true,request:1},origin:'http://127.0.0.1:8793'}]);
  choices.resize(300,{top:100,bottom:140});
  assert.equal(pickerList.style.maxHeight,'153px');
  assert.ok(parseFloat(pickerList.style.top)+parseFloat(pickerList.style.maxHeight)<=300-8);
  choices.resize(700,{top:100,bottom:140});assert.equal(pickerList.style.maxHeight,'240px');
  choices.resize(300,{top:240,bottom:280});
  assert.equal(pickerList.style.top,'8px','picker opens upward when space below is smaller');
  assert.equal(pickerList.style.maxHeight,'233px');assert.equal(pickerList.attributes['data-side'],'up');
  assert.equal(key(pickerList,'End'),true);assert.equal(pickerList.attributes['aria-activedescendant'],'voice-option-49');
  key(pickerList,'Home');key(pickerList,'ArrowDown');key(pickerList,'Enter');
  assert.equal(choices.nodes.get('voice').value,'1');assert.equal(choices.nodes.get('position').textContent,'2 / 50');
  assert.equal(choices.nodes.get('voice-picker-label').textContent,'학습용 · 2');assert.equal(pickerList.hidden,true);
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
  const awaitingButton=awaiting.nodes.get('voice-picker-button'),awaitingList=awaiting.nodes.get('voice-picker-list');
  awaitingButton.focus();key(awaitingButton,'End');
  assert.equal(awaitingButton.attributes['aria-expanded'],'true');assert.equal(awaitingList.hidden,true);
  assert.equal(awaiting.document.activeElement,awaitingButton,'the list does not take focus while the parent layout is pending');
  assert.equal(awaitingList.style.top,undefined);
  awaiting.resize(300,{top:100,bottom:140});assert.equal(awaitingList.style.top,undefined,'resize before acknowledgment cannot show or position the small pending popup');
  awaiting.messages.message({origin:'https://other.example',source:awaiting.parent,data:{type:'koww:voice-picker-ready',request:1}});
  awaiting.messages.message({origin:'http://127.0.0.1:8793',source:{},data:{type:'koww:voice-picker-ready',request:1}});
  for(const request of [0,-1,1.5,'1',undefined])awaiting.ready(request);
  assert.equal(awaitingList.hidden,true,'only an authentic acknowledgment for this open request can show the list');
  awaiting.ready(1);assert.equal(awaitingList.hidden,false);assert.equal(awaitingList.style.maxHeight,'153px');
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
  loading.ready(1);loading.nodes.get('voice-picker-button').handlers.click();
  assert.deepEqual(loading.parentMessages,[]);assert.equal(loading.nodes.get('voice-picker-list').hidden,true);
  resolveQueue();for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));
  loading.nodes.get('voice-picker-button').handlers.click();assert.equal(loading.nodes.get('voice-picker-list').hidden,true);
  loading.ready(1);assert.equal(loading.nodes.get('voice-picker-list').hidden,false);assert.equal(loading.calls.length,1);
  const parentClose=await screen('reference-preview',choiceItems);
  parentClose.nodes.get('play-single').handlers.click();parentClose.nodes.get('voice-picker-button').handlers.click();
  parentClose.messages.message({origin:'https://other.example',source:parentClose.parent,data:{type:'koww:voice-picker-close'}});
  parentClose.messages.message({origin:'http://127.0.0.1:8793',source:{},data:{type:'koww:voice-picker-close'}});
  assert.equal(parentClose.nodes.get('voice-picker-list').hidden,false,'foreign messages cannot close the reference picker');
  parentClose.messages.message({origin:'http://127.0.0.1:8793',source:parentClose.parent,data:{type:'koww:voice-picker-close'}});
  assert.equal(parentClose.nodes.get('voice-picker-list').hidden,true);
  assert.equal(parentClose.nodes.get('audio').paused,false,'clicking the parent outside the picker does not interrupt playback');
  assert.equal(parentClose.nodes.get('audio').played,1);
  assert.equal(hard.nodes.get('voice-picker').hidden,true);
  assert.equal(hard.nodes.get('reference-playback-actions').hidden,true);
  assert.deepEqual(hard.parentMessages,[],'hard-example playback never requests reference-picker expansion');
  const hardPages=await screen('hard-preview',Array.from({length:7},(_,i)=>({id:`hard-${i}`,text:'돌아와',audio:`/hard-preview/audio/${i}?project=bora`})));
  assert.equal(hardPages.nodes.get('listening-list').children.length,5);
  hardPages.nodes.get('list-next').handlers.click();
  assert.equal(hardPages.nodes.get('listening-list').children.length,2);
  assert.equal(hardPages.nodes.get('voice').value,'0','browsing a hard-example list page never selects or plays an audio');
  hardPages.nodes.get('listening-list').children[1].handlers.click();
  assert.equal(hardPages.nodes.get('voice').value,'6');
  assert.equal(hardPages.nodes.get('audio').played,0);
  assert.equal(hardPages.calls.length,1);
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
