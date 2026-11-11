"use strict";
const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../korean_openwakeword/resources/listening_player.js"), "utf8");

function surface(name) {
  let document;
  const geometry={rect:{top:200,bottom:240,left:30,width:300}};
  const listeners = new Map(), docListeners = new Map(), observers = [], frames = [];
  const add = (map, type, callback, capture) => { const rows = map.get(type) || []; rows.push({callback,capture}); map.set(type,rows); };
  class Element {
    constructor(tag) { this.tagName=tag.toUpperCase();this.children=[];this.parentElement=null;this.handlers=new Map();this.attributes={};this.hidden=false;this.disabled=false;this.scrollTop=0;this.clientHeight=120;this.offsetHeight=36;this.className="";this.style={setProperty(key,value){this[key]=value;}};this.focusCount=0; }
    get isConnected() { return this.root || Boolean(this.parentElement?.isConnected); }
    get offsetTop() { return Math.max(0,this.parentElement?.children.indexOf(this)||0)*36; }
    append(...children) { for(const node of children){node.parentElement=this;this.children.push(node);} }
    replaceChildren(...children) { for(const node of this.children)node.parentElement=null;this.children=[];this.append(...children); }
    remove() { const parent=this.parentElement;if(parent)parent.children=parent.children.filter(row=>row!==this);this.parentElement=null; }
    contains(target) { return target===this||this.children.some(node=>node.contains(target)); }
    closest() { for(let item=this;item;item=item.parentElement)if(item.hidden||item.inert)return item;return null; }
    getClientRects() { for(let item=this;item;item=item.parentElement)if(item.hidden||item.inert||item.style.display==="none")return [];return this.isConnected?[{}]:[]; }
    getBoundingClientRect() { return {...geometry.rect}; }
    setAttribute(key,value) { this.attributes[key]=String(value); }
    getAttribute(key) { return this.attributes[key]??null; }
    removeAttribute(key) { delete this.attributes[key]; }
    addEventListener(type,callback,capture) { add(this.handlers,type,callback,capture); }
    fire(type,event={}) { event.target??=this;for(const {callback} of this.handlers.get(type)||[])callback(event); }
    focus() { this.focusCount++;document.activeElement=this; }
  }
  const html=new Element("html"),body=new Element("body");html.root=true;html.append(body);
  document={documentElement:html,body,hidden:false,activeElement:null,
    createElement:tag=>new Element(tag),querySelectorAll:selector=>selector==="iframe"?frames.filter(row=>row.isConnected):[],
    addEventListener:(type,callback,capture)=>add(docListeners,type,callback,capture)};
  const context=vm.createContext({document,URL,location:{origin:"http://127.0.0.1:8793",href:`http://127.0.0.1:8793/${name}/?project=bora`},
    innerHeight:900,innerWidth:1400,addEventListener:(type,callback,capture)=>add(listeners,type,callback,capture),
    MutationObserver:class {constructor(callback){this.callback=callback;observers.push(this);}observe(){this.active=true;}disconnect(){this.active=false;}}});
  context.window=context;context.parent=context;
  const fireWindow=(type,event={})=>{for(const {callback}of listeners.get(type)||[])callback(event);};
  const fireDocument=(type,event={})=>{
    event.preventDefault??=()=>{event.prevented=true;};event.stopPropagation??=()=>{event.stopped=true;};
    for(const {callback}of docListeners.get(type)||[])callback(event);return event;
  };
  vm.runInContext(source,context);
  function picker(id="picker",onOpenChange) {
    const container=new Element("div"),button=new Element("button"),label=new Element("span"),popup=new Element("div"),list=new Element("div"),selected=[];
    list.id=id;button.append(label);popup.append(list);container.append(button,popup);body.append(container);
    const handle=context.KowwListening.createPicker({container,button,label,popup,list,onSelect:index=>selected.push(index),onOpenChange});
    handle.setItems([{label:"학습용 1",badge:"미청취"},{label:"학습용 2",badge:"미청취"}]);
    return {container,button,popup,list,handle,selected,open:()=>button.fire("click"),isOpen:()=>button.getAttribute("aria-expanded")==="true"};
  }
  return {context,document,body,Element,frames,observers,picker,fireDocument,fireWindow,docListeners,listeners,geometry,
    mutate:()=>{for(const observer of observers)if(observer.active)observer.callback([]);}};
}

function attach(parent,child,src) {
  const frame=new parent.Element("iframe");frame.setAttribute("src",src||child.context.location.href);parent.body.append(frame);parent.frames.push(frame);
  const sent=[];
  const parentProxy={postMessage(message,origin){sent.push({direction:"up",message,origin});parent.fireWindow("message",{data:message,origin:child.context.location.origin,source:childProxy});}};
  const childProxy={postMessage(message,origin){sent.push({direction:"down",message,origin});child.fireWindow("message",{data:message,origin:parent.context.location.origin,source:parentProxy});}};
  frame.contentWindow=childProxy;child.context.parent=parentProxy;
  return {frame,parentProxy,childProxy,sent};
}

let passed=0;
function test(label,run){run();passed++;console.log("  ✓ "+label);}

test("outside capture closes without cancelling background/control actions; internal option/scroll survives",()=>{
  const s=surface("review"),p=s.picker();p.open();assert.equal(p.popup.hidden,false);
  s.fireDocument("pointerdown",{target:p.list.children[1]});assert.equal(p.isOpen(),true);
  s.fireDocument("scroll",{target:p.list});assert.equal(p.isOpen(),true);
  p.list.children[1].fire("click");assert.deepEqual(p.selected,[1]);assert.equal(p.isOpen(),false);
  p.open();const volume=new s.Element("input");s.body.append(volume);let changed=0;
  volume.addEventListener("pointerdown",()=>changed++);
  const event=s.fireDocument("pointerdown",{target:volume});volume.fire("pointerdown",event);
  assert.equal(p.isOpen(),false);assert.equal(changed,1);assert.equal(event.prevented,undefined);assert.equal(event.stopped,undefined);
  assert.equal(s.docListeners.get("pointerdown")[0].capture,true);
  p.open();s.fireDocument("scroll",{target:s.body});assert.equal(p.isOpen(),false);
});

test("one registry handles replacement, Escape, hidden/detached views and repeat script loads",()=>{
  const s=surface("review"),a=s.picker("a"),b=s.picker("b");a.open();b.open();assert.equal(a.isOpen(),false);assert.equal(b.isOpen(),true);
  const event=s.fireDocument("keydown",{key:"Escape",target:b.list});assert.equal(b.isOpen(),false);assert.equal(event.prevented,true);assert.equal(b.button.focusCount,1);
  a.open();a.container.hidden=true;s.mutate();assert.equal(a.isOpen(),false);assert.equal(s.observers.some(row=>row.active),false);
  b.open();b.container.remove();s.mutate();assert.equal(b.isOpen(),false);assert.equal(s.observers.some(row=>row.active),false);
  for(let index=0;index<20;index++){const current=s.picker("dynamic"+index);current.open();current.container.remove();s.mutate();}
  assert.equal(s.docListeners.get("pointerdown").length,1);assert.equal(s.docListeners.get("scroll").length,1);assert.equal(s.listeners.get("resize").length,1);
  const messages=s.listeners.get("message").length;vm.runInContext(source,s.context);
  assert.equal(s.docListeners.get("pointerdown").length,1);assert.equal(s.listeners.get("message").length,messages);
});

test("parent heading/settings/footer closes every child picker; child inside gestures preserve its picker",()=>{
  const parent=surface("studio"),review=surface("review"),reference=surface("reference-preview");
  const reviewLink=attach(parent,review),referenceLink=attach(parent,reference);
  const rp=review.picker(),ref=reference.picker(),history=parent.picker("history");
  for(const name of ["heading","settings","footer"]){
    rp.open();ref.open();const target=new parent.Element("button");target.id=name;parent.body.append(target);
    const event=parent.fireDocument("pointerdown",{target});assert.equal(rp.isOpen(),false);assert.equal(ref.isOpen(),false);assert.equal(event.prevented,undefined);
  }
  history.open();rp.open();ref.open();
  review.fireDocument("pointerdown",{target:rp.list});assert.equal(rp.isOpen(),true);assert.equal(ref.isOpen(),false);assert.equal(history.isOpen(),false);
  review.fireDocument("scroll",{target:rp.list});assert.equal(rp.isOpen(),true);
  assert.ok(reviewLink.sent.every(row=>row.origin===parent.context.location.origin));
  assert.ok(referenceLink.sent.every(row=>row.origin===parent.context.location.origin));
});

test("only the real same-origin parent or current same-origin iframe may broadcast closes",()=>{
  const parent=surface("studio"),child=surface("review"),link=attach(parent,child),p=child.picker();p.open();
  child.fireWindow("message",{data:{type:"koww:close-pickers"},source:link.parentProxy,origin:"https://evil.test"});assert.equal(p.isOpen(),true);
  child.fireWindow("message",{data:{type:"koww:close-pickers"},source:{},origin:child.context.location.origin});assert.equal(p.isOpen(),true);
  child.fireWindow("message",{data:{type:"koww:other-message"},source:link.parentProxy,origin:child.context.location.origin});assert.equal(p.isOpen(),true);
  const history=parent.picker("history");history.open();
  parent.fireWindow("message",{data:{type:"koww:close-pickers"},source:{},origin:parent.context.location.origin});assert.equal(history.isOpen(),true);
  link.frame.setAttribute("src","https://evil.test/frame");
  parent.fireWindow("message",{data:{type:"koww:close-pickers"},source:link.childProxy,origin:parent.context.location.origin});assert.equal(history.isOpen(),true);
  link.frame.setAttribute("src",child.context.location.href);link.frame.remove();
  parent.fireWindow("message",{data:{type:"koww:close-pickers"},source:link.childProxy,origin:parent.context.location.origin});assert.equal(history.isOpen(),true);
  child.fireWindow("message",{data:{type:"koww:close-pickers"},source:link.parentProxy,origin:child.context.location.origin});assert.equal(p.isOpen(),false);
});

test("a pending reference layout reveal stays closed after outside close, and pagehide releases registry",()=>{
  const s=surface("reference-preview"),changes=[];
  const p=s.picker("reference",(open,request)=>changes.push({open,request}));p.open();const request=changes[0].request;
  assert.equal(p.popup.hidden,true);s.fireDocument("pointerdown",{target:s.body});p.handle.reveal(request);
  assert.equal(p.isOpen(),false);assert.equal(p.popup.hidden,true);assert.equal(changes.at(-1).open,false);
  p.open();p.handle.reveal(changes.at(-1).request);assert.equal(p.popup.hidden,false);
  s.fireWindow("pagehide");assert.equal(p.isOpen(),false);assert.equal(s.observers.some(row=>row.active),false);
});

function geometryPicker(height,rect,count=50,selected=0,onOpenChange) {
  const s=surface("review");s.context.innerHeight=height;s.geometry.rect={...s.geometry.rect,...rect};
  const p=s.picker("geometry",onOpenChange);
  p.handle.setItems(Array.from({length:count},(_,i)=>({label:`목소리 ${i+1}`})),selected);
  Object.defineProperty(p.list,"clientHeight",{get(){return Math.min(count*36+8,parseFloat(p.list.style.maxHeight)||0);}});
  for(const [i,row]of p.list.children.entries())Object.defineProperty(row,"offsetTop",{get:()=>4+i*36});
  const bounds=()=>({top:parseFloat(p.popup.style.top),height:parseFloat(p.popup.style.maxHeight),listHeight:parseFloat(p.list.style.maxHeight)});
  return {s,p,bounds};
}

test("long lists use 360px while short lists retain natural height and 36px rows",()=>{
  const large=geometryPicker(1080,{top:200,bottom:240});large.p.open();
  assert.deepEqual(large.bounds(),{top:239,height:360,listHeight:358});assert.equal(large.p.popup.getAttribute("data-side"),"down");
  const short=geometryPicker(1080,{top:200,bottom:240},2);short.p.open();assert.equal(short.bounds().height,82);assert.equal(short.bounds().listHeight,80);
  const css=fs.readFileSync(path.join(__dirname,"../korean_openwakeword/resources/audio_listening.css"),"utf8");
  assert.match(css,/\.voice-picker-popup\{[^}]*box-sizing:border-box[^}]*max-height:360px/);
  assert.match(css,/\.voice-picker-list\{[^}]*box-sizing:border-box[^}]*max-height:358px/);
  assert.match(css,/\.voice-picker-option\{[^}]*height:36px/);
});

test("placement chooses available side and stays within viewport margins including clipped anchors",()=>{
  for(const [height,rect,side,expected]of [
    [700,{top:500,bottom:540},"up",360],
    [180,{top:70,bottom:110},"down",63],
    [170,{top:20,bottom:60},"down",103],
    [200,{top:-100,bottom:-60},"down",184],
    [200,{top:240,bottom:280},"up",184],
  ]){
    const view=geometryPicker(height,rect);view.p.open();const bounds=view.bounds();
    assert.equal(view.p.popup.getAttribute("data-side"),side);assert.equal(bounds.height,expected);
    assert.ok(bounds.top>=8);assert.ok(bounds.top+bounds.height<=height-8);
  }
});

test("selected item remains visible through open resize and close/reopen",()=>{
  const view=geometryPicker(1080,{top:200,bottom:240},50,49);view.p.open();
  function selectedVisible(){const row=view.p.list.children[49];assert.ok(row.offsetTop>=view.p.list.scrollTop);assert.ok(row.offsetTop+36<=view.p.list.scrollTop+view.p.list.clientHeight);}
  selectedVisible();
  view.s.context.innerHeight=300;view.s.geometry.rect={...view.s.geometry.rect,top:240,bottom:280};view.s.fireWindow("resize");
  assert.equal(view.p.popup.getAttribute("data-side"),"up");assert.equal(view.bounds().height,233);selectedVisible();
  view.p.list.fire("keydown",{key:"Home",preventDefault(){}});assert.equal(view.p.list.scrollTop,4);
  view.p.handle.close();view.s.context.innerHeight=1080;view.s.geometry.rect={...view.s.geometry.rect,top:200,bottom:240};view.p.open();
  assert.equal(view.bounds().height,360);selectedVisible();
});

test("parent-expanded reveal measures fresh geometry and rejects stale request IDs",()=>{
  const changes=[],view=geometryPicker(170,{top:20,bottom:60},50,49,(open,request)=>changes.push({open,request}));
  view.p.open();const first=changes.at(-1).request;assert.equal(view.p.popup.hidden,true);
  view.p.handle.reveal(first+1);assert.equal(view.p.popup.hidden,true);
  view.s.context.innerHeight=900;view.s.geometry.rect={...view.s.geometry.rect,top:200,bottom:240};
  view.p.handle.reveal(first);assert.equal(view.p.popup.hidden,false);assert.equal(view.bounds().height,360);
  view.s.fireDocument("pointerdown",{target:view.s.body});view.p.handle.reveal(first);assert.equal(view.p.popup.hidden,true);
  view.s.context.innerHeight=170;view.s.geometry.rect={...view.s.geometry.rect,top:20,bottom:60};view.p.open();
  const second=changes.at(-1).request;assert.ok(second>first);view.p.handle.reveal(first);assert.equal(view.p.popup.hidden,true);
  view.s.context.innerHeight=900;view.s.geometry.rect={...view.s.geometry.rect,top:200,bottom:240};view.p.handle.reveal(second);
  assert.equal(view.p.popup.hidden,false);assert.equal(view.bounds().height,360);
});

console.log(`Shared picker outside-close: ${passed} tests passed`);
