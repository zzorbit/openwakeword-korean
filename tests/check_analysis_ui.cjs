"use strict";
const assert = require("node:assert/strict"), fs = require("node:fs"), path = require("node:path"), vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname,"../korean_openwakeword/resources/analysis.js"),"utf8");
const flush = async () => { for(let i=0;i<15;i++) await new Promise(resolve=>setImmediate(resolve)); };
const tests=[];
async function test(name,action) { await action();tests.push(name); }
function sample() { return {token:"local-token",wakeword:"보라야",status:"idle",selection_allowed:true,
  runtime:{threshold:.986,consecutive_frames:2},default_candidate:"mean",
  candidate_definitions:[{id:"member1",kind:"single"},{id:"member2",kind:"single"},{id:"mean",kind:"ensemble"}],
  models:[{id:"member2",clean:{recall:1},noise_cell_mean:.95,negative:{fa_per_hour:.1}},{id:"member1",clean:{recall:.9},negative:{}},{id:"mean",clean:{recall:.98},noise_cell_mean:.92,negative:{fa_per_hour:0}}],
  adoptions:{member2:{passed:true},member1:{passed:false},mean:{passed:false}},selection:null}; }
function browser(initial=sample()) {
  class Element {
    constructor(tag="div") {this.tagName=tag;this.textContent="";this.value="";this.children=[];this.handlers={};this.hidden=false;this.disabled=false;this.attributes={};}
    append(...children){this.children.push(...children);} replaceChildren(...children){this.children=children;}
    addEventListener(name,handler){this.handlers[name]=handler;} setAttribute(name,value){this.attributes[name]=value;}
  }
  const elements=new Map(),calls=[],listeners={},intervals=[];
  const ids=["error","title","active","original-runtime","candidate","models","thresholds","consecutive","job","comparisons","comparison","select-setting","saved","saved-model","saved-performance","directory","download","microphone","select","compare"];
  ids.forEach(id=>elements.set(id,new Element()));
  const state={view:initial,fail:false,hold:false};
  function find(node){if(node.type==="radio"&&node.checked)return node;for(const child of node.children){const found=find(child);if(found)return found;}return null;}
  const document={hidden:false,getElementById:id=>elements.get(id),createElement:tag=>new Element(tag),addEventListener:(name,handler)=>listeners[name]=handler,querySelector:()=>find(elements.get("comparisons"))};
  const fetch=async (url,options={})=>{calls.push({url,options});if(state.fail)throw new Error("network interrupted");if(url==="api/status")return {ok:true,json:async()=>structuredClone(state.view)};state.view={...state.view,status:"running",action:url.endsWith("compare")?"compare":"select"};return {ok:true,json:async()=>({status:"running"})};};
  const context=vm.createContext({document,fetch,AbortController,location:{pathname:"/analysis/fixture/"},setTimeout,clearTimeout,setInterval:handler=>{intervals.push(handler);return 1;},clearInterval:()=>{}});
  context.window=context;context.addEventListener=(name,handler)=>listeners[name]=handler;
  vm.runInContext(source,context);
  return {state,elements,calls,intervals,listeners,click:id=>elements.get(id).handlers.click(),refresh:async()=>{await intervals[0]();await flush();}};
}
(async()=>{
  await test("opening selection shows frozen model identities and never launches comparison or selection",async()=>{
    const b=browser();await flush();assert.equal(b.calls.length,1);assert.equal(b.calls[0].url,"api/status");
    assert.equal(b.elements.get("models").children[0].children[0].textContent,"모델 2");
    assert.match(b.elements.get("active").textContent,/앙상블/);assert.equal(b.elements.get("thresholds").value,"0.986");
    assert.equal(b.elements.get("candidate").value,"mean");
  });
  await test("explicit selection keeps original measured runtime and uses the local CSRF token",async()=>{
    const b=browser();await flush();b.elements.get("candidate").value="member2";b.click("select");await flush();
    const call=b.calls.find(call=>call.url==="api/select");assert.equal(call.options.headers["X-Koww-Token"],"local-token");
    assert.deepEqual(JSON.parse(call.options.body),{candidate:"member2",thresholds:[.986],consecutive:[2]});
    assert.equal(b.elements.get("select").disabled,true);assert.match(b.elements.get("job").textContent,/별도 실행 묶음/);
  });
  await test("invalid, blank and oversized comparison choices never submit",async()=>{
    for(const value of ["", "NaN", "1", ".5,", ".1,.2,.3,.4,.5,.6,.7,.8"]){
      const b=browser();await flush();b.elements.get("thresholds").value=value;b.click("compare");await flush();
      assert.equal(b.calls.some(call=>call.options.method==="POST"),false);assert.equal(b.elements.get("error").hidden,false);
    }
  });
  await test("comparison is explicit, bounded and does not export or apply a model",async()=>{
    const b=browser();await flush();b.elements.get("thresholds").value=".7,.8";b.elements.get("consecutive").value="1,2";b.click("compare");await flush();
    const call=b.calls.find(call=>call.url==="api/compare");assert.deepEqual(JSON.parse(call.options.body),{candidate:"mean",thresholds:[.7,.8],consecutive:[1,2]});
    assert.equal(b.calls.some(call=>call.url==="api/select"),false);assert.equal(b.elements.get("select-setting").disabled,true);
  });
  await test("cached comparison candidate object binds the selected exact configuration",async()=>{
    const s=sample();s.status="completed";s.comparison_id="a".repeat(32);s.comparison={candidate:{id:"member2"},comparisons:[{runtime:{threshold:.7,consecutive_frames:1},adoption:{passed:true},models:[s.models[0]]}]};
    const b=browser(s);await flush();assert.equal(b.elements.get("comparison").hidden,false);
    assert.equal(b.elements.get("comparisons").children[0].children[3].textContent,"100.00%");
    b.click("select-setting");await flush();const call=b.calls.find(call=>call.url==="api/select");
    assert.deepEqual(JSON.parse(call.options.body),{candidate:"member2",thresholds:[.7],consecutive:[1],comparison_id:s.comparison_id});
  });
  await test("a completed selection exposes the versioned bundle and microphone link without changing frozen metrics",async()=>{
    const s=sample();s.status="completed";s.selection={candidate:"member2",runtime:{threshold:.7,consecutive_frames:1}};s.selected={candidate:"member2",runtime:s.selection.runtime,bundle:"F:/workspace/project/.studio/analysis/jobs/version/bundle"};s.download_url="/analysis/fixture/download/version";
    const b=browser(s);await flush();assert.equal(b.elements.get("saved").hidden,false);assert.match(b.elements.get("active").textContent,/모델 2.*0.7/);
    assert.equal(b.elements.get("thresholds").value,"0.7");assert.match(b.elements.get("microphone").href,/mic\/fixture\/$/);
    assert.equal(b.elements.get("models").children[0].children[1].textContent,"100.00%");
  });
  await test("a finished selection clears old comparison buttons so stale settings cannot be submitted",async()=>{
    const s=sample();s.status="completed";s.comparison_id="b".repeat(32);s.comparison={candidate:{id:"member2"},comparisons:[{runtime:{threshold:.7,consecutive_frames:1},models:[s.models[0]]}]};
    const b=browser(s);await flush();assert.equal(b.elements.get("select-setting").disabled,false);
    b.state.view=sample();b.state.view.status="completed";await b.refresh();
    assert.equal(b.elements.get("select-setting").disabled,true);assert.equal(b.elements.get("comparison").hidden,true);
    b.click("select-setting");await flush();assert.equal(b.calls.some(call=>call.url==="api/select"),false);
  });
  await test("a changed selected runtime shows its own measured statistics separately from the frozen comparison",async()=>{
    const s=sample();s.status="completed";s.selection={candidate:"member2",runtime:{threshold:.7,consecutive_frames:1}};
    s.selected={candidate:"member2",runtime:s.selection.runtime,bundle:"F:/workspace/selected",summary:{clean:{recall:.88},noise_cell_mean:.81,negative:{fa_per_hour:.4}}};
    const b=browser(s);await flush();
    assert.match(b.elements.get("active").textContent,/0\.7.*연속 1회/);
    assert.match(b.elements.get("original-runtime").textContent,/기존 평가 설정.*0\.986.*연속 2회/);
    assert.equal(b.elements.get("models").children[0].children[1].textContent,"100.00%");
    assert.match(b.elements.get("saved-performance").textContent,/88\.00%.*81\.00%.*0\.400회/);
    assert.doesNotMatch(b.elements.get("saved-performance").textContent,/100\.00%|95\.00%/);
  });
  await test("legacy selected bundles without their own summary never inherit original model statistics",async()=>{
    const s=sample();s.status="completed";s.selection={candidate:"member2",runtime:{threshold:.7,consecutive_frames:1}};
    s.selected={candidate:"member2",runtime:s.selection.runtime,bundle:"F:/workspace/selected"};
    const b=browser(s);await flush();
    assert.equal(b.elements.get("saved-performance").textContent,"해당 설정의 비교 기록을 확인하세요.");
    assert.doesNotMatch(b.elements.get("saved-performance").textContent,/%|시간당/);
  });
  await test("Final lock disables all mutation controls",async()=>{
    const s=sample();s.selection_allowed=false;const b=browser(s);await flush();
    for(const id of ["select","compare","select-setting","candidate","thresholds","consecutive"])assert.equal(b.elements.get(id).disabled,true);
  });
  await test("connection loss retains the last table and cannot fabricate a completed result",async()=>{
    const b=browser();await flush();b.state.fail=true;await b.refresh();assert.equal(b.elements.get("models").children.length,3);assert.match(b.elements.get("error").textContent,/network/);
    b.state.fail=false;await b.refresh();assert.equal(b.elements.get("error").hidden,true);
  });
  process.stdout.write(`${tests.length} model selection UI tests passed\n`);
})().catch(error=>{console.error(error);process.exitCode=1;});
