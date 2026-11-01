/* Portable frozen result viewer: no API, no inference, no hidden audio gain. */
"use strict";
const {report,clips}=window.KOWW_RESULTS;
const ui=window.KowwResultsUI;
const $=id=>document.getElementById(id);
const labels={clean:"Clean",environment:"환경소음",music:"음악·전자음",competing_speech:"다른 사람 말소리"};
$("title").textContent=`${report.wakeword} · 성능 시험 결과`;
$("scope").textContent=evaluationScopeText(report,clips);
$("noiseCoverage").textContent=evaluationNoiseCoverageText(clips);
ui.renderSummary(report);
ui.renderConditions(report);
let queue=[],index=0,continuous=false,token=0,currentSource="main",currentClip=null,playing=false,loading=false,listPage=0;
const audio=$("audio");
function transport(){const active=playing||loading;$("resume").textContent=active?"일시정지":"재생";$("resume").setAttribute("aria-pressed",String(active));$("resume").setAttribute("aria-busy",String(loading));$("resume").disabled=!currentClip&&!queue.length;}
function stop(){continuous=false;token++;loading=false;playing=false;audio.pause();transport();$("playState").textContent="일시정지";}
const opt=(select,value,text)=>{const o=document.createElement("option");o.value=value;o.textContent=text;select.append(o);};
function renderList(){
  const pages=Math.max(1,Math.ceil(queue.length/5));listPage=Math.min(listPage,pages-1);const rows=[];
  for(let position=listPage*5;position<Math.min(queue.length,(listPage+1)*5);position++){
    const clip=queue[position],button=document.createElement("button"),title=document.createElement("span"),badge=document.createElement("span"),label=ui.clipLabel(clip);
    button.type="button";button.className="voice-row";button.setAttribute("data-number",String(position+1));button.setAttribute("aria-current",String(currentSource==="main"&&position===index));button.setAttribute("aria-label",`${position+1}. ${label}`);
    title.className="listen-row-title";title.textContent=label;badge.className="listen-badge";badge.textContent=String(position+1);button.append(title,badge);button.onclick=()=>move(position);rows.push(button);
  }
  $("clip-list").replaceChildren(...rows);$("clip-list-position").textContent=`${listPage+1} / ${pages}`;$("clip-list-prev").disabled=listPage===0;$("clip-list-next").disabled=listPage+1>=pages;
}
const levels=[...new Set(clips.filter(c=>c.kind==="positive"&&c.category!=="clean").map(c=>c.snr_db))].sort((a,b)=>b-a);
levels.forEach(s=>opt($("snr"),s,ui.snrLabel(s)));
if(levels.includes(10))$("snr").value="10";
for(const option of $("reverb").options)option.disabled=!clips.some(c=>c.kind==="positive"&&(c.reverb_condition||"none")===option.value);
function rebuild(){stop();currentSource="main";queue=clips.filter(c=>c.kind==="positive"&&c.category!=="clean"&&Number(c.snr_db)===Number($("snr").value)&&($("category").value==="all"||c.category===$("category").value)&&(c.reverb_condition||"none")===$("reverb").value);queue.sort((a,b)=>(a.original_order||0)-(b.original_order||0)||String(a.original_id).localeCompare(String(b.original_id))||["environment","music","competing_speech"].indexOf(a.category)-["environment","music","competing_speech"].indexOf(b.category)||(a.noise_section||0)-(b.noise_section||0));index=0;$("item").replaceChildren();$("original").replaceChildren();queue.forEach((c,i)=>opt($("item"),i,`${i+1} · 음성 ${c.original_order||c.original_id} · ${labels[c.category]} · 구간 ${c.noise_section||1}`));const seen=new Set();queue.forEach((c,i)=>{const key=c.original_id||c.sample_id;if(!seen.has(key)){seen.add(key);opt($("original"),i,`음성 ${c.original_order||key}`);}});if(queue.length)show(queue[0]);else{currentClip=null;audio.removeAttribute("src");$("responses").tBodies[0].replaceChildren();$("clipTitle").textContent="선택한 조건으로 시험한 오디오가 없습니다.";}$("playState").textContent="";progress();transport();}
function progress(){$("queue").textContent=queue.length?`${(index+1).toLocaleString("ko-KR")} / ${queue.length.toLocaleString("ko-KR")}`:"목록 없음";$("prev").disabled=index<=0;$("next").disabled=index>=queue.length-1;$("item").value=String(index);listPage=Math.floor(index/5);renderList();}
function show(c){currentClip=c;audio.src=c.path;$("clipTitle").textContent=ui.clipLabel(c);const background=evaluationBackgroundTypeText(c);$("clipInfo").textContent=`${c.duration_s.toFixed(2)}초${c.calls?.length?` · 호출어 위치 ${c.calls[0].start_s.toFixed(2)}–${c.calls[0].end_s.toFixed(2)}초`:" · 호출어 없음"}${background?` · 배경: ${background}`:""}`;$("download").href=c.path;$("download").download="";$("hash").textContent=`SHA256 ${c.sha256}`;$("sample-id").textContent=`파일 이름: ${c.sample_id}`;ui.renderResponses(report,c);}
async function playCurrent(resume=false){
  if((!resume&&!queue.length)||(resume&&!currentClip))return;
  const mine=++token;loading=true;transport();
  if(!resume){currentSource="main";show(queue[index]);}
  progress();
  try{await audio.play();if(mine!==token){if(!continuous)audio.pause();return;}loading=false;playing=audio.paused===false;transport();$("playState").textContent="재생 중";}
  catch(e){if(mine!==token)return;continuous=false;loading=false;playing=false;transport();$("playState").textContent=`재생 실패: ${e.message}. 재생을 다시 눌러주세요.`;}
}
function move(position){if(!Number.isInteger(position)||position<0||position>=queue.length)return;index=position;currentSource="main";if(continuous)return playCurrent();token++;audio.pause();playing=false;loading=false;show(queue[index]);progress();transport();$("playState").textContent="";}
$("start").onclick=()=>{index=0;continuous=true;return playCurrent();};
$("resume").onclick=()=>{if(playing||loading||audio.paused===false)return stop();continuous=currentSource==="main";return playCurrent(true);};
$("pause").onclick=stop;
audio.onplay=()=>{playing=true;transport();$("playState").textContent="재생 중";};
audio.onpause=()=>{if(audio.ended)return;playing=false;if(!loading){continuous=false;$("playState").textContent="일시정지";}transport();};
audio.onended=()=>{playing=false;loading=false;if(continuous&&currentSource==="main"&&index+1<queue.length){index++;return playCurrent();}const finished=continuous&&currentSource==="main";continuous=false;transport();$("playState").textContent=finished?"선택한 목록을 모두 들었습니다.":"재생 완료";};
audio.onerror=()=>{continuous=false;token++;loading=false;playing=false;transport();$("playState").textContent="오디오를 불러올 수 없습니다. 파일 경로와 서버 상태를 확인하세요.";};
$("prev").onclick=()=>move(index-1);$("next").onclick=()=>move(index+1);
$("clip-list-prev").onclick=()=>{if(listPage>0){listPage--;renderList();}};$("clip-list-next").onclick=()=>{if((listPage+1)*5<queue.length){listPage++;renderList();}};
for(const id of ["item","original"])$(id).onchange=()=>move(Number($(id).value));
$("snr").onchange=rebuild;$("category").onchange=rebuild;$("reverb").onchange=rebuild;
clips.filter(c=>c.category==="clean"||c.kind==="negative").forEach(c=>opt($("extra"),c.sample_id,ui.clipLabel(c)));
$("extra").onchange=()=>{const c=clips.find(c=>c.sample_id===$("extra").value);if(c){stop();currentSource="extra";show(c);transport();renderList();$("playState").textContent="추가 자료 · 재생 버튼을 눌러 들으세요.";}};
rebuild();
window.KowwListening?.enhance(audio,$("resume").parentElement,{play:$("resume"),previous:$("prev"),next:$("next")});
