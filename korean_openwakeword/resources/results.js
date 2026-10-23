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
let queue=[],index=0,continuous=false,token=0;
const audio=$("audio");
const opt=(select,value,text)=>{const o=document.createElement("option");o.value=value;o.textContent=text;select.append(o);};
const levels=[...new Set(clips.filter(c=>c.kind==="positive"&&c.category!=="clean").map(c=>c.snr_db))].sort((a,b)=>b-a);
levels.forEach(s=>opt($("snr"),s,ui.snrLabel(s)));
if(levels.includes(10))$("snr").value="10";
for(const option of $("reverb").options)option.disabled=!clips.some(c=>c.kind==="positive"&&(c.reverb_condition||"none")===option.value);
function rebuild(){continuous=false;token++;audio.pause();queue=clips.filter(c=>c.kind==="positive"&&c.category!=="clean"&&Number(c.snr_db)===Number($("snr").value)&&($("category").value==="all"||c.category===$("category").value)&&(c.reverb_condition||"none")===$("reverb").value);queue.sort((a,b)=>(a.original_order||0)-(b.original_order||0)||String(a.original_id).localeCompare(String(b.original_id))||["environment","music","competing_speech"].indexOf(a.category)-["environment","music","competing_speech"].indexOf(b.category)||(a.noise_section||0)-(b.noise_section||0));index=0;$("item").replaceChildren();$("original").replaceChildren();queue.forEach((c,i)=>opt($("item"),i,`${i+1} · 음성 ${c.original_order||c.original_id} · ${labels[c.category]} · 구간 ${c.noise_section||1}`));const seen=new Set();queue.forEach((c,i)=>{const key=c.original_id||c.sample_id;if(!seen.has(key)){seen.add(key);opt($("original"),i,`음성 ${c.original_order||key}`);}});if(queue.length)show(queue[0]);else $("clipTitle").textContent="선택한 조건으로 시험한 오디오가 없습니다.";progress();}
function progress(){$("queue").textContent=queue.length?`${(index+1).toLocaleString("ko-KR")} / ${queue.length.toLocaleString("ko-KR")}개`:"목록 없음";$("prev").disabled=index<=0;$("next").disabled=index>=queue.length-1;$("item").value=String(index);}
function show(c){audio.src=c.path;$("clipTitle").textContent=ui.clipLabel(c);const background=evaluationBackgroundTypeText(c);$("clipInfo").textContent=`${c.duration_s.toFixed(2)}초${c.calls?.length?` · 호출어 위치 ${c.calls[0].start_s.toFixed(2)}–${c.calls[0].end_s.toFixed(2)}초`:" · 호출어 없음"}${background?` · 배경: ${background}`:""}`;$("download").href=c.path;$("download").download="";$("hash").textContent=`SHA256 ${c.sha256}`;$("sample-id").textContent=`파일 이름: ${c.sample_id}`;ui.renderResponses(report,c);}
async function playCurrent(resume=false){if(!queue.length)return;const mine=++token;if(!resume)show(queue[index]);progress();try{await audio.play();if(mine===token)$("playState").textContent="재생 중 · 마지막 항목에서 종료합니다.";}catch(e){continuous=false;$("playState").textContent=`재생 실패: ${e.message}. 이어 듣기를 눌러 다시 시작하세요.`;}}
$("start").onclick=()=>{index=0;continuous=true;playCurrent();};$("resume").onclick=()=>{continuous=true;playCurrent(true);};$("pause").onclick=()=>{continuous=false;token++;audio.pause();$("playState").textContent="일시정지";};
audio.onended=()=>{if(!continuous)return;if(index+1<queue.length){index++;playCurrent();}else{continuous=false;$("playState").textContent="선택한 목록을 모두 들었습니다.";}};
audio.onerror=()=>{continuous=false;$("playState").textContent="오디오를 불러올 수 없습니다. 파일 경로와 서버 상태를 확인하세요.";};
$("prev").onclick=()=>{if(index>0){index--;continuous?playCurrent():(show(queue[index]),progress());}};$("next").onclick=()=>{if(index+1<queue.length){index++;continuous?playCurrent():(show(queue[index]),progress());}};
for(const id of ["item","original"])$(id).onchange=()=>{index=Number($(id).value);continuous?playCurrent():(show(queue[index]),progress());};
$("snr").onchange=rebuild;$("category").onchange=rebuild;$("reverb").onchange=rebuild;
clips.filter(c=>c.category==="clean"||c.kind==="negative").forEach(c=>opt($("extra"),c.sample_id,ui.clipLabel(c)));
$("extra").onchange=()=>{const c=clips.find(c=>c.sample_id===$("extra").value);if(c){continuous=false;token++;audio.pause();show(c);$("playState").textContent="추가 자료 · 재생 버튼을 눌러 들으세요.";}};
rebuild();
