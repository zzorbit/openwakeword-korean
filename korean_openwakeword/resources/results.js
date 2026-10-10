/* Portable frozen result viewer: no API, no inference, no hidden audio gain. */
"use strict";
const {report,clips}=window.KOWW_RESULTS;
const $=id=>document.getElementById(id), pct=x=>x==null?"미측정":`${(x*100).toFixed(2)}%`, num=(x,n=4)=>x==null?"미측정":Number(x).toFixed(n);
const labels={clean:"Clean",environment:"환경소음",music:"음악·전자음",competing_speech:"한국어 대화·방송"};
const cell=(tr,value)=>{const td=document.createElement("td");td.textContent=value;tr.append(td);};
$("title").textContent=`${report.wakeword} · 학습 결과`;
$("scope").textContent=`${report.stage==="final"?"동결 모델 Final 평가":"개발 평가"} · ${report.sessions}개 오디오${report.smoke_test_only?" · 기능 시험용 결과":""}`;
for(const m of report.models){const tr=document.createElement("tr");if(m.kind==="ensemble")tr.className="ensemble";[m.kind==="ensemble"?"앙상블":m.rank,m.id,`${m.primary.successes}/${m.primary.calls} · ${pct(m.primary.recall)}`,pct(m.clean.recall),pct(m.noise_cell_mean),`${m.negative.events}회 / ${m.negative.hours.toFixed(2)}h · ${num(m.negative.fa_per_hour,3)}/h`,m.primary.latency_p95_s==null?"미측정":`${Math.round(m.primary.latency_p95_s*1000)}ms`].forEach(v=>cell(tr,v));$("summary").tBodies[0].append(tr);
for(const [k,v] of Object.entries(m.conditions)){const r=document.createElement("tr");[m.id,k,`${v.successes}/${v.calls}`,pct(v.recall)].forEach(x=>cell(r,x));$("conditions").tBodies[0].append(r);}}
$("denominators").textContent="소음 평균: 실제 측정한 환경·음악·한국어 음성 × SNR20/15/10/5 조건의 균등 평균. 같은 비호출 자료를 여러 모델이 평가해도 독립 시간을 합산하지 않습니다.";
let queue=[],index=0,continuous=false,token=0;
const audio=$("audio");
const opt=(select,value,text)=>{const o=document.createElement("option");o.value=value;o.textContent=text;select.append(o);};
const levels=[...new Set(clips.filter(c=>c.kind==="positive"&&c.category!=="clean").map(c=>c.snr_db))].sort((a,b)=>b-a);
levels.forEach(s=>opt($("snr"),s,`SNR ${s} dB${s===0?" · 한계 진단":""}`));
if(levels.includes(10))$("snr").value="10";
function rebuild(){continuous=false;token++;audio.pause();queue=clips.filter(c=>c.kind==="positive"&&c.category!=="clean"&&Number(c.snr_db)===Number($("snr").value)&&($("category").value==="all"||c.category===$("category").value));queue.sort((a,b)=>(a.original_order||0)-(b.original_order||0)||String(a.original_id).localeCompare(String(b.original_id))||["environment","music","competing_speech"].indexOf(a.category)-["environment","music","competing_speech"].indexOf(b.category)||(a.noise_section||0)-(b.noise_section||0));index=0;$("item").replaceChildren();$("original").replaceChildren();queue.forEach((c,i)=>opt($("item"),i,`${i+1} · 원본 ${c.original_order||c.original_id} · ${labels[c.category]} · 구간 ${c.noise_section||1}`));const seen=new Set();queue.forEach((c,i)=>{const key=c.original_id||c.sample_id;if(!seen.has(key)){seen.add(key);opt($("original"),i,`원본 ${c.original_order||key}`);}});if(queue.length)show(queue[0]);else $("clipTitle").textContent="해당 소음 조건의 평가 오디오가 없습니다.";progress();}
function progress(){$("queue").textContent=queue.length?`${index+1} / ${queue.length} · 선택한 SNR 고정 · 남은 오디오 약 ${Math.ceil(queue.slice(index).reduce((s,c)=>s+c.duration_s,0)/60)}분`:"목록 없음";$("prev").disabled=index<=0;$("next").disabled=index>=queue.length-1;$("item").value=String(index);}
function show(c){audio.src=c.path;$("clipTitle").textContent=`${c.sample_id} · ${labels[c.category]||c.category}`;$("clipInfo").textContent=`전체 ${c.duration_s.toFixed(2)}초${c.calls?.length?` · 호출 ${c.calls[0].start_s.toFixed(2)}–${c.calls[0].end_s.toFixed(2)}초`:" · 비호출"}${c.snr_db!=null?` · SNR ${c.snr_db} dB`:""}`;$("download").href=c.path;$("download").download="";$("hash").textContent=`SHA256 ${c.sha256}`;$("responses").tBodies[0].replaceChildren();for(const [id,r]of Object.entries(c.results)){const tr=document.createElement("tr");[id,c.kind==="negative"?`오호출 ${r.false_activations}회`:r.success?"감지":"누락",num(r.max_score),num(r.clean_score_drop),r.latency_s==null?"—":`${Math.round(r.latency_s*1000)}ms`].forEach(v=>cell(tr,v));$("responses").tBodies[0].append(tr);}}
async function playCurrent(resume=false){if(!queue.length)return;const mine=++token;if(!resume)show(queue[index]);progress();try{await audio.play();if(mine===token)$("playState").textContent="재생 중 · 마지막 항목에서 종료합니다.";}catch(e){continuous=false;$("playState").textContent=`재생 실패: ${e.message}. 이어 듣기를 눌러 다시 시작하세요.`;}}
$("start").onclick=()=>{index=0;continuous=true;playCurrent();};$("resume").onclick=()=>{continuous=true;playCurrent(true);};$("pause").onclick=()=>{continuous=false;token++;audio.pause();$("playState").textContent="일시정지";};
audio.onended=()=>{if(!continuous)return;if(index+1<queue.length){index++;playCurrent();}else{continuous=false;$("playState").textContent="선택한 목록을 모두 들었습니다.";}};
audio.onerror=()=>{continuous=false;$("playState").textContent="오디오를 불러올 수 없습니다. 파일 경로와 서버 상태를 확인하세요.";};
$("prev").onclick=()=>{if(index>0){index--;continuous?playCurrent():(show(queue[index]),progress());}};$("next").onclick=()=>{if(index+1<queue.length){index++;continuous?playCurrent():(show(queue[index]),progress());}};
for(const id of ["item","original"])$(id).onchange=()=>{index=Number($(id).value);continuous?playCurrent():(show(queue[index]),progress());};
$("snr").onchange=rebuild;$("category").onchange=rebuild;
clips.filter(c=>c.category==="clean"||c.kind==="negative").forEach(c=>opt($("extra"),c.sample_id,`${c.sample_id} · ${labels[c.category]}`));
$("extra").onchange=()=>{const c=clips.find(c=>c.sample_id===$("extra").value);if(c){continuous=false;token++;audio.pause();show(c);$("playState").textContent="추가 자료 · 재생 버튼을 눌러 들으세요.";}};
rebuild();
