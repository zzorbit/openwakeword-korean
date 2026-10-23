"use strict";
(() => {
  const $ = id => document.getElementById(id);
  let view = null, pending = false, timer = null, comparisonKey = "", settings = [], refreshing = false, fetchFailed = false;
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const percent = value => finite(value) ? `${(value * 100).toFixed(2)}%` : "결과 없음";
  const rate = value => finite(value) ? `${value.toFixed(3)}회` : "결과 없음";
  function label(id) {
    const definitions = view?.candidate_definitions || [];
    if (definitions.find(row => row.id === id)?.kind === "ensemble") return "앙상블";
    const index = definitions.filter(row => row.kind === "single").findIndex(row => row.id === id);
    return index >= 0 ? `모델 ${index + 1}` : "모델";
  }
  function comparisonCandidate() { return view.comparison?.selected_candidate?.id || view.comparison?.candidate?.id || view.comparison?.candidate; }
  function error(text = "") { $("error").textContent = text; $("error").hidden = !text; }
  async function api(path, options = {}) {
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(path, {...options, signal:controller.signal, cache:"no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data?.error || "작업을 처리하지 못했습니다.");
      return data;
    } catch (e) { if (e.name === "AbortError") throw new Error("상태 확인이 늦어지고 있습니다. 실행 중인 작업은 유지됩니다."); throw e; }
    finally { clearTimeout(timeout); }
  }
  function cell(row, text) { const td=document.createElement("td"); td.textContent=String(text); row.append(td); return td; }
  function controls() {
    const busy = pending || view?.status === "running", locked = view && !view.selection_allowed;
    for (const id of ["candidate","select","compare","thresholds","consecutive","select-setting"]) $(id).disabled = !view || busy || locked || id === "select-setting" && !settings.length;
  }
  function parse(id, integers, maximum) {
    const values = $(id).value.split(",").map(value => value.trim());
    if (!values.length || values.length > maximum || values.some(value => !value || !Number.isFinite(Number(value)))) throw new Error("쉼표로 구분한 숫자를 확인하세요.");
    const numbers = values.map(Number);
    if (numbers.some(value => integers ? !Number.isInteger(value) || value < 1 || value > 10 : value <= 0 || value >= 1)) throw new Error(integers ? "연속 감지 수는 1~10의 정수입니다." : "감지 기준은 0과 1 사이의 숫자입니다.");
    return [...new Set(numbers)];
  }
  function draw() {
    $("title").textContent = `‘${view.wakeword}’ 모델 비교`;
    const candidate = view.selection?.candidate || view.default_candidate;
    const runtime = view.selection?.runtime || view.runtime;
    $("active").textContent = `현재 시험 모델: ${label(candidate)} · 감지 기준 ${runtime.threshold} · 연속 ${runtime.consecutive_frames}회`;
    $("original-runtime").textContent = `기존 평가 설정: 감지 기준 ${view.runtime.threshold} · 연속 ${view.runtime.consecutive_frames}회`;
    const previous = $("candidate").value;
    $("candidate").replaceChildren(); $("models").replaceChildren();
    for (const model of view.models) {
      const option = document.createElement("option"); option.value = model.id; option.textContent = label(model.id); $("candidate").append(option);
      const row=document.createElement("tr");
      [label(model.id),percent(model.clean?.recall),percent(model.noise_cell_mean),rate(model.negative?.fa_per_hour), view.adoptions[model.id]?.passed ? "충족" : "미달"].forEach(value=>cell(row,value));
      $("models").append(row);
    }
    $("candidate").value = view.models.some(row=>row.id === previous) ? previous : candidate;
    if (!$("thresholds").value) $("thresholds").value = String(runtime.threshold);
    if (!$("consecutive").value) $("consecutive").value = String(runtime.consecutive_frames);
    const busy = view.status === "running";
    $("job").textContent = busy ? (view.action === "select" ? "선택한 모델을 별도 실행 묶음으로 저장하고 있습니다." : "저장된 점수로 비교하고 있습니다. 이 창을 닫아도 작업은 계속됩니다.") : view.status === "interrupted" ? "분석이 중단됐습니다. 기존 결과를 보존했으며 다시 실행할 수 있습니다." : "";
    if (busy && view.action === "compare" && view.progress) $("job").textContent = `민감도 비교 ${view.progress.done} / ${view.progress.total}개 설정 계산 완료${view.progress.done === view.progress.total ? " · 기록 정리 중" : ""}`;
    if (view.error) error(view.error);
    if (!view.comparison) {
      settings=[]; comparisonKey=""; $("comparisons").replaceChildren(); $("comparison").hidden=true;
    }
    if (view.comparison && comparisonKey !== view.comparison_id) {
      comparisonKey = view.comparison_id; settings = view.comparison.comparisons || [];
      $("comparisons").replaceChildren();
      settings.forEach((value,index)=>{
        const row=document.createElement("tr"), td=document.createElement("td"), radio=document.createElement("input");
        radio.type="radio"; radio.name="configuration"; radio.value=String(index); radio.checked=index === 0; radio.setAttribute("aria-label",`설정 ${index+1} 선택`); td.append(radio); row.append(td);
        const model=value.models.find(item=>item.id === comparisonCandidate());
        [String(value.runtime.threshold),String(value.runtime.consecutive_frames),percent(model?.clean?.recall),percent(model?.noise_cell_mean),rate(model?.negative?.fa_per_hour),value.adoption?.passed ? "충족" : "미달"].forEach(text=>cell(row,text));
        $("comparisons").append(row);
      });
      $("comparison").hidden = !settings.length;
    }
    $("saved").hidden = !view.selected;
    if (view.selected) {
      $("saved-model").textContent = `${label(view.selected.candidate)} · 감지 기준 ${view.selected.runtime.threshold} · 연속 ${view.selected.runtime.consecutive_frames}회`;
      const summary = view.selected.summary;
      $("saved-performance").textContent = summary
        ? `이 설정의 시험 결과: 조용할 때 ${percent(summary.clean?.recall)} · 소음 평균 ${percent(summary.noise_cell_mean)} · 시간당 오작동 ${rate(summary.negative?.fa_per_hour)}`
        : "해당 설정의 비교 기록을 확인하세요.";
      $("directory").textContent = view.selected.bundle;
      $("download").href = view.download_url;
      $("microphone").href = "../../mic/" + location.pathname.split("/").filter(Boolean)[1] + "/";
    }
    controls();
  }
  async function refresh() {
    if (refreshing) return;
    refreshing=true;
    try { view=await api("api/status"); if (fetchFailed) error(""); fetchFailed=false; draw(); } catch(e) { fetchFailed=true; error(e.message); }
    finally { refreshing=false; }
  }
  async function submit(action, payload) {
    if (pending || view?.status === "running") return;
    pending=true; controls(); error("");
    try {
      await api(`api/${action}`,{method:"POST",headers:{"Content-Type":"application/json","X-Koww-Token":view.token},body:JSON.stringify(payload)});
      if (action === "compare") { settings=[]; $("comparison").hidden=true; comparisonKey=""; }
      await refresh();
    } catch(e) { error(e.message); }
    finally { pending=false; controls(); }
  }
  $("compare").addEventListener("click",()=>{ try { submit("compare",{candidate:$("candidate").value,thresholds:parse("thresholds",false,7),consecutive:parse("consecutive",true,3)}); } catch(e) {error(e.message);} });
  $("select").addEventListener("click",()=>submit("select",{candidate:$("candidate").value,thresholds:[view.runtime.threshold],consecutive:[view.runtime.consecutive_frames]}));
  $("select-setting").addEventListener("click",()=>{
    const selected=document.querySelector('input[name="configuration"]:checked');
    const setting=selected && settings[Number(selected.value)]; if (!setting) return;
    submit("select",{candidate:comparisonCandidate(),thresholds:[setting.runtime.threshold],consecutive:[setting.runtime.consecutive_frames],comparison_id:view.comparison_id});
  });
  document.addEventListener("visibilitychange",()=>{if (!document.hidden) refresh();});
  window.addEventListener("pagehide",()=>clearInterval(timer));
  refresh(); timer=setInterval(()=>{if (!document.hidden && !pending) refresh();},3000);
})();
