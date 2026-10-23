/* Presentation of frozen results. These helpers never recalculate or alter scores. */
"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const count = value => finite(value) ? value.toLocaleString("ko-KR") : "—";
  const percent = value => finite(value) ? `${(value * 100).toFixed(2)}%` : "결과 없음";
  const categories = ["environment", "music", "competing_speech"];
  const levels = [20, 15, 10, 5, 0];
  const noiseNames = {clean: "조용한 환경", environment: "환경소음", music: "음악·전자음", competing_speech: "다른 사람 말소리"};
  const reverbNames = {none: "울림 추가 없음", moderate: "보통 울림", strong: "강한 울림"};
  function noiseLabel(category) { return noiseNames[category] || "기타 배경"; }
  function reverbLabel(group) { return reverbNames[group] || "기타 울림"; }
  function snrLabel(value) {
    const level = value == null || value === "" ? NaN : Number(value);
    const text = {20: "소음 아주 작음", 15: "소음 작음", 10: "소음 중간", 5: "소음 큼", 0: "목소리와 같은 크기"};
    return `${text[level] || "소음 크기"} · SNR ${Number.isFinite(level) ? level : "—"} dB`;
  }
  function definitions(report) { return Array.isArray(report.candidate_definitions) ? report.candidate_definitions : []; }
  function isEnsemble(report, id) {
    const definition = definitions(report).find(row => row.id === id);
    const model = (report.models || []).find(row => row.id === id);
    const deployed = report.deployment_candidate?.id === id ? report.deployment_candidate : null;
    return (definition?.kind || model?.kind || deployed?.kind || (id === "mean" ? "ensemble" : "single")) === "ensemble";
  }
  function singleIds(report) {
    const declared = definitions(report).filter(row => row.kind === "single").map(row => row.id);
    if (declared.length) return [...new Set(declared)];
    return [...new Set((report.models || []).filter(row => !isEnsemble(report, row.id)).map(row => row.id))]
      .sort((left, right) => String(left).localeCompare(String(right), "en", {numeric: true}));
  }
  function modelLabel(report, id) {
    if (isEnsemble(report, id)) return "앙상블";
    const position = singleIds(report).indexOf(id);
    return position < 0 ? "모델" : `모델 ${position + 1}`;
  }
  function memberCount(report, id) {
    const declared = definitions(report).find(row => row.id === id);
    const deployed = report.deployment_candidate;
    return declared?.members?.length || (deployed?.id === id ? deployed.members?.length : 0) || singleIds(report).length;
  }
  function appendCell(row, text, title) {
    const node = document.createElement("td");
    node.textContent = String(text);
    if (title) node.title = title;
    row.append(node);
    return node;
  }
  function body(id) { return $(id)?.tBodies?.[0]; }
  function statisticText(value) {
    if (value?.calls === 0) return "시험 안 함";
    if (!value || !finite(value.recall)) return "결과 없음";
    return `${percent(value.recall)}${finite(value.successes) && finite(value.calls) ? `\n${count(value.successes)} / ${count(value.calls)}회` : ""}`;
  }
  function reactionTime(value) { return finite(value) ? `${Math.abs(value).toFixed(2)}초${value < 0 ? " 전" : ""}` : "결과 없음"; }
  function reasonText(reason) {
    const text = String(reason);
    if (text.includes("FP duration below")) return "오작동 시험 시간 부족";
    if (text.includes("Clean recall")) return "조용한 환경 성공률";
    if (text.includes("FA/hour")) return "시간당 오작동 횟수";
    if (text.includes("Latency")) return "반응 시간";
    if (text.includes("No evaluated candidate")) return "비교할 모델 없음";
    for (const category of categories) {
      if (!text.includes(category)) continue;
      const match = text.match(/SNR\s*(\d+(?:\.\d+)?)/i);
      return `${noiseLabel(category)} 성공률${match ? ` · SNR ${Number(match[1])} dB` : ""}`;
    }
    return "추가 확인 필요";
  }
  function renderSummary(report) {
    const rows = body("summary"), reasons = $("candidateReasons");
    if (!rows) return;
    rows.replaceChildren(); reasons?.replaceChildren();
    const decisions = report.candidate_adoptions || {};
    for (const model of report.models || []) {
      const row = document.createElement("tr");
      if (isEnsemble(report, model.id)) row.className = "ensemble";
      const decision = decisions[model.id] || (report.adoption?.candidate === model.id ? report.adoption : null);
      const negative = model.negative || {};
      const rate = finite(negative.fa_per_hour) ? `${negative.fa_per_hour.toFixed(3)}회` : "결과 없음";
      const actual = finite(negative.events) && finite(negative.hours) ? `${count(negative.events)}회 / ${negative.hours.toFixed(2)}시간` : "";
      const values = [isEnsemble(report, model.id) ? "—" : model.rank ?? "—", modelLabel(report, model.id),
        statisticText(model.primary), statisticText(model.clean), percent(model.noise_cell_mean),
        `${rate}${actual ? `\n${actual}` : ""}`, reactionTime(model.primary?.latency_p95_s),
        !decision || typeof decision.passed !== "boolean" ? "결과 없음" : decision.passed ? "충족" : "미달"];
      values.forEach((text, index) => appendCell(row, text, index === 6 ? "감지된 호출의 95%가 이 시간 안에 반응했습니다. 호출을 마친 시점이 기준입니다." : null));
      rows.append(row);
      if (decision?.passed === false && reasons) {
        const line = document.createElement("p");
        const items = [...new Set((decision.reasons || []).map(reasonText))];
        line.textContent = `${modelLabel(report, model.id)}: ${items.join(" · ") || "추가 확인 필요"}`;
        reasons.append(line);
      }
    }
    const actualId = report.deployment_candidate?.id || report.selected_candidate || report.adoption?.candidate;
    if ($("candidateStatus")) {
      const size = actualId && isEnsemble(report, actualId) ? memberCount(report, actualId) : 0;
      $("candidateStatus").textContent = actualId
        ? `현재 실행 모델: ${modelLabel(report, actualId)}${size ? ` · 모델 ${size}개 점수 평균` : ""}`
        : "선택한 모델 없음";
    }
    if ($("denominators")) $("denominators").textContent = "종합 성공률은 소음과 목소리가 같은 크기인 시험을 제외합니다. 소음 평균은 울림 없이 소음이 아주 작을 때부터 클 때까지의 평균입니다.";
  }
  function option(select, value, text) {
    const node = document.createElement("option"); node.value = value; node.textContent = text; select.append(node);
  }
  function conditionValue(values, category, level) {
    for (const [key, value] of Object.entries(values || {})) {
      const separator = key.lastIndexOf(":");
      if (key.slice(0, separator) === category && Number(key.slice(separator + 1)) === level) return value;
    }
    return undefined;
  }
  function cleanValue(result) {
    if (result?.clean) return result.clean;
    return Object.entries(result?.conditions || {}).find(([key]) => key.startsWith("clean:"))?.[1];
  }
  function renderConditions(report) {
    const modelSelect = $("condition-model"), reverbSelect = $("condition-reverb"), rows = body("conditions");
    if (!modelSelect || !reverbSelect || !rows) return;
    const models = report.models || [];
    const previous = modelSelect.value;
    modelSelect.replaceChildren();
    models.forEach(model => option(modelSelect, model.id, modelLabel(report, model.id)));
    const selected = report.selected_candidate || report.adoption?.candidate;
    modelSelect.value = models.some(model => model.id === previous) ? previous
      : models.some(model => model.id === selected) ? selected : models[0]?.id || "";
    function draw() {
      const model = models.find(value => value.id === modelSelect.value);
      const previousGroup = reverbSelect.value;
      const groups = model ? [...new Set(["none", ...Object.keys(model.reverb_conditions || {})])] : [];
      reverbSelect.replaceChildren();
      groups.forEach(group => option(reverbSelect, group, reverbLabel(group)));
      reverbSelect.value = groups.includes(previousGroup) ? previousGroup : "none";
      drawValues();
    }
    function drawValues() {
      rows.replaceChildren();
      const model = models.find(value => value.id === modelSelect.value);
      const group = reverbSelect.value || "none";
      const result = group === "none"
        ? {conditions: model?.conditions, clean: model?.clean}
        : model?.reverb_conditions?.[group];
      if ($("condition-clean")) $("condition-clean").textContent = `배경 소음 없이: ${statisticText(cleanValue(result))}`;
      for (const category of categories) {
        const row = document.createElement("tr"); appendCell(row, noiseLabel(category));
        for (const level of levels) appendCell(row, statisticText(conditionValue(result?.conditions, category, level)));
        rows.append(row);
      }
      modelSelect.disabled = !models.length; reverbSelect.disabled = !models.length;
    }
    modelSelect.onchange = draw; reverbSelect.onchange = drawValues; draw();
  }
  function renderResponses(report, clip) {
    const rows = body("responses"), scores = body("scores");
    if (!rows) return;
    rows.replaceChildren(); scores?.replaceChildren();
    for (const [id, result] of Object.entries(clip.results || {})) {
      const row = document.createElement("tr");
      const response = clip.kind === "negative"
        ? finite(result.false_activations) ? `오작동 ${count(result.false_activations)}회` : "결과 없음"
        : typeof result.success === "boolean" ? result.success ? "감지" : "놓침" : "결과 없음";
      const latency = clip.kind === "negative" || result.success === false ? "—" : reactionTime(result.latency_s);
      [modelLabel(report, id), response, latency].forEach(value => appendCell(row, value)); rows.append(row);
      if (scores) {
        const detail = document.createElement("tr");
        [modelLabel(report, id), finite(result.max_score) ? result.max_score.toFixed(4) : "결과 없음",
          finite(result.clean_score_drop) ? result.clean_score_drop.toFixed(4) : "결과 없음"]
          .forEach(value => appendCell(detail, value));
        scores.append(detail);
      }
    }
  }
  function clipLabel(clip) {
    if (clip.kind === "negative") return `호출어 없는 음원 · ${noiseLabel(clip.category)}`;
    return `${clip.original_order ? `음성 ${clip.original_order}` : "호출어 음성"} · ${noiseLabel(clip.category)}`;
  }
  const api = {modelLabel, noiseLabel, snrLabel, reverbLabel, renderSummary, renderConditions, renderResponses, clipLabel};
  if (typeof window !== "undefined") window.KowwResultsUI = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
