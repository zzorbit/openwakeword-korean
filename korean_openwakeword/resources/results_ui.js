/* Presentation of frozen results. These helpers never recalculate or alter scores. */
"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const count = value => finite(value) ? value.toLocaleString("ko-KR") : "—";
  const percent = value => finite(value) ? `${(value * 100).toFixed(2)}%` : "결과 없음";
  const categories = ["environment", "music", "competing_speech"];
  const levels = [20, 15, 10, 5, 0];
  let dashboardView = "models", modelView = "recommendations";
  let compactAudio = null;
  let resultClipPicker;
  function fitAudioControls() {
    if (typeof window === "undefined" || !Number.isFinite(window.innerHeight)) return;
    const compact = window.innerHeight < 520;
    if (compact === compactAudio) return;
    for (const id of ["audio-filter-details", "clip-select-details"]) {
      const detail = $(id); if (detail && compactAudio === null) detail.open = false;
    }
    compactAudio = compact;
  }
  function activateTabs(prefix, names, chosen, focus = false) {
    names.forEach(name => {
      const tab = $(`${prefix}-tab-${name}`), panel = $(`${prefix}-panel-${name}`);
      if (!tab || !panel) return;
      const selected = name === chosen;
      tab.setAttribute?.("aria-selected", String(selected)); tab.tabIndex = selected ? 0 : -1;
      panel.hidden = !selected;
      if (focus && selected) tab.focus?.();
    });
  }
  function dashboard(view, focus = false) {
    if (!["models", "conditions", "audio", "criteria"].includes(view)) return;
    if (view !== "audio") resultClipPicker?.close();
    dashboardView = view; activateTabs("results", ["models", "conditions", "audio", "criteria"], view, focus);
  }
  function modelDashboard(view, focus = false) {
    if (!["recommendations", "evaluation", "targets"].includes(view)) return;
    modelView = view; activateTabs("model", ["recommendations", "evaluation", "targets"], view, focus);
  }
  function bindTabs(prefix, names, select) {
    names.forEach(name => {
      const tab = $(`${prefix}-tab-${name}`);
      if (!tab) return;
      tab.onclick = () => select(name);
      tab.onkeydown = event => {
        const available = names.filter(value => {
          const item = $(`${prefix}-tab-${value}`);
          return item && !item.hidden && !item.disabled;
        });
        const position = available.indexOf(name);
        let next;
        if (event.key === "ArrowRight") next = (position + 1) % available.length;
        if (event.key === "ArrowLeft") next = (position - 1 + available.length) % available.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = available.length - 1;
        if (next === undefined) return;
        event.preventDefault(); select(available[next], true);
      };
    });
  }
  function initDashboard() {
    if (!$("results-tab-models")) return;
    if (typeof window !== "undefined" && window.parent && window.parent !== window) document.body?.classList?.add("embedded");
    bindTabs("results", ["models", "conditions", "audio", "criteria"], dashboard);
    bindTabs("model", ["recommendations", "evaluation", "targets"], modelDashboard);
    dashboard(dashboardView); modelDashboard(modelView);
    for (const view of ["player", "reactions", "settings"]) {
      const tab = $(`audio-view-${view}`);
      if (tab) tab.onclick = () => {
        if (view !== "player") resultClipPicker?.close();
        $("results-panel-audio")?.setAttribute?.("data-audio-view", view);
        for (const name of ["player", "reactions", "settings"]) $(`audio-view-${name}`)?.setAttribute?.("aria-pressed", String(name === view));
      };
    }
  }
  function pauseFromParent(event) {
    if (!window.parent || window.parent === window || !window.location?.origin
        || event.origin !== window.location.origin || event.source !== window.parent
        || event.data?.type !== "koww:pause") return;
    resultClipPicker?.close();
    const pause = $("pause");
    // Reuse the viewer's own pause path so queue state and in-flight autoplay
    // cancellation stay consistent in both static and paged exports.
    if (typeof pause?.onclick === "function") pause.onclick();
    else $("audio")?.pause?.();
  }
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
  function thresholdText(value) {
    if (!finite(value) || value <= 0 || value >= 1) return "결과 없음";
    const rounded = Number(value.toPrecision(6));
    return String(rounded > 0 && rounded < 1 ? rounded : value);
  }
  function renderRecommendations(report) {
    const section = $("runtimeRecommendations"), rows = body("recommendations");
    if (!section || !rows) return;
    const source = report.runtime_recommendations;
    const candidates = source?.schema_version === 1 && Array.isArray(source.candidates) ? source.candidates : [];
    const known = new Set((report.models || []).map(row => row.id));
    const measured = candidates.filter(row => known.has(row.id) && finite(row.runtime?.threshold)
      && row.runtime.threshold > 0 && row.runtime.threshold < 1
      && Number.isInteger(row.runtime.consecutive_frames) && row.runtime.consecutive_frames > 0 && row.metrics);
    rows.replaceChildren(); section.hidden = !measured.length;
    const reasons = $("recommendationReasons"), failures = $("recommendationFailures");
    reasons?.replaceChildren();
    if (failures) failures.hidden = true;
    const baseline = $("baselineRuntime");
    if (baseline) baseline.hidden = !measured.length;
    const recommendationTab = $("model-tab-recommendations");
    if (recommendationTab) recommendationTab.hidden = !measured.length;
    if (!measured.length) {
      if (modelView === "recommendations") modelDashboard("evaluation");
      return;
    }
    for (const candidate of measured) {
      const row = document.createElement("tr"), metric = candidate.metrics;
      if (isEnsemble(report, candidate.id)) row.className = "ensemble";
      const passed = candidate.passed ?? candidate.adoption?.passed;
      const hard = metric.hard_conditions;
      const availability = candidate.recommendation_available;
      const status = availability === false ? "절충 후보" : availability === true ? "추천" : "이전 추천";
      [modelLabel(report, candidate.id), status, thresholdText(candidate.runtime.threshold), `${candidate.runtime.consecutive_frames}회`,
        percent(metric.clean?.recall), hard?.complete === false ? "자료 부족" : percent(hard?.cell_mean),
        hard?.complete === false ? "자료 부족" : percent(hard?.worst_recall), finite(metric.negative?.fa_per_hour)
          ? `${metric.negative.fa_per_hour.toFixed(3)}회` : "결과 없음", passed === true ? "충족" : passed === false ? "미달" : "결과 없음"]
        .forEach(value => appendCell(row, value));
      if (hard?.complete && finite(hard.worst_recall)) {
        row.children[6].title = Object.values(hard.conditions || {}).filter(cell => cell.recall === hard.worst_recall)
          .map(cell => `${reverbLabel(cell.reverb_condition)} · ${noiseLabel(cell.category)}${finite(cell.snr_db) ? ` · SNR ${cell.snr_db} dB` : ""}`).join(" / ");
      }
      rows.append(row);
      if ((passed === false || availability === false) && reasons) {
        const line = document.createElement("p");
        const items = [...new Set([...(candidate.recommendation_reasons || []),
          ...(candidate.adoption?.reasons || [])].map(reasonText))];
        line.textContent = `${modelLabel(report, candidate.id)}${availability === false ? " · 추천 없음" : ""}: ${items.join(" · ") || "추가 확인 필요"}`;
        reasons.append(line);
        if (failures) failures.hidden = false;
      }
    }
    const hardPolicy = measured.some(row => row.metrics.hard_conditions);
    const step = source.search_policy?.threshold_step;
    const faLimit = source.recommendation_criteria?.fa_per_hour;
    const originalFaLimit = source.adoption_criteria?.fa_per_hour ?? report.adoption_criteria?.fa_per_hour;
    const declaredSearch = finite(step) && step > 0 && step < 1 && finite(faLimit) && faLimit >= 0
      ? ` 감지 기준 ${step} 간격 · 시간당 오작동 ${faLimit}회 이하.` : "";
    if ($("recommendationScope")) $("recommendationScope").textContent = hardPolicy
      ? `조용할 때의 성공률·반응 시간 기준을 지키며 강한 소음과 울림의 조건별 평균 성공률이 높은 값을 고릅니다.${declaredSearch}`
      : "감지 기준과 연속 감지 수를 함께 비교한 이전 추천입니다.";
    const unavailable = measured.some(row => row.recommendation_available === false);
    const unverified = measured.some(row => row.deployment_backend_verified !== true);
    const previousGoal = finite(originalFaLimit) && finite(faLimit) && originalFaLimit !== faLimit
      ? `기존 평가 목표의 오작동 한도는 시간당 ${originalFaLimit}회이며, 추천 기준과 별도로 표시합니다. ` : "";
    if ($("recommendationNote")) $("recommendationNote").textContent = `${unavailable ? "추천 기준을 만족하는 조합이 없는 모델은 절충 후보를 표시합니다. 마이크에는 그 모델의 기존 설정을 유지합니다. " : ""}${previousGoal}${unverified ? "저장 점수 기준 비교이며 마이크 실행 결과는 별도로 확인합니다." : "실행 연산 방식에서도 추천 설정을 확인했습니다."}`;
    if (baseline) baseline.textContent = `평가 당시 설정${report.runtime ? ` · 감지 기준 ${thresholdText(report.runtime.threshold)} · 연속 ${report.runtime.consecutive_frames}회` : ""}의 결과입니다. 추천값 결과는 ‘추천 감지 설정’에서 확인하세요.`;
  }
  function reasonText(reason) {
    const text = String(reason);
    if (text.includes("Hard noise/reverberation conditions")) return "강소음·울림 시험 자료 부족";
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
  function criteriaFor(report) {
    const value = report.adoption_criteria;
    if (!value || value.schema_version !== 1) return null;
    const recalls = ["clean_recall", "snr10_recall", "snr5_recall"];
    if (recalls.some(key => !finite(value[key]) || value[key] < 0 || value[key] > 1)) return null;
    if (["fa_per_hour", "latency_p95_s", "minimum_negative_hours"].some(key => !finite(value[key]) || value[key] < 0)) return null;
    return value;
  }
  function targetChecks(model, criteria) {
    const checks = [];
    const recall = (value, exact = false) => exact ? `${value * 100}%` : percent(value);
    const metric = (unit, digits) => (value, exact = false) => `${exact ? value : Number(value.toFixed(digits))}${unit}`;
    const add = (label, actual, target, format, lowerIsBetter = false) => {
      const passed = finite(actual) && (lowerIsBetter ? actual <= target : actual >= target);
      let measured = finite(actual) ? format(actual) : "결과 없음";
      // Keep a small failure visible rather than rounding it to the target.
      if (finite(actual) && !passed && measured === format(target)) measured = format(actual, true);
      checks.push({label, target: `${format(target)} ${lowerIsBetter ? "이하" : "이상"}`,
        actual: measured, result: !finite(actual) ? "결과 없음" : passed ? "충족" : "미달"});
    };
    add("조용한 환경 성공률", model.clean?.recall, criteria.clean_recall, recall);
    for (const category of categories) {
      for (const [level, key] of [[10, "snr10_recall"], [5, "snr5_recall"]]) {
        add(`${noiseLabel(category)} · ${level === 10 ? "소음 중간" : "소음 큼"} (SNR ${level} dB)`,
          conditionValue(model.conditions, category, level)?.recall, criteria[key], recall);
      }
    }
    add("시간당 오작동", model.negative?.fa_per_hour, criteria.fa_per_hour, metric("회", 3), true);
    add("반응 시간 · 감지된 호출의 95%", model.primary?.latency_p95_s, criteria.latency_p95_s, metric("초", 2), true);
    add("오작동 시험 시간", model.negative?.hours, criteria.minimum_negative_hours, metric("시간", 2));
    return checks;
  }
  function renderTargets(report) {
    const criteria = criteriaFor(report), summary = $("criteriaSummary"), note = $("criteriaNote");
    const select = $("target-model"), rows = body("targetComparison");
    if (summary) summary.textContent = criteria
      ? `목표 기준: 조용할 때 ${percent(criteria.clean_recall)} 이상 · 소음 중간(SNR 10 dB) ${percent(criteria.snr10_recall)} 이상 · 소음 큼(SNR 5 dB) ${percent(criteria.snr5_recall)} 이상 · 시간당 오작동 ${Number(criteria.fa_per_hour.toFixed(3))}회 이하 · 반응 시간 ${Number(criteria.latency_p95_s.toFixed(2))}초 이내`
      : "평가 당시 목표 기준이 기록되지 않아 상세 비교를 표시할 수 없습니다.";
    if (note) note.textContent = criteria
      ? `소음 기준은 울림을 추가하지 않은 환경소음·음악·다른 사람 말소리 각각에 적용합니다. 오작동 시험 ${Number(criteria.minimum_negative_hours.toFixed(2))}시간 이상을 포함해 모든 기준을 만족해야 ‘충족’입니다.`
      : "";
    if ($("targetControls")) $("targetControls").hidden = !criteria;
    if ($("targetComparison")) $("targetComparison").hidden = !criteria;
    if (!select || !rows) return;
    const previous = select.value, models = report.models || [];
    select.replaceChildren(); rows.replaceChildren(); select.disabled = !criteria || !models.length;
    if (!criteria) return;
    $("candidateReasons")?.replaceChildren();
    models.forEach(model => option(select, model.id, modelLabel(report, model.id)));
    const selected = report.deployment_candidate?.id || report.selected_candidate || report.adoption?.candidate;
    select.value = models.some(model => model.id === previous) ? previous
      : models.some(model => model.id === selected) ? selected : models[0]?.id || "";
    function draw() {
      rows.replaceChildren();
      const model = models.find(value => value.id === select.value);
      if (!model) return;
      for (const check of targetChecks(model, criteria)) {
        const row = document.createElement("tr");
        [check.label, check.target, check.actual, check.result].forEach(value => appendCell(row, value));
        rows.append(row);
      }
    }
    select.onchange = draw; draw();
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
        ? `평가 당시 실행 모델: ${modelLabel(report, actualId)}${size ? ` · 모델 ${size}개 점수 평균` : ""}`
        : "선택한 모델 없음";
    }
    if ($("denominators")) $("denominators").textContent = "종합 성공률은 소음과 목소리가 같은 크기인 시험을 제외합니다. 소음 평균은 울림 없이 소음이 아주 작을 때부터 클 때까지의 평균입니다.";
    renderTargets(report);
    renderRecommendations(report);
    renderOverview(report);
    renderSearchCriteria(report);
  }
  function renderSearchCriteria(report) {
    const target = $("searchCriteria");
    if (!target) return;
    const recommendation = report.runtime_recommendations;
    const search = recommendation?.search_policy;
    const criteria = recommendation?.recommendation_criteria;
    const counts = Array.isArray(search?.consecutive_frames) ? search.consecutive_frames : [];
    const consecutive = counts.length && counts.every((value, index) => Number.isInteger(value) && value > 0 && (!index || value === counts[index - 1] + 1));
    const values = [
      ["감지 기준 간격", finite(search?.threshold_step) ? String(search.threshold_step) : "기록 없음"],
      ["연속 감지 수", consecutive ? counts.length === 1 ? `${counts[0]}회` : `${counts[0]}~${counts[counts.length - 1]}회 비교` : counts.length ? counts.map(value => `${value}회`).join(" · ") : "기록 없음"],
      ["조용할 때 성공률", finite(criteria?.clean_recall) ? `${percent(criteria.clean_recall)} 이상` : "기록 없음"],
      ["오호출 허용량", finite(criteria?.fa_per_hour) ? `시간당 ${criteria.fa_per_hour}회 이하` : "기록 없음"],
      ["반응 속도", finite(criteria?.latency_p95_s) ? `95%가 ${criteria.latency_p95_s}초 이내` : "기록 없음"],
      ["우선 비교", recommendation?.candidates?.some(row => row.metrics?.hard_conditions) ? "강한 소음·울림" : "이전 추천 방식"]
    ];
    target.replaceChildren();
    for (const [name, value] of values) {
      const item = document.createElement("div"), term = document.createElement("dt"), description = document.createElement("dd");
      term.textContent = name; description.textContent = value; item.append(term, description); target.append(item);
    }
    const note = $("searchCriteriaNote");
    if (note) note.textContent = criteria
      ? "위 기준을 지키면서 소음·울림 성공률이 높은 감지 설정을 고릅니다."
      : "이 평가에는 자동 추천 기준이 기록되지 않았습니다. 평가 당시 목표는 상세 결과에서 확인할 수 있습니다.";
  }
  function renderOverview(report) {
    const table = body("storySummary"), source = $("overview-source"), modelSelect = $("overview-model");
    if (!table || !source || !modelSelect) return;
    const recommendations = report.runtime_recommendations?.schema_version === 1 ? report.runtime_recommendations.candidates || [] : [];
    const previousSource = source.value, previousModel = modelSelect.value;
    const models = [...(report.models || [])].sort((left, right) => {
      const order = [...singleIds(report), ...report.models.filter(row => isEnsemble(report, row.id)).map(row => row.id)];
      return order.indexOf(left.id) - order.indexOf(right.id);
    });
    source.replaceChildren();
    if (recommendations.length) option(source, "recommended", "추천값 결과");
    option(source, "baseline", "평가 당시 결과");
    source.value = ["recommended", "baseline"].includes(previousSource) && (previousSource !== "recommended" || recommendations.length)
      ? previousSource : recommendations.length ? "recommended" : "baseline";
    modelSelect.replaceChildren(); models.forEach(model => option(modelSelect, model.id, modelLabel(report, model.id)));
    modelSelect.value = models.some(model => model.id === previousModel) ? previousModel : models.find(model => isEnsemble(report, model.id))?.id || models[0]?.id || "";
    const valuesFor = model => {
      const chosen = recommendations.find(row => row.id === model.id);
      const available = chosen && chosen.recommendation_available !== false && chosen.metrics && finite(chosen.runtime?.threshold)
        && chosen.runtime.threshold > 0 && chosen.runtime.threshold < 1 && Number.isInteger(chosen.runtime?.consecutive_frames) && chosen.runtime.consecutive_frames > 0;
      const recommended = source.value === "recommended";
      const metric = recommended ? available ? chosen.metrics : null : model;
      const runtime = recommended ? available ? chosen.runtime : null : report.runtime;
      const hard = metric?.hard_conditions;
      return [modelLabel(report, model.id), percent(metric?.clean?.recall),
        hard?.complete === false ? "자료 부족" : percent(hard?.cell_mean),
        finite(metric?.negative?.fa_per_hour) ? `${metric.negative.fa_per_hour.toFixed(3)}회` : "결과 없음",
        runtime && finite(runtime.threshold) && Number.isInteger(runtime.consecutive_frames)
          ? `${thresholdText(runtime.threshold)} · ${runtime.consecutive_frames}회` : recommended ? "추천 없음" : "기록 없음"];
    };
    function draw() {
      table.replaceChildren();
      for (const model of models) {
        const row = document.createElement("tr"); if (isEnsemble(report, model.id)) row.className = "ensemble";
        valuesFor(model).forEach(value => appendCell(row, value)); table.append(row);
      }
      const metrics = $("storySummaryMetrics"), model = models.find(row => row.id === modelSelect.value);
      metrics?.replaceChildren();
      if (metrics && model) {
        const labels = ["조용할 때", "강한 소음·울림", "오호출 / 시간", "감지 기준 · 연속"];
        valuesFor(model).slice(1).forEach((value, index) => {
          const item = document.createElement("div"), term = document.createElement("dt"), description = document.createElement("dd");
          term.textContent = labels[index]; description.textContent = value; item.append(term, description); metrics.append(item);
        });
      }
      const note = $("overview-note");
      if (note) note.textContent = source.value === "recommended"
        ? "추천 기준을 만족한 설정의 결과입니다. 추천이 없는 모델은 평가 당시 결과에서 확인하세요."
        : "저장된 평가 당시 설정의 결과입니다. 강한 소음·울림 평균이 기록되지 않은 평가는 ‘결과 없음’으로 표시합니다.";
    }
    source.onchange = draw; modelSelect.onchange = draw; draw();
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
    const snrSelect = $("condition-snr"), mobile = $("conditionMetrics");
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
      mobile?.replaceChildren();
      if (mobile && snrSelect) for (const category of categories) {
        const item = document.createElement("div"), term = document.createElement("dt"), description = document.createElement("dd");
        term.textContent = noiseLabel(category); description.textContent = statisticText(conditionValue(result?.conditions, category, Number(snrSelect.value)));
        item.append(term, description); mobile.append(item);
      }
      modelSelect.disabled = !models.length; reverbSelect.disabled = !models.length;
    }
    if (snrSelect) snrSelect.onchange = drawValues;
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
  function mountClipPicker(move, labelAt, pageSize) {
    const listening = window.KowwListening, container = $("result-clip-picker");
    if (!listening?.createPicker || !container) return null;
    const pager = $("clip-list-pager"), previous = $("clip-list-prev"), next = $("clip-list-next");
    const play = $("resume"), controls = play.parentElement;
    let page = 0, renderedPage = -1, length = 0, selected = -1, extraLabel = "";
    const picker = listening.createPicker({
      container, button: $("result-clip-picker-button"), label: $("result-clip-picker-label"),
      popup: $("result-clip-picker-popup"), list: $("result-clip-picker-list"),
      onSelect: offset => move(page * pageSize + offset),
    });
    resultClipPicker = picker;
    container.hidden = false;
    $("clipTitle").hidden = true;
    $("results-panel-audio").classList.add("listen-voice-compact");
    pager.hidden = true;
    controls.after(pager);
    listening.enhance($("audio"), controls, {
      play, previous: $("prev"), next: $("next"), volume: $("result-volume"),
    });

    function render() {
      const start = page * pageSize, end = Math.min(length, start + pageSize);
      const localIndex = !extraLabel && selected >= start && selected < end ? selected - start : -1;
      const display = extraLabel || (length ? labelAt(selected) : "선택한 조건의 음원 없음");
      if (renderedPage !== page) {
        const entries = [];
        for (let position = start; position < end; position++) {
          const label = labelAt(position);
          entries.push({label, fullLabel: `${count(position + 1)}번째 음원 · ${label}`});
        }
        picker.setItems(entries, localIndex, display);
        renderedPage = page;
      } else {
        picker.setSelected(localIndex, display);
      }
      pager.hidden = length <= pageSize;
      previous.disabled = page === 0;
      next.disabled = end >= length;
      $("clip-list-position").textContent = `${count(start + 1)}–${count(end)} / ${count(length)}`;
      if (extraLabel) $("queue").textContent = "추가 음원";
    }
    previous.onclick = () => { if (!previous.disabled) { page--; render(); } };
    next.onclick = () => { if (!next.disabled) { page++; render(); } };
    return (total, index, extra = "", reset = false) => {
      if (reset || total !== length) renderedPage = -1;
      if (reset || total !== length || index !== selected) page = Math.floor(index / pageSize);
      length = total;
      selected = index;
      extraLabel = extra;
      render();
    };
  }
  initDashboard();
  fitAudioControls();
  if (typeof window !== "undefined") window.addEventListener?.("resize", fitAudioControls);
  if (typeof window !== "undefined") window.addEventListener?.("message", pauseFromParent);
  if (typeof window !== "undefined") window.addEventListener?.("keydown", event => {
    if (event.key === "Escape") document.querySelectorAll?.("details[open]")?.forEach(detail => { detail.open = false; });
  });
  const api = {modelLabel, noiseLabel, snrLabel, reverbLabel, renderSummary, renderConditions, renderResponses, renderRecommendations, clipLabel,
    dashboard, modelDashboard, mountClipPicker, dashboardState: () => ({view: dashboardView, modelView})};
  if (typeof window !== "undefined") window.KowwResultsUI = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
