/* Presentation of frozen results. These helpers never recalculate or alter scores. */
"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const count = value => finite(value) ? value.toLocaleString("ko-KR") : "—";
  const percent = value => finite(value) ? `${(value * 100).toFixed(2)}%` : "결과 없음";
  const categories = ["environment", "music", "competing_speech"];
  const levels = [20, 15, 10, 5, 0];
  const strongReverbs = ["moderate", "strong"];
  const strongSnrs = [10, 5, 0];
  const strongConditionCount = strongReverbs.length * strongSnrs.length * categories.length;
  const worstReverb = "strong";
  const worstSnr = 0;
  const detectionLabels = {baseline: "기본 감지 기준", recommended: "추천 감지 기준"};
  const resultsTabs = ["models", "conditions", "audio"];
  const reportDiagnostics = new WeakSet();
  let dashboardView = "models";
  let compactAudio = null;
  let resultClipPicker;
  let overviewHelp;
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
    if (!resultsTabs.includes(view)) return;
    if (view !== "audio") resultClipPicker?.close();
    if (view !== "models") overviewHelp?.hide();
    dashboardView = view; activateTabs("results", resultsTabs, view, focus);
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
    bindTabs("results", resultsTabs, dashboard);
    dashboard(dashboardView);
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
    overviewHelp?.hide();
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
  function appendCell(row, text) {
    const node = document.createElement("td");
    node.textContent = String(text);
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
  function validSettings(runtime) {
    return finite(runtime?.threshold) && runtime.threshold > 0 && runtime.threshold < 1
      && Number.isInteger(runtime.consecutive_frames) && runtime.consecutive_frames > 0;
  }
  function measuredSettings(candidate) {
    return Boolean(candidate?.metrics && validSettings(candidate.runtime));
  }
  function recommendationApplies(candidate) {
    if (!candidate) return false;
    if (candidate.recommendation_available == null || candidate.recommendation_available === true) return true;
    // Standalone archived reports keep their original flags.
    const checks = candidate.constraint_checks;
    const rate = candidate.metrics?.negative?.fa_per_hour;
    return candidate.recommendation_available === false && checks?.false_activations === false && finite(rate) && rate >= 0
      && ["clean", "latency", "negative_duration", "hard_conditions"].every(key => checks[key] === true)
      && Object.entries(checks).every(([key, passed]) => key === "false_activations" || passed === true);
  }
  function matchRuntime(left, right) {
    return validSettings(left) && validSettings(right) && Object.keys(left).length === Object.keys(right).length
      && Object.entries(left).every(([key, value]) => right[key] === value);
  }
  function matchesDisplayedConditions(conditions) {
    if (!Array.isArray(conditions) || conditions.length !== strongConditionCount) return false;
    const declared = new Set(conditions.map(cell => JSON.stringify([cell?.reverb_condition, cell?.category, cell?.snr_db])));
    return strongReverbs.every(room => categories.every(category =>
      strongSnrs.every(snr => declared.has(JSON.stringify([room, category, snr])))));
  }
  function acousticRecall(metric, matchingMetric, rooms, snrs) {
    if (!metric) return undefined;
    const sources = [metric, matchingMetric].filter(Boolean).map(record => ({
      rooms: record.reverb_conditions,
      hardCells: Object.values(record.hard_conditions?.conditions ?? {})
    }));
    const cells = rooms.flatMap(room => snrs.flatMap(snr => categories.map(category => {
      for (const source of sources) {
        const cell = conditionValue(source.rooms?.[room]?.conditions, category, snr)
          ?? source.hardCells.find(entry => entry?.category === category &&
            entry.reverb_condition === room && entry.snr_db === snr);
        if (cell !== undefined) return cell;
      }
      return undefined;
    })));
    if (!cells.every(cell => finite(cell?.recall) && cell.recall >= 0 && cell.recall <= 1 && cell.calls !== 0)) return undefined;
    return cells.reduce((sum, cell) => sum + cell.recall, 0) / cells.length;
  }
  function renderSummary(report) {
    renderOverview(report);
    const recommendation = report.runtime_recommendations;
    if (recommendation?.report_diagnostics_pending && !reportDiagnostics.has(report)) {
      reportDiagnostics.add(report);
      fetch("report-diagnostics.json").then(response => {
        if (!response.ok) throw new Error("평가 자료를 불러오지 못했습니다.");
        return response.json();
      }).then(result => {
        if (result?.schema_version !== 1 || !Array.isArray(result.candidates)) throw new Error("평가 자료 형식이 올바르지 않습니다.");
        return result;
      }).catch(() => ({...recommendation, report_diagnostics_pending: false})).then(result => {
        if (report.runtime_recommendations !== recommendation) return;
        report.runtime_recommendations = result;
        renderOverview(report);
      });
    }
  }
  function recommendationDetails(recommendation, matchesDisplay) {
    const search = recommendation?.search_policy;
    const criteria = recommendation?.recommendation_criteria;
    if (!search && !criteria) return [];
    const conditionCount = Array.isArray(search?.hard_conditions) ? search.hard_conditions.length : 0;
    const hasConditionPriority = conditionCount > 0 || recommendation?.candidates?.some(row => row.metrics?.hard_conditions);
    const thresholds = Array.isArray(search?.coarse_thresholds) ? search.coarse_thresholds.filter(finite) : [];
    const step = finite(search?.threshold_step) ? String(search.threshold_step) : null;
    const min = Math.min(...thresholds), max = Math.max(...thresholds);
    const thresholdRange = min === max ? String(min) : `${min}~${max}`;
    const counts = Array.isArray(search?.consecutive_frames) ? search.consecutive_frames : [];
    const consecutive = counts.length && counts.every((value, index) => Number.isInteger(value) && value > 0 && (!index || value === counts[index - 1] + 1));
    const values = [
      thresholds.length
        ? ["감지 기준 범위", step && min !== max ? `${thresholdRange} · ${step} 간격` : thresholdRange]
        : ["감지 기준 간격", step ?? "기록 없음"],
      ["연속 감지 수", consecutive ? counts.length === 1 ? `${counts[0]}회` : `${counts[0]}~${counts[counts.length - 1]}회 비교` : counts.length ? counts.map(value => `${value}회`).join(" · ") : "기록 없음"],
      ["조용할 때 감지율", finite(criteria?.clean_recall) ? `${percent(criteria.clean_recall)} 이상` : "기록 없음"],
      ["오호출 허용량", finite(criteria?.fa_per_hour) ? `시간당 ${criteria.fa_per_hour}회 이하` : "기록 없음"],
      ["반응 시간", finite(criteria?.latency_p95_s) ? `95%가 ${criteria.latency_p95_s}초 이내` : "기록 없음"],
      ...(finite(criteria?.minimum_negative_hours) ? [["비호출 평가 시간", `${criteria.minimum_negative_hours}시간 이상`]] : []),
      ["우선 비교", hasConditionPriority
        ? `${matchesDisplay ? "강한 소음·울림" : "소음·울림 시험 평균"}${conditionCount ? ` · ${conditionCount}가지` : ""}`
        : "기록 없음"]
    ];
    return values.map(([label, value]) => ({label, value}));
  }
  function appendMetric(list, label, value, rowClass) {
    const item = document.createElement("div"), term = document.createElement("dt"), description = document.createElement("dd");
    if (rowClass) item.className = rowClass;
    term.textContent = label; description.textContent = value; item.append(term, description); list.append(item);
    return description;
  }
  function createOverviewHelp() {
    const popup = $("overview-help"), title = $("overview-help-title");
    const content = $("overview-help-content"), close = $("overview-help-close");
    if (!popup?.showPopover || !title || !content || !close) return null;
    let active;
    function resetTrigger() {
      active?.setAttribute("aria-expanded", "false");
      active = null;
    }
    function hide(restoreFocus = false) {
      const trigger = active;
      if (popup.matches(":popover-open")) popup.hidePopover();
      resetTrigger();
      if (restoreFocus) trigger?.focus();
    }
    function position() {
      if (!active || !popup.matches(":popover-open")) return;
      // Use iframe-local coordinates and keep the panel inside the viewport.
      const anchor = active.getBoundingClientRect(), rect = popup.getBoundingClientRect();
      const margin = 12, gap = 6;
      const width = window.innerWidth, height = window.innerHeight;
      const left = anchor.left + (anchor.width - rect.width) / 2;
      const below = anchor.bottom + gap, above = anchor.top - rect.height - gap;
      const top = below + rect.height <= height - margin || above < margin ? below : above;
      popup.style.left = `${Math.max(margin, Math.min(left, width - rect.width - margin))}px`;
      popup.style.top = `${Math.max(margin, Math.min(top, height - rect.height - margin))}px`;
    }
    function bind(trigger, help) {
      trigger.setAttribute("aria-haspopup", "dialog");
      trigger.setAttribute("aria-controls", popup.id || "overview-help");
      trigger.setAttribute("aria-expanded", "false");
      trigger.setAttribute("popovertarget", popup.id || "overview-help");
      trigger.onclick = event => {
        event.preventDefault();
        if (active === trigger && popup.matches(":popover-open")) {
          hide(true);
          return;
        }
        hide();
        title.textContent = help.title;
        content.replaceChildren();
        for (const text of help.paragraphs) {
          const paragraph = document.createElement("p");
          paragraph.textContent = text;
          content.append(paragraph);
        }
        if (help.models?.length) {
          const list = document.createElement("dl");
          list.className = "overview-help-models";
          for (const {label, value} of help.models) appendMetric(list, label, value);
          content.append(list);
        }
        active = trigger;
        popup.showPopover({source: trigger});
        trigger.setAttribute("aria-expanded", "true");
        content.scrollTop = 0;
        position();
        close.focus({preventScroll: true});
      };
    }
    close.onclick = () => hide(true);
    popup.ontoggle = () => { if (!popup.matches(":popover-open")) resetTrigger(); };
    window.addEventListener?.("resize", position);
    return {bind, hide};
  }
  function metricHelp(criteria) {
    return {
      hard: {title: "강한 소음·울림", paragraphs: [
        `울림: ${strongReverbs.map(reverbLabel).join(", ")}\n소음 크기: 호출어보다 10 dB 또는 5 dB 작게, 또는 호출어와 같은 크기 (SNR ${strongSnrs.map(level => `${level} dB`).join(", ")})`,
        `총 ${strongConditionCount}가지 조건의 감지율을 같은 비중으로 평균합니다.`
      ]},
      worst: {title: "최악 조건", paragraphs: [
        `울림: ${reverbLabel(worstReverb)}\n소음 크기: 호출어와 같은 크기 (SNR ${worstSnr} dB)`,
        "이 조건에서 배경 소리별 감지율을 같은 비중으로 평균합니다."
      ]},
      latency: {title: "반응 시간", paragraphs: ["감지에 성공한 호출의 약 95%가 이 시간 안에 감지됐습니다(P95).",
        "시간은 발화 종료 시점을 기준으로 잽니다. 값에 ‘전’이 붙으면 발화가 끝나기 전에 감지됐다는 뜻입니다."]},
      criteria: {title: "평가 기준", paragraphs: ["모델을 비교할 때 참고할 기준입니다."], models: [
        {label: "소음·울림 없음", value: finite(criteria?.clean_recall) ? `${Number((criteria.clean_recall * 100).toFixed(2))}% 이상` : "높을수록 좋습니다."},
        {label: "종합·소음·최악 조건", value: "감지율이 높을수록 좋습니다."},
        {label: "오호출", value: finite(criteria?.fa_per_hour) ? `시간당 ${criteria.fa_per_hour}회 이하` : "적을수록 좋습니다."},
        {label: "반응 시간", value: finite(criteria?.latency_p95_s) ? `감지된 호출의 95%가 ${criteria.latency_p95_s}초 이내` : "짧을수록 좋습니다."}
      ]}
    };
  }
  function metricHeading(cell, text, help) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "overview-help-button";
    button.textContent = text;
    overviewHelp.bind(button, help);
    cell.replaceChildren(button);
  }
  function renderOverview(report) {
    const table = body("storySummary"), source = $("overview-source"), modelSelect = $("overview-model");
    if (!table || !source || !modelSelect) return;
    const calibration = report.runtime_recommendations;
    const matchesDisplay = matchesDisplayedConditions(calibration?.search_policy?.hard_conditions);
    const recordedCriteria = recommendationDetails(calibration, matchesDisplay);
    const sharedHelp = metricHelp(calibration?.recommendation_criteria);
    overviewHelp ??= createOverviewHelp();
    const candidates = calibration?.schema_version === 1 && Array.isArray(calibration.candidates) ? calibration.candidates : [];
    const models = report.models || [];
    const order = [...singleIds(report), ...models.filter(row => isEnsemble(report, row.id)).map(row => row.id)];
    const sorted = [...models].sort((left, right) => order.indexOf(left.id) - order.indexOf(right.id));
    const byModel = new Map(candidates.map(candidate => [candidate.id, candidate]));
    const hasComparison = models.some(model => measuredSettings(byModel.get(model.id)));
    const previousSource = source.value, previousModel = modelSelect.value;
    const deployed = report.deployment_candidate?.id || report.selected_candidate || report.adoption?.candidate;
    source.replaceChildren();
    if (hasComparison) option(source, "recommended", detectionLabels.recommended);
    option(source, "baseline", detectionLabels.baseline);
    source.value = hasComparison && previousSource !== "baseline" ? "recommended" : "baseline";
    source.disabled = !hasComparison;
    modelSelect.replaceChildren(); sorted.forEach(model => option(modelSelect, model.id, modelLabel(report, model.id)));
    modelSelect.value = models.some(model => model.id === previousModel) ? previousModel
      : models.some(model => model.id === deployed) ? deployed : sorted[0]?.id || "";

    function overviewFor(model, compareSettings, settingsLabel) {
      const candidate = byModel.get(model.id);
      const tested = measuredSettings(candidate) ? candidate : null;
      const metric = compareSettings ? tested?.metrics : model;
      const runtime = compareSettings ? tested?.runtime : report.runtime;
      const sameRuntimeMetrics = !compareSettings && matchRuntime(runtime, candidate?.baseline?.runtime)
        ? candidate.baseline.metrics : null;
      const negative = metric?.negative;
      const rawFalse = finite(negative?.events) && finite(negative?.hours) ? `${count(negative.events)}회 / ${negative.hours.toFixed(2)}시간` : "";
      const clean = metric?.clean;
      const cleanCount = finite(clean?.recall) && finite(clean.successes) && finite(clean.calls) ? `${count(clean.successes)} / ${count(clean.calls)}회` : "";
      const settings = compareSettings && tested && !recommendationApplies(tested) ? "추천값 없음"
        : validSettings(runtime) ? `${thresholdText(runtime.threshold)} · ${runtime.consecutive_frames}회` : "기록 없음";
      const conditionValueText = recall => metric && recall === undefined
        ? (compareSettings && calibration?.report_diagnostics_pending ? "계산 중" : "자료 부족") : percent(recall);
      const hardValue = conditionValueText(acousticRecall(metric, sameRuntimeMetrics, strongReverbs, strongSnrs));
      const worstValue = conditionValueText(acousticRecall(metric, sameRuntimeMetrics, [worstReverb], [worstSnr]));
      const falseRate = finite(negative?.fa_per_hour) ? `${Number(negative.fa_per_hour.toFixed(3))}회` : "결과 없음";
      const entries = [
        {label: "종합 감지율", value: percent(metric?.primary?.recall)},
        {label: "소음·울림 없음", value: clean?.calls === 0 ? "시험 안 함" : percent(clean?.recall), title: cleanCount},
        {label: "강한 소음·울림", value: hardValue, help: "hard"},
        {label: "최악 조건", value: worstValue, help: "worst"},
        {label: "오호출/시간", value: falseRate, title: rawFalse, help: "fa"},
        {label: "반응 시간", value: reactionTime(metric?.primary?.latency_p95_s), help: "latency"},
        {label: settingsLabel, value: settings, help: "settings"}
      ];
      return entries;
    }
    function draw() {
      overviewHelp?.hide();
      const compareSettings = source.value === "recommended";
      const settingsLabel = compareSettings ? detectionLabels.recommended : detectionLabels.baseline;
      const recommendationLimit = calibration?.recommendation_criteria?.fa_per_hour;
      const selectionMetric = matchesDisplay ? `모델 비교의 ‘강한 소음·울림’과 같은 ${strongConditionCount}가지 조건`
        : "추천 비교에 기록된 소음·울림 시험 조건";
      const selection = finite(recommendationLimit) ? [
        `다른 추천 조건을 충족하는 설정 중 오호출이 시간당 ${recommendationLimit}회 이하인 후보를 먼저 봅니다. 그중 ${selectionMetric}에서 평균 감지율이 가장 높은 설정을 추천합니다.`,
        "상한 이내에 드는 후보가 없으면, 다른 추천 조건을 충족한 설정 중 오호출이 가장 적은 설정을 추천합니다."
      ] : [];
      const explanations = {
        ...sharedHelp,
        fa: {title: "오호출/시간", paragraphs: ["호출어가 없는 음원에서 호출어를 잘못 감지한 횟수를 1시간 기준으로 환산한 값입니다.",
          ...(finite(recommendationLimit) ? [`감지 기준을 추천할 때 오호출 상한은 시간당 ${recommendationLimit}회입니다.`] : [])]},
        settings: {title: settingsLabel, models: compareSettings ? recordedCriteria : [], paragraphs: [
          "앞 숫자는 감지 점수의 임계값이고, 뒤의 횟수는 그 점수 이상이 연속으로 나와야 하는 횟수입니다.",
          ...(compareSettings
            ? ["모델마다 저장된 평가 점수에 여러 감지 기준을 적용해 비교했습니다.", ...selection,
              ...(!recordedCriteria.length ? ["이 평가에는 추천 기준이 기록되지 않았습니다."] : [])]
            : ["이 보고서의 성능 평가에 사용한 설정입니다."])
        ]}
      };
      for (const button of document.querySelectorAll?.("[data-overview-help]") ?? []) {
        const key = button.dataset.overviewHelp;
        if (key === "settings") button.textContent = settingsLabel;
        overviewHelp?.bind(button, explanations[key]);
      }
      table.replaceChildren();
      const metrics = $("storySummaryMetrics");
      metrics?.replaceChildren();
      for (const model of sorted) {
        const entries = overviewFor(model, compareSettings, settingsLabel);
        const label = modelLabel(report, model.id);
        const row = document.createElement("tr");
        if (isEnsemble(report, model.id)) row.className = "ensemble";
        appendCell(row, label);
        for (const {value, title} of entries) {
          const cell = appendCell(row, value);
          if (title) cell.title = title;
        }
        table.append(row);
        if (metrics && model.id === modelSelect.value) {
          for (const entry of entries) {
            const description = appendMetric(metrics, entry.label, entry.value);
            if (entry.title) description.title = entry.title;
            if (overviewHelp && entry.help) {
              metricHeading(description.previousElementSibling, entry.label, explanations[entry.help]);
            }
          }
        }
      }
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
  const api = {modelLabel, noiseLabel, snrLabel, reverbLabel, renderSummary, renderConditions, renderResponses, clipLabel,
    dashboard, mountClipPicker, dashboardState: () => ({view: dashboardView})};
  if (typeof window !== "undefined") window.KowwResultsUI = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})();
