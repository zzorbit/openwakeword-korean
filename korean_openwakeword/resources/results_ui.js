/* Presentation of frozen results. These helpers never recalculate or alter scores. */
"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const count = value => finite(value) ? value.toLocaleString("ko-KR") : "—";
  const percent = value => finite(value) ? `${(value * 100).toFixed(2)}%` : "결과 없음";
  const categories = ["environment", "music", "competing_speech"];
  const levels = [20, 15, 10, 5, 0];
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
    if (!["models", "conditions", "audio", "criteria"].includes(view)) return;
    if (view !== "audio") resultClipPicker?.close();
    if (view !== "models") overviewHelp?.hide();
    dashboardView = view; activateTabs("results", ["models", "conditions", "audio", "criteria"], view, focus);
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
  function memberCount(report, id) {
    const declared = definitions(report).find(row => row.id === id);
    const deployed = report.deployment_candidate;
    return declared?.members?.length || (deployed?.id === id ? deployed.members?.length : 0) || singleIds(report).length;
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
  const recommendationMissed = "추천 기준 미달";
  function validSettings(runtime) {
    return finite(runtime?.threshold) && runtime.threshold > 0 && runtime.threshold < 1
      && Number.isInteger(runtime.consecutive_frames) && runtime.consecutive_frames > 0;
  }
  function measuredSettings(candidate) {
    return Boolean(candidate?.metrics && validSettings(candidate.runtime));
  }
  function recommendationReasons(candidate, criteria) {
    return [...new Set((candidate.recommendation_reasons || []).map(reason =>
      String(reason).includes("FA/hour") && finite(criteria?.fa_per_hour)
        ? `오호출 한도 ${criteria.fa_per_hour}회/시간`
        : String(reason).includes("Hard noise/reverberation conditions") ? "강소음·울림 시험 자료 부족" : goalReason(reason)))];
  }
  function goalReason(reason) {
    const text = String(reason);
    if (categories.some(category => text.includes(category))) return "소음 조건";
    if (text.includes("FP duration below")) return "오호출 시험 시간";
    if (text.includes("Clean recall")) return "조용할 때 감지율";
    if (text.includes("FA/hour")) return "시간당 오호출";
    if (text.includes("Latency")) return "반응 시간";
    return "추가 확인 필요";
  }
  function criteriaFor(report) {
    const value = report.adoption_criteria;
    if (value?.schema_version !== 1) return null;
    const recalls = ["clean_recall", "snr10_recall", "snr5_recall"];
    if (recalls.some(key => !finite(value[key]) || value[key] < 0 || value[key] > 1)) return null;
    if (["fa_per_hour", "latency_p95_s", "minimum_negative_hours"].some(key => !finite(value[key]) || value[key] < 0)) return null;
    return value;
  }
  function matchRuntime(left, right) {
    return validSettings(left) && validSettings(right) && Object.keys(left).length === Object.keys(right).length
      && Object.entries(left).every(([key, value]) => right[key] === value);
  }
  function recordedHardConditions(model, policy) {
    if (!Array.isArray(policy) || !policy.length) return null;
    // Average only the declared cells, using detection rates already recorded in this report.
    const cells = policy.map(({reverb_condition, category, snr_db}) => {
      const result = reverb_condition === "none" ? model : model.reverb_conditions?.[reverb_condition];
      const stats = category === "clean" ? cleanValue(result) : conditionValue(result?.conditions, category, snr_db);
      return {reverb_condition, category, snr_db, recall: stats?.recall};
    });
    if (cells.some(cell => !finite(cell.recall))) return {complete: false};
    return {complete: true, conditions: cells, cell_mean: cells.reduce((sum, cell) => sum + cell.recall, 0) / cells.length,
      worst_recall: Math.min(...cells.map(cell => cell.recall))};
  }
  function renderSummary(report) {
    renderOverview(report);
    renderCriteria(report);
  }
  function renderCriteria(report) {
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
      ["조용할 때 감지율", finite(criteria?.clean_recall) ? `${percent(criteria.clean_recall)} 이상` : "기록 없음"],
      ["오호출 허용량", finite(criteria?.fa_per_hour) ? `시간당 ${criteria.fa_per_hour}회 이하` : "기록 없음"],
      ["반응 시간", finite(criteria?.latency_p95_s) ? `95%가 ${criteria.latency_p95_s}초 이내` : "기록 없음"],
      ["우선 비교", recommendation?.candidates?.some(row => row.metrics?.hard_conditions) ? "강한 소음·울림" : "기록 없음"]
    ];
    target.replaceChildren();
    for (const [name, value] of values) appendMetric(target, name, value);
    const note = $("searchCriteriaNote");
    if (note) note.textContent = criteria
      ? "위 기준을 지키면서 소음·울림 감지율이 높은 설정을 고릅니다."
      : "이 평가에는 추천 기준이 기록되지 않았습니다.";
    const original = $("evaluationCriteria"), goals = criteriaFor(report);
    if (!original) return;
    const group = original.closest?.(".evaluation-criteria");
    if (group) group.hidden = !goals;
    original.replaceChildren();
    if (!goals) return;
    for (const [name, value] of [
      ["조용할 때 감지율", `${percent(goals.clean_recall)} 이상`],
      ["울림 없는 소음 조건", `SNR 10 dB ${percent(goals.snr10_recall)} 이상 · SNR 5 dB ${percent(goals.snr5_recall)} 이상`],
      ["오호출 허용량", `시간당 ${goals.fa_per_hour}회 이하`],
      ["반응 시간", `95%가 ${goals.latency_p95_s}초 이내`],
      ["오호출 시험 시간", `${goals.minimum_negative_hours}시간 이상`]
    ]) appendMetric(original, name, value);
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
        if (help.conditions?.length) {
          const list = document.createElement("ul");
          for (const text of help.conditions) {
            const item = document.createElement("li");
            item.textContent = text;
            list.append(item);
          }
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
  function metricHelp(policy) {
    const hard = ["비교에 포함된 소음·울림 조건의 감지율을 같은 비중으로 평균한 값입니다."];
    if (Array.isArray(policy) && policy.length) {
      const noiseLevels = [...new Set(policy.map(cell => cell.snr_db).filter(finite))].sort((left, right) => right - left);
      const noise = [...new Set(policy.map(cell => cell.category).filter(category => category !== "clean"))].map(noiseLabel);
      const rooms = [...new Set(policy.filter(cell => cell.category !== "clean").map(cell => cell.reverb_condition))].map(reverbLabel);
      const groups = [`비교 조건 ${policy.length}개`];
      if (rooms.length) groups.push(`울림 · ${rooms.join(", ")}`);
      if (noise.length) groups.push(`배경 · ${noise.join(", ")}`);
      if (noiseLevels.length) groups.push(`소음 크기 · SNR ${noiseLevels.join(", ")} dB`);
      hard.push(groups.join("\n"));
      if (policy.some(cell => cell.category === "clean" && cell.reverb_condition !== "none")) {
        hard.push("배경 소음 없이 울림만 있는 조건도 포함합니다.");
      }
    }
    return {
      hard: {title: "강한 소음·울림", paragraphs: hard},
      worst: {title: "최약 조건", paragraphs: ["강한 소음·울림 비교에 포함된 조건 중 감지율이 가장 낮은 값입니다.",
        "모델의 수치를 누르면 해당 시험 조건을 볼 수 있습니다."]}
    };
  }
  function weakestConditions(hard) {
    return Object.values(hard.conditions || {}).filter(cell => cell.recall === hard.worst_recall)
      .map(({reverb_condition: reverb, category, snr_db: snr}) =>
        `${reverbLabel(reverb)} · ${noiseLabel(category)}${finite(snr) ? ` · SNR ${snr} dB` : ""}`);
  }
  function metricButton(cell, text, className, help) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = text;
    if (className === "overview-value-button") button.setAttribute("aria-label", `${help.title} ${text}, 설명 열기`);
    overviewHelp.bind(button, help);
    cell.replaceChildren(button);
  }
  function renderOverview(report) {
    const table = body("storySummary"), source = $("overview-source"), modelSelect = $("overview-model");
    if (!table || !source || !modelSelect) return;
    const calibration = report.runtime_recommendations;
    const explanations = metricHelp(calibration?.search_policy?.hard_conditions);
    overviewHelp ??= createOverviewHelp();
    if (overviewHelp) for (const button of document.querySelectorAll("[data-overview-help]")) {
      overviewHelp.bind(button, explanations[button.dataset.overviewHelp]);
    }
    const candidates = calibration?.schema_version === 1 && Array.isArray(calibration.candidates) ? calibration.candidates : [];
    const models = report.models || [];
    const order = [...singleIds(report), ...models.filter(row => isEnsemble(report, row.id)).map(row => row.id)];
    const sorted = [...models].sort((left, right) => order.indexOf(left.id) - order.indexOf(right.id));
    const byModel = new Map(candidates.map(candidate => [candidate.id, candidate]));
    const hasComparison = models.some(model => measuredSettings(byModel.get(model.id)));
    const hasMissedRecommendation = candidates.some(candidate => candidate.recommendation_available === false && measuredSettings(candidate));
    const previousSource = source.value, previousModel = modelSelect.value;
    const deployed = report.deployment_candidate?.id || report.selected_candidate || report.adoption?.candidate;
    source.replaceChildren();
    if (hasComparison) option(source, "recommended", "추천값 결과");
    option(source, "baseline", "평가 당시 결과");
    source.value = hasComparison && previousSource !== "baseline" ? "recommended" : "baseline";
    source.disabled = !hasComparison;
    modelSelect.replaceChildren(); sorted.forEach(model => option(modelSelect, model.id, modelLabel(report, model.id)));
    modelSelect.value = models.some(model => model.id === previousModel) ? previousModel
      : models.some(model => model.id === deployed) ? deployed : sorted[0]?.id || "";

    function overviewFor(model) {
      const compareSettings = source.value === "recommended";
      const candidate = byModel.get(model.id);
      const tested = measuredSettings(candidate) ? candidate : null;
      const metric = compareSettings ? tested?.metrics : model;
      const runtime = compareSettings ? tested?.runtime : report.runtime;
      const sameRuntimeHard = !compareSettings && matchRuntime(runtime, candidate?.baseline?.runtime)
        ? candidate.baseline.metrics?.hard_conditions : null;
      const hard = metric?.hard_conditions ?? sameRuntimeHard
        ?? (compareSettings ? null : recordedHardConditions(model, calibration?.search_policy?.hard_conditions));
      const status = compareSettings && tested?.recommendation_available === false
        ? [recommendationMissed, ...recommendationReasons(tested, calibration.recommendation_criteria)].join(" · ") : "";
      const goal = compareSettings ? tested?.adoption
        : report.candidate_adoptions?.[model.id] || (report.adoption?.candidate === model.id ? report.adoption : null);
      const passed = (compareSettings ? tested?.passed : undefined) ?? goal?.passed;
      const negative = metric?.negative;
      const rawFalse = finite(negative?.events) && finite(negative?.hours) ? `${count(negative.events)}회 / ${negative.hours.toFixed(2)}시간` : "";
      const clean = metric?.clean;
      const cleanCount = finite(clean?.recall) && finite(clean.successes) && finite(clean.calls) ? `${count(clean.successes)} / ${count(clean.calls)}회` : "";
      const dryNoise = finite(metric?.noise_cell_mean) ? `울림 없는 소음 평균 ${percent(metric.noise_cell_mean)}` : "";
      const settings = validSettings(runtime) ? `${thresholdText(runtime.threshold)} · ${runtime.consecutive_frames}회` : "기록 없음";
      const entries = [
        {label: "종합 감지율", value: percent(metric?.primary?.recall)},
        {label: "조용할 때", value: clean?.calls === 0 ? "시험 안 함" : percent(clean?.recall), title: cleanCount},
        {label: "강한 소음·울림", value: hard?.complete === false ? "자료 부족" : percent(hard?.cell_mean),
          help: "hard", measured: hard?.complete !== false && finite(hard?.cell_mean), title: dryNoise},
        {label: "최약 조건", value: hard?.complete === false ? "자료 부족" : percent(hard?.worst_recall),
          help: "worst", measured: hard?.complete !== false && finite(hard?.worst_recall)},
        {label: "오호출/시간", value: finite(negative?.fa_per_hour) ? `${negative.fa_per_hour.toFixed(3)}회` : "결과 없음", title: rawFalse},
        {label: "반응 시간", value: reactionTime(metric?.primary?.latency_p95_s)},
        {label: "감지 설정", value: settings, title: "감지 기준 · 연속 감지 횟수"},
        {label: "평가 목표", value: passed === true ? "충족" : passed === false ? "미달" : "기록 없음",
          title: passed === false ? [...new Set((goal?.reasons || []).map(goalReason))].join(" · ") : ""}
      ];
      return {entries, status, hard};
    }
    function showMetric(cell, entry, model, hard) {
      if (!overviewHelp || !entry.measured) {
        if (entry.title) cell.title = entry.title;
        return;
      }
      const {help, label, value} = entry;
      const mode = source.value === "recommended" ? "추천값 결과" : "평가 당시 결과";
      const paragraphs = [`${mode} · ${value}`];
      const conditions = help === "worst" ? weakestConditions(hard) : [];
      if (help === "hard") {
        paragraphs.push(...explanations.hard.paragraphs);
        if (entry.title) paragraphs.push(entry.title);
      } else if (!conditions.length) {
        paragraphs.push("이 보고서에는 세부 시험 조건이 기록되지 않았습니다.");
      }
      metricButton(cell, value, "overview-value-button", {title: `${modelLabel(report, model.id)} · ${label}`, paragraphs, conditions});
    }
    function draw() {
      overviewHelp?.hide();
      table.replaceChildren();
      const metrics = $("storySummaryMetrics");
      metrics?.replaceChildren();
      for (const model of sorted) {
        const {entries, status, hard} = overviewFor(model);
        const row = document.createElement("tr");
        if (isEnsemble(report, model.id)) row.className = "ensemble";
        const name = appendCell(row, modelLabel(report, model.id));
        if (status) {
          const detail = document.createElement("small");
          detail.className = "recommendation-status"; detail.textContent = status;
          name.append(detail);
        }
        for (const entry of entries) showMetric(appendCell(row, entry.value), entry, model, hard);
        table.append(row);
        if (metrics && model.id === modelSelect.value) {
          if (status) appendMetric(metrics, "추천 상태", status, "overview-status-row").className = "recommendation-status";
          for (const entry of entries) {
            const description = appendMetric(metrics, entry.label, entry.value);
            if (entry.label === "반응 시간") description.setAttribute("aria-describedby", "overview-latency-note");
            if (overviewHelp && entry.help) {
              metricButton(description.previousElementSibling, entry.label, "overview-help-button", explanations[entry.help]);
            }
            showMetric(description, entry, model, hard);
          }
        }
      }
      const note = $("overview-note");
      if (note) note.textContent = source.value === "recommended"
        ? hasMissedRecommendation ? "미달 모델은 마이크에서 평가 당시 설정을 사용합니다." : "저장된 점수로 감지 설정을 비교했습니다."
        : deployed ? `평가 당시 실행 모델 · ${modelLabel(report, deployed)}${isEnsemble(report, deployed) ? ` (${memberCount(report, deployed)}개 모델)` : ""}` : "";
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
