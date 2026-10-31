"use strict";
/* Presentation adapter. Worker state and all human decisions stay authoritative. */
(() => {
  const definitions = [
    ["entry", "호출어 입력", "호출어를 입력해 주세요."],
    ["scale", "학습 규모 선택", "학습 규모를 선택해 주세요."],
    ["materials", "준비할 자료", "준비할 자료를 확인해 주세요."],
    ["downloads", "자료 다운로드", "자료 다운로드"],
    ["voices", "합성 목소리 준비", "합성 목소리 준비"],
    ["pronunciation", "호출어 발음 확인", "호출어 발음 확인"],
    ["positive-generation", "호출어 음성 생성", "호출어 음성 생성"],
    ["phrases", "유사어 입력", "유사어 문구 입력"],
    ["hard-generation", "유사어 음성 생성", "유사어 음성 생성"],
    ["hard-review", "유사어 발음 확인", "유사어 발음 확인"],
    ["features", "증강·특징 추출", "소음·울림 증강과 특징 추출"],
    ["training", "모델 학습", "모델 학습"],
    ["evaluation", "성능 평가", "성능 평가"],
    ["results", "학습 결과", "학습 결과"],
    ["microphone", "마이크 시험", "마이크로 시험"],
  ];
  const mapping = {
    frontend: "downloads", data: "downloads", data_download: "downloads", data_review: "downloads", data_connect: "downloads",
    references: "voices", references_review: "voices", pilot: "pronunciation", pilot_review: "pronunciation",
    generate: "positive-generation", positives: "positive-generation", hard_phrases: "phrases",
    hard_generation: "hard-generation", hard_review: "hard-review", hard_connect: "hard-review",
    features: "features", train: "training", training: "training", package: "training",
    evaluate: "evaluation", development: "evaluation", evaluation: "evaluation", test: "results", results: "results",
  };
  const order = definitions.map(([id]) => id);
  const done = value => ["done", "complete", "completed", "finished"].includes(value);
  const running = value => ["running", "starting", "queued", "pausing"].includes(value);
  const stageId = project => project?.stage_id || (typeof project?.stage === "object" ? project.stage.id : project?.stage);
  const stageStep = project => done(project?.status) ? "results" : mapping[stageId(project)] || "downloads";
  function frontier(project) {
    if (!project) return 2;
    if (done(project.status)) return (project.results?.mic_url || project.mic_url) ? 14 : 13;
    // Only actual current/completed backend stages unlock later screens.
    return Math.max(3, order.indexOf(stageStep(project)));
  }
  const titles = { check: "동작 확인용", trial: "시험 모델용", production: "상용 모델용" };
  const shortDescriptions = { check: "라이브러리 동작 확인", trial: "호출 성능 비교·시험", production: "많은 자료로 본격 학습" };
  const count = value => Number.isFinite(value) ? value.toLocaleString("ko-KR") : "—";
  const $ = id => document.getElementById(id);
  const node = (tag, text, className) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (className) result.className = className;
    return result;
  };
  const button = (text, className, action) => {
    const result = node("button", text, className); result.type = "button";
    if (action) result.addEventListener("click", action);
    return result;
  };
  function create(bridge) {
    // Core DOM-only unit tests continue to exercise studio.js independently.
    if (!document.querySelector || !document.body || !$('wakeword')) return null;
    const legacy = document.querySelector("main.dashboard-shell");
    legacy.classList.add("story-source");
    document.body.classList.add("story-mode");
    const shell = node("main", undefined, "story-shell"); shell.id = "story-shell";
    const atmosphere = node("div", undefined, "story-atmosphere"); atmosphere.setAttribute("aria-hidden", "true");
    atmosphere.append(node("div", undefined, "story-background-state story-background-previous"), node("div", undefined, "story-background-state story-background-current"));
    const toolbar = node("header", undefined, "story-toolbar");
    toolbar.append($("project-switch"));
    const settingsButton = button("설정", "story-tool", () => openDialog(settingsDialog));
    const logButton = button("작업 기록", "story-tool", () => { $("log-details").open = true; openDialog(logDialog); bridge.logs(); });
    toolbar.append(settingsButton, logButton);
    const notices = node("div", undefined, "story-notices"); notices.append($("connection"), $("message"));
    const index = node("nav", undefined, "story-index"); index.setAttribute("aria-label", "진행 단계");
    const stage = node("section", undefined, "story-stage");
    const panel = node("div", undefined, "story-panel"); panel.id = "story-panel";
    const heading = node("h1"); heading.id = "story-title"; heading.tabIndex = -1;
    panel.setAttribute("aria-labelledby", heading.id);
    const context = node("p", undefined, "story-context");
    const content = node("div", undefined, "story-content");
    const entry = node("div", undefined, "story-entry"); entry.append($("wakeword"));
    const scale = node("div", undefined, "story-scale"); scale.append($("presets"));
    const materials = node("div", undefined, "story-materials");
    const materialsTabs = node("div", undefined, "story-material-tabs"); materialsTabs.setAttribute("role", "tablist"); materialsTabs.setAttribute("aria-label", "자료 구분");
    const materialColumns = node("div", undefined, "story-material-columns");
    const materialPages = node("div", undefined, "story-pagination");
    const materialBack = button("‹", "story-small", () => { materialPage = Math.max(0, materialPage - 1); renderMaterials(); }); materialBack.setAttribute("aria-label", "앞 자료");
    const materialPageText = node("span");
    const materialNext = button("›", "story-small", () => { materialPage++; renderMaterials(); }); materialNext.setAttribute("aria-label", "뒤 자료");
    materialPages.append(materialBack, materialPageText, materialNext);
    materials.append(materialsTabs, materialColumns, materialPages);
    const live = node("div", undefined, "story-live");
    const statusCard = document.querySelector(".current-card");
    const liveSummary = node("div", undefined, "story-live-summary");
    const currentTitle = $("current-title");
    liveSummary.append(statusCard, $("counts"));
    const downloads = $("downloads-section"), review = $("review-section"), input = $("input-section"), results = $("results-section");
    const downloadPages = node("div", undefined, "story-pagination");
    const downloadBack = button("‹", "story-small", () => { downloadPage = Math.max(0, downloadPage - 1); paginateDownloads(); }); downloadBack.setAttribute("aria-label", "앞 다운로드 목록");
    const downloadPageText = node("span");
    const downloadNext = button("›", "story-small", () => { downloadPage++; paginateDownloads(); }); downloadNext.setAttribute("aria-label", "뒤 다운로드 목록");
    downloadPages.append(downloadBack, downloadPageText, downloadNext);
    const voiceCounts = node("div", undefined, "story-voice-counts");
    const historic = node("div", undefined, "story-history");
    const historicalTitle = node("p", undefined, "story-history-title"), historicalDetail = node("p", undefined, "story-history-detail");
    historic.append(historicalTitle, historicalDetail);
    const listenActions = node("div", undefined, "story-listen-actions");
    const referenceButton = button("목소리 듣기", "story-secondary", () => openListening("reference"));
    const hardButton = button("유사어 전체 듣기", "story-secondary", () => openListening("hard"));
    listenActions.append(referenceButton, hardButton);
    live.append(liveSummary, voiceCounts, downloads, downloadPages, review, input, results, historic, listenActions);
    // Keep every legacy ID in the live document. Core rendering and event
    // bindings also visit hidden screens, so detaching inactive panels would
    // make document.getElementById return null.
    content.append(entry, scale, materials, live);
    const footer = node("div", undefined, "story-actions");
    const back = button("이전", "story-button", () => navigate(order.indexOf(current) - 1));
    const next = button("다음", "story-button", advance);
    footer.append(back, next); panel.append(heading, context, content, footer);
    const loader = node("div", undefined, "story-loader"); loader.hidden = true; loader.setAttribute("role", "status");
    const loaderDots = node("span", undefined, "story-loading-dots"); loaderDots.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) loaderDots.append(node("i"));
    const loaderText = node("span"); loader.append(loaderDots, loaderText);
    stage.append(panel, loader); shell.append(atmosphere, toolbar, notices, index, stage);
    document.body.append(shell);
    const settingsDialog = makeDialog("세부 설정");
    const settingSummary = node("div", undefined, "story-setting-summary");
    const location = $("storage-directory").parentElement;
    settingSummary.append(location, $("system-summary"));
    settingsDialog.content.append(settingSummary, $("advanced"), $("setup-voices"), $("storage-budget-details"));
    const timingDetails = statusCard.querySelector(".task-details");
    if (timingDetails) settingsDialog.content.append(timingDetails);
    const logDialog = makeDialog("작업 기록"); logDialog.content.append($("log-details"));
    const listenDialog = makeDialog("목소리 듣기");
    listenDialog.content.classList.add("story-listen-body");
    listenDialog.content.append($("reference-preview"), $("hard-preview"));
    listenDialog.dialog.addEventListener("close", () => {
      listening = null; bridge.pause($("reference-preview-frame")); bridge.pause($("hard-preview-frame"));
      $("reference-preview").open = false; $("hard-preview").open = false;
      render();
    });
    let current = "entry", projectKey = "", maximum = 2, following = true, transitioning = false, transitionTimer = null, creationPending = false;
    let materialTab = "training", materialPage = 0, materialKey = "", materialData = null, downloadPage = 0, listening = null;
    let lastFrontier = "", lastProject = null;
    const ticks = definitions.map(([id, label], position) => {
      const tick = button(undefined, "story-index-step", () => navigate(position));
      tick.dataset.step = id; tick.setAttribute("aria-label", label);
      tick.append(node("span", undefined, "story-index-tick"), node("span", label, "story-index-label"));
      tick.addEventListener("pointerenter", () => emphasize(position));
      tick.addEventListener("focus", () => emphasize(position));
      tick.addEventListener("pointerleave", () => emphasize(-1));
      tick.addEventListener("blur", () => emphasize(-1));
      index.append(tick); return tick;
    });
    for (const [id, label] of [["training", "학습용"], ["evaluation", "공통 평가용"]]) {
      const tab = button(label, "story-material-tab", () => { materialTab = id; materialPage = 0; renderMaterials(); });
      tab.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        materialTab = event.key === "Home" ? "training" : event.key === "End" ? "evaluation" : materialTab === "training" ? "evaluation" : "training";
        materialPage = 0; renderMaterials();
        [...materialsTabs.children].find(button => button.dataset.group === materialTab)?.focus();
      });
      tab.dataset.group = id; tab.setAttribute("role", "tab"); materialsTabs.append(tab);
    }
    function makeDialog(title) {
      const dialog = node("dialog", undefined, "story-dialog");
      const top = node("div", undefined, "story-dialog-heading"), titleNode = node("h2", title);
      const close = button("닫기", "story-tool", () => dialog.close());
      top.append(titleNode, close); const body = node("div", undefined, "story-dialog-content");
      dialog.append(top, body); shell.append(dialog);
      dialog.addEventListener("cancel", () => { for (const frame of dialog.querySelectorAll("iframe")) bridge.pause(frame); });
      dialog.addEventListener("close", () => { for (const frame of dialog.querySelectorAll("iframe")) bridge.pause(frame); });
      return { dialog, content: body, title: titleNode };
    }
    function openDialog(value) { if (!value.dialog.open) value.dialog.showModal(); }
    function pauseMedia() {
      for (const id of ["review-frame", "reference-preview-frame", "hard-preview-frame", "result-frame"]) bridge.pause($(id));
      if (listenDialog.dialog.open) listenDialog.dialog.close();
    }
    function emphasize(position) {
      ticks.forEach((tick, i) => {
        const distance = Math.abs(i - position), emphasis = position >= 0 && distance <= 2;
        tick.classList.toggle("is-emphasized", position === i);
        tick.style.setProperty("--tick-width", position < 0 ? "8px" : distance === 0 ? "40px" : distance === 1 ? "26px" : distance === 2 ? "16px" : "8px");
        tick.style.setProperty("--tick-opacity", position === i ? ".95" : emphasis ? ".6" : ".4");
      });
    }
    function stopTransition() {
      clearTimeout(transitionTimer); transitionTimer = null; transitioning = false; loader.hidden = true;
      panel.classList.remove("is-leaving"); panel.inert = false; panel.removeAttribute("aria-busy");
    }
    function navigate(position, automatic = false) {
      const state = bridge.state();
      if (transitioning || position < 0 || position > maximum || position >= order.length || state.busy || position === order.indexOf(current)) return;
      if (!state.project && position > 0 && !$("wakeword").value.trim()) { bridge.message("호출어를 입력해 주세요.", true); $("wakeword").focus(); return; }
      const backwards = position < order.indexOf(current);
      pauseMedia(); transitioning = true; panel.inert = true; panel.setAttribute("aria-busy", "true");
      panel.classList.add("is-leaving"); loaderText.textContent = backwards ? "전 단계로 이동 중" : "다음 단계로 이동 중"; loader.hidden = false;
      following = automatic || (state.project && position === frontier(state.project));
      transitionTimer = setTimeout(() => {
        current = order[position]; stopTransition(); render(); heading.focus({ preventScroll: true });
      }, 1500);
    }
    async function advance() {
      const state = bridge.state(), project = state.project, position = order.indexOf(current);
      if (transitioning || state.busy) return;
      if (!project) {
        if (current === "materials") {
          // Start real preparation immediately. The presentation waits for
          // both its result and the same minimum used by local transitions.
          // Polls may receive the created model meanwhile, but must not reveal
          // it early or allow another creation request.
          pauseMedia(); creationPending = true; transitioning = true;
          panel.inert = true; panel.setAttribute("aria-busy", "true"); panel.classList.add("is-leaving");
          loaderText.textContent = "다음 단계로 이동 중"; loader.hidden = false;
          const minimum = new Promise(resolve => { transitionTimer = setTimeout(resolve, 1500); });
          try {
            const result = await Promise.allSettled([minimum, (async () => bridge.create())()]);
            if (result[1].status === "rejected") bridge.message(result[1].reason?.message || "모델 준비를 시작하지 못했습니다. 다시 확인해 주세요.", true);
          } finally {
            creationPending = false; stopTransition(); render(); heading.focus({ preventScroll: true });
          }
          return;
        }
        navigate(position + 1); return;
      }
      if (position < maximum) { navigate(position + 1); return; }
      if (running(project.status)) { await bridge.action("pause"); return; }
      if (project.review || project.input) return;
      if (project.migration) { if (project.migration.state === "available") await bridge.action("migrate"); return; }
      if (["paused", "failed", "error", "interrupted"].includes(project.status)) await bridge.action("resume");
      else if (!done(project.status)) await bridge.action("start");
    }
    function openListening(kind) {
      if (!lastProject || transitioning) return;
      listening = kind; listenDialog.title.textContent = kind === "hard" ? "유사어 전체 듣기" : "목소리 듣기";
      bridge.preview(kind);
      $("reference-preview").hidden = kind !== "reference";
      $("hard-preview").hidden = kind !== "hard";
      $(kind === "hard" ? "hard-preview" : "reference-preview").open = true;
      bridge.view("review", kind); openDialog(listenDialog);
    }
    function setView(view, mode) { if (!listening) bridge.view(view, mode); }
    function renderMaterials() {
      if (!materialData) return;
      const small = matchMedia("(max-width:760px), (max-height:700px)").matches;
      const size = innerHeight < 560 ? 2 : innerHeight < 720 ? 4 : 6;
      const activeRows = materialData[materialTab] || [], pages = Math.max(1, Math.ceil(activeRows.length / size));
      materialPage = Math.min(materialPage, pages - 1);
      const sections = [];
      for (const [key, label] of [["training", "학습용 자료"], ["evaluation", "성능 평가용 자료"]]) {
        const section = node("section", undefined, "story-material-section"); section.hidden = small && key !== materialTab;
        const head = node("div", undefined, "story-material-heading");
        head.append(node("h2", label), node("span", key === "training" ? materialData.label : "공통 평가"));
        const list = node("dl", undefined, "story-material-list"), rows = materialData[key] || [];
        const shown = small ? rows.slice(materialPage * size, (materialPage + 1) * size) : rows;
        for (const item of shown) {
          const row = node("div", undefined, "story-material-row"), amount = node("dd");
          const names = [...new Set((item.sources || []).map(source => source.name === "RIRS_NOISES" ? "OpenSLR RIR" : source.name))];
          row.append(node("dt", item.name)); amount.append(node("strong", item.amount), node("span", names.join(" · ") || item.description)); row.append(amount); list.append(row);
        }
        section.append(head, list); sections.push(section);
      }
      materialColumns.replaceChildren(...sections);
      for (const tab of materialsTabs.children) { tab.setAttribute("aria-selected", String(tab.dataset.group === materialTab)); tab.tabIndex = tab.dataset.group === materialTab ? 0 : -1; }
      materialPages.hidden = !small || pages <= 1; materialBack.disabled = materialPage === 0; materialNext.disabled = materialPage >= pages - 1;
      materialPageText.textContent = `${materialPage + 1} / ${pages}`;
    }
    function paginateDownloads() {
      const groups = [...$("downloads-list").children], small = innerHeight < 720 || innerWidth < 760;
      const size = innerHeight < 500 ? 1 : innerHeight < 600 ? 2 : 4, pages = Math.max(1, Math.ceil(groups.length / size));
      downloadPage = Math.min(downloadPage, pages - 1);
      groups.forEach((group, i) => { group.hidden = small && Math.floor(i / size) !== downloadPage; });
      downloadPages.hidden = !small || pages <= 1 || downloads.hidden;
      downloadBack.disabled = downloadPage === 0; downloadNext.disabled = downloadPage >= pages - 1;
      downloadPageText.textContent = `${downloadPage + 1} / ${pages}`;
    }
    function renderVoiceCounts(project) {
      const targets = project.reference_counts || {}, preview = project.reference_preview || {};
      const values = [["학습용 목소리", "train"], ["공통 평가용 목소리", "validation"]].map(([label, key]) => {
        const row = node("div", undefined, "story-voice-row"), amount = node("span");
        row.append(node("strong", label)); amount.textContent = `${count(preview[key] ?? 0)} / ${count(targets[key])}개`; row.append(amount);
        if (Number.isFinite(targets[key]) && targets[key] > 0) { const bar = node("progress"); bar.max = targets[key]; bar.value = preview[key] || 0; bar.setAttribute("aria-label", label + " 파일 검사 통과"); row.append(bar); }
        return row;
      });
      voiceCounts.replaceChildren(...values);
    }
    function renderHistory(project, position) {
      const isComplete = done(project.status) || position < maximum;
      historicalTitle.textContent = isComplete ? "이 단계를 마쳤습니다." : "진행 상태를 확인하고 있습니다.";
      const settings = project.settings || {}, text = {
        entry: `호출어 · ${project.wakeword || ""}`,
        scale: titles[project.preset] || project.preset_label || "저장된 학습 규모",
        downloads: "다운로드·검사·자료 연결 완료",
        voices: "자동 검사한 목소리를 호출어 음성 생성에 사용합니다.",
        pronunciation: "호출어 발음 검수 결과가 저장되어 있습니다.",
        "positive-generation": [Number.isFinite(settings.originals) ? `학습용 ${count(settings.originals)}개` : "", Number.isFinite(settings.validation_originals) ? `공통 평가용 ${count(settings.validation_originals)}개` : ""].filter(Boolean).join(" · "),
        phrases: "저장한 유사어 문구로 음성을 생성했습니다.",
        "hard-generation": Number.isFinite(settings.hard_samples_per_phrase) ? `유사어 문구당 ${count(settings.hard_samples_per_phrase)}개` : "유사어 음성 생성 완료",
        "hard-review": "유사어 발음 검수 결과가 저장되어 있습니다.",
        features: "소음·울림 증강과 학습 특징 준비 완료",
        training: Number.isFinite(settings.updates) ? `모델 3개 · 모델당 ${count(settings.updates)}회 학습` : "독립 모델 3개 학습 완료",
        evaluation: "저장된 채점과 모델별 감지 설정을 결과 화면에서 확인하세요.",
      };
      historicalDetail.textContent = text[current] || "기존 작업과 기록은 그대로 보존됩니다.";
    }
    function render() {
      if (creationPending) return;
      const state = bridge.state(), project = state.project;
      const key = project?.id || "setup";
      if (key !== projectKey) {
        pauseMedia(); stopTransition(); projectKey = key; following = true; materialPage = 0; materialKey = "";
        maximum = frontier(project); current = project ? stageStep(project) : "entry"; lastFrontier = project ? stageStep(project) : "entry";
      }
      lastProject = project; maximum = frontier(project);
      bridge.voiceDesign?.(project ? project.reference_design : state.recipes.find(recipe => recipe.id === state.selectedPreset)?.reference_design);
      const actual = project ? stageStep(project) : current;
      if (project && actual !== lastFrontier) {
        lastFrontier = actual;
        if (following && !transitioning && current !== actual) navigate(order.indexOf(actual), true);
      }
      if (order.indexOf(current) > maximum) current = actual;
      const position = order.indexOf(current), historical = Boolean(project && (position < maximum || done(project.status)));
      shell.dataset.step = current; shell.dataset.model = project ? "existing" : "new";
      heading.textContent = definitions[position]?.[2] || "학습 상태";
      document.title = definitions[position]?.[1] || "한국어 호출어 학습";
      context.textContent = project && position >= 3 ? `‘${project.wakeword || ""}’ · ${titles[project.preset] || "호출어 학습"}` : "";
      context.hidden = !context.textContent;
      logButton.hidden = !project;
      index.hidden = position === 0;
      ticks.forEach((tick, i) => {
        tick.hidden = i > position; tick.disabled = transitioning || state.busy || i > maximum;
        tick.setAttribute("aria-current", i === position ? "step" : "false");
      });
      if (!transitioning) {
        back.hidden = position === 0; back.disabled = state.busy || position === 0;
        next.hidden = current === "microphone";
        next.disabled = state.busy;
        next.textContent = "다음"; next.title = "";
        if (!project) next.disabled = state.busy || !state.recipeReady || (current === "materials" && $("create").disabled);
        else if (position >= maximum && !done(project.status)) {
          if (running(project.status)) { next.textContent = "일시정지"; next.title = "현재 단계가 끝난 뒤 멈춥니다."; next.disabled = state.busy || project.status === "pausing"; }
          else if (project.review || project.input) { next.textContent = "다음"; next.disabled = true; }
          else if (project.migration) { next.textContent = "업데이트 적용"; next.disabled = state.busy || project.migration.state !== "available"; }
          else next.textContent = ["paused", "failed", "error", "interrupted"].includes(project.status) ? "이어서 진행" : "시작";
        }
      }
      // Hidden panels stay connected; polling never rebuilds their inputs or
      // embedded documents, and never resets a review's listening position.
      const wanted = !project && current === "entry" ? entry : !project && current === "scale" ? scale : current === "materials" ? materials : live;
      for (const unit of [entry, scale, materials, live]) unit.hidden = unit !== wanted;
      if (!project) {
        for (const preset of $("presets").children) {
          const radio = preset.querySelector('input[name="preset"]');
          const description = preset.querySelector(".preset-description");
          if (radio) { preset.setAttribute("data-selected", String(radio.checked)); if (description) description.textContent = shortDescriptions[radio.value] || description.textContent; }
        }
        settingsDialog.content.querySelector("#advanced").hidden = false;
      } else {
        // Existing model settings are displayed, never overwritten by a newly
        // selected preset or by the setup form's advanced-value overrides.
        $("advanced").hidden = true;
        if (current === "results" || current === "microphone") setView(current === "microphone" ? "mic" : "results");
        else if (!historical && (project.review || project.input)) setView("review", "required");
        else setView("overview");
        const frameStep = ["results", "microphone"].includes(current);
        results.hidden = !frameStep || !(project.results?.url || project.results_url);
        review.hidden = historical || frameStep || !project.review || running(project.status);
        input.hidden = historical || frameStep || !project.input || running(project.status);
        liveSummary.hidden = historical || frameStep;
        currentTitle.hidden = true;
        downloads.hidden = current !== "downloads" || historical || !project.progress?.files?.length;
        voiceCounts.hidden = current !== "voices";
        historic.hidden = !historical || frameStep || current === "voices";
        if (!historic.hidden) renderHistory(project, position);
        if (!voiceCounts.hidden) renderVoiceCounts(project);
        referenceButton.hidden = !project.reference_preview?.url || !["voices", "pronunciation"].includes(current);
        hardButton.hidden = !project.hard_preview?.url || !["hard-generation", "hard-review", "phrases"].includes(current);
        listenActions.hidden = referenceButton.hidden && hardButton.hidden;
        paginateDownloads();
      }
      if (current === "materials") {
        const recipe = state.recipes.find(recipe => recipe.id === (project?.preset || state.selectedPreset));
        const values = project ? { ...recipe, ...project.settings, evaluation_policy: project.evaluation_policy || recipe?.evaluation_policy } : recipe;
        const fingerprint = JSON.stringify([values, !project ? [$("adv-originals").value, $("adv-generic").value, $("adv-korean").value] : null]);
        if (fingerprint !== materialKey) {
          materialKey = fingerprint;
          materialData = values ? { ...bridge.materials(values, Boolean(project)), label: titles[project?.preset || state.selectedPreset] || "학습 규모" } : null;
          if (project && materialData && project.settings) {
            const s = project.settings, training = materialData.training;
            const update = (id, value) => { const item = training.find(row => row.id === id); if (item && value !== undefined) item.amount = value; };
            update("positive", Number.isFinite(s.originals) ? `${count(s.originals)}개` : undefined);
            update("generic", Number.isFinite(s.generic_hours) ? `${count(s.generic_hours)}시간` : undefined);
            update("korean", Number.isFinite(s.korean_hours) ? `${count(s.korean_hours)}시간` : undefined);
            update("hard", Number.isFinite(s.hard_samples_per_phrase) ? `문구당 ${count(s.hard_samples_per_phrase)}개` : undefined);
          }
          renderMaterials();
        }
      }
      const details = project ? `${project.directory || ""}` : "";
      settingSummary.title = details;
      if (timingDetails) timingDetails.hidden = !project;
      $("storage-budget-details").hidden = Boolean(project) || !$("storage-budget-content").children.length;
      settingsDialog.content.querySelector(".storage-location").hidden = Boolean(project);
      if (project) {
        let saved = settingsDialog.content.querySelector(".story-saved-settings");
        if (!saved) { saved = node("p", undefined, "story-saved-settings"); settingSummary.prepend(saved); }
        saved.textContent = [project.wakeword, titles[project.preset], project.directory].filter(Boolean).join(" · ");
      } else settingsDialog.content.querySelector(".story-saved-settings")?.remove();
      if (listening) {
        $("reference-preview").hidden = listening !== "reference";
        $("hard-preview").hidden = listening !== "hard";
      }
    }
    $("wakeword").addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.isComposing && current === "entry") { event.preventDefault(); advance(); }
    });
    window.addEventListener("resize", () => { if (current === "materials") renderMaterials(); if (current === "downloads") paginateDownloads(); });
    window.addEventListener("pagehide", () => { clearTimeout(transitionTimer); pauseMedia(); });
    return { render };
  }
  window.KowwStory = Object.freeze({ create, definitions, stageStep, frontier });
})();
