"use strict";
/* Presentation adapter. Worker state and all human decisions stay authoritative. */
(() => {
  const definitions = [
    ["entry", "호출어 입력", "호출어를 입력해 주세요."],
    ["scale", "학습 규모 선택", "학습 규모를 선택해 주세요."],
    ["materials", "준비할 자료", "준비할 자료를 확인해 주세요."],
    ["downloads", "자료 다운로드", "자료를 다운로드합니다."],
    ["voices", "합성 목소리 준비", "합성 목소리를 준비합니다."],
    ["voice-listening", "합성 목소리 듣기", "합성 목소리를 들어보세요."],
    ["pronunciation", "호출어 발음 확인", "호출어 발음을 확인하세요."],
    ["positive-generation", "호출어 음성 생성", "호출어 음성을 생성합니다."],
    ["phrases", "유사어 입력", "유사어 문구를 입력하세요."],
    ["hard-generation", "유사어 음성 생성", "유사어 음성을 생성합니다."],
    ["hard-review", "유사어 발음 확인", "유사어 발음을 확인하세요."],
    ["features", "증강·특징 추출", "소음·울림을 적용하고 특징을 추출합니다."],
    ["training", "모델 학습", "모델을 학습합니다."],
    ["evaluation", "성능 평가", "모델의 성능을 평가합니다."],
    ["results", "학습 결과", "학습 결과를 확인하세요."],
    ["microphone", "마이크 시험", "마이크로 호출어를 시험하세요."],
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
  const downloadCheckpoint = project => project?.download_checkpoint?.policy === "explicit_download_v1" ? project.download_checkpoint : null;
  const navigationCheckpoint = project => project?.navigation_checkpoint?.policy === "explicit_story_v1" ? project.navigation_checkpoint : null;
  const navigationGroups = [
    ["downloads", ["frontend", "data_download", "data_review", "data_connect"]],
    ["voices", ["references", "references_review"]], ["pronunciation", ["pilot", "pilot_review"]],
    ["positive-generation", ["positives"]], ["phrases", ["hard_phrases"]],
    ["hard-generation", ["hard_generation"]], ["hard-review", ["hard_review", "hard_connect"]],
    ["features", ["features"]], ["training", ["training", "package"]],
    ["evaluation", ["development", "evaluation"]], ["results", ["results"]],
  ];
  const stageStep = project => done(project?.status) ? "results"
    : navigationCheckpoint(project)?.state === "awaiting_next" && order.includes(navigationCheckpoint(project).step) ? navigationCheckpoint(project).step
    : downloadCheckpoint(project)?.state === "awaiting_next" ? "downloads" : mapping[stageId(project)] || "downloads";
  function completedStep(project, step) {
    if (step === "voice-listening") step = "voices";
    const records = Array.isArray(project?.stages) ? project.stages.filter(record => mapping[record.id] === step) : [];
    return records.length > 0 && records.every(record => done(record.status));
  }
  function completedNavigationStep(project, step) {
    if (step === "voice-listening") step = "voices";
    if (!navigationCheckpoint(project)) return completedStep(project, step);
    const position = navigationGroups.findIndex(([id]) => id === step);
    return position >= 0 && navigationGroups.slice(0, position + 1).flatMap(([, stages]) => stages)
      .every(id => project.stages?.some(record => record.id === id && done(record.status)));
  }
  function frontier(project) {
    if (!project) return 2;
    if (done(project.status)) return order.indexOf((project.results?.mic_url || project.mic_url) ? "microphone" : "results");
    // Only actual current/completed backend stages unlock later screens.
    const current = stageStep(project), position = Math.max(3, order.indexOf(current));
    const navigation = navigationCheckpoint(project);
    return current === "voices" && completedNavigationStep(project, "voices")
      && (!navigation || (navigation.state === "awaiting_next" && project.status === "paused")) ? order.indexOf("voice-listening") : position;
  }
  const titles = { check: "동작 확인용", trial: "시험 모델용", production: "상용 모델용" };
  const shortDescriptions = { check: "라이브러리 동작 확인", trial: "호출 성능 비교·시험", production: "많은 자료로 본격 학습" };
  const count = value => Number.isFinite(value) ? value.toLocaleString("ko-KR") : "—";
  function pageCapacity(items, available, rowHeight, columns = 1, gap = 12, pagerSpace = 44) {
    if (!items) return 1;
    const rows = Math.ceil(items / columns);
    if (rows * rowHeight + Math.max(0, rows - 1) * gap <= available) return items;
    // Reserve the pager only when it is needed. Use a fixed count on all
    // pages, including pages whose longer labels make their rows taller.
    return Math.max(1, Math.floor((Math.max(0, available - pagerSpace) + gap) / (rowHeight + gap))) * columns;
  }
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
    // A saved model is a read-only view; leave the new-model draft untouched.
    const savedEntry = node("div", undefined, "story-entry");
    const savedWord = node("input"); savedWord.id = "story-saved-wakeword";
    savedWord.type = "text"; savedWord.readOnly = true;
    savedWord.setAttribute("aria-label", "호출어"); savedEntry.append(savedWord);
    const scale = node("div", undefined, "story-scale"); scale.append($("presets"));
    const savedScale = node("div", undefined, "story-scale story-saved-scale");
    const savedPresets = node("div", undefined, "preset-grid"); savedPresets.id = "story-saved-presets";
    savedPresets.setAttribute("role", "group"); savedPresets.setAttribute("aria-label", "저장된 학습 규모"); savedScale.append(savedPresets);
    for (const [id, title] of Object.entries(titles)) {
      const card = node("label", undefined, "preset"), radio = node("input");
      radio.type = "radio"; radio.name = "saved-preset"; radio.value = id; radio.disabled = true;
      const line = node("div"); line.append(radio, node("span", title, "preset-title"));
      card.append(line, node("p", shortDescriptions[id], "preset-description")); savedPresets.append(card);
    }
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
    const migrationNote = $("migration-note");
    liveSummary.append(statusCard, $("counts"));
    const downloads = $("downloads-section"), review = $("review-section"), input = $("input-section"), results = $("results-section");
    const downloadPages = node("div", undefined, "story-pagination");
    const downloadBack = button("‹", "story-small", () => { downloadPage = Math.max(0, downloadPage - 1); paginateDownloads(); }); downloadBack.setAttribute("aria-label", "앞 다운로드 목록");
    const downloadPageText = node("span");
    const downloadNext = button("›", "story-small", () => { downloadPage++; paginateDownloads(); }); downloadNext.setAttribute("aria-label", "뒤 다운로드 목록");
    downloadPages.append(downloadBack, downloadPageText, downloadNext);
    const downloadActions = node("div", undefined, "story-download-actions"); downloadActions.hidden = true;
    const downloadPrimary = button("다운로드 시작", "story-button story-download-primary", controlDownload); downloadPrimary.disabled = true;
    downloadActions.append(downloadPrimary);
    const voiceCounts = node("div", undefined, "story-voice-counts");
    const voiceListening = node("div", undefined, "story-voice-listening");
    const voiceListeningStatus = node("p", undefined, "story-listening-status"); voiceListeningStatus.hidden = true;
    const referencePreview = $("reference-preview");
    voiceListening.append(referencePreview, voiceListeningStatus);
    const historic = node("div", undefined, "story-history");
    const historyContent = node("div", undefined, "story-history-content");
    const historyPages = node("div", undefined, "story-pagination");
    const historyBack = button("‹", "story-small", () => { historyPage = Math.max(0, historyPage - 1); renderHistory(lastProject); }); historyBack.setAttribute("aria-label", "앞 완료 내역");
    const historyPageText = node("span");
    const historyNext = button("›", "story-small", () => { historyPage++; renderHistory(lastProject); }); historyNext.setAttribute("aria-label", "뒤 완료 내역");
    historyPages.append(historyBack, historyPageText, historyNext);
    const savedReview = node("div", undefined, "story-saved-review");
    const savedReviewCompletion = node("p", "검수한 음원과 저장된 판정을 확인할 수 있습니다.", "listen-completion");
    const savedReviewWorkspace = node("div", undefined, "listen-workspace");
    const savedReviewSidebar = node("aside", undefined, "listen-sidebar"); savedReviewSidebar.setAttribute("aria-label", "검수한 음원 목록");
    const savedReviewList = node("div", undefined, "listen-list");
    const savedReviewListPager = node("div", undefined, "listen-list-pager");
    const savedReviewListPrevious = button("‹", undefined, () => { historyReviewPage--; renderHistoryReviewList(true); }); savedReviewListPrevious.setAttribute("aria-label", "앞 음원 목록");
    const savedReviewListPosition = node("span");
    const savedReviewListNext = button("›", undefined, () => { historyReviewPage++; renderHistoryReviewList(true); }); savedReviewListNext.setAttribute("aria-label", "뒤 음원 목록");
    savedReviewListPager.append(savedReviewListPrevious, savedReviewListPosition, savedReviewListNext);
    savedReviewSidebar.append(savedReviewList, savedReviewListPager);
    const savedReviewPlayer = node("div", undefined, "listen-player");
    const savedReviewSelect = node("select", undefined, "listen-select"); savedReviewSelect.setAttribute("aria-label", "저장된 검수 음원");
    const savedReviewHeading = node("div", undefined, "listen-heading");
    const savedReviewText = node("strong", undefined, "listen-title"), savedReviewPosition = node("span", undefined, "listen-position"); savedReviewHeading.append(savedReviewText, savedReviewPosition);
    const savedReviewAudio = node("audio", undefined, "listen-audio"); savedReviewAudio.controls = true; savedReviewAudio.preload = "none";
    const savedReviewControls = node("div", undefined, "listen-controls");
    const savedReviewPrevious = button("이전", undefined, () => selectHistoryReview(historyReviewIndex - 1)); savedReviewPrevious.setAttribute("aria-label", "이전 음원");
    const savedReviewToggle = button("재생", undefined, () => historyPlaybackWanted || savedReviewAudio.paused === false ? stopHistoryPlayback() : playHistoryReview());
    const savedReviewNext = button("다음", undefined, () => selectHistoryReview(historyReviewIndex + 1)); savedReviewNext.setAttribute("aria-label", "다음 음원");
    savedReviewControls.append(savedReviewPrevious, savedReviewToggle, savedReviewNext);
    const savedReviewDecisions = node("div", undefined, "story-saved-review-decisions listen-decisions"); savedReviewDecisions.setAttribute("role", "group"); savedReviewDecisions.setAttribute("aria-label", "저장된 검수 결과");
    const savedReviewInputs = [];
    for (const [value, label] of [["keep", "정상"], ["discard", "제외"], ["review", "다시 확인"]]) {
      const choice = node("label"), radio = node("input"); radio.type = "radio"; radio.name = "saved-review-decision"; radio.value = value; radio.disabled = true;
      choice.append(radio, node("span", label)); savedReviewDecisions.append(choice); savedReviewInputs.push(radio);
    }
    const savedReviewNote = node("p", undefined, "story-history-note listen-status");
    savedReviewSelect.hidden = true;
    savedReviewPlayer.append(savedReviewHeading, savedReviewAudio, savedReviewControls, savedReviewSelect, savedReviewDecisions, savedReviewNote);
    savedReviewWorkspace.append(savedReviewSidebar, savedReviewPlayer);
    savedReview.append(savedReviewCompletion, savedReviewWorkspace);
    historic.append(historyContent, savedReview, historyPages);
    const listenActions = node("div", undefined, "story-listen-actions");
    const referenceButton = button("목소리 듣기", "story-secondary", () => openListening("reference"));
    const hardButton = button("유사어 전체 듣기", "story-secondary", () => openListening("hard"));
    listenActions.append(referenceButton, hardButton);
    // A completed stage hides live progress, but its compatibility update must
    // remain reachable through the existing explicit migration control.
    live.append(migrationNote, liveSummary, voiceCounts, downloads, downloadPages, downloadActions, review, input, results, historic, listenActions);
    // Keep every legacy ID in the live document. Core rendering and event
    // bindings also visit hidden screens, so detaching inactive panels would
    // make document.getElementById return null.
    content.append(entry, savedEntry, scale, savedScale, materials, live, voiceListening);
    const footer = node("div", undefined, "story-actions");
    const back = button("이전", "story-button", () => navigate(order.indexOf(current) - 1));
    const operation = button("일시정지", "story-button story-operation", controlJob); operation.hidden = true;
    const next = button("다음", "story-button", advance);
    footer.append(back, operation, next); panel.append(heading, context, content, footer);
    const loader = node("div", undefined, "story-loader"); loader.hidden = true; loader.setAttribute("role", "status");
    const loaderDots = node("span", undefined, "story-loading-dots"); loaderDots.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) loaderDots.append(node("i"));
    const loaderText = node("span"); loader.append(loaderDots, loaderText);
    stage.append(panel, loader); shell.append(atmosphere, toolbar, notices, index, stage);
    document.body.append(shell);
    window.KowwListening?.enhance(savedReviewAudio, savedReviewControls, {
      play: savedReviewToggle, previous: savedReviewPrevious, next: savedReviewNext,
    });
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
    listenDialog.content.append($("hard-preview"));
    listenDialog.dialog.addEventListener("close", () => {
      listening = null; bridge.pause($("reference-preview-frame")); bridge.pause($("hard-preview-frame"));
      $("hard-preview").open = false;
      render();
    });
    let current = "entry", projectKey = "", maximum = 2, setupReached = 0, following = true, transitioning = false, transitionSession = null, creationPending = false;
    let materialTab = "training", materialPage = 0, materialKey = "", materialData = null, downloadPage = 0, listening = null;
    let historyPage = 0, historyKey = "", historyReviewItems = [], historyReviewIndex = 0;
    let historyReviewPage = 0, historyReviewListKey = "", historyReviewRows = [];
    let historyPlaybackWanted = false, historyPlaybackVersion = 0;
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
      voiceListening.classList.remove("is-selecting");
      for (const id of ["review-frame", "reference-preview-frame", "hard-preview-frame", "result-frame"]) bridge.pause($(id));
      stopHistoryPlayback(true);
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
      const session = transitionSession; transitionSession = null;
      if (session) { session.cancel(); for (const timer of session.timers) clearTimeout(timer); }
      transitioning = false; creationPending = false; loader.hidden = true;
      loader.classList.remove("is-finishing");
      panel.classList.remove("is-leaving"); panel.inert = false; panel.removeAttribute("aria-busy");
    }
    function activeTransition(session) { return transitionSession === session; }
    function beginTransition(backwards) {
      pauseMedia();
      const session = { timers: new Set() };
      session.aborted = new Promise(resolve => { session.cancel = () => resolve(false); });
      transitionSession = session; transitioning = true;
      panel.inert = true; panel.setAttribute("aria-busy", "true"); panel.classList.add("is-leaving");
      loaderText.textContent = backwards ? "전 단계로 이동 중" : "다음 단계로 이동 중";
      return session;
    }
    function cssMilliseconds(value) {
      return String(value || "0s").split(",").map(part => {
        const number = parseFloat(part); return Number.isFinite(number) ? number * (part.trim().endsWith("ms") ? 1 : 1000) : 0;
      });
    }
    function cssMotionTime(element, prefix) {
      const style = typeof getComputedStyle === "function" ? getComputedStyle(element) : {};
      const durations = cssMilliseconds(style[`${prefix}Duration`]), delays = cssMilliseconds(style[`${prefix}Delay`]);
      return Math.max(0, ...durations.map((duration, index) => duration + delays[index % delays.length]));
    }
    async function waitForDuration(session, delay) {
      if (!activeTransition(session)) return false;
      if (delay <= 0) return true;
      let timer;
      const ready = new Promise(resolve => { timer = setTimeout(() => resolve(true), delay); session.timers.add(timer); });
      const result = await Promise.race([ready, session.aborted]);
      clearTimeout(timer); session.timers.delete(timer);
      return result && activeTransition(session);
    }
    async function waitForMotion(element, session) {
      if (!activeTransition(session)) return false;
      // getAnimations() includes only this element; the loader's infinite dots
      // are children and must never hold up navigation.
      if (typeof element.getAnimations !== "function") {
        const duration = Math.max(cssMotionTime(element, "transition"), cssMotionTime(element, "animation"));
        return waitForDuration(session, duration);
      }
      const animations = element.getAnimations().filter(animation => {
        const end = animation.effect?.getComputedTiming?.().endTime;
        return (!animation.effect?.target || animation.effect.target === element)
          && animation.playState !== "finished" && animation.playState !== "idle" && Number.isFinite(end);
      });
      if (!animations.length) return true;
      const remaining = Math.max(...animations.map(animation => Math.max(0,
        (animation.effect.getComputedTiming().endTime - Number(animation.currentTime || 0)) / Math.abs(animation.playbackRate || 1))));
      let timer;
      const watchdog = new Promise(resolve => {
        timer = setTimeout(() => {
          if (!activeTransition(session)) { resolve(false); return; }
          // Exceptional recovery only: reach the final visual state before
          // proceeding if a browser never delivers its completion promise.
          try { for (const animation of animations) animation.finish(); resolve(true); }
          catch { resolve(false); }
        }, Math.max(1000, remaining + 500));
        session.timers.add(timer);
      });
      const finished = Promise.allSettled(animations.map(animation => animation.finished))
        .then(results => results.every(result => result.status === "fulfilled"));
      const result = await Promise.race([finished, watchdog, session.aborted]);
      clearTimeout(timer); session.timers.delete(timer);
      return result && activeTransition(session);
    }
    async function runTransition(session, preparation, commit) {
      const prepared = Promise.resolve(preparation).then(value => ({ ok: true, value }), error => ({ ok: false, error }));
      try {
        if (!await waitForMotion(panel, session)) return;
        loader.hidden = false;
        if (!await waitForMotion(loader, session)) return;
        // Hold the fully visible notice for at least 0.8 seconds. Server
        // preparation runs concurrently and can keep it visible for longer.
        const [result, held] = await Promise.all([
          Promise.race([prepared, session.aborted]), waitForDuration(session, 800),
        ]);
        if (!held || !activeTransition(session)) return;
        if (!result.ok) bridge.message(result.error?.message || "모델 준비를 시작하지 못했습니다. 다시 확인해 주세요.", true);
        loader.classList.add("is-finishing");
        if (!await waitForMotion(loader, session)) return;
        commit?.();
      } catch (error) {
        if (activeTransition(session)) bridge.message("화면을 전환하지 못했습니다. 다시 시도해 주세요.", true);
      } finally {
        if (activeTransition(session)) { stopTransition(); render(); heading.focus({ preventScroll: true }); }
      }
    }
    function navigate(position, automatic = false) {
      const state = bridge.state();
      if (transitioning || position < 0 || position > maximum || position >= order.length || state.busy || position === order.indexOf(current)) return false;
      if (!state.project && position > 0 && !$("wakeword").value.trim()) { bridge.message("호출어를 입력해 주세요.", true); $("wakeword").focus(); return false; }
      const backwards = position < order.indexOf(current);
      const session = beginTransition(backwards);
      following = automatic || (state.project && position === frontier(state.project));
      void runTransition(session, undefined, () => { current = order[position]; historyPage = 0; historyKey = ""; });
      return true;
    }
    async function controlDownload() {
      const state = bridge.state(), project = state.project;
      if (transitioning || state.busy || !canStartDownloads(project)) return;
      await bridge.action("start");
    }
    function humanGate(project) { return Boolean(project?.review || project?.input || project?.migration); }
    function canStartDownloads(project) {
      const navigation = navigationCheckpoint(project), download = downloadCheckpoint(project);
      return Boolean(project && current === "downloads" && stageStep(project) === "downloads"
        && !humanGate(project) && !completedStep(project, "downloads")
        && (!navigation || navigation.state === "awaiting_start") && (!download || download.state === "awaiting_start")
        && ["ready", "idle", "pending", "created"].includes(project.status));
    }
    function releaseStep(project) {
      if (!project || humanGate(project) || project.status !== "paused") return "";
      const checkpoint = navigationCheckpoint(project);
      if (checkpoint?.state === "awaiting_next") {
        const step = current === "voice-listening" ? "voices" : current;
        if (step === checkpoint.step && completedNavigationStep(project, step)) return step;
      }
      if (current === "downloads" && downloadCheckpoint(project)?.state === "awaiting_next"
        && completedStep(project, "downloads")) return "downloads";
      return "";
    }
    function operationAction(project) {
      if (!project || humanGate(project) || current !== stageStep(project) || done(project.status)
        || navigationCheckpoint(project)?.state === "awaiting_next" || downloadCheckpoint(project)?.state === "awaiting_next"
        || ["entry", "scale", "materials", "voice-listening", "microphone"].includes(current)) return "";
      if (running(project.status)) return "pause";
      if (["paused", "failed", "error", "interrupted"].includes(project.status)) return "resume";
      if (current !== "downloads" && ["ready", "idle", "pending", "created"].includes(project.status)) return "start";
      return "";
    }
    async function controlJob() {
      const state = bridge.state(), action = operationAction(state.project);
      if (!action || transitioning || state.busy || state.project.status === "pausing") return;
      await bridge.action(action);
    }
    async function advance() {
      const state = bridge.state(), project = state.project, position = order.indexOf(current);
      if (transitioning || state.busy) return;
      if (!project) {
        if (current === "materials") {
          // Create the model without starting transfers. The presentation waits
          // for its result and the same minimum used by local transitions.
          // Polls may receive the created model meanwhile, but must not reveal
          // it early or allow another creation request.
          const session = beginTransition(false); creationPending = true;
          await runTransition(session, (async () => bridge.create({ navigationCheckpoints: true }))());
          return;
        }
        navigate(position + 1); return;
      }
      if (current === stageStep(project) && humanGate(project)) return;
      if (position < maximum) { navigate(position + 1); return; }
      if (releaseStep(project)) await bridge.action("resume");
    }
    function openListening(kind) {
      if (!lastProject || transitioning) return;
      if (kind === "reference") { navigate(order.indexOf("voice-listening")); return; }
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
      const list = $("downloads-list"), groups = [...list.children];
      downloadPages.hidden = true;
      if (downloads.hidden || !groups.length) return;
      // Measure with the pager removed and every row present. The download
      // panel has a definite remaining height, independent of the list size.
      groups.forEach(group => { group.hidden = false; });
      const style = window.getComputedStyle?.(list), liveStyle = window.getComputedStyle?.(live);
      const columns = style?.gridTemplateColumns?.trim().split(/\s+/).length || (innerWidth > 760 && innerHeight > 700 ? 2 : 1);
      const gap = parseFloat(style?.rowGap) || 0, liveGap = parseFloat(liveStyle?.rowGap) || 0;
      const sectionBounds = downloads.getBoundingClientRect?.(), listBounds = list.getBoundingClientRect?.();
      const available = sectionBounds && listBounds ? Math.max(0, sectionBounds.bottom - listBounds.top) : Math.max(80, innerHeight - 340);
      const rowHeight = Math.max(0, ...groups.map(group => group.getBoundingClientRect?.().height || 0)) || 80;
      const size = pageCapacity(groups.length, available, rowHeight, columns, gap, 32 + liveGap);
      const pages = Math.max(1, Math.ceil(groups.length / size));
      downloadPage = Math.min(downloadPage, pages - 1);
      groups.forEach((group, i) => { group.hidden = Math.floor(i / size) !== downloadPage; });
      downloadPages.hidden = pages <= 1;
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
    function quantity(value, unit = "개") {
      if (!Number.isFinite(value) || value < 0) return "—";
      if (unit === "bytes") {
        if (value < 1024) return `${count(value)} B`;
        const index = Math.min(4, Math.floor(Math.log(value) / Math.log(1024)));
        return `${(value / 1024 ** index).toLocaleString("ko-KR", { maximumFractionDigits: 1 })} ${["B", "KiB", "MiB", "GiB", "TiB"][index]}`;
      }
      const label = { files: "개", items: "개", samples: "개", updates: "회", rows: "행", hours: "시간", audio_hours: "시간" }[unit] || unit;
      return `${count(value)}${label}`;
    }
    function historyRow(row) {
      const item = node("div", undefined, "story-history-row");
      const line = node("div", undefined, "story-history-line");
      const totalKnown = Number.isFinite(row.total) && row.total > 0, doneKnown = Number.isFinite(row.done) && row.done >= 0;
      const amount = totalKnown ? `${doneKnown ? row.unit === "bytes" ? quantity(row.done, row.unit) : count(row.done) : "—"} / ${quantity(row.total, row.unit)}` : doneKnown ? quantity(row.done, row.unit) : "수량 기록 없음";
      line.append(node("strong", row.label || "자료"), node("span", amount)); item.append(line);
      if (totalKnown && doneKnown) {
        const bar = node("progress"); bar.max = row.total; bar.value = Math.min(row.total, row.done);
        bar.setAttribute("aria-label", (row.label || "자료") + " 저장된 진행률"); item.append(bar);
        line.append(node("span", `${Math.min(100, row.done / row.total * 100).toLocaleString("ko-KR", { maximumFractionDigits: 1 })}%`, "story-history-percent"));
      }
      return item;
    }
    function historyPlayerActive() {
      return !savedReview.hidden && !historic.hidden && !live.hidden && !transitioning
        && ["pronunciation", "hard-review"].includes(current) && !document.hidden;
    }
    function historyControls() {
      savedReviewPrevious.disabled = historyReviewIndex <= 0;
      savedReviewNext.disabled = historyReviewIndex >= historyReviewItems.length - 1;
      const item = historyReviewItems[historyReviewIndex];
      savedReviewToggle.disabled = !bridge.mediaUrl?.(item?.audio_url || item?.audio);
      const playing = historyPlaybackWanted || savedReviewAudio.paused === false;
      savedReviewToggle.textContent = playing ? "일시정지" : "재생";
      savedReviewToggle.setAttribute("aria-pressed", String(playing));
    }
    function stopHistoryPlayback(release = false) {
      historyPlaybackWanted = false; historyPlaybackVersion++; savedReviewAudio.pause?.();
      if (release) { savedReviewAudio.removeAttribute("src"); savedReviewAudio.load?.(); }
      historyControls();
    }
    async function playHistoryReview() {
      if (!historyPlayerActive()) return;
      if (!savedReviewAudio.getAttribute("src")) updateHistoryReview();
      if (!savedReviewAudio.getAttribute("src")) return;
      const version = ++historyPlaybackVersion;
      historyPlaybackWanted = true; historyControls();
      try {
        await savedReviewAudio.play();
        if (version !== historyPlaybackVersion || !historyPlayerActive() || !historyPlaybackWanted) {
          if (!historyPlayerActive() || !historyPlaybackWanted) savedReviewAudio.pause?.();
          return;
        }
        historyControls();
      } catch {
        if (version === historyPlaybackVersion) {
          historyPlaybackWanted = false; historyControls(); savedReviewNote.textContent = "음원을 재생하지 못했습니다.";
        }
      }
    }
    function selectHistoryReview(position, continuePlayback = historyPlaybackWanted || savedReviewAudio.paused === false) {
      if (!Number.isInteger(position) || position < 0 || position >= historyReviewItems.length) return;
      historyReviewIndex = position; return updateHistoryReview(continuePlayback);
    }
    function renderHistoryReviewList(keepPage = false) {
      const key = JSON.stringify([lastProject?.id, current, historyReviewItems.map(item => [item.id, item.text, item.label, item.decision])]);
      if (key !== historyReviewListKey) {
        historyReviewListKey = key;
        const labels = { keep: "정상", discard: "제외", review: "다시 확인" };
        historyReviewRows = historyReviewItems.map((item, i) => {
          const row = button(undefined, "voice-row", () => selectHistoryReview(i));
          const label = current === "pronunciation" ? `목소리 ${i + 1}` : item.text || item.label || `음원 ${i + 1}`;
          row.dataset.number = String(i + 1);
          row.append(node("span", label, "listen-row-title"), node("span", labels[item.decision] || "기록 없음", "listen-badge"));
          const accessibleLabel = current === "pronunciation" ? label : `음원 ${i + 1} · ${label}`;
          row.setAttribute("aria-label", `${accessibleLabel} · ${labels[item.decision] || "판정 기록 없음"}`);
          return row;
        });
        savedReviewList.replaceChildren(...historyReviewRows);
      }
      const pages = Math.max(1, Math.ceil(historyReviewItems.length / 5));
      historyReviewPage = keepPage ? Math.max(0, Math.min(pages - 1, historyReviewPage)) : Math.floor(historyReviewIndex / 5);
      historyReviewRows.forEach((row, i) => {
        row.hidden = Math.floor(i / 5) !== historyReviewPage;
        row.setAttribute("aria-current", String(i === historyReviewIndex)); row.setAttribute("aria-pressed", String(i === historyReviewIndex));
      });
      savedReviewListPrevious.disabled = historyReviewPage === 0; savedReviewListNext.disabled = historyReviewPage >= pages - 1;
      savedReviewListPosition.textContent = `${historyReviewPage + 1} / ${pages}`; savedReviewListPager.hidden = pages <= 1;
    }
    function updateHistoryReview(continuePlayback = false) {
      const item = historyReviewItems[historyReviewIndex];
      if (!item) return;
      stopHistoryPlayback();
      const label = current === "pronunciation" ? `목소리 ${historyReviewIndex + 1}` : item.text || item.label || `음원 ${historyReviewIndex + 1}`;
      savedReviewText.textContent = label;
      savedReviewPosition.textContent = `${historyReviewIndex + 1} / ${historyReviewItems.length}`;
      savedReviewSelect.value = String(historyReviewIndex);
      const source = bridge.mediaUrl?.(item.audio_url || item.audio) || "";
      if (source) savedReviewAudio.setAttribute("src", source); else savedReviewAudio.removeAttribute("src");
      savedReviewAudio.hidden = !source; savedReviewAudio.load?.();
      for (const radio of savedReviewInputs) { radio.checked = radio.value === item.decision; radio.parentElement.setAttribute("data-checked", String(radio.checked)); }
      savedReviewNote.textContent = !source ? "저장된 음원을 찾을 수 없습니다." : !["keep", "discard", "review"].includes(item.decision) ? "판정 기록이 없습니다." : "";
      renderHistoryReviewList();
      historyControls();
      if (continuePlayback && source) return playHistoryReview();
    }
    savedReviewSelect.addEventListener("change", () => {
      const position = Number(savedReviewSelect.value);
      return selectHistoryReview(position);
    });
    savedReviewAudio.addEventListener("play", () => {
      if (!historyPlayerActive() || !savedReviewAudio.getAttribute("src")) { stopHistoryPlayback(); return; }
      historyPlaybackWanted = true; historyControls();
    });
    savedReviewAudio.addEventListener("pause", () => {
      if (savedReviewAudio.paused && !savedReviewAudio.ended) { historyPlaybackWanted = false; historyPlaybackVersion++; historyControls(); }
    });
    savedReviewAudio.addEventListener("ended", () => {
      if (savedReviewAudio.ended && historyPlaybackWanted && historyPlayerActive() && historyReviewItems.length) return selectHistoryReview((historyReviewIndex + 1) % historyReviewItems.length, true);
    });
    savedReviewAudio.addEventListener("error", () => {
      if (savedReviewAudio.getAttribute("src")) { stopHistoryPlayback(); savedReviewNote.textContent = "음원을 재생하지 못했습니다."; }
    });
    document.addEventListener?.("visibilitychange", () => { if (document.hidden) stopHistoryPlayback(true); });
    function historyHeight() {
      if (!stage.getBoundingClientRect || typeof getComputedStyle !== "function") return Math.max(80, innerHeight - 210);
      const rect = stage.getBoundingClientRect(), style = getComputedStyle(stage);
      const gap = parseFloat(getComputedStyle(panel).rowGap) || 0;
      const contextHeight = context.getBoundingClientRect().height;
      let available = rect.height - (parseFloat(style.paddingTop) || 0) - (parseFloat(style.paddingBottom) || 0)
        - heading.getBoundingClientRect().height - contextHeight - footer.getBoundingClientRect().height
        - gap * (contextHeight > 0 ? 3 : 2);
      if (!listenActions.hidden) available -= listenActions.getBoundingClientRect().height + (parseFloat(getComputedStyle(live).rowGap) || 0);
      if (!migrationNote.hidden) available -= migrationNote.getBoundingClientRect().height + (parseFloat(getComputedStyle(live).rowGap) || 0);
      return Math.max(80, available);
    }
    function renderHistory(project) {
      const view = project?.completed_stage_views?.[current];
      historic.dataset.kind = current;
      const valid = completedStep(project, current) && view?.completed === true;
      const fingerprint = JSON.stringify([project?.id, current, valid, view, project?.migration, historyPage, innerWidth, innerHeight]);
      if (fingerprint === historyKey) return;
      historyKey = fingerprint; savedReview.hidden = true; historyPages.hidden = true;
      const unknown = () => historyContent.replaceChildren(node("p", "저장된 세부 내역이 없습니다.", "story-history-note"));
      if (!valid) { stopHistoryPlayback(true); unknown(); return; }
      const reviewItems = Array.isArray(view.review?.items) ? view.review.items : [];
      if (["pronunciation", "hard-review"].includes(current) && reviewItems.length) {
        const sameQueue = JSON.stringify(historyReviewItems) === JSON.stringify(reviewItems);
        historyReviewItems = reviewItems;
        if (!sameQueue) historyReviewIndex = 0;
        savedReview.hidden = false; historyContent.replaceChildren();
        if (!sameQueue || savedReviewSelect.children.length !== reviewItems.length) {
          savedReviewSelect.replaceChildren(...reviewItems.map((item, i) => { const option = node("option", `${i + 1}. ${item.text || item.label || "음원"}`); option.value = String(i); return option; }));
          updateHistoryReview();
        } else if (!savedReviewAudio.getAttribute("src")) updateHistoryReview();
        return;
      }
      stopHistoryPlayback(true);
      if (current === "phrases") {
        const phrases = Array.isArray(view.phrases) ? view.phrases.filter(value => typeof value === "string") : [];
        if (!phrases.length) { unknown(); return; }
        const size = pageCapacity(phrases.length, historyHeight(), 44, innerWidth > 600 ? 2 : 1, 10), pages = Math.ceil(phrases.length / size);
        historyPage = Math.min(historyPage, pages - 1);
        const grid = node("div", undefined, "story-saved-phrases");
        phrases.slice(historyPage * size, (historyPage + 1) * size).forEach((text, i) => {
          const field = node("input"); field.type = "text"; field.readOnly = true; field.value = text;
          field.setAttribute("aria-label", `저장된 유사어 ${historyPage * size + i + 1}`); grid.append(field);
        });
        historyContent.replaceChildren(grid); renderHistoryPages(pages); return;
      }
      const files = Array.isArray(view.progress?.files) ? view.progress.files : Array.isArray(view.files) ? view.files : [];
      let rows = Array.isArray(view.rows) ? view.rows : [];
      if (current === "downloads" && files.length) {
        const grouped = new Map();
        for (const file of files) {
          const key = file.group || file.id; if (!grouped.has(key)) grouped.set(key, []); grouped.get(key).push(file);
        }
        rows = [...grouped.values()].flatMap(group => {
          if (!group.every(file => (file.unit || "bytes") === "bytes")) {
            // Automatically selected corpora report admitted hours (or other
            // real quantities), not a byte archive. Preserve each original
            // quantity and its unknown values instead of calling it one file.
            return group.map(file => ({ label: file.label || file.group_label || "자료", done: file.done, total: file.total, unit: file.unit || "bytes" }));
          }
          const complete = group.filter(file => ["complete", "reused"].includes(file.state)).length;
          const knownBytes = group.length > 0 && group.every(file => Number.isFinite(file.done));
          const bytes = knownBytes ? group.reduce((sum, file) => sum + file.done, 0) : null;
          const totalBytes = group.every(file => Number.isFinite(file.total)) ? group.reduce((sum, file) => sum + file.total, 0) : null;
          return [{ label: group[0].group_label || group[0].label || "자료", done: complete, total: group.length, unit: "files", bytes, totalBytes, files: group }];
        });
      }
      if (!rows.length && view.progress) rows = [view.progress];
      if (!rows.length) { unknown(); return; }
      // Measure real row heights at the rendered width before choosing a
      // page size. A tall narrow viewport still has a single-column list.
      const list = node("div", undefined, "story-history-rows");
      const allItems = rows.map(row => {
        const item = historyRow(row);
        if (Number.isFinite(row.bytes) || Number.isFinite(row.totalBytes)) item.append(node("p", Number.isFinite(row.totalBytes) ? `${quantity(row.bytes, "bytes")} / ${quantity(row.totalBytes, "bytes")}` : quantity(row.bytes, "bytes"), "story-history-note"));
        return item;
      });
      list.append(...allItems); historyContent.replaceChildren(list);
      const columns = current === "downloads" && innerWidth > 600 ? 2 : 1;
      const style = typeof getComputedStyle === "function" ? getComputedStyle(list) : null;
      const gap = style ? parseFloat(style.rowGap) || 0 : 18;
      const rowHeight = Math.max(...allItems.map(item => item.getBoundingClientRect?.().height || 70));
      const size = pageCapacity(rows.length, historyHeight(), rowHeight, columns, gap), pages = Math.ceil(rows.length / size);
      historyPage = Math.min(historyPage, pages - 1);
      allItems.forEach((item, i) => { item.hidden = Math.floor(i / size) !== historyPage; });
      renderHistoryPages(pages);
    }
    function renderHistoryPages(pages) {
      historyPages.hidden = pages <= 1;
      historyBack.disabled = historyPage === 0; historyNext.disabled = historyPage >= pages - 1;
      historyPageText.textContent = `${historyPage + 1} / ${pages}`;
    }
    function render() {
      if (creationPending) return;
      const state = bridge.state(), project = state.project;
      const key = project?.id || "setup";
      if (key !== projectKey) {
        pauseMedia(); stopTransition(); projectKey = key; following = true; materialPage = 0; materialKey = "";
        maximum = frontier(project); current = project ? stageStep(project) : "entry"; lastFrontier = project ? stageStep(project) : "entry";
        historyPage = 0; historyKey = ""; setupReached = 0;
      }
      lastProject = project; maximum = frontier(project);
      bridge.voiceDesign?.(project ? project.reference_design : state.recipes.find(recipe => recipe.id === state.selectedPreset)?.reference_design);
      const actual = project ? stageStep(project) : current;
      if (project && actual !== lastFrontier) {
        if (!following || current === actual) lastFrontier = actual;
        else if (!transitioning && !state.busy && navigate(order.indexOf(actual), true)) lastFrontier = actual;
      }
      if (order.indexOf(current) > maximum) current = actual;
      const checkpoint = downloadCheckpoint(project), navigation = navigationCheckpoint(project);
      const position = order.indexOf(current), historical = Boolean(project && (position < maximum || done(project.status)
        || (current === "downloads" && checkpoint?.state === "awaiting_next" && completedStep(project, "downloads"))
        || (navigation?.state === "awaiting_next" && current === navigation.step && completedNavigationStep(project, current))));
      shell.dataset.step = current; shell.dataset.model = project ? "existing" : "new";
      shell.dataset.history = String(historical);
      shell.dataset.downloadList = "false";
      const preparingResults = Boolean(project && current === "results" && !done(project.status) && stageStep(project) === "results");
      heading.textContent = preparingResults ? "학습 결과를 준비합니다." : definitions[position]?.[2] || "학습 상태";
      document.title = definitions[position]?.[1] || "한국어 호출어 학습";
      context.textContent = project && position >= 3 ? `‘${project.wakeword || ""}’ · ${titles[project.preset] || "호출어 학습"}` : "";
      context.hidden = !context.textContent;
      logButton.hidden = !project;
      if (!project) setupReached = Math.max(setupReached, position);
      const visibleThrough = project ? maximum : setupReached;
      index.hidden = visibleThrough === 0;
      ticks.forEach((tick, i) => {
        tick.hidden = i > visibleThrough; tick.disabled = transitioning || state.busy || i > maximum;
        tick.setAttribute("aria-current", i === position ? "step" : "false");
      });
      if (!transitioning) {
        back.hidden = position === 0; back.disabled = state.busy || position === 0;
        next.hidden = current === "microphone";
        next.disabled = state.busy;
        next.textContent = "다음"; next.title = "";
        if (!project) next.disabled = state.busy || !state.recipeReady || (current === "materials" && $("create").disabled);
        else next.disabled = state.busy || (current === stageStep(project) && humanGate(project)) || (position >= maximum && !releaseStep(project));
      }
      const action = operationAction(project);
      operation.hidden = !action;
      operation.disabled = !action || state.busy || transitioning || project?.status === "pausing";
      operation.textContent = project?.status === "pausing" ? "정지 중" : action === "pause" ? "일시정지" : action === "resume" ? "재개" : "시작";
      operation.title = action === "pause" ? "현재 단계가 끝난 뒤 멈춥니다." : "";
      footer.classList.toggle("has-operation", !operation.hidden);
      downloadActions.hidden = !canStartDownloads(project);
      downloadPrimary.disabled = state.busy || transitioning || !canStartDownloads(project);
      migrationNote.hidden = !project?.migration;
      // Hidden panels stay connected; polling never rebuilds their inputs or
      // embedded documents, and never resets a review's listening position.
      const wanted = current === "entry" ? (project ? savedEntry : entry) : current === "scale" ? (project ? savedScale : scale)
        : current === "materials" ? materials : current === "voice-listening" ? voiceListening : live;
      for (const unit of [entry, savedEntry, scale, savedScale, materials, live, voiceListening]) unit.hidden = unit !== wanted;
      if (project && savedWord.value !== (project.wakeword || "")) savedWord.value = project.wakeword || "";
      if (project) for (const card of savedPresets.children) {
        const radio = card.querySelector("input"); radio.checked = radio.value === project.preset;
        card.setAttribute("data-selected", String(radio.checked));
      }
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
        referencePreview.open = current === "voice-listening" && Boolean(project.reference_preview?.url);
        if (preparingResults) setView("overview");
        else if (current === "voice-listening") setView("review", "reference");
        else if (current === "results" || current === "microphone") setView(current === "microphone" ? "mic" : "results");
        else if (!historical && (project.review || project.input)) setView("review", "required");
        else setView("overview");
        const frameStep = !preparingResults && ["results", "microphone"].includes(current) && Boolean(project.results?.url || project.results_url);
        results.hidden = !frameStep || !(project.results?.url || project.results_url);
        review.hidden = historical || frameStep || !project.review || running(project.status);
        input.hidden = historical || frameStep || !project.input || running(project.status);
        liveSummary.hidden = historical || frameStep;
        currentTitle.hidden = true;
        downloads.hidden = current !== "downloads" || historical || !project.progress?.files?.length;
        shell.dataset.downloadList = String(!downloads.hidden);
        voiceCounts.hidden = current !== "voices";
        historic.hidden = !historical || frameStep || ["voices", "entry", "scale", "materials"].includes(current);
        referenceButton.hidden = !project.reference_preview?.url || current !== "pronunciation";
        hardButton.hidden = !project.hard_preview?.url || !["hard-generation", "hard-review", "phrases"].includes(current);
        listenActions.hidden = referenceButton.hidden && hardButton.hidden;
        if (!historic.hidden) renderHistory(project);
        if (!voiceCounts.hidden) renderVoiceCounts(project);
        voiceListeningStatus.hidden = Boolean(project.reference_preview?.url);
        voiceListeningStatus.textContent = voiceListeningStatus.hidden ? "" : "저장된 목소리 듣기 자료가 없습니다.";
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
    downloads.addEventListener("toggle", event => {
      if (event.target?.tagName === "DETAILS") Promise.resolve().then(paginateDownloads);
    }, true);
    window.addEventListener("resize", () => { if (current === "materials") renderMaterials(); if (current === "downloads") paginateDownloads(); if (lastProject && !historic.hidden) renderHistory(lastProject); });
    window.addEventListener("pointerdown", () => {
      if (!voiceListening.classList.contains("is-selecting")) return;
      voiceListening.classList.remove("is-selecting");
      $("reference-preview-frame")?.contentWindow?.postMessage({ type: "koww:voice-picker-close" }, window.location.origin);
    });
    window.addEventListener("message", event => {
      const frame = $("reference-preview-frame"), origin = window.location?.origin;
      if (!origin || event.origin !== origin || !frame?.contentWindow || event.source !== frame.contentWindow
        || event.data?.type !== "koww:voice-picker" || typeof event.data.open !== "boolean") return;
      voiceListening.classList.toggle("is-selecting", event.data.open && current === "voice-listening" && !transitioning);
    });
    window.addEventListener("pagehide", () => { stopTransition(); pauseMedia(); });
    return { render };
  }
  window.KowwStory = Object.freeze({ create, definitions, stageStep, frontier, completedStep, pageCapacity, downloadCheckpoint, navigationCheckpoint });
})();
