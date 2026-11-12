"use strict";
/* Presentation adapter. Worker state and all human decisions stay authoritative. */
(() => {
  const definitions = [
    ["entry", "호출어 입력", "‘보라야’와 같은 한국어 호출어를 입력해 주세요."],
    ["tools", "학습 도구 준비", "CPU 또는 NVIDIA GPU에서 학습에 사용할 실행 환경을 준비합니다."],
    ["scale", "학습 규모 선택", "사용 목적과 환경에 맞는 학습 규모를 선택하세요."],
    ["materials", "준비 자료", "학습과 평가에 사용할 자료의 종류와 용량을 확인하세요."],
    ["downloads", "자료 다운로드", "학습과 평가에 필요한 자료를 내려받고 검사합니다."],
    ["voices", "기준 목소리 생성", "호출어 음성을 만드는 데 쓸 기준 목소리를 생성합니다."],
    ["voice-listening", "기준 목소리 검수", "목소리가 요청한 성별과 연령대에 맞고 서로 구분되는지 들어보세요."],
    ["call-pilot-generation", "호출어 샘플 생성", "기준 목소리마다 호출어 샘플을 하나씩 만듭니다."],
    ["call-pilot-review", "호출어 발음 사전 검수", "호출어 샘플을 연속으로 듣고, 문제가 있는 음원만 부적합으로 표시하세요.\n재생된 음원은 자동으로 적합 판정하며, 부적합한 음원만 모아 다시 생성합니다."],
    ["positive-generation", "호출어 음성 생성", "학습용과 평가용 기준 목소리로 전체 호출어 음성을 생성합니다."],
    ["pronunciation", "호출어 발음 검수", "호출어가 정상적으로 생성되었는지 검수합니다.\n'제외'를 선택하지 않을 경우 '정상'으로 자동 지정됩니다."],
    ["phrases", "유사어 입력", "호출어와 유사한 문구를 입력하세요."],
    ["hard-generation", "유사어 음성 생성", "입력한 유사어의 전체 음성을 학습용 기준 목소리로 생성합니다."],
    ["hard-review", "유사어 발음 검수", "대표 음원을 연속으로 듣고, 문제가 있는 음원만 부적합으로 표시하세요.\n끝까지 들은 음원은 적합 판정하며, 다음을 누르면 저장하고 적합한 유사어를 학습에 사용합니다."],
    ["features", "증강·특징 추출", "음원에 소음과 울림을 더하고 학습에 쓸 특징을 추출합니다."],
    ["training", "모델 학습", "호출어를 감지하도록 모델을 학습합니다."],
    ["evaluation", "성능 평가", "여러 소음과 울림 조건에서 모델을 평가합니다."],
    ["results", "학습 결과", "모델별 성능과 추천 감지 설정을 비교해 보세요."],
    ["microphone", "마이크 시험", "마이크로 호출어를 말하고 모델의 반응을 확인해 주세요."],
  ];
  const mapping = {
    frontend: "downloads", data: "downloads", data_download: "downloads", data_review: "downloads", data_connect: "downloads",
    references: "voices", references_review: "voices", pilot: "pronunciation", pilot_review: "pronunciation",
    generate: "positive-generation", positives: "positive-generation", positive_review: "pronunciation", hard_phrases: "phrases",
    hard_pilot: "hard-review", hard_generation: "hard-generation", hard_review: "hard-review", hard_connect: "hard-review",
    features: "features", train: "training", training: "training", package: "training",
    evaluate: "evaluation", development: "evaluation", evaluation: "evaluation", test: "results", results: "results",
  };
  const prepostOrder = definitions.map(([id]) => id);
  const order = prepostOrder.filter(id => id !== "call-pilot-generation" && id !== "call-pilot-review");
  const workLabels = {
    voices: "합성 목소리",
    "call-pilot-generation": "호출어 샘플",
    "positive-generation": "호출어 음성",
    "hard-generation": "유사어 음성",
    features: "특징 추출",
    training: "모델 학습",
    evaluation: "성능 평가",
  };
  const generationSteps = new Set(["voices", "call-pilot-generation", "positive-generation", "hard-generation"]);
  const callReviewSteps = new Set(["call-pilot-review", "pronunciation"]);
  const pronunciationSteps = new Set([...callReviewSteps, "hard-review"]);
  const standardLayoutSteps = new Set([
    ...Object.keys(workLabels), ...pronunciationSteps, "phrases", "voice-listening",
  ]);
  const featureLabels = { positive: "호출어 음성", generic_negative: "일반 비호출", korean_negative: "한국어 비호출", hard_negative: "유사어 음성" };
  const done = value => ["done", "complete", "completed", "finished"].includes(value);
  const running = value => ["running", "starting", "queued", "pausing"].includes(value);
  const stageId = project => project?.stage_id || (typeof project?.stage === "object" ? project.stage.id : project?.stage);
  const sampleFirstHardReview = project => project?.settings?.hard_review_policy === "sample_before_generation_v2"
    || Boolean(project?.stages?.some(record => record.id === "hard_pilot"));
  const postgenerationPositiveReview = project => project?.settings?.pilot_review_policy === "one_per_voice_after_generation_v3";
  const twoPhasePositiveReview = project => project?.settings?.pilot_review_policy === "one_per_voice_before_and_after_generation_v4";
  const separateReferencePreparation = project => (project?.settings?.reference_preparation_policy
    || project?.reference_preparation?.policy || project?.reference_preparation_policy) === "initial_candidate_batches_v1";
  const referencePreparationNeeded = project => separateReferencePreparation(project) && project?.reference_preparation?.required === true;
  const referencePreparationReached = project => referencePreparationNeeded(project)
    && !(navigationCheckpoint(project)?.step === "downloads" && navigationCheckpoint(project)?.state === "awaiting_next")
    && (navigationCheckpoint(project)?.step === "voices" || ["references", "references_review"].includes(stageId(project))
      || project.reference_preparation.operation === "helper" && (project.reference_preparation.can_start === true
        || project.reference_preparation.can_resume === true || project.reference_preparation.can_pause === true
        || ["running", "pausing", "paused", "failed", "completed"].includes(project.reference_preparation.status)));
  const mergedReferenceReview = project => !separateReferencePreparation(project) && (project?.reference_preview?.candidate_selection === true
    || project?.settings?.reference_selection_policy === "candidate_pool_v3");
  function displayOrder(project) {
    const prepost = twoPhasePositiveReview(project);
    const screens = [...(prepost ? prepostOrder : order)];
    if (project && !done(project.status)) {
      if (!prepost && !postgenerationPositiveReview(project)) screens.splice(screens.indexOf("positive-generation"), 2, "pronunciation", "positive-generation");
      if (sampleFirstHardReview(project)) screens.splice(screens.indexOf("hard-generation"), 2, "hard-review", "hard-generation");
    }
    return mergedReferenceReview(project) ? screens.filter(id => id !== "voices") : screens;
  }
  function stageScreen(project, id) {
    if (id === "frontend" && project?.status === "waiting_input" && project.input?.kind === "dependency_setup") return "tools";
    if (mergedReferenceReview(project) && ["references", "references_review"].includes(id)) return "voice-listening";
    if (id === "hard_connect" && sampleFirstHardReview(project)) return "hard-generation";
    if (twoPhasePositiveReview(project)) {
      if (id === "pilot") return "call-pilot-generation";
      if (id === "pilot_review") return "call-pilot-review";
    }
    return mapping[id];
  }
  const downloadCheckpoint = project => project?.download_checkpoint?.policy === "explicit_download_v1" ? project.download_checkpoint : null;
  const navigationCheckpoint = project => project?.navigation_checkpoint?.policy === "explicit_story_v1" ? project.navigation_checkpoint : null;
  // Tool preparation and downloads share the saved worker checkpoint.
  const navigationStart = [
    ["downloads", ["frontend", "data_download", "data_review", "data_connect"]],
    ["voices", ["references", "references_review"]],
  ];
  const navigationAfterHard = [
    ["features", ["features"]], ["training", ["training", "package"]],
    ["evaluation", ["development", "evaluation"]], ["results", ["results"]],
  ];
  const stageStep = project => {
    if (done(project?.status)) return "results";
    if (referencePreparationReached(project) && project.reference_preparation.ready !== true) return "voices";
    const checkpoint = navigationCheckpoint(project);
    const step = mergedReferenceReview(project) && checkpoint?.step === "voices" ? "voice-listening" : checkpoint?.step;
    return checkpoint?.state === "awaiting_next" && displayOrder(project).includes(step) ? step
      : downloadCheckpoint(project)?.state === "awaiting_next" ? "downloads" : stageScreen(project, stageId(project)) || "downloads";
  };
  function completedStep(project, step) {
    if (step === "tools") return project?.stages?.some(record => record.id === "frontend" && done(record.status)) === true;
    if (step === "voices" && separateReferencePreparation(project)) return project.reference_preparation?.ready === true;
    if (step === "voice-listening" && !mergedReferenceReview(project)) step = "voices";
    const records = Array.isArray(project?.stages) ? project.stages.filter(record => stageScreen(project, record.id) === step) : [];
    return records.length > 0 && records.every(record => done(record.status));
  }
  function completedNavigationStep(project, step) {
    if (step === "voice-listening") step = "voices";
    if (!navigationCheckpoint(project)) return completedStep(project, step);
    const positive = twoPhasePositiveReview(project)
      ? [["call-pilot-generation", ["pilot"]], ["call-pilot-review", ["pilot_review"]], ["positive-generation", ["positives"]], ["pronunciation", ["positive_review"]]]
      : postgenerationPositiveReview(project)
        ? [["positive-generation", ["positives"]], ["pronunciation", ["pilot_review"]]]
        : [["pronunciation", ["pilot", "pilot_review"]], ["positive-generation", ["positives"]]];
    const hard = sampleFirstHardReview(project)
      ? [["hard-review", ["hard_pilot", "hard_review"]], ["hard-generation", ["hard_generation", "hard_connect"]]]
      : [["hard-generation", ["hard_generation"]], ["hard-review", ["hard_review", "hard_connect"]]];
    const groups = [...navigationStart, ...positive, ["phrases", ["hard_phrases"]], ...hard, ...navigationAfterHard];
    const position = groups.findIndex(([id]) => id === step);
    return position >= 0 && groups.slice(0, position + 1).flatMap(([, stages]) => stages)
      .every(id => project.stages?.some(record => record.id === id && done(record.status)));
  }
  function frontier(project) {
    const screens = displayOrder(project);
    if (!project) return screens.indexOf("materials");
    if (done(project.status)) return screens.indexOf((project.results?.mic_url || project.mic_url) ? "microphone" : "results");
    if (referencePreparationReached(project) && project.reference_preparation.ready !== true) return screens.indexOf("voices");
    // Only actual current/completed backend stages unlock later screens.
    const current = stageStep(project), position = Math.max(screens.indexOf("materials"), screens.indexOf(current));
    const navigation = navigationCheckpoint(project);
    return current === "voices" && completedNavigationStep(project, "voices")
      && (!navigation || (navigation.state === "awaiting_next" && project.status === "paused")) ? screens.indexOf("voice-listening") : position;
  }
  const titles = { check: "동작 확인용", trial: "시험 모델용", production: "상용 모델용" };
  const shortDescriptions = { check: "라이브러리 동작 확인", trial: "호출 성능 비교·시험", production: "많은 자료로 본격 학습" };
  const toolBackends = { cpu: ["CPU", "NVIDIA GPU 없이 실행"], cu128: ["NVIDIA GPU", "CUDA 12.8 사용"] };
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
    if (!document.querySelector || !document.body || !$('wakeword')) return null;
    const sourcePanels = document.querySelector("main.dashboard-shell");
    if (!sourcePanels) return null;
    // Controls from these panels are mounted into the chapter view below.
    sourcePanels.classList.add("story-source");
    document.body.classList.add("story-mode");
    const shell = node("main", undefined, "story-shell"); shell.id = "story-shell";
    const atmosphere = node("div", undefined, "story-atmosphere"); atmosphere.setAttribute("aria-hidden", "true");
    atmosphere.append(node("div", undefined, "story-background-state story-background-previous"), node("div", undefined, "story-background-state story-background-current"));
    const toolbar = node("header", undefined, "story-toolbar");
    const settingsContainer = node("div", undefined, "story-settings");
    const settingsMenu = node("div", undefined, "story-settings-menu");
    settingsMenu.id = "story-settings-menu"; settingsMenu.hidden = true;
    settingsMenu.setAttribute("role", "group"); settingsMenu.setAttribute("aria-label", "설정 메뉴");
    const settingsButton = button("설정", "story-tool story-settings-trigger", () => setSettingsOpen(settingsMenu.hidden));
    settingsButton.id = "story-settings-button";
    settingsButton.setAttribute("aria-expanded", "false"); settingsButton.setAttribute("aria-controls", settingsMenu.id);
    const projectSwitch = $("project-switch"), projectSelect = $("project-select"), newProjectButton = $("new-project");
    const projectLabel = node("label", "작업한 호출어"); projectLabel.setAttribute("for", projectSelect.id);
    projectSwitch.replaceChildren(projectLabel, projectSelect);
    newProjectButton.textContent = "새 호출어 생성";
    const detailButton = button("세부 설정", "story-settings-action", () => { setSettingsOpen(false); openDialog(settingsDialog); });
    detailButton.id = "story-settings-details";
    const logButton = button("작업 기록", "story-settings-action", () => { setSettingsOpen(false); $("log-details").open = true; openDialog(logDialog); bridge.logs(); });
    logButton.id = "story-settings-log";
    settingsMenu.append(newProjectButton, projectSwitch, detailButton, logButton);
    settingsContainer.append(settingsButton, settingsMenu); toolbar.append(settingsContainer);
    function setSettingsOpen(open, restoreFocus = false) {
      settingsMenu.hidden = !open;
      settingsButton.setAttribute("aria-expanded", String(open));
      if (restoreFocus) settingsButton.focus();
    }
    newProjectButton.addEventListener("click", () => setSettingsOpen(false));
    projectSelect.addEventListener("change", () => setSettingsOpen(false, true));
    settingsContainer.addEventListener("keydown", event => {
      if (event.key === "Escape" && !settingsMenu.hidden) {
        event.preventDefault(); setSettingsOpen(false, true);
      } else if (event.key === "ArrowDown" && event.target === settingsButton) {
        event.preventDefault(); setSettingsOpen(true);
        (newProjectButton.hidden || newProjectButton.disabled ? detailButton : newProjectButton).focus();
      }
    });
    settingsContainer.addEventListener("focusout", event => {
      if (event.relatedTarget && !settingsContainer.contains(event.relatedTarget)) setSettingsOpen(false);
    });
    document.addEventListener("pointerdown", event => {
      if (!settingsMenu.hidden && !settingsContainer.contains(event.target)) setSettingsOpen(false);
    });
    window.addEventListener("blur", () => setSettingsOpen(false));
    const notices = node("div", undefined, "story-notices"); notices.append($("connection"), $("message"));
    const index = node("nav", undefined, "story-index"); index.setAttribute("aria-label", "진행 단계");
    const stage = node("section", undefined, "story-stage");
    const panel = node("div", undefined, "story-panel"); panel.id = "story-panel";
    const header = node("header", undefined, "story-panel-heading");
    const heading = node("h1"); heading.id = "story-title"; heading.tabIndex = -1;
    const subtitle = node("p", undefined, "story-subtitle"); subtitle.id = "story-subtitle";
    const hardGenerationSummary = node("span", undefined, "story-hard-generation-summary"); hardGenerationSummary.hidden = true;
    heading.setAttribute("aria-describedby", subtitle.id);
    header.append(heading, subtitle);
    panel.setAttribute("aria-labelledby", heading.id);
    const content = node("div", undefined, "story-content");
    const entry = node("div", undefined, "story-entry");
    const wakeword = $("wakeword"), wakewordFeedback = node("p", undefined, "story-field-error");
    wakewordFeedback.id = "story-wakeword-error"; wakewordFeedback.hidden = true;
    wakewordFeedback.setAttribute("aria-live", "polite");
    wakeword.setAttribute("aria-describedby", wakewordFeedback.id);
    entry.append(wakeword, wakewordFeedback);
    // A saved model is a read-only view; leave the new-model draft untouched.
    const savedEntry = node("div", undefined, "story-entry");
    const savedWord = node("input"); savedWord.id = "story-saved-wakeword";
    savedWord.type = "text"; savedWord.readOnly = true;
    savedWord.setAttribute("aria-label", "호출어"); savedEntry.append(savedWord);
    const toolPicker = node("fieldset", undefined, "story-tool-options");
    toolPicker.id = "story-tool-picker"; toolPicker.append(node("legend", "연산 장치", "sr-only"));
    const toolInputs = new Map();
    for (const [backend, [label, description]] of Object.entries(toolBackends)) {
      const card = node("label", undefined, "dependency-option"), radio = node("input");
      radio.id = `story-tool-${backend}`; radio.type = "radio"; radio.name = "setup-backend"; radio.value = backend;
      radio.addEventListener("change", () => { if (radio.checked) setupBackend = backend; });
      card.append(radio, node("strong", label), node("span", description));
      toolInputs.set(backend, radio); toolPicker.append(card);
    }
    const scale = node("div", undefined, "story-scale"); scale.append($("presets"));
    const savedScale = node("div", undefined, "story-scale story-saved-scale");
    const savedPresets = node("div", undefined, "preset-grid"); savedPresets.id = "story-saved-presets";
    savedPresets.setAttribute("role", "group"); savedPresets.setAttribute("aria-label", "저장된 학습 규모"); savedScale.append(savedPresets);
    for (const [id, title] of Object.entries(titles)) {
      const card = node("label", undefined, "preset"), radio = node("input");
      radio.type = "radio"; radio.name = "saved-preset"; radio.value = id; radio.disabled = true;
      const line = node("div"); line.append(radio, node("span", title, "preset-title"));
      card.append(line, node("p", shortDescriptions[id], "preset-description"));
      savedPresets.append(card);
    }
    const materials = node("div", undefined, "story-materials");
    const materialsTabs = node("div", undefined, "story-material-tabs"); materialsTabs.setAttribute("role", "tablist"); materialsTabs.setAttribute("aria-label", "자료 구분");
    const materialColumns = node("div", undefined, "story-material-columns"); materialColumns.tabIndex = 0; materialColumns.setAttribute("aria-label", "준비 자료 목록");
    const toolMaterials = node("section", undefined, "story-tool-materials"); toolMaterials.id = "story-tool-materials"; toolMaterials.hidden = true;
    toolMaterials.setAttribute("aria-labelledby", "story-tool-materials-title");
    const locationStatus = node("div", undefined, "story-location-status"); locationStatus.hidden = true;
    const locationFeedback = node("p"); locationFeedback.setAttribute("role", "status");
    const locationRetry = button("다시 확인", "story-secondary", () => bridge.retryLocation()); locationRetry.hidden = true;
    locationStatus.append(locationFeedback, locationRetry);
    materials.append(materialsTabs, materialColumns, toolMaterials, locationStatus);
    const live = node("div", undefined, "story-live");
    const statusCard = document.querySelector(".current-card");
    const liveSummary = node("div", undefined, "story-live-summary");
    const downloadActivity = node("p", undefined, "story-history-note");
    downloadActivity.id = "story-download-activity";
    const downloadActivityLabel = node("span"), downloadActivityIndicator = node("i", undefined, "story-progress-indicator");
    downloadActivity.hidden = true;
    Object.assign(downloadActivity.style, { minHeight: "22px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" });
    downloadActivity.append(downloadActivityLabel, downloadActivityIndicator);
    const currentTitle = $("current-title");
    const migrationNote = $("migration-note");
    liveSummary.append(statusCard, $("counts"));
    const downloads = $("downloads-section"), review = $("review-section"), input = $("input-section"), results = $("results-section");
    $("hard-phrase-fields").classList.add("story-phrase-grid");
    let phraseInputPending = false;
    const voiceCounts = node("div", undefined, "story-voice-counts");
    let workRowsKey = "", workRowViews = [];
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
    const savedReviewSidebar = node("aside", undefined, "listen-sidebar"); savedReviewSidebar.setAttribute("aria-label", "저장된 음원 목록");
    const savedReviewList = node("div", undefined, "listen-list");
    const savedReviewListPager = node("div", undefined, "listen-list-pager");
    const savedReviewListPrevious = button("‹", undefined, () => { historyReviewPage--; renderHistoryReviewList(true); }); savedReviewListPrevious.setAttribute("aria-label", "앞 음원 목록");
    const savedReviewListPosition = node("span");
    const savedReviewListNext = button("›", undefined, () => { historyReviewPage++; renderHistoryReviewList(true); }); savedReviewListNext.setAttribute("aria-label", "뒤 음원 목록");
    savedReviewListPager.append(savedReviewListPrevious, savedReviewListPosition, savedReviewListNext);
    savedReviewSidebar.append(savedReviewList, savedReviewListPager);
    const savedReviewPlayer = node("div", undefined, "listen-player");
    const savedReviewSelect = node("select", undefined, "listen-select"); savedReviewSelect.setAttribute("aria-label", "저장된 음원");
    const savedReviewHeading = node("div", undefined, "listen-heading");
    const savedReviewText = node("strong", undefined, "listen-title"), savedReviewPosition = node("span", undefined, "listen-position"); savedReviewHeading.append(savedReviewText, savedReviewPosition);
    const savedReviewPickerContainer = node("div", undefined, "voice-picker");
    const savedReviewPickerButton = node("button", undefined, "voice-picker-button");
    savedReviewPickerButton.type = "button";
    savedReviewPickerButton.setAttribute("aria-label", "저장된 음원");
    savedReviewPickerButton.setAttribute("aria-haspopup", "listbox");
    savedReviewPickerButton.setAttribute("aria-controls", "saved-review-voice-list");
    const savedReviewPickerLabel = node("span", undefined, "voice-picker-label");
    const savedReviewPickerPopup = node("div", undefined, "voice-picker-popup");
    const savedReviewPickerList = node("div", undefined, "voice-picker-list");
    savedReviewPickerList.id = "saved-review-voice-list";
    savedReviewPickerList.setAttribute("aria-label", "저장된 음원 목록");
    savedReviewPickerList.setAttribute("role", "listbox");
    savedReviewPickerList.tabIndex = -1;
    savedReviewPickerButton.append(savedReviewPickerLabel);
    savedReviewPickerPopup.append(savedReviewPickerList);
    savedReviewPickerContainer.append(savedReviewPickerButton, savedReviewPickerPopup);
    savedReviewPickerContainer.hidden = true;
    savedReviewHeading.prepend(savedReviewPickerContainer);
    const savedReviewAudio = node("audio", undefined, "listen-audio"); savedReviewAudio.controls = true; savedReviewAudio.preload = "none";
    const savedReviewControls = node("div", undefined, "listen-controls");
    const savedReviewPrevious = button("이전", undefined, () => selectHistoryReview(historyReviewIndex - 1)); savedReviewPrevious.setAttribute("aria-label", "이전 음원");
    const savedReviewToggle = button("재생", undefined, () => historyPlaybackWanted || savedReviewAudio.paused === false ? stopHistoryPlayback() : playHistoryReview());
    const savedReviewNext = button("다음", undefined, () => selectHistoryReview(historyReviewIndex + 1)); savedReviewNext.setAttribute("aria-label", "다음 음원");
    savedReviewControls.append(savedReviewPrevious, savedReviewToggle, savedReviewNext);
    const savedReviewVolume = node("input");
    savedReviewVolume.type = "range";
    savedReviewVolume.min = "0"; savedReviewVolume.max = "100"; savedReviewVolume.step = "1";
    savedReviewVolume.hidden = true;
    const savedReviewDecisions = node("div", undefined, "story-saved-review-decisions listen-decisions"); savedReviewDecisions.setAttribute("role", "group"); savedReviewDecisions.setAttribute("aria-label", "저장된 검수 결과");
    const savedReviewLabels = { keep: "정상", discard: "제외", review: "다시 확인" };
    const savedReviewInputs = [];
    for (const [value, label] of [["keep", "정상"], ["discard", "제외"]]) {
      const choice = node("label"), radio = node("input"); radio.type = "radio"; radio.name = "saved-review-decision"; radio.value = value; radio.disabled = true;
      choice.append(radio, node("span", label)); savedReviewDecisions.append(choice); savedReviewInputs.push(radio);
    }
    const savedReviewNote = node("p", undefined, "story-history-note listen-status");
    savedReviewSelect.hidden = true;
    savedReviewPlayer.append(savedReviewHeading, savedReviewAudio, savedReviewControls, savedReviewSelect, savedReviewDecisions, savedReviewNote);
    savedReviewWorkspace.append(savedReviewSidebar, savedReviewPlayer);
    savedReview.append(savedReviewCompletion, savedReviewWorkspace);
    historic.append(historyContent, savedReview, historyPages);
    const evaluationPlan = node("div", undefined, "story-evaluation-plan");
    const evaluationPlanTitle = node("h2");
    const evaluationPlanDetail = node("p");
    evaluationPlan.append(evaluationPlanTitle, evaluationPlanDetail);
    evaluationPlan.hidden = true;
    const listenActions = node("div", undefined, "story-listen-actions");
    const hardButton = button("유사어 전체 듣기", "story-secondary", openHardListening);
    const positiveButton = button("호출어 전체 듣기", "story-secondary", openPositiveListening);
    listenActions.append(hardButton, positiveButton);
    // A completed stage hides live progress, but its compatibility update must
    // remain reachable through the existing explicit migration control.
    live.append(migrationNote, voiceCounts, liveSummary, downloadActivity, downloads, review, input, results, historic, evaluationPlan, listenActions);
    // Core forms and event bindings retain their IDs when moved into chapters.
    // Inactive controls stay connected so their drafts survive navigation.
    content.append(entry, savedEntry, toolPicker, scale, savedScale, materials, live, voiceListening);
    const footer = node("div", undefined, "story-actions");
    const back = button("이전", "story-button", () => navigate(screens.indexOf(current) - 1));
    const operation = button("일시정지", "story-button story-operation", controlJob); operation.hidden = true;
    const next = button("다음", "story-button", advance);
    footer.append(back, operation, next); panel.append(header, content, footer);
    const loader = node("div", undefined, "story-loader"); loader.hidden = true; loader.setAttribute("role", "status");
    const loaderDots = node("span", undefined, "story-loading-dots"); loaderDots.setAttribute("aria-hidden", "true");
    for (let i = 0; i < 3; i++) loaderDots.append(node("i"));
    const loaderText = node("span"); loader.append(loaderDots, loaderText);
    stage.append(panel, loader); shell.append(atmosphere, toolbar, notices, index, stage);
    document.body.append(shell);
    const savedReviewTransport = window.KowwListening?.enhance(savedReviewAudio, savedReviewControls, {
      play: savedReviewToggle, previous: savedReviewPrevious, next: savedReviewNext,
      volume: savedReviewVolume,
    });
    savedReviewVolume.hidden = true;
    const savedReviewPicker = window.KowwListening?.createPicker?.({
      container: savedReviewPickerContainer, button: savedReviewPickerButton, label: savedReviewPickerLabel,
      popup: savedReviewPickerPopup, list: savedReviewPickerList, onSelect: position => selectHistoryReview(position),
    });
    const settingsDialog = makeDialog("세부 설정");
    const settingSummary = node("div", undefined, "story-setting-summary");
    const location = $("storage-directory").parentElement;
    settingSummary.append(location, $("system-summary"));
    settingsDialog.content.append(settingSummary, $("advanced"), $("setup-voices"), $("storage-budget-details"));
    const timingDetails = statusCard.querySelector(".task-details");
    if (timingDetails) settingsDialog.content.append(timingDetails);
    const logDialog = makeDialog("작업 기록"); logDialog.content.append($("log-details"));
    const listenDialog = makeDialog("유사어 전체 듣기");
    listenDialog.dialog.classList.add("story-listen-dialog");
    listenDialog.dialog.setAttribute("aria-describedby", "hard-preview-counts");
    listenDialog.content.classList.add("story-listen-body");
    const positiveFrame = node("iframe", undefined, "story-positive-preview-frame");
    positiveFrame.title = "호출어 전체 듣기"; positiveFrame.hidden = true;
    listenDialog.content.append($("hard-preview"), positiveFrame);
    listenDialog.dialog.addEventListener("close", () => {
      listening = null; bridge.pause($("reference-preview-frame")); bridge.pause($("hard-preview-frame"));
      $("hard-preview").open = false;
      positiveFrame.removeAttribute("src"); positiveFrame.hidden = true;
      render();
    });
    let current = "entry", projectKey = "", maximum = frontier(null), setupReached = 0, following = true, transitioning = false, transitionSession = null, creationPending = false;
    let setupBackend = null;
    let hardConnectionIntent = null;
    const connectingHard = project => current === "hard-review" && stageId(project) === "hard_connect"
      && running(project?.status) && !project.review && !project.input;
    let materialTab = "training", materialKey = "", materialData = null, listening = null;
    let historyPage = 0, historyKey = "", historyReviewItems = [], historyReviewIndex = 0;
    let historyReviewPage = 0, historyReviewListKey = "", historyReviewQueueKey = "", historyReviewRows = [];
    let historyPlaybackWanted = false, historyPlaybackVersion = 0;
    let lastFrontier = "", lastProject = null, screens = order;
    let referenceReviewComplete = false;
    let preparationErrorShown = false;
    let referenceGroup = null, referenceApprovalPending = false, referenceMode = null;
    let referenceApprovalContext = null, referenceApprovalRequest = 0, boundaryNextIntent = null;
    let referenceFrameElement = null, referenceFrameWindow = null, referenceFrameUrl = "";
    let preflightState = null, preflightFinalizePending = false, preflightCommitted = false, preflightFrameKey = "", preflightFrameWindow = null;
    let preflightRequest = 0;
    let preflightResumeIntent = null, preflightPickerRequest = null, preflightPickerLastRequest = 0;
    let preflightHeightReceived = false, preflightStateReceived = false;
    let preflightFramePhase = null;
    let hardReviewArrivalPending = false, hardReviewRevealFailed = false;
    const reviewWirePhase = project => (callReviewRoute(project)?.phase || preflightFramePhase) === "hard" ? { phase: "hard" } : {};
    let repairGenerationSeen = "", repairReviewStaged = "", repairReviewConcealed = "", repairRevealFailed = "";
    function postRepair(project) {
      const value = project?.postgeneration_repair;
      return twoPhasePositiveReview(project) && stageId(project) === "positive_review" && value && typeof value === "object" ? value : null;
    }
    function repairQueueKey(project) { return project?.review?.queue_sha256 ? `${project.id}|${project.review.queue_sha256}` : ""; }
    function repairProgressVisible(project) {
      return current === "pronunciation" && Boolean(postRepair(project)) && (!project.review || repairReviewConcealed === repairQueueKey(project));
    }
    function revealRepairReview(project, reload = false) {
      const queue = repairQueueKey(project); if (!queue || transitioning) return;
      repairReviewStaged = queue; repairReviewConcealed = queue; repairRevealFailed = "";
      if (reload) { const frame = $("review-frame"); frame.setAttribute("src", frame.getAttribute("src")); }
      const session = beginTransition(false); session.repairQueue = queue;
      void runTransition(session, Promise.resolve(true), () => { repairReviewConcealed = ""; });
    }
    function revealHardReview(reload = false) {
      const session = beginTransition(false);
      session.hardReviewArrival = true;
      hardReviewRevealFailed = false;
      if (reload) {
        const frame = $("review-frame"), url = frame.getAttribute("src");
        if (url) {
          chapterFrameStates.delete("review-frame");
          frame.setAttribute("src", url);
        }
      }
      void runTransition(session);
    }
    function clearPreflightPicker() {
      preflightPickerRequest = null; shell.classList.remove("preflight-picker-open"); shell.style.setProperty("--preflight-picker-height", "");
    }
    function finishBoundaryNext(intent, target) {
      if (boundaryNextIntent !== intent) return;
      boundaryNextIntent = null; intent.target = target; intent.resolve(true);
    }
    function cancelBoundaryNext(message) {
      const intent = boundaryNextIntent; boundaryNextIntent = null;
      if (intent) intent.reject(Error(message || "저장 후 다음 단계 상태를 확인하지 못했습니다. 다시 진행하세요."));
    }
    function beginBoundaryNext(project, route) {
      const session = beginTransition(false), intent = { project: project.id, route, session, saved: false, issued: false, inFlight: false };
      const preparation = new Promise((resolve, reject) => { intent.resolve = resolve; intent.reject = reject; });
      boundaryNextIntent = intent; following = true;
      const watchdog = setTimeout(() => { if (boundaryNextIntent === intent) cancelBoundaryNext(); }, 90000);
      session.timers.add(watchdog);
      void runTransition(session, preparation, () => { current = intent.target; historyPage = 0; historyKey = ""; });
      return intent;
    }
    function resumeSavedBoundary() {
      const intent = boundaryNextIntent; if (!intent) return;
      const state = bridge.state(), project = state.project, checkpoint = navigationCheckpoint(project), stage = stageId(project);
      if (!activeTransition(intent.session) || project?.id !== intent.project || document.hidden) { cancelBoundaryNext(); return; }
      if (project.error || ["failed", "error", "interrupted", "stopped", "cancelled"].includes(project.status) || project.migration && !project.process_alive) {
        cancelBoundaryNext("작업 상태를 확인한 뒤 다시 진행하세요."); return;
      }
      if (!intent.saved || state.busy || intent.inFlight) return;
      if (checkpoint?.step === intent.route.step && checkpoint.state === "awaiting_next") {
        if (![intent.route.sourceStage, ...intent.route.destinationStages].includes(stage)) {
          cancelBoundaryNext("저장한 단계와 현재 대기 단계가 다릅니다. 상태를 확인하세요."); return;
        }
        const referenceReady = intent.route.step !== "voices" || project.reference_preview?.review_complete || referenceReviewComplete;
        if (!intent.issued && referenceReady && project.status === "paused" && !project.process_alive && !humanGate(project)
          && completedNavigationStep(project, intent.route.step)) {
          intent.issued = true; intent.inFlight = true;
          void Promise.resolve().then(async () => {
            if (boundaryNextIntent !== intent || !activeTransition(intent.session)) return;
            const latest = bridge.state(), candidate = latest.project, boundary = navigationCheckpoint(candidate);
            if (document.hidden || latest.busy || candidate?.id !== intent.project || candidate.status !== "paused" || candidate.process_alive
              || humanGate(candidate) || boundary?.step !== intent.route.step || boundary.state !== "awaiting_next"
              || ![intent.route.sourceStage, ...intent.route.destinationStages].includes(stageId(candidate))
              || !completedNavigationStep(candidate, intent.route.step)) { intent.issued = false; return; }
            if (await bridge.action("resume") === false && boundaryNextIntent === intent) cancelBoundaryNext("다음 단계가 시작되지 않았습니다. 다시 진행하세요.");
          }).catch(error => { if (boundaryNextIntent === intent) cancelBoundaryNext(error?.message); })
            .finally(() => { intent.inFlight = false; if (boundaryNextIntent === intent) { resumeSavedBoundary(); render(); } });
        }
        return;
      }
      if (intent.route.destinationStages.includes(stage) && checkpoint?.step !== intent.route.step) {
        finishBoundaryNext(intent, intent.route.target); return;
      }
      // Old projects can continue directly after input without explicit story
      // checkpoints. Never invent a resume boundary for those saved workflows.
      if (!checkpoint) { finishBoundaryNext(intent, current); return; }
      if (stage !== intent.route.sourceStage || project.input && JSON.stringify(project.input) !== intent.route.inputKey) {
        cancelBoundaryNext("저장 중 작업 단계나 입력 요청이 바뀌었습니다. 현재 화면을 확인하세요.");
      }
    }
    function expandPreflightPicker(request) {
      const project = bridge.state().project, frame = $("review-frame");
      if (!reviewingPreflight(project) || transitioning) { clearPreflightPicker(); return; }
      if (request < preflightPickerLastRequest) return;
      const rect = $("review-section")?.getBoundingClientRect?.();
      const baseline = parseFloat(shell.style.getPropertyValue?.("--preflight-content-height") || shell.style["--preflight-content-height"]) || rect?.height || 200;
      const center = Number.isFinite(rect?.top) && Number.isFinite(rect?.height) ? rect.top + rect.height / 2 : NaN;
      const footerRect = footer.getBoundingClientRect?.();
      const bottomEdge = footerRect?.height > 0 && Number.isFinite(footerRect.top) && footerRect.top > center
        ? Math.min(innerHeight - 8, footerRect.top - 8) : innerHeight - 8;
      const available = Number.isFinite(center) && center > 8 && center < innerHeight - 8
        ? 2 * Math.min(center - 8, bottomEdge - center) : innerHeight - 48;
      const height = Math.max(baseline, Math.min(720, Math.floor(available)));
      preflightPickerLastRequest = request; preflightPickerRequest = request; shell.style.setProperty("--preflight-picker-height", `${height}px`); shell.classList.add("preflight-picker-open");
      frame.getBoundingClientRect?.();
      frame.contentWindow?.postMessage?.({ type: "koww:preflight-picker-ready", ...reviewWirePhase(project), project: project.id, queue_sha256: project.review.queue_sha256,
        request: preflightRequest, picker_request: request }, window.location.origin);
    }
    function collapsePreflightPicker(request, queue) {
      if (request < preflightPickerLastRequest || preflightPickerRequest !== null && request !== preflightPickerRequest) return;
      const project = bridge.state().project, frame = $("review-frame");
      preflightPickerLastRequest = request; clearPreflightPicker();
      // Apply the parent's smaller iframe before releasing the child's
      // centering surface. Both documents keep the same global anchor.
      frame.getBoundingClientRect?.();
      frame.contentWindow?.postMessage?.({ type: "koww:preflight-picker-closed", ...reviewWirePhase(project), project: project.id,
        queue_sha256: queue, request: preflightRequest, picker_request: request }, window.location.origin);
    }
    function cancelPreflightNext(message) {
      const intent = preflightResumeIntent; preflightResumeIntent = null;
      if (intent) intent.reject(Error(message || "다음 단계로 이동하지 못했습니다. 상태를 확인한 뒤 다시 진행하세요."));
    }
    function finishPreflightNext(intent, target) {
      if (preflightResumeIntent !== intent) return;
      preflightResumeIntent = null; intent.target = target; intent.resolve(true);
    }
    function issuePreflightResume(intent, phase) {
      if (intent.inFlight || intent[phase]) return;
      intent[phase] = true; intent.inFlight = true;
      void Promise.resolve().then(async () => {
        if (preflightResumeIntent !== intent || !activeTransition(intent.session)) return;
        try {
          const state = bridge.state(), project = state.project, checkpoint = navigationCheckpoint(project);
          if (project?.id !== intent.project || document.hidden) { cancelPreflightNext(); return; }
          if (state.busy || project.status !== "paused" || project.review || project.input || project.process_alive
            || (phase === "processingStarted" ? stageId(project) !== intent.route.stage
              : checkpoint?.step !== intent.route.step || checkpoint.state !== "awaiting_next")) { intent[phase] = false; return; }
          if (await bridge.action("resume") === false && preflightResumeIntent === intent) cancelPreflightNext("다음 단계가 시작되지 않았습니다. 상태를 확인한 뒤 다시 진행하세요.");
        }
        catch (error) { if (preflightResumeIntent === intent) cancelPreflightNext(error?.message); }
        finally {
          intent.inFlight = false;
          if (preflightResumeIntent === intent) {
            resumeFinalizedPreflight();
            render();
          }
        }
      });
    }
    function resumeFinalizedPreflight() {
      const intent = preflightResumeIntent; if (!intent) return;
      const state = bridge.state(), project = state.project, checkpoint = navigationCheckpoint(project), stage = stageId(project);
      if (!activeTransition(intent.session) || project?.id !== intent.project || document.hidden) { cancelPreflightNext(); return; }
      const route = intent.route;
      if (route.phase === "hard" && intent.phase === "processing" && connectingHard(project)) {
        hardConnectionIntent = { project: project.id, target: route.target };
        finishPreflightNext(intent, route.step); return;
      }
      if (intent.phase === "repair") {
        const repair = postRepair(project);
        if (route.phase !== "post" || stage !== route.stage || project.migration && !project.process_alive) {
          cancelPreflightNext("다시 생성하는 작업의 상태가 바뀌었습니다. 현재 상태를 확인해 주세요."); return;
        }
        if (!repair || repair.source_queue_sha256 !== intent.queue) return;
        if (!project.review || project.review.queue_sha256 !== intent.queue) { finishPreflightNext(intent, route.step); return; }
        return;
      }
      const completedHardDestination = route.phase === "hard" && checkpoint?.step === route.target && checkpoint.state === "awaiting_next"
        && completedNavigationStep(project, route.target);
      const reachedDestination = intent.phase === "processing" && (route.destinationStages.includes(stage) || completedHardDestination)
        && !(checkpoint?.step === route.step && checkpoint.state === "awaiting_next")
        && (route.phase !== "hard" || !checkpoint || checkpoint.step === route.target);
      const destinationInput = route.phase === "post" && reachedDestination && stage === "hard_phrases"
        && project.status === "waiting_input" && project.input?.kind === "hard_phrases";
      if (["failed", "error", "interrupted", "stopped", "cancelled"].includes(project.status) || project.error || project.input && !destinationInput || project.migration && !project.process_alive) {
        cancelPreflightNext("작업이 중지됐습니다. 상태를 확인한 뒤 다시 진행하세요."); return;
      }
      if (project.review && project.review.queue_sha256 !== intent.queue && !reachedDestination) {
        if (route.phase === "pre" && intent.phase === "retry" && stage === route.stage) { finishPreflightNext(intent, stageStep(project)); return; }
        cancelPreflightNext("검수 목록이 바뀌었습니다. 현재 상태를 확인한 뒤 다시 진행하세요."); return;
      }
      if (intent.phase === "finalizing" || state.busy || intent.inFlight) return;
      if (intent.phase === "retry") {
        if (route.phase !== "pre") { cancelPreflightNext("검수 단계가 바뀌었습니다. 현재 상태를 확인하세요."); return; }
        if (stage === "pilot" || stage === "pilot_review" && !project.review) finishPreflightNext(intent, "call-pilot-generation");
        else if (stage !== "pilot_review") cancelPreflightNext("재생성 단계가 바뀌었습니다. 현재 상태를 확인하세요.");
        return;
      }
      if (checkpoint?.step === route.step && checkpoint.state === "awaiting_next") {
        if (![route.stage, ...(route.transitStages || []), ...route.destinationStages].includes(stage)) { cancelPreflightNext("작업 단계가 바뀌었습니다. 현재 상태를 확인한 뒤 다시 진행하세요."); return; }
        if (project.status === "paused" && !project.review && !project.process_alive && completedNavigationStep(project, route.step))
          issuePreflightResume(intent, "boundaryReleased");
        return;
      }
      if (reachedDestination) {
        if (route.phase === "post" && !destinationInput) return;
        finishPreflightNext(intent, route.target); return;
      }
      if (route.transitStages?.includes(stage)) {
        if (project.review || project.status === "paused" && !project.process_alive) cancelPreflightNext("자료 연결 작업의 상태를 확인한 뒤 다시 진행하세요.");
        return;
      }
      if (stage !== route.stage) { cancelPreflightNext("검수 단계가 바뀌었습니다. 현재 상태를 확인하세요."); return; }
      if (project.status === "paused" && !project.review && !project.process_alive) issuePreflightResume(intent, "processingStarted");
    }
    function retainReviewFrame(project) {
      const intent = preflightResumeIntent, frame = $("review-frame");
      return Boolean(intent?.phase === "finalizing" && activeTransition(intent.session) && project?.id === intent.project
        && !project.error && !["failed", "error", "interrupted", "stopped", "cancelled"].includes(project.status)
        && (!project.review || project.review.queue_sha256 === intent.queue)
        && intent.route.retainStages.includes(stageId(project))
        && frame === intent.frame && frame?.contentWindow === intent.source && (frame.getAttribute("src") || "") === intent.url
        && preflightRequest === intent.request);
    }
    function syncPreflightFrame() {
      const project = bridge.state().project, frame = $("review-frame");
      const queue = retainReviewFrame(project) ? preflightResumeIntent.queue : project?.review?.queue_sha256;
      const key = JSON.stringify([project?.id, queue, frame?.getAttribute("src") || ""]);
      if (key !== preflightFrameKey || preflightFrameWindow !== frame?.contentWindow) {
        clearPreflightPicker();
        if (preflightResumeIntent && (preflightResumeIntent.phase === "finalizing"
          || project?.review?.queue_sha256 === preflightResumeIntent.queue && stageId(project) === preflightResumeIntent.route.stage))
          cancelPreflightNext("검수 화면이 바뀌었습니다. 다시 진행하세요.");
        preflightFrameKey = key; preflightState = null; preflightFinalizePending = false; preflightCommitted = false;
        preflightFramePhase = callReviewRoute(project)?.phase || (project?.review?.queue_id === "hard" ? "hard" : null);
        preflightHeightReceived = false; preflightStateReceived = false; preflightPickerLastRequest = 0;
        preflightRequest++;
        preflightFrameWindow = frame?.contentWindow || null;
        shell.classList.remove("preflight-draft-review"); shell.style.setProperty("--preflight-content-height", "");
      }
      return frame;
    }
    $("review-frame").addEventListener("load", () => {
      clearPreflightPicker();
      const loadingProject = bridge.state().project;
      if (preflightResumeIntent && (preflightResumeIntent.phase === "finalizing"
        || loadingProject?.review?.queue_sha256 === preflightResumeIntent.queue && stageId(loadingProject) === preflightResumeIntent.route.stage))
        cancelPreflightNext("검수 화면을 다시 불러왔습니다. 다시 진행하세요.");
      syncPreflightFrame(); preflightState = null; preflightFinalizePending = false; preflightRequest++;
      preflightHeightReceived = false; preflightStateReceived = false; preflightPickerLastRequest = 0;
      const project = bridge.state().project;
      if (reviewingPreflight(project)) $("review-frame").contentWindow?.postMessage?.(
        { type: "koww:preflight-status", ...reviewWirePhase(project), project: project.id, queue_sha256: project.review.queue_sha256, request: preflightRequest }, window.location.origin);
    });
    let chapterRequestSequence = 0;
    const chapterFrames = ["reference-preview-frame", "review-frame", "result-frame"];
    const chapterFrameStates = new Map();
    const callReviewLayoutWaiters = new Set();
    function checkCallReviewLayoutWaiters() { for (const check of [...callReviewLayoutWaiters]) check(); }
    function callReviewLayoutReady() {
      const project = bridge.state().project;
      return Boolean(reviewingPreflight(project) && preflightHeightReceived && preflightStateReceived
        && preflightState?.queue_sha256 === project.review.queue_sha256);
    }
    function preparingReviewQueue(project) {
      const stage = current === "call-pilot-review" && twoPhasePositiveReview(project) ? "pilot_review"
        : current === "hard-review" && sampleFirstHardReview(project) ? "hard_review" : "";
      return Boolean(stage && current === stageStep(project) && stageId(project) === stage && !project.error && !project.input
        && (running(project.status) || project.status === "waiting_review" && !reviewingPreflight(project)));
    }
    async function waitForReviewState(session, stillCurrent, isReady) {
      let check;
      const ready = new Promise((resolve, reject) => {
        check = () => {
          if (!activeTransition(session)) { resolve(false); return; }
          const project = bridge.state().project;
          if (project?.id !== session.project || !stillCurrent(project)) {
            reject(Error("검수 화면이 바뀌었습니다. 다시 불러와 주세요.")); return;
          }
          if (isReady(project)) resolve(true);
        };
        callReviewLayoutWaiters.add(check); check();
      });
      try { return await Promise.race([ready, session.aborted]); }
      finally { callReviewLayoutWaiters.delete(check); }
    }
    async function waitForCallReviewLayout(session) {
      const project = bridge.state().project;
      if (!reviewingPreflight(project)) return true;
      const step = current, reviewStage = stageId(project);
      return waitForReviewState(session, next => current === step && stageId(next) === reviewStage && reviewingPreflight(next),
        callReviewLayoutReady);
    }
    function cancelReferenceApproval() {
      if (referenceApprovalContext) clearTimeout(referenceApprovalContext.watchdog);
      referenceApprovalContext = null; referenceApprovalPending = false;
    }
    function clearReferenceGroup() { referenceGroup = null; cancelReferenceApproval(); }
    function clearReferenceHeight() { shell.style.setProperty("--reference-content-height", ""); }
    function syncReferenceFrame() {
      const frame = $("reference-preview-frame"), source = frame?.contentWindow || null, url = frame?.getAttribute("src") || "";
      if (frame !== referenceFrameElement || source !== referenceFrameWindow || url !== referenceFrameUrl) {
        clearReferenceGroup(); clearReferenceHeight(); referenceMode = null; referenceReviewComplete = false;
        referenceFrameElement = frame; referenceFrameWindow = source; referenceFrameUrl = url;
      }
      return frame;
    }
    $("reference-preview-frame").addEventListener("load", () => {
      syncReferenceFrame(); clearReferenceGroup(); clearReferenceHeight(); referenceMode = null; referenceReviewComplete = false;
      if (current === "voice-listening") render();
      const frame = $("reference-preview-frame"), project = bridge.state().project;
      if (project?.id && project.id === lastProject?.id) frame.contentWindow?.postMessage?.(
        { type: "koww:reference-status", project: project.id }, window.location.origin);
    });
    let hoveredIndex = null, focusedIndex = null;
    const ticks = new Map(definitions.map(([id, label]) => {
      const tick = button(undefined, "story-index-step", () => navigate(screens.indexOf(id)));
      tick.dataset.step = id; tick.setAttribute("aria-label", label);
      tick.append(node("span", undefined, "story-index-tick"), node("span", label, "story-index-label"));
      tick.addEventListener("pointerenter", event => {
        if (event.pointerType === "touch" || !interactiveIndex(id)) return;
        hoveredIndex = id; updateIndexDistances();
      });
      tick.addEventListener("pointerleave", () => {
        if (hoveredIndex === id) hoveredIndex = null;
        updateIndexDistances();
      });
      tick.addEventListener("pointerdown", () => {
        if (!interactiveIndex(id)) return;
        focusedIndex = null; updateIndexDistances();
      });
      tick.addEventListener("focus", () => {
        if (!interactiveIndex(id) || !tick.matches?.(":focus-visible")) return;
        focusedIndex = id; updateIndexDistances();
      });
      tick.addEventListener("blur", () => {
        if (focusedIndex === id) focusedIndex = null;
        updateIndexDistances();
      });
      return [id, tick];
    }));
    index.append(...order.map(id => ticks.get(id)));
    function interactiveIndex(id) {
      const tick = ticks.get(id);
      return Boolean(tick && screens.includes(id) && !tick.hidden && !tick.disabled && !index.hidden && !index.inert && !transitioning);
    }
    function updateIndexDistances() {
      if (!interactiveIndex(hoveredIndex)) hoveredIndex = null;
      if (!interactiveIndex(focusedIndex)) focusedIndex = null;
      const center = screens.indexOf(hoveredIndex || focusedIndex);
      screens.forEach((id, position) => {
        ticks.get(id).dataset.distance = String(center < 0 ? 3 : Math.min(Math.abs(position - center), 3));
      });
    }
    function clearIndexMagnification() {
      hoveredIndex = focusedIndex = null; updateIndexDistances();
    }
    index.addEventListener("pointerleave", () => { hoveredIndex = null; updateIndexDistances(); });
    window.addEventListener("blur", clearIndexMagnification);
    document.addEventListener?.("visibilitychange", () => { if (document.hidden) clearIndexMagnification(); });
    for (const [id, label] of [["training", "학습용"], ["evaluation", "공통 평가용"]]) {
      const tab = button(label, "story-material-tab", () => { materialTab = id; renderMaterials(); });
      tab.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        materialTab = event.key === "Home" ? "training" : event.key === "End" ? "evaluation" : materialTab === "training" ? "evaluation" : "training";
        renderMaterials();
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
      clearPreflightPicker();
      voiceListening.classList.remove("is-selecting");
      for (const id of ["review-frame", "reference-preview-frame", "hard-preview-frame", "result-frame"]) bridge.pause($(id));
      stopHistoryPlayback(true);
      if (listenDialog.dialog.open) listenDialog.dialog.close();
    }
    function stopTransition() {
      cancelReferenceApproval();
      const session = transitionSession; transitionSession = null;
      if (preflightResumeIntent?.session === session) preflightResumeIntent = null;
      if (boundaryNextIntent?.session === session) boundaryNextIntent = null;
      if (session) { session.cancel(); session.waiters?.clear(); for (const timer of session.timers) clearTimeout(timer); }
      transitioning = false; creationPending = false; loader.hidden = true;
      loader.classList.remove("is-finishing");
      index.classList.remove("is-transitioning"); index.inert = false;
      clearIndexMagnification();
      panel.classList.remove("is-leaving"); panel.inert = false; panel.removeAttribute("aria-busy");
    }
    function activeTransition(session) { return transitionSession === session; }
    function beginTransition(backwards, initial = false) {
      cancelReferenceApproval();
      if (!initial) pauseMedia();
      const session = { timers: new Set(), sourceStep: current, sourceFollowing: following, project: bridge.state().project?.id || "", waiters: new Map() };
      session.aborted = new Promise(resolve => { session.cancel = () => resolve(false); });
      transitionSession = session; transitioning = true;
      if (!initial) index.classList.add("is-transitioning"); index.inert = true;
      clearIndexMagnification();
      panel.inert = true; panel.setAttribute("aria-busy", "true"); if (!initial) panel.classList.add("is-leaving");
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
    function chapterFrameUrl(frame) {
      const value = frame?.getAttribute("src");
      if (!value) return "";
      try {
        const url = new URL(value, window.location.href || window.location.origin + "/");
        if (url.origin !== window.location.origin || !["http:", "https:"].includes(url.protocol)) return "";
        url.hash = ""; return url.href;
      } catch { return ""; }
    }
    function destinationFrames() {
      const project = bridge.state().project;
      if (!project) return [];
      if (current === "voice-listening" && project.reference_preview?.url) return [$("reference-preview-frame")];
      if (["results", "microphone"].includes(current) && !results.hidden) return [$("result-frame")];
      if (!review.hidden && historic.hidden && project.review) return [$("review-frame")];
      return [];
    }
    function requestChapterFrame(session, waiter) {
      if (!activeTransition(session)) return;
      const state = chapterFrameStates.get(waiter.id);
      waiter.resolve?.(false);
      waiter.promise = new Promise((resolve, reject) => { waiter.resolve = resolve; waiter.reject = reject; });
      waiter.ready = false; waiter.generation = state?.generation;
      waiter.frame = state?.frame; waiter.source = state?.source; waiter.url = state?.url;
      waiter.request = ++chapterRequestSequence;
      if (!state?.loaded || !state.url || typeof state.source?.postMessage !== "function") return;
      if (waiter.id === "review-frame" && reviewingPreflight(bridge.state().project)) {
        preflightHeightReceived = false; preflightStateReceived = false; preflightState = null; preflightRequest++;
        state.source.postMessage({ type: "koww:preflight-status", ...reviewWirePhase(bridge.state().project), project: waiter.project,
          queue_sha256: bridge.state().project.review.queue_sha256, request: preflightRequest }, window.location.origin);
      }
      state.source.postMessage({ type: "koww:chapter-request", project: waiter.project,
        request: waiter.request, url: state.url }, window.location.origin);
    }
    function refreshChapterWaiters(session, loadedFrame) {
      for (const id of chapterFrames) {
        const frame = $(id), url = chapterFrameUrl(frame), previous = chapterFrameStates.get(id), previousGeneration = previous?.generation;
        let state = previous;
        if (!state || frame !== state.frame || frame?.contentWindow !== state.source || url !== state.url) {
          state = { frame, source: frame?.contentWindow, url, loaded: false, generation: (previous?.generation || 0) + 1 };
          chapterFrameStates.set(id, state);
        }
        if (frame === loadedFrame) { state.loaded = Boolean(url); state.generation++; }
        if (session && activeTransition(session) && state.generation !== previousGeneration) session.contentVersion = (session.contentVersion || 0) + 1;
        const waiter = session?.waiters.get(id);
        if (waiter && activeTransition(session) && waiter.generation !== state.generation) requestChapterFrame(session, waiter);
      }
    }
    for (const id of chapterFrames) $(id).addEventListener("load", () => {
      refreshChapterWaiters(transitionSession, $(id));
    });
    window.addEventListener("message", event => {
      const session = transitionSession, data = event.data;
      if (!session || !activeTransition(session) || event.origin !== window.location.origin
        || data?.type !== "koww:chapter-ready" || data.project !== bridge.state().project?.id
        || !Number.isSafeInteger(data.request) || data.request < 1) return;
      const waiter = [...session.waiters.values()].find(value => value.request === data.request);
      if (!waiter || waiter.project !== data.project || event.source !== waiter.source
        || $(waiter.id) !== waiter.frame || waiter.frame.contentWindow !== waiter.source
        || chapterFrameUrl(waiter.frame) !== waiter.url || data.url !== waiter.url) return;
      if (data.status === "ready") { waiter.ready = true; waiter.resolve(true); }
      else if (data.status === "error") {
        const detail = typeof data.message === "string" ? data.message.slice(0, 240) : "";
        waiter.reject(Error(detail || "다음 화면의 자료를 불러오지 못했습니다."));
      }
    });
    async function chapterPaint(session) {
      if (document.fonts?.ready) await Promise.race([document.fonts.ready, session.aborted]);
      for (let i = 0; i < 2 && activeTransition(session); i++) {
        if (typeof requestAnimationFrame !== "function") break;
        await Promise.race([new Promise(resolve => requestAnimationFrame(resolve)), session.aborted]);
      }
      return activeTransition(session);
    }
    async function waitForChapter(session) {
      let timer;
      const reviewQueuePending = preparingReviewQueue(bridge.state().project);
      const timeout = new Promise((_, reject) => {
        if (!reviewQueuePending && !destinationFrames().some(Boolean) && !document.fonts?.ready) return;
        timer = setTimeout(() => reject(Error("화면을 불러오는 시간이 길어지고 있습니다. 다시 이동해 주세요.")), current === "microphone" ? 45000 : 20000);
        session.timers.add(timer);
      });
      try {
        const ready = (async () => {
          const project = bridge.state().project?.id || "";
          if (project && ["scale", "materials"].includes(current)) {
            await bridge.prepareRecipes?.();
            if (!activeTransition(session)) return false;
          }
          if (reviewQueuePending) {
            const step = current, reviewStage = stageId(bridge.state().project);
            const queueAvailable = next => !running(next.status) && !next.error && reviewingPreflight(next) && !review.hidden;
            if (!await waitForReviewState(session, next => current === step && stageId(next) === reviewStage
                && (preparingReviewQueue(next) || queueAvailable(next)), queueAvailable)) return false;
          }
          refreshChapterWaiters(session);
          for (const frame of destinationFrames().filter(Boolean)) {
            frame.setAttribute("loading", "eager");
            const waiter = { id: frame.id, project };
            session.waiters.set(frame.id, waiter); requestChapterFrame(session, waiter);
          }
          while (activeTransition(session)) {
            await Promise.all([...session.waiters.values()].map(waiter => waiter.promise));
            if (!await waitForCallReviewLayout(session)) return false;
            const version = session.contentVersion || 0;
            if (!await chapterPaint(session)) return false;
            if ([...session.waiters.values()].every(waiter => waiter.ready)
              && (!reviewingPreflight(bridge.state().project) || callReviewLayoutReady())
              && version === (session.contentVersion || 0)) {
              session.readyVersion = session.contentVersion || 0; return true;
            }
          }
          return false;
        })();
        return await Promise.race([ready, timeout, session.aborted]);
      } finally {
        clearTimeout(timer); session.timers.delete(timer); session.waiters.clear();
      }
    }
    async function prepareInitialView() {
      // Boot has no user navigation intent. Build the authoritative current
      // chapter, then hold its reveal until child geometry and paint settle.
      stopTransition();
      const project = bridge.state().project;
      if (project) { current = stageStep(project); lastFrontier = current; following = true; historyKey = ""; }
      render();
      const session = beginTransition(false, true);
      try {
        if (!await waitForChapter(session)) throw Error("화면 준비가 중단됐습니다. 다시 불러와 주세요.");
        return true;
      } finally {
        if (activeTransition(session)) { stopTransition(); render(); }
      }
    }
    async function runTransition(session, preparation, commit) {
      const prepared = Promise.resolve(preparation).then(value => ({ ok: true, value }), error => ({ ok: false, error }));
      try {
        if (!await waitForMotion(panel, session)) return;
        loader.hidden = false;
        // Build the destination behind the opaque notice. Embedded documents
        // can load and measure here, before the notice starts disappearing.
        const content = (async () => {
          const result = await Promise.race([prepared, session.aborted]);
          if (!activeTransition(session) || !result) return false;
          if (!result.ok) throw result.error;
          if (creationPending) { creationPending = false; session.createdProject = bridge.state().project?.id || ""; }
          commit?.(); session.staged = true; render();
          if (!activeTransition(session)) return false;
          return waitForChapter(session);
        })().then(ok => ({ ok }), error => ({ ok: false, error }));
        if (!await waitForMotion(loader, session)) return;
        // Hold the fully visible notice for at least 0.8 seconds. Server
        // preparation runs concurrently and can keep it visible for longer.
        const [result, held] = await Promise.all([
          Promise.race([content, session.aborted]), waitForDuration(session, 800),
        ]);
        if (!held || !activeTransition(session)) return;
        if (!result?.ok) {
          if (session.hardReviewArrival) hardReviewRevealFailed = true;
          if (session.repairQueue === repairQueueKey(bridge.state().project)) { repairReviewConcealed = session.repairQueue; repairRevealFailed = session.repairQueue; }
          if (session.staged) { current = session.sourceStep; following = session.sourceFollowing; historyKey = ""; render(); }
          bridge.message(result?.error?.message || "화면을 준비하지 못했습니다. 다시 이동해 주세요.", true);
        }
        while (activeTransition(session)) {
          if (result?.ok && session.readyVersion !== (session.contentVersion || 0) && !await waitForChapter(session)) return;
          const version = session.contentVersion || 0;
          loader.classList.add("is-finishing");
          if (!await waitForMotion(loader, session)) return;
          if (version === (session.contentVersion || 0) || !result?.ok) break;
          loader.classList.remove("is-finishing");
          if (!await waitForMotion(loader, session) || !await waitForChapter(session)) return;
        }
      } catch (error) {
        if (activeTransition(session)) {
          if (session.hardReviewArrival) hardReviewRevealFailed = true;
          if (session.repairQueue === repairQueueKey(bridge.state().project)) { repairReviewConcealed = session.repairQueue; repairRevealFailed = session.repairQueue; }
          if (session.staged) { current = session.sourceStep; following = session.sourceFollowing; historyKey = ""; render(); }
          bridge.message(error?.message || "화면을 전환하지 못했습니다. 다시 시도해 주세요.", true);
        }
      } finally {
        if (activeTransition(session)) { stopTransition(); render(); heading.focus({ preventScroll: true }); }
      }
    }
    let composingWakeword = false;
    function showWakewordError(error) {
      if (wakewordFeedback.textContent !== error) wakewordFeedback.textContent = error;
      wakewordFeedback.hidden = !error;
      if (error) wakeword.setAttribute("aria-invalid", "true");
      else wakeword.removeAttribute("aria-invalid");
    }
    function navigate(position, automatic = false) {
      const state = bridge.state();
      if (transitioning || position < 0 || position > maximum || position >= screens.length || state.busy || position === screens.indexOf(current)) return false;
      if (!state.project && position > screens.indexOf(current)) {
        if (composingWakeword) return false;
        const error = bridge.wakewordError(wakeword.value);
        showWakewordError(error);
        if (error) { wakeword.focus(); return false; }
      }
      const backwards = position < screens.indexOf(current), target = screens[position];
      if (!automatic) hardConnectionIntent = null;
      const session = beginTransition(backwards);
      following = automatic || (state.project && (target === stageStep(state.project) || position === frontier(state.project)));
      void runTransition(session, undefined, () => { current = target; historyPage = 0; historyKey = ""; });
      return true;
    }
    function humanGate(project) { return Boolean(project?.review || project?.input || (project?.migration && !project.process_alive)); }
    function draftBackend(state) {
      return setupBackend || (state.dependencySetup?.recommended_backend === "cu128" ? "cu128" : "cpu");
    }
    function waitingToolChoice(project) {
      return project?.status === "waiting_input" && stageStep(project) === "tools"
        && project.input?.kind === "dependency_setup" && typeof bridge.input === "function";
    }
    function editingPhrases(project) {
      return Boolean(project && current === "phrases" && stageId(project) === "hard_phrases"
        && project.status === "waiting_input" && project.input?.kind === "hard_phrases"
        && !project.review && !project.migration && project.process_alive !== true && typeof bridge.input === "function");
    }
    function canStartDataPreparation(project) {
      const navigation = navigationCheckpoint(project), download = downloadCheckpoint(project);
      return Boolean(project && !humanGate(project) && !completedStep(project, current)
        && (!navigation || navigation.state === "awaiting_start") && (!download || download.state === "awaiting_start")
        && ["ready", "idle", "pending", "created"].includes(project.status));
    }
    function releaseStep(project) {
      if (!project || humanGate(project) || project.status !== "paused") return "";
      if (["voices", "voice-listening"].includes(current) && referencePreparationNeeded(project) && project.reference_preparation.ready !== true) return "";
      const checkpoint = navigationCheckpoint(project);
      if (checkpoint?.state === "awaiting_next") {
        const step = current === "voice-listening" ? "voices" : current;
        const reference = project.reference_preview;
        if (step === "voices" && reference?.review_required
          && (current !== "voice-listening" || !(reference.review_complete || referenceReviewComplete))) return "";
        if (step === checkpoint.step && completedNavigationStep(project, step)) return step;
      }
      if (current === "downloads" && downloadCheckpoint(project)?.state === "awaiting_next"
        && completedStep(project, "downloads")) return "downloads";
      return "";
    }
    function reviewingReferenceGroup(project) {
      return Boolean(project && current === "voice-listening" && referenceMode === "candidate_pool_v2"
        && project.reference_preview?.review_required && !(project.reference_preview.review_complete || referenceReviewComplete));
    }
    function callReviewRoute(project) {
      if (current === "hard-review") {
        if (project?.review?.queue_id && project.review.queue_id !== "hard") return null;
        const sampleFirst = sampleFirstHardReview(project);
        return { phase: "hard", step: current, stage: "hard_review", target: sampleFirst ? "hard-generation" : "features",
          destinationStages: sampleFirst ? ["hard_generation", "hard_connect"] : ["features"],
          transitStages: sampleFirst ? [] : ["hard_connect"],
          retainStages: ["hard_review", "hard_connect", "hard_generation", "features"] };
      }
      if (!twoPhasePositiveReview(project)) return null;
      if (current === "call-pilot-review") {
        if (project.review?.queue_id && project.review.queue_id !== "pilot") return null;
        return { phase: "pre", step: current, stage: "pilot_review", target: "positive-generation",
          destinationStages: ["positives", "positive_review"], retainStages: ["pilot", "pilot_review", "positives", "positive_review"] };
      }
      if (current === "pronunciation") return { phase: "post", step: current, stage: "positive_review", target: "phrases",
        destinationStages: ["hard_phrases"], retainStages: ["positive_review", "hard_phrases"] };
      return null;
    }
    function reviewingPreflight(project) {
      const route = callReviewRoute(project);
      return Boolean(project && route && current === stageStep(project) && stageId(project) === route.stage && !done(project.status)
        && project.review?.queue_sha256);
    }
    function preflightReady(project) {
      return Boolean(reviewingPreflight(project) && preflightState?.ready && !preflightFinalizePending && !preflightCommitted
        && preflightState.queue_sha256 === project.review.queue_sha256 && typeof $("review-frame").contentWindow?.postMessage === "function");
    }
    function referenceGroupReady(project) {
      const navigation = navigationCheckpoint(project), frame = $("reference-preview-frame");
      return Boolean(reviewingReferenceGroup(project) && referenceGroup?.ready && !referenceApprovalPending
        && (!referencePreparationNeeded(project) || project.reference_preparation.ready === true)
        && project.status === "paused" && !humanGate(project) && navigation?.state === "awaiting_next"
        && navigation.step === "voices" && completedNavigationStep(project, "voices")
        && frame === referenceFrameElement && frame.contentWindow === referenceFrameWindow
        && (frame.getAttribute("src") || "") === referenceFrameUrl && typeof frame.contentWindow?.postMessage === "function");
    }
    function operationAction(project) {
      if (hardReviewRevealFailed && current === "hard-review" && stageId(project) === "hard_review"
          && (preparingReviewQueue(project) || reviewingPreflight(project)) && !project.error) return "load_hard_review";
      if (repairRevealFailed && repairReviewConcealed === repairQueueKey(project) && current === "pronunciation") return "load_review";
      if (project && current === "voices" && referencePreparationNeeded(project) && !humanGate(project)
        && project.reference_preparation.operation === "helper" && project.reference_preparation.ready !== true) {
        const preparation = project.reference_preparation;
        if (preparation.status === "pausing" || preparation.can_pause === true) return "prepare_pause";
        if (preparation.can_resume === true) return "prepare_resume";
        if (preparation.can_start === true) return "prepare_start";
        return "";
      }
      if (!project || humanGate(project) || current !== stageStep(project) || done(project.status)
        || navigationCheckpoint(project)?.state === "awaiting_next" || downloadCheckpoint(project)?.state === "awaiting_next"
        || ["entry", "scale", "materials", "microphone"].includes(current)
        || current === "voice-listening" && !(mergedReferenceReview(project) && ["references", "references_review"].includes(stageId(project)))) return "";
      if (running(project.status)) return "pause";
      if (current === "downloads" && project.dependency_install_pending === true) return "start";
      if (["paused", "failed", "error", "interrupted"].includes(project.status)) return "resume";
      if (current === "downloads") return canStartDataPreparation(project) ? "start" : "";
      if (["ready", "idle", "pending", "created"].includes(project.status)) return "start";
      return "";
    }
    async function controlJob() {
      const state = bridge.state(), action = operationAction(state.project);
      if (!action || transitioning || state.busy || state.project.status === "pausing"
        || current === "voices" && referencePreparationNeeded(state.project) && state.project.reference_preparation.status === "pausing") return;
      hardConnectionIntent = null;
      if (action === "load_hard_review") { revealHardReview(true); return; }
      if (action === "load_review") { revealRepairReview(state.project, true); return; }
      await bridge.action(action);
    }
    async function advance() {
      const state = bridge.state(), project = state.project, position = screens.indexOf(current);
      if (transitioning || state.busy || phraseInputPending || referenceApprovalContext) return;
      if (!project) {
        if (current === "materials") {
          if ($("create").disabled) return;
          // Create the model without starting transfers. The presentation waits
          // for its result and the same minimum used by local transitions.
          // Polls may receive the created model meanwhile, but must not reveal
          // it early or allow another creation request.
          const session = beginTransition(false); creationPending = true;
          await runTransition(session, (async () => bridge.create({
            navigationCheckpoints: true, dependencyBackend: draftBackend(state),
          }))());
          return;
        }
        navigate(position + 1); return;
      }
      if (current === "tools" && waitingToolChoice(project)) {
        navigate(screens.indexOf("scale"));
        return;
      }
      if (current === "materials" && waitingToolChoice(project)) {
        const session = beginTransition(false);
        following = true;
        await runTransition(session, (async () => {
          const saved = await bridge.input();
          const latest = bridge.state().project;
          if (saved !== true || latest?.id !== project.id || latest.dependency_install_pending !== true)
            throw new Error("학습 도구 선택을 저장하지 못했습니다. 다시 진행해 주세요.");
        })(), () => { current = "downloads"; historyPage = 0; historyKey = ""; });
        return;
      }
      if (reviewingReferenceGroup(project)) {
        syncReferenceFrame();
        if (!referenceGroupReady(project)) return;
        referenceApprovalContext = { project: project.id, group: referenceGroup.group, frame: referenceFrameElement,
          source: referenceFrameWindow, url: referenceFrameUrl, request: ++referenceApprovalRequest };
        const approval = referenceApprovalContext;
        approval.watchdog = setTimeout(() => {
          if (referenceApprovalContext !== approval) return;
          referenceApprovalContext = null; referenceApprovalPending = false;
          bridge.message?.("목소리 승인 응답을 확인하지 못했습니다. 저장 상태를 확인한 뒤 다시 진행하세요.", true); render();
        }, 45000);
        referenceApprovalPending = true; render();
        $("reference-preview-frame").contentWindow.postMessage({ type: "koww:reference-approve", project: project.id,
          group: referenceGroup.group, request: referenceApprovalContext.request }, window.location.origin);
        return;
      }
      if (reviewingPreflight(project)) {
        syncPreflightFrame(); if (!preflightReady(project)) return;
        preflightFinalizePending = true; next.disabled = true;
        const session = beginTransition(false), intent = { project: project.id, queue: project.review.queue_sha256,
          frame: $("review-frame"), source: $("review-frame").contentWindow, url: $("review-frame").getAttribute("src") || "",
          request: preflightRequest, phase: "finalizing", route: callReviewRoute(project), session };
        const preparation = new Promise((resolve, reject) => { intent.resolve = resolve; intent.reject = reject; });
        preflightResumeIntent = intent; following = true;
        if (intent.route.phase === "post") loaderText.textContent = "검수 결과를 저장하고 있습니다...";
        const watchdog = setTimeout(() => { if (preflightResumeIntent === intent) cancelPreflightNext("다음 단계 준비 상태를 확인하지 못했습니다. 다시 진행하세요."); }, intent.route.phase === "post" ? 90000 : 45000);
        intent.watchdog = watchdog; session.timers.add(watchdog);
        void runTransition(session, preparation, () => { current = intent.target; historyPage = 0; historyKey = ""; });
        render();
        $("review-frame").contentWindow.postMessage({ type: "koww:preflight-finalize", ...reviewWirePhase(project), project: project.id,
          queue_sha256: project.review.queue_sha256, request: preflightRequest }, window.location.origin); return;
      }
      if (editingPhrases(project)) {
        const inputKey = JSON.stringify(project.input);
        phraseInputPending = true; render();
        try {
          const saved = await bridge.input();
          const latest = bridge.state().project;
          if (saved === false || latest?.id !== project.id || document.hidden || current !== "phrases"
            || saved !== true && latest.status === "waiting_input" && JSON.stringify(latest.input) === inputKey) return;
          const intent = beginBoundaryNext(latest, { step: "phrases", sourceStage: "hard_phrases", inputKey,
            target: sampleFirstHardReview(project) ? "hard-review" : "hard-generation", destinationStages: ["hard_pilot", "hard_generation", "hard_review"] });
          intent.saved = true; resumeSavedBoundary();
        }
        catch (error) { bridge.message?.(error?.message || "유사어를 저장하지 못했습니다. 다시 진행하세요."); }
        finally { phraseInputPending = false; render(); }
        return;
      }
      if (current === stageStep(project) && humanGate(project)) return;
      if (position < maximum) { navigate(position + 1); return; }
      if (releaseStep(project)) await bridge.action("resume");
    }
    function openHardListening() {
      if (!lastProject || transitioning) return;
      listening = "hard";
      stopHistoryPlayback(true);
      bridge.pause($("review-frame"));
      listenDialog.title.textContent = "유사어 전체 듣기";
      listenDialog.dialog.setAttribute("aria-describedby", "hard-preview-counts");
      bridge.preview("hard");
      $("reference-preview").hidden = true;
      $("hard-preview").hidden = false;
      $("hard-preview").open = true;
      bridge.view("review", "hard"); openDialog(listenDialog);
    }
    function openPositiveListening() {
      const url = lastProject?.positive_preview?.url;
      if (transitioning || typeof url !== "string" || !url.startsWith("/positive-preview/?")) return;
      stopHistoryPlayback(true);
      bridge.pause($("review-frame"));
      listening = "positive";
      listenDialog.title.textContent = "호출어 전체 듣기";
      listenDialog.dialog.removeAttribute("aria-describedby");
      $("hard-preview").hidden = true;
      positiveFrame.setAttribute("src", url); positiveFrame.hidden = false;
      openDialog(listenDialog);
    }
    function setView(view, mode) { if (!listening) bridge.view(view, mode); }
    function renderToolMaterials(backend, packages, downloadBytes) {
      toolMaterials.hidden = !Object.hasOwn(toolBackends, backend) || !Array.isArray(packages) || !packages.length;
      if (toolMaterials.hidden) return;
      const specs = new Map(packages.map(spec => spec.split(/==|>=/, 2)));
      const [torchVersion, torchBuild] = specs.get("torch")?.split("+") ?? [];
      const cuda = backend === "cu128" && torchBuild?.match(/^cu(\d{2})(\d)$/);
      const tools = [];
      if (specs.has("torch")) tools.push(torchVersion ? `PyTorch ${torchVersion}` : "PyTorch");
      if (specs.has("torchaudio")) tools.push("TorchAudio");
      if (!tools.length) { toolMaterials.hidden = true; return; }

      const header = node("div", undefined, "story-material-heading");
      const heading = node("h2", "연산 도구"); heading.id = "story-tool-materials-title";
      const device = toolBackends[backend][0];
      header.append(heading, node("span", cuda ? `${device} · CUDA ${cuda[1]}.${cuda[2]}` : device));
      const capacity = Number.isSafeInteger(downloadBytes) && downloadBytes > 0
        ? `다운로드 약 ${bridge.formatBytes(downloadBytes)}` : "다운로드 용량 미확정";
      const list = node("dl", undefined, "story-material-list");
      const row = node("div", undefined, "story-material-row");
      const title = node("dt"), amount = node("dd");
      title.append(node("span", tools.join(" · "), "story-material-source"));
      const download = node("span", capacity, "story-material-capacity");
      download.title = "PyTorch·TorchAudio 배포 파일 기준";
      amount.append(node("strong", `${tools.length}개`), download);
      row.append(title, amount); list.append(row);
      toolMaterials.replaceChildren(header, list);
    }
    function renderMaterials() {
      if (!materialData) return;
      const small = matchMedia("(max-width:760px), (max-height:700px)").matches;
      const sections = [];
      for (const [key, label] of [["training", "학습용 자료"], ["evaluation", "성능 평가용 자료"]]) {
        const section = node("section", undefined, "story-material-section"); section.hidden = small && key !== materialTab;
        const head = node("div", undefined, "story-material-heading");
        const total = materialData.downloadTotals?.[key];
        const totalLabel = node("span", total?.label || "용량 미정", "story-material-total");
        totalLabel.setAttribute("aria-label", `${label} 합산 다운로드 용량: ${totalLabel.textContent}`);
        totalLabel.title = "합산 다운로드 용량 · 재사용·공유 파일은 중복 제외 · 생성 파일과 모델·도구 공간은 별도" + (total?.unknown ? " · 미정인 자료는 합계에서 제외" : "");
        head.append(node("h2", label), totalLabel);
        const list = node("dl", undefined, "story-material-list"), rows = materialData[key] || [];
        for (const item of rows) {
          const row = node("div", undefined, "story-material-row"), amount = node("dd");
          const names = [...new Set((item.sources || []).map(source => (source.name === "RIRS_NOISES" ? "OpenSLR RIR" : source.name) + (source.downloadReuse === "학습용 자료와 공유" ? "(공유)" : "")))];
          const title = node("dt");
          title.append(node("span", item.name, "story-material-name"), node("span", names.join(" · ") || item.description, "story-material-source"));
          row.append(title);
          const capacity = node("span", item.downloadLabel || "다운로드 용량 미확정", "story-material-capacity");
          capacity.title = (item.sources || []).map(source => `${source.name}: ${source.downloadReuse || (Number.isFinite(source.downloadBytes) ? `${source.downloadEstimated ? "약 " : ""}${bridge.formatBytes(source.downloadBytes)}` : "용량 미확정")}`).join("\n");
          amount.append(node("strong", item.amount), capacity); row.append(amount); list.append(row);
        }
        section.append(head, list); sections.push(section);
      }
      materialColumns.replaceChildren(...sections);
      for (const tab of materialsTabs.children) { tab.setAttribute("aria-selected", String(tab.dataset.group === materialTab)); tab.tabIndex = tab.dataset.group === materialTab ? 0 : -1; }
    }
    function chapterWorkActive(project, historical) {
      if (current === "voices" && separateReferencePreparation(project)) return !historical && referencePreparationNeeded(project)
        && ["running", "pausing"].includes(project.reference_preparation.status) && !humanGate(project);
      return !historical && Boolean(project) && stageStep(project) === current && running(project.status)
        && project.process_alive !== false && !project.input && !project.review;
    }
    function preparationRow(project, label, active, statusText = "준비 중") {
      return { label, indeterminate: true, active,
        statusText: active ? statusText : project.status === "paused" ? "일시정지"
          : ["error", "failed", "interrupted", "cancelled", "stopped"].includes(project.status) ? "확인 필요" : "대기 중" };
    }
    function generationActivity(progress) {
      const phase = progress?.activity?.phase || progress?.phase;
      const fallback = { tts_model_loading: "음성 생성 모델 준비 중", tts_prompt_preparation: "기준 목소리 복제 정보 준비 중",
        tts_qualification: "음성 생성 가속 확인 중", tts_finalizing: "생성 음원 검증·저장 중",
        reference_preparation_finalizing: "기준 목소리 후보 검증 중" }[phase];
      if (!fallback) return "";
      const label = progress?.activity?.label ?? (progress.kind === "studio_task_progress_v1" ? progress.label : "");
      return (typeof label === "string" && label.trim() ? label.trim().slice(0, 120) : fallback) + phaseElapsed(progress);
    }
    function workRows(project, historical, active, completed) {
      if (connectingHard(project)) return [{ label: "유사어 학습 자료 연결", statusText: "확인 중", indeterminate: true, active: true }];
      const stage = stageId(project);
      if (current === "results" && stage === "results" && !historical && !completed) {
        return [preparationRow(project, "학습 결과 정리", active, "정리 중")];
      }
      if (repairProgressVisible(project)) {
        const repair = postRepair(project), done = Number.isFinite(repair.done) && repair.done >= 0 ? repair.done : null;
        const total = Number.isFinite(repair.total) && repair.total > 0 ? repair.total : null;
        return [{ label: repair.label || "부적합 목소리 호출어 다시 생성", done, total, unit: repair.unit || "개",
          active, completed: repair.state === "waiting_review" && total !== null && done >= total }];
      }
      if (current === "voices") {
        if (separateReferencePreparation(project)) {
          const preparation = project.reference_preparation || {};
          const finalizing = active && (stage === "references_review" || preparation.rows?.length > 0
            && preparation.rows.every(row => row.completed === true));
          const activity = active ? generationActivity(preparation) || generationActivity(project.progress)
            || (finalizing ? "기준 목소리 후보 검증 중" : "") : "";
          const activitySplit = preparation.rows?.find(row => row.active && !row.completed)?.split || "train";
          return ["train", "validation"].map(split => {
            const row = preparation.rows?.find(row => row.split === split) || {};
            const totalKnown = Number.isSafeInteger(row.total) && row.total > 0;
            return { ...row, label: row.label || (split === "train" ? "학습용 목소리 후보" : "공통 평가용 목소리 후보"),
              detail: [totalKnown ? `총 ${count(row.total)}개` : "", split === activitySplit ? activity : ""].filter(Boolean).join(" · "),
              completed: row.completed === true, active: row.active === true && !row.completed
                && ["running", "pausing"].includes(preparation.status) };
          });
        }
        const targets = project.reference_counts || {}, preview = project.reference_preview || {};
        const banks = [["학습용 목소리", "train"], ["공통 평가용 목소리", "validation"]].map(([label, key]) => {
          const total = targets[key], count = preview[key] ?? 0;
          return { label, done: count, total, completed: Number.isFinite(total) && total > 0 && count >= total };
        });
        // The worker generates the two voice banks in this order.
        const activeBank = active && stage === "references"
          ? banks.find(bank => bank.done < bank.total) : null;
        const activity = active ? generationActivity(project.progress) : "";
        return banks.map(bank => ({ ...bank, detail: [
          Number.isSafeInteger(bank.total) && bank.total > 0 ? `총 ${count(bank.total)}개` : "",
          bank === (activeBank || banks[0]) ? activity : "",
        ].filter(Boolean).join(" · "), active: bank === activeBank }));
      }
      const sampleLabel = current === "pronunciation" && stage === "pilot" ? "호출어 검수용 샘플"
        : current === "hard-review" && stage === "hard_pilot" ? "유사어 검수용 샘플" : "";
      const label = workLabels[current] || sampleLabel;
      if (!label) return [];
      if (historical) {
        if (!generationSteps.has(current)) return [];
        const view = project.completed_stage_views?.[current];
        return view?.completed && completed && Array.isArray(view.rows)
          ? view.rows.map(row => ({ ...row, completed: true })) : [];
      }
      // Only an explicit, task-scoped backend preparation record identifies
      // startup. A genuine 0-update training snapshot remains numeric.
      const preparation = project.training_preparation;
      if (current === "training" && ["training", "train"].includes(stage) && !completed
          && ["training_preparation_v1", "studio_task_progress_v1"].includes(preparation?.kind)
          && ["feature_validation", "feature_cache_verification", "device_selection", "model_initialization", "gpu_cache_loading",
            "training_finalizing"].includes(preparation.phase)) {
        const preparationLabel = preparation.label || "학습 자료 및 모델 준비";
        if (preparation.unit === "bytes" && Number.isFinite(preparation.done) && preparation.done >= 0
            && Number.isFinite(preparation.total) && preparation.total > 0 && preparation.done <= preparation.total)
          return [{ ...preparation, label: preparationLabel, active, completed: false }];
        return [preparationRow(project, preparationLabel, active, preparation.phase === "training_finalizing" ? "정리 중" : "준비 중")];
      }
      const progress = project.progress;
      const generation = generationSteps.has(current) || Boolean(sampleLabel);
      const detail = generation && active && !completed && stageScreen(project, stage) === current ? generationActivity(progress) : "";
      const done = progress?.done ?? progress?.completed ?? progress?.current;
      const evaluationPhaseLabel = current === "evaluation"
        ? { validation: "평가 음원 검증", finalizing: "평가 결과 정리", preparation_validation: "평가 자료 원본 검증",
          preparation_generation: "평가 자료 생성", preparation_finalizing: "평가 자료 등록·검증" }[progress?.phase] : "";
      if (!Number.isFinite(done) || done < 0 || !Number.isFinite(progress.total)
        || progress.total <= 0 || /단계/.test(progress.unit || "")) {
        if (current === "features" && stage === "features" && !completed) {
          return [preparationRow(project, "자료 확인 및 특징 추출 준비", active, "확인 중")];
        }
        if (current === "evaluation" && ["development", "evaluation"].includes(stage) && !completed) {
          const progressLabel = /단계/.test(progress?.unit || "") ? "" : progress?.label;
          const statusText = { validation: "검증 중", scoring: "평가 중", finalizing: "정리 중", runtime_recommendation: "비교 중",
            preparation_validation: "검증 중", preparation_generation: "생성 중", preparation_finalizing: "등록·검증 중" }[progress?.phase] || "준비 중";
          return [preparationRow(project, evaluationPhaseLabel || progressLabel || (stage === "development" ? "평가 자료 준비" : "성능 평가 준비"), active, statusText)];
        }
        if (current === "training" && stage === "package" && !completed)
          return [preparationRow(project, "학습 모델 정리", active, "정리 중")];
        if (!completed && stageScreen(project, stage) === current && generation)
          return [{ ...preparationRow(project, sampleLabel || label, active), detail }];
        return [];
      }
      const progressLabel = current === "features" ? featureLabels[progress.label] || progress.label : evaluationPhaseLabel || progress.label;
      return [{ ...progress, done, label: sampleLabel || (generationSteps.has(current) ? label : progressLabel || label), detail,
        active: active && !completed, completed }];
    }
    const progressStatus = (active, completed) => completed ? "complete" : active ? "active" : "";
    function updateProgressIndicator(indicator, status) {
      indicator.dataset.status = status;
      indicator.hidden = !status;
      indicator.setAttribute("aria-hidden", String(status !== "complete"));
      indicator.setAttribute("role", "img");
      indicator.setAttribute("aria-label", "완료");
    }
    function workProgressRow() {
      const row = node("div", undefined, "story-voice-row");
      const title = node("strong"), indicator = node("i", undefined, "story-progress-indicator"), amount = node("span"), bar = node("progress");
      const detail = node("span", undefined, "story-generation-summary"); detail.hidden = true;
      row.append(title, amount, bar);
      return { row, title, detail, indicator, amount, bar };
    }
    function updateWorkProgressRow({ title, detail, indicator, amount, bar }, { label = "자료", detail: detailText = "", done, total, unit = "개", active = false, completed = false, indeterminate = false, statusText = "" }) {
      const doneKnown = Number.isFinite(done) && done >= 0, totalKnown = Number.isFinite(total) && total > 0;
      if (title.dataset.label !== label) { title.textContent = label; title.dataset.label = label; }
      detail.hidden = !detailText;
      if (detail.textContent !== detailText) detail.textContent = detailText;
      if (current === "hard-generation" && hardGenerationSummary.parentElement !== title) title.append(hardGenerationSummary);
      if (detail.parentElement !== title) title.append(detail);
      if (indicator.parentElement !== title) title.append(indicator);
      updateProgressIndicator(indicator, progressStatus(active, completed));
      amount.textContent = statusText || (totalKnown ? progressAmount(done, total, unit) : quantity(done, unit));
      bar.hidden = !(doneKnown && totalKnown) && !indeterminate;
      if (indeterminate) {
        bar.removeAttribute("value"); bar.removeAttribute("max"); bar.dataset.active = String(active);
        bar.setAttribute("aria-label", label + " · 확인 중"); return;
      }
      if (!bar.hidden) {
        bar.max = total; bar.value = Math.min(done, total);
        bar.dataset.active = String(active);
        bar.setAttribute("aria-label", label + (generationSteps.has(current) ? " 파일 검사 통과" : " 진행률") + (active ? " · 진행 중" : ""));
      }
    }
    function renderEvaluationPlan(project) {
      const policy = project?.evaluation_policy;
      evaluationPlan.hidden = current !== "evaluation" || !Number.isSafeInteger(policy?.positive_sessions)
        || policy.positive_sessions <= 0 || !policy.snrs?.length || policy.noise_sections !== 1;
      if (evaluationPlan.hidden) return;
      const conditions = 1 + 3 * policy.snrs.length;
      const dry = policy.dry_positive_sessions ?? policy.positive_sessions;
      const lines = [`잔향 없음 · 원본 ${count(policy.validation_originals)}개 × ${conditions}조건 = ${count(dry)}회`];
      if (policy.reverb_positive_sessions > 0) {
        lines.push(`일반·강한 잔향 · 대표 원본 ${count(policy.reverb_originals)}개 × ${conditions}조건 × ${policy.reverb_groups.length}강도 = ${count(policy.reverb_positive_sessions)}회`);
      }
      lines.push(`${conditions}조건 = 배경음 없음 1조건 + 배경음 3종(말소리·환경음·음악/전자음) × 소음 강도 ${policy.snrs.length}단계`);
      const cells = project.results?.common_evaluation_assessment?.reverberation?.condition_counts;
      const actualCalls = Array.isArray(cells) && cells.length && cells.every(cell => Number.isSafeInteger(cell?.samples) && cell.samples >= 0)
        ? cells.reduce((sum, cell) => sum + cell.samples, 0) : NaN;
      const saved = project.completed_stage_views?.evaluation;
      const totalSessions = saved?.completed ? saved.rows?.[0]?.done : NaN;
      const haveActualCounts = Number.isSafeInteger(totalSessions) && Number.isSafeInteger(actualCalls)
        && actualCalls > 0 && totalSessions >= actualCalls;
      const negativeSessions = haveActualCounts ? totalSessions - actualCalls : null;
      if (haveActualCounts && actualCalls !== policy.positive_sessions) {
        lines.unshift(`계획한 호출 감지 시험 ${count(policy.positive_sessions)}회`);
      }
      const hours = project.results?.negative_hours ?? policy.development_negative_target_hours?.[0];
      const duration = Number.isFinite(hours) ? ` 약 ${hours.toLocaleString("ko-KR", { maximumFractionDigits: 1 })}시간` : "";
      lines.push(`오호출 시험 · 호출어 없는 배경${duration}`);
      evaluationPlanTitle.textContent = haveActualCounts
        ? `호출 감지 시험 ${count(actualCalls)}회 + 오호출 시험 ${count(negativeSessions)}세션 = 전체 ${count(totalSessions)}건`
        : `호출 감지 시험 계획 ${count(policy.positive_sessions)}회`;
      evaluationPlanDetail.textContent = lines.join("\n");
    }
    function quantity(value, unit = "개") {
      if (!Number.isFinite(value) || value < 0) return "—";
      if (unit === "bytes") {
        if (value < 1024) return `${count(value)} B`;
        const index = Math.min(4, Math.floor(Math.log(value) / Math.log(1024)));
        return `${(value / 1024 ** index).toLocaleString("ko-KR", { maximumFractionDigits: 1 })} ${["B", "KiB", "MiB", "GiB", "TiB"][index]}`;
      }
      if (unit === "seconds") {
        const seconds = Math.floor(value);
        return [[Math.floor(seconds / 3600), "시간"], [Math.floor(seconds % 3600 / 60), "분"], [seconds % 60, "초"]]
          .filter(([part]) => part > 0).map(([part, label]) => `${count(part)}${label}`).join(" ") || "0초";
      }
      const label = { files: "개", items: "개", samples: "개", updates: "회", "학습 업데이트": "회", rows: "행",
        hours: "시간", audio_hours: "시간", frames: "프레임", sources: "개 원본", datasets: "개 자료", batches: "개 묶음", steps: "단계" }[unit] || unit;
      return `${count(value)}${label}`;
    }
    function progressAmount(done, total, unit) {
      const formatted = ["bytes", "seconds"].includes(unit) ? quantity(done, unit)
        : Number.isFinite(done) && done >= 0 ? count(done) : "—";
      return `${formatted} / ${quantity(total, unit)}`;
    }
    function phaseElapsed(progress) {
      const seconds = progress?.phase_elapsed_seconds;
      return Number.isFinite(seconds) && seconds >= 1 ? ` · ${quantity(seconds, "seconds")} 경과` : "";
    }
    function historyRowView() {
      const item = node("div", undefined, "story-history-row");
      const line = node("div", undefined, "story-history-line");
      const title = node("strong"), amount = node("span"), percent = node("span", undefined, "story-history-percent");
      const bar = node("progress");
      line.append(title, amount, percent); item.append(line, bar);
      return { item, title, amount, percent, bar };
    }
    function historyRow(row, completed = false) {
      const view = historyRowView();
      updateHistoryRow(view, row);
      if (completed) {
        const indicator = node("i", undefined, "story-progress-indicator");
        updateProgressIndicator(indicator, "complete");
        view.title.append(indicator);
      }
      if (view.bar.hidden) view.bar.remove();
      return view.item;
    }
    function updateHistoryRow({ title, amount, percent, bar }, row) {
      const totalKnown = Number.isFinite(row.total) && row.total > 0, doneKnown = Number.isFinite(row.done) && row.done >= 0;
      const label = row.label || "자료";
      if (title.textContent !== label) title.textContent = label;
      amount.textContent = totalKnown ? `${doneKnown ? row.unit === "bytes" ? quantity(row.done, row.unit) : count(row.done) : "—"} / ${quantity(row.total, row.unit)}` : doneKnown ? quantity(row.done, row.unit) : "수량 기록 없음";
      const byBytes = Number.isFinite(row.bytes) && Number.isFinite(row.totalBytes) && row.totalBytes > 0;
      const fraction = row.verification ? Number.isFinite(row.verificationPercent) ? row.verificationPercent / 100 : null
        : byBytes ? row.bytes / row.totalBytes : totalKnown && doneKnown ? row.done / row.total : null;
      const checking = row.quality === "checking";
      const awaitingResult = row.quality === "waiting" || (checking && fraction >= 1);
      const percentage = checking ? Math.floor(fraction * 1000) / 10 : Math.min(100, fraction * 100);
      percent.textContent = fraction === null || awaitingResult ? ""
        : `${percentage.toLocaleString("ko-KR", { maximumFractionDigits: 1 })}%`;
      bar.hidden = fraction === null && !row.verification;
      bar.dataset.quality = row.quality || "";
      bar.dataset.active = String(Boolean(row.active));
      if (fraction !== null || row.verification) {
        bar.max = row.verification ? 100 : byBytes ? row.totalBytes : row.total;
        if (awaitingResult || fraction === null) bar.removeAttribute("value");
        else bar.value = Math.min(bar.max, row.verification ? row.verificationPercent : byBytes ? row.bytes : row.done);
        bar.setAttribute("aria-label", (row.label || "자료") + (row.verification ? " 현재 파일 무결성 검사 진행률" : row.quality ? " 품질 검사 진행률" : " 진행률") + (row.active ? " · 진행 중" : ""));
      }
    }
    const downloadGroupOrder = ["generic_negative", "zeroth", "audioset", "musan", "seoul", "fsd50k", "common_rir", "official_music", "mit_rir"];
    let downloadRowViews = new Map();
    const downloadList = node("div", undefined, "story-history-rows");
    function renderDownloadActivity(project, active, toolsComplete) {
      downloadActivity.hidden = !active && !toolsComplete;
      if (downloadActivity.hidden) return;
      const progress = project.progress || {}, stage = stageId(project);
      const defaults = { frontend: "학습·평가 실행환경을 확인하고 있습니다...", data_download: "자료 파일을 준비하고 있습니다...",
        data_review: "학습·평가 자료를 검사하고 있습니다...", data_connect: "검증한 자료를 연결하고 평가용 배경 음원을 준비하고 있습니다..." };
      const label = typeof progress.label === "string" && progress.label.trim() && !/단계/.test(progress.unit || "")
        ? progress.label.trim().slice(0, 240) : defaults[stage] || "자료 준비 상태를 확인하고 있습니다...";
      const numeric = Number.isFinite(progress.done) && progress.done >= 0 && Number.isFinite(progress.total) && progress.total > 0
        && !/단계|bytes|바이트/.test(progress.unit || "");
      const text = toolsComplete ? "학습 도구 준비를 마쳤습니다." : label + (numeric ? ` · ${progressAmount(progress.done, progress.total, progress.unit || "개")}` : "")
        + (stage === "frontend" ? phaseElapsed(progress) : "");
      downloadActivityLabel.textContent = text + " "; downloadActivity.title = text;
      updateProgressIndicator(downloadActivityIndicator, toolsComplete ? "complete" : "active");
    }
    function renderStoryDownloads(files, verification, quality, activeDownload) {
      const groups = new Map(), complete = file => ["complete", "reused"].includes(file.state);
      const fileActive = file => activeDownload && ["downloading", "verifying", "retrying"].includes(file.state);
      const sum = (files, key) => files.every(file => Number.isFinite(file[key])) ? files.reduce((value, file) => value + file[key], 0) : null;
      const rank = key => downloadGroupOrder.includes(key) ? downloadGroupOrder.indexOf(key) : downloadGroupOrder.length;
      for (const file of files) {
        const key = file.group || file.id;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(file);
      }
      const rows = [...groups].sort(([a], [b]) => rank(a) - rank(b)).flatMap(([key, group]) => {
        if (group.some(file => (file.unit || "bytes") !== "bytes")) {
          return group.map(file => ({ ...file, key: file.id, label: file.label || file.group_label || "자료",
            active: fileActive(file), completed: complete(file) }));
        }
        return [{ key, label: group[0].group_label || group[0].label || "자료", done: group.filter(complete).length,
          total: group.length, unit: "files", bytes: sum(group, "done"), totalBytes: sum(group, "total"),
          failed: group.some(file => file.state === "failed"),
          active: group.some(fileActive),
          completed: group.every(complete),
          verifying: group.some(file => file.state === "verifying"),
          hashPercent: group.filter(file => file.state === "verifying").length === 1
            && group.some(file => file.id === verification?.id && file.state === "verifying") ? verification.percent : null,
          reused: group.filter(file => file.state === "reused").length,
          cacheCandidate: group.some(file => file.cache_candidate && file.state === "waiting") }];
      });
      const active = new Map(), items = rows.map(row => {
        let view = downloadRowViews.get(row.key);
        if (!view) {
          view = historyRowView();
          view.titleIndicator = node("i", undefined, "story-progress-indicator");
          view.note = node("p", undefined, "story-history-note");
          view.noteDetail = node("span", undefined, "story-history-note-detail");
          view.inspectionStatus = node("span", undefined, "story-quality-status");
          view.inspectionIndicator = node("i", undefined, "story-progress-indicator story-quality-indicator");
          view.inspectionLabel = node("span", undefined, "story-quality-label");
          view.inspectionStatus.append(view.inspectionIndicator, view.inspectionLabel);
          view.note.append(view.noteDetail, view.inspectionStatus);
          view.item.append(view.note);
        }
        const inspection = (row.group || row.key) === "generic_negative" && row.completed ? quality : null;
        const gauge = inspection && Number.isFinite(inspection.total)
          ? { ...row, done: inspection.done, total: inspection.total, unit: "rows", bytes: null, totalBytes: null }
          : row.verifying || row.state === "verifying" ? { ...row, verification: true, verificationPercent: row.hashPercent } : row;
        const inspectionActive = Boolean(inspection && inspection.state !== "complete");
        updateHistoryRow(view, { ...gauge, active: inspection ? inspectionActive : row.active, quality: inspection?.state });
        if (view.titleIndicator.parentElement !== view.title) view.title.append(view.titleIndicator);
        updateProgressIndicator(view.titleIndicator, inspection ? "" : progressStatus(row.active, row.completed));
        const sizeNote = row.failed || row.state === "failed" ? "다운로드 실패"
          : Number.isFinite(row.totalBytes) ? `${quantity(row.bytes, "bytes")} / ${quantity(row.totalBytes, "bytes")}`
            : row.unit === "files" && Object.hasOwn(row, "totalBytes")
              ? Number.isFinite(row.bytes) && row.bytes > 0 ? `${quantity(row.bytes, "bytes")} / —` : "전체 용량 확인 전"
              : "";
        const checkPercent = Number.isFinite(row.hashPercent) ? ` · ${row.hashPercent}%` : "";
        const reuseNote = row.verifying || row.state === "verifying" ? "파일 무결성 확인 중" + checkPercent
          : row.state === "reused" || (row.reused > 0 && row.reused === row.total) ? "기존 자료 재사용"
          : row.reused > 0 ? `기존 자료 ${count(row.reused)}개 재사용` : row.cacheCandidate ? "기존 파일 확인 대기" : "";
        const detail = [sizeNote, reuseNote].filter(Boolean).join(" · ");
        view.noteDetail.textContent = detail && inspection ? detail + " · " : detail;
        view.inspectionStatus.hidden = !inspection;
        updateProgressIndicator(view.inspectionIndicator, progressStatus(inspectionActive, inspection?.state === "complete"));
        view.inspectionLabel.textContent = inspection ? inspection.state === "complete" ? "검사 완료" : "품질 검사 중" : "";
        view.note.hidden = !detail && !inspection;
        view.item.hidden = false;
        active.set(row.key, view);
        return view.item;
      });
      downloadRowViews = active;
      if (historyContent.firstElementChild !== downloadList) historyContent.replaceChildren(downloadList);
      if (downloadList.children.length !== items.length || items.some((item, i) => downloadList.children[i] !== item)) {
        downloadList.replaceChildren(...items);
      }
      paginateHistoryRows(downloadList, items, innerWidth > 600 ? 2 : 1);
    }
    function historyPlayerActive() {
      return !savedReview.hidden && !historic.hidden && !live.hidden && !transitioning
        && pronunciationSteps.has(current) && !document.hidden;
    }
    function historyControls() {
      savedReviewPrevious.disabled = historyReviewIndex <= 0;
      savedReviewNext.disabled = historyReviewIndex >= historyReviewItems.length - 1;
      const item = historyReviewItems[historyReviewIndex];
      savedReviewToggle.disabled = !bridge.mediaUrl?.(item?.audio_url || item?.audio);
      savedReviewVolume.disabled = savedReviewToggle.disabled;
      const playing = historyPlaybackWanted || savedReviewAudio.paused === false;
      savedReviewToggle.textContent = playing ? "일시정지" : "재생";
      savedReviewToggle.setAttribute("aria-pressed", String(playing));
    }
    function stopHistoryPlayback(release = false) {
      historyPlaybackWanted = false; historyPlaybackVersion++; savedReviewAudio.pause?.();
      if (release) {
        savedReviewPicker?.close();
        savedReviewAudio.removeAttribute("src"); savedReviewAudio.load?.();
      }
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
    function historyReviewBadge(item) {
      const labels = callReviewSteps.has(current) ? { keep: "적합", discard: "부적합", review: "다시 확인" } : savedReviewLabels;
      return item.saved_review === false ? "미검수" : labels[item.decision] || "기록 없음";
    }
    function renderHistoryReviewList(keepPage = false) {
      const key = JSON.stringify([lastProject?.id, current, historyReviewItems.map(item => [item.id, item.text, item.label, historyReviewBadge(item), item.voice?.fullLabel])]);
      if (key !== historyReviewListKey) {
        historyReviewListKey = key;
        historyReviewRows = historyReviewItems.map((item, i) => {
          const row = button(undefined, "voice-row", () => selectHistoryReview(i));
          const label = item.voice?.fullLabel || item.text || item.label || `음원 ${i + 1}`;
          const badge = historyReviewBadge(item);
          row.dataset.number = String(i + 1);
          row.append(node("span", label, "listen-row-title"), node("span", badge, "listen-badge"));
          const accessibleLabel = callReviewSteps.has(current) ? label : `음원 ${i + 1} · ${label}`;
          row.setAttribute("aria-label", `${accessibleLabel} · ${badge}`);
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
      const label = item.voice?.fullLabel || item.text || item.label || `음원 ${historyReviewIndex + 1}`;
      savedReviewText.textContent = label;
      savedReviewPosition.textContent = `${historyReviewIndex + 1} / ${historyReviewItems.length}`;
      savedReviewSelect.value = String(historyReviewIndex);
      const source = bridge.mediaUrl?.(item.audio_url || item.audio) || "";
      if (source) savedReviewAudio.setAttribute("src", source); else savedReviewAudio.removeAttribute("src");
      savedReviewAudio.hidden = !source; savedReviewAudio.load?.();
      const missingReview = item.saved_review === false;
      savedReviewDecisions.hidden = missingReview || callReviewSteps.has(current);
      for (const radio of savedReviewInputs) { radio.checked = !missingReview && radio.value === item.decision; radio.parentElement.setAttribute("data-checked", String(radio.checked));
        radio.parentElement.children[1].textContent = callReviewSteps.has(current) ? (radio.value === "keep" ? "적합" : "부적합") : savedReviewLabels[radio.value]; }
      savedReviewNote.textContent = [
        !source ? "저장된 음원을 찾을 수 없습니다." : "",
        missingReview ? "저장된 발음 검수 기록이 없습니다."
          : item.decision === "review" ? "저장된 판정 · 다시 확인" : !["keep", "discard"].includes(item.decision) ? "판정 기록이 없습니다." : "",
      ].filter(Boolean).join(" ");
      if (savedReviewPickerContainer.hidden) renderHistoryReviewList();
      else savedReviewPicker.setSelected(historyReviewIndex);
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
      if (!savedReviewAudio.ended || !historyPlaybackWanted || !historyPlayerActive()) return;
      if (historyReviewIndex + 1 < historyReviewItems.length) return selectHistoryReview(historyReviewIndex + 1, true);
      stopHistoryPlayback();
    });
    savedReviewAudio.addEventListener("error", () => {
      if (savedReviewAudio.getAttribute("src")) { stopHistoryPlayback(); savedReviewNote.textContent = "음원을 재생하지 못했습니다."; }
    });
    document.addEventListener?.("visibilitychange", () => { if (document.hidden) stopHistoryPlayback(true); });
    function historyHeight() {
      if (!stage.getBoundingClientRect || typeof getComputedStyle !== "function") return Math.max(80, innerHeight - 210);
      const rect = stage.getBoundingClientRect(), style = getComputedStyle(stage);
      const gap = parseFloat(getComputedStyle(panel).rowGap) || 0;
      let available = rect.height - (parseFloat(style.paddingTop) || 0) - (parseFloat(style.paddingBottom) || 0)
        - header.getBoundingClientRect().height - footer.getBoundingClientRect().height - gap * 2;
      const liveGap = parseFloat(getComputedStyle(live).rowGap) || 0;
      for (const extra of [listenActions, migrationNote, evaluationPlan, review, input, liveSummary, downloadActivity]) {
        if (!extra.hidden) available -= (extra.getBoundingClientRect?.().height || parseFloat(extra.style.minHeight) || 0) + liveGap;
      }
      return Math.max(80, available);
    }
    function renderHistory(project) {
      const view = project?.completed_stage_views?.[current];
      historic.dataset.kind = current;
      const valid = completedStep(project, current) && view?.completed === true;
      const downloadFiles = current === "downloads" ? valid ? view.progress?.files || view.files || project.download_files
        : project.download_files || project.progress?.files : null;
      const activeDownload = Boolean(downloadFiles?.length && stageId(project) === "data_download"
        && chapterWorkActive(project, shell.dataset.history === "true"));
      const event = !valid && downloadFiles?.length ? project.progress?.current_event : null;
      const verification = event?.phase === "hash" && Number.isFinite(event.done) && event.done >= 0
        && Number.isFinite(event.total) && event.total > 0
        ? { id: project.progress.current_asset, percent: Math.min(100, Math.floor(event.done / event.total * 100)) } : null;
      const progress = project?.progress;
      let quality = downloadFiles?.length ? project.download_quality : null;
      if (!quality && downloadFiles?.length && !valid && stageId(project) === "data_download" && !project.error) {
        quality = progress?.stage === "data_download" && progress.phase === "quality" && progress.unit === "rows"
          ? { state: "checking", done: progress.done, total: progress.total } : { state: "waiting" };
      }
      const downloadSpace = downloadFiles?.length ? [project?.error, Boolean(project?.review), Boolean(project?.input), downloadActivity.hidden] : null;
      const fingerprint = JSON.stringify([project?.id, current, valid, view, downloadFiles, verification, quality, activeDownload, downloadSpace, project?.migration, historyPage, innerWidth, innerHeight]);
      if (fingerprint === historyKey) return;
      historyKey = fingerprint; savedReview.hidden = true; historyPages.hidden = true;
      const unknown = () => historyContent.replaceChildren(node("p", "저장된 세부 내역이 없습니다.", "story-history-note"));
      if (downloadFiles?.length) { stopHistoryPlayback(true); renderStoryDownloads(downloadFiles, verification, quality, activeDownload); return; }
      if (!valid) { stopHistoryPlayback(true); unknown(); return; }
      const reviewItems = Array.isArray(view.review?.items) ? view.review.items : [];
      if (pronunciationSteps.has(current) && reviewItems.length) {
        const queueKey = JSON.stringify([project.id, current, reviewItems]);
        const sameQueue = historyReviewQueueKey === queueKey;
        historyReviewQueueKey = queueKey;
        const compact = Boolean(savedReviewPicker && savedReviewTransport);
        const wasCompact = !savedReviewPickerContainer.hidden;
        const layoutChanged = compact !== wasCompact;
        savedReviewWorkspace.classList.toggle("listen-voice-compact", compact);
        savedReviewSidebar.hidden = compact;
        savedReviewPickerContainer.hidden = !compact;
        savedReviewText.hidden = compact;
        savedReviewPrevious.hidden = compact;
        savedReviewNext.hidden = compact;
        savedReviewVolume.hidden = !compact;
        const groupCounts = new Map([["학습용", 0], ["공통 평가용", 0]]);
        historyReviewItems = callReviewSteps.has(current) ? reviewItems.map((item, i) => {
          if (typeof item.label === "string" && item.label.trim())
            return { ...item, voice: { label: item.label, fullLabel: item.label } };
          const grouped = groupCounts.has(item.group);
          const groupLabel = grouped ? item.group : "목소리";
          const position = grouped ? groupCounts.get(groupLabel) + 1 : i + 1;
          if (grouped) groupCounts.set(groupLabel, position);
          const voiceNumber = Number.isSafeInteger(item.voice_number) && item.voice_number > 0 ? item.voice_number : position;
          const fullLabel = grouped ? `${groupLabel} · 목소리 ${voiceNumber}` : `목소리 ${voiceNumber}`;
          const label = grouped ? `${groupLabel} · ${voiceNumber}` : fullLabel;
          return { ...item, voice: { label, fullLabel, groupLabel, voiceNumber } };
        }) : reviewItems;
        savedReviewCompletion.hidden = compact || callReviewSteps.has(current);
        if (!sameQueue) historyReviewIndex = 0;
        savedReview.hidden = false; historyContent.replaceChildren();
        if (compact && (!sameQueue || layoutChanged)) {
          savedReviewPicker.setItems(historyReviewItems.map((item, i) => {
            const name = item.text || item.label || `음원 ${i + 1}`;
            const entry = item.voice || { label: name, fullLabel: `유사어 음원 ${i + 1} · ${name}` };
            return { ...entry, badge: historyReviewBadge(item) };
          }));
        }
        if (!sameQueue || savedReviewSelect.children.length !== reviewItems.length) {
          savedReviewSelect.replaceChildren(...historyReviewItems.map((item, i) => {
            const label = item.voice && groupCounts.has(item.group) ? item.voice.fullLabel : `${i + 1}. ${item.text || item.label || "음원"}`;
            const option = node("option", label); option.value = String(i); return option;
          }));
          updateHistoryReview();
        } else if (layoutChanged || !savedReviewAudio.getAttribute("src")) updateHistoryReview();
        return;
      }
      stopHistoryPlayback(true);
      if (current === "phrases") {
        const phrases = Array.isArray(view.phrases) ? view.phrases.filter(value => typeof value === "string") : [];
        if (!phrases.length) { unknown(); return; }
        const size = pageCapacity(phrases.length, historyHeight(), 44, innerWidth > 600 ? 2 : 1, 10), pages = Math.ceil(phrases.length / size);
        historyPage = Math.min(historyPage, pages - 1);
        const grid = node("div", undefined, "story-saved-phrases story-phrase-grid");
        phrases.slice(historyPage * size, (historyPage + 1) * size).forEach((text, i) => {
          const field = node("input"); field.type = "text"; field.readOnly = true; field.value = text;
          field.setAttribute("aria-label", `저장된 유사어 ${historyPage * size + i + 1}`); grid.append(field);
        });
        historyContent.replaceChildren(grid); renderHistoryPages(pages); return;
      }
      let rows = Array.isArray(view.rows) ? view.rows : [];
      if (!rows.length && view.progress) rows = [view.progress];
      if (!rows.length) { unknown(); return; }
      // Measure real row heights at the rendered width before choosing a
      // page size. A tall narrow viewport still has a single-column list.
      const workLayout = Boolean(workLabels[current]);
      const list = node("div", undefined, workLayout ? "story-history-rows story-work-history-rows" : "story-history-rows");
      const allItems = rows.map(row => {
        const workView = workLayout ? workProgressRow() : null;
        if (workView) {
          updateWorkProgressRow(workView, { ...row, completed: true });
          if (workView.bar.hidden) workView.bar.remove();
        }
        const item = workView?.row || historyRow(row, current === "downloads");
        if (Number.isFinite(row.bytes) || Number.isFinite(row.totalBytes)) item.append(node("p", Number.isFinite(row.totalBytes) ? `${quantity(row.bytes, "bytes")} / ${quantity(row.totalBytes, "bytes")}` : quantity(row.bytes, "bytes"), "story-history-note"));
        return item;
      });
      list.append(...allItems); historyContent.replaceChildren(list);
      paginateHistoryRows(list, allItems, current === "downloads" && innerWidth > 600 ? 2 : 1);
    }
    function paginateHistoryRows(list, allItems, columns) {
      if (current === "downloads") {
        const available = historyHeight(), rowCount = Math.ceil(allItems.length / columns);
        const budget = rowCount ? available / rowCount : available;
        const densities = budget >= 82 ? ["normal", "compact", "tight"] : budget >= 64 ? ["compact", "tight"] : ["tight"];
        historyPage = 0; historyPages.hidden = true;
        allItems.forEach(item => { item.hidden = false; });
        list.style.setProperty("--download-list-height", `${Math.floor(available)}px`);
        let required = 0;
        for (const density of densities) {
          list.dataset.density = density;
          const style = typeof getComputedStyle === "function" ? getComputedStyle(list) : null;
          const gap = style ? parseFloat(style.rowGap) || 0 : 4;
          const rowHeight = Math.max(0, ...allItems.map(item => item.getBoundingClientRect?.().height || 70));
          required = list.scrollHeight || rowCount * rowHeight + Math.max(0, rowCount - 1) * gap;
          if (required <= available) break;
        }
        list.dataset.overflow = String(required > available);
        return;
      }
      const style = typeof getComputedStyle === "function" ? getComputedStyle(list) : null;
      const gap = style ? parseFloat(style.rowGap) || 0 : 18;
      const rowHeight = Math.max(...allItems.map(item => item.getBoundingClientRect?.().height || 70));
      const size = pageCapacity(allItems.length, historyHeight(), rowHeight, columns, gap), pages = Math.ceil(allItems.length / size);
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
      downloadActivity.hidden = true;
      const key = project?.id || "setup";
      const nextScreens = displayOrder(project);
      if (nextScreens.length !== screens.length || nextScreens.some((id, position) => id !== screens[position])) {
        screens = nextScreens;
        index.replaceChildren(...screens.map(id => ticks.get(id)));
        index.style.setProperty("--story-index-count", screens.length);
      }
      if (key !== projectKey) {
        hardConnectionIntent = null;
        hardReviewArrivalPending = false; hardReviewRevealFailed = false;
        const createdHere = transitionSession?.createdProject === key && Boolean(key) && key !== "setup";
        pauseMedia(); if (!createdHere) stopTransition(); projectKey = key; following = true; materialKey = "";
        referenceReviewComplete = false; clearReferenceGroup(); clearReferenceHeight(); referenceMode = null;
        repairGenerationSeen = ""; repairReviewStaged = ""; repairReviewConcealed = ""; repairRevealFailed = "";
        voiceListening.classList.remove("candidate-review");
        shell.classList.remove("candidate-reference-review");
        maximum = frontier(project); current = project ? stageStep(project) : "entry"; lastFrontier = project ? stageStep(project) : "entry";
        if (!project) setupBackend = null;
        historyPage = 0; historyKey = ""; setupReached = 0;
      }
      syncReferenceFrame(); syncPreflightFrame();
      resumeFinalizedPreflight();
      resumeSavedBoundary();
      const repair = postRepair(project), repairedQueue = repairQueueKey(project);
      const repairKey = repair ? `${project.id}|${repair.round ?? repair.source_queue_sha256 ?? "repair"}` : "";
      if (repairReviewConcealed && repairReviewConcealed !== repairedQueue) { repairReviewConcealed = ""; repairRevealFailed = ""; }
      if (current === "pronunciation" && repair && !project.review) repairGenerationSeen = repairKey;
      if (current === "pronunciation" && repair && project.review) {
        if (transitioning && !repairReviewConcealed) repairReviewStaged = repairedQueue;
        else if (!transitioning && !state.busy && repairGenerationSeen === repairKey && repairReviewStaged !== repairedQueue) revealRepairReview(project);
      }
      lastProject = project; maximum = frontier(project);
      if (hardConnectionIntent) {
        if (hardConnectionIntent.project !== project?.id || project.error || project.review || project.input
          || ["failed", "error", "interrupted", "cancelled", "stopped"].includes(project.status)) hardConnectionIntent = null;
        else if (!transitioning && !state.busy && current === "hard-review" && project.status === "paused"
          && navigationCheckpoint(project)?.step === "hard-review" && navigationCheckpoint(project)?.state === "awaiting_next"
          && !project.process_alive && completedNavigationStep(project, "hard-review")) {
          hardConnectionIntent = null; following = true;
          void Promise.resolve().then(() => bridge.action("resume")).catch(error => bridge.message?.(error.message, true));
        }
      }
      shell.classList.toggle("merged-reference-review", mergedReferenceReview(project));
      shell.classList.toggle("separate-reference-preparation", separateReferencePreparation(project));
      bridge.voiceDesign?.(project ? project.reference_design : state.recipes.find(recipe => recipe.id === state.selectedPreset)?.reference_design);
      const actual = project ? stageStep(project) : current;
      if (project && actual !== lastFrontier) {
        if (!following || current === actual) lastFrontier = actual;
        else if (!transitioning && !state.busy && navigate(screens.indexOf(actual), true)) lastFrontier = actual;
      }
      if (screens.indexOf(current) > maximum) current = actual;
      const reviewStage = stageId(project);
      if (current === "hard-review" && sampleFirstHardReview(project) && reviewStage === "hard_pilot") hardReviewArrivalPending = true;
      else if (hardReviewArrivalPending) {
        hardReviewArrivalPending = false;
        if (current === "hard-review" && !transitioning && (preparingReviewQueue(project) || reviewingPreflight(project)))
          revealHardReview();
      }
      if (reviewStage !== "hard_review") hardReviewRevealFailed = false;
      const checkpoint = downloadCheckpoint(project), navigation = navigationCheckpoint(project);
      const preparationPending = current === "voices" && referencePreparationNeeded(project) && project.reference_preparation.ready !== true;
      const liveTools = Boolean(project && current === "tools" && actual === current);
      const preparingTools = current === "downloads" && stageId(project) === "frontend";
      const position = screens.indexOf(current), historical = Boolean(!preparationPending && project && (position < maximum && !liveTools || done(project.status)
        || (current === "downloads" && checkpoint?.state === "awaiting_next" && completedStep(project, "downloads"))
        || (navigation?.state === "awaiting_next" && current === navigation.step && completedNavigationStep(project, current))));
      shell.dataset.step = current; shell.dataset.model = project ? "existing" : "new";
      shell.dataset.layout = standardLayoutSteps.has(current) ? "standard" : "";
      shell.dataset.history = String(historical);
      shell.classList.toggle("post-repair-generating", repairProgressVisible(project));
      shell.classList.toggle("hard-connecting", connectingHard(project));
      const preparingResults = Boolean(project && current === "results" && !done(project.status) && stageStep(project) === "results");
      shell.classList.toggle("preparing-results", preparingResults);
      const definition = definitions.find(([id]) => id === current);
      const [, title = "학습 상태", description = "현재 작업 상태를 확인해 주세요."] = definition || [];
      const needsToolChoice = waitingToolChoice(project);
      heading.textContent = title;
      hardGenerationSummary.hidden = true; hardGenerationSummary.textContent = "";
      const savedReviewDescription = historical && pronunciationSteps.has(current);
      subtitle.textContent = savedReviewDescription ? "대표 음원과 저장된 판정을 확인합니다." : description;
      if (current === "tools" && (needsToolChoice || !project)) subtitle.textContent = "음성 생성과 모델 학습에 사용할 연산 장치를 선택하세요.";
      if (current === "voices" && separateReferencePreparation(project)) subtitle.textContent = "성별·연령별로 검수할 기준 목소리 후보를 미리 생성합니다.";
      if (current === "voice-listening" && !project?.reference_preview?.review_required) {
        subtitle.textContent = "생성 설정이 비슷한 목소리를 묶음별로 들어보세요.";
      }
      if (twoPhasePositiveReview(project)) {
        if (current === "positive-generation") subtitle.textContent = "사전 검수에서 승인한 기준 목소리로 전체 호출어 음성을 생성합니다.";
        if (current === "pronunciation" && !historical) subtitle.textContent = "대표 음원을 연속으로 듣고, 문제가 있는 음원만 부적합으로 표시하세요.\n끝까지 들은 음원은 적합 판정하며, 다음을 누르면 저장하고 승인을 확인합니다.";
        if (current === "pronunciation" && repair) {
          const retained = Number.isSafeInteger(repair.retained_count) && repair.retained_count >= 0 ? repair.retained_count : 0;
          const count = Number.isSafeInteger(repair.recheck_count) && repair.recheck_count > 0 ? repair.recheck_count : null;
          subtitle.textContent = repairProgressVisible(project)
            ? `적합 판정 ${retained}개는 보존하고, 부적합한 목소리의 호출어만 다시 생성합니다.`
            : `적합 판정 ${retained}개는 보존했습니다. 다시 생성한 ${count ?? "음원"}${count === null ? "" : "개"}만 끝까지 듣고 검수하세요.`;
        }
      }
      if (preparingResults) subtitle.textContent = "학습 결과를 정리하고 모델별 감지 설정을 비교합니다.";
      if (connectingHard(project)) subtitle.textContent = "검수한 유사어를 학습 자료에 연결하고 원본의 무결성을 확인하고 있습니다.";
      document.title = definition ? title : "한국어 호출어 학습";
      logButton.hidden = !project;
      if (!project) setupReached = Math.max(setupReached, position);
      const visibleThrough = project ? maximum : setupReached;
      index.hidden = visibleThrough === 0;
      screens.forEach((id, i) => {
        const tick = ticks.get(id);
        tick.hidden = i > visibleThrough; tick.disabled = state.busy || i > maximum;
        tick.setAttribute("aria-current", i === position ? "step" : "false");
      });
      updateIndexDistances();
      if (!transitioning) {
        back.hidden = position === 0;
        const backDisabled = Boolean(state.busy || position === 0);
        if (back.disabled !== backDisabled) back.disabled = backDisabled;
        next.hidden = current === "microphone";
        let nextDisabled = Boolean(!project
          ? state.busy || !state.recipeReady || (current === "materials" && $("create").disabled)
          : state.busy || (current === stageStep(project) && humanGate(project)) || (position >= maximum && !releaseStep(project)));
        if (reviewingReferenceGroup(project)) nextDisabled = Boolean(state.busy || !referenceGroupReady(project));
        if (reviewingPreflight(project)) nextDisabled = Boolean(state.busy || !preflightReady(project));
        if (current === "hard-review" && hardReviewRevealFailed) nextDisabled = true;
        if (editingPhrases(project)) nextDisabled = Boolean(state.busy || phraseInputPending);
        if (needsToolChoice && (current === "tools" || current === "materials")) nextDisabled = Boolean(state.busy);
        nextDisabled = nextDisabled || Boolean(referenceApprovalContext);
        if (next.disabled !== nextDisabled) next.disabled = nextDisabled;
        if (next.textContent !== "다음") next.textContent = "다음";
        if (next.title) next.title = "";
      }
      const action = operationAction(project);
      operation.hidden = !action;
      const preparationPausing = current === "voices" && referencePreparationNeeded(project) && project.reference_preparation.status === "pausing";
      const operationDisabled = Boolean(!action || state.busy || transitioning || project?.status === "pausing" || preparationPausing);
      if (operation.disabled !== operationDisabled) operation.disabled = operationDisabled;
      const startsToolsAndDownloads = preparingTools && action === "start";
      const operationText = project?.status === "pausing" || preparationPausing ? "정지 대기" : ["pause", "prepare_pause"].includes(action) ? "일시정지" : ["resume", "prepare_resume"].includes(action) ? "재개"
        : action === "prepare_start" ? "생성 시작"
        : ["load_review", "load_hard_review"].includes(action) ? "다시 열기"
        : startsToolsAndDownloads ? "설치·다운로드 시작"
        : current === "downloads" ? "다운로드 시작" : "시작";
      if (operation.textContent !== operationText) operation.textContent = operationText;
      const operationTitle = action === "pause" ? "현재 단계가 끝난 뒤 멈춥니다." : action === "prepare_pause" ? "진행 중인 후보 생성 작업을 일시정지합니다." : "";
      if (operation.title !== operationTitle) operation.title = operationTitle;
      footer.classList.toggle("has-operation", !operation.hidden);
      footer.classList.toggle("has-tool-start", startsToolsAndDownloads);
      migrationNote.hidden = !project?.migration || project.process_alive === true;
      // Hidden panels stay connected; polling never rebuilds their inputs or
      // embedded documents, and never resets a review's listening position.
      const pickerBackend = needsToolChoice ? ($("dependency-cu128").checked ? "cu128" : "cpu")
        : project ? project.dependency_backend : draftBackend(state);
      const showToolPicker = current === "tools" && !liveTools && toolInputs.has(pickerBackend);
      const wanted = showToolPicker ? toolPicker : current === "entry" ? (project ? savedEntry : entry) : current === "scale" ? (project ? savedScale : scale)
        : current === "materials" ? materials : current === "voice-listening" ? voiceListening : live;
      for (const unit of [entry, savedEntry, toolPicker, scale, savedScale, materials, live, voiceListening]) unit.hidden = unit !== wanted;
      if (showToolPicker) {
        toolPicker.disabled = Boolean(project || state.busy);
        for (const [backend, radio] of toolInputs) radio.checked = backend === pickerBackend;
      }
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
        review.hidden = historical || frameStep || !project.review || running(project.status) || repairProgressVisible(project)
          || current === "hard-review" && hardReviewRevealFailed;
        const dependencySetup = project.input?.kind === "dependency_setup";
        input.hidden = historical || frameStep || !project.input || running(project.status) || dependencySetup && current !== "tools";
        const hasDownloadRows = current === "downloads" && !preparingTools
          && Boolean(project.download_files?.length || project.progress?.files?.length);
        const chapterComplete = completedStep(project, current);
        const chapterActive = !chapterComplete && chapterWorkActive(project, historical);
        renderDownloadActivity(project, (current === "tools" || preparingTools || hasDownloadRows) && chapterActive && !project.error,
          current === "tools" && chapterComplete);
        liveSummary.hidden = historical || frameStep || repairProgressVisible(project) && !project.error
          || preparingResults && !project.error
          || connectingHard(project) && !project.error
          || editingPhrases(project) && !project.error
          || ((dependencySetup && !input.hidden || !downloadActivity.hidden || hasDownloadRows
            || (["call-pilot-review", "hard-review"].includes(current) && !review.hidden)) && !project.error);
        currentTitle.hidden = true;
        const preparationError = current === "voices" && separateReferencePreparation(project) && typeof project.reference_preparation?.error === "string"
          ? project.reference_preparation.error : "";
        const repairError = repairProgressVisible(project) && typeof repair?.error === "string" ? repair.error : "";
        if (preparationError || repairError || preparationErrorShown) {
          const errorText = preparationError || repairError || project.error || "";
          $("project-error").textContent = errorText; $("project-error").hidden = !errorText;
          preparationErrorShown = Boolean(preparationError || repairError);
        }
        downloads.hidden = true;
        $("progress").dataset.active = String(chapterActive);
        $("progress-label").dataset.status = progressStatus(chapterActive, chapterComplete);
        $("progress").setAttribute("aria-label", chapterComplete ? "현재 작업 진행률 · 완료" : "현재 작업 진행률");
        const rows = workRows(project, historical, chapterActive, chapterComplete);
        const generation = generationSteps.has(current) || current === "hard-review" && stageId(project) === "hard_pilot"
          || current === "pronunciation" && stageId(project) === "pilot";
        $("eta").hidden = chapterActive && project.status !== "pausing" && (generation && (project.progress?.eta_eligible === false
          || current === "voices" && separateReferencePreparation(project) && rows.every(row => row.completed))
          || current === "training" && Boolean(project.training_preparation));
        if (current === "hard-generation") {
          const phrases = project.completed_stage_views?.phrases?.phrases;
          const phraseCount = Array.isArray(phrases) ? phrases.length : 0;
          const perPhrase = project.settings?.hard_samples_per_phrase;
          const savedTotal = rows.length === 1 ? rows[0].total : null;
          const knownTotal = Number.isSafeInteger(savedTotal) && savedTotal > 0;
          const knownPerPhrase = Number.isSafeInteger(perPhrase) && perPhrase > 0;
          const total = knownTotal ? savedTotal : !sampleFirstHardReview(project) && knownPerPhrase ? phraseCount * perPhrase : null;
          const generationCount = knownTotal && knownPerPhrase && total % perPhrase === 0 ? total / perPhrase : phraseCount;
          if (generationCount > 0 && Number.isSafeInteger(total) && total > 0
              && (!knownPerPhrase || total === generationCount * perPhrase)) {
            hardGenerationSummary.textContent = `유사어 ${count(generationCount)}종${knownPerPhrase ? ` · 문구당 ${count(perPhrase)}개` : ""} · 총 ${count(total)}개`;
            hardGenerationSummary.hidden = false;
          }
        }
        voiceCounts.hidden = rows.length === 0;
        shell.dataset.workRows = String(!voiceCounts.hidden);
        historic.hidden = !(historical || hasDownloadRows) || frameStep || !voiceCounts.hidden || ["tools", "voices", "entry", "scale", "materials"].includes(current);
        hardButton.hidden = current !== "hard-review" || !project.hard_preview?.url
          || project.hard_preview.available === false || !(project.hard_preview.count > 0);
        positiveButton.hidden = repairProgressVisible(project) || current !== "pronunciation" || !project.positive_preview?.url
          || project.positive_preview.available === false || !(project.positive_preview.count > 0);
        listenActions.hidden = hardButton.hidden && positiveButton.hidden;
        renderEvaluationPlan(project);
        if (!historic.hidden) renderHistory(project);
        if (!voiceCounts.hidden) {
          const key = JSON.stringify([project.id, current, rows.length]);
          if (key !== workRowsKey) {
            workRowsKey = key;
            workRowViews = rows.map(workProgressRow);
            voiceCounts.replaceChildren(...workRowViews.map(view => view.row));
          }
          rows.forEach((row, index) => updateWorkProgressRow(workRowViews[index], row));
        }
        voiceListeningStatus.hidden = Boolean(project.reference_preview?.url);
        voiceListeningStatus.textContent = voiceListeningStatus.hidden ? "" : "저장된 목소리 듣기 자료가 없습니다.";
      }
      if (current === "materials") {
        locationStatus.hidden = Boolean(project || state.busy || !state.recipeReady || !$("create").disabled);
        const error = state.locationError || "";
        const feedback = locationStatus.hidden ? "" : error || "모델 저장 위치를 확인하고 있습니다.";
        if (locationFeedback.textContent !== feedback) locationFeedback.textContent = feedback;
        locationFeedback.className = error ? "story-field-error" : "";
        locationRetry.hidden = !error;
        const recipe = state.recipes.find(recipe => recipe.id === (project?.preset || state.selectedPreset));
        const values = project ? { ...recipe, ...project.settings, evaluation_policy: project.evaluation_policy || recipe?.evaluation_policy } : recipe;
        const toolPlan = project?.dependency_setup ?? state.dependencySetup;
        const toolPackages = needsToolChoice ? toolPlan?.packages?.[pickerBackend]
          : project?.dependency_packages ?? toolPlan?.packages?.[pickerBackend];
        const toolDownloadBytes = needsToolChoice ? toolPlan?.download_bytes?.[pickerBackend]
          : project?.dependency_download_bytes ?? toolPlan?.download_bytes?.[pickerBackend];
        const fingerprint = JSON.stringify([values, pickerBackend, toolPackages, toolDownloadBytes,
          !project ? [$("adv-originals").value, $("adv-generic").value, $("adv-korean").value] : null]);
        if (fingerprint !== materialKey) {
          materialKey = fingerprint;
          materialData = values ? bridge.materials(values, Boolean(project)) : null;
          if (project && materialData && project.settings) {
            const s = project.settings, training = materialData.training;
            const update = (id, value) => { const item = training.find(row => row.id === id); if (item && value !== undefined) item.amount = value; };
            update("positive", Number.isFinite(s.originals) ? `${count(s.originals)}개` : undefined);
            update("generic", Number.isFinite(s.generic_hours) ? `${count(s.generic_hours)}시간` : undefined);
            update("korean", Number.isFinite(s.korean_hours) ? `${count(s.korean_hours)}시간` : undefined);
            update("hard", Number.isFinite(s.hard_samples_per_phrase) ? `문구당 ${count(s.hard_samples_per_phrase)}개` : undefined);
          }
          renderMaterials();
          renderToolMaterials(pickerBackend, toolPackages, toolDownloadBytes);
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
        $("reference-preview").hidden = true;
        $("hard-preview").hidden = listening !== "hard";
      }
      if (transitionSession) {
        const queueKey = preparingReviewQueue(project) || reviewingPreflight(project)
          ? JSON.stringify([project.id, current, stageId(project), project.review?.queue_sha256 || ""]) : "";
        if ((transitionSession.reviewQueueKey || "") !== queueKey) {
          transitionSession.reviewQueueKey = queueKey;
          transitionSession.contentVersion = (transitionSession.contentVersion || 0) + 1;
        }
      }
      refreshChapterWaiters(transitionSession);
      checkCallReviewLayoutWaiters();
    }
    wakeword.addEventListener("compositionstart", () => { composingWakeword = true; showWakewordError(""); });
    wakeword.addEventListener("compositionend", () => { composingWakeword = false; showWakewordError(bridge.wakewordError(wakeword.value)); });
    wakeword.addEventListener("input", event => {
      if (!composingWakeword && !event?.isComposing && !wakewordFeedback.hidden) showWakewordError(bridge.wakewordError(wakeword.value));
    });
    wakeword.addEventListener("blur", () => {
      if (!composingWakeword && wakeword.value.trim()) showWakewordError(bridge.wakewordError(wakeword.value));
    });
    wakeword.addEventListener("keydown", event => {
      if (event.key === "Enter" && !composingWakeword && !event.isComposing && current === "entry") { event.preventDefault(); advance(); }
    });
    window.addEventListener("resize", () => { if (current === "materials") renderMaterials(); if (lastProject && !historic.hidden) renderHistory(lastProject); if (preflightPickerRequest) expandPreflightPicker(preflightPickerRequest); });
    window.addEventListener("pointerdown", () => {
      if (!voiceListening.classList.contains("is-selecting")) return;
      voiceListening.classList.remove("is-selecting");
      $("reference-preview-frame")?.contentWindow?.postMessage({ type: "koww:voice-picker-close" }, window.location.origin);
    });
    window.addEventListener("message", event => {
      const reviewFrame = $("review-frame"), project = bridge.state().project, message = event.data;
      const activeReview = reviewingPreflight(project), retainedFinalize = retainReviewFrame(project);
      const inactiveClose = message?.type === "koww:preflight-picker" && message.open === false
        && preflightFrameWindow === reviewFrame?.contentWindow;
      const queue = activeReview ? project.review.queue_sha256 : retainedFinalize ? preflightResumeIntent.queue
        : inactiveClose ? project?.review?.queue_sha256 : null;
      if (event.origin !== window.location?.origin || event.source !== reviewFrame?.contentWindow || !queue
        || message?.project !== project.id || message?.queue_sha256 !== queue) return;
      syncPreflightFrame();
      if (!Number.isSafeInteger(message.request) || message.request !== preflightRequest) return;
      const phase = activeReview ? callReviewRoute(project).phase : retainedFinalize ? preflightResumeIntent.route.phase : preflightFramePhase;
      if (phase === "hard" ? message.phase !== "hard" : message.phase === "hard") return;
      if (message.type === "koww:preflight-picker" && message.open === false && Number.isSafeInteger(message.picker_request) && message.picker_request > 0) {
        collapsePreflightPicker(message.picker_request, queue); return;
      }
      if (message.type === "koww:preflight-held") {
        const intent = preflightResumeIntent;
        if (!activeReview || !intent || !["post", "hard"].includes(intent.route.phase) || intent.phase !== "finalizing"
          || !preflightFinalizePending || preflightCommitted || !Number.isSafeInteger(message.held_count) || message.held_count < 1 || message.held_count > 100000
          || intent.project !== project.id || intent.queue !== message.queue_sha256 || intent.frame !== reviewFrame || intent.source !== event.source
          || intent.request !== message.request || (reviewFrame.getAttribute("src") || "") !== intent.url || !activeTransition(intent.session)) return;
        clearTimeout(intent.watchdog); intent.session.timers.delete(intent.watchdog);
        preflightFinalizePending = false; preflightResumeIntent = null;
        intent.phase = "held"; intent.target = intent.route.step; intent.resolve(true);
        return;
      }
      if (!activeReview && message.type !== "koww:preflight-finalized") {
        if (message.type === "koww:preflight-state" && message.error === true && typeof message.ready === "boolean" && typeof message.pending === "boolean") {
          preflightFinalizePending = false; cancelPreflightNext("검수 저장 응답을 확인하지 못했습니다. 상태를 확인한 뒤 다시 진행하세요."); render();
        }
        return;
      }
      if (message.type === "koww:preflight-picker" && typeof message.open === "boolean" && Number.isSafeInteger(message.picker_request) && message.picker_request > 0) {
        if (message.open) expandPreflightPicker(message.picker_request);
        return;
      }
      if (message.type === "koww:preflight-height" && Number.isSafeInteger(message.height) && message.height > 0 && message.height <= 100000) {
        if (preflightPickerRequest) return;
        const height = `${message.height}px`;
        const previousHeight = shell.style.getPropertyValue?.("--preflight-content-height") || shell.style["--preflight-content-height"];
        if (!preflightHeightReceived || previousHeight !== height) {
          if (transitionSession) transitionSession.contentVersion = (transitionSession.contentVersion || 0) + 1;
        }
        preflightHeightReceived = true;
        shell.classList.add("preflight-draft-review");shell.style.setProperty("--preflight-content-height", height);
        checkCallReviewLayoutWaiters(); return;
      }
      if (message.type === "koww:preflight-state" && typeof message.ready === "boolean" && typeof message.pending === "boolean" && typeof message.error === "boolean") {
        preflightState = { queue_sha256: message.queue_sha256, ready: message.ready && !message.pending && !message.error };
        preflightStateReceived = true;
        shell.classList.add("preflight-draft-review");
        if (message.error) { preflightFinalizePending = false; if (preflightResumeIntent?.phase === "finalizing") cancelPreflightNext("검수 저장 상태를 확인한 뒤 다시 진행하세요."); }
        else if (!preflightResumeIntent && message.ready && !message.pending) preflightFinalizePending = false;
        checkCallReviewLayoutWaiters(); render(); return;
      }
      if (message.type === "koww:preflight-finalized" && ["resume", "none", "retry", "repair"].includes(message.workflow_action) && preflightFinalizePending && !preflightCommitted) {
        const intent = preflightResumeIntent;
        if (!intent || intent.phase !== "finalizing" || intent.project !== project.id || intent.queue !== message.queue_sha256
          || intent.frame !== reviewFrame || intent.source !== event.source || intent.request !== message.request || !activeTransition(intent.session)) return;
        if (intent.route.phase === "hard" && message.workflow_action !== "resume"
          || intent.route.phase === "post" && !["resume", "repair"].includes(message.workflow_action)
          || intent.route.phase !== "post" && message.workflow_action === "repair"
          || message.workflow_action === "repair" && typeof message.repair_pending !== "boolean") {
          preflightFinalizePending = false; cancelPreflightNext("본검수 승인 상태를 확인한 뒤 다시 진행하세요."); render(); return;
        }
        if (["post", "hard"].includes(intent.route.phase) && message.workflow_action === "resume") {
          // Saving is bounded; authoritative corpus admission can legitimately
          // take longer on HDDs and is monitored through worker state instead.
          clearTimeout(intent.watchdog); intent.session.timers.delete(intent.watchdog);
        }
        preflightCommitted = true;
        intent.phase = message.workflow_action === "repair" ? "repair" : message.workflow_action === "retry" ? "retry" : "processing";
        if (message.workflow_action === "none") intent.processingStarted = true;
        void (async () => {
          await bridge.refresh?.();
          resumeFinalizedPreflight();
          render();
        })();
      }
    });
    window.addEventListener("message", event => {
      const frame = $("reference-preview-frame"), origin = window.location?.origin;
      if (!origin || event.origin !== origin || !frame?.contentWindow || event.source !== frame.contentWindow) return;
      syncReferenceFrame();
      const matchingProject = typeof event.data?.project === "string" && Boolean(lastProject?.id) && event.data.project === lastProject.id;
      if (event.data?.type === "koww:reference-layout" && matchingProject
        && ["candidate_pool_v2", "legacy"].includes(event.data.mode)) {
        const candidates = event.data.mode === "candidate_pool_v2";
        referenceMode = event.data.mode;
        if (!candidates) { clearReferenceGroup(); clearReferenceHeight(); }
        voiceListening.classList.toggle("candidate-review", candidates);
        shell.classList.toggle("candidate-reference-review", candidates);
        if (current === "voice-listening") render();
        return;
      }
      if (event.data?.type === "koww:reference-height" && matchingProject
        && event.data.mode === "candidate_pool_v2" && referenceMode === "candidate_pool_v2"
        && Number.isSafeInteger(event.data.height) && event.data.height > 0 && event.data.height <= 100000) {
        shell.style.setProperty("--reference-content-height", `${event.data.height}px`);
        return;
      }
      if (event.data?.type === "koww:reference-group" && matchingProject
        && typeof event.data.group === "string" && event.data.group.length > 0 && event.data.group.length <= 160
        && event.data.group === event.data.group.trim() && !/[\u0000-\u001f]/.test(event.data.group)
        && typeof event.data.ready === "boolean") {
        referenceMode = "candidate_pool_v2";
        referenceGroup = { group: event.data.group, ready: event.data.ready }; referenceApprovalPending = false;
        if (current === "voice-listening") render();
        return;
      }
      if (event.data?.type === "koww:reference-approved" && matchingProject) {
        const approval = referenceApprovalContext, message = event.data;
        if (!approval || approval.project !== projectKey || approval.frame !== frame || approval.source !== event.source
          || approval.url !== (frame.getAttribute("src") || "") || message.request !== approval.request || message.group !== approval.group
          || typeof message.success !== "boolean" || typeof message.complete !== "boolean") return;
        clearTimeout(approval.watchdog); referenceApprovalContext = null; referenceApprovalPending = false;
        if (current !== "voice-listening" || document.hidden || !message.success) { render(); return; }
        if (message.complete) {
          referenceReviewComplete = true; referenceGroup = null;
          const project = bridge.state().project;
          const intent = beginBoundaryNext(project, { step: "voices", sourceStage: stageId(project),
            target: screens[screens.indexOf(current) + 1], destinationStages: ["pilot", "pilot_review"] });
          intent.saved = true; resumeSavedBoundary();
        }
        render(); return;
      }
      if (event.data?.type === "koww:reference-review" && matchingProject
        && typeof event.data.complete === "boolean") {
        referenceReviewComplete = event.data.complete;
        if (referenceReviewComplete) { referenceGroup = null; if (!referenceApprovalContext) referenceApprovalPending = false; }
        if (current === "voice-listening") render();
        return;
      }
      if (event.data?.type !== "koww:voice-picker" || typeof event.data.open !== "boolean"
        || !Number.isSafeInteger(event.data.request) || event.data.request < 1) return;
      const open = event.data.open && current === "voice-listening" && !transitioning;
      voiceListening.classList.toggle("is-selecting", open);
      if (open) {
        // Resolve the expanded frame before its popup becomes visible.
        frame.getBoundingClientRect();
        frame.contentWindow.postMessage({ type: "koww:voice-picker-ready", request: event.data.request }, origin);
      }
    });
    window.addEventListener("pagehide", () => { stopTransition(); pauseMedia(); });
    return { render, retainReviewFrame, prepareInitialView, ownsDownloadRows: true };
  }
  window.KowwStory = Object.freeze({ create, definitions, stageStep, frontier, completedStep, pageCapacity, downloadCheckpoint, navigationCheckpoint });
})();
