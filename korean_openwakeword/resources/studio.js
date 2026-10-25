"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const names = { check: "동작 확인용", trial: "시험 모델용", production: "상용 모델용" };
  const steps = [
    ["data", "자료 준비"], ["references", "합성 목소리"], ["pilot", "호출어 확인"],
    ["generate", "호출어 생성"], ["hard", "유사어 준비"], ["features", "증강·특징 추출"], ["train", "모델 학습"],
    ["evaluate", "성능 평가"], ["test", "결과·시험"]
  ];
  const stepGroups = {
    data: ["data", "frontend", "data_download", "data_review", "data_connect"],
    references: ["references", "references_review"], pilot: ["pilot", "pilot_review"],
    generate: ["generate", "positives"], hard: ["hard_phrases", "hard_generation", "hard_review", "hard_connect"],
    features: ["features"], train: ["train", "training", "package"],
    evaluate: ["evaluate", "development", "evaluation"], test: ["test", "results"]
  };
  const statuses = {
    ready: "준비됨", idle: "준비됨", pending: "준비됨", created: "준비됨",
    running: "진행 중", starting: "시작 중", queued: "준비 중", pausing: "멈추는 중",
    paused: "일시정지", interrupted: "중단됨", waiting_review: "검수 대기", needs_review: "검수 대기", awaiting_review: "검수 대기",
    waiting_input: "확인 필요", needs_input: "확인 필요", awaiting_input: "확인 필요", waiting: "확인 필요",
    failed: "오류 확인 필요", error: "오류 확인 필요", complete: "완료", completed: "완료", finished: "완료"
  };
  const activeStatuses = new Set(["running", "starting", "queued", "pausing"]);
  const completeStatuses = new Set(["complete", "completed", "finished"]);
  const number = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString("ko-KR") : "—";
  function duration(value) {
    if (value === null || value === undefined || value === "" || !Number.isFinite(Number(value)) || Number(value) < 0) return "";
    const seconds = Math.floor(Number(value));
    if (seconds < 60) return `${seconds}초`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}분`;
    const hours = Math.floor(minutes / 60);
    return `${hours}시간${minutes % 60 ? " " + minutes % 60 + "분" : ""}`;
  }
  function bytes(value) {
    if (value === null || value === undefined || value === "" || !Number.isFinite(Number(value)) || Number(value) < 0) return "";
    const size = Number(value);
    if (size < 1024) return `${number(size)} B`;
    const unit = Math.min(4, Math.floor(Math.log(size) / Math.log(1024)));
    return `${(size / 1024 ** unit).toLocaleString("ko-KR", { maximumFractionDigits: 1 })} ${["B", "KiB", "MiB", "GiB", "TiB"][unit]}`;
  }
  function storageEstimate(recipe, selected = {}) {
    // This preview configures new projects. Historical workflow processing
    // remains on the server; it is not a second set of setup presets.
    if (recipe.material_acquisition_policy !== "automatic_v3") return null;
    const space = recipe.automatic_space_plan, components = recipe.automatic_source_plan?.components;
    if (!space || !Array.isArray(components)) return null;
    const fixed = [space.base_archive_bytes, space.base_working_bytes, space.future_buffer_bytes];
    const originals = selected.originals ?? recipe.originals, rounds = selected.rounds ?? recipe.rounds;
    const targets = { generic_negative: selected.genericHours ?? recipe.generic_hours,
      korean_negative: selected.koreanHours ?? recipe.korean_hours };
    const archive = space.transient_source_working_bytes;
    if (![...fixed, originals, rounds, ...Object.values(targets), archive].every(value => Number.isFinite(value) && value >= 0)) return null;
    const totals = {}, providers = new Set(["audioset_unbalanced", "mls", "fma_full", "yodas2"]);
    for (const item of components) {
      if (!providers.has(item.provider || item.id) || !Object.hasOwn(targets, item.pool)
        || !Number.isFinite(item.target_hours) || item.target_hours < 0) return null;
      totals[item.pool] = (totals[item.pool] || 0) + item.target_hours;
    }
    const rawHours = Object.keys(totals).reduce((sum, pool) => sum + targets[pool], 0);
    const seconds = rawHours * 3600;
    const reference = Math.ceil(seconds * 16000 * 2);
    const known = Math.ceil(seconds / 2) * 16 * 96 * 4 + Math.ceil(originals * rounds * 16 * 96 * 4);
    const fmaRate = space.source_forecast_bytes_per_hour?.fma_full ?? 125000000;
    if (!Number.isFinite(fmaRate) || fmaRate < 0) return null;
    const fmaPolicy = recipe.fma_acquisition_policy ?? space.fma_acquisition_policy;
    if (fmaPolicy !== undefined && fmaPolicy !== "per_track_flac_prefetch_v1") return null;
    const fmaBytes = components.reduce((sum, item) => (item.provider || item.id) === "fma_full" && totals[item.pool] > 0
      ? sum + targets[item.pool] * item.target_hours / totals[item.pool] * fmaRate : sum, 0);
    const transient = Math.ceil(Math.max(archive, fmaPolicy ? 0 : fmaBytes));
    return { basis: "wav_reference", reference, known, retained: 0, transient,
      total: fixed.reduce((sum, value) => sum + value, 0) + reference + known + transient };
  }
  function storageBudget(recipe, selected = {}) {
    const budget = recipe.storage_budget;
    const numeric = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
    const record = value => value !== null && typeof value === "object" && !Array.isArray(value);
    const keys = ["negative_hours", "negative_feature_rows", "positive_feature_rows", "tts_slots", "reference_count", "hard_originals"];
    const hard = budget?.hard_generation, pendingHard = budget?.variables?.hard_originals === null;
    const validPendingHard = record(hard) && hard.policy === "per_phrase_v1"
      && Number.isInteger(hard.samples_per_phrase) && hard.samples_per_phrase > 0
      && hard.phrase_count === null && hard.total_originals === null
      && hard.status === "pending_phrase_count" && hard.included_in_reference_subtotal === false
      && hard.tts_slots_per_phrase === hard.samples_per_phrase
      && [hard.tts_audio_bytes_per_phrase, hard.feature_bytes_per_phrase, hard.incremental_bytes_per_phrase].every(numeric)
      && hard.incremental_bytes_per_phrase === hard.tts_audio_bytes_per_phrase + hard.feature_bytes_per_phrase
      && budget.tts_slots_scope === "excluding_unselected_hard_phrases";
    const validVariables = input => keys.every(key => key === "hard_originals" && pendingHard ? input[key] === null : numeric(input[key]));
    if (!record(budget) || budget.version !== 1 || budget.complete !== false
      || budget.total_required_bytes !== null || budget.peak_required_bytes !== null
      || !numeric(budget.reference_subtotal_bytes) || !record(budget.variables)
      || (pendingHard && !validPendingHard) || !validVariables(budget.variables) || !Array.isArray(budget.rows)
      || !Array.isArray(budget.assumptions) || !budget.assumptions.every(item => typeof item === "string")
      || !Array.isArray(budget.unknowns) || !budget.unknowns.every(item => record(item)
        && typeof item.id === "string" && typeof item.label === "string" && typeof item.detail === "string")) return null;
    const values = {};
    for (const [key, field] of [["originals", "originals"], ["rounds", "rounds"], ["genericHours", "generic_hours"], ["koreanHours", "korean_hours"]]) {
      values[key] = Object.hasOwn(selected, key) ? selected[key] : recipe[field];
      if (!numeric(values[key])) return null;
    }
    if (!Number.isInteger(values.originals) || !Number.isInteger(values.rounds) || !numeric(recipe.originals)) return null;
    const components = recipe.automatic_source_plan?.components;
    if (!Array.isArray(components)) return null;
    // Published feature-only pools have no stored waveform or newly extracted
    // window rows. Only raw pools present in this recipe scale with hours.
    const rawHours = (components.some(item => item.pool === "generic_negative") ? values.genericHours : 0)
      + (components.some(item => item.pool === "korean_negative") ? values.koreanHours : 0);
    const variables = { ...budget.variables, negative_hours: rawHours,
      negative_feature_rows: Math.ceil(rawHours * 3600 / 2),
      positive_feature_rows: values.originals * values.rounds,
      tts_slots: budget.variables.tts_slots + values.originals - recipe.originals };
    if (!validVariables(variables)) return null;
    const classes = new Set(["published", "measured", "calculated", "reference", "unresolved"]), ids = new Set(), rows = [];
    let baseline = 0, subtotal = 0;
    const calculate = (terms, input) => Math.ceil(Object.entries(terms).reduce((sum, [key, coefficient]) => sum + coefficient * (key === "constant" ? 1 : input[key]), 0));
    for (const row of budget.rows) {
      if (!record(row) || typeof row.id !== "string" || ids.has(row.id) || typeof row.label !== "string"
        || typeof row.detail !== "string" || !classes.has(row.classification)) return null;
      ids.add(row.id);
      if (row.bytes === null) {
        if (row.classification !== "unresolved" || (row.terms !== undefined && row.terms !== null && (!record(row.terms) || Object.keys(row.terms).length))) return null;
        rows.push({ ...row }); continue;
      }
      if (!numeric(row.bytes) || row.classification === "unresolved" || !record(row.terms)
        || !Object.keys(row.terms).length || !Object.entries(row.terms).every(([key, value]) => (key === "constant" || keys.includes(key)) && numeric(value))) return null;
      if (pendingHard && row.terms.hard_originals) return null;
      if (Math.abs(calculate(row.terms, budget.variables) - row.bytes) > 1) return null;
      const size = calculate(row.terms, variables);
      if (!numeric(size)) return null;
      baseline += row.bytes; subtotal += size; rows.push({ ...row, bytes: size });
    }
    if (Math.abs(baseline - budget.reference_subtotal_bytes) > 1) return null;
    return { ...budget, variables, rows, reference_subtotal_bytes: subtotal };
  }
  function eta(value) {
    if (Array.isArray(value) && value.length === 2) {
      const low = duration(value[0]), high = duration(value[1]);
      return low && high ? `${low}–${high}` : "";
    }
    if (value && typeof value === "object") return eta([value.min ?? value.low, value.max ?? value.high]);
    return duration(value);
  }
  function progressValue(project) {
    const progress = project.progress;
    if (!progress || typeof progress !== "object") return null;
    const rawDone = progress.done ?? progress.completed ?? progress.current;
    if (rawDone === undefined || rawDone === null || progress.total === undefined || progress.total === null) return null;
    const done = Number(rawDone), total = Number(progress.total);
    if (!Number.isFinite(done) || !Number.isFinite(total) || done < 0 || total <= 0) return null;
    return { done, total, unit: progress.unit || "개", label: progress.label || "", percent: Math.min(100, Math.max(0, done / total * 100)) };
  }
  function evaluating(project) {
    return project && activeStatuses.has(project.status) && (project.stage_id || (typeof project.stage === "object" ? project.stage?.id : project.stage)) === "evaluation";
  }
  function requestBody(form, preset) {
    const result = { wakeword: form.wakeword.trim(), preset };
    const advanced = {};
    const integerKeys = new Set(["positive_originals", "augmentation_rounds", "updates"]);
    for (const [key, raw] of Object.entries(form.advanced || {})) {
      if (raw === "" || raw === null || raw === undefined) continue;
      const value = Number(raw);
      if (!Number.isFinite(value) || value < (integerKeys.has(key) ? 1 : 0) || (integerKeys.has(key) && !Number.isInteger(value))) throw new Error("세부 설정의 수량과 시간을 확인하세요.");
      advanced[key] = value;
    }
    if (Object.keys(advanced).length) result.advanced = advanced;
    if (!result.wakeword) throw new Error("학습할 호출어를 입력하세요.");
    return result;
  }
  function localUrl(value) {
    if (!value || typeof value !== "string") return "";
    try { const url = new URL(value, window.location.href); return url.origin === window.location.origin ? url.pathname + url.search + url.hash : ""; } catch { return ""; }
  }
  function webUrl(value) {
    if (!value || typeof value !== "string") return "";
    try { const url = new URL(value); return ["http:", "https:"].includes(url.protocol) ? url.href : ""; } catch { return ""; }
  }
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }
  const backgroundTypes = {
    rain: "비", wind: "바람", vacuum_cleaner: "청소기", washing_machine: "세탁기", engine: "엔진", train: "기차",
    keyboard_typing: "키보드", door_wood_knock: "문 두드림", glass_breaking: "유리 충격·파손", footsteps: "발걸음",
    clock_alarm: "알람", clock_tick: "시계 전자·기계음", car_horn: "경적", siren: "사이렌", church_bells: "종소리",
    music_instrumental_piano_drums: "피아노·드럼 연주", music_instrumental_guitar_drums: "기타·드럼 연주", music_singing: "보컬 멜로디", music_vocal_pad: "보컬 패드 음악",
    korean_read_speech: "한국어 낭독", korean_speech: "한국어 음성", korean_natural_conversation: "한국어 자연 대화",
    korean_conversation: "한국어 대화", korean_broadcast: "한국어 방송", korean_overlapping_speech: "겹치는 한국어 음성",
    korean_market_conversation: "시장 대화", korean_restaurant_conversation: "식당 대화", korean_subway_conversation: "지하철 대화",
    korean_street_conversation: "거리 대화", korean_transit_announcement: "교통 안내방송",
    dog: "개 짖음", cat: "고양이", chirping_birds: "새소리", crying_baby: "아기 울음",
    sneezing: "재채기", clapping: "박수", breathing: "숨소리", coughing: "기침", laughing: "웃음", snoring: "코골이",
    pouring_water: "물 붓기", toilet_flush: "변기 물 내림", water_drops: "물방울", can_opening: "캔 따기",
    mouse_click: "마우스 클릭", door_wood_creaks: "문 삐걱거림", airplane: "비행기", helicopter: "헬리콥터",
    thunderstorm: "천둥", sea_waves: "파도", music_jazz: "재즈", music_techno: "테크노", music_hiphop: "힙합",
    music_rock_vocals: "보컬 록", music_rock: "록", music_cinematic_orchestral: "영화풍 관현악",
    music_house: "하우스 음악", music_orchestral_loop: "관현악 루프"
  };
  function backgroundTypeNames(source) {
    const types = source.type_labels || source.eval_types || [];
    return Array.isArray(types) ? [...new Set(types.map(value => backgroundTypes[value] || value).filter(value => typeof value === "string" && value))] : [];
  }
  function replace(node, children) { node.replaceChildren(...children); }

  let snapshot = { projects: [] }, recipes = [], selectedId = "", selectedPreset = "check";
  let newProject = false, busy = false, refreshing = false, pollTimer = null, recipeReady = false;
  let locationTimer = null, locationRevision = 0, modelLocation = { key: "", directory: "" };
  let reviewIdentity = "", referencePreviewIdentity = "", inputIdentity = "", resultIdentity = "", resultMode = "results", logProject = "";
  let hardPhraseFields = [];
  const selectedProject = () => (snapshot.projects || []).find(project => project.id === selectedId);

  async function api(path, method = "GET", payload) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), method === "GET" ? 15000 : 60000);
    const headers = { Accept: "application/json" };
    if (payload !== undefined) headers["Content-Type"] = "application/json";
    if (snapshot.token) headers["X-Koww-Token"] = snapshot.token;
    try {
      const response = await fetch(path, { method, headers, signal: controller.signal, ...(payload === undefined ? {} : { body: JSON.stringify(payload) }) });
      const text = await response.text();
      let body;
      try { body = text ? JSON.parse(text) : {}; } catch { throw new Error(response.ok ? "서버 응답을 읽지 못했습니다." : `요청에 실패했습니다 (${response.status}).`); }
      if (!response.ok) {
        const error = new Error(body?.error?.message || body?.error || body?.message || `요청에 실패했습니다 (${response.status}).`);
        error.status = response.status;
        if (typeof body?.code === "string") error.code = body.code;
        throw error;
      }
      return body;
    } catch (error) {
      if (error.name === "AbortError") {
        const timeoutError = new Error("서버 응답이 늦어지고 있습니다. 진행 상태를 다시 확인하세요. 같은 작업을 중복 실행하지 않습니다.");
        timeoutError.code = "request_timeout";
        throw timeoutError;
      }
      throw error;
    } finally { clearTimeout(timer); }
  }
  function message(text, error = false) {
    $("message").textContent = text || "";
    $("message").hidden = !text;
    $("message").className = "notice" + (error ? " error" : "");
  }
  function creationBody() {
    return requestBody({ wakeword: $("wakeword").value, advanced: { positive_originals: $("adv-originals").value, augmentation_rounds: $("adv-rounds").value, updates: $("adv-updates").value, generic_hours: $("adv-generic").value, korean_hours: $("adv-korean").value } }, selectedPreset);
  }
  function locationReady(body) {
    try { return Boolean(modelLocation.directory) && modelLocation.key === JSON.stringify(body || creationBody()); }
    catch { return false; }
  }
  function previewLocation() {
    clearTimeout(locationTimer); locationTimer = null;
    const revision = ++locationRevision;
    modelLocation = { key: "", directory: "" };
    let body;
    try { body = creationBody(); }
    catch {
      $("storage-directory").textContent = $("wakeword").value.trim() ? "입력한 값을 확인하세요." : "호출어를 입력하면 표시됩니다.";
      render(); return;
    }
    if (!recipeReady) { render(); return; }
    $("storage-directory").textContent = "저장 위치 확인 중…";
    render();
    locationTimer = setTimeout(async () => {
      locationTimer = null;
      try {
        const result = await api("/api/model-location", "POST", body);
        if (revision !== locationRevision) return;
        if (typeof result.directory !== "string" || !result.directory.trim()) throw new Error("서버에서 저장 위치를 받지 못했습니다.");
        modelLocation = { key: JSON.stringify(body), directory: result.directory };
        $("storage-directory").textContent = result.directory;
      } catch (error) {
        if (revision !== locationRevision) return;
        $("storage-directory").textContent = "저장 위치 확인 실패: " + error.message;
      }
      render();
    }, 200);
  }
  function renderPresets() {
    const nodes = recipes.map(recipe => {
      const label = element("label", undefined, "preset");
      const radio = element("input"); radio.type = "radio"; radio.name = "preset"; radio.value = recipe.id; radio.checked = selectedPreset === recipe.id;
      radio.addEventListener("change", () => { selectedPreset = recipe.id; renderRecipe(); previewLocation(); });
      const title = element("span", recipe.label || names[recipe.id] || recipe.id, "preset-title");
      const heading = element("div"); heading.append(radio, title);
      const detail = element("p", recipe.description || "", "preset-description");
      const summary = element("p", undefined, "preset-summary");
      const originals = recipe.originals ?? recipe.positive_originals, rounds = recipe.rounds ?? recipe.augmentation_rounds, updates = recipe.updates;
      if (originals !== undefined) summary.append(element("span", `호출어 ${number(originals)}개`));
      if (rounds !== undefined) summary.append(element("span", `증강 R${number(rounds)}`));
      if (updates !== undefined) summary.append(element("span", `학습 ${number(updates)}회 / 모델`));
      if (Number.isInteger(recipe.train_references) && recipe.train_references > 0) summary.append(element("span", `학습 기준 목소리 ${number(recipe.train_references)}개`));
      label.append(heading, detail, summary); return label;
    });
    replace($("presets"), nodes.length ? nodes : [element("p", "사용 가능한 작업 규모가 없습니다. 서버 설정을 확인하세요.", "muted")]);
  }
  function originalSize(value) {
    if (!Number.isFinite(value) || value <= 0) return "";
    return value >= 1e9 ? `${(value / 1e9).toFixed(2)} GB` : value >= 1e6 ? `${(value / 1e6).toFixed(1)} MB` : `${Math.round(value / 1e3)} KB`;
  }
  function materialGroups(recipe) {
    const hours = value => `${Number.isFinite(value) ? value.toLocaleString("ko-KR", { maximumFractionDigits: 1 }) : "—"}시간`, count = value => `${number(value)}개`;
    const selected = (id, fallback) => $(id).value !== "" && Number.isFinite(Number($(id).value)) && Number($(id).value) >= 0 ? Number($(id).value) : fallback;
    const originals = selected("adv-originals", recipe.originals ?? recipe.positive_originals);
    const genericHours = selected("adv-generic", recipe.generic_hours ?? recipe.automatic_target_hours?.generic_nominal ?? 0);
    const koreanHours = selected("adv-korean", recipe.korean_hours ?? recipe.automatic_target_hours?.korean_train ?? 0);
    const policy = recipe.evaluation_policy || recipe.evaluation_materials_policy || {};
    const assets = recipe.assets || [], sources = recipe.sources || assets;
    const asset = id => sources.find(item => item.id === id) || assets.find(item => item.id === id);
    const size = item => item?.selected_download_bytes ?? item?.bytes ?? item?.size_bytes ?? item?.total_bytes ?? item?.totalbytes;
    const bundleSize = ids => {
      const values = ids.map(id => size(asset(id))).filter(Number.isFinite);
      return values.length ? values.reduce((sum, value) => sum + value, 0) : undefined;
    };
    const source = (name, content, amount, rank, record, fileBytes = size(record)) => ({ name, content, amount, rank, bytes: fileBytes, bytesLabel: record?.download_mode === "selected_members" ? "다운로드" : "원본", url: record?.url || record?.source_url });
    const fallback = record => {
      const time = record.target_hours ?? record.hours;
      return source(record.display_name || record.name || record.id || record.dataset_id || "공개 자료",
        record.description || backgroundTypeNames(record).join(" · ") || "공개 오디오 자료",
        time !== undefined ? hours(time) : record.file_count !== undefined ? count(record.file_count) : "1묶음",
        time === undefined ? null : time * 3600, record);
    };
    const suppliedGeneric = recipe.automatic_target_hours?.generic_nominal ?? asset("generic")?.hours ?? 0;
    const suppliedKorean = recipe.automatic_target_hours?.korean_train ?? asset("zeroth")?.hours ?? 0;
    const negativePlan = recipe.training_negative_plan;
    const automaticPlan = recipe.material_acquisition_policy === "automatic_v3" && recipe.automatic_source_plan?.version === 3
      && recipe.automatic_source_plan.enabled === true && Array.isArray(recipe.automatic_source_plan.components) ? recipe.automatic_source_plan : null;
    const hasNegativePlan = ((negativePlan?.id === "commercial_negative_v1" && negativePlan.version === 1)
      || (negativePlan?.id === "commercial_negative_v2" && negativePlan.version === 2)
      || (negativePlan?.id === "commercial_negative_v3" && negativePlan.version === 3)) && Array.isArray(negativePlan.components);
    const plannedSources = (pool, target) => {
      const components = negativePlan.components.filter(item => item.pool === pool && Number.isFinite(item.target_hours) && item.target_hours > 0);
      const total = components.reduce((sum, item) => sum + item.target_hours, 0);
      return components.map(item => {
        const records = Array.isArray(item.sources) ? item.sources : [];
        const amount = item.target_hours * target / total;
        const name = records.map(record => record.name).filter(Boolean).join(" · ") || item.name;
        const selectiveMusic = negativePlan.version === 3 && records.some(record => record.id === "fma_full" && record.download_mode === "selective");
        const content = selectiveMusic ? "필요한 원곡만 선택 다운로드 · 원본 MP3 보존"
          : [...new Set([item.name, item.description].filter(Boolean))].join(" · ");
        return source(name, content, `준비 목표 ${hours(amount)}`, amount * 3600, records.length === 1 ? records[0] : undefined);
      });
    };
    const automaticSources = (pool, target) => {
      const components = (automaticPlan?.components || []).filter(item => item.pool === pool && Number.isFinite(item.target_hours) && item.target_hours > 0);
      const total = components.reduce((sum, item) => sum + item.target_hours, 0);
      const providers = { yodas2: ["YODAS2 한국어", "한국어 웹 음성 · 목표량 자동 선정", "https://huggingface.co/datasets/espnet/yodas2"] };
      return components.map(item => {
        const provider = providers[item.provider] || [item.name || item.provider, "목표량 자동 선정", undefined];
        const amount = item.target_hours * target / total;
        return source(provider[0], provider[1], `준비 목표 ${hours(amount)}`, amount * 3600, { source_url: item.source_url || provider[2] });
      });
    };
    const automaticGeneric = automaticSources("generic_negative", genericHours), automaticKorean = automaticSources("korean_negative", koreanHours);
    // Planned raw-audio hours never add the opaque legacy 2,000h feature file.
    const generic = hasNegativePlan ? plannedSources("generic_negative", genericHours) : asset("generic") ? [source("openWakeWord", "일반 음성·음악·환경음의 특징 데이터", hours(suppliedGeneric), suppliedGeneric * 3600, asset("generic"))] : [];
    if (!hasNegativePlan && automaticGeneric.length) generic.splice(0, generic.length, ...automaticGeneric);
    const korean = hasNegativePlan ? plannedSources("korean_negative", koreanHours) : automaticKorean.length ? automaticKorean : asset("zeroth") ? [source("Zeroth Korean", "한국어 낭독 음성", hours(suppliedKorean), suppliedKorean * 3600, asset("zeroth"))] : [];
    if (!hasNegativePlan) {
      for (const [items, target, available, content] of [[generic, genericHours, automaticGeneric.length ? genericHours : suppliedGeneric, "일반 비호출 음원·특징 자료"], [korean, koreanHours, automaticKorean.length ? koreanHours : suppliedKorean, "한국어 대화·낭독 등 음성"]]) {
        if (target > available) items.push(source("추가 자료", `직접 준비할 ${content}`, hours(target - available), (target - available) * 3600));
      }
    }
    const reuseNegativeWaveforms = recipe.noise_material_policy === "reuse_negative_waveforms";
    const noise = reuseNegativeWaveforms ? [
      source("일반 비호출 자료", "학습용 원본 음원 재사용", "원본 재사용", null),
      source("한국어 비호출 자료", "학습용 한국어 원본 음원 재사용", "원본 재사용", null)
    ] : [];
    if (!reuseNegativeWaveforms) {
      if (asset("fma")) noise.push(source("FMA", "음악 120곡 · 곡당 30초", "약 1시간", 3600, asset("fma"), bundleSize(["fma", "fma_metadata"])));
      if (automaticKorean.length) noise.push(...automaticKorean.map(item => ({ ...item, content: "학습용 한국어 원본 음원 재사용", amount: "원본 재사용", bytes: undefined })));
      else if (asset("zeroth")) noise.push(source("Zeroth Korean", "학습용 한국어 낭독 재사용", hours(suppliedKorean), suppliedKorean * 3600, asset("zeroth")));
      if (asset("audioset")) noise.push(source("AudioSet", "여러 종류의 일반 오디오", "1묶음", null, asset("audioset"), bundleSize(["audioset", "audioset_segments", "audioset_ontology"])));
    }
    const mit = asset("mit_rir") || recipe.additional_download;
    const hardPerPhrase = Number.isInteger(recipe.hard_samples_per_phrase) && recipe.hard_samples_per_phrase > 0
      ? recipe.hard_samples_per_phrase : null;
    const hardAmount = hardPerPhrase ? `문구당 ${count(hardPerPhrase)}` : count(recipe.hard_originals ?? 0);
    const training = [
      { id: "positive", name: "호출어 음성", description: "호출어 합성 음성", amount: count(originals), sources: [source("Qwen TTS", Number.isInteger(recipe.train_references) && recipe.train_references > 0 ? `기준 목소리 ${number(recipe.train_references)}개 · 호출어만 말한 음성` : "호출어만 말한 음성", count(originals), originals)] },
      { id: "hard", name: "유사어 음성", description: "호출어로 오인할 수 있는 다른 단어·표현", amount: hardAmount, sources: [source("Qwen TTS", hardPerPhrase ? "문구 수는 직접 선택 · 총량은 문구 수에 따라 결정" : "헷갈리기 쉬운 단어·표현을 합성한 음성", hardAmount, hardPerPhrase ? null : recipe.hard_originals ?? 0)] },
      { id: "generic", name: "일반 비호출 자료", description: "일반 음성 · 음악 · 환경음", amount: hours(genericHours), sources: generic },
      { id: "korean", name: "한국어 비호출 자료", description: hasNegativePlan || automaticKorean.length || koreanHours > suppliedKorean ? "한국어 대화 · 낭독 등 음성" : "한국어 낭독 음성", amount: hours(koreanHours), sources: korean },
      { id: "noise", name: "소음 증강 음원", description: reuseNegativeWaveforms ? "학습용 비호출 원본 음원 재사용" : "일반 오디오 · 음악 · 한국어 음성", amount: reuseNegativeWaveforms ? "재사용" : `${number(noise.length)}종`, sources: noise },
      { id: "rir", name: "잔향 자료", description: "공간의 울림을 재현하는 파일", amount: mit ? count(mit.files ?? mit.file_count) : "0개", sources: mit ? [source("MIT RIR", "공간의 울림 특성", count(mit.files ?? mit.file_count), mit.files ?? mit.file_count, mit)] : [] }
    ];
    const isEvaluation = item => ["validation", "development"].includes(item.split) || /held_out|evaluation|validation/.test(item.role || "");
    const knownTrain = new Set(["generic", "zeroth", "fma", "audioset", "mit_rir"]);
    // Old preview payloads may still contain this retired standalone test clip.
    const retiredTrain = new Set(["dns_fan"]);
    for (const item of sources.filter(item => !isEvaluation(item) && item.role !== "provider_metadata" && !knownTrain.has(item.id) && !retiredTrain.has(item.id))) {
      training.push({ id: `extra-${training.length}`, name: item.display_name || item.name || item.id || "공개 자료", description: item.description || "추가 학습 자료", amount: fallback(item).amount, sources: [fallback(item)] });
    }
    const buckets = policy.negative_hours_by_bucket || {};
    const evalSources = [...sources.filter(isEvaluation), ...(recipe.evaluation_sources || [])]
      .filter(item => !["common_development_v4", "common_development_v5"].includes(policy.id) || item.id === "openslr28_common_rir_v1" || Object.keys(buckets).some(key => item.id === `common_30h_${key}`));
    const bucketRecord = key => evalSources.find(item => item.id === `common_30h_${key}` || item.dataset_id === key);
    const bucketHours = key => buckets[key] ?? bucketRecord(key)?.target_hours ?? 0;
    const bucketLabels = { korean_conversation: "대화", korean_interview: "인터뷰", korean_read: "낭독", other_speech: "다른 언어", music: "음악", environment: "환경음", electronic: "전자음" };
    const content = keys => keys.filter(key => bucketHours(key) > 0).map(key => `${bucketLabels[key]} ${hours(bucketHours(key))}`).join(" · ");
    const groupSource = (name, keys, description = content(keys)) => {
      const records = keys.map(bucketRecord).filter(Boolean), total = keys.reduce((sum, key) => sum + bucketHours(key), 0);
      // Bucket records can point to the same archive; its size is never additive.
      const knownSizes = records.map(size).filter(Number.isFinite);
      return source(name, description, hours(total), total * 3600, records[0], knownSizes.length ? Math.max(...knownSizes) : undefined);
    };
    const generalKeys = ["other_speech", "music", "environment", "electronic"], koreanKeys = ["korean_conversation", "korean_interview", "korean_read"];
    const evaluation = [];
    const validationOriginals = recipe.validation_originals ?? policy.validation_originals;
    const references = recipe.validation_references ?? policy.validation_references;
    if (validationOriginals !== undefined) evaluation.push({ id: "positive", name: "호출어 음성", description: `${number(references)}가지 음색 계열의 단독 호출`, amount: count(validationOriginals), sources: [source("Qwen TTS", `${number(references)}가지 음색 계열로 말한 호출어`, count(validationOriginals), validationOriginals)] });
    if (Object.keys(buckets).length) {
      const general = [groupSource("MUSAN", ["other_speech", "music"]), groupSource("FSD50K", ["environment", "electronic"])].filter(item => item.rank > 0);
      const speech = [groupSource("Seoul Corpus", ["korean_conversation", "korean_interview"]), groupSource("Zeroth Korean", ["korean_read"], "한국어 낭독 음성")].filter(item => item.rank > 0);
      const totalHours = Object.values(buckets).reduce((sum, value) => sum + value, 0);
      evaluation.push(
        { id: "generic", name: "일반 비호출 자료", description: content(generalKeys), amount: hours(generalKeys.reduce((sum, key) => sum + bucketHours(key), 0)), sources: general },
        { id: "korean", name: "한국어 비호출 자료", description: content(koreanKeys), amount: hours(koreanKeys.reduce((sum, key) => sum + bucketHours(key), 0)), sources: speech },
        { id: "noise", name: "소음 증강 음원", description: "위 평가용 비호출 음원을 호출어에 섞어 사용", amount: hours(totalHours), sources: [...general, ...speech] }
      );
    }
    // Older policies retain their supplemental sources; v4/v5 filter them above.
    for (const item of evalSources.filter(item => item.id !== "openslr28_common_rir_v1" && (!Object.keys(buckets).length || !item.id?.startsWith("common_30h_")))) {
      evaluation.push({ id: `extra-${evaluation.length}`, name: fallback(item).name, description: fallback(item).content, amount: fallback(item).amount, sources: [fallback(item)] });
    }
    const rir = evalSources.find(item => item.id === "openslr28_common_rir_v1");
    if (rir) {
      const description = `가상 공간 ${number(rir.simulated_room_count ?? rir.room_count ?? policy.rir_rooms)}개 · 중간/강한 울림`;
      evaluation.push({ id: "rir", name: "잔향 자료", description, amount: count(rir.file_count), sources: [source("RIRS_NOISES", description, count(rir.file_count), rir.file_count, rir)] });
    }
    return { training, evaluation, policy };
  }
  let activeMaterial = null;
  function renderRecipe() {
    const recipe = recipes.find(item => item.id === selectedPreset);
    if (!recipe) return;
    renderVoiceDesign("setup-voices", recipe.reference_design);
    const placeholders = { "adv-originals": recipe.originals ?? recipe.positive_originals, "adv-rounds": recipe.rounds ?? recipe.augmentation_rounds, "adv-updates": recipe.updates, "adv-generic": recipe.generic_hours, "adv-korean": recipe.korean_hours };
    for (const [id, value] of Object.entries(placeholders)) $(id).placeholder = value === undefined ? "규모 기본값" : String(value);
    const groups = materialGroups(recipe), sections = [], openId = activeMaterial?.id;
    activeMaterial = null;
    const show = (entry, open) => { entry.panel.hidden = !open; entry.button.setAttribute("aria-expanded", String(open)); };
    for (const [key, label] of [["training", "학습용 자료"], ["evaluation", "성능 평가용 자료"]]) {
      const items = groups[key];
      if (!items.length) continue;
      const section = element("section"), heading = element("div", undefined, "materials-heading"), title = element("h3", label);
      title.id = `materials-${key}-title`; section.setAttribute("aria-labelledby", title.id); heading.append(title);
      if (key === "evaluation") heading.append(element("span", "공통 평가"));
      const list = element("ul", undefined, "materials-list");
      for (const item of items) {
        const id = `materials-${key}-${item.id}`, row = element("li", undefined, "material"), button = element("button", undefined, "material-button");
        button.type = "button"; button.setAttribute("aria-expanded", "false"); button.setAttribute("aria-controls", `${id}-sources`);
        button.setAttribute("aria-labelledby", `${id}-name`); button.setAttribute("aria-describedby", `${id}-description ${id}-amount`);
        const text = element("span", undefined, "material-copy"), name = element("span", item.name, "material-name"), description = element("span", item.description, "material-description"), amount = element("span", item.amount, "material-amount");
        name.id = `${id}-name`; description.id = `${id}-description`; amount.id = `${id}-amount`; text.append(name, description);
        const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg"), path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        icon.setAttribute("class", "chevron"); icon.setAttribute("viewBox", "0 0 24 24"); icon.setAttribute("aria-hidden", "true"); icon.setAttribute("focusable", "false");
        for (const [key, value] of Object.entries({ d: "m6 9 6 6 6-6", fill: "none", stroke: "currentColor", "stroke-width": "1.8", "stroke-linecap": "round", "stroke-linejoin": "round" })) path.setAttribute(key, value);
        icon.append(path); button.append(text, amount, icon);
        const panel = element("div", undefined, "source-panel"); panel.id = `${id}-sources`; panel.hidden = true;
        panel.setAttribute("role", "region"); panel.setAttribute("aria-label", `${label} · ${item.name} 출처`);
        const sources = element("ul", undefined, "source-list");
        const ordered = item.sources.map((source, index) => ({ ...source, index })).sort((a, b) => a.rank == null ? (b.rank == null ? a.index - b.index : 1) : b.rank == null ? -1 : b.rank - a.rank || a.index - b.index);
        for (const source of ordered) {
          const row = element("li", undefined, "source-item"), text = element("span", undefined, "source-copy"), quantity = element("span", source.amount, "source-quantity");
          const href = webUrl(source.url), name = element(href ? "a" : "span", source.name, "source-name");
          if (href) { name.href = href; name.target = "_blank"; name.rel = "noopener"; }
          text.append(name, element("span", source.content, "source-content"));
          const size = originalSize(source.bytes);
          if (size) quantity.append(element("span", `${source.bytesLabel || "원본"} ${size}`, "file-size"));
          row.append(text, quantity); sources.append(row);
        }
        panel.append(sources); row.append(button, panel); list.append(row);
        const current = { id, button, panel };
        if (id === openId) { show(current, true); activeMaterial = current; }
        button.addEventListener("click", () => { const open = panel.hidden; if (activeMaterial) show(activeMaterial, false); show(current, open); activeMaterial = open ? current : null; });
      }
      section.append(heading, list);
      if (key === "evaluation") {
        const summary = element("div", undefined, "materials-summary"), policy = groups.policy;
        const totalHours = Object.values(policy.negative_hours_by_bucket || {}).reduce((sum, value) => sum + value, 0);
        if (policy.positive_sessions !== undefined) { const line = element("span", "호출 시험"); line.append(element("strong", `${number(policy.positive_sessions)}회`)); summary.append(line); }
        if (totalHours) { const line = element("span", "비호출 시험"); line.append(element("strong", `${number(totalHours)}시간 이상`)); summary.append(line); }
        if (summary.children.length) section.append(summary);
      }
      sections.push(section);
    }
    replace($("source-preview"), sections);
    const selected = (id, value) => $(id).value !== "" && Number.isFinite(Number($(id).value)) && Number($(id).value) >= 0 ? Number($(id).value) : value;
    const values = { originals: selected("adv-originals", recipe.originals), rounds: selected("adv-rounds", recipe.rounds), genericHours: selected("adv-generic", recipe.generic_hours), koreanHours: selected("adv-korean", recipe.korean_hours) };
    $("storage-budget-details").hidden = true;
    replace($("storage-budget-content"), []);
    if (Object.hasOwn(recipe, "storage_budget")) {
      const budget = storageBudget(recipe, values);
      $("source-plan-note").hidden = false;
      $("source-plan-note").textContent = budget
        ? `전체 필요 공간: 미확정. ${budget.hard_generation?.status === "pending_phrase_count" ? "유사어 분량을 제외한 " : ""}계산 가능한 항목의 비교 합계 ${bytes(budget.reference_subtotal_bytes)}.`
        : "전체 필요 공간: 미확정. 용량 계산 근거를 확인할 수 없습니다.";
      if (budget) {
        const wrap = element("div", undefined, "storage-table-wrap"), table = element("table", undefined, "storage-table");
        table.setAttribute("aria-label", "저장 공간 항목별 계산 근거");
        const head = element("thead"), headings = element("tr"), body = element("tbody");
        for (const label of ["항목", "용량", "구분", "근거"]) { const cell = element("th", label); cell.setAttribute("scope", "col"); headings.append(cell); }
        head.append(headings);
        const classifications = { published: "고정 파일", measured: "실측", calculated: "형식 계산", reference: "참조", unresolved: "미확정" };
        const unknowns = budget.unknowns.filter(item => !budget.rows.some(row => row.id === item.id && row.bytes === null));
        const rows = [...budget.rows, ...unknowns.map(item => ({ ...item, bytes: null, classification: "unresolved" }))];
        for (const row of rows) {
          const line = element("tr");
          const pendingPhraseCount = row.id === "hard_features" && budget.hard_generation?.status === "pending_phrase_count";
          const amount = row.bytes === null ? (pendingPhraseCount ? "문구 수 미정" : "미확정") : bytes(row.bytes);
          line.append(element("th", row.label), element("td", amount),
            element("td", classifications[row.classification]), element("td", row.detail));
          line.children[0].setAttribute("scope", "row"); body.append(line);
        }
        table.append(head, body); wrap.append(table);
        const content = [element("p", "비교 합계는 실제 FLAC 저장량이나 최대 필요 공간이 아닙니다.", "hint"), wrap];
        if (budget.hard_generation?.status === "pending_phrase_count") {
          content.push(element("p", `유사어 문구 1종당 참고 용량 ${bytes(budget.hard_generation.incremental_bytes_per_phrase)}가 추가됩니다.`, "hint"));
        }
        if (budget.assumptions.length) {
          const list = element("ul", undefined, "storage-assumptions");
          for (const assumption of budget.assumptions) list.append(element("li", assumption));
          content.push(list);
        }
        replace($("storage-budget-content"), content); $("storage-budget-details").hidden = false;
      }
      return;
    }
    const automatic = recipe.material_acquisition_policy === "automatic_v3";
    $("source-plan-note").hidden = !automatic;
    if (automatic) {
      const known = bytes(recipe.known_download_bytes);
      const additional = recipe.automatic_source_plan?.components?.length > 0;
      const estimate = storageEstimate(recipe, values);
      const free = snapshot.system?.free_bytes ?? snapshot.system?.disk_free_bytes;
      $("source-plan-note").textContent = estimate?.basis === "wav_reference"
        ? `WAV 기준 참고 예산 ${bytes(estimate.total)}${estimate.transient > 0 ? "(임시 " + bytes(estimate.transient) + " 포함)" : ""}. 학습 음원은 무손실 FLAC으로 보관하며 실제 용량은 변환 후 확인합니다.${Number.isFinite(free) && estimate.total > free ? " 현재 여유가 참고 예산보다 적습니다." : ""} 다운로드량은 같으며 원본 재처리 시 다시 받습니다.`
        : additional
        ? `목표량에 맞춰 원본을 자동 선정합니다.${known ? " 고정 자료 " + known + " 외" : ""} 추가 다운로드 용량은 목록 확정 후 확인합니다.`
        : `필요한 공개 자료를 자동 준비합니다.${known ? " 고정 다운로드 " + known + "." : ""} 실제 준비량과 검수 승인은 별도로 확인합니다.`;
    }
  }
  function renderSystem() {
    const system = snapshot.system || {};
    const gpu = system.gpu?.name || system.GPU?.name || system.gpu_name;
    const free = bytes(system.free_bytes ?? system.disk_free_bytes);
    const parts = [gpu ? `GPU: ${gpu}` : "", free ? `저장 공간 여유 ${free}` : ""];
    $("system-summary").textContent = parts.filter(Boolean).join(" · ");
  }
  function renderSelector() {
    const projects = snapshot.projects || [];
    const selector = $("project-select");
    replace(selector, projects.map(project => {
      let label = project.wakeword || project.id;
      if (projects.filter(other => other.wakeword === project.wakeword).length > 1) {
        const folder = String(project.directory || "").split(/[\\/]/).filter(Boolean).pop();
        label += " · " + (folder || project.id);
      }
      const option = element("option", label); option.value = project.id; return option;
    }));
    if (newProject) { const option = element("option", "새 모델"); option.value = ""; selector.append(option); selector.value = ""; }
    else selector.value = selectedId;
    $("project-switch").hidden = !projects.length;
    selector.hidden = !projects.length; $("new-project").hidden = !projects.length;
    selector.disabled = busy; $("new-project").disabled = busy;
  }
  function renderStages(project) {
    const provided = Array.isArray(project.stages) ? project.stages : [];
    const stageId = project.stage && typeof project.stage === "object" ? project.stage.id : provided.find(item => item.id === project.stage || item.label === project.stage)?.id || project.stage;
    const currentIndex = steps.findIndex(([id]) => stepGroups[id].includes(stageId));
    const complete = completeStatuses.has(project.status);
    replace($("stages"), steps.map(([id, label], index) => {
      const records = provided.filter(item => stepGroups[id].includes(item.id));
      // An unknown stage never becomes a guessed percentage or a fabricated completion.
      const done = records.length ? records.every(record => completeStatuses.has(record.status) || record.status === "done") : complete || currentIndex > index;
      const current = currentIndex === index;
      const item = element("li", undefined, "stage" + (done ? " done" : current ? " current" : ""));
      if (current) item.setAttribute("aria-current", "step");
      item.append(element("div", undefined, "stage-bar"), element("span", `${done ? "✓ " : ""}${label}`)); return item;
    }));
  }
  function renderCounts(project) {
    const values = project.counts;
    const records = Array.isArray(values) ? values : values && typeof values === "object" ? Object.entries(values).map(([label, value]) => ({ label, value })) : [];
    const labels = { tts_generated: "생성 시도", positive_generated: "생성 음원", technical_passed: "파일 검사 통과", reviewed: "사람 청취", admitted: "학습 사용 판정", trained_models: "학습 완료 모델" };
    // Only supplied counters are shown; generation is never called human approval.
    replace($("counts"), records.map(record => { const item = element("span"); item.append(element("span", (labels[record.label || record.id] || record.label || record.id || "자료") + " "), element("strong", `${typeof record.value === "number" ? number(record.value) : String(record.value ?? "—")}${record.total !== undefined ? " / " + number(record.total) : ""}${record.unit || ""}`)); return item; }));
    $("counts").hidden = !records.length;
    const training = project.training_materials;
    $("training-material-coverage").hidden = !training;
    if (training) {
      const hours = value => Number.isFinite(value) ? `${value.toLocaleString("ko-KR", { maximumFractionDigits: 2 })}시간` : "확인 전";
      const lines = ["generic_negative", "korean_negative"].map(role => {
        const label = role === "generic_negative" ? "일반 비호출" : "한국어 비호출";
        const target = training.targets?.[role], nominal = training.generic_publisher_nominal_hours;
        if (role === "generic_negative" && Number.isFinite(nominal) && nominal > 0 && Number.isFinite(target) && target <= nominal) {
          return element("li", `${label}: 공개 특징 명목 ${hours(nominal)} / 목표 ${hours(target)}`);
        }
        return element("li", `${label}: 기술 검사 ${hours(training.candidate?.[role])} · 검수 승인 ${hours(training.admitted?.[role])} / 목표 ${hours(target)}`);
      });
      const storage = training.storage;
      if (storage?.storage_policy === "canonical_flac_only_v1"
        && Number.isFinite(storage.canonical_storage_bytes) && storage.canonical_storage_bytes >= 0) {
        lines.push(element("li", `학습용 FLAC 실제 보관량: ${bytes(storage.canonical_storage_bytes)}`));
      }
      replace($("training-material-coverage-list"), lines);
    }
    const materials = project.evaluation_materials;
    $("material-coverage").hidden = !materials;
    if (materials) {
      const names = {korean_conversation:"한국어 대화",korean_interview:"한국어 인터뷰",korean_read:"한국어 낭독",other_speech:"다른 언어",music:"음악",environment:"환경음",electronic:"전자음"};
      const targets = materials.candidate?.target_hours_by_bucket || materials.admitted?.target_hours_by_bucket || {};
      $("material-coverage-note").textContent = materials.scope;
      const candidate = materials.candidate?.core_coverage || materials.candidate;
      const admitted = materials.admitted?.core_coverage || materials.admitted;
      const lines = Object.entries(targets).map(([key,target]) => element("li", `${names[key]||key}: 기술 검사 ${(candidate?.actual_hours_by_bucket?.[key]||0).toFixed(2)}h · 검수 승인 ${(admitted?.actual_hours_by_bucket?.[key]||0).toFixed(2)}h / ${number(target)}h`));
      for (const [stage, label] of [["candidate", "기술 검사"], ["admitted", "검수 승인"]]) {
        const duration = materials.duration?.[stage];
        if (!duration) continue;
        const measured = (value, factor, unit) => Number.isFinite(value) ? `${(value * factor).toFixed(3)}${unit}` : "확인 전";
        lines.push(element("li", `${label}: 기본 ${measured(duration.core_hours, 1, "h")} · 초기화 보충 ${measured(duration.initialization_margin_hours, 60, "분")} · 채점 가능 ${measured(duration.eligible_hours, 1, "h")}`, "material-duration"));
      }
      replace($("material-coverage-list"), lines);
    }
  }
  function frameSource(frame, url) {
    const safe = localUrl(url);
    if (!safe) { frame.removeAttribute("src"); return; }
    if (frame.getAttribute("src") !== safe) frame.setAttribute("src", safe);
  }
  function renderVoiceDesign(id, design) {
    const groups = design?.counts, train = groups?.train, validation = groups?.validation;
    const show = Boolean(train && validation && Number.isInteger(train.count) && Number.isInteger(validation.count));
    $(id).hidden = !show;
    if (!show) { replace($(id + "-rows"), []); return; }
    const labels = {
      man:"남성", woman:"여성", male:"남성", female:"여성",
      child:"어린이", teenager:"청소년",
      "young adult":"젊은 성인", "middle-aged":"중년", "older adult":"고령", adult:"성인(연령 미지정)",
      young_adult:"젊은 성인", middle_aged:"중년", older_adult:"고령",
      neutral:"담담하게", friendly:"친근하게", urgent:"다급하게", projected:"또렷하게",
      soft:"부드럽게", normal:"보통", strong:"강하게"
    };
    const distribution = (group, axis) => {
      const entries = Object.entries(group[axis] || {}).filter(([, count]) => Number.isInteger(count) && count > 0);
      if (!entries.length) return "미지정";
      if (axis === "timbre") {
        const values = entries.map(([,count]) => count), low = Math.min(...values), high = Math.max(...values);
        return `${number(entries.length)}종 · 유형당 ${number(low)}${high === low ? "" : "–" + number(high)}개`;
      }
      return entries.map(([label,count]) => `${labels[label] || label} ${number(count)}개`).join(" · ");
    };
    $(id + "-train").textContent = `학습용 ${number(train.count)}개`;
    $(id + "-validation").textContent = `공통 평가용 ${number(validation.count)}개`;
    replace($(id + "-rows"), [["gender","성별"],["age","연령"],["timbre","음색"],["style","말투"],["vocal_effort","발성 강도"]].map(([axis,label]) => {
      const row = element("tr"), heading = element("th",label); heading.setAttribute("scope","row");
      row.append(heading, element("td",distribution(train,axis)), element("td",distribution(validation,axis))); return row;
    }));
  }
  function renderReferencePreview(project) {
    const preview = project?.reference_preview, url = localUrl(preview?.url);
    const show = Boolean(url && Number.isInteger(preview?.count) && preview.count > 0);
    const details = $("reference-preview"), frame = $("reference-preview-frame");
    const identity = show ? project.id + ":" + url : "";
    if (referencePreviewIdentity !== identity) {
      details.open = false;
      frameSource(frame, "");
      referencePreviewIdentity = identity;
    }
    details.hidden = !show;
    if (!show) return;
    $("reference-preview-counts").textContent = [preview.train, preview.validation].every(value => Number.isInteger(value) && value >= 0)
      ? `학습용 ${number(preview.train)}개 · 공통 평가용 ${number(preview.validation)}개`
      : `생성된 기준 목소리 ${number(preview.count)}개`;
    frameSource(frame, details.open ? url : "");
  }
  function renderReview(project) {
    const review = project.review;
    const url = localUrl(review?.url);
    const show = Boolean(review && url && !activeStatuses.has(project.status));
    $("review-section").hidden = !show;
    if (!show) { if (reviewIdentity) frameSource($("review-frame"), ""); reviewIdentity = ""; renderVoiceDesign("review-voices", null); return; }
    const identity = project.id + ":" + (review.queue_id || review.id || url);
    const references = /^references(?:_|$)/.test(review.queue_id || "");
    const counts = project.reference_counts;
    const voiceCounts = counts && [counts.train, counts.validation].every(value => Number.isInteger(value) && value > 0)
      ? `학습용 ${number(counts.train)}개 · 공통 평가용 ${number(counts.validation)}개 · 총 ${number(counts.train + counts.validation)}개. ` : "";
    $("review-title").textContent = references ? "합성 목소리 확인" : review.title || "오디오 검수";
    $("review-description").textContent = references ? voiceCounts + "목소리와 음질을 확인하세요. 호출어 발음은 다음 단계에서 확인합니다." : review.description || "끝까지 들으면 자동으로 보존됩니다. 문제가 있는 항목만 바꾼 뒤 검수 완료·저장을 누르세요.";
    renderVoiceDesign("review-voices", references ? project.reference_design : null);
    if (reviewIdentity !== identity) { frameSource($("review-frame"), url); reviewIdentity = identity; }
  }
  function renderInput(project) {
    const input = project.input;
    const show = Boolean(input && !activeStatuses.has(project.status));
    $("input-section").hidden = !show;
    if (!show) { inputIdentity = ""; $("hard-generation-counts").hidden = true; return; }
    const identity = project.id + ":" + (input.id || input.kind);
    const hard = input.kind === "hard_phrases";
    const dependencies = input.kind === "dependency_setup";
    const editable = dependencies || hard || input.kind === "material_admissions";
    $("input-form").hidden = !editable;
    $("input-title").textContent = input.title || (!editable ? "작업을 확인해주세요" : dependencies ? "학습 도구를 준비해주세요" : hard ? "유사어 후보를 정해주세요" : "추가 자료를 연결해주세요");
    $("input-description").textContent = input.description || (!editable ? "자동으로 해결할 수 없는 문제가 있어 멈췄습니다. 아래 작업 기록에서 원인과 필요한 조치를 확인하세요." : dependencies ? "이 가상 환경에 필요한 음성 생성·학습 도구를 설치합니다. 사용할 연산 환경을 선택하세요." : hard ? "호출어와 헷갈릴 수 있는 표현을 입력하세요." : "자동으로 준비할 수 없는 자료가 있습니다. 안내에 맞는 검수·승인 기록을 연결하면 이어서 진행합니다.");
    $("input-label").textContent = input.label || (dependencies ? "설치할 연산 도구" : hard ? "유사어 문구" : "자료 승인 기록 · 한 줄에 경로 하나씩");
    $("input-label").hidden = false; $("input-hint").hidden = false;
    $("input-label").htmlFor = dependencies ? "dependency-backend" : hard ? "hard-phrase-1" : "input-value";
    $("input-value").hidden = dependencies || hard; $("input-value").required = !dependencies && !hard;
    $("hard-phrase-editor").hidden = !hard;
    $("dependency-backend").hidden = !dependencies;
    $("input-submit").textContent = dependencies ? "선택한 도구 설치하고 계속" : "저장하고 이어서 진행";
    $("input-hint").textContent = input.hint || (dependencies ? "현재 가상 환경에만 설치합니다. NVIDIA GPU가 없다면 CPU용을 선택하세요." : hard ? "확인한 후보로 소수의 검수용 음성을 먼저 만듭니다." : "직접 준비한 자료의 승인 기록이 필요합니다. 원본 음원 경로만 입력하면 승인된 것으로 처리하지 않습니다.");
    if (identity !== inputIdentity) {
      $("input-value").value = (input.phrases || input.values || input.suggestions || []).join("\n");
      hardPhraseFields = [];
      $("hard-phrase-fields").replaceChildren();
      if (hard) for (const value of (input.phrases || input.values || input.suggestions || []).length ? (input.phrases || input.values || input.suggestions) : [""]) addHardPhraseField(value);
      $("dependency-backend").value = input.recommended_backend === "cu128" ? "cu128" : "cpu";
      $("material-mode").value = "source";
      const source = input.source || {};
      $("source-directory").value = source.directory || ""; $("source-dataset").value = source.dataset || "zeroth";
      $("source-pool").value = source.pool || input.pool || ({ mls: "generic_negative", yodas2: "korean_negative" }[$("source-dataset").value]) || "korean_negative";
      $("source-hours").value = String(source.hours ?? input.remaining_hours ?? input.required_hours ?? input.target_hours ?? "");
      for (const [id, key] of [["source-metadata", "metadata"], ["source-url", "source_url"], ["source-license", "license"]]) $(id).value = source[key] || "";
      $("source-metadata-details").open = ["mls", "yodas2"].includes($("source-dataset").value);
      inputIdentity = identity;
    }
    renderInputMode(project);
    renderHardGenerationCounts(project);
    $("input-submit").disabled = busy;
    $("hard-phrase-add").disabled = busy;
    for (const field of hardPhraseFields) { field.input.disabled = busy; field.remove.disabled = busy; }
  }
  function hardPhraseValues() {
    return hardPhraseFields.map(field => field.input.value.trim()).filter(Boolean);
  }
  function addHardPhraseField(value = "", focus = false) {
    const row = document.createElement("div"), input = document.createElement("input"), remove = document.createElement("button");
    row.className = "hard-phrase-field"; input.type = "text"; input.value = value;
    input.placeholder = "유사어 입력"; input.maxLength = 16; input.autocomplete = "off";
    remove.type = "button"; remove.className = "hard-phrase-remove"; remove.textContent = "×";
    const field = { row, input, remove };
    input.addEventListener("input", () => renderHardGenerationCounts(selectedProject()));
    input.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.isComposing) { event.preventDefault(); if (!busy) addHardPhraseField("", true); }
    });
    remove.addEventListener("click", () => {
      if (busy) return;
      const position = hardPhraseFields.indexOf(field);
      hardPhraseFields = hardPhraseFields.filter(item => item !== field);
      $("hard-phrase-fields").replaceChildren(...hardPhraseFields.map(item => item.row));
      if (!hardPhraseFields.length) addHardPhraseField();
      labelHardPhraseFields();
      hardPhraseFields[Math.min(position, hardPhraseFields.length - 1)].input.focus();
      renderHardGenerationCounts(selectedProject());
    });
    row.append(input, remove); hardPhraseFields.push(field); $("hard-phrase-fields").append(row);
    labelHardPhraseFields(); renderHardGenerationCounts(selectedProject());
    if (focus) input.focus();
  }
  function labelHardPhraseFields() {
    hardPhraseFields.forEach((field, index) => {
      field.input.id = `hard-phrase-${index + 1}`;
      field.input.setAttribute("aria-label", `유사어 ${index + 1}`);
      field.remove.setAttribute("aria-label", `유사어 ${index + 1} 삭제`);
    });
  }
  function renderHardGenerationCounts(project) {
    const input = project?.input, perPhrase = input?.samples_per_phrase;
    const show = input?.kind === "hard_phrases" && Number.isInteger(perPhrase) && perPhrase > 0;
    const counts = $("hard-generation-counts");
    counts.hidden = !show;
    if (!show) { counts.textContent = ""; return; }
    const phrases = new Set(hardPhraseValues());
    counts.textContent = `유사어 ${number(phrases.size)}종 · 문구당 ${number(perPhrase)}개 · 총 ${number(phrases.size * perPhrase)}개`;
  }
  function renderInputMode(project) {
    const material = project?.input?.kind === "material_admissions";
    const source = material && $("material-mode").value === "source";
    const rir = source && $("source-pool").value === "rir";
    const downloadReceipt = source && ["mls", "yodas2"].includes($("source-dataset").value);
    $("material-controls").hidden = !material; $("source-fields").hidden = !source;
    $("source-directory").required = source;
    $("source-hours").required = source && !rir;
    $("source-hours-label").hidden = rir;
    $("source-rir-note").hidden = !rir;
    $("source-format-note").hidden = !downloadReceipt;
    $("source-metadata").required = downloadReceipt;
    $("source-metadata").placeholder = downloadReceipt ? "download_receipt.json 경로" : "해당 자료의 목록 파일";
    if (!material) return;
    $("input-label").hidden = source; $("input-value").hidden = source; $("input-value").required = !source;
    $("input-hint").hidden = source;
    $("input-submit").textContent = source ? "자료 준비하고 검수하기" : "승인 자료 연결하고 계속";
  }
  function renderResultFrame(project) {
    const result = project.results || {};
    const resultUrl = localUrl(result.url || project.results_url), micUrl = localUrl(result.mic_url || project.mic_url);
    const analysisUrl = resultUrl ? `/analysis/${encodeURIComponent(project.id)}/` : "";
    const show = Boolean(resultUrl);
    $("results-section").hidden = !show;
    if (!show) { if (resultIdentity) frameSource($("result-frame"), ""); $("evaluation-coverage").hidden = true; resultIdentity = ""; resultMode = "results"; return; }
    if (resultIdentity !== project.id + ":" + resultUrl) { resultMode = "results"; resultIdentity = project.id + ":" + resultUrl; }
    if (!micUrl && resultMode === "mic") resultMode = "results";
    $("mic-tab").hidden = !micUrl;
    $("analysis-tab").hidden = !analysisUrl;
    $("analysis-tab").className = "tab" + (resultMode === "analysis" ? " active" : "");
    $("analysis-tab").setAttribute("aria-selected", String(resultMode === "analysis"));
    $("result-tab").className = "tab" + (resultMode === "results" ? " active" : "");
    $("mic-tab").className = "tab" + (resultMode === "mic" ? " active" : "");
    $("result-tab").setAttribute("aria-selected", String(resultMode === "results")); $("mic-tab").setAttribute("aria-selected", String(resultMode === "mic"));
    const url = resultMode === "mic" ? micUrl : resultMode === "analysis" ? analysisUrl : resultUrl;
    frameSource($("result-frame"), url); $("result-open").href = url;
    $("result-frame").title = resultMode === "mic" ? "마이크로 호출어 시험" : resultMode === "analysis" ? "모델 선택과 감지 민감도 비교" : "성능과 평가 오디오";
    $("results-note").textContent = resultMode === "mic" ? "마이크를 허용한 뒤 호출어를 말해보세요." : resultMode === "analysis" ? "선택한 모델과 감지 설정을 별도로 저장해 시험합니다." : "같은 시험 오디오에서 모델별 성능을 비교한 결과입니다.";
    const assessment = result.common_evaluation_assessment;
    $("evaluation-coverage").hidden = !assessment || resultMode !== "results";
    if (assessment) {
      const shortages = Array.isArray(assessment.shortages) ? assessment.shortages : [];
      $("evaluation-coverage-summary").textContent = assessment.coverage_targets_passed ? "시험 자료 준비 완료" : `시험 자료 확인 필요 · ${number(shortages.length)}개 항목`;
      replace($("evaluation-shortages"), shortages.map(item => element("li", item)));
    }
  }
  const downloadStates = { waiting: "대기", downloading: "다운로드", verifying: "검증", complete: "완료", reused: "기존 파일", retrying: "재시도", failed: "실패" };
  let downloadProject = "", downloadGroups = new Map();
  const downloadNumber = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
  function downloadChildren(parent, children) {
    if (parent.children.length !== children.length || children.some((child, index) => parent.children[index] !== child)) replace(parent, children);
  }
  function downloadRow(tag) {
    const node = element(tag, undefined, "download-item"), line = element("div", undefined, "download-line");
    const view = { node, name: element("span", undefined, "download-name"), state: element("span"),
      amount: element("span", undefined, "download-amount"), speed: element("span", undefined, "download-speed"),
      bar: element("progress"), error: element("p", undefined, "download-error") };
    view.bar.max = 100; view.bar.hidden = true; view.error.hidden = true;
    line.append(view.name, view.state, view.amount, view.speed); node.append(line, view.bar, view.error);
    return view;
  }
  function updateDownload(view, row, label) {
    const state = Object.hasOwn(downloadStates, row.state) ? row.state : "unknown";
    view.name.textContent = label;
    view.state.textContent = downloadStates[state] || "상태 확인 중";
    view.state.className = "download-state " + state;
    const unit = row.unit ?? "bytes", knownUnit = ["bytes", "hours", "files"].includes(unit);
    const quantity = value => unit === "bytes" ? bytes(value) : `${number(value)}${unit === "hours" ? "시간" : "개"}`;
    const done = downloadNumber(row.done), total = downloadNumber(row.total);
    const unknown = unit === "bytes" ? "전체 용량 확인 전" : "전체 분량 확인 전";
    view.amount.textContent = unit === "mixed" ? "" : !knownUnit ? "진행량 확인 전"
      : total ? `${done ? quantity(row.done) + " / " : "전체 "}${quantity(row.total)}`
      : done && row.done > 0 ? `${quantity(row.done)} · ${unknown}` : unknown;
    // Local progress may contain a resumed partial or a verified cached file.
    // Only measured bytes from the current transfer can carry a live rate.
    view.speed.textContent = unit === "bytes" && state === "downloading" && downloadNumber(row.network_bytes) && row.network_bytes > 0 && downloadNumber(row.bytes_per_second) && row.bytes_per_second > 0
      ? `${bytes(row.bytes_per_second)}/초` : "";
    view.bar.hidden = !knownUnit || !done || !total || row.total <= 0;
    if (!view.bar.hidden) { view.bar.value = Math.min(100, row.done / row.total * 100); view.bar.setAttribute("aria-label", label + " 확보 진행률"); }
    view.error.textContent = state === "failed" && typeof row.error === "string" ? row.error : "";
    view.error.hidden = !view.error.textContent;
  }
  function downloadGroupProgress(rows) {
    const sum = key => rows.every(row => downloadNumber(row[key])) ? rows.reduce((total, row) => total + row[key], 0) : null;
    const finished = row => ["complete", "reused"].includes(row.state);
    const state = rows.every(row => row.state === "reused") ? "reused" : rows.every(finished) ? "complete"
      : ["failed", "downloading", "retrying", "verifying", "waiting"].find(state => rows.some(row => row.state === state)) || "unknown";
    const units = new Set(rows.map(row => row.unit ?? "bytes")), unit = units.size === 1 ? [...units][0] : "mixed";
    return { state, unit, done: unit === "mixed" ? null : sum("done"), total: unit === "mixed" ? null : sum("total"), network_bytes: sum("network_bytes"),
      bytes_per_second: rows.reduce((sum, row) => sum + (row.state === "downloading" && downloadNumber(row.network_bytes) && row.network_bytes > 0 && downloadNumber(row.bytes_per_second) ? row.bytes_per_second : 0), 0) };
  }
  function renderDownloadFiles(group) {
    if (!group.details?.open) return;
    const children = group.rows.map(row => {
      let view = group.files.get(row.id);
      if (!view) { view = downloadRow("li"); group.files.set(row.id, view); }
      updateDownload(view, row, row.label || row.id); return view.node;
    });
    const ids = new Set(group.rows.map(row => row.id));
    for (const id of group.files.keys()) if (!ids.has(id)) group.files.delete(id);
    downloadChildren(group.list, children);
  }
  function renderDownloads(project) {
    const input = Array.isArray(project.progress?.files) ? project.progress.files : [];
    const files = [...new Map(input.filter(row => row && typeof row.id === "string" && row.id).map(row => [row.id, row])).values()];
    $("downloads-section").hidden = !files.length;
    if (downloadProject !== project.id || !files.length) {
      downloadProject = project.id; downloadGroups = new Map(); replace($("downloads-list"), []);
    }
    if (!files.length) { $("downloads-count").textContent = ""; return; }
    $("downloads-count").textContent = `${number(files.filter(row => ["complete", "reused"].includes(row.state)).length)} / ${number(files.length)}개 준비 완료`;
    const groups = new Map();
    for (const row of files) {
      const key = typeof row.group === "string" && row.group ? row.group : "file:" + row.id;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    const nodes = [];
    for (const [key, rows] of groups) {
      const multiple = rows.length > 1;
      let group = downloadGroups.get(key);
      if (!group || Boolean(group.details) !== multiple) {
        const node = element("li", undefined, "download-group"), view = downloadRow(multiple ? "summary" : "div");
        group = { node, view, rows, files: new Map() };
        if (multiple) {
          group.details = element("details", undefined, "download-details"); group.list = element("ul", undefined, "download-files");
          view.node.className += " download-summary"; group.details.append(view.node, group.list); node.append(group.details);
          group.details.addEventListener("toggle", () => renderDownloadFiles(group));
        } else node.append(view.node);
        downloadGroups.set(key, group);
      }
      group.rows = rows;
      const singleLabel = rows[0].group === "common_rir" && typeof rows[0].group_label === "string" && rows[0].group_label
        ? rows[0].group_label : rows[0].label || rows[0].id;
      const label = multiple ? `${rows[0].group_label || rows[0].group || "자료 묶음"} · ${number(rows.filter(row => ["complete", "reused"].includes(row.state)).length)} / ${number(rows.length)}개 완료` : singleLabel;
      updateDownload(group.view, multiple ? downloadGroupProgress(rows) : rows[0], label);
      renderDownloadFiles(group); nodes.push(group.node);
    }
    for (const key of downloadGroups.keys()) if (!groups.has(key)) downloadGroups.delete(key);
    downloadChildren($("downloads-list"), nodes);
  }
  function renderProject(project) {
    const running = activeStatuses.has(project.status), complete = completeStatuses.has(project.status), waiting = Boolean(project.review || project.input) || /wait|need|await/.test(project.status || "");
    $("project-title").textContent = `‘${project.wakeword || project.id}’`;
    $("project-purpose").textContent = names[project.preset] || project.preset_label || "호출어 학습";
    $("project-directory").textContent = project.directory || project.root || project.path || "";
    $("project-status").textContent = statuses[project.status] || project.status_label || "상태 확인 중";
    $("project-status").className = "status-pill" + (complete ? " complete" : project.error ? " failed" : waiting ? " waiting" : running ? " running" : "");
    renderStages(project);
    const stageLabel = project.stage && typeof project.stage === "object" ? project.stage.label : steps.find(([id]) => id === project.stage)?.[1];
    $("current-category").textContent = waiting ? "사용자 확인 대기" : complete ? "완료" : "현재 작업";
    const stageId = typeof project.stage === "object" ? project.stage?.id : project.stage;
    const voiceTitle = stageId === "references" ? "합성 목소리 만들기"
      : stageId === "references_review" ? project.stage?.kind === "task" ? stageLabel : "합성 목소리 확인" : "";
    $("current-title").textContent = voiceTitle || project.stage_label || project.title || stageLabel || (complete ? "학습과 평가를 마쳤습니다" : waiting ? "안내를 확인해주세요" : "시작할 준비가 되었습니다");
    $("current-description").textContent = project.next_action || project.description || (waiting ? "아래에서 필요한 확인을 마치면 이어서 진행합니다." : complete ? "아래에서 결과를 확인하고 호출어를 시험할 수 있습니다." : running ? "진행 상태를 자동으로 갱신합니다." : "시작하면 준비된 작업부터 순서대로 진행합니다.");
    if (complete) $("current-description").textContent = "아래에서 모델 성능을 비교하고 시험 오디오를 들어보세요.";
    $("resume-note").hidden = complete;
    const migration = project.migration;
    $("migration-note").hidden = !migration;
    $("migration-reason").textContent = migration?.reason || "";
    $("migrate").hidden = migration?.state !== "available";
    $("migrate").disabled = busy || running;
    if (evaluating(project) && ["summary", "finalizing"].includes(project.progress?.phase)) $("current-description").textContent = "채점을 마쳤습니다. 통계와 결과 파일을 정리하고 있습니다.";
    $("start").hidden = Boolean(migration) || running || complete || waiting || project.status === "paused" || project.status === "failed" || project.status === "error";
    $("resume").hidden = Boolean(migration) || running || complete || waiting || !["paused", "failed", "error", "interrupted"].includes(project.status);
    $("pause").hidden = !running;
    for (const id of ["start", "resume", "pause"]) $(id).disabled = busy || project.status === "pausing";
    const progress = progressValue(project);
    $("progress-section").hidden = !progress;
    if (progress) {
      const stageCount = /단계/.test(progress.unit);
      const unit = { items: "개", samples: "개", files: "개 파일", datasets: "개 자료 묶음", updates: "회 학습", rows: "행", hours: "시간", audio_hours: "시간" }[progress.unit] || progress.unit;
      const quantity = ["bytes", "B", "바이트"].includes(unit) ? `${bytes(progress.done)} / ${bytes(progress.total)}` : `${number(progress.done)} / ${number(progress.total)}${unit}`;
      $("progress-label").textContent = stageCount ? `전체 절차 ${number(progress.done)} / ${number(progress.total)}단계 완료` : `${progress.label ? progress.label + " · " : ""}${quantity}`;
      $("progress-percent").textContent = stageCount ? "" : `${progress.percent.toFixed(1)}%`;
      $("progress").hidden = stageCount; $("progress").value = progress.percent;
    }
    const currentElapsed = duration(project.current_stage_elapsed_seconds), elapsed = duration(project.elapsed_seconds), remaining = eta(project.eta_seconds);
    $("elapsed").textContent = currentElapsed ? `이 작업 경과 ${currentElapsed}${elapsed ? " · 전체 경과 " + elapsed : ""}` : elapsed ? `전체 경과 ${elapsed}` : "";
    if (project.timing) {
      const measured = project.timing;
      const parts = [`전체 ${duration(measured.total_seconds) || "0초"}`];
      if (Number.isFinite(measured.active_seconds)) parts.push(`실제 처리 ${duration(measured.active_seconds)}`);
      if (measured.review_wait_seconds > 0) parts.push(`사용자 확인 대기 ${duration(measured.review_wait_seconds)}`);
      if (measured.paused_seconds > 0) parts.push(`일시정지 ${duration(measured.paused_seconds)}`);
      if (measured.unclassified_seconds > 0) parts.push(`기타 경과 ${duration(measured.unclassified_seconds)}`);
      $("elapsed").textContent = parts.join(" · ");
      if (measured.legacy_estimate) $("elapsed").title = "과거 기록에서 확인 가능한 구간만 분리했습니다.";
    }
    $("eta").textContent = remaining ? `이 작업 예상 남은 시간 ${remaining}` : running ? "예상 시간: 처리 속도를 측정한 뒤 표시합니다" : "";
    const transfer = project.progress || {};
    if (transfer.bytes_per_second > 0 && Number.isFinite(transfer.bytes_per_second)) {
      const speed = `${bytes(transfer.bytes_per_second)}/초`;
      $("eta").textContent = remaining ? `${speed} · 이 작업 예상 남은 시간 ${remaining}` : speed;
    }
    if (transfer.phase === "retry") $("eta").textContent = `연결 재시도 ${number(transfer.attempt)} / ${number(transfer.retry_limit)}`;
    if (evaluating(project)) $("eta").textContent = progress && !/단계/.test(progress.unit) ? "음원 수 기준 · 긴 비호출 음원은 더 오래 걸립니다." : "평가 음원의 무결성과 형식을 확인하고 있습니다.";
    $("project-error").hidden = !project.error;
    $("project-error").textContent = project.error && typeof project.error === "object" ? project.error.message || JSON.stringify(project.error) : project.error || "";
    renderDownloads(project); renderCounts(project); renderReferencePreview(project); renderReview(project); renderInput(project); renderResultFrame(project);
    if (logProject !== project.id) { logProject = project.id; $("log").textContent = "작업 기록 보기를 열면 최신 기록을 불러옵니다."; if ($("log-details").open) loadLog(); }
  }
  function render() {
    const project = selectedProject();
    renderSelector(); renderSystem();
    $("setup").hidden = Boolean(project) && !newProject;
    $("project").hidden = !project || newProject;
    $("create").disabled = busy || !recipeReady || !locationReady();
    if (project && !newProject) renderProject(project);
    else {
      renderReferencePreview(null);
      if (reviewIdentity || resultIdentity) {
        frameSource($("review-frame"), ""); frameSource($("result-frame"), "");
        reviewIdentity = ""; resultIdentity = "";
      }
    }
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    let receivedState = false;
    try {
      const state = await api("/api/state"); receivedState = true; snapshot = state;
      if (!Array.isArray(snapshot.projects)) snapshot.projects = [];
      if (!newProject && !snapshot.projects.some(project => project.id === selectedId)) selectedId = snapshot.active_project_id || snapshot.projects[0]?.id || "";
      $("connection").hidden = true; render();
    } catch (error) {
      const progressPending = error.status === 503 && error.code === "progress_read_pending";
      const delayed = progressPending || error.code === "request_timeout";
      $("connection").hidden = false;
      $("connection").className = "notice" + (delayed ? "" : " error");
      $("connection").setAttribute("role", delayed ? "status" : "alert");
      $("connection").textContent = progressPending
        ? "진행 상태 파일을 읽지 못했습니다. 마지막으로 받은 상태를 표시하며 자동으로 다시 확인합니다."
        : delayed
        ? "진행 상태 확인이 늦어지고 있습니다. 마지막으로 받은 상태를 표시하며 자동으로 다시 확인합니다."
        : (receivedState
        ? "진행 상태를 화면에 표시하지 못했습니다. 페이지를 새로고침하세요. 저장된 작업은 사라지지 않습니다.\n"
        : "서버에 연결하지 못했습니다. 서버를 실행한 창이 열려 있는지 확인하세요. 저장된 작업은 사라지지 않습니다.\n") + error.message;
    } finally { refreshing = false; }
  }
  async function runAction(action) {
    if (busy || !selectedId) return;
    const id = selectedId; busy = true; message(""); render();
    try { await api(`/api/projects/${encodeURIComponent(id)}/${action}`, "POST", {}); await refresh(); }
    catch (error) { message(error.message, true); await refresh(); }
    finally { busy = false; render(); }
  }
  async function create(event) {
    event.preventDefault(); if (busy) return;
    let body;
    try { body = creationBody(); }
    catch (error) { message(error.message, true); return; }
    if (!recipeReady || !locationReady(body)) { previewLocation(); return; }
    busy = true; message("모델 학습 폴더를 만들고 있습니다."); render();
    try {
      const result = await api("/api/projects", "POST", body), project = result.project || result;
      if (!project.id) throw new Error("모델 생성 결과에 식별자가 없습니다. 중복 생성 전에 목록을 확인하세요.");
      selectedId = project.id; newProject = false;
      await refresh();
      if (!activeStatuses.has(project.status)) await api(`/api/projects/${encodeURIComponent(selectedId)}/start`, "POST", {});
      message(""); await refresh();
    } catch (error) { message(error.message, true); await refresh(); }
    finally { busy = false; render(); }
  }
  async function submitInput(event) {
    event.preventDefault(); if (busy) return;
    const project = selectedProject(); if (!project?.input) return;
    if (!["dependency_setup", "hard_phrases", "material_admissions"].includes(project.input.kind)) return;
    const values = project.input.kind === "hard_phrases" ? hardPhraseValues() : $("input-value").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
    const dependencies = project.input.kind === "dependency_setup";
    const sourceMode = project.input.kind === "material_admissions" && $("material-mode").value === "source";
    if (!dependencies && !sourceMode && !values.length) { message("안내에 맞게 한 항목 이상 입력하세요.", true); return; }
    let body = dependencies ? { dependency_backend: $("dependency-backend").value } : project.input.kind === "hard_phrases" ? { phrases: values } : { admissions: values };
    if (sourceMode) {
      const source = { directory: $("source-directory").value.trim(), dataset: $("source-dataset").value, pool: $("source-pool").value };
      if (!source.directory) { message("음원 폴더를 입력하세요.", true); return; }
      if (source.pool !== "rir") {
        source.hours = Number($("source-hours").value);
        if (!Number.isFinite(source.hours) || source.hours <= 0) { message("준비할 시간을 확인하세요.", true); return; }
      }
      for (const [id, key] of [["source-metadata", "metadata"], ["source-url", "source_url"], ["source-license", "license"]]) { const value = $(id).value.trim(); if (value) source[key] = value; }
      if (["mls", "yodas2"].includes(source.dataset) && !source.metadata) { message("메타데이터에 download_receipt.json 경로를 입력하세요.", true); return; }
      body = { source };
    }
    busy = true; message(""); render();
    try { await api(`/api/projects/${encodeURIComponent(project.id)}/input`, "POST", body); await refresh(); }
    catch (error) { message(error.message, true); }
    finally { busy = false; render(); }
  }
  async function loadLog() {
    const id = selectedId; if (!id || !$("log-details").open) return;
    try {
      const result = await api(`/api/projects/${encodeURIComponent(id)}/log`);
      if (id === selectedId) $("log").textContent = typeof result === "string" ? result : result.text || result.log || (Array.isArray(result.lines) ? result.lines.join("\n") : "아직 작업 기록이 없습니다.");
    } catch (error) { if (id === selectedId) $("log").textContent = error.message; }
  }
  async function boot() {
    $("create-form").addEventListener("submit", create);
    for (const id of ["adv-originals", "adv-rounds", "adv-updates", "adv-generic", "adv-korean"]) $(id).addEventListener("input", () => { renderRecipe(); previewLocation(); });
    $("wakeword").addEventListener("input", previewLocation);
    $("input-form").addEventListener("submit", submitInput);
    $("hard-phrase-add").addEventListener("click", () => { if (!busy) addHardPhraseField("", true); });
    $("material-mode").addEventListener("change", () => renderInputMode(selectedProject()));
    $("source-dataset").addEventListener("change", () => {
      const dataset = $("source-dataset").value;
      const pool = { zeroth: "korean_negative", audioset: "noise", mls: "generic_negative", yodas2: "korean_negative", fma: "noise", mit_rir: "rir" }[dataset];
      if (pool) $("source-pool").value = pool;
      if (["mls", "yodas2"].includes(dataset)) $("source-metadata-details").open = true;
      renderInputMode(selectedProject());
    });
    $("source-pool").addEventListener("change", () => renderInputMode(selectedProject()));
    $("project-select").addEventListener("change", () => { selectedId = $("project-select").value; newProject = !selectedId; message(""); render(); if (newProject) previewLocation(); });
    $("new-project").addEventListener("click", () => { newProject = true; message(""); render(); previewLocation(); $("wakeword").focus(); });
    for (const action of ["start", "pause", "resume", "migrate"]) $(action).addEventListener("click", () => runAction(action));
    $("refresh-log").addEventListener("click", loadLog);
    $("log-details").addEventListener("toggle", loadLog);
    $("reference-preview").addEventListener("toggle", () => renderReferencePreview(newProject ? null : selectedProject()));
    $("result-tab").addEventListener("click", () => { resultMode = "results"; if (selectedProject()) renderResultFrame(selectedProject()); });
    $("mic-tab").addEventListener("click", () => { resultMode = "mic"; if (selectedProject()) renderResultFrame(selectedProject()); });
    $("analysis-tab").addEventListener("click", () => { resultMode = "analysis"; if (selectedProject()) renderResultFrame(selectedProject()); });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
    await refresh();
    try {
      const result = await api("/api/recipes"); recipes = Array.isArray(result.presets) ? result.presets : [];
      if (!recipes.some(recipe => recipe.id === selectedPreset)) selectedPreset = recipes[0]?.id || "check";
      recipeReady = recipes.length > 0; renderPresets(); renderRecipe(); previewLocation();
    } catch (error) { message("작업 규모를 불러오지 못했습니다. " + error.message, true); }
    pollTimer = setInterval(async () => { if (!document.hidden) { await refresh(); if ($("log-details").open) await loadLog(); } }, 3000);
    window.addEventListener("pagehide", () => { clearInterval(pollTimer); clearTimeout(locationTimer); locationRevision++; }, { once: true });
  }
  window.KowwStudio = Object.freeze({ duration, bytes, storageEstimate, storageBudget, eta, progressValue, requestBody, localUrl, webUrl });
  if (!window.KOWW_STUDIO_TEST) boot();
})();
