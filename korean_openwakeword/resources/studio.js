"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const names = { check: "동작 확인용", trial: "시험 모델용", production: "상용 모델용" };
  const steps = [
    ["data", "자료 준비"], ["references", "참조 음색"], ["pilot", "호출어 확인"],
    ["generate", "음성 생성"], ["features", "증강·특징 추출"], ["train", "모델 학습"],
    ["evaluate", "성능 평가"], ["test", "결과·시험"]
  ];
  const stepGroups = {
    data: ["data", "frontend", "data_download", "data_review", "data_connect"],
    references: ["references", "references_review"], pilot: ["pilot", "pilot_review"],
    generate: ["generate", "positives", "hard_phrases", "hard_generation", "hard_review", "hard_connect"],
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
    const unit = Math.min(3, Math.floor(Math.log(size) / Math.log(1024)));
    return `${(size / 1024 ** unit).toLocaleString("ko-KR", { maximumFractionDigits: 1 })} ${["B", "KiB", "MiB", "GiB"][unit]}`;
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
  function requestBody(form, preset) {
    const result = { wakeword: form.wakeword.trim(), preset };
    if (form.directory.trim()) result.directory = form.directory.trim();
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
  let reviewIdentity = "", inputIdentity = "", resultIdentity = "", resultMode = "results", logProject = "";
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
      if (!response.ok) throw new Error(body.error?.message || body.error || body.message || `요청에 실패했습니다 (${response.status}).`);
      return body;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("서버 응답이 늦어지고 있습니다. 진행 상태를 다시 확인하세요. 같은 작업을 중복 실행하지 않습니다.");
      throw error;
    } finally { clearTimeout(timer); }
  }
  function message(text, error = false) {
    $("message").textContent = text || "";
    $("message").hidden = !text;
    $("message").className = "notice" + (error ? " error" : "");
  }
  function renderPresets() {
    const nodes = recipes.map(recipe => {
      const label = element("label", undefined, "preset");
      const radio = element("input"); radio.type = "radio"; radio.name = "preset"; radio.value = recipe.id; radio.checked = selectedPreset === recipe.id;
      radio.addEventListener("change", () => { selectedPreset = recipe.id; renderRecipe(); });
      const title = element("span", recipe.label || names[recipe.id] || recipe.id, "preset-title");
      const heading = element("div"); heading.append(radio, title);
      const detail = element("p", recipe.description || "", "preset-description");
      const summary = element("p", undefined, "preset-summary");
      const originals = recipe.originals ?? recipe.positive_originals, rounds = recipe.rounds ?? recipe.augmentation_rounds, updates = recipe.updates;
      if (originals !== undefined) summary.append(element("span", `호출어 ${number(originals)}개`));
      if (rounds !== undefined) summary.append(element("span", `증강 R${number(rounds)}`));
      if (updates !== undefined) summary.append(element("span", `학습 ${number(updates)}회 / 모델`));
      label.append(heading, detail, summary); return label;
    });
    replace($("presets"), nodes.length ? nodes : [element("p", "사용 가능한 작업 규모가 없습니다. 서버 설정을 확인하세요.", "muted")]);
  }
  function originalSize(value) {
    if (!Number.isFinite(value) || value <= 0) return "";
    return value >= 1e9 ? `${(value / 1e9).toFixed(2)} GB` : value >= 1e6 ? `${(value / 1e6).toFixed(1)} MB` : `${Math.round(value / 1e3)} KB`;
  }
  function materialGroups(recipe) {
    const hours = value => `${number(value)}시간`, count = value => `${number(value)}개`;
    const selected = (id, fallback) => $(id).value !== "" && Number.isFinite(Number($(id).value)) && Number($(id).value) >= 0 ? Number($(id).value) : fallback;
    const originals = selected("adv-originals", recipe.originals ?? recipe.positive_originals);
    const genericHours = selected("adv-generic", recipe.generic_hours ?? recipe.automatic_target_hours?.generic_nominal ?? 0);
    const koreanHours = selected("adv-korean", recipe.korean_hours ?? recipe.automatic_target_hours?.korean_train ?? 0);
    const policy = recipe.evaluation_policy || recipe.evaluation_materials_policy || {};
    const assets = recipe.assets || [], sources = recipe.sources || assets;
    const asset = id => sources.find(item => item.id === id) || assets.find(item => item.id === id);
    const size = item => item?.bytes ?? item?.size_bytes ?? item?.total_bytes ?? item?.totalbytes;
    const bundleSize = ids => {
      const values = ids.map(id => size(asset(id))).filter(Number.isFinite);
      return values.length ? values.reduce((sum, value) => sum + value, 0) : undefined;
    };
    const source = (name, content, amount, rank, record, fileBytes = size(record)) => ({ name, content, amount, rank, bytes: fileBytes, url: record?.url || record?.source_url });
    const fallback = record => {
      const time = record.target_hours ?? record.hours;
      return source(record.display_name || record.name || record.id || record.dataset_id || "공개 자료",
        record.description || backgroundTypeNames(record).join(" · ") || "공개 오디오 자료",
        time !== undefined ? hours(time) : record.file_count !== undefined ? count(record.file_count) : "1묶음",
        time === undefined ? null : time * 3600, record);
    };
    const suppliedGeneric = recipe.automatic_target_hours?.generic_nominal ?? asset("generic")?.hours ?? 0;
    const suppliedKorean = recipe.automatic_target_hours?.korean_train ?? asset("zeroth")?.hours ?? 0;
    const generic = asset("generic") ? [source("openWakeWord", "일반 음성·음악·환경음의 특징 데이터", hours(suppliedGeneric), suppliedGeneric * 3600, asset("generic"))] : [];
    const korean = asset("zeroth") ? [source("Zeroth Korean", "한국어 낭독 음성", hours(suppliedKorean), suppliedKorean * 3600, asset("zeroth"))] : [];
    for (const [items, target, available, content] of [[generic, genericHours, suppliedGeneric, "일반 비호출 음원·특징 자료"], [korean, koreanHours, suppliedKorean, "한국어 대화·낭독 등 음성"]]) {
      if (target > available) items.push(source("추가 자료", `직접 준비할 ${content}`, hours(target - available), (target - available) * 3600));
    }
    const noise = [];
    if (asset("fma")) noise.push(source("FMA", "음악 120곡 · 곡당 30초", "약 1시간", 3600, asset("fma"), bundleSize(["fma", "fma_metadata"])));
    if (asset("zeroth")) noise.push(source("Zeroth Korean", "학습용 한국어 낭독 재사용", hours(suppliedKorean), suppliedKorean * 3600, asset("zeroth")));
    if (asset("dns_fan")) noise.push(source("DNS-Challenge", "팬 작동 소리", "10초", 10, asset("dns_fan")));
    if (asset("audioset")) noise.push(source("AudioSet", "여러 종류의 일반 오디오", "1묶음", null, asset("audioset"), bundleSize(["audioset", "audioset_segments", "audioset_ontology"])));
    const mit = asset("mit_rir") || recipe.additional_download;
    const training = [
      { id: "positive", name: "호출어 음성", description: "호출어 합성 음성", amount: count(originals), sources: [source("Qwen TTS", "호출어만 말한 음성", count(originals), originals)] },
      { id: "hard", name: "유사어 음성", description: "호출어로 오인할 수 있는 다른 단어·표현", amount: count(recipe.hard_originals ?? 0), sources: [source("Qwen TTS", "헷갈리기 쉬운 단어·표현을 합성한 음성", count(recipe.hard_originals ?? 0), recipe.hard_originals ?? 0)] },
      { id: "generic", name: "일반 비호출 자료", description: "일반 음성 · 음악 · 환경음", amount: hours(genericHours), sources: generic },
      { id: "korean", name: "한국어 비호출 자료", description: koreanHours > suppliedKorean ? "한국어 대화 · 낭독 등 음성" : "한국어 낭독 음성", amount: hours(koreanHours), sources: korean },
      { id: "noise", name: "소음 증강 음원", description: "일반 오디오 · 음악 · 팬 소음 · 한국어 음성", amount: `${number(noise.length)}종`, sources: noise },
      { id: "rir", name: "잔향 자료", description: "공간의 울림을 재현하는 파일", amount: mit ? count(mit.files ?? mit.file_count) : "0개", sources: mit ? [source("MIT RIR", "공간의 울림 특성", count(mit.files ?? mit.file_count), mit.files ?? mit.file_count, mit)] : [] }
    ];
    const isEvaluation = item => ["validation", "development"].includes(item.split) || /held_out|evaluation|validation/.test(item.role || "");
    const knownTrain = new Set(["generic", "zeroth", "fma", "audioset", "dns_fan", "mit_rir"]);
    for (const item of sources.filter(item => !isEvaluation(item) && item.role !== "provider_metadata" && !knownTrain.has(item.id))) {
      training.push({ id: `extra-${training.length}`, name: item.display_name || item.name || item.id || "공개 자료", description: item.description || "추가 학습 자료", amount: fallback(item).amount, sources: [fallback(item)] });
    }
    const buckets = policy.negative_hours_by_bucket || {};
    const evalSources = [...sources.filter(isEvaluation), ...(recipe.evaluation_sources || [])]
      .filter(item => policy.id !== "common_development_v4" || item.id === "openslr28_common_rir_v1" || Object.keys(buckets).some(key => item.id === `common_30h_${key}`));
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
    // Older policies retain their supplemental sources; v4 filters them above.
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
          if (size) quantity.append(element("span", `원본 ${size}`, "file-size"));
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
    replace(selector, projects.map(project => { const option = element("option", project.wakeword || project.id); option.value = project.id; return option; }));
    if (newProject) { const option = element("option", "새 프로젝트"); option.value = ""; selector.append(option); selector.value = ""; }
    else selector.value = selectedId;
    selector.hidden = !projects.length; $("new-project").hidden = !projects.length;
    selector.disabled = busy; $("new-project").disabled = busy;
  }
  function renderStages(project) {
    const provided = Array.isArray(project.stages) ? project.stages : [];
    const stageId = typeof project.stage === "object" ? project.stage.id : provided.find(item => item.id === project.stage || item.label === project.stage)?.id || project.stage;
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
    const labels = { tts_generated: "음성 생성", positive_generated: "음성 생성", technical_passed: "자동 검사 통과", reviewed: "사람 청취", admitted: "학습 사용 판정", trained_models: "학습 완료 모델" };
    // Only supplied counters are shown; generation is never called human approval.
    replace($("counts"), records.map(record => { const item = element("span"); item.append(element("span", (labels[record.label || record.id] || record.label || record.id || "자료") + " "), element("strong", `${typeof record.value === "number" ? number(record.value) : String(record.value ?? "—")}${record.total !== undefined ? " / " + number(record.total) : ""}${record.unit || ""}`)); return item; }));
    $("counts").hidden = !records.length;
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
  function renderReview(project) {
    const review = project.review;
    const url = localUrl(review?.url);
    const show = Boolean(review && url && !activeStatuses.has(project.status));
    $("review-section").hidden = !show;
    if (!show) { if (reviewIdentity) frameSource($("review-frame"), ""); reviewIdentity = ""; return; }
    const identity = project.id + ":" + (review.queue_id || review.id || url);
    $("review-title").textContent = review.title || "오디오 검수";
    $("review-description").textContent = review.description || "끝까지 들으면 자동으로 보존됩니다. 문제가 있는 항목만 바꾼 뒤 검수 완료·저장을 누르세요.";
    if (reviewIdentity !== identity) { frameSource($("review-frame"), url); reviewIdentity = identity; }
  }
  function renderInput(project) {
    const input = project.input;
    const show = Boolean(input && !activeStatuses.has(project.status));
    $("input-section").hidden = !show;
    if (!show) { inputIdentity = ""; return; }
    const identity = project.id + ":" + (input.id || input.kind);
    const hard = input.kind === "hard_phrases";
    const dependencies = input.kind === "dependency_setup";
    const editable = dependencies || hard || input.kind === "material_admissions";
    $("input-form").hidden = !editable;
    $("input-title").textContent = input.title || (!editable ? "작업을 확인해주세요" : dependencies ? "학습 도구를 준비해주세요" : hard ? "유사어 후보를 정해주세요" : "추가 자료를 연결해주세요");
    $("input-description").textContent = input.description || (!editable ? "자동으로 해결할 수 없는 문제가 있어 멈췄습니다. 아래 작업 기록에서 원인과 필요한 조치를 확인하세요." : dependencies ? "이 가상 환경에 필요한 음성 생성·학습 도구를 설치합니다. 사용할 연산 환경을 선택하세요." : hard ? "호출어와 헷갈릴 수 있는 표현을 확인합니다. 호출어 자체나 같은 낱말이 통째로 포함된 표현은 넣지 않습니다." : "자동으로 준비할 수 없는 자료가 있습니다. 안내에 맞는 검수·승인 기록을 연결하면 이어서 진행합니다.");
    $("input-label").textContent = input.label || (dependencies ? "설치할 연산 도구" : hard ? "유사어 · 한 줄에 하나씩" : "자료 승인 기록 · 한 줄에 경로 하나씩");
    $("input-label").hidden = false; $("input-hint").hidden = false;
    $("input-label").htmlFor = dependencies ? "dependency-backend" : "input-value";
    $("input-value").hidden = dependencies; $("input-value").required = !dependencies;
    $("dependency-backend").hidden = !dependencies;
    $("input-submit").textContent = dependencies ? "선택한 도구 설치하고 계속" : "저장하고 이어서 진행";
    $("input-hint").textContent = input.hint || (dependencies ? "현재 가상 환경에만 설치합니다. NVIDIA GPU가 없다면 CPU용을 선택하세요." : hard ? "확인한 후보로 소수의 검수용 음성을 먼저 만듭니다." : "직접 준비한 자료의 승인 기록이 필요합니다. 원본 음원 경로만 입력하면 승인된 것으로 처리하지 않습니다.");
    if (identity !== inputIdentity) {
      $("input-value").value = (input.phrases || input.values || input.suggestions || []).join("\n");
      $("dependency-backend").value = input.recommended_backend === "cu128" ? "cu128" : "cpu";
      $("material-mode").value = "source";
      const source = input.source || {};
      $("source-directory").value = source.directory || ""; $("source-dataset").value = source.dataset || "zeroth";
      $("source-pool").value = source.pool || input.pool || "korean_negative";
      $("source-hours").value = String(source.hours ?? input.remaining_hours ?? input.required_hours ?? input.target_hours ?? "");
      for (const [id, key] of [["source-metadata", "metadata"], ["source-url", "source_url"], ["source-license", "license"]]) $(id).value = source[key] || "";
      inputIdentity = identity;
    }
    renderInputMode(project);
    $("input-submit").disabled = busy;
  }
  function renderInputMode(project) {
    const material = project?.input?.kind === "material_admissions";
    const source = material && $("material-mode").value === "source";
    const rir = source && $("source-pool").value === "rir";
    $("material-controls").hidden = !material; $("source-fields").hidden = !source;
    $("source-directory").required = source;
    $("source-hours").required = source && !rir;
    $("source-hours-label").hidden = rir;
    $("source-rir-note").hidden = !rir;
    if (!material) return;
    $("input-label").hidden = source; $("input-value").hidden = source; $("input-value").required = !source;
    $("input-hint").hidden = source;
    $("input-submit").textContent = source ? "자료 준비하고 검수하기" : "승인 자료 연결하고 계속";
  }
  function renderResultFrame(project) {
    const result = project.results || {};
    const resultUrl = localUrl(result.url || project.results_url), micUrl = localUrl(result.mic_url || project.mic_url);
    const show = Boolean(resultUrl);
    $("results-section").hidden = !show;
    if (!show) { if (resultIdentity) frameSource($("result-frame"), ""); $("evaluation-coverage").hidden = true; resultIdentity = ""; resultMode = "results"; return; }
    if (resultIdentity !== project.id + ":" + resultUrl) { resultMode = "results"; resultIdentity = project.id + ":" + resultUrl; }
    if (!micUrl) resultMode = "results";
    $("mic-tab").hidden = !micUrl;
    $("result-tab").className = "tab" + (resultMode === "results" ? " active" : "");
    $("mic-tab").className = "tab" + (resultMode === "mic" ? " active" : "");
    $("result-tab").setAttribute("aria-selected", String(resultMode === "results")); $("mic-tab").setAttribute("aria-selected", String(resultMode === "mic"));
    const url = resultMode === "mic" ? micUrl : resultUrl;
    frameSource($("result-frame"), url); $("result-open").href = url;
    $("result-frame").title = resultMode === "mic" ? "마이크로 호출어 시험" : "성능과 평가 오디오";
    $("results-note").textContent = resultMode === "mic" ? "마이크 허용 후 시작하세요. 이 시험 음성을 학습이나 설정 조정에 자동 사용하지 않습니다." : result.note || result.scope || "채점 결과와 실제 평가 오디오를 함께 확인하세요. 이 결과만으로 모든 화자와 환경의 성능을 보장하지 않습니다.";
    const assessment = result.common_evaluation_assessment;
    $("evaluation-coverage").hidden = !assessment || resultMode === "mic";
    if (assessment) {
      const shortages = Array.isArray(assessment.shortages) ? assessment.shortages : [];
      $("evaluation-coverage-summary").textContent = assessment.coverage_targets_passed ? "공통 평가의 자료·표본 기준 충족 · 독립 최종 평가는 별도" : `공통 평가의 자료·표본 기준 미충족 · 부족 항목 ${number(shortages.length)}개 보기`;
      replace($("evaluation-shortages"), shortages.map(item => element("li", item)));
    }
  }
  function renderProject(project) {
    const running = activeStatuses.has(project.status), complete = completeStatuses.has(project.status), waiting = Boolean(project.review || project.input) || /wait|need|await/.test(project.status || "");
    $("project-title").textContent = `‘${project.wakeword || project.id}’`;
    $("project-purpose").textContent = names[project.preset] || project.preset_label || "호출어 학습";
    $("project-directory").textContent = project.directory || project.root || project.path || "";
    $("project-status").textContent = statuses[project.status] || project.status_label || "상태 확인 중";
    $("project-status").className = "status-pill" + (complete ? " complete" : project.error ? " failed" : waiting ? " waiting" : running ? " running" : "");
    renderStages(project);
    const stageLabel = typeof project.stage === "object" ? project.stage.label : steps.find(([id]) => id === project.stage)?.[1];
    $("current-category").textContent = waiting ? "사용자 확인 대기" : complete ? "완료" : "현재 작업";
    $("current-title").textContent = project.stage_label || project.title || stageLabel || (complete ? "학습과 평가를 마쳤습니다" : waiting ? "안내를 확인해주세요" : "시작할 준비가 되었습니다");
    $("current-description").textContent = project.next_action || project.description || (waiting ? "아래에서 필요한 확인을 마치면 이어서 진행합니다." : complete ? "아래에서 결과를 확인하고 호출어를 시험할 수 있습니다." : running ? "진행 상태를 자동으로 갱신합니다." : "시작하면 준비된 작업부터 순서대로 진행합니다.");
    $("start").hidden = running || complete || waiting || project.status === "paused" || project.status === "failed" || project.status === "error";
    $("resume").hidden = running || complete || waiting || !["paused", "failed", "error", "interrupted"].includes(project.status);
    $("pause").hidden = !running;
    for (const id of ["start", "resume", "pause"]) $(id).disabled = busy || project.status === "pausing";
    const progress = progressValue(project);
    $("progress-section").hidden = !progress;
    if (progress) {
      const stageCount = /단계/.test(progress.unit);
      const unit = { items: "개", samples: "개", files: "개 파일", datasets: "개 자료 묶음", updates: "회 학습", rows: "행" }[progress.unit] || progress.unit;
      const quantity = ["bytes", "B", "바이트"].includes(unit) ? `${bytes(progress.done)} / ${bytes(progress.total)}` : `${number(progress.done)} / ${number(progress.total)}${unit}`;
      $("progress-label").textContent = stageCount ? `전체 절차 ${number(progress.done)} / ${number(progress.total)}단계 완료` : `${progress.label ? progress.label + " · " : ""}${quantity}`;
      $("progress-percent").textContent = stageCount ? "" : `${progress.percent.toFixed(1)}%`;
      $("progress").hidden = stageCount; $("progress").value = progress.percent;
    }
    const currentElapsed = duration(project.current_stage_elapsed_seconds), elapsed = duration(project.elapsed_seconds), remaining = eta(project.eta_seconds);
    $("elapsed").textContent = currentElapsed ? `이 작업 경과 ${currentElapsed}${elapsed ? " · 전체 경과 " + elapsed : ""}` : elapsed ? `전체 경과 ${elapsed}` : "";
    $("eta").textContent = remaining ? `이 작업 예상 남은 시간 ${remaining}` : running ? "예상 시간: 처리 속도를 측정한 뒤 표시합니다" : "";
    $("project-error").hidden = !project.error; $("project-error").textContent = typeof project.error === "object" ? project.error.message || JSON.stringify(project.error) : project.error || "";
    renderCounts(project); renderReview(project); renderInput(project); renderResultFrame(project);
    if (logProject !== project.id) { logProject = project.id; $("log").textContent = "작업 기록 보기를 열면 최신 기록을 불러옵니다."; if ($("log-details").open) loadLog(); }
  }
  function render() {
    const project = selectedProject();
    renderSelector(); renderSystem();
    $("setup").hidden = Boolean(project) && !newProject;
    $("project").hidden = !project || newProject;
    $("create").disabled = busy || !recipeReady;
    if (project && !newProject) renderProject(project);
    else if (reviewIdentity || resultIdentity) {
      frameSource($("review-frame"), ""); frameSource($("result-frame"), "");
      reviewIdentity = ""; resultIdentity = "";
    }
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      const state = await api("/api/state"); snapshot = state;
      if (!Array.isArray(snapshot.projects)) snapshot.projects = [];
      if (!newProject && !snapshot.projects.some(project => project.id === selectedId)) selectedId = snapshot.active_project_id || snapshot.projects[0]?.id || "";
      $("connection").hidden = true; render();
    } catch (error) {
      $("connection").hidden = false;
      $("connection").textContent = "서버에 연결하지 못했습니다. 서버를 실행한 창이 열려 있는지 확인하세요. 저장된 작업은 사라지지 않습니다.\n" + error.message;
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
    try { body = requestBody({ wakeword: $("wakeword").value, directory: $("directory").value, advanced: { positive_originals: $("adv-originals").value, augmentation_rounds: $("adv-rounds").value, updates: $("adv-updates").value, generic_hours: $("adv-generic").value, korean_hours: $("adv-korean").value } }, selectedPreset); }
    catch (error) { message(error.message, true); return; }
    busy = true; message("프로젝트를 만들고 있습니다."); render();
    try {
      const result = await api("/api/projects", "POST", body), project = result.project || result;
      if (!project.id) throw new Error("프로젝트 생성 결과에 식별자가 없습니다. 중복 생성 전에 목록을 확인하세요.");
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
    const values = $("input-value").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
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
    for (const id of ["adv-originals", "adv-generic", "adv-korean"]) $(id).addEventListener("input", renderRecipe);
    $("input-form").addEventListener("submit", submitInput);
    $("material-mode").addEventListener("change", () => renderInputMode(selectedProject()));
    $("source-dataset").addEventListener("change", () => { const pool = { zeroth: "korean_negative", audioset: "noise", fma: "noise", mit_rir: "rir" }[$("source-dataset").value]; if (pool) $("source-pool").value = pool; renderInputMode(selectedProject()); });
    $("source-pool").addEventListener("change", () => renderInputMode(selectedProject()));
    $("project-select").addEventListener("change", () => { selectedId = $("project-select").value; newProject = !selectedId; message(""); render(); });
    $("new-project").addEventListener("click", () => { newProject = true; message(""); render(); $("wakeword").focus(); });
    for (const action of ["start", "pause", "resume"]) $(action).addEventListener("click", () => runAction(action));
    $("refresh-log").addEventListener("click", loadLog);
    $("log-details").addEventListener("toggle", loadLog);
    $("result-tab").addEventListener("click", () => { resultMode = "results"; if (selectedProject()) renderResultFrame(selectedProject()); });
    $("mic-tab").addEventListener("click", () => { resultMode = "mic"; if (selectedProject()) renderResultFrame(selectedProject()); });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
    await refresh();
    try {
      const result = await api("/api/recipes"); recipes = Array.isArray(result.presets) ? result.presets : [];
      if (!recipes.some(recipe => recipe.id === selectedPreset)) selectedPreset = recipes[0]?.id || "check";
      if (result.default_directory) $("directory").placeholder = result.default_directory;
      if (result.storage_note) $("storage-note").textContent = result.storage_note;
      recipeReady = recipes.length > 0; renderPresets(); renderRecipe(); render();
    } catch (error) { message("작업 규모를 불러오지 못했습니다. " + error.message, true); }
    pollTimer = setInterval(async () => { if (!document.hidden) { await refresh(); if ($("log-details").open) await loadLog(); } }, 3000);
    window.addEventListener("pagehide", () => clearInterval(pollTimer), { once: true });
  }
  window.KowwStudio = Object.freeze({ duration, bytes, eta, progressValue, requestBody, localUrl, webUrl });
  if (!window.KOWW_STUDIO_TEST) boot();
})();
