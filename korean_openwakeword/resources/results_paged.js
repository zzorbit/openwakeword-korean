/* Frozen results, loaded in bounded pages. No inference or audio gain. */
"use strict";
(() => {
  const {report, viewer} = window.KOWW_RESULTS;
  const ui = window.KowwResultsUI;
  const $ = id => document.getElementById(id);
  const labels = {clean: "조용한 원본", environment: "환경소음", music: "음악·전자음", competing_speech: "다른 사람 말소리"};
  const OPTION_LIMIT = 100, EXTRA_LIMIT = 99, CACHE_LIMIT = 2;
  const cache = new Map(), pending = new Map();
  const audio = $("audio");
  let metadata = [], queue = [], extras = [], extraPage = 0;
  let index = 0, continuous = false, token = 0, optionPage = -1, currentSource = "main";
  const option = (select, value, text) => { const node = document.createElement("option"); node.value = String(value); node.textContent = text; select.append(node); };
  function summary() {
    $("title").textContent = `${report.wakeword} · 성능 시험 결과`;
    ui.renderSummary(report);
    ui.renderConditions(report);
  }
  async function json(path) {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`평가 자료를 불러오지 못했습니다 (${response.status}).`);
    return response.json();
  }
  async function detail(sourceIndex) {
    const page = Math.floor(sourceIndex / viewer.page_size);
    let rows;
    if (cache.has(page)) { rows = cache.get(page); cache.delete(page); cache.set(page, rows); }
    else {
      if (!pending.has(page)) {
        const pagePath = new URL(`clips/page-${String(page).padStart(6, "0")}.json`, new URL(viewer.index_path, window.location.href));
        const request = json(pagePath.href).then(value => {
          if (value.start !== page * viewer.page_size || !Array.isArray(value.clips) || value.clips.length > viewer.page_size) throw new Error("평가 자료 페이지 형식이 올바르지 않습니다.");
          cache.set(page, value.clips);
          while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
          return value.clips;
        }).finally(() => pending.delete(page));
        pending.set(page, request);
      }
      rows = await pending.get(page);
    }
    const clip = rows[sourceIndex % viewer.page_size];
    if (!clip || clip.sample_id !== metadata[sourceIndex][0]) throw new Error("평가 자료 목록과 오디오가 일치하지 않습니다.");
    return clip;
  }
  function show(clip) {
    audio.src = clip.path;
    $("clipTitle").textContent = ui.clipLabel(clip);
    const background = evaluationBackgroundTypeText(clip);
    $("clipInfo").textContent = `${clip.duration_s.toFixed(2)}초${clip.calls?.length ? ` · 호출어 위치 ${clip.calls[0].start_s.toFixed(2)}–${clip.calls[0].end_s.toFixed(2)}초` : " · 호출어 없음"}${background ? ` · 배경: ${background}` : ""}`;
    $("download").href = clip.path; $("download").download = "";
    $("hash").textContent = `SHA256 ${clip.sha256}`;
    $("sample-id").textContent = `파일 이름: ${clip.sample_id}`;
    ui.renderResponses(report, clip);
  }
  function progress() {
    $("queue").textContent = queue.length ? `${(index + 1).toLocaleString("ko-KR")} / ${queue.length.toLocaleString("ko-KR")}개` : "목록 없음";
    $("prev").disabled = index <= 0; $("next").disabled = index >= queue.length - 1;
    const page = Math.floor(index / OPTION_LIMIT);
    if (page !== optionPage) {
      optionPage = page; $("item").replaceChildren();
      for (let position = page * OPTION_LIMIT; position < Math.min(queue.length, (page + 1) * OPTION_LIMIT); position++) { const clip = metadata[queue[position]]; option($("item"), position, `${position + 1} · 음성 ${clip[6] || clip[5]} · ${labels[clip[2]] || clip[2]}`); }
    }
    $("item").value = String(index);
    $("original").value = queue.length ? String(metadata[queue[index]][6] || "") : "";
  }
  async function activate(sourceIndex, autoplay = false, source = "main") {
    const mine = ++token; audio.pause(); currentSource = source;
    $("playState").textContent = "오디오 정보를 불러오는 중…";
    try {
      const clip = await detail(sourceIndex);
      if (mine !== token) return;
      show(clip);
      $("playState").textContent = source === "extra" ? "추가 자료 · 재생 버튼을 눌러 들으세요." : "재생 버튼을 눌러 들으세요.";
      if (autoplay) { await audio.play(); if (mine === token) $("playState").textContent = "재생 중 · 마지막 항목에서 종료합니다."; }
    } catch (error) { if (mine === token) { continuous = false; $("playState").textContent = error.message + " 다시 선택하거나 이어 듣기를 눌러주세요."; } }
  }
  function move(position, autoplay = continuous) {
    if (!Number.isInteger(position) || position < 0 || position >= queue.length) return;
    index = position; progress(); return activate(queue[index], autoplay);
  }
  function rebuild() {
    continuous = false; token++; audio.pause(); optionPage = -1;
    queue = [];
    for (let position = 0; position < metadata.length; position++) { const clip = metadata[position]; if (clip[1] === "positive" && clip[2] !== "clean" && Number(clip[3]) === Number($("snr").value) && ($("category").value === "all" || clip[2] === $("category").value) && (clip[4] || "none") === $("reverb").value) queue.push(position); }
    const categoryOrder = ["environment", "music", "competing_speech"];
    queue.sort((left, right) => { const a = metadata[left], b = metadata[right]; return (a[6] || 0) - (b[6] || 0) || String(a[5]).localeCompare(String(b[5])) || categoryOrder.indexOf(a[2]) - categoryOrder.indexOf(b[2]) || left - right; });
    index = 0; progress();
    if (queue.length) return activate(queue[0]);
    audio.removeAttribute("src"); $("responses").tBodies[0].replaceChildren(); $("clipTitle").textContent = "선택한 조건으로 시험한 오디오가 없습니다."; $("playState").textContent = "";
  }
  function renderExtras() {
    const start = extraPage * EXTRA_LIMIT, end = Math.min(extras.length, start + EXTRA_LIMIT);
    $("extra").replaceChildren(); option($("extra"), "", "추가 자료 선택");
    for (let position = start; position < end; position++) { const clip = metadata[extras[position]]; option($("extra"), position, `${position + 1} · ${ui.clipLabel({kind:clip[1],category:clip[2],original_order:clip[6]})}`); }
    $("extra-range").textContent = extras.length ? `${start + 1}–${end} / ${extras.length}개` : "추가 자료 없음";
    $("extra-prev").disabled = extraPage <= 0; $("extra-next").disabled = end >= extras.length;
  }
  function bind() {
    $("start").onclick = () => { continuous = true; return move(0, true); };
    $("resume").onclick = () => { continuous = true; return move(index, true); };
    $("pause").onclick = () => { continuous = false; token++; audio.pause(); $("playState").textContent = "일시정지"; };
    audio.onended = () => { if (!continuous || currentSource !== "main") return; if (index + 1 < queue.length) return move(index + 1, true); continuous = false; $("playState").textContent = "선택한 목록을 모두 들었습니다."; };
    audio.onerror = () => { continuous = false; $("playState").textContent = "오디오를 불러올 수 없습니다. 파일 경로와 서버 상태를 확인하세요."; };
    $("prev").onclick = () => move(index - 1); $("next").onclick = () => move(index + 1);
    $("item").onchange = () => move(Number($("item").value));
    const original = () => { const value = Number($("original").value), position = queue.findIndex(sourceIndex => metadata[sourceIndex][6] === value); if (position >= 0) return move(position); $("playState").textContent = "이 조건에 해당하는 음성 번호가 없습니다."; };
    $("original-go").onclick = original; $("original").onchange = original;
    for (const id of ["snr", "category", "reverb"]) $(id).onchange = rebuild;
    $("extra-prev").onclick = () => { if (extraPage > 0) { extraPage--; renderExtras(); } };
    $("extra-next").onclick = () => { if ((extraPage + 1) * EXTRA_LIMIT < extras.length) { extraPage++; renderExtras(); } };
    $("extra").onchange = () => { if ($("extra").value === "") return; const position = Number($("extra").value); if (Number.isInteger(position) && extras[position] !== undefined) { continuous = false; return activate(extras[position], false, "extra"); } };
  }
  async function boot() {
    summary(); bind();
    if (viewer?.schema_version !== 1 || !Number.isInteger(viewer.page_size) || viewer.page_size < 1 || viewer.page_size > 128) throw new Error("지원하지 않는 평가 화면 형식입니다.");
    const data = await json(viewer.index_path);
    if (data.schema_version !== 1 || !Array.isArray(data.clips)) throw new Error("평가 목록 형식이 올바르지 않습니다.");
    metadata = data.clips;
    $("scope").textContent = data.scope_text; $("noiseCoverage").textContent = data.noise_coverage_text;
    const levels = new Set(), reverbGroups = new Set();
    for (let position = 0; position < metadata.length; position++) { const clip = metadata[position]; if (clip[1] === "positive") { reverbGroups.add(clip[4] || "none"); if (clip[2] !== "clean") levels.add(clip[3]); } if (clip[2] === "clean" || clip[1] === "negative") extras.push(position); }
    [...levels].sort((a, b) => b - a).forEach(level => option($("snr"), level, ui.snrLabel(level)));
    if (levels.has(10)) $("snr").value = "10";
    for (const node of $("reverb").options) node.disabled = !reverbGroups.has(node.value);
    renderExtras(); await rebuild();
  }
  window.KowwResults = {boot, detail, rebuild, move, state: () => ({index, queueLength: queue.length, cachePages: [...cache.keys()], pendingPages: [...pending.keys()], source: currentSource, continuous}), OPTION_LIMIT, CACHE_LIMIT};
  if (!window.KOWW_RESULTS_TEST) boot().catch(error => { $("playState").textContent = error.message; });
})();
