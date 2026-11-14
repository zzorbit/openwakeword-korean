/* Frozen results, loaded in bounded pages. No inference or audio gain. */
"use strict";
(() => {
  const chapter = window.KowwChapterReady;
  const {report, viewer} = window.KOWW_RESULTS;
  const ui = window.KowwResultsUI;
  const $ = id => document.getElementById(id);
  const OPTION_LIMIT = 100, CACHE_LIMIT = 2;
  const cache = new Map(), pending = new Map();
  const audio = $("audio");
  let metadata = [], folder = null, index = 0, continuous = false;
  let token = 0, playRequest = null, loadedIndex = null, playing = false, loading = false;
  let updatePicker, selection;

  function describe([, kind, category, snr_db, reverb_condition, original_id, original_order]) {
    return {kind, category, snr_db, reverb_condition, original_id, original_order};
  }

  function transport() {
    const active = playing || loading;
    $("resume").textContent = active ? "일시정지" : "재생";
    $("resume").setAttribute("aria-pressed", String(active));
    $("resume").setAttribute("aria-busy", String(loading));
    $("resume").disabled = !folder;
  }

  function stop() {
    token++;
    continuous = false; playRequest = null; loading = false; playing = false;
    audio.pause();
    transport();
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

  function progress() {
    const length = folder?.clips.length || 0;
    $("queue").textContent = length ? `${(index + 1).toLocaleString("ko-KR")} / ${length.toLocaleString("ko-KR")}` : "목록 없음";
    $("prev").disabled = index <= 0;
    $("next").disabled = index + 1 >= length;
    updatePicker?.(folder, index);
  }

  async function activate(sourceIndex, {autoplay = false, resume = false} = {}) {
    const mine = ++token;
    loading = true; playing = false; playRequest = null;
    if (!resume) audio.pause();
    $("playState").textContent = "";
    transport();
    try {
      if (!resume) {
        const clip = await detail(sourceIndex);
        if (mine !== token) return false;
        audio.src = clip.path;
        loadedIndex = sourceIndex;
        $("clipTitle").textContent = ui.clipLabel(clip);
        ui.renderResponses(report, clip);
      }
      if (autoplay) {
        playRequest = mine;
        await audio.play();
        if (mine !== token) {
          if (playRequest === null) audio.pause();
          return false;
        }
      }
      loading = false; playing = audio.paused === false;
      transport();
      return true;
    } catch (error) {
      if (mine === token) {
        continuous = false; playRequest = null; loading = false; playing = false;
        transport();
        $("playState").textContent = error.message + " 다시 선택하거나 재생을 눌러주세요.";
      }
      return false;
    }
  }

  function move(position, autoplay = continuous) {
    if (!Number.isInteger(position) || !folder || position < 0 || position >= folder.clips.length) return;
    index = position;
    continuous = autoplay && folder.continuous;
    progress();
    selection = activate(folder.clips[index], {autoplay});
    return selection;
  }

  function selectFolder(nextFolder, position = 0) {
    if (!Number.isInteger(position) || !nextFolder?.clips || position < 0 || position >= nextFolder.clips.length) return;
    if (nextFolder === folder) return move(position);
    folder = nextFolder;
    return move(position, false);
  }

  function bind() {
    $("start").onclick = () => move(0, true);
    $("resume").onclick = () => {
      if (playing || loading || audio.paused === false) return stop();
      if (!folder) return;
      continuous = folder.continuous;
      const sourceIndex = folder.clips[index];
      return activate(sourceIndex, {autoplay: true, resume: loadedIndex === sourceIndex});
    };
    $("pause").onclick = stop;
    $("prev").onclick = () => move(index - 1);
    $("next").onclick = () => move(index + 1);
    audio.onplay = () => { playing = true; transport(); $("playState").textContent = ""; };
    audio.onpause = () => {
      if (audio.ended || audio.paused === false) return;
      playing = false;
      if (!loading) { continuous = false; playRequest = null; }
      transport();
    };
    audio.onended = () => {
      playing = false; loading = false; playRequest = null;
      if (continuous && index + 1 < folder.clips.length) return move(index + 1, true);
      continuous = false;
      transport();
    };
    audio.onerror = () => {
      if (loadedIndex !== folder?.clips[index]) return;
      loadedIndex = null;
      token++;
      continuous = false; playRequest = null; loading = false; playing = false;
      transport();
      $("playState").textContent = "오디오를 불러올 수 없습니다. 파일 경로와 서버 상태를 확인하세요.";
    };
  }

  async function boot() {
    chapter?.loading();
    try {
      $("title").textContent = `${report.wakeword} · 성능 시험 결과`;
      ui.renderSummary(report);
      ui.renderConditions(report);
      bind();
      if (viewer?.schema_version !== 1 || !Number.isInteger(viewer.page_size) || viewer.page_size < 1 || viewer.page_size > 128) throw new Error("지원하지 않는 평가 화면 형식입니다.");
      const data = await json(viewer.index_path);
      if (data.schema_version !== 1 || !Array.isArray(data.clips)) throw new Error("평가 목록 형식이 올바르지 않습니다.");
      metadata = data.clips;
      const library = ui.buildClipFolders(metadata, describe);
      updatePicker = ui.mountClipPicker?.(library, selectFolder, sourceIndex => ui.clipLabel(describe(metadata[sourceIndex])), OPTION_LIMIT);
      if (!updatePicker) {
        window.KowwListening?.enhance(audio, $("resume").parentElement, {
          play: $("resume"), previous: $("prev"), next: $("next"),
        });
      }
      if (library.defaultFolder) {
        selectFolder(library.defaultFolder);
        let displayed, observed;
        // A selection made while the default clip is loading also prepares the chapter.
        do {
          observed = selection;
          displayed = await observed;
        } while (observed !== selection);
        if (!displayed) {
          chapter?.fail($("playState").textContent || "평가 화면 준비가 중단되었습니다. 다시 열어 주세요.");
          return false;
        }
      } else {
        folder = null; index = 0; loadedIndex = null;
        stop();
        audio.removeAttribute("src");
        $("responses").tBodies[0].replaceChildren();
        $("clipTitle").textContent = "시험한 오디오가 없습니다.";
        $("playState").textContent = "";
        progress();
      }
      chapter?.ready();
      return true;
    } catch (error) {
      chapter?.fail(error?.message || "평가 자료를 불러오지 못했습니다.");
      throw error;
    }
  }
  window.KowwResults = {boot, detail, move, selectFolder, state: () => ({
    folder, index, queueLength: folder?.clips.length || 0,
    cachePages: [...cache.keys()], pendingPages: [...pending.keys()], continuous,
  }), OPTION_LIMIT, CACHE_LIMIT};
  if (!window.KOWW_RESULTS_TEST) boot().catch(error => { $("playState").textContent = error.message; });
})();
