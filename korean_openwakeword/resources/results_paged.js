/* Frozen results, loaded in bounded pages. No inference or audio gain. */
"use strict";
(() => {
  const chapter = window.KowwChapterReady;
  const {report, viewer} = window.KOWW_RESULTS;
  const ui = window.KowwResultsUI;
  const $ = id => document.getElementById(id);
  const OPTION_LIMIT = 100, CACHE_LIMIT = 2;
  const cache = new Map(), pending = new Map();
  let metadata = [], player;

  function describe([, kind, category, snr_db, reverb_condition, original_id, original_order]) {
    return {kind, category, snr_db, reverb_condition, original_id, original_order};
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

  async function boot() {
    chapter?.loading();
    try {
      $("title").textContent = `${report.wakeword} · 성능 시험 결과`;
      ui.renderSummary(report);
      ui.renderConditions(report);
      if (viewer?.schema_version !== 1 || !Number.isInteger(viewer.page_size) || viewer.page_size < 1 || viewer.page_size > 128) throw new Error("지원하지 않는 평가 화면 형식입니다.");
      const data = await json(viewer.index_path);
      if (data.schema_version !== 1 || !Array.isArray(data.clips)) throw new Error("평가 목록 형식이 올바르지 않습니다.");
      metadata = data.clips;
      player = ui.createClipPlayer(report, ui.buildClipFolders(metadata, describe), {
        loadClip: detail,
        labelAt: index => ui.clipLabel(describe(metadata[index])),
        pageSize: OPTION_LIMIT,
      });
      player.start();
      if (!await player.waitForSelection()) {
        chapter?.fail($("playState").textContent || "평가 화면 준비가 중단되었습니다. 다시 열어 주세요.");
        return false;
      }
      chapter?.ready();
      return true;
    } catch (error) {
      chapter?.fail(error?.message || "평가 자료를 불러오지 못했습니다.");
      throw error;
    }
  }

  window.KowwResults = {
    boot, detail, move: (...args) => player?.move(...args), selectFolder: (...args) => player?.selectFolder(...args),
    state: () => ({
      ...(player?.state() || {folder: null, index: 0, queueLength: 0, continuous: false}),
      cachePages: [...cache.keys()], pendingPages: [...pending.keys()],
    }),
    OPTION_LIMIT, CACHE_LIMIT,
  };
  if (!window.KOWW_RESULTS_TEST) boot().catch(error => { $("playState").textContent = error.message; });
})();
