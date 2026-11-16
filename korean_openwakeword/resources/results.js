/* Portable result viewer for inline clip details. */
"use strict";
(() => {
  const chapter = window.KowwChapterReady;
  const $ = id => document.getElementById(id);
  chapter?.loading();
  try {
    const {report, clips} = window.KOWW_RESULTS;
    const ui = window.KowwResultsUI;
    $("title").textContent = `${report.wakeword} · 성능 시험 결과`;
    ui.renderSummary(report);
    ui.renderConditions(report);
    const player = ui.createClipPlayer(report, ui.buildClipFolders(clips), {
      loadClip: index => clips[index],
      labelAt: index => ui.clipLabel(clips[index]),
    });
    if (!player.start()) throw new Error($("playState").textContent);
    chapter?.ready();
  } catch (error) {
    const message = error?.message || "평가 자료를 불러오지 못했습니다.";
    const status = $("playState");
    if (status) status.textContent = message;
    chapter?.fail(message);
  }
})();
