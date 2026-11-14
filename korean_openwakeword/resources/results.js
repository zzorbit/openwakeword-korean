/* Portable frozen result viewer: no API, no inference, no hidden audio gain. */
"use strict";
(() => {
  const chapter = window.KowwChapterReady;
  const $ = id => document.getElementById(id);
  chapter?.loading();
  try {
    const {report, clips} = window.KOWW_RESULTS;
    const ui = window.KowwResultsUI;
    const audio = $("audio");
    const library = ui.buildClipFolders(clips);
    let folder = null, index = 0, continuous = false;
    let token = 0, playRequest = null, playing = false, loading = false;

    $("title").textContent = `${report.wakeword} · 성능 시험 결과`;
    ui.renderSummary(report);
    ui.renderConditions(report);
    const updatePicker = ui.mountClipPicker?.(library, selectFolder, sourceIndex => ui.clipLabel(clips[sourceIndex]));
    if (!updatePicker) {
      window.KowwListening?.enhance(audio, $("resume").parentElement, {
        play: $("resume"), previous: $("prev"), next: $("next"),
      });
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

    function progress() {
      const length = folder?.clips.length || 0;
      $("queue").textContent = length ? `${(index + 1).toLocaleString("ko-KR")} / ${length.toLocaleString("ko-KR")}` : "목록 없음";
      $("prev").disabled = index <= 0;
      $("next").disabled = index + 1 >= length;
      updatePicker?.(folder, index);
    }

    async function playCurrent() {
      if (!folder) return;
      const mine = ++token;
      playRequest = mine; loading = true;
      $("playState").textContent = "";
      transport();
      try {
        await audio.play();
        if (mine !== token) {
          if (playRequest === null) audio.pause();
          return;
        }
        loading = false; playing = audio.paused === false;
        transport();
      } catch (error) {
        if (mine !== token) return;
        continuous = false; playRequest = null; loading = false; playing = false;
        transport();
        $("playState").textContent = `재생 실패: ${error.message}. 재생을 다시 눌러주세요.`;
      }
    }

    function move(position, autoplay = continuous) {
      if (!Number.isInteger(position) || !folder || position < 0 || position >= folder.clips.length) return;
      stop();
      index = position;
      continuous = autoplay && folder.continuous;
      const clip = clips[folder.clips[index]];
      audio.src = clip.path;
      $("clipTitle").textContent = ui.clipLabel(clip);
      $("playState").textContent = "";
      ui.renderResponses(report, clip);
      progress();
      return autoplay ? playCurrent() : true;
    }

    function selectFolder(nextFolder, position = 0) {
      if (!Number.isInteger(position) || !nextFolder?.clips || position < 0 || position >= nextFolder.clips.length) return;
      if (nextFolder === folder) return move(position);
      folder = nextFolder;
      return move(position, false);
    }

    $("start").onclick = () => move(0, true);
    $("resume").onclick = () => {
      if (playing || loading || audio.paused === false) return stop();
      if (!folder) return;
      continuous = folder.continuous;
      return playCurrent();
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
      token++;
      continuous = false; playRequest = null; loading = false; playing = false;
      transport();
      $("playState").textContent = "오디오를 불러올 수 없습니다. 파일 경로와 서버 상태를 확인하세요.";
    };

    if (library.defaultFolder) selectFolder(library.defaultFolder);
    else {
      $("clipTitle").textContent = "시험한 오디오가 없습니다.";
      progress(); transport();
    }
    chapter?.ready();
  } catch (error) {
    const message = error?.message || "평가 자료를 불러오지 못했습니다.";
    const status = $("playState");
    if (status) status.textContent = message;
    chapter?.fail(message);
  }
})();
