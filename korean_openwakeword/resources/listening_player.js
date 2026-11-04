/* Compact media controls only. Review decisions remain in their own controller. */
"use strict";
(() => {
  const mounted = new WeakMap();
  const clock = value => Number.isFinite(value) && value >= 0 ? `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}` : "0:00";
  function enhance(audio, controls, {play, previous, next} = {}) {
    if (!audio || !controls || !play || !previous || !next) return null;
    if (mounted.has(audio)) return mounted.get(audio);
    const seek = document.createElement("input"), time = document.createElement("span"), mute = document.createElement("button");
    seek.type = "range"; seek.min = "0"; seek.max = "1000"; seek.step = "1"; seek.value = "0";
    seek.className = "listen-seek"; seek.setAttribute("aria-label", "재생 위치");
    time.className = "listen-time"; time.setAttribute("aria-label", "재생 시간");
    mute.type = "button"; mute.className = "listen-mute";
    mute.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path class="listen-sound-wave" d="M16 8c2 2 2 6 0 8m3-11c4 4 4 10 0 14"/></svg>';
    controls.classList.add("is-compact"); play.classList.add("listen-play");
    for (const [button, direction, label] of [[previous, "previous", "이전 음원"], [next, "next", "다음 음원"]]) {
      button.classList.add("listen-skip", "listen-skip-" + direction);
      if (!button.getAttribute("aria-label")) button.setAttribute("aria-label", label);
    }
    controls.insertBefore(seek, next); controls.insertBefore(time, next); controls.append(mute);
    function sync() {
      const known = Number.isFinite(audio.duration) && audio.duration > 0;
      seek.disabled = !known; seek.value = known ? String(Math.round(Math.min(1, audio.currentTime / audio.duration) * 1000)) : "0";
      const progress = known && Number.isFinite(audio.currentTime) ? Math.max(0, Math.min(1, audio.currentTime / audio.duration)) * 100 : 0;
      seek.style.setProperty("--listen-progress", `${progress}%`);
      seek.setAttribute("aria-valuetext", `${clock(audio.currentTime)} / ${clock(audio.duration)}`);
      time.textContent = `${clock(audio.currentTime)} / ${clock(audio.duration)}`;
    }
    function syncMute() {
      mute.setAttribute("aria-label", audio.muted ? "소리 켜기" : "음소거");
      mute.setAttribute("aria-pressed", String(audio.muted));
    }
    seek.addEventListener("input", () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) audio.currentTime = Number(seek.value) / 1000 * audio.duration;
      sync();
    });
    mute.addEventListener("click", () => { audio.muted = !audio.muted; syncMute(); });
    for (const event of ["timeupdate", "loadedmetadata", "durationchange", "emptied"]) audio.addEventListener(event, sync);
    audio.addEventListener("volumechange", syncMute);
    // Hide the browser's duplicate play row only after its replacement exists.
    audio.controls = false; audio.classList.add("listen-audio-enhanced");
    const handle = {sync, seek, mute}; mounted.set(audio, handle); sync(); syncMute();
    return handle;
  }
  window.KowwListening = Object.freeze({enhance});
})();
