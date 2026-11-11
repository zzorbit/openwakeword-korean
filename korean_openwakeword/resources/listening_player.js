/* Shared listening controls. Review decisions remain in their own controller. */
"use strict";
(() => {
  if (window.KowwListening) return;
  const mounted = new WeakMap();
  // Only open pickers are retained here. Closed/discarded views do not leave
  // document listeners holding their DOM and controller closures alive.
  const openPickers = new Set();
  let pickerObserver = null;
  function observePickers() {
    if (!openPickers.size) { pickerObserver?.disconnect(); return; }
    if (typeof MutationObserver !== "function" || !document.documentElement) return;
    pickerObserver ||= new MutationObserver(prunePickers);
    pickerObserver.observe(document.documentElement, {subtree: true, childList: true,
      attributes: true, attributeFilter: ["hidden", "inert", "disabled", "class", "style"]});
  }
  function prunePickers() {
    for (const picker of [...openPickers]) if (!picker.usable()) picker.close();
  }
  function closePickers(restoreFocus = false) {
    for (const picker of [...openPickers]) picker.close(restoreFocus);
  }
  function ownOrigin() {
    const origin = window.location?.origin;
    return origin && origin !== "null" ? origin : null;
  }
  function childFrames() {
    const origin = ownOrigin();
    if (!origin) return [];
    return [...(document.querySelectorAll?.("iframe") || [])].filter(frame => {
      try { return frame.contentWindow && new URL(frame.getAttribute("src") || "", window.location.href).origin === origin; }
      catch { return false; }
    });
  }
  function sendClose(target) {
    const origin = ownOrigin();
    if (!origin || !target?.postMessage) return;
    try { target.postMessage({type: "koww:close-pickers"}, origin); } catch { /* A detached/navigating frame needs no close. */ }
  }
  function closeChildPickers(except) {
    for (const frame of childFrames()) if (frame.contentWindow !== except) sendClose(frame.contentWindow);
  }
  function notifyParent() {
    if (window.parent && window.parent !== window) sendClose(window.parent);
  }
  // Capture before another control handles the gesture. Closing never cancels
  // that control's pointer action, changes playback, or saves a review answer.
  document.addEventListener?.("pointerdown", event => {
    prunePickers();
    for (const picker of [...openPickers]) if (!picker.contains(event)) picker.close();
    closeChildPickers(); notifyParent();
  }, true);
  document.addEventListener?.("scroll", event => {
    prunePickers();
    for (const picker of [...openPickers]) if (!picker.popupContains(event.target)) picker.close();
  }, true);
  document.addEventListener?.("keydown", event => {
    if (event.key !== "Escape") return;
    prunePickers();
    if (openPickers.size) { event.preventDefault(); closePickers(true); }
  }, true);
  document.addEventListener?.("visibilitychange", () => { if (document.hidden) closePickers(); });
  window.addEventListener?.("resize", () => {
    prunePickers(); for (const picker of [...openPickers]) picker.resize();
  });
  window.addEventListener?.("pagehide", () => closePickers());
  window.addEventListener?.("blur", () => closePickers());
  window.addEventListener?.("message", event => {
    if (event.data?.type !== "koww:close-pickers" || !ownOrigin() || event.origin !== ownOrigin()) return;
    if (window.parent && window.parent !== window && event.source === window.parent) {
      closePickers(); closeChildPickers(); return;
    }
    if (!childFrames().some(frame => frame.contentWindow === event.source)) return;
    closePickers(); closeChildPickers(event.source); notifyParent();
  });
  const clock = value => Number.isFinite(value) && value >= 0 ? `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}` : "0:00";
  function enhance(audio, controls, {play, previous, next, volume} = {}) {
    if (!audio || !controls || !play || Boolean(previous) !== Boolean(next)) return null;
    if (mounted.has(audio)) return mounted.get(audio);
    const seek = document.createElement("input"), time = document.createElement("span"), mute = document.createElement("button");
    seek.type = "range"; seek.min = "0"; seek.max = "1000"; seek.step = "1"; seek.value = "0";
    seek.className = "listen-seek"; seek.setAttribute("aria-label", "재생 위치");
    time.className = "listen-time"; time.setAttribute("aria-label", "재생 시간");
    mute.type = "button"; mute.className = "listen-mute";
    mute.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24"><path d="M4 9h4l5-4v14l-5-4H4z"/><path class="listen-sound-wave" d="M16 8c2 2 2 6 0 8m3-11c4 4 4 10 0 14"/></svg>';
    controls.classList.add("is-compact"); play.classList.add("listen-play");
    for (const [button, direction, label] of [[previous, "previous", "이전 음원"], [next, "next", "다음 음원"]]) {
      if (!button) continue;
      button.classList.add("listen-skip", "listen-skip-" + direction);
      if (!button.getAttribute("aria-label")) button.setAttribute("aria-label", label);
    }
    if (next) {
      controls.insertBefore(seek, next);
      controls.insertBefore(time, next);
    } else {
      controls.append(seek, time);
    }
    controls.append(mute);
    if (volume) {
      volume.type = "range"; volume.min = "0"; volume.max = "100"; volume.step = "1";
      if (!volume.getAttribute("aria-label")) volume.setAttribute("aria-label", "음량");
      volume.classList.add("listen-volume");
      mute.classList.add("listen-inline-mute");
      controls.append(volume);
      volume.hidden = false;
      volume.addEventListener("input", () => {
        const level = Number(volume.value);
        if (!Number.isFinite(level)) return;
        audio.volume = Math.max(0, Math.min(100, level)) / 100;
        if (audio.volume > 0) audio.muted = false;
        syncMute();
      });
    }
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
      if (!volume) return;
      const level = Math.round(Math.max(0, Math.min(1, Number.isFinite(audio.volume) ? audio.volume : 1)) * 100);
      volume.value = String(level);
      volume.style.setProperty("--listen-progress", `${level}%`);
      volume.setAttribute("aria-valuetext", `${level}%${audio.muted ? " · 음소거" : ""}`);
      volume.setAttribute("title", `음량 ${level}%`);
      mute.setAttribute("data-muted", String(audio.muted || level === 0));
      mute.setAttribute("title", audio.muted ? "소리 켜기" : `음소거 · 음량 ${level}%`);
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
  function createPicker({container, button, label, popup, list, onSelect, onOpenChange, optionPrefix = `${list.id}-option`}) {
    const accessibleName = button.getAttribute("aria-label") || "음원 선택";
    let rows = [], selected = -1, active = 0, open = false, ready = false, request = 0;
    const usable = () => !document.hidden && container.isConnected !== false && button.isConnected !== false
      && !container.hidden && !button.hidden && !button.disabled && !container.closest?.("[hidden],[inert]")
      && (typeof container.getClientRects !== "function" || container.getClientRects().length > 0);
    const registryEntry = {close, usable,
      contains: event => Boolean(container.contains(event.target) || popup.contains(event.target)
        || event.composedPath?.().some(node => node === container || node === popup)),
      popupContains: target => popup.contains(target),
      resize: () => { if (open && ready) { position(); highlight(active); } }};

    function position() {
      const rect = button.getBoundingClientRect(), viewport = Math.max(0, window.innerHeight),
        margin = Math.min(8, Math.floor(viewport / 2)), gap = -1;
      const below = Math.max(0, viewport - rect.bottom - gap - margin);
      const above = Math.max(0, rect.top - gap - margin);
      const height = Math.max(0, Math.floor(Math.min(360, rows.length * 36 + 10,
        Math.max(below, above), viewport - margin * 2)));
      const down = below >= above;
      const side = down ? "down" : "up";
      const desiredTop = down ? rect.bottom + gap : rect.top - gap - height;
      const top = Math.max(margin, Math.min(desiredTop, viewport - margin - height));
      container.setAttribute("data-side", side);
      popup.setAttribute("data-side", side);
      popup.style.maxHeight = `${height}px`;
      list.style.maxHeight = `${Math.max(0, height - 2)}px`;
      popup.style.top = `${Math.floor(top)}px`;
      popup.style.left = `${Math.max(0, Math.min(rect.left, window.innerWidth - rect.width))}px`;
      popup.style.width = `${Math.max(0, Math.min(rect.width, window.innerWidth))}px`;
    }

    function highlight(index) {
      if (!rows.length) return;
      rows[active]?.element.removeAttribute("data-active");
      active = Math.max(0, Math.min(rows.length - 1, index));
      const row = rows[active].element;
      row.setAttribute("data-active", "true");
      list.setAttribute("aria-activedescendant", row.id);
      if (!open || !ready) return;
      if (row.offsetTop < list.scrollTop) list.scrollTop = row.offsetTop;
      else if (row.offsetTop + row.offsetHeight > list.scrollTop + list.clientHeight) {
        list.scrollTop = row.offsetTop + row.offsetHeight - list.clientHeight;
      }
    }

    function close(restoreFocus = false) {
      openPickers.delete(registryEntry); observePickers();
      if (!open) return;
      open = false; ready = false;
      popup.hidden = true; list.hidden = true;
      container.removeAttribute("data-side");
      popup.removeAttribute("data-side");
      button.setAttribute("aria-expanded", "false");
      onOpenChange?.(false, request);
      if (restoreFocus && usable()) button.focus();
    }

    function reveal(layoutRequest) {
      if (!open || ready || layoutRequest !== request || !rows.length) return;
      if (!usable()) { close(); return; }
      ready = true;
      position();
      popup.hidden = false; list.hidden = false;
      highlight(active);
      list.focus();
    }

    function show() {
      if (!rows.length || !usable()) return;
      if (open) {
        if (ready) list.focus();
        return;
      }
      closePickers(); open = true; ready = false;
      openPickers.add(registryEntry); observePickers();
      request++;
      button.setAttribute("aria-expanded", "true");
      highlight(selected);
      if (onOpenChange) onOpenChange(true, request);
      else reveal(request);
    }

    function choose(index) {
      if (!open || !ready || !rows[index]) return;
      onSelect(index);
      close(true);
    }

    function setSelected(index, unselectedLabel = "") {
      const row = rows[index];
      if ((!row && index !== -1) || (row && selected === index)) return;
      rows[selected]?.element.setAttribute("aria-selected", "false");
      selected = index;
      row?.element.setAttribute("aria-selected", "true");
      label.textContent = row?.label || unselectedLabel;
      const description = row?.fullLabel || unselectedLabel;
      button.setAttribute("aria-label", description ? `${accessibleName}: ${description}` : accessibleName);
    }

    function setBadge(index, value) {
      const row = rows[index];
      if (!row || row.badgeText === value) return;
      row.badgeText = value;
      if (!row.badge && value) {
        row.badge = document.createElement("span");
        row.badge.className = "voice-picker-badge";
        row.element.append(row.badge);
      }
      if (row.badge) row.badge.textContent = value;
      row.element.setAttribute("aria-label", value ? `${row.fullLabel} · ${value}` : row.fullLabel);
    }

    function setItems(entries, selectedIndex = 0, unselectedLabel = "") {
      close();
      selected = -1; active = 0;
      rows = entries.map((entry, index) => {
        const element = document.createElement("div");
        const group = document.createElement("span");
        element.id = `${optionPrefix}-${index}`;
        element.className = "voice-picker-option";
        element.setAttribute("role", "option");
        element.setAttribute("aria-selected", "false");
        const fullLabel = entry.fullLabel || entry.label;
        element.setAttribute("aria-label", fullLabel);
        group.className = "voice-picker-group";
        group.textContent = entry.groupLabel || entry.label;
        element.append(group);
        if (entry.voiceNumber != null) {
          const number = document.createElement("span");
          number.className = "voice-picker-number";
          number.textContent = String(entry.voiceNumber);
          element.append(number);
        }
        if (entry.detailLabel) {
          const detail = document.createElement("span");
          detail.className = "voice-picker-detail";
          detail.textContent = entry.detailLabel;
          element.append(detail);
        }
        element.addEventListener("click", () => choose(index));
        element.addEventListener("pointermove", () => { if (active !== index) highlight(index); });
        return {element, label: entry.label, fullLabel};
      });
      list.replaceChildren(...rows.map(row => row.element));
      list.removeAttribute("aria-activedescendant");
      entries.forEach((entry, index) => { if (entry.badge) setBadge(index, entry.badge); });
      button.disabled = rows.length === 0;
      setSelected(rows.length ? selectedIndex : -1, unselectedLabel);
    }

    const selectionKeys = ["ArrowDown", "ArrowUp", "Home", "End"];
    button.addEventListener("click", () => { if (open) close(true); else show(); });
    button.addEventListener("keydown", event => {
      if (open && (event.key === "Escape" || event.key === "Tab")) {
        if (event.key === "Escape") event.preventDefault();
        close(event.key === "Escape");
        return;
      }
      if (![...selectionKeys, "Enter", " "].includes(event.key)) return;
      event.preventDefault();
      if (open && (event.key === "Enter" || event.key === " ")) close(true);
      else {
        show();
        if (!open) return;
        if (event.key === "Home") highlight(0);
        if (event.key === "End") highlight(rows.length - 1);
      }
    });
    list.addEventListener("keydown", event => {
      if (!open || !ready) return;
      if (event.key === "Tab") { close(); return; }
      if (![...selectionKeys, "Enter", " ", "Escape"].includes(event.key)) return;
      event.preventDefault();
      if (event.key === "Escape") close(true);
      else if (event.key === "Enter" || event.key === " ") choose(active);
      else if (event.key === "Home") highlight(0);
      else if (event.key === "End") highlight(rows.length - 1);
      else highlight(active + (event.key === "ArrowDown" ? 1 : -1));
    });
    popup.hidden = true; list.hidden = true;
    button.setAttribute("aria-expanded", "false");
    return {setItems, setSelected, setBadge, close, reveal};
  }

  window.KowwListening = Object.freeze({enhance, createPicker, closePickers: () => closePickers()});
})();

/* A chapter reports rendered UI readiness, independently of playback or review decisions. */
(() => {
  if (window.KowwChapterReady) return;
  let ownUrl;
  try { ownUrl = new URL(location.href); ownUrl.hash = ""; } catch { return; }
  const pathProject = ownUrl.pathname.match(/^\/(?:results|mic)\/([^/]+)\//)?.[1];
  let project = ownUrl.searchParams.get("project") || "";
  if (!project && pathProject) { try { project = decodeURIComponent(pathProject); } catch { return; } }
  let status = "loading", message = "", revision = 0, delivery = 0, active = true, pending = null;
  const paint = () => new Promise(resolve => {
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => resolve());
    else if (typeof setTimeout === "function") setTimeout(resolve, 0);
    else resolve();
  });
  function schedule() {
    const ticket = ++delivery, version = revision, request = pending;
    if (!active || !request || status === "loading") return;
    Promise.resolve(document.fonts?.ready).catch(() => {}).then(paint).then(paint).then(() => {
      if (!active || ticket !== delivery || version !== revision || request !== pending || status === "loading") return;
      const response = {type: "koww:chapter-ready", project, request: request.id, url: ownUrl.href, status};
      if (status === "error") response.message = message;
      window.parent.postMessage(response, ownUrl.origin);
    });
  }
  function update(next, error = "") {
    revision++; status = next; message = typeof error === "string" ? error.trim().slice(0, 500) : "";
    if (status === "error" && !message) message = "화면을 불러오지 못했습니다. 다시 시도해 주세요.";
    schedule();
  }
  window.addEventListener("message", event => {
    if (window.parent === window || event.source !== window.parent || event.origin !== ownUrl.origin
        || event.data?.type !== "koww:chapter-request" || event.data.project !== project
        || !Number.isSafeInteger(event.data.request) || event.data.request < 1
        || typeof event.data.url !== "string" || !/^https?:\/\//.test(event.data.url)) return;
    let requested;
    try { requested = new URL(event.data.url); requested.hash = ""; } catch { return; }
    if (requested.href !== ownUrl.href) return;
    pending = {id: event.data.request}; schedule();
  });
  window.addEventListener("pagehide", () => { active = false; pending = null; delivery++; });
  window.addEventListener("pageshow", () => { active = true; schedule(); });
  window.KowwChapterReady = Object.freeze({
    loading: () => update("loading"),
    ready: () => update("ready"),
    fail: error => update("error", error),
  });
})();
