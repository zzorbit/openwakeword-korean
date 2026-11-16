"use strict";

(() => {
  if (window.parent && window.parent !== window) document.body?.classList?.add("embedded");
  const $ = id => document.getElementById(id);
  const config = JSON.parse($("koww-page-config")?.textContent || "{}");
  const apiUrl = path => (config.api_base || "") + path;
  const ui = Object.fromEntries(["wakeword", "microphone", "connect", "refresh", "retry", "deviceHint", "detection-settings", "thresholdSetting", "consecutiveSetting", "trial-mode", "trial-threshold", "trial-consecutive", "reset-detection", "start", "stop", "statusHeading", "statusText", "statusDot", "level", "levelText", "elapsed", "frameCount", "lastEvent", "lastEventTime", "model-bars", "threshold"].map(id => [id, $(id)]));
  const rows = new Map();
  let info = null;
  let run = null;
  let closingSession = false;
  let refreshing = false;
  let probe = null;
  let initializing = false;
  let defaultRuntime = null;
  let thresholdEdited = false;
  let recommendedRuntimes = null;
  function showDeviceHint(text) {
    if (ui.deviceHint.textContent !== text) ui.deviceHint.textContent = text;
  }

  function displayThreshold(value) {
    const rounded = Number(value.toPrecision(6));
    return String(rounded > 0 && rounded < 1 ? rounded : value);
  }

  function validRuntime(runtime) {
    return Number.isFinite(runtime?.threshold) && runtime.threshold > 0 && runtime.threshold < 1
      && Number.isInteger(runtime.consecutive_frames) && runtime.consecutive_frames >= 1 && runtime.consecutive_frames <= 10;
  }

  function showTrialRuntime(runtime, runtimes = null) {
    const unavailable = info?.runtime_recommendations?.candidates?.some(row => row.recommendation_available === false);
    ui.threshold.textContent = runtimes ? (unavailable ? "모델별 추천·기존값 적용" : "모델별 추천값 적용")
      : `감지 기준 ${displayThreshold(runtime.threshold)} · 연속 ${runtime.consecutive_frames}회`;
    for (const [id, row] of rows) {
      const value = runtimes ? runtimes[id] : runtime;
      const valid = validRuntime(value);
      row.runtimeLabel.textContent = valid ? `${displayThreshold(value.threshold)} · ${value.consecutive_frames}회` : "—";
      row.thresholdMarker.setAttribute("style", valid ? `left:${value.threshold * 100}%` : "display:none");
    }
  }

  function syncDetectionDisplay() {
    const recommended = ui["trial-mode"].value === "recommended" && recommendedRuntimes;
    if (recommended) showTrialRuntime(null, recommendedRuntimes);
    else {
      const threshold = Number(ui["trial-threshold"].value), consecutive_frames = Number(ui["trial-consecutive"].value);
      if (validRuntime({threshold, consecutive_frames})) showTrialRuntime({threshold, consecutive_frames});
    }
    ui.thresholdSetting.hidden = !!recommended;
    ui.consecutiveSetting.hidden = !!recommended;
    ui["detection-settings"].className = `detection-settings${recommended ? " recommended" : ""}`;
  }

  function setDefaultFields() {
    ui["trial-threshold"].value = displayThreshold(defaultRuntime.threshold);
    ui["trial-consecutive"].value = String(defaultRuntime.consecutive_frames);
    thresholdEdited = false;
  }

  function resetDetection() {
    if (!defaultRuntime || run || probe || closingSession) return;
    ui["trial-mode"].value = "manual";
    setDefaultFields();
    syncControls(); resetMetrics();
  }

  function trialSettings() {
    if (ui["trial-mode"].value === "recommended") {
      if (!recommendedRuntimes) throw new Error("모델별 추천 설정을 아직 사용할 수 없습니다.");
      return { mode: "recommended" };
    }
    const rawThreshold = ui["trial-threshold"].value.trim();
    const rawConsecutive = ui["trial-consecutive"].value.trim();
    const parsedThreshold = Number(rawThreshold), consecutive = Number(rawConsecutive);
    if (!rawThreshold || !Number.isFinite(parsedThreshold) || parsedThreshold <= 0 || parsedThreshold >= 1) {
      throw new Error("감지 기준은 0보다 크고 1보다 작은 숫자를 입력하세요.");
    }
    if (!rawConsecutive || !Number.isInteger(consecutive) || consecutive < 1 || consecutive > 10) {
      throw new Error("연속 감지 수는 1~10의 정수를 입력하세요.");
    }
    // Keep the frozen default's full precision unless the user changes it.
    const threshold = !thresholdEdited && rawThreshold === displayThreshold(defaultRuntime.threshold)
      ? defaultRuntime.threshold : parsedThreshold;
    return { threshold, consecutive_frames: consecutive };
  }

  function setStatus(heading, text, phase = "idle") {
    if (ui.statusHeading.textContent !== heading) ui.statusHeading.textContent = heading;
    if (ui.statusText.textContent !== text) ui.statusText.textContent = text;
    const dotClass = `status-dot ${phase}`;
    if (ui.statusDot.className !== dotClass) ui.statusDot.className = dotClass;
  }

  function syncControls() {
    const settingsLocked = !info || !!run || !!probe || closingSession;
    ui.start.disabled = settingsLocked || document.hidden;
    ui.stop.disabled = !run && !probe;
    ui.microphone.disabled = !!run || !!probe;
    ui.connect.disabled = !!run || !!probe || refreshing || document.hidden || !window.isSecureContext || !navigator.mediaDevices?.getUserMedia;
    ui.refresh.disabled = !!run || refreshing || !!probe || !navigator.mediaDevices?.enumerateDevices;
    ui.retry.disabled = initializing;
    ui["trial-mode"].disabled = settingsLocked || !recommendedRuntimes;
    for (const control of ["trial-threshold", "trial-consecutive"]) ui[control].disabled = settingsLocked || ui["trial-mode"].value === "recommended";
    ui["reset-detection"].disabled = settingsLocked;
  }

  async function request(path, options = {}, timeoutMs = 5000, controller = new AbortController(), timeoutMessage = "서버 응답이 늦어 시험을 중지했습니다. 서버 상태를 확인한 뒤 다시 시작하세요.") {
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(apiUrl(path), { ...options, signal: controller.signal, cache: "no-store" });
      let data;
      try { data = await response.json(); } catch (_) { throw new Error("서버 응답을 읽을 수 없습니다. 터미널의 서버 상태를 확인하세요."); }
      if (!response.ok) {
        if (response.status === 409) throw new Error("다른 창에서 시험 중이거나 이전 연결이 정리 중입니다. 다른 시험을 중지한 뒤 다시 시작하세요.");
        throw new Error(data.error || data.message || `서버 오류 (${response.status})`);
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error(timeoutMessage);
      throw error;
    } finally { clearTimeout(timer); }
  }

  function headers(extra = {}) { return { "X-Koww-Token": info.token, ...extra }; }
  function isCurrent(target) { return run === target; }

  function renderModels(models) {
    const bars = ui["model-bars"];
    bars.replaceChildren();
    rows.clear();
    for (const model of models) {
      const item = document.createElement("div"), heading = document.createElement("div");
      const name = document.createElement("span"), events = document.createElement("span");
      item.className = "mic-model-row"; heading.className = "mic-model-top";
      name.textContent = model.label; events.textContent = "감지 0회"; events.className = "mic-event-count";
      heading.append(name, events);

      const response = document.createElement("div"), track = document.createElement("div");
      const fill = document.createElement("span"), marker = document.createElement("span");
      const score = document.createElement("span"), runtime = document.createElement("span");
      response.className = "mic-model-response"; track.className = "mic-score-track";
      fill.className = "mic-score-fill"; marker.className = "mic-score-marker";
      track.setAttribute("role", "progressbar");
      track.setAttribute("aria-label", `${model.label} 현재 점수`);
      track.setAttribute("aria-valuemin", "0"); track.setAttribute("aria-valuemax", "1"); track.setAttribute("aria-valuenow", "0");
      score.className = "mic-current-score"; score.textContent = "0.00";
      runtime.className = "mic-runtime-label"; runtime.textContent = "—";
      track.append(fill, marker); response.append(track, score, runtime);
      item.append(heading, response); bars.append(item);
      rows.set(model.id, {item, scoreTrack: track, scoreFill: fill, thresholdMarker: marker, scoreLabel: score, eventLabel: events, runtimeLabel: runtime, flashUntil: 0});
    }
  }

  function resetMetrics() {
    ui.level.value = 0;
    ui.levelText.textContent = "대기 중";
    ui.elapsed.textContent = "0.0초";
    ui.frameCount.textContent = "0개 구간";
    ui.lastEvent.textContent = "아직 없음";
    ui.lastEventTime.textContent = "";
    renderModels(info.models);
    syncDetectionDisplay();
  }

  async function refreshDevices({expectedProbe = null, requested = false} = {}) {
    if (refreshing || !navigator.mediaDevices?.enumerateDevices) return;
    const activeRun = run;
    const activeDeviceId = activeRun?.activeDeviceId;
    refreshing = true; syncControls();
    if (requested) showDeviceHint("마이크 목록을 읽는 중입니다.");
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "audioinput");
      if (expectedProbe && probe !== expectedProbe) return;
      const selected = ui.microphone.value;
      ui.microphone.replaceChildren(new Option("시스템 기본 마이크", ""));
      let number = 0;
      for (const device of devices) {
        if (!device.deviceId || device.deviceId === "default") continue;
        ui.microphone.append(new Option(device.label || `마이크 ${++number} (권한 허용 후 이름 표시)`, device.deviceId));
      }
      const stillListed = Array.from(ui.microphone.options).some(option => option.value === selected);
      ui.microphone.value = stillListed ? selected : "";
      if (selected && !stillListed) showDeviceHint("선택한 마이크가 목록에 없어 시스템 기본 마이크로 바꿨습니다.");
      else if (expectedProbe) showDeviceHint("마이크를 확인했습니다. 시험을 시작하면 입력 레벨을 볼 수 있습니다.");
      else if (requested) showDeviceHint(devices.length ? "" : "표시할 입력 장치가 없습니다. 연결을 눌러 마이크를 확인하세요.");
      if (activeDeviceId && run === activeRun && !devices.some(device => device.deviceId === activeDeviceId)) {
        stop("선택한 마이크 연결이 끊겨 시험을 중지했습니다.", true);
      }
    } catch (error) {
      if (!expectedProbe || probe === expectedProbe) showDeviceHint(`마이크 목록을 읽지 못했습니다: ${error.message}`);
    } finally {
      refreshing = false; syncControls();
    }
  }

  function cancelProbe() {
    const target = probe;
    probe = null;
    for (const track of target?.stream?.getTracks() || []) track.stop();
    if (target) showDeviceHint("마이크 연결을 취소했습니다. 다시 연결하려면 연결을 누르세요.");
  }

  async function connectMicrophone() {
    if (probe || run || refreshing || document.hidden || !window.isSecureContext || !navigator.mediaDevices?.getUserMedia) return;
    const target = {};
    probe = target;
    showDeviceHint("브라우저에서 마이크 사용을 허용하세요. 중지를 누르면 연결을 취소합니다.");
    syncControls();
    try {
      const deviceId = ui.microphone.value;
      target.stream = await navigator.mediaDevices.getUserMedia({ audio: deviceId ? { deviceId: { exact: deviceId } } : true, video: false });
      if (probe !== target) return;
      await refreshDevices({expectedProbe: target});
    } catch (error) {
      if (probe === target) showDeviceHint(microphoneError(error));
    } finally {
      for (const track of target.stream?.getTracks() || []) track.stop();
      if (probe === target) probe = null;
      syncControls();
    }
  }

  function closeResources(target) {
    if (target.watchdog) clearInterval(target.watchdog);
    target.queue.length = 0;
    target.frameController?.abort();
    if (target.worklet) target.worklet.port.onmessage = null;
    for (const node of [target.source, target.worklet, target.silence]) { try { node?.disconnect(); } catch (_) {} }
    for (const track of target.stream?.getTracks() || []) { track.onended = null; track.onmute = null; track.stop(); }
    if (target.context) target.context.onstatechange = null;
    if (target.context && target.context.state !== "closed") target.context.close().catch(() => {});
  }

  async function releaseServerSession(target, message, hadError) {
    closingSession = true;
    if (!hadError) setStatus("시험 연결 정리 중", "", "preparing");
    try {
      // A cancelled start can still allocate a session before its response arrives.
      const sessionId = target.sessionId || (await target.startRequest)?.session_id;
      if (sessionId) {
        await request("/api/stop", { method: "POST", headers: headers({ "Content-Type": "application/json" }),
          body: JSON.stringify({ session_id: sessionId }), keepalive: true }, 5000, new AbortController(),
          "서버가 5초 안에 시험 종료에 응답하지 않았습니다.");
      }
      if (!hadError) setStatus("시험 중지", message);
    } catch (error) {
      const detail = `이전 시험 연결을 정리하지 못했습니다. ${error.message}`;
      setStatus("시험 중지 · 확인 필요", hadError ? `${message} ${detail}` : detail, "error");
    } finally {
      closingSession = false;
      syncControls();
    }
  }

  function stop(message = "마이크 사용을 중지했습니다. 다시 시작하면 새 시험이 시작됩니다.", error = false) {
    cancelProbe();
    const target = run;
    if (!target) { syncControls(); return; }
    run = null;
    closeResources(target);
    ui.level.value = 0;
    ui.levelText.textContent = "마이크 중지됨";
    const hasServerSession = Boolean(target.sessionId || target.startRequest);
    if (error || !hasServerSession) setStatus(error ? "시험 중지 · 확인 필요" : "시험 중지", message, error ? "error" : "idle");
    if (hasServerSession) void releaseServerSession(target, message, error);
    syncControls();
  }

  function updateFrame(target, data) {
    if (data.session_id !== target.sessionId || data.sequence !== target.sequence - 1 || !Number.isFinite(data.time_s) || !Array.isArray(data.responses)) throw new Error("오디오 구간의 순서가 맞지 않아 시험을 중지했습니다.");
    ui.elapsed.textContent = `${data.time_s.toFixed(1)}초`;
    ui.frameCount.textContent = `${target.sequence.toLocaleString()}개 구간`;
    ui.level.value = Math.min(1, Math.max(0, Number(data.rms) * 8));
    const peak = Number(data.peak), rms = Number(data.rms);
    ui.levelText.textContent = peak >= 0.999 ? "입력이 너무 큽니다 · 마이크에서 조금 멀어지세요." : rms < 0.0001 ? "거의 무음" : `입력 RMS ${Number.isFinite(rms) ? rms.toFixed(4) : "—"}`;
    if (data.eligible) {
      setStatus(`‘${info.wakeword}’를 불러보세요`, "마이크 소리를 분석하고 있습니다. 주변 소음이 있는 상태에서도 시험해 보세요.", "listening");
    } else {
      const remaining = Math.max(0, Number(info.warmup_s) - data.time_s);
      setStatus("모델 준비 중", `${remaining.toFixed(1)}초 뒤에 호출어를 말하세요. 준비 중인 구간은 감지에 포함하지 않습니다.`, "preparing");
    }
    const detected = [];
    for (const response of data.responses) {
      const row = rows.get(response.id);
      if (!row || !Number.isFinite(response.score) || !Number.isFinite(response.max_score) || !Number.isInteger(response.events)) throw new Error("모델 반응을 읽을 수 없어 시험을 중지했습니다.");
      const bounded = Math.min(1, Math.max(0, response.score));
      row.scoreTrack.setAttribute("aria-valuenow", String(bounded));
      row.scoreFill.setAttribute("style", `width:${bounded * 100}%`);
      row.scoreLabel.textContent = response.score.toFixed(2);
      row.eventLabel.textContent = `감지 ${response.events.toLocaleString()}회`;
      if (response.event) { detected.push(response.label); row.flashUntil = performance.now() + 700; }
      row.item.className = `mic-model-row${performance.now() < row.flashUntil ? " detected" : ""}`;
    }
    if (detected.length) {
      ui.lastEvent.textContent = detected.join(" · ");
      ui.lastEventTime.textContent = `분석한 오디오의 ${data.time_s.toFixed(2)}초 지점`;
    }
  }

  async function drain(target) {
    if (target.draining) return;
    target.draining = true;
    try {
      while (isCurrent(target) && target.queue.length) {
        const pcm = target.queue.shift();
        const sequence = target.sequence++;
        target.frameController = new AbortController();
        const data = await request("/api/frame", { method: "POST", headers: headers({ "Content-Type": "application/octet-stream", "X-Koww-Session": target.sessionId, "X-Koww-Sequence": String(sequence) }), body: pcm }, 5000, target.frameController);
        if (isCurrent(target)) updateFrame(target, data);
      }
    } catch (error) { if (isCurrent(target)) stop(error.message, true); }
    finally { target.draining = false; target.frameController = null; }
  }

  function onFrame(target, event) {
    if (!isCurrent(target)) return;
    if (event.data.type === "error") { stop(event.data.message, true); return; }
    if (event.data.type !== "frame" || !(event.data.pcm instanceof ArrayBuffer) || event.data.pcm.byteLength !== info.frame_samples * 2) { stop("마이크 오디오 구간이 유효하지 않아 시험을 중지했습니다.", true); return; }
    if (target.queue.length + (target.draining ? 1 : 0) >= 12) {
      stop("실시간 분석이 입력 속도를 따라가지 못해 시험을 중지했습니다. 다른 작업을 줄인 뒤 다시 시작하세요. 오디오 구간을 버린 채 감지를 계속하지 않습니다.", true);
      return;
    }
    target.lastFrameAt = performance.now();
    target.queue.push(event.data.pcm);
    drain(target);
  }

  function microphoneError(error) {
    const messages = {
      NotAllowedError: "마이크 권한이 거부됐습니다. 주소창의 사이트 권한에서 마이크를 허용한 뒤 다시 시작하세요.",
      NotFoundError: "사용 가능한 마이크가 없습니다. 입력 장치를 연결한 뒤 목록을 새로고침하세요.",
      NotReadableError: "마이크를 열 수 없습니다. 다른 프로그램의 마이크 사용을 종료한 뒤 다시 시작하세요.",
      OverconstrainedError: "선택한 마이크를 사용할 수 없습니다. 목록을 새로고침하고 다른 장치를 선택하세요."
    };
    return messages[error.name] || error.message || "마이크를 시작하지 못했습니다.";
  }

  async function start() {
    if (!info || run || probe || closingSession || document.hidden) return;
    let settings;
    try { settings = trialSettings(); }
    catch (error) { setStatus("감지 설정을 확인하세요", error.message, "error"); return; }
    const target = { queue: [], sequence: 0, draining: false, sessionId: null, startRequest: null, settings };
    run = target;
    showDeviceHint("");
    syncControls(); resetMetrics();
    setStatus("마이크 준비 중", "마이크 사용 권한이 나오면 허용하세요.", "preparing");
    try {
      const deviceId = ui.microphone.value;
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { ...(deviceId ? { deviceId: { exact: deviceId } } : {}), channelCount: { ideal: 1 }, sampleRate: { ideal: 16000 }, echoCancellation: { ideal: false }, noiseSuppression: { ideal: false }, autoGainControl: { ideal: false } }, video: false });
      if (!isCurrent(target)) { stream.getTracks().forEach(track => track.stop()); return; }
      target.stream = stream;
      const track = stream.getAudioTracks()[0];
      target.activeDeviceId = track.getSettings().deviceId || deviceId;
      track.onended = () => { if (isCurrent(target)) stop("마이크 연결이 끊겨 시험을 중지했습니다.", true); };
      track.onmute = () => { if (isCurrent(target)) stop("마이크 입력이 일시 중단되어 시험을 중지했습니다. 다시 시작하세요.", true); };
      const context = new AudioContext({ sampleRate: info.sample_rate, latencyHint: "interactive" });
      target.context = context;
      if (context.sampleRate !== 16000) throw new Error("이 브라우저가 16 kHz 마이크 분석을 지원하지 않습니다. 최신 Chrome 또는 Edge에서 열어주세요.");
      await context.audioWorklet.addModule(config.worklet_url || "/microphone_worklet.js");
      if (!isCurrent(target)) return;
      target.startRequest = request("/api/start", { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify(target.settings) });
      const session = await target.startRequest;
      target.startRequest = null;
      if (!isCurrent(target)) return;
      if (!session.session_id) throw new Error("시험 연결 정보를 읽을 수 없습니다.");
      target.sessionId = session.session_id;
      const recommended = target.settings.mode === "recommended";
      const matches = recommended
        ? info.models.every(model => validRuntime(session.runtimes?.[model.id])
          && session.runtimes[model.id].threshold === recommendedRuntimes[model.id].threshold
          && session.runtimes[model.id].consecutive_frames === recommendedRuntimes[model.id].consecutive_frames)
        : session.runtime?.threshold === target.settings.threshold && session.runtime?.consecutive_frames === target.settings.consecutive_frames;
      if (!matches) {
        throw new Error("시험 감지 설정을 확인할 수 없습니다. 화면을 새로고침한 뒤 다시 시작하세요.");
      }
      renderModels(session.models || info.models);
      showTrialRuntime(session.runtime, recommended ? session.runtimes : null);
      target.source = context.createMediaStreamSource(stream);
      target.worklet = new AudioWorkletNode(context, "koww-microphone", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1], channelCountMode: "max" });
      target.silence = context.createGain();
      target.silence.gain.value = 0;
      target.worklet.port.onmessage = event => onFrame(target, event);
      target.worklet.onprocessorerror = () => { if (isCurrent(target)) stop("마이크 오디오 처리가 중단됐습니다. 다시 시작하세요.", true); };
      target.source.connect(target.worklet); target.worklet.connect(target.silence); target.silence.connect(context.destination);
      await context.resume();
      if (!isCurrent(target)) return;
      if (context.state !== "running") throw new Error("브라우저 오디오를 시작하지 못했습니다. 페이지를 다시 열어주세요.");
      target.lastFrameAt ??= performance.now();
      context.onstatechange = () => {
        if (isCurrent(target) && context.state !== "running") stop("브라우저가 마이크 처리를 일시 중단했습니다. 다시 시작하세요.", true);
      };
      target.watchdog = setInterval(() => {
        if (isCurrent(target) && performance.now() - target.lastFrameAt > 3000) stop("마이크 오디오가 3초 동안 들어오지 않아 시험을 중지했습니다. 장치 연결을 확인한 뒤 다시 시작하세요.", true);
      }, 250);
      setStatus("모델 준비 중", `${info.warmup_s}초 뒤에 호출어를 말하세요.`, "preparing");
      refreshDevices();
    } catch (error) {
      target.startRequest = null;
      if (isCurrent(target)) stop(microphoneError(error), true);
    }
  }

  ui.start.addEventListener("click", start);
  ui.stop.addEventListener("click", () => stop());
  ui.connect.addEventListener("click", connectMicrophone);
  ui.refresh.addEventListener("click", () => refreshDevices({requested: true}));
  ui.retry.addEventListener("click", initialize);
  ui["trial-threshold"].addEventListener("input", () => { if (!info || run || probe) return; thresholdEdited = true; resetMetrics(); });
  ui["trial-consecutive"].addEventListener("input", () => { if (info && !run && !probe) resetMetrics(); });
  ui["trial-mode"].addEventListener("change", () => { if (run || probe) return; syncControls(); if (info) resetMetrics(); });
  ui["reset-detection"].addEventListener("click", resetDetection);
  window.addEventListener("pagehide", () => { if (run || probe) stop("페이지를 떠나 마이크 시험을 중지했습니다."); });
  document.addEventListener("visibilitychange", () => { if (document.hidden && (run || probe)) stop("다른 창으로 이동하여 시험을 중지했습니다. 돌아온 뒤 다시 시작하세요."); syncControls(); });
  window.addEventListener("message", event => {
    if (!window.parent || window.parent === window || !window.location?.origin || event.origin !== window.location.origin
        || event.source !== window.parent || event.data?.type !== "koww:pause") return;
    if (run || probe) stop("화면을 이동하여 마이크 시험을 중지했습니다.");
  });
  navigator.mediaDevices?.addEventListener("devicechange", () => { if (!probe) refreshDevices(); });

  async function initialize() {
    if (initializing || run || closingSession) return;
    window.KowwChapterReady?.loading();
    let initializationError = "";
    initializing = true;
    ui.retry.hidden = true;
    setStatus("모델 연결 중", "이 컴퓨터의 시험 모델을 준비하고 있습니다. 처음에는 잠시 걸릴 수 있습니다.", "preparing");
    syncControls();
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode || !window.AudioContext) throw new Error("마이크 시험은 localhost의 최신 Chrome 또는 Edge에서 사용할 수 있습니다.");
      info = await request("/api/info", {}, 30000, new AbortController(), "모델 정보를 30초 안에 받지 못했습니다. 잠시 뒤 모델 연결을 다시 시도하세요.");
      if (info.sample_rate !== 16000 || info.frame_samples !== 1280 || !Array.isArray(info.models) || !info.models.length || !info.token || !Number.isFinite(info.warmup_s)) throw new Error("지원하지 않는 마이크 시험 설정입니다. 서버를 확인하세요.");
      ui.wakeword.textContent = `‘${info.wakeword}’`;
      document.title = `${info.wakeword} 마이크 시험 · Korean openWakeWord`;
      defaultRuntime = info.default_runtime || { threshold: info.threshold, consecutive_frames: info.consecutive_frames };
      if (!Number.isFinite(defaultRuntime.threshold) || defaultRuntime.threshold <= 0 || defaultRuntime.threshold >= 1
          || !Number.isInteger(defaultRuntime.consecutive_frames) || defaultRuntime.consecutive_frames < 1 || defaultRuntime.consecutive_frames > 10) {
        throw new Error("모델의 기본 감지 설정을 읽을 수 없습니다.");
      }
      defaultRuntime = { threshold: defaultRuntime.threshold, consecutive_frames: defaultRuntime.consecutive_frames };
      recommendedRuntimes = info.recommended_runtimes && info.models.every(model => validRuntime(info.recommended_runtimes[model.id]))
        ? info.recommended_runtimes : null;
      const recommendedOption = Array.from(ui["trial-mode"].options).find(option => option.value === "recommended");
      if (recommendedOption) {
        recommendedOption.disabled = !recommendedRuntimes;
        recommendedOption.textContent = info.runtime_recommendations?.candidates?.some(row => row.recommendation_available === false)
          ? "모델별 추천·기존값" : "모델별 추천값";
      }
      ui["trial-mode"].value = recommendedRuntimes ? "recommended" : "manual";
      setDefaultFields();
      renderModels(info.models);
      syncDetectionDisplay();
      setStatus("시험 준비 완료", "");
    } catch (error) {
      initializationError = error.message || "모델 정보를 읽지 못했습니다.";
      info = null;
      ui.retry.hidden = false;
      setStatus("모델 연결을 확인하세요", `${error.message || "모델 정보를 읽지 못했습니다."} 모델 연결 다시 시도를 눌러주세요.`, "error");
    } finally {
      initializing = false;
      syncControls();
      if (initializationError) window.KowwChapterReady?.fail(ui.statusText.textContent);
      else window.KowwChapterReady?.ready();
    }
  }
  refreshDevices();
  initialize();
})();
