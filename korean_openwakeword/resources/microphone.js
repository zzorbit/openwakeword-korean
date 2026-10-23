"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const ui = Object.fromEntries(["wakeword", "microphone", "connect", "refresh", "retry", "deviceHint", "start", "stop", "connection", "statusHeading", "statusText", "statusDot", "level", "levelText", "elapsed", "frameCount", "lastEvent", "lastEventTime", "models", "threshold", "captureSettings"].map(id => [id, $(id)]));
  const rows = new Map();
  let info = null;
  let run = null;
  let generation = 0;
  let refreshing = false;
  let probe = null;
  let initializing = false;

  function setStatus(heading, text, phase = "idle") {
    ui.statusHeading.textContent = heading;
    ui.statusText.textContent = text;
    ui.statusDot.className = `status-dot ${phase}`;
  }

  function syncControls() {
    ui.start.disabled = !info || !!run || !!probe || document.hidden;
    ui.stop.disabled = !run && !probe;
    ui.microphone.disabled = !!run || !!probe;
    ui.connect.disabled = !!run || !!probe || refreshing || document.hidden || !window.isSecureContext || !navigator.mediaDevices?.getUserMedia;
    ui.refresh.disabled = !!run || refreshing || !!probe || !navigator.mediaDevices?.enumerateDevices;
    ui.retry.disabled = initializing;
  }

  async function request(path, options = {}, timeoutMs = 5000, controller = new AbortController(), timeoutMessage = "서버 응답이 늦어 시험을 중지했습니다. 서버 상태를 확인한 뒤 다시 시작하세요.") {
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, { ...options, signal: controller.signal, cache: "no-store" });
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
  function isCurrent(target) { return run === target && !target.cancelled && target.generation === generation; }

  function renderModels(models) {
    ui.models.replaceChildren();
    rows.clear();
    for (const model of models) {
      const tr = document.createElement("tr");
      if (model.kind === "ensemble") tr.className = "ensemble";
      const cells = [model.label, "0.0000", "0.0000", "0", "—"].map(text => {
        const td = document.createElement("td");
        td.textContent = text;
        tr.append(td);
        return td;
      });
      rows.set(model.id, { tr, cells, ensemble: model.kind === "ensemble", flashUntil: 0 });
      ui.models.append(tr);
    }
  }

  function resetMetrics() {
    ui.level.value = 0;
    ui.levelText.textContent = "대기 중";
    ui.elapsed.textContent = "0.0초";
    ui.frameCount.textContent = "0개 구간";
    ui.lastEvent.textContent = "아직 없음";
    ui.lastEventTime.textContent = "시작 후 호출어를 말하세요.";
    renderModels(info.models);
  }

  function renderSettings(stream, context) {
    const settings = stream.getAudioTracks()[0].getSettings();
    const settingBoolean = value => typeof value === "boolean" ? value ? "켜짐" : "꺼짐" : "브라우저가 제공하지 않음";
    const entries = [
      ["마이크", stream.getAudioTracks()[0].label || "선택한 입력 장치"],
      ["입력 샘플레이트", settings.sampleRate ? `${settings.sampleRate.toLocaleString()} Hz` : "브라우저가 제공하지 않음"],
      ["분석 샘플레이트", `${context.sampleRate.toLocaleString()} Hz`],
      ["입력 채널 수", settings.channelCount ?? "브라우저가 제공하지 않음"],
      ["에코 제거", settingBoolean(settings.echoCancellation)],
      ["소음 억제", settingBoolean(settings.noiseSuppression)],
      ["자동 음량 조절", settingBoolean(settings.autoGainControl)]
    ];
    ui.captureSettings.replaceChildren();
    for (const [name, value] of entries) {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = name; dd.textContent = String(value);
      ui.captureSettings.append(dt, dd);
    }
  }

  async function refreshDevices(expectedProbe = null) {
    if (refreshing || !navigator.mediaDevices?.enumerateDevices) return;
    refreshing = true; syncControls();
    const selected = ui.microphone.value;
    try {
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "audioinput");
      if (expectedProbe && probe !== expectedProbe) return;
      ui.microphone.replaceChildren(new Option("시스템 기본 마이크", ""));
      let number = 0;
      for (const device of devices) {
        if (!device.deviceId || device.deviceId === "default") continue;
        ui.microphone.append(new Option(device.label || `마이크 ${++number} (권한 허용 후 이름 표시)`, device.deviceId));
      }
      if (Array.from(ui.microphone.options).some(option => option.value === selected)) ui.microphone.value = selected;
      ui.deviceHint.textContent = devices.some(device => device.label) ? "입력 장치를 선택한 뒤 시험을 시작하세요." : "마이크 연결을 누르고 사용을 허용하면 장치 이름을 확인할 수 있습니다.";
      if (run?.activeDeviceId && !devices.some(device => device.deviceId === run.activeDeviceId)) {
        stop("선택한 마이크 연결이 끊겨 시험을 중지했습니다.", true);
      }
    } catch (error) { if (!expectedProbe || probe === expectedProbe) ui.deviceHint.textContent = `마이크 목록을 읽지 못했습니다: ${error.message}`; }
    finally { refreshing = false; syncControls(); }
  }

  function cancelProbe() {
    const target = probe;
    probe = null;
    for (const track of target?.stream?.getTracks() || []) track.stop();
    if (target) ui.deviceHint.textContent = "마이크 연결을 취소했습니다. 다시 연결하려면 마이크 연결을 누르세요.";
  }

  async function connectMicrophone() {
    if (probe || run || refreshing || document.hidden || !window.isSecureContext || !navigator.mediaDevices?.getUserMedia) return;
    const target = {};
    probe = target;
    ui.deviceHint.textContent = "브라우저의 마이크 사용 요청에서 허용을 선택하세요. 중지를 누르면 연결을 취소합니다.";
    syncControls();
    try {
      target.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      if (probe !== target) return;
      await refreshDevices(target);
    } catch (error) {
      if (probe === target) ui.deviceHint.textContent = microphoneError(error);
    } finally {
      for (const track of target.stream?.getTracks() || []) track.stop();
      if (probe === target) probe = null;
      syncControls();
    }
  }

  function closeResources(target) {
    target.cancelled = true;
    if (target.watchdog) clearInterval(target.watchdog);
    target.queue.length = 0;
    target.frameController?.abort();
    if (target.worklet) target.worklet.port.onmessage = null;
    for (const node of [target.source, target.worklet, target.silence]) { try { node?.disconnect(); } catch (_) {} }
    for (const track of target.stream?.getTracks() || []) { track.onended = null; track.onmute = null; track.stop(); }
    if (target.context) target.context.onstatechange = null;
    if (target.context && target.context.state !== "closed") target.context.close().catch(() => {});
  }

  function notifyStop(sessionId, keepalive = false) {
    if (!sessionId || !info) return Promise.resolve();
    return fetch("/api/stop", { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: JSON.stringify({ session_id: sessionId }), keepalive }).catch(() => {});
  }

  function stop(message = "마이크 사용을 중지했습니다. 다시 시작하면 새 시험이 시작됩니다.", error = false, keepalive = false) {
    cancelProbe();
    const target = run;
    generation++;
    run = null;
    if (target) {
      closeResources(target);
      notifyStop(target.sessionId, keepalive);
    }
    ui.level.value = 0;
    ui.levelText.textContent = "마이크 중지됨";
    ui.connection.textContent = "연결 종료";
    setStatus(error ? "시험 중지 · 확인 필요" : "시험 중지", message, error ? "error" : "idle");
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
      row.cells[1].textContent = response.score.toFixed(4);
      row.cells[2].textContent = response.max_score.toFixed(4);
      row.cells[3].textContent = response.events.toLocaleString();
      row.cells[4].textContent = Number.isFinite(response.last_event_s) ? `${response.last_event_s.toFixed(2)}초` : "—";
      if (response.event) { detected.push(response.label); row.flashUntil = performance.now() + 700; }
      row.tr.className = `${row.ensemble ? "ensemble " : ""}${performance.now() < row.flashUntil ? "detected" : ""}`.trim();
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
    if (!info || run || probe || document.hidden) return;
    const target = { generation: ++generation, cancelled: false, queue: [], sequence: 0, draining: false, sessionId: null };
    run = target;
    syncControls(); resetMetrics();
    ui.connection.textContent = "마이크 연결 중";
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
      await context.audioWorklet.addModule("/microphone_worklet.js");
      if (!isCurrent(target)) return;
      const session = await request("/api/start", { method: "POST", headers: headers({ "Content-Type": "application/json" }), body: "{}" });
      if (!session.session_id) throw new Error("시험 연결 정보를 읽을 수 없습니다.");
      target.sessionId = session.session_id;
      if (!isCurrent(target)) { notifyStop(target.sessionId, true); return; }
      renderModels(session.models || info.models);
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
      renderSettings(stream, context);
      ui.connection.textContent = "이 컴퓨터에서 분석 중";
      setStatus("모델 준비 중", `${info.warmup_s}초 뒤에 호출어를 말하세요.`, "preparing");
      refreshDevices();
    } catch (error) { if (isCurrent(target)) stop(microphoneError(error), true); }
  }

  ui.start.addEventListener("click", start);
  ui.stop.addEventListener("click", () => stop());
  ui.connect.addEventListener("click", connectMicrophone);
  ui.refresh.addEventListener("click", () => refreshDevices());
  ui.retry.addEventListener("click", initialize);
  window.addEventListener("pagehide", () => { if (run || probe) stop("페이지를 떠나 마이크 시험을 중지했습니다.", false, true); });
  document.addEventListener("visibilitychange", () => { if (document.hidden && (run || probe)) stop("다른 창으로 이동하여 시험을 중지했습니다. 돌아온 뒤 다시 시작하세요.", false, true); syncControls(); });
  navigator.mediaDevices?.addEventListener("devicechange", () => { if (!probe) refreshDevices(); });

  async function initialize() {
    if (initializing || run) return;
    initializing = true;
    ui.retry.hidden = true;
    ui.connection.textContent = "모델 연결 중";
    setStatus("모델 연결 중", "이 컴퓨터의 시험 모델을 준비하고 있습니다. 처음에는 잠시 걸릴 수 있습니다.", "preparing");
    syncControls();
    try {
      if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode || !window.AudioContext) throw new Error("마이크 시험은 localhost의 최신 Chrome 또는 Edge에서 사용할 수 있습니다.");
      info = await request("/api/info", {}, 30000, new AbortController(), "모델 정보를 30초 안에 받지 못했습니다. 잠시 뒤 모델 연결을 다시 시도하세요.");
      if (info.sample_rate !== 16000 || info.frame_samples !== 1280 || !Array.isArray(info.models) || !info.models.length || !info.token || !Number.isFinite(info.warmup_s)) throw new Error("지원하지 않는 마이크 시험 설정입니다. 서버를 확인하세요.");
      ui.wakeword.textContent = `‘${info.wakeword}’`;
      document.title = `${info.wakeword} 마이크 시험 · Korean openWakeWord`;
      ui.threshold.textContent = Number.isFinite(info.threshold) ? `고정 임계값 ${info.threshold.toFixed(4)}` : "동봉 실행 설정 사용";
      renderModels(info.models);
      ui.connection.textContent = "모델 준비됨";
      setStatus("시작할 준비가 됐습니다", "시험 시작을 누르면 마이크를 연결합니다.");
    } catch (error) {
      info = null;
      ui.retry.hidden = false;
      ui.connection.textContent = "모델 연결 실패";
      setStatus("모델 연결을 확인하세요", `${error.message || "모델 정보를 읽지 못했습니다."} 모델 연결 다시 시도를 눌러주세요.`, "error");
    } finally {
      initializing = false;
      syncControls();
    }
  }
  refreshDevices();
  initialize();
})();
