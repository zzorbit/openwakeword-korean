"""Memory-only microphone trials through the existing CPU streaming runtime.

Only a loopback browser may submit ordered, exact PCM16 frames. This module
does not record audio, select/tune models, or admit human recordings to data.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import urlsplit
import hmac
import json
import math
import secrets
import threading
import time

import numpy as np

from .events import EventDetector
from .web_page import render_page_config


class MicrophoneError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class MicrophoneService:
    """One ordered microphone stream; callers cannot interleave model history."""
    def __init__(self, detector, idle_timeout=60, clock=time.monotonic, model_labels=None):
        if idle_timeout <= 0:
            raise ValueError("Idle timeout must be positive")
        self.detector = detector
        self.runtime = dict(detector.runtime)
        if self.runtime["sample_rate"] != 16000 or self.runtime["frame_samples"] != 1280:
            raise ValueError("Microphone input requires the canonical 16kHz / 80ms runtime")
        self.token = secrets.token_urlsafe(32)
        labels = model_labels or {}
        self.models = [{"id": key, "label": labels.get(key, f"단일 모델 {i + 1}"), "kind": "single"}
                       for i, key in enumerate(detector.keys)]
        self._ensemble_id = None
        if len(detector.keys) == 3:
            self._ensemble_id = "mean"
            while self._ensemble_id in detector.keys:
                self._ensemble_id = "_" + self._ensemble_id
            self.models.append({"id": self._ensemble_id, "label": "3개 평균 앙상블", "kind": "ensemble"})
        self._lock = threading.Lock()
        self._clock = clock
        self._idle_timeout = idle_timeout
        self._session = None
        self._last_activity = 0.0
        self._next_sequence = 0
        self._gates = {}
        self._stats = {}

    def info(self):
        return {"wakeword": self.detector.manifest["wakeword"],
                "bundle_id": self.detector.manifest.get("id", ""),
                "sample_rate": 16000, "frame_samples": 1280,
                "warmup_s": self.runtime["warmup_s"],
                "threshold": self.runtime["threshold"],
                "consecutive_frames": self.runtime["consecutive_frames"],
                "default_runtime": {key: self.runtime[key] for key in ("threshold", "consecutive_frames")},
                "token": self.token, "models": self.models,
                "audio_saved": False, "training_or_tuning": False,
                "inference_device": "CPU ONNX"}

    def _close(self):
        self._session = None
        self._next_sequence = 0
        self._gates = {}
        self._stats = {}
        self.detector.reset()

    def _expire(self):
        if self._session and self._clock() - self._last_activity >= self._idle_timeout:
            self._close()
            return True
        return False

    def expire(self):
        with self._lock:
            return self._expire()

    def active(self):
        """Check capture ownership without exposing session IDs or microphone audio."""
        with self._lock:
            self._expire()
            return self._session is not None

    def start(self, settings=None):
        with self._lock:
            self._expire()
            if self._session:
                raise MicrophoneError("다른 시험이 진행 중입니다. 먼저 중지하세요.", 409)
            settings = {} if settings is None else settings
            if not isinstance(settings, dict) or set(settings) - {"threshold", "consecutive_frames"}:
                raise MicrophoneError("감지 기준과 연속 감지 수만 설정할 수 있습니다.")
            runtime = dict(self.runtime, **settings)
            threshold, consecutive = runtime["threshold"], runtime["consecutive_frames"]
            if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                    or not math.isfinite(threshold) or not 0 < threshold < 1):
                raise MicrophoneError("감지 기준은 0보다 크고 1보다 작은 숫자여야 합니다.")
            if type(consecutive) is not int or not 1 <= consecutive <= 10:
                raise MicrophoneError("연속 감지 수는 1~10의 정수여야 합니다.")
            self.detector.reset()
            self._session = secrets.token_urlsafe(24)
            self._last_activity = self._clock()
            self._next_sequence = 0
            # Trial-only gates share the unchanged scores and warmup timeline.
            # The bundle's runtime, model weights and offline evaluation stay
            # untouched, including the detector's original ensemble gate.
            self._gates = {model["id"]: EventDetector(runtime["threshold"],
                runtime["consecutive_frames"], runtime["cooldown_s"])
                for model in self.models}
            self._stats = {m["id"]: {"max_score": 0.0, "events": 0, "last_event_s": None}
                           for m in self.models}
            return {"session_id": self._session, "runtime": runtime,
                    "models": self.models, "audio_saved": False}

    def _require_session(self, session_id):
        if self._expire():
            raise MicrophoneError("입력이 오래 끊겨 시험을 종료했습니다. 다시 시작하세요.", 410)
        if not self._session or not isinstance(session_id, str) or not hmac.compare_digest(self._session, session_id):
            raise MicrophoneError("이 시험은 종료됐거나 다른 세션입니다. 다시 시작하세요.", 409)

    def process_frame(self, session_id, sequence, payload):
        with self._lock:
            self._require_session(session_id)
            if type(sequence) is not int or sequence != self._next_sequence:
                self._close()
                raise MicrophoneError("음성 순서가 끊겨 시험을 중단했습니다. 다시 시작하세요.", 409)
            if not isinstance(payload, bytes) or len(payload) != 2560:
                self._close()
                raise MicrophoneError("음성은 1,280개 PCM16 샘플이어야 합니다.")
            # The wire format is little-endian; never change gain or resample.
            frame = np.frombuffer(payload, dtype="<i2").astype(np.int16, copy=True)
            try:
                row = self.detector.process(frame)
                values = row["member_scores"]
                if len(values) != len(self.detector.keys) or not np.isfinite(values).all():
                    raise RuntimeError("Invalid runtime member scores")
                responses = []
                for index, model in enumerate(self.models):
                    key = model["id"]
                    if key == self._ensemble_id:
                        score = float(row["score"])
                    else:
                        score = float(values[index])
                    event = self._gates[key].update(score, row["time_s"], row["eligible"])["event"]
                    stat = self._stats[key]
                    stat["max_score"] = max(stat["max_score"], score)
                    if event:
                        stat["events"] += 1
                        stat["last_event_s"] = row["time_s"]
                    responses.append({**model, "score": score, "event": bool(event), **stat})
                scaled = frame.astype(np.float64) / 32768.0
                response = {"session_id": self._session, "sequence": sequence,
                    "time_s": row["time_s"], "eligible": bool(row["eligible"]),
                    "rms": float(np.sqrt(np.mean(scaled ** 2))),
                    "peak": float(np.max(np.abs(scaled))), "responses": responses}
                # Reject impossible/non-finite detector output before JSON reaches UI.
                json.dumps(response, allow_nan=False)
            except Exception:
                self._close()
                raise
            self._next_sequence += 1
            self._last_activity = self._clock()
            return response

    def stop(self, session_id):
        with self._lock:
            self._expire()
            if self._session:
                self._require_session(session_id)
                self._close()
            return {"stopped": True, "audio_saved": False}


def make_server(service, port=8792):
    """Build the localhost-only server without starting a background thread."""
    resources = files("korean_openwakeword") / "resources"
    routes = {"/": ("microphone.html", "text/html; charset=utf-8"),
              "/microphone.js": ("microphone.js", "text/javascript; charset=utf-8"),
              "/microphone_worklet.js": ("microphone_worklet.js", "text/javascript; charset=utf-8"),
              "/microphone.css": ("microphone.css", "text/css; charset=utf-8")}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            # No audio, transcript, microphone metadata or per-frame logs.
            pass

        def _respond(self, status, payload, content_type="application/json; charset=utf-8"):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Permissions-Policy", "microphone=(self)")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; worker-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def _guard(self, mutate=False):
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            host = self.headers.get("Host", "")
            if host not in hosts:
                raise MicrophoneError("로컬 주소로 접속하세요.", 403)
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{host}":
                raise MicrophoneError("다른 사이트에서 마이크 시험에 접근할 수 없습니다.", 403)
            if mutate and not hmac.compare_digest(self.headers.get("X-Koww-Token", ""), service.token):
                raise MicrophoneError("화면을 새로고침한 뒤 다시 시작하세요.", 403)

        def _body(self, kind):
            if self.headers.get("Transfer-Encoding"):
                raise MicrophoneError("Chunked requests are not supported", 400)
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise MicrophoneError("Content-Length is required", 411)
            if not 0 <= length <= 4096:
                raise MicrophoneError("Request exceeds the bounded microphone frame", 413)
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != kind:
                raise MicrophoneError("Unsupported microphone request type", 415)
            body = self.rfile.read(length)
            if len(body) != length:
                raise MicrophoneError("Incomplete microphone request", 400)
            if kind == "application/json":
                try:
                    body = json.loads(body)
                except (ValueError, UnicodeError):
                    raise MicrophoneError("Invalid JSON request", 400)
                if not isinstance(body, dict):
                    raise MicrophoneError("Expected a JSON object", 400)
            return body

        def do_GET(self):
            try:
                self._guard()
                path = urlsplit(self.path).path
                if path == "/api/info":
                    return self._respond(200, service.info())
                if path not in routes:
                    raise MicrophoneError("Not found", 404)
                name, content_type = routes[path]
                body = (resources / name).read_bytes()
                if name == "microphone.html":
                    body = render_page_config(body.decode("utf-8")).encode("utf-8")
                return self._respond(200, body, content_type)
            except MicrophoneError as error:
                self._respond(error.status, {"error": str(error)})

        def do_POST(self):
            try:
                self._guard(mutate=True)
                path = urlsplit(self.path).path
                if path == "/api/start":
                    body = self._body("application/json")
                    value = service.start(body)
                elif path == "/api/frame":
                    payload = self._body("application/octet-stream")
                    try:
                        sequence = int(self.headers.get("X-Koww-Sequence", ""))
                    except ValueError:
                        raise MicrophoneError("A numeric microphone sequence is required")
                    value = service.process_frame(self.headers.get("X-Koww-Session"), sequence, payload)
                elif path == "/api/stop":
                    body = self._body("application/json")
                    if not isinstance(body.get("session_id"), str):
                        raise MicrophoneError("A microphone session id is required")
                    value = service.stop(body["session_id"])
                else:
                    raise MicrophoneError("Not found", 404)
                self._respond(200, value)
            except MicrophoneError as error:
                self.close_connection = True
                self._respond(error.status, {"error": str(error)})
            except TimeoutError:
                self.close_connection = True
                self._respond(408, {"error": "음성 전달 시간이 초과됐습니다. 다시 시작하세요."})
            except Exception:
                self.close_connection = True
                self._respond(500, {"error": "모델 처리에 실패해 시험을 중단했습니다. 다시 시작하세요."})

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def service_actions(self):
            service.expire()

    return Server(("127.0.0.1", port), Handler)


def serve_microphone(bundle=None, port=8792):
    from .runtime import WakeWordDetector
    if not 1 <= port <= 65535:
        raise ValueError("마이크 시험 포트는 1~65535 사이여야 합니다.")
    print("마이크 시험 모델을 준비하고 있습니다…", flush=True)
    service = MicrophoneService(WakeWordDetector(bundle))
    try:
        server = make_server(service, port)
    except OSError as error:
        raise RuntimeError(f"포트 {port}를 열 수 없습니다. 이미 열린 화면을 사용하거나 --port로 바꾸세요.") from error
    print(f"마이크 시험: http://127.0.0.1:{server.server_port}/", flush=True)
    print(f"호출어: {service.info()['wakeword']} · 녹음 저장 없음 · 종료: Ctrl+C", flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        with service._lock:
            if service._session:
                service._close()
        server.server_close()

