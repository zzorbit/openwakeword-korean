"""Memory-only live microphone protocol checks; no hardware or model downloads."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import threading

import numpy as np
import pytest

from korean_openwakeword.events import EventDetector
from korean_openwakeword.microphone import MicrophoneError, MicrophoneService, make_server


class FakeDetector:
    """Stateful scores exercise stream isolation without importing ONNX or Torch."""
    manifest = {"wakeword": "보라야"}
    runtime = dict(sample_rate=16000, frame_samples=1280, warmup_s=2,
                   threshold=.5, consecutive_frames=2, cooldown_s=1)
    keys = ["head1", "head2", "head3"]

    def __init__(self):
        self.resets = 0
        self.frames = []
        self.received = []

    def reset(self):
        self.resets += 1
        self.frames = []

    def process(self, frame):
        assert frame.dtype == np.int16
        assert frame.shape == (1280,)
        self.frames.append(frame.copy())
        self.received.append(frame.copy())
        number = len(self.frames)
        score = .8 if number % 2 else .3
        eligible = number > 25
        return dict(time_s=number * .08, score=score,
                    member_scores=[score - .1, score, score + .1],
                    eligible=eligible, event=eligible and number % 2 == 1,
                    candidate=number % 2 == 1,
                    suppression=None if eligible else "warmup",
                    above_threshold=score >= .5, qualified=score >= .5,
                    consecutive_count=int(score >= .5))


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


def pcm(samples=None):
    if samples is None:
        samples = np.zeros(1280, dtype="<i2")
    return np.asarray(samples, dtype="<i2").tobytes()


@pytest.fixture
def detector():
    return FakeDetector()


@pytest.fixture
def service(detector):
    return MicrophoneService(detector)


def assert_status(status, function, *args):
    with pytest.raises(MicrophoneError) as caught:
        function(*args)
    assert caught.value.status == status
    assert str(caught.value)
    return caught.value


def test_info_preserves_bundle_rules_without_starting_stream(service, detector):
    info = service.info()
    assert info["wakeword"] == "보라야"
    assert info["sample_rate"] == detector.runtime["sample_rate"]
    assert info["frame_samples"] == detector.runtime["frame_samples"]
    assert info["threshold"] == detector.runtime["threshold"]
    assert info["warmup_s"] == detector.runtime["warmup_s"]
    assert [model["id"] for model in info["models"]] == [*detector.keys, "mean"]
    assert info["audio_saved"] is False and info["training_or_tuning"] is False
    assert isinstance(info["token"], str) and len(info["token"]) >= 24
    assert detector.resets == 0 and detector.frames == []


def test_active_capture_and_selected_model_labels_keep_original_identity(detector):
    service = MicrophoneService(detector, model_labels={"head1": "모델 2"})
    assert service.info()["models"][0]["label"] == "모델 2"
    assert not service.active()
    started = service.start()
    assert service.active()
    service.stop(started["session_id"])
    assert not service.active()


def test_exact_little_endian_pcm_and_model_responses_are_not_modified(service, detector):
    started = service.start()
    samples = np.resize(np.array([-32768, -1234, 0, 1234, 32767], dtype="<i2"), 1280)
    row = service.process_frame(started["session_id"], 0, pcm(samples))
    np.testing.assert_array_equal(detector.frames[0], samples)
    assert started["runtime"] == detector.runtime
    assert len(started["models"]) == 4
    assert row["time_s"] == .08 and row["eligible"] is False
    assert [response["score"] for response in row["responses"]] == pytest.approx([.7, .8, .9, .8])
    assert row["peak"] == pytest.approx(1.)
    assert row["rms"] == pytest.approx(np.sqrt(np.mean((samples.astype(np.float64) / 32768) ** 2)))


def test_warmup_and_events_remain_the_detector_decisions(service):
    session = service.start()["session_id"]
    rows = [service.process_frame(session, index, pcm()) for index in range(27)]
    assert all(not row["eligible"] and not any(response["event"] for response in row["responses"])
               for row in rows[:25])
    assert rows[25]["eligible"] and not rows[25]["responses"][-1]["event"]
    assert rows[26]["eligible"] and rows[26]["responses"][-1]["event"]
    assert rows[-1]["responses"][-1]["events"] == 1
    assert rows[-1]["responses"][-1]["last_event_s"] == pytest.approx(2.16)
    assert [response["max_score"] for response in rows[-1]["responses"]] == pytest.approx([.7, .8, .9, .8])
    assert rows[-1]["time_s"] == pytest.approx(2.16)
    assert rows[-1]["peak"] == 0 and rows[-1]["rms"] == 0


def test_single_stream_rejects_second_start_and_other_session_without_reset(service, detector):
    session = service.start()["session_id"]
    assert_status(409, service.start)
    assert_status(409, service.process_frame, "not-this-session", 0, pcm())
    assert_status(409, service.stop, "not-this-session")
    assert detector.resets == 1
    service.process_frame(session, 0, pcm())
    assert len(detector.frames) == 1


def test_stop_then_restart_resets_history_and_uses_new_identifier(service, detector):
    first = service.start()["session_id"]
    service.process_frame(first, 0, pcm())
    assert service.stop(first)["stopped"]
    second = service.start()["session_id"]
    assert second != first and detector.resets == 3 and not detector.frames
    assert_status(409, service.stop, first)
    row = service.process_frame(second, 0, pcm())
    assert row["time_s"] == .08


@pytest.mark.parametrize("sequence", [1, -1, 100])
def test_first_frame_must_have_sequence_zero(service, detector, sequence):
    session = service.start()["session_id"]
    assert_status(409, service.process_frame, session, sequence, pcm())
    assert not detector.frames
    # A gap cannot silently skip input. The invalidated stream must be restarted.
    assert_status(409, service.process_frame, session, 0, pcm())
    restarted = service.start()["session_id"]
    service.process_frame(restarted, 0, pcm())


@pytest.mark.parametrize("sequence", [0, 2])
def test_repeated_or_missing_frame_terminates_stream(service, detector, sequence):
    session = service.start()["session_id"]
    service.process_frame(session, 0, pcm())
    assert_status(409, service.process_frame, session, sequence, pcm())
    assert len(detector.received) == 1
    assert_status(409, service.process_frame, session, 1, pcm())


@pytest.mark.parametrize("length", [0, 1, 2558, 2562, 8192])
def test_wrong_frame_length_never_reaches_detector(service, detector, length):
    session = service.start()["session_id"]
    assert_status(400, service.process_frame, session, 0, bytes(length))
    assert not detector.frames
    assert_status(409, service.process_frame, session, 0, pcm())
    service.start()


def test_idle_stream_expires_and_old_session_cannot_touch_replacement(detector):
    clock = Clock()
    service = MicrophoneService(detector, idle_timeout=60, clock=clock)
    first = service.start()["session_id"]
    clock.now += 61
    assert_status(410, service.process_frame, first, 0, pcm())
    assert not detector.frames
    second = service.start()["session_id"]
    assert_status(409, service.stop, first)
    service.process_frame(second, 0, pcm())


def test_processing_activity_extends_idle_timeout(detector):
    clock = Clock()
    service = MicrophoneService(detector, idle_timeout=60, clock=clock)
    session = service.start()["session_id"]
    clock.now += 59
    service.process_frame(session, 0, pcm())
    clock.now += 59
    assert service.process_frame(session, 1, pcm())["time_s"] == .16


def test_simultaneous_start_has_exactly_one_owner(service, detector):
    barrier = threading.Barrier(2)
    def start():
        barrier.wait(timeout=3)
        try:
            return service.start()
        except MicrophoneError as error:
            return error.status
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start(), range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert results.count(409) == 1 and detector.resets == 1


@pytest.mark.parametrize("name", ["single", "mean"])
def test_single_model_has_no_phantom_ensemble_and_preserves_event_rule(name):
    class SingleDetector(FakeDetector):
        keys = [name]

        def reset(self):
            super().reset()
            self.gate = EventDetector(self.runtime["threshold"],
                                      self.runtime["consecutive_frames"], self.runtime["cooldown_s"])

        def process(self, frame):
            row = super().process(frame)
            row.update(score=.8, member_scores=[.8])
            row.update(self.gate.update(.8, row["time_s"], row["eligible"]))
            return row

    detector = SingleDetector()
    service = MicrophoneService(detector)
    assert [model["id"] for model in service.info()["models"]] == [name]
    session = service.start()["session_id"]
    rows = [service.process_frame(session, sequence, pcm()) for sequence in range(28)]
    assert all(len(row["responses"]) == 1 for row in rows)
    assert not any(row["responses"][0]["event"] for row in rows[:25])
    assert rows[25]["responses"][0]["event"]
    assert not rows[26]["responses"][0]["event"]
    assert rows[-1]["responses"][0]["events"] == 1


@pytest.mark.parametrize("names", [["mean", "head2", "head3"], ["mean", "_mean", "__mean"]])
def test_member_named_mean_keeps_independent_score_and_statistics(names):
    detector = FakeDetector()
    detector.keys = names
    service = MicrophoneService(detector)
    ids = [model["id"] for model in service.info()["models"]]
    assert ids[:3] == names and len(ids) == len(set(ids)) == 4
    assert ids[-1] not in names
    session = service.start()["session_id"]
    rows = [service.process_frame(session, sequence, pcm()) for sequence in range(27)]
    # A member's basename must not turn its raw score into the ensemble score.
    assert [response["score"] for response in rows[0]["responses"]] == pytest.approx([.7, .8, .9, .8])
    last = {response["id"]: response for response in rows[-1]["responses"]}
    assert last[names[0]]["max_score"] == pytest.approx(.7)
    assert last[ids[-1]]["max_score"] == pytest.approx(.8)
    assert last[names[0]]["events"] == 0
    assert last[ids[-1]]["events"] == 1
    assert last[names[0]]["last_event_s"] is None
    assert last[ids[-1]]["last_event_s"] == pytest.approx(2.16)


def test_detector_failure_terminates_stream_without_reusing_partial_history(service, detector, monkeypatch):
    session = service.start()["session_id"]
    original = detector.process
    def failure(frame):
        raise RuntimeError("fixture inference failure")
    monkeypatch.setattr(detector, "process", failure)
    with pytest.raises(RuntimeError, match="fixture inference failure"):
        service.process_frame(session, 0, pcm())
    assert_status(409, service.process_frame, session, 0, pcm())
    monkeypatch.setattr(detector, "process", original)
    restarted = service.start()["session_id"]
    assert service.process_frame(restarted, 0, pcm())["time_s"] == .08


def test_cli_microphone_defaults_and_explicit_bundle(tmp_path):
    from korean_openwakeword.cli import parser
    argument_parser = parser()
    default = argument_parser.parse_args(["mic"])
    assert default.bundle is None and default.port == 8792
    bundle = tmp_path / "portable bundle"
    explicit = argument_parser.parse_args(["mic", "--bundle", str(bundle), "--port", "8800"])
    assert explicit.bundle == bundle and explicit.port == 8800
    assert "마이크" in argument_parser.format_help()


def test_cli_microphone_dispatch_does_not_load_training_or_model(monkeypatch, tmp_path):
    from korean_openwakeword.cli import dispatch, parser
    observed = []
    def serve(bundle, port):
        observed.append((bundle, port))
        return "served"
    monkeypatch.setattr("korean_openwakeword.microphone.serve_microphone", serve)
    bundle = tmp_path / "own_bundle"
    args = parser().parse_args(["mic", "--bundle", str(bundle), "--port", "8800"])
    assert dispatch(args) == "served"
    assert observed == [(bundle, 8800)]


@pytest.fixture
def web(service, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server = make_server(service, port=0)
    assert server.server_address[0] in ("127.0.0.1", "localhost")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    port = server.server_address[1]
    def request(method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()
    yield request, service, port
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    # Testing receives live audio in RAM only; it must not create recordings or scores.
    assert not list(tmp_path.iterdir())


def api(web, path, body, *, headers=None):
    request, service, _ = web
    common = {"X-Koww-Token": service.info()["token"], "Content-Type": "application/json"}
    common.update(headers or {})
    status, response_headers, payload = request("POST", path, body, common)
    return status, response_headers, json.loads(payload)


def test_local_static_assets_and_public_info_only(web):
    request, service, _ = web
    for path in ("/", "/microphone.js", "/microphone.css", "/microphone_worklet.js"):
        status, headers, body = request("GET", path)
        assert status == 200 and body
        assert headers.get("Access-Control-Allow-Origin") is None
        if path == "/":
            assert '<script id="koww-page-config" type="application/json">{}</script>' in body.decode()
            assert 'src="microphone.js"' in body.decode() and 'href="microphone.css"' in body.decode()
    status, _, body = request("GET", "/api/info")
    assert status == 200
    assert json.loads(body)["token"] == service.info()["token"]
    for path in ("/project.json", "/manifest.json", "/../README.md", "/%2e%2e/README.md"):
        assert request("GET", path)[0] == 404


def test_http_start_frame_stop_round_trip_preserves_pcm(web, detector):
    request, service, _ = web
    status, _, started = api(web, "/api/start", b"{}")
    assert status == 200
    frame = np.full(1280, -3210, dtype="<i2")
    status, _, body = request("POST", "/api/frame", pcm(frame), {
        "X-Koww-Token": service.info()["token"],
        "Content-Type": "application/octet-stream",
        "X-Koww-Session": started["session_id"], "X-Koww-Sequence": "0"})
    assert status == 200 and json.loads(body)["time_s"] == .08
    np.testing.assert_array_equal(detector.frames[0], frame)
    status, _, stopped = api(web, "/api/stop", json.dumps({"session_id": started["session_id"]}))
    assert status == 200 and stopped["stopped"]


def test_http_session_conflict_does_not_overwrite_owner(web):
    status, _, first = api(web, "/api/start", b"{}")
    assert status == 200
    assert api(web, "/api/start", b"{}")[0] == 409
    assert api(web, "/api/stop", b'{"session_id":"wrong"}')[0] == 409
    assert api(web, "/api/stop", json.dumps({"session_id": first["session_id"]}))[0] == 200


@pytest.mark.parametrize("token", [None, "invalid"])
def test_mutating_http_requests_require_unpredictable_token(web, detector, token):
    request, _, _ = web
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Koww-Token"] = token
    status, _, body = request("POST", "/api/start", b"{}", headers)
    assert status == 403 and json.loads(body)["error"]
    assert detector.resets == 0


@pytest.mark.parametrize("origin", ["https://example.org", "null", "http://127.0.0.1:1", "http://localhost:1"])
def test_foreign_origin_cannot_read_token_or_start_microphone(web, detector, origin):
    request, service, _ = web
    assert request("GET", "/api/info", headers={"Origin": origin})[0] == 403
    status, _, _ = request("POST", "/api/start", b"{}", {
        "Origin": origin, "X-Koww-Token": service.info()["token"], "Content-Type": "application/json"})
    assert status == 403 and detector.resets == 0


@pytest.mark.parametrize("host", ["example.org", "127.0.0.1:1", "localhost:1", "127.0.0.1.evil.org"])
def test_dns_rebinding_host_cannot_read_service(web, host):
    request, _, _ = web
    assert request("GET", "/api/info", headers={"Host": host})[0] in (400, 403)


def test_same_origin_localhost_is_allowed(web):
    request, _, port = web
    status, _, body = request("GET", "/api/info", headers={
        "Host": f"localhost:{port}", "Origin": f"http://localhost:{port}"})
    assert status == 200 and json.loads(body)["wakeword"] == "보라야"


def test_json_media_type_and_body_limits(web, detector):
    assert api(web, "/api/start", b"{}", headers={"Content-Type": "text/plain"})[0] == 415
    assert api(web, "/api/start", b"not-json")[0] == 400
    assert api(web, "/api/start", b"x" * 4097)[0] == 413
    assert api(web, "/api/unknown", b"{}")[0] == 404
    assert detector.resets == 0


def test_frame_media_type_is_required_before_processing(web, detector):
    request, service, _ = web
    _, _, started = api(web, "/api/start", b"{}")
    status, _, body = request("POST", "/api/frame", pcm(), {
        "X-Koww-Token": service.info()["token"], "Content-Type": "application/json",
        "X-Koww-Session": started["session_id"], "X-Koww-Sequence": "0"})
    assert status == 415 and json.loads(body)["error"]
    assert not detector.frames
