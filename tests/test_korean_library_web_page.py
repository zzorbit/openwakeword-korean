"""Page configuration is data; localhost review still saves bound answers."""
import http.client
import json
from pathlib import Path
import re
import threading
from types import SimpleNamespace
from http.server import ThreadingHTTPServer

import pytest

from korean_openwakeword import review
from korean_openwakeword.common import digest, read_json
from korean_openwakeword.project import create_project
from korean_openwakeword.web_page import render_page_config


def config_in(page):
    return json.loads(re.search(r'<script id="koww-page-config" type="application/json">(.*?)</script>', page).group(1))


def test_page_config_is_nonexecutable_json_and_escapes_html_and_line_separators():
    value = '</script><img src=x onerror="bad()">&\u2028\u2029'
    template = '<head><!--KOWW_PAGE_CONFIG--></head><script src="client.js"></script>'
    page = render_page_config(template, {"token": value, "queue_url": "/queue?one=1&two=2"})
    assert config_in(page) == {"token": value, "queue_url": "/queue?one=1&two=2"}
    assert page.count("</script>") == 2
    assert '<img' not in page and "\u2028" not in page and "\u2029" not in page
    assert "\\u003c" in page and "\\u003e" in page and "\\u0026" in page
    assert render_page_config("<!--KOWW_PAGE_CONFIG-->").endswith("{}</script>")


@pytest.mark.parametrize("template", ["no slot", "<!--KOWW_PAGE_CONFIG--><!--KOWW_PAGE_CONFIG-->"])
def test_page_config_requires_one_explicit_slot(template):
    with pytest.raises(ValueError, match="one configuration slot"):
        render_page_config(template, {"token": "fixture"})


def test_page_config_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        render_page_config("<!--KOWW_PAGE_CONFIG-->", {"value": float("nan")})


def test_standalone_review_config_audio_and_authenticated_origin_save(tmp_path, monkeypatch):
    project = create_project(tmp_path / "word", "보라야", "bora")
    root = project.parent
    audio = root / "review" / "preview.wav"
    audio.write_bytes(b"exact fixture audio")
    queue = root / "review" / "queue.jsonl"
    queue.write_text(json.dumps({"sample_id": "one", "path": str(audio), "sha256": digest(audio),
        "voice_family_id": "synthetic-fixture", "text": "샘플 문장"}) + "\n", encoding="utf-8")
    captured = []
    # Capture the public command's handler, then mount it on a random test
    # port. Origin remains its configured command address, not an open CORS rule.
    def capture(address, handler):
        assert address == ("127.0.0.1", 8790)
        captured.append(handler)
        return SimpleNamespace(serve_forever=lambda: None)
    monkeypatch.setattr(review, "ThreadingHTTPServer", capture)
    review.serve_review(project, queue, "references", 8790)
    server = ThreadingHTTPServer(("127.0.0.1", 0), captured[0])
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    worker.start()

    def request(method, path, payload=None, origin=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        headers = {"Content-Type": "application/json"}
        if origin is not None:
            headers["Origin"] = origin
        body = None if payload is None else json.dumps(payload).encode()
        try:
            connection.request(method, path, body, headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    try:
        status, body = request("GET", "/")
        assert status == 200 and config_in(body.decode()) == {}
        status, body = request("GET", "/queue")
        assert status == 200 and json.loads(body)["items"][0]["audio"] == "/audio/0"
        assert request("GET", "/audio/0") == (200, audio.read_bytes())
        assert request("GET", "/audio/-1")[0] == 404
        answers = {"one": {"decision": "keep", "listened": True}}
        assert request("POST", "/save", answers, "https://foreign.invalid")[0] == 400
        assert not (root / "review/references_results.json").exists()
        assert request("POST", "/save", {"one": {"decision": "keep", "listened": False}}, "http://127.0.0.1:8790")[0] == 400
        status, body = request("POST", "/save", answers, "http://127.0.0.1:8790")
        assert status == 200 and json.loads(body)["kept"] == 1
        assert read_json(root / "review/references_results.json")["decisions"] == answers
        assert read_json(root / "approval.json")["accepted_reference_families"] == ["synthetic-fixture"]
        audio.write_bytes(b"changed")
        assert request("GET", "/audio/0")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()
