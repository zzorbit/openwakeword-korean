"""One complete review queue, automatic keep after listening, server-side saving."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import json
import mimetypes

from .common import digest, identity, now, read_json, read_rows, resolved, write_json, snapshot_json
from .project import load_project


def apply_review(project, queue, decisions, kind):
    root, config = load_project(project)
    queue = Path(queue).resolve()
    rows = read_rows(queue)
    by_id = {r["sample_id"]: r for r in rows}
    if not rows or len(by_id) != len(rows) or kind not in ("references", "pilot", "content"):
        raise ValueError("A nonempty unique review queue and supported review kind are required")
    if set(decisions) != set(by_id):
        raise ValueError("Every queued item must be listened to or explicitly rejected/review-held")
    for key, value in decisions.items():
        if value.get("decision") not in ("keep", "discard", "review"):
            raise ValueError("Invalid disposition")
        if value["decision"] == "keep" and not value.get("listened"):
            raise ValueError("Unheard audio cannot be silently marked listened/kept")
        if digest(resolved(root, by_id[key]["path"])) != by_id[key]["sha256"]:
            raise ValueError("Review audio changed")
    report = {"kind": kind + "_listening_review", "queue_sha256": digest(queue), "wakeword": config["wakeword"],
        "queued_audio_sha256": {r["sample_id"]: r["sha256"] for r in rows},
        "decisions": decisions, "saved_at": now(), "source_scope": "Human dispositions entered through the local review screen"}
    write_json(root / "review" / (kind + "_results.json"), report)
    archive = root / "review" / "evidence" / (kind + "_" + digest(root / "review" / (kind + "_results.json")) + ".json")
    write_json(archive, report)
    kept = [r for r in rows if decisions[r["sample_id"]]["decision"] == "keep"]
    approval = read_json(root / "approval.json")
    if kind == "references":
        reviewed_families = {r["voice_family_id"] for r in rows}
        kept_families = {r["voice_family_id"] for r in kept}
        approval["accepted_reference_families"] = sorted(
            (set(approval.get("accepted_reference_families", [])) - reviewed_families) | kept_families)
        approval["reference_review_sha256"] = digest(root / "review" / (kind + "_results.json"))
        approval.setdefault("reference_review_evidence", []).append({"path": str(archive.relative_to(root)), "sha256": digest(archive)})
        write_json(root / "approval.json", approval)
    elif kind == "pilot":
        if any(r.get("tts_settings_id") != identity(config["tts"]) for r in rows) or len({r.get("generation_method") for r in rows}) != 1 or rows[0].get("generation_method") not in ("clone", "design"):
            raise ValueError("Pilot method/settings do not match this project's generation policy")
        if not kept or any(decisions[r["sample_id"]]["decision"] == "review" for r in rows):
            approval.update(state="review_incomplete_or_policy_rejected", generation_policy_approved=False)
            write_json(root / "approval.json", approval)
            return {"state": "review_incomplete_or_policy_rejected", "kept": len(kept)}
        # A rejected pilot must not silently approve its unmodified generation method.
        if len(kept) != len(rows):
            approval.update(state="pilot_exceptions_need_policy_fix", generation_policy_approved=False)
            write_json(root / "approval.json", approval)
            return {"state": "pilot_exceptions_need_policy_fix", "kept": len(kept), "total": len(rows)}
        approval.update(state="accepted", generation_policy_approved=True, human_reference_used=False,
            wakeword=config["wakeword"], tts_settings_id=identity(config["tts"]), generation_method=rows[0]["generation_method"],
            pilot_review_sha256=digest(root / "review" / (kind + "_results.json")),
            accepted_pilot_items=len(kept), approval_scope="generation policy and representative pronunciation; not every production clip was listened to")
        write_json(root / "approval.json", approval)
    elif kind == "content":
        if any(decisions[r["sample_id"]]["decision"] == "review" for r in rows):
            return {"state": "content_exceptions_pending", "kept": len(kept)}
        hashes = []
        for row in kept:
            if row.get("provenance") == "qwen_tts":
                from .data import validate_hard_negative
                validate_hard_negative(row, config)
                if row.get("split") != "train":
                    raise ValueError("Synthetic Hard Negative review requires train split")
            elif row.get("provenance") != "public_dataset" or not row.get("source_url") or not row.get("license"):
                raise ValueError("Content review requires public provenance or a synthetic-only Hard Negative phrase")
            binding = row.get("reviewed_source", {"path": row["path"], "sha256": row["sha256"]})
            if digest(resolved(root, binding["path"])) != binding["sha256"]:
                raise ValueError("Content preview's underlying source changed")
            hashes.append(binding["sha256"])
        record = {"state": "accepted", "content_review": "accepted", "accepted_source_sha256": sorted(set(hashes)), "evidence": str(archive.relative_to(root)), "evidence_sha256": digest(archive), "scope": "Source admission based on supplied listening previews and declared metadata; not a full STT sweep"}
        binding = snapshot_json(root, record, "content_admission")
        write_json(root / "review" / "content_admission.json", record)
        return {"state": "saved", "kept": len(kept), "total": len(rows), "admission": binding,
                "note": "Use the immutable admission binding in material manifests; the summary can change after another review."}
    else:
        raise ValueError("Stable review purposes are references, pilot or content")
    return {"state": "saved", "kept": len(kept), "total": len(rows), "policy_approved": approval.get("generation_policy_approved")}


def serve_review(project, queue, kind="pilot", port=8790):
    root, config = load_project(project)
    queue = Path(queue).resolve()
    rows = read_rows(queue)
    if not rows:
        raise ValueError("Empty review queue")
    snapshot = digest(queue)
    purpose = "합성 참조 음색이 자연스러운 한국어이며 서로 충분히 다른지 확인합니다." if kind == "references" else f"대량 생성 전에 '{config['wakeword']}' 단독 호출이 명확하고 불필요한 말이 없는지 확인합니다."
    if kind == "content":
        purpose = f"비호출·증강 자료의 내용 범주와 사용 적합성을 확인합니다. '{config['wakeword']}' 호출이 들리거나 자료가 깨졌으면 폐기합니다."
    payload = {"purpose": purpose, "wakeword": config["wakeword"], "queue_sha256": snapshot, "items": [{"id": r["sample_id"], "text": r.get("text", r.get("category", "비호출 자료")), "audio": f"/audio/{i}"} for i, r in enumerate(rows)]}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                from importlib.resources import files
                body = (files("korean_openwakeword") / "resources" / "review.html").read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)
            elif path == "/queue":
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers(); self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())
            elif path.startswith("/audio/"):
                try:
                    index = int(path.split("/")[-1])
                    if not 0 <= index < len(rows):
                        raise ValueError("Bad index")
                    source = resolved(root, rows[index]["path"])
                    if digest(source) != rows[index]["sha256"]:
                        raise ValueError("Changed audio")
                    self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("Content-Length", str(source.stat().st_size)); self.end_headers()
                    with source.open("rb") as f:
                        __import__("shutil").copyfileobj(f, self.wfile)
                except (ValueError, IndexError):
                    self.send_error(404)
            else:
                self.send_error(404)
        def do_POST(self):
            try:
                if self.headers.get("Origin") not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                    raise ValueError("Review saving requires the local review page's Origin")
                if urlparse(self.path).path != "/save" or digest(queue) != snapshot:
                    raise ValueError("Queue changed or unsupported route")
                if int(self.headers.get("Content-Length", "0")) > 1000000:
                    raise ValueError("Review response too large")
                decisions = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                result = apply_review(project, queue, decisions, kind)
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False).encode())
            except (ValueError, KeyError) as e:
                self.send_response(400); self.end_headers(); self.wfile.write(str(e).encode())
    print(f"Review: http://127.0.0.1:{port}/ · {len(rows)} items · saves directly to {root / 'review'}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
