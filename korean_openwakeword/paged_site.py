"""Bounded display data for large frozen evaluations; never change scoring."""
import json
import math
from pathlib import Path

from .common import write_json


PAGE_SIZE = 128
INDEX_FIELDS = ("sample_id", "kind", "category", "snr_db", "reverb_condition",
                "original_id", "original_order", "duration_s")
DISPLAY_FIELDS = INDEX_FIELDS + ("path", "sha256", "noise_section",
    "noise_eval_type", "noise_eval_label")
RESULT_FIELDS = ("success", "false_activations", "max_score", "clean_score_drop", "latency_s")


def iter_clip_array(path, chunk_size=65536):
    """Incrementally decode an ordinary JSON array, including UTF-8/BOM files.

    Only one object and a bounded input chunk are retained. Invalid separators,
    incomplete objects and trailing data fail instead of publishing a subset.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("JSON stream chunk size must be a positive integer")
    def invalid_constant(value):
        raise ValueError("Non-finite number in evaluation clips: " + value)
    decoder = json.JSONDecoder(parse_constant=invalid_constant)
    with Path(path).open(encoding="utf-8-sig") as stream:
        buffer, position, ended = "", 0, False

        def refill():
            nonlocal buffer, position, ended
            buffer = buffer[position:]
            position = 0
            value = stream.read(chunk_size)
            buffer += value
            ended = not value

        def token():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n":
                    position += 1
                if position < len(buffer):
                    return buffer[position]
                if ended:
                    return ""
                refill()

        if token() != "[":
            raise ValueError("Evaluation clips must be a JSON array")
        position += 1
        first = True
        while True:
            current = token()
            if current == "]":
                position += 1
                break
            if not first:
                if current != ",":
                    raise ValueError("Missing separator in evaluation clips")
                position += 1
                if token() in ("", "]"):
                    raise ValueError("Incomplete evaluation clip array")
            elif not current:
                raise ValueError("Incomplete evaluation clip array")
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as error:
                    if ended:
                        raise ValueError("Invalid evaluation clip JSON") from error
                    if len(buffer) - position > 16 * 1024 ** 2:
                        raise ValueError("Evaluation clip exceeds the bounded JSON record size") from error
                    refill()
            if not isinstance(row, dict):
                raise ValueError("Each evaluation clip must be an object")
            position = end
            first = False
            yield row
        if token():
            raise ValueError("Trailing data after evaluation clips")


def _display_clip(row):
    result = {key: row[key] for key in DISPLAY_FIELDS if key in row}
    if "calls" in row:
        result["calls"] = ([{key: call[key] for key in ("start_s", "end_s") if key in call}
                           for call in row["calls"]] if row["calls"] is not None else None)
    result["results"] = {model: {key: value[key] for key in RESULT_FIELDS if key in value}
                         for model, value in row.get("results", {}).items()}
    return result


class _Scope:
    def __init__(self):
        self.originals, self.positive, self.variants = set(), 0, 0
        self.synthetic = True
        self.noise = {name: {"count": 0, "sources": set(), "types": set()}
                      for name in ("environment", "music", "competing_speech")}

    def add(self, row):
        if row.get("kind") != "positive":
            return
        self.positive += 1
        self.synthetic &= row.get("provenance") == "qwen_tts"
        self.variants += row.get("category") != "clean"
        for call in row.get("calls") or [{}]:
            self.originals.add(str(call.get("original_utterance_id") or row.get("original_id") or row.get("sample_id")))
        group = self.noise.get(row.get("category"))
        if group is not None:
            group["count"] += 1
            source = row.get("noise_source_group_id") or row.get("noise_source_id")
            if source:
                group["sources"].add(source)
            if row.get("noise_eval_type"):
                group["types"].add(row["noise_eval_type"])

    def texts(self, report):
        hours = next((model["negative"]["hours"] for model in report.get("models", [])
                      if type((model.get("negative") or {}).get("hours")) in (int, float)
                      and math.isfinite(model["negative"]["hours"])), None)
        adoption = report.get("adoption") or {}
        duration_gate = "장시간 기준 미확인"
        if report.get("stage") == "final" and type(report.get("final_negative_duration_passed")) is bool:
            duration_gate = "장시간 기준 통과" if report["final_negative_duration_passed"] else "장시간 기준 미달"
        elif any(isinstance(reason, str) and reason.startswith("Development FP duration below ")
                 for reason in adoption.get("reasons", [])):
            duration_gate = "장시간 기준 미달"
        elif hours is not None and adoption.get("passed") is True:
            duration_gate = "장시간 기준 통과"
        time = "시간 미측정" if hours is None else f"{hours * 60:.2f}분" if hours < 1 else f"{hours:.2f}시간"
        parts = ["동결 모델 Final 평가" if report.get("stage") == "final" else "개발 평가 (Final 아님)",
            f"{'합성 ' if self.positive and self.synthetic else ''}호출 원본 {len(self.originals):,}개",
            f"소음 변형 {self.variants:,}개", f"비호출 평가 {time}", duration_gate]
        if report.get("smoke_test_only"):
            parts.append("기능 시험용 결과")
        labels = {"environment": "환경음", "music": "음악·전자음", "competing_speech": "겹치는 말소리"}
        coverage = []
        for category, values in self.noise.items():
            if values["count"]:
                sources = f"원본 {len(values['sources']):,}개" if values["sources"] else "원본 수 미기록"
                types = f"{len(values['types']):,}종" if values["types"] else "세부 종류 미기록"
                coverage.append(f"{labels[category]} {sources} · {types}")
        return " · ".join(parts), "시험에 사용한 배경 음원: " + (" / ".join(coverage) or "미측정")


def write_paged_viewer(report, clips, output, page_size=PAGE_SIZE):
    """Export every item into a compact index and bounded detail pages.

    Clip order, recorded scalar scores and audio hashes are preserved. Full
    provenance and call diagnostics stay in the immutable evaluation archive.
    """
    if type(page_size) is not int or not 1 <= page_size <= PAGE_SIZE:
        raise ValueError("Viewer page size must be an integer from 1 to 128")
    output = Path(output)
    (output / "clips").mkdir(parents=True, exist_ok=True)
    index, page, scope = [], [], _Scope()
    pages, maximum = 0, 0

    def publish_page():
        nonlocal pages, maximum, page
        path = output / "clips" / f"page-{pages:06d}.json"
        if path.exists():
            raise ValueError("Viewer pages are immutable")
        write_json(path, {"start": len(index) - len(page), "clips": page})
        maximum = max(maximum, path.stat().st_size)
        pages += 1
        page = []

    for row in clips:
        scope.add(row)
        index.append([row.get(key) for key in INDEX_FIELDS])
        page.append(_display_clip(row))
        if len(page) == page_size:
            publish_page()
    if page:
        publish_page()
    scope_text, noise_text = scope.texts(report)
    index_path = output / "viewer-index.json"
    # Compact tuples avoid retaining 112k diagnostic objects in the browser.
    index_path.write_text(json.dumps({"schema_version": 1, "scope_text": scope_text,
        "noise_coverage_text": noise_text, "clips": index}, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False), encoding="utf-8")
    boot = output / "results-data.js"
    boot.write_text("window.KOWW_RESULTS=" + json.dumps({"report": report, "viewer": {
        "schema_version": 1, "index_path": "viewer-index.json", "page_size": page_size}},
        ensure_ascii=False, allow_nan=False).replace("<", "\\u003c") + ";\n", encoding="utf-8")
    return {"audio_files": len(index), "detail_pages": pages, "index_bytes": index_path.stat().st_size,
            "boot_bytes": boot.stat().st_size, "maximum_page_bytes": maximum}
