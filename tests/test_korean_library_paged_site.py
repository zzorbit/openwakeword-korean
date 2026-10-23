"""Bounded static viewers keep every scored clip and recorded display value."""
import copy
import json
from pathlib import Path
import tracemalloc

import pytest

from korean_openwakeword.paged_site import iter_clip_array, write_paged_viewer
from korean_openwakeword.common import digest, write_json
from korean_openwakeword import site


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _boot(output):
    wrapper = (output / "results-data.js").read_text(encoding="utf-8")
    assert wrapper.startswith("window.KOWW_RESULTS=") and wrapper.endswith(";\n")
    return json.loads(wrapper.removeprefix("window.KOWW_RESULTS=").removesuffix(";\n"))


def _clip(number=0, **changes):
    row = {"sample_id": f"소리_{number:06d}", "kind": "positive", "category": "environment",
           "snr_db": 0, "reverb_condition": "none", "original_id": f"original_{number}",
           "original_order": number, "duration_s": 0.0, "path": f"audio/{number:064x}.wav",
           "sha256": f"{number:064x}", "noise_section": 0,
           "noise_eval_type": "unknown_provider_type", "noise_eval_label": "실제 기록된 배경",
           "provenance": "qwen_tts", "noise_source_group_id": "canonical-source",
           "calls": [{"start_s": 0.0, "end_s": 0.31,
                      "original_utterance_id": f"utterance_{number}", "private_metadata": "archive only"}],
           "results": {"head": {"success": False, "false_activations": 0, "max_score": 0.0,
                                "clean_score_drop": None, "latency_s": 0.0,
                                "scores": [0.1] * 50, "frames": [{"score": .1}] * 50}},
           "unneeded_archive_metadata": {"text": "보존할 원본 정보"}}
    row.update(changes)
    return row


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 31, 65536])
def test_streamed_array_preserves_unicode_escapes_nesting_and_order(tmp_path, chunk_size):
    rows = [{"sample_id": "보라야\U0001f3b5", "quote": '"\\\n\t',
             "nested": {"calls": [{"start_s": 0}, None, [], {}]}, "value": -1.25e-20},
            {"sample_id": "끝", "bool": False, "empty": "", "null": None}]
    path = tmp_path / "clips.json"
    path.write_text(" \r\n" + json.dumps(rows, ensure_ascii=False) + "\n\t ", encoding="utf-8")
    assert list(iter_clip_array(path, chunk_size=chunk_size)) == rows
    path.write_text(" \n [ \t ] \r\n", encoding="utf-8")
    assert list(iter_clip_array(path, chunk_size=chunk_size)) == []


@pytest.mark.parametrize("text", [
    "", "{}", "null", '"[]"', "[", "[{", '[{"sample_id":"one"}',
    "[{},]", "[{} {}]", "[{},", "[]{}", "[{}][{}]", "[{}] trailing",
    "\u00a0[]", "[]\u00a0", "[{}\u00a0]",
    '[{"value":1e}]', '[{"value":NaN}]', '[{"value":Infinity}]',
    '[{"value":-Infinity}]', '[{"value":"unterminated}]', '[{"value":"bad\ncontrol"}]',
])
def test_streamed_array_rejects_truncated_malformed_or_trailing_json(tmp_path, text):
    path = tmp_path / "clips.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_clip_array(path, chunk_size=3))


def test_streamed_array_accepts_utf8_bom_without_changing_recorded_text(tmp_path):
    path = tmp_path / "clips.json"
    path.write_text('\ufeff[{"sample_id":"보라야"}]', encoding="utf-8")
    assert list(iter_clip_array(path, chunk_size=1)) == [{"sample_id": "보라야"}]


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_streamed_array_rejects_unbounded_or_invalid_chunk_size(tmp_path, chunk_size):
    path = tmp_path / "clips.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_clip_array(path, chunk_size=chunk_size))


def test_streaming_reads_bounded_chunks_and_releases_consumed_json(tmp_path, monkeypatch):
    path = tmp_path / "clips.json"
    count = 3000
    with path.open("w", encoding="utf-8") as stream:
        stream.write("[")
        for number in range(count):
            if number:
                stream.write(",")
            json.dump({"sample_id": str(number), "discarded": "한" * 1024}, stream, ensure_ascii=False)
        stream.write("]")
    original_open, reads = Path.open, []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            assert 0 < size <= 4096, "The parser must not read the entire manifest"
            reads.append(size)
            return self.stream.read(size)

    def opened(candidate, *args, **kwargs):
        stream = original_open(candidate, *args, **kwargs)
        return BoundedReader(stream) if candidate == path else stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", opened)
        tracemalloc.start()
        try:
            for number, row in enumerate(iter_clip_array(path, chunk_size=4096)):
                assert row["sample_id"] == str(number)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert number + 1 == count
    assert len(reads) > 100
    # The 9 MiB UTF-8 input must not remain in Python memory after consumption.
    assert peak < 1024 * 1024


def test_paged_viewer_projects_exact_display_values_without_mutating_archive_objects(tmp_path):
    rows = [_clip(), _clip(1, results={"head": {"max_score": .975, "success": None,
                                              "latency_s": None, "scores": [.975]}, "empty": {}})]
    report = {"stage": "development", "wakeword": "<보라야>", "models": [{"id": "head"}],
              "archive_metadata": {"keep": [1, 2]}}
    before = copy.deepcopy((report, rows))
    summary = write_paged_viewer(report, iter(rows), tmp_path, page_size=128)
    boot = _boot(tmp_path)
    assert boot == {"report": report, "viewer": {"schema_version": 1,
                    "index_path": "viewer-index.json", "page_size": 128}}
    assert "<보라야>" not in (tmp_path / "results-data.js").read_text(encoding="utf-8")
    detail = _read(tmp_path / "clips/page-000000.json")
    assert detail["start"] == 0
    first, second = detail["clips"]
    for key in ("sample_id", "kind", "category", "snr_db", "reverb_condition", "original_id",
                "original_order", "duration_s", "path", "sha256", "noise_section", "noise_eval_type",
                "noise_eval_label"):
        assert first[key] == rows[0][key]
    assert first["calls"] == [{"start_s": 0.0, "end_s": 0.31}]
    assert first["results"] == {"head": {"success": False, "false_activations": 0, "max_score": 0.0,
                                        "clean_score_drop": None, "latency_s": 0.0}}
    assert second["results"] == {"head": {"max_score": .975, "success": None, "latency_s": None}, "empty": {}}
    assert "unneeded_archive_metadata" not in first
    assert "provenance" not in first and "noise_source_group_id" not in first
    assert summary["audio_files"] == 2 and summary["detail_pages"] == 1
    assert (report, rows) == before


def test_paged_viewer_keeps_every_row_in_bounded_pages_with_stable_index_positions(tmp_path):
    count, produced = 259, []

    def one_pass_rows():
        for number in range(count):
            produced.append(number)
            # Each archive row is much larger than the values used by the UI.
            yield _clip(number, results={"head": {"success": number % 2 == 0,
                            "max_score": number / count, "scores": [0.2] * 2000}})

    summary = write_paged_viewer({"stage": "final", "models": []}, one_pass_rows(), tmp_path, page_size=128)
    index = _read(tmp_path / "viewer-index.json")
    assert index["schema_version"] == 1
    assert index["clips"] == [[f"소리_{i:06d}", "positive", "environment", 0, "none",
                               f"original_{i}", i, 0.0] for i in range(count)]
    pages = sorted((tmp_path / "clips").glob("page-*.json"))
    assert [path.name for path in pages] == ["page-000000.json", "page-000001.json", "page-000002.json"]
    details = [_read(path) for path in pages]
    assert [value["start"] for value in details] == [0, 128, 256]
    assert [len(value["clips"]) for value in details] == [128, 128, 3]
    actual = [row for page in details for row in page["clips"]]
    assert [row["sample_id"] for row in actual] == [f"소리_{i:06d}" for i in range(count)]
    assert [row["results"]["head"]["max_score"] for row in actual] == [i / count for i in range(count)]
    assert produced == list(range(count))
    assert summary["audio_files"] == count and summary["detail_pages"] == 3
    assert summary["boot_bytes"] == (tmp_path / "results-data.js").stat().st_size
    assert summary["index_bytes"] == (tmp_path / "viewer-index.json").stat().st_size
    assert summary["maximum_page_bytes"] == max(path.stat().st_size for path in pages)
    assert summary["maximum_page_bytes"] < 128 * 1024
    assert summary["boot_bytes"] < 1024


def test_sparse_rows_preserve_missing_metadata_and_missing_scores(tmp_path):
    row = {"sample_id": "sparse", "kind": "negative", "results": {"unmeasured": {}}}
    write_paged_viewer({"stage": "development", "models": []}, iter([row]), tmp_path)
    assert _read(tmp_path / "viewer-index.json")["clips"] == [
        ["sparse", "negative", None, None, None, None, None, None]]
    assert _read(tmp_path / "clips/page-000000.json")["clips"] == [row]


def test_scope_and_noise_coverage_use_all_original_evidence_before_projection(tmp_path):
    report = {"stage": "development", "models": [{"negative": {"hours": .2}}, {"negative": {"hours": .2}}],
              "adoption": {"passed": False, "reasons": ["Development FP duration below 20h"]}}
    rows = [_clip(i, original_id="one-recording", calls=[{"start_s": 0, "end_s": 1,
              "original_utterance_id": "first"}, {"start_s": 1.5, "end_s": 2,
              "original_utterance_id": "second"}], snr_db=snr,
              noise_source_group_id="one-canonical-recording", noise_source_id=f"alias-{snr}",
              noise_eval_type="rain") for i, snr in enumerate((20, 15, 10, 5, 0))]
    rows += [_clip(5, original_id="one-recording", category="clean", calls=[], provenance="human"),
             _clip(6, category="music", calls=[], original_id="second-recording",
                   noise_source_group_id=None, noise_source_id="music-source", noise_eval_type=None),
             _clip(7, kind="negative", noise_source_group_id="not-positive-background",
                   noise_eval_type="wind", duration_s=999),
             _clip(8, category="competing_speech", calls=[], original_id="second-recording",
                   noise_source_group_id=None, noise_source_id=None, noise_eval_type=None)]
    before = copy.deepcopy((report, rows))
    write_paged_viewer(report, iter(rows), tmp_path, page_size=2)
    index = _read(tmp_path / "viewer-index.json")
    assert index["scope_text"] == ("개발 평가 (Final 아님) · 호출 원본 4개 · 소음 변형 7개 · "
                                   "비호출 평가 12.00분 · 장시간 기준 미달")
    assert index["noise_coverage_text"] == ("시험에 사용한 배경 음원: 환경음 원본 1개 · 1종 / "
        "음악·전자음 원본 1개 · 세부 종류 미기록 / 겹치는 말소리 원본 수 미기록 · 세부 종류 미기록")
    assert len(index["clips"]) == len(rows)
    assert (report, rows) == before


@pytest.mark.parametrize("report,expected", [
    ({"stage": "final", "final_negative_duration_passed": True,
      "models": [{"negative": {"hours": 123.456}}]},
     "동결 모델 Final 평가 · 호출 원본 0개 · 소음 변형 0개 · 비호출 평가 123.46시간 · 장시간 기준 통과"),
    ({"stage": "development", "models": [], "smoke_test_only": True},
     "개발 평가 (Final 아님) · 호출 원본 0개 · 소음 변형 0개 · 비호출 평가 시간 미측정 · 장시간 기준 미확인 · 기능 시험용 결과"),
])
def test_empty_viewer_records_unmeasured_scope_and_report_duration_truth(tmp_path, report, expected):
    summary = write_paged_viewer(report, iter(()), tmp_path)
    index = _read(tmp_path / "viewer-index.json")
    assert index["scope_text"] == expected
    assert index["noise_coverage_text"] == "시험에 사용한 배경 음원: 미측정"
    assert index["clips"] == []
    assert summary["audio_files"] == 0 and summary["detail_pages"] == 0


@pytest.mark.parametrize("large_archive", [False, True])
def test_export_uses_pages_for_large_count_or_large_archive_without_changing_scored_files(
        tmp_path, monkeypatch, large_archive):
    evaluation, output = tmp_path / "evaluation", tmp_path / "site"
    evaluation.mkdir()
    source = evaluation / "scored.wav"
    source.write_bytes(b"RIFF exact previously-scored fixture bytes")
    checksum = digest(source)
    count = 1 if large_archive else 1001
    rows = [_clip(i, path=str(source), sha256=checksum,
                  results={"head": {"success": False, "max_score": 0.0}}) for i in range(count)]
    if large_archive:
        rows[0]["unneeded_archive_metadata"] = "x" * (4 * 1024 * 1024)
    report = {"stage": "development", "wakeword": "보라야", "models": [{"id": "head"}]}
    write_json(evaluation / "clips.json", rows)
    assert ((evaluation / "clips.json").stat().st_size > 4 * 1024 * 1024) is large_archive
    write_json(evaluation / "report.json", report)
    write_json(evaluation / "receipt.json", {"clips_sha256": digest(evaluation / "clips.json"),
               "report_sha256": digest(evaluation / "report.json")})
    protected = {path: digest(path) for path in evaluation.iterdir()}
    original_read_json = site.read_json

    def bounded_read_json(path):
        if large_archive and Path(path).name == "clips.json":
            pytest.fail("Large clip archives must stream rather than load through read_json")
        return original_read_json(path)

    monkeypatch.setattr(site, "read_json", bounded_read_json)
    result = site.export_site(evaluation, output)
    assert result["audio_files"] == count
    assert {path: digest(path) for path in protected} == protected
    assert (output / "audio" / (checksum + ".wav")).read_bytes() == source.read_bytes()
    assert len(list((output / "audio").glob("*.wav"))) == 1
    boot = _boot(output)
    assert boot["report"] == report and "clips" not in boot
    index = _read(output / boot["viewer"]["index_path"])
    assert len(index["clips"]) == count
    exported = [row for page in sorted((output / "clips").glob("page-*.json"))
                for row in _read(page)["clips"]]
    assert len(exported) == count
    assert [row["sample_id"] for row in exported] == [row["sample_id"] for row in rows]
    assert [row["results"]["head"]["max_score"] for row in exported] == [0.0] * count
    assert all(row["path"] == "audio/" + checksum + ".wav" for row in exported)
    assert (output / "results-data.js").stat().st_size < 4096
