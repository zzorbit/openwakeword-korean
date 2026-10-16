"""CPU-only display scope; scoring metadata and audio paths stay unchanged."""
import json
import importlib.util
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from korean_openwakeword.common import digest, read_json, write_json
from korean_openwakeword.site import export_site


SCOPE_SCRIPT = Path(__file__).resolve().parents[1] / "korean_openwakeword/resources/results_scope.js"


def _scope(report, clips):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the browser's pure scope formatter")
    code = "const {evaluationScopeText}=require(" + json.dumps(str(SCOPE_SCRIPT)) + ");" \
        "let s='';process.stdin.setEncoding('utf8');process.stdin.on('data',x=>s+=x);" \
        "process.stdin.on('end',()=>{const d=JSON.parse(s);process.stdout.write(evaluationScopeText(d.report,d.clips));});"
    return subprocess.run([node, "-e", code], input=json.dumps({"report": report, "clips": clips}, ensure_ascii=False),
                          text=True, encoding="utf-8", capture_output=True, check=True).stdout


def test_scope_counts_originals_once_and_uses_eligible_negative_hours_not_model_sum():
    report = {"stage": "development", "sessions": 5, "models": [
        {"negative": {"hours": .2}}, {"negative": {"hours": .2}}, {"negative": {"hours": .2}}],
        "adoption": {"passed": False, "reasons": ["Development FP duration below 20h"]}}
    clips = [
        {"sample_id": "clean_a", "original_id": "a", "kind": "positive", "category": "clean", "provenance": "qwen_tts"},
        {"sample_id": "noise_a", "original_id": "a", "kind": "positive", "category": "music", "provenance": "qwen_tts"},
        {"sample_id": "noise2_a", "original_id": "a", "kind": "positive", "category": "environment", "provenance": "qwen_tts"},
        {"sample_id": "clean_b", "original_id": "b", "kind": "positive", "category": "clean", "provenance": "qwen_tts"},
        {"sample_id": "negative", "kind": "negative", "category": "environment", "duration_s": 999},
    ]
    before = json.dumps({"report": report, "clips": clips}, sort_keys=True)
    assert _scope(report, clips) == "개발 평가 (Final 아님) · 합성 호출 원본 2개 · 소음 변형 2개 · 비호출 평가 12.00분 · 장시간 기준 미달"
    assert json.dumps({"report": report, "clips": clips}, sort_keys=True) == before


@pytest.mark.parametrize("passed,label", [(True, "장시간 기준 통과"), (False, "장시간 기준 미달")])
def test_final_duration_status_comes_from_actual_final_report_boolean(passed, label):
    text = _scope({"stage": "final", "final_negative_duration_passed": passed,
                   "models": [{"negative": {"hours": 123.456}}]}, [])
    assert text == f"동결 모델 Final 평가 · 호출 원본 0개 · 소음 변형 0개 · 비호출 평가 123.46시간 · {label}"
    assert "Final 아님" not in text


def test_missing_scope_evidence_is_unmeasured_not_assumed_passing():
    text = _scope({"stage": "development", "models": [], "smoke_test_only": True}, [])
    assert "비호출 평가 시간 미측정" in text
    assert "장시간 기준 미확인" in text
    assert text.endswith("기능 시험용 결과")


def test_multi_call_original_ids_and_human_provenance_remain_truthful():
    clips = [{"kind": "positive", "category": "clean", "sample_id": "recording", "provenance": "human",
              "calls": [{"original_utterance_id": "one"}, {"original_utterance_id": "two"}]}]
    text = _scope({"stage": "development", "models": [{"negative": {"hours": 25}}],
                   "adoption": {"passed": True}}, clips)
    assert "호출 원본 2개" in text and "합성" not in text
    assert "장시간 기준 통과" in text


def _coverage(clips):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the browser's pure coverage formatter")
    code = "const {evaluationNoiseCoverageText}=require(" + json.dumps(str(SCOPE_SCRIPT)) + ");" \
        "let s='';process.stdin.setEncoding('utf8');process.stdin.on('data',x=>s+=x);" \
        "process.stdin.on('end',()=>process.stdout.write(evaluationNoiseCoverageText(JSON.parse(s))));"
    return subprocess.run([node, "-e", code], input=json.dumps(clips), text=True,
                          encoding="utf-8", capture_output=True, check=True).stdout


def test_noise_coverage_does_not_count_snr_variants_or_negative_sessions_as_new_sources():
    clips = [{"kind": "positive", "category": "environment", "snr_db": snr,
              "noise_source_group_id": "one-canonical-recording", "noise_source_id": "alias-" + str(snr),
              "noise_eval_type": "rain"} for snr in (20, 15, 10, 5, 0)]
    clips += [{"kind": "positive", "category": "environment", "noise_source_group_id": "second-recording",
               "noise_eval_type": "wind"},
              {"kind": "negative", "category": "environment", "noise_source_group_id": "not-a-positive-background",
               "noise_eval_type": "traffic"}]
    original = json.dumps(clips, sort_keys=True)
    assert _coverage(clips) == "시험에 사용한 배경 음원: 환경음 원본 2개 · 2종"
    assert json.dumps(clips, sort_keys=True) == original


def test_legacy_noise_coverage_uses_recorded_source_ids_without_inventing_subtypes():
    assert _coverage([{"kind": "positive", "category": "music", "noise_source_id": "original"}]) == \
        "시험에 사용한 배경 음원: 음악·전자음 원본 1개 · 세부 종류 미기록"
    assert _coverage([{"kind": "positive", "category": "competing_speech"}]) == \
        "시험에 사용한 배경 음원: 겹치는 한국어 음성 원본 수 미기록 · 세부 종류 미기록"
    assert _coverage([]) == "시험에 사용한 배경 음원: 미측정"


def test_background_detail_uses_actual_recorded_type_without_inventing_a_speech_style():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the browser's pure type formatter")
    clips = [{"noise_eval_type": value} for value in (
        "korean_read_speech", "korean_market_conversation", "korean_transit_announcement", "unknown_provider_type")]
    clips.append({"category": "competing_speech"})
    code = "const {evaluationBackgroundTypeText}=require(" + json.dumps(str(SCOPE_SCRIPT)) + ");" \
        "let s='';process.stdin.setEncoding('utf8');process.stdin.on('data',x=>s+=x);" \
        "process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(evaluationBackgroundTypeText))));"
    result = subprocess.run([node, "-e", code], input=json.dumps(clips), text=True,
                            encoding="utf-8", capture_output=True, check=True)
    assert json.loads(result.stdout) == ["한국어 낭독", "한국어 시장 대화", "한국어 교통 안내방송", "unknown_provider_type", ""]


def test_export_adds_scope_asset_without_changing_scores_or_scored_audio(tmp_path):
    evaluation, site = tmp_path / "evaluation", tmp_path / "site"
    evaluation.mkdir()
    source = evaluation / "fixture.wav"
    sf.write(source, np.zeros(1600), 16000, subtype="PCM_16")
    clips = [{"sample_id": "clean", "kind": "positive", "category": "clean", "original_id": "original",
              "path": str(source), "sha256": digest(source), "results": {"single": {"success": True, "max_score": .9}}}]
    report = {"stage": "development", "models": [{"id": "single", "negative": {"hours": 0}}]}
    write_json(evaluation / "clips.json", clips)
    write_json(evaluation / "report.json", report)
    write_json(evaluation / "receipt.json", {"clips_sha256": digest(evaluation / "clips.json"),
               "report_sha256": digest(evaluation / "report.json")})
    protected = {path: digest(path) for path in (source, evaluation / "report.json", evaluation / "clips.json", evaluation / "receipt.json")}
    export_site(evaluation, site)
    assert {path: digest(path) for path in protected} == protected
    assert (site / "results_scope.js").read_bytes() == SCOPE_SCRIPT.read_bytes()
    html = (site / "index.html").read_text(encoding="utf-8")
    assert html.index('src="results_scope.js"') < html.index('src="results.js"')
    payload = (site / "results-data.js").read_text(encoding="utf-8")
    exported = json.loads(payload.removeprefix("window.KOWW_RESULTS=").removesuffix(";\n"))
    assert exported["report"] == report
    assert exported["clips"][0]["results"] == clips[0]["results"]
    assert digest(site / exported["clips"][0]["path"]) == clips[0]["sha256"]
    assert read_json(site / "provenance.json")["same_scored_WAVs"] is True


def _post_export_fixture(tmp_path):
    evaluation, site = tmp_path / "evaluation", tmp_path / "site"
    evaluation.mkdir()
    source = evaluation / "fixture.wav"
    sf.write(source, np.zeros(1600), 16000, subtype="PCM_16")
    clips = [{"sample_id": "clean", "kind": "positive", "category": "clean", "original_id": "original",
              "path": str(source), "sha256": digest(source), "results": {"single": {"success": True}}}]
    report = {"stage": "development", "models": [{"id": "single", "negative": {"hours": 0}}]}
    write_json(evaluation / "clips.json", clips)
    write_json(evaluation / "report.json", report)
    write_json(evaluation / "receipt.json", {"clips_sha256": digest(evaluation / "clips.json"),
               "report_sha256": digest(evaluation / "report.json")})
    export_site(evaluation, site)
    (site / "index.html").write_text("<html>Old display</html>", encoding="utf-8")
    (site / "results.js").write_text("// old display script\n", encoding="utf-8")
    (site / "results_scope.js").unlink()
    helper_path = Path(__file__).resolve().parents[1] / "reports/bora_independent_20261013/patch_result_scope_ui.py"
    spec = importlib.util.spec_from_file_location("scope_patch_helper_test", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, evaluation, site, source


@pytest.mark.legacy_parity
def test_post_export_scope_helper_preserves_metadata_scores_and_audio(tmp_path):
    helper, evaluation, site, source = _post_export_fixture(tmp_path)
    protected = [source, evaluation / "report.json", evaluation / "clips.json", evaluation / "receipt.json",
                 site / "results-data.js", site / "provenance.json"]
    before = {path: digest(path) for path in protected}
    result = helper.patch_scope_ui(evaluation / "report.json", site, SCOPE_SCRIPT.parent)
    assert result["state"] == "completed"
    assert {path: digest(path) for path in protected} == before
    evidence = read_json(result["receipt"])
    assert evidence["protected_metadata_sha256_before"] == evidence["protected_metadata_sha256_after"]
    assert evidence["audio_files_read_or_modified"] == 0
    assert evidence["new_inference_or_training"] is False
    assert (Path(evidence["old_ui_preserved_at"]) / "index.html").read_text(encoding="utf-8") == "<html>Old display</html>"
    assert helper.patch_scope_ui(evaluation / "report.json", site, SCOPE_SCRIPT.parent)["state"] == "already_current"


@pytest.mark.legacy_parity
def test_post_export_scope_helper_refuses_site_scores_changed_since_export(tmp_path):
    helper, evaluation, site, _ = _post_export_fixture(tmp_path)
    data = site / "results-data.js"
    wrapper = data.read_text(encoding="utf-8")
    payload = json.loads(wrapper.removeprefix("window.KOWW_RESULTS=").removesuffix(";\n"))
    payload["clips"][0]["results"]["single"]["success"] = False
    data.write_text("window.KOWW_RESULTS=" + json.dumps(payload) + ";\n", encoding="utf-8")
    before = digest(site / "index.html")
    with pytest.raises(ValueError, match="scores or scored clip metadata changed"):
        helper.patch_scope_ui(evaluation / "report.json", site, SCOPE_SCRIPT.parent)
    assert digest(site / "index.html") == before
