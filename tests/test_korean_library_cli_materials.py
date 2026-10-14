"""The documented relative-path commands work from the checkout directory."""
from pathlib import Path

import numpy as np
import soundfile as sf

from korean_openwakeword.cli import main, parser
from korean_openwakeword.common import read_json, read_rows
from korean_openwakeword.project import create_project, load_project
from korean_openwakeword.review import apply_pool_review


def test_documented_prepare_review_admit_path_chain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = create_project(Path("projects/bora"), "보라야", "bora")
    source = Path("shared/zeroth/train_data_01")
    for identifier in ("001", "002", "003", "004"):
        folder = source / identifier
        folder.mkdir(parents=True)
        uid = identifier + "_0001"
        values = .1 * np.sin(np.arange(16000 * 4) * (200 + int(identifier)) * 2 * np.pi / 16000)
        sf.write(folder / (uid + ".wav"), values, 16000, subtype="PCM_16")
        (folder / (identifier + ".trans.txt")).write_text(uid + " 일반적인 한국어 문장\n", encoding="utf-8")
    main(["index-data", "--dataset", "zeroth", "--input", str(source),
          "--output", "shared/zeroth_index.jsonl", "--source-url", "https://example.org/fixture",
          "--license", "test-fixture-only"])
    main(["prepare-materials", "--project", str(project), "--input", str(source),
          "--source-index", "shared/zeroth_index.jsonl", "--output", "projects/bora/prepared/korean_v1"])
    prepared = Path("projects/bora/prepared/korean_v1")
    receipt = prepared / "preparation_receipt.json"
    assert receipt.is_file()
    assert not (Path("projects/bora/projects")).exists()
    queue, sources = prepared / "representatives.jsonl", prepared / "train_sources.jsonl"
    preview_rows = read_rows(queue)
    decisions = {r["sample_id"]: {"decision": "keep", "listened": True}
                 for r in preview_rows}
    # Simulated fixture answers exercise the protocol, not real human approval.
    apply_pool_review(project, queue.resolve(), sources.resolve(), decisions, True)
    main(["admit-materials", "--project", str(project), "--preparation", str(receipt),
          "--pools", "korean_negative", "--connect"])
    root, config = load_project(project)
    windows = read_rows(root / config["manifests"]["korean_negative"])
    assert windows and all(r["split"] == "train" and r["window_policy"] == "fixed_2s_no_audio_copy" for r in windows)
    assert all(r["end_s"] - r["start_s"] <= 2 for r in windows)
    first = (root / config["manifests"]["korean_negative"]).read_bytes()
    main(["admit-materials", "--project", str(project), "--preparation", str(receipt),
          "--pools", "korean_negative", "--connect"])
    assert first == (root / config["manifests"]["korean_negative"]).read_bytes()
    assert read_json(receipt)["Final_accessed"] is False


def test_candidate_options_and_relative_selection_dispatch(monkeypatch, tmp_path):
    from korean_openwakeword import selection
    calls = []
    monkeypatch.setattr(selection, "select_candidate", lambda *args: calls.append(args) or {"candidate": args[-1]})
    monkeypatch.chdir(tmp_path)
    main(["select-candidate", "--project", "projects/bora/project.json", "--bundle", "models/candidate",
          "--evaluation", "reports/dev", "--output", "models/selected", "--candidate", "member3_seed20260930"])
    assert calls[0][-1] == "member3_seed20260930"
    assert parser().parse_args(["evaluate", "--project", "p.json", "--bundle", "b", "--output", "e", "--candidate", "single"]).candidate == "single"
    assert parser().parse_args(["tune", "--project", "p.json", "--bundle", "b", "--evaluation", "e", "--output", "t",
                               "--thresholds", ".9", "--candidate", "single"]).candidate == "single"
