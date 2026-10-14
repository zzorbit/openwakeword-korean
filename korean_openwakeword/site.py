"""Export completed results and their exact scored WAVs as a static site."""
from importlib.resources import files
from pathlib import Path
import json
import shutil

from .common import digest, read_json, write_json


def export_site(evaluation, output):
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    receipt = read_json(evaluation / "receipt.json")
    for name in ("report.json", "clips.json"):
        if digest(evaluation / name) != receipt[name.split(".")[0] + "_sha256"]:
            raise ValueError("Completed evaluation report/clip binding changed")
    report, clips = read_json(evaluation / "report.json"), read_json(evaluation / "clips.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Site exports are immutable")
    (output / "audio").mkdir(parents=True, exist_ok=True)
    for row in clips:
        source = Path(row["path"])
        if digest(source) != row["sha256"]:
            raise ValueError("Scored WAV changed; cannot attach old scores to new audio")
        name = row["sha256"] + ".wav"
        dest = output / "audio" / name
        if not dest.exists():
            shutil.copyfile(source, dest)
        row["path"] = "audio/" + name
        if digest(dest) != row["sha256"]:
            raise ValueError("Copied WAV changed")
    root = files("korean_openwakeword") / "resources"
    for name in ("results.html", "results_scope.js", "results.js", "results.css"):
        (output / ("index.html" if name == "results.html" else name)).write_bytes((root / name).read_bytes())
    text = json.dumps({"report": report, "clips": clips}, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (output / "results-data.js").write_text("window.KOWW_RESULTS=" + text + ";\n", encoding="utf-8")
    write_json(output / "provenance.json", {"evaluation_receipt_sha256": digest(evaluation / "receipt.json"),
        "audio": {r["sample_id"]: r["sha256"] for r in clips}, "models": report["models"],
        "stage": report["stage"], "same_scored_WAVs": True, "new_inference_or_training": False})
    return {"site": str(output / "index.html"), "audio_files": len(clips), "stage": report["stage"]}
