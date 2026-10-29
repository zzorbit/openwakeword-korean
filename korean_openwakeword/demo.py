"""Re-score the small packaged Development sample and export its exact audio UI."""
from pathlib import Path
import json
from .assets import export_sample
from .common import read_json, resolved
from .project import create_project
from .evaluate import evaluate
from .site import export_site


def run_demo(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo destination must be empty")
    project = create_project(output / "project", "춘식아", "chunsik_demo")
    bundle = output / "sample_model"
    export_sample(bundle)
    rows = read_json(bundle / "audio_manifest.json")
    for row in rows:
        row["path"] = str(resolved(bundle, row["path"]))
        row.pop("existing_results", None)
    manifest = project.parent / "data" / "eval" / "development.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    evaluate(project, bundle, output / "evaluation", backend="cpu")
    return export_site(output / "evaluation", output / "site")
