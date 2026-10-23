"""Run every tests/check_*.cjs browser contract and retain individual logs."""
import argparse
import json
from pathlib import Path
import subprocess


def run_browser_checks(repository, output, *, node="node", timeout=120):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    checks = sorted(path for path in (repository / "tests").glob("check_*.cjs") if path.is_file())
    if not checks:
        raise ValueError("No browser checks found in tests/check_*.cjs")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for check in checks:
        try:
            completed = subprocess.run([node, str(check)], cwd=repository, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
            returncode = completed.returncode
            log = completed.stdout + completed.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            returncode, log = None, str(error) + "\n"
        log_path = output / (check.stem + ".log")
        log_path.write_text(log, encoding="utf-8")
        result = {"name": check.name, "passed": returncode == 0,
                  "returncode": returncode, "log": log_path.name}
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {check.name}", flush=True)
    report = {"passed": all(result["passed"] for result in results), "checks": results}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(".test-tmp/browser-checks"))
    args = parser.parse_args()
    return 0 if run_browser_checks(args.repository, args.output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
