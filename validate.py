"""
validate.py — Standalone validation script for the text classification pipeline.

Runs all artifact and consistency checks, prints PASS/FAIL per item,
and exits with code 0 (all pass) or 1 (any fail).
"""

import json
import pathlib
import subprocess
import sys

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Check runner
# ─────────────────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Print a single PASS/FAIL line, accumulate the global counters, return ok."""
    global _passed, _failed
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if detail:
        print(f"         >> {detail}")
    if ok:
        _passed += 1
    else:
        _failed += 1
    return ok


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> tuple[dict | None, str | None]:
    """Return (data, None) on success, or (None, error_msg) on any failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh), None
    except FileNotFoundError:
        return None, f"{path} not found"
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON at line {exc.lineno}: {exc.msg}"
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Required artifact files exist
# ─────────────────────────────────────────────────────────────────────────────

def check_files_exist() -> None:
    section("1. Required artifact files exist")

    # Resolve winner name so we check the right .joblib file
    winner_name = None
    sel_data, _ = _load_json("model_selection_report.json")
    if sel_data:
        winner_name = sel_data.get("winner")

    required = [
        "data_validation_report.json",
        "preprocessing_preview.json",
        "split_report.json",
        "metrics.json",
        "model_selection_report.json",
        "error_analysis.json",
        "test_predictions.csv",
        "vectorizer.joblib",
        f"{winner_name}.joblib" if winner_name else "???.joblib  (winner unknown)",
        "run_manifest.json",
        "safeguards_report.json",
    ]

    for artifact in required:
        exists = pathlib.Path(artifact).exists()
        check(artifact, exists, "file not found" if not exists else "")


# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — All JSON artifacts are valid JSON
# ─────────────────────────────────────────────────────────────────────────────

_JSON_ARTIFACTS = [
    "data_validation_report.json",
    "preprocessing_preview.json",
    "split_report.json",
    "metrics.json",
    "model_selection_report.json",
    "error_analysis.json",
    "run_manifest.json",
    "safeguards_report.json",
]


def check_json_valid() -> None:
    section("2. All JSON artifacts are valid JSON")

    for artifact in _JSON_ARTIFACTS:
        if not pathlib.Path(artifact).exists():
            check(artifact, False, "file not found - skipping JSON parse")
            continue
        _, err = _load_json(artifact)
        check(artifact, err is None, err or "")


# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — CSV files have required columns
# ─────────────────────────────────────────────────────────────────────────────

def check_csv_columns() -> None:
    section("3. CSV files have required columns")

    specs = [
        ("train.csv", {"id", "text", "label"}),
        ("test.csv",  {"id", "text"}),
    ]

    for path, required in specs:
        label = f"{path} has columns: {', '.join(sorted(required))}"
        if not pathlib.Path(path).exists():
            check(label, False, "file not found")
            continue
        try:
            actual  = set(pd.read_csv(path, nrows=0).columns)
            missing = required - actual
            check(label, not missing,
                  f"missing columns: {sorted(missing)}" if missing else "")
        except Exception as exc:
            check(label, False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — metrics.json contains at least 3 models
# ─────────────────────────────────────────────────────────────────────────────

def check_metrics_model_count() -> None:
    section("4. metrics.json contains at least 3 models")

    data, err = _load_json("metrics.json")
    if err:
        check("metrics.json contains >= 3 models", False, err)
        return

    models   = data.get("models", {})
    n        = len(models)
    check("metrics.json contains >= 3 models", n >= 3,
          f"found {n} model(s): {list(models.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — Winner in model_selection_report.json matches a key in metrics.json
# ─────────────────────────────────────────────────────────────────────────────

def check_winner_in_metrics() -> None:
    section("5. Winner in model_selection_report.json matches a key in metrics.json")

    sel_data, sel_err = _load_json("model_selection_report.json")
    met_data, met_err = _load_json("metrics.json")

    if sel_err or met_err:
        check("Winner exists in metrics.json models", False, sel_err or met_err)
        return

    winner = sel_data.get("winner")
    keys   = list(met_data.get("models", {}).keys())

    check(
        f"Winner '{winner}' is a key in metrics.json",
        winner in keys,
        f"available keys: {keys}" if winner not in keys else "",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — test_predictions.csv has a row for every id in test.csv
# ─────────────────────────────────────────────────────────────────────────────

def check_prediction_coverage() -> None:
    section("6. test_predictions.csv has a row for every id in test.csv")

    for path in ("test.csv", "test_predictions.csv"):
        if not pathlib.Path(path).exists():
            check("Prediction coverage", False, f"{path} not found")
            return

    try:
        test_ids = set(pd.read_csv("test.csv")["id"])
        pred_ids = set(pd.read_csv("test_predictions.csv")["id"])
        missing  = test_ids - pred_ids
        extra    = pred_ids - test_ids

        detail = ""
        if missing:
            detail = f"missing predictions for IDs: {sorted(missing)}"
        elif extra:
            detail = f"extra IDs in predictions not in test.csv: {sorted(extra)}"

        check(
            f"All {len(test_ids)} test ID(s) have a prediction",
            not missing,
            detail,
        )
    except Exception as exc:
        check("Prediction coverage", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Check 7 — predict.py can load artifacts and run inference
# ─────────────────────────────────────────────────────────────────────────────

def check_predict_inference() -> None:
    section("7. predict.py can load artifacts and run inference")

    if not pathlib.Path("predict.py").exists():
        check("predict.py --text 'test input' runs without error",
              False, "predict.py not found")
        return

    try:
        result = subprocess.run(
            [sys.executable, "predict.py", "--text", "test input"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        ok = result.returncode == 0 and "Predicted label:" in result.stdout

        detail = ""
        if not ok:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            detail = (stderr or stdout)[:200] if (stderr or stdout) else \
                     f"'Predicted label:' not found in output: {result.stdout!r}"

        check("predict.py --text 'test input' runs without error", ok, detail)

    except subprocess.TimeoutExpired:
        check("predict.py --text 'test input' runs without error",
              False, "timed out after 30 seconds")
    except Exception as exc:
        check("predict.py --text 'test input' runs without error", False, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 52)
    print("  Pipeline Validation")
    print("=" * 52)

    check_files_exist()
    check_json_valid()
    check_csv_columns()
    check_metrics_model_count()
    check_winner_in_metrics()
    check_prediction_coverage()
    check_predict_inference()

    print("\n" + "=" * 52)
    print(f"  PASSED: {_passed} checks")
    print(f"  FAILED: {_failed} checks")
    print("=" * 52)

    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
