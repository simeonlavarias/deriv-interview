"""
predict.py — Standalone inference CLI for the text classification pipeline.

Usage:
    python predict.py --text "The app is easy to use"

Output:
    Predicted label: positive
    Confidence score: 0.823        # predict_proba models
    Decision score:   1.042        # decision_function-only models (e.g. LinearSVC)
"""

import argparse
import json
import re
import sys

import joblib


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — identical rules to pipeline.py stage 3
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    """Lowercase → strip surrounding whitespace → collapse internal whitespace."""
    text = str(text).lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_artifacts():
    """
    Read the winner name from model_selection_report.json, then load
    the fitted vectorizer and winning model from their .joblib files.
    """
    try:
        with open("model_selection_report.json", "r", encoding="utf-8") as fh:
            winner_name = json.load(fh)["winner"]
    except FileNotFoundError:
        sys.exit(
            "ERROR: model_selection_report.json not found. "
            "Run pipeline.py first to generate all required artifacts."
        )

    try:
        vectorizer = joblib.load("vectorizer.joblib")
    except FileNotFoundError:
        sys.exit("ERROR: vectorizer.joblib not found. Run pipeline.py first.")

    model_path = f"{winner_name}.joblib"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        sys.exit(f"ERROR: {model_path} not found. Run pipeline.py first.")

    return winner_name, vectorizer, model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer(text: str, vectorizer, model) -> tuple[str, float | None, str | None]:
    """
    Preprocess *text*, vectorize it, and run the model.

    Returns
    -------
    label      : predicted class string
    score      : numeric confidence or decision score (None if unavailable)
    score_label: human-readable name for the score type
    """
    cleaned = _preprocess(text)
    X       = vectorizer.transform([cleaned])

    label     = model.predict(X)[0]
    classes   = list(model.classes_)
    class_idx = classes.index(label)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)           # (1, n_classes)
        score = round(float(proba[0, class_idx]), 3)
        return label, score, "Confidence score"

    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)         # (1,) binary or (1, n_classes) multi
        if raw.ndim == 1:
            score = round(float(raw[0]), 3)
        else:
            score = round(float(raw[0, class_idx]), 3)
        return label, score, "Decision score"

    return label, None, None


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a text string using the trained pipeline artifacts."
    )
    parser.add_argument(
        "--text",
        required=True,
        metavar="TEXT",
        help='Text to classify, e.g. --text "The app is easy to use"',
    )
    args = parser.parse_args()

    _, vectorizer, model = _load_artifacts()
    label, score, score_label = _infer(args.text, vectorizer, model)

    print(f"Predicted label: {label}")
    if score is not None:
        print(f"{score_label}: {score:.3f}")


if __name__ == "__main__":
    main()
