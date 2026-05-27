"""
pipeline.py — Local text classification pipeline.
Stages 1–8: data loading, validation, preprocessing, splitting,
feature extraction, model training, evaluation, and winner selection.
"""

import datetime
import json
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DATA_LOADED
# ─────────────────────────────────────────────────────────────────────────────

def stage_data_loaded():
    print("STAGE 1 - DATA_LOADED")

    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")

    with open("config.json", "r", encoding="utf-8") as fh:
        config = json.load(fh)

    return train_df, test_df, config


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DATA_VALIDATED
# ─────────────────────────────────────────────────────────────────────────────

def stage_data_validated(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("STAGE 2 - DATA_VALIDATED")

    checks = []
    errors = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        entry = {"check": name, "passed": passed}
        if detail:
            entry["detail"] = detail
        checks.append(entry)
        if not passed:
            errors.append(detail or name)

    # ── Required columns ──────────────────────────────────────────────────────
    train_required = {"id", "text", "label"}
    test_required  = {"id", "text"}

    missing_train = train_required - set(train_df.columns)
    if missing_train:
        record("train_columns", False,
               f"train.csv is missing columns: {sorted(missing_train)}")
    else:
        record("train_columns", True)

    missing_test = test_required - set(test_df.columns)
    if missing_test:
        record("test_columns", False,
               f"test.csv is missing columns: {sorted(missing_test)}")
    else:
        record("test_columns", True)

    # ── At least 2 distinct labels in train ───────────────────────────────────
    if "label" in train_df.columns:
        n_labels = train_df["label"].nunique()
        if n_labels < 2:
            record("distinct_labels", False,
                   f"train.csv has only {n_labels} distinct label(s); need at least 2")
        else:
            record("distinct_labels", True,
                   f"{n_labels} distinct labels: {sorted(train_df['label'].unique())}")

    # ── No empty text fields ──────────────────────────────────────────────────
    if "text" in train_df.columns:
        n_empty = train_df["text"].astype(str).str.strip().eq("").sum()
        if n_empty:
            record("train_empty_text", False,
                   f"train.csv has {n_empty} empty text field(s)")
        else:
            record("train_empty_text", True)

    if "text" in test_df.columns:
        n_empty = test_df["text"].astype(str).str.strip().eq("").sum()
        if n_empty:
            record("test_empty_text", False,
                   f"test.csv has {n_empty} empty text field(s)")
        else:
            record("test_empty_text", True)

    # ── Unique IDs within each file ───────────────────────────────────────────
    if "id" in train_df.columns:
        n_dup = int(train_df["id"].duplicated().sum())
        if n_dup:
            record("train_unique_ids", False,
                   f"train.csv has {n_dup} duplicate ID(s)")
        else:
            record("train_unique_ids", True)

    if "id" in test_df.columns:
        n_dup = int(test_df["id"].duplicated().sum())
        if n_dup:
            record("test_unique_ids", False,
                   f"test.csv has {n_dup} duplicate ID(s)")
        else:
            record("test_unique_ids", True)

    # ── Persist report ────────────────────────────────────────────────────────
    report = {"passed": len(errors) == 0, "checks": checks}
    with open("data_validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if errors:
        print("ERROR: Data validation failed:")
        for msg in errors:
            print(f"  ✗ {msg}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — TEXT_PREPROCESSED
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Lowercase → strip surrounding whitespace → collapse internal whitespace."""
    text = str(text).lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def stage_text_preprocessed(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("STAGE 3 - TEXT_PREPROCESSED")

    # Capture 3 before/after examples before mutating the dataframe
    examples = [
        {
            "id":     int(row["id"]),
            "before": str(row["text"]),
            "after":  _clean(row["text"]),
        }
        for _, row in train_df.head(3).iterrows()
    ]

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df["text"] = train_df["text"].apply(_clean)
    test_df["text"]  = test_df["text"].apply(_clean)

    with open("preprocessing_preview.json", "w", encoding="utf-8") as fh:
        json.dump({"examples": examples}, fh, indent=2)

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SPLIT_CREATED
# ─────────────────────────────────────────────────────────────────────────────

def stage_split_created(
    train_df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("STAGE 4 - SPLIT_CREATED")

    seed     = config["random_seed"]
    val_frac = config["validation_split"]

    X_train_idx, X_val_idx = train_test_split(
        train_df.index,
        test_size=val_frac,
        random_state=seed,
        stratify=train_df["label"],
    )

    train_split = train_df.loc[X_train_idx].copy()
    val_split   = train_df.loc[X_val_idx].copy()

    labels_info: dict[str, dict[str, int]] = {}
    for label in sorted(train_df["label"].unique()):
        labels_info[label] = {
            "train":      int((train_split["label"] == label).sum()),
            "validation": int((val_split["label"]   == label).sum()),
        }

    split_report = {
        "random_seed":     seed,
        "train_size":      len(train_split),
        "validation_size": len(val_split),
        "labels":          labels_info,
    }

    with open("split_report.json", "w", encoding="utf-8") as fh:
        json.dump(split_report, fh, indent=2)

    return train_split, val_split


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — FEATURES_FIT
# ─────────────────────────────────────────────────────────────────────────────

def stage_features_fit(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    config: dict,
) -> tuple:
    print("STAGE 5 - FEATURES_FIT")

    vec_cfg    = config["vectorizer"]
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(vec_cfg["ngram_range"]),  # list → tuple for sklearn
        max_features=vec_cfg["max_features"],
        min_df=vec_cfg["min_df"],
    )

    # Fit on training text only — validation and test are never seen during fit
    X_train = vectorizer.fit_transform(train_split["text"])
    X_val   = vectorizer.transform(val_split["text"])
    y_train = train_split["label"].values
    y_val   = val_split["label"].values

    joblib.dump(vectorizer, "vectorizer.joblib")

    return X_train, X_val, y_train, y_val, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — MODELS_TRAINED
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(name: str, seed: int):
    """Return a freshly instantiated sklearn classifier by pipeline name."""
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, max_iter=1000)
    if name == "linear_svm":
        # LinearSVC uses random_state only to break ties in the dual solver
        return LinearSVC(random_state=seed, max_iter=2000)
    if name == "naive_bayes":
        # MultinomialNB is fully deterministic; random_state is not accepted
        return MultinomialNB()
    raise ValueError(f"Unknown model name in config: {name!r}")


def stage_models_trained(X_train, y_train, config: dict) -> dict:
    print("STAGE 6 - MODELS_TRAINED")

    seed   = config["random_seed"]
    models = {}

    for name in config["models"]:
        clf = _build_model(name, seed)
        clf.fit(X_train, y_train)
        joblib.dump(clf, f"{name}.joblib")
        models[name] = clf

    return models


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — MODELS_EVALUATED
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_one(clf, X_val, y_val) -> dict:
    """Compute accuracy, macro averages, and per-class metrics for one model."""
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    macro  = report["macro avg"]

    metrics: dict = {
        "accuracy":        round(float(accuracy_score(y_val, y_pred)), 4),
        "macro_precision": round(float(macro["precision"]), 4),
        "macro_recall":    round(float(macro["recall"]),    4),
        "macro_f1":        round(float(macro["f1-score"]),  4),
        "per_class":       {},
    }

    # Per-class rows — skip the aggregate keys injected by sklearn
    _skip = {"accuracy", "macro avg", "weighted avg"}
    for label, scores in report.items():
        if label in _skip:
            continue
        metrics["per_class"][label] = {
            "precision": round(float(scores["precision"]), 4),
            "recall":    round(float(scores["recall"]),    4),
            "f1":        round(float(scores["f1-score"]),  4),
            "support":   int(scores["support"]),
        }

    return metrics


def stage_models_evaluated(models: dict, X_val, y_val) -> dict:
    print("STAGE 7 - MODELS_EVALUATED")

    all_metrics = {
        name: _evaluate_one(clf, X_val, y_val)
        for name, clf in models.items()
    }

    with open("metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"models": all_metrics}, fh, indent=2)

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — WINNER_SELECTED
# ─────────────────────────────────────────────────────────────────────────────

# Maps config selection_metric names to metrics dict keys
_METRIC_KEY: dict[str, str] = {
    "macro_f1":        "macro_f1",
    "macro_precision": "macro_precision",
    "macro_recall":    "macro_recall",
    "accuracy":        "accuracy",
}


def stage_winner_selected(
    models: dict,
    all_metrics: dict,
    config: dict,
) -> tuple[str, object]:
    print("STAGE 8 - WINNER_SELECTED")

    selection_metric = config["selection_metric"]
    primary_key      = _METRIC_KEY[selection_metric]

    # Flat list of candidates for sorting and reporting
    candidates = [
        {
            "model":           name,
            "macro_f1":        m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall":    m["macro_recall"],
            "accuracy":        m["accuracy"],
        }
        for name, m in all_metrics.items()
    ]

    # Deterministic three-level sort:
    #   1. primary metric descending   (higher is better)
    #   2. macro_precision descending  (tie-break 1)
    #   3. model name ascending        (tie-break 2 — alphabetical)
    rankings = sorted(
        candidates,
        key=lambda c: (-c[primary_key], -c["macro_precision"], c["model"]),
    )

    winner = rankings[0]
    second = rankings[1] if len(rankings) > 1 else None

    # Identify which level of the sort actually decided the outcome
    tie_breaking_applied = False
    tie_breaking_steps: list[str] = []

    if second is None or winner[primary_key] > second[primary_key]:
        # Clear win — no tie to break
        reason = (
            f"'{winner['model']}' selected with the highest {selection_metric} "
            f"({winner[primary_key]:.4f}). No tie-breaking needed."
        )
    elif winner["macro_precision"] > second["macro_precision"]:
        # Tied on primary; precision separated winner from runner-up
        tie_breaking_applied = True
        tie_breaking_steps   = ["macro_precision"]
        reason = (
            f"'{winner['model']}' selected after tie on {selection_metric} "
            f"({winner[primary_key]:.4f}). Resolved by higher macro_precision "
            f"({winner['macro_precision']:.4f} vs {second['macro_precision']:.4f})."
        )
    else:
        # Tied on primary and precision; alphabetical name was decisive
        tie_breaking_applied = True
        tie_breaking_steps   = ["macro_precision", "alphabetical_name"]
        reason = (
            f"'{winner['model']}' selected after tie on both {selection_metric} "
            f"({winner[primary_key]:.4f}) and macro_precision "
            f"({winner['macro_precision']:.4f}). Resolved alphabetically by model name."
        )

    report = {
        "selection_metric":        selection_metric,
        "winner":                  winner["model"],
        "reason":                  reason,
        "tie_breaking_applied":    tie_breaking_applied,
        "tie_breaking_steps_used": tie_breaking_steps,
        "rankings": [
            {"rank": i + 1, **c} for i, c in enumerate(rankings)
        ],
    }

    with open("model_selection_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return winner["model"], models[winner["model"]]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9 — ERROR_ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _get_top_tokens(text: str, vectorizer, clf, pred_class_idx: int, n: int = 3) -> list[str]:
    """
    Return the n vocabulary tokens in *text* that contributed most strongly
    toward the predicted class, based on TF-IDF weight × model coefficient.
    Returns an empty list when the model exposes no per-feature weights.
    """
    try:
        x_row        = vectorizer.transform([text])          # (1, n_features) sparse
        feature_names = vectorizer.get_feature_names_out()
        _, col_idx   = x_row.nonzero()

        if hasattr(clf, "coef_"):
            # LogisticRegression / LinearSVC:
            #   binary  → coef_ shape (1, n_features); positive = toward classes_[1]
            #   multi   → coef_ shape (n_classes, n_features)
            w = clf.coef_
            if w.shape[0] == 1:
                row_w = w[0] if pred_class_idx == 1 else -w[0]
            else:
                row_w = w[pred_class_idx]
        elif hasattr(clf, "feature_log_prob_"):
            # MultinomialNB
            row_w = clf.feature_log_prob_[pred_class_idx]
        else:
            return []

        scored = [
            (feature_names[j], float(x_row[0, j]) * float(row_w[j]))
            for j in col_idx
        ]
        return [tok for tok, _ in sorted(scored, key=lambda x: -x[1])[:n]]
    except Exception:
        return []


def _confidence(clf, x_row, pred_class_idx: int) -> tuple[float | None, str]:
    """
    Return (score, method) for a single-sample sparse row.
    Priority: predict_proba → decision_function → (None, 'null').
    """
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(x_row)             # (1, n_classes)
        return round(float(p[0, pred_class_idx]), 4), "predict_proba"

    if hasattr(clf, "decision_function"):
        df = clf.decision_function(x_row)        # (1,) binary or (1, n_classes) multi
        if df.ndim == 1:
            raw = float(df[0])
        else:
            raw = float(df[0, pred_class_idx])
        return round(raw, 4), "decision_function"

    return None, "null"


def stage_error_analysis(
    models: dict,
    vectorizer,
    val_split: pd.DataFrame,
    X_val,
    y_val,
    config: dict,
) -> None:
    print("STAGE 9 - ERROR_ANALYSIS")

    with open("model_selection_report.json", "r", encoding="utf-8") as fh:
        winner_name = json.load(fh)["winner"]
    clf     = models[winner_name]
    classes = list(clf.classes_)          # class ordering sklearn uses
    top_k   = config["top_k_error_examples"]

    y_pred   = clf.predict(X_val)
    val_rows = val_split.reset_index(drop=True)   # aligns row i with X_val[i]

    errors: list[dict] = []
    for i, (true, pred) in enumerate(zip(y_val, y_pred)):
        if true == pred:
            continue

        pred_idx            = classes.index(pred)
        score, score_method = _confidence(clf, X_val[i], pred_idx)
        top_tokens          = _get_top_tokens(
            val_rows.iloc[i]["text"], vectorizer, clf, pred_idx
        )

        reason = (
            f"Model predicted '{pred}' ({score_method}: {score}) "
            f"but true label was '{true}'."
        )
        if top_tokens:
            reason += f" Top contributing tokens: {top_tokens}."

        errors.append({
            "id":                  int(val_rows.iloc[i]["id"]),
            "text":                val_rows.iloc[i]["text"],
            "true_label":          str(true),
            "predicted_label":     str(pred),
            "confidence_or_score": score,
            "reason":              reason,
        })

    # Sort by most-confident wrong prediction first, then cap at top_k
    errors.sort(key=lambda r: abs(r["confidence_or_score"] or 0), reverse=True)
    errors = errors[:top_k]

    with open("error_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "winner_model": winner_name,
                "total_errors": len(errors),
                "top_k":        top_k,
                "errors":       errors,
            },
            fh,
            indent=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 10 — TEST_PREDICTIONS_GENERATED
# ─────────────────────────────────────────────────────────────────────────────

def stage_test_predictions_generated(
    winner_model,
    vectorizer,
    test_df: pd.DataFrame,
) -> None:
    print("STAGE 10 - TEST_PREDICTIONS_GENERATED")

    # test_df already carries preprocessed text from stage 3
    X_test = vectorizer.transform(test_df["text"])
    y_pred = winner_model.predict(X_test)

    pd.DataFrame({"id": test_df["id"].values, "predicted_label": y_pred}).to_csv(
        "test_predictions.csv", index=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 11 — REPORT_EXPORTED
# ─────────────────────────────────────────────────────────────────────────────

def stage_report_exported(
    config: dict,
    train_df: pd.DataFrame,       # full preprocessed training set (pre-split)
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    winner_name: str,
    all_metrics: dict,
) -> None:
    print("STAGE 11 - REPORT_EXPORTED")

    # ── run_manifest.json ─────────────────────────────────────────────────────
    wm = all_metrics[winner_name]

    manifest = {
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "random_seed":    config["random_seed"],
        "files_read":     ["train.csv", "test.csv", "config.json"],
        "models_trained": config["models"],
        "winning_model":  winner_name,
        "key_metrics": {
            "accuracy":        wm["accuracy"],
            "macro_f1":        wm["macro_f1"],
            "macro_precision": wm["macro_precision"],
            "macro_recall":    wm["macro_recall"],
        },
        "artifact_paths": {
            "vectorizer": "vectorizer.joblib",
            "models":     {n: f"{n}.joblib" for n in config["models"]},
            "reports": [
                "data_validation_report.json",
                "preprocessing_preview.json",
                "split_report.json",
                "metrics.json",
                "model_selection_report.json",
                "error_analysis.json",
                "test_predictions.csv",
                "run_manifest.json",
                "safeguards_report.json",
            ],
        },
    }

    with open("run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # ── safeguards_report.json ────────────────────────────────────────────────
    checks: list[dict] = []

    # 1. Class imbalance — any single class > 70 % of the full training set
    label_counts = train_df["label"].value_counts()
    total        = len(train_df)
    imbalanced   = {lbl: cnt for lbl, cnt in label_counts.items() if cnt / total > 0.70}
    if imbalanced:
        detail = "; ".join(
            f"'{lbl}' = {cnt/total:.1%}" for lbl, cnt in imbalanced.items()
        )
        checks.append({
            "check":   "class_imbalance",
            "warning": True,
            "detail":  f"High class imbalance detected — {detail} (threshold: 70%).",
        })
    else:
        dist = {lbl: f"{cnt/total:.1%}" for lbl, cnt in label_counts.items()}
        checks.append({
            "check":   "class_imbalance",
            "warning": False,
            "detail":  f"All classes within acceptable range. Distribution: {dist}.",
        })

    # 2. Any training class absent from the validation split
    missing_in_val = set(train_df["label"].unique()) - set(val_split["label"].unique())
    if missing_in_val:
        checks.append({
            "check":   "validation_class_coverage",
            "warning": True,
            "detail":  (
                f"Class(es) {sorted(missing_in_val)} present in training "
                f"but absent from validation split."
            ),
        })
    else:
        checks.append({
            "check":   "validation_class_coverage",
            "warning": False,
            "detail":  "All training classes are represented in the validation split.",
        })

    # 3. Exact duplicate texts shared between train and validation splits
    train_texts = set(train_split["text"])
    val_texts   = set(val_split["text"])
    overlap     = train_texts & val_texts
    if overlap:
        checks.append({
            "check":    "train_val_text_overlap",
            "warning":  True,
            "detail":   (
                f"{len(overlap)} exact duplicate text(s) found between "
                f"train and validation splits."
            ),
            "examples": sorted(overlap)[:3],
        })
    else:
        checks.append({
            "check":   "train_val_text_overlap",
            "warning": False,
            "detail":  "No exact duplicate texts between train and validation splits.",
        })

    safeguards = {
        "total_warnings": sum(1 for c in checks if c["warning"]),
        "checks":         checks,
    }

    with open("safeguards_report.json", "w", encoding="utf-8") as fh:
        json.dump(safeguards, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    train_df, test_df, config = stage_data_loaded()
    stage_data_validated(train_df, test_df)
    train_df, test_df = stage_text_preprocessed(train_df, test_df)
    train_split, val_split = stage_split_created(train_df, config)

    X_train, X_val, y_train, y_val, vectorizer = stage_features_fit(
        train_split, val_split, config
    )
    models      = stage_models_trained(X_train, y_train, config)
    all_metrics = stage_models_evaluated(models, X_val, y_val)
    winner_name, winner_model = stage_winner_selected(models, all_metrics, config)

    stage_error_analysis(models, vectorizer, val_split, X_val, y_val, config)
    stage_test_predictions_generated(winner_model, vectorizer, test_df)
    stage_report_exported(config, train_df, train_split, val_split, winner_name, all_metrics)


if __name__ == "__main__":
    main()
