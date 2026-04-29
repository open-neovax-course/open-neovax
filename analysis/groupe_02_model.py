"""
Global scoring model for neoepitope ranking.

Usage:
    python analysis/score_analysis.py --generate
    python analysis/groupe_02_model.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

SCORES_ONE = ANALYSIS_DIR / "scores_patient_one.csv"

SEED = 42
CV_FOLDS = 5
LABEL_MAP = {"GOLD": 1, "GOOD": 1, "MEDIOCRE": 0, "BAD": 0, "TRAP": 0}


def load_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python analysis/score_analysis.py --generate` first."
        )
    df = pd.read_csv(path)
    if "label" not in df.columns and "note" in df.columns:
        df["label"] = df["note"].astype(str).str.split().str[0]
    return df


def prepare_binary_data(df: pd.DataFrame):
    df = df.copy()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(LABEL_MAP)].copy()
    features = [
        c for c in df.columns if c not in {"candidate_id", "label", "note", "ic50_nm"}
    ]
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df["label"].map(LABEL_MAP).astype(int)
    return X, y, features


def build_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaled = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    plain = [("imputer", SimpleImputer(strategy="median"))]

    return {
        "LogisticRegression": Pipeline(
            scaled
            + [
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000, class_weight="balanced", random_state=SEED
                    ),
                )
            ]
        ),
        "RandomForest": Pipeline(
            plain
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "GaussianNB": Pipeline(scaled + [("model", GaussianNB())]),
        "MLP": Pipeline(
            scaled
            + [
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        alpha=0.01,
                        learning_rate_init=0.005,
                        max_iter=3000,
                        random_state=SEED,
                    ),
                )
            ]
        ),
    }


def proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-scores))


def evaluate_models(X: pd.DataFrame, y: pd.Series):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    results = []

    print("\nCROSS-VALIDATION (patient_one)")
    for name, model in build_models().items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        model.fit(X, y)
        row = {
            "name": name,
            "model": model,
            "acc_mean": float(acc.mean()),
            "acc_std": float(acc.std()),
            "auc_mean": float(auc.mean()),
            "auc_std": float(auc.std()),
        }
        results.append(row)
        print(
            f"  {name:<20} accuracy={row['acc_mean']:.3f} +/- {row['acc_std']:.3f} "
            f"| auc={row['auc_mean']:.3f} +/- {row['auc_std']:.3f}"
        )

    return sorted(results, key=lambda r: (r["acc_mean"], r["auc_mean"]), reverse=True)


def main() -> None:
    train = load_scores(SCORES_ONE)
    X_train, y_train, features = prepare_binary_data(train)

    print("=" * 60)
    print("GLOBAL SCORING MODEL - GROUPE 02")
    print("=" * 60)
    print(f"Label map: {LABEL_MAP}")
    print(f"Training set: {len(X_train)} candidates, {len(features)} score modules")
    print("Models compared: LogisticRegression, RandomForest, GaussianNB, MLP")

    results = evaluate_models(X_train, y_train)
    best = results[0]
    print(
        f"\nBest model: {best['name']} "
        f"(accuracy={best['acc_mean']:.3f}, auc={best['auc_mean']:.3f})"
    )


if __name__ == "__main__":
    main()
