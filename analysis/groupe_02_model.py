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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

SCORES_ONE = ANALYSIS_DIR / "scores_patient_one.csv"
SCORES_ZERO = ANALYSIS_DIR / "scores_patient_zero.csv"

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


def feature_importance(best, X: pd.DataFrame, y: pd.Series, features: list[str]) -> pd.Series:
    from sklearn.inspection import permutation_importance

    model = best["model"].named_steps["model"]
    if hasattr(model, "feature_importances_"):
        s = pd.Series(model.feature_importances_, index=features, dtype=float)
    elif hasattr(model, "coef_"):
        s = pd.Series(np.abs(model.coef_[0]), index=features, dtype=float)
        if s.sum() > 0:
            s = s / s.sum()
    else:
        perm = permutation_importance(
            best["model"], X, y, n_repeats=30, random_state=SEED, scoring="accuracy"
        )
        s = pd.Series(perm.importances_mean, index=features, dtype=float)
    return s.sort_values(ascending=False)


def print_importance(importance: pd.Series, model_name: str) -> None:
    print("\nFEATURE IMPORTANCE")
    for feat, val in importance.items():
        print(f"  {feat:<24} {val:>7.3f}  {'#' * max(1, int(round(max(val, 0) * 50)))}")

    out = ANALYSIS_DIR / f"groupe_02_feature_importance_{model_name.lower()}.png"
    top = importance.head(12).sort_values()
    plt.figure(figsize=(10, 6))
    plt.barh(top.index, top.values, color="#2f6db3")
    plt.xlabel("Importance")
    plt.title(f"Top scoring modules - {model_name}")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\nFeature importance plot saved to {out}")


def rank_patient_zero(results, features: list[str]):
    zero = load_scores(SCORES_ZERO)
    X_zero = zero[features].apply(pd.to_numeric, errors="coerce")
    best_row, best_rank = results[0], None

    for row in results:
        scores = proba(row["model"], X_zero)
        ranking = pd.DataFrame(
            {"candidate_id": zero["candidate_id"], "score": scores, "label": zero["label"]}
        ).sort_values("score", ascending=False, ignore_index=True)

        print(f"\nFINAL RANKING (patient_zero) - {row['name']}")
        for i, r in ranking.iterrows():
            marker = "  <--" if r["candidate_id"] == "CAND_01" else ""
            print(
                f"  {i + 1:>2}. {r['candidate_id']:<8} {r['score']:.3f}  {r['label']}{marker}"
            )

        cand_rank = int(ranking.index[ranking["candidate_id"] == "CAND_01"][0]) + 1
        row["cand01_rank"] = cand_rank
        print(f"\nCAND_01 rank with {row['name']}: {cand_rank}")
        if best_rank is None or cand_rank < best_rank:
            best_row, best_rank = row, cand_rank

    print("\nPATIENT_ZERO COMPARISON")
    for row in results:
        print(
            f"  {row['name']:<22} cv_acc={row['acc_mean']:.3f}  "
            f"cv_auc={row['auc_mean']:.3f}  CAND_01_rank={row['cand01_rank']}"
        )
    print(f"\nBest patient_zero ranking model: {best_row['name']}")
    return best_row


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

    importance = feature_importance(best, X_train, y_train, features)
    print_importance(importance, best["name"])
    rank_patient_zero(results, features)


if __name__ == "__main__":
    main()
