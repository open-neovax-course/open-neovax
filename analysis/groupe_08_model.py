"""
groupe_08_model.py — Global scoring model — Groupe 08
======================================================

Objective
---------
Improve the ranking of neo-epitope candidates by training a better
ML model than the baseline (RandomForest in score_analysis.py).

Strategy
--------
1. Compare 4 models: LogReg, RandomForest, GradientBoosting, SVM
2. Select the best by cross-validation (AUC-ROC)
3. Show feature importance
4. Produce a final ranking of patient_zero candidates

How to run
----------
    python analysis/groupe_08_model.py

Expected output
---------------
    - Cross-validation scores for each model
    - Feature importance ranking
    - Final ranking of patient_zero (CAND_01 should be near the top)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ══════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════

ANALYSIS_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ANALYSIS_DIR / "scores_patient_one.csv"
TEST_PATH = ANALYSIS_DIR / "scores_patient_zero.csv"


# ══════════════════════════════════════════════════════════════════════
#  LABEL ENCODING
# ══════════════════════════════════════════════════════════════════════


def encode_label(label: str) -> float:
    """Encode labels as binary: 1 = good candidate, 0 = bad candidate.

    GOLD and GOOD → 1 (we want the vaccine to target these)
    BAD and TRAP  → 0 (these should be excluded)
    Others        → NaN (ambiguous, removed from training)
    """
    if label in ("GOLD", "GOOD"):
        return 1.0
    if label in ("BAD", "TRAP"):
        return 0.0
    return np.nan


# ══════════════════════════════════════════════════════════════════════
#  LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════════════════


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load train and test sets, encode labels, return X_train, y_train, X_test."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train["y"] = train["label"].apply(encode_label)
    train = train.dropna(subset=["y"])

    feature_cols = [
        c for c in train.columns if c not in ("candidate_id", "label", "y")
    ]

    X_train = train[feature_cols].fillna(0.0)
    y_train = train["y"]

    # Test set may not have labels — keep candidate_id for ranking
    X_test = test[feature_cols].fillna(0.0)

    return X_train, y_train, X_test, test["candidate_id"]


# ══════════════════════════════════════════════════════════════════════
#  MODELS TO COMPARE
# ══════════════════════════════════════════════════════════════════════

MODELS: dict = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "SVM": SVC(probability=True, kernel="rbf", random_state=42),
}


# ══════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════


def compare_models(
    X: np.ndarray, y: pd.Series
) -> tuple[str, dict[str, float]]:
    """Run 5-fold stratified cross-validation for all models.

    Returns the name of the best model and all CV scores.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: dict[str, float] = {}

    print("\n=== CROSS-VALIDATION (5-fold, AUC-ROC) ===")
    for name, model in MODELS.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        mean = scores.mean()
        results[name] = mean
        print(f"  {name:<25} AUC = {mean:.3f}  (±{scores.std():.3f})")

    best_name = max(results, key=results.__getitem__)
    print(f"\n  → Best model: {best_name} (AUC = {results[best_name]:.3f})")
    return best_name, results


# ══════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════


def show_feature_importance(
    model: RandomForestClassifier | GradientBoostingClassifier,
    feature_names: list[str],
) -> None:
    """Print feature importance if the model supports it."""
    if not hasattr(model, "feature_importances_"):
        print("\n  (Feature importance not available for this model)")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 60)
        marker = " ← notre module A1" if feat == "A_hydrophobicity_kd" else ""
        print(f"  {feat:<35} {imp:.4f}  {bar}{marker}")


# ══════════════════════════════════════════════════════════════════════
#  FINAL RANKING
# ══════════════════════════════════════════════════════════════════════


def show_ranking(
    model,
    X_test_scaled: np.ndarray,
    candidate_ids: pd.Series,
) -> None:
    """Print the final ranking of patient_zero candidates."""
    probas = model.predict_proba(X_test_scaled)[:, 1]
    ranking = pd.DataFrame({"candidate_id": candidate_ids, "score": probas})
    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)

    print("\n=== FINAL RANKING (patient_zero) ===")
    print(f"  {'Rank':<6} {'Candidate':<15} {'Score':<8}")
    print("  " + "-" * 30)
    for i, row in ranking.iterrows():
        marker = " ✓ CAND_01" if row["candidate_id"] == "CAND_01" else ""
        print(f"  {i+1:<6} {row['candidate_id']:<15} {row['score']:.3f}{marker}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    print("Open-NeoVax — Groupe 08 — ML Model")
    print("=" * 45)

    # Load data
    X_train, y_train, X_test, candidate_ids = load_data()
    feature_names = list(X_train.columns)

    print(f"\nTraining set : {len(X_train)} candidates, {len(feature_names)} features")
    print(f"Test set     : {len(X_test)} candidates")
    print(f"Label balance: {int(y_train.sum())} good / {int((y_train == 0).sum())} bad")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compare models
    best_name, _ = compare_models(X_train_scaled, y_train)

    # Fit best model on full training set
    best_model = MODELS[best_name]
    best_model.fit(X_train_scaled, y_train)

    # Feature importance
    show_feature_importance(best_model, feature_names)

    # Final ranking
    show_ranking(best_model, X_test_scaled, candidate_ids)


if __name__ == "__main__":
    main()
