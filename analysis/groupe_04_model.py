"""
Group 04 — ML Global Scoring Model
====================================

Trains 5 classifiers to combine module scores and rank neoepitope candidates.

Usage:
    python analysis/groupe_04_model.py

Prerequisites:
    python analysis/score_analysis.py --generate
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPARATION
# ──────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
# MEDIOCRE is ambiguous — we drop it from training


def load_score_matrix(filename: str) -> pd.DataFrame:
    path = ANALYSIS_DIR / filename
    if not path.exists():
        print(f"[ERROR] {path} not found. Run: python analysis/score_analysis.py --generate")
        sys.exit(1)
    return pd.read_csv(path)


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Filter to GOLD/GOOD/BAD/TRAP, encode labels, return (X, y)."""
    df = df[df["label"].isin(LABEL_MAP)].copy()
    y = df["label"].map(LABEL_MAP)
    X = df.drop(columns=["candidate_id", "label"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 2. MODELS
# ──────────────────────────────────────────────────────────────────────────────

MODELS: dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    "SVM (RBF)":           SVC(kernel="rbf", probability=True, random_state=42),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
}


# ──────────────────────────────────────────────────────────────────────────────
# 3. CROSS-VALIDATION COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def compare_models(X_scaled, y: pd.Series) -> str:
    """5-fold cross-validation for all models. Returns name of best model."""
    print("=" * 65)
    print("MODEL COMPARISON — 5-fold cross-validation (patient_one)")
    print("=" * 65)
    print(f"{'Model':<25}  {'Accuracy':>10}  {'Std':>8}  {'AUC':>8}")
    print("-" * 65)

    best_name, best_acc = "", 0.0
    for name, model in MODELS.items():
        acc_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
        auc_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")
        mean_acc = acc_scores.mean()
        print(
            f"{name:<25}  {mean_acc:>10.3f}  {acc_scores.std():>8.3f}  {auc_scores.mean():>8.3f}"
        )
        if mean_acc > best_acc:
            best_acc, best_name = mean_acc, name
    print("-" * 65)
    print(f"Best model: {best_name} (accuracy = {best_acc:.3f})\n")
    return best_name


# ──────────────────────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

def _bar(value: float, scale: float = 40) -> str:
    return "#" * max(0, int(abs(value) * scale))


def show_feature_importance(X: pd.DataFrame, X_scaled, y: pd.Series) -> None:
    print("=" * 65)
    print("FEATURE IMPORTANCE")
    print("=" * 65)

    # --- Logistic Regression (signed coefficients) ---
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    coefs = pd.Series(lr.coef_[0], index=X.columns)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    print("\n[Logistic Regression — coefficients]")
    print(f"  {'Feature':<35}  {'Coef':>7}  Direction")
    print("  " + "-" * 60)
    for feat, coef in coefs.items():
        direction = "→ GOOD/GOLD" if coef > 0 else "→ BAD/TRAP"
        marker = " ◄ our module" if feat == "C_binding_quality" else ""
        print(f"  {feat:<35}  {coef:>+7.3f}  {_bar(coef, 12)} {direction}{marker}")

    # --- Random Forest (impurity importance) ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\n[Random Forest — feature importance]")
    print(f"  {'Feature':<35}  {'Importance':>10}")
    print("  " + "-" * 55)
    for feat, imp in rf_imp.items():
        marker = " ◄ our module" if feat == "C_binding_quality" else ""
        print(f"  {feat:<35}  {imp:>10.3f}  {_bar(imp, 60)}{marker}")

    # --- Gradient Boosting (impurity importance) ---
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_scaled, y)
    gb_imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("\n[Gradient Boosting — feature importance]")
    print(f"  {'Feature':<35}  {'Importance':>10}")
    print("  " + "-" * 55)
    for feat, imp in gb_imp.items():
        marker = " ◄ our module" if feat == "C_binding_quality" else ""
        print(f"  {feat:<35}  {imp:>10.3f}  {_bar(imp, 60)}{marker}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# 5. RANKING — patient_zero
# ──────────────────────────────────────────────────────────────────────────────

def rank_patient_zero(
    X_train: pd.DataFrame,
    X_train_scaled,
    y_train: pd.Series,
    best_model_name: str,
    scaler: StandardScaler,
) -> None:
    print("=" * 65)
    print("RANKING — patient_zero (validation set)")
    print("=" * 65)

    df_zero = load_score_matrix("scores_patient_zero.csv")
    candidate_ids = df_zero["candidate_id"].tolist()
    labels_zero = df_zero["label"].tolist()

    X_zero = df_zero.drop(columns=["candidate_id", "label"])
    X_zero = X_zero.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    # Align to training columns
    X_zero = X_zero.reindex(columns=X_train.columns, fill_value=0.0)
    X_zero_scaled = scaler.transform(X_zero)

    # Evaluate all models on patient_zero
    print(f"\n{'Model':<25}  {'Accuracy':>10}  {'AUC':>8}")
    print("-" * 50)
    for name, model in MODELS.items():
        model.fit(X_train_scaled, y_train)
        y_zero_true = [LABEL_MAP.get(l, -1) for l in labels_zero]
        # Only evaluate on candidates with known labels
        known_mask = [l in LABEL_MAP for l in labels_zero]
        if sum(known_mask) == 0:
            continue
        y_true_k = [y for y, m in zip(y_zero_true, known_mask) if m]
        X_zero_k = X_zero_scaled[[i for i, m in enumerate(known_mask) if m]]
        y_pred_k = model.predict(X_zero_k)
        acc = sum(a == b for a, b in zip(y_true_k, y_pred_k)) / len(y_true_k)
        try:
            auc = roc_auc_score(y_true_k, model.predict_proba(X_zero_k)[:, 1])
        except Exception:
            auc = float("nan")
        print(f"{name:<25}  {acc:>10.3f}  {auc:>8.3f}")
    print()

    # Final ranking with best model
    best_model = MODELS[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    proba = best_model.predict_proba(X_zero_scaled)[:, 1]

    ranking = sorted(zip(candidate_ids, proba, labels_zero), key=lambda x: -x[1])

    print(f"Final ranking (best model: {best_model_name})")
    print(f"  {'#':>3}  {'Candidate':<12}  {'P(GOOD)':>8}  Label")
    print("  " + "-" * 42)
    for i, (cid, prob, label) in enumerate(ranking, 1):
        marker = " ← SUCCESS" if cid == "CAND_01" and i == 1 else (" ← CAND_01" if cid == "CAND_01" else "")
        print(f"  {i:>3}. {cid:<12}  {prob:>8.3f}  {label}{marker}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 6. REAL DATA — Spearman correlation + REAL vs DECOY AUC
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_real_data(
    X_train: pd.DataFrame,
    X_train_scaled,
    y_train: pd.Series,
    scaler: StandardScaler,
) -> None:
    print("=" * 65)
    print("REAL DATA — patient_real (experimental validation)")
    print("=" * 65)

    df_real_raw = pd.read_csv(PROJECT_ROOT / "data" / "patient_real.csv")
    df_real_scores = load_score_matrix("scores_patient_real.csv")

    # Merge ic50_nm from raw data
    df_real = df_real_scores.merge(
        df_real_raw[["candidate_id", "ic50_nm"]], on="candidate_id", how="left"
    )

    X_real = df_real.drop(columns=["candidate_id", "label", "ic50_nm"], errors="ignore")
    X_real = X_real.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_real = X_real.reindex(columns=X_train.columns, fill_value=0.0)
    X_real_scaled = scaler.transform(X_real)

    # 6a — Spearman correlation: pipeline score vs IC50 (REAL only)
    df_real_only = df_real[df_real["label"] == "REAL"].copy()
    df_real_only = df_real_only[df_real_only["ic50_nm"].notna()]

    if not df_real_only.empty:
        print("\n[Spearman correlation — pipeline score vs IC50 (REAL candidates)]")
        print(f"  {'Model':<25}  {'rho':>8}  {'p-value':>10}  Interpretation")
        print("  " + "-" * 65)

        for name, model in MODELS.items():
            model.fit(X_train_scaled, y_train)
            idx = df_real_only.index.tolist()
            X_real_only = X_real_scaled[
                [list(df_real.index).index(i) for i in idx]
            ]
            scores = model.predict_proba(X_real_only)[:, 1]
            ic50 = df_real_only["ic50_nm"].values
            # Lower IC50 = better binder → negative correlation expected
            rho, pval = spearmanr(scores, ic50)
            interp = "good (neg corr with IC50)" if rho < -0.2 else ("weak" if abs(rho) < 0.2 else "inverted")
            print(f"  {name:<25}  {rho:>8.3f}  {pval:>10.4f}  {interp}")

    # 6b — REAL vs DECOY AUC
    print("\n[REAL vs DECOY classification — AUC]")
    print(f"  {'Model':<25}  {'AUC':>8}")
    print("  " + "-" * 40)

    y_real_binary = (df_real["label"] == "REAL").astype(int)
    for name, model in MODELS.items():
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_real_scaled)[:, 1]
        try:
            auc = roc_auc_score(y_real_binary, proba)
        except Exception:
            auc = float("nan")
        print(f"  {name:<25}  {auc:>8.3f}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nGroup 04 — ML Global Scoring Model")
    print("Module: C_hla_delta_binding (PSSM-based WT vs MUT delta score)\n")

    # Load training data
    df_train = load_score_matrix("scores_patient_one.csv")
    X_train, y_train = prepare_training_data(df_train)

    print(
        f"Training set: {len(X_train)} candidates "
        f"({y_train.sum()} positive, {(y_train == 0).sum()} negative) "
        f"| {X_train.shape[1]} features\n"
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Steps 2-6
    best_model_name = compare_models(X_train_scaled, y_train)
    show_feature_importance(X_train, X_train_scaled, y_train)
    rank_patient_zero(X_train, X_train_scaled, y_train, best_model_name, scaler)
    evaluate_real_data(X_train, X_train_scaled, y_train, scaler)


if __name__ == "__main__":
    main()
