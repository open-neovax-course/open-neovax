"""
Open-NeoVax — Global Scoring Model (Group 11)
==============================================

Train an ML model to combine pipeline module scores optimally and rank
neoepitope candidates.  Addresses issue #39.

Usage
-----
    python analysis/groupe_11_model.py

Inputs (produced by ``python analysis/score_analysis.py --generate``)
------
    analysis/scores_patient_one.csv   — 75 training candidates
    analysis/scores_patient_zero.csv  — 18 validation candidates
    analysis/scores_patient_real.csv  — 138 real/decoy candidates
    data/patient_real.csv             — original file with ic50_nm values
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent

PATIENT_ONE_CSV = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_CSV = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL_CSV = ANALYSIS_DIR / "scores_patient_real.csv"
PATIENT_REAL_RAW_CSV = PROJECT_ROOT / "data" / "patient_real.csv"

OUTPUT_IMPORTANCE_PNG = ANALYSIS_DIR / "groupe_11_feature_importance.png"
OUTPUT_ROC_PNG = ANALYSIS_DIR / "groupe_11_roc_real_vs_decoy.png"
OUTPUT_SPEARMAN_PNG = ANALYSIS_DIR / "groupe_11_spearman_ic50.png"

RANDOM_STATE = 42

# Labels mapped to binary class
LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
# MEDIOCRE and NEUTRAL are ambiguous — removed from training


# ══════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & PREPARATION
# ══════════════════════════════════════════════════════════════════════


def _check_files() -> bool:
    """Return True if all required score matrices exist."""
    missing = [p for p in (PATIENT_ONE_CSV, PATIENT_ZERO_CSV) if not p.exists()]
    if missing:
        print("ERROR: missing score matrices:")
        for p in missing:
            print(f"  {p}")
        print("Run:  python analysis/score_analysis.py --generate")
        return False
    return True


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a score CSV and fill missing numeric values with 0.0."""
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("candidate_id", "label")]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def load_data() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    Optional[np.ndarray],
    list[str],
    StandardScaler,
]:
    """Load, encode, scale, and align training and validation sets.

    Returns
    -------
    df_one       : full patient_one dataframe (including filtered rows)
    X_train      : scaled feature matrix for training (MEDIOCRE/NEUTRAL removed)
    y_train      : binary labels for training
    df_zero      : full patient_zero dataframe
    X_zero       : scaled feature matrix for patient_zero (all rows)
    y_zero       : binary labels for patient_zero (None for unknown labels)
    feature_cols : list of feature column names
    scaler       : fitted StandardScaler
    """
    df_one = _load_csv(PATIENT_ONE_CSV)
    df_zero = _load_csv(PATIENT_ZERO_CSV)

    # Identify feature columns (same order for both sets)
    feature_cols = [c for c in df_one.columns if c not in ("candidate_id", "label")]

    # Align patient_zero to the same feature columns (fill missing with 0)
    for col in feature_cols:
        if col not in df_zero.columns:
            df_zero[col] = 0.0

    print(f"  patient_one  : {len(df_one)} candidates, {len(feature_cols)} features")
    print(f"  patient_zero : {len(df_zero)} candidates")

    # Encode labels
    df_one["y"] = df_one["label"].map(LABEL_MAP)
    train_mask = df_one["y"].notna()
    n_before = len(df_one)
    n_after = train_mask.sum()
    print(
        f"  After filtering ambiguous labels: {n_after} training samples "
        f"(removed {n_before - n_after})"
    )

    X_train_raw = df_one.loc[train_mask, feature_cols].values.astype(float)
    y_train = df_one.loc[train_mask, "y"].values.astype(int)

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    X_zero = scaler.transform(df_zero[feature_cols].values.astype(float))

    # Build a binary label array for patient_zero (keep NaN for unknowns)
    df_zero["y"] = df_zero["label"].map(LABEL_MAP)

    return (
        df_one,
        X_train,
        y_train,
        df_zero,
        X_zero,
        df_zero["y"].values,
        feature_cols,
        scaler,
    )


# ══════════════════════════════════════════════════════════════════════
#  2. MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[object, str]:
    """Run 5-fold CV for three classifiers and return the best one.

    Parameters
    ----------
    X_train : scaled feature matrix
    y_train : binary label vector

    Returns
    -------
    best_model  : unfitted best estimator instance
    best_name   : display name of the best model
    """
    models: list[tuple[str, object]] = [
        (
            "Logistic Regression    ",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        (
            "Random Forest          ",
            RandomForestClassifier(
                max_depth=5, n_estimators=100, random_state=RANDOM_STATE
            ),
        ),
        (
            "Gradient Boosting      ",
            GradientBoostingClassifier(
                max_depth=3, n_estimators=100, random_state=RANDOM_STATE
            ),
        ),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_score = -1.0
    best_model = None
    best_name = ""

    for name, clf in models:
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        mean, std = scores.mean(), scores.std()
        print(f"  {name}  accuracy = {mean:.3f} (+/- {std:.3f})")
        if mean > best_score:
            best_score = mean
            best_model = clf
            best_name = name.strip()

    print(f"\n  Best model: {best_name}  (CV accuracy = {best_score:.3f})")
    return best_model, best_name


# ══════════════════════════════════════════════════════════════════════
#  3. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════


def analyse_feature_importance(
    best_model: object,
    best_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
) -> None:
    """Compute and plot feature importances; save PNG.

    Shows:
    - RF built-in importances (if available)
    - LR coefficients (if available)
    - Permutation importance on the best fitted model
    """
    # Fit best model on full training set
    best_model.fit(X_train, y_train)
    # NOTE: model stays fitted — reused by rank_patient_zero() and analyse_patient_real()

    # --- Permutation importance (model-agnostic) ---
    perm = permutation_importance(
        best_model,
        X_train,
        y_train,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="accuracy",
    )
    perm_imp = perm.importances_mean

    # --- Built-in importance (RF or GB) ---
    builtin_imp: Optional[np.ndarray] = None
    builtin_label = "Built-in importance"
    if hasattr(best_model, "feature_importances_"):
        builtin_imp = best_model.feature_importances_
        builtin_label = "Tree feature importance"
    elif hasattr(best_model, "coef_"):
        builtin_imp = best_model.coef_[0]
        builtin_label = "LR coefficient (+ → GOOD)"

    # --- Print text table (sorted by permutation importance) ---
    order = np.argsort(perm_imp)[::-1]
    print(f"\n  {'Feature':<38}  Perm.Imp")
    print(f"  {'-'*38}  --------")
    for i in order:
        bar = "#" * max(0, int(perm_imp[i] * 80))
        print(f"  {feature_cols[i]:<38}  {perm_imp[i]:+.4f}  {bar}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Feature Importance — {best_name}\n(Group 11)", fontsize=13)

    # Left: permutation importance
    sorted_feats = [feature_cols[i] for i in order]
    sorted_perm = perm_imp[order]
    colors_perm = ["#2196F3" if v >= 0 else "#F44336" for v in sorted_perm]
    axes[0].barh(sorted_feats, sorted_perm, color=colors_perm)
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Permutation Importance")
    axes[0].set_xlabel("Mean accuracy decrease")
    axes[0].invert_yaxis()

    # Right: built-in importance or coefficients
    if builtin_imp is not None:
        bi_order = np.argsort(np.abs(builtin_imp))[::-1]
        sorted_bi_feats = [feature_cols[i] for i in bi_order]
        sorted_bi = builtin_imp[bi_order]
        colors_bi = ["#4CAF50" if v >= 0 else "#FF5722" for v in sorted_bi]
        axes[1].barh(sorted_bi_feats, sorted_bi, color=colors_bi)
        axes[1].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[1].set_title(builtin_label)
        axes[1].invert_yaxis()
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMPORTANCE_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_IMPORTANCE_PNG.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════
#  4. FINAL RANKING ON PATIENT_ZERO
# ══════════════════════════════════════════════════════════════════════


def rank_patient_zero(
    best_model: object,
    X_zero: np.ndarray,
    df_zero: pd.DataFrame,
) -> None:
    """Predict probabilities on patient_zero and print ranked list."""
    proba = best_model.predict_proba(X_zero)[:, 1]
    df_out = df_zero[["candidate_id", "label"]].copy()
    df_out["prob"] = proba
    df_out = df_out.sort_values("prob", ascending=False).reset_index(drop=True)

    for rank, row in df_out.iterrows():
        marker = "  <--" if row["candidate_id"] == "CAND_01" else ""
        print(
            f"  {rank + 1:>2}. {row['candidate_id']:<10}  {row['prob']:.4f}  "
            f"{row['label']}{marker}"
        )

    # Check success criterion
    top_id = df_out.iloc[0]["candidate_id"]
    cand01_rank = df_out[df_out["candidate_id"] == "CAND_01"].index[0] + 1
    if cand01_rank <= 3:
        print(f"\n  [OK] CAND_01 ranked #{cand01_rank} — success criterion met.")
    else:
        print(f"\n  [!] CAND_01 ranked #{cand01_rank} — top candidate is {top_id}.")


# ══════════════════════════════════════════════════════════════════════
#  5 & 6. REAL DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════


def analyse_patient_real(
    best_model: object,
    feature_cols: list[str],
    scaler: StandardScaler,
) -> None:
    """Run bonus analyses on patient_real data."""
    if not PATIENT_REAL_CSV.exists():
        print("  scores_patient_real.csv not found — skipping.")
        return
    if not PATIENT_REAL_RAW_CSV.exists():
        print("  data/patient_real.csv not found — skipping IC50 analysis.")

    df_real = _load_csv(PATIENT_REAL_CSV)

    # Align features
    for col in feature_cols:
        if col not in df_real.columns:
            df_real[col] = 0.0
    X_real = scaler.transform(df_real[feature_cols].values.astype(float))
    proba_real = best_model.predict_proba(X_real)[:, 1]
    df_real["model_prob"] = proba_real

    # ── Task B: REAL vs DECOY AUC ────────────────────────────────────
    real_mask = df_real["label"] == "REAL"
    decoy_mask = df_real["label"] == "DECOY"
    y_rd = np.where(real_mask, 1, np.where(decoy_mask, 0, np.nan))
    valid = ~np.isnan(y_rd)
    if valid.sum() > 0:
        auc = roc_auc_score(y_rd[valid].astype(int), proba_real[valid])
        fpr, tpr, _ = roc_curve(y_rd[valid].astype(int), proba_real[valid])
        print(f"  AUC REAL vs DECOY = {auc:.4f}")

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — REAL vs DECOY (Group 11)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(OUTPUT_ROC_PNG, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {OUTPUT_ROC_PNG.relative_to(PROJECT_ROOT)}")
    else:
        print("  No REAL/DECOY labels found in patient_real scores.")

    # ── Task A: Spearman correlation with IC50 ───────────────────────
    if not PATIENT_REAL_RAW_CSV.exists():
        return

    df_raw = pd.read_csv(PATIENT_REAL_RAW_CSV)
    # Keep only REAL candidates with valid IC50 values
    df_raw_real = df_raw[(df_raw["label"] == "REAL") & df_raw["ic50_nm"].notna()][
        ["candidate_id", "ic50_nm"]
    ].copy()

    df_merged = df_real[real_mask][["candidate_id", "model_prob"]].merge(
        df_raw_real, on="candidate_id", how="inner"
    )

    if len(df_merged) < 3:
        print("  Not enough REAL candidates with IC50 for Spearman correlation.")
        return

    # Lower IC50 = better binder → higher model score expected → negative rho
    rho, p_val = stats.spearmanr(df_merged["model_prob"], df_merged["ic50_nm"])
    print(f"  Spearman rho = {rho:.4f}  (p = {p_val:.4f})")
    if p_val < 0.05:
        print("  [OK] Statistically significant correlation with IC50.")
    else:
        print("  [!] Correlation with IC50 not significant (p >= 0.05).")

    # Scatter plot: model probability vs IC50
    plt.figure(figsize=(6, 5))
    plt.scatter(
        df_merged["ic50_nm"],
        df_merged["model_prob"],
        color="#4CAF50",
        edgecolors="k",
        linewidth=0.5,
        s=60,
        alpha=0.8,
    )
    plt.xlabel("IC50 (nM) — lower = better binder")
    plt.ylabel("Model probability (REAL)")
    plt.title(
        f"Model prob vs IC50 — Spearman ρ = {rho:.3f} (p = {p_val:.3f})\n(Group 11)"
    )
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUT_SPEARMAN_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_SPEARMAN_PNG.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point — run all analysis steps sequentially."""
    sep = "═" * 43

    print(sep)
    print("  Open-NeoVax — Global Scoring Model (Group 11)")
    print(sep)
    print()

    if not _check_files():
        sys.exit(1)

    # 1. Load data
    print("Loading data...")
    (df_one, X_train, y_train, df_zero, X_zero, y_zero, feature_cols, scaler) = (
        load_data()
    )

    # 2. Model comparison
    print()
    print("─── MODEL COMPARISON (5-fold CV) " + "─" * 10)
    best_model, best_name = compare_models(X_train, y_train)

    # 3. Feature importance
    print()
    print("─── FEATURE IMPORTANCE " + "─" * 20)
    analyse_feature_importance(best_model, best_name, X_train, y_train, feature_cols)

    # 4. Final ranking on patient_zero
    # best_model is already fitted from step 3 (feature importance)
    print()
    print("─── FINAL RANKING (patient_zero) " + "─" * 10)
    rank_patient_zero(best_model, X_zero, df_zero)

    # 5 & 6. Patient_real analysis
    print()
    print("─── REAL DATA ANALYSIS " + "─" * 20)
    analyse_patient_real(best_model, feature_cols, scaler)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
