"""
Global Scoring Model — Open-NeoVax
=========================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent

PATIENT_ONE_CSV = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_CSV = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL_CSV = ANALYSIS_DIR / "scores_patient_real.csv"
PATIENT_REAL_RAW_CSV = PROJECT_ROOT / "data" / "patient_real.csv"

LABEL_MAP_BINARY = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
RANDOM_STATE = 42

BIO_MEANING = {
    "C_total_binding": "Aggregated MHC binding",
    "C_hla_delta_binding": "Delta binding (MUT vs WT)",
    "C_anchoring_P2": "P2 anchor quality",
    "C_hla_anchor_p9": "P9/C-term anchor quality",
    "C_binding_quality": "Overall HLA binding",
    "D_mutation_in_window": "Mutation in TCR window",
    "B_sanity_check": "Sequence validity check",
    "A_hydrophobicity_kd": "Kyte-Doolittle hydrophobicity",
    "A_delta_wt_vs_mut": "Physicochemical delta",
    "A_tcr_contact_potential": "TCR contact probability",
    "B_tap_transport_score": "TAP transport efficiency",
    "B_proteasome_cterm": "C-term proteasomal cleavage",
    "B_erap_nterm_proxy": "N-term ERAP trimming",
    "D1_exact_self_similarity": "Human proteome self-similarity",
    "d1_exact_self_similarity": "Human proteome self-similarity",
    "A_hybrid_complexity": "Sequence complexity",
    "A_net_charge": "Peptide net charge",
    "A_mutation_surprisal": "Mutation surprisal score",
    "D_wt_presented": "WT HLA presentation",
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA ENGINE
# ──────────────────────────────────────────────────────────────────────────────


def load_and_preprocess():
    print("=" * 50)
    print("1. DATA LOADING & PREPROCESSING")
    print("=" * 50)

    df_one = pd.read_csv(PATIENT_ONE_CSV)
    df_zero = pd.read_csv(PATIENT_ZERO_CSV)
    df_real = (
        pd.read_csv(PATIENT_REAL_CSV) if PATIENT_REAL_CSV.exists() else pd.DataFrame()
    )

    feature_cols = [c for c in df_one.columns if c not in ("candidate_id", "label")]

    for df in [df_one, df_zero, df_real]:
        if not df.empty:
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
            df[feature_cols] = df[feature_cols].fillna(0.0)

    df_one_bin = df_one[df_one["label"].isin(LABEL_MAP_BINARY)].copy()
    y_train = df_one_bin["label"].map(LABEL_MAP_BINARY).values.astype(int)
    X_train_raw = df_one_bin[feature_cols].copy()

    p1_thresholds = X_train_raw.quantile(0.01)

    X_train_clip = X_train_raw.clip(lower=p1_thresholds, axis=1)
    X_zero_clip = df_zero[feature_cols].clip(lower=p1_thresholds, axis=1)
    X_real_clip = (
        df_real[feature_cols].clip(lower=p1_thresholds, axis=1)
        if not df_real.empty
        else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clip)
    X_zero_scaled = scaler.transform(X_zero_clip)
    X_real_scaled = scaler.transform(X_real_clip) if X_real_clip is not None else None

    print(f"Features: {len(feature_cols)} | Training instances: {len(y_train)}\n")
    return (
        feature_cols,
        X_train_scaled,
        y_train,
        X_zero_scaled,
        df_zero,
        X_real_scaled,
        df_real,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. REGULARIZED MODEL ARENA
# ──────────────────────────────────────────────────────────────────────────────


def train_regularized_models(X_train, y_train):
    print("=" * 50)
    print("2. MODEL TRAINING (Cross-Validation)")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=2, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
    }

    trained_models = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"{name:<25} CV Accuracy: {scores.mean():.3f} (± {scores.std():.3f})")
        model.fit(X_train, y_train)
        trained_models[name] = model

    print()
    return trained_models


def get_consensus_proba(models, X):
    p_rf = models["Random Forest"].predict_proba(X)[:, 1]
    p_gb = models["Gradient Boosting"].predict_proba(X)[:, 1]
    p_lr = models["Logistic Regression"].predict_proba(X)[:, 1]
    return (0.4 * p_rf) + (0.4 * p_gb) + (0.2 * p_lr)


# ──────────────────────────────────────────────────────────────────────────────
# 3. BIOLOGICAL INTERPRETABILITY & SHAP
# ──────────────────────────────────────────────────────────────────────────────


def interpret_biology(models, X_train, y_train, feature_cols, X_zero):
    print("=" * 50)
    print("3. BIOLOGICAL INTERPRETABILITY (Permutation & SHAP)")
    print("=" * 50)

    rf_model = models["Random Forest"]
    perm = permutation_importance(
        rf_model, X_train, y_train, n_repeats=30, random_state=RANDOM_STATE
    )
    perm_imp = perm.importances_mean

    print("Top 5 Biological Drivers (Global Importance):")
    order = np.argsort(perm_imp)[::-1]
    for i in order[:5]:
        feat = feature_cols[i]
        meaning = BIO_MEANING.get(feat, "General feature")
        print(f"  • {feat:<28} | {meaning}")

    print("\nGenerating SHAP Summary Plot...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_zero)
    sv_pos = (
        shap_values[1]
        if isinstance(shap_values, list)
        else (shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values)
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv_pos, X_zero, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "groupe_03_shap_summary.png", dpi=150)
    plt.close()
    print("  [Saved] -> analysis/groupe_03_shap_summary.png\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4. CONSENSUS RANKING
# ──────────────────────────────────────────────────────────────────────────────


def rank_patient_zero(models, X_zero, df_zero):
    print("=" * 50)
    print("4. FINAL RANKING (Ensemble Consensus)")
    print("=" * 50)

    proba = get_consensus_proba(models, X_zero)
    df_rank = df_zero[["candidate_id", "label"]].copy()
    df_rank["prob"] = proba
    df_rank = df_rank.sort_values("prob", ascending=False).reset_index(drop=True)

    for rank, row in df_rank.iterrows():
        marker = "  <-- TARGET" if row["candidate_id"] == "CAND_01" else ""
        print(f"  {rank + 1:>2}. {row['candidate_id']:<10} {row['prob']:.4f}  \
            {row['label']}{marker}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. REAL-WORLD BENCHMARKING
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_real_world(models, X_real, df_real):
    print("\n" + "=" * 50)
    print("5. REAL-WORLD BENCHMARKING (patient_real)")
    print("=" * 50)

    if X_real is None or not PATIENT_REAL_RAW_CSV.exists():
        print("Real-world data missing. Skipping evaluation.")
        return

    proba_real = get_consensus_proba(models, X_real)
    df_real["prob"] = proba_real

    # Classification: REAL vs DECOY
    real_mask = df_real["label"] == "REAL"
    decoy_mask = df_real["label"] == "DECOY"
    y_true = np.where(real_mask, 1, np.where(decoy_mask, 0, np.nan))
    valid = ~np.isnan(y_true)

    if valid.sum() > 0:
        auc = roc_auc_score(y_true[valid], proba_real[valid])
        print(f"REAL vs DECOY Classification AUC: {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_true[valid], proba_real[valid])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, color="#e74c3c", label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("Consensus: REAL vs DECOY ROC")
        plt.legend(loc="lower right")
        plt.savefig(ANALYSIS_DIR / "groupe_03_roc_curve.png", dpi=150)
        plt.close()
        print("  [Saved] -> analysis/groupe_03_roc_curve.png")

    # Correlation: Score vs IC50
    df_raw = pd.read_csv(PATIENT_REAL_RAW_CSV)
    df_raw_real = df_raw[(df_raw["label"] == "REAL") & df_raw["ic50_nm"].notna()]
    merged = df_real[real_mask].merge(
        df_raw_real[["candidate_id", "ic50_nm"]], on="candidate_id"
    )

    if len(merged) > 3:
        rho, p_val = stats.spearmanr(merged["prob"], merged["ic50_nm"])
        print(f"\nSpearman Correlation vs IC50: {rho:.3f} (p={p_val:.4f})")
        print("*(Note: Negative indicates High Model Prob = Low IC50 = Strong Binder)*")

        plt.figure(figsize=(7, 5))
        plt.scatter(merged["prob"], merged["ic50_nm"], alpha=0.7, edgecolors="k")
        plt.xlabel("Ensemble Consensus Score")
        plt.ylabel("Measured IC50 (nM)")
        plt.title(f"Score vs IC50 Binding Affinity (rho={rho:.3f})")
        plt.yscale("log")
        plt.savefig(ANALYSIS_DIR / "groupe_03_ic50_scatter.png", dpi=150)
        plt.close()
        print("  [Saved] -> analysis/groupe_03_ic50_scatter.png\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feat_col, X_train, y_train, X_zero, df_zero, X_real, df_real = load_and_preprocess()
    models = train_regularized_models(X_train, y_train)
    interpret_biology(models, X_train, y_train, feat_col, X_zero)
    rank_patient_zero(models, X_zero, df_zero)
    evaluate_real_world(models, X_real, df_real)
