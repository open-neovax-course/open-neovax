"""
Global Scoring Model — Open-NeoVax
=========================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


ANALYSIS_DIR = Path(__file__).resolve().parent
PATIENT_ONE_CSV = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_CSV = ANALYSIS_DIR / "scores_patient_zero.csv"

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

    feature_cols = [c for c in df_one.columns if c not in ("candidate_id", "label")]
    
    for df in [df_one, df_zero]:
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clip)
    X_zero_scaled = scaler.transform(X_zero_clip)

    print(f"Features: {len(feature_cols)} | Training instances: {len(y_train)}")
    print("Outliers clipped to 1st percentile to protect scaler.\n")
    
    return feature_cols, X_train_scaled, y_train, X_zero_scaled, df_zero

# ──────────────────────────────────────────────────────────────────────────────
# 2. REGULARIZED MODEL ARENA
# ──────────────────────────────────────────────────────────────────────────────

def train_regularized_models(X_train, y_train):
    print("=" * 50)
    print("2. MODEL TRAINING (Cross-Validation)")
    print("=" * 50)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Anti-Overfitting applied to all models
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE
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

    print("\n[INFO] Models are regularized to prevent memorization.\n")
    return trained_models

def get_consensus_proba(models, X):
    """Combines RF, GB, and LR probabilities into a stable ensemble score."""
    p_rf = models["Random Forest"].predict_proba(X)[:, 1]
    p_gb = models["Gradient Boosting"].predict_proba(X)[:, 1]
    p_lr = models["Logistic Regression"].predict_proba(X)[:, 1]
    
    # Ensemble weighting: 40% RF, 40% GB, 20% LR
    return (0.4 * p_rf) + (0.4 * p_gb) + (0.2 * p_lr)

# ──────────────────────────────────────────────────────────────────────────────
# 3. CONSENSUS RANKING
# ──────────────────────────────────────────────────────────────────────────────

def rank_patient_zero(models, X_zero, df_zero):
    print("=" * 50)
    print("3. FINAL RANKING (Ensemble Consensus)")
    print("=" * 50)
    
    proba = get_consensus_proba(models, X_zero)
    df_rank = df_zero[["candidate_id", "label"]].copy()
    df_rank["prob"] = proba
    df_rank = df_rank.sort_values("prob", ascending=False).reset_index(drop=True)
    
    for rank, row in df_rank.iterrows():
        marker = "  <-- TARGET" if row["candidate_id"] == "CAND_01" else ""
        print(f"  {rank + 1:>2}. {row['candidate_id']:<10} {row['prob']:.4f}  {row['label']}{marker}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    feat_cols, X_train, y_train, X_zero, df_zero = load_and_preprocess()
    models = train_regularized_models(X_train, y_train)
    rank_patient_zero(models, X_zero, df_zero)