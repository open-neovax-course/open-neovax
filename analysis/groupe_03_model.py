"""
Global Scoring Model — Open-NeoVax
=========================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


ANALYSIS_DIR = Path(__file__).resolve().parent
PATIENT_ONE_CSV = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_CSV = ANALYSIS_DIR / "scores_patient_zero.csv"

LABEL_MAP_BINARY = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
RANDOM_STATE = 42

BIO_MEANING = {
    "A_tcr_contact_potential": "TCR contact probability",
    "A_mutation_surprisal": "Mutation surprisal score",
    "A_net_charge": "Peptide net charge",
    "A_hydrophobicity_kd": "Kyte-Doolittle hydrophobicity",
    "A_delta_wt_vs_mut": "Physicochemical delta",
    "A_hybrid_complexity": "Sequence complexity",
    "B_sanity_check": "Sequence validity check",
    "B_proteasome_cterm": "C-term proteasomal cleavage",
    "B_erap_nterm_proxy": "N-term ERAP trimming",
    "B_tap_transport_score": "TAP transport efficiency",
    "C_total_binding": "Aggregated MHC binding",
    "C_anchoring_P2": "P2 anchor quality",
    "C_hla_anchor_p9": "P9/C-term anchor quality",
    "C_hla_delta_binding": "Delta binding (MUT vs WT)",
    "C_binding_quality": "Overall HLA binding",
    "D1_exact_self_similarity": "Human proteome self-similarity",
    "d1_exact_self_similarity": "Human proteome self-similarity",
    "D_mutation_in_window": "Mutation in TCR window",
    "D_wt_presented": "WT HLA presentation",
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def load_and_preprocess():
    print("=" * 50)
    print("1. DATA LOADING & ROBUST PREPROCESSING")
    print("=" * 50)
    
    df_one = pd.read_csv(PATIENT_ONE_CSV)
    df_zero = pd.read_csv(PATIENT_ZERO_CSV)

    feature_cols = [c for c in df_one.columns if c not in ("candidate_id", "label")]
    
    # Safely align columns and fill NaNs
    for df in [df_one, df_zero]:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        df[feature_cols] = df[feature_cols].fillna(0.0)

    # Filter targets (Drop MEDIOCRE)
    df_one_bin = df_one[df_one["label"].isin(LABEL_MAP_BINARY)].copy()
    y_train = df_one_bin["label"].map(LABEL_MAP_BINARY).values.astype(int)
    X_train_raw = df_one_bin[feature_cols].copy()

    # OUTLIER CLIPPING: Find 1st percentile threshold
    p1_thresholds = X_train_raw.quantile(0.01)
    
    # Clip training and validation data
    X_train_clip = X_train_raw.clip(lower=p1_thresholds, axis=1)
    X_zero_clip = df_zero[feature_cols].clip(lower=p1_thresholds, axis=1)

    # SCALING
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clip)
    X_zero_scaled = scaler.transform(X_zero_clip)

    print(f"Features: {len(feature_cols)} | Training instances: {len(y_train)}")
    print("Outliers clipped to 1st percentile to protect scaler.\n")
    
    return feature_cols, X_train_scaled, y_train, X_zero_scaled, df_zero

# ──────────────────────────────────────────────────────────────────────────────
# 2. BASELINE TRAINING & RANKING
# ──────────────────────────────────────────────────────────────────────────────

def train_baseline_and_rank(X_train, y_train, X_zero, df_zero):
    print("=" * 50)
    print("2. BASELINE MODEL RANKING (patient_zero)")
    print("=" * 50)
    
    # Basic baseline model
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    
    # Rank
    proba = rf.predict_proba(X_zero)[:, 1]
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
    train_baseline_and_rank(X_train, y_train, X_zero, df_zero)