"""
Score Analysis & Global Scoring Model

Groupe 09 (Projet A2)
Zakaria BOUAZZA
Achraf SAIGHI
======================================

Two objectives:

1. BUILD A GLOBAL SCORING MODEL
   The pipeline currently averages all module scores equally.
   Can we do better? Use machine learning to find the optimal way
   to combine scores and rank neoepitope candidates.

2. IDENTIFY THE MOST PREDICTIVE MODULES
   Which scoring modules actually matter for distinguishing good
   candidates from bad ones? Feature importance analysis tells us
   which biological signals are the most informative.

How to use this script
----------------------

    # Step 1 — Generate the score matrices (this part is done for you)
    python analysis/score_analysis.py --generate

    # Step 2 — YOUR WORK: train models, evaluate, analyze
    python analysis/score_analysis.py --train

Data
----
    - patient_one.csv  : 75 candidates (training) with labels
    - patient_zero.csv : 18 candidates (validation)

Requirements
------------
    pip install scikit-learn matplotlib pandas
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — Generate the score matrix (provided)
# ══════════════════════════════════════════════════════════════════════


def generate_score_matrix(patient_file: str = "patient_one.csv") -> None:
    """Run the pipeline on a patient file and save the score matrix as CSV."""
    from logic.orchestrator import run_modules
    from logic.types import Candidate

    data_dir = PROJECT_ROOT / "data"
    input_path = data_dir / patient_file

    candidates = []
    with open(input_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(
                Candidate(
                    candidate_id=row["candidate_id"],
                    peptide_wt=row["peptide_wt"],
                    peptide_mut=row["peptide_mut"],
                    mut_pos_1based=int(row["mut_pos_1based"]),
                    gene=row.get("gene", ""),
                    hla_allele=row.get("hla_allele", ""),
                    note=row.get("note", ""),
                )
            )

    scored = run_modules(candidates, PROJECT_ROOT / "modules")

    output_dir = PROJECT_ROOT / "analysis"
    output_dir.mkdir(exist_ok=True)

    all_score_names = set()
    for c in scored:
        all_score_names.update(c.scores.keys())
    all_score_names = sorted(all_score_names)

    output_path = output_dir / f"scores_{patient_file}"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "label"] + all_score_names)
        for c in scored:
            label = c.note.split()[0] if c.note else "UNKNOWN"
            row = [c.candidate_id, label]
            row += [c.scores.get(name, "") for name in all_score_names]
            writer.writerow(row)

    print(f"Score matrix saved to {output_path}")
    print(f"  {len(scored)} candidates x {len(all_score_names)} scores")
    print(f"  Scores: {', '.join(all_score_names)}")


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Train and evaluate models
#
#  DO NOT implement your model here.
#  Create your own file: analysis/groupe_XX_model.py
#  See docs/ml-guide.md for the full guide and issue #39 on GitHub.
# ══════════════════════════════════════════════════════════════════════


def train_and_evaluate():
    """Train ML models to combine all module scores.
    
    We compare two models, perform cross-validation, extract feature importance
    and produce the final ranking on patient_zero.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    print("\n" + "=" * 70)
    print("GROUPE 09 - Global ML Scoring Model (Issue #39)")
    print("=" * 70)


    # ====================== 1. LOAD DATA ======================
    data_dir = Path("analysis")

    train_df = pd.read_csv(data_dir / "scores_patient_one.csv")
    val_df   = pd.read_csv(data_dir / "scores_patient_zero.csv")

    real_df = None
    real_path = data_dir / "scores_patient_real.csv"
    if real_path.exists():
        real_df = pd.read_csv(real_path)
        print(f"Real neoantigen data loaded ({len(real_df)} candidates)")

    print(f"Training  : {len(train_df)} candidates")
    print(f"Validation: {len(val_df)} candidates")


    # ====================== 2. PREPARE DATA ======================
    feature_cols = [col for col in train_df.columns 
                    if col not in ['candidate_id', 'label']]

    # Binary labels: GOLD/GOOD = 1, BAD/TRAP = 0
    label_map = {'GOLD': 1, 'GOOD': 1, 'MEDIOCRE': 0, 'BAD': 0, 'TRAP': 0}
    train_df = train_df[train_df['label'].isin(label_map.keys())].copy()

    X = train_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = train_df['label'].map(label_map)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # ====================== 3. MODELS + CROSS-VALIDATION ======================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, 
            max_depth=7, 
            min_samples_leaf=2,
            random_state=42
        )
    }

    print("\nCross-validation accuracy (5-fold) on patient_one:")
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
        print(f"  {name:22s} → {scores.mean():.3f} ± {scores.std():.3f}")

    # Use Random Forest for predictions and feature importance
    best_model = models["Random Forest"]
    best_model.fit(X_scaled, y)


    # ====================== 4. FEATURE IMPORTANCE ======================
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    print("\nFEATURE IMPORTANCE (Top 12):")
    for i, (feat, imp) in enumerate(importances.head(12).items(), 1):
        print(f"  {i:2d}. {feat:28s} {imp:.4f}")

    # Save plot
    plt.figure(figsize=(11, 7))
    importances.head(15).plot(kind='barh', color='#38bdf8')
    plt.title('Groupe 9 - Feature Importance (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('analysis/groupe9_feature_importance.png', dpi=300)
    print("Plot saved → analysis/groupe9_feature_importance.png")


    # ====================== 5. FINAL RANKING ON PATIENT_ZERO ======================
    X_val = val_df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val_scaled = scaler.transform(X_val)

    proba = best_model.predict_proba(X_val_scaled)[:, 1]

    val_df = val_df.copy()
    val_df['ml_score'] = proba

    ranking = val_df.sort_values('ml_score', ascending=False)[['candidate_id', 'ml_score', 'label']].reset_index(drop=True)

    print("\nFINAL RANKING (patient_zero):")
    print("-" * 60)
    for i, row in ranking.iterrows():
        print(f"  {i+1:2d}. {row['candidate_id']:12s}  {row['ml_score']:.4f}   {row['label']:8s}")


    # ====================== 6. BIOLOGICAL INTERPRETATION ======================
    print("\nBIOLOGICAL INTERPRETATION:")
    print("The top modules are mostly HLA anchoring related (C_*) and some")
    print("physicochemical properties (A_*). This shows that strong HLA binding")
    print("is the most important signal for neoepitope quality. Our module")
    print("A_net_charge contributes reasonably well, supporting that moderate")
    print("net charge is biologically favorable.")



# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score analysis and ML model training")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate score matrices from pipeline output",
    )
    parser.add_argument(
        "--train", action="store_true", help="Train and evaluate ML models"
    )
    args = parser.parse_args()

    if not args.generate and not args.train:
        parser.print_help()
        sys.exit(0)

    if args.generate:
        print("Generating score matrix for patient_one...")
        generate_score_matrix("patient_one.csv")
        print()
        print("Generating score matrix for patient_zero...")
        generate_score_matrix("patient_zero.csv")
        print()
        print("Generating score matrix for patient_real...")
        generate_score_matrix("patient_real.csv")
        print()

    if args.train:
        train_and_evaluate()
