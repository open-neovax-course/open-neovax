"""
Score Analysis & Global Scoring Model
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
#  STEP 2 — Train and evaluate models (YOUR WORK)
# ══════════════════════════════════════════════════════════════════════

# Label encoding: higher = better candidate
LABEL_MAP = {"GOLD": 4, "GOOD": 3, "MEDIOCRE": 2, "BAD": 1, "TRAP": 0, "NEUTRAL": 2}


def train_and_evaluate():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    # -------------------------
    # LOAD DATA
    # -------------------------
    train = pd.read_csv("analysis/scores_patient_one.csv")
    test = pd.read_csv("analysis/scores_patient_zero.csv")

    # -------------------------
    # LABEL ENCODING
    # -------------------------
    def encode(label):
        if label in ["GOLD", "GOOD"]:
            return 1
        elif label in ["BAD", "TRAP"]:
            return 0
        else:
            return np.nan

    train = train.dropna(subset=["label"])
    train["y"] = train["label"].apply(encode)
    train = train.dropna(subset=["y"])

    X_train = train.drop(columns=["candidate_id", "label", "y"])
    y_train = train["y"]

    X_test = test.drop(columns=["candidate_id", "label"])

    # -------------------------
    # SCALE FEATURES
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # MODELS
    # -------------------------
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    print("\n=== CROSS VALIDATION ===")

    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"{name}: {scores.mean():.3f}")

    # -------------------------
    # FIT BEST MODEL (RF)
    # -------------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importances.head(10))

    # -------------------------
    # PREDICT TEST SET
    # -------------------------
    test["score"] = rf.predict_proba(X_test_scaled)[:, 1]

    ranking = test.sort_values("score", ascending=False)

    print("\n=== FINAL RANKING (patient_zero) ===")
    print(ranking[["candidate_id", "score"]].head(18))


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

    if args.train:
        train_and_evaluate()
