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
    """
    YOUR WORK HERE.

    You have two CSV files in analysis/:
        - scores_patient_one.csv  (75 candidates — training)
        - scores_patient_zero.csv (18 candidates — validation)

    Each row = one candidate. Columns:
        - candidate_id : identifier
        - label        : GOLD, GOOD, MEDIOCRE, BAD, or TRAP
        - remaining columns : one per scoring module (float values)

    Objectives:
        1. Train a model to classify candidates as good (GOLD/GOOD) vs bad (BAD/TRAP)
        2. Evaluate with cross-validation on patient_one
        3. Validate on patient_zero — does CAND_01 (GOLD) rank first?
        4. Extract feature importance — which modules matter most?

    Suggested approach:
        - Load the CSV with pandas
        - Convert labels to binary: GOLD/GOOD = 1, BAD/TRAP = 0 (remove MEDIOCRE)
        - Standardize features (StandardScaler)
        - Try at least 2 models (e.g. LogisticRegression, RandomForestClassifier)
        - Use cross_val_score for evaluation
        - Use feature_importances_ (Random Forest) to find top modules
        - Predict on patient_zero and print the final ranking

    Useful imports:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report
    """
    print("TODO: implement train_and_evaluate()")
    print()
    print("Start by loading the score matrices:")
    print("  analysis/scores_patient_one.csv  (training)")
    print("  analysis/scores_patient_zero.csv (validation)")
    print()
    print("See the docstring above for guidance.")


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
