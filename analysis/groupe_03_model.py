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

import pandas as pd
import argparse
import csv
import sys
from pathlib import Path
from scipy.stats import spearmanr

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

def load_data():
    """
    Load the data from the csv files and labelised it.
    return X as the data and y as the labels.
    """

    df = pd.read_csv("analysis/scores_patient_one.csv")
    # Encode labels
    label_map = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}

    # Remove MEDIOCRE (ambiguous) or assign it
    df = df[df["label"].isin(label_map.keys())]

    y = df["label"].map(label_map)
    X = df.drop(columns=["candidate_id", "label"])

    # Fill missing values with 0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    return X, y

def train_and_evaluate():
    """Placeholder — see docs/ml-guide.md for instructions."""

    print("Create your model in analysis/groupe_XX_model.py")
    print("See docs/ml-guide.md for the full guide.")


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
