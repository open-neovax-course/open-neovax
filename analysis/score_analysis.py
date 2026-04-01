from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from logic.types import Candidate
from modules.groupe_06 import get_score


def load_candidates(csv_path: Path) -> pd.DataFrame:
    """Load candidate table from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def row_to_candidate(row: pd.Series) -> Candidate:
    """Convert one CSV row into a Candidate object."""
    return Candidate(
        candidate_id=row.get("candidate_id", ""),
        peptide_wt=row.get("peptide_wt", ""),
        peptide_mut=row.get("peptide_mut", ""),
        mut_pos_1based=row.get("mut_pos_1based", 0),
    )


def infer_label_from_note(note: object) -> str:
    """
    Extract a coarse label from the 'note' column if available.
    Examples:
    - 'GOLD — ...' -> 'GOLD'
    - 'GOOD — ...' -> 'GOOD'
    - 'BAD — ...'  -> 'BAD'
    - 'TRAP — ...' -> 'TRAP'
    """
    if note is None:
        return ""
    text = str(note).strip().upper()

    for label in ["GOLD", "GOOD", "MEDIOCRE", "BAD", "TRAP", "REAL", "DECOY"]:
        if text.startswith(label):
            return label
    return ""


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Run the group module on every candidate row."""
    rows = []

    for _, row in df.iterrows():
        candidate = row_to_candidate(row)
        score_name, score_value = get_score(candidate)

        result = dict(row)

        # Add label if possible
        if "label" not in result:
            result["label"] = infer_label_from_note(result.get("note", ""))

        # Add module score
        result[score_name] = score_value
        rows.append(result)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a simple score matrix for Open-NeoVax candidates"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/patient_zero.csv",
        help="Input candidate CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="analysis/score_matrix_patient_zero.csv",
        help="Output score matrix CSV",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate score matrix",
    )
    args = parser.parse_args()

    if not args.generate:
        print("Nothing to do. Use --generate to build the score matrix.")
        return

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    df = load_candidates(input_path)
    scored_df = compute_scores(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)

    print(f"Loaded {len(df)} candidates from: {input_path}")
    print(f"Saved score matrix to: {output_path}")
    print("Columns:")
    for col in scored_df.columns:
        print(f" - {col}")


if __name__ == "__main__":
    main()
