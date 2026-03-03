"""Loading and validation of candidates from a CSV file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from logic.types import Candidate

REQUIRED_COLUMNS = {"candidate_id", "peptide_wt", "peptide_mut", "mut_pos_1based"}
OPTIONAL_COLUMNS = {"gene", "hla_allele", "note"}


def load_candidates(csv_path: str | Path) -> list[Candidate]:
    """Load a list of candidates from a CSV file.

    Raises:
        FileNotFoundError: if the file does not exist
        ValueError: if required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    _validate_columns(df)

    return [_row_to_candidate(row) for _, row in df.iterrows()]


def _validate_columns(df: pd.DataFrame) -> None:
    """Check that all required columns are present."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _row_to_candidate(row: pd.Series) -> Candidate:
    """Convert a DataFrame row into a Candidate object."""
    return Candidate(
        candidate_id=str(row["candidate_id"]),
        peptide_wt=str(row["peptide_wt"]),
        peptide_mut=str(row["peptide_mut"]),
        mut_pos_1based=int(row["mut_pos_1based"]),
        gene=str(row.get("gene", "") or ""),
        hla_allele=str(row.get("hla_allele", "") or ""),
        note=str(row.get("note", "") or ""),
    )
