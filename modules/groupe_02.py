"""
C2 — C-terminal HLA anchor (P9) (Group 02)
===========================================

Models the main C-terminal anchor of HLA-I binding by scoring the last
residue (P9 / PΩ) of the mutant peptide.

Uses the provided HLA-A*02:01 PSSM (hla_pssm_A0201.csv), taking the
value in column P9 for the C-terminal amino acid. Returns a neutral
score when the peptide is not a valid 9-mer, the residue is invalid,
or the PSSM cannot be used.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Import the Candidate type.
# TYPE_CHECKING is False at runtime → no circular import issues.
# This still lets your editor (VS Code, PyCharm) provide autocompletion.
# ──────────────────────────────────────────────────────────────────────
if TYPE_CHECKING:
    from logic.types import Candidate


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# Path to the data/ directory at the project root.
# Useful if your module needs to load an external file
# (PSSM matrix, self-peptide list, etc.).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Name of the score returned by this module.
# IMPORTANT: this name must be unique across all modules!
# Convention: <department>_<concept>[_detail]
SCORE_NAME = "C_hla_anchor_p9"

NEUTRAL_SCORE = 0.0
## valid amino acids that can be present in our peptide
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

## {amino acid : P9 score}
_PSSM_P9: dict[str, float] | None


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _load_pssm() -> dict[str, float] | None:
    """Loading pssm and extracting P9

    Returns dict with {amino_acid: P9 score} or Non
    """

    pssm_path = DATA_DIR / "hla_pssm_A0201.csv"

    try:
        df = pd.read_csv(pssm_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    # Checking if needed columns are present
    if "aa" not in df.columns or "P9" not in df.columns:
        return None

    p9_scores: dict[str, float] = {}

    for _, row in df.iterrows():
        aa = str(row["aa"]).strip().upper()

        ## Check if amino acids have expected format
        if len(aa) != 1 or not aa.isalpha():
            continue

        try:
            value = float(row["P9"])
        except (TypeError, ValueError):
            continue

        ## Check if the value is finite(no inf or nan)
        if not math.isfinite(value):
            continue

        p9_scores[aa] = value

    return p9_scores or None


## Loaded values of P9 for each amino acid
_PSSM_P9 = _load_pssm()


def _valid_peptide(peptide: str) -> bool:
    """Return true is peptide is valid for our module
    It is a str, length is 9, last amino acid is valid
    """
    if not isinstance(peptide, str):
        return False

    peptide = peptide.strip().upper()

    if len(peptide) < 8 or len(peptide) > 11:
        return False

    ## last aa is valid
    return peptide[-1] in _VALID_AA


def _compute_P9_score(peptide: str) -> float:
    """Example internal function.

    Replace this computation with your biological logic.
    """
    if not _valid_peptide(peptide):
        return NEUTRAL_SCORE

    if _PSSM_P9 is None:
        return NEUTRAL_SCORE

    aa = peptide.strip().upper()[-1]

    return float(_PSSM_P9.get(aa, NEUTRAL_SCORE))


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute the score for a given candidate.

    This is THE ONLY FUNCTION that the pipeline calls.
    It must ALWAYS return a tuple (str, float):
      - str  : the unique name of your score
      - float : the computed value (not NaN, not inf)

    Parameters
    ----------
    candidate : Candidate
        Object containing neo-epitope information:
        - candidate.peptide_mut  (str)  : mutated sequence
        - candidate.peptide_wt   (str)  : wild-type sequence
        - candidate.mut_pos_1based (int): mutation position
        - candidate.gene         (str)  : source gene
        - candidate.hla_allele   (str)  : target HLA allele

    Returns
    -------
    tuple[str, float]
        (score_name, score_value)
    """
    # 1. Get the sequence to analyze
    peptide = candidate.peptide_mut

    # 2. Compute the score using your logic
    score_value = _compute_P9_score(peptide)

    # 3. Return the result in the expected format
    return (SCORE_NAME, score_value)
