"""
C5 — Total HLA binding score (weighted PSSM) (Group 13)
========================================================

Computes a global HLA-A*02:01 binding estimate by summing the PSSM
(Position-Specific Scoring Matrix) log-odds scores across all 9
positions of the mutant peptide.

Anchor positions P2 and P9, which dominate peptide–MHC contact, are
weighted double (×2) to reflect their biological importance.

Hypothesis
----------
A high weighted sum means the peptide carries favorable residues at
every position (especially the anchors), making it a strong HLA binder
and therefore a good neo-epitope vaccine candidate.

Score name: C_total_binding
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from logic.types import Candidate

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SCORE_NAME = "C_total_binding"

# 20 standard amino acids
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Anchor positions (P2, P9) count double
WEIGHTS: dict[str, float] = {
    "P1": 1.0,
    "P2": 2.0,
    "P3": 1.0,
    "P4": 1.0,
    "P5": 1.0,
    "P6": 1.0,
    "P7": 1.0,
    "P8": 1.0,
    "P9": 2.0,
}

# ══════════════════════════════════════════════════════════════════════
#  PSSM LOADING (cached)
# ══════════════════════════════════════════════════════════════════════

_PSSM: pd.DataFrame | None = None
_PSSM_LOADED: bool = False


def _load_pssm() -> pd.DataFrame | None:
    """Load the PSSM matrix once and cache it globally."""
    global _PSSM, _PSSM_LOADED
    if _PSSM_LOADED:
        return _PSSM

    path = DATA_DIR / "hla_pssm_A0201.csv"
    try:
        df = pd.read_csv(path, index_col=0)
        _PSSM = df
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        _PSSM = None

    _PSSM_LOADED = True
    return _PSSM


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


def _compute_total_binding(peptide: str) -> float:
    """Return the weighted PSSM sum for a 9-mer peptide."""
    pssm = _load_pssm()
    if pssm is None:
        return 0.0

    peptide = peptide.upper()
    total = 0.0

    for i, aa in enumerate(peptide):
        col = f"P{i + 1}"
        weight = WEIGHTS.get(col, 1.0)
        if aa in pssm.index:
            total += weight * float(pssm.loc[aa, col])

    return total


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute the total weighted PSSM binding score.

    Parameters
    ----------
    candidate : Candidate
        Neo-epitope candidate with a ``peptide_mut`` attribute.

    Returns
    -------
    tuple[str, float]
        (``"C_total_binding"``, weighted sum of PSSM scores).
        Returns 0.0 for invalid peptides.
    """
    peptide = candidate.peptide_mut

    # Guard: must be a non-empty string
    if not isinstance(peptide, str) or not peptide:
        return (SCORE_NAME, 0.0)

    peptide = peptide.strip().upper()

    # Guard: must be a 9-mer
    if len(peptide) != 9:
        return (SCORE_NAME, 0.0)

    # Guard: all characters must be valid amino acids
    if not all(aa in _VALID_AA for aa in peptide):
        return (SCORE_NAME, 0.0)

    score = _compute_total_binding(peptide)
    return (SCORE_NAME, score)
