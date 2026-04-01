"""
A8 — TCR contact potential (Group 11)
=====================================

Scores how visible the mutated peptide may be to the T cell receptor.
It looks at positions P4-P7 of the mutated peptide.

Higher scores mean more visible residues at these positions.
Lower scores mean less visible residues at these positions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
SCORE_NAME = "A_tcr_contact_potential"

TCR_SALIENCY = {
    "W": 1.0,
    "F": 0.9,
    "Y": 0.9,
    "K": 0.8,
    "R": 0.8,
    "D": 0.7,
    "E": 0.7,
    "H": 0.6,
    "N": 0.5,
    "Q": 0.5,
    "M": 0.4,
    "L": 0.3,
    "I": 0.3,
    "V": 0.2,
    "T": 0.2,
    "S": 0.2,
    "C": 0.2,
    "P": 0.1,
    "A": 0.1,
    "G": 0.0,
}

TCR_POSITIONS = [3, 4, 5, 6]


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _tcr_contact_score(peptide: str) -> float:
    """Compute average saliency at TCR-exposed positions P4-P7."""
    if not peptide or len(peptide) < 8:
        return 0.0

    peptide = peptide.upper()
    total = sum(TCR_SALIENCY.get(peptide[i], 0.0) for i in TCR_POSITIONS)
    return total / len(TCR_POSITIONS)


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
    score_value = _tcr_contact_score(peptide)

    # 3. Return the result in the expected format
    return (SCORE_NAME, score_value)
