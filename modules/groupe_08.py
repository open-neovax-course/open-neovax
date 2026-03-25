"""
Group 08 — Department A (A1)
Physicochemical property: Hydrophobicity
Using the Kyte-Doolittle scale.

Hypothesis:
Hydrophobic peptides may influence HLA interaction properties.
This module computes the mean Kyte-Doolittle hydrophobicity
of the mutated peptide.

Important:
This function must NEVER raise exceptions.
Invalid inputs return a neutral score (0.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.types import Candidate


# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

SCORE_NAME = "A1_hydrophobicity_kyte_doolittle"

# Kyte-Doolittle hydropathy scale
_KYTE_DOOLITTLE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


# ══════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """
    Compute mean hydrophobicity of the mutated peptide.

    Returns
    -------
    tuple[str, float]
        (score_name, mean_hydrophobicity)

    Rules
    -----
    - Must never raise exceptions.
    - Invalid peptide → return 0.0.
    - Unknown amino acids contribute 0.0.
    """

    try:
        peptide = getattr(candidate, "peptide_mut", None)
    except Exception:
        return (SCORE_NAME, 0.0)

    if not isinstance(peptide, str) or not peptide:
        return (SCORE_NAME, 0.0)

    peptide = peptide.strip().upper()

    if len(peptide) == 0:
        return (SCORE_NAME, 0.0)

    scores: list[float] = []

    for aa in peptide:
        scores.append(_KYTE_DOOLITTLE.get(aa, 0.0))

    if len(scores) == 0:
        return (SCORE_NAME, 0.0)

    mean_score = sum(scores) / len(scores)

    if not isinstance(mean_score, float):
        return (SCORE_NAME, 0.0)

    if mean_score != mean_score:  # NaN check
        return (SCORE_NAME, 0.0)

    return (SCORE_NAME, mean_score)
