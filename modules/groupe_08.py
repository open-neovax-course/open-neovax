"""
Group 08 — Department A (A1)
Physicochemical property: Hydrophobicity
Using the Kyte-Doolittle scale.
"""

from __future__ import annotations

from logic.types import Candidate

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

SCORE_NAME = "A1_hydrophobicity_kyte_doolittle"

# Kyte-Doolittle hydropathy scale
# Source concept: Kyte & Doolittle hydropathy index
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
# PUBLIC FUNCTION (mandatory entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: Candidate) -> tuple[str, float]:
    """
    Compute mean hydrophobicity of the mutated peptide.

    Hypothesis:
    Hydrophobic peptides may show improved interaction with HLA binding
    pockets and increased stability.

    Returns
    -------
    tuple[str, float]
        (score_name, mean_hydrophobicity)

    Raises
    ------
    ValueError
        If peptide is invalid or contains unknown amino acids.
    """

    peptide = candidate.peptide_mut

    if not isinstance(peptide, str) or not peptide:
        raise ValueError("Invalid peptide sequence.")

    peptide = peptide.strip().upper()

    scores: list[float] = []

    for aa in peptide:
        if aa not in _KYTE_DOOLITTLE:
            raise ValueError(f"Unknown amino acid: {aa}")
        scores.append(_KYTE_DOOLITTLE[aa])

    mean_score = sum(scores) / len(scores)

    return SCORE_NAME, mean_score
