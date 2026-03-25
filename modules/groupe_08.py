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
    Position-aware hydrophobicity score with mutation effect (delta model).

    Improvements:
    - Anchors (P2, P9): hydrophobic = good
    - Exposed (P4–P7): hydrophobic = bad
    - Now includes delta between WT and MUT
    """

    pep_mut = getattr(candidate, "peptide_mut", None)
    pep_wt = getattr(candidate, "peptide_wt", None)

    if not isinstance(pep_mut, str) or len(pep_mut) < 8:
        return (SCORE_NAME, 0.0)
    if not isinstance(pep_wt, str) or len(pep_wt) != len(pep_mut):
        pep_wt = pep_mut  # fallback to neutral comparison

    pep_mut = pep_mut.strip().upper()
    pep_wt = pep_wt.strip().upper()

    # --- anchors (P2, P9)
    def _anchor_score(seq: str) -> float:
        vals = [_KYTE_DOOLITTLE.get(seq[1], 0.0), _KYTE_DOOLITTLE.get(seq[-1], 0.0)]
        return sum(vals) / len(vals)

    # --- exposed (P4–P7)
    def _exposed_score(seq: str) -> float:
        idxs = list(range(3, min(7, len(seq))))
        if not idxs:
            return 0.0
        vals = [_KYTE_DOOLITTLE.get(seq[i], 0.0) for i in idxs]
        return sum(vals) / len(vals)

    # Compute hydrophobicity for WT and MUT
    anchor_mut = _anchor_score(pep_mut)
    anchor_wt = _anchor_score(pep_wt)
    exposed_mut = _exposed_score(pep_mut)
    exposed_wt = _exposed_score(pep_wt)

    # Weight anchors higher, exposed lower
    delta_anchor = 2.0 * (anchor_mut - anchor_wt)
    delta_exposed = 1.5 * (exposed_mut - exposed_wt)

    score = delta_anchor - delta_exposed

    # Safe fallback
    if not isinstance(score, float) or score != score:
        return (SCORE_NAME, 0.0)

    return (SCORE_NAME, score)
