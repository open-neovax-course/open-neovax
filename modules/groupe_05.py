"""
GROUPE 05 — Open-NeoVax
=============================

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
SCORE_NAME = "A_delta_wt_vs_mut"

AA_GROUPS: dict[str, str] = {
    "A": "hydrophobic",
    "V": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "F": "hydrophobic",
    "W": "hydrophobic",
    "P": "hydrophobic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "Y": "polar",
    "C": "polar",
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "D": "negative",
    "E": "negative",
    "G": "special",
}
# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.

GROUP_DISTANCE = {
    ("hydrophobic", "hydrophobic"): 0.0,
    ("hydrophobic", "polar"): -0.3,
    ("hydrophobic", "positive"): -0.6,
    ("hydrophobic", "negative"): -0.8,
    ("hydrophobic", "special"): -0.4,
    ("polar", "polar"): 0.0,
    ("polar", "positive"): -0.3,
    ("polar", "negative"): -0.5,
    ("polar", "special"): -0.2,
    ("positive", "positive"): 0.0,
    ("positive", "negative"): -1.0,
    ("positive", "special"): -0.5,
    ("negative", "negative"): 0.0,
    ("negative", "special"): -0.5,
    ("special", "special"): 0.0,
}


def _group_distance(g1: str, g2: str) -> float:
    if (g1, g2) in GROUP_DISTANCE:
        return GROUP_DISTANCE[(g1, g2)]
    if (g2, g1) in GROUP_DISTANCE:
        return GROUP_DISTANCE[(g2, g1)]
    return -0.5


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute the delta WT vs MUT physicochemical score
    A4 Delta WT vs MUT basé sur une matrice physico-chimique simplifiée.
    Returns

    Parameters
    ----------
    candidate : Candidate
        - candidate.peptide_mut : mutated sequence
        - candidate.peptide_wt  : wild-type sequence
    Returns
    -------
    tuple[str, float]
        ("A_delta_wt_vs_mut", score) where:
    Score proche de 0 : mutation conservative
    Score négatif   : mutation non-conservative (impact forte)
    """
    try:
        wt = candidate.peptide_wt.strip().upper()
        mut = candidate.peptide_mut.strip().upper()

        if not wt or not mut or len(wt) != len(mut):
            return (SCORE_NAME, -1.0)

        if wt == mut:
            return (SCORE_NAME, 0.0)

        distances = []
        for a, b in zip(wt, mut):
            g1 = AA_GROUPS.get(a, "special")
            g2 = AA_GROUPS.get(b, "special")
            distances.append(_group_distance(g1, g2))

        score = sum(distances) / len(distances)
        return (SCORE_NAME, score)

    except Exception:
        return (SCORE_NAME, -1.0)
