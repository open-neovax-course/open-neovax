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


def _compute_something(peptide: str) -> float:
    """Example internal function.

    Replace this computation with your biological logic.
    """
    # A constant score for demonstration purposes.
    # Your real module will do something useful here!
    _ = peptide  # avoid "unused parameter" warning
    return 0.0


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute the delta WT vs MUT physicochemical score

    Parameters
    ----------
    candidate : Candidate
        - candidate.peptide_mut : mutated sequence
        - candidate.peptide_wt  : wild-type sequence

    Returns
    -------
    tuple[str, float]
        ("A_delta_wt_vs_mut", score)
        score = fraction of non-conservative mutations in [0, 1]
    """
    try:
        mut = candidate.peptide_mut.strip().upper()
        wt = candidate.peptide_wt.strip().upper()

        if not mut or not wt or len(mut) != len(wt):
            return (SCORE_NAME, -1.0)
        if mut == wt:
            return (SCORE_NAME, 0.0)

        score = sum(
            AA_GROUPS.get(a, "?") != AA_GROUPS.get(b, "?") for a, b in zip(wt, mut)
        ) / len(mut)

        return (SCORE_NAME, score)

    except Exception:
        return (SCORE_NAME, -1.0)
