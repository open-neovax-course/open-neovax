"""
Template module — Open-NeoVax
=============================

This file is a TEMPLATE for student modules.
It shows the exact structure your module must follow.

You can copy and rename it to start your own module.
For example: cp template_module.py groupe_01.py

THIS FILE IS NOT EXECUTED by the pipeline (it is ignored by the orchestrator).
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
SCORE_NAME = "D_mutation_in_window"

# Scoring policy: neutral value for valid candidates, strong penalty for
# inconsistent mutation data.
NEUTRAL_SCORE = 0.0
INVALID_PENALTY = -1000.0


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _validate_mutation_in_window(
    peptide_wt: str, peptide_mut: str, mut_pos_1based: int
) -> bool:
    """Return True when WT/MUT sequences and declared position are consistent."""

    # WT must be a non-empty string.
    if not isinstance(peptide_wt, str) or not peptide_wt:
        return False

    # MUT must be a non-empty string.
    if not isinstance(peptide_mut, str) or not peptide_mut:
        return False

    # WT and MUT must describe the same peptide window length.
    if len(peptide_wt) != len(peptide_mut):
        return False

    # Mutation position must be an integer.
    if not isinstance(mut_pos_1based, int):
        return False

    # Position is 1-based and must fall inside the peptide.
    if mut_pos_1based < 1 or mut_pos_1based > len(peptide_mut):
        return False

    # If WT and MUT are identical, there is no mutation.
    if peptide_wt == peptide_mut:
        return False

    # Declared mutation position must contain a residue change.
    index = mut_pos_1based - 1
    return peptide_wt[index] != peptide_mut[index]


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
    # 1. Compute the score using your logic
    is_valid = _validate_mutation_in_window(
        candidate.peptide_wt, candidate.peptide_mut, candidate.mut_pos_1based
    )
    score_value = NEUTRAL_SCORE if is_valid else INVALID_PENALTY

    # 2. Return the result in the expected format
    return (SCORE_NAME, score_value)
