"""Group 12 module — D1 exact self-similarity.

This module penalizes candidates whose mutated peptide
exactly matches a known human peptide (self similarity).
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
HUMAN_PEPTIDES_PATH = DATA_DIR / "human_peptides_small.txt"

# Name of the score returned by this module.
# IMPORTANT: this name must be unique across all modules!
# Convention: <department>_<concept>[_detail]
SCORE_NAME = "D1_exact_self_similarity"
PENALTY = -1000.0


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _load_human_peptides(path: Path) -> set[str]:
    """Load the human peptide corpus from a text file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        return set()


HUMAN_PEPTIDES = _load_human_peptides(HUMAN_PEPTIDES_PATH)


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute D1 exact self-similarity score.

    Returns a strong negative penalty if the mutated peptide
    exactly matches a known human peptide.
    """
    peptide = candidate.peptide_mut.upper()

    # Safety: empty peptide → neutral score
    if not peptide:
        return (SCORE_NAME, 0.0)

    # Exact match with human peptide corpus → strong penalty
    if peptide in HUMAN_PEPTIDES:
        return (SCORE_NAME, PENALTY)

    # Otherwise → no penalty
    return (SCORE_NAME, 0.0)
