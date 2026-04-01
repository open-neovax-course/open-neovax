"""
D2 — Approximate self-peptide match (Group 02)
===============================================

Models how dissimilar a mutant peptide is from a reference “self” peptide
by counting residue mismatches along the sequence (Hamming distance).

Intended to capture approximate self-reactivity: peptides that are
farther from self receive higher scores (less self-like), while
peptides very close to self receive lower scores (more self-like).
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
SCORE_NAME = "D_self_approx_match"
_SELF_PEPTIDES = None

# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _load_self_peptides():
    global _SELF_PEPTIDES
    if _SELF_PEPTIDES is not None:
        return _SELF_PEPTIDES
    path = DATA_DIR / "human_peptides_small.txt"
    if not path.exists():
        _SELF_PEPTIDES = set()
        return _SELF_PEPTIDES
    with open(path) as f:
        _SELF_PEPTIDES = set(line.strip() for line in f)
    return _SELF_PEPTIDES


def _hamming(a, b) -> int:
    """Compute the number of differences in sequences."""
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def _D_distance_score(candidate: "Candidate") -> tuple[str, float]:
    """Example internal function.

    Replace this computation with your biological logic.
    """
    pep = candidate.peptide_mut
    if not pep:
        return (SCORE_NAME, 0.0)
    pep = pep.strip().upper()
    corpus = _load_self_peptides()
    same_len = [p for p in corpus if len(p) == len(pep)]
    if not same_len:
        return (SCORE_NAME, 0.0)
    min_dist = min(_hamming(pep, p) for p in same_len)
    if min_dist == 0:
        score = -10
    elif min_dist == 1:
        score = -5
    elif min_dist == 2:
        score = -3
    else:
        score = -1 / min_dist

    return (SCORE_NAME, float(score))


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
    return _D_distance_score(candidate)
