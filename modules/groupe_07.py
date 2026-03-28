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
SCORE_NAME = "B_TAP_score"

#valide acides animés :
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Acides aminés hydrophobes (favorisés par TAP)
HYDROPHOBIC_RESIDUES = set("AILMFWVY")

# Acides aminés chargés (défavorisés par TAP)
CHARGED_AA =set("DEKRH")
POSITIVE_RESIDUES = set("KRH")
NEGATIVE_RESIDUES = set("DE")

ALLOWED_LENGTHS = range(8, 12)


# Préférences pour le C-terminus 
# Plus le score est élevé, plus l'acide aminé est favorisé
C_TERMINAL_BONUS = {
    "L": 1.0,
    "I": 0.9,
    "V": 0.9,
    "F": 0.8,
    "Y": 0.7,
    "M": 0.7,
    "A": 0.4,
    "T": 0.2,
    "S": 0.1,
    "W": 0.5,
}

EMPTY_OR_INVALID_PENALTY = -2.0
OUT_OF_RANGE_LENGTH_PENALTY = -1.5

HYDROPHOBIC_TARGET_RATIO = 0.40
HYDROPHOBIC_TOLERANCE = 0.25
MAX_ABS_CHARGE_RATIO = 0.35
EXCESS_CHARGED_RATIO = 0.45



# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.

def _is_valid_peptide(peptide: str) -> bool:
    """Renvoie True si le peptide ne contient que des acides aminés standard."""
    return bool(peptide) and all(residue in VALID_AMINO_ACIDS for residue in peptide)


def _cterm_score(peptide: str) -> float:
    """Le C-terminus, un facteur clé de la préférence de transport TAP."""
    return C_TERMINAL_BONUS.get(peptide[-1], -0.6)


def _hydrophobicity_score(peptide: str) -> float:
    """privilégier une proportion équilibrée de résidus hydrophobes."""
    hydrophobic_count = sum(1 for aa in peptide if aa in HYDROPHOBIC_RESIDUES)
    ratio = hydrophobic_count / len(peptide)
    deviation = abs(ratio - HYDROPHOBIC_TARGET_RATIO)

    if deviation <= HYDROPHOBIC_TOLERANCE:
        return 0.8 - (deviation / HYDROPHOBIC_TOLERANCE) * 0.8
    return -0.8


def _charge_score(peptide: str) -> float:
    """Penalize extreme net charge or too many charged residues."""
    positive_count = sum(1 for aa in peptide if aa in POSITIVE_RESIDUES)
    negative_count = sum(1 for aa in peptide if aa in NEGATIVE_RESIDUES)
    charged_count = positive_count + negative_count

    net_charge_ratio = (positive_count - negative_count) / len(peptide)
    charged_ratio = charged_count / len(peptide)

    score = 0.4
    if abs(net_charge_ratio) > MAX_ABS_CHARGE_RATIO:
        score -= 0.7
    if charged_ratio > EXCESS_CHARGED_RATIO:
        score -= 0.7
    return score


def _compute_tap_score(peptide: str) -> float:
    """Compute the TAP transport proxy score for a valid peptide."""
    return _cterm_score(peptide) + _hydrophobicity_score(peptide) + _charge_score(peptide)
#    # Normalisation entre -1 et 1 (ajuste les bornes selon tes tests)
#     # Si nos scores vont de -2.5 à +2.5 par exemple :
#     normalized = max(-1.0, min(1.0, raw_score / 2.5))
#     return normalized


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Return the TAP transport proxy score for a candidate.

    Parameters
    ----------
    candidate:
        Candidate object containing at least `peptide_mut`.

    Returns
    -------
    tuple[str, float]
        The score name and a finite float score.

    Error policy
    ------------
    - Empty or invalid peptide: explicit penalty.
    - Length outside 8-11 aa: explicit penalty.
    - Unexpected attribute issues: explicit penalty.
    """
    try:
        peptide = str(candidate.peptide_mut).strip().upper()
    except (AttributeError, TypeError, ValueError):
        return (SCORE_NAME, EMPTY_OR_INVALID_PENALTY)

    if not _is_valid_peptide(peptide):
        return (SCORE_NAME, EMPTY_OR_INVALID_PENALTY)

    if len(peptide) not in ALLOWED_LENGTHS:
        return (SCORE_NAME, OUT_OF_RANGE_LENGTH_PENALTY)

    return (SCORE_NAME, float(_compute_tap_score(peptide)))








# def get_score(candidate: "Candidate") -> tuple[str, float]:
#     """Compute the score for a given candidate.

#     This is THE ONLY FUNCTION that the pipeline calls.
#     It must ALWAYS return a tuple (str, float):
#       - str  : the unique name of your score
#       - float : the computed value (not NaN, not inf)

#     Parameters
#     ----------
#     candidate : Candidate
#         Object containing neo-epitope information:
#         - candidate.peptide_mut  (str)  : mutated sequence
#         - candidate.peptide_wt   (str)  : wild-type sequence
#         - candidate.mut_pos_1based (int): mutation position
#         - candidate.gene         (str)  : source gene
#         - candidate.hla_allele   (str)  : target HLA allele

#     Returns
#     -------
#     tuple[str, float]
#         (score_name, score_value)
#     """
#     # 1. Get the sequence to analyze
#     peptide = candidate.peptide_mut

#     # 2. Compute the score using your logic
#     score_value = _compute_something(peptide)

#     # 3. Return the result in the expected format
#     return (SCORE_NAME, score_value)