"""
Groupe 07 - TAP transport scoring module(B2)

A peptide is more likely to be transported into the ER by TAP
if its C-terminal residue is hydrophobic and its overall net charge is moderate.
Score > 0: compatible with TAP transport.
Score < 0: disfavored by TAP.
Score -2.0: invalid peptide (wrong length or non-standard amino acids).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.types import Candidate


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SCORE_NAME = "B_tap_transport_score"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 8
MAX_LENGTH = 11

# Threshold beyond which the net charge is considered excessive.
CHARGE_THRESHOLD = 2

# Penalty applied when the peptide is invalid
INVALID_PENALTY = -2.0

# Relative weights of the two criteria in the final score
WEIGHT_CTERMINUS = 0.6
WEIGHT_CHARGE = 0.4

# TAP C-terminal preference scores, based on van Endert et al.(1994).
# Hydrophobic/aromatic residues are favored; charged/rigid are penalized.
CTERMINUS_SCORES: dict[str, float] = {
    # Strongly favored (hydrophobic / aromatic)
    "L": 1.0,
    "I": 1.0,
    "V": 0.8,
    "F": 1.0,
    "Y": 0.8,
    "M": 0.7,
    "W": 0.6,
    # Neutral
    "A": 0.2,
    "G": 0.0,
    "S": 0.0,
    "T": 0.0,
    "C": 0.1,
    "N": 0.0,
    "Q": 0.0,
    "H": -0.2,
    # Disfavored (charged or rigid)
    "P": -1.0,
    "D": -0.8,
    "E": -0.8,
    "K": -0.6,
    "R": -0.6,
}

# Net charge per amino acid
AA_CHARGE: dict[str, int] = {
    "K": +1,
    "R": +1,
    "D": -1,
    "E": -1,
    "H": 0,   # histidine treated as neutral
}


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════


def _is_valid_peptide(peptide: str) -> bool:
    """Check that the peptide can be processed by this module.
    A peptide is valid if:
    - it is not empty,
    - its length is between MIN_LENGTH and MAX_LENGTH,
    - it contains only standard amino acids.
    """
    if not peptide:
        return False
    if not (MIN_LENGTH <= len(peptide) <= MAX_LENGTH):
        return False
    if not set(peptide).issubset(VALID_AA):
        return False
    return True


def _score_cterminus(peptide: str) -> float:
    """Return the TAP preference score for the C-terminal residue."""
    last_aa = peptide[-1]
    return CTERMINUS_SCORES.get(last_aa, 0.0)


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