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


def _score_charge(peptide: str) -> float:
    """Return a score based on the net charge of the peptide.

    - Moderate charge (|q| <= CHARGE_THRESHOLD) -> neutral score (0.0).
    - Excessive charge -> progressive penalty proportional to the excess.

    The penalty is capped at -1.0 to remain on a consistent scale
    with the C-terminal score.
    """
    net_charge = sum(AA_CHARGE.get(aa, 0) for aa in peptide)
    excess = abs(net_charge) - CHARGE_THRESHOLD
    if excess <= 0:
        return 0.0
    # Progressive penalty, capped at -1.0
    penalty = -min(excess * 0.2, 1.0)
    return penalty


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute the TAP transport proxy score for a neo-epitope candidate.

    Only candidate.peptide_mut is used. Returns INVALID_PENALTY for
    malformed inputs without raising an exception.
    """

    peptide = candidate.peptide_mut

    # Invalid input: return explicit penalty without raising an exception
    if not _is_valid_peptide(peptide):
        return (SCORE_NAME, INVALID_PENALTY)

    # Compute both components
    cterm_score = _score_cterminus(peptide)
    charge_score = _score_charge(peptide)

    # Weighted combination
    final_score = WEIGHT_CTERMINUS * cterm_score + WEIGHT_CHARGE * charge_score

    return (SCORE_NAME, float(final_score))
