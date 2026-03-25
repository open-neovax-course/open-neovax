"""
Groupe 09
Template module — Open-NeoVax

Hypothesis:
Peptides with a moderate overall charge are favored.
Highly positive or highly negative peptides are penalized.

This is a qualitative proxy, not an exact physicochemical measurement.
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
SCORE_NAME = "A_net_charge"

# ──────────────── Charge table ──────────────
# K,R,H are considered positively charged
# D,E are negatively charged
# All others are 0
AA_CHARGE = {
    "K": 1,
    "R": 1,
    "H": 1,  # can be considered weakly positive
    "D": -1,
    "E": -1
}


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
def _compute_net_charge(peptide: str) -> float:
    """Compute a simple net charge proxy for the peptide."""
    if not peptide:
        return 0.0
    net = sum(AA_CHARGE.get(aa.upper(), 0) for aa in peptide)
    # normalize by peptide length to penalize extremes in longer peptides
    return net / len(peptide)

def _compute_score(net_charge: float) -> float:
    """
    Penalize extreme charges:
    - net_charge near 0 → score close to 0
    - high positive or negative net_charge → negative score
    """
    # simple heuristic: subtract squared deviation from 0
    return 0.0 - net_charge**2

# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════

def get_score(candidate: "Candidate") -> tuple[str, float]:
    """
    Compute the net charge / pI proxy score for a candidate peptide.

    Parameters
    ----------
    candidate : Candidate
        Must have attribute peptide_mut (str)

    Returns
    -------
    tuple[str, float]
        (SCORE_NAME, score)
    """
    peptide = getattr(candidate, "peptide_mut", "")

    # handle invalid inputs
    if not isinstance(peptide, str) or len(peptide) == 0 or not peptide.isalpha():
        return SCORE_NAME, -1.0  # penalty for invalid/empty peptide

    net_charge = _compute_net_charge(peptide)
    score_value = _compute_score(net_charge)
    return SCORE_NAME, float(score_value)
