"""Group 12 module — D5 wild-type presented bonus."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.types import Candidate


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCORE_NAME = "D_wt_presented"


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════


def _compute_placeholder(peptide: str) -> float:
    _ = peptide
    return 0.0


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute D5 wild-type presentation score."""
    wt = candidate.peptide_wt
    score_value = _compute_placeholder(wt)
    return (SCORE_NAME, score_value)