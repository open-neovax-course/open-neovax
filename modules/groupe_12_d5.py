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
HUMAN_PEPTIDES_PATH = DATA_DIR / "human_peptides_small.txt"
SCORE_NAME = "D_wt_presented"
BONUS = 1.0

# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════


def _load_human_peptides(path: Path) -> set[str]:
    """Load the human peptide corpus from a text file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return {line.strip().upper() for line in f if line.strip()}
    except OSError:
        return set()


HUMAN_PEPTIDES = _load_human_peptides(HUMAN_PEPTIDES_PATH)


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute D5 wild-type presentation score."""
    wt = candidate.peptide_wt

    if not isinstance(wt, str) or not wt:
        return (SCORE_NAME, 0.0)

    wt = wt.strip().upper()

    if wt in HUMAN_PEPTIDES:
        return (SCORE_NAME, BONUS)

    return (SCORE_NAME, 0.0)
