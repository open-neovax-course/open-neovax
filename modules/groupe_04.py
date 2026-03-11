"""Group 04 scoring module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

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

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SCORE_NAME = "c_hla_delta_binding"


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _load_pssm() -> pd.DataFrame:
    """Load the HLA-A*02:01 PSSM used for delta binding."""
    return pd.read_csv(DATA_DIR / "hla_pssm_A0201.csv", index_col=0)


def _pssm_score(peptide: str, pssm: pd.DataFrame) -> float:
    """Compute the raw PSSM score for a 9-mer peptide."""
    if len(peptide) != 9:
        return 0.0

    total = 0.0
    for pos, aa in enumerate(peptide, start=1):
        column = f"P{pos}"
        if aa not in pssm.index or column not in pssm.columns:
            return 0.0
        total += float(pssm.loc[aa, column])
    return total


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Return the mutant-vs-wildtype delta binding score."""
    pssm = _load_pssm()
    score_wt = _pssm_score(candidate.peptide_wt, pssm)
    score_mut = _pssm_score(candidate.peptide_mut, pssm)
    delta = score_mut - score_wt
    return (SCORE_NAME, delta)
