"""Group 04 — C6 scoring module.
Hypothesis: A mutant peptide is a better neoepitope candidate if it binds
HLA-A*02:01 strongly in absolute terms — i.e., each position carries a
favorable amino acid according to the PSSM, independent of the wild-type
sequence.
"""

from __future__ import annotations

from pathlib import Path
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

# Path to the data/ directory at the project root.
# Useful if your module needs to load an external file
# (PSSM matrix, self-peptide list, etc.).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Name of the score returned by this module.
# IMPORTANT: this name must be unique across all modules!
# Convention: <department>_<concept>[_detail]
SCORE_NAME = "C_binding_quality"
REQUIRED_PSSM_COLUMNS = [f"P{i}" for i in range(1, 10)]


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.
def _load_pssm() -> pd.DataFrame | None:
    """Load and validate the HLA-A*02:01 PSSM used for delta binding."""
    try:
        pssm = pd.read_csv(DATA_DIR / "hla_pssm_A0201.csv", index_col=0)
    except Exception:
        return None

    if not isinstance(pssm, pd.DataFrame) or pssm.empty or not pssm.index.is_unique:
        return None

    missing_columns = [col for col in REQUIRED_PSSM_COLUMNS if col not in pssm.columns]
    if missing_columns:
        return None

    try:
        pssm.loc[:, REQUIRED_PSSM_COLUMNS] = pssm.loc[:, REQUIRED_PSSM_COLUMNS].apply(
            pd.to_numeric
        )
    except Exception:
        return None

    return pssm


def _normalize_peptide(peptide: object, pssm: pd.DataFrame | None) -> str | None:
    """Return a cleaned 9-mer peptide accepted by the PSSM, else None."""
    if not isinstance(pssm, pd.DataFrame) or not isinstance(peptide, str):
        return None

    peptide = peptide.strip().upper()
    if len(peptide) != 9:
        return None

    if not all(aa in pssm.index for aa in peptide):
        return None

    return peptide


def _pssm_score(peptide: str, pssm: pd.DataFrame) -> float:
    """Compute the raw PSSM score for a validated 9-mer peptide."""
    total = 0.0
    for pos, aa in enumerate(peptide, start=1):
        column = f"P{pos}"
        try:
            total += float(pssm.loc[aa, column])
        except (TypeError, ValueError):
            return 0.0
    return total


# ══════════════════════════════════════════════════════════════════════
#  LOCAL CACHE
# ══════════════════════════════════════════════════════════════════════

_PSSM = _load_pssm()


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Compute how good does mutated peptide bind (more is better)

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
    pssm = _PSSM
    peptide_wt = _normalize_peptide(getattr(candidate, "peptide_wt", None), pssm)
    peptide_mut = _normalize_peptide(getattr(candidate, "peptide_mut", None), pssm)
    if pssm is None or peptide_wt is None or peptide_mut is None:
        return (SCORE_NAME, 0.0)

    def _position_ratio(aa: str, col: str) -> float:
        score = pssm.loc[aa, col]
        best, worst = pssm[col].max(), pssm[col].min()
        return (score - worst) / (best - worst) if best != worst else 0.5

    ratios = [
        _position_ratio(aa, col) for aa, col in zip(peptide_mut, REQUIRED_PSSM_COLUMNS)
    ]
    return (SCORE_NAME, sum(ratios) / len(ratios))
