"""
C3 — Secondary anchors — Open-NeoVax
"""

from __future__ import annotations

import warnings
from functools import lru_cache
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
SCORE_NAME = "C_delta_binding"


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.

_score_weights = {(0, "P1"): 0.45, (2, "P3"): 0.3, (6, "P7"): 0.15}


def _secondary_anchoring(pep: str) -> float:
    """Calculate a score for a given peptide based on given weights.

    Parameters
    ----------
        pep (str): sequence of the peptide
    """
    pssm = _get_pssm_matrix()
    score = 0
    for (pos, label), weight in _score_weights.items():
        score += pssm.get(pep[pos], {}).get(label, 0.0) * weight
    return score


@lru_cache(maxsize=1)
def _get_pssm_matrix() -> dict[str, dict[str, float]]:
    """Retrieve the PSSM matrix for HLA-A*02:01."""
    return _load_pssm_from_csv(DATA_DIR / "hla_pssm_A0201.csv")


def _load_pssm_from_csv(file_path: Path) -> dict[str, dict[str, float]]:
    """Load a PSSM matrix from a CSV file.

    Parameters
    ----------
        file_path (Path)

    Returns
    ----------
        dict[str, dict[str, float]]

    If the file does not exist, returns an empty dictionary.
    """
    if not file_path.exists():
        return {}
    try:
        data = pd.read_csv(file_path, index_col=0)
        data = data.to_dict(orient="index")
    except pd.errors.EmptyDataError:
        return {}
    except pd.errors.ParserError:
        return {}
    except Exception as e:
        warnings.warn(f"Error while loading PSSM file: {e}")
        return {}
    return data


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
    peptide = candidate.peptide_mut

    if len(peptide) < 7:
        return (SCORE_NAME, 0.0)

    score_value = _secondary_anchoring(peptide.upper())

    return (SCORE_NAME, score_value)
