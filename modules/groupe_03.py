"""
C1 — Anchor position P2 (PSSM HLA-I) module — Open-NeoVax
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
SCORE_NAME = "C_anchoring_P2"


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.


def _lookup_score(amino: str) -> float:
    """Lookup the score for a given amino acid at position P2 in the peptide.

    Parameters
    ----------
        amino (str): single-letter code of the amino acid at position P2
    """
    pssm = _get_pssm_matrix()
    return pssm.get(amino, {}).get("P2", 0.0)


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

    if type(peptide) is not str or len(peptide) < 2:
        return (SCORE_NAME, 0.0)

    score_value = _lookup_score(peptide[1].upper())

    return (SCORE_NAME, score_value)
