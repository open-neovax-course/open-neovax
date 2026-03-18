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

import numpy as np

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
SCORE_NAME = "A_complexity_shannon"


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL FUNCTIONS (private)
# ══════════════════════════════════════════════════════════════════════
#
# You can define as many internal functions as you need.
# They will never be called by the pipeline.
# By convention, prefix them with _ to indicate they are private.

PHYSICO_GROUPS = {
    "A": "hydrophobic",
    "V": "hydrophobic",
    "I": "hydrophobic",
    "L": "hydrophobic",
    "M": "hydrophobic",
    "F": "hydrophobic",
    "W": "hydrophobic",
    "P": "hydrophobic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "Y": "polar",
    "C": "polar",
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "D": "negative",
    "E": "negative",
    "G": "special",
}


def _get_group_entropy(peptide: str) -> float:
    groups = [PHYSICO_GROUPS.get(aa, "unknown") for aa in peptide]
    # compute Shannon entropy on group frequencies
    H = 0.0
    freq_dict = dict()
    N = len(groups)
    Hmax = np.log2(5)

    for i in groups:
        if i in freq_dict.keys():
            freq_dict[i] += 1
        else:
            freq_dict[i] = 1

    for char in freq_dict:
        pi = freq_dict[char] / N
        H += pi * np.log2(pi)

    return float(-1 * H / Hmax)


def _get_shannon(peptide: str) -> float:
    """
    Internal function that computes the Shannon Entropy of a given peptide
    """
    H = 0.0
    freq_dict = dict()
    N = len(peptide)

    for i in peptide:
        if i in freq_dict.keys():
            freq_dict[i] += 1
        else:
            freq_dict[i] = 1

    for char in freq_dict:
        pi = freq_dict[char] / N
        H += pi * np.log2(pi)

    Hmax = np.log2(N)
    return float(-1 * H / Hmax)


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC FUNCTION (module entry point)
# ══════════════════════════════════════════════════════════════════════


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """
    Calcule l'entropie de Shannon normalisée d'un peptide.
    Renvoie une valeur entre 0 (pire, invalide, ou très répétitif)
    et 1 (complexité maximale).
    """
    # 1. Get the sequence to analyze
    peptide = candidate.peptide_mut

    if not isinstance(peptide, str) or not peptide:
        return (SCORE_NAME, 0.0)

    peptide = peptide.upper()
    N = len(peptide)

    if N < 8 or N > 11:
        return (SCORE_NAME, 0.0)

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for aa in peptide:
        if aa not in valid_aa:
            return (SCORE_NAME, 0.0)

    # 2. Compute the score using your logic
    mixed_score = (_get_shannon(peptide) + _get_group_entropy(peptide)) * 1 / 2
    k = 10  # Pente de la courbe (plus k est haut, plus c'est tranché)
    x0 = 0.5  # Le point d'inflexion

    score_value = 1 / (1 + np.exp(-k * (mixed_score - x0)))

    # 3. Return the result in the expected format
    return (SCORE_NAME, score_value)
