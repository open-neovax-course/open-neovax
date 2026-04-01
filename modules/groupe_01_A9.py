"""Module A9 — mutation surprisal score.

This module measures how statistically surprising amino-acid substitutions are,
using human proteome amino-acid frequencies.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.types import Candidate


SCORE_NAME = "A_mutation_surprisal"

# Approximate amino-acid frequencies in the human proteome.
AA_FREQ = {
    "A": 0.070,
    "R": 0.056,
    "N": 0.036,
    "D": 0.047,
    "C": 0.014,
    "Q": 0.037,
    "E": 0.071,
    "G": 0.066,
    "H": 0.026,
    "I": 0.046,
    "L": 0.099,
    "K": 0.057,
    "M": 0.021,
    "F": 0.037,
    "P": 0.063,
    "S": 0.083,
    "T": 0.054,
    "W": 0.011,
    "Y": 0.027,
    "V": 0.060,
}


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Return mean mutation surprisal over all mutated positions.

    For each mutated position, the contribution is:
            -log2(freq_mut) + log2(freq_wt)

    This is equivalent to log2(freq_wt / freq_mut).
    """
    peptide_wt = candidate.peptide_wt
    peptide_mut = candidate.peptide_mut

    if (
        not isinstance(peptide_wt, str)
        or not isinstance(peptide_mut, str)
        or not peptide_wt
        or not peptide_mut
        or len(peptide_wt) != len(peptide_mut)
    ):
        return (SCORE_NAME, 0.0)

    wt = peptide_wt.upper()
    mut = peptide_mut.upper()

    total_surprisal = 0.0
    n_mutations = 0

    for aa_wt, aa_mut in zip(wt, mut):
        if aa_wt == aa_mut:
            continue

        if aa_wt not in AA_FREQ or aa_mut not in AA_FREQ:
            return (SCORE_NAME, 0.0)

        total_surprisal += -math.log2(AA_FREQ[aa_mut]) + math.log2(AA_FREQ[aa_wt])
        n_mutations += 1

    if n_mutations == 0:
        return (SCORE_NAME, 0.0)

    return (SCORE_NAME, total_surprisal / n_mutations)
