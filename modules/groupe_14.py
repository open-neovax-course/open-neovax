"""
Module A7 — Aromatic amino acid content (Group 14, Eliot RABIN)

Hypothesis: F, W, Y are bulky and protrude from the peptide, creating
pi-stacking / Van der Waals contacts with the TCR.  More aromatics →
more "visible" to the immune system.  Score = fraction of F/W/Y
(high = good, low = less distinctive).

Limitations: doesn't account for residue position, treats F/W/Y equally,
and very aromatic peptides may aggregate (caught by A1/B4 instead).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.types import Candidate

SCORE_NAME = "A_aromaticity"
AROMATIC = set("FWY")


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Return the aromatic fraction of the mutant peptide.

    Parameters
    ----------
    candidate : Candidate
        We only look at ``candidate.peptide_mut``.

    Returns
    -------
    tuple[str, float]
        ``("A_aromaticity", fraction)`` where *fraction* is between
        0.0 (no aromatics) and 1.0 (all aromatic).
    """
    pep = candidate.peptide_mut

    if not isinstance(pep, str) or not pep:
        return (SCORE_NAME, 0.0)

    pep = pep.strip().upper()
    if len(pep) == 0:
        return (SCORE_NAME, 0.0)

    count = sum(1 for aa in pep if aa in AROMATIC)
    return (SCORE_NAME, count / len(pep))
