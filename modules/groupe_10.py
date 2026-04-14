"""
Module B3 — ERAP trimming (N-terminal proxy).

Hypothesis: ERAP enzymes trim the N-terminal of peptides before HLA
loading. Some N-terminal amino acids are trimmed efficiently, others
block the process. A disfavored N-terminal means less likely presented.

Computation: look up the first amino acid in a preference table based
on ERAP1 trimming biases (Saveanu et al., 2005).

Limitations: only the first amino acid is used, ERAP1 vs ERAP2 not
distinguished, values are a simplified heuristic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.types import Candidate

SCORE_NAME = "B_erap_nterm_proxy"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids
MIN_LENGTH = 8
MAX_LENGTH = 11
PENALTY = -1.0  # worst possible score, same as glycine (cleaved too fast)

# ERAP1 trimming preference table (Saveanu et al., 2005).
# Positive values = resistant to ERAP → peptide preserved → good for presentation.
# Negative values = well cleaved by ERAP → peptide shortened → bad for presentation.
ERAP_TRIMMING = {
    # Resistant to ERAP → peptide preserved → good for presentation
    "P": 1.0,  # proline blocks the enzyme (peptide bond to proline is hard to cleave)
    "K": 0.7,  # basic residues resist trimming
    "R": 0.7,
    "D": 0.5,  # acidic residues resist (slower)
    "E": 0.5,
    "H": 0.4,
    "N": 0.2,
    "Q": 0.2,
    "C": 0.1,
    # Neutral
    "V": 0.0,
    "I": 0.0,
    "T": 0.0,
    "W": 0.0,
    "Y": 0.0,
    # Well cleaved by ERAP → peptide shortened → bad for presentation
    "L": -0.5,  # bulky hydrophobic — preferred ERAP substrates
    "M": -0.5,
    "F": -0.5,
    "A": -0.8,  # small residues — very well cleaved
    "S": -0.8,
    "G": -1.0,  # glycine — cleaved extremely fast
}


def get_score(candidate: "Candidate") -> tuple[str, float]:
    """Return the ERAP N-terminal trimming score for a candidate."""
    try:
        peptide = candidate.peptide_mut

        # reject missing or non-string input
        if not peptide or not isinstance(peptide, str):
            return (SCORE_NAME, PENALTY)

        peptide = peptide.strip().upper()

        if not peptide:
            return (SCORE_NAME, PENALTY)

        # MHC-I ligands are 8-11 aa; anything else is out of scope
        if not (MIN_LENGTH <= len(peptide) <= MAX_LENGTH):
            return (SCORE_NAME, PENALTY)

        first_aa = peptide[0]

        # non-standard amino acid — cannot score
        if first_aa not in VALID_AA:
            return (SCORE_NAME, PENALTY)

        # look up the ERAP preference for the N-terminal residue
        score = ERAP_TRIMMING.get(first_aa, 0.0)
        return (SCORE_NAME, score)

    except Exception:
        return (SCORE_NAME, PENALTY)
