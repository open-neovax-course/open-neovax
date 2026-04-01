"""
Module B3 — ERAP trimming (N-terminal proxy).
"""

SCORE_NAME = "B_erap_nterm_proxy"

ERAP_TRIMMING = {
    "P": 1.0,
    "K": 0.7,
    "R": 0.7,
    "D": 0.5,
    "E": 0.5,
    "H": 0.4,
    "N": 0.2,
    "Q": 0.2,
    "C": 0.1,
    "V": 0.0,
    "I": 0.0,
    "T": 0.0,
    "W": 0.0,
    "Y": 0.0,
    "L": -0.5,
    "M": -0.5,
    "F": -0.5,
    "A": -0.8,
    "S": -0.8,
    "G": -1.0,
}


def get_score(candidate):
    try:
        peptide = candidate.peptide_mut

        if not peptide:
            return (SCORE_NAME, -1.0)

        first_aa = peptide[0].upper()
        score = ERAP_TRIMMING.get(first_aa, 0.0)
        return (SCORE_NAME, score)

    except Exception:
        return (SCORE_NAME, -1.0)
