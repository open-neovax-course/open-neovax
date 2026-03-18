"""
Module B3 — ERAP trimming (N-terminal proxy).
"""

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
DISFAVORED_NTERM = {"D", "E", "P", "R", "K"}


def get_score(candidate):
    score_name = "B_erap_nterm_proxy"

    try:
        peptide = candidate.peptide_mut

        if not peptide:
            return (score_name, -1.0)

        first_aa = peptide[0].upper()

        if first_aa not in VALID_AA:
            return (score_name, -1.0)

        if first_aa in DISFAVORED_NTERM:
            return (score_name, -1.0)

        return (score_name, 0.0)

    except Exception:
        return (score_name, -1.0)