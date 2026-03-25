SCORE_NAME = "B_sanity_check"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def get_score(candidate):
    try:
        mut = candidate.peptide_mut
        wt = candidate.peptide_wt

        if not mut or not wt:
            return (SCORE_NAME, -10.0)

        mut = mut.strip().upper()
        wt = wt.strip().upper()

        if not mut or not wt:
            return (SCORE_NAME, -10.0)

        penalties = 0

        if not (8 <= len(mut) <= 11):
            penalties += 1

        if not all(aa in VALID_AA for aa in mut):
            penalties += 1

        if wt == mut:
            penalties += 1

        if penalties == 0:
            return (SCORE_NAME, 1.0)

        return (SCORE_NAME, -float(penalties))

    except Exception:
        return (SCORE_NAME, -10.0)
