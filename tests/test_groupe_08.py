import math
from logic.types import Candidate
from modules.groupe_08 import get_score, SCORE_NAME

def _make_candidate(peptide_mut: str, peptide_wt: str = "AAAAAAAAA") -> Candidate:
    return Candidate(
        candidate_id="TEST_08",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )

def test_hydrophobic_anchor_improvement():
    """Mutation that improves anchor hydrophobicity should score positive."""
    # WT has A@P2 (low hydro), MUT has L@P2 (high hydro)
    cand = _make_candidate(peptide_wt="SAMAFTIAV", peptide_mut="SLMAFTIAV")
    _, score = get_score(cand)
    assert score > 0.0

def test_wt_equals_mut_scores_zero():
    """No mutation = no delta = score 0."""
    cand = _make_candidate(peptide_wt="SLMAFTIAV", peptide_mut="SLMAFTIAV")
    _, score = get_score(cand)
    assert score == 0.0

def test_exposed_hydrophobicity_penalty():
    """Mutation that makes exposed center more hydrophobic should score lower."""
    # WT: polar center (D), MUT: hydrophobic center (I)
    _, score_polar = get_score(_make_candidate("AAADAAAAA", "AAAAAAAAA"))
    _, score_hydro = get_score(_make_candidate("AAAIAAAAA", "AAAAAAAAA"))
    assert score_hydro < score_polar

def test_invalid_inputs_no_crash():
    """Ensure CI remains green with robust error handling."""
    _, score = get_score(_make_candidate(None))
    assert score == 0.0