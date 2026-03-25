from logic.types import Candidate
from modules.groupe_08 import get_score


def _make_cand(mut: str, wt: str = "AAAAAAAAA") -> Candidate:
    return Candidate(
        candidate_id="08", peptide_wt=wt, peptide_mut=mut, mut_pos_1based=5
    )


def test_position_reflects_impact():
    """AC: Same residue has different impact at anchor vs exposed."""
    _, score_anchor = get_score(_make_cand("ALAAAAAAV", "AAAAAAAAV"))
    _, score_exposed = get_score(_make_cand("AAALAAAVV", "AAAAAAAAV"))

    assert score_anchor > score_exposed


def test_cand11_vs_cand01():
    """AC: CAND_11 (all hydrophobic) scores lower than CAND_01 (mixed)."""
    _, score_11 = get_score(_make_cand("ILVMILMVL", "AAAAAAAAA"))
    _, score_01 = get_score(_make_cand("ALADNAAAV", "AAAAAAAAA"))

    assert score_11 < score_01


def test_anchor_good_exposed_polar_vs_all_hydrophobic():
    """AC: Specific comparison test."""
    _, score_perfect = get_score(_make_cand("FLRDKREAV", "AAAAAAAAA"))
    _, score_greasy = get_score(_make_cand("FLLLLLLLV", "AAAAAAAAA"))

    assert score_perfect > score_greasy


def test_ci_robustness():
    """Verify neutral return for invalid inputs to keep CI green."""
    assert get_score(_make_cand("Short"))[1] == 0.0
    assert isinstance(get_score(_make_cand("AAAAAAAAA"))[1], float)
