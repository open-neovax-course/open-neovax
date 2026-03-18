from logic.types import Candidate
from modules.groupe_10 import get_score


def make_candidate(peptide_mut):
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )


def test_nominal_favored_nterm():
    name, value = get_score(make_candidate("SLMAFTIAV"))
    assert name == "B_erap_nterm_proxy"
    assert isinstance(value, float)
    assert value == 0.0


def test_penalized_nterm():
    name, value = get_score(make_candidate("DLMAFTIAV"))
    assert name == "B_erap_nterm_proxy"
    assert isinstance(value, float)
    assert value == -1.0


def test_empty_peptide():
    name, value = get_score(make_candidate(""))
    assert name == "B_erap_nterm_proxy"
    assert isinstance(value, float)
    assert value == -1.0
