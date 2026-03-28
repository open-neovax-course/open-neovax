from logic.types import Candidate
from modules.groupe_09 import get_score


def make_candidate(peptide_mut: str):
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )


# Nominal case
def test_get_score_nominal():
    name, value = get_score(make_candidate("ACDEKRH"))
    assert name == "A_net_charge"
    assert isinstance(value, float)
    assert value < 0  # slightly charged, penalized if extreme


# Edge case: empty peptide
def test_get_score_empty():
    name, value = get_score(make_candidate(""))
    assert name == "A_net_charge"
    assert value == -1.0


# Invalid-input case: invalid characters
def test_get_score_invalid():
    name, value = get_score(make_candidate("1234!@#"))
    assert name == "A_net_charge"
    assert value == -1.0


# Case: highly positively charged
def test_highly_positive():
    name, value = get_score(make_candidate("KKKKKKKKK"))
    assert value < 0


# Case: highly negatively charged
def test_highly_negative():
    name, value = get_score(make_candidate("DDDDDDDDD"))
    assert value < 0


# Case: neutral peptide
def test_neutral_peptide():
    name, value = get_score(make_candidate("ACGTFPQIL"))
    assert value == 0.0
