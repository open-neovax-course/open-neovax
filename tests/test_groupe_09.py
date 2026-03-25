import pytest

from logic.types import Candidate
from modules.groupe_09 import get_score


def make_candidate(peptide_mut):
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )


# Test nominal
def test_get_score_nominal():
    name, value = get_score(make_candidate("ACDE"))

    assert isinstance(name, str)
    assert isinstance(value, float)


# Edge case (séquence vide)
def test_get_score_empty_sequence():
    name, value = get_score(make_candidate(""))

    assert isinstance(name, str)
    assert isinstance(value, float)


# Invalid input
def test_get_score_invalid_sequence():
    with pytest.raises(Exception):
        get_score(make_candidate("1234"))
