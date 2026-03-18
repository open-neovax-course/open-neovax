import math

from logic.types import Candidate
from modules.groupe_08 import get_score


def make_candidate(peptide_mut: str, peptide_wt: str | None = None):
    return Candidate(
        candidate_id="test",
        peptide_wt=peptide_wt or peptide_mut,
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
        gene="TEST",
        hla_allele="A*02:01",
        note=None,
    )


def test_get_score_returns_correct_type():
    candidate = make_candidate("AAAAAAAAA")
    name, value = get_score(candidate)

    assert isinstance(name, str)
    assert isinstance(value, float)


def test_get_score_not_nan():
    candidate = make_candidate("AAAAAAAAA")
    _, value = get_score(candidate)

    assert not math.isnan(value)


def test_get_score_returns_tuple():
    candidate = make_candidate("AAAAAAAAA")
    result = get_score(candidate)

    assert isinstance(result, tuple)
    assert len(result) == 2

