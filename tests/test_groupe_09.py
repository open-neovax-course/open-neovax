import pytest

from modules.groupe_09 import get_score


class DummyCandidate:
    def __init__(self, sequence):
        self.peptide_mut = sequence
        self.peptide_wt = sequence
        self.mut_pos_1based = 1
        self.gene = "TEST"
        self.hla_allele = "HLA-A02:01"
        self.scores = {}


def test_get_score_nominal():
    candidate = DummyCandidate("ACDE")
    score_name, value = get_score(candidate)

    assert isinstance(score_name, str)
    assert isinstance(value, float)


def test_get_score_empty_sequence():
    candidate = DummyCandidate("")
    score_name, value = get_score(candidate)

    assert isinstance(score_name, str)
    assert isinstance(value, float)


def test_get_score_invalid_sequence():
    candidate = DummyCandidate("1234")

    with pytest.raises(Exception):
        get_score(candidate)
