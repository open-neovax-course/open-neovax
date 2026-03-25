import pytest

from modules.groupe_09 import get_score


class DummyCandidate:
    def __init__(self, sequence):
        self.sequence = sequence
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
