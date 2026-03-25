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
    name, value = get_score(make_candidate("1234"))

    assert isinstance(name, str)
    assert isinstance(value, float)
# Invalid input
def test_get_score_invalid_sequence():
    with pytest.raises(Exception):
        get_score(make_candidate("1234"))
