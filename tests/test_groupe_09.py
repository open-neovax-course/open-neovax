import pytest
from modules.groupe_09 import get_score


class DummyCandidate:
    def __init__(self, sequence):
        self.sequence = sequence
        self.scores = {}


# ✅ Test nominal (cas standard)
def test_get_score_nominal():
    candidate = DummyCandidate("ACDE")
    score_name, value = get_score(candidate)

    assert isinstance(score_name, str)
    assert isinstance(value, float)


# ✅ Test edge case (séquence vide)
def test_get_score_empty_sequence():
    candidate = DummyCandidate("")
    score_name, value = get_score(candidate)

    assert isinstance(score_name, str)
    assert isinstance(value, float)


# ❌ Test invalid input (séquence invalide)
def test_get_score_invalid_sequence():
    candidate = DummyCandidate("1234")  # caractères non valides

    with pytest.raises(Exception):
        get_score(candidate)