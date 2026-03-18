from __future__ import annotations

import math

from logic.types import Candidate
from modules.groupe_07 import get_score

"""Tests for the template module."""


import math

from logic.types import Candidate
from modules.groupe_07 import get_score

def _make_candidate(peptide_mut: str) -> Candidate:
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )


def test_returns_expected_score_name_and_float():
    name, value = get_score(_make_candidate("SLYNTVATL"))

    assert name == "B_tap_transport_proxy"
    assert isinstance(value, float)


def test_score_value_is_finite():
    _, value = get_score(_make_candidate("SLYNTVATL"))

    assert not math.isnan(value)
    assert not math.isinf(value)


def test_invalid_amino_acid_gets_explicit_penalty():
    name, value = get_score(_make_candidate("SLYNTVAXL"))

    assert name == "B_tap_transport_proxy"
    assert value == -2.0


def test_invalid_length_gets_explicit_penalty():
    name, value = get_score(_make_candidate("AAAA"))

    assert name == "B_tap_transport_proxy"
    assert value == -1.5


def test_hydrophobic_c_terminus_scores_better_than_charged_one():
    _, favored_value = get_score(_make_candidate("SLYNTVATL"))
    _, penalized_value = get_score(_make_candidate("RRRDEEDDK"))

    assert favored_value > penalized_value
