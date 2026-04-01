"""Tests for group 12 D5 module."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules.groupe_12_d5 import get_score


def _make_candidate(peptide_wt: str = "AAAAAAAAA") -> Candidate:
    return Candidate(
        candidate_id="TEST_D5_01",
        peptide_wt=peptide_wt,
        peptide_mut="AAAAAAAAV",
        mut_pos_1based=9,
    )


def test_returns_tuple_of_two():
    result = get_score(_make_candidate())
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_return_types():
    name, value = get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, (int, float))


def test_score_name_is_expected():
    name, _ = get_score(_make_candidate())
    assert name == "D_wt_presented"


def test_score_value_finite():
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)