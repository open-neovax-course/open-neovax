"""Tests for group 12 module."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules import groupe_12
from modules.groupe_12 import get_score


def _make_candidate(peptide_mut: str = "AAAAAAAAV") -> Candidate:
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
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


def test_score_name_not_empty():
    name, _ = get_score(_make_candidate())
    assert name != ""


def test_score_value_finite():
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


def test_exact_match_returns_penalty():
    original_corpus = groupe_12.HUMAN_PEPTIDES
    try:
        groupe_12.HUMAN_PEPTIDES = {"PEPTIDE123"}
        _, value = get_score(_make_candidate("PEPTIDE123"))
        assert value == groupe_12.PENALTY
    finally:
        groupe_12.HUMAN_PEPTIDES = original_corpus


def test_absent_peptide_returns_zero():
    original_corpus = groupe_12.HUMAN_PEPTIDES
    try:
        groupe_12.HUMAN_PEPTIDES = {"PEPTIDE123"}
        _, value = get_score(_make_candidate("DIFFERENT999"))
        assert value == 0.0
    finally:
        groupe_12.HUMAN_PEPTIDES = original_corpus


def test_empty_peptide_returns_zero():
    _, value = get_score(_make_candidate(""))
    assert value == 0.0