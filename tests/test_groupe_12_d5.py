"""Tests for group 12 D5 module."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules import groupe_12_d5
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


def test_wt_found_returns_bonus():
    original_corpus = groupe_12_d5.HUMAN_PEPTIDES
    try:
        groupe_12_d5.HUMAN_PEPTIDES = {"SAMAFTIAV"}
        _, value = get_score(_make_candidate("SAMAFTIAV"))
        assert value == 1.0
    finally:
        groupe_12_d5.HUMAN_PEPTIDES = original_corpus


def test_wt_absent_returns_zero():
    original_corpus = groupe_12_d5.HUMAN_PEPTIDES
    try:
        groupe_12_d5.HUMAN_PEPTIDES = {"SAMAFTIAV"}
        _, value = get_score(_make_candidate("DIFFERENTWT"))
        assert value == 0.0
    finally:
        groupe_12_d5.HUMAN_PEPTIDES = original_corpus


def test_empty_wt_returns_zero():
    _, value = get_score(_make_candidate(""))
    assert value == 0.0