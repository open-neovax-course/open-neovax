"""Tests for the template module."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules.groupe_05 import get_score


def _make_candidate(
    peptide_wt="AAAAAAAAA",
    peptide_mut="AAAAAAAAV",
) -> Candidate:
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt=peptide_wt,
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


def test_identical_sequences():
    """Test that identical sequences return 0.0."""
    cand = _make_candidate(peptide_wt="SLMAFTIAV", peptide_mut="SLMAFTIAV")
    _, score = get_score(cand)
    assert score == 0.0


def test_conservative_mutation():
    """L -> V (both hydrophobic) should score between 0.0 and 0.3"""
    cand = _make_candidate(peptide_wt="SLMAFTIAV", peptide_mut="SVMAFTIAV")
    _, score = get_score(cand)
    assert 0.0 < score < 0.3


def test_radical_mutation():
    """L -> D (hydrophobic -> negative) should score > 0.5"""
    cand = _make_candidate(peptide_wt="SLMAFTIAV", peptide_mut="SDMAFTIAV")
    _, score = get_score(cand)
    assert score > 0.5


def test_different_lengths():
    """Peptides de longueurs différentes should score -1.0."""
    cand = _make_candidate(peptide_wt="SLMAFTIAV", peptide_mut="SLMAF")
    _, score = get_score(cand)
    assert score == -1.0


def test_granularity():
    """
    L -> D (hydrophobic -> negative) should score worse than L-> S (hydrophobic
    -> polar)

    """
    cand_negative = _make_candidate(peptide_wt="SLMAFTIAV", peptide_mut="SDMAFTIAV")
    _, score_negative = get_score(cand_negative)

    cand_polar = _make_candidate(peptide_wt="SLMAFTIAV", peptide_mut="SSMAFTIAV")
    _, score_polar = get_score(cand_polar)

    assert score_negative > score_polar
