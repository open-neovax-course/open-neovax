"""Tests for module A9 (mutation surprisal)."""

from __future__ import annotations

import math

import pytest

from logic.types import Candidate
from modules.groupe_01_A9 import SCORE_NAME, get_score


def _make_candidate(peptide_wt: str, peptide_mut: str) -> Candidate:
    return Candidate(
        candidate_id="TEST_A9",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=2,
    )


def test_l_to_w_scores_higher_than_l_to_i():
    cand_rare = _make_candidate("LLL", "LWL")  # L -> W (common -> rare)
    cand_common = _make_candidate("LLL", "LIL")  # L -> I (common -> common)

    name_rare, score_rare = get_score(cand_rare)
    name_common, score_common = get_score(cand_common)

    assert name_rare == SCORE_NAME
    assert name_common == SCORE_NAME
    assert score_rare > score_common


def test_rare_to_common_is_negative():
    cand = _make_candidate("WWW", "WAW")  # W -> A (rare -> common)
    _, score = get_score(cand)
    assert score < 0.0


def test_no_mutation_returns_zero():
    cand = _make_candidate("SLMAFTIAV", "SLMAFTIAV")
    _, score = get_score(cand)
    assert score == 0.0


def test_invalid_length_returns_zero():
    cand = _make_candidate("SLMAFTIAV", "SLMAF")
    _, score = get_score(cand)
    assert score == 0.0


def test_score_is_finite():
    cand = _make_candidate("SLMAFTIAV", "SWMAFTIAV")
    _, score = get_score(cand)
    assert not math.isnan(score)
    assert not math.isinf(score)


def test_invalid_amino_acid_returns_zero():
    cand = _make_candidate("SLMAFTIAV", "SXMAFTIAV")
    _, score = get_score(cand)
    assert score == pytest.approx(0.0)
