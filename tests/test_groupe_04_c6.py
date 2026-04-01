"""Tests for the groupe_04_c6 module."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import modules.groupe_04_c6 as groupe_04_c6
from logic.types import Candidate
from modules.groupe_04_c6 import get_score

# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════


def _make_candidate(
    peptide_mut: str = "AAAAAAAAV", peptide_wt: str = "AAAAAAAAA"
) -> Candidate:
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=9,
    )


# ══════════════════════════════════════════════════════════════════════
#  Return shape
# ══════════════════════════════════════════════════════════════════════


def test_returns_tuple_of_two():
    result = get_score(_make_candidate())
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_return_types():
    name, value = get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, (int, float))


def test_score_name_matches_constant():
    name, _ = get_score(_make_candidate())
    assert name == groupe_04_c6.SCORE_NAME


def test_score_value_finite():
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


# ══════════════════════════════════════════════════════════════════════
#  Score range & correctness
# ══════════════════════════════════════════════════════════════════════


def test_score_in_range():
    _, value = get_score(_make_candidate())
    assert 0.0 <= value <= 1.0


def test_perfect_binder_scores_one():
    """Peptide built from argmax of each PSSM column should score 1.0."""
    cand = _make_candidate(peptide_mut="ILFFFFFFV")
    _, score = get_score(cand)
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_worst_binder_scores_zero():
    """Peptide built from argmin of each PSSM column should score 0.0."""
    cand = _make_candidate(peptide_mut="DDDDDDDDP", peptide_wt="DDDDDDDDA")
    _, score = get_score(cand)
    assert math.isclose(score, 0.0, abs_tol=1e-9)


def test_gold_candidate_scores_high():
    """CAND_01-like peptide with L@P2 V@P9 should score high."""
    cand = _make_candidate(peptide_mut="SLMAFTIAV")
    _, score = get_score(cand)
    assert score > 0.6


def test_bad_candidate_scores_low():
    """Peptide with D@P2 K@P9 should score low."""
    cand = _make_candidate(peptide_mut="DDKREKDRK", peptide_wt="AAAAAAAAA")
    _, score = get_score(cand)
    assert score < 0.4


def test_uniform_pssm_column_scores_half(monkeypatch):
    """When best == worst at every position, each ratio falls back to 0.5."""
    flat_pssm = pd.DataFrame(
        {
            col: {aa: 1.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
            for col in groupe_04_c6.REQUIRED_PSSM_COLUMNS
        }
    )
    monkeypatch.setattr(groupe_04_c6, "_PSSM", flat_pssm)
    _, score = get_score(_make_candidate())
    assert math.isclose(score, 0.5, abs_tol=1e-9)


# ══════════════════════════════════════════════════════════════════════
#  Wild-type independence
# ══════════════════════════════════════════════════════════════════════


def test_score_independent_of_wildtype():
    """Same peptide_mut with different peptide_wt must produce the same score."""
    _, score_a = get_score(
        _make_candidate(peptide_mut="SLMAFTIAV", peptide_wt="AAAAAAAAA")
    )
    _, score_b = get_score(
        _make_candidate(peptide_mut="SLMAFTIAV", peptide_wt="SLMAFTIAA")
    )
    assert score_a == score_b


# ══════════════════════════════════════════════════════════════════════
#  Input validation — fallback to 0.0
# ══════════════════════════════════════════════════════════════════════


def test_none_peptide_mut_returns_neutral_score():
    candidate = _make_candidate()
    candidate.peptide_mut = None
    name, value = get_score(candidate)
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0



def test_lowercase_peptides_are_accepted():
    candidate = _make_candidate(peptide_mut="aaaaaaaav", peptide_wt="aaaaaaaaa")
    name, value = get_score(candidate)
    assert name == groupe_04_c6.SCORE_NAME
    assert isinstance(value, float)
    assert not math.isnan(value)
    assert not math.isinf(value)


@pytest.mark.parametrize(
    ("peptide_wt", "peptide_mut"),
    [
        ("AAAAAAAAA", "AAAAAAAA"),  # mut too short
        ("AAAAAAAAA", "AAAAAAAAAA"),  # mut too long
        ("AAAAAAAAA", ""),  # mut empty
        ("AAAAAAAAA", "         "),  # mut whitespace
    ],
)
def test_wrong_length_peptides_return_neutral_score(peptide_wt, peptide_mut):
    candidate = Candidate(
        candidate_id="TEST_LEN",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )
    name, value = get_score(candidate)
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0


@pytest.mark.parametrize(
    ("peptide_wt", "peptide_mut"),
    [
        ("AAAAAAAAA", "AAAAAA*AV"),  # special char in mut
        ("AAAAAAAAA", "AAAAAAZAV"),  # Z not in PSSM
        ("AAAAAAAAA", "AAA AAAAV"),  # space in mut
    ],
)
def test_invalid_amino_acids_return_neutral_score(peptide_wt, peptide_mut):
    candidate = Candidate(
        candidate_id="TEST_AA",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )
    name, value = get_score(candidate)
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0


# ══════════════════════════════════════════════════════════════════════
#  PSSM failure modes
# ══════════════════════════════════════════════════════════════════════


def test_missing_pssm_returns_neutral_score(monkeypatch):
    monkeypatch.setattr(groupe_04_c6, "_PSSM", None)
    name, value = get_score(_make_candidate())
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0


def test_empty_pssm_returns_neutral_score(monkeypatch):
    monkeypatch.setattr(groupe_04_c6, "_PSSM", pd.DataFrame())
    name, value = get_score(_make_candidate())
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0


def test_missing_pssm_columns_returns_neutral_score(monkeypatch):
    partial = pd.DataFrame({"P1": [1.0], "P2": [2.0]}, index=["A"])
    monkeypatch.setattr(groupe_04_c6, "_PSSM", partial)
    name, value = get_score(_make_candidate())
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0


def test_duplicate_pssm_index_returns_neutral_score(monkeypatch):
    dup = pd.DataFrame(
        {col: [1.0, 2.0] for col in groupe_04_c6.REQUIRED_PSSM_COLUMNS},
        index=["A", "A"],
    )
    monkeypatch.setattr(groupe_04_c6, "_PSSM", dup)
    name, value = get_score(_make_candidate())
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0


def test_non_dataframe_pssm_returns_neutral_score(monkeypatch):
    monkeypatch.setattr(groupe_04_c6, "_PSSM", object())
    name, value = get_score(_make_candidate())
    assert name == groupe_04_c6.SCORE_NAME
    assert value == 0.0
