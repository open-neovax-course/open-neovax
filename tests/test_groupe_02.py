"""Tests for the template module."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules.groupe_02 import get_score


def _make_candidate(
    candidate_id: str = "TEST_01",
    peptide_wt: str = "AAAAAAAAA",
    peptide_mut: str = "AAAAAAAAV",
    mut_pos_1based: int = 9,
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=mut_pos_1based,
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


def test_nominal_case():
    """Test Nominal : Un peptide standard de 9 acides aminés qui se termine par une lettre valide."""
    candidate = _make_candidate()
    name, value = get_score(candidate)
    assert isinstance(value, (int, float))
    assert not math.isnan(value)
    assert not math.isinf(value)
    assert value == 2.0


def test_edge_case_short_peptide():
    """Test Edge-case : Un peptide trop court ou vide ne doit pas faire planter le module."""
    candidate = _make_candidate(peptide_wt="AA", peptide_mut="AAV", mut_pos_1based=2)
    name, value = get_score(candidate)
    assert isinstance(value, (int, float))
    assert not math.isnan(value)
    assert not math.isinf(value)

    candidate = _make_candidate(peptide_wt="", peptide_mut="", mut_pos_1based=0)
    _, value = get_score(candidate)
    assert isinstance(value, (int, float))
    assert not math.isnan(value)
    assert not math.isinf(value)


def test_invalid_case():
    """Test Invalid : Un peptide trop court ou vide ne doit pas faire planter le module."""

    candidate = _make_candidate(
        peptide_wt="111111111", peptide_mut="111111111", mut_pos_1based=9
    )
    name, value = get_score(candidate)
    assert isinstance(value, (int, float))
    assert math.isnan(value)
    assert math.isinf(value)


def test_invalid_case_mut_pos_1based():
    """Test Invalid : Un peptide trop long ou vide ne doit pas faire planter le module."""
    candidate = _make_candidate(
        mut_pos_1based=18,
        peptide_wt="AAAAAAAAAAAAAAAAAA",
        peptide_mut="AAAAAAAAAAAAAAAAAV",
    )
    name, value = get_score(candidate)
    assert isinstance(value, (int, float))
    assert math.isnan(value)
    assert math.isinf(value)
