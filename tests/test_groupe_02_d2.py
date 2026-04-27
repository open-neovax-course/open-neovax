"""Tests for groupe_02_d2 (approximate self-similarity / D2)."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules import groupe_02_d2


def _make_candidate(
    candidate_id: str = "TEST_01",
    peptide_wt: str = "AAAAAAAAA",
    peptide_mut: str = "AAAAAAAAA",
    mut_pos_1based: int = 9,
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=mut_pos_1based,
    )


def test_returns_tuple_of_two():
    result = groupe_02_d2.get_score(_make_candidate())
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_return_types():
    name, value = groupe_02_d2.get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, (int, float))


def test_score_name_not_empty():
    name, _ = groupe_02_d2.get_score(_make_candidate())
    assert name == groupe_02_d2.SCORE_NAME


def test_score_value_finite():
    _, value = groupe_02_d2.get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


def test_exact_self_peptide_lowest_score(monkeypatch):
    """Distance 0 should get the exact distance 0.0, distance 1 -> 1.0."""
    # Control the self-peptide corpus: only one peptide of length 9.
    monkeypatch.setattr(groupe_02_d2, "_SELF_PEPTIDES", {"AAAAAAAAA"})

    candidate_exact = _make_candidate(peptide_mut="AAAAAAAAA")
    candidate_d1 = _make_candidate(peptide_mut="AAAAAAAAV")  # distance 1

    _, score_exact = groupe_02_d2.get_score(candidate_exact)
    _, score_d1 = groupe_02_d2.get_score(candidate_d1)

    assert score_exact == 0.0
    assert score_d1 == 1.0


def test_distance_two_larger_than_distance_one(monkeypatch):
    """Distance 2 should get 2.0, distance 1 should get 1.0."""
    monkeypatch.setattr(groupe_02_d2, "_SELF_PEPTIDES", {"AAAAAAAAA"})

    candidate_d1 = _make_candidate(peptide_mut="AAAAAAAAV")  # distance 1
    candidate_d2 = _make_candidate(peptide_mut="AAAAAAAVV")  # distance 2

    _, score_d1 = groupe_02_d2.get_score(candidate_d1)
    _, score_d2 = groupe_02_d2.get_score(candidate_d2)

    assert score_d1 == 1.0
    assert score_d2 == 2.0


def test_distance_ge_three_actual_distance(monkeypatch):
    """Peptides far from self get their actual distance."""
    monkeypatch.setattr(groupe_02_d2, "_SELF_PEPTIDES", {"AAAAAAAAA"})

    # At least three differences from "AAAAAAAAA".
    far_mutant = "VVVAAAAAA"
    _, score_far = groupe_02_d2.get_score(_make_candidate(peptide_mut=far_mutant))

    assert score_far == 3.0


def test_lowercase_peptide_equivalent_to_uppercase(monkeypatch):
    """Lowercase mutant peptide should behave like uppercase version."""
    monkeypatch.setattr(groupe_02_d2, "_SELF_PEPTIDES", {"AAAAAAAAA"})

    upper_candidate = _make_candidate(peptide_mut="AAAAAAAAA")
    lower_candidate = _make_candidate(peptide_mut="aaaaaaaaa")

    _, score_upper = groupe_02_d2.get_score(upper_candidate)
    _, score_lower = groupe_02_d2.get_score(lower_candidate)

    assert score_upper == score_lower


def test_empty_peptide_returns_neutral_score():
    """Empty peptide must return a neutral score and not crash."""
    candidate = _make_candidate(peptide_wt="", peptide_mut="", mut_pos_1based=0)
    name, value = groupe_02_d2.get_score(candidate)

    assert name == groupe_02_d2.SCORE_NAME
    assert value == 0.0


def test_missing_corpus_returns_neutral_score(monkeypatch):
    """Missing or empty corpus should yield a neutral score (peptide length)."""
    monkeypatch.setattr(groupe_02_d2, "_SELF_PEPTIDES", set())

    name, value = groupe_02_d2.get_score(_make_candidate(peptide_mut="AAAAAAAAA"))

    assert name == groupe_02_d2.SCORE_NAME
    assert value == 0.0


def test_no_same_length_in_corpus_returns_neutral_score(monkeypatch):
    """If no self peptide has the same length,
    score must be neutral (peptide length)."""
    # Corpus only has 4-mers, while candidate peptide is a 9-mer.
    monkeypatch.setattr(groupe_02_d2, "_SELF_PEPTIDES", {"AAAA", "BBBB"})

    name, value = groupe_02_d2.get_score(_make_candidate(peptide_mut="AAAAAAAAA"))

    assert name == groupe_02_d2.SCORE_NAME
    assert value == 0.0


def test_edge_case_short_peptide():
    """A too-short peptide must not crash the module and stays finite."""
    candidate = _make_candidate(peptide_wt="AA", peptide_mut="AA", mut_pos_1based=2)
    name, value = groupe_02_d2.get_score(candidate)

    assert name == groupe_02_d2.SCORE_NAME
    assert isinstance(value, (int, float))
    assert not math.isnan(value)
    assert not math.isinf(value)
