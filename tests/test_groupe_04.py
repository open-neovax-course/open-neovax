"""Tests for the template module."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

import modules.groupe_04 as groupe_04
from logic.types import Candidate


def _make_candidate() -> Candidate:
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt="AAAAAAAAA",
        peptide_mut="AAAAAAAAV",
        mut_pos_1based=9,
    )


def test_return_types():
    name, value = groupe_04.get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, (int, float))


def test_score_name_not_empty():
    name, _ = groupe_04.get_score(_make_candidate())
    assert name != ""


def test_score_value_finite():
    _, value = groupe_04.get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


def test_non_string_peptide_returns_neutral_score():
    candidate = _make_candidate()
    candidate.peptide_mut = None

    name, value = groupe_04.get_score(candidate)

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_non_string_wildtype_peptide_returns_neutral_score():
    candidate = _make_candidate()
    candidate.peptide_wt = None

    name, value = groupe_04.get_score(candidate)

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_missing_candidate_attribute_returns_neutral_score():
    candidate = SimpleNamespace(peptide_mut="AAAAAAAAV")

    name, value = groupe_04.get_score(candidate)

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_missing_pssm_file_returns_neutral_score(monkeypatch):
    monkeypatch.setattr(groupe_04, "_PSSM", None)

    name, value = groupe_04.get_score(_make_candidate())

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_malformed_pssm_values_return_neutral_score(monkeypatch):
    malformed = pd.DataFrame(
        {column: ["bad", "bad"] for column in groupe_04.REQUIRED_PSSM_COLUMNS},
        index=["A", "V"],
    )

    monkeypatch.setattr(groupe_04, "_PSSM", malformed)

    name, value = groupe_04.get_score(_make_candidate())

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_missing_pssm_columns_return_neutral_score(monkeypatch):
    missing_columns = pd.DataFrame({"P1": [1.0], "P2": [2.0]}, index=["A"])

    monkeypatch.setattr(groupe_04, "_PSSM", missing_columns)

    name, value = groupe_04.get_score(_make_candidate())

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_empty_pssm_returns_neutral_score(monkeypatch):
    monkeypatch.setattr(groupe_04, "_PSSM", pd.DataFrame())

    name, value = groupe_04.get_score(_make_candidate())

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_duplicate_pssm_index_returns_neutral_score(monkeypatch):
    duplicate_index = pd.DataFrame(
        {column: [1.0, 2.0] for column in groupe_04.REQUIRED_PSSM_COLUMNS},
        index=["A", "A"],
    )

    monkeypatch.setattr(groupe_04, "_PSSM", duplicate_index)

    name, value = groupe_04.get_score(_make_candidate())

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_non_dataframe_pssm_returns_neutral_score(monkeypatch):
    monkeypatch.setattr(groupe_04, "_PSSM", object())

    name, value = groupe_04.get_score(_make_candidate())

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


@pytest.mark.parametrize(
    ("peptide_wt", "peptide_mut"),
    [
        ("AAAAAAAA", "AAAAAAAAV"),
        ("AAAAAAAAAA", "AAAAAAAAV"),
        ("AAAAAAAAA", "AAAAAAAA"),
        ("AAAAAAAAA", "AAAAAAAAAA"),
        ("", "AAAAAAAAV"),
        ("AAAAAAAAA", ""),
        ("         ", "AAAAAAAAV"),
        ("AAAAAAAAA", "         "),
    ],
)
def test_wrong_length_peptides_return_neutral_score(peptide_wt, peptide_mut):
    candidate = Candidate(
        candidate_id="TEST_LEN",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )

    name, value = groupe_04.get_score(candidate)

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


@pytest.mark.parametrize(
    ("peptide_wt", "peptide_mut"),
    [
        ("AAAAAAAAX", "AAAAAAAAV"),
        ("AAAAAAA1A", "AAAAAAAAV"),
        ("AAAAAAA-A", "AAAAAAAAV"),
        ("AAAAAAAAA", "AAAAAA*AV"),
        ("AAAAAAAAA", "AAAAAAZAV"),
        ("AAAAAAAAA", "AAA AAAAV"),
    ],
)
def test_invalid_amino_acids_return_neutral_score(peptide_wt, peptide_mut):
    candidate = Candidate(
        candidate_id="TEST_AA",
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=1,
    )

    name, value = groupe_04.get_score(candidate)

    assert name == groupe_04.SCORE_NAME
    assert value == 0.0


def test_lowercase_peptides_are_accepted():
    candidate = Candidate(
        candidate_id="TEST_LOWER",
        peptide_wt="aaaaaaaaa",
        peptide_mut="aaaaaaaav",
        mut_pos_1based=9,
    )

    name, value = groupe_04.get_score(candidate)

    assert name == groupe_04.SCORE_NAME
    assert isinstance(value, float)
    assert not math.isnan(value)
    assert not math.isinf(value)
