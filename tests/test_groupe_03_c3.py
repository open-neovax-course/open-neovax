"""Tests for the C3 module."""

from __future__ import annotations

import math
from time import perf_counter_ns
from unittest.mock import patch

from logic.types import Candidate
from modules.groupe_03_c3 import get_score


def _make_candidate() -> Candidate:
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt="AAAAAAAAA",
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


def test_score_name_not_empty():
    name, _ = get_score(_make_candidate())
    assert name != ""


def test_score_value_finite():
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


def test_valid():
    _, value = get_score(_make_candidate())
    assert value == 0.2 * 0.75 + 0.1 * 0.15 + 0.1 * 0.1


def test_mut_too_short():
    candidate = Candidate(
        candidate_id="TEST_02",
        peptide_wt="AA",
        peptide_mut="AAA",
        mut_pos_1based=2,
    )
    _, value = get_score(candidate)
    assert value == 0.0


def test_unknown_amino_acid():
    candidate = Candidate(
        candidate_id="TEST_UNKNOWN",
        peptide_wt="AAAAAAAA",
        peptide_mut="XAXAXAAA",
        mut_pos_1based=2,
    )
    _, value = get_score(candidate)
    assert value == 0.0


@patch("modules.groupe_03_c3.Path.exists")
def test_missing_pssm_file(mock_exists):
    mock_exists.return_value = False  # simulate missing file

    # erase cache
    from modules.groupe_03_c3 import _get_pssm_matrix

    _get_pssm_matrix.cache_clear()

    candidate = Candidate(
        candidate_id="TEST_MISSING_FILE",
        peptide_wt="AAAAAAAA",
        peptide_mut="AAAAAAAA",
        mut_pos_1based=1,
    )
    _, value = get_score(candidate)
    assert value == 0.0
    _get_pssm_matrix.cache_clear()


def test_caching():
    # erase cache
    from modules.groupe_03_c3 import _get_pssm_matrix

    _get_pssm_matrix.cache_clear()

    start = perf_counter_ns()
    _ = get_score(_make_candidate())
    end = perf_counter_ns()
    first_call = end - start
    start = perf_counter_ns()
    _ = get_score(_make_candidate())
    end = perf_counter_ns()
    second_call = end - start
    assert second_call < first_call

    _get_pssm_matrix.cache_clear()
