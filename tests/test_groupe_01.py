from __future__ import annotations

import math
import pytest

from logic.types import Candidate
from modules.groupe_01 import get_score, SCORE_NAME


def _make_candidate(peptide_mut: str = "SLMAFTIAV") -> Candidate:
    """Helper to create a Candidate with a specific mutated peptide."""
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )

# ── Nominal Tests ───────────────────────────────────────────────────

def test_nominal_standard_peptide():
    """Test with a standard 9-mer peptide (nominal case)."""
    name, value = get_score(_make_candidate("SLMAFTIAV"))
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert 0.0 < value <= 1.0  # Entropy should be between 0 and 1

def test_nominal_extreme_entropy():
    """Test the bounds of the Shannon entropy calculation."""
    # Minimum entropy: All amino acids are the same (should be 0.0)
    _, val_min = get_score(_make_candidate("AAAAAAAAA"))
    assert val_min == pytest.approx(0.0, abs=1e-5)

    # Maximum entropy: All amino acids are unique (should be 1.0)
    # Length 9, 9 distinct characters
    _, val_max = get_score(_make_candidate("ACDEFGHIK"))
    assert val_max == pytest.approx(1.0, abs=1e-5)

# ── Edge-Case Tests ─────────────────────────────────────────────────

def test_edge_case_length_bounds():
    """Test lengths exactly at the boundaries (8 and 11) and outside."""
    # Valid lengths
    _, val_8 = get_score(_make_candidate("ACDEFGHIK"[:8]))
    assert val_8 > 0.0
    _, val_11 = get_score(_make_candidate("ACDEFGHIKLM"))
    assert val_11 > 0.0

    # Invalid lengths (should return 0.0 penalty)
    _, val_short = get_score(_make_candidate("ACDEFGH")) # length 7
    assert val_short == 0.0
    _, val_long = get_score(_make_candidate("ACDEFGHIKLMN")) # length 12
    assert val_long == 0.0

def test_edge_case_case_insensitivity():
    """Test that the module handles lowercase letters gracefully."""
    _, val_upper = get_score(_make_candidate("SLMAFTIAV"))
    _, val_lower = get_score(_make_candidate("slmaftiav"))
    assert val_upper == val_lower

# ── Invalid-Input Tests ─────────────────────────────────────────────

def test_invalid_input_characters():
    """Test peptides containing invalid amino acids (e.g., X, *)."""
    _, value_x = get_score(_make_candidate("SLXMAFTIA"))
    assert value_x == 0.0
    
    _, value_star = get_score(_make_candidate("SL*MAFTIA"))
    assert value_star == 0.0

def test_invalid_input_empty():
    """Test with an empty peptide string."""
    _, value = get_score(_make_candidate(""))
    assert value == 0.0

def test_invalid_input_none():
    """Test with a None type instead of a string."""
    candidate = _make_candidate()
    candidate.peptide_mut = None  # Force invalid type
    _, value = get_score(candidate)
    assert value == 0.0