"""Tests for module A7 — Aromatic amino acid content (group 14)."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules.groupe_14 import get_score


def _make_candidate(peptide_mut: str = "SLMAFTIAV") -> Candidate:
    """Quick helper so we don't repeat the Candidate boilerplate."""
    return Candidate(
        candidate_id="TEST",
        peptide_wt="SLMAATIAV",
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )


# Nominal cases


def test_nominal_cand01():
    """SLMAFTIAV has one F at position 5, so 1/9 aromatic."""
    name, value = get_score(_make_candidate("SLMAFTIAV"))
    assert name == "A_aromaticity"
    assert math.isclose(value, 1 / 9, rel_tol=1e-9)


def test_all_aromatic():
    """A peptide made entirely of F/W/Y should hit 1.0."""
    _, value = get_score(_make_candidate("FWYFWYFWY"))
    assert math.isclose(value, 1.0, rel_tol=1e-9)


def test_mixed_aromatics():
    """FAYGLWATV has F, Y, W → 3 out of 9."""
    _, value = get_score(_make_candidate("FAYGLWATV"))
    assert math.isclose(value, 3 / 9, rel_tol=1e-9)


# Edge cases


def test_no_aromatics():
    """All alanines → 0.0, nothing aromatic in sight."""
    _, value = get_score(_make_candidate("AAAAAAAAA"))
    assert value == 0.0


def test_short_peptide():
    """Shorter than the 8-11 aa range but should still work fine."""
    _, value = get_score(_make_candidate("FW"))
    assert math.isclose(value, 1.0, rel_tol=1e-9)


def test_single_residue():
    """Just one Y → 1.0 (edge but valid)."""
    _, value = get_score(_make_candidate("Y"))
    assert math.isclose(value, 1.0, rel_tol=1e-9)


# Invalid inputs


def test_empty_peptide():
    """Empty string should give 0.0, not crash."""
    name, value = get_score(_make_candidate(""))
    assert name == "A_aromaticity"
    assert value == 0.0


def test_whitespace_only():
    """Spaces only → treated as empty."""
    _, value = get_score(_make_candidate("   "))
    assert value == 0.0


def test_non_standard_characters():
    """FX*A → 1 aromatic (F) out of 4 total characters = 0.25."""
    _, value = get_score(_make_candidate("FX*A"))
    assert math.isclose(value, 1 / 4, rel_tol=1e-9)


def test_all_invalid_characters():
    """Nothing valid at all → 0.0."""
    _, value = get_score(_make_candidate("X*123"))
    assert value == 0.0


def test_return_types():
    """Basic contract: must return (str, float)."""
    name, value = get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, float)


def test_score_finite():
    """No NaN or inf sneaking through."""
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)
