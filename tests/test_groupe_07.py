"""Tests for the template module."""

from __future__ import annotations

import math
import pytest

from types import SimpleNamespace

from logic.types import Candidate
from modules.groupe_07 import get_score, SCORE_NAME
from logic.data_loader import load_candidates
from modules.groupe_07 import (
    get_score,
    SCORE_NAME,
    _is_valid_peptide,
    _score_cterminus,
    _score_charge,
    VALID_AA,
    MIN_LENGTH,
    MAX_LENGTH,
    CHARGE_THRESHOLD,
)



def _make_candidate() -> Candidate:
    return Candidate(
        candidate_id="TEST_01",
        peptide_wt="AAAAAAAAA",
        peptide_mut="AAAAAAAAV",
        mut_pos_1based=9,
    )


def test_returns_tuple_of_two():
    """Verify that get_score returns a tuple of exactly two elements."""
    result = get_score(_make_candidate())
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_return_types():
    """Verify that the return values have correct types."""
    name, value = get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, (int, float))


def test_score_name_not_empty():
    """Verify that the score name is not an empty string."""
    name, _ = get_score(_make_candidate())
    assert name != ""


def test_score_value_finite():
    """Verify that the score value is finite (not NaN or infinity)."""
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


# ══════════════════════════════════════════════════════════════════════
#  Nominal Tests (Biologically Expected Cases)
# ══════════════════════════════════════════════════════════════════════

def test_nominal():
    """Verify that get_score returns the correct score name."""
    name, value = get_score(_make_candidate())
    assert name == SCORE_NAME 

def test_hydrophobic_cterm_scores():
    """Hydrophobic C-terminus should score higher than charged C-terminus."""
    _, score_L = get_score(SimpleNamespace(peptide_mut="SLMAFTIAL"))
    _, score_D = get_score(SimpleNamespace(peptide_mut="SLMAFTIAD"))
    assert score_L > score_D

def test_favorable_cterminus_gives_positive_score():
    """Peptide with hydrophobic C-terminus (V) should have a positive score."""
    _, value = get_score(SimpleNamespace(peptide_mut="SLMAFTIAV"))
    assert value > 0.0, "V en C-terminus est favorable pour TAP"

def test_unfavorable_cterminus_gives_negative_score():
    """Peptide with charged C-terminus (D) should have a negative score."""
    _, value = get_score(SimpleNamespace(peptide_mut="SLMAFTIAD"))
    assert value < 0.0, "D en C-terminus est défavorisé par TAP"

def test_proline_cterminus_gives_negative_score():
    """Proline at C-terminus is the worst residue for TAP."""
    _, value = get_score(SimpleNamespace(peptide_mut="SLMAFTIAP"))
    assert value < 0.0, "P en C-terminus est fortement pénalisé"

def test_highly_charged_peptide_penalized():
    """Highly charged peptide should be penalized even with good C-terminus."""    # KKKKKKKIV : 5 K (+5 charge), C-terminus V (favorable)
    _, value_charged = get_score(SimpleNamespace(peptide_mut="KKKKKKKIV"))
    _, value_normal = get_score(SimpleNamespace(peptide_mut="SLMAFTIAV"))
    assert value_charged < value_normal, (
        "Un peptide très chargé doit scorer moins qu'un peptide normal"
    )

def test_moderate_charge_no_penalty():
    """Moderate charge (|q| <= 2) should not incur charge penalty."""    # SLMAKTIAV : 1 K (+1 charge), C-terminus V
    _, value_moderate = get_score(SimpleNamespace(peptide_mut="SLMAKTIAV"))
    _, value_neutral = get_score(SimpleNamespace(peptide_mut="SLMAFTIAV"))
    # Les deux doivent être positifs et proches (seul le C-terminus diffère peu)
    assert value_moderate > 0.0

