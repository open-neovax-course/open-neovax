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


# ══════════════════════════════════════════════════════════════════════
#  Internal Functions Unit Tests
# ══════════════════════════════════════════════════════════════════════

def test_is_valid_peptide():
    """Test the peptide validation function."""
    assert _is_valid_peptide("SLMAFTIAV") is True      # 9 aa, standard
    assert _is_valid_peptide("AAAAAAAA") is True       # 8 aa (min length)
    assert _is_valid_peptide("A" * 11) is True         # 11 aa (max length)
    
    assert _is_valid_peptide("") is False              # Empty
    assert _is_valid_peptide("AAAAAAA") is False       # 7 aa (too short)
    assert _is_valid_peptide("A" * 12) is False        # 12 aa (too long)
    assert _is_valid_peptide("SLMAFTXAV") is False     # Invalid character X
    assert _is_valid_peptide("SLMAFT*AV") is False     # Invalid character *
    assert _is_valid_peptide("123456789") is False     # Non-amino acid characters


def test_score_cterminus():
    """Test C-terminus scoring based on the predefined table."""
    # Favored hydrophobic residues
    assert _score_cterminus("SLMAFTIAL") == 1.0    # L
    assert _score_cterminus("SLMAFTIAI") == 1.0    # I
    assert _score_cterminus("SLMAFTIAF") == 1.0    # F
    
    # Moderately favored
    assert _score_cterminus("SLMAFTIAV") == 0.8    # V
    assert _score_cterminus("SLMAFTIAY") == 0.8    # Y
    
    # Neutral residues
    assert _score_cterminus("SLMAFTIAA") == 0.2    # A
    assert _score_cterminus("SLMAFTIAG") == 0.0    # G
    assert _score_cterminus("SLMAFTIAS") == 0.0    # S
    
    # Disfavored residues
    assert _score_cterminus("SLMAFTIAP") == -1.0   # P (worst)
    assert _score_cterminus("SLMAFTIAD") == -0.8   # D
    assert _score_cterminus("SLMAFTIAK") == -0.6   # K

def test_score_charge():
    """Test charge-based scoring."""
    # Neutral or moderate charge (|q| ≤ 2) → score = 0
    assert _score_charge("SLMAFTIAV") == 0.0       # 0 net charge
    assert _score_charge("SLMAKTIAA") == 0.0       # +1 charge
    assert _score_charge("SLMARTIAA") == 0.0       # +1 charge
    assert _score_charge("SLMADTIAA") == 0.0       # -1 charge
    assert _score_charge("KKAAVV") == 0.0          # +2 charges
    
    # Excessive charge (|q| > 2) → progressive penalty
    score_3plus = _score_charge("KKKAAAA")         # +3 charges
    assert score_3plus < 0.0
    assert abs(score_3plus - (-0.2)) < 0.0001     # Use tolerance for floating point
    
    score_5plus = _score_charge("KKKKKAAA")        # +5 charges
    assert score_5plus < 0.0
    assert abs(score_5plus - (-0.6)) < 0.0001     # Use tolerance for floating point
    
    score_7plus = _score_charge("KKKKKKKAA")       # +7 charges
    assert abs(score_7plus - (-1.0)) < 0.0001     # excess=5 → capped at -1.0
    
    # Negative excessive charge
    score_neg = _score_charge("DDDDDAAA")          # -5 charges
    assert score_neg < 0.0
    assert abs(score_neg - (-0.6)) < 0.0001


# ══════════════════════════════════════════════════════════════════════
#  Combination Tests (Weighted Scores)
# ══════════════════════════════════════════════════════════════════════

def test_weighted_combination():
    """Verify that final score correctly combines C-terminus and charge components."""
    # Case 1: Good C-terminus, neutral charge → high positive score
    _, score = get_score(SimpleNamespace(peptide_mut="SLMAFTIAL"))
    assert score >= 0.6 or abs(score - 0.6) < 0.0001  # L = 1.0 * 0.6 = 0.6
    
    # Case 2: Bad C-terminus, neutral charge → negative score
    _, score = get_score(SimpleNamespace(peptide_mut="SLMAFTIAP"))
    assert score <= -0.6 or abs(score - (-0.6)) < 0.0001  # P = -1.0 * 0.6 = -0.6
    
    # Case 3: Good C-terminus but excessive charge → reduced score
    score_bonus = get_score(SimpleNamespace(peptide_mut="SLMAFTIAL"))[1]
    score_charged = get_score(SimpleNamespace(peptide_mut="KKKKKKIAL"))[1]
    assert score_charged < score_bonus


# ══════════════════════════════════════════════════════════════════════
#  Tests edge cases
# ══════════════════════════════════════════════════════════════════════

def test_empty_peptide_returns_penalty():
    """Empty peptide should return the invalid penalty without crashing."""
    name, value = get_score(SimpleNamespace(peptide_mut=""))
    assert isinstance(value, float)
    assert value < 0.0

def test_too_short_peptide_returns_penalty():
    """Peptide shorter than MIN_LENGTH should be penalized."""
    name, value = get_score(SimpleNamespace(peptide_mut="SLMAV"))
    assert isinstance(value, float)
    assert value < 0.0

def test_too_long_peptide_returns_penalty():
    """Peptide longer than MAX_LENGTH should be penalized."""
    name, value = get_score(SimpleNamespace(peptide_mut="SLMAFTIAAAAAV"))
    assert isinstance(value, float)
    assert value < 0.0

def test_invalid_amino_acid_returns_penalty():
    """Peptide with invalid characters (X, *) should be penalized."""
    name, value = get_score(SimpleNamespace(peptide_mut="SLMAFTX*V"))
    assert isinstance(value, float)
    assert value < 0.0

def test_invalid_input_never_raises():
    """No absurd input should cause an exception."""
    absurd_inputs = ["", "XXXXX", "123456789", "*********", "A" * 20]
    for seq in absurd_inputs:
        try:
            name, value = get_score(SimpleNamespace(peptide_mut=seq))
            assert isinstance(value, float)
        except Exception as e:
            pytest.fail(f"Exception levée pour '{seq}' : {e}")



def test_length_extremes():
    """Test at the boundaries of acceptable length (8 and 11 aa)."""
    # Minimum length (8 aa)
    min_peptide = "AAAAAAAA"   # 8 aa, A at C-terminus
    _, score_min = get_score(SimpleNamespace(peptide_mut=min_peptide))
    assert isinstance(score_min, float)
    assert not math.isnan(score_min)
    
    # Maximum length (11 aa)
    max_peptide = "A" * 11      # 11 aa, A at C-terminus
    _, score_max = get_score(SimpleNamespace(peptide_mut=max_peptide))
    assert isinstance(score_max, float)
    assert not math.isnan(score_max)


def test_boundary_lengths():
    """Test just outside the acceptable length limits (7 and 12 aa)."""
    too_short = get_score(SimpleNamespace(peptide_mut="AAAAAAA"))[1]   # 7 aa
    too_long = get_score(SimpleNamespace(peptide_mut="A" * 12))[1]     # 12 aa
    
    assert too_short == -2.0
    assert too_long == -2.0


# ══════════════════════════════════════════════════════════════════════
#  Robustness Tests
# ══════════════════════════════════════════════════════════════════════

def test_handle_all_amino_acids():
    """Verify that all standard amino acids are supported."""
    for aa in VALID_AA:
        peptide = "A" * 8 + aa  # 8 A + target aa at C-terminus
        try:
            _, value = get_score(SimpleNamespace(peptide_mut=peptide))
            assert isinstance(value, float)
            assert not math.isnan(value)
        except Exception as e:
            pytest.fail(f"Failed for amino acid {aa}: {e}")


def test_peptide_with_mixed_charge():
    """Test with positive and negative charges that cancel each other."""
    neutral_peptide = "SLMAKDIAV"  # K (+1) and D (-1) = 0
    _, score_neutral = get_score(SimpleNamespace(peptide_mut=neutral_peptide))
    
    positive_peptide = "SLMAKIAV"   # K only
    _, score_positive = get_score(SimpleNamespace(peptide_mut=positive_peptide))
    
    assert abs(score_neutral - score_positive) < 0.1


def test_consistency_wt_vs_mut():
    """Verify that score depends only on mutated peptide, not wild-type."""
    candidate = SimpleNamespace(
        peptide_wt="SLMAFTIAV",
        peptide_mut="SLMAFTIAL"
    )
    _, score_mut = get_score(candidate)
    
    candidate.peptide_wt = "WWWWWWWWWW"
    _, score_mut2 = get_score(candidate)
    assert score_mut == score_mut2



# ══════════════════════════════════════════════════════════════════════
#  Mathematical Property Tests
# ══════════════════════════════════════════════════════════════════════

def test_score_monotonicity():
    """Verify that score is monotonic: better C-terminus yields better score."""
    peptides = [
        ("SLMAFTIAL", 1.0),   # L
        ("SLMAFTIAV", 0.8),   # V  
        ("SLMAFTIAA", 0.2),   # A
        ("SLMAFTIAS", 0.0),   # S
        ("SLMAFTIAK", -0.6),  # K
        ("SLMAFTIAP", -1.0),  # P
    ]
    
    scores = []
    for pep, _ in peptides:
        _, score = get_score(SimpleNamespace(peptide_mut=pep))
        scores.append(score)
    
    for i in range(len(scores)-1):
        assert scores[i] >= scores[i+1], \
            f"Order not preserved: {peptides[i][0]} ({scores[i]:.3f}) > {peptides[i+1][0]} ({scores[i+1]:.3f})"


def test_charge_penalty_progressive():
    """Verify that charge penalty increases progressively with excess charge."""
    base_pep = "SLMAFTIAL"  # L at C-terminus, 0 net charge
    
    scores = []
    charges = []
    
    for k_count in range(0, 6):
        pep = "K" * k_count + base_pep
        _, score = get_score(SimpleNamespace(peptide_mut=pep))
        scores.append(score)
        charges.append(k_count)
    
    for i in range(1, len(scores)):
        if charges[i] > CHARGE_THRESHOLD:
            assert scores[i] <= scores[i-1], \
                f"Score should decrease: charge {charges[i-1]}→{charges[i]}, score {scores[i-1]:.3f}→{scores[i]:.3f}"


def test_score_range():
    """Verify that all scores fall within the expected range [-1.0, 0.6]."""
    test_peptides = [
        "SLMAFTIAL",   # L - good
        "SLMAFTIAV",   # V - moderate
        "SLMAFTIAA",   # A - neutral
        "SLMAFTIAS",   # S - neutral
        "SLMAFTIAK",   # K - bad
        "SLMAFTIAP",   # P - worst
        "KKKKKKIAL",   # L with excessive charge
        "KKKKKKIAP",   # P with excessive charge
    ]
    
    for pep in test_peptides:
        _, score = get_score(SimpleNamespace(peptide_mut=pep))
        assert -1.0 <= score <= 0.6, f"Score {score:.3f} for {pep} outside bounds [-1.0, 0.6]"




# ══════════════════════════════════════════════════════════════════════
#  Real Data Integration Test 
# ══════════════════════════════════════════════════════════════════════


def test_try_me():
    """Display scores for candidates from patient_zero.csv file."""    
    try:
        candidates = load_candidates("data/patient_zero.csv")
        # Just test first 10 to keep output short
        for candidate in candidates[:10]:
            name, value = get_score(candidate)
            assert name == SCORE_NAME
            assert isinstance(value, float)
        print("test_try_me: verified first 10 candidates")
    except FileNotFoundError:
        pytest.skip("File data/patient_zero.csv not found")