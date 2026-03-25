"""
Unit tests for module B1 — groupe_06
Proteasome C-terminal proxy (Department B)

Test coverage:
    1. Nominal case          — valid peptide, favoured C-terminus
    2. Nominal case          — valid peptide, disfavoured C-terminus
    3. Edge case             — very short peptide (single AA)
    4. Edge case             — non-standard / unknown amino acid at C-terminus
    5. Invalid input         — empty peptide string
    6. Invalid input         — None as peptide_mut
    7. Invalid input         — peptide_mut is not a string (integer)
    8. Contract              — return type is always (str, float)
"""

# Standard library imports
# (none in this file)

# Third-party imports
import pytest

# Local application imports
from logic.types import Candidate
from modules.groupe_06 import SCORE_NAME, get_score

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_candidate(peptide_mut: str = "SLMAFTIAV") -> Candidate:
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )


# ---------------------------------------------------------------------------
# 1. Nominal tests
# ---------------------------------------------------------------------------


def test_nominal_favoured_cterm():
    """Peptide ending with L (hydrophobic) → positive score."""
    candidate = _make_candidate("SLMAFTIAL")  # ends with L
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value > 0.0, "L at C-terminus should be favoured (score > 0)"


def test_nominal_disfavoured_cterm():
    """Peptide ending with K (charged) → negative score."""
    candidate = _make_candidate("SLMAFTIAK")  # ends with K
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value < 0.0, "K at C-terminus should be disfavoured (score < 0)"


def test_nominal_neutral_cterm():
    """Peptide ending with G (neutral) → score == 0.0."""
    candidate = _make_candidate("SLMAFTIAG")  # ends with G
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert value == 0.0, "G at C-terminus should be neutral (score == 0.0)"


# ---------------------------------------------------------------------------
# 2. Edge-case tests
# ---------------------------------------------------------------------------


def test_single_amino_acid_peptide():
    """Single valid AA peptide — should not crash and return a float."""
    candidate = _make_candidate("V")  # V is favoured
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value > 0.0


def test_non_standard_cterm():
    """Peptide ending with a non-standard character (X) → neutral penalty."""
    candidate = _make_candidate("SLMAFTIX")  # X is not a standard AA
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)


def test_lowercase_peptide():
    """Module should handle lowercase residues gracefully (case-insensitive)."""
    candidate = _make_candidate("slmaftial")  # lowercase, ends with l
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)
    assert value > 0.0, "Lowercase 'l' should be treated as L (favoured)"


# ---------------------------------------------------------------------------
# 3. Invalid-input tests
# ---------------------------------------------------------------------------


def test_empty_peptide():
    """Empty string as peptide_mut → returns a float, does not crash."""
    candidate = _make_candidate("")
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)


def test_none_peptide(monkeypatch):
    """None as git pushpeptide_mut → returns a float, does not crash."""
    candidate = _make_candidate("PLACEHOLDER")
    monkeypatch.setattr(candidate, "peptide_mut", None)
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)


def test_integer_peptide(monkeypatch):
    """Non-string peptide_mut (int) → returns a float, does not crash."""
    candidate = _make_candidate("PLACEHOLDER")
    monkeypatch.setattr(candidate, "peptide_mut", 12345)
    name, value = get_score(candidate)
    assert name == SCORE_NAME
    assert isinstance(value, float)


# ---------------------------------------------------------------------------
# 4. Contract tests (return-type guarantees)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "peptide",
    [
        "SLMAFTIAV",  # standard 9-mer, ends V (favoured)
        "AAAAAAAAAA",  # 10-mer, ends A (moderate)
        "GILGFVFTL",  # known HLA-A*02:01 ligand, ends L
        "FMYSDFHFI",  # ends I (favoured)
        "EIYKRWII",   # ends I (favoured)
    ],
)
def test_return_type_contract(peptide):
    """get_score must always return (str, float) for any valid peptide."""
    candidate = _make_candidate(peptide)
    name, value = get_score(candidate)
    assert isinstance(name, str)
    assert isinstance(value, float)
    assert name == SCORE_NAME
