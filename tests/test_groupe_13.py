"""Tests for groupe_13 module — C5: Total PSSM binding score (weighted)."""

from __future__ import annotations

import math

from logic.types import Candidate
from modules.groupe_13 import SCORE_NAME, WEIGHTS, get_score

# ── Helpers ──────────────────────────────────────────────────────────


def _make_candidate(
    peptide_mut: str = "SLMAFTIAV",
    peptide_wt: str = "SAMAFTIAV",
    candidate_id: str = "TEST_01",
    mut_pos_1based: int = 2,
) -> Candidate:
    """Create a Candidate with sensible defaults."""
    return Candidate(
        candidate_id=candidate_id,
        peptide_wt=peptide_wt,
        peptide_mut=peptide_mut,
        mut_pos_1based=mut_pos_1based,
    )


# ── Contract tests (return type & format) ────────────────────────────


def test_returns_tuple_of_two():
    """get_score must return a 2-tuple."""
    result = get_score(_make_candidate())
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_return_types():
    """First element is str, second is numeric."""
    name, value = get_score(_make_candidate())
    assert isinstance(name, str)
    assert isinstance(value, (int, float))


def test_score_name():
    """Score name must match the module constant."""
    name, _ = get_score(_make_candidate())
    assert name == SCORE_NAME
    assert name == "C_total_binding"


def test_score_value_finite():
    """Score must never be NaN or inf."""
    _, value = get_score(_make_candidate())
    assert not math.isnan(value)
    assert not math.isinf(value)


# ── Nominal tests ────────────────────────────────────────────────────


def test_cand01_gold_high_score():
    """CAND_01 (GOLD) with L@P2 and V@P9 should get a high positive score."""
    # SLMAFTIAV: L@P2=2.0, V@P9=2.0 (both best anchors)
    _, score = get_score(_make_candidate("SLMAFTIAV"))
    assert score > 5.0  # strong binder


def test_cand12_bad_low_score():
    """CAND_12 (BAD) with D@P2 and K@P9 should get a low/negative score."""
    # DDKREKDRK: D@P2=-2.5, K@P9=-2.5 (worst anchors)
    _, score = get_score(_make_candidate("DDKREKDRK"))
    assert score < -5.0  # poor binder


def test_cand01_scores_higher_than_cand12():
    """CAND_01 must score strictly higher than CAND_12 (acceptance criterion)."""
    _, score_gold = get_score(_make_candidate("SLMAFTIAV"))
    _, score_bad = get_score(_make_candidate("DDKREKDRK"))
    assert score_gold > score_bad


def test_cand15_bad_p9_proline():
    """CAND_15 with P@P9 (-3.0) should pull the total down significantly."""
    # GLTDAFNQP: P@P9=-3.0 (worst C-terminal anchor)
    _, score = get_score(_make_candidate("GLTDAFNQP"))
    assert score < 0.0  # negative overall due to terrible anchor


def test_anchor_weighting_matters():
    """Verify that P2 and P9 have weight 2.0 in the WEIGHTS dict."""
    assert WEIGHTS["P2"] == 2.0
    assert WEIGHTS["P9"] == 2.0
    for pos in ["P1", "P3", "P4", "P5", "P6", "P7", "P8"]:
        assert WEIGHTS[pos] == 1.0


# ── Edge-case tests ──────────────────────────────────────────────────


def test_empty_peptide():
    """Empty peptide returns neutral score 0.0."""
    _, value = get_score(_make_candidate(""))
    assert value == 0.0


def test_wrong_length_peptide():
    """Peptides that are not 9-mers return 0.0."""
    _, val_short = get_score(_make_candidate("SLMAFTI"))  # 7-mer
    _, val_long = get_score(_make_candidate("SLMAFTIAVAL"))  # 11-mer
    assert val_short == 0.0
    assert val_long == 0.0


def test_case_insensitivity():
    """Lowercase and uppercase should produce the same score."""
    _, upper = get_score(_make_candidate("SLMAFTIAV"))
    _, lower = get_score(_make_candidate("slmaftiav"))
    assert upper == lower


# ── Invalid-input tests ──────────────────────────────────────────────


def test_invalid_amino_acids():
    """Peptide with non-standard characters (X, *) returns 0.0."""
    _, val_x = get_score(_make_candidate("SLXMAFTIA"))
    assert val_x == 0.0

    _, val_star = get_score(_make_candidate("SL*MAFTIA"))
    assert val_star == 0.0


def test_none_peptide():
    """None peptide must not crash; returns 0.0."""
    c = _make_candidate()
    c.peptide_mut = None  # type: ignore[assignment]
    _, value = get_score(c)
    assert value == 0.0


def test_numeric_peptide():
    """Numeric string returns 0.0 (not valid amino acids)."""
    _, value = get_score(_make_candidate("123456789"))
    assert value == 0.0
