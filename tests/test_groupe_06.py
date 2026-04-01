from logic.types import Candidate
from modules.groupe_06 import get_score


def _make_candidate(peptide_mut="SLMAFTIAV"):
    return Candidate(
        candidate_id="TEST",
        peptide_wt="AAAAAAAAA",
        peptide_mut=peptide_mut,
        mut_pos_1based=5,
    )


def test_nominal_favored_cterm():
    name, value = get_score(_make_candidate("SLMAFTIAV"))  # C-term = V
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == 1.0


def test_nominal_neutral_cterm():
    name, value = get_score(_make_candidate("SLMAFTIAA"))  # C-term = A
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == 0.0


def test_nominal_disfavored_cterm():
    name, value = get_score(_make_candidate("SLMAFTIAD"))  # C-term = D
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == -1.0


def test_edge_empty_peptide():
    name, value = get_score(_make_candidate(""))
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == -2.0


def test_invalid_non_standard_amino_acid_z():
    name, value = get_score(_make_candidate("SLMAFTIAZ"))  # Z invalid
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == -2.0


def test_invalid_non_standard_amino_acid_x():
    name, value = get_score(_make_candidate("SLMAFTIAX"))  # X invalid
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == -2.0


def test_invalid_stop_symbol():
    name, value = get_score(_make_candidate("SLMAFTIA*"))  # * invalid
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == -2.0


def test_lowercase_and_spaces_are_handled():
    name, value = get_score(_make_candidate("  slmaftiav  "))
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == 1.0


def test_none_peptide_is_safe():
    name, value = get_score(_make_candidate(None))
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == -2.0


def test_short_peptide_is_safe():
    name, value = get_score(_make_candidate("A"))
    assert name == "B_proteasome_cterm"
    assert isinstance(value, float)
    assert value == 0.0
