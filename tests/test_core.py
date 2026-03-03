"""Tests for the Open-NeoVax core: data_loader, orchestrator, scoring."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from logic.data_loader import load_candidates
from logic.orchestrator import discover_modules, run_modules
from logic.scoring import aggregate, compute_total_scores, normalize_scores
from logic.types import Candidate

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_candidate(cid: str = "C1", scores: dict | None = None) -> Candidate:
    return Candidate(
        candidate_id=cid,
        peptide_wt="AAAAAAAAA",
        peptide_mut="AAAAAAAAV",
        mut_pos_1based=9,
        scores=scores or {},
    )


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.csv"
    p.write_text(textwrap.dedent(content).strip())
    return p


# ═══════════════════════════════════════════════════════════════════
#  TestDataLoader
# ═══════════════════════════════════════════════════════════════════


class TestDataLoader:
    def test_load_valid_csv(self, tmp_path):
        csv = _write_csv(
            tmp_path,
            """\
            candidate_id,peptide_wt,peptide_mut,mut_pos_1based
            C1,AAAA,AAAV,4
            C2,BBBB,BBBV,4
            """,
        )
        candidates = load_candidates(csv)
        assert len(candidates) == 2
        assert candidates[0].candidate_id == "C1"
        assert candidates[1].peptide_mut == "BBBV"

    def test_missing_columns_raises(self, tmp_path):
        csv = _write_csv(
            tmp_path,
            """\
            candidate_id,peptide_wt
            C1,AAAA
            """,
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            load_candidates(csv)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_candidates("/nonexistent/path.csv")

    def test_optional_columns_absent(self, tmp_path):
        csv = _write_csv(
            tmp_path,
            """\
            candidate_id,peptide_wt,peptide_mut,mut_pos_1based
            C1,AAAA,AAAV,4
            """,
        )
        candidates = load_candidates(csv)
        assert candidates[0].gene == ""
        assert candidates[0].hla_allele == ""

    def test_load_patient_zero(self):
        csv_path = DATA_DIR / "patient_zero.csv"
        if not csv_path.exists():
            pytest.skip("patient_zero.csv not found")
        candidates = load_candidates(csv_path)
        assert len(candidates) == 18
        assert candidates[0].candidate_id == "CAND_01"


# ═══════════════════════════════════════════════════════════════════
#  TestOrchestrator
# ═══════════════════════════════════════════════════════════════════


class TestOrchestrator:
    def test_discover_modules_empty(self, tmp_path):
        d = tmp_path / "mods"
        d.mkdir()
        (d / "__init__.py").write_text("")
        assert discover_modules(d) == []

    def test_discover_modules_skips_template_and_private(self, tmp_path):
        d = tmp_path / "mods"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "template_module.py").write_text("")
        (d / "_hidden.py").write_text("")
        (d / "good_module.py").write_text("def get_score(c): return ('test', 1.0)\n")
        names = discover_modules(d)
        assert names == ["good_module"]

    def test_run_with_template_module(self):
        """template_module is ignored by discover, so no scores are added."""
        candidates = [_make_candidate()]
        result = run_modules(candidates, "modules")
        assert result[0].scores == {}

    def test_broken_module_does_not_crash(self, tmp_path):
        d = tmp_path / "mods"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "broken.py").write_text("raise RuntimeError('boom')\n")

        candidates = [_make_candidate()]
        result = run_modules(candidates, d)
        assert result[0].scores == {}

    def test_module_bad_return_ignored(self, tmp_path):
        d = tmp_path / "mods"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "bad_return.py").write_text("def get_score(c): return 'not a tuple'\n")

        candidates = [_make_candidate()]
        result = run_modules(candidates, d)
        assert result[0].scores == {}

    def test_working_module(self, tmp_path):
        d = tmp_path / "mods"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "good.py").write_text("def get_score(c): return ('test_score', 42.0)\n")

        candidates = [_make_candidate()]
        result = run_modules(candidates, d)
        assert result[0].scores["test_score"] == 42.0


# ═══════════════════════════════════════════════════════════════════
#  TestScoring
# ═══════════════════════════════════════════════════════════════════


class TestScoring:
    def test_normalize_simple(self):
        c1 = _make_candidate("C1", {"s": 0.0})
        c2 = _make_candidate("C2", {"s": 10.0})
        normalize_scores([c1, c2])
        assert c1.scores["s"] == 0.0
        assert c2.scores["s"] == 1.0

    def test_normalize_constant_gives_half(self):
        c1 = _make_candidate("C1", {"s": 5.0})
        c2 = _make_candidate("C2", {"s": 5.0})
        normalize_scores([c1, c2])
        assert c1.scores["s"] == 0.5
        assert c2.scores["s"] == 0.5

    def test_equal_weights(self):
        c1 = _make_candidate("C1", {"a": 1.0, "b": 0.0})
        c2 = _make_candidate("C2", {"a": 0.0, "b": 1.0})
        compute_total_scores([c1, c2])
        assert c1.scores["total_score"] == pytest.approx(0.5)
        assert c2.scores["total_score"] == pytest.approx(0.5)

    def test_custom_weights(self):
        c1 = _make_candidate("C1", {"a": 1.0, "b": 0.0})
        weights = {"a": 0.8, "b": 0.2}
        compute_total_scores([c1], weights)
        assert c1.scores["total_score"] == pytest.approx(0.8)

    def test_sort_descending(self):
        c1 = _make_candidate("C1", {"s": 0.2})
        c2 = _make_candidate("C2", {"s": 0.8})
        c3 = _make_candidate("C3", {"s": 0.5})
        result = compute_total_scores([c1, c2, c3])
        assert result[0].candidate_id == "C2"
        assert result[-1].candidate_id == "C1"


# ═══════════════════════════════════════════════════════════════════
#  TestEndToEnd
# ═══════════════════════════════════════════════════════════════════


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """load -> run_modules -> aggregate end-to-end."""
        csv = _write_csv(
            tmp_path,
            """\
            candidate_id,peptide_wt,peptide_mut,mut_pos_1based
            C1,AAAA,AAAV,4
            C2,BBBB,BBBV,4
            """,
        )

        d = tmp_path / "mods"
        d.mkdir()
        (d / "__init__.py").write_text("")
        (d / "test_mod.py").write_text(
            "def get_score(c): return ('len_score', float(len(c.peptide_mut)))\n"
        )

        candidates = load_candidates(csv)
        candidates = run_modules(candidates, d)
        candidates = aggregate(candidates)

        assert len(candidates) == 2
        assert "total_score" in candidates[0].scores
        assert "len_score" in candidates[0].scores

    def test_patient_zero_pipeline(self):
        """Full pipeline on patient_zero.csv with the real modules/ directory."""
        csv_path = DATA_DIR / "patient_zero.csv"
        if not csv_path.exists():
            pytest.skip("patient_zero.csv not found")

        candidates = load_candidates(csv_path)
        candidates = run_modules(candidates, "modules")
        candidates = aggregate(candidates)

        assert len(candidates) == 18
        for c in candidates:
            assert "total_score" in c.scores
