"""Benchmark Open-NeoVax against NetMHCpan 4.1.

This script implements the NetMHCpan comparison

Usage:
    python analysis/netmhcpan_benchmark.py
    python analysis/netmhcpan_benchmark.py --dataset patient_zero
    python analysis/netmhcpan_benchmark.py --dataset patient_real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logic.data_loader import load_candidates
from logic.orchestrator import run_modules
from logic.scoring import aggregate

DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
MODULES_DIR = PROJECT_ROOT / "modules"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

DATASETS = {
    "patient_zero": {
        "csv": DATA_DIR / "patient_zero.csv",
        "netmhcpan": ANALYSIS_DIR / "netmhcpan_patient_zero.txt",
        "output": ANALYSIS_DIR / "netmhcpan_benchmark_patient_zero.csv",
    },
    "patient_real": {
        "csv": DATA_DIR / "patient_real.csv",
        "netmhcpan": ANALYSIS_DIR / "netmhcpan_patient_real.txt",
        "output": ANALYSIS_DIR / "netmhcpan_benchmark_patient_real.csv",
    },
}


def _is_valid_peptide(peptide: str) -> bool:
    """Return True only for non-empty peptides with valid amino-acid letters."""
    return bool(peptide) and all(char in VALID_AA for char in peptide)


def parse_netmhcpan_output(path: Path) -> pd.DataFrame:
    """Parse NetMHCpan text output into a dataframe."""
    rows: list[dict[str, object]] = []

    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.split()

            # Skip short lines, headers, and non-data rows.
            if len(parts) < 13:
                continue
            if not parts[0].isdigit():
                continue
            if not parts[1].startswith("HLA-"):
                continue

            peptide = parts[2].strip().upper()
            if not _is_valid_peptide(peptide):
                continue

            identity = parts[10].strip()
            try:
                score_el = float(parts[11])
                rank_el = float(parts[12])
            except ValueError:
                continue

            bind_level = "--"
            if len(parts) >= 15 and parts[13] == "<=":
                bind_level = parts[14]

            rows.append(
                {
                    "identity": identity,
                    "peptide_mut": peptide,
                    "score_el": score_el,
                    "netmhcpan_rank_pct": rank_el,
                    "bind_level": bind_level,
                }
            )

    if not rows:
        raise ValueError(f"No NetMHCpan rows could be parsed from {path.name}.")

    df = pd.DataFrame(rows)
    df = df.sort_values("netmhcpan_rank_pct", ascending=True)
    df = df.drop_duplicates(subset=["identity", "peptide_mut"], keep="first")
    return df


def score_dataset(csv_path: Path) -> pd.DataFrame:
    """Run the current pipeline and return a scored dataframe."""
    candidates = load_candidates(csv_path)
    scored = run_modules(candidates, MODULES_DIR)
    scored = aggregate(scored)

    rows: list[dict[str, object]] = []
    for candidate in scored:
        normalized_scores = {
            name: value
            for name, value in candidate.scores.items()
            if name != "total_score"
        }

        if normalized_scores:
            best_module = max(normalized_scores, key=normalized_scores.get)
            worst_module = min(normalized_scores, key=normalized_scores.get)
        else:
            best_module = ""
            worst_module = ""

        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "peptide_mut": candidate.peptide_mut,
                "hla_allele": candidate.hla_allele,
                "gene": candidate.gene,
                "note": candidate.note,
                "pipeline_score": candidate.scores.get("total_score", 0.0),
                "best_module": best_module,
                "best_module_score": normalized_scores.get(best_module, 0.0),
                "worst_module": worst_module,
                "worst_module_score": normalized_scores.get(worst_module, 0.0),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("pipeline_score", ascending=False).reset_index(drop=True)
    df["our_rank"] = range(1, len(df) + 1)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Open-NeoVax pipeline against NetMHCpan 4.1."
    )
    parser.add_argument(
        "--dataset",
        choices=["patient_zero", "patient_real", "all"],
        default="all",
        help="Dataset to benchmark",
    )
    args = parser.parse_args()

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset_name in targets:
        csv_path = DATASETS[dataset_name]["csv"]
        netmhcpan_path = DATASETS[dataset_name]["netmhcpan"]

        if not csv_path.exists():
            print(f"Missing dataset: {csv_path}")
            continue

        pipeline_df = score_dataset(csv_path)
        print(f"{dataset_name}: scored {len(pipeline_df)} candidates with pipeline")

        if not netmhcpan_path.exists():
            print(f"Missing NetMHCpan output: {netmhcpan_path}")
            continue

        netmhcpan_df = parse_netmhcpan_output(netmhcpan_path)
        print(f"{dataset_name}: parsed {len(netmhcpan_df)} NetMHCpan rows")


if __name__ == "__main__":
    main()
