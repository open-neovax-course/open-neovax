"""Benchmark Open-NeoVax against NetMHCpan 4.1.

This script implements the NetMHCpan comparison issue from scratch.

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

DATA_DIR = PROJECT_ROOT / "data"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
MODULES_DIR = PROJECT_ROOT / "modules"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
META_COLUMNS = {
    "candidate_id",
    "peptide_wt",
    "peptide_mut",
    "mut_pos_1based",
    "gene",
    "hla_allele",
    "note",
    "ic50_nm",
    "label",
}

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


def _label_from_note(note: str) -> str:
    if not isinstance(note, str) or not note.strip():
        return "UNKNOWN"
    return note.split("—", 1)[0].strip().split()[0].upper()


def _is_valid_peptide(peptide: str) -> bool:
    """
    Initial checking to prevent edge cases like CAND_16 that are not valid
    """
    return bool(peptide) and all(char in VALID_AA for char in peptide)


def parse_netmhcpan_output(path: Path) -> pd.DataFrame:
    """Parse NetMHCpan txt"""
    rows: list[dict[str, object]] = []

    with open(path, encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) < 13:  ## too short lines
                continue
            if not parts[0].isdigit():  ## first tokend should be a number
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

            bind_level = ""
            if len(parts) >= 15 and parts[13] == "<=":  ## saving binding level
                bind_level = parts[14]

            rows.append(
                {
                    "identity": identity,
                    "peptide_mut": peptide,
                    "score_el": score_el,
                    "netmhcpan_rank_pct": rank_el,
                    "bind_level": bind_level or "--",
                }
            )

    if not rows:
        raise ValueError(f"No NetMHCpan rows could be parsed from {path.name}.")

    df = pd.DataFrame(rows)
    df = df.sort_values("netmhcpan_rank_pct", ascending=True)
    df = df.drop_duplicates(subset=["identity", "peptide_mut"], keep="first")
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
        netmhcpan_path = DATASETS[dataset_name]["netmhcpan"]
        if not netmhcpan_path.exists():
            print(f"Missing NetMHCpan output: {netmhcpan_path}")
            continue
        df = parse_netmhcpan_output(netmhcpan_path)
        print(f"{dataset_name}: parsed {len(df)} NetMHCpan rows")


if __name__ == "__main__":
    main()
