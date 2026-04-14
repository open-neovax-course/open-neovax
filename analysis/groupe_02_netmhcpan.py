"""Benchmark Open-NeoVax against NetMHCpan 4.1."""

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


def _label_from_note(note: str) -> str:
    if not isinstance(note, str) or not note.strip():
        return "UNKNOWN"
    return note.split("—", 1)[0].strip().split()[0].upper()


def _is_valid_peptide(peptide: str) -> bool:
    return bool(peptide) and all(char in VALID_AA for char in peptide)


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
        print(f"Selected dataset: {dataset_name}")


if __name__ == "__main__":
    main()
