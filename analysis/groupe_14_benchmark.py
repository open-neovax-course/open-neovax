"""
Benchmark: pipeline ranking vs NetMHCpan 4.1 predictions.

Issue #49 -- Group 14 (Eliot RABIN)

Steps:
  1. Go to https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
  2. Input type = Peptide, allele = HLA-A*02:01
  3. Paste peptides, submit, copy output to:
       analysis/netmhcpan_patient_zero.txt
  4. python analysis/groupe_14_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
NETMHCPAN_ZERO = ANALYSIS_DIR / "netmhcpan_patient_zero.txt"
SCORES_ZERO = ANALYSIS_DIR / "scores_patient_zero.csv"
RAW_ZERO = PROJECT_ROOT / "data" / "patient_zero.csv"

# columns that are metadata, not scores
META_COLS = {
    "candidate_id",
    "peptide_wt",
    "peptide_mut",
    "mut_pos_1based",
    "gene",
    "hla_allele",
    "note",
    "label",
    "ic50_nm",
}


def parse_netmhcpan(filepath: Path) -> pd.DataFrame:
    """Read a NetMHCpan 4.1 text output file.

    Looks for data lines (start with a number) and pulls out
    the peptide, EL score/rank, BA score/rank, affinity, etc.
    """
    if not filepath.exists():
        print(f"  File not found: {filepath}")
        return pd.DataFrame()

    lines = filepath.read_text(encoding="utf-8").splitlines()

    results = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 14:
            continue
        # data lines start with an integer (Pos column)
        try:
            int(parts[0])
        except ValueError:
            continue

        peptide = parts[2]

        # EL score + rank
        try:
            score_el = float(parts[11])
            rank_el = float(parts[12])
        except (ValueError, IndexError):
            score_el = None
            rank_el = None

        # BA score, affinity, rank
        try:
            affinity_nm = float(parts[14])
            rank_ba = float(parts[15])
        except (ValueError, IndexError):
            affinity_nm = None
            rank_ba = None

        bind_level = parts[16] if len(parts) > 16 else ""

        results.append(
            {
                "peptide": peptide,
                "netmhcpan_score_el": score_el,
                "netmhcpan_rank_el": rank_el,
                "netmhcpan_affinity_nm": affinity_nm,
                "netmhcpan_rank_ba": rank_ba,
                "netmhcpan_bind_level": bind_level,
            }
        )

    if not results:
        print(f"  Could not parse any data from {filepath}")
        return pd.DataFrame()

    return pd.DataFrame(results)


def load_pipeline_scores(path: Path) -> pd.DataFrame:
    """Load the score matrix CSV and add an aggregated pipeline_score."""
    if not path.exists():
        print(f"  {path} not found -- run score_analysis.py --generate first")
        return pd.DataFrame()

    df = pd.read_csv(path)
    score_cols = [c for c in df.columns if c not in META_COLS and not c.startswith("_")]

    nums = df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # min-max normalize each module to [0,1] then average
    for col in nums.columns:
        lo, hi = nums[col].min(), nums[col].max()
        if hi > lo:
            nums[col] = (nums[col] - lo) / (hi - lo)
        else:
            nums[col] = 0.5

    df["pipeline_score"] = nums.mean(axis=1)
    return df


def show_ranking_table(merged, pipe_col, nmhc_col, name):
    """Print the comparison table + Spearman + top disagreements."""
    if merged.empty:
        print(f"  No data for {name}, skipping.\n")
        return

    # rank both sides (1 = best candidate)
    merged["our_rank"] = merged[pipe_col].rank(ascending=False).astype(int)
    merged["nmhc_rank"] = merged[nmhc_col].rank(ascending=True).astype(int)
    merged["delta"] = (merged["our_rank"] - merged["nmhc_rank"]).abs()

    rho, pval = spearmanr(merged[pipe_col], merged[nmhc_col])

    print("=" * 65)
    print(f"  {name}")
    print("=" * 65)

    # header
    print(
        f"  {'ID':>12}  {'ours':>5}  {'nmhc':>5}"
        f"  {'delta':>5}  {'%Rank':>7}  {'score':>7}"
        f"  {'label':>8}"
    )
    print(f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*5}" f"  {'-'*7}  {'-'*7}  {'-'*8}")

    for _, r in merged.sort_values("our_rank").iterrows():
        cid = r.get("candidate_id", r.get("peptide", "?"))
        lbl = r.get("label", "")
        if isinstance(lbl, str) and lbl:
            lbl = lbl.split()[0]
        print(
            f"  {str(cid):>12}  {int(r['our_rank']):>5}"
            f"  {int(r['nmhc_rank']):>5}"
            f"  {int(r['delta']):>5}"
            f"  {r[nmhc_col]:>7.3f}"
            f"  {r[pipe_col]:>7.3f}"
            f"  {str(lbl):>8}"
        )

    print(f"\n  Spearman rho = {rho:+.3f}  (p = {pval:.4f})")
    if abs(rho) < 0.3:
        print("  Weak correlation -- we measure" " different things than NetMHCpan")
    elif rho < -0.3:
        print("  Negative correlation (expected):" " our score tracks binding")
    print()

    # top 3 biggest disagreements
    worst = merged.nlargest(3, "delta")
    print("  Biggest disagreements:")
    for _, r in worst.iterrows():
        cid = r.get("candidate_id", r.get("peptide", "?"))
        pep = r.get("peptide_mut", r.get("peptide", ""))
        lbl = r.get("label", "")
        ours = int(r["our_rank"])
        theirs = int(r["nmhc_rank"])
        print(f"    {cid} ({pep}) -- ours #{ours}," f" NetMHCpan #{theirs}")
        if ours < theirs:
            print(
                "      We rank higher: pipeline catches"
                " something beyond pure binding"
            )
        else:
            print(
                "      NetMHCpan ranks higher: our PSSM" " is probably too simple here"
            )
        if isinstance(lbl, str) and lbl:
            print(f"      Label: {lbl.split()[0]}")
        print()


def main():
    print()
    print("Benchmark: Pipeline vs NetMHCpan 4.1 (Group 14)")

    print("\npatient_zero (18 candidates)\n")

    nmhc_zero = parse_netmhcpan(NETMHCPAN_ZERO)
    if nmhc_zero.empty:
        print("No NetMHCpan data for patient_zero.\n")
    else:
        raw = pd.read_csv(RAW_ZERO)
        pipe = load_pipeline_scores(SCORES_ZERO)

        if not pipe.empty:
            merged = raw[["candidate_id", "peptide_mut", "note"]].merge(
                nmhc_zero,
                left_on="peptide_mut",
                right_on="peptide",
                how="inner",
            )
            merged = merged.merge(
                pipe[["candidate_id", "pipeline_score"]],
                on="candidate_id",
                how="inner",
            )
            merged["label"] = merged["note"].apply(
                lambda x: (str(x).split()[0] if pd.notna(x) else "")
            )

            show_ranking_table(
                merged,
                "pipeline_score",
                "netmhcpan_rank_el",
                "patient_zero",
            )


if __name__ == "__main__":
    main()
