"""Open-NeoVax NetMHCpan 4.1 benchmark (Group 11).

This script compares our pipeline ranking against REAL NetMHCpan output.

Required file:
    - analysis/netmhcpan_patient_zero.txt

Optional bonus file:
    - analysis/netmhcpan_patient_real.txt

patient_zero job reference:
    - Job ID: 69DE019E0000C966E5CB314A

Usage:
    python analysis/groupe_11_netmhcpan.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

PATIENT_ZERO_RAW = DATA_DIR / "patient_zero.csv"
PATIENT_REAL_RAW = DATA_DIR / "patient_real.csv"
SCORES_ZERO_CSV = ANALYSIS_DIR / "scores_patient_zero.csv"
SCORES_REAL_CSV = ANALYSIS_DIR / "scores_patient_real.csv"
NETMHCPAN_ZERO_TXT = ANALYSIS_DIR / "netmhcpan_patient_zero.txt"
NETMHCPAN_REAL_TXT = ANALYSIS_DIR / "netmhcpan_patient_real.txt"


def _normalize_peptide(seq: str) -> str:
    """Normalize peptide strings for sequence-based joins."""
    return seq.upper().replace("*", "X")


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════


def _load_patient_peptides(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["candidate_id", "peptide_mut", "hla_allele"]
    if "label" in df.columns:
        cols.append("label")
    if "note" in df.columns:
        cols.append("note")
    return df[cols]


def _pipeline_mean_score(scores_csv: Path) -> pd.DataFrame:
    """Aggregate all module scores into a mean score per candidate."""
    df = pd.read_csv(scores_csv)
    feat_cols = [c for c in df.columns if c not in ("candidate_id", "label")]
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["pipeline_score"] = df[feat_cols].mean(axis=1)
    return df[["candidate_id", "label", "pipeline_score"]]


def _print_peptides_for_netmhcpan(df_peps: pd.DataFrame) -> None:
    print("  Peptides (one per line) - paste into NetMHCpan 4.1, HLA-A*02:01:")
    for _, row in df_peps.iterrows():
        print(f"    {row['peptide_mut']:<12s}  # {row['candidate_id']}")


def _parse_netmhcpan_output(path: Path) -> pd.DataFrame:
    """Parse NetMHCpan text output rows into a DataFrame."""
    rows: list[dict[str, object]] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = line.split()
        if len(tokens) < 13:
            continue
        if not tokens[0].isdigit():
            continue
        if not tokens[1].startswith("HLA-"):
            continue

        peptide = _normalize_peptide(tokens[2])

        if len(tokens) >= 15 and tokens[-2] == "<=" and tokens[-1] in {"SB", "WB"}:
            score_el_token = tokens[-4]
            rank_el_token = tokens[-3]
            bind = tokens[-1]
        else:
            score_el_token = tokens[-2]
            rank_el_token = tokens[-1]
            bind = "--"

        try:
            score_el = float(score_el_token)
            rank_el = float(rank_el_token)
        except ValueError:
            continue

        rows.append(
            {
                "peptide_mut": peptide,
                "score_el": score_el,
                "netmhcpan_rank_pct": rank_el,
                "netmhcpan_bind": bind,
            }
        )

    if not rows:
        raise ValueError(f"No parseable NetMHCpan lines found in {path.name}.")

    df = pd.DataFrame(rows)
    df = df.sort_values("netmhcpan_rank_pct", ascending=True)
    return df.drop_duplicates(subset=["peptide_mut"], keep="first")


def _netmhcpan_by_candidate(
    df_peps: pd.DataFrame, df_net: pd.DataFrame
) -> pd.DataFrame:
    """Attach parsed NetMHCpan predictions to candidate IDs by peptide sequence."""
    df_map = df_peps[["candidate_id", "peptide_mut"]].copy()
    df_map["peptide_mut"] = df_map["peptide_mut"].apply(_normalize_peptide)

    merged = df_map.merge(df_net, on="peptide_mut", how="left")
    missing = merged[merged["netmhcpan_rank_pct"].isna()]["candidate_id"].tolist()
    if missing:
        print(
            "  Warning: no NetMHCpan prediction parsed for "
            f"{len(missing)} candidates: {', '.join(missing)}"
        )
    return merged.dropna(subset=["netmhcpan_rank_pct"])


# ══════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS — patient_zero
# ══════════════════════════════════════════════════════════════════════


def analyse_patient_zero() -> None:
    sep = "-" * 60
    print(sep)
    print("  NetMHCpan 4.1 benchmark - patient_zero")
    print(sep)

    if not PATIENT_ZERO_RAW.exists() or not SCORES_ZERO_CSV.exists():
        print("  patient_zero inputs not found - skipping.")
        return

    df_peps = _load_patient_peptides(PATIENT_ZERO_RAW)

    if not NETMHCPAN_ZERO_TXT.exists():
        print(
            f"  Missing {NETMHCPAN_ZERO_TXT.name} in analysis/. "
            "Download NetMHCpan output and save it there."
        )
        print()
        _print_peptides_for_netmhcpan(df_peps)
        return

    try:
        df_net = _parse_netmhcpan_output(NETMHCPAN_ZERO_TXT)
    except ValueError as exc:
        print(f"  {exc}")
        return

    df_net_by_id = _netmhcpan_by_candidate(df_peps, df_net)
    if df_net_by_id.empty:
        print("  No candidates matched parsed NetMHCpan rows.")
        return

    df_pipe = _pipeline_mean_score(SCORES_ZERO_CSV)
    df = df_pipe.merge(
        df_net_by_id[["candidate_id", "netmhcpan_rank_pct", "netmhcpan_bind"]],
        on="candidate_id",
        how="inner",
    )

    if len(df) < 3:
        print(f"  Only {len(df)} comparable candidates - skipping statistics.")
        return

    df["our_rank"] = (
        df["pipeline_score"].rank(ascending=False, method="min").astype(int)
    )
    df["netmhcpan_rank"] = (
        df["netmhcpan_rank_pct"].rank(ascending=True, method="min").astype(int)
    )
    df["delta"] = df["our_rank"] - df["netmhcpan_rank"]

    rho, p_val = stats.spearmanr(df["our_rank"], df["netmhcpan_rank"])

    print()
    print("  Comparison table (sorted by |delta|):")
    header = (
        f"  {'candidate':<10s}  {'our':>4s}  {'netmhcp':>7s}  "
        f"{'%rank':>7s}  {'bind':>4s}  label"
    )
    print(
        header
    )
    print(f"  {'-' * 10}  {'-' * 4}  {'-' * 7}  {'-' * 7}  {'-' * 4}  ---------")
    ordered = df.reindex(df["delta"].abs().sort_values(ascending=False).index)
    for _, r in ordered.iterrows():
        print(
            f"  {r['candidate_id']:<10s}  {r['our_rank']:>4d}  "
            f"{r['netmhcpan_rank']:>7d}  {r['netmhcpan_rank_pct']:>7.3f}  "
            f"{r['netmhcpan_bind']:>4s}  {r['label']}"
        )

    print()
    print("  Top 3 disagreements:")
    for _, r in ordered.head(3).iterrows():
        print(
            f"  - {r['candidate_id']}: our #{int(r['our_rank'])}, "
            f"NetMHCpan #{int(r['netmhcpan_rank'])}, delta={int(r['delta']):+d}, "
            f"label={r['label']}"
        )

    print()
    print(f"  Spearman rho (rank vs rank): {rho:+.3f} (p = {p_val:.4f})")


# ══════════════════════════════════════════════════════════════════════
#  BONUS — patient_real (69 REAL peptides with IC50)
# ══════════════════════════════════════════════════════════════════════


def analyse_patient_real() -> None:
    sep = "-" * 60
    print()
    print(sep)
    print("  Bonus - NetMHCpan vs our pipeline vs IC50 on patient_real")
    print(sep)

    if not PATIENT_REAL_RAW.exists() or not SCORES_REAL_CSV.exists():
        print("  patient_real files not found - skipping.")
        return

    if not NETMHCPAN_REAL_TXT.exists():
        print(
            f"  Optional file {NETMHCPAN_REAL_TXT.name} not found - "
            "skipping bonus NetMHCpan correlation."
        )
        return

    df_raw = pd.read_csv(PATIENT_REAL_RAW)
    df_real = df_raw[(df_raw["label"] == "REAL") & df_raw["ic50_nm"].notna()][
        ["candidate_id", "peptide_mut", "ic50_nm"]
    ].copy()
    df_real["peptide_mut"] = df_real["peptide_mut"].apply(_normalize_peptide)

    try:
        df_net = _parse_netmhcpan_output(NETMHCPAN_REAL_TXT)
    except ValueError as exc:
        print(f"  {exc}")
        return

    df_real = df_real.merge(
        df_net[["peptide_mut", "netmhcpan_rank_pct"]],
        on="peptide_mut",
        how="inner",
    )

    df_pipe = _pipeline_mean_score(SCORES_REAL_CSV)
    df = df_real.merge(df_pipe, on="candidate_id", how="inner")

    if len(df) < 5:
        print(f"  Only {len(df)} comparable REAL candidates - skipping correlations.")
        return

    rho_pipe, p_pipe = stats.spearmanr(df["pipeline_score"], df["ic50_nm"])
    rho_net, p_net = stats.spearmanr(df["netmhcpan_rank_pct"], df["ic50_nm"])

    print(f"  N REAL candidates with IC50 and NetMHCpan: {len(df)}")
    print(
        f"  Our pipeline vs IC50     :  Spearman rho = {rho_pipe:+.3f}  "
        f"(p = {p_pipe:.4f})   "
        "[expected negative]"
    )
    print(
        f"  NetMHCpan %Rank vs IC50  :  Spearman rho = {rho_net:+.3f}  "
        f"(p = {p_net:.4f})   "
        "[expected positive]"
    )


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    print("============================================================")
    print("Open-NeoVax - NetMHCpan 4.1 benchmark (Group 11)")
    print("============================================================")
    analyse_patient_zero()
    analyse_patient_real()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
