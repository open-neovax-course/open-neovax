"""Benchmark — Compare Open-NeoVax pipeline ranking with NetMHCpan 4.1.

Issue #49 — Groupe 07 (Module B2 TAP transport)

Usage:
    1. Go to https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/
       Paste peptides from data/patient_zero.csv, HLA-A*02:01
       Save raw output as analysis/netmhcpan_patient_zero.txt
       (Optional) Same for REAL peptides from data/patient_real.csv
       Save as analysis/netmhcpan_patient_real.txt
    2. python analysis/groupe_07_netmhcpan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ANALYSIS_DIR = PROJECT_ROOT / "analysis"
PATIENT_ONE = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL = ANALYSIS_DIR / "scores_patient_real.csv"
PATIENT_REAL_RAW = PROJECT_ROOT / "data" / "patient_real.csv"
NETMHCPAN_ZERO = ANALYSIS_DIR / "netmhcpan_patient_zero.txt"
NETMHCPAN_REAL = ANALYSIS_DIR / "netmhcpan_patient_real.txt"

LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
ORDINAL_MAP = {"TRAP": 0, "BAD": 1, "MEDIOCRE": 2, "GOOD": 3, "GOLD": 4}
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
RANDOM_STATE = 42

# Amino acids valides pour NetMHCpan
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def is_valid_peptide(peptide: str) -> bool:
    """Vérifie si un peptide contient uniquement des acides aminés valides."""
    return bool(peptide and all(c in VALID_AA for c in peptide))


# =============================================================================
# HELPERS
# =============================================================================


def _label(raw: str) -> str:
    clean = raw.strip().upper().split("—")[0].strip()
    # Si le label est "REAL" ou "DECOY", garde-les comme tels
    if clean in {"REAL", "DECOY"}:
        return clean
    return clean if clean in {**LABEL_MAP, **ORDINAL_MAP} else "UNKNOWN"


def _label_col(df: pd.DataFrame) -> str | None:
    return next((c for c in ("label", "note") if c in df.columns), None)


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


# =============================================================================
# DATA
# =============================================================================


def load_pipeline_ranking(scores_path: Path, peptide_src: Path) -> pd.DataFrame:
    """Train RF on patient_one, score scores_path. Returns ranked DataFrame."""
    if not PATIENT_ONE.exists() or not scores_path.exists():
        print(f"[ERROR] Score matrices not found ({scores_path.name}).")
        print("  -> Run: python analysis/score_analysis.py --generate")
        sys.exit(1)

    df_tr = pd.read_csv(PATIENT_ONE)
    df_tr["_label"] = df_tr[_label_col(df_tr)].astype(str).apply(_label)
    df_tr = df_tr[df_tr["_label"].isin(LABEL_MAP)].copy()
    y_tr = df_tr["_label"].map(LABEL_MAP)
    feats = _feat_cols(df_tr)
    X_tr = df_tr[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    rf.fit(X_tr_s, y_tr)

    df_val = pd.read_csv(scores_path)
    col = _label_col(df_val)
    labels = (
        df_val[col].astype(str).apply(_label).tolist()
        if col
        else ["UNKNOWN"] * len(df_val)
    )

    peptide_map = (
        pd.read_csv(peptide_src).set_index("candidate_id")["peptide_mut"].to_dict()
    )
    peptides = [peptide_map.get(cid, "") for cid in df_val["candidate_id"].tolist()]

    X_val = df_val[_feat_cols(df_val)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_val = X_val.reindex(columns=feats, fill_value=0.0)
    probas = rf.predict_proba(scaler.transform(X_val))[:, 1]

    df = pd.DataFrame(
        {
            "candidate_id": df_val["candidate_id"].tolist(),
            "peptide_mut": peptides,
            "label": labels,
            "pipeline_score": probas,
        }
    )
    df = df.sort_values("pipeline_score", ascending=False).reset_index(drop=True)
    df["pipeline_rank"] = range(1, len(df) + 1)
    return df


def parse_netmhcpan(path: Path) -> pd.DataFrame:
    """Parse NetMHCpan 4.1 raw text output -> DataFrame(peptide, rank_el)."""
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            if len(parts) < 13:
                continue
            if parts[0] != "1" or parts[4] != "0":
                continue
            pep = parts[2]
            if not (8 <= len(pep) <= 11) or not is_valid_peptide(pep):
                continue
            try:
                rows.append({"peptide": pep, "rank_el": float(parts[12])})
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(rows).drop_duplicates(subset="peptide")
    print(f"  Parsed {len(df)} peptides from {path.name}")
    return df


# =============================================================================
# BENCHMARK
# =============================================================================


def merge_and_rank(df_pipe: pd.DataFrame, df_nmhc: pd.DataFrame) -> pd.DataFrame:
    df = df_pipe.merge(df_nmhc, left_on="peptide_mut", right_on="peptide", how="inner")
    if df.empty:
        return df
    df = df.sort_values("rank_el").reset_index(drop=True)
    df["netmhcpan_rank"] = range(1, len(df) + 1)
    df = df.sort_values("pipeline_score", ascending=False).reset_index(drop=True)
    df["pipeline_rank"] = range(1, len(df) + 1)
    df["rank_delta"] = df["netmhcpan_rank"] - df["pipeline_rank"]
    return df[
        [
            "candidate_id",
            "peptide_mut",
            "label",
            "pipeline_score",
            "pipeline_rank",
            "rank_el",
            "netmhcpan_rank",
            "rank_delta",
        ]
    ]


def print_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("  No data to display.")
        return
    print(
        f"\n  {'Candidate':12s}  {'Label':8s}  {'P.score':>7s}  {'P.rank':>6s}  "
        f"{'%Rank_EL':>8s}  {'N.rank':>6s}  {'Delta':>6s}"
    )
    print("  " + "-" * 62)
    for _, r in df.sort_values("pipeline_rank").iterrows():
        print(
            f"  {r['candidate_id']:12s}  {r['label']:8s}  {r['pipeline_score']:7.3f}  "
            f"{int(r['pipeline_rank']):6d}  {r['rank_el']:8.3f}  "
            f"{int(r['netmhcpan_rank']):6d}  {int(r['rank_delta']):+6d}"
        )


def print_disagreements(df: pd.DataFrame, top_n: int = 3) -> None:
    if df.empty:
        return
    print(
        f"\n  Top {top_n} disagreements  "
        "(+ = we rank higher than NetMHCpan | - = NetMHCpan ranks us higher)"
    )
    top = df.reindex(df["rank_delta"].abs().sort_values(ascending=False).index).head(
        top_n
    )
    for _, r in top.iterrows():
        delta = int(r["rank_delta"])
        reason = (
            "our modules (TAP, self-similarity, mutation pos) push this candidate up"
            if delta > 0
            else "NetMHCpan detects binding affinity our simplified proxy misses"
        )
        print(f"  [{r['candidate_id']}  {r['label']}  delta={delta:+d}]  {reason}")


def spearman_ic50(df_merged: pd.DataFrame) -> None:
    """Compare pipeline vs NetMHCpan correlation with measured IC50."""
    if not PATIENT_REAL_RAW.exists() or df_merged.empty:
        return
    df_ic50 = pd.read_csv(PATIENT_REAL_RAW)[["candidate_id", "ic50_nm"]]
    df = df_merged.merge(df_ic50, on="candidate_id", how="left")
    df["ic50_nm"] = pd.to_numeric(df["ic50_nm"], errors="coerce")
    df = df.dropna(subset=["ic50_nm"])
    if len(df) < 5:
        print("  [SKIP] Not enough REAL candidates with IC50.")
        return

    rho_p, p_p = spearmanr(df["pipeline_score"], df["ic50_nm"])
    rho_n, p_n = spearmanr(df["rank_el"], df["ic50_nm"])

    print(f"\n  IC50 correlation (n={len(df)} REAL candidates)")
    print(f"  Pipeline vs IC50:     rho = {rho_p:+.3f}  (p = {p_p:.4f})")
    print(f"  NetMHCpan vs IC50:    rho = {rho_n:+.3f}  (p = {p_n:.4f})")

    if abs(rho_n) > abs(rho_p):
        print(f"  NetMHCpan correlates better (gap = {abs(rho_n)-abs(rho_p):.3f})")
    else:
        print("  Our pipeline correlates as well as NetMHCpan!")


# =============================================================================
# PLOTS
# =============================================================================


def plot_comparison(df: pd.DataFrame, suffix: str = "") -> None:
    if df.empty or len(df) < 2:
        return
    colour_map = {
        "GOLD": "#e74c3c",
        "GOOD": "#e67e22",
        "BAD": "#3498db",
        "TRAP": "#9b59b6",
        "MEDIOCRE": "#95a5a6",
    }
    rho, p = spearmanr(df["pipeline_rank"], df["netmhcpan_rank"])

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, grp in df.groupby("label"):
        ax.scatter(
            grp["netmhcpan_rank"],
            grp["pipeline_rank"],
            label=label,
            color=colour_map.get(label, "#7f8c8d"),
            s=70,
            alpha=0.85,
            edgecolors="white",
            zorder=3,
        )
    n = len(df)
    ax.plot([1, n], [1, n], "k--", lw=1, alpha=0.4, label="Perfect agreement")
    ax.set_xlabel("NetMHCpan rank (1 = best binder)")
    ax.set_ylabel("Pipeline rank (1 = top candidate)")
    ax.set_title(
        f"Pipeline vs NetMHCpan — {suffix}\nSpearman rho = {rho:+.3f} (p={p:.4f})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_plot(ANALYSIS_DIR / f"netmhcpan_comparison_{suffix}.png")

    df_s = df.sort_values("rank_delta")
    colours = ["#e74c3c" if d > 0 else "#3498db" for d in df_s["rank_delta"]]
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.55), 4))
    ax.bar(df_s["candidate_id"], df_s["rank_delta"], color=colours, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Rank delta (NetMHCpan − Pipeline)")
    ax.set_title(f"Rank differences — {suffix} (Red = we rank higher)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _save_plot(ANALYSIS_DIR / f"netmhcpan_delta_{suffix}.png")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print("\nOpen-NeoVax — NetMHCpan Benchmark — Groupe 07 (B2 TAP)")
    print("=" * 60)

    # patient_zero
    print("\n[1/2] patient_zero benchmark")
    print("-" * 40)

    df_pipe_zero = load_pipeline_ranking(
        PATIENT_ZERO, PROJECT_ROOT / "data" / "patient_zero.csv"
    )
    print(f"  {len(df_pipe_zero)} candidates ranked")

    df_nmhc_zero = parse_netmhcpan(NETMHCPAN_ZERO)
    if df_nmhc_zero.empty:
        print(f"\n  [INFO] Save NetMHCpan output as {NETMHCPAN_ZERO}")
        for _, r in df_pipe_zero.iterrows():
            if is_valid_peptide(r["peptide_mut"]):
                print(f"    {r['peptide_mut']}   # {r['candidate_id']}")
        return

    df_zero = merge_and_rank(df_pipe_zero, df_nmhc_zero)
    print_table(df_zero)

    rho, p = spearmanr(df_zero["pipeline_rank"], df_zero["netmhcpan_rank"])
    print(f"\n  Spearman rho = {rho:+.3f} (p={p:.4f}) n={len(df_zero)}")
    print_disagreements(df_zero)

    df_zero.to_csv(ANALYSIS_DIR / "benchmark_patient_zero.csv", index=False)
    plot_comparison(df_zero, "patient_zero")

    # patient_real (bonus)
    print("\n[2/2] patient_real benchmark (bonus)")
    print("-" * 40)

    if not PATIENT_REAL.exists():
        print(f"  [SKIP] {PATIENT_REAL.name} not found.")
    else:
        df_pipe_real = load_pipeline_ranking(PATIENT_REAL, PATIENT_REAL_RAW)
        df_pipe_real["pipeline_rank"] = range(1, len(df_pipe_real) + 1)
        print(f"  {len(df_pipe_real)} candidates ranked")

        df_nmhc_real = parse_netmhcpan(NETMHCPAN_REAL)
        if df_nmhc_real.empty:
            print(f"  [INFO] Save NetMHCpan output as {NETMHCPAN_REAL}")
        else:
            df_real = merge_and_rank(df_pipe_real, df_nmhc_real)
            print_table(df_real)
            rho_r, p_r = spearmanr(df_real["pipeline_rank"], df_real["netmhcpan_rank"])
            print(f"\n  Spearman rho = {rho_r:+.3f} (p={p_r:.4f}) n={len(df_real)}")
            print_disagreements(df_real)
            spearman_ic50(df_real)
            df_real.to_csv(ANALYSIS_DIR / "benchmark_patient_real.csv", index=False)
            plot_comparison(df_real, "patient_real")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
