"""
Groupe 15 — Comparaison pipeline vs NetMHCpan 4.1
Issue #49
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ─────────────────────────────────────────────
# 1. DONNÉES NETMHCPAN
# ─────────────────────────────────────────────

netmhcpan_data = [
    ("CAND_01", "SLMAFTIAV", 0.049, "SB"),
    ("CAND_02", "VLMSFYLAV", 0.213, "SB"),
    ("CAND_03", "SMYDPAKAV", 0.097, "SB"),
    ("CAND_04", "ALNSYTIRV", 0.023, "SB"),
    ("CAND_05", "GLAFQYPEL", 0.188, "SB"),
    ("CAND_06", "KLTDFINAV", 0.006, "SB"),
    ("CAND_07", "YVDEHGTKL", 0.064, "SB"),
    ("CAND_08", "KADSGTFRL", 1.164, "WB"),
    ("CAND_09", "FTQEYGNAL", 1.772, "WB"),
    ("CAND_10", "YLNRQFGAV", 0.865, "WB"),
    ("CAND_11", "ILVMILMVL", 3.149, "NB"),
    ("CAND_12", "DDKREKDRK", 95.000, "NB"),
    ("CAND_13", "ALGTFINQL", 0.107, "SB"),
    ("CAND_14", "SLNRTFRLV", 1.334, "WB"),
    ("CAND_15", "GLTDAFNQP", 4.955, "NB"),
    ("CAND_16", "ALXDF*NQV", None,  "N/A"),
    ("CAND_17", "FATEDHGKL", 8.408, "NB"),
    ("CAND_18", "GLWDPFNAV", 0.017, "SB"),
]

# ─────────────────────────────────────────────
# 2. SCORES PIPELINE (depuis scores_patient_zero.csv)
# ─────────────────────────────────────────────

pipeline_data = {
    "CAND_01": {"label": "GOLD",    "C_total_binding": 9.5,   "C_binding_quality": 0.875},
    "CAND_02": {"label": "GOLD",    "C_total_binding": 9.7,   "C_binding_quality": 0.885},
    "CAND_03": {"label": "GOOD",    "C_total_binding": 6.2,   "C_binding_quality": 0.598},
    "CAND_04": {"label": "GOOD",    "C_total_binding": 8.8,   "C_binding_quality": 0.786},
    "CAND_05": {"label": "GOOD",    "C_total_binding": 6.6,   "C_binding_quality": 0.648},
    "CAND_06": {"label": "TRAP",    "C_total_binding": 7.8,   "C_binding_quality": 0.692},
    "CAND_07": {"label": "MEDIOCRE","C_total_binding": 3.4,   "C_binding_quality": 0.507},
    "CAND_08": {"label": "MEDIOCRE","C_total_binding": 1.2,   "C_binding_quality": 0.564},
    "CAND_09": {"label": "MEDIOCRE","C_total_binding": 2.6,   "C_binding_quality": 0.620},
    "CAND_10": {"label": "GOOD",    "C_total_binding": 8.5,   "C_binding_quality": 0.754},
    "CAND_11": {"label": "BAD",     "C_total_binding": 8.9,   "C_binding_quality": 0.894},
    "CAND_12": {"label": "BAD",     "C_total_binding": -12.8, "C_binding_quality": 0.208},
    "CAND_13": {"label": "NEUTRAL", "C_total_binding": 7.8,   "C_binding_quality": 0.771},
    "CAND_14": {"label": "BAD",     "C_total_binding": 8.5,   "C_binding_quality": 0.767},
    "CAND_15": {"label": "BAD",     "C_total_binding": -2.4,  "C_binding_quality": 0.554},
    "CAND_16": {"label": "BAD",     "C_total_binding": 0.0,   "C_binding_quality": 0.0},
    "CAND_17": {"label": "MEDIOCRE","C_total_binding": 0.9,   "C_binding_quality": 0.487},
    "CAND_18": {"label": "TRAP",    "C_total_binding": 7.2,   "C_binding_quality": 0.627},
}

# ─────────────────────────────────────────────
# 3. CONSTRUCTION DU TABLEAU COMPARATIF
# ─────────────────────────────────────────────

rows = []
for cand_id, peptide, rank_el, bind_level in netmhcpan_data:
    p = pipeline_data[cand_id]
    rows.append({
        "candidate_id":       cand_id,
        "label":              p["label"],
        "peptide":            peptide,
        "netmhcpan_%rank":    rank_el,
        "bind_level":         bind_level,
        "C_total_binding":    p["C_total_binding"],
        "C_binding_quality":  p["C_binding_quality"],
    })

df = pd.DataFrame(rows)

# Ranking pipeline : plus C_total_binding est élevé, meilleur le rang
df["pipeline_rank"] = df["C_total_binding"].rank(ascending=False).astype(int)

# Ranking NetMHCpan : plus %Rank est bas, meilleur le rang
df_valid = df[df["netmhcpan_%rank"].notna()].copy()
df_valid["netmhcpan_rank"] = df_valid["netmhcpan_%rank"].rank(
    ascending=True
).astype(int)

df["netmhcpan_rank"] = df["candidate_id"].map(
    df_valid.set_index("candidate_id")["netmhcpan_rank"]
)
df["delta"] = (df["pipeline_rank"] - df["netmhcpan_rank"]).abs()

# ─────────────────────────────────────────────
# 4. SPEARMAN CORRELATION
# ─────────────────────────────────────────────

df_corr = df_valid.copy()
rho, pvalue = spearmanr(
    df_corr["pipeline_rank"],
    df_corr["netmhcpan_rank"]
)

print("=" * 60)
print("  GROUPE 15 — Pipeline vs NetMHCpan 4.1 (Issue #49)")
print("=" * 60)
print(f"\nSpearman correlation : rho = {rho:.3f}  (p = {pvalue:.4f})")
if pvalue < 0.05:
    print("  → Corrélation statistiquement significative (p < 0.05)")
else:
    print("  → Corrélation non significative (p >= 0.05)")

# ─────────────────────────────────────────────
# 5. TABLEAU COMPARATIF
# ─────────────────────────────────────────────

print("\n── Tableau comparatif ──")
print(f"{'ID':<10} {'Label':<10} {'Pipeline':>10} {'NetMHCpan':>10} "
      f"{'Delta':>7} {'%Rank':>8} {'BindLevel':<10}")
print("-" * 65)

df_sorted = df.sort_values("pipeline_rank")
for _, row in df_sorted.iterrows():
    rank_str = f"{int(row['netmhcpan_rank'])}" \
        if pd.notna(row["netmhcpan_rank"]) else "N/A"
    rank_el_str = f"{row['netmhcpan_%rank']:.3f}" \
        if pd.notna(row["netmhcpan_%rank"]) else "N/A"
    delta_str = f"{int(row['delta'])}" if pd.notna(row["delta"]) else "N/A"
    print(f"{row['candidate_id']:<10} {row['label']:<10} "
          f"{int(row['pipeline_rank']):>10} {rank_str:>10} "
          f"{delta_str:>7} {rank_el_str:>8} {row['bind_level']:<10}")

# ─────────────────────────────────────────────
# 6. TOP 3 DÉSACCORDS
# ─────────────────────────────────────────────

print("\n── Top 3 désaccords ──")
top3 = df[df["delta"].notna()].nlargest(3, "delta")
for _, row in top3.iterrows():
    print(f"\n{row['candidate_id']} ({row['label']}) :")
    print(f"  Pipeline rank    : {int(row['pipeline_rank'])}")
    print(f"  NetMHCpan rank   : {int(row['netmhcpan_rank'])}")
    print(f"  Delta            : {int(row['delta'])}")
    print(f"  %Rank NetMHCpan  : {row['netmhcpan_%rank']}")
    print(f"  C_total_binding  : {row['C_total_binding']}")

# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter : pipeline rank vs netmhcpan rank
ax = axes[0]
colors = {"GOLD": "gold", "GOOD": "green", "MEDIOCRE": "orange",
          "BAD": "red", "TRAP": "purple", "NEUTRAL": "gray"}
for _, row in df_corr.iterrows():
    color = colors.get(row["label"], "blue")
    ax.scatter(row["pipeline_rank"], row["netmhcpan_rank"],
               color=color, s=100, zorder=3)
    ax.annotate(row["candidate_id"],
                (row["pipeline_rank"], row["netmhcpan_rank"]),
                textcoords="offset points", xytext=(5, 5), fontsize=7)
ax.plot([1, 17], [1, 17], "k--", alpha=0.3, label="Perfect agreement")
ax.set_xlabel("Pipeline rank (lower = better)")
ax.set_ylabel("NetMHCpan rank (lower = better)")
ax.set_title(f"Pipeline vs NetMHCpan\nSpearman ρ = {rho:.3f} (p={pvalue:.4f})")
ax.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=c, markersize=8, label=l)
    for l, c in colors.items()
] + [plt.Line2D([0], [0], linestyle='--', color='k',
                alpha=0.3, label='Perfect agreement')],
    fontsize=7, loc="upper left")

# Bar chart des deltas
ax2 = axes[1]
df_delta = df[df["delta"].notna()].sort_values("delta", ascending=False)
bar_colors = [colors.get(l, "blue") for l in df_delta["label"]]
ax2.barh(df_delta["candidate_id"], df_delta["delta"], color=bar_colors)
ax2.set_xlabel("Delta (|pipeline_rank - netmhcpan_rank|)")
ax2.set_title("Désaccords Pipeline vs NetMHCpan")
ax2.axvline(x=3, color="red", linestyle="--", alpha=0.5, label="Seuil delta=3")
ax2.legend()

plt.tight_layout()
out = "analysis/groupe_15_netmhcpan_comparison.png"
plt.savefig(out)
plt.close()
print(f"\n→ Plot sauvegardé : {out}")
print("\nTerminé !")