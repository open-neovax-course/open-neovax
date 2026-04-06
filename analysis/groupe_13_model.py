"""
Open-NeoVax — Modele ML global (Groupe 13)
===========================================

Pipeline ML pour le classement des candidats neo-epitopes.
On compare plusieurs modeles pour trouver la meilleure combinaison
des scores modules, puis on analyse l'importance des features.

Utilisation:
    python analysis/score_analysis.py --generate   # generer les matrices
    python analysis/groupe_13_model.py              # lancer le modele

Auteurs: Christian Essome Ndoumin, Morel Talekeudjeu Goudjou
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import (
    GradientBoostingClassifier,  # noqa: E402
    RandomForestClassifier,  # noqa: E402
)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (
    auc,  # noqa: E402
    classification_report,  # noqa: E402
    roc_curve,  # noqa: E402
)
from sklearn.model_selection import (
    cross_val_score,  # noqa: E402
    train_test_split,  # noqa: E402
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCORES_ONE = PROJECT_ROOT / "analysis" / "scores_patient_one.csv"
SCORES_ZERO = PROJECT_ROOT / "analysis" / "scores_patient_zero.csv"
SCORES_REAL = PROJECT_ROOT / "analysis" / "scores_patient_real.csv"
RAW_REAL = PROJECT_ROOT / "data" / "patient_real.csv"

LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
SEED = 42

# Colonnes qui ne sont pas des scores
META_COLS = {
    "candidate_id",
    "label",
    "ic50_nm",
    "note",
    "peptide_wt",
    "peptide_mut",
    "mut_pos_1based",
    "gene",
    "hla_allele",
}


def load_data(path):
    """Charge un CSV de scores et verifie qu'il existe."""
    if not path.exists():
        print(f"ERREUR: {path} introuvable.")
        print("  -> Lancer: python analysis/score_analysis.py --generate")
        sys.exit(1)
    return pd.read_csv(path)


def get_feature_cols(df):
    """Retourne les colonnes de scores (pas les metadonnees)."""
    return [c for c in df.columns if c not in META_COLS and not c.startswith("_")]


def prepare_train_data(df):
    """Prepare X et y pour l'entrainement (binaire: GOLD/GOOD vs BAD/TRAP)."""
    df = df.copy()
    df["_label_clean"] = df["label"].str.strip().str.upper()
    df = df[df["_label_clean"].isin(LABEL_MAP)].copy()
    y = df["_label_clean"].map(LABEL_MAP).astype(int)
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y, feat_cols


def main():
    analysis_dir = PROJECT_ROOT / "analysis"

    print("\n" + "=" * 60)
    print("Open-NeoVax — ML Pipeline (Groupe 13)")
    print("=" * 60)

    # --- Chargement des donnees ---
    print("\n[1/6] Chargement des donnees d'entrainement...")
    df_train = load_data(SCORES_ONE)
    X_train, y_train, feat_cols = prepare_train_data(df_train)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"  {len(y_train)} candidats x {len(feat_cols)} features")
    print(f"  Positifs (GOLD/GOOD): {n_pos} | Negatifs (BAD/TRAP): {n_neg}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Comparaison de modeles ---
    print("\n[2/6] Comparaison des modeles (cross-validation 5-fold)...")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED
        ),
    }

    best_name, best_acc = None, 0.0
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        print(f"  {name:25s}  accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")
        if scores.mean() > best_acc:
            best_acc = scores.mean()
            best_name = name

    print(f"\n  Meilleur modele: {best_name} ({best_acc:.3f})")

    # On entraine tous les modeles sur tout le jeu
    for model in models.values():
        model.fit(X_train_scaled, y_train)

    rf = models["Random Forest"]
    lr = models["Logistic Regression"]

    # --- Importance des features ---
    print("\n[3/6] Analyse d'importance des features...")

    # Gini (Random Forest)
    imp_gini = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(
        ascending=False
    )

    print("\n" + "=" * 60)
    print("IMPORTANCE — Random Forest (Gini)")
    print("=" * 60)
    for feat, val in imp_gini.items():
        bar = "#" * int(val * 50)
        print(f"  {feat:35s}  {val:.3f}  {bar}")

    # Coefficients Logistic Regression
    coefs = pd.Series(lr.coef_[0], index=feat_cols)
    coefs_sorted = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    print("\n" + "=" * 60)
    print("IMPORTANCE — Logistic Regression (coefficients)")
    print("=" * 60)
    for feat, val in coefs_sorted.items():
        direction = "GOOD" if val > 0 else "BAD"
        print(f"  {feat:35s}  {val:+.3f}  (predit {direction})")

    # Permutation importance
    perm = permutation_importance(
        rf, X_train_scaled, y_train, n_repeats=30, random_state=SEED
    )
    imp_perm = pd.Series(perm.importances_mean, index=feat_cols).sort_values(
        ascending=False
    )

    print("\n" + "=" * 60)
    print("IMPORTANCE — Permutation (RF)")
    print("=" * 60)
    for feat, val in imp_perm.items():
        print(f"  {feat:35s}  {val:.3f}")

    # Graphique importance
    fig, ax = plt.subplots(figsize=(10, 6))
    imp_gini.sort_values().plot(kind="barh", ax=ax, color="#2ecc71")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Importance des modules — Random Forest (Groupe 13)")
    fig.tight_layout()
    fig.savefig(analysis_dir / "groupe_13_feature_importance.png", dpi=150)
    plt.close(fig)
    print("\n  Plot sauvegarde -> groupe_13_feature_importance.png")

    # --- Classement patient_zero ---
    print("\n[4/6] Classement final (patient_zero)...")
    df_zero = load_data(SCORES_ZERO)
    ids_zero = df_zero["candidate_id"].tolist()
    labels_zero = df_zero["label"].tolist()

    X_zero = (
        df_zero[get_feature_cols(df_zero)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .reindex(columns=feat_cols, fill_value=0.0)
    )
    X_zero_scaled = scaler.transform(X_zero)

    probas = rf.predict_proba(X_zero_scaled)[:, 1]
    ranking = sorted(zip(ids_zero, probas, labels_zero), key=lambda t: -t[1])

    print("\n" + "=" * 60)
    print("CLASSEMENT FINAL — patient_zero")
    print("=" * 60)
    cand01_rank = None
    for i, (cid, prob, label) in enumerate(ranking, 1):
        tag = "  <-- CIBLE" if label == "GOLD" else ""
        if cid == "CAND_01":
            cand01_rank = i
        print(f"  {i:3d}. {cid:10s}  P(GOOD) = {prob:.3f}  {label}{tag}")

    if cand01_rank == 1:
        print("\n  SUCCES: CAND_01 est classe #1!")
    elif cand01_rank:
        print(f"\n  CAND_01 est classe #{cand01_rank}")

    # Hold-out rapide pour les metriques
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=SEED
    )
    rf_ho = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED)
    rf_ho.fit(X_tr, y_tr)

    print("\n" + "=" * 60)
    print("RAPPORT DE CLASSIFICATION (hold-out 80/20)")
    print("=" * 60)
    print(
        classification_report(
            y_te, rf_ho.predict(X_te), target_names=["BAD/TRAP", "GOOD/GOLD"]
        )
    )

    # --- Donnees reelles: correlation Spearman avec IC50 ---
    print("[5/6] Donnees reelles — correlation Spearman (score vs IC50)...")
    print("=" * 60)

    if RAW_REAL.exists() and SCORES_REAL.exists():
        raw = pd.read_csv(RAW_REAL)
        scores_real = pd.read_csv(SCORES_REAL)

        # Seulement les REAL ont des IC50 mesures
        real_raw = raw[raw["label"] == "REAL"].dropna(subset=["ic50_nm"])
        real_ids = set(real_raw["candidate_id"])

        real_merged = scores_real[scores_real["candidate_id"].isin(real_ids)].merge(
            real_raw[["candidate_id", "ic50_nm"]], on="candidate_id"
        )

        X_real = (
            real_merged[get_feature_cols(real_merged)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .reindex(columns=feat_cols, fill_value=0.0)
        )

        pipeline_scores = rf.predict_proba(scaler.transform(X_real))[:, 1]
        ic50_values = real_merged["ic50_nm"].values

        rho, pvalue = spearmanr(pipeline_scores, ic50_values)
        print(f"  Candidats REAL avec IC50: {len(real_merged)}")
        print(f"  Spearman rho:  {rho:+.3f}")
        print(f"  p-value:       {pvalue:.4f}")
        if rho < 0:
            print("  -> Correlation negative: score eleve = IC50 bas = bon binder")
        else:
            print("  -> Pas de correlation negative avec IC50")

        # Scatter plot score vs IC50
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            pipeline_scores, ic50_values, alpha=0.6, edgecolors="k", linewidth=0.5
        )
        ax.set_xlabel("Score ML (P(GOOD))")
        ax.set_ylabel("IC50 (nM)")
        ax.set_title(f"Score pipeline vs IC50 — Spearman rho={rho:.3f} (Groupe 13)")
        fig.tight_layout()
        fig.savefig(analysis_dir / "groupe_13_spearman_ic50.png", dpi=150)
        plt.close(fig)
        print("  Plot sauvegarde -> groupe_13_spearman_ic50.png")
    else:
        print("  [SKIP] Fichiers patient_real non trouves.")
    print()

    # --- REAL vs DECOY (AUC) ---
    print("[6/6] Donnees reelles — classification REAL vs DECOY (AUC)...")
    print("=" * 60)

    if SCORES_REAL.exists():
        df_rd = pd.read_csv(SCORES_REAL)
        df_rd = df_rd[df_rd["label"].isin(["REAL", "DECOY"])].copy()
        y_rd = (df_rd["label"] == "REAL").astype(int)

        X_rd = (
            df_rd[get_feature_cols(df_rd)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .reindex(columns=feat_cols, fill_value=0.0)
        )

        probas_rd = rf.predict_proba(scaler.transform(X_rd))[:, 1]
        fpr, tpr, _ = roc_curve(y_rd, probas_rd)
        auc_score = auc(fpr, tpr)

        n_real = int((y_rd == 1).sum())
        n_decoy = int((y_rd == 0).sum())
        print(f"  REAL: {n_real} | DECOY: {n_decoy}")
        print(f"  AUC: {auc_score:.3f}")
        if auc_score > 0.6:
            print("  -> Le modele distingue partiellement REAL de DECOY")
        else:
            print("  -> Le modele ne distingue pas bien REAL de DECOY")

        # Courbe ROC
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title("REAL vs DECOY — Courbe ROC (Groupe 13)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(analysis_dir / "groupe_13_real_vs_decoy_roc.png", dpi=150)
        plt.close(fig)
        print("  Plot sauvegarde -> groupe_13_real_vs_decoy_roc.png")
    else:
        print("  [SKIP] scores_patient_real.csv introuvable.")
    print()

    # --- Interpretation biologique ---
    print("=" * 60)
    print("INTERPRETATION BIOLOGIQUE")
    print("=" * 60)

    top3 = imp_gini.head(3).index.tolist()
    print(f"\n  Top 3 features (Gini): {', '.join(top3)}")

    bio_meaning = {
        "C_total_binding": "Score global HLA: somme ponderee PSSM (9 positions)",
        "C_hla_delta_binding": "Difference de binding MUT vs WT — mutations favorables",
        "C_anchoring_P2": "Ancrage en P2: L ou M en P2 = bon binder HLA-A*02:01",
        "C_hla_anchor_p9": "Ancrage P9: V ou L en C-term = stabilite complexe",
        "C_binding_quality": "Qualite globale du binding HLA",
        "D_mutation_in_window": "Mutation dans la fenetre TCR (P3-P8) = immunogenicite",
        "B_sanity_check": "Verification de coherence du candidat",
        "A_hydrophobicity_kd": "Hydrophobicite: stabilite dans le sillon MHC",
        "A_delta_wt_vs_mut": "Changement physicochimique MUT vs WT",
        "A_tcr_contact_potential": "Potentiel de contact TCR en surface",
        "B_tap_transport_score": "Transport TAP vers le reticulum endoplasmique",
        "B_proteasome_cterm": "Signal de clivage proteasomal en C-terminal",
        "B_erap_nterm_proxy": "Trimming N-terminal dans le RE",
        "D1_exact_self_similarity": "Similarite avec le soi — eviter auto-immunite",
        "A_hybrid_complexity": "Complexite de la sequence peptidique",
        "A_net_charge": "Charge nette du peptide",
        "A_mutation_surprisal": "Surprise de la mutation (information theory)",
        "A_aromaticity": "Contenu en acides amines aromatiques (F, W, Y)",
        "D_wt_presented": "Le peptide WT est-il presente par le HLA?",
    }

    print()
    for feat in top3:
        meaning = bio_meaning.get(feat, "Contribue a la qualite du candidat")
        print(f"  - {feat}: {meaning}")

    # Importance par departement
    dept_imp = {}
    for feat, val in imp_gini.items():
        dept = feat[0].upper()
        dept_imp[dept] = dept_imp.get(dept, 0.0) + val

    dept_labels = {
        "A": "Physicochimique",
        "B": "Processing",
        "C": "HLA binding",
        "D": "Securite/Self",
    }
    print("\n  Importance par departement:")
    for dept in sorted(dept_imp, key=dept_imp.get, reverse=True):
        label = dept_labels.get(dept, dept)
        print(f"    Dept {dept} ({label}): {dept_imp[dept]:.3f}")

    top_dept = max(dept_imp, key=dept_imp.get)
    print(
        f"\n  Conclusion: le departement {top_dept} ({dept_labels.get(top_dept, '')}) "
        "est le plus predictif,"
    )
    print("  ce qui est coherent avec le role central de la presentation HLA")
    print("  dans la selection des neo-epitopes.")
    print()


if __name__ == "__main__":
    main()
