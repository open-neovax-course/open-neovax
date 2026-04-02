"""ML pipeline — Global scoring model (Group 14, Eliot RABIN).

We train classifiers on all the module scores to figure out the best
way to rank neo-epitope candidates, and which modules actually matter.
Also tests our model against real experimental data (IC50 + decoys).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
SCORES_ONE = PROJECT_ROOT / "analysis" / "scores_patient_one.csv"
SCORES_ZERO = PROJECT_ROOT / "analysis" / "scores_patient_zero.csv"
SCORES_REAL = PROJECT_ROOT / "analysis" / "scores_patient_real.csv"
RAW_REAL = PROJECT_ROOT / "data" / "patient_real.csv"

# We want to select GOLD/GOOD and reject BAD/TRAP
LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
ALL_LABELS = {"GOLD", "GOOD", "MEDIOCRE", "NEUTRAL", "BAD", "TRAP"}

RANDOM_STATE = 42

# Columns that aren't scores (so we skip them when building X)
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


def _clean_label(raw: str) -> str:
    """'GOLD — nice anchors ...' -> 'GOLD'. Unknown stuff -> 'UNKNOWN'."""
    token = raw.strip().upper().split("—")[0].strip()
    return token if token in ALL_LABELS else "UNKNOWN"


def _feature_cols(df):
    """Grab only score columns, skip metadata and internal stuff."""
    return [c for c in df.columns if c not in META_COLS and not c.startswith("_")]


def _load_csv(path):
    """Load a CSV or bail out with a helpful message."""
    if not path.exists():
        print(f"[ERROR] {path} not found")
        print("  -> Run: python analysis/score_analysis.py --generate")
        sys.exit(1)
    return pd.read_csv(path)


# Importance printing helpers


def _print_importance_table(imp, title):
    """Print feature importances with little ASCII bars."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    for feat, val in imp.items():
        bar = "#" * int(abs(val) * 50)
        print(f"  {feat:35s}  {val:+.3f}  {bar}")
    print()


def _save_importance_plot(imp, title, path):
    """Horizontal bar chart saved to disk."""
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {path}\n")


# =========================================================================
# Main pipeline
# =========================================================================


def main() -> None:
    analysis_dir = PROJECT_ROOT / "analysis"

    print("\n" + "=" * 60)
    print("Open-NeoVax — ML Pipeline (Group 14, Eliot RABIN)")
    print("=" * 60)

    # Load training data (patient_one)

    print("\n[1/7] Loading training data (patient_one)...")
    df_train = _load_csv(SCORES_ONE)

    # Figure out where the labels are
    label_col = "label" if "label" in df_train.columns else "note"
    df_train["_lbl"] = df_train[label_col].astype(str).apply(_clean_label)

    # Drop MEDIOCRE — too ambiguous for binary classification
    df_train = df_train[df_train["_lbl"].isin(LABEL_MAP)].copy()
    y_train = df_train["_lbl"].map(LABEL_MAP)

    feature_names = _feature_cols(df_train)
    X_train = df_train[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    print(f"  {len(y_train)} candidates x {len(feature_names)} features")
    print(f"  Positives (GOLD/GOOD): {n_pos} | Negatives (BAD/TRAP): {n_neg}\n")

    # Standardize

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Compare 3 models with 5-fold cross-validation

    print("[2/7] Comparing models...")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    print("=" * 60)
    print("CROSS-VALIDATION ACCURACY (5-fold, patient_one)")
    print("=" * 60)
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        print(f"  {name:25s}  {scores.mean():.3f}  (+/- {scores.std():.3f})")
    print()

    # Train all models on the full dataset for later use
    for model in models.values():
        model.fit(X_train_scaled, y_train)

    rf = models["Random Forest"]
    lr = models["Logistic Regression"]

    # Feature importance — which modules actually matter?

    print("[3/7] Feature importance analysis...\n")

    # Gini importance (built into Random Forest)
    imp_gini = pd.Series(rf.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )

    _print_importance_table(imp_gini, "FEATURE IMPORTANCE — Random Forest (Gini)")
    _save_importance_plot(
        imp_gini,
        "Feature importance — Random Forest (Group 14)",
        analysis_dir / "groupe_14_importance_rf.png",
    )

    # LR coefficients (positive = pushes toward GOOD/GOLD)
    coefs = pd.Series(lr.coef_[0], index=feature_names)
    imp_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
    _print_importance_table(imp_coefs, "IMPORTANCE — LR (coefficients)")

    # Permutation importance (shuffle each feature, see what breaks)
    perm_result = permutation_importance(
        rf, X_train_scaled, y_train, n_repeats=30, random_state=RANDOM_STATE
    )
    imp_perm = pd.Series(perm_result.importances_mean, index=feature_names).sort_values(
        ascending=False
    )
    _print_importance_table(imp_perm, "IMPORTANCE — Permutation (RF)")

    # Validate on patient_zero — does CAND_01 end up on top?

    print("[4/7] Validation on patient_zero...")
    df_zero = _load_csv(SCORES_ZERO)
    ids_zero = df_zero["candidate_id"].tolist()

    lcol_zero = "label" if "label" in df_zero.columns else "note"
    labels_zero = df_zero[lcol_zero].astype(str).apply(_clean_label).tolist()

    X_zero = (
        df_zero[_feature_cols(df_zero)]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    X_zero = X_zero.reindex(columns=feature_names, fill_value=0.0)
    X_zero_scaled = scaler.transform(X_zero)

    # Rank by P(GOOD/GOLD) — higher = better candidate
    probas_zero = rf.predict_proba(X_zero_scaled)[:, 1]
    ranking = sorted(zip(ids_zero, probas_zero, labels_zero), key=lambda t: -t[1])

    print("=" * 60)
    print("FINAL RANKING (patient_zero)")
    print("=" * 60)
    cand01_rank = None
    for i, (cid, prob, label) in enumerate(ranking, 1):
        tag = "  <-- TARGET" if label == "GOLD" else ""
        if cid == "CAND_01":
            cand01_rank = i
        print(f"  {i:3d}.  {cid:10s}  {prob:.3f}  {label}{tag}")

    print()
    if cand01_rank == 1:
        print("  SUCCESS: CAND_01 is ranked #1!")
    elif cand01_rank:
        print(f"  WARNING: CAND_01 is ranked #{cand01_rank} (target: #1)")
    print()

    # Quick sanity check: hold-out classification report
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    rf_ho = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    rf_ho.fit(X_tr, y_tr)
    print("=" * 60)
    print("CLASSIFICATION REPORT (hold-out 80/20)")
    print("=" * 60)
    print(
        classification_report(
            y_te, rf_ho.predict(X_te), target_names=["BAD/TRAP", "GOOD/GOLD"]
        )
    )

    # Visualizations (heatmap + ROC + confusion matrix)

    print("[5/7] Generating visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Open-NeoVax — Group 14", fontsize=13)

    # Heatmap: how each patient_zero candidate scores across modules
    ax = axes[0]
    heatmap_data = pd.DataFrame(X_zero_scaled, columns=feature_names)
    heatmap_data.index = labels_zero
    im = ax.imshow(heatmap_data.T.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(labels_zero)))
    ax.set_xticklabels(labels_zero, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title("Score heatmap (patient_zero)")
    plt.colorbar(im, ax=ax)

    # ROC curve from cross-validated predictions on training data
    ax = axes[1]
    y_proba_cv = cross_val_predict(
        RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        X_train_scaled,
        y_train,
        cv=5,
        method="predict_proba",
    )[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_proba_cv)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"RF (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (cross-validated)")
    ax.legend()

    # Confusion matrix from the hold-out split we already did
    ax = axes[2]
    cm = confusion_matrix(y_te, rf_ho.predict(X_te))
    ConfusionMatrixDisplay(cm, display_labels=["BAD/TRAP", "GOOD/GOLD"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix (hold-out 20%)")

    fig.tight_layout()
    viz_path = analysis_dir / "groupe_14_visualizations.png"
    fig.savefig(viz_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {viz_path}\n")

    # Real experimental data

    # Spearman correlation — does our score track real IC50?
    print("[6/7] Real data — Spearman correlation with IC50...")
    print("=" * 60)
    print("REAL DATA — Spearman correlation (pipeline score vs IC50)")
    print("=" * 60)

    if RAW_REAL.exists() and SCORES_REAL.exists():
        raw = pd.read_csv(RAW_REAL)
        scores_real = pd.read_csv(SCORES_REAL)

        # Only REAL candidates have measured IC50 values
        real_raw = raw[raw["label"] == "REAL"].dropna(subset=["ic50_nm"])
        real_ids = set(real_raw["candidate_id"])

        real_merged = scores_real[scores_real["candidate_id"].isin(real_ids)].merge(
            real_raw[["candidate_id", "ic50_nm"]], on="candidate_id"
        )

        X_real = (
            real_merged[_feature_cols(real_merged)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .reindex(columns=feature_names, fill_value=0.0)
        )

        pipeline_scores = rf.predict_proba(scaler.transform(X_real))[:, 1]
        ic50_values = real_merged["ic50_nm"].values

        rho, pvalue = spearmanr(pipeline_scores, ic50_values)
        print(f"  REAL candidates with IC50: {len(real_merged)}")
        print(f"  Spearman rho:  {rho:+.3f}")
        print(f"  p-value:       {pvalue:.4f}")
        # Lower IC50 = stronger binder, so we hope for negative correlation
        if rho < 0:
            print("  -> Negative correlation: higher score = lower IC50 (good!)")
        else:
            print("  -> Positive/no correlation: pipeline doesn't track IC50 well")
    else:
        print("  [SKIP] Real data files not found.")
    print()

    # REAL vs DECOY
    # can we tell real neoantigens from shuffled fakes?
    print("[7/7] Real data — REAL vs DECOY classification...")
    print("=" * 60)
    print("REAL DATA — REAL vs DECOY classification (AUC)")
    print("=" * 60)

    if SCORES_REAL.exists():
        df_real = pd.read_csv(SCORES_REAL)
        df_real = df_real[df_real["label"].isin(["REAL", "DECOY"])].copy()
        y_rd = (df_real["label"] == "REAL").astype(int)

        X_rd = (
            df_real[_feature_cols(df_real)]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .reindex(columns=feature_names, fill_value=0.0)
        )

        probas_rd = rf.predict_proba(scaler.transform(X_rd))[:, 1]
        fpr_rd, tpr_rd, _ = roc_curve(y_rd, probas_rd)
        auc_rd = auc(fpr_rd, tpr_rd)

        print(f"  REAL: {(y_rd == 1).sum()} | DECOY: {(y_rd == 0).sum()}")
        print(f"  AUC: {auc_rd:.3f}")
        if auc_rd > 0.6:
            print("  -> Model can partially distinguish REAL from DECOY")
        else:
            print("  -> Model does not distinguish REAL from DECOY well")
        print()

        # Save a dedicated ROC for this
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr_rd, tpr_rd, color="#e74c3c", lw=2, label=f"AUC = {auc_rd:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("REAL vs DECOY — ROC Curve (Group 14)")
        ax.legend()
        fig.tight_layout()
        roc_path = analysis_dir / "groupe_14_real_vs_decoy_roc.png"
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved -> {roc_path}\n")
    else:
        print("  [SKIP] scores_patient_real.csv not found.\n")

    # feature selection

    print("[BONUS] Feature selection — remove weakest features...")
    print("=" * 60)
    print("FEATURE SELECTION (drop least important, re-train)")
    print("=" * 60)

    # Try dropping the bottom 25% of features by Gini importance
    n_keep = max(3, len(feature_names) * 3 // 4)
    top_feats = imp_gini.head(n_keep).index.tolist()

    X_sel = X_train[top_feats]
    scaler_sel = StandardScaler()
    X_sel_scaled = scaler_sel.fit_transform(X_sel)

    rf_sel = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    scores_sel = cross_val_score(
        rf_sel, X_sel_scaled, y_train, cv=5, scoring="accuracy"
    )
    scores_full = cross_val_score(
        RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    print(f"  All {len(feature_names)} features:  {scores_full.mean():.3f}")
    print(f"  Top {n_keep} features:    {scores_sel.mean():.3f}")
    dropped = [f for f in feature_names if f not in top_feats]
    print(f"  Dropped: {', '.join(dropped)}")
    if scores_sel.mean() >= scores_full.mean():
        print("  -> Fewer features, same or better accuracy!")
    else:
        diff = scores_full.mean() - scores_sel.mean()
        print(f"  -> Accuracy dropped by {diff:.3f}")
    print()

    # ordinal regression

    print("[BONUS] Ordinal regression (predict label 0-4)...")
    print("=" * 60)
    print("ORDINAL REGRESSION (predict 0=TRAP .. 4=GOLD)")
    print("=" * 60)

    from sklearn.ensemble import RandomForestRegressor

    ordinal_map = {"GOLD": 4, "GOOD": 3, "MEDIOCRE": 2, "BAD": 1, "TRAP": 0}

    df_ord = _load_csv(SCORES_ONE)
    lcol_ord = "label" if "label" in df_ord.columns else "note"
    df_ord["_lbl"] = df_ord[lcol_ord].astype(str).apply(_clean_label)
    df_ord = df_ord[df_ord["_lbl"].isin(ordinal_map)].copy()
    y_ord = df_ord["_lbl"].map(ordinal_map)
    X_ord = (
        df_ord[_feature_cols(df_ord)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    X_ord = X_ord.reindex(columns=feature_names, fill_value=0.0)

    scaler_ord = StandardScaler()
    X_ord_scaled = scaler_ord.fit_transform(X_ord)

    rfr = RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    from sklearn.model_selection import cross_val_score as cvs

    r2 = cvs(rfr, X_ord_scaled, y_ord, cv=5, scoring="r2")
    print(f"  {len(y_ord)} candidates (including MEDIOCRE)")
    print(f"  R² (5-fold CV): {r2.mean():.3f} (+/- {r2.std():.3f})")

    # Train and predict on patient_zero
    rfr.fit(X_ord_scaled, y_ord)
    ord_preds = rfr.predict(scaler_ord.transform(X_zero))
    ord_ranking = sorted(zip(ids_zero, ord_preds, labels_zero), key=lambda t: -t[1])

    print("\n  Ordinal ranking (patient_zero):")
    for i, (cid, pred, lbl) in enumerate(ord_ranking, 1):
        tag = "  <-- TARGET" if lbl == "GOLD" else ""
        print(f"    {i:3d}. {cid:10s}  pred={pred:.2f}  {lbl}{tag}")
    print()

    # Biological interpretation

    print("=" * 60)
    print("BIOLOGICAL INTERPRETATION")
    print("=" * 60)

    # What each score means biologically (for the top features)
    score_meanings = {
        "C_total_binding": "Overall HLA binding.",
        "C_hla_delta_binding": "Delta binding MUT vs WT.",
        "C_anchoring_P2": "P2 anchor (HLA-A*02:01).",
        "C_hla_anchor_p9": "P9 C-terminal anchor.",
        "D_mutation_in_window": "Mutation in TCR window.",
        "B_sanity_check": "Catches broken candidates.",
        "A_hydrophobicity_kd": "MHC groove stability.",
        "A_delta_wt_vs_mut": "Physicochemical delta.",
        "A_tcr_contact_potential": "TCR surface contact.",
        "B_tap_transport_score": "TAP transport to ER.",
        "B_proteasome_cterm": "C-term cleavage signal.",
        "B_erap_nterm_proxy": "N-term ER trimming.",
        "d1_exact_self_similarity": "Self-similarity check.",
        "A_aromaticity": "Aromatic F/W/Y content.",
        "A_hybrid_complexity": "Sequence complexity.",
        "A_net_charge": "Net charge balance.",
    }

    top3 = imp_gini.head(3).index.tolist()
    print(f"\n  Top 3 features (Gini): {', '.join(top3)}")
    print()
    for feat in top3:
        meaning = score_meanings.get(feat, "contributes to candidate quality")
        print(f"  - {feat}: {meaning}")

    # Which department contributes the most overall?
    dept_totals: dict[str, float] = {}
    for feat, val in imp_gini.items():
        dept = feat[0].upper()
        dept_totals[dept] = dept_totals.get(dept, 0.0) + val

    top_dept = max(dept_totals, key=dept_totals.get)
    dept_names = {
        "A": "physicochemical",
        "B": "processing",
        "C": "HLA binding",
        "D": "safety",
    }

    print()
    print(
        f"  Overall: {dept_names.get(top_dept, top_dept)} scores (dept {top_dept})"
        " contribute the most,"
    )
    print("  confirming that MHC presentation is the primary bottleneck.")
    print("  Other departments provide useful complementary signals.")
    print()

    print("Done.\n")


if __name__ == "__main__":
    main()
