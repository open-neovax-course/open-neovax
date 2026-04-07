"""ML pipeline — Global scoring model for neoepitope ranking.

Objective: learn the optimal combination of module scores to distinguish
good candidates (GOLD, GOOD) from bad ones (BAD, TRAP), and identify
which scoring modules are the most predictive.

Group 07 — Module B2 (TAP transport proxy)

Usage:
    python analysis/groupe_07_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
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

ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# =============================================================================
# CONSTANTS
# =============================================================================

PATIENT_ONE = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL = ANALYSIS_DIR / "scores_patient_real.csv"
PATIENT_REAL_RAW = PROJECT_ROOT / "data" / "patient_real.csv"

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

def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


# =============================================================================
# HELPERS
# =============================================================================


def _label(raw: str) -> str:
    """Extract label keyword from raw string (handles annotated notes)."""
    clean = raw.strip().upper().split("—")[0].strip()
    return clean if clean in {**LABEL_MAP, **ORDINAL_MAP} else "UNKNOWN"


def _label_col(df: pd.DataFrame) -> str | None:
    """Return first available label column name."""
    return next((c for c in ("label", "note") if c in df.columns), None)


def _feat_cols(df: pd.DataFrame) -> list[str]:
    """Return score feature columns (exclude metadata)."""
    return [c for c in df.columns if c not in META_COLS]


def _save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}")


# =============================================================================
# DATA LOADING
# =============================================================================


def load_training(ordinal: bool = False) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load patient_one scores. Binary (MEDIOCRE excluded) or ordinal (0-4)."""
    if not PATIENT_ONE.exists():
        print(f"[ERROR] {PATIENT_ONE} not found.")
        print("  -> Run: python analysis/score_analysis.py --generate")
        sys.exit(1)

    df = pd.read_csv(PATIENT_ONE)
    col = _label_col(df)
    df["_label"] = df[col].astype(str).apply(_label)

    label_map = ORDINAL_MAP if ordinal else LABEL_MAP
    df = df[df["_label"].isin(label_map)].copy()

    y = df["_label"].map(label_map)
    feats = _feat_cols(df)
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y, feats


def load_validation(
    train_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load patient_zero scores aligned to training columns."""
    if not PATIENT_ZERO.exists():
        print(f"[ERROR] {PATIENT_ZERO} not found.")
        sys.exit(1)

    df = pd.read_csv(PATIENT_ZERO)
    ids = df["candidate_id"].tolist()
    col = _label_col(df)
    labels = (
        df[col].astype(str).apply(_label).tolist() if col else ["UNKNOWN"] * len(df)
    )

    X = df[_feat_cols(df)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.reindex(columns=train_columns, fill_value=0.0)
    return X, ids, labels


def load_patient_real(train_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patient_real scores and merge IC50 from raw file."""
    if not PATIENT_REAL.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(PATIENT_REAL)

    if PATIENT_REAL_RAW.exists():
        df_raw = pd.read_csv(PATIENT_REAL_RAW)[["candidate_id", "ic50_nm"]]
        df = df.merge(df_raw, on="candidate_id", how="left")

    X = df[_feat_cols(df)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.reindex(columns=train_columns, fill_value=0.0)
    return X, df


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================


def importance_rf(model: RandomForestClassifier, feature_names: list[str]) -> pd.Series:
    """Built-in Gini importance from Random Forest."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )


def importance_lr(model: LogisticRegression, feature_names: list[str]) -> pd.Series:
    """Absolute coefficients from Logistic Regression."""
    coefs = pd.Series(model.coef_[0], index=feature_names)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


def importance_permutation(
    model, X_scaled: np.ndarray, y: pd.Series, feature_names: list[str]
) -> pd.Series:
    """Permutation importance — model-agnostic."""
    result = permutation_importance(
        model, X_scaled, y, n_repeats=30, random_state=RANDOM_STATE
    )
    return pd.Series(result.importances_mean, index=feature_names).sort_values(
        ascending=False
    )


def print_importances(imp: pd.Series, title: str, top_n: int = 10) -> None:
    print(f"\n{title}")
    for feat, val in imp.head(top_n).items():
        bar = "#" * int(abs(val) * 50)
        print(f"  {feat:35s}  {val:+.3f}  {bar}")
    print()


# =============================================================================
# RANKING
# =============================================================================


def print_ranking(
    model, X_scaled: np.ndarray, ids: list[str], labels: list[str], title: str
) -> None:
    """Print candidates ranked by predicted probability of being GOOD/GOLD."""
    probas = model.predict_proba(X_scaled)[:, 1]
    ranking = sorted(zip(ids, probas, labels), key=lambda x: -x[1])

    print(f"\n- {title}\n")
    cand01_rank = None
    for i, (cid, prob, label) in enumerate(ranking, 1):
        marker = "  <-- TARGET" if label == "GOLD" else ""
        if cid == "CAND_01":
            cand01_rank = i
        print(f"  {i:3d}.  {cid:10s}  {prob:.3f}  {label}{marker}")

    print()
    if cand01_rank == 1:
        print("  SUCCESS: CAND_01 is ranked #1!")
    elif cand01_rank:
        print(f"  WARNING: CAND_01 is ranked #{cand01_rank} (target: #1)")
    print()


# =============================================================================
# BONUS 1 — FEATURE SELECTION
# =============================================================================


def feature_selection(
    X_scaled: np.ndarray, y: pd.Series, feature_names: list[str], imp: pd.Series
) -> None:
    """Try removing least important features and compare accuracy."""
    print("- Feature selection (remove least important)")
    n = len(feature_names)
    top_feats = imp.index.tolist()

    for label, k in {
        "All features": n,
        "Top 75%": max(1, int(n * 0.75)),
        "Top 50%": max(1, int(n * 0.50)),
        "Top 25%": max(1, int(n * 0.25)),
    }.items():
        selected = [feature_names.index(f) for f in top_feats[:k]]
        scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=RANDOM_STATE
            ),
            X_scaled[:, selected],
            y,
            cv=5,
            scoring="accuracy",
        )
        print(
            f"  {label:15s} ({k:2d} features)  acc = {scores.mean():.3f} +/- {scores.std():.3f}"
        )
    print()


# =============================================================================
# BONUS 2 — VISUALIZATIONS
# =============================================================================


def visualizations(
    rf: RandomForestClassifier,
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    X_zero_scaled: np.ndarray,
    ids_zero: list[str],
    feature_names: list[str],
    y_te: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Generate heatmap, ROC curve and confusion matrix."""
    print("- Visualizations")

    # Heatmap (patient_zero)
    fig, ax = plt.subplots(figsize=(12, 8))
    df_plot = pd.DataFrame(X_zero_scaled, columns=feature_names, index=ids_zero)
    im = ax.imshow(df_plot.T.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(ids_zero)))
    ax.set_xticklabels(ids_zero, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("Score heatmap (patient_zero)")
    plt.colorbar(im, ax=ax)
    _save_plot(ANALYSIS_DIR / "score_heatmap.png")

    # ROC curve (cross-validated on patient_one)
    y_proba_cv = cross_val_predict(
        rf, X_train_scaled, y_train, cv=5, method="predict_proba"
    )[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_proba_cv)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"RF (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (cross-validated, patient_one)")
    ax.legend()
    _save_plot(ANALYSIS_DIR / "roc_curve.png")

    # Confusion matrix (hold-out)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["BAD/TRAP", "GOOD/GOLD"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (hold-out 20%)")
    _save_plot(ANALYSIS_DIR / "confusion_matrix.png")


# =============================================================================
# BONUS 3 — HYPERPARAMETER TUNING
# =============================================================================


def hyperparameter_tuning(X_scaled: np.ndarray, y: pd.Series) -> None:
    """Try to beat 90% cross-validation accuracy with tuned Random Forest."""
    print("- Hyperparameter tuning (Random Forest)")
    best_score, best_params = 0.0, {}

    for n_est in (100, 200, 300):
        for depth in (3, 5, 7, None):
            scores = cross_val_score(
                RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth, random_state=RANDOM_STATE
                ),
                X_scaled,
                y,
                cv=5,
                scoring="accuracy",
            )
            mean = scores.mean()
            marker = "  <-- best so far" if mean > best_score else ""
            print(
                f"  n_estimators={n_est:3d}  max_depth={str(depth) if depth else 'None':4s}  acc={mean:.3f} +/- {scores.std():.3f}{marker}"
            )
            if mean > best_score:
                best_score = mean
                best_params = {"n_estimators": n_est, "max_depth": depth}

    print(
        f"\n  Best accuracy: {best_score:.3f} -> Beat 90%? {'YES' if best_score >= 0.90 else 'NOT YET'}"
    )
    print(f"  Best params  : {best_params}\n")


# =============================================================================
# BONUS 4 — ORDINAL REGRESSION
# =============================================================================


def ordinal_regression(
    X_zero_raw: pd.DataFrame, ids_zero: list[str], labels_zero: list[str]
) -> None:
    """Predict ordinal label 0-4 instead of binary classification."""
    X_ord, y_ord, ord_feats = load_training(ordinal=True)
    if X_ord.empty:
        print("  [SKIP] Could not load ordinal data.\n")
        return

    scaler_ord = StandardScaler()
    X_ord_scaled = scaler_ord.fit_transform(X_ord)

    reg = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE
    )
    mae = -cross_val_score(
        reg, X_ord_scaled, y_ord, cv=5, scoring="neg_mean_absolute_error"
    ).mean()

    print(f"  Candidates (all labels): {len(y_ord)}")
    print(f"  Label distribution     : {y_ord.value_counts().sort_index().to_dict()}")
    print(f"  MAE (5-fold CV)        : {mae:.3f}  (scale: 0=TRAP ... 4=GOLD)")
    print(
        f"  {'Good result: < 0.5 grade off.' if mae < 0.5 else 'Moderate result: consider tuning.'}\n"
    )

    reg.fit(X_ord_scaled, y_ord)
    preds_train = reg.predict(X_ord_scaled)
    inv_map = {v: k for k, v in ORDINAL_MAP.items()}

    print("  Sample predictions — patient_one (first 10):")
    print(f"  {'Actual':10s}  {'Predicted':10s}  {'Score':>5s}  {'Error':>6s}")
    print("  " + "-" * 40)
    for actual, pred in list(zip(y_ord, preds_train))[:10]:
        print(
            f"  {inv_map.get(int(round(actual)), '?'):10s}  {inv_map.get(int(round(pred)), '?'):10s}  {pred:5.2f}  {abs(actual - pred):6.2f}"
        )

    # Ordinal ranking on patient_zero
    X_zero = X_zero_raw.reindex(columns=ord_feats, fill_value=0.0)
    preds_val = reg.predict(scaler_ord.transform(X_zero))
    ranking = sorted(zip(ids_zero, preds_val, labels_zero), key=lambda x: -x[1])

    print("\n  Ordinal ranking — patient_zero:")
    print(f"  {'#':>3}  {'ID':10s}  {'Score':>7s}  Label")
    print("  " + "-" * 35)
    cand01_rank = None
    for i, (cid, score, label) in enumerate(ranking, 1):
        marker = "  <-- TARGET" if label == "GOLD" else ""
        if cid == "CAND_01":
            cand01_rank = i
        print(f"  {i:3}.  {cid:10s}  {score:5.2f}/4  {label}{marker}")

    print()
    if cand01_rank == 1:
        print("  SUCCESS: CAND_01 is ranked #1 with ordinal regression!")
    elif cand01_rank:
        print(f"  WARNING: CAND_01 is ranked #{cand01_rank} (target: #1)")
    print()


# =============================================================================
# PATIENT_REAL EVALUATION
# =============================================================================


def evaluate_patient_real(
    rf: RandomForestClassifier,
    gb: GradientBoostingClassifier,
    scaler: StandardScaler,
    feature_names: list[str],
) -> dict:
    """Evaluate on patient_real: Spearman correlation with IC50 + REAL vs DECOY AUC."""
    if not PATIENT_REAL.exists():
        print(f"  [SKIP] {PATIENT_REAL} not found.\n")
        return {}

    X_real, df = load_patient_real(feature_names)
    if X_real.empty:
        return {}

    X_real_scaled = scaler.transform(X_real)
    df = df.copy()
    df["rf_score"] = rf.predict_proba(X_real_scaled)[:, 1]
    df["gb_score"] = gb.predict_proba(X_real_scaled)[:, 1]

    results = {}

    # Task 1: Spearman correlation with IC50 (REAL only)
    print("  Task 1 — Spearman correlation with IC50")
    print("  " + "-" * 40)
    real_mask = df["label"].astype(str).str.upper().str.startswith("REAL")
    df_real = df[real_mask].copy()
    df_real["ic50_nm"] = pd.to_numeric(
        df_real.get("ic50_nm", pd.Series(dtype=float)), errors="coerce"
    )
    df_real = df_real.dropna(subset=["ic50_nm"])

    print(f"  REAL candidates with valid IC50: {len(df_real)}")

    if len(df_real) > 5:
        rho_rf, p_rf = spearmanr(df_real["rf_score"], df_real["ic50_nm"])
        rho_gb, p_gb = spearmanr(df_real["gb_score"], df_real["ic50_nm"])
        results.update({"rho_rf": rho_rf, "p_rf": p_rf, "rho_gb": rho_gb, "p_gb": p_gb})

        print(f"  RF : rho = {rho_rf:+.3f}  (p = {p_rf:.4f})")
        print(f"  GB : rho = {rho_gb:+.3f}  (p = {p_gb:.4f})")
        print(
            "  Note: negative rho = higher score -> lower IC50 -> biologically valid."
            if rho_rf < 0 and p_rf < 0.05
            else (
                "  No significant correlation (expected: modules are proxies)."
                if p_rf >= 0.05
                else f"  Unexpected direction: rho = {rho_rf:+.3f}"
            )
        )

        # Scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            df_real["rf_score"],
            df_real["ic50_nm"],
            color="#e74c3c",
            alpha=0.7,
            s=60,
            edgecolors="white",
        )
        for _, row in df_real.iterrows():
            ax.annotate(
                row["candidate_id"],
                (row["rf_score"], row["ic50_nm"]),
                fontsize=6,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlabel("Pipeline RF score (higher = more likely GOOD/GOLD)")
        ax.set_ylabel("Experimental IC50 (nM, lower = stronger binder)")
        ax.set_title(
            f"Pipeline score vs IC50 — patient_real\nSpearman rho = {rho_rf:+.3f}  (p = {p_rf:.4f})"
        )
        ax.grid(True, alpha=0.3)
        _save_plot(ANALYSIS_DIR / "patient_real_spearman.png")

    print()

    # Task 2: REAL vs DECOY classification
    print("  Task 2 — REAL vs DECOY classification")
    print("  " + "-" * 40)
    binary = df["label"].astype(str).str.upper().str.startswith("REAL").astype(int)
    auc_rf = roc_auc_score(binary, df["rf_score"])
    auc_gb = roc_auc_score(binary, df["gb_score"])
    results.update(
        {
            "auc_rf": auc_rf,
            "auc_gb": auc_gb,
            "n_real": binary.sum(),
            "n_decoy": (binary == 0).sum(),
        }
    )

    print(f"  REAL: {binary.sum()}  |  DECOY: {(binary == 0).sum()}")
    print(f"  AUC — Random Forest    : {auc_rf:.3f}")
    print(f"  AUC — Gradient Boosting: {auc_gb:.3f}")
    print(
        "  Pipeline clearly separates REAL from DECOY."
        if auc_rf >= 0.80
        else (
            "  Partial separation of REAL from DECOY."
            if auc_rf >= 0.65
            else "  Weak discrimination, training labels may not generalize."
        )
    )

    # Interpretation
    print("\n  Interpretation")
    print("  " + "-" * 40)
    if "rho_rf" in results:
        rho, p = results["rho_rf"], results["p_rf"]
        if rho < 0 and p < 0.05:
            print(f"  rho = {rho:+.3f} (p={p:.4f}) — significant negative correlation.")
            print(
                "  Higher pipeline score -> lower IC50 -> stronger binder. Pipeline is biologically valid."
            )
        elif p >= 0.05:
            print(
                f"  rho = {rho:+.3f} (p={p:.4f}) — no significant correlation with IC50."
            )
            print("  Expected: our modules are proxies, not affinity predictors.")
        else:
            print(f"  rho = {rho:+.3f} (p={p:.4f}) — unexpected direction.")
    if "auc_rf" in results:
        auc_val = results["auc_rf"]
        print(
            f"\n  AUC = {auc_val:.3f} — {'pipeline clearly separates REAL from DECOY.' if auc_val >= 0.80 else 'partial separation.' if auc_val >= 0.65 else 'weak discrimination.'}"
        )

    # Save table
    out_csv = ANALYSIS_DIR / "patient_real_evaluation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Full table saved -> {out_csv}\n")
    return results


def shap_analysis(model, X_test, feature_names, candidate_ids, labels_raw) -> None:
    """SHAP explainability for patient_zero.

    Saves:
      - analysis/shap_summary.png
      - analysis/shap_waterfall_cand_01.png
      - analysis/shap_waterfall_gold.png
      - analysis/shap_waterfall_bad.png
    """
    try:
        import shap
    except ImportError:
        print("[SKIP] SHAP not installed. Run: pip install shap")
        return

    print("=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)

    X_df = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.TreeExplainer(model)

    raw = explainer.shap_values(X_df)

    if isinstance(raw, list):
        shap_pos = raw[1]
        base_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else float(explainer.expected_value)
        )
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_pos = raw[:, :, 1]
        base_value = (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else float(explainer.expected_value)
        )
    else:
        shap_pos = raw
        base_value = float(explainer.expected_value)

    if shap_pos.shape != X_df.shape:
        print(f"[SKIP] SHAP shape mismatch: {shap_pos.shape} vs {X_df.shape}")
        return

    # 1. Summary plot
    plt.figure(figsize=(12, max(6, len(feature_names) * 0.4)))
    shap.summary_plot(shap_pos, X_df, feature_names=feature_names, show=False)
    plt.title("SHAP Summary — patient_zero")
    plt.tight_layout()
    out_summary = ANALYSIS_DIR / "shap_summary.png"
    plt.savefig(out_summary, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {out_summary}")

    # Trouver les indices
    cand01_idx = next(
        (i for i, cid in enumerate(candidate_ids) if cid == "CAND_01"), None
    )
    gold_idx = next(
        (i for i, lbl in enumerate(labels_raw) if lbl in ("GOLD", "GOOD")), None
    )
    bad_idx = next(
        (i for i, lbl in enumerate(labels_raw) if lbl in ("BAD", "TRAP")), None
    )

    targets = []
    if cand01_idx is not None:
        targets.append(("cand_01", cand01_idx))
    if gold_idx is not None:
        targets.append(("gold", gold_idx))
    if bad_idx is not None:
        targets.append(("bad", bad_idx))

    if not targets:
        print("[SKIP] No candidate found for waterfall plots.")
        return

    probas = model.predict_proba(X_test)[:, 1]

    # 2. Waterfall plots - CORRECTION: créer une nouvelle figure pour CHAQUE plot
    for tag, idx in targets:
        # CRÉER UNE NOUVELLE FIGURE
        fig, ax = plt.subplots(figsize=(10, max(5, len(feature_names) * 0.35)))

        exp = shap.Explanation(
            values=shap_pos[idx],
            base_values=base_value,
            data=X_df.iloc[idx].values,
            feature_names=feature_names,
        )

        # Dessiner le waterfall plot sur l'axe
        shap.waterfall_plot(exp, max_display=12, show=False)

        # Ajouter le titre APRÈS avoir dessiné
        if tag == "cand_01":
            rank = np.argsort(-probas).tolist().index(idx) + 1
            plt.title(
                f"SHAP Waterfall — {candidate_ids[idx]} [{labels_raw[idx]}]\n"
                f"Rank #{rank} | Predicted GOOD/GOLD probability = {probas[idx]:.3f}"
            )
        else:
            plt.title(
                f"SHAP Waterfall — {candidate_ids[idx]} [{labels_raw[idx]}]\n"
                f"Predicted GOOD/GOLD probability = {probas[idx]:.3f}"
            )

        plt.tight_layout()
        out_path = ANALYSIS_DIR / f"shap_waterfall_{tag}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)  # Fermer la figure spécifique
        print(f"  Plot saved -> {out_path}")

    # 3. Brief interpretation
    mean_abs = np.abs(shap_pos).mean(axis=0)
    top_features = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    print("\nBrief interpretation:")
    print("  Most important modules globally:")
    for feat, val in top_features.head(5).items():
        print(f"    - {feat}: {val:.4f}")

    if cand01_idx is not None:
        rank = np.argsort(-probas).tolist().index(cand01_idx) + 1
        print(f"\n  Why CAND_01 is ranked #{rank}:")
        cand_contrib = pd.Series(shap_pos[cand01_idx], index=feature_names).sort_values(
            key=np.abs, ascending=False
        )
        for feat, val in cand_contrib.head(5).items():
            arrow = "pushes up" if val > 0 else "pulls down"
            print(f"    - {feat}: {val:+.4f} ({arrow})")

    if gold_idx is not None:
        print(f"\n  Why {candidate_ids[gold_idx]} (GOLD) is ranked high:")
        gold_contrib = pd.Series(shap_pos[gold_idx], index=feature_names).sort_values(
            key=np.abs, ascending=False
        )
        for feat, val in gold_contrib.head(5).items():
            arrow = "pushes up" if val > 0 else "pulls down"
            print(f"    - {feat}: {val:+.4f} ({arrow})")

    if bad_idx is not None:
        print(f"\n  Why {candidate_ids[bad_idx]} (BAD) is ranked low:")
        bad_contrib = pd.Series(shap_pos[bad_idx], index=feature_names).sort_values(
            key=np.abs, ascending=False
        )
        for feat, val in bad_contrib.head(5).items():
            arrow = "pushes up" if val > 0 else "pulls down"
            print(f"    - {feat}: {val:+.4f} ({arrow})")

    print()


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    print("\nOpen-NeoVax -- ML Pipeline -- groupe 07 (B2 TAP)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & standardize training data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading training data (patient_one)...")
    X_train_raw, y_train, feature_names = load_training()
    print(f"  {len(y_train)} candidates  x  {len(feature_names)} features")
    print(
        f"  Positives (GOLD/GOOD): {y_train.sum()} | Negatives (BAD/TRAP): {(y_train == 0).sum()}\n"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # ------------------------------------------------------------------
    # 2. Cross-validation comparison
    # ------------------------------------------------------------------
    print("[2/6] Comparing models (cross-validation)...")
    print("- CROSS-VALIDATION ACCURACY (5-fold, patient_one)")
    cv_models = {
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
    for name, model in cv_models.items():
        scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        print(f"  {name:25s}  {scores.mean():.3f}  (+/- {scores.std():.3f})")
    print()

    # ------------------------------------------------------------------
    # 3. Train final models
    # ------------------------------------------------------------------
    print("[3/6] Training final models on full patient_one dataset...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE
    )
    rf.fit(X_train_scaled, y_train)
    lr.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 4. Feature importance — 3 methods
    # ------------------------------------------------------------------
    print("[4/6] Feature importance analysis...\n")

    imp_rf = importance_rf(rf, feature_names)
    imp_lr = importance_lr(lr, feature_names)
    imp_perm = importance_permutation(rf, X_train_scaled, y_train, feature_names)

    print_importances(imp_rf, "FEATURE IMPORTANCE -- Random Forest (Gini, built-in)")
    print_importances(
        imp_lr, "FEATURE IMPORTANCE -- Logistic Regression (coefficients)"
    )
    print_importances(
        imp_perm, "FEATURE IMPORTANCE -- Permutation importance (Random Forest)"
    )

    # Feature importance plot
    imp_rf.plot(kind="barh", figsize=(10, 6))
    plt.xlabel("Importance")
    plt.title("Feature importance -- Random Forest (groupe 07)")
    _save_plot(ANALYSIS_DIR / "feature_importance_rf.png")

    # Hold-out classification report
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    rf_ho = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    rf_ho.fit(X_tr, y_tr)

    y_pred_te = rf_ho.predict(X_te)
    print("\nClassification Report (hold-out 20%):")
    print(
        classification_report(y_te, y_pred_te, target_names=["BAD/TRAP", "GOOD/GOLD"])
    )

    # ------------------------------------------------------------------
    # 5. Validate on patient_zero
    # ------------------------------------------------------------------
    print("\n[5/6] Validation on patient_zero...")
    X_zero_raw, ids_zero, labels_zero = load_validation(feature_names)
    X_zero_scaled = scaler.transform(X_zero_raw)

    print_ranking(
        rf,
        X_zero_scaled,
        ids_zero,
        labels_zero,
        "FINAL RANKING — Random Forest (patient_zero)",
    )
    print_ranking(
        gb,
        X_zero_scaled,
        ids_zero,
        labels_zero,
        "FINAL RANKING — Gradient Boosting (patient_zero)",
    )

    # ------------------------------------------------------------------
    # 6. Bonus analyses
    # ------------------------------------------------------------------
    print("[6/6] Bonus analyses...\n")

    # Bonus 1 — Feature selection
    feature_selection(X_train_scaled, y_train, feature_names, imp_rf)

    # Bonus 2 — Visualizations
    visualizations(
        rf,
        X_train_scaled,
        y_train,
        X_zero_scaled,
        ids_zero,
        feature_names,
        y_te,
        y_pred_te,
    )

    # Bonus 3 — Hyperparameter tuning
    hyperparameter_tuning(X_train_scaled, y_train)

    # Bonus 4 — Ordinal regression
    print("- Ordinal regression (predict label 0-4)")
    ordinal_regression(X_zero_raw, ids_zero, labels_zero)

    # ------------------------------------------------------------------
    # Patient_real evaluation
    # ------------------------------------------------------------------
    print("- Evaluating on patient_real.csv...")
    evaluate_patient_real(rf, gb, scaler, feature_names)
    shap_analysis(rf, X_zero_scaled, feature_names, ids_zero, labels_zero)

    print("=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
