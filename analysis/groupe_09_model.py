"""ML pipeline — Global scoring model for neoepitope ranking.

Group 09 — Module A2 (Peptide Net Charge)
Objective: Analyze how physicochemical properties, specifically net charge,
contribute to the global neoepitope ranking.

Usage:
    python analysis/groupe_9_model.py
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
    StratifiedKFold,
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
ORDINAL_MAP = {"TRAP": 0, "BAD": 1,
               "MEDIOCRE": 2, "GOOD": 3, "GOLD": 4}

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
    "target",
    "target_ordinal",
    "target_real",
    "_label",
}

RANDOM_STATE = 42
CV = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE)


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
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {path}")


def _check_file(path: Path) -> None:
    if not path.exists():
        print(f"[ERROR] {path} not found.")
        print(
            "  -> Run: python analysis/score_analysis.py --generate")
        sys.exit(1)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_scores(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def inspect_dataset(name: str, df: pd.DataFrame) -> None:
    col = _label_col(df)
    print(f"\n=== {name} ===")
    print(f"  Shape : {df.shape}")
    if col:
        print(
            f"  Labels:\n{df[col].value_counts().to_string()}")


def load_training(
    ordinal: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load patient_one scores. Binary (MEDIOCRE excluded) or ordinal (0-4)."""
    _check_file(PATIENT_ONE)
    df = pd.read_csv(PATIENT_ONE)
    col = _label_col(df)
    df["_label"] = df[col].astype(str).apply(_label)

    label_map = ORDINAL_MAP if ordinal else LABEL_MAP
    df = df[df["_label"].isin(label_map)].copy()

    y = df["_label"].map(label_map)
    feats = _feat_cols(df)
    X = df[feats].apply(
        pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y, feats


def load_validation(
    train_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load patient_zero scores aligned to training columns."""
    _check_file(PATIENT_ZERO)
    df = pd.read_csv(PATIENT_ZERO)
    ids = df["candidate_id"].tolist()
    col = _label_col(df)
    labels = (
        df[col].astype(str).apply(_label).tolist()
        if col
        else ["UNKNOWN"] * len(df)
    )
    X = df[_feat_cols(df)].apply(
        pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.reindex(columns=train_columns, fill_value=0.0)
    return X, ids, labels


def load_patient_real(
    train_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patient_real scores and merge IC50 from raw file."""
    if not PATIENT_REAL.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(PATIENT_REAL)
    if PATIENT_REAL_RAW.exists():
        df_raw = pd.read_csv(PATIENT_REAL_RAW)[
            ["candidate_id", "ic50_nm"]]
        df = df.merge(df_raw, on="candidate_id", how="left")
    X = df[_feat_cols(df)].apply(
        pd.to_numeric, errors="coerce").fillna(0.0)
    X = X.reindex(columns=train_columns, fill_value=0.0)
    return X, df


# =============================================================================
# CROSS-VALIDATION
# =============================================================================


def compare_models(X_scaled: np.ndarray, y: pd.Series) -> None:
    """Cross-validation accuracy + AUC comparison for all models."""
    print(
        "\n=== CROSS-VALIDATION (5-fold stratified, patient_one) ===")
    models = {
        "Logistic Regression  ": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest        ": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        ),
        "Gradient Boosting    ": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=RANDOM_STATE
        ),
    }
    for name, model in models.items():
        acc = cross_val_score(
            model, X_scaled, y, cv=CV, scoring="accuracy")
        roc = cross_val_score(
            model, X_scaled, y, cv=CV, scoring="roc_auc")
        print(
            f"  {name}  acc={acc.mean():.3f} (+/-{acc.std():.3f})  "
            f"auc={roc.mean():.3f} (+/-{roc.std():.3f})"
        )


# =============================================================================
# FEATURE IMPORTANCE — 3 METHODS
# =============================================================================


def importance_rf(
    model: RandomForestClassifier, feature_names: list[str]
) -> pd.Series:
    """Built-in Gini importance from Random Forest."""
    return pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)


def importance_lr(
    model: LogisticRegression, feature_names: list[str]
) -> pd.Series:
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


def plot_feature_importance(imp: pd.Series) -> None:
    imp.head(15).plot(
        kind="barh", figsize=(10, 6), color="#2980b9")
    plt.xlabel("Importance")
    plt.title(
        "Feature Importance — Random Forest (groupe 09)")
    plt.gca().invert_yaxis()
    _save_plot(ANALYSIS_DIR /
               "groupe09_feature_importance.png")


# =============================================================================
# RANKING
# =============================================================================


def print_ranking(
    model, X_scaled: np.ndarray, ids: list[str], labels: list[str], title: str
) -> None:
    """Print candidates ranked by predicted probability of being GOOD/GOLD."""
    probas = model.predict_proba(X_scaled)[:, 1]
    ranking = sorted(
        zip(ids, probas, labels), key=lambda x: -x[1])

    print(f"\n=== {title} ===\n")
    cand01_rank = None
    for i, (cid, prob, label) in enumerate(ranking, 1):
        marker = "  <-- TARGET" if label == "GOLD" else ""
        if cid == "CAND_01":
            cand01_rank = i
        print(
            f"  {i:3d}.  {cid:10s}  {prob:.3f}  {label}{marker}")

    print()
    if cand01_rank == 1:
        print("  SUCCESS: CAND_01 is ranked #1!")
    elif cand01_rank:
        print(
            f"  WARNING: CAND_01 is ranked #{cand01_rank} (target: #1)")


# =============================================================================
# BONUS 1 — FEATURE SELECTION
# =============================================================================


def feature_selection(
    X_scaled: np.ndarray,
    y: pd.Series,
    feature_names: list[str],
    imp: pd.Series,
) -> None:
    """Try removing least important features and compare accuracy."""
    print(
        "\n=== BONUS 1 — Feature selection (remove least important) ===")
    n = len(feature_names)
    top_feats = imp.index.tolist()

    for label, k in {
        "All features": n,
        "Top 75%": max(1, int(n * 0.75)),
        "Top 50%": max(1, int(n * 0.50)),
        "Top 25%": max(1, int(n * 0.25)),
    }.items():
        selected = [feature_names.index(
            f) for f in top_feats[:k]]
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
            f"  {label:15s} ({k:2d} features)  "
            f"acc={scores.mean():.3f} +/-{scores.std():.3f}"
        )


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
    print("\n=== BONUS 2 — Visualizations ===")

    # Heatmap (patient_zero)
    fig, ax = plt.subplots(figsize=(12, 8))
    df_plot = pd.DataFrame(
        X_zero_scaled, columns=feature_names, index=ids_zero
    )
    im = ax.imshow(df_plot.T.values,
                   aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(ids_zero)))
    ax.set_xticklabels(ids_zero, rotation=90, fontsize=8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title("Score heatmap — patient_zero (groupe 09)")
    plt.colorbar(im, ax=ax)
    _save_plot(ANALYSIS_DIR / "groupe09_score_heatmap.png")

    # ROC curve (cross-validated on patient_one)
    y_proba_cv = cross_val_predict(
        rf, X_train_scaled, y_train, cv=5, method="predict_proba"
    )[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_proba_cv)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2,
            label=f"RF (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "ROC Curve (cross-validated, patient_one) — groupe 09")
    ax.legend()
    _save_plot(ANALYSIS_DIR / "groupe09_roc_curve.png")

    # Confusion matrix (hold-out)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[
            "BAD/TRAP", "GOOD/GOLD"]
    ).plot(ax=ax, colorbar=False)
    ax.set_title(
        "Confusion Matrix (hold-out 20%) — groupe 09")
    _save_plot(ANALYSIS_DIR /
               "groupe09_confusion_matrix.png")


# =============================================================================
# BONUS 3 — HYPERPARAMETER TUNING
# =============================================================================


def hyperparameter_tuning(X_scaled: np.ndarray, y: pd.Series) -> None:
    """Try to beat 90% cross-validation accuracy with tuned Random Forest."""
    print(
        "\n=== BONUS 3 — Hyperparameter tuning (Random Forest) ===")
    best_score, best_params = 0.0, {}

    for n_est in (100, 200, 300):
        for depth in (3, 5, 7, None):
            scores = cross_val_score(
                RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=RANDOM_STATE,
                ),
                X_scaled,
                y,
                cv=5,
                scoring="accuracy",
            )
            mean = scores.mean()
            marker = "  <-- best so far" if mean > best_score else ""
            print(
                f"  n_estimators={n_est:3d}  max_depth={str(depth):4s}  "
                f"acc={mean:.3f} +/-{scores.std():.3f}{marker}"
            )
            if mean > best_score:
                best_score = mean
                best_params = {
                    "n_estimators": n_est, "max_depth": depth}

    print(
        f"\n  Best accuracy: {best_score:.3f} -> "
        f"Beat 90%? {'YES' if best_score >= 0.90 else 'NOT YET'}"
    )
    print(f"  Best params  : {best_params}")


# =============================================================================
# BONUS 4 — ORDINAL REGRESSION
# =============================================================================


def ordinal_regression(
    X_zero_raw: pd.DataFrame, ids_zero: list[str], labels_zero: list[str]
) -> None:
    """Predict ordinal label 0-4 instead of binary classification."""
    print(
        "\n=== BONUS 4 — Ordinal regression (predict label 0-4) ===")
    X_ord, y_ord, ord_feats = load_training(ordinal=True)
    if X_ord.empty:
        print("  [SKIP] Could not load ordinal data.")
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
    print(
        f"  Label distribution     : "
        f"{y_ord.value_counts().sort_index().to_dict()}"
    )
    print(
        f"  MAE (5-fold CV)        : {mae:.3f}  (scale: 0=TRAP ... 4=GOLD)")
    print(
        "  Good result: < 0.5 grade off."
        if mae < 0.5
        else "  Moderate result."
    )

    reg.fit(X_ord_scaled, y_ord)

    X_zero = X_zero_raw.reindex(
        columns=ord_feats, fill_value=0.0)
    preds_val = reg.predict(scaler_ord.transform(X_zero))
    ranking = sorted(
        zip(ids_zero, preds_val, labels_zero), key=lambda x: -x[1]
    )

    print("\n  Ordinal ranking — patient_zero:")
    print(f"  {'#':>3}  {'ID':10s}  {'Score':>7s}  Label")
    print("  " + "-" * 35)
    cand01_rank = None
    for i, (cid, score, label) in enumerate(ranking, 1):
        marker = "  <-- TARGET" if label == "GOLD" else ""
        if cid == "CAND_01":
            cand01_rank = i
        print(
            f"  {i:3}.  {cid:10s}  {score:5.2f}/4  {label}{marker}")

    print()
    if cand01_rank == 1:
        print(
            "  SUCCESS: CAND_01 is ranked #1 with ordinal regression!")
    elif cand01_rank:
        print(
            f"  WARNING: CAND_01 is ranked #{cand01_rank} (target: #1)")


# =============================================================================
# PATIENT_REAL EVALUATION
# =============================================================================


def evaluate_real_vs_decoy(
    model, df: pd.DataFrame, feature_cols: list[str]
) -> None:
    """AUC for REAL vs DECOY classification."""
    df = df.copy()
    label_map = {"REAL": 1, "DECOY": 0}
    df = df[df["label"].isin(label_map)]
    df["target_real"] = df["label"].map(label_map)

    X = df[feature_cols]
    y = df["target_real"]
    probs = model.predict_proba(X)[:, 1]

    auc_score = roc_auc_score(y, probs)
    auc_flipped = roc_auc_score(y, 1 - probs)

    print("\n=== REAL vs DECOY ===")
    print(f"  AUC         : {auc_score:.3f}")
    print(f"  AUC flipped : {auc_flipped:.3f}")
    if auc_score >= 0.80:
        print("  Pipeline clearly separates REAL from DECOY.")
    elif auc_score >= 0.65:
        print("  Partial separation of REAL from DECOY.")
    else:
        print(
            "  Weak discrimination — training labels may not generalize.")


def evaluate_ic50_correlation(
    model, scores_df: pd.DataFrame, feature_cols: list[str]
) -> None:
    """Spearman correlation between model scores and IC50."""
    df = scores_df.copy()

    if PATIENT_REAL_RAW.exists():
        raw = pd.read_csv(PATIENT_REAL_RAW)[
            ["candidate_id", "ic50_nm"]]
        df = df.merge(raw, on="candidate_id", how="left")

    df = df[df["label"].astype(
        str).str.upper() == "REAL"].copy()
    df["ic50_nm"] = pd.to_numeric(
        df.get("ic50_nm", pd.Series(dtype=float)), errors="coerce"
    )
    df = df.dropna(subset=["ic50_nm"])

    print("\n=== IC50 CORRELATION (REAL candidates only) ===")
    print(f"  REAL candidates with valid IC50: {len(df)}")

    if len(df) < 5:
        print(
            "  [SKIP] Not enough REAL candidates with IC50 data.")
        return

    X = df[feature_cols]
    probs = model.predict_proba(X)[:, 1]
    rho, pvalue = spearmanr(probs, df["ic50_nm"])

    print(f"  Spearman rho : {rho:+.3f}")
    print(f"  p-value      : {pvalue:.3e}")
    if rho < 0 and pvalue < 0.05:
        print(
            "  Negative rho = higher score -> "
            "lower IC50 -> biologically valid."
        )
    elif pvalue >= 0.05:
        print(
            "  No significant correlation (expected: "
            "modules are proxies, not affinity predictors)."
        )
    else:
        print(f"  Unexpected direction: rho = {rho:+.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        probs,
        df["ic50_nm"],
        color="#e74c3c",
        alpha=0.7,
        s=60,
        edgecolors="white",
    )
    for i, (_, row) in enumerate(df.iterrows()):
        ax.annotate(
            row["candidate_id"],
            (probs[i], row["ic50_nm"]),
            fontsize=6,
            ha="left",
            va="bottom",
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel(
        "Pipeline RF score (higher = more likely GOOD/GOLD)")
    ax.set_ylabel(
        "Experimental IC50 (nM, lower = stronger binder)")
    ax.set_title(
        f"Pipeline score vs IC50 — patient_real\n"
        f"Spearman rho={rho:+.3f}  p={pvalue:.4f}"
    )
    ax.grid(True, alpha=0.3)
    _save_plot(ANALYSIS_DIR /
               "groupe09_patient_real_spearman.png")


# =============================================================================
# BIOLOGICAL INTERPRETATION
# =============================================================================


def biological_interpretation(
    imp_rf: pd.Series, imp_lr: pd.Series, imp_perm: pd.Series
) -> None:
    """Print consensus feature ranking and biological commentary."""
    print("\n=== BIOLOGICAL INTERPRETATION ===")

    # Consensus top features (average rank across 3 methods)
    all_feats = set(imp_rf.index) | set(
        imp_lr.index) | set(imp_perm.index)
    n = len(all_feats)
    rank_rf = {f: i for i, f in enumerate(imp_rf.index)}
    rank_lr = {f: i for i, f in enumerate(imp_lr.index)}
    rank_perm = {f: i for i, f in enumerate(imp_perm.index)}
    avg_rank = {
        f: np.mean([rank_rf.get(f, n), rank_lr.get(
            f, n), rank_perm.get(f, n)])
        for f in all_feats
    }
    consensus = sorted(avg_rank.items(), key=lambda x: x[1])

    print(
        "\n  Consensus top features (avg rank across RF / LR / permutation):"
    )
    for feat, rank in consensus[:10]:
        print(f"    {feat:35s}  avg_rank={rank:.1f}")

    print("""
  Biological interpretation:

  * C (HLA anchoring) — C_anchoring_P2, C_hla_anchor_p9:
    Anchor residues at P2 and PΩ (P9) are the primary determinants
    of pMHC complex stability. Strong anchoring ensures the peptide
    is presented at the cell surface long enough for T-cell recognition.

  * A (physicochemical) — A_hybrid_complexity, A_hydrophobicity:
    Sequence complexity and hydrophobicity drive TCR recognition.
    Overly simple or hydrophilic peptides are poorly immunogenic.

  * D (self-similarity) — D_self_match, D_approx_self_match:
    Peptides too similar to self are eliminated by central tolerance
    and will not trigger an effective CD8+ T-cell response.
    Self-divergence is both a safety and efficacy criterion.

  * B (processing) — B_proteasome, B_tap:
    Proteasomal cleavage and TAP transport determine whether the
    peptide reaches the ER for HLA loading. Without this, even
    a perfect binder is never presented.

  Least important modules (net charge, length filters) act as
  coarse sanity checks — no discriminant signal once other modules
  are accounted for.
""")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    print("\nOpen-NeoVax -- ML Pipeline -- groupe 09")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load & inspect
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    _check_file(PATIENT_ONE)
    _check_file(PATIENT_ZERO)

    df_one = load_scores(PATIENT_ONE)
    df_zero = load_scores(PATIENT_ZERO)
    inspect_dataset("PATIENT_ONE", df_one)
    inspect_dataset("PATIENT_ZERO", df_zero)

    # ------------------------------------------------------------------
    # 2. Prepare training data
    # ------------------------------------------------------------------
    print("\n[2/6] Preparing training data...")
    X_train_raw, y_train, feature_names = load_training()
    print(
        f"  {len(y_train)} candidates  x  {len(feature_names)} features")
    print(
        f"  Positives (GOLD/GOOD): {y_train.sum()} | "
        f"Negatives (BAD/TRAP): {(y_train == 0).sum()}"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # ------------------------------------------------------------------
    # 3. Cross-validation comparison
    # ------------------------------------------------------------------
    print("\n[3/6] Comparing models...")
    compare_models(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 4. Train final models
    # ------------------------------------------------------------------
    print(
        "\n[4/6] Training final models on full patient_one...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=RANDOM_STATE
    )
    lr = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE
    )

    rf.fit(X_train_scaled, y_train)
    lr.fit(X_train_scaled, y_train)
    gb.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Feature importance — 3 methods
    # ------------------------------------------------------------------
    print("\n[5/6] Feature importance analysis...")

    imp_rf = importance_rf(rf, feature_names)
    imp_lr = importance_lr(lr, feature_names)
    imp_perm = importance_permutation(
        rf, X_train_scaled, y_train, feature_names
    )

    print_importances(
        imp_rf, "FEATURE IMPORTANCE — Random Forest (Gini)")
    print_importances(
        imp_lr, "FEATURE IMPORTANCE — Logistic Regression (coefficients)"
    )
    print_importances(
        imp_perm, "FEATURE IMPORTANCE — Permutation importance")

    plot_feature_importance(imp_rf)

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
        classification_report(
            y_te, y_pred_te, target_names=[
                "BAD/TRAP", "GOOD/GOLD"]
        )
    )

    # ------------------------------------------------------------------
    # 6. Validate on patient_zero
    # ------------------------------------------------------------------
    print("\n[6/6] Validation on patient_zero...")
    X_zero_raw, ids_zero, labels_zero = load_validation(
        feature_names)
    X_zero_scaled = scaler.transform(X_zero_raw)

    print_ranking(
        rf,
        X_zero_scaled,
        ids_zero,
        labels_zero,
        "FINAL RANKING — Random Forest",
    )
    print_ranking(
        gb,
        X_zero_scaled,
        ids_zero,
        labels_zero,
        "FINAL RANKING — Gradient Boosting",
    )

    # ------------------------------------------------------------------
    # Bonus analyses
    # ------------------------------------------------------------------
    feature_selection(
        X_train_scaled, y_train, feature_names, imp_rf)
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
    hyperparameter_tuning(X_train_scaled, y_train)
    ordinal_regression(X_zero_raw, ids_zero, labels_zero)

    # ------------------------------------------------------------------
    # Patient_real evaluation
    # ------------------------------------------------------------------
    print("\n--- Patient Real Evaluation ---")
    if PATIENT_REAL.exists():
        df_real = load_scores(PATIENT_REAL)
        inspect_dataset("PATIENT_REAL", df_real)
        X_real_raw, _ = load_patient_real(feature_names)
        if not X_real_raw.empty:
            evaluate_real_vs_decoy(
                rf, df_real, feature_names)
            evaluate_ic50_correlation(
                rf, df_real, feature_names)
    else:
        print(f"  [SKIP] {PATIENT_REAL} not found.")

    # ------------------------------------------------------------------
    # Biological interpretation
    # ------------------------------------------------------------------
    biological_interpretation(imp_rf, imp_lr, imp_perm)

    print("=" * 60)
    print("Pipeline complete. Plots saved in analysis/")
    print("=" * 60)


if __name__ == "__main__":
    main()
