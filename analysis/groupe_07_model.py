"""ML pipeline — Global scoring model for neoepitope ranking.

Objective: learn the optimal combination of module scores to distinguish
good candidates (GOLD, GOOD) from bad ones (BAD, TRAP), and identify
which scoring modules are the most predictive.

"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONSTANTS
# =============================================================================

PATIENT_ONE = PROJECT_ROOT / "analysis" / "scores_patient_one.csv"
PATIENT_ZERO = PROJECT_ROOT / "analysis" / "scores_patient_zero.csv"

LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}

RANDOM_STATE = 42

META_COLS = {
    "candidate_id",
    "peptide_wt",
    "peptide_mut",
    "mut_pos_1based",
    "gene",
    "hla_allele",
    "note",
    "label",
}

# =============================================================================
# HELPERS
# =============================================================================


def _extract_label_keyword(raw: str) -> str:
    """Extract the label keyword from a raw string.

    Handles both clean labels ('GOLD') and annotated notes
    ('GOLD — L@P2 V@P9 radical mutation at P4').

    Returns the keyword if found in LABEL_MAP, else 'UNKNOWN'.
    """
    clean = raw.strip().upper().split("—")[0].strip()
    return clean if clean in LABEL_MAP else "UNKNOWN"


def _get_label_column(df: pd.DataFrame) -> str | None:
    """Return the first available label column name, or None."""
    for col in ("label", "note"):
        if col in df.columns:
            return col
    return None


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are score features (not metadata)."""
    return [c for c in df.columns if c not in META_COLS]


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================


def load_training(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load and prepare the training score matrix (patient_one).

    - Detects label column automatically ('label' or 'note').
    - Removes MEDIOCRE candidates (ambiguous for binary classification).
    - Fills missing values with 0.

    Returns
    -------
    X : DataFrame of numeric features (not yet scaled)
    y : Series of binary labels (1 = GOLD/GOOD, 0 = BAD/TRAP)
    feature_cols : list of feature column names
    """
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        print("  -> Run: python analysis/score_analysis.py --generate")
        sys.exit(1)

    df = pd.read_csv(path)

    label_col = _get_label_column(df)
    if label_col is None:
        print("[ERROR] No 'label' or 'note' column found in training data.")
        sys.exit(1)

    # Extract and encode labels robustly (handles 'GOLD — ...' notes)
    df["_label"] = df[label_col].astype(str).apply(_extract_label_keyword)

    # Remove MEDIOCRE and UNKNOWN
    df = df[df["_label"].isin(LABEL_MAP)].copy()

    y = df["_label"].map(LABEL_MAP)
    feature_cols = _feature_columns(df)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X, y, feature_cols


def load_validation(
    path: Path,
    train_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load and prepare the validation score matrix (patient_zero).

    Aligns columns to match the training set exactly:
    - Missing columns are filled with 0.
    - Extra columns are dropped.

    Returns
    -------
    X            : DataFrame aligned to train_columns
    candidate_ids: list of candidate IDs
    labels_raw   : list of raw label strings (for display)
    """
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        print("  -> Run: python analysis/score_analysis.py --generate")
        sys.exit(1)

    df = pd.read_csv(path)

    candidate_ids = df["candidate_id"].tolist()

    label_col = _get_label_column(df)
    if label_col:
        labels_raw = df[label_col].astype(str).apply(_extract_label_keyword).tolist()
    else:
        labels_raw = ["UNKNOWN"] * len(df)

    feature_cols = _feature_columns(df)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Align to training columns — missing -> 0, extra -> dropped
    X = X.reindex(columns=train_columns, fill_value=0.0)

    return X, candidate_ids, labels_raw


# =============================================================================
# FEATURE IMPORTANCE — 3 METHODS
# =============================================================================


def importance_rf(
    model: RandomForestClassifier,
    feature_names: list[str],
) -> pd.Series:
    """Built-in Gini importance from a trained Random Forest."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )


def importance_lr(
    model: LogisticRegression,
    feature_names: list[str],
) -> pd.Series:
    """Absolute coefficients from a trained Logistic Regression.
    A large positive coefficient -> predicts GOLD/GOOD.
    A large negative coefficient -> predicts BAD/TRAP.
    """
    coefs = pd.Series(model.coef_[0], index=feature_names)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


def importance_permutation(
    model,
    X_scaled,
    y: pd.Series,
    feature_names: list[str],
) -> pd.Series:
    """Permutation importance — model-agnostic.
    Measures accuracy drop when each feature is randomly shuffled.
    Works with any sklearn-compatible model.
    """
    result = permutation_importance(
        model, X_scaled, y, n_repeats=30, random_state=RANDOM_STATE
    )
    return pd.Series(result.importances_mean, index=feature_names).sort_values(
        ascending=False
    )


def print_importances(importances: pd.Series, title: str) -> None:
    """Print a feature importance table with inline ASCII bar chart."""
    print("=" * 60)
    print(title)
    print("=" * 60)
    for feat, imp in importances.items():
        bar = "#" * int(abs(imp) * 50)
        print(f"  {feat:35s}  {imp:+.3f}  {bar}")
    print()


def plot_importances(
    importances: pd.Series,
    title: str,
    path: Path,
) -> None:
    """Save a horizontal bar chart of feature importances to disk."""
    importances.plot(kind="barh", figsize=(10, 6))
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {path}\n")


# =============================================================================
# MODEL COMPARISON
# =============================================================================


def compare_models(X_scaled, y: pd.Series) -> None:
    """Compare 3 models using 5-fold cross-validation (accuracy).

    Models
    ------
    - Logistic Regression : linear baseline
    - Random Forest       : ensemble, captures interactions
    - Gradient Boosting   : strongest ensemble
    """
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
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
        print(f"  {name:25s}  {scores.mean():.3f}  (+/- {scores.std():.3f})")
    print()


# =============================================================================
# FINAL RANKING
# =============================================================================


def print_ranking(
    model,
    X_scaled,
    candidate_ids: list[str],
    labels_raw: list[str],
) -> None:
    """Rank candidates by predicted probability of being GOOD/GOLD.

    Prints a numbered list. GOLD candidates are flagged with '<-- TARGET'.
    Checks whether CAND_01 reaches rank #1.
    """
    probas = model.predict_proba(X_scaled)[:, 1]

    ranking = sorted(
        zip(candidate_ids, probas, labels_raw),
        key=lambda x: -x[1],
    )

    print("=" * 60)
    print("FINAL RANKING (patient_zero)")
    print("=" * 60)

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
    else:
        print("  INFO: CAND_01 not found in patient_zero.")
    print()


# =============================================================================
# FEATURE SELECTION (remove least important features)
# =============================================================================


def feature_selection(
    X_scaled,
    y: pd.Series,
    feature_names: list[str],
    importances: pd.Series,
) -> None:
    """Try removing the least important features and compare accuracy.

    Strategy: train with top-N features for N in [all, 75%, 50%, 25%].
    Prints cross-validation accuracy for each subset.
    """
    print("=" * 60)
    print("Feature selection (remove least important)")
    print("=" * 60)

    n_total = len(feature_names)
    thresholds = {
        "All features": n_total,
        "Top 75%": max(1, int(n_total * 0.75)),
        "Top 50%": max(1, int(n_total * 0.50)),
        "Top 25%": max(1, int(n_total * 0.25)),
    }

    top_features = importances.index.tolist()  # already sorted by importance

    for label, k in thresholds.items():
        selected = [feature_names.index(f) for f in top_features[:k]]
        X_sub = X_scaled[:, selected]
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        )
        scores = cross_val_score(model, X_sub, y, cv=5, scoring="accuracy")
        print(
            f"  {label:15s} ({k:2d} features)"
            f"  acc = {scores.mean():.3f} +/- {scores.std():.3f}"
        )
    print()


# =============================================================================
# VISUALIZATIONS
# =============================================================================


def visualizations(
    rf,
    lr,
    X_train_scaled,
    y_train: pd.Series,
    X_zero_scaled,
    y_zero_raw: list[str],
    feature_names: list[str],
    analysis_dir: Path,
) -> None:
    """Generate heatmap, ROC curve and confusion matrix plots."""

    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        confusion_matrix,
        roc_curve,
    )
    from sklearn.model_selection import cross_val_predict

    print("=" * 60)
    print("Visualizations")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Open-NeoVax", fontsize=13)

    # --- Plot 1: Score heatmap (patient_zero) ---
    ax = axes[0]
    X_zero_df = pd.DataFrame(X_zero_scaled, columns=feature_names)
    X_zero_df.index = y_zero_raw
    im = ax.imshow(X_zero_df.T.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(y_zero_raw)))
    ax.set_xticklabels(y_zero_raw, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title("Score heatmap (patient_zero)")
    plt.colorbar(im, ax=ax)

    # --- Plot 2: ROC curve (cross-validated on patient_one) ---
    ax = axes[1]
    y_proba_cv = cross_val_predict(
        rf, X_train_scaled, y_train, cv=5, method="predict_proba"
    )[:, 1]
    fpr, tpr, _ = roc_curve(y_train, y_proba_cv)
    from sklearn.metrics import auc

    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"RF (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (cross-validated, patient_one)")
    ax.legend()

    # --- Plot 3: Confusion matrix (hold-out split) ---
    ax = axes[2]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    rf_ho = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    rf_ho.fit(X_tr, y_tr)
    cm = confusion_matrix(y_te, rf_ho.predict(X_te))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["BAD/TRAP", "GOOD/GOLD"],
    )
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (hold-out 20%)")

    plt.tight_layout()
    out_path = analysis_dir / "visualizations.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out_path}\n")


# =============================================================================
# Hyperparameter tuning  : BEAT 90% ACCURACY
# =============================================================================


def Hyperparameter_tuning(X_scaled, y: pd.Series) -> None:
    """Try to beat 90% cross-validation accuracy with a tuned model.
    Tests several combinations and reports the best configuration found."""

    best_score = 0.0
    best_params: dict = {}

    for n_est in (100, 200, 300):
        for depth in (3, 5, 7, None):
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                random_state=RANDOM_STATE,
            )
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
            mean = scores.mean()
            depth_label = str(depth) if depth else "None"
            marker = "  <-- best so far" if mean > best_score else ""
            print(
                f"  n_estimators={n_est:3d}  max_depth={depth_label:4s}"
                f"  acc={mean:.3f} +/- {scores.std():.3f}{marker}"
            )
            if mean > best_score:
                best_score = mean
                best_params = {"n_estimators": n_est, "max_depth": depth}

    print()
    status = "YES" if best_score >= 0.90 else "NOT YET"
    print(f"  Best accuracy: {best_score:.3f} -> Beat 90%? {status}")
    print(f"  Best params  : {best_params}\n")


# =============================================================================
# ORDINAL REGRESSION :predict label 0-4 instead of binary
# =============================================================================

ORDINAL_MAP = {"TRAP": 0, "BAD": 1, "MEDIOCRE": 2, "GOOD": 3, "GOLD": 4}


def load_training_ordinal(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load training data with ordinal labels (0=TRAP ... 4=GOLD).

    Keeps ALL candidates including MEDIOCRE (needed for ordinal scale).
    """
    df = pd.read_csv(path)
    label_col = _get_label_column(df)
    if label_col is None:
        return pd.DataFrame(), pd.Series(dtype=float), []

    df["_label"] = df[label_col].astype(str).apply(_extract_label_keyword)
    df = df[df["_label"].isin(ORDINAL_MAP)].copy()
    y = df["_label"].map(ORDINAL_MAP)
    feature_cols = [c for c in _feature_columns(df) if not c.startswith("_")]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, y, feature_cols


def ordinal_regression(path_train: Path, path_val: Path) -> None:
    """Train a Gradient Boosting Regressor to predict ordinal labels (0-4).

    Keeps ALL candidates including MEDIOCRE (justified: it sits at position 2
    on the ordinal scale TRAP=0, BAD=1, MEDIOCRE=2, GOOD=3, GOLD=4).

    Evaluates with MAE (cross-validated), shows predictions vs actual on
    training samples, then produces an ordinal ranking of patient_zero.
    """

    # --- Load training data (all labels including MEDIOCRE) ---
    X_ord, y_ord, feature_cols = load_training_ordinal(path_train)
    if X_ord.empty:
        print("  [SKIP] Could not load ordinal data.\n")
        return

    inv_map = {v: k for k, v in ORDINAL_MAP.items()}

    scaler_ord = StandardScaler()
    X_ord_scaled = scaler_ord.fit_transform(X_ord)

    regressor = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE
    )

    mae_scores = cross_val_score(
        regressor, X_ord_scaled, y_ord, cv=5, scoring="neg_mean_absolute_error"
    )
    mae = -mae_scores.mean()

    print(f"  Candidates (all labels): {len(y_ord)}")
    print(
        f"  Label distribution     : " f"{y_ord.value_counts().sort_index().to_dict()}"
    )
    print(f"  MAE (5-fold CV)        : {mae:.3f}  (scale: 0=TRAP ... 4=GOLD)")
    print("  (MAE < 0.5 = off by less than half a grade on average)\n")
    if mae < 0.5:
        print("  Good result: predictions are on average < 0.5 grade off.")
    else:
        print("  Moderate result: consider more features or tuning.")

    regressor.fit(X_ord_scaled, y_ord)

    preds_train = regressor.predict(X_ord_scaled)
    print("\n  Sample predictions — patient_one (first 10):")
    print(f"  {'Actual':10s}  {'Predicted':10s}  {'Score':>5s}  {'Error':>6s}")
    print("  " + "-" * 40)
    for actual, pred in list(zip(y_ord, preds_train))[:10]:
        label_actual = inv_map.get(int(round(actual)), "?")
        label_pred = inv_map.get(int(round(pred)), "?")
        error = abs(actual - pred)
        print(f"  {label_actual:10s}  {label_pred:10s}  " f"{pred:5.2f}  {error:6.2f}")

    # Ordinal ranking on patient_zero
    df_val = pd.read_csv(path_val)
    label_col_val = _get_label_column(df_val)
    if label_col_val:
        labels_val = (
            df_val[label_col_val].astype(str).apply(_extract_label_keyword).tolist()
        )
    else:
        labels_val = ["UNKNOWN"] * len(df_val)

    val_feature_cols = _feature_columns(df_val)
    X_val = (
        df_val[val_feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .reindex(columns=feature_cols, fill_value=0.0)
    )

    X_val_scaled = scaler_ord.transform(X_val)
    preds_val = regressor.predict(X_val_scaled)

    ranking = sorted(
        zip(df_val["candidate_id"].tolist(), preds_val, labels_val),
        key=lambda x: -x[1],
    )

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
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    print("\nOpen-NeoVax -- ML Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load training data (patient_one)
    # ------------------------------------------------------------------
    print("\n[1/5] Loading training data (patient_one)...")
    X_train_raw, y_train, feature_names = load_training(PATIENT_ONE)
    print(f"  {len(y_train)} candidates  x  {len(feature_names)} features")
    print(
        f"  Positives (GOLD/GOOD): {y_train.sum()} | "
        f"Negatives (BAD/TRAP): {(y_train == 0).sum()}\n"
    )

    # ------------------------------------------------------------------
    # 2. Standardize — fit ONLY on training data
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # ------------------------------------------------------------------
    # 3. Compare models with cross-validation
    # ------------------------------------------------------------------
    print("[2/5] Comparing models (cross-validation)...")
    compare_models(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 4. Train final models on full training set
    # ------------------------------------------------------------------
    print("[3/5] Training final models on full patient_one dataset...")

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    rf.fit(X_train_scaled, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)

    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=RANDOM_STATE
    )
    gb.fit(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # 5. Feature importance — 3 methods
    # ------------------------------------------------------------------
    print("[4/5] Feature importance analysis...\n")

    # Method 1: Random Forest built-in (Gini)
    imp_rf = importance_rf(rf, feature_names)
    print_importances(imp_rf, "FEATURE IMPORTANCE -- Random Forest (Gini, built-in)")
    plot_importances(
        imp_rf,
        "Feature importance -- Random Forest (groupe 07)",
        PROJECT_ROOT / "analysis" / "groupe_07_importance_rf.png",
    )

    # Method 2: Logistic Regression coefficients
    imp_lr = importance_lr(lr, feature_names)
    print_importances(
        imp_lr, "FEATURE IMPORTANCE -- Logistic Regression (coefficients)"
    )

    # Method 3: Permutation importance (model-agnostic)
    imp_perm = importance_permutation(rf, X_train_scaled, y_train, feature_names)
    print_importances(
        imp_perm, "FEATURE IMPORTANCE -- Permutation importance (Random Forest)"
    )

    # Hold-out classification report (sanity check)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    rf_ho = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    rf_ho.fit(X_tr, y_tr)
    print("=" * 60)
    print("CLASSIFICATION REPORT -- hold-out split (patient_one, 80/20)")
    print("=" * 60)
    print(
        classification_report(
            y_te, rf_ho.predict(X_te), target_names=["BAD/TRAP", "GOOD/GOLD"]
        )
    )

    # ------------------------------------------------------------------
    # 6. Validate on patient_zero (never seen during training)
    # ------------------------------------------------------------------
    print("[5/5] Validation on patient_zero...")
    X_zero_raw, ids_zero, labels_zero = load_validation(PATIENT_ZERO, feature_names)

    # Use scaler.transform (NOT fit_transform) — scaler was fit on patient_one only
    X_zero_scaled = scaler.transform(X_zero_raw)

    print_ranking(rf, X_zero_scaled, ids_zero, labels_zero)
    print("Done.")

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    feature_selection(X_train_scaled, y_train, feature_names, imp_rf)

    # ------------------------------------------------------------------
    # Visualizations (heatmap, ROC curve, confusion matrix)
    # ------------------------------------------------------------------
    visualizations(
        rf,
        lr,
        X_train_scaled,
        y_train,
        X_zero_scaled,
        labels_zero,
        feature_names,
        PROJECT_ROOT / "analysis",
    )
    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------
    Hyperparameter_tuning(X_train_scaled, y_train)

    # ------------------------------------------------------------------
    # Ordinal regression
    # ------------------------------------------------------------------
    ordinal_regression(PATIENT_ONE, PATIENT_ZERO)


if __name__ == "__main__":
    main()
