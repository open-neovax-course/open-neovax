"""ML pipeline — Global scoring model for neoepitope ranking.
 
Objective: learn the optimal combination of module scores to distinguish
good candidates (GOLD, GOOD) from bad ones (BAD, TRAP), and identify
which scoring modules are the most predictive.
 
Group 07
"""

from __future__ import annotations
 
import sys
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(_file_).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONSTANTS
# =============================================================================

PATIENT_ONE  = PROJECT_ROOT / "analysis" / "scores_patient_one.csv"
PATIENT_ZERO = PROJECT_ROOT / "analysis" / "scores_patient_zero.csv"

LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}

RANDOM_STATE = 42

META_COLS = {
    "candidate_id", "peptide_wt", "peptide_mut",
    "mut_pos_1based", "gene", "hla_allele", "note", "label",
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
        labels_raw = (
            df[label_col].astype(str).apply(_extract_label_keyword).tolist()
        )
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
    return (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
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
    return (
        pd.Series(result.importances_mean, index=feature_names)
        .sort_values(ascending=False)
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
# MAIN PIPELINE
# =============================================================================
 
def main() -> None:
    print("\nOpen-NeoVax -- ML Pipeline -- groupe 07")
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
 
 
if _name_ == "_main_":
    main()