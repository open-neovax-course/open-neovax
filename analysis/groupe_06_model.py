from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_score_matrix(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def normalize_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def build_binary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only:
    GOLD / GOOD -> 1
    BAD / TRAP  -> 0
    Drop MEDIOCRE and other labels.
    """
    label_map = {
        "GOLD": 1,
        "GOOD": 1,
        "BAD": 0,
        "TRAP": 0,
    }

    work = df.copy()

    if "label" not in work.columns:
        raise ValueError("Input score matrix must contain a 'label' column.")

    work["label_str"] = work["label"].map(normalize_label)
    work = work[work["label_str"].isin(label_map.keys())].copy()
    work["y"] = work["label_str"].map(label_map)

    return work


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "candidate_id",
        "label",
        "label_str",
        "y",
        "dataset",
        "patient",
        "note",
        "gene",
        "hla_allele",
        "peptide_wt",
        "peptide_mut",
        "mut_pos_1based",
    }

    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }

    print("\n=== Cross-validation results (accuracy) ===")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"{name}: mean={scores.mean():.4f} std={scores.std():.4f}")


def fit_logistic_regression(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def fit_random_forest(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def show_feature_importance(X: pd.DataFrame, y: pd.Series) -> None:
    print("\n=== Feature importance / coefficients ===")

    logreg_pipeline = fit_logistic_regression(X, y)
    logreg = logreg_pipeline.named_steps["clf"]
    coef = pd.Series(logreg.coef_[0], index=X.columns)
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

    print("\nTop Logistic Regression coefficients:")
    print(coef.head(10))

    rf_pipeline = fit_random_forest(X, y)
    rf = rf_pipeline.named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )

    print("\nTop Random Forest importances:")
    print(importances.head(10))


def rank_patient_zero(train_df: pd.DataFrame, zero_df: pd.DataFrame) -> pd.DataFrame:
    train_bin = build_binary_dataset(train_df)
    feature_cols = get_feature_columns(train_bin)

    if not feature_cols:
        raise ValueError("No numeric feature columns found for training.")

    X_train = train_bin[feature_cols]
    y_train = train_bin["y"]

    model = fit_logistic_regression(X_train, y_train)

    missing_cols = [col for col in feature_cols if col not in zero_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in patient_zero: {missing_cols}")

    X_zero = zero_df[feature_cols].copy()
    zero_rank = zero_df.copy()
    zero_rank["pred_proba_good"] = model.predict_proba(X_zero)[:, 1]

    sort_cols = ["pred_proba_good"]
    ascending = [False]
    if "candidate_id" in zero_rank.columns:
        sort_cols.append("candidate_id")
        ascending.append(True)

    zero_rank = zero_rank.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    keep_cols = [c for c in ["candidate_id", "label", "pred_proba_good"] if c in zero_rank.columns]
    return zero_rank[keep_cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group 06 ML model for Open-NeoVax score matrices"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="analysis/scores_patient_one.csv",
        help="Path to score matrix for patient_one",
    )
    parser.add_argument(
        "--zero_csv",
        type=str,
        default="analysis/scores_patient_zero.csv",
        help="Path to score matrix for patient_zero",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="analysis/patient_zero_ranking_groupe_06.csv",
        help="Output ranking CSV",
    )
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    zero_path = Path(args.zero_csv)
    output_path = Path(args.output_csv)

    train_df = load_score_matrix(train_path)
    zero_df = load_score_matrix(zero_path)

    train_bin = build_binary_dataset(train_df)
    feature_cols = get_feature_columns(train_bin)

    if not feature_cols:
        raise ValueError("No usable numeric feature columns found in training CSV.")

    X = train_bin[feature_cols]
    y = train_bin["y"]

    print("=== Dataset summary ===")
    print(f"Training rows kept for binary task: {len(train_bin)}")
    print(f"Number of features: {len(feature_cols)}")
    print("Features:")
    for col in feature_cols:
        print(f" - {col}")

    evaluate_models(X, y)
    show_feature_importance(X, y)

    ranking_df = rank_patient_zero(train_df, zero_df)

    print("\n=== Patient zero ranking ===")
    print(ranking_df)

    ranking_df.to_csv(output_path, index=False)
    print(f"\nSaved ranking to: {output_path}")


if __name__ == "__main__":
    main()
