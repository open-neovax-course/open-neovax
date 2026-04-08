from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANALYSIS_DIR = Path("analysis")
PATIENT_ONE_PATH = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_PATH = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL_PATH = ANALYSIS_DIR / "scores_patient_real.csv"


def load_scores(csv_path: Path) -> pd.DataFrame:
    """Load a score matrix from CSV."""
    return pd.read_csv(csv_path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns only."""
    excluded = {"candidate_id", "label", "target", "target_ordinal"}
    return [col for col in df.columns if col not in excluded]


def inspect_dataset(name: str, df: pd.DataFrame) -> None:
    """Print a quick summary of a dataset."""
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts(dropna=False))

    feature_cols = get_feature_columns(df)
    print(f"\nNumber of feature columns: {len(feature_cols)}")
    print("Feature columns:")
    print(feature_cols)


def encode_labels_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert labels to binary and remove ambiguous ones."""
    df = df.copy()

    label_map = {
        "GOLD": 1,
        "GOOD": 1,
        "BAD": 0,
        "TRAP": 0,
    }

    df = df[df["label"].isin(label_map.keys())]
    df["target"] = df["label"].map(label_map)

    return df


def prepare_ml_data(df: pd.DataFrame):
    """Prepare X and y for binary classification."""
    df = encode_labels_binary(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["target"]

    return X, y


def evaluate_logistic_regression(X, y) -> None:
    """Evaluate logistic regression with cross-validation."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("\n=== LOGISTIC REGRESSION ===")
    print(f"CV accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f}")
    print(f"Std accuracy: {scores.std():.3f}")


def evaluate_random_forest(X, y) -> None:
    """Evaluate a random forest classifier with cross-validation."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("\n=== RANDOM FOREST ===")
    print(f"CV accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f}")
    print(f"Std accuracy: {scores.std():.3f}")


def compute_feature_importance(X, y, feature_names) -> None:
    """Train RandomForest and display feature importance."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print("\n=== FEATURE IMPORTANCE ===")
    for name, score in feature_importance:
        bar = "#" * int(score * 50)
        print(f"{name:30s} {score:.3f} {bar}")


def rank_candidates(model, df) -> None:
    """Rank candidates using predicted probabilities."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    probs = model.predict_proba(X)[:, 1]

    ranked_df = df.copy()
    ranked_df["score"] = probs
    ranked_df = ranked_df.sort_values(by="score", ascending=False)

    print("\n=== FINAL RANKING (patient_zero) ===")
    for i, row in enumerate(ranked_df.itertuples(), 1):
        print(f"{i:2d}. {row.candidate_id:10s} {row.score:.3f} {row.label}")


def encode_labels_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """Encode ordered labels for regression."""
    df = df.copy()

    label_map = {
        "TRAP": 0,
        "BAD": 1,
        "MEDIOCRE": 2,
        "GOOD": 3,
        "GOLD": 4,
    }

    df = df[df["label"].isin(label_map.keys())]
    df["target_ordinal"] = df["label"].map(label_map)

    return df


def prepare_ml_data_ordinal(df: pd.DataFrame):
    """Prepare X and y for ordinal regression."""
    df = encode_labels_ordinal(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["target_ordinal"]

    return X, y


def evaluate_random_forest_regressor(X, y) -> None:
    """Evaluate an ordinal regression model with cross-validation."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")

    print("\n=== RANDOM FOREST REGRESSOR ===")
    print(f"CV neg MAE scores: {scores}")
    print(f"Mean neg MAE: {scores.mean():.3f}")
    print(f"Std neg MAE: {scores.std():.3f}")
    print(f"Mean MAE: {-scores.mean():.3f}")


def rank_candidates_regression(model, df) -> None:
    """Rank candidates using predicted regression scores."""
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    preds = model.predict(X)

    ranked_df = df.copy()
    ranked_df["pred_score"] = preds
    ranked_df = ranked_df.sort_values(by="pred_score", ascending=False)

    print("\n=== FINAL RANKING (patient_zero, ordinal regression) ===")
    for i, row in enumerate(ranked_df.itertuples(), 1):
        marker = " <--" if row.candidate_id == "CAND_01" else ""
        print(f"{i:2d}. {row.candidate_id:10s} {row.pred_score:.3f} {row.label}{marker}")


def main() -> None:
    patient_one_df = load_scores(PATIENT_ONE_PATH)
    patient_zero_df = load_scores(PATIENT_ZERO_PATH)
    patient_real_df = load_scores(PATIENT_REAL_PATH)

    inspect_dataset("PATIENT_ONE", patient_one_df)
    inspect_dataset("PATIENT_ZERO", patient_zero_df)
    inspect_dataset("PATIENT_REAL", patient_real_df)

    # Binary classification
    X_train, y_train = prepare_ml_data(patient_one_df)

    print("\n=== TRAINING DATA ===")
    print("X shape:", X_train.shape)
    print("y distribution:")
    print(y_train.value_counts())

    evaluate_logistic_regression(X_train, y_train)
    evaluate_random_forest(X_train, y_train)
    compute_feature_importance(X_train, y_train, X_train.columns.tolist())

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    final_model.fit(X_train, y_train)

    rank_candidates(final_model, patient_zero_df)

    # Ordinal regression
    X_ord, y_ord = prepare_ml_data_ordinal(patient_one_df)

    print("\n=== ORDINAL TRAINING DATA ===")
    print("X shape:", X_ord.shape)
    print("y distribution:")
    print(y_ord.value_counts().sort_index())

    evaluate_random_forest_regressor(X_ord, y_ord)

    final_regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    final_regressor.fit(X_ord, y_ord)

    rank_candidates_regression(final_regressor, patient_zero_df)


if __name__ == "__main__":
    main()