from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

ANALYSIS_DIR = Path("analysis")
PATIENT_ONE_PATH = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_PATH = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL_PATH = ANALYSIS_DIR / "scores_patient_real.csv"


def load_scores(csv_path: Path) -> pd.DataFrame:
    """Load a score matrix from CSV."""
    df = pd.read_csv(csv_path)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"candidate_id", "label", "target"}
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

    # mapping
    label_map = {
        "GOLD": 1,
        "GOOD": 1,
        "BAD": 0,
        "TRAP": 0,
    }

    # keep only known labels
    df = df[df["label"].isin(label_map.keys())]

    # encode
    df["target"] = df["label"].map(label_map)

    return df


def prepare_ml_data(df: pd.DataFrame):
    """Prepare X and y for ML."""
    df = encode_labels_binary(df)

    feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df["target"]

    return X, y

def evaluate_logistic_regression(X, y) -> None:
    """Evaluate a logistic regression model with cross-validation."""
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
    """Evaluate a random forest model with cross-validation."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("\n=== RANDOM FOREST ===")
    print(f"CV accuracy scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.3f}")
    print(f"Std accuracy: {scores.std():.3f}")

def compute_feature_importance(X, y, feature_names):
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

    def rank_candidates(model, df):
    """Rank candidates using predicted probabilities."""
    feature_cols = get_feature_columns(df)

    X = df[feature_cols]

    probs = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["score"] = probs

    df = df.sort_values(by="score", ascending=False)

    print("\n=== FINAL RANKING (patient_zero) ===")
    for i, row in enumerate(df.itertuples(), 1):
        print(f"{i:2d}. {row.candidate_id:10s} {row.score:.3f} {row.label}")


def main() -> None:
    patient_one_df = load_scores(PATIENT_ONE_PATH)
    patient_zero_df = load_scores(PATIENT_ZERO_PATH)
    patient_real_df = load_scores(PATIENT_REAL_PATH)

    inspect_dataset("PATIENT_ONE", patient_one_df)
    inspect_dataset("PATIENT_ZERO", patient_zero_df)
    inspect_dataset("PATIENT_REAL", patient_real_df)

    # Prepare training data
    X_train, y_train = prepare_ml_data(patient_one_df)

    print("\n=== TRAINING DATA ===")
    print("X shape:", X_train.shape)
    print("y distribution:")
    print(y_train.value_counts())

    evaluate_logistic_regression(X_train, y_train)

    evaluate_random_forest(X_train, y_train)

    compute_feature_importance(
        X_train,
        y_train,
        X_train.columns.tolist()
    )

    # Train final model on all data
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    final_model.fit(X_train, y_train)

    # Ranking on patient_zero
    rank_candidates(final_model, patient_zero_df)

if __name__ == "__main__":
    main()