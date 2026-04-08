from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path("analysis")
PATIENT_ONE_PATH = ANALYSIS_DIR / "scores_patient_one.csv"
PATIENT_ZERO_PATH = ANALYSIS_DIR / "scores_patient_zero.csv"
PATIENT_REAL_PATH = ANALYSIS_DIR / "scores_patient_real.csv"


def load_scores(csv_path: Path) -> pd.DataFrame:
    """Load a score matrix from CSV."""
    df = pd.read_csv(csv_path)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return ML feature columns (exclude identifiers and labels)."""
    excluded = {"candidate_id", "label"}
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


def main() -> None:
    patient_one_df = load_scores(PATIENT_ONE_PATH)
    patient_zero_df = load_scores(PATIENT_ZERO_PATH)
    patient_real_df = load_scores(PATIENT_REAL_PATH)

    inspect_dataset("PATIENT_ONE", patient_one_df)
    inspect_dataset("PATIENT_ZERO", patient_zero_df)
    inspect_dataset("PATIENT_REAL", patient_real_df)


if __name__ == "__main__":
    main()