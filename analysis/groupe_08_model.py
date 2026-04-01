from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "analysis" / "scores_patient_one.csv"
TEST_PATH = PROJECT_ROOT / "analysis" / "scores_patient_zero.csv"


def train_and_evaluate():
    # -------------------------
    # LOAD DATA
    # -------------------------
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # -------------------------
    # LABEL ENCODING (good vs bad; ignore MEDIOCRE/others)
    # -------------------------
    def encode(label):
        if label in ["GOLD", "GOOD"]:
            return 1
        elif label in ["BAD", "TRAP"]:
            return 0
        else:
            return np.nan

    train = train.dropna(subset=["label"]).copy()
    train["y"] = train["label"].apply(encode)
    train = train.dropna(subset=["y"]).copy()

    # -------------------------
    # FEATURES / TARGET
    # -------------------------
    X_train = train.drop(columns=["candidate_id", "label", "y"], errors="ignore")
    y_train = train["y"].astype(int)

    X_test = test.drop(columns=["candidate_id", "label"], errors="ignore")

    # robustesse si NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # -------------------------
    # SCALE FEATURES
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # MODELS
    # -------------------------
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    print("\n=== CROSS VALIDATION (patient_one) ===")
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
        print(f"{name}: mean={scores.mean():.3f}  std={scores.std():.3f}")

    # -------------------------
    # FIT BEST MODEL (choose RF for ranking + feature importance)
    # -------------------------
    rf = models["RandomForest"]
    rf.fit(X_train_scaled, y_train)

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    importance_df = (
        pd.DataFrame({"feature": X_train.columns, "importance": rf.feature_importances_})
        .sort_values(by="importance", ascending=False)
    )

    print("\n=== FEATURE IMPORTANCE (top 15) ===")
    print(importance_df.head(15))

    # -------------------------
    # PREDICT + FINAL RANKING (patient_zero)
    # -------------------------
    test = test.copy()
    test["ml_score"] = rf.predict_proba(X_test_scaled)[:, 1]
    ranking = test.sort_values("ml_score", ascending=False)

    print("\n=== FINAL RANKING (patient_zero) ===")
    print(ranking[["candidate_id", "label", "ml_score"]])

    return ranking, importance_df


if __name__ == "__main__":
    train_and_evaluate()