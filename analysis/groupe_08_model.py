import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================

train_df = pd.read_csv("analysis/scores_patient_one.csv")
test_df = pd.read_csv("analysis/scores_patient_zero.csv")


# =========================
# LABEL ENCODING
# =========================


def encode_label(x):
    return 1 if x in ["GOLD", "GOOD"] else 0


train_df["y"] = train_df["label"].apply(encode_label)


# =========================
# FEATURES / TARGET
# =========================

drop_cols = ["candidate_id", "label", "y"]
X = train_df.drop(columns=drop_cols)
y = train_df["y"]

X_test = test_df.drop(columns=["candidate_id", "label"], errors="ignore")

X = X.fillna(0)
X_test = X_test.fillna(0)


# =========================
# STANDARDIZATION
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# =========================
# MODELS
# =========================

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}


# =========================
# CROSS VALIDATION
# =========================

print("\n=== CROSS VALIDATION ===")

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f}")


# =========================
# FEATURE IMPORTANCE
# =========================

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_scaled, y)

importances = rf.feature_importances_

importance_df = pd.DataFrame(
    {"feature": X.columns, "importance": importances}
).sort_values(by="importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance_df.head(15))


# =========================
# FINAL RANKING
# =========================

rf.fit(X_scaled, y)
proba = rf.predict_proba(X_test_scaled)[:, 1]

test_df["ml_score"] = proba

ranking = test_df.sort_values(by="ml_score", ascending=False)

print("\n=== FINAL RANKING ===")
print(ranking[["candidate_id", "ml_score"]])
