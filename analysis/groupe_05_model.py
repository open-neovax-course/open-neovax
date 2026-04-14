"""
Groupe 05 — Global scoring model
ML-based candidate ranking & module importance analysis
Issue #39

Description:
    Ce script entraîne des modèles ML (Logistic Regression, Random Forest,
    Gradient Boosting) pour combiner les scores des modules et classer
    les candidats néoépitopes.

Bonus:
    - Feature selection (remove least important features)
    - Visualizations (heatmap, ROC, confusion matrix)
    - Ordinal regression (predict 0=TRAP .. 4=GOLD)

Outputs:
    - analysis/ranking_groupe_05.csv
    - analysis/visualizations_groupe_05.png

"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA

df_train = pd.read_csv("analysis/scores_patient_one.csv")
df_test = pd.read_csv("analysis/scores_patient_zero.csv")

# 2. PREPARE DATA

label_map = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}

df_train_clean = df_train[df_train["label"].isin(label_map.keys())].copy()

y_train = df_train_clean["label"].map(label_map)

X_train = df_train_clean.drop(columns=["candidate_id", "label"])
X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)

candidate_ids = df_test["candidate_id"]
labels_test = df_test["label"]

X_test = df_test.drop(columns=["candidate_id", "label"])
X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

X_test = X_test[X_train.columns]

# 3. STANDARDIZE

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODEL COMPARISON

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=3, random_state=42
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    results[name] = {"model": model, "mean_acc": scores.mean(), "std_acc": scores.std()}

best_name = max(results, key=lambda x: results[x]["mean_acc"])
best_model = results[best_name]["model"]
best_model.fit(X_train_scaled, y_train)

print("=" * 60)
print("MODEL COMPARISON (5-fold cross-validation)")
print("=" * 60)
for name, res in results.items():
    print(f"{name:30s}  accuracy = {res['mean_acc']:.3f} (+/- {res['std_acc']:.3f})")
print(f"\nBest model: {best_name}")

# 5. FEATURE IMPORTANCE

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Méthode 1: Logistic Regression coefficients
print("\n[Method 1] Logistic Regression coefficients (top 10):")
lr_model = models["Logistic Regression"]
lr_model.fit(X_train_scaled, y_train)

coefs = pd.Series(lr_model.coef_[0], index=X_train.columns)
coefs_sorted = coefs.abs().sort_values(ascending=False)

for i, feat in enumerate(coefs_sorted.head(10).index, 1):
    direction = "-> GOOD" if coefs[feat] > 0 else "-> BAD"
    print(f"  {i:2d}. {feat:35s}  {coefs[feat]:+.3f}  {direction}")

# Méthode 2: Random Forest Gini importance
print("\n[Method 2] Random Forest Gini importance (top 10):")
rf_model = models["Random Forest"]
rf_model.fit(X_train_scaled, y_train)

imp_gini = pd.Series(rf_model.feature_importances_, index=X_train.columns)
imp_gini = imp_gini.sort_values(ascending=False)

for i, (feat, imp) in enumerate(imp_gini.head(10).items(), 1):
    print(f"  {i:2d}. {feat:35s}  {imp:.3f}")

# Méthode 3: Permutation importance
print("\n[Method 3] Permutation importance (top 10):")
perm_result = permutation_importance(
    best_model, X_train_scaled, y_train, n_repeats=10, random_state=42
)
imp_perm = pd.Series(perm_result.importances_mean, index=X_train.columns)
imp_perm = imp_perm.sort_values(ascending=False)

for i, (feat, imp) in enumerate(imp_perm.head(10).items(), 1):
    print(f"  {i:2d}. {feat:35s}  {imp:.3f}")


# 6. FINAL RANKING


proba = best_model.predict_proba(X_test_scaled)[:, 1]
ranking = sorted(zip(candidate_ids, proba, labels_test), key=lambda x: -x[1])

with open("analysis/ranking_groupe_05.csv", "w") as f:
    f.write("rank,candidate_id,probability,true_label\n")
    for i, (cid, prob, label) in enumerate(ranking, 1):
        f.write(f"{i},{cid},{prob:.6f},{label}\n")

print("\n" + "=" * 60)
print("FINAL RANKING (patient_zero)")
print("=" * 60)
for i, (cid, prob, label) in enumerate(ranking, 1):
    marker = " <--" if label == "GOLD" else ""
    print(f"  {i:3d}. {cid}  {prob:.3f}  {label}{marker}")


# 7.1 BONUS : FEATURE SELECTION

print("\n" + "=" * 60)
print("FEATURE SELECTION (remove least important features)")
print("=" * 60)

n_keep = max(3, int(len(X_train.columns) * 0.75))
top_features = imp_gini.head(n_keep).index.tolist()
dropped_features = [f for f in X_train.columns if f not in top_features]

X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]

scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_test_sel_scaled = scaler_sel.transform(X_test_sel)

rf_full = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_sel = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

scores_full = cross_val_score(
    rf_full, X_train_scaled, y_train, cv=5, scoring="accuracy"
)
scores_sel = cross_val_score(
    rf_sel, X_train_sel_scaled, y_train, cv=5, scoring="accuracy"
)

print(
    f"  All {len(X_train.columns)} features      : {scores_full.mean():.3f} "
    f"(+/- {scores_full.std():.3f})"
)
print(
    f"  Top {n_keep} features ({len(dropped_features)} removed): "
    f"{scores_sel.mean():.3f} (+/- {scores_sel.std():.3f})"
)

if scores_sel.mean() >= scores_full.mean():
    print("  Result: Accuracy improved or stayed the same")
else:
    print(f"  Result: Accuracy dropped by {scores_full.mean() - scores_sel.mean():.3f}")


# 7.2 BONUS : VISUALISATIONS (heatmap, ROC, confusion matrix)

print("\n" + "=" * 60)
print("BONUS 2: VISUALIZATIONS")
print("=" * 60)


X_tr, X_te, y_tr, y_te = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)
rf_ho = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_ho.fit(X_tr, y_tr)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Heatmap (patient_zero)
ax = axes[0]
X_zero_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
X_zero_df.index = labels_test
im = ax.imshow(X_zero_df.T.values, aspect="auto", cmap="RdYlGn")
ax.set_xticks(range(len(labels_test)))
ax.set_xticklabels(labels_test, rotation=90, fontsize=7)
ax.set_yticks(range(len(X_train.columns)))
ax.set_yticklabels(X_train.columns, fontsize=7)
ax.set_title("Score heatmap (patient_zero)")
plt.colorbar(im, ax=ax)

# Plot 2: ROC curve
ax = axes[1]
y_proba = cross_val_predict(
    rf_full, X_train_scaled, y_train, cv=5, method="predict_proba"
)[:, 1]
fpr, tpr, _ = roc_curve(y_train, y_proba)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"RF (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (cross-validated)")
ax.legend()

# Plot 3: Confusion matrix
ax = axes[2]
y_pred = rf_ho.predict(X_te)
cm = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["BAD/TRAP", "GOOD/GOLD"])
disp.plot(ax=ax, colorbar=False)
ax.set_title("Confusion Matrix (hold-out 20%)")

plt.tight_layout()
plt.savefig("analysis/visualizations_groupe_05.png", dpi=150)
print("  Plot saved to: analysis/visualizations_groupe_05.png")

# 7.3 BONUS: ORDINAL REGRESSION

print("\n" + "=" * 60)
print("ORDINAL REGRESSION (predict 0=TRAP .. 4=GOLD)")
print("=" * 60)


ordinal_map_v2 = {"GOLD": 3, "GOOD": 2, "BAD": 1, "TRAP": 0}

df_ord_v2 = df_train[df_train["label"].isin(ordinal_map_v2.keys())].copy()
y_ord_v2 = df_ord_v2["label"].map(ordinal_map_v2)

X_ord_v2 = df_ord_v2.drop(columns=["candidate_id", "label"])
X_ord_v2 = X_ord_v2.apply(pd.to_numeric, errors="coerce").fillna(0.0)
X_ord_v2 = X_ord_v2[X_train.columns]

print(f"  Candidates (GOLD/GOOD/BAD/TRAP): {len(y_ord_v2)}")
print(f"  Label distribution: {y_ord_v2.value_counts().sort_index().to_dict()}")

scaler_ord_v2 = StandardScaler()
X_ord_v2_scaled = scaler_ord_v2.fit_transform(X_ord_v2)

# Random Forest
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
mae_scores = cross_val_score(
    rf_reg, X_ord_v2_scaled, y_ord_v2, cv=5, scoring="neg_mean_absolute_error"
)
mae = -mae_scores.mean()

print(f"  MAE (5-fold CV): {mae:.3f} (scale: 0=TRAP, 1=BAD, 2=GOOD, 3=GOLD)")


# Entrainer et predire
rf_reg.fit(X_ord_v2_scaled, y_ord_v2)
X_test_ord_scaled = scaler_ord_v2.transform(X_test)
ord_preds = rf_reg.predict(X_test_ord_scaled)

ord_preds_rounded = [int(round(p)) for p in ord_preds]
ord_preds_rounded = [max(0, min(3, p)) for p in ord_preds_rounded]

inv_ordinal = {3: "GOLD", 2: "GOOD", 1: "BAD", 0: "TRAP"}
ord_ranking = sorted(
    zip(candidate_ids, ord_preds, ord_preds_rounded, labels_test), key=lambda x: -x[1]
)

print("\n  Ordinal ranking — patient_zero (top 10):")
print(f"  {'#':>3}  {'ID':10s}  {'Score':>6s}  {'Pred':>8s}  {'True':>8s}")
print("  " + "-" * 45)

cand01_rank = None
for i, (cid, score, pred_round, label) in enumerate(ord_ranking[:10], 1):
    pred_label = inv_ordinal.get(pred_round, "?")
    marker = " <- TARGET" if cid == "CAND_01" else ""
    if cid == "CAND_01":
        cand01_rank = i
    print(f"  {i:3}.  {cid:10s}  {score:5.2f}   {pred_label:8s}  {label:8s}{marker}")

print()
if cand01_rank == 1:
    print("  SUCCESS: CAND_01 is ranked #1!")
elif cand01_rank and cand01_rank <= 3:
    print(f"  GOOD: CAND_01 is ranked #{cand01_rank}")
elif cand01_rank:
    print(f"  INFO: CAND_01 is ranked #{cand01_rank}")
