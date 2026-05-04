# Machine Learning Guide — Global Scoring Model

## What are we trying to do?

The pipeline has 10+ scoring modules. Each one captures a different biological signal:
hydrophobicity, HLA anchor quality, self-similarity, proteasome cleavage, etc.
Right now, the final ranking is a **simple average** of all scores — every module
counts equally. But that's probably not optimal: the anchor P2 score (C1) is likely
more predictive of a good neoepitope than the net charge score (A2).

**Your job**: use machine learning to find the best way to combine these scores,
and figure out which modules actually matter.

---

## The two objectives

### 1. Build a global scoring model

Instead of averaging, train a model that learns the **optimal combination** of scores
to distinguish good candidates (GOLD, GOOD) from bad ones (BAD, TRAP).

### 2. Identify the most important modules

Once the model is trained, extract **feature importance** — a ranking of which
modules contribute the most to the prediction. This answers the biological question:
what matters most for neoepitope quality?

---

## The data

### Score matrices

Run the provided script to generate the score matrices:

```bash
python analysis/score_analysis.py --generate
```

This produces two CSV files:

**`analysis/scores_patient_one.csv`** — Training set (75 candidates)

| candidate_id | label | C_anchoring_P2 | A_hybrid_complexity | ... |
|-------------|-------|---------------|--------------------|----|
| P1_001 | GOLD | 2.0 | 0.85 | ... |
| P1_016 | GOOD | 1.0 | 0.72 | ... |
| P1_036 | MEDIOCRE | -0.5 | 0.41 | ... |
| P1_051 | BAD | -2.5 | 0.33 | ... |
| P1_071 | TRAP | 0.0 | 0.60 | ... |

**`analysis/scores_patient_zero.csv`** — Validation set (18 candidates)

Same format. This is the dataset you already know — CAND_01 (GOLD) should end up
at the top of your ranking.

**`analysis/scores_patient_real.csv`** — Real experimental data (138 candidates)

This dataset contains **real neoantigen data** from published research
(dbPepNeo2.0, HNSCC tumors):

- **69 REAL candidates**: true WT/MUT peptide pairs from cancer patients with
  measured IC50 binding affinity (3-48 nM). Source: experimental mass spectrometry
  and binding assays.
- **69 DECOY candidates**: same peptides with middle positions (P3-P8) shuffled,
  anchors (P1, P2, P9) preserved. Standard negative controls used in
  computational immunology benchmarks.

The `ic50_nm` column contains the measured binding affinity for REAL candidates
(empty for DECOYs). Lower IC50 = stronger binder = better candidate.

Two analyses are possible with this dataset:

1. **Spearman correlation** (REAL only): does your pipeline ranking correlate
   with the experimentally measured IC50?
   ```python
   from scipy.stats import spearmanr
   rho, pvalue = spearmanr(pipeline_scores, ic50_values)
   ```

2. **Classification REAL vs DECOY**: can your model distinguish real neoantigen
   candidates from shuffled decoys? Train on patient_one, predict on patient_real,
   measure AUC.

### Labels

Each candidate has a label in the `note` column:

| Label | Meaning | Numeric encoding |
|-------|---------|:----------------:|
| GOLD | Excellent candidate — perfect anchors, good properties | 4 |
| GOOD | Good candidate — decent anchors, reasonable properties | 3 |
| MEDIOCRE | Average — weak anchors or charged middle | 2 |
| BAD | Poor candidate — terrible anchors, extreme properties, self-match | 1 |
| TRAP | Edge case — WT==MUT, invalid chars, position out of range | 0 |

### Features

Each column (except `candidate_id` and `label`) is one **feature** — the output
of one scoring module. The feature names are the score names you defined
(e.g. `C_anchoring_P2`, `A_hybrid_complexity`, `B_proteasome_cterm`).

---

## Machine learning concepts you need

### What is classification?

Classification is a type of supervised learning: you have labeled data (X, y)
and you train a model to predict the label y from the features X.

In our case:
- **X** = the score matrix (one row per candidate, one column per module)
- **y** = the label (GOLD/GOOD vs BAD/TRAP)

We simplify to **binary classification**:
- Class 1 (positive): GOLD and GOOD candidates — we want to select them
- Class 0 (negative): BAD and TRAP candidates — we want to reject them
- MEDIOCRE: ambiguous — you can remove them from training, or assign them to one class

```python
import pandas as pd

df = pd.read_csv("analysis/scores_patient_one.csv")

# Encode labels
label_map = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}

# Remove MEDIOCRE (ambiguous) or assign it
df = df[df["label"].isin(label_map.keys())]

y = df["label"].map(label_map)
X = df.drop(columns=["candidate_id", "label"])

# Fill missing values with 0
X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
```

### Feature standardization

Different modules produce scores in different ranges: C_anchoring_P2 might be
in [-3, +3] while B_sanity_check is in [-10, +1]. If you don't standardize,
models like Logistic Regression will be biased toward high-magnitude features.

**StandardScaler** transforms each feature to have mean=0 and std=1:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

> **Note**: fit the scaler on the training set only. When you evaluate on
> patient_zero, use `scaler.transform(X_test)` (not `fit_transform`).

### Train/test split and cross-validation

You never evaluate a model on the same data you trained it on — that would be
cheating (overfitting). Two strategies:

**1. Hold-out split**: split your data into train (80%) and test (20%).

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

**2. Cross-validation** (recommended with small datasets like ours): split the
data into K folds, train on K-1 folds, test on the remaining one. Repeat K times.
The average score is a robust estimate of model performance.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

With 5-fold cross-validation on 55 candidates (after removing MEDIOCRE), each
fold has ~11 candidates. It's small, but it's what we have.

**3. Final validation**: after training, evaluate on patient_zero (18 candidates)
— a completely independent dataset the model has never seen.

---

## Models you can try

### 1. Logistic Regression — the baseline

The simplest classification model. It learns a **linear combination** of features
(like a weighted sum) and passes it through a sigmoid function to get a probability.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# The learned weights tell you how important each feature is
print(dict(zip(X.columns, model.coef_[0])))
```

**When to use**: good starting point, fast, interpretable weights.
**Limitation**: can only learn linear combinations — if two features interact
(e.g. "good anchor AND low self-similarity → GOLD"), it cannot capture that.

### 2. Random Forest — captures interactions

An ensemble of decision trees. Each tree learns a different set of rules
(e.g. "if C_anchoring_P2 > 1.0 and D_self_exact_match < -5.0 then GOLD").
The forest votes on the final prediction.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Feature importance: how much does each feature reduce prediction error?
print(dict(zip(X.columns, model.feature_importances_)))
```

**When to use**: captures non-linear interactions, provides feature importance.
**Limitation**: can overfit on small datasets — use `max_depth` to limit tree size.

### 3. Support Vector Machine (SVM)

Finds the hyperplane that best separates the two classes, with maximum margin.
With a non-linear kernel (RBF), it can capture complex decision boundaries.

```python
from sklearn.svm import SVC

model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

# SVM does not directly give feature importance
# But you can use permutation importance (see below)
```

**When to use**: good with small datasets, robust to outliers.
**Limitation**: no built-in feature importance (need permutation importance).

### 4. Gradient Boosting — the heavy hitter

Builds trees sequentially: each new tree corrects the errors of the previous ones.
Often the best-performing model, but slower and can overfit.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
)
model.fit(X_train, y_train)

# Feature importance (same API as Random Forest)
print(dict(zip(X.columns, model.feature_importances_)))
```

**When to use**: when you want maximum accuracy and have enough data.
**Limitation**: more hyperparameters to tune, slower training.

### 5. K-Nearest Neighbors (KNN) — the lazy learner

No training step at all. To classify a new candidate, it looks at the K most
similar candidates in the training set and takes the majority vote.

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

**When to use**: simple baseline, good to visualize decision boundaries.
**Limitation**: sensitive to feature scaling, no feature importance.

### Model comparison

Try at least 2-3 models and compare their cross-validation scores:

```python
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3),
    "SVM (RBF)": SVC(kernel="rbf", probability=True),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    print(f"{name:25s}  accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Feature importance — which modules matter?

This is the key biological insight of the project. Feature importance tells you
how much each scoring module contributes to the model's predictions.

### Method 1: Built-in importance (Random Forest, Gradient Boosting)

Tree-based models compute importance during training. For each feature, it measures
how much the feature reduces prediction error (Gini impurity or entropy) across
all trees in the forest.

```python
import pandas as pd

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_scaled, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("Feature importance:")
for feat, imp in importances.items():
    bar = "#" * int(imp * 50)
    print(f"  {feat:35s}  {imp:.3f}  {bar}")
```

### Method 2: Logistic Regression coefficients

In Logistic Regression, each feature has a **weight** (coefficient). A large
positive weight means the feature strongly predicts class 1 (GOOD/GOLD).
A large negative weight means it predicts class 0 (BAD/TRAP).

```python
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

coefs = pd.Series(lr.coef_[0], index=X.columns)
coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

print("Logistic Regression coefficients:")
for feat, coef in coefs.items():
    direction = "+" if coef > 0 else "-"
    print(f"  {feat:35s}  {coef:+.3f}  ({direction} = predicts {'GOOD' if coef > 0 else 'BAD'})")
```

### Method 3: Permutation importance (works with any model)

Shuffle one feature at a time and measure how much accuracy drops.
If shuffling a feature destroys accuracy, it was important.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_scaled, y, n_repeats=30, random_state=42)

importances = pd.Series(result.importances_mean, index=X.columns)
importances = importances.sort_values(ascending=False)

print("Permutation importance:")
for feat, imp in importances.items():
    print(f"  {feat:35s}  {imp:.3f}")
```

**Advantage**: works with ANY model (SVM, KNN, etc.).
**Disadvantage**: slower (needs to re-evaluate the model many times).

### Visualizing feature importance

```python
import matplotlib.pyplot as plt

importances.plot(kind="barh", figsize=(10, 6))
plt.xlabel("Importance")
plt.title("Which scoring modules matter most?")
plt.tight_layout()
plt.savefig("analysis/feature_importance.png")
plt.show()
```

---

## Evaluating your model

### Metrics

**Accuracy**: fraction of correct predictions. Simple but misleading if classes
are imbalanced (e.g. 70% GOOD, 30% BAD → predicting all GOOD gives 70% accuracy).

**Precision**: of all candidates predicted as GOOD, how many actually are?
**Recall**: of all actual GOOD candidates, how many did we find?
**F1-score**: harmonic mean of precision and recall.

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=["BAD/TRAP", "GOOD/GOLD"]))
```

### The final ranking

The real goal is not just classification but **ranking**. Use `predict_proba`
to get a probability for each candidate, then sort:

```python
# Predict probabilities on patient_zero
proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of being GOOD/GOLD

# Build ranking
ranking = sorted(zip(candidate_ids, proba, labels), key=lambda x: -x[1])

print("Final ranking:")
for i, (cid, prob, label) in enumerate(ranking, 1):
    print(f"  {i:3d}. {cid}  {prob:.3f}  {label}")
```

**Success criterion**: CAND_01 (GOLD) should be near the top.

---

## Practical tips

### Start simple

1. Load the data, encode labels, standardize
2. Train a Logistic Regression
3. Print cross-validation accuracy
4. Print feature importance
5. Predict on patient_zero and print the ranking

That's a complete pipeline. Then improve.

### Common mistakes

- **Fitting the scaler on test data**: always `fit` on training, `transform` on test
- **Not removing MEDIOCRE**: these ambiguous candidates add noise to binary classification
- **Overfitting**: if cross-validation accuracy is 60% but training accuracy is 100%, your model memorizes instead of learning. Reduce `max_depth` or `n_estimators`
- **Forgetting to align columns**: patient_zero might have different columns than patient_one if some modules failed on some candidates. Use `X_test = X_test[X_train.columns]`

### Going further

- **Hyperparameter tuning**: try different values of `max_depth`, `n_estimators`, `C`, `k` — use `GridSearchCV`
- **Feature selection**: remove the least important features and re-train — does accuracy improve?
- **Regression instead of classification**: predict the ordinal label (0-4) instead of binary (0/1) — use `RandomForestRegressor`
- **Visualization**: confusion matrix heatmap, ROC curve, PCA/t-SNE of the score matrix

---

## Deliverables

Create `analysis/groupe_XX_model.py` with:

1. Data loading and preparation
2. At least 2 models compared with cross-validation
3. Feature importance analysis (table or plot)
4. Final ranking of patient_zero candidates
5. A short interpretation: which modules matter and why (biologically)?

Run your script with:
```bash
python analysis/groupe_XX_model.py
```

It should print the results to the terminal. Save any plots to `analysis/`.
