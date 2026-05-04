"""
Groupe 15 - Global Scoring Model
ML-based candidate ranking & module importance analysis
Issue #39
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ANALYSIS_DIR = Path(__file__).resolve().parent

LABEL_ORDER = {"TRAP": 0, "BAD": 1, "MEDIOCRE": 2, "GOOD": 3, "GOLD": 4, "NEUTRAL": 2}
BINARY_MAP = {"GOLD": 1, "GOOD": 1, "MEDIOCRE": 0, "BAD": 0, "TRAP": 0, "NEUTRAL": 0}

# ─────────────────────────────────────────────
#  1. CHARGEMENT & PRÉPARATION
# ─────────────────────────────────────────────


def load_scores(filename: str) -> pd.DataFrame:
    path = ANALYSIS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable.\n"
            "Lance d'abord : python analysis/score_analysis.py --generate"
        )
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    feature_cols = [
        c for c in df.columns if c not in ("candidate_id", "label", "target")
    ]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Remplace les valeurs aberrantes (-1000, -10, -20)
    # par des valeurs minimales clippées
    for col in X.columns:
        p1 = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=p1)
    X = X.fillna(0)

    y_ord = df["label"].map(LABEL_ORDER).fillna(2).astype(int)  # ordinal 0-4
    y_binary = df["label"].map(BINARY_MAP).fillna(0).astype(int)  # binaire

    return X, y_ord, y_binary, df["candidate_id"], feature_cols


# ─────────────────────────────────────────────
#  2. ENTRAÎNEMENT
# ─────────────────────────────────────────────


def train_models(X, y_ord, y_binary, feature_cols):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n── Cross-validation (5-fold) ──")

    # Modèle 1 : Logistic Regression (binaire)
    lr = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y_binary, cv=cv, scoring="accuracy")
    lr.fit(X_scaled, y_binary)
    print(
        f"LogisticRegression (binaire)   : "
        f"{lr_scores.mean():.3f} ± {lr_scores.std():.3f}"
    )

    # Modèle 2 : Random Forest (ordinal)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    rf_scores = cross_val_score(rf, X, y_ord, cv=cv, scoring="accuracy")
    rf.fit(X, y_ord)
    print(
        f"RandomForest (ordinal 0-4)     : "
        f"{rf_scores.mean():.3f} ± {rf_scores.std():.3f}"
    )

    # Modèle 3 : Gradient Boosting (ordinal)
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42
    )
    gb_scores = cross_val_score(gb, X, y_ord, cv=cv, scoring="accuracy")
    gb.fit(X, y_ord)
    print(
        f"GradientBoosting (ordinal 0-4) : "
        f"{gb_scores.mean():.3f} ± {gb_scores.std():.3f}"
    )

    return {
        "LogisticRegression": (lr, scaler),
        "RandomForest": (rf, None),
        "GradientBoosting": (gb, None),
    }


# ─────────────────────────────────────────────
#  3. IMPORTANCE DES FEATURES
# ─────────────────────────────────────────────


def print_feature_importance(model, feature_cols, model_name):
    print(f"\nFEATURE IMPORTANCE ({model_name}):")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])
        importances = importances / importances.sum()

    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in ranked[:10]:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<35} {imp:.3f}  {bar}")

    top_n = 10
    names = [r[0] for r in ranked[:top_n]]
    values = [r[1] for r in ranked[:top_n]]

    plt.figure(figsize=(10, 5))
    plt.barh(names[::-1], values[::-1], color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} features — {model_name} (Groupe 15)")
    plt.tight_layout()
    out = ANALYSIS_DIR / f"groupe_15_importance_{model_name.lower()}.png"
    plt.savefig(out)
    plt.close()
    print(f"  → Plot sauvegardé : {out}")


# ─────────────────────────────────────────────
#  4. SCORING DE CONSENSUS (ensemble)
# ─────────────────────────────────────────────


def consensus_score(trained, X, X_scaled):
    """Combine les 3 modèles : score normalisé entre 0 et 1."""
    rf, _ = trained["RandomForest"]
    gb, _ = trained["GradientBoosting"]
    lr, _ = trained["LogisticRegression"]

    # Probabilité d'être GOLD (classe 4) pour RF et GB
    rf_proba = rf.predict_proba(X)
    gb_proba = gb.predict_proba(X)

    # Score ordinal normalisé : somme pondérée des probabilités × rang
    classes = rf.classes_
    rf_score = (
        np.array([sum(p * c for p, c in zip(row, classes)) for row in rf_proba]) / 4
    )
    gb_score = (
        np.array([sum(p * c for p, c in zip(row, gb.classes_)) for row in gb_proba]) / 4
    )

    # LR : proba classe positive
    lr_score = lr.predict_proba(X_scaled)[:, 1]

    # Moyenne pondérée (GB et RF ont plus de poids car ordinal)
    final = 0.35 * rf_score + 0.45 * gb_score + 0.20 * lr_score
    return final


# ─────────────────────────────────────────────
#  5. VALIDATION PATIENT ZERO
# ─────────────────────────────────────────────


def validate_patient_zero(trained, feature_cols):
    df_val = load_scores("scores_patient_zero.csv")
    X_val = df_val[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Même clipping que l'entraînement
    df_train = load_scores("scores_patient_one.csv")
    X_train = df_train[feature_cols].apply(pd.to_numeric, errors="coerce")
    for col in X_val.columns:
        p1 = X_train[col].quantile(0.01)
        X_val[col] = X_val[col].clip(lower=p1)
    X_val = X_val.fillna(0)

    _, scaler = trained["LogisticRegression"]
    X_val_scaled = scaler.transform(X_val)

    scores = consensus_score(trained, X_val, X_val_scaled)
    ids = df_val["candidate_id"].values
    labels = df_val["label"].values

    ranking = sorted(zip(ids, scores, labels), key=lambda x: x[1], reverse=True)

    print("\nFINAL RANKING (patient_zero) — Ensemble (RF+GB+LR):")
    for i, (cid, score, label) in enumerate(ranking, 1):
        arrow = "  <-- ★" if label == "GOLD" else ""
        print(f"  {i:>2}. {cid:<10} {score:.3f}  {label}{arrow}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  GROUPE 15 — ML Scoring Model (Issue #39)")
    print("=" * 55)

    df_train = load_scores("scores_patient_one.csv")
    X, y_ord, y_binary, ids, feature_cols = prepare_data(df_train)
    print(f"\nDonnées : {len(X)} candidats, {len(feature_cols)} features")
    distrib = pd.Series(y_ord).value_counts().sort_index().to_dict()
    print(f"Distribution ordinale : {distrib}")

    trained = train_models(X, y_ord, y_binary, feature_cols)

    rf, _ = trained["RandomForest"]
    gb, _ = trained["GradientBoosting"]
    lr, _ = trained["LogisticRegression"]
    print_feature_importance(rf, feature_cols, "RandomForest")
    print_feature_importance(gb, feature_cols, "GradientBoosting")
    print_feature_importance(lr, feature_cols, "LogisticRegression")

    validate_patient_zero(trained, feature_cols)
    print("\nTerminé !")
