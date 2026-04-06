"""ML pipeline — Global scoring model for neoepitope ranking.

Objective: learn the optimal combination of module scores to distinguish
good candidates (GOLD, GOOD) from bad ones (BAD, TRAP), and identify
which scoring modules are the most predictive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONSTANTS (unchanged from original)
# =============================================================================

PATIENT_ONE = PROJECT_ROOT / "analysis" / "scores_patient_one.csv"
PATIENT_ZERO = PROJECT_ROOT / "analysis" / "scores_patient_zero.csv"
PATIENT_REAL = PROJECT_ROOT / "analysis" / "scores_patient_real.csv"
PATIENT_REAL_RAW = PROJECT_ROOT / "data" / "patient_real.csv"

LABEL_MAP = {"GOLD": 1, "GOOD": 1, "BAD": 0, "TRAP": 0}
ORDINAL_MAP = {"TRAP": 0, "BAD": 1, "MEDIOCRE": 2, "GOOD": 3, "GOLD": 4}

META_COLS = {
    "candidate_id",
    "peptide_wt",
    "peptide_mut",
    "mut_pos_1based",
    "gene",
    "hla_allele",
    "note",
    "label",
    "ic50_nm",
}

RANDOM_STATE = 42


# =============================================================================
# DATA LOADER (unified, no redundancy)
# =============================================================================


class DataLoader:
    """Unified data loader for both binary and ordinal modes."""

    @staticmethod
    def _extract_label_keyword(raw: str) -> str:
        """Extract the label keyword from a raw string.

        Returns the keyword if found in LABEL_MAP, else 'UNKNOWN'.
        """
        clean = raw.strip().upper().split("—")[0].strip()
        return clean if clean in LABEL_MAP else "UNKNOWN"

    @staticmethod
    def _get_label_column(df: pd.DataFrame) -> str | None:
        """Return the first available label column name, or None."""
        for col in ("label", "note"):
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _feature_columns(df: pd.DataFrame) -> list[str]:
        """Return columns that are score features (not metadata)."""
        return [c for c in df.columns if c not in META_COLS]

    def load_training(
        self, path: Path, ordinal: bool = False
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Load training data (patient_one).

        Args:
            path: Path to CSV file
            ordinal: If True, use ordinal labels 0-4 (keeps MEDIOCRE).
                     If False, use binary labels 1/0 (removes MEDIOCRE).

        Returns:
            X: DataFrame of numeric features
            y: Series of labels (binary or ordinal)
            feature_cols: List of feature column names
        """
        if not path.exists():
            print(f"[ERROR] File not found: {path}")
            print("  -> Run: python analysis/score_analysis.py --generate")
            sys.exit(1)

        df = pd.read_csv(path)
        label_col = self._get_label_column(df)
        if label_col is None:
            print("[ERROR] No 'label' or 'note' column found in training data.")
            sys.exit(1)

        df["_label"] = df[label_col].astype(str).apply(self._extract_label_keyword)

        if ordinal:
            print("hoal")
            # Keep all labels (including MEDIOCRE) for ordinal regression
            label_map = ORDINAL_MAP
            df = df[df["_label"].isin(ORDINAL_MAP)].copy()
        else:
            print("not ordinal")
            # Remove MEDIOCRE for binary classification
            label_map = LABEL_MAP
            df = df[df["_label"].isin(LABEL_MAP)].copy()

        y = df["_label"].map(label_map)
        feature_cols = self._feature_columns(df)
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        return X, y, feature_cols

    def load_validation(
        self, path: Path, train_columns: list[str]
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Load validation data (patient_zero) aligned to training columns."""
        if not path.exists():
            print(f"[ERROR] File not found: {path}")
            print("  -> Run: python analysis/score_analysis.py --generate")
            sys.exit(1)

        df = pd.read_csv(path)
        candidate_ids = df["candidate_id"].tolist()

        label_col = self._get_label_column(df)
        if label_col:
            labels_raw = (
                df[label_col].astype(str).apply(self._extract_label_keyword).tolist()
            )
        else:
            labels_raw = ["UNKNOWN"] * len(df)

        feature_cols = self._feature_columns(df)
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X.reindex(columns=train_columns, fill_value=0.0)

        return X, candidate_ids, labels_raw

    def load_patient_real(
        self, path: Path, train_columns: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load patient_real data with IC50 merged."""
        if not path.exists():
            return pd.DataFrame(), pd.DataFrame()

        df = pd.read_csv(path)

        # Merge IC50 from raw data if available
        if PATIENT_REAL_RAW.exists():
            df_raw = pd.read_csv(PATIENT_REAL_RAW)[["candidate_id", "ic50_nm"]]
            df = df.merge(df_raw, on="candidate_id", how="left")

        feature_cols = self._feature_columns(df)
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X.reindex(columns=train_columns, fill_value=0.0)

        return X, df


# =============================================================================
# VISUALIZER (separate class for all plots)
# =============================================================================


class Visualizer:
    """Handles all visualizations."""

    def __init__(self, output_dir: Path = PROJECT_ROOT / "analysis"):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def save_plot(self, filename: str) -> None:
        """Save current plot and close."""
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
        print(f"  Plot saved -> {self.output_dir / filename}")

    def plot_feature_importance(
        self, importances: pd.Series, title: str, filename: str
    ) -> None:
        """Horizontal bar chart of feature importances."""
        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind="barh", ax=ax)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_heatmap(
        self,
        X_scaled: np.ndarray,
        labels: list[str],
        feature_names: list[str],
        filename: str,
    ) -> None:
        """Heatmap of standardized scores."""
        fig, ax = plt.subplots(figsize=(12, 8))
        df_plot = pd.DataFrame(X_scaled, columns=feature_names, index=labels)
        im = ax.imshow(df_plot.T.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_title("Score heatmap (patient_zero)")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        self.save_plot(filename)

    def plot_roc_curve(
        self, y_true: pd.Series, y_proba: np.ndarray, model_name: str, filename: str
    ) -> float:
        """ROC curve with AUC score."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            fpr, tpr, color="#e74c3c", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})"
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (cross-validated, patient_one)")
        ax.legend()
        plt.tight_layout()
        self.save_plot(filename)
        return roc_auc

    def plot_confusion_matrix(
        self, y_true: pd.Series, y_pred: pd.Series, filename: str
    ) -> None:
        """Confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["BAD/TRAP", "GOOD/GOLD"]
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix (hold-out 20%)")
        plt.tight_layout()
        self.save_plot(filename)

    def plot_spearman(
        self, df_real: pd.DataFrame, score_col: str, filename: str
    ) -> tuple[float, float]:
        """Scatter plot: pipeline score vs experimental IC50."""
        rho, pvalue = spearmanr(df_real[score_col], df_real["ic50_nm"])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            df_real[score_col],
            df_real["ic50_nm"],
            color="#e74c3c",
            alpha=0.7,
            s=60,
            edgecolors="white",
        )

        for _, row in df_real.iterrows():
            ax.annotate(
                row["candidate_id"],
                (row[score_col], row["ic50_nm"]),
                fontsize=6,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel("Pipeline RF score (higher = more likely GOOD/GOLD)", fontsize=11)
        ax.set_ylabel("Experimental IC50 (nM, lower = tighter binder)", fontsize=11)
        ax.set_title(
            f"Pipeline score vs experimental IC50 — patient_real\nSpearman ρ = {rho:+.3f}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save_plot(filename)
        return rho, pvalue


# =============================================================================
# MODEL TRAINER
# =============================================================================


class ModelTrainer:
    """Train, evaluate, and analyze models."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models: dict[str, object] = {}

    def get_models(self) -> dict[str, object]:
        """Return dictionary of models to train."""
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=self.random_state
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=self.random_state
            ),
        }

    def cross_validate(self, X_scaled: np.ndarray, y: pd.Series) -> None:
        """Compare models using 5-fold cross-validation (accuracy)."""
        print("- CROSS-VALIDATION ACCURACY (5-fold, patient_one)")
        for name, model in self.get_models().items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
            print(f"  {name:25s}  {scores.mean():.3f}  (+/- {scores.std():.3f})")
        print()

    def train_all(self, X_scaled: np.ndarray, y: pd.Series) -> dict[str, object]:
        """Train all models on full dataset."""
        self.models = {}
        for name, model in self.get_models().items():
            model.fit(X_scaled, y)
            self.models[name] = model
        return self.models

    def importance_rf(
        self, model: RandomForestClassifier, feature_names: list[str]
    ) -> pd.Series:
        """Built-in Gini importance from Random Forest."""
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(
            ascending=False
        )

    def importance_lr(
        self, model: LogisticRegression, feature_names: list[str]
    ) -> pd.Series:
        """Absolute coefficients from Logistic Regression."""
        coefs = pd.Series(model.coef_[0], index=feature_names)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    def importance_permutation(
        self, model, X_scaled: np.ndarray, y: pd.Series, feature_names: list[str]
    ) -> pd.Series:
        """Permutation importance — model-agnostic."""
        result = permutation_importance(
            model, X_scaled, y, n_repeats=30, random_state=self.random_state
        )
        return pd.Series(result.importances_mean, index=feature_names).sort_values(
            ascending=False
        )

    def hyperparameter_tuning(self, X_scaled: np.ndarray, y: pd.Series) -> None:
        """Try to beat 90% cross-validation accuracy with tuned Random Forest."""
        print("- Hyperparameter tuning (Random Forest)")
        best_score = 0.0
        best_params = {}

        for n_est in (100, 200, 300):
            for depth in (3, 5, 7, None):
                model = RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth, random_state=self.random_state
                )
                scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
                mean = scores.mean()
                depth_label = str(depth) if depth else "None"
                marker = "  <-- best so far" if mean > best_score else ""
                print(
                    f"  n_estimators={n_est:3d}  max_depth={depth_label:4s}  "
                    f"acc={mean:.3f} +/- {scores.std():.3f}{marker}"
                )
                if mean > best_score:
                    best_score = mean
                    best_params = {"n_estimators": n_est, "max_depth": depth}

        print()
        status = "YES" if best_score >= 0.90 else "NOT YET"
        print(f"  Best accuracy: {best_score:.3f} -> Beat 90%? {status}")
        print(f"  Best params  : {best_params}\n")

    def train_ordinal(
        self, X_scaled: np.ndarray, y: pd.Series
    ) -> GradientBoostingRegressor:
        """Train ordinal regression model (predicts 0-4 scale)."""
        regressor = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=self.random_state
        )
        regressor.fit(X_scaled, y)
        return regressor

    def cross_validate_ordinal(self, X_scaled: np.ndarray, y: pd.Series) -> float:
        """Cross-validate ordinal regression with MAE."""
        regressor = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=self.random_state
        )
        mae_scores = cross_val_score(
            regressor, X_scaled, y, cv=5, scoring="neg_mean_absolute_error"
        )
        return -mae_scores.mean()


# =============================================================================
# EVALUATOR
# =============================================================================


class Evaluator:
    """Evaluate models on validation sets."""

    def __init__(self, visualizer: Visualizer):
        self.viz = visualizer

    def print_ranking(
        self,
        models: dict[str, object],
        X_scaled: np.ndarray,
        candidate_ids: list[str],
        labels_raw: list[str],
    ) -> None:
        """Rank candidates by predicted probability of being GOOD/GOLD."""
        for model_name, model in models.items():
            if model_name not in ["Random Forest", "Gradient Boosting"]:
                continue

            probas = model.predict_proba(X_scaled)[:, 1]
            ranking = sorted(
                zip(candidate_ids, probas, labels_raw), key=lambda x: -x[1]
            )

            print("- FINAL RANKING (patient_zero)\n")
            cand01_rank = None
            for i, (cid, prob, label) in enumerate(ranking, 1):
                marker = "  <-- TARGET" if label == "GOLD" else ""
                if cid == "CAND_01":
                    cand01_rank = i
                print(f"  {i:3d}.  {cid:10s}  {prob:.3f}  {label}{marker}")

            print()
            if cand01_rank == 1:
                print("  SUCCESS: CAND_01 is ranked #1!")
            elif cand01_rank:
                print(f"  WARNING: CAND_01 is ranked #{cand01_rank} (target: #1)")
            else:
                print("  INFO: CAND_01 not found in patient_zero.")
            print()

    def evaluate_ordinal_ranking(
        self,
        regressor,
        scaler: StandardScaler,
        feature_names: list[str],
        X_zero_raw: pd.DataFrame,
        ids_zero: list[str],
        labels_zero: list[str],
    ) -> None:
        """Produce ordinal ranking for patient_zero."""
        X_zero = X_zero_raw.reindex(columns=feature_names, fill_value=0.0)
        X_zero_scaled = scaler.transform(X_zero)
        preds = regressor.predict(X_zero_scaled)

        ranking = sorted(zip(ids_zero, preds, labels_zero), key=lambda x: -x[1])

        print("\n  Ordinal ranking — patient_zero:")
        print(f"  {'#':>3}  {'ID':10s}  {'Score':>7s}  Label")
        print("  " + "-" * 35)
        cand01_rank = None
        for i, (cid, score, label) in enumerate(ranking, 1):
            marker = "  <-- TARGET" if label == "GOLD" else ""
            if cid == "CAND_01":
                cand01_rank = i
            print(f"  {i:3}.  {cid:10s}  {score:5.2f}/4  {label}{marker}")

        print()
        if cand01_rank == 1:
            print("  SUCCESS: CAND_01 is ranked #1 with ordinal regression!")
        elif cand01_rank:
            print(f"  WARNING: CAND_01 is ranked #{cand01_rank} (target: #1)")
        print()

    def run_ordinal_pipeline(
        self, loader, trainer, feature_names, X_zero_raw, ids_zero, labels_zero
    ):
        """Run full ordinal regression pipeline (bonus)."""
        X_ord_raw, y_ord, ord_feature_names = loader.load_training(
            PATIENT_ONE, ordinal=True
        )

        if not X_ord_raw.empty:
            scaler_ord = StandardScaler()
            X_ord_scaled = scaler_ord.fit_transform(X_ord_raw)

            mae = trainer.cross_validate_ordinal(X_ord_scaled, y_ord)
            print(f"  Candidates (all labels): {len(y_ord)}")
            label_dist = y_ord.value_counts().sort_index().to_dict()
            print(f"  Label distribution     : {label_dist}")
            print(f"  MAE (5-fold CV)        : {mae:.3f}  (scale: 0=TRAP ... 4=GOLD)")
            if mae < 0.5:
                print("  Predictions are on average < 0.5 grade off.\n")
            else:
                print("  Consider more features or tuning.\n")

            ord_reg = trainer.train_ordinal(X_ord_scaled, y_ord)

            # Sample predictions
            preds_train = ord_reg.predict(X_ord_scaled)
            inv_map = {v: k for k, v in ORDINAL_MAP.items()}
            print("  Sample predictions — patient_one (first 10):")
            print(f"  {'Actual':10s}  {'Predicted':10s}  {'Score':>5s}  {'Error':>6s}")
            print("  " + "-" * 40)
            for actual, pred in list(zip(y_ord, preds_train))[:10]:
                label_actual = inv_map.get(int(round(actual)), "?")
                label_pred = inv_map.get(int(round(pred)), "?")
                error = abs(actual - pred)
                print(
                    f"  {label_actual:10s}  {label_pred:10s}  {pred:5.2f}  {error:6.2f}"
                )

            self.evaluate_ordinal_ranking(
                ord_reg,
                scaler_ord,
                ord_feature_names,
                X_zero_raw,
                ids_zero,
                labels_zero,
            )
        else:
            print("  [SKIP] Could not load ordinal data.\n")

    def evaluate_patient_real(
        self,
        model_rf: RandomForestClassifier,
        model_gb: GradientBoostingClassifier,
        scaler: StandardScaler,
        feature_names: list[str],
    ) -> dict:
        """Evaluate on patient_real: Spearman correlation + REAL/DECOY AUC."""
        loader = DataLoader()

        if not PATIENT_REAL.exists():
            print(
                f"  [SKIP] {PATIENT_REAL} not found.\n"
                "  -> Run: python analysis/score_analysis.py --generate\n"
            )
            return {}

        X_real, df = loader.load_patient_real(PATIENT_REAL, feature_names)
        if X_real.empty:
            return {}

        X_real_scaled = scaler.transform(X_real)
        rf_proba = model_rf.predict_proba(X_real_scaled)[:, 1]
        gb_proba = model_gb.predict_proba(X_real_scaled)[:, 1]

        df = df.copy()
        df["rf_score"] = rf_proba
        df["gb_score"] = gb_proba

        results = {}

        # Task 1: Spearman correlation with IC50
        real_mask = df["label"].astype(str).str.upper().str.startswith("REAL")
        df_real = df[real_mask].copy().dropna(subset=["ic50_nm"])

        if len(df_real) > 0:
            rho_rf, p_rf = spearmanr(df_real["rf_score"], df_real["ic50_nm"])
            rho_gb, p_gb = spearmanr(df_real["gb_score"], df_real["ic50_nm"])
            results["spearman_rho_rf"] = rho_rf
            results["spearman_p_rf"] = p_rf
            results["spearman_rho_gb"] = rho_gb
            results["spearman_p_gb"] = p_gb

            print(
                f"\n  Spearman correlation (RF vs IC50): ρ = {rho_rf:+.3f} (p={p_rf:.4f})"
            )
            print(
                f"  Spearman correlation (GB vs IC50): ρ = {rho_gb:+.3f} (p={p_gb:.4f})"
            )

            # Plot with RF (or both? choix: RF)
            rho, pvalue = self.viz.plot_spearman(
                df_real, "rf_score", "patient_real_spearman.png"
            )

            if rho < 0 and pvalue < 0.05:
                print(
                    "   Significant negative correlation: higher score -> lower IC50 -> biologically valid."
                )
            elif pvalue >= 0.05:
                print(
                    "   No significant correlation with IC50 (expected: modules are proxies)."
                )
            else:
                print(f"  Unexpected direction: ρ = {rho:+.3f}")

        # Task 2: REAL vs DECOY classification
        binary = df["label"].astype(str).str.upper().str.startswith("REAL").astype(int)
        auc_rf = roc_auc_score(binary, df["rf_score"])
        auc_gb = roc_auc_score(binary, df["gb_score"])
        n_real = binary.sum()
        n_decoy = (binary == 0).sum()

        print(f"\n  REAL vs DECOY ({n_real} REAL / {n_decoy} DECOY):")
        print(f"    AUC — Random Forest: {auc_rf:.3f}")
        print(f"    AUC — Gradient Boosting: {auc_gb:.3f}")
        results["auc_rf"] = auc_rf
        results["auc_gb"] = auc_gb

        if auc_rf >= 0.80:
            print("   Pipeline clearly separates REAL from DECOY.")
        elif auc_rf >= 0.65:
            print("   Partial separation of REAL from DECOY.")
        else:
            print("   Weak discrimination, training labels may not generalize.")

        # Save full table
        out_csv = PROJECT_ROOT / "analysis" / "patient_real_evaluation.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n  Full table saved -> {out_csv}\n")

        return results


# =============================================================================
# SHAP EXPLAINABILITY
# =============================================================================
class ShapExplainer:
    """SHAP explainability for per-candidate predictions."""

    def __init__(self, visualizer):
        self.viz = visualizer

    def explain(
        self,
        model,
        X_test: "np.ndarray",
        X_test_raw: "pd.DataFrame",
        candidate_ids: list,
        labels_raw: list,
    ) -> None:
        """
        Generate SHAP explanations for model predictions.
        Deliverables:
          - analysis/shap_summary.png         (global beeswarm)
          - analysis/shap_waterfall_cand_01.png
          - analysis/shap_waterfall_gold.png
          - analysis/shap_waterfall_bad.png
        """
        try:
            import shap
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
            return

        print("=" * 60)
        print("SHAP EXPLAINABILITY")
        print("=" * 60)

        # ── Rebuild a clean DataFrame from the scaled array ───────────────
        feature_names = list(X_test_raw.columns)
        X_df = pd.DataFrame(X_test, columns=feature_names)

        # ── Compute SHAP values ───────────────────────────────────────────
        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_df)

        if isinstance(raw_shap, list):
            # Classic format — two arrays, one per class
            sv = raw_shap[1]  # class 1 = GOOD/GOLD
            base = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else float(explainer.expected_value)
            )
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            # New format — single 3-D array (n_samples, n_features, n_classes)
            sv = raw_shap[:, :, 1]
            base = (
                explainer.expected_value[1]
                if hasattr(explainer.expected_value, "__len__")
                else float(explainer.expected_value)
            )
        else:
            # Already a plain 2-D array (regression or single-output)
            sv = raw_shap
            base = float(explainer.expected_value)

        # Safety check — shapes must now match
        assert (
            sv.shape == X_df.shape
        ), f"SHAP shape mismatch: sv={sv.shape}  X_df={X_df.shape}"

        probas = model.predict_proba(X_test)[:, 1]

        # ── 1. Global summary (beeswarm) ──────────────────────────────────
        print("\n[1/3] Generating SHAP summary plot...")
        plt.figure(figsize=(12, max(6, len(feature_names) * 0.45)))
        shap.summary_plot(sv, X_df, feature_names=feature_names, show=False)
        plt.title(
            "SHAP Summary — patient_zero\n"
            "Each dot = one candidate.  Red = high feature value.  "
            "Right = pushes toward GOOD/GOLD.",
            fontsize=10,
            pad=10,
        )
        plt.tight_layout()
        self.viz.save_plot("shap_summary.png")
        print("  → Red dot on the right  : high score value pushes toward GOOD/GOLD")
        print("  → Blue dot on the left  : low score value pushes toward BAD/TRAP")

        # ── 2. Waterfall plots (per candidate) ────────────────────────────
        print("\n[2/3] Generating waterfall plots...")

        cand01_idx = (
            candidate_ids.index("CAND_01") if "CAND_01" in candidate_ids else None
        )
        gold_idx = next(
            (i for i, lbl in enumerate(labels_raw) if lbl in ("GOLD", "GOOD")), None
        )
        bad_idx = next(
            (i for i, lbl in enumerate(labels_raw) if lbl in ("BAD", "TRAP")), None
        )

        targets = []
        if cand01_idx is not None:
            targets.append(("cand_01", cand01_idx))
        if gold_idx is not None and gold_idx != cand01_idx:
            targets.append(("gold", gold_idx))
        if bad_idx is not None:
            targets.append(("bad", bad_idx))

        for tag, idx in targets:
            cid = candidate_ids[idx]
            label = labels_raw[idx] if idx < len(labels_raw) else "UNKNOWN"

            exp = shap.Explanation(
                values=sv[idx],
                base_values=base,
                data=X_df.iloc[idx].values,
                feature_names=feature_names,
            )

            plt.figure(figsize=(10, max(5, len(feature_names) * 0.42)))
            shap.waterfall_plot(exp, show=False, max_display=12)
            plt.title(
                f"SHAP Waterfall — {cid}  [{label}]  "
                f"(predicted prob = {probas[idx]:.3f})\n"
                "Each bar = contribution of one module to the final prediction.",
                fontsize=9,
                pad=8,
            )
            plt.tight_layout()
            self.viz.save_plot(f"shap_waterfall_{tag}.png")

        # ── 3. Biological interpretation ──────────────────────────────────
        print("\n[3/3] Biological interpretation")
        print("─" * 50)

        mean_abs = np.abs(sv).mean(axis=0)
        top_importance = pd.Series(mean_abs, index=feature_names).sort_values(
            ascending=False
        )

        print("\n  Mean |SHAP| per module (global):")
        for feat, val in top_importance.items():
            bar = "█" * max(1, int(val * 60))
            print(f"    {feat:35s}  {val:.4f}  {bar}")

        top_feat = top_importance.index[0]
        print(f"\n  ► Most impactful module globally : '{top_feat}'")

        # Why CAND_01 is where it is
        if cand01_idx is not None:
            rank = int(np.argsort(-probas).tolist().index(cand01_idx)) + 1
            print(f"\n  Why CAND_01 is ranked #{rank}:")
            cand_shap = sv[cand01_idx]
            contributions = sorted(
                zip(feature_names, cand_shap), key=lambda x: -abs(x[1])
            )
            for feat, val in contributions[:5]:
                arrow = "↑" if val > 0 else "↓"
                print(f"    {arrow} {feat}: {val:+.4f}")

        # Why a BAD candidate is at the bottom
        if bad_idx is not None:
            cid_bad = candidate_ids[bad_idx]
            rank_bad = int(np.argsort(-probas).tolist().index(bad_idx)) + 1
            print(f"\n  Why {cid_bad} (BAD) is ranked #{rank_bad}:")
            bad_shap = sv[bad_idx]
            contributions = sorted(zip(feature_names, bad_shap), key=lambda x: x[1])
            for feat, val in contributions[:5]:
                arrow = "↑" if val > 0 else "↓"
                print(f"    {arrow} {feat}: {val:+.4f}")

        print(
            "\n  ► Reading the waterfall plots:\n"
            "    f(x) = final predicted probability for this candidate.\n"
            "    E[f(x)] = baseline probability (average over all candidates).\n"
            "    Red bars push the score UP (toward GOOD/GOLD).\n"
            "    Blue bars push the score DOWN (toward BAD/TRAP).\n"
            "    The bar length = magnitude of that module's contribution.\n"
        )
        print("=" * 60)


# =============================================================================
# FEATURE SELECTION
# =============================================================================


def feature_selection(
    X_scaled: np.ndarray, y: pd.Series, feature_names: list[str], importances: pd.Series
) -> None:
    """Try removing the least important features and compare accuracy."""
    print("- Feature selection (remove least important)")

    n_total = len(feature_names)
    thresholds = {
        "All features": n_total,
        "Top 75%": max(1, int(n_total * 0.75)),
        "Top 50%": max(1, int(n_total * 0.50)),
        "Top 25%": max(1, int(n_total * 0.25)),
    }

    top_features = importances.index.tolist()

    for label, k in thresholds.items():
        selected = [feature_names.index(f) for f in top_features[:k]]
        X_sub = X_scaled[:, selected]
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE
        )
        scores = cross_val_score(model, X_sub, y, cv=5, scoring="accuracy")
        print(
            f"  {label:15s} ({k:2d} features)  acc = {scores.mean():.3f} "
            f"+/- {scores.std():.3f}"
        )
    print()


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main():
    print("\nOpen-NeoVax -- ML Pipeline")
    print("=" * 60)

    # Initialize components
    loader = DataLoader()
    viz = Visualizer()
    trainer = ModelTrainer()
    evaluator = Evaluator(viz)

    # -------------------------------------------------------------------------
    # 1. Load training data (binary classification)
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading training data (patient_one)...")
    X_train_raw, y_train, feature_names = loader.load_training(
        PATIENT_ONE, ordinal=False
    )
    print(f"  {len(y_train)} candidates  x  {len(feature_names)} features")
    print(
        f"  Positives (GOLD/GOOD): {y_train.sum()} | "
        f"Negatives (BAD/TRAP): {(y_train == 0).sum()}\n"
    )

    # -------------------------------------------------------------------------
    # 2. Standardize — fit ONLY on training data
    # -------------------------------------------------------------------------
    X_train_scaled = trainer.scaler.fit_transform(X_train_raw)

    # -------------------------------------------------------------------------
    # 3. Compare models with cross-validation
    # -------------------------------------------------------------------------
    print("[2/6] Comparing models (cross-validation)...")
    trainer.cross_validate(X_train_scaled, y_train)

    # -------------------------------------------------------------------------
    # 4. Train final models on full training set
    # -------------------------------------------------------------------------
    print("[3/6] Training final models on full patient_one dataset...")
    models = trainer.train_all(X_train_scaled, y_train)

    rf = models["Random Forest"]
    lr = models["Logistic Regression"]
    gb = models["Gradient Boosting"]

    # -------------------------------------------------------------------------
    # 5. Feature importance analysis
    # -------------------------------------------------------------------------
    print("[4/6] Feature importance analysis...\n")

    imp_rf = trainer.importance_rf(rf, feature_names)
    imp_lr = trainer.importance_lr(lr, feature_names)
    imp_perm = trainer.importance_permutation(
        rf, X_train_scaled, y_train, feature_names
    )

    # Print feature importances
    print("FEATURE IMPORTANCE -- Random Forest (Gini, built-in)")
    for feat, imp in imp_rf.head(10).items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:35s}  {imp:.3f}  {bar}")

    print("\nFEATURE IMPORTANCE -- Logistic Regression (coefficients)")
    for feat, coef in imp_lr.head(10).items():
        direction = "+" if coef > 0 else "-"
        bar = "#" * int(abs(coef) * 30)
        print(
            f"  {feat:35s}  {coef:+.3f}  {bar}  "
            f"({direction} = predicts {'GOOD' if coef > 0 else 'BAD'})"
        )

    print("\nFEATURE IMPORTANCE -- Permutation importance (Random Forest)")
    for feat, imp in imp_perm.head(10).items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:35s}  {imp:.3f}  {bar}")

    # Plot feature importances
    viz.plot_feature_importance(
        imp_rf, "Feature importance -- Random Forest", "feature_importance_rf.png"
    )

    # Hold-out classification report
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    rf_ho = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=RANDOM_STATE
    )
    rf_ho.fit(X_tr, y_tr)
    print("\nClassification Report (hold-out 20%):")
    print(
        classification_report(
            y_te, rf_ho.predict(X_te), target_names=["BAD/TRAP", "GOOD/GOLD"]
        )
    )

    # -------------------------------------------------------------------------
    # 6. Validate on patient_zero
    # -------------------------------------------------------------------------
    print("\n[5/6] Validation on patient_zero...")
    X_zero_raw, ids_zero, labels_zero = loader.load_validation(
        PATIENT_ZERO, feature_names
    )
    X_zero_scaled = trainer.scaler.transform(X_zero_raw)

    evaluator.print_ranking(
        {"Random Forest": rf, "Gradient Boosting": gb},
        X_zero_scaled,
        ids_zero,
        labels_zero,
    )

    # -------------------------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------------------------
    feature_selection(X_train_scaled, y_train, feature_names, imp_rf)

    # -------------------------------------------------------------------------
    # Visualizations (heatmap, ROC curve, confusion matrix)
    # -------------------------------------------------------------------------
    viz.plot_heatmap(X_zero_scaled, ids_zero, feature_names, "score_heatmap.png")

    y_proba_cv = cross_val_predict(
        rf, X_train_scaled, y_train, cv=5, method="predict_proba"
    )[:, 1]
    viz.plot_roc_curve(y_train, y_proba_cv, "Random Forest", "roc_curve.png")
    viz.plot_confusion_matrix(y_te, rf_ho.predict(X_te), "confusion_matrix.png")

    # -------------------------------------------------------------------------
    # Hyperparameter tuning
    # -------------------------------------------------------------------------
    trainer.hyperparameter_tuning(X_train_scaled, y_train)

    # -------------------------------------------------------------------------
    # Ordinal regression
    # -------------------------------------------------------------------------
    print("[6/6] Ordinal regression ...")
    evaluator.run_ordinal_pipeline(
        loader, trainer, feature_names, X_zero_raw, ids_zero, labels_zero
    )

    # -------------------------------------------------------------------------
    # Patient real evaluation
    # -------------------------------------------------------------------------
    print("\n[Bonus] Evaluating on patient_real.csv...")
    evaluator.evaluate_patient_real(rf, gb, trainer.scaler, feature_names)

    # -------------------------------------------------------------------------
    # SHAP explainability
    # -------------------------------------------------------------------------
    print("\n[Bonus] SHAP explainability...")
    shap_explainer = ShapExplainer(viz)
    shap_explainer.explain(
        model=rf,
        X_test=X_zero_scaled,
        X_test_raw=X_zero_raw,
        candidate_ids=ids_zero,
        labels_raw=labels_zero,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
