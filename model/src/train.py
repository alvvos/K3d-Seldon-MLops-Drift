from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from .data import load_or_create_dataset
    from .schema import (
        CLASS_NAMES,
        DEFAULT_DECISION_THRESHOLD,
        FEATURE_NAMES,
        MODEL_VERSION,
        MONETARY_FEATURES,
        PROFILE_FEATURES,
        RATE_AND_SCORE_FEATURES,
        TARGET_COLUMN,
    )
except ImportError:
    from data import load_or_create_dataset
    from schema import (
        CLASS_NAMES,
        DEFAULT_DECISION_THRESHOLD,
        FEATURE_NAMES,
        MODEL_VERSION,
        MONETARY_FEATURES,
        PROFILE_FEATURES,
        RATE_AND_SCORE_FEATURES,
        TARGET_COLUMN,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = MODEL_ROOT / "data"
DEFAULT_ARTIFACTS_DIR = MODEL_ROOT / "artifacts"


def build_preprocessor() -> ColumnTransformer:
    """Construye el ColumnTransformer diferenciado por tipo de distribución."""
    monetary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    rates_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    profile_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("monetary", monetary_pipeline, MONETARY_FEATURES),
            ("rates_and_scores", rates_pipeline, RATE_AND_SCORE_FEATURES),
            ("profile", profile_pipeline, PROFILE_FEATURES),
        ],
        remainder="drop",
    )


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define los clasificadores candidatos a evaluar mediante validación cruzada."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.85,
            random_state=random_state,
        ),
    }


def _extract_feature_importances(pipeline: Pipeline) -> Dict[str, float]:
    """Extrae importancias o magnitudes de coeficientes ordenadas por feature."""
    ordered_features: List[str] = MONETARY_FEATURES + RATE_AND_SCORE_FEATURES + PROFILE_FEATURES
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        raw = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        raw = np.abs(np.asarray(classifier.coef_, dtype=float)).ravel()
        total = float(np.sum(raw))
        raw = raw / total if total > 0 else raw
    else:
        return {}

    importances = {
        feat: round(float(score), 4) for feat, score in zip(ordered_features, raw.tolist())
    }
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))


def _compute_feature_stats(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calcula estadísticas descriptivas de referencia por variable."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in FEATURE_NAMES:
        series = X[col].dropna()
        stats[col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std(ddof=0)), 4),
            "min": round(float(series.min()), 4),
            "p05": round(float(series.quantile(0.05)), 4),
            "median": round(float(series.median()), 4),
            "p95": round(float(series.quantile(0.95)), 4),
            "max": round(float(series.max()), 4),
        }
    return stats


def train_and_evaluate(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_ARTIFACTS_DIR,
    n_samples: int = 5000,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    reference_sample_size: int = 1000,
    force_regenerate: bool = False,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Ejecuta el flujo completo de entrenamiento, selección y persistencia."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "credit_risk.csv"
    df = load_or_create_dataset(
        csv_path=dataset_path,
        n_samples=n_samples,
        random_state=random_state,
        force_regenerate=force_regenerate,
    )
    logger.info(
        "Dataset cargado (%s): %d filas, tasa de default=%.2f%%",
        dataset_path,
        len(df),
        df[TARGET_COLUMN].mean() * 100.0,
    )

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_cv_auc = -1.0
    best_pipeline: Pipeline | None = None

    for name, estimator in get_candidate_models(random_state=random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator),
            ]
        )
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))
        cv_results[name] = {
            "cv_roc_auc_mean": round(mean_auc, 4),
            "cv_roc_auc_std": round(std_auc, 4),
        }
        logger.info("Modelo candidato %-20s | CV ROC-AUC: %.4f (+/- %.4f)", name, mean_auc, std_auc)

        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None
    logger.info(
        "Modelo ganador: %s (CV ROC-AUC=%.4f). Reentrenando en train...",
        best_name,
        best_cv_auc,
    )
    best_pipeline.fit(X_train, y_train)

    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DEFAULT_DECISION_THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, y_pred).tolist()

    test_metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_proba)), 4),
        "confusion_matrix": cm,
    }
    logger.info(
        "Métricas en test | ROC-AUC=%.4f | F1=%.4f | Precision=%.4f | Recall=%.4f",
        test_metrics["roc_auc"],
        test_metrics["f1"],
        test_metrics["precision"],
        test_metrics["recall"],
    )

    # Guardar muestra limpia de referencia para el detector de Data Drift (KSDrift)
    ref_clean = X_train.fillna(X_train.median(numeric_only=True))
    ref_n = min(reference_sample_size, len(ref_clean))
    ref_sample = ref_clean.sample(n=ref_n, random_state=random_state)
    reference_path = data_dir / "reference_data.csv"
    ref_sample.to_csv(reference_path, index=False)
    logger.info("Distribución de referencia guardada en %s (%d muestras)", reference_path, ref_n)

    model_path = output_dir / "model.joblib"
    joblib.dump(best_pipeline, model_path)

    metadata: Dict[str, Any] = {
        "model_name": "credit-risk-classifier",
        "model_version": MODEL_VERSION,
        "selected_algorithm": best_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "class_names": CLASS_NAMES,
        "decision_threshold": DEFAULT_DECISION_THRESHOLD,
        "dataset": {
            "n_total": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate_train": round(float(y_train.mean()), 4),
            "positive_rate_test": round(float(y_test.mean()), 4),
        },
        "cv_comparison": cv_results,
        "test_metrics": test_metrics,
        "feature_importances": _extract_feature_importances(best_pipeline),
        "feature_stats": _compute_feature_stats(X_train),
    }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Artefactos persistidos en %s y %s", model_path, metadata_path)

    return best_pipeline, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y exporta el modelo de riesgo crediticio."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force-regenerate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        force_regenerate=args.force_regenerate,
    )
