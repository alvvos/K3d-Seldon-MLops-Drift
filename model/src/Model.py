from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    from .schema import CLASS_NAMES, DEFAULT_DECISION_THRESHOLD, FEATURE_NAMES, MODEL_VERSION
except ImportError:
    from schema import CLASS_NAMES, DEFAULT_DECISION_THRESHOLD, FEATURE_NAMES, MODEL_VERSION

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_artifact_path(
    env_var: str,
    filename: str,
    explicit_path: Optional[str | Path],
) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value)

    src_dir = Path(__file__).resolve().parent
    candidates = [
        src_dir / "artifacts" / filename,
        src_dir.parent / "artifacts" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


class Model:
    """Envoltorio Seldon Core para el clasificador de riesgo crediticio."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        metadata_path: Optional[str | Path] = None,
    ) -> None:
        self.model_path = _resolve_artifact_path("MODEL_PATH", "model.joblib", model_path)
        self.metadata_path = _resolve_artifact_path("METADATA_PATH", "metadata.json", metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto del modelo en '{self.model_path}'. "
                "Ejecuta 'make train' o 'python model/src/train.py' primero."
            )

        self.pipeline = joblib.load(self.model_path)
        self.metadata: Dict[str, Any] = self._load_metadata(self.metadata_path)
        self.feature_names: List[str] = list(self.metadata.get("feature_names", FEATURE_NAMES))
        self.model_version: str = str(self.metadata.get("model_version", MODEL_VERSION))
        self.algorithm: str = str(self.metadata.get("selected_algorithm", "unknown"))

        threshold_env = os.getenv("DECISION_THRESHOLD")
        self.decision_threshold: float = (
            float(threshold_env)
            if threshold_env is not None
            else float(self.metadata.get("decision_threshold", DEFAULT_DECISION_THRESHOLD))
        )

        # Métricas internas exportadas a Prometheus vía Seldon Core (`metrics()`)
        self._last_batch_size: int = 0
        self._last_high_risk_count: int = 0
        self._last_latency_ms: float = 0.0
        self._last_avg_default_prob: float = 0.0
        self._error_count: int = 0

        self.ready: bool = True
        logger.info(
            "Modelo '%s' v%s (%s) cargado desde %s | threshold=%.2f",
            self.metadata.get("model_name", "credit-risk-classifier"),
            self.model_version,
            self.algorithm,
            self.model_path,
            self.decision_threshold,
        )

    @staticmethod
    def _load_metadata(path: Path) -> Dict[str, Any]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        logger.warning("No se encontró metadata en %s; usando configuración por defecto.", path)
        return {}

    def _validate_and_prepare(
        self,
        X: Any,
        features_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if X is None:
            raise ValueError("El tensor de entrada X no puede ser None.")

        try:
            arr = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("La entrada X debe contener únicamente valores numéricos.") from exc

        if arr.size == 0:
            raise ValueError("La entrada X está vacía.")

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(
                f"Se esperaba un tensor 2D (n_samples, n_features); recibido {arr.ndim}D."
            )

        if np.isinf(arr).any():
            raise ValueError("La entrada X contiene valores infinitos no permitidos.")

        expected_n = len(self.feature_names)
        if arr.shape[1] != expected_n:
            raise ValueError(
                f"Número de features inválido: recibidas {arr.shape[1]}, esperadas {expected_n} "
                f"({self.feature_names})."
            )

        if features_names is not None and len(features_names) > 0:
            provided = list(features_names)
            if len(provided) != expected_n:
                raise ValueError(
                    f"features_names tiene {len(provided)} nombres, pero se esperaban {expected_n}."
                )
            if set(provided) != set(self.feature_names):
                missing = sorted(set(self.feature_names) - set(provided))
                extra = sorted(set(provided) - set(self.feature_names))
                raise ValueError(
                    f"features_names no coincide con el esquema esperado. "
                    f"Faltan={missing}, sobran={extra}."
                )
            df = pd.DataFrame(arr, columns=provided)[self.feature_names]
        else:
            df = pd.DataFrame(arr, columns=self.feature_names)

        return df

    def predict(
        self,
        X: Iterable[Any],
        features_names: Optional[Sequence[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """Devuelve probabilidades [[p_approved, p_default], ...] para cada muestra."""
        start = time.perf_counter()
        try:
            df = self._validate_and_prepare(X, features_names=features_names)
            probas = self.pipeline.predict_proba(df)
        except Exception:
            self._error_count += 1
            logger.exception("Error durante la inferencia del modelo.")
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        default_probs = probas[:, 1]
        high_risk_count = int(np.sum(default_probs >= self.decision_threshold))

        self._last_batch_size = int(len(df))
        self._last_high_risk_count = high_risk_count
        self._last_latency_ms = round(latency_ms, 3)
        self._last_avg_default_prob = round(float(np.mean(default_probs)), 4)

        logger.info(
            "Inferencia completada | batch=%d | high_risk=%d | avg_p_default=%.4f | latency=%.2fms",
            self._last_batch_size,
            self._last_high_risk_count,
            self._last_avg_default_prob,
            self._last_latency_ms,
        )
        return np.round(probas, 6).tolist()

    def class_names(self) -> List[str]:
        return list(self.metadata.get("class_names", CLASS_NAMES))

    def tags(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "algorithm": self.algorithm,
            "decision_threshold": self.decision_threshold,
        }

    def metrics(self) -> List[Dict[str, Any]]:
        """Métricas custom recogidas por el sidecar de Seldon y expuestas a Prometheus."""
        return [
            {
                "type": "COUNTER",
                "key": "credit_risk_predictions_total",
                "value": self._last_batch_size,
            },
            {
                "type": "COUNTER",
                "key": "credit_risk_high_risk_predictions_total",
                "value": self._last_high_risk_count,
            },
            {
                "type": "GAUGE",
                "key": "credit_risk_avg_default_probability",
                "value": self._last_avg_default_prob,
            },
            {
                "type": "TIMER",
                "key": "credit_risk_inference_duration_ms",
                "value": self._last_latency_ms,
            },
        ]

    def health_status(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.ready else "not_ready",
            "model_version": self.model_version,
            "algorithm": self.algorithm,
        }

    def init_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.metadata.get("model_name", "credit-risk-classifier"),
            "versions": [self.model_version],
            "platform": "scikit-learn",
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP64",
                    "shape": [-1, len(self.feature_names)],
                }
            ],
            "outputs": [
                {
                    "name": "probabilities",
                    "datatype": "FP64",
                    "shape": [-1, len(self.class_names())],
                }
            ],
        }
