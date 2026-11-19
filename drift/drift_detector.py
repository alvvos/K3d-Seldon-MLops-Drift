from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from alibi_detect.cd import KSDrift

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "model" / "data" / "reference_data.csv"
SELDON_ENDPOINT = "http://localhost:8080/api/v0.1/predictions"

DEFAULT_FEATURE_NAMES: List[str] = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "interest_rate",
    "debt_to_income",
    "credit_score",
    "num_open_accounts",
]


def load_reference_data(
    reference_path: Path = DEFAULT_REFERENCE_PATH,
) -> Tuple[np.ndarray, List[str]]:
    """Carga la distribución de referencia X_ref generada durante el entrenamiento."""
    path = Path(reference_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de referencia en '{path}'. "
            "Ejecuta 'make train' antes de lanzar la auditoría de drift."
        )

    df = pd.read_csv(path)
    feature_names = list(df.columns)
    X_ref = df.to_numpy(dtype=np.float32)
    return X_ref, feature_names


def build_detector(
    X_ref: np.ndarray,
    p_val: float = 0.05,
    correction: str = "bonferroni",
) -> KSDrift:
    """Inicializa el detector Kolmogorov-Smirnov multivariante con corrección Bonferroni."""
    return KSDrift(X_ref, p_val=p_val, correction=correction)


def sample_healthy_batch(
    X_ref: np.ndarray,
    batch_size: int = 120,
    random_state: int = 101,
) -> np.ndarray:
    """Genera un lote en distribución (sin drift) muestreando de la referencia."""
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_ref), size=batch_size, replace=True)
    return X_ref[idx].astype(np.float32).copy()


def simulate_macroeconomic_drift(
    X_ref: np.ndarray,
    feature_names: List[str],
    batch_size: int = 120,
    random_state: int = 202,
) -> np.ndarray:
    """Simula un escenario de recesión (menores ingresos, mayor tipo de interés y DTI)."""
    batch = sample_healthy_batch(X_ref, batch_size=batch_size, random_state=random_state).copy()
    col_idx = {name: i for i, name in enumerate(feature_names)}

    if "annual_income" in col_idx:
        batch[:, col_idx["annual_income"]] *= 0.72
    if "interest_rate" in col_idx:
        batch[:, col_idx["interest_rate"]] += 4.8
    if "debt_to_income" in col_idx:
        batch[:, col_idx["debt_to_income"]] = np.clip(
            batch[:, col_idx["debt_to_income"]] + 0.14, 0.03, 0.85
        )
    return batch


def simulate_credit_score_drift(
    X_ref: np.ndarray,
    feature_names: List[str],
    batch_size: int = 120,
    random_state: int = 303,
) -> np.ndarray:
    """Simula un desplazamiento focalizado en la distribución de credit_score."""
    batch = sample_healthy_batch(X_ref, batch_size=batch_size, random_state=random_state).copy()
    col_idx = {name: i for i, name in enumerate(feature_names)}

    if "credit_score" in col_idx:
        batch[:, col_idx["credit_score"]] = np.clip(
            batch[:, col_idx["credit_score"]] - 110.0, 300.0, 850.0
        )
    return batch


def _query_seldon(
    batch: np.ndarray,
    feature_names: List[str],
    endpoint: str,
) -> Optional[Dict[str, Any]]:
    payload = {
        "data": {
            "names": feature_names,
            "ndarray": np.round(batch, 4).tolist(),
        }
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=3)
        logger.info("Respuesta del clúster Seldon: HTTP %d", resp.status_code)
        if resp.status_code == 200:
            body = resp.json()
            ndarray = body.get("data", {}).get("ndarray")
            if ndarray:
                probas = np.asarray(ndarray, dtype=float)
                if probas.ndim == 2 and probas.shape[1] >= 2:
                    default_probs = probas[:, 1]
                else:
                    default_probs = probas
                return {
                    "status_code": resp.status_code,
                    "avg_default_probability": round(float(np.mean(default_probs)), 4),
                    "high_risk_ratio": round(float(np.mean(default_probs >= 0.5)), 4),
                }
        return {"status_code": resp.status_code}
    except requests.exceptions.RequestException:
        logger.warning(
            "Clúster no alcanzable en %s (ejecutando solo test estadístico local)",
            endpoint,
        )
        return None


def evaluate_batch(
    detector: KSDrift,
    batch: np.ndarray,
    feature_names: List[str],
    label: str,
    endpoint: Optional[str] = SELDON_ENDPOINT,
    p_val: float = 0.05,
) -> Dict[str, Any]:
    """Evalúa un lote frente a la referencia y devuelve métricas detalladas por feature."""
    logger.info("=== Evaluando: %s (n=%d) ===", label, len(batch))

    seldon_summary = _query_seldon(batch, feature_names, endpoint) if endpoint is not None else None

    preds = detector.predict(batch, drift_type="batch", return_p_val=True, return_distance=True)
    is_drift = bool(preds["data"]["is_drift"])
    p_vals = np.asarray(preds["data"]["p_val"], dtype=float)
    distances = np.asarray(preds["data"]["distance"], dtype=float)
    bonferroni_threshold = p_val / max(len(feature_names), 1)

    feature_breakdown: Dict[str, Dict[str, Any]] = {}
    drifted_features: List[str] = []
    for name, dist, pv in zip(feature_names, distances.tolist(), p_vals.tolist()):
        feat_drifted = pv < bonferroni_threshold
        if feat_drifted:
            drifted_features.append(name)
        feature_breakdown[name] = {
            "ks_distance": round(dist, 4),
            "p_value": round(pv, 6),
            "drifted": feat_drifted,
        }

    if is_drift:
        logger.error(
            "[ALERTA MLOPS] ¡Data Drift detectado! Features afectadas: %s. Reentrenamiento.",
            drifted_features,
        )
    else:
        logger.info("[OK] Distribución dentro de los umbrales estadísticos (alpha=%.3f).", p_val)

    for name, stats in feature_breakdown.items():
        flag = "DRIFT" if stats["drifted"] else "OK"
        logger.info(
            "  %-18s | KS=%.4f | p=%.6f | [%s]",
            name,
            stats["ks_distance"],
            stats["p_value"],
            flag,
        )

    if seldon_summary and "avg_default_probability" in seldon_summary:
        logger.info(
            "  Predicción Seldon  | avg_p_default=%.4f | high_risk_ratio=%.2f%%",
            seldon_summary["avg_default_probability"],
            seldon_summary["high_risk_ratio"] * 100.0,
        )

    logger.info("")
    return {
        "scenario": label,
        "batch_size": int(len(batch)),
        "is_drift": is_drift,
        "bonferroni_threshold": round(bonferroni_threshold, 6),
        "drifted_features": drifted_features,
        "features": feature_breakdown,
        "seldon_inference": seldon_summary,
    }


def run_drift_simulation(
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    endpoint: Optional[str] = SELDON_ENDPOINT,
    p_val: float = 0.05,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Ejecuta la auditoría completa sobre escenarios sano y con drift realista."""
    X_ref, feature_names = load_reference_data(reference_path)
    logger.info(
        "Distribución de referencia cargada desde %s | shape=%s | features=%s",
        reference_path,
        X_ref.shape,
        feature_names,
    )

    detector = build_detector(X_ref, p_val=p_val, correction="bonferroni")

    scenarios = [
        ("Lote sano (in-distribution)", sample_healthy_batch(X_ref)),
        (
            "Lote con drift macroeconómico (recesión)",
            simulate_macroeconomic_drift(X_ref, feature_names),
        ),
        (
            "Lote con drift focalizado (credit_score)",
            simulate_credit_score_drift(X_ref, feature_names),
        ),
    ]

    results = [
        evaluate_batch(
            detector=detector,
            batch=batch,
            feature_names=feature_names,
            label=label,
            endpoint=endpoint,
            p_val=p_val,
        )
        for label, batch in scenarios
    ]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_path": str(reference_path),
        "reference_samples": int(len(X_ref)),
        "p_val": p_val,
        "correction": "bonferroni",
        "scenarios": results,
    }

    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Reporte de auditoría guardado en %s", report_path)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoría de Data Drift con KS + Bonferroni.")
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--endpoint", type=str, default=SELDON_ENDPOINT)
    parser.add_argument("--p-val", type=float, default=0.05)
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_drift_simulation(
        reference_path=args.reference_path,
        endpoint=args.endpoint,
        p_val=args.p_val,
        report_path=args.report_path,
    )
