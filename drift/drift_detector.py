import logging

import numpy as np
import requests
from alibi_detect.cd import KSDrift

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SELDON_ENDPOINT = "http://localhost:8080/api/v0.1/predictions"


def run_drift_simulation():
    np.random.seed(42)

    logger.info("Generando distribución de referencia X_ref ~ U(0, 1)")
    X_ref = np.random.uniform(0, 1, size=(1000, 4))

    cd = KSDrift(X_ref, p_val=0.05, correction="bonferroni")

    X_healthy = np.random.uniform(0, 1, size=(50, 4))
    _evaluate_batch(cd, X_healthy, "Lote bueno")

    X_drifted = np.random.normal(loc=2.5, scale=0.5, size=(50, 4))
    _evaluate_batch(cd, X_drifted, "Lote con drift")


def _evaluate_batch(detector: KSDrift, batch: np.ndarray, label: str):
    logger.info(f"Evaluando: {label}")

    try:
        payload = {"data": {"ndarray": batch.tolist()}}
        resp = requests.post(SELDON_ENDPOINT, json=payload, timeout=2)
        logger.info(f"Respuesta del clúster Seldon: HTTP {resp.status_code}")
    except requests.exceptions.RequestException:
        logger.warning(
            "Clúster no alcanzable en localhost:8080 (ejecutando solo test estadístico local)"
        )

    preds = detector.predict(batch, drift_type="batch", return_p_val=True, return_distance=True)
    is_drift = bool(preds["data"]["is_drift"])
    p_vals = np.round(preds["data"]["p_val"], 5)
    distances = np.round(preds["data"]["distance"], 4)

    if is_drift:
        logger.error("[ALERTA MLOPS] ¡Data Drift detectado! Reentrenamiento requerido.")
    else:
        logger.info("[OK] Distribución dentro de los umbrales estadísticos.")

    logger.info(f"Estadístico K-S (Distancia máx): {distances} | p-valores: {p_vals}\n")


if __name__ == "__main__":
    run_drift_simulation()
