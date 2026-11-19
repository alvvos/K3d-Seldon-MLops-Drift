from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "model" / "artifacts" / "metadata.json"

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

DEFAULT_FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (22.0, 68.0),
    "annual_income": (30000.0, 145000.0),
    "employment_years": (1.0, 22.0),
    "loan_amount": (4000.0, 50000.0),
    "interest_rate": (5.5, 14.5),
    "debt_to_income": (0.10, 0.45),
    "credit_score": (540.0, 815.0),
    "num_open_accounts": (2.0, 13.0),
}


def _load_schema_and_ranges() -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    if not METADATA_PATH.exists():
        return DEFAULT_FEATURE_NAMES, DEFAULT_FEATURE_RANGES

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = list(metadata.get("feature_names", DEFAULT_FEATURE_NAMES))
    stats = metadata.get("feature_stats", {})
    ranges: Dict[str, Tuple[float, float]] = {}

    for feat in feature_names:
        if feat in stats:
            low = float(stats[feat].get("p05", DEFAULT_FEATURE_RANGES[feat][0]))
            high = float(stats[feat].get("p95", DEFAULT_FEATURE_RANGES[feat][1]))
            ranges[feat] = (low, high)
        else:
            ranges[feat] = DEFAULT_FEATURE_RANGES[feat]

    return feature_names, ranges


FEATURE_NAMES, FEATURE_RANGES = _load_schema_and_ranges()


def _sample_applicant() -> List[float]:
    return [round(random.uniform(*FEATURE_RANGES[feat]), 4) for feat in FEATURE_NAMES]


def _sample_high_risk_applicant() -> List[float]:
    return [
        round(random.uniform(22.0, 30.0), 1),  # age
        round(random.uniform(22000.0, 36000.0), 2),  # annual_income
        round(random.uniform(0.2, 2.5), 1),  # employment_years
        round(random.uniform(20000.0, 42000.0), 2),  # loan_amount
        round(random.uniform(15.0, 23.0), 2),  # interest_rate
        round(random.uniform(0.42, 0.68), 4),  # debt_to_income
        round(random.uniform(430.0, 560.0), 1),  # credit_score
        float(random.randint(10, 18)),  # num_open_accounts
    ]


class MLOpsLoadTest(HttpUser):
    wait_time = between(0.5, 2.0)
    endpoint = "/api/v0.1/predictions"

    def _send_prediction(self, rows: List[List[float]], request_name: str) -> None:
        payload = {
            "data": {
                "names": FEATURE_NAMES,
                "ndarray": rows,
            }
        }
        headers = {"Content-Type": "application/json"}

        with self.client.post(
            self.endpoint,
            data=json.dumps(payload),
            headers=headers,
            catch_response=True,
            name=request_name,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:120]}")
                return

            try:
                body = response.json()
                preds = body["data"]["ndarray"]
                if len(preds) != len(rows) or len(preds[0]) != 2:
                    response.failure(f"Formato inesperado en respuesta: {preds[:1]}")
                else:
                    response.success()
            except (ValueError, KeyError, IndexError) as exc:
                response.failure(f"Error parseando respuesta JSON: {exc}")

    @task(7)
    def predict_single(self) -> None:
        self._send_prediction([_sample_applicant()], "Seldon inferencia (single)")

    @task(2)
    def predict_batch(self) -> None:
        batch = [_sample_applicant() for _ in range(16)]
        self._send_prediction(batch, "Seldon inferencia (batch x16)")

    @task(1)
    def predict_high_risk(self) -> None:
        self._send_prediction([_sample_high_risk_applicant()], "Seldon inferencia (high-risk)")
