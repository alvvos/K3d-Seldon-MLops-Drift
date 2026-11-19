from __future__ import annotations

from typing import Dict, List

MODEL_VERSION = "2.0.0"
TARGET_COLUMN = "default"
CLASS_NAMES: List[str] = ["approved", "default"]
DEFAULT_DECISION_THRESHOLD = 0.50

MONETARY_FEATURES: List[str] = [
    "annual_income",
    "loan_amount",
]

RATE_AND_SCORE_FEATURES: List[str] = [
    "interest_rate",
    "debt_to_income",
    "credit_score",
]

PROFILE_FEATURES: List[str] = [
    "age",
    "employment_years",
    "num_open_accounts",
]

FEATURE_NAMES: List[str] = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "interest_rate",
    "debt_to_income",
    "credit_score",
    "num_open_accounts",
]

FEATURE_SPECS: Dict[str, Dict[str, object]] = {
    "age": {
        "description": "Edad del solicitante en años",
        "min": 21.0,
        "max": 75.0,
        "unit": "years",
    },
    "annual_income": {
        "description": "Ingresos brutos anuales del solicitante",
        "min": 15000.0,
        "max": 350000.0,
        "unit": "USD",
    },
    "employment_years": {
        "description": "Antigüedad laboral continua",
        "min": 0.0,
        "max": 45.0,
        "unit": "years",
    },
    "loan_amount": {
        "description": "Importe solicitado del préstamo",
        "min": 1000.0,
        "max": 120000.0,
        "unit": "USD",
    },
    "interest_rate": {
        "description": "Tipo de interés anual aplicado al préstamo",
        "min": 3.5,
        "max": 28.0,
        "unit": "percent",
    },
    "debt_to_income": {
        "description": "Ratio deuda mensual sobre ingresos mensuales (DTI)",
        "min": 0.02,
        "max": 0.85,
        "unit": "ratio",
    },
    "credit_score": {
        "description": "Puntuación crediticia estilo FICO",
        "min": 300.0,
        "max": 850.0,
        "unit": "score",
    },
    "num_open_accounts": {
        "description": "Número de líneas de crédito abiertas actualmente",
        "min": 1.0,
        "max": 25.0,
        "unit": "count",
    },
}
