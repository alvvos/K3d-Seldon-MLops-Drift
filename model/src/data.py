from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .schema import FEATURE_NAMES, TARGET_COLUMN
except ImportError:
    from schema import FEATURE_NAMES, TARGET_COLUMN


def generate_credit_risk_dataset(
    n_samples: int = 5000,
    random_state: int = 42,
    missing_rate: float = 0.025,
) -> pd.DataFrame:
    """Genera un dataset sintético reproducible de riesgo crediticio.

    Las variables presentan correlaciones realistas entre sí (por ejemplo,
    mayor puntuación crediticia implica menor tipo de interés y mayor
    antigüedad laboral suele asociarse con mayores ingresos) y una frontera
    no lineal para la probabilidad de impago (`default`).
    """
    rng = np.random.default_rng(random_state)

    age = np.clip(rng.normal(loc=41.0, scale=11.5, size=n_samples), 21.0, 75.0)
    max_emp = np.maximum(age - 18.0, 0.5)
    employment_years = np.clip(
        rng.gamma(shape=2.6, scale=3.2, size=n_samples),
        0.0,
        max_emp,
    )

    # Ingresos con cola pesada (log-normal), correlacionados con experiencia laboral
    base_log_income = rng.normal(loc=10.85, scale=0.45, size=n_samples)
    annual_income = np.clip(
        np.exp(base_log_income) + employment_years * 1100.0,
        18000.0,
        320000.0,
    )

    # Puntuación crediticia (300 - 850) influida por estabilidad laboral e ingresos
    credit_score = np.clip(
        rng.normal(loc=665.0, scale=78.0, size=n_samples)
        + 1.8 * employment_years
        + 0.00025 * (annual_income - 60000.0),
        300.0,
        850.0,
    )

    # Importe del préstamo proporcional a ingresos con variabilidad
    loan_amount = np.clip(
        annual_income * rng.uniform(0.10, 0.55, size=n_samples)
        + rng.normal(0.0, 2500.0, size=n_samples),
        1500.0,
        100000.0,
    )

    # Tipo de interés inversamente relacionado con el credit_score
    interest_rate = np.clip(
        21.5 - 0.018 * credit_score + rng.normal(0.0, 1.8, size=n_samples),
        4.0,
        26.0,
    )

    # Debt-to-income (DTI) influido por relación préstamo/ingresos
    loan_to_income = loan_amount / np.maximum(annual_income, 1.0)
    debt_to_income = np.clip(
        0.14 + 0.42 * loan_to_income + rng.normal(0.0, 0.08, size=n_samples),
        0.03,
        0.78,
    )

    num_open_accounts = np.clip(
        np.round(rng.poisson(lam=6.5, size=n_samples) + 0.04 * (age - 30.0)),
        1.0,
        24.0,
    )

    # Log-odds no lineales con interacciones crediticias (tasa objetivo ~28%-32%)
    high_dti_low_score = ((debt_to_income > 0.34) & (credit_score < 670.0)).astype(float)
    young_high_loan = ((age < 32.0) & (loan_to_income > 0.30)).astype(float)
    short_tenure_risk = (employment_years < 3.5).astype(float) * np.maximum(
        interest_rate - 8.5, 0.0
    )

    logits = (
        -1.35
        + 3.6 * (debt_to_income - 0.28)
        + 8.0 * np.maximum(debt_to_income - 0.36, 0.0) ** 2
        + 0.12 * (interest_rate - 9.5)
        - 0.011 * (credit_score - 670.0)
        + 3.8 * np.maximum(loan_to_income - 0.30, 0.0)
        - 0.06 * np.minimum(employment_years, 15.0)
        + 2.20 * high_dti_low_score
        + 1.80 * young_high_loan
        + 0.45 * short_tenure_risk
        + 0.10 * np.maximum(num_open_accounts - 9.0, 0.0)
        + rng.normal(0.0, 0.30, size=n_samples)
    )
    default_prob = 1.0 / (1.0 + np.exp(-logits))
    default = (rng.uniform(0.0, 1.0, size=n_samples) < default_prob).astype(int)

    df = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "annual_income": np.round(annual_income, 2),
            "employment_years": np.round(employment_years, 1),
            "loan_amount": np.round(loan_amount, 2),
            "interest_rate": np.round(interest_rate, 2),
            "debt_to_income": np.round(debt_to_income, 4),
            "credit_score": np.round(credit_score, 1),
            "num_open_accounts": num_open_accounts.astype(float),
            TARGET_COLUMN: default,
        }
    )

    # Introducir un porcentaje controlado de nulos en columnas seleccionadas
    if missing_rate > 0.0:
        nullable_cols = ["annual_income", "employment_years", "debt_to_income", "credit_score"]
        for col in nullable_cols:
            mask = rng.uniform(0.0, 1.0, size=n_samples) < missing_rate
            df.loc[mask, col] = np.nan

    return df[FEATURE_NAMES + [TARGET_COLUMN]]


def load_or_create_dataset(
    csv_path: Path,
    n_samples: int = 5000,
    random_state: int = 42,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Carga el CSV de datos si existe; en caso contrario lo genera y persiste."""
    if csv_path.exists() and not force_regenerate:
        return pd.read_csv(csv_path)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_credit_risk_dataset(n_samples=n_samples, random_state=random_state)
    df.to_csv(csv_path, index=False)
    return df
