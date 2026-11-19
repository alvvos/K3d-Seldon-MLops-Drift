from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from model.src.Model import Model
from model.src.schema import CLASS_NAMES, FEATURE_NAMES
from model.src.train import train_and_evaluate


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, Path]:
    base_dir = tmp_path_factory.mktemp("credit_model")
    data_dir = base_dir / "data"
    output_dir = base_dir / "artifacts"
    train_and_evaluate(
        data_dir=data_dir,
        output_dir=output_dir,
        n_samples=800,
        cv_folds=3,
        random_state=42,
        reference_sample_size=200,
    )
    return output_dir / "model.joblib", output_dir / "metadata.json"


def _build_model(trained_artifacts: Tuple[Path, Path]) -> Model:
    model_path, metadata_path = trained_artifacts
    return Model(model_path=model_path, metadata_path=metadata_path)


def test_model_initialization_and_seldon_hooks(trained_artifacts: Tuple[Path, Path]) -> None:
    model = _build_model(trained_artifacts)

    assert model.ready is True
    assert model.feature_names == FEATURE_NAMES
    assert model.class_names() == CLASS_NAMES

    health = model.health_status()
    assert health["status"] == "ok"

    tags = model.tags()
    assert "model_version" in tags
    assert "algorithm" in tags

    init_meta = model.init_metadata()
    assert init_meta["platform"] == "scikit-learn"
    assert init_meta["inputs"][0]["shape"] == [-1, len(FEATURE_NAMES)]


def test_model_predict_probabilities_and_risk_ordering(
    trained_artifacts: Tuple[Path, Path],
) -> None:
    model = _build_model(trained_artifacts)

    # Orden: age, annual_income, employment_years, loan_amount, interest_rate, dti, score, accts
    low_risk = [48.0, 130000.0, 15.0, 12000.0, 6.2, 0.14, 790.0, 4.0]
    high_risk = [24.0, 26000.0, 0.5, 32000.0, 21.5, 0.58, 460.0, 15.0]

    preds = model.predict([low_risk, high_risk])

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert len(preds[0]) == 2
    assert np.isclose(sum(preds[0]), 1.0, atol=1e-4)
    assert np.isclose(sum(preds[1]), 1.0, atol=1e-4)
    # El perfil de alto riesgo debe obtener una probabilidad de default claramente superior
    assert preds[1][1] > preds[0][1]

    metrics = {m["key"]: m["value"] for m in model.metrics()}
    assert metrics["credit_risk_predictions_total"] == 2
    assert metrics["credit_risk_inference_duration_ms"] >= 0.0


def test_model_predict_handles_1d_nans_and_shuffled_feature_names(
    trained_artifacts: Tuple[Path, Path],
) -> None:
    model = _build_model(trained_artifacts)
    sample = [38.0, 72000.0, 6.0, 18000.0, 10.2, 0.28, 680.0, 6.0]

    preds_1d = model.predict(sample)
    assert len(preds_1d) == 1

    # Reordenar columnas usando features_names debe producir exactamente la misma predicción
    reversed_names = list(reversed(FEATURE_NAMES))
    reversed_sample = list(reversed(sample))
    preds_reordered = model.predict([reversed_sample], features_names=reversed_names)
    assert np.allclose(preds_1d, preds_reordered, atol=1e-6)

    # Valores faltantes (NaN) deben ser imputados por el pipeline sin fallar
    sample_with_nan = [38.0, np.nan, 6.0, 18000.0, 10.2, np.nan, 680.0, 6.0]
    preds_nan = model.predict([sample_with_nan])
    assert len(preds_nan) == 1
    assert np.isclose(sum(preds_nan[0]), 1.0, atol=1e-4)


def test_model_predict_validation_errors(trained_artifacts: Tuple[Path, Path]) -> None:
    model = _build_model(trained_artifacts)

    with pytest.raises(ValueError, match="vacía"):
        model.predict([])

    with pytest.raises(ValueError, match="Número de features inválido"):
        model.predict([[1.0, 2.0, 3.0, 4.0]])

    with pytest.raises(ValueError, match="infinitos"):
        model.predict([[40.0, np.inf, 5.0, 10000.0, 9.0, 0.25, 700.0, 5.0]])

    with pytest.raises(ValueError, match="no coincide con el esquema"):
        model.predict(
            [[40.0, 60000.0, 5.0, 10000.0, 9.0, 0.25, 700.0, 5.0]],
            features_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
        )
