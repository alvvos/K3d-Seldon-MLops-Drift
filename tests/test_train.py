from __future__ import annotations

import json
from pathlib import Path

from model.src.data import generate_credit_risk_dataset
from model.src.schema import FEATURE_NAMES, TARGET_COLUMN
from model.src.train import train_and_evaluate


def test_generate_credit_risk_dataset_schema_and_balance() -> None:
    df = generate_credit_risk_dataset(n_samples=600, random_state=42)

    assert len(df) == 600
    assert list(df.columns) == FEATURE_NAMES + [TARGET_COLUMN]
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}
    default_rate = float(df[TARGET_COLUMN].mean())
    assert 0.10 <= default_rate <= 0.50
    assert df[FEATURE_NAMES].isna().sum().sum() > 0


def test_train_and_evaluate_exports_artifacts_and_passes_quality_gate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "artifacts"

    pipeline, metadata = train_and_evaluate(
        data_dir=data_dir,
        output_dir=output_dir,
        n_samples=800,
        cv_folds=3,
        random_state=42,
        reference_sample_size=200,
    )

    assert pipeline is not None
    assert (output_dir / "model.joblib").exists()
    assert (output_dir / "metadata.json").exists()
    assert (data_dir / "credit_risk.csv").exists()
    assert (data_dir / "reference_data.csv").exists()

    loaded_meta = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert loaded_meta["feature_names"] == FEATURE_NAMES
    assert set(loaded_meta["cv_comparison"].keys()) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert loaded_meta["selected_algorithm"] in loaded_meta["cv_comparison"]
    assert loaded_meta["test_metrics"]["roc_auc"] >= 0.75
    assert metadata["test_metrics"]["roc_auc"] == loaded_meta["test_metrics"]["roc_auc"]
