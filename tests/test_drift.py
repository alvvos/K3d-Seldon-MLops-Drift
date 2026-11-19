from __future__ import annotations

from pathlib import Path

from drift.drift_detector import (
    build_detector,
    evaluate_batch,
    load_reference_data,
    sample_healthy_batch,
    simulate_credit_score_drift,
    simulate_macroeconomic_drift,
)
from model.src.data import generate_credit_risk_dataset
from model.src.schema import FEATURE_NAMES


def test_drift_detector_healthy_and_drifted_credit_scenarios(tmp_path: Path) -> None:
    df = generate_credit_risk_dataset(n_samples=1000, random_state=42, missing_rate=0.0)
    ref_path = tmp_path / "reference_data.csv"
    df[FEATURE_NAMES].to_csv(ref_path, index=False)

    X_ref, feature_names = load_reference_data(ref_path)
    assert X_ref.shape == (1000, len(FEATURE_NAMES))
    assert feature_names == FEATURE_NAMES

    detector = build_detector(X_ref, p_val=0.05, correction="bonferroni")

    healthy = sample_healthy_batch(X_ref, batch_size=120, random_state=101)
    res_healthy = evaluate_batch(
        detector=detector,
        batch=healthy,
        feature_names=feature_names,
        label="healthy",
        endpoint=None,
    )
    assert res_healthy["is_drift"] is False
    assert res_healthy["drifted_features"] == []

    macro_drifted = simulate_macroeconomic_drift(
        X_ref, feature_names=feature_names, batch_size=120, random_state=202
    )
    res_macro = evaluate_batch(
        detector=detector,
        batch=macro_drifted,
        feature_names=feature_names,
        label="macro_drift",
        endpoint=None,
    )
    assert res_macro["is_drift"] is True
    assert {"annual_income", "interest_rate", "debt_to_income"}.issubset(
        set(res_macro["drifted_features"])
    )

    score_drifted = simulate_credit_score_drift(
        X_ref, feature_names=feature_names, batch_size=120, random_state=303
    )
    res_score = evaluate_batch(
        detector=detector,
        batch=score_drifted,
        feature_names=feature_names,
        label="credit_score_drift",
        endpoint=None,
    )
    assert res_score["is_drift"] is True
    assert "credit_score" in res_score["drifted_features"]
