import numpy as np
from alibi_detect.cd import KSDrift


def test_ks_drift_bonferroni_behavior() -> None:
    rng = np.random.default_rng(42)
    x_ref = rng.uniform(0.0, 1.0, size=(1000, 4))
    detector = KSDrift(x_ref, p_val=0.05, correction="bonferroni")

    x_healthy = rng.uniform(0.0, 1.0, size=(50, 4))
    res_healthy = detector.predict(x_healthy, drift_type="batch")
    assert res_healthy["data"]["is_drift"] == 0

    x_drifted = rng.normal(loc=2.5, scale=0.5, size=(50, 4))
    res_drifted = detector.predict(x_drifted, drift_type="batch")
    assert res_drifted["data"]["is_drift"] == 1
