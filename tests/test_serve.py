import numpy as np

from model.src.Model import Model


def test_model_initialization() -> None:
    model = Model()
    assert model.ready is True


def test_model_predict_shape_and_values() -> None:
    model = Model()
    batch = [
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    preds = model.predict(batch)

    assert isinstance(preds, list)
    assert len(preds) == 2
    assert np.isclose(preds[0], 15.0)
    assert np.isclose(preds[1], 0.0)
