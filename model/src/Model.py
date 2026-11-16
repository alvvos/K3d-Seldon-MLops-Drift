import logging

import numpy as np

logger = logging.getLogger(__name__)


class Model:
    def __init__(self):
        logger.info("Modelo listo")
        self.ready = True

    def predict(self, X, features_names=None, meta=None):
        logger.info(f"Petición de inferencia recibida; tensor shape: {np.shape(X)}")
        X_array = np.array(X)
        predictions = np.sum(X_array * 1.5, axis=1)

        return predictions.tolist()
