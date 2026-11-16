import json
import random

from locust import HttpUser, between, task


class MLOpsLoadTest(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_endpoint(self):
        features = [[random.uniform(0, 1) for _ in range(4)]]
        payload = {"data": {"ndarray": features}}

        endpoint = "/api/v0.1/predictions"

        headers = {"Content-Type": "application/json"}

        with self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            catch_response=True,
            name="Seldon inferencia",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Fallo en inferencia: Status {response.status_code}")
