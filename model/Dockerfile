FROM python:3.10-slim

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ /app/

# Entrena y empaqueta los artefactos con la versión exacta de Python/scikit-learn del contenedor
RUN python train.py --data-dir /app/data --output-dir /app/artifacts

RUN useradd --create-home --uid 8888 appuser && \
    chown -R appuser:appuser /app
USER 8888

ENV MODEL_NAME=Model
ENV API_TYPE=REST
ENV SERVICE_TYPE=MODEL
ENV PERSISTENCE=0
ENV MODEL_PATH=/app/artifacts/model.joblib
ENV METADATA_PATH=/app/artifacts/metadata.json

EXPOSE 5000 9000

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
