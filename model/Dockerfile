FROM python:3.10-slim

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY src/ /app/


ENV MODEL_NAME=Model
ENV API_TYPE=REST
ENV SERVICE_TYPE=MODEL
ENV PERSISTENCE=0

EXPOSE 5000 9000

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE