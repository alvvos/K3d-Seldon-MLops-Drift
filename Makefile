CLUSTER_NAME := mlops-cluster
IMAGE_NAME := alvarosalis/mlops-model:v2
SELDON_VERSION := v1.17.1

.PHONY: help create-cluster delete-cluster install-seldon \
        install-monitoring apply-monitoring port-forward-grafana \
        train build-model load-image deploy-model status \
        lint test test-load check-drift clean

help:
	@echo "Comandos disponibles:"
	@echo "  make create-cluster        Levanta el clúster K3d con LoadBalancer en :8080"
	@echo "  make install-seldon        Instala el operador Seldon Core en el clúster"
	@echo "  make install-monitoring    Instala Prometheus + Grafana (kube-prometheus-stack)"
	@echo "  make apply-monitoring      Aplica ServiceMonitor, alertas y dashboard de Grafana"
	@echo "  make port-forward-grafana  Expone Grafana en http://localhost:3000"
	@echo "  make train                 Entrena el modelo y exporta artefactos + reference_data"
	@echo "  make deploy-model          Compila imagen, la inyecta en K3d y aplica k8s/ (Kustomize)"
	@echo "  make status                Muestra Pods, Services, Ingress, HPA y PDB activos"
	@echo "  make lint                  Formatea y corrige código con Ruff"
	@echo "  make test                  Ejecuta tests unitarios e integración con Pytest"
	@echo "  make test-load             Arranca Locust contra http://localhost:8080"
	@echo "  make check-drift           Ejecuta auditoría Kolmogorov-Smirnov + Bonferroni"
	@echo "  make delete-cluster        Destruye el clúster K3d"

# --- INFRAESTRUCTURA K3D / SELDON ---
create-cluster:
	@echo "Creando clúster K3d '$(CLUSTER_NAME)'..."
	k3d cluster create $(CLUSTER_NAME) --agents 1 -p "8080:80@loadbalancer"
	kubectl cluster-info

delete-cluster:
	@echo "Eliminando clúster K3d '$(CLUSTER_NAME)'..."
	k3d cluster delete $(CLUSTER_NAME)

install-seldon:
	@echo "Instalando Seldon Core Operator ($(SELDON_VERSION))..."
	kubectl create namespace seldon-system --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install seldon-core seldon-core-operator \
		--repo https://storage.googleapis.com/seldon-charts \
		--set usageMetrics.enabled=false \
		--set istio.enabled=false \
		--namespace seldon-system \
		--version $(SELDON_VERSION)
	@echo "Esperando a que el operador Seldon esté listo..."
	kubectl wait --for=condition=Ready pod -l app=seldon-operator -n seldon-system --timeout=120s

# --- MONITORIZACIÓN (PROMETHEUS + GRAFANA) ---
install-monitoring:
	@echo "Instalando kube-prometheus-stack en namespace 'monitoring'..."
	kubectl apply -f k8s/monitoring/namespace.yaml
	helm upgrade --install kube-prometheus-stack kube-prometheus-stack \
		--repo https://prometheus-community.github.io/helm-charts \
		--namespace monitoring \
		-f k8s/monitoring/prometheus-values.yaml
	$(MAKE) apply-monitoring

apply-monitoring:
	@echo "Aplicando ServiceMonitor, PrometheusRule y dashboard de Grafana..."
	kubectl apply -k k8s/monitoring

port-forward-grafana:
	@echo "Exponiendo Grafana en http://localhost:3000 (admin / mlops-local-password)..."
	kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80

# --- ENTRENAMIENTO Y DESPLIEGUE DEL MODELO ---
train:
	@echo "Entrenando pipeline de riesgo crediticio..."
	python model/src/train.py

build-model:
	@echo "Compilando imagen Docker del modelo ($(IMAGE_NAME))..."
	docker build -t $(IMAGE_NAME) ./model

load-image: build-model
	@echo "Inyectando imagen local en los nodos de K3d..."
	k3d image import $(IMAGE_NAME) -c $(CLUSTER_NAME)

deploy-model: load-image
	@echo "Desplegando recursos base, SeldonDeployment e Ingress con Kustomize..."
	kubectl apply -k k8s
	@echo "Esperando a que el Pod del modelo pase a Ready..."
	kubectl wait --for=condition=Ready pod -l seldon-deployment-id=mlops-model --timeout=120s

status:
	kubectl get pods,svc,ingress,hpa,pdb

# --- CALIDAD DE CÓDIGO Y TESTS ---
lint:
	@echo "Ejecutando Ruff..."
	ruff check --fix .
	ruff format .

test:
	@echo "Ejecutando Pytest..."
	PYTHONPATH=. pytest -v

# --- PRUEBAS DE CARGA Y GOBERNANZA ---
test-load:
	@echo "Abriendo Locust en http://localhost:8089 (objetivo http://localhost:8080)..."
	locust -f locust/locustfile.py --host http://localhost:8080

check-drift:
	@echo "Ejecutando detector de Data Drift sobre referencia real..."
	python drift/drift_detector.py --report-path drift/reports/latest_drift_report.json

clean: delete-cluster
