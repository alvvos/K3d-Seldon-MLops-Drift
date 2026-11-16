CLUSTER_NAME := mlops-cluster
IMAGE_NAME := alvarosalis/mlops-model:v1
SELDON_VERSION := v1.17.1

.PHONY: help create-cluster delete-cluster install-seldon \
        build-model load-image deploy-model status \
        lint test test-load check-drift clean

help:
	@echo "Comandos disponibles:"
	@echo "  make create-cluster   Levanta el clúster K3d con LoadBalancer en :8080"
	@echo "  make install-seldon   Instala el operador Seldon Core en el clúster"
	@echo "  make deploy-model     Compila imagen, la inyecta en K3d y aplica YAML + Ingress"
	@echo "  make status           Muestra Pods, Services e Ingress activos"
	@echo "  make lint             Formatea y corrige código con Ruff"
	@echo "  make test             Ejecuta tests unitarios con Pytest"
	@echo "  make test-load        Arranca Locust contra http://localhost:8080"
	@echo "  make check-drift      Ejecuta auditoría Kolmogorov-Smirnov + Bonferroni"
	@echo "  make delete-cluster   Destruye el clúster K3d"

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
	helm install seldon-core seldon-core-operator \
		--repo https://storage.googleapis.com/seldon-charts \
		--set usageMetrics.enabled=false \
		--set istio.enabled=false \
		--namespace seldon-system \
		--version $(SELDON_VERSION)
	@echo "Esperando a que el operador Seldon esté listo..."
	kubectl wait --for=condition=Ready pod -l app=seldon-operator -n seldon-system --timeout=120s

# --- DESPLIEGUE DEL MODELO ---
build-model:
	@echo "Compilando imagen Docker del modelo..."
	docker build -t $(IMAGE_NAME) ./model

load-image: build-model
	@echo "Inyectando imagen local en los nodos de K3d..."
	k3d image import $(IMAGE_NAME) -c $(CLUSTER_NAME)

deploy-model: load-image
	@echo "Desplegando SeldonDeployment e Ingress..."
	kubectl apply -f k8s/seldon/seldon-deployment.yaml
	kubectl apply -f k8s/seldon/ingress.yaml
	@echo "Esperando a que el Pod del modelo pase a Ready..."
	kubectl wait --for=condition=Ready pod -l seldon-deployment-id=mlops-model --timeout=120s

status:
	kubectl get pods,svc,ingress,hpa

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
	@echo "Ejecutando detector de Data Drift..."
	python drift/drift_detector.py

clean: delete-cluster