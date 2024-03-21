run-app:
	python app.py

start-mlflow-server:
	mlflow server --host 0.0.0.0 --port 5000

run-all:
	@echo "Starting MLflow server..."
	@mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db &
	@echo "MLflow server started!"
	@echo "Starting Flask app..."
	@python app.py

.PHONY: run-app start-mlflow-server run-all
