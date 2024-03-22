# Define variables
PYTHON = python
PIP = pip
MLFLOW = mlflow
FLASK = flask
FLAKE8 = flake8

# Target to install dependencies
install:
	$(PIP) install -r requirements.txt

# Target to run tests
test:
	$(PYTHON) -m pytest tests/

# Target to start MLflow server
mlflow:
	$(MLFLOW) ui

# Target to run the Flask app
run:
	$(PYTHON) app.py

# Target to lint the code
lint:
	$(FLAKE8) . --select E --ignore=E402,E501 .

# Target to run all tasks
all: install test mlflow run lint
