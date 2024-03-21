import torch
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mlflow

def evaluate_model(model, X_test, y_test):

  X_test = X_test.astype(np.float32)

  le = LabelEncoder()
  y_test = le.fit_transform(y_test)

  X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
  y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
  with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = (outputs > 0.5).float()
    accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)
    print(3*'\n')
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, predictions))

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)