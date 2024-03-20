import torch
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"Accuracy: {accuracy}")
        print(classification_report(y_test, predictions))

        #Print everything
