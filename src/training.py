import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert X_train to a compatible data type
    X_train = X_train.astype(np.float32)
    
    # Encode target labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    # Reshape y_train_encoded to a 2D array
    y_train_encoded = y_train_encoded.reshape(-1, 1)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

def save_model(model, path):
    torch.save(model.state_dict(), path)
