from flask import Flask, request, jsonify
import torch
from src.preprocessing import Preprocessor
from model.model import FNN
import mlflow
import pandas as pd

app = Flask(__name__)

# Load the trained model using MLflow
def load_model():
    with mlflow.start_run() as run:
        # Load the model
        model_uri = 'runs:/9a3ee2317e30495f9068c5788875b47b/model/trained_models/attrition_predictor_model.pth'  # Update with your MLflow run ID
        model = mlflow.pytorch.load_model(model_uri)
    return model

# Preprocess data using the Preprocessor class
def preprocess_data(df):
    preprocessor = Preprocessor(df)
    X, _ = preprocessor.define_target_and_independent_features()
    X = preprocessor.remove_features_with_zero_variance(X)
    numerical, categorical = preprocessor.separate_feature_into_numerical_and_categorical(X)
    numerical = preprocessor.remove_outliers(numerical)
    numerical_scaled = preprocessor.scale_numerical_features(numerical)
    X_ca = preprocessor.select_k_best_categorical_features(categorical, None)
    X_all = preprocessor.join_all_features(numerical_scaled, X_ca)
    return X_all

# Load the model
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json

        # Preprocess the input data
        input_data = pd.DataFrame(data)
        processed_data = preprocess_data(input_data)

        # Convert the processed data to a tensor
        input_tensor = torch.tensor(processed_data.values, dtype=torch.float32)

        # Make predictions using the model
        with torch.no_grad():
            predictions = model(input_tensor)
            predictions = (predictions > 0.5).float().numpy().flatten().tolist()

        # Return the predictions
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
