from src.preprocessing import Preprocessor
from model.model import FNN
from src.training import train_model, save_model
from src.evaluating import evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('data/HR-Employee-Attrition.csv')

# Preprocessing
preprocessor = Preprocessor(df)
X, Y = preprocessor.define_target_and_independent_features()
X = preprocessor.remove_features_with_zero_variance(X)
numerical, categorical = preprocessor.separate_feature_into_numerical_and_categorical(X)
numerical = preprocessor.remove_outliers(numerical)
numerical_scaled = preprocessor.scale_numerical_features(numerical)
X_ca = preprocessor.select_k_best_categorical_features(categorical, Y)
X_all = preprocessor.join_all_features(numerical_scaled, X_ca)
X_s, Y_s = preprocessor.oversample_data(X_all, Y)
X_train, x_test, y_train, y_test = preprocessor.split_data(X_s, Y_s)

# Model building
input_size = X_train.shape[1]
model = FNN(input_size)

# Model training
train_model(model, X_train, y_train)

# Save the trained model
save_model(model, 'attrition_predictor_model.pth')

# Model evaluation
evaluate_model(model, x_test, y_test)
