import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preprocessing import preprocess_data
from data_modelling2 import load_or_train_model as load_lstm_model
from data_modelling3 import preprocess_data as preprocess_data3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to train models
def train_models():
    """ Train LSTM and traditional ML models on SOH data. """
    X, y, feature_scaler, target_scaler = preprocess_data3()
    
    # Flatten for ML models
    X_reshaped = X.reshape(X.shape[0], -1)  # Reshape to 2D

    # Split into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # Load LSTM model
    lstm_model = load_lstm_model()
    
    # Train ML models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Reshape for LSTM
    X_test_lstm = X_test.reshape(X_test.shape[0], 50, 35)

    return {
        "LSTM": lstm_model,
        "Random Forest": rf_model,
        "Gradient Boosting": gb_model,
        "Linear Regression": lr_model
    }, X_test, y_test, target_scaler

# Function to evaluate models
def evaluate_model(y_true, y_pred):
    """ Compute MAE, MSE, and R² scores. """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2
