import logging
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from data_extraction import load_data
from data_visualization import (
    plot_soc_over_time, plot_variables, plot_slope_vs_speed,
    plot_soh_over_time, plot_soh_distribution
)
from data_preprocessing import preprocess_data
from data_modelling2 import load_or_train_model, evaluate_and_forecast
from data_comparision import train_models, evaluate_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# 🚀 Initialize FastAPI
app = FastAPI(title="Battery Analytics API")

# ✅ CORS Configuration (Allow React Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"]
)

# ✅ Load Data Endpoint
@app.get("/data/")
def get_data():
    try:
        df = load_data()
        df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

# 📈 **Dropdown for Plot Selection**
@app.get("/plots/")
def get_plot_options():
    return {"available_plots": ["soc_over_time", "variables", "slope_vs_speed", "soh_over_time", "soh_distribution"]}

# 📊 **Generate Plots Endpoint**
@app.get("/plot/{plot_type}/")
def get_plot(plot_type: str):
    try:
        df = load_data()
        os.makedirs("plots", exist_ok=True)

        plot_functions = {
            "soc_over_time": plot_soc_over_time,
            "variables": plot_variables,
            "slope_vs_speed": plot_slope_vs_speed,
            "soh_over_time": plot_soh_over_time,
            "soh_distribution": plot_soh_distribution
        }

        if plot_type not in plot_functions:
            raise HTTPException(status_code=400, detail=f"Invalid plot type: {plot_type}")

        # ✅ Generate plot
        plot_path = plot_functions[plot_type](df)

        # Debugging step: Ensure plot_path is valid
        logging.info(f"Generated plot at: {plot_path}")
        if not plot_path or not os.path.exists(plot_path):
            raise ValueError(f"Plot function did not return a valid file path for {plot_type}")

        return FileResponse(plot_path)

    except Exception as e:
        logging.error(f"Error generating plot: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating plot: {e}")

# 🔮 **Prediction Endpoint**
@app.get("/predict/")
def predict_soh():
    try:
        df = load_data()
        X, y, feature_scaler, target_scaler = preprocess_data()

        # Ensure we have enough data points
        sequence_length = 50
        if X.shape[0] < sequence_length:
            raise HTTPException(status_code=400, detail=f"Not enough data points! Need at least {sequence_length}, but got {X.shape[0]}.")

        # Extract last 50 timesteps
        input_sequence = X[-sequence_length:].reshape(1, sequence_length, -1)

        # Load model
        model = load_or_train_model()

        # Predict and forecast
        y_pred_original, y_test_original, forecasted_soh = evaluate_and_forecast(
            model, X, y, target_scaler, weeks=4
        )

        return JSONResponse(content={"forecasted_soh": list(map(float, forecasted_soh))})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# 📊 **Model Comparison Endpoint (With MAE, MSE, R²)**
@app.get("/compare_models/")
def compare_models():
    try:
        models, X_test, y_test, target_scaler = train_models()
        results = {}

        for name, model in models.items():
            X_test_reshaped = X_test.reshape(X_test.shape[0], 50, 35) if name == "LSTM" else X_test
            
            # Predict
            y_pred = model.predict(X_test_reshaped)

            # Convert back to original scale
            y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
            y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

            # Calculate Metrics
            mae = mean_absolute_error(y_test_original, y_pred_original)
            mse = mean_squared_error(y_test_original, y_pred_original)
            r2 = r2_score(y_test_original, y_pred_original)

            # Store results
            results[name] = {
                "MAE": round(mae, 5),
                "MSE": round(mse, 5),
                "R²": round(r2, 5)
            }

        return JSONResponse(content=results)
    
    except Exception as e:
        logging.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error comparing models: {e}")
