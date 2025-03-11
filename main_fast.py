import logging
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from data_extraction import load_data
from data_visualization import (
    plot_soc_over_time, plot_variables, plot_slope_vs_speed,
    plot_soh_over_time, plot_soh_distribution
)
from data_preprocessing import preprocess_data
from data_modelling2 import load_or_train_model, evaluate_and_forecast, visualize_results
from data_comparision import train_models, evaluate_model

# ✅ Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# 🚀 Initialize FastAPI
app = FastAPI(title="Battery Analytics API")

# ✅ Load Data
@app.get("/data/")
def get_data():
    try:
        df = load_data()
        df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

# 📈 Generate Plots
@app.get("/plot/{plot_type}/")
def get_plot(plot_type: str):
    df = load_data()
    os.makedirs("plots", exist_ok=True)  # Ensure plots directory exists

    plot_functions = {
        "soc_over_time": plot_soc_over_time,
        "variables": plot_variables,
        "slope_vs_speed": plot_slope_vs_speed,
        "soh_over_time": plot_soh_over_time,
        "soh_distribution": plot_soh_distribution
    }

    if plot_type not in plot_functions:
        raise HTTPException(status_code=400, detail=f"Invalid plot type: {plot_type}")

    try:
        plot_path = plot_functions[plot_type](df)

        # Debugging step: Ensure plot_path is valid
        logging.info(f"Generated plot at: {plot_path}")
        if not plot_path or not os.path.exists(plot_path):
            raise ValueError(f"Plot function did not return a valid file path for {plot_type}")

        return FileResponse(plot_path)
    except Exception as e:
        logging.error(f"Error generating plot: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating plot: {e}")
# 🔮 Prediction Endpoint
class PredictionInput(BaseModel):
    input_data: list

@app.post("/predict/")
def predict_soh(data: PredictionInput):
    try:
        input_data = np.array(data.input_data)
        X, y, feature_scaler, target_scaler = preprocess_data()
        sequence_length, num_features = X.shape[1], X.shape[2]
        
        if input_data.shape != (sequence_length, num_features):
            raise ValueError("Input shape mismatch")
        
        input_sequence = feature_scaler.transform(input_data)
        input_sequence = input_sequence.reshape(1, sequence_length, -1)
        
        model = load_or_train_model()
        y_pred_original, y_test_original, forecasted_soh = evaluate_and_forecast(
            model, X, y, target_scaler, weeks=4
        )
        
        return JSONResponse(content={"forecasted_soh": forecasted_soh.tolist()})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# 📊 Model Comparison Endpoint
@app.get("/compare_models/")
def compare_models():
    try:
        models, X_test, y_test, target_scaler = train_models()
        results = {}
        for name, model in models.items():
            X_test_reshaped = X_test.reshape(X_test.shape[0], 50, 35) if name == "LSTM" else X_test
            y_pred = model.predict(X_test_reshaped)
            results[name] = evaluate_model(y_test, y_pred)
        
        return JSONResponse(content=results)
    except Exception as e:
        logging.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error comparing models: {e}")
