from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from fastapi.responses import JSONResponse, FileResponse
import logging
import os
import pandas as pd
from data_extraction import load_data
from data_visualization import (
    plot_soc_over_time, plot_variables, plot_slope_vs_speed,
    plot_soh_over_time, plot_soh_distribution
)
from data_preprocessing import preprocess_data
from data_modelling2 import load_or_train_model, evaluate_and_forecast
from data_comparision import train_models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

# 🚀 Initialize FastAPI
app = FastAPI(title="Battery Analytics API")

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (React Frontend URL should be set in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
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

# ✅ API Endpoints (Prediction, Visualization, Model Comparison)
@app.get("/predict/")
def predict_soh():
    try:
        df = load_data()
        X, y, feature_scaler, target_scaler = preprocess_data()

        sequence_length = 50
        if X.shape[0] < sequence_length:
            raise HTTPException(status_code=400, detail=f"Not enough data points! Need at least {sequence_length}, but got {X.shape[0]}.")

        input_sequence = X[-sequence_length:].reshape(1, sequence_length, -1)  

        model = load_or_train_model()
        y_pred_original, y_test_original, forecasted_soh = evaluate_and_forecast(model, X, y, target_scaler, weeks=4)

        return JSONResponse(content={"forecasted_soh": list(map(float, forecasted_soh))})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

