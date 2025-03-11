import logging
import os
import streamlit as st
import pandas as pd
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
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

# 🚀 **Streamlit App Title**
st.title("🔋 Battery Analytics and Prediction App")

# 📌 Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Plots", "Prediction", "Model Comparison"])

# ✅ **Load Data with Caching**
@st.cache_data
def get_data():
    try:
        df = load_data()
        df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc']).dt.strftime('%Y-%m-%d %H:%M:%S')
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        return None

# 🏠 **Home Page**
if page == "Home":
    st.write("Welcome to the **Battery Analytics and Prediction App** 🚀")
    st.write("Use the navigation menu to explore data, visualize plots, or make predictions.")

# 📊 **Data Exploration**
elif page == "Data Exploration":
    st.header("📊 Data Exploration")
    df = get_data()
    if df is not None:
        st.write("🔍 Here's a preview of the data:")
        st.dataframe(df)

# 📈 **Data Visualization**
elif page == "Plots":
    st.header("📈 Data Visualization")
    df = get_data()

    if df is not None:
        plot_type = st.selectbox("Select a plot type", [
            "SoC Over Time", "Variables", "Slope vs Speed", "SoH Over Time", "SoH Distribution"
        ])

        os.makedirs("plots", exist_ok=True)  # Ensure directory exists

        try:
            if plot_type == "SoC Over Time":
                plot_path = plot_soc_over_time(df)
            elif plot_type == "Variables":
                file_paths = plot_variables(df, save_path="plots")
                plot_path = file_paths[0] if file_paths else None
            elif plot_type == "Slope vs Speed":
                plot_path = plot_slope_vs_speed(df)
            elif plot_type == "SoH Over Time":
                plot_path = plot_soh_over_time(df)
            elif plot_type == "SoH Distribution":
                plot_path = plot_soh_distribution(df)
            
            if plot_path and os.path.exists(plot_path):
                st.image(plot_path, caption=plot_type, use_container_width=True)
            else:
                st.error(f"❌ Error: Plot function did not return a valid file path for {plot_type}.")
        except Exception as e:
            st.error(f"❌ Error generating {plot_type} plot: {e}")
            logging.error(f"Error generating {plot_type} plot: {e}")

# 🔮 **Prediction**
elif page == "Prediction":
    st.header("🔮 Battery SoH Prediction")
    st.write("Enter input data as a **JSON array** (must be 50 timestamps × 35 features).")

    input_data = st.text_area("📌 Enter JSON input data:")

    if st.button("Predict"):
        try:
            if not input_data:
                st.error("⚠️ Please provide valid input data.")
                raise ValueError("Empty input data.")

            # 🔍 **Parse JSON input safely**
            try:
                input_data = json.loads(input_data)
                if not isinstance(input_data, list):
                    raise ValueError("Input must be a list of lists (time-series format).")
            except json.JSONDecodeError as e:
                st.error(f"❌ Invalid JSON format: {e}")
                logging.error(f"Invalid JSON input: {e}")
                raise

            # ✅ **Load Preprocessed Data**
            X, y, feature_scaler, target_scaler = preprocess_data()
            sequence_length = 50
            num_features = X.shape[2]  

            input_sequence = np.array(input_data)
            if input_sequence.shape[1] != num_features: 
                st.error(f"❌ Feature mismatch: Expected {num_features}, got {input_sequence.shape[1]}")
                raise ValueError("Input feature count does not match training data.")

            if input_sequence.shape[0] != sequence_length: 
                st.error(f"❌ Sequence length mismatch: Expected {sequence_length}, got {input_sequence.shape[0]}")
                raise ValueError("Input sequence length does not match training data.")

            input_sequence = feature_scaler.transform(input_sequence)
            input_sequence = input_sequence.reshape(1, sequence_length, -1)  # ✅ Reshape for LSTM model

            # 🔍 **Load or Train Model**
            model = load_or_train_model()
            y_pred_original, y_test_original, forecasted_soh = evaluate_and_forecast(
                model, X, y, target_scaler, weeks=4
            )

            # ✅ **Display Prediction Results**
            st.success("✅ Prediction successful!")
            st.write("📌 **Predicted SoH Values for Next 4 Weeks:**", forecasted_soh)

            # 📊 **Plot Actual vs Predicted SoH**
            st.subheader("📊 Actual vs Predicted SoH")
            visualize_results(y_pred_original, y_test_original, forecasted_soh, weeks=4, st=st)

        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")
            logging.error(f"Error making prediction: {e}")

# 📊 **Model Comparison**
elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")

    with st.spinner("🔄 Training models... Please wait ⏳"):
        models, X_test, y_test, target_scaler = train_models()
        st.success("✅ Models trained successfully!")

    results = {}
    for name, model in models.items():
        if name == "LSTM":
            X_test_reshaped = X_test.reshape(X_test.shape[0], 50, 35)  # ✅ Reshape for LSTM
        else:
            X_test_reshaped = X_test  # ML models use 2D data

        y_pred = model.predict(X_test_reshaped)
        results[name] = evaluate_model(y_test, y_pred)

    comparison_df = pd.DataFrame(results, index=["MAE", "MSE", "R²"]).T

    # ✅ **Display Model Metrics**
    st.write("### 📋 Model Evaluation Metrics")
    st.dataframe(comparison_df)

    # 📊 **Plot Model Comparison**
    st.subheader("📊 Model Performance Metrics")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["MAE", "MSE", "R²"]
    colors = ["blue", "orange", "green"]

    for i, metric in enumerate(metrics):
        ax[i].bar(comparison_df.index, comparison_df[metric], color=colors[i])
        ax[i].set_xlabel("Model")
        ax[i].set_ylabel(metric)
        ax[i].set_title(f"Comparison of Models - {metric}")

    plt.tight_layout()
    st.pyplot(fig)
