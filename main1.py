import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained LSTM model & scalers
lstm_model = load_model('lstm_model_weekly.keras')
feature_scaler = joblib.load('feature_scaler.pkl')
target_scaler = joblib.load('target_scaler.pkl')

# Function to preprocess user input
def preprocess_input(features, feature_scaler, sequence_length=50, num_features=35):
    features = np.array(features).reshape(sequence_length, num_features)
    scaled_features = feature_scaler.transform(features)
    return scaled_features.reshape(1, sequence_length, num_features)

# Streamlit UI
st.title("🔋 Battery SoH Prediction with LSTM")
st.write("Enter feature values for a 50x35 input sequence to predict the State of Health (SoH).")

# User input form
feature_input = []
for i in range(50):
    row = st.text_input(f"Row {i+1} (Comma-separated 35 values)", key=f"row_{i}")
    if row:
        feature_input.append([float(x) for x in row.split(",")])

# Predict button
if st.button("Predict SoH"):
    if len(feature_input) == 50 and all(len(row) == 35 for row in feature_input):
        try:
            input_sequence = preprocess_input(feature_input, feature_scaler)
            predicted_soh = lstm_model.predict(input_sequence)
            predicted_soh_original = target_scaler.inverse_transform(predicted_soh.reshape(-1, 1))[0, 0]
            
            st.success(f"🔋 Predicted SoH: **{predicted_soh_original:.2f}%**")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.warning("❗ Please enter valid feature values for all 50 rows.")


    

    
