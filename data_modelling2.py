import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from data_preprocessing import preprocess_data

# Preprocess data
X, y, feature_scaler, target_scaler = preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train or load the model
def load_or_train_model():
    try:
        model = load_model('lstm_model_weekly.keras')
        print("Model loaded successfully.")
    except:
        print("No saved model found. Training a new model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_lstm_model(input_shape)
        model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)
        model.save('lstm_model_weekly.keras')
        joblib.dump(feature_scaler, 'feature_scaler.pkl')
        joblib.dump(target_scaler, 'target_scaler.pkl')
    return model

# Evaluate and forecast
def evaluate_and_forecast(model, X_test, y_test, target_scaler, weeks=4):
    # Evaluate the model on the test data
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss:", test_loss)
    print("Test MAE:", test_mae)

    # Predict on the test data
    y_pred = model.predict(X_test)
    y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Forecast for future weeks
    forecasted_soh = []
    input_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])  # Start with the last sequence

    for _ in range(weeks):
        # Predict the next SOH value
        predicted_soh = model.predict(input_sequence)
        forecasted_soh.append(target_scaler.inverse_transform(predicted_soh.reshape(-1, 1))[0, 0])

        # Update the input sequence for the next prediction
        predicted_soh_feature = np.zeros((1, input_sequence.shape[2]))
        predicted_soh_feature[0, -1] = predicted_soh  # Replace the last feature (SOH) with the predicted value

        # Shift the sequence and append the new prediction
        next_input = np.append(input_sequence[0, 1:], predicted_soh_feature, axis=0)
        input_sequence = next_input.reshape(1, next_input.shape[0], next_input.shape[1])

    return y_pred_original, y_test_original, forecasted_soh

# Visualization - modified for Streamlit integration
# Visualization - modified for Streamlit integration
# Visualization - modified for Streamlit integration
def visualize_results(y_pred_original, y_test_original, forecasted_soh, weeks, st):
    # Generate weekly time-based x-axis
    week_labels = np.arange(1, len(y_test_original) + 1)  # Weeks for actual vs predicted plot
    forecast_weeks = np.arange(1, weeks + 1)  # Ensures values are 1, 2, 3, 4

    # Create a new figure with a wider layout
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Increased width for clarity

    # Plot actual vs predicted SOH with better x-axis spacing
    axs[0].plot(week_labels, y_test_original, label="Actual SOH", color='blue', marker='o', markersize=4)
    axs[0].plot(week_labels, y_pred_original, label="Predicted SOH", color='orange', marker='s', markersize=4)
    axs[0].set_xticks(week_labels[::3])  # Show every 3rd week to prevent label merging
    axs[0].tick_params(axis='x', rotation=45)  # Rotate x-axis labels slightly for better readability
    axs[0].set_xlabel("Weeks")
    axs[0].set_ylabel("SOH")
    axs[0].set_title("Actual vs Predicted SOH Over Time")
    axs[0].legend()

    # Plot forecasted SOH with correct week labels
    axs[1].plot(forecast_weeks, forecasted_soh, label="Forecasted SOH", marker='o', color='green', markersize=6)
    axs[1].set_xticks(forecast_weeks)  # Ensure x-axis shows 1, 2, 3, 4 without merging
    axs[1].tick_params(axis='x', rotation=0)  # No rotation needed here
    axs[1].set_xlabel("Weeks Ahead")
    axs[1].set_ylabel("SOH")
    axs[1].set_title(f"Forecasted SOH for Next {weeks} Weeks")
    axs[1].legend()

    # Layout adjustment and display with Streamlit
    plt.tight_layout()
    st.pyplot(fig)  # Use st.pyplot to display the figure in Streamlit

