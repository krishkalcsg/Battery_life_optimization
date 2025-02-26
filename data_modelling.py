import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_preprocessing import preprocess_data  # Import preprocess_data function
from tensorflow.keras.models import load_model
import joblib  # Use joblib for saving/loading scalers

# Step 1: Load preprocessed data
X, y, feature_scaler, target_scaler = preprocess_data()

# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Step 3: Define the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))  # 64 LSTM units
    model.add(Dropout(0.2))  # Add dropout to prevent overfitting
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Function to train and save the model and scalers
def train_and_save_model():
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    model = create_lstm_model(input_shape)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # Use 20% of training data for validation
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # Save the trained model
    model.save('lstm_model.keras')

    # Save the scalers using joblib
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

    return model

# Load the model and scalers if already saved or train a new one if not saved
def load_or_train_model():
    try:
        model = load_model('lstm_model.keras')  # Try to load the saved model
        feature_scaler = joblib.load('feature_scaler.pkl')  # Load the saved feature scaler
        target_scaler = joblib.load('target_scaler.pkl')  # Load the saved target scaler
        print("Model and scalers loaded successfully.")
    except:
        print("Model or scalers not found. Training a new model...")
        model = train_and_save_model()  # Train and save the model and scalers
        feature_scaler, target_scaler = None, None  # Set scalers to None for now, since they are re-trained
    return model, feature_scaler, target_scaler

# Step 4: Evaluate and make predictions (if necessary)
def evaluate_and_predict(model, X_test, y_test, target_scaler):
    # Evaluate the model on the test data
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print("Test Loss:", test_loss)
    print("Test Mean Absolute Error:", test_mae)

    # Step 5: Make predictions
    y_pred = model.predict(X_test)

    # Initialize default values in case target_scaler is None
    y_pred_original_scale, y_actual_original_scale = None, None

    # Reverse scaling: Convert predictions and actual values back to original scale (only for 'soh')
    if target_scaler is not None:
        y_pred_original_scale = target_scaler.inverse_transform(y_pred.reshape(-1, 1))  # Reshape for inverse transform
        y_actual_original_scale = target_scaler.inverse_transform(y_test.reshape(-1, 1))  # Reshape for inverse transform

        # Print the first 5 predictions and their corresponding actual values in original scale
        print("Predicted values (original scale):", y_pred_original_scale[:5].flatten())
        print("Actual values (original scale):", y_actual_original_scale[:5].flatten())

        # Check the first few target values (soh) in original scale
        print("First 5 target values (soh) in original scale:", target_scaler.inverse_transform(y[:5].reshape(-1, 1)).flatten())

    return y_pred_original_scale, y_actual_original_scale


# Train and load model or load existing model
model, feature_scaler, target_scaler = load_or_train_model()

# Evaluate and make predictions
y_pred_original_scale, y_actual_original_scale = evaluate_and_predict(model, X_test, y_test, target_scaler)
