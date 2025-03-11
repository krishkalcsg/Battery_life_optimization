# Updated data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_extraction import load_data

def preprocess_data():
    # Load data from data_extraction.py
    df = load_data()

    # Debug: Print dataset columns
    print("Columns in dataset:", df.columns.tolist())

    # Step 1: Feature engineering on 'timestamp_data_utc'
    df['year'] = df['timestamp_data_utc'].dt.year
    df['month'] = df['timestamp_data_utc'].dt.month
    df['week'] = df['timestamp_data_utc'].dt.isocalendar().week
    df['day'] = df['timestamp_data_utc'].dt.day
    df['hour'] = df['timestamp_data_utc'].dt.hour

    # Step 2: Handle missing numerical values
    numerical_columns = [
        'elv_spy', 'speed', 'soc', 'amb_temp', 'regenwh', 'Motor Pwr(w)', 
        'Aux Pwr(100w)', 'Motor Temp', 'Torque Nm', 'rpm', 'capacity',
        'ref_consumption', 'wind_mph', 'wind_kph', 'wind_degree',
        'Frontal_Wind', 'Veh_deg', 'totalVehicles', 'speedAvg', 'max_speed', 
        'radius', 'step', 'acceleration(m/s²)', 'actualBatteryCapacity(Wh)', 
        'speed(m/s)', 'speedFactor', 'totalEnergyConsumed(Wh)', 
        'totalEnergyRegenerated(Wh)', 'lon', 'lat', 'alt', 'slope(º)', 
        'completedDistance(km)', 'mWh', 'remainingRange(km)'
    ]

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

    # Step 3: Normalize numerical columns
    feature_scaler = MinMaxScaler()
    df[numerical_columns] = feature_scaler.fit_transform(df[numerical_columns])

    # Step 4: Normalize target column 'soh'
    target_scaler = MinMaxScaler()
    df['soh'] = target_scaler.fit_transform(df[['soh']])

    # Step 5: Handle missing timestamps
    categorical_columns = ['timestamp_data_utc']
    df = df.dropna(subset=categorical_columns)

    # Step 6: Convert data to LSTM-compatible format
    sequence_length = 50
    features = numerical_columns

    sequences = []
    targets = []
    for i in range(len(df) - sequence_length):
        seq = df[features].iloc[i:i + sequence_length].values
        target = df['soh'].iloc[i + sequence_length]
        sequences.append(seq)
        targets.append(target)

    X = np.array(sequences)
    y = np.array(targets)

    return X, y, feature_scaler, target_scaler

# Run preprocessing pipeline
X, y, feature_scaler, target_scaler = preprocess_data()

print("Shape of X (features):", X.shape)
print("Shape of y (targets):", y.shape)
print("First 5 values of 'soh' in y:", y[:5])
# Print the first sequence (50 samples, 35 features)
print("First sequence of X (50 samples, 35 features):")
print(X[0])  # Print the first sequence, shape should be (50, 35)
# Pretty print the first sequence
