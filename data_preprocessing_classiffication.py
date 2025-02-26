import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
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

    # **Convert SOH into classification labels**
    bins = [60, 70, 80, 90]  # Adjusted bins to fit your data range
    labels = ['Low', 'Medium', 'High']
    df['soh_class'] = pd.cut(df['soh'], bins=bins, labels=labels)

    # Drop NaN rows after binning (if any)
    df = df.dropna(subset=['soh_class'])

    # Encode categorical labels into numbers
    label_encoder = LabelEncoder()
    df['soh_class'] = label_encoder.fit_transform(df['soh_class'])

    # Check class distribution
    class_counts = df['soh_class'].value_counts()
    print("\nClass distribution after binning:\n", class_counts)

    # Step 5: Handle missing timestamps
    df = df.dropna(subset=['timestamp_data_utc'])

    # Step 6: Convert data to LSTM-compatible format
    sequence_length = 50
    features = numerical_columns

    sequences = []
    targets = []
    for i in range(len(df) - sequence_length):
        seq = df[features].iloc[i:i + sequence_length].values
        target = df['soh_class'].iloc[i + sequence_length]
        sequences.append(seq)
        targets.append(target)

    X_raw = np.array(sequences)
    y_raw = np.array(targets)

    # Step 7: Handle Class Imbalance with SMOTE (Only if needed)
    print("\nClass distribution before SMOTE:", Counter(y_raw))

    if len(set(y_raw)) > 1:  # Only apply SMOTE if multiple classes exist
        min_class_size = min(Counter(y_raw).values())  # Get smallest class count
        smote_k = min(5, min_class_size - 1)  # Ensure k_neighbors <= min_class_size - 1

        if smote_k > 0:
            smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=smote_k)
            X_resampled, y_resampled = smote.fit_resample(
                X_raw.reshape(X_raw.shape[0], -1), y_raw
            )
            X_balanced = X_resampled.reshape(X_resampled.shape[0], sequence_length, len(features))
            y_balanced = y_resampled
            print("SMOTE applied successfully.")
        else:
            print("Skipping SMOTE: Not enough samples for oversampling.")
            X_balanced, y_balanced = X_raw, y_raw
    else:
        print("Skipping SMOTE: Only one class found in y.")
        X_balanced, y_balanced = X_raw, y_raw

    print("\nFinal Class distribution:", Counter(y_balanced))
    print("Shape of X (features):", X_balanced.shape)
    print("Shape of y (targets):", y_balanced.shape)
    print("First 5 class labels of 'soh':", y_balanced[:5])

    return X_balanced, y_balanced, feature_scaler, label_encoder

# Run preprocessing pipeline
X, y, feature_scaler, label_encoder = preprocess_data()
