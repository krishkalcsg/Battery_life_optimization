import numpy as np
import tensorflow as tf
import pandas as pd
from data_preprocessing import preprocess_data  # Import preprocess_data from the other script
import matplotlib.pyplot as plt

# Load the preprocessed data
X, y, feature_scaler, target_scaler = preprocess_data()

# Define the Generator
def build_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim=35, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(50 * 35, activation='sigmoid'))  # Generate a sequence of 50 time steps, each with 35 features
    model.add(tf.keras.layers.Reshape((50, 35)))  # Reshape output to (50, 35)
    return model

# Define the Discriminator
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(50, 35)))  # Flatten the input (50 time steps, 35 features per time step)
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Real or fake classification
    return model

# Define the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile models
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN (example setup)
def train_gan(X_train, epochs=10000, batch_size=64):
    for epoch in range(epochs):
        # Select a random batch of real data
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_sequences = X_train[idx]

        # Generate fake data using the generator
        noise = np.random.normal(0, 1, (batch_size, 35))  # Generate noise to pass to the generator
        fake_sequences = generator.predict(noise)

        # Train the discriminator (real = 1, fake = 0)
        discriminator_loss_real = discriminator.train_on_batch(real_sequences, np.ones((batch_size, 1)))
        discriminator_loss_fake = discriminator.train_on_batch(fake_sequences, np.zeros((batch_size, 1)))

        # Train the generator (the discriminator is frozen)
        noise = np.random.normal(0, 1, (batch_size, 35))
        generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Discriminator Loss: {discriminator_loss_real[0] + discriminator_loss_fake[0]} | Generator Loss: {generator_loss}")

# Train the GAN model
train_gan(X, epochs=100)

# Detect anomalies
def detect_anomalies(X_test, discriminator, threshold=0.5):
    predictions = discriminator.predict(X_test)
    anomalies = predictions < threshold  # Anomalies if predicted as fake
    return anomalies

# Example anomaly detection
anomalies = detect_anomalies(X, discriminator)
print("Anomalies detected:", np.sum(anomalies))

# Revert the data back to original scale
# Revert the data back to original scale
anomalous_data = X[anomalies.flatten()]

# Flatten the data to 2D for inverse transformation (only flatten time steps if needed)
anomalous_data_flat = anomalous_data.reshape((-1, anomalous_data.shape[-1]))

# Inverse transform
anomalous_data_original_scale = feature_scaler.inverse_transform(anomalous_data_flat)

# If necessary, reshape back to the original 3D structure (e.g., [batch_size, time_steps, features])
anomalous_data_original_scale = anomalous_data_original_scale.reshape((-1, 50, 35))  # Adjust based on your specific data structure

# Convert to DataFrame with original columns
numerical_columns = ['elv_spy', 'speed', 'soc', 'amb_temp', 'regenwh', 'Motor Pwr(w)', 
                     'Aux Pwr(100w)', 'Motor Temp', 'Torque Nm', 'rpm', 'capacity', 
                     'ref_consumption', 'wind_mph', 'wind_kph', 'wind_degree', 
                     'Frontal_Wind', 'Veh_deg', 'totalVehicles', 'speedAvg', 'max_speed', 
                     'radius', 'step', 'acceleration(m/s²)', 'actualBatteryCapacity(Wh)', 
                     'speed(m/s)', 'speedFactor', 'totalEnergyConsumed(Wh)', 
                     'totalEnergyRegenerated(Wh)', 'lon', 'lat', 'alt', 'slope(º)', 
                     'completedDistance(km)', 'mWh', 'remainingRange(km)']

# Ensure the shape matches the column length
anomalous_df = pd.DataFrame(anomalous_data_original_scale.reshape((-1, 35)), columns=numerical_columns)

# Plot anomaly results (example: anomaly count by feature)
anomalous_df.plot(kind='box', figsize=(12, 6))
plt.title('Anomalous Data Distribution')
plt.xticks(rotation=90)
plt.show()

# Visualize the anomalies for one feature (e.g., 'speed')
plt.figure(figsize=(10, 6))
plt.plot(anomalous_df['speed'], 'ro', label='Anomalies')
plt.title('Anomalous Speed Data')
plt.xlabel('Index')
plt.ylabel('Speed')
plt.legend()
plt.show()

