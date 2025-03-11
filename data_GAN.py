import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data

# Load and preprocess data
X, y, feature_scaler, target_scaler = preprocess_data()

# Flatten input for GAN training
input_dim = X.shape[1] * X.shape[2]
X_flattened = X.reshape(X.shape[0], input_dim)

# ✅ Build Generator Model
def build_generator(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(alpha=0.2),
        BatchNormalization(),
        Dense(input_dim, activation='sigmoid')  # Output in range [0,1]
    ])
    return model

# ✅ Build Discriminator Model
def build_discriminator(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification (Normal/Anomalous)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Instantiate models
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)

# ✅ Combine Generator + Discriminator to create the GAN
discriminator.trainable = False  # Freeze discriminator when training GAN
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy')

# ✅ Training Hyperparameters
batch_size = 64
epochs = 2000
real_label = 0.9  # Label smoothing
fake_label = 0.1  # Fake label instead of 0

# ✅ Progressive Noise Decay (reduces over time)
initial_noise_std = 1.0
final_noise_std = 0.3

# ✅ Training Loop
for epoch in range(epochs):
    # Compute current noise level (progressively reducing noise)
    noise_std = initial_noise_std - ((initial_noise_std - final_noise_std) * (epoch / epochs))
    
    # Generate Fake Data
    noise = np.random.normal(0, noise_std, (batch_size, input_dim))
    generated_data = generator.predict(noise)

    # Get Real Data
    idx = np.random.randint(0, X_flattened.shape[0], batch_size)
    real_data = X_flattened[idx]

    # Train Discriminator
    x_combined = np.vstack((real_data, generated_data))
    y_combined = np.hstack((np.full(batch_size, real_label), np.full(batch_size, fake_label)))  # Label smoothing
    d_loss = discriminator.train_on_batch(x_combined, y_combined)

    # Train Generator
    noise = np.random.normal(0, noise_std, (batch_size, input_dim))
    g_loss = gan.train_on_batch(noise, np.full(batch_size, real_label))  # Try to fool discriminator

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss = {d_loss[0]:.4f}, G Loss = {g_loss:.4f}")

# ✅ Adaptive Threshold for Anomaly Detection
threshold = np.percentile(discriminator.predict(X_flattened), 40)  # Adjusted threshold to 40th percentile
anomaly_scores = discriminator.predict(X_flattened)
anomaly_labels = np.where(anomaly_scores < threshold, 1, 0)  # 1 = Anomalous, 0 = Normal

print("Anomaly Distribution:", np.unique(anomaly_labels, return_counts=True))
print(f"Adaptive Anomaly Threshold: {threshold:.4f}")

# ✅ Prepare SOH data for classification
soh_test = y.reshape(-1, 1)

# ✅ Step 15: Convert SOH Back to Percentage
soh_normal_scaled = soh_test[anomaly_labels == 0].reshape(-1, 1)
soh_anomalous_scaled = soh_test[anomaly_labels == 1].reshape(-1, 1)

# Inverse transform to get original SOH percentages
soh_normal = target_scaler.inverse_transform(soh_normal_scaled).flatten()
soh_anomalous = target_scaler.inverse_transform(soh_anomalous_scaled).flatten()

# ✅ Step 16: Display SOH Percentage Ranges
print(f"Average SOH for Normal: {np.mean(soh_normal):.2f}%")
print(f"Average SOH for Anomalous: {np.mean(soh_anomalous):.2f}%")

# ✅ Step 17: Plot SOH Distributions
plt.figure(figsize=(8, 5))
plt.hist(soh_normal, bins=20, alpha=0.7, label="Normal SOH", color='blue')
plt.hist(soh_anomalous, bins=20, alpha=0.7, label="Anomalous SOH", color='red')
plt.xlabel("SOH (%)")
plt.ylabel("Frequency")
plt.legend()
plt.title("SOH Distribution for Normal vs. Anomalous")
plt.show()
