import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import preprocess_data  # Correct import from your preprocessing script

# 1. Build Generator Model
def build_generator(latent_dim, sequence_length, num_features):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=latent_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(sequence_length * num_features, activation='tanh'))
    model.add(layers.Reshape((sequence_length, num_features)))  # Reshape to match the input shape
    return model

# 2. Build Discriminator Model
def build_discriminator(sequence_length, num_features):
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(sequence_length, num_features)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output (real vs fake)
    return model

# 3. Build GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False  # We will train only the generator while training the GAN
    model = tf.keras.Sequential([generator, discriminator])
    return model

# Compile GAN
def compile_gan(generator, discriminator):
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

# 4. Train GAN
def train_gan(generator, discriminator, gan, X_train, epochs=100, batch_size=32, latent_dim=100):
    for epoch in range(epochs):
        # Train discriminator on real and fake data
        real_data = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))  # Real data label = 1
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))  # Fake data label = 0
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))  # We want generator to fool the discriminator

        if epoch % 10 == 0:
            print(f"{epoch}/{epochs}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# 5. Anomaly Detection Based on Reconstruction Error
def detect_anomalies(generator, X_test, threshold=0.5, latent_dim=100):
    noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))
    generated_data = generator.predict(noise)
    
    anomaly_scores = []
    
    for i in range(X_test.shape[0]):
        real_data = X_test[i]
        fake_data = generated_data[i]
        
        # Calculate reconstruction error (mean absolute error)
        error = np.mean(np.abs(real_data - fake_data))  # Absolute error between real and generated
        anomaly_scores.append(error)
    
    # Flag data with high error as anomalous
    anomalies = np.array(anomaly_scores) > threshold
    return anomalies

# 6. Main function to run everything
def run_anomaly_detection():
    # Load the preprocessed data
    X, y, feature_scaler, target_scaler = preprocess_data()

    # GAN configuration
    latent_dim = 100  # Latent dimension size
    sequence_length = X.shape[1]  # Length of sequences (50 in your case)
    num_features = X.shape[2]  # Number of features (35 in your case)

    # Build and compile the GAN model
    generator = build_generator(latent_dim, sequence_length, num_features)
    discriminator = build_discriminator(sequence_length, num_features)
    gan = compile_gan(generator, discriminator)

    # Train the GAN model
    train_gan(generator, discriminator, gan, X, epochs=100, batch_size=32, latent_dim=latent_dim)

    # Detect anomalies
    anomalies = detect_anomalies(generator, X)

    # Print detected anomalies
    print("Detected anomalies:", np.where(anomalies)[0])

if __name__ == "__main__":
    run_anomaly_detection()
