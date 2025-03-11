import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt

# Load the preprocessed data from data_preprocessing.py
X, y, feature_scaler, target_scaler = preprocess_data()

# Check the shapes of the data to confirm
print("Shape of X (features):", X.shape)
print("Shape of y (targets):", y.shape)

# Step 1: Reshape the data for regression models
# Flatten the sequences into a 2D array where each sequence becomes one sample.
X_reshaped = X.reshape(X.shape[0], -1)  # Flatten the time-series data

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Step 3: Initialize the models
# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Linear Regression
lr_model = LinearRegression()

# Step 4: Train the models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Step 5: Make predictions using the trained models
rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

# Step 6: Reverse scaling to get the predicted SOH values in the original scale (percentage)
rf_predictions_original = target_scaler.inverse_transform(rf_predictions.reshape(-1, 1)) * 100
gb_predictions_original = target_scaler.inverse_transform(gb_predictions.reshape(-1, 1)) * 100
lr_predictions_original = target_scaler.inverse_transform(lr_predictions.reshape(-1, 1)) * 100

# Also reverse scale the actual SOH values for evaluation
y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)) * 100

# Step 7: Evaluate the models

# Random Forest Evaluation
rf_mae = mean_absolute_error(y_test_original, rf_predictions_original)
rf_mse = mean_squared_error(y_test_original, rf_predictions_original)
rf_r2 = r2_score(y_test_original, rf_predictions_original)

# Gradient Boosting Evaluation
gb_mae = mean_absolute_error(y_test_original, gb_predictions_original)
gb_mse = mean_squared_error(y_test_original, gb_predictions_original)
gb_r2 = r2_score(y_test_original, gb_predictions_original)

# Linear Regression Evaluation
lr_mae = mean_absolute_error(y_test_original, lr_predictions_original)
lr_mse = mean_squared_error(y_test_original, lr_predictions_original)
lr_r2 = r2_score(y_test_original, lr_predictions_original)

# Print the evaluation metrics for each model
print("Random Forest - MAE:", rf_mae, "MSE:", rf_mse, "R²:", rf_r2)
print("Gradient Boosting - MAE:", gb_mae, "MSE:", gb_mse, "R²:", gb_r2)
print("Linear Regression - MAE:", lr_mae, "MSE:", lr_mse, "R²:", lr_r2)

# Step 8: Plot predictions vs actual values for comparison

plt.figure(figsize=(12, 6))

# Random Forest predictions vs actual values
plt.subplot(131)
plt.scatter(y_test_original, rf_predictions_original)
plt.title("Random Forest Regressor")
plt.xlabel('Actual SOH (%)')
plt.ylabel('Predicted SOH (%)')

# Gradient Boosting predictions vs actual values
plt.subplot(132)
plt.scatter(y_test_original, gb_predictions_original)
plt.title("Gradient Boosting Regressor")
plt.xlabel('Actual SOH (%)')
plt.ylabel('Predicted SOH (%)')

# Linear Regression predictions vs actual values
plt.subplot(133)
plt.scatter(y_test_original, lr_predictions_original)
plt.title("Linear Regression")
plt.xlabel('Actual SOH (%)')
plt.ylabel('Predicted SOH (%)')

plt.tight_layout()
plt.show()
