# Battery_life_optimization

Battery Analytics API
Project Overview
The Battery Analytics API is a FastAPI-based application designed to predict and analyze the State of Health (SoH) of electric vehicle (EV) batteries. It processes user-uploaded CSV data, stored in MongoDB, to provide SoH forecasts using LSTM and traditional machine learning models (Random Forest, Gradient Boosting, Linear Regression). The API also generates visualizations and battery parameter analytics, accessible via a web interface (e.g., at http://localhost:3000). This project supports analysts in monitoring battery performance, optimizing usage, and predicting degradation over time.
Modules and Their Functionality
1. create_users.py

Purpose: Manages user creation and updates in the MongoDB database.
Functionality:
Creates 10 users (user1 to user10) with hashed passwords (password1 to password10) using Argon2.
Assigns the analyst role to each user.
Uses pymongo for MongoDB interactions and passlib for password hashing.
Logs operations to create_users_log.txt.


Usage: Run to initialize users in the ev_database.users collection.

2. data_comparision.py

Purpose: Trains and evaluates machine learning models for SoH prediction.
Functionality:
Splits preprocessed data into training and testing sets.
Trains Random Forest, Gradient Boosting, and Linear Regression models on flattened data.
Trains an LSTM model using data_modelling2.py.
Evaluates models using MAE, MSE, R², and accuracy within a 5% tolerance.
Logs results to app.log.


Usage: Used by the API to train models for user-specific data and compare performance.

3. data_modelling2.py

Purpose: Implements LSTM-based time series forecasting for SoH.
Functionality:
Creates an LSTM model with two LSTM layers (128 and 64 units), dropout (0.3), and dense layers.
Loads existing models or trains new ones, saving them as lstm_model_<username>.keras.
Evaluates and forecasts SoH for a specified number of weeks.
Visualizes actual vs. predicted SoH and future forecasts using Matplotlib or Streamlit.
Logs operations to app.log.


Usage: Provides LSTM predictions for the API’s analytics endpoint.

4. setup_mongodb.py

Purpose: Initializes the MongoDB database and collections.
Functionality:
Creates the ev_database with collections: users, csv_uploads, sessions.
Sets up indexes for efficient querying (e.g., unique username in users).
Logs setup status to app.log.


Usage: Run before using the API to ensure the database is configured.

5. data_visualization.py

Purpose: Generates visualizations and calculates battery parameters.
Functionality:
Creates JSON data for plots: SoC over time, SoH over time, slope vs. speed, and variable-specific time series.
Calculates parameters like initial/current SoH, charging time, distance, energy consumption, and regeneration efficiency.
Handles missing columns gracefully, logging errors to app.log.


Usage: Used by the API to deliver visualizations and analytics to the frontend.

6. data_modelling3.py

Purpose: Trains and evaluates traditional ML models for SoH prediction.
Functionality:
Trains Random Forest, Gradient Boosting, and Linear Regression on preprocessed data.
Evaluates models using MAE, MSE, and R².
Visualizes predictions vs. actual SoH using scatter plots.
Logs results to app.log.


Usage: Provides baseline ML model comparisons, typically run standalone for testing.

7. data_extraction.py

Purpose: Loads and processes user-specific data from MongoDB.
Functionality:
Fetches the latest CSV data from the csv_uploads collection for a given user.
Standardizes column names (e.g., slopeº to slope_deg, SoC(%) to soc).
Performs feature engineering on timestamp_data_utc (extracts year, month, week, day, hour).
Drops duplicate columns and handles missing data.
Logs operations to app.log.


Usage: Core data loading module for preprocessing and analytics.

8. main_fas.py

Purpose: FastAPI application serving the Battery Analytics API.
Functionality:
Handles user authentication via JWT tokens (OAuth2 with Argon2 hashing).
Provides endpoints for CSV uploads, model training, forecasting, and analytics.
Supports CORS for frontend integration (e.g., http://localhost:3000).
Integrates with other modules for data processing, modeling, and visualization.
Logs API operations to app.log.


Usage: The main entry point for running the API server.

9. data_preprocessing.py

Purpose: Preprocesses data for model training.
Functionality:
Selects valid numerical features, excluding the target soh.
Applies MinMaxScaler to features and target.
Creates sequences for LSTM (default length: 50 timesteps).
Handles missing/invalid data, ensuring no NaN or infinite values.
Logs preprocessing steps to app.log.


Usage: Prepares data for data_comparision.py and data_modelling2.py.

10. fix_csv_columns.py

Purpose: Standardizes and corrects CSV column names.
Functionality:
Renames columns to consistent formats (e.g., Motor Pwr(w) to motor_pwrw).
Drops invalid or duplicate columns (e.g., soc.1).
Verifies required columns, raising errors if missing.
Saves corrected CSV to a specified output path.
Logs operations to fix_csv_log.txt.


Usage: Run before uploading CSVs to ensure compatibility with the API.

Why These Algorithms Were Chosen
1. LSTM (Long Short-Term Memory)

Reason: LSTMs excel in time-series forecasting, capturing long-term dependencies in sequential data like battery SoH, which evolves over time based on usage patterns (e.g., soc, speed, amb_temp).
Advantages:
Handles sequential data with variable-length dependencies.
Robust to noise in high-frequency data (e.g., per-minute readings).
Suitable for forecasting future SoH values (e.g., 4 weeks ahead).


Implementation: Used in data_modelling2.py with a sequence length of 50 timesteps, leveraging two LSTM layers with dropout to prevent overfitting.

2. Random Forest

Reason: Random Forest is a robust ensemble method that handles non-linear relationships and feature interactions, making it ideal for modeling complex battery data with multiple features (e.g., motor_pwrw, slope_deg).
Advantages:
Reduces overfitting through bagging and feature randomness.
Provides feature importance for interpretability.
Performs well on tabular data after flattening LSTM sequences.


Implementation: Used in data_comparision.py and data_modelling3.py with 100 estimators for stable predictions.

3. Gradient Boosting

Reason: Gradient Boosting captures complex patterns through sequential tree building, complementing Random Forest by focusing on correcting errors in predictions.
Advantages:
High accuracy for regression tasks like SoH prediction.
Handles imbalanced feature importance effectively.
Robust to outliers in battery data.


Implementation: Used in data_comparision.py and data_modelling3.py with 100 estimators, tuned for regression.

4. Linear Regression

Reason: Serves as a baseline model to provide a simple, interpretable benchmark for comparing against more complex models (LSTM, Random Forest, Gradient Boosting).
Advantages:
Fast training and inference.
Interpretable coefficients for understanding feature impacts.
Useful for detecting linear trends in SoH degradation.


Implementation: Used in data_comparision.py and data_modelling3.py for quick benchmarking.

Rationale for Multiple Models: Combining LSTM for time-series forecasting with traditional ML models (Random Forest, Gradient Boosting, Linear Regression) allows for robust comparison, leveraging LSTM’s strength in sequential data and traditional models’ effectiveness in tabular data. This ensures both accuracy and interpretability for SoH predictions.
Setup and Execution Steps
1. Clone the Repository
Clone the project repository to your local machine:
git clone <repository-url>
cd <repository-directory>

2. Install Required Libraries
Install the necessary Python libraries using pip. Create a virtual environment for isolation:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Create a requirements.txt file with the following content:
fastapi==0.115.0
uvicorn==0.30.6
pymongo==4.8.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
tensorflow==2.17.0
matplotlib==3.9.2
python-jose==3.3.0
passlib[argon2]==1.7.4
python-dotenv==1.0.1

Install the libraries:
pip install -r requirements.txt

3. Set Up MongoDB

Install MongoDB: Download and install MongoDB Community Edition (https://www.mongodb.com/try/download/community).
Start MongoDB: Run the MongoDB server locally:mongod --dbpath /path/to/data/db

Ensure mongodb://localhost:27017 is accessible.
Run Database Setup: Initialize the ev_database and collections:python setup_mongodb.py



4. Create Users
Create or update users in the MongoDB database:
python create_users.py

This creates 10 users (user1 to user10) with passwords (password1 to password10) and the analyst role.
5. Prepare CSV Data

Fix CSV Columns: If your CSV files have inconsistent column names, run:python fix_csv_columns.py <input_csv_path> <output_csv_path>

Example:python fix_csv_columns.py D:\final_merge_batches\synthetic_weeeekly_data_150weeks.csv D:\final_merge_batches\synthetic_weeeekly_data_150weeks_corrected.csv


Upload CSV: Use the API’s /upload-csv endpoint or manually insert CSV data into the csv_uploads collection in MongoDB.

6. Run the FastAPI Application
Start the FastAPI server using uvicorn:
uvicorn main_fas:app --host 0.0.0.0 --port 8000


Access the API at http://localhost:8000.
Use the interactive API docs at http://localhost:8000/docs for testing endpoints.
The frontend (if available) can connect to http://localhost:3000.

7. Test the API

Authenticate: Use the /token endpoint with a username (user1) and password (password1) to obtain a JWT token.
Upload CSV: Use the /upload-csv endpoint to upload corrected CSV files.
Train Models: Call the /train endpoint to train models for a user.
Get Analytics: Use the /analytics endpoint to retrieve SoH predictions, visualizations, and battery parameters.

8. Visualize Results

The /analytics endpoint returns JSON data for plots (e.g., SoC over time, SoH forecasts).
Use the frontend at http://localhost:3000 to view visualizations, or run data_modelling2.py or data_modelling3.py standalone for Matplotlib plots.

Running the FastAPI Application
To run the FastAPI application in main_fas.py, use:
uvicorn main_fas:app --host 0.0.0.0 --port 8000


host: 0.0.0.0 allows access from any network interface.
port: 8000 is the default port; adjust if needed.
Ensure MongoDB is running and the database is set up before starting the server.

Required Libraries
The following libraries are required, based on the imports across all modules:

fastapi==0.115.0: For building the API.
uvicorn==0.30.6: For running the FastAPI server.
pymongo==4.8.0: For MongoDB interactions.
pandas==2.2.2: For data manipulation and CSV processing.
numpy==1.26.4: For numerical computations.
scikit-learn==1.5.1: For traditional ML models and metrics.
tensorflow==2.17.0: For LSTM model training and inference.
matplotlib==3.9.2: For plotting results.
python-jose==3.3.0: For JWT authentication.
passlib[argon2]==1.7.4: For password hashing with Argon2.
python-dotenv==1.0.1: For environment variable management.

Install them using the requirements.txt file provided above.
Notes

Data Requirements: Ensure CSV files contain required columns (e.g., timestamp_data_utc, soh, soc). Use fix_csv_columns.py to standardize column names.
MongoDB Configuration: Update MONGODB_URI in a .env file if using a non-default MongoDB instance.
Security: Replace the SECRET_KEY in main_fas.py with a secure key in production.
Customization: Update the frontend URL in main_fas.py (CORS allow_origins) if using a different frontend host.
Troubleshooting: Check app.log or fix_csv_log.txt for errors during setup or execution.

Example Workflow

Start MongoDB and run setup_mongodb.py.
Create users with create_users.py.
Fix a CSV file using fix_csv_columns.py.
Start the FastAPI server with uvicorn main_fas:app --host 0.0.0.0 --port 8000.
Upload a CSV via the /upload-csv endpoint.
Train models using the /train endpoint.
Access analytics and visualizations via the /analytics endpoint or the frontend.

For further assistance, refer to the API documentation at http://localhost:8000/docs or check the logs for debugging.
