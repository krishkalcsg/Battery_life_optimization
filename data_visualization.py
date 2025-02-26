import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from data_extraction import load_data
import re

def plot_soc_over_time(df: pd.DataFrame, save_path: str = "soc_over_time.png"):
    df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc'])
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp_data_utc'], df['SoC(%)'], color='blue', label='State of Charge')
    plt.title('State of Charge Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('SoC (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_variables(df: pd.DataFrame, save_path: str = "plots"):
    df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc'])

    os.makedirs(save_path, exist_ok=True)

    variables = [
        'elv_spy', 'speed', 'soc', 'amb_temp', 'regenwh', 'Motor Pwr(w)',
        'Aux Pwr(100w)', 'Motor Temp', 'Torque Nm', 'rpm', 'route_id', 'longitude',
        'latitude', 'altitude', 'car_id', 'time_diff', 'route_description', 'capacity',
        'ref_consumption', 'wind_mph', 'wind_kph', 'wind_degree', 'Frontal_Wind', 'Veh_deg',
        'totalVehicles', 'speedAvg', 'max_speed', 'radius', 'step',
        'actualBatteryCapacity(Wh)', 'SoC(%)', 'speed(m/s)', 'speedFactor',
        'totalEnergyConsumed(Wh)', 'totalEnergyRegenerated(Wh)', 'lon', 'lat', 'alt',
        'slope(º)', 'completedDistance(km)', 'mWh', 'remainingRange(km)', 'soh'
    ]

    file_paths = []
    for var in variables:
        if var in df.columns:
            plt.figure(figsize=(12, 5))
            plt.plot(df['timestamp_data_utc'], df[var], label=var, linewidth=2)
            plt.xlabel("Timestamp (Weeks)")
            plt.ylabel(var)
            plt.title(f"Time Series Plot of {var}")
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)

            # ✅ Clean variable name for safe filenames
            safe_var_name = re.sub(r"[^a-zA-Z0-9_]", "_", var)  # Replace special characters
            plot_filename = f"{save_path}/{safe_var_name}_plot.png"

            plt.savefig(plot_filename)
            plt.close()
            file_paths.append(plot_filename)
        else:
            print(f"Column '{var}' not found in the dataset.")

    return file_paths


def plot_slope_vs_speed(df, save_path="plots/slope_vs_speed.png"):
    if "slope(º)" not in df.columns or "speed(m/s)" not in df.columns:
        raise ValueError("Missing required columns: 'slope(º)' or 'speed(m/s)' in the DataFrame.")

    plt.figure(figsize=(8, 6))
    plt.scatter(df["speed(m/s)"], df["slope(º)"], alpha=0.5)
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Slope (º)")
    plt.title("Slope vs Speed Scatter Plot")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path)
    plt.close()

    return save_path  # Ensure the function returns the correct file path

def plot_soh_over_time(df: pd.DataFrame, save_path: str = "soh_over_time.png"):
    df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc'])
    if 'soh' not in df.columns:
        print("Column 'soh' not found in the dataset.")
        return None
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp_data_utc'], df['soh'], color='green', label='State of Health')
    plt.title('State of Health Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('soh')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_soh_distribution(df: pd.DataFrame, save_path: str = "soh_distribution.png"):
    if 'soh' not in df.columns:
        print("Column 'soh' not found in the dataset.")
        return None
    plt.figure(figsize=(10, 6))
    df['soh'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of State of Health (SoH) Values')
    plt.xlabel('soh')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path

if __name__ == "__main__":
    df = load_data()
    plot_soh_distribution(df)
