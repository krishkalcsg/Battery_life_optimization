import pandas as pd

def load_data():
    # Load the dataset from the CSV file
    file_path = r"D:/final_merge_batches/final_merge_batches/synthetic_weeeekly_data_150weeks.csv"
    df = pd.read_csv(file_path)
    
    # Ensure the 'timestamp_data_utc' column is in datetime format
    df['timestamp_data_utc'] = pd.to_datetime(df['timestamp_data_utc'])
    
    return df
df = load_data()

# Retrieve and print the list of column names
column_names = df.columns.tolist()
print("Column names:", column_names)

df = load_data()
print(df.head())












