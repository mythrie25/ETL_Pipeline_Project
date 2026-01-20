# DATA PIPELINE: ETL PROCESS
# Author: Mythri
# Tools: Pandas, NumPy, Scikit-learn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# -----------------------------
# 1. EXTRACT: Load the dataset
# -----------------------------
def extract_data(file_path):
    print("Step 1: Extracting data...")
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  # Remove extra spaces in column names
    print("Data extracted successfully!")
    print("Columns found:", list(data.columns))
    return data

# ------------------------------------
# 2. PREPROCESS: Handle missing values
# ------------------------------------
def preprocess_data(data):
    print("\nStep 2: Preprocessing data...")
    numeric_features = data.select_dtypes(include=[np.number])
    categorical_features = data.select_dtypes(include=[object])

    # Handle missing numerical values
    num_imputer = SimpleImputer(strategy="mean")
    numeric_features = pd.DataFrame(
        num_imputer.fit_transform(numeric_features),
        columns=numeric_features.columns
    )

    # Handle missing categorical values ONLY if present
    if not categorical_features.empty:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        categorical_features = pd.DataFrame(
            cat_imputer.fit_transform(categorical_features),
            columns=categorical_features.columns
        )
    else:
        print("No categorical columns found.")
        categorical_features = pd.DataFrame()  # keep as empty

    print("Missing values handled.")
    print("Numeric columns:", list(numeric_features.columns))
    print("Categorical columns:", list(categorical_features.columns))
    return numeric_features, categorical_features

# ---------------------------------
# 3. TRANSFORM: Encode & Scale data
# ---------------------------------
def transform_data(numeric_data, categorical_data):
    print("\nStep 3: Transforming data (encoding & scaling)...")

    # Encode categorical columns ONLY if present
    if not categorical_data.empty:
        encoder = LabelEncoder()
        for column in categorical_data.columns:
            categorical_data[column] = encoder.fit_transform(categorical_data[column])
    else:
        print("No categorical columns to encode.")

    # Scale numerical columns
    scaler = StandardScaler()
    numeric_data = pd.DataFrame(
        scaler.fit_transform(numeric_data),
        columns=numeric_data.columns
    )

    print("Data transformed successfully!")
    return pd.concat([numeric_data, categorical_data], axis=1)

# -----------------------------
# 4. LOAD: Save processed data
# -----------------------------
def load_data(final_data):
    print("\nStep 4: Loading data (saving to CSV)...")
    final_data.to_csv("processed_data.csv", index=False)
    print("Processed data saved as 'processed_data.csv'.")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline():
    print("=== ETL PIPELINE STARTED ===\n")

    # CSV file path
    file_path = r"C:\Users\mythr\OneDrive\Desktop\pipeline project\Customer Purchase Data.csv"

    # 1. Extract
    data = extract_data(file_path)

    # 2. Split features and target
    target_column = "Purchase_Frequency"  # <-- change if your target is different
    if target_column not in data.columns:
        raise KeyError(f"Column '{target_column}' not found in CSV. Check your file.")

    X = data.drop(target_column, axis=1)
    y = data[target_column]
    print("\nFeatures and target separated successfully.")

    # 3. Preprocess and transform
    numeric_data, categorical_data = preprocess_data(X)
    final_features = transform_data(numeric_data, categorical_data)

    # 4. Split dataset into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        final_features, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

    # 5. Load/save processed training data
    load_data(X_train)

    print("\n=== ETL PIPELINE EXECUTED SUCCESSFULLY ===")

# -----------------------------
# Execute pipeline
# -----------------------------
if __name__ == "__main__":
    run_pipeline()
