import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def read_and_merge_sensor_data(folder_path):
    acc = pd.read_csv(os.path.join(folder_path, "Accelerometer.csv"), delimiter='\t')
    gyro = pd.read_csv(os.path.join(folder_path, "Gyroscope.csv"), delimiter='\t')
    mag = pd.read_csv(os.path.join(folder_path, "Magnetometer.csv"), delimiter='\t')

    # Replace commas with dots and convert to numeric
    for df in [acc, gyro, mag]:
        df.replace(",", ".", regex=True, inplace=True)
        df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce')

    # Drop time column (first column)
    acc = acc.iloc[:, 1:]
    gyro = gyro.iloc[:, 1:]
    mag = mag.iloc[:, 1:]

    # Ensure equal length
    min_len = min(len(acc), len(gyro), len(mag))
    acc = acc.iloc[:min_len]
    gyro = gyro.iloc[:min_len]
    mag = mag.iloc[:min_len]

    # Concatenate all sensor readings horizontally
    merged = pd.concat([acc, gyro, mag], axis=1)
    merged.dropna(inplace=True)  # Drop any rows with NaN values

    return merged

def load_all_data():
    base_path = "dataset"
    all_data = []
    labels = []

    road_types = {
        "dirt_road": "Dirt Road",
        "medium_road": "Medium Road",
        "good_road": "Good Road"
    }

    for folder, label in road_types.items():
        path = os.path.join(base_path, folder)
        if os.path.exists(path):
            print(f"📂 Processing: {label}")
            merged_data = read_and_merge_sensor_data(path)
            all_data.append(merged_data)
            labels += [label] * len(merged_data)

    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data, labels

def main():
    print("🔁 Loading and preprocessing data...")
    X, y = load_all_data()

    if X.empty:
        print("❌ No valid data found. Check if the sensor files are correctly formatted.")
        return

    print("🔢 Scaling features and encoding labels...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("🧠 Training classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y_encoded)

    print("💾 Saving model and encoder...")
    joblib.dump(model, "road_condition_model.pkl")
    joblib.dump(encoder, "label_encoder.pkl")

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
