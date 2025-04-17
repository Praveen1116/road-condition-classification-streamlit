import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and label encoder
model = joblib.load("road_condition_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🚗 Road Condition Classifier")
st.write("Upload sensor files to classify road condition as Dirt, Medium, or Good.")

# File uploads
acc_file = st.file_uploader("📥 Upload Accelerometer.csv", type=["csv"])
gyro_file = st.file_uploader("📥 Upload Gyroscope.csv", type=["csv"])
mag_file = st.file_uploader("📥 Upload Magnetometer.csv", type=["csv"])

# Function to preprocess uploaded files
def preprocess_uploaded_files(acc_file, gyro_file, mag_file):
    try:
        acc = pd.read_csv(acc_file, delimiter='\t')
        gyro = pd.read_csv(gyro_file, delimiter='\t')
        mag = pd.read_csv(mag_file, delimiter='\t')

        # Rename first column to "Time (s)" just in case
        acc.columns.values[0] = "Time (s)"
        gyro.columns.values[0] = "Time (s)"
        mag.columns.values[0] = "Time (s)"

        # Replace commas with dots and convert to numeric
        for df in [acc, gyro, mag]:
            df.replace(",", ".", regex=True, inplace=True)
            df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce')

        # Fix time column
        time_series = acc["Time (s)"].astype(str).str.replace(",", ".", regex=False)
        time_series = pd.to_numeric(time_series, errors='coerce')

        # Drop 'Time (s)' and keep aligned data
        acc = acc.drop(columns=["Time (s)"])
        gyro = gyro.drop(columns=["Time (s)"])
        mag = mag.drop(columns=["Time (s)"])

        # Match lengths
        min_len = min(len(acc), len(gyro), len(mag), len(time_series))
        acc = acc.iloc[:min_len]
        gyro = gyro.iloc[:min_len]
        mag = mag.iloc[:min_len]
        time_series = time_series.iloc[:min_len]

        # Concatenate all features
        combined = pd.concat([acc, gyro, mag], axis=1)
        combined.fillna(combined.mean(), inplace=True)

        return combined, time_series, acc, gyro, mag

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None, None, None, None

# Function to plot 3-axis sensor data
def plot_sensor_data(time, df, title):
    try:
        if df is not None and not df.empty and time is not None:
            time = pd.to_numeric(time, errors='coerce')
            df = df.apply(pd.to_numeric, errors='coerce')

            # Combine and clean
            combined = pd.concat([time, df], axis=1).dropna()
            time_cleaned = combined.iloc[:, 0]
            x = combined.iloc[:, 1]
            y = combined.iloc[:, 2]
            z = combined.iloc[:, 3]

            fig, ax = plt.subplots()
            ax.plot(time_cleaned, x, label="X", alpha=0.8)
            ax.plot(time_cleaned, y, label="Y", alpha=0.8)
            ax.plot(time_cleaned, z, label="Z", alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Sensor Value")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning(f"⚠️ No valid data to plot for {title}.")
    except Exception as e:
        st.error(f"Plotting error in {title}: {e}")

# Predict button
if st.button("🧠 Predict Road Condition"):
    if acc_file and gyro_file and mag_file:
        data, time_series, acc_data, gyro_data, mag_data = preprocess_uploaded_files(acc_file, gyro_file, mag_file)

        if data is not None and not data.empty:
            sample = data.mean().values.reshape(1, -1)
            prediction = model.predict(sample)
            confidence = model.predict_proba(sample)

            label = encoder.inverse_transform(prediction)[0]
            confidence_percent = np.max(confidence) * 100

            st.success(f"🏁 **Predicted Road Condition: {label}**")
            st.info(f"📊 Model Confidence: {confidence_percent:.2f}%")

            # Show plots
            st.subheader("📈 Sensor Data Visualization")
            if time_series is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    plot_sensor_data(time_series, acc_data, "Accelerometer")
                with col2:
                    plot_sensor_data(time_series, gyro_data, "Gyroscope")
                with col3:
                    plot_sensor_data(time_series, mag_data, "Magnetometer")
        else:
            st.warning("⚠️ Could not process data. Please check the files.")
    else:
        st.warning("📂 Please upload all three files to proceed.")
