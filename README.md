
# 🚧 Road Condition Classification using Mobile Sensor Data (Streamlit + ML)

This project is a machine learning-based application that classifies road conditions into **Dirt Road**, **Medium Road**, and **Good Road** using **mobile sensor data** — specifically **Accelerometer**, **Gyroscope**, and **Magnetometer**.  
Built with **Python**, **Scikit-learn**, and **Streamlit**, the application takes sensor `.csv` files as input and predicts road conditions in real time through a user-friendly web interface.

---

## 🔍 Problem Statement

Accurate road condition monitoring is crucial for:
- Autonomous driving systems 🚗
- Smart city planning 🏙️
- Driver safety applications 🛡️

Manually inspecting road quality is inefficient. This project aims to automate that process using mobile sensor data and machine learning.

---

## 🧠 Tech Stack

- **Frontend/UI**: Streamlit (Python)
- **Backend**: Python
- **Machine Learning Model**: Random Forest Classifier
- **Tools**: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- **Data Input**: CSV files for each sensor (Accelerometer, Gyroscope, Magnetometer)

---

## ⚙️ Features

- 📥 Upload CSV files from sensors.
- 🤖 Pre-trained **Random Forest Classifier** to predict:
  - `Dirt Road`
  - `Medium Road`
  - `Good Road`
- 📊 Live visualization of sensor data.
- 📈 Display of model confidence/probability.
- 🧠 Label encoding using pre-trained `label_encoder.pkl`.

---

## 📂 Project Structure

```
road-condition-classification-streamlit/
│
├── road-condition-model.pkl        # Trained Random Forest model
├── label_encoder.pkl               # Label encoder for target labels
├── app.py                          # Streamlit app
├── data/                           # Sample sensor CSVs
│   ├── accelerometer.csv
│   ├── gyroscope.csv
│   └── magnetometer.csv
├── utils/                          # Utility functions
├── README.md                       # Project description
└── requirements.txt                # Dependencies
```

---

## 🧪 Model Performance

- Achieves high accuracy on **Dirt** and **Good** roads.
- Medium road classification has occasional misclassifications due to overlap in sensor patterns — future work includes advanced feature engineering or deep learning models to improve accuracy.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/road-condition-classification-streamlit.git
cd road-condition-classification-streamlit
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📌 Future Enhancements

- 📈 Improve model performance for medium road classification
- 📦 Add support for real-time mobile sensor integration (e.g., via APIs)
- 🧠 Experiment with deep learning models like LSTM or CNN for time-series data

---

## 🙌 Contributions

Feel free to open issues, suggest features, or contribute to this project!  
Pull requests are welcome.
