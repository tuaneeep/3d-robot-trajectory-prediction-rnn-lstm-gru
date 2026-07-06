# 3D Robot Trajectory Prediction using RNN, LSTM, and GRU

This project focuses on predicting the 3D trajectory of a robot using time-series forecasting. We implement and compare three popular Recurrent Neural Network architectures: **Simple RNN**, **LSTM**, and **GRU**, using a synthetic dataset to evaluate their performance in spatial coordinate prediction.

## 🚀 Project Overview
The objective is to analyze how different recurrent architectures learn from sequential coordinate data $(x, y, z)$ and predict future positions. This is a classic time-series regression task in autonomous robotics.

## 📊 Dataset
*   **Origin:** Synthetic Data generated via a Random Walk model.
*   **Volume:** 12,000 coordinate samples.
*   **Features:** Spatial coordinates $(x, y, z)$.
*   **Preprocessing:** Data is windowed using a sliding window technique (`window_size=20`) and normalized using `MinMaxScaler` to ensure stable training.

## 🛠 Model Architectures
1.  **Simple RNN:** Used as the baseline model.
2.  **LSTM (Long Short-Term Memory):** Designed to capture long-term dependencies in the trajectory.
3.  **GRU (Gated Recurrent Unit):** An optimized architecture aimed at efficient sequence modeling.

## 📈 Performance Results
The models were evaluated based on Mean Squared Error (MSE) and Mean Absolute Error (MAE) on a test trajectory.

| Model | MSE (Total) | MAE (Total) |
| :--- | :--- | :--- |
| **Simple RNN** | 0.000682 | 0.014729 |
| **LSTM** | 0.000804 | 0.015951 |
| **GRU** | **0.000534** | **0.013579** |

*Note: The GRU architecture demonstrated the highest accuracy and stability for this specific dataset.*

## 🛠 Tech Stack
*   **Language:** Python 3.x
*   **Deep Learning:** TensorFlow / Keras
*   **Data Processing:** NumPy, Pandas, Scikit-learn
*   **Visualization:** Matplotlib

## ⚙️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
