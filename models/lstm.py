import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. TẠO DỮ LIỆU CỬA SỔ TRƯỢT (Giữ nguyên logic chuẩn)
def create_sliding_window_data(tensor, window_size=10):
    X_list, y_list = [], []
    for i in range(tensor.shape[0]): # Duyệt qua 120 quỹ đạo
        traj = tensor[i]
        for j in range(tensor.shape[1] - window_size):
            X_list.append(traj[j : j + window_size, :])
            y_list.append(traj[j + window_size, :])
    return np.array(X_list), np.array(y_list)

# Giả sử trajectories_tensor của bạn là (120, 100, 3)
window_size = 10
X_big, y_big = create_sliding_window_data(trajectories_tensor, window_size)

print(f"Tổng số mẫu dữ liệu: {X_big.shape}")

# 2. CHIA DỮ LIỆU VÀ XÁO TRỘN
X_train, X_test, y_train, y_test = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

# 3. XÂY DỰNG MÔ HÌNH LSTM (Đã sửa tên biến thành model_lstm)
model_lstm = Sequential([
    LSTM(64, input_shape=(window_size, 3), activation='tanh'),
    Dense(32, activation='relu'),
    Dense(3)
])

model_lstm.compile(optimizer='adam', loss='mse')

# 4. HUẤN LUYỆN
print("Đang huấn luyện mô hình model_lstm trên 10,800 mẫu...")
model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

# LƯU MÔ HÌNH VỚI TÊN lstm.h5
model_lstm.save('/content/drive/MyDrive/lstm.h5')
print("Đã lưu mô hình thành công vào file: lstm.h5")

# 5. DỰ ĐOÁN VÀ KIỂM TRA SAI SỐ
predictions_lstm = model_lstm.predict(X_test)
mse_lstm = np.mean((y_test - predictions_lstm)**2)
print(f"Sai số MSE của LSTM sau khi huấn luyện: {mse_lstm:.8f}")
