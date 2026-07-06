import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense # Import GRU
from sklearn.model_selection import train_test_split

# 1. TẠO DỮ LIỆU CỬA SỔ TRƯỢT (Giữ nguyên logic chuẩn của bạn)
def create_sliding_window_data(tensor, window_size=10):
    X_list, y_list = [], []
    for i in range(tensor.shape[0]):
        traj = tensor[i]
        for j in range(tensor.shape[1] - window_size):
            X_list.append(traj[j : j + window_size, :])
            y_list.append(traj[j + window_size, :])
    return np.array(X_list), np.array(y_list)

# Giả sử trajectories_tensor là (120, 100, 3)
window_size = 10
X_big, y_big = create_sliding_window_data(trajectories_tensor, window_size)

# 2. CHIA DỮ LIỆU
X_train, X_test, y_train, y_test = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

# 3. XÂY DỰNG MÔ HÌNH GRU
model_gru = Sequential([
    # GRU thường chỉ cần 2 cổng (Update và Reset) thay vì 3 cổng như LSTM
    GRU(64, input_shape=(window_size, 3), activation='tanh'),
    Dense(32, activation='relu'),
    Dense(3)
])

model_gru.compile(optimizer='adam', loss='mse')

# 4. HUẤN LUYỆN
print("Đang huấn luyện mô hình GRU...")
model_gru.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

# LƯU MÔ HÌNH VỚI TÊN gru.h5
model_gru.save('/content/drive/MyDrive/gru.h5')
print("Đã lưu mô hình GRU thành công!")

# 5. KIỂM TRA SAI SỐ
predictions = model_gru.predict(X_test)
mse = np.mean((y_test - predictions)**2)
print(f"Sai số MSE của GRU: {mse:.8f}")
