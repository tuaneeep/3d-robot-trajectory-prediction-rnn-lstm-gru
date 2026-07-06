import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. TẠO DỮ LIỆU CỬA SỔ TRƯỢT (Giữ nguyên logic)
def create_sliding_window_data(tensor, window_size=10):
    X_list, y_list = [], []
    for i in range(tensor.shape[0]): # Duyệt qua 120 quỹ đạo
        traj = tensor[i]
        for j in range(tensor.shape[1] - window_size): # Trượt để tạo mẫu
            X_list.append(traj[j : j + window_size, :])
            y_list.append(traj[j + window_size, :])
    return np.array(X_list), np.array(y_list)


window_size = 10
X_big, y_big = create_sliding_window_data(trajectories_tensor, window_size)

print(f"Dữ liệu mẫu sau khi trượt cửa sổ: {X_big.shape}") # (10800, 10, 3)

# 2. CHIA DỮ LIỆU VÀ XÁO TRỘN
X_train, X_test, y_train, y_test = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

# 3. XÂY DỰNG MÔ HÌNH RNN
# SimpleRNN là dạng cơ bản nhất của mạng nơ-ron hồi quy
model_rnn = Sequential([
    # Lớp RNN với 64 đơn vị
    SimpleRNN(64, input_shape=(window_size, 3), activation='tanh'),

    # Lớp Dense trung gian
    Dense(32, activation='relu'),

    # Lớp đầu ra cho 3 tọa độ X, Y, Z
    Dense(3)
])

model_rnn.compile(optimizer='adam', loss='mse')

# 4. HUẤN LUYỆN
print("Đang huấn luyện mô hình RNN trên 10,800 mẫu...")
model_rnn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

# LƯU MÔ HÌNH VỚI TÊN rnn.h5
model_rnn.save('/content/drive/MyDrive/rnn.h5')
print("Đã lưu mô hình thành công vào file: rnn.h5")

# 5. DỰ ĐOÁN VÀ KIỂM TRA SAI SỐ
predictions = model_rnn.predict(X_test)
mse = np.mean((y_test - predictions)**2)
print(f"Sai số MSE của RNN sau khi huấn luyện: {mse:.8f}")
