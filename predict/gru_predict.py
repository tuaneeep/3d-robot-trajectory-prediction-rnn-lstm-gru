import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # Add this import
from tensorflow.keras.models import load_model

model_path = '/content/drive/MyDrive/gru.h5'
custom_dict = {'mse': tf.keras.losses.MeanSquaredError()}
model = load_model(model_path, custom_objects=custom_dict)

# 1. KHỞI TẠO CỬA SỔ VỚI ĐIỂM (0,0,0)
# Giả sử chúng ta dùng quỹ đạo số 26 để kiểm tra
test_traj = trajectories_tensor[26]
window_size = 10

# Tạo cửa sổ ban đầu: 9 điểm đầu là 0, điểm cuối là thực tế điểm đầu tiên (0,0,0)
current_window = np.zeros((1, window_size, 3))
current_window[0, -1, :] = test_traj[0]

step_by_step_preds = []

# 2. VÒNG LẶP DỰ ĐOÁN VÀ CẬP NHẬT
# Chúng ta sẽ dự đoán từ điểm thứ 1 đến điểm thứ 99
for i in range(len(test_traj) - 1):
    # AI dự đoán điểm tiếp theo (i+1) dựa trên cửa sổ hiện tại
    prediction = model.predict(current_window, verbose=0)
    step_by_step_preds.append(prediction[0])

    # CẬP NHẬT ĐIỂM THỰC TẾ:
    # Thay vì dùng 'prediction', ta lấy điểm 'test_traj[i+1]' thực tế để nạp vào cửa sổ
    actual_next_point = test_traj[i+1].reshape(1, 1, 3)
    current_window = np.append(current_window[:, 1:, :], actual_next_point, axis=1)

step_by_step_preds = np.array(step_by_step_preds)

# 3. VẼ BIỂU ĐỒ SO SÁNH 3 TRỤC (Đã sửa để hiển thị ngang)
t_actual = np.arange(len(test_traj))
t_pred = np.arange(1, len(test_traj)) # AI bắt đầu đoán từ điểm thứ 2 (index 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True) # Thay đổi từ (3, 1) thành (1, 3) và sharex=True thành sharey=True
labels = ['X', 'Y', 'Z']

for i in range(3):
    axes[i].plot(t_actual, test_traj[:, i], label='Thực tế ', color='blue', alpha=0.5, linewidth=2)
    axes[i].plot(t_pred, step_by_step_preds[:, i], label='Dự đoán', color='red', linestyle='--', linewidth=1.5)
    axes[i].set_xlabel('Timesteps') # Di chuyển xlabel vào vòng lặp hoặc set chung
    axes[i].set_ylabel(f'Tọa độ {labels[i]}')
    axes[i].legend()
    axes[i].grid(True)

fig.suptitle('GRU quỹ đạo 26', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Điều chỉnh layout để tránh tiêu đề bị che
plt.show()
