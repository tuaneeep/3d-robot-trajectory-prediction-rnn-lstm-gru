import pandas as pd
import numpy as np

# 1. Đọc dữ liệu từ file CSV
# Cập nhật đường dẫn file để trỏ đến vị trí trên Google Drive
file_path = '/content/drive/MyDrive/robot_trajectory_randomized_movement.csv'
df = pd.read_csv(file_path)

# 2. Chuyển đổi thành Tensor (Numpy Array)
# Mỗi quỹ đạo có 100 điểm, mỗi điểm có 3 tọa độ (x, y, z)
# reshape(-1, 100, 3) sẽ tự động tính số lượng quỹ đạo (12000 / 100 = 120)
trajectories_tensor = df.values.reshape(-1, 100, 3)

# 3. Lưu Tensor vào file .npy để sử dụng sau này
np.save('robot_trajectories.npy', trajectories_tensor)

# 4. Code "Show" để kiểm tra tọa độ
print(f"Cấu trúc Tensor: {trajectories_tensor.shape}")


# Hiển thị quỹ đạo thứ 2 (Index 1)
# Lấy toàn bộ 100 điểm của quỹ đạo này
trajectory_3 = trajectories_tensor[2]

print("Dữ liệu của quỹ đạo thứ 3:")
print(f"Điểm bắt đầu (Point 0): {trajectory_3[0]}")
print(f"Điểm thứ ba   (Point 2): {trajectory_3[2]}")
print(f"Điểm cuối cùng (Point 99): {trajectory_3[-1]}")

# Nếu bạn muốn xem 5 điểm đầu tiên của quỹ đạo thứ 2 theo dạng bảng
print("\n5 điểm đầu tiên của quỹ đạo thứ 3:")
print(trajectory_3[:5])
