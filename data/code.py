import numpy as np
import pandas as pd
import os

# Tạo thư mục data nếu chưa có
os.makedirs('data', exist_ok=True)

# Sinh dữ liệu
np.random.seed(42)
num_samples = 12000
steps = np.random.normal(loc=0, scale=0.01, size=(num_samples, 3))
trajectory = np.cumsum(steps, axis=0)

# Lưu file
df = pd.DataFrame(trajectory, columns=['x', 'y', 'z'])
df.to_csv('data/robot_trajectory_randomized_movement.csv', index=False)
print("File đã được tạo tại data/robot_trajectory_randomized_movement.csv")
