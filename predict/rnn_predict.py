# --- CODE TÍNH SAI SỐ ---

# 1. Lấy các giá trị thực tế tương ứng (từ index 1 đến hết)
actual_values = test_traj[1:]

# 2. Tính toán MSE (Sai số bình phương trung bình)
mse = np.mean((actual_values - step_by_step_preds)**2)

# 3. Tính toán MAE (Sai số tuyệt đối trung bình)
mae = np.mean(np.abs(actual_values - step_by_step_preds))

# 4. Tính thêm sai số riêng cho từng trục X, Y, Z (nếu cần)
mse_per_axis = np.mean((actual_values - step_by_step_preds)**2, axis=0)

print("-" * 30)
print(f"Kết quả sai số trên Quỹ đạo 26:")
print(f"Sai số MSE (Tổng hợp): {mse:.8f}")
print(f"Sai số MAE (Tổng hợp): {mae:.8f}")
print("-" * 30)
print(f"MSE Trục X: {mse_per_axis[0]:.8f}")
print(f"MSE Trục Y: {mse_per_axis[1]:.8f}")
print(f"MSE Trục Z: {mse_per_axis[2]:.8f}")
