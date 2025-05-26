import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from joblib import dump, load
import torch
from sklearn.preprocessing import StandardScaler

# 读取数据
capacity_path = os.path.join('dataset', 'capacity')
try:
    capacity = load(capacity_path)
except FileNotFoundError:
    print(f"Error: File {capacity_path} does not exist")
    exit()

# 支持多个电池
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']

all_cycles_for_plotting = []
all_cap_for_plotting = [] # Stores cleaned capacity arrays for each battery, used for plotting

plt.figure(figsize=(10, 6))
for battery_name in Battery_list:
    data = capacity[battery_name]
    cycles = np.array(data[0])
    cap = np.array(data[1])

    # 计算相邻点的变化量
    diff = np.diff(cap)
    std_diff = np.std(diff)

    # 标记异常点：与前一个点变化大于3σ
    abnormal = np.zeros_like(cap, dtype=bool)
    if len(diff) > 0: # Ensure diff is not empty
        abnormal[1:] = np.abs(diff) > 3 * std_diff
    else:
        # Handle cases with very few data points where diff might be empty
        pass


    # 输出被剔除的异常点
    if np.any(abnormal):
        print(f"{battery_name} removed {np.sum(abnormal)} points as outliers (diff > 3σ):")
        print(list(zip(cycles[abnormal], cap[abnormal])))

    # 保留正常点
    cap = cap[~abnormal]
    cycles = cycles[~abnormal]

    all_cycles_for_plotting.append(cycles)
    all_cap_for_plotting.append(cap) # Add the cleaned capacity array for this battery
    # 绘制每个电池的容量退化曲线
    if len(cycles) > 0 and len(cap) > 0:
        plt.plot(cycles, cap, marker='o', linestyle='-', linewidth=1.5, alpha=0.8, label=battery_name)

plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.title('Capacity Degradation of Batteries')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join('dataset', 'batteries_capacity.png'), dpi=300)

# --- Create (X,Y) pairs from original sequences BEFORE shuffling ---
all_X_segments = []
all_Y_targets = []
window_size = 1  # Define window size for creating X, Y pairs

# Iterate through the cleaned capacity data for each battery
for single_battery_cap_array in all_cap_for_plotting:
    if len(single_battery_cap_array) > window_size:
        for i in range(len(single_battery_cap_array) - window_size):
            X_segment = single_battery_cap_array[i : i + window_size]
            Y_target = single_battery_cap_array[i + window_size]
            # Reshape X_segment to (window_size, 1 feature)
            all_X_segments.append(X_segment.reshape(window_size, 1))
            # Y_target is scalar, reshape to (1 feature,) for consistent stacking
            all_Y_targets.append(np.array([Y_target]))

if not all_X_segments:
    print("Error: No (X,Y) data pairs were generated. Check input data and window_size.")
    exit()

# Convert lists of pairs to NumPy arrays
master_X = np.array(all_X_segments)  # Shape: (total_num_pairs, window_size, num_features=1)
master_Y = np.array(all_Y_targets)   # Shape: (total_num_pairs, num_features=1)

# --- Dataset splitting (on the correctly formed X,Y pairs) ---
num_samples = master_X.shape[0]
if num_samples == 0:
    print("Error: master_X is empty after pair generation.")
    exit()
    
indices = np.arange(num_samples)
np.random.seed(42) # for reproducibility
np.random.shuffle(indices)

train_end = int(0.6 * num_samples)
val_end = int(0.8 * num_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

train_X_raw = master_X[train_idx]
train_Y_raw = master_Y[train_idx]
val_X_raw = master_X[val_idx]
val_Y_raw = master_Y[val_idx]
test_X_raw = master_X[test_idx]
test_Y_raw = master_Y[test_idx]

# --- Normalization ---
scaler = StandardScaler()
# Fit scaler on training data. Reshape X data to 2D (samples*window_size, features) for scaler.
# Since window_size=1 and features=1, train_X_raw is (num_samples, 1, 1).
# Reshape to (num_samples, 1) for scaler.
scaler.fit(train_X_raw.reshape(-1, 1)) # Fit on the capacity values
dump(scaler, os.path.join('dataset', 'scaler_data'))

# Transform X data
train_X = scaler.transform(train_X_raw.reshape(-1, 1)).reshape(train_X_raw.shape)
val_X = scaler.transform(val_X_raw.reshape(-1, 1)).reshape(val_X_raw.shape)
test_X = scaler.transform(test_X_raw.reshape(-1, 1)).reshape(test_X_raw.shape)

# Transform Y data (as it's also capacity)
train_Y = scaler.transform(train_Y_raw.reshape(-1, 1)).reshape(train_Y_raw.shape)
val_Y = scaler.transform(val_Y_raw.reshape(-1, 1)).reshape(val_Y_raw.shape)
test_Y = scaler.transform(test_Y_raw.reshape(-1, 1)).reshape(test_Y_raw.shape)


# --- Convert to PyTorch Tensors ---
def make_data_labels(x_data, y_label):
    x_data = torch.tensor(x_data).float()
    y_label = torch.tensor(y_label).float()
    return x_data, y_label

train_X, train_Y = make_data_labels(train_X, train_Y)
val_X, val_Y = make_data_labels(val_X, val_Y)
test_X, test_Y = make_data_labels(test_X, test_Y)

# 保存数据集
dump(train_X, os.path.join('dataset', 'train_X'))
dump(train_Y, os.path.join('dataset', 'train_Y'))
dump(val_X, os.path.join('dataset', 'val_X'))
dump(val_Y, os.path.join('dataset', 'val_Y'))
dump(test_X, os.path.join('dataset', 'test_X'))
dump(test_Y, os.path.join('dataset', 'test_Y'))

print('Data Shapes:')
print("Training Set X:", train_X.size(), "Training Set Y:", train_Y.size())
print("Validation Set X:", val_X.size(), "Validation Set Y:", val_Y.size())
print("Test Set X:", test_X.size(), "Test Set Y:", test_Y.size())
