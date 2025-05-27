import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from joblib import dump, load
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 读取数据
capacity_path = os.path.join('dataset', 'capacity')  # 假设 capacity 是 .joblib 文件
try:
    capacity = load(capacity_path)
except FileNotFoundError:
    print(f"Error: File {capacity_path} does not exist")
    exit()

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']  # 4 个数据集的名字
# Battery_list = ['B0005'] 
# 绘制容量退化曲线
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
plt.style.use('seaborn-v0_8-whitegrid')

# 自定义颜色
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

# for idx, name in enumerate(Battery_list):
#     data_cycles = capacity[name][0]
#     data_capacity = capacity[name][1]
    
#     # Determine split point for plotting
#     split_idx = int(len(data_cycles) * 0.8)
    
#     # Plot training data part for this battery
#     plt.plot(data_cycles[:split_idx], data_capacity[:split_idx],
#              color=colors[idx],
#              linestyle='-',  # Solid line for train
#              linewidth=1.5,
#              alpha=0.7,
#              label=f'{name} - Train')
             
#     # Plot testing data part for this battery
#     plt.plot(data_cycles[split_idx:], data_capacity[split_idx:],
#              color=colors[idx],
#              linestyle='--', # Dashed line for test
#              linewidth=1.5,
#              alpha=0.7,
#              label=f'{name} - Test')

for idx, name in enumerate(Battery_list):
    data_cycles = capacity[name][0]
    data_capacity = np.array(capacity[name][1])

    # 异常值处理：与前一个点相比，变化幅度大于3σ则删除
    diff = np.diff(data_capacity)
    std_diff = np.std(diff)
    abnormal = np.zeros_like(data_capacity, dtype=bool)
    abnormal[1:] = np.abs(diff) > 3 * std_diff

    # 输出被剔除的异常点
    if np.any(abnormal):
        print(f"{name} removed {np.sum(abnormal)} points as outliers (diff > 3σ):")
        print(list(zip(np.array(data_cycles)[abnormal], data_capacity[abnormal])))

    # 保留正常点
    data_capacity = data_capacity[~abnormal]
    data_cycles = np.array(data_cycles)[~abnormal]

    # Determine split point for plotting
    split_idx = int(len(data_cycles) * 0.8)

    # Plot training data part for this battery
    plt.plot(data_cycles[:split_idx], data_capacity[:split_idx],
             color=colors[idx],
             linestyle='-',  # Solid line for train
             linewidth=1.5,
             alpha=0.7,
             label=f'{name} - Train')

    # Plot testing data part for this battery
    plt.plot(data_cycles[split_idx:], data_capacity[split_idx:],
             color=colors[idx],
             linestyle='--', # Dashed line for test
             linewidth=1.5,
             alpha=0.7,
             label=f'{name} - Test')

plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.title('Capacity Degradation (80% Train, 20% Test per Battery)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout() # Added for better layout
plt.savefig(os.path.join('dataset', 'capacity_degradation.png'), dpi=300)

# 数据预处理
def make_data_labels(x_data, y_label):
    '''
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    x_data = torch.tensor(x_data).float()
    y_label = torch.tensor(y_label).float()
    return x_data, y_label

def data_window_maker(time_series, window_size):
    '''
        参数:
        time_series: 时间序列数据(为numpy数组格式)
        window_size: 滑动窗口大小

        返回:
        data_x: 特征数据
        data_y: 标签数据
    '''
    data_x = []
    data_y = []
    data_len = time_series.shape[0]
    # Ensure there's enough data to create at least one window
    if data_len <= window_size:
        # Return empty arrays if not enough data for a single window + label
        return np.array([]).reshape(0, window_size, time_series.shape[1] if time_series.ndim > 1 else 1), \
               np.array([]).reshape(0, time_series.shape[1] if time_series.ndim > 1 else 1)

    for i in range(data_len - window_size):
        data_x.append(time_series[i:i + window_size, :])  # 输入特征
        data_y.append(time_series[i + window_size, :])  # 输出标签
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x, data_y = make_data_labels(data_x, data_y)
    return data_x, data_y

def make_wind_dataset(data, window_size):
    '''
        参数:
        data: 数据集(为numpy数组格式)
        window_size: 滑动窗口大小

        返回:
        data_x: 特征数据
        data_y: 标签数据
    '''
    if data.size == 0: # Handle empty input data
        # Infer feature dimension for empty tensor creation
        # This assumes data would have at least one feature column if not empty
        # Adjust if data can be 1D (time_series.shape[1] would error)
        num_features = 1 # Default to 1 if cannot infer, or adjust as needed
        if data.ndim > 1 and data.shape[1] > 0:
            num_features = data.shape[1]
        elif train_X.ndim == 3 and train_X.shape[2] > 0 : # Try to infer from global train_X if available
             num_features = train_X.shape[2]


        return torch.empty(0, window_size, num_features), torch.empty(0, num_features)
    data_x, data_y = data_window_maker(data, window_size)
    return data_x, data_y

# 数据归一化和滑动窗口处理
# Splitting each battery's data: 80% for training, 20% for testing
all_train_data_list = []
all_test_data_list = []

for name in Battery_list:
    # capacity[name][1] contains the capacity values (time series data)
    target_capacity_data = capacity[name][1] 
    target_capacity_data = np.array(target_capacity_data).reshape(-1, 1)

    n_samples = len(target_capacity_data)
    split_point = int(n_samples * 0.8)
    
    train_part = target_capacity_data[:split_point]
    test_part = target_capacity_data[split_point:]
    
    # Ensure parts are not empty and have enough data for at least one window later
    # window_size is 1, so len > 1 is needed for data_window_maker to produce output
    if len(train_part) > 1: # Adjusted from > 0 to > window_size
        all_train_data_list.append(train_part)
    if len(test_part) > 1: # Adjusted from > 0 to > window_size
        all_test_data_list.append(test_part)

# Combine all training parts and all testing parts
if not all_train_data_list:
    raise ValueError("No training data was generated. Check battery data, split logic, and window_size requirements.")
train_data = np.concatenate(all_train_data_list, axis=0)

if not all_test_data_list:
    print("Warning: No test data was generated or test data is too short. Test set will be empty or effectively empty after windowing.")
    # Ensure test_data has the same number of columns as train_data for scaler compatibility
    num_features_train = train_data.shape[1] if train_data.ndim > 1 else 1
    test_data = np.array([]).reshape(0, num_features_train)
else:
    test_data = np.concatenate(all_test_data_list, axis=0)


# Normalize the training set
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
dump(scaler, os.path.join('dataset', 'scaler_data'))  # 保存归一化模型

# Apply the same normalization to test set
if test_data.size > 0:
    test_data = scaler.transform(test_data)
else:
    print("Test data is empty, skipping transformation.")


# Sliding window processing
window_size = 1 # Defined earlier, ensure it's consistent

# 处理训练集
# This global variable is used by make_wind_dataset in case of empty data
# It's a bit of a workaround for inferring feature dimensions.
train_X_global_ref = None # Placeholder

train_X, train_Y = make_wind_dataset(train_data, window_size)
train_X_global_ref = train_X # Store a reference

# 处理测试集
if test_data.size > 0:
    test_X, test_Y = make_wind_dataset(test_data, window_size)
else:
    # Create empty tensors if test_data is empty, try to match feature dim from train_X
    num_features = 1
    if train_X_global_ref is not None and train_X_global_ref.ndim == 3 and train_X_global_ref.shape[2] > 0:
        num_features = train_X_global_ref.shape[2]
    elif train_Y.ndim == 2 and train_Y.shape[1] > 0: # Fallback to train_Y features
        num_features = train_Y.shape[1]

    test_X = torch.empty(0, window_size, num_features) 
    test_Y = torch.empty(0, num_features)


# 保存数据集
dump(train_X, os.path.join('dataset', 'train_X'))
dump(train_Y, os.path.join('dataset', 'train_Y'))
dump(test_X, os.path.join('dataset', 'test_X'))
dump(test_Y, os.path.join('dataset', 'test_Y'))

print('Data Shapes:')
print("Training Set X:", train_X.size(), "Training Set Y:", train_Y.size())
print("Test Set X:", test_X.size(), "Test Set Y:", test_Y.size())
