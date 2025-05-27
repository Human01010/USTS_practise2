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

# 绘制容量退化曲线
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
plt.style.use('seaborn-v0_8-whitegrid')

# 自定义颜色和标记
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
marker_list = ['x', '+', '.', '*']

# 异常值处理：与前一个点相比，变化幅度大于3σ就删除
for idx, name in enumerate(Battery_list):
    data = capacity[name]
    cycles = np.array(data[0])
    cap = np.array(data[1])

    diff = np.diff(cap)
    std_diff = np.std(diff)
    abnormal = np.zeros_like(cap, dtype=bool)
    abnormal[1:] = np.abs(diff) > 3 * std_diff

    # 输出被剔除的异常点
    if np.any(abnormal):
        print(f"{name} removed {np.sum(abnormal)} points as outliers (diff > 3σ):")
        print(list(zip(cycles[abnormal], cap[abnormal])))

    # 保留正常点
    cap = cap[~abnormal]
    cycles = cycles[~abnormal]

    # 更新capacity中的数据，后续数据集划分用去除异常值后的数据
    capacity[name] = (cycles, cap)

    # 绘制容量退化曲线
    plt.plot(cycles, cap,
             marker=marker_list[idx],
             color=colors[idx],
             markersize=8,
             linestyle='-',
             linewidth=1.5,
             alpha=0.7,
             label=name)

plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.title('Capacity Degradation at Ambient Temperature of 24°C')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
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
    data_x, data_y = data_window_maker(data, window_size)
    return data_x, data_y

# ================== 数据集划分 ==================
# B0005全部作为训练集
train_data_list = []
b5_data = np.array(capacity['B0005'][1]).reshape(-1, 1)
train_data_list.append(b5_data)

# 其他电池前10%作为训练，后90%作为测试
test_data_list = []
for name in ['B0006', 'B0007', 'B0018']:
    data = np.array(capacity[name][1]).reshape(-1, 1)
    split_idx = int(len(data) * 0.1)
    # 前10%加入训练集
    train_data_list.append(data[:split_idx])
    # 后90%加入测试集
    test_data_list.append(data[split_idx:])

# 合并训练集和测试集
train_data = np.concatenate(train_data_list, axis=0)
test_data = np.concatenate(test_data_list, axis=0)

# 归一化
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
dump(scaler, os.path.join('dataset', 'scaler_data'))  # 保存归一化模型

# 测试集归一化
test_data = scaler.transform(test_data)

# 滑动窗口
window_size = 1

# 处理训练集
train_X, train_Y = make_wind_dataset(train_data, window_size)

# 处理测试集
test_X, test_Y = make_wind_dataset(test_data, window_size)

# 保存数据集
dump(train_X, os.path.join('dataset', 'train_X'))
dump(train_Y, os.path.join('dataset', 'train_Y'))
dump(test_X, os.path.join('dataset', 'test_X'))
dump(test_Y, os.path.join('dataset', 'test_Y'))

print('Data Shapes:')
print("Training Set:", train_X.size(), train_Y.size())
print("Test Set:", test_X.size(), test_Y.size())