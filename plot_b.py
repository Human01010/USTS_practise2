import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# %matplotlib inline # 如果在 Jupyter notebook 中运行，保留此行

from tqdm.notebook import tqdm
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau # 导入学习率调度器

# 转换时间格式，将字符串转换成 datatime 格式
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# 加载 mat 文件
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]

    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


# 提取锂电池参数
def getBatteryFeatures(Battery, feature_choose = 1):
    cycle, capacity, mean_voltage, duration, voltage_slope, mean_current  = [], [], [], [], [], []
    i = 1
    res = [cycle, capacity]
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1

            # 计算电压相关特征
            mean_voltage.append(np.mean(Bat['data']['Voltage_measured']))

            # 时间相关特征
            discharge_time = Bat['data']['Time']
            cur_duration = discharge_time[-1] - discharge_time[0]
            duration.append(cur_duration)

            discharge_current = Bat['data']['Current_measured']
            discharge_voltage = Bat['data']['Voltage_measured']
            voltage_slope.append((discharge_voltage[-1] - discharge_voltage[0]) / len(discharge_voltage))
            
            # 计算电流相关特征
            mean_current.append(np.mean(discharge_current))
    # print(voltage_slope.shape, mean_current.shape, mean_voltage.shape, duration.shape)
    if feature_choose == 2:
        res.append(voltage_slope)
        res.append(mean_current)
    else:
        res.append(mean_voltage)
        res.append(duration)


    return res

# --- 添加数据清洗函数 ---
def clean_capacity_data(battery_data): 
    """
    Cleans battery data by removing cycles with anomalous capacity increases
    using a 3-sigma rule on the differences between consecutive capacity values.
    Assumes battery_data is a list: [cycles, capacities, feature1, feature2, ...]
    where each element is a list or NumPy array.
    """
    cycles_orig = np.array(battery_data[0], dtype=float)
    capacities_orig = np.array(battery_data[1], dtype=float)
    
    if len(cycles_orig) == 0 or len(capacities_orig) == 0:
        # print("Warning: Empty cycles or capacities list.")
        return battery_data
    
    # Ensure consistent lengths for core data for diff calculation
    min_len = min(len(cycles_orig), len(capacities_orig))
    if min_len < len(cycles_orig) or min_len < len(capacities_orig):
        # print(f"Warning: Truncating cycles/capacities to shortest length: {min_len}")
        cycles_orig = cycles_orig[:min_len]
        capacities_orig = capacities_orig[:min_len]

    other_features_orig = [np.array(f, dtype=float)[:min_len] for f in battery_data[2:]]

    if len(capacities_orig) < 3: # Need at least 3 capacity points for 2 differences for std calculation
        return [cycles_orig, capacities_orig] + other_features_orig # Return (potentially truncated) original

    capacity_diffs = np.diff(capacities_orig)
    
    if len(capacity_diffs) < 2: # Need at least 2 differences for a meaningful std
        # print("Not enough capacity differences to apply 3-sigma rule.")
        return [cycles_orig, capacities_orig] + other_features_orig

    mean_diff = np.mean(capacity_diffs)
    std_diff = np.std(capacity_diffs)
    
    safe_std_diff = max(std_diff, 1e-9) 
    # Threshold for identifying an anomalous *increase*
    increase_threshold = mean_diff + 3 * safe_std_diff

    kept_indices = [0] # Always keep the first data point
    for i in range(1, len(capacities_orig)):
        current_increase = capacities_orig[i] - capacities_orig[i-1]
        # If the increase is anomalously large, mark for removal (i.e., don't keep)
        if current_increase > increase_threshold:
            # print(f"Cycle index {i} (Cycle {cycles_orig[i]}): Capacity increase {current_increase:.4f} > threshold {increase_threshold:.4f}. Removing.")
            continue 
        kept_indices.append(i)

    cleaned_cycles = cycles_orig[kept_indices]
    cleaned_capacities = capacities_orig[kept_indices]
    cleaned_other_features = [f[kept_indices] for f in other_features_orig]
    
    # print(f"Data cleaning: Original points: {len(capacities_orig)}, Kept points: {len(cleaned_capacities)}")
    return [cleaned_cycles, cleaned_capacities] + cleaned_other_features


# 获取锂电池充电或放电时的测试数据
def getBatteryValues(Battery, Type='charge'):
    data=[]
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = r'./battery_dataset/'

Battery = {}  # {name: [cycles, capacities, feature1, feature2]}
# feature_choose = int(input('Choose features: 1 for voltage and time, 2 for current and slope: '))
feature_choose = 1 # Default to voltage and time
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    raw_battery_features = getBatteryFeatures(data,feature_choose=feature_choose)              # 放电时的容量数据
    cleaned_battery_features = clean_capacity_data(raw_battery_features)
    # cleaned_battery_features = raw_battery_features

    Battery[name] = cleaned_battery_features

# 定义模型结果文件
model_result_files = {
    'TFT_BiLSTM': 'BiLSTM_transformer_results_b.npy',
    'LSTM': 'LSTM_results_b.npy',
    'CNN_LSTM_Skip': 'CNN_LSTM_Skip_results_b.npy'
}

# 存储所有模型的预测数据
# 结构: {battery_name: {'cycles': [], 'ground_truth': [], 'predictions': {model_name: []}}}
all_plot_data = {}

# 加载各个模型的结果
for model_name, file_name in model_result_files.items():
    if model_name == 'TFT_BiLSTM':
        model_name = 'BiLSTM_transformer'
    file_path = os.path.join(dir_path, file_name) # 假设结果文件也在 dir_path 或根据实际情况修改
    if not os.path.exists(file_path):
        # 如果文件不在 dir_path，尝试当前目录
        file_path = file_name
        if not os.path.exists(file_path):
            print(f"警告: 结果文件 {file_name} 未在 {dir_path} 或当前目录找到。跳过此模型。")
            continue
    
    try:
        # 加载 .npy 文件，allow_pickle=True 是因为数据结构中包含字典
        results_array = np.load(file_path, allow_pickle=True)
        
        for record in results_array:
            battery_name = record['name']
            
            if battery_name not in all_plot_data:
                all_plot_data[battery_name] = {
                    'cycles': record['cycles'],
                    'ground_truth': record['ground_truth'],
                    'predictions': {}
                }
            elif (not np.array_equal(all_plot_data[battery_name]['cycles'], record['cycles']) or
                  not np.array_equal(all_plot_data[battery_name]['ground_truth'], record['ground_truth'])):
                print(f"警告: 电池 {battery_name} 的 'cycles' 或 'ground_truth' 数据在文件 {file_name} 中不一致。将使用首次加载的数据。")

            all_plot_data[battery_name]['predictions'][model_name] = record['prediction']
            
    except Exception as e:
        print(f"加载或处理文件 {file_path} 时出错: {e}")
        continue

# 绘制预测结果
battery_names_to_plot = sorted(list(all_plot_data.keys()))

if not battery_names_to_plot:
    print("没有加载到用于绘图的数据。")
else:
    num_batteries = len(battery_names_to_plot)
    # 设置子图布局，例如每行最多显示2个子图
    ncols = 2 
    nrows = (num_batteries + ncols - 1) // ncols  # 向上取整计算行数
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 5), squeeze=False)
    axes = axes.flatten() # 将axes数组展平，方便索引

    for i, battery_name in enumerate(battery_names_to_plot):
        ax = axes[i]
        data_for_battery = all_plot_data[battery_name]
        
        # 1. 绘制该电池所有周期的完整容量衰减曲线
        if battery_name in Battery:
            full_cycles = Battery[battery_name][0]  # 从 Battery 字典获取完整周期数据
            full_capacities = Battery[battery_name][1] # 从 Battery 字典获取完整容量数据
            ax.plot(full_cycles, full_capacities, label='Full Capacity Degradation', linestyle='-', color='grey', alpha=0.7, linewidth=2)
        else:
            print(f"警告: 电池 {battery_name} 的完整数据未在 Battery 字典中找到。")

        # 2. 获取测试集的数据点
        test_cycles = data_for_battery['cycles']
        test_ground_truth = data_for_battery['ground_truth']
        
        # 绘制测试集上的实际值
        # ax.plot(test_cycles, test_ground_truth, label='Test Set Ground Truth', marker='o', linestyle='-', color='black')
        
        # 3. 绘制各个模型在测试集上的预测值
        model_predictions = data_for_battery['predictions']
        for model_name, prediction in model_predictions.items():
            # 确保预测值与测试集的周期对齐
            ax.plot(test_cycles, prediction, label=model_name, marker='x', linestyle='--')
            
        ax.set_title(f'Battery {battery_name} - Capacity Degradation and Predictions')
        ax.set_xlabel('Cycles')
        ax.set_ylabel('Capacity') 
        ax.legend()
        ax.grid(True)
        
    # 如果子图数量多于电池数量，隐藏多余的子图
    for j in range(num_batteries, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout() # 调整子图布局，防止重叠
    plt.show()