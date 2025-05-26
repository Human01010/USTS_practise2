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
# %matplotlib inline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


# 提取锂电池容量
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]


# 获取锂电池充电或放电时的测试数据
def getBatteryValues(Battery, Type='charge'):
    data=[]
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = 'C:/Users\Timothy\PycharmProjects\pythonProject\Practise2\mission3/NASA\dataset/'

Battery = {}
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    Battery[name] = getBatteryCapacity(data)              # 放电时的容量数据

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018'] # 4 个数据集的名字
Battery = np.load('C:/Users\Timothy\PycharmProjects\pythonProject\Practise2\mission3/NASA\dataset/NASA.npy', allow_pickle=True)
Battery = Battery.item()

fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
c = 0
for name,color in zip(Battery_list, color_list):
    df_result = Battery[name]
    ax.plot(df_result[0], df_result[1], color, label=name)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
plt.legend()

def build_instances(sequence, window_size):
    #sequence: list of capacity
    x, y = [],[]
    for i in range(len(sequence) - window_size):
        features = sequence[i:i+window_size]
        target = sequence[i+window_size]

        x.append(features)
        y.append(target)
        
    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)


def split_dataset(data_sequence, train_ratio=0.0, capacity_threshold=0.0):
    if capacity_threshold > 0:
        max_capacity = max(data_sequence)
        capacity = max_capacity * capacity_threshold
        point = [i for i in range(len(data_sequence)) if data_sequence[i] < capacity]
    else:
        point = int(train_ratio + 1)
        if 0 < train_ratio <= 1:
            point = int(len(data_sequence) * train_ratio)
    train_data, test_data = data_sequence[:point], data_sequence[point:]
    
    return train_data, test_data


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence = data_dict[name][1]
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    train_x, train_y = build_instances(train_data, window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_instances(v[1], window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    
    for i in range(len(y_test)-1):
        if y_test[i] <= threshold >= y_test[i+1]:
            true_re = i - 1
            break
            
    for i in range(len(y_predict)-1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
            
    score = abs(true_re - pred_re)/true_re
    if score > 1: score = 1
        
    return score


def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return rmse
    
    
def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Net(nn.Module):
    def __init__(self, hidden_dim=8, num_layers=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):           # x shape: (batch_size, feature_size, 1)
        out, _ = self.cell(x)       # out shape: (batch_size, feature_size, hidden_dim)
        out = out[:, -1, :]         # 取序列最后一个时间步的输出作为预测  
        out = self.linear(out)      # out shape: (batch_size, 1)
        return out
    
def train(lr, feature_size, hidden_dim=64, num_layers=2, weight_decay=0.0, mode='RNN', epochs=1000, seed=0, device='cpu', metric='rmse'):
    score_list, result_list = [], []
    setup_seed(seed)
    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, feature_size)
        test_sequence = train_data + test_data
        test_x, test_y = build_instances(test_sequence, feature_size)
        # print('sample size: {}'.format(len(train_x)))
        
        model = Net(hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        loss_list, y_ = [0], []
        mae, rmse, re = 1, 1, 1
        score_, score = 1,1
        for epoch in range(epochs):
            X = np.reshape(train_x/Rated_Capacity,(-1, feature_size, 1))   # (batch_size, feature_size, 1)
            y = np.reshape(train_y/Rated_Capacity,(-1,1))          # shape 为 (batch_size, 1)

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output = model(X)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)
            optimizer.zero_grad()            
            loss.backward()                    
            optimizer.step()                   

            if (epoch + 1) % 100 == 0:
                x = np.reshape(test_x/Rated_Capacity,(-1, feature_size, 1))
                x = torch.from_numpy(x).to(device) 
                pred = model(x) 
                point_list = pred.data.cpu().numpy() * Rated_Capacity
                point_list = point_list.reshape(-1)
                
                y_.append(point_list)                                 
                loss_list.append(loss)
                rmse = evaluation(y_test=test_y, y_predict=y_[-1])
                re = relative_error(y_test=test_y, y_predict=y_[-1], threshold=Rated_Capacity*0.7)
                #print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, re))
            
            if metric == 're':
                score = [re]
            else:
                score = [rmse]
                
            if (loss < 1e-3) and (score_[0] < score[0]):
                break
            score_ = score.copy()
            
        score_list.append(score_)
        result_list.append(train_data.copy()[:-1] + list(y_[-1]))
        
    return score_list, result_list

feature_size = 16
epochs = 500
lr = 0.001           # learning rate
hidden_dim = 128
num_layers = 1
weight_decay = 0.0
mode = 'LSTM'        # RNN, LSTM, GRU
Rated_Capacity = 2.0
metric = 'rmse'
device = 'cpu'

SCORE = []
for seed in tqdm(range(4)):
    print('seed: ', seed)
    score_list, _ = train(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, 
                         weight_decay=weight_decay, mode=mode, epochs=epochs, seed=seed, device=device, metric=metric)
    print(np.array(score_list))
    print(metric + ': for this seed: {:<6.4f}'.format(np.mean(np.array(score_list))))
    for s in score_list:
        SCORE.append(s)
    print('------------------------------------------------------------------')
print(metric + ': mean: {:<6.4f}'.format(np.mean(np.array(SCORE))))

# feature_size = 16
# epochs = 500
# lr = 0.001           # learning rate
# hidden_dim = 128
# num_layers = 1
# weight_decay = 0.0
# mode = 'LSTM'        # RNN, LSTM, GRU
# Rated_Capacity = 2.0
# metric = 'rmse'
# device = 'cpu'

# SCORE = []
# print('seed: ', seed)
# score_list, prediction_list = train(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, 
#                                     weight_decay=weight_decay, mode=mode, epochs=epochs, seed=seed, device=device, metric=metric)

# print(np.array(score_list))
# print(metric + ' for this seed: {:<6.4f}'.format(np.mean(np.array(score_list))))

# fig,ax = plt.subplots(2, 2, figsize=(24, 18))

# for i in range(2):
#     for j in range(2):
#         t = i + j
#         battery_name = Battery_list[t]
#         test_data = Battery[battery_name][1]
#         predict_data = prediction_list[t]
#         x = [t for t in range(len(test_data))]
#         threshold = [Rated_Capacity*0.7] * len(test_data)
#         ax[i][j].plot(x, test_data, 'c', label='test data')
#         ax[i][j].plot(x, predict_data, 'b', label='predicted data')
#         ax[i][j].plot(x, threshold, 'black', ls=':', label='stop line')
#         ax[i][j].legend()
#         ax[i][j].set_xlabel('Discharge cycles', fontsize=15)
#         ax[i][j].set_ylabel('Capacity (Ah)', fontsize=15)
#         ax[i][j].set_title('test v.s. prediction of battery ' + battery_name, fontsize=20)
# plt.show()
def predict(model, test_x, feature_size, device, Rated_Capacity, test_data_length):
    model.eval()
    with torch.no_grad():
        x = np.reshape(test_x / Rated_Capacity, (-1, feature_size, 1))
        x = torch.from_numpy(x).to(device)
        pred = model(x)
        point_list = pred.data.cpu().numpy().reshape(-1) * Rated_Capacity
        
        # 确保预测结果与测试数据长度一致
        if len(point_list) > test_data_length:
            point_list = point_list[:test_data_length]
        elif len(point_list) < test_data_length:
            point_list = np.pad(point_list, (0, test_data_length - len(point_list)), 'constant')
        
        return point_list

seed = 0

# 定义模型和训练参数
feature_size = 16
epochs = 500
lr = 0.001           # learning rate
hidden_dim = 128
num_layers = 1
weight_decay = 0.0
mode = 'LSTM'        # RNN, LSTM, GRU
Rated_Capacity = 2.0
metric = 'rmse'
device = 'cpu'

# 训练模型
print('seed: ', seed)
setup_seed(seed)
model = Net(hidden_dim=hidden_dim, num_layers=num_layers, mode=mode).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

# 获取训练和测试数据
train_x, train_y, train_data, test_data = get_train_test(Battery, Battery_list[0], feature_size)
test_sequence = train_data + test_data
test_x, test_y = build_instances(test_sequence, feature_size)

# 训练模型
for epoch in range(epochs):
    model.train()
    X = np.reshape(train_x / Rated_Capacity, (-1, feature_size, 1))
    y = np.reshape(train_y / Rated_Capacity, (-1, 1))
    X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 使用 predict 函数进行预测
prediction_list = predict(model, test_x, feature_size, device, Rated_Capacity, len(test_data))

# # 绘制预测结果
# fig, ax = plt.subplots(2, 2, figsize=(24, 18))
# for i in range(2):
#     for j in range(2):
#         plot_idx = i * 2 + j # Changed variable name from t to avoid confusion if t is used elsewhere
#         battery_name = Battery_list[plot_idx]
#
#         # Original data for the battery
#         original_test_data_for_plot = Battery[battery_name][1]
#         original_x_indices = list(range(len(original_test_data_for_plot)))
#
#         # Predictions for this battery
#         # Ensure prediction_list is correctly indexed if it's a list of lists/arrays
#         current_predict_data = prediction_list[plot_idx]
#
#         # Determine the common length for plotting to avoid ValueError
#         # Based on the error: len(original_x_indices) is 168, len(current_predict_data) is 151
#         common_len = min(len(original_x_indices), len(current_predict_data))
#
#         # Slice all data to the common length
#         x_to_plot = original_x_indices[:common_len]
#         test_data_to_plot = original_test_data_for_plot[:common_len]
#         predict_data_to_plot = current_predict_data[:common_len]
#
#         threshold_to_plot = [Rated_Capacity * 0.7] * common_len # Adjust threshold length as well
#
#         ax[i][j].plot(x_to_plot, test_data_to_plot, 'c', label='test data')
#         ax[i][j].plot(x_to_plot, predict_data_to_plot, 'b', label='predicted data')
#         ax[i][j].plot(x_to_plot, threshold_to_plot, 'black', ls=':', label='stop line')
#         ax[i][j].legend()
#         ax[i][j].set_xlabel('Discharge cycles', fontsize=15)
#         ax[i][j].set_ylabel('Capacity (Ah)', fontsize=15)
#         ax[i][j].set_title('test v.s. prediction of battery ' + battery_name, fontsize=20)
# plt.show()

#提取更多特征并计算PCC
# ------------------------- 特征提取 -------------------------
def extract_features(battery_data):
    """从电池数据中提取多个特征"""
    features = []
    for Bat in battery_data:
        if Bat['type'] == 'discharge':
            # 提取放电容量作为目标变量
            capacity = Bat['data']['Capacity'][0]
            
            # 提取放电阶段的统计特征
            discharge_voltage = Bat['data']['Voltage_measured']
            discharge_current = Bat['data']['Current_measured']
            
            # 计算电压相关特征
            mean_voltage = np.mean(discharge_voltage)
            min_voltage = np.min(discharge_voltage)
            voltage_slope = (discharge_voltage[-1] - discharge_voltage[0]) / len(discharge_voltage)
            
            # 计算电流相关特征
            mean_current = np.mean(discharge_current)
            
            # 时间相关特征
            discharge_time = Bat['data']['Time']
            duration = discharge_time[-1] - discharge_time[0]
            
            features.append({
                'capacity': capacity,
                'mean_voltage': mean_voltage,
                'min_voltage': min_voltage,
                'voltage_slope': voltage_slope,
                'mean_current': mean_current,
                'duration': duration
            })
    return pd.DataFrame(features)

# ------------------------- 数据加载与特征提取 -------------------------
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
Battery_features = {}  # 存储每个电池的特征数据

for name in Battery_list:
    print(f'Processing features for {name}...')
    path = dir_path + name + '.mat'
    raw_data = loadMat(path)
    df_features = extract_features(raw_data)  # 提取特征
    Battery_features[name] = df_features

# 合并所有电池的特征数据
all_features = pd.concat([Battery_features[name] for name in Battery_list], ignore_index=True)

# ------------------------- PCC计算与可视化 -------------------------
# 计算皮尔逊相关系数
correlation_matrix = all_features.corr(method='pearson')

# 提取各特征与容量的相关性
capacity_corr = correlation_matrix['capacity'].sort_values(ascending=False)
print("\n各特征与容量的皮尔逊相关系数：")
print(capacity_corr)

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 12})
plt.title("Feature Correlation Matrix (Pearson)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
