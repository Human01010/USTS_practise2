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

from tqdm.notebook import tqdm
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset # 引入 DataLoader 和 TensorDataset

# 转换时间格式，将字符串转换成 datatime 格式 (保持不变)
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

# 加载 mat 文件 (保持不变)
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
feature_choose = int(input('Choose features: 1 for voltage and time, 2 for current and slope: '))
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    raw_battery_features = getBatteryFeatures(data,feature_choose=feature_choose)              # 放电时的容量数据
    cleaned_battery_features = clean_capacity_data(raw_battery_features)

    Battery[name] = cleaned_battery_features


'''plot figures for capacity degradation'''
fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
c = 0
for name,color in zip(Battery_list, color_list):
    df_result = Battery[name]
    ax.plot(df_result[0], df_result[1], color, label=name)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
plt.legend()

# # plot the feature1 and feature2
# for i in range(2, len(df_result)):
#     fig, ax = plt.subplots(1, figsize=(12, 8))
#     color_list = ['b:', 'g--', 'r-.', 'c.']
#     c = 0
#     for name,color in zip(Battery_list, color_list):
#         df_result = Battery[name]
#         ax.plot(df_result[0], df_result[i], color, label=name)
#     ax.set(xlabel='Discharge cycles', ylabel='Feature' + str(i-1), title='Feature degradation at ambient temperature of 24°C')
#     plt.legend()


# plt.show()


'''
data processing for model training
'''

# build_instances 函数用于构建滑动窗口实例
def build_instances(sequence, window_size):
    x, y = [],[]
    sequence = np.array(sequence) 
    for i in range(len(sequence[0]) - window_size):
        features = sequence[:,i:i+window_size]
        target = sequence[0,i+window_size]

        x.append(features)
        y.append(target)
        
    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)


# get_train_test_a 函数用于在单个电池数据上进行划分
def get_train_test_a(data_dict, name, window_size=8):
    if not isinstance(data_dict.get(name), list) or len(data_dict[name]) < 2:
         print(f"Warning: Data for battery {name} is missing or incomplete.")
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    data_sequence = data_dict[name][1:] # Exclude cycles, only keep [capacities, feature1, feature2]
    
    # 创建 (feature, target) 对
    x_all, y_all = build_instances(data_sequence, window_size)
    x_all = np.transpose(x_all, (0, 2, 1)) # 交换轴 1 和 2
    
    # Check if build_instances returned empty arrays due to insufficient data
    if x_all.shape[0] == 0:
        print(f"Warning: Not enough instances built for battery {name} with window size {window_size}.")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    train_x, temp_x, train_y, temp_y = train_test_split(
        x_all, y_all, test_size=0.4, random_state=42 # random_state 用于可复现性
    )
    
    # 从临时集中分割出验证集和测试集 (各占原始数据的20%，即临时集的50%)
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=42 # 0.5 * 0.4 = 0.2
    )
    
    return train_x, train_y, val_x, val_y, test_x, test_y

# 评估函数 
def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_predict)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_test - y_predict) / (y_test + epsilon))) * 100
    return mae, mse, rmse, r2, mape
    
# 设置随机种子 
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

'''build net'''
# 模型定义 (保持不变，输入特征现在是 window_size, 3)
class Net(nn.Module):
    def __init__(self, input_size=3, hidden_dim=8, num_layers=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        # input_size 应该是每个时间步的特征数，这里是3 (容量+2个特征)
        # 序列长度是 window_size
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_dim=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):           # x shape: (batch_size, window_size, input_size=3)
        out, _ = self.cell(x)       # out shape: (batch_size, window_size, hidden_dim)
        out = out[:, -1, :]         # 取序列最后一个时间步的输出作为预测  (batch_size, hidden_dim)
        out = self.linear(out)      # out shape: (batch_size, 1)
        return out


def cal_feature_target_mean_std(all_train_x, all_train_y):
    feature_means = np.mean(all_train_x, axis=(0, 1)) # shape: (num_features,)
    feature_stds = np.std(all_train_x, axis=(0, 1))   # shape: (num_features,)
    feature_stds = np.maximum(feature_stds, 1e-5) # Avoid division by zero

    # Calculate mean/std across all samples (axis=0) for the target
    target_mean = np.mean(all_train_y) # shape: scalar
    target_std = np.std(all_train_y)   # shape: scalar
    target_std = target_std if target_std > 1e-5 else 1e-5 # Avoid division by zero

    return feature_means, feature_stds, target_mean, target_std


'''train for model - Modified for combined training'''
def train_val_test(lr, window_size, hidden_dim=64, num_layers=2, weight_decay=0.0, mode='RNN', epochs=1000, seed=0, device='cuda'):
    setup_seed(seed)
    batch_size = 128 

    # Collect train, validation, and test data from ALL batteries
    print("Collecting and splitting data from all batteries...")
    all_train_x_list = []
    all_train_y_list = []
    all_val_x_list = []
    all_val_y_list = []
    individual_test_data_list = [] # Store test data for individual battery evaluation later

    for name in Battery_list:
        print(f'Processing data for battery {name}...')
        # get_train_test_a splits data *within* each battery based on percentage
        train_x, train_y, val_x, val_y, test_x, test_y = get_train_test_a(Battery, name, window_size=window_size)

        all_train_x_list.append(train_x)
        all_train_y_list.append(train_y)

        all_val_x_list.append(val_x)
        all_val_y_list.append(val_y)

        # Store test data and related info for later individual evaluation
        individual_test_data_list.append({
            'name': name,
            'test_x': test_x,
            'test_y': test_y,
        })

    
    all_train_x = np.concatenate(all_train_x_list, axis=0)
    all_train_y = np.concatenate(all_train_y_list, axis=0)

    all_val_x = np.concatenate(all_val_x_list, axis=0) if all_val_x_list else np.array([])
    all_val_y = np.concatenate(all_val_y_list, axis=0) if all_val_y_list else np.array([])


    print(f"\nCombined training data shape: X={all_train_x.shape}, y={all_train_y.shape}")
    print(f"Combined validation data shape: X={all_val_x.shape}, y={all_val_y.shape}")
    print(f"Number of batteries with test data: {len(individual_test_data_list)}")


    # Standardize data based on the combined training set 
    print("Calculating standardization statistics from combined training data...")
    feature_means, feature_stds, target_mean, target_std = cal_feature_target_mean_std(all_train_x, all_train_y)

    # Standardize the Combined Training Data
    train_x_scaled = (all_train_x - feature_means) / feature_stds
    train_y_scaled = (all_train_y - target_mean) / target_std

    # Standardize the Combined Validation Data (using training set stats)
    # Only standardize if there is validation data
    val_x_scaled = (all_val_x - feature_means) / feature_stds if all_val_x.shape[0] > 0 else np.array([])
    val_y_scaled = (all_val_y - target_mean) / target_std if all_val_y.shape[0] > 0 else np.array([])


    # Create PyTorch DataLoaders for combined training and validation data
    X_train_tensor = torch.from_numpy(train_x_scaled).float().to(device)
    y_train_tensor = torch.from_numpy(train_y_scaled.reshape(-1, 1)).float().to(device) # Ensure target is (N, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create validation dataloader only if validation data exists
    val_dataloader = None

    X_val_tensor = torch.from_numpy(val_x_scaled).float().to(device)
    y_val_tensor = torch.from_numpy(val_y_scaled.reshape(-1, 1)).float().to(device) # Ensure target is (N, 1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Determine the actual number of features per time step
    actual_input_features = all_train_x.shape[-1] # Last dimension of x_all shape is num_features

    model = Net(input_size=actual_input_features, hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() # Loss is calculated on standardized values

    losses = [] # Store loss for the combined training process
    eval_losses = [] # Store evaluation loss (MSE)
    best_eval_rmse = float('inf') # For potential early stopping (optional)


    model.train() # Set model to training mode
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_dataloader:
            # batch_X shape: (batch_size, window_size, num_features)
            # batch_y shape: (batch_size, 1)
            optimizer.zero_grad()
            output = model(batch_X) # output shape: (batch_size, 1)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        losses.append(avg_loss)

        # Evaluate on validation set periodically
        model.eval() # Set model to evaluation mode
        total_eval_loss = 0
        all_val_preds_scaled = []
        all_val_gt_scaled = []
        with torch.no_grad():
            for eval_batch_X, eval_batch_y in val_dataloader:
                    eval_output = model(eval_batch_X)
                    total_eval_loss += criterion(eval_output, eval_batch_y).item()
                    all_val_preds_scaled.append(eval_output.cpu().numpy())
                    all_val_gt_scaled.append(eval_batch_y.cpu().numpy())

            avg_eval_loss = total_eval_loss / len(val_dataloader)
            eval_losses.append(avg_eval_loss)

            # Calculate metrics on de-standardized data
            all_val_preds_scaled = np.concatenate(all_val_preds_scaled).reshape(-1)
            all_val_gt_scaled = np.concatenate(all_val_gt_scaled).reshape(-1)

            # De-standardize using training set stats!
            val_preds_original = all_val_preds_scaled * target_std + target_mean
            val_gt_original = all_val_gt_scaled * target_std + target_mean
            if (epoch + 1) % 100 == 0 or epoch == 0: 
                # Calculate evaluation metrics
                eval_mae, eval_mse, eval_rmse, eval_r2, eval_mape = evaluation(y_test=val_gt_original, y_predict=val_preds_original)

                print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.6f}, Validation Loss: {avg_eval_loss:.6f}')
                print(f'Validation Metrics: MAE:{eval_mae:<6.4f} | MSE:{eval_mse:<6.6f} | RMSE:{eval_rmse:<6.6f} | R2:{eval_r2:<6.4f} | MAPE:{eval_mape:<6.4f}%')

        model.train() # Set model back to training mode

        # Print training loss every 100 epochs or at the beginning/end
        if (epoch + 1) % 100 == 0:
             print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.6f}')

    print("\n--- Testing trained model on individual battery test sets ---")
    score_list = [] 
    all_test_results = [] 

    model.eval() # Set model to evaluation mode *once* before starting test loop
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for battery_test_data in individual_test_data_list:
            name = battery_test_data['name']
            test_x = battery_test_data['test_x']
            test_y = battery_test_data['test_y']

            print(f'Evaluating on battery test data: {name} (Instances: {test_x.shape[0]})...')

            # Standardize test data using the *training set's* mean and std
            test_x_scaled = (test_x - feature_means) / feature_stds
            X_test_tensor = torch.from_numpy(test_x_scaled).float().to(device)
            pred_test_scaled = model(X_test_tensor) # Predict scaled values
            
            # De-standardize the predictions using the *training set's* target mean and std
            pred_test_numpy_original = pred_test_scaled.data.cpu().numpy().reshape(-1) * target_std + target_mean

            test_y_numpy_original = test_y.reshape(-1)

            test_mae, test_mse, test_rmse, test_r2, test_mape = evaluation(y_test=test_y_numpy_original, y_predict=pred_test_numpy_original)
            print(f'Test results for {name}: MAE:{test_mae:<6.4f} | MSE:{test_mse:<6.6f} | RMSE:{test_rmse:<6.6f} | R2:{test_r2:<6.4f} | MAPE:{test_mape:<6.4f}%')

            score_list.append([name, test_mae, test_mse, test_rmse, test_r2, test_mape]) # Store name and scores
            all_test_results.append({ # Store results for potential plotting
                'name': name,
                'ground_truth': test_y_numpy_original,
                'prediction': pred_test_numpy_original
            })

    # Return scores, combined training loss, and individual test results
    return score_list, [losses], all_test_results, eval_losses, all_train_y_list, all_val_y_list    

'''setting and training for overall performance'''

window_size = 16 # This is the sequence length (feature_size in original code comment)
epochs = 500
lr = 0.001           # learning rate
hidden_dim = 128
num_layers = 1
weight_decay = 0.0
mode = 'LSTM'        # RNN, LSTM, GRU
Rated_Capacity = 2.0 # Note: Rated_Capacity is no longer used for normalization in train_val_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

seed = 1

# Call the modified train_val_test function
score_list , loss_list, all_test_results, eval_losses, all_train_y_list, all_val_y_list = train_val_test(
    lr=lr, 
    window_size=window_size, # Pass window_size
    hidden_dim=hidden_dim, num_layers=num_layers, 
    weight_decay=weight_decay, mode=mode, epochs=epochs, seed=seed, device=device
)


# Extract numerical scores for averaging
numerical_scores = np.array([score[1:] for score in score_list]) # Exclude battery name
avg_scores = np.mean(numerical_scores, axis=0)

for score in score_list:
    name, mae, mse, rmse, r2, mape = score
    print(f"Scores for {name}: MAE:{mae:<6.4f} | MSE:{mse:<6.6f} | RMSE:{rmse:<6.6f} | R2:{r2:<6.4f} | MAPE:{mape:<6.4f}%")

print(f"\nAverage Scores (across batteries): MAE:{avg_scores[0]:<6.4f} | MSE:{avg_scores[1]:<6.6f} | RMSE:{avg_scores[2]:<6.6f} | R2:{avg_scores[3]:<6.4f} | MAPE:{avg_scores[4]:<6.4f}%")



# 可视化 loss_list (
print("\n--- Plotting Training Loss (Combined Data) ---")

if loss_list and loss_list[0]:
    combined_losses = loss_list[0] # loss_list is a list containing one list of losses
    plt.figure(figsize=(10, 6))
    plt.plot(combined_losses, label='Training Loss (Combined Data)')
    plt.title('Training Loss on Combined Battery Data')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No training loss data to plot.")

# 可视化 evaluation loss
print("\n--- Plotting Evaluation Loss (Combined Data) ---")
if eval_losses:
    plt.figure(figsize=(10, 6))
    # Assuming evaluation happens every 100 epochs + epoch 0
    eval_epochs = [0] + list(range(99, epochs, 100)) # Epoch numbers where evaluation occurred
    # Adjust x-axis labels if evaluation frequency is different
    plt.plot(eval_losses, label='Evaluation Loss (Combined Data)', color='orange')
    plt.title('Evaluation Loss on Combined Battery Data')
    plt.xlabel(f'Evaluation Point') # Adjusted label
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
     print("No evaluation loss data to plot.")