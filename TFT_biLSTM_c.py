# 代码： import numpy as np
# ... (保留所有之前的导入)
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

from tqdm.notebook import tqdm
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        # res.append(mean_voltage)
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
feature_choose = 1

for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    raw_battery_features = getBatteryFeatures(data,feature_choose=feature_choose)              # 放电时的容量数据

    # --- 添加清洗步骤 ---
    # print(f"Cleaning data for {name}...")
    cleaned_battery_features = clean_capacity_data(raw_battery_features)
    # cleaned_battery_features = raw_battery_features
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


'''
data processing for model training
'''

def build_instances(sequence, window_size):
    #sequence: list of capacity
    x, y = [],[]
    sequence = np.array(sequence) 
    for i in range(len(sequence[0]) - window_size):
        features = sequence[:,i:i+window_size]
        target = sequence[0,i+window_size]

        x.append(features)
        y.append(target)
    
    x = np.transpose(np.array(x), (0, 2, 1))
        
    return x.astype(np.float32), np.array(y).astype(np.float32)


# B0005(对应i=0)数据作为训练数据，其他电池中前10%的数据用作训练，后90%数据作为测试
def get_train_test_c(data_dict, window_size=8):
    all_train_x, all_train_y = [], []
    all_test_x, all_test_y = [], []
    all_test_cycles = []
    test_battery_lengths = []
    

    for i, name in enumerate(Battery_list):
        data_sequence = data_dict[name][1:] # 忽略cycle数据，只取特征和容量
        original_cycles = data_dict[name][0]
        
        # 将列表数据转换为NumPy数组，以便于切片
        data_sequence_np = np.array(data_sequence)

        if name == 'B0005':
            # B0005的全部数据用于训练
            x_instances, y_instances = build_instances(data_sequence_np, window_size)
            if len(x_instances) > 0:
                all_train_x.append(x_instances)
                all_train_y.append(y_instances)
        else:
            # 其他电池的数据
            # 直接在 build_instances 之后的结果上分割
            x_instances, y_instances = build_instances(data_sequence_np, window_size)
            
            if len(x_instances) > 0:
                num_instances = len(x_instances)
                split_idx = int(0.1 * num_instances)

                train_x_battery = x_instances[:split_idx]
                train_y_battery = y_instances[:split_idx]
                
                test_x_battery = x_instances[split_idx:]
                test_y_battery = y_instances[split_idx:]

                y_all_cycles = original_cycles[window_size:]
                test_cycles = y_all_cycles[split_idx:] # Store test cycles

                if len(train_x_battery) > 0:
                    all_train_x.append(train_x_battery)
                    all_train_y.append(train_y_battery)
                if len(test_x_battery) > 0:
                    all_test_x.append(test_x_battery)
                    all_test_y.append(test_y_battery)
                    all_test_cycles.append(test_cycles)
                    test_battery_lengths.append(len(test_y_battery))

    # 合并所有电池的数据
    final_train_x = np.concatenate(all_train_x, axis=0) if len(all_train_x) > 0 else np.array([])
    final_train_y = np.concatenate(all_train_y, axis=0) if len(all_train_y) > 0 else np.array([])
    final_test_x = np.concatenate(all_test_x, axis=0) if len(all_test_x) > 0 else np.array([])
    final_test_y = np.concatenate(all_test_y, axis=0) if len(all_test_y) > 0 else np.array([])
    final_test_cycles = np.concatenate(all_test_cycles, axis=0) if len(all_test_cycles) > 0 else np.array([])


    return final_train_x, final_train_y, final_test_x, final_test_y, final_test_cycles, test_battery_lengths

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_predict)
    mape = np.mean(np.abs((y_test - y_predict) / y_test)) * 100
    return mae, mse, rmse, r2, mape
    


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


'''build net - Simplified TFT with LSTM Feature Extraction'''

class LSTM_TFT(nn.Module):
    def __init__(self, input_features: int, lstm_hidden_size: int, lstm_num_layers: int,
                 tft_hidden_size: int, tft_num_heads: int, tft_num_layers: int, output_dim: int = 1):
        """
        TFT-inspired model with an LSTM block for initial feature extraction.
        LSTM processes (batch, seq_len, features) -> (batch, seq_len, lstm_features).
        Transformer processes (batch, seq_len, lstm_features) -> prediction.
        Expects standardized inputs.
        """
        super(LSTM_TFT, self).__init__()

        self.input_features = input_features
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.tft_hidden_size = tft_hidden_size
        self.tft_num_heads = tft_num_heads
        self.tft_num_layers = tft_num_layers
        self.output_dim = output_dim

        # --- LSTM Feature Extraction Block ---
        # LSTM processes (batch, seq_len, input_size) and outputs (batch, seq_len, hidden_size * num_directions)
        self.lstm_block = nn.LSTM(
            input_size=input_features, # Input is the original features per time step
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True, # Input/output tensors are (batch, seq, feature)
            # 修改点 1: 将 bidirectional 设置为 True
            bidirectional=True
        )
        # Output shape after Bidirectional LSTM: (batch_size, sequence_length, lstm_hidden_size * 2)

        # --- TFT Core (using LSTM output features as input) ---
        # 输入嵌入层现在需要接收双向 LSTM 的输出维度
        self.input_embedding = nn.Linear(lstm_hidden_size * 2, tft_hidden_size)

        # Transformer Encoder Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=tft_hidden_size,
            nhead=tft_num_heads,
            dim_feedforward=tft_hidden_size * 4,
            activation='gelu',
            batch_first=True
        )

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=tft_num_layers)

        # Output layer
        self.output_layer = nn.Linear(tft_hidden_size, output_dim)


    def forward(self, x):
        """
        Forward pass for the LSTM-TFT model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_features).
                              This represents the history window, expected to be standardized.
        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_dim). This output is
                          the standardized predicted value.
        """
        # x shape: (batch_size, sequence_length, input_features)

        # 1. Apply LSTM Feature Extraction
        # lstm_out shape: (batch_size, sequence_length, lstm_hidden_size * 2) due to bidirectional=True
        # hn, cn are the final hidden/cell states, we usually don't need them for sequence-to-sequence prediction
        lstm_out, (hn, cn) = self.lstm_block(x)

        # 2. Input Embedding (on LSTM output)
        # embedded_x shape: (batch_size, sequence_length, tft_hidden_size)
        # This layer correctly takes the 2 * lstm_hidden_size input from lstm_out
        embedded_x = self.input_embedding(lstm_out)

        # 3. Temporal Processing via Transformer Encoder
        # encoded_output shape: (batch_size, sequence_length, tft_hidden_size)
        encoded_output = self.transformer_encoder(embedded_x)

        # 4. Get representation for prediction (last time step)
        # last_step_output shape: (batch_size, tft_hidden_size)
        last_step_output = encoded_output[:, -1, :]

        # 5. Output Layer
        # out shape: (batch_size, output_dim). This is the standardized prediction.
        out = self.output_layer(last_step_output)

        return out


'''train for model'''
# Added LSTM specific parameters to the function signature, removed CNN parameters
def train_val_test(lr, window_size, lstm_hidden_size, lstm_num_layers,
                   tft_hidden_size, tft_num_heads, tft_num_layers,
                   weight_decay=0.0, epochs=1000, seed=0, device='cuda'):

    score_list = []
    all_test_results = []

    setup_seed(seed)
    batch_size = 64 #8

    actual_input_features = len(Battery[Battery_list[0]]) - 1

    # feature_size is now window_size
    train_x, train_y, test_x, test_y, test_cycles, test_battery_lengths = get_train_test_c(Battery, window_size=window_size)

    print('test battery lengths:', test_battery_lengths)  

    feature_means = np.mean(train_x, axis=(0, 1))
    feature_stds = np.std(train_x, axis=(0, 1))
    # Add a small epsilon to std dev to prevent division by zero for constant features
    feature_stds = np.maximum(feature_stds, 1e-5)

    # For target (train_y), mean/stddev across all samples
    target_mean = np.mean(train_y)
    target_std = np.std(train_y)
    target_std = target_std if target_std > 1e-5 else 1e-5


    # Apply standardization using training stats
    train_x_scaled = (train_x - feature_means) / feature_stds
    train_y_scaled = (train_y - target_mean) / target_std

    test_x_scaled = (test_x - feature_means) / feature_stds # Use train stats for test data

    print('train_x_scaled shape:', train_x_scaled.shape)
    print('train_y_scaled shape:', train_y_scaled.shape)
    print('test_x_scaled shape:', test_x_scaled.shape)


    # Convert scaled data to PyTorch tensors
    X_train_tensor = torch.from_numpy(train_x_scaled).to(device)
    y_train_tensor = torch.from_numpy(train_y_scaled.reshape(-1, 1)).to(device)

    X_test_tensor = torch.from_numpy(test_x_scaled).to(device)


    # Initialize the LSTM_TFT model
    model = LSTM_TFT(
        input_features=actual_input_features, # Original number of features per time step
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        tft_hidden_size=tft_hidden_size,
        tft_num_heads=tft_num_heads,
        tft_num_layers=tft_num_layers,
        output_dim=1 # Predicting a single value (capacity)
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() # Loss is calculated on standardized values
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    losses = []

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train() # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss) # 将当前epoch的平均损失传递给调度器

        if (epoch + 1) % 100 == 0 or epoch == 1: 
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Current LR: {current_lr:.7f}')
   
    print("--- Training complete ---")

    # 训练结束后，在测试集上评估最终模型
    model.eval() 
    with torch.no_grad():
        pred_test_scaled = model(X_test_tensor)  # Model predicts standardized values
        pred_test_numpy_original = pred_test_scaled.data.cpu().numpy().reshape(-1) * target_std + target_mean

    test_y_numpy_original = test_y.reshape(-1)



    cur = 0
    for i in range(3):
        name = Battery_list[i+1]
        length = test_battery_lengths[i]
        # Extract the cycles and predictions for this battery
        test_cycles_for_battery = test_cycles[cur:cur+length]
        pred_test_for_battery = pred_test_numpy_original[cur:cur+length]
        test_y_for_battery = test_y_numpy_original[cur:cur+length]

        test_mae, test_mse, test_rmse, test_r2, test_mape = evaluation(y_test=test_y_for_battery, y_predict=pred_test_for_battery)
        # print(f'Test results for {name}: MAE:{test_mae:<6.4f} | MSE:{test_mse:<6.6f} | RMSE:{test_rmse:<6.6f} | R2:{test_r2:<6.4f} | MAPE:{test_mape:<6.4f}%')
        score_list.append([test_mae, test_mse, test_rmse, test_r2, test_mape])

        cur += length

        # Store test results for visualization
        all_test_results.append({
            'name': name,
            'cycles': np.array(test_cycles_for_battery),
            'ground_truth': np.array(test_y_for_battery),
            'prediction': np.array(pred_test_for_battery)
        })

    return score_list, [losses], all_test_results


'''setting and training for overall performance'''
window_size = 16
epochs = 600
lr = 0.0002
weight_decay = 0.0

# Hyperparameters for the LSTM block (replaces CNN params)
lstm_hidden_size = 64 # Hidden size of the LSTM layers
lstm_num_layers = 1   # Number of LSTM layers

# Hyperparameters for the TFT (Transformer) core
tft_hidden_size = 128 # Dimension of the hidden layers in the transformer
tft_num_heads = 4     # Number of attention heads in the transformer
tft_num_layers = 2    # Number of transformer encoder layers


metric = 'rmse' # Unused variable, kept for context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

seed = 1

# Call the training function with new LSTM and TFT parameters
score_list ,loss_list, all_test_results = train_val_test(
    lr=lr,
    window_size=window_size,
    lstm_hidden_size=lstm_hidden_size,
    lstm_num_layers=lstm_num_layers,
    tft_hidden_size=tft_hidden_size,
    tft_num_heads=tft_num_heads,
    tft_num_layers=tft_num_layers,
    weight_decay=weight_decay,
    epochs=epochs,
    seed=seed,
    device=device
)
np.save("BiLSTM_Transformer_results_c.npy", all_test_results)

print("\n--- Overall Test Scores ---")
print(score_list, Battery_list)
for i, name in enumerate(Battery_list):
    if i != 0:
        print(f"Scores for {name}: MAE:{score_list[i-1][0]:<6.4f} | MSE:{score_list[i-1][1]:<6.4f} | RMSE:{score_list[i-1][2]:<6.4f} | R2:{score_list[i-1][3]:<6.4f} | MAPE:{score_list[i-1][4]:<6.4f}%")
# calculate average scores
avg_scores = np.mean(score_list, axis=0)
print(f"Average Scores: MAE:{avg_scores[0]:<6.4f} | MSE:{avg_scores[1]:<6.4f} | RMSE:{avg_scores[2]:<6.4f} | R2:{avg_scores[3]:<6.4f} | MAPE:{avg_scores[4]:<6.4f}%")


# plot train loss
plt.figure(figsize=(12, 6))
plt.plot(loss_list[0], label='Train Loss')
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()


# --- 添加对测试电池测试集上的预测可视化 ---
print("\n--- Plotting Predictions vs Ground Truth on Test Sets ---")

# Filter out batteries that had no test data
plotable_results = [res for res in all_test_results if len(res['cycles']) > 0]


# Determine grid size based on the number of batteries with test results
num_plots = len(plotable_results)
rows = math.ceil(num_plots / 2) if num_plots > 0 else 0
cols = 2 if num_plots > 1 else 1

# Adjust figure size based on the number of rows and columns
fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6)) # Adjust figure size

# 检查axs是单个Axes对象还是Axes数组
if num_plots == 1:
    axes_to_plot = [axs]
else:
    axes_to_plot = axs.flatten()

for i, result in enumerate(plotable_results):
    ax = axes_to_plot[i] # Get the current subplot

    battery_name = result['name'] # 获取电池名称
    # 从全局的 Battery 字典中获取该电池的所有周期和容量数据
    full_cycles = Battery[battery_name][0]
    full_capacities = Battery[battery_name][1]

    # Plot Ground Truth (使用所有周期的容量数据进行绘制)
    ax.plot(full_cycles, full_capacities, label='Ground Truth (All Cycles)', marker='o', linestyle='-', markersize=4)

    # Plot Prediction (预测数据仍然只存在于测试周期)
    ax.plot(result['cycles'], result['prediction'], label='Prediction (Test Cycles)', marker='o', linestyle='-', markersize=4)

    ax.set_title(f'Battery {result["name"]} Test Prediction')
    ax.set_xlabel('Discharge cycles')
    ax.set_ylabel('Capacity (Ah)')
    ax.legend()
    ax.grid(True)

# Hide any unused subplots if fewer batteries had test data than available subplots
if num_plots > 0: # Only execute if there was at least one plotable result
    for j in range(num_plots, len(axes_to_plot)):
         fig.delaxes(axes_to_plot[j]) # Use axes_to_plot for consistent indexing

plt.tight_layout() # Adjust layout
plt.show()
