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

# Load Battery Data
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = r'./battery_dataset/'

Battery = {}  # {name: [cycles, capacities, feature1, feature2]}
feature_choose = 1 # Default to voltage and time features
# feature_choose = int(input('Choose features: 1 for voltage and time, 2 for current and slope: ')) # Uncomment to allow user input

for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    battery_features = getBatteryFeatures(data, feature_choose=feature_choose)
    raw_battery_features = getBatteryFeatures(data,feature_choose=feature_choose)              # 放电时的容量数据
    cleaned_battery_features = clean_capacity_data(raw_battery_features)

    Battery[name] = cleaned_battery_features


'''plot figures for capacity degradation'''
# Keep the original capacity degradation plot as it's useful
fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
c = 0
for name,color in zip(Battery_list, color_list):
    df_result = Battery[name]
    if df_result and len(df_result) > 1 and len(df_result[0]) > 0: # Ensure data exists
        ax.plot(df_result[0], df_result[1], color, label=name)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
plt.legend()
plt.show() # Add plt.show() to display this plot immediately


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
        
    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)


# Modified to return train/test splits for a single battery AND test cycles
def get_train_test_b(data_dict, name, window_size=8):
    # data_sequence contains [capacities, feature1, feature2, ...]

    data_sequence = data_dict[name][1:] 
    original_cycles = data_dict[name][0] 

    # Create (feature, target) pairs using build_instances
    # x_all shape: (num_instances, num_features_per_cycle, window_size)
    # y_all shape: (num_instances,)
    x_all, y_all = build_instances(data_sequence, window_size)

    # Get the cycles corresponding to the y_all targets
    # The target y_all[i] corresponds to the capacity at cycle original_cycles[i + window_size]
    if len(original_cycles) > window_size:
        y_all_cycles = original_cycles[window_size : window_size + len(y_all)]
    else:
         y_all_cycles = np.array([]) # Should be empty if y_all is empty

    if len(x_all) == 0: 
         return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


    # --- Transpose x_all to match expected RNN input shape (batch, seq_len, input_size) ---
    x_all = np.transpose(x_all, (0, 2, 1)) # Shape: (num_instances, window_size, num_features_per_cycle)


    # Calculate split index (e.g., 80% for training)
    split_idx = int(len(x_all) * 0.8)

    # Split data
    train_x = x_all[:split_idx]
    train_y = y_all[:split_idx]

    test_x = x_all[split_idx:]
    test_y = y_all[split_idx:]
    test_cycles = y_all_cycles[split_idx:] # Split corresponding cycles


    # Ensure even if splits are empty, NumPy arrays are returned
    if len(train_x) == 0:
        train_x, train_y = np.array([]), np.array([])
    if len(test_x) == 0:
        test_x, test_y, test_cycles = np.array([]), np.array([]), np.array([])

    print(f"Battery {name}: Train instances: {len(train_x)}, Test instances: {len(test_x)}")

    return train_x, train_y, test_x, test_y , test_cycles


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


'''build net - CNN-LSTM-SKIP Model'''
class CNN_LSTM_SKIP(nn.Module):
    def __init__(self, input_features: int, window_size: int,
                 cnn_channels: int, cnn_kernel_size: int, cnn_pool_kernel_size: int,
                 lstm_hidden_size: int, lstm_num_layers: int, output_dim: int = 1):
        """
        CNN-LSTM model with a skip connection from the original input to the output.
        Expects standardized inputs of shape (batch_size, window_size, input_features).
        """
        super(CNN_LSTM_SKIP, self).__init__()

        self.input_features = input_features
        self.window_size = window_size
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_pool_kernel_size = cnn_pool_kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.output_dim = output_dim

        # --- CNN Block ---
        # Conv1d expects input shape (batch, channels, seq_len)
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=cnn_channels, 
                      kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2), # Use padding to potentially preserve sequence length before pooling
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=cnn_pool_kernel_size, stride=cnn_pool_kernel_size) # Use stride = kernel_size for non-overlapping pooling
        )
        
        # Need to handle cases where window_size is smaller than kernel or pool size
        conv_out_len = window_size # assuming padding preserves length
        if conv_out_len < cnn_pool_kernel_size:
            self.lstm_seq_len = 1 # Pool reduces to size 1 if input is smaller than kernel
        else:
            self.lstm_seq_len = math.floor((conv_out_len - cnn_pool_kernel_size) / cnn_pool_kernel_size) + 1
            
        # --- LSTM Block ---
        # LSTM expects input shape (batch, seq_len, features)
        # The features for LSTM are the output channels from CNN after pooling
        self.lstm_block = nn.LSTM(
            input_size=cnn_channels, # Features per time step for LSTM
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True, # Input/output tensors are (batch, seq, feature)
            # bidirectional=True # Optional: can make it bidirectional
            bidirectional=False # Starting with unidirectional
        )
        # If bidirectional is True, lstm_out_features = lstm_hidden_size * 2
        lstm_out_features = lstm_hidden_size * (2 if self.lstm_block.bidirectional else 1)


        # --- Skip Connection Linear Layer ---
        self.skip_linear = nn.Linear(input_features, lstm_out_features)


        # --- Output Layer ---
        # Takes the combined features from LSTM and Skip Connection
        self.output_layer = nn.Linear(lstm_out_features, output_dim)


    def forward(self, x_original):
        """
        Forward pass for the CNN-LSTM-SKIP model.
        Args:
            x_original (torch.Tensor): Input tensor of shape (batch_size, window_size, input_features).
                              This represents the history window, expected to be standardized.
        Returns:
            torch.Tensor: Predicted output of shape (batch_size, output_dim). This output is
                          the standardized predicted value.
        """
        # x_original shape: (batch_size, window_size, input_features)
        # Conv1d requires (batch, channels, seq_len), so transpose x_original
        cnn_input = x_original.permute(0, 2, 1)
        
        # cnn_out shape: (batch_size, cnn_channels, self.lstm_seq_len)
        cnn_out = self.cnn_block(cnn_input)

        # LSTM requires (batch, seq_len, features), so transpose cnn_out
        # lstm_input shape: (batch_size, self.lstm_seq_len, cnn_channels)
        lstm_input = cnn_out.permute(0, 2, 1)

        # lstm_out shape: (batch_size, self.lstm_seq_len, lstm_out_features)
        lstm_out, (hn, cn) = self.lstm_block(lstm_input)

        # 4. Get representation for prediction (last time step of LSTM output)
        # lstm_last_step shape: (batch_size, lstm_out_features)
        lstm_last_step = lstm_out[:, -1, :]

        # Get the features from the last time step of the original input
        original_last_step = x_original[:, -1, :]  # shape: (batch_size, input_features)
        
        skip_out = self.skip_linear(original_last_step) # (batch_size, lstm_out_features)

        # 6. Combine LSTM output and Skip Connection output
        # combined shape: (batch_size, lstm_out_features)
        combined = lstm_last_step + skip_out

        out = self.output_layer(combined)

        return out


'''train for model'''
def train_val_test(lr, window_size, hidden_dim=64, num_layers=2, weight_decay=0.0, mode='RNN', epochs=1000, seed=0, device='cuda'):
    """
    Trains a single model on combined training data from all batteries and evaluates on individual test sets.
    """
    setup_seed(seed)
    batch_size = 64 # Define batch size for DataLoader

    all_train_x_list = []
    all_train_y_list = []
    individual_test_data_list = []

    # --- 1. Prepare Data: Split each battery and collect training data ---
    print("\n--- Preparing data: Splitting batteries and collecting training data ---")
    for name in Battery_list:
        train_x, train_y, test_x, test_y, test_cycles = get_train_test_b(Battery, name, window_size=window_size)

        all_train_x_list.append(train_x)
        all_train_y_list.append(train_y)

        individual_test_data_list.append({
            'name': name,
            'test_x': test_x, # shape: (N_test_battery, window_size, num_features_per_cycle)
            'test_y': test_y, # shape: (N_test_battery,)
            'test_cycles': test_cycles # shape: (N_test_battery,)
        })

    all_train_x = np.concatenate(all_train_x_list, axis=0) #shape: (Total_train_samples, window_size, num_features_per_cycle)
    all_train_y = np.concatenate(all_train_y_list, axis=0)

    print(f"\nCombined training data shape: X={all_train_x.shape}, y={all_train_y.shape}")

    actual_input_features = all_train_x.shape[-1]



    # -- Standardize Data: Calculate stats on combined training data ---
    feature_means = np.mean(all_train_x, axis=(0, 1))
    feature_stds = np.std(all_train_x, axis=(0, 1))
    # Add a small epsilon to std dev to prevent division by zero for constant features
    feature_stds = np.maximum(feature_stds, 1e-6)

    target_mean = np.mean(all_train_y)
    target_std = np.std(all_train_y)
    target_std = target_std if target_std > 1e-6 else 1e-6 # Prevent division by zero

    # Standardize the Combined Training Data
    train_x_scaled = (all_train_x - feature_means) / feature_stds
    train_y_scaled = (all_train_y - target_mean) / target_std

    X_train_tensor = torch.from_numpy(train_x_scaled).float().to(device)
    y_train_tensor = torch.from_numpy(train_y_scaled.reshape(-1, 1)).float().to(device)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN_LSTM_SKIP(
        input_features=actual_input_features,
        window_size=window_size,
        cnn_channels=64, # Number of output channels for CNN
        cnn_kernel_size=3, # Kernel size for CNN
        cnn_pool_kernel_size=2, # Pooling kernel size
        lstm_hidden_size=hidden_dim,
        lstm_num_layers=num_layers,
        output_dim=1 # Output dimension (1 for regression)
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() # Loss is calculated on standardized values
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True) # Use patience 20, factor 0.5

    losses = [] # Store loss for the combined training process

    # Single training loop using the combined data dataloader
    print(f"\n--- Starting training for {epochs} epochs on combined data ---")
    model.train() # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_X) # model expects (batch, seq_len, input_size)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        # Step the scheduler based on the average loss
        scheduler.step(avg_loss)

        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 1 or epoch == epochs: # Print more frequently
             current_lr = optimizer.param_groups[0]['lr']
             print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Current LR: {current_lr:.7f}')

    print("--- Training complete ---")

    # --- Evaluate on Individual Battery Test Sets ---
    print("\n--- Evaluating trained model on individual battery test sets ---")

    score_list = [] # Store evaluation scores per battery
    all_test_results = [] # Store prediction results per battery for plotting

    model.eval()
    with torch.no_grad(): # Disable gradient calculation for evaluation
        for battery_test_data in individual_test_data_list:
            name = battery_test_data['name']
            test_x = battery_test_data['test_x']
            test_y = battery_test_data['test_y']
            test_cycles = battery_test_data['test_cycles']

            test_x_scaled = (test_x - feature_means) / feature_stds
            X_test_tensor = torch.from_numpy(test_x_scaled).float().to(device)

            pred_test_scaled = model(X_test_tensor)  # output is standardized
            pred_test_numpy_original = pred_test_scaled.data.cpu().numpy().reshape(-1) * target_std + target_mean  # De-standardize

            test_y_numpy_original = test_y.reshape(-1)

            test_mae, test_mse, test_rmse, test_r2, test_mape = evaluation(y_test=test_y_numpy_original, y_predict=pred_test_numpy_original)
            print(f'Test results for {name}: MAE:{test_mae:<6.4f} | MSE:{test_mse:<6.6f} | RMSE:{test_rmse:<6.6f} | R2:{test_r2:<6.4f} | MAPE:{test_mape:<6.4f}%')

            # Store scores and prediction results
            score_list.append([test_mae, test_mse, test_rmse, test_r2, test_mape])
            all_test_results.append({
                'name': name,
                'cycles': test_cycles, # Original cycle numbers for test data
                'ground_truth': test_y_numpy_original,
                'prediction': pred_test_numpy_original
            })

    score_array = np.array(score_list)
    avg_scores = np.nanmean(score_array, axis=0) 

    print(f"Average Scores: MAE:{avg_scores[0]:<6.4f} | MSE:{avg_scores[1]:<6.4f} | RMSE:{avg_scores[2]:<6.4f} | R2:{avg_scores[3]:<6.4f} | MAPE:{avg_scores[4]:<6.4f}%")


    # Return the calculated scores, the single combined loss list, and the test results for plotting
    return score_list, losses, all_test_results # Return combined losses, not a list containing the list


'''setting and training for overall performance'''
# These settings apply to the single model trained on combined data
window_size = 8      # Lookback window size (sequence length)
epochs = 500
lr = 0.001           # learning rate
hidden_dim = 128     # Hidden size for the RNN/LSTM/GRU cell
num_layers = 1       # Number of layers for the RNN/LSTM/GRU cell
weight_decay = 0.0   # L2 regularization
mode = 'LSTM'        # RNN, LSTM, GRU mode
# Rated_Capacity = 2.0 # Rated capacity is not used directly for scaling anymore as we use Z-score normalization
metric = 'rmse'      # Primary metric to track (not used for early stopping in this version, but useful)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# Set a seed for reproducibility
seed = 1
setup_seed(seed)


# Call the modified training function
score_list, combined_losses, all_test_results = train_val_test(
    lr=lr,
    window_size=window_size,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    weight_decay=weight_decay,
    mode=mode,
    epochs=epochs,
    seed=seed,
    device=device
)


np.save("CNN_LSTM_Skip_results_b.npy", all_test_results)

# --- Plotting Combined Training Loss ---
print("\n--- Plotting Combined Training Loss ---")
plt.figure(figsize=(10, 6))
plt.plot(combined_losses, label='Training Loss (Combined Data)')
plt.title('Training Loss on Combined Battery Data')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()


# --- Plotting Predictions vs Ground Truth on Test Sets for Each Battery ---
print("\n--- Plotting Predictions vs Ground Truth on Test Sets per Battery ---")

# Filter out batteries that had no test data instances created (e.g., not enough cycles)
plotable_results = [res for res in all_test_results if len(res['cycles']) > 0]

if not plotable_results:
     print("No battery had enough test data instances to plot predictions.")
else:
    # Determine grid size based on the number of batteries with test results
    num_plots = len(plotable_results)
    rows = math.ceil(num_plots / 2)
    cols = 2 if num_plots > 1 else 1 # Use 2 columns if more than one plot

    # Adjust figure size based on the number of rows and columns
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6)) # Adjust figure size

    # Flatten the axs array for easy indexing if it's a grid
    if num_plots == 1:
        axes_to_plot = [axs] # Wrap single axis in a list
    else:
        axes_to_plot = axs.flatten() # Flatten the grid

    for i, result in enumerate(plotable_results):
        ax = axes_to_plot[i] # Get the current subplot

        battery_name = result['name']
        # Retrieve the full cycle and capacity data for plotting the complete ground truth curve
        full_data = Battery.get(battery_name) # Use .get() to safely access
        if full_data and len(full_data) > 1:
             full_cycles = full_data[0]
             full_capacities = full_data[1]
             # Plot Ground Truth (Using all available cycles for comparison)
             if len(full_cycles) > 0: # Check if full data is not empty
                 ax.plot(full_cycles, full_capacities, label='Ground Truth (All Cycles)', marker='o', linestyle='-', markersize=4)
             else:
                 print(f"Warning: No full capacity data available for {battery_name} to plot.")
        else:
             print(f"Warning: Full data missing or incomplete for {battery_name}. Cannot plot full ground truth.")
             full_cycles = np.array([]) # Set to empty to avoid errors


        # Plot Prediction (Predictions only exist for the test cycles)
        if len(result['cycles']) > 0: # Ensure test results are not empty
            ax.plot(result['cycles'], result['prediction'], label='Prediction (Test Cycles)', marker='o', linestyle='-', markersize=4)
        else:
             print(f"Warning: No test prediction data available for {battery_name} to plot.")


        ax.set_title(f'Battery {result["name"]} Test Prediction')
        ax.set_xlabel('Discharge cycles')
        ax.set_ylabel('Capacity (Ah)')
        ax.legend()
        ax.grid(True)

        # Optional: Highlight the training/test split point on the full ground truth line
        if len(full_cycles) > 0 and len(result['cycles']) > 0:
             # Find the cycle number right before the test set starts
             first_test_cycle = result['cycles'][0]
             # Find the index of this cycle in the full cycles list
             split_point_index = np.where(full_cycles == first_test_cycle)[0]
             if len(split_point_index) > 0:
                 split_point_index = split_point_index[0] # Get the first occurrence
                 # Draw a vertical line at the start of the test cycles
                 ax.axvline(x=full_cycles[split_point_index], color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
                 ax.legend() # Update legend to include the vertical line label


    # Hide any unused subplots if fewer batteries had test data than available subplots
    if num_plots > 0:
        for j in range(num_plots, len(axes_to_plot)):
             fig.delaxes(axes_to_plot[j]) # Remove unused axes

    plt.tight_layout() # Adjust layout
    plt.show()