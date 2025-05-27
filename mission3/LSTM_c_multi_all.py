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
# %matplotlib inline # If running in Jupyter, uncomment this line

from tqdm.notebook import tqdm
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset # Added for DataLoader

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

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
# Assuming the script is run from the directory containing 'battery_dataset'
dir_path = r'./battery_dataset/'
# Or specify absolute path if needed: dir_path = r'C:\your\path\to\battery_dataset\'


Battery = {}  # {name: [cycles, capacities, feature1, feature2]}
# feature_choose = int(input('Choose features: 1 for voltage and time, 2 for current and slope: '))
feature_choose = 1

for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = os.path.join(dir_path, name + '.mat') # Use os.path.join for cross-platform compatibility
    data = loadMat(path)
    raw_battery_features = getBatteryFeatures(data,feature_choose=feature_choose)              # 放电时的容量数据
    cleaned_battery_features = clean_capacity_data(raw_battery_features)
    # cleaned_battery_features = raw_battery_features

    Battery[name] = cleaned_battery_features


'''plot figures for capacity degradation'''

fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b:', 'g--', 'r-.', 'c.']
# c = 0 # This variable was unused, removed it.
for name,color in zip(Battery_list, color_list):
    if name in Battery and len(Battery[name]) > 1 and len(Battery[name][0]) > 0: # Check if data exists
        df_result = Battery[name]
        ax.plot(df_result[0], df_result[1], color, label=name)
    else:
        print(f"Warning: No capacity data to plot for battery {name}.")

ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
plt.legend()
plt.show()



'''
data processing for model training
'''

def build_instances(sequence, window_size):
    x, y = [],[]
    # Ensure sequence is a numpy array
    sequence = np.array(sequence)
    num_features = sequence.shape[0] # Number of features per time step

    # Check if sequence has enough data for the window size
    if sequence.shape[1] < window_size + 1: # Need window_size history + 1 target
        return np.array([]).astype(np.float32), np.array([]).astype(np.float32)

    for i in range(window_size, sequence.shape[1]):
        features_window = sequence[:, i - window_size : i]
        target = sequence[0, i]

        features_transposed = features_window.T # Transpose from (num_features, window_size) to (window_size, num_features)

        x.append(features_transposed)
        y.append(target)

    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)


# B0005(对应i=0)数据作为训练数据，其他电池中前10%的数据用作训练，后90%数据作为测试
def get_train_test_c(data_dict, window_size=8):
    all_train_x, all_train_y = [], []
    all_test_x, all_test_y = [], []
    all_test_cycles = [] # Store test cycles for visualization
    test_battery_lengths = [] # Store number of test samples per battery

    for i, name in enumerate(Battery_list):
        # data_sequence: [cycles, capacity, feature1, feature2]
        data_features_only = data_dict[name][1:] 
        original_cycles = data_dict[name][0] 

        # build_instances now returns x with shape (num_instances, window_size, num_features)
        x_instances, y_instances = build_instances(data_features_only, window_size)

        if len(x_instances) > 0:
            # Cycles corresponding to the generated instances (targets)
            # The first target is at index `window_size` of the *original* sequence
            cycles_for_instances = original_cycles[window_size:]

            if name == 'B0005':
                # B0005's entire valid data for training
                all_train_x.append(x_instances)
                all_train_y.append(y_instances)
            else:
                # Other batteries: 10% train, 90% test
                num_instances = len(x_instances)
                split_idx = int(0.1 * num_instances) # 10% for train

                # Ensure there's at least one instance for training/testing if data exists
                if num_instances > 0 and split_idx == 0 and num_instances >= 2:
                     split_idx = 1 # Use at least one sample for training if possible
                elif num_instances > 0 and split_idx == num_instances: # If 10% results in using all data as train
                     split_idx = num_instances - 1 # Use at least one sample for test if possible

                train_x_battery = x_instances[:split_idx]
                train_y_battery = y_instances[:split_idx]

                test_x_battery = x_instances[split_idx:]
                test_y_battery = y_instances[split_idx:]
                # Get corresponding test cycles segment
                test_cycles_battery = cycles_for_instances[split_idx:]

                if len(train_x_battery) > 0:
                    all_train_x.append(train_x_battery)
                    all_train_y.append(train_y_battery)
                if len(test_x_battery) > 0:
                    all_test_x.append(test_x_battery)
                    all_test_y.append(test_y_battery)
                    all_test_cycles.append(test_cycles_battery) # Store test cycles for this battery
                    test_battery_lengths.append(len(test_y_battery)) # Store length for this battery

    # Concatenate data from all batteries
    final_train_x = np.concatenate(all_train_x, axis=0) if len(all_train_x) > 0 else np.array([]).reshape(0, window_size, -1)
    final_train_y = np.concatenate(all_train_y, axis=0) if len(all_train_y) > 0 else np.array([])
    final_test_x = np.concatenate(all_test_x, axis=0) if len(all_test_x) > 0 else np.array([]).reshape(0, window_size, -1)
    final_test_y = np.concatenate(all_test_y, axis=0) if len(all_test_y) > 0 else np.array([])
    final_test_cycles = np.concatenate(all_test_cycles, axis=0) if len(all_test_cycles) > 0 else np.array([]) # Concatenate test cycles

    return final_train_x, final_train_y, final_test_x, final_test_y, final_test_cycles, test_battery_lengths

# Modified evaluation function to include MAPE
def evaluation(y_test, y_predict):
    """
    Calculates evaluation metrics. Expects original scale values.
    """
    # Ensure inputs are numpy arrays
    y_test = np.asarray(y_test)
    y_predict = np.asarray(y_predict)

    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_predict)

    non_zero_mask = y_test > 0.1 # Using a threshold like 0.1Ah to avoid division by near-zero values

    y_test_nonzero = y_test[non_zero_mask]
    y_predict_nonzero = y_predict[non_zero_mask] # Ensure alignment

    if len(y_test_nonzero) > 0:
       mape = np.mean(np.abs((y_test_nonzero - y_predict_nonzero) / y_test_nonzero)) * 100
    else:
       mape = np.nan # Cannot calculate MAPE if all true values are zero or below threshold

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

'''build net'''

class Net(nn.Module):
    # input_size should be the number of features per time step (e.g., capacity, feature1, feature2 = 3)
    def __init__(self, input_size, hidden_dim=8, num_layers=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        # RNN/LSTM expects input shape (batch_size, seq_len, input_size)
        # Here, seq_len is window_size, input_size is num_features
        if mode == 'LSTM':
             self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # elif mode == 'GRU':
            #  self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_first=True)
        elif mode == 'RNN':
             self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
             raise ValueError("mode must be 'LSTM', 'GRU', or 'RNN'") # Added error handling for mode


        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):           # x shape: (batch_size, window_size, num_features)
        # out shape: (batch_size, window_size, hidden_dim)
        # hn, cn are final hidden/cell states, we don't need them for sequence-to-one prediction
        out, _ = self.cell(x)
        # Take output from the last time step for prediction
        out = out[:, -1, :]         # out shape: (batch_size, hidden_dim)
        out = self.linear(out)      # out shape: (batch_size, 1)
        return out


'''train for model'''
# Added return of test predictions and ground truth for plotting
# Modified scaling to use mean-variance normalization
def train_val_test(lr, window_size, hidden_dim=64, num_layers=2, weight_decay=0.0, mode='RNN', epochs=1000, seed=0, device='cuda'):

    setup_seed(seed)

    # Get the actual number of features per time step (capacity + chosen features)
    # data_dict[name][1:] has shape (num_features, total_cycles)
    # Need to check if any battery has data after feature extraction
    example_battery_data = None
    for name in Battery_list:
        if name in Battery and len(Battery[name]) > 1:
            example_battery_data = Battery[name]
            break

    if example_battery_data is None:
        print("Error: No battery data loaded or processed successfully.")
        return None, None, None, None

    # Number of features per step is total features - cycles (index 0)
    num_features_per_step = len(example_battery_data) - 1


    # get_train_test_c now returns test cycles and battery lengths
    train_x, train_y, test_x, test_y, test_cycles, test_battery_lengths  = get_train_test_c(Battery, window_size=window_size)

    print(f"Train data shape: {train_x.shape}, Train labels shape: {train_y.shape}")
    print('test battery lengths:', test_battery_lengths)    

    feature_means = np.mean(train_x, axis=(0, 1), keepdims=True)
    feature_stds = np.std(train_x, axis=(0, 1), keepdims=True)   
    feature_stds = np.maximum(feature_stds, 1e-5)

    target_mean = np.mean(train_y) # Scalar
    target_std = np.std(train_y)   # Scalar
    target_std = target_std if target_std > 1e-5 else 1e-5

    # Apply standardization using training stats
    train_x_scaled = (train_x - feature_means) / feature_stds
    train_y_scaled = (train_y - target_mean) / target_std

    # Use train stats for test data scaling
    test_x_scaled = (test_x - feature_means) / feature_stds

    X_train_tensor = torch.from_numpy(train_x_scaled).float().to(device) # Use .float() to ensure float32
    y_train_tensor = torch.from_numpy(train_y_scaled.reshape(-1, 1)).float().to(device) # Use .float()

    X_test_tensor = torch.from_numpy(test_x_scaled).float().to(device)

    model = Net(input_size=num_features_per_step, hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss() # Loss is calculated on standardized values

    losses = []

    model.train() # Set model to training mode
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    batch_size = 32 # Example batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_dataloader:
            # batch_X shape: (batch_size, window_size, num_features_per_step)
            # batch_y shape: (batch_size, 1)

            optimizer.zero_grad()
            output = model(batch_X) # output shape: (batch_size, 1)
            loss = criterion(output, batch_y) # Calculate loss on scaled values
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0 or epoch == 1:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.6f}')

    print("--- Training complete ---")


    # Evaluation on test set
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Predict on test data (output is scaled)
        # Process test data in batches to avoid potential memory issues for large test sets
        test_dataset = TensorDataset(X_test_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No shuffling for evaluation

        pred_test_scaled_list = []

        for batch_X_test in test_dataloader:
                batch_X_test = batch_X_test[0].to(device)
                batch_pred_scaled = model(batch_X_test)
                pred_test_scaled_list.append(batch_pred_scaled.cpu().numpy()) # Move to CPU and convert to numpy

        pred_test_scaled = np.concatenate(pred_test_scaled_list, axis=0) # Concatenate batch results
        # Convert predictions back to original scale using target mean and std dev
        pred_test_numpy = pred_test_scaled.reshape(-1) * target_std + target_mean



    # test_y is already in original scale, reshape for consistency
    test_y_numpy = test_y.reshape(-1)

    test_mae, test_mse, test_rmse, test_r2, test_mape = evaluation(y_test=test_y_numpy, y_predict=pred_test_numpy)
    print('--- Test results ---')
    overall_metrics = [test_mae, test_mse, test_rmse, test_r2, test_mape]

    # Prepare results for plotting (group by battery)
    all_test_results = []
    current_idx = 0

    test_battery_names = Battery_list[1:] # Get names of batteries used for testing

    for i, length in enumerate(test_battery_lengths):
        if current_idx + length > len(test_y_numpy):
            print(f"Error: Indexing out of bounds when grouping battery test results for {test_battery_names[i]}. Check test_battery_lengths vs actual test data length.")
            break # Avoid error
        battery_name = test_battery_names[i]
        # Extract the segment for the current battery
        cycles_segment = test_cycles[current_idx : current_idx + length]
        ground_truth_segment = test_y_numpy[current_idx : current_idx + length]
        prediction_segment = pred_test_numpy[current_idx : current_idx + length]
        # sigle battery test evaluation
        battery_mae, battery_mse, battery_rmse, battery_r2, battery_mape = evaluation(y_test=ground_truth_segment, y_predict=prediction_segment)
        print(f'Battery {battery_name} Test results : MAE:{battery_mae:<6.4f} | MSE:{battery_mse:<6.6f} | RMSE:{battery_rmse:<6.6f} | R2:{battery_r2:<6.4f} | MAPE:{battery_mape:<6.4f}%')

        all_test_results.append({
            'name': battery_name,
            'cycles': cycles_segment,
            'ground_truth': ground_truth_segment,
            'prediction': prediction_segment
        })
        current_idx += length # Move index for the next battery
    
    print(f'Average Test results : MAE:{test_mae:<6.4f} | MSE:{test_mse:<6.6f} | RMSE:{test_rmse:<6.6f} | R2:{test_r2:<6.4f} | MAPE:{test_mape:<6.4f}%')
    
    return overall_metrics, losses, all_test_results, test_battery_lengths


'''setting and training for overall performance'''
# feature_size is effectively the window_size
window_size = 16
epochs = 500
lr = 0.001           # learning rate
hidden_dim = 128
num_layers = 1
weight_decay = 0.0
mode = 'LSTM'        # RNN, LSTM, GRU
# Rated_Capacity is no longer directly used for scaling, it can be removed or kept for context
# Rated_Capacity = 2.0

metric = 'rmse' # Unused variable, kept for context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

seed = 1

# Call the training function, now receiving overall metrics, losses, and test results
overall_metrics, loss_list, all_test_results, test_battery_lengths = train_val_test(
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

np.save("LSTM_results_c.npy", all_test_results)

# Check if training completed successfully and test data was available
if overall_metrics is not None and len(all_test_results) > 0 and len(overall_metrics) > 0 and not np.isnan(overall_metrics[0]):
    # The overall metrics are already printed inside train_val_test
    print("\n--- Plotting Results ---")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Train Loss (Scaled)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plotting Predictions vs Ground Truth on Test Sets ---
    print("\n--- Plotting Predictions vs Ground Truth on Test Sets ---")

    # Filter out batteries that had no test data (e.g., if window_size was too large)
    plotable_results = [res for res in all_test_results if len(res['cycles']) > 0]

    if len(plotable_results) > 0:
        # Determine grid size based on the number of batteries with test results
        num_plots = len(plotable_results)
        # Ensure at least one row and column if there are plots
        rows = max(1, math.ceil(num_plots / 2))
        cols = 2 if num_plots > 1 else 1

        # Adjust figure size based on the number of rows and columns
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6)) # Adjust figure size

        # Get flat list of axes for easy iteration
        if num_plots == 1:
            axes_to_plot = [axs] # axs is a single Axes object
        else:
            axes_to_plot = axs.flatten() # axs is an array of Axes objects

        for i, result in enumerate(plotable_results):
            ax = axes_to_plot[i] # Get the current subplot axis

            battery_name = result['name']
            # Get the full original data for this battery to plot the complete ground truth curve
            # Check if the battery exists and has data
            if battery_name in Battery and len(Battery[battery_name]) > 1:
                 full_cycles = Battery[battery_name][0]
                 full_capacities = Battery[battery_name][1]
                 # Plot full Ground Truth curve
                 ax.plot(full_cycles, full_capacities, label=f'Ground Truth ({battery_name})', marker='o', linestyle='-', markersize=4)
            else:
                 print(f"Warning: Full data for battery {battery_name} not found for plotting ground truth.")


            # Plot Prediction curve (only on the test cycles)
            if len(result['cycles']) > 0:
                 ax.plot(result['cycles'], result['prediction'], label=f'Prediction ({battery_name} Test Set)', marker='x', linestyle='--', markersize=4)
            else:
                 print(f"Warning: No prediction data for battery {battery_name} test set.")


            ax.set_title(f'Battery {battery_name} Test Prediction')
            ax.set_xlabel('Discharge cycles')
            ax.set_ylabel('Capacity (Ah)')
            ax.legend()
            ax.grid(True)

        # Hide any unused subplots if the number of plotable results is odd
        if num_plots > 0 and num_plots < len(axes_to_plot):
            for j in range(num_plots, len(axes_to_plot)):
                 fig.delaxes(axes_to_plot[j]) # Remove unused axis

        plt.tight_layout() # Adjust layout to prevent overlap
        plt.show()
    else:
        print("No test data available for plotting predictions.")
else:
    print("Training failed, no test data available, or evaluation metrics are NaN.")