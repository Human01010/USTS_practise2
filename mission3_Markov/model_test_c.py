import torch
from joblib import load
import torch.utils.data as Data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- Configuration ---
MODEL_PATH = 'model/best_model_TransformerBiLSTM.pt' # Model saved by model_train.py
SCALER_PATH = 'dataset/scaler_data'       # Scaler saved by data_process_c.py
TRAIN_X_PATH = 'dataset/train_X'          # train_X saved by data_process_c.py
TRAIN_Y_PATH = 'dataset/train_Y'          # train_Y saved by data_process_c.py
TEST_X_PATH = 'dataset/test_X'            # test_X saved by data_process_c.py
TEST_Y_PATH = 'dataset/test_Y'            # test_Y saved by data_process_c.py

RESULTS_DIR = 'model_c_results'           # Directory to save plots for this test script
BATCH_SIZE = 32                           # Should match batch_size used in model_train.py

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure Matplotlib for Chinese characters if needed
try:
    matplotlib.rc("font", family='Microsoft YaHei') # Or 'SimHei' or another font that supports Chinese
except RuntimeError:
    print("Chinese font 'Microsoft YaHei' not found, using default. Plots might not display Chinese characters correctly.")

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model ---
try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Load Scaler ---
try:
    scaler = load(SCALER_PATH)
    print(f"Scaler loaded successfully from {SCALER_PATH}")
except FileNotFoundError:
    print(f"Error: Scaler file not found at {SCALER_PATH}")
    exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# --- Load Data and Create DataLoaders ---
try:
    print("Loading data processed by data_process_c.py...")
    train_X_loaded = load(TRAIN_X_PATH)
    train_Y_loaded = load(TRAIN_Y_PATH)
    test_X_loaded = load(TEST_X_PATH)
    test_Y_loaded = load(TEST_Y_PATH)

    # Ensure data are PyTorch tensors (data_process_c.py saves them as tensors)
    if not isinstance(train_X_loaded, torch.Tensor):
        train_X_loaded = torch.tensor(train_X_loaded, dtype=torch.float32)
    if not isinstance(train_Y_loaded, torch.Tensor):
        train_Y_loaded = torch.tensor(train_Y_loaded, dtype=torch.float32)
    if not isinstance(test_X_loaded, torch.Tensor):
        test_X_loaded = torch.tensor(test_X_loaded, dtype=torch.float32)
    if not isinstance(test_Y_loaded, torch.Tensor):
        test_Y_loaded = torch.tensor(test_Y_loaded, dtype=torch.float32)

    print(f"  Train X shape: {train_X_loaded.shape}, Train Y shape: {train_Y_loaded.shape}")
    print(f"  Test X shape: {test_X_loaded.shape}, Test Y shape: {test_Y_loaded.shape}")

    train_dataset_c = Data.TensorDataset(train_X_loaded, train_Y_loaded)
    # shuffle=False for consistent plotting order, drop_last to match training if it was used
    train_loader_c = Data.DataLoader(dataset=train_dataset_c, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    test_dataset_c = Data.TensorDataset(test_X_loaded, test_Y_loaded)
    test_loader_c = Data.DataLoader(dataset=test_dataset_c, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print("DataLoaders for data_process_c.py outputs created successfully.")

except FileNotFoundError as e:
    print(f"Error: Data file not found. {e}")
    print("Please ensure data_process_c.py has been run successfully.")
    exit()
except Exception as e:
    print(f"Error loading or processing data: {e}")
    exit()

# --- Helper Functions ---
def predict_c(model, data_loader, device):
    """Generates predictions for the given data_loader."""
    true_values_list = []
    predicted_values_list = []
    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            true_values_list.append(labels.cpu().numpy())
            predicted_values_list.append(outputs.cpu().numpy())
    # Concatenate batches into single numpy arrays
    true_values_np = np.concatenate(true_values_list, axis=0)
    predicted_values_np = np.concatenate(predicted_values_list, axis=0)
    return true_values_np, predicted_values_np

def inverse_transform_c(scaler, data_np):
    """Applies inverse transformation to the data."""
    # Scaler expects 2D array (n_samples, n_features)
    # Data from predict_c will be (N, 1) if output_dim is 1
    if data_np.ndim == 1:
        data_np = data_np.reshape(-1, 1)
    return scaler.inverse_transform(data_np)

def plot_train_results_c(true_values, predicted_values, title_prefix, filename):
    """Plots training results split into two subplots."""
    mid_index = len(true_values) // 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.style.use('seaborn-v0_8-whitegrid')
    true_color = '#1f77b4'
    predicted_color = '#ff7f0e'

    # First subplot
    axes[0].plot(true_values[:mid_index], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[0].plot(predicted_values[:mid_index], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[0].set_title(f'{title_prefix} - Part 1')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Capacity (Ah)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Second subplot
    axes[1].plot(true_values[mid_index:], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[1].plot(predicted_values[mid_index:], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[1].set_title(f'{title_prefix} - Part 2')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Capacity (Ah)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close(fig)

def plot_results_c(true_values, predicted_values, title, filename):
    """Plots true vs. predicted values for a single plot."""
    plt.figure(figsize=(12, 6), dpi=100)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.style.use('seaborn-v0_8-whitegrid')
    true_color = '#1f77b4'
    predicted_color = '#ff7f0e'

    plt.plot(true_values, label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    plt.plot(predicted_values, label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Capacity (Ah)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

def calculate_metrics_c(y_true, y_pred):
    """Calculates regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE can be sensitive to zero true values, adding epsilon for stability
    mape = np.mean(np.abs((y_true - y_pred) / (y_true.clip(min=1e-8)))) * 100 # clip to avoid division by zero
    return mse, rmse, mae, r2, mape

# --- Main Execution ---
if __name__ == "__main__":
    # --- Evaluate on Training Set (from data_process_c.py) ---
    if train_loader_c is not None and len(train_loader_c.dataset) > 0:
        print("\n--- Evaluating on Training Set (Data from data_process_c.py) ---")
        train_true_norm, train_pred_norm = predict_c(model, train_loader_c, device)

        train_true_original = inverse_transform_c(scaler, train_true_norm)
        train_pred_original = inverse_transform_c(scaler, train_pred_norm)

        plot_train_results_c(train_true_original, train_pred_original,
                             'Training Set Prediction (data_process_c)',
                             'train_c_results_split.png')

        train_mse, train_rmse, train_mae, train_r2, train_mape = calculate_metrics_c(train_true_original, train_pred_original)
        print("Training Set Metrics (data_process_c):")
        print(f"  R^2 Score: {train_r2:.4f}")
        print(f"  MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.2f}%")
    else:
        print("\nTraining data loader is empty or not available. Skipping training set evaluation.")

    # --- Evaluate on Test Set (from data_process_c.py) ---
    if test_loader_c is not None and len(test_loader_c.dataset) > 0:
        print("\n--- Evaluating on Test Set (Data from data_process_c.py) ---")
        test_true_norm, test_pred_norm = predict_c(model, test_loader_c, device)

        test_true_original = inverse_transform_c(scaler, test_true_norm)
        test_pred_original = inverse_transform_c(scaler, test_pred_norm)

        plot_results_c(test_true_original, test_pred_original,
                       'Test Set Prediction (data_process_c)',
                       'test_c_results.png')

        test_mse, test_rmse, test_mae, test_r2, test_mape = calculate_metrics_c(test_true_original, test_pred_original)
        print("Test Set Metrics (data_process_c):")
        print(f"  R^2 Score: {test_r2:.4f}")
        print(f"  MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%")
    else:
        print("\nTest data loader is empty or not available. Skipping test set evaluation.")

    # No validation set evaluation as data_process_c.py does not create a separate validation set.

    print(f"\n--- Testing with data_process_c.py outputs complete. Results saved in '{RESULTS_DIR}' ---")