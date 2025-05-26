import torch
from joblib import load
import torch.utils.data as Data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- Configuration ---
MODEL_PATH = 'model/best_model_TransformerBiLSTM.pt'
SCALER_PATH = 'dataset/scaler_data' # Saved by data_process_b.py
TRAIN_X_PATH = 'dataset/train_X'    # Saved by data_process_b.py
TRAIN_Y_PATH = 'dataset/train_Y'    # Saved by data_process_b.py
TEST_X_PATH = 'dataset/test_X'      # Saved by data_process_b.py
TEST_Y_PATH = 'dataset/test_Y'      # Saved by data_process_b.py
RESULTS_DIR = 'model_b_results'     # Directory to save plots for this test script
BATCH_SIZE = 32                     # Should match batch_size used in model_train.py

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure Matplotlib for Chinese characters if needed
try:
    matplotlib.rc("font", family='Microsoft YaHei') # Or 'SimHei'
except RuntimeError:
    print("Chinese font not found, using default.")

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model ---
try:
    # Ensure the model is loaded onto the correct device from the start
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
    train_X_loaded = load(TRAIN_X_PATH)
    train_Y_loaded = load(TRAIN_Y_PATH)
    test_X_loaded = load(TEST_X_PATH)
    test_Y_loaded = load(TEST_Y_PATH)

    # Convert to PyTorch tensors if they are not already (joblib might load them as tensors or numpy arrays)
    if not isinstance(train_X_loaded, torch.Tensor):
        train_X_loaded = torch.tensor(train_X_loaded, dtype=torch.float32)
    if not isinstance(train_Y_loaded, torch.Tensor):
        train_Y_loaded = torch.tensor(train_Y_loaded, dtype=torch.float32)
    if not isinstance(test_X_loaded, torch.Tensor):
        test_X_loaded = torch.tensor(test_X_loaded, dtype=torch.float32)
    if not isinstance(test_Y_loaded, torch.Tensor):
        test_Y_loaded = torch.tensor(test_Y_loaded, dtype=torch.float32)

    print(f"Data loaded successfully: ")
    print(f"  Train X shape: {train_X_loaded.shape}, Train Y shape: {train_Y_loaded.shape}")
    print(f"  Test X shape: {test_X_loaded.shape}, Test Y shape: {test_Y_loaded.shape}")


    train_dataset_b = Data.TensorDataset(train_X_loaded, train_Y_loaded)
    train_loader_b = Data.DataLoader(dataset=train_dataset_b, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for consistent plotting

    test_dataset_b = Data.TensorDataset(test_X_loaded, test_Y_loaded)
    test_loader_b = Data.DataLoader(dataset=test_dataset_b, batch_size=BATCH_SIZE, shuffle=False)

except FileNotFoundError as e:
    print(f"Error: Data file not found. {e}")
    exit()
except Exception as e:
    print(f"Error loading or processing data: {e}")
    exit()


# --- Helper Functions (adapted from user's model_test.py) ---
def predict(model, data_loader, device):
    true_values = []
    predicted_values = []
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        for data, label in data_loader:
            # data and label are already tensors from DataLoader
            data, label = data.to(device), label.to(device)
            pred = model(data)
            true_values.extend(label.cpu().numpy()) # Use extend for list of arrays/tensors
            predicted_values.extend(pred.cpu().numpy())
    # Convert list of (batch_size, 1) arrays/tensors to a single (N, 1) numpy array
    return np.vstack(true_values), np.vstack(predicted_values)


def inverse_transform(scaler, data):
    # Scaler expects 2D array (n_samples, n_features)
    # Data from predict will be (N,1) already if output_dim is 1
    if data.ndim == 1:
        data = data.reshape(-1,1)
    return scaler.inverse_transform(data)

def plot_train_results(true_values, predicted_values, title_prefix, save_filename):
    mid_index = len(true_values) // 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.style.use('seaborn-v0_8-whitegrid')
    true_color = '#1f77b4'
    predicted_color = '#ff7f0e'

    axes[0].plot(true_values[:mid_index-10], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[0].plot(predicted_values[:mid_index-10], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[0].set_title(f'{title_prefix} - Part 1')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Capacity (Ah)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(true_values[mid_index+10:], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[1].plot(predicted_values[mid_index+10:], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[1].set_title(f'{title_prefix} - Part 2')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Capacity (Ah)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_filename), dpi=300)
    print(f"Plot saved to {os.path.join(RESULTS_DIR, save_filename)}")
    plt.close(fig)

def plot_results(true_values, predicted_values, title, save_filename):
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
    plt.savefig(os.path.join(RESULTS_DIR, save_filename), dpi=300)
    print(f"Plot saved to {os.path.join(RESULTS_DIR, save_filename)}")
    plt.close()

def calculate_metrics(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    # MAPE can be sensitive to zero true values, adding epsilon for stability
    mape = np.mean(np.abs((true_values - predicted_values) / (true_values + 1e-8))) * 100
    return mse, rmse, mae, r2, mape

# --- Main Execution ---
if __name__ == "__main__":
    # --- Evaluate on Training Set (from data_process_b.py) ---
    if train_loader_b.dataset: # Check if dataset is not empty
        print("\n--- Evaluating on Training Set (Data from data_process_b.py) ---")
        train_true_norm, train_pred_norm = predict(model, train_loader_b, device)

        train_true_original = inverse_transform(scaler, train_true_norm)
        train_pred_original = inverse_transform(scaler, train_pred_norm)

        plot_train_results(train_true_original, train_pred_original,
                           'Training Set Prediction Results (data_process_b)',
                           'train_b_results_split.png')

        train_mse, train_rmse, train_mae, train_r2, train_mape = calculate_metrics(train_true_original, train_pred_original)
        print("Training Set Metrics (data_process_b):")
        print(f"  R^2 Score: {train_r2:.4f}")
        print(f"  MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.2f}%")
    else:
        print("\nTraining data is empty or not loaded, skipping training set evaluation.")

    # --- Evaluate on Test Set (from data_process_b.py) ---
    if test_loader_b.dataset: # Check if dataset is not empty
        print("\n--- Evaluating on Test Set (Data from data_process_b.py) ---")
        test_true_norm, test_pred_norm = predict(model, test_loader_b, device)

        test_true_original = inverse_transform(scaler, test_true_norm)
        test_pred_original = inverse_transform(scaler, test_pred_norm)

        plot_results(test_true_original, test_pred_original,
                     'Test Set Prediction Results (data_process_b)',
                     'test_b_results.png')

        test_mse, test_rmse, test_mae, test_r2, test_mape = calculate_metrics(test_true_original, test_pred_original)
        print("Test Set Metrics (data_process_b):")
        print(f"  R^2 Score: {test_r2:.4f}")
        print(f"  MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%")
    else:
        print("\nTest data is empty or not loaded, skipping test set evaluation.")

    print("\n--- Testing with data_process_b.py outputs complete ---")