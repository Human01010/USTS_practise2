import torch
from joblib import load
import torch.utils.data as Data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Import DataLoaders from model_train.py
# This assumes model_train.py is in the same directory or accessible via PYTHONPATH
try:
    from model_train import train_loader, val_loader, test_loader
except ImportError:
    print("Could not import DataLoaders from model_train.py.")
    print("Please ensure model_train.py has been run and DataLoaders are accessible,")
    print("or modify this script to load data independently.")
    exit()


matplotlib.rc("font", family='Microsoft YaHei') # Or 'SimHei' or another font that supports Chinese

# --- Configuration ---
MODEL_PATH = 'model/best_model_TransformerBiLSTM.pt'
SCALER_PATH = 'dataset/scaler_data' # As saved in data_process_a.py
RESULTS_DIR = 'model_a_results' # Directory to save plots for this script (model_test_a.py)

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# --- Helper Functions ---
def predict(model, data_loader, device):
    """Generates predictions for the given data_loader."""
    true_values = []
    predicted_values = []
    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            true_values.extend(labels.cpu().numpy())
            predicted_values.extend(outputs.cpu().numpy())
    return np.array(true_values), np.array(predicted_values)

def inverse_transform_data(scaler, data):
    """Applies inverse transformation to the data."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return scaler.inverse_transform(data)

def calculate_metrics(y_true, y_pred):
    """Calculates regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE can be sensitive to zero true values
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100 # Added epsilon for stability
    return mse, rmse, mae, r2, mape

def plot_predictions(true_values, predicted_values, title, filename):
    """Plots true vs. predicted values."""
    plt.figure(figsize=(12, 6), dpi=100)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12}) # Consistent with model_train.py
    plt.style.use('seaborn-v0_8-whitegrid') # Consistent with model_train.py

    true_color = '#1f77b4'
    predicted_color = '#ff7f0e'

    plt.plot(true_values, label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    plt.plot(predicted_values, label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    plt.title(title)
    plt.xlabel('Discharge Cycles (or Sample Index)')
    plt.ylabel('Capacity (Ah) - Original Scale')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    print(f"Plot saved to {os.path.join(RESULTS_DIR, filename)}")
    plt.close()

def plot_train_results_split(true_values, predicted_values, title, filename_prefix):
    """Plots training results split into two subplots (as in the user's model_test.py)."""
    mid_index = len(true_values) // 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    plt.style.use('seaborn-v0_8-whitegrid')
    true_color = '#1f77b4'
    predicted_color = '#ff7f0e'

    # First subplot
    axes[0].plot(true_values[:mid_index-10], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[0].plot(predicted_values[:mid_index-10], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[0].set_title(f'{title} - Part 1 (e.g., B0005)')
    axes[0].set_xlabel('Discharge Cycles (or Sample Index)')
    axes[0].set_ylabel('Capacity (Ah)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Second subplot
    axes[1].plot(true_values[mid_index+10:], label='True Values', color=true_color, linewidth=1.5, alpha=0.8)
    axes[1].plot(predicted_values[mid_index+10:], label='Predicted Values', color=predicted_color, linewidth=1.5, linestyle='--', alpha=0.8)
    axes[1].set_title(f'{title} - Part 2 (e.g., B0006)')
    axes[1].set_xlabel('Discharge Cycles (or Sample Index)')
    axes[1].set_ylabel('Capacity (Ah)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{filename_prefix}_split_results_a.png"), dpi=300)
    print(f"Split plot saved to {os.path.join(RESULTS_DIR, f'{filename_prefix}_split_results_a.png')}")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    datasets = {
        "Train": train_loader,
        "Validation": val_loader,
        "Test": test_loader
    }

    for name, data_loader_instance in datasets.items():
        if data_loader_instance is None:
            print(f"Skipping {name} set as DataLoader is not available.")
            continue

        print(f"\n--- Evaluating on {name} Set ---")

        # Predict
        true_norm, pred_norm = predict(model, data_loader_instance, device)

        # Inverse transform
        # Ensure the shapes are correct for the scaler, typically (n_samples, n_features)
        # The target variable (Y) is usually a single feature for RUL.
        true_original = inverse_transform_data(scaler, true_norm.reshape(-1, 1))
        pred_original = inverse_transform_data(scaler, pred_norm.reshape(-1, 1))

        # Calculate metrics
        mse, rmse, mae, r2, mape = calculate_metrics(true_original, pred_original)
        print(f"{name} Set Metrics:")
        print(f"  R^2 Score: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        # Plot predictions
        plot_title = f'{name} Set: True vs. Predicted RUL'
        plot_filename = f'{name.lower()}_predictions_a.png'
        plot_predictions(true_original, pred_original, plot_title, plot_filename)

        # Special plotting for training set if desired (like in user's example)
        if name == "Train":
             plot_train_results_split(true_original, pred_original, 'Training Set Prediction Results', 'train')


    print("\n--- Testing Complete ---")