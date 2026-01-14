import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
import pickle
import sys

# Add project root to path to import model_arch
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
_GAME_RACER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

from model_arch import SimpleModel

# ============================================================
# 1. Configuration Loading and Path Setup
# ============================================================

parser = argparse.ArgumentParser()
# Unique command line argument: configuration file path
default_config_path = os.path.join(_GAME_RACER_ROOT, "config", "train_q_args.yaml")
parser.add_argument("--config", default=default_config_path, help="Path to training config yaml")
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"Configuration file not found at: {args.config}")

print(f"Loading configuration from: {args.config}")
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Extract configuration blocks
DATA_CFG = cfg['data']
TRAIN_CFG = cfg['training']
MODEL_CFG = cfg['model']
DEVICE_CFG = cfg['device']

# Directory setup
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
Q_MODELS_DIR = os.path.join(_PROJECT_ROOT, "q_models")
os.makedirs(Q_MODELS_DIR, exist_ok=True)


# ============================================================
# 2. Helper Functions
# ============================================================

def wrap(x):
    """Handle track cyclic wrapping (approx. 150m)"""
    return (
        (x > 75.) * (x - 150.087)
        + (x < -75.) * (x + 150.087)
        + (x <= 75.) * (x >= -75.) * x
    )

def get_num_cars_from_collect_yaml(project_root):
    """Attempt to read number of cars from collect.yaml"""
    collect_yaml_path = os.path.join(project_root, "config", "collect_args.yaml")
    if os.path.exists(collect_yaml_path):
        with open(collect_yaml_path, 'r') as f:
            c = yaml.safe_load(f)
            return c.get('sim', {}).get('num_cars', 3)
    return 0


# ============================================================
# 3. Dataset Loading and Dimension Calculation
# ============================================================

# Construct filename suffix
suffix = ""
if DATA_CFG['use_mpc_suffix']:
    suffix = "_mpc"

data_filename = f"{DATA_CFG['name']}{suffix}.pkl"
data_path = os.path.join(DATA_DIR, data_filename)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

print(f"Loading data from {data_path}...")
if DATA_CFG['use_small_dataset']:
    data = np.load(data_path, allow_pickle=True)[:241]
else:
    data = np.load(data_path, allow_pickle=True)

# Determine number of cars (Priority: train_args > collect.yaml > Inference)
if 'num_cars' in DATA_CFG:
    num_cars = DATA_CFG['num_cars']
else:
    num_cars = get_num_cars_from_collect_yaml(_GAME_RACER_ROOT)

# Verify dimensions
feats_per_car = 13
expected_input_dim = feats_per_car * num_cars
actual_dim = data.shape[2]

if actual_dim != expected_input_dim:
    raise ValueError(
        f"Dimension mismatch: Calculated input dim ({expected_input_dim}) "
        f"!= Actual data dim ({actual_dim}). "
        f"Please check 'num_cars' ({num_cars}) or data source."
    )

input_dim = actual_dim
print(f"Cars: {num_cars} | Input Dimension: {input_dim}")


# ============================================================
# 4. Determine Model Suffix (for saving and loading)
# ============================================================

S = int(DATA_CFG['horizon_s'])
save_suffix = ""
fs = int(MODEL_CFG['hidden_size']) # 128

if DATA_CFG['use_small_dataset']:
    save_suffix = "_small"

if S < 250:
    save_suffix += "_myopic"

if S < 50:
    save_suffix += "_s"

if DATA_CFG['use_mpc_suffix']:
    save_suffix += "_mpc"


# ============================================================
# 5. Model Initialization (N cars)
# ============================================================

models = []
optimizers = []
loss_fn = nn.MSELoss()
use_cuda = DEVICE_CFG['use_cuda'] and torch.cuda.is_available()

print("Initializing models...")
for k in range(num_cars):
    m = SimpleModel(input_dim, [fs, fs, 64], 1)
    
    # Load pretrained weights (if available)
    load_name = MODEL_CFG['load_model_name']
    if load_name != "none":
        if not load_name.endswith(".pth"): load_name += ".pth"
        full_path = os.path.join(Q_MODELS_DIR, load_name)
        
        # Try loading specific weights for a car (e.g. model_multi0.pth)
        specific_path = full_path.replace(".pth", f"_{k}.pth")
        
        if os.path.exists(specific_path):
            m.load_state_dict(torch.load(specific_path))
            print(f"  [Car {k}] Loaded specific weights: {specific_path}")
        elif os.path.exists(full_path):
            m.load_state_dict(torch.load(full_path))
            print(f"  [Car {k}] Loaded base weights: {full_path}")
    
    if use_cuda:
        m = m.cuda()
    
    m.train()
    models.append(m)
    optimizers.append(torch.optim.Adam(m.parameters(), lr=float(TRAIN_CFG['learning_rate'])))

if use_cuda:
    print("Training on GPU.")
else:
    print("Training on CPU.")


# ============================================================
# 6. Data Preprocessing
# ============================================================

print("Preprocessing Targets (Y)...")
discount_factor = float(TRAIN_CFG['discount_factor'])

Y_list = []
masks_list = []
# Convert to Tensor to avoid repeated conversion in loop
_data_tensor = torch.tensor(data).float()

for k in range(num_cars):
    # Calculate Progress Delta (Car k)
    # data indices 0..num_cars-1 are 's'
    y_k_np = data[:, 1:, k] - data[:, :-1, k]
    y_k_np = wrap(y_k_np)
    
    _Y_k = torch.tensor(y_k_np).float()
    
    # Discounted cumulative future rewards
    Y_k = _Y_k[:, :-S].clone()
    for i in range(1, S):
        Y_k += (discount_factor ** i) * _Y_k[:, i:-S + i]
    
    # Mask to clean abnormal data
    mask_k = (Y_k < 600.) & (Y_k >= -100.)
    
    if use_cuda:
        Y_k = Y_k.cuda()
        mask_k = mask_k.cuda()
        
    Y_list.append(Y_k)
    masks_list.append(mask_k)


print("Preprocessing States (X)...")
X = data[:, :, :].copy()

# Relative position transformation logic (Generalized for N cars)
s0 = data[:, :, 0].copy()

# 1. All non-Ego cars relative to Ego (Car 0)
for k in range(1, num_cars):
    X[:, :, k] -= s0
    X[:, :, k] = wrap(X[:, :, k])

# 2. Replace Ego car's position with (relative distance of last two cars) to maintain relativity
if num_cars >= 2:
    idx_last = num_cars - 1
    idx_second_last = num_cars - 2
    X[:, :, 0] = data[:, :, idx_last] - data[:, :, idx_second_last]
    X[:, :, 0] = wrap(X[:, :, 0])
else:
    X[:, :, 0] = 0.0

X = torch.tensor(X).float()
X[torch.isnan(X)] = 0.0

if use_cuda:
    X = X.cuda()


# ============================================================
# 7. Training Loop
# ============================================================

def save_model(m, name):
    # Use the save_suffix generated based on config
    path = os.path.join(Q_MODELS_DIR, f"{name}{save_suffix}.pth")
    torch.save(m.state_dict(), path)
    return path

n_iters = int(TRAIN_CFG['n_iters'])
batch_size = int(TRAIN_CFG['batch_size'])

print(f"Starting training for {n_iters} iterations...")

try:
    for i in range(n_iters):
        epoch_losses = [0.0] * num_cars

        for j in range(0, X.shape[0], batch_size):
            end = min(j + batch_size, X.shape[0])
            
            # Slicing corresponds to Y's [:-S]
            xb = X[j:end, :-S - 1]

            for k in range(num_cars):
                preds = models[k](xb)
                
                loss = loss_fn(
                    preds.squeeze() * masks_list[k][j:end],
                    Y_list[k][j:end] * masks_list[k][j:end]
                )
                
                optimizers[k].zero_grad()
                loss.backward()
                optimizers[k].step()
                
                epoch_losses[k] += loss.item()

        # Print logs
        if i % 100 == 0:
            loss_str = ", ".join([f"{l:.4f}" for l in epoch_losses])
            print(f"Iter {i}/{n_iters} | Losses: [{loss_str}]")

        # Periodic save
        if i % 1000 == 0 and i > 0:
            for k in range(num_cars):
                save_model(models[k], f"model_multi{k}")

except KeyboardInterrupt:
    print("\nTraining interrupted manually.")

# ============================================================
# 8. Final Save
# ============================================================
print("Saving final models...")
for k in range(num_cars):
    path = save_model(models[k], f"model_multi{k}")
    print(f"Saved: {path}")

print("Done.")