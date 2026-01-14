import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
import sys

# Add project root to path
_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
_GAME_RACER_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

from model_arch import SimpleModel

# ============================================================
# 1. Configuration Loading
# ============================================================

parser = argparse.ArgumentParser()
default_config = os.path.join(_GAME_RACER_ROOT, "config", "train_q_rel_args.yaml")
parser.add_argument("--config", default=default_config, help="Path to config yaml")
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"Config not found: {args.config}")

print(f"Loading config from: {args.config}")
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg['data']
TRAIN_CFG = cfg['training']
MODEL_CFG = cfg['model']
DEVICE_CFG = cfg['device']

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
Q_MODELS_REL_DIR = os.path.join(_PROJECT_ROOT, "q_models_rel")
os.makedirs(Q_MODELS_REL_DIR, exist_ok=True)


# ============================================================
# 2. Helper Functions
# ============================================================

def wrap(x):
    """Handle track cyclic wrapping (~150m)"""
    return (
        (x > 75.) * (x - 150.087)
        + (x < -75.) * (x + 150.087)
        + (x <= 75.) * (x >= -75.) * x
    )

def get_num_cars(project_root, explicit_val=None):
    if explicit_val: return explicit_val
    
    collect_yaml = os.path.join(project_root, "config", "collect_args.yaml")
    if os.path.exists(collect_yaml):
        with open(collect_yaml, 'r') as f:
            c = yaml.safe_load(f)
            return c.get('sim', {}).get('num_cars', 3)
    return 0


# ============================================================
# 3. Dataset Loading
# ============================================================

suffix = ""
if DATA_CFG['use_mpc_suffix']:
    suffix = "_mpc"

data_path = os.path.join(DATA_DIR, f"{DATA_CFG['name']}{suffix}.pkl")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

print(f"Loading data from {data_path}...")
if DATA_CFG['use_small_dataset']:
    data = np.load(data_path, allow_pickle=True)[:241]
else:
    data = np.load(data_path, allow_pickle=True)

# Determine Num Cars
if 'num_cars' in DATA_CFG:
    num_cars = DATA_CFG['num_cars']
else:
    num_cars = get_num_cars(_GAME_RACER_ROOT)

# Infer dimensions
feats_per_car = 13
input_dim = feats_per_car * num_cars
actual_dim = data.shape[2]

if actual_dim != input_dim:
    raise ValueError(
        f"Dimension mismatch: Calculated input dim ({input_dim}) "
        f"!= Actual data dim ({actual_dim}). "
        f"Please check 'num_cars' ({num_cars}) or data source."
    )

print(f"Cars: {num_cars} | Input Dim: {input_dim}")


# ============================================================
# 4. Model Initialization
# ============================================================

save_suffix = ""
S = int(DATA_CFG['horizon_s'])
fs = int(MODEL_CFG['hidden_size'])

if DATA_CFG['use_small_dataset']: save_suffix = "_small"
if S < 250: save_suffix += "_myopic"
if S < 50: save_suffix += "_s"
if DATA_CFG['use_mpc_suffix']: save_suffix += "_mpc"

models = []
optimizers = []
loss_fn = nn.MSELoss()
use_cuda = DEVICE_CFG['use_cuda'] and torch.cuda.is_available()

print("Initializing models...")
for k in range(num_cars):
    m = SimpleModel(input_dim, [fs, fs, 64], 1)
    
    load_name = MODEL_CFG['load_model_name']
    if load_name != "none":
        if not load_name.endswith(".pth"): load_name += ".pth"
        path = os.path.join(Q_MODELS_REL_DIR, load_name)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path))
            print(f"  [Car {k}] Loaded weights: {path}")

    if use_cuda:
        m = m.cuda()
    
    m.train()
    models.append(m)
    optimizers.append(torch.optim.Adam(m.parameters(), lr=float(TRAIN_CFG['learning_rate'])))


# ============================================================
# 5. Data Preprocessing (Generalized Relative Logic)
# ============================================================

print("Preprocessing States (X)...")
X = data[:, :, :].copy()

# 1. Transform State to Relative Space
# Standardize: Everything relative to Car 0
# For indices 0..num_cars-1 (which correspond to 's' positions)
s0 = data[:, :, 0].copy()

# Subtract s0 from all car positions (including s0 itself temporarily)
for k in range(num_cars):
    X[:, :, k] -= s0
    X[:, :, k] = wrap(X[:, :, k])

# Handle X[:,:,0] special case (Consistent with previous logic)
# Original code: X[:,:,0] = s2 - s1 (Relative dist of opponents)
# New logic for N cars: Last Opponent - Second Last Opponent
if num_cars >= 2:
    idx_a = num_cars - 1
    idx_b = num_cars - 2
    # Note: We must calculation this from raw DATA, not modified X, to be safe, 
    # or ensure X logic holds. Raw data is safer.
    X[:, :, 0] = data[:, :, idx_a] - data[:, :, idx_b]
    X[:, :, 0] = wrap(X[:, :, 0])
else:
    X[:, :, 0] = 0.0

X_tensor = torch.tensor(X).float()
X_tensor[torch.isnan(X_tensor)] = 0.0
if use_cuda: X_tensor = X_tensor.cuda()


print("Preprocessing Targets (Y - Relative Advantage)...")
# Logic: Reward is based on (My Position) - (Best Opponent Position)
# Or in the original code's specific 3-car case:
# Y  (Car 0): -(max(s2, s1) - s0)  [Relative deficit against best opponent]
# Y_ (Car 1): -(max(s2, s0) - s1) 
# ... and looking at the change (Delta) of this metric.

Y_list = []
masks_list = []
discount = float(TRAIN_CFG['discount_factor'])

# Convert raw data to tensor for calculation
raw_s = torch.tensor(data[:, :, :num_cars]).float() # [Episodes, Time, Cars]

for k in range(num_cars):
    # 1. Identify Opponents
    opp_indices = [i for i in range(num_cars) if i != k]
    
    if not opp_indices:
        # Single car case
        relative_val = raw_s[:, :, k] * 0.0 
    else:
        # Get positions relative to Car 0 (Standard Frame)
        # Note: Original code used complex relative logic. 
        # Here we simplify: Score = (My S - Best Opponent S)
        
        # Get my position
        my_s = raw_s[:, :, k]
        
        # Easiest way: Compute (Opponent - Me), wrap it, then take Max.
        
        # diffs[i] = Opp_i - Me
        diffs = []
        for opp_idx in opp_indices:
            d = raw_s[:, :, opp_idx] - my_s
            # Apply wrap function (numpy style logic on tensor)
            d = (d > 75.) * (d - 150.087) + (d < -75.) * (d + 150.087) + (d <= 75.) * (d >= -75.) * d
            diffs.append(d)
        
        diffs_tensor = torch.stack(diffs, dim=0) # [Num_Opp, Eps, Time]
        
        # Max diff = How far ahead is the leading opponent?
        # If I am leading, max diff is negative (or small).
        max_diff, _ = torch.max(diffs_tensor, dim=0)
        
        # Original Metric Y = -max(diffs). 
        # i.e. if leader is +10m ahead, max_diff is +10, Y is -10.
        # if I am +10m ahead of everyone, max_diff is -10, Y is +10.
        relative_val = -max_diff

    # 2. Compute Delta (Change in relative value)
    # Y_t = Rel_Val_{t+1} - Rel_Val_{t} (Change in advantage)
    # Original code equation: -max(next) + max(curr) == -(max(next) - max(curr))
    # It essentially measures: "Did I catch up?" or "Did I pull away?"
    
    # Note: Original code logic was:
    # Y = -np.maximum(X[:, 1:, 2], X[:, 1:, 1]) + np.maximum(X[:, :-1, 2], X[:, :-1, 1])
    # Where X indices were relative positions.
    # The logic `relative_val` above captures the snapshot. 
    # The delta is simply:
    
    y_delta = relative_val[:, 1:] - relative_val[:, :-1]
    y_delta = (y_delta > 75.) * (y_delta - 150.087) + (y_delta < -75.) * (y_delta + 150.087) + (y_delta <= 75.) * (y_delta >= -75.) * y_delta
    
    # 3. Discounted Sum
    Y_k = y_delta[:, :-S].clone()
    for i in range(1, S):
        Y_k += (discount ** i) * y_delta[:, i:-S + i]
        
    mask_k = (Y_k < 600.) & (Y_k >= -100.)

    if use_cuda:
        Y_k = Y_k.cuda()
        mask_k = mask_k.cuda()
    
    Y_list.append(Y_k)
    masks_list.append(mask_k)


# ============================================================
# 6. Training Loop
# ============================================================

def save_model(m, name):
    path = os.path.join(Q_MODELS_REL_DIR, f"{name}{save_suffix}.pth")
    torch.save(m.state_dict(), path)
    return path

n_iters = int(TRAIN_CFG['n_iters'])
batch_size = int(TRAIN_CFG['batch_size'])

print(f"Starting training for {n_iters} iterations...")

try:
    for i in range(n_iters):
        losses = [0.0] * num_cars

        for j in range(0, X_tensor.shape[0], batch_size):
            end = min(j + batch_size, X_tensor.shape[0])
            
            xb = X_tensor[j:end, :-S - 1]

            for k in range(num_cars):
                preds = models[k](xb)
                
                loss = loss_fn(
                    preds.squeeze() * masks_list[k][j:end],
                    Y_list[k][j:end] * masks_list[k][j:end]
                )
                
                optimizers[k].zero_grad()
                loss.backward()
                optimizers[k].step()
                
                losses[k] += loss.item()

        if i % 100 == 0:
            loss_str = ", ".join([f"{l:.4f}" for l in losses])
            print(f"Iter {i}/{n_iters} | Losses: [{loss_str}]")

        if i % 1000 == 0 and i > 0:
            for k in range(num_cars):
                save_model(models[k], f"model_multi{k}")

except KeyboardInterrupt:
    print("Interrupted.")

# Final Save
print("Saving final models...")
for k in range(num_cars):
    p = save_model(models[k], f"model_multi{k}")
    print(f"Saved: {p}")