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

from model_arch import SimpleModel

# ============================================================
# 1. Configuration Loading & Path Setup
# ============================================================

parser = argparse.ArgumentParser()
default_config = os.path.join(_PROJECT_ROOT, "config", "train_p_args.yaml")
parser.add_argument("--config", default=default_config, help="Path to config yaml")
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"Config file not found: {args.config}")

print(f"Loading configuration from: {args.config}")
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

DATA_CFG = cfg['data']
TRAIN_CFG = cfg['training']
MODEL_CFG = cfg['model']
DEVICE_CFG = cfg['device']

# Directory setup
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
Q_MODELS_DIR = os.path.join(_PROJECT_ROOT, "q_models")
Q_MODELS_REL_DIR = os.path.join(_PROJECT_ROOT, "q_models_rel")
P_MODELS_DIR = os.path.join(_PROJECT_ROOT, "p_models")
P_MODELS_REL_DIR = os.path.join(_PROJECT_ROOT, "p_models_rel")

os.makedirs(P_MODELS_DIR, exist_ok=True)
os.makedirs(P_MODELS_REL_DIR, exist_ok=True)

# Select directories based on 'use_rel_paths'
use_rel = MODEL_CFG.get('use_rel_paths', False)
Q_DIR = Q_MODELS_REL_DIR if use_rel else Q_MODELS_DIR
P_DIR = P_MODELS_REL_DIR if use_rel else P_MODELS_DIR

device = torch.device("cuda" if DEVICE_CFG['use_cuda'] and torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Dataset Loading & Dimension Inference
# ============================================================

suffix = ""
if DATA_CFG['use_mpc_suffix']: suffix = "_mpc"

data_path = os.path.join(DATA_DIR, f"{DATA_CFG['name']}{suffix}.pkl")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

# Determine load size (Small vs Full)
max_n = 241 if DATA_CFG['use_small_dataset'] else 241 * 9
print(f"Loading data from {data_path} (max entries: {max_n})...")
data = np.load(data_path, allow_pickle=True)[:max_n]

# Infer number of cars
if 'num_cars' in DATA_CFG:
    num_cars = DATA_CFG['num_cars']
else:
    # Attempt to read from collect.yaml or infer default
    collect_yaml = os.path.join(_PROJECT_ROOT, "config", "collect.yaml")
    if os.path.exists(collect_yaml):
        with open(collect_yaml, 'r') as f:
            c = yaml.safe_load(f)
            num_cars = c.get('sim', {}).get('num_cars', 3)
    else:
        num_cars = 0 # Fallback default

# Calculate dimensions
feats_per_car = 13
input_dim = feats_per_car * num_cars
print(f"Configuration: {num_cars} cars | Input Dimension: {input_dim}")

if data.shape[-1] != input_dim:
    raise ValueError(
        f"Dimension mismatch: Calculated input dim ({input_dim}) "
        f"!= Actual data dim ({data.shape[-1]}). "
        f"Please check 'num_cars' ({num_cars}) or data source."
    )


# ============================================================
# 3. Model Initialization (Teachers & Potential P)
# ============================================================

S = int(DATA_CFG['horizon_s'])
S_ = 500 - S  # Valid range for training P
fs = int(MODEL_CFG['hidden_size'])

# Determine suffix for saving models
save_suffix = ""
if DATA_CFG['use_small_dataset']: save_suffix = "_small"
if S < 250: save_suffix += "_myopic"
if S < 50: save_suffix += "_s"
if DATA_CFG['use_mpc_suffix']: save_suffix += "_mpc"

# --- Load Teacher Models (Q) Dynamically ---
teacher_models = []
teacher_base_name = MODEL_CFG['teacher_base_name'] # e.g. "model_multi"

print(f"Loading {num_cars} Teacher Models (Base: {teacher_base_name})...")

for k in range(num_cars):
    m = SimpleModel(input_dim, [fs, fs, 64], 1).to(device)
    
    # Try exact name first, then with suffix
    filename_exact = f"{teacher_base_name}{k}.pth"
    filename_suffixed = f"{teacher_base_name}{k}{save_suffix}.pth"
    
    path_candidates = [
        os.path.join(Q_DIR, filename_exact),
        os.path.join(Q_DIR, filename_suffixed)
    ]
    
    loaded = False
    for p in path_candidates:
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, map_location=device))
            print(f"  [Car {k}] Loaded: {p}")
            loaded = True
            break
    
    if not loaded:
        print(f"  [Warning] Teacher for Car {k} not found. Initialized randomly.")

    # Freeze teachers
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    teacher_models.append(m)


# --- Initialize Potential Model (P) ---
# Width multiplier W = num_cars to accommodate weights from all teachers
W = num_cars 
model_p = SimpleModel(input_dim, [W*fs, W*fs, W*64], 1).to(device)

load_p_name = MODEL_CFG['load_p_model_name']

if load_p_name == "none":
    print("Initializing P-Model from Teachers (Block-wise copy)...")
    
    # 1. Zero out existing weights
    for layer_idx in [0, 2, 4, 6]:
        model_p.fc[layer_idx].weight.data.zero_()
        model_p.fc[layer_idx].bias.data.zero_()

    # 2. Block Copy Logic
    # Copy weights from N teachers into N diagonal blocks of P
    
    # Layer 0 (Input -> Hidden1)
    for k in range(num_cars):
        w_t = teacher_models[k].fc[0].weight.data
        b_t = teacher_models[k].fc[0].bias.data
        model_p.fc[0].weight.data[k*fs : (k+1)*fs, :] = w_t
        model_p.fc[0].bias.data[k*fs : (k+1)*fs] = b_t

    # Layer 2 (Hidden1 -> Hidden2) - Block Diagonal
    for k in range(num_cars):
        w_t = teacher_models[k].fc[2].weight.data
        b_t = teacher_models[k].fc[2].bias.data
        model_p.fc[2].weight.data[k*fs : (k+1)*fs, k*fs : (k+1)*fs] = w_t
        model_p.fc[2].bias.data[k*fs : (k+1)*fs] = b_t

    # Layer 4 (Hidden2 -> Latent) - Block Diagonal
    # Teacher latent=64, P latent=W*64
    for k in range(num_cars):
        w_t = teacher_models[k].fc[4].weight.data # [64, fs]
        b_t = teacher_models[k].fc[4].bias.data   # [64]
        model_p.fc[4].weight.data[k*64 : (k+1)*64, k*fs : (k+1)*fs] = w_t
        model_p.fc[4].bias.data[k*64 : (k+1)*64] = b_t

    # Layer 6 (Latent -> Output Scalar) - Summation
    # The output of P is a scalar that roughly sums the Q values
    for k in range(num_cars):
        w_t = teacher_models[k].fc[6].weight.data # [1, 64]
        b_t = teacher_models[k].fc[6].bias.data   # [1]
        model_p.fc[6].weight.data[:, k*64 : (k+1)*64] = w_t
        model_p.fc[6].bias.data += b_t

    # BatchNorm Stats (Copy from Teacher 0 as baseline)
    model_p.batch_norm.running_mean = teacher_models[0].batch_norm.running_mean
    model_p.batch_norm.running_var = teacher_models[0].batch_norm.running_var
    model_p.batch_norm.momentum = 0.0

else:
    # Load existing P
    p_path = os.path.join(P_DIR, f"{load_p_name}.pth")
    if os.path.exists(p_path):
        model_p.load_state_dict(torch.load(p_path, map_location=device))
        print(f"Loaded P-Model: {p_path}")
    else:
        print(f"Warning: P-Model {p_path} not found. Initialized randomly.")

# Force BN to eval mode (as usually done in transfer learning/fine-tuning here)
model_p.batch_norm.eval()


# ============================================================
# 4. Data Preparation (Relative Features)
# ============================================================

def wrap_np(x):
    # Wrap track length approx 150.087
    return (
        (x > 75.0) * (x - 150.087)
        + (x < -75.0) * (x + 150.087)
        + ((x <= 75.0) & (x >= -75.0)) * x
    )

print("Preprocessing data features...")
X = data[:, :, :].copy()

# Generalized Relative Logic for N cars
# Indices 0 to num_cars-1 represent 's' positions.
s0 = data[:, :, 0].copy()

# 1. Make all opponents (1..N-1) relative to Ego (0)
for k in range(1, num_cars):
    X[:, :, k] = wrap_np(data[:, :, k] - s0)

# 2. Ego relative logic: Last Opponent - Second Last Opponent
if num_cars >= 2:
    idx_a = num_cars - 1
    idx_b = num_cars - 2
    X[:, :, 0] = wrap_np(data[:, :, idx_a] - data[:, :, idx_b])
else:
    X[:, :, 0] = 0.0

X = torch.tensor(X, dtype=torch.float32, device=device)
X[torch.isnan(X)] = 0.0

# Counterfactual X_ (Shift environment)
# Env part is everything EXCEPT the parameters (last 5 * num_cars dims)
param_dim = 5 * num_cars
X_ = X.clone()
if X.shape[0] >= 2:
    # Shift env features by 1 time step
    X_[1:, :, :-param_dim] = X[:-1, :, :-param_dim]


# ============================================================
# 5. Training Loop
# ============================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model_p.fc.parameters(), lr=float(TRAIN_CFG['learning_rate']))
batch_size = int(TRAIN_CFG['batch_size'])
n_iters = int(TRAIN_CFG['n_iters'])
warmup_iters = int(TRAIN_CFG['warmup_iters'])

# --- Warmup Phase ---
# Train P(s) to approximate Sum(Q_k(s))
print(f"Starting Warmup ({warmup_iters} iterations)...")
model_p.train()
for it in range(warmup_iters):
    total = 0.0
    for j in range(0, X.shape[0], batch_size):
        if j + batch_size > X.shape[0]: break
        
        xb = X[j : j+batch_size]
        with torch.no_grad():
            yb = sum([m(xb) for m in teacher_models])
        
        pred = model_p(xb)
        loss = loss_fn(pred, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    
    if it % 50 == 0: print(f"  Warmup Iter {it}: Loss {total:.4f}")

model_p.eval()

# --- Main Training Phase (Regret Matching) ---
print(f"Starting Main Training ({n_iters} iterations)...")
model_p.train()
min_loss = float("inf")

try:
    for i in range(n_iters):
        total_loss = 0.0
        
        # Sequential batching loop
        for j in range(0, X.shape[0] - batch_size, batch_size):
            # Define current (t) and counterfactual (t') indices
            idx_curr = slice(j, j + batch_size)
            idx_next = slice(j + 1, j + batch_size + 1)
            
            if idx_next.stop > X.shape[0]: break

            x_curr = X[idx_curr]
            x_counter = X_[idx_next]
            
            # 1. Calculate Ground Truth Regret from Teachers
            # Regret_k = Q_k(s) - Q_k(s')
            regrets = []
            with torch.no_grad():
                for k in range(num_cars):
                    v_curr = teacher_models[k](x_curr)
                    v_next = teacher_models[k](x_counter)
                    regrets.append(v_curr - v_next)
            
            # 2. Calculate Potential Model Difference
            # P_diff = P(s) - P(s')
            p_curr = model_p(x_curr)
            p_next = model_p(x_counter)
            p_diff = p_curr - p_next
            
            # 3. Calculate Loss (Sum of MSEs)
            # Minimizing divergence from the regret of all players
            loss_sum = 0
            for k in range(num_cars):
                # Only consider valid time horizon S_
                l = loss_fn(p_diff[:, :S_], regrets[k][:, :S_])
                loss_sum += l
                
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            
            total_loss += loss_sum.item()

        # Save best model
        if total_loss < min_loss:
            min_loss = total_loss
            save_path = os.path.join(P_DIR, f"model_multi{save_suffix}.pth")
            torch.save(model_p.state_dict(), save_path)

        if i % 100 == 0:
            print(f"Iter {i} | Total Loss {total_loss:.6f}")

except KeyboardInterrupt:
    print("\nTraining interrupted manually.")


# ============================================================
# 6. Save Regret Info (Restored & Generalized)
# ============================================================

print("Collecting final regret statistics...")

# Containers for N cars
final_diffs = [[] for _ in range(num_cars)]
max_gts = [0.0] * num_cars

# Run one pass without gradients to collect stats
model_p.eval()
with torch.no_grad():
    for j in range(0, X.shape[0] - batch_size, batch_size):
        idx_curr = slice(j, j + batch_size)
        idx_next = slice(j + 1, j + batch_size + 1)
        if idx_next.stop > X.shape[0]: break

        x_curr = X[idx_curr]
        x_counter = X_[idx_next]

        # Potential Difference
        p_diff = model_p(x_curr) - model_p(x_counter)

        for k in range(num_cars):
            # GT Regret
            v_curr = teacher_models[k](x_curr)
            v_next = teacher_models[k](x_counter)
            gt_k = v_curr - v_next
            
            # Difference = Predicted Regret - GT Regret
            diff_k = p_diff - gt_k
            
            # Slice to valid horizon
            valid_diff = diff_k[:, :S_]
            valid_gt = gt_k[:, :S_]
            
            final_diffs[k].append(valid_diff.detach().cpu())
            
            # Track max ground truth value for normalization/analysis
            current_max = torch.max(torch.abs(valid_gt)).item()
            if current_max > max_gts[k]:
                max_gts[k] = current_max

# Aggregate and Save
regret_info = {}
for k in range(num_cars):
    # Dynamic keys: losses1, losses2... max_gt1, max_gt2...
    regret_info[f"losses{k+1}"] = final_diffs[k] 
    regret_info[f"max_gt{k+1}"] = max_gts[k]

regret_info["min_loss"] = min_loss

out_dir = P_DIR
save_name = f"regret_info_multi{save_suffix}.pkl"
save_path = os.path.join(out_dir, save_name)

with open(save_path, "wb") as f:
    pickle.dump(regret_info, f)

print(f"Training finished. Model saved to {P_DIR}")
print(f"Regret info saved to: {save_path}")