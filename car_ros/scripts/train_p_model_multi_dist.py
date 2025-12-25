import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse

from model_arch import SimpleModel


# ============================================================
# Path configuration
# ============================================================

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
Q_MODELS_DIR = os.path.join(_PROJECT_ROOT, "q_models")
Q_MODELS_REL_DIR = os.path.join(_PROJECT_ROOT, "q_models_rel")
P_MODELS_DIR = os.path.join(_PROJECT_ROOT, "p_models")
P_MODELS_REL_DIR = os.path.join(_PROJECT_ROOT, "p_models_rel")

os.makedirs(P_MODELS_DIR, exist_ok=True)
os.makedirs(P_MODELS_REL_DIR, exist_ok=True)


# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--model_name1", default="none")
parser.add_argument("--model_name2", default="none")
parser.add_argument("--model_name3", default="none")
parser.add_argument("--model_name", default="none")  # optional pre-trained potential model
parser.add_argument("--data_name", default="none")
parser.add_argument("--batch_size", default=241, type=int)
parser.add_argument("--small", action="store_true")
parser.add_argument("--rel", action="store_true")
parser.add_argument("--mpc", action="store_true")
parser.add_argument("--s", default=350, type=int)  # horizon parameter S
parser.add_argument("--cuda", action="store_true")

# training knobs (optional override)
parser.add_argument("--n_iters", default=100000, type=int)
parser.add_argument("--warmup_iters", default=200, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--discount", default=0.98, type=float)

args = parser.parse_args()


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset loading
# ============================================================

data_name = args.data_name + ("_mpc" if args.mpc else "")
data_path = os.path.join(DATA_DIR, f"{data_name}.pkl")
if not os.path.exists(data_path):
    raise FileNotFoundError(data_path)

# original logic: small -> 241, else -> 241*9
max_n = 241 if args.small else 241 * 9
data = np.load(data_path, allow_pickle=True)[:max_n]

# Expect data = (N,T,39)
if not (isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[-1] == 39):
    raise ValueError(f"Expected data shape (N,T,39), got {getattr(data,'shape',None)}")


# ============================================================
# Hyperparameters and suffix
# ============================================================

S = int(args.s)
suffix = ""
fs = 128

if args.small:
    suffix = "_small"
if S < 250:
    suffix += "_myopic"
if S < 50:
    suffix += "_s"
if args.mpc:
    suffix += "_mpc"

discount_factor = float(args.discount)
n_iters = int(args.n_iters)
warmup_iters = int(args.warmup_iters)
learning_rate = float(args.lr)

# The code uses S_ = 500 - S (same as your original)
S_ = 500 - S
if S_ <= 0:
    raise ValueError(f"S_ = 500 - S must be > 0, got S={S}, S_={S_}")


# ============================================================
# Model initialization
# ============================================================

model = SimpleModel(39, [fs, fs, 64], 1).to(device)
model_ = SimpleModel(39, [fs, fs, 64], 1).to(device)
model__ = SimpleModel(39, [fs, fs, 64], 1).to(device)
model_p = SimpleModel(39, [3 * fs, 3 * fs, 3 * 64], 1).to(device)


def load_model(m, base_name, directory, map_location):
    path = os.path.join(directory, f"{base_name}{suffix}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sd = torch.load(path, map_location=map_location)
    m.load_state_dict(sd)


# Load teacher models
q_dir = Q_MODELS_REL_DIR if args.rel else Q_MODELS_DIR
load_model(model, args.model_name1, q_dir, map_location=device)
load_model(model_, args.model_name2, q_dir, map_location=device)
load_model(model__, args.model_name3, q_dir, map_location=device)

# Freeze teachers
for m in (model, model_, model__):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


# ============================================================
# Initialize / load potential model_p
# ============================================================

if args.model_name == "none":
    # Zero out the Linear layers (indices 0,2,4,6 in your Sequential)
    for layer in [0, 2, 4, 6]:
        model_p.fc[layer].weight.data.zero_()
        model_p.fc[layer].bias.data.zero_()

    # Block-wise init: model_p(x) ≈ model(x)+model_(x)+model__(x)
    # Layer 0: stack weights/biases
    model_p.fc[0].weight.data[:fs, :] = model.fc[0].weight.data
    model_p.fc[0].weight.data[fs:2 * fs, :] = model_.fc[0].weight.data
    model_p.fc[0].weight.data[2 * fs:, :] = model__.fc[0].weight.data

    model_p.fc[0].bias.data[:fs] = model.fc[0].bias.data
    model_p.fc[0].bias.data[fs:2 * fs] = model_.fc[0].bias.data
    model_p.fc[0].bias.data[2 * fs:] = model__.fc[0].bias.data

    # Layer 2: block diagonal
    model_p.fc[2].weight.data[:fs, :fs] = model.fc[2].weight.data
    model_p.fc[2].weight.data[fs:2 * fs, fs:2 * fs] = model_.fc[2].weight.data
    model_p.fc[2].weight.data[2 * fs:, 2 * fs:] = model__.fc[2].weight.data

    model_p.fc[2].bias.data[:fs] = model.fc[2].bias.data
    model_p.fc[2].bias.data[fs:2 * fs] = model_.fc[2].bias.data
    model_p.fc[2].bias.data[2 * fs:] = model__.fc[2].bias.data

    # Layer 4: block diagonal but output dims are 3*64
    model_p.fc[4].weight.data[:64, :fs] = model.fc[4].weight.data
    model_p.fc[4].weight.data[64:2 * 64, fs:2 * fs] = model_.fc[4].weight.data
    model_p.fc[4].weight.data[2 * 64:, 2 * fs:] = model__.fc[4].weight.data

    model_p.fc[4].bias.data[:64] = model.fc[4].bias.data
    model_p.fc[4].bias.data[64:2 * 64] = model_.fc[4].bias.data
    model_p.fc[4].bias.data[2 * 64:] = model__.fc[4].bias.data

    # Layer 6: sum three heads into 1 output
    model_p.fc[6].weight.data[:, :64] = model.fc[6].weight.data
    model_p.fc[6].weight.data[:, 64:2 * 64] = model_.fc[6].weight.data
    model_p.fc[6].weight.data[:, 2 * 64:] = model__.fc[6].weight.data
    model_p.fc[6].bias.data = model.fc[6].bias.data + model_.fc[6].bias.data + model__.fc[6].bias.data

    # Copy BN running stats and freeze them effectively
    model_p.batch_norm.running_mean = model.batch_norm.running_mean
    model_p.batch_norm.running_var = model.batch_norm.running_var
    model_p.batch_norm.momentum = 0.0

else:
    # Load existing potential model
    pot_dir = P_MODELS_REL_DIR if args.rel else P_MODELS_DIR
    pot_path = os.path.join(pot_dir, f"{args.model_name}.pth")
    if not os.path.exists(pot_path):
        raise FileNotFoundError(pot_path)
    model_p.load_state_dict(torch.load(pot_path, map_location=device))

# For safety: keep BN in eval mode even if we later call train()
model_p.batch_norm.eval()


# ============================================================
# Training data preparation
# ============================================================

X = data[:, :, :].copy()

# relative s features: 0,1,2
X[:, :, 0] = data[:, :, 2] - data[:, :, 1]  # opp1 - opp
X[:, :, 1] = data[:, :, 1] - data[:, :, 0]  # opp - ego
X[:, :, 2] = data[:, :, 2] - data[:, :, 0]  # opp1 - ego


def wrap_np(x):
    # keep values in [-75,75] using track length ~150.087
    return (
        (x > 75.0) * (x - 150.087)
        + (x < -75.0) * (x + 150.087)
        + ((x <= 75.0) & (x >= -75.0)) * x
    )


X[:, :, 0] = wrap_np(X[:, :, 0])
X[:, :, 1] = wrap_np(X[:, :, 1])
X[:, :, 2] = wrap_np(X[:, :, 2])

X = torch.tensor(X, dtype=torch.float32, device=device)
X[torch.isnan(X)] = 0.0

# Counterfactual input X_: shift env-part (all but last 15 dims) by one sample
X_ = X.clone()
if X.shape[0] >= 2:
    X_[1:, :, :-15] = X[:-1, :, :-15]
# X_[0] keeps its own env part (no previous sample)


# ============================================================
# Training setup
# ============================================================

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model_p.fc.parameters(), lr=learning_rate)

batch_size = int(args.batch_size)


# ============================================================
# Warm-up: align model_p(x) with sum of teacher values
# (Highly recommended for stability / equivalence with original script)
# ============================================================

model_p.train()
for it in range(warmup_iters):
    total = 0.0
    for j in range(0, X.shape[0], batch_size):
        if j + batch_size > X.shape[0]:
            break

        xb = X[j:j + batch_size]
        with torch.no_grad():
            yb = model(xb) + model_(xb) + model__(xb)

        pred = model_p(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    if it % 20 == 0:
        print(f"[Warmup] iter={it} loss={total:.6f}")

model_p.eval()


# ============================================================
# Main training (regret learning)
# ============================================================

losses1, losses2, losses3 = [], [], []
max_gt1 = max_gt2 = max_gt3 = 0.0
min_loss = float("inf")

model_p.train()

for i in range(n_iters):
    total_loss = 0.0

    for j in range(0, X.shape[0], args.batch_size):
        if j + args.batch_size > X.shape[0]:
            break

        with torch.no_grad():
            v1 = model(X[j+2:j+args.batch_size-1:3])
            v2 = model_(X[j:j+args.batch_size-1:3])
            v3 = model__(X[j+1:j+args.batch_size-1:3])

            v1_ = model(X_[j+3:j+args.batch_size:3])
            v2_ = model_(X_[j+1:j+args.batch_size:3])
            v3_ = model__(X_[j+2:j+args.batch_size:3])

            gt  = v2 - v2_
            gt_ = v3 - v3_
            gt__= v1 - v1_

        preds  = model_p(X[j:j+args.batch_size-1:3]) - model_p(X_[j+1:j+args.batch_size:3])
        preds_ = model_p(X[j+1:j+args.batch_size-1:3]) - model_p(X_[j+2:j+args.batch_size:3])
        preds__= model_p(X[j+2:j+args.batch_size-1:3]) - model_p(X_[j+3:j+args.batch_size:3])

        loss  = loss_fn(preds[:, :S_], gt[:, :S_])
        loss_ = loss_fn(preds_[:, :S_], gt_[:, :S_])
        loss__= loss_fn(preds__[:, :S_], gt__[:, :S_])

        optimizer.zero_grad()
        (loss + loss_ + loss__).backward()
        optimizer.step()

        total_loss += loss.item() + loss_.item() + loss__.item()

        if i == n_iters - 1:
            losses1.append((preds[:, :S_] - gt[:, :S_]).detach().cpu())
            losses2.append((preds_[:, :S_] - gt_[:, :S_]).detach().cpu())
            losses3.append((preds__[:, :S_] - gt__[:, :S_]).detach().cpu())

            max_gt1 = max(max_gt1, torch.max(torch.abs(gt[:, :S_])).item())
            max_gt2 = max(max_gt2, torch.max(torch.abs(gt_[:, :S_])).item())
            max_gt3 = max(max_gt3, torch.max(torch.abs(gt__[:, :S_])).item())

    if total_loss < min_loss:
        min_loss = total_loss
        out_dir = P_MODELS_REL_DIR if args.rel else P_MODELS_DIR
        torch.save(model_p.state_dict(),
                   os.path.join(out_dir, f"model_multi{suffix}.pth"))

    print(f"Iter {i} | Loss {total_loss:.6f}")

# ============================================================
# Save regret info
# ============================================================

regret_info = {
    "losses1": losses1,
    "losses2": losses2,
    "losses3": losses3,
    "max_gt1": max_gt1,
    "max_gt2": max_gt2,
    "max_gt3": max_gt3,
    "min_loss": min_loss
}
out_dir = P_MODELS_REL_DIR if args.rel else P_MODELS_DIR
with open(os.path.join(out_dir, f"regret_info_multi1{suffix}.pkl"), "wb") as f:
    pickle.dump(regret_info, f)

print("Training finished. Regret info saved.")
