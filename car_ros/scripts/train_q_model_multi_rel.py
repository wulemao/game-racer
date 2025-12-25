import os
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model_arch import SimpleModel


# ============================================================
# Path configuration
# ============================================================

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))          # project_root/scripts
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                       # project_root

DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
Q_MODELS_REL_DIR = os.path.join(_PROJECT_ROOT, "q_models_rel")

os.makedirs(Q_MODELS_REL_DIR, exist_ok=True)

# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="none",
                    help="Name of the pre-trained model")
parser.add_argument("--data_name", default="none",
                    help="Dataset name (without extension)")
parser.add_argument("--cuda", action="store_true",
                    help="Use CUDA")
parser.add_argument("--small", action="store_true",
                    help="Use small dataset")
parser.add_argument("--s", default=350,
                    help="Discount horizon")
parser.add_argument("--mpc", action="store_true",
                    help="Use MPC dataset")

args = parser.parse_args()


# ============================================================
# Dataset loading
# ============================================================

suffix = ""
if args.mpc:
    suffix = "_mpc"

data_path = os.path.join(
    DATA_DIR,
    f"{args.data_name}{suffix}.pkl"
)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

if args.small:
    data = np.load(data_path, allow_pickle=True)[:241]
else:
    data = np.load(data_path, allow_pickle=True)


# ============================================================
# Hyperparameters
# ============================================================

discount_factor = 1.0
n_iters = 50000
learning_rate = 0.001
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


# ============================================================
# Model initialization
# ============================================================

model = SimpleModel(39, [fs, fs, 64], 1)
model_ = SimpleModel(39, [fs, fs, 64], 1)
model__ = SimpleModel(39, [fs, fs, 64], 1)

if args.model_name != "none":
    model_path = args.model_name
    if not model_path.endswith(".pth"):
        model_path += ".pth"

    if not os.path.isabs(model_path):
        model_path = os.path.join(Q_MODELS_REL_DIR, model_path)

    model.load_state_dict(torch.load(model_path))

if args.cuda:
    model = model.cuda()
    model_ = model_.cuda()
    model__ = model__.cuda()


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer_ = torch.optim.Adam(model_.parameters(), lr=learning_rate)
optimizer__ = torch.optim.Adam(model__.parameters(), lr=learning_rate)


# ============================================================
# Data preprocessing
# ============================================================

X = data[:, :, :].copy()

X[:, :, 0] = data[:, :, 2] - data[:, :, 1]
X[:, :, 1] -= data[:, :, 0]
X[:, :, 2] -= data[:, :, 0]

def wrap(x):
    return (
        (x > 75.) * (x - 150.087) +
        (x < -75.) * (x + 150.087) +
        (x <= 75.) * (x >= -75.) * x
    )

X[:, :, 0] = wrap(X[:, :, 0])
X[:, :, 1] = wrap(X[:, :, 1])
X[:, :, 2] = wrap(X[:, :, 2])

Y = -np.maximum(X[:, 1:, 2], X[:, 1:, 1]) + np.maximum(X[:, :-1, 2], X[:, :-1, 1])
Y_ = -np.maximum(X[:, 1:, 0], -X[:, 1:, 1]) + np.maximum(X[:, :-1, 0], -X[:, :-1, 1])
Y__ = -np.maximum(-X[:, 1:, 2], -X[:, 1:, 0]) + np.maximum(-X[:, :-1, 2], -X[:, :-1, 0])

Y = wrap(Y)
Y_ = wrap(Y_)
Y__ = wrap(Y__)

_Y = torch.tensor(Y).float()
_Y_ = torch.tensor(Y_).float()
_Y__ = torch.tensor(Y__).float()

Y = _Y[:, :-S].clone()
Y_ = _Y_[:, :-S].clone()
Y__ = _Y__[:, :-S].clone()

for i in range(1, S):
    Y += (discount_factor ** i) * _Y[:, i:-S + i]
    Y_ += (discount_factor ** i) * _Y_[:, i:-S + i]
    Y__ += (discount_factor ** i) * _Y__[:, i:-S + i]

mask = (Y < 600.) & (Y >= -100.)
mask_ = (Y_ < 600.) & (Y_ >= -100.)
mask__ = (Y__ < 600.) & (Y__ >= -100.)

X = torch.tensor(X).float()
X[torch.isnan(X)] = 0.0

if args.cuda:
    X = X.cuda()
    Y = Y.cuda()
    Y_ = Y_.cuda()
    Y__ = Y__.cuda()
    mask = mask.cuda()
    mask_ = mask_.cuda()
    mask__ = mask__.cuda()


# ============================================================
# Training
# ============================================================

model.train()
model_.train()
model__.train()

batch_size = 4096


def save_model(m, base_name):
    path = os.path.join(Q_MODELS_REL_DIR, f"{base_name}{suffix}.pth")
    torch.save(m.state_dict(), path)
    return path


for i in range(n_iters):
    total_loss = 0.0
    total_loss_ = 0.0
    total_loss__ = 0.0

    for j in range(0, X.shape[0], batch_size):
        end = min(j + batch_size, X.shape[0])

        xb = X[j:end, :-S - 1]

        preds = model(xb)
        preds_ = model_(xb)
        preds__ = model__(xb)

        loss = loss_fn(
            preds.squeeze() * mask[j:end],
            Y[j:end] * mask[j:end]
        )
        loss_ = loss_fn(
            preds_.squeeze() * mask_[j:end],
            Y_[j:end] * mask_[j:end]
        )
        loss__ = loss_fn(
            preds__.squeeze() * mask__[j:end],
            Y__[j:end] * mask__[j:end]
        )

        optimizer.zero_grad()
        optimizer_.zero_grad()
        optimizer__.zero_grad()

        loss.backward()
        loss_.backward()
        loss__.backward()

        optimizer.step()
        optimizer_.step()
        optimizer__.step()

        total_loss += loss.item()
        total_loss_ += loss_.item()
        total_loss__ += loss__.item()

    print("Iteration:", i, "Loss:", total_loss, total_loss_, total_loss__)

    if i % 1000 == 0:
        save_model(model, "model_multi0")
        save_model(model_, "model_multi1")
        save_model(model__, "model_multi2")


save_model(model, "model_multi0")
save_model(model_, "model_multi1")
save_model(model__, "model_multi2")

