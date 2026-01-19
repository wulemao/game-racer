import os
import numpy as np
import pickle

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

datas = []
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    in_path = os.path.join(DATA_DIR, f"data{i}_multi_mpc.pkl")
    data = np.load(in_path, allow_pickle=True)[:241, :, :]
    print(data.shape)
    datas.append(data)

data_large = np.concatenate(datas, axis=0)
print(data_large.shape)

out_path = os.path.join(DATA_DIR, "data_large_multi_mpc.pkl")
with open(out_path, "wb") as f:
    pickle.dump(data_large, f)
