import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

# Constants for default values
SAMPLES_PER_CLIENT = 6000

# CNN builder
import torch.nn as nn

def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(64 * 14 * 14, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 10)
    )

# Data splitting

def split_indices(ds, pi, n=SAMPLES_PER_CLIENT):
    labels = np.array(ds.targets)
    idx = []
    for k in range(10):
        k_idx = np.where(labels == k)[0]
        n_k = int(pi[k] * n)
        sample = np.random.choice(k_idx, n_k, replace=(n_k > len(k_idx)))
        idx.extend(sample.tolist())
    idx = idx[:n]
    if len(idx) < n:
        all_idx = np.arange(len(ds))
        rem = np.setdiff1d(all_idx, idx)
        idx.extend(np.random.choice(rem, n - len(idx), False).tolist())
    return idx

# Test filtering

def filter_test(ds, pi, n=2000):
    labels = np.array(ds.targets)
    idx = []
    for k in range(10):
        k_idx = np.where(labels == k)[0]
        n_k = int(pi[k] * n)
        idx.extend(np.random.choice(k_idx, n_k, replace=(n_k > len(k_idx))).tolist())
    idx = idx[:n]
    if len(idx) < n:
        rem = np.setdiff1d(np.arange(len(ds)), idx)
        idx.extend(np.random.choice(rem, n - len(idx), False).tolist())
    return DataLoader(Subset(ds, idx), batch_size=n)

# Ensemble evaluation

def ensemble_eval(clients, loader):
    total = robust = succ = fail = 0
    agc = sum(c.is_affected for c in clients)
    for Xb, yb in tqdm(loader):
        logits = torch.stack([c.model.eval()(Xb).detach() for c in clients])
        preds = torch.argmax(logits, dim=2).numpy()
        print(preds)
        for i in range(preds.shape[1]):
            votes = preds[:, i]
            non = [votes[j] for j, c in enumerate(clients) if not c.is_affected]
            if non:
                vals, cts = np.unique(non, return_counts=True)
                sorted_cts = np.sort(cts)[::-1]
                if sorted_cts[0] > (sorted_cts[1] if len(sorted_cts) > 1 else 0) + agc:
                    robust += 1; total += 1; continue
            vals, cts = np.unique(votes, return_counts=True)
            tied = vals[cts == cts.max()]
            final = tied[0] if tied.size == 1 else random.choice(tied.tolist())
            total += 1
            if final == yb[i].item(): succ += 1
            else: fail += 1
    return robust/total, succ/total, fail/total
