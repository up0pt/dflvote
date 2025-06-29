import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
from client import Client

# Constants for default values
SAMPLES_PER_CLIENT = 6000

# CNN builder
import torch.nn as nn

# Data splitting

def split_indices(ds, pi, n=SAMPLES_PER_CLIENT):
    # Avoid Error: Convert ds.targets to a list and then to a numpy array, whether it's a Tensor or a list
    if hasattr(ds.targets, "tolist"):
        labels_list = ds.targets.tolist()
    else:
        labels_list = list(ds.targets)
    labels = np.array(labels_list)
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
    # Avoid Error: Convert ds.targets to a list and then to a numpy array, whether it's a Tensor or a list
    if hasattr(ds.targets, "tolist"):
        labels_list = ds.targets.tolist()
    else:
        labels_list = list(ds.targets)
    labels = np.array(labels_list)
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

def ensemble_eval(clients: list[Client], loader: DataLoader):
    total = robust = succ = fail = 0
    agc = sum(c.is_affected for c in clients)
    for Xb, yb in tqdm(loader):
        logits = torch.stack([c.model.eval()(Xb).detach() for c in clients])
        preds = torch.argmax(logits, dim=2).numpy()
        print(preds)
        for i in range(preds.shape[1]):
            votes = preds[:, i]
            non = [votes[j] for j, c in enumerate(clients) if not c.is_affected]
            total += 1
            if not non:
                assert NotImplementedError('all clients are affected')
            vals, cts = np.unique(non, return_counts=True)
            tied = vals[cts == cts.max()]
            sorted_cts = np.sort(cts)[::-1]
            if yb[i].item not in tied:
                fail += 1
                continue
            elif sorted_cts[0] > (sorted_cts[1] if len(sorted_cts) > 1 else 0) + agc:
                robust += 1.0/tied.size
                continue
            else: 
                succ += 1
    return robust/total, succ/total, fail/total
