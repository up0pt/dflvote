import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Literal
import random
from client import Client

# Constants for default values
SAMPLES_PER_CLIENT = 6000
LABEL_NUM = 10

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
    for k in range(LABEL_NUM):
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


def ensemble_eval(
    clients: list[Client],
    loader: DataLoader,
    device: Literal['cuda', 'cpu'] = 'cuda'
) -> tuple[float, float, float]:
    # Ensure clients list is not empty
    if not clients:
        raise AssertionError("clients must be a list and have at least one Client")

    # Determine device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = 'cpu'
    dev = torch.device(device)
    print(f"Evaluating on device: {dev}")

    # Move models to device
    for c in clients:
        c.model.to(dev)
        c.model.eval()

    total = robust = succ = tie = fail = 0.0
    agc = sum(c.is_affected for c in clients)

    for Xb, yb in tqdm(loader):
        # Transfer batch to device
        Xb = Xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)

        # Compute logits per client
        logits = torch.stack([c.model(Xb).detach() for c in clients], dim=0)
        preds = torch.argmax(logits, dim=2).cpu().numpy()  # shape: num_clients x batch_size

        for i in range(preds.shape[1]):
            votes = preds[:, i]
            non_vals = [votes[j] for j, c in enumerate(clients) if not c.is_affected]
            total += 1
            if not non_vals:
                raise NotImplementedError('All clients are affected')

            vals, cts = np.unique(non_vals, return_counts=True)
            tied = vals[cts == cts.max()]
            sorted_cts = np.sort(cts)[::-1]

            true_label = yb[i].item()
            print(tied)
            if true_label not in tied:
                fail += 1
            elif sorted_cts[0] > (sorted_cts[1] if len(sorted_cts) > 1 else 0) + agc and true_label == tied[0]:
                robust += 1
            elif sorted_cts[0] > (sorted_cts[1] if len(sorted_cts) > 1 else 0) and true_label == tied[0]:
                succ += 1
            elif true_label in tied:
                # equals else
                tie += 1
            else:
                raise AssertionError('ensamble logic has unpredicted conditional branch')

    return robust/total, succ/total, tie/total, fail/total