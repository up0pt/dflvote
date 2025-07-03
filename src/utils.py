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
    ensamble_clients: list[Client],
    all_clients: dict[int, Client],
    num_all_clients: int, 
    loader: DataLoader,
    max_attackers: int,
    num_trials: int = 10,
    device: Literal['cuda', 'cpu'] = 'cuda',
) -> dict[int, tuple[float, float, float, float, float]]:
    # Ensure clients list is not empty
    if not ensamble_clients:
        raise AssertionError("ensamble clients must be a list and have at least one Client")

    # Determine device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = 'cpu'
    dev = torch.device(device)
    print(f"Evaluating on device: {dev}")

    # Move models to device
    for c in ensamble_clients:
        c.model.to(dev)
        c.model.eval()

    # Precompute all predictions for all clients and all samples
    all_preds = []
    all_labels = []
    for Xb, yb in loader:
        Xb = Xb.to(dev, non_blocking=True)
        logits = torch.stack([c.model(Xb).detach() for c in ensamble_clients], dim=0)
        preds = torch.argmax(logits, dim=2).cpu().numpy()  # shape: num_clients x batch_size
        all_preds.append(preds)
        all_labels.append(yb.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=1)  # shape: num_clients x total_samples
    all_labels = np.concatenate(all_labels, axis=0)  # shape: total_samples

    num_clients, total_samples = all_preds.shape

    group_ids = [c.get_group_id() for _, c in all_clients.items()]

    results = {}

    random.seed(42) #INFO: to keep the same attackes between target clients
    for n_attacker in range(1, max_attackers+1):
        rob_sum = succ_sum = tie_sum = fail_sum = all_affected_sum = 0.0
        for trial in range(num_trials):
            attacker_ids = set(random.sample(range(num_all_clients), n_attacker))
            # Find all group_ids that have attackers
            attacker_groups = set(group_ids[aid] for aid in attacker_ids)
            if len(attacker_groups) > n_attacker:
                raise AssertionError('Wrong Logic')
            # All clients in these groups are affected
            affected_ids = set(idx for idx, gid in enumerate(group_ids) if gid in attacker_groups)
            print(f"attackers_id: {attacker_ids}, attacker_groups: {attacker_groups}, affected_ids: {affected_ids}")
            robust = succ = tie = fail = all_affected = 0.0
            for i in range(total_samples):
                votes = all_preds[:, i]
                non_vals = [votes[j] for j in range(num_clients) if ensamble_clients[j].id not in affected_ids]
                if not non_vals:
                    all_affected += 1
                    continue
                vals, cts = np.unique(non_vals, return_counts=True)
                tied = vals[cts == cts.max()]
                sorted_cts = np.sort(cts)[::-1]
                true_label = all_labels[i]
                agc = len(affected_ids)
                if true_label not in tied:
                    fail += 1
                elif sorted_cts[0] > (sorted_cts[1] if len(sorted_cts) > 1 else 0) + agc and true_label == tied[0]:
                    robust += 1
                elif sorted_cts[0] > (sorted_cts[1] if len(sorted_cts) > 1 else 0) and true_label == tied[0]:
                    succ += 1
                elif true_label in tied:
                    tie += 1
                else:
                    raise AssertionError('ensemble logic has unpredicted conditional branch')
            total = robust + succ + tie + fail + all_affected
            if total == 0:
                continue
            rob_sum += robust / total
            succ_sum += succ / total
            tie_sum += tie / total
            fail_sum += fail / total
            all_affected_sum += all_affected / total
        results[n_attacker] = (
            rob_sum / num_trials,
            succ_sum / num_trials,
            tie_sum / num_trials,
            fail_sum / num_trials,
            all_affected_sum / num_trials
        )
    return results