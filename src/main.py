import argparse
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Decentralized FL with configurable attackers')
parser.add_argument('--num_clients', type=int, default=8, help='Total number of clients')
parser.add_argument('--num_attackers', type=int, default=2, help='Number of attacker clients')
parser.add_argument('--attackers', type=int, nargs='*', help='Specific attacker client IDs (overrides random)')
parser.add_argument('--groups', type=str, help='Client groups as a Python list of lists, e.g. "[[1,3,5,7],[2,4,6,8]]"')
args = parser.parse_args()

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Model builder
def create_model(input_dim, hidden_dim=128):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 10)
    )

# Dirichlet distributions for two client groups
alpha = [0.5] * 10
torch_dist1 = np.random.dirichlet(alpha)
torch_dist2 = np.random.dirichlet(alpha)

# Configuration
D_NUM_CLIENTS = args.num_clients
SAMPLES_PER_CLIENT = 6000
INPUT_DIM = 28 * 28
BATCH_SIZE = 128
GLOBAL_EPOCHS = 5
LEARNING_RATE = 0.01
# Default groups: split first half and second half
default_groups = [list(range(1, D_NUM_CLIENTS//2+1)), list(range(D_NUM_CLIENTS//2+1, D_NUM_CLIENTS+1))]
# Allow override via CLI argument --groups

# parse groups after args
if args.groups:
    try:
        GROUPS = ast.literal_eval(args.groups)
    except Exception as e:
        print(f"Error parsing groups argument: {e}")
        GROUPS = default_groups
else:
    GROUPS = default_groups

# Load MNIST
tf = transforms.ToTensor()
mnist_train = datasets.MNIST(root='.', train=True, download=True, transform=tf)
mnist_test = datasets.MNIST(root='.', train=False, download=True, transform=tf)

# Helper: non-IID indices by Dirichlet
def create_non_iid_indices(dataset, pi):
    labels = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else np.array(dataset.targets)
    idx = []
    for k in range(10):
        k_idx = np.where(labels == k)[0]
        n_k = int(pi[k] * SAMPLES_PER_CLIENT)
        replace_flag = n_k > len(k_idx)
        if len(k_idx) > 0:
            sampled = np.random.choice(k_idx, n_k, replace=replace_flag)
            idx.extend(sampled.tolist())
    if len(idx) < SAMPLES_PER_CLIENT:
        all_idx = np.arange(len(dataset))
        remaining = np.setdiff1d(all_idx, idx)
        need = SAMPLES_PER_CLIENT - len(idx)
        add = np.random.choice(remaining, need, replace=False)
        idx.extend(add.tolist())
    elif len(idx) > SAMPLES_PER_CLIENT:
        idx = idx[:SAMPLES_PER_CLIENT]
    return idx

# Initialize clients
d_clients = {}
for i in range(1, D_NUM_CLIENTS+1):
    dist = torch_dist1 if i <= D_NUM_CLIENTS//2 else torch_dist2
    indices = create_non_iid_indices(mnist_train, dist)
    loader = DataLoader(Subset(mnist_train, indices), batch_size=BATCH_SIZE, shuffle=True)
    d_clients[i] = {'model': create_model(INPUT_DIM), 'loader': loader, 'is_mal': False}

# Designate attackers
def designate_attackers(client_ids, num_attackers, specific=None):
    if specific:
        attackers = [aid for aid in specific if aid in client_ids]
        if len(attackers) < num_attackers:
            remaining = list(set(client_ids) - set(attackers))
            attackers += random.sample(remaining, num_attackers - len(attackers))
    else:
        attackers = random.sample(client_ids, num_attackers)
    for aid in attackers:
        d_clients[aid]['is_mal'] = True
    return attackers

client_ids = list(d_clients.keys())
attackers = designate_attackers(client_ids, args.num_attackers, args.attackers)
print(f"Attackers: {attackers}")

# DFL training per group
for gid, group in enumerate(GROUPS, start=1):
    print(f"Training group {gid}: Clients {group}")
    group_clients = [d_clients[i] for i in group]
    for epoch in range(GLOBAL_EPOCHS):
        for c in group_clients:
            optimizer = optim.SGD(c['model'].parameters(), lr=LEARNING_RATE)
            c['model'].train()
            for Xb, yb in c['loader']:
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(c['model'](Xb), yb)
                loss.backward()
                optimizer.step()
        avg_state = {}
        for k in group_clients[0]['model'].state_dict():
            avg_state[k] = torch.mean(torch.stack([c['model'].state_dict()[k].float() for c in group_clients]), dim=0)
        for c in group_clients:
            c['model'].load_state_dict(avg_state)
    torch.save(avg_state, f"group_{gid}_model.pth")
    affected = any(c['is_mal'] for c in group_clients)
    for c in group_clients:
        c['is_affected'] = affected

# Select one model per group for ensemble
def select_models(groups):
    return [d_clients[random.choice(g)]['model'] for g in groups]

selected_models = select_models(GROUPS)

# Prepare test loaders by distribution
def filter_test_by_dist(dataset, pi, n_samples=2000):
    labels = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else np.array(dataset.targets)
    idx = []
    for k in range(10):
        k_idx = np.where(labels == k)[0]
        n_k = int(pi[k] * n_samples)
        replace_flag = n_k > len(k_idx)
        if len(k_idx) > 0:
            sampled = np.random.choice(k_idx, n_k, replace=replace_flag)
            idx.extend(sampled.tolist())
    if len(idx) < n_samples:
        all_idx = np.arange(len(dataset))
        remaining = np.setdiff1d(all_idx, idx)
        need = n_samples - len(idx)
        add = np.random.choice(remaining, need, replace=False)
        idx.extend(add.tolist())
    elif len(idx) > n_samples:
        idx = idx[:n_samples]
    return DataLoader(Subset(dataset, idx), batch_size=n_samples)

loader1 = filter_test_by_dist(mnist_test, torch_dist1)
loader2 = filter_test_by_dist(mnist_test, torch_dist2)
loader_avg = filter_test_by_dist(mnist_test, np.ones(10)/10)

# Ensemble evaluation with tie-breaking

# Ensemble evaluation with three-category outputs
def ensemble_eval(models, loader):
    total = robust = succ_if_noattack = fail = 0
    attacker_group_count = sum(getattr(m, 'is_affected', False) for m in models)
    for Xb, yb in loader:
        logits = torch.stack([m.eval()(Xb).detach() for m in models])  # (M,B,10)
        preds = torch.argmax(logits, dim=2).numpy()  # (M,B)
        for i in range(preds.shape[1]):
            votes = preds[:, i]
            # non-affected votes count
            non_votes = [votes[j] for j, m in enumerate(models) if not getattr(m, 'is_affected', False)]
            if non_votes:
                vals_n, cnts_n = np.unique(non_votes, return_counts=True)
                sorted_cnts = np.sort(cnts_n)[::-1]
                first_non = sorted_cnts[0]
                second_non = sorted_cnts[1] if len(sorted_cnts) > 1 else 0
            else:
                first_non = second_non = 0
            total += 1
            # category 1: robust success under attack
            if first_non > second_non + attacker_group_count:
                robust += 1
                continue
            # determine final vote
            vals, counts = np.unique(votes, return_counts=True)
            tied = vals[counts == counts.max()]
            if tied.size == 1:
                final = tied[0]
            else:
                attacker_votes = [votes[j] for j, m in enumerate(models) if getattr(m, 'is_affected', False)]
                if any(v in tied for v in attacker_votes):
                    pass  # attack influence already considered
                final = random.choice(tied.tolist())
            # category 2/3 based on final vs true label
            if final == yb[i].item():
                succ_if_noattack += 1
            else:
                fail += 1
    return robust/total, succ_if_noattack/total, fail/total

# Run evaluations
res1 = ensemble_eval(selected_models, loader1)
res2 = ensemble_eval(selected_models, loader2)
res_avg = ensemble_eval(selected_models, loader_avg)
print(f"Dist1 -> Robust={res1[0]:.3f}, Succ={res1[1]:.3f}, Fail={res1[2]:.3f}")
print(f"Dist2 -> Robust={res2[0]:.3f}, Succ={res2[1]:.3f}, Fail={res2[2]:.3f}")
print(f"Avg   -> Robust={res_avg[0]:.3f}, Succ={res_avg[1]:.3f}, Fail={res_avg[2]:.3f}")

# Plot stacked area of three categories
labels = ['Dist1', 'Dist2', 'Avg']
robusts = np.array([res1[0], res2[0], res_avg[0]])
succs = np.array([res1[1], res2[1], res_avg[1]])
fails = np.array([res1[2], res2[2], res_avg[2]])
x = np.arange(len(labels))

fig, ax = plt.subplots()
ax.stackplot(x, robusts, succs, fails, labels=['Robust Success','Success','Failure'], alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Rate')
ax.legend()
plt.title('untarget backdoor attack')
plt.tight_layout()
plt.savefig('untarget_backdoor_attack.png')
plt.show()