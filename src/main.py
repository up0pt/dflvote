import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms # type: ignore[import]
import matplotlib.pyplot as plt
import random
import ast

from client import Client
from group import Group
from utils import split_indices, filter_test, ensemble_eval

# Command-line parsing
parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=8)
parser.add_argument('--num_attackers', type=int, default=2)
parser.add_argument('--attackers', type=int, nargs='*')
parser.add_argument('--groups', type=str)
parser.add_argument('--dists', type=str)
parser.add_argument('--is_targeted', type=bool, default=False)
parser.add_argument('--dataset', choices=["MNIST", 'CIFAR10'], default="MNIST")
args = parser.parse_args()

# Reproducibility
np.random.seed(0); 
torch.manual_seed(0); 
random.seed(0)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data preparation
match args.dataset:
    case "MNIST":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_ds = datasets.MNIST('.', train=True, download=True, transform=tf)
        test_ds = datasets.MNIST('.', train=False, download=True, transform=tf)
    case "CIFAR10":
        raise NotImplementedError('CIFAR10 is not implemented')

# Setup groups
default_groups = [list(range(0, args.num_clients//2)), list(range(args.num_clients//2, args.num_clients))]
if args.groups:
    try: GROUP_IDS = ast.literal_eval(args.groups)
    except: 
        print("Invalid group format. Using default groups.")
        GROUP_IDS = default_groups
else:
    GROUP_IDS = default_groups

# Dirichlet splits
# TODO: vary alpha between groups
alpha = [5]*10
# Setup distribution
default_distribution = [np.random.dirichlet(alpha)]* (args.num_clients//2) + [np.random.dirichlet(alpha)] * (args.num_clients - args.num_clients//2)
if args.dists:
    raise NotImplementedError('')
else:
    distribution = default_distribution

# Initialize clients
clients = {}

# クライアント生成ループ
for i in range(0, args.num_clients):
    pi = distribution[i]  # 属するグループに対応した分布
    inds = split_indices(train_ds, pi)
    loader = DataLoader(Subset(train_ds, inds), batch_size=128, shuffle=True)
    clients[i] = Client(i, loader)

# Designate attackers
client_ids = list(clients.keys())
if args.attackers:
    attackers = args.attackers[:args.num_attackers]
else:
    attackers = random.sample(client_ids, args.num_attackers)
for aid in attackers:
    clients[aid].is_mal = True
print(f"Attackers: {attackers}")

# クライアントのグループ化
groups = {}
for gid, gids in enumerate(GROUP_IDS):
    grp_clients = [clients[i] for i in gids]
    grp = Group(gid, grp_clients)
    for client in grp_clients:
        client.set_group_id(gid)
    groups[gid] = grp

# Train and assign
for gid, grp in groups.items():
    print(f"Training Group {gid}")
    grp.train(5, 0.001)

# Select and evaluate
test_loaders = [filter_test(test_ds, dist) for dist in distribution]
test_loader_avg = filter_test(test_ds, np.ones(10)/10)

results = {}

for target_id, target_client in clients.items():
    # 1) 全グループからランダムに１クライアントずつ抽出
    ensemble_clients: list[Client] = []
    for _, group in groups.items():
        ensemble_clients.append(group.select_model(target_id))
    # 2) 対象クライアントのテストローダーを使う
    test_loader = test_loaders[target_id]

    # 3) 評価
    robust, success, failure = ensemble_eval(ensemble_clients, test_loader)
    results[target_id] = (robust, success, failure)


labels = [f'Dist {i}' for i in range(1, args.num_clients+1)]
ordered = [results[k] for k in sorted(results.keys())]

# 2) 各リストに分解
rob, suc, fai = zip(*ordered)

# Plot
x = range(args.num_clients)
fig, ax = plt.subplots()
ax.stackplot(x, rob, suc, fai, labels=['Robust','Success','Fail'], alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.title('target backdoor attack')
plt.tight_layout()
if args.is_targeted == False:
    plt.savefig('target_backdoor_attack.png')
    plt.show()
