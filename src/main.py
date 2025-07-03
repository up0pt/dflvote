import argparse
import ast
import json
import logging
import os
import random
import pathlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms  # type: ignore[import]

from client import Client
from group import Group
from utils import split_indices, filter_test, ensemble_eval
from generate_dists import generate_dirichlet_dist

# Command-line parsing
parser = argparse.ArgumentParser()
parser.add_argument('--num_clients', type=int, default=8)
parser.add_argument('--num_attackers_limit', type=int, default=4)
parser.add_argument('--groups', type=str)
parser.add_argument('--dists_groups', type=str)
parser.add_argument('--is_targeted', type=bool, default=False)
parser.add_argument('--dataset', choices=["MNIST", "CIFAR10"], default="MNIST")
parser.add_argument('--epoch', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=10)
parser.add_argument('--num_train_data', type=int, default=6000)
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
        # 前処理
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # MNIST 全訓練データ取得
        full = datasets.MNIST('.', train=True, download=True, transform=tf)

        # 9:1 に分割するインデックスだけ作成
        train_size = int(len(full) * 0.9)   # 54000
        valid_size = len(full) - train_size  # 6000
        train_idx, valid_idx = random_split(list(range(len(full))), [train_size, valid_size])

        # 元データをコピーせず、同じ transform を使い回す
        train_ds = datasets.MNIST('.', train=True, download=False, transform=tf)
        valid_ds = datasets.MNIST('.', train=True, download=False, transform=tf)

        # data / targets をインデックスでフィルタリングして上書き
        train_ds.data   = full.data[train_idx]
        train_ds.targets= full.targets[train_idx]

        valid_ds.data   = full.data[valid_idx]
        valid_ds.targets= full.targets[valid_idx]

        # test はそのまま
        test_ds = datasets.MNIST('.', train=False, download=True, transform=tf)

    case "CIFAR10":
        # 前処理
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # CIFAR10 全訓練データ取得
        full = datasets.CIFAR10('.', train=True, download=True, transform=tf)

        # 9:1 に分割するインデックスだけ作成
        train_size = int(len(full) * 0.9)   # 45000
        valid_size = len(full) - train_size  # 5000
        train_idx, valid_idx = random_split(list(range(len(full))), [train_size, valid_size])

        # 元データをコピーせず、同じ transform を使い回す
        train_ds = datasets.CIFAR10('.', train=True, download=False, transform=tf)
        valid_ds = datasets.CIFAR10('.', train=True, download=False, transform=tf)

        # data / targets をインデックスでフィルタリングして上書き
        train_ds.data   = full.data[train_idx]
        train_ds.targets= np.array(full.targets)[train_idx]

        valid_ds.data   = full.data[valid_idx]
        valid_ds.targets= np.array(full.targets)[valid_idx]

        # test はそのまま
        test_ds = datasets.CIFAR10('.', train=False, download=True, transform=tf)
    case other:
        raise NotImplementedError('Not implemented dataset')

# Setup groups
default_groups = [list(range(0, args.num_clients//2)), list(range(args.num_clients//2, args.num_clients))]
if args.groups:
    try: GROUP_IDS = ast.literal_eval(args.groups)
    except: 
        print("Invalid group format. Using default groups.")
        GROUP_IDS = default_groups
else:
    GROUP_IDS = default_groups

args.num_attackers_limit = min(args.num_attackers_limit, len(GROUP_IDS)-1)

# Dirichlet splits
# TODO: vary alpha between groups
alpha = [args.alpha]*10
# Setup distribution
dist_groups = [list(range(0, args.num_clients//2)), list(range(args.num_clients//2, args.num_clients))]
if args.dists_groups:
    try: 
        dist_groups = ast.literal_eval(args.dists_groups)
        distribution = generate_dirichlet_dist(args.num_clients, dist_groups, alpha)

    except: 
        print("Invalid dists format. Using default dists.")
        dist_groups = [list(range(0, args.num_clients//2)), list(range(args.num_clients//2, args.num_clients))]
        distribution = generate_dirichlet_dist(args.num_clients, dist_groups, alpha)
else:
    distribution = generate_dirichlet_dist(args.num_clients, dist_groups, alpha)

# mkdir for data store
exp_name = f"{args.dataset}_cl{args.num_clients}_attclimit{args.num_attackers_limit}_datasize{args.num_train_data}_distg{dist_groups[0]}_groups{GROUP_IDS[0]}_ep{args.epoch}_alpha{args.alpha}"
now = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = Path("experiments") / f"{exp_name}_{now}"
exp_dir.mkdir(parents=True, exist_ok=True)
models_dir = exp_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
print(f"Experiment directory created: {exp_dir}")

# Initialize clients
clients = {}

# クライアント生成ループ
for i in range(0, args.num_clients):
    pi = distribution[i]  # 属するグループに対応した分布
    inds_train = split_indices(train_ds, pi, args.num_train_data)
    inds_valid = split_indices(valid_ds, pi, args.num_train_data)
    train_loader = DataLoader(Subset(train_ds, inds_train), batch_size=128, shuffle=True)
    valid_loader = DataLoader(Subset(valid_ds, inds_valid), batch_size=128, shuffle=True)
    clients[i] = Client(i, train_loader, valid_loader, dataset_name=args.dataset)


# クライアントのグループ化
groups = {}
for gid, gids in enumerate(GROUP_IDS):
    grp_clients = [clients[i] for i in gids]
    grp = Group(gid, grp_clients, exp_dir)
    for client in grp_clients:
        client.set_group_id(gid)
    groups[gid] = grp

# Train and assign
for gid, grp in groups.items():
    print(f"Training Group {gid}")
    grp.train(args.epoch, args.lr)

# Select and evaluate
test_loaders = [filter_test(test_ds, dist) for dist in distribution]
# test_loader_avg = filter_test(test_ds, np.ones(10)/10)

nested_results_dict = defaultdict(dict)
for target_id, target_client in clients.items():
    # 1) 全グループからランダムに１クライアントずつ抽出
    ensemble_clients: list[Client] = []
    for _, group in groups.items():
        ensemble_clients.append(group.select_model(target_id))
    # 2) 対象クライアントのテストローダーを使う
    test_loader = test_loaders[target_id]

    # 3) 評価
    results_various_attackers = ensemble_eval(ensemble_clients, clients, args.num_clients, test_loader, args.num_attackers_limit)
    # results: {n_attacker: (rob, suc, tie, fail)}
    for att_num, result in results_various_attackers.items():
        rob, suc, tie, fail = result
        if att_num not in nested_results_dict:
            nested_results_dict[att_num] = {
                "rob": {},
                "suc": {},
                "tie": {},
                "fail": {}
            }
        nested_results_dict[att_num]["rob"][target_id] = rob  # rob[target_id]
        nested_results_dict[att_num]["suc"][target_id] = suc  # suc[]
        nested_results_dict[att_num]["tie"][target_id] = tie  # tie[]
        nested_results_dict[att_num]["fail"][target_id] = fail  # fail[]


labels = [f'Dist {i}' for i in range(1, args.num_clients+1)]

for att_num, listed_results in nested_results_dict.items():
    # Plot
    x = range(args.num_clients)
    fig, ax = plt.subplots()
    rob = [listed_results["rob"][i] for i in x]
    suc = [listed_results["suc"][i] for i in x]
    tie = [listed_results["tie"][i] for i in x]
    fai = [listed_results["fail"][i] for i in x]
    ax.bar(x, rob, label='Robust', alpha=0.6)
    ax.bar(x, suc, bottom=rob, label='Success wo Attack', alpha=0.6)
    ax.bar(x, tie, bottom=np.array(rob) + np.array(suc), label='Tie wo Attack', alpha=0.6)
    ax.bar(x, fai, bottom=np.array(rob) + np.array(suc) + np.array(tie), label='Fail wo Attack', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # rotate labels 90° so they’re vertical
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    ax.grid(True)
    ax.legend()
    plt.title(f'backdoor attack: attackers {att_num}, alpha {args.alpha}, train data size {args.num_train_data}\nGROUP {GROUP_IDS}')
    plt.tight_layout()
    if args.is_targeted == False:
        plt.savefig(exp_dir / f'target_backdoor_attack_{att_num}.png')
        plt.show()

# Plot Dirichlet distributions for each client as a heatmap

dist_matrix = np.stack(distribution)  # shape: (num_clients, num_classes)
plt.figure(figsize=(10, 6))
sns.heatmap(
    dist_matrix,
    annot=True,
    cmap="Blues",  # 1色の濃淡
    cbar=True,
    xticklabels=[f'Class {i}' for i in range(10)],
    yticklabels=[f'Client {i}' for i in range(args.num_clients)],
    linewidths=0.5,
    linecolor='gray',
    fmt=".2f"
)
plt.xlabel('Class')
plt.ylabel('Client')
plt.title('Dirichlet Distribution per Client (Heatmap)')
plt.tight_layout()
plt.savefig(exp_dir / 'dirichlet_distributions_heatmap.png')
plt.show()


# Set up logger
log_file = exp_dir / "experiment.log"
logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger()

# Save important experiment info
info = {
    "exp_dir": str(exp_dir),
    "dataset": args.dataset,
    "num_clients": args.num_clients,
    "num_attackers_limit": args.num_attackers_limit,
    "is_targeted": args.is_targeted,
    "groups": GROUP_IDS,
    "alpha": args.alpha,
    "distribution": [dist.tolist() for dist in distribution],
    "epoch": args.epoch,
    "lr": args.lr,
    "num_train_data": args.num_train_data,
    "results": {
        att_num: {
            "rob": nested_results_dict[att_num]["rob"],
            "suc": nested_results_dict[att_num]["suc"],
            "tie": nested_results_dict[att_num]["tie"],
            "fail": nested_results_dict[att_num]["fail"]
        }
        for att_num in nested_results_dict
    }
}
info_file = exp_dir / "experiment_info.json"
with open(info_file, "w") as f:
    json.dump(info, f, indent=2)

logger.info(f"Experiment info saved to {info_file}")
