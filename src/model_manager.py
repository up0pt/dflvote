import json
from pathlib import Path
import hashlib
from datetime import datetime
from dataclasses import dataclass
from typing import Literal
import torch
import inspect

from group import EarlyStopping

@dataclass
class ModelInfo:
    model_name: str
    param_hash: str
    init_seed: int
    architecture: str # str(model)でわかる情報で一応十分？

@dataclass
class TrainInfo:
    dataset: Literal['MNIST', 'CIFAR10'] #TODO: if you add new dataset, please add a type hint.
    datasize: int
    seed: int
    model_info: ModelInfo
    num_clients: int
    groups: list[list[int]]
    group_id: int
    dists: list[list[float]]
    epoch: int
    early_stopping: EarlyStopping
    agg_method: Literal['FedAvg']

class ModelManager:
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)

    def get_exp_name(self, config: dict) -> str:
        """一意なハッシュ名 or パラメータによる実験名を生成"""
        key_fields = [
            f"cl{config['num_clients']}",
            f"attc{config['num_attack_clients']}",
            f"alpha{config['alpha']}",
        ]
        dataset = config.get("dataset", "UnknownDataset")
        exp_name = f"{dataset}/{'_'.join(key_fields)}"
        return exp_name

    def get_model_path(self, config: dict, round_num: int, create=False) -> Path:
        """対応するモデルファイルのパスを返す"""
        exp_name = self.get_exp_name(config)
        timestamp = config.get("timestamp", None)
        if timestamp is None:
            raise ValueError("configに'timestamp'が必要です（例: '20250701_1530'）")

        run_dir = self.base_dir / exp_name / f"run_{timestamp}"
        if create:
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / f"model_round_{round_num}.pth"

    def save_config(self, config: dict):
        exp_name = self.get_exp_name(config)
        timestamp = config["timestamp"]
        run_dir = self.base_dir / exp_name / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self, config_path: Path) -> dict:
        with open(config_path) as f:
            return json.load(f)

    def list_models(self, config: dict) -> list:
        exp_name = self.get_exp_name(config)
        timestamp = config["timestamp"]
        run_dir = self.base_dir / exp_name / f"run_{timestamp}"
        return sorted(run_dir.glob("model_round_*.pth"))


if __name__ == "__main__":
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(64 * 14 * 14, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 10)
    )
    print(model.__str__())

