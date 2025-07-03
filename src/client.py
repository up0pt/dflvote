import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

class Client:
    def __init__(self, cid: int, train_loader: DataLoader, valid_loader: DataLoader, dataset_name: str = "MNIST"):
        self.id = cid
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = create_model(dataset_name)
        self.is_mal: bool = False
        self.is_affected: bool = False
        self.group_id: int | None = None 

    def set_group_id(self, group_id: int) -> None:
        self.group_id = group_id

    def get_group_id(self) -> int:
        return self.group_id

    def set_affected(self, is_affected: bool) -> None:
        self.is_affected = is_affected
        
def create_model(dataset_name: str):
    match dataset_name:
        case "MNIST":
            return nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(64 * 14 * 14, 128), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(128, 10)
            )
        case "CIFAR10":
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),  # (B, 3, 32, 32) → (B, 32, 32, 32)
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), # → (B, 64, 32, 32)
                nn.MaxPool2d(2),                                                  # → (B, 64, 16, 16)
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 256), nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 10)
            )
        case other:
            raise NotImplementedError("No model is implemented for this dataset")
