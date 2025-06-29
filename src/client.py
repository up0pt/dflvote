import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

class Client:
    def __init__(self, cid: int, loader: DataLoader):
        self.id = cid
        self.loader = loader
        self.model = create_model()
        self.is_mal: bool = False
        self.is_affected: bool = False
        self.group_id: int | None = None 

    def set_group_id(self, group_id: int) -> None:
        self.group_id = group_id

    def set_affected(self, is_affected: bool) -> None:
        self.is_affected = is_affected
        
def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(64 * 14 * 14, 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, 10)
    )
