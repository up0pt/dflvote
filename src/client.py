import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random

from utils import create_model

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
        
