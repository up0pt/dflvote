import pathlib
import torch
import torch.optim as optim
import torch.nn as nn
import random
import logging
from typing import Literal
from pathlib import Path
from torch.utils.data import DataLoader

from client import Client


class Group:
    def __init__(self, gid: int, clients: list[Client], dir_path: Path, choose_vote_model: Literal['random', 'nearest'] = 'random') -> None:
        self.id: int = gid
        self.clients: list[Client] = clients
        self.dir_path: Path = dir_path
        self.has_mal: bool = False
        self.choose_vote_model = choose_vote_model
        self.compute_has_mal()

    def compute_has_mal(self):
        self.has_mal = any(c.is_mal for c in self.clients)
        for client in self.clients:
            client.set_affected(self.has_mal)

    def train(
        self,
        epochs: int,
        lr: float
    ) -> None:
        # Setup logging
        log_path = self.dir_path / 'train.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Move all client models to device
        for c in self.clients:
            c.model.to(device)

        early_stopping = EarlyStopping()
        for epoch in range(1, epochs + 1):
            logging.info(f"Epoch {epoch}/{epochs} - Start training")
            # Local training
            for c in self.clients:
                c.model.train()
                optimizer = optim.SGD(c.model.parameters(), lr=lr)
                for Xb, yb in c.train_loader:
                    Xb = Xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(c.model(Xb), yb)
                    loss.backward()
                    optimizer.step()
                    
            # Validation
            logging.info(f"Epoch {epoch}/{epochs} - Start validation")
            self.clients[0].model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for Xv, yv in c.valid_loader:
                    Xv = Xv.to(device, non_blocking=True)
                    yv = yv.to(device, non_blocking=True)
                    outputs = self.clients[0].model(Xv)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == yv).sum().item()
                    total += yv.size(0)
            val_acc = correct / total if total > 0 else 0.0
            logging.info(f"Epoch {epoch}/{epochs} - Validation Accuracy: {val_acc:.4f}")

            early_stopping(val_acc)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
            # Aggregate: average states
            state_dicts = [c.model.state_dict() for c in self.clients]
            avg_dict = {}
            for key in state_dicts[0].keys():
                stacked = torch.stack(
                    [sd[key].float().to(device) for sd in state_dicts],
                    dim=0
                )
                avg_dict[key] = stacked.mean(dim=0)

            # Update client models with averaged parameters
            for c in self.clients:
                c.model.load_state_dict(avg_dict)

            
        # Save final model to CPU
        save_models_dir = self.dir_path / 'models'
        save_models_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_models_dir / f"group_{self.id}_model.pth"
        save_dict = {k: v.cpu() for k, v in avg_dict.items()}
        torch.save(save_dict, save_path)
        logging.info(f"Saved final model at {save_path}")

    def select_model(self, voted_at_id: int):
        match self.choose_vote_model:
            case 'random':
                return random.choice(self.clients)
            case other:
                raise ValueError("Group has not implemented voting clients selection.")

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_threshold = 0.88  # この精度を超えたら early stopping の監視開始

    def __call__(self, val_acc):
        # 精度が閾値を超えていない間はカウントしない
        if val_acc < self.val_acc_threshold:
            return

        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
