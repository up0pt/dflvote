import torch
import torch.optim as optim
import torch.nn as nn
import random
from utils import ensemble_eval
from client import Client

class Group:
    def __init__(self, gid: int, clients: list[Client]) -> None:
        self.id: int = gid
        self.clients: list[Client] = clients
        self.has_mal: bool = False
        self.chose_vote_model: bool = False
        self.compute_has_mal()

    def compute_has_mal(self):
        self.has_mal = any(c.is_mal for c in self.clients)
        for client in self.clients:
            client.set_affected(self.has_mal)

    def train(self, epochs: int, lr: float):
        # TODO: do full mesh dfl (now star or CFL)
        for _ in range(epochs):
            # local training
            for c in self.clients:
                optimizer = optim.SGD(c.model.parameters(), lr=lr)
                c.model.train()
                for Xb, yb in c.loader:
                    optimizer.zero_grad()
                    loss = nn.CrossEntropyLoss()(c.model(Xb), yb)
                    loss.backward(); optimizer.step()
            # aggregate
            avg_dict = {
                k: torch.mean(torch.stack([c.model.state_dict()[k].float() for c in self.clients]), dim=0)
                for k in self.clients[0].model.state_dict()
            }
            for c in self.clients:
                c.model.load_state_dict(avg_dict)
        torch.save(avg_dict, f"group_{self.id}_model.pth")

    def select_model(self, voted_at_id: int):
        if self.chose_vote_model:
            return random.choice(self.clients)
        else:
            ValueError("Group has not implemented voting clients selection.")

