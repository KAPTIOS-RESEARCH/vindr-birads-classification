import torch
from torch import nn
from tqdm import tqdm
from src.core.trainer import BaseTrainer
from sklearn.metrics import roc_auc_score


class BinaryTrainer(BaseTrainer):

    def __init__(self, model: nn.Module, parameters: dict, device: str):
        super(BinaryTrainer, self).__init__(model, parameters, device)
        if not self.criterion:
            self.criterion = nn.BCEWithLogitsLoss()

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0

        all_preds = []
        all_targets = []
        all_probs = []

        with tqdm(train_loader, desc="Running training phase") as pbar:
            for data, targets in train_loader:
                data, targets = data.to(
                    self.device), targets.unsqueeze(1).to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                pbar.update(1)

        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)

        train_metric = roc_auc_score(
            all_targets.detach().numpy(), all_probs.detach().numpy())
        train_loss /= len(train_loader)
        return train_loss, train_metric

    def test(self, val_loader):
        self.model.eval()

        test_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            with tqdm(val_loader, leave=False, desc="Running testing phase") as pbar:
                for data, targets in val_loader:
                    data, targets = data.to(
                        self.device), targets.unsqueeze(1).to(self.device)
                    logits = self.model(data)
                    loss = self.criterion(logits, targets)
                    test_loss += loss.item()
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).int()
                    all_targets.append(targets.cpu())
                    all_probs.append(probs.cpu())
                    all_preds.append(preds.cpu())
                    pbar.update(1)
        all_targets = torch.cat(all_targets)
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)

        test_metric = roc_auc_score(
            all_targets.detach().numpy(), all_probs.detach().numpy())

        return test_loss, test_metric
