from torch import nn
import torch
import torchmetrics

class Accuracy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # argmax and compare
        y = y.long()
        predicted = torch.argmax(y_hat, dim=1)
        correct = (predicted == y).sum().item()
        total = y.size(0)
        accuracy = correct / total
        return accuracy

import torch

def precision(outputs, labels, average='macro'):
    # get the index of the max log-probability
    predicted = torch.argmax(outputs, dim=1)
    # calculate precision
    p = torchmetrics.functional.precision(predicted, labels, num_classes=outputs.shape[1], average=average, task="multiclass")
    return p.item()

def recall(outputs, labels, average='macro'):
    # get the index of the max log-probability
    predicted = torch.argmax(outputs, dim=1)
    # calculate recall
    r = torchmetrics.functional.recall(predicted, labels, num_classes=outputs.shape[1], average=average, task="multiclass")
    return r.item()

def f1_score(outputs, labels, average='macro'):
    # get the index of the max log-probability
    predicted = torch.argmax(outputs, dim=1)
    # calculate f1-score
    f1 = torchmetrics.functional.f1_score(predicted, labels, num_classes=outputs.shape[1], average=average,  task="multiclass")
    return f1.item()
