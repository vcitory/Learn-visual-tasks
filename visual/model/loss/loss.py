import torch
from torch import nn

criterion = nn.MSELoss()

def loss_with_l2(outputs, labels, model, lambda_=0.01):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.sum(torch.square(param))
    return criterion(outputs, labels) + lambda_ * l2_reg