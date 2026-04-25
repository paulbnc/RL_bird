import torch
import torch.nn as nn

class Naive(nn.Module):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, w, tr = 0.3):
        return torch.rand(self.batch_size) < tr