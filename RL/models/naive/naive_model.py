import torch

class Naive:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def action(self, w):
        return torch.rand(self.batch_size) < 0.3