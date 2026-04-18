import torch

class Naive:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def action(self, w, tr = 0.3):
        return torch.rand(self.batch_size) < tr