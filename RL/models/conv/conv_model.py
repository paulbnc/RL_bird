import torch.nn as nn
from RL.functions.utils.size import conv_out_size

class ConvNN_small(nn.Module):

    def __init__(self, view_width: int, view_height: int):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten(start_dim=1)

        
        h, w = view_height, view_width

        for _ in range(3):
            h = conv_out_size(h)
            w = conv_out_size(w)

        self.flat_size = 32 * h * w

        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, w):
        w = w.unsqueeze(1)
        w = self.relu(self.conv1(w))
        w = self.relu(self.conv2(w))
        w = self.relu(self.conv3(w))
        w = self.flat(w)
        w = self.relu(self.fc1(w))
        w = self.fc2(w)
        return w