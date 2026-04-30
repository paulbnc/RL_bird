import torch
import torch.nn as nn





class LinearNN_small(nn.Module):

    def __init__(
                    self,
                    view_width:int,
                    view_height:int     
                ):
        
        super().__init__()

        self.fc1 = nn.Linear(view_height*view_width, view_height*view_width // 2)
        self.fc2 = nn.Linear(view_height*view_width // 2, view_height*view_width // 4)
        self.fc3 = nn.Linear(view_height*view_width // 4, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, w):
        w = self.flat(w)
        w = self.relu(self.fc1(w))
        w = self.relu(self.fc2(w))
        w = self.fc3(w)
        return w