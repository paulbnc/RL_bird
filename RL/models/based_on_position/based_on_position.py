import torch.nn as nn
import torch

class model_position(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f1 = nn.Linear(4, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Linear(64, 2)
        self.act = nn.ReLU()
        
    
    def forward(self, obs):
        state = obs["screen"]
        vy  = obs["vy"]
        features = get_distance(state, vy)   # (Batch, 4)
        x = self.f3(self.act(self.f2(self.act(self.f1(features)))))
        return x

def get_distance(state, vy):
    # Pour chaque oiseau du batch, on calcule 3 features de distance.
    # 1 -> bord droit oiseau - bord gauche tuyau
    # 2 -> centre oiseau - haut du trou
    # 3 -> centre oiseau - bas du trou

    B, _, H, W = state.shape
    features = torch.zeros(B, 4, device=state.device)
 
    for i in range(B):
        world = state[i, 0]    # (H, W)  1=tuyau
        bird  = state[i, 1]    # (H, W)  1=oiseau
 
        
        bird_cols = bird.any(dim=0).nonzero(as_tuple=True)[0]  # colonnes occupées
        bird_rows = bird.any(dim=1).nonzero(as_tuple=True)[0]  # lignes  occupées
 
        if bird_cols.numel() == 0 or bird_rows.numel() == 0:
            # Oiseau hors-cadre
            continue
 
        bird_x_right  = int(bird_cols.max().item())            # bord droit
        bird_center_y = float(bird_rows.float().mean().item()) # centre Y
 
        
        pipe_col = None
        for col in range(bird_x_right + 1, W):
            if world[:, col].any():
                pipe_col = col
                break

        dist_h = pipe_col - bird_x_right

        free_rows = (world[:, pipe_col] != 1.0).nonzero(as_tuple=True)[0]
 
    
        gap_top    = float(free_rows.min().item())
        gap_bottom = float(free_rows.max().item())
 
        features[i, 0] = dist_h                   
        features[i, 1] = (bird_center_y - gap_top)    
        features[i, 2] = (bird_center_y - gap_bottom)
        features[i, 3] = vy[i]
 
    return features




