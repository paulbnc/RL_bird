from game_controller import Game
import time
from torchvision.utils import save_image
import os
import torch

s = time.time()
batch_size=10
game = Game(batch_size=batch_size, difficulty=3)
print(f"\nSTAT PERF : monde de {batch_size} batchs généré en {abs(s-time.time())} sec.\n")

i=1
for w in game.world:
    if i > 10:
        break

    # couleurs définies comme vecteurs RGB
    grass = torch.tensor([0./255., 153./255., 0./255.]).view(3, 1, 1)  # vert herbe
    sky   = torch.tensor([102./255., 178./255., 255./255.]).view(3, 1, 1)  # bleu ciel

    # interpolation
    var = w * grass + (1 - w) * sky

    save_image(var.float(), os.path.join("game", "plots", "test_world", f"test_world_{i}.png"))
    i += 1







