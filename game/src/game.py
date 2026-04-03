from game_controller import Game
import time
from torchvision.utils import save_image
import os

s = time.time()
batch_size=100
game = Game(batch_size=batch_size)
print(f"\nSTAT PERF : monde de {batch_size} batchs généré en {abs(s-time.time())} sec.\n")

i=1
for w in game.world:
    if i>5:
        break
    save_image(w.float(), os.path.join("game", "plots", f"test_world_{i}.png"))
    i+=1







