from game.src.services.game import Game
import time
import game.src.services.GIF as GIF
import random
import torch


epochs = 10
batch_size = 10

VIEW_WIDTH = 200

game = Game(
                batch_size=batch_size, 
                difficulty=3,
                height=100,
                width=1000
            )

for e in range(epochs):

    s = time.time()
    game.reset_world()
    print(f"\nSTAT PERF epoch {e} : monde de {batch_size} batchs généré en {abs(s-time.time())} sec.\n")

    i = random.randint(0, game.batch_size - 1)

    # Nombre de frames : on déplace la fenêtre de `tick` px à chaque frame
    # jusqu'à ce qu'elle atteigne la fin du monde
    n_frames = (game.world_width - VIEW_WIDTH) // game.tick

    w = torch.zeros(1, n_frames, game.world_height, VIEW_WIDTH)

    for t in range(n_frames):
        x = t * game.tick  # position gauche de la fenêtre
        w[0, t] = game.world[i, :, x : x + VIEW_WIDTH]

    GIF.gif(w, idx=[0], name=f"game_sample_{e}")