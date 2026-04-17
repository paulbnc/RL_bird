from game.src.services.game_objects import Game
from RL.models.naive.naive_model import Naive
import time
import game.src.services.GIF as GIF
import random
import torch

epochs = 10
batch_size = 10

game = Game(
    batch_size=batch_size,
    difficulty=2,
    height=100,
    width=1000,
    VIEW_WIDTH=200
)

model = Naive(batch_size)

for e in range(epochs):

    s = time.time()
    game.reset_world()
    print(f"\nSTAT PERF epoch {e} : monde de {batch_size} batchs généré en {abs(s-time.time()):.4f} sec.\n")

    n_frames = (game.world_width - game.VIEW_WIDTH) // game.tick
    w = torch.zeros(game.batch_size, n_frames, game.world_height, game.VIEW_WIDTH)

    row_idx = torch.arange(game.world_height).unsqueeze(0)
    col_idx = torch.arange(game.VIEW_WIDTH).unsqueeze(0)

    for t in range(n_frames):

        w[:, t] = game.step()
        game.t += 1

        actions = model.action(w[:, t])
        game.flappy.step(actions)
        game.flappy.update_collisions()

        alive_mask = game.flappy.alive.bool()
        tops = game.flappy.y.long().clamp(0, game.world_height - game.flappy.height - 1)
        bird_rows = (row_idx >= tops.unsqueeze(1)) & (row_idx < (tops + game.flappy.height).unsqueeze(1))
        bird_cols = (col_idx >= game.flappy.x_col) & (col_idx < game.flappy.x_col + game.flappy.width)
        bird_mask = bird_rows.unsqueeze(2) & bird_cols.unsqueeze(0)
        bird_mask = bird_mask & alive_mask[:, None, None]
        w[:, t][bird_mask] = 0.5

        done_mask = ~alive_mask
        if done_mask.any():
            game.reset_dead(done_mask)

    i = random.randint(0, game.batch_size - 1)
    GIF.gif(w[i], name=f"game_sample_{e}")
    print(f"\nSTAT PERF epoch {e} : epoch complétée en {abs(s-time.time()):.4f} sec.\n")