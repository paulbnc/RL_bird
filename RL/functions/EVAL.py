from game.src.services.game_objects import Game
import time
import game.src.services.GIF as GIF
import torch


def _eval(
        model,
        batch_size:int,
        difficulty:int,
        height:int,
        width:int,
        VIEW_WIDTH:int,
        save:int,
        idx_save:int,
        threshold:float=0.5):
    
    game = Game(
                    batch_size=batch_size,
                    difficulty=difficulty,
                    height=height,
                    width=width,
                    VIEW_WIDTH=VIEW_WIDTH
                )

    s = time.time()
    game.reset_world()
    print(f"\nSTAT PERF : {batch_size} mondes générés en {abs(s-time.time()):.4f} sec.\n")

    model.eval()


    n_frames = (game.world_width - game.VIEW_WIDTH) // game.tick
    w = torch.zeros(game.batch_size, n_frames, game.world_height, game.VIEW_WIDTH)


    for t in range(n_frames):

        w[:, t] = game.step()
        game.t += 1

        actions = model(w[:, t])>threshold
        game.flappy.step(actions)
        bird_mask, done_mask = game.flappy.update_collisions()

        w[:, t][bird_mask] = 0.5
        if done_mask.any():
            game.reset_dead(done_mask)

    if save>0:
        if save>batch_size:
            save=batch_size
        for i in range(save):
            GIF.gif(w[i], name=f"eval_{idx_save}_sample_{i}")

    print(f"\nSTAT PERF : {batch_size} mondes en eval complétés en {abs(s-time.time()):.4f} sec.\n")