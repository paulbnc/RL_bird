from game.src.services.game_objects import Game
import time
import game.src.services.GIF as GIF
import torch
from numpy import mean as _mean


def _train_classic(
                    model,
                    epochs:int,
                    lr:float,
                    criterion,
                    optimizer,
                    batch_size:int,
                    threshold:float,
                    difficulty:int,
                    height:int,
                    width:int,
                    VIEW_WIDTH:int,
                    freq:int,
                    path,
                    verbose:bool=True
                ):
    
    game = Game(
                    batch_size=batch_size,
                    difficulty=difficulty,
                    height=height,
                    width=width,
                    VIEW_WIDTH=VIEW_WIDTH
                )

    model.train()


    if verbose:
        print("\n\n")
        print(f'''initializing training with parameters :
                model : {model},
                epochs : {epochs},
                criterion : {criterion},
                optimizer : {optimizer},
                batch_size : {batch_size},
                learning rate : {lr},
                threshold : {threshold},
                difficulty : {difficulty},
                height : {height},
                width : {width},
                view_width : {VIEW_WIDTH},
                freq : {freq},
                path : {path},
                verbose : {verbose}
                ''')
        print("\n\n")


    LOSSES = []
    temps = []

    for e in range(epochs):
        if verbose:
            print(f"\nEPOCH {e+1}/{epochs} ///")
            s = time.time()

        
        game.reset_world()


        n_frames = (game.world_width - game.VIEW_WIDTH) // game.tick
        w = torch.zeros(game.batch_size, n_frames, game.world_height, game.VIEW_WIDTH)


        for t in range(n_frames):


            # À AJOUTER : REWARDS, LOSS, STEP, etc.


            w[:, t] = game.step()
            game.t += 1

            actions = model(w[:, t])>threshold
            game.flappy.step(actions)
            bird_mask, done_mask = game.flappy.update_collisions()

            w[:, t][bird_mask] = 0.5
            if done_mask.any():
                game.reset_dead(done_mask)

        if verbose:
            temps.append(abs(s-time.time()))
            print(f"complétée en {temps[e]:.4f} sec. Estimation de temps restant : {(_mean(temps)*(epochs-e-1))/60} minutes.\n")