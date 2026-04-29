from game.src.services.game_objects import Game
import time
import game.src.services.GIF as GIF
import torch
from numpy import mean as _mean
import torch
import os
from loss_functions.DQN import no_replay_loss

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





def _train_dqn_no_replay(
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
                    gamma:float,
                    model_path,
                    plots_path=os.path.join("game","plots", "gifs_last"),
                    verbose:bool=True
                ):
    

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)


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
                model_path : {model_path},
                plots_path : {plots_path},
                verbose : {verbose},
                gamma : {gamma}
                ''')
        print("\n\n")


    LOSSES = []
    best_loss = float('inf')

    best_model = model.state_dict()

    temps = []

    for e in range(epochs):
        if verbose:
            print(f"\nEPOCH {e+1}/{epochs} ///")
            s = time.time()

        
        game.reset_world()


        n_frames = (game.world_width - game.VIEW_WIDTH) // game.tick
        w = torch.zeros(game.batch_size, n_frames, game.world_height, game.VIEW_WIDTH)


        for t in range(n_frames):
            optimizer.zero_grad()


            w[:, t] = game.step()
            game.t += 1

            Q_actions = model(w[:, t]) #(batch, 2) => [:,0] le Q du saut, [:,1] le Q du non-saut

            game.flappy.step(Q_actions.)
            bird_mask, done_mask = game.flappy.update_collisions()


            reward = game.flappy.reward()

            loss = no_replay_loss(
                gamma,
                model,
                w[:,t],
                game.step(t),
                Q_actions,
                reward
            )
            
            LOSSES.append(loss)

            loss.backward()
            optimizer.step()

            w[:, t][bird_mask] = 0.5
            if done_mask.any():
                game.reset_dead(done_mask)
            
        if e%freq==0:
            torch.save(model.state_dict(), os.path.join(model_path, f"epoch_{e}.pth"))
            GIF.gif(w[0], folder=plots_path, name=f"_epoch_{e}", e=e)


        if loss<best_loss:
            best_loss=loss
            best_model = model.state_dict()


        if verbose:
            temps.append(abs(s-time.time()))
            print(f"complétée en {temps[e]:.4f} sec. Estimation de temps restant : {(_mean(temps)*(epochs-e-1))/60} minutes.\n")


    torch.save(best_model, os.path.join(model_path, "best.pth"))
    print("\n\nentraînement complet\n\n")

    return temps, LOSSES, best_loss