from game.src.services.game_objects import Game
import time
import game.src.services.GIF as GIF
import torch
from numpy import mean as _mean
import torch
import os
from RL.functions.loss_functions.DQN import no_replay_loss
from tqdm import tqdm
from RL.functions.utils.log import _log



def _train_dqn_no_replay(
                    model,
                    epochs:int,
                    optimizer,
                    batch_size:int,
                    lr:float,
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

    log_file = os.path.join(model_path, "console.txt")

    game = Game(
                    batch_size=batch_size,
                    difficulty=difficulty,
                    height=height,
                    width=width,
                    VIEW_WIDTH=VIEW_WIDTH
                )

    model.train()

    _log("\n\n", log_file, verbose)
    _log(f'''initializing training with parameters :
            model : {model},
            epochs : {epochs},
            criterion : loss_no_replay,
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
            ''', log_file, verbose)
    _log("\n\n", log_file, verbose)

    LOSSES = []
    best_loss = float('inf')

    best_model = model.state_dict()

    temps = []

    for e in range(epochs):
        _log(f"\nEPOCH {e+1}/{epochs} ///", log_file, verbose)
        s = time.time()

        game.reset_world()

        n_frames = (game.world_width - game.VIEW_WIDTH) // game.tick
        w = torch.zeros(game.batch_size, n_frames, game.world_height, game.VIEW_WIDTH)

        for t in tqdm(range(n_frames)):
            optimizer.zero_grad()

            w[:, t] = game.step()
            game.t += 1

            Q_actions = model(w[:, t])

            game.flappy.step(Q_actions.argmax(dim=1)==1)

            bird_mask, done_mask = game.flappy.update_collisions()

            reward = game.flappy.reward()

            loss = no_replay_loss(
                gamma,
                model,
                w[:,t],
                game.step(game.t-1),
                Q_actions.argmax(dim=1),
                reward
            )

            LOSSES.append(loss.item())

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

        temps.append(abs(s-time.time()))
        _log(f"complétée en {int(temps[e]/60)}min{round(60*(temps[e]/60 - int(temps[e]/60)),0)}sec. Estimation de temps restant : {round((_mean(temps)*(epochs-e-1))/60, 0)} minutes.\n", log_file, verbose)

    torch.save(best_model, os.path.join(model_path, "best.pth"))
    _log("\n\nentraînement complet\n\n", log_file, verbose)

    return temps, LOSSES, best_loss.item()