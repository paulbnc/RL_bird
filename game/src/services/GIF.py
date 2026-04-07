import os
import torch
from PIL import Image
import numpy as np


def gif(world, idx: list, folder="gifs", name="game", fps=12):
    path = os.path.join("game", "plots", folder)
    os.makedirs(path, exist_ok=True)

    cpt = 0
    for i in idx:
        save_gif(color(world[i]), path, filename=name + f"_{i}", fps=fps)
        cpt += 1
        if cpt > 100:
            raise ValueError("nombre de gifs maximum atteint")


def save_gif(colored_world: torch.Tensor,
             path,
             filename: str = "game",
             fps: int = 12):
    frames = []
    duration_ms = int(1000 / fps)

    for step in colored_world:
        frame_np = (step.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame_np, mode="RGB"))

    gif_path = os.path.join(path, filename + ".gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms
    )
    print(f"GIF sauvegardé : {gif_path}")


def color(world: torch.Tensor):
    # size(world) = (duration, height, view_width)
    tunnels = torch.tensor([0. / 255., 153. / 255., 0. / 255.]).view(3, 1, 1)
    sky     = torch.tensor([102. / 255., 178. / 255., 255. / 255.]).view(3, 1, 1)

    w = world.unsqueeze(1)  # (duration, 1, H, W)
    return w * tunnels + (1 - w) * sky  # (duration, 3, H, W)