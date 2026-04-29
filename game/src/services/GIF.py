import os
import torch
from PIL import Image
import numpy as np

idx_gifs = 0

def gif(world, folder=os.path.join("game", "plots", "gifs"), name="game", fps=12, e=None):
    global idx_gifs

    os.makedirs(folder, exist_ok=True)

    if e is None:
        save_gif(color(world), folder, filename=name + f"_{idx_gifs}", fps=fps)
        idx_gifs+=1
    else:
        save_gif(color(world), folder, filename=name + f"_{e}", fps=fps)


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
    bird    = torch.tensor([173. / 255., 0. / 255., 0. / 255.]).view(3, 1, 1)

    w = world.unsqueeze(1)  # (duration, 1, H, W)

    tunnel_mask = (w == 1).float()
    bird_mask   = (w == 0.5).float()
    sky_mask    = (1 - tunnel_mask - bird_mask)

    return tunnel_mask * tunnels + bird_mask * bird + sky_mask * sky