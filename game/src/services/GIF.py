import os
import torch
from PIL import Image
import numpy as np

idx_gifs = 0

def gif(world, folder=os.path.join("game", "plots", "gifs"), name="game", fps=12, e=None, use_flappy_png=True):
    global idx_gifs

    os.makedirs(folder, exist_ok=True)

    if e is None:
        save_gif(color(world, use_flappy_png=use_flappy_png), folder, filename=name + f"_{idx_gifs}", fps=fps)
        idx_gifs+=1
    else:
        save_gif(color(world, use_flappy_png=use_flappy_png), folder, filename=name + f"_{e}", fps=fps)


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


import os
import torch
from PIL import Image
import torchvision.transforms as T


def color(
    world: torch.Tensor,
    use_flappy_png:bool,
    flappy_png_path: str = os.path.join("game", "static", "flappy.png")
):
    # world.shape = (duration, height, width)

    if not(use_flappy_png):
        tunnels = torch.tensor([0. / 255., 153. / 255., 0. / 255.]).view(3, 1, 1) 
        sky = torch.tensor([102. / 255., 178. / 255., 255. / 255.]).view(3, 1, 1) 
        bird = torch.tensor([173. / 255., 0. / 255., 0. / 255.]).view(3, 1, 1) 
        w = world.unsqueeze(1) # (duration, 1, H, W) 
        tunnel_mask = (w == 1).float() 
        bird_mask = (w == 0.5).float() 
        sky_mask = (1 - tunnel_mask - bird_mask) 
        return tunnel_mask * tunnels + bird_mask * bird + sky_mask * sky


    duration, H, W = world.shape

    tunnels = torch.tensor([0. / 255., 153. / 255., 0. / 255.]).view(3, 1, 1)
    sky     = torch.tensor([102. / 255., 178. / 255., 255. / 255.]).view(3, 1, 1)

    w = world.unsqueeze(1)

    tunnel_mask = (w == 1).float()
    sky_mask    = 1 - tunnel_mask

    rgb = tunnel_mask * tunnels + sky_mask * sky

    img = Image.open(flappy_png_path).convert("RGBA")

    bird_sprite = T.ToTensor()(img)

    bird_rgb   = bird_sprite[:3]
    bird_alpha = bird_sprite[3:4]

    sprite_h, sprite_w = bird_rgb.shape[1:]



    for t in range(duration):

        positions = (world[t] == 0.5).nonzero(as_tuple=False)

        if len(positions) == 0:
            continue

        y, x = positions[0]

        y = int(y)
        x = int(x)

        top  = y - sprite_h // 2
        left = x - sprite_w // 2

        top_clip = max(top, 0)
        left_clip = max(left, 0)

        bottom_clip = min(top + sprite_h, H)
        right_clip = min(left + sprite_w, W)

        sprite_top = top_clip - top
        sprite_left = left_clip - left

        sprite_bottom = sprite_top + (bottom_clip - top_clip)
        sprite_right = sprite_left + (right_clip - left_clip)

        alpha = bird_alpha[
            :,
            sprite_top:sprite_bottom,
            sprite_left:sprite_right
        ]

        bird_part = bird_rgb[
            :,
            sprite_top:sprite_bottom,
            sprite_left:sprite_right
        ]

        background = rgb[
            t,
            :,
            top_clip:bottom_clip,
            left_clip:right_clip
        ]

        rgb[
            t,
            :,
            top_clip:bottom_clip,
            left_clip:right_clip
        ] = bird_part * alpha + background * (1 - alpha)

    return rgb


def save_png(path: str, world: torch.Tensor, use_flappy_png):
    colored = color(world[0].unsqueeze(0), use_flappy_png=use_flappy_png)
    frame_np = (colored[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(frame_np, mode="RGB").save(path)
    print(f"PNG sauvegardé : {path}")