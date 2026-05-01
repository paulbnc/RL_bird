from game.src.services.game_objects import Game
from game.src.services.GIF import save_png
import os

def generate_world(difficulty, height, width):

    game = Game(
        batch_size=1,
        difficulty=difficulty,
        height=height,
        width=width
    )

    game.reset_world()
    path = os.path.join("game", "plots", "world_test_generation_png", f"dif{difficulty}.png")
    save_png(path, game.world)
