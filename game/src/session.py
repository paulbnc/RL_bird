import torch

def get_difficulty_params(difficulty):
    return {
        'speed':difficulty, #dans l'idée, 1 2 ou 3
        'tunnels':3*difficulty
    }

class Bird:
    def __init__(self,
                 height,
                 width):
        self.top_left_position = (height//2, width//2)
        self.width = 50
        self.height = 50

    def step(self, intensity_move):
        if intensity_move!=0:
            self.compute_movement(intensity_move)

    def compute_movement(self, intensity_move):
        raise NotImplementedError


class Game:
    def __init__(self, 
                 batch_size=1, 
                 duration=1,
                 difficulty=1,
                 height=500,
                 width=10000):
        self.batch_size = batch_size
        self.duration = duration
        self.world_height = height
        self.world_width = width
        self.speed = get_difficulty_params(difficulty)['speed']
        self.tunnels = get_difficulty_params(difficulty)['tunnels']
        self.bird = Bird(height, width)
        self.world = self.generate_world()

    def generate_world(self):
        #idée : générer aléatoirement tout le terrain (sans flappy) avec le nombre de self.tunnels définis
        #si possible : batcher la générations de plusieurs worlds selon self.batch_size
        #raise NotImplementedError

        world = torch.zeros(size=(self.batch_size, self.duration, self.world_height, self.world_width))



    def step(self, actions):
        #idée : faire défiler l'écran selon le temps qui avance (nombre de ticks à définir selon self.speed)
        #encore une fois gérer le traitement par batchs
        #dans actions, bien définir si flappy saute et à quelle intensité (voir class Bird)
        raise NotImplementedError