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


        start = self.world_width//10 #pour ne générer aucun tuyau/tunnel avant un dixième de la map

        #maintenant, on veut placer, à partir de start, self.tunnels tuyaux : pour cela on va
        #les placer (pour commencer, on verra plus tard pour complexifier) uniformément le long de [start:width]
        
        tunnel_width = 100 #arbitraire

        #le schéma est le suivant ; l'écart entre les tuyaux, 
        #en n'ayant pas d'écart au début et un écart à la fin, est ~(self.width-start-self.tunnels*tunnel_width)//self.tunnels
        tunnel_distance = (self.world_width-start-self.tunnels*tunnel_width)//self.tunnels

        #le pattern des tuyaux est le même toujours, sauf si la difficulté change (+ de tuyaux dans ce cas).
        #ce qui change entre les parties, c'est la zone de "vide", i.e. la zone du tuyau que l'on peut traverser.
        #on va donc placer cette zone aléatoirement mais jamais trop loin de la précédente, pour
        # laisser la partie "jouable" (processus de génération selon une chaîne de markov) ; disons ne pas éloigner 
        #la zone suivante + que d'un tier de la hauteur du jeu self.world_height.
        

        hole_size = self.world_height // 4 #taille du trou
        max_shift = self.world_height // 3 #le 1/3 dont on parlait pour la chaîne de Markov

        world = torch.zeros(size=(self.batch_size, self.world_height, self.world_width))
        
        #ci-dessous, position du premier trou (batchée)
        hole_center = torch.randint(
            low=hole_size // 2,
            high=self.world_height - hole_size // 2,
            size=(self.batch_size,)
        )

        for t in range(self.tunnels):

            x_start = start + t * (tunnel_width + tunnel_distance)
            x_end = x_start + tunnel_width
            
            shift = torch.randint(-max_shift, max_shift + 1, (self.batch_size,))
            hole_center = torch.clamp(
                hole_center + shift,
                hole_size // 2,
                self.world_height - hole_size // 2
            )

            hole_min = hole_center - hole_size // 2
            hole_max = hole_center + hole_size // 2

            for b in range(self.batch_size):
                world[b, :, x_start:x_end] = 1 #on place le tuyau
                world[b, hole_min[b]:hole_max[b], x_start:x_end] = 0 #on place le trou / l'ouverture

        return world



    def step(self, actions):
        #idée : faire défiler l'écran selon le temps qui avance (nombre de ticks à définir selon self.speed)
        #encore une fois gérer le traitement par batchs
        #dans actions, bien définir si flappy saute et à quelle intensité (voir class Bird)
        raise NotImplementedError