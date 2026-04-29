import torch




def get_difficulty_params(difficulty):
    if not(difficulty in (1,2,3)):
         raise IndexError
    return {
        'speed':difficulty, #dans l'idée, 1 2 ou 3
        'tunnels':3*difficulty
    }






class Bird:
    def __init__(
                    self, 
                    game,
                    gravity = 4,
                    strength = -8.,
                    max_fall_speed = 10.
                ):
        
        self.game = game

        self.height = game.world_height // 10
        self.width = self.height

        self.x_col = (game.VIEW_WIDTH // 100) + 5

        self.y = torch.full((game.batch_size,), float(game.world_height // 2))
        self.vy = torch.zeros(game.batch_size) #l'axe x n'est pas représenté car fixe

        self.gravity = gravity
        self.strength = strength
        self.max_fall_speed = max_fall_speed

        self.alive = torch.ones(size=(self.game.batch_size,))



    def step(self, actions:torch.BoolTensor):
        #actions : BoolTensor (batch_size,) : 1=clic/saut, 0=rien
        self.vy[actions] = self.strength
        self.vy = torch.clamp(self.vy + self.gravity, self.strength, self.max_fall_speed)
        self.y += self.vy



    def update_collisions(self):
        H = self.game.world_height

        tops    = self.y.long()
        bottoms = (self.y + self.height).long()

        out_of_bounds = (tops < 0) | (bottoms >= H)

        x_global = (self.game.t * self.game.tick + self.x_col).clamp(0, self.game.world_width - self.width - 1)

        col_idx   = (x_global.unsqueeze(1).unsqueeze(2) + torch.arange(self.width).unsqueeze(0).unsqueeze(0)).expand(-1, H, -1)
        col_slice = self.game.world.gather(2, col_idx)                # (B, H, bird_width)

        row_idx   = torch.arange(H).unsqueeze(0)
        bird_rows = (row_idx >= tops.unsqueeze(1)) & (row_idx < bottoms.unsqueeze(1))

        hit_wall = (col_slice * bird_rows.unsqueeze(-1)).sum(dim=(1, 2)) > 0

        self.alive = (~out_of_bounds & ~hit_wall).float()

        row_idx = torch.arange(self.game.world_height).unsqueeze(0)
        col_idx = torch.arange(self.game.VIEW_WIDTH).unsqueeze(0)


        alive_mask = self.alive.bool()
        tops = self.y.long().clamp(0, self.game.world_height - self.height - 1)
        bird_rows = (row_idx >= tops.unsqueeze(1)) & (row_idx < (tops + self.height).unsqueeze(1))
        bird_cols = (col_idx >= self.x_col) & (col_idx < self.x_col + self.width)
        bird_mask = bird_rows.unsqueeze(2) & bird_cols.unsqueeze(0)
        bird_mask = bird_mask & alive_mask[:, None, None]

        done_mask = ~alive_mask

        return bird_mask, done_mask





class Game:
    def __init__(self,
                 batch_size,
                 difficulty=1,
                 height=100,
                 width=1000,
                 VIEW_WIDTH=200):
        self.batch_size = batch_size
        self.world_height = height
        self.world_width = width
        self.speed = get_difficulty_params(difficulty)['speed']
        self.tunnels = get_difficulty_params(difficulty)['tunnels']
        self.VIEW_WIDTH = VIEW_WIDTH

        #idée : faire défiler l'écran selon le temps qui avance (nombre de ticks à définir selon self.speed)
        #encore une fois gérer le traitement par batchs
        #dans actions, bien définir si flappy saute et à quelle intensité (voir class Bird)
        #partons sur un tick = 2*self.speed pixels
        self.tick = 2*self.speed
        self.duration = self.world_width//self.tick

        #le monde est un tenseur de taille (batch_size, hauteur, largeur)
        self.world = torch.Tensor(size=(self.batch_size,
                                        self.world_height, 
                                        self.world_width)) #vide avant de reset dès le début
        

        self.flappy = Bird(game=self)

        self.t = torch.zeros(self.batch_size, dtype=torch.long)
        
    def reset_world(self):
        self.world[:,] = self.generate_world() #état initial

    def generate_world(self):
        #idée : générer aléatoirement tout le terrain (sans flappy) avec le nombre de self.tunnels définis
        #si possible : batcher la générations de plusieurs worlds selon self.batch_size

        start = self.world_width//10 #pour ne générer aucun tuyau/tunnel avant un dixième de la map

        #maintenant, on veut placer, à partir de start, self.tunnels tuyaux : pour cela on va
        #les placer (pour commencer, on verra plus tard pour complexifier) uniformément le long de [start:width]
        
        tunnel_width = (self.world_width//9)//3

        #le schéma est le suivant ; l'écart entre les tuyaux, 
        #en n'ayant pas d'écart au début et un écart à la fin, est ~(self.width-start-self.tunnels*tunnel_width)//self.tunnels
        tunnel_distance = (self.world_width-start-self.tunnels*tunnel_width)//self.tunnels

        #le pattern des tuyaux est le même toujours, sauf si la difficulté change (+ de tuyaux dans ce cas).
        #ce qui change entre les parties, c'est la zone de "vide", i.e. la zone du tuyau que l'on peut traverser.
        #on va donc placer cette zone aléatoirement mais jamais trop loin de la précédente, pour
        # laisser la partie "jouable" (processus de génération selon une chaîne de markov) ; disons ne pas éloigner 
        #la zone suivante + que d'un tier de la hauteur du jeu self.world_height.

        hole_size = self.world_height // 3 #taille du trou
        max_shift = self.world_height // 3 #le 1/3 dont on parlait pour la chaîne de Markov

        world = torch.zeros(size=(self.batch_size, self.world_height, self.world_width))
        world[:, :, 0] = 1 #bordures
        world[:, 0, :] = 1 
        world[:, :, self.world_width-1] = 1 
        world[:, self.world_height-1, :] = 1 

        #ci-dessous, position du premier trou (batchée)
        hole_center = torch.randint(
            low=hole_size // 2,
            high=self.world_height - hole_size // 2,
            size=(self.batch_size,)
        )

        # indices de lignes et de colonnes pour le masquage vectorisé
        row_idx = torch.arange(self.world_height)   # (H,)

        for t in range(self.tunnels):

            x_start = start + t * (tunnel_width + tunnel_distance)
            x_end = x_start + tunnel_width
            
            shift = torch.randint(-max_shift, max_shift + 1, (self.batch_size,))
            hole_center = torch.clamp(
                hole_center + shift,
                hole_size // 2,
                self.world_height - hole_size // 2
            )

            hole_min = hole_center - hole_size // 2   # (B,)
            hole_max = hole_center + hole_size // 2   # (B,)

            # masque mur : toutes les lignes sont 1, sauf la zone du trou
            # is_wall : (B, H) — True pour les lignes qui restent mur
            is_wall = (row_idx.unsqueeze(0) < hole_min.unsqueeze(1)) | \
                    (row_idx.unsqueeze(0) >= hole_max.unsqueeze(1))

            # on place le tuyau (1) sur la tranche, puis on perce le trou (0) vectoriellement
            world[:, :, x_start:x_end] = is_wall.unsqueeze(2).float()

        return world
    

    def step(self, t=None):
        if t is None:
            x = self.t * self.tick                                    #(batch,) positions gauches
        else:
            x = t * self.tick                                    #(batch,) positions gauches

        col_idx = x.unsqueeze(1).unsqueeze(2) + \
                torch.arange(self.VIEW_WIDTH).unsqueeze(0).unsqueeze(0)  #(batch, 1, VIEW_WIDTH)
        col_idx = col_idx.expand(-1, self.world_height, -1)                
        return self.world.gather(2, col_idx)
    

    def reset_dead(self, done_mask):
        new_worlds = self.generate_world()
        self.world[done_mask] = new_worlds[done_mask]
        self.flappy.y[done_mask]     = self.world_height / 2
        self.flappy.vy[done_mask]    = 0.0
        self.flappy.alive[done_mask] = 1.0
        self.t[done_mask]            = 0                