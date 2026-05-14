
  Flappy Bird — Deep Reinforcement Learning
================================================================================
Paul Buannic & Arthur Devilliers

--------------------------------------------------------------------------------
  STRUCTURE DU PROJET
--------------------------------------------------------------------------------

```
RL_bird/
├── main.py
├── requirements.txt
├── game/
│   └── src/
│       ├── services/
│       │   ├── game_objects.py
│       │   ├── GIF.py
│       │   └── testing.py
│       ├── tests/
│       │   └── generate_world_test.py
│       └── play.py
├── RL/
│   ├── functions/
│   │   ├── TRAIN.py
│   │   ├── EVAL.py
│   │   ├── loss_functions/
│   │   │   └── DQN.py
│   │   └── utils/
│   │       ├── get_actions.py
│   │       ├── log.py
│   │       └── size.py
│   └── models/
│       ├── naive/
│       │   └── naive_model.py
│       ├── linear/
│       │   └── linear_model.py
│       ├── conv/
│       │   └── conv_model.py
│       └── based_on_position/
│           └── based_on_position.py
├── checkpoints/
└── game/plots/
```

--------------------------------------------------------------------------------
  UTILISATION
--------------------------------------------------------------------------------

Tout passe par main.py via des arguments en ligne de commande.

  Modes disponibles (--type)
  --------------------------
  eval              Évalue un modèle chargé et exporte des GIFs
  train_no_replay   Entraînement DQN sans experience replay
  train_replay      Entraînement DQN avec experience replay
  distances         Entraînement avec le modèle basé sur les distances
  test_world        Génère et sauvegarde une image PNG d'un monde

  Modèles disponibles (--model)
  -----------------------------
  naive             Baseline aléatoire (évaluation uniquement)
  small_linear      Réseau fully connected
  conv_small        Réseau convolutif
  distances         Features géométriques hand-crafted

  Arguments
  ---------
  ```
  -ty  / --type                   Mode d'exécution           (défaut : eval)
  -M   / --model                  Modèle à utiliser          (défaut : naive)
  -e   / --epochs                 Nombre d'epochs            (défaut : 100)
  -lr  / --lr                     Learning rate              (défaut : 0.0002)
  -B   / --batch_size             Taille du batch            (défaut : 8)
  -op  / --optimizer              Optimiseur Adam/SGD        (défaut : Adam)
  -g   / --gamma                  Facteur d'actualisation    (défaut : 0.98)
  -eps / --epsilon                ε start, end, decay        (défaut : 1. 0.05 0.99)
  -dif / --difficulty             Difficulté du monde 1-3    (défaut : 2)
  -H   / --height                 Hauteur du monde           (défaut : 200)
  -W   / --width                  Largeur du monde           (défaut : 800)
  -VW  / --view_width             Largeur de la vue agent    (défaut : 100)
  -P   / --path                   Dossier de sauvegarde      (défaut : checkpoints)
  -PP  / --plots_path             Dossier des GIFs           (défaut : game/plots/gifs_last)
  -freq / --freq                  Fréquence de sauvegarde    (défaut : 10)
  -LOAD / --load_model            Chemin vers un .pth        (défaut : None)
  -REPN / --experience_replay_size  Taille du replay buffer  (défaut : 12000)
  -DR  / --dead_reward            Pénalité de mort           (défaut : -100.)
  -AR  / --alive_reward           Récompense de survie       (défaut : 2.)
  -TSR / --tunnel_start_reward    Récompense entrée tunnel   (défaut : 50.)
  -TER / --tunnel_end_reward      Récompense sortie tunnel   (défaut : 50.)
  -SV  / --save                   Nombre de GIFs à sauver    (défaut : 1)
  -V   / --verbose                Affichage console 0/1      (défaut : 1)
```
  Exemples
  --------
  # Évaluer le modèle aléatoire
  ```python main.py --type eval --model naive --batch_size 4```

  # Entraîner le CNN sans replay
  ```python main.py --type train_no_replay --model conv_small --epochs 200 --difficulty 2 --path checkpoints/mon_run```

  # Reprendre depuis un checkpoint
  ```python main.py --type train_no_replay --model conv_small --load_model checkpoints/mon_run/epoch_50.pth```

  # Entraîner avec experience replay
  ```python main.py --type train_replay --model conv_small --epochs 500 --experience_replay_size 12000 --path checkpoints/replay_run```

  # Visualiser un monde généré
  ```python main.py --type test_world --difficulty 3 --height 200 --width 800```


--------------------------------------------------------------------------------
  DESCRIPTION DES MODULES
--------------------------------------------------------------------------------

  game/src/services/game_objects.py
  ----------------------------------
  C'est le coeur du simulateur. Ce fichier définit l'environnement de jeu dans
  son intégralité : génération procédurale des niveaux, physique de l'oiseau,
  détection de collisions et calcul des récompenses. Tout est représenté sous
  forme de tenseurs PyTorch, ce qui permet de faire tourner plusieurs parties
  en parallèle.

  Classe Game :
    Génère le monde sous la forme d'un tenseur (batch_size, hauteur, largeur).
    La difficulté est paramétrée sur 3 niveaux : elle influe sur la vitesse de
    défilement et le nombre de tuyaux placés sur le niveau. Les tuyaux sont
    répartis à partir du premier dixième de la carte ; la position verticale
    du trou dans chaque tuyau suit une chaîne de Markov (décalage aléatoire
    borné) pour garantir que la partie reste jouable. La méthode step() expose
    une fenêtre glissante de largeur VIEW_WIDTH — ce que l'agent voit à chaque
    frame. reset_dead() régénère uniquement les agents morts sans interrompre
    les autres.

  Classe Bird :
    Gère la physique de l'oiseau (gravité, vitesse verticale, saut), la
    détection de collisions avec les tuyaux et les bords, et le calcul des
    récompenses. Les récompenses sont entièrement configurables : pénalité de
    mort, récompense de survie par frame, bonus d'entrée et de sortie de tuyau.

  --

  RL/functions/TRAIN.py
  ----------------------
  Ce fichier contient les trois boucles d'entraînement. Elles partagent la
  même structure générale — générer un monde, jouer une partie frame par frame,
  calculer la loss DQN, mettre à jour le modèle — et se distinguent par la
  façon dont les transitions sont utilisées pour l'apprentissage.

  _train_dqn_no_replay :
    Mise à jour du modèle à chaque frame, directement sur la transition
    (état précédent, action, récompense, état suivant) produite à l'instant t.
    Simple et rapide, mais sensible à la corrélation temporelle des données.

  _train_dqn_replay :
    Même collecte de transitions, mais celles-ci sont stockées dans un buffer
    circulaire de taille N. À chaque frame, un mini-batch est tiré aléatoirement
    dans ce buffer pour la mise à jour, ce qui brise la corrélation temporelle
    et stabilise l'apprentissage.

  _train_distances_no_replay :
    Identique à no_replay, mais l'état passé au modèle inclut en plus la
    vitesse verticale de l'oiseau (vy), nécessaire au modèle basé sur les
    distances.

  Dans les trois cas : les poids sont sauvegardés toutes les `freq` epochs,
  le meilleur modèle (loss minimale) est conservé sous best.pth, et tous les
  messages sont écrits simultanément en console et dans console.txt.

  --

  RL/functions/EVAL.py
  ---------------------
  Lance une évaluation en mode inférence sur plusieurs parties simultanées.
  Sauvegarde jusqu'à `save` GIFs dans plots_path. Aucune mise à jour de poids
  n'est effectuée.

  --

  RL/functions/loss_functions/DQN.py
  -----------------------------------
  Implémente la loss DQN standard. La cible est calculée sans gradient sur les
  poids courants (détachée), ce qui correspond à la formulation classique avec
  réseau cible figé à theta_{i-1} :

    L = mean[ ( r + gamma * max_a'[ Q(s', a' ; theta_prev) ] - Q(s, a ; theta) )^2 ]

  --

  RL/models/
  -----------
  Quatre modèles sont disponibles, tous compatibles avec la même interface :
  forward retourne un tenseur de taille (batch_size, 2), soit une valeur Q par
  action possible.

  naive_model.py :
    Le modèle Naive tire ses actions aléatoirement selon un seuil de
    probabilité. Il sert uniquement de baseline lors de l'évaluation ; il
    n'a pas de paramètres et ne peut pas être entraîné.

  linear_model.py :
    LinearNN_small est un réseau fully connected à 3 couches. Il prend en
    entrée la vue du monde aplatie (hauteur × largeur), ce qui le rend
    relativement lourd pour de grandes résolutions.

  conv_model.py :
    ConvNN_small est un réseau convolutif à 3 couches Conv2D suivies de 2
    couches FC. Il prend en entrée 2 canaux : la vue du monde et le masque
    binaire indiquant la position de l'oiseau. La taille de la couche FC est
    calculée dynamiquement selon les dimensions de la vue (via conv_out_size).

  based_on_position.py :
    model_position ne travaille pas sur les pixels bruts. Il calcule d'abord
    4 features géométriques à partir de la vue : distance horizontale au
    prochain tuyau, écart entre le centre de l'oiseau et le bord haut du trou,
    écart avec le bord bas, et vitesse verticale vy. Ces features sont ensuite
    passées à un MLP à 3 couches (4 → 64 → 64 → 2).

  --

  RL/functions/utils/
  --------------------
  log.py       : _log() écrit un message à la fois en console et dans un
                 fichier texte. Utilisé partout dans TRAIN.py.

  size.py      : conv_out_size() calcule la dimension de sortie d'une couche
                 Conv2D selon kernel_size, stride et padding. Utilisé dans
                 ConvNN_small pour dimensionner la couche FC automatiquement.

  get_actions.py : action_index() convertit un label textuel ("saut" / "rien")
                   en indice entier (1 / 0).

  --

  game/src/services/GIF.py
  -------------------------
  Fonctions d'export visuel. color() applique une colorisation au tenseur
  binaire du monde : vert pour les tuyaux, bleu pour le ciel, rouge pour
  l'oiseau. gif() sauvegarde une séquence de frames colorisées en fichier .gif.
  save_png() sauvegarde une frame unique en .png. Les fichiers produits sont
  stockés dans game/plots/.

  --

  game/src/services/testing.py
  -----------------------------
  Utilitaire simple : génère un monde avec les paramètres donnés et le
  sauvegarde en PNG dans game/plots/world_test_generation_png/. Appelé via
  --type test_world.

  --

  game/src/tests/generate_world_test.py
  --------------------------------------
  Script de test indépendant qui instancie un Game, joue plusieurs epochs
  complètes avec le modèle aléatoire et mesure les performances (temps de
  génération, temps par epoch). Exporte un GIF par epoch.

  --

  game/src/play.py
  -----------------
  Non implémenté.


--------------------------------------------------------------------------------
  SAUVEGARDES
--------------------------------------------------------------------------------

À chaque run d'entraînement, le dossier --path contiendra :

```  checkpoints/mon_run/
  ├── best.pth          Meilleurs poids (loss minimale sur tout l'entraînement)
  ├── epoch_0.pth       Checkpoints intermédiaires (fréquence --freq)
  ├── epoch_10.pth
  ├── ...
  ├── loss.png          Courbe de la loss par epoch
  ├── time.png          Courbe des temps d'epoch
  └── console.txt       Log complet : paramètres, progression, temps estimé```

Les GIFs d'évaluation périodique sont dans le dossier --plots_path.
Les checkpoints des expériences menées sont dans checkpoints/.
Les GIFs correspondants sont dans game/plots/.
Se référer au rapport PDF pour l'analyse des résultats.

================================================================================