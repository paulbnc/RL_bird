import argparse
from RL.functions.EVAL import _eval
from RL.functions.TRAIN import _train_dqn_no_replay, _train_dqn_replay, _train_distances_no_replay
import torch
import os
import matplotlib.pyplot as plt
from game.src.services.testing import generate_world




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train a model.')


    parser.add_argument("-ty", "--type", type=str, default="eval",
                        help="Choose between [\"eval\", \"train_no_replay\", \"train_replay\", \"distances\", \"test_world\"]. Default eval")

    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Number of epochs for training. Default 100")
    
    parser.add_argument("-lr", "--lr", type=float, default=0.0002,
                      help="The learning rate to use for training. Default 0.0002")
    
    parser.add_argument("-B", "--batch_size", type=int, default=8,
                        help="Size of mini-batches. Default 8")
    
    parser.add_argument("-op", "--optimizer", type=str, default='Adam',
                        help="which optimizer to use. Write between : ['Adam', 'SGD']. Default Adam.")
    
    parser.add_argument("-P", "--path", type=str, default='checkpoints', 
                        help="path to folder for model savings. Default checkpoints")

    parser.add_argument("-freq", "--freq", type=int, default=10,
                        help="freq to save model. Default 10")
    
    parser.add_argument("-dif", "--difficulty", type=int, default=2,
                        help="difficulty (hint from paul : 1, 2, or 3, difficulty increases fast). Default 2")

    parser.add_argument("-H", "--height", type=int, default=200,
                        help="height of the world. Default 200")

    parser.add_argument("-W", "--width", type=int, default=800,
                        help="width of the world. Default 800")

    parser.add_argument("-VW", "--view_width", type=int, default=100,
                        help="width of what the model sees. Default 100")
    
    parser.add_argument("-TR", "--threshold", type=float, default=0.5,
                        help="threshold for decisions, default 0.5")

    parser.add_argument("-SV", "--save", type=int, default=1,
                        help="number of gifs to save during eval. Max batch size, min 0. Default 1")

    parser.add_argument("-M", "--model", type=str, default='naive',
                        help="which model to train. Choose between : ['naive', 'small_linear', 'conv_small', 'distances']. Default naive.")

    parser.add_argument("-PP", "--plots_path", type=str, default=os.path.join("game","plots","gifs_last"),
                        help="path for plots")

    parser.add_argument("-V", "--verbose", type=int, default=1,
                        help="1 for verbose, 0 for not. default 1")

    parser.add_argument("-g", "--gamma", type=float, default=0.98,
                        help="facteur d'actualisation / gamma : default 0.98 (futur=>proche de 1)")
    
    parser.add_argument("-eps", "--epsilon", nargs=3, type=float, default=[1., 0.05, 0.99],
                        help="probabilité que le modèle choisisse une action aléatoire : ~ mutation aléatoire. Renseigner 3 floats : epsilon_start epsilon_end decay. Defaut 1. 0.05 0.99")

    parser.add_argument("-DR", "--dead_reward", type=float, default=-100.,
                        help="pénalisation de la mort d'un individu : default -100.")

    parser.add_argument("-AR", "--alive_reward", type=float, default=2.,
                        help="récompense pour un individu d'être resté en vie : default 2.")

    parser.add_argument("-TSR", "--tunnel_start_reward", type=float, default=50.,
                        help="récompense pour un individu d'être entré dans un tuyau : default 50.")
    
    parser.add_argument("-TER", "--tunnel_end_reward", type=float, default=50.,
                        help="récompense pour un individu d'être sorti d'un tuyau : default 50.")

    parser.add_argument("-LOAD", "--load_model", type=str, default=None,
                        help="model to load (path) for training/eval. Default None.")

    parser.add_argument("-REPN", "--experience_replay_size", type=int, default=12000,
                        help="size of experience replay, default 12000 (best = batchsize*n_frames*number of epochs to remember)")

    args = parser.parse_args()

    if args.type=="test_world":
        generate_world(args.difficulty, args.height, args.width)
        exit(0)

    rewards = {"dead":args.dead_reward,
               "alive":args.alive_reward,
               "tunnel_start":args.tunnel_start_reward,
               "tunnel_end":args.tunnel_end_reward}

    #########



    if args.model=="naive":
        from RL.models.naive.naive_model import Naive
        model = Naive(args.batch_size)

    elif args.model=='small_linear':
        from RL.models.linear.linear_model import LinearNN_small
        model = LinearNN_small(
            view_height=args.height,
            view_width=args.view_width
        )
    elif args.model=='conv_small':
        from RL.models.conv.conv_model import ConvNN_small
        model = ConvNN_small(
            view_height=args.height,
            view_width=args.view_width
        )
    elif args.model=='distances':
        from RL.models.based_on_position.based_on_position import model_position, get_distance
        model = model_position()

    else:
        print(f"\n\nmodele {args.model} introuvable\n")
        raise Exception

    if args.load_model is not None:
        print(f"\n\nchargement de {args.load_model}\n\n")
        state_dict = torch.load(args.load_model)
        model.load_state_dict(state_dict)


    if args.model=='naive' and args.type!='eval':
        print("\n\nimpossible d'entraîner le réseau aléatoire.\n")
        raise Exception



    if args.type=="eval":
        _eval(
                model,
                batch_size=args.batch_size,
                difficulty=args.difficulty,
                height=args.height,
                width=args.width,
                VIEW_WIDTH=args.view_width,
                save=args.save,
                idx_save=1
            )
        

        
    elif args.type=="train_no_replay":

        if args.optimizer=="Adam":
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        elif args.optimizer=="SGD":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
        else:
            print(f"optimizer inconnu : {args.optimizer}")
            raise NameError


        temps, LOSSES, best_loss = _train_dqn_no_replay(
                                model=model,
                                epochs=args.epochs,
                                lr=args.lr,
                                optimizer=optimizer,
                                threshold=args.threshold,
                                difficulty=args.difficulty,
                                height=args.height,
                                width=args.width,
                                VIEW_WIDTH=args.view_width,
                                freq=args.freq,
                                gamma=args.gamma,
                                model_path=args.path,
                                plots_path=args.plots_path,
                                verbose=bool(args.verbose),
                                batch_size=args.batch_size,
                                rewards=rewards,
                                epsilon=args.epsilon
                            )
        
        plt.figure()
        plt.plot(LOSSES)
        plt.xlabel("parties jouées")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.savefig(os.path.join(args.path, "loss.png"))
        plt.close()

        plt.figure()
        plt.plot(temps)
        plt.xlabel("itérations")
        plt.ylabel("time")
        plt.title("Training time")
        plt.savefig(os.path.join(args.path, "time.png"))
        plt.close()

        print(f"best loss {best_loss}. \n\n****PLOTS DE LOSSES ET TIMES dans {args.path}\n\n")

    elif args.type=="train_replay":

        if args.optimizer=="Adam":
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        elif args.optimizer=="SGD":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
        else:
            print(f"optimizer inconnu : {args.optimizer}")
            raise NameError


        temps, LOSSES, best_loss = _train_dqn_replay(
                                N=args.experience_replay_size,
                                model=model,
                                epochs=args.epochs,
                                lr=args.lr,
                                optimizer=optimizer,
                                threshold=args.threshold,
                                difficulty=args.difficulty,
                                height=args.height,
                                width=args.width,
                                VIEW_WIDTH=args.view_width,
                                freq=args.freq,
                                gamma=args.gamma,
                                model_path=args.path,
                                plots_path=args.plots_path,
                                verbose=bool(args.verbose),
                                batch_size=args.batch_size,
                                rewards=rewards,
                                epsilon=args.epsilon
                            )
        
        plt.figure()
        plt.plot(LOSSES)
        plt.xlabel("parties jouées")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.savefig(os.path.join(args.path, "loss.png"))
        plt.close()

        plt.figure()
        plt.plot(temps)
        plt.xlabel("itérations")
        plt.ylabel("time")
        plt.title("Training time")
        plt.savefig(os.path.join(args.path, "time.png"))
        plt.close()

        print(f"best loss {best_loss}. \n\n****PLOTS DE LOSSES ET TIMES dans {args.path}\n\n")
    
    
    elif args.type=="distances":

        if args.optimizer=="Adam":
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        elif args.optimizer=="SGD":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
        else:
            print(f"optimizer inconnu : {args.optimizer}")
            raise NameError


        temps, LOSSES, best_loss = _train_distances_no_replay(
                                N=args.experience_replay_size,
                                model=model,
                                epochs=args.epochs,
                                lr=args.lr,
                                optimizer=optimizer,
                                threshold=args.threshold,
                                difficulty=args.difficulty,
                                height=args.height,
                                width=args.width,
                                VIEW_WIDTH=args.view_width,
                                freq=args.freq,
                                gamma=args.gamma,
                                model_path=args.path,
                                plots_path=args.plots_path,
                                verbose=bool(args.verbose),
                                batch_size=args.batch_size,
                                rewards=rewards,
                                epsilon=args.epsilon
                            )
        
        plt.figure()
        plt.plot(LOSSES)
        plt.xlabel("parties jouées")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.savefig(os.path.join(args.path, "loss.png"))
        plt.close()

        plt.figure()
        plt.plot(temps)
        plt.xlabel("itérations")
        plt.ylabel("time")
        plt.title("Training time")
        plt.savefig(os.path.join(args.path, "time.png"))
        plt.close()

        print(f"best loss {best_loss}. \n\n****PLOTS DE LOSSES ET TIMES dans {args.path}\n\n")