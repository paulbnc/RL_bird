import argparse
from RL.functions.EVAL import _eval
from RL.functions.TRAIN import _train_classic, _train_dqn_no_replay
import torch
import os

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train a model.')


    parser.add_argument("-ty", "--type", type=str, default="eval",
                        help="Choose between \"eval\" and \"train_no_replay\". Default eval")

    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Number of epochs for training. Default 100")
    
    parser.add_argument("-lr", "--lr", type=float, default=0.0002,
                      help="The learning rate to use for training. Default 0.0002")
    
    parser.add_argument("-B", "--batch_size", type=int, default=16,
                        help="Size of mini-batches. Default 16")
    
    parser.add_argument("-op", "--optimizer", type=str, default='Adam',
                        help="which optimizer to use. Write between : ['Adam', 'SGD']. Default Adam.")
    
    parser.add_argument("-P", "--path", type=str, default='checkpoints', 
                        help="path to folder for model savings. Default checkpoints")

    parser.add_argument("-freq", "--freq", type=int, default=10,
                        help="freq to save model. Default 10")
    
    parser.add_argument("-dif", "--difficulty", type=int, default=2,
                        help="difficulty (hint from paul : 1, 2, or 3, difficulty increases fast). Default 2")

    parser.add_argument("-H", "--height", type=int, default=100,
                        help="height of the world. Default 100")

    parser.add_argument("-W", "--width", type=int, default=1000,
                        help="width of the world. Default 1000")

    parser.add_argument("-VW", "--view_width", type=int, default=200,
                        help="width of what the model sees. Default 200")
    
    parser.add_argument("-TR", "--threshold", type=float, default=0.5,
                        help="threshold for decisions, default 0.5")

    parser.add_argument("-SV", "--save", type=int, default=1,
                        help="number of gifs to save during eval. Max batch size, min 0. Default 1")

    parser.add_argument("-M", "--model", type=str, default='naive',
                        help="which model to train. Choose between : ['naive','small_linear']. Default naive.")
    
    parser.add_argument("-PP", "--plots_path", type=str, default=os.path.join("game","plots","gifs_last"),
                        help="path for plots")
    
    parser.add_argument("-V", "--verbose", type=int, default=1,
                        help="1 for verbose, 0 for not. default 1")

    parser.add_argument("-g", "--gamma", type=float, default=0.95,
                        help="facteur d'actualisation / gamma : default 0.95 (futur=>proche de 1)")


    args = parser.parse_args()


    #########

    if args.model=='naive' and args.type=='train':
        print("\n\nimpossible d'entraîner le réseau aléatoire.\n")
        raise Exception


    if args.model=="naive":
        from RL.models.naive.naive_model import Naive
        model = Naive(args.batch_size)

    elif args.model=='small_linear':
        from RL.models.linear.linear_model import LinearNN_small
        model = LinearNN_small(
            view_height=args.height,
            view_width=args.view_width
        )

    else:
        print(f"\n\nmodele {args.model} introuvable\n")
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


        _train_dqn_no_replay(
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
                                batch_size=args.batch_size
                            )