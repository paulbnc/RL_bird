import argparse

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    
    parser.add_argument("-lr", "--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    
    parser.add_argument("-B", "--batch_size", type=int, default=16,
                        help="Size of mini-batches")
    
    parser.add_argument("-op", "--optimizer", type=str, default='Adam',
                        help="which optimizer to use. Write between : ['Adam', 'SGD']. Default Adam.")
    
    parser.add_argument("-P", "--path", type=str, default='checkpoints', 
                        help="path to folder for model savings")

    parser.add_argument("-freq", "--freq", type=int, default=10,
                        help="freq to save model")
    
    parser.add_argument("-M", "--model", type=str, default='naive',
                        help="which model to train. Choose between : ['naive',]. Default naive.")


    args = parser.parse_args()