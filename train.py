from datetime import datetime

import torch

from chess import ChessBoard
from model import ChessModel
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 64,
    'numIters': 500,                                # Total number of training iterations
    'num_simulations': 20,                         # Total number of MCTS simulations to run when deciding on a move to play
    'numEps': 5,                                    # Number of full games (episodes) to run during each iteration
    'numItersForTrainExamplesHistory': 20,
    'epochs': 5,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights
}

game = ChessBoard()
model = ChessModel()

trainer = Trainer(game, model, args)
trainer.learn()