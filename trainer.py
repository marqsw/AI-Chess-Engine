import math
import os
import timeit
from copy import deepcopy

import numpy as np
from random import shuffle

import torch
import torch.optim as optim
import torch.nn as nn

import chess
import model
from mcts import MCTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:

    def __init__(self, game: chess.ChessBoard, model: model.ChessModel, args: dict) -> None:
        """
        Initialise the trainer
        @param game: game board to use
        @param model: neural network to use
        @param args: parameters
        """
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, self.model, self.args)
        self.temp_board = chess.ChessBoard()
        self.loss_pi = nn.CrossEntropyLoss()
        self.loss_v = nn.MSELoss()

    def execute_episode(self) -> list:
        """
        Play one game against itself using the current model.
        @return:
        """
        train_examples = []
        current_player = 1
        self.temp_board.state = self.game.state

        counter = 0

        while True:

            # initialise and expand the search tree
            self.mcts = MCTS(self.temp_board, self.model, self.args)
            root = self.mcts.run(self.model, self.temp_board.state, to_play=1)

            # select and save the best move
            visit_sum = sum(node.visit_count for node in root.children.values())
            action_probs = {k: v.visit_count / visit_sum for k, v in root.children.items()}
            action_probs = self.temp_board.action_prob_to_planes(action_probs)
            train_examples.append((self.temp_board.state, current_player, action_probs))

            # play the best move
            action = root.select_action(math.e ** (30 - counter) if counter < 30 else 1)

            self.temp_board.make_move(*action)

            # switch player
            current_player = current_player * -1
            self.temp_board.state = self.temp_board.view_from(-1)
            reward = self.temp_board.get_winner()

            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    ret.append(
                        (hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player)))
                    )

                print(self.temp_board.state_to_str(self.temp_board.state), self.temp_board.get_winner() * current_player)

                return ret

            counter += 1

    def learn(self) -> None:
        """
        Generate games by self-playing and learn from the games.
        @return:
        """
        filename = self.args['checkpoint_path']
        self.load_checkpoint(folder=".", filename=filename)

        for i in range(1, self.args['numIters'] + 1):

            print("{}/{}".format(i, self.args['numIters']))

            train_examples = []

            for eps in range(self.args['numEps']):
                start = timeit.default_timer()
                iteration_train_examples = self.execute_episode()
                train_examples.extend(iteration_train_examples)
                print(f'time taken: {timeit.default_timer() - start}')

            shuffle(train_examples)
            # noinspection PyTypeChecker
            self.train(train_examples)
            self.save_checkpoint(folder=".", filename=filename)

    def train(self,
              examples: list[
                  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, chess.Piece, int], np.ndarray, int
              ]) -> None:
        """
        Train the neural network using the generated games.
        @param examples: generated games
        @return:
        """
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        pi_losses = []
        v_losses = []

        for epoch in range(self.args['epochs']):
            self.model.train()

            batch_idx = 0

            while batch_idx < int(len(examples) / self.args['batch_size']):
                # print(batch_idx)
                sample_ids = np.random.randint(len(examples), size=self.args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                tmp = []

                for board in boards:
                    self.temp_board.state = board
                    tmp.append(self.temp_board.get_feature_planes())

                boards = torch.FloatTensor(np.concatenate(tmp))
                target_pis = torch.FloatTensor(np.concatenate(pis))
                target_vs = torch.FloatTensor(np.stack(np.array([vs]).astype(np.float64), axis=1))

                # predict
                boards = boards.contiguous().cuda()
                target_pis = target_pis.contiguous().cuda()
                target_vs = target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.model(boards)

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                pi_losses.append(float(l_pi))
                v_losses.append(float(l_v))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1

            print()
            print("Policy Loss", np.mean(pi_losses))
            print("Value Loss", np.mean(v_losses))

            with open('log.txt', 'w' if not os.path.isfile('log.txt') else 'a') as f:
                f.write(f"Policy Loss {np.mean(pi_losses)}\nValue Loss {np.mean(v_losses)}\n")

    def save_checkpoint(self, folder: str, filename: str) -> None:
        """
        Save the weights to file.
        @param folder: path to folder containing the save file
        @param filename: name of the save file
        @return:
        """
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(self.model.state_dict(), filepath)

    def load_checkpoint(self, folder: str, filename: str) -> None:
        """
        Load the weights from file.
        @param folder: path to folder containing the save file
        @param filename: name of the save 	file
        @return:
        """
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            self.model.load_state_dict(torch.load(filepath))
            print(f'loaded {filepath}')
