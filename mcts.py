import math
from copy import deepcopy

import numpy as np
import torch

import chess
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ucb_score(parent: 'Node', child: 'Node') -> float:
    """
    Calculate the ucb score for action between the parent and the child.
    @param parent: Parent node
    @param child: Child node
    @return:
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0

    return value_score + prior_score


class Node:
    def __init__(self, prior: float, to_play: int) -> None:
        """
        Node of the Monte Carlo Tree Search.
        @param prior: probabiltiy from neural network
        @param to_play: colour to play
        """
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children: dict[tuple, Node] = {}
        self.state = None

    def expanded(self) -> bool:
        """
        Whether if the node is expanded.
        @return: Status of expansion
        """
        return len(self.children) > 0

    def value(self) -> float:
        """
        Value of the node.
        @return: value of the node
        """
        if self.visit_count == 0:
            return 0

        return self.value_sum / self.visit_count

    def select_action(self, temperature: float) -> tuple:
        """
        Select the best action acccording to visit counts and temperature (randomness)
        @param temperature: temperature
        @return: action to take
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = actions[np.random.choice(len(actions))]
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = actions[np.random.choice(len(actions), p=visit_count_distribution)]

        return action

    def select_child(self) -> tuple[tuple, 'Node']:
        """
        Select the child with the highest UCB score.
        @return: action-state pair of the selected child
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self,
               state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, chess.Piece, int],
               to_play: int,
               action_probs: dict[tuple, float]
               ) -> None:
        """
        Expand the node and give prior probability to its children
        @param state: node's board state
        @param to_play: colour to move
        @param action_probs: action-probability pairs
        @return:
        """
        self.to_play = to_play
        self.state = state
        for a, prob in action_probs.items():
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self) -> str:
        """
        Return the state of the node.
        @return:
        """
        prior = "{0:.2f}".format(self.prior)
        return f"{chess.ChessBoard.state_to_str(self.state) if self.state is not None else ''} Prior: {prior} Count: {self.visit_count} Value: {self.value()}"


class MCTS:
    def __init__(self, game: chess.ChessBoard, model: model.ChessModel, args: dict):
        """
        Initisalise the search tree
        @param game: chessboard to play on
        @param model: neural network to use
        @param args: parameters
        """
        self.game = game
        self.model = model
        self.args = args
        self.temp_board = chess.ChessBoard()

    def run(self, model: model.ChessModel,
            state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, chess.Piece, int],
            to_play: int
            ) -> Node:
        """
        Run the search tree
        @param model: neural network to use
        @param state: board state to start from
        @param to_play: side to play
        @return: starting node of the search tree
        """
        root = Node(0, to_play)

        self.temp_board.state = state
        self.temp_board.state = self.temp_board.view_from(to_play)

        # EXPAND root
        action_probs, value = model.predict(self.temp_board.get_feature_planes().to(device))
        action_probs = self.game.to_action_prob_pair(action_probs)
        action_probs = {k: v / sum(action_probs.values()) for k, v in action_probs.items()}
        root.expand(state, to_play, action_probs)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            state = deepcopy(parent.state)

            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            self.temp_board.state = state
            self.temp_board.make_move(*action)
            # Get the board from the perspective of the other player
            self.temp_board.state = self.temp_board.view_from(-1)

            # The value of the new state from the perspective of the other player
            value = self.temp_board.get_winner()

            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = model.predict(self.temp_board.get_feature_planes().to(device))
                action_probs = self.temp_board.to_action_prob_pair(action_probs)
                action_probs = {k: v / sum(action_probs.values()) for k, v in action_probs.items()}
                node.expand(self.temp_board.state, parent.to_play * -1, action_probs)

            self.backpropagate(search_path, value, parent.to_play * -1)
        return root

    def backpropagate(self, search_path: list, value: float, to_play: int) -> None:
        """
        Backpropagate the evaluation back to the root after simulation, updating the values.
        @param search_path: path leading to root node
        @param value: value of the current node
        @param to_play: colour to play of the current node
        @return:
        """

        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1