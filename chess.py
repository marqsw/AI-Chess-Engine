import textwrap
from collections import deque
from enum import Enum

import numpy as np
import torch


class Piece(Enum):
    WHITE = 0
    BLACK = 1
    PAWNS = 2
    KNIGHTS = 3
    BISHOPS = 4
    ROOKS = 5
    QUEENS = 6
    KINGS = 7


# noinspection PyTypeChecker
class ChessBoard:
    NOT_A_FILE = np.uint64(0xfefefefefefefefe)
    NOT_H_FILE = np.uint64(0x7f7f7f7f7f7f7f7f)

    FIRST_RANK = np.uint64(0x00000000000000ff)
    EIGHTH_RANK = np.uint64(0xff00000000000000)

    EMPTY = np.uint64(0)
    UNIVERSE = np.uint64(0xffffffffffffffff)

    WHITE_EN_PASSANT = np.uint64(0x000000000000ff00)
    BLACK_EN_PASSANT = np.uint64(0x00ff000000000000)

    SYMBOLS = (
        (
            '',
            '',
            '♟︎',
            '♞',
            '♝',
            '♜',
            '♛',
            '♚'
        ),
        (
            '',
            '',
            '♙',
            '♘',

            '♗',
            '♖',
            '♕',
            '♔'
        )
    )

    def __init__(self, cache_steps: int = 40):
        """
        Initialise chessboard with optional parameters
        @param cache_steps: number of neural network input feature planes to cache
        """
        self.pieces = np.array(
            [
                [
                    np.uint64(0),
                    np.uint64(0),
                    np.uint64(0x000000000000ff00),  # white pawns
                    np.uint64(0x0000000000000042),  # white knights
                    np.uint64(0x0000000000000024),  # white bishops
                    np.uint64(0x0000000000000081),  # white rooks
                    np.uint64(0x0000000000000008),  # white queens
                    np.uint64(0x0000000000000010),  # white kings
                ],

                [
                    np.uint64(0),
                    np.uint64(0),
                    np.uint64(0x00ff000000000000),  # black pawns
                    np.uint64(0x4200000000000000),  # black knights
                    np.uint64(0x2400000000000000),  # black bishops
                    np.uint64(0x8100000000000000),  # black rooks
                    np.uint64(0x0800000000000000),  # black queens
                    np.uint64(0x1000000000000000),  # black kings
                ],
            ], dtype=np.uint64
        )

        # generate reachable square by the pieces on the next move, useful for king checks
        self.attacks = (
            (
                lambda: self.white_pawn_attacks(self.get_pawns(Piece.WHITE)),
                lambda: self.knight_attacks(self.get_knights(Piece.WHITE)),
                lambda: self.bishop_attacks(self.get_bishops(Piece.WHITE)),
                lambda: self.rook_attacks(self.get_rooks(Piece.WHITE)),
                lambda: self.queen_attacks(self.get_queens(Piece.WHITE)),
                lambda: self.king_attacks(self.get_kings(Piece.WHITE)),
            ),
            (
                lambda: self.black_pawn_attacks(self.get_pawns(Piece.BLACK)),
                lambda: self.knight_attacks(self.get_knights(Piece.BLACK)),
                lambda: self.bishop_attacks(self.get_bishops(Piece.BLACK)),
                lambda: self.rook_attacks(self.get_rooks(Piece.BLACK)),
                lambda: self.queen_attacks(self.get_queens(Piece.BLACK)),
                lambda: self.king_attacks(self.get_kings(Piece.BLACK)),
            )
        )

        # generate moves purely based on pieces, not taking in king checks
        self.pseudo_legal_gen = (
            (
                lambda x: np.uint64(0),
                lambda x: np.uint64(0),
                lambda x: self.white_pawn_moves(x),
                lambda x: self.knight_attacks(x) & ~ self.occupancy[Piece.WHITE.value],
                lambda x: self.bishop_attacks(x) & ~ self.occupancy[Piece.WHITE.value],
                lambda x: self.rook_attacks(x) & ~ self.occupancy[Piece.WHITE.value],
                lambda x: self.queen_attacks(x) & ~ self.occupancy[Piece.WHITE.value],
                lambda x: self.white_king_moves(x),
            ),
            (
                lambda x: np.uint64(0),
                lambda x: np.uint64(0),
                lambda x: self.black_pawn_moves(x),
                lambda x: self.knight_attacks(x) & ~ self.occupancy[Piece.BLACK.value],
                lambda x: self.bishop_attacks(x) & ~ self.occupancy[Piece.BLACK.value],
                lambda x: self.rook_attacks(x) & ~ self.occupancy[Piece.BLACK.value],
                lambda x: self.queen_attacks(x) & ~ self.occupancy[Piece.BLACK.value],
                lambda x: self.black_king_moves(x)

            )
        )

        # special rule flags
        self.en_passant_captures = np.array([np.uint64(0), np.uint64(0)])
        self.kingside_castling = np.array([True, True])
        self.queenside_castling = np.array([True, True])

        # to be updated by self.update_board()
        self.occupancy: list[np.uint64] = [None, None]
        self.empty = None

        self.colour_to_move = Piece.WHITE
        self.halfmove_clock = 0

        # a list of previous states and feature planes
        self.history: list[tuple] = []
        self.feature_plane_cache = deque([np.zeros((12, 8, 8))] * cache_steps, maxlen=cache_steps)

        self.update_board()

        self.test_board = None

    # -- interfaces --

    @property
    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Piece, int]:
        """
        @return: data representing current state of the chessboard
        """
        return (
            self.pieces.copy(),
            self.en_passant_captures.copy(),
            self.kingside_castling.copy(),
            self.queenside_castling.copy(),
            self.colour_to_move,
            self.halfmove_clock
        )

    @state.setter
    def state(self, value: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Piece, int]:
        """
        @param value: data representing the state to be applied
        """
        (
            self.pieces,
            self.en_passant_captures,
            self.kingside_castling,
            self.queenside_castling,
            self.colour_to_move,
            self.halfmove_clock,
        ) \
            = value

        self.update_board()

    def view_from(self, player: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Piece, int]:
        """
        Return the board state from the specified side.
        @return: flipped board data
        """
        if player == 1:
            return self.state

        return (
            np.flip(self.rotate_180(self.pieces.copy()), 0),
            np.flip(self.rotate_180(self.en_passant_captures.copy()), 0),
            np.flip(self.kingside_castling, 0),
            np.flip(self.queenside_castling, 0),
            Piece(abs(self.colour_to_move.value - 1)),
            self.halfmove_clock
        )

    @staticmethod
    def isolate_bits(b: np.uint64) -> list[np.uint64]:
        """
        Separate each bit onto standalone boards.
        @param b: bitboard with multiple pieces
        @return: list of boards containing one piece only
        """
        bits = []
        with np.errstate(over='ignore'):
            while b:
                bits.append(b & -b)
                b &= b - np.uint64(1)

        return bits

    def make_move(self, b: np.uint64, dest: np.uint64, colour: Piece = None, piece_type: Piece = None,
                  promotion_type: Piece = Piece.QUEENS, force=False):
        """
        Make a move on the board.
        @param b: current position of the piece
        @param dest: intended destination of the piece
        @param colour: colour of the piece to be moved
        @param piece_type: type of the piece to be moved
        @param promotion_type: type to be promoted to (if it is a pawn)
        @param force: force the move even it might be illegal
        """
        # DO NOT accept pieces that does not exist
        if not b & self.pieces[colour.value][piece_type.value]:
            return

        # check legality
        if not force and dest not in self.gen_legal(b, colour, piece_type):
            return

        self.history.append(self.state)

        enemy_index = abs(colour.value - 1)

        # 50-move rule
        if not len(np.argwhere(self.pieces & dest)) and piece_type is not Piece.PAWNS:  # no capture and not pawn
            self.halfmove_clock += 1
        else:
            self.halfmove_clock = 0

        # en passant capture
        if piece_type == Piece.PAWNS:
            if (colour == Piece.WHITE) and (self.south_one(dest) & self.en_passant_captures[Piece.WHITE.value]):
                self.pieces[Piece.BLACK.value][Piece.PAWNS.value] &= ~ self.south_one(dest)
            elif (colour == Piece.BLACK) and (self.north_one(dest) & self.en_passant_captures[Piece.BLACK.value]):
                self.pieces[Piece.WHITE.value][Piece.PAWNS.value] &= ~ self.north_one(dest)

        self.en_passant_captures &= np.uint64(0)

        if piece_type == Piece.PAWNS:
            if colour == Piece.WHITE:
                if dest & (b << np.uint64(16)):
                    self.en_passant_captures[Piece.BLACK.value] |= dest  # en passant

                if dest & self.EIGHTH_RANK:
                    self.pieces[Piece.WHITE.value][Piece.PAWNS.value] &= ~ b
                    piece_type = promotion_type

            else:
                if dest & (b >> np.uint64(16)):
                    self.en_passant_captures[Piece.WHITE.value] |= dest  # en passant

                if dest & self.FIRST_RANK:
                    self.pieces[Piece.BLACK.value][Piece.PAWNS.value] &= ~ b
                    piece_type = promotion_type

        # castling
        if piece_type == Piece.ROOKS:
            if colour == Piece.WHITE:
                if b & np.uint64(1):
                    self.queenside_castling[Piece.WHITE.value] = False

                elif b & np.uint64(128):
                    self.kingside_castling[Piece.WHITE.value] = False

            if colour == Piece.BLACK:
                if b & np.uint64(72057594037927936):
                    self.queenside_castling[Piece.BLACK.value] = False

                if b & np.uint64(9223372036854775808):
                    self.kingside_castling[Piece.BLACK.value] = False

        if piece_type == Piece.KINGS:
            # castle
            if not dest & self.king_attacks(b):
                if dest & self.east_one(self.east_one(b)):
                    self.pieces[colour.value][Piece.ROOKS.value] &= ~ self.east_attacks(b, self.empty)
                    self.pieces[colour.value][Piece.ROOKS.value] |= self.west_one(dest)
                    self.pieces[enemy_index] &= ~ self.west_one(dest)

                elif dest & self.west_one(self.west_one(b)):
                    self.pieces[colour.value][Piece.ROOKS.value] &= ~ self.west_attacks(b, self.empty)
                    self.pieces[colour.value][Piece.ROOKS.value] |= self.east_one(dest)
                    self.pieces[enemy_index] &= ~ self.east_one(dest)

            if colour == Piece.WHITE:
                self.queenside_castling[Piece.WHITE.value] = False
                self.kingside_castling[Piece.WHITE.value] = False

            elif colour == Piece.BLACK:
                self.queenside_castling[Piece.BLACK.value] = False
                self.kingside_castling[Piece.BLACK.value] = False

        # move piece
        self.pieces[colour.value][piece_type.value] &= ~ b
        self.pieces[colour.value][piece_type.value] |= dest
        self.pieces[enemy_index] &= ~ dest
        self.update_board()
        self.colour_to_move = Piece(enemy_index)

    def gen_legal(self, b: np.uint64, colour: Piece, piece_type: Piece) -> list[np.uint64]:
        """
        Generate legal moves of a selected piece.
        @param b: bitboard of the piece square
        @param colour: colour of the piece
        @param piece_type: type of the piece
        @return: a list of bitboards each representing a legal final position after one move
        """
        moves = self.isolate_bits(self.pseudo_legal_gen[colour.value][piece_type.value](b))

        if self.test_board is None:
            self.test_board = ChessBoard(cache_steps=1)

        for move in moves.copy():
            self.test_board.pieces = self.pieces.copy()
            self.test_board.update_board()
            self.test_board.make_move(b, move, colour, piece_type, force=True)

            if self.test_board.get_kings(colour) & self.test_board.gen_attacks(Piece(abs(colour.value - 1))):
                moves.remove(move)

        return moves

    def gen_attacks(self, colour: Piece) -> np.uint64:
        """
        Generate squares that could be reached by a colour on the next move.
        @param colour: colour of the side to be generated
        @return: a bitboard representing all the square that can be reached by the specified colour on the next move
        """
        return np.bitwise_or.reduce([f() for f in self.attacks[colour.value]])

    def update_board(self):
        """
        Update overall board occupancy. To be used after modifying pieces' position.
        """
        self.occupancy[0] = np.bitwise_or.reduce(self.pieces[0])
        self.occupancy[1] = np.bitwise_or.reduce(self.pieces[1])
        self.empty = ~ np.bitwise_or.reduce(self.occupancy)
        self.feature_plane_cache.append([self.bitboard_to_ndarray(i) for i in self.pieces[:, 2:].flatten()])

    def get_winner(self) -> None | int:
        """
        Return the result of the game
        @return:
            None: Game is not ended
            1: White winsc
            -1: Black wins
            0: Stalemate
        """
        moves = [False, False]

        # get move for every piece
        for colour, colour_set in enumerate(self.pieces):
            for piece_type, pieces in enumerate(colour_set):
                for piece in self.isolate_bits(pieces):
                    if len(self.gen_legal(piece, Piece(colour), Piece(piece_type))):
                        moves[colour] = True
                        break

                    if moves[colour] is True:
                        break

        return (
            # 50 move rules
            0 if self.halfmove_clock > 100 else

            # game not ended
            None if (moves[0] & moves[1]) else

            # checkmates
            1 if moves[0] and self.get_kings(Piece.BLACK) & self.gen_attacks(Piece.WHITE) else
            -1 if moves[1] and self.get_kings(Piece.WHITE) & self.gen_attacks(Piece.BLACK) else

            # stalemate if both have no moves
            0
        )

    # -- neural network --

    def get_feature_planes(self) -> torch.Tensor:
        """
        Get input feature planes from current state for the neural network.
        @return: ndarray of shape (cache_steps + 6, 8, 8)
        """

        constant_valued_input = \
            [self.bitboard_to_ndarray(i) for i in np.concatenate(
                [
                    np.uint64(self.kingside_castling),  # kingside castling
                    np.uint64(self.queenside_castling),  # queenside castling

                    # single values as list for concatenation
                    [
                        np.uint64(self.halfmove_clock),  # no progress count
                        np.uint64(self.colour_to_move.value),  # player's colour
                    ]
                ]
            )
             ]

        return torch.FloatTensor(
            np.concatenate((np.concatenate(self.feature_plane_cache), constant_valued_input))
        ).unsqueeze(dim=0)

    def to_action_prob_pair(self, feature_planes: torch.Tensor) -> dict[tuple, float]:
        """
        Interpret move probabilities output from neural network.
        @param feature_planes: output planes from neural network
        @return: dictionary of action-value pairs
        """
        feature_planes = feature_planes.cpu().flatten()
        action_list: list[tuple] = []
        prob_list: list[float] = []

        for piece_type, pieces in enumerate(self.pieces[Piece.WHITE.value]):  # always looking from white's perspective
            piece_type = Piece(piece_type)

            for piece in self.isolate_bits(pieces):  # separate pieces
                piece_index = self.get_indices(piece)[0]
                promotion_type = Piece.QUEENS

                if piece_type == Piece.KNIGHTS:
                    knight_move_indices = np.array(
                        [
                            piece << np.uint64(17),  # noNoEa
                            piece << np.uint64(10),  # noEaEa
                            piece >> np.uint64(6),  # soEaEa
                            piece >> np.uint64(15),  # soSoEa
                            piece >> np.uint64(17),  # soSoWe
                            piece >> np.uint64(10),  # soWeWe
                            piece << np.uint64(6),  # noWeWe
                            piece << np.uint64(15)  # noNoWe
                        ]
                    )

                for move in self.gen_legal(piece, Piece.WHITE, piece_type):
                    move_index = self.get_indices(move)[0]

                    if piece_type == Piece.KNIGHTS:
                        # noinspection PyUnboundLocalVariable
                        plane_index = 56 + np.where(knight_move_indices & move)[0][0]

                    elif piece_type == Piece.PAWNS:
                        plane_index = []
                        if move & self.EIGHTH_RANK:  # under-promotion
                            if move & self.north_attacks(piece, self.UNIVERSE):
                                plane_index = [64, 67, 70]
                            elif move & self.northeast_attacks(piece, self.UNIVERSE):
                                plane_index = [65, 68, 71]
                            elif move & self.northwest_attacks(piece, self.UNIVERSE):
                                plane_index = [66, 69, 72]

                            prob = feature_planes[plane_index]
                            max_index = np.argmax(prob)

                            promotion_type = Piece(int(max_index) + 3)
                            plane_index = int(prob[max_index])

                        else:
                            if move & self.north_attacks(piece, self.UNIVERSE):
                                plane_index = abs(int(piece_index) - int(move_index)) // 8 - 1
                            elif move & self.northeast_attacks(piece, self.UNIVERSE):
                                plane_index = abs(int(piece_index) - int(move_index)) // 9 + 6
                            elif move & self.northwest_attacks(piece, self.UNIVERSE):
                                plane_index = abs(int(piece_index) - int(move_index)) // 7 + 48

                    else:
                        plane_index = abs(int(piece_index) - int(move_index))

                        if move & self.north_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index // 8 - 1
                        elif move & self.northeast_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index // 9 + 6
                        elif move & self.east_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index + 13
                        elif move & self.southeast_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index // 7 + 20
                        elif move & self.south_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index // 8 + 27
                        elif move & self.southwest_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index // 9 + 34
                        elif move & self.west_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index + 41
                        elif move & self.northwest_attacks(piece, self.UNIVERSE):
                            plane_index = plane_index // 7 + 48

                    action_list.append((piece, move, Piece.WHITE, piece_type, promotion_type))
                    prob_list.append(feature_planes[plane_index])

        return dict(zip(action_list, prob_list))

    def action_prob_to_planes(self,
                              action_probs: dict[tuple, float]
                              ) -> np.ndarray:
        feature_planes = np.zeros((1, 73), dtype=float)

        for (piece, move, piece_colour, piece_type, promotion_type), prob in action_probs.items():
            piece_index = self.get_indices(piece)[0]
            move_index = self.get_indices(move)[0]

            if piece_type == Piece.KNIGHTS:
                knight_move_indices = np.array(
                    [
                        piece << np.uint64(17),  # noNoEa
                        piece << np.uint64(10),  # noEaEa
                        piece >> np.uint64(6),  # soEaEa
                        piece >> np.uint64(15),  # soSoEa
                        piece >> np.uint64(17),  # soSoWe
                        piece >> np.uint64(10),  # soWeWe
                        piece << np.uint64(6),  # noWeWe
                        piece << np.uint64(15)  # noNoWe
                    ]
                )

                plane_index = 56 + np.where(knight_move_indices & move)[0][0]

            elif piece_type == Piece.PAWNS:
                if move & self.EIGHTH_RANK:
                    if move & self.north_attacks(piece, self.UNIVERSE):
                        plane_index = 64 + 3 * (promotion_type.value - 3)
                    elif move & self.northeast_attacks(piece, self.UNIVERSE):
                        plane_index = 65 + 3 * (promotion_type.value - 3)
                    elif move & self.northwest_attacks(piece, self.UNIVERSE):
                        plane_index = 66 + 3 * (promotion_type.value - 3)

                else:
                    if move & self.north_attacks(piece, self.UNIVERSE):
                        plane_index = abs(int(piece_index) - int(move_index)) // 8 - 1
                    elif move & self.northeast_attacks(piece, self.UNIVERSE):
                        plane_index = abs(int(piece_index) - int(move_index)) // 9 + 6
                    elif move & self.northwest_attacks(piece, self.UNIVERSE):
                        plane_index = abs(int(piece_index) - int(move_index)) // 7 + 48

            else:
                plane_index = abs(int(piece_index) - int(move_index))
                if move & self.north_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index // 8 - 1
                elif move & self.northeast_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index // 9 + 6
                elif move & self.east_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index + 13
                elif move & self.southeast_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index // 7 + 20
                elif move & self.south_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index // 8 + 27
                elif move & self.southwest_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index // 9 + 34
                elif move & self.west_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index + 41
                elif move & self.northwest_attacks(piece, self.UNIVERSE):
                    plane_index = plane_index // 7 + 48

            # noinspection PyUnboundLocalVariable
            feature_planes[0][plane_index] += prob

            feature_planes /= np.sum(feature_planes)

        return feature_planes

    # -- conversion --

    @staticmethod
    def bitboard_to_ndarray(b: np.uint64) -> np.ndarray:
        """
        Convert a bitboard into a 2D ndarray
        @param b: bitboard to be converted
        @return: two-dimensional 8x8 ndarray starting with square index 0 on topleft
        """
        mask = np.uint64(1) << np.arange(64, dtype=np.uint64)
        return np.reshape((b & mask) >> np.arange(64, dtype=np.uint64), (8, 8)).astype(np.int8)

    @staticmethod
    def get_indices(b: np.uint64) -> list[np.uint64]:
        """
        Return occupied square indices.
        @param b: bitboard with occupancy
        @return: a list of occupied indices
        """
        indices = []
        with np.errstate(over='ignore'):
            while b:
                indices.append(np.uint64(np.log2(b & -b)))
                b &= b - np.uint64(1)  # Reset LS1B

        return indices

    @staticmethod
    def index_to_board(index: int) -> np.uint64:
        """
        Convert a square index into bitboard.
        @param index: square index of a square
        @return: bitboard of the specified squared
        """
        return np.uint64(1) << np.uint64(index)

    @staticmethod
    def coord_to_index(coord: tuple[int, int]) -> int:
        """
        Convert a pair of coordinates to square index
        @param coord: coordinates to be converted
        @return: converted square index from coordinates
        """
        return coord[0] + coord[1] * 8

    @staticmethod
    def index_to_coord(index) -> tuple[int, int]:
        """
        Convert square index into a pair of coordinates
        @param index: square index to be converted
        @return: converted coordinates from square index
        """
        return int(index % 8), int(index // 8)

    # -- visualisation --

    @staticmethod
    def state_to_str(state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Piece, int]) -> str:
        return_str = ['．'] * 64

        for piece_colour in [Piece.WHITE, Piece.BLACK]:
            for piece_type in list(Piece)[2:]:
                for index in ChessBoard.get_indices(state[0][piece_colour.value][piece_type.value]):
                    return_str[index] = ChessBoard.SYMBOLS[piece_colour.value][piece_type.value]

        return ''.join(reversed([''.join(return_str[i: i + 8]) + '\n' for i in range(0, len(return_str), 8)]))

    @staticmethod
    def bitboard_to_str(b: np.uint) -> str:
        """
        Convert a bitboard into a string of 0s and 1s.
        @param b: bitboard to be converted
        @return: a string of 0s and 1s to be printed for visualisation
        """

        result = ''

        for i in textwrap.wrap(format(b, '064b'), 8):
            result += i[::-1] + '\n'
        return result.replace('0', '．').replace('1', 'Ｏ')

    # -- piece sets --
    def get_pawns(self, colour: Piece) -> np.uint64:
        """
        Determine the positions of pawns of a specified colour.
        @param colour: colour of the pawns to acquire
        @return: a bitboard representing the positions of the pawns
        """
        return self.pieces[colour.value, Piece.PAWNS.value]

    def get_knights(self, colour: Piece) -> np.uint64:
        """
        Determine the positions of knights of a specified colour.
        @param colour: colour of the knights to acquire
        @return: a bitboard representing the positions of the knights
        """
        return self.pieces[colour.value, Piece.KNIGHTS.value]

    def get_bishops(self, colour: Piece) -> np.uint64:
        """
        Determine the positions of bishops of a specified colour.
        @param colour: colour of the bishops to acquire
        @return: a bitboard representing the positions of the bishops
        """
        return self.pieces[colour.value, Piece.BISHOPS.value]

    def get_rooks(self, colour: Piece) -> np.uint64:
        """
        Determine the positions of rooks of a specified colour.
        @param colour: colour of the rooks to acquire
        @return: a bitboard representing the position of the rooks
        """
        return self.pieces[colour.value, Piece.ROOKS.value]

    def get_queens(self, colour: Piece) -> np.uint64:
        """
        Determine the positions of queens of a specified colour.
        @param colour: colour of the queens to acquire
        @return: a bitboard representing the position of the queens
        """
        return self.pieces[colour.value, Piece.QUEENS.value]

    def get_kings(self, colour: Piece) -> np.uint64:
        """
        Determine the positions of kings of a specified colour.
        @param colour: colour of the kings to acquire
        @return: a bitboard representing the position of the kings
        """
        return self.pieces[colour.value, Piece.KINGS.value]

    # -- legal move generations --

    def white_pawn_moves(self, pawns: np.uint64) -> np.uint64:
        """
        Pseudo-legal move generation of black pawns.
        @param pawns: bitboard representing positions of the white pawns
        @return: a bitboard representing pseudo-legal moves of the white pawns
        """
        # advancement
        moves = self.north_one(pawns) & self.empty

        # en passant
        moves |= self.north_one(self.north_one(self.south_one(moves) & self.WHITE_EN_PASSANT)) & self.empty

        # en passant captures
        moves |= self.north_one(self.east_one(pawns) & self.en_passant_captures[Piece.WHITE.value])
        moves |= self.north_one(self.west_one(pawns) & self.en_passant_captures[Piece.WHITE.value])

        # captures
        moves |= self.white_pawn_attacks(pawns) & self.occupancy[Piece.BLACK.value]

        return moves & ~ self.occupancy[Piece.WHITE.value]

    def black_pawn_moves(self, pawns: np.uint64) -> np.uint64:
        """
        Pseudo-legal move generation of white pawns.
        @param pawns: bitboard representing positions of the black pawns
        @return: a bitboard representing pseudo-legal moves of the black pawns
        """
        # advancement
        moves = self.south_one(pawns) & self.empty

        # en passant
        moves |= self.south_one(self.south_one(self.north_one(moves) & self.BLACK_EN_PASSANT)) & self.empty

        # en passant captures
        moves |= self.south_one(self.east_one(pawns) & self.en_passant_captures[Piece.BLACK.value])
        moves |= self.south_one(self.west_one(pawns) & self.en_passant_captures[Piece.BLACK.value])

        # captures
        moves |= self.black_pawn_attacks(pawns) & self.occupancy[Piece.WHITE.value]

        return moves & ~ self.occupancy[Piece.BLACK.value]

    def white_king_moves(self, kings: np.uint64) -> np.uint64:
        """
        Pseudo-legal move generation of white kings.
        @param kings: bitboard representing the positions of the white kings
        @return: a btiboard representing pseudo-legal moves of the white kings
        """
        moves = self.king_attacks(kings)

        if (self.kingside_castling[Piece.WHITE.value]
                and self.east_attacks(kings, self.empty) & self.get_rooks(Piece.WHITE)
                and self.east_one(self.get_kings(Piece.WHITE)) &
                ~ np.bitwise_or.reduce(self.gen_attacks(Piece.BLACK))
        ):
            moves |= self.east_one(self.east_one(kings))

        if (self.queenside_castling[Piece.WHITE.value]
                and self.west_attacks(kings, self.empty) & self.get_rooks(Piece.WHITE)
                and self.west_one(self.get_kings(Piece.WHITE)) &
                ~ np.bitwise_or.reduce(self.gen_attacks(Piece.BLACK))

        ):
            moves |= self.west_one(self.west_one(kings))

        return moves & ~ self.occupancy[Piece.WHITE.value]

    def black_king_moves(self, kings: np.uint64) -> np.uint64:
        """
        Pseudo-legal move generation of the black kings.
        @param kings: bitboard representing the positions of the black kings
        @return: a bitboard representing pseudo-legal moves of the black kings
        """
        moves = self.king_attacks(kings)

        if (self.kingside_castling[Piece.BLACK.value]
                and self.east_attacks(kings, self.empty) & self.get_rooks(Piece.BLACK)
                and self.east_one(self.get_kings(Piece.BLACK)) &
                ~ np.bitwise_or.reduce(self.gen_attacks(Piece.WHITE))
        ):
            moves |= self.east_one(self.east_one(kings))

        if (self.queenside_castling[Piece.BLACK.value]
                and self.west_attacks(kings, self.empty) & self.get_rooks(Piece.BLACK)
                and self.west_one(self.get_kings(Piece.BLACK)) &
                ~ np.bitwise_or.reduce(self.gen_attacks(Piece.WHITE))
        ):
            moves |= self.west_one(self.west_one(kings))

        return moves & ~ self.occupancy[Piece.BLACK.value]

    # -- piece attacks  --

    def white_pawn_attacks(self, pawns: np.uint64) -> np.uint64:
        """
        Generate squares could be attacked by white pawns on the next move.
        @param pawns: bitboard representing the positions of the white pawns
        @return: a bitboard representing squares that can be attacked by white pawns on the next move
        """
        return self.northeast_one(pawns) | self.northwest_one(pawns)

    def black_pawn_attacks(self, pawns: np.uint64) -> np.uint64:
        """
        Generate squares could be attacked by black pawns on the next move.
        @param pawns: bitboard representing the positions of the black pawns
        @return: a bitboard representing squares that can be attacked by the black pawns on the next move
        """
        return self.southeast_one(pawns) | self.southwest_one(pawns)

    @staticmethod
    def knight_attacks(knights: np.uint64) -> np.uint64:
        """
        Generate square could be attacked by knights on the next move.
        @param knights: bitboard representing the positions of the knights
        @return: a bitboard representing squares that can be attacked by the knights on the next move
        """
        l1 = (knights >> np.uint64(1)) & np.uint64(0x7f7f7f7f7f7f7f7f)
        l2 = (knights >> np.uint64(2)) & np.uint64(0x3f3f3f3f3f3f3f3f)
        r1 = (knights << np.uint64(1)) & np.uint64(0xfefefefefefefefe)
        r2 = (knights << np.uint64(2)) & np.uint64(0xfcfcfcfcfcfcfcfc)
        h1 = l1 | r1
        h2 = l2 | r2
        return (h1 << np.uint64(16)) | (h1 >> np.uint64(16)) | (h2 << np.uint64(8)) | (h2 >> np.uint64(8))

    def bishop_attacks(self, bishops: np.uint64) -> np.uint64:
        """
        Generate squares could be attacked by the bishops on the next move.
        @param bishops: bitboard representing the positions of the bishops
        @return: a bitboard representing square that can be attacked by the bishops on the next move
        """
        return (
                self.northeast_attacks(bishops, self.empty) |
                self.southeast_attacks(bishops, self.empty) |
                self.southwest_attacks(bishops, self.empty) |
                self.northwest_attacks(bishops, self.empty)
        )

    def rook_attacks(self, rooks: np.uint64) -> np.uint64:
        """
        Generate squares could be attacked by the bishops on the next move.
        @param rooks: bitboard representing the positions of the rooks
        @return: a bitboard representing squares that can be attacked by the rooks on the next move
        """
        return (
                self.south_attacks(rooks, self.empty) |
                self.north_attacks(rooks, self.empty) |
                self.east_attacks(rooks, self.empty) |
                self.west_attacks(rooks, self.empty)
        )

    def queen_attacks(self, queens: np.uint64) -> np.uint64:
        """
        Generate squares could be attacked by the queens on the next move.
        @param queens: bitboard representing the positions of the queens
        @return: a bitboard representing squares that can be attacked by the queens on the next move
        """
        return self.rook_attacks(queens) | self.bishop_attacks(queens)

    def king_attacks(self, kings: np.uint64) -> np.uint64:
        """
        Generate squares could be attacked by the kings on the next move.
        @param kings: bitboard representing the positions of the kings.
        @return: a bitboard representing squares that can be attacked by the kings on the next move
        """
        attacks = self.east_one(kings) | self.west_one(kings)
        kings |= attacks
        attacks |= self.north_one(kings) | self.south_one(kings)
        return attacks

    # -- board transformations --
    @staticmethod
    def south_one(b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards south by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return b >> np.uint64(8)

    @staticmethod
    def north_one(b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards north by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return b << np.uint64(8)

    def east_one(self, b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards east by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return (b << np.uint64(1)) & self.NOT_A_FILE

    def northeast_one(self, b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards northeast by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return (b << np.uint64(9)) & self.NOT_A_FILE

    def southeast_one(self, b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards southeast by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return (b >> np.uint64(7)) & self.NOT_A_FILE

    def west_one(self, b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards west by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return (b >> np.uint64(1)) & self.NOT_H_FILE

    def southwest_one(self, b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards southwest by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return (b >> np.uint64(9)) & self.NOT_H_FILE

    def northwest_one(self, b: np.uint64) -> np.uint64:
        """
        Shift bitboard towards northwest by one square.
        @param b: bitboard to be shifted
        @return: transformed bitboard
        """
        return (b << np.uint(7)) & self.NOT_H_FILE

    @staticmethod
    def gen_shift(x: np.uint64, s: np.uint64) -> np.uint64:
        """
        General shift of a 64-bit integer.
        @param x: 64-bit integer to be shifted
        @param s: direction and magnitude of the shift (positive is leftward and vice versa)
        @return: a shifted 64-bit integer
        """
        return np.uint64(x << s) if s > 0 else np.uint64(x >> -s)

    @staticmethod
    def flip_vertical(b: np.uint64) -> np.uint64:
        """
        Vertically flip the bitboard.
        @param b: bitboard to be flipped
        @return: transformed bitboard
        """
        k1 = np.uint64(0x00FF00FF00FF00FF)
        k2 = np.uint64(0x0000FFFF0000FFFF)

        b = ((b >> np.uint64(8)) & k1) | ((b & k1) << np.uint64(8))
        b = ((b >> np.uint64(16)) & k2) | ((b & k2) << np.uint64(16))
        b = (b >> np.uint64(32)) | (b << np.uint64(32))

        return b

    @staticmethod
    def mirror_horizontal(b: np.uint64) -> np.uint64:
        """
        Horizontally mirror the bitboard.
        @param b: bitboard to be mirrored
        @return: transformed bitboard
        """
        k1 = np.uint64(0x5555555555555555)
        k2 = np.uint64(0x3333333333333333)
        k4 = np.uint64(0x0f0f0f0f0f0f0f0f)

        b = ((b >> np.uint64(1)) & k1) | ((b & k1) << np.uint64(1))
        b = ((b >> np.uint64(2)) & k2) | ((b & k2) << np.uint64(2))
        b = ((b >> np.uint64(4)) & k4) | ((b & k4) << np.uint64(4))

        return b

    @staticmethod
    def flip_diagonal_a1h8(b: np.uint64) -> np.uint64:
        """
        Flip the bitboard along the A1-H8 diagonal.
        @param b: bitboard to be flipped
        @return: transformed bitboard
        """
        k1 = np.uint64(0x5500550055005500)
        k2 = np.uint64(0x3333000033330000)
        k4 = np.uint64(0x0f0f0f0f00000000)

        t = k4 & (b ^ (b << np.uint64(28)))
        b ^= t ^ (t >> np.uint64(28))

        t = k2 & (b ^ (b << np.uint64(14)))
        b ^= t ^ (t >> np.uint64(14))

        t = k1 & (b ^ (b << np.uint64(7)))
        b ^= t ^ (t >> np.uint64(7))

        return b

    @staticmethod
    def flip_diagonal_a8h1(b: np.uint64) -> np.uint64:
        """
        Flip the bitboard along the A8-H1 diagonal.
        @param b: bitboard to be flipped
        @return: transformed bitboard
        """
        k1 = np.uint64(0xaa00aa00aa00aa00)
        k2 = np.uint64(0xcccc0000cccc0000)
        k4 = np.uint64(0xf0f0f0f00f0f0f0f)

        t = b ^ (b << np.uint64(36))
        b ^= k4 & (t ^ (b >> np.uint64(36)))

        t = k2 & (b ^ (b << np.uint64(18)))
        b ^= t ^ (t >> np.uint64(18))

        t = k1 & (b ^ (b << np.uint64(9)))
        b ^= t ^ (t >> np.uint64(9))

        return b

    def rotate_180(self, b: np.uint64) -> np.uint64:
        """
        Rotate bitboard by 180 degrees
        @param b: bitboard to be rotated
        @return: transformed bitboard
        """
        return self.mirror_horizontal(self.flip_vertical(b))

    def rotate_90_clockwise(self, b: np.uint64) -> np.uint64:
        """
        Rotate bitboard by 90 degrees in clockwise direction
        @param b: bitboard to be rotated
        @return: transformed bitboard
        """
        return self.flip_vertical(self.flip_diagonal_a1h8(b))

    def rotate_90_anticlockwise(self, b: np.uint64) -> np.uint64:
        """
        Rotate bitboard by 90 degrees in anti-clockwise direction
        @param b: bitboard to be rotated
        @return: transformed bitboard
        """
        return self.flip_diagonal_a1h8(self.flip_vertical(b))

    # -- flood fill --
    @staticmethod
    def south_attacks(rooks: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares south of certain squares.
        @param rooks: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares south of the specified squares filled
        """
        flood = rooks
        flood |= (rooks := (rooks >> np.uint64(8)) & empty)
        flood |= (rooks := (rooks >> np.uint64(8)) & empty)
        flood |= (rooks := (rooks >> np.uint64(8)) & empty)
        flood |= (rooks := (rooks >> np.uint64(8)) & empty)
        flood |= (rooks := (rooks >> np.uint64(8)) & empty)
        flood |= (rooks >> np.uint64(8)) & empty
        return flood >> np.uint64(8)

    @staticmethod
    def north_attacks(rooks: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares north of certain squares.
        @param rooks: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares north of the specified squares filled
        """
        flood = rooks
        flood |= (rooks := (rooks << np.uint64(8)) & empty)
        flood |= (rooks := (rooks << np.uint64(8)) & empty)
        flood |= (rooks := (rooks << np.uint64(8)) & empty)
        flood |= (rooks := (rooks << np.uint64(8)) & empty)
        flood |= (rooks := (rooks << np.uint64(8)) & empty)
        flood |= (rooks << np.uint64(8)) & empty
        return flood << np.uint64(8)

    @staticmethod
    def east_attacks(rooks: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares east of certain squares.
        @param rooks: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares east of the specified squares filled
        """
        notA = np.uint64(0xfefefefefefefefe)
        flood = rooks
        empty &= notA
        flood |= (rooks := (rooks << np.uint64(1)) & empty)
        flood |= (rooks := (rooks << np.uint64(1)) & empty)
        flood |= (rooks := (rooks << np.uint64(1)) & empty)
        flood |= (rooks := (rooks << np.uint64(1)) & empty)
        flood |= (rooks := (rooks << np.uint64(1)) & empty)
        flood |= (rooks << np.uint64(1)) & empty
        return (flood << np.uint64(1)) & notA

    @staticmethod
    def west_attacks(rooks: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares west of certain squares.
        @param rooks: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares west of the specified squares filled
        """
        notH = np.uint64(0x7f7f7f7f7f7f7f7f)
        flood = rooks
        empty &= notH
        flood |= (rooks := (rooks >> np.uint64(1)) & empty)
        flood |= (rooks := (rooks >> np.uint64(1)) & empty)
        flood |= (rooks := (rooks >> np.uint64(1)) & empty)
        flood |= (rooks := (rooks >> np.uint64(1)) & empty)
        flood |= (rooks := (rooks >> np.uint64(1)) & empty)
        flood |= (rooks >> np.uint64(1)) & empty
        return (flood >> np.uint64(1)) & notH

    @staticmethod
    def northeast_attacks(bishops: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares northeast of certain squares.
        @param bishops: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares northeast of the specified squares filled
        """
        notA = np.uint64(0xfefefefefefefefe)
        flood = bishops
        empty &= notA
        flood |= (bishops := (bishops << np.uint64(9)) & empty)
        flood |= (bishops := (bishops << np.uint64(9)) & empty)
        flood |= (bishops := (bishops << np.uint64(9)) & empty)
        flood |= (bishops := (bishops << np.uint64(9)) & empty)
        flood |= (bishops := (bishops << np.uint64(9)) & empty)
        flood |= (bishops << np.uint64(9)) & empty
        return (flood << np.uint64(9)) & notA

    @staticmethod
    def southeast_attacks(bishops: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares southeast of certain squares.
        @param bishops: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares southeast of the specified squares filled
        """
        notA = np.uint64(0xfefefefefefefefe)
        flood = bishops
        empty &= notA
        flood |= (bishops := (bishops >> np.uint64(7)) & empty)
        flood |= (bishops := (bishops >> np.uint64(7)) & empty)
        flood |= (bishops := (bishops >> np.uint64(7)) & empty)
        flood |= (bishops := (bishops >> np.uint64(7)) & empty)
        flood |= (bishops := (bishops >> np.uint64(7)) & empty)
        flood |= (bishops >> np.uint64(7)) & empty
        return (flood >> np.uint64(7)) & notA

    @staticmethod
    def southwest_attacks(bishops: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares southwest of certain squares.
        @param bishops: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares southwest of the specified squares filled
        """
        notH = np.uint64(0x7f7f7f7f7f7f7f7f)
        flood = bishops
        empty &= notH
        flood |= (bishops := (bishops >> np.uint64(9)) & empty)
        flood |= (bishops := (bishops >> np.uint64(9)) & empty)
        flood |= (bishops := (bishops >> np.uint64(9)) & empty)
        flood |= (bishops := (bishops >> np.uint64(9)) & empty)
        flood |= (bishops := (bishops >> np.uint64(9)) & empty)
        flood |= (bishops >> np.uint64(9)) & empty
        return (flood >> np.uint64(9)) & notH

    @staticmethod
    def northwest_attacks(bishops: np.uint64, empty: np.uint64) -> np.uint64:
        """
        Fill squares northwest of certain squares.
        @param bishops: bitboard representing the square
        @param empty: bitboard representing empty squares
        @return: bitboard with squares northwest of the specified squares filled
        """
        notH = np.uint64(0x7f7f7f7f7f7f7f7f)
        flood = bishops
        empty &= notH
        flood |= (bishops := (bishops << np.uint64(7)) & empty)
        flood |= (bishops := (bishops << np.uint64(7)) & empty)
        flood |= (bishops := (bishops << np.uint64(7)) & empty)
        flood |= (bishops := (bishops << np.uint64(7)) & empty)
        flood |= (bishops := (bishops << np.uint64(7)) & empty)
        flood |= (bishops << np.uint64(7)) & empty
        return (flood << np.uint64(7)) & notH

