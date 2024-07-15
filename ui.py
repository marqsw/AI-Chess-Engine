import json
import math
import os
import sys
import webbrowser
from abc import ABC, abstractmethod
from enum import Enum
from tkinter import filedialog as fd
from typing import Callable

import numpy as np
import pygame

import mcts
import model
from JSONFileManager import JSONFileManager
from chess import ChessBoard, Piece
from default_configs import ui_config

pygame.init()

config = JSONFileManager('.config/ui_config.json', ui_config)

config.save()

while True:
    try:
        # ui and images
        BORDER_RADIUS = config.data['border radius']
        PADDING = config.data['padding']
        PIECE_PATH = config.data['piece path']

        # text
        FONT_PATH = config.data['font path']
        FONT_SIZE = config.data['font size']
        FONT = pygame.font.Font(FONT_PATH, FONT_SIZE)

        # colours
        BASE_COLOUR = pygame.Color(config.data['base colour'])
        TINT_COLOUR = pygame.Color(config.data['tint colour'])
        ALPHA = int(config.data['alpha'] * 255)

        BOARD_TINT_STRENGTH = config.data['board tint strength']
        WHITE = pygame.Color(255, 255, 255)
        BLACK = pygame.Color(0, 0, 0)

        # ui
        DRAG_ACTIVATION = config.data[
            'drag activation']  # distance the cursor have travelled to be categorised as dragging
        PROMOTION_UI_SIZE = config.data['promotion ui size']
        break
    except:
        config.restore_default()
        config.save()


class Weight(Enum):
    """
    UI elements visual weights, used for determining tint and opacity of the background of elements.
    """
    INVISIBLE = 0
    BACKGROUND = 0.2
    UNIMPORTANT = 0.3
    SECONDARY = 0.4
    PRIMARY = 0.5


class UIElement(ABC):
    """
    Base class of all the UI elements. Specifying the universal functions required.
    """
    instances: list['UIElement'] = []
    mouse_click_pos: tuple[int, int] = None

    def __init__(self, container: 'UIContainer', column_num: int = 1, row_num: int = 1, background: bool = True):
        """
        Initialise a UI element with attributes
        @param container: an outer container of the element
        @param column_num: relative width of the element
        @param row_num: relative height of the element
        @param background: whether a coloured background should be drawn
        """
        UIElement.instances.append(self)
        self.container = container

        self.colour = BASE_COLOUR.lerp(TINT_COLOUR, self.WEIGHT.value)
        self.colour.a = ALPHA

        self.column_num = column_num
        self.row_num = row_num

        self.background = background

        self.on_hold = False

    # -- properties --
    @property
    @abstractmethod
    def WEIGHT(self) -> Weight:
        """
        Retrieve the visual weight of the element.
        @return: Weight Enum class representing the visual weight of the element
        """
        pass

    @property
    def border_radius(self) -> int:
        """
        Retrieve the border radius of the element.
        @return: Integer representing the border radius of the element
        """
        return self.container.border_radius

    @property
    def padding(self) -> int:
        """
        Retrieve the padding of the element.
        @return: Integer representing the padding of the element
        """
        return self.container.padding

    @property
    def rect(self) -> pygame.Rect:
        """
        Retrieve the rect area given to the element to be drawn on.
        @return: Rect object representing the available area to be drawn on
        """
        return self.container.get_rect(self)

    @property
    def padded_rect(self) -> pygame.Rect:
        """
        Retrieve the rect area after padding.
        @return: Rect object representing the padded area
        """
        return self.rect.inflate(- self.padding, - self.padding)

    @property
    def on_hover(self) -> bool:
        """
        Retrieve cursor collision state.
        @return: boolean representing whether cursor is hovering on the element
        """
        return self.padded_rect.collidepoint(pygame.mouse.get_pos())

    @property
    def on_drag(self) -> bool:
        """
        Retrieve dragging state.
        @return: boolean representing whether the element is being dragged
        """
        return (
                self.on_hover
                and (math.hypot(
            *tuple(map(lambda x, y: x - y, UIElement.mouse_click_pos, pygame.mouse.get_pos()))) > DRAG_ACTIVATION
                     if UIElement.mouse_click_pos is not None else False
                     )
        )

    @property
    def on_click(self) -> bool:
        """
        Retrieve the click state of the element.
        @return: boolean representing whether the element is currently being clicked
        """
        return (
                self.on_hover
                and pygame.mouse.get_pressed()[0]
                and not self.on_drag
        )

    # -- methods --
    def render(self, canvas: pygame.Surface) -> None:
        """
        Render the element onto the given Surface after updating cursor states.
        @param canvas: Surface for the element to be drawn on
        """
        if issubclass(type(self), UIBaseContainer):  # base container's responsibility but a core function
            if pygame.mouse.get_pressed()[0]:
                if UIElement.mouse_click_pos is None:
                    UIElement.mouse_click_pos = pygame.mouse.get_pos()

                    UIElement.mouse_drag = math.hypot(
                        *map(lambda x, y: x - y, UIElement.mouse_click_pos, pygame.mouse.get_pos())
                    ) > DRAG_ACTIVATION
            else:
                UIElement.mouse_click_pos = None
                UIElement.mouse_drag = False

        # draw background if on hover
        if self.on_hover and self.WEIGHT is not Weight.BACKGROUND and self.background:
            canvas.blit(self.gen_squircle(self.padded_rect.size), self.padded_rect)

        # mouse button clicked
        if self.on_click and not self.on_hold:
            self.on_click_action(canvas)
            self.on_hold = True

        # mouse button released
        elif self.on_hold and self.on_hover and not pygame.mouse.get_pressed()[0]:
            self.on_release_action(canvas)
            self.on_hold = False

        # dragging
        elif self.on_drag:
            self.on_drag_action(canvas)


        elif self.on_hold:
            self.on_hold_action(canvas)

        # hover
        elif self.on_hover:
            self.on_hover_action(canvas)

        # cursor escaped (PRISON BREAK!!)
        else:
            self.on_hold = False

        self.draw(canvas)

    def draw(self, canvas: pygame.Surface) -> None:
        """
        Draw the element on a Surface without background and updating cursor states. Subclasses should define their appearance here.
        @param canvas: Canvas to be drawn on
        """
        pass

    def on_click_action(self, canvas: pygame.Surface) -> None:
        """
        Function called when being clicked. Subclasses should define their click action here.
        @param canvas: canvas to be drawn on
        """
        pass

    def on_hold_action(self, canvas: pygame.Surface) -> None:
        """
        Function called when being hold onto. Subclasses should define their hold action here.
        @param canvas: canvas to be drawn on
        """
        pass

    def on_release_action(self, canvas: pygame.Surface) -> None:
        """
        Function called when being clicked and released. Subclasses should define their press action here.
        @param canvas: canvas to be drawn on
        """
        pass

    def on_drag_action(self, canvas: pygame.Surface) -> None:
        """
        Function called when being dragged. Subclasses should define their drag action here.
        @param canvas: canvas to be drawn on
        """
        pass

    def on_hover_action(self, canvas: pygame.Surface) -> None:
        """
        Function called when on hover. Subclasses should define their on hover action here.
        @param canvas: Canvas to be drawn on
        """
        pass

    def gen_squircle(self, size: tuple[int, int]) -> pygame.Surface:
        """
        Generate a rounded rectangle background
        @param size: size of the rectangle
        @return: Surface containing the rounded rectangle
        """
        canvas = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(canvas, self.colour, (0, 0, *size), border_radius=self.border_radius)
        return canvas


class UIContainer(UIElement):
    """
    Container of UI Elements, all UI elements should be contained within a container. The container governs the
    positions and sizes of the elements. This allows dynamic widget size and placement with varying window size.
    There are two types of container: UIContainer and UIBaseContainer. UIContainer can be placed inside UIContainer
    or UIBaseContainer. While UIBaseContainer can only be placed on top of a pygame Surface, intended to interface
    between Pygame window and this UI library.
    """
    WEIGHT = Weight.UNIMPORTANT

    def __init__(self, container: 'UIContainer', column_num: int = 1, row_num: int = 1, background: bool = True):
        super().__init__(container, column_num, row_num, background)
        self.elements: list[list[UIElement]] = []
        self.element_rects: list[list[pygame.Rect]] = []

    @property
    def on_click(self) -> False:
        """
        Container NEVER steals the spotlight of other elements.
        @return: The UIContainer is NOT on a click.
        """
        return False

    def update_elements(self) -> None:
        """
        Update all the elements' positions and sizes.
        """
        topleft = list(self.padded_rect.topleft)
        self.element_rects = []

        try:
            cell_height = self.padded_rect.height / sum(
                [
                    max([element.row_num for element in row])
                    for row in self.elements
                ]
            )

            for row in self.elements:
                topleft[0] = self.padded_rect.topleft[0]
                rects = []

                element_height = cell_height * max([element.row_num for element in row])

                cell_width = self.padded_rect.width / sum([element.column_num for element in row])
                for element in row:
                    element_width = cell_width * element.column_num

                    rects.append(
                        pygame.Rect(*topleft, element_width, element_height)
                    )

                    topleft[0] += element_width

                self.element_rects.append(rects)
                topleft[1] += element_height

        except ZeroDivisionError:
            return

    def get_rect(self, element: UIElement) -> pygame.Rect:
        """
        Retrieve a specified element rect.
        @param element: target element
        @return: rect of the element
        """
        for row_index, row in enumerate(self.elements):
            if element in row:
                return self.element_rects[row_index][row.index(element)]

    def render(self, canvas: pygame.Surface) -> None:
        super().render(canvas)
        self.update_elements()

        if self.background:
            canvas.blit(self.gen_squircle(self.padded_rect.size), self.padded_rect.topleft)

        for row in self.elements:
            for element in row:
                element.render(canvas)


class UIBaseContainer(UIContainer):
    """
    Specialised UIContainer to interface between other UIElements and Pygame Surface. This should contain all other
    UIElements.
    """
    WEIGHT = Weight.BACKGROUND

    def __init__(self, container: pygame.Surface, rect: pygame.Rect):
        super().__init__(container)
        self._rect = rect

    @property
    def rect(self) -> pygame.Rect:
        return self._rect

    @rect.setter
    def rect(self, value):
        self._rect = value

    @property
    def border_radius(self) -> int:
        return int(min(self.rect.size) * BORDER_RADIUS)

    @property
    def padding(self) -> int:
        return min(self.rect.size) * PADDING

    def render(self, canvas: pygame.Surface = None):
        super().render(self.container)


class Button(UIElement):
    """
    A clickable button with text or image.
    """
    WEIGHT = Weight.SECONDARY

    def __init__(
            self, container: 'UIContainer',
            func: Callable = None,
            text: str = None,
            image: pygame.Surface = None,
            column_num: int = 1,
            row_num: int = 1,
            background: bool = True):
        """
        Inistalise the button with the required attributes.
        @param container: UIContainer of the Button
        @param func: function to be run after clicking
        @param text: text to be displayed on the button
        @param image: image to be displayed on the button
        @param column_num: relative width of the button
        @param row_num: relative height of the button
        @param background: whether a coloured background should be drawn
        """
        super().__init__(container, column_num, row_num, background)
        self.func = func if func is not None else (lambda: None)

        self.text = text
        self.image = image

    def on_release_action(self, canvas: pygame.Surface) -> None:
        self.func()

    def on_hold_action(self, canvas: pygame.Surface) -> None:
        canvas.blit(self.gen_squircle(self.padded_rect.size), self.padded_rect.topleft)

    def draw(self, canvas: pygame.Surface) -> None:
        canvas.blit(self.gen_squircle(self.padded_rect.size), self.padded_rect.topleft)

        if self.image is not None:
            dimension = min(self.padded_rect.inflate(- self.padding, - self.padding).size)
            canvas.blit(
                pygame.transform.smoothscale(self.image, (dimension, dimension)),
                pygame.Rect(*self.padded_rect.center, 0, 0).inflate(dimension, dimension).topleft
            )
        elif self.text is not None:
            text = FONT.render(self.text, True, TINT_COLOUR)
            canvas.blit(
                text,
                pygame.Rect(*self.padded_rect.center, 0, 0).inflate(text.get_size())
            )


class ChessboardUI(UIElement):
    WEIGHT = Weight.PRIMARY

    def __init__(self, container: 'UIContainer', column_num: int = 1, row_num: int = 1,
                 background: bool = False):
        """
        Initialise a chessboard with the required attributes.
        @param container: UIContainer of the chessboard
        @param board: bitboard to be used
        @param column_num: relative width of the chessboard
        @param row_num: relatvie height of the chessboard
        @param background: whether a colooured background should be drawn
        """
        super().__init__(container, column_num, row_num, background)
        self.promote_square = None
        self.active_promotion_ui: PromotionUI = None
        self.overlay = None
        self.mouse_board = None
        self.mouse_pos = None
        self.square_size = None
        self.mouse_index = None
        self.board_rect = None
        self.board = ChessBoard()
        self.flip_board = False

        self.selected_piece: np.uint64 = np.uint64(0)
        self.selected_piece_image: pygame.Surface = None
        self.selected_piece_colour: Piece = None
        self.selected_piece_type: Piece = None
        self.selected_piece_moves: list[np.uint64] = []
        self.piece_images = (
            (
                None,
                None,
                pygame.image.load(f'{PIECE_PATH}/wP.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wN.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wB.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wR.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wQ.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wK.png').convert_alpha()
            ),
            (
                None,
                None,
                pygame.image.load(f'{PIECE_PATH}/bP.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bN.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bB.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bR.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bQ.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bK.png').convert_alpha()
            )
        )

    @property
    def on_hover(self) -> bool:
        return self.board_rect.collidepoint(pygame.mouse.get_pos())

    def render(self, canvas: pygame.Surface) -> None:
        self.square_size = int(min(self.padded_rect.size) / 8)
        self.board_rect = pygame.Rect(*self.rect.center, 0, 0).inflate(self.square_size * 8, self.square_size * 8)
        self.mouse_pos = tuple(map(lambda x, y: x - y, pygame.mouse.get_pos(), self.board_rect))
        self.mouse_index = self.board.coord_to_index(
            (int(self.mouse_pos[0] // self.square_size), 7 - int(self.mouse_pos[1] // self.square_size))
        )
        self.mouse_board = self.board.index_to_board(self.mouse_index)

        self.overlay = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)

        super().render(canvas)

    def on_click_action(self, canvas: pygame.Surface, freeplay=True) -> None:
        """
        Function called when being clicked. Subclasses should define their click action here.
        @param canvas: canvas to be drawn on
        @param freeplay: whether the board is on freeplay mode
        @return:
        """
        if (
                self.active_promotion_ui is None
                and
                (bool(np.bitwise_or.reduce(
                    self.board.pieces[self.board.colour_to_move.value]) & self.mouse_board) or freeplay)
        ):

            if (not self.mouse_index in self.selected_piece_moves and not freeplay) or not self.selected_piece:
                self.selected_piece = np.bitwise_or.reduce(self.board.occupancy) & self.mouse_board

                for i, colour in enumerate(self.board.pieces):
                    for j, piece_type in enumerate(colour):
                        image = self.piece_images[i][j]

                        if image is None:
                            continue

                        if self.selected_piece & piece_type:
                            self.selected_piece_image = image
                            self.selected_piece_colour = Piece(i)
                            self.selected_piece_type = Piece(j)

                            self.selected_piece_moves = self.board.gen_legal(
                                self.selected_piece,
                                self.selected_piece_colour,
                                self.selected_piece_type,
                            )

                            if len(self.selected_piece_moves):
                                self.selected_piece_moves = self.board.get_indices(
                                    np.bitwise_or.reduce(self.selected_piece_moves)
                                )

    def on_drag_action(self, canvas: pygame.Surface) -> None:
        """
        Function called when being dragged. Subclasses should define their drag action here.
        @param canvas: canvas to be drawn on
        """
        if self.selected_piece and self.active_promotion_ui is None:
            self.overlay.blit(pygame.transform.smoothscale(
                self.piece_images[self.selected_piece_colour.value][self.selected_piece_type.value],
                (self.square_size, self.square_size)
            ),
                pygame.Rect(*self.mouse_pos, 0, 0).inflate(self.square_size, self.square_size)
            )

    def on_release_action(self, canvas: pygame.Surface) -> None:
        """
        Function called when being clicked and released. Subclasses should define their press action here.
        @param canvas: canvas to be drawn on
        """
        if (
                self.selected_piece
                # and self.board.get_indices(self.mouse_board)[0] in self.selected_piece_moves
                and self.active_promotion_ui is None
                and not (self.mouse_board & self.selected_piece)
        ):
            self.make_move()

    def make_move(self, freeplay=False):
        """
        To make a move on the internal board when the user make a move on the GUI.
        @param freeplay: whether if the board is on freeplay mode
        @return:
        """

        if (
                freeplay or (
                self.selected_piece
                and self.board.get_indices(self.mouse_board)[0] in self.selected_piece_moves
                and self.selected_piece & np.bitwise_or.reduce(self.board.pieces[self.board.colour_to_move.value])
        )):
            # promotion
            if self.selected_piece_type is Piece.PAWNS:
                if self.selected_piece_colour is Piece.WHITE and self.mouse_board & self.board.EIGHTH_RANK:
                    self.active_promotion_ui = PromotionUI(
                        self,
                        self.board.colour_to_move,
                        pygame.Rect(*self.board_rect.center, 0, 0).inflate(self.square_size * PROMOTION_UI_SIZE,
                                                                           self.square_size * PROMOTION_UI_SIZE)
                    )
                    self.promote_square = self.mouse_board
                    return

                elif self.selected_piece_colour is Piece.BLACK and self.mouse_board & self.board.FIRST_RANK:
                    self.active_promotion_ui = PromotionUI(
                        self,
                        self.board.colour_to_move,
                        pygame.Rect(*self.board_rect.center, 0, 0).inflate(self.square_size * PROMOTION_UI_SIZE,
                                                                           self.square_size * PROMOTION_UI_SIZE)
                    )
                    self.promote_square = self.mouse_board
                    return

            self.board.make_move(
                self.selected_piece,
                self.mouse_board,
                self.selected_piece_colour,
                self.selected_piece_type,
                force=freeplay
            )

            if self.board.get_winner() is None:
                self.ai_action()
                if self.board.get_winner() is not None:
                    self.game_ends_action()
            else:
                self.game_ends_action()

        self.selected_piece = np.uint64(0)

    def draw(self, canvas: pygame.Surface, freeplay=False) -> None:
        """
        Draw the element on a Surface without background and updating cursor states. Subclasses should define their appearance here.
        @param canvas: Canvas to be drawn on
        @param freeplay: whether the board is on freeplay mode
        @return:
        """
        light_colour = WHITE.lerp(TINT_COLOUR, BOARD_TINT_STRENGTH)
        dark_colour = BLACK.lerp(TINT_COLOUR, BOARD_TINT_STRENGTH)

        # board
        coord_y = self.board_rect.y
        for file in range(8):
            coord_x = self.board_rect.x
            for rank in range(8):
                square_colour = light_colour if (rank + file) % 2 == 0 else dark_colour
                pygame.draw.rect(canvas, square_colour, (coord_x, coord_y, self.square_size, self.square_size))
                coord_x += self.square_size
            coord_y += self.square_size

        # piece
        for i, colour in enumerate(self.board.pieces):
            for j, piece_type in enumerate(colour):
                image = self.piece_images[i][j]

                if image is None:
                    continue

                self.blit_piece(
                    canvas,
                    image,
                    self.board_rect,
                    *self.board.get_indices(piece_type & ~ self.selected_piece)
                )

        if self.selected_piece and (self.selected_piece_colour == self.board.colour_to_move or freeplay):
            transparent_piece_image = self.selected_piece_image.copy()
            transparent_piece_image.set_alpha(ALPHA)

            self.blit_piece(
                canvas,
                transparent_piece_image,
                self.board_rect,
                *self.selected_piece_moves
            )

        # blit piece in original position if not on drag
        if not self.on_drag and self.selected_piece:
            self.blit_piece(
                canvas,
                self.selected_piece_image,
                self.board_rect,
                *self.board.get_indices(self.selected_piece)
            ), self.board_rect.topleft

        # overlays
        canvas.blit(self.overlay, self.board_rect)

        if self.active_promotion_ui is not None:
            self.active_promotion_ui.render(canvas)

            if type(self) is PassandPlayChessboardUI:
                self.board.state = self.board.view_from(-1)

            if self.active_promotion_ui.promotion_piece is not None:
                self.board.make_move(
                    self.selected_piece,
                    self.promote_square,
                    self.selected_piece_colour,
                    self.selected_piece_type,
                    self.active_promotion_ui.promotion_piece
                )

                self.active_promotion_ui = None
                self.selected_piece = np.uint64(0)

                self.ai_action()

            if type(self) is PassandPlayChessboardUI:
                self.board.state = self.board.view_from(-1)

    def blit_piece(self,
                   canvas: pygame.Surface,
                   image: pygame.Surface,
                   rect: pygame.Rect,
                   *indices: np.uint64) -> pygame.Surface:
        """
        Draw the pieces.
        @param canvas: canvas to be drawn on
        @param image: image of the piece
        @param rect: rect of chessboard
        @param indices: square indices of the pieces
        @return:
        """

        square_size = tuple(map(lambda x: int(x / 8), rect.size))
        topleft = rect.topleft
        for index in indices:
            rank, file = self.board.index_to_coord(index)

            canvas.blit(
                pygame.transform.smoothscale(image, square_size),
                (topleft[0] + rank * square_size[0], topleft[1] + (7 - file) * square_size[1])
            )

    def undo_board(self) -> None:
        """
        Undo the latest move.
        @return:
        """
        if len(self.board.history):
            self.board.state = self.board.history.pop()
            self.board.update_board()
            self.selected_piece = np.uint64(0)

    def reset_board(self) -> None:
        """
        Set the board to original state.
        @return:
        """
        self.__init__(self.container, self.column_num, self.row_num, self.background)

    def ai_action(self) -> None:
        """
        AI action to be taken after a move.
        @return:
        """
        pass

    def game_ends_action(self) -> None:
        """
        Actions to be taken after the game ends.
        @return:
        """
        pass


class PlayWhiteChessboardUI(ChessboardUI):
    """
    Playing from white's perspective.
    """

    def __init__(self, container: 'UIContainer', column_num: int = 1, row_num: int = 1,
                 background: bool = False):
        super().__init__(container, column_num, row_num, background)
        self.model = model.ChessModel()
        self.temp_board = ChessBoard()

    def ai_action(self) -> None:
        self.board.state = self.temp_board.state = self.board.view_from(-1)
        search_tree = mcts.MCTS(self.temp_board, self.model, {'num_simulations': 20})
        root = search_tree.run(self.model, self.board.state, to_play=1)
        self.board.make_move(*root.select_action(0))
        self.board.state = self.board.view_from(-1)

    def undo_board(self) -> None:
        super().undo_board()
        super().undo_board()


class PlayBlackChessboardUI(PlayWhiteChessboardUI):
    """
    PLay from black's perspective.
    """

    def __init__(self, container: 'UIContainer', column_num: int = 1, row_num: int = 1,
                 background: bool = False):
        super().__init__(container, column_num, row_num, background)
        self.piece_images = tuple(reversed(self.piece_images))
        self.board.state = self.board.view_from(-1)
        self.board.colour_to_move = Piece.WHITE
        self.ai_action()


class SelfPlayChessboardUI(PlayWhiteChessboardUI):
    """
    Board for self-playing. (without training)
    """

    def on_click_action(self, canvas: pygame.Surface, freeplay=True) -> None:
        pass

    def make_move(self, freeplay=False) -> None:
        self.temp_board.state = self.board.state
        search_tree = mcts.MCTS(self.temp_board, self.model, {'num_simulations': 20})
        root = search_tree.run(self.model, self.board.state, to_play=1)
        self.board.make_move(*root.select_action(0))
        self.ai_action()


class PassandPlayChessboardUI(ChessboardUI):
    """
    Board for pass and play mode.
    """

    def __init__(self, container: 'UIContainer', column_num: int = 1, row_num: int = 1,
                 background: bool = False):
        super().__init__(container, column_num, row_num, background)

    def make_move(self) -> None:
        moved = (self.selected_piece
                 and self.board.get_indices(self.mouse_board)[0] in self.selected_piece_moves
                 and self.selected_piece & np.bitwise_or.reduce(self.board.pieces[self.board.colour_to_move.value])
                 )
        super().make_move()
        if moved:
            self.piece_images = tuple(reversed(self.piece_images))
            self.board.state = self.board.view_from(-1)
            self.board.colour_to_move = Piece.WHITE
            self.ai_action()

    def undo_board(self) -> None:
        if len(self.board.history):
            self.piece_images = tuple(reversed(self.piece_images))

        super().undo_board()


class FreePlayChessboardUI(ChessboardUI):
    """
    Board for freeplay mode, no rules apply here.
    """

    def __init__(self, container: 'UIContainer', column_num: int = 1, row_num: int = 1,
                 background: bool = False):
        super().__init__(container, column_num, row_num, background)

    def make_move(self, force=True):
        super().make_move(force)

    def on_click_action(self, canvas: pygame.Surface, freeplay=True) -> None:
        super().on_click_action(True)

    def draw(self, canvas: pygame.Surface, freeplay=True) -> None:
        super().draw(canvas, freeplay)


class PromotionUI(UIContainer):
    """
    User interface for promotion.
    """
    WEIGHT = Weight.SECONDARY

    def __init__(self, container: ChessboardUI, piece_colour: Piece, board_rect: pygame.Rect, column_num: int = 1,
                 row_num: int = 1,
                 background: bool = True):
        super().__init__(container, column_num, row_num, background)

        self.piece_colour = piece_colour
        self._rect = board_rect
        self.promotion_piece = None

        self.piece_images = (
            (
                pygame.image.load(f'{PIECE_PATH}/wN.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wB.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wR.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/wQ.png').convert_alpha(),
            ),
            (
                pygame.image.load(f'{PIECE_PATH}/bN.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bB.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bR.png').convert_alpha(),
                pygame.image.load(f'{PIECE_PATH}/bQ.png').convert_alpha(),
            )
        )

        self.elements = [
            [
                Button(self, self.to_queen, image=self.piece_images[piece_colour.value][3]),
                Button(self, self.to_bishop, image=self.piece_images[piece_colour.value][1]),
            ],
            [
                Button(self, self.to_knight, image=self.piece_images[piece_colour.value][0]),
                Button(self, self.to_rook, image=self.piece_images[piece_colour.value][2])
            ]
        ]

    def to_queen(self) -> None:
        self.promotion_piece = Piece.QUEENS

    def to_bishop(self) -> None:
        self.promotion_piece = Piece.BISHOPS

    def to_knight(self) -> None:
        self.promotion_piece = Piece.KNIGHTS

    def to_rook(self) -> None:
        self.promotion_piece = Piece.ROOKS

    @property
    def rect(self) -> pygame.Rect:
        return self._rect


class Game(UIBaseContainer):
    def __init__(self, container: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Game playing UI.
        @param container: Pygame window containing the game UI
        @param rect: rect to be rendered on
        """
        super().__init__(container, rect)
        self.main_menu()
        self.piece_images = (
            pygame.image.load(f'{PIECE_PATH}/wP.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/wN.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/wB.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/wR.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/wQ.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/wK.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/bP.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/bN.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/bB.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/bR.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/bQ.png').convert_alpha(),
            pygame.image.load(f'{PIECE_PATH}/bK.png').convert_alpha()
        )

    def render(self, canvas: pygame.Surface = None) -> None:
        try:
            super().render(canvas)
        except AttributeError:  # elements won't be not found when any element renewed interface
            super().render(canvas)

    def main_menu(self) -> None:
        """
        Menu screen of the chess game.
        @return:
        """

        # -- GAME OPTIONS --
        game_sector = UIContainer(self, row_num=3)

        ai_sector = UIContainer(game_sector)
        play_white_button = Button(
            container=ai_sector,
            text='Play White',
            func=lambda: self.main_game_screen(PlayWhiteChessboardUI),
        )
        play_black_button = Button(
            container=ai_sector,
            text='Play Black',
            func=lambda: self.main_game_screen(PlayBlackChessboardUI)
        )
        self_play_button = Button(
            container=ai_sector,
            text='AI vs AI',
            func=self.self_play_screen
        )
        ai_sector.elements = [
            [play_white_button, play_black_button],
            [self_play_button]
        ]

        other_options = UIContainer(game_sector)
        pass_and_play_button = Button(
            container=other_options,
            text='Pass & Play',
            func=lambda: self.main_game_screen(PassandPlayChessboardUI)
        )
        free_play_button = Button(
            container=other_options,
            text='Freeplay',
            func=lambda: self.main_game_screen(FreePlayChessboardUI)
        )
        replay_button = Button(
            container=other_options,
            text='Replay Game',
        )

        def replay_game() -> None:
            file_path = fd.askopenfilename()

            # check file extension
            if os.path.splitext(file_path)[-1] != '.chess':
                return
            else:
                with open(file_path) as f:
                    data_dict: dict = json.load(f)

            history = data_dict['history']

            for index, step in enumerate(history):
                history[index] = (
                    np.array(step[0], dtype=np.uint64),
                    np.array(step[1], dtype=np.uint64),
                    np.array(step[2], dtype=np.uint64),
                    np.array(step[3], dtype=np.uint64),
                    Piece(step[4]),
                    step[5]
                )

            self.replay_game_screen(getattr(sys.modules[__name__], data_dict['mode']), history)

        replay_button.func = replay_game

        other_options.elements = [
            [replay_button],
            [pass_and_play_button, free_play_button],
        ]

        game_sector.elements = [
            [ai_sector, other_options]
        ]

        # -- CONFIG BUTTON --
        config_button = Button(
            container=self,
            text='Open Config File',
            func=lambda: webbrowser.open(os.path.abspath(config.path))
        )

        # -- EXIT BUTTON --
        exit_button = Button(
            container=self,
            text='Exit Game',
            func=lambda: sys.exit()
        )

        self.elements = [
            [game_sector],
            [config_button, exit_button]
        ]

    def main_game_screen(self, chessboardUI: ChessboardUI) -> None:
        """
        Game playing UI
        @param chessboardUI: Chessboard mode to use
        @return:
        """
        board_card = UIContainer(self, row_num=4)
        board_ui = chessboardUI(board_card)
        board_card.elements = [[board_ui]]

        control_centre = UIContainer(self)
        undo_button = Button(
            container=control_centre,
            text='Undo',
            func=board_ui.undo_board
        )
        reset_button = Button(
            container=control_centre,
            text='Reset',
            func=board_ui.reset_board
        )
        control_centre.elements = [
            [undo_button, reset_button]
        ]

        exit_button = Button(
            container=self,
            text='Main Menu',
            func=lambda: self.main_menu()
        )

        self.elements = [
            [board_card],
            [control_centre],
            [exit_button]
        ]

        def game_ends_action() -> None:
            board_ui.on_click_action = lambda x: None
            save_button = Button(self, text="Save Game")

            def save_file() -> None:
                f = fd.asksaveasfile(mode='w', defaultextension=".chess")
                if f is None:
                    return

                save_list = []

                history = board_ui.board.history.copy()
                history.append(board_ui.board.state)
                history.append(board_ui.board.state) if type(board_ui) in (
                    PlayWhiteChessboardUI, PlayBlackChessboardUI) else None

                for state in history:
                    item = []

                    for i in state:
                        if isinstance(i, np.ndarray):
                            item.append(i.tolist())
                        elif isinstance(i, Piece):
                            item.append(i.value)
                        else:
                            item.append(i)
                    save_list.append(item)

                save_dict = {
                    "mode": type(board_ui).__name__,
                    "history": save_list
                }

                json.dump(save_dict, f)
                f.close()
                self.main_menu()

            save_button.func = save_file
            no_button = Button(self, text="Main Menu", func=self.main_menu)

            board_ui.container.container = self

            self.elements = [
                [board_ui.container],
                [save_button, no_button]
            ]

        chessboardUI.game_ends_action = lambda x: game_ends_action()

    def self_play_screen(self) -> None:
        """
        Special UI for self-playing mode.
        @return:
        """
        board_card = UIContainer(self, row_num=4)
        board_ui = SelfPlayChessboardUI(board_card)
        board_card.elements = [[board_ui]]

        control_centre = UIContainer(self)
        undo_button = Button(
            container=control_centre,
            text='Undo',
            func=board_ui.undo_board
        )

        make_move_button = Button(
            container=control_centre,
            text='Make Move',
            func=board_ui.make_move
        )

        reset_button = Button(
            container=control_centre,
            text='Reset',
            func=board_ui.reset_board
        )
        control_centre.elements = [
            [undo_button, make_move_button, reset_button]
        ]

        exit_button = Button(
            container=self,
            text='Main Menu',
            func=lambda: self.main_menu()
        )

        self.elements = [
            [board_card],
            [control_centre],
            [exit_button]
        ]

        def game_ends_action() -> None:
            save_button = Button(self, text="Save Game")

            def save_file() -> None:
                f = fd.asksaveasfile(mode='w', defaultextension=".chess")
                if f is None:
                    return

                save_list = []

                history = board_ui.board.history.copy()
                history.append(board_ui.board.state)
                history.append(board_ui.board.state) if type(board_ui) in (
                    PlayWhiteChessboardUI, PlayBlackChessboardUI) else None

                for state in history:
                    item = []

                    for i in state:
                        if isinstance(i, np.ndarray):
                            item.append(i.tolist())
                        elif isinstance(i, Piece):
                            item.append(i.value)
                        else:
                            item.append(i)
                    save_list.append(item)

                save_dict = {
                    "mode": type(board_ui).__name__,
                    "history": save_list
                }

                json.dump(save_dict, f)
                f.close()
                self.main_menu()

            save_button.func = save_file
            no_button = Button(self, text="Main Menu", func=self.main_menu)

            board_ui.container.container = self

            self.elements = [
                [board_ui.container],
                [save_button, no_button]
            ]

    def replay_game_screen(self, chessboardUI: ChessboardUI, history: list) -> None:
        """
        Special UI for replaying games
        @param chessboardUI: game mode
        @param history: game history
        @return:
        """
        board_card = UIContainer(self, row_num=4)
        board_ui = chessboardUI(board_card)
        board_card.elements = [[board_ui]]
        board_ui.on_click_action = lambda x: None

        board_ui.board.state = history[0 if chessboardUI.__name__ != PlayBlackChessboardUI.__name__ else 1]
        board_ui.board.history = history

        self.current_step = 0 if chessboardUI.__name__ != PlayBlackChessboardUI.__name__ else 1

        control_centre = UIContainer(self)

        def previous_func():
            if chessboardUI.__name__ is not FreePlayChessboardUI.__name__:
                if self.current_step > 1:
                    self.current_step -= 2
            else:
                if self.current_step > 0:
                    self.current_step -= 1

            board_ui.board.state = board_ui.board.history[self.current_step]

        previous = Button(
            container=control_centre,
            text='Previous',
            func=previous_func
        )

        def next_func():
            if chessboardUI.__name__ is not FreePlayChessboardUI.__name__:
                if self.current_step + 2 <= len(board_ui.board.history) - 1:
                    self.current_step += 2
            else:
                if self.current_step < len(board_ui.board.history) - 1:
                    self.current_step += 1

            board_ui.board.state = board_ui.board.history[self.current_step]

        next = Button(
            container=control_centre,
            text='Next',
            func=next_func
        )
        control_centre.elements = [
            [previous, next]
        ]

        exit_button = Button(
            container=self,
            text='Main Menu',
            func=lambda: self.main_menu()
        )

        self.elements = [
            [board_card],
            [control_centre],
            [exit_button]
        ]