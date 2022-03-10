from typing import Tuple, TypeVar
import numpy as np

from games.game_types import Player
from games.game_types import Point
from games.game_types import Move
from games.abstract_game_state import AbstractGameState


Self = TypeVar("Self", bound="ZeroEncoder")


class ZeroEncoder():
    def __init__(self, board_size: int):
        """[summary]
            Plane channels:
            0. black stones
            1. white stones
            2. fill 1 if player is black. fill 0 if player is white

        Args:
            board_size (int): [description]
        """
        self.name = "ZeroEncoder"
        self.board_size = board_size
        self.num_planes = 3

    @classmethod
    def name(cls) -> str:
        return cls.name

    def shape(self) -> Tuple[int, int, int]:
        return self.num_planes, self.board_size, self.board_size

    def encode(self, game_state: AbstractGameState) -> np.ndarray:
        encoded_board = np.zeros(self.shape(), dtype=float)
        player = game_state.player
        if player == Player.black:
            encoded_board[2] = 1.

        board_grid = game_state.board.get_grid()
        black_stones = np.where(board_grid == Player.black.value)
        white_stones = np.where(board_grid == Player.white.value)

        encoded_board[0][black_stones] = 1.0
        encoded_board[1][white_stones] = 1.0

        return encoded_board

    def encode_move(self, move: Move) -> int:
        """[summary]
            Encode Move object to move index.
            If board_size is N,
            0 ~ N-1 : point on board
            N       : pass turn
        Args:
            move (Move): [description]

        Returns:
            int: Move index.
        """
        if move.is_play:
            return (self.board_size * move.point.row + move.point.col)
        else:
            return self.board_size * self.board_size

    def decode_move_index(self, move_index: int) -> Move:
        if move_index is self.board_size * self.board_size:
            return Move.pass_turn()

        row = move_index // self.board_size
        col = move_index % self.board_size
        return Move.play(Point(row, col))

    def num_moves(self) -> int:
        return self.board_size * self.board_size

    @classmethod
    def new_encoder(board_size: int) -> Self:
        return ZeroEncoder(board_size)
