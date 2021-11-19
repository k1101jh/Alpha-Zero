import numpy as np

from codes.types import Player
from codes.types import Point
from codes.types import Move


class ZeroEncoder():
    def __init__(self, board_size):
        # 0. black stones
        # 1. white stones
        # 2. fill 1 if player is black. fill 0 if player is white
        self.board_size = board_size
        self.num_planes = 3

    def name(self):
        return "ZeroEncoder"

    def shape(self):
        return self.num_planes, self.board_size, self.board_size

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape(), dtype=float)
        player = game_state.player
        if player == Player.black:
            board_tensor[2] = 1.

        board_grid = game_state.board.get_grid()
        black_stones = np.where(board_grid == 1)
        white_stones = np.where(board_grid == 0)

        board_tensor[0][black_stones] = 1.0
        board_tensor[1][white_stones] = 1.0

        return board_tensor

    def encode_move(self, move):
        """
        return move idx on flatten board
        if board_size is N,
        0 ~ N-1 : point on board
        N       : pass turn
        """
        if move.is_play:
            return (self.board_size * move.point.row + move.point.col)
        else:
            return self.board_size * self.board_size

    def decode_move_index(self, move_index):
        if move_index is self.board_size * self.board_size:
            return Move.pass_turn()

        row = move_index // self.board_size
        col = move_index % self.board_size
        return Move.play(Point(row, col))

    def num_moves(self):
        return self.board_size * self.board_size

    @classmethod
    def new_encoder(board_size):
        return ZeroEncoder(board_size)
