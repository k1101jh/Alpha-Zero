import numpy as np

from codes.types import Player
from codes.types import Point
from codes.types import Move
from codes.games.tic_tac_toe.game import Board
from codes.games.tic_tac_toe.game import GameState


class ZeroEncoder():
    def __init__(self, board_size):
        # 0. black stones
        # 1. white stones
        # 2. fill 1 if player is black. fill 0 if player is white
        self.board_size = board_size
        self.num_planes = 3

    def name(self):
        return "TicTacToeEncoder"

    def shape(self):
        return self.num_planes, self.board_size, self.board_size

    def encode(self, game_state):
        encoded_game_state = np.zeros(self.shape(), dtype=float)
        player = game_state.player
        if player == Player.black:
            encoded_game_state[2] = 1.

        board_grid = game_state.board.get_grid()
        black_stones = np.where(board_grid == 1)
        white_stones = np.where(board_grid == 2)

        encoded_game_state[0][black_stones] = 1.0
        encoded_game_state[1][white_stones] = 1.0

        return encoded_game_state

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
        if move_index is self.board_size + self.board_size:
            return Move.pass_turn()

        row = move_index // self.board_size
        col = move_index % self.board_size
        return Move.play(Point(row, col))

    @classmethod
    def new_encoder(board_size):
        return ZeroEncoder(board_size)


if __name__ == "__main__":
    board = Board()
    board.grid = np.array([[1, 2, 0],
                           [2, 1, 0],
                           [0, 1, 2]], dtype=np.int8)
    game_state = GameState(board, Player.white)
    encoder = ZeroEncoder(3)

    encoded_game_state = encoder.encode(game_state)

    for k in range(encoder.num_planes):
        for i in range(3):
            print(encoded_game_state[k][i])
        print('')
