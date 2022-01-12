import numpy as np

from codes.game_types import Player
from codes import utils


class Board:
    def __init__(self, board_size):
        """[summary]
            Game board.
        Args:
            board_size (int): Size of board.
        """

        self.board_size = board_size
        self.grid = np.zeros((self.board_size, self.board_size), dtype=np.int)
        self.player_num_stones = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_empty_points = self.board_size * self.board_size

    def __deepcopy__(self, memodict={}):
        """[summary]

        Args:
            memodict (dict, optional): [description]. Defaults to {}.

        Returns:
            Board: Copied board.
        """

        copy_object = Board(self.board_size)
        copy_object.grid = np.copy(self.grid)
        copy_object.player_num_stones = utils.copy_dict(self.player_num_stones)
        copy_object.num_empty_points = self.num_empty_points

        return copy_object

    def place_stone(self, player, point):
        """
            put stone on grid.
        """

        self.grid[point] = player.value
        self.player_num_stones[player] += 1
        self.num_empty_points -= 1

    def get(self, point):
        """
            get stone on grid.
        """

        return self.grid[point]

    def get_grid(self):
        return self.grid

    def remain_point_nums(self):
        return self.num_empty_points
