from typing import Iterable
import numpy as np
import copy

from games.game_types import Player, Point
from utils import copy_dict


class Board:
    def __init__(self, board_size: int):
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

    def __deepcopy__(self, memo):
        """[summary]
        Args:
            memodict (dict, optional): [description]. Defaults to {}.

        Returns:
            Board: Copied board.
        """
        copy_object = Board(self.board_size)
        copy_object.grid = np.copy(self.grid)
        copy_object.player_num_stones = copy_dict(self.player_num_stones)
        copy_object.num_empty_points = copy.deepcopy(self.num_empty_points)

        return copy_object

    def place_stone(self, player: Player, point: Point) -> None:
        """[summary]
            put stone on grid.
        Args:
            player (Player): [description]
            point (Point): [description]
        """
        self.grid[point] = player.value
        self.player_num_stones[player] += 1
        self.num_empty_points -= 1

    def get(self, point: Point) -> int:
        """[summary]
            get stone on grid.
        Args:
            point (Point): [description]

        Returns:
            [type]: [description]
        """
        return self.grid[point]
    
    def get_board_size(self) -> int:
        return self.board_size

    def get_grid(self) -> np.ndarray:
        return self.grid

    def get_num_empty_points(self) -> int:
        return self.num_empty_points
    
    def get_player_num_stones(self, player: Player) -> int:
        return self.player_num_stones[player]

    def set_stones(self, player: Player, points: Iterable[Point]) -> None:
        for point in points:
            self.grid[point] = player.value
