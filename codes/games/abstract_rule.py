from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Tuple

from games.game_types import Player


class AbstractRule(metaclass=ABCMeta):
    def __init__(self, board_size: int):
        """[summary]
        Args:
            board_size (int): Size of board.
        """
        super().__init__()
        self.board_size = board_size
        self.list_dy = [-1, -1, -1, 0, 1, 1, 1, 0]
        self.list_dx = [-1, 0, 1, 1, 1, 0, -1, -1]

    def get_dx_dy(self, direction: int) -> Tuple[int, int]:
        return self.list_dx[direction], self.list_dy[direction]

    @abstractmethod
    def check_game_over(self, game_state) -> Tuple[Optional[Player], bool]:
        pass
