from abc import ABCMeta
from abc import abstractmethod


class AbstractRule(metaclass=ABCMeta):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size

    def is_on_grid(self, point):
        return 0 <= point.row < self.board_size and 0 <= point.col < self.board_size

    @abstractmethod
    def check_game_over(self, game_state) -> bool:
        pass
