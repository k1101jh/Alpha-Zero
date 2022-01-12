from abc import ABCMeta
from abc import abstractmethod


class AbstractRule(metaclass=ABCMeta):
    def __init__(self, board_size):
        """[summary]

        Args:
            board_size (int): Size of board.
        """

        super().__init__()
        self.board_size = board_size
        self.list_dx = [-1, 1, -1, 1, 0, 0, 1, -1]
        self.list_dy = [0, 0, -1, 1, -1, 1, -1, 1]

    def get_dx_dy(self, direction):
        return self.list_dx[direction], self.list_dy[direction]

    @abstractmethod
    def check_game_over(self, game_state) -> bool:
        pass
