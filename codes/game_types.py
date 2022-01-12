import enum
from collections import namedtuple

game_name_dict = {
    "TicTacToe": "tic_tac_toe",
    "Omok": "omok",
    "MiniOmok": "omok",
}

board_size_dict = {
    "TicTacToe": 3,
    "MiniOmok": 9,
    "Omok": 15,
}


class Player(enum.Enum):
    black = 1
    white = 2
    both = 3

    @property
    def other(self):
        assert self is not Player.both
        return Player.black if self == Player.white else Player.white

    @property
    def forbidden(self):
        return FORBIDDEN_POINT[self]


FORBIDDEN_POINT = {
    Player.black: 4,
    Player.white: 5,
    Player.both: 6,
}


class Point(namedtuple('Point', 'row col')):
    pass


class Move:
    def __init__(self, point=None, is_pass=False):
        """[summary]

        Args:
            point (Point, optional): move point. Defaults to None.
            is_pass (bool, optional): Passed turn with no move. Defaults to False.
        """
        assert (point is not None) ^ is_pass
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)


class UIEvent(enum.Enum):
    BOARD = 1
    VISIT_COUNTS = 2
    LAST_MOVE = 3
    GAME_OVER = 4
