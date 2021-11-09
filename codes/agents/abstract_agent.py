from abc import ABCMeta
from abc import abstractmethod


class Agent(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def select_move(self, game_state):
        raise NotImplementedError()