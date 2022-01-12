from abc import ABCMeta
from abc import abstractmethod


class Agent(metaclass=ABCMeta):
    def __init__(self):
        """[summary]
        """

        pass

    def set_input_queue(self, input_queue):
        pass

    @abstractmethod
    def select_move(self, game_state):
        raise NotImplementedError()
