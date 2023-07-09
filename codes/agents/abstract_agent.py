from abc import ABCMeta
from abc import abstractmethod
from multiprocessing import Queue
from typing import List, Optional, Tuple
from games.game_components import Move

from games.abstract_game_state import AbstractGameState


class AbstractAgent(metaclass=ABCMeta):
    def __init__(self):
        pass

    def set_input_queue(self, input_queue: Queue) -> None:
        pass

    @abstractmethod
    def select_move(self, game_state: AbstractGameState) -> Tuple[Move, Optional[List]]:
        raise NotImplementedError()
