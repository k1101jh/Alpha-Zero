from multiprocessing import Queue
from typing import List, Optional, Tuple
from agents.abstract_agent import AbstractAgent
from games.game_components import Move
from games.abstract_game_state import AbstractGameState
from utils import point_from_coords


class Human(AbstractAgent):
    def __init__(self):
        """[summary]
            Human Player.
        """
        super().__init__()
        self.input_queue = None

    def set_input_queue(self, input_queue: Queue) -> None:
        self.input_queue = input_queue

    def select_move(self, game_state: AbstractGameState) -> Tuple[Move, Optional[List]]:
        if self.input_queue is None:
            while True:
                try:
                    inp = input()
                    point = point_from_coords(inp.strip())
                    move = Move.play(point)
                    if game_state.check_valid_move(move):
                        break
                except ValueError:
                    continue
        else:
            while True:
                self.input_queue.queue.clear()
                move = self.input_queue.get()
                self.input_queue.task_done()
                if game_state.check_valid_move(move):
                    break

        return move, None
