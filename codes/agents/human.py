from codes.agents.abstract_agent import Agent
from codes.types import Move
from codes.utils import point_from_coords


class Human(Agent):
    def __init__(self):
        super().__init__()
        self.input_queue = None

    def set_input_queue(self, input_queue):
        self.input_queue = input_queue

    def select_move(self, game_state):
        if self.input_queue is None:
            while True:
                try:
                    inp = input('-- ')
                    point = point_from_coords(inp.strip())
                    move = Move.play(point)
                    if game_state.check_valid_move(move):
                        break
                except ValueError:
                    continue
        else:
            while True:
                self.input_queue.empty()
                move = self.input_queue.get()
                self.input_queue.task_done()
                if game_state.check_valid_move(move):
                    break

        return move, None
