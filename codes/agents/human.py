from codes.agents.abstract_agent import Agent
from codes.types import Move
from codes.utils import point_from_coords


class Human(Agent):
    def __init__(self):
        super().__init__()
        self.queue = None

    def set_input_queue(self, queue):
        self.queue = queue

    def select_move(self, game_state):
        if self.queue is None:
            while True:
                inp = input('-- ')
                point = point_from_coords(inp.strip())
                move = Move.play(point)
                if game_state.check_valid_move(move):
                    break
        else:
            while True:
                move = self.queue.get()
                self.queue.task_done()
                if game_state.check_vaild_move(move):
                    break

        return move
