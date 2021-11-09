import time
import queue

from codes.utils import print_board
from codes.utils import print_move
from codes.utils import print_winner


class CUI:
    def __init__(self, game, players):
        self.board_queue = queue.Queue()
        self.move_queue = queue.Queue()
        self.game = game(players, self.board_queue, self.move_queue)
        self.board_size = self.game.get_board_size()

    def run(self):
        game_state = self.game.get_game_state()
        print_board(game_state.board)
        self.game.start()

        while not self.game.get_game_state().game_over:
            player, move = self.move_queue.get()
            board = self.board_queue.get()
            self.queue_task_done()
            print_move(player, move)
            print_board(board)
            time.sleep(0.2)

        winner = self.game.get_game_state().winner
        print_winner(winner)

    def queue_task_done(self):
        self.move_queue.task_done()
        self.board_queue.task_done()
