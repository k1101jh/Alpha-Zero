import queue

from codes.games.game import Game
from codes.utils import print_turn
from codes.utils import print_board
from codes.utils import print_move
from codes.utils import print_winner


class CUI:
    def __init__(self, game_state_constructor, rule_constructor, players):
        self.board_queue = queue.Queue()
        self.move_queue = queue.Queue()
        self.game = Game(game_state_constructor, rule_constructor, players, self.board_queue, self.move_queue)
        self.board_size = self.game.get_board_size()

    def run(self):
        game_state = self.game.get_game_state()
        print_board(game_state.board)
        self.game.start()

        while not game_state.game_over:
            print_turn(game_state)
            player, move = self.move_queue.get()
            board = self.board_queue.get()
            print_move(player, move)
            print_board(board)
            game_state = self.game.get_game_state()
            self.queue_task_done()

        winner = game_state.winner
        print_winner(winner)

    def queue_task_done(self):
        self.move_queue.task_done()
        self.board_queue.task_done()
