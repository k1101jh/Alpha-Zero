import sys
import queue
import time

from codes.games.game import Game
from codes.utils import print_turn
from codes.utils import print_board
from codes.utils import print_move
from codes.utils import print_visit_count
from codes.utils import print_winner


class CUI:
    def __init__(self, game_state_constructor, rule_constructor, players):
        self.board_queue = queue.Queue()
        self.move_queue = queue.Queue()
        self.visit_count_queue = queue.Queue()
        self.game = Game(game_state_constructor, rule_constructor, players, self.board_queue, self.move_queue, self.visit_count_queue)
        self.board_size = self.game.get_board_size()

    def run(self):
        game_state = self.game.get_game_state()
        self.game.start()

        while not game_state.game_over:
            print_turn(game_state)
            board = self.board_queue.get()
            player, move = self.move_queue.get()

            # print visit count
            visit_counts = self.visit_count_queue.get()
            if visit_counts is not None:
                print_visit_count(visit_counts)
            #

            if move is not None:
                print_move(player, move)
            print_board(board)
            game_state = self.game.get_game_state()
            self.queue_task_done()

        winner = game_state.winner
        print_winner(winner)

    def queue_task_done(self):
        self.move_queue.task_done()
        self.board_queue.task_done()
        self.visit_count_queue.task_done()
