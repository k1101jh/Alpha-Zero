import queue

from codes.games.game import Game
from codes.utils import print_turn
from codes.utils import print_board
from codes.utils import print_move
from codes.utils import print_visit_count
from codes.utils import print_winner
from codes.game_types import UIEvent


class CUI:
    def __init__(self, game_type, rule_type, players):
        """[summary]
            Play game on CUI.
        Args:
            game_type ([type]): [description]
            rule_type ([type]): [description]
            players ([type]): [description]
        """

        self.event_queue = queue.Queue()
        self.game = Game(game_type, rule_type, players, self.event_queue)
        self.board_size = self.game.get_board_size()

    def run(self):
        self.game.start()
        game_over = False

        while not game_over:
            event, val = self.event_queue.get()
            if event == UIEvent.BOARD:
                print_board(val)
            elif event == UIEvent.VISIT_COUNTS:
                print_visit_count(val)
            elif event == UIEvent.LAST_MOVE:
                print_move(val)
                print_turn(self.game.get_game_state())
            elif event == UIEvent.GAME_OVER:
                print_winner(val)
                game_over = True

            self.event_queue.task_done()

    def get_game(self):
        return self.game
