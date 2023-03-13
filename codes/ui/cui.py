import queue
from typing import Tuple
from agents.abstract_agent import AbstractAgent

from games.game import Game
from utils import print_turn
from utils import print_board
from utils import print_move
from utils import print_visit_count
from utils import print_winner
from games.game_components import UIEvent
from configuration import Configuration


class CUI:
    def __init__(self, config: Configuration, players: Tuple[AbstractAgent, AbstractAgent]):
        """[summary]
            Play game on CUI.
        Args:
            game_type ([type]): [description]
            rule_type ([type]): [description]
            players ([type]): [description]
        """
        self.event_queue = queue.Queue()
        self.game = Game(config, players, self.event_queue)
        self.board_size = config.board_size

    def run(self) -> None:
        self.game.start()
        game_over = False

        while not game_over:
            event, val = self.event_queue.get()
            self.event_queue.task_done()
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

    def get_game(self) -> Game:
        return self.game
