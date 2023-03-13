from multiprocessing import Queue
import threading
from typing import Dict
# import yappi

from configuration import Configuration
from agents.abstract_agent import AbstractAgent
from games.game_components import Move, Player, UIEvent
from games.abstract_game_state import AbstractGameState

from configuration import get_game_state_constructor, get_rule_constructor


class Game(threading.Thread):
    def __init__(self, config: Configuration,
                 players: Dict[Player, AbstractAgent], event_queue: Queue = None):
        """[summary]
        Args:
            game_type ([type]): [description]
            rule_type ([type]): [description]
            players (dictionary): Dict with two players.
            event_queue ([type], optional): Event queue to communicate with UI. Defaults to None.
        """
        super().__init__()
        self.daemon = True

        self.board_size = config.board_size
        self.game_state_constructor = get_game_state_constructor(config.game_type)
        self.rule_constructor = get_rule_constructor(config.rule_type)
        
        self.rule = self.rule_constructor(self.board_size)
        self.players = players
        self.event_queue = event_queue
        
        # initialize game
        self.game_state: AbstractGameState = None
        self.init_game()

    def init_game(self) -> None:
        self.game_state = self.game_state_constructor.new_game(self.board_size, self.rule)

    def enqueue(self, event: UIEvent, val) -> None:
        if self.event_queue is not None:
            self.event_queue.put((event, val))
            self.event_queue.join()

    def get_game_state(self) -> AbstractGameState:
        return self.game_state

    def run(self) -> None:
        self.enqueue(UIEvent.BOARD, self.game_state.get_board())
        while not self.game_state.game_over:
            # yappi.set_clock_type("wall")
            # yappi.start()
            move, visit_counts = self.players[self.game_state.player].select_move(self.game_state)
            self.game_state = self.game_state.apply_move(move)

            self.enqueue(UIEvent.VISIT_COUNTS, visit_counts)
            self.enqueue(UIEvent.BOARD, self.game_state.board)
            self.enqueue(UIEvent.LAST_MOVE, [self.game_state.player.other, move])
            
            if self.game_state.check_game_over():
                break
            # yappi.stop()
            # yappi.get_func_stats().print_all()

        winner = self.game_state.winner
        self.enqueue(UIEvent.GAME_OVER, winner)

    def get_cur_player(self) -> Player:
        return self.players[self.game_state.player]

    def get_cur_player_move(self) -> Move:
        return self.players[self.game_state.player].select_move(self.game_state)
