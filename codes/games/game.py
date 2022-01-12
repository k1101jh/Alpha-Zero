import threading
from codes import utils
from codes import types
from codes.types import UIEvent


class Game(threading.Thread):
    def __init__(self, game_type, rule_type,
                 players, event_queue=None):
        super().__init__()
        self.daemon = True

        self.board_size = types.board_size_dict[game_type]
        self.game_state_constructor = utils.get_game_state_constructor(game_type)
        self.rule_constructor = utils.get_rule_constructor(game_type, rule_type)
        self.game_state = self.game_state_constructor.new_game(self.board_size, self.rule_constructor)
        self.players = players
        self.event_queue = event_queue

    def init_game(self):
        self.game_state = self.game_state_constructor.new_game(self.board_size, self.rule_constructor)

    def enqueue(self, event, val):
        self.event_queue.put((event, val))
        self.event_queue.join()

    def get_board_size(self):
        return self.game_state.board.board_size

    def get_game_state(self):
        return self.game_state

    def run(self):
        self.enqueue(UIEvent.BOARD, self.game_state.board)
        while not self.game_state.game_over:
            move, visit_counts = self.players[self.game_state.player].select_move(self.game_state)
            self.game_state = self.game_state.apply_move(move)

            if self.game_state.check_game_over():
                break

            self.enqueue(UIEvent.VISIT_COUNTS, visit_counts)
            self.enqueue(UIEvent.BOARD, self.game_state.board)
            self.enqueue(UIEvent.LAST_MOVE, [self.game_state.player.other, move])

        winner = self.game_state.get_winner()
        self.enqueue(UIEvent.VISIT_COUNTS, visit_counts)
        self.enqueue(UIEvent.BOARD, self.game_state.board)
        self.enqueue(UIEvent.LAST_MOVE, [self.game_state.player.other, move])
        self.enqueue(UIEvent.GAME_OVER, winner)

    def get_cur_player(self):
        return self.players[self.game_state.player]

    def get_cur_player_move(self):
        return self.players[self.game_state.player].select_move(self.game_state)
