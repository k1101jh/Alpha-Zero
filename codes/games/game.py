import threading

from codes.types import Player


class Game(threading.Thread):
    def __init__(self, game_state_constructor, rule_constructor,
                 players, board_queue=None, move_queue=None):
        super().__init__()
        self.daemon = True

        self.game_state_constructor = game_state_constructor
        self.rule_constructor = rule_constructor
        self.game_state = game_state_constructor.new_game(rule_constructor)
        self.players = players
        self.board_queue = board_queue
        self.move_queue = move_queue

    def init_game(self):
        self.game_state = self.game_state_constructor.new_game(self.rule_constructor)

    def get_board_size(self):
        return self.game_state.board.board_size

    def get_game_state(self):
        return self.game_state

    def run(self):
        while self.game_state.check_can_play():
            move = self.players[self.game_state.player].select_move(self.game_state)
            self.game_state = self.game_state.apply_move(move)
            if self.game_state.check_game_over():
                break

            if(self.board_queue is not None and self.move_queue is not None):
                self.board_queue.put(self.game_state.board, False)
                self.move_queue.put([self.game_state.player.other, move], False)
                self.board_queue.join()
                self.move_queue.join()

        if self.game_state.winner == Player.both:
            if(self.board_queue is not None and self.move_queue is not None):
                self.board_queue.put(self.game_state.board)
                self.move_queue.put([self.game_state.player.other, move])
                self.board_queue.join()
                self.move_queue.join()

    def get_cur_player(self):
        return self.players[self.game_state.player]

    def get_cur_player_move(self):
        return self.players[self.game_state.player].select_move(self.game_state)
