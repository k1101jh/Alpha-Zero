# -*- coding:utf-8 -*-
import numpy as np
import threading

from codes.types import Player
from codes.games.tic_tac_toe.rule import TicTacToeRule


class Board:
    def __init__(self):
        self.board_size = 3
        self.grid = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.player_num_stones = {
            Player.black: 0,
            Player.white: 0,
        }

    def __deepcopy__(self, memodict={}):
        copy_object = Board()
        copy_object.grid = np.copy(self.grid)

        return copy_object

    def place_stone(self, player, point):
        """
        put stone on grid
        """
        self.grid[point] = player.value
        self.player_num_stones[player] += 1

    def is_on_grid(self, point):
        """
        check is point is on grid
        """
        return 0 <= point.row < self.board_size and 0 <= point.col < self.board_size

    def get(self, point):
        """
        get stone on grid
        """
        return self.grid[point]

    def get_grid(self):
        return self.grid


class GameState:
    def __init__(self, rule, board, player, last_move):
        self.rule = rule
        self.board = board
        self.player = player
        self.game_over = False
        self.winner = None
        self.last_move = last_move

        self.num_empty_points = self.board.board_size * self.board.board_size

        print(self.num_empty_points)

    def apply_move(self, move, change_turn=False):
        """
        apply move on board
        """
        if move.is_play:
            self.board.place_stone(self.player, move.point)

        if change_turn:
            self.player = self.player.other
        self.num_empty_points -= 1
        self.last_move = move

    def change_turn(self):
        self.player = self.player.other

    def check_valid_move(self, move):
        """
        if point on board is 0, return True
        """
        return self.rule.is_on_grid(move.point) and self.board.get(move.point) == 0

    def check_game_over(self):
        self.game_over = self.rule.check_game_over(self)
        if self.game_over:
            self.winner = self.player
        return self.game_over

    def check_can_play(self):
        """
        check empty point remains
        if there are no empty points, set self.game_over to True and self.winner to Player.both
        """
        if self.num_empty_points == 0:
            self.game_over = True
            self.winner = Player.both
            return False
        else:
            return True

    @classmethod
    def new_game(cls):
        board = Board()
        rule = TicTacToeRule(board.board_size)
        return GameState(rule, board, Player.black, None)


class TicTacToe(threading.Thread):
    def __init__(self, players, board_queue, move_queue):
        super().__init__()
        self.daemon = True

        self.game_state = GameState.new_game()
        self.players = players
        self.board_queue = board_queue
        self.move_queue = move_queue

    def init_game(self):
        self.game_state = GameState.new_game()

    def get_board_size(self):
        return self.game_state.board.board_size

    def run(self):
        while self.game_state.check_can_play():
            self.board_queue.join()
            self.move_queue.join()

            move = self.players[self.game_state.player].select_move(self.game_state)
            self.game_state.apply_move(move)

            self.board_queue.put(self.game_state.board)
            self.move_queue.put([self.game_state.player, move])

            if self.game_state.check_game_over():
                break

            self.game_state.change_turn()

    def get_game_state(self):
        return self.game_state

    def get_cur_player(self):
        return self.players[self.game_state.player]

    def get_cur_player_move(self):
        return self.players[self.game_state.player].select_move(self.game_state)
