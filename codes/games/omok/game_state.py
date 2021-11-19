# -*- coding:utf-8 -*-
import numpy as np
import copy

from codes.types import Player
from codes.types import Point
from codes.types import board_size_dict
from codes import utils


class Board:
    def __init__(self):
        self.board_size = board_size_dict['Omok']
        self.grid = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.player_num_stones = {
            Player.black: 0,
            Player.white: 0,
        }

    def __deepcopy__(self, memodict={}):
        copy_object = Board()
        copy_object.grid = np.copy(self.grid)
        copy_object.player_num_stones = utils.copy_dict(self.player_num_stones)

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

    def remain_point_nums(self):
        return self.board_size * self.board_size - self.player_num_stones[Player.black] - self.player_num_stones[Player.white]


class GameState:
    def __init__(self, rule, board, player, last_move):
        self.rule = rule
        self.board = board
        self.player = player
        self.game_over = False
        self.winner = None
        self.last_move = last_move

    def __deepcopy__(self, memodict={}):
        copy_object = GameState(self.rule, self.board, self.player, self.last_move)
        copy_object.game_over = self.game_over
        copy_object.winner = self.winner

        return copy_object

    def apply_move(self, move):
        """
        apply move on board
        move can't be pass
        """
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.player, move.point)

        return GameState(self.rule, next_board, self.player.other, last_move=move)

    def change_turn(self):
        self.player = self.player.other

    def check_valid_move(self, move):
        """
        if point on board is 0, return True
        """
        return self.rule.is_on_grid(move.point) and self.board.get(move.point) == 0

    def check_valid_move_idx(self, move_idx):
        point = Point(move_idx // self.board.board_size, move_idx % self.board.board_size)
        return self.board.get(point) == 0

    def check_game_over(self):
        self.game_over = self.rule.check_game_over(self)
        if self.game_over:
            self.winner = self.player.other
        return self.game_over

    def check_can_play(self):
        """
        check empty point remains
        if there are no empty points, set self.game_over to True and self.winner to Player.both
        """
        if self.board.remain_point_nums() == 0:
            self.game_over = True
            self.winner = Player.both
            return False
        else:
            return not self.game_over

    @classmethod
    def new_game(cls, rule_constructor):
        board = Board()
        rule = rule_constructor(board.board_size)
        return GameState(rule, board, Player.black, None)
