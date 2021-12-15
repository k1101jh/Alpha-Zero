# -*- coding:utf-8 -*-
import numpy as np
import copy

from codes.types import Player
from codes.types import Point
from codes import utils


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
        copy_object.player_num_stones = utils.copy_dict(self.player_num_stones)

        return copy_object

    def place_stone(self, player, point):
        """
        put stone on grid
        """
        self.grid[point] = player.value
        self.player_num_stones[player] += 1

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
        return utils.is_on_grid(move.point, self.board.board_size) and self.board.get(move.point) == 0

    def check_valid_move_idx(self, move_idx):
        point = Point(move_idx // self.board.board_size, move_idx % self.board.board_size)
        return self.board.get(point) == 0

    def check_game_over(self):
        if self.game_over:
            return self.game_over
        else:
            if self.rule.check_game_over(self):
                # 규칙에 의한 게임 종료 확인
                self.game_over = True
                self.winner = self.player.other
            elif self.board.remain_point_nums() == 0:
                # 규칙에 의해 게임이 종료되지 않았고
                # 돌을 놓을 자리가 없는 경우 무승부
                self.game_over = True
                self.winner = Player.both
            return self.game_over

    @classmethod
    def new_game(cls, rule_constructor):
        board = Board()
        rule = rule_constructor(board.board_size)
        return GameState(rule, board, Player.black, None)
