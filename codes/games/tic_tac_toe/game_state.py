# -*- coding:utf-8 -*-
import copy

from codes.games.board import Board
from codes.game_types import Player
from codes.game_types import Point
from codes import utils


class GameState:
    def __init__(self, rule, board, player, last_move):
        """[summary]

        Args:
            rule (AbstractRule): [description]
            board (Board): Game board.
            player ([type]): [description]
            last_move (Move): Last movement.
        """
        self.rule = rule
        self.board = board
        self.player = player
        self.game_over = False
        self.winner = None
        self.last_move = last_move

    def __deepcopy__(self, memo):
        """[summary]

        Args:
            memodict (dict, optional): [description]. Defaults to {}.

        Returns:
            GameState: Copied GameState.
        """
        copy_object = GameState(self.rule, self.board, self.player, self.last_move)
        copy_object.game_over = self.game_over
        copy_object.winner = self.winner

        return copy_object

    def apply_move(self, move):
        """        apply move on board
        move can't be pass
        """
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.player, move.point)

        return GameState(self.rule, next_board, self.player.other, last_move=move)

    def change_turn(self):
        self.player = self.player.other

    def check_valid_move(self, move):
        """        if point on board is 0, return True
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

    def get_winner(self):
        return self.winner

    @classmethod
    def new_game(cls, board_size, rule_constructor):
        board = Board(board_size)
        rule = rule_constructor(board_size)
        return GameState(rule, board, Player.black, None)
