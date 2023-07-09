# -*- coding:utf-8 -*-
import copy
from typing import TypeVar

from games.abstract_game_state import AbstractGameState
from games.game_components import Move, Player, Point
from games.board import Board
from games.tic_tac_toe.tic_tac_toe_rule import TicTacToeRule
import utils


Self = TypeVar("Self", bound="TicTacToeGameState")


class TicTacToeGameState(AbstractGameState):
    def __init__(self, rule: TicTacToeRule, board: Board, player: Player, last_move: Move):
        """[summary]
        Args:
            rule (AbstractRule): [description]
            board (Board): Game board.
            player ([type]): [description]
            last_move (Move): Last movement.
        """
        super().__init__(rule, board, player, last_move)

    def apply_move(self, move: Move) -> Self:
        """[summary]
            Apply move on board.
            Move can't be pass.
        Args:
            move (Move): [description]

        Returns:
            GameState: Next GameState which move is applied.
        """
        next_board = copy.deepcopy(self.board)
        next_board.place_stone(self.player, move.point)

        return TicTacToeGameState(self.rule, next_board, self.player.other, last_move=move)

    def check_valid_move(self, move: Move) -> bool:
        """[summary]
            If point on board is 0, return True.
        Args:
            move ([type]): [description]

        Returns:
            bool: Is move valid.
        """
        return utils.is_on_grid(move.point, self.get_board_size()) and self.board.get(move.point) == 0

    def check_valid_move_idx(self, move_idx: int) -> bool:
        point = Point(move_idx // self.get_board_size(), move_idx % self.get_board_size())
        return self.board.get(point) == 0
