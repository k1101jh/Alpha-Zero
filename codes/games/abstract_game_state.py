# -*- coding:utf-8 -*-
from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, TypeVar

from games.abstract_rule import AbstractRule
from games.board import Board
from games.game_components import Move, Player


Self = TypeVar("Self", bound="AbstractGameState")


class AbstractGameState(metaclass=ABCMeta):
    def __init__(self, rule: AbstractRule, board: Board, player: Player, last_move: Optional[Move]):
        """[summary]
        Args:
            rule (AbstractRule): [description]
            board (Board): Game board.
            player ([type]): [description]
            last_move (Move): Last movement.
        """
        self.rule: AbstractRule = rule
        self.board: Board = board
        self.player: Player = player
        self.game_over: bool = False
        self.winner: Player = None
        self.last_move: Move = last_move
    
    def get_board(self) -> Board:
        return self.board

    def get_board_size(self) -> int:
        return self.board.board_size

    def check_game_over(self) -> bool:
        if not self.game_over:
            self.winner, self.game_over = self.rule.check_game_over(self)
        return self.game_over

    def get_num_available_points(self) -> int:
        return self.board.num_empty_points

    @classmethod
    def new_game(cls, board_size: int, rule: AbstractRule) -> Self:
        board = Board(board_size)
        return cls(rule, board, Player.black, None)

    @abstractmethod
    def apply_move(self, move: Move) -> Self:
        pass

    @abstractmethod
    def check_valid_move(self, move: Move) -> bool:
        pass

    @abstractmethod
    def check_valid_move_idx(self, move_idx: int) -> bool:
        pass
