from typing import Optional, Tuple
from games.game_components import Player, Point
from games.abstract_game_state import AbstractGameState
from games.abstract_rule import AbstractRule
import utils


class TicTacToeRule(AbstractRule):
    def __init__(self, board_size: int):
        """[summary]
        Args:
            board_size (int): Size of board.
        """
        super().__init__(board_size)

    def get_stone_count(self, game_state: AbstractGameState, point: Point, direction: int) -> int:
        cnt = 1
        player_stone = game_state.player.other.value
        for i in range(2):
            dx, dy = self.get_dx_dy(direction + i * 4)
            next_point = point
            while True:
                next_point = Point(next_point.row + dy, next_point.col + dx)
                if utils.is_on_grid(next_point, self.board_size):
                    if game_state.board.get(next_point) == player_stone:
                        cnt += 1
                    else:
                        break
                else:
                    break
        return cnt

    def check_game_over(self, game_state: AbstractGameState) -> Tuple[Optional[Player], bool]:
        game_over = False
        winner = None
        
        if game_state.last_move.is_play:
            game_over = self.is_three(game_state, game_state.last_move.point)
            if game_over:
                winner = game_state.player.other                
            elif game_state.board.get_num_empty_points() == 0:
                game_over = True
                winner = Player.both
                
        return winner, game_over

    def is_three(self, game_state: AbstractGameState, point: Point) -> bool:
        for i in range(4):
            cnt = self.get_stone_count(game_state, point, i)
            if cnt == 3:
                return True
        return False
