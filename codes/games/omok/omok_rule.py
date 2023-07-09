from typing import Optional, Tuple
from games.game_components import Player, Point
from games.abstract_game_state import AbstractGameState
from games.abstract_rule import AbstractRule
import utils


class OmokFreeRule(AbstractRule):
    """
    direction:
    2 4 6
    0   1
    7 5 3
    """
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
        
        # 마지막 이동이 돌을 놓는 행동이었을 경우
        if game_state.last_move.is_play:
            game_over = self.is_five(game_state, game_state.last_move.point)
            if game_over:
                winner = game_state.player.other
            elif game_state.get_board().get_num_empty_points() == 0:
                game_over = True
                winner = Player.both
            else:
                winner = None
        
        # 더 이상 둘 곳이 없으면 게임 종료
        if game_state.board.get_num_empty_points() == 0:
            game_over = True
            winner = Player.both

        return winner, game_over

    def is_five(self, game_state: AbstractGameState, point: Point) -> bool:
        for i in range(4):
            cnt = self.get_stone_count(game_state, point, i)
            if cnt == 5:
                return True
        return False


class OmokRenjuRule(OmokFreeRule):
    """
    direction:
    2 4 6
    0   1
    7 5 3
    """
    def __init__(self, board_size: int):
        """[summary]
        Args:
            board_size (int): Size of board.
        """
        super().__init__(board_size)
