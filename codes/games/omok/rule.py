from codes.types import Point
from codes.games.abstract_rule import AbstractRule
from codes import utils


class FreeRule(AbstractRule):
    def __init__(self, board_size):
        super().__init__(board_size)

    def get_stone_count(self, game_state, point, direction):
        cnt = 1
        player_stone = game_state.player.other.value
        for i in range(2):
            dx, dy = self.get_dx_dy(direction * 2 + i)
            next_point = point
            while True:
                next_point = Point(next_point.row + dx, next_point.col + dy)
                if utils.is_on_grid(next_point, self.board_size):
                    if game_state.board.get(next_point) == player_stone:
                        cnt += 1
                    else:
                        break
                else:
                    break
        return cnt

    def check_game_over(self, game_state) -> bool:
        game_over = False
        if game_state.last_move.is_play:
            game_over = self.is_five(game_state, game_state.last_move.point)
        return game_over

    def is_five(self, game_state, point):
        for i in range(4):
            cnt = self.get_stone_count(game_state, point, i)
            if cnt == 5:
                return True
        return False
