# -*- coding: utf-8 -*-
import os
import sys
import io
import importlib
import math

from codes.game_types import Player
from codes.game_types import Point
from codes.game_types import game_name_dict

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

COLS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
EMPTY = 0
STONE_TO_CHAR = {
    EMPTY: '━╋━',
    Player.black.value: ' ○ ',
    Player.white.value: ' ● ',
}


def get_rule_constructor(game_name, rule_name):
    module = importlib.import_module(f'codes.games.{game_name_dict[game_name]}.rule')
    constructor = getattr(module, rule_name)
    return constructor


def get_game_state_constructor(name):
    module = importlib.import_module(f'codes.games.{game_name_dict[name]}.game_state')
    constructor = getattr(module, "GameState")
    return constructor


def print_turn(game_state):
    print(f'{game_state.player.name} turn!')
    sys.stdout.flush()


def print_move(player_move):
    if player_move is not None:
        player = player_move[0]
        move = player_move[1]
        if move.is_pass:
            move_str = 'passes'
        else:
            move_str = '%s%d' % (COLS[move.point.col], move.point.row + 1)
        print('%s %s' % (player, move_str))
        sys.stdout.flush()


def print_board(board):
    for row in range(board.board_size - 1, -1, -1):
        bump = " " if row <= board.board_size else ""
        line = []
        for col in range(0, board.board_size):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%2d %s' % (bump, row + 1, ''.join(line)))
    print('     ' + '  '.join(COLS[:board.board_size]))
    sys.stdout.flush()


def print_visit_count(visit_counts):
    board_size = int(math.sqrt(len(visit_counts)))
    for row in range(board_size - 1, -1, -1):
        bump = " " if row <= board_size else ""
        print('\n%s%2d' % (bump, row + 1), end='')
        for col in range(0, board_size):
            visit_count = visit_counts[row * board_size + col]
            print('%4d ' % (visit_count), end='')
        print('')
    print('      ' + '    '.join(COLS[:board_size]))
    sys.stdout.flush()


def print_winner(winner):
    if winner is Player.both:
        print("DRAW!!!")
    else:
        print(winner.name, "WINS!!!")
    sys.stdout.flush()


def point_from_coords(coords):
    col = COLS.index(coords[0])
    row = int(coords[1:]) - 1
    return Point(row=row, col=col)


def is_on_grid(point, board_size):
    """
        check point is on grid
    """

    return 0 <= point.row < board_size and 0 <= point.col < board_size


def get_agent_filename(game_name, version, postfix="", prefix=""):
    cur_file_path = os.path.abspath(__file__)
    project_path = os.path.dirname(os.path.dirname(cur_file_path))
    dir_path = os.path.join(project_path, f'save_files/{game_name}/saved_models')
    file_name = f'{postfix}-v{version}{prefix}.pth'

    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, file_name)


def copy_list(input_list):
    ret = input_list.copy()
    for idx, item in enumerate(ret):
        ret[idx] = item
    return ret


def copy_dict(input_dict):
    ret = input_dict.copy()
    for key, value in ret.items():
        ret[key] = value

    return ret
