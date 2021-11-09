# -*- coding: utf-8 -*-
import sys
import io

from codes.types import Player
from codes.types import Point

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

COLS = 'ABCDEFGHJKLMNOPQRSTUVWXYZ'
EMPTY = 0
STONE_TO_CHAR = {
    EMPTY: '━╋━',
    Player.black.value: ' ● ',
    Player.white.value: ' ○ ',
}


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    else:
        move_str = '%s%d' % (COLS[move.point.col], move.point.row + 1)
    print('%s %s' % (player, move_str))


def print_board(board):
    for row in range(board.board_size - 1, -1, -1):
        bump = " " if row <= board.board_size else ""
        line = []
        for col in range(0, board.board_size):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%2d %s' % (bump, row + 1, ''.join(line)))
    print('     ' + '  '.join(COLS[:board.board_size]))


def print_winner(winner):
    if winner is Player.both:
        print("DRAW!!!")
    else:
        print(winner.name, "WINS!!!")


def point_from_coords(coords):
    col = COLS.index(coords[0])
    row = int(coords[1:]) - 1
    return Point(row=row, col=col)


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
