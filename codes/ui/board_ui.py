import pygame
import threading

from codes.game_types import Player
from codes.game_types import Point
from codes.ui.sprites import StoneSprite
from codes.ui.sprites import VisitCountSprite


BOARD_COLOR = (255, 165, 0)


class BoardUI():
    def __init__(self, board_size, board_length):
        """[summary]

        Args:
            board_size (int): Size of board.
            board_length (int): Length of board ui.
        """

        super().__init__()
        self.daemon = True
        self.board_size = board_size
        self.board_length = board_length
        self.board_rec_size = [self.board_length, self.board_length]
        self.line_interval = self.board_length // (self.board_size + 1)

        self.stone_surface = pygame.Surface(self.board_rec_size, pygame.SRCALPHA)
        self.stone_group = pygame.sprite.RenderPlain()

        self.visit_count_surface = pygame.Surface(self.board_rec_size, pygame.SRCALPHA)
        self.visit_count_group = pygame.sprite.RenderPlain()

        self.surface = pygame.Surface(self.board_rec_size)
        self.surface.fill(BOARD_COLOR)

        self.lock = threading.Lock()

    def get_stone_coord(self, stone_point: Point):
        return [stone_point.col * self.line_interval + self.line_interval,
                stone_point.row * self.line_interval + self.line_interval]

    def point_from_mouse_coords(self, pos):
        row = int((pos[1] - self.line_interval * 0.5) // self.line_interval)
        col = int((pos[0] - self.line_interval * 0.5) // self.line_interval)

        point = Point(row=row, col=col)
        return point

    def set_stone_sprite(self, board):
        """[summary]
            Put stones on board ui.
        Args:
            board (Board): [description]
        """

        self.lock.acquire()
        self.stone_group.empty()
        for row in range(self.board_size):
            for col in range(self.board_size):
                stone_point = Point(row, col)
                stone = board.get(stone_point)
                if stone != 0:
                    stone_coord = self.get_stone_coord(stone_point)
                    if stone == Player.black.value:
                        StoneSprite(self.stone_group, Player.black, stone_coord, self.line_interval)
                    else:
                        StoneSprite(self.stone_group, Player.white, stone_coord, self.line_interval)
        self.lock.release()

    def set_visit_count_sprite(self, visit_counts):
        self.lock.acquire()
        self.visit_count_group.empty()
        if visit_counts is not None:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    counter_coord = self.get_stone_coord(Point(row, col))
                    VisitCountSprite(self.visit_count_group, counter_coord, self.line_interval, str(visit_counts[int(row*self.board_size + col)]))
        self.lock.release()

    def render(self):
        self.lock.acquire()
        self.render_board()
        self.render_stone()
        if len(self.visit_count_group) > 0:
            self.render_visit_counts()
        self.lock.release()

    def render_board(self):
        self.surface.fill(BOARD_COLOR)
        # 첫 번째 줄 위치
        line_start_point = int(self.line_interval)
        line_end_point = int(self.board_size * self.line_interval)
        line_thickness = int(self.line_interval / 20)
        for line_num in range(1, self.board_size + 1):
            axis = int(line_num * self.line_interval)
            pygame.draw.line(self.surface, pygame.Color('Black'),
                             [axis, line_start_point], [axis, line_end_point], line_thickness)
            pygame.draw.line(self.surface, pygame.Color('Black'),
                             [line_start_point, axis], [line_end_point, axis], line_thickness)

    def render_stone(self):
        self.stone_surface.fill((0, 0, 0, 0))
        self.stone_group.draw(self.stone_surface)
        self.surface.blit(self.stone_surface, (0, 0))

    def render_visit_counts(self):
        self.visit_count_surface.fill((0, 0, 0, 0))
        self.visit_count_group.draw(self.visit_count_surface)
        self.surface.blit(self.visit_count_surface, (0, 0))
