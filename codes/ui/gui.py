import sys
import pygame
import queue
import threading

from codes.games.game import Game
from codes.types import Point
from codes.types import Player
from codes.types import Move
from codes.ui.board_ui import BoardUI
from codes.ui.menu_ui import MenuUI
from codes import utils


FPS = 30
BOARD_LENGTH = 800
MENU_WIDTH = 400
MENU_HEIGHT = BOARD_LENGTH
WINDOW_WIDTH = BOARD_LENGTH + MENU_WIDTH
WINDOW_HEIGHT = BOARD_LENGTH
WINDOW_REC_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
MENU_REC_SIZE = (MENU_WIDTH, MENU_HEIGHT)
BOARD_REC_SIZE = (BOARD_LENGTH, BOARD_LENGTH)
BOARD_POS = (0, 0)
MENU_POS = (BOARD_LENGTH, 0)


class GUI:
    def __init__(self, game_state_constructor, rule_constructor, players):
        self.board_queue = queue.Queue()
        self.move_queue = queue.Queue()
        self.visit_count_queue = queue.Queue()
        self.mouse_input_queue = queue.Queue()
        self.lock = threading.Lock()

        players[Player.black].set_input_queue(self.mouse_input_queue)
        players[Player.white].set_input_queue(self.mouse_input_queue)
        self.game = Game(game_state_constructor,
                         rule_constructor,
                         players,
                         self.board_queue,
                         self.move_queue,
                         self.visit_count_queue)
        self.board_size = self.game.get_board_size()

        # pygame setting
        pygame.init()
        pygame.display.set_caption("Alpha Zero")
        pygame.event.set_allowed([pygame.QUIT,
                                  pygame.MOUSEBUTTONUP])

        self.menu_ui = MenuUI(self.game, MENU_REC_SIZE, MENU_POS)
        self.board_ui = BoardUI(self.board_size,
                                BOARD_LENGTH,
                                self.board_queue,
                                self.move_queue,
                                self.visit_count_queue, self.lock)

        self.clock = pygame.time.Clock()
        self.display_surf = pygame.display.set_mode(WINDOW_REC_SIZE)

    def run(self):
        self.game.start()
        self.board_ui.start()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_coord = pygame.mouse.get_pos()
                    self.menu_ui.check_button_clicked(mouse_coord)
                    move = Move(self.board_ui.point_from_mouse_coords(mouse_coord))
                    if utils.is_on_grid(move.point, self.board_size):
                        self.mouse_input_queue.put(move)
            self.lock.acquire()
            self.render()
            self.lock.release()

    def queue_task_done(self):
        self.move_queue.task_done()
        self.board_queue.task_done()

    def _render_menu(self):
        self.menu_ui.render()
        self.display_surf.blit(self.menu_ui.surface, MENU_POS)

    def _render_board(self):
        self.board_ui.render()
        self.display_surf.blit(self.board_ui.surface, BOARD_POS)

    def render(self):
        self._render_menu()
        self._render_board()
        pygame.display.update()
        self.clock.tick(FPS)
