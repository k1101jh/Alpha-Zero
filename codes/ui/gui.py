import sys
import pygame
import queue

from codes.games.game import Game
from codes.game_types import Player
from codes.game_types import Move
from codes.ui.board_ui import BoardUI
from codes.ui.menu_ui import MenuUI
from codes import utils
from codes.game_types import UIEvent


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
    def __init__(self, game_type, rule_type, players):
        """[summary]
            Play game on GUI.
        Args:
            game_type ([type]): [description]
            rule_type ([type]): [description]
            players (dict): [description]
        """
        self.event_queue = queue.Queue()
        self.mouse_input_queue = queue.Queue()

        self.game_type = game_type
        self.rule_type = rule_type
        self.players = players

        self.players[Player.black].set_input_queue(self.mouse_input_queue)
        self.players[Player.white].set_input_queue(self.mouse_input_queue)
        self.game = Game(game_type,
                         rule_type,
                         players,
                         self.event_queue)
        self.board_size = self.game.get_board_size()

        # pygame setting
        pygame.init()
        pygame.display.set_caption("Alpha Zero")
        pygame.event.set_allowed([pygame.QUIT,
                                  pygame.MOUSEBUTTONUP])

        self.menu_ui = MenuUI(self, MENU_REC_SIZE, MENU_POS)
        self.board_ui = BoardUI(self.board_size,
                                BOARD_LENGTH)

        self.clock = pygame.time.Clock()
        self.display_surf = pygame.display.set_mode(WINDOW_REC_SIZE)

    def new_game_start(self):
        self.game = Game(self.game_type,
                         self.rule_type,
                         self.players,
                         self.event_queue)
        self.mouse_input_queue.queue.clear()
        self.game.start()

    def run(self):
        self.game.start()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_coord = pygame.mouse.get_pos()
                    # 메뉴 버튼이 눌렸는지 확인
                    self.menu_ui.check_button_clicked(mouse_coord)
                    move = Move(self.board_ui.point_from_mouse_coords(mouse_coord))
                    # 마우스로 돌을 놓는 경우
                    if utils.is_on_grid(move.point, self.board_size):
                        self.mouse_input_queue.put(move)

            # 메뉴에 현재 플레이어 표시
            game_state = self.game.get_game_state()
            if not game_state.game_over:
                self.menu_ui.update_turn(game_state.player)

            # 이벤트 처리
            if(not self.event_queue.empty()):
                event, val = self.event_queue.get()
                if event == UIEvent.BOARD:
                    self.board_ui.set_stone_sprite(val)
                elif event == UIEvent.VISIT_COUNTS:
                    self.board_ui.set_visit_count_sprite(val)
                elif event == UIEvent.GAME_OVER:
                    self.menu_ui.game_over(val)

                self.event_queue.task_done()
            self.render()

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

    def get_game(self):
        return self.game
