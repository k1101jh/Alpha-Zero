from typing import Tuple
import pygame

from games.game_components import Player
from ui.button import Button


pygame.font.init()
BASIC_FONT = pygame.font.SysFont('freesans', 24, bold=True)
LARGE_FONT = pygame.font.SysFont('freesans', 32, bold=True)


class MenuUI:
    def __init__(self, gui, size: Tuple[int, int], pos: Tuple[int, int]):
        """[summary]
            Menu to show informations, buttons.
        Args:
            gui ([type]): [description]
            size ([type]): Size of menu ui.
            pos ([type]): Position of menu ui.
        """
        self.size = size
        self.gui = gui
        self.pos = pos

        self.surface = pygame.Surface(self.size)
        self.rect = pygame.Rect(0, 0, self.size[0], self.size[1])

        self.buttons = []
        # 새 게임 버튼 생성
        # 게임이 끝난 경우에만 새 게임 버튼을 누를 수 있음
        self.new_game_button = Button('New Game')
        self.new_game_button.set_functions(self.gui.new_game_start, self.new_game_button.deactivate)
        self.buttons.append(self.new_game_button)

        self.num_buttons = len(self.buttons)
        self.button_width = self.size[0] // self.num_buttons
        self.button_height = 75
        self.button_size = [self.button_width, self.button_height]

        for idx, button in enumerate(self.buttons):
            button_x = self.pos[0] + idx * self.button_width
            button.locate((button_x, self.pos[1]), self.button_size)

        # 버튼 초기화
        self.new_game_button.deactivate()

        # 색상 설정
        self.bg_color = pygame.Color('Black')
        self.text_color = pygame.Color('White')

        # 텍스트 설정
        self.text_surf = BASIC_FONT.render("Initial text", True, pygame.Color('White'))
        self.text_rect = self.text_surf.get_rect()
        self.text_topleft = (30, 100)
        self.text_rect.topleft = self.text_topleft
        self.surface.blit(self.text_surf, self.text_rect)

    def check_button_clicked(self, pos: Tuple[int, int]) -> None:
        for button in self.buttons:
            button.click(pos)

    def render(self) -> None:
        self.surface.fill(self.bg_color)

        # render buttons
        for i, button in enumerate(self.buttons):
            button.render_text()
            self.surface.blit(button.surface, (i * self.button_width, 0))

        # render texts
        self.text_rect = self.text_surf.get_rect()
        self.text_rect.topleft = self.text_topleft
        self.surface.blit(self.text_surf, self.text_rect)

    def update_text(self, text: str) -> None:
        self.text_surf = BASIC_FONT.render(text, True, self.text_color)

    @classmethod
    def get_turn_text(cls, next_player: Player) -> str:
        if next_player is Player.black:
            return "Black Turn!"
        else:
            return "White Turn!"

    @classmethod
    def get_win_text(cls, winner: Player) -> str:
        if winner is Player.black:
            return "Black win!!"
        elif winner is Player.white:
            return "White win!!"
        else:
            return "Draw!!"

    def update_turn(self, next_player: Player) -> None:
        self.update_text(self.get_turn_text(next_player))

    def game_over(self, winner: Player) -> None:
        self.update_text(self.get_win_text(winner))
        self.new_game_button.activate()
