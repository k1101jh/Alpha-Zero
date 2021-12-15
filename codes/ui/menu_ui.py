import pygame

from codes.types import Player
from codes.ui.button import Button


pygame.font.init()
BASIC_FONT = pygame.font.SysFont('freesans', 24, bold=True)
LARGE_FONT = pygame.font.SysFont('freesans', 32, bold=True)


class MenuUI:
    def __init__(self, game, size, pos):
        self.size = size
        self.game = game
        self.pos = pos

        self.surface = pygame.Surface(self.size)
        self.rect = pygame.Rect(0, 0, self.size[0], self.size[1])

        self.buttons = []
        # self.new_game_button = Button('New Game', self.game.init_game)
        # self.buttons.append(self.new_game_button)

        # self.num_buttons = len(self.buttons)
        # self.button_width = self.size[0] // self.num_buttons
        # self.button_height = 75
        # self.button_size = [self.button_width, self.button_height]

        # self.new_game_button.locate((self.pos[0], self.pos[1]), self.button_size)

    def check_button_clicked(self, pos):
        for button in self.buttons:
            button.click(pos)

    def render(self):
        self.surface.fill(pygame.Color('Black'))

        # print turn
        if self.game.game_state.game_over:
            self.win_msg(self.game.game_state.winner)
        else:
            self.turn_msg(self.game.game_state.player)

        # render buttons
        for i, button in enumerate(self.buttons):
            button.render_text()
            self.surface.blit(button.surface, (i * self.button_width, 0))

    def turn_msg(self, next_player):
        if next_player is Player.black:
            turn_surf = BASIC_FONT.render("Black's Turn!", True, pygame.Color('White'))
        else:
            turn_surf = BASIC_FONT.render("White's Turn!", True, pygame.Color('White'))
        turn_rect = turn_surf.get_rect()
        turn_rect.topleft = (30, 100)
        self.surface.blit(turn_surf, turn_rect)

    def win_msg(self, winner):
        if winner is Player.black:
            win_surf = BASIC_FONT.render("Black win!!", True, pygame.Color('White'))
        elif winner is Player.white:
            win_surf = BASIC_FONT.render("White win!!", True, pygame.Color('White'))
        else:
            win_surf = BASIC_FONT.render("Draw!!", True, pygame.Color('White'))
        win_rect = win_surf.get_rect()
        win_rect.topleft = (30, 100)
        self.surface.blit(win_surf, win_rect)
