"""
reference: https://pythonprogramming.altervista.org/buttons-in-pygame/
"""

import pygame


class Button:
    def __init__(self, text, *game_functions):
        self.__font = pygame.font.SysFont("freesans", 24, bold=True)
        self.__text = text
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Black"))
        self.pos = None
        self.size = None
        self.game_functions = game_functions
        self.__activated = True

        self.background_color = (255, 255, 255)

    def locate(self, pos, size=None):
        self.pos = pos
        if size is not None:
            self.size = size

        assert self.pos is not None and self.size is not None
        self.rect = pygame.Rect(self.pos[0], self.pos[1], self.size[0], self.size[1])
        self.surface = pygame.Surface(self.size)

    def update_text(self, text):
        self.__text = text
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Black"))
        self.render_text()

    def render_text(self):
        self.surface.fill(self.background_color)
        self.border = pygame.draw.rect(self.surface, "Black", [0, 0, self.size[0], self.size[1]], 1)
        self.surface.blit(self.__text_obj, (5, self.size[1] // 2))

    def click(self, mouse_coord):
        if self.__activated:
            if self.rect.collidepoint(mouse_coord[0], mouse_coord[1]):
                print("clicked!")
                for function in self.game_functions:
                    function()

    def activate(self):
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Black"))
        self.render_text()
        self.__activated = True

    def deactivate(self):
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Gray"))
        self.render_text()
        self.__activated = False
