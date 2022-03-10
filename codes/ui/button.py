# reference: https://pythonprogramming.altervista.org/buttons-in-pygame/

from typing import Callable, Iterable, Tuple
import pygame


class Button:
    def __init__(self, text: str):
        """[summary]
        Args:
            text (string): Text to show on button.
        """
        self.__font = pygame.font.SysFont("freesans", 24, bold=True)
        self.__text = text
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Black"))
        self.pos = None
        self.size = None
        self.game_functions: Iterable[Callable[[None], None]] = None
        self.rect = None
        self.surface = None
        self.__activated = True

        self.background_color = (255, 255, 255)

    def set_functions(self, *functions: Iterable[Callable[[None], None]]) -> None:
        self.game_functions = functions

    def locate(self, pos: Tuple[int, int], size: Tuple[int, int] = None) -> None:
        self.pos = pos
        if size is not None:
            self.size = size

        assert self.pos is not None and self.size is not None
        self.rect = pygame.Rect(self.pos[0], self.pos[1], self.size[0], self.size[1])
        self.surface = pygame.Surface(self.size)

    def update_text(self, text: str) -> None:
        self.__text = text
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Black"))
        self.render_text()

    def render_text(self) -> None:
        self.surface.fill(self.background_color)
        self.border = pygame.draw.rect(self.surface, "Black", [0, 0, self.size[0], self.size[1]], 1)
        self.surface.blit(self.__text_obj, (5, self.size[1] // 2))

    def click(self, mouse_coord: Tuple[int, int]) -> None:
        if self.__activated:
            if self.rect.collidepoint(mouse_coord[0], mouse_coord[1]):
                print("clicked!")
                if(self.game_functions is not None):
                    for function in self.game_functions:
                        function()

    def activate(self) -> None:
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Black"))
        self.render_text()
        self.__activated = True

    def deactivate(self) -> None:
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Gray"))
        self.render_text()
        self.__activated = False
