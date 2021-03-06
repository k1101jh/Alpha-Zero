from typing import Tuple
import pygame

from games.game_types import Player


PLAYER_COLOR = {
    Player.black: "black",
    Player.white: "white",
}


class StoneSprite(pygame.sprite.Sprite):
    def __init__(self, group, player: Player, pos: Tuple[int, int], size: Tuple[int, int]):
        """[summary]
        Args:
            group ([type]): [description]
            player ([type]): [description]
            pos (list): [description]
            size ([type]): [description]
        """
        super().__init__(group)
        global PLAYER_COLOR

        self.pos = pos
        self.size = size

        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=self.pos)

        pygame.draw.circle(self.image,
                           PLAYER_COLOR[player],
                           [self.size // 2, self.size // 2],
                           self.size // 2)


class VisitCountSprite(pygame.sprite.Sprite):
    def __init__(self, group, pos: Tuple[int, int], size: Tuple[int, int], text: str):
        """[summary]
        Args:
            group ([type]): [description]
            pos ([type]): [description]
            size ([type]): [description]
            text (string): visit count.
        """
        super().__init__(group)
        self.__font = pygame.font.SysFont("freesans", 24, bold=True)
        self.__text = text
        self.__text_obj = self.__font.render(self.__text, True, pygame.Color("Red"))

        self.pos = pos
        self.size = size

        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=self.pos)

        self.image.blit(self.__text_obj, (10, self.size // 2))
