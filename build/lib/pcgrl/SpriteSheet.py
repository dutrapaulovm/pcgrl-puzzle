# -*- coding: utf-8 -*-
import sys
import os
import pygame
from pygame.locals import *

class SpriteSheet(object):
    def __init__(self, fileName):        
        self.sheet = pygame.image.load(fileName).convert_alpha()        

    def image_at(self, rectangle):
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size, pygame.SRCALPHA, 32).convert_alpha()
        image.blit(self.sheet, (0, 0), rect)
        return image