# -*- coding: utf-8 -*-
import sys
import os
import pygame
from pygame.locals import *

from pcgrl.SpriteSheet import SpriteSheet

class Sprite(pygame.sprite.Sprite):
    
    def __init__(self, tile, x, y, w = 32, h = 32, id = 0):        
        pygame.sprite.Sprite.__init__(self)
        
        self.sprites = None
        try:            
            #Load the spritesheet of frames for this player
            self.sprites = SpriteSheet(tile)
        except:
            print("An exception occurred when try to load file: {}".format(tile))
                    
        self.image = self.sprites.image_at((0, 0, w, h))
        
        #Set position
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
                
        #Set speed and direction
        self.changeX = x
        self.changeY = y        
        self.id = 0  
        self.name = "" 
        self.parent = None
        self.xscale = w
        self.yscale = h
        self.width  = w
        self.height = h        

    def move_to(self, x, y):
        self.changeX = x
        self.changeY = y
        self.rect.x = x * self.xscale
        self.rect.y = y * self.xscale                 
        
    def move_to_xy(self, x, y):
        self.rect.x = x
        self.rect.y = y
        self.changeX = int(x / self.xscale)
        self.changeY = int(y / self.yscale)            

    def getX(self):
        return self.rect.x
    
    def getY(self):
        return self.rect.y
        
    def update(self):        
        pass

    def create(self):        
        pass
    
    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def reset(self):        
        self.changeX = 0
        self.changeY = 0
        self.rect.x = self.changeX            
        self.rect.y = self.changeY         

    def do(self, event):
        pass
    
    def __str__(self):
        return "[ Id = " + str(self.id) + " , Name " + self.name +  " ] "