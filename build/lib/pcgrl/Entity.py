# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import time     
import random
import pygame, sys
from pygame.locals import *
from pcgrl.Sprite import * 
from pcgrl.Entity import * 
from enum import Enum

RESOURCES_PATH = os.path.dirname(__file__) + "/resources/"

class Actions(Enum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    STOP  = 4

    def __str__(self):
        return format(self.value)                                            
        
    def __int__(self):
        return self.value
    
class Entity(Sprite):
        
    def __init__(self, x = 0, y = 0, tile_width = 64, tile_height = 64, id = 0):        
        self.id = id
        self.name = "Entity"             
        super().__init__(RESOURCES_PATH + "agent.png", x, y, tile_width, tile_height)  

    def update(self):        
        self.rect.x = self.changeX            
        self.rect.y = self.changeY

    def move_right(self):                
        new_pos = min(self.rect.x + self.parent.get_state_width(), self.parent.get_width()-self.parent.get_state_width())
        self.changeX = new_pos

    def move_left(self):
        new_pos = max(self.rect.x - self.parent.get_state_width(), 0)
        self.changeX = new_pos    
    
    def move_up(self):                
        new_pos = max(self.rect.y - self.parent.get_state_height(), 0)
        self.changeY = new_pos
    
    def move_down(self):
        new_pos = min(self.rect.y + self.parent.get_state_height(), self.parent.get_height()-self.parent.get_state_height())
        self.changeY = new_pos
        
    def set_pos(self, x, y):                
        self.changeX = x
        self.changeY = y
        self.rect.x = self.changeX            
        self.rect.y = self.changeY           

    def add_posX(self, value):                
        self.changeX = value        
        self.rect.x += self.changeX                    

    def add_posY(self, value):                        
        self.changeY = value      
        self.rect.y += self.changeY         

    def set_posX(self, x):        
        self.changeX = x    
        self.rect.x = self.changeX                
    
    def set_posY(self, y):        
        self.changeY = y        
        self.rect.y = self.changeY            

    def draw(self, screen):        
        x = self.rect.x
        y = self.rect.y        
        self.parent.draw_text(x, y-10, str(x) + ", " + str(y))                        
        #print(str(x) + ", " + str(y))
        screen.blit(self.image, self.rect)
        
    def get_vector2(self):
        pos = pygame.Vector2(self.getX(), self.getY())
        return pos
                
    def __str__(self):
        return "(x,y)="+str(self.rect.x)+", "+str(self.rect.y)