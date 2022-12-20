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

from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("minimap", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "minimap/")

TILE_SIZE = 16
class Grass(Sprite):
    ID = 0
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Grass"
        dir_tile = f"tile{tile_width}"
        p = random.randrange(0, 100)
        
        if (p > 50):
            id_image = 1        
        else:
            id_image = 2

        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "grass{}.png".format(id_image))                        
        
        super().__init__(path, x, y, tile_width, tile_height)        
"""
class Crossing(Sprite):        
    ID = 2
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 2):                
        self.id = id
        self.name = "Crossing"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_10.png", x, y, tile_width, tile_height)        

class CrossingTop(Sprite):        
    ID = 3
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 3):        
        self.id = id
        self.name = "CrossingTop"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_11.png", x, y, tile_width, tile_height)        

class CrossingBottom(Sprite):
    ID = 4       
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 4):        
        self.id = id
        self.name = "CrossingBottom"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_12.png", x, y, tile_width, tile_height)        

class RoadVertical(Sprite):
    ID = 5
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 5):
        self.id = id
        self.name = "RoadVertical"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_08.png", x, y, tile_width, tile_height)

class RoadHorizontal(Sprite):        
    ID = 6
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 6):        
        self.id = id
        self.name = "RoadHorizontal"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_09.png", x, y, tile_width, tile_height)

class RoadTopLeft(Sprite):        
    ID = 7
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 7):        
        self.id = id
        self.name = "RoadTopLeft"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_22.png", x, y, tile_width, tile_height)

class RoadTopRight(Sprite):
    ID = 8
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 8):        
        self.id = id
        self.name = "RoadTopRight"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_22.png", x, y, tile_width, tile_height)
"""
class Trees(Sprite):
    ID = 1
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 8):        
        self.id = id
        self.name = "Trees"
        dir_tile = f"tile{tile_width}"
        p = random.randrange(0, 100)
        
        if (p > 50):
            id_image = "0094"
        else:
            id_image = "0095"

        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "tile_{}.png".format(id_image))   
        super().__init__(path, x, y, tile_width, tile_height)                

class RockGold(Sprite):
    ID = 2
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 7):
        self.id = id
        self.name = "RockGold"
        dir_tile = f"tile{tile_width}"
        p = random.randrange(0, 100)
        
        if (p > 50):
            id_image = "0101"
        else:
            id_image = "0124"

        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "tile_{}.png".format(id_image))   
        super().__init__(path, x, y, tile_width, tile_height)                

class Rock(Sprite):
    ID = 3
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 7):
        self.id = id
        self.name = "Rock"
        dir_tile = f"tile{tile_width}"
        p = random.randrange(0, 100)
        
        if (p > 50):
            id_image = "0098"
        else:
            id_image = "0099"

        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "tile_{}.png".format(id_image)) 
        super().__init__(path, x, y, tile_width, tile_height)                

class House1(Sprite):
    ID = 4
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 8):        
        self.id = id
        self.name = "Trees"
        dir_tile = f"tile{tile_width}"
        
        id_image = "0143"        

        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "tile_{}.png".format(id_image)) 
        super().__init__(path, x, y, tile_width, tile_height)                

class Person(Sprite):
    ID = 5
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Person"
        dir_tile = f"tile{tile_width}"
        p = random.randrange(0, 100)
        
        if (p > 50):
            id_image = "0132"
        else:
            id_image = "0133"

        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "tile_{}.png".format(id_image)) 
        super().__init__(path, x, y, tile_width, tile_height)  


class Fruit(Sprite):
    ID = 6
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 7):
        self.id = id
        self.name = "Fruit"
        dir_tile = f"tile{tile_width}"
        id_image = "0102"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "tile_{}.png".format(id_image)) 
        super().__init__(path, x, y, tile_width, tile_height)  

class Warehouse(Sprite):
    ID = 7
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 8):        
        self.id = id
        self.name = "Warehouse"
        dir_tile = f"tile{tile_width}"
        id_image = "0146"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "tile_{}.png".format(id_image)) 
        super().__init__(path, x, y, tile_width, tile_height)  
     
class Castle(Sprite):
    ID = 8
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Castle"
        dir_tile = f"tile{tile_width}"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "structure/medievalStructure_02.png")
        super().__init__(path, x, y, tile_width, tile_height)  
        

        
