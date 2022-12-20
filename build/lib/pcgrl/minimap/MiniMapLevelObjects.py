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

#path = os.path.abspath(os.path.join("minimap", os.pardir))
#RESOURCES_PATH = os.path.join(path, "pcgrl/resources/minimap/")
from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("minimap", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "minimap/")
TILE_SIZE = 64
class Grass(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "Grass"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_58.png", x, y, tile_width, tile_height)


class Crossing(Sprite):        
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 2):                
        self.id = id
        self.name = "Crossing"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_10.png", x, y, tile_width, tile_height)        

class CrossingTop(Sprite):        
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 3):        
        self.id = id
        self.name = "CrossingTop"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_11.png", x, y, tile_width, tile_height)        

class CrossingBottom(Sprite):
    ID = -1       
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 4):        
        self.id = id
        self.name = "CrossingBottom"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_12.png", x, y, tile_width, tile_height)        

class RoadVertical(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 5):
        self.id = id
        self.name = "RoadVertical"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_08.png", x, y, tile_width, tile_height)

class RoadHorizontal(Sprite):        
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 6):        
        self.id = id
        self.name = "RoadHorizontal"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_09.png", x, y, tile_width, tile_height)

class RoadTopLeft(Sprite):        
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 7):        
        self.id = id
        self.name = "RoadTopLeft"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_22.png", x, y, tile_width, tile_height)

class RoadTopRight(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 8):        
        self.id = id
        self.name = "RoadTopRight"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_22.png", x, y, tile_width, tile_height)

class Trees(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 8):        
        self.id = id
        self.name = "Trees"
        super().__init__(RESOURCES_PATH + "tile/medievalTile_44.png", x, y, tile_width, tile_height)

class House1(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 8):        
        self.id = id
        self.name = "Trees"
        super().__init__(RESOURCES_PATH + "structure/medievalStructure_17.png", x, y, tile_width, tile_height)

class Warehouse(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 8):        
        self.id = id
        self.name = "Warehouse"
        super().__init__(RESOURCES_PATH + "structure/medievalStructure_21.png", x, y, tile_width, tile_height)        
        

class RockGold(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 7):
        self.id = id
        self.name = "RockGold"
        super().__init__(RESOURCES_PATH + "environment/medievalEnvironment_19.png", x, y, tile_width, tile_height)

class Rock(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 7):
        self.id = id
        self.name = "Rock"
        super().__init__(RESOURCES_PATH + "environment/medievalEnvironment_10.png", x, y, tile_width, tile_height)
 
class Fruit(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 7):
        self.id = id
        self.name = "Fruit"
        super().__init__(RESOURCES_PATH + "environment/medievalEnvironment_20.png", x, y, tile_width, tile_height)  
 
 
 
 
class Block(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "Block"
        super().__init__(RESOURCES_PATH + "tile/block_05.png", x, y, tile_width, tile_height) 


     
class Castle(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "Castle"
        super().__init__(RESOURCES_PATH + "structure/medievalStructure_02.png", x, y, tile_width, tile_height)         
        
class Person(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "Person"
        super().__init__(RESOURCES_PATH + "/unit/medievalUnit_06.png", x, y, tile_width, tile_height)         
        
LEVELOBJECTS = {
    "1": Crossing,
    "2": CrossingTop,
    "3": CrossingBottom,
    "4": RoadVertical,
    "5": RoadHorizontal,
    "6": RoadTopLeft,
    "7": RoadTopRight    
}