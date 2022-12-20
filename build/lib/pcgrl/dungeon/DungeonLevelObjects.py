# -*- coding: utf-8 -*-
from pcgrl.maze import *
from pcgrl.Sprite import * 

from pygame.locals import *

#path = os.path.abspath(os.path.join("dungeon", os.pardir))
#RESOURCES_PATH = os.path.join(path, "pcgrl/resources/dungeon/")
from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("dungeon", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "dungeon/")

TILE_SIZE = 16

class Ground(Sprite):
    ID = 0
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 0):
        self.id = id
        self.name = "Ground"        
        dir_tile = f"tile{tile_width}"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "floor_1.png")        
        super().__init__(path, x, y, tile_width, tile_height)

class Block(Sprite):
    ID = 1
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Block"
        dir_tile = f"tile{tile_width}"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")                
        super().__init__(path, x, y, tile_width, tile_height)