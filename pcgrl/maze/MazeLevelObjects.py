# -*- coding: utf-8 -*-
from pcgrl.maze import *
from pcgrl.Sprite import * 

from pygame.locals import *

#path = os.path.abspath(os.path.join("maze", os.pardir))
#RESOURCES_PATH = os.path.join(path, "pcgrl/resources/maze/")
from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("maze", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "maze/")
TILE_SIZE = 64
class Ground(Sprite):
    ID = 0
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 0):
        self.id = id
        self.name = "Ground"        
        super().__init__(RESOURCES_PATH + "tile64/ground.png", x, y, tile_width, tile_height)

class Block(Sprite):
    ID = 1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "Block"
        super().__init__(RESOURCES_PATH + "tile64/block.png", x, y, tile_width, tile_height)

class CoinGold(Sprite):
    ID = 2
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 2):
        self.id = id
        self.name = "CoinGold"
        super().__init__(RESOURCES_PATH + "tile64/coinGold.png", x, y, tile_width, tile_height)