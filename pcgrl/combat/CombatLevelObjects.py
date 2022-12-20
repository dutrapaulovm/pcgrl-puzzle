# -*- coding: utf-8 -*-
from pcgrl.maze import *
from pcgrl.Sprite import * 

from pygame.locals import *

path = os.path.abspath(os.path.join("maze", os.pardir))
RESOURCES_COMBAT_PATH = os.path.join(path, "pcgrl/resources/combat/")

class TankBlue(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "TankBlue"        
        super().__init__(RESOURCES_PATH + "tile/tank_blue.png", x, y, tile_width, tile_height)

class TankDark(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "TankDark"        
        super().__init__(RESOURCES_PATH + "tile/tank_dark.png", x, y, tile_width, tile_height)

class TankGreen(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "TankGreen"        
        super().__init__(RESOURCES_PATH + "tile/tank_green.png", x, y, tile_width, tile_height)

class TankRed(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "TankRed"
        super().__init__(RESOURCES_PATH + "tile/tank_red.png", x, y, tile_width, tile_height)
        
class Grass(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "Grass"        
        super().__init__(RESOURCES_PATH + "tile/tileGrass2.png", x, y, tile_width, tile_height)

class Block(Sprite):
    ID = -1
    def __init__(self, x, y, tile_width = 64, tile_height = 64, id = 1):
        self.id = id
        self.name = "Block"
        super().__init__(RESOURCES_PATH + "tile/block_05.png", x, y, tile_width, tile_height)