# -*- coding: utf-8 -*-
from pcgrl.maze import *
from pcgrl.Sprite import * 

from pygame.locals import *

from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("zelda", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "zelda/")

TILE_SIZE = 64

class Ground(Sprite):
    ID = 0
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 0):
        self.id = id
        self.name = "Ground"        
        super().__init__(RESOURCES_PATH + "tile64/ground.png", x, y, tile_width, tile_height)

class Block(Sprite):
    ID = 1
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Block"
        super().__init__(RESOURCES_PATH + "tile64/block.png", x, y, tile_width, tile_height)
        #rect = pygame.Rect(x, y,  tile_width, tile_height)
        #hitbox = pygame.sprite.Sprite()
        #hitbox.image = pygame.Surface((tile_width, tile_height))
        #hitbox.rect = rect       
        
        #col = pygame.sprite.collide_rect(self, sprite2)
        #collide = rect1.colliderect(rect2)
        #color = (255, 0, 0) if collide else (255, 255, 255)
        

class DoorEntrance(Sprite):
    ID = 2
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 2):
        self.id = id
        self.name = "Entrance"
        super().__init__(RESOURCES_PATH + "tile64/doorEntrance.png", x, y, tile_width, tile_height)        

class DoorExit(Sprite):
    ID = 3
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 3):
        self.id = id
        self.name = "Exit"
        super().__init__(RESOURCES_PATH + "tile64/doorExit.png", x, y, tile_width, tile_height)

class Coin(Sprite):
    ID = 4
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 4):
        self.id = id
        self.name = "Coin"
        super().__init__(RESOURCES_PATH + "tile64/coin.png", x, y, tile_width, tile_height)        

class Key(Sprite):
    ID = 5
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 5):
        self.id = id
        self.name = "Key"
        super().__init__(RESOURCES_PATH + "tile64/key.png", x, y, tile_width, tile_height)                

class Player(Sprite):
    ID = 6
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 6):
        self.id = id
        self.name = "Player"
        super().__init__(RESOURCES_PATH + "tile64/player.png", x, y, tile_width, tile_height)                        

class Enemy(Sprite):
    ID = 7
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 7):
        self.id = id
        self.name = "Enemy"
        super().__init__(RESOURCES_PATH + "tile64/enemy.png", x, y, tile_width, tile_height)                                

class Weapon(Sprite):
    ID = 8
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 8):
        self.id = id
        self.name = "Weapon"
        super().__init__(RESOURCES_PATH + "tile64/weapon.png", x, y, tile_width, tile_height)