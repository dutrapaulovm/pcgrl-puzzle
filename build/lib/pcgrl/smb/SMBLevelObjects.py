# -*- coding: utf-8 -*-
from pcgrl.maze import *
from pcgrl.Sprite import * 

from pygame.locals import *

from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("smb", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "smb/")

TILE_SIZE = 16
ACC = 0.5
FRIC = -0.12
FPS = 60

class Background(Sprite):
    ID = 0
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 0):
        self.id = id
        self.name = "Ground"        
        super().__init__(RESOURCES_PATH + "background.png", x, y, tile_width, tile_height)

class Ground(Sprite):
    ID = 1
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Block"
        super().__init__(RESOURCES_PATH + "ground.png", x, y, tile_width, tile_height)

class Ladder(Sprite):
    ID = 2
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Ladder"
        super().__init__(RESOURCES_PATH + "ladder.png", x, y, tile_width, tile_height)

class Block(Sprite):
    ID = 3
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Block"
        super().__init__(RESOURCES_PATH + "block.png", x, y, tile_width, tile_height)

class DoorEntrance(Sprite):
    ID = 4
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 2):
        self.id = id
        self.name = "Entrance"
        super().__init__(RESOURCES_PATH + "doorEntrance.png", x, y, tile_width, tile_height)        

class DoorExit(Sprite):
    ID = 5
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 3):
        self.id = id
        self.name = "Exit"
        super().__init__(RESOURCES_PATH + "doorExit.png", x, y, tile_width, tile_height)

class Key(Sprite):
    ID = 6
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 5):
        self.id = id
        self.name = "Key"
        super().__init__(RESOURCES_PATH + "key.png", x, y, tile_width, tile_height)                

class Coin(Sprite):
    ID = 7
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 4):
        self.id = id
        self.name = "Coin"
        super().__init__(RESOURCES_PATH + "coin.png", x, y, tile_width, tile_height)        

class Player(Sprite):
    ID = 8
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 6):
        self.id = id
        vec = pygame.math.Vector2 #2 for two dimensional        
        self.name = "Player"
        super().__init__(RESOURCES_PATH + "player.png", x, y, tile_width, tile_height)                        
        self.vec = pygame.math.Vector2 #2 for two dimensional
        self.pos = vec((x, y))
        self.vel = vec(0,0)
        self.acc = vec(0,0)
        self.jumping = False
        self.platforms = pygame.sprite.Group()
    """
    def do(self):
        self.acc = self.vec(0,0.5)
    
        pressed_keys = pygame.key.get_pressed()
                
        if pressed_keys[K_LEFT]:
            self.acc.x = -ACC
        if pressed_keys[K_RIGHT]:
            self.acc.x = ACC
                 
        self.acc.x += self.vel.x * FRIC
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc
         
        if self.pos.x > 720: #WIDTH:
            self.pos.x = 0
        if self.pos.x < 0:
            self.pos.x = 720#WIDTH
             
        self.rect.midbottom = self.pos
 
    def jump(self): 
        hits = pygame.sprite.spritecollide(self, platforms, False)
        if hits and not self.jumping:
           self.jumping = True
           self.vel.y = -15
 
    def cancel_jump(self):
        if self.jumping:
            if self.vel.y < -3:
                self.vel.y = -3
                 
    def update(self):
        super().update()
        hits = pygame.sprite.spritecollide(self, platforms, False)
        if self.vel.y > 0:        
            if hits:
                if self.pos.y < hits[0].rect.bottom:               
                    self.pos.y = hits[0].rect.top +1
                    self.vel.y = 0
                    self.jumping = False        
    """
    
class Enemy(Sprite):
    ID = 9
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 7):
        self.id = id
        self.name = "Enemy"
        super().__init__(RESOURCES_PATH + "enemy.png", x, y, tile_width, tile_height)