# -*- coding: utf-8 -*-
import time
from cv2 import resizeWindow
from pcgrl.mazecoin import *
from pcgrl.Sprite import * 

from pygame.locals import *

#path = os.path.abspath(os.path.join("mazecoin", os.pardir))
#RESOURCES_PATH = os.path.join(path, "pcgrl/resources/mazecoin/")
from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("mazecoin", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "mazecoin/")

TILE_SIZE = 64

class Ground(Sprite):
    ID = 0
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 0):
        self.id = id
        self.name = "Ground"        
        dir_tile = f"tile{tile_width}"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "ground.png")
        super().__init__(path, x, y, tile_width, tile_height)

class Block(Sprite):
    ID = 1
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Block"
        dir_tile = f"tile{tile_width}"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        super().__init__(path, x, y, tile_width, tile_height)

    def create(self):
        """
        rect = pygame.Rect(self.rect.x+1, self.rect.y, (self.width*self.parent.scale), self.width)
        aux = pygame.sprite.Sprite()
        aux.image = pygame.Surface((self.width, self.width))
        aux.rect = rect
        b = pygame.sprite.spritecollide(aux, self.parent.bases, False)         
        if (b):
            pass
        """



class CoinGold(Sprite):
    ID = 2
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 2):
        self.id = id
        self.name = "CoinGold"
        dir_tile = f"tile{tile_width}"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "coinGold.png")
        super().__init__(path, x, y, tile_width, tile_height)

# -*- coding: utf-8 -*-
from enum import Enum        

class Player(Sprite):
    ID = 3
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 3):
        self.id = id
        self.name = "Player"            
        dir_tile = f"tile{tile_width}"
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "player.png")
        super().__init__(path, x, y, tile_width, tile_height)
        
    def draw(self, screen):
        super().draw(screen)
        #self.parent.draw_text_ext(x=self.getX(), y=self.getY(), text="{},{}".format(self.getX(), self.getY()), color=Color(0,0,0), font=self.parent.fntHUD)
        

    def update(self):                   
        self.rect.x = self.changeX            
        self.rect.y = self.changeY

    def move_right(self):                
        new_pos = min(self.rect.x + self.width, self.parent.get_width()-(self.width*2))                

        x = new_pos
        y = self.getY()       
        #print("{},{}".format(x,y))
        if not self.is_wall_collision(x, y):
            self.changeX = new_pos            
        
        if self.is_wall_collision(x + self.width, y):        
            return -2

        return -1

    def move_left(self):
        new_pos = max(self.rect.x - self.width, self.width)
        x = new_pos
        y = self.getY()       
        #print("{},{}".format(x,y))
        if not self.is_wall_collision(x, y):
            self.changeX = new_pos           
        
        if self.is_wall_collision(x - self.width, y):        
            return -2

        return -1
    
    def move_up(self):                
        new_pos = max(self.rect.y - self.height, self.height)
        x = self.getX()       
        y = new_pos        
        if not self.is_wall_collision(x, y):        
            self.changeY = new_pos                
        
        if self.is_wall_collision(x, y- self.height):        
            return -2

        return -1    
    
    def move_down(self):
        new_pos = min(self.rect.y + self.height, self.parent.get_height()-(self.height*2))
        x = self.getX()
        y = new_pos
        #print("{},{}".format(x,y))
        if not self.is_wall_collision(x, y):        
            self.changeY = new_pos
        
        if self.is_wall_collision(x , y+ self.height):                    
            return -2

        return -1

    def step(self, action):

        action_map = {
            0 : 'right',
            1 : 'left',
            2 : 'down',
            3 : 'up',        
            4 : 'no_op'
        }
        reward = 0.0
        if action == 1:
            reward = self.move_right()
        elif action == 2:
            reward = self.move_left()
        elif action == 3:
            reward = self.move_down()
        elif action == 4:
            reward = self.move_up()           
        
        if self.is_coin_collision(self.rect.x, self.rect.y):
            r = 100
            return r

        return reward       

    def do(self, event):        
        pass
        """
        if event.type == pygame.QUIT:
            return True            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                self.move_down()
            elif event.key == pygame.K_LEFT:
                self.move_left()
            elif event.key == pygame.K_RIGHT:
                self.move_right()
            elif event.key == pygame.K_UP:
                self.move_up()          
        """
    def is_wall_collision(self, x, y):
        rect = pygame.Rect(x, y, (self.width*self.parent.scale), self.width)
        aux = pygame.sprite.Sprite()
        aux.image = pygame.Surface((self.width, self.width))
        aux.rect = rect                
        aux.image.fill(pygame.Color(20, 20, 20))
        self.parent.screen.blit(aux.image, self.rect)
        b = pygame.sprite.spritecollide(aux, self.parent.bases, False) 
        #if (b):
        #    print("Colidiu com a parede")
        #aux = None
        #rect = None
        return b

    def is_coin_collision(self, x, y):
        rect = pygame.Rect(x, y, self.width, self.width)
        aux = pygame.sprite.Sprite()
        aux.image = pygame.Surface((self.width, self.width))
        aux.rect = rect                
        list_sprites = pygame.sprite.spritecollide(aux, self.parent.levelObjects, True)
        b = len(list_sprites)
        for e in list_sprites:     
            self.parent.levelObjects.remove(e)        
        return b
