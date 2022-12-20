# -*- coding: utf-8 -*-
from pcgrl.maze import *
from pcgrl.Sprite import * 

from pygame.locals import *
import random

from pcgrl import PCGRLPUZZLE_RESOURCES_PATH
path = os.path.abspath(os.path.join("zelda", os.pardir))
RESOURCES_PATH = os.path.join(PCGRLPUZZLE_RESOURCES_PATH, "zelda/")

TILE_SIZE = 16

class Ground(Sprite):
    ID = 0
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 0):
        self.id = id
        self.name = "Ground"        

        dir_tile = f"tile{tile_width}"

        p = random.randrange(1, 100)
        
        if (p > 50):
            id_image = 1
        elif (p >= 11) and (p <= 50):
            id_image = 2          
        elif (p <= 10):
            id_image = 3
        else:
            id_image = 1

        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "floor{}.png".format(id_image))   

        #super().__init__(RESOURCES_PATH + "tile64/ground.png", x, y, tile_width, tile_height)
        super().__init__(path, x, y, tile_width, tile_height)

class Block(Sprite):
    ID = 1
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 1):
        self.id = id
        self.name = "Block"
        dir_tile = f"tile{tile_width}"
        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "wall.png")   
        super().__init__(path, x, y, tile_width, tile_height)
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
        dir_tile = f"tile{tile_width}"
        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "doorEntrance.png")   
        super().__init__(path, x, y, tile_width, tile_height)

        #super().__init__(RESOURCES_PATH + "tile64/doorEntrance.png", x, y, tile_width, tile_height)        

class DoorExit(Sprite):
    ID = 3
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 3):
        self.id = id
        self.name = "Exit"
        dir_tile = f"tile{tile_width}"
        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "doorExit.png")   
        super().__init__(path, x, y, tile_width, tile_height)

        #super().__init__(RESOURCES_PATH + "tile64/doorExit.png", x, y, tile_width, tile_height)

class Coin(Sprite):
    ID = 4
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 4):
        self.id = id
        self.name = "Coin"
        dir_tile = f"tile{tile_width}"
        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "coin.png")   
        super().__init__(path, x, y, tile_width, tile_height)        
        #super().__init__(RESOURCES_PATH + "tile64/coin.png", x, y, tile_width, tile_height)        

class Key(Sprite):
    ID = 5
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 5):
        self.id = id
        self.name = "Key"
        dir_tile = f"tile{tile_width}"
        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "key.png")   
        super().__init__(path, x, y, tile_width, tile_height)         
        #super().__init__(RESOURCES_PATH + "tile64/key.png", x, y, tile_width, tile_height)                

class Player(Sprite):
    ID = 6
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 6):
        self.id = id
        self.name = "Player"            
        dir_tile = f"tile{tile_width}"
        id_image = random.randrange(1, 4)        
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "player{}.png".format(id_image)) 
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


class Enemy(Sprite):
    ID = 7
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 7):
        self.id = id
        self.name = "Enemy"
        dir_tile = f"tile{tile_width}"
        id_image = random.randrange(1, 6)

        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "enemy{}.png".format(id_image))    
        super().__init__(path, x, y, tile_width, tile_height)
        #super().__init__(RESOURCES_PATH + "tile64/enemy.png", x, y, tile_width, tile_height)                                

class Weapon(Sprite):
    ID = 8
    def __init__(self, x, y, tile_width = TILE_SIZE, tile_height = TILE_SIZE, id = 8):
        self.id = id
        self.name = "Weapon"
        dir_tile = f"tile{tile_width}"
        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        id_image = random.randrange(1, 5)

        #path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "block.png")
        path = "{}{}/{}".format(RESOURCES_PATH, dir_tile, "weapon{}.png".format(id_image))  

        super().__init__(path, x, y, tile_width, tile_height)        
        #super().__init__(RESOURCES_PATH + "tile64/weapon.png", x, y, tile_width, tile_height)