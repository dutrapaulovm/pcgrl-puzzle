import time
import pygame
import pytmx
import os
import time

class TiledMap():
    """ This is creating the surface on which you make the draw updates """
    def __init__(self):
        self.parent_dir = os.path.dirname(__file__) + "/pcgrl/mapbuilder/resources/maps/"
        self.gameMap = pytmx.load_pygame(self.parent_dir + "map4_4_2.tmx", pixelalpha=True)
        self.mapwidth = self.gameMap.tilewidth * self.gameMap.width
        self.mapheight = self.gameMap.tileheight * self.gameMap.height
        self.fntHUD      = pygame.font.Font('freesansbold.ttf', 24)     
        self.fntSmallHUD = pygame.font.Font('freesansbold.ttf', 16)        

    def draw_text_ext(self,x = 0, y = 0, text = "", color = (200, 000, 000), font = None, screen=None):
        text = str(text)        
        text = font.render(text, True, color)
        screen.blit(text, (x, y))  

    def draw(self, screen):
        for layer in self.gameMap.visible_layers:
            if isinstance(layer, pytmx.TiledTileLayer):
                for x, y, gid in layer:
                    tile = self.gameMap.get_tile_image_by_gid(gid)                                        
                    if tile:                        
                        self.draw_text_ext(x * self.gameMap.tilewidth, y * self.gameMap.tileheight, text=str(gid), color=Color(0,0,0), font=self.fntHUD, screen=screen)
                        screen.blit(tile, (x * self.gameMap.tilewidth, y * self.gameMap.tileheight))    
    
# -*- coding: utf-8 -*-
import pygame, sys
from pygame import draw
from pygame.locals import *
from pcgrl.Utils import *
import os

#Background color
BACKGROUND = (20, 20, 20)

class Game():    
    def __init__(self, w = 960, h = 540, tile_w = 32, tile_h = 32):
        pygame.init()
        self.running  = True
        self.clock    = pygame.time.Clock()
        self.updating = True   
        self.env      = None
        self.width  = w
        self.height = h      
        self.tile_width  = tile_w
        self.tile_height = tile_h
        self.scale    = 1                 
        #self.screen   = pygame.display.set_mode((self.width, self.height),HWSURFACE|DOUBLEBUF|RESIZABLE)                        
        self.screen   = pygame.display.set_mode((self.width, self.height))                        
        self.gamescreen = pygame.Surface((self.width, self.height))                        
        self.fullscreen = False     
        self.front      = pygame.sprite.Group([])
        self.enemies    = pygame.sprite.Group([])
        self.bases      = pygame.sprite.Group([])
        self.background = pygame.sprite.Group([])
        self.structure  = pygame.sprite.Group([])
        self.players    = pygame.sprite.Group([])        
        self.fntDefault = pygame.font.SysFont('FFF Intelligent', 16)
        self.map = TiledMap()        
        
    def noisy(self, val):
        between_val = val / 2.0
        if (val < between_val):
            return int(round(self.env.np_random.uniform(val, between_val)))
        else:
            return int(round(self.env.np_random.uniform(between_val, val)))        

    def get_width(self):
        return self.screen.get_width()

    def get_height(self):
        return self.screen.get_height()        

    def get_state_width(self):
        return self.tile_width
    
    def get_state_height(self):
        return self.tile_height

    def get_rows(self):
        return int(self.get_height() / self.get_state_height())
    
    def get_cols(self):
        return int(self.get_width() / self.get_state_width())

    def draw_text(self,x = 0, y = 0, text = "", color = (200, 000, 000)):
        self.draw_text_ext(x, y , text, color, self.fntDefault)
        #text = str(text)        
        #text = self.fntDefault.render(text, True, color)
        #self.gamescreen.blit(text, (x, y))

    def draw_text_ext(self,x = 0, y = 0, text = "", color = (200, 000, 000), font = None):
        text = str(text)        
        text = font.render(text, True, color)
        self.gamescreen.blit(text, (x, y))        

    def draw_hud(self, screen):
        pass

    def run(self):
        while self.running:
            self.render()
            
    def render_rgb(self, tick = 0):
        ar = None
        self.update()  
        self.draw()
        ar = pygame.surfarray.array3d(self.gamescreen).swapaxes(0, 1)  # swap because pygame                                         
        self.clock.tick(tick)             
        return ar
        
                
    def render(self, mode="human", tick = 60):

        ar = None
                
        for event in pygame.event.get():
            self.do(event) 

        if (mode == "human"):  
            self.gamescreen.fill(BACKGROUND)           
                        
            self.update()        
            self.draw()            

            scaled_win = pygame.transform.smoothscale(self.gamescreen, self.screen.get_size())
            self.screen.blit(scaled_win, (0, 0))        
            # Refresh Screen            
            pygame.display.flip()                
            # Create the PixelArray.
            ar = pygame.PixelArray (scaled_win)                    
            self.clock.tick(tick)             
            
        else:
            self.update()  
            ar = pygame.surfarray.array3d(self.gamescreen).swapaxes(0, 1)  # swap because pygame                                         
            #return rgb                  
            #scaled_win = pygame.transform.smoothscale(self.gamescreen, self.screen.get_size())
            #ar = pygame.PixelArray (scaled_win)                    
            
        return ar

    def addBackground(self, object):
        object.parent = self
        self.background.add(object)

    def addFront(self, object):
        object.parent = self
        self.front.add(object)
    
    def addBases(self, object):        
        object.parent = self
        self.bases.add(object)           
        
    def addStructure(self, object):        
        object.parent = self
        self.structure.add(object)                   

    def addEnemies(self, object):
        object.parent = self
        self.enemies.add(object) 
        
    def addPlayers(self, object):
        object.parent = self
        self.players.add(object)     
        
    def save_screen(self, file):
        pygame.image.save(self.screen, file)              
        
    def close(self):
        self.running = False
        pygame.quit()
        sys.exit()

    def do(self, event):
        if event.type == QUIT:
            self.running = False
            pygame.quit()
            sys.exit()
        elif event.type == VIDEORESIZE:            
            if not self.fullscreen:
                self.screen = pygame.display.set_mode((event.w, event.h), RESIZABLE)                    
        elif event.type == KEYDOWN:            
            if event.key == K_F2:
                if self.scale > 2:
                    self.scale = 1
                else:
                    self.scale += 1

                self.screen = pygame.display.set_mode((self.width*self.scale, self.height*self.scale),HWSURFACE|DOUBLEBUF|RESIZABLE)

            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == K_f:
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    self.screen = pygame.display.set_mode((self.screen.get_width(), self.screen.get_height()),FULLSCREEN)
                else:
                    self.screen = pygame.display.set_mode((self.screen.get_width(), self.screen.get_height()), RESIZABLE)

        for obj in self.background:
            obj.do(event)
        for obj in self.bases:
            obj.do(event)
        for obj in self.enemies:
            obj.do(event)
        for obj in self.structure:
            obj.do(event)                        
        for obj in self.players:
            obj.do(event)                                    
        for obj in self.front:
            obj.do(event)                            

    def update(self):
        if self.updating:            
            for obj in self.background:
                obj.update()            
            for obj in self.bases:
                obj.update()      
            for obj in self.enemies:
                obj.update()        
            for obj in self.structure:
                obj.update()
            for obj in self.players:
                obj.update()                
            for obj in self.front:
                obj.update()                   
        

    def draw(self):        
        for obj in self.background:            
            obj.draw(self.gamescreen)                    
        self.map.draw(self.gamescreen)
        for obj in self.bases:            
            obj.draw(self.gamescreen)
        for obj in self.enemies:            
            obj.draw(self.gamescreen)
        for obj in self.structure:            
            obj.draw(self.gamescreen)         
        for obj in self.players:            
            obj.draw(self.gamescreen)                           
        for obj in self.front:            
            obj.draw(self.gamescreen)                        

        self.draw_hud(self.gamescreen)

        pygame.display.update()           

    def check_collision(self, sprite1, sprite2):
        col = pygame.sprite.collide_rect(sprite1, sprite2)
        return col 

    def check_spritecollision(self, spr1, spr2):
        col = pygame.sprite.spritecollide(spr1, spr2, False)
        return col     

game = Game(w=256, h=256, tile_h=64, tile_w=64)
game.run()