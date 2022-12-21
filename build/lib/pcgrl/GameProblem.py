# -*- coding: utf-8 -*-
import pygame, sys
from gym.utils import seeding
from pygame import draw
from pygame.locals import *
from pcgrl.Utils import *
import json
import os

#Background color
BACKGROUND = (20, 20, 20)

class GameProblem():    
    def __init__(self, w = 960, h = 540, tile_w = 32, tile_h = 32, show_screen = True, render_game = True):        
        
        self.render_game = render_game
        
        self.width  = w
        self.height = h     
        self.tile_width  = tile_w
        self.tile_height = tile_h
        self.scale = 1   
        self.cols = int(w / tile_w)
        self.rows = int(h / tile_h)        
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        pygame.init()
        self.np_random  = None
        self.system_quit = True
        self.running  = True
        self.clock    = pygame.time.Clock()
        self.updating = True   
        self.env      = None
        #self.screen   = pygame.display.set_mode((self.width, self.height),HWSURFACE|DOUBLEBUF|RESIZABLE)                        
        self.gamescreen = None
        self.screen = None
        if show_screen:
            self.screen   = pygame.display.set_mode((round(self.width*self.scale), round(self.height*self.scale)))                        
            self.gamescreen = pygame.Surface((self.width, self.height))                  
        else:
            pygame.display.set_mode((1, 1))
            self.screen = pygame.Surface((self.width, self.height)).convert()  # we'll use gym to render. So just use a surface as the screen!       

        self.fullscreen = False     
        self.front      = pygame.sprite.Group([])
        self.enemies    = pygame.sprite.Group([])
        self.bases      = pygame.sprite.Group([])
        self.background = pygame.sprite.Group([])
        self.background_first = pygame.sprite.Group([])
        self.structure  = pygame.sprite.Group([])
        self.levelObjects  = pygame.sprite.Group([])
        self.players    = pygame.sprite.Group([])        
        self.fntDefault = pygame.font.SysFont('FFF Intelligent', 16)        
        self.blocked = True
        self.dic_tiles = {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]           
    
    def get_information(self):
        info = {"width" : self.get_width(), "height" : self.get_height(),
                "cols" : self.get_cols(), "rows" : self.get_rows(),
                "dim" : self.get_dim(), "state_height" : self.get_state_height(),
                "state_width" : self.get_state_width() }
        return info      

    def in_range(self, value, low, high):        
        return value >= low and value <= high        

    def border_offset(self):
        return (0, 0)

    def get_dim(self):
        return (self.get_width(), self.get_height() )        

    def noisy(self, val):
        between_val = val / 2.0
        if (val < between_val):
            return int(round(self.env.np_random.uniform(val, between_val)))
        else:
            return int(round(self.env.np_random.uniform(between_val, val)))            
    
    def create(self):    
        for obj in self.background_first:
            obj.create()                
        for obj in self.background:
            obj.create()                
        for obj in self.bases:
            obj.create()                
        for obj in self.enemies:
            obj.create()                
        for obj in self.structure:
            obj.create()                
        for obj in self.levelObjects:
            obj.create()                
        for obj in self.players:
            obj.create()                
        for obj in self.front:
            obj.create()                

    def clear_layers(self):     
        if self.render_game:   
            self.background.empty()                        
            self.bases.empty()                        
            self.front.empty()
            self.enemies.empty()
            self.structure.empty()
            self.levelObjects.empty()
            self.players.empty()

    def clear(self):           
        if self.render_game:        
            self.clear_layers()
            self.update()

    def remove_tile(self, x, y, group = None):
        if self.render_game:   
            if not self.blocked:
            
                state_w = self.get_state_width()
                state_h = self.get_state_height()

                rect = pygame.Rect(x, y,  state_w, state_h)
                aux = pygame.sprite.Sprite()
                aux.image = pygame.Surface((state_w, state_h))
                aux.rect = rect
                
                if (not group is None):
                    collide = pygame.sprite.spritecollide(aux, self.front, True)
                    collide = pygame.sprite.spritecollide(aux, self.bases, True)
                    collide = pygame.sprite.spritecollide(aux, self.background, True)
                    collide = pygame.sprite.spritecollide(aux, self.enemies, True)
                    collide = pygame.sprite.spritecollide(aux, self.structure, True)
                    collide = pygame.sprite.spritecollide(aux, self.levelObjects, True)
                    collide = pygame.sprite.spritecollide(aux, self.players, True)        
                else:
                    collide = pygame.sprite.spritecollide(aux, group, True)        
    
    """
    Get the current reward of current position

    Returns:
        float: the current reward 
    """    
    def compute_reward(self, new_stats, old_stats):        
        reward = 0.0
        rewards_info = { }
        return reward, rewards_info 
    
    def count_occurrences(self, map, elements):   
        
        array = np.array(map)
                
        unique, counts = np.unique(array, return_counts=True)
                
        search = np.array(elements)
                
        counts = [counts[i] for i, x in enumerate(unique) if x in search]
        
        return counts
    
    def counter_occurrences(self, map, id = None):
        _map = np.array(map)
        _map = list(_map.flatten())    
        _map = collections.Counter(_map)        
        if (id == None):
            return _map
        else:
            return _map[id]    

    def calc_regions(self, map, tile, ignore_values = None):
        """
        if (replaces_values is not None):
            _map = map.copy()            
            
            for tile in replaces_values:
                _map[_map == tile] = passable
                                 
            map = _map        
        """
        positions = array_el_locations(map)     
        
        map_flood_fill = map.copy()
                                                        
        num_regions = 0
   
        positions = positions[tile]
        
        for row, col in positions:             
            num_tiles_colored = flood_fill_v4(map_flood_fill, row, col, ignore_values)
            if (num_tiles_colored > 0):
                num_regions += 1
                                                                
        return num_regions    

    def calc_tiles(self, map, dic_tiles, tile):
        _map = np.array(map)
        _map = list(_map.flatten())    
        _map = collections.Counter(_map)        
        t = dic_tiles[tile]
        return _map[t]

    def get_tile_positions(self, tiles, map):        
        max_row = map.shape[0]
        max_col = map.shape[1]
        new_map = []
        for row in range(max_row):
            for col in range(max_col):
                id = int(map[row][col])
                if id in tiles:
                    new_map.append((row, col))
        return new_map

    def get_width(self):
        return self.screen.get_width()

    def get_height(self):
        return self.screen.get_height()        

    def get_state_width(self):
        return self.tile_width
    
    def get_state_height(self):
        return self.tile_height

    def get_rows(self):
        return self.rows #int(self.get_height() / self.get_state_height())
    
    def get_cols(self):
        return self.cols #int(self.get_width() / self.get_state_width())

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
        
        if self.render_game:           
            for event in pygame.event.get():
                self.do(event) 

            if (mode == "human"):  
                self.gamescreen.fill(BACKGROUND)           
                
                self.update()        
                self.draw()
                scaled_win = pygame.transform.scale(self.gamescreen, self.screen.get_size())
                #scaled_win = pygame.transform.smoothscale(self.gamescreen, self.screen.get_size())
                self.screen.blit(scaled_win, (0, 0))        
                # Refresh Screen            
                pygame.display.flip()                
                # Create the PixelArray.                
                ar = pygame.surfarray.array3d(self.gamescreen).swapaxes(0, 1)
                self.clock.tick(tick)             
            else:
                self.update()
                ar = pygame.surfarray.array3d(self.gamescreen).swapaxes(0, 1)  # swap because pygame                                         
            
        return ar

    def addBackground_first(self, object):
        if self.render_game:
            object.parent = self
            self.background_first.add(object)

    def addBackground(self, object):
        if self.render_game:
            object.parent = self
            self.background.add(object)

    def addFront(self, object):
        if self.render_game:
            object.parent = self
            self.front.add(object)
    
    def addBases(self, object):        
        if self.render_game:
            object.parent = self
            self.bases.add(object)           
        
    def addStructure(self, object):        
        if self.render_game:
            object.parent = self
            self.structure.add(object)                   

    def addEnemies(self, object):
        if self.render_game:
            object.parent = self
            self.enemies.add(object) 

    def addLevelObjects(self, object):
        if self.render_game:
            object.parent = self
            self.levelObjects.add(object)         

    def addPlayers(self, object):
        if self.render_game:
            object.parent = self
            self.players.add(object)     
        
    def save_screen(self, file):
        if self.render_game:
            pygame.image.save(self.screen, file)              
        
    def close(self):
        self.running = False
        pygame.quit()
        sys.exit()

    def quit(self):
        self.running = False
        pygame.quit()        

    def do(self, event):
        if self.render_game:        
            quit = False
            
            if event.type == QUIT:
                self.running = False
                pygame.quit()
                if self.system_quit:
                    sys.exit()
            elif event.type == VIDEORESIZE:            
                if not self.fullscreen:
                    self.screen = pygame.display.set_mode((event.w*self.scale, event.h*self.scale), HWSURFACE|DOUBLEBUF|RESIZABLE)                    
            elif event.type == KEYDOWN:            
                if event.key == K_F2:
                    if self.scale > 2:
                        self.scale = 1
                    else:
                        self.scale += 1

                    self.screen = pygame.display.set_mode((self.width*self.scale, self.height*self.scale),HWSURFACE|DOUBLEBUF|RESIZABLE)

                if event.key == K_ESCAPE:
                    pygame.quit()
                    if self.system_quit:
                        sys.exit()
                if event.key == K_f:
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        self.screen = pygame.display.set_mode((self.screen.get_width(), self.screen.get_height()),FULLSCREEN)
                    else:
                        self.screen = pygame.display.set_mode((self.screen.get_width()*self.scale, self.screen.get_height()*self.scale), HWSURFACE|DOUBLEBUF|RESIZABLE)

            for obj in self.background_first:
                obj.do(event)
            for obj in self.background:
                obj.do(event)
            for obj in self.bases:
                obj.do(event)
            for obj in self.enemies:
                obj.do(event)
            for obj in self.structure:
                obj.do(event)
            for obj in self.levelObjects:
                obj.do(event)                                    
            for obj in self.players:
                obj.do(event)
            for obj in self.front:
                obj.do(event)                            

    def update(self):
        if self.render_game:
            if self.updating:
                for obj in self.background_first:
                    obj.update()                
                for obj in self.background:
                    obj.update()
                for obj in self.bases:
                    obj.update()      
                for obj in self.enemies:
                    obj.update()        
                for obj in self.structure:
                    obj.update()
                for obj in self.levelObjects:
                    obj.update()                
                for obj in self.players:
                    obj.update()                
                for obj in self.front:
                    obj.update()                                          

    def draw(self):        
        if self.render_game:
            for obj in self.background_first:            
                obj.draw(self.gamescreen)                    
            for obj in self.background:            
                obj.draw(self.gamescreen)        
            for obj in self.bases:            
                obj.draw(self.gamescreen)
            for obj in self.enemies:            
                obj.draw(self.gamescreen)
            for obj in self.structure:            
                obj.draw(self.gamescreen)         
            for obj in self.levelObjects:            
                obj.draw(self.gamescreen)                     
            for obj in self.players:            
                obj.draw(self.gamescreen)                           
            for obj in self.front:            
                obj.draw(self.gamescreen)                        

            self.draw_hud(self.gamescreen)

            pygame.display.update()           
    
    def create_tile(self, tile):
        pass

    def draw_rect_alpha(surface, color, rect):        
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        #surface.blit(shape_surf, rect)
        
    def draw_rect_tiles(self, surface):
        if self.render_game:        
            background = pygame.Surface(surface.get_size())
            ts, w, h, c1, c2 = 50, *surface.get_size(), (160, 160, 160), (192, 192, 192)
            tiles = [((x*ts, y*ts, ts, ts), c1 if (x+y) % 2 == 0 else c2) for x in range((w+ts-1)//ts) for y in range((h+ts-1)//ts)]
            for rect, color in tiles:
                pygame.draw.rect(background, color, rect)        
    
    def check_collision(self, sprite1, sprite2):
        if self.render_game:        
            col = pygame.sprite.collide_rect(sprite1, sprite2)
            return col 
        return None

    def check_spritecollision(self, spr1, spr2):
        if self.render_game:        
            col = pygame.sprite.spritecollide(spr1, spr2, False)
            return col
        return None

    def range_reward(self, new_value, old_value, low, high, reward, weight = 1):
        
        if new_value >= low and new_value <= high and old_value >= low and old_value <= high:        
            return 0
                                
        if (self.in_range(new_value, low , high) and old_value > high):        
            return (1 + (old_value - high) + new_value) * weight
        
        if (self.in_range(new_value, low , high) and old_value < low):        
            return (1 + (low - old_value) + new_value) * weight
                
        if (self.in_range(old_value, low ,  high) and new_value < low):        
            mi = min(new_value, old_value)
            mx = max(new_value, old_value)
            return (mi - mx) * weight
            
        if (self.in_range(old_value, low ,  high) and new_value > high):        
            mi = min(new_value, old_value)
            mx = max(new_value, old_value)
            return (mi - mx) * weight
                
        if (not self.in_range(old_value, low, high) and not self.in_range(new_value, low, high)):
            mi = min(new_value, old_value)
            mx = max(new_value, old_value)
            r = mi - mx
            if (mx == mi):
                r = low - (new_value + old_value) - high - reward 
            return r * weight
                                    
        return -reward * weight                    