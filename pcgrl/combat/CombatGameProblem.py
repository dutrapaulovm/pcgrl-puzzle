# -*- coding: utf-8 -*-
import pygame, sys
import numpy as np
import pandas as pd

from pcgrl.combat import *
from pcgrl.combat.CombatLevelObjects import *
from pcgrl.GameProblem import GameProblem
from pcgrl.Utils import *
from pygame import draw
from pygame import font
from pygame.locals import *
import csv

class CombatGameProblem(GameProblem):    
    def __init__(self, width = 512, height = 512, rows = 0, cols = 0, border = False):
        
        self.border = border

        offset_border = 0
        
        if (self.border):
            offset_border = 2

        cols = cols + offset_border
        rows = rows + offset_border
        self.cols = cols
        self.rows = rows

        if cols > 0 and rows > 0:            
            width  = cols * 64
            height = rows * 64
        else:            
            width = 16 * 64
            height = 8 * 64
                            
        super().__init__(w = width, h = height, tile_w = 64, tile_h = 64)            

        self.fntHUD      = pygame.font.Font('freesansbold.ttf', 24)     
        self.fntSmallHUD = pygame.font.Font('freesansbold.ttf', 16)
        self.action      = 0   
        self.action_leveldesigner = 0
        self.tiles             = {}
        self.tile              = 0
        self.reward            = 0
        self.tilesConnections  = ""
        self.right             = 0
        self.left              = 0
        self.up                = 0
        self.down              = 0
        self.state             = 0
        self.gen_map           = False
        self.showinfo          = False
        self.leveldesigners    = {}        
        self._range_blocks      = np.array([200, int( (self.get_cols() * self.get_rows()))])        
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):
                y = row * self.get_state_height()
                x = col * self.get_state_width()
                ground = Ground(x, y)
                self.addBackground(ground)         
        
        self.tiles = ["Ground", "Block"]                
        
        self.rewards = {
            "Ground"                 : 0,
            "Block"                  : 1,            
            "regions_ground"         : 1
        }

        Ground.ID = 0
        Block.ID = 1

    def border_offset(self):
        return (1, 1)

    def get_info(self):
        params = { "block_min" : self._range_blocks[0], "block_max" : self._range_blocks[1] }
        return params
        
    def do(self, event):
        super().do(event)
        if event.type == KEYDOWN:
            if event.key == K_F12:
                self.reset(self.np_random)
                    
    def step(self, entity):                                
        reward = 0.0                        
        return reward        

    def reset(self, np_random):
        self.np_random = np_random
        self.bases.empty()                        
        self.front.empty()
        self.enemies.empty()
        self.structure.empty()                
        self.generate_map(np_random)
        self.update()

    def clear(self):        
        self.background.empty()                        
        self.bases.empty()                        
        self.front.empty()
        self.enemies.empty()
        self.structure.empty()
        self.update()
    
    def get_tile_name(self, id):
        return self.tiles[id]

    def get_tiles(self):
        return self.tiles        
    
    def load_map(self, path_map):   
        data = []
        with open(path_map, newline='') as csvfile:
            data = list(csv.reader(csvfile))            
            
        data = np.array(data).astype("int") 
        
        self.map = data
        self.__create()    
        
    def create_map(self, data):    
        #for row in range(self.map.shape[0]):
        #    for col in range(self.map.shape[1]):
        #        self.remove_tile(col * 64, row * 64, self.map[row, col])

        self.map = data
        self.__create()            

    def generate_map(self, random):          
        border = 0    
        if self.border:
            border = 2

        self.map = np.zeros((self.get_rows()-border, self.get_cols()-border))
        self.map = np.array(self.map).astype("int") 
        self.map = fast_pad(self.map, 1)               
        self.cols = self.map.shape[1] 
        self.rows = self.map.shape[0]
        print(self.map)
        print(self.map.shape)
        print(self.cols, self.rows)
        self.__create()


    def change_tile(self, x, y, val):
        
        col = int(x / self.get_state_width())
        row = int(y / self.get_state_height())
        
        state_w = self.get_state_width()
        state_h = self.get_state_height()

        rect = pygame.Rect(x, y,  state_w, state_h)
        aux = pygame.sprite.Sprite()
        aux.image = pygame.Surface((state_w, state_h))
        aux.rect = rect
                                    
        collide = pygame.sprite.spritecollide(aux, self.bases, True)
        collide = pygame.sprite.spritecollide(aux, self.structure, True)
                
        tile = 0

        if val == Ground.ID:
            tile  = Ground(id = Ground.ID, x = col * state_w, y = row * state_h)                    
            self.addBackground(tile)            
        if val == Block.ID:
            tile  = Block(id = Block.ID, x = col * state_w, y = row * state_h)                    
            self.addBases(tile)        

        self.map[row, col] = val
               
    def get_tiles(self):
        return self.tiles    

    def get_map_stats(self):
        map = self.convert_map_to_string(self.get_tiles(), self.map)        
        map_locations = self.get_tile_locations(map, self.get_tiles())                
        map_stats = {
            "Ground"      : self.calc_certain_tile(map_locations,  ["Ground"]),
            "Block"      : self.calc_certain_tile(map_locations,  ["Block"]),            
            "regions_ground"    : self.calc_num_regions(map, map_locations, ["Ground"])
        }
        return map_stats

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
    
    def update_rewards(self, scale = 0.02):        
        for key, rew in self.rewards.items():
            self.rewards[key] = self.rewards[key] * (1 + scale)           

    """
    Get the current reward of current position

    Returns:
        float: the current reward 
    """
    def compute_reward(self, new_stats, old_stats):
        
        reward = 0.0
        
        new_stats = new_stats["map_stats"]
        old_stats = old_stats["map_stats"]                               
        
        reward_block = self.range_reward(new_stats["Block"], old_stats["Block"], self._range_blocks[0], self._range_blocks[1], self.rewards["Block"])
        
        reward_regions_ground = self.range_reward(new_stats["regions_ground"], old_stats["regions_ground"], 1, 1, self.rewards["regions_ground"])

        rewards = {                        
            "Block" : reward_block,
            "regions_ground": reward_regions_ground
        }

        for key, rew in rewards.items():            
            r = rew * self.rewards[key]            
            reward += r            

        rewards_info = {                        
            "Block": reward_block,
            "regions_ground": reward_regions_ground
        }

        return reward, rewards_info           

    def get_positions(self, tiles):
        tiles = ["Gound", "Block"]        
        map = self.convert_map_to_string(tiles, self.map)
        map_locations = self.get_tile_locations(map, self.get_tiles())
        positions = self.get_certain_tiles(map_locations, self.get_tiles())
        
        return positions

    def is_done(self, stats):
        """
        Check if problem is over. This method test if problem satisfying quality based on current stats
        """    
        map_stats = stats["map_stats"] 
        regions_ground = map_stats["regions_ground"]                
        
        block = self.in_range(map_stats["Block"], self._range_blocks[0], self._range_blocks[1])
           
        done = block and regions_ground == 1
        return done  
            
    def __create(self):
        
        state_w = self.get_state_width()
        state_h = self.get_state_height()
        
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):                
                val = self.map[row, col]                                
                if val == Ground.ID:
                    tile  = Ground(id =Ground.ID, x = col * state_w, y = row * state_h)                    
                    self.addBackground(tile)                   
                if val == Block.ID:
                    tile  = Block(id =Block.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)                                            

    def draw(self):
        super().draw()
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):
                y = row * self.get_state_height()
                x = col * self.get_state_width()
                #state = encodedXY(x, y, self.get_cols(), self.get_width(), self.get_state_width(), self.get_state_height())
                #self.draw_text(x, y, str(state))
                #self.draw_text(x, y, str(self.map[row, col]))

    def draw_hud(self, screen):
        space_line    = 32
        current_line  = 0
        current_line += space_line
        if (not self.env is None):
            text = "Rewards: " + str(self.env.current_reward)
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)