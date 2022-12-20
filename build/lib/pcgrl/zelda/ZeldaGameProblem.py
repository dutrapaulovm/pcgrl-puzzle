# -*- coding: utf-8 -*-
from itertools import count
from tkinter import E
import pygame, sys
import numpy as np
import pandas as pd

from pcgrl.zelda import *
from pcgrl.zelda.ZeldaLevelObjects import *
from pcgrl.GameProblem import GameProblem
from pcgrl.Utils import *
from pygame import draw
from pygame import font
from pygame.locals import *
import csv

class ZeldaGameProblem(GameProblem):    

    def __init__(self, rows = 0, cols = 0, border = False):

        self.border = border
        self.tile_size = 64

        offset_border = 0
        
        if (self.border):
            offset_border = 2

        cols = cols + offset_border
        rows = rows + offset_border

        if cols > 0 and rows > 0:            
            self.width  = cols * self.tile_size
            self.height = rows * self.tile_size
        else:            
            self.width = 16 * self.tile_size
            self.height = 8 * self.tile_size        
        
        super().__init__(w = self.width, h = self.height, tile_w = self.tile_size, tile_h = self.tile_size)            

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
        self._range_coins      = {"min" : 4, "max" : 10}
        self._range_enemies      = {"min" : 2, "max" : 6}
        self.show_hud = False

        self.neighbors = 0
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):
                y = row * self.get_state_height()
                x = col * self.get_state_width()
                ground = Ground(x, y)
                self.addBackground(ground)
                        
        self.tiles = ["Ground", "Block", "DoorEntrance", "DoorExit", "Coin", "Key", "Player", "Enemy", "Weapon"]        
        self.dic_tiles = convert_strarray_to_dic(self.tiles)
        """
        self.dic_tiles = {}
        for i in range(len(self.tiles)):
            tile = self.tiles[i]
            self.dic_tiles[tile] = i
        """
    def border_offset(self):
        return (1, 1)

    def get_info(self):
        params = { }
        return params
        
    def do(self, event):
        super().do(event)
        if event.type == KEYDOWN:
            if event.key == K_F12:
                self.reset(self.np_random)
                    
    def step(self, entity):                                
        reward = 0.0                        
        return reward

    def reset(self, np_random = None):
        self.np_random = np_random
        self.clear_layers()
        self.generate_map(np_random)                
        self.update()                
    
    def get_tile_name(self, id):
        return self.tiles[id]

    def get_tiles(self):
        return self.tiles        

    def place_objects(self, obj_id, num_objs):
        tiles = [Ground.ID]
        map_locations = self.get_tile_positions(tiles, self.map)
        for j in range(num_objs):            
            index = self.np_random.randint(len(map_locations))            
            position = map_locations[index]
            row = position[0]
            col = position[1]

            while (self.map[row][col] == obj_id):
                index = self.np_random.randint(len(map_locations))
                position = map_locations[index]
                row = position[0]
                col = position[1]

            self.change_tile(col * self.get_state_width(), row * self.get_state_height(), obj_id)            
    
    def load_map(self, path_map):   
        data = []
        with open(path_map, newline='') as csvfile:
            data = list(csv.reader(csvfile))            
            
        data = np.array(data).astype("int") 
        
        self.map = data
        self.__create()
        
    def render_map(self):
        self.__create()
        self.render(tick=0)            
        
    def create_map(self, data):    
        self.map = data
        if not self.blocked:
            self.__create()  
            
    def update_map(self):
        self.clear()
        self.__create() 

    def generate_map(self, random = None):                  
        border = 0    
        
        if self.border:
            border = self.border_offset()[0] + self.border_offset()[1] 

        self.map = np.zeros((self.get_rows()-border, self.get_cols()-border))
        self.map = np.array(self.map).astype("int") 
        if self.border:
            self.map = fast_pad(self.map, 1)               
        self.__create()

    def change_tile(self, x, y, val):
        if not self.blocked:
            col = int(x / self.get_state_width())
            row = int(y / self.get_state_height())
            
            state_w = self.get_state_width()
            state_h = self.get_state_height()

            rect = pygame.Rect(x, y,  state_w, state_h)
            aux = pygame.sprite.Sprite()
            aux.image = pygame.Surface((state_w, state_h))
            aux.rect = rect

            collide = pygame.sprite.spritecollide(aux, self.bases, True)
            collide = pygame.sprite.spritecollide(aux, self.background, True)
            collide = pygame.sprite.spritecollide(aux, self.enemies, True)
            collide = pygame.sprite.spritecollide(aux, self.structure, True)
            collide = pygame.sprite.spritecollide(aux, self.levelObjects, True)
            collide = pygame.sprite.spritecollide(aux, self.players, True)
                    
            tile = self.create_tile(val, x = col * state_w, y = row * state_h)                         

        self.map[row, col] = val
               
    def get_tiles(self):
        return self.tiles

    def get_map_stats(self):
     
        map_stats = {
            "Ground"      : self.calc_tiles(self.map, self.dic_tiles,  "Ground"),
            "Block"      : self.calc_tiles(self.map, self.dic_tiles,  "Block"),
            "DoorEntrance"   : self.calc_tiles(self.map, self.dic_tiles, "DoorEntrance"),
            "DoorExit"   : self.calc_tiles(self.map, self.dic_tiles, "DoorExit"),
            "Coin"   : self.calc_tiles(self.map, self.dic_tiles, "Coin"),
            "Key"   : self.calc_tiles(self.map, self.dic_tiles, "Key"),
            "Player"   : self.calc_tiles(self.map, self.dic_tiles, "Player"),
            "Enemy"   : self.calc_tiles(self.map, self.dic_tiles, "Enemy"),
            "Weapon"   : self.calc_tiles(self.map, self.dic_tiles, "Weapon"),
            "regions_ground" : self.calc_regions(self.map, Ground.ID, [Ground.ID, Player.ID, Enemy.ID, Coin.ID, Key.ID, Weapon.ID] )
        }         
  
        return map_stats                  

    def compute_reward(self, new_stats, old_stats):        
        reward = 0.0
        agent = self.env.agent              
        #if DoorEntrance.ID in agent.pieces[0] and DoorExit.ID in agent.pieces[agent.total_board_pieces-1]:
        #    reward = 5

        map_stats = new_stats["map_stats"] 
        old_map_stats = old_stats["map_stats"] 
        door_entrance = map_stats["DoorEntrance"]                
        coins = map_stats["Coin"]
        key = map_stats["Key"]
        weapon = map_stats["Weapon"]
        enemy = map_stats["Enemy"]        
        player = map_stats["Player"]
        door_exit = map_stats["DoorExit"]                
        regions_ground = map_stats["regions_ground"]    

        reward += self.range_reward(regions_ground, old_map_stats["regions_ground"], 1, 1, 1)
        reward += self.range_reward(player, old_map_stats["Player"], 1, 1, 2)        
        reward += self.range_reward(key, old_map_stats["Key"], 1, 1, 2)
        reward += self.range_reward(weapon, old_map_stats["Weapon"], 1, 1, 2)
        reward += self.range_reward(enemy, old_map_stats["Enemy"],  self._range_enemies["min"],  self._range_enemies["max"], 1)
        reward += self.range_reward(coins, old_map_stats["Coin"],  self._range_coins["min"],  self._range_coins["max"], 1)
        reward += self.range_reward(door_exit, old_map_stats["DoorExit"], 1, 1, 3)
        reward += self.range_reward(door_entrance, old_map_stats["DoorEntrance"], 1, 1, 3)        

        rewards_info = { }
        return reward, rewards_info   

    def is_done(self, stats):
        
        """
        Check if problem is over. This method test if problem satisfying quality based on current stats
        """    
        map_stats = stats["map_stats"] 
        door_entrance = map_stats["DoorEntrance"]                
        coins = map_stats["Coin"]
        key = map_stats["Key"]
        enemy = map_stats["Enemy"]
        weapon = map_stats["Weapon"]        
        player = map_stats["Player"]
        door_exit = map_stats["DoorExit"]                
        regions_ground = map_stats["regions_ground"]
        
        done = regions_ground == 1 and \
            door_exit == 1 and door_entrance == 1 and \
            self.in_range(coins, self._range_coins["min"], self._range_coins["max"]) and \
            self.in_range(enemy, self._range_enemies["min"], self._range_enemies["max"]) and \
            key == 1 and player == 1 and \
            weapon == 1

        return done  

    def is_neighbors(self, tile, tile_neighbors):
        positions = self.get_tile_positions([tile], self.map)
        for row, col in positions:                    
            n = neighbors(row, col)            
            for row, col in n:
                if (self.map[row][col] == tile_neighbors):                    
                    return True;                   
        return False

    def counter_neighbors(self, tile, tiles_neighbors):
        positions = self.get_tile_positions(tile, self.map)
        counter = 0
        for row, col in positions:                    
            n = neighbors(row, col)                        
            for row, col in n:
                if (row >= 0 and row < self.map.shape[0] and \
                    col >= 0 and col < self.map.shape[1] and \
                    self.map[row][col] in tiles_neighbors):                    
                    counter += 1

        return counter                    
    
    def create_tile(self, tile, x, y):        
        val = tile
        if val == Ground.ID:
            tile  = Ground(id =Ground.ID, x = x, y = y)                    
            self.addBackground(tile)                   
        elif val == Block.ID:
            tile  = Block(id =Block.ID, x = x, y = y)                    
            self.addBases(tile)                                            
        elif val == DoorEntrance.ID:
            tile  = DoorEntrance(id = DoorEntrance.ID, x = x, y = y)                    
            self.addBases(tile)
        elif val == DoorExit.ID:
            tile  = DoorExit(id = DoorExit.ID, x = x, y = y)                    
            self.addBases(tile)                   
        elif val == Coin.ID:
            tile  = Coin(id = Coin.ID, x = x, y = y)                    
            self.addLevelObjects(tile)                        
        elif val == Key.ID:
            tile  = Key(id = Key.ID, x = x, y = y)                    
            self.addLevelObjects(tile) 
        elif val == Player.ID:
            tile  = Player(id = Player.ID, x = x, y = y)                    
            self.addPlayers(tile)  
        elif val == Enemy.ID:
            tile  = Enemy(id = Enemy.ID, x = x, y = y)                    
            self.addPlayers(tile)            
        elif val == Weapon.ID:
            tile  = Weapon(id = Weapon.ID, x = x, y = y)                    
            self.addLevelObjects(tile)        
        else:
            assert False, "unknown tile in decode '%s'" % tile

        return tile

    def __create(self):
        
        state_w = self.get_state_width()
        state_h = self.get_state_height()
        
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):                
                val = self.map[row, col]
                tile = self.create_tile(val, x = col * state_w, y = row * state_h)

    def draw_hud(self, screen):
        space_line    = 32
        current_line  = 0
        current_line += space_line
        map_stats = self.get_map_stats()
        if (not self.env is None) and self.show_hud:
            text = "Rewards: " + str(self.env._reward)
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)
            current_line += space_line
            text = "Changes: " + str(self.env.counter_changes)
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)            
            current_line += space_line
            text = "Tiles Grounds: " + str(map_stats["Ground"])
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        
            current_line += space_line
            text = "Tiles Blocks: " + str(map_stats["Block"])
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        
            current_line += space_line
            text = "Tile DoorEntrance: " + str(map_stats["DoorEntrance"])
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        
            current_line += space_line
            text = "Tile DoorExit: " + str(map_stats["DoorExit"])
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                                    
            current_line += space_line
            text = "Regions ground: " + str(map_stats["regions_ground"])
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        
            current_line += space_line
            text = "Max segments: " + str(self.env.max_segment)
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        
            current_line += space_line
            text = "Segments: " + str(self.env.agent.grid)
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        
            current_line += space_line
            text = "Rows: {}, Cols: {} ".format(self.get_rows(), self.get_cols())
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        
            current_line += space_line            
            text = "Entropy {}".format(entropy(self.env.agent.grid))
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)          