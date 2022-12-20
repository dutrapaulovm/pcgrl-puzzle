# -*- coding: utf-8 -*-
import pygame, sys
import numpy as np
import pandas as pd

from pcgrl.Utils import *
from pcgrl.minimap import *
from pcgrl.minimap.MiniMapLevelObjects import *
from pcgrl.GameProblem import GameProblem
from pygame import draw
from pygame import font
from pygame.locals import *
import csv
    
class InfoRewards:
    def __init__(self):
        self.pos = pygame.Vector2(0,0)
        self.tile = 0
        self.reward = -1        
        self.direction = 0

class MiniMapGameProblem(GameProblem): 
               
    def __init__(self, width = 512, height = 512, rows = 0, cols = 0):

        Grass.ID = 0
        Trees.ID = 1
        RockGold.ID = 2
        Rock.ID = 3
        House1.ID = 4        
        Person.ID = 5
        Fruit.ID = 6
        Warehouse.ID = 7                   
        
        self.tile_size = 64
        
        if cols > 0 and rows > 0:
            width = cols * self.tile_size
            height = rows * self.tile_size
        else:            
            width = 16 * self.tile_size
            height = 8 * self.tile_size

        self.cols = cols
        self.rows = rows
                                        
        super().__init__(w = width, h = height, tile_w = 64, tile_h = 64)
        
        if self.render_game:
            for row in range(self.get_rows()):
                for col in range(self.get_cols()):
                    y = row * self.get_state_height()
                    x = col * self.get_state_width()
                    grass = Grass(x, y)
                    self.addBackground_first(grass)        
                            
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
        self._range_fruit      = np.array([4, 8])
        self._range_rocks      = np.array([6, 10])        
        self._range_rockgolds  = np.array([6, 10])
        self._range_houses     = np.array([4, 20])
        self._range_trees      = np.array([10, int( (self.get_cols() * self.get_rows()) * 0.80)])
        self._range_person     = np.array([4, 8])
        self._range_ware_house = np.array([1, 2])
        self._max_regions      = 20
        self._min_objects      = 21
        self.show_hud = False
        
        self.tiles = ["Grass", "Trees", "RockGold", "Rock", "House1", "Person", "Fruit", "Warehouse"]                       
        
                
    def get_info(self):
        params = { "rock_min" : self._range_rocks[0], "rock_max" : self._range_rocks[1],
                   "rockgolds_min" : self._range_rockgolds[0], "rockgold_max" : self._range_rockgolds[1],
                   "tree_min" : self._range_trees[0], "tree_max" : self._range_trees[1],
                   "person_min" : self._range_person[0], "person_max" : self._range_person[1],
                   "fruit_min" : self._range_fruit[0], "fruit_max" : self._range_fruit[1],
                   "warehouse_min" : self._range_ware_house[0], "warehouse_max" : self._range_ware_house[1]
                 }
        return params
    
    def border_offset(self):
        return (0, 0)

    def do(self, event):
        super().do(event)
        if event.type == KEYDOWN:
            if event.key == K_F12:
                self.reset(self.np_random)
                    
    def step(self, entity):
                        
        reward = -1
                        
        if (pygame.sprite.spritecollide(entity, self.bases, False)):
            reward = -600
                        
        if (pygame.sprite.spritecollide(entity, self.structure, False)):
            reward = -600
        
        tile = None            
        hit = pygame.sprite.groupcollide(self.players, self.front, False, False)
        
        if hit:
            for base in hit:                
                tile = base
                reward = 100
                #print("Tile collisions. %i "%(tile.id))
                break
                        
        return reward   
    
    def update_map(self):
        self.clear()
        self.__create()          

    def reset(self, np_random):        
        self.np_random = np_random
        if self.render_game:
            self.bases.empty()                        
            self.front.empty()
            self.enemies.empty()
            self.structure.empty()                
            self.generate_map(np_random)
            self.update()    
    
    def get_tile_name(self, id):
        return self.tiles[id]  

    def clear(self):                
        self.clear_layers()
        self.update()    
    
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

    def generate_map(self, random):   
        self.map = np.zeros((self.get_rows(), self.get_cols()))        
        self.map = np.array(self.map).astype("int") 
        self.__create()
                            
    def get_tiles(self):
        return self.tiles

    def get_map_stats(self):
        map = self.convert_map_to_string(self.get_tiles(), self.map)        
        map_locations = self.get_tile_locations(map, self.get_tiles())                
        map_stats = {
            "Grass"      : self.calc_certain_tile(map_locations,  ["Grass"]),
            "Trees"      : self.calc_certain_tile(map_locations,  ["Trees"]),
            "RockGold"   : self.calc_certain_tile(map_locations,  ["RockGold"]),
            "Rock"       : self.calc_certain_tile(map_locations,  ["Rock"]),
            "House1"     : self.calc_certain_tile(map_locations,  ["House1"]),
            "Person"     : self.calc_certain_tile(map_locations,  ["Person"]),
            "Fruit"      : self.calc_certain_tile(map_locations,  ["Fruit"]),
            "Warehouse"  : self.calc_certain_tile(map_locations,  ["Warehouse"]),
            "regions_person"    : self.calc_num_regions(map, map_locations, ["Person"]),
            "regions_grass"    : self.calc_num_regions(map, map_locations, ["Grass"]),
            "regions_trees"    : self.calc_num_regions(map, map_locations, ["Trees"]),
            "regions_house"    : self.calc_num_regions(map, map_locations, ["House1"]),
            "regions_rock"     : self.calc_num_regions(map, map_locations, ["Rock"]),
            "regions_rockgold" : self.calc_num_regions(map, map_locations, ["RockGold"])
        }
        return map_stats       

    def in_range(self, value, low, high):        
        return value >= low and value <= high            

    def _calc_rewardsV2(self, tile, row, col, connections, r = 1):                

        max_x = self.map.shape[1]
        max_y = self.map.shape[0]                                      
        
        left  = max(col - 1, 0)
        right = min(col + 1, max_x-1)        
        up    = max(row - 1, 0)
        down  = min(row + 1, max_y-1)          
        
        self.tile = tile
        
        reward       = 0
        count_chunks = 0
                                        
        infoRewards = []
        for dir in connections:
            con = connections[dir]
            for c in con:
                info = InfoRewards()
                info.reward = -1
                if (dir == "left"):
                    if self.map[row, left] == c  or (left <= 0 and self.map[row, left] == c):
                        count_chunks += 1                        
                        info.pos = pygame.Vector2(left, row)
                        info.tile = self.map[row, left]
                        info.reward = r                        
                elif (dir == "right"):
                    if self.map[row, right] == c  or (right >= self.map.shape[1] and self.map[row, right] == c):
                        count_chunks += 1 
                        info.pos = pygame.Vector2(right, row)
                        info.tile = self.map[row, right]
                        info.reward = r                                                
                elif (dir == "up"):                                        
                    if self.map[up, col] == c  or (up <= 0 and self.map[up, col] == c):                 
                        count_chunks += 1
                        info.pos = pygame.Vector2(col, up)
                        info.tile = self.map[up, col]
                        info.reward = r                       
                elif (dir == "down"):
                    if self.map[down, col] == c  or (down >= self.map.shape[0] and self.map[down, col] == c):
                        count_chunks += 1
                        info = InfoRewards()
                        info.pos = pygame.Vector2(col, down)
                        info.tile = self.map[down, col]
                        info.reward = r
                info.direction = (info.tile + (4 * info.tile) - (info.tile-1) + 0) - 1
                infoRewards.append(info)
                                                                        
        if count_chunks >= 4:
            reward = 1
        else:
            reward = -3
                                                        
        return reward
    
    def _calc_rewards(self, connections, states = None, r = 1):        
      
        tile  = states["tile"]   
        row   = states["row"]       
        col   = states["col"]       
        
        right = states["right"]
        left  = states["left"]
        up    = states["up"]
        down  = states["down"]
        tile  = states["tile"]
        
        self.tile = tile
        
        reward       = 0
        count_chunks = 0
                        
        infoRewards = []
        for dir in connections:
            con = connections[dir]
            for c in con:
                info = InfoRewards()
                info.reward = -1
                if (dir == "left"):
                    if self.map[row, left] == c or left <= 0:
                        count_chunks += 1                        
                        info.pos = pygame.Vector2(left, row)
                        info.tile = self.map[row, left]
                        info.reward = r                        
                elif (dir == "right"):
                    if self.map[row, right] == c or right >= self.map.shape[1]:
                        count_chunks += 1 
                        info.pos = pygame.Vector2(right, row)
                        info.tile = self.map[row, right]
                        info.reward = r                                                
                elif (dir == "up"):                                        
                    if self.map[up, col] == c or up <= 0:                  
                        count_chunks += 1
                        info.pos = pygame.Vector2(col, up)
                        info.tile = self.map[up, col]
                        info.reward = r                       
                elif (dir == "down"):
                    if self.map[down, col] == c or down >= self.map.shape[0]:
                        count_chunks += 1
                        info = InfoRewards()
                        info.pos = pygame.Vector2(col, down)
                        info.tile = self.map[down, col]
                        info.reward = r
                info.direction = (info.tile + (4 * info.tile) - (info.tile-1) + 0) - 1
                infoRewards.append(info)
                                                        
        if count_chunks >= 4:
            reward = 1
        else:
            reward = -500 
                                                
        return reward, infoRewards

    def is_connect(self):
        tiles = ["Grass", "Trees", "RockGold", "Rock", "House1", "Person", "Fruit", "Warehouse"]        
        map = self.convert_map_to_string(tiles, self.map)
        map_locations = self.get_tile_locations(map, self.get_tiles())
        positions = self.get_certain_tiles(map_locations, self.get_tiles())
        
        for (col, row) in positions:
            tile = self.map[row, col]            
            reward = self.compute_rewards_by_tileV2(tile, row, col)
            if self.map[row, col] > 0:
                if (reward < 0):
                    return False                                
        return True

    def get_positions(self, tiles):
        tiles = ["Grass", "Trees", "RockGold", "Rock", "House1", "Person", "Fruit", "Warehouse"]        
        map = self.convert_map_to_string(tiles, self.map)
        map_locations = self.get_tile_locations(map, self.get_tiles())
        positions = self.get_certain_tiles(map_locations, self.get_tiles())        
        return positions
        
    def compute_rewards_by_tileV2(self, tile, row, col):
        
        reward = 0
        
        if (tile == Grass.ID):
                        
            reward = 0
            #self.env._rewardsmap[row, col] = 0
             
        if (tile == Fruit.ID):
            
            reward = 1
            #self.env._rewardsmap[row, col] = 4
                
        elif (tile == Person.ID): 
            
            left  = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
            right = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
            up    = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
            down  = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
                                    
            connections = {
                "left"  : left,
                "right" : right,
                "up"    : up,
                "down"  : down
            }
                                
            reward = self._calc_rewardsV2(tile, row, col, connections)
                                            
        elif (tile == Castle.ID): 
            
            left  = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
            right = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
            up    = [Trees.ID, Grass.ID, Person.ID, Fruit.ID]
            down  = [Grass.ID, Person.ID, Fruit.ID]

            connections = {
                "left"  : left,
                    "right" : right,
                    "up"    : up,
                    "down"  : down
            }

            reward = self._calc_rewardsV2(tile, row, col, connections)
        
        elif (tile == Trees.ID): #Cross
            left  = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID, Person.ID, Fruit.ID]
            right = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID, Person.ID, Fruit.ID]
            up    = [Trees.ID, Grass.ID,  RockGold.ID, Rock.ID, Person.ID, Fruit.ID]
            down  = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID, Person.ID, Fruit.ID]

            connections = {
                "left"  : left,
                "right" : right,
                "up"    : up,
                "down"  : down
            }
            
            reward = self._calc_rewardsV2(tile, row, col, connections)

        elif (tile == Warehouse.ID): 
            
            left  = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID, Rock.ID, RockGold.ID]
            right = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID, Rock.ID, RockGold.ID]
            up    = [Trees.ID, Grass.ID,  Person.ID, House1.ID, Fruit.ID, Rock.ID, RockGold.ID]
            down  = [Grass.ID, Person.ID, House1.ID, Fruit.ID]

            connections = {
                "left"  : left,
                "right" : right,
                "up"    : up,
                "down"  : down
            }
            
            reward = self._calc_rewardsV2(tile, row, col, connections)

        elif (tile == House1.ID): 
            
            left  = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
            right = [Trees.ID, House1.ID, Grass.ID, Person.ID, Fruit.ID]
            up    = [Trees.ID, Grass.ID, Person.ID, House1.ID, Fruit.ID]
            down  = [Grass.ID, Person.ID, House1.ID, Fruit.ID]

            connections = {
                "left"  : left,
                    "right" : right,
                    "up"    : up,
                    "down"  : down
            }
            
            reward = self._calc_rewardsV2(tile, row, col, connections)
            
        elif (tile == RockGold.ID or tile == Rock.ID): #Rock
            
            left  = [Trees.ID, Grass.ID, RockGold.ID, Rock.ID, Fruit.ID]
            right = [Trees.ID, Grass.ID, RockGold.ID, Rock.ID, Fruit.ID]
            up    = [Trees.ID, Grass.ID, RockGold.ID, Rock.ID, Fruit.ID]
            down  = [Trees.ID, Grass.ID, RockGold.ID, Rock.ID, Fruit.ID]

            connections = {
                "left"  : left,
                    "right" : right,
                    "up"    : up,
                    "down"  : down
            }

            reward = self._calc_rewardsV2(tile, row, col, connections)
                        
        return reward      
        
    
    def compute_rewards_by_tile(self, tile, states):
        
        pc = []
        pd = []        
        
        reward = -1
                
        if (tile == Grass.ID):             
            reward = 1
            
        if (tile == Fruit.ID):             
            reward = 1
        
        if (tile == Warehouse.ID):             
            reward = 1            
            
                                            
        elif (tile == Castle.ID): 
            
            left  = [Trees.ID, House1.ID, Grass.ID]
            right = [Trees.ID, House1.ID, Grass.ID]
            up    = [Trees.ID, Grass.ID]
            down  = [Grass.ID]

            connections = {
                "left"  : left,
                 "right" : right,
                 "up"    : up,
                 "down"  : down
            }

            notConnections = {
                 "left"  : [RockGold.ID, Rock.ID],
                 "right" : [RockGold.ID, Rock.IDD],
                 "up"    : [House1.ID, RockGold.ID, Rock.ID],
                 "down"  : [House1.ID, Trees.ID, RockGold.ID, Rock.ID]
                }

            reward, pc, pd = self._compute_reward(connections, notConnects=notConnections,states=states)
        
        elif (tile == Trees.ID): #Cross
            left  = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID]
            right = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID]
            up    = [Trees.ID, Grass.ID,  RockGold.ID, Rock.ID]
            down  = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID]

            connections = {
                "left"  : left,
                "right" : right,
                "up"    : up,
                "down"  : down
            }
            
            notConnections = {
                 "left"  : [],
                 "right" : [],
                 "up"    : [House1.ID],
                 "down"  : []
                }
            
            reward, pc, pd = self._compute_reward(connections, notConnects=notConnections, states=states)

        elif (tile == House1.ID): 
            
            left  = [Trees.ID, House1.ID, Grass.ID]
            right = [Trees.ID, House1.ID, Grass.ID]
            up    = [Trees.ID, Grass.ID]
            down  = [Grass.ID]

            connections = {
                "left"  : left,
                 "right" : right,
                 "up"    : up,
                 "down"  : down
            }
            
            notConnections = {
                 "left"  : [RockGold.ID, Rock.ID],
                 "right" : [RockGold.ID, Rock.ID],
                 "up"    : [House1.ID, RockGold.ID, Rock.ID],
                 "down"  : [House1.ID, Trees.ID, RockGold.ID, Rock.ID]
                }

            reward, pc, pd = self._compute_reward(connections, notConnects=notConnections,states=states)
            
        elif (tile == RockGold.ID or tile == Rock.ID): #Rock
            
            left  = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID]
            right = [Trees.ID, House1.ID, Grass.ID, RockGold.ID, Rock.ID]
            up    = [Trees.ID, Grass.ID, RockGold.ID, Rock.ID, RoadVertical.ID]
            down  = [Trees.ID, Grass.ID, RockGold.ID, Rock.ID, RoadVertical.ID]

            connections = {
                "left"  : left,
                 "right" : right,
                 "up"    : up,
                 "down"  : down
            }

            notConnections = {
                 "left"  : [House1.ID],
                 "right" : [House1.ID],
                 "up"    : [House1.ID],
                 "down"  : [House1.ID]
                }

            reward, pc, pd = self._compute_reward(connections, notConnects=notConnections, states=states)        
                        
        return reward, pc, pd       

    def is_done(self, stats):
        """
        Check if problem is over. This method test if problem satisfying quality based on current stats
        """    
        map_stats = stats["map_stats"] 
        regions_tree = map_stats["regions_trees"]        
        regions_grass = map_stats["regions_grass"]        
        
        tiles = ["Person", "Rock", "RockGold", "Trees", "House1", "Fruit", "Warehouse"]
        
        count_objects = 0
        
        for key in tiles:
            count_objects += map_stats[key]
        
        person = self.in_range(map_stats["Person"], self._range_person[0], self._range_person[1])
        rocks = self.in_range(map_stats["Rock"], self._range_rocks[0], self._range_rocks[1])
        rockgold = self.in_range(map_stats["RockGold"], self._range_rockgolds[0], self._range_rockgolds[1])
        trees = self.in_range(map_stats["Trees"], self._range_trees[0], self._range_trees[1])
        house = self.in_range(map_stats["House1"], self._range_houses[0], self._range_houses[1])
        fruit = self.in_range(map_stats["Fruit"], self._range_fruit[0], self._range_fruit[1])
        warehouse = self.in_range(map_stats["Warehouse"], self._range_ware_house[0], self._range_ware_house[1])
           
        done = (rocks and rockgold and trees and house and person and fruit and warehouse) and regions_grass >= 1 and regions_tree >= 6
        return done and count_objects >= self._min_objects and self.is_connect()
        
    def remove_tile(self, x, y, val):
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

    def change_tile(self, x, y, val):
        if self.render_game:
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
                        
                tile = 0
                """
                if val == Crossing.ID:
                    tile = Crossing(id = Crossing.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)
                if val == RoadVertical.ID:
                    tile  = RoadVertical(id = RoadVertical.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)
                if val == RoadHorizontal.ID:
                    tile  = RoadHorizontal(id = RoadHorizontal.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)
                if val == CrossingBottom.ID:
                    tile  = CrossingBottom(id = CrossingBottom.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)
                """
                #if val == Grass.ID:
                #    tile  = Grass(id = Grass.ID, x = col * state_w, y = row * state_h)                    
                #    self.addBackground(tile)             
                if val == Trees.ID:
                    tile  = Trees(id = Trees.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)            
                if val == House1.ID:
                    tile  = House1(id = House1.ID, x = col * state_w, y = row * state_h)                    
                    self.addStructure(tile)
                if val == Warehouse.ID:
                    tile  = Warehouse(id = Warehouse.ID, x = col * state_w, y = row * state_h)                    
                    self.addStructure(tile)            
                if val == RockGold.ID:
                    tile  = RockGold(id = RockGold.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)
                if val == Rock.ID:
                    tile  = Rock(id = Rock.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)
                if val == Fruit.ID:
                    tile  = Fruit(id = Fruit.ID, x = col * state_w, y = row * state_h)                    
                    self.addBases(tile)
                if val == Person.ID:
                    tile  = Person(id = Person.ID, x = col * state_w, y = row * state_h)                    
                    self.addStructure(tile)             

        self.map[row, col] = val
            
    def __create(self):

        if self.render_game:    
            state_w = self.get_state_width()
            state_h = self.get_state_height()
            for row in range(self.get_rows()):
                for col in range(self.get_cols()):                                
                    val = self.map[row, col]                                     
                    #if val == 0:
                    #    tile  = Grass(id =Grass.ID, x = col * state_w, y = row * state_h)                    
                    #    self.addBackground(tile)                                                     
                    if val == Trees.ID:
                        tile  = Trees(id =Trees.ID, x = col * state_w, y = row * state_h)                    
                        self.addBases(tile)            
                    if val == House1.ID:
                        tile  = House1(id =House1.ID, x = col * state_w, y = row * state_h)                    
                        self.addStructure(tile)  
                    if val == Warehouse.ID:
                        tile  = Warehouse(id = Warehouse.ID, x = col * state_w, y = row * state_h)                    
                        self.addStructure(tile)                     
                    if val == RockGold.ID:
                        tile  = RockGold(id = RockGold.ID, x = col * state_w, y = row * state_h)                    
                        self.addBases(tile)  
                    if val == Rock.ID:
                        tile  = Rock(id = Rock.ID, x = col * state_w, y = row * state_h)                    
                        self.addBases(tile)
                    if val == Fruit.ID:
                        tile  = Fruit(id = Fruit.ID, x = col * state_w, y = row * state_h)                    
                        self.addBases(tile)                    
                    if val == Person.ID:
                        tile  = Person(id = Person.ID, x = col * state_w, y = row * state_h)                    
                        self.addStructure(tile)                                           

    def draw_hud(self, screen):
        if (not self.env is None) and self.show_hud:            
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
                text = "Regions grass: " + str(map_stats["regions_grass"])
                self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)
                current_line += space_line
                text = "Regions trees: " + str(map_stats["regions_trees"])
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