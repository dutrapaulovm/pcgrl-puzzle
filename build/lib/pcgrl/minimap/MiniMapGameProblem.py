# -*- coding: utf-8 -*-
import numpy as np

from pcgrl.minimap import *
from pcgrl.minimap.MiniMapLevelObjects import *
from pcgrl.GameProblem import GameProblem
from pcgrl.Utils import *
from pygame.locals import *
import csv   

class MiniMapGameProblem(GameProblem): 
               
    def __init__(self, rows = 0, cols = 0, border = False, tile_size = TILE_SIZE):

        self.border = border
        self.tile_size = tile_size

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
        
        """   
        self._range_fruit      = np.array([1, 100])
        self._range_rocks      = np.array([1, 100])        
        self._range_rockgolds  = np.array([1, 100])
        self._range_houses     = np.array([1, 100])
        self._range_trees      = np.array([1, 100000])
        self._range_person     = np.array([1, 100])
        self._range_ware_house = np.array([1, 100])
        """    
        
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

        self.neighbors = 0
        
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):
                y = row * self.get_state_height()
                x = col * self.get_state_width()
                ground = Grass(x, y)
                self.addBackground_first(ground)
        
        self.tiles = ["Grass", "Trees", "RockGold", "Rock", "House1", "Person", "Fruit", "Warehouse"]                       
        self.dic_tiles = convert_strarray_to_dic(self.tiles)
        
    def border_offset(self):
        return (0, 0)

    def get_info(self):
        params = {}
        return params

    def do(self, event):
        super().do(event)
        if event.type == KEYDOWN:
            if event.key == K_F12:
                self.reset(self.np_random)
                    
    def step(self, action):                         
        reward = 0
        return reward, False   

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
        tiles = ["Grass"]
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
        with open(path_map) as fc:
            creader = csv.reader(fc) # add settings as needed
            dt = [r for r in creader]                
        data = np.array(dt)                 
            
        data = np.array(data).astype("int") 
        
        self.map = data
        self.clear()
        self.__create()
        
    def render_map(self):        
        self.__create()
        self.render(tick=0)
        
    def create_map(self, data):    
        self.map = data
        if not self.blocked:
            self.__create()  
            
    def update_map(self, data = None):
        if (not data is None):
            self.map = data
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

            self.map[row, col] = tile
               
    def get_tiles(self):
        return self.tiles  

    def get_map_stats(self):

        map_stats = {
            "Grass"      : self.calc_tiles(self.map, self.dic_tiles, "Grass"),
            "Trees"      : self.calc_tiles(self.map, self.dic_tiles,  "Trees"),
            "RockGold"   : self.calc_tiles(self.map, self.dic_tiles,  "RockGold"),
            "Rock"       : self.calc_tiles(self.map, self.dic_tiles,  "Rock"),
            "House1"     : self.calc_tiles(self.map, self.dic_tiles,  "House1"),
            "Person"     : self.calc_tiles(self.map, self.dic_tiles,  "Person"),
            "Fruit"      : self.calc_tiles(self.map, self.dic_tiles,  "Fruit"),
            "Warehouse"  : self.calc_tiles(self.map, self.dic_tiles,  "Warehouse"),
            "regions_person"    : self.calc_tiles(self.map, self.dic_tiles, "Person"),
            "regions_grass"    : self.calc_tiles(self.map, self.dic_tiles, "Grass"),
            "regions_trees"    : self.calc_tiles(self.map, self.dic_tiles, "Trees"),
            "regions_house"    : self.calc_tiles(self.map, self.dic_tiles, "House1"),
            "regions_rock"     : self.calc_tiles(self.map, self.dic_tiles, "Rock"),
            "regions_rockgold" : self.calc_tiles(self.map, self.dic_tiles, "RockGold")
        }
        return map_stats  

    def compute_reward(self, new_stats, old_stats):        
        reward = 0.0
        rewards_info = { }
        return reward, rewards_info  

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



        done = (rocks and rockgold and trees and house and person and fruit and warehouse) and regions_grass == 1 and regions_tree >= 6
        return done and count_objects >= self._min_objects #and (len(positionsp) == len(exist_path))
    
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

            self.map[row, col] = tile

    def __create(self):
        
        state_w = self.get_state_width()
        state_h = self.get_state_height()
        
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):                
                val = self.map[row, col]
                self.create_tile(val, x = col * state_w, y = row * state_h)    
        
        super().create()          
            
    def create_tile(self, tile, x, y): 
        val = tile                                                     
        if val == Trees.ID:
            tile  = Trees(id =Trees.ID, x = x, y = y)                    
            self.addBases(tile)            
        if val == House1.ID:
            tile  = House1(id =House1.ID, x = x, y = y)                    
            self.addStructure(tile)  
        if val == Warehouse.ID:
            tile  = Warehouse(id = Warehouse.ID, x = x, y = y)                    
            self.addStructure(tile)                     
        if val == RockGold.ID:
            tile  = RockGold(id = RockGold.ID, x = x, y = y)                    
            self.addBases(tile)  
        if val == Rock.ID:
            tile  = Rock(id = Rock.ID, x = x, y = y)                    
            self.addBases(tile)
        if val == Fruit.ID:
            tile  = Fruit(id = Fruit.ID, x = x, y = y)                    
            self.addBases(tile)                    
        if val == Person.ID:
            tile  = Person(id = Person.ID, x = x, y = y)                    
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
