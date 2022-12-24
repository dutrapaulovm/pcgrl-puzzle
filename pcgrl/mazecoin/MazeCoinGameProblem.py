# -*- coding: utf-8 -*-
import numpy as np

from pcgrl.mazecoin import *
from pcgrl.mazecoin.MazeCoinLevelObjects import *
from pcgrl.GameProblem import GameProblem
from pcgrl.Utils import *
from pygame.locals import *
import csv

class MazeCoinGameProblem(GameProblem):    

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
        
        super(MazeCoinGameProblem, self).__init__(w = self.width, h = self.height, tile_w = self.tile_size, tile_h = self.tile_size)            
        self.show_hud = False
        self.fntHUD      = pygame.font.Font('freesansbold.ttf', 24)     
        self.fntSmallHUD = pygame.font.Font('freesansbold.ttf', 16)
        self.action      = 0   
        self.action_leveldesigner = 0
        self.tiles             = {}                        
        self.showinfo          = False
        self.leveldesigners    = {}              
        self.player = None 
        self.show_game_hud = False
        self.number_coins = 0

        for row in range(self.get_rows()):
            for col in range(self.get_cols()):
                y = row * self.get_state_height()
                x = col * self.get_state_width()
                ground = Ground(x, y)
                self.addBackground_first(ground)         
        
        self.tiles = ["Ground", "Block", "CoinGold", "Player"]
        self.dic_tiles = convert_strarray_to_dic(self.tiles)         
                
        self._range_coins = {"min" : 1, "max" : 2}

        Ground.ID = 0
        Block.ID = 1 
        CoinGold.ID = 2   
        Player.ID = 3    

    def border_offset(self):
        return (1, 1)

    def get_info(self):
        params = {}
        return params
        
    def do(self, event):
        super().do(event)
        if event.type == KEYDOWN:
            if event.key == K_F12:
                self.reset(self.np_random)
                    
    def step(self, action):     
        reward = self.player.step(action)                                                  
        
        if len(self.levelObjects.sprites()) == 0:
            return 50, True

        return reward, False

    def reset(self, np_random):
        self.np_random = np_random
        self.bases.empty()                        
        self.front.empty()
        self.enemies.empty()
        self.structure.empty()                
        self.generate_map(np_random)
        self.update()
    
    def get_tile_name(self, id):
        return self.tiles[id]

    def get_tiles(self):
        return self.tiles        

    def place_objects(self, obj_id, num_objs):
        tiles = ["Ground"]
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
            collide = pygame.sprite.spritecollide(aux, self.structure, True)
                    
            tile = 0

            #if val == Ground.ID:
            #    tile  = Ground(id = Ground.ID, x = col * state_w, y = row * state_h)                    
            #    self.addBackground(tile)            
            if val == Block.ID:
                tile  = Block(id = Block.ID, x = col * state_w, y = row * state_h, tile_height=self.tile_size, tile_width=self.tile_size)                    
                self.addBases(tile)        
            if val == CoinGold.ID:
                tile  = CoinGold(id = CoinGold.ID, x = col * state_w, y = row * state_h, tile_height=self.tile_size, tile_width=self.tile_size)                    
                self.addLevelObjects(tile)   
            if val == Player.ID:
                tile  = Player(id =Player.ID, x = col * state_w, y = row * state_h, tile_height=self.tile_size, tile_width=self.tile_size)                    
                self.addPlayers(tile)                         
                
        self.map[row, col] = val
               
    def get_tiles(self):
        return self.tiles

    def get_map_stats(self):        
            
        map_stats = {
            "Ground"          : self.calc_tiles(self.map, self.dic_tiles, "Ground"),
            "Block"           : self.calc_tiles(self.map, self.dic_tiles, "Block"),
            "CoinGold"        : self.calc_tiles(self.map, self.dic_tiles, "CoinGold"),
            "Player"          : self.calc_tiles(self.map, self.dic_tiles, "Player"),
            "regions_ground"  : self.calc_regions(self.map, Ground.ID, [Ground.ID, Player.ID, CoinGold.ID])
        }     

        return map_stats          

    def compute_reward(self, new_stats, old_stats):        
        reward = 0.0

        map_stats = new_stats["map_stats"] 
        old_map_stats = old_stats["map_stats"]         
        coins = map_stats["CoinGold"]                     
        player = map_stats["Player"]            
        regions_ground = map_stats["regions_ground"]    

        reward += self.range_reward(regions_ground, old_map_stats["regions_ground"], 1, 1, 1)
        reward += self.range_reward(player, old_map_stats["Player"], 1, 1, 2)        
        reward += self.range_reward(coins, old_map_stats["CoinGold"],  self._range_coins["min"],  self._range_coins["max"], 1)

        rewards_info = { }
        return reward, rewards_info          

    def is_done(self, stats):
        """
        Check if problem is over. This method test if problem satisfying quality based on current stats
        """    
        map_stats = stats["map_stats"] 
        regions_ground = map_stats["regions_ground"]
        coin_gold = map_stats["CoinGold"]
        players = map_stats["Player"]
        coin_gold = self.in_range(coin_gold, self._range_coins["min"], self._range_coins["max"])
        done = regions_ground == 1 and coin_gold and players == 1
        return done  
            
    def __create(self):
        
        state_w = self.get_state_width()
        state_h = self.get_state_height()
        
        for row in range(self.get_rows()):
            for col in range(self.get_cols()):                
                val = self.map[row, col]                  
                if val == Block.ID:
                    tile  = Block(id =Block.ID, x = col * state_w, y = row * state_h, tile_height=self.tile_size, tile_width=self.tile_size)                    
                    self.addBases(tile)                                            
                if val == CoinGold.ID:
                    tile  = CoinGold(id = CoinGold.ID, x = col * state_w, y = row * state_h, tile_height=self.tile_size, tile_width=self.tile_size)                    
                    self.addLevelObjects(tile)                     
                if val == Player.ID:
                    tile  = Player(id = Player.ID, x = col * state_w, y = row * state_h, tile_height=self.tile_size, tile_width=self.tile_size) 
                    self.player = tile                   
                    self.addPlayers(tile)                                                                
        
        self.number_coins = len(self.levelObjects.sprites())

        super().create()


    def draw_hud(self, screen):
        if self.show_game_hud:
            space_line    = 32
            current_line  = 0
            current_line += space_line
            text = "Coins {}/{}".format( len(self.levelObjects.sprites()), self.number_coins )
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)
            #current_line += space_line
            #text = "Episode {}".format(self.env.episode)
            #self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)            
            current_line += space_line
            text = "Steps {}".format(self.env.n_steps)
            self.draw_text_ext(x=16, y=current_line, text=text, color=Color(0,0,0), font=self.fntHUD)                        

        if (not self.env is None) and self.show_hud:
            space_line    = 32
            current_line  = 0
            current_line += space_line
            map_stats = self.get_map_stats()            
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