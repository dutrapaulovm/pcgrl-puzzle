import os
import pandas as pd
import csv
import numpy as np
from gym.utils import seeding
from gym import spaces
import time
from pcgrl.Utils import *
from pcgrl.Grid import Grid
import json

MAP_PATH = os.path.dirname(__file__) + "/maps/mapbuilder"


class Generator:    
    
    def __init__(self, seed = None, path = MAP_PATH, piece_size = (4, 4), dim = (256, 256), n_models = -1):
        
        """
            Generator used to load models        
            
        Args:
            seed (int, optional): Seed for the pseudo random generators. Defaults to None.
            path (_type_, optional): path to load load models. Defaults to MAP_PATH.
            piece_size (tuple, optional): Size of piece each model. Defaults to (4, 4).
            dim (tuple, optional): Dimension each models. Defaults to (256, 256).
            n_models (int, optional): Number of models to load. If value less or equal than 0, the generator load all models. Defaults to -1.
        """
        self.path     = path
        self.piece_w  = piece_size[1]
        self.piece_h  = piece_size[0]
        self.pieces       = []
        self.max_cols     = 0
        self.max_rows     = 0

        self.pos_row  = 0
        self.pos_col  = 0 
        self.curr_row = 0
        self.curr_col = 0
        self.n_models = n_models
                
        self.load_pieces()
        
        self.action_space = spaces.Discrete( len(self.pieces) )   
        self.action_space.seed(seed=seed)                      
        
        self.seed(seed=seed)        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed=seed)  
        return [seed]    
    
    def get_piece(self, segment):
         piece = self.pieces[segment]
         return piece
     
    def random_piece(self):
        action = self.action_space.sample()
        return self.get_piece(action)                       
    
    def load_pieces(self):
        #start = timer()
        #print("Start: ", start)
        for file in os.listdir(self.path):
            if file[-3:] in {'csv'}: 
                pathfile = os.path.join(self.path, file)
                #dt = pd.read_csv(pathfile).values       
                dt = None
                with open(pathfile) as fc:
                    creader = csv.reader(fc) # add settings as needed
                    dt = [r for r in creader]                
                dt = np.array(dt)
                max_rows, max_cols = int(dt.shape[0] / self.piece_h), int(dt.shape[1] / self.piece_w)
                pos_row, pos_col   = 0, 0
                curr_col           = 0
                curr_row           = 0
                #print("Peças: ", file,  (max_cols , max_rows), dt.shape)
                #time.sleep(2)                
                for x in range(max_cols * max_rows):                                    

                    piece = np.zeros( (self.piece_h , self.piece_w))     
                            
                    for ay in range(piece.shape[0]):
                        for ax in range(piece.shape[1]):                     
                            tile = int(dt[(pos_row+ay)][(pos_col+ax)])                
                            piece[ay][ax] = int(tile)
                    
                    self.pieces.append(piece)           
                    #print(file, piece)
                    #time.sleep(1)                     
                    pos_row, pos_col, curr_row, curr_col = self.next(pos_row, pos_col, max_cols, max_rows, curr_col, curr_row)                                                 
        end = timer()        
        #print("End: ", end)
        #time_ela = timedelta(seconds=end-start)
        #print("Time elapsed: ", time_ela) 
        #print("Pieces: ", len(self.pieces))  
    
    def reset(self):
        self.curr_col = 0
        self.curr_row = 0
        self.pos_row = 0
        self.pos_col = 0        

    def next(self, pos_row, pos_col, max_col, max_row, curr_col, curr_row):
        
        pos_col  += self.piece_w
        curr_col += 1
                    
        if curr_col >= max_col:
            pos_col   = 0
            curr_col  = 0
            pos_row  += self.piece_w 
            curr_row += 1
            if curr_row >= max_row:
                pos_row  = 0    
                curr_row = 0
                
        return pos_row, pos_col, curr_row, curr_col

    def build_map_xy(self, map, piece, x, y, offset = (0,0)):
        piece = self.get_piece(piece)
        max_col, max_row  = piece.shape
        self.pos_row = (y * self.piece_h)
        self.pos_col = (x * self.piece_w)
        self.curr_col = x
        self.curr_row = y
        #print()
        #print(self.pos_row, self.pos_col)        
        #print((y, x), (self.piece_h, self.piece_h))        
        #print("Map: ", map.shape)
        #print("Row, Col: ", (max_row, max_col))
        for ax in range(max_col):
            for ay in range(max_row):                
                tile = int(piece[ay][ax])
                #print("AY, AX: ", self.pos_row+ay+offset[0], self.pos_col+ax+offset[1])        
                map[(self.pos_row+ay+offset[0])][(self.pos_col+ax+offset[1])] = tile                        
        
        return map, piece
    
    def build_map(self, map, piece,  offset = (0, 0)):
        piece = self.get_piece(piece)
        self.max_cols = int(map.shape[1] / self.piece_w)        
        self.max_rows = int(map.shape[0] / self.piece_h)        
        max_col, max_row  = piece.shape
        for ax in range(max_col):
            for ay in range(max_row):                
                tile = int(piece[ay][ax])
                map[(self.pos_row+ay+offset[0])][(self.pos_col+ax+offset[1])] = tile                

        self.pos_row, self.pos_col, self.curr_row, self.curr_col = self.next(self.pos_row, self.pos_col, self.max_cols, self.max_rows, self.curr_col, self.curr_row)
        return map, piece
    
    def random_map(self, w = 15, h = 9):
        
        pos_row, pos_col = 0, 0
        curr_col = 0
        curr_row = 0        
        _map = np.zeros((h , w))                
        self.max_cols = int(w / self.piece_w)
        self.max_rows = int(h / self.piece_h)
        
        for i in range(self.max_cols * self.max_rows):            
            piece = self.random_piece()
            max_col, max_row  = piece.shape            
            act = 0
            for ax in range(max_col):
                for ay in range(max_row):                
                    tile = int(piece[ay][ax])
                    _map[(pos_row+ay)][(pos_col+ax)] = tile
                    act += 1
            pos_row, pos_col, curr_row, curr_col = self.next(pos_row, pos_col, self.max_cols, self.max_rows, curr_col, curr_row)
        
        return _map
    
    def step(action):
        return []