import os
import csv
import numpy as np
import time
import json

MAP_PATH = os.path.dirname(__file__) + "/"


class Generator:    
    
    def __init__(self, path = MAP_PATH, piece_size = (8, 8), loadmap = False, border = True):
        
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
        self.border = border    
        if not loadmap:
            self.load_pieces() 
        else:
            self.load_map()       

    
    def get_piece(self, segment):
         piece = np.array(self.pieces[segment], dtype="int")
         return piece
     
    def random_piece(self):
        action = self.action_space.sample()
        return self.get_piece(action)                       

    def load_map(self):

        dt = None
        with open(self.path) as fc:
            creader = csv.reader(fc) # add settings as needed
            dt = [r for r in creader]                
        dt = np.array(dt)
        
        max_rows, max_cols = int(dt.shape[0] / self.piece_h), int(dt.shape[1] / self.piece_w)
        
        if (self.border):
            max_rows, max_cols = int((dt.shape[0] - 2) / self.piece_h), int((dt.shape[1] - 2) / self.piece_w)
        
        pos_row, pos_col   = 0, 0
        if (self.border):
            pos_row, pos_col   = 1, 1
            #max_rows, max_cols = max_rows, max_rows

        curr_col           = 0
        curr_row           = 0
        print(max_rows, max_cols)
        for x in range(max_cols * max_rows):                                    

            piece = np.zeros( (self.piece_h , self.piece_w))     
            
            for ay in range(piece.shape[0]):
                for ax in range(piece.shape[1]):                     
                    tile = int(dt[(pos_row+ay)][(pos_col+ax)])                
                    piece[ay][ax] = int(tile)            
            #print(piece)            
            #time.sleep(5)
            #print()
            self.pieces.append(piece)           

            pos_row, pos_col, curr_row, curr_col = self.next(pos_row, pos_col, max_cols, max_rows, curr_col, curr_row)                                                 

    def load_pieces(self):

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
                if (self.border):
                    pos_row, pos_col   = 1, 1                
                curr_col           = 0
                curr_row           = 0
                for x in range(max_cols * max_rows):                                    

                    piece = np.zeros( (self.piece_h , self.piece_w))     
                            
                    for ay in range(piece.shape[0]):
                        for ax in range(piece.shape[1]):                     
                            tile = int(dt[(pos_row+ay)][(pos_col+ax)])                
                            piece[ay][ax] = int(tile)
                    
                    self.pieces.append(piece)           

                    pos_row, pos_col, curr_row, curr_col = self.next(pos_row, pos_col, max_cols, max_rows, curr_col, curr_row)                                                 
    
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
    
    def build_map(self, map, piece,  offset = (0, 0), rotate = False):
        piece = self.get_piece(piece)

        if (rotate):
            piece = np.rot90(piece)

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

    def count(self):
        return len(self.pieces)
