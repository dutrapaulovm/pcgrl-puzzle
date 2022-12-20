import numpy as np
import random
import os
import pandas as pd
from pcgrl.Utils import *
from pcgrl.minimap import MiniMapGameProblem
pos_row, pos_col = 0, 0
max_col  = 4
max_row  = 4
curr_col = 0
curr_row = 0

from pcgrl.Generator import Generator

def next(pos_row, pos_col, max_col, max_row, curr_col, curr_row):
    pos_col  += 4        
    curr_col += 1
        
    if curr_col >= max_col:
        pos_col   = 0
        curr_col  = 0
        pos_row  += 4        
        curr_row += 1
        if curr_row >= max_row:
            pos_row  = 0    
            curr_row = 0
            
    return pos_row, pos_col, curr_row, curr_col


path = os.path.dirname(__file__) + "/map/"

maps_list = []
_map1 = []
for file in os.listdir(path):
    if file[-3:] in {'csv'}: 
        pathfile = os.path.join(path, file)
        dt = pd.read_csv(pathfile).values       
        max_rows, max_cols = int(dt.shape[0] / 4), int(dt.shape[1] / 4)
        pos_row, pos_col   = 0, 0
        curr_col           = 0
        curr_row           = 0  
        _map1              = dt  
        print(dt)
        print("Shape: ", dt.shape)        
        print("Entropy", entropy(dt))        
        r1 = np.var(dt)    
        print("\nvariance: ", r1)           
        #game = MinimapGameProblem(cols=dt.shape[1], rows=dt.shape[0])        
        #game.create_map(np.array(dt))
        #game.run()         
        for x in range(max_cols * max_rows):                                    
                    
            piece = np.zeros((4,4))     
                    
            for ay in range(piece.shape[0]):
                for ax in range(piece.shape[1]):                     
                    tile = int(dt[(pos_row+ay)][(pos_col+ax)])                
                    piece[ay][ax] = int(tile)
            
            maps_list.append(piece)
            #print(piece)
            #print()
            '''
            unique, counts = np.unique(piece, return_counts=True)
            print(np.asarray((unique, counts)).T) 
            x = piece
            
            print(x)
            r1 = np.mean(x)
            r2 = np.average(x)
            assert np.allclose(r1, r2)
            print("\nMean: ", r1)
            r1 = np.std(x)
            r2 = np.sqrt(np.mean((x - np.mean(x)) ** 2 ))
            assert np.allclose(r1, r2)
            print("\nstd: ", 1)
            r1 = np.var(x)
            r2 = np.mean((x - np.mean(x)) ** 2 )
            assert np.allclose(r1, r2)
            print("\nvariance: ", r1)
            '''
            pos_row, pos_col, curr_row, curr_col = next(pos_row, pos_col, max_col, max_row, curr_col, curr_row)                                  
        
        print(piece)

print()
print("Gerando um novo Mapa")
pos_row, pos_col = 0, 0

_map = np.zeros((8,16))
last_piece = np.zeros((4,4))
pie = np.zeros((4,4))
reward = 0.0
for i in range(9):
    last_piece = pie
    action = random.randint(0, len(maps_list)-1)    
    pie = maps_list[action]
 #  print(pie)
    max_col, max_row  = pie.shape
#   print(pie.shape)
    act = 0
    for ax in range(max_col):
        for ay in range(max_row):                
            _map[(pos_row+ay)][(pos_col+ax)] = int(pie[ay][ax])
            act += 1
    pos_row, pos_col, curr_row, curr_col = next(pos_row, pos_col, max_col, max_row, curr_col, curr_row)                          
    js_kl = js_divergence(last_piece, pie)
    print()
    print("JS KL:", js_kl)    
    reward += js_kl
    
print("Reward: ", reward)    
        
_map_aux = _map1.copy()        
_map_aux[0][0] = 6
_map_aux[0][1] = 6

print(_map)
print("Shape: ", _map.shape)        
print("Entropy", entropy(_map))        
r1 = np.var(_map)    
print("\nvariance: ", r1)    
js_kl = js_divergence(_map, _map1)
print("KL V2: ", js_kl)

game = MiniMapGameProblem(cols=_map.shape[1], rows=_map.shape[0])
game.create_map(np.array(_map))
game.run()
game.close()
    
'''
pos_row, pos_col = 0, 0    

_map = np.ones((max_row * 4,  max_col * 4))
print(_map.shape)
print(_map)
low  = 0
high = 6
for x in range(max_col * max_row):    
    action = [random.randint(low, high) for p in range(0, 16)]
    _map[pos_row][pos_col] = action[0]
    act = 0
    for ax in range(max_col):
        for ay in range(max_row+1):                
            _map[(pos_row+ay)][(pos_col+ax)] = action[act]
            act += 1                                                    
    pos_row, pos_col, curr_row, curr_col = next(pos_row, pos_col, max_col, max_row, curr_col, curr_row)                  
print(_map)
'''