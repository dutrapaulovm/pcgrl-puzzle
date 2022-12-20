import numpy as np
import time
import os
import pandas as pd
from pcgrl.Utils import *
import gym
from pcgrl import *
from pcgrl.Generator import Generator
from pcgrl.zelda import ZeldaGameProblem
from pcgrl.zelda import *
import time
import numpy as np
from pcgrl.Utils import *

path_main = os.path.abspath(os.path.join("test", os.pardir))
path_piece = os.path.join(path_main, "pcgrl/maps/Zelda")
path_save = os.path.join(path_main, "test/map/zelda")
game = ZeldaGameProblem(rows=16, cols=24, border = True)
game.map = np.zeros((game.get_rows(), game.get_cols()))        
game.map = np.array(game.map).astype("int")

offset = game.border_offset()
dim = np.array(game.get_dim()).copy()
dim[0] = dim[0] - (offset[0] * game.tile_height) * 2 
dim[1] = dim[1] - (offset[1] * game.tile_width) * 2 
generator = Generator(path=path_piece, piece_size=(8, 8), dim = dim )
total_pieces = int((dim[0] / game.tile_height) / 8 * (dim[1] / game.tile_width) /8)
print("Total segments: ", generator.action_space.n)
root = os.path.abspath(os.path.dirname(__file__))
map_id = 1
generator.seed(seed=round(time.time() * 1000))    
last_piece = np.zeros((8,8))

action_space = gym.spaces.Box(0, generator.action_space.n, shape=(total_pieces,), dtype='int32')
pieces = []
actions = []
rewards = []
action = 0
index = 0
game.reset()
actions =  [ 43, 107, 102, 31, 149,  56]
#actions = [ 66,  41, 114,  32,  44, 116]
#while game.running:               
for a in actions:
    action = a#generator.action_space.sample()        
    game.map, piece = generator.build_map(game.map, action, offset=game.border_offset())             
    pieces.append(piece)
    actions.append(action)
    #js_kl = js_divergence(last_piece, piece)
    action += 1
            
    r = 0
    c = 0
    
    print()                    
    for p in range(len(pieces)-1):                        
        #print(pieces[p])
        #print(piece)
        js = js_divergence(pieces[p], piece)
        print("\t", js)
        if (js <= 0):            
            c += 1    
                
        if (c > 0):
            a = 6
            b = c + 1 + 1.80
            r = -(1 / math.sqrt( a / b ) * b )
        else:
            r = 0
                    
    #print("R: {}, C: {},  P: {}, Avg: {} , Std {}, H: {}".format(r, c, actions, np.mean(actions), np.std(actions), entropy(actions)))        
    last_piece = piece                    
    
    if (len(pieces) >= 6):
        actions = []
        print("Board: {} = Pieces{}, Map: {}".format(entropy(game.map), actions, entropy(actions)))
        print("R: {}, C: {},  P: {}, Avg: {} , Std {}, H: {}".format(r, c, actions, np.mean(actions), np.std(actions), entropy(actions)))        
        pieces  = []
        actions = []
        
        #if r < 0:            
        df = pd.DataFrame(game.map)   
        ps = path_save + "/Map"+str(index)+"JS.csv"
        print(f"Saving CSV to {ps}")     
        df.to_csv(path_save + "/Map"+str(index)+"JS.csv", header=False, index=False)              
        game.render_map()
        ps = path_save + "/Map"+str(index)+"JS.png"
        print(f"Saving IMAGE to {ps}")     
        game.save_screen(ps)                                  
        index += 1  
            
        game.reset()         
            
    game.render()
    #time.sleep(0.5)     
    map_id += 1