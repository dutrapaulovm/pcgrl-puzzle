from time import sleep
from gym.utils import seeding
from gym import spaces
from collections import deque
from numpy.core.fromnumeric import reshape
import pandas as pad
from pcgrl import PCGRLPUZZLE_MAP_PATH
from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.dungeon import DungeonGameProblem
from pcgrl.dungeon import *
from pcgrl.Utils import *
from pcgrl.callbacks import BasePCGRLCallback

class DungeonEnv(BasePCGRLEnv):
    
    def __init__(self, 
                seed = None, 
                show_logger = False, 
                save_logger = False,
                save_image_level = False,
                path = "", 
                rep = None, 
                action_change = False, 
                action_rotate = False,
                agent = None,
                reward_change_penalty = None,                
                piece_size = 8, 
                board = (3,2),
                env_rewards=False,
                path_models = "dungeon/",
                callback = BasePCGRLCallback()):        

        self.rows = board[0] * min(piece_size, piece_size * 2) #Row               
        self.cols = board[1] * min(piece_size, piece_size * 2) #Col
        
        game = DungeonGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale = 2
        self.action_change = action_change
        self.action_rotate = action_rotate
        super(DungeonEnv, self).__init__(seed = seed, game = game, 
                                        env_rewards=env_rewards, save_image_level = save_image_level, 
                                        save_logger = save_logger, show_logger=show_logger, 
                                        rep=rep, path=path, piece_size = piece_size, 
                                        action_change=action_change,
                                        action_rotate=action_rotate,
                                        agent=agent,
                                        reward_change_penalty=reward_change_penalty,                                          
                                        board = board, path_models = path_models, callback=callback)
        
        self.current_reward = 0        
        self.counter_done    = 0        
        self.info = {}        
        self.name = "DungeonEnv"