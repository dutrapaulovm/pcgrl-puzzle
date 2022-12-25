from time import sleep
from gym.utils import seeding
from gym import spaces
from collections import deque
from numpy.core.fromnumeric import reshape
import pandas as pad
from pcgrl.Agents import LevelDesignerAgentBehavior, Behaviors
from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.smb import SMBGameProblem
from pcgrl.smb import *
from pcgrl.Utils import *
from pcgrl.callbacks import BasePCGRLCallback

from pcgrl import PCGRLPUZZLE_MAP_PATH
class SMBEnv(BasePCGRLEnv):
    def __init__(self, 
                 seed = None, 
                 show_logger = False, 
                 save_logger = False, 
                 save_image_level = False, 
                 rendered = False,
                 path = "", 
                 rep = None, 
                 action_change = False, 
                 action_rotate = False, 
                 agent = None,
                 max_changes = 61,
                 reward_change_penalty = None,                 
                 piece_size = 8, 
                 board = (6, 1), 
                 path_models = "smb/",
                 env_rewards = False,
                 callback = BasePCGRLCallback()):
        
        self.rep = rep          
        self.rows = board[0] * min(piece_size, piece_size * 2) #Row               
        self.cols = board[1] * min(piece_size, piece_size * 2) #Col
        game = SMBGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale    = 2            
        self.action_change = action_change  
        self.action_rotate = action_rotate      
        super(SMBEnv, self).__init__(seed = seed, env_rewards=env_rewards,
                                    game = game, 
                                    action_change=action_change, 
                                    action_rotate=action_rotate,
                                    rendered = rendered,
                                    reward_change_penalty=reward_change_penalty,
                                    agent=agent,
                                    max_changes=max_changes,
                                    save_image_level = save_image_level, 
                                    save_logger=save_logger, 
                                    show_logger=show_logger, rep=rep, path=path, piece_size = piece_size, board = board, path_models = path_models, callback = callback)
        self.current_reward   = 0
        self.counter_done     = 0        
        self.cols = 60
        self.rows = 8        
        self.name = "SMBEnv"           