from time import sleep
from gym.utils import seeding
from gym import spaces
from collections import deque
from numpy.core.fromnumeric import reshape
import pandas as pad
from pcgrl.Agents import *
from pcgrl.BasePCGRLEnv import BasePCGRLEnv
from pcgrl.maze import MazeGameProblem
from pcgrl.maze import *
from pcgrl.Utils import *
from pcgrl import PCGRLPUZZLE_MAP_PATH
from pcgrl.callbacks import BasePCGRLCallback

class MazeEnv(BasePCGRLEnv):
    def __init__(self, 
                seed = None, 
                show_logger = False, 
                save_logger = False, 
                save_image_level = False, 
                path = "", 
                rep = None, 
                action_change = False, 
                piece_size = 8, 
                board = (3,2), 
                env_rewards=False,
                path_models = "mazev2/",
                callback = BasePCGRLCallback()):
        
        self.rep = rep 
        self.cols = board[0] * piece_size #Col
        self.rows = board[1] * piece_size #Row                
                                 
        game = MazeGameProblem(cols = self.cols, rows = self.rows, border = True)
        game.scale    = 2            
        self.action_change = action_change
        super(MazeEnv, self).__init__(seed = seed, game = game, env_rewards=env_rewards,
                                       save_image_level = save_image_level, save_logger=save_logger, show_logger=show_logger, rep=rep, path=path, piece_size = piece_size, board = board, path_models = path_models, callback = callback)
        self.current_reward = 0        
        self.counter_done   = 0        
        self.cols = 24
        self.rows = 16        
        self.name = "MazeEnv"